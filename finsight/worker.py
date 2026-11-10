"""Research worker: claims jobs and runs the workflow durably.

One dedicated database session per worker holds the job's advisory lock, performs every
application write and backs the LangGraph checkpointer. If that session fails, or ownership
is lost, the worker exits immediately (exit code 3) instead of reconnecting; a replacement
worker can take the job over only after acquiring the lock, and resumes from the last
checkpoint. A separate heartbeat session proves liveness; a frozen process stops
heartbeating and is terminated by the supervisor.

    uv run python -m finsight.worker [--once] [--stages demo] [--delay analysis=3] [--heartbeat 5]
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from finsight.graph import Cancelled, DemoStages, JobContext, Stages, build_graph
from finsight.settings import get_settings
from finsight.storage import jobs
from finsight.storage.db import connect

LOST_AUTHORITY_EXIT = 3


class Heartbeat(threading.Thread):
    def __init__(self, url: str, job_id, token, interval: float) -> None:
        super().__init__(daemon=True)
        self.url, self.job_id, self.token, self.interval = url, job_id, token, interval
        self.stop_event = threading.Event()
        self.lost = threading.Event()

    def run(self) -> None:
        conn = connect(self.url, application_name="finsight-heartbeat")
        try:
            while not self.stop_event.is_set():
                try:
                    jobs.heartbeat(conn, self.job_id, self.token)
                except (jobs.OwnershipLost, psycopg.Error):
                    self.lost.set()
                    return
                self.stop_event.wait(self.interval)
        finally:
            conn.close()


class LivenessFile(threading.Thread):
    """Touches FINSIGHT_HEARTBEAT_FILE so the container's watchdog can kill a frozen worker."""

    def __init__(self, path: str, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.path, self.interval = Path(path), interval

    def run(self) -> None:
        while True:
            self.path.touch()
            time.sleep(self.interval)


def build_stages(job: dict, conn, token: uuid.UUID, delays: dict | None = None) -> Stages:
    """Stages for a job, from the model configuration frozen at admission."""
    from finsight.ingestion.sec import PostgresRateLimiter, SecSource
    from finsight.stages import ResearchStages

    config = job["model_config"]
    if config["provider"] == "demo":
        return DemoStages(delays=delays or {})
    settings = get_settings()
    if settings.replay_dir is not None:
        return replay_stages(job, conn, token, settings.replay_dir, settings.state_dir / "artifacts")
    if config["provider"] == "qwen":
        from finsight.reasoning.qwen import QwenProvider
        provider = QwenProvider(config["model"], config.get("base_url") or settings.qwen_base_url,
                                max_tokens=config.get("max_tokens") or 2048,
                                enable_thinking=bool(config.get("enable_thinking")))
    elif config["provider"] == "claude":
        from finsight.reasoning.claude import ClaudeProvider
        from finsight.spend import SpendLedger
        provider = ClaudeProvider(config["model"], effort=settings.effort_default,
                                  max_tokens=config.get("max_tokens") or settings.max_tokens_default,
                                  ledger=SpendLedger(conn, settings.spend_limit_usd))
    else:
        raise ValueError("recorded model responses are only available in replay mode (FINSIGHT_REPLAY_DIR)")
    source = None
    if settings.sec_user_agent:
        source = SecSource(settings.sec_user_agent, settings.state_dir / "artifacts", conn=conn,
                           rate_limiter=PostgresRateLimiter(conn))
    return ResearchStages(conn=conn, job_id=job["id"], token=token, source=source, provider=provider, model_config=config)


def replay_stages(job: dict, conn, token: uuid.UUID, replay_dir: Path, store: Path) -> Stages:
    """Offline: SEC responses from the bundle, model responses from its recordings; anything missing fails the job."""
    from finsight.ingestion.sec import SecSource
    from finsight.reasoning.recorded import RecordedProvider
    from finsight.replay import REPLAY_USER_AGENT, ReplayTransport
    from finsight.stages import ResearchStages

    config = job["model_config"]
    if config["provider"] != "recorded":
        raise ValueError(f"replay mode serves recorded responses only, not provider {config['provider']!r}")
    provider = RecordedProvider(Path(replay_dir) / "recordings.json", config["model"])
    source = SecSource(REPLAY_USER_AGENT, store, conn=conn, transport=ReplayTransport(Path(replay_dir) / "sec"))
    return ResearchStages(conn=conn, job_id=job["id"], token=token, source=source, provider=provider, model_config=config)


class SerializedSaver(PostgresSaver):
    """PostgresSaver whose calls hold the session lock that graph nodes also hold (one session, no concurrent use)."""

    def __init__(self, conn, lock=None) -> None:
        super().__init__(conn)
        self.session_lock = lock or threading.RLock()

    def get_tuple(self, config):
        with self.session_lock:
            return super().get_tuple(config)

    def list(self, *args, **kwargs):
        with self.session_lock:
            return iter(list(super().list(*args, **kwargs)))

    def put(self, *args, **kwargs):
        with self.session_lock:
            return super().put(*args, **kwargs)

    def put_writes(self, *args, **kwargs):
        with self.session_lock:
            return super().put_writes(*args, **kwargs)

    def delete_thread(self, *args, **kwargs):
        with self.session_lock:
            return super().delete_thread(*args, **kwargs)


def run_job(conn, saver: PostgresSaver, job: dict, token: uuid.UUID, stages: Stages, artifacts: Path) -> str:
    ctx = JobContext(conn=conn, job_id=job["id"], token=token, artifacts=artifacts,
                     **({"session_lock": saver.session_lock} if hasattr(saver, "session_lock") else {}))
    graph = build_graph(ctx, stages).compile(checkpointer=saver)
    config = {"configurable": {"thread_id": str(job["id"])}}
    snapshot = graph.get_state(config)
    try:
        if snapshot.values:  # resume from the last checkpoint
            final = graph.invoke(None, config, durability="sync")
        else:
            final = graph.invoke({"job_id": str(job["id"]), "request": job["request"]}, config, durability="sync")
    except Cancelled:
        jobs.finish(conn, job["id"], token, "cancelled", "cancellation requested")
        return "cancelled"
    except (jobs.OwnershipLost, psycopg.OperationalError, psycopg.InterfaceError):
        raise
    except Exception as exc:  # a stage error ends the job with a clear outcome instead of crashing the worker
        jobs.finish(conn, job["id"], token, "failed", f"{type(exc).__name__}: {exc}"[:500])
        return "failed"
    outcome = final["reporting"]["outcome"]
    unresolved = final["reporting"].get("report", {}).get("unresolved") or []
    jobs.finish(conn, job["id"], token, outcome, None if outcome == "completed" else "; ".join(unresolved)[:500] or None)
    return outcome


def serve(url: str, worker_id: str, stages, artifacts: Path, *, once: bool = False,
          poll: float = 1.0, heartbeat_interval: float = 5.0) -> int:
    """``stages`` is a Stages object, or a factory ``(job, conn, token) -> Stages``."""
    if os.environ.get("FINSIGHT_HEARTBEAT_FILE"):
        LivenessFile(os.environ["FINSIGHT_HEARTBEAT_FILE"], min(heartbeat_interval, 5.0)).start()
    conn = connect(url, application_name=f"finsight-worker:{worker_id}")
    saver = SerializedSaver(conn)
    saver.setup()
    while True:
        claimed = jobs.claim(conn, worker_id)
        if claimed is None:
            if once:
                return 0
            time.sleep(poll)
            continue
        job, token = claimed
        beat = Heartbeat(url, job["id"], token, heartbeat_interval)
        beat.start()
        try:
            try:
                job_stages = stages(job, conn, token) if callable(stages) else stages
            except (ValueError, OSError, KeyError) as exc:      # bad configuration fails the job, never the worker
                jobs.finish(conn, job["id"], token, "failed", f"configuration: {exc}"[:500])
                job_stages = None
            if job_stages is not None:
                run_job(conn, saver, job, token, job_stages, artifacts)
        except (jobs.OwnershipLost, psycopg.OperationalError, psycopg.InterfaceError) as exc:
            print(f"worker {worker_id}: lost authority over job {job['id']} ({type(exc).__name__}); exiting",
                  file=sys.stderr, flush=True)
            if os.environ.get("FINSIGHT_WORKER_RAISE_ON_LOST"):    # tests only: surface a traceback instead of killing pytest
                raise
            os._exit(LOST_AUTHORITY_EXIT)  # never reconnect and continue writing
        finally:
            beat.stop_event.set()
        jobs.release(conn, job["id"])
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--delay", action="append", default=[], metavar="STAGE=SECONDS")
    parser.add_argument("--heartbeat", type=float, default=5.0)
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}:{os.getpid()}")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    delays = {k: float(v) for k, v in (d.split("=", 1) for d in args.delay)}
    return serve(args.database_url or settings.database_url, args.worker_id,
                 lambda job, conn, token: build_stages(job, conn, token, delays), settings.state_dir / "artifacts",
                 once=args.once, heartbeat_interval=args.heartbeat)


if __name__ == "__main__":
    raise SystemExit(main())
