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


def run_job(conn, saver: PostgresSaver, job: dict, token: uuid.UUID, stages: Stages, artifacts: Path) -> str:
    ctx = JobContext(conn=conn, job_id=job["id"], token=token, artifacts=artifacts)
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
    outcome = final["reporting"]["outcome"]
    jobs.finish(conn, job["id"], token, outcome)
    return outcome


def serve(url: str, worker_id: str, stages: Stages, artifacts: Path, *, once: bool = False,
          poll: float = 1.0, heartbeat_interval: float = 5.0) -> int:
    conn = connect(url, application_name=f"finsight-worker:{worker_id}")
    saver = PostgresSaver(conn)
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
            run_job(conn, saver, job, token, stages, artifacts)
        except (jobs.OwnershipLost, psycopg.OperationalError, psycopg.InterfaceError) as exc:
            print(f"worker {worker_id}: lost authority over job {job['id']} ({type(exc).__name__}); exiting",
                  file=sys.stderr, flush=True)
            os._exit(LOST_AUTHORITY_EXIT)  # never reconnect and continue writing
        finally:
            beat.stop_event.set()
        jobs.release(conn, job["id"])
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stages", default="demo", choices=["demo"])
    parser.add_argument("--delay", action="append", default=[], metavar="STAGE=SECONDS")
    parser.add_argument("--heartbeat", type=float, default=5.0)
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}:{os.getpid()}")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    delays = {k: float(v) for k, v in (d.split("=", 1) for d in args.delay)}
    stages = DemoStages(delays=delays)
    return serve(args.database_url or settings.database_url, args.worker_id, stages, settings.state_dir / "artifacts",
                 once=args.once, heartbeat_interval=args.heartbeat)


if __name__ == "__main__":
    raise SystemExit(main())
