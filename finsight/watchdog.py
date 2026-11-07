"""Watchdog parent for the worker container (needs no Docker socket).

It starts the worker as a child process. The worker touches a liveness file from a separate thread every few
seconds (``FINSIGHT_HEARTBEAT_FILE``). A frozen worker (SIGSTOP, or a process that stops scheduling) stops
touching it; after ``--timeout`` seconds the watchdog SIGKILLs the child. In Postgres the supervisor has already
terminated the frozen worker's sessions (or will), so its authority over the job ended either way.

Whenever the child exits, including exit code 3 (lost authority), a new worker starts after a short backoff with
fresh database sessions; it can take a job over only by acquiring the job's advisory lock. SIGTERM/SIGINT are
forwarded to the child and end the watchdog.

    python -m finsight.watchdog [--timeout 30] -- python -m finsight.worker --heartbeat 5
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _age(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return float("inf")


def run(command: list[str], heartbeat_file: Path, *, timeout: float = 30.0, startup_grace: float = 60.0,
        poll: float = 1.0, backoff: float = 2.0, max_starts: int | None = None) -> list[dict]:
    """Supervise ``command``; returns the event log (used by tests; runs forever when max_starts is None)."""
    events: list[dict] = []
    stopping = {"flag": False}
    child: dict = {"proc": None}

    def forward(signum, _frame):
        stopping["flag"] = True
        if child["proc"] and child["proc"].poll() is None:
            child["proc"].send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
    starts = 0
    try:
        while not stopping["flag"] and (max_starts is None or starts < max_starts):
            heartbeat_file.unlink(missing_ok=True)
            env = {**os.environ, "FINSIGHT_HEARTBEAT_FILE": str(heartbeat_file)}
            proc = child["proc"] = subprocess.Popen(command, env=env)
            starts += 1
            started = time.monotonic()
            events.append({"event": "started", "pid": proc.pid})
            while proc.poll() is None and not stopping["flag"]:
                age = _age(heartbeat_file)
                limit = startup_grace if age == float("inf") else timeout
                if (age == float("inf") and time.monotonic() - started > limit) or (age != float("inf") and age > limit):
                    proc.kill()  # SIGKILL works on a stopped process too
                    proc.wait()
                    events.append({"event": "killed", "pid": proc.pid, "heartbeat_age_s": round(age, 1)})
                    print(f"watchdog: worker {proc.pid} heartbeat {age:.0f}s old; killed", file=sys.stderr, flush=True)
                    break
                time.sleep(poll)
            code = proc.wait()
            events.append({"event": "exited", "pid": proc.pid, "code": code})
            if not stopping["flag"]:
                time.sleep(backoff)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    return events


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m finsight.watchdog [--timeout S] -- COMMAND...")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--heartbeat-file", type=Path, default=Path("/tmp/finsight-worker.alive"))
    args = parser.parse_args(argv[:split])
    run(argv[split + 1:], args.heartbeat_file, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
