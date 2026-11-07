"""Supervisor: ends the authority of workers whose heartbeat has stopped.

A process stopped with SIGSTOP keeps its TCP connections open, so neither keepalives nor
in-process timers detect it. The supervisor runs independently, finds running jobs with a
stale heartbeat and terminates the owner's dedicated session and heartbeat session with
``pg_terminate_backend``. That releases the job's advisory lock (so another worker can take
over) and guarantees the frozen worker's later writes fail; it works across PID namespaces,
including Docker containers. Inside a worker container a watchdog parent can additionally
kill the frozen process.

    uv run python -m finsight.supervisor [--timeout 30] [--interval 5] [--once]
"""

from __future__ import annotations

import argparse
import json
import time

from finsight.settings import get_settings
from finsight.storage import jobs
from finsight.storage.db import connect


def sweep(conn, timeout_seconds: int) -> list[dict]:
    terminated = []
    for job in jobs.stale_owners(conn, timeout_seconds):
        pids = [p for p in (job["owner_backend_pid"], job["heartbeat_backend_pid"]) if p]
        results = {pid: conn.execute("SELECT pg_terminate_backend(%s) AS ok", (pid,)).fetchone()["ok"] for pid in pids}
        conn.execute("INSERT INTO job_events (job_id, kind, detail) VALUES (%s, 'owner_terminated', %s)",
                     (job["id"], json.dumps({"worker": job["owner_worker"], "backends": results,
                                             "heartbeat_at": str(job["heartbeat_at"])})))
        terminated.append({"job_id": str(job["id"]), "backends": results})
    return terminated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=30, help="seconds without a heartbeat before termination")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    conn = connect(args.database_url or get_settings().database_url, application_name="finsight-supervisor")
    while True:
        for item in sweep(conn, args.timeout):
            print(json.dumps(item), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
