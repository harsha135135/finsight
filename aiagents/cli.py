"""Command line entry points. Nothing runs in the background implicitly."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import load_job
from .engine import Engine
from .store import RunBusy, Store


def _job(path):
    job, base_dir = load_job(path)
    if job.provider == "openai" and job.model is None:
        model = os.environ.get("OPENAI_MODEL", "").strip()
        if not model:
            raise ValueError("Set OPENAI_MODEL or the job's model field before submitting an OpenAI job")
        job = replace(job, model=model)
    return job, base_dir


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autonomous research and reports from configured evidence")
    p.add_argument("--state-dir", default=".aiagents", help="Database, checkpoints, and report directory")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a job through planning, analysis, review, and report generation")
    run.add_argument("job")
    resume = commands.add_parser("resume", help="Continue a failed/interrupted run, or regenerate completed reports")
    resume.add_argument("run_id")
    commands.add_parser("runs", help="List the latest 100 runs")
    status = commands.add_parser("status", help="Inspect one run and its artifacts")
    status.add_argument("run_id")
    status.add_argument("--events", action="store_true")
    schedule = commands.add_parser("schedule", help="Manage persistent interval schedules")
    actions = schedule.add_subparsers(dest="action", required=True)
    add = actions.add_parser("add")
    add.add_argument("name")
    add.add_argument("job")
    add.add_argument("--every", type=int, default=86400, metavar="SECONDS")
    actions.add_parser("list")
    disable = actions.add_parser("disable")
    disable.add_argument("name")
    worker = commands.add_parser("worker", help="Run due schedules in the foreground; Ctrl-C stops")
    worker.add_argument("--once", action="store_true", help="Process pending scheduled runs once, then exit")
    worker.add_argument("--poll", type=int, default=10, metavar="SECONDS")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    store = None
    try:
        store = Store(args.state_dir)
        engine = Engine(store)
        if args.command == "run":
            job, base_dir = _job(args.job)
            result = engine.execute(store.create(job.to_dict(), base_dir))
            _print(result)
            return 0 if result["status"] == "completed" else 1
        if args.command == "resume":
            result = engine.execute(args.run_id)
            _print(result)
            return 0 if result["status"] == "completed" else 1
        if args.command == "runs":
            _print(store.runs())
        elif args.command == "status":
            result = store.get(args.run_id)
            if args.events:
                result["events"] = store.events(args.run_id)
            collection = store.checkpoint(args.run_id, "collection")
            if collection:
                result["source_errors"] = collection["errors"]
            if store.checkpoint(args.run_id, "final"):
                result["report_directory"] = str(store.root / "reports" / args.run_id)
            _print(result)
        elif args.command == "schedule":
            if args.action == "add":
                job, base_dir = _job(args.job)
                schedule_id = store.add_schedule(args.name, job.to_dict(), base_dir, args.every)
                _print({"schedule_id": schedule_id, "name": args.name, "status": "registered", "starts": "When a worker is running"})
            elif args.action == "list":
                _print(store.schedules())
            elif args.action == "disable":
                store.disable_schedule(args.name)
                _print({"name": args.name, "status": "disabled", "note": "Already queued runs remain available"})
        elif args.command == "worker":
            if not 1 <= args.poll <= 60:
                raise ValueError("--poll must be between 1 and 60 seconds")
            while True:
                store.enqueue_due()
                failed = False
                for run_id in store.pending_scheduled():
                    try:
                        result = engine.execute(run_id)
                    except RunBusy:
                        continue
                    _print(result)
                    failed = failed or result["status"] != "completed"
                if args.once:
                    return 1 if failed else 0
                time.sleep(args.poll)
        return 0
    except KeyboardInterrupt:
        print("Stopped. Interrupted runs retain their checkpoints.", file=sys.stderr)
        return 130
    except (ValueError, OSError, sqlite3.Error, RunBusy) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if store:
            store.close()
