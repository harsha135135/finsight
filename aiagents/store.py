"""Durable checkpoints, an audit trail, and a small persistent scheduler."""

from __future__ import annotations

import fcntl
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class RunBusy(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class Store:
    def __init__(self, root: str | Path = ".aiagents"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "state.db", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, job TEXT NOT NULL, base_dir TEXT NOT NULL,
                status TEXT NOT NULL, calls INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL, updated REAL NOT NULL, error TEXT,
                schedule_id TEXT, due_time REAL,
                UNIQUE(schedule_id, due_time)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL REFERENCES runs(id), stage TEXT NOT NULL,
                result TEXT NOT NULL, PRIMARY KEY(run_id, stage)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                timestamp REAL NOT NULL, kind TEXT NOT NULL, details TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schedules (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, job TEXT NOT NULL,
                base_dir TEXT NOT NULL, interval INTEGER NOT NULL,
                next_due REAL NOT NULL, enabled INTEGER NOT NULL DEFAULT 1
            );
        """)

    def close(self) -> None:
        self.db.close()

    def _insert_run(self, job: dict, base_dir: Path, schedule_id=None, due_time=None) -> str:
        run_id = uuid.uuid4().hex
        now = time.time()
        self.db.execute(
            "INSERT INTO runs(id,job,base_dir,status,created,updated,schedule_id,due_time) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, json.dumps(job), str(base_dir.resolve()), "queued", now, now, schedule_id, due_time),
        )
        return run_id

    def create(self, job: dict, base_dir: Path) -> str:
        with self.db:
            return self._insert_run(job, base_dir)

    def get(self, run_id: str) -> dict:
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown run: {run_id}")
        result = dict(row)
        result["job"] = json.loads(result["job"])
        return result

    def runs(self) -> list[dict]:
        return [dict(row) for row in self.db.execute(
            "SELECT id,status,calls,created,updated,error,schedule_id FROM runs ORDER BY created DESC LIMIT 100"
        )]

    def set_status(self, run_id: str, status: str, error: str | None = None) -> None:
        with self.db:
            self.db.execute("UPDATE runs SET status=?,error=?,updated=? WHERE id=?",
                            (status, error, time.time(), run_id))
            self._event(run_id, "status", {"status": status, "error": error})

    def checkpoint(self, run_id: str, stage: str):
        row = self.db.execute("SELECT result FROM checkpoints WHERE run_id=? AND stage=?", (run_id, stage)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, run_id: str, stage: str, result: dict | list) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES(?,?,?)",
                            (run_id, stage, json.dumps(result, allow_nan=False)))
            self._event(run_id, "checkpoint", {"stage": stage})

    def _event(self, run_id: str, kind: str, details: dict) -> None:
        self.db.execute("INSERT INTO events(run_id,timestamp,kind,details) VALUES(?,?,?,?)",
                        (run_id, time.time(), kind, json.dumps(details)))

    def event(self, run_id: str, kind: str, details: dict) -> None:
        with self.db:
            self._event(run_id, kind, details)

    def events(self, run_id: str) -> list[dict]:
        self.get(run_id)
        return [{**dict(row), "details": json.loads(row["details"])} for row in self.db.execute(
            "SELECT timestamp,kind,details FROM events WHERE run_id=? ORDER BY id", (run_id,)
        )]

    def consume_call(self, run_id: str, stage: str, limit: int) -> None:
        with self.db:
            changed = self.db.execute(
                "UPDATE runs SET calls=calls+1,updated=? WHERE id=? AND calls<?",
                (time.time(), run_id, limit),
            ).rowcount
            if not changed:
                raise BudgetExceeded(f"Model request limit ({limit}) reached; start a new job with a revised budget")
            self._event(run_id, "model_attempt", {"stage": stage})

    @contextmanager
    def lock(self, run_id: str, kind: str = "run"):
        if kind not in ("run", "schedule") or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run ID")
        locks = self.root / "locks"
        locks.mkdir(exist_ok=True)
        with (locks / f"{kind}-{run_id}.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunBusy(f"{kind.title()} {run_id} is already executing") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def add_schedule(self, name: str, job: dict, base_dir: Path, interval: int) -> str:
        if not name.strip() or type(interval) is not int or interval < 60:
            raise ValueError("Schedule needs a name and an interval of at least 60 seconds")
        schedule_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO schedules VALUES(?,?,?,?,?,?,1)",
                            (schedule_id, name, json.dumps(job), str(base_dir.resolve()), interval, time.time()))
        return schedule_id

    def schedules(self) -> list[dict]:
        return [dict(row) for row in self.db.execute(
            "SELECT id,name,interval,next_due,enabled FROM schedules ORDER BY name"
        )]

    def disable_schedule(self, name: str) -> None:
        with self.db:
            if not self.db.execute("UPDATE schedules SET enabled=0 WHERE name=?", (name,)).rowcount:
                raise ValueError(f"Unknown schedule: {name}")

    def enqueue_due(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        ids = []
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            for row in self.db.execute("SELECT * FROM schedules WHERE enabled=1 AND next_due<=?", (now,)).fetchall():
                # Coalesce missed intervals. Do not create another run while this schedule has unfinished work.
                pending = self.db.execute(
                    "SELECT 1 FROM runs WHERE schedule_id=? AND status IN ('queued','running','interrupted')",
                    (row["id"],),
                ).fetchone()
                if not pending:
                    ids.append(self._insert_run(json.loads(row["job"]), Path(row["base_dir"]), row["id"], row["next_due"]))
                self.db.execute("UPDATE schedules SET next_due=? WHERE id=?", (now + row["interval"], row["id"]))
        return ids

    def pending_scheduled(self) -> list[str]:
        return [row[0] for row in self.db.execute(
            "SELECT id FROM runs WHERE schedule_id IS NOT NULL AND status IN ('queued','running','interrupted') ORDER BY created"
        )]
