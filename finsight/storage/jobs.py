"""Research job records: admission, claiming, ownership, heartbeats, events and outcomes.

Ownership rules:

* A worker claims a job on its **dedicated session** with ``FOR UPDATE SKIP LOCKED`` and then
  takes a session-level advisory lock for that job on the same session. That session also
  backs the LangGraph checkpointer, so losing it ends the worker's authority over both the
  application records and the checkpoints; the worker must exit rather than reconnect.
* Every worker write is conditional on the job's current ``owner_token``.
* A ``running`` job whose advisory lock is free (its owner's session is gone) can be taken over
  by another worker, which resumes from the last checkpoint with a new token.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import psycopg

TERMINAL = ("completed", "insufficient_evidence", "blocked", "failed", "cancelled")


class OwnershipLost(RuntimeError):
    """This worker no longer owns the job; it must stop without writing anything else."""


def lock_key(job_id: uuid.UUID | str) -> int:
    return int.from_bytes(uuid.UUID(str(job_id)).bytes[:8], "big", signed=True)


def create_job(conn: psycopg.Connection, request: dict, model_config: dict,
               idempotency_key: str | None = None) -> dict:
    job_id = uuid.uuid4()
    row = conn.execute(
        "INSERT INTO research_jobs (id, request, model_config, idempotency_key) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (idempotency_key) DO NOTHING RETURNING *",
        (job_id, json.dumps(request), json.dumps(model_config), idempotency_key)).fetchone()
    if row is None:  # same idempotency key: return the existing job
        row = conn.execute("SELECT * FROM research_jobs WHERE idempotency_key = %s", (idempotency_key,)).fetchone()
    else:
        conn.execute("INSERT INTO job_events (job_id, kind, detail) VALUES (%s, 'admitted', %s)",
                     (job_id, json.dumps({"model_config": model_config})))
    return row


def get_job(conn: psycopg.Connection, job_id: uuid.UUID | str) -> dict | None:
    return conn.execute("SELECT * FROM research_jobs WHERE id = %s", (job_id,)).fetchone()


def events(conn: psycopg.Connection, job_id: uuid.UUID | str, limit: int = 100) -> list[dict]:
    return conn.execute("SELECT at, kind, stage, detail FROM job_events WHERE job_id = %s ORDER BY id DESC LIMIT %s",
                        (job_id, limit)).fetchall()


def request_cancel(conn: psycopg.Connection, job_id: uuid.UUID | str) -> dict | None:
    with conn.transaction():
        row = conn.execute("UPDATE research_jobs SET cancel_requested = true, updated_at = now(), "
                           "status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END, "
                           "finished_at = CASE WHEN status = 'queued' THEN now() ELSE finished_at END "
                           "WHERE id = %s RETURNING *", (job_id,)).fetchone()
        if row:
            conn.execute("INSERT INTO job_events (job_id, kind) VALUES (%s, 'cancel_requested')", (job_id,))
    return row


def claim(conn: psycopg.Connection, worker_id: str) -> tuple[dict, uuid.UUID] | None:
    """Claim a queued job, or take over a running job whose owner's session is gone.

    Must be called on the worker's dedicated session: the advisory lock it takes is held by that
    session until it ends.
    """
    with conn.transaction():
        candidates = conn.execute(
            "SELECT id, status FROM research_jobs WHERE status IN ('queued', 'running') "
            "ORDER BY (status = 'running') DESC, created_at FOR UPDATE SKIP LOCKED LIMIT 20").fetchall()
        for candidate in candidates:
            if not conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (lock_key(candidate["id"]),)).fetchone()["ok"]:
                continue  # a live session owns it
            token = uuid.uuid4()
            row = conn.execute(
                "UPDATE research_jobs SET status = 'running', owner_token = %s, owner_worker = %s, "
                "owner_backend_pid = pg_backend_pid(), heartbeat_at = now(), updated_at = now() "
                "WHERE id = %s RETURNING *", (token, worker_id, candidate["id"])).fetchone()
            kind = "taken_over" if candidate["status"] == "running" else "claimed"
            conn.execute("INSERT INTO job_events (job_id, kind, detail) VALUES (%s, %s, %s)",
                         (candidate["id"], kind, json.dumps({"worker": worker_id})))
            return row, token
    return None


def release(conn: psycopg.Connection, job_id: uuid.UUID | str) -> None:
    conn.execute("SELECT pg_advisory_unlock(%s)", (lock_key(job_id),))


def owned_update(conn: psycopg.Connection, job_id: uuid.UUID | str, token: uuid.UUID, **fields) -> dict:
    """Update job fields only if this worker still owns the job."""
    assignments = ", ".join(f"{name} = %s" for name in fields) + ", updated_at = now()"
    values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    row = conn.execute(f"UPDATE research_jobs SET {assignments} WHERE id = %s AND owner_token = %s RETURNING *",
                       (*values, job_id, token)).fetchone()
    if row is None:
        raise OwnershipLost(f"job {job_id}: ownership lost")
    return row


def owned_event(conn: psycopg.Connection, job_id: uuid.UUID | str, token: uuid.UUID, kind: str,
                stage: str | None = None, detail: dict | None = None) -> None:
    row = conn.execute(
        "INSERT INTO job_events (job_id, kind, stage, detail) SELECT %s, %s, %s, %s "
        "WHERE EXISTS (SELECT 1 FROM research_jobs WHERE id = %s AND owner_token = %s) RETURNING id",
        (job_id, kind, stage, json.dumps(detail or {}), job_id, token)).fetchone()
    if row is None:
        raise OwnershipLost(f"job {job_id}: ownership lost")


def heartbeat(conn: psycopg.Connection, job_id: uuid.UUID | str, token: uuid.UUID) -> datetime:
    """Called from a separate heartbeat session; a frozen process stops heartbeating entirely."""
    row = conn.execute("UPDATE research_jobs SET heartbeat_at = now(), heartbeat_backend_pid = pg_backend_pid() "
                       "WHERE id = %s AND owner_token = %s AND status = 'running' RETURNING heartbeat_at",
                       (job_id, token)).fetchone()
    if row is None:
        raise OwnershipLost(f"job {job_id}: ownership lost")
    return row["heartbeat_at"]


def finish(conn: psycopg.Connection, job_id: uuid.UUID | str, token: uuid.UUID, status: str,
           reason: str | None = None) -> dict:
    if status not in TERMINAL:
        raise ValueError(f"{status!r} is not a terminal status")
    with conn.transaction():
        row = owned_update(conn, job_id, token, status=status, outcome_reason=reason, stage=None)
        conn.execute("UPDATE research_jobs SET finished_at = now() WHERE id = %s", (job_id,))
        owned_event(conn, job_id, token, "finished", detail={"status": status, "reason": reason})
    return row


def stale_owners(conn: psycopg.Connection, seconds: int) -> list[dict]:
    return conn.execute(
        "SELECT id, owner_worker, owner_backend_pid, heartbeat_backend_pid, heartbeat_at FROM research_jobs "
        "WHERE status = 'running' AND heartbeat_at < now() - make_interval(secs => %s)", (seconds,)).fetchall()
