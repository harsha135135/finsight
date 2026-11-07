"""The single authoritative spend ledger for paid model calls (FinSight jobs and ModelForge runs).

Every paid attempt is a row in ``model_attempts`` that moves through
``reserved -> sent -> completed | unknown | failed``:

* ``reserve`` locks the ``spend_limit`` row, adds up committed spend (completed costs plus the
  reservations of reserved/sent/unknown attempts) and refuses the call if the new reservation
  would exceed the limit, so concurrent clients cannot overspend together.
* ``unknown``: the request may have been accepted but no response was recorded (timeout,
  connection loss, crash after sending). It keeps its reservation, stays visible, and is never
  resent automatically; a retry is a new, separately reserved attempt.
* ``failed``: definitely not charged (e.g. HTTP 4xx before processing, 429, 529).
* A completed attempt for the same job, stage, attempt and request hash is reused on re-run.

A file-backed ledger is deliberately not offered for paid calls; offline tests use recorded
providers and never reach this module.
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal


class BudgetExceeded(RuntimeError):
    pass


OPEN = ("reserved", "sent", "unknown")


class SpendLedger:
    def __init__(self, conn, limit_usd: Decimal) -> None:
        self.conn = conn
        conn.execute("INSERT INTO spend_limit (id, limit_usd) VALUES (1, %s) ON CONFLICT (id) DO UPDATE "
                     "SET limit_usd = EXCLUDED.limit_usd", (limit_usd,))

    def committed(self) -> Decimal:
        row = self.conn.execute(
            "SELECT coalesce(sum(CASE WHEN status = 'completed' THEN coalesce(cost_usd, 0) "
            "WHEN status IN ('reserved', 'sent', 'unknown') THEN reserved_usd ELSE coalesce(cost_usd, 0) END), 0) AS v "
            "FROM model_attempts WHERE paid").fetchone()
        return Decimal(row["v"])

    def completed_response(self, job_id, stage: str, attempt: int, request_sha256: str) -> dict | None:
        return self.conn.execute(
            "SELECT * FROM model_attempts WHERE job_id IS NOT DISTINCT FROM %s AND stage = %s AND attempt = %s "
            "AND request_sha256 = %s AND status = 'completed'", (job_id, stage, attempt, request_sha256)).fetchone()

    def reserve(self, *, scope: str, job_id, stage: str, attempt: int, request_sha256: str, provider: str,
                model: str, amount_usd: Decimal) -> uuid.UUID:
        attempt_id = uuid.uuid4()
        with self.conn.transaction():
            limit = Decimal(self.conn.execute("SELECT limit_usd FROM spend_limit WHERE id = 1 FOR UPDATE").fetchone()["limit_usd"])
            committed = self.committed()
            if committed + amount_usd > limit:
                raise BudgetExceeded(f"reserving ${amount_usd} would take committed spend from ${committed} past the "
                                     f"${limit} limit")
            self.conn.execute(
                "INSERT INTO model_attempts (id, scope, job_id, stage, attempt, request_sha256, provider, model, status, "
                "paid, reserved_usd) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'reserved', true, %s)",
                (attempt_id, scope, job_id, stage, attempt, request_sha256, provider, model, amount_usd))
        return attempt_id

    def reserve_many(self, entries: list[dict], *, run_cap_usd: Decimal | None = None) -> list[uuid.UUID]:
        """Reserve a whole batch in one transaction: all rows or none. Each entry has the keyword arguments
        of ``reserve``. Refused past the global limit, and past ``run_cap_usd`` for this run alone."""
        total = sum((Decimal(e["amount_usd"]) for e in entries), Decimal(0))
        if run_cap_usd is not None and total > run_cap_usd:
            raise BudgetExceeded(f"the batch's worst case ${total} exceeds its run cap ${run_cap_usd}")
        ids = [uuid.uuid4() for _ in entries]
        with self.conn.transaction():
            limit = Decimal(self.conn.execute("SELECT limit_usd FROM spend_limit WHERE id = 1 FOR UPDATE").fetchone()["limit_usd"])
            committed = self.committed()
            if committed + total > limit:
                raise BudgetExceeded(f"reserving ${total} would take committed spend from ${committed} past the "
                                     f"${limit} limit")
            with self.conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO model_attempts (id, scope, job_id, stage, attempt, request_sha256, provider, model, "
                    "status, paid, reserved_usd) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'reserved', true, %s)",
                    [(i, e["scope"], e.get("job_id"), e["stage"], e["attempt"], e["request_sha256"], e["provider"],
                      e["model"], Decimal(e["amount_usd"])) for i, e in zip(ids, entries)])
        return ids

    def mark_sent_many(self, attempt_ids: list[uuid.UUID], note: str | None = None) -> None:
        self.conn.execute("UPDATE model_attempts SET status = 'sent', error = coalesce(%s, error), updated_at = now() "
                          "WHERE id = ANY(%s) AND status IN ('reserved', 'unknown')", (note, list(attempt_ids)))

    def unknown_many(self, attempt_ids: list[uuid.UUID], error: str) -> None:
        self.conn.execute("UPDATE model_attempts SET status = 'unknown', error = %s, updated_at = now() "
                          "WHERE id = ANY(%s) AND status IN ('reserved', 'sent')", (error, list(attempt_ids)))

    def failed_many(self, attempt_ids: list[uuid.UUID], error: str) -> None:
        """Definitely not charged (for example, the batch was rejected before creation)."""
        self.conn.execute("UPDATE model_attempts SET status = 'failed', cost_usd = 0, error = %s, updated_at = now() "
                          "WHERE id = ANY(%s) AND status IN ('reserved', 'sent')", (error, list(attempt_ids)))

    def mark_sent(self, attempt_id: uuid.UUID) -> None:
        self.conn.execute("UPDATE model_attempts SET status = 'sent', updated_at = now() WHERE id = %s AND status = 'reserved'",
                          (attempt_id,))

    def complete(self, attempt_id: uuid.UUID, *, cost_usd: Decimal, usage: dict, response: dict | None,
                 latency_ms: int, model_version: str | None, stop_reason: str | None) -> None:
        self.conn.execute(
            "UPDATE model_attempts SET status = 'completed', cost_usd = %s, usage = %s, response = %s, latency_ms = %s, "
            "model_version = %s, stop_reason = %s, updated_at = now() WHERE id = %s",
            (cost_usd, json.dumps(usage), json.dumps(response) if response is not None else None, latency_ms,
             model_version, stop_reason, attempt_id))

    def failed(self, attempt_id: uuid.UUID, error: str) -> None:
        """Definitely not charged."""
        self.conn.execute("UPDATE model_attempts SET status = 'failed', cost_usd = 0, error = %s, updated_at = now() "
                          "WHERE id = %s", (error, attempt_id))

    def unknown(self, attempt_id: uuid.UUID, error: str) -> None:
        """Outcome unknown: keep the reservation; never resent automatically."""
        self.conn.execute("UPDATE model_attempts SET status = 'unknown', error = %s, updated_at = now() WHERE id = %s",
                          (error, attempt_id))

    def open_attempts(self) -> list[dict]:
        return self.conn.execute("SELECT id, scope, job_id, stage, attempt, status, reserved_usd, created_at, error "
                                 "FROM model_attempts WHERE paid AND status IN ('reserved', 'sent', 'unknown') "
                                 "ORDER BY created_at").fetchall()
