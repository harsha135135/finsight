import uuid

import pytest

from finsight.storage import jobs
from finsight.storage.db import connect, migrate
import os

TEST_DB = os.environ.get("FINSIGHT_TEST_DATABASE_URL")

pytestmark = pytest.mark.postgres
REQUEST = {"company": "MMM", "question": "FY2018 revenue, operating margin and capex", "periods": ["FY2018"]}
MODEL = {"provider": "recorded", "model": "fixture"}


def test_migrations_are_idempotent_and_refuse_edits(pg, tmp_path):
    assert migrate(pg) == []
    (tmp_path / "0001_init.sql").write_text("SELECT 1;")
    with pytest.raises(RuntimeError, match="changed after it was applied"):
        migrate(pg, tmp_path)


def test_admission_is_idempotent(pg):
    first = jobs.create_job(pg, REQUEST, MODEL, idempotency_key="k1")
    again = jobs.create_job(pg, REQUEST, MODEL, idempotency_key="k1")
    assert first["id"] == again["id"] and first["status"] == "queued"


def test_only_one_session_can_own_a_job(pg):
    job = jobs.create_job(pg, REQUEST, MODEL)
    a, b = connect(TEST_DB, application_name="finsight-a"), connect(TEST_DB, application_name="finsight-b")
    claimed = jobs.claim(a, "worker-a")
    assert claimed and claimed[0]["id"] == job["id"]
    assert jobs.claim(b, "worker-b") is None          # a live session holds the advisory lock
    a.close(); b.close()


def test_takeover_after_owner_session_ends_and_stale_writes_fail(pg):
    job = jobs.create_job(pg, REQUEST, MODEL)
    a = connect(TEST_DB, application_name="finsight-a")
    _, token_a = jobs.claim(a, "worker-a")
    jobs.owned_update(a, job["id"], token_a, stage="research")
    a.close()                                          # the owner's dedicated session is gone
    b = connect(TEST_DB, application_name="finsight-b")
    row, token_b = jobs.claim(b, "worker-b")
    assert row["id"] == job["id"] and token_b != token_a
    kinds = [e["kind"] for e in jobs.events(pg, job["id"])]
    assert "taken_over" in kinds
    with pytest.raises(jobs.OwnershipLost):            # the old token no longer authorizes writes
        jobs.owned_update(pg, job["id"], token_a, stage="analysis")
    with pytest.raises(jobs.OwnershipLost):
        jobs.heartbeat(pg, job["id"], token_a)
    jobs.finish(b, job["id"], token_b, "completed")
    assert jobs.get_job(pg, job["id"])["status"] == "completed"
    b.close()


def test_cancel_queued_job_immediately(pg):
    job = jobs.create_job(pg, REQUEST, MODEL)
    assert jobs.request_cancel(pg, job["id"])["status"] == "cancelled"
    assert jobs.request_cancel(pg, uuid.uuid4()) is None
