"""Durability tests: real worker processes against Postgres (skipped unless FINSIGHT_TEST_DATABASE_URL is set)."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from finsight.graph import DemoStages
from finsight.storage import jobs
from finsight.supervisor import sweep
from finsight.worker import LOST_AUTHORITY_EXIT, serve

pytestmark = pytest.mark.postgres
TEST_DB = os.environ.get("FINSIGHT_TEST_DATABASE_URL")
REQUEST = {"company": "MMM", "question": "FY2018 revenue", "periods": ["FY2018"]}
MODEL = {"provider": "demo", "model": "demo"}


def _worker(tmp_path: Path, *extra: str) -> subprocess.Popen:
    env = {**os.environ, "FINSIGHT_STATE_DIR": str(tmp_path)}
    return subprocess.Popen([sys.executable, "-m", "finsight.worker", "--once", "--database-url", TEST_DB,
                             "--heartbeat", "0.3", *extra], env=env)


def _wait_for(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    raise AssertionError("condition not reached in time")


def _count(pg, job_id, kind, stage=None):
    sql = "SELECT count(*) AS n FROM job_events WHERE job_id = %s AND kind = %s" + (" AND stage = %s" if stage else "")
    return pg.execute(sql, (job_id, kind, stage) if stage else (job_id, kind)).fetchone()["n"]


def test_demo_job_completes_with_one_report(pg, tmp_path):
    job = jobs.create_job(pg, REQUEST, MODEL)
    assert serve(TEST_DB, "inproc", DemoStages(), tmp_path, once=True, heartbeat_interval=0.3) == 0
    assert jobs.get_job(pg, job["id"])["status"] == "completed"
    report = pg.execute("SELECT * FROM reports WHERE job_id = %s", (job["id"],)).fetchone()
    assert Path(report["json_path"]).exists() and _count(pg, job["id"], "stage_result") == 5


def test_sigkill_mid_node_resumes_without_repeating_stages_or_reports(pg, tmp_path):
    job = jobs.create_job(pg, REQUEST, MODEL)
    first = _worker(tmp_path, "--delay", "analysis=30")
    _wait_for(lambda: jobs.get_job(pg, job["id"])["stage"] == "analysis")
    first.send_signal(signal.SIGKILL)
    first.wait()
    assert jobs.get_job(pg, job["id"])["status"] == "running"
    second = _worker(tmp_path)
    assert second.wait(timeout=120) == 0
    assert jobs.get_job(pg, job["id"])["status"] == "completed"
    assert _count(pg, job["id"], "stage_result", "research") == 1      # completed stage reused, not re-run
    assert _count(pg, job["id"], "stage_result", "analysis") == 1
    assert _count(pg, job["id"], "taken_over") == 1
    assert pg.execute("SELECT count(*) AS n FROM reports WHERE job_id = %s", (job["id"],)).fetchone()["n"] == 1


def test_frozen_worker_is_terminated_and_cannot_write_or_publish(pg, tmp_path):
    job = jobs.create_job(pg, REQUEST, MODEL)
    frozen = _worker(tmp_path, "--delay", "analysis=8")
    _wait_for(lambda: jobs.get_job(pg, job["id"])["stage"] == "analysis")
    frozen.send_signal(signal.SIGSTOP)                                   # its TCP connections stay open
    time.sleep(1.5)
    terminated = sweep(pg, timeout_seconds=1)
    assert terminated and terminated[0]["job_id"] == str(job["id"])
    replacement = _worker(tmp_path)
    assert replacement.wait(timeout=120) == 0
    report_before = pg.execute("SELECT * FROM reports WHERE job_id = %s", (job["id"],)).fetchone()
    events_before = pg.execute("SELECT count(*) AS n FROM job_events WHERE job_id = %s", (job["id"],)).fetchone()["n"]
    frozen.send_signal(signal.SIGCONT)                                   # it wakes up mid-node and tries to continue
    assert frozen.wait(timeout=60) == LOST_AUTHORITY_EXIT
    assert jobs.get_job(pg, job["id"])["status"] == "completed"
    assert pg.execute("SELECT * FROM reports WHERE job_id = %s", (job["id"],)).fetchone() == report_before
    assert pg.execute("SELECT count(*) AS n FROM job_events WHERE job_id = %s", (job["id"],)).fetchone()["n"] == events_before
