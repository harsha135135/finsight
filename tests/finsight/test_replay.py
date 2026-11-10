"""Offline replay (mandatory in CI): the real pipeline re-runs the FY2025 demo job from the committed bundle with no
network, GPU or key, reproduces the original outcome, labels the report as a replay with its provenance, and fails
closed when a recording or an SEC response is missing."""

import json
import os
import shutil
from pathlib import Path

import httpx
import pytest

from finsight.replay import ReplayMiss, ReplayTransport
from finsight.storage import jobs
from finsight.worker import replay_stages, serve

BUNDLE = Path("evals/demo/replay")
TEST_DB = os.environ.get("FINSIGHT_TEST_DATABASE_URL")


def test_transport_refuses_anything_not_in_the_bundle():
    transport = ReplayTransport(BUNDLE / "sec")
    client = httpx.Client(transport=transport)
    assert client.get("https://data.sec.gov/submissions/CIK0000066740.json").status_code == 200
    with pytest.raises(ReplayMiss):
        client.get("https://data.sec.gov/submissions/CIK0000000001.json")
    with pytest.raises(ReplayMiss):
        client.post("https://data.sec.gov/submissions/CIK0000066740.json")


def _replay(pg, tmp_path, bundle: Path):
    spec = json.loads((BUNDLE / "job.json").read_text())
    job = jobs.create_job(pg, spec["request"], spec["model_config"])
    seen = {}

    def factory(j, conn, token):
        stages = replay_stages(j, conn, token, bundle, tmp_path / "store")
        seen["transport"] = stages.source.client._transport
        return stages
    serve(TEST_DB, "replay-test", factory, tmp_path, once=True, heartbeat_interval=0.3)
    row = pg.execute("SELECT * FROM reports WHERE job_id = %s", (job["id"],)).fetchone()
    report = json.loads(Path(row["json_path"]).read_text()) if row else None
    return spec, jobs.get_job(pg, job["id"]), report, row, seen


@pytest.mark.postgres
def test_replay_reproduces_the_fy2025_demo_outcome(pg, tmp_path):
    spec, status, report, row, seen = _replay(pg, tmp_path, BUNDLE)
    assert status["status"] == spec["expected_status"] == "completed", status["outcome_reason"]
    assert [[f["metric"], f["status"], f.get("value")] for f in report["findings"]] == spec["expected_findings"]
    statuses = {f["metric"]: f["status"] for f in report["findings"]}
    assert statuses["operating_margin"] == "accepted"
    assert {statuses[m] for m in ("revenue", "capex", "net_income")} == {"rejected"}   # wrong scale, caught by XBRL
    assert report["filing"]["accession"] == "0000066740-26-000014"
    replay = report["model"]["replay"]
    assert report["model"]["provider"] == "recorded" and replay["label"].startswith("CACHED REPLAY")
    assert replay["responses"][0]["job_id"] == "488aac69-a9f0-4910-be38-b17717bbef20"
    assert replay["responses"][0]["recorded_at"].startswith("2026-11-09")
    assert "CACHED REPLAY" in Path(row["markdown_path"]).read_text()
    index = json.loads((BUNDLE / "sec" / "index.json").read_text())["responses"]
    assert seen["transport"].requests and set(seen["transport"].requests) <= set(index)    # nothing outside the bundle


@pytest.mark.postgres
def test_replay_fails_closed_when_a_recording_is_missing(pg, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    data = json.loads((bundle / "recordings.json").read_text())
    data["entries"] = {}
    (bundle / "recordings.json").write_text(json.dumps(data))
    _, status, _, _, _ = _replay(pg, tmp_path, bundle)
    assert status["status"] == "failed" and "model call failed" in status["outcome_reason"]
    attempts = pg.execute("SELECT error FROM model_attempts WHERE job_id = %s", (status["id"],)).fetchall()
    assert attempts and all("no recording" in a["error"] for a in attempts)


@pytest.mark.postgres
def test_replay_fails_closed_when_an_sec_response_is_missing(pg, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    index = json.loads((bundle / "sec" / "index.json").read_text())
    index["responses"] = {u: e for u, e in index["responses"].items() if "companyfacts" not in u}
    (bundle / "sec" / "index.json").write_text(json.dumps(index))
    _, status, report, _, _ = _replay(pg, tmp_path, bundle)
    assert status["status"] == "failed" and "replay never falls back to the network" in (status["outcome_reason"] or "")


@pytest.mark.postgres
def test_replay_mode_refuses_live_providers_without_crashing_the_worker(pg, tmp_path):
    job = jobs.create_job(pg, {"company": "MMM", "period": "FY2025", "metrics": ["revenue"], "question": "revenue"},
                          {"provider": "qwen", "model": "Qwen/Qwen3-4B", "contract": "finsight-report-v1"})
    serve(TEST_DB, "replay-test", lambda j, conn, token: replay_stages(j, conn, token, BUNDLE, tmp_path), tmp_path,
          once=True, heartbeat_interval=0.3)
    status = jobs.get_job(pg, job["id"])
    assert status["status"] == "failed" and "replay mode serves recorded responses only" in status["outcome_reason"]
