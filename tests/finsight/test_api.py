import os

import pytest
from fastapi.testclient import TestClient

from finsight.api.app import create_app
from finsight.graph import DemoStages
from finsight.worker import serve

pytestmark = pytest.mark.postgres
TEST_DB = os.environ.get("FINSIGHT_TEST_DATABASE_URL")
KEY = {"X-API-Key": "test-key"}
BODY = {"company": "MMM", "question": "Report FY2018 revenue, operating margin and capital expenditure.",
        "period": "FY2018", "metrics": ["revenue", "operating_margin", "capex"], "model": {"provider": "demo"}}


def _client():
    return TestClient(create_app(TEST_DB, api_key="test-key"))


def test_api_requires_key_and_validates(pg):
    with _client() as client:
        assert client.post("/research-jobs", json=BODY).status_code == 401
        assert client.post("/research-jobs", json={**BODY, "unexpected": 1}, headers=KEY).status_code == 422


def test_submit_survives_api_restart_and_returns_report_after_worker(pg, tmp_path):
    with _client() as client:
        created = client.post("/research-jobs", json=BODY, headers=KEY)
        assert created.status_code == 202
        job_id = created.json()["id"]
        assert client.get(f"/research-jobs/{job_id}/report", headers=KEY).status_code == 404
    with _client() as restarted:                                   # a new API process sees the same job
        status = restarted.get(f"/research-jobs/{job_id}", headers=KEY).json()
        assert status["status"] == "queued" and status["model_config"]["contract"] == "finsight-report-v1"
    serve(TEST_DB, "api-test", DemoStages(), tmp_path, once=True, heartbeat_interval=0.3)
    with _client() as client:
        assert client.get(f"/research-jobs/{job_id}", headers=KEY).json()["status"] == "completed"
        report = client.get(f"/research-jobs/{job_id}/report", headers=KEY)
        assert report.status_code == 200 and report.json()["report"]["request"]["company"] == "MMM"
        assert client.get(f"/research-jobs/{job_id}/report?format=markdown", headers=KEY).text.startswith("# Report")
        assert client.get(f"/research-jobs/{job_id}/evidence", headers=KEY).status_code == 200


def test_idempotency_and_cancel(pg):
    with _client() as client:
        first = client.post("/research-jobs", json={**BODY, "idempotency_key": "abc"}, headers=KEY).json()
        again = client.post("/research-jobs", json={**BODY, "idempotency_key": "abc"}, headers=KEY).json()
        assert first["id"] == again["id"]
        cancelled = client.post(f"/research-jobs/{first['id']}/cancel", headers=KEY).json()
        assert cancelled["status"] == "cancelled"
        assert client.get("/research-jobs/00000000-0000-0000-0000-000000000000", headers=KEY).status_code == 404
