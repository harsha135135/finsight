import json
import os
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from finsight.api.app import create_app
from finsight.ingestion.sec import SecSource
from finsight.watcher import check

pytestmark = pytest.mark.postgres
TEST_DB = os.environ.get("FINSIGHT_TEST_DATABASE_URL")
KEY = {"X-API-Key": "test-key"}
FIX = Path("tests/fixtures/sec")


def _client():
    return TestClient(create_app(TEST_DB, api_key="test-key"))


def test_dashboard_page_has_a_strict_csp_and_no_embedded_key(pg):
    with _client() as client:
        page = client.get("/dashboard")
        assert page.status_code == 200 and "default-src 'self'" in page.headers["content-security-policy"]
        assert "test-key" not in page.text and "<script src=\"/dashboard/app.js\">" in page.text
        assert client.get("/dashboard/app.js").status_code == 200 and client.get("/dashboard/other.js").status_code == 422
        assert client.get("/health").status_code == 401                       # data endpoints need the key


def test_health_models_job_list_pagination_and_claude_gate(pg):
    with _client() as client:
        health = client.get("/health", headers=KEY).json()
        assert health["database"] == "ok" and health["spend"]["claude_enabled"] is False
        assert all(o["provider"] != "claude" for o in client.get("/models", headers=KEY).json()["options"])
        body = {"company": "MMM", "question": "q", "period": "FY2018", "metrics": ["revenue"], "model": {"provider": "demo"}}
        for n in range(3):
            assert client.post("/research-jobs", json={**body, "idempotency_key": f"k{n}"}, headers=KEY).status_code == 202
        first = client.get("/research-jobs?limit=2", headers=KEY).json()
        assert len(first["items"]) == 2 and first["next_before"]
        rest = client.get(f"/research-jobs?limit=2&before={first['next_before']}", headers=KEY).json()
        assert len(rest["items"]) == 1 and rest["next_before"] is None
        refused = client.post("/research-jobs", json={**body, "model": {"provider": "claude"}}, headers=KEY)
        assert refused.status_code == 403 and "disabled" in refused.json()["detail"]


def _source(conn, tmp_path, extra_filing: dict | None = None):
    data = json.loads((FIX / "mmm-submissions.json").read_text())
    if extra_filing:
        recent = data["filings"]["recent"]
        for key in recent:
            recent[key] = [extra_filing.get(key, "")] + list(recent[key])
    routes = {"https://data.sec.gov/submissions/CIK0000066740.json": json.dumps(data).encode(),
              "https://data.sec.gov/submissions/CIK0000066740-submissions-001.json": (FIX / "mmm-submissions-001.json").read_bytes()}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=routes[str(r.url)]) if str(r.url) in routes else httpx.Response(404))
    return SecSource("FinSight test admin@example.com", tmp_path, conn=conn, transport=transport)


def test_watcher_records_a_baseline_then_each_new_filing_once_and_analysis_is_idempotent(pg, tmp_path):
    pg.execute("INSERT INTO watchlist (ticker, cik, auto_analyze) VALUES ('MMM', '0000066740', true)")
    recent_dates = json.loads((FIX / "mmm-submissions.json").read_text())["filings"]["recent"]["filingDate"]
    today = max(date.fromisoformat(d) for d in recent_dates) + timedelta(days=30)
    first = check(pg, _source(pg, tmp_path), "MMM", today=today, auto_model="qwen-test")
    assert first["baseline"] is True and first["new"] == [] and first["jobs"] == []
    baseline = pg.execute("SELECT count(*) AS n FROM filings_seen WHERE baseline").fetchone()["n"]
    new = {"accessionNumber": "0000066740-99-000001", "form": "10-K", "filingDate": today.isoformat(),
           "reportDate": (today - timedelta(days=40)).isoformat(), "primaryDocument": "mmm-new-10k.htm"}
    second = check(pg, _source(pg, tmp_path, new), "MMM", today=today, auto_model="qwen-test")
    assert second["baseline"] is False and second["new"] == ["0000066740-99-000001"] and len(second["jobs"]) == 1
    again = check(pg, _source(pg, tmp_path, new), "MMM", today=today, auto_model="qwen-test")
    assert again["new"] == [] and again["jobs"] == []                                       # recorded once
    assert pg.execute("SELECT count(*) AS n FROM filings_seen WHERE NOT baseline").fetchone()["n"] == 1 and baseline > 0
    with _client() as client:
        feed = client.get("/filings", headers=KEY).json()["items"]
        assert [f["accession"] for f in feed] == ["0000066740-99-000001"]
        a = client.post("/filings/0000066740-99-000001/analyze", json={"model": "qwen-test"}, headers=KEY).json()
        b = client.post("/filings/0000066740-99-000001/analyze", json={"model": "qwen-test"}, headers=KEY).json()
        assert a["id"] == b["id"] == second["jobs"][0]                                    # idempotent per accession + metrics
