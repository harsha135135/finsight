"""Stage 6 acceptance, offline: the demo request yields a cited report; an unsupported question yields
insufficient evidence. SEC is mocked with fixtures (real trimmed 3M companyfacts, synthetic 10-K HTML)."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from finsight.contracts import ModelAttempt
from finsight.ingestion.sec import SecSource
from finsight.reasoning.base import Generation
from finsight.stages import ResearchStages
from finsight.storage import jobs
from finsight.worker import serve

pytestmark = pytest.mark.postgres
TEST_DB = os.environ.get("FINSIGHT_TEST_DATABASE_URL")
FIX = Path("tests/fixtures/sec")
FY2018_URL = "https://www.sec.gov/Archives/edgar/data/66740/000155837019000470/mmm-20181231x10k.htm"


def _sec_transport():
    routes = {"https://www.sec.gov/files/company_tickers.json": json.dumps({"0": {"cik_str": 66740, "ticker": "MMM", "title": "3M CO"}}).encode(),
              "https://data.sec.gov/submissions/CIK0000066740.json": (FIX / "mmm-submissions.json").read_bytes(),
              "https://data.sec.gov/submissions/CIK0000066740-submissions-001.json": (FIX / "mmm-submissions-001.json").read_bytes(),
              "https://data.sec.gov/api/xbrl/companyfacts/CIK0000066740.json": (FIX / "mmm-companyfacts-fy2018.json").read_bytes(),
              FY2018_URL: (FIX / "synthetic-10k.html").read_bytes()}
    return httpx.MockTransport(lambda r: httpx.Response(200, content=routes[str(r.url)]) if str(r.url) in routes
                               else httpx.Response(404))


class ScriptedAnalyst:
    """Stands in for a model: reads the supplied passages and answers in the report contract."""
    provider, model, ledger = "scripted", "scripted", None

    def generate(self, system, user, schema, **kwargs):
        rows = dict(re.findall(r"^\[([^\]]+)\] ([^|\n]+) \|", user, re.M))
        find = lambda label: next((pid for pid, text in rows.items() if text.strip().lower().startswith(label)), None)
        requested = re.findall(r"^## metric (\w+), period (FY\d{4})", user, re.M)
        findings = []
        for metric, period in requested:
            op = lambda name, pid, text, value: {"name": name, "passage_id": pid, "original_text": text, "value": value,
                                                  "scale": "millions", "unit": "USD", "period": period}
            if metric == "revenue" and find("net sales"):
                findings.append({"metric": metric, "period": period, "status": "answered", "unit": "USD", "program": [],
                                 "operands": [op("revenue", find("net sales"), "32,765", "32765")], "missing": None,
                                 "explanation": [{"text": "Net sales were $32,765 million.", "passage_ids": [find("net sales")]}]})
            elif metric == "operating_margin" and find("operating income") and find("net sales"):
                findings.append({"metric": metric, "period": period, "status": "answered", "unit": "ratio", "missing": None,
                                 "operands": [op("operating_income", find("operating income"), "7,207", "7207"),
                                              op("revenue", find("net sales"), "32,765", "32765")],
                                 "program": [{"op": "divide", "args": ["operating_income", "revenue"]}], "explanation": []})
            elif metric == "capex" and find("purchases of property"):
                findings.append({"metric": metric, "period": period, "status": "answered", "unit": "USD", "program": [],
                                 "operands": [op("capex", find("purchases of property"), "(1,577)", "-1577")],
                                 "explanation": [], "missing": None})
            else:
                findings.append({"metric": metric, "period": period, "status": "insufficient_evidence", "operands": [],
                                 "program": [], "unit": None, "explanation": [], "missing": f"no passage reports {metric}"})
        attempt = ModelAttempt(requested_model="scripted", stage="analysis", attempt=kwargs.get("attempt", 1), status="ok",
                               started_at=datetime.now(timezone.utc))
        return Generation(text=json.dumps({"findings": findings, "unresolved": []}), attempts=[attempt])


def _run(pg, tmp_path, request):
    job = jobs.create_job(pg, request, {"provider": "recorded", "model": "scripted", "contract": "finsight-report-v1"})
    factory = lambda j, conn, token: ResearchStages(conn=conn, job_id=j["id"], token=token,
                                                    source=SecSource("FinSight test admin@example.com", tmp_path, conn=conn,
                                                                     transport=_sec_transport()),
                                                    provider=ScriptedAnalyst(), model_config=j["model_config"])
    serve(TEST_DB, "pipeline-test", factory, tmp_path, once=True, heartbeat_interval=0.3)
    row = pg.execute("SELECT * FROM reports WHERE job_id = %s", (job["id"],)).fetchone()
    return jobs.get_job(pg, job["id"]), (json.loads(Path(row["json_path"]).read_text()) if row else None), row


def test_demo_request_produces_a_cited_report_that_matches_xbrl(pg, tmp_path):
    status, report, row = _run(pg, tmp_path, {"company": "MMM", "period": "FY2018", "metrics": [],
                                              "question": "Using 3M's FY2018 annual report, report revenue, operating margin and "
                                                          "capital expenditure. Show the calculations and cite each source."})
    assert status["status"] == "completed", status["outcome_reason"]
    findings = {f["metric"]: f for f in report["findings"]}
    assert set(findings) == {"revenue", "operating_margin", "capex"} and all(f["status"] == "accepted" for f in findings.values())
    assert findings["revenue"]["value"] == "32765000000" and findings["capex"]["value"] == "-1577000000"
    assert findings["operating_margin"]["value"].startswith("0.2199")
    assert all("(agrees)" in f["checks"][0] for f in findings.values())
    assert report["filing"]["accession"] == "0001558370-19-000470" and report["filing"]["form"] == "10-K"
    assert report["filing"]["amendments"] == ["0000066740-19-999999"]                 # recorded, not used
    assert all(op["source_period"] == "2018" for f in findings.values() for op in f["operands"])
    md = Path(row["markdown_path"]).read_text()
    assert "## Calculations and sources" in md and "0001558370-19-000470" in md and "22.0%" in md


def test_metric_missing_from_the_evidence_reports_insufficient_evidence(pg, tmp_path):
    status, report, _ = _run(pg, tmp_path, {"company": "MMM", "period": "FY2018", "metrics": ["net_income"],
                                            "question": "What was 3M's FY2018 net income?"})
    assert status["status"] == "insufficient_evidence"
    assert report["findings"][0]["status"] == "insufficient_evidence" and report["unresolved"]


def test_unsupported_question_reports_insufficient_evidence_without_a_model_call(pg, tmp_path):
    status, report, row = _run(pg, tmp_path, {"company": "MMM", "period": "FY2018", "metrics": [],
                                              "question": "How many employees did 3M have in Antarctica in FY2018?"})
    assert status["status"] == "insufficient_evidence"
    assert "supported:" in report["unresolved"][0] and report["filing"] == {}
    assert pg.execute("SELECT count(*) AS n FROM model_attempts WHERE job_id = %s", (status["id"],)).fetchone()["n"] == 0
    assert "no filing was used" in Path(row["markdown_path"]).read_text()


def test_period_without_a_filing_reports_insufficient_evidence(pg, tmp_path):
    status, report, _ = _run(pg, tmp_path, {"company": "MMM", "period": "FY2031", "metrics": ["revenue"],
                                            "question": "What was 3M's FY2031 revenue?"})
    assert status["status"] == "insufficient_evidence" and "no 10-K for fiscal year 2031" in report["unresolved"][0]


def test_figure_that_disagrees_with_xbrl_is_rejected(pg, tmp_path, monkeypatch):
    import finsight.stages as stages
    real = stages.annual_fact

    def restated_revenue(facts, concepts, accession, end):
        fact = real(facts, concepts, accession, end)
        return {**fact, "value": fact["value"] + 1_000_000} if fact and fact["concept"] == "Revenues" else fact
    monkeypatch.setattr(stages, "annual_fact", restated_revenue)
    status, report, _ = _run(pg, tmp_path, {"company": "MMM", "period": "FY2018", "metrics": ["revenue", "capex"],
                                            "question": "FY2018 revenue and capex"})
    findings = {f["metric"]: f for f in report["findings"]}
    assert findings["capex"]["status"] == "accepted" and status["status"] == "completed"
    assert findings["revenue"]["status"] == "rejected" and "differs from XBRL" in " ".join(findings["revenue"]["issues"])


def test_unreachable_model_server_fails_operationally_and_recovers(pg, tmp_path, monkeypatch):
    state = {"up": False, "calls": 0}

    def server_models(self):
        if not state["up"]:
            raise httpx.ConnectError("tunnel down")
        return ["scripted"]
    original = ScriptedAnalyst.generate

    def counted(self, *args, **kwargs):
        state["calls"] += 1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(ScriptedAnalyst, "server_models", server_models, raising=False)
    monkeypatch.setattr(ScriptedAnalyst, "generate", counted)
    request = {"company": "MMM", "period": "FY2018", "metrics": ["revenue"], "question": "FY2018 revenue"}
    down, _, _ = _run(pg, tmp_path, dict(request, question="FY2018 revenue (down)"))
    assert down["status"] == "failed" and "model server unavailable" in down["outcome_reason"] and state["calls"] == 0
    state["up"] = True
    up, report, _ = _run(pg, tmp_path, dict(request, question="FY2018 revenue (up)"))
    assert up["status"] == "completed" and report["findings"][0]["status"] == "accepted"
