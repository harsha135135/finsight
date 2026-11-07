import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from finsight.ingestion.sec import SecSource, SourceError, select_facts

FIX = Path("tests/fixtures/sec")
UA = "FinSight test admin@example.com"


def _transport(calls: list | None = None, overrides: dict | None = None):
    routes = {"https://data.sec.gov/submissions/CIK0000066740.json": (200, (FIX / "mmm-submissions.json").read_bytes()),
              "https://data.sec.gov/submissions/CIK0000066740-submissions-001.json": (200, (FIX / "mmm-submissions-001.json").read_bytes()),
              "https://data.sec.gov/api/xbrl/companyfacts/CIK0000066740.json": (200, (FIX / "mmm-companyfacts-fy2018.json").read_bytes()),
              "https://www.sec.gov/Archives/edgar/data/66740/000155837019000470/mmm-20181231x10k.htm": (200, b"<html><body>3M 10-K</body></html>")}
    routes.update(overrides or {})

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        status, payload = routes.get(str(request.url), (404, b""))
        if isinstance(payload, dict):
            return httpx.Response(status, headers=payload)
        return httpx.Response(status, content=payload)
    return httpx.MockTransport(handler)


def _source(tmp_path, **kwargs):
    return SecSource(UA, tmp_path, transport=_transport(**kwargs))


def test_user_agent_needs_contact(tmp_path):
    with pytest.raises(SourceError, match="contact"):
        SecSource("FinSight research prototype", tmp_path)


def test_fy2018_resolves_to_the_original_10k_and_records_the_amendment(tmp_path):
    filings, fye = _source(tmp_path).filings("0000066740")
    assert fye == "1231" and any(f.accession == "0000066740-06-000001" for f in filings)   # older page was read
    choice = SecSource.choose_annual(filings, 2018, fye)
    assert choice.filing.accession == "0001558370-19-000470" and choice.filing.form == "10-K"
    assert [a.form for a in choice.amendments] == ["10-K/A"]
    with pytest.raises(SourceError, match="filed by"):
        SecSource.choose_annual(filings, 2018, fye, cutoff=date(2019, 1, 31))           # before the 10-K was filed


def test_facts_are_selected_by_accession_dates_and_unit(tmp_path):
    facts, snapshot = _source(tmp_path).company_facts_snapshot("0000066740")
    assert Path(snapshot["artifact_path"]).exists() and snapshot["kind"] == "sec_companyfacts"
    acc = "0001558370-19-000470"
    revenue = select_facts(facts, "Revenues", "USD", acc, date(2018, 1, 1), date(2018, 12, 31))
    opinc = select_facts(facts, "OperatingIncomeLoss", "USD", acc, date(2018, 1, 1), date(2018, 12, 31))
    capex = select_facts(facts, "PaymentsToAcquirePropertyPlantAndEquipment", "USD", acc, date(2018, 1, 1), date(2018, 12, 31))
    assert (revenue["as_reported"], opinc["as_reported"], capex["as_reported"]) == (32_765_000_000, 7_207_000_000, 1_577_000_000)
    assert revenue["later_filings"] and all(f["accn"] != acc for f in revenue["later_filings"])   # later comparatives kept apart
    prior = select_facts(facts, "Revenues", "USD", acc, date(2017, 1, 1), date(2017, 12, 31))
    assert prior["as_reported"] is not None                                                   # the prior year as reported in FY2018's 10-K


def test_redirects_are_validated_and_sizes_limited(tmp_path, monkeypatch):
    bad = {"https://data.sec.gov/submissions/CIK0000000001.json": (302, {"location": "http://data.sec.gov/x"}),
           "https://data.sec.gov/submissions/CIK0000000002.json": (302, {"location": "https://evil.example/x"})}
    source = _source(tmp_path, overrides=bad)
    for cik in ("0000000001", "0000000002"):
        with pytest.raises(SourceError, match="only https"):
            source.submissions(cik)
    monkeypatch.setitem(__import__("finsight.ingestion.sec", fromlist=["MAX_BYTES"]).MAX_BYTES, "json", 10)
    with pytest.raises(SourceError, match="exceeds"):
        source.submissions("0000066740")


@pytest.mark.postgres
def test_repeated_request_reuses_the_stored_filing(pg, tmp_path):
    calls: list = []
    source = SecSource(UA, tmp_path, conn=pg, transport=_transport(calls))
    filings, fye = source.filings("0000066740")
    filing = SecSource.choose_annual(filings, 2018, fye).filing
    first = source.fetch_document(filing)
    downloads = sum("Archives" in c for c in calls)
    second = source.fetch_document(filing)
    assert downloads == 1 and sum("Archives" in c for c in calls) == 1                          # served from the store
    assert first["sha256"] == second["sha256"] and second["accession"] == "0001558370-19-000470"


@pytest.mark.postgres
def test_shared_rate_limit_spaces_requests(pg):
    import time
    from finsight.ingestion.sec import PostgresRateLimiter
    limiter = PostgresRateLimiter(pg, per_second=10)
    start = time.monotonic()
    for _ in range(4):
        limiter.wait("data.sec.gov")
    assert time.monotonic() - start >= 0.25
