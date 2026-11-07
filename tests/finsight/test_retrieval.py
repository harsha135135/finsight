from datetime import datetime, timezone
from pathlib import Path

import pytest

from finsight.ingestion.html import parse_filing
from finsight.retrieval.search import label_matches, period_matches, search, store_passages

HTML = Path("tests/fixtures/sec/synthetic-10k.html").read_text()


def test_metric_and_period_checks_are_strict():
    row = "Net sales | Years ended December 31 2018: $ 32,765 | Years ended December 31 2017: $ 31,657  [table: x]"
    assert label_matches(row, ["net sales"], True) and period_matches(row, 2018, True)
    assert not period_matches(row, 2016, True)
    assert not label_matches("Cost of sales | 2018: 1", ["net sales"], True)


@pytest.mark.postgres
def test_retrieval_finds_metric_and_period_together_or_reports_missing(pg):
    pg.execute("INSERT INTO documents (id, source_uri, sha256, artifact_path, retrieved_at) VALUES "
               "('sec:test', 'https://www.sec.gov/x', 'abc', '/tmp/x', %s)", (datetime.now(timezone.utc),))
    store_passages(pg, parse_filing(HTML, "sec:test", "mmm18"))
    revenue = search(pg, ["sec:test"], "revenue", 2018)
    assert revenue.status == "found" and revenue.best.text.startswith("Net sales") and revenue.best.section == "item_8"
    capex = search(pg, ["sec:test"], "capex", 2018)
    assert capex.best.text.startswith("Purchases of property, plant and equipment")
    assert search(pg, ["sec:test"], "operating_income", 2018).best.text.startswith("Operating income")
    missing_year = search(pg, ["sec:test"], "revenue", 2016)
    assert missing_year.status == "missing" and "2016" in missing_year.reason
    missing_metric = search(pg, ["sec:test"], "net_income", 2018)
    assert missing_metric.status == "missing"
