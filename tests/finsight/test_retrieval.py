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


# Row shapes copied from 3M's FY2018 10-K (accession 0001558370-19-000470): segment, geographic, quarterly and
# non-GAAP tables share labels with the consolidated statements and must rank below them.
REAL_ROWS = [
    ("item_8", "170", "Net sales | 2018: $ 32,765 | 2017: $ 31,657 | 2016: $ 30,109  [table: Years ended December 31; units: millions]"),
    ("item_8", "170", "Cost of sales | 2018: 16,682 | 2017: 16,055 | 2016: 15,118  [table: Years ended December 31; units: millions]"),
    ("item_8", "170", "Operating income | 2018: 7,207 | 2017: 7,692 | 2016: 7,027  [table: Years ended December 31; units: millions]"),
    ("item_7", "72", "Net sales (millions) | Year ended December 31, 2018 United States: $ 12,840 | Year ended December 31, 2018 Asia Pacific: $ 10,254  [table: Worldwide; units: millions]"),
    ("item_7", "104", "Sales (millions) | 2018: $ 12,267 | 2017: $ 11,866 | 2016: $ 11,217  [table: Industrial Business (37.4% of consolidated sales):; units: millions]"),
    ("item_7", "104", "Operating income (millions) | 2018: $ 2,737 | 2017: $ 2,490 | 2016: $ 2,528  [table: Industrial Business (37.4% of consolidated sales):; units: millions]"),
    ("item_8", "265", "Net sales | First Quarter: 8,278 | Second Quarter: 8,390 | Third Quarter: 8,152 | Fourth Quarter: 7,945 | Year 2018: 32,765  [table: NOTE 20. Quarterly Data (Unaudited); units: millions]"),
    ("item_7", "166", "Purchases of property, plant and equipment (PP&E) | 2018: (1,577) | 2017: (1,373) | 2016: (1,420)  [table: Free Cash Flow (non-GAAP measure):; units: millions]"),
    ("item_8", "175", "Purchases of property, plant and equipment (PP&E) | 2018: (1,577) | 2017: (1,373) | 2016: (1,420)  [table: Years ended December 31; units: millions]"),
    ("item_8", "170", "Net income including noncontrolling interest | 2018: $ 5,363 | 2017: $ 4,869 | 2016: $ 5,058  [table: Years ended December 31; units: millions]"),
    ("item_8", "170", "Net income attributable to 3M | 2018: $ 5,349 | 2017: $ 4,858 | 2016: $ 5,050  [table: Years ended December 31; units: millions]"),
    ("item_8", None, "In May 2014, the FASB issued ASU No. 2014-09, Revenue from Contracts with Customers, which supersedes revenue recognition guidance; revenue and sales are discussed at length here, and the update applies to revenue in 2018."),
]


@pytest.mark.postgres
def test_real_filing_row_shapes_rank_consolidated_statements_first(pg):
    from finsight.contracts import Passage, PassageLocation
    pg.execute("INSERT INTO documents (id, source_uri, sha256, artifact_path, retrieved_at) VALUES "
               "('sec:real', 'https://www.sec.gov/y', 'def', '/tmp/y', %s)", (datetime.now(timezone.utc),))
    passages = [Passage(id=f"real:{section}:{i}", document_id="sec:real", section=section,
                        location=PassageLocation(html_section=section, table=table, row=i if table else None),
                        start=i * 1000, end=i * 1000 + len(text), text=text)
                for i, (section, table, text) in enumerate(REAL_ROWS)]
    store_passages(pg, passages)
    best = {m: search(pg, ["sec:real"], m, 2018).best.text for m in ("revenue", "operating_income", "capex", "net_income")}
    assert best["revenue"].startswith("Net sales | 2018: $ 32,765")
    assert best["operating_income"].startswith("Operating income | 2018: 7,207")
    assert best["capex"].endswith("[table: Years ended December 31; units: millions]")
    assert best["net_income"].startswith("Net income attributable to 3M")
    assert search(pg, ["sec:real"], "revenue", 2015).status == "missing"
