from pathlib import Path

from finsight.ingestion.html import parse_filing

HTML = Path("tests/fixtures/sec/synthetic-10k.html").read_text()


def test_table_rows_keep_labels_periods_units_and_titles():
    passages = parse_filing(HTML, "sec:test", "mmm18")
    by_label = {p.text.split(" | ")[0]: p for p in passages if p.location.table is not None}
    sales = by_label["Net sales"]
    assert "2018: $ 32,765" in sales.text and "2017: $ 31,657" in sales.text
    assert "units: millions" in sales.text and "Consolidated Statement of Income" in sales.text
    assert sales.section == "item_8" and sales.location.html_section == "item_8" and sales.location.row is not None
    assert "(207)" in by_label["Other expense (income), net"].text           # split parenthesis cells re-joined
    capex = by_label["Purchases of property, plant and equipment (PP&E)"]
    assert "2018: (1,577)" in capex.text and "Cash Flows" in capex.text


def test_text_sections_ids_offsets_and_hidden_content():
    passages = parse_filing(HTML, "sec:test", "mmm18")
    mdna = [p for p in passages if p.section == "item_7"]
    assert any("22.0 percent" in p.text for p in mdna)
    assert not any("hidden xbrl" in p.text for p in passages)
    assert len({p.id for p in passages}) == len(passages)                  # stable, unique ids
    assert all(a.end <= b.start for a, b in zip(passages, passages[1:]))   # ordered, non-overlapping offsets
    assert parse_filing(HTML, "sec:test", "mmm18") == passages              # deterministic
