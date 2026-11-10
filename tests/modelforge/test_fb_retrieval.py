from modelforge import fb_retrieval as R


def test_page_coverage():
    assert R.page_coverage({3, 4}, [3, 4]) == "all"
    assert R.page_coverage({3}, [3, 4]) == "some"
    assert R.page_coverage({9}, [3, 4]) == "none"


def test_table_needle_needs_value_year_and_label_in_one_passage():
    needle = {"kind": "table_value", "label": "Merchandise inventories", "year": 2019, "value": "5,409"}
    assert R.needle_reached(needle, ["Merchandise inventories | 2019: 5,409 | 2018: 5,209  [table: page 40]"])
    assert R.needle_reached(needle, ["Merchandiseinventories 5,409 5,209 February 2, 2019"])          # squashed extraction
    assert not R.needle_reached(needle, ["Consolidated Balance Sheets (in millions) 2019 2018 Current assets:"])  # header only
    assert not R.needle_reached(needle, ["Merchandise inventories", "2019: 5,409"])                       # split across passages


def test_line_needle_uses_its_label_for_bare_figures():
    needle = {"kind": "evidence_line", "text": "1,093 $", "label": "Cash and cash equivalents"}
    assert R.needle_reached(needle, ["Cash and cash equivalents 1,093 $ 1,511"])
    assert not R.needle_reached(needle, ["Total assets 1,093"])


def test_needles_from_gold_evidence_lines():
    q = {"answer": "The quick ratio was 0.96", "justification": "(15,754-5,280)/10,936",
         "evidence": [{"evidence_text": "Total current assets\n15,754\nInventories\n5,280\nTotal current liabilities\n10,936\nOther 77"}]}
    needles = R.evidence_needles(q, None)
    assert [n["text"] for n in needles] == ["15,754", "5,280", "10,936"]
    assert needles[0]["label"] == "Total current assets"
