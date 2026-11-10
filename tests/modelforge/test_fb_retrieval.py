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


def test_text3_query_drops_company_and_instruction_tail_and_normalises_years():
    from finsight.retrieval.search import query_terms_v3
    terms = query_terms_v3("Does Corning have positive working capital based on FY2022 data? If working capital is not a useful "
                           "or relevant metric for a company like this, then please state that and explain why.", "Corning CORNING")
    assert "corning" not in terms and "useful" not in terms and "explain" not in terms
    assert "2022" in terms and {"current", "assets", "liabilities"} <= set(terms)
    assert "purchases" not in terms                     # working capital is not capital expenditure
    assert len(terms) == len(set(terms))
    assert "2021" in query_terms_v3("revenue in FY21", None)
    assert "2023" in query_terms_v3("inventory in Q2 FY2023", None) and "2023" in query_terms_v3("Q2'23 margin", None)


def test_statement_title_detection():
    from finsight.ingestion.pdf import statement_title
    assert statement_title(["Table of Contents", "Consolidated Statements of Cash Flows"]) == "Consolidated Statements Of Cash Flows"
    assert statement_title(["SQUARE, INC.", "CONSOLIDATED", "BALANCE", "SHEETS"]) == "Consolidated Balance Sheets"
    assert statement_title(["Results of operations", "Net sales"]) is None
