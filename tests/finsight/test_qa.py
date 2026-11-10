from pathlib import Path

from finsight.contracts.qa import QAOutput
from finsight.contracts import Passage, PassageLocation
from finsight.ingestion.html import parse_filing
from finsight.qa import requested_years, validate_qa

PASSAGES = {p.id: p for p in parse_filing(Path("tests/fixtures/sec/synthetic-10k.html").read_text(), "sec:test", "mmm18")}
SALES, OPINC, CAPEX = "mmm18:item_8:t0r2", "mmm18:item_8:t0r3", "mmm18:item_8:t1r1"
TEXT = Passage(id="t1", document_id="d", section="s", location=PassageLocation(), start=0, end=40,
               text="Research spending was $1,821 million in the year.")
SUPPLIED = {**PASSAGES, "t1": TEXT}


def op(name, pid, text, value, period="FY2018", scale="millions"):
    return {"name": name, "passage_id": pid, "original_text": text, "value": value, "scale": scale, "unit": "USD", "period": period}


def qa(**fields):
    return QAOutput.model_validate({"operands": [], "program": [], "claims": [], "boolean": None, "answer_unit": None,
                                    "missing": None, **fields})


def test_numeric_answer_value_is_the_calculation_and_periods_are_checked_against_the_source():
    margin = qa(answer_type="numeric", operands=[op("oi", OPINC, "7,207", "7207"), op("rev", SALES, "32,765", "32765")],
                program=[{"op": "divide", "args": ["oi", "rev"]}])
    result = validate_qa(margin, SUPPLIED, {2018})
    assert result.status == "accepted" and result.value.startswith("0.2199") and result.unit == "ratio"
    assert [o.period for o in result.operands] == ["pass", "pass"]
    wrong_column = qa(answer_type="numeric", operands=[op("rev", SALES, "31,657", "31657", period="FY2018")])
    assert validate_qa(wrong_column, SUPPLIED, {2018}).operands[0].period == "reject"      # source says 2017
    not_asked = qa(answer_type="numeric", operands=[op("rev", SALES, "32,765", "32765")])
    assert validate_qa(not_asked, SUPPLIED, {2016}).status == "rejected"                   # 2018 figure, 2016 asked


def test_unplaceable_figure_is_unverified_strict_rejects_lenient_accepts():
    answer = qa(answer_type="numeric", operands=[op("rd", "t1", "$1,821 million", "1821")])
    strict, lenient = validate_qa(answer, SUPPLIED, {2018}), validate_qa(answer, SUPPLIED, {2018}, strict=False)
    assert strict.status == "rejected" and lenient.status == "accepted" and lenient.period_unverified == 1


def test_column_dates_map_to_fiscal_years_with_the_company_year_end():
    from finsight.qa import fiscal_year
    best_buy = (1, 28)                                                   # fiscal years end in late January
    assert fiscal_year("January 28, 2023", best_buy) == 2023 and fiscal_year("July 29, 2023", best_buy) == 2024
    assert fiscal_year("February 2, 2019", best_buy) == 2019             # 53-week year, within 7 days
    assert fiscal_year("December 31, 2022", (6, 30)) == 2023             # Amcor: year ends June 30
    assert fiscal_year("Years ended December 31 2018", (12, 31)) == 2018 and fiscal_year("2018") == 2018
    assert fiscal_year("2018 versus 2017") is None


def test_boolean_narrative_and_insufficient_evidence():
    assert validate_qa(qa(answer_type="boolean", boolean="yes"), SUPPLIED).status == "rejected"          # no claims
    claim = {"text": "Net sales were 32,765 in 2018.", "passage_ids": [SALES]}
    assert validate_qa(qa(answer_type="boolean", boolean="yes", claims=[claim]), SUPPLIED).status == "accepted"
    invented = {"text": "Net sales were 40,000.", "passage_ids": [SALES]}
    assert validate_qa(qa(answer_type="narrative", claims=[invented]), SUPPLIED).status == "rejected"
    assert validate_qa(qa(answer_type="narrative", claims=[claim]), SUPPLIED).status == "accepted"
    assert validate_qa(qa(answer_type="insufficient_evidence", missing="no FY2016 figures"), SUPPLIED).status == "insufficient_evidence"
    several = qa(answer_type="numeric", operands=[op("a", SALES, "32,765", "32765"), op("b", OPINC, "7,207", "7207")])
    assert validate_qa(several, SUPPLIED, {2018}).status == "rejected"                      # no program


def test_requested_years_cover_financebench_phrasings():
    assert requested_years("Q22023 revenue by region") == {2023}
    assert requested_years("as of FY 2023 and FY2022") == {2022, 2023}
    assert requested_years("between FY21 and FY22") == {2021, 2022}
    assert requested_years("quick ratio by Jun'23 close") == {2023}
    assert requested_years("Q2'23 operating margin") == {2023}
    assert requested_years("Does 3M maintain a stable trend of dividend distribution?") == set()
    assert requested_years("the FY2017 - FY2019 3 year average") == {2017, 2018, 2019}


def test_model_facing_schema_requires_the_evidence_each_answer_type_needs():
    from finsight.reasoning.qa_prompt import qa_schema
    branches = {b["properties"]["answer_type"]["const"]: b for b in qa_schema()["anyOf"]}
    assert set(branches) == {"numeric", "boolean", "narrative", "insufficient_evidence"}
    assert branches["numeric"]["properties"]["operands"]["minItems"] == 1
    assert branches["boolean"]["properties"]["claims"]["minItems"] == 1 and branches["narrative"]["properties"]["claims"]["minItems"] == 1
    assert branches["numeric"]["properties"]["operands"]["items"]["$ref"].endswith("QAOperand")


def test_every_requested_year_must_be_covered_by_a_placed_figure():
    one_year = qa(answer_type="numeric", operands=[op("rev", SALES, "32,765", "32765")])
    result = validate_qa(one_year, SUPPLIED, {2017, 2018})                 # growth question, only the 2018 figure cited
    assert result.status == "rejected" and result.coverage == "missing" and result.missing_years == [2017]
    both = qa(answer_type="numeric", operands=[op("r18", SALES, "32,765", "32765"), op("r17", SALES, "31,657", "31657", period="FY2017")],
              program=[{"op": "subtract", "args": ["r18", "r17"]}])
    assert validate_qa(both, SUPPLIED, {2017, 2018}).coverage == "complete"
    unplaced = qa(answer_type="numeric", operands=[op("rd", "t1", "$1,821 million", "1821")])
    lenient = validate_qa(unplaced, SUPPLIED, {2018}, strict=False)
    assert lenient.coverage == "unverified" and lenient.status == "accepted"   # unknown, not a period rejection


def test_claim_numbers_parse_fix_drops_trailing_punctuation():
    from finsight.qa import claim_numbers
    text = "net sales rose from 14,544 to 14,694, so margin improved"
    assert claim_numbers(text, "v1.6") == ["14,544", "14,694,"]
    assert claim_numbers(text, "parse") == ["14,544", "14,694"]


def test_reference_exemptions_apply_only_inside_identified_spans():
    from finsight.qa import claim_numbers
    assert claim_numbers("December 31, 2022 revenue was 31 million", "v1.7") == ["2022", "31"]   # the figure 31 is still checked
    assert claim_numbers("Item 9.01 lists the exhibits", "v1.7") == []
    assert claim_numbers("Note 6 shows 6% growth", "v1.7") == ["6"]
    assert claim_numbers("acquired on November 2, 2021, for 389", "v1.7") == ["2021", "389"]
    assert claim_numbers("Exhibit 21.1 and Section 12(b)", "v1.7") == []
    assert claim_numbers("dec 3.5 million", "v1.7") == ["3.5"]                                       # not a date: decimal follows
    assert claim_numbers("revenue of 1,234.5 and 0.62% and (5,022)", "v1.7") == ["1,234.5", "0.62", "5,022"]
    assert claim_numbers("Item 9.01", "parse") == ["9.01"]                                           # policy off: still a figure


def test_date_day_number_no_longer_rejects_a_supported_claim():
    passage = Passage(id="p1", document_id="d", section="s", location=PassageLocation(), start=0, end=60,
                      text="Total assets were $9,912 million at the end of 2022.")
    out = qa(answer_type="boolean", boolean="yes", claims=[{"text": "Assets were 9,912 million at December 31, 2022.", "passage_ids": ["p1"]}])
    assert validate_qa(out, {"p1": passage}, claim_policy="v1.6").status == "rejected"
    assert validate_qa(out, {"p1": passage}, claim_policy="v1.7").status == "accepted"


def test_month_prefixed_words_are_not_dates():
    from finsight.qa import claim_numbers
    assert claim_numbers("operating margin 22% and sales decreased 12 percent", "v1.7") == ["22", "12"]
    assert claim_numbers("declined 5 points; market 30 share; junior 4", "v1.7") == ["5", "30", "4"]
    assert claim_numbers("Dec. 31, 2022 and September 30 and May 31 and Sept 30", "v1.7") == ["2022"]
