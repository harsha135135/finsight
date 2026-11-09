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
