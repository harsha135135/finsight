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


def test_fiscal_year_named_by_its_start_year_uses_the_offset():
    answer = qa(answer_type="numeric", operands=[op("rev", SALES, "32,765", "32765", period="FY2017")])
    assert validate_qa(answer, SUPPLIED, {2017}, fiscal_offset=1).status == "accepted"


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
