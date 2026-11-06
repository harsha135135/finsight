"""Includes the flagship fixture: operating margin change FY2024 → FY2025."""

from finsight.contracts import AnswerOutput, Passage, ReasonCode
from finsight.validation import validate_answer

PASSAGES = {
    "is-2025": Passage(id="is-2025", document_id="acme-10k-2025", section="Item 8", start=0, end=120,
                       text="(in millions) FY2025: Revenue 125.0; Operating income 25.0"),
    "is-2024": Passage(id="is-2024", document_id="acme-10k-2025", section="Item 8", start=121, end=240,
                       text="(in millions) FY2024: Revenue 100.0; Operating income 15.0"),
}


def operand(name, text, value, period, pid):
    return {"name": name, "original_text": text, "value": value, "unit": "USD", "scale": "millions",
            "period": period, "passage_id": pid}


def margin_answer(**overrides):
    data = {
        "abstain": False, "abstain_reason": None,
        "operands": [
            operand("rev25", "Revenue 125.0", "125.0", "FY2025", "is-2025"),
            operand("oi25", "Operating income 25.0", "25.0", "FY2025", "is-2025"),
            operand("rev24", "Revenue 100.0", "100.0", "FY2024", "is-2024"),
            operand("oi24", "Operating income 15.0", "15.0", "FY2024", "is-2024"),
        ],
        "program": [
            {"op": "divide", "args": ["oi25", "rev25"]}, {"op": "to_percent", "args": ["#0"]},
            {"op": "divide", "args": ["oi24", "rev24"]}, {"op": "to_percent", "args": ["#2"]},
            {"op": "subtract", "args": ["#1", "#3"]},
        ],
        "answer": "5", "answer_unit": "percentage points",
        "claims": [{"text": "Operating margin rose from 15% to 20%.", "passage_ids": ["is-2025", "is-2024"],
                    "calculation_steps": [1, 3, 4], "kind": "calculated", "confidence": "high"}],
    }
    data.update(overrides)
    return AnswerOutput.model_validate(data)


def test_operating_margin_change_is_answered_with_executed_value():
    result = validate_answer(margin_answer(), PASSAGES, required_periods=["FY2024", "FY2025"])
    assert result.reason_code is ReasonCode.ANSWERED, result.issues
    assert result.answer == "5"
    assert result.calculation.values[1] == "20" and result.calculation.values[3] == "15"


def test_missing_year_evidence_leads_to_evidence_abstention_when_model_abstains():
    only_2025 = {"is-2025": PASSAGES["is-2025"]}
    abstained = AnswerOutput(abstain=True, abstain_reason="FY2024 figures are not in the evidence", operands=[],
                             program=[], answer=None, answer_unit=None, claims=[])
    assert validate_answer(abstained, only_2025).reason_code is ReasonCode.ABSTAIN_EVIDENCE_ABSENT


def test_answering_without_required_period_is_a_contradiction_not_an_abstention():
    answer = margin_answer(operands=margin_answer().model_dump()["operands"][:2],
                           program=[{"op": "divide", "args": ["oi25", "rev25"]}], answer="0.2")
    result = validate_answer(answer, PASSAGES, required_periods=["FY2024", "FY2025"])
    assert result.reason_code is ReasonCode.VALIDATION_CONTRADICTION


def test_citing_an_unsupplied_passage_is_invalid_output():
    bad = margin_answer().model_dump()
    bad["operands"][0]["passage_id"] = "invented"
    assert validate_answer(AnswerOutput.model_validate(bad), PASSAGES).reason_code is ReasonCode.MODEL_OUTPUT_INVALID


def test_operand_value_not_in_cited_text_is_a_contradiction():
    bad = margin_answer().model_dump()
    bad["operands"][1]["value"] = "30.0"
    result = validate_answer(AnswerOutput.model_validate(bad), PASSAGES)
    assert result.reason_code is ReasonCode.VALIDATION_CONTRADICTION


def test_stated_answer_must_match_execution():
    result = validate_answer(margin_answer(answer="7"), PASSAGES)
    assert result.reason_code is ReasonCode.VALIDATION_CONTRADICTION


def test_malformed_program_is_invalid_output():
    result = validate_answer(margin_answer(program=[{"op": "add", "args": ["#3", "oi25"]}]), PASSAGES)
    assert result.reason_code is ReasonCode.MODEL_OUTPUT_INVALID


def test_abstaining_while_answering_is_invalid_output():
    result = validate_answer(margin_answer(abstain=True), PASSAGES)
    assert result.reason_code is ReasonCode.MODEL_OUTPUT_INVALID


def test_rounded_stated_answer_is_consistent():
    answer = margin_answer(program=[{"op": "divide", "args": ["oi24", "rev25"]}], answer="0.12")
    assert validate_answer(answer, PASSAGES).reason_code is ReasonCode.ANSWERED
