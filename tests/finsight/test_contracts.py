from decimal import Decimal

import pytest
from pydantic import ValidationError

from finsight.contracts import AnswerOutput, Operand, Passage, ResearchJob, Scale


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ResearchJob(task="answer", question="q", surprise=True)


def test_decimals_round_trip_as_strings():
    operand = Operand(name="revenue_fy2025", original_text="$1.2 billion", value=Decimal("1.2"),
                      unit="USD", scale=Scale.BILLIONS, period="FY2025", passage_id="p1")
    dumped = operand.model_dump(mode="json")
    assert dumped["value"] == "1.2"
    assert Operand.model_validate(dumped) == operand
    assert operand.base_value == Decimal("1200000000.0")


def test_passage_offsets_must_be_ordered():
    with pytest.raises(ValidationError):
        Passage(id="p1", document_id="d1", section="Item 7", start=10, end=2, text="x")


def test_operand_names_must_be_identifiers():
    with pytest.raises(ValidationError):
        Operand(name="1bad name", original_text="1", value=Decimal(1), passage_id="p")


def test_answer_schema_is_strict_json_schema():
    schema = AnswerOutput.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {"abstain", "operands", "program", "answer", "claims"}
