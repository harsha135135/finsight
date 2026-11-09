"""Contract ``finsight-qa-v1``: one answer to a free-text question about filings, of one of four types.

numeric               cited operands and a program; the value is the calculator's result (there is no model-typed
                      final number, so it cannot disagree with the calculation)
boolean               yes or no, with at least one cited claim
narrative             at least one cited claim; every figure in a claim must be in the passages it cites
insufficient_evidence the evidence does not answer the question; a reason, no operands

Separate from the FinQA protocol ``answer-v4`` and the report contract ``finsight-report-v1`` (both unchanged).
The model-facing schema is flat (``answer_type`` plus nullable fields). The answer type comes first: with the
evidence first, models emitted empty operand lists before deciding what they were answering (smoke test).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from finsight.contracts.models import CalculationStep, Contract, Scale

QA_CONTRACT = "finsight-qa-v1"
Short = Annotated[str, StringConstraints(min_length=1, max_length=200)]


class QAOperand(Contract):
    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
    passage_id: Short
    original_text: Short
    value: Annotated[str, StringConstraints(pattern=r"^-?[0-9]+(\.[0-9]+)?$", max_length=40)]
    scale: Scale
    unit: Annotated[str, StringConstraints(min_length=1, max_length=20)]
    period: Annotated[str, StringConstraints(min_length=2, max_length=40)]


class QAClaim(Contract):
    text: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    passage_ids: list[Short] = Field(min_length=1, max_length=8)


class QAOutput(Contract):
    answer_type: Literal["numeric", "boolean", "narrative", "insufficient_evidence"]
    boolean: Literal["yes", "no"] | None = None
    claims: list[QAClaim] = Field(default_factory=list, max_length=6)
    operands: list[QAOperand] = Field(default_factory=list, max_length=10)
    program: list[CalculationStep] = Field(default_factory=list, max_length=10)
    answer_unit: str | None = Field(default=None, max_length=40)
    missing: str | None = Field(default=None, max_length=1000)


class OperandCheck(Contract):
    name: str
    passage_id: str
    original_text: str
    value: str
    scale: Scale
    declared_period: str
    source_year: str | None
    supported: bool
    period: Literal["pass", "reject", "unverified"]
    note: str | None = None


class ClaimCheck(Contract):
    text: str
    passage_ids: list[str]
    supported: bool
    issues: list[str] = Field(default_factory=list)


class QAResult(Contract):
    contract: str = QA_CONTRACT
    answer_type: str
    status: Literal["accepted", "rejected", "insufficient_evidence"]
    value: str | None = None            # numeric: the calculator's result in base units
    unit: str | None = None
    boolean: str | None = None
    operands: list[OperandCheck] = Field(default_factory=list)
    claims: list[ClaimCheck] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    period_unverified: int = 0          # operands the source could not place (accepted only in lenient mode)
