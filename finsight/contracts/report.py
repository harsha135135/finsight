"""Application contract ``finsight-report-v1``: independently validated findings.

Separate from the FinQA benchmark protocol (``answer-v4``, unchanged). A model returns one
``FindingOutput`` per requested metric; each is validated on its own, so revenue, operating
margin and capital expenditure succeed or fail independently. Financial quantities must come
from cited operands: program literals are limited to ``PERMITTED_CONSTANTS``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from finsight.contracts.models import CalculationStep, Contract, Scale

REPORT_CONTRACT = "finsight-report-v1"
PERMITTED_CONSTANTS = frozenset({Decimal(1), Decimal(100), Decimal(1000), Decimal(1_000_000), Decimal(1_000_000_000)})

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
Text = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class EvidenceOperand(Contract):
    name: Identifier
    passage_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    original_text: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    value: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    scale: Scale
    unit: Annotated[str, StringConstraints(min_length=1, max_length=20)]
    period: Annotated[str, StringConstraints(min_length=2, max_length=32)]


class ClaimOutput(Contract):
    text: Text
    passage_ids: list[str] = Field(min_length=1, max_length=8)


class FindingOutput(Contract):
    metric: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    period: Annotated[str, StringConstraints(min_length=2, max_length=32)]
    status: Literal["answered", "insufficient_evidence"]
    operands: list[EvidenceOperand] = Field(default_factory=list, max_length=8)
    program: list[CalculationStep] = Field(default_factory=list, max_length=8)
    unit: str | None = None
    explanation: list[ClaimOutput] = Field(default_factory=list, max_length=6)
    missing: str | None = Field(default=None, max_length=1000)


class ReportOutput(Contract):
    findings: list[FindingOutput] = Field(min_length=1, max_length=12)
    unresolved: list[Text] = Field(default_factory=list, max_length=12)


class ValidatedOperand(Contract):
    name: str
    passage_id: str
    original_text: str
    value: str
    scale: Scale
    unit: str
    period: str
    source_period: str | None
    location: dict


class ValidatedClaim(Contract):
    text: str
    passage_ids: list[str]
    supported: bool
    issues: list[str] = Field(default_factory=list)


class ValidatedFinding(Contract):
    metric: str
    period: str
    status: Literal["accepted", "rejected", "insufficient_evidence"]
    value: str | None = None
    unit: str | None = None
    operands: list[ValidatedOperand] = Field(default_factory=list)
    calculation: list[dict] = Field(default_factory=list)
    claims: list[ValidatedClaim] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
