"""Pydantic contracts for evidence, operands, calculations, results and usage."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=20_000)]


class Contract(BaseModel):
    """Base for all contracts: unknown fields are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Task(StrEnum):
    REPORT = "report"
    ANSWER = "answer"


class ValueKind(StrEnum):
    REPORTED = "reported"
    CALCULATED = "calculated"
    INTERPRETATION = "interpretation"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Scale(StrEnum):
    UNITS = "units"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"

    @property
    def multiplier(self) -> Decimal:
        return {
            Scale.UNITS: Decimal(1),
            Scale.THOUSANDS: Decimal(1_000),
            Scale.MILLIONS: Decimal(1_000_000),
            Scale.BILLIONS: Decimal(1_000_000_000),
        }[self]


class ReasonCode(StrEnum):
    """Why an item ended the way it did. Failures are never relabelled as abstentions."""

    ANSWERED = "ANSWERED"
    ABSTAIN_EVIDENCE_ABSENT = "ABSTAIN_EVIDENCE_ABSENT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    VALIDATION_CONTRADICTION = "VALIDATION_CONTRADICTION"
    OPERATIONAL = "OPERATIONAL"


class Op(StrEnum):
    """Operations the calculator interpreter permits. Nothing else is executable."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    EXP = "exp"
    GREATER = "greater"
    PERCENT_CHANGE = "percent_change"
    TO_PERCENT = "to_percent"
    TABLE_SUM = "table_sum"
    TABLE_AVERAGE = "table_average"
    TABLE_MAX = "table_max"
    TABLE_MIN = "table_min"


# --------------------------------------------------------------------------- evidence


class Document(Contract):
    id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")]
    source_uri: NonEmpty
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    accession: Annotated[str, StringConstraints(pattern=r"^\d{10}-\d{2}-\d{6}$")] | None = None
    filing_date: date | None = None
    company: str | None = None
    form: str | None = None
    parser_version: NonEmpty


class PassageLocation(Contract):
    page: int | None = Field(default=None, ge=1)
    table: str | None = None
    row: int | None = Field(default=None, ge=0)


class Passage(Contract):
    id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:#-]{0,199}$")]
    document_id: NonEmpty
    section: NonEmpty
    location: PassageLocation = PassageLocation()
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: NonEmpty

    @model_validator(mode="after")
    def _offsets(self) -> "Passage":
        if self.end < self.start:
            raise ValueError("passage end offset precedes start offset")
        return self


# --------------------------------------------------------------------------- answers


class Operand(Contract):
    """A figure taken from evidence, kept with its original text and its normalization."""

    name: Identifier
    original_text: NonEmpty
    value: Decimal
    unit: Annotated[str, StringConstraints(min_length=1, max_length=16)] = "number"
    scale: Scale = Scale.UNITS
    period: str | None = None
    passage_id: NonEmpty
    kind: ValueKind = ValueKind.REPORTED

    @field_validator("value")
    @classmethod
    def _finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("operand value must be finite")
        return value

    @property
    def base_value(self) -> Decimal:
        return self.value * self.scale.multiplier


class CalculationStep(Contract):
    """One interpreter step. Args are operand names, '#n' step references or numeric literals."""

    op: Op
    args: list[Annotated[str, StringConstraints(min_length=1, max_length=200)]] = Field(min_length=1, max_length=8)


class Calculation(Contract):
    steps: list[CalculationStep] = Field(min_length=1, max_length=32)
    values: list[str] = Field(default_factory=list, description="Executed value of each step, as strings")
    result: str | None = None


class Claim(Contract):
    text: NonEmpty
    passage_ids: list[str] = Field(default_factory=list, max_length=20)
    calculation_steps: list[int] = Field(default_factory=list, max_length=32)
    kind: ValueKind
    confidence: Confidence


class OperandOutput(Contract):
    """Model-facing operand: the value is a string so JSON schemas stay strict and portable."""

    name: Identifier
    original_text: NonEmpty
    value: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    unit: Annotated[str, StringConstraints(min_length=1, max_length=16)]
    scale: Scale
    period: str | None
    passage_id: NonEmpty


class AnswerOutput(Contract):
    """What a model returns for the `answer` task (Claude and Qwen use the same schema)."""

    abstain: bool
    abstain_reason: str | None
    operands: list[OperandOutput] = Field(max_length=24)
    program: list[CalculationStep] = Field(max_length=32)
    answer: str | None = Field(description="Final value as a plain number, 'yes'/'no', or null when abstaining")
    answer_unit: str | None
    claims: list[Claim] = Field(max_length=12)


# --------------------------------------------------------------------------- jobs and results


class Budgets(Contract):
    max_model_calls: int = Field(default=12, ge=1, le=200)
    max_cost_usd: Decimal = Field(default=Decimal("0.50"), ge=0, le=Decimal("50"))
    max_retries: int = Field(default=2, ge=0, le=5)
    max_revisions: int = Field(default=1, ge=0, le=3)


class ModelSettings(Contract):
    provider: Annotated[str, StringConstraints(pattern=r"^(anthropic|qwen|demo)$")]
    model: NonEmpty
    revision: str | None = None
    effort: str | None = None
    max_tokens: int = Field(default=16_000, ge=256, le=128_000)
    temperature: float | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class ResearchJob(Contract):
    task: Task
    question: Annotated[str, StringConstraints(min_length=1, max_length=4_000)]
    company: str | None = None
    periods: list[str] = Field(default_factory=list, max_length=12)
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    cutoff_date: date | None = None
    budgets: Budgets = Budgets()
    require_approval: bool = False


class ResearchResult(Contract):
    task: Task
    reason_code: ReasonCode
    answer: str | None = None
    answer_unit: str | None = None
    operands: list[Operand] = Field(default_factory=list)
    calculation: Calculation | None = None
    claims: list[Claim] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)


class ModelAttempt(Contract):
    """One provider request. Usage is the provider's raw accounting, never a derived total."""

    requested_model: NonEmpty
    served_model: str | None = None
    stage: NonEmpty
    attempt: int = Field(ge=1)
    status: Annotated[str, StringConstraints(pattern=r"^(ok|retryable_error|error|uncertain|truncated|refused)$")]
    input_tokens_uncached: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_5m_tokens: int | None = Field(default=None, ge=0)
    cache_write_1h_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    ttft_ms: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = None
    started_at: datetime
    error: str | None = None


class Prediction(Contract):
    """One evaluated item. Kept in the denominator whatever its outcome."""

    item_id: NonEmpty
    config_id: NonEmpty
    reason_code: ReasonCode
    answer: str | None = None
    executed: str | None = None
    correct: bool | None = None
    issues: list[str] = Field(default_factory=list)
    attempts: list[ModelAttempt] = Field(default_factory=list)
    raw_output: str | None = None
