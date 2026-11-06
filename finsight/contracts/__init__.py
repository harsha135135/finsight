"""Shared, dependency-light contracts used by FinSight and ModelForge.

Every model rejects unknown fields. Decimal values serialize as JSON strings so
figures survive round trips without binary floating-point drift.
"""

from .models import (
    AnswerOutput,
    Budgets,
    Calculation,
    CalculationStep,
    Claim,
    Confidence,
    Document,
    ModelAttempt,
    ModelSettings,
    Op,
    Operand,
    OperandOutput,
    Passage,
    PassageLocation,
    Prediction,
    ReasonCode,
    ResearchJob,
    ResearchResult,
    Scale,
    Task,
    ValueKind,
)

__all__ = [
    "AnswerOutput",
    "Budgets",
    "Calculation",
    "CalculationStep",
    "Claim",
    "Confidence",
    "Document",
    "ModelAttempt",
    "ModelSettings",
    "Op",
    "Operand",
    "OperandOutput",
    "Passage",
    "PassageLocation",
    "Prediction",
    "ReasonCode",
    "ResearchJob",
    "ResearchResult",
    "Scale",
    "Task",
    "ValueKind",
]
