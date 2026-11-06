"""Evaluation items: a question with frozen evidence and (optionally) a gold answer."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from finsight.contracts import CalculationStep, Passage
from finsight.contracts.models import Contract


class EvalItem(Contract):
    id: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    dataset: str
    group: str = Field(description="Filing-level grouping key; splits never separate a group")
    company: str | None = None
    year: str | None = None
    question: str
    passages: list[Passage]
    rows: dict[str, list[str]] = Field(default_factory=dict, description="Numeric table rows keyed by passage id")
    answerable: bool = True
    gold_answer: str | None = None
    gold_steps: list[CalculationStep] | None = None
    gold_passage_ids: list[str] = Field(default_factory=list)
    meta: dict[str, str] = Field(default_factory=dict)
