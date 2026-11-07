"""One provider interface for every model FinSight and ModelForge call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from finsight.contracts import ModelAttempt


@dataclass
class Generation:
    """Raw model text plus per-attempt accounting. ``text`` is None when no usable output arrived."""

    text: str | None
    attempts: list[ModelAttempt] = field(default_factory=list)


class ModelProvider(Protocol):
    provider: str
    model: str

    def generate(self, system: str, user: str, schema: dict, *, job_id=None, stage: str = "answer",
                 attempt: int = 1, scope: str = "finsight") -> Generation: ...
