"""Model adapters share one interface so the runner treats every model identically."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from finsight.contracts import ModelAttempt
from modelforge.items import EvalItem


@dataclass
class Generation:
    """Raw model text plus provider accounting. ``text`` is None when no output arrived."""

    text: str | None
    attempts: list[ModelAttempt] = field(default_factory=list)


class Adapter(Protocol):
    config_id: str
    settings: dict

    def answer(self, item: EvalItem, system: str, user: str, schema: dict) -> Generation: ...


def make_adapter(name: str, **options) -> Adapter:
    if name == "gold":
        from modelforge.adapters.demo import GoldAdapter
        return GoldAdapter()
    if name == "abstain":
        from modelforge.adapters.demo import AbstainAdapter
        return AbstainAdapter()
    if name == "qwen":
        from modelforge.adapters.qwen import QwenAdapter
        return QwenAdapter(**options)
    raise ValueError(f"unknown adapter {name!r}; choose gold, abstain or qwen")
