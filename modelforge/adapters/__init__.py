"""Model adapters share one interface so the runner treats every model identically."""

from __future__ import annotations

from typing import Protocol

from finsight.reasoning.base import Generation
from modelforge.items import EvalItem

__all__ = ["Adapter", "Generation", "make_adapter"]


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
    if name == "claude":
        from modelforge.adapters.claude import ClaudeAdapter
        return ClaudeAdapter(**options)
    if name == "recorded":
        from modelforge.adapters.recorded import RecordedAdapter
        return RecordedAdapter(**options)
    raise ValueError(f"unknown adapter {name!r}; choose gold, abstain, qwen, claude or recorded")
