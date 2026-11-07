"""ModelForge adapter over recorded responses (offline evaluation of saved model outputs)."""

from __future__ import annotations

from pathlib import Path

from finsight.reasoning.base import Generation
from finsight.reasoning.recorded import RecordedProvider
from modelforge.items import EvalItem


class RecordedAdapter(RecordedProvider):
    def __init__(self, recordings: str | Path, model: str = "recorded", label: str | None = None) -> None:
        super().__init__(Path(recordings), model)
        self.config_id = label or f"recorded-{model}"

    def answer(self, item: EvalItem, system: str, user: str, schema: dict) -> Generation:
        return self.generate(system, user, schema, scope="modelforge")
