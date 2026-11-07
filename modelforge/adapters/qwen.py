"""ModelForge adapter for Qwen on vLLM: the FinSight provider plus the FinQA adapter interface.

The same chat template settings, decoding parameters and JSON schema are used for the
unchanged and the fine-tuned model; only ``model`` differs (the base checkpoint or the LoRA
adapter name registered with ``--lora-modules``).
"""

from __future__ import annotations

from finsight.reasoning.base import Generation
from finsight.reasoning.qwen import QwenProvider
from modelforge.items import EvalItem


class QwenAdapter(QwenProvider):
    def answer(self, item: EvalItem, system: str, user: str, schema: dict) -> Generation:
        return self.generate(system, user, schema, stage="answer", scope="modelforge")
