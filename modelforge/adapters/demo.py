"""Deterministic offline adapters used to validate the harness itself.

* ``gold``: answers with the gold program expressed in the answer contract (an upper
  bound that should score 100% on aligned answerable items).
* ``abstain``: always abstains (a lower bound that exercises abstention scoring).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from finsight.contracts import ModelAttempt
from modelforge.adapters import Generation
from modelforge.datasets.finqa import gold_answer_output
from modelforge.items import EvalItem


def _attempt(model: str) -> ModelAttempt:
    return ModelAttempt(requested_model=model, served_model=model, stage="answer", attempt=1, status="ok",
                        input_tokens_uncached=0, output_tokens=0, latency_ms=0, estimated_cost_usd=0,
                        started_at=datetime.now(timezone.utc))


class GoldAdapter:
    config_id = "demo-gold"
    settings = {"provider": "demo", "model": "gold-program-oracle"}

    def answer(self, item: EvalItem, system: str, user: str, schema: dict) -> Generation:
        if not item.answerable:
            text = json.dumps({"abstain": True, "abstain_reason": "gold evidence removed", "operands": [],
                               "program": [], "answer": None, "answer_unit": None, "claims": []})
        else:
            output = gold_answer_output(item)
            text = output.model_dump_json() if output else json.dumps({"unaligned": True})
        return Generation(text=text, attempts=[_attempt("gold-program-oracle")])


class AbstainAdapter:
    config_id = "demo-abstain"
    settings = {"provider": "demo", "model": "always-abstain"}

    def answer(self, item: EvalItem, system: str, user: str, schema: dict) -> Generation:
        text = json.dumps({"abstain": True, "abstain_reason": "always abstains", "operands": [], "program": [],
                           "answer": None, "answer_unit": None, "claims": []})
        return Generation(text=text, attempts=[_attempt("always-abstain")])
