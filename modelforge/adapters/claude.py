"""ModelForge adapter for Claude: the FinSight provider (same contract, ledger and accounting).

Paid runs always go through the shared Postgres spend ledger (``FINSIGHT_DATABASE_URL``) with
the project limit from settings; there is no unmetered mode for evaluations.
"""

from __future__ import annotations

from finsight.reasoning.base import Generation
from finsight.reasoning.claude import ClaudeProvider
from finsight.settings import get_settings
from finsight.spend import SpendLedger
from finsight.storage.db import connect
from modelforge.items import EvalItem


class ClaudeAdapter(ClaudeProvider):
    def __init__(self, model: str, effort: str = "medium", max_tokens: int = 16_000, label: str | None = None,
                 ledger: SpendLedger | None = None, client=None) -> None:
        settings = get_settings()
        if ledger is None:
            ledger = SpendLedger(connect(settings.database_url, application_name="modelforge-claude"),
                                 settings.spend_limit_usd)
        super().__init__(model, effort=effort, max_tokens=max_tokens, ledger=ledger, client=client)
        self.config_id = label or f"claude-{model}-{effort}"

    def answer(self, item: EvalItem, system: str, user: str, schema: dict) -> Generation:
        return self.generate(system, user, schema, stage=f"finqa:{item.id}", attempt=1, scope="modelforge")
