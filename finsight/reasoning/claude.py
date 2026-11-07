"""Claude through the official Anthropic SDK, with structured JSON output and a shared spend ledger.

Request: ``output_config.format`` (JSON schema) plus an explicit ``effort``; no temperature or
other sampling parameters (current models reject them), no server-side fallbacks (a fallback
model would make usage and model attribution ambiguous; off by project decision), no forced
tool choice, and SDK retries off so every retry is a separately reserved, recorded attempt.

Outcomes: ``refusal`` -> refused; ``max_tokens`` -> truncated (both block publication);
connection loss or timeout after sending -> uncertain (ledger: unknown, reservation kept, never
resent automatically); 429/529/5xx -> retryable_error (not charged); other 4xx -> error.
Cost comes from the raw ``usage`` object (uncached input, 5-minute and 1-hour cache writes,
cache reads, output) at dated prices.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import anthropic

from finsight.contracts import ModelAttempt
from finsight.pricing import TokenUsage, cost, reservation
from finsight.reasoning.base import Generation
from finsight.spend import SpendLedger


class ClaudeProvider:
    provider = "anthropic"

    def __init__(self, model: str, *, effort: str = "medium", max_tokens: int = 16_000,
                 ledger: SpendLedger | None = None, client: anthropic.Anthropic | None = None,
                 allow_unmetered: bool = False) -> None:
        if ledger is None and not allow_unmetered:
            raise ValueError("paid Claude calls need the shared Postgres spend ledger")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.ledger = ledger
        if client is None:
            from finsight.settings import anthropic_client
            client = anthropic_client()
        self.client = client
        self.settings = {"provider": "anthropic", "model": model, "effort": effort, "max_tokens": str(max_tokens),
                         "structured_output": "output_config.format json_schema", "fallbacks": "off",
                         "sampling": "server defaults (none sent)"}

    def request(self, system: str, user: str, schema: dict) -> dict:
        return {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}],
                "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": self.effort}}

    def generate(self, system: str, user: str, schema: dict, *, job_id=None, stage: str = "answer",
                 attempt: int = 1, scope: str = "finsight") -> Generation:
        body = self.request(system, user, schema)
        request_sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if self.ledger is not None:
            done = self.ledger.completed_response(job_id, stage, attempt, request_sha)
            if done is not None:  # a node re-run after a crash reuses the recorded response
                return Generation(text=(done["response"] or {}).get("text"), attempts=[])
        started = datetime.now(timezone.utc)
        prompt_estimate = len(json.dumps(body)) // 3 + 50          # conservative: about 3 characters per token
        attempt_id = None
        if self.ledger is not None:
            attempt_id = self.ledger.reserve(scope=scope, job_id=job_id, stage=stage, attempt=attempt,
                                             request_sha256=request_sha, provider=self.provider, model=self.model,
                                             amount_usd=reservation(self.model, prompt_estimate, self.max_tokens))
            self.ledger.mark_sent(attempt_id)
        t0 = time.perf_counter()

        def failure(status: str, error: str, *, unknown: bool) -> Generation:
            if attempt_id is not None:
                (self.ledger.unknown if unknown else self.ledger.failed)(attempt_id, error)
            return Generation(text=None, attempts=[ModelAttempt(
                requested_model=self.model, stage=stage, attempt=attempt, status=status, started_at=started,
                latency_ms=int((time.perf_counter() - t0) * 1000), error=error)])

        try:
            response = self.client.messages.create(**body)
        except anthropic.APITimeoutError as exc:          # subclass of APIConnectionError: check it first
            return failure("uncertain", f"timeout: {exc}", unknown=True)
        except anthropic.APIConnectionError as exc:
            return failure("uncertain", f"connection: {exc}", unknown=True)
        except anthropic.RateLimitError as exc:
            return failure("retryable_error", f"429: {exc.message}", unknown=False)
        except anthropic.APIStatusError as exc:
            status = "retryable_error" if exc.status_code >= 500 else "error"
            return failure(status, f"{exc.status_code}: {exc.message}", unknown=False)
        latency = int((time.perf_counter() - t0) * 1000)
        usage = TokenUsage.from_anthropic(response.usage)
        spent = cost(self.model, usage)
        text = next((b.text for b in response.content if b.type == "text"), None)
        status = {"refusal": "refused", "max_tokens": "truncated"}.get(response.stop_reason, "ok")
        detail = getattr(response, "stop_details", None)
        error = (f"refusal category {getattr(detail, 'category', None)}" if status == "refused" else
                 "hit max_tokens" if status == "truncated" else None)
        if attempt_id is not None:
            self.ledger.complete(attempt_id, cost_usd=spent, usage=usage.__dict__,
                                 response={"text": text, "stop_reason": response.stop_reason, "id": response.id},
                                 latency_ms=latency, model_version=response.model, stop_reason=response.stop_reason)
        record = ModelAttempt(
            requested_model=self.model, served_model=response.model, stage=stage, attempt=attempt, status=status,
            input_tokens_uncached=usage.input_uncached, cache_read_tokens=usage.cache_read,
            cache_write_5m_tokens=usage.cache_write_5m, cache_write_1h_tokens=usage.cache_write_1h, output_tokens=usage.output, latency_ms=latency,
            estimated_cost_usd=spent, started_at=started, error=error)
        return Generation(text=text if status == "ok" else None, attempts=[record])
