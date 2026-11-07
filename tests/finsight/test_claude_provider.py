import json
import threading
from decimal import Decimal

import anthropic
import httpx2
import pytest

from finsight.pricing import TokenUsage, cost, reservation
from finsight.reasoning.claude import ClaudeProvider
from finsight.spend import BudgetExceeded, SpendLedger

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
          "additionalProperties": False}
USAGE = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 5000,
         "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000}}


def _message(stop_reason="end_turn", text='{"answer": "42"}', **extra):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
            "usage": USAGE, **extra}


def _client(handler, seen):
    def wrapped(request):
        seen.append(request)
        return handler(request)
    return anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(wrapped)))


def _provider(handler, seen, ledger=None, model="claude-sonnet-5-5"):
    return ClaudeProvider(model, effort="medium", max_tokens=2000, ledger=ledger, client=_client(handler, seen),
                          allow_unmetered=ledger is None)


def test_payload_uses_structured_output_and_sends_no_sampling_fallbacks_or_forced_tools():
    seen = []
    generation = _provider(lambda r: httpx2.Response(200, json=_message()), seen).generate("system", "question", SCHEMA)
    body = json.loads(seen[0].content)
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "medium"}
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 2000
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice", "fallbacks", "thinking"):
        assert forbidden not in body
    assert "anthropic-beta" not in seen[0].headers
    assert generation.text == '{"answer": "42"}' and generation.attempts[0].status == "ok"


def test_cost_from_raw_usage_includes_both_cache_write_ttls():
    seen = []
    attempt = _provider(lambda r: httpx2.Response(200, json=_message()), seen).generate("s", "u", SCHEMA).attempts[0]
    assert (attempt.cache_write_5m_tokens, attempt.cache_write_1h_tokens, attempt.cache_read_tokens) == (1000, 2000, 5000)
    # 1000*2 + 1000*2.50 + 2000*4 + 5000*0.10 + 200*10 = 15,000 per million
    assert attempt.estimated_cost_usd == Decimal("0.015000")


def test_haiku_prompt_tier_and_batch_discount():
    small = TokenUsage(input_uncached=50_000, output=1000)
    large = TokenUsage(input_uncached=60_000, cache_read=50_000, output=1000)        # prompt 110k > 100k
    assert cost("claude-haiku-5-5", small) == Decimal("0.005500")
    assert cost("claude-haiku-5-5", large) == Decimal("0.035000")                    # 60k*0.50+50k*0.05+1k*2.50
    assert cost("claude-sonnet-5-5", small, batch=True) == cost("claude-sonnet-5-5", small) / 2
    assert reservation("claude-sonnet-5-5", 3000, 16000) == Decimal("0.166000")


@pytest.mark.parametrize("stop_reason,status", [("refusal", "refused"), ("max_tokens", "truncated")])
def test_refusal_and_truncation_block_output(stop_reason, status):
    seen = []
    extra = {"stop_details": {"type": "refusal", "category": "cyber", "explanation": "declined"}} if stop_reason == "refusal" else {}
    generation = _provider(lambda r: httpx2.Response(200, json=_message(stop_reason, **extra)), seen).generate("s", "u", SCHEMA)
    assert generation.text is None and generation.attempts[0].status == status


def test_paid_calls_require_the_shared_ledger():
    with pytest.raises(ValueError, match="ledger"):
        ClaudeProvider("claude-sonnet-5-5")


@pytest.mark.postgres
def test_unknown_outcome_keeps_reservation_and_completed_responses_are_reused(pg):
    ledger = SpendLedger(pg, Decimal("2.00"))
    seen = []

    def lost(request):
        raise httpx2.ConnectError("connection reset after send")
    generation = _provider(lost, seen, ledger).generate("s", "u", SCHEMA, stage="analysis", attempt=1)
    assert generation.attempts[0].status == "uncertain"
    (open_attempt,) = ledger.open_attempts()
    assert open_attempt["status"] == "unknown" and Decimal(open_attempt["reserved_usd"]) > 0
    assert ledger.committed() == Decimal(open_attempt["reserved_usd"])           # still counted against the limit

    ok = _provider(lambda r: httpx2.Response(200, json=_message()), seen, ledger)
    first = ok.generate("s", "u", SCHEMA, stage="analysis", attempt=2)               # a retry is a new attempt
    calls = len(seen)
    again = ok.generate("s", "u", SCHEMA, stage="analysis", attempt=2)               # re-run after a crash
    assert len(seen) == calls and again.text == first.text                           # reused, no second call
    overloaded = _provider(lambda r: httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error",
                                                                                            "message": "busy"}}), seen, ledger)
    assert overloaded.generate("s", "u", SCHEMA, stage="analysis", attempt=3).attempts[0].status == "retryable_error"
    statuses = sorted(r["status"] for r in pg.execute("SELECT status FROM model_attempts").fetchall())
    assert statuses == ["completed", "failed", "unknown"]


@pytest.mark.postgres
def test_concurrent_reservations_cannot_exceed_the_limit(pg):
    import os
    from finsight.storage.db import connect
    SpendLedger(pg, Decimal("0.20"))
    results = []

    def reserve():
        conn = connect(os.environ["FINSIGHT_TEST_DATABASE_URL"], application_name="finsight-ledger-test")
        try:
            SpendLedger(conn, Decimal("0.20")).reserve(scope="modelforge", job_id=None, stage="answer", attempt=1,
                                                       request_sha256="x", provider="anthropic", model="claude-sonnet-5-5",
                                                       amount_usd=Decimal("0.15"))
            results.append("ok")
        except BudgetExceeded:
            results.append("refused")
        finally:
            conn.close()
    threads = [threading.Thread(target=reserve) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == ["ok", "refused", "refused", "refused"]
