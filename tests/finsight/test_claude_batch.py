"""ModelForge Claude batch runs against the shared ledger, offline (real SDK, mocked HTTP)."""

import json
import uuid
from decimal import Decimal

import anthropic
import httpx2
import pytest

from finsight.spend import BudgetExceeded, SpendLedger
from modelforge import claude_batch

pytestmark = pytest.mark.postgres
USAGE = {"input_tokens": 2500, "output_tokens": 400, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def _batch(status="in_progress", n=3, batch_id="msgbatch_01test"):
    ended = status == "ended"
    return {"id": batch_id, "type": "message_batch", "processing_status": status,
            "request_counts": {"processing": 0 if ended else n, "succeeded": n - 1 if ended else 0,
                               "errored": 1 if ended else 0, "canceled": 0, "expired": 0},
            "created_at": "2099-01-01T00:00:00Z", "expires_at": "2099-01-02T00:00:00Z", "ended_at": None,
            "cancel_initiated_at": None, "archived_at": None,
            "results_url": f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results" if ended else None}


def _result(custom_id, kind, stop_reason="end_turn"):
    if kind == "errored":
        return {"custom_id": custom_id, "result": {"type": "errored", "error": {"type": "error",
                "error": {"type": "api_error", "message": "boom"}}}}
    return {"custom_id": custom_id, "result": {"type": "succeeded", "message": {
        "id": f"msg_{custom_id}", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "{}"}], "stop_reason": stop_reason, "stop_sequence": None, "usage": USAGE}}}


class Api:
    """Routes for the batch endpoints; `create` can be told to time out."""

    def __init__(self, create="ok"):
        self.create_mode, self.calls, self.created = create, [], 0

    def __call__(self, request):
        path = request.url.path
        self.calls.append((request.method, path))
        if path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 2500})
        if request.method == "POST" and path.endswith("/messages/batches"):
            self.created += 1
            if self.create_mode == "timeout":
                raise httpx2.ReadTimeout("timed out", request=request)
            return httpx2.Response(200, json=_batch())
        if path.endswith("/results"):
            lines = [_result("i0000", "succeeded"), _result("i0001", "succeeded", "max_tokens"), _result("i0002", "errored")]
            return httpx2.Response(200, content="\n".join(json.dumps(x) for x in lines).encode(),
                                   headers={"content-type": "application/binary"})
        if request.method == "GET" and "/messages/batches/" in path:
            return httpx2.Response(200, json=_batch("ended"))
        if request.method == "GET" and path.endswith("/messages/batches"):
            return httpx2.Response(200, json={"data": [_batch("in_progress")], "has_more": False,
                                              "first_id": "msgbatch_01test", "last_id": "msgbatch_01test"})
        return httpx2.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": path}})


def _client(api):
    return anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api)))


@pytest.fixture
def run(tmp_path, monkeypatch, pg):
    monkeypatch.setattr(claude_batch, "ROOT", tmp_path / "batches")
    name = f"t-{uuid.uuid4().hex[:8]}"
    claude_batch.plan(name, "finqa-grouped-v2", "val_dev_eval", "claude-haiku-5-5", "medium", limit=3)
    return name, SpendLedger(pg, Decimal("2.00")), tmp_path


def test_plan_uses_valid_custom_ids_and_count_bounds_the_worst_case(run):
    name, ledger, _ = run
    summary = claude_batch.count(name, _client(Api()), run_cap_usd=Decimal("0.25"))
    assert summary["requests"] == 3 and summary["input_tokens_total"] == 7500
    # 3 x (2500 input x $0.05/M + max_tokens x $0.25/M) must stay under the cap; auto caps at 8000 tokens
    assert summary["max_tokens"] == 8000 and Decimal(summary["worst_case_usd"]) <= Decimal("0.25")
    with pytest.raises(ValueError, match="leaves only"):
        claude_batch.count(name, _client(Api()), run_cap_usd=Decimal("0.0005"))


def test_submit_refuses_past_the_run_cap_without_calling_the_api(run):
    name, ledger, _ = run
    claude_batch.count(name, _client(Api()), run_cap_usd=Decimal("0.25"), max_tokens=2000)
    plan = json.loads((claude_batch._dir(name) / "plan.json").read_text())
    plan["run_cap_usd"] = "0.0001"
    (claude_batch._dir(name) / "plan.json").write_text(json.dumps(plan))
    api = Api()
    with pytest.raises(BudgetExceeded):
        claude_batch.submit(name, _client(api), ledger)
    assert api.created == 0 and ledger.committed() == 0


def test_create_timeout_keeps_reservations_unknown_and_never_resubmits(run):
    name, ledger, _ = run
    claude_batch.count(name, _client(Api()), run_cap_usd=Decimal("0.25"), max_tokens=2000)
    api = Api(create="timeout")
    with pytest.raises(anthropic.APITimeoutError):
        claude_batch.submit(name, _client(api), ledger)
    assert {a["status"] for a in ledger.open_attempts()} == {"unknown"} and ledger.committed() > 0
    with pytest.raises(ValueError, match="never resubmitted"):
        claude_batch.submit(name, _client(api), ledger)
    assert api.created == 1
    state = claude_batch.reconcile(name, _client(Api()), ledger)         # finds the one matching batch
    assert state["status"] == "submitted" and state["batch_id"] == "msgbatch_01test"
    assert {a["status"] for a in ledger.open_attempts()} == {"sent"}


def test_collect_completes_with_actual_cost_fails_errored_unbilled_and_scores(run):
    name, ledger, tmp_path = run
    claude_batch.count(name, _client(Api()), run_cap_usd=Decimal("0.25"), max_tokens=2000)
    claude_batch.submit(name, _client(Api()), ledger)
    out = claude_batch.collect(name, _client(Api()), ledger, runs_dir=tmp_path / "runs")
    rows = {r["stop_reason"] or r["error"]: r for r in ledger.conn.execute(
        "SELECT status, cost_usd, stop_reason, error FROM model_attempts ORDER BY stage").fetchall()}
    per_success = Decimal(2500) * Decimal("0.05") / 10**6 + Decimal(400) * Decimal("0.25") / 10**6
    assert rows["end_turn"]["status"] == "completed" and rows["end_turn"]["cost_usd"] == per_success
    assert rows["max_tokens"]["status"] == "completed"                                   # billed, scored as truncated
    errored = next(r for r in rows.values() if r["status"] == "failed")
    assert errored["cost_usd"] == 0 and "api_error" in errored["error"]
    assert Decimal(out["actual_cost_usd"]) == 2 * per_success and ledger.open_attempts() == []
    predictions = [json.loads(l) for l in (tmp_path / "runs" / name / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 3 and (tmp_path / "runs" / name / "scorecard.json").exists()
    assert sorted(p["reason_code"] for p in predictions).count("OPERATIONAL") == 2           # truncated + errored
