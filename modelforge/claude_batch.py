"""Claude reference runs through the Message Batches API (half price), metered by the shared ledger.

Steps, each resumable from the files in ``artifacts/claude-batches/<name>/``:

plan     requests for every item, with the same body ``ClaudeProvider.request`` sends (FinQA protocol
         ``answer-v4``, structured output, explicit effort, no sampling parameters). ``custom_id`` must match
         ``^[a-zA-Z0-9_-]{1,64}$`` and FinQA ids contain ``/`` and ``::``, so ids are ``i0000``... with a mapping.
count    exact input tokens per request from the free token-counting endpoint, and the worst case:
         every request uses its exact input plus the full ``max_tokens`` (thinking is billed as output),
         at batch rates. ``--max-tokens auto`` picks the largest allowance that keeps the worst case under
         the run cap.
submit   reserves every request at its worst case in one ledger transaction (refused past the run cap or
         the global limit), writes ``creating`` to the state file, creates the batch and saves its id at once.
         A create call that times out or loses its connection may or may not have created a batch: all
         attempts become ``unknown`` and keep their reservations; nothing is resubmitted (``reconcile`` looks
         for a matching batch instead).
collect  when the batch has ended: succeeded requests complete with their actual cost; errored, expired and
         canceled requests fail unbilled. Results are scored by the normal runner, so the scorecard has the
         same format as every other run.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import anthropic

from finsight.contracts import ModelAttempt
from finsight.pricing import PRICES_AS_OF, TokenUsage, cost, reservation
from finsight.reasoning.base import Generation
from finsight.spend import SpendLedger
from modelforge import prompts, runner, splits

ROOT = Path("artifacts/claude-batches")
CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
        raise ValueError(f"bad run name {name!r}")
    return ROOT / name


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    tmp.replace(path)


def _counts(request_counts) -> dict:
    return request_counts.model_dump() if hasattr(request_counts, "model_dump") else dict(vars(request_counts))


def _body(model: str, max_tokens: int, effort: str, item) -> dict:
    return {"model": model, "max_tokens": max_tokens, "system": prompts.SYSTEM,
            "messages": [{"role": "user", "content": prompts.user_message(item)}],
            "output_config": {"format": {"type": "json_schema", "schema": prompts.answer_schema()}, "effort": effort}}


def plan(name: str, manifest_name: str, selection: str, model: str, effort: str, limit: int | None = None) -> dict:
    out = _dir(name)
    if (out / "state.json").exists():
        raise ValueError(f"{out} already has a submitted batch; use a new name")
    manifest = splits.load(manifest_name)
    ids = (manifest.get("samples", {}).get(selection) or manifest.get("slices", {}).get(selection)
           or manifest["splits"].get(selection))
    items = splits.resolve_items(manifest, ids[:limit] if limit else ids)
    requests = [{"custom_id": f"i{n:04d}", "item_id": item.id} for n, item in enumerate(items)]
    assert all(CUSTOM_ID.match(r["custom_id"]) for r in requests)
    data = {"name": name, "manifest": manifest_name, "manifest_sha256": manifest["sha256"], "selection": selection,
            "model": model, "effort": effort, "prompt_version": prompts.PROMPT_VERSION,
            "prompt_sha256": prompts.fingerprint(), "prices_as_of": PRICES_AS_OF, "limit": limit, "requests": requests,
            "created_at": datetime.now(timezone.utc)}
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "plan.json", data)
    return data


def _items(data: dict) -> dict:
    manifest = splits.load(data["manifest"])
    if manifest["sha256"] != data["manifest_sha256"]:
        raise ValueError("the manifest changed since the plan was made")
    return {i.id: i for i in splits.resolve_items(manifest, [r["item_id"] for r in data["requests"]])}


def count(name: str, client, *, run_cap_usd: Decimal, max_tokens: int | str = "auto") -> dict:
    """Exact input tokens (free endpoint) and the worst-case cost at batch rates."""
    out = _dir(name)
    data = _read(out / "plan.json")
    items = _items(data)
    for r in data["requests"]:
        if "input_tokens" not in r:
            body = _body(data["model"], 1, data["effort"], items[r["item_id"]])
            r["input_tokens"] = client.messages.count_tokens(
                model=body["model"], system=body["system"], messages=body["messages"],
                output_config=body["output_config"]).input_tokens
            _write(out / "plan.json", data)  # keep counts made so far
    n = len(data["requests"])
    input_cost = sum((cost(data["model"], TokenUsage(input_uncached=r["input_tokens"]), batch=True)
                      for r in data["requests"]), Decimal(0))
    largest = max(r["input_tokens"] for r in data["requests"])
    output_rate = (cost(data["model"], TokenUsage(input_uncached=largest, output=1_000_000), batch=True)
                   - cost(data["model"], TokenUsage(input_uncached=largest), batch=True)) / Decimal(1_000_000)
    if max_tokens == "auto":
        allowance = int((run_cap_usd - input_cost) / (n * output_rate)) // 100 * 100
        max_tokens = min(allowance, 8000)
    max_tokens = int(max_tokens)
    if max_tokens < 1024:
        raise ValueError(f"the run cap ${run_cap_usd} leaves only {max_tokens} output tokens per request")
    worst = sum((reservation(data["model"], r["input_tokens"], max_tokens, batch=True) for r in data["requests"]),
                Decimal(0))
    data.update(max_tokens=max_tokens, run_cap_usd=str(run_cap_usd), input_tokens_total=sum(r["input_tokens"] for r in data["requests"]),
                worst_case_usd=str(worst.quantize(Decimal("0.0001"))), input_cost_usd=str(input_cost.quantize(Decimal("0.0001"))))
    if worst > run_cap_usd:
        raise ValueError(f"worst case ${worst:.4f} exceeds the run cap ${run_cap_usd} at max_tokens={max_tokens}")
    _write(out / "plan.json", data)
    return {k: data[k] for k in ("name", "model", "effort", "max_tokens", "input_tokens_total", "input_cost_usd",
                                 "worst_case_usd", "run_cap_usd")} | {"requests": n}


def submit(name: str, client, ledger: SpendLedger) -> dict:
    out = _dir(name)
    data = _read(out / "plan.json")
    if "worst_case_usd" not in data:
        raise ValueError("run `count` first: the worst case must be known before anything is reserved")
    if (out / "state.json").exists():
        state = _read(out / "state.json")
        raise ValueError(f"batch already {state['status']} (id {state.get('batch_id')}); never resubmitted automatically")
    items = _items(data)
    requests, entries = [], []
    for r in data["requests"]:
        body = _body(data["model"], data["max_tokens"], data["effort"], items[r["item_id"]])
        r["request_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        requests.append({"custom_id": r["custom_id"], "params": body})
        entries.append({"scope": "modelforge", "job_id": None, "stage": f"finqa:{r['item_id']}", "attempt": 1,
                        "request_sha256": r["request_sha256"], "provider": "anthropic", "model": data["model"],
                        "amount_usd": reservation(data["model"], r["input_tokens"], data["max_tokens"], batch=True)})
    attempt_ids = ledger.reserve_many(entries, run_cap_usd=Decimal(data["run_cap_usd"]))
    for r, attempt_id in zip(data["requests"], attempt_ids):
        r["attempt_id"] = str(attempt_id)
    _write(out / "plan.json", data)
    state = {"status": "creating", "requested_at": datetime.now(timezone.utc), "batch_id": None}
    _write(out / "state.json", state)
    try:
        batch = client.messages.batches.create(requests=requests)
    except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:  # created or not: unknown
        ledger.unknown_many(attempt_ids, f"batch create outcome unknown: {type(exc).__name__}")
        state.update(status="create_unknown", error=str(exc)[:300])
        _write(out / "state.json", state)
        raise
    except anthropic.APIStatusError as exc:                                   # rejected: no batch exists
        ledger.failed_many(attempt_ids, f"batch rejected: {exc.status_code}")
        state.update(status="rejected", error=f"{exc.status_code}: {exc.message}"[:300])
        _write(out / "state.json", state)
        raise
    state.update(status="submitted", batch_id=batch.id, submitted_at=datetime.now(timezone.utc))
    _write(out / "state.json", state)
    ledger.mark_sent_many(attempt_ids, note=f"batch {batch.id}")
    return state


def reconcile(name: str, client, ledger: SpendLedger) -> dict:
    """After an unknown create: adopt the one batch created around then with the same request count."""
    out = _dir(name)
    data, state = _read(out / "plan.json"), _read(out / "state.json")
    if state["status"] != "create_unknown":
        return state
    since = datetime.fromisoformat(str(state["requested_at"])) - timedelta(minutes=1)
    n = len(data["requests"])
    matches = [b for b in client.messages.batches.list(limit=50)
               if b.created_at >= since and sum(_counts(b.request_counts).values()) == n]
    if len(matches) == 1:
        state.update(status="submitted", batch_id=matches[0].id, reconciled_at=datetime.now(timezone.utc))
        _write(out / "state.json", state)
        ledger.mark_sent_many([r["attempt_id"] for r in data["requests"]], note=f"batch {matches[0].id} (reconciled)")
    else:
        state["reconcile_note"] = f"{len(matches)} candidate batches; attempts stay unknown"
        _write(out / "state.json", state)
    return state


class _Collected:
    """Adapter that hands the runner the already-collected batch results."""

    def __init__(self, data: dict, generations: dict[str, Generation]) -> None:
        self.generations = generations
        self.config_id = f"claude-{data['model']}-{data['effort']}-batch"
        self.settings = {"provider": "anthropic", "model": data["model"], "effort": data["effort"],
                         "max_tokens": str(data["max_tokens"]), "api": "message batches (50% price)",
                         "thinking": "adaptive (model default)", "sampling": "none sent"}

    def answer(self, item, system, user, schema) -> Generation:
        return self.generations[item.id]


def collect(name: str, client, ledger: SpendLedger, runs_dir: Path = Path("artifacts/runs")) -> dict:
    out = _dir(name)
    data, state = _read(out / "plan.json"), _read(out / "state.json")
    if state["status"] not in ("submitted", "ended"):
        raise ValueError(f"batch state is {state['status']}")
    batch = client.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        return {"status": batch.processing_status, "counts": _counts(batch.request_counts)}
    by_custom = {r["custom_id"]: r for r in data["requests"]}
    generations, raw = {}, []
    for result in client.messages.batches.results(state["batch_id"]):
        r = by_custom[result.custom_id]
        kind = result.result.type
        started = datetime.fromisoformat(str(state["submitted_at"]))
        if kind == "succeeded":
            message = result.result.message
            usage = TokenUsage.from_anthropic(message.usage)
            spent = cost(data["model"], usage, batch=True)
            text = next((b.text for b in message.content if b.type == "text"), None)
            status = {"refusal": "refused", "max_tokens": "truncated"}.get(message.stop_reason, "ok")
            ledger.complete(r["attempt_id"], cost_usd=spent, usage=usage.__dict__,
                            response={"text": text, "stop_reason": message.stop_reason, "id": message.id,
                                      "batch_id": state["batch_id"]},
                            latency_ms=None, model_version=message.model, stop_reason=message.stop_reason)
            attempt = ModelAttempt(requested_model=data["model"], served_model=message.model, stage="answer", attempt=1,
                                   status=status, input_tokens_uncached=usage.input_uncached,
                                   cache_read_tokens=usage.cache_read, cache_write_5m_tokens=usage.cache_write_5m,
                                   cache_write_1h_tokens=usage.cache_write_1h, output_tokens=usage.output,
                                   estimated_cost_usd=spent, started_at=started,
                                   error="hit max_tokens" if status == "truncated" else
                                   "refusal" if status == "refused" else None)
            generations[r["item_id"]] = Generation(text=text if status == "ok" else None, attempts=[attempt], raw=text)
            raw.append({"custom_id": result.custom_id, "item_id": r["item_id"], "type": kind, "text": text,
                        "stop_reason": message.stop_reason, "usage": usage.__dict__, "cost_usd": str(spent)})
        else:  # errored, expired, canceled: not billed
            outer = getattr(result.result, "error", None)                 # ErrorResponse {type, error: {type}}
            error = getattr(getattr(outer, "error", None), "type", None) or getattr(outer, "type", None) or kind
            ledger.failed(r["attempt_id"], f"batch result {kind}: {error}")
            generations[r["item_id"]] = Generation(text=None, attempts=[ModelAttempt(
                requested_model=data["model"], stage="answer", attempt=1,
                status="error" if kind == "errored" else "retryable_error", started_at=started,
                error=f"batch result {kind}: {error}")])
            raw.append({"custom_id": result.custom_id, "item_id": r["item_id"], "type": kind, "error": str(error)})
    missing = set(by_custom) - {x["custom_id"] for x in raw}
    for custom_id in missing:  # never returned: treat as not billed, keep it visible
        r = by_custom[custom_id]
        ledger.failed(r["attempt_id"], "batch result missing")
        generations[r["item_id"]] = Generation(text=None, attempts=[ModelAttempt(
            requested_model=data["model"], stage="answer", attempt=1, status="error",
            started_at=datetime.now(timezone.utc), error="batch result missing")])
    (out / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in raw))
    state.update(status="ended", collected_at=datetime.now(timezone.utc),
                 actual_cost_usd=str(sum((Decimal(x.get("cost_usd", "0")) for x in raw), Decimal(0))))
    _write(out / "state.json", state)
    run_dir = runs_dir / name
    runner.run(data["manifest"], data["selection"], _Collected(data, generations), run_dir, limit=data.get("limit"))
    return {"status": "ended", "run_dir": str(run_dir), "actual_cost_usd": state["actual_cost_usd"],
            "counts": _counts(batch.request_counts)}
