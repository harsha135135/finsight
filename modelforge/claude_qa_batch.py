"""Claude reference on the frozen finsight-qa interface (FinanceBench development only) via the Message Batches API.

Inputs are the frozen item files the local models used (hashes checked against evals/financebench/freeze-qa-v1.6.json),
the same system prompt and user messages, and the Claude variant of the same schema. Exact input tokens come from the
free token-counting endpoint; the output allowance is chosen so the worst case (every request uses its full allowance,
at batch prices) stays under the run cap. All requests are reserved in one ledger transaction (run cap and global
limit). A create call with an unknown outcome is never resubmitted. Results are written in the local runs' format so
modelforge.qa_score applies unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import anthropic

from finsight.contracts import Passage
from finsight.pricing import TokenUsage, cost, reservation
from finsight.reasoning.qa_prompt import QA_PROMPT_VERSION, SYSTEM, claude_schema, user_message
from finsight.spend import SpendLedger
from modelforge.claude_batch import _counts, _read, _write

ROOT = Path("artifacts/claude-batches")
FROZEN = {"supplied": Path("artifacts/runs/20261110-qa15-fb-supplied-base/items.jsonl"),
          "full": Path("artifacts/runs/20261110-qa15-fb-full-base/items.jsonl")}
FREEZE = Path("evals/financebench/freeze-qa-v1.6.json")
MAX_TOKENS = 5600       # the output allowance of the 2026-11-10 development batch; fixed so later runs match it
EFFORT = "medium"


def _check(path: Path, expected_sha256: str) -> list[dict]:
    body = Path(path).read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError(f"{path} does not match its expected sha256")
    return [json.loads(line) for line in body.decode().splitlines() if line]


def _items(mode: str, inputs: dict | None = None) -> list[dict]:
    """Items for a mode; ``inputs`` (from the plan) maps mode -> {"path", "sha256"}; default: the v1.6 dev freeze."""
    if inputs:
        return _check(Path(inputs[mode]["path"]), inputs[mode]["sha256"])
    return _check(FROZEN[mode], json.loads(FREEZE.read_text())["inputs_sha256"][mode])


def _body(model: str, max_tokens: int, effort: str, item: dict) -> dict:
    passages = [Passage.model_validate(p) for p in item["passages"]]
    return {"model": model, "max_tokens": max_tokens, "system": SYSTEM,
            "messages": [{"role": "user", "content": user_message(item["question"], passages)}],
            "output_config": {"format": {"type": "json_schema", "schema": claude_schema()}, "effort": effort}}


def plan(name: str, model: str = "claude-haiku-5-5", effort: str = EFFORT, *, inputs: dict | None = None,
         freeze: Path = FREEZE, run_prefix: str = "20261110-qa16-fb", label: str = "finsight-qa-v1.6",
         max_tokens: int = MAX_TOKENS) -> dict:
    """``inputs``: mode -> {"path": items.jsonl, "sha256": expected}; modes are run in the order given."""
    out = ROOT / name
    if (out / "state.json").exists():
        raise ValueError(f"{out} already has a submitted batch")
    modes = list(inputs) if inputs else ["supplied", "full"]
    requests = [{"custom_id": f"{mode[0]}{n:03d}", "mode": mode, "item_id": item["id"]}
                for mode in modes for n, item in enumerate(_items(mode, inputs))]
    data = {"name": name, "model": model, "effort": effort, "prompt_version": QA_PROMPT_VERSION, "max_tokens": max_tokens,
            "inputs": inputs, "modes": modes, "run_prefix": run_prefix, "label": label, "freeze": str(freeze),
            "freeze_sha256": hashlib.sha256(Path(freeze).read_bytes()).hexdigest(), "requests": requests,
            "created_at": datetime.now(timezone.utc)}
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "plan.json", data)
    return {k: v for k, v in data.items() if k != "requests"} | {"requests": len(requests)}


def _lookup(data: dict) -> dict:
    inputs = data.get("inputs")
    return {(mode, item["id"]): item for mode in data.get("modes", ["supplied", "full"]) for item in _items(mode, inputs)}


def count(name: str, client, *, run_cap_usd: Decimal) -> dict:
    out = ROOT / name
    data = _read(out / "plan.json")
    items = _lookup(data)
    for r in data["requests"]:
        if "input_tokens" not in r:
            body = _body(data["model"], 1, data["effort"], items[(r["mode"], r["item_id"])])
            r["input_tokens"] = client.messages.count_tokens(model=body["model"], system=body["system"], messages=body["messages"],
                                                             output_config=body["output_config"]).input_tokens
            _write(out / "plan.json", data)
    n = len(data["requests"])
    input_cost = sum((cost(data["model"], TokenUsage(input_uncached=r["input_tokens"]), batch=True) for r in data["requests"]), Decimal(0))
    # the output allowance is fixed by the plan (identical conditions across runs); the cap only bounds the worst case
    max_tokens = data.get("max_tokens") or MAX_TOKENS
    worst = sum((reservation(data["model"], r["input_tokens"], max_tokens, batch=True) for r in data["requests"]), Decimal(0))
    if worst > run_cap_usd:
        raise ValueError(f"worst case ${worst:.4f} exceeds the run cap ${run_cap_usd}")
    data.update(max_tokens=max_tokens, run_cap_usd=str(run_cap_usd), input_tokens_total=sum(r["input_tokens"] for r in data["requests"]),
                input_cost_usd=str(input_cost.quantize(Decimal("0.0001"))), worst_case_usd=str(worst.quantize(Decimal("0.0001"))))
    _write(out / "plan.json", data)
    return {k: data[k] for k in ("model", "effort", "max_tokens", "input_tokens_total", "input_cost_usd", "worst_case_usd", "run_cap_usd")} | {"requests": n}


def submit(name: str, client, ledger: SpendLedger) -> dict:
    out = ROOT / name
    data = _read(out / "plan.json")
    items = _lookup(data)
    if "worst_case_usd" not in data:
        raise ValueError("run count first")
    if (out / "state.json").exists():
        raise ValueError(f"batch already {_read(out / 'state.json')['status']}; never resubmitted automatically")
    requests, entries = [], []
    for r in data["requests"]:
        body = _body(data["model"], data["max_tokens"], data["effort"], items[(r["mode"], r["item_id"])])
        r["request_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        requests.append({"custom_id": r["custom_id"], "params": body})
        entries.append({"scope": "modelforge", "job_id": None, "stage": f"financebench-qa:{r['mode']}:{r['item_id']}", "attempt": 1,
                        "request_sha256": r["request_sha256"], "provider": "anthropic", "model": data["model"],
                        "amount_usd": reservation(data["model"], r["input_tokens"], data["max_tokens"], batch=True)})
    ids = ledger.reserve_many(entries, run_cap_usd=Decimal(data["run_cap_usd"]))
    for r, attempt_id in zip(data["requests"], ids):
        r["attempt_id"] = str(attempt_id)
    _write(out / "plan.json", data)
    state = {"status": "creating", "requested_at": datetime.now(timezone.utc), "batch_id": None}
    _write(out / "state.json", state)
    try:
        batch = client.messages.batches.create(requests=requests)
    except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
        ledger.unknown_many(ids, f"batch create outcome unknown: {type(exc).__name__}")
        state.update(status="create_unknown", error=str(exc)[:300])
        _write(out / "state.json", state)
        raise
    except anthropic.APIStatusError as exc:
        ledger.failed_many(ids, f"batch rejected: {exc.status_code}")
        state.update(status="rejected", error=f"{exc.status_code}: {exc.message}"[:300])
        _write(out / "state.json", state)
        raise
    state.update(status="submitted", batch_id=batch.id, submitted_at=datetime.now(timezone.utc))
    _write(out / "state.json", state)
    ledger.mark_sent_many(ids, note=f"batch {batch.id}")
    return state


def collect(name: str, client, ledger: SpendLedger, runs_dir: Path = Path("artifacts/runs")) -> dict:
    out = ROOT / name
    data, state = _read(out / "plan.json"), _read(out / "state.json")
    if state["status"] not in ("submitted", "ended"):
        raise ValueError(f"batch state is {state['status']}")
    batch = client.messages.batches.retrieve(state["batch_id"])
    if batch.processing_status != "ended":
        return {"status": batch.processing_status, "counts": _counts(batch.request_counts)}
    by_custom = {r["custom_id"]: r for r in data["requests"]}
    lines = {mode: {} for mode in data.get("modes", ["supplied", "full"])}
    spent = Decimal(0)
    for result in client.messages.batches.results(state["batch_id"]):
        r = by_custom[result.custom_id]
        if result.result.type == "succeeded":
            message = result.result.message
            usage = TokenUsage.from_anthropic(message.usage)
            c = cost(data["model"], usage, batch=True)
            spent += c
            text = next((b.text for b in message.content if b.type == "text"), None)
            status = {"refusal": "refused", "max_tokens": "truncated"}.get(message.stop_reason, "ok")
            ledger.complete(r["attempt_id"], cost_usd=c, usage=usage.__dict__,
                            response={"text": text, "stop_reason": message.stop_reason, "id": message.id, "batch_id": state["batch_id"]},
                            latency_ms=None, model_version=message.model, stop_reason=message.stop_reason)
            lines[r["mode"]][r["item_id"]] = {"item_id": r["item_id"], "text": text if status == "ok" else None, "raw": text,
                                              "status": status, "error": None, "latency_ms": None,
                                              "input_tokens": usage.input_uncached, "output_tokens": usage.output}
        else:
            outer = getattr(result.result, "error", None)
            error = getattr(getattr(outer, "error", None), "type", None) or getattr(outer, "type", None) or result.result.type
            ledger.failed(r["attempt_id"], f"batch result {result.result.type}: {error}")
            lines[r["mode"]][r["item_id"]] = {"item_id": r["item_id"], "text": None, "raw": None, "status": "error",
                                              "error": f"batch result {result.result.type}: {error}", "latency_ms": None,
                                              "input_tokens": None, "output_tokens": None}
    inputs = data.get("inputs")
    for mode, preds in lines.items():
        source = Path(inputs[mode]["path"]) if inputs else FROZEN[mode]
        run = runs_dir / f"{data.get('run_prefix', '20261110-qa16-fb')}-{mode}-haiku"
        run.mkdir(parents=True, exist_ok=True)
        (run / "items.jsonl").write_bytes(source.read_bytes())
        order = [json.loads(line)["id"] for line in source.read_text().splitlines() if line]
        (run / "predictions.jsonl").write_text("".join(json.dumps(preds[i]) + "\n" for i in order if i in preds))
        (run / "config.json").write_text(json.dumps({"label": f"{data.get('label', 'finsight-qa-v1.6')} fb-{mode} Claude batch", "model": data["model"],
                                                     "effort": data["effort"], "max_tokens": data["max_tokens"], "batch_id": state["batch_id"],
                                                     "prompt_version": data["prompt_version"], "freeze_sha256": data["freeze_sha256"]}, indent=1) + "\n")
    state.update(status="ended", collected_at=datetime.now(timezone.utc), actual_cost_usd=str(spent))
    _write(out / "state.json", state)
    return {"status": "ended", "actual_cost_usd": str(spent), "counts": _counts(batch.request_counts)}
