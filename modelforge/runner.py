"""Run one adapter over frozen manifest items and write predictions plus a scorecard.

Runs are resumable: predictions are appended to ``predictions.jsonl`` as each item
finishes, and a restarted run skips items already recorded. Every item stays in
the denominator whatever its outcome.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from finsight.contracts import AnswerOutput, Prediction, ReasonCode
from finsight.validation import validate_answer
from modelforge import prompts, splits, stats
from modelforge.adapters import Adapter
from modelforge.datasets.finqa import matches_gold
from modelforge.items import EvalItem

SCORECARD_VERSION = "1"


def _commit() -> str:
    build_commit = Path("BUILD_COMMIT")
    if not Path(".git").exists() and build_commit.exists():
        return build_commit.read_text().strip()
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate_item(item: EvalItem, adapter: Adapter) -> Prediction:
    generation = adapter.answer(item, prompts.SYSTEM, prompts.user_message(item), prompts.answer_schema())
    base = {"item_id": item.id, "config_id": adapter.config_id, "attempts": generation.attempts,
            "raw_output": generation.text}
    if generation.text is None:
        status = generation.attempts[-1].status if generation.attempts else "error"
        return Prediction(reason_code=ReasonCode.OPERATIONAL, issues=[f"no output ({status})"],
                          correct=False, **base)
    try:
        output = AnswerOutput.model_validate_json(generation.text)
    except ValidationError as exc:
        return Prediction(reason_code=ReasonCode.MODEL_OUTPUT_INVALID, issues=[str(exc).splitlines()[0]],
                          correct=False, **base)
    rows = {k: [Decimal(v) for v in values] for k, values in item.rows.items()}
    result = validate_answer(output, {p.id: p for p in item.passages}, rows,
                             as_written=item.meta.get("calc_convention") == "as_written")
    if item.answerable:
        correct = False
        if result.reason_code is ReasonCode.ANSWERED and item.gold_answer is not None:
            executed = result.answer == "yes" if result.answer in {"yes", "no"} else Decimal(result.answer)
            correct = matches_gold(executed, item.gold_answer)
    else:
        correct = result.reason_code is ReasonCode.ABSTAIN_EVIDENCE_ABSENT
    return Prediction(reason_code=result.reason_code, answer=output.answer, executed=result.answer,
                      correct=correct, issues=result.issues, **base)


def run(manifest_name: str, selection: str, adapter: Adapter, out_dir: Path, limit: int | None = None) -> Path:
    manifest = splits.load(manifest_name)
    ids = (manifest.get("samples", {}).get(selection) or manifest.get("slices", {}).get(selection)
           or manifest["splits"].get(selection))
    if ids is None:
        raise KeyError(f"{selection!r} is neither a sample nor a split of {manifest_name}")
    ids = ids[:limit] if limit else ids
    items = splits.resolve_items(manifest, ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    done = set()
    if predictions_path.exists():
        done = {json.loads(line)["item_id"] for line in predictions_path.read_text().splitlines() if line.strip()}
    config = {
        "scorecard_version": SCORECARD_VERSION, "manifest": manifest_name, "manifest_sha256": manifest["sha256"],
        "dataset_revision": manifest["revision"], "selection": selection, "item_ids": ids,
        "adapter": adapter.config_id, "settings": adapter.settings, "prompt_version": prompts.PROMPT_VERSION,
        "prompt_sha256": prompts.fingerprint(), "commit": _commit(),
        "tolerance": "FinQA: executed value rounded to 5 decimals as float equals gold",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = out_dir / "config.json"
    if config_path.exists() and done:
        previous = json.loads(config_path.read_text())
        keys = ("manifest_sha256", "selection", "item_ids", "adapter", "settings", "prompt_sha256", "scorecard_version")
        changed = [k for k in keys if previous.get(k) != config.get(k)]
        if changed:
            raise ValueError(f"{out_dir} holds predictions from a different configuration ({', '.join(changed)}); "
                             "use a new output directory")
        config["started_at"] = previous.get("started_at", config["started_at"])
    config_path.write_text(json.dumps(config, indent=1) + "\n")
    with predictions_path.open("a", encoding="utf-8") as handle:
        for item in items:
            if item.id in done:
                continue
            prediction = evaluate_item(item, adapter)
            handle.write(prediction.model_dump_json() + "\n")
            handle.flush()
    return write_scorecard(out_dir, items)


def _percent_scale_error(prediction: Prediction, item: EvalItem) -> bool:
    """Wrong answer that would be right divided by 100 (a percent-vs-fraction convention error)."""
    if prediction.correct or prediction.executed in (None, "yes", "no") or item.gold_answer in (None, "yes", "no"):
        return False
    return matches_gold(Decimal(prediction.executed) / 100, item.gold_answer)


def summarize(predictions: list[Prediction], items: list[EvalItem]) -> dict:
    by_id = {p.item_id: p for p in predictions}
    groups = {i.id: i.group for i in items}
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    missing = [i.id for i in items if i.id not in by_id]

    def rate(subset, predicate):
        return {i.id: 1.0 if (i.id in by_id and predicate(by_id[i.id])) else 0.0 for i in subset}

    def ci(values):
        if not values:
            return None
        mean, low, high = stats.cluster_bootstrap_mean(values, groups)
        return {"mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)], "n": len(values)}

    attempts = [a for p in predictions for a in p.attempts]
    latencies = sorted(a.latency_ms for a in attempts if a.latency_ms is not None)
    ttfts = sorted(a.ttft_ms for a in attempts if a.ttft_ms is not None)
    costs = [a.estimated_cost_usd for a in attempts]

    def pct(values, q):
        return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))] if values else None

    return {
        "items": len(items), "answerable": len(answerable), "unanswerable": len(unanswerable),
        "missing_predictions": len(missing),
        "accuracy_answerable": ci(rate(answerable, lambda p: bool(p.correct))),
        "abstention_recall_unanswerable": ci(rate(unanswerable, lambda p: p.reason_code is ReasonCode.ABSTAIN_EVIDENCE_ABSENT)),
        "false_abstention_answerable": ci(rate(answerable, lambda p: p.reason_code is ReasonCode.ABSTAIN_EVIDENCE_ABSENT)),
        "invalid_output_rate": ci(rate(items, lambda p: p.reason_code is ReasonCode.MODEL_OUTPUT_INVALID)),
        "contradiction_rate": ci(rate(items, lambda p: p.reason_code is ReasonCode.VALIDATION_CONTRADICTION)),
        "operational_failure_rate": ci(rate(items, lambda p: p.reason_code is ReasonCode.OPERATIONAL)),
        "reason_codes": dict(sorted(Counter(p.reason_code.value for p in predictions).items())),
        "diagnostic_percent_scale_errors": sum(
            1 for i in answerable if i.id in by_id and _percent_scale_error(by_id[i.id], i)),
        "truncated_outputs": sum(a.status == "truncated" for a in attempts),
        "latency_ms": {"p50": pct(latencies, 0.5), "p95": pct(latencies, 0.95)} if latencies else None,
        "ttft_ms": {"p50": pct(ttfts, 0.5), "p95": pct(ttfts, 0.95)} if ttfts else None,
        "tokens": {"input": sum(a.input_tokens_uncached or 0 for a in attempts),
                   "output": sum(a.output_tokens or 0 for a in attempts)},
        "cost_usd": (str(sum(costs, Decimal(0))) if costs and all(c is not None for c in costs) else "unavailable"),
    }


def write_scorecard(out_dir: Path, items: list[EvalItem] | None = None) -> Path:
    """Regenerate the scorecard deterministically from stored predictions and config."""
    config = json.loads((out_dir / "config.json").read_text())
    predictions = [Prediction.model_validate_json(line)
                   for line in (out_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if items is None:
        items = splits.resolve_items(splits.load(config["manifest"]), config["item_ids"])
    summary = summarize(predictions, items)
    payload = {"config": {k: v for k, v in config.items() if k not in {"item_ids", "started_at"}}, "summary": summary}
    (out_dir / "scorecard.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    lines = [f"# Scorecard: {config['adapter']} on {config['manifest']}/{config['selection']}", "",
             f"- Manifest sha256: `{config['manifest_sha256'][:16]}` · dataset revision `{config['dataset_revision'][:12]}`",
             f"- Prompt `{config['prompt_version']}` sha256 `{config['prompt_sha256'][:16]}` · commit `{config['commit'][:12]}`",
             f"- Settings: {json.dumps(config['settings'], sort_keys=True)}",
             f"- Tolerance: {config['tolerance']}", "", "| Metric | Mean | 95% CI (filing-clustered) | n |", "|---|---:|---|---:|"]
    for key in ("accuracy_answerable", "abstention_recall_unanswerable", "false_abstention_answerable",
                "invalid_output_rate", "contradiction_rate", "operational_failure_rate"):
        value = summary[key]
        if value:
            lines.append(f"| {key} | {value['mean']:.4f} | {value['ci95'][0]:.4f} – {value['ci95'][1]:.4f} | {value['n']} |")
    lines += ["", f"Reason codes: {json.dumps(summary['reason_codes'])}",
              f"Latency ms: {summary['latency_ms']} · TTFT ms: {summary['ttft_ms']} · tokens: {summary['tokens']} · cost USD: {summary['cost_usd']}",
              f"Missing predictions: {summary['missing_predictions']} · truncated outputs: {summary['truncated_outputs']}", ""]
    path = out_dir / "scorecard.md"
    path.write_text("\n".join(lines))
    return path

