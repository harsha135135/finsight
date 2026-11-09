"""Run one adapter over frozen manifest items and write predictions plus a scorecard.

Runs are resumable: predictions are appended to ``predictions.jsonl`` as each item
finishes, and a restarted run skips items already recorded. Every item stays in
the denominator whatever its outcome.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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

SCORECARD_VERSION = "3"


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


def run(manifest_name: str, selection: str, adapter: Adapter, out_dir: Path, limit: int | None = None,
        concurrency: int = 1) -> Path:
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
        "concurrency": concurrency,
        "server": adapter.server_info() if hasattr(adapter, "server_info") else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = out_dir / "config.json"
    if config_path.exists() and done:
        previous = json.loads(config_path.read_text())
        keys = ("manifest_sha256", "selection", "item_ids", "adapter", "settings", "prompt_sha256", "scorecard_version",
                "concurrency", "server")
        changed = [k for k in keys if previous.get(k) != config.get(k)]
        if changed:
            raise ValueError(f"{out_dir} holds predictions from a different configuration ({', '.join(changed)}); "
                             "use a new output directory")
        config["started_at"] = previous.get("started_at", config["started_at"])
    config_path.write_text(json.dumps(config, indent=1) + "\n")
    todo = [item for item in items if item.id not in done]
    lock = threading.Lock()
    session_start = time.monotonic()  # the WSL wall clock can jump; durations use a monotonic clock
    with predictions_path.open("a", encoding="utf-8") as handle:
        def work(item: EvalItem) -> None:
            prediction = evaluate_item(item, adapter)
            with lock:
                handle.write(prediction.model_dump_json() + "\n")
                handle.flush()
        if concurrency <= 1:
            for item in todo:
                work(item)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(work, todo))
    if todo:
        # Wall-clock per session, stored so the scorecard (and throughput) regenerates from saved files.
        with (out_dir / "timing.jsonl").open("a") as timing:
            timing.write(json.dumps({"items": len(todo), "seconds": round(time.monotonic() - session_start, 3),
                                     "concurrency": concurrency}) + "\n")
    return write_scorecard(out_dir, items)


def run_items(items: list[EvalItem], adapter: Adapter, out_dir: Path, label: str, extra: dict | None = None) -> Path:
    """Run already-built items (e.g. FinanceBench) with the same validation as `run`; resumable. Scoring is done
    by the dataset's own scorer, not the FinQA scorecard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    done = set()
    if predictions_path.exists():
        done = {json.loads(line)["item_id"] for line in predictions_path.read_text().splitlines() if line.strip()}
    config = {"selection": label, "item_ids": [i.id for i in items], "adapter": adapter.config_id,
              "settings": adapter.settings, "prompt_version": prompts.PROMPT_VERSION, "prompt_sha256": prompts.fingerprint(),
              "items_sha256": hashlib.sha256("".join(i.model_dump_json() for i in items).encode()).hexdigest(),
              "commit": _commit(), "server": adapter.server_info() if hasattr(adapter, "server_info") else None,
              "started_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
    config_path = out_dir / "config.json"
    if config_path.exists() and done:
        previous = json.loads(config_path.read_text())
        changed = [k for k in ("item_ids", "adapter", "settings", "prompt_sha256", "items_sha256", "server") if previous.get(k) != config.get(k)]
        if changed:
            raise ValueError(f"{out_dir} holds predictions from a different configuration ({', '.join(changed)})")
        config["started_at"] = previous["started_at"]
    config_path.write_text(json.dumps(config, indent=1) + "\n")
    session_start = time.monotonic()
    todo = [i for i in items if i.id not in done]
    with predictions_path.open("a", encoding="utf-8") as handle:
        for item in todo:
            handle.write(evaluate_item(item, adapter).model_dump_json() + "\n")
            handle.flush()
    if todo:
        with (out_dir / "timing.jsonl").open("a") as timing:
            timing.write(json.dumps({"items": len(todo), "seconds": round(time.monotonic() - session_start, 3), "concurrency": 1}) + "\n")
    return out_dir


def _percent_scale_error(prediction: Prediction, item: EvalItem) -> bool:
    """Wrong answer that would be right divided by 100 (a percent-vs-fraction convention error)."""
    if prediction.correct or prediction.executed in (None, "yes", "no") or item.gold_answer in (None, "yes", "no"):
        return False
    return matches_gold(Decimal(prediction.executed) / 100, item.gold_answer)


def _throughput(timing: list[dict], output_tokens: int) -> dict | None:
    seconds = sum(t["seconds"] for t in timing)
    if not timing or seconds <= 0:
        return None
    items = sum(t["items"] for t in timing)
    return {"items_per_second": round(items / seconds, 4), "output_tokens_per_second": round(output_tokens / seconds, 2),
            "wall_seconds": round(seconds, 1), "sessions": len(timing),
            "concurrency": sorted({t["concurrency"] for t in timing})}


def _lenient_correct(prediction: Prediction, item: EvalItem) -> bool:
    """Would this answer be correct if quotes could add or drop currency symbols and thousands separators?

    A secondary diagnostic, defined before any fine-tuned run, to separate quote formatting from reasoning.
    """
    if prediction.correct:
        return True
    if not item.answerable or item.gold_answer is None or not prediction.raw_output:
        return False
    try:
        output = AnswerOutput.model_validate_json(prediction.raw_output)
    except ValidationError:
        return False
    rows = {k: [Decimal(v) for v in values] for k, values in item.rows.items()}
    result = validate_answer(output, {p.id: p for p in item.passages}, rows,
                             as_written=item.meta.get("calc_convention") == "as_written", lenient_quotes=True)
    if result.reason_code is not ReasonCode.ANSWERED or result.answer is None:
        return False
    executed = result.answer == "yes" if result.answer in {"yes", "no"} else Decimal(result.answer)
    return matches_gold(executed, item.gold_answer)


def _evidence_subcategories(prediction: Prediction, item: EvalItem) -> set[str]:
    """Why an answer failed evidence validation. These are kept apart: not every subcategory means the
    answer contradicts its evidence (a quote may differ only in formatting, or cite the wrong passage)."""
    import re as _re

    from finsight.normalize import _squash, _squash_lenient, parse_figure

    passages = {p.id: p.text for p in item.passages}
    operands = {}
    try:
        operands = {o.name: o for o in AnswerOutput.model_validate_json(prediction.raw_output or "").operands}
    except ValidationError:
        pass
    found: set[str] = set()
    for issue in prediction.issues:
        if "is not supported by passage" in issue:
            match = _re.match(r"operand (\S+) \(", issue)
            operand = operands.get(match.group(1)) if match else None
            if operand is None:
                found.add("operand not supported (details unavailable)")
                continue
            cited = passages.get(operand.passage_id, "")
            if _squash(operand.original_text) in _squash(cited):
                found.add("value or scale disagrees with the quote" if parse_figure(operand.original_text)
                          else "quote is not a single figure")
            elif _squash_lenient(operand.original_text) in _squash_lenient(cited):
                found.add("quote formatting differs from the passage (e.g. added $ or commas)")
            elif any(_squash(operand.original_text) in _squash(t) for pid, t in passages.items() if pid != operand.passage_id):
                found.add("figure is in a different passage than the one cited")
            else:
                found.add("quoted text not found in the evidence")
        elif "mixes units" in issue:
            found.add("adds or subtracts mixed units")
        elif "required period" in issue:
            found.add("required period missing")
        elif "stated answer" in issue:
            found.add("stated answer disagrees with the executed program")
        else:
            found.add("calculation error")
    return found


def summarize(predictions: list[Prediction], items: list[EvalItem], timing: list[dict] | None = None) -> dict:
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
        "evidence_validation_failure_rate": ci(rate(items, lambda p: p.reason_code is ReasonCode.VALIDATION_CONTRADICTION)),
        "evidence_validation_subcategories": dict(sorted(Counter(
            c for i in items if i.id in by_id and by_id[i.id].reason_code is ReasonCode.VALIDATION_CONTRADICTION
            for c in _evidence_subcategories(by_id[i.id], i)).items())),
        "operational_failure_rate": ci(rate(items, lambda p: p.reason_code is ReasonCode.OPERATIONAL)),
        "reason_codes": dict(sorted(Counter(p.reason_code.value for p in predictions).items())),
        "diagnostic_lenient_quote_accuracy_answerable": ci(
            {i.id: 1.0 if (i.id in by_id and _lenient_correct(by_id[i.id], i)) else 0.0 for i in answerable}),
        "diagnostic_percent_scale_errors": sum(
            1 for i in answerable if i.id in by_id and _percent_scale_error(by_id[i.id], i)),
        "truncated_outputs": sum(a.status == "truncated" for a in attempts),
        "latency_ms": {"p50": pct(latencies, 0.5), "p95": pct(latencies, 0.95)} if latencies else None,
        "ttft_ms": {"p50": pct(ttfts, 0.5), "p95": pct(ttfts, 0.95)} if ttfts else None,
        "tokens": {"input": sum(a.input_tokens_uncached or 0 for a in attempts),
                   "output": sum(a.output_tokens or 0 for a in attempts)},
        "throughput": _throughput(timing or [], sum(a.output_tokens or 0 for a in attempts)),
        "cost_usd": (str(sum(costs, Decimal(0))) if costs and all(c is not None for c in costs) else "unavailable"),
    }


def write_scorecard(out_dir: Path, items: list[EvalItem] | None = None) -> Path:
    """Regenerate the scorecard deterministically from stored predictions and config."""
    config = json.loads((out_dir / "config.json").read_text())
    predictions = [Prediction.model_validate_json(line)
                   for line in (out_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if items is None:
        items = splits.resolve_items(splits.load(config["manifest"]), config["item_ids"])
    timing_path = out_dir / "timing.jsonl"
    timing = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()] if timing_path.exists() else []
    summary = summarize(predictions, items, timing)
    payload = {"config": {k: v for k, v in config.items() if k not in {"item_ids", "started_at"}}, "summary": summary}
    (out_dir / "scorecard.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    lines = [f"# Scorecard: {config['adapter']} on {config['manifest']}/{config['selection']}", "",
             f"- Manifest sha256: `{config['manifest_sha256'][:16]}` · dataset revision `{config['dataset_revision'][:12]}`",
             f"- Prompt `{config['prompt_version']}` sha256 `{config['prompt_sha256'][:16]}` · commit `{config['commit'][:12]}`",
             f"- Settings: {json.dumps(config['settings'], sort_keys=True)}",
             f"- Tolerance: {config['tolerance']}", "", "| Metric | Mean | 95% CI (filing-clustered) | n |", "|---|---:|---|---:|"]
    for key in ("accuracy_answerable", "abstention_recall_unanswerable", "false_abstention_answerable",
                "invalid_output_rate", "evidence_validation_failure_rate", "operational_failure_rate",
                "diagnostic_lenient_quote_accuracy_answerable"):
        value = summary[key]
        if value:
            lines.append(f"| {key} | {value['mean']:.4f} | {value['ci95'][0]:.4f} – {value['ci95'][1]:.4f} | {value['n']} |")
    lines += ["", f"Reason codes: {json.dumps(summary['reason_codes'])}",
              "Evidence-validation failures by subcategory (items; one item can have several): "
              f"{json.dumps(summary['evidence_validation_subcategories'])}",
              f"Latency ms: {summary['latency_ms']} · TTFT ms: {summary['ttft_ms']} · tokens: {summary['tokens']} · cost USD: {summary['cost_usd']}",
              f"Throughput: {summary['throughput']} · server: {json.dumps(config.get('server'), sort_keys=True)}",
              f"Missing predictions: {summary['missing_predictions']} · truncated outputs: {summary['truncated_outputs']}", ""]
    path = out_dir / "scorecard.md"
    path.write_text("\n".join(lines))
    return path



def agreement(run_a: Path, run_b: Path) -> dict:
    """How often two runs of the same items agree (fresh reruns are measured, not assumed identical)."""
    def load(run: Path) -> dict[str, Prediction]:
        return {p.item_id: p for p in (Prediction.model_validate_json(line)
                                       for line in (run / "predictions.jsonl").read_text().splitlines() if line.strip())}
    a, b = load(run_a), load(run_b)
    common = sorted(a.keys() & b.keys())
    if not common:
        return {"common_items": 0}
    same_raw = [i for i in common if a[i].raw_output == b[i].raw_output]
    same_outcome = [i for i in common if (a[i].reason_code, a[i].executed) == (b[i].reason_code, b[i].executed)]
    same_correct = [i for i in common if a[i].correct == b[i].correct]
    return {"common_items": len(common), "identical_raw_output": round(len(same_raw) / len(common), 4),
            "identical_outcome": round(len(same_outcome) / len(common), 4),
            "identical_correctness": round(len(same_correct) / len(common), 4),
            "differing_items": sorted(set(common) - set(same_outcome))[:50]}


def compare(run_a: Path, run_b: Path) -> dict:
    """Paired comparison of two runs on the same items (A minus B), with filing-clustered 95% CIs.

    Both runs must cover the same manifest items with the same prompt; metrics are recomputed from the
    saved predictions, so the report regenerates exactly.
    """
    configs = [json.loads((run / "config.json").read_text()) for run in (run_a, run_b)]
    for key in ("manifest_sha256", "item_ids", "prompt_sha256"):
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"runs differ in {key}; a paired comparison needs the same items and prompt")
    items = splits.resolve_items(splits.load(configs[0]["manifest"]), configs[0]["item_ids"])
    groups = {i.id: i.group for i in items}
    runs = []
    for run in (run_a, run_b):
        runs.append({p.item_id: p for p in (Prediction.model_validate_json(line)
                                            for line in (run / "predictions.jsonl").read_text().splitlines() if line.strip())})
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    metrics = {
        "accuracy_answerable": (answerable, lambda p, i: bool(p.correct)),
        "diagnostic_lenient_quote_accuracy_answerable": (answerable, lambda p, i: _lenient_correct(p, i)),
        "abstention_recall_unanswerable": (unanswerable, lambda p, i: p.reason_code is ReasonCode.ABSTAIN_EVIDENCE_ABSENT),
        "false_abstention_answerable": (answerable, lambda p, i: p.reason_code is ReasonCode.ABSTAIN_EVIDENCE_ABSENT),
        "invalid_output_rate": (items, lambda p, i: p.reason_code is ReasonCode.MODEL_OUTPUT_INVALID),
        "evidence_validation_failure_rate": (items, lambda p, i: p.reason_code is ReasonCode.VALIDATION_CONTRADICTION),
        "operational_failure_rate": (items, lambda p, i: p.reason_code is ReasonCode.OPERATIONAL),
    }
    result: dict = {"a": configs[0]["adapter"], "b": configs[1]["adapter"], "items": len(items), "metrics": {}}
    for name, (subset, predicate) in metrics.items():
        if not subset:  # e.g. an all-unanswerable slice has no answerable metrics
            result["metrics"][name] = {"a": None, "b": None, "a_minus_b": None, "ci95": None, "n": 0}
            continue
        values = [{i.id: 1.0 if (i.id in run and predicate(run[i.id], i)) else 0.0 for i in subset} for run in runs]
        mean_a = stats.cluster_bootstrap_mean(values[0], groups)[0] if subset else None
        mean_b = stats.cluster_bootstrap_mean(values[1], groups)[0] if subset else None
        diff, low, high = stats.paired_difference(values[0], values[1], groups) if subset else (None, None, None)
        result["metrics"][name] = {"a": round(mean_a, 4), "b": round(mean_b, 4), "a_minus_b": round(diff, 4),
                                   "ci95": [round(low, 4), round(high, 4)], "n": len(subset)}
    flips = Counter((runs[1][i.id].correct, runs[0][i.id].correct) for i in answerable if i.id in runs[0] and i.id in runs[1])
    result["answerable_correctness_transitions_b_to_a"] = {
        "both_correct": flips[(True, True)], "fixed_by_a": flips[(False, True)],
        "broken_by_a": flips[(True, False)], "both_wrong": flips[(False, False)]}
    return result


def relabel(run_dir: Path, labels_path: Path, out_dir: Path) -> Path:
    """Copy a run with corrected labels: unanswerable variants not verified by the audit are excluded.

    The source run is left untouched; the copy records the label file and its hash, the excluded ids
    and the source run, and its scorecard regenerates from the copied predictions like any other run.
    Excluded variants are not valid negatives (derivable or ambiguous) and have no gold answer, so they
    leave every denominator.
    """
    import hashlib
    import shutil

    labels = json.loads(labels_path.read_text())
    excluded = {vid for vid, entry in labels["candidates"].items() if entry["decision"] != "verified_unanswerable"}
    config = json.loads((run_dir / "config.json").read_text())
    dropped = sorted(i for i in config["item_ids"] if i in excluded)
    config["item_ids"] = [i for i in config["item_ids"] if i not in excluded]
    config["relabel"] = {"labels": str(labels_path), "labels_version": labels.get("version"),
                         "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                         "excluded_items": dropped, "source_run": str(run_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    lines = [line for line in (run_dir / "predictions.jsonl").read_text().splitlines()
             if line.strip() and json.loads(line)["item_id"] not in excluded]
    (out_dir / "predictions.jsonl").write_text("\n".join(lines) + "\n")
    if (run_dir / "timing.jsonl").exists():
        shutil.copy(run_dir / "timing.jsonl", out_dir / "timing.jsonl")  # throughput refers to the source run
    return write_scorecard(out_dir)


def _program_outcome(prediction: Prediction, item: EvalItem) -> tuple[bool | None, bool | None]:
    """(program correct, stated answer consistent) for an answerable item whose program executed.

    The program executes only after every evidence check passes; ``None`` means it never ran
    (invalid output, failed evidence checks, abstention, no output). Consistency is undefined
    when the stated answer could not be parsed.
    """
    if not item.answerable or item.gold_answer is None or not prediction.raw_output:
        return None, None
    try:
        output = AnswerOutput.model_validate_json(prediction.raw_output)
    except ValidationError:
        return None, None
    rows = {k: [Decimal(v) for v in values] for k, values in item.rows.items()}
    result = validate_answer(output, {p.id: p for p in item.passages}, rows,
                             as_written=item.meta.get("calc_convention") == "as_written")
    if result.calculation is None:
        return None, None
    value = result.calculation.result
    executed = value == "yes" if value in {"yes", "no"} else Decimal(value)
    correct = matches_gold(executed, item.gold_answer)
    if result.reason_code is ReasonCode.ANSWERED:
        return correct, True
    if result.reason_code is ReasonCode.VALIDATION_CONTRADICTION:
        return correct, False
    return correct, None


def diagnose(run_dir: Path, baseline_dir: Path | None = None) -> dict:
    """Analyses beyond the frozen scorecard format, written to diagnostics.json.

    * program accuracy: the executed program's value is correct, whatever the stated answer says
      (FinSight takes final numbers from the calculator, so this isolates operand and formula choice);
    * arithmetic consistency: among executed programs with a parseable stated answer, how often the
      stated answer agrees with the program. Inconsistency does not rule out memorization.
    """
    def analyse(run: Path) -> tuple[dict, dict, dict]:
        config = json.loads((run / "config.json").read_text())
        items = splits.resolve_items(splits.load(config["manifest"]), config["item_ids"])
        preds = {p.item_id: p for p in (Prediction.model_validate_json(line)
                                        for line in (run / "predictions.jsonl").read_text().splitlines() if line.strip())}
        outcomes = {i.id: _program_outcome(preds[i.id], i) for i in items if i.answerable and i.id in preds}
        return config, {i.id: i.group for i in items}, outcomes

    config, groups, outcomes = analyse(run_dir)
    program = {k: 1.0 if v[0] else 0.0 for k, v in outcomes.items()}
    consistent = [v[1] for v in outcomes.values() if v[1] is not None]
    executed_wrong_stated = sum(1 for v in outcomes.values() if v[1] is False and v[0])
    mean, low, high = stats.cluster_bootstrap_mean(program, groups)
    report = {"run": str(run_dir), "adapter": config["adapter"],
              "program_accuracy_answerable": {"mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)], "n": len(program)},
              "arithmetic_consistency": {"rate": round(sum(consistent) / len(consistent), 4) if consistent else None,
                                         "n_executed_with_stated_answer": len(consistent)},
              "inconsistent_but_program_correct": executed_wrong_stated,
              "note": "inconsistency shows stated answer and program disagree; it does not rule out memorization"}
    if baseline_dir is not None:
        _, _, base = analyse(baseline_dir)
        base_program = {k: 1.0 if v[0] else 0.0 for k, v in base.items()}
        diff, low, high = stats.paired_difference(program, base_program, groups)
        report["paired_program_accuracy_vs_baseline"] = {"baseline": str(baseline_dir), "a_minus_b": round(diff, 4),
                                                         "ci95": [round(low, 4), round(high, 4)]}
    (run_dir / "diagnostics.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report
