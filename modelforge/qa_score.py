"""Scoring for ``finsight-qa-v1`` runs (rules: docs/financebench-qa-plan.md, gold: evals/financebench/gold-dev-v1.json)."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from finsight.contracts import Passage
from finsight.contracts.qa import QAOutput
from finsight.qa import requested_years, validate_qa
from modelforge import stats

FISCAL = {k: tuple(v) if v else None for k, v in json.loads(Path("evals/financebench/fiscal-years-dev-v1.json").read_text())["fiscal_year_end"].items()}


def _load(run: Path) -> tuple[dict, dict]:
    items = {i["id"]: i for i in map(json.loads, (run / "items.jsonl").read_text().splitlines())}
    preds = {p["item_id"]: p for p in map(json.loads, (run / "predictions.jsonl").read_text().splitlines())}
    return items, preds


def _parse(pred: dict):
    if pred["status"] != "ok" or not pred.get("text"):
        return None, "operational" if pred["status"] in ("truncated", "error", "retryable_error", "uncertain") else "invalid"
    try:
        return QAOutput.model_validate_json(pred["text"]), None
    except ValueError:
        return None, "invalid"


def _validate(output, item: dict, requested: set[int], company: str | None, strict: bool):
    supplied = {p["id"]: Passage.model_validate(p) for p in item["passages"]}
    return validate_qa(output, supplied, requested, fiscal_year_end=FISCAL.get((company or "").upper().replace(" ", "").replace("&", "")), strict=strict)


def _scale_for(question: str) -> Decimal:
    q = question.lower()
    return Decimal(10) ** (9 if "billion" in q else 6 if "million" in q else 3 if "thousand" in q else 0)


def numeric_outcome(value: str | None, gold: dict, question: str) -> str:
    if value is None:
        return "no_value"
    v, g = Decimal(value), Decimal(gold["gold_value"])
    decimals = gold["decimals"]
    percent = gold["gold_unit"] in ("percent", "percentage points")
    base = v / _scale_for(question) if not percent else v

    def close(x: Decimal) -> bool:
        if gold["sign_policy"] == "magnitude":
            x, target = abs(x), abs(g)
        else:
            target = g
        if target == 0:
            return x == 0
        try:
            return abs(x - target) <= abs(target) * Decimal("0.01") or round(x, decimals) == round(target, decimals)
        except InvalidOperation:
            return False
    candidates = [base] + ([base * 100] if percent else [])
    if any(close(c) for c in candidates):
        return "correct"
    if any(close(c * Decimal(10) ** k) for c in candidates for k in (-9, -6, -3, 3, 6, 9)):
        return "scale_slip"
    return "wrong"


def score_financebench(run: Path, gold: dict, review: dict | None = None, retrieval: dict | None = None) -> dict:
    items, preds = _load(run)
    rows = []
    for item_id, item in items.items():
        g = gold["questions"][item_id]
        pred = preds[item_id]
        output, failure = _parse(pred)
        requested = set(g["requested_years"])
        row = {"id": item_id, "company": item["company"], "expected": g["expected_type"], "latency_ms": pred.get("latency_ms"),
               "retrieved": (retrieval or {}).get(item_id, {}).get("gold_pages_retrieved") if retrieval else None,
               "disputed": "disputed" in g}
        if failure:
            row.update(answer_type=None, stage="output_or_operational", correct_strict=False, correct_lenient=False)
            rows.append(row)
            continue
        strict = _validate(output, item, requested, item["company"], True)
        lenient = _validate(output, item, requested, item["company"], False)
        row.update(answer_type=output.answer_type, status_strict=strict.status, status_lenient=lenient.status,
                   unverified=lenient.period_unverified, period_rejects=sum(o.period == "reject" for o in lenient.operands),
                   claims_supported=sum(c.supported for c in lenient.claims), claims=len(lenient.claims))
        if output.answer_type == "insufficient_evidence":
            stage, outcome = "insufficient_evidence", None
        elif g["expected_type"] == "numeric":
            outcome = numeric_outcome(lenient.value, g, item["question"]) if output.answer_type == "numeric" else None
            stage = ("wrong_type" if output.answer_type != "numeric" else
                     "validation_rejection" if lenient.status != "accepted" else
                     "correct" if outcome == "correct" else "wrong_answer")
        elif g["expected_type"] == "boolean":
            outcome = output.boolean if output.answer_type == "boolean" else None
            stage = ("wrong_type" if output.answer_type != "boolean" else
                     "validation_rejection" if lenient.status != "accepted" else
                     "correct" if outcome == g["gold_boolean"] else "wrong_answer")
        else:
            outcome = (review or {}).get(f"{run.name}::{item_id}")
            stage = "correct" if outcome == "correct" else "partial" if outcome == "partial" else "wrong_answer"
        correct = stage == "correct"
        row.update(outcome=outcome, stage=stage, correct_lenient=correct,
                   correct_strict=correct and (strict.status == "accepted" or g["expected_type"] == "narrative"))
        if row["retrieved"] == "none" and stage != "correct":
            row["stage"] = "retrieval"
        rows.append(row)
    return {"run": run.name, "rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind in ("numeric", "boolean", "narrative"):
        sub = [r for r in rows if r["expected"] == kind]
        out[kind] = {"n": len(sub), "correct_lenient": sum(r["correct_lenient"] for r in sub),
                     "correct_strict": sum(r["correct_strict"] for r in sub),
                     "partial": sum(r.get("stage") == "partial" for r in sub),
                     "stages": dict(Counter(r["stage"] for r in sub)),
                     "answer_types": dict(Counter(str(r.get("answer_type")) for r in sub))}
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms"))
    out["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[max(0, int(len(lat) * 0.95) - 1)]} if lat else None
    out["retrieval"] = dict(Counter(str(r.get("retrieved")) for r in rows))
    out["correct_by_retrieval"] = {k: sum(r["correct_lenient"] for r in rows if r.get("retrieved") == k) for k in ("all", "some", "none")}
    return out


def paired(a: dict, b: dict, kind: str, key: str = "correct_lenient") -> dict:
    ra = {r["id"]: float(r[key]) for r in a["rows"] if r["expected"] == kind}
    rb = {r["id"]: float(r[key]) for r in b["rows"] if r["expected"] == kind}
    diff, low, high = stats.paired_difference(ra, rb, {r["id"]: r["company"] for r in a["rows"]})
    return {"a_minus_b": round(diff, 3), "ci95": [round(low, 3), round(high, 3)], "n": len(ra)}


def finqa_value_matches(lenient, output: QAOutput, gold_answer: str | None) -> bool:
    """For the control only: the answer matches FinQA's gold in base units or as written (both conventions occur)."""
    from finsight.calc import execute
    from modelforge.datasets.finqa import matches_gold

    if gold_answer is None or lenient.value is None:
        return False
    values = [Decimal(lenient.value)]
    try:
        raw = {op.name: Decimal(op.value) for op in output.operands}
        values.append(execute(output.program, raw)[-1] if output.program else raw[output.operands[0].name])
    except Exception:  # noqa: BLE001
        pass
    gold = "yes" if gold_answer == "yes" else gold_answer
    return any(matches_gold(v, gold) for v in values)


def score_controls(run: Path) -> dict:
    items, preds = _load(run)
    tallies = {"negative": Counter(), "positive": Counter()}
    for item_id, item in items.items():
        output, failure = _parse(preds[item_id])
        role = item["role"]
        if failure:
            tallies[role]["invalid_or_operational"] += 1
            continue
        if output.answer_type == "insufficient_evidence":
            tallies[role]["insufficient_evidence"] += 1
            continue
        if output.answer_type != "numeric":
            tallies[role][f"answered_{output.answer_type}"] += 1
            continue
        lenient = _validate(output, item, set(item["requested_years"]), None, strict=False)
        rejects = sum(o.period == "reject" for o in lenient.operands)
        unverified = lenient.period_unverified
        verdict = "period_rejected" if rejects else "period_unverified" if unverified else "period_passed"
        if role == "positive":
            correct = finqa_value_matches(lenient, output, item.get("gold_answer"))
            tallies[role][("correct_" if correct else "wrong_") + verdict] += 1
        else:
            tallies[role]["numeric_" + verdict] += 1
    return {"run": run.name, "negative": dict(tallies["negative"]), "positive": dict(tallies["positive"])}


def review_sheet(runs: list[Path], gold: dict, seed: int = 20261109) -> list[dict]:
    entries = []
    for run in runs:
        items, preds = _load(run)
        for item_id, item in items.items():
            if gold["questions"][item_id]["expected_type"] != "narrative":
                continue
            output, failure = _parse(preds[item_id])
            if failure or output.answer_type == "insufficient_evidence":
                continue
            entries.append({"key": f"{run.name}::{item_id}", "question": item["question"], "gold_id": item_id,
                            "answer_type": output.answer_type, "boolean": output.boolean,
                            "claims": [c.text for c in output.claims][:5],
                            "numeric": [o.original_text for o in output.operands][:4]})
    random.Random(seed).shuffle(entries)
    for n, e in enumerate(entries, 1):
        e["blind_id"] = f"Q{n:03d}"
    return entries
