"""Score FinanceBench development runs with the rules fixed in docs/financebench-dev-plan.md.

Numeric questions: gold = first number in the gold answer (percentage if it carries %); model value = the validator's
executed answer, else the stated answer; correct within 1% relative or equal at the gold's displayed decimals; a
fraction is accepted for a percentage gold; scale slips (x1e3/x1e6) are counted separately, never as correct.
Narrative questions are judged by a blind rubric review (see `review_sheet` / `apply_review`).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from modelforge import stats

GOLD_NUMBER = re.compile(r"^\s*\(?\s*\$?\s*(-?[\d,]*\.?\d+)\s*\)?\s*(%)?")
STAGES = ["extraction", "retrieval", "output_or_operational", "false_abstention", "validation_rejection", "wrong_answer", "correct"]


def numeric_gold(answer: str):
    match = GOLD_NUMBER.match(str(answer))
    if not match:
        return None
    text = match.group(1).replace(",", "")
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    decimals = len(text.split(".")[1]) if "." in text else 0
    return value, bool(match.group(2)), decimals


def _close(value: Decimal, gold: Decimal, decimals: int) -> bool:
    if gold == 0:
        return value == 0
    if abs(value - gold) <= abs(gold) * Decimal("0.01"):
        return True
    try:
        return round(value, decimals) == round(gold, decimals)
    except InvalidOperation:
        return False


def numeric_outcome(value: Decimal | None, gold) -> str:
    if value is None:
        return "no_value"
    g, percent, decimals = gold
    candidates = [value] + ([value * 100] if percent else [])
    if any(_close(c, g, decimals) for c in candidates):
        return "correct"
    if any(_close(c * k, g, decimals) for c in candidates for k in (Decimal(10) ** 3, Decimal(10) ** 6, Decimal(10) ** -3, Decimal(10) ** -6)):
        return "scale_slip"
    return "wrong"


def model_value(prediction: dict) -> Decimal | None:
    for raw in (prediction.get("executed"), _stated(prediction)):
        if raw in (None, "yes", "no"):
            continue
        try:
            return Decimal(str(raw).replace(",", "").replace("$", "").replace("%", "").strip())
        except InvalidOperation:
            continue
    return None


def _stated(prediction: dict):
    try:
        return json.loads(prediction.get("raw_output") or "{}").get("answer")
    except json.JSONDecodeError:
        return None


def cited_pages(prediction: dict, item) -> set[int]:
    try:
        output = json.loads(prediction.get("raw_output") or "{}")
    except json.JSONDecodeError:
        return set()
    pages = {p.id: p.location.page for p in item.passages}
    ids = {op.get("passage_id") for op in output.get("operands") or []}
    ids |= {pid for c in output.get("claims") or [] for pid in (c.get("passage_ids") or [])}
    return {pages[i] for i in ids if i in pages}


def score_run(run_dir: Path, items: list, questions: dict, review: dict | None = None, retrieval: dict | None = None) -> dict:
    preds = {p["item_id"]: p for p in map(json.loads, (run_dir / "predictions.jsonl").read_text().splitlines())}
    rows = []
    for item in items:
        q, p = questions[item.id], preds[item.id]
        gold = numeric_gold(q["answer"])
        kind = "numeric" if gold else "narrative"
        code = p["reason_code"]
        gold_pages = {int(x) for x in item.meta["gold_pages"].split(",")}
        retrieved_ok = None if retrieval is None else retrieval[item.id]["gold_page_retrieved"]
        if kind == "numeric":
            outcome = numeric_outcome(model_value(p), gold) if code not in ("MODEL_OUTPUT_INVALID", "OPERATIONAL", "ABSTAIN_EVIDENCE_ABSENT") else None
            correct = outcome == "correct" and code == "ANSWERED"
            stated_correct = outcome == "correct"
        else:
            verdict = (review or {}).get(f"{run_dir.name}::{item.id}")
            outcome = verdict
            correct = verdict == "correct"
            stated_correct = correct
        if retrieved_ok is False:
            stage = "retrieval"
        elif code in ("MODEL_OUTPUT_INVALID", "OPERATIONAL"):
            stage = "output_or_operational"
        elif code == "ABSTAIN_EVIDENCE_ABSENT":
            stage = "false_abstention"
        elif kind == "numeric" and code == "VALIDATION_CONTRADICTION":
            stage = "validation_rejection"
        elif correct:
            stage = "correct"
        else:
            stage = "wrong_answer"
        latency = sum(a.get("latency_ms") or 0 for a in p.get("attempts") or []) or None
        rows.append({"id": item.id, "company": item.company, "type": q["question_type"], "kind": kind, "reason_code": code,
                     "outcome": outcome, "correct": correct, "stated_correct": stated_correct, "stage": stage,
                     "validated": code == "ANSWERED", "cites_gold_page": bool(cited_pages(p, item) & gold_pages),
                     "retrieved_gold_page": retrieved_ok, "latency_ms": latency})
    return {"run": run_dir.name, "rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind in ("numeric", "narrative"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        n = len(sub)
        out[kind] = {"n": n, "correct": sum(r["correct"] for r in sub), "accuracy": round(sum(r["correct"] for r in sub) / n, 3),
                     "outcomes": dict(Counter(str(r["outcome"]) for r in sub)), "stages": dict(Counter(r["stage"] for r in sub)),
                     "false_abstention": round(sum(r["reason_code"] == "ABSTAIN_EVIDENCE_ABSENT" for r in sub) / n, 3),
                     "validated": round(sum(r["validated"] for r in sub) / n, 3),
                     "cites_gold_page": round(sum(r["cites_gold_page"] for r in sub) / n, 3)}
        if kind == "numeric":
            out[kind]["stated_answer_correct"] = sum(r["stated_correct"] for r in sub)
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    out["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1]} if lat else None
    out["by_type_correct"] = {t: f"{sum(r['correct'] for r in rows if r['type'] == t)}/{sum(r['type'] == t for r in rows)}"
                              for t in sorted({r["type"] for r in rows})}
    return out


def paired(a: dict, b: dict, kind: str) -> dict:
    ra = {r["id"]: float(r["correct"]) for r in a["rows"] if r["kind"] == kind}
    rb = {r["id"]: float(r["correct"]) for r in b["rows"] if r["kind"] == kind}
    groups = {r["id"]: r["company"] for r in a["rows"]}
    diff, low, high = stats.paired_difference(ra, rb, groups)
    return {"a_minus_b": round(diff, 3), "ci95": [round(low, 3), round(high, 3)], "n": len(ra)}


def review_sheet(runs: list[Path], items_by_mode: dict, questions: dict, seed: int = 20261109) -> list[dict]:
    """Blind sheet: every answered narrative output, labels hidden behind a shuffled key."""
    entries = []
    for run in runs:
        mode = "supplied" if "-supplied-" in run.name else "full_document"
        preds = {p["item_id"]: p for p in map(json.loads, (run / "predictions.jsonl").read_text().splitlines())}
        for item in items_by_mode[mode]:
            q = questions[item.id]
            if numeric_gold(q["answer"]):
                continue
            p = preds[item.id]
            if p["reason_code"] in ("MODEL_OUTPUT_INVALID", "OPERATIONAL", "ABSTAIN_EVIDENCE_ABSENT"):
                continue
            out = json.loads(p["raw_output"])
            entries.append({"key": f"{run.name}::{item.id}", "question": q["question"], "gold": str(q["answer"]),
                            "justification": str(q.get("justification") or "")[:400], "answer": out.get("answer"),
                            "claims": [c.get("text") for c in out.get("claims") or []][:4]})
    random.Random(seed).shuffle(entries)
    for n, e in enumerate(entries, 1):
        e["blind_id"] = f"N{n:03d}"
    return entries
