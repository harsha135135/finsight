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

DEV_FISCAL_FILES = ("evals/financebench/fiscal-years-dev-v1.json",)

# Scoring versions (validator claim policy + scorer rules). Records pin the version they were scored with, so their
# scores can be regenerated exactly. "v1.6": frozen 2026-11-10. "v1.7-parse": bug fixes only (claim numbers parsed
# without trailing punctuation; an explicit gold unit scale honoured when the question states none). "v1.7": plus the
# policy change exempting dates' day numbers and numbered document references in claims.
VERSIONS = {"v1.6": {"claim_policy": "v1.6", "gold_unit_scale": False},
            "v1.6+unit": {"claim_policy": "v1.6", "gold_unit_scale": True},
            "v1.7-parse": {"claim_policy": "parse", "gold_unit_scale": True},
            "v1.7": {"claim_policy": "v1.7", "gold_unit_scale": True}}
CURRENT = "v1.7"


def load_fiscal(paths=DEV_FISCAL_FILES) -> dict:
    table: dict = {}
    for path in paths:
        for key, value in json.loads(Path(path).read_text())["fiscal_year_end"].items():
            value = tuple(value) if value else None
            if key in table and table[key] != value:
                raise ValueError(f"conflicting fiscal-year entries for {key} in {path}")
            table[key] = value
    return table


FISCAL = load_fiscal()


def company_key(company: str) -> str:
    return company.upper().replace(" ", "").replace("&", "")


def fiscal_year_end(company: str | None, table: dict | None = None):
    """None for FinQA items (no company); an unknown company is an error, never a silent calendar-year fallback."""
    if company is None:
        return None
    table = FISCAL if table is None else table
    key = company_key(company)
    if key not in table:
        raise KeyError(f"no fiscal-year entry for company {company!r} (key {key!r})")
    return table[key]


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


def _validate(output, item: dict, requested: set[int], company: str | None, strict: bool, version: str = CURRENT, fiscal: dict | None = None):
    supplied = {p["id"]: Passage.model_validate(p) for p in item["passages"]}
    return validate_qa(output, supplied, requested, fiscal_year_end=fiscal_year_end(company, fiscal), strict=strict,
                       claim_policy=VERSIONS[version]["claim_policy"])


def _scale_for(question: str) -> Decimal:
    q = question.lower()
    return Decimal(10) ** (9 if "billion" in q else 6 if "million" in q else 3 if "thousand" in q else 0)


_GOLD_UNIT_SCALE = {"USD thousands": 3, "USD millions": 6, "USD billions": 9}


def numeric_outcome(value: str | None, gold: dict, question: str, version: str = CURRENT) -> str:
    if value is None:
        return "no_value"
    v, g = Decimal(value), Decimal(gold["gold_value"])
    decimals = gold["decimals"]
    percent = gold["gold_unit"] in ("percent", "percentage points")
    scale = _scale_for(question)
    if VERSIONS[version]["gold_unit_scale"] and scale == 1 and gold["gold_unit"] in _GOLD_UNIT_SCALE:
        scale = Decimal(10) ** _GOLD_UNIT_SCALE[gold["gold_unit"]]
    base = v / scale if not percent else v

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


def period_verdict(result) -> str:
    """period_rejected: a figure placed in another year, or a requested year not covered while every figure was placed;
    unverified: no period rejection, but some figure (or the coverage) could not be placed; passed otherwise."""
    if not result.operands:
        return "not_checked"            # rejected before any period check (e.g. duplicate operand names)
    if any(o.period == "reject" for o in result.operands) or result.coverage == "missing":
        return "period_rejected"
    if result.period_unverified or result.coverage == "unverified":
        return "unverified"
    return "passed"


def score_controls_v2(run: Path, version: str = CURRENT) -> dict:
    """Native period controls: per role, period verdicts of numeric answers plus how many are accepted in strict and
    lenient mode (on negatives every accepted numeric answer used wrong-period evidence)."""
    items, preds = _load(run)
    out = {"negative": Counter(), "positive": Counter()}
    for item_id, item in items.items():
        output, failure = _parse(preds[item_id])
        role = item["role"]
        if failure or output.answer_type != "numeric":
            out[role]["insufficient_evidence" if output and output.answer_type == "insufficient_evidence" else "other_or_invalid"] += 1
            continue
        requested = requested_years(item["question"])
        res = _validate(output, item, requested, None, strict=False, version=version)
        strict = _validate(output, item, requested, None, strict=True, version=version)
        verdict = period_verdict(res)
        out[role]["accepted_lenient"] += int(res.status == "accepted")
        out[role]["accepted_strict"] += int(strict.status == "accepted")
        if role == "positive":
            correct = finqa_value_matches(res, output, item.get("gold_answer"))
            out[role][("correct_" if correct else "wrong_") + verdict] += 1
        else:
            out[role]["answered_" + verdict] += 1
    return {k: dict(v) for k, v in out.items()}


def trust_rows(run: Path, gold: dict, review: dict | None = None, version: str = CURRENT, fiscal: dict | None = None) -> list[dict]:
    """One row per question: answer type, correctness, strict and lenient status (for tables and version transitions)."""
    items, preds = _load(run)
    rows = []
    for item_id, item in items.items():
        g = gold["questions"][item_id]
        kind = g["expected_type"]
        row = {"id": item_id, "company": item["company"], "kind": kind, "outcome": None, "correct": False,
               "strict": None, "lenient": None, "period": None, "gold_value_but_rejected": None, "strict_only_unverified": False}
        output, failure = _parse(preds[item_id])
        if failure:
            rows.append(row | {"outcome": "invalid_or_operational"})
            continue
        if output.answer_type == "insufficient_evidence":
            rows.append(row | {"outcome": "declined"})
            continue
        requested = set(g["requested_years"])
        strict = _validate(output, item, requested, item["company"], True, version, fiscal)
        lenient = _validate(output, item, requested, item["company"], False, version, fiscal)
        if kind == "numeric":
            correct = output.answer_type == "numeric" and lenient.value is not None and numeric_outcome(lenient.value, g, item["question"], version) == "correct"
            if output.answer_type == "numeric" and lenient.status != "accepted":
                loose = _validate(output, item, set(), item["company"], False, version, fiscal)    # value without period constraints
                if loose.value is not None and numeric_outcome(loose.value, g, item["question"], version) == "correct":
                    row["gold_value_but_rejected"] = period_verdict(lenient)
            row["strict_only_unverified"] = correct and strict.status != "accepted" and lenient.status == "accepted"
        elif kind == "boolean":
            correct = output.answer_type == "boolean" and output.boolean == g["gold_boolean"]
        else:
            correct = (review or {}).get(f"{run.name}::{item_id}") == "correct"
        rows.append(row | {"outcome": "answered", "correct": correct, "strict": strict.status, "lenient": lenient.status,
                           "period": period_verdict(lenient) if lenient.operands else None,
                           "issues": lenient.issues + [i for c in lenient.claims for i in c.issues]})
    return rows


def trust_table(run: Path, gold: dict, review: dict | None = None, version: str = CURRENT, fiscal: dict | None = None) -> dict:
    """Per answer type: coverage (questions with an accepted answer), correct among accepted (strict and lenient),
    unsupported (answered but rejected), declined, invalid; for gold-matching numeric answers, period rejections vs
    strict-only rejections because a period could not be placed."""
    table = {k: Counter() for k in ("numeric", "boolean", "narrative")}
    for r in trust_rows(run, gold, review, version, fiscal):
        t = table[r["kind"]]
        t["n"] += 1
        if r["outcome"] in ("invalid_or_operational", "declined"):
            t["invalid_or_operational" if r["outcome"] == "invalid_or_operational" else "declined"] += 1
            continue
        if r["gold_value_but_rejected"]:
            t["gold_value_but_rejected_" + r["gold_value_but_rejected"]] += 1
        if r["strict_only_unverified"]:
            t["gold_value_strict_only_rejection_unverified"] += 1
        for mode in ("strict", "lenient"):
            if r[mode] == "accepted":
                t[f"accepted_{mode}"] += 1
                t[f"correct_accepted_{mode}"] += int(r["correct"])
        if r["lenient"] != "accepted":
            t["unsupported"] += 1
        t["correct_any"] += int(r["correct"])
    return {k: dict(v) for k, v in table.items()}


def transitions(before: list[dict], after: list[dict]) -> dict:
    """Acceptance changes between two scoring versions of the same run, by mode and correctness."""
    b = {r["id"]: r for r in before}
    out = Counter()
    for r in after:
        old = b[r["id"]]
        for mode in ("strict", "lenient"):
            was, now = old[mode] == "accepted", r[mode] == "accepted"
            if was != now:
                out[f"{mode}_{'newly_accepted' if now else 'newly_rejected'}_{'correct' if r['correct'] else 'incorrect'}"] += 1
        if old["correct"] != r["correct"]:
            out["correctness_changed"] += 1
    return dict(out)
