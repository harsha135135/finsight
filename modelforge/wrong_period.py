"""Wrong-period analysis: account for every outcome, then replay answers through FinSight's period validator.

Outcome classes (sum to the slice size per model):
  correct_abstention   the model abstained
  wrong_year           answered (accepted or contradicted by the FinQA validator) with at least one operand that the
                       source places in an evidence year other than the year the question asks about
  other_wrong          answered, but no operand can be placed in another year
  invalid              invalid output or operational failure

Operand years are read independently of the validator under test: for table rows, from the FinQA row's own
``[columns: ...]`` trailer by cell position; for text, the single year in the sentence holding the quoted figure.

Replay: each answer becomes a ``finsight-report-v1`` finding (period = the asked year; operands, declared periods and
program exactly as the model gave them) and is checked by ``finsight.findings.validate_finding`` against the same
evidence, with FinQA table rows re-expressed in FinSight's row format (``label | 2014: v | 2013: v``, same cells).
A rejection counts as a period catch only when the period issue is present (FinQA questions are not one of
FinSight's metrics, so a metric issue is always added and is ignored here). Positive control: the same models'
correct answers to the original questions, checked against the original year; any period rejection there is a
false rejection.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from finsight.contracts import AnswerOutput, Passage, PassageLocation
from finsight.contracts.report import EvidenceOperand, FindingOutput
from finsight.findings import validate_finding
from finsight.normalize import _squash
from modelforge import splits

YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?![\ds])")
PERIOD_ISSUE = "the source places"


def _row_parts(text: str) -> tuple[str, list[str], list[str]] | None:
    body, sep, trailer = text.partition("  [columns:")
    if not sep:
        return None
    cells = body.split(" | ")
    headers = trailer.rstrip("]").split(" | ")
    return cells[0], cells[1:], headers[1:]


def operand_year(passage: Passage, original_text: str) -> str | None:
    quoted = _squash(original_text)
    if not quoted:
        return None
    if passage.id.startswith("table_"):
        parts = _row_parts(passage.text)
        if parts is None:
            return None
        _, cells, headers = parts
        if len(cells) != len(headers):
            return None
        for cell, header in zip(cells, headers):
            if quoted == _squash(cell) or _squash(cell).endswith(quoted):
                years = YEAR.findall(header)
                return years[0] if len(years) == 1 else None
        return None
    for sentence in re.split(r"(?<=[.;])\s+", passage.text):
        if quoted in _squash(sentence):
            years = YEAR.findall(sentence)
            return years[0] if len(years) == 1 else None
    return None


def finsight_passages(item) -> tuple[dict[str, Passage], int]:
    """The item's evidence as FinSight would hold it; returns (passages, rows that could not be converted)."""
    out, unconvertible = {}, 0
    for p in item.passages:
        parts = _row_parts(p.text) if p.id.startswith("table_") else None
        if parts is None:
            out[p.id] = p
            continue
        label, cells, headers = parts
        if len(cells) != len(headers):
            unconvertible += 1
            out[p.id] = p.model_copy(update={"location": PassageLocation(table=p.location.table, row=p.location.row)})
            continue
        text = " | ".join([label] + [f"{h.strip()}: {c.strip()}" for h, c in zip(headers, cells)]) + f"  [table: {p.location.table}]"
        out[p.id] = p.model_copy(update={"text": text, "end": p.start + len(text),
                                         "location": PassageLocation(table=p.location.table, row=p.location.row)})
    return out, unconvertible


def _name(raw: str, n: int) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", raw or "")[:64]
    return name if re.match(r"^[A-Za-z]", name) else f"op{n}_{name}"[:64]


def replay(output: AnswerOutput, passages: dict[str, Passage], year: str) -> dict:
    operands = [EvidenceOperand(name=_name(op.name, n), passage_id=op.passage_id, original_text=op.original_text[:200] or "?",
                                value=str(op.value)[:40] or "?", scale=op.scale, unit=(op.unit or "unit")[:20],
                                period=(op.period or "unspecified")[:32] if len(op.period or "unspecified") >= 2 else "unspecified")
                for n, op in enumerate(output.operands)]
    renamed = {op.name: e.name for op, e in zip(output.operands, operands)}
    program = [step.model_copy(update={"args": [renamed.get(a, a) for a in step.args]}) for step in output.program]
    finding = FindingOutput(metric="finqa_question", period=f"FY{year}", status="answered", operands=operands,
                            program=program, unit=output.answer_unit, explanation=[], missing=None)
    validated = validate_finding(finding, passages)
    period_issues = [i for i in validated.issues if PERIOD_ISSUE in i]
    return {"period_caught": bool(period_issues), "period_issues": period_issues,
            "other_issues": [i for i in validated.issues if PERIOD_ISSUE not in i and "unknown metric" not in i]}


def _load(run: Path) -> dict:
    return {p["item_id"]: p for p in map(json.loads, (run / "predictions.jsonl").read_text().splitlines())}


def analyse(slice_runs: dict[str, Path], original_runs: dict[str, Path]) -> dict:
    manifest = splits.load(splits.WRONG_PERIOD_NAME)
    items = {i.id: i for i in splits.resolve_items(manifest, manifest["samples"]["val_dev_wrong_period"])}
    originals = {i.id: i for i in splits.resolve_items(splits.load("finqa-grouped-v2"), sorted({k.split("::")[0] for k in items}))}
    unconvertible_rows = sum(finsight_passages(i)[1] for i in items.values())
    report: dict = {"items": len(items), "unconvertible_table_rows": unconvertible_rows, "models": {}}
    for model, run in slice_runs.items():
        preds = _load(run)
        classes, replays, examples = Counter(), Counter(), []
        for item_id, item in items.items():
            p = preds[item_id]
            code = p["reason_code"]
            if code == "ABSTAIN_EVIDENCE_ABSENT":
                classes["correct_abstention"] += 1
                continue
            if code in ("MODEL_OUTPUT_INVALID", "OPERATIONAL"):
                classes["invalid"] += 1
                continue
            output = AnswerOutput.model_validate_json(p["raw_output"])
            asked = item.meta["shifted_to"]
            by_id = {q.id: q for q in item.passages}
            years = {operand_year(by_id[op.passage_id], op.original_text) if op.passage_id in by_id else None
                     for op in output.operands}
            other = {y for y in years if y and y != asked}
            accepted = code == "ANSWERED"
            if not other:
                classes["other_wrong"] += 1
                continue
            classes["wrong_year_" + ("accepted_by_finqa_validator" if accepted else "rejected_by_finqa_validator")] += 1
            passages, _ = finsight_passages(item)
            r = replay(output, passages, asked)
            replays["caught" if r["period_caught"] else "missed"] += 1
            if not r["period_caught"]:
                declared = [op.period for op in output.operands]
                replays["missed_with_declared_other_years" if all(YEAR.search(d or "") for d in declared) and len(declared) > 1 else "missed_other"] += 1
                if len(examples) < 3:
                    examples.append({"item": item_id, "asked": asked, "operand_years": sorted(other),
                                     "declared": declared, "issues": r["other_issues"][:2]})
        control, control_examples = Counter(), []
        opreds = _load(original_runs[model])
        for item_id, item in items.items():
            base_id = item_id.split("::")[0]
            p = opreds.get(base_id)
            if not p or not p["correct"]:
                continue
            passages, _ = finsight_passages(originals[base_id])
            r = replay(AnswerOutput.model_validate_json(p["raw_output"]), passages, item.meta["shifted_from"])
            control["falsely_rejected" if r["period_caught"] else "accepted"] += 1
            if r["period_caught"] and len(control_examples) < 3:
                control_examples.append({"item": base_id, "year": item.meta["shifted_from"], "issue": r["period_issues"][0][:200]})
        report["models"][model] = {"outcomes": dict(classes), "outcomes_total": sum(classes.values()),
                                   "replay_of_wrong_year_answers": dict(replays), "missed_examples": examples,
                                   "control_correct_original_answers": dict(control), "control_false_rejection_examples": control_examples}
    return report
