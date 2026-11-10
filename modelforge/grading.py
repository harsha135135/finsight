"""Blind grading of narrative answers (procedure frozen 2026-11-10; rules in docs/evaluation-rules.md).

* Every non-declined narrative answer from every model being compared goes on one sheet, shuffled with a fixed seed,
  with model and run hidden behind blind ids. The grader sees the question, the gold answer and the answer's claims
  (and figures, if the model answered a narrative question with numbers), never the model, the validation status or
  any other model's grade.
* Decoys: previously graded development answers (with their recorded verdicts) are mixed in, about one in five, so the
  grader's consistency is measured on the same sheet.
* Verdicts use the frozen rubric (RUBRIC). "partial" never counts as correct in a headline number; it is reported
  separately.
* The sheet is written as CSV for a person to fill in. Independent human review is preferred for headline narrative
  results; if a Claude model grades instead, the results say so.
* The verdicts are committed before the key is applied; ``apply`` maps blind ids back and reports decoy agreement.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

RUBRIC = ("correct: conclusion and key facts match the gold; partial: right direction but incomplete or one error "
          "(e.g. right segment with a different-basis figure, two of three items, a bare answer where the gold also gives "
          "facts); incorrect: wrong or no conclusion, or a figure that does not answer a which/what/why question")
VERDICTS = ("correct", "partial", "incorrect")


def blind_sheet(runs: list[Path], gold: dict, answers: dict[str, str], *, seed: int, decoys: list[dict] | None = None,
                decoy_share: float = 0.2) -> tuple[list[dict], dict]:
    """runs: run folders to grade; answers: question id -> gold answer text; decoys: entries from an earlier sheet with
    a recorded ``verdict``. Returns (sheet rows, key: blind id -> {"key": "run::item", "decoy": bool, "verdict"?})."""
    from modelforge.qa_score import review_sheet

    entries = review_sheet(runs, gold, seed=seed)
    rng = random.Random(seed)
    pool = list(decoys or [])
    n_decoys = min(len(pool), round(len(entries) * decoy_share / (1 - decoy_share))) if pool else 0
    chosen = rng.sample(pool, n_decoys) if n_decoys else []
    rows = [dict(e, decoy=False) for e in entries] + [dict(d, decoy=True) for d in chosen]
    rng.shuffle(rows)
    sheet, key = [], {}
    for n, row in enumerate(rows, 1):
        blind = f"G{n:03d}"
        key[blind] = {"key": row["key"], "decoy": row["decoy"], **({"verdict": row["verdict"]} if row["decoy"] else {})}
        sheet.append({"blind_id": blind, "question": row["question"], "gold": answers[row["gold_id"]],
                      "answer_type": row["answer_type"], "boolean": row.get("boolean"),
                      "claims": " / ".join(c or "" for c in row["claims"]), "figures": ", ".join(row.get("numeric") or [])})
    return sheet, key


def write_csv(sheet: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "verdict", "question", "gold", "answer_type", "boolean", "claims", "figures"])
        writer.writeheader()
        for row in sheet:
            writer.writerow({**row, "verdict": ""})


def read_csv(path: Path) -> dict[str, str]:
    with path.open() as handle:
        verdicts = {r["blind_id"]: r["verdict"].strip().lower() for r in csv.DictReader(handle)}
    bad = {k: v for k, v in verdicts.items() if v not in VERDICTS}
    if bad:
        raise ValueError(f"verdicts must be one of {VERDICTS}; got {dict(list(bad.items())[:5])}")
    return verdicts


def apply(verdicts: dict[str, str], key: dict, grader: str) -> dict:
    """Map blind verdicts back to run::item and measure agreement on decoys."""
    missing = sorted(set(key) - set(verdicts))
    if missing:
        raise ValueError(f"{len(missing)} sheet rows have no verdict, e.g. {missing[:3]}")
    graded = {k["key"]: verdicts[b] for b, k in key.items() if not k["decoy"]}
    decoys = [(verdicts[b], k["verdict"]) for b, k in key.items() if k["decoy"]]
    return {"grader": grader, "rubric": RUBRIC, "verdicts": graded,
            "decoy_agreement": f"{sum(a == b for a, b in decoys)}/{len(decoys)}",
            "decoy_disagreements": [(b, k["key"], k["verdict"], verdicts[b]) for b, k in key.items()
                                    if k["decoy"] and verdicts[b] != k["verdict"]],
            "claude_grades_claude_disclosure": "claude" in grader.lower()}


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
