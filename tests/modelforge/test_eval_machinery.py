"""Evaluation machinery for the sealed runs, tested on synthetic fixtures (no held-out data is read)."""

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelforge import fiscal
from modelforge.datasets import financebench as fb


# ----------------------------------------------------------------------------- held-out accessor

def _manifest():
    return {"samples": {"dev": ["d1"], "held_out": ["h1", "h2"]}, "companies": {"dev": ["DevCo"], "held_out": ["HeldCo"]}}


QUESTIONS = [{"financebench_id": "d1", "company": "DevCo"}, {"financebench_id": "h1", "company": "HeldCo"},
             {"financebench_id": "h2", "company": "HeldCo"}]


def test_held_out_is_sealed_without_authorization(tmp_path):
    with pytest.raises(PermissionError, match="sealed"):
        fb.held_out_questions(_manifest(), tmp_path / "missing.json", questions=QUESTIONS)


def test_held_out_opens_only_for_the_authorized_freeze(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"frozen": true}')
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"split": "held_out", "authorized_by": "user, in chat, 2026-11-10", "freeze": str(freeze),
                                "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest()}))
    held = fb.held_out_questions(_manifest(), auth, require_committed=False, questions=QUESTIONS)
    assert [q["financebench_id"] for q in held] == ["h1", "h2"]
    freeze.write_text('{"frozen": false}')                       # changed after authorization
    with pytest.raises(PermissionError, match="changed"):
        fb.held_out_questions(_manifest(), auth, require_committed=False, questions=QUESTIONS)


def test_held_out_requires_a_committed_authorization(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"split": "held_out", "authorized_by": "x", "freeze": str(freeze),
                                "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest()}))
    with pytest.raises(PermissionError, match="committed"):
        fb.held_out_questions(_manifest(), auth, questions=QUESTIONS)


# ----------------------------------------------------------------------------- fiscal-year derivation

def test_cover_dates_and_52_53_week_years():
    text = "ANNUAL REPORT ... For the fiscal year ended January 28, 2023 ... for the year ended Feb. 1, 2020"
    assert fiscal.cover_dates(text) == [date(2023, 1, 28), date(2020, 2, 1)]
    assert fiscal.fiscal_year_end([date(2023, 1, 28), date(2020, 2, 1), date(2019, 2, 2)]) == (1, 28)
    assert fiscal.fiscal_year_end([date(2022, 12, 2), date(2021, 11, 27)]) == (11, 27)      # wraps nothing, 5 days
    assert fiscal.fiscal_year_end([date(2022, 12, 31), date(2023, 1, 2)]) == (12, 31)        # wraps the year end
    with pytest.raises(ValueError):
        fiscal.fiscal_year_end([date(2022, 6, 30), date(2022, 12, 31)])


def test_fiscal_year_rule_matches_the_validator():
    assert fiscal.fiscal_year(date(2023, 2, 2), (1, 28)) == 2023
    assert fiscal.fiscal_year(date(2022, 6, 30), (6, 30)) == 2022 and fiscal.fiscal_year(date(2022, 7, 31), (6, 30)) == 2023
    assert fiscal.fiscal_year(date(2022, 9, 30), None) == 2022


def test_derived_dev_fiscal_years_agree_with_the_hand_file():
    derived = json.loads(Path("evals/financebench/fiscal-years-dev-derived-v1.json").read_text())
    assert all(derived["agreement_with_hand_file"].values()) and len(derived["agreement_with_hand_file"]) == 16


# ----------------------------------------------------------------------------- label-audit denominators

def test_final_denominators_need_every_decision_and_keep_only_verified_negatives():
    from modelforge.audit import final_denominators
    audit = {"version": "t", "candidates": {"a::u": {"decision": "verified_unanswerable"}, "b::u": {"decision": "derivable"},
                                            "c::u": {"decision": "ambiguous"}}}
    d = final_denominators(audit, ["a", "b", "c"])
    assert d["items"] == ["a", "b", "c", "a::u"] and d["excluded"] == ["b::u", "c::u"]
    audit["candidates"]["d::u"] = {"decision": None}
    with pytest.raises(ValueError):
        final_denominators(audit, ["a"])


# ----------------------------------------------------------------------------- Claude batch: fixed settings

class _Client:
    def __init__(self, tokens):
        self.messages = SimpleNamespace(count_tokens=lambda **kw: SimpleNamespace(input_tokens=tokens))


def _items_file(tmp_path, n=3):
    passage = {"id": "p1", "document_id": "d", "section": "s", "location": {}, "start": 0, "end": 5, "text": "hello"}
    path = tmp_path / "items.jsonl"
    path.write_text("".join(json.dumps({"id": f"q{i}", "question": "What?", "passages": [passage]}) + "\n" for i in range(n)))
    return path


def test_batch_plan_keeps_max_tokens_fixed_and_checks_the_cap(tmp_path, monkeypatch):
    from modelforge import claude_qa_batch as B
    monkeypatch.setattr(B, "ROOT", tmp_path / "batches")
    items = _items_file(tmp_path)
    inputs = {"full": {"path": str(items), "sha256": hashlib.sha256(items.read_bytes()).hexdigest()}}
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    B.plan("t", inputs=inputs, freeze=freeze, run_prefix="x", label="t")
    out = B.count("t", _Client(10_000), run_cap_usd=Decimal("0.25"))
    assert out["max_tokens"] == 5600 and Decimal(out["worst_case_usd"]) <= Decimal("0.25")
    B.plan("t2", inputs=inputs, freeze=freeze)
    with pytest.raises(ValueError, match="exceeds the run cap"):
        B.count("t2", _Client(10_000), run_cap_usd=Decimal("0.001"))
    items.write_text("changed\n")
    with pytest.raises(ValueError, match="sha256"):
        B.plan("t3", inputs=inputs, freeze=freeze)


# ----------------------------------------------------------------------------- narrative grading

def test_blind_sheet_hides_models_mixes_decoys_and_requires_every_verdict(tmp_path):
    from modelforge import grading
    gold = {"questions": {"q1": {"expected_type": "narrative"}, "q2": {"expected_type": "narrative"}}}
    for name in ("run-a", "run-b"):
        run = tmp_path / name
        run.mkdir()
        (run / "items.jsonl").write_text("".join(json.dumps({"id": q, "question": f"Why {q}?", "passages": [], "company": "C"}) + "\n"
                                                 for q in ("q1", "q2")))
        out = {"answer_type": "narrative", "claims": [{"text": f"{name} says so", "passage_ids": ["p1"]}], "operands": [], "program": [],
               "boolean": None, "answer_unit": None, "missing": None}
        (run / "predictions.jsonl").write_text("".join(json.dumps({"item_id": q, "status": "ok", "text": json.dumps(out)}) + "\n"
                                                       for q in ("q1", "q2")))
    decoys = [{"key": f"old::d{i}", "question": "Old?", "gold_id": "q1", "answer_type": "narrative", "boolean": None,
               "claims": ["old"], "numeric": [], "verdict": "partial"} for i in range(5)]
    sheet, key = grading.blind_sheet([tmp_path / "run-a", tmp_path / "run-b"], gold, {"q1": "A", "q2": "B"}, seed=1, decoys=decoys)
    assert len(sheet) == 5 and sum(k["decoy"] for k in key.values()) == 1          # 4 answers + 1 decoy (one in five)
    assert all("run-" not in json.dumps(row) or "says so" in row["claims"] for row in sheet)   # no run names outside claims
    grading.write_csv(sheet, tmp_path / "sheet.csv")
    assert (tmp_path / "sheet.csv").read_text().startswith("blind_id,verdict,")
    verdicts = {row["blind_id"]: ("partial" if key[row["blind_id"]]["decoy"] else "correct") for row in sheet}
    result = grading.apply(verdicts, key, grader="human reviewer")
    assert result["decoy_agreement"] == "1/1" and len(result["verdicts"]) == 4 and not result["claude_grades_claude_disclosure"]
    with pytest.raises(ValueError):
        grading.apply({}, key, grader="x")
