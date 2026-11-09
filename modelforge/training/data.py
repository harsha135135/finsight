"""Build supervised fine-tuning data from the frozen train split only.

Targets come from deterministic sources (FinQA gold programs aligned to cited
evidence, and abstentions for gold-evidence-removed variants). Claude outputs and
FinanceBench are never used as training data. Validation items are written
separately for evaluation loss and adapter selection; test items are never read.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from modelforge import prompts, splits
from modelforge.datasets.finqa import gold_answer_output

DATA_VERSION = "1"
ABSTAIN = {"abstain": True, "abstain_reason": "The evidence does not contain the figures needed.",
           "operands": [], "program": [], "answer": None, "answer_unit": None, "claims": []}


def _example(item, completion: str, kind: str) -> dict:
    return {"id": item.id, "kind": kind,
            "prompt": [{"role": "system", "content": prompts.SYSTEM},
                       {"role": "user", "content": prompts.user_message(item)}],
            "completion": [{"role": "assistant", "content": completion}]}


def build(manifest_name: str = "finqa-grouped", out_dir: Path = Path("artifacts/training/finqa-grouped-v1"),
          abstain_fraction: float = 0.2, seed: int = 20261106) -> dict:
    manifest = splits.load(manifest_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"data_version": DATA_VERSION, "manifest": manifest_name, "manifest_sha256": manifest["sha256"],
                    "prompt_version": prompts.PROMPT_VERSION, "prompt_sha256": prompts.fingerprint(), "seed": seed,
                    "sources": "FinQA gold programs aligned to cited evidence + gold-evidence-removed abstentions"}
    for split in ("train", "val"):
        items = splits.resolve_items(manifest, manifest["splits"][split])
        answer, abstain, skipped = [], [], 0
        for item in items:
            target = gold_answer_output(item)
            if target is None:
                skipped += 1
            else:
                answer.append(_example(item, target.model_dump_json(), "answer"))
            variant = splits.unanswerable_variant(item)
            if variant is not None:
                abstain.append(_example(variant, json.dumps(ABSTAIN), "abstain"))
        rng = random.Random(seed)
        rng.shuffle(abstain)
        abstain = abstain[: int(len(answer) * abstain_fraction)]
        rows = answer + abstain
        rng.shuffle(rows)
        path = out_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        report[split] = {"examples": len(rows), "answer": len(answer), "abstain": len(abstain),
                         "unaligned_skipped": skipped, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out_dir / "data_manifest.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


# ----------------------------------------------------------------------------- v2

DATA_VERSION_V2 = "2"
NEGATIVE_FILTER = ("drop a train variant if any automatic audit flag fires, or if a table row was removed while "
                   "two or more numeric rows remain in the table (totals and components often rebuild it); "
                   "on the 134 hand-audited validation variants this kept 27, of which 93% were verified "
                   "unanswerable (in-sample estimate; unfiltered 65%)")


def _numeric_rows(passages) -> int:
    import re
    return sum(1 for p in passages if p.id.startswith("table_") and len(re.findall(r"\d", p.text.split("  [columns")[0])) >= 3)


def keep_train_negative(item, variant) -> bool:
    from modelforge.audit import auto_flags
    if auto_flags(item, variant):
        return False
    removed_table = any(pid.startswith("table_") for pid in item.gold_passage_ids)
    return not (removed_table and _numeric_rows(variant.passages) >= 2)


def build_v2(manifest_name: str = "finqa-grouped-v2", out_dir: Path = Path("artifacts/training/finqa-grouped-v2-a20"),
             abstain_fraction: float = 0.2, seed: int = 20261106) -> dict:
    """Training data v2: filtered train negatives paired with their answerable originals; selection rows
    from val_select only (answerable items plus hand-verified negatives). val_dev and test are never read."""
    manifest = splits.load(manifest_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"data_version": DATA_VERSION_V2, "manifest": manifest_name, "manifest_sha256": manifest["sha256"],
                    "prompt_version": prompts.PROMPT_VERSION, "prompt_sha256": prompts.fingerprint(), "seed": seed,
                    "abstain_fraction": abstain_fraction, "negative_filter": NEGATIVE_FILTER,
                    "sources": "FinQA gold programs aligned to cited evidence; train negatives filtered automatically and "
                               "paired with their answerable originals; selection negatives hand-verified (audit file bound "
                               "to the manifest)"}
    rng = random.Random(seed)
    train_items = splits.resolve_items(manifest, manifest["splits"]["train"])
    answer, candidates, skipped, filtered = {}, [], 0, 0
    for item in train_items:
        target = gold_answer_output(item)
        if target is None:
            skipped += 1
            continue
        answer[item.id] = _example(item, target.model_dump_json(), "answer")
        variant = splits.unanswerable_variant(item)
        if variant is not None:
            if keep_train_negative(item, variant):
                candidates.append(variant)
            else:
                filtered += 1
    rng.shuffle(candidates)
    negatives = candidates[: int(len(answer) * abstain_fraction)]
    rows = list(answer.values()) + [_example(v, json.dumps(ABSTAIN), "abstain") for v in negatives]
    rng.shuffle(rows)
    path = out_dir / "train.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report["train"] = {"examples": len(rows), "answer": len(answer), "abstain": len(negatives),
                       "negatives_available_after_filter": len(candidates), "negatives_filtered_out": filtered,
                       "abstain_paired_with_answerable_original": all(v.id.split("::")[0] in answer for v in negatives),
                       "unaligned_skipped": skipped, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    select_items = splits.resolve_items(manifest, manifest["splits"]["val_select"] + manifest["samples"]["val_select_unanswerable"])
    val_rows, skipped = [], 0
    for item in select_items:
        if item.answerable:
            target = gold_answer_output(item)
            if target is None:
                skipped += 1
                continue
            val_rows.append(_example(item, target.model_dump_json(), "answer"))
        else:
            val_rows.append(_example(item, json.dumps(ABSTAIN), "abstain"))
    rng.shuffle(val_rows)
    path = out_dir / "val.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_rows), encoding="utf-8")
    report["val"] = {"examples": len(val_rows), "answer": sum(r["kind"] == "answer" for r in val_rows),
                     "abstain": sum(r["kind"] == "abstain" for r in val_rows), "unaligned_skipped": skipped,
                     "source": "val_select only", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out_dir / "data_manifest.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


# ----------------------------------------------------------------------------- v2b

DATA_VERSION_V2B = "2b"


def build_v2b(manifest_name: str = "finqa-grouped-v2", base_dir: Path = Path("artifacts/training/finqa-grouped-v2-a20"),
              out_dir: Path = Path("artifacts/training/finqa-grouped-v2b-p10"), per_100_answerable: int = 10,
              seed: int = 20261109) -> dict:
    """v2a's training rows unchanged, plus wrong-period negatives (splits.PERIOD_SHIFT_RULE) made from training
    questions that are in v2a as answerable examples; the selection file is v2a's, byte for byte."""
    manifest = splits.load(manifest_name)
    base_train = (base_dir / "train.jsonl").read_bytes()
    base_rows = [json.loads(line) for line in base_train.decode().splitlines() if line]
    answer_ids = {r["id"] for r in base_rows if r["kind"] == "answer"}
    items = splits.resolve_items(manifest, sorted(answer_ids))
    candidates = [v for item in items if (v := splits.period_shift_variant(item)) is not None]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = candidates[: len(answer_ids) * per_100_answerable // 100]
    rows = base_rows + [_example(v, json.dumps(ABSTAIN), "abstain") for v in chosen]
    rng.shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "train.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out_dir / "val.jsonl").write_bytes((base_dir / "val.jsonl").read_bytes())
    base_manifest = json.loads((base_dir / "data_manifest.json").read_text())
    clean = sum(r["kind"] == "abstain" for r in base_rows)
    report = {**{k: base_manifest[k] for k in ("manifest", "manifest_sha256", "prompt_version", "prompt_sha256",
                                                "negative_filter")},
              "data_version": DATA_VERSION_V2B, "seed": seed, "base_data": str(base_dir),
              "base_train_sha256": hashlib.sha256(base_train).hexdigest(),
              "period_shift": {"version": splits.PERIOD_SHIFT_VERSION, "rule": splits.PERIOD_SHIFT_RULE,
                               "per_100_answerable": per_100_answerable, "eligible": len(candidates),
                               "chosen": len(chosen), "ids": sorted(v.id for v in chosen)},
              "sources": base_manifest["sources"] + "; plus wrong-period negatives paired with their answerable originals",
              "train": {"examples": len(rows), "answer": len(answer_ids), "abstain_clean": clean,
                        "abstain_period_shift": len(chosen), "abstain": clean + len(chosen),
                        "abstain_paired_with_answerable_original": all(v.id.split("::")[0] in answer_ids for v in chosen),
                        "base_rows_unchanged": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
              "val": {**base_manifest["val"], "copied_from": str(base_dir / "val.jsonl")}}
    (out_dir / "data_manifest.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report
