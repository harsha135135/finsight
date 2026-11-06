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
