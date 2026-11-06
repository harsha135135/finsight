"""Leakage-controlled splits and frozen, hashed manifests.

Rules for ``finqa-grouped`` (reported under its own name, never as the official split):

* test = the official FinQA test split, unchanged;
* val  = official dev minus any filing group present in test;
* train = official train minus any filing group present in val or test, minus any
  item sharing a substantial passage (normalized text, >= 40 chars) with val/test.

Unanswerable variants remove an item's gold passages and keep it only if at least
one figure the gold program needs no longer appears anywhere in the evidence.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

from finsight.contracts import Passage
from finsight.normalize import figures_in
from modelforge.datasets import finqa
from modelforge.items import EvalItem

BUILDER_VERSION = "2"
MIN_DEDUP_CHARS = 40
MANIFEST_DIR = Path("evals/manifests")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def passage_hashes(item: EvalItem) -> set[str]:
    return {hashlib.sha256(_norm(p.text).encode()).hexdigest()
            for p in item.passages if len(_norm(p.text)) >= MIN_DEDUP_CHARS}


def canonical_hash(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sample(ids: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(ids), min(n, len(ids))))


def unanswerable_variant(item: EvalItem) -> EvalItem | None:
    if not item.gold_passage_ids or not item.gold_steps:
        return None
    remaining = [p for p in item.passages if p.id not in set(item.gold_passage_ids)]
    needed = {a for s in item.gold_steps for a in s.args if not a.startswith(("#", "row:"))}
    # Table operations need their gold row; removing it makes the item unanswerable.
    uses_rows = any(a.startswith("row:") for s in item.gold_steps for a in s.args)
    present = {format(f.value.normalize(), "f") for p in remaining for f in figures_in(p.text)}
    if not uses_rows and needed and needed <= present:
        return None
    gold_rows = {pid for pid in item.gold_passage_ids if pid.startswith("table_")}
    return item.model_copy(update={
        "id": f"{item.id}::unanswerable", "passages": remaining, "answerable": False, "gold_answer": None,
        "gold_steps": None, "gold_passage_ids": [],
        "rows": {k: v for k, v in item.rows.items() if k not in gold_rows},
        "meta": {**item.meta, "variant": "gold_passages_removed"},
    })


def build_finqa(revision: str = finqa.REVISION, seed: int = 20261106) -> list[dict]:
    items = {split: list(finqa.iter_items(split, revision)) for split in finqa.SPLITS}
    official = {
        "name": "finqa-official", "dataset": "finqa", "revision": revision, "builder_version": BUILDER_VERSION,
        "rules": "Official FinQA splits, unchanged. Filing groups overlap across splits (see report).",
        "splits": {"train": sorted(i.id for i in items["train"]), "val": sorted(i.id for i in items["dev"]),
                   "test": sorted(i.id for i in items["test"])},
    }
    groups = {s: {i.group for i in v} for s, v in items.items()}
    official["report"] = {
        "groups": {s: len(g) for s, g in groups.items()},
        "test_groups_also_in_train": len(groups["test"] & groups["train"]),
        "test_groups_also_in_dev": len(groups["test"] & groups["dev"]),
    }

    test_hashes = set().union(*(passage_hashes(i) for i in items["test"]))
    val = [i for i in items["dev"] if i.group not in groups["test"]]
    val_groups = {i.group for i in val}
    val_hashes = set().union(*(passage_hashes(i) for i in val))
    train_group_ok = [i for i in items["train"] if i.group not in groups["test"] and i.group not in val_groups]
    train = [i for i in train_group_ok if not (passage_hashes(i) & (test_hashes | val_hashes))]
    test = items["test"]
    train_companies = {i.company for i in train}

    aligned = {i.id for i in val + test if finqa.gold_answer_output(i) is not None}
    unanswerable_val = [u.id for i in val if (u := unanswerable_variant(i))]
    unanswerable_test = [u.id for i in test if (u := unanswerable_variant(i))]
    grouped = {
        "name": "finqa-grouped", "dataset": "finqa", "revision": revision, "builder_version": BUILDER_VERSION,
        "rules": __doc__.strip(),
        "splits": {"train": sorted(i.id for i in train), "val": sorted(i.id for i in val),
                   "test": sorted(i.id for i in test)},
        "samples": {
            "val_200": _sample([i.id for i in val], 200, seed),
            "test_300": _sample([i.id for i in test], 300, seed),
            "unanswerable_val_50": _sample(unanswerable_val, 50, seed),
            "unanswerable_test_100": _sample(unanswerable_test, 100, seed),
        },
        "slices": {
            "test_company_unseen_in_train": sorted(i.id for i in test if i.company not in train_companies),
            # Items whose gold program can be expressed with figures quoted from the evidence.
            "val_contract_answerable": sorted(i.id for i in val if i.id in aligned),
            "test_contract_answerable": sorted(i.id for i in test if i.id in aligned),
        },
        "report": {
            "train_dropped_shared_group": len(items["train"]) - len(train_group_ok),
            "train_dropped_shared_passage": len(train_group_ok) - len(train),
            "val_dropped_shared_group": len(items["dev"]) - len(val),
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "unanswerable_available": {"val": len(unanswerable_val), "test": len(unanswerable_test)},
            "test_companies_unseen_in_train": len({i.company for i in test} - train_companies),
            "contract_answerable": {"val": sum(i.id in aligned for i in val), "test": sum(i.id in aligned for i in test)},
        },
        "seed": seed,
    }
    for manifest in (official, grouped):
        manifest["sha256"] = canonical_hash(manifest)
    return [official, grouped]


def write(manifests: Iterable[dict], directory: Path = MANIFEST_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for manifest in manifests:
        path = directory / f"{manifest['name']}.json"
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def load(name: str, directory: Path = MANIFEST_DIR) -> dict:
    manifest = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if canonical_hash(manifest) != manifest["sha256"]:
        raise ValueError(f"manifest {name} does not match its recorded sha256; it was modified after freezing")
    return manifest


def resolve_items(manifest: dict, ids: Iterable[str]) -> list[EvalItem]:
    """Rebuild EvalItems (including unanswerable variants) for manifest ids."""
    wanted = list(ids)
    base = {i.split("::")[0] for i in wanted}
    found: dict[str, EvalItem] = {}
    for split in finqa.SPLITS:
        for item in finqa.iter_items(split, manifest["revision"]):
            if item.id in base:
                found[item.id] = item
                if (variant := unanswerable_variant(item)) is not None:
                    found[variant.id] = variant
    missing = [i for i in wanted if i not in found]
    if missing:
        raise KeyError(f"{len(missing)} manifest ids could not be rebuilt, e.g. {missing[:3]}")
    return [found[i] for i in wanted]


def verify(manifest: dict) -> dict:
    """Recompute leakage checks for a grouped manifest."""
    if manifest["name"] == "finqa-official":
        return {"name": manifest["name"], "checked": "hash only (official split is not leakage-controlled)"}
    split_items = {s: resolve_items(manifest, ids) for s, ids in manifest["splits"].items()}
    groups = {s: {i.group for i in v} for s, v in split_items.items()}
    hashes = {s: set().union(*(passage_hashes(i) for i in v)) for s, v in split_items.items()}
    return {
        "name": manifest["name"],
        "shared_groups": {f"{a}/{b}": len(groups[a] & groups[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "shared_passages": {f"{a}/{b}": len(hashes[a] & hashes[b]) for a, b in (("train", "val"), ("train", "test"))},
    }


def passages_for_prompt(passages: list[Passage]) -> str:
    return "\n".join(f"[{p.id}] {p.text}" for p in passages)
