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
from decimal import Decimal, InvalidOperation
from pathlib import Path

from finsight.contracts import Passage
from finsight.normalize import figures_in
from modelforge.datasets import finqa
from modelforge.items import EvalItem

BUILDER_VERSION = "3"
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


def _needed_literals(program: str) -> tuple[set[tuple[str, str | None]], bool]:
    """Figures a FinQA program reads from the evidence, as (written value, unit) pairs.

    Constants (``const_*``), step references and ``none`` are not evidence. A percent
    literal such as ``7.81%`` must appear in the evidence as 7.81 with unit "%".
    Returns the literals and whether the program uses table operations.
    """
    needed: set[tuple[str, str | None]] = set()
    uses_rows = False
    for op, args in re.findall(r"([a-z_]+)\(([^()]*)\)", program):
        if op.startswith("table_"):
            uses_rows = True
            continue
        for arg in (a.strip() for a in args.split(",")):
            if arg.startswith(("#", "const_")) or arg == "none":
                continue
            percent = arg.endswith("%")
            try:
                value = Decimal(arg.rstrip("%").replace(",", ""))
            except InvalidOperation:
                continue
            needed.add((format(value.normalize(), "f"), "%" if percent else None))
    return needed, uses_rows


def unanswerable_variant(item: EvalItem) -> EvalItem | None:
    if not item.gold_passage_ids or not item.gold_steps:
        return None
    remaining = [p for p in item.passages if p.id not in set(item.gold_passage_ids)]
    needed, uses_rows = _needed_literals(item.meta.get("program", ""))
    figures = [(format(f.value.normalize(), "f"), f.unit) for p in remaining for f in figures_in(p.text)]
    values = {v for v, _ in figures}
    percents = {v for v, unit in figures if unit == "%"}
    still_present = all((v in percents) if unit == "%" else (v in values) for v, unit in needed)
    removed_gold_row = uses_rows and any(pid.startswith("table_") for pid in item.gold_passage_ids)
    if not needed and not removed_gold_row:
        return None
    if still_present and not removed_gold_row:
        return None
    gold_rows = {pid for pid in item.gold_passage_ids if pid.startswith("table_")}
    return item.model_copy(update={
        "id": f"{item.id}::unanswerable", "passages": remaining, "answerable": False, "gold_answer": None,
        "gold_steps": None, "gold_passage_ids": [],
        "rows": {k: v for k, v in item.rows.items() if k not in gold_rows},
        "meta": {**item.meta, "variant": "gold_passages_removed"},
    })


PERIOD_SHIFT_VERSION = "2"   # v2: "2019s" (FinQA's encoding of an apostrophe, as in "management 2019s") is not a year
PERIOD_SHIFT_RULE = ("an answerable question that names a year the evidence covers; that year is moved to one the "
                     "evidence never mentions (2-4 years before or after every year in the question and evidence, "
                     "direction seeded by the item id; absent as a 4-digit year and as FY/'YY forms); forecasting or "
                     "hypothetical questions are skipped because they stay answerable for an absent year")
_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?![\ds])")
_HYPOTHETICAL = re.compile(r"\b(expect\w*|project\w*|forecast\w*|estimat\w*|if|assum\w*|would|will|next|trend|"
                           r"continu\w*|predict\w*|anticipat\w*)\b", re.I)


def period_shift_variant(item: EvalItem) -> EvalItem | None:
    """Wrong-period negative: the question asks about a year the evidence does not cover (see PERIOD_SHIFT_RULE)."""
    if not item.answerable or not item.gold_steps or _HYPOTHETICAL.search(item.question):
        return None
    text = " ".join(p.text for p in item.passages).lower()
    evidence = {int(y) for y in _YEAR.findall(text)}
    asked = sorted({int(y) for y in _YEAR.findall(item.question)})
    covered = [y for y in asked if y in evidence]
    if not covered:
        return None
    rng = random.Random(f"period-shift-v{PERIOD_SHIFT_VERSION}:{item.id}")
    source = rng.choice(covered)
    low, high = min(evidence | set(asked)), max(evidence | set(asked))
    options = [y for y in (*(low - k for k in (2, 3, 4)), *(high + k for k in (2, 3, 4))) if 1990 <= y <= 2035]
    rng.shuffle(options)
    for target in options:
        short = f"{target % 100:02d}"
        if re.search(rf"(?<!\d){target}(?!\d)", text) or re.search(rf"(?:fy|')\s*{short}(?!\d)", text):
            continue
        return item.model_copy(update={
            "id": f"{item.id}::period_shift", "question": re.sub(rf"(?<!\d){source}(?![\ds])", str(target), item.question),
            "answerable": False, "gold_answer": None, "gold_steps": None, "gold_passage_ids": [],
            "meta": {**item.meta, "variant": "period_shift", "shifted_from": str(source), "shifted_to": str(target),
                     "period_shift_version": PERIOD_SHIFT_VERSION}})
    return None


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


V2_NAME = "finqa-grouped-v2"


def _check_bound_audit(manifest: dict) -> None:
    """A manifest built from audited labels records the audit file's hash; refuse to resolve if it changed."""
    audit = (manifest.get("variant_generator") or {}).get("audit")
    if audit:
        path = Path(audit["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != audit["sha256"]:
            raise ValueError(f"manifest {manifest['name']} is bound to {path} sha256 {audit['sha256'][:12]}, "
                             f"found {actual and actual[:12]}; the labels changed after the manifest was frozen")


def build_v2(audit_path: Path = Path("evals/labels/finqa-unanswerable-audit-v1.json")) -> dict:
    """finqa-grouped-v2: validation split by filing group into val_select and val_dev, negatives from audited labels.

    train and test are the finqa-grouped v1 splits. Every unanswerable example in a v2 selection
    was reviewed by hand and verified (see modelforge/audit.py); the manifest binds the audit file's
    hash and the variant generator, so later label edits cannot silently change it.
    """
    parent = load("finqa-grouped")
    audit = json.loads(audit_path.read_text())
    undecided = [k for k, v in audit["candidates"].items() if v["decision"] is None]
    if undecided:
        raise ValueError(f"{len(undecided)} audit candidates have no decision, e.g. {undecided[:3]}")
    select_groups, dev_groups = set(audit["partition"]["val_select_groups"]), set(audit["partition"]["val_dev_groups"])
    val = resolve_items(parent, parent["splits"]["val"])
    verified = {k: v["pool"] for k, v in audit["candidates"].items() if v["decision"] == "verified_unanswerable"}
    val_select = sorted(i.id for i in val if i.group in select_groups)
    val_dev = sorted(i.id for i in val if i.group in dev_groups)
    dev_negatives = sorted(k for k, pool in verified.items() if pool == "val_dev")
    select_negatives = sorted(k for k, pool in verified.items() if pool == "val_select")
    aligned_dev = sorted(i.id for i in val if i.group in dev_groups and finqa.gold_answer_output(i) is not None)
    manifest = {
        "name": V2_NAME, "dataset": "finqa", "revision": parent["revision"], "builder_version": "4",
        "parent": {"name": parent["name"], "sha256": parent["sha256"]},
        "rules": build_v2.__doc__.strip(),
        "variant_generator": {"name": "gold_passages_removed", "builder_version": BUILDER_VERSION,
                              "audit": {"path": str(audit_path), "version": audit["version"],
                                        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()}},
        "splits": {"train": parent["splits"]["train"], "val_select": val_select, "val_dev": val_dev,
                   "test": parent["splits"]["test"]},
        "samples": {"val_dev_eval": sorted(val_dev + dev_negatives), "val_dev_unanswerable": dev_negatives,
                    "val_select_unanswerable": select_negatives},
        "slices": {"val_dev_contract_answerable": aligned_dev},
        "report": {"val_groups": {"select": len(select_groups), "dev": len(dev_groups)},
                   "sizes": {"val_select": len(val_select), "val_dev": len(val_dev),
                             "val_dev_unanswerable": len(dev_negatives), "val_select_unanswerable": len(select_negatives)},
                   "audit_decisions": {d: sum(v["decision"] == d for v in audit["candidates"].values())
                                       for d in ("verified_unanswerable", "derivable", "ambiguous")},
                   "development_note": "val_dev guides choices and is development data; the test split stays reserved"},
        "seed": audit["partition"]["seed"],
    }
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def resolve_items(manifest: dict, ids: Iterable[str]) -> list[EvalItem]:
    """Rebuild EvalItems (including unanswerable variants) for manifest ids."""
    _check_bound_audit(manifest)
    wanted = list(ids)
    base = {i.split("::")[0] for i in wanted}
    wanted_set = set(wanted)
    found: dict[str, EvalItem] = {}
    for split in finqa.SPLITS:
        for item in finqa.iter_items(split, manifest["revision"]):
            if item.id in base:
                found[item.id] = item
                if (variant := unanswerable_variant(item)) is not None:
                    found[variant.id] = variant
                if f"{item.id}::period_shift" in wanted_set and (shifted := period_shift_variant(item)) is not None:
                    found[shifted.id] = shifted
    missing = [i for i in wanted if i not in found]
    if missing:
        raise KeyError(f"{len(missing)} manifest ids could not be rebuilt, e.g. {missing[:3]}")
    return [found[i] for i in wanted]


COMPARISON_NAME = "finqa-grouped-comparison"


def build_comparison(parent_name: str = "finqa-grouped", answerable_n: int = 250, unanswerable_n: int = 50,
                     seed: int = 20261106) -> dict:
    """Shared evaluation sets for comparing models, derived deterministically from the frozen parent.

    Kept in its own manifest (recording the parent's sha256) so the parent and the
    training data that references it stay unchanged. Built and committed before any
    model is run on test items.
    """
    parent = load(parent_name)
    samples = parent["samples"]
    answerable = _sample(samples["test_300"], answerable_n, seed)
    unanswerable = _sample(samples["unanswerable_test_100"], unanswerable_n, seed)
    manifest = {
        "name": COMPARISON_NAME, "dataset": parent["dataset"], "revision": parent["revision"],
        "builder_version": BUILDER_VERSION, "parent": {"name": parent_name, "sha256": parent["sha256"]},
        "rules": ("test_shared_300: every model (baseline Qwen, tuned Qwen, Claude) runs these same items; "
                  f"{answerable_n} answerable items sampled from test_300 plus {unanswerable_n} declared unanswerable "
                  "items sampled from unanswerable_test_100. test_expanded_1247: every official test item plus "
                  "unanswerable_test_100, for the local models only; reported separately and never compared as an "
                  "aggregate against test_shared_300. val_shared_250: val_200 plus unanswerable_val_50."),
        "splits": {},
        "samples": {
            "test_shared_300": sorted(answerable + unanswerable),
            "test_expanded_1247": sorted(parent["splits"]["test"] + samples["unanswerable_test_100"]),
            "val_shared_250": sorted(samples["val_200"] + samples["unanswerable_val_50"]),
        },
        "declared_unanswerable": {
            "test_shared_300": sorted(unanswerable),
            "test_expanded_1247": sorted(samples["unanswerable_test_100"]),
            "val_shared_250": sorted(samples["unanswerable_val_50"]),
        },
        "seed": seed,
    }
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def _family(item_id: str) -> str:
    return item_id.split("::")[0]


def verify(manifest: dict) -> dict:
    """Recompute leakage checks for a grouped manifest, or parentage checks for a comparison manifest."""
    if manifest["name"] == V2_NAME:
        _check_bound_audit(manifest)
        names = ("train", "val_select", "val_dev", "test")
        items = {s: resolve_items(manifest, manifest["splits"][s]) for s in names}
        groups = {s: {i.group for i in items[s]} for s in names}
        families = {s: {_family(i) for i in manifest["splits"][s]} for s in names}
        for sample in ("val_dev_eval", "val_dev_unanswerable"):
            families["val_dev"] |= {_family(i) for i in manifest["samples"][sample]}
        families["val_select"] |= {_family(i) for i in manifest["samples"]["val_select_unanswerable"]}
        pairs = [(a, b) for k, a in enumerate(names) for b in names[k + 1:]]
        audit = json.loads(Path(manifest["variant_generator"]["audit"]["path"]).read_text())["candidates"]
        negatives = manifest["samples"]["val_dev_unanswerable"] + manifest["samples"]["val_select_unanswerable"]
        return {"name": manifest["name"],
                "shared_groups": {f"{a}/{b}": len(groups[a] & groups[b]) for a, b in pairs},
                "shared_families": {f"{a}/{b}": len(families[a] & families[b]) for a, b in pairs},
                "negatives_all_verified": all(audit.get(i, {}).get("decision") == "verified_unanswerable" for i in negatives),
                "sizes": manifest["report"]["sizes"]}
    if manifest["name"] == COMPARISON_NAME:
        parent = load(manifest["parent"]["name"])
        expanded = set(manifest["samples"]["test_expanded_1247"])
        return {"name": manifest["name"], "parent_sha256_matches": parent["sha256"] == manifest["parent"]["sha256"],
                "shared_within_expanded": set(manifest["samples"]["test_shared_300"]) <= expanded,
                "sizes": {k: len(v) for k, v in manifest["samples"].items()},
                "declared_unanswerable": {k: len(v) for k, v in manifest["declared_unanswerable"].items()},
                "declared_match_variant_ids": all(i.endswith("::unanswerable") for v in manifest["declared_unanswerable"].values() for i in v)}
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


WRONG_PERIOD_NAME = "finqa-grouped-v2-wrong-period"


def build_wrong_period(base_name: str = V2_NAME) -> dict:
    """Diagnostic slice (secondary): wrong-period versions of val_dev's answerable questions. Never used for
    training or selection; the main development metric stays val_dev_eval."""
    base = load(base_name)
    items = resolve_items(base, base["splits"]["val_dev"])
    ids = sorted(v.id for i in items if (v := period_shift_variant(i)) is not None)
    manifest = {"name": WRONG_PERIOD_NAME, "dataset": base["dataset"], "revision": base["revision"],
                "builder_version": BUILDER_VERSION, "base_manifest": base_name, "base_sha256": base["sha256"],
                "variant_generator": {"period_shift": PERIOD_SHIFT_VERSION, "rule": PERIOD_SHIFT_RULE},
                "rules": "Secondary diagnostic: every val_dev answerable item the period-shift rule applies to.",
                "samples": {"val_dev_wrong_period": ids}}
    manifest["sha256"] = canonical_hash(manifest)
    return manifest
