"""Audit of synthetic unanswerable examples (validation pool), recorded per example.

A synthetic unanswerable variant removes an item's gold passages. Removing a needed
literal does not prove the answer cannot be derived another way, so every retained
negative evaluation example gets a recorded manual review. Automatic flags, computed
from the gold program and the remaining passages only (never from model outputs),
order the review and are stored next to each decision:

* the final answer, or an intermediate step value, appears in the remaining evidence;
* a needed literal appears in the remaining evidence in another format
  (3,460 / 3.46 billion / a percentage);
* the label of a removed table row reappears in another remaining table row.

Decisions: ``verified_unanswerable`` (kept as a negative), ``derivable`` (excluded;
the reason says how) or ``ambiguous`` (excluded).
"""

from __future__ import annotations

import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from finsight.calc import CalcError, execute
from finsight.normalize import figures_in
from modelforge import splits
from modelforge.items import EvalItem

AUDIT_VERSION = "finqa-unanswerable-audit-v1"
DEFAULT_PATH = Path("evals/labels/finqa-unanswerable-audit-v1.json")
DECISIONS = ("verified_unanswerable", "derivable", "ambiguous")
PARTITION_SEED = 20261107
DEV_CANDIDATES, SELECT_CANDIDATES = 60, 30
_STOP = set("what is the of in to for a an and or by was were be as at from on with how much did does do than "
            "that this these those its it their there total percent percentage change ratio average".split())


def partition_val_groups(groups: list[str], seed: int = PARTITION_SEED) -> tuple[list[str], list[str]]:
    """Split validation filing groups in half: (val_select, val_dev). Families stay with their group."""
    ordered = sorted(set(groups))
    random.Random(seed).shuffle(ordered)
    half = len(ordered) // 2
    return sorted(ordered[:half]), sorted(ordered[half:])


def _values(figures) -> list[Decimal]:
    out = []
    for f in figures:
        for v in {f.value, f.base_value}:
            out.append(v)
            for k in (3, 6, 9):  # a table "in millions" figure vs a prose "billion" figure
                out.append(f.base_value / Decimal(10) ** k)
    return out


def _close(a: Decimal, b: Decimal, rel: Decimal = Decimal("0.005")) -> bool:
    if b == 0:
        return a == 0
    return abs(a - b) <= abs(b) * rel


def _present(target: Decimal, passages) -> list[str]:
    hits = []
    for p in passages:
        candidates = _values(figures_in(p.text))
        # Absolute values: a parenthesised "(6,826)" parses as negative but is the same reported figure.
        if any(_close(abs(v), abs(target)) or _close(abs(v), abs(target) * 100) for v in candidates):
            hits.append(p.id)
    return hits


def _needed_numbers(program: str) -> list[Decimal]:
    out = []
    for op, args in re.findall(r"([a-z_]+)\(([^()]*)\)", program):
        if op.startswith("table_"):
            continue
        for arg in (a.strip() for a in args.split(",")):
            if arg.startswith(("#", "const_")) or arg == "none":
                continue
            try:
                out.append(Decimal(arg.rstrip("%").replace(",", "")))
            except InvalidOperation:
                continue
    return out


def _row_label(text: str) -> str:
    return text.split(" | ")[0].strip().lower()


def auto_flags(item: EvalItem, variant: EvalItem) -> list[str]:
    flags = []
    rows = {k: [Decimal(v) for v in values] for k, values in item.rows.items()}
    try:
        step_values = execute(item.gold_steps, {}, rows) if item.gold_steps else []
    except CalcError:
        step_values = []
    numeric = [(i, v) for i, v in enumerate(step_values) if not isinstance(v, bool)]
    for index, value in numeric:
        if abs(value) in (Decimal(0), Decimal(1)):
            continue
        hits = _present(value, variant.passages)
        if hits:
            kind = "final answer" if index == len(step_values) - 1 else f"intermediate value (step {index})"
            flags.append(f"{kind} {value:.6g} appears in {', '.join(hits[:3])}")
    for literal in _needed_numbers(item.meta.get("program", "")):
        hits = _present(literal, variant.passages)
        if hits:
            flags.append(f"needed literal {literal} appears (possibly in another format) in {', '.join(hits[:3])}")
    removed_tables = [p for p in item.passages if p.id in item.gold_passage_ids and p.id.startswith("table_")]
    remaining_labels = {_row_label(p.text): p.id for p in variant.passages if p.id.startswith("table_")}
    for p in removed_tables:
        label = _row_label(p.text)
        if label and label in remaining_labels:
            flags.append(f"removed row label {label!r} reappears in {remaining_labels[label]}")
    return flags


def candidates(revision: str | None = None) -> dict:
    """Seeded candidate negatives from each validation partition (deterministic)."""
    manifest = splits.load("finqa-grouped")
    val = splits.resolve_items(manifest, manifest["splits"]["val"])
    select_groups, dev_groups = partition_val_groups([i.group for i in val])
    pools = {"val_select": set(select_groups), "val_dev": set(dev_groups)}
    out = {}
    for pool, wanted in (("val_dev", DEV_CANDIDATES), ("val_select", SELECT_CANDIDATES)):
        variants = [(i, v) for i in val if i.group in pools[pool] and (v := splits.unanswerable_variant(i))]
        rng = random.Random(f"{PARTITION_SEED}-{pool}")
        chosen = sorted(rng.sample(sorted(variants, key=lambda x: x[1].id), min(wanted, len(variants))),
                        key=lambda x: x[1].id)
        for item, variant in chosen:
            out[variant.id] = {"pool": pool, "flags": auto_flags(item, variant), "decision": None, "reason": None}
    return {"version": AUDIT_VERSION,
            "generator": {"name": "gold_passages_removed", "builder_version": splits.BUILDER_VERSION},
            "rules": __doc__.strip(), "partition": {"seed": PARTITION_SEED, "val_select_groups": select_groups,
                                                     "val_dev_groups": dev_groups},
            "candidates": out}


def render(variant_id: str) -> str:
    """Compact view of one candidate for manual review."""
    manifest = splits.load("finqa-grouped")
    base = splits.resolve_items(manifest, [variant_id.split("::")[0]])[0]
    variant = splits.unanswerable_variant(base)
    rows = {k: [Decimal(v) for v in values] for k, values in base.rows.items()}
    try:
        values = execute(base.gold_steps, {}, rows) if base.gold_steps else []
    except CalcError:
        values = []
    words = {w for w in re.findall(r"[a-z]+", base.question.lower()) if w not in _STOP and len(w) > 2}
    scored = sorted(variant.passages, key=lambda p: -len(words & set(re.findall(r"[a-z]+", p.text.lower()))))
    lines = [f"== {variant_id}", f"Q: {base.question}",
             f"gold program: {base.meta.get('program')}  -> steps {[f'{v:.6g}' if not isinstance(v, bool) else v for v in values]}  gold {base.gold_answer}",
             "removed: " + " || ".join(f"[{p.id}] {p.text[:160]}" for p in base.passages if p.id in base.gold_passage_ids),
             "flags: " + ("; ".join(auto_flags(base, variant)) or "none"),
             "remaining, most related:"]
    lines += [f"  [{p.id}] {p.text[:220]}" for p in scored[:6]]
    return "\n".join(lines)


def write(audit: dict, path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
    return path


def load(path: Path = DEFAULT_PATH) -> dict:
    return json.loads(path.read_text())


def decide(path: Path, decisions: dict[str, tuple[str, str]]) -> None:
    """Record manual decisions: {variant_id: (decision, reason)}."""
    audit = load(path)
    for variant_id, (decision, reason) in decisions.items():
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision {decision!r}")
        if variant_id not in audit["candidates"]:
            raise KeyError(variant_id)
        audit["candidates"][variant_id].update(decision=decision, reason=reason)
    write(audit, path)
