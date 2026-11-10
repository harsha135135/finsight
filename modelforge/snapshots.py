"""Committed snapshots of the evaluation items behind recorded FinQA runs, so scorecards regenerate from a fresh clone
without downloading FinQA.

A snapshot is the exact list of EvalItems a run was scored on (rebuilt from the pinned dataset revision and the
manifest), stored gzipped under its content hash. ``evals/snapshots/index.json`` maps (manifest name, manifest sha256,
sha256 of the ordered item ids) to the snapshot. When the dataset is available, ``check`` rebuilds the items and
confirms the snapshot is byte-identical.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from modelforge.items import EvalItem

ROOT = Path("evals/snapshots")
INDEX = ROOT / "index.json"


def _ids_hash(item_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(list(item_ids)).encode()).hexdigest()


def key(manifest: str, manifest_sha256: str, item_ids: list[str]) -> str:
    return f"{manifest}:{manifest_sha256[:16]}:{_ids_hash(item_ids)[:16]}"


def _body(items: list[EvalItem]) -> bytes:
    return "".join(i.model_dump_json() + "\n" for i in items).encode()


def _index() -> dict:
    return json.loads(INDEX.read_text()) if INDEX.exists() else {"format": "modelforge-item-snapshots-v1", "snapshots": {}}


def write(manifest: str, manifest_sha256: str, item_ids: list[str], items: list[EvalItem]) -> str:
    body = _body(items)
    digest = hashlib.sha256(body).hexdigest()
    path = ROOT / "items" / f"{digest[:16]}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body, mtime=0))
    index = _index()
    index["snapshots"][key(manifest, manifest_sha256, item_ids)] = {"file": str(path), "sha256": digest, "items": len(items)}
    INDEX.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    return digest


def load(manifest: str, manifest_sha256: str, item_ids: list[str]) -> list[EvalItem] | None:
    entry = _index()["snapshots"].get(key(manifest, manifest_sha256, item_ids))
    if entry is None:
        return None
    body = gzip.decompress(Path(entry["file"]).read_bytes())
    if hashlib.sha256(body).hexdigest() != entry["sha256"]:
        raise ValueError(f"snapshot {entry['file']} does not match its sha256")
    items = [EvalItem.model_validate_json(line) for line in body.decode().splitlines() if line]
    if [i.id for i in items] != list(item_ids):
        raise ValueError(f"snapshot {entry['file']} holds different items than the run")
    return items


def for_run(run_dir: Path) -> list[EvalItem] | None:
    config = json.loads((run_dir / "config.json").read_text())
    return load(config["manifest"], config["manifest_sha256"], config["item_ids"])


def check(run_dir: Path) -> bool:
    """The snapshot equals the items rebuilt from the dataset (needs the dataset)."""
    from modelforge import splits

    config = json.loads((run_dir / "config.json").read_text())
    rebuilt = splits.resolve_items(splits.load(config["manifest"]), config["item_ids"])
    snap = for_run(run_dir)
    return snap is not None and _body(snap) == _body(rebuilt)
