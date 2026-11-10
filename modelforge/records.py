"""Frozen experiment records: artifact hashes plus score regeneration.

A record lists every artifact of an experiment (adapter, checkpoints, run folders, reports) with its sha256, and the
run folders whose scores must regenerate from their saved predictions. ``verify`` reports two separate things:

* **hashes verified**: every artifact, prediction and config file is byte-identical to what was recorded (and a run's
  ``items.jsonl`` matches the ``items_sha256`` in its config, where present);
* **scores regenerated**: the scores were recomputed from the saved predictions with the scoring code and version the
  record pins, and they match. FinQA runs regenerate their scorecard files byte-identically; FinanceBench trust records
  carry a ``regenerate`` block (scoring version, gold, fiscal-year files, reviews, expected tables). Older FinanceBench
  records without one are hash-only: their hashes can be verified, their scores are not regenerated.

    uv run modelforge record verify evals/records/qwen3-4b-finqa-v1.json
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from modelforge.envinfo import sha256_dir, sha256_file

SCORECARD_FILES = ("scorecard.json", "scorecard.md")


def artifact_hash(path: Path) -> str | None:
    if path.is_dir():
        return sha256_dir(path)
    return sha256_file(path)


def regenerate_matches(run_dir: Path) -> dict:
    """Regenerate a run's scorecard in a temporary copy and compare bytes with the stored files."""
    from modelforge.runner import write_scorecard

    from modelforge import snapshots

    if not any((run_dir / name).exists() for name in SCORECARD_FILES):
        return {}  # e.g. FinanceBench runs: scored by modelforge.qa_score, regenerated through a record's regenerate block
    items = snapshots.for_run(run_dir)      # committed item snapshot; falls back to rebuilding from the dataset
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / run_dir.name
        shutil.copytree(run_dir, copy)
        write_scorecard(copy, items)
        return {name: (copy / name).read_bytes() == (run_dir / name).read_bytes()
                for name in SCORECARD_FILES if (run_dir / name).exists()}


def build(name: str, description: str, artifacts: dict[str, str], runs: list[str], facts: dict) -> dict:
    return {
        "name": name, "description": description, "facts": facts,
        "artifacts": {label: {"path": path, "sha256": artifact_hash(Path(path))} for label, path in artifacts.items()},
        "runs": {run: {"predictions_sha256": sha256_file(Path(run) / "predictions.jsonl"),
                       "config_sha256": sha256_file(Path(run) / "config.json"),
                       "scorecard_sha256": sha256_file(Path(run) / "scorecard.json")} for run in runs},
    }


def _items_match(path: Path) -> bool | None:
    config, items = path / "config.json", path / "items.jsonl"
    if not (config.exists() and items.exists()):
        return None
    expected = json.loads(config.read_text()).get("items_sha256")
    return None if expected is None else sha256_file(items) == expected


def regenerate_financebench(spec: dict) -> dict:
    """Recompute each trust table from saved predictions with the pinned scoring version; compare with the record."""
    from modelforge import qa_score

    gold = json.loads(Path(spec["gold"]).read_text())
    fiscal = qa_score.load_fiscal(spec.get("fiscal", qa_score.DEV_FISCAL_FILES))
    out = {}
    for key, entry in spec["tables"].items():
        review = json.loads(Path(entry["review"]).read_text())["verdicts"] if entry.get("review") else None
        table = qa_score.trust_table(Path(entry["run"]), gold, review, version=spec["version"], fiscal=fiscal)
        out[key] = table == entry["expected"]
    return out


# Large or machine-local artifacts that are not in the repository: adapters and checkpoints (separate download),
# training data caches, GPU probe and serving logs. Missing ones are reported as "external", not as failures.
EXTERNAL_PREFIXES = ("artifacts/adapters/", "artifacts/training/", "artifacts/probe/", "artifacts/serving/",
                     "artifacts/recovery/", "artifacts/train-")


def is_external(path: str) -> bool:
    return path.startswith(EXTERNAL_PREFIXES)


def verify(record_path: Path) -> dict:
    record = json.loads(record_path.read_text())
    report: dict = {"record": record["name"], "artifacts": {}, "runs": {}}
    for label, entry in record["artifacts"].items():
        path = Path(entry["path"])
        report["artifacts"][label] = ("external (not in this checkout)" if not path.exists() and is_external(entry["path"])
                                      else "missing" if not path.exists()
                                      else "ok" if artifact_hash(path) == entry["sha256"] else "CHANGED")
    regenerated_runs, hash_only_runs = 0, 0
    for run, entry in record["runs"].items():
        path = Path(run)
        if not path.exists():
            report["runs"][run] = "missing"
            continue
        same_inputs = (sha256_file(path / "predictions.jsonl") == entry["predictions_sha256"]
                       and sha256_file(path / "config.json") == entry["config_sha256"]
                       and _items_match(path) is not False)
        regenerated = regenerate_matches(path)
        if regenerated:
            regenerated_runs += 1
        else:
            hash_only_runs += 1
        report["runs"][run] = ("ok" if same_inputs and all(regenerated.values())
                               else f"inputs {'ok' if same_inputs else 'CHANGED'}, regenerate {regenerated}")
    if "regenerate" in record:
        specs = record["regenerate"] if isinstance(record["regenerate"], list) else [record["regenerate"]]
        try:
            tables = {f"{spec['version']}:{k}": v for spec in specs for k, v in regenerate_financebench(spec).items()}
        except FileNotFoundError as exc:
            tables = {"missing": False}
            report["regenerate_error"] = str(exc)
        report["score_tables"] = {k: "regenerated" if v else "MISMATCH" for k, v in tables.items()}
    values = [v for v in list(report["artifacts"].values()) + list(report["runs"].values()) if not v.startswith("external")]
    external = sum(v.startswith("external") for v in report["artifacts"].values())
    report["hashes_verified"] = ("all" if all(v == "ok" for v in values) and not external else
                                 f"all present ({external} external artifacts not in this checkout)" if all(v == "ok" for v in values)
                                 else "NO")
    report["missing"] = sum(v == "missing" for v in values)
    report["external_not_present"] = external
    if "regenerate" in record:
        report["scores"] = "regenerated" if all(v == "regenerated" for v in report["score_tables"].values()) else "MISMATCH"
    elif regenerated_runs and not hash_only_runs:
        report["scores"] = "regenerated"
    elif regenerated_runs:
        report["scores"] = f"regenerated for {regenerated_runs} runs; hash-only for {hash_only_runs}"
    else:
        report["scores"] = "not regenerated (hash-only record)"
    report["passed"] = all(v == "ok" for v in values) and report["scores"] != "MISMATCH"
    return report
