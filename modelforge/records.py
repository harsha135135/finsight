"""Frozen experiment records: artifact hashes plus a scorecard-regeneration check.

A record lists every artifact of an experiment (adapter, checkpoints, run folders,
reports) with its sha256, and the run folders whose scorecards must regenerate
byte-identically from their saved predictions. ``verify`` recomputes all of it, so
an experiment stays reproducible after later code or data changes.

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

    if not any((run_dir / name).exists() for name in SCORECARD_FILES):
        return {}  # e.g. FinanceBench runs: scored by modelforge.financebench_score, hashed with the record
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / run_dir.name
        shutil.copytree(run_dir, copy)
        write_scorecard(copy)
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


def verify(record_path: Path) -> dict:
    record = json.loads(record_path.read_text())
    report: dict = {"record": record["name"], "artifacts": {}, "runs": {}}
    for label, entry in record["artifacts"].items():
        path = Path(entry["path"])
        report["artifacts"][label] = ("missing" if not path.exists()
                                      else "ok" if artifact_hash(path) == entry["sha256"] else "CHANGED")
    for run, entry in record["runs"].items():
        path = Path(run)
        if not path.exists():
            report["runs"][run] = "missing"
            continue
        same_inputs = (sha256_file(path / "predictions.jsonl") == entry["predictions_sha256"]
                       and sha256_file(path / "config.json") == entry["config_sha256"])
        regenerated = regenerate_matches(path)
        report["runs"][run] = ("ok" if same_inputs and all(regenerated.values())
                               else f"inputs {'ok' if same_inputs else 'CHANGED'}, regenerate {regenerated}")
    values = list(report["artifacts"].values()) + list(report["runs"].values())
    report["passed"] = all(v == "ok" for v in values)
    report["missing"] = sum(v == "missing" for v in values)
    return report
