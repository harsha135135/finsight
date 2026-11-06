"""Render reproducible, source-linked local report artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _text(value: str) -> str:
    # Untrusted fields are plain inline text, never Markdown/HTML supplied by a model.
    value = " ".join(value.split()).replace("\\", "\\\\")
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in "`*_{}[]()#!|":
        value = value.replace(character, "\\" + character)
    return value


def markdown(result: dict) -> str:
    analysis = result["analysis"]
    lines = [f"# {_text(analysis['title'])}", "", f"Run: `{result['run_id']}` · {result['generated_at']}", "",
             f"Mode: **{result['provider']}** · Domain: **{result['domain']}**", ""]
    if result["provider"] == "demo":
        lines += ["> DEMO: deterministic source excerpts. No AI model was called; the inputs have not been independently assessed.", ""]
    lines += ["## Research goal", "", _text(result["goal"]), "", "## Summary", "", _text(analysis["summary"]), "",
              "## Findings", ""]
    for finding in analysis["findings"]:
        refs = " ".join(f"[{sid}](#source-{sid.lower()})" for sid in finding["source_ids"])
        lines += [f"- {_text(finding['claim'])} {refs} — confidence: {finding['confidence']}"]
    if result["metrics"]:
        lines += ["", "## Calculated metrics", "", "| Metric | Value | Unit | Source | Formula |", "|---|---:|---|---|---|"]
        for metric in result["metrics"]:
            cells = [metric["name"], "Unavailable" if metric["value"] is None else str(metric["value"]),
                     metric["unit"], metric["source_id"], metric["formula"]]
            lines += ["| " + " | ".join(_text(str(c)) for c in cells) + " |"]
    for heading, values in (("Risks", analysis["risks"]), ("Limitations", analysis["limitations"] + result["limitations"])):
        lines += ["", f"## {heading}", ""]
        lines += [f"- {_text(item)}" for item in values] or ["None identified."]
    lines += ["", "## Sources", ""]
    for source in result["sources"]:
        lines += [f"### Source {_text(source['id'])}", "", _text(source["title"]), "",
                  f"Location: {_text(source['uri'])}", "",
                  f"Retrieved: {source['retrieved_at']} · SHA-256: `{source['sha256']}`", ""]
    review_description = ("Passed deterministic reference and structure checks. No model review was performed."
                          if result["provider"] == "demo" else
                          "Passed automated citation checks and a separate model review.")
    lines += ["## Review", "", review_description + " This does not independently establish factual correctness.", ""]
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_reports(root: Path, result: dict) -> dict:
    directory = root / "reports" / result["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    md = directory / "report.md"
    data = directory / "report.json"
    _atomic_write(md, markdown(result))
    _atomic_write(data, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return {"markdown": str(md), "json": str(data)}
