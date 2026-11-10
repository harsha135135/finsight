"""Render a ``finsight-report-v1`` report as Markdown. Model text is always rendered as plain inline text."""

from __future__ import annotations

from decimal import Decimal


def _text(value) -> str:
    # Untrusted fields are plain inline text, never Markdown/HTML supplied by a model.
    value = " ".join(str(value).split()).replace("\\", "\\\\")
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in "`*_{}[]()#!|":
        value = value.replace(character, "\\" + character)
    return value


def _amount(value: str | None, unit: str | None) -> str:
    if value is None:
        return "—"
    number = Decimal(value)
    if unit and unit.lower() in ("ratio", "fraction"):
        return f"{number * 100:.1f}%"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.0f} million {unit or ''}".strip()
    return f"{number:,} {unit or ''}".strip()


def markdown(report: dict) -> str:
    filing = report["filing"]
    source = (f"Source: {_text(filing.get('form'))} accession {_text(filing.get('accession'))}, period ending "
              f"{_text(filing.get('period_end'))}, filed {_text(filing.get('filing_date'))} "
              f"([original]({filing.get('source_uri')}), sha256 `{str(filing.get('sha256'))[:12]}`, retrieved "
              f"{_text(filing.get('retrieved_at'))}); amounts as reported in this filing"
              if filing.get("accession") else "Source: no filing was used (see unresolved information)")
    lines = [f"# {_text(report['company'])}: {_text(report['period'])}", "", source, "", "## Findings", "",
             "| Metric | Period | Value | Status |", "|---|---|---:|---|"]
    for f in report["findings"]:
        lines.append(f"| {_text(f['metric'])} | {_text(f['period'])} | {_text(_amount(f.get('value'), f.get('unit')))} | "
                     f"{_text(f['status'])} |")
    lines += ["", "## Calculations and sources", ""]
    for f in report["findings"]:
        if f["status"] != "accepted":
            continue
        lines.append(f"**{_text(f['metric'])} ({_text(f['period'])})**")
        for op in f["operands"]:
            lines.append(f"- {_text(op['name'])} = {_text(op['original_text'])} ({_text(op['scale'])}, "
                         f"{_text(op.get('source_period'))}) from passage `{op['passage_id']}`")
        for step in f.get("calculation", []):
            lines.append(f"- {_text(step['op'])}({_text(', '.join(step['args']))}) = {_text(step['value'])}")
        for check in f.get("checks", []):
            lines.append(f"- check: {_text(check)}")
        for claim in f.get("claims", []):
            mark = "supported" if claim["supported"] else "unsupported"
            lines.append(f"- {_text(claim['text'])} ({mark}; passages {', '.join(claim['passage_ids'])})")
        lines.append("")
    lines += ["## Unresolved and missing information", ""]
    lines += [f"- {_text(item)}" for item in report["unresolved"]] or ["- none"]
    model = report["model"]
    lines += ["", "## Run", "", f"- Outcome: {_text(report['outcome'])}",
              f"- Model: {_text(model.get('provider'))} {_text(model.get('model'))} (contract {_text(model.get('contract'))})",
              *([f"- **{_text(model['replay']['label'])}**; responses recorded "
                 + ", ".join(sorted({_text(str(r.get('recorded_at'))[:10]) for r in model['replay']['responses']}))] if model.get("replay") else []),
              f"- Generated: {_text(report['generated_at'])}",
              f"- Usage: {_text(report['usage'])}", ""]
    return "\n".join(lines)
