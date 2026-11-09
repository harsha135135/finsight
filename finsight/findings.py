"""Validation for ``finsight-report-v1`` findings. Each check establishes one thing:

| Check | Establishes |
|---|---|
| schema (Pydantic) | the output has the required structure |
| citation lookup | every cited passage was supplied |
| figure matching | each quoted figure appears in its passage and normalizes to the operand |
| source period | the figure sits under the requested period in the source (column header or sentence) |
| units | additive steps do not mix units; the finding's unit comes from its operands (a division of two
  same-unit figures is a ratio), never from the model's label |
| permitted literals | every financial quantity is a cited operand |
| Decimal execution | the calculation executes exactly |
| metric check | the operands are the requested metric (row label or sentence matches its synonyms) |

The metric check is the deterministic part of claim review; a model-based review of meaning
runs after it in the workflow.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from finsight.calc import CalcError, execute, format_value
from finsight.contracts import Passage
from finsight.contracts.report import (PERMITTED_CONSTANTS, FindingOutput, ReportOutput, ValidatedClaim,
                                       ValidatedFinding, ValidatedOperand)
from finsight.normalize import _squash, supports
from finsight.retrieval.search import DERIVED, METRICS, label_matches

ADDITIVE = {"add", "subtract"}


def year_of(period: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", period)
    return int(match.group(0)) if match else None


def source_period(passage: Passage, original_text: str) -> str | None:
    """The period the source assigns to a quoted figure: its column header, or a year in its sentence."""
    quoted = _squash(original_text)
    if passage.location.table is not None:
        body = passage.text.split("  [")[0]
        for segment in body.split(" | ")[1:]:
            header, _, value = segment.rpartition(":")
            if quoted and quoted in _squash(value) or _squash(value).endswith(quoted):
                match = re.search(r"(19|20)\d{2}", header)
                return match.group(0) if match else None
        return None
    for sentence in re.split(r"(?<=[.;])\s+", passage.text):
        if quoted in _squash(sentence):
            years = re.findall(r"\b(?:19|20)\d{2}\b", sentence)
            return years[0] if len(years) == 1 else None
    return None


def _metric_ok(metric: str, finding: FindingOutput, supplied: dict[str, Passage]) -> list[str]:
    def matches(op, key):
        passage = supplied[op.passage_id]
        return label_matches(passage.text, METRICS[key]["terms"], passage.location.table is not None)
    if metric in METRICS:
        return [] if any(matches(op, metric) for op in finding.operands) else \
            [f"no operand is {metric} (row label or sentence does not match its synonyms)"]
    if metric in DERIVED:
        missing = [k for k in DERIVED[metric] if not any(matches(op, k) for op in finding.operands)]
        return [f"{metric} needs a cited {k} operand" for k in missing]
    return [f"unknown metric {metric!r}"]


def validate_finding(finding: FindingOutput, supplied: dict[str, Passage]) -> ValidatedFinding:
    base = {"metric": finding.metric, "period": finding.period}
    if finding.status == "insufficient_evidence":
        return ValidatedFinding(**base, status="insufficient_evidence",
                                issues=[finding.missing or "model reported insufficient evidence"])
    issues: list[str] = []
    if not finding.operands:
        return ValidatedFinding(**base, status="rejected", issues=["answered without cited operands"])
    unknown = sorted({op.passage_id for op in finding.operands} - set(supplied))
    if unknown:
        return ValidatedFinding(**base, status="rejected", issues=[f"cites passages that were not supplied: {unknown}"])
    names = [op.name for op in finding.operands]
    if len(set(names)) != len(names):
        return ValidatedFinding(**base, status="rejected", issues=["duplicate operand names"])
    requested_year = year_of(finding.period)
    # A second period is allowed only alongside a figure the source places in the requested year: otherwise an
    # answer built entirely from other years would pass by declaring those years (seen in the wrong-period replay).
    has_requested = requested_year is not None and any(
        op.passage_id in supplied and source_period(supplied[op.passage_id], op.original_text) == str(requested_year)
        for op in finding.operands)
    operands = []
    for op in finding.operands:
        passage = supplied[op.passage_id]
        try:
            value = Decimal(op.value)
        except InvalidOperation:
            issues.append(f"operand {op.name} value {op.value!r} is not a number")
            continue
        if not supports(passage.text, op.original_text, value, op.scale):
            issues.append(f"operand {op.name} ({op.original_text!r}) is not supported by passage {op.passage_id}")
        found_period = source_period(passage, op.original_text)
        if requested_year and str(requested_year) != found_period and not (has_requested and _other_period_allowed(finding, op)):
            issues.append(f"operand {op.name}: the source places {op.original_text!r} under "
                          f"{found_period or 'no identifiable period'}, not {requested_year}")
        operands.append(ValidatedOperand(name=op.name, passage_id=op.passage_id, original_text=op.original_text,
                                         value=op.value, scale=op.scale, unit=op.unit, period=op.period,
                                         source_period=found_period, location=passage.location.model_dump()))
    issues += _metric_ok(finding.metric, finding, supplied)
    by_name = {op.name: op for op in finding.operands}
    for index, step in enumerate(finding.program):
        for arg in step.args:
            if arg in by_name or arg.startswith(("#", "row:")):
                continue
            try:
                literal = Decimal(arg)
            except InvalidOperation:
                issues.append(f"step {index} argument {arg!r} is not an operand, step reference or constant")
                continue
            if literal not in PERMITTED_CONSTANTS:
                issues.append(f"step {index} uses literal {arg}: financial quantities must be cited operands")
        if step.op.value in ADDITIVE:
            units = {by_name[a].unit for a in step.args if a in by_name}
            if len(units) > 1:
                issues.append(f"step {index} {step.op.value} mixes units {sorted(units)}")
    if issues:
        return ValidatedFinding(**base, status="rejected", operands=operands, issues=issues)
    values = {op.name: Decimal(op.value) * op.scale.multiplier for op in finding.operands}
    try:
        results = execute(finding.program, values) if finding.program else [values[finding.operands[0].name]]
    except CalcError as exc:
        return ValidatedFinding(**base, status="rejected", operands=operands, issues=[str(exc)])
    calculation = [{"op": s.op.value, "args": s.args, "value": format_value(v)} for s, v in zip(finding.program, results)]
    claims = [_claim(c, supplied, results[-1]) for c in finding.explanation]
    return ValidatedFinding(**base, status="accepted", value=format_value(results[-1]),
                            unit=_result_unit(finding, by_name), operands=operands, calculation=calculation,
                            claims=claims)


def _result_unit(finding, by_name) -> str | None:
    """Unit of the result, derived from the cited operands rather than the model's declared unit:
    a division of two same-unit figures is a ratio; additive steps keep the operands' common unit;
    anything else is left unlabelled (None) rather than guessed."""
    units = {op.unit for op in finding.operands}
    ops = {step.op.value for step in finding.program}
    last = finding.program[-1] if finding.program else None
    if last is not None and last.op.value == "divide":
        named = [by_name[a] for a in last.args if a in by_name]
        return "ratio" if len(named) == len(last.args) and len({op.unit for op in named}) == 1 else None
    if len(units) == 1 and ops <= ADDITIVE:
        return units.pop()
    return None


def _other_period_allowed(finding: FindingOutput, op) -> bool:
    """Growth-style findings legitimately cite a second period; the operand must declare it."""
    return year_of(op.period) is not None and year_of(op.period) != year_of(finding.period) and len(finding.operands) > 1


def _claim(claim, supplied: dict[str, Passage], result) -> ValidatedClaim:
    issues = [f"cites passage {p} that was not supplied" for p in claim.passage_ids if p not in supplied]
    cited = " ".join(supplied[p].text for p in claim.passage_ids if p in supplied)
    for number in re.findall(r"\d[\d,]*(?:\.\d+)?", claim.text):   # a sentence's full stop is not a decimal point
        plain = number.replace(",", "")
        if plain and _squash(number) not in _squash(cited) and plain not in format_value(result):
            if not re.fullmatch(r"(19|20)\d{2}", plain):
                issues.append(f"number {number} is neither in the cited passages nor the computed result")
    return ValidatedClaim(text=claim.text, passage_ids=claim.passage_ids, supported=not issues, issues=issues)


def validate_report(output: ReportOutput, supplied: dict[str, Passage]) -> list[ValidatedFinding]:
    return [validate_finding(f, supplied) for f in output.findings]
