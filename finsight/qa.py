"""Validation for ``finsight-qa-v1`` answers (see finsight.contracts.qa). Independent of FinSight's metric list.

Period check per numeric operand, against the source (not the model's declaration alone):
  pass        the source places the figure in a year, that year is the operand's declared period, and (when the
              question names periods) it is one of the requested years
  reject      the source places the figure in a different year from the declaration or from every requested year
  unverified  the source cannot place the figure (no year in its column header or sentence)
Strict mode rejects an answer with any unverified operand; lenient mode accepts it and counts it. A column date is
mapped to the company's fiscal year with its fiscal-year end (so "July 29, 2023" is FY2024 for a year ending in late
January); companies that name a fiscal year by the calendar year in which it starts are not handled. At year
granularity a quarter column and a year-to-date column are not told apart.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from finsight.calc import CalcError, execute, format_value
from finsight.contracts import Passage
from finsight.contracts.qa import ClaimCheck, OperandCheck, QAOutput, QAResult
from finsight.contracts.report import PERMITTED_CONSTANTS

# finsight-qa-v1.5: counts used by standard formulas (averages over 2-5 periods, months, day counts) are allowed;
# financial quantities must still be cited operands. finsight-report-v1 keeps PERMITTED_CONSTANTS unchanged.
QA_CONSTANTS = PERMITTED_CONSTANTS | {__import__('decimal').Decimal(n) for n in (2, 3, 4, 5, 12, 360, 365)}
from finsight.normalize import _squash, supports

_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
ADDITIVE = {"add", "subtract"}


_MONTH = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b")


def fiscal_year(text: str, fiscal_year_end: tuple[int, int] | None = None) -> int | None:
    """The fiscal year a column header or sentence assigns: a full date is mapped with the company's fiscal-year end
    (the year Y whose year-end, plus 7 days, is on or after the date); otherwise a single plain year."""
    from datetime import date, timedelta

    dates = {(int(y), _MONTH[m.lower()[:3]], int(d)) for m, d, y in _DATE.findall(text)}
    if len(dates) == 1:
        year, month, day = dates.pop()
        if fiscal_year_end is None:
            return year
        fm, fd = fiscal_year_end
        when = date(year, month, min(day, 28) if month == 2 else day)
        for candidate in (year - 1, year, year + 1):
            if when <= date(candidate, fm, min(fd, 28) if fm == 2 else fd) + timedelta(days=7):
                return candidate
        return None
    years = {int(y) for y in _YEAR.findall(text)}
    return years.pop() if len(years) == 1 else None


def source_text(passage: Passage, original_text: str) -> str | None:
    """The column header (table row) or the sentence (text) the source gives a quoted figure."""
    quoted = _squash(original_text)
    if not quoted:
        return None
    if passage.location.table is not None:
        body = passage.text.split("  [")[0]
        for segment in body.split(" | ")[1:]:
            header, _, value = segment.rpartition(":")
            if quoted in _squash(value) or _squash(value).endswith(quoted):
                return header
        return None
    for sentence in re.split(r"(?<=[.;])\s+", passage.text):
        if quoted in _squash(sentence):
            return sentence
    return None


def requested_years(question: str) -> set[int]:
    """Fiscal years a question asks about: 4-digit years, FY2023 / FY 2023 / FY23, Q22023 / Q2'23, Jun'23, and
    ranges such as FY2017 - FY2019 (expanded; 'from 2012 to 2018' is two endpoints, not a range)."""
    text = question.replace("’", "'")
    years = {int(y) for y in _YEAR.findall(text)}
    for a, b in re.findall(r"(?i)(?:fy\s?)?((?:19|20)\d{2})\s*(?:-|–|through)\s*(?:fy\s?)?((?:19|20)\d{2})", text):
        if 0 < int(b) - int(a) <= 10:
            years |= set(range(int(a), int(b) + 1))
    years |= {int(m) for m in re.findall(r"(?i)\bq[1-4]\s*'?((?:19|20)\d{2})\b", text)}
    years |= {2000 + int(m) for m in re.findall(r"(?i)\bfy\s?'?(\d{2})\b", text)}
    years |= {2000 + int(m) for m in re.findall(r"(?i)(?:\bq[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s?'(\d{2})\b", text)}
    return years


# Claim-check policies. "v1.6": every digit run (a trailing comma is kept: "14,694," fails to match). "parse": numbers
# are digit groups only (fixes the trailing comma). "v1.7": "parse" plus two exempt reference spans that are not
# financial figures: the day in a calendar date ("December 31, 2022", "May 31") and numbered references to a
# document part ("Item 9.01", "Note 6", "Exhibit 21.1"). Only digits inside such a span are exempt.
CLAIM_POLICIES = ("v1.6", "parse", "v1.7")
_MONTH_NAME = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_REFERENCE_SPANS = re.compile(
    rf"(?i)\b{_MONTH_NAME}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\.\d)(?!,\d)"                          # date: month name + day
    r"|\b(?:item|note|part|exhibit|section|schedule|rule)\s+\d+[a-z]?(?:\.\d+)*\b(?![\d,]*\s*%)")   # document reference


def claim_numbers(text: str, policy: str = "v1.7") -> list[str]:
    if policy not in CLAIM_POLICIES:
        raise ValueError(f"unknown claim policy {policy!r}")
    if policy == "v1.6":
        return re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    if policy == "v1.7":
        text = _REFERENCE_SPANS.sub(lambda m: " " * len(m.group(0)), text)
    return re.findall(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|(?<![\d.,])\d+(?:\.\d+)?(?!\d)", text)


def _claim(claim, supplied: dict[str, Passage], result: Decimal | None, policy: str = "v1.7") -> ClaimCheck:
    issues = [f"cites passage {p} that was not supplied" for p in claim.passage_ids if p not in supplied]
    cited = " ".join(supplied[p].text for p in claim.passage_ids if p in supplied)
    for number in claim_numbers(claim.text, policy):
        plain = number.replace(",", "")
        if re.fullmatch(r"(19|20)\d{2}", plain) or _squash(number) in _squash(cited):
            continue
        if result is not None and plain in format_value(result):
            continue
        issues.append(f"number {number} is not in the cited passages")
    return ClaimCheck(text=claim.text, passage_ids=claim.passage_ids, supported=not issues, issues=issues)


def _result_unit(output: QAOutput) -> str | None:
    units = {op.unit for op in output.operands}
    last = output.program[-1] if output.program else None
    by_name = {op.name: op for op in output.operands}
    if last is not None and last.op.value == "divide":
        named = [by_name[a] for a in last.args if a in by_name]
        return "ratio" if len(named) == len(last.args) and len({op.unit for op in named}) == 1 else None
    if len(units) == 1 and {s.op.value for s in output.program} <= ADDITIVE:
        return units.pop()
    return None


def validate_qa(output: QAOutput, supplied: dict[str, Passage], requested: set[int] | None = None, *,
                fiscal_year_end: tuple[int, int] | None = None, strict: bool = True, claim_policy: str = "v1.7") -> QAResult:
    requested = requested or set()
    kind = output.answer_type
    if kind == "insufficient_evidence":
        return QAResult(answer_type=kind, status="insufficient_evidence", issues=[output.missing or "no reason given"])
    issues: list[str] = []
    claims: list[ClaimCheck] = []
    operands: list[OperandCheck] = []
    value: Decimal | None = None
    coverage, missing_years = None, []
    if kind == "numeric":
        if not output.operands:
            return QAResult(answer_type=kind, status="rejected", issues=["numeric answer without cited operands"])
        names = [op.name for op in output.operands]
        if len(set(names)) != len(names):
            return QAResult(answer_type=kind, status="rejected", issues=["duplicate operand names"])
        values = {}
        for op in output.operands:
            passage = supplied.get(op.passage_id)
            try:
                number = Decimal(op.value)
            except InvalidOperation:
                issues.append(f"operand {op.name} value {op.value!r} is not a number")
                continue
            ok = passage is not None and supports(passage.text, op.original_text, number, op.scale)
            if passage is None:
                issues.append(f"operand {op.name} cites passage {op.passage_id} that was not supplied")
            elif not ok:
                issues.append(f"operand {op.name} ({op.original_text!r}) is not supported by passage {op.passage_id}")
            header = source_text(passage, op.original_text) if passage is not None else None
            source = fiscal_year(header, fiscal_year_end) if header else None
            declared = _YEAR.search(op.period) or re.search(r"(?i)fy\s?'?(\d{2})\b", op.period)
            declared_year = (int(declared.group(1)) if len(declared.group(1)) == 4 else 2000 + int(declared.group(1))) if declared else None
            if source is None:
                period, note = "unverified", "the source gives no year for this figure"
            elif declared_year is not None and source != declared_year:
                period, note = "reject", f"the source places it in {source}, declared {op.period}"
            elif requested and source not in requested:
                period, note = "reject", f"the source places it in {source}, not a requested year {sorted(requested)}"
            else:
                period, note = "pass", None
            if period == "reject":
                issues.append(f"operand {op.name}: {note}")
            operands.append(OperandCheck(name=op.name, passage_id=op.passage_id, original_text=op.original_text,
                                         value=op.value, scale=op.scale, declared_period=op.period,
                                         source_year=str(source) if source is not None else None, supported=ok,
                                         period=period, note=note))
            values[op.name] = number * op.scale.multiplier
        for index, step in enumerate(output.program):
            for arg in step.args:
                if arg in values or arg.startswith("#"):
                    continue
                try:
                    literal = Decimal(arg)
                except InvalidOperation:
                    issues.append(f"step {index} argument {arg!r} is not an operand, step reference or constant")
                    continue
                if literal not in QA_CONSTANTS:
                    issues.append(f"step {index} uses literal {arg}: financial quantities must be cited operands")
        unverified = sum(o.period == "unverified" for o in operands)
        # Coverage: every requested year must be covered by a figure the source places in it (a growth answer built
        # only from the later year, or an average missing a year, does not answer the question).
        placed = {int(o.source_year) for o in operands if o.period == "pass" and o.source_year}
        missing_years = sorted(requested - placed) if requested else []
        coverage = "complete" if not missing_years else "unverified" if unverified else "missing"
        if coverage == "missing":
            issues.append(f"requested year(s) {missing_years} not covered by any cited figure")
        if strict and unverified:
            issues.append(f"{unverified} operand(s) could not be placed in a period by the source")
        if len(output.operands) > 1 and not output.program:
            issues.append("several operands but no program to combine them")
        if not issues:
            try:
                results = execute(output.program, values) if output.program else [values[output.operands[0].name]]
                value = results[-1]
            except (CalcError, KeyError, InvalidOperation) as exc:
                issues.append(f"calculation failed: {exc}")
    elif kind == "boolean" and output.boolean is None:
        issues.append("boolean answer without yes or no")
    if kind in ("boolean", "narrative"):
        if not output.claims:
            issues.append(f"{kind} answer without cited claims")
    claims = [_claim(c, supplied, value, claim_policy) for c in output.claims]
    if kind in ("boolean", "narrative"):
        issues += [f"claim {n}: {i}" for n, c in enumerate(claims) for i in c.issues]
    status = "rejected" if issues else "accepted"
    return QAResult(answer_type=kind, status=status, value=format_value(value) if value is not None else None,
                    unit=_result_unit(output) if kind == "numeric" else None, boolean=output.boolean,
                    operands=operands, claims=claims, issues=issues,
                    period_unverified=sum(o.period == "unverified" for o in operands),
                    coverage=coverage if kind == "numeric" else None, missing_years=missing_years if kind == "numeric" else [])
