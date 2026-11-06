"""Deterministic checks on a model's answer, with explicit reason codes.

Outcomes are kept distinct (they must not be merged into "abstained"):

* ``ABSTAIN_EVIDENCE_ABSENT`` – the model declared the evidence insufficient.
* ``MODEL_OUTPUT_INVALID`` – contradictory fields, unknown citations, malformed programs.
* ``VALIDATION_CONTRADICTION`` – operands or results disagree with the supplied evidence.
* ``ANSWERED`` – every check passed; the answer is the *executed* program result.

Citation existence is a structural check only; whether a passage truly supports a
claim is measured separately by the evaluation harness.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation

from finsight.calc import CalcError, execute, format_value
from finsight.contracts import (
    AnswerOutput,
    Calculation,
    Op,
    Operand,
    Passage,
    ReasonCode,
    ResearchResult,
    Scale,
    Task,
)
from finsight.normalize import supports

CONSISTENCY_REL_TOL = Decimal("0.005")
_ADDITIVE = {Op.ADD, Op.SUBTRACT}
_ANSWER_SCALE = re.compile(r"\b(thousand|million|billion)s?\b", re.I)


def _result(code: ReasonCode, issues: list[str], **fields) -> ResearchResult:
    return ResearchResult(task=Task.ANSWER, reason_code=code, issues=issues, **fields)


def _parse_answer(text: str) -> Decimal | bool | None:
    cleaned = text.strip().lower().replace(",", "").replace("$", "").replace("%", "").strip()
    if cleaned in {"yes", "no"}:
        return cleaned == "yes"
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _consistent(stated: Decimal | bool, executed: Decimal | bool, answer_unit: str | None) -> bool:
    if isinstance(stated, bool) or isinstance(executed, bool):
        return stated == executed
    candidates = [executed]
    if answer_unit and (match := _ANSWER_SCALE.search(answer_unit)):
        candidates.append(executed / Scale(match.group(1).lower() + "s").multiplier)
    for target in candidates:
        exponent = stated.as_tuple().exponent
        decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
        if round(target, decimals) == stated:
            return True
        if target != 0 and abs((stated - target) / target) <= CONSISTENCY_REL_TOL:
            return True
        if target == 0 and stated == 0:
            return True
    return False


def validate_answer(
    output: AnswerOutput,
    supplied: Mapping[str, Passage],
    rows: Mapping[str, Sequence[Decimal]] | None = None,
    required_periods: Sequence[str] = (),
) -> ResearchResult:
    """Validate a model answer against the passages actually supplied to it."""
    if output.abstain:
        if output.answer is not None or output.program:
            return _result(ReasonCode.MODEL_OUTPUT_INVALID, ["abstained but also returned an answer or program"])
        return _result(ReasonCode.ABSTAIN_EVIDENCE_ABSENT, [output.abstain_reason or "evidence insufficient"])

    if output.answer is None or not output.program:
        return _result(ReasonCode.MODEL_OUTPUT_INVALID, ["answered without an answer value or program"])

    cited = {o.passage_id for o in output.operands} | {pid for c in output.claims for pid in c.passage_ids}
    unknown = sorted(cited - set(supplied))
    if unknown:
        return _result(ReasonCode.MODEL_OUTPUT_INVALID, [f"cites passages that were not supplied: {', '.join(unknown)}"])

    names = [o.name for o in output.operands]
    if len(names) != len(set(names)):
        return _result(ReasonCode.MODEL_OUTPUT_INVALID, ["operand names are not unique"])

    operands: list[Operand] = []
    for item in output.operands:
        try:
            value = Decimal(item.value.replace(",", ""))
        except InvalidOperation:
            return _result(ReasonCode.MODEL_OUTPUT_INVALID, [f"operand {item.name} value {item.value!r} is not a number"])
        if not value.is_finite():
            return _result(ReasonCode.MODEL_OUTPUT_INVALID, [f"operand {item.name} is not finite"])
        operands.append(Operand(
            name=item.name, original_text=item.original_text, value=value, unit=item.unit,
            scale=item.scale, period=item.period, passage_id=item.passage_id,
        ))

    issues: list[str] = []
    for operand in operands:
        if not supports(supplied[operand.passage_id].text, operand.original_text, operand.value, operand.scale):
            issues.append(f"operand {operand.name} ({operand.original_text!r}) is not supported by passage {operand.passage_id}")

    by_name = {o.name: o for o in operands}
    for index, step in enumerate(output.program):
        if step.op in _ADDITIVE:
            units = {by_name[a].unit for a in step.args if a in by_name and by_name[a].unit != "number"}
            if len(units) > 1:
                issues.append(f"step {index} {step.op.value} mixes units {sorted(units)}")

    periods = {o.period for o in operands if o.period}
    for period in required_periods:
        if period not in periods:
            issues.append(f"no operand supports required period {period}")

    if issues:
        return _result(ReasonCode.VALIDATION_CONTRADICTION, issues, operands=operands)

    try:
        values = execute(output.program, {o.name: o.base_value for o in operands}, rows)
    except CalcError as exc:
        code = ReasonCode.MODEL_OUTPUT_INVALID if exc.kind == "structure" else ReasonCode.VALIDATION_CONTRADICTION
        return _result(code, [str(exc)], operands=operands)

    executed = values[-1]
    calculation = Calculation(steps=output.program, values=[format_value(v) for v in values], result=format_value(executed))
    stated = _parse_answer(output.answer)
    if stated is None:
        return _result(ReasonCode.MODEL_OUTPUT_INVALID, [f"answer {output.answer!r} is not a number or yes/no"],
                       operands=operands, calculation=calculation)
    if not _consistent(stated, executed, output.answer_unit):
        return _result(ReasonCode.VALIDATION_CONTRADICTION,
                       [f"stated answer {output.answer} does not match executed result {format_value(executed)}"],
                       operands=operands, calculation=calculation)

    return _result(ReasonCode.ANSWERED, [], answer=format_value(executed), answer_unit=output.answer_unit,
                   operands=operands, calculation=calculation, claims=output.claims)
