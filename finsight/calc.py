"""A small, safe Decimal interpreter for calculation programs.

Programs are lists of steps such as ``subtract(revenue_fy2025, revenue_fy2024)``
followed by ``divide(#0, revenue_fy2024)``. Arguments may be operand names,
``#n`` references to earlier steps, plain numeric literals, or ``row:<key>``
references to numeric table rows (for FinQA-style table operations).

Model-generated text is never executed: only the operations in ``Op`` exist.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext

from finsight.contracts import CalculationStep, Op

PRECISION = 50
MAX_EXPONENT = 100
_LITERAL = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")
_STEP_REF = re.compile(r"^#(\d{1,3})$")
_ROW_REF = re.compile(r"^row:(.+)$", re.S)
_ARITY = {
    Op.ADD: 2, Op.SUBTRACT: 2, Op.MULTIPLY: 2, Op.DIVIDE: 2, Op.EXP: 2, Op.GREATER: 2,
    Op.PERCENT_CHANGE: 2, Op.TO_PERCENT: 1,
    Op.TABLE_SUM: 1, Op.TABLE_AVERAGE: 1, Op.TABLE_MAX: 1, Op.TABLE_MIN: 1,
}
_TABLE_OPS = {Op.TABLE_SUM, Op.TABLE_AVERAGE, Op.TABLE_MAX, Op.TABLE_MIN}

Value = Decimal | bool


class CalcError(ValueError):
    """Raised for an invalid program ("structure") or an undefined result ("math")."""

    def __init__(self, message: str, kind: str = "structure") -> None:
        super().__init__(message)
        self.kind = kind


def _resolve(arg: str, index: int, values: list[Value], operands: Mapping[str, Decimal]) -> Decimal:
    if match := _STEP_REF.match(arg):
        ref = int(match.group(1))
        if ref >= index:
            raise CalcError(f"step {index} references #{ref}, which is not an earlier step")
        value = values[ref]
        if isinstance(value, bool):
            raise CalcError(f"step {index} uses the yes/no result of step {ref} as a number")
        return value
    if _LITERAL.match(arg):
        return Decimal(arg)
    if arg in operands:
        return operands[arg]
    raise CalcError(f"step {index} argument {arg!r} is not an operand, step reference or literal")


def _row(arg: str, index: int, rows: Mapping[str, Sequence[Decimal]]) -> Sequence[Decimal]:
    match = _ROW_REF.match(arg)
    if not match or match.group(1) not in rows:
        raise CalcError(f"step {index} table argument {arg!r} is not a known row")
    row = rows[match.group(1)]
    if not row:
        raise CalcError(f"step {index} table row {arg!r} has no numeric cells")
    return row


def execute(
    steps: Sequence[CalculationStep],
    operands: Mapping[str, Decimal] | None = None,
    rows: Mapping[str, Sequence[Decimal]] | None = None,
) -> list[Value]:
    """Execute steps in order and return every step's value. The last value is the result."""
    operands = operands or {}
    rows = rows or {}
    if not steps:
        raise CalcError("program has no steps")
    values: list[Value] = []
    with localcontext() as context:
        context.prec = PRECISION
        context.traps[DivisionByZero] = True
        context.traps[InvalidOperation] = True
        context.traps[Overflow] = True
        for index, step in enumerate(steps):
            arity = _ARITY[step.op]
            if len(step.args) != arity:
                raise CalcError(f"step {index} {step.op.value} takes {arity} argument(s), got {len(step.args)}")
            try:
                if step.op in _TABLE_OPS:
                    row = _row(step.args[0], index, rows)
                    if step.op is Op.TABLE_SUM:
                        value: Value = sum(row, Decimal(0))
                    elif step.op is Op.TABLE_AVERAGE:
                        value = sum(row, Decimal(0)) / len(row)
                    elif step.op is Op.TABLE_MAX:
                        value = max(row)
                    else:
                        value = min(row)
                elif step.op is Op.TO_PERCENT:
                    value = _resolve(step.args[0], index, values, operands) * 100
                else:
                    a = _resolve(step.args[0], index, values, operands)
                    b = _resolve(step.args[1], index, values, operands)
                    if step.op is Op.ADD:
                        value = a + b
                    elif step.op is Op.SUBTRACT:
                        value = a - b
                    elif step.op is Op.MULTIPLY:
                        value = a * b
                    elif step.op is Op.DIVIDE:
                        value = a / b
                    elif step.op is Op.EXP:
                        if abs(b) > MAX_EXPONENT:
                            raise CalcError(f"step {index} exponent {b} exceeds {MAX_EXPONENT}", "math")
                        value = a ** (int(b) if b == b.to_integral_value() else b)
                    elif step.op is Op.GREATER:
                        value = a > b
                    else:  # PERCENT_CHANGE: (new - old) / old * 100
                        value = (a - b) / b * 100
            except (DivisionByZero, InvalidOperation, Overflow, ZeroDivisionError) as exc:
                raise CalcError(f"step {index} {step.op.value} is undefined: {type(exc).__name__}", "math") from None
            values.append(value)
    return values


def format_value(value: Value) -> str:
    """Render a step value without exponent notation; yes/no for comparisons."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = format(value.normalize(), "f")
    return "0" if text in {"-0", ""} else text
