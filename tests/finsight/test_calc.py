from decimal import Decimal

import pytest

from finsight.calc import CalcError, execute, format_value
from finsight.contracts import CalculationStep as S


def run(*steps, operands=None, rows=None):
    return execute([S(op=op, args=list(args)) for op, *args in steps], operands, rows)


def test_finqa_style_program_with_step_references():
    values = run(("subtract", "5829", "5735"), ("divide", "#0", "5735"))
    assert format_value(values[0]) == "94"
    assert round(values[1], 5) == Decimal("0.01639")


def test_named_operands_and_percentages():
    values = run(("divide", "op_inc", "revenue"), ("to_percent", "#0"),
                 operands={"op_inc": Decimal("25"), "revenue": Decimal("125")})
    assert values[-1] == Decimal("20")
    assert run(("percent_change", "125", "100"))[-1] == Decimal("25")


def test_table_operations():
    rows = {"net revenue": [Decimal("1"), Decimal("2"), Decimal("6")]}
    assert run(("table_average", "row:net revenue"), rows=rows)[-1] == Decimal("3")
    assert run(("table_max", "row:net revenue"), rows=rows)[-1] == Decimal("6")
    assert run(("table_sum", "row:net revenue"), rows=rows)[-1] == Decimal("9")


def test_greater_returns_yes_no():
    assert format_value(run(("greater", "3", "2"))[-1]) == "yes"


@pytest.mark.parametrize("steps,kind", [
    ((("divide", "1", "0"),), "math"),
    ((("add", "1"),), "structure"),
    ((("add", "#1", "1"),), "structure"),
    ((("add", "unknown_name", "1"),), "structure"),
    ((("greater", "2", "1"), ("add", "#0", "1")), "structure"),
    ((("table_sum", "row:missing"),), "structure"),
    ((("exp", "2", "1000"),), "math"),
])
def test_invalid_programs_raise_with_kind(steps, kind):
    with pytest.raises(CalcError) as info:
        run(*steps)
    assert info.value.kind == kind


def test_python_injection_is_just_an_unknown_argument():
    with pytest.raises(CalcError):
        run(("add", "__import__('os').system('true')", "1"))
