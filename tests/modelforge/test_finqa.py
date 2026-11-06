from decimal import Decimal

from finsight.calc import execute
from finsight.contracts import ReasonCode
from finsight.validation import validate_answer
from modelforge.datasets import finqa
from modelforge.splits import unanswerable_variant


def test_reference_number_semantics():
    assert finqa.finqa_number("7.81%") == Decimal("0.0781")
    assert finqa.finqa_number("const_m1") == Decimal(-1)
    assert finqa.finqa_number("1,234") == Decimal("1234.0")
    assert finqa.finqa_row(["$ 5735", "(12)"]) is None  # FinQA drops text after '(' → empty cell


def test_float_tie_rounding_matches_reference():
    # 1.276875 is the RE/2010 case: FinQA's float rounding gives 1.27687, Decimal half-even gives 1.27688.
    assert finqa.matches_gold(Decimal("1.276875"), "1.27687")


def test_program_conversion_and_execution(finqa_example):
    steps, rows = finqa.parse_program(finqa_example["qa"]["program"], finqa_example["table"])
    assert finqa.matches_gold(execute(steps, {}, rows)[-1], "0.01639")


def test_table_operation_conversion():
    table = [["", "2019", "2018"], ["net revenue", "$ 10", "$ 20"]]
    steps, rows = finqa.parse_program("table_average(net revenue, none)", table)
    assert steps[0].args == ["row:table_1"] and execute(steps, {}, rows)[-1] == Decimal(15)


def test_item_conversion_and_gold_target_validates(finqa_example):
    item = finqa.to_item(finqa_example, "test")
    assert item.group == "ETR/2016" and item.meta["calc_convention"] == "as_written"
    target = finqa.gold_answer_output(item)
    assert target is not None
    assert {o.original_text for o in target.operands} == {"$5,829", "$5,735"}
    rows = {k: [Decimal(v) for v in vs] for k, vs in item.rows.items()}
    result = validate_answer(target, {p.id: p for p in item.passages}, rows, as_written=True)
    assert result.reason_code is ReasonCode.ANSWERED


def test_scaled_figures_align_as_written(finqa_example):
    finqa_example["qa"]["program"] = "subtract(5.8, 5.7)"
    finqa_example["qa"]["exe_ans"] = 0.1
    finqa_example["pre_text"].append("the 2014 figure was $ 5.7 billion .")
    item = finqa.to_item(finqa_example, "test")
    target = finqa.gold_answer_output(item)
    assert target is not None and target.operands[0].scale.value == "billions"
    result = validate_answer(target, {p.id: p for p in item.passages}, {}, as_written=True)
    assert result.reason_code is ReasonCode.ANSWERED and result.answer == "0.1"


def test_unanswerable_variant_removes_gold_evidence(finqa_example):
    item = finqa.to_item(finqa_example, "test")
    variant = unanswerable_variant(item)
    assert variant is not None and not variant.answerable
    assert {"table_1", "table_2"}.isdisjoint({p.id for p in variant.passages})
