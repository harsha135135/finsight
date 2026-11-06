"""FinQA adapter (MIT licence): download at a pinned commit, convert to EvalItems.

Number and table-row parsing mirror FinQA's reference executor (``str_to_num`` and
``process_row``) so the calculator suite measures our interpreter against the
benchmark's own semantics. Executing *gold* programs measures the calculator only;
it says nothing about a model's reasoning.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

from finsight.calc import format_value
from finsight.contracts import AnswerOutput, CalculationStep, Claim, Confidence, Op, OperandOutput, Passage, Scale, ValueKind
from finsight.normalize import figures_in
from modelforge.items import EvalItem

REPO = "czyssrs/FinQA"
REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
SPLITS = ("train", "dev", "test")
_FINQA_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater",
              "table_max", "table_min", "table_sum", "table_average"}


class FinQAProgramError(ValueError):
    pass


def data_dir(revision: str = REVISION) -> Path:
    root = Path(os.environ.get("FINSIGHT_DATA_DIR", Path.home() / ".cache" / "finsight" / "datasets"))
    return root / "finqa" / revision


def download(revision: str = REVISION) -> Path:
    directory = data_dir(revision)
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = directory / f"{split}.json"
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"https://raw.githubusercontent.com/{REPO}/{revision}/dataset/{split}.json"
        with httpx.stream("GET", url, timeout=600, follow_redirects=False) as response:
            response.raise_for_status()
            temporary = target.with_suffix(".part")
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        temporary.replace(target)
    return directory


def load_raw(split: str, revision: str = REVISION) -> list[dict]:
    path = data_dir(revision) / f"{split}.json"
    if not path.exists():
        download(revision)
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- reference semantics


def finqa_number(text: str) -> Decimal | None:
    """FinQA's ``str_to_num``: commas removed, '%' divides by 100, ``const_m1`` is -1."""
    text = text.replace(",", "")
    try:
        return Decimal(str(float(text)))
    except ValueError:
        pass
    if "%" in text:
        try:
            return Decimal(str(float(text.replace("%", "")))) / 100
        except ValueError:
            return None
    if "const" in text:
        token = text.replace("const_", "")
        token = "-1" if token == "m1" else token
        try:
            return Decimal(token)
        except InvalidOperation:
            return None
    return None


def finqa_row(cells: list[str]) -> list[Decimal] | None:
    """FinQA's ``process_row``: strip '$', drop anything after '(', parse each cell."""
    values = []
    for cell in cells:
        number = finqa_number(cell.replace("$", "").strip().split("(")[0].strip())
        if number is None:
            return None
        values.append(number)
    return values


def _tokenize(program: str) -> list[tuple[str, list[str]]]:
    steps = []
    for raw in re.findall(r"([a-z_]+)\(([^()]*)\)", program):
        op, args = raw
        steps.append((op, [a.strip() for a in args.split(", ")] if ", " in args else [a.strip() for a in args.split(",")]))
    if not steps:
        raise FinQAProgramError(f"cannot parse program {program!r}")
    return steps


def parse_program(program: str, table: list[list[str]]) -> tuple[list[CalculationStep], dict[str, list[Decimal]]]:
    """Convert a FinQA program string into interpreter steps plus the table rows it uses."""
    labels = {row[0]: index for index, row in enumerate(table) if row}
    steps: list[CalculationStep] = []
    rows: dict[str, list[Decimal]] = {}
    for op, args in _tokenize(program):
        if op not in _FINQA_OPS:
            raise FinQAProgramError(f"unknown FinQA operation {op!r}")
        if op.startswith("table_"):
            label = args[0]
            if label not in labels:
                raise FinQAProgramError(f"table row {label!r} not found")
            key = f"table_{labels[label]}"
            numeric = finqa_row(table[labels[label]][1:])
            if numeric is None:
                raise FinQAProgramError(f"table row {label!r} has non-numeric cells")
            rows[key] = numeric
            steps.append(CalculationStep(op=Op(op), args=[f"row:{key}"]))
            continue
        if len(args) != 2:
            raise FinQAProgramError(f"{op} expects two arguments, got {args}")
        converted = []
        for arg in args:
            if arg.startswith("#"):
                converted.append(arg)
            else:
                number = finqa_number(arg)
                if number is None:
                    raise FinQAProgramError(f"argument {arg!r} is not numeric")
                converted.append(format_value(number))
        steps.append(CalculationStep(op=Op(op), args=converted))
    return steps, rows


def gold_text(exe_ans) -> str:
    if isinstance(exe_ans, str):
        return exe_ans.strip().lower()
    return format_value(Decimal(str(exe_ans)))


def matches_gold(executed: Decimal | bool, gold: str) -> bool:
    """FinQA's comparison: executed results are rounded to 5 decimals *as Python floats*.

    Float rounding (not Decimal half-even) is deliberate: it reproduces the reference
    executor exactly, including how it resolves ties such as 1.276875.
    """
    if isinstance(executed, bool):
        return gold in {"yes", "no"} and (gold == "yes") == executed
    if gold in {"yes", "no"}:
        return False
    return round(float(executed), 5) == round(float(gold), 5)


# ----------------------------------------------------------------- conversion


def _group(filename: str) -> str:
    return filename.rsplit("/", 1)[0]


def to_item(example: dict, split: str) -> EvalItem:
    filename = example["filename"]
    company, year = filename.split("/")[:2]
    text_sentences = list(example["pre_text"]) + list(example["post_text"])
    passages: list[Passage] = []
    offset = 0
    for index, sentence in enumerate(text_sentences):
        sentence = sentence.strip()
        if not sentence or sentence == ".":
            continue
        section = "pre_text" if index < len(example["pre_text"]) else "post_text"
        passages.append(Passage(id=f"text_{index}", document_id=filename, section=section,
                                start=offset, end=offset + len(sentence), text=sentence))
        offset += len(sentence) + 1
    header = " | ".join(cell for cell in example["table_ori"][0]) if example["table_ori"] else ""
    rows: dict[str, list[str]] = {}
    for index, (row_ori, row) in enumerate(zip(example["table_ori"], example["table"])):
        text = " | ".join(row_ori)
        if index > 0 and header:
            text = f"{text}  [columns: {header}]"
        if not text.strip(" |"):
            continue
        passages.append(Passage(id=f"table_{index}", document_id=filename, section="table",
                                location={"table": "0", "row": index}, start=offset, end=offset + len(text), text=text))
        offset += len(text) + 1
        numeric = finqa_row(row[1:]) if index > 0 and len(row) > 1 else None
        if numeric:
            rows[f"table_{index}"] = [format_value(v) for v in numeric]
    qa = example["qa"]
    try:
        gold_steps, _ = parse_program(qa["program"], example["table"])
    except FinQAProgramError:
        gold_steps = None
    present = {p.id for p in passages}
    return EvalItem(
        id=example["id"], dataset="finqa", group=_group(filename), company=company, year=year,
        question=qa["question"], passages=passages, rows=rows, answerable=True,
        gold_answer=gold_text(qa["exe_ans"]), gold_steps=gold_steps,
        gold_passage_ids=[key for key in qa.get("gold_inds", {}) if key in present],
        meta={"split": split, "program": qa["program"], "calc_convention": "as_written"},
    )


def iter_items(split: str, revision: str = REVISION) -> Iterator[EvalItem]:
    for example in load_raw(split, revision):
        yield to_item(example, split)


def _find_figure(value: Decimal, unit: str | None, passages: list[Passage]):
    """Find a figure with this value; plain literals prefer non-percent figures."""
    fallback = None
    for passage in passages:
        for figure in figures_in(passage.text):
            if figure.value != value:
                continue
            if unit is not None:
                if figure.unit == unit:
                    return passage, figure
            elif figure.unit != "%":
                return passage, figure
            elif fallback is None:
                fallback = (passage, figure)
    return None if unit is not None else fallback


def gold_answer_output(item: EvalItem) -> AnswerOutput | None:
    """Express a FinQA gold program in FinSight's answer contract, with cited operands.

    Each literal becomes an operand only when that figure is found in a passage
    (gold passages first); figures keep their written scale (FinQA computes with
    figures as written). A percent literal such as ``7.81%`` becomes an operand of
    7.81 with unit "%" plus an explicit ``divide(v, 100)`` step. Returns None when
    the program cannot be aligned to the evidence; such items are counted and
    excluded from training targets.
    """
    if not item.gold_steps or item.gold_answer is None:
        return None
    raw_steps = [[a.strip() for a in args.split(",")]
                 for args in re.findall(r"[a-z_]+\(([^()]*)\)", item.meta.get("program", ""))]
    ordered = sorted(item.passages, key=lambda p: (p.id not in item.gold_passage_ids, p.id))
    operands: dict[str, OperandOutput] = {}
    by_literal: dict[str, str] = {}
    prefix: list[CalculationStep] = []
    percent_steps: dict[str, int] = {}

    def operand_for(raw_arg: str) -> str | None:
        percent = raw_arg.endswith("%")
        literal = raw_arg.rstrip("%").replace(",", "").strip()
        key = raw_arg
        if key in by_literal:
            return by_literal[key]
        try:
            value = Decimal(literal)
        except InvalidOperation:
            return None
        found = _find_figure(value, "%" if percent else None, ordered)
        if not found:
            return None
        passage, figure = found
        name = f"v{len(operands)}"
        operands[name] = OperandOutput(name=name, original_text=figure.original, value=format_value(value),
                                       unit=figure.unit, scale=figure.scale, period=None, passage_id=passage.id)
        by_literal[key] = name
        return name

    # First pass: percent literals need conversion steps placed before the program.
    for raw in raw_steps:
        for raw_arg in raw:
            if raw_arg.endswith("%") and raw_arg not in percent_steps:
                name = operand_for(raw_arg)
                if name is None:
                    return None
                percent_steps[raw_arg] = len(prefix)
                prefix.append(CalculationStep(op=Op.DIVIDE, args=[name, "100"]))
    shift = len(prefix)
    steps: list[CalculationStep] = []
    for step, raw in zip(item.gold_steps, raw_steps):
        args: list[str] = []
        for arg, raw_arg in zip(step.args, raw):
            if arg.startswith("#"):
                args.append(f"#{int(arg[1:]) + shift}")
            elif arg.startswith("row:") or raw_arg.startswith("const_"):
                args.append(arg)
            elif raw_arg.endswith("%"):
                args.append(f"#{percent_steps[raw_arg]}")
            else:
                name = operand_for(raw_arg)
                if name is None:
                    return None
                args.append(name)
        steps.append(CalculationStep(op=step.op, args=args))
    program = prefix + steps
    cited = sorted({o.passage_id for o in operands.values()} | set(item.gold_passage_ids))
    return AnswerOutput(
        abstain=False, abstain_reason=None, operands=list(operands.values()), program=program,
        answer=item.gold_answer, answer_unit=None,
        claims=[Claim(text=f"The answer is {item.gold_answer}.", passage_ids=cited,
                      calculation_steps=list(range(len(program))), kind=ValueKind.CALCULATED,
                      confidence=Confidence.HIGH)],
    )
