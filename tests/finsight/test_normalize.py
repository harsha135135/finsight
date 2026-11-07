from decimal import Decimal

from finsight.contracts import Scale
from finsight.normalize import parse_figure, supports


def test_parse_scaled_currency():
    figure = parse_figure("$1.2 billion")
    assert (figure.value, figure.unit, figure.scale) == (Decimal("1.2"), "USD", Scale.BILLIONS)
    assert figure.base_value == Decimal("1200000000.0")


def test_parse_parenthesized_negative_and_percent():
    assert parse_figure("(1,234)").value == Decimal("-1234")
    assert parse_figure("12.5%").unit == "%"


def test_bare_m_is_not_a_scale_without_currency():
    assert parse_figure("5 m").scale is Scale.UNITS
    assert parse_figure("$5m").scale is Scale.MILLIONS


def test_supports_requires_text_and_matching_normalization():
    passage = "Total net revenue was $1,577 million in 2018, compared with $1,373 million."
    assert supports(passage, "$1,577 million", Decimal("1577"), Scale.MILLIONS)
    assert supports(passage, "$1,577 million", Decimal("1.577"), Scale.BILLIONS)
    assert not supports(passage, "$1,577 million", Decimal("1577"), Scale.UNITS)
    assert not supports(passage, "$1,600 million", Decimal("1600"), Scale.MILLIONS)


def test_table_cell_without_scale_word_may_use_heading_scale():
    assert supports("net revenue | 5829 | 5735", "5829", Decimal("5829"), Scale.MILLIONS)


def test_finqa_symbol_spacing_does_not_block_support():
    from decimal import Decimal
    from finsight.contracts import Scale
    from finsight.normalize import supports
    passage = "cash and investments totaling $ 5.5 billion , of which 83.6 % was held abroad"
    assert supports(passage, "$5.5 billion", Decimal("5.5"), Scale.BILLIONS)
    assert supports(passage, "83.6%", Decimal("83.6"), Scale.UNITS)
    assert not supports(passage, "$5.5 billion", Decimal("5500000000"), Scale.BILLIONS)


def test_spaces_between_digits_still_matter():
    from decimal import Decimal
    from finsight.contracts import Scale
    from finsight.normalize import supports
    assert not supports("totals were 1 234 and 9", "1234", Decimal("1234"), Scale.UNITS)
