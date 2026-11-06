"""Parse figures as written in filings and normalize them to declared base units.

``"$1.2 billion"`` → value 1.2, scale billions → base value 1,200,000,000 USD.
``"(1,234)"`` → -1234. The original text is always kept next to the normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from finsight.contracts import Scale

_SCALE_WORDS = {
    "thousand": Scale.THOUSANDS, "thousands": Scale.THOUSANDS, "k": Scale.THOUSANDS,
    "million": Scale.MILLIONS, "millions": Scale.MILLIONS, "mm": Scale.MILLIONS, "m": Scale.MILLIONS,
    "billion": Scale.BILLIONS, "billions": Scale.BILLIONS, "bn": Scale.BILLIONS, "b": Scale.BILLIONS,
}
_CURRENCY = {"$": "USD", "us$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP"}
_NUMBER = re.compile(
    r"""(?P<neg_paren>\()?\s*
        (?P<currency>us\$|\$|€|£|usd|eur|gbp)?\s*
        (?P<sign>[-−–])?\s*
        (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
        (?:\s*(?P<percent>%|percent\b))?
        (?:\s*(?P<scale>thousands?|millions?|billions?|bn|mm|k|m|b)\b)?
        (?:\s*(?P<close_paren>\)))?""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class ParsedFigure:
    original: str
    value: Decimal
    unit: str
    scale: Scale

    @property
    def base_value(self) -> Decimal:
        return self.value * self.scale.multiplier


def parse_figure(text: str, default_unit: str = "number") -> ParsedFigure | None:
    """Parse the first figure in ``text``; return None when there is no number."""
    match = _NUMBER.search(text.strip())
    if not match:
        return None
    try:
        value = Decimal(match.group("number").replace(",", ""))
    except InvalidOperation:
        return None
    negative = bool(match.group("sign")) or bool(match.group("neg_paren") and match.group("close_paren"))
    if negative:
        value = -value
    unit = default_unit
    if match.group("percent"):
        unit = "%"
    elif match.group("currency"):
        unit = _CURRENCY[match.group("currency").lower()]
    scale_word = (match.group("scale") or "").lower()
    # A bare single-letter suffix only counts as a scale next to a currency figure ("$3m").
    if scale_word in {"m", "b", "k"} and unit not in _CURRENCY.values():
        scale_word = ""
    scale = _SCALE_WORDS.get(scale_word, Scale.UNITS)
    return ParsedFigure(original=match.group(0).strip(), value=value, unit=unit, scale=scale)


def figures_in(text: str) -> list[ParsedFigure]:
    """All figures in a passage, in order."""
    found = []
    for match in _NUMBER.finditer(text):
        parsed = parse_figure(match.group(0))
        if parsed is not None:
            found.append(parsed)
    return found


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("−", "-").replace("–", "-")).strip().lower()


def supports(passage_text: str, original_text: str, value: Decimal, scale: Scale) -> bool:
    """Whether the cited passage contains ``original_text`` and it normalizes to the operand.

    The original text must appear in the passage (whitespace- and case-insensitive),
    and parsing it must give the same base value as ``value`` at the declared ``scale``.
    A figure written without a scale word may carry a scale stated elsewhere (for
    example a table heading "in millions"), so its raw number may match ``value`` directly.
    """
    if _squash(original_text) not in _squash(passage_text):
        return False
    parsed = parse_figure(original_text)
    if parsed is None:
        return False
    declared = value * scale.multiplier
    if parsed.scale is not Scale.UNITS:
        return parsed.base_value == declared
    return parsed.value == value or parsed.value == declared
