"""Dated Claude prices (USD per million tokens) and cost from raw Anthropic usage.

Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-11-07). Cache
writes cost 1.25x (5-minute) and 2x (1-hour) base input; cache reads are 0.05x on Opus 5.5 and
Sonnet 5.5 and 0.1x on Haiku 5.5. The Batch API halves every category. Haiku 5.5 charges a
higher tier when a request's prompt (all input tokens, cached or not) exceeds 100,000 tokens.
Costs are computed from the raw ``usage`` object: ``input_tokens`` is uncached input only,
and ``cache_creation`` breaks writes down by TTL.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

PRICES_AS_OF = "2026-11-07"
PRICES_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class Rates:
    input: Decimal
    write_5m: Decimal
    write_1h: Decimal
    cache_read: Decimal
    output: Decimal


def _r(*values: str) -> Rates:
    return Rates(*(Decimal(v) for v in values))


PRICES: dict[str, dict[str, Rates]] = {
    "claude-opus-5-5": {"standard": _r("4", "5", "8", "0.20", "20")},
    "claude-sonnet-5-5": {"standard": _r("2", "2.50", "4", "0.10", "10")},
    "claude-haiku-5-5": {"standard": _r("0.10", "0.125", "0.20", "0.01", "0.50"),
                         "over_100k_prompt": _r("0.50", "0.625", "1", "0.05", "2.50")},
}
HAIKU_PROMPT_THRESHOLD = 100_000


@dataclass(frozen=True)
class TokenUsage:
    input_uncached: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0
    output: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_uncached + self.cache_write_5m + self.cache_write_1h + self.cache_read

    @classmethod
    def from_anthropic(cls, usage) -> "TokenUsage":
        """From the SDK's raw usage: TTL breakdown when present, else all writes count as 5-minute."""
        breakdown = getattr(usage, "cache_creation", None)
        writes = getattr(usage, "cache_creation_input_tokens", 0) or 0
        w5 = getattr(breakdown, "ephemeral_5m_input_tokens", None) if breakdown else None
        w1 = getattr(breakdown, "ephemeral_1h_input_tokens", None) if breakdown else None
        if w5 is None and w1 is None:
            w5, w1 = writes, 0
        return cls(input_uncached=getattr(usage, "input_tokens", 0) or 0, cache_write_5m=w5 or 0, cache_write_1h=w1 or 0,
                   cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0, output=getattr(usage, "output_tokens", 0) or 0)


def rates_for(model: str, prompt_tokens: int) -> Rates:
    if model not in PRICES:
        raise KeyError(f"no dated price for {model!r}; add it to finsight/pricing.py from {PRICES_SOURCE}")
    table = PRICES[model]
    if "over_100k_prompt" in table and prompt_tokens > HAIKU_PROMPT_THRESHOLD:
        return table["over_100k_prompt"]
    return table["standard"]


def cost(model: str, usage: TokenUsage, *, batch: bool = False) -> Decimal:
    rates = rates_for(model, usage.prompt_tokens)
    total = (usage.input_uncached * rates.input + usage.cache_write_5m * rates.write_5m
             + usage.cache_write_1h * rates.write_1h + usage.cache_read * rates.cache_read
             + usage.output * rates.output) / MILLION
    return (total / 2 if batch else total).quantize(Decimal("0.000001"))


def reservation(model: str, prompt_tokens_estimate: int, max_tokens: int, *, batch: bool = False) -> Decimal:
    """Conservative pre-call reservation: uncached input plus the full output allowance (thinking is output)."""
    return cost(model, TokenUsage(input_uncached=prompt_tokens_estimate, output=max_tokens), batch=batch)
