"""Dry-run Claude cost estimates for a manifest selection (no API calls).

Input tokens are estimated from the exact prompts and schema at about 3.5 characters per token
(``--count-tokens`` would use the free token-counting endpoint, which needs an API key). Output
is a range, because adaptive thinking tokens are billed as output and are unknown until a pilot
measures them: low = the answer JSON only, expected = answer plus moderate thinking, high =
answer plus heavy thinking. Estimates are provisional until measured usage replaces them.
"""

from __future__ import annotations

import json
from decimal import Decimal

from finsight.pricing import PRICES_AS_OF, TokenUsage, cost
from modelforge import prompts, splits

OUTPUT_TOKENS = {"low": 250, "expected": 900, "high": 2500}
CHARS_PER_TOKEN = 3.5


def prompt_tokens(item) -> int:
    text = prompts.SYSTEM + prompts.user_message(item) + json.dumps(prompts.answer_schema())
    return int(len(text) / CHARS_PER_TOKEN) + 30


def estimate(manifest_name: str, selection: str, models: list[str], *, batch: bool = False) -> dict:
    manifest = splits.load(manifest_name)
    ids = (manifest.get("samples", {}).get(selection) or manifest.get("slices", {}).get(selection)
           or manifest["splits"].get(selection))
    items = splits.resolve_items(manifest, ids)
    inputs = [prompt_tokens(i) for i in items]
    out = {"manifest": manifest_name, "selection": selection, "items": len(items), "batch": batch,
           "input_tokens_total": sum(inputs), "prices_as_of": PRICES_AS_OF, "assumed_output_tokens_per_item": OUTPUT_TOKENS,
           "models": {}}
    for model in models:
        out["models"][model] = {
            scenario: str(sum((cost(model, TokenUsage(input_uncached=n, output=tokens), batch=batch) for n in inputs),
                              Decimal(0)).quantize(Decimal("0.01")))
            for scenario, tokens in OUTPUT_TOKENS.items()}
    return out
