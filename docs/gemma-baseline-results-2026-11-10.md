# Gemma 4 E4B vs base Qwen3-4B on FinanceBench development, supplied evidence (2026-11-10)

Protocol written before any output: `docs/gemma-baseline-plan.md`, with an addendum on feasibility, also written before
any output. Development data; the held-out half is still sealed. Scorer v1.7. Both models got the same frozen item
bytes, contract, prompt, guided JSON, `max_tokens` 1536 and context 6,656, each with its own chat template. Thinking
was off for both. For Gemma this was checked in the rendered prompt (no `<|think|>`) and in all 75 outputs (no thinking
markers).

| | Gemma 4 E4B-it QAT w4a16 | Qwen3-4B (bf16) |
|---|---|---|
| **Primary: numeric questions correct and strictly accepted (of 33)** | **2** | **10** |
| Numeric correct among strictly accepted | 2 / 14 | 10 / 15 |
| Numeric correct (any) / declined / invalid | 2 / 9 / 1 | 13 / 0 / 0 |
| Boolean correct (of 20); correct among strictly accepted | 6; 6 / 12 | 13; 11 / 15 |
| Median / p95 latency per question | 17.8 s / 62.8 s | 2.9 s / 8.9 s |
| Peak GPU memory during the run | 10.9 GB, with 3 GB of weights offloaded to CPU | — (Qwen serving at 0.87 of 12 GB) |

**Uncertainty.** Gemma minus Qwen on the primary endpoint: −8 questions (−0.24 per question). The 95% interval from a
bootstrap that resamples companies is −0.40 to −0.10 and excludes zero. Exact McNemar gives Gemma-only 0, Qwen-only 8,
p = 0.008.

**Decision (predeclared).** Gemma fails all three criteria: quality (worse, with an interval that excludes zero),
memory (it does not fit the 12 GB card at this context without CPU offload) and latency. Base Qwen3-4B stays the local
model for the sealed run.

**What went wrong for Gemma.**
- It answered 15 of 33 numeric questions with the narrative type. Only 1 of those 15 states the gold figure; most say
  the evidence is missing although it was supplied.
- It declined 9 numeric questions.
- It wrote malformed passage ids 21 times (e.g. `)[p6]`).

The interface (short passage ids, the reference format, the permitted constants) was refined on Qwen-family outputs,
so this contract is not neutral ground; Gemma's malformed ids show some of that cost. Most failures are substantive,
though: only 1 of the 15 narrative-typed answers states the figure. This is a practical deployment comparison
(different family, size and quantization), not a controlled test of architecture.

**Feasibility.** Without offload, the weights filled the card. With offload, vLLM 0.33's compiler failed on Gemma 4.
Offload plus eager mode loaded; that is a disclosed deviation and the cause of the high latency.

**Answer to the question this baseline was set up for.** The FinanceBench limits are not specific to our Qwen setup.
The other current local model in the same size class does worse, under an interface refined on Qwen outputs. With the right evidence
supplied, the stronger model (Haiku: 28 of 33 under v1.7) is what changes the result.
