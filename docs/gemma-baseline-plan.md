# Gemma 4 E4B local baseline: predeclared protocol (written before any Gemma output, 2026-11-10)

**Question.** Is the poor FinanceBench performance specific to our Qwen setup, or does another current local model
face similar limits? One bounded comparison, no fine-tuning. It is a practical deployment comparison (different
family, size and quantization), not a controlled test of architecture.

**Model.** `google/gemma-4-E4B-it-qat-w4a16-ct` at revision `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb` (Apache-2.0;
Google's quantization-aware-trained checkpoint in compressed-tensors w4a16 format for vLLM; about 8B total
parameters, "4B effective", mostly bf16 per-layer embeddings).

**Fixed conditions (identical to the base-Qwen run it is compared with).**
- Items: the 75 FinanceBench development questions, supplied-evidence mode, the same `items.jsonl` bytes
  (sha256 `3a64aa0d…`, as in `evals/financebench/freeze-qa-v1.6.json`).
- Contract finsight-qa-v1, prompt finsight-qa-v1.5, the same system and user messages, the same JSON schema,
  validator and scorer **v1.7**. Each model gets its own chat template.
- Serving: vLLM 0.33.0, `--compact-json`, `--max-model-len 6656`, `--language-model-only`, no LoRA, greedy
  decoding with guided JSON, `max_tokens` 1536, seed 0.
- Thinking off. For Gemma this means the thinking trigger is not requested; it is verified from the rendered prompt
  and from the outputs (not assumed from the absence of a reasoning parser).
- Narrative answers are not graded in this comparison; the endpoints below are deterministic.

**Feasibility gate.** Load with `--language-model-only` and measure VRAM. If it does not fit in the 12 GB card, retry
once with `--cpu-offload-gb` (recorded; latency reported). If it still does not fit, stop and report. The Qwen server
is restored afterwards whatever happens.

**Primary endpoint.** Supplied-evidence numeric questions (33) answered correctly **and accepted under strict
validation**. Base Qwen under v1.7: 10 of 33.

**Uncertainty.** Paired difference (Gemma − Qwen) on the 33 questions with a 95% interval from a bootstrap that
resamples companies (`modelforge.stats.paired_difference`, 10,000 resamples, seed fixed). Exact McNemar is reported
too, but it is not the decision rule (33 questions, grouped by company).

**Decision (Gemma becomes the local model for the sealed FinanceBench run only if all hold).**
1. Quality: the primary endpoint is at least 5 questions higher than Qwen's and the company-clustered 95% interval
   excludes zero; and strict correct-among-accepted is not more than 10 points below Qwen's.
2. Memory: it runs on the RTX 4070 at 6,656 tokens without CPU offload.
3. Latency: median latency per question at most twice Qwen's in the same setting.

Otherwise Qwen stays the local model. If the interval includes zero, the result is reported as **inconclusive**, not as
"no difference". If Gemma is lower and the interval excludes zero, it is reported as worse.

**Also reported.** Lenient results; boolean accuracy; correct/accepted, coverage and unsupported counts per answer
type; invalid outputs; latency p50/p95 and time to first token; peak GPU memory (an `nvidia-smi` sampler during the
run); quantization; the chat template's sha256 and the rendered prompt's thinking check.

## Addendum: feasibility outcome (recorded before any scored Gemma output)
- **Attempt 1** (as declared, no offload): the weights filled the card (11,963 MiB peak), leaving no memory for the KV
  cache, so vLLM refused to start.
- **Attempt 2** (the declared retry, `--cpu-offload-gb 3`): vLLM 0.33's torch.compile failed on the offloaded layers
  (`Unsupported: setattr() on OrderedDict`). This is a software incompatibility, not a memory limit.
- **Attempt 3, a disclosed deviation** (`--cpu-offload-gb 3 --enforce-eager`): the model loaded. A one-item smoke test
  returned valid JSON with no thinking text, in 36 s against Qwen's median of 2.9 s.
- **Consequence:** the memory criterion (runs without CPU offload) already fails, so Gemma cannot become the local model
  for the sealed run. The quality comparison still runs because it answers the question this baseline was set up for.
  Latency is reported, but offload and eager mode make it unrepresentative of Gemma on a larger card.
- **Context:** the longest supplied item is 6,913 Gemma tokens and 6,717 Qwen tokens, so it exceeds 6,656 for both. Qwen's
  run already recorded it as an HTTP 400. Keeping 6,656 preserves parity.
