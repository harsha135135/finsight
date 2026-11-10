# Final configuration for the sealed evaluations: for review (2026-11-10)

Nothing here has touched the FinanceBench held-out half or the FinQA test pool. Once you approve, the freeze file is
built from a clean, committed tree (`uv run python scripts/make_freeze.py --out evals/freeze-final-v2.json`), committed
once and never edited. The rules are in `docs/evaluation-rules.md`.

## FinanceBench held-out half (75 questions, 16 companies, 44 documents; no FinQA-training overlap)

| Item | Frozen choice | Why |
|---|---|---|
| Local model | **Qwen3-4B base** @ `1cfa9a72`, bf16, vLLM 0.33.0, `--compact-json`, context 6,656, `max_tokens` 1536, thinking off, greedy guided JSON | Fine-tuned adapters showed no FinanceBench gain on dev; Gemma 4 E4B was worse and needs CPU offload on the 12 GB card |
| Reference | **Haiku 5.5**, Message Batches, effort medium, `max_tokens` 5600 | The development settings, unchanged |
| Modes | supplied evidence and full document | Separates reasoning from retrieval |
| Contract and prompt | finsight-qa-v1, prompt finsight-qa-v1.5 | Unchanged since the v1.6 freeze |
| Validator and scorer | **v1.7** (parsing fix, gold-unit fix, date and reference exemptions) | Each change was measured separately on dev |
| Headline | **strict** validation; lenient also reported | Lenient accepted wrong-period answers in the controls |
| Retrieval | pdf-2 passages + **text-3**, 25 passages / 8,000 characters | Evidence complete 18 → 26 of 75 on dev |
| Fiscal years | `modelforge/fiscal.py` rule (10-K cover pages, at most a 7-day window) | Reproduces the dev hand file for 16/16 companies |
| Exclusions | Only missing or unparseable documents leave the denominator; disputed golds stay in, with a sensitivity line | Fixed before any inference |
| Narrative grading | Frozen blind procedure with decoys; partial never counts as correct | See the reviewer decision below |

## FinQA test pool (`finqa-grouped-comparison/test_shared_300`)

| Item | Frozen choice |
|---|---|
| Local models | Qwen3-4B base and the **v1** adapter, the local model with the highest dev accuracy (54.3% on val_dev, a development figure) |
| Protocol | answer-v4, FinQA serving defaults (context 5,120), the protocol the development results used |
| Reference | Haiku 5.5, Message Batches, effort medium, `max_tokens` 2600 (the val_dev settings) |
| Negatives | Unanswerable variants count only if the test-pool label audit verifies them; the audit and denominators are committed before inference |

## Estimated cost (exact counts only after authorization, because counting needs the sealed items)
- FinanceBench held-out Haiku, 150 requests: on dev the worst case was $0.25 at 5,600 output tokens and the actual
  cost $0.06 (supplied) and $0.03 (full). Cap: $0.25.
- FinQA test Haiku, about 300 requests: the val_dev run (294 items) had a worst case of $0.25 and cost $0.11. Cap: $0.25.
- Spend so far: $0.2057. Worst case after both: about $0.71 of $2.00.

## Decisions for you
1. **Approve this configuration**, or name changes. A change after the freeze means a new freeze version, never an
   edit.
2. **Narrative grader for the headline:** the frozen rules prefer an independent human. Will you (or someone you
   choose) grade the blind CSV sheet (about 30–40 rows per mode, about 1 in 5 decoys)? Otherwise Claude grades, and
   the results say so.
3. **Authorize opening** the FinanceBench held-out half and the FinQA test pool. This is recorded in
   `evals/authorizations/`. Then, in order:
   1. held-out preparation with the frozen procedures (fiscal years, gold, ingestion and retrieval, evidence needles,
      the FinQA test label audit, denominators), hashed and committed before any inference;
   2. the local runs;
   3. one Haiku batch per dataset, each after you approve its exact token count.

## Known limitations to state with the results
- The interface was refined on development outputs, mostly from Qwen-family models.
- Narrative grades depend on the grader.
- Retrieval still delivers none of the needed evidence for 32 of 75 dev questions.
- In full-document mode, Haiku attempts more questions under text-3; coverage rises and strict-accepted accuracy falls
  (dev: 14/14 → 18/28).
- Development results were used to make choices; only the sealed runs give unbiased estimates.
