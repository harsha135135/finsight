# Wrong-period analysis and FinanceBench development results (2026-11-09)

No training, no Claude calls; the FinQA test set and FinanceBench's held-out half were not read. Plan written before
generation: `docs/financebench-dev-plan.md`. Records: `evals/records/financebench-dev-v2a-v2b.json`.

## 1. Wrong-period outcomes (193 frozen items)

| Model | Correct abstention | Wrong-year answer (FinQA validator accepted / rejected) | Other wrong answer | Invalid |
|---|---:|---:|---:|---:|
| base | 30 | 52 (19 / 33) | 71 | 40 |
| v1 | 24 | 93 (80 / 13) | 76 | 0 |
| v2a | 34 | 91 (77 / 14) | 68 | 0 |
| v2b | **132** | **37** (31 / 6) | 24 | 0 |

Operand years were read from FinQA's own column headers by cell position (independent of the validator tested).

**Replay through FinSight's period validator** (`finsight.findings.validate_finding`, rows re-expressed in FinSight's
`label | 2014: v` format; 177 FinQA rows could not be converted and stay unplaceable):
- Before the fix: v1 93/93, v2a 91/91, v2b 37/37 wrong-year answers rejected for the period, but base only 11/52:
  41 passed because every operand *declared* another year, which the growth-figure exception allowed even when no
  operand was from the requested year.
- Fix (committed): a declared second period is allowed only next to a figure from the requested year. After: 52/52
  for base, all others unchanged; the positive control did not change.
- Positive control (the same models' correct answers to the original questions): 75-83% of the fine-tuned models'
  correct answers are also rejected (base: 13 of 25), in every case because an operand has no declared period (the
  second year of a two-year question, or a figure in a sentence without a single year). FinQA-format outputs leave
  operand periods empty; so on this format the validator catches wrong years but is not precise. In FinSight's own
  contract the model declares periods (as in the 3M demo).

## 2. FinanceBench development: v2a vs v2b

**Before the models:** `evidence_page_num` is a 0-based page index (92/92 verified); every development evidence page is
present in our extraction with 99.9% of FinanceBench's numbers (their page text drops spaces; ours does not).
Retrieval (single-document, question text only): gold page in the 10 retrieved passages for 33% untuned, **44% tuned on
development** (2/23 → 5/23 for domain-relevant questions).

| Mode | Model | Numeric correct (n=31) | Narrative correct / partial (n=44) | False abstention num / nar | Latency p50 / p95 |
|---|---|---:|---:|---:|---:|
| Supplied evidence | v2a | 5 (16%; stated value right: 8) | 2 / 4 | 3% / 14% | 3.5 / 7.2 s |
| Supplied evidence | v2b | 2 (6%; stated: 6) | 4 / 5 | 3% / 11% | 3.7 / 7.3 s |
| Full document | v2a | 2 (6%) | 1 / 0 | 29% / 45% | 3.4 / 5.1 s |
| Full document | v2b | 4 (13%) | 1 / 2 | 13% / 25% | 3.8 / 6.7 s |

Paired v2b − v2a (company-clustered): supplied numeric −9.7 pts (−21.9, 0.0), narrative +4.5 (0.0, +9.4); full
document numeric +6.5 (0.0, +15.2), narrative 0.0. None is distinguishable.

**Errors by first failing stage** (numeric, supplied: v2a / v2b): validation rejection 14 / 18 (mostly the stated
answer disagreeing with the model's own program; 2 / 3 were correct values whose quotes FinanceBench's line-split text
could not match), wrong answer 8 / 9, output 3 / 1, false abstention 1 / 1. Full document: retrieval 14 of 31 numeric
and 28 of 44 narrative questions fail before the model; then wrong answer and validation. Scale slips (an answer off
by 10^3 or 10^6): 3-4 per run.

**Narrative questions.** The answer-v4 contract can only express a number or yes/no; the models' claims are templated
("The answer is X."), so most narrative answers are a bare number. Blind review of all 116 answered outputs: 8 correct,
11 partial, 97 incorrect (`evals/financebench/narrative-review-dev-v1.json`). One gold label is disputed (Pfizer
"spinning off Upjohn as of Q2 2023"; that separation completed in 2020); excluding it changes no total.

**Reading.** The FinQA-trained adapters do not carry over to FinanceBench in this contract: both answer few numeric
questions correctly even with the evidence supplied, and v2b's abstention gains do not show up as a measurable
difference here. In full-document mode, retrieval misses the gold page for over half the questions. These are
development results, one run each.
