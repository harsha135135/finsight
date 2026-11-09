# FinanceBench development under finsight-qa-v1.5: base, v2a, v2b (2026-11-10)

Plan: `docs/financebench-qa-plan.md`. No training, no Claude calls; FinanceBench's held-out half and the FinQA test set
were not read. No configuration is selected here. Record: `evals/records/financebench-dev-qa15.json`.

## Interface correction before these results (stated plainly)
The first scored run (v1.4, recorded as measured in `artifacts/runs/20261110-qa-fb-*`) was confounded by the interface:
FinQA-trained adapters cut passage ids containing colons (`3M_2018_10K:v2:p60:r18` → `3M_2018_10K`; v2a lost 45 of 74
answers to that alone), models named intermediate steps instead of `#n`, and averages and day counts need constants
(2, 3, 365) the contract did not allow. v1.5 (short ids p1..pn, a multi-step example, those constants for this contract
only) was rerun on FinanceBench only. These changes were made after seeing development outputs; the held-out half gives
the honest number.

## Results (75 development questions; 33 numeric, 20 boolean, 22 narrative)

| Mode | Model | Numeric correct (lenient / strict) | Boolean correct | Narrative correct / partial | Latency p50 / p95 |
|---|---|---:|---:|---:|---:|
| Supplied evidence | base | 12 / 9 | 11 | 9 / 7 | 2.9 / 8.9 s |
| Supplied evidence | v2a | 11 / 8 | 7 | 7 / 3 | 2.6 / 5.1 s |
| Supplied evidence | v2b | 7 / 3 | 11 | 5 / 4 | 2.1 / 5.5 s |
| Full document | base | 1 / 0 | 7 | 2 / 3 | 3.1 / 29.6 s |
| Full document | v2a | 3 / 1 | 2 | 2 / 3 | 2.7 / 5.1 s |
| Full document | v2b | 2 / 0 | 8 | 1 / 4 | 2.0 / 4.3 s |

Strict = an operand whose period the source cannot place counts as a rejection. Narrative: blind review of all 124
answered outputs (26 correct, 30 partial, 68 incorrect; `evals/financebench/narrative-review-dev-qa15-v1.json`).

Paired differences vs base (company-clustered) all include zero; the largest: supplied numeric v2b −15.2 pts (−35.7,
0.0), supplied narrative v2b −18.2 (−50.0, 0.0), supplied boolean v2a −20.0 (−52.6, +8.3). Fine-tuning on FinQA did not
measurably improve transfer to FinanceBench, and v2b trends worse on numeric and narrative answers.

## Errors by stage
- Supplied, numeric: wrong answers dominate for the adapters (v2a 18, v2b 21 of 33), validation rejections for base
  (15; mostly unsupported quotes and program errors).
- Full document: retrieval is the first failing stage for about half of all questions (gold pages: all retrieved 25,
  some 5, none 45). Correct answers despite no gold page are mostly booleans (a coin-flip floor), not evidence.

## Period validator
- Native FinQA controls: correct native answers rejected for their period: **0** of 9 / 85 / 84 (base / v2a / v2b);
  about 36% of their figures are unverified (headers or sentences without a single year). Wrong-year answers that pass:
  8 / 11 / 4 under the frozen rule, 7 / 2 / 1 with "from A to B" read as two endpoints (the frozen rule expanded it into
  every year between; corrected after these runs). Remaining gap: the check does not require every requested year to be
  covered.
- Real filings (gold-matching numeric answers): **0** rejected; 3-4 per model unverified in supplied mode.

## What users can trust today (development evidence)
- A numeric answer that the validator accepts with every period passing, from supplied evidence: the period check made
  no false rejections in either control, and accepted answers are calculator results over cited figures.
- Not yet: full-document answers (retrieval misses the gold page for most questions) and narrative answers (at best
  9 of 22 correct).
