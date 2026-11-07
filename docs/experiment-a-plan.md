# Experiment (a): clean negatives at v1's mix (written before results, 2026-11-07)

**Question.** Does replacing v1's training negatives with filter-cleaned negatives, at the same mix and with
everything else fixed, change abstention and answerable accuracy?

**Training (launched 15:14 UTC; `artifacts/adapters/qwen3-4b-finqa-v2a` on the PC).**

| | v1 | v2a (this run) |
|---|---|---|
| Base | Qwen/Qwen3-4B @ 1cfa9a72 | same |
| Answerable rows | 1,428 | identical ids and content (sha256 of the set matches) |
| Negatives | 285 (247 paired with their answerable original) | 283, all paired; 227 differ from v1's |
| Mix | 285 / 1,713 = 16.6%; 20.0 per 100 answerable | 283 / 1,711 = 16.5%; 19.8 per 100 answerable |
| Negative quality | uncleaned (hand audit of a similar pool: ~65% truly unanswerable) | rule-filtered, ~93% on the hand-audited validation sample (in-sample estimate); **not hand-verified** |
| QLoRA, lr, schedule, seed, max length, steps | r16/α32, 2e-4 cosine, seed 42, 4,096, 210 steps | same |
| Checkpoint selection | lowest eval loss every 30 steps, on 200 rows of the old validation pool | same rule, on all 237 `val_select` rows (215 + 22 hand-verified negatives) |
| Checkpoints kept | last 3 (150/180/210) | all 7 (disk only; no effect on training) |
| Data order | seed-42 shuffle of v1's file | seed-42 shuffle of a different file, so the order differs |

**Evaluation (fixed now).** `finqa-grouped-v2/val_dev_eval` (254 answerable + 40 hand-verified negatives), same
vLLM protocol as the v1/base runs (default serve command, one request at a time, answer-v4), paired against
`20261107-v2dev-v1` and `20261107-v2dev-base`. Reported together, with group-resampled 95% CIs:
answerable accuracy, abstention recall on the 40 verified negatives, false abstention on answerable items, plus
invalid output and evidence-validation failure.

**Known asymmetry.** v1's original selection rows touched 49 of the 66 `val_dev` filing groups; v2a's selection
(`val_select`) touched none. Any v2a-vs-v1 difference on `val_dev` is measured with that asymmetry, which, if
anything, favours v1. The exposed/unexposed split is reported for both.

**Reading the result.** 40 negatives give wide intervals (±13 points at best): a change in abstention recall smaller
than about 20 points will not be distinguishable from noise. A drop in answerable accuracy beyond the CI is a
cost to report, not to explain away. Wrong-period negatives are a separate, later comparison.
