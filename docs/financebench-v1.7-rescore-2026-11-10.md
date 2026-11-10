# FinanceBench development: validator and scorer v1.7 (2026-11-10)

Saved development outputs were rescored, not regenerated. The frozen v1.6 numbers stay recorded
(`docs/financebench-trust-2026-11-10.md`). Both versions regenerate from saved predictions:
`uv run modelforge record verify evals/records/financebench-dev-trust-v17.json` reports `scores: regenerated`.

## Changes, applied one at a time

| Step | Kind | Change |
|---|---|---|
| 1 | scorer bug fix | an explicit gold unit ("USD millions") sets the scale when the question states none |
| 2 | parsing bug fix | numbers in claims are digit groups; trailing punctuation ("14,694,") is no longer part of the number |
| 3 | **policy change** (broadens acceptance) | in claims, the day of a calendar date ("December 31, 2022") and numbered document references ("Item 9.01", "Note 6", "Exhibit 21.1") are not treated as financial figures; only digits inside such a span are exempt (tested: "December 31, 2022 revenue was 31" still checks the 31) |

Not changed: a dash ("—") is still not accepted as the figure zero; derived figures stated in prose still fail the claim check.

Fix after review: the first v1.7 date pattern matched any word beginning with a month prefix ("margin 22%", "decreased
12"), which would have exempted ordinary figures. It now matches only real month names and abbreviations ("Dec.",
"September", "Sept"), and tests check that "margin 22%", "decreased 12", "declined 5" and "market 30" are still
checked. Rescoring with the corrected pattern changed no table, transition or control result on the development data.

## Effect of each step (changes in acceptance, by correctness)

| Step | Correct newly accepted | Incorrect newly accepted | Where |
|---|---|---|---|
| 1 scorer fix | — | — | one answer re-scored correct for base and for Haiku (supplied) |
| 2 parsing fix | 3 (Haiku supplied) | 2 (v2a supplied, base full) | strict and lenient alike |
| 3 policy | 5 (Haiku supplied) | 1 (v2a supplied) | strict and lenient alike |

Period controls (FinQA, 193 + 193) are unchanged by all three steps: wrong-period answers accepted strict 0 / 0 / 0,
lenient 14 / 57 / 18 (base, v2a, v2b).

## v1.7 tables (correct overall · correct/accepted strict · correct/accepted lenient)

| Mode | Model | Numeric (33) | Boolean (20) | Narrative (22) |
|---|---|---|---|---|
| Supplied | base | 13 · 10/15 · 13/18 | 13 · 11/15 · 11/15 | 9 · 7/15 · 7/16 |
| Supplied | v2a | 11 · 8/22 · 11/28 | 8 · 7/14 · 7/15 | 7 · 2/12 · 2/13 |
| Supplied | v2b | 7 · 3/20 · 7/27 | 12 · 11/16 · 11/17 | 5 · 4/17 · 4/18 |
| Supplied | Haiku | 28 · 21/24 · 28/31 | 18 · 17/18 · 17/19 | 17 · 16/19 · 16/19 |
| Full doc | base | 1 · 0/3 · 1/6 | 7 · 7/14 · 7/14 | 2 · 2/14 · 2/15 |
| Full doc | v2a | 3 · 1/8 · 3/19 | 3 · 2/12 · 2/13 | 2 · 1/11 · 1/12 |
| Full doc | v2b | 2 · 0/9 · 2/17 | 13 · 8/14 · 8/14 | 1 · 1/14 · 1/15 |
| Full doc | Haiku | 8 · 2/2 · 8/8 | 7 · 5/5 · 5/5 | 7 · 7/10 · 7/10 |

Also new in the scorer: fiscal-year tables load from a list of files, and an unknown company raises instead of falling
back to calendar years (needed before any held-out scoring). Record verification now reports "hashes verified" and
"scores regenerated" separately; the older FinanceBench records are hash-only.
