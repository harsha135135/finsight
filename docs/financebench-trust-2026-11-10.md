# Which answers can users trust? FinanceBench development, frozen interface v1.6 (2026-11-10)

Frozen interface (`evals/financebench/freeze-qa-v1.6.json`): prompt finsight-qa-v1.5, validator v1.6 (requested-year
coverage), identical inputs for all models (hash-checked). Local outputs were rescored, not regenerated; Claude Haiku 5.5
ran once through the Message Batches API on the same inputs ($0.0607 actual; worst case $0.2494 reserved under the
$0.25 cap; all 150 succeeded, none truncated). Held-out half and FinQA test not read. Development data only.

Each cell: **correct (any) / n · correct / accepted (strict) · correct / accepted (lenient) · unsupported · declined**.
"Accepted" means the validator's checks passed; it is not correctness. "Unsupported" = answered but rejected.

| Mode | Model | Numeric (33) | Boolean (20) | Narrative (22) |
|---|---|---|---|---|
| Supplied | base | 12 · 9/15 · 12/18 · 15 · 0 | 13 · 11/15 · 11/15 · 4 · 0 | 9 · 7/15 · 7/16 · 5 · 0 |
| Supplied | v2a | 11 · 8/22 · 11/28 · 5 · 0 | 8 · 7/13 · 7/14 · 5 · 0 | 7 · 2/11 · 2/12 · 10 · 0 |
| Supplied | v2b | 7 · 3/20 · 7/27 · 6 · 0 | 12 · 11/16 · 11/17 · 2 · 0 | 5 · 4/17 · 4/18 · 4 · 0 |
| Supplied | **Haiku** | **27 · 20/24 · 27/31** · 1 · 0 | **18 · 13/14 · 13/15** · 5 · 0 | **17 · 12/15 · 12/15** · 5 · 1 |
| Full doc | base | 1 · 0/3 · 1/6 · 25 · 2 | 7 · 7/14 · 7/14 · 2 · 0 | 2 · 2/13 · 2/14 · 5 · 0 |
| Full doc | v2a | 3 · 1/8 · 3/19 · 8 · 6 | 3 · 2/12 · 2/13 · 5 · 1 | 2 · 1/11 · 1/12 · 8 · 1 |
| Full doc | v2b | 2 · 0/9 · 2/17 · 12 · 4 | 13 · 8/14 · 8/14 · 5 · 1 | 1 · 1/14 · 1/15 · 5 · 2 |
| Full doc | **Haiku** | 8 · **2/2 · 8/8** · 2 · **23** | 7 · **5/5 · 5/5** · 2 · **13** | 7 · **7/10 · 7/10** · 2 · **10** |

Narrative: blind rubric; Haiku's outputs were graded mixed with 30 already-graded local outputs as decoys (agreement
29/30; the one difference was an item first graded from a truncated view).

**Period validator (native FinQA controls, coverage rule applied to saved outputs).** Wrong-year answers passing:
0 / 0 / 0 (base, v2a, v2b). Correct answers rejected for their period: 0 / 0 / 0. Correct answers rejected only in strict
mode because a figure's period could not be placed: 4 / 32 / 32 of 9 / 87 / 85. On FinanceBench, gold-valued numeric
answers rejected for their period: 0 for every model; strict-only rejections (unplaceable period): base 3, v2a 3,
v2b 4, Haiku 7 (supplied).

**Reading.**
- With the right evidence supplied, the pipeline supports correct answers when the model is strong enough: Haiku answers
  27 of 33 numeric questions correctly, and 20 of its 24 strictly accepted numeric answers are correct. The local
  models' accepted answers are much less reliable (base 9/15, v2a 8/22, v2b 3/20 strict). The remaining limitation with
  supplied evidence is the local models, not the evidence or validation pipeline.
- With whole documents, retrieval is the limitation: Haiku declines most questions whose evidence was not retrieved
  (23 of 33 numeric), and the answers it does give are accepted-and-correct (8/8 numeric, 5/5 boolean lenient).
- Acceptance is a check, not a guarantee: even for Haiku, 4 of 24 strictly accepted numeric answers are wrong (correct
  figures from the right year combined into the wrong metric is possible and is not caught by the validator).
