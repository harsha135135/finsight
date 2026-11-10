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

Narrative: blind rubric; Haiku's outputs were graded mixed with 30 already-graded local outputs as decoys, with model
labels hidden. Re-grading agreement on the decoys was 29/30; the one difference (an Amcor restructuring answer from base,
full-document mode: incorrect before, partial now) is left as originally recorded in the local scores. The blinding is
weak: Haiku's answers are longer and more specific in style and are often recognizable, so the narrative comparison
carries grader bias that the decoys measure only partly. The numeric and boolean results are scored deterministically.

Paired test, supplied numeric (exact McNemar on the same 33 questions): Haiku correct where base is not 15, the reverse
0 (p = 6e-5); vs v2a 16 / 0 (p = 3e-5); vs v2b 20 / 0 (p = 2e-6). Development data, 33 questions.

**Period validator: strict and lenient differ sharply (native FinQA controls, coverage rule applied to saved outputs).**
- Negative controls (the requested year's evidence removed, another year's left; 193 items). Numeric answers accepted:
  strict 0 / 0 / 0; lenient **14 of 171, 57 of 162, 18 of 52** answered (base, v2a, v2b). Lenient mode accepts a figure
  whose period the source does not name, so it does not protect against wrong-period answers; only strict mode does.
  ("0 escapes" in an earlier draft counted only answers whose every figure was placed.)
- Positive controls: correct answers rejected for their period 0 / 0 / 0; correct answers rejected only in strict mode
  because a figure's period could not be placed 4 / 32 / 32 (of 9 / 87 / 85 correct).
- FinanceBench: gold-valued numeric answers rejected for their period: 0 for every model. Strict-only rejections
  (unplaceable period): base 3, v2a 3, v2b 4, Haiku 7 (supplied) and 6 (full).

**What the validator costs a strong model (Haiku, correct answers it rejected, by reason).**

| Reason | Supplied | Full doc |
|---|---|---|
| Numeric, strict only: a figure's period could not be placed | 7 | 6 |
| Claim check: number parsed with a trailing comma ("14,694," / "June 30,") — a parsing defect | 3 | 0 |
| Claim check: day of month in a date ("December 31, 2022") or a section number ("Item 9.01") | 5 | 0 |
| Claim check: a derived figure stated in prose (boolean answers carry no program) | 1 | 2 |
| Operand: a dash meaning zero ("—") not accepted as a figure | 1 | 0 |

**Haiku's accepted-but-wrong numeric answers (4, all accepted in both modes).**
- 3M PP&E: correct figure, but the program divided by 1,000 to express billions although the contract already scales
  figures — a scale error the interface invites.
- General Mills free cash flow: subtracted a capex figure printed in parentheses, so 3,676.2 − (−460.8) — correct figures,
  wrong calculation.
- Pfizer/Upjohn: cited total separation costs ($700 million) where the gold expects 77.78 (a different quantity).
- Amcor adjusted EBITDA: actually correct ($2,018 million) — scored wrong by a scorer defect (below).

**Defects found after the freeze (not fixed; the frozen numbers above stand).**
- Scorer: an explicit gold unit ("USD millions") is ignored when the question states no scale (one item). With it
  honoured: supplied numeric base 13/33 (strict 10/15, lenient 13/18) and Haiku 28/33 (strict 21/24, lenient 28/31);
  other cells unchanged.
- Validator: trailing punctuation is kept on numbers in claims (3 correct Haiku answers rejected; local models also
  affected). Dates' day numbers and section numbers are treated as figures.
- Scale slips (right digits, wrong scale) count as wrong for every model: supplied base 1, v2a 5, v2b 5, Haiku 2
  (including the scorer defect). Counting them correct would not change the ordering (Haiku 29, v2a 16, base 13, v2b 12).

**Reading.**
- With the right evidence supplied, the local models are the main limitation: Haiku answers 27 of 33 numeric questions
  correctly (28 with the scorer defect fixed) against 12, 11 and 7, and is right on every numeric question any local
  model gets right (paired test above).
- The validator also costs real coverage on a strong model: strict mode rejects 7 of Haiku's 27 correct supplied numeric
  answers (strict coverage 24/33 vs lenient 31/33), and the claim check rejects 10 correct supplied boolean or narrative
  answers, 3 of them through a parsing defect. Lenient mode keeps that coverage but accepted 14–57 wrong-period answers
  in the controls, so neither mode is both safe and complete yet.
- With whole documents, retrieval is the limitation: Haiku declines most questions whose evidence was not retrieved
  (23 of 33 numeric), and the answers it does give are accepted-and-correct (8/8 numeric, 5/5 boolean, lenient).
- Acceptance is a check, not a guarantee: even for Haiku, 3 of 24 strictly accepted numeric answers are genuinely wrong
  (scale, sign, and a different quantity), none caught by the validator.
- Narrative grades for Haiku come from a Claude model grading another Claude model, on top of the style caveat above.
