# Retrieval text-2 vs text-3 with Haiku (FinanceBench development, full documents, 2026-11-10)

One approved Haiku 5.5 batch ran with the adopted retrieval (pdf-2 + text-3), using the same item bytes base Qwen got
(`evals/financebench/freeze-dev-text3-v1.json`) and the unchanged settings: effort medium, `max_tokens` 5600, Message
Batches. 75 of 75 requests succeeded, with none truncated or refused and no retries. **Actual cost $0.0319** (worst case
$0.1264 reserved under the $0.15 cap). Total Claude spend is now $0.2057 of $2.

Same 75 questions, same prompt, contract and scorer (v1.7); only the retrieved passages differ. Narrative answers from
both Haiku runs were graded together on one blind sheet with the frozen procedure (`modelforge/grading.py`), by Claude:
Claude graded Claude's answers. Decoy agreement was 6/6, and the regraded text-2 answers matched the earlier grades 12/12.
This confirms the adopted retrieval; it is not a new tuning round. Development data; the held-out sets are still
sealed.

## Haiku

Each cell: correct overall · correct among strictly accepted · correct among leniently accepted · declined (abstained) · unsupported · invalid.

| Retrieval | Numeric (33) | Boolean (20) | Narrative (22) |
|---|---|---|---|
| text-2 | 8 · 2/2 · 8/8 · 23 · 2 · 0 | 7 · 5/5 · 5/5 · 13 · 2 · 0 | 7 · 7/10 · 7/10 · 10 · 2 · 0 |
| text-3 | 10 · 2/5 · 10/13 · 18 · 1 · 1 | 9 · 8/11 · 8/11 · 8 · 1 · 0 | 8 · 8/12 · 8/12 · 8 · 2 · 0 |

**Paired, all 75 questions** (text-3 minus text-2; 95% interval from a bootstrap resampling companies; exact McNemar):

| Measure | text-2 | text-3 | Difference | 95% interval | McNemar (text-3 only / text-2 only, p) |
|---|---|---|---|---|---|
| Correct | 22 | 27 | +6.7 points | 0.0 to +12.7 | 7 / 2, p = 0.18 |
| Correct and strictly accepted | 14 | 18 | +5.3 points | 0.0 to +10.8 | 6 / 2, p = 0.29 |

By type, correct: numeric 8 → 10 (interval −5.6 to +18.2), boolean 7 → 9, narrative 7 → 8. None of these is
statistically clear on its own.

**Coverage against accepted-answer accuracy.** With text-3, Haiku declines less (46 → 34 declines across types) and
the strictly accepted answers double (14 → 28). Their accuracy falls from 14/14 to 18/28. Under text-2 it answered
mostly when the evidence was clearly present, and was right each time. Under text-3 it answers more questions,
gets more right in total, and more wrong answers pass the validator.

**Questions that changed**, with evidence-audit coverage text-2 → text-3:
- **Improved (7):**
  - Adobe operating margin (boolean): declined → correct; evidence partial → complete.
  - Amcor key agenda (narrative): declined → correct.
  - Amcor adjusted EBITDA (numeric): wrong → correct. Only the correct value now passes lenient validation.
  - AmEx operating-margin relevance (narrative): wrong → correct.
  - Corning effective tax rate (numeric): declined → correct.
  - Netflix current liabilities (numeric): declined → correct; evidence none → complete.
  - Pfizer (boolean): declined → correct; evidence none → complete.
- **Regressed (2):**
  - Amcor restructuring (narrative): correct → incorrect, with the same complete evidence.
  - Block cash from operations (numeric): correct → wrong; evidence none → complete, but a different figure was cited.
- **Abstention changed without a gain (9):** seven questions went from declined to answered wrong:
  - 3M PP&E: the known scale error;
  - 3M capital intensity;
  - Best Buy inventories;
  - Best Buy acquisitions: graded partial;
  - Best Buy cash flows: graded partial;
  - Corning working capital;
  - Pfizer/Upjohn: gold disputed.

  Two more: one Corning question went from declined to invalid output, and one Boeing question from wrong to declined.

## Base Qwen3-4B (same comparison; narrative answers not graded for Qwen)

| Measure (numeric + boolean, 53) | text-2 | text-3 | 95% interval | McNemar |
|---|---|---|---|---|
| Correct | 8 | 16 | +6.0 to +24.6 points | 8 / 0, p = 0.008 |
| Correct and strictly accepted | 7 | 13 | +1.8 to +20.4 points | 7 / 1, p = 0.07 |

## Reading
- text-3 helps the local model clearly and the strong model modestly. Neither result changes the adopted
  configuration, and no further retrieval tuning follows from it.
- For the application: better retrieval makes Haiku attempt more questions. Some newly attempted answers are wrong
  and still pass validation, so in full-document mode coverage rose and accepted-answer accuracy fell. The
  strict-accepted accuracy is the number to quote with any coverage figure.
- Retrieval remains a limit: 32 of 75 questions still get none of the needed evidence.
