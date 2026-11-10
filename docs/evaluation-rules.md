# Evaluation rules for the sealed runs (frozen 2026-11-10, before any held-out data is opened)

These rules apply to the FinanceBench held-out half and the FinQA test pool. The code that implements them is fixed
and tested on development or synthetic data. After authorization, held-out preparation only feeds data into these
procedures; it needs no new evaluation logic.

## Opening order
1. Commit the final configuration as a freeze file: prompt, schemas, validator and scorer version, retrieval version,
   serving settings, model revisions, Haiku settings, and the code hashes.
2. You authorize opening the held-out half in chat. The authorization is committed as
   `evals/authorizations/financebench-heldout.json` and names the freeze file and its sha256.
   `held_out_questions()` refuses to return anything until both files are committed and unchanged.
3. Prepare the held-out inputs with the frozen procedures:
   - fiscal years: `modelforge/fiscal.py`;
   - gold labels;
   - document ingestion and retrieval: `modelforge/fb_retrieval.py` with the frozen version;
   - exclusions, final denominators and the FinQA test label audit.

   Their hashes go into a second file that cites the freeze's sha256, committed **before any inference**.
4. Run each model once. Score once with the frozen scorer. There are no reruns, no fallbacks to another model and no
   changes afterwards. An unknown paid outcome stays unknown.

## Fiscal years (`modelforge/fiscal.py`)
- Read each held-out company's 10-K cover pages and collect the "for the (fiscal) year ended <date>" dates. The
  fiscal-year end is the earliest (month, day) of the smallest window holding all of them; the window may span at most
  7 days, which is the validator's tolerance. A wider spread stops for a person to resolve.
- A company with no 10-K in the corpus gets null, which means calendar years. It is listed for a manual look.
- Check on development data: the rule gives the same fiscal year as the hand-made file for every cover date of all
  16 development companies (`evals/financebench/fiscal-years-dev-derived-v1.json`).
- The scorer raises on a company missing from the fiscal-year files. It never falls back silently.

## Exclusions and denominators
- **FinanceBench.** The headline denominator is every held-out question. A question leaves the denominator only if
  its source document is missing or unparseable; such questions are listed and coverage is reported.
  - Invalid output, operational failures, declines and validation rejections all stay in the denominator, as not correct.
  - A gold label found doubtful while building the gold (before inference) is recorded as disputed with its reason.
    It stays in the headline, and a sensitivity line without the disputed items is reported beside it.
- **FinQA test pool.** Every answerable item counts. An unanswerable variant counts only if the label audit verified it
  (`verified_unanswerable`); derivable and ambiguous variants are excluded and listed (`audit.final_denominators`).
  The audit decisions and the resulting denominators are committed before inference. An undecided variant blocks
  freezing.

## Headline measures
- Per answer type (numeric, boolean, narrative):
  - correct overall;
  - correct among accepted answers, in **strict** validation (the headline) and lenient;
  - coverage (accepted answers);
  - unsupported (answered but rejected), declined and invalid.
- Numeric answers are scored by the frozen scorer (v1.7 rules), within 1% or at the gold's stated rounding.
  Scale slips (right digits, wrong scale) count as wrong.
- Paired comparisons use a bootstrap that resamples companies (FinanceBench) or filings (FinQA). Exact McNemar is
  shown as well.
- Retrieval (full-document mode) gets the page coverage (all, some or none of the gold pages) and the evidence audit:
  whether the needed figures or lines reached the model. Held-out needles come from the same rule
  (`fb_retrieval.evidence_needles`).

## Narrative grading (`modelforge/grading.py`)
- **Rubric (frozen):** *correct*: the conclusion and key facts match the gold. *partial*: the right direction but
  incomplete or with one error. *incorrect*: wrong or no conclusion, or a figure that does not answer a which/what/why
  question.
- **Partial** answers never count as correct in a headline number; they are reported separately.
- **Blinding:** all models' narrative answers go on one shuffled sheet with blind ids. The grader sees the question,
  the gold and the answer's claims and figures, never the model, the validation status or other grades. About one row
  in five is a previously graded development answer (a decoy), so the grader's consistency is measured.
- **Reviewer:** independent human review is preferred for the headline narrative results: the sheet is a CSV for a
  person to fill in. If a Claude model grades instead, the results say "Claude graded Claude's answers" and report the
  decoy agreement.
- The verdicts are committed before the key is applied.

## Claude reference runs
- Haiku 5.5 through the Message Batches API: the same frozen inputs as the local models, `max_tokens` 5600, effort
  medium (the 2026-11-10 development settings).
- Exact input tokens are counted before submission. The run is capped (the worst case, if every request used its full
  allowance, must fit) and needs your approval of the estimate.
