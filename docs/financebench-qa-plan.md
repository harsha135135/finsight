# FinanceBench under a versioned answer contract: base, v2a, v2b (written before generation, 2026-11-09)

Supersedes the answer-v4 FinanceBench evaluation as the comparison of record; those results are kept unchanged
(`docs/financebench-dev-2026-11-09.md`). No training, no Claude calls; the FinQA test set and FinanceBench's held-out
half are not read. No configuration is selected here.

**Contract `finsight-qa-v1`** (`finsight/contracts/qa.py`, validator `finsight/qa.py`, prompt `finsight-qa-v1.0`):
numeric (cited operands + program; the value is the calculator's result), boolean (yes/no + cited claims), narrative
(cited claims; every figure in a claim must be in its cited passages), insufficient_evidence. Flat schema, evidence
fields before the answer type. answer-v4, finsight-report-v1 and `validate_finding` are unchanged.

**Period check** (numeric operands, against the source): pass / reject / unverified (the source gives no year).
Reported strict (unverified = rejected) and lenient (unverified accepted, counted). Requested years from the question by
a fixed rule (hand-checked on all 75; ranges expanded). Column dates mapped to fiscal years with each company's
fiscal-year end (`evals/financebench/fiscal-years-dev-v1.json`; all development 10-Ks name the fiscal year by its end
year). Limitations: quarter vs year-to-date columns are not distinguished; start-year fiscal naming is not handled.

**Extraction `pdf-2`** (`finsight.ingestion.pdf.parse_pdf_v2`): numeric table rows under a year/date header become
`label | 2018: v | 2017: v` passages (cell count must equal header count; otherwise left as text, so its figures are
unverified, never guessed). Alignment audit (`alignment-audit-dev-v1.json`): for 23 of 31 numeric questions with named
line items, every line item is a pdf-2 row with all requested years; failures are layouts needing character
positions (Adobe column-separated text, Walmart FY2019 one word per line, General Mills split headers). Tuned on
development.

**Evidence.** Supplied mode now uses FinSight's own pdf-2 extraction of the gold pages (changed from FinanceBench's
`evidence_text`, which drops spaces and splits figures across lines, caused format-only rejections and cannot be read
by the period check); supplied and full-document modes therefore differ only by retrieval. Full document: text-2 over
pdf-2 passages, up to 25 passages within 8,000 characters (limit chosen on development), frozen per question in
`retrieval-dev-pdf2-text-2.json`: all gold pages retrieved for 25 questions, some for 5, none for 45.

**Gold** (`evals/financebench/gold-dev-v1.json`, frozen before any output): expected type per question (33 numeric,
20 boolean, 22 narrative, by hand), numeric value/unit/decimals/sign policy (magnitude accepted only for outflows shown in
parentheses: 3M capex, Activision capex % of revenue, General Mills dividend payout), boolean polarity, requested
years; one disputed gold (Pfizer "spinning off Upjohn as of Q2 2023"), reported with and without.

**Scoring.** Numeric: the calculator's value (base units) converted to the gold's unit; correct within 1% relative or
at the gold's displayed decimals; fraction accepted for percent/percentage-point golds; scale slips counted separately;
correct only if the validator accepts the answer (strict and lenient reported). Boolean: polarity + validator
acceptance. Narrative: blind rubric review (correct / partial / incorrect; model and mode hidden). Answering with the
wrong type, insufficient evidence, invalid output and truncation are counted separately. Full document: correctness
crossed with gold pages retrieved (all / some / none). Paired base vs each tuned model, company-clustered; with these
sizes, differences under roughly 15 points cannot be told apart.

**Period-validator controls with native outputs** (FinQA, same contract): negative = the 193 frozen wrong-period
items; positive = the original questions behind them. FinQA rows are given in the same `label | year: value` form the
validator reads. Reported: catch rate (negative), false-rejection rate among correct native answers (positive),
unverified rate; FinanceBench's correct numeric answers serve as a second positive control on real filings.

**Serving.** One vLLM server with v2a and v2b, `--compact-json`, `--max-model-len 6656`, thinking off for all three
models, `max_tokens` 1,536, one request at a time. Over-length or truncated outputs count as operational failures.
Order: smoke test on unscored inputs → native period controls → supplied (base, v2a, v2b) → full document.
