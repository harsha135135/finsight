# FinanceBench development evaluation: v2a vs v2b (written before any generation, 2026-11-09)

**Scope.** The 75 development questions of `financebench-v1` (16 companies incl. 3M). Held-out ids and companies are
asserted out of every code path. No training, no Claude calls. No configuration is selected here; this reports the
comparison.

**Contract and serving.** FinQA protocol `answer-v4` with the `as_written` calculation convention in both modes (the
format v2a/v2b were trained on: this is a transfer test). It can express a number or yes/no plus cited claims, so
numeric (31) and narrative (44) results are reported separately, never pooled. One vLLM server with both adapters,
default FinQA serve settings except `--max-model-len 6144` (8,192 did not fit); one request at a time; `max_tokens`
1,024. Truncated or over-length outputs count as operational failures.

**Modes.**
- *Supplied evidence*: FinanceBench's annotated `evidence_text` for the question (1-3 spans), split into passages of up
  to 700 characters by line.
- *Full document*: FinSight's own extraction (`pdf-1`) and general retrieval `text-2` (question text only, the question's
  own document, 10 passages / 8,000 characters), frozen per question in `evals/financebench/retrieval-dev-text-2.json`.
  Page recall: 0.333 untuned (text-1), **0.440 tuned on development** (text-2); the held-out half gives the honest
  number.

**Numeric scoring (31 questions; fixed now).** Gold = the first number in the gold answer, in the gold's unit (a gold
with % is a percentage). The model's value = the validator's executed answer, else its stated answer. Correct when
within 1% relative of gold, or equal after rounding to the gold's displayed decimals; a fraction is accepted for a
percentage gold (×100). Scale slips (×10^3, ×10^6) are not correct; they are counted separately. Primary: correct and
accepted by the validator; secondary: stated answer correct regardless of validation.

**Narrative rubric (44 questions; fixed now).** Each answered output (answer + cited claims) is judged against the
gold answer and justification: *correct* (conclusion and key facts match, no material error), *partial* (right
direction but incomplete or one minor error), *incorrect* (wrong conclusion or facts, or unsupported). Abstained and
invalid are counted separately. Review is blind: model and mode labels stripped, order shuffled, every answer read.
Disputed gold labels are decided before unblinding and scores reported with and without them.

**Metrics.** Correctness (as above), citation support (validator evidence checks pass; full-document: a cited passage
is on a gold page), false abstention (every question is answerable), retrieval failure (no gold page retrieved),
latency p50/p95; all by question type.

**Error stages (first failing stage, in this order).** extraction (gold page missing or empty in our text) →
retrieval (no gold page retrieved; full-document only) → output/operational (invalid, truncated) → false abstention
→ validation rejection → wrong answer. Extraction check already done: 92/92 evidence pages present, number recall
0.999 (`evals/financebench/extraction-check-dev-v1.json`).

**Statistics.** v2a vs v2b paired on identical items, group-resampled by company. With 31 numeric and 44 narrative
questions, intervals will be roughly ±12-18 points; smaller differences cannot be told apart.
