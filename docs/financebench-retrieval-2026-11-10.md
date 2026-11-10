# FinanceBench full-document retrieval: bounded revision (development data, 2026-11-10)

**Reproducibility first.** The frozen text-2 retrieval now regenerates from committed code
(`modelforge/fb_retrieval.py`): 75 of 75 passage lists identical, coverage 25 / 5 / 45; the 40 development PDFs
re-parse to the stored pdf-2 passages (ids and text).

**Evidence, not just pages.** A fixed audit (`evals/financebench/evidence-audit-dev-v1.json`, written before any
change was measured) lists per question the figures with their row labels and years, or the answer-bearing lines of
the gold evidence, that must reach the model inside one retrieved passage. Under text-2, 11 of the 25 questions whose
gold pages were all retrieved received none of the needed evidence: the page's header chunk arrived, not the row.

| Configuration | Gold pages all / some / none | Any gold page | Evidence complete / partial / none | Needles reached (of 239) |
|---|---|---|---|---|
| pdf-2 + text-2 (frozen) | 25 / 5 / 45 | 0.40 | 18 / 11 / 46 | 70 |
| **pdf-2 + text-3 (adopted)** | **34 / 6 / 35** | **0.53** | **26 / 17 / 32** | **97** |
| pdf-3 + text-3 (one revision; not adopted) | 36 / 6 / 33 | 0.56 | 28 / 14 / 33 | 96 |

text-3 changes, chosen from the failure analysis rather than per question: the document's own company name and the
standard instruction tails are dropped from the query; FY20 / Q2 FY2023 forms normalise to years; a few finance
synonyms are added and "working capital" no longer expands to capital expenditure; ranking is length-normalised;
exhibit-index and subsidiary-list pages are ranked down. No question category got worse by more than one question.
pdf-3 (statement titles on table rows) is a parser change; its page gain did not carry into evidence coverage, so the
simpler configuration is kept. All of this is tuned on these 75 questions; the held-out half gives the honest number.

**Effect on answers (base Qwen3-4B, full-document mode, scorer v1.7; only the retrieved passages differ).**

| Retrieval | Numeric correct / correct-and-strictly-accepted | Boolean correct / correct-and-strictly-accepted |
|---|---|---|
| text-2 | 1 of 33 / 0 of 3 accepted | 7 of 20 / 7 of 14 accepted |
| text-3 | 3 of 33 / 1 of 5 accepted | 13 of 20 / 12 of 19 accepted |

Narrative answers were not graded for this comparison. Retrieval still limits: 32 questions get none of the needed
evidence, and multi-page evidence (15 questions) is almost never complete.
