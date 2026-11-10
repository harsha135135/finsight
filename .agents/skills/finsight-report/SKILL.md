---
name: finsight-report
description: Build a sourced financial brief on a public company and/or commodities with the FinSight engine (SEC filings and XBRL as originally reported, plus FRED market and macro series). Use when asked for a company or market report, a trend analysis, or "what's happening with <ticker> / <commodity>". Every figure in the report is traced to the engine's context pack and checked before the report is delivered.
---

# FinSight report

FinSight builds a **context pack** (`pack.json`): company figures from SEC filings as originally reported (latest
quarter, year-over-year from the same filing, trailing twelve months, operating margin, capex, recent filings) and
market series from FRED (latest value and date, 1-month/1-year change, 52-week range, trend), each with its source
and as-of date. You write the report from that pack; a checker rejects any number that is not in it.

Run everything from the FinSight repository root (the directory containing `pyproject.toml` with `finsight`).

## 1. Build the pack

```bash
uv run finsight commodities                      # market series keys
uv run finsight brief --company MMM --commodity copper --commodity wti --out reports/MMM-<date>
```

- `--company` takes a ticker or CIK; `--commodity` is repeatable. Choose series that matter for the company's industry
  (`pack.json` → `company.industry`): energy and airlines → `wti`, `brent`, `natgas`; industrials and electrical →
  `copper`, `aluminum`; food → `wheat`, `corn`; add `ust10y`, `cpi` or `usd` for macro context when useful.
- For a commodity-only brief, omit `--company`.
- Needs `FINSIGHT_SEC_USER_AGENT` (company data) and `FINSIGHT_FRED_API_KEY` (market data) in `.env`. If `pack.json`
  lists `problems`, tell the user what is missing instead of filling the gap.

## 2. Write `report.md` in the same folder

Read `pack.json` and `draft.md` (a deterministic draft with every insight and source). Then write a clearer report:

1. **Summary**: three to five sentences on the current picture.
2. **Company performance**: revenue, margin, capex, trailing twelve months, recent filings.
3. **Market context**: the selected series: level, recent change, position in the 52-week range, trend.
4. **What to watch**: open questions the data raises (upcoming filings, a series near its range edge), phrased as
   questions, not predictions.
5. **Sources**: accession numbers and FRED series ids from the pack, with as-of dates.

Rules:
- Every number must come from the pack, written as it appears in an insight's text or `figures` (for example
  `$6.58 bn`, `+1.0%`, `23.7%`). Do not compute new figures, round differently, or convert units.
- Give each figure its as-of date. Monthly FRED series are published with a lag; say so. Label derived fourth quarters.
- A company and a commodity moving together is co-movement, not causation: do not write that one drives the other.
- No investment advice, price targets or recommendations.
- Optional outside context (news, management commentary) may come from a web search, with a link, as qualitative
  statements only: no numbers from outside the pack.

## 3. Check, then deliver

```bash
uv run finsight check-report reports/MMM-<date>/report.md
```

If it lists numbers not found in the pack, remove or correct them and run it again until it passes. Then give the user
the report and its path, and state how fresh the data is (latest filing date, latest observation per series).

FinSight itself makes no model calls here: the writing happens in your Codex session, so it is not metered by
FinSight's spend ledger.
