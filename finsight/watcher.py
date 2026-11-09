"""Filing watcher: polls SEC EDGAR for each watchlist company and records new filings once (by accession).

The first check of a newly added ticker records all its matching filings as a baseline (not "new"), so only
filings that appear later are new. Later checks record
filings that appeared since, and - only when the ticker has auto-analysis switched on - create a research job for a
new 10-K with local Qwen (the research pipeline analyses annual reports), idempotent per accession and metric set. Requests share FinSight's Postgres rate
limit and use the SEC contact from FINSIGHT_SEC_USER_AGENT.

    uv run python -m finsight.watcher [--interval 900] [--once]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

from finsight.ingestion.sec import Filing, PostgresRateLimiter, SecSource, SourceError
from finsight.settings import get_settings
from finsight.storage import jobs
from finsight.storage.db import connect

AUTO_METRICS = ["revenue", "operating_margin", "capex", "net_income"]


def _fiscal_period(filing: Filing) -> str | None:
    return f"FY{filing.report_date.year}" if filing.report_date else None


def check(conn, source: SecSource, ticker: str, *, today: date | None = None, auto_model: str | None = None) -> dict:
    today = today or date.today()
    row = conn.execute("SELECT * FROM watchlist WHERE ticker = %s", (ticker,)).fetchone()
    first = row["last_checked_at"] is None
    filings, _ = source.filings(row["cik"])
    new, jobs_created = [], []
    for f in sorted(filings, key=lambda f: f.filing_date):
        if f.form not in row["forms"]:
            continue
        inserted = conn.execute(
            "INSERT INTO filings_seen (accession, ticker, cik, form, filing_date, report_date, primary_document, url, baseline) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (accession) DO NOTHING RETURNING accession",
            (f.accession, ticker, f.cik, f.form, f.filing_date, f.report_date, f.primary_document, f.url, first)).fetchone()
        if inserted is None or first:
            continue
        new.append(f.accession)
        if row["auto_analyze"] and auto_model and f.form == "10-K" and _fiscal_period(f):  # the pipeline analyses annual reports
            job = create_analysis(conn, ticker, f, auto_model)
            jobs_created.append(str(job["id"]))
    conn.execute("UPDATE watchlist SET last_checked_at = now(), last_error = NULL, name = coalesce(name, %s) WHERE ticker = %s",
                 (source.names.get(row["cik"]), ticker))
    return {"ticker": ticker, "baseline": first, "new": new, "jobs": jobs_created}


def create_analysis(conn, ticker: str, filing: Filing, model: str, metrics: list[str] | None = None) -> dict:
    metrics = metrics or AUTO_METRICS
    request = {"company": ticker, "question": f"Report {', '.join(metrics)} for {ticker} {_fiscal_period(filing)} from its {filing.form}.",
               "period": _fiscal_period(filing), "metrics": metrics, "cutoff_date": None}
    config = {"provider": "qwen", "model": model, "base_url": get_settings().qwen_base_url, "max_tokens": 2048,
              "enable_thinking": False, "contract": "finsight-report-v1"}
    job = jobs.create_job(conn, request, config, f"filing:{filing.accession}:{','.join(sorted(metrics))}")
    conn.execute("UPDATE filings_seen SET job_ids = job_ids || %s::jsonb WHERE accession = %s AND NOT job_ids @> %s::jsonb",
                 (json.dumps([str(job["id"])]), filing.accession, json.dumps([str(job["id"])])))
    return job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=900.0, help="seconds between rounds")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    conn = connect(settings.database_url, application_name="finsight-watcher")
    if not settings.sec_user_agent:
        print("watcher: FINSIGHT_SEC_USER_AGENT is not set; nothing to do", file=sys.stderr)
        return 2
    source = SecSource(settings.sec_user_agent, settings.state_dir / "artifacts", conn=conn, rate_limiter=PostgresRateLimiter(conn))
    while True:
        for row in conn.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall():
            try:
                print(json.dumps(check(conn, source, row["ticker"], auto_model=settings.watcher_model)), flush=True)
            except (SourceError, OSError) as exc:
                conn.execute("UPDATE watchlist SET last_checked_at = now(), last_error = %s WHERE ticker = %s",
                             (f"{type(exc).__name__}: {exc}"[:300], row["ticker"]))
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
