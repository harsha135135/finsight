"""Endpoints and static files for the FinSight dashboard (served at /dashboard on 127.0.0.1).

The page never embeds the API key: it asks once, keeps it in sessionStorage and sends X-API-Key. Model text is
rendered as text, never HTML, and the page is served with a strict Content-Security-Policy (no third-party scripts).
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from finsight.retrieval.search import METRICS
from finsight.settings import get_settings
from finsight.spend import SpendLedger

STATIC = Path(__file__).parent / "static"
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
TREND_CONCEPTS = {"revenue": METRICS["revenue"]["concepts"], "operating_income": METRICS["operating_income"]["concepts"],
                  "net_income": METRICS["net_income"]["concepts"], "capex": METRICS["capex"]["concepts"]}


class WatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z.-]+$")
    auto_analyze: bool = False
    forms: list[Literal["10-K", "10-Q", "8-K"]] = Field(default_factory=lambda: ["10-K", "10-Q", "8-K"])


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=200)
    metrics: list[Literal["revenue", "operating_income", "operating_margin", "capex", "net_income"]] = Field(
        default_factory=lambda: ["revenue", "operating_margin", "capex", "net_income"], min_length=1)


def annual_series(facts: dict, concepts: list[str]) -> dict:
    """Full-year values per fiscal year as reported in each 10-K (the filing's own figure) plus later restatements."""
    for concept in concepts:
        entries = facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {}).get("USD", [])
        annual = []
        for e in entries:
            if e.get("form") not in ("10-K", "10-K/A") or not e.get("start"):
                continue
            days = (datetime.fromisoformat(e["end"]) - datetime.fromisoformat(e["start"])).days
            if 350 <= days <= 380:
                annual.append(e)
        if not annual:
            continue
        by_period: dict = {}
        for e in annual:
            by_period.setdefault(e["end"], []).append(e)
        points = []
        for end, group in sorted(by_period.items()):
            group.sort(key=lambda e: (e["filed"], e["form"] != "10-K"))
            first = group[0]                       # the earliest filed report of this period: the original figure
            later = sorted({e["val"] for e in group if e["val"] != first["val"]})
            points.append({"period_end": end, "value": first["val"], "accession": first["accn"], "form": first["form"],
                           "filed": first["filed"], "restated_values": later})
        return {"concept": concept, "unit": "USD", "points": points[-10:]}
    return {"concept": None, "unit": "USD", "points": []}


def register(app, pool, authorized) -> None:
    settings = get_settings()

    @app.get("/dashboard", include_in_schema=False)
    def page():
        return FileResponse(STATIC / "index.html", headers={"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff"})

    @app.get("/dashboard/{name}", include_in_schema=False)
    def asset(name: Literal["app.js", "app.css"]):
        return FileResponse(STATIC / name, media_type="text/javascript" if name.endswith(".js") else "text/css",
                            headers={"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff"})

    @app.get("/health", dependencies=[Depends(authorized)])
    def health() -> dict:
        out: dict = {"database": "ok"}
        with pool.connection() as conn:
            conn.execute("SELECT 1")
            beat = conn.execute("SELECT max(heartbeat_at) AS at, count(*) FILTER (WHERE status = 'running') AS running, "
                                "count(*) FILTER (WHERE status = 'queued') AS queued FROM research_jobs").fetchone()
            ledger = SpendLedger(conn, settings.spend_limit_usd)
            committed = ledger.committed()
            watched = conn.execute("SELECT count(*) AS n, max(last_checked_at) AS checked FROM watchlist").fetchone()
        out["jobs"] = {"running": beat["running"], "queued": beat["queued"], "last_heartbeat_at": beat["at"]}
        try:
            served = httpx.get(f"{settings.qwen_base_url.rstrip('/')}/v1/models", timeout=2.0).json()["data"]
            out["model_server"] = {"reachable": True, "models": [m["id"] for m in served]}
        except (httpx.HTTPError, KeyError, ValueError):
            out["model_server"] = {"reachable": False, "models": []}
        out["sec_contact_configured"] = bool(settings.sec_user_agent)
        out["spend"] = {"committed_usd": str(committed), "limit_usd": str(settings.spend_limit_usd),
                        "remaining_usd": str(Decimal(settings.spend_limit_usd) - committed), "claude_enabled": settings.allow_claude}
        out["watchlist"] = {"tickers": watched["n"], "last_checked_at": watched["checked"]}
        return out

    @app.get("/models", dependencies=[Depends(authorized)])
    def models() -> dict:
        try:
            served = [m["id"] for m in httpx.get(f"{settings.qwen_base_url.rstrip('/')}/v1/models", timeout=2.0).json()["data"]]
        except (httpx.HTTPError, KeyError, ValueError):
            served = []
        options = [{"provider": "qwen", "model": m} for m in served]
        if settings.allow_claude:
            options.append({"provider": "claude", "model": settings.model_analyze, "paid": True})
        return {"options": options, "note": "no configuration has been selected as the default yet"}

    @app.get("/research-jobs", dependencies=[Depends(authorized)])
    def list_jobs(limit: int = Query(default=25, ge=1, le=100), before: datetime | None = None) -> dict:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, status, stage, outcome_reason, request, model_config, created_at, finished_at FROM research_jobs "
                "WHERE (%s::timestamptz IS NULL OR created_at < %s) ORDER BY created_at DESC LIMIT %s",
                (before, before, limit)).fetchall()
        items = [{"id": str(r["id"]), "status": r["status"], "stage": r["stage"], "outcome_reason": r["outcome_reason"],
                  "company": r["request"].get("company"), "period": r["request"].get("period"),
                  "metrics": r["request"].get("metrics"), "question": r["request"].get("question"),
                  "model": r["model_config"].get("model"), "provider": r["model_config"].get("provider"),
                  "created_at": r["created_at"], "finished_at": r["finished_at"]} for r in rows]
        return {"items": items, "next_before": items[-1]["created_at"] if len(items) == limit else None}

    @app.get("/passages", dependencies=[Depends(authorized)])
    def passages(ids: str = Query(min_length=1, max_length=4000)) -> dict:
        wanted = [i for i in ids.split(",") if i][:50]
        with pool.connection() as conn:
            rows = conn.execute("SELECT p.id, p.section, p.location, p.text, d.source_uri, d.accession, d.form "
                                "FROM passages p JOIN documents d ON d.id = p.document_id WHERE p.id = ANY(%s)", (wanted,)).fetchall()
        return {"passages": rows}

    @app.get("/companies/{ticker}/trends", dependencies=[Depends(authorized)])
    def trends(ticker: str) -> dict:
        from finsight.ingestion.sec import PostgresRateLimiter, SecSource, SourceError
        if not settings.sec_user_agent:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "FINSIGHT_SEC_USER_AGENT is not set")
        with pool.connection() as conn:
            source = SecSource(settings.sec_user_agent, settings.state_dir / "artifacts", conn=conn, rate_limiter=PostgresRateLimiter(conn))
            try:
                cik = source.resolve_cik(ticker)
                cached = conn.execute("SELECT artifact_path, retrieved_at FROM source_snapshots WHERE kind = 'sec_companyfacts' "
                                      "AND source_uri LIKE %s AND retrieved_at > now() - interval '1 day' ORDER BY retrieved_at DESC LIMIT 1",
                                      (f"%CIK{cik}.json",)).fetchone()
                if cached and Path(cached["artifact_path"]).exists():
                    facts, retrieved = json.loads(Path(cached["artifact_path"]).read_text()), cached["retrieved_at"]
                else:
                    facts, snapshot = source.company_facts_snapshot(cik)
                    retrieved = snapshot["retrieved_at"]
            except SourceError as exc:
                raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        series = {metric: annual_series(facts, concepts) for metric, concepts in TREND_CONCEPTS.items()}
        return {"ticker": ticker.upper(), "cik": cik, "entity": facts.get("entityName"), "snapshot_retrieved_at": retrieved,
                "basis": "full-year values as reported in each 10-K (the filing's own figure); later different values listed as restatements",
                "series": series}

    @app.get("/watchlist", dependencies=[Depends(authorized)])
    def watchlist() -> dict:
        with pool.connection() as conn:
            rows = conn.execute("SELECT w.*, (SELECT count(*) FROM filings_seen f WHERE f.ticker = w.ticker AND NOT f.baseline) AS new_filings "
                                "FROM watchlist w ORDER BY ticker").fetchall()
        return {"items": rows}

    @app.post("/watchlist", status_code=status.HTTP_201_CREATED, dependencies=[Depends(authorized)])
    def watch(request: WatchRequest) -> dict:
        from finsight.ingestion.sec import PostgresRateLimiter, SecSource, SourceError
        if not settings.sec_user_agent:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "FINSIGHT_SEC_USER_AGENT is not set")
        ticker = request.ticker.upper()
        with pool.connection() as conn:
            source = SecSource(settings.sec_user_agent, settings.state_dir / "artifacts", conn=conn, rate_limiter=PostgresRateLimiter(conn))
            try:
                cik = source.resolve_cik(ticker)
            except SourceError as exc:
                raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
            conn.execute("INSERT INTO watchlist (ticker, cik, forms, auto_analyze) VALUES (%s, %s, %s, %s) "
                         "ON CONFLICT (ticker) DO UPDATE SET forms = EXCLUDED.forms, auto_analyze = EXCLUDED.auto_analyze",
                         (ticker, cik, request.forms, request.auto_analyze))
        return {"ticker": ticker, "cik": cik}

    @app.delete("/watchlist/{ticker}", dependencies=[Depends(authorized)])
    def unwatch(ticker: str) -> dict:
        with pool.connection() as conn:
            conn.execute("DELETE FROM watchlist WHERE ticker = %s", (ticker.upper(),))
        return {"ticker": ticker.upper(), "removed": True}

    @app.get("/filings", dependencies=[Depends(authorized)])
    def filings(limit: int = Query(default=50, ge=1, le=200), include_baseline: bool = False) -> dict:
        with pool.connection() as conn:
            rows = conn.execute("SELECT * FROM filings_seen WHERE (%s OR NOT baseline) ORDER BY filing_date DESC, first_seen_at DESC LIMIT %s",
                                (include_baseline, limit)).fetchall()
        return {"items": rows}

    @app.post("/filings/{accession}/analyze", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(authorized)])
    def analyze(accession: str, request: AnalyzeRequest) -> dict:
        from finsight.ingestion.sec import Filing
        from finsight.watcher import create_analysis
        with pool.connection() as conn:
            row = conn.execute("SELECT * FROM filings_seen WHERE accession = %s", (accession,)).fetchone()
            if row is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "filing not in the feed")
            if row["form"] != "10-K" or row["report_date"] is None:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "the research pipeline analyses annual 10-K filings")
            filing = Filing(cik=row["cik"], accession=accession, form=row["form"], filing_date=row["filing_date"],
                            report_date=row["report_date"], primary_document=row["primary_document"])
            job = create_analysis(conn, row["ticker"], filing, request.model, list(request.metrics))
        return {"id": str(job["id"]), "status": job["status"]}


def register_insights(app, pool, authorized) -> None:
    """Insights endpoints: the same context pack the CLI and the Claude skill use."""
    from finsight.insights import build_pack, commodity_insight, company_pack
    from finsight.market import CATALOG, FredSource, MarketError, analytics
    settings = get_settings()

    def fred():
        return FredSource(settings.fred_api_key.get_secret_value() if settings.fred_api_key else None, settings.state_dir / "artifacts")

    @app.get("/market/catalog", dependencies=[Depends(authorized)])
    def catalog() -> dict:
        return {"series": [{"key": k, "series_id": s, "title": t, "unit": u} for k, (s, t, u) in CATALOG.items()],
                "configured": bool(settings.fred_api_key)}

    @app.get("/insights/commodity/{key}", dependencies=[Depends(authorized)])
    def commodity(key: str) -> dict:
        try:
            series = fred().series(key)
        except MarketError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE if "not set" in str(exc) else status.HTTP_404_NOT_FOUND, str(exc)) from exc
        a = analytics(series)
        return {"analytics": a, "insight": commodity_insight(a) if a.get("available") else None,
                "observations": [[d.isoformat(), v] for d, v in series.observations[-520:]]}

    @app.get("/insights/company/{ticker}", dependencies=[Depends(authorized)])
    def company(ticker: str) -> dict:
        from finsight.ingestion.sec import PostgresRateLimiter, SecSource, SourceError
        if not settings.sec_user_agent:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "FINSIGHT_SEC_USER_AGENT is not set")
        with pool.connection() as conn:
            source = SecSource(settings.sec_user_agent, settings.state_dir / "artifacts", conn=conn, rate_limiter=PostgresRateLimiter(conn))
            try:
                return build_pack(company_pack(source, ticker), [])
            except SourceError as exc:
                raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
