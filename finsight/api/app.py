"""FinSight HTTP API.

Accepts a research request and returns a job id immediately (202); a separate worker process
does the research and records progress in PostgreSQL, so restarting the API loses nothing.
The model configuration is resolved and frozen at admission. Every route requires the
``X-API-Key`` header.

    uv run uvicorn finsight.api.app:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import hmac
import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from finsight.settings import get_settings
from finsight.storage import jobs

CONTRACT_VERSION = "finsight-report-v1"


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["qwen", "claude", "recorded", "demo"]
    model: str | None = Field(default=None, max_length=200)
    max_tokens: int | None = Field(default=None, ge=1, le=64_000)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: str = Field(min_length=1, max_length=64, description="ticker or company name")
    question: str = Field(min_length=1, max_length=4_000)
    period: str = Field(min_length=2, max_length=32, description="e.g. FY2018")
    metrics: list[str] = Field(default_factory=list, max_length=20)
    cutoff_date: date | None = None
    model: ModelChoice
    idempotency_key: str | None = Field(default=None, max_length=200)


def resolve_model_config(choice: ModelChoice) -> dict:
    """Freeze the full model configuration at admission (later setting changes do not affect the job)."""
    settings = get_settings()
    defaults = {"qwen": {"model": "Qwen/Qwen3-4B", "base_url": settings.qwen_base_url, "max_tokens": 1024},
                "claude": {"model": settings.model_analyze, "max_tokens": settings.max_tokens_default},
                "recorded": {"model": "fixtures", "max_tokens": None},
                "demo": {"model": "demo", "max_tokens": None}}[choice.provider]
    return {"provider": choice.provider, "model": choice.model or defaults["model"],
            "max_tokens": choice.max_tokens or defaults["max_tokens"], "base_url": defaults.get("base_url"),
            "contract": CONTRACT_VERSION}


def create_app(database_url: str | None = None, api_key: str | None = None) -> FastAPI:
    settings = get_settings()
    url = database_url or settings.database_url
    key = api_key if api_key is not None else (settings.api_key.get_secret_value() if settings.api_key else None)
    if not key:
        raise RuntimeError("FINSIGHT_API_KEY is not set; the API refuses to run without authentication")
    pool = ConnectionPool(url, open=False, min_size=1, max_size=8,
                          kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0,
                                  "application_name": "finsight-api"})

    @asynccontextmanager
    async def lifespan(_app):
        pool.open(wait=True)
        yield
        pool.close()

    app = FastAPI(title="FinSight", version="0.1.0", lifespan=lifespan)

    def authorized(x_api_key: str | None = Header(default=None)) -> None:
        if not x_api_key or not hmac.compare_digest(x_api_key, key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")

    def job_or_404(conn, job_id: UUID) -> dict:
        job = jobs.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return job

    @app.post("/research-jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(authorized)])
    def submit(request: ResearchRequest) -> dict:
        payload = request.model_dump(mode="json", exclude={"model", "idempotency_key"})
        with pool.connection() as conn:
            job = jobs.create_job(conn, payload, resolve_model_config(request.model), request.idempotency_key)
        return {"id": str(job["id"]), "status": job["status"], "links": {"self": f"/research-jobs/{job['id']}"}}

    @app.get("/research-jobs/{job_id}", dependencies=[Depends(authorized)])
    def read(job_id: UUID) -> dict:
        with pool.connection() as conn:
            job = job_or_404(conn, job_id)
            usage = conn.execute(
                "SELECT count(*) AS attempts, coalesce(sum((usage->>'input_tokens')::bigint), 0) AS input_tokens, "
                "coalesce(sum((usage->>'output_tokens')::bigint), 0) AS output_tokens, "
                "coalesce(sum(cost_usd), 0) AS cost_usd, count(*) FILTER (WHERE status = 'unknown') AS unknown_outcomes "
                "FROM model_attempts WHERE job_id = %s", (job_id,)).fetchone()
            recent = jobs.events(conn, job_id, limit=20)
        return {"id": str(job["id"]), "status": job["status"], "stage": job["stage"],
                "outcome_reason": job["outcome_reason"], "cancel_requested": job["cancel_requested"],
                "request": job["request"], "model_config": job["model_config"],
                "created_at": job["created_at"], "finished_at": job["finished_at"],
                "usage": {k: (str(v) if k == "cost_usd" else v) for k, v in usage.items()},
                "events": [{"at": e["at"], "kind": e["kind"], "stage": e["stage"]} for e in recent]}

    @app.get("/research-jobs/{job_id}/report", dependencies=[Depends(authorized)])
    def report(job_id: UUID, format: Literal["json", "markdown"] = Query(default="json")):
        with pool.connection() as conn:
            job = job_or_404(conn, job_id)
            row = conn.execute("SELECT * FROM reports WHERE job_id = %s", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no report yet (status {job['status']})")
        if format == "markdown":
            return PlainTextResponse(Path(row["markdown_path"]).read_text(), media_type="text/markdown")
        return {"job_id": str(job_id), "status": job["status"], "sha256": row["sha256"],
                "report": json.loads(Path(row["json_path"]).read_text())}

    @app.get("/research-jobs/{job_id}/evidence", dependencies=[Depends(authorized)])
    def evidence(job_id: UUID, limit: int = Query(default=200, ge=1, le=1000)) -> dict:
        with pool.connection() as conn:
            job_or_404(conn, job_id)
            documents = conn.execute(
                "SELECT d.id, d.source_uri, d.accession, d.form, d.fiscal_year, d.period_end, d.sha256, d.retrieved_at "
                "FROM job_documents j JOIN documents d ON d.id = j.document_id WHERE j.job_id = %s", (job_id,)).fetchall()
            passages = conn.execute(
                "SELECT p.id, p.document_id, p.section, p.location, p.text FROM job_documents j "
                "JOIN passages p ON p.document_id = j.document_id WHERE j.job_id = %s ORDER BY p.id LIMIT %s",
                (job_id, limit)).fetchall()
        return {"job_id": str(job_id), "documents": documents, "passages": passages}

    @app.post("/research-jobs/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(authorized)])
    def cancel(job_id: UUID) -> dict:
        with pool.connection() as conn:
            row = jobs.request_cancel(conn, job_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return {"id": str(job_id), "status": row["status"], "cancel_requested": True}

    return app


def __getattr__(name: str):  # `uvicorn finsight.api.app:app` builds the app from settings on first access
    if name == "app":
        return create_app()
    raise AttributeError(name)
