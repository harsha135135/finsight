"""Export finished research jobs from the database into a sanitized, committable bundle.

A bundle keeps what is needed to show and replay a job without the live stack: the request, the frozen model
configuration, the event trail (stage results: evidence, model output, validation, review, report), the model
attempts' responses and usage, the cited and retrieved passages, the linked documents' metadata and the report.

Sanitizing removes worker identity (hostnames, backend pids, ownership tokens), local endpoints, absolute paths and
idempotency keys, and the export refuses to write if anything that looks like an email address, an API key, a
private IP address or a home-directory path remains.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LOCAL_ENDPOINT = "<local-model-endpoint>"
_FORBIDDEN = {
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"),
    "private IP address": re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    "home-directory path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+"),
    "local endpoint": re.compile(r"https?://(?:localhost|127\.0\.0\.1|host\.docker\.internal)[:/]"),
}
_DROP_JOB_FIELDS = ("owner_token", "owner_worker", "owner_backend_pid", "heartbeat_backend_pid", "heartbeat_at", "idempotency_key")


def _relative(path: str | None) -> str | None:
    if not path:
        return path
    marker = ".finsight/"
    return path[path.index(marker):] if marker in path else Path(path).name


def _scrub(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "base_url":
                out[k] = LOCAL_ENDPOINT
            elif k == "worker":
                out[k] = "<worker>"
            elif k in ("artifact_path", "json_path", "markdown_path", "companyfacts_snapshot") and isinstance(v, str):
                out[k] = _relative(v)
            else:
                out[k] = _scrub(v)
        return out
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def problems(text: str) -> list[str]:
    return [f"{name}: {m.group(0)[:40]!r}" for name, rx in _FORBIDDEN.items() for m in rx.finditer(text)]


def _rows(conn, sql: str, params) -> list[dict]:
    cur = conn.execute(sql, params)
    names = [d.name for d in cur.description]
    return [dict(r) if isinstance(r, dict) else dict(zip(names, r)) for r in cur.fetchall()]


def export_job(conn, job_id: str) -> dict:
    job = _rows(conn, "SELECT * FROM research_jobs WHERE id = %s", (job_id,))[0]
    for field in _DROP_JOB_FIELDS:
        job.pop(field, None)
    events = _rows(conn, "SELECT at, kind, stage, detail FROM job_events WHERE job_id = %s ORDER BY id", (job_id,))
    attempts = _rows(conn, "SELECT stage, attempt, provider, model, status, paid, reserved_usd, cost_usd, response, usage, latency_ms,"
                           " model_version, stop_reason, error, created_at FROM model_attempts WHERE job_id = %s ORDER BY created_at, attempt", (job_id,))
    claims = _rows(conn, "SELECT finding_key, kind, status, payload FROM claims WHERE job_id = %s ORDER BY id", (job_id,))
    documents = _rows(conn, "SELECT d.id, d.source_uri, d.sha256, d.artifact_path, d.accession, d.cik, d.company, d.form, d.fiscal_year,"
                            " d.period_end, d.filing_date, d.content_type, d.size_bytes, d.retrieved_at, d.parser_version"
                            " FROM job_documents jd JOIN documents d ON d.id = jd.document_id WHERE jd.job_id = %s", (job_id,))
    report = _rows(conn, "SELECT json_path, markdown_path, sha256, created_at FROM reports WHERE job_id = %s", (job_id,))
    passage_ids = sorted(set(re.findall(r'"(?:passage_id|id)": "([^"]+:[^"]+)"', json.dumps([e["detail"] for e in events], default=str))))
    passages = _rows(conn, "SELECT id, document_id, section, location, start_offset, end_offset, text FROM passages WHERE id = ANY(%s) ORDER BY id",
                     (passage_ids,)) if passage_ids else []
    bundle = {"job": job, "events": events, "model_attempts": attempts, "claims": claims, "documents": documents,
              "report": report[0] if report else None, "passages": passages}
    return _scrub(json.loads(json.dumps(bundle, default=str)))


def write_bundle(conn, job_ids: list[str], out: Path, note: str) -> dict:
    jobs = [export_job(conn, j) for j in job_ids]
    body = json.dumps({"bundle": "finsight-demo-bundle-v1", "note": note, "jobs": jobs}, indent=1, sort_keys=True) + "\n"
    found = problems(body)
    if found:
        raise ValueError("refusing to write an unsanitized bundle: " + "; ".join(sorted(set(found))[:10]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    return {"path": str(out), "jobs": len(jobs), "passages": sum(len(j["passages"]) for j in jobs), "bytes": len(body)}
