"""One-time capture of the offline replay bundle for the FY2025 demo job (488aac69), kept for provenance.

    FINSIGHT_TEST_DATABASE_URL=... uv run python scripts/capture_replay.py

* SEC: the 10-K and the companyfacts snapshot are served from the local cache and must match the sha256 the original job
  recorded; the small index files (company tickers, submissions) are fetched live once, with the user agent from .env
  (the user agent is never written to the bundle).
* Model: the pipeline is re-run against a scratch database; at the model call the exported response of the original
  attempt (evals/demo/fy2025/bundle.json) is returned and stored under the key of the request rebuilt now, with the
  original provenance. The capture refuses to write unless the rebuilt evidence equals the original job's evidence
  passages (ids and text) and the outcome matches the original report.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsight.contracts import ModelAttempt  # noqa: E402
from finsight.ingestion.sec import SecSource  # noqa: E402
from finsight.reasoning.base import Generation  # noqa: E402
from finsight.reasoning.recorded import request_key  # noqa: E402
from finsight.replay import write_recordings, write_sec_bundle  # noqa: E402
from finsight.settings import get_settings  # noqa: E402
from finsight.stages import ResearchStages  # noqa: E402
from finsight.storage import jobs  # noqa: E402
from finsight.storage.db import connect  # noqa: E402
from finsight.worker import serve  # noqa: E402

OUT = Path("evals/demo/replay")
BUNDLE = json.loads(Path("evals/demo/fy2025/bundle.json").read_text())
JOB = next(j for j in BUNDLE["jobs"] if j["job"]["id"].startswith("488aac69"))
RESEARCH = next(e["detail"] for e in JOB["events"] if e["stage"] == "research")
TENK_URL = RESEARCH["filing"]["source_uri"]
FACTS_URL = RESEARCH["companyfacts_snapshot"]["source_uri"]
LOCAL = {TENK_URL: (Path(".finsight/artifacts/sec/0000066740/0000066740-26-000014/mmm-20251231.htm"), RESEARCH["filing"]["sha256"], "text/html"),
         FACTS_URL: (Path(RESEARCH["companyfacts_snapshot"]["artifact_path"]), RESEARCH["companyfacts_snapshot"]["sha256"], "application/json")}


class CaptureTransport(httpx.BaseTransport):
    def __init__(self, user_agent: str) -> None:
        self.live = httpx.Client(headers={"User-Agent": user_agent}, timeout=30)
        self.responses: dict[str, tuple[bytes, str]] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in LOCAL:
            path, digest, ctype = LOCAL[url]
            body = path.read_bytes()
            assert hashlib.sha256(body).hexdigest() == digest, f"{path} does not match the original job's sha256"
        else:
            live = self.live.get(url)
            live.raise_for_status()
            body, ctype = live.content, live.headers.get("content-type", "application/json")
        self.responses[url] = (body, ctype)
        return httpx.Response(200, content=body, headers={"content-type": ctype})


class CaptureProvider:
    """Returns the original attempt's response text and records it under the rebuilt request's key."""
    provider = "recorded"

    def __init__(self, model: str) -> None:
        self.model = model
        self.attempts = {a["attempt"]: a for a in JOB["model_attempts"]}
        self.entries: dict[str, dict] = {}
        self.requests: list[str] = []

    def generate(self, system, user, schema, *, attempt=1, **kwargs):
        original = self.attempts[attempt]
        key = request_key(self.model, system, user, schema)
        self.requests.append(user)
        self.entries[key] = {"text": original["response"]["text"], "provenance": {
            "source": "exported model_attempts row of the original job (evals/demo/fy2025/bundle.json)",
            "job_id": JOB["job"]["id"], "stage": original["stage"], "attempt": attempt,
            "model": original["model"], "model_version": original["model_version"], "recorded_at": original["created_at"],
            "usage": original["usage"], "latency_ms": original["latency_ms"], "original_status": original["status"],
            "original_model_config": JOB["job"]["model_config"],
            "prompt_version": next(e["detail"].get("prompt_version") for e in JOB["events"] if e["stage"] == "analysis"),
            "request": {"how": "rebuilt deterministically from the bundled filing and retrieval at capture time; the original "
                               "run stored no request hash (fixed since: unpaid attempts now record it)",
                        "evidence_matches_original": None, "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                        "user_sha256": hashlib.sha256(user.encode()).hexdigest(),
                        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()},
            "captured_at": datetime.now(timezone.utc).isoformat()}}
        return Generation(text=original["response"]["text"], attempts=[ModelAttempt(
            requested_model=self.model, served_model=self.model, stage="analysis", attempt=attempt, status="ok",
            latency_ms=0, started_at=datetime.now(timezone.utc))])


def main() -> int:
    url = os.environ["FINSIGHT_TEST_DATABASE_URL"]
    conn = connect(url)
    for table in ("reports", "claims", "job_events", "model_attempts", "job_documents", "research_jobs"):
        conn.execute(f"DELETE FROM {table}")
    transport = CaptureTransport(get_settings().sec_user_agent)
    provider = CaptureProvider(JOB["job"]["model_config"]["model"])
    config = {"provider": "recorded", "model": provider.model, "contract": JOB["job"]["model_config"]["contract"], "max_tokens": None}
    job = jobs.create_job(conn, JOB["job"]["request"], config)
    with tempfile.TemporaryDirectory() as tmp:
        factory = lambda j, c, token: ResearchStages(conn=c, job_id=j["id"], token=token, provider=provider, model_config=j["model_config"],
                                                     source=SecSource(get_settings().sec_user_agent, Path(tmp), conn=c, transport=transport))
        serve(url, "replay-capture", factory, Path(tmp), once=True, heartbeat_interval=0.5)
        status = jobs.get_job(conn, job["id"])
        report = json.loads(Path(conn.execute("SELECT json_path FROM reports WHERE job_id = %s", (job["id"],)).fetchone()["json_path"]).read_text())
    rebuilt = conn.execute("SELECT detail FROM job_events WHERE job_id = %s AND stage = 'research'", (job["id"],)).fetchone()["detail"]
    same_evidence = [(p["id"], p["text"]) for p in rebuilt["passages"]] == [(p["id"], p["text"]) for p in RESEARCH["passages"]]
    original = json.loads(Path("evals/demo/fy2025/488aac69.report.json").read_text())
    same_outcome = (status["status"] == JOB["job"]["status"] and
                    [(f["metric"], f["status"], f.get("value")) for f in report["findings"]] ==
                    [(f["metric"], f["status"], f.get("value")) for f in original["findings"]])
    print("evidence matches original:", same_evidence, "| outcome matches original:", same_outcome, "| status:", status["status"])
    if not (same_evidence and same_outcome and len(provider.entries) == len(JOB["model_attempts"])):
        print("refusing to write the bundle"); return 1
    for entry in provider.entries.values():
        entry["provenance"]["request"]["evidence_matches_original"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    notes = {u: "original bytes from the job's cache (sha256 as recorded by the original job)" for u in LOCAL}
    write_sec_bundle(OUT / "sec", transport.responses, notes)
    write_recordings(OUT / "recordings.json", provider.entries,
                     "FY2025 3M 10-K demo job 488aac69 (qwen3-4b-finqa-v1, 2026-11-09); responses are the original model outputs")
    (OUT / "job.json").write_text(json.dumps({"request": JOB["job"]["request"], "model_config": config,
                                              "original_job": JOB["job"]["id"], "expected_status": JOB["job"]["status"],
                                              "expected_findings": [(f["metric"], f["status"], f.get("value")) for f in original["findings"]]},
                                             indent=1) + "\n")
    print("wrote", OUT, {u[:60]: len(b) for u, (b, _) in transport.responses.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
