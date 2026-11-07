"""The research workflow as a LangGraph graph with durable, idempotent stages.

Stages: research → analysis → validation → review → reporting. Each node:

1. checks cancellation and ownership before doing anything;
2. reuses its stored result if it already completed (a node re-run after a crash must not
   repeat side effects or model calls);
3. persists its result as a ``stage_result`` event, conditional on ownership, before returning.

The graph is compiled per job with a ``PostgresSaver`` on the worker's dedicated session and
invoked with ``durability="sync"``, so every completed step is checkpointed before the next.
The stage implementations are pluggable (``Stages``); ``DemoStages`` is deterministic and offline.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from finsight.storage import jobs

STAGES = ("research", "analysis", "validation", "review", "reporting")


class Cancelled(RuntimeError):
    pass


class JobState(TypedDict, total=False):
    job_id: str
    request: dict
    research: dict
    analysis: dict
    validation: dict
    review: dict
    reporting: dict


class Stages(Protocol):
    def research(self, request: dict) -> dict: ...
    def analysis(self, request: dict, research: dict) -> dict: ...
    def validation(self, request: dict, analysis: dict, research: dict) -> dict: ...
    def review(self, request: dict, validation: dict) -> dict: ...
    def reporting(self, request: dict, review: dict) -> dict: ...


@dataclass
class DemoStages:
    """Deterministic offline stages; ``delays`` (seconds per stage) let tests interrupt a node mid-way."""

    delays: dict[str, float] = field(default_factory=dict)

    def _wait(self, stage: str) -> None:
        if self.delays.get(stage):
            time.sleep(self.delays[stage])

    def research(self, request):
        self._wait("research")
        return {"documents": ["demo:filing"], "passages": ["demo:p1"]}

    def analysis(self, request, research):
        self._wait("analysis")
        return {"findings": [{"metric": "revenue", "value": "1", "unit": "USD", "passages": research["passages"]}]}

    def validation(self, request, analysis, research):
        self._wait("validation")
        return {"accepted": analysis["findings"], "rejected": []}

    def review(self, request, validation):
        self._wait("review")
        return {"outcome": "completed", "findings": validation["accepted"], "unresolved": []}

    def reporting(self, request, review):
        self._wait("reporting")
        return {"outcome": review["outcome"], "report": {"request": request, "findings": review["findings"],
                                                          "unresolved": review["unresolved"]}}


@dataclass
class JobContext:
    conn: Any                       # the worker's dedicated session (lock + checkpointer)
    job_id: uuid.UUID
    token: uuid.UUID
    artifacts: Path


def _stored_result(ctx: JobContext, stage: str) -> dict | None:
    row = ctx.conn.execute("SELECT detail FROM job_events WHERE job_id = %s AND kind = 'stage_result' AND stage = %s "
                           "ORDER BY id DESC LIMIT 1", (ctx.job_id, stage)).fetchone()
    return row["detail"] if row else None


def _check(ctx: JobContext) -> None:
    job = ctx.conn.execute("SELECT owner_token, cancel_requested FROM research_jobs WHERE id = %s", (ctx.job_id,)).fetchone()
    if job is None or job["owner_token"] != ctx.token:
        raise jobs.OwnershipLost(f"job {ctx.job_id}: ownership lost")
    if job["cancel_requested"]:
        raise Cancelled(f"job {ctx.job_id}: cancellation requested")


def _node(ctx: JobContext, stage: str, run):
    def node(state: JobState) -> dict:
        _check(ctx)
        stored = _stored_result(ctx, stage)
        if stored is not None:
            return {stage: stored}
        jobs.owned_update(ctx.conn, ctx.job_id, ctx.token, stage=stage)
        result = run(state)
        if stage == "reporting":
            result = {**result, "report_sha256": write_report(ctx, result["report"])}
        with ctx.conn.transaction():
            jobs.owned_event(ctx.conn, ctx.job_id, ctx.token, "stage_result", stage=stage, detail=result)
        return {stage: result}
    return node


def write_report(ctx: JobContext, report: dict) -> str:
    """Write the report files atomically and record them once per job (idempotent)."""
    body = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    # Per-token folder: a stale worker can only ever write into its own folder, which no report row references.
    directory = ctx.artifacts / "reports" / str(ctx.job_id) / str(ctx.token)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"json": directory / "report.json", "markdown": directory / "report.md"}
    for path, text in ((paths["json"], body), (paths["markdown"], f"# Report {ctx.job_id}\n\n```json\n{body}```\n")):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
    row = ctx.conn.execute(
        "INSERT INTO reports (job_id, json_path, markdown_path, sha256) SELECT %s, %s, %s, %s "
        "WHERE EXISTS (SELECT 1 FROM research_jobs WHERE id = %s AND owner_token = %s) "
        "ON CONFLICT (job_id) DO NOTHING RETURNING job_id",
        (ctx.job_id, str(paths["json"]), str(paths["markdown"]), digest, ctx.job_id, ctx.token)).fetchone()
    if row is None:
        _check(ctx)  # raises if ownership was lost; otherwise the report already exists (re-run after a crash)
    return digest


def build_graph(ctx: JobContext, stages: Stages) -> StateGraph:
    graph = StateGraph(JobState)
    graph.add_node("research", _node(ctx, "research", lambda s: stages.research(s["request"])))
    graph.add_node("analysis", _node(ctx, "analysis", lambda s: stages.analysis(s["request"], s["research"])))
    graph.add_node("validation", _node(ctx, "validation",
                                       lambda s: stages.validation(s["request"], s["analysis"], s["research"])))
    graph.add_node("review", _node(ctx, "review", lambda s: stages.review(s["request"], s["validation"])))
    graph.add_node("reporting", _node(ctx, "reporting", lambda s: stages.reporting(s["request"], s["review"])))
    graph.add_edge(START, "research")
    for a, b in zip(STAGES, STAGES[1:]):
        graph.add_edge(a, b)
    graph.add_edge("reporting", END)
    return graph
