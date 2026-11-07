"""Real workflow stages for a research job.

research   -> resolve the company, choose the filing for the requested period, store the original,
              parse it, retrieve evidence per metric, and take XBRL as-reported facts for cross-checks
analysis   -> one model call in the ``finsight-report-v1`` contract, with one bounded repair when the
              output is malformed
validation -> per-finding checks (finsight.findings)
review     -> cross-checks accepted figures against XBRL as reported in the same filing; decides the outcome
reporting  -> the report (JSON + Markdown)

Every stage returns JSON-serializable data and passes forward what later stages need, because a
resumed job rebuilds the stage object and has only the stored results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from finsight.contracts import Passage, PassageLocation
from finsight.contracts.report import REPORT_CONTRACT, ReportOutput
from finsight.findings import validate_report, year_of
from finsight.ingestion.html import PARSER_VERSION, parse_filing
from finsight.ingestion.sec import SecSource
from finsight.reasoning.base import ModelProvider
from finsight.reasoning.prompt import PROMPT_VERSION, SYSTEM, report_schema, user_message
from finsight.reporting import markdown
from finsight.retrieval.search import DERIVED, METRICS, search, store_passages
from finsight.storage import jobs

KEYWORDS = {"revenue": ("revenue", "sales"), "operating_margin": ("operating margin",),
            "operating_income": ("operating income",), "capex": ("capital expenditure", "capex", "purchases of property"),
            "net_income": ("net income", "net earnings")}
MAX_PASSAGES_PER_METRIC = 4


def infer_metrics(question: str) -> list[str]:
    text = question.lower()
    found = [m for m, words in KEYWORDS.items() if any(w in text for w in words)]
    if "operating_margin" in found and "operating_income" in found and "operating income" not in text.replace("operating margin", ""):
        found.remove("operating_income")
    return found


def base_metrics(metrics: list[str]) -> list[str]:
    out = []
    for m in metrics:
        for b in (DERIVED.get(m) or (m,)):
            if b not in out:
                out.append(b)
    return out


def _jsonable(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def annual_fact(facts: dict, concepts: list[str], accession: str, end: date) -> dict | None:
    """As-reported full-year value for the period ending ``end`` in this filing (first matching concept)."""
    for concept in concepts:
        entries = facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {}).get("USD", [])
        for e in entries:
            if e.get("accn") != accession or e.get("end") != end.isoformat() or not e.get("start"):
                continue
            days = (end - date.fromisoformat(e["start"])).days
            if 350 <= days <= 380:
                later = sorted({x["val"] for x in entries if x.get("end") == e["end"] and x.get("start") == e["start"]
                                and x.get("accn") != accession and x["val"] != e["val"]})
                return {"concept": concept, "value": e["val"], "start": e["start"], "end": e["end"],
                        "accession": accession, "later_different_values": later}
    return None


@dataclass
class ResearchStages:
    conn: object
    job_id: object
    token: object
    source: SecSource | None
    provider: ModelProvider
    model_config: dict

    # ------------------------------------------------------------------ research
    def research(self, request: dict) -> dict:
        if self.source is None:
            raise RuntimeError("SEC ingestion is not configured (set FINSIGHT_SEC_USER_AGENT with a contact email)")
        year = year_of(request["period"])
        metrics = request.get("metrics") or infer_metrics(request["question"])
        if not metrics:
            raise RuntimeError("the question names no supported metric")
        cutoff = date.fromisoformat(request["cutoff_date"]) if request.get("cutoff_date") else None
        cik = self.source.resolve_cik(request["company"])
        filings, fye = self.source.filings(cik)
        choice = SecSource.choose_annual(filings, year, fye, cutoff)
        document = self.source.fetch_document(choice.filing)
        html = Path(document["artifact_path"]).read_text(errors="replace")
        short = f"{choice.filing.accession}".replace("-", "")[-10:]
        passages = parse_filing(html, document["id"], short)
        store_passages(self.conn, passages)
        self.conn.execute("UPDATE documents SET parser_version = %s, company = %s WHERE id = %s",
                          (PARSER_VERSION, self.source.names.get(cik), document["id"]))
        self.conn.execute("INSERT INTO job_documents (job_id, document_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                          (self.job_id, document["id"]))
        facts, snapshot = self.source.company_facts_snapshot(cik)
        evidence, supplied, xbrl = {}, {}, {}
        for metric in base_metrics(metrics):
            found = search(self.conn, [document["id"]], metric, year, limit=MAX_PASSAGES_PER_METRIC)
            evidence[metric] = {"status": found.status, "reason": found.reason,
                                "passage_ids": [c.id for c in found.candidates]}
            for c in found.candidates:
                supplied[c.id] = c.text
            xbrl[metric] = annual_fact(facts, METRICS[metric]["concepts"], choice.filing.accession, choice.filing.report_date)
        filing = {k: document.get(k) for k in ("id", "source_uri", "sha256", "accession", "form", "period_end",
                                               "filing_date", "retrieved_at")}
        filing.update(reason=choice.reason, amendments=[a.accession for a in choice.amendments])
        return _jsonable({"company": self.source.names.get(cik) or request["company"], "cik": cik, "fiscal_year": year,
                          "period": request["period"], "metrics": metrics, "filing": filing, "evidence": evidence,
                          "passages": [{"id": k, "text": v} for k, v in supplied.items()], "xbrl": xbrl,
                          "companyfacts_snapshot": snapshot})

    # ------------------------------------------------------------------ analysis
    def analysis(self, request: dict, research: dict) -> dict:
        requested = [(m, research["period"]) for m in research["metrics"]]
        missing = {m: e["reason"] for m, e in research["evidence"].items() if e["status"] == "missing"}
        if not research["passages"]:
            return {"status": "ok", "skipped_model": True, "output": {"findings": [
                {"metric": m, "period": p, "status": "insufficient_evidence", "operands": [], "program": [], "unit": None,
                 "explanation": [], "missing": "no passage reports this metric"} for m, p in requested], "unresolved": []}}
        system = SYSTEM
        user = user_message(research["company"], research["filing"], requested, research["passages"], missing)
        schema = report_schema()
        error = None
        for attempt in (1, 2):
            prompt = user if error is None else f"{user}\n\nYour previous output was invalid ({error}). Return valid JSON."
            generation = self.provider.generate(system, prompt, schema, job_id=self.job_id, stage="analysis",
                                                attempt=attempt, scope="finsight")
            self._record_unpaid(generation, attempt)
            if generation.text is None:
                status = generation.attempts[-1].status if generation.attempts else "error"
                if status in ("refused", "truncated"):
                    return {"status": "blocked", "reason": f"model output {status}"}
                if status == "uncertain":
                    return {"status": "failed", "reason": "model outcome unknown; not resent automatically"}
                return {"status": "failed", "reason": f"model call failed ({status})"}
            try:
                output = ReportOutput.model_validate_json(generation.text)
                return {"status": "ok", "output": output.model_dump(mode="json"), "attempts": attempt,
                        "prompt_version": PROMPT_VERSION}
            except ValueError as exc:
                error = str(exc).splitlines()[0][:300]
                if attempt == 1:
                    repairs = jobs.get_job(self.conn, self.job_id)["repairs"] or {}
                    jobs.owned_update(self.conn, self.job_id, self.token,
                                      repairs={**repairs, "malformed_output": repairs.get("malformed_output", 0) + 1})
        return {"status": "failed", "reason": f"malformed model output after one repair: {error}"}

    def _record_unpaid(self, generation, attempt: int) -> None:
        if getattr(self.provider, "ledger", None) is not None:
            return  # paid calls are recorded by the ledger itself
        for record in generation.attempts:
            self.conn.execute(
                "INSERT INTO model_attempts (id, scope, job_id, stage, attempt, request_sha256, provider, model, status, "
                "paid, usage, latency_ms, model_version, error) VALUES (gen_random_uuid(), 'finsight', %s, 'analysis', %s, "
                "'unpaid', %s, %s, %s, false, %s, %s, %s, %s) ON CONFLICT (job_id, stage, attempt) DO NOTHING",
                (self.job_id, attempt, self.provider.provider, self.provider.model,
                 "completed" if record.status == "ok" else "failed",
                 json.dumps({"input_tokens": record.input_tokens_uncached, "output_tokens": record.output_tokens}),
                 record.latency_ms, record.served_model, record.error))

    # ------------------------------------------------------------------ validation
    def validation(self, request: dict, analysis: dict, research: dict) -> dict:
        carry = {k: research[k] for k in ("company", "period", "metrics", "filing", "evidence", "xbrl")}
        if analysis["status"] != "ok":
            return {"status": analysis["status"], "reason": analysis["reason"], "findings": [], "research": carry}
        ids = [p["id"] for p in research["passages"]]
        rows = self.conn.execute("SELECT * FROM passages WHERE id = ANY(%s)", (ids,)).fetchall()
        supplied = {r["id"]: Passage(id=r["id"], document_id=r["document_id"], section=r["section"],
                                     location=PassageLocation(**r["location"]), start=r["start_offset"],
                                     end=r["end_offset"], text=r["text"]) for r in rows}
        validated = [f.model_dump(mode="json") for f in validate_report(ReportOutput.model_validate(analysis["output"]),
                                                                        supplied)]
        answered = {f["metric"] for f in validated}
        for metric in research["metrics"]:
            if metric not in answered:
                validated.append({"metric": metric, "period": research["period"], "status": "insufficient_evidence",
                                  "issues": ["the model returned no finding for this metric"], "operands": [],
                                  "calculation": [], "claims": [], "value": None, "unit": None})
        return {"status": "ok", "findings": validated, "research": carry}

    # ------------------------------------------------------------------ review
    def review(self, request: dict, validation: dict) -> dict:
        research = validation["research"]
        unresolved = []
        if validation["status"] != "ok":
            return {"outcome": "blocked" if validation["status"] == "blocked" else "failed",
                    "reason": validation.get("reason"), "findings": [], "unresolved": [validation.get("reason")],
                    "research": research}
        findings = []
        for f in validation["findings"]:
            f = dict(f, checks=[])
            if f["status"] == "accepted":
                f["checks"], problems = self._xbrl_checks(f, research["xbrl"])
                unresolved += problems
            elif f["status"] == "rejected":
                unresolved.append(f"{f['metric']} {f['period']}: rejected ({'; '.join(f['issues'])})")
            else:
                unresolved.append(f"{f['metric']} {f['period']}: insufficient evidence ({'; '.join(f['issues'])})")
            findings.append(f)
        accepted = [f for f in findings if f["status"] == "accepted"]
        outcome = ("completed" if accepted else
                   "insufficient_evidence" if all(f["status"] == "insufficient_evidence" for f in findings) else "failed")
        return {"outcome": outcome, "findings": findings, "unresolved": unresolved, "research": research}

    @staticmethod
    def _xbrl_checks(finding: dict, xbrl: dict) -> tuple[list[str], list[str]]:
        value = Decimal(finding["value"])
        metric = finding["metric"]
        if metric in METRICS:
            fact = xbrl.get(metric)
            if not fact:
                return ["no XBRL fact in this filing to cross-check"], []
            agrees = abs(value) == abs(Decimal(fact["value"]))
            note = f"XBRL {fact['concept']} as reported in {fact['accession']}: {fact['value']}"
            later = f"; later filings report {fact['later_different_values']}" if fact["later_different_values"] else ""
            return [note + (" (agrees)" if agrees else " (differs)") + later], \
                [] if agrees else [f"{metric}: {value} differs from {note}"]
        if metric in DERIVED:
            parts = [xbrl.get(k) for k in DERIVED[metric]]
            if not all(parts):
                return ["components lack XBRL facts to cross-check"], []
            expected = Decimal(parts[0]["value"]) / Decimal(parts[1]["value"])
            agrees = abs(expected - value) <= Decimal("0.0005")
            note = f"XBRL {parts[0]['concept']} / {parts[1]['concept']} = {expected:.4f}"
            return [note + (" (agrees)" if agrees else " (differs)")], [] if agrees else [f"{metric}: {value} vs {note}"]
        return [], []

    # ------------------------------------------------------------------ reporting
    def reporting(self, request: dict, review: dict) -> dict:
        research = review["research"]
        usage = self.conn.execute(
            "SELECT count(*) AS attempts, coalesce(sum((usage->>'input_tokens')::bigint), 0) AS input_tokens, "
            "coalesce(sum((usage->>'output_tokens')::bigint), 0) AS output_tokens, coalesce(sum(cost_usd), 0) AS cost_usd, "
            "count(*) FILTER (WHERE status = 'unknown') AS unknown_outcomes FROM model_attempts WHERE job_id = %s",
            (self.job_id,)).fetchone()
        report = _jsonable({
            "contract": REPORT_CONTRACT, "company": research["company"], "period": research["period"],
            "question": request["question"], "outcome": review["outcome"], "filing": research["filing"],
            "findings": review["findings"], "unresolved": [u for u in review["unresolved"] if u],
            "model": {k: self.model_config.get(k) for k in ("provider", "model", "max_tokens", "contract")},
            "generated_at": datetime.now(timezone.utc), "usage": dict(usage)})
        return {"outcome": review["outcome"], "report": report, "markdown": markdown(report)}
