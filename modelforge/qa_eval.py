"""Runs under the ``finsight-qa-v1`` contract: FinanceBench development items (supplied / full document) and the
FinQA native period controls. Raw outputs are stored with the exact items so validation and scoring can be redone
offline without generating again."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from finsight.contracts import Passage, PassageLocation
from finsight.reasoning.qa_prompt import QA_PROMPT_VERSION, SYSTEM, qa_schema, user_message
from finsight.qa import requested_years
from modelforge import splits, wrong_period


def _passage(row: dict) -> Passage:
    loc = row["location"] if "location" in row else {"page": row.get("page"), "table": row.get("table"), "row": row.get("row")}
    return Passage(id=row["id"], document_id=row.get("document_id", "doc"), section=row.get("section") or f"page_{loc.get('page')}",
                   location=PassageLocation(**{k: loc.get(k) for k in ("page", "html_section", "table", "row")}),
                   start=row.get("start_offset", 0), end=row.get("end_offset", len(row["text"])), text=row["text"])


def financebench_items(manifest: dict, mode: str, conn=None,
                       retrieval_path: Path = Path("evals/financebench/retrieval-dev-pdf2-text-2.json")) -> list[dict]:
    from modelforge.datasets import financebench as fb

    retrieved = json.loads(retrieval_path.read_text())["questions"] if mode == "full_document" else {}
    items = []
    for q in fb.dev_questions(manifest):
        if mode == "supplied":
            pages = [e["evidence_page_num"] + 1 for e in q["evidence"]]
            rows = conn.execute("SELECT id, document_id, section, location, text, start_offset, end_offset FROM passages "
                                "WHERE document_id = %s AND (location->>'page')::int = ANY(%s) ORDER BY start_offset",
                                (f"financebench:{q['doc_name']}:pdf-2", pages)).fetchall()
            passages = [_passage(dict(r)) for r in rows]
        else:
            passages = [_passage(r) for r in retrieved[q["financebench_id"]]["passages"]]
        # finsight-qa-v1.5: short ids (p1, p2, ...) in the prompt; FinQA-trained models shortened long ids with colons
        aliases = {f"p{n + 1}": p.id for n, p in enumerate(passages)}
        passages = [p.model_copy(update={"id": f"p{n + 1}"}) for n, p in enumerate(passages)]
        items.append({"id": q["financebench_id"], "question": q["question"], "company": q["company"], "doc": q["doc_name"],
                      "passages": [p.model_dump(mode="json") for p in passages], "passage_aliases": aliases})
    return items


def finqa_control_items() -> list[dict]:
    """Negative control: the 193 frozen wrong-period items; positive control: their original questions. FinQA table
    rows are given in FinSight's 'label | year: value' form (the form the period check reads)."""
    manifest = splits.load(splits.WRONG_PERIOD_NAME)
    shifted = splits.resolve_items(manifest, manifest["samples"]["val_dev_wrong_period"])
    originals = {i.id: i for i in splits.resolve_items(splits.load("finqa-grouped-v2"), sorted({s.id.split("::")[0] for s in shifted}))}
    items = []
    for s in shifted:
        for item, role in ((s, "negative"), (originals[s.id.split("::")[0]], "positive")):
            passages, _ = wrong_period.finsight_passages(item)
            items.append({"id": item.id, "role": role, "question": item.question, "company": item.company,
                          "gold_answer": item.gold_answer, "calc_convention": item.meta.get("calc_convention"),
                          "requested_years": sorted(requested_years(item.question)),
                          "shifted_from": s.meta["shifted_from"], "shifted_to": s.meta["shifted_to"],
                          "passages": [p.model_dump(mode="json") for p in passages.values()]})
    return items


def run(items: list[dict], provider, out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    config = {"label": label, "contract": "finsight-qa-v1", "prompt_version": QA_PROMPT_VERSION,
              "schema_sha256": hashlib.sha256(json.dumps(qa_schema(), sort_keys=True).encode()).hexdigest(),
              "items_sha256": hashlib.sha256((out_dir / "items.jsonl").read_bytes()).hexdigest(),
              "model": provider.model, "settings": provider.settings, "server": provider.server_info(),
              "started_at": datetime.now(timezone.utc).isoformat()}
    config_path, preds_path = out_dir / "config.json", out_dir / "predictions.jsonl"
    done = {json.loads(l)["item_id"] for l in preds_path.read_text().splitlines()} if preds_path.exists() else set()
    if done and config_path.exists():
        previous = json.loads(config_path.read_text())
        changed = [k for k in ("prompt_version", "schema_sha256", "items_sha256", "model", "settings", "server") if previous.get(k) != config.get(k)]
        if changed:
            raise ValueError(f"{out_dir} holds outputs from a different configuration ({changed})")
        config["started_at"] = previous["started_at"]
    config_path.write_text(json.dumps(config, indent=1) + "\n")
    with preds_path.open("a") as handle:
        for item in items:
            if item["id"] in done:
                continue
            passages = [Passage.model_validate(p) for p in item["passages"]]
            t0 = time.monotonic()
            generation = provider.generate(SYSTEM, user_message(item["question"], passages), qa_schema(), stage="qa")
            attempt = generation.attempts[-1] if generation.attempts else None
            handle.write(json.dumps({"item_id": item["id"], "text": generation.text, "raw": generation.raw,
                                     "status": attempt.status if attempt else "error", "error": attempt.error if attempt else None,
                                     "latency_ms": attempt.latency_ms if attempt else int((time.monotonic() - t0) * 1000),
                                     "input_tokens": attempt.input_tokens_uncached if attempt else None,
                                     "output_tokens": attempt.output_tokens if attempt else None}) + "\n")
            handle.flush()
    return out_dir
