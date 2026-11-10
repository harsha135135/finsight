"""FinanceBench full-document evidence: PDF ingestion into FinSight passages, retrieval runs, and evidence coverage.

Committed so the frozen retrieval files regenerate from code:

* ``ingest(conn, docs, pdf_record, manifest, parser)`` parses the original PDFs into ``financebench:<doc>:<parser>``
  documents and passages (``verify_ingested`` re-parses and compares ids and text with the database);
* ``retrieve(conn, questions, version, parser)`` runs FinSight's text retrieval per question within its own document
  and records every passage handed to the model;
* ``page_coverage`` (all / some / none of the gold pages) and ``evidence_coverage`` (whether the figures, headers and
  text the answer needs actually reach the model within the context budget; see ``evals/financebench/evidence-audit-*``).

Questions are passed in by the caller (development questions from ``modelforge.datasets.financebench.dev_questions``);
nothing here reads the held-out half.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from finsight.retrieval.search import search_text, store_passages

PARSERS = ("pdf-2", "pdf-3")


def document_id(doc: str, parser: str) -> str:
    return f"financebench:{doc}:{parser}"


def _parse(path: Path, doc: str, parser: str):
    from finsight.ingestion import pdf

    if parser == "pdf-2":
        return pdf.parse_pdf_v2(path, document_id(doc, parser), f"{doc}:v2")
    if parser == "pdf-3":
        return pdf.parse_pdf_v3(path, document_id(doc, parser), f"{doc}:v3")
    raise ValueError(f"unknown parser {parser!r}")


def ingest(conn, docs: list[str], pdf_record: dict, manifest: dict, parser: str = "pdf-2") -> dict:
    counts = {}
    for doc in docs:
        entry = pdf_record[doc]
        parsed = _parse(Path(entry["path"]), doc, parser)
        conn.execute("INSERT INTO documents (id, source_uri, sha256, artifact_path, retrieved_at, company, parser_version) "
                     "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                     (document_id(doc, parser), f"{manifest['documents'][doc]['doc_link']}#parser={parser}", entry["sha256"],
                      entry["path"], datetime.now(timezone.utc), doc.split("_")[0], parser))
        counts[doc] = store_passages(conn, parsed.passages)
    return counts


def verify_ingested(conn, docs: list[str], pdf_record: dict, parser: str = "pdf-2") -> dict:
    """Re-parse each PDF and compare passage ids and text with what the database holds."""
    out = {}
    for doc in docs:
        parsed = _parse(Path(pdf_record[doc]["path"]), doc, parser)
        stored = {r["id"]: r["text"] for r in conn.execute("SELECT id, text FROM passages WHERE document_id = %s",
                                                            (document_id(doc, parser),)).fetchall()}
        fresh = {p.id: p.text for p in parsed.passages}
        out[doc] = "identical" if stored == fresh else f"differs: stored {len(stored)}, parsed {len(fresh)}, " \
                                                      f"text mismatches {sum(stored.get(k) != v for k, v in fresh.items())}"
    return out


def page_coverage(retrieved_pages: set[int], gold_pages: list[int]) -> str:
    hit = set(gold_pages) & retrieved_pages
    return "all" if hit == set(gold_pages) else "some" if hit else "none"


def company_name(question: dict) -> str:
    """The document's company as FinanceBench names it, plus the document name's company token."""
    return f"{question['company']} {question['doc_name'].split('_')[0]}"


def gold_pages(question: dict) -> list[int]:
    return sorted({e["evidence_page_num"] + 1 for e in question["evidence"]})


def retrieve(conn, questions: list[dict], version: str, parser: str = "pdf-2", *, limit: int = 25,
             char_budget: int = 8000) -> dict:
    out = {}
    for q in questions:
        rows = search_text(conn, [document_id(q["doc_name"], parser)], q["question"], limit=limit,
                           char_budget=char_budget, version=version, company=company_name(q) if version == "text-3" else None)
        passages = [{"id": r["id"], "page": r["location"]["page"], "row": r["location"].get("row"),
                     "table": r["location"].get("table"), "text": r["text"],
                     "sha256": hashlib.sha256(r["text"].encode()).hexdigest()} for r in rows]
        pages = gold_pages(q)
        out[q["financebench_id"]] = {"doc": q["doc_name"], "gold_pages": pages, "passages": passages,
                                     "gold_pages_retrieved": page_coverage({p["page"] for p in passages}, pages)}
    return summarize(out, version, parser, limit, char_budget)


def summarize(questions: dict, version: str, parser: str, limit: int, char_budget: int) -> dict:
    n = len(questions)
    counts = {k: sum(q["gold_pages_retrieved"] == k for q in questions.values()) for k in ("all", "none", "some")}
    return {"parser_version": parser, "retrieval": f"{version} over {parser} passages; single-document; up to {limit} passages "
                                                    f"within {char_budget:,} characters; query = question text only",
            "gold_pages_retrieved": counts,
            "page_recall": {"all_gold_pages": round(counts["all"] / n, 3), "any_gold_page": round((counts["all"] + counts["some"]) / n, 3)},
            "questions": questions}


def compare(a: dict, b: dict) -> dict:
    """Per-question passage-set equality between two retrieval files (ids and text hashes)."""
    same = [k for k in a["questions"] if [(p["id"], p["sha256"]) for p in a["questions"][k]["passages"]]
            == [(p["id"], p["sha256"]) for p in b["questions"].get(k, {}).get("passages", [])]]
    return {"questions": len(a["questions"]), "identical_passage_lists": len(same),
            "coverage_a": a["gold_pages_retrieved"], "coverage_b": b["gold_pages_retrieved"]}


def breakdown(retrieval: dict, questions: list[dict], page_counts: dict[str, int]) -> dict:
    """Coverage by question type, number of gold pages and document length."""
    by_id = {q["financebench_id"]: q for q in questions}
    groups: dict[str, dict[str, int]] = {}
    for qid, r in retrieval["questions"].items():
        q = by_id[qid]
        pages = page_counts.get(r["doc"], 0)
        for key in (f"type:{q['question_type']}", "gold:multi-page" if len(r["gold_pages"]) > 1 else "gold:single-page",
                    "doc:>100 pages" if pages > 100 else "doc:<=100 pages"):
            g = groups.setdefault(key, {"all": 0, "some": 0, "none": 0})
            g[r["gold_pages_retrieved"]] += 1
    return dict(sorted(groups.items()))


def write(path: Path, data: dict) -> str:
    body = json.dumps(data, indent=1, sort_keys=True) + "\n"
    path.write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()


# ----------------------------------------------------------------------------- evidence-content audit
# A page can be retrieved without the row or sentence the answer needs. The audit fixes, per development question and
# before any retrieval change is measured, the evidence "needles" that must reach the model:
#   * numeric questions with an alignment-audit entry: each needed row label with its value for each requested year;
#   * otherwise: the answer-bearing lines of FinanceBench's gold evidence text (lines holding a figure the answer or
#     justification uses, else lines sharing two content words with the answer; at most six per question).
# A needle is reached when one retrieved passage holds all of its figures, at least half of its content words and,
# for table values, the requested year.

_FIG = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+\.\d+|(?<![\w.,])\d{2,}(?![\d,])")
_WORD = re.compile(r"[a-z][a-z&-]{3,}")
_STOP = {"that", "this", "with", "from", "were", "have", "which", "their", "also", "than", "into", "during", "other", "such",
         "these", "those", "over", "under", "total", "year", "years", "ended", "company", "million", "millions", "billion"}


def figures(text: str) -> set[str]:
    return {f.replace(",", "") for f in _FIG.findall(text or "") if not re.fullmatch(r"(19|20)\d{2}", f)}


def words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def evidence_needles(question: dict, alignment: dict | None) -> list[dict]:
    if alignment and alignment.get("items"):
        needles = []
        for item in alignment["items"]:
            for year in alignment.get("requested") or []:
                value = (item.get("values") or {}).get(str(year))
                if value:
                    needles.append({"kind": "table_value", "label": item["name"], "year": year, "value": value})
        if needles:
            return needles
    answer = f"{question.get('answer') or ''} {question.get('justification') or ''}"
    used, answer_words = figures(answer), words(answer)
    lines = [ln.strip() for e in question["evidence"] for ln in e["evidence_text"].splitlines() if ln.strip()]
    labels = {}
    for i, ln in enumerate(lines):          # a line holding only figures takes the nearest preceding text line as its label
        if figures(ln) and not words(ln):
            labels[i] = next((lines[j] for j in range(i - 1, max(-1, i - 6), -1) if words(lines[j])), "")
    keep = [i for i, ln in enumerate(lines) if len(ln) > 3]
    chosen = ([i for i in keep if figures(lines[i]) & used] or [i for i in keep if len(words(lines[i]) & answer_words) >= 2]
              or [i for i in keep if figures(lines[i])] or [i for i in keep if len(lines[i]) >= 30][:3])
    return [{"kind": "evidence_line", "text": lines[i], **({"label": labels[i]} if labels.get(i) else {})} for i in chosen[:6]]


def _squash(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def needle_reached(needle: dict, passages: list[str]) -> bool:
    if needle["kind"] == "table_value":
        value = needle["value"].strip("()$ ").replace(",", "")
        label = words(needle["label"])
        for text in passages:
            low = text.lower()
            label_ok = not label or len(words(low) & label) >= len(label) / 2 or _squash(needle["label"]) in _squash(text)
            if value in text.replace(",", "") and str(needle["year"]) in text and label_ok:
                return True
        return False
    need_figs, need_words = figures(needle["text"]), words(needle["text"] + " " + needle.get("label", ""))
    for text in passages:
        if not need_figs <= figures(text):
            continue
        if not need_words or len(words(text) & need_words) >= len(need_words) / 2 or \
                (needle.get("label") and _squash(needle["label"]) in _squash(text)):
            return True
    return False


def evidence_coverage(retrieval: dict, audit: dict) -> dict:
    per_q, counts = {}, {"complete": 0, "partial": 0, "none": 0, "no_needles": 0}
    for qid, r in retrieval["questions"].items():
        needles = audit["questions"][qid]["needles"]
        texts = [p["text"] for p in r["passages"]]
        reached = [needle_reached(n, texts) for n in needles]
        label = ("no_needles" if not needles else "complete" if all(reached) else "partial" if any(reached) else "none")
        counts[label] += 1
        per_q[qid] = {"evidence": label, "reached": sum(reached), "needles": len(needles), "pages": r["gold_pages_retrieved"]}
    cross = {}
    for v in per_q.values():
        cross[f"pages {v['pages']} / evidence {v['evidence']}"] = cross.get(f"pages {v['pages']} / evidence {v['evidence']}", 0) + 1
    return {"evidence": counts, "pages_vs_evidence": dict(sorted(cross.items())), "questions": per_q}
