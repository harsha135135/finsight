"""FinanceBench (CC-BY-NC 4.0): evaluation only, at a pinned commit.

Never used for training or prompt development on held-out companies. Gold answers, justifications and evidence
are scoring material: full-document mode gives the application no gold locations and no answers; supplied-evidence
mode gives the model the annotated evidence text by design (it measures reasoning, not retrieval).

Documents. Every one of the 84 documents the 150 open-source questions use is present in the repository at the
pinned commit as the benchmark's own PDF (category "original"). The other two categories exist for documents that
are missing or unreadable: "verified_equivalent" (another rendition shown to be the same filing by provenance,
e.g. the SEC accession in the document link, and content checks that do not read gold evidence) and
"adapted_substitute" (chosen because it contains the evidence; reported only as a separate adapted-corpus result).

Split. Companies, not questions, are divided into development and held-out halves with a fixed seed before any
iteration, so no company's filings inform choices that are then scored on that company. 3M is in development (it is
the demo company).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.request
from collections import Counter
from pathlib import Path

REPO = "patronus-ai/financebench"
REVISION = "cc39aeb4afdf33909ee1412188bf89035950c2eb"           # 2024-12-03, "Replace corrupted pdf"
FILES = {"financebench_open_source.jsonl": "a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
         "financebench_document_information.jsonl": "1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575"}
SPLIT_SEED = 20261107
# For the FinQA overlap check only (FinQA ids are TICKER/YEAR/page). Block traded as SQ during FinQA's period.
TICKERS = {"3M": "MMM", "AES Corporation": "AES", "AMD": "AMD", "Activision Blizzard": "ATVI", "Adobe": "ADBE",
           "Amazon": "AMZN", "Amcor": "AMCR", "American Express": "AXP", "American Water Works": "AWK",
           "Best Buy": "BBY", "Block": "SQ", "Boeing": "BA", "CVS Health": "CVS", "Coca-Cola": "KO", "Corning": "GLW",
           "Costco": "COST", "Foot Locker": "FL", "General Mills": "GIS", "JPMorgan": "JPM",
           "Johnson & Johnson": "JNJ", "Kraft Heinz": "KHC", "Lockheed Martin": "LMT", "MGM Resorts": "MGM",
           "Microsoft": "MSFT", "Netflix": "NFLX", "Nike": "NKE", "Paypal": "PYPL", "PepsiCo": "PEP", "Pfizer": "PFE",
           "Ulta Beauty": "ULTA", "Verizon": "VZ", "Walmart": "WMT"}


def data_dir(revision: str = REVISION) -> Path:
    root = Path(os.environ.get("FINSIGHT_DATA_DIR", Path.home() / ".cache" / "finsight" / "datasets"))
    return root / "financebench" / revision


def download(revision: str = REVISION) -> Path:
    out = data_dir(revision)
    out.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        path = out / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            url = f"https://raw.githubusercontent.com/{REPO}/{revision}/data/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (fixed https URL)
                body = response.read()
            if hashlib.sha256(body).hexdigest() != sha:
                raise ValueError(f"{name}: sha256 mismatch for revision {revision}")
            path.write_bytes(body)
    return out


def load(revision: str = REVISION) -> tuple[list[dict], dict[str, dict]]:
    out = download(revision)
    questions = [json.loads(line) for line in (out / "financebench_open_source.jsonl").read_text().splitlines() if line]
    documents = {d["doc_name"]: d for d in (json.loads(line) for line in
                 (out / "financebench_document_information.jsonl").read_text().splitlines() if line)}
    return questions, documents


def split_companies(questions: list[dict], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Seeded company split balancing question counts; 3M goes to development."""
    counts = Counter(q["company"] for q in questions)
    companies = sorted(counts)
    random.Random(seed).shuffle(companies)
    companies.remove("3M")
    halves = {"dev": ["3M"], "held_out": []}
    totals = {"dev": counts["3M"], "held_out": 0}
    for company in sorted(companies, key=lambda c: -counts[c]):  # largest first, then fill the smaller half
        side = min(totals, key=lambda s: (totals[s], s != "held_out"))
        halves[side].append(company)
        totals[side] += counts[company]
    return {k: sorted(v) for k, v in halves.items()}


def build_manifest(finqa_train_ids: list[str]) -> dict:
    questions, documents = load()
    halves = split_companies(questions)
    side = {c: s for s, cs in halves.items() for c in cs}
    finqa_company_years = {tuple(i.split("/")[:2]) for i in finqa_train_ids}
    overlap = sorted({f"{q['doc_name']} ~ FinQA {TICKERS[q['company']]}/{documents[q['doc_name']]['doc_period']}"
                      for q in questions
                      if (TICKERS[q["company"]], str(documents[q["doc_name"]]["doc_period"])) in finqa_company_years})
    used = sorted({q["doc_name"] for q in questions})
    manifest = {
        "name": "financebench-v1", "dataset": "financebench", "repo": REPO, "revision": REVISION, "files": FILES,
        "license": "CC-BY-NC-4.0; evaluation only",
        "split_seed": SPLIT_SEED, "companies": halves,
        "samples": {s: sorted(q["financebench_id"] for q in questions if side[q["company"]] == s) for s in halves},
        "counts": {s: dict(Counter(q["question_type"] for q in questions if side[q["company"]] == s)) | {
            "questions": sum(side[q["company"]] == s for q in questions),
            "documents": len({q["doc_name"] for q in questions if side[q["company"]] == s})} for s in halves},
        "documents": {d: {"category": "original", "source": f"pdfs/{d}.pdf at {REVISION}",
                          "doc_link": documents[d]["doc_link"], "doc_type": documents[d]["doc_type"],
                          "doc_period": documents[d]["doc_period"]} for d in used},
        "document_coverage": f"{len(used)} of {len(used)} documents available as the benchmark's original PDF",
        "finqa_train_overlap": {"rule": "same ticker and year (FinanceBench doc_period vs the FinQA id's year folder)",
                                "matches": overlap},
    }
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest
