"""Evidence retrieval: Postgres full-text search with metric synonyms, then metric + period checks.

Finding the word "revenue" is not enough: a candidate counts only when the metric matches
its row label (or the sentence) **and** the requested period appears with a value. If no
passage satisfies both, retrieval reports the evidence as missing instead of guessing.
Embeddings are not used until measured retrieval failures justify them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

METRICS: dict[str, dict] = {
    "revenue": {"terms": ["net sales", "total net sales", "revenues", "revenue", "total revenues", "net revenues",
                          "sales"],
                "concepts": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]},
    "operating_income": {"terms": ["operating income", "income from operations", "operating profit"],
                         "concepts": ["OperatingIncomeLoss"]},
    "capex": {"terms": ["purchases of property, plant and equipment", "capital expenditures",
                        "purchases of property and equipment", "additions to property, plant and equipment",
                        "capital spending"],
              "concepts": ["PaymentsToAcquirePropertyPlantAndEquipment"]},
    "net_income": {"terms": ["net income", "net earnings"], "concepts": ["NetIncomeLoss"]},
}
DERIVED = {"operating_margin": ("operating_income", "revenue")}
_SECTION_BONUS = {"item_8": 1.0, "item_7": 0.5}


@dataclass
class Candidate:
    id: str
    document_id: str
    section: str
    location: dict
    text: str
    rank: float
    label_match: bool
    period_match: bool

    @property
    def score(self) -> float:
        return self.rank + 2.0 * self.label_match + 2.0 * self.period_match + _SECTION_BONUS.get(self.section, 0.0)


@dataclass
class Retrieval:
    metric: str
    period: int
    status: str                         # "found" | "missing"
    candidates: list[Candidate] = field(default_factory=list)
    reason: str | None = None

    @property
    def best(self) -> Candidate | None:
        good = [c for c in self.candidates if c.label_match and c.period_match]
        return good[0] if good else None


def store_passages(conn, passages) -> int:
    rows = [(p.id, p.document_id, p.section, p.location.model_dump_json(), p.start, p.end, p.text) for p in passages]
    with conn.cursor() as cursor:
        cursor.executemany("INSERT INTO passages (id, document_id, section, location, start_offset, end_offset, text) "
                           "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING", rows)
    return len(rows)


def _label(text: str) -> str:
    return text.split(" | ")[0].strip().lower()


def label_matches(text: str, terms: list[str], is_table: bool) -> bool:
    target = _label(text) if is_table else text.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", target) for t in terms)


def period_matches(text: str, year: int, is_table: bool) -> bool:
    if is_table:  # a column header naming the year must carry a value in this row
        return bool(re.search(rf"\b{year}\b[^|:]*:\s*[^|\[\s]", text))
    return bool(re.search(rf"\b{year}\b", text))


def search(conn, document_ids: list[str], metric: str, year: int, limit: int = 10) -> Retrieval:
    if metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; known: {sorted(METRICS)} (derived: {sorted(DERIVED)})")
    terms = METRICS[metric]["terms"]
    query = " || ".join(["phraseto_tsquery('english', %s)"] * len(terms))
    rows = conn.execute(
        f"WITH q AS (SELECT ({query}) AS query) SELECT p.id, p.document_id, p.section, p.location, p.text, "
        f"ts_rank_cd(p.tsv, q.query) AS rank FROM passages p, q WHERE p.document_id = ANY(%s) AND p.tsv @@ q.query "
        f"ORDER BY rank DESC LIMIT 200", (*terms, document_ids)).fetchall()
    candidates = []
    for row in rows:
        is_table = row["location"].get("table") is not None
        candidates.append(Candidate(id=row["id"], document_id=row["document_id"], section=row["section"],
                                    location=row["location"], text=row["text"], rank=float(row["rank"]),
                                    label_match=label_matches(row["text"], terms, is_table),
                                    period_match=period_matches(row["text"], year, is_table)))
    candidates.sort(key=lambda c: c.score, reverse=True)
    top = candidates[:limit]
    if any(c.label_match and c.period_match for c in candidates):
        return Retrieval(metric, year, "found", top)
    reason = ("no passage mentions the metric" if not candidates else
              f"passages mention the metric but none reports it for {year}")
    return Retrieval(metric, year, "missing", top, reason)
