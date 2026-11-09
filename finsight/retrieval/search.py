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
_SECTION_BONUS = {"item_8": 1.0, "item_7": 0.5, "item_6": 0.5}
# Tables that report a slice of the company (segments, geographies, quarters) or non-GAAP measures share row
# labels with the consolidated statements; they rank below them.
_SCOPED = re.compile(r"quarter|three months|six months|nine months|segment|business \(|% of consolidated|"
                     r"non-gaap|geographic|united states|asia pacific|europe|latin america", re.I)
_PERIOD_WORDS = re.compile(r"\b(?:years?|fiscal|ended|ending|as of|january|february|march|april|may|june|july|august|"
                           r"september|october|november|december|\d{1,2})\b|[*,.:()\s]", re.I)
_EXACT_EXTRA = {"net_income": re.compile(r"^net (?:income|earnings) attributable to (?!non)[\w .,&'-]+$")}


def _cells(text: str) -> tuple[str, list[tuple[str, str]], str]:
    body, _, meta = text.partition("  [table:")
    parts = body.split(" | ")
    cells = [(c.rpartition(": ")[0], c.rpartition(": ")[2]) for c in parts[1:]]
    return parts[0].strip(), cells, meta


def _norm_label(label: str) -> str:
    value = label.lower()
    value = re.sub(r"\((?:in )?(?:millions|thousands|billions)[^)]*\)|\([a-z&]{2,6}\)|[*\u2020\u2021]|\(\d\)", "", value)
    value = re.sub(r"^total\s+", "", value.strip())
    return " ".join(re.sub(r"[\s:,.]+$", "", value).split())


def label_quality(text: str, metric: str, is_table: bool) -> str | None:
    """'exact' when the row label is the metric itself, 'partial' when it only contains a metric term."""
    terms = METRICS[metric]["terms"]
    if not is_table:
        return "partial" if label_matches(text, terms, False) else None
    label = _norm_label(_cells(text)[0])
    if label in terms or (metric in _EXACT_EXTRA and _EXACT_EXTRA[metric].match(label)):
        return "exact"
    return "partial" if label_matches(text, terms, True) else None


def period_quality(text: str, year: int, is_table: bool) -> str | None:
    """'clean' when a column header is just the fiscal year (e.g. "2018", "Years ended December 31, 2018"),
    'qualified' when the header adds a scope (quarter, geography); None when no value is reported for the year."""
    if not is_table:
        return "clean" if period_matches(text, year, False) else None
    found = None
    for header, value in _cells(text)[1]:
        if re.search(rf"\b{year}\b", header) and value.strip() and value.strip() not in ("—", "-", "$"):
            if not _PERIOD_WORDS.sub("", header.replace(str(year), "")):
                return "clean"
            found = "qualified"
    return found


def _sentence_reports(text: str, terms: list[str], year: int) -> bool:
    """Prose counts only when one sentence names the metric, the year and a figure."""
    for sentence in re.split(r"(?<=[.;])\s+", text):
        low = sentence.lower()
        if (re.search(rf"\b{year}\b", low) and re.search(r"\$?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|%)", low)
                and any(re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", low) for t in terms)):
            return True
    return False


@dataclass
class Candidate:
    id: str
    document_id: str
    section: str
    location: dict
    text: str
    rank: float
    label: str | None                   # exact | partial | None
    period: str | None                  # clean | qualified | None
    scoped: bool = False
    prose_reports: bool = False

    @property
    def is_table(self) -> bool:
        return self.location.get("table") is not None

    @property
    def label_match(self) -> bool:
        return self.label == "exact" if self.is_table else self.prose_reports

    @property
    def period_match(self) -> bool:
        return self.period is not None

    @property
    def score(self) -> float:
        bonus = _SECTION_BONUS.get(self.section, 0.0)
        if self.is_table:
            return (min(self.rank, 1.0) + {"exact": 4.0, "partial": 1.0}.get(self.label, 0.0)
                    + {"clean": 3.0, "qualified": 1.0}.get(self.period, 0.0) + bonus - 3.0 * self.scoped)
        return min(self.rank, 1.0) + 2.0 * self.prose_reports + 0.5 * bonus


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
        f"ORDER BY rank DESC LIMIT 5000", (*terms, document_ids)).fetchall()
    candidates = []
    for row in rows:
        is_table = row["location"].get("table") is not None
        _, _, meta = _cells(row["text"])
        headers = " ".join(h for h, _ in _cells(row["text"])[1]) if is_table else ""
        candidates.append(Candidate(id=row["id"], document_id=row["document_id"], section=row["section"],
                                    location=row["location"], text=row["text"], rank=float(row["rank"]),
                                    label=label_quality(row["text"], metric, is_table),
                                    period=period_quality(row["text"], year, is_table),
                                    scoped=is_table and bool(_SCOPED.search(meta + " " + headers)),
                                    prose_reports=not is_table and _sentence_reports(row["text"], terms, year)))
    candidates.sort(key=lambda c: (c.score, -len(c.text)), reverse=True)
    top = candidates[:limit]
    if any(c.label_match and c.period_match for c in candidates):
        return Retrieval(metric, year, "found", top)
    reason = ("no passage mentions the metric" if not candidates else
              f"passages mention the metric but none reports it for {year}")
    return Retrieval(metric, year, "missing", top, reason)


# ----------------------------------------------------------------------------- general questions

TEXT_SEARCH_VERSION = "text-1"
_QUERY_WORD = re.compile(r"[a-z][a-z0-9]{2,}|(?:19|20)\d{2}")


def query_terms(question: str) -> list[str]:
    """Words from a free-text question for an OR full-text query (FY2018 -> 2018; English stop words are dropped
    by Postgres)."""
    text = re.sub(r"\bfy\s?((?:19|20)\d{2})\b", r"\1", question.lower())
    seen: list[str] = []
    for word in _QUERY_WORD.findall(text):
        if word not in seen:
            seen.append(word)
    return seen


# text-2 (tuned on FinanceBench development questions; the held-out half gives the honest recall)
TEXT_SEARCH_V2 = "text-2"
INSTRUCTION_WORDS = {"give", "response", "question", "relying", "details", "shown", "answer", "using", "primarily",
                     "information", "assume", "public", "equities", "analyst", "following", "based", "what", "which",
                     "company", "much", "many", "calculate", "round", "decimal", "places", "reported", "provide",
                     "fiscal", "year", "amount", "usd", "millions", "billions", "thousands",
                     "units", "help", "please", "between", "rely", "basis", "according", "describe", "shows", "data"}
EXPANSIONS = {"capex": ["purchases", "property", "plant", "equipment"], "capital": ["purchases", "property", "plant", "equipment"],
              "revenue": ["sales", "revenues"], "revenues": ["sales"], "sales": ["revenue", "revenues"],
              "cogs": ["cost", "sales"], "ebitda": ["depreciation", "amortization", "operating", "income"],
              "margin": ["income", "sales"], "profitability": ["income", "margin"], "liquidity": ["cash", "current", "assets", "liabilities"],
              "leverage": ["debt", "equity"], "dividend": ["dividends", "paid"], "dividends": ["paid"],
              "inventory": ["inventories"], "inventories": ["inventory"], "receivable": ["receivables"], "payable": ["payables"]}


def search_text(conn, document_ids: list[str], question: str, *, limit: int = 10, char_budget: int = 8000,
                version: str = "text-2") -> list[dict]:
    """General retrieval for any question within the given documents (single-document setting): Postgres full-text
    OR query over the question's words, ranked by cover density; top passages up to ``limit`` and ``char_budget``.
    text-1 uses every word; text-2 (tuned on FinanceBench development questions) drops instruction words and adds a
    few financial synonyms."""
    terms = query_terms(question)
    if version == "text-2":
        terms = [t for t in terms if t not in INSTRUCTION_WORDS]
        terms += [x for t in list(terms) for x in EXPANSIONS.get(t, []) if x not in terms]
    if not terms:
        return []
    rows = conn.execute(
        "WITH q AS (SELECT to_tsquery('english', %s) AS query) SELECT p.id, p.document_id, p.section, p.location, p.text, "
        "p.start_offset, p.end_offset, ts_rank_cd(p.tsv, q.query) AS rank FROM passages p, q "
        "WHERE p.document_id = ANY(%s) AND p.tsv @@ q.query ORDER BY rank DESC, p.id LIMIT 200",
        (" | ".join(terms), document_ids)).fetchall()
    chosen, used = [], 0
    for row in rows:
        if len(chosen) >= limit or used + len(row["text"]) > char_budget:
            continue
        chosen.append(dict(row))
        used += len(row["text"])
    return chosen
