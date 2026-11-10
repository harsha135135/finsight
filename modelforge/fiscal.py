"""Fiscal-year-end derivation for FinanceBench companies (rule frozen 2026-11-10, before any held-out document is read).

Rule:
1. For each company, read the cover page (first three pages) of every 10-K of that company in the corpus and collect
   the dates in "for the (fiscal) year ended <Month D, YYYY>".
2. Companies with 52/53-week years report different dates in different years. The fiscal-year end (month, day) is the
   earliest date of the smallest window that holds all of them; the window may not exceed 7 days, which is the
   tolerance the validator applies (`finsight.qa.fiscal_year`). A wider spread is an error for a person to resolve.
3. A company with no 10-K in the corpus gets null, which the validator reads as calendar years. It is listed, because
   any period question about such a company deserves a manual look.

The output has the same format as ``evals/financebench/fiscal-years-dev-v1.json``, plus the evidence (document and cover
date) behind each entry. ``agreement`` checks the rule against the hand-made development file: every cover date must
get the same fiscal year under both.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                        "september", "october", "november", "december"], 1)}
_COVER = re.compile(r"(?i)for\s+the\s+(?:fiscal\s+)?year\s+ended:?\s+([A-Za-z]+)\.?\s+(\d{1,2})\s*,?\s*((?:19|20)\d{2})")
TOLERANCE_DAYS = 7


def cover_dates(text: str) -> list[date]:
    out = []
    for month, day, year in _COVER.findall(text):
        m = _MONTHS.get(month.lower()) or next((v for k, v in _MONTHS.items() if k.startswith(month.lower()[:3])), None)
        if m:
            out.append(date(int(year), m, int(day)))
    return out


def fiscal_year_end(dates: list[date]) -> tuple[int, int]:
    """Earliest (month, day) of the smallest window (wrapping the year end) holding every date's (month, day)."""
    if not dates:
        raise ValueError("no cover dates")
    days = sorted({(d.month, d.day) for d in dates})
    as_ordinal = [date(2001, m, d).timetuple().tm_yday for m, d in days]
    best = None
    for i, start in enumerate(as_ordinal):
        spread = max((o - start) % 365 for o in as_ordinal)
        if best is None or spread < best[0]:
            best = (spread, days[i])
    if best[0] > TOLERANCE_DAYS:
        raise ValueError(f"cover dates {days} spread over {best[0]} days (more than {TOLERANCE_DAYS})")
    return best[1]


def first_pages_text(pdf: Path, pages: int = 3) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        parts = []
        for i in range(min(pages, len(doc))):
            page = doc[i]
            try:
                parts.append(page.get_textpage().get_text_range())
            finally:
                page.close()
        return "\n".join(parts)
    finally:
        doc.close()


def company_key(company: str) -> str:
    return company.upper().replace(" ", "").replace("&", "")


def derive(company_docs: dict[str, list[tuple[str, str, Path]]]) -> dict:
    """company -> [(doc_name, doc_type, pdf_path)]; returns the fiscal-year file content with evidence."""
    table, evidence, nulls = {}, {}, []
    for company, docs in sorted(company_docs.items()):
        found = []
        for name, doc_type, path in docs:
            if doc_type.lower() != "10k":
                continue
            found += [(name, d.isoformat()) for d in cover_dates(first_pages_text(path))[:1]]
        key = company_key(company)
        if not found:
            table[key] = None
            nulls.append(key)
            continue
        table[key] = list(fiscal_year_end([date.fromisoformat(d) for _, d in found]))
        evidence[key] = found
    return {"rule": __doc__.split("Rule:")[1].split("The output")[0].strip(), "fiscal_year_end": table,
            "evidence": evidence, "no_10k_in_corpus": nulls}


def fiscal_year(day: date, fye: tuple[int, int] | None) -> int:
    """The validator's rule: the year Y whose fiscal-year end plus 7 days is on or after the date."""
    if fye is None:
        return day.year
    end = date(day.year, *fye) + timedelta(days=TOLERANCE_DAYS)
    return day.year if day <= end else day.year + 1


def agreement(derived: dict, hand: dict) -> dict:
    """Same fiscal year for every cover date under the derived and the hand-made (month, day)."""
    out = {}
    for key, found in derived["evidence"].items():
        h, d = hand["fiscal_year_end"].get(key), derived["fiscal_year_end"][key]
        out[key] = all(fiscal_year(date.fromisoformat(day), tuple(h) if h else None) ==
                       fiscal_year(date.fromisoformat(day), tuple(d)) for _, day in found)
    for key in derived["no_10k_in_corpus"]:
        out[key] = hand["fiscal_year_end"].get(key) is None
    return out
