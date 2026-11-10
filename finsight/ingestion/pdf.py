"""PDF filings (e.g. FinanceBench's original PDFs): page text through PDFium, split into page-anchored passages.

Each passage records its 1-based page, so a citation can be checked against the original. Lines are kept intact,
so where the PDF lays out a table row on one line its label and figures stay together; no further table structure
is inferred. Pages without extractable text (scanned images) are counted, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium

from finsight.contracts import Passage, PassageLocation

PARSER_VERSION = "pdf-1"
MAX_CHARS = 700


@dataclass
class PdfParse:
    passages: list[Passage] = field(default_factory=list)
    pages: int = 0
    empty_pages: list[int] = field(default_factory=list)
    chars: int = 0


def parse_pdf(path: Path, document_id: str, doc_short: str) -> PdfParse:
    result = PdfParse()
    offset = 0
    pdf = pdfium.PdfDocument(str(path))
    try:
        result.pages = len(pdf)
        for index in range(len(pdf)):
            page_no = index + 1
            page = pdf[index]
            try:
                text = page.get_textpage().get_text_range()
            finally:
                page.close()
            lines = [" ".join(line.split()) for line in text.replace("\r", "\n").split("\n")]
            lines = [line for line in lines if line]
            if not lines:
                result.empty_pages.append(page_no)
                continue
            chunks: list[list[str]] = [[]]
            for line in lines:
                if chunks[-1] and sum(len(x) + 1 for x in chunks[-1]) + len(line) > MAX_CHARS:
                    chunks.append([])
                chunks[-1].append(line)
            for n, chunk in enumerate(chunks):
                body = "\n".join(chunk)
                result.passages.append(Passage(
                    id=f"{doc_short}:p{page_no}:c{n}", document_id=document_id, section=f"page_{page_no}",
                    location=PassageLocation(page=page_no), start=offset, end=offset + len(body), text=body))
                offset += len(body) + 1
                result.chars += len(body)
    finally:
        pdf.close()
    return result


# ----------------------------------------------------------------------------- pdf-2: table rows with column years

PARSER_VERSION_V2 = "pdf-2"
_CELL = re.compile(r"^(?:\(?\$?\(?-?[\d,]*\.?\d+\)?%?\)?|[—–-])$")
_SPAN = re.compile(r"(?i)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+)?(?:19|20)\d{2}(?!\d)")
_HEADER_NOISE = re.compile(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?|\b(?:19|20)\d{2}\b")


def _split_row(line: str) -> tuple[str, list[str]]:
    tokens = line.split()
    cells: list[str] = []
    i = len(tokens)
    while i > 0:
        token = tokens[i - 1]
        if token == "$":
            i -= 1
            continue
        balanced = token.count("(") == token.count(")")          # "6)" from "(Note 6)" is not a value cell
        if _CELL.match(token) and balanced and any(ch.isdigit() for ch in token) or token in ("—", "–", "-"):
            cells.insert(0, token)
            i -= 1
            continue
        break
    return " ".join(tokens[:i]), cells


def _header(line: str) -> list[str] | None:
    """Column labels if the line lists two or more years (or dates) and no other figures."""
    spans = [m.group(0) for m in _SPAN.finditer(line)]
    if len(spans) < 2:
        return None
    rest = _HEADER_NOISE.sub(" ", line)
    return None if re.search(r"\d", rest) else spans


def table_rows(lines: list[str]) -> list[tuple[int, str] | tuple[int, None]]:
    """For each line: (index, row text in 'label | col: value' form) when it is a numeric row under a header with the
    same number of columns; (index, None) otherwise. A label line followed by a values-only line is merged."""
    out, header = [], None
    i = 0
    while i < len(lines):
        line = lines[i]
        found = _header(line)
        if found:
            header = found
            out.append((i, None))
            i += 1
            continue
        label, cells = _split_row(line)
        if header and not cells and i + 1 < len(lines):
            next_label, next_cells = _split_row(lines[i + 1])
            if not next_label and len(next_cells) == len(header):
                out.append((i, None))
                out.append((i + 1, f"{label} | " + " | ".join(f"{h}: {c}" for h, c in zip(header, next_cells))))
                i += 2
                continue
        if header and label and len(cells) == len(header):
            out.append((i, f"{label} | " + " | ".join(f"{h}: {c}" for h, c in zip(header, cells))))
        else:
            out.append((i, None))
        i += 1
    return out


_STATEMENT = re.compile(r"(?i)\b(consolidated\s+)?(balance\s+sheets?|statements?\s+of\s+(?:consolidated\s+)?(?:cash\s+flows?|operations|income|"
                        r"earnings|comprehensive\s+(?:income|loss|earnings)|financial\s+(?:position|condition)|"
                        r"(?:stockholders|shareholders|shareowners)[’']?\s+equity))\b")


def statement_title(lines: list[str]) -> str | None:
    """The financial statement a page presents, from its first lines (e.g. 'Consolidated Balance Sheets')."""
    head = " ".join(lines[:8])
    m = _STATEMENT.search(head)
    return " ".join(m.group(0).split()).title() if m else None


def parse_pdf_v3(path: Path, document_id: str, doc_short: str) -> PdfParse:
    """pdf-3: pdf-2 plus the page's statement title on each table row ('[Consolidated Balance Sheets] label | 2018: v
    ...'), so a question naming the statement can find its rows. Text chunks are unchanged."""
    return parse_pdf_v2(path, document_id, doc_short, label_statements=True)


def parse_pdf_v2(path: Path, document_id: str, doc_short: str, *, label_statements: bool = False) -> PdfParse:
    """pdf-2: like pdf-1, but numeric table rows under a year/date header become row passages
    ('label | 2018: v | 2017: v  [table: pN]') that FinSight's period checks can read; other lines become text chunks."""
    result = PdfParse()
    offset = 0
    pdf = pdfium.PdfDocument(str(path))
    try:
        result.pages = len(pdf)
        for index in range(len(pdf)):
            page_no = index + 1
            page = pdf[index]
            try:
                text = page.get_textpage().get_text_range()
            finally:
                page.close()
            lines = [" ".join(line.split()) for line in text.replace("\r", "\n").split("\n")]
            lines = [line for line in lines if line]
            if not lines:
                result.empty_pages.append(page_no)
                continue
            rows = dict(table_rows(lines))
            title = statement_title(lines) if label_statements else None
            chunk: list[str] = []
            counters = {"c": 0, "r": 0}

            def emit(body: str, *, row: bool) -> None:
                nonlocal offset
                kind = "r" if row else "c"
                n = counters[kind]
                counters[kind] += 1
                location = PassageLocation(page=page_no, table=f"p{page_no}", row=n) if row else PassageLocation(page=page_no)
                text_ = f"{body}  [table: page {page_no}]" if row else body
                result.passages.append(Passage(id=f"{doc_short}:p{page_no}:{kind}{n}", document_id=document_id,
                                               section=f"page_{page_no}", location=location, start=offset,
                                               end=offset + len(text_), text=text_))
                offset += len(text_) + 1
                result.chars += len(text_)

            for i, line in enumerate(lines):
                row = rows.get(i)
                if row:
                    if chunk:
                        emit("\n".join(chunk), row=False)
                        chunk = []
                    emit(f"[{title}] {row}" if title else row, row=True)
                    continue
                if chunk and sum(len(x) + 1 for x in chunk) + len(line) > MAX_CHARS:
                    emit("\n".join(chunk), row=False)
                    chunk = []
                chunk.append(line)
            if chunk:
                emit("\n".join(chunk), row=False)
    finally:
        pdf.close()
    return result
