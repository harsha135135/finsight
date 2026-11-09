"""PDF filings (e.g. FinanceBench's original PDFs): page text through PDFium, split into page-anchored passages.

Each passage records its 1-based page, so a citation can be checked against the original. Lines are kept intact,
so where the PDF lays out a table row on one line its label and figures stay together; no further table structure
is inferred. Pages without extractable text (scanned images) are counted, never guessed.
"""

from __future__ import annotations

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
