"""Parse an SEC filing's HTML into passages that keep table structure, units and periods.

* Text: one passage per paragraph-like block (long blocks split at sentence ends), with
  character offsets into the extracted text.
* Tables: one passage per data row. Each carries its row label, the column headers (usually
  periods such as 2018 / 2017), the table's units ("in millions") and the nearest title
  above it, so retrieval can check metric and period together, e.g.
  ``Net sales | 2018: $ 32,765 | 2017: $ 31,657  [table: Consolidated Statement of Income; units: millions]``.
* Sections follow "Item N." headings (item_7, item_8, ...); passage ids are stable
  (``<doc>:<section>:p<n>`` / ``<doc>:<section>:t<table>r<row>``) and locations name the
  section, table and row.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from finsight.contracts import Passage, PassageLocation

PARSER_VERSION = "sec-html-1"
_BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "title"}
_SKIP = {"script", "style", "head"}
_VOID = {"br", "img", "hr", "meta", "link", "input", "col", "wbr", "area", "base", "source"}
_ITEM = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\-—]", re.I)
_UNITS = re.compile(r"\(?\s*(?:in\s+|dollars\s+in\s+|amounts\s+in\s+)?(thousands|millions|billions)\b", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MAX_TEXT = 1500


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


class _Collector(HTMLParser):
    """Turns HTML into an ordered stream of ('text', str) and ('table', rows) events."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, object]] = []
        self.buffer: list[str] = []
        self.skip_depth = 0
        self.hidden: list[str] = []           # open tags inside a display:none element
        self.tables: list[dict] = []          # stack for nested tables

    def _flush_text(self) -> None:
        text = _clean("".join(self.buffer))
        self.buffer = []
        if text:
            self.events.append(("text", text))

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in _SKIP or tag == "ix:header":
            self.skip_depth += 1
            return
        if tag not in _VOID and (self.hidden or "display:none" in (attributes.get("style") or "").replace(" ", "").lower()):
            self.hidden.append(tag)
        if self.skip_depth or self.hidden:
            return
        if tag == "table":
            if not self.tables:
                self._flush_text()
            self.tables.append({"rows": [], "row": None, "cell": None})
        elif tag == "tr" and self.tables:
            self.tables[-1]["row"] = []
        elif tag in ("td", "th") and self.tables:
            span = int(attributes.get("colspan") or 1) if str(attributes.get("colspan") or "1").isdigit() else 1
            self.tables[-1]["cell"] = {"text": [], "span": span}
        elif tag in _BLOCK and not self.tables:
            self._flush_text()

    def handle_endtag(self, tag):
        if tag in _SKIP or tag == "ix:header":     # skipped tags first, mirroring handle_starttag
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if self.skip_depth:
            return
        if tag in ("td", "th") and self.tables and self.tables[-1]["cell"] is not None:
            table = self.tables[-1]
            if table["row"] is not None:
                table["row"].append((_clean("".join(table["cell"]["text"])), table["cell"]["span"]))
            table["cell"] = None
        elif tag == "tr" and self.tables and self.tables[-1]["row"] is not None:
            table = self.tables[-1]
            if any(text for text, _ in table["row"]):
                table["rows"].append(table["row"])
            table["row"] = None
        elif tag == "table" and self.tables:
            table = self.tables.pop()
            if self.tables:   # nested table: fold its text into the enclosing cell
                text = " ".join(t for row in table["rows"] for t, _ in row)
                if self.tables[-1]["cell"] is not None:
                    self.tables[-1]["cell"]["text"].append(" " + text)
            else:
                self.events.append(("table", table["rows"]))
        elif tag in _BLOCK and not self.tables:
            self._flush_text()

    def handle_data(self, data):
        if self.skip_depth or self.hidden:
            return
        if self.tables and self.tables[-1]["cell"] is not None:
            self.tables[-1]["cell"]["text"].append(data)
        elif not self.tables:
            self.buffer.append(data)

    def close(self):
        super().close()
        self._flush_text()


def _expand(row: list[tuple[str, int]]) -> list[str]:
    cells: list[str] = []
    for text, span in row:
        cells.extend([text] + [""] * (span - 1))
    return cells


def _row_values(cells: list[str]) -> list[tuple[int, str]]:
    """Values with their original column index; '$', '(' attach forward and ')', '%' attach backward."""
    values: list[tuple[int, str]] = []
    pending = ""
    for index, cell in enumerate(cells[1:], start=1):
        text = cell.strip()
        if not text:
            continue
        if text in ("$", "(", "$("):
            pending = (pending + text).replace("$(", "$ (")
            continue
        if text in (")", "%", ")%") and values:
            col, value = values[-1]
            values[-1] = (col, value + text)
            continue
        joined = f"{pending} {text}".strip() if pending.endswith("$") else f"{pending}{text}"
        values.append((index, joined.replace("$ (", "$(")))
        pending = ""
    return values


def _is_numeric(cell: str) -> bool:
    return bool(re.search(r"\d", cell)) and not _YEAR.fullmatch(cell.strip()) and \
        bool(re.fullmatch(r"[$()\-\u2014\u2013\s\d.,%]+", cell.strip()))


def _forward_fill(row: list[str]) -> list[str]:
    filled, last = [], ""
    for index, cell in enumerate(row):
        if index == 0:
            filled.append(cell)
            continue
        last = cell or last
        filled.append(last)
    return filled


def _table_passages(rows: list[list[tuple[str, int]]], title: str, units: str | None) -> list[tuple[str, int]]:
    grid = [_expand(r) for r in rows]
    width = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (width - len(r)) for r in grid]
    first_data = next((i for i, r in enumerate(grid) if any(_is_numeric(c) for c in r[1:] if c not in ("$", "(", ")"))),
                      len(grid))
    headers = [_forward_fill(r) for r in grid[:first_data]]
    out = []
    for index, row in enumerate(grid[first_data:], start=first_data):
        values = _row_values(row)
        if not values:
            continue
        label = row[0] or "(unlabelled row)"
        parts = []
        for col, value in values:
            header = _clean(" ".join(dict.fromkeys(h[col] for h in headers if col < len(h) and h[col])))
            parts.append(f"{header}: {value}" if header else value)
        context = "; ".join(x for x in (f"table: {title}" if title else "", f"units: {units}" if units else "") if x)
        out.append((f"{label} | " + " | ".join(parts) + (f"  [{context}]" if context else ""), index))
    return out


def parse_filing(html: str, document_id: str, doc_short: str) -> list[Passage]:
    collector = _Collector()
    collector.feed(html)
    collector.close()
    passages: list[Passage] = []
    section = "front"
    offset = 0
    recent_text: list[str] = []
    counters: dict[str, int] = {}
    table_index = 0

    def add(text: str, kind: str, location: PassageLocation, key: str) -> None:
        nonlocal offset
        passages.append(Passage(id=f"{doc_short}:{section}:{key}", document_id=document_id, section=section,
                                location=location, start=offset, end=offset + len(text), text=text))
        offset += len(text) + 1

    for kind, payload in collector.events:
        if kind == "text":
            text = str(payload)
            match = _ITEM.match(text)
            if match and len(text) < 200:
                section = f"item_{match.group(1).lower()}"
            recent_text = (recent_text + [text])[-4:]
            for chunk in _split(text):
                counters[section] = counters.get(section, 0) + 1
                add(chunk, "text", PassageLocation(html_section=section), f"p{counters[section]}")
        else:
            rows = payload
            title = next((t for t in reversed(recent_text) if len(t) < 160 and not _UNITS.search(t)
                          and not _ITEM.match(t)), "")
            unit_text = " ".join(recent_text[-2:] + [c for r in rows[:3] for c, _ in r])
            unit_match = _UNITS.search(unit_text)
            for text, row_index in _table_passages(rows, title, unit_match.group(1).lower() if unit_match else None):
                add(text, "table", PassageLocation(html_section=section, table=str(table_index), row=row_index),
                    f"t{table_index}r{row_index}")
            table_index += 1
    return passages


def _split(text: str) -> list[str]:
    if len(text) <= _MAX_TEXT:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if current and len(current) + len(sentence) + 1 > _MAX_TEXT:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts
