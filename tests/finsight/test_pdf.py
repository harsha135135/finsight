from finsight.ingestion.pdf import parse_pdf


def _pdf(pages: list[list[str]]) -> bytes:
    """A minimal valid PDF with Helvetica text lines (an empty list makes a page without text)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        ops = "".join(f"BT /F1 10 Tf 50 {750 - 14 * i} Td ({line}) Tj ET\n" for i, line in enumerate(lines))
        objects.append(f"<< /Length {len(ops)} >>\nstream\n{ops}endstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objects)} 0 R "
                       f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pages_become_page_anchored_passages_and_textless_pages_are_counted(tmp_path):
    path = tmp_path / "f.pdf"
    path.write_bytes(_pdf([["Consolidated Statement of Income", "Net sales 32,765 31,657"], [],
                           ["Purchases of property, plant and equipment (1,577)"]]))
    parsed = parse_pdf(path, "fb:3M_2018_10K", "mmm18")
    assert parsed.pages == 3 and parsed.empty_pages == [2]
    first, last = parsed.passages[0], parsed.passages[-1]
    assert first.location.page == 1 and "Net sales 32,765 31,657" in first.text and first.id == "mmm18:p1:c0"
    assert last.location.page == 3 and "(1,577)" in last.text
