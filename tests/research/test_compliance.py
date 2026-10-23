import pytest

from paper.release.compliance import (check_anonymity, check_fonts, check_log, check_page_size, check_pdf,
                                      check_text)

FONTS_OK = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
IHAIZV+CMUSerif-Roman-Identity-H     CID Type 0C       Identity-H       yes yes yes      4  0
"""
FONTS_BAD = FONTS_OK + """Helvetica                            Type 1            Custom           no  no  no       9  0
[none]                               Type 3            Custom           yes no  no      11  0
"""


def test_pending_markers_and_unresolved_references_are_rejected():
    assert check_text("Results are in Table 2 and Section 4.") == []
    problems = check_text("see [PENDING: abstract] and Table ?? and [?]")
    assert any("PENDING" in p for p in problems) and any("??" in p for p in problems) and any("[?]" in p for p in problems)


def test_fonts_must_be_embedded_and_not_type3():
    assert check_fonts(FONTS_OK) == []
    problems = check_fonts(FONTS_BAD)
    assert any("not embedded: Helvetica" in p for p in problems) and any("Type 3" in p for p in problems)


def test_log_warnings_for_references_citations_and_overfull_boxes():
    assert check_log("Output written on main.pdf (3 pages).") == []
    log = ("LaTeX Warning: Reference `sec:x' on page 2 undefined on input line 4.\n"
           "LaTeX Warning: Citation `smith2020' on page 3 undefined on input line 9.\n"
           "Overfull \\hbox (12.5pt too wide) in paragraph at lines 10--12\n"
           "Overfull \\hbox (0.4pt too wide) in paragraph at lines 20--21\n")
    problems = check_log(log)
    assert any("sec:x" in p for p in problems) and any("smith2020" in p for p in problems)
    assert sum("Overfull" in p for p in problems) == 1  # under 1pt is invisible and tolerated


def test_page_size_is_a4():
    assert check_page_size("Page size:       595.28 x 841.89 pts (A4)\n") == []
    assert check_page_size("Page size:       612 x 792 pts (letter)\n") != []


def test_anonymity_only_for_a_venue_edition():
    deny = ["Pylypenko", "0009-0004-9610-9242", "cyberdid"]
    assert check_anonymity("Anonymous authors. Code: https://anonymous.4open.science/r/x", {"Author": ""}, deny) == []
    problems = check_anonymity("See github.com/cyberdid/archive-index", {"Author": "Oleksandr Pylypenko"}, deny)
    assert any("cyberdid" in p for p in problems) and any("metadata" in p for p in problems)


@pytest.mark.parametrize(("text", "pending"), [("Finished text", False), ("[PENDING: abstract]", True)])
def test_pdf_placeholder_gate_uses_a_fixed_fixture(tmp_path, text, pending):
    # Exercise real Poppler extraction independently of the evolving manuscript.
    content = f"BT /F1 12 Tf 50 780 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(data)
    problems = check_pdf(pdf, log=None, edition="preprint")
    assert any("PENDING" in problem for problem in problems) is pending
