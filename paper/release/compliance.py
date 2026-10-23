"""Formatting gate for every built PDF of the paper (plan Task 13, revision b).

A preprint edition must have no placeholders, no unresolved references or citations, every font embedded and none of
Type 3, A4 pages, and no visibly overfull lines. A venue edition must also be anonymous: no author name, ORCID,
e-mail or personal repository in the text or the PDF metadata.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

OVERFULL_TOLERANCE_PT = 1.0
A4_PTS = (595.28, 841.89)


def check_text(text: str) -> list[str]:
    problems = [f"placeholder left: {m.group(0)[:60]}" for m in re.finditer(r"\[PENDING:[^\]]*\]?", text)]
    if re.search(r"(?<!\?)\?\?(?!\?)", text):
        problems.append("unresolved reference '??' in the text")
    if "[?]" in text:
        problems.append("unresolved citation '[?]' in the text")
    return problems


def check_fonts(pdffonts_output: str) -> list[str]:
    problems = []
    for line in pdffonts_output.splitlines()[2:]:
        if not line.strip():
            continue
        name = line.split()[0]
        if "Type 3" in line:
            problems.append(f"Type 3 font: {name}")
        # columns from the right: object ID (2 fields), uni, sub, emb
        fields = line.split()
        if fields[-5] != "yes":
            problems.append(f"font not embedded: {name}")
    return problems


def check_log(log: str) -> list[str]:
    problems = [f"undefined reference: {m.group(1)}" for m in re.finditer(r"Reference `([^']+)' on page \d+ undefined", log)]
    problems += [f"undefined citation: {m.group(1)}" for m in re.finditer(r"Citation `([^']+)' on page \d+ undefined", log)]
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*lines (\d+)--(\d+)", log):
        if float(m.group(1)) >= OVERFULL_TOLERANCE_PT:
            problems.append(f"Overfull line {m.group(1)}pt at source lines {m.group(2)}--{m.group(3)}")
    return problems


def check_page_size(pdfinfo_output: str) -> list[str]:
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", pdfinfo_output)
    if not m:
        return ["page size not found"]
    w, h = float(m.group(1)), float(m.group(2))
    return [] if abs(w - A4_PTS[0]) < 1 and abs(h - A4_PTS[1]) < 1 else [f"page size {w} x {h} pts is not A4"]


def check_anonymity(text: str, metadata: dict, deny: list[str]) -> list[str]:
    problems = [f"identifying string in text: {d}" for d in deny if d.lower() in text.lower()]
    for key, value in metadata.items():
        problems += [f"identifying string in PDF metadata {key}: {d}" for d in deny if d.lower() in str(value).lower()]
    return problems


def _run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def check_pdf(pdf: Path, log: Path | None, edition: str = "preprint", deny: list[str] | None = None) -> list[str]:
    text = _run("pdftotext", "-layout", str(pdf), "-")
    info = _run("pdfinfo", str(pdf))
    problems = check_text(text) + check_fonts(_run("pdffonts", str(pdf))) + check_page_size(info)
    if log is not None and log.exists():
        problems += check_log(log.read_text(encoding="utf-8", errors="replace"))
    if edition == "venue":
        metadata = dict(line.split(":", 1) for line in info.splitlines() if ":" in line)
        problems += check_anonymity(text, metadata, deny or [])
    return problems


def main() -> int:
    import sys

    manuscript = Path("paper/manuscript")
    edition = sys.argv[1] if len(sys.argv) > 1 else "preprint"
    failed = 0
    for lang in ("en", "uk"):
        problems = check_pdf(manuscript / f"article-{lang}.pdf", manuscript / "tex" / lang / "main.log", edition)
        for p in problems:
            print(f"{lang}: {p}")
        failed += len(problems)
    print(f"{failed} problems")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
