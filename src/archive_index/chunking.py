"""Split documents into retrieval-sized chunks without cutting paragraphs in half."""

from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def pack_paragraphs(text: str, max_chars: int = 1200) -> list[str]:
    """Greedily pack blank-line separated paragraphs into chunks of at most max_chars.

    A paragraph longer than max_chars is hard-split, so no chunk exceeds the limit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)
            )
            continue
        if current and len(current) + 2 + len(paragraph) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs; text before the first heading gets heading ''."""
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def chunk_markdown(text: str, max_chars: int = 1200) -> list[tuple[str, str]]:
    """Chunk markdown by heading first, then pack each section. Returns (heading, chunk)."""
    out: list[tuple[str, str]] = []
    for heading, body in split_markdown_sections(text):
        out.extend((heading, piece) for piece in pack_paragraphs(body, max_chars))
    return out
