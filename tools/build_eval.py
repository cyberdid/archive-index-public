"""Build the evaluation set from the real index (plan G1).

Targets (plan G1): 60-80 queries; 6 exact queries for each of 11 formats (wiki, pdf, docx, pptx, xlsx,
csv, html, sessions, git, audio, images); 8 absent queries; 6 descriptive queries from tools/final_holdout.py.
Each exact query records its source (file path and locator) and its language (en, uk, ru, or cyr when the
script alone cannot tell Ukrainian from Russian). Languages are balanced inside each format where the index
allows it; the report prints what was actually reached.

- exact: literal phrases taken from indexed fragments. The phrase is in the source by construction, so the
  label is exact. It is NOT checked by a person: a human review of samples is still needed.
- absent: random letter strings that are in no document (must return no word matches).
- described: descriptions with known projects from tools/final_holdout.py.
Writes evals/queries.jsonl (default), one JSON object per line.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

from tools.final_holdout import DESCRIBED

FORMATS = ("wiki", "pdf", "docx", "pptx", "xlsx", "csv", "html", "sessions", "git", "audio", "images")
OWN_COLLECTIONS = {"wiki": "wiki", "sessions": "sessions", "git": "git", "audio": "audio", "images": "images"}
SUFFIX_FORMAT = {"pdf": "pdf", "docx": "docx", "pptx": "pptx", "xlsx": "xlsx", "csv": "csv",
                 "html": "html", "htm": "html"}
PER_FORMAT = 6
ABSENT = 8
DESCRIBED_COUNT = 6


def fmt(path: str, collection: str) -> str | None:
    """Format label for the G1 target list, or None when the file is not one of the target formats."""
    if collection in OWN_COLLECTIONS:
        return OWN_COLLECTIONS[collection]
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    return SUFFIX_FORMAT.get(suffix)


def language(text: str) -> str:
    """en / uk / ru by script and letters; 'cyr' when Cyrillic has no telling letters."""
    lowered = text.lower()
    cyrillic = sum("Ѐ" <= ch <= "ӿ" for ch in lowered)
    latin = sum("a" <= ch <= "z" for ch in lowered)
    if cyrillic == 0:
        return "en" if latin else "other"
    if latin > cyrillic:
        return "en"
    if any(ch in "ыэъё" for ch in lowered):
        return "ru"
    if any(ch in "іїєґ" for ch in lowered):
        return "uk"
    return "cyr"


def pick_balanced(rng: random.Random, pool: list, count: int) -> list:
    """Round-robin over languages uk, ru, en so each format mixes languages when the index allows."""
    buckets: dict[str, list] = {}
    for item in pool:
        buckets.setdefault(item[-1], []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen: list = []
    order = ["uk", "ru", "en", "cyr", "other"]
    while len(chosen) < count and any(buckets.get(lang) for lang in order):
        for lang in order:
            if len(chosen) >= count:
                break
            if buckets.get(lang):
                chosen.append(buckets[lang].pop())
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".local/share/archive-index/archive.sqlite3")
    parser.add_argument("--out", type=Path, default=Path("evals/queries.jsonl"))
    parser.add_argument("--per-format", type=int, default=PER_FORMAT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT c.id, c.collection, d.path, c.locator, c.body FROM chunks c JOIN documents d ON d.id = c.doc_id"
        " WHERE c.dimension > 0 OR c.collection IN ('tables','code','docs','wiki','git','sessions','run_logs')"
    ).fetchall()
    word = re.compile(r"[^\W_]{2,}")
    by_format: dict[str, list] = {}
    for cid, collection, path, locator, body in rows:
        label = fmt(path, collection)
        if label is None:
            continue
        spans = list(word.finditer(body))
        if len(spans) < 8 or body[spans[0].start():spans[5].end()].strip() == "":
            continue
        by_format.setdefault(label, []).append((cid, path, locator, body, spans, language(body)))
    out: list[dict] = []
    reached: dict[str, Counter] = {}
    for label in FORMATS:
        pool = by_format.get(label, [])
        chosen = pick_balanced(rng, pool, args.per_format)
        reached[label] = Counter(item[-1] for item in chosen)
        for cid, path, locator, body, spans, lang in chosen:
            start = rng.randrange(0, max(1, len(spans) - 6))
            # The expected phrase is a literal slice of the stored text; the query is its words.
            literal = body[spans[start].start():spans[start + 5].end()]
            query = " ".join(m.group(0) for m in spans[start:start + 6])
            out.append({"kind": "exact", "format": label, "lang": lang, "query": query,
                        "expect_phrase": literal, "source_path": path, "source_locator": locator})
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(ABSENT):
        out.append({"kind": "absent", "format": "-", "lang": "-",
                    "query": "".join(rng.choice(alphabet) for _ in range(12))})
    for query, project in DESCRIBED[:DESCRIBED_COUNT]:
        out.append({"kind": "described", "format": "-", "lang": "-", "query": query, "expect_project": project})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    langs = Counter(x["lang"] for x in out if x["kind"] == "exact")
    print(json.dumps({"total": len(out), "by_kind": {
        k: sum(1 for x in out if x["kind"] == k) for k in ("exact", "absent", "described")},
        "exact_by_lang": dict(langs),
        "missing_formats": [f for f in FORMATS if not reached.get(f) or sum(reached[f].values()) == 0],
        "per_format": {f: sum(reached[f].values()) for f in FORMATS if f in reached}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
