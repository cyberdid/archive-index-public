"""UA-PAR query set: format, quotas and validation (plan Task 5, spec section 4)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from archive_index.lexical import query_expression, tokenize

CATEGORY_QUOTA = {"known_item": 40, "topical": 60, "technical": 40, "cross_lingual": 30, "no_answer": 30}
MIN_UK_SHARE = 0.40
MIN_RU_SHARE = 0.20
MAX_SHARED_RUN = 5  # tokens; a longer run shared with the target document means the query was copied
COPY_CHECKED = ("topical", "cross_lingual")


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    lang: str
    category: str
    author: str
    target: str = ""  # doc_id the query was written for (empty for no-answer and open topical queries)


def load_queries(path: Path) -> list[Query]:
    with path.open(encoding="utf-8") as fh:
        rows = csv.DictReader(fh, delimiter="\t")
        return [Query(r["qid"], r["text"], r["lang"], r["category"], r["author"], r.get("target") or "") for r in rows]


def write_queries(path: Path, queries: list[Query]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["qid", "text", "lang", "category", "author", "target"])
        for q in queries:
            writer.writerow([q.qid, q.text, q.lang, q.category, q.author, q.target])


def longest_shared_run(query: str, doc_text: str) -> int:
    """Length in tokens of the longest contiguous token sequence the query and the document share."""
    a, b = tokenize(query), tokenize(doc_text)
    if not a or not b:
        return 0
    best = 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, start=1):
            if token == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def validate(queries: list[Query], manifest: list[dict], store=None, texts: dict[str, str] | None = None) -> list[str]:
    """Every problem with the query set, as readable lines; an empty list means the set is ready."""
    problems: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if q.qid in seen:
            problems.append(f"{q.qid}: duplicate qid")
        seen.add(q.qid)
        if not q.text.strip():
            problems.append(f"{q.qid}: empty text")
        if q.category not in CATEGORY_QUOTA:
            problems.append(f"{q.qid}: unknown category {q.category}")
    for category, quota in CATEGORY_QUOTA.items():
        n = sum(1 for q in queries if q.category == category)
        if n < quota:
            problems.append(f"category {category}: {n} queries, quota {quota}")
    if queries:
        uk = sum(1 for q in queries if q.lang == "uk") / len(queries)
        ru = sum(1 for q in queries if q.lang == "ru") / len(queries)
        if uk < MIN_UK_SHARE:
            problems.append(f"uk share {uk:.2f} below {MIN_UK_SHARE}")
        if ru < MIN_RU_SHARE:
            problems.append(f"ru share {ru:.2f} below {MIN_RU_SHARE}")
    known = {e["doc_id"] for e in manifest}
    for q in queries:
        if q.target and q.target not in known:
            problems.append(f"{q.qid}: target {q.target} is not in the manifest")
        if q.category in COPY_CHECKED and q.target and texts and q.target in texts:
            run = longest_shared_run(q.text, texts[q.target])
            if run > MAX_SHARED_RUN:
                problems.append(f"{q.qid}: copied from {q.target} ({run} shared tokens)")
    if store is not None:
        for q in queries:
            if q.category == "no_answer":
                n = store.count_lexical(query_expression(q.text), collections=None, project=None)
                if n:
                    problems.append(f"{q.qid}: no_answer query «{q.text}» matches {n} fragments")
    return problems
