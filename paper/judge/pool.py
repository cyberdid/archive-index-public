"""Depth-k pooling of all system runs, judging sheets, and TREC qrels (plan Task 7)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

SNIPPET_CHARS = 1200  # text a judge sees per document


def read_trec_run(path: Path) -> list[tuple[str, list[tuple[str, float]]]]:
    rows: dict[str, list[tuple[str, float]]] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        qid, _q0, doc, _rank, score, _run = line.split()
        if qid not in rows:
            rows[qid] = []
            order.append(qid)
        rows[qid].append((doc, float(score)))
    return [(qid, rows[qid]) for qid in order]


def build_pool(runs: dict[str, list], depth: int = 10) -> dict[str, list[str]]:
    """Union of every system's top-depth documents per query (sorted, so the pool does not reveal its origin)."""
    pool: dict[str, set[str]] = {}
    for rows in runs.values():
        for qid, run in rows:
            pool.setdefault(qid, set()).update(doc for doc, _ in run[:depth])
    return {qid: sorted(docs) for qid, docs in sorted(pool.items())}


def export_sheet(pool: dict[str, list[str]], queries: dict[str, str], snippets: dict[str, str], path: Path,
                 annotator: str, seed: int = 2026) -> None:
    """One row per (query, document), shuffled per annotator; no system names, empty grade column."""
    rows = [(qid, queries[qid], doc, " ".join(snippets.get(doc, "").split())[:SNIPPET_CHARS]) for qid, docs in pool.items()
            for doc in docs]
    random.Random(f"{seed}-{annotator}").shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["qid", "query", "doc_id", "snippet", "grade"])
        for row in rows:
            writer.writerow([*row, ""])


def read_sheet(path: Path) -> dict[tuple[str, str], int]:
    grades = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["grade"].strip() != "":
                grades[(row["qid"], row["doc_id"])] = int(row["grade"])
    return grades


def write_qrels(path: Path, grades: dict[tuple[str, str], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for (qid, doc), grade in sorted(grades.items()):
            fh.write(f"{qid} 0 {doc} {grade}\n")
