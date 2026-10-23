"""Judging sheets from the runs, and an LLM judge that fills a sheet's grade column (plan Task 7, step 5).

A judge sees the query and the beginning of the document: its title and its first chunks in order, up to
SNIPPET_CHARS (the lead of a note, the header of a file or table). The text is the same for every system, so the sheet
does not depend on which system retrieved the document.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from urllib.parse import unquote
from pathlib import Path

from paper.corpus.build import read_manifest
from paper.judge.pool import SNIPPET_CHARS, build_pool, export_sheet, read_trec_run
from paper.queries.schema import load_queries

# Current protocol: Gemini primary, Gemma 4 12B initial sensitivity, and Gemini within-model disagreement review.
# Gemma 4 E4B is ancillary; GPT-6-Luna wrote the queries and is comparison only. The historical primary.tsv/claude
# label remains available to preserve old sheets; it is not part of the current qrels workflow.
ANNOTATORS = {"primary.tsv": "claude", "gemini.tsv": "gemini", "gemma.tsv": "gemma4", "gemma12.tsv": "gemma4-12b",
              "gpt.tsv": "gpt-6-luna"}
SAVE_EVERY = 25  # grades between rewrites of the sheet; an interruption loses at most this many


_LINK = re.compile(r"https?://\S+")


def readable(text: str) -> str:
    """Percent-encoded links (wiki source lines) decoded, so they do not eat the judge's character budget."""
    return _LINK.sub(lambda m: unquote(m.group(0)), text)


def snippets_from_index(db: Path, corpus_root: Path) -> dict[str, str]:
    """doc_id (path relative to the corpus root) -> 'title\nfirst chunks', at least SNIPPET_CHARS when the document
    is that long; a heading-only first chunk is followed by the text under it."""
    root = corpus_root.resolve()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT d.path, d.title, c.body FROM documents d JOIN chunks c ON c.doc_id = d.id"
            " ORDER BY d.id, c.ord")
        snippets: dict[str, str] = {}
        for path, title, body in rows:
            try:
                doc = Path(path).resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            text = snippets.get(doc)
            if text is None:
                snippets[doc] = f"{title}\n{readable(body)}"
            elif len(text) < SNIPPET_CHARS:
                snippets[doc] = f"{text}\n{readable(body)}"
    finally:
        conn.close()
    return snippets


def add_transcripts(snippets: dict[str, str], manifest: list[dict]) -> dict[str, str]:
    """A text judge cannot hear a recording: speech documents get their human FLEURS transcript appended."""
    out = dict(snippets)
    for entry in manifest:
        if entry.get("part") == "speech" and entry.get("transcript") and entry["doc_id"] in out:
            out[entry["doc_id"]] += f"\nСказано в записі: {entry['transcript']}"
    return out


def refuse_graded(path: Path) -> None:
    """A sheet that already holds grades is never rewritten: judging hours would be lost."""
    if path.exists() and any(row["grade"].strip() for row in _read(path)[1]):
        raise FileExistsError(f"{path} already holds grades; move it away to rebuild the sheet")


def make_sheets(run_dir: Path, queries_path: Path, db: Path, corpus_root: Path, out_dir: Path,
                depth: int = 10, names: list[str] | None = None) -> dict[str, int]:
    """Pool the top-depth documents of every run and write one shuffled sheet per judge (names: some of ANNOTATORS)."""
    runs = {path.stem: read_trec_run(path) for path in sorted(run_dir.glob("*.trec"))}
    pool = build_pool(runs, depth=depth)
    queries = {q.qid: q.text for q in load_queries(queries_path)}
    snippets = add_transcripts(snippets_from_index(db, corpus_root),
                               read_manifest(corpus_root / "manifest.jsonl"))
    missing = sorted({doc for docs in pool.values() for doc in docs} - set(snippets))
    if missing:
        raise ValueError(f"{len(missing)} pooled documents have no text in {db}: {missing[:3]}")
    names = names or list(ANNOTATORS)
    for name in names:
        refuse_graded(out_dir / name)
    for name in names:
        export_sheet(pool, queries, snippets, out_dir / name, ANNOTATORS[name])
    return {"systems": len(runs), "queries": len(pool), "pairs": sum(len(docs) for docs in pool.values())}


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _write(path: Path, header: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def judge_sheet(path: Path, judge, save_every: int = SAVE_EVERY) -> dict[str, int]:
    """Fill empty grades with judge(query, snippet) -> 0|1|2|None; resumes where an interrupted run stopped.

    An answer that gives no grade stays empty and is counted, never guessed.
    """
    header, rows = _read(path)
    report = {"graded": 0, "unparsed": 0, "already": 0}
    pending = 0
    try:
        for row in rows:
            if row["grade"].strip():
                report["already"] += 1
                continue
            grade = judge(row["query"], row["snippet"])
            if grade is None:
                report["unparsed"] += 1
                continue
            row["grade"] = str(grade)
            report["graded"] += 1
            pending += 1
            if pending >= save_every:
                _write(path, header, rows)
                pending = 0
    finally:
        if pending:
            _write(path, header, rows)
    return report


def judge_sheet_batched(path: Path, judge_batch, log_path: Path, batch_size: int = 50) -> dict[str, int]:
    """Fill empty grades batch by batch with judge_batch([(id, query, snippet)]) -> (grades by id, call record).

    The sheet is saved after every batch, so a stopped run resumes at the first ungraded row; every call is appended
    to log_path (JSONL) with the (qid, doc_id) pairs it graded. Pairs a reply leaves out stay empty and are counted.
    """
    header, rows = _read(path)
    todo = [i for i, row in enumerate(rows) if not row["grade"].strip()]
    report = {"graded": 0, "unparsed": 0, "already": len(rows) - len(todo), "calls": 0}
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        grades, record = judge_batch([(str(i), rows[i]["query"], rows[i]["snippet"]) for i in batch])
        for i in batch:
            if str(i) in grades:
                rows[i]["grade"] = str(grades[str(i)])
                report["graded"] += 1
            else:
                report["unparsed"] += 1
        record = {**record, "pairs": [[rows[i]["qid"], rows[i]["doc_id"]] for i in batch],
                  "grades_by_pair": [{"qid": rows[i]["qid"], "doc_id": rows[i]["doc_id"],
                                      "grade": grades[str(i)]} for i in batch if str(i) in grades],
                  "graded": sum(1 for i in batch if str(i) in grades)}
        # Log the received result first. A failed log must never leave saved, unlogged grades;
        # a failed sheet save leaves the full result available in the log for audit/recovery.
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        _write(path, header, rows)
        report["calls"] += 1
        if not grades and record.get("status") not in (None, "SUCCESS"):
            # a spent quota or a broken CLI fails every call alike; resume later from the first ungraded row
            raise RuntimeError(f"judge call failed ({record.get('status')}): {record.get('stderr', '')[-300:]}")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="pool the runs and write the judging sheets")
    make.add_argument("--runs", type=Path, default=Path("paper/data/runs"))
    make.add_argument("--queries", type=Path, default=Path("paper/data/queries.tsv"))
    make.add_argument("--index", type=Path, default=Path("paper/data/indexes/full.sqlite3"))
    make.add_argument("--corpus", type=Path, default=Path("paper/data/ua-par"))
    make.add_argument("--out", type=Path, default=Path("paper/data/judgments"))
    judge = sub.add_parser("judge", help="fill a sheet: gemma (E4B, llama-server), gemma12 (12B, Ollama), "
                                          "gemini (agy) or gpt (codex exec)")
    judge.add_argument("--judge", choices=("gemma", "gemma12", "gemini", "gpt"), required=True)
    judge.add_argument("--sheet", type=Path, required=True)
    judge.add_argument("--url", default="http://127.0.0.1:8282")
    judge.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()
    if args.command == "make":
        print(json.dumps(make_sheets(args.runs, args.queries, args.index, args.corpus, args.out)))
    elif args.judge == "gemma":
        from paper.judge.llm_judge import judge_with_llama

        print(json.dumps(judge_sheet(args.sheet, lambda q, s: judge_with_llama(q, s, url=args.url))))
    elif args.judge == "gemma12":
        from paper.judge.llm_judge import judge_with_ollama

        print(json.dumps(judge_sheet(args.sheet, judge_with_ollama)))
    else:
        from paper.judge.llm_judge import judge_batch_with_agy, judge_batch_with_codex

        batch_judge = judge_batch_with_agy if args.judge == "gemini" else judge_batch_with_codex
        log = args.sheet.with_suffix(".calls.jsonl")
        print(json.dumps(judge_sheet_batched(args.sheet, batch_judge, log, batch_size=args.batch)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
