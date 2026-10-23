"""Prepare/evaluate the 50-query non-note extension, kept separate from the frozen 200 queries.

The extension's primary labels are initial Gemini, with initial Gemma 12B sensitivity;
GPT is comparison-only. This is exploratory target-coverage evidence, not an identified
experiment on query-author self-preference (the query populations also differ).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from paper.eval.metrics import read_run
from paper.eval.report_main import build_report
from paper.judge.finalize import sha256, write_json
from paper.judge.pool import build_pool, read_sheet, read_trec_run
from paper.judge.report import judge_report
from paper.judge.sheets import make_sheets
from paper.queries.extend import COUNTS, accept
from paper.queries.schema import load_queries
from paper.runs.systems import SYSTEMS

DATA = Path("paper/data")
QUERIES = DATA / "queries-ext.tsv"
RUNS = DATA / "runs-extension"
SHEETS = DATA / "judgments-extension"
PREPARED = SHEETS / "prepared.json"
JUDGES = ("gemini", "gemma12", "gpt")


def inputs() -> tuple[list, dict, dict]:
    queries = load_queries(QUERIES)
    manifest_path = DATA / "ua-par/manifest.jsonl"
    manifest = {r["doc_id"]: r for r in map(json.loads, manifest_path.read_text().splitlines())}
    if len(queries) != 50 or len({q.qid for q in queries}) != 50:
        raise ValueError("extension requires 50 unique queries")
    if {q.qid for q in queries} != {f"q{i:03d}" for i in range(201, 251)}:
        raise ValueError("extension query IDs must stay separate from the frozen population")
    parts = Counter(manifest[q.target]["part"] for q in queries)
    if dict(parts) != COUNTS or any(q.author != "qwen3.8-27b" or q.lang not in ("uk", "ru") for q in queries):
        raise ValueError("extension authorship/language/target quotas differ from the protocol")
    files = {p.stem: p for p in RUNS.glob("*.trec")}
    if set(files) != set(SYSTEMS):
        raise ValueError("extension requires the ten system runs")
    qids = {q.qid for q in queries}
    for name, path in files.items():
        run = read_run(path)
        # A paraphrased answerable query may legitimately have no lexical matches.
        missing_dense = SYSTEMS[name].mode != "lexical" and set(run) != qids
        if set(run) - qids or missing_dense or any(doc not in manifest for docs in run.values() for doc in docs):
            raise ValueError(f"foreign/missing query or document in {path}")
    digests = {str(p): sha256(p) for p in (QUERIES, DATA / "queries-ext.log.jsonl", manifest_path,
                                         *sorted(files.values()))}
    return queries, manifest, digests


def prepare() -> dict:
    queries, manifest, digests = inputs()
    if PREPARED.exists():
        raise FileExistsError("extension sheets already prepared; never overwrite them")
    result = make_sheets(RUNS, QUERIES, DATA / "indexes/full.sqlite3", DATA / "ua-par", SHEETS,
                         names=[f"{name}.tsv" for name in JUDGES])
    # Targets absent from the pool are still checked against their indexed lead.
    from paper.judge.sheets import snippets_from_index
    leads = snippets_from_index(DATA / "indexes/full.sqlite3", DATA / "ua-par")
    if any(not accept(q.text, leads[q.target]) for q in queries):
        raise ValueError("extension query violates word-count or copying rules")
    result.update(source_sha256=digests, primary="gemini_initial", sensitivity="gemma12_initial",
                  gpt="comparison-only", target_parts=dict(Counter(manifest[q.target]["part"] for q in queries)),
                  distinct_query_wordings=len({" ".join(q.text.casefold().split()) for q in queries}),
                  languages=dict(Counter(q.lang for q in queries)),
                  repository_target_suffixes=dict(Counter(Path(q.target).suffix.lower() for q in queries
                                                         if manifest[q.target]["part"] == "code")),
                  sheet_prepared_sha256={name: sha256(SHEETS / f"{name}.tsv") for name in JUDGES})
    write_json(PREPARED, result)
    return result


def evaluate() -> dict:
    import hashlib
    import io

    queries, manifest, digests = inputs()
    prepared = json.loads(PREPARED.read_text())
    if prepared["source_sha256"] != digests:
        raise ValueError("extension inputs changed after preparation")
    pool = build_pool({p.stem: read_trec_run(p) for p in RUNS.glob("*.trec")}, depth=10)
    expected = {(q, doc) for q, docs in pool.items() for doc in docs}
    judges, missing_ancillary = {}, []
    for name in JUDGES:
        path = SHEETS / f"{name}.tsv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            header, rows = reader.fieldnames, list(reader)
        keys = [(r["qid"], r["doc_id"]) for r in rows]
        grades = read_sheet(path)
        if len(keys) != len(expected) or set(keys) != expected or set(grades) - expected:
            raise ValueError(f"{name}: duplicate or foreign extension grades")
        if any(type(g) is not int or g not in (0, 1, 2) for g in grades.values()):
            raise ValueError(f"{name}: invalid extension grade")
        blank = io.StringIO(newline="")
        writer = csv.DictWriter(blank, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({**r, "grade": ""} for r in rows)
        if hashlib.sha256(blank.getvalue().encode()).hexdigest() != prepared["sheet_prepared_sha256"][name]:
            raise ValueError(f"{name}: extension query/snippet/order changed")
        if set(grades) != expected:
            if name == "gpt":
                missing_ancillary.append(name)
                continue
            raise ValueError(f"{name}: incomplete extension grades")
        judges[name] = grades
    runs = {p.stem: read_run(p) for p in RUNS.glob("*.trec")}
    report = {"protocol": __doc__.strip(), "provenance": prepared, "queries": len(queries),
              "judge_diagnostics": judge_report(judges, queries), "variants": {},
              "missing_ancillary_assessments": missing_ancillary}
    # Two records can share a wording while targeting different files. Retain
    # the frozen 50-record analysis, and expose a deterministic 49-wording
    # sensitivity rather than silently assuming all records are independent.
    seen, unique_queries, dropped = set(), [], []
    for query in sorted(queries, key=lambda q: q.qid):
        wording = ' '.join(query.text.casefold().split())
        if wording in seen:
            dropped.append(query.qid)
        else:
            seen.add(wording)
            unique_queries.append(query)
    report['unique_wording_sensitivity'] = {
        'selection': 'Keep the lowest query ID for each casefolded whitespace-normalized wording; do not select by score.',
        'queries': len(unique_queries), 'dropped_query_ids': dropped, 'variants': {},
    }
    for name in ("gemini", "gemma12"):
        qrels = {q.qid: {doc: grade for (qid, doc), grade in judges[name].items() if qid == q.qid} for q in queries}
        report["variants"][f"{name}_initial"] = build_report(queries, qrels, runs)
        report['unique_wording_sensitivity']['variants'][f'{name}_initial'] = build_report(unique_queries, qrels, runs)
    report["sheet_sha256"] = {name: sha256(SHEETS / f"{name}.tsv") for name in JUDGES}
    write_json(Path("paper/results/extension.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else evaluate()
    print(json.dumps({k: v for k, v in result.items() if k not in ("variants", "judge_diagnostics")}))
