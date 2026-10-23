"""Blind cross-family adjudication of the initial Gemini/Gemma 12B disagreements (audit of 2026-10-22).

The within-model Gemini review is not independent. Here Claude (another model family, and not the author of the
queries) grades every disagreement pair blind: same rubric, same query and snippet, no initial grades shown. The
median of the three grades (Gemini, Gemma 12B, Claude) replaces each disagreement; agreed pairs keep their grade.
The result is one more labelled qrels variant ('median3'); it does not replace the protocol's primary qrels.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import statistics
from pathlib import Path

from paper.judge.agreement import disagreements
from paper.judge.finalize import DEFAULT_SNAPSHOT, load_snapshot, sha256, write_json
from paper.judge.sheets import _read, judge_sheet_batched

CLAUDE_MODEL = "claude-opus-5-5-medium"
BLIND_SEED = 2027  # differs from the review's shuffle, so batch neighbours differ too
DEFAULT_SHEET = Path("paper/data/judgments/claude-blind.tsv")
COLUMNS = ["qid", "query", "doc_id", "snippet", "grade"]


def prepare_blind(data, path: Path) -> dict:
    """Exclusive creation of the blind sheet: disagreement pairs only, original query and snippet, no grades."""
    keys = disagreements(data.grades("gemini"), data.grades("gemma12"))
    random.Random(BLIND_SEED).shuffle(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for key in keys:
            row = data.sheets["gemini"][key]
            writer.writerow({"qid": row["qid"], "query": row["query"], "doc_id": row["doc_id"],
                             "snippet": row["snippet"], "grade": ""})
    record = {"kind": "blind cross-family adjudication", "model": CLAUDE_MODEL, "disagreements": len(keys),
              "seed": BLIND_SEED, "prepared_sha256": sha256(path), "provenance": data.provenance}
    write_json(path.with_suffix(".prepared.json"), record)
    return record


def _third(data, path: Path, allow_empty: bool = False) -> dict:
    header, rows = _read(path)
    expected = set(disagreements(data.grades("gemini"), data.grades("gemma12")))
    keys = [(r["qid"], r["doc_id"]) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("blind sheet must hold exactly the initial disagreements")
    valid = ("", "0", "1", "2") if allow_empty else ("0", "1", "2")
    if any(r["grade"].strip() not in valid for r in rows):
        raise ValueError("blind sheet has ungraded or invalid rows")
    for r in rows:
        original = data.sheets["gemini"][(r["qid"], r["doc_id"])]
        for column in ("query", "snippet"):
            if r[column] != original[column]:
                raise ValueError(f"{column} changed for {(r['qid'], r['doc_id'])}")
    prepared = json.loads(path.with_suffix(".prepared.json").read_text())
    if (header != COLUMNS or prepared.get("provenance") != data.provenance
            or prepared.get("model") != CLAUDE_MODEL or prepared.get("seed") != BLIND_SEED):
        raise ValueError("blind sheet prepared provenance differs from the current protocol")
    blank = io.StringIO(newline="")
    writer = csv.DictWriter(blank, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({**r, "grade": ""} for r in rows)
    if hashlib.sha256(blank.getvalue().encode()).hexdigest() != prepared.get("prepared_sha256"):
        raise ValueError("blind sheet differs from its prepared hash after clearing grades")
    return {(r["qid"], r["doc_id"]): int(r["grade"]) for r in rows if r["grade"].strip()}


def median_variant(data, path: Path) -> dict:
    gemini, gemma = data.grades("gemini"), data.grades("gemma12")
    third = _third(data, path)
    merged = dict(gemini)
    for key, grade in third.items():
        merged[key] = int(statistics.median((gemini[key], gemma[key], grade)))
    return merged


def adjudication_summary(data, path: Path) -> dict:
    gemini, gemma = data.grades("gemini"), data.grades("gemma12")
    third = _third(data, path)
    sided = {"gemini": 0, "gemma12": 0, "neither": 0}
    for key, grade in third.items():
        sided["gemini" if grade == gemini[key] else "gemma12" if grade == gemma[key] else "neither"] += 1
    return {"kind": "blind cross-family adjudication", "model": CLAUDE_MODEL, "disagreements": len(third),
            "sided_with": sided, "sheet_sha256": sha256(path)}


def evaluate(data, path: Path, out: Path, qrels_path: Path = Path("paper/data/qrels-median3.txt"),
             reference: str = "adaptive") -> dict:
    """ua_par_median3.json in the format of the other variants, plus adjudication.json."""
    from paper.eval.judge_sensitivity import CAVEATS, _non_answer, _population, _target_coverage, pool_coverage
    from paper.eval.report_main import build_report

    grades = median_variant(data, path)
    qrels: dict = {}
    for (qid, doc), grade in grades.items():
        qrels.setdefault(qid, {})[doc] = grade
    report = build_report(data.queries, qrels, data.runs, reference=reference)
    summary = adjudication_summary(data, path)
    report.update(variant="median3", provenance={**data.provenance, "adjudication": summary},
                  population=_population(data, grades), pool_coverage=pool_coverage(data), caveats=CAVEATS,
                  non_answer_diagnostics=_non_answer(data, grades), target_diagnostics=_target_coverage(data, grades))
    write_json(out / "ua_par_median3.json", report)
    write_json(out / "adjudication.json", summary)
    with qrels_path.open("w", encoding="utf-8") as fh:
        for (qid, doc), grade in sorted(grades.items()):
            fh.write(f"{qid} 0 {doc} {grade}\n")
    return report


def main() -> int:
    import argparse
    import json

    from paper.judge.llm_judge import judge_batch_with_agy

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "judge", "evaluate"))
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--out", type=Path, default=Path("paper/results"))
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()
    data = load_snapshot(args.snapshot)
    if args.command == "prepare":
        print(json.dumps({k: v for k, v in prepare_blind(data, args.sheet).items() if k != "provenance"}))
    elif args.command == "judge":
        _third(data, args.sheet, allow_empty=True)
        judge = lambda pairs: judge_batch_with_agy(pairs, model=CLAUDE_MODEL)  # noqa: E731
        log = args.sheet.with_suffix(".calls.jsonl")
        print(json.dumps(judge_sheet_batched(args.sheet, judge, log, batch_size=args.batch)))
    else:
        report = evaluate(data, args.sheet, args.out)
        print(json.dumps(report["provenance"]["adjudication"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
