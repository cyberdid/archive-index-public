"""The author's blind audit of the LLM relevance labels (plan: author spot check; audit of 2026-10-22).

A stratified sample (the same number of pairs for each final grade 0, 1, 2) is shown in random order with the same
query and snippet the judges saw and no grades of any judge. Once the author has graded every row, compare_audit
reports agreement of each judge with the author. Until then the paper may not claim a human check.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from paper.judge.agreement import cohen_kappa

COLUMNS = ["item", "qid", "query", "doc_id", "snippet", "grade", "note"]
SEED = 2028
DEFAULT_SHEET = Path("paper/data/judgments/human-audit.tsv")


def prepare_audit(final: dict, snippets: dict, queries: dict, path: Path, per_grade: int = 50,
                  seed: int = SEED) -> dict:
    """final: (qid, doc) -> grade of the reported qrels; exclusive creation, never overwrites a sheet."""
    rng = random.Random(seed)
    chosen = []
    for grade in (0, 1, 2):
        keys = sorted(k for k, g in final.items() if g == grade)
        chosen += rng.sample(keys, min(per_grade, len(keys)))
    rng.shuffle(chosen)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item, (qid, doc) in enumerate(chosen, start=1):
            writer.writerow({"item": item, "qid": qid, "query": queries.get(qid, ""), "doc_id": doc,
                             "snippet": snippets[(qid, doc)], "grade": "", "note": ""})
    return {"pairs": len(chosen), "per_grade": per_grade, "seed": seed}


def compare_audit(path: Path, judges: dict[str, dict]) -> dict:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if any(r["grade"].strip() not in ("0", "1", "2") for r in rows):
        raise ValueError("the audit sheet has ungraded or invalid rows; the author has not finished it")
    human = {(r["qid"], r["doc_id"]): int(r["grade"]) for r in rows}
    if not human or len(human) != len(rows):
        raise ValueError("the audit sheet is empty or contains duplicate pairs")
    out = {"pairs": len(human), "judges": {}}
    for name, grades in judges.items():
        keys = [k for k in human if k in grades]
        if len(keys) != len(human):
            raise ValueError(f"judge {name} does not cover every audited pair")
        out["judges"][name] = {
            "pairs": len(keys),
            "kappa": cohen_kappa({k: human[k] for k in keys}, {k: grades[k] for k in keys}),
            "exact_agreement": sum(human[k] == grades[k] for k in keys) / len(keys),
            "mean_judge_minus_human": float(np.mean([grades[k] - human[k] for k in keys])),
        }
    return out


def main() -> int:
    import argparse
    import json

    from paper.eval.metrics import read_qrels
    from paper.judge.pool import read_sheet
    from paper.queries.schema import load_queries

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "compare"))
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    args = parser.parse_args()
    data = Path("paper/data")
    if args.command == "prepare":
        final = {(q, d): g for q, docs in read_qrels(data / "qrels.txt").items() for d, g in docs.items()}
        with (data / "judgments" / "gemini.tsv").open(encoding="utf-8", newline="") as fh:
            snippets = {(r["qid"], r["doc_id"]): r["snippet"] for r in csv.DictReader(fh, delimiter="\t")}
        queries = {q.qid: q.text for q in load_queries(data / "queries.tsv")}
        print(json.dumps(prepare_audit(final, snippets, queries, args.sheet)))
    else:
        judges = {name: read_sheet(data / "judgments" / f"{name}.tsv") for name in ("gemini", "gemma12", "gemma", "gpt")}
        judges["reviewed_qrels"] = {(q, d): g for q, docs in read_qrels(data / "qrels.txt").items()
                                    for d, g in docs.items()}
        out = compare_audit(args.sheet, judges)
        Path("paper/results/human_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
