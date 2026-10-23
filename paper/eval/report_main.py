"""UA-PAR results: every system, overall and by category and language, with bootstrap CIs and Holm-adjusted
comparisons against the full system (plan Task 8, step 5). Writes paper/results/ua_par.json."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from paper.eval.metrics import DEFAULT_MEASURES, per_query, read_qrels, read_run
from paper.eval.stats import bootstrap_ci, compare_all


def _summary(scores: dict[str, dict[str, float]], qids: list[str]) -> dict:
    out = {}
    for measure in DEFAULT_MEASURES:
        values = [scores[q][measure] for q in qids if q in scores]
        if values:
            mean, low, high = bootstrap_ci(values)
            out[measure] = {"mean": mean, "ci_low": low, "ci_high": high, "n": len(values)}
    return out


def build_report(queries, qrels, runs: dict[str, dict], reference: str = "adaptive") -> dict:
    judged = [q for q in queries if q.qid in qrels and q.category != "no_answer"]
    qids = [q.qid for q in judged]
    scores = {name: per_query({q: qrels[q] for q in qids}, run) for name, run in runs.items()}
    report = {"queries": len(qids), "systems": {}, "comparisons": {}}
    for name, s in scores.items():
        report["systems"][name] = {
            "all": _summary(s, qids),
            "by_category": {c: _summary(s, [q.qid for q in judged if q.category == c])
                            for c in sorted({q.category for q in judged})},
            "by_language": {lang: _summary(s, [q.qid for q in judged if q.lang == lang])
                            for lang in sorted({q.lang for q in judged})},
        }
    if reference in scores:
        for measure in DEFAULT_MEASURES:
            per_system = {name: {q: s[q][measure] for q in qids} for name, s in scores.items()}
            report["comparisons"][measure] = [asdict(c) for c in compare_all(per_system, reference)]
    return report


def main() -> int:
    from paper.queries.schema import load_queries

    queries = load_queries(Path("paper/data/queries.tsv"))
    qrels = read_qrels(Path("paper/data/qrels.txt"))
    runs = {p.stem: read_run(p) for p in sorted(Path("paper/data/runs").glob("*.trec"))}
    report = build_report(queries, qrels, runs)
    out = Path("paper/results/ua_par.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}: {report['queries']} judged queries, {len(runs)} systems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
