"""Agreement between the LLM judges and their checks against known answers (plan Task 7, spec section 4).

Kappa is linear-weighted (grades 0-2); its confidence interval resamples queries, not pairs, because the pairs of one
query share the query and are not independent. Known-answer checks: the document a known-item query was written for
should get 2, and every pooled document of a no-answer query should get 0.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from paper.judge.agreement import GRADES, cohen_kappa
from paper.judge.pool import read_sheet
from paper.queries.schema import load_queries

SHEETS = {"gemini": "gemini.tsv", "gemma4-12b": "gemma12.tsv", "gemma4-e4b": "gemma.tsv", "gpt-6-luna": "gpt.tsv",
          "claude": "primary.tsv"}
N_RESAMPLES = 2000
SEED = 2026


def _weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    span = max(GRADES) - min(GRADES)
    w = 1 - np.abs(np.subtract.outer(np.array(GRADES), np.array(GRADES))) / span
    observed = np.mean(w[a, b])
    pa = np.bincount(a, minlength=len(GRADES)) / len(a)
    pb = np.bincount(b, minlength=len(GRADES)) / len(b)
    expected = pa @ w @ pb
    return 1.0 if expected == 1 else float((observed - expected) / (1 - expected))


def kappa_ci(a: dict, b: dict, n: int = N_RESAMPLES, seed: int = SEED, level: float = 0.95):
    """(kappa, low, high) on the pairs both judges graded; the interval resamples queries with replacement."""
    keys = sorted(set(a) & set(b))
    kappa = cohen_kappa({k: a[k] for k in keys}, {k: b[k] for k in keys})
    qids = sorted({q for q, _ in keys})
    by_query = {q: [] for q in qids}
    for i, (q, _) in enumerate(keys):
        by_query[q].append(i)
    ga = np.array([a[k] for k in keys])
    gb = np.array([b[k] for k in keys])
    groups = [np.array(by_query[q]) for q in qids]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        stats.append(_weighted_kappa(ga[idx], gb[idx]))
    alpha = (1 - level) / 2
    return kappa, float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def known_answer_checks(grades: dict, queries) -> dict:
    by_qid = {q.qid: q for q in queries}
    targets = [grades[(q.qid, q.target)] for q in queries
               if q.target and q.category != "no_answer" and (q.qid, q.target) in grades]
    no_answer = [g for (qid, _doc), g in grades.items() if by_qid[qid].category == "no_answer"]
    share = lambda values, grade: round(sum(1 for v in values if v == grade) / len(values), 4) if values else None
    return {"target": {"pairs": len(targets), "share_2": share(targets, 2), "share_0": share(targets, 0)},
            "no_answer": {"pairs": len(no_answer), "share_0": share(no_answer, 0), "share_2": share(no_answer, 2)},
            "mean_grade": float(np.mean(list(grades.values()))) if grades else None,
            "distribution": {g: sum(1 for v in grades.values() if v == g) for g in GRADES}}


def judge_report(judges: dict[str, dict], queries, n: int = N_RESAMPLES) -> dict:
    by_qid = {q.qid: q for q in queries}
    report = {"judges": {name: {"graded": len(g), **known_answer_checks(g, queries)} for name, g in judges.items()},
              "pairs": []}
    for name_a, name_b in combinations(judges, 2):
        a, b = judges[name_a], judges[name_b]
        keys = set(a) & set(b)
        if not keys:
            continue
        kappa, low, high = kappa_ci(a, b, n=n)
        entry = {"a": name_a, "b": name_b, "n": len(keys), "kappa": kappa, "low": low, "high": high,
                 "exact_agreement": sum(1 for k in keys if a[k] == b[k]) / len(keys),
                 "mean_a_minus_b": float(np.mean([a[k] - b[k] for k in keys])),
                 "by_category": {}, "by_language": {}}
        for field, out in (("category", entry["by_category"]), ("lang", entry["by_language"])):
            for value in sorted({getattr(by_qid[q], field) for q, _ in keys}):
                part = [k for k in keys if getattr(by_qid[k[0]], field) == value]
                out[value] = {"n": len(part), "kappa": cohen_kappa({k: a[k] for k in part}, {k: b[k] for k in part})}
        report["pairs"].append(entry)
    return report


def main() -> int:
    import json

    sheets = Path("paper/data/judgments")
    judges = {name: read_sheet(sheets / file) for name, file in SHEETS.items() if (sheets / file).exists()}
    judges = {name: grades for name, grades in judges.items() if grades}
    report = judge_report(judges, load_queries(Path("paper/data/queries.tsv")))
    out = Path("paper/results/judges.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
