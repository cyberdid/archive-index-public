"""Controlled encoder comparison (audit of 2026-10-22): E5 against EG2 on the same 4137 text documents.

dense_eg2_text and adaptive_text (paper.runs.systems.CONTROLS) were not in the judged pool, so some of their top-10
documents may be unjudged. Every metric is reported twice: unjudged = non-relevant (as trec_eval does), and on the
condensed run that drops unjudged documents (Sakai's condensed-list evaluation), together with the unjudged share.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper.eval.metrics import per_query, read_qrels, read_run
from paper.eval.stats import bootstrap_ci, randomization_test

MEASURES = ("nDCG@10", "R@100", "Success@1")
QRELS_VARIANTS = {"gemini_reviewed": "qrels-gemini-reviewed.txt", "gemini_initial": "qrels-gemini.txt",
                  "gemma12_initial": "qrels-gemma12.txt", "median3": "qrels-median3.txt"}
COMPARED = ("dense_e5", "dense_eg2", "dense_eg2_text", "adaptive", "adaptive_text")


def unjudged_at(qrels: dict, runs: dict, k: int = 10) -> float:
    total = unjudged = 0
    for qid, judged in qrels.items():
        top = sorted(runs.get(qid, {}), key=lambda d: -runs[qid][d])[:k]
        total += len(top)
        unjudged += sum(1 for doc in top if doc not in judged)
    return unjudged / total if total else 0.0


def condensed(qrels: dict, runs: dict) -> dict:
    return {qid: {doc: score for doc, score in run.items() if doc in qrels.get(qid, {})} for qid, run in runs.items()}


def compare_encoders(qrels: dict, runs: dict[str, dict], reference: str = "dense_e5", n: int = 10_000) -> dict:
    """Mean metrics per system, and paired differences (system - reference) with bootstrap CI and sign-flip p."""
    qids = sorted(qrels)
    scores, out = {}, {"queries": len(qids), "reference": reference, "systems": {}, "comparisons": {}}
    for name, run in runs.items():
        plain = per_query(qrels, run, MEASURES)
        dense = per_query(qrels, condensed(qrels, run), ("nDCG@10",))
        scores[name] = {**{m: [plain[q][m] for q in qids] for m in MEASURES},
                        "nDCG@10_condensed": [dense[q]["nDCG@10"] for q in qids]}
        out["systems"][name] = {**{m: float(np.mean(v)) for m, v in scores[name].items()},
                                "unjudged@10": unjudged_at(qrels, run)}
    for name in runs:
        if name == reference:
            continue
        out["comparisons"][name] = {}
        for metric in (*MEASURES, "nDCG@10_condensed"):
            diffs = np.array(scores[name][metric]) - np.array(scores[reference][metric])
            mean, low, high = bootstrap_ci(diffs, n=n)
            out["comparisons"][name][metric] = {"mean": mean, "low": low, "high": high,
                                                "p": randomization_test(scores[name][metric], scores[reference][metric], n=n)}
    return out


def main() -> int:
    from archive_index.embed import Embedder
    from paper.queries.schema import load_queries
    from paper.runs.systems import CONTROLS, run_system, write_trec_run

    queries = load_queries(Path("paper/data/queries.tsv"))
    corpus, indexes = Path("paper/data/ua-par"), Path("paper/data/indexes")
    control_dir = Path("paper/data/runs-control")
    embedders = {"eg2": Embedder()}
    for name, config in CONTROLS.items():
        write_trec_run(control_dir / f"{name}.trec", name, run_system(config, queries, corpus, indexes, embedders))
    runs = {name: read_run(Path("paper/data/runs") / f"{name}.trec") for name in COMPARED if name not in CONTROLS}
    runs.update({name: read_run(control_dir / f"{name}.trec") for name in CONTROLS})
    answerable = {q.qid for q in queries if q.category != "no_answer"}
    report = {"note": __doc__.strip().splitlines()[0], "variants": {}}
    for variant, filename in QRELS_VARIANTS.items():
        path = Path("paper/data") / filename
        if path.exists():
            qrels = {qid: judged for qid, judged in read_qrels(path).items() if qid in answerable}
            report["variants"][variant] = compare_encoders(qrels, runs)
    out = Path("paper/results/encoder_control.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0




def covers_fix_main() -> int:
    """Adaptive systems re-run after the whole-word coverage fix, compared with the frozen runs on the same qrels."""
    from archive_index.embed import Embedder
    from paper.queries.schema import load_queries
    from paper.runs.systems import SYSTEMS, run_system, write_trec_run

    queries = load_queries(Path("paper/data/queries.tsv"))
    corpus, indexes, fixed_dir = Path("paper/data/ua-par"), Path("paper/data/indexes"), Path("paper/data/runs-covers-fix")
    embedders = {"eg2": Embedder()}
    names = [name for name in SYSTEMS if name.startswith("adaptive")]
    runs = {name: read_run(Path("paper/data/runs") / f"{name}.trec") for name in ("dense_eg2", "dense_e5", *names)}
    changed = {}
    for name in names:
        rows = run_system(SYSTEMS[name], queries, corpus, indexes, embedders)
        write_trec_run(fixed_dir / f"{name}.trec", f"{name}_fixed", rows)
        runs[f"{name}_fixed"] = read_run(fixed_dir / f"{name}.trec")
        top = lambda run, q: sorted(run.get(q, {}), key=lambda d: -run[q][d])[:10]  # noqa: E731
        changed[name] = sum(top(runs[name], q.qid) != top(runs[f"{name}_fixed"], q.qid) for q in queries)
    answerable = {q.qid for q in queries if q.category != "no_answer"}
    report = {"note": covers_fix_main.__doc__, "queries_with_changed_top10": changed, "variants": {}}
    for variant, filename in QRELS_VARIANTS.items():
        path = Path("paper/data") / filename
        if path.exists():
            qrels = {qid: judged for qid, judged in read_qrels(path).items() if qid in answerable}
            report["variants"][variant] = compare_encoders(qrels, runs, reference="adaptive_fixed")
    out = Path("paper/results/covers_fix.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(covers_fix_main() if sys.argv[1:] == ["covers-fix"] else main())
