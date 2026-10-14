"""Run evals/queries.jsonl against the index and write a report (plan G2). Standard library + archive_index.

exact:     hit if one of the top-k fragments contains the phrase (case and spacing ignored).
absent:    pass if the word search finds nothing.
described: hit if one of the top-5 fragments belongs to the expected project.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

from archive_index.embed import Embedder
from archive_index.search import VectorCache, hybrid
from archive_index.store import Store


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".local/share/archive-index/archive.sqlite3")
    parser.add_argument("--queries", type=Path, default=Path("evals/queries.jsonl"))
    parser.add_argument("--reports", type=Path, default=Path("evals/reports"))
    parser.add_argument("--baseline", type=Path, default=None,
                        help="an earlier report (.json): the markdown shows the change per format (before/after)")
    args = parser.parse_args()
    queries = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line]
    store = Store(args.db)
    cache = VectorCache(store)
    embedder = Embedder()
    exact_rank: dict[str, list[int | None]] = defaultdict(list)
    misses: list[dict] = []  # exact queries whose phrase is not in the top 10: listed in the report
    absent_ok = []
    absent_similarity: list[float] = []
    described_ok = []
    for q in queries:
        if q["kind"] == "exact":
            hits = hybrid(store, cache, q["query"], embedder.query(q["query"]), k=10, per_document=0)
            rank = None
            for i, h in enumerate(hits, start=1):
                body = store.chunk_rows([h.chunk_id]).get(h.chunk_id)
                if body and _norm(q["expect_phrase"]) in _norm(body.body):
                    rank = i
                    break
            exact_rank[q["format"]].append(rank)
            if rank is None:
                misses.append({"format": q["format"], "query": q["query"], "source": q.get("source_path", ""),
                               "locator": q.get("source_locator", ""),
                               "top": [h.title[:40] for h in hits[:3]]})
        elif q["kind"] == "absent":
            found = hybrid(store, cache, q["query"], None, mode="lexical", k=5)
            absent_ok.append(len(found) == 0)
            # Meaning side for absent words (plan G2): how close the nearest fragment is in vector space.
            near = hybrid(store, cache, q["query"], embedder.query(q["query"]), mode="semantic", k=1)
            absent_similarity.append(near[0].similarity if near and near[0].similarity is not None else 0.0)
        else:
            hits = hybrid(store, cache, q["query"], embedder.query(q["query"]), k=5, per_document=0)
            described_ok.append(any(q["expect_project"].lower() in (h.project + " " + h.path).lower()
                                    for h in hits))
    store.close()

    def metrics(ranks: list[int | None]) -> dict[str, float]:
        n = len(ranks) or 1
        return {
            "n": len(ranks),
            "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
            "hit@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
            "hit@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
            "mrr@10": round(sum(1 / r for r in ranks if r and r <= 10) / n, 3),
        }

    all_exact = [r for ranks in exact_rank.values() for r in ranks]
    report = {
        "exact_overall": metrics(all_exact),
        "exact_by_format": {f: metrics(r) for f, r in sorted(exact_rank.items())},
        "absent_pass_rate": round(sum(absent_ok) / (len(absent_ok) or 1), 3),
        "absent_semantic_top1": {"mean": round(sum(absent_similarity) / (len(absent_similarity) or 1), 3),
                                 "max": round(max(absent_similarity, default=0.0), 3)},
        "described_hit@5": round(sum(described_ok) / (len(described_ok) or 1), 3),
        "queries": len(queries),
        "misses": misses,
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    (args.reports / f"{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Оцінка archive-index — {stamp}", "", f"Запитів: {len(queries)}", "",
             "## Точні фрази", "", "| формат | n | hit@1 | hit@3 | hit@5 | MRR@10 |", "|---|---:|---:|---:|---:|---:|"]
    for f, m in [("усі", report["exact_overall"])] + list(report["exact_by_format"].items()):
        lines.append(f"| {f} | {m['n']} | {m['hit@1']} | {m['hit@3']} | {m['hit@5']} | {m['mrr@10']} |")
    lines += ["", f"Відсутні слова: без збігів у {report['absent_pass_rate']:.0%} запитів.",
              f"Описові запити: hit@5 = {report['described_hit@5']:.0%}."]
    lines += ["", f"Відсутні слова, семантична схожість найближчого фрагмента: середня "
              f"{report['absent_semantic_top1']['mean']}, максимум {report['absent_semantic_top1']['max']}."]
    if args.baseline is not None:
        before = json.loads(args.baseline.read_text(encoding="utf-8"))
        lines += ["", f"## Зміна від {args.baseline.name} (до → після)", "", "| формат | hit@1 | hit@5 | MRR@10 |",
                  "|---|---|---|---|"]
        pairs = [("усі", before["exact_overall"], report["exact_overall"])]
        for f, m in report["exact_by_format"].items():
            if f in before.get("exact_by_format", {}):
                pairs.append((f, before["exact_by_format"][f], m))
        for f, b, a in pairs:
            lines.append(f"| {f} | {b['hit@1']} → {a['hit@1']} | {b['hit@5']} → {a['hit@5']} | "
                         f"{b['mrr@10']} → {a['mrr@10']} |")
    (args.reports / f"{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
