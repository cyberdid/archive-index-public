"""Private field study (plan Task 11): the author's real archive and the existing private sets in evals/.

Only aggregate numbers leave this script: counts and hit rates with bootstrap CIs per set and query kind. No query
text, path or snippet is written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from paper.eval.stats import bootstrap_ci

SETS = {"main": "evals/queries.jsonl", "holdout7": "evals/queries_holdout_seed7.jsonl",
        "holdout11": "evals/queries_holdout_seed11.jsonl"}


def _ci(values) -> dict:
    mean, low, high = bootstrap_ci([float(v) for v in values])
    return {"mean": mean, "ci_low": low, "ci_high": high}


def aggregate(results) -> dict:
    """results: (set, kind, outcome); outcome is a rank or None for exact, a bool for absent and described."""
    out: dict = {}
    for name in sorted({r[0] for r in results}):
        rows = [r for r in results if r[0] == name]
        exact = [r[2] for r in rows if r[1] == "exact"]
        absent = [r[2] for r in rows if r[1] == "absent"]
        described = [r[2] for r in rows if r[1] == "described"]
        out[name] = {}
        if exact:
            out[name]["exact"] = {"n": len(exact), "hit@1": _ci([r == 1 for r in exact]),
                                  "hit@5": _ci([bool(r) and r <= 5 for r in exact]),
                                  "mrr@10": _ci([1 / r if r and r <= 10 else 0.0 for r in exact])}
        if absent:
            out[name]["absent"] = {"n": len(absent), "pass": _ci(absent)}
        if described:
            out[name]["described"] = {"n": len(described), "hit@5": _ci(described)}
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def run_private(db: Path) -> dict:
    from archive_index.embed import Embedder
    from archive_index.search import VectorCache, hybrid
    from archive_index.store import Store

    store = Store(db)
    cache = VectorCache(store)
    embedder = Embedder()
    results = []
    try:
        for name, path in SETS.items():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                q = json.loads(line)
                if q["kind"] == "exact":
                    hits = hybrid(store, cache, q["query"], embedder.query(q["query"]), k=10, per_document=0)
                    bodies = store.chunk_rows([h.chunk_id for h in hits])
                    rank = next((i for i, h in enumerate(hits, start=1) if h.chunk_id in bodies
                                 and _norm(q["expect_phrase"]) in _norm(bodies[h.chunk_id].body)), None)
                    results.append((name, "exact", rank))
                elif q["kind"] == "absent":
                    results.append((name, "absent", not hybrid(store, cache, q["query"], None, mode="lexical", k=5)))
                else:
                    hits = hybrid(store, cache, q["query"], embedder.query(q["query"]), k=5, per_document=0)
                    results.append((name, "described", any(q["expect_project"].lower() in (h.project + " " + h.path).lower()
                                                           for h in hits)))
        stats = {"documents": store.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                 "chunks": store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]}
    finally:
        store.close()
    return {"archive": stats, "sets": aggregate(results)}


def main() -> int:
    out = Path("paper/results/field_aggregate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run_private(Path.home() / ".local/share/archive-index/archive.sqlite3")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
