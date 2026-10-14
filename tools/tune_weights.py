"""Choose the lexical weight of the hybrid fusion on a DEV set and report it on a HOLDOUT set.

DEV (seed 101) picks the weight; the weight with the best DEV recall@5 wins (ties: smaller weight).
HOLDOUT (seed 202, disjoint chunks) is never used for the choice; it reports the real effect.
Semantic and lexical rankings are computed once per query, then fused for every weight.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

from archive_index.embed import Embedder
from archive_index.lexical import query_expression
from archive_index.search import RRF_K, VectorCache
from archive_index.store import Store

WEIGHTS = [1.0, 1.5, 2.0, 3.0, 5.0]


def sample_exact(db: Path, n: int, seed: int, exclude: set[int]) -> list[tuple[str, int]]:
    conn = sqlite3.connect(db)
    rows = [r for r in conn.execute(
        "SELECT id, body FROM chunks WHERE collection IN ('wiki','git','sessions')"
        " AND length(body) > 300") if r[0] not in exclude]
    conn.close()
    rng = random.Random(seed)
    out: list[tuple[str, int]] = []
    for cid, body in rng.sample(rows, min(len(rows), n * 2)):
        words = [w for w in body.split() if len(w) > 3 and w.isalpha()]
        if len(words) < 8:
            continue
        start = rng.randrange(0, max(1, len(words) - 6))
        out.append((" ".join(words[start:start + 6]), cid))
        if len(out) == n:
            break
    return out


def fuse(lex: list[int], sem: list[int], weight: float) -> list[int]:
    scores: dict[int, float] = {}
    for rank, cid in enumerate(lex, start=1):
        scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank)
    for rank, cid in enumerate(sem, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda c: (-scores[c], c))[:5]


def rankings(store, cache, embedder, queries):
    out = []
    for query, gold in queries:
        lex = [cid for cid, _ in store.lexical(query_expression(query), collections=None,
                                               project=None, limit=100)]
        sem = cache.semantic(embedder.query(query), collections=None, project=None, pool=100)
        out.append((gold, lex, sem))
    return out


def recall(rows, weight: float) -> float:
    return round(sum(gold in fuse(lex, sem, weight) for gold, lex, sem in rows) / len(rows), 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".local/share/archive-index/archive.sqlite3")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dev = sample_exact(args.db, args.n, seed=101, exclude=set())
    dev_ids = {gold for _, gold in dev}
    hold = sample_exact(args.db, args.n, seed=202, exclude=dev_ids)

    store = Store(args.db)
    cache = VectorCache(store)
    embedder = Embedder()
    dev_rows = rankings(store, cache, embedder, dev)
    hold_rows = rankings(store, cache, embedder, hold)

    dev_scores = {w: recall(dev_rows, w) for w in WEIGHTS}
    best = max(WEIGHTS, key=lambda w: (dev_scores[w], -w))
    report = {
        "dev_queries": len(dev_rows), "holdout_queries": len(hold_rows),
        "dev_recall_at_5": dev_scores,
        "holdout_recall_at_5": {w: recall(hold_rows, w) for w in WEIGHTS},
        "chosen_weight": best,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
