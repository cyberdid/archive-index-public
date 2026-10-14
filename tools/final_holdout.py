"""Independent final check. Nothing here was used to choose the merge rule.

- DESCRIBED: 12 Ukrainian/English descriptions written BEFORE running anything, each with a gold project.
- EXACT: 40 phrases from a fresh random seed (303) that was not used in any earlier choice.
Compares lexical-only, semantic-only, lexical_first and adaptive. Reports recall@5 for each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from archive_index.embed import Embedder
from archive_index.lexical import query_expression
from archive_index.search import VectorCache
from archive_index.store import Store
from tools.tune_weights import sample_exact

DESCRIBED = [
    # Add your own pairs (description, project name); the query builder uses the first six.
]


def run(store, cache, embedder, pairs_exact, pairs_desc):
    def lex_ids(q):
        return [cid for cid, _ in store.lexical(query_expression(q), collections=None, project=None, limit=100)]

    def sem_ids(q):
        return cache.semantic(embedder.query(q), collections=None, project=None, pool=100)

    def first_lexical(q, lex):
        return None if not lex else store.chunk_rows([lex[0]]).get(lex[0])

    from archive_index.search import _covers

    def adaptive(q, lex, sem):
        if lex and _covers(store, q, lex[0]) < 0.9:
            first, second = sem, lex
        else:
            first, second = lex, sem
        order = []
        for cid in first + second:
            if cid not in order:
                order.append(cid)
        return order[:5]

    def lexfirst(q, lex, sem):
        order = []
        for cid in lex + sem:
            if cid not in order:
                order.append(cid)
        return order[:5]

    results = {}
    exact_rows = [(q, g, lex_ids(q), sem_ids(q)) for q, g in pairs_exact]
    results["exact"] = {
        "lexical": round(sum(g in lex[:5] for _, g, lex, _ in exact_rows) / len(exact_rows), 3),
        "semantic": round(sum(g in sem[:5] for _, g, _, sem in exact_rows) / len(exact_rows), 3),
        "lexical_first": round(sum(g in lexfirst(q, lex, sem) for q, g, lex, sem in exact_rows) / len(exact_rows), 3),
        "adaptive": round(sum(g in adaptive(q, lex, sem) for q, g, lex, sem in exact_rows) / len(exact_rows), 3),
    }

    desc_rows = []
    for q, project in pairs_desc:
        lex, sem = lex_ids(q), sem_ids(q)
        desc_rows.append((q, project, lex, sem))

    def hit(ids, project):
        rows = store.chunk_rows(ids[:5])
        return any(project.lower() in (rows[i].project + " " + rows[i].path).lower()
                   for i in ids[:5] if i in rows)

    results["described"] = {
        "lexical": sum(hit(lex, p) for _, p, lex, _ in desc_rows),
        "semantic": sum(hit(sem, p) for _, p, _, sem in desc_rows),
        "lexical_first": sum(hit(lexfirst(q, lex, sem), p) for q, p, lex, sem in desc_rows),
        "adaptive": sum(hit(adaptive(q, lex, sem), p) for q, p, lex, sem in desc_rows),
        "of": len(desc_rows),
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".local/share/archive-index/archive.sqlite3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    exact = sample_exact(args.db, 40, seed=303, exclude=set())
    store = Store(args.db)
    cache = VectorCache(store)
    embedder = Embedder()
    results = run(store, cache, embedder, exact, DESCRIBED)
    results["exact_queries"] = len(exact)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
