"""Draw a search as a picture of neurons: command line wrapper around archive_index.viz.

usage: python tools/visualize_search.py "запит" [-k 40] [--mode hybrid] [--near 0.8] [--out viz/x.html]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archive_index.cli import DEFAULT_DB  # noqa: E402
from archive_index.embed import make_embedder  # noqa: E402
from archive_index.search import VectorCache, hybrid  # noqa: E402
from archive_index.store import Store  # noqa: E402
from archive_index.viz import build_graph, fetch_vectors, render  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=40)
    parser.add_argument("--mode", choices=("hybrid", "lexical", "semantic"), default="hybrid")
    parser.add_argument("--near", type=float, default=0.8, help="cosine threshold for links between hits")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=ROOT / "viz" / "search.html")
    args = parser.parse_args()

    store = Store(args.db)
    cache = VectorCache(store)
    embedder = make_embedder()
    qv = None if args.mode == "lexical" else np.asarray(embedder.query(args.query), dtype=np.float32)
    hits = hybrid(store, cache, args.query, qv, k=args.k, mode=args.mode, per_document=0)
    vectors = fetch_vectors(store, [h.chunk_id for h in hits])
    nodes, links = build_graph(args.query, hits, vectors, qv, args.near)
    render(args.query, nodes, links, args.out)
    store.close()
    print(f"wrote {args.out}: {len(hits)} hits, {len(links)} links "
          f"({sum(1 for l in links if l['kind'] == 'near')} between hits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
