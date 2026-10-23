"""Exact/ANN vector scaling (audit of 2026-10-22): optimized exact NumPy top-k search,
against exact and HNSW indexes from Faiss, on collections grown from the real UA-PAR EG2 vectors.

The real vectorized chunks come first; larger collections add noisy copies (Gaussian noise, renormalized).
These synthetic copies do not establish the semantics of a larger real archive. Queries are the 200 UA-PAR queries
embedded by EG2. Recall@10 of an approximate index is measured against the exact top 10 on the same collection.

  export     (archive-index environment) write paper/data/ann/base.npy and queries.npy
  run        (isolated environment with faiss-cpu 1.15.1) measure and write paper/results/ann_scaling.json
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np

SIZES = (100_000, 300_000, 1_000_000, 2_000_000)
SIGMA = 0.05
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = (64, 256)
K = 10
DATA = Path("paper/data/ann")


def scaled_collection(base: np.ndarray, n: int, sigma: float = SIGMA, seed: int = 2026) -> np.ndarray:
    """n unit vectors: the real ones first, then noisy copies of randomly chosen real ones."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, base.shape[1]), dtype=np.float32)
    m = min(n, len(base))
    out[:m] = base[:m]
    for start in range(m, n, 100_000):
        stop = min(n, start + 100_000)
        picks = base[rng.integers(0, len(base), size=stop - start)]
        noisy = picks + rng.normal(scale=sigma, size=picks.shape).astype(np.float32)
        out[start:stop] = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
    return out


def exact_top_k(matrix: np.ndarray, queries: np.ndarray, k: int = K) -> np.ndarray:
    sims = queries @ matrix.T
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def recall_at_k(exact: np.ndarray, approx: np.ndarray, k: int = K) -> float:
    return float(np.mean([len(set(e[:k]) & set(a[:k])) / k for e, a in zip(exact, approx)]))


def _per_query_ms(fn, queries) -> dict:
    ms = []
    for q in queries:
        started = time.perf_counter()
        fn(q[None, :])
        ms.append((time.perf_counter() - started) * 1000)
    return {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)), "mean": float(np.mean(ms))}


def run(checkpoint: Path | None = None) -> dict:
    import faiss

    base, queries = np.load(DATA / "base.npy"), np.load(DATA / "queries.npy")
    inputs = {}
    for name in ("base.npy", "queries.npy"):
        with (DATA / name).open("rb") as source:
            inputs[name] = hashlib.file_digest(source, "sha256").hexdigest()
    results = {"faiss": faiss.__version__, "numpy": np.__version__, "sigma": SIGMA, "k": K, "real_vectors": len(base),
               "queries": len(queries), "hnsw": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
               "source_sha256": inputs, "planned_sizes": list(SIZES), "seed": 2026,
               "faiss_threads": faiss.omp_get_max_threads(), "complete": False, "sizes": {}}
    if checkpoint is not None and checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        for key in results.keys() - {"sizes", "complete"}:
            if previous.get(key) != results[key]:
                raise ValueError(f"ANN checkpoint does not match current inputs/settings: {key}")
        results["sizes"] = previous["sizes"]
    for n in SIZES:
        if str(n) in results["sizes"]:
            continue
        print(f"starting {n} vectors", flush=True)
        matrix = scaled_collection(base, n)
        # A full (200, 2M) argpartition allocates multi-GB score/index arrays.
        # Query blocks preserve the exact reference while bounding temporary memory.
        exact = np.vstack([exact_top_k(matrix, queries[i:i + 4]) for i in range(0, len(queries), 4)])
        row = {"matrix_mb": matrix.nbytes / 2**20,
               "numpy_exact_ms": _per_query_ms(lambda q: exact_top_k(matrix, q), queries)}
        flat = faiss.IndexFlatIP(matrix.shape[1])
        flat.add(matrix)
        row["faiss_flat_ms"] = _per_query_ms(lambda q: flat.search(q, K), queries)
        del flat  # do not retain a second exact copy while constructing HNSW
        started = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(matrix.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        hnsw.add(matrix)
        row["hnsw_build_s"] = time.perf_counter() - started
        row["hnsw_extra_mb"] = (hnsw.ntotal * HNSW_M * 2 * 4) / 2**20  # level-0 links, an estimate stated as such
        for ef in HNSW_EF_SEARCH:
            hnsw.hnsw.efSearch = ef
            _, approx = hnsw.search(queries, K)
            row[f"hnsw_ef{ef}"] = {"recall@10": recall_at_k(exact, approx),
                                   **_per_query_ms(lambda q: hnsw.search(q, K), queries)}
        results["sizes"][str(n)] = row
        results["complete"] = all(str(size) in results["sizes"] for size in SIZES)
        if checkpoint is not None:
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, checkpoint)
        print(n, json.dumps(row), flush=True)
        del matrix, hnsw
    return results


def export() -> None:
    from archive_index.embed import Embedder
    from archive_index.store import Store
    from paper.queries.schema import load_queries

    store = Store(Path("paper/data/indexes/full.sqlite3"))
    try:
        _ids, matrix, _cols, _projects = store.all_vectors()
    finally:
        store.close()
    embedder = Embedder()
    queries = np.stack([embedder.query(q.text) for q in load_queries(Path("paper/data/queries.tsv"))])
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "base.npy", np.asarray(matrix, dtype=np.float32))
    np.save(DATA / "queries.npy", np.asarray(queries, dtype=np.float32))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["export"]:
        export()
    else:
        out = Path("paper/results/ann_scaling.json")
        run(checkpoint=out)
        print(f"wrote {out}")
