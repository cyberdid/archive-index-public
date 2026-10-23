"""The ten systems of the paper (spec section 5) and their runs in TREC format (plan Task 6).

Every system goes through archive_index.search.hybrid on an index built by archive-index itself, so baselines and
ablations differ only in the switches below. Runs are document-level: the best-ranked chunk stands for its document.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from archive_index.search import VectorCache, hybrid
from archive_index.store import Store


@dataclass(frozen=True)
class SystemConfig:
    index_db: str
    mode: str            # hybrid | lexical | semantic
    strategy: str        # adaptive | rrf
    lemmatize: bool
    identifiers: bool
    phrase_rerank: bool
    embed_backend: str   # none | eg2 | e5
    embed_dim: int | None
    collections: tuple[str, ...] | None = None  # None: every collection of the index


SYSTEMS: dict[str, SystemConfig] = {
    "bm25_plain": SystemConfig("plain.sqlite3", "lexical", "adaptive", False, True, True, "none", None),
    "bm25_lemma": SystemConfig("full.sqlite3", "lexical", "adaptive", True, True, True, "none", None),
    "dense_eg2": SystemConfig("full.sqlite3", "semantic", "adaptive", True, True, True, "eg2", None),
    "dense_e5": SystemConfig("e5.sqlite3", "semantic", "adaptive", True, True, True, "e5", None),
    "rrf": SystemConfig("full.sqlite3", "hybrid", "rrf", True, True, True, "eg2", None),
    "adaptive": SystemConfig("full.sqlite3", "hybrid", "adaptive", True, True, True, "eg2", None),
    "adaptive-phrase": SystemConfig("full.sqlite3", "hybrid", "adaptive", True, True, False, "eg2", None),
    "adaptive-ident": SystemConfig("full.sqlite3", "hybrid", "adaptive", True, False, True, "eg2", None),
    "adaptive-short": SystemConfig("noshort.sqlite3", "hybrid", "adaptive", True, True, True, "eg2", None),
    "dense_eg2_256": SystemConfig("full.sqlite3", "semantic", "adaptive", True, True, True, "eg2", 256),
}
# Controls outside the pooled ten: EG2 on the E5 candidate universe (the same 4137 text documents, no benchmark
# pictures or speech), so the encoder comparison is not confounded by the index contents (audit of 2026-10-22).
TEXT_COLLECTIONS = ("wiki", "docs", "tables", "code")
CONTROLS: dict[str, SystemConfig] = {
    "dense_eg2_text": SystemConfig("full.sqlite3", "semantic", "adaptive", True, True, True, "eg2", None,
                                   TEXT_COLLECTIONS),
    "adaptive_text": SystemConfig("full.sqlite3", "hybrid", "adaptive", True, True, True, "eg2", None,
                                  TEXT_COLLECTIONS),
}
CHUNK_POOL = 400  # chunks fetched per query before collapsing to documents


def to_document_run(chunk_hits, corpus_root: Path) -> list[tuple[str, float]]:
    """Document ids (paths relative to the corpus root) in rank order, one per document; hits outside are dropped."""
    root = str(corpus_root.resolve()).rstrip("/") + "/"
    seen: set[str] = set()
    docs: list[str] = []
    for hit in chunk_hits:
        path = str(Path(hit.path).resolve())
        if not path.startswith(root):
            continue
        doc_id = path[len(root):]
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return [(doc, float(len(docs) - i)) for i, doc in enumerate(docs)]


def write_trec_run(path: Path, run_id: str, rows: list[tuple[str, list[tuple[str, float]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid, run in rows:
            for rank, (doc, score) in enumerate(run, start=1):
                fh.write(f"{qid} Q0 {doc} {rank} {score:.6f} {run_id}\n")


def _truncate(cache: VectorCache, dim: int) -> None:
    """Matryoshka truncation of the stored document vectors, renormalized, to match a truncated query."""
    cache.ensure()
    if cache.matrix.size:
        m = cache.matrix[:, :dim]
        cache.matrix = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def _query_vector(config: SystemConfig, text: str, embedders: dict):
    if config.mode == "lexical" or config.embed_backend == "none":
        return None
    v = np.asarray(embedders[config.embed_backend].query(text), dtype=np.float32)
    if config.embed_dim:
        v = v[: config.embed_dim]
        v = v / max(float(np.linalg.norm(v)), 1e-12)
    return v


def run_system(config: SystemConfig, queries, corpus_root: Path, index_dir: Path, embedders: dict,
               k: int = 100, timings: list[float] | None = None,
               cache_factory=None) -> list[tuple[str, list[tuple[str, float]]]]:
    """timings, when given, receives each query's wall time in ms (query embedding plus search).

    cache_factory(store) -> VectorCache replaces the stored vectors (e.g. by GGUF-quantized ones).
    """
    store = Store(index_dir / config.index_db)
    try:
        cache = cache_factory(store) if cache_factory else VectorCache(store)
        if config.embed_dim:
            _truncate(cache, config.embed_dim)
        rows = []
        for q in queries:
            started = time.perf_counter()
            hits = hybrid(store, cache, q.text, _query_vector(config, q.text, embedders), k=CHUNK_POOL,
                          collections=list(config.collections) if config.collections else None,
                          pool=CHUNK_POOL, mode=config.mode, strategy=config.strategy, per_document=0,
                          lemmatize=config.lemmatize, identifiers=config.identifiers,
                          phrase_rerank=config.phrase_rerank)
            if timings is not None:
                timings.append((time.perf_counter() - started) * 1000)
            rows.append((q.qid, to_document_run(hits, corpus_root)[:k]))
        return rows
    finally:
        store.close()
