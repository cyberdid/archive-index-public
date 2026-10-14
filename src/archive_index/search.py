"""Hybrid search: FTS5 BM25 and cosine similarity, fused with reciprocal rank fusion."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import re

import numpy as np

from .lexical import query_expression
from .store import Store

RRF_K = 60
SNIPPET_CHARS = 280


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    collection: str
    project: str
    title: str
    path: str
    locator: str
    heading: str
    snippet: str
    occurred_at: str
    score: float
    via: str  # "lexical", "semantic" or "both"
    similarity: float | None = None  # cosine similarity when the chunk was found by meaning
    more_in_document: int = 0  # other matching fragments of the same document that are not shown


class VectorCache:
    """Holds all chunk vectors in memory; rebuilt when the store changes."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.ids = np.zeros(0, dtype=np.int64)
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.collections = np.array([], dtype=object)
        self.projects = np.array([], dtype=object)
        self._loaded = False

    def refresh(self) -> None:
        ids, matrix, cols, projs = self.store.all_vectors()
        self.ids, self.matrix = ids, matrix
        self.collections = np.array(cols, dtype=object)
        self.projects = np.array(projs, dtype=object)
        self._loaded = True

    def ensure(self) -> None:
        if not self._loaded:
            self.refresh()

    def semantic(self, query_vector: np.ndarray, *, collections, project, pool: int) -> list[int]:
        return [cid for cid, _ in self.semantic_scored(query_vector, collections=collections,
                                                       project=project, pool=pool)]

    def semantic_scored(self, query_vector: np.ndarray, *, collections, project,
                        pool: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        self.ensure()
        if self.matrix.size == 0:
            return []
        mask = np.ones(len(self.ids), dtype=bool)
        if collections:
            mask &= np.isin(self.collections, list(collections))
        if project is not None:
            mask &= np.array([p.startswith(project) for p in self.projects], dtype=bool)
        if allowed is not None:  # format filter: only chunks of the wanted file types
            mask &= np.isin(self.ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
        candidates = np.nonzero(mask)[0]
        if candidates.size == 0:
            return []
        sims = self.matrix[candidates] @ query_vector
        ranked = np.argsort(-sims)[:pool]
        return [(int(self.ids[candidates[i]]), float(sims[i])) for i in ranked]


COVERAGE_MIN = 0.9


def _covers(store: Store, query: str, chunk_id: int) -> float:
    """Share of query words that occur in the top lexical chunk (1.0 = the phrase is there)."""
    from .lexical import tokenize

    tokens = tokenize(query)
    row = store.chunk_rows([chunk_id]).get(chunk_id)
    if not tokens or row is None:
        return 0.0
    body = row.body.lower()
    return sum(t in body for t in tokens) / len(tokens)


_WORD = re.compile(r"[^\W_]+")


def _phrase_close(query: str, body: str, slack: int = 3) -> bool:
    """True when the query words occur in the body in order and close together (a phrase with slack).

    Numbers are where BM25 cannot tell a row from its neighbours: "192 168 169 tcp" as a sequence
    separates the right row from rows that only share some of the numbers.
    """
    from .lexical import tokenize

    query_words = tokenize(query)
    if len(query_words) < 2:
        return False
    body_words = _WORD.findall(body.lower())
    n = len(query_words)
    for start, word in enumerate(body_words):
        if word != query_words[0]:
            continue
        need, end = 1, start
        for pos in range(start + 1, min(len(body_words), start + n + slack + 1)):
            if body_words[pos] == query_words[need]:
                need += 1
                end = pos
                if need == n:
                    break
        if need == n and end - start + 1 <= n + slack:
            return True
    return False


def _snippet(body: str) -> str:
    flat = " ".join(body.split())
    return flat[:SNIPPET_CHARS] + ("…" if len(flat) > SNIPPET_CHARS else "")


def hybrid(
    store: Store,
    cache: VectorCache,
    query: str,
    query_vector: np.ndarray | None,
    *,
    collections: list[str] | None = None,
    project: str | None = None,
    k: int = 10,
    pool: int = 100,
    mode: str = "hybrid",
    strategy: str = "adaptive",
    lexical_weight: float = 1.0,
    per_document: int = 3,
    suffixes: tuple[str, ...] | None = None,
) -> list[Hit]:
    """strategy='lexical_first' (default): exact-word matches first, semantic results fill the rest.
    strategy='rrf': reciprocal rank fusion with weight lexical_weight (kept for comparison)."""
    if mode not in ("hybrid", "lexical", "semantic"):
        raise ValueError(f"unknown mode {mode!r}")
    lexical_ids: list[int] = []
    if mode in ("hybrid", "lexical"):
        match = query_expression(query)
        lexical_ids = [
            cid for cid, _ in store.lexical(
                match, collections=collections, project=project, limit=pool, suffixes=suffixes
            )
        ]
    semantic_ids: list[int] = []
    similarity_of: dict[int, float] = {}
    if mode in ("hybrid", "semantic") and query_vector is not None:
        allowed = store.chunk_ids_with_suffixes(suffixes) if suffixes else None
        scored = cache.semantic_scored(
            query_vector, collections=collections, project=project, pool=pool, allowed=allowed
        )
        semantic_ids = [cid for cid, _ in scored]
        similarity_of = {cid: sim for cid, sim in scored}

    scores: dict[int, float] = {}
    sources: dict[int, set[str]] = {}
    for ranking, label in ((lexical_ids, "lexical"), (semantic_ids, "semantic")):
        for rank, cid in enumerate(ranking, start=1):
            sources.setdefault(cid, set()).add(label)
    if strategy in ("adaptive", "lexical_first"):
        exact = bool(lexical_ids) and (
            strategy == "lexical_first"
            or _covers(store, query, lexical_ids[0]) >= COVERAGE_MIN
        )
        if exact:
            # Exact phrase found: show only word matches, phrase-close matches first (BM25 order kept inside).
            rows_for_phrase = store.chunk_rows(lexical_ids)
            close = {cid for cid in lexical_ids
                     if cid in rows_for_phrase and _phrase_close(query, rows_for_phrase[cid].body)}
            order = [cid for cid in lexical_ids if cid in close] + [cid for cid in lexical_ids if cid not in close]
        elif strategy == "adaptive" and lexical_ids:
            order = list(dict.fromkeys(semantic_ids + lexical_ids))  # descriptive: meaning first
        else:
            order = list(dict.fromkeys(lexical_ids + semantic_ids))  # no word matches: meaning only
        for rank, cid in enumerate(order, start=1):
            scores[cid] = 1.0 / rank
        ranked = order
    elif strategy == "rrf":
        for ranking, weight in ((lexical_ids, lexical_weight), (semantic_ids, 1.0)):
            for rank, cid in enumerate(ranking, start=1):
                scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank)
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    if not ranked:
        return []

    rows = store.chunk_rows(ranked)
    ranked = [cid for cid in ranked if cid in rows]
    # Candidates per document over the whole ranking, so "+N more" is exact.
    in_document: Counter = Counter(rows[cid].path for cid in ranked)
    kept: list[int] = []
    kept_per_document: Counter = Counter()
    for cid in ranked:
        if len(kept) >= k:
            break
        path = rows[cid].path
        if per_document and kept_per_document[path] >= per_document:
            continue
        kept_per_document[path] += 1
        kept.append(cid)

    hits: list[Hit] = []
    for cid in kept:
        row = rows[cid]
        via = "both" if len(sources[cid]) == 2 else next(iter(sources[cid]))
        hits.append(Hit(
            chunk_id=cid, collection=row.collection, project=row.project, title=row.title,
            path=row.path, locator=row.locator, heading=row.heading,
            snippet=_snippet(row.body), occurred_at=row.occurred_at,
            score=round(scores[cid], 6), via=via,
            similarity=round(similarity_of[cid], 4) if cid in similarity_of else None,
            more_in_document=in_document[row.path] - kept_per_document[row.path],
        ))
    return hits
