"""Measure the actual archive vector-ranking path and audit exact-reference agreement.

The first scaling pass uses an optimized argpartition top-10. archive-index
instead gathers candidate rows, sorts all scores, and constructs 400 tuples.
This follow-up calls that actual source method on the same synthetic matrices.
It excludes embedding, SQLite cache-version checks, lexical search and collapse.
Strict-ID agreement can differ between exact methods because of tied scores or
floating-point arithmetic; a score-threshold diagnostic is reported separately.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from paper.eval.ann_scaling import DATA, K, _per_query_ms, exact_top_k, recall_at_k, scaled_collection

OUT = Path('paper/results/ann_reference_audit.json')
SOURCE = Path('paper/results/ann_scaling.json')
SCORE_TOLERANCE = 1e-6


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run() -> dict:
    import faiss
    from archive_index.search import VectorCache

    original = json.loads(SOURCE.read_text())
    if not original.get('complete'):
        raise ValueError('finish the first scaling pass before the reference audit')
    inputs = {name: digest(DATA / name) for name in ('base.npy', 'queries.npy')}
    if inputs != original['source_sha256']:
        raise ValueError('ANN vectors changed since the first pass')
    base, queries = np.load(DATA / 'base.npy'), np.load(DATA / 'queries.npy')
    result = {
        'schema': 1, 'complete': False, 'source_ann_sha256': digest(SOURCE),
        'source_sha256': inputs, 'search_source_sha256': digest(Path('src/archive_index/search.py')),
        'faiss': faiss.__version__, 'numpy': np.__version__, 'queries': len(queries),
        'faiss_threads': faiss.omp_get_max_threads(), 'candidate_pool': 400,
        'score_tolerance': SCORE_TOLERANCE, 'planned_sizes': original['planned_sizes'],
        'scope': __doc__.strip(), 'sizes': {},
    }
    if result['faiss'] != original['faiss'] or result['numpy'] != original['numpy']:
        raise ValueError('use the same numerical library versions as the first pass')
    if OUT.exists():
        previous = json.loads(OUT.read_text())
        for key in result.keys() - {'sizes', 'complete'}:
            if previous.get(key) != result[key]:
                raise ValueError(f'reference audit checkpoint differs: {key}')
        result['sizes'] = previous['sizes']

    class ExportedVectorCache(VectorCache):
        # Reuse the actual semantic_scored method. Only its database-version
        # poll is excluded, because the experiment already has in-memory arrays.
        def __init__(self, matrix):
            self.matrix = matrix
            self.ids = np.arange(len(matrix), dtype=np.int64)

        def ensure(self):
            pass

    for size in original['planned_sizes']:
        key = str(size)
        if key in result['sizes']:
            continue
        print(f'archive path / exact-reference audit: {size}', flush=True)
        matrix = scaled_collection(base, size, sigma=original['sigma'], seed=original['seed'])
        exact = np.vstack([exact_top_k(matrix, queries[i:i+4]) for i in range(0, len(queries), 4)])
        cache = ExportedVectorCache(matrix)
        observed = []

        def archive_query(query):
            ranked = cache.semantic_scored(query[0], collections=None, project=None, pool=400)
            observed.append([cid for cid, _ in ranked[:K]])

        archive_query(queries[:1])  # warm the candidate-gather allocation
        observed.clear()
        row = {'archive_numpy_exact_ms': _per_query_ms(archive_query, queries),
               'archive_strict_id_recall_vs_numpy': recall_at_k(exact, np.asarray(observed)),
               'load_average_before_flat': list(os.getloadavg())}
        del cache, observed
        flat = faiss.IndexFlatIP(matrix.shape[1])
        flat.add(matrix)
        flat_ids = np.vstack([flat.search(query[None, :], K)[1] for query in queries])
        row['flat_strict_id_recall_vs_numpy'] = recall_at_k(exact, flat_ids)
        thresholds = np.array([np.min(matrix[ids] @ query) for ids, query in zip(exact, queries)])
        scores = np.array([matrix[ids] @ query for ids, query in zip(flat_ids, queries)])
        row['flat_results_above_reference_kth_score_with_tolerance'] = float(np.mean(
            scores >= thresholds[:, None] - SCORE_TOLERANCE))
        row['measured_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        result['sizes'][key] = row
        result['complete'] = set(result['sizes']) == {str(n) for n in original['planned_sizes']}
        temporary = OUT.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(temporary, OUT)
        print(size, json.dumps(row), flush=True)
        del matrix, flat, exact, flat_ids, scores
    return result


if __name__ == '__main__':
    run()
