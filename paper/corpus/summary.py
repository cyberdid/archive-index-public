"""Rebuild corpus populations from the manifest, query sheet and read-only indexes.

Raw chunks include lexical-only rows (dimension=0). They must not be reported as
embedding vectors, or as the population re-embedded by the GGUF experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

TEXT_COLLECTIONS = ('wiki', 'docs', 'tables', 'code')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(data: Path) -> dict:
    corpus = data / 'ua-par'
    manifest = corpus / 'manifest.jsonl'
    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    by_id = {entry['doc_id']: entry for entry in entries}
    if len(entries) != len(by_id):
        raise ValueError('duplicate manifest document id')
    with (data / 'queries.tsv').open() as stream:
        queries = list(csv.DictReader(stream, delimiter='\t'))
    out = {
        'source_sha256': {'manifest.jsonl': sha256(manifest), 'queries.tsv': sha256(data / 'queries.tsv')},
        'manifest_files': len(entries),
        'manifest_by_part': dict(Counter(entry['part'] for entry in entries)),
        'queries': len(queries),
        'queries_by_category': dict(Counter(q['category'] for q in queries)),
        'queries_by_language': dict(Counter(q['lang'] for q in queries)),
        'target_parts': dict(Counter(by_id[q['target']]['part'] for q in queries if q.get('target'))),
        'indexes': {},
        'caveats': [
            'The manifest describes collected files, not all of them are indexable by archive-index.',
            'The E5 baseline uses text-only input; its document universe differs from the multimodal EG2 indexes.',
            'All 170 authored answerable query targets are notes; findings do not establish retrieval quality for targets in every archive format.',
            'Raw chunks include lexical-only rows. Text-collection vector counts restrict collections to wiki, docs, tables and code; they include image-rendered PDF pages. Text-vector counts exclude image/audio input hashes.',
        ],
    }
    for name in ('e5', 'full', 'noshort', 'plain'):
        path = data / 'indexes' / f'{name}.sqlite3'
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            connection.execute('BEGIN')
            documents = connection.execute('SELECT path FROM documents').fetchall()
            ids = [Path(row[0]).resolve().relative_to(corpus.resolve()).as_posix() for row in documents]
            missing = set(ids) - by_id.keys()
            if missing:
                raise ValueError(f'{name}: indexed documents outside manifest: {len(missing)}')
            chunks, vectors = connection.execute('SELECT count(*),sum(dimension>0) FROM chunks').fetchone()
            text_vectors = connection.execute(
                'SELECT count(*) FROM chunks WHERE dimension>0 AND collection IN (?,?,?,?)',
                TEXT_COLLECTIONS).fetchone()[0]
            rendered_pages = connection.execute(
                "SELECT count(*) FROM chunks WHERE dimension>0 AND collection IN (?,?,?,?) AND body_sha LIKE 'img:%'",
                TEXT_COLLECTIONS).fetchone()[0]
            pure_text_vectors = connection.execute(
                "SELECT count(*) FROM chunks WHERE dimension>0 AND body_sha NOT LIKE 'img:%' AND body_sha NOT LIKE 'aud:%'").fetchone()[0]
            out['indexes'][name] = {
                'documents': len(documents), 'chunks': chunks, 'vectors': int(vectors or 0),
                'text_collection_vectors': text_vectors, 'text_vectors': pure_text_vectors,
                'text_collection_image_vectors': rendered_pages,
                'by_manifest_part': dict(Counter(by_id[doc]['part'] for doc in ids)),
                'vector_dimensions': {str(dim): count for dim, count in connection.execute(
                    'SELECT dimension,count(*) FROM chunks WHERE dimension>0 GROUP BY dimension')},
            }
        finally:
            connection.close()
        # SQLite sidecars can contain committed data not yet checkpointed. Keep
        # the WAL fingerprint when nonempty; the transient shared-memory file is
        # an implementation detail, not an input to the scientific population.
        out['source_sha256'][f'indexes/{path.name}'] = sha256(path)
        wal = Path(str(path) + '-wal')
        if wal.is_file() and wal.stat().st_size:
            out['source_sha256'][f'indexes/{wal.name}'] = sha256(wal)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('paper/data'))
    parser.add_argument('--out', type=Path, default=Path('paper/results/corpus_summary.json'))
    args = parser.parse_args()
    result = summarize(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['indexes'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
