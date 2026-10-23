"""Bind central manuscript claims and the built PDFs to inspectable measured sources.

This is a machine consistency record. It never certifies human relevance,
source review, licensing, release or submission approval.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

from paper.release.compliance import check_pdf
from paper.report.validate import validate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('paper/manuscript')
    results = Path('paper/results')
    errors = validate(results, root / 'generated')
    if errors:
        raise ValueError('\n'.join(errors))
    generated = json.loads((root / 'generated/manifest.json').read_text())
    if generated['unavailable_planned_artifacts']:
        raise ValueError('scientific measurements remain incomplete')
    sources = {p.name: json.loads(p.read_text()) for p in results.glob('*.json')}
    bib = root / 'tex/refs.bib'
    source_review = json.loads((root / 'source_manifest.json').read_text())
    if source_review['bibliography_sha256'] != sha(bib):
        raise ValueError('bibliography changed after the recorded source review')
    keys = set(re.findall(r'^@\w+\{([^,]+),', bib.read_text(), re.M))
    if keys != {r['key'] for r in source_review['sources']}:
        raise ValueError('source review does not cover the current bibliography entries')
    claims = []

    def add(name, filename, pointer, scope, unit='score', display=None):
        value = sources[filename]
        for key in pointer.strip('/').split('/'):
            value = value[int(key)] if isinstance(value, list) else value[key]
        row = {'id': name, 'scope': scope, 'source_file': filename, 'json_pointer': pointer,
               'value': value, 'unit': unit, 'display': display or str(value), 'human_verified': False}
        claims.append(row)
        return value

    for system in ('dense_e5', 'dense_eg2', 'adaptive', 'bm25_lemma', 'bm25_plain'):
        filename = 'ua_par_gemini_reviewed.json'
        pointer = f'/systems/{system}/all/nDCG@10/mean'
        mean = sources[filename]['systems'][system]['all']['nDCG@10']['mean']
        add('primary-' + system, filename, pointer, '170 answerable original queries; reviewed Gemini', display=f'{mean:.3f}')
    for system in ('dense_eg2', 'adaptive', 'dense_e5'):
        filename = 'extension.json'
        pointer = f'/variants/gemini_initial/systems/{system}/all/nDCG@10/mean'
        mean = sources[filename]['variants']['gemini_initial']['systems'][system]['all']['nDCG@10']['mean']
        add('extension-' + system, filename, pointer, '50-record extension; initial Gemini', display=f'{mean:.3f}')
    for lang in ('en', 'uk'):
        text = (root / f'tex/{lang}/main.tex').read_text()
        abstract = text.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
        for claim in claims:
            display = claim['display'].replace('.', ',') if lang == 'uk' else claim['display']
            if display not in abstract:
                raise ValueError(f'{lang} abstract missing source-backed value for {claim["id"]}: {display}')
    for key in ('manifest_files', 'queries'):
        add('corpus-' + key, 'corpus_summary.json', '/' + key, 'collected corpus / original query population', 'count')
    for key in ('documents', 'chunks', 'vectors', 'text_collection_vectors', 'text_vectors', 'text_collection_image_vectors'):
        add('full-index-' + key, 'corpus_summary.json', '/indexes/full/' + key, 'full public index', 'count')
    add('pool-pairs', 'judges.json', '/judges/gemini/graded', 'initial original-query pool', 'count')
    add('extension-unique', 'extension.json', '/unique_wording_sensitivity/queries', 'deterministic distinct-wording sensitivity', 'count')
    for key in ('documents', 'chunks'):
        add('private-aggregate-' + key, 'field_aggregate.json', '/archive/' + key, 'private archive aggregate only', 'count')
    for key, filename, pointer in (
        ('q4-ordinary-delta', 'efficiency.json', '/quantization/retrieval/quality/gemini_reviewed/comparisons/gguf_Q4_K_M/nDCG@10/mean'),
        ('q4-condensed-delta', 'efficiency.json', '/quantization/retrieval/quality/gemini_reviewed/comparisons/gguf_Q4_K_M/nDCG@10_condensed/mean'),
        ('q4-unjudged', 'efficiency.json', '/quantization/retrieval/quality/gemini_reviewed/systems/gguf_Q4_K_M/unjudged@10'),
        ('ann-2m-optimized-ms', 'ann_scaling.json', '/sizes/2000000/numpy_exact_ms/p50'),
        ('ann-2m-archive-ms', 'ann_reference_audit.json', '/sizes/2000000/archive_numpy_exact_ms/p50'),
        ('ann-2m-hnsw-recall', 'ann_scaling.json', '/sizes/2000000/hnsw_ef256/recall@10')):
        unit = 'ms' if key.endswith('-ms') else ('fraction' if key.endswith(('recall', 'unjudged')) else 'nDCG difference')
        add(key, filename, pointer, 'GGUF coverage or synthetic vector component measurement', unit)
    with (root / 'claims.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(claims[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(claims)
    compliance = {lang: check_pdf(root / f'article-{lang}.pdf', root / f'tex/{lang}/main.log') for lang in ('en', 'uk')}
    unexpected = {lang: [error for error in rows if not error.startswith('placeholder left: [PENDING:')]
                  for lang, rows in compliance.items()}
    if any(unexpected.values()):
        raise ValueError(f'PDF formatting/reference problems: {unexpected}')
    watched = [bib, root / 'source_manifest.json', root / 'claims.csv', root / 'generated/manifest.json',
               *(root / f'article-{lang}.pdf' for lang in ('en', 'uk'))]
    record = {'schema': 1, 'status': 'machine consistency checked; author approval pending',
              'claims': len(claims), 'source_sha256': generated['source_sha256'],
              'artifact_sha256': {p.relative_to(root).as_posix(): sha(p) for p in watched},
              'pdf_compliance': compliance, 'human_verified': False, 'submission_ready': False,
              'limits': 'Abstract numbers and generated content are checked against source values. This does not prove every prose interpretation or human validity.'}
    (root / 'consistency_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(f'Bound {len(claims)} central claims, bibliography/source review, generated artifacts and both PDFs; human approval remains false.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
