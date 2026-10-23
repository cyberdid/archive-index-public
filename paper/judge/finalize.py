"""Validate frozen judging inputs and prepare, run, and merge a Gemini disagreement review.

Original sheets are read only. The review is a within-model sensitivity procedure, not an
independent third judge. Its A/B order is balanced, shuffled, and reproducible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from paper.eval.stats import N_RESAMPLES, SEED
from paper.judge.agreement import cohen_kappa, disagreements, merge_grades
from paper.judge.llm_judge import AGY_MODEL, batch_prompt, judge_batch_with_agy
from paper.judge.pool import build_pool, read_trec_run, write_qrels
from paper.judge.sheets import _read, judge_sheet_batched
from paper.queries.schema import load_queries

DEFAULT_SNAPSHOT = Path('paper/data/judgments/snapshot-2026-10-22')
DEFAULT_REVIEW = Path('paper/data/judgments/gemini-review.tsv')
MODELS = {'gemini_initial': AGY_MODEL, 'gemma12_initial': 'gemma4:12b',
          'gemini_reviewed': AGY_MODEL + ' (within-model disagreement review)',
          'gemma_initial': 'gemma4 E4B (ancillary)', 'gpt_initial': 'gpt-6-luna/xhigh (comparison only)'}
REVIEW_KIND = 'within-model Gemini review'
REVIEW_COLUMNS = ['qid', 'query', 'doc_id', 'snippet', 'grade', 'gemini_grade', 'gemma12_grade',
                  'judge_a', 'grade_a', 'judge_b', 'grade_b']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


def _grade(value, context: str, allow_empty: bool = False):
    if allow_empty and value == '':
        return None
    if value not in ('0', '1', '2'):
        raise ValueError(f'{context}: invalid or empty grade {value!r}; expected 0, 1, or 2')
    return int(value)


def _sheet(path: Path, allow_empty: bool = False) -> dict:
    header, rows = _read(path)
    required = {'qid', 'doc_id', 'query', 'snippet', 'grade'}
    if len(set(header)) != len(header) or not required <= set(header):
        raise ValueError(f'{path}: invalid sheet columns')
    out = {}
    for row in rows:
        key = (row['qid'], row['doc_id'])
        if key in out:
            raise ValueError(f'{path}: duplicate pair {key}')
        _grade(row['grade'], f'{path}: {key}', allow_empty)
        if not row['snippet'] or not row['query']:
            raise ValueError(f'{path}: empty query or snippet for {key}')
        out[key] = row
    return out


@dataclass
class Snapshot:
    root: Path
    queries: list
    runs: dict
    sheets: dict
    pool_keys: set
    provenance: dict

    def grades(self, name: str) -> dict:
        return {key: int(row['grade']) for key, row in self.sheets[name].items()}


def _verify_snapshot_lock(root: Path) -> None:
    lock = root / 'inputs.json'
    if not lock.exists():
        return
    try:
        entries = json.loads(lock.read_text(encoding='utf-8'))['files']
        if not isinstance(entries, list) or not entries:
            raise ValueError('empty files list')
        seen = set()
        for entry in entries:
            relative = entry['snapshot']
            path = root / relative
            if relative in seen or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('duplicate or escaping path')
            seen.add(relative)
            if not path.is_file() or sha256(path) != entry['sha256']:
                raise ValueError(f'{relative}: missing file or digest mismatch')
        required = {'queries.tsv', 'manifest.jsonl', 'gemini.tsv', 'gemma12.tsv'}
        required.update(p.relative_to(root).as_posix() for p in (root / 'runs').glob('*.trec'))
        required.update(f'{n}.tsv' for n in ('gemma', 'gpt') if (root / f'{n}.tsv').exists())
        if not required <= seen:
            raise ValueError(f'unlocked inputs: {sorted(required - seen)}')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ValueError(f'snapshot hash lock {lock}: {exc}') from exc


def load_snapshot(root: Path, expected_systems=None, allow_empty_queries=()) -> Snapshot:
    """Refuse partial pools/grades, unknown keys, changed snippets and missing system files.

    A TREC file has no row for an empty retrieval. Missing no-answer queries are recorded;
    missing answerable queries require an explicit ``system:qid`` declaration.
    """
    _verify_snapshot_lock(root)
    if expected_systems is None:
        from paper.runs.systems import SYSTEMS
        expected_systems = tuple(SYSTEMS)
    queries = load_queries(root / 'queries.tsv')
    qmap = {q.qid: q for q in queries}
    if not queries or len(qmap) != len(queries):
        raise ValueError('empty query list or duplicate query ID')
    docs = {json.loads(line)['doc_id'] for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()}
    paths = sorted((root / 'runs').glob('*.trec'))
    if {p.stem for p in paths} != set(expected_systems):
        raise ValueError(f'run system set mismatch: expected {sorted(expected_systems)}, got {[p.stem for p in paths]}')
    allowed = set(allow_empty_queries)
    used_allowed = set()
    runs, pool_runs, empty_queries = {}, {}, {}
    for path in paths:
        seen, ranks = set(), {}
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                qid, _, doc, rank, score, _ = line.split()
                rank, score = int(rank), float(score)
            except ValueError as exc:
                raise ValueError(f'{path}:{lineno}: invalid TREC row') from exc
            if qid not in qmap:
                raise ValueError(f'{path}: unknown query {qid}')
            if doc not in docs:
                raise ValueError(f'{path}: unknown document {doc}')
            if (qid, doc) in seen:
                raise ValueError(f'{path}: duplicate pair {(qid, doc)}')
            if not math.isfinite(score) or rank != len(ranks.setdefault(qid, [])) + 1:
                raise ValueError(f'{path}: invalid score or nonconsecutive rank for {qid}')
            ranks[qid].append(rank)
            seen.add((qid, doc))
        if not seen:
            raise ValueError(f'{path}: empty run')
        empty_queries[path.stem] = sorted(set(qmap) - set(ranks))
        for qid in empty_queries[path.stem]:
            declaration = f'{path.stem}:{qid}'
            if qmap[qid].category != 'no_answer' and declaration not in allowed:
                raise ValueError(f'{path}: missing answerable query {qid}; declare intentional empty retrieval explicitly')
            if declaration in allowed:
                used_allowed.add(declaration)
        pool_runs[path.stem] = read_trec_run(path)
        runs[path.stem] = {qid: dict(rows) for qid, rows in pool_runs[path.stem]}
    if allowed != used_allowed:
        raise ValueError(f'unused/unknown empty-query declarations: {sorted(allowed - used_allowed)}')
    pool = build_pool(pool_runs, depth=10)
    pool_keys = {(qid, doc) for qid, ds in pool.items() for doc in ds}
    if set(pool) != set(qmap):
        raise ValueError(f'pool missing query rows: {sorted(set(qmap) - set(pool))}')
    names = ['gemini', 'gemma12'] + [name for name in ('gemma', 'gpt') if (root / f'{name}.tsv').exists()]
    sheets = {name: _sheet(root / f'{name}.tsv') for name in names}
    for name, rows in sheets.items():
        for (qid, doc), row in rows.items():
            if qid not in qmap:
                raise ValueError(f'{name}: unknown query {qid}')
            if doc not in docs:
                raise ValueError(f'{name}: unknown document {doc}')
            if row['query'] != qmap[qid].text:
                raise ValueError(f'{name}: query text differs from query list for {qid}')
        if set(rows) != pool_keys:
            raise ValueError(f'{name}: sheet does not equal depth-10 union pool: '
                             f'{len(pool_keys - set(rows))} missing, {len(set(rows) - pool_keys)} extra')
        for key, row in rows.items():
            if row['snippet'] != sheets['gemini'][key]['snippet']:
                raise ValueError(f'{name}: snippet differs from Gemini for {key}')
    inputs = [root / 'queries.tsv', root / 'manifest.jsonl', *paths, *(root / f'{n}.tsv' for n in names)]
    population = {'total': len(queries), 'answerable': sum(q.category != 'no_answer' for q in queries),
                  'no_answer': sum(q.category == 'no_answer' for q in queries),
                  'metric_query_ids': sorted(q.qid for q in queries if q.category != 'no_answer'),
                  'no_answer_query_ids': sorted(q.qid for q in queries if q.category == 'no_answer')}
    provenance = {'input_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(inputs)},
                  'models': {f'{n}_initial': MODELS[f'{n}_initial'] for n in names},
                  'query_population': population, 'seed': SEED, 'bootstrap_resamples': N_RESAMPLES,
                  'randomization_resamples': N_RESAMPLES, 'pool_depth': 10, 'pool_pairs': len(pool_keys),
                  'expected_systems': sorted(expected_systems), 'empty_run_queries': empty_queries,
                  'declared_empty_answerable_queries': sorted(allowed)}
    return Snapshot(root, queries, runs, sheets, pool_keys, provenance)


def _review_rows(data: Snapshot) -> list[dict]:
    a, b = data.grades('gemini'), data.grades('gemma12')
    keys = disagreements(a, b)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    rows = []
    for index, key in enumerate(keys):
        first, second = ('gemini', 'gemma12') if index % 2 == 0 else ('gemma12', 'gemini')
        initial = {'gemini': a[key], 'gemma12': b[key]}
        rows.append({**data.sheets['gemini'][key], 'grade': '', 'gemini_grade': str(a[key]),
                     'gemma12_grade': str(b[key]), 'judge_a': first, 'grade_a': str(initial[first]),
                     'judge_b': second, 'grade_b': str(initial[second])})
    return rows


def prepare_review(data: Snapshot, path: Path) -> dict:
    """Exclusive creation: an existing review (graded or empty) is never overwritten."""
    rows = _review_rows(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    record = {'review_kind': REVIEW_KIND, 'disagreements': len(rows), 'seed': SEED,
              'review_prepared_sha256': sha256(path), 'provenance': data.provenance}
    write_json(path.with_suffix('.prepared.json'), record)
    return record


def validate_review(data: Snapshot, path: Path, allow_empty=False) -> dict:
    prepared_path = path.with_suffix('.prepared.json')
    try:
        prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'review requires readable prepared provenance: {prepared_path}') from exc
    if prepared.get('provenance') != data.provenance:
        raise ValueError('review prepared provenance differs from the current snapshot')
    actual = _sheet(path, allow_empty=allow_empty)
    expected = {(r['qid'], r['doc_id']): r for r in _review_rows(data)}
    if set(actual) != set(expected):
        raise ValueError('review keys must equal exactly the initial disagreements')
    for key, row in actual.items():
        for column in REVIEW_COLUMNS:
            if column != 'grade' and row.get(column) != expected[key][column]:
                raise ValueError(f'review changed original {column} for {key}')
    ungraded = io.StringIO(newline='')
    writer = csv.DictWriter(ungraded, fieldnames=REVIEW_COLUMNS, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows({**row, 'grade': ''} for row in actual.values())
    digest = hashlib.sha256(ungraded.getvalue().encode('utf-8')).hexdigest()
    if digest != prepared.get('review_prepared_sha256'):
        raise ValueError('review differs from prepared hash after clearing grades')
    return actual


def judge_review(data: Snapshot, path: Path, log_path: Path, batch_size: int = 50,
                 judge_batch=None) -> dict:
    """Run/resume Gemini review using the existing batch executor, schema and call logger."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    validate_review(data, path, allow_empty=True)
    _, rows = _read(path)
    labels = {str(i): (r['grade_a'], r['grade_b']) for i, r in enumerate(rows)}

    def prompt_builder(pairs):
        # Only neutral grade labels enter the prompt; original snippet bytes stay in the TSV.
        annotated = [(pid, query, f'{snippet}\n\nПочаткові незалежні оцінки — A: {labels[pid][0]}; B: {labels[pid][1]}.')
                     for pid, query, snippet in pairs]
        return ('Це повторний розгляд розбіжностей тією самою моделлю Gemini (within-model Gemini review). '
                'Оціни документ заново за тією самою шкалою. Обидві початкові оцінки мають однакову вагу; '
                'обери власну оцінку 0, 1 або 2.\n\n' + batch_prompt(annotated))

    def review_batch(pairs):
        grades, record = (judge_batch or judge_batch_with_agy)(pairs, prompt_builder=prompt_builder)
        return grades, {**record, 'review_kind': REVIEW_KIND, 'provenance': data.provenance,
                        'neutral_initial_grades': {pid: list(labels[pid]) for pid, _, _ in pairs}}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    return judge_sheet_batched(path, review_batch, log_path, batch_size=batch_size)


def merged_variants(data: Snapshot, review: Path | None = None) -> dict:
    variants = {f'{name}_initial': data.grades(name) for name in data.sheets}
    if review is not None:
        rows = validate_review(data, review)
        third = {key: int(row['grade']) for key, row in rows.items()}
        variants['gemini_reviewed'] = merge_grades(variants['gemini_initial'], variants['gemma12_initial'], third)
    return variants


def merge_review(data: Snapshot, review: Path, out: Path) -> dict:
    variants = merged_variants(data, review)  # validate everything before creating any qrels
    out.mkdir(parents=True, exist_ok=True)
    for name, filename in [('gemini_initial', 'qrels-gemini.txt'), ('gemma12_initial', 'qrels-gemma12.txt'),
                           ('gemini_reviewed', 'qrels-gemini-reviewed.txt'), ('gemini_reviewed', 'qrels.txt')]:
        target = out / filename
        if target.resolve().is_relative_to(data.root.resolve()):
            raise ValueError('qrels output must be outside the frozen snapshot')
        write_qrels(target, variants[name])
    write_json(out / 'provenance.json', {**data.provenance, 'review_sha256': sha256(review),
               'review_kind': REVIEW_KIND, 'review_model': AGY_MODEL,
               'reviewed_disagreements': len(disagreements(data.grades('gemini'), data.grades('gemma12')))})
    return variants


def add_snapshot_arguments(parser):
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--allow-empty-query', action='append', default=[], metavar='SYSTEM:QID',
                        help='explicitly certify an intentional empty retrieval for an answerable query')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'prepare', 'judge', 'merge'):
        cmd = sub.add_parser(command)
        add_snapshot_arguments(cmd)
        if command != 'validate': cmd.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
        if command == 'judge':
            cmd.add_argument('--batch', type=int, default=50)
            cmd.add_argument('--log', type=Path)
        if command == 'merge': cmd.add_argument('--out', type=Path, default=Path('paper/data'))
    args = parser.parse_args()
    data = load_snapshot(args.snapshot, allow_empty_queries=args.allow_empty_query)
    if args.command == 'validate':
        result = {'provenance': data.provenance,
                  'disagreements': len(disagreements(data.grades('gemini'), data.grades('gemma12'))),
                  'weighted_kappa': cohen_kappa(data.grades('gemini'), data.grades('gemma12'))}
    elif args.command == 'prepare': result = prepare_review(data, args.review)
    elif args.command == 'judge':
        result = judge_review(data, args.review, args.log or args.review.with_suffix('.calls.jsonl'), args.batch)
    else:
        variants = merge_review(data, args.review, args.out)
        result = {'out': str(args.out), 'queries': len(data.queries), 'pairs': len(variants['gemini_reviewed'])}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
