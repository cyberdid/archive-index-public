"""Hand-checked snapshot and disagreement review fixtures; no live judge calls."""
import csv
import json
from pathlib import Path

import pytest


def write_sheet(path, rows):
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / 'snapshot'
    (root / 'runs').mkdir(parents=True)
    (root / 'queries.tsv').write_text(
        'qid\ttext\tlang\tcategory\tauthor\n'
        'q1\tquestion one\tuk\ttopical\thuman\n'
        'q2\tquestion two\tru\ttechnical\thuman\n'
        'q0\tunknown\tuk\tno_answer\thuman\n', encoding='utf-8')
    (root / 'manifest.jsonl').write_text(''.join(json.dumps({'doc_id': d}) + '\n' for d in 'abc'))
    for system, docs in [('adaptive', 'ab'), ('other', 'ba')]:
        (root / 'runs' / f'{system}.trec').write_text(
            f'q1 Q0 {docs[0]} 1 2 {system}\nq1 Q0 {docs[1]} 2 1 {system}\n'
            f'q2 Q0 c 1 2 {system}\n' + (f'q0 Q0 a 1 2 {system}\n' if system == 'adaptive' else ''))
    base = [('q1', 'question one', 'a', 'original a'), ('q1', 'question one', 'b', 'original b'),
            ('q2', 'question two', 'c', 'original c'), ('q0', 'unknown', 'a', 'original a')]
    for name, grades in [('gemini', [2, 0, 0, 0]), ('gemma12', [0, 2, 0, 0])]:
        write_sheet(root / f'{name}.tsv', [dict(zip(('qid', 'query', 'doc_id', 'snippet', 'grade'), (*row, grade)))
                                         for row, grade in zip(base, grades)])
    return root


def load(root):
    from paper.judge.finalize import load_snapshot
    return load_snapshot(root, expected_systems=('adaptive', 'other'))


def test_snapshot_validates_all_input_and_records_provenance(snapshot):
    data = load(snapshot)
    assert len(data.pool_keys) == 4
    assert data.provenance == load(snapshot).provenance
    assert data.provenance['seed'] == 2026
    assert data.provenance['query_population']['total'] == 3
    assert data.provenance['query_population']['answerable'] == 2
    assert data.provenance['empty_run_queries']['other'] == ['q0']
    assert len(data.provenance['input_sha256']['gemini.tsv']) == 64
    assert data.provenance['models']['gemini_initial'] == 'gemini-3.8-flash-medium'


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'duplicate'), ('invalid', 'grade'), ('boolean', 'grade'), ('empty', 'grade'),
    ('missing_pair', 'pool'), ('extra_pair', 'pool'), ('unknown_doc', 'unknown document'),
    ('unknown_query', 'unknown query'), ('different_query', 'query text'), ('different_snippet', 'snippet'),
    ('missing_run', 'system'), ('empty_run', 'empty run'), ('unknown_run_query', 'unknown query'),
    ('unknown_run_doc', 'unknown document'), ('missing_answerable', 'missing answerable'),
])
def test_snapshot_refuses_corrupt_or_incomplete_data(snapshot, mutation, match):
    path = snapshot / 'gemma12.tsv'
    rows = read_rows(path)
    if mutation == 'duplicate': rows.append(rows[0].copy())
    elif mutation == 'invalid': rows[0]['grade'] = '3'
    elif mutation == 'boolean': rows[0]['grade'] = 'True'
    elif mutation == 'empty': rows[0]['grade'] = ''
    elif mutation == 'missing_pair': rows.pop()
    elif mutation == 'extra_pair': rows.append({**rows[0], 'doc_id': 'c'})
    elif mutation == 'unknown_doc': rows[0]['doc_id'] = 'missing'
    elif mutation == 'unknown_query': rows[0]['qid'] = 'missing'
    elif mutation == 'different_query': rows[0]['query'] = 'altered'
    elif mutation == 'different_snippet': rows[0]['snippet'] = 'altered'
    elif mutation == 'missing_run': (snapshot / 'runs/other.trec').unlink()
    elif mutation == 'empty_run': (snapshot / 'runs/other.trec').write_text('')
    elif mutation == 'unknown_run_query':
        with (snapshot / 'runs/other.trec').open('a') as fh: fh.write('bad Q0 a 1 1 other\n')
    elif mutation == 'unknown_run_doc':
        with (snapshot / 'runs/other.trec').open('a') as fh: fh.write('q0 Q0 missing 1 1 other\n')
    elif mutation == 'missing_answerable':
        run = snapshot / 'runs/other.trec'
        run.write_text('\n'.join(l for l in run.read_text().splitlines() if not l.startswith('q2 ')) + '\n')
    write_sheet(path, rows)
    with pytest.raises(ValueError, match=match): load(snapshot)


def test_prepare_preserves_originals_and_only_disagreements(snapshot, tmp_path):
    from paper.judge.finalize import prepare_review
    data = load(snapshot)
    original = {p: p.read_bytes() for p in snapshot.rglob('*') if p.is_file()}
    review = tmp_path / 'review.tsv'
    prepare_review(data, review)
    rows = read_rows(review)
    assert {(r['qid'], r['doc_id']) for r in rows} == {('q1', 'a'), ('q1', 'b')}
    assert {r['query'] for r in rows} == {'question one'}
    assert {r['snippet'] for r in rows} == {'original a', 'original b'}
    assert {r['judge_a'] for r in rows} == {'gemini', 'gemma12'}
    assert all(r['grade'] == '' for r in rows)
    for r in rows:
        assert r['grade_a'] == r[f"{r['judge_a']}_grade"]
        assert r['grade_b'] == r[f"{r['judge_b']}_grade"]
    copy = tmp_path / 'copy.tsv'
    prepare_review(data, copy)
    assert copy.read_bytes() == review.read_bytes()
    assert {p: p.read_bytes() for p in original} == original
    rows[0]['grade'] = '2'
    write_sheet(review, rows)
    with pytest.raises(FileExistsError): prepare_review(data, review)
    assert read_rows(review)[0]['grade'] == '2'


def complete_review(data, path):
    from paper.judge.finalize import prepare_review
    prepare_review(data, path)
    rows = read_rows(path)
    for r in rows: r['grade'] = '0' if r['doc_id'] == 'a' else '2'
    write_sheet(path, rows)


def test_merge_only_overrides_disagreements_and_keeps_all_queries(snapshot, tmp_path):
    from paper.judge.finalize import merge_review
    from paper.eval.metrics import read_qrels
    data = load(snapshot)
    review = tmp_path / 'review.tsv'
    complete_review(data, review)
    variants = merge_review(data, review, tmp_path / 'out')
    assert variants['gemini_reviewed'] == {('q1', 'a'): 0, ('q1', 'b'): 2, ('q2', 'c'): 0, ('q0', 'a'): 0}
    assert variants['gemini_initial'][('q1', 'a')] == 2
    assert read_qrels(tmp_path / 'out/qrels.txt') == {'q1': {'a': 0, 'b': 2}, 'q2': {'c': 0}, 'q0': {'a': 0}}
    assert (tmp_path / 'out/qrels.txt').read_bytes() == (tmp_path / 'out/qrels-gemini-reviewed.txt').read_bytes()
    assert json.loads((tmp_path / 'out/provenance.json').read_text())['review_sha256']


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'empty', 'invalid', 'snippet', 'initial_grade'])
def test_merge_refuses_inexact_or_tampered_review_before_writing(snapshot, tmp_path, mutation):
    from paper.judge.finalize import merge_review
    data = load(snapshot)
    review = tmp_path / 'review.tsv'
    complete_review(data, review)
    rows = read_rows(review)
    if mutation == 'missing': rows.pop()
    elif mutation == 'extra': rows.append({**rows[0], 'qid': 'q2', 'query': 'question two', 'doc_id': 'c'})
    elif mutation == 'duplicate': rows.append(rows[0].copy())
    elif mutation == 'empty': rows[0]['grade'] = ''
    elif mutation == 'invalid': rows[0]['grade'] = '-1'
    elif mutation == 'snippet': rows[0]['snippet'] = 'tampered'
    elif mutation == 'initial_grade': rows[0]['gemini_grade'] = '1'
    write_sheet(review, rows)
    with pytest.raises(ValueError): merge_review(data, review, tmp_path / 'out')
    assert not (tmp_path / 'out/qrels.txt').exists()


def test_review_uses_same_rubric_neutral_grades_and_resumes_with_logs(snapshot, tmp_path):
    from paper.judge.finalize import judge_review, prepare_review
    from paper.judge.llm_judge import GUIDELINE
    data = load(snapshot)
    review = tmp_path / 'review.tsv'
    prepare_review(data, review)
    prompts = []
    def fake_agy(pairs, **kwargs):
        prompt = kwargs['prompt_builder'](pairs)
        prompts.append(prompt)
        assert GUIDELINE.splitlines()[0] in prompt
        assert 'A: ' in prompt and 'B: ' in prompt
        assert 'original ' in prompt
        assert 'Gemini' in prompt  # explicitly discloses within-model review
        assert 'gemma12' not in prompt
        return {pid: 1 for pid, _, _ in pairs}, {'status': 'SUCCESS', 'model': 'test-model'}
    log = tmp_path / 'calls.jsonl'
    first = judge_review(data, review, log, batch_size=1, judge_batch=fake_agy)
    second = judge_review(data, review, log, batch_size=1, judge_batch=fake_agy)
    assert first['graded'] == 2 and second['already'] == 2
    assert len(prompts) == 2
    records = [json.loads(l) for l in log.read_text().splitlines()]
    assert all(r['review_kind'] == 'within-model Gemini review' for r in records)
    assert all(r['provenance']['input_sha256'] == data.provenance['input_sha256'] for r in records)


def test_agy_accepts_custom_prompt_builder_and_rejects_boolean_grades():
    from paper.judge.llm_judge import judge_batch_with_agy, parse_batch
    def runner(argv, **kwargs):
        assert argv[argv.index('-p') + 1] == 'custom review'
        return type('Done', (), {'returncode': 0, 'stderr': '', 'stdout': json.dumps({
            'status': 'SUCCESS', 'response': '{"grades":[{"id":"1","grade":2}]}'})})()
    grades, _ = judge_batch_with_agy([('1', 'q', 's')], runner=runner, prompt_builder=lambda pairs: 'custom review')
    assert grades == {'1': 2}
    assert parse_batch('{"grades":[{"id":"1","grade":true}]}', ['1']) == {}


def test_snapshot_manifest_hash_lock_refuses_changed_frozen_run(snapshot):
    from paper.judge.finalize import sha256
    files = [{'snapshot': p.relative_to(snapshot).as_posix(), 'sha256': sha256(p)}
             for p in snapshot.rglob('*') if p.is_file()]
    (snapshot / 'inputs.json').write_text(json.dumps({'files': files}))
    assert len(load(snapshot).pool_keys) == 4
    run = snapshot / 'runs/other.trec'
    run.write_text(run.read_text().replace(' 2 other', ' 3 other'))
    with pytest.raises(ValueError, match='hash lock'): load(snapshot)


def test_review_rejects_changed_initial_agreement_row_provenance(snapshot, tmp_path):
    from paper.judge.finalize import merge_review
    review = tmp_path / 'review.tsv'
    complete_review(load(snapshot), review)
    # Changing an AGREEMENT row leaves the disagreement keys and prepared review text unchanged.
    for name in ('gemini', 'gemma12'):
        path = snapshot / f'{name}.tsv'
        rows = read_rows(path)
        next(r for r in rows if r['qid'] == 'q2')['grade'] = '1'
        write_sheet(path, rows)
    with pytest.raises(ValueError, match='provenance'): merge_review(load(snapshot), review, tmp_path / 'out')


def test_review_requires_prepared_provenance(snapshot, tmp_path):
    from paper.judge.finalize import merge_review
    review = tmp_path / 'review.tsv'
    complete_review(load(snapshot), review)
    review.with_suffix('.prepared.json').unlink()
    with pytest.raises(ValueError, match='prepared'): merge_review(load(snapshot), review, tmp_path / 'out')


def test_batch_never_saves_grades_when_call_log_cannot_be_written(snapshot, tmp_path, monkeypatch):
    from paper.judge.sheets import judge_sheet_batched
    sheet = tmp_path / 'review.tsv'
    rows = read_rows(snapshot / 'gemini.tsv')
    for row in rows: row['grade'] = ''
    write_sheet(sheet, rows)
    before = sheet.read_bytes()
    log = tmp_path / 'calls.jsonl'
    original_open = Path.open
    def fail_log(path, *args, **kwargs):
        if path == log: raise OSError('log disk full')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_log)
    with pytest.raises(OSError, match='log disk full'):
        judge_sheet_batched(sheet, lambda pairs: ({pairs[0][0]: 2}, {'status': 'SUCCESS'}), log, batch_size=1)
    assert sheet.read_bytes() == before


def test_batch_logs_received_grades_before_a_sheet_save_failure(snapshot, tmp_path, monkeypatch):
    from paper.judge import sheets
    sheet = tmp_path / 'review.tsv'
    rows = read_rows(snapshot / 'gemini.tsv')
    for row in rows: row['grade'] = ''
    write_sheet(sheet, rows)
    log = tmp_path / 'calls.jsonl'
    def fail_write(*args): raise OSError('sheet disk full')
    monkeypatch.setattr(sheets, '_write', fail_write)
    with pytest.raises(OSError, match='sheet disk full'):
        sheets.judge_sheet_batched(sheet, lambda pairs: ({pairs[0][0]: 2}, {'status': 'SUCCESS'}), log, batch_size=1)
    record = json.loads(log.read_text())
    assert record['grades_by_pair'] == [{'qid': 'q1', 'doc_id': 'a', 'grade': 2}]
    assert all(r['grade'] == '' for r in read_rows(sheet))


def test_parse_batch_refuses_all_duplicate_id_entries():
    from paper.judge.llm_judge import parse_batch
    text = '{"grades":[{"id":"1","grade":0},{"id":"1","grade":2},{"id":"2","grade":1}]}'
    assert parse_batch(text, ['1', '2']) == {'2': 1}
