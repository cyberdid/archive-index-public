"""Blind cross-family adjudication of Gemini/Gemma 12B disagreements (no live judge calls)."""
import json

import pytest

from test_judge_finalize import load, read_rows, snapshot, write_sheet  # noqa: F401  (fixture)


def test_blind_sheet_holds_only_disagreements_and_no_initial_grades(snapshot, tmp_path):
    from paper.judge.adjudicate import prepare_blind

    data = load(snapshot)
    sheet = tmp_path / 'claude-blind.tsv'
    record = prepare_blind(data, sheet)
    rows = read_rows(sheet)
    assert list(rows[0]) == ['qid', 'query', 'doc_id', 'snippet', 'grade']
    assert sorted((r['qid'], r['doc_id']) for r in rows) == [('q1', 'a'), ('q1', 'b')]
    assert all(r['grade'] == '' for r in rows) and record['disagreements'] == 2
    assert {r['snippet'] for r in rows} == {'original a', 'original b'}
    with pytest.raises(FileExistsError):
        prepare_blind(data, sheet)


def test_median_of_three_changes_only_disagreements(snapshot, tmp_path):
    from paper.judge.adjudicate import median_variant, prepare_blind

    data = load(snapshot)
    sheet = tmp_path / 'claude-blind.tsv'
    prepare_blind(data, sheet)
    rows = read_rows(sheet)
    for r in rows:
        r['grade'] = '1' if r['doc_id'] == 'a' else '2'  # q1/a: 2,0,1 -> 1; q1/b: 0,2,2 -> 2
    write_sheet(sheet, rows)
    assert median_variant(data, sheet) == {('q1', 'a'): 1, ('q1', 'b'): 2, ('q2', 'c'): 0, ('q0', 'a'): 0}


def test_median_variant_refuses_a_partial_or_foreign_sheet(snapshot, tmp_path):
    from paper.judge.adjudicate import median_variant, prepare_blind

    data = load(snapshot)
    sheet = tmp_path / 'claude-blind.tsv'
    prepare_blind(data, sheet)
    with pytest.raises(ValueError, match='ungraded'):
        median_variant(data, sheet)
    rows = read_rows(sheet)
    for r in rows:
        r['grade'] = '0'
    rows[0]['doc_id'] = 'c'
    write_sheet(sheet, rows)
    with pytest.raises(ValueError, match='disagreements'):
        median_variant(data, sheet)


def test_adjudication_summary_counts_which_initial_judge_the_third_sided_with(snapshot, tmp_path):
    from paper.judge.adjudicate import adjudication_summary, prepare_blind

    data = load(snapshot)
    sheet = tmp_path / 'claude-blind.tsv'
    prepare_blind(data, sheet)
    rows = read_rows(sheet)
    for r in rows:
        r['grade'] = '2' if r['doc_id'] == 'a' else '1'  # a: sides with Gemini (2); b: neither (Gemini 0, Gemma 2)
    write_sheet(sheet, rows)
    summary = adjudication_summary(data, sheet)
    assert summary['disagreements'] == 2
    assert summary['sided_with'] == {'gemini': 1, 'gemma12': 0, 'neither': 1}
    json.dumps(summary)
