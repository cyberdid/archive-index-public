"""Independent arithmetic checks for judge sensitivity and tie handling."""
import json
import math

import pytest

from test_judge_finalize import complete_review, load, snapshot


def test_hand_computed_rank_reversal_population_and_zero_relevance(snapshot, tmp_path):
    from paper.eval.judge_sensitivity import build_sensitivity, write_sensitivity
    data = load(snapshot)
    review = tmp_path / 'review.tsv'
    complete_review(data, review)
    reports, sensitivity = build_sensitivity(data, review)
    initial = reports['gemini_initial']
    reviewed = reports['gemini_reviewed']
    # q1: ideal at rank one -> 1; rank two -> 1/log2(3). q2 has no relevant docs -> 0.
    assert initial['systems']['adaptive']['all']['nDCG@10']['mean'] == pytest.approx(0.5)
    assert initial['systems']['other']['all']['nDCG@10']['mean'] == pytest.approx(1 / (2 * math.log2(3)))
    assert reviewed['systems']['adaptive']['all']['nDCG@10']['mean'] == pytest.approx(1 / (2 * math.log2(3)))
    assert reviewed['systems']['other']['all']['nDCG@10']['mean'] == pytest.approx(0.5)
    assert initial['queries'] == reviewed['queries'] == 2
    assert initial['population']['answerable_zero_relevant_query_ids'] == ['q2']
    assert initial['population']['answerable_zero_relevant_count'] == 1
    assert initial['provenance']['query_population']['total'] == 3
    assert set(initial['systems']['adaptive']['by_category']) == {'technical', 'topical'}
    assert initial['non_answer_diagnostics']['systems']['other']['empty_retrieval_count'] == 1
    changes = sensitivity['ordering_changes']['gemini_reviewed']['nDCG@10']
    assert changes == [{'systems': ['adaptive', 'other'], 'initial': 'adaptive > other', 'variant': 'other > adaptive',
                        'kind': 'reversal'}]
    assert sensitivity['rankings']['gemini_initial']['nDCG@10'] == [['adaptive'], ['other']]
    assert sensitivity['rankings']['gemini_reviewed']['nDCG@10'] == [['other'], ['adaptive']]
    assert sensitivity['provenance']['review_sha256']
    assert 'incomplete' in sensitivity['caveats']['R@100']
    assert 'same provider' in sensitivity['caveats']['judge_independence']
    assert set(sensitivity['within_judge_comparisons']) == set(reports)
    [comparison] = reviewed['comparisons']['nDCG@10']
    difference = 0.5 - 1 / (2 * math.log2(3))
    assert comparison['diff'] == pytest.approx(difference)
    assert comparison['ci_low'] == pytest.approx(0)
    assert comparison['ci_high'] == pytest.approx(2 * difference)
    out = tmp_path / 'results'
    write_sensitivity(reports, sensitivity, out)
    assert json.loads((out / 'ua_par_gemini_initial.json').read_text()) == initial
    markdown = (out / 'judge_sensitivity.md').read_text()
    assert 'adaptive' in markdown and 'other' in markdown and 'reversal' in markdown
    assert 'nDCG@10' in markdown and 'q2' in markdown and 'SHA-256' in markdown
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    again_reports, again = build_sensitivity(data, review)
    write_sensitivity(again_reports, again, out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_rank_ties_are_groups_and_changes_distinguish_tie_transitions():
    from paper.eval.judge_sensitivity import ordering_changes, ranking_groups
    assert ranking_groups({'c': .3, 'b': .5, 'a': .5}) == [['a', 'b'], ['c']]
    assert ordering_changes({'a': .5, 'b': .5}, {'a': .5, 'b': .4}) == [
        {'systems': ['a', 'b'], 'initial': 'a = b', 'variant': 'a > b', 'kind': 'tie broken'}]
    assert ordering_changes({'a': .5, 'b': .4}, {'a': .5, 'b': .5})[0]['kind'] == 'tie introduced'
    assert ordering_changes({'a': .5, 'b': .5}, {'a': .5, 'b': .5}) == []


def test_initial_sensitivity_never_promotes_initial_grades_to_canonical(snapshot, tmp_path):
    from paper.eval.judge_sensitivity import build_sensitivity, write_sensitivity
    reports, sensitivity = build_sensitivity(load(snapshot))
    assert set(reports) == {'gemini_initial', 'gemma12_initial'}
    assert sensitivity['review_status'] == 'not supplied; no canonical qrels produced'
    write_sensitivity(reports, sensitivity, tmp_path / 'out')
    assert not list(tmp_path.rglob('qrels*.txt'))


def test_pool_relative_recall_exposes_unjudged_tail(snapshot):
    from paper.eval.judge_sensitivity import pool_coverage
    data = load(snapshot)
    # The fixture's q1 pool has a,b; c is known but unjudged for q1.
    data.runs['other']['q1']['c'] = -1
    coverage = pool_coverage(data)
    assert coverage['other']['at_100']['retrieved_pairs'] == 4
    assert coverage['other']['at_100']['judged_pairs'] == 3
    assert coverage['other']['at_100']['judged_fraction'] == pytest.approx(.75)


def test_planned_rq1_contrast_has_hand_computed_paired_difference_and_ci():
    from paper.eval.judge_sensitivity import planned_rq1_contrast
    queries = ['q1', 'q2']
    scores = {'bm25_lemma': {'q1': {'nDCG@10': 1.0}, 'q2': {'nDCG@10': 0.5}},
              'bm25_plain': {'q1': {'nDCG@10': 0.0}, 'q2': {'nDCG@10': 0.5}}}
    result = planned_rq1_contrast(scores, queries)
    assert result['measure'] == 'nDCG@10'
    assert result['mean_difference'] == pytest.approx(.5)
    assert result['ci_low'] == 0 and result['ci_high'] == 1
    assert result['p'] == 1
    assert result['n'] == 2
    assert 'planned RQ1' in result['label']


def test_target_diagnostics_use_pool_denominator_and_record_excluded_targets(snapshot, tmp_path):
    from paper.eval.judge_sensitivity import build_sensitivity
    path = snapshot / 'queries.tsv'
    path.write_text(path.read_text().replace('category\tauthor\n', 'category\tauthor\ttarget\n')
                    .replace('topical\thuman\n', 'topical\thuman\ta\n')
                    .replace('technical\thuman\n', 'technical\thuman\tb\n')
                    .replace('no_answer\thuman\n', 'no_answer\thuman\t\n'))
    reports, _ = build_sensitivity(load(snapshot))
    diagnostic = reports['gemini_initial']['target_diagnostics']
    assert diagnostic['target_total'] == 2
    assert diagnostic['target_in_pool'] == 1 and diagnostic['target_outside_pool'] == 1
    assert diagnostic['target_grade_denominator'] == 1
    assert diagnostic['pooled_target_grade_counts'] == {'0': 0, '1': 0, '2': 1}
    assert diagnostic['pooled_target_grade_2_share'] == 1
