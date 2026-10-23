import math

import pytest

from paper.eval.metrics import per_query
from paper.eval.stats import bootstrap_ci, compare_all, holm, randomization_test


def test_ndcg_hand_computed():
    qrels = {"q1": {"d1": 2, "d2": 1}, "q2": {"d4": 1}}
    run = {"q1": {"d2": 3.0, "d1": 2.0, "d3": 1.0}, "q2": {"d5": 2.0, "d4": 1.0}}
    scores = per_query(qrels, run, measures=("nDCG@10", "RR@10", "Success@1"))
    q1 = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert scores["q1"]["nDCG@10"] == pytest.approx(q1, abs=1e-4)
    assert scores["q2"]["nDCG@10"] == pytest.approx(1 / math.log2(3), abs=1e-4)
    assert scores["q2"]["RR@10"] == pytest.approx(0.5) and scores["q1"]["Success@1"] == 1.0


def test_query_without_any_hit_scores_zero():
    scores = per_query({"q1": {"d1": 1}}, {"q1": {"x": 1.0}}, measures=("nDCG@10",))
    assert scores["q1"]["nDCG@10"] == 0.0


def test_bootstrap_of_a_constant_difference():
    assert bootstrap_ci([0.1] * 30) == pytest.approx((0.1, 0.1, 0.1))


def test_randomization_of_identical_systems_is_one():
    x = [0.2, 0.5, 0.9, 0.1]
    assert randomization_test(x, x) == 1.0


def test_randomization_detects_a_clear_difference():
    assert randomization_test([1.0] * 40, [0.0] * 40) < 0.001


def test_holm():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_compare_all_against_reference():
    per_system = {"adaptive": {f"q{i}": 0.8 for i in range(30)}, "bm25": {f"q{i}": 0.5 for i in range(30)}}
    [c] = compare_all(per_system, reference="adaptive")
    assert c.system == "bm25" and c.diff == pytest.approx(-0.3) and c.p_holm < 0.001
    assert c.ci_low == pytest.approx(-0.3) and c.ci_high == pytest.approx(-0.3)


def test_report_groups_by_system_category_and_language(tmp_path):
    from paper.eval.report_main import build_report
    from paper.queries.schema import Query

    queries = [Query(f"q{i}", "x", "uk" if i % 2 else "ru", "topical" if i < 4 else "technical", "claude")
               for i in range(8)]
    qrels = {q.qid: {"good": 2} for q in queries}
    runs = {"adaptive": {q.qid: {"good": 2.0, "bad": 1.0} for q in queries},
            "bm25_plain": {q.qid: {"bad": 2.0, "good": 1.0} for q in queries}}
    report = build_report(queries, qrels, runs)
    assert report["systems"]["adaptive"]["all"]["nDCG@10"]["mean"] == pytest.approx(1.0)
    assert set(report["systems"]["adaptive"]["by_category"]) == {"topical", "technical"}
    assert set(report["systems"]["adaptive"]["by_language"]) == {"uk", "ru"}
    [cmp] = report["comparisons"]["nDCG@10"]
    assert cmp["system"] == "bm25_plain" and cmp["diff"] < 0
