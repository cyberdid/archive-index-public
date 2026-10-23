import random

import pytest

from paper.judge.agreement import cohen_kappa
from paper.queries.schema import Query


def _grades(seed, n_queries=30, per_query=8):
    rng = random.Random(seed)
    return {(f"q{q}", f"d{d}"): rng.choice((0, 0, 1, 2)) for q in range(n_queries) for d in range(per_query)}


def test_kappa_ci_point_estimate_is_cohen_kappa():
    from paper.judge.report import kappa_ci

    a = _grades(1)
    b = {k: (g if random.Random(str(k)).random() < 0.7 else 0) for k, g in a.items()}
    kappa, low, high = kappa_ci(a, b, n=300)
    assert kappa == pytest.approx(cohen_kappa(a, b))
    assert low < kappa < high


def test_kappa_ci_of_identical_judges_is_one():
    from paper.judge.report import kappa_ci

    a = _grades(2)
    assert kappa_ci(a, dict(a), n=200) == (1.0, 1.0, 1.0)


def test_known_answer_checks():
    from paper.judge.report import known_answer_checks

    queries = [Query("q1", "антена", "uk", "known_item", "gpt-6-luna", "radio/a.md"),
               Query("q2", "GX-1 прошивка", "uk", "no_answer", "gpt-6-luna")]
    grades = {("q1", "radio/a.md"): 2, ("q1", "radio/b.md"): 1, ("q2", "x.md"): 0, ("q2", "y.md"): 1}
    checks = known_answer_checks(grades, queries)
    assert checks["target"] == {"pairs": 1, "share_2": 1.0, "share_0": 0.0}
    assert checks["no_answer"] == {"pairs": 2, "share_0": 0.5, "share_2": 0.0}
    assert checks["mean_grade"] == pytest.approx(1.0)


def test_judge_report_covers_every_pair_of_judges():
    from paper.judge.report import judge_report

    judges = {"a": _grades(3), "b": _grades(4), "c": _grades(5)}
    queries = [Query(f"q{i}", "x", "uk", "topical", "gpt-6-luna") for i in range(30)]
    report = judge_report(judges, queries, n=100)
    assert [(p["a"], p["b"]) for p in report["pairs"]] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert all(p["n"] == 240 and p["low"] <= p["kappa"] <= p["high"] for p in report["pairs"])
    assert set(report["judges"]) == {"a", "b", "c"} and report["judges"]["a"]["graded"] == 240
    assert set(report["pairs"][0]["by_category"]) == {"topical"} and set(report["pairs"][0]["by_language"]) == {"uk"}
