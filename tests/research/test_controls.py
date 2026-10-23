import pytest


def _run(docs):
    return {doc: float(len(docs) - i) for i, doc in enumerate(docs)}


def test_unjudged_share_counts_top10_documents_without_a_grade():
    from paper.eval.controls import unjudged_at

    qrels = {"q1": {"a": 2, "b": 0}, "q2": {"c": 1}}
    runs = {"q1": _run(["a", "x", "b", "y"]), "q2": _run(["c"])}
    assert unjudged_at(qrels, runs, k=10) == pytest.approx(2 / 5)


def test_condensed_run_drops_unjudged_documents_and_keeps_order():
    from paper.eval.controls import condensed

    qrels = {"q1": {"a": 2, "b": 0}}
    run = condensed(qrels, {"q1": _run(["x", "b", "y", "a"])})
    assert sorted(run["q1"], key=lambda d: -run["q1"][d]) == ["b", "a"]


def test_control_comparison_reports_both_encoders_on_one_population():
    from paper.eval.controls import compare_encoders

    qrels = {f"q{i}": {"a": 2, "b": 0, "c": 1} for i in range(6)}
    runs = {"dense_e5": {q: _run(["a", "c", "b"]) for q in qrels},
            "dense_eg2_text": {q: _run(["b", "c", "a"]) for q in qrels},
            "dense_eg2": {q: _run(["b", "a", "c"]) for q in qrels}}
    out = compare_encoders(qrels, runs, reference="dense_e5", n=500)
    assert out["queries"] == 6
    assert out["systems"]["dense_e5"]["nDCG@10"] > out["systems"]["dense_eg2_text"]["nDCG@10"]
    diff = out["comparisons"]["dense_eg2_text"]["nDCG@10"]
    assert diff["mean"] < 0 and diff["high"] < 0 and diff["p"] < 0.05
    assert set(out["systems"]["dense_eg2"]) >= {"nDCG@10", "nDCG@10_condensed", "unjudged@10"}
