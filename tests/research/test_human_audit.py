import csv

import pytest


def _qrels(n_queries=40):
    grades = {}
    for q in range(n_queries):
        for d in range(6):
            grades[(f"q{q:03d}", f"doc{d}")] = (q + d) % 3
    return grades


def _snippets(grades):
    return {key: f"text of {key[1]} for {key[0]}" for key in grades}


def test_audit_sheet_is_blind_stratified_and_never_overwritten(tmp_path):
    from paper.judge.human_audit import prepare_audit

    grades = _qrels()
    path = tmp_path / "human-audit.tsv"
    record = prepare_audit(grades, _snippets(grades), {f"q{q:03d}": f"query {q}" for q in range(40)}, path,
                           per_grade=10)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert list(rows[0]) == ["item", "qid", "query", "doc_id", "snippet", "grade", "note"]
    assert len(rows) == 30 and all(r["grade"] == "" for r in rows)
    keys = [(r["qid"], r["doc_id"]) for r in rows]
    assert sorted(grades[k] for k in keys) == [0] * 10 + [1] * 10 + [2] * 10
    assert [grades[k] for k in keys] != sorted(grades[k] for k in keys)  # shuffled, so the order leaks nothing
    assert record["pairs"] == 30 and "grades" not in path.read_text(encoding="utf-8").split("\n")[0]
    with pytest.raises(FileExistsError):
        prepare_audit(grades, _snippets(grades), {}, path, per_grade=10)


def test_audit_comparison_needs_a_complete_sheet_and_reports_kappa_per_judge(tmp_path):
    from paper.judge.human_audit import compare_audit, prepare_audit

    grades = _qrels()
    path = tmp_path / "human-audit.tsv"
    prepare_audit(grades, _snippets(grades), {f"q{q:03d}": f"query {q}" for q in range(40)}, path, per_grade=5)
    with pytest.raises(ValueError, match="ungraded"):
        compare_audit(path, {"gemini": grades})
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        r["grade"] = str(grades[(r["qid"], r["doc_id"])])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    shifted = {k: min(2, g + 1) for k, g in grades.items()}
    out = compare_audit(path, {"gemini": grades, "lenient": shifted})
    assert out["pairs"] == 15 and out["judges"]["gemini"]["kappa"] == pytest.approx(1.0)
    assert out["judges"]["lenient"]["mean_judge_minus_human"] > 0
