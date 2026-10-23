import json

from paper.field.run_private import aggregate
from paper.release.check import check_release, deny_list


def test_released_files_contain_no_private_markers(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "a.md").write_text("UA-PAR corpus, CC BY-SA 4.0\n", encoding="utf-8")
    assert check_release(clean, deny=["/home/someone"]) == []
    leaky = tmp_path / "leaky"
    (leaky / "sub").mkdir(parents=True)
    (leaky / "sub" / "b.md").write_text("ok\nsee /home/someone/Projects/x\n", encoding="utf-8")
    (leaky / "c.bin").write_bytes(b"\x00\xff/home/someone\x00")
    problems = check_release(leaky, deny=["/home/someone"])
    assert any(p.startswith("sub/b.md:2:") for p in problems) and any(p.startswith("c.bin:") for p in problems)


def test_deny_list_takes_private_query_texts_and_projects(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"kind": "exact", "query": "синтетичний довгий запит для перевірки захисту",
                             "expect_phrase": "x", "source_path": "/home/someone/Projects/Secret/wiki/a.md"}) + "\n"
                 + json.dumps({"kind": "described", "query": "ok", "expect_project": "Hidden"}) + "\n",
                 encoding="utf-8")
    deny = deny_list([q], home="/home/someone")
    assert "/home/someone" in deny and "синтетичний довгий запит для перевірки захисту" in deny
    assert "Secret" in deny and "Hidden" in deny
    assert "ok" not in deny  # too short to be a marker; would flag every file


def test_field_aggregate_holds_numbers_only():
    results = [("main", "exact", 1), ("main", "exact", None), ("main", "exact", 4), ("main", "absent", True),
               ("main", "described", False), ("main", "described", True)]
    out = aggregate(results)
    assert out["main"]["exact"]["n"] == 3 and out["main"]["exact"]["hit@5"]["mean"] == 2 / 3
    assert out["main"]["absent"]["pass"]["mean"] == 1.0 and out["main"]["described"]["hit@5"]["mean"] == 0.5
    text = json.dumps(out)
    assert all(isinstance(v, (int, float)) for v in _leaves(out))
    assert "енерг" not in text


def _leaves(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _leaves(v)
        else:
            yield v
