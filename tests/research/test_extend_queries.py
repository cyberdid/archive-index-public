import pytest


def _manifest():
    entries = []
    for i in range(30):
        entries.append({"doc_id": f"radio/docs/uk-{i}.pdf", "part": "documents", "lang": "uk" if i % 3 else "ru"})
        entries.append({"doc_id": f"opendata/tables/t{i}.csv", "part": "tables", "lang": "uk"})
        entries.append({"doc_id": f"code-x/src/m{i}.py", "part": "code", "lang": "mixed"})
        entries.append({"doc_id": f"radio/notes/uk-{i}.md", "part": "notes", "lang": "uk"})
    return entries


def test_targets_are_sampled_per_part_and_never_notes():
    from paper.queries.extend import choose_targets

    texts = {e["doc_id"]: "x" * 400 for e in _manifest()}
    targets = choose_targets(_manifest(), texts, {"documents": 4, "tables": 3, "code": 3}, seed=1)
    assert [t["part"] for t in targets].count("documents") == 4 and len(targets) == 10
    assert not any(t["part"] == "notes" for t in targets)
    assert targets == choose_targets(_manifest(), texts, {"documents": 4, "tables": 3, "code": 3}, seed=1)


def test_targets_skip_documents_without_enough_text():
    from paper.queries.extend import choose_targets

    texts = {e["doc_id"]: ("x" * 400 if not e["doc_id"].endswith("0.csv") else "x") for e in _manifest()}
    targets = choose_targets(_manifest(), texts, {"tables": 30}, seed=1)
    assert all(not t["doc_id"].endswith("0.csv") for t in targets)


def test_prompt_states_the_rules_and_the_language():
    from paper.queries.extend import query_prompt

    prompt = query_prompt("Звіт про антени", "Текст документа про антени.", "ru", "known_item")
    assert "русском" in prompt or "російською" in prompt
    assert "Текст документа про антени." in prompt and "3" in prompt and "10" in prompt


def test_accept_rejects_copies_empty_and_overlong_answers():
    from paper.queries.extend import accept

    doc = "антена ретранслятора на даху лабораторії працює на частоті сто сорок шість мегагерц"
    assert accept("де стоїть антена ретранслятора", doc)
    assert not accept("антена ретранслятора на даху лабораторії працює", doc)  # 7 shared tokens in a row
    assert not accept("", doc)
    assert not accept(" ".join(["слово"] * 16), doc)


def test_new_queries_get_their_own_ids_and_author():
    from paper.queries.extend import to_queries

    rows = [{"doc_id": "radio/docs/uk-1.pdf", "part": "documents", "lang": "uk", "category": "known_item",
             "text": "звіт про антени"}]
    queries = to_queries(rows, start=201, author="qwen3.8-27b")
    assert queries[0].qid == "q201" and queries[0].author == "qwen3.8-27b" and queries[0].target == "radio/docs/uk-1.pdf"
