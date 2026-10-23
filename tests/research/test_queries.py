import numpy as np

from paper.queries.schema import CATEGORY_QUOTA, Query, load_queries, longest_shared_run, validate


def _q(i, category, lang="uk", text=None, target=""):
    return Query(qid=f"q{i:03d}", text=text or f"запит номер {i} про антени", lang=lang, category=category,
                 author="claude", target=target)


def _full_set():
    queries, i = [], 0
    for category, n in CATEGORY_QUOTA.items():
        for k in range(n):
            lang = "uk" if k % 3 != 2 else "ru"
            queries.append(_q(i, category, lang))
            i += 1
    return queries


def test_category_quotas():
    assert CATEGORY_QUOTA == {"known_item": 40, "topical": 60, "technical": 40, "cross_lingual": 30, "no_answer": 30}
    assert validate(_full_set(), manifest=[]) == []
    short = [q for q in _full_set() if not (q.category == "technical" and q.qid.endswith("0"))]
    assert any("technical" in p for p in validate(short, manifest=[]))


def test_language_shares_are_checked():
    only_en = [Query(q.qid, q.text, "en", q.category, q.author) for q in _full_set()]
    problems = validate(only_en, manifest=[])
    assert any("uk share" in p for p in problems) and any("ru share" in p for p in problems)


def test_topical_query_shares_at_most_five_token_ngram():
    assert longest_shared_run("як налаштувати точку доступу вдома", "Щоб налаштувати точку доступу вдома…") == 4
    assert longest_shared_run("антена для ретранслятора", "Ретранслятор потребує доброї антени") <= 1
    manifest = [{"doc_id": "radio/notes/uk-1.md", "part": "notes"}]
    queries = _full_set()
    queries[45] = _q(45, "topical", text="як налаштувати точку доступу вдома швидко і просто", target="radio/notes/uk-1.md")
    problems = validate(queries, manifest,
                        texts={"radio/notes/uk-1.md": "Щоб налаштувати точку доступу вдома швидко і просто…"})
    assert any("q045" in p and "copied" in p for p in problems)


def test_unknown_target_and_duplicate_ids_are_reported():
    queries = _full_set()
    queries[0] = _q(0, "known_item", target="missing/doc.md")
    queries[1] = Query(queries[2].qid, queries[1].text, "uk", "known_item", "claude")
    problems = validate(queries, manifest=[])
    assert any("missing/doc.md" in p for p in problems) and any("duplicate" in p for p in problems)


def test_no_answer_queries_have_no_word_match(tmp_path):
    from archive_index.lexical import index_terms
    from archive_index.store import ChunkRecord, Store

    store = Store(tmp_path / "n.sqlite3")
    body = "Антена ретранслятора працює на даху"
    store.replace_document(collection="wiki", path="/x/a.md", size=1, mtime_ns=1, sha256="a", title="a", project="p",
                           chunks=[ChunkRecord(ord=0, heading="", locator="s", body=body, terms=index_terms(body),
                                               occurred_at="", vector=np.ones(4, dtype=np.float32), body_sha="a")])
    queries = _full_set()
    queries[-1] = _q(len(queries) - 1, "no_answer", text="антена")
    problems = validate(queries, manifest=[], store=store)
    assert any("no_answer" in p and "антена" in p for p in problems)
    store.close()


def test_load_queries_reads_tsv(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text("qid\ttext\tlang\tcategory\tauthor\ttarget\nq001\tантена\tuk\ttopical\tclaude\t\n", encoding="utf-8")
    assert load_queries(path) == [Query("q001", "антена", "uk", "topical", "claude", "")]
