"""Switches used by the paper's baselines and ablations (plan Task 2). Defaults keep the system unchanged."""

import numpy as np

from archive_index.embed import E5Embedder, Embedder
from archive_index.indexer import run_index
from archive_index.lexical import index_terms, query_expression
from archive_index.search import VectorCache, hybrid
from archive_index.store import ChunkRecord, Store


def test_plain_terms_have_no_lemmas():
    assert index_terms("точку доступу", lemmatize=False).split() == ["точку", "доступу"]
    assert "точка" in index_terms("точку доступу").split()


def test_plain_query_has_no_lemmas():
    assert '"точка"' in query_expression("точку")
    assert '"точка"' not in query_expression("точку", lemmatize=False)


def test_identifier_rule_can_be_switched_off():
    assert " AND " in query_expression("36.101 антена")
    assert " AND " not in query_expression("36.101 антена", identifiers=False)


def _doc(store, path, body, rank_seed):
    vec = np.ones(4, dtype=np.float32)
    store.replace_document(collection="docs", path=path, size=1, mtime_ns=1, sha256=path, title="t", project="p",
                           chunks=[ChunkRecord(ord=0, heading="", locator="text", body=body,
                                               terms=index_terms(body), occurred_at="", vector=vec,
                                               body_sha=path + str(rank_seed))])


def test_phrase_rerank_off_keeps_bm25_order(tmp_path):
    store = Store(tmp_path / "p.sqlite3")
    # far apart but repeated: higher BM25; adjacent once inside a longer text: lower BM25
    _doc(store, "/far", "антена " * 6 + "слово " * 6 + "ретранслятор " * 6, 1)  # gap 6 > slack 3
    _doc(store, "/near", "інший текст " * 30 + "антена ретранслятор " + "ще текст " * 30, 2)
    cache = VectorCache(store)
    on = [h.path for h in hybrid(store, cache, "антена ретранслятор", None, mode="lexical", k=2, per_document=0)]
    off = [h.path for h in hybrid(store, cache, "антена ретранслятор", None, mode="lexical", k=2, per_document=0,
                                  phrase_rerank=False)]
    assert off == ["/far", "/near"]
    assert on == ["/near", "/far"]
    store.close()


class _Fake:
    def documents(self, texts, titles=None):
        return np.ones((len(texts), 4), dtype=np.float32) if texts else np.zeros((0, 0), np.float32)

    def query(self, text, prompt="SearchQuery"):
        return np.ones(4, dtype=np.float32)


def test_min_letters_zero_keeps_page_numbers(tmp_path):
    root = tmp_path / "Projects" / "p"
    root.mkdir(parents=True)
    (root / "note.md").write_text("# Номер\n\n3\n\n# Текст\n\nДовгий абзац про антену і ретранслятор у лабораторії.\n",
                                  encoding="utf-8")
    counts = {}
    for name, letters in (("default", 20), ("off", 0)):
        store = Store(tmp_path / f"{name}.sqlite3")
        run_index(store, _Fake(), projects_root=tmp_path / "Projects", home=tmp_path / "h", collections=("wiki",),
                  min_letters=letters)
        counts[name] = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        store.close()
    assert counts["off"] == counts["default"] + 1


def test_index_without_lemmas_stores_plain_terms(tmp_path):
    root = tmp_path / "Projects" / "p"
    root.mkdir(parents=True)
    (root / "note.md").write_text("# Мережа\n\nНалаштовуємо точку доступу в лабораторії сьогодні.\n", encoding="utf-8")
    store = Store(tmp_path / "plain.sqlite3")
    run_index(store, _Fake(), projects_root=tmp_path / "Projects", home=tmp_path / "h", collections=("wiki",),
              lemmatize=False)
    terms = " ".join(r[0] for r in store.conn.execute("SELECT terms FROM chunks_fts"))
    assert "точку" in terms.split() and "точка" not in terms.split()
    store.close()


def test_e5_prefixes():
    e = E5Embedder.__new__(E5Embedder)
    assert e.query_text("x") == "query: x" and e.passage_text("y") == "passage: y"


def test_truncated_dimension(monkeypatch):
    class Model:
        def encode(self, items, **kwargs):
            if isinstance(items, list):
                return np.ones((len(items), 768), dtype=np.float32) / np.sqrt(768)
            return np.arange(1, 769, dtype=np.float32)

    emb = Embedder(truncate_dim=256)
    monkeypatch.setattr(emb, "_load", lambda: Model())
    q = emb.query("x")
    assert q.shape == (256,) and abs(float(np.linalg.norm(q)) - 1.0) < 1e-5
    d = emb.documents(["a", "b"])
    assert d.shape == (2, 256) and np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
