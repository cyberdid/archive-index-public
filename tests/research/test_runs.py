from pathlib import Path

import numpy as np

from archive_index.indexer import run_index
from archive_index.search import Hit
from paper.queries.schema import Query
from paper.runs.systems import SYSTEMS, SystemConfig, run_system, to_document_run, write_trec_run

SPEC_SYSTEMS = {"bm25_plain", "bm25_lemma", "dense_eg2", "dense_e5", "rrf", "adaptive", "adaptive-phrase",
                "adaptive-ident", "adaptive-short", "dense_eg2_256"}


def _hit(cid, path, score=0.5):
    return Hit(chunk_id=cid, collection="wiki", project="p", title="t", path=path, locator="s", heading="",
               snippet="", occurred_at="", score=score, via="lexical")


def test_registry_has_all_spec_systems():
    assert set(SYSTEMS) == SPEC_SYSTEMS
    assert SYSTEMS["bm25_plain"] == SystemConfig("plain.sqlite3", "lexical", "adaptive", False, True, True, "none", None)
    assert SYSTEMS["dense_eg2_256"].embed_dim == 256 and SYSTEMS["rrf"].strategy == "rrf"
    assert SYSTEMS["adaptive-phrase"].phrase_rerank is False and SYSTEMS["adaptive-ident"].identifiers is False
    assert SYSTEMS["adaptive-short"].index_db == "noshort.sqlite3" and SYSTEMS["dense_e5"].embed_backend == "e5"


def test_chunks_collapse_to_documents_in_rank_order(tmp_path):
    root = tmp_path / "ua-par"
    hits = [_hit(1, str(root / "a/notes/x.md")), _hit(2, str(root / "b/docs/y.pdf")), _hit(3, str(root / "a/notes/x.md")),
            _hit(4, "/elsewhere/z.md"), _hit(5, str(root / "c/z.md"))]
    run = to_document_run(hits, root)
    assert [doc for doc, _ in run] == ["a/notes/x.md", "b/docs/y.pdf", "c/z.md"]
    scores = [s for _, s in run]
    assert scores == sorted(scores, reverse=True)


def test_trec_run_format(tmp_path):
    path = tmp_path / "r.trec"
    write_trec_run(path, "adaptive", [("q001", [("a/x.md", 3.0), ("b/y.pdf", 2.0)])])
    assert path.read_text(encoding="utf-8").splitlines() == ["q001 Q0 a/x.md 1 3.000000 adaptive",
                                                             "q001 Q0 b/y.pdf 2 2.000000 adaptive"]


class _Fake:
    def documents(self, texts, titles=None):
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8) if texts else np.zeros((0, 0), np.float32)

    def query(self, text, prompt="SearchQuery"):
        return np.ones(8, dtype=np.float32) / np.sqrt(8)


def test_every_run_uses_corpus_document_ids(tmp_path):
    from archive_index.store import Store

    root = tmp_path / "ua-par"
    (root / "radio" / "notes").mkdir(parents=True)
    (root / "radio" / "notes" / "uk-1.md").write_text("# Антена\n\nАнтена ретранслятора на даху лабораторії.\n", encoding="utf-8")
    (root / "radio" / "notes" / "uk-2.md").write_text("# Кабель\n\nКоаксіальний кабель до антени і приймача.\n", encoding="utf-8")
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    store = Store(index_dir / "full.sqlite3")
    run_index(store, _Fake(), projects_root=root, home=tmp_path / "h", collections=("wiki",))
    store.close()
    manifest_ids = {"radio/notes/uk-1.md", "radio/notes/uk-2.md"}
    queries = [Query("q001", "антена", "uk", "topical", "claude")]
    for name in ("bm25_lemma", "dense_eg2", "adaptive", "dense_eg2_256"):
        rows = run_system(SYSTEMS[name], queries, root, index_dir, embedders={"eg2": _Fake()}, k=10)
        docs = {doc for _, run in rows for doc, _ in run}
        assert docs and docs <= manifest_ids, name


def test_indexes_never_read_the_users_home(tmp_path, monkeypatch):
    """The benchmark indexes must contain the corpus only: no Claude or Codex sessions from the real home."""
    from archive_index.store import Store
    from paper.runs.run_all import INDEXES, build_indexes

    root = tmp_path / "ua-par"
    (root / "radio" / "notes").mkdir(parents=True)
    (root / "radio" / "notes" / "uk-1.md").write_text("# Антена\n\nАнтена ретранслятора на даху лабораторії.\n", encoding="utf-8")
    fake_home = tmp_path / "home"
    (fake_home / ".claude" / "projects" / "x").mkdir(parents=True)
    (fake_home / ".claude" / "projects" / "x" / "s.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    paths = build_indexes(root, tmp_path / "idx", embedders={"eg2": _Fake(), "e5": _Fake()})
    assert set(paths) == set(INDEXES)
    for path in paths.values():
        store = Store(path)
        collections = {r[0] for r in store.conn.execute("SELECT DISTINCT collection FROM documents")}
        store.close()
        assert collections and "sessions" not in collections and "run_logs" not in collections


def test_a_relative_corpus_path_gives_the_same_documents_and_runs(tmp_path, monkeypatch):
    """Indexes keep absolute paths; a relative --corpus once duplicated every document on a rebuild."""
    from archive_index.store import Store
    from paper.runs.run_all import build_indexes, run_all

    root = tmp_path / "ua-par"
    (root / "radio" / "notes").mkdir(parents=True)
    (root / "radio" / "notes" / "uk-1.md").write_text("# Антена\n\nАнтена ретранслятора на даху лабораторії.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    fakes = {"eg2": _Fake(), "e5": _Fake()}
    build_indexes(root, tmp_path / "idx", embedders=fakes)
    build_indexes(Path("ua-par"), Path("idx"), embedders=fakes)
    store = Store(tmp_path / "idx" / "full.sqlite3")
    paths = [r[0] for r in store.conn.execute("SELECT path FROM documents")]
    store.close()
    assert paths == [str(root / "radio" / "notes" / "uk-1.md")]
    queries = [Query("q001", "антена", "uk", "topical", "gpt-6-luna")]
    written = run_all(queries, Path("runs"), Path("ua-par"), Path("idx"), embedders=fakes, k=10)
    assert "radio/notes/uk-1.md" in written["bm25_lemma"].read_text(encoding="utf-8")


def test_text_only_control_never_returns_pictures(tmp_path):
    """dense_eg2_text searches the E5 candidate universe: the same index restricted to the text collections."""
    from PIL import Image

    from archive_index.store import Store
    from paper.runs.systems import CONTROLS

    class MediaFake(_Fake):
        def image(self, path):
            return np.ones(8, dtype=np.float32) / np.sqrt(8)

    root = tmp_path / "ua-par"
    (root / "radio" / "notes").mkdir(parents=True)
    (root / "radio" / "notes" / "uk-1.md").write_text("# Антена\n\nАнтена на даху.\n", encoding="utf-8")
    (root / "images").mkdir()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(root / "images" / "антена.png")
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    store = Store(index_dir / "full.sqlite3")
    run_index(store, MediaFake(), projects_root=root, home=tmp_path / "h", collections=("wiki", "images"))
    store.close()
    queries = [Query("q001", "антена", "uk", "topical", "gpt-6-luna")]
    full = {doc for _, run in run_system(SYSTEMS["dense_eg2"], queries, root, index_dir, {"eg2": _Fake()}) for doc, _ in run}
    text = {doc for _, run in run_system(CONTROLS["dense_eg2_text"], queries, root, index_dir, {"eg2": _Fake()})
            for doc, _ in run}
    assert full == {"radio/notes/uk-1.md", "images/антена.png"} and text == {"radio/notes/uk-1.md"}
    assert set(CONTROLS) == {"dense_eg2_text", "adaptive_text"} and not set(CONTROLS) & set(SYSTEMS)
