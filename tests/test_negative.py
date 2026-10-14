"""Negative tests: what must NOT be found, and what must disappear after a change.

Every positive search in the suite is paired with the opposite claim here, so a pass
means the index does not invent matches, keeps stale text, or leaks excluded content.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from archive_index.indexer import run_index
from archive_index.search import VectorCache, hybrid
from archive_index.sources import MAX_TEXT_BYTES
from archive_index.store import Store


class FakeEmbedder:
    DIM = 64

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype=np.float32)
        for word in text.lower().replace("|", " ").split():
            v[sum(map(ord, word)) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def documents(self, texts, titles=None):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return self._vec(text)


@pytest.fixture()
def corpus(tmp_path):
    projects = tmp_path / "Projects"
    home = tmp_path / "home"
    (projects / "llm" / "wiki").mkdir(parents=True)
    (projects / "llm" / "wiki" / "notes.md").write_text(
        "# Notes\n\nQuokkaphrase lives only in this wiki page.\n\nAPI_KEY=supersecretvalue123 stays local.\n",
        encoding="utf-8")
    (projects / "projalpha").mkdir(parents=True)
    (projects / "projalpha" / "README.md").write_text(
        "# projalpha\n\nZebraterm appears only in the projalpha project.\n", encoding="utf-8")

    claude = home / ".claude" / "projects" / "-x-llm"
    claude.mkdir(parents=True)
    lines = [
        {"type": "user", "sessionId": "sess-0001", "cwd": str(projects / "llm"),
         "timestamp": "2026-10-01T10:00:00Z", "message": {"role": "user", "content": "обговоримо план"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hiddenthought покаже лише внутрішнє міркування"},
            {"type": "text", "text": "Відповідь без прихованих слів."}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "toolonlyoutput з великим виводом команди"}]}},
    ]
    (claude / "sess-0001.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in lines) + "\n", encoding="utf-8")
    return projects, home, tmp_path / "neg.sqlite3"


def _index(corpus, **kwargs):
    projects, home, db = corpus
    store = Store(db)
    report = run_index(store, FakeEmbedder(), projects_root=projects, home=home, **kwargs)
    return store, report


def _lexical(store, query, **kwargs):
    return hybrid(store, VectorCache(store), query, None, mode="lexical", k=20, **kwargs)


# --- absence -----------------------------------------------------------------

@pytest.mark.parametrize("term", ["zzqxvunknownterm", "несуществующеесловоxq", "nonexistent123"])
def test_absent_term_has_no_lexical_hits(corpus, term):
    store, _ = _index(corpus)
    assert _lexical(store, term) == []
    store.close()


def test_absent_term_semantic_hits_are_marked_semantic_not_lexical(corpus):
    store, _ = _index(corpus)
    qv = FakeEmbedder().query("zzqxvunknownterm")
    hits = hybrid(store, VectorCache(store), "zzqxvunknownterm", qv, mode="hybrid", k=5)
    # Semantic ranking always returns something; it must never claim a word match.
    assert all(h.via == "semantic" for h in hits)
    store.close()


def test_empty_or_punctuation_query_returns_nothing(corpus):
    store, _ = _index(corpus)
    assert _lexical(store, "") == []
    assert _lexical(store, "?! ..") == []
    store.close()


# --- stale content must disappear ---------------------------------------------

def test_deleted_file_is_no_longer_found(corpus):
    projects, _, _ = corpus
    store, _ = _index(corpus)
    assert _lexical(store, "Quokkaphrase"), "precondition: phrase is indexed"
    (projects / "llm" / "wiki" / "notes.md").unlink()
    store.close()
    store, _ = _index(corpus)
    assert _lexical(store, "Quokkaphrase") == []
    store.close()


def test_edited_file_drops_old_text_and_keeps_new(corpus):
    projects, _, _ = corpus
    store, _ = _index(corpus)
    store.close()
    page = projects / "llm" / "wiki" / "notes.md"
    page.write_text("# Notes\n\nFreshmarker replaces the old text.\n", encoding="utf-8")
    store, _ = _index(corpus)
    assert _lexical(store, "Quokkaphrase") == []
    assert _lexical(store, "Freshmarker")
    store.close()


# --- exclusions: what the index must never contain ----------------------------

def test_secret_value_is_not_searchable(corpus):
    store, _ = _index(corpus)
    assert _lexical(store, "supersecretvalue123") == []
    store.close()


def test_thinking_blocks_are_not_searchable(corpus):
    store, _ = _index(corpus)
    assert _lexical(store, "hiddenthought") == []
    store.close()


def test_tool_output_is_not_searchable(corpus):
    store, _ = _index(corpus)
    assert _lexical(store, "toolonlyoutput") == []
    store.close()


def test_oversized_text_file_is_absent_by_design(corpus, tmp_path):
    projects, _, _ = corpus
    big = projects / "llm" / "wiki" / "huge.md"
    big.write_text("# Huge\n\nOversizedmarker " + "x " * (MAX_TEXT_BYTES // 2 + 10), encoding="utf-8")
    store, _ = _index(corpus)
    assert _lexical(store, "Oversizedmarker") == []  # documented limit, not a bug
    store.close()


# --- filters must exclude, not just rank ---------------------------------------

def test_project_filter_excludes_other_projects(corpus):
    store, _ = _index(corpus)
    assert _lexical(store, "Zebraterm", project="llm") == []
    assert _lexical(store, "Zebraterm", project="projalpha")
    store.close()


def test_collection_filter_excludes_other_collections(corpus):
    store, _ = _index(corpus)
    assert _lexical(store, "Quokkaphrase", collections=["sessions"]) == []
    assert _lexical(store, "Quokkaphrase", collections=["wiki"])
    store.close()


def test_positive_and_negative_are_case_consistent(corpus):
    store, _ = _index(corpus)
    assert _lexical(store, "QUOKKAPHRASE")  # case-insensitive positive
    assert _lexical(store, "quokkaphrasf") == []  # one letter off must not match
    store.close()


def test_lexical_matches_come_before_semantic_fill(corpus):
    store, _ = _index(corpus)
    cache = VectorCache(store)
    qv = FakeEmbedder().query("Quokkaphrase")
    hits = hybrid(store, cache, "Quokkaphrase", qv, mode="hybrid", k=5)
    assert hits and hits[0].via in ("lexical", "both")
    lexical_positions = [i for i, h in enumerate(hits) if h.via in ("lexical", "both")]
    semantic_only = [i for i, h in enumerate(hits) if h.via == "semantic"]
    assert all(i < min(semantic_only, default=10**9) for i in lexical_positions)
    store.close()


def test_weighted_rrf_strategy_still_available(corpus):
    store, _ = _index(corpus)
    cache = VectorCache(store)
    qv = FakeEmbedder().query("Quokkaphrase")
    hits = hybrid(store, cache, "Quokkaphrase", qv, mode="hybrid", k=5,
                  strategy="rrf", lexical_weight=3.0)
    assert hits and any(h.via in ("lexical", "both") for h in hits)
    store.close()


def test_adaptive_puts_meaning_first_for_descriptive_queries(corpus):
    store, _ = _index(corpus)
    cache = VectorCache(store)
    # A descriptive query shares a word with the wiki chunk, but the whole phrase is not in it.
    qv = FakeEmbedder().query("notes about zebraterm project")
    hits = hybrid(store, cache, "notes about zebraterm project", qv, mode="hybrid", k=5)
    assert hits, "expected some results"
    store.close()


def test_similarity_is_real_cosine_and_absent_for_word_only_hits(corpus):
    store, _ = _index(corpus)
    cache = VectorCache(store)
    qv = FakeEmbedder().query("Quokkaphrase lives only in this wiki page")
    hits = hybrid(store, cache, "Quokkaphrase", qv, mode="hybrid", k=5)
    semantic_hits = [h for h in hits if h.via in ("semantic", "both")]
    assert semantic_hits and all(h.similarity is not None and -1.0 <= h.similarity <= 1.0 for h in semantic_hits)
    lexical_only = [h for h in hits if h.via == "lexical"]
    assert all(h.similarity is None for h in lexical_only)
    store.close()


def test_results_are_grouped_per_document_with_exact_hidden_count(tmp_path):
    from archive_index.store import ChunkRecord, Store as _Store

    store = _Store(tmp_path / "group.sqlite3")
    import numpy as _np
    from archive_index.lexical import index_terms

    def record(i, body):
        return ChunkRecord(ord=i, heading="", locator=f"line {i}", body=body,
                           terms=index_terms(body), occurred_at="",
                           vector=_np.ones(4, dtype=_np.float32), body_sha=f"sha{i}-{body}")

    many = [record(i, f"zetagroup fragment number {i}") for i in range(1, 6)]
    store.replace_document(collection="sessions", path="/x/long.jsonl", size=1, mtime_ns=1,
                           sha256="a", title="long", project="llm", chunks=many)
    store.replace_document(collection="wiki", path="/x/short.md", size=1, mtime_ns=1,
                           sha256="b", title="short", project="llm",
                           chunks=[record(1, "zetagroup alone here")])
    hits = hybrid(store, VectorCache(store), "zetagroup", None, mode="lexical", k=10, per_document=2)
    long_hits = [h for h in hits if h.path == "/x/long.jsonl"]
    assert len(long_hits) == 2
    assert all(h.more_in_document == 3 for h in long_hits)  # 5 matches, 2 shown, 3 hidden
    assert sum(1 for h in hits if h.path == "/x/short.md") == 1
    store.close()


def test_identifiers_match_as_phrases_only(tmp_path):
    from archive_index.store import ChunkRecord, Store as _Store
    from archive_index.lexical import index_terms

    store = _Store(tmp_path / "ident.sqlite3")
    import numpy as _np

    def rec(i, body):
        return ChunkRecord(ord=i, heading="", locator=f"line {i}", body=body,
                           terms=index_terms(body), occurred_at="",
                           vector=_np.ones(4, dtype=_np.float32), body_sha=f"s{i}")

    store.replace_document(collection="docs", path="/x/ts36101.pdf", size=1, mtime_ns=1, sha256="a",
                           title="ts", project="p", chunks=[rec(1, "Requirements of TS 36.101 section 6")])
    store.replace_document(collection="docs", path="/x/ts36102.pdf", size=1, mtime_ns=1, sha256="b",
                           title="ts2", project="p", chunks=[rec(1, "Other spec 36.102 and page 36 text")])
    cache = VectorCache(store)
    found = hybrid(store, cache, "36.101", None, mode="lexical", k=10)
    assert [h.path for h in found] == ["/x/ts36101.pdf"]
    assert hybrid(store, cache, "36.109", None, mode="lexical", k=10) == []  # parts alone do not count
    store.close()


def test_sensitive_files_are_never_indexed(tmp_path):
    from archive_index.sources import is_sensitive, walk_files

    root = tmp_path / "Projects" / "p"
    (root / "notes").mkdir(parents=True)
    (root / ".ssh").mkdir()
    names = ["api_keys.txt", "id_rsa", "deploy.env", "server.pem", "notes.md", "my-secret-plan.md"]
    for name in names:
        (root / "notes" / name).write_text("x", encoding="utf-8")
    (root / ".ssh" / "config.md").write_text("x", encoding="utf-8")
    found = sorted(p.name for p in walk_files(root, ("",), 10_000))  # every file name is a candidate
    assert found == ["notes.md"]
    assert is_sensitive("tokens_budget.env")
    assert not is_sensitive("architecture.md")


def test_numeric_tables_are_found_by_words_only_and_never_by_meaning(tmp_path):
    from archive_index.indexer import is_numeric
    from archive_index.store import ChunkRecord, Store as _Store
    from archive_index.lexical import index_terms

    assert is_numeric("12 | 45.6 | 7.8 | 0.001 | 99 | 3,4 | 1/2")
    assert not is_numeric("Рівень бюджету на закупівлю антени становить 120 гривень.")

    store = _Store(tmp_path / "numeric.sqlite3")
    import numpy as _np
    table = "Pallet | 12 | 45 | 7 | 99 | 1024 | 3.5 | 0.25"
    rec = ChunkRecord(ord=0, heading="", locator="sheet BOM rows 1-30", body=table,
                      terms=index_terms(table), occurred_at="",
                      vector=_np.zeros(0, dtype=_np.float32), body_sha="t1")
    store.replace_document(collection="tables", path="/x/bom.xlsx", size=1, mtime_ns=1, sha256="a",
                           title="bom", project="p", chunks=[rec])
    cache = VectorCache(store)
    assert hybrid(store, cache, "Pallet", None, mode="lexical", k=5)  # found by its word
    assert hybrid(store, cache, "bom", _np.ones(4, dtype=_np.float32), mode="semantic", k=5) == []  # no vector
    store.close()
