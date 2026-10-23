from pathlib import Path

import numpy as np

from archive_index.indexer import run_index
from archive_index.search import VectorCache, hybrid
from archive_index.store import Store


class FakeEmbedder:
    def documents(self, texts, titles=None):
        return np.ones((len(texts), 4), dtype=np.float32) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return np.ones(4, dtype=np.float32)


def test_code_is_opt_in_per_project_and_skips_vendor_folders(tmp_path):
    root = tmp_path / "Projects"
    (root / "llm" / "app").mkdir(parents=True)
    (root / "llm" / "vendor").mkdir(parents=True)
    (root / "other").mkdir(parents=True)
    (root / "llm" / "app" / "core.py").write_text(
        "def route_packets(x):\n    return x\n\n\nclass Scheduler:\n    pass\n", encoding="utf-8")
    (root / "llm" / "vendor" / "lib.py").write_text("def vendorfn(): pass\n", encoding="utf-8")
    (root / "other" / "x.py").write_text("def otherfn(): pass\n", encoding="utf-8")
    store = Store(tmp_path / "c.sqlite3")
    report = run_index(store, FakeEmbedder(), projects_root=root, home=tmp_path / "h",
                       collections=("code",), code_projects={"llm"})
    assert report.errors == []
    paths = [r[0] for r in store.conn.execute("SELECT path FROM documents WHERE collection='code'")]
    assert len(paths) == 1 and paths[0].endswith("core.py")
    hits = hybrid(store, VectorCache(store), "route_packets", None, mode="lexical", k=3)
    assert hits and hits[0].locator == "line 1"
    assert hybrid(store, VectorCache(store), "otherfn", None, mode="lexical", k=3) == []
    store.close()


def test_code_without_projects_is_not_indexed(tmp_path):
    root = tmp_path / "Projects"
    (root / "llm").mkdir(parents=True)
    (root / "llm" / "a.py").write_text("def f(): pass\n", encoding="utf-8")
    store = Store(tmp_path / "c2.sqlite3")
    run_index(store, FakeEmbedder(), projects_root=root, home=tmp_path / "h", collections=("code",))
    assert "code" not in store.stats()
    store.close()


def test_repository_readme_and_docs_join_the_code_collection(tmp_path):
    from archive_index.embed import query_prompt
    from archive_index.sources import discover_code

    proj = tmp_path / "llm" / "tool"
    (proj / "docs").mkdir(parents=True)
    (proj / "src").mkdir(parents=True)
    (proj / "README.md").write_text("# tool", encoding="utf-8")
    (proj / "docs" / "guide.md").write_text("guide", encoding="utf-8")
    (proj / "notes.md").write_text("личні нотатки", encoding="utf-8")
    (proj / "src" / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    names = sorted(p.name for p in discover_code(tmp_path, {"llm"}))
    assert names == ["README.md", "a.py", "guide.md"]
    assert query_prompt(["code"]) == "CodeRetrieval"
    assert query_prompt(["docs", "wiki"]) == "SearchQuery"
    assert query_prompt(None) == "SearchQuery"


def test_named_repository_docs_are_not_also_indexed_as_wiki(tmp_path):
    root = tmp_path / "Projects"
    repo = root / "code-tool"
    (repo / "docs").mkdir(parents=True)
    (root / "notes").mkdir(parents=True)
    (repo / "README.md").write_text("# repo readme\n\nrepository documentation", encoding="utf-8")
    (repo / "docs" / "guide.md").write_text("# guide\n\nrepository guide", encoding="utf-8")
    (root / "notes" / "note.md").write_text("# personal note\n\nprivate note", encoding="utf-8")
    store = Store(tmp_path / "overlap.sqlite3")
    try:
        report = run_index(store, FakeEmbedder(), projects_root=root, home=tmp_path / "h",
                           collections=("wiki", "code"), code_projects={"code-tool"})
        assert report.errors == []
        rows = {Path(path).relative_to(root).as_posix(): collection for path, collection
                in store.conn.execute("SELECT path, collection FROM documents")}
        assert rows == {
            "code-tool/README.md": "code",
            "code-tool/docs/guide.md": "code",
            "notes/note.md": "wiki",
        }
    finally:
        store.close()


def test_mcp_server_accepts_opt_in_collections():
    from archive_index import mcp_server

    assert "code" in mcp_server.ALL_COLLECTIONS
    assert "config" in mcp_server.ALL_COLLECTIONS
    assert "logs" in mcp_server.ALL_COLLECTIONS

