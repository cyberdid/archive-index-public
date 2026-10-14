import numpy as np
from PIL import Image

from archive_index.indexer import PDF_PASS_PAGES, run_index
from archive_index.store import Store


class FakeEmbedder:
    def documents(self, texts, titles=None):
        return np.ones((len(texts), 4), dtype=np.float32) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return np.ones(4, dtype=np.float32)

    def image(self, path):
        return np.ones(4, dtype=np.float32)


def test_large_pdf_is_indexed_in_passes(tmp_path):
    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    pages = [Image.new("RGB", (120, 160), (i * 5 % 255, 80, 200)) for i in range(PDF_PASS_PAGES + 5)]
    pages[0].save(projects / "book.pdf", save_all=True, append_images=pages[1:])
    store = Store(tmp_path / "s.sqlite3")
    run_index(store, FakeEmbedder(), projects_root=tmp_path / "Projects", home=tmp_path / "h",
              collections=("docs",))
    first = store.conn.execute("SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.doc_id"
                               " WHERE d.path LIKE '%book.pdf'").fetchone()[0]
    assert first == PDF_PASS_PAGES
    run_index(store, FakeEmbedder(), projects_root=tmp_path / "Projects", home=tmp_path / "h",
              collections=("docs",))
    total = store.conn.execute("SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.doc_id"
                               " WHERE d.path LIKE '%book.pdf'").fetchone()[0]
    assert total == PDF_PASS_PAGES + 5
    run_index(store, FakeEmbedder(), projects_root=tmp_path / "Projects", home=tmp_path / "h",
              collections=("docs",))
    again = store.conn.execute("SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.doc_id"
                               " WHERE d.path LIKE '%book.pdf'").fetchone()[0]
    assert again == total  # finished documents are not re-embedded
    store.close()


def test_repeated_passes_never_duplicate_pages(tmp_path):
    # Regression: a PDF indexed before passes existed has read_line=1; later passes must not add its pages again.
    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    pages = [Image.new("RGB", (120, 160), (i * 9 % 255, 60, 180)) for i in range(40)]
    pages[0].save(projects / "old.pdf", save_all=True, append_images=pages[1:])
    store = Store(tmp_path / "s.sqlite3")
    kw = dict(projects_root=tmp_path / "Projects", home=tmp_path / "h", collections=("docs",))
    for _ in range(4):
        run_index(store, FakeEmbedder(), **kw)
    rows = store.conn.execute("SELECT COUNT(*), COUNT(DISTINCT locator) FROM chunks c JOIN documents d "
                              "ON d.id=c.doc_id WHERE d.path LIKE '%old.pdf'").fetchone()
    assert rows[0] == rows[1] == 40  # one fragment per page, no repeats
    store.close()


def test_remove_duplicate_chunks_keeps_one_copy(tmp_path):
    store = Store(tmp_path / "d.sqlite3")
    from archive_index.store import ChunkRecord
    vec = np.ones(4, dtype=np.float32)
    rec = lambda i, body: ChunkRecord(ord=i, heading="", locator="page 1", body=body,
                                      terms=body, occurred_at="", vector=vec, body_sha=body)
    store.replace_document(collection="docs", path="/x/a.pdf", size=1, mtime_ns=1, sha256="s",
                           title="a", project="p", chunks=[rec(0, "same"), rec(1, "same"), rec(2, "other")])
    assert store.remove_duplicate_chunks("docs") == 1
    assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    store.close()
