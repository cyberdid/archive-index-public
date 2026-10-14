import shutil

import numpy as np
import pytest
from PIL import Image

from archive_index.indexer import run_index
from archive_index.store import Store


class FakeEmbedder:
    def __init__(self):
        self.images = 0

    def documents(self, texts, titles=None):
        return np.ones((len(texts), 4), dtype=np.float32) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return np.ones(4, dtype=np.float32)

    def image(self, path):
        self.images += 1
        return np.ones(4, dtype=np.float32)


@pytest.mark.skipif(shutil.which("pdftoppm") is None or shutil.which("pdftotext") is None,
                    reason="poppler utilities not installed")
def test_picture_only_pdf_page_is_embedded_as_image(tmp_path):
    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    Image.new("RGB", (300, 200), (30, 90, 200)).save(projects / "scan.pdf")  # no text layer
    embedder = FakeEmbedder()
    store = Store(tmp_path / "scan.sqlite3")
    report = run_index(store, embedder, projects_root=tmp_path / "Projects", home=tmp_path / "h",
                       collections=("docs",))
    assert report.errors == []
    assert embedder.images == 1
    rows = store.conn.execute(
        "SELECT c.locator, c.dimension FROM chunks c JOIN documents d ON d.id = c.doc_id"
        " WHERE d.path LIKE '%scan.pdf'").fetchall()
    assert rows == [("page 1 (image)", 4)]
    store.close()
