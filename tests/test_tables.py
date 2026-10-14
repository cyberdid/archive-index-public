import numpy as np
import openpyxl

from archive_index.indexer import run_index
from archive_index.search import VectorCache, hybrid
from archive_index.sources import parse_table
from archive_index.store import Store


class FakeEmbedder:
    def documents(self, texts, titles=None):
        return np.ones((len(texts), 4), dtype=np.float32) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return np.ones(4, dtype=np.float32)


def _xlsx(path, header, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BOM"
    ws.append(header)
    for r in rows:
        ws.append(r)
    wb.save(path)


def test_same_numbers_are_told_apart_by_their_headers(tmp_path):
    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    _xlsx(projects / "budget.xlsx", ["Стаття", "Сума UAH"], [["антени", 120], ["кабелі", 45]])
    _xlsx(projects / "stock.xlsx", ["Склад", "Залишок шт"], [["антени", 120], ["кабелі", 45]])
    store = Store(tmp_path / "t.sqlite3")
    report = run_index(store, FakeEmbedder(), projects_root=tmp_path / "Projects",
                       home=tmp_path / "h", collections=("tables",))
    assert report.errors == []
    hits = hybrid(store, VectorCache(store), "Сума UAH", None, mode="lexical", k=5)
    assert hits and {h.path.split("/")[-1] for h in hits} == {"budget.xlsx"}
    rows = store.conn.execute(
        "SELECT d.path, c.body FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.collection='tables'"
    ).fetchall()
    for path, body in rows:
        if path.endswith("budget.xlsx"):
            assert "Сума UAH" in body and "Залишок" not in body
        else:
            assert "Залишок шт" in body and "Сума UAH" not in body
    store.close()


def test_csv_with_semicolons_is_read_with_its_delimiter(tmp_path):
    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    f = projects / "data.csv"
    f.write_text("назва;кількість\nплата;4\nмодем;2\n", encoding="utf-8")
    doc = parse_table(f, tmp_path / "Projects")
    assert doc.units and "назва | кількість" in doc.units[0].text
    assert "модем | 2" in doc.units[0].text


def test_numeric_windows_are_lexical_only(tmp_path):
    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    rows = [[i, i * 2, i * 3, i * 4] for i in range(1, 40)]
    _xlsx(projects / "nums.xlsx", ["a", "b", "c", "d"], rows)
    store = Store(tmp_path / "n.sqlite3")
    run_index(store, FakeEmbedder(), projects_root=tmp_path / "Projects", home=tmp_path / "h",
              collections=("tables",))
    dims = [r[0] for r in store.conn.execute("SELECT dimension FROM chunks WHERE collection='tables'")]
    assert dims and all(d == 0 for d in dims)  # all-number windows have no vectors
    store.close()


def test_long_binary_cells_are_cut(tmp_path):
    from archive_index.sources import TABLE_CELL_CHARS, _sheet_units

    blob = "ab" * 50_000
    units = _sheet_units("s", [["id", "data"], ["1", blob]])
    assert units and len(units[0].text) < TABLE_CELL_CHARS + 200
