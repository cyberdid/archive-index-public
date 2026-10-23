import json

import numpy as np
import pytest
from PIL import Image

from archive_index.indexer import run_index
from archive_index.store import Store
from paper.queries.schema import Query


class _Fake:
    def documents(self, texts, titles=None):
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

    def query(self, text, prompt="SearchQuery"):
        return np.ones(8, dtype=np.float32) / np.sqrt(8)

    def image(self, path):
        return np.ones(8, dtype=np.float32) / np.sqrt(8)


def test_merge_results_keeps_other_sections(tmp_path):
    from paper.eval.run_efficiency import merge_results

    path = tmp_path / "efficiency.json"
    merge_results(path, "queries/gpu", {"p50": 1.0})
    merge_results(path, "index/gpu", {"seconds": 2.0})
    merge_results(path, "queries/gpu", {"p50": 3.0})
    assert json.loads(path.read_text()) == {"queries": {"gpu": {"p50": 3.0}}, "index": {"gpu": {"seconds": 2.0}}}


def test_gpu_sampler_reports_baseline_and_peak():
    from paper.eval.run_efficiency import GpuMemorySampler

    readings = iter([500, 500, 2100, 4100, 3000] + [3000] * 1000)
    with GpuMemorySampler(reader=lambda: next(readings), interval=0.001) as sampler:
        import time
        time.sleep(0.05)
    assert sampler.baseline_mib == 500 and sampler.peak_mib == 4100 and sampler.peak_mib - sampler.baseline_mib == 3600


def test_subset_copies_the_requested_number_per_part_deterministically(tmp_path):
    from paper.eval.run_efficiency import make_subset

    root = tmp_path / "ua-par"
    entries = []
    for part, n in (("notes", 6), ("images", 4)):
        for i in range(n):
            path = root / part / f"{i}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{part} {i}", encoding="utf-8")
            entries.append({"doc_id": f"{part}/{i}.txt", "part": part})
    a = make_subset(entries, root, tmp_path / "a", {"notes": 3, "images": 2})
    b = make_subset(entries, root, tmp_path / "b", {"notes": 3, "images": 2})
    assert a == b and len(a) == 5 and sum(d.startswith("notes/") for d in a) == 3
    assert all((tmp_path / "a" / d).read_text(encoding="utf-8") == (root / d).read_text(encoding="utf-8") for d in a)


def test_text_chunk_rows_leave_out_pictures(tmp_path):
    from paper.eval.run_efficiency import text_chunk_rows

    root = tmp_path / "c"
    (root / "radio").mkdir(parents=True)
    (root / "radio" / "a.md").write_text("# Антена\n\nАнтена на даху лабораторії.\n", encoding="utf-8")
    Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "radio" / "photo.png")
    store = Store(tmp_path / "full.sqlite3")
    run_index(store, _Fake(), projects_root=root, home=tmp_path / "h", collections=("wiki", "images"))
    rows = text_chunk_rows(store)
    store.close()
    assert len(rows) == 1 and rows[0][1] == "Антена" and "даху" in rows[0][2]


def test_run_system_searches_a_given_vector_cache(tmp_path):
    from archive_index.search import VectorCache
    from paper.runs.systems import CONTROLS, run_system

    root = tmp_path / "ua-par"
    (root / "radio").mkdir(parents=True)
    (root / "radio" / "a.md").write_text("# Антена\n\nАнтена на даху.\n", encoding="utf-8")
    (root / "radio" / "b.md").write_text("# Кабель\n\nКабель до приймача.\n", encoding="utf-8")
    (tmp_path / "idx").mkdir()
    store = Store(tmp_path / "idx" / "full.sqlite3")
    run_index(store, _Fake(), projects_root=root, home=tmp_path / "h", collections=("wiki",))
    ids = [r[0] for r in store.conn.execute("SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id"
                                            " WHERE d.path LIKE '%b.md'")]
    store.close()

    def cache_factory(store):
        cache = VectorCache(store)
        cache.ensure()
        cache.matrix = np.array([[0.0] * 8 if cid not in ids else [1.0] + [0.0] * 7 for cid in cache.ids],
                                dtype=np.float32)
        return cache

    class Query1(_Fake):
        def query(self, text, prompt="SearchQuery"):
            return np.array([1.0] + [0.0] * 7, dtype=np.float32)

    rows = run_system(CONTROLS["dense_eg2_text"], [Query("q1", "x", "uk", "topical", "gpt-6-luna")], root,
                      tmp_path / "idx", {"eg2": Query1()}, k=2, cache_factory=cache_factory)
    assert rows[0][1][0][0] == "radio/b.md"


def test_environment_names_the_hardware_and_package_versions():
    from paper.eval.run_efficiency import environment

    env = environment()
    assert env["python"].count(".") >= 2 and "Linux" in env["os"]
    assert env["packages"]["numpy"] and env["packages"]["sentence-transformers"]
    assert env["cpu"] and env["logical_cpus"] >= 1 and env["memory_gb"] > 0
