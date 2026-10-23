import os

import pytest

from paper.eval.efficiency import core_usage, latency_summary, parse_proc_stat, thread_cpu_seconds


def test_latency_percentiles():
    out = latency_summary([10, 20, 30, 40, 100])
    assert out == pytest.approx({"p50": 30.0, "p95": 88.0, "mean": 40.0, "n": 5})


STAT_A = "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0 0 0\ncpu1 50 0 50 400 0 0 0 0 0 0\nintr 1\n"
STAT_B = "cpu  250 0 100 950 0 0 0 0 0 0\ncpu0 150 0 50 400 0 0 0 0 0 0\ncpu1 100 0 50 750 0 0 0 0 0 0\nintr 2\n"


def test_per_core_usage_between_two_samples():
    a, b = parse_proc_stat(STAT_A), parse_proc_stat(STAT_B)
    assert set(a) == {"cpu0", "cpu1"}
    usage = core_usage(a, b)
    assert usage["cpu0"] == pytest.approx(100.0)  # 100 busy jiffies of 100
    assert usage["cpu1"] == pytest.approx(50 / 400 * 100)


def test_thread_cpu_of_this_process():
    threads = thread_cpu_seconds(os.getpid())
    assert threads and all(v >= 0 for v in threads.values())


def test_run_system_reports_per_query_latency(tmp_path):
    from archive_index.indexer import run_index
    from archive_index.store import Store
    from paper.queries.schema import Query
    from paper.runs.systems import SYSTEMS, run_system
    from test_runs import _Fake

    root = tmp_path / "ua-par"
    (root / "radio" / "notes").mkdir(parents=True)
    (root / "radio" / "notes" / "uk-1.md").write_text("# Антена\n\nАнтена ретранслятора на даху.\n", encoding="utf-8")
    (tmp_path / "idx").mkdir()
    store = Store(tmp_path / "idx" / "full.sqlite3")
    run_index(store, _Fake(), projects_root=root, home=tmp_path / "h", collections=("wiki",))
    store.close()
    timings = []
    queries = [Query("q1", "антена", "uk", "topical", "claude"), Query("q2", "даху", "uk", "topical", "claude")]
    run_system(SYSTEMS["adaptive"], queries, root, tmp_path / "idx", {"eg2": _Fake()}, k=10, timings=timings)
    assert len(timings) == 2 and all(t > 0 for t in timings)


def test_reference_cosines():
    import numpy as np

    from paper.eval.efficiency import reference_cosines

    ref = {"text0": np.array([1.0, 0.0]), "text1": np.array([0.0, 1.0])}
    out = reference_cosines(np.array([[1.0, 0.0], [1.0, 1.0]]), ref)
    assert out["worst"] == pytest.approx(2 ** -0.5) and out["mean"] == pytest.approx((1 + 2 ** -0.5) / 2)
