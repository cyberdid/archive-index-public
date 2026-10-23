"""RQ5: indexing cost, query latency, per-core and per-thread CPU use, and quantized embedders (plan Task 10)."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def latency_summary(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()),
            "n": int(len(a))}


def parse_proc_stat(text: str) -> dict[str, tuple[int, int]]:
    """Per core: (busy jiffies, total jiffies); idle and iowait count as not busy."""
    cores = {}
    for line in text.splitlines():
        name, *fields = line.split()
        if name.startswith("cpu") and name != "cpu":
            values = [int(v) for v in fields[:8]]
            idle = values[3] + values[4]
            cores[name] = (sum(values) - idle, sum(values))
    return cores


def read_proc_stat() -> dict[str, tuple[int, int]]:
    return parse_proc_stat(Path("/proc/stat").read_text())


def core_usage(before: dict, after: dict) -> dict[str, float]:
    usage = {}
    for core in before:
        busy = after[core][0] - before[core][0]
        total = after[core][1] - before[core][1]
        usage[core] = 100.0 * busy / total if total else 0.0
    return usage


def thread_cpu_seconds(pid: int) -> dict[str, float]:
    """utime + stime per thread of pid, keyed '<tid>:<name>'."""
    out = {}
    for task in Path(f"/proc/{pid}/task").iterdir():
        try:
            stat = (task / "stat").read_text()
        except OSError:
            continue
        name = stat[stat.index("(") + 1: stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2:].split()
        out[f"{task.name}:{name}"] = (int(fields[11]) + int(fields[12])) / CLOCK_TICKS
    return out


def irq_counts() -> dict[str, list[int]]:
    """Per-IRQ interrupt counts per core from /proc/interrupts, to show where device interrupts land."""
    lines = Path("/proc/interrupts").read_text().splitlines()
    ncpu = len(lines[0].split())
    out = {}
    for line in lines[1:]:
        parts = line.split()
        if parts and parts[0].endswith(":"):
            counts = [int(p) for p in parts[1:1 + ncpu] if p.isdigit()]
            if len(counts) == ncpu:
                out[parts[0][:-1] + " " + " ".join(parts[1 + ncpu:])[:40]] = counts
    return out


def _irq_delta(before: dict, after: dict, top: int = 8) -> dict:
    delta = {k: [b - a for a, b in zip(before[k], after[k])] for k in before if k in after}
    busiest = sorted(delta.items(), key=lambda kv: -sum(kv[1]))[:top]
    return {k: v for k, v in busiest if sum(v)}


def measure_index(args: list[str], db: Path, env: dict | None = None) -> dict:
    """Run one `archive index` command; wall seconds, disk bytes (db + WAL), peak RSS, per-core CPU, IRQ placement."""
    stat_before, irq_before = read_proc_stat(), irq_counts()
    started = time.perf_counter()
    subprocess.run(args, check=True, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL)
    seconds = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    disk = sum(p.stat().st_size for p in db.parent.glob(db.name + "*"))
    return {"seconds": seconds, "disk_bytes": disk, "peak_rss_mb": usage.ru_maxrss / 1024,
            "cpu_user_s": usage.ru_utime, "cpu_sys_s": usage.ru_stime,
            "per_core_pct": core_usage(stat_before, read_proc_stat()), "irq_delta": _irq_delta(irq_before, irq_counts())}


def measure_queries(config, queries, corpus_root: Path, index_dir: Path, embedders: dict, warmup: int = 5) -> dict:
    """Per-query latency of one system, with per-core and per-thread CPU over the timed run."""
    from paper.runs.systems import run_system

    run_system(config, queries[:warmup], corpus_root, index_dir, embedders, k=10)
    timings: list[float] = []
    threads_before, stat_before, irq_before = thread_cpu_seconds(os.getpid()), read_proc_stat(), irq_counts()
    run_system(config, queries, corpus_root, index_dir, embedders, k=10, timings=timings)
    threads_after = thread_cpu_seconds(os.getpid())
    busy_threads = {t: round(threads_after[t] - threads_before.get(t, 0.0), 3) for t in threads_after
                    if threads_after[t] - threads_before.get(t, 0.0) > 0.01}
    return {"latency_ms": latency_summary(timings), "per_core_pct": core_usage(stat_before, read_proc_stat()),
            "thread_cpu_s": busy_threads, "irq_delta": _irq_delta(irq_before, irq_counts())}


def write_results(results: dict, path: Path = Path("paper/results/efficiency.json")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def reference_cosines(vectors: np.ndarray, reference) -> dict:
    """Cosine of each vector to the bf16 SentenceTransformers reference (evals/reference.npz, keys text0..)."""
    cos = []
    for i, v in enumerate(np.asarray(vectors, dtype=np.float64)):
        r = np.asarray(reference[f"text{i}"], dtype=np.float64)
        cos.append(float(v @ r / (np.linalg.norm(v) * np.linalg.norm(r))))
    return {"worst": min(cos), "mean": float(np.mean(cos)), "n": len(cos)}


def serve_gguf(server_bin: Path, gguf: Path, port: int, gpu: bool, *,
               context_tokens: int = 2048, batch_tokens: int | None = None,
               parallel: int | None = None) -> subprocess.Popen:
    """Start llama-server in embedding mode and wait until it answers /health."""
    import urllib.request

    args = [str(server_bin), "-m", str(gguf), "--embeddings", "--pooling", "mean",
            "-ngl", "99" if gpu else "0", "--host", "127.0.0.1", "--port", str(port), "-c", str(context_tokens)]
    if batch_tokens is not None:
        args += ["-b", str(batch_tokens), "-ub", str(batch_tokens)]
    if parallel is not None:
        args += ["-np", str(parallel)]
    proc = subprocess.Popen(args,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if b"ok" in r.read():
                    return proc
        except OSError:
            time.sleep(1)
    proc.kill()
    raise RuntimeError(f"llama-server did not start for {gguf}")
