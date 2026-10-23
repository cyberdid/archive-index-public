"""RQ5 measurements on the built UA-PAR corpus (plan Task 10, step 5), merged into paper/results/efficiency.json.

  queries --device gpu|cpu   per-query latency of each system, per-core/per-thread CPU, IRQ placement, peak RSS/VRAM
  index --device gpu|cpu     indexing a fixed sample of the corpus with the archive CLI (seconds, disk, RSS, VRAM)
  gguf                       retrieval quality and speed of the GGUF builds of EG2 (f16, Q8_0, Q4_K_M) on the
                             4137 text documents, against the SentenceTransformers bf16 model on the same documents

The device is chosen by CUDA_VISIBLE_DEVICES ('' for CPU), set by this script for the child work it starts.
"""

from __future__ import annotations

import json
import hashlib
import os
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

RESULTS = Path("paper/results/efficiency.json")
CORPUS = Path("paper/data/ua-par")
INDEXES = Path("paper/data/indexes")
QUERY_SYSTEMS = ("bm25_plain", "bm25_lemma", "dense_eg2", "dense_e5", "rrf", "adaptive", "dense_eg2_256")
SUBSET = {"notes": 300, "documents": 40, "tables": 15, "images": 60, "speech": 60}
# EG2 on the CPU embeds about one chunk per second here, so the CPU run indexes a fifth of the GPU sample;
# throughput is compared per chunk.
SUBSET_CPU = {"notes": 60, "documents": 8, "tables": 3, "images": 12, "speech": 12}
SUBSET_COLLECTIONS = ("wiki", "docs", "tables", "images", "audio")
GGUF_DIR = Path.home() / ".local/share/archive-index/gguf"
GGUF = {"f16": "embeddinggemma2-f16.gguf", "Q8_0": "embeddinggemma2-Q8_0.gguf", "Q4_K_M": "embeddinggemma2-Q4_K_M.gguf"}
LLAMA_SERVER = Path.home() / ".local/src/llama.cpp/build-cuda/bin/llama-server"


PACKAGES = ("numpy", "torch", "transformers", "sentence-transformers", "pymorphy3", "pymorphy3-dicts-uk",
            "pymorphy3-dicts-ru", "ir-measures", "pytrec-eval-terrier")


def environment() -> dict:
    """Hardware and software of the measurements."""
    import importlib.metadata
    import platform
    import sqlite3

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    memory_kb = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemTotal")))
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    llama = subprocess.run(["git", "-C", str(LLAMA_SERVER.parents[2]), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip() or None
    return {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}", "cpu": cpu,
            "logical_cpus": os.cpu_count(), "memory_gb": round(memory_kb / 2**20, 1), "gpu": gpu,
            "sqlite": sqlite3.sqlite_version, "llama_cpp_commit": llama,
            "packages": {name: version(name) for name in PACKAGES}}


def merge_results(path: Path, section: str, value) -> None:
    """Write value at a 'a/b' section of the JSON file, keeping every other section."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    node = data
    *parents, leaf = section.split("/")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _nvidia_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


class GpuMemorySampler:
    """Samples GPU memory in a thread: baseline (first reading) and peak over the block."""

    def __init__(self, reader=_nvidia_used_mib, interval: float = 0.2) -> None:
        self.reader, self.interval = reader, interval
        self.baseline_mib = self.peak_mib = None
        self._stop = threading.Event()

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self.reader()
            self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline_mib = self.peak_mib = self.reader()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        return {"baseline_mib": self.baseline_mib, "peak_mib": self.peak_mib,
                "added_mib": self.peak_mib - self.baseline_mib}


def make_subset(entries: list[dict], corpus_root: Path, out: Path, counts: dict[str, int], seed: int = 2026) -> list[str]:
    """Copy a fixed random sample of each corpus part, keeping relative paths; returns the doc ids."""
    rng = random.Random(seed)
    chosen = []
    for part, n in counts.items():
        ids = sorted(e["doc_id"] for e in entries if e["part"] == part)
        chosen += rng.sample(ids, min(n, len(ids)))
    for doc in chosen:
        target = out / doc
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(corpus_root / doc, target)
    return chosen


def text_chunk_rows(store) -> list[tuple[int, str, str]]:
    """Vectorized rows in text collections, in id order; includes stored text of raster PDF pages.

    The full index has21image-embedded PDF-page rows in these collections. GGUF
    encodes their stored body text; it cannot reproduce the reference image input.
    """
    from paper.runs.systems import TEXT_COLLECTIONS

    marks = ",".join("?" for _ in TEXT_COLLECTIONS)
    return store.conn.execute(
        f"SELECT c.id, d.title, c.body FROM chunks c JOIN documents d ON d.id = c.doc_id"
        f" WHERE c.collection IN ({marks}) AND c.dimension > 0 ORDER BY c.id", TEXT_COLLECTIONS).fetchall()


def _device_env(device: str) -> dict:
    return {"CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else {}


def measure_query_latency(device: str) -> dict:
    from archive_index.embed import E5Embedder, Embedder
    from paper.eval.efficiency import measure_queries
    from paper.queries.schema import load_queries
    from paper.runs.systems import SYSTEMS

    queries = load_queries(Path("paper/data/queries.tsv"))
    embedders = {"eg2": Embedder(), "e5": E5Embedder()}
    out = {}
    for name in QUERY_SYSTEMS:
        with GpuMemorySampler() as gpu:
            # warm-up over every query: each system reads its own index file, and a five-query warm-up left the
            # first measured system with a cold page cache (bm25_plain looked slower than bm25_lemma)
            result = measure_queries(SYSTEMS[name], queries, CORPUS, INDEXES, embedders, warmup=len(queries))
        out[name] = {**result, "gpu_memory": gpu.summary()}
        print(name, json.dumps(result["latency_ms"]), flush=True)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


def measure_subset_indexing(device: str) -> dict:
    from paper.corpus.build import read_manifest
    from paper.eval.efficiency import measure_index

    entries = read_manifest(CORPUS / "manifest.jsonl")
    with tempfile.TemporaryDirectory(dir="paper/data") as tmp:
        root, home = Path(tmp) / "subset", Path(tmp) / "home"
        home.mkdir()
        sample = SUBSET_CPU if device == "cpu" else SUBSET
        docs = make_subset(entries, CORPUS, root, sample)
        db = Path(tmp) / "subset.sqlite3"
        args = [sys.executable, "-c", "import sys; from archive_index.cli import main; sys.exit(main(sys.argv[1:]))",
                "--db", str(db), "index", "--projects-root", str(root), "--home", str(home)]
        for collection in SUBSET_COLLECTIONS:
            args += ["-c", collection]
        with GpuMemorySampler() as gpu:
            result = measure_index(args, db, env=_device_env(device))
        import sqlite3

        conn = sqlite3.connect(db)
        chunks = dict(conn.execute("SELECT collection, count(*) FROM chunks GROUP BY collection").fetchall())
        conn.close()
    return {**result, "gpu_memory": gpu.summary(), "sample": sample, "files": len(docs), "chunks": chunks,
            "full_index_disk_bytes": {p.name: p.stat().st_size for p in sorted(INDEXES.glob("*.sqlite3"))}}


def measure_gguf() -> dict:
    from archive_index.embed import LlamaServerEmbedder
    from archive_index.search import VectorCache
    from archive_index.store import Store
    from paper.eval.controls import QRELS_VARIANTS, compare_encoders
    from paper.eval.efficiency import latency_summary, serve_gguf
    from paper.eval.metrics import read_qrels, read_run
    from paper.queries.schema import load_queries
    from paper.runs.systems import CONTROLS, run_system, write_trec_run

    queries = load_queries(Path("paper/data/queries.tsv"))
    store = Store(INDEXES / "full.sqlite3")
    rows = text_chunk_rows(store)
    store.close()
    runs = {"dense_eg2_text": read_run(Path("paper/data/runs-control/dense_eg2_text.trec"))}
    settings = {"context_tokens": 8192, "batch_tokens": 2048, "parallel": 4}
    out, port = {"text_chunks": len(rows), "builds": {}, "server_settings": settings,
                 "complete": False, "query_warmup": len(queries)}, 8291
    cache_dir = Path("paper/data/gguf-vectors")
    cache_dir.mkdir(parents=True, exist_ok=True)
    row_digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    for quant, filename in GGUF.items():
        from paper.judge.finalize import sha256, write_json

        cache_path, metadata_path = cache_dir / f"{quant}.npy", cache_dir / f"{quant}.json"
        fingerprint = {"rows_sha256": row_digest, "model_sha256": sha256(GGUF_DIR / filename),
                       "llama_cpp_commit": environment()["llama_cpp_commit"], "server_settings": settings,
                       "embedding_prefix": "title: {title or 'none'} | text: {text}"}
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        server = serve_gguf(LLAMA_SERVER, GGUF_DIR / filename, port, gpu=True, **settings)
        try:
            embedder = LlamaServerEmbedder(f"http://127.0.0.1:{port}")
            if (cache_path.exists() and metadata.get("fingerprint") == fingerprint
                    and metadata.get("vectors_sha256") == sha256(cache_path)):
                vectors = np.load(cache_path)
                embed_seconds = metadata["embed_seconds"]
                print(quant, "using validated embedding cache", flush=True)
            else:
                started = time.perf_counter()
                batches = []
                for i in range(0, len(rows), 32):
                    batch = rows[i:i + 32]
                    batches.append(embedder.documents([b for _, _, b in batch], [t for _, t, _ in batch]))
                    if i % 2048 == 0:
                        print(quant, f"embedded {i + len(batch)}/{len(rows)}", flush=True)
                vectors = np.vstack(batches)
                embed_seconds = time.perf_counter() - started
                np.save(cache_path, vectors)
                metadata = {"fingerprint": fingerprint, "embed_seconds": embed_seconds,
                            "vectors_sha256": sha256(cache_path)}
                write_json(metadata_path, metadata)
            if vectors.shape != (len(rows), 768) or not np.isfinite(vectors).all():
                raise ValueError("invalid GGUF embedding cache shape/values")
            index_of = {cid: n for n, (cid, _, _) in enumerate(rows)}

            def cache_factory(store, vectors=vectors, index_of=index_of):
                cache = VectorCache(store)
                cache.ensure()
                matrix = np.zeros((len(cache.ids), vectors.shape[1]), dtype=np.float32)
                for row, cid in enumerate(cache.ids):
                    if int(cid) in index_of:
                        matrix[row] = vectors[index_of[int(cid)]]
                cache.matrix = matrix
                return cache

            timings: list[float] = []
            run_system(CONTROLS["dense_eg2_text"], queries, CORPUS, INDEXES, {"eg2": embedder},
                       cache_factory=cache_factory)  # full-query warm-up, as in the ST latency experiment
            result = run_system(CONTROLS["dense_eg2_text"], queries, CORPUS, INDEXES, {"eg2": embedder},
                                timings=timings, cache_factory=cache_factory)
            write_trec_run(Path("paper/data/runs-gguf") / f"{quant}.trec", f"gguf_{quant}", result)
            runs[f"gguf_{quant}"] = read_run(Path("paper/data/runs-gguf") / f"{quant}.trec")
            out["builds"][quant] = {"file_mb": (GGUF_DIR / filename).stat().st_size / 2**20,
                                    "embed_chunks_per_s": len(rows) / embed_seconds,
                                    "query_latency_ms_gpu_server": latency_summary(timings),
                                    "embedding_cache": metadata,
                                    "run_sha256": sha256(Path("paper/data/runs-gguf") / f"{quant}.trec")}
            merge_results(RESULTS, "quantization/retrieval", out)
        finally:
            server.terminate()
            server.wait()
        print(quant, json.dumps(out["builds"][quant]), flush=True)
    answerable = {q.qid for q in queries if q.category != "no_answer"}
    out["quality"] = {}
    for variant, filename in QRELS_VARIANTS.items():
        path = Path("paper/data") / filename
        if path.exists():
            qrels = {qid: judged for qid, judged in read_qrels(path).items() if qid in answerable}
            out["quality"][variant] = compare_encoders(qrels, runs, reference="dense_eg2_text")
    out["complete"] = True
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("queries", "index", "gguf", "env"))
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.command == "queries":
        if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            os.execvpe(sys.executable, [sys.executable, "-m", "paper.eval.run_efficiency", "queries", "--device", "cpu"],
                       {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        merge_results(RESULTS, f"queries/{args.device}", measure_query_latency(args.device))
    elif args.command == "env":
        merge_results(RESULTS, "environment", environment())
    elif args.command == "index":
        merge_results(RESULTS, f"index/{args.device}", measure_subset_indexing(args.device))
    else:
        merge_results(RESULTS, "quantization/retrieval", measure_gguf())
    print(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
