"""Build the four benchmark indexes and write one TREC run per system (plan Task 6, step 5).

The indexes hold the corpus only: run_index gets an empty temporary home, so no Claude or Codex sessions of the
person running it can enter a benchmark index.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from archive_index.indexer import run_index
from archive_index.store import Store
from paper.queries.schema import load_queries
from paper.runs.systems import SYSTEMS, run_system, write_trec_run

TEXT_AND_MEDIA = ("wiki", "docs", "tables", "images", "audio", "code")
TEXT_ONLY = ("wiki", "docs", "tables", "code")
# index file -> (embedder key, collections, lemmatize, min_letters)
INDEXES = {
    "full.sqlite3": ("eg2", TEXT_AND_MEDIA, True, 20),
    "plain.sqlite3": ("eg2", TEXT_AND_MEDIA, False, 20),
    "noshort.sqlite3": ("eg2", TEXT_AND_MEDIA, True, 0),
    "e5.sqlite3": ("e5", TEXT_ONLY, True, 20),
}


def default_embedders() -> dict:
    from archive_index.embed import E5Embedder, Embedder

    return {"eg2": Embedder(), "e5": E5Embedder()}


def build_indexes(corpus_root: Path, index_dir: Path, embedders: dict | None = None) -> dict[str, Path]:
    embedders = embedders or default_embedders()
    corpus_root = corpus_root.resolve()  # stored document paths are absolute; a relative root duplicates them
    index_dir.mkdir(parents=True, exist_ok=True)
    code_projects = {p.name for p in corpus_root.iterdir() if p.is_dir() and p.name.startswith("code-")}
    paths = {}
    with tempfile.TemporaryDirectory() as empty_home:
        for name, (key, collections, lemmatize, min_letters) in INDEXES.items():
            store = Store(index_dir / name)
            try:
                run_index(store, embedders[key], projects_root=corpus_root, home=Path(empty_home),
                          collections=collections, code_projects=code_projects, lemmatize=lemmatize,
                          min_letters=min_letters)
            finally:
                store.close()
            paths[name] = index_dir / name
    return paths


def run_all(queries, out_dir: Path, corpus_root: Path, index_dir: Path, embedders: dict | None = None,
            k: int = 100) -> dict[str, Path]:
    embedders = embedders or default_embedders()
    written = {}
    for name, config in SYSTEMS.items():
        rows = run_system(config, queries, corpus_root, index_dir, embedders, k=k)
        path = out_dir / f"{name}.trec"
        write_trec_run(path, name, rows)
        written[name] = path
    return written


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("paper/data/ua-par"))
    parser.add_argument("--indexes", type=Path, default=Path("paper/data/indexes"))
    parser.add_argument("--queries", type=Path, default=Path("paper/data/queries.tsv"))
    parser.add_argument("--out", type=Path, default=Path("paper/data/runs"))
    parser.add_argument("--build", action="store_true", help="build the four indexes first")
    args = parser.parse_args()
    embedders = default_embedders()
    if args.build:
        build_indexes(args.corpus, args.indexes, embedders)
    written = run_all(load_queries(args.queries), args.out, args.corpus, args.indexes, embedders)
    for name, path in written.items():
        print(name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
