"""Command line: `archive index`, `archive search`, `archive stats`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .indexer import ALL_COLLECTIONS, COLLECTIONS, run_index
from .search import VectorCache, hybrid
from .logs import get_logger
from .store import Store

log = get_logger("cli")

DEFAULT_DB = Path.home() / ".local" / "share" / "archive-index" / "archive.sqlite3"
DEFAULT_PROJECTS = Path.home() / "Projects"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="archive", description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="index sources (incremental)")
    idx.add_argument("--projects-root", type=Path, default=DEFAULT_PROJECTS)
    idx.add_argument("--home", type=Path, default=Path.home())
    idx.add_argument("-c", "--collection", action="append", choices=ALL_COLLECTIONS)
    idx.add_argument("--session-limit", type=int, default=None,
                     help="index only the first N session files (for trial runs)")
    idx.add_argument("--force", action="store_true", help="re-index unchanged files")
    idx.add_argument("--project", action="append", default=None,
                     help="project folder for the code collection (repeatable), e.g. --project llm")
    idx.add_argument("--no-lemmas", action="store_true", help="index words only, without lemmas (ablation)")
    idx.add_argument("--min-letters", type=int, default=20,
                     help="drop document fragments with fewer letters; 0 keeps everything (ablation)")
    idx.add_argument("--skip-screenshots", action="store_true",
                     help="leave screenshots out of the image collection")

    srch = sub.add_parser("search", help="hybrid search")
    srch.add_argument("query")
    srch.add_argument("-c", "--collection", action="append", choices=ALL_COLLECTIONS)
    srch.add_argument("--project", default=None, help="project prefix, e.g. llm")
    srch.add_argument("-k", "--limit", dest="k", type=int, default=8,
                      help="how many results to print")
    srch.add_argument("--per-document", type=int, default=3,
                      help="max fragments shown per document; 0 turns grouping off")
    srch.add_argument("--mode", choices=("hybrid", "lexical", "semantic"), default="hybrid")
    srch.add_argument("--grouped", action="store_true", help="top-k per collection")
    srch.add_argument("--json", action="store_true")


    sub.add_parser("stats", help="counts per collection")
    sub.add_parser("rebuild-terms", help="recompute word terms (e.g. after a lexical rule change)")
    return parser


def _embedder():
    from .embed import make_embedder

    return make_embedder()


def _acquire_lock(db: Path):
    """Non-blocking exclusive lock so the timer and the session hook never index in parallel."""
    import fcntl

    db.parent.mkdir(parents=True, exist_ok=True)
    handle = open(db.with_suffix(".lock"), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _append_run_log(db: Path, collections, report) -> None:
    """One JSON line per run: when it ran, what it touched. Lets daily runs be checked later."""
    from datetime import datetime, timezone

    line = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collections": list(collections),
        "indexed": report.indexed,
        "skipped": report.skipped,
        "chunks": report.chunks,
        "errors": len(report.errors),
    }
    path = db.parent / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None, embedder=None) -> int:
    """Logs every command and any failure, then runs it."""
    import sys

    log.info("command %s", " ".join(argv if argv is not None else sys.argv[1:]))
    try:
        return _run(argv, embedder)
    except SystemExit:
        raise
    except Exception:
        log.exception("command failed")
        raise


def _run(argv: list[str] | None = None, embedder=None) -> int:
    args = _build_parser().parse_args(argv)
    store = Store(args.db)
    try:
        if args.command == "rebuild-terms":
            from .lexical import index_terms

            count = store.rebuild_terms(index_terms)
            print(json.dumps({"chunks_rebuilt": count}, ensure_ascii=False))
            return 0

        if args.command == "stats":
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
            return 0

        if args.command == "index":
            lock = _acquire_lock(args.db)
            if lock is None:
                print("archive index: another run is in progress; skipping", file=sys.stderr)
                return 0
            emb = embedder or _embedder()
            collections = tuple(args.collection) if args.collection else COLLECTIONS
            report = run_index(
                store, emb, projects_root=args.projects_root, home=args.home,
                collections=collections, session_limit=args.session_limit,
                force=args.force, progress=lambda m: print(m, file=sys.stderr),
                code_projects=set(args.project or []), skip_screenshots=args.skip_screenshots,
                lemmatize=not args.no_lemmas, min_letters=args.min_letters,
            )
            _append_run_log(args.db, collections, report)
            print(json.dumps({
                "indexed_documents": report.indexed, "skipped": report.skipped,
                "chunks_written": report.chunks, "redactions": report.redactions,
                "errors": report.errors[:20], "error_count": len(report.errors),
            }, ensure_ascii=False, indent=2))
            return 0

        if args.command == "search":
            emb = embedder or _embedder()
            from .embed import query_prompt

            qv = emb.query(args.query, prompt=query_prompt(args.collection)) if args.mode != "lexical" else None
            cache = VectorCache(store)
            if args.grouped and args.collection:
                groups = args.collection
            elif args.grouped:
                groups = [c for c in COLLECTIONS]
            else:
                groups = [None]
            results = {}
            for group in groups:
                hits = hybrid(store, cache, args.query, qv,
                              collections=[group] if group else args.collection,
                              project=args.project, k=args.k, mode=args.mode,
                              per_document=args.per_document)
                results[group or "all"] = [h.__dict__ for h in hits]
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
            else:
                for name, hits in results.items():
                    print(f"== {name} ({len(hits)})")
                    for h in hits:
                        sim = "   -  " if h["similarity"] is None else f"{h['similarity']:.4f}"
                        print(f"  [{h['via']:8s}] схожість {sim} {h['collection']}/{h['project'] or '-'}"
                              f" · {h['title'][:70]} · {h['locator']}")
                        if h.get("more_in_document"):
                            print(f"      +{h['more_in_document']} ще фрагментів у цьому документі")
                        print(f"      {h['snippet'][:200]}")
            return 0
    finally:
        store.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
