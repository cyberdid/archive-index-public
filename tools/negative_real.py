"""Negative check on the REAL index, cross-checked against grep over the same files.

For N random tokens that should not exist anywhere:
  - the archive (lexical mode, all collections) must return 0 hits;
  - grep over the text sources (md, txt, py, json, Claude JSONL) must return 0 files.
A mismatch in either direction is reported, so both missed and invented matches are visible.
Known gap: PDFs, office files and Codex JSONL are not grep-checked (too slow); they are covered by tests.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
from pathlib import Path

from archive_index.search import VectorCache, hybrid
from archive_index.store import Store

CYR = "абвгдежзиклмнопрстуфхцчшщюя"


def make_tokens(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    tokens = set()
    while len(tokens) < n:
        if rng.random() < 0.5:
            tokens.add("".join(rng.choice(string.ascii_lowercase) for _ in range(12)))
        else:
            tokens.add("".join(rng.choice(CYR) for _ in range(12)))
    return sorted(tokens)


def grep_files(tokens: list[str], root: Path, claude_root: Path) -> dict[str, list[str]]:
    """Return {token: [files]} using one grep pass per source group."""
    pattern = Path("/tmp/archive_negative_patterns.txt")
    pattern.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    found: dict[str, list[str]] = {t: [] for t in tokens}
    for base, includes in (
        (root, ["--include=*.md", "--include=*.txt", "--include=*.py", "--include=*.json"]),
        (claude_root, ["--include=*.jsonl"]),
    ):
        proc = subprocess.run(
            ["grep", "-rIlF", "-f", str(pattern), *includes,
             "--exclude-dir=.git", "--exclude-dir=node_modules", "--exclude-dir=venv",
             "--exclude-dir=.venv", str(base)],
            capture_output=True, text=True, check=False,
        )
        for path in filter(None, proc.stdout.splitlines()):
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            for token in tokens:
                if token in text:
                    found[token].append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path.home() / ".local/share/archive-index/archive.sqlite3")
    parser.add_argument("--projects", type=Path, default=Path.home() / "Projects")
    parser.add_argument("--claude", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokens = make_tokens(args.n, args.seed)
    store = Store(args.db)
    cache = VectorCache(store)
    archive_hits = {t: len(hybrid(store, cache, t, None, mode="lexical", k=20)) for t in tokens}
    store.close()
    grep_hits = grep_files(tokens, args.projects, args.claude)

    mismatches = []
    for t in tokens:
        a, g = archive_hits[t], len(grep_hits[t])
        if a != 0 or g != 0:
            mismatches.append({"token": t, "archive_hits": a, "grep_files": grep_hits[t][:5]})
    report = {
        "tokens": len(tokens),
        "archive_zero": sum(1 for t in tokens if archive_hits[t] == 0),
        "grep_zero": sum(1 for t in tokens if not grep_hits[t]),
        "mismatches": mismatches,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, ensure_ascii=False))
    print(f"mismatches: {len(mismatches)}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
