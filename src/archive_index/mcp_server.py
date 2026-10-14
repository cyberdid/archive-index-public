"""MCP server exposing the archive to Claude Code and other MCP clients (stdio)."""

from __future__ import annotations

from pathlib import Path

from .indexer import ALL_COLLECTIONS
from .search import VectorCache, hybrid
from .store import Store

DEFAULT_DB = Path.home() / ".local" / "share" / "archive-index" / "archive.sqlite3"


class ArchiveService:
    """Holds one store, one vector cache and one embedder for the server's lifetime."""

    def __init__(self, db_path: Path = DEFAULT_DB, embedder=None) -> None:
        self.store = Store(db_path)
        self.cache = VectorCache(self.store)
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import Embedder

            self._embedder = Embedder()
        return self._embedder

    def search(self, query: str, collections: list[str] | None = None,
               project: str | None = None, limit: int = 8, mode: str = "hybrid") -> list[dict]:
        if collections:
            unknown = set(collections) - set(ALL_COLLECTIONS)
            if unknown:
                raise ValueError(f"unknown collections: {sorted(unknown)}")
        qv = self.embedder.query(query) if mode != "lexical" else None
        hits = hybrid(self.store, self.cache, query, qv, collections=collections,
                      project=project, k=max(1, min(limit, 50)), mode=mode)
        return [h.__dict__ for h in hits]

    def stats(self) -> dict:
        return self.store.stats()


def build_server(db_path: Path = DEFAULT_DB, embedder=None):
    from mcp.server.fastmcp import FastMCP

    service = ArchiveService(db_path, embedder)
    mcp = FastMCP("archive-index")

    @mcp.tool()
    def archive_search(query: str, collections: list[str] | None = None,
                       project: str | None = None, limit: int = 8,
                       mode: str = "hybrid") -> list[dict]:
        """Search local project files, wiki, docs, tables, git, sessions, run_logs, images, audio, video, code, config, logs.

        collections: any of wiki, docs, tables, git, sessions, run_logs, images, audio, video, code, config, logs (default: all active).
        project: project folder prefix under ~/Projects, e.g. "llm" or "projalpha".
        mode: hybrid (default), lexical (exact words) or semantic (meaning).
        """
        return service.search(query, collections, project, limit, mode)

    @mcp.tool()
    def archive_stats() -> dict:
        """Document and chunk counts per collection."""
        return service.stats()

    mcp.service = service  # type: ignore[attr-defined]
    return mcp


if __name__ == "__main__":
    build_server().run()
