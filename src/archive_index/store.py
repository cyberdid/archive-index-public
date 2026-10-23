"""SQLite storage: documents, chunks, FTS5 lexical index and float32 vectors."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    collection TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    project TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    project TEXT NOT NULL,
    ord INTEGER NOT NULL,
    heading TEXT NOT NULL,
    locator TEXT NOT NULL,
    body TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    body_sha TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_collection ON chunks(collection);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    terms, body, heading, tokenize = 'unicode61 remove_diacritics 0'
);
"""


@dataclass(frozen=True)
class ChunkRecord:
    ord: int
    heading: str
    locator: str
    body: str
    terms: str
    occurred_at: str
    vector: np.ndarray
    body_sha: str = ""


@dataclass(frozen=True)
class DocumentState:
    doc_id: int
    size: int
    mtime_ns: int
    sha256: str
    read_offset: int = 0
    read_line: int = 1


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: int
    collection: str
    project: str
    title: str
    path: str
    locator: str
    heading: str
    body: str
    occurred_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        # Several programs share this database (the interface, the timers, the indexer). WAL lets readers
        # and the writer run together; the long timeout waits for a short lock instead of failing.
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()
        if created:
            os.chmod(self.path, 0o600)  # transcripts can contain local content

    def _migrate(self) -> None:
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(chunks)")}
        if "body_sha" not in columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN body_sha TEXT NOT NULL DEFAULT ''")
        doc_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(documents)")}
        if "read_offset" not in doc_columns:
            self.conn.execute("ALTER TABLE documents ADD COLUMN read_offset INTEGER NOT NULL DEFAULT 0")
        if "read_line" not in doc_columns:
            self.conn.execute("ALTER TABLE documents ADD COLUMN read_line INTEGER NOT NULL DEFAULT 1")
        self.conn.execute("CREATE INDEX IF NOT EXISTS chunks_body_sha ON chunks(body_sha)")

    def vector_cache(self, path: str) -> dict[str, bytes]:
        """Vectors of already-embedded chunks of one document, keyed by body hash."""
        rows = self.conn.execute(
            "SELECT c.body_sha, c.vector FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " WHERE d.path = ? AND c.body_sha != ''",
            (path,),
        ).fetchall()
        return {sha: blob for sha, blob in rows}

    def close(self) -> None:
        self.conn.close()

    # documents -------------------------------------------------------------

    def document_state(self, path: str) -> DocumentState | None:
        row = self.conn.execute(
            "SELECT id, size, mtime_ns, sha256, read_offset, read_line FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        return DocumentState(*row) if row else None

    def set_resume_point(self, doc_id: int, *, size: int, mtime_ns: int,
                         read_offset: int, read_line: int) -> None:
        self.conn.execute(
            "UPDATE documents SET size = ?, mtime_ns = ?, read_offset = ?, read_line = ? WHERE id = ?",
            (size, mtime_ns, read_offset, read_line, doc_id),
        )
        self.conn.commit()

    def trim_chunks_from_line(self, doc_id: int, line: int) -> None:
        """Delete chunks whose locator is 'line N' with N >= line (the turn being re-read)."""
        ids = [r[0] for r in self.conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ? AND locator LIKE 'line %'"
            " AND CAST(SUBSTR(locator, 6) AS INTEGER) >= ?",
            (doc_id, line),
        )]
        self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
        self.conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in ids])
        self.conn.commit()

    def append_chunks(self, doc_id: int, *, collection: str, project: str,
                      chunks: list[ChunkRecord]) -> None:
        cur = self.conn.cursor()
        start = cur.execute("SELECT COALESCE(MAX(ord), -1) + 1 FROM chunks WHERE doc_id = ?",
                            (doc_id,)).fetchone()[0]
        for offset, chunk in enumerate(chunks):
            vector = np.asarray(chunk.vector, dtype=np.float32)
            cur.execute(
                "INSERT INTO chunks (doc_id, collection, project, ord, heading, locator, body,"
                " occurred_at, dimension, vector, body_sha) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, collection, project, start + offset, chunk.heading, chunk.locator,
                 chunk.body, chunk.occurred_at, vector.size, vector.tobytes(), chunk.body_sha),
            )
            cur.execute("INSERT INTO chunks_fts (rowid, terms, body, heading) VALUES (?, ?, ?, ?)",
                        (cur.lastrowid, chunk.terms, chunk.body, chunk.heading))
        self.conn.commit()

    def document_project(self, doc_id: int) -> str:
        row = self.conn.execute("SELECT project FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row[0] if row else ""

    def touch_document(self, doc_id: int, mtime_ns: int) -> None:
        self.conn.execute("UPDATE documents SET mtime_ns = ? WHERE id = ?", (mtime_ns, doc_id))
        self.conn.commit()

    def replace_document(
        self,
        *,
        collection: str,
        path: str,
        size: int,
        mtime_ns: int,
        sha256: str,
        title: str,
        project: str,
        chunks: list[ChunkRecord],
    ) -> int:
        """Insert or replace one document and all of its chunks atomically."""
        cur = self.conn.cursor()
        old = cur.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
        if old:
            self._delete_chunks(old[0])
            cur.execute("DELETE FROM documents WHERE id = ?", (old[0],))
        cur.execute(
            "INSERT INTO documents (collection, path, size, mtime_ns, sha256, title, project, indexed_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (collection, path, size, mtime_ns, sha256, title, project, _now()),
        )
        doc_id = cur.lastrowid
        for chunk in chunks:
            vector = np.asarray(chunk.vector, dtype=np.float32)
            cur.execute(
                "INSERT INTO chunks (doc_id, collection, project, ord, heading, locator, body,"
                " occurred_at, dimension, vector, body_sha) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id, collection, project, chunk.ord, chunk.heading, chunk.locator,
                    chunk.body, chunk.occurred_at, vector.size, vector.tobytes(), chunk.body_sha,
                ),
            )
            chunk_id = cur.lastrowid
            cur.execute(
                "INSERT INTO chunks_fts (rowid, terms, body, heading) VALUES (?, ?, ?, ?)",
                (chunk_id, chunk.terms, chunk.body, chunk.heading),
            )
        self.conn.commit()
        return doc_id

    def locators(self, doc_id: int) -> set[str]:
        """Locators (page or section labels) already stored for a document."""
        return {r[0] for r in self.conn.execute("SELECT locator FROM chunks WHERE doc_id = ?", (doc_id,))}

    def remove_short_fragments(self, collections: tuple[str, ...], min_letters: int) -> int:
        """Delete text fragments with fewer letters than min_letters, in documents that have longer ones too."""
        removed = 0
        marks = ",".join("?" for _ in collections)
        doc_ids = [r[0] for r in self.conn.execute(
            f"SELECT id FROM documents WHERE collection IN ({marks})", collections)]
        for doc_id in doc_ids:
            rows = self.conn.execute(
                "SELECT id, body FROM chunks WHERE doc_id = ? AND locator NOT LIKE '%(image)%'",
                (doc_id,)).fetchall()
            short = [cid for cid, body in rows if sum(ch.isalpha() for ch in body) < min_letters]
            if not short or len(short) == len(rows):
                continue  # all short (keep the document) or nothing short
            self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in short])
            self.conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in short])
            removed += len(short)
        self.conn.commit()
        return removed

    def remove_duplicate_chunks(self, collection: str) -> int:
        """Delete repeated fragments: same document, locator and text, keeping the first one."""
        ids = [r[0] for r in self.conn.execute(
            """SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
               WHERE d.collection = ? AND EXISTS (
                   SELECT 1 FROM chunks o WHERE o.doc_id = c.doc_id AND o.locator = c.locator
                   AND o.body = c.body AND o.id < c.id)""", (collection,))]
        self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
        self.conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in ids])
        self.conn.commit()
        return len(ids)

    def _delete_chunks(self, doc_id: int) -> None:
        ids = [r[0] for r in self.conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
        self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def remove_missing(self, collection: str, present_paths: set[str]) -> int:
        """Drop documents of a collection whose source file no longer exists."""
        rows = self.conn.execute(
            "SELECT id, path FROM documents WHERE collection = ?", (collection,)
        ).fetchall()
        removed = 0
        for doc_id, path in rows:
            if path not in present_paths:
                self._delete_chunks(doc_id)
                self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                removed += 1
        self.conn.commit()
        return removed

    # search ----------------------------------------------------------------

    def lexical(
        self,
        match: str,
        *,
        collections: list[str] | None,
        project: str | None,
        limit: int,
        suffixes: tuple[str, ...] | None = None,
    ) -> list[tuple[int, float]]:
        if not match:
            return []
        sql = (
            "SELECT f.rowid, bm25(chunks_fts) AS score FROM chunks_fts f"
            " JOIN chunks c ON c.id = f.rowid WHERE chunks_fts MATCH ?"
        )
        params: list[object] = [match]
        sql, params = _apply_filters(sql, params, collections, project, suffixes)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        return [(int(r[0]), float(r[1])) for r in self.conn.execute(sql, params)]

    def chunk_ids_with_suffixes(self, suffixes: tuple[str, ...]) -> set[int]:
        """Chunk ids whose document path ends with one of the suffixes (format filter for vector search)."""
        sql = "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE " + " OR ".join(
            "lower(d.path) LIKE ?" for _ in suffixes)
        return {row[0] for row in self.conn.execute(sql, [f"%{s}" for s in suffixes])}

    def count_lexical(self, match: str, *, collections: list[str] | None, project: str | None,
                      suffixes: tuple[str, ...] | None = None) -> int:
        """Exact number of chunks that match the word query under the same filters."""
        if not match:
            return 0
        sql = (
            "SELECT COUNT(*) FROM chunks_fts f JOIN chunks c ON c.id = f.rowid"
            " WHERE chunks_fts MATCH ?"
        )
        params: list[object] = [match]
        sql, params = _apply_filters(sql, params, collections, project, suffixes)
        return int(self.conn.execute(sql, params).fetchone()[0])

    def rebuild_terms(self, make_terms, batch: int = 2000) -> int:
        """Recompute the FTS5 terms column from stored text, from the same source indexing used.
        Vectors are not touched."""
        from .indexer import chunk_term_source

        total = 0
        last = 0
        while True:
            rows = self.conn.execute(
                "SELECT c.id, d.collection, d.path, d.title, c.heading, c.body"
                " FROM chunks c JOIN documents d ON d.id = c.doc_id"
                " WHERE c.id > ? ORDER BY c.id LIMIT ?", (last, batch),
            ).fetchall()
            if not rows:
                break
            self.conn.executemany(
                "UPDATE chunks_fts SET terms = ? WHERE rowid = ?",
                [(make_terms(chunk_term_source(collection, path, title, heading, body)), cid)
                 for cid, collection, path, title, heading, body in rows],
            )
            self.conn.commit()
            total += len(rows)
            last = rows[-1][0]
        return total

    def all_vectors(self) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Return (chunk_ids, matrix, collections, projects) for every chunk."""
        rows = self.conn.execute(
            "SELECT id, dimension, vector, collection, project FROM chunks WHERE dimension > 0 ORDER BY id"
        ).fetchall()
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32), [], []
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        matrix = np.vstack([np.frombuffer(r[2], dtype=np.float32, count=r[1]) for r in rows])
        return ids, matrix, [r[3] for r in rows], [r[4] for r in rows]

    def chunk_rows(self, ids: list[int]) -> dict[int, ChunkRow]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            "SELECT c.id, c.collection, c.project, d.title, d.path, c.locator, c.heading,"
            " c.body, c.occurred_at FROM chunks c JOIN documents d ON d.id = c.doc_id"
            f" WHERE c.id IN ({marks})",
            ids,
        ).fetchall()
        return {r[0]: ChunkRow(*r) for r in rows}

    def stats(self) -> dict[str, dict[str, int]]:
        rows = self.conn.execute(
            "SELECT d.collection, COUNT(DISTINCT d.id), COUNT(c.id) FROM documents d"
            " LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.collection ORDER BY d.collection"
        ).fetchall()
        return {r[0]: {"documents": r[1], "chunks": r[2]} for r in rows}


def _apply_filters(
    sql: str, params: list[object], collections: list[str] | None, project: str | None,
    suffixes: tuple[str, ...] | None = None,
) -> tuple[str, list[object]]:
    if collections:
        sql += " AND c.collection IN (" + ",".join("?" for _ in collections) + ")"
        params.extend(collections)
    if project is not None:
        sql += " AND c.project LIKE ?"
        params.append(f"{project}%")
    if suffixes:  # format filter applied to the candidates, not to the top results
        sql += " AND c.doc_id IN (SELECT id FROM documents WHERE " + " OR ".join(
            "lower(path) LIKE ?" for _ in suffixes) + ")"
        params.extend(f"%{suffix}" for suffix in suffixes)
    return sql, params
