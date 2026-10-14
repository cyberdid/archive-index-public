"""Build and refresh the archive: parse sources, redact, chunk, embed, store."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .chunking import pack_paragraphs
from .redact import redact
from .sources import (
    Document, discover_audio, discover_claude_sessions, discover_images, discover_codex_sessions,
    discover_documents, discover_tables, parse_table, LEGACY_TARGET, discover_html, discover_text,
    discover_notebooks, discover_config, discover_logs, parse_legacy, parse_html, parse_text,
    discover_code, parse_code, parse_epub, discover_video, parse_video,
    parse_notebook, parse_config, parse_log,
    discover_git_repos, discover_markdown, discover_run_logs, parse_claude_session,
    parse_codex_session, parse_docx, parse_git_history, parse_markdown, parse_pdf,
    parse_pptx, parse_run_log,
)
from .store import ChunkRecord, Store

COLLECTIONS = ("wiki", "docs", "tables", "git", "sessions", "run_logs", "images", "audio", "video")
OPT_IN = ("config", "logs", "code")  # searched only when named explicitly with -c
ALL_COLLECTIONS = COLLECTIONS + OPT_IN
AUDIO_WINDOW_S = 20.0
AUDIO_STEP_S = 15.0


@dataclass
class IndexReport:
    indexed: dict[str, int] = field(default_factory=dict)
    skipped: int = 0
    chunks: int = 0
    reused_vectors: int = 0
    numeric_only: int = 0
    image_chunks: int = 0
    audio_chunks: int = 0
    redactions: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add_indexed(self, collection: str, chunks: int) -> None:
        self.indexed[collection] = self.indexed.get(collection, 0) + 1
        self.chunks += chunks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _body_sha(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


MIN_LETTERS = 20  # fragments of documents with fewer letters are page numbers and punctuation: no meaning
SHORT_FRAGMENT_COLLECTIONS = ("docs", "wiki")


def letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def drop_short_fragments(collection: str, pieces: list[tuple]) -> list[tuple]:
    """Remove text fragments with fewer than MIN_LETTERS letters, unless every text fragment is short.

    Images and sound are never dropped here. A document that is only short text keeps it, so short notes
    stay searchable.
    """
    if collection not in SHORT_FRAGMENT_COLLECTIONS:
        return pieces
    text = [p for p in pieces if not p[4] and p[5] is None]
    if not text or all(letter_count(p[3]) < MIN_LETTERS for p in text):
        return pieces
    return [p for p in pieces if p[4] or p[5] is not None or letter_count(p[3]) >= MIN_LETTERS]


def _make_chunks(doc: Document, max_chars: int, report: IndexReport) -> list[tuple]:
    """Return (heading, locator, occurred_at, body, image, audio) after redaction and packing."""
    out: list[tuple] = []
    for unit in doc.units:
        if unit.image:
            out.append((unit.heading, unit.locator, unit.occurred_at, unit.text, unit.image, None))
            continue
        if unit.audio is not None:
            out.append((unit.heading, unit.locator, unit.occurred_at, unit.text, "", unit.audio))
            continue
        clean, counts = redact(unit.text)
        for kind, n in counts.items():
            report.redactions[kind] = report.redactions.get(kind, 0) + n
        for piece in pack_paragraphs(clean, max_chars):
            out.append((unit.heading, unit.locator, unit.occurred_at, piece, "", None))
    return drop_short_fragments(doc.collection, out)


NUMERIC_SHARE = 0.6  # above this share of digits and symbols the text is numbers, not meaning


def is_numeric(text: str) -> bool:
    """True for tables and number dumps: mostly digits and symbols (plan A4)."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False
    numeric = sum(1 for ch in chars if ch.isdigit() or (not ch.isalnum()))
    return numeric / len(chars) > NUMERIC_SHARE


def _image_key(path: str) -> str:
    return "img:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _audio_key(samples) -> str:
    return "aud:" + hashlib.sha256(np.asarray(samples, dtype=np.float32).tobytes()).hexdigest()


def _records_for(embedder, cache: dict[str, bytes], doc: Document, pieces, report: IndexReport,
                 ord_base: int = 0) -> list[ChunkRecord]:
    """Embed only chunks whose text, image or sound is not already in cache; reuse the rest.

    Numeric chunks get an empty vector (dimension 0): they are found by exact words only.
    Rendered pages and video frames are embedded as images; sound windows as audio.
    """
    from .lexical import index_terms

    vectors: list[np.ndarray | None] = [None] * len(pieces)
    missing: list[int] = []
    keys: list[str] = []
    for i, (_, _, _, body, image, audio) in enumerate(pieces):
        if audio is not None:
            key = _audio_key(audio)
        elif image:
            key = _image_key(image)
        else:
            key = _body_sha(body)
        keys.append(key)
        if audio is None and not image and is_numeric(body):
            vectors[i] = np.zeros(0, dtype=np.float32)
            report.numeric_only += 1
            continue
        blob = cache.get(key)
        if blob is not None:
            vectors[i] = np.frombuffer(blob, dtype=np.float32).copy()
            report.reused_vectors += 1
        elif image:
            vectors[i] = embedder.image(image)
            report.image_chunks += 1
        elif audio is not None:
            vectors[i] = embedder.audio(audio)
            report.audio_chunks += 1
        else:
            missing.append(i)
    if missing:
        fresh = embedder.documents([pieces[i][3] for i in missing], [doc.title] * len(missing))
        for i, vector in zip(missing, fresh):
            vectors[i] = vector
    records = []
    for ord_, ((heading, locator, at, body, _image, _audio), vector, key) in enumerate(
            zip(pieces, vectors, keys)):
        records.append(ChunkRecord(
            ord=ord_base + ord_, heading=heading, locator=locator, body=body,
            terms=index_terms(f"{doc.title} {heading} {body}"),
            occurred_at=at, vector=vector, body_sha=key,
        ))
    return records


def index_document(
    store: Store,
    embedder,
    doc: Document,
    *,
    state_key: str,
    size: int,
    mtime_ns: int,
    sha: str,
    report: IndexReport,
    max_chars: int = 1200,
) -> None:
    from .lexical import index_terms

    pieces = _make_chunks(doc, max_chars, report)
    records = _records_for(embedder, store.vector_cache(state_key), doc, pieces, report)
    store.replace_document(
        collection=doc.collection, path=state_key, size=size, mtime_ns=mtime_ns, sha256=sha,
        title=doc.title, project=doc.project, chunks=records,
    )
    report.add_indexed(doc.collection, len(records))


def _head_sha(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read(4096)).hexdigest()


def _index_session(store: Store, embedder, path: Path, projects_root: Path,
                   report: IndexReport, max_chars: int, force: bool) -> None:
    """Index a Claude or Codex transcript, re-reading only what changed since the last run.

    A file whose first 4 KB are unchanged and that did not shrink is read from the saved
    resume point: the last user turn is re-read (its old chunks are trimmed first) and any
    new turns are appended. Unchanged chunk texts reuse their stored vectors.
    """
    parse = parse_claude_session if ".claude" in path.parts else parse_codex_session
    stat = path.stat()
    key = str(path)
    state = store.document_state(key)
    if state and not force and state.size == stat.st_size and state.mtime_ns == stat.st_mtime_ns:
        report.skipped += 1
        return
    head = _head_sha(path)
    if state and not force and state.sha256 == head and stat.st_size >= state.size:
        cache = store.vector_cache(key)
        doc, r_off, r_line = parse(path, projects_root, state.read_offset, state.read_line)
        store.trim_chunks_from_line(state.doc_id, state.read_line)
        pieces = _make_chunks(doc, max_chars, report)
        records = _records_for(embedder, cache, doc, pieces, report)
        store.append_chunks(state.doc_id, collection="sessions",
                            project=store.document_project(state.doc_id), chunks=records)
        store.set_resume_point(state.doc_id, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                               read_offset=r_off, read_line=r_line)
        report.add_indexed("sessions", len(records))
        return
    doc, r_off, r_line = parse(path, projects_root, 0, 1)
    index_document(store, embedder, doc, state_key=key, size=stat.st_size,
                   mtime_ns=stat.st_mtime_ns, sha=head, report=report, max_chars=max_chars)
    new_state = store.document_state(key)
    store.set_resume_point(new_state.doc_id, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                           read_offset=r_off, read_line=r_line)


PDF_PASS_PAGES = 30  # pages per indexing pass; a large PDF is finished over several runs (plan D2)


def _page_of(locator: str) -> int:
    import re

    match = re.search(r"page (\d+)", locator)
    return int(match.group(1)) if match else 0


def _pdf_pages(path: Path) -> int:
    """Page count from pdfinfo: cheap, so an unchanged PDF is not parsed again on every run."""
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=60).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return 0


def _index_pdf_staged(store: Store, embedder, path: Path, projects_root: Path,
                      report: IndexReport, max_chars: int, force: bool) -> None:
    """Index a PDF in passes of PDF_PASS_PAGES pages; read_line is the last page covered.

    An unchanged PDF gets only the fragments whose page is not in the index yet, so a repeated
    pass never duplicates a page. A complete, unchanged PDF is skipped without being parsed.
    """
    stat = path.stat()
    key = str(path)
    state = store.document_state(key)
    pages = _pdf_pages(path)
    if state and not force and state.size == stat.st_size and state.mtime_ns == stat.st_mtime_ns \
            and pages and state.read_line >= pages:
        report.skipped += 1
        return
    sha = _sha256(path)
    unchanged = state is not None and not force and state.sha256 == sha and state.size == stat.st_size
    if unchanged and pages and state.read_line >= pages:
        if state.mtime_ns != stat.st_mtime_ns:
            store.touch_document(state.doc_id, stat.st_mtime_ns)
        report.skipped += 1
        return
    done = state.read_line if unchanged else 0
    target = min(pages, done + PDF_PASS_PAGES) if pages else done + PDF_PASS_PAGES
    doc = parse_pdf(path, projects_root, max_page=target)  # render only this pass's image pages
    upto = [u for u in doc.units if _page_of(u.locator) <= target]
    if unchanged:
        have = store.locators(state.doc_id)
        fresh = [u for u in upto if u.locator not in have]
        part = Document(doc.collection, doc.path, doc.title, doc.project, fresh)
        pieces = _make_chunks(part, max_chars, report)
        records = _records_for(embedder, store.vector_cache(key), part, pieces, report)
        if records:
            store.append_chunks(state.doc_id, collection=part.collection,
                                project=store.document_project(state.doc_id), chunks=records)
    else:
        part = Document(doc.collection, doc.path, doc.title, doc.project, upto)
        pieces = _make_chunks(part, max_chars, report)
        records = _records_for(embedder, store.vector_cache(key), part, pieces, report)
        store.replace_document(collection=part.collection, path=key, size=stat.st_size,
                               mtime_ns=stat.st_mtime_ns, sha256=sha, title=part.title,
                               project=part.project, chunks=records)
    report.add_indexed("docs", len(records))
    new_state = store.document_state(key)
    store.set_resume_point(new_state.doc_id, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                           read_offset=0, read_line=target)


def _index_file(
    store: Store, embedder, path: Path, parse: Callable[[Path], Document],
    report: IndexReport, max_chars: int, force: bool,
) -> None:
    stat = path.stat()
    state = store.document_state(str(path))
    if state and not force and state.size == stat.st_size and state.mtime_ns == stat.st_mtime_ns:
        report.skipped += 1
        return
    sha = _sha256(path)
    if state and not force and state.sha256 == sha:
        store.touch_document(state.doc_id, stat.st_mtime_ns)
        report.skipped += 1
        return
    doc = parse(path)
    index_document(
        store, embedder, doc, state_key=str(path), size=stat.st_size,
        mtime_ns=stat.st_mtime_ns, sha=sha, report=report, max_chars=max_chars,
    )


def _index_git(store: Store, embedder, repo: Path, projects_root: Path,
               report: IndexReport, max_chars: int, force: bool) -> None:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=30).stdout.strip()
    if not head:
        return
    key = str(repo / ".git")
    state = store.document_state(key)
    if state and not force and state.sha256 == head:
        report.skipped += 1
        return
    doc = parse_git_history(repo, projects_root)
    index_document(store, embedder, doc, state_key=key, size=0, mtime_ns=0,
                   sha=head, report=report, max_chars=max_chars)


def _index_image(store: Store, embedder, path: Path, projects_root: Path,
                 report: IndexReport, force: bool) -> None:
    from .lexical import index_terms
    from .sources import project_of

    stat = path.stat()
    key = str(path)
    state = store.document_state(key)
    if state and not force and state.size == stat.st_size and state.mtime_ns == stat.st_mtime_ns:
        report.skipped += 1
        return
    sha = _sha256(path)
    if state and not force and state.sha256 == sha:
        store.touch_document(state.doc_id, stat.st_mtime_ns)
        report.skipped += 1
        return
    rel = str(path.resolve().relative_to(projects_root.resolve())) if path.is_absolute() else str(path)
    vector = embedder.image(key)
    record = ChunkRecord(
        ord=0, heading="", locator="image", body=f"image: {rel}",
        terms=index_terms(rel.replace("/", " ").replace("_", " ")),
        occurred_at="", vector=vector, body_sha=sha,
    )
    store.replace_document(
        collection="images", path=key, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        sha256=sha, title=path.name, project=project_of(path, projects_root), chunks=[record],
    )
    report.add_indexed("images", 1)


def _index_audio(store: Store, embedder, path: Path, projects_root: Path,
                 report: IndexReport, force: bool) -> None:
    import numpy as np
    from .audio_io import read_audio, window_ranges
    from .lexical import index_terms
    from .sources import _transcript_for_audio, parse_audio_transcript_cues, project_of

    stat = path.stat()
    transcript_path = _transcript_for_audio(path)
    t_stat = transcript_path.stat() if transcript_path else None
    doc_mtime_ns = max(stat.st_mtime_ns, t_stat.st_mtime_ns) if t_stat else stat.st_mtime_ns
    key = str(path)
    state = store.document_state(key)
    if state and not force and state.size == stat.st_size and state.mtime_ns == doc_mtime_ns:
        report.skipped += 1
        return
    audio_sha = _sha256(path)
    doc_sha = (
        hashlib.sha256(f"{audio_sha}:{_sha256(transcript_path)}".encode()).hexdigest()
        if transcript_path
        else audio_sha
    )
    if state and not force and state.sha256 == doc_sha:
        store.touch_document(state.doc_id, doc_mtime_ns)
        report.skipped += 1
        return
    cache = store.vector_cache(key)
    cues = parse_audio_transcript_cues(path)
    samples = read_audio(path)
    duration_s = len(samples) / 16_000
    records: list[ChunkRecord] = []
    stem_words = path.stem.replace("_", " ")
    for ord_, (start_s, end_s) in enumerate(window_ranges(duration_s, AUDIO_WINDOW_S, AUDIO_STEP_S)):
        chunk_sha = hashlib.sha256(f"{audio_sha}:{ord_}".encode()).hexdigest()
        if chunk_sha in cache:
            vector = np.frombuffer(cache[chunk_sha], dtype=np.float32).copy()
            report.reused_vectors += 1
        else:
            start, end = int(start_s * 16_000), min(len(samples), int(end_s * 16_000))
            vector = embedder.audio(np.asarray(samples[start:end], dtype=np.float32))
        overlap_lines = [text for cs, ce, text in cues if start_s < ce and cs < end_s]
        if overlap_lines:
            clean_text, r_counts = redact(" ".join(overlap_lines))
            for kind, n in r_counts.items():
                report.redactions[kind] = report.redactions.get(kind, 0) + n
            transcript_text = clean_text.strip()
        else:
            transcript_text = ""
        header = f"audio: {path.name} at {start_s:.0f}-{end_s:.0f}s"
        body = f"{header}\n{transcript_text}" if transcript_text else header
        term_source = f"{stem_words} {transcript_text}" if transcript_text else stem_words
        records.append(ChunkRecord(
            ord=ord_, heading="", locator=f"{start_s:.0f}-{end_s:.0f}s",
            body=body,
            terms=index_terms(term_source),
            occurred_at="", vector=vector,
            body_sha=chunk_sha,
        ))
    store.replace_document(
        collection="audio", path=key, size=stat.st_size, mtime_ns=doc_mtime_ns,
        sha256=doc_sha, title=path.name, project=project_of(path, projects_root), chunks=records,
    )
    report.add_indexed("audio", len(records))


def run_index(
    store: Store,
    embedder,
    *,
    projects_root: Path,
    home: Path,
    collections: tuple[str, ...] = COLLECTIONS,
    max_chars: int = 1200,
    session_limit: int | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
    code_projects: set[str] | None = None,
    skip_screenshots: bool = False,
) -> IndexReport:
    report = IndexReport()
    say = progress or (lambda _msg: None)
    errors = report.errors

    def guarded(label: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # one bad file must not stop the whole run
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if "wiki" in collections:
        present: set[str] = set()
        for path in discover_markdown(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_markdown(q, projects_root),
                report, max_chars, force))
        say(f"wiki: {len(present)} markdown files seen")
        if session_limit is None:
            store.remove_missing("wiki", present)

    if "docs" in collections:
        parsers = {".pdf": parse_pdf, ".docx": parse_docx, ".pptx": parse_pptx, ".epub": parse_epub}
        parsers.update({suffix: parse_legacy for suffix in LEGACY_TARGET})
        present = set()
        # Small files first, so new documents show up in search quickly; large ones follow.
        for path in sorted(discover_documents(projects_root), key=lambda p: p.stat().st_size):
            present.add(str(path))
            if path.suffix.lower() == ".pdf":
                guarded(str(path), lambda p=path: _index_pdf_staged(
                    store, embedder, p, projects_root, report, max_chars, force))
                continue
            parser = parsers[path.suffix.lower()]
            guarded(str(path), lambda p=path, f=parser: _index_file(
                store, embedder, p, lambda q: f(q, projects_root),
                report, max_chars, force))
        for path in discover_html(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_html(q, projects_root),
                report, max_chars, force))
        for path in discover_text(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_text(q, projects_root),
                report, max_chars, force))
        for path in discover_notebooks(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_notebook(q, projects_root),
                report, max_chars, force))
        say(f"docs: {len(present)} documents seen")
        if session_limit is None:
            store.remove_missing("docs", present)

    if "config" in collections:
        present = set()
        for path in discover_config(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_config(q, projects_root),
                report, max_chars, force))
        say(f"config: {len(present)} files seen")
        if session_limit is None:
            store.remove_missing("config", present)

    if "code" in collections:
        if not code_projects:
            say("code: skipped, name the projects with --project (code is opt-in per project)")
        else:
            present = set()
            for path in discover_code(projects_root, code_projects):
                present.add(str(path))
                guarded(str(path), lambda p=path: _index_file(
                    store, embedder, p, lambda q: parse_code(q, projects_root),
                    report, max_chars, force))
            say(f"code: {len(present)} source files seen for {sorted(code_projects)}")
            if session_limit is None:
                store.remove_missing("code", present)

    if "logs" in collections:
        present = set()
        for path in discover_logs(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_log(q, projects_root),
                report, max_chars, force))
        say(f"logs: {len(present)} files seen")
        if session_limit is None:
            store.remove_missing("logs", present)

    if "tables" in collections:
        present = set()
        for path in discover_tables(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_table(q, projects_root),
                report, max_chars, force))
        say(f"tables: {len(present)} spreadsheets seen")
        if session_limit is None:
            store.remove_missing("tables", present)

    if "video" in collections:
        present = set()
        for path in discover_video(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_video(q, projects_root),
                report, max_chars, force))
        say(f"video: {len(present)} videos seen")
        if session_limit is None:
            store.remove_missing("video", present)

    if "git" in collections:
        repos = discover_git_repos(projects_root)
        for repo in repos:
            guarded(str(repo), lambda r=repo: _index_git(
                store, embedder, r, projects_root, report, max_chars, force))
        say(f"git: {len(repos)} repositories seen")

    if "sessions" in collections:
        files = discover_claude_sessions(home) + discover_codex_sessions(home)
        if session_limit is not None:
            files = files[:session_limit]
        for path in files:
            guarded(str(path), lambda p=path: _index_session(
                store, embedder, p, projects_root, report, max_chars, force))
        if session_limit is None:
            # Remove transcripts that no longer exist on disk (e.g. deleted sessions).
            store.remove_missing("sessions", {str(p) for p in files})
        say(f"sessions: {len(files)} session files seen")

    if "images" in collections:
        present = set()
        for path in discover_images(projects_root, skip_screenshots=skip_screenshots):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_image(
                store, embedder, p, projects_root, report, force))
        say(f"images: {len(present)} images seen")
        if session_limit is None:
            store.remove_missing("images", present)

    if "audio" in collections:
        present = set()
        for path in discover_audio(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_audio(
                store, embedder, p, projects_root, report, force))
        say(f"audio: {len(present)} recordings seen")
        if session_limit is None:
            store.remove_missing("audio", present)

    if "run_logs" in collections:
        present = set()
        for path in discover_run_logs(projects_root):
            present.add(str(path))
            guarded(str(path), lambda p=path: _index_file(
                store, embedder, p, lambda q: parse_run_log(q, projects_root),
                report, max_chars, force))
        say(f"run_logs: {len(present)} run logs seen")
        if session_limit is None:
            store.remove_missing("run_logs", present)

    return report
