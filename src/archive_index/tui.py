"""Terminal UI for the archive: search, filter, read, open. Run with `archive-tui`."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Static

from .indexer import ALL_COLLECTIONS
from .lexical import query_expression, tokenize
from .logs import get_logger
from .search import Hit, VectorCache, hybrid
from .store import Store

MODES = ("hybrid", "lexical", "semantic")
FORMATS = ("все", "pdf", "docx", "pptx", "xlsx", "csv", "html", "md", "txt", "png", "audio")
AUDIO_EXT = (".wav", ".mp3", ".opus", ".ogg", ".flac", ".m4a")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".avi")  # opened at the second too (frames and sound windows)
CODE_EXT = (".py", ".c", ".h", ".cpp", ".ts", ".js", ".rs", ".go", ".sh")


FORMAT_SUFFIXES = {
    "pdf": (".pdf",), "docx": (".docx",), "pptx": (".pptx",), "xlsx": (".xlsx", ".xlsm", ".xls", ".ods"),
    "csv": (".csv", ".tsv"), "html": (".html", ".htm"), "md": (".md",), "txt": (".txt",),
    "png": (".png", ".jpg", ".jpeg", ".webp"),
}


def format_suffixes(fmt: str) -> tuple[str, ...] | None:
    """File suffixes of a format name; None for 'все'. The filter runs in the search, not after it."""
    if fmt == "все":
        return None
    if fmt == "audio":
        return AUDIO_EXT
    return FORMAT_SUFFIXES.get(fmt, ("." + fmt,))


def format_matches(path: str, fmt: str) -> bool:
    suffixes = format_suffixes(fmt)
    return suffixes is None or path.lower().endswith(suffixes)


PLAYER = "mpv"          # plays from a given second (--start), as the plan names
FALLBACK_PLAYER = "gst-play-1.0"  # plays from a given second (-s, -a for an exact seek)


def _positioned_player() -> tuple[str, list[str]] | None:
    """(program, arguments before the path) for the first installed player that can start at a second."""
    if shutil.which(PLAYER):
        return PLAYER, ["--start={second}"]
    if shutil.which(FALLBACK_PLAYER):
        return FALLBACK_PLAYER, ["-a", "-s", "{second}"]
    return None
CLIP_CACHE = Path.home() / ".cache" / "archive-index" / "clips"


def audio_span(locator: str) -> tuple[float, float] | None:
    """'42-62s' -> (42.0, 62.0)."""
    import re

    match = re.fullmatch(r"(\d+)-(\d+)s", locator.strip())
    return (float(match.group(1)), float(match.group(2))) if match else None


def media_second(path: str, locator: str) -> float | None:
    """Second to start playback at: a sound window ('42-62s') or a video frame ('frame 01:25')."""
    import re

    lowered = path.lower()
    if lowered.endswith(AUDIO_EXT + VIDEO_EXT):
        span = audio_span(locator)
        if span:
            return span[0]
    frame = re.fullmatch(r"frame (\d+):(\d+)", locator.strip())
    if frame and lowered.endswith(VIDEO_EXT):
        return float(int(frame.group(1)) * 60 + int(frame.group(2)))
    return None


def audio_clip(path: str, locator: str) -> Path | None:
    """Cut the matched window out of the recording so a plain player starts at the match."""
    import hashlib
    import wave

    import numpy as np
    from .audio_io import read_audio

    span = audio_span(locator)
    if span is None:
        return None
    start, end = span
    key = hashlib.sha256(f"{path}|{start}|{end}".encode()).hexdigest()[:24]
    out = CLIP_CACHE / f"{key}.wav"
    if out.exists():
        return out
    samples = np.asarray(read_audio(Path(path)), dtype=np.float32)
    piece = samples[int(start * 16000): int(end * 16000)]
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(piece, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return out


def browser_command(path: Path) -> list[str]:
    """Command that opens an HTML file. ARCHIVE_BROWSER names the browser (e.g. google-chrome); by default the
    system handler is used, which may be an app that takes focus from the terminal."""
    import os

    browser = os.environ.get("ARCHIVE_BROWSER", "").strip()
    if browser:
        return [browser, str(path)]
    return ["xdg-open", str(path)]


def open_command(path: str, locator: str) -> list[str]:
    """Program and arguments that open the file at the place of the match."""
    lowered = path.lower()
    second = media_second(path, locator)
    player = _positioned_player()
    if second is not None and player is not None:
        program, args = player
        return [program, *[a.format(second=int(second)) for a in args], path]  # -a: accurate seek, not the previous keyframe
    page = locator.split()[1] if locator.startswith("page ") else ""
    page = page.split("(")[0].strip() if page else ""
    if lowered.endswith(".pdf") and page.isdigit():
        return ["evince", f"--page-index={page}", path]
    if lowered.endswith(CODE_EXT) and locator.startswith("line ") and locator[5:].isdigit():
        return ["code", "-g", f"{path}:{locator[5:]}"]
    return ["xdg-open", path]


def open_target(path: str, locator: str) -> str:
    """The file to hand to the opener. Without a positioned player, audio spans become a cut clip."""
    if path.lower().endswith(AUDIO_EXT) and audio_span(locator) and _positioned_player() is None:  # clip only for audio
        clip = audio_clip(path, locator)
        if clip is not None:
            return str(clip)
    return path
PER_DOCUMENT = 3     # fragments shown per document before grouping ("+N" shows the rest)
DEFAULT_LIMIT = 200  # results shown per query when the limit field is empty (the table scrolls)
MAX_LIMIT = 5000
FILTERS = ("all",) + ALL_COLLECTIONS
FORMAT_KEY_NOTE = "формат: f"
DEFAULT_DB = Path.home() / ".local" / "share" / "archive-index" / "archive.sqlite3"
log = get_logger("tui")


class Backend:
    """Everything that touches the database or the model. Used from one thread only."""

    def __init__(self, db_path: Path, embedder=None) -> None:
        self.db_path = db_path
        self.store = Store(db_path)
        self.cache = VectorCache(self.store)
        self._embedder = embedder
        self._stamp = None

    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import make_embedder

            self._embedder = make_embedder()
        return self._embedder

    def _refresh_if_changed(self) -> None:
        stamp = self.store.conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
        if stamp != self._stamp:
            self.cache.refresh()
            self._stamp = stamp

    def search(self, query: str, mode: str, collection: str, project: str,
               min_similarity: float = 0.0, limit: int = DEFAULT_LIMIT, fmt: str = "все"):
        """Return (hits, seconds, bodies, lexical_total). lexical_total is None in semantic mode."""
        started = time.monotonic()
        self._refresh_if_changed()
        log.debug("backend search start mode=%s collection=%s", mode, collection)
        collections = None if collection == "all" else [collection]
        project_filter = project or None
        lexical_total = None
        suffixes = format_suffixes(fmt)
        if mode != "semantic":
            lexical_total = self.store.count_lexical(query_expression(query), collections=collections,
                                                     project=project_filter, suffixes=suffixes)
        if mode == "hybrid" and lexical_total == 0:
            # No word of the query is in the archive. Showing the nearest vectors here would look like answers,
            # so the list stays empty; the user asks for meaning-only results with the 's' key.
            elapsed = time.monotonic() - started
            log.info("no word matches query=%r; semantic results need the s key", query)
            return [], elapsed, {}, lexical_total
        from .embed import query_prompt

        qv = None if mode == "lexical" else self.embedder.query(
            query, prompt=query_prompt([collection] if collection not in (None, "", "all") else []))
        pool = max(limit * 2, 100)
        hits = hybrid(self.store, self.cache, query, qv, collections=collections,
                      project=project_filter, k=limit, pool=pool, mode=mode,
                      per_document=PER_DOCUMENT, suffixes=suffixes)
        if min_similarity > 0:
            # Hits found only by words have no similarity and are kept; meaning-hits must pass the bar.
            hits = [h for h in hits if h.similarity is None or h.similarity >= min_similarity]
        bodies = {hit.chunk_id: self.body(hit.chunk_id) for hit in hits}
        elapsed = time.monotonic() - started
        log.debug("backend search end hits=%d seconds=%.3f", len(hits), elapsed)
        return hits, elapsed, bodies, lexical_total

    def body(self, chunk_id: int) -> str:
        row = self.store.chunk_rows([chunk_id]).get(chunk_id)
        return row.body if row else ""

    def graph_data(self, query: str, hits, mode: str):
        """Nodes and links of the shown results (the same data for the picture and the text view)."""
        import numpy as np

        from .viz import build_graph, fetch_vectors

        qv = None if mode == "lexical" else np.asarray(self.embedder.query(query), dtype=np.float32)
        vectors = fetch_vectors(self.store, [h.chunk_id for h in hits])
        return build_graph(query, hits, vectors, qv, near=0.8)

    def make_graph(self, query: str, hits, mode: str) -> Path:
        """Write the neuron picture of the shown results to an HTML file and return its path."""
        import time

        from .viz import render

        nodes, links = self.graph_data(query, hits, mode)
        out = self.db_path.parent / "graphs" / f"{time.strftime('%Y%m%d-%H%M%S')}.html"
        render(query, nodes, links, out)
        log.info("graph written %s hits=%d links=%d", out, len(hits), len(links))
        return out

    def document_fragments(self, path: str, limit: int = 200) -> list[tuple[str, str]]:
        """(locator, text) of every fragment of one document, in reading order (plan A1 expansion)."""
        rows = self.store.conn.execute(
            """SELECT c.locator, c.body FROM chunks c JOIN documents d ON d.id = c.doc_id
               WHERE d.path = ? ORDER BY c.ord LIMIT ?""", (path, limit)).fetchall()
        return [(locator, body) for locator, body in rows]


class GraphScreen(ModalScreen):
    """The graph as text, on its own screen: Esc or q closes it, arrows scroll."""

    BINDINGS = [Binding("escape", "close", "Закрити"), Binding("q", "close", "Закрити")]
    DEFAULT_CSS = """
    GraphScreen { align: center middle; background: $background 92%; }
    #graph-box { width: 98%; height: 96%; border: round $accent; padding: 0 1; background: $surface; }
    """

    def __init__(self, text) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="graph-box"):
            yield Static(self._text)

    def action_close(self) -> None:
        self.dismiss()


class QueryInput(Input):
    """Search field. On an empty field '/' is the hotkey that focuses it, so it is not typed as text."""

    async def _on_key(self, event) -> None:
        if event.character == "/" and not self.value:
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)


class ArchiveApp(App):
    TITLE = "Archive"
    CSS = """
    #bar { height: auto; padding: 0 1; }
    #query { width: 3fr; }
    #project { width: 1fr; }
    #minsim { width: 1fr; }
    #limit { width: 14; }
    #main { height: 1fr; }
    #results { width: 3fr; }
    #detail { width: 2fr; padding: 0 1; border-left: solid $accent; overflow-y: auto; }
    #status { height: 1; padding: 0 1; background: $panel; }
    """
    BINDINGS = [
        Binding("slash", "focus_query", "Пошук"),
        Binding("p", "focus_project", "Проєкт"),
        Binding("m", "cycle_mode", "Режим"),
        Binding("f", "cycle_filter", "Колекція"),
        Binding("e", "cycle_format", "Формат"),
        Binding("o", "open_selected", "Відкрити"),
        Binding("right", "expand_document", "Розгорнути"),
        Binding("i", "index_now", "Індексувати"),
        Binding("s", "semantic_search", "Семантично"),
        Binding("g", "show_graph", "Граф"),
        Binding("b", "show_graph_browser", "Граф у браузері"),
        Binding("escape", "focus_results", "До результатів"),
        Binding("ctrl+q", "quit", "Вихід"),
    ]

    def __init__(self, db_path: Path = DEFAULT_DB, backend: Backend | None = None) -> None:
        super().__init__()
        self.db_path = db_path
        self._backend = backend
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._hits: list[Hit] = []
        self._bodies: dict[int, str] = {}
        self.mode = "hybrid"
        self.filter_index = 0
        self.format_index = 0
        self._last_query = ""

    # layout -----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        header = Header(show_clock=True)
        header.icon = "A"  # plain ASCII: Textual's default glyph is missing from many terminal fonts
        yield header
        with Horizontal(id="bar"):
            yield QueryInput(placeholder="запит: слова або опис (Enter)", id="query")
            yield Input(placeholder="проєкт, напр. llm", id="project")
            yield Input(placeholder="мін. схожість, напр. 0.6", id="minsim")
            yield Input(placeholder=f"ліміт, {DEFAULT_LIMIT}", id="limit")
        with Horizontal(id="main"):
            yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
            with Vertical(id="detail-wrap"):
                yield Static("Введи запит і натисни Enter.", id="detail")
        yield Static("", id="status")
        yield Footer()

    def _handle_exception(self, error: Exception) -> None:
        """Every unhandled error goes to the log with its traceback before Textual shows its own screen."""
        log.error("unhandled error in the interface", exc_info=error)
        super()._handle_exception(error)

    def on_mount(self) -> None:
        log.info("tui start db=%s", self.db_path)
        table = self.query_one("#results", DataTable)
        table.add_columns("тип", "колекція", "проєкт", "назва", "місце", "схожість")
        self.query_one("#query", Input).focus()
        self._set_status("готово · модель завантажиться при першому запиті")

    def on_unmount(self) -> None:
        log.info("tui stop")
        self._exec.shutdown(wait=False, cancel_futures=True)

    # actions ----------------------------------------------------------------

    def action_focus_query(self) -> None:
        self.query_one("#query", Input).focus()

    def action_focus_project(self) -> None:
        self.query_one("#project", Input).focus()

    def action_focus_results(self) -> None:
        self.query_one("#results", DataTable).focus()

    def action_cycle_mode(self) -> None:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        self._set_status(f"режим: {self.mode}")
        self._run_search()

    def action_cycle_filter(self) -> None:
        self.filter_index = (self.filter_index + 1) % len(FILTERS)
        self._set_status(f"колекція: {FILTERS[self.filter_index]}")
        self._run_search()

    def action_cycle_format(self) -> None:
        self.format_index = (self.format_index + 1) % len(FORMATS)
        self._set_status(f"формат: {FORMATS[self.format_index]}")
        self._run_search()

    def action_open_selected(self) -> None:
        hit = self._selected()
        if hit is None:
            return
        target = hit.path
        if not Path(target).exists():
            self._set_status(f"файл не знайдено: {target}")
            return
        try:
            target = open_target(target, hit.locator)
        except Exception as exc:  # unreadable audio: open the original file instead
            self._set_status(f"не вдалось вирізати фрагмент, відкриваю файл: {exc}")
        command = open_command(target, hit.locator)
        log.info("open %s", command)
        try:
            subprocess.Popen(command, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.exception("open failed %s", command)
            self._set_status(f"не вдалось відкрити ({command[0]}): {exc}")
            return
        place = f" на {hit.locator}" if command[0] != "xdg-open" or target != hit.path else ""
        self._set_status(f"відкрито{place}: {Path(target).name}")

    def action_show_graph(self) -> None:
        """The shown results as a graph inside the terminal: groups, similarity bars and close neighbours."""
        if not self._hits:
            self._set_status("спочатку зроби пошук: граф будується з результатів на екрані")
            return
        log.info("graph in terminal requested hits=%d query=%r", len(self._hits), self._last_query)
        self._set_status("будую граф…")
        future = self._exec.submit(self._graph_data_in_thread, self._last_query, list(self._hits),
                                   getattr(self, "_searched_mode", self.mode))
        future.add_done_callback(self._show_graph_screen)

    def _graph_data_in_thread(self, query, hits, mode):
        if self._backend is None:
            self._backend = Backend(self.db_path)
        return self._backend.graph_data(query, hits, mode)

    def _show_graph_screen(self, future) -> None:
        try:
            nodes, links = future.result()
        except Exception as exc:
            log.exception("graph failed")
            self._post(self._set_status, f"граф не вдався: {type(exc).__name__}: {exc} (деталі в лозі)")
            return
        from .viz import text_graph

        text = text_graph(self._last_query, nodes, links)
        log.info("graph in terminal shown nodes=%d links=%d", len(nodes), len(links))
        self._post(self.push_screen, GraphScreen(text))
        self._post(self._set_status, "граф у терміналі · Esc — закрити")

    def action_show_graph_browser(self) -> None:
        """The same results as the neuron picture in a browser (HTML file)."""
        if not self._hits:
            self._set_status("спочатку зроби пошук: граф будується з результатів на екрані")
            return
        log.info("graph requested hits=%d query=%r", len(self._hits), self._last_query)
        self._set_status("будую граф…")
        future = self._exec.submit(self._graph_in_thread, self._last_query, list(self._hits),
                                   getattr(self, "_searched_mode", self.mode))
        future.add_done_callback(self._open_graph)

    def _graph_in_thread(self, query, hits, mode):
        if self._backend is None:
            self._backend = Backend(self.db_path)
        return self._backend.make_graph(query, hits, mode)

    def _open_graph(self, future) -> None:
        try:
            path = future.result()
        except Exception as exc:
            log.exception("graph failed")
            self._post(self._set_status, f"граф не вдався: {type(exc).__name__}: {exc} (деталі в лозі)")
            return
        command = browser_command(path)
        log.info("graph opens with %s", command[0])
        try:
            subprocess.Popen(command, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.exception("graph open failed %s", command)
            self._post(self._set_status, f"граф збережено, але браузер не відкрився: {path} ({exc})")
            return
        self._post(self._set_status, f"граф відкрито в браузері: {path.name}")

    def action_expand_document(self) -> None:
        """Show every indexed fragment of the selected document in the detail pane (plan A1: Enter or →)."""
        hit = self._selected()
        if hit is None:
            return
        log.info("expand document path=%s", hit.path)
        future = self._exec.submit(self._fragments_in_thread, hit.path)
        future.add_done_callback(lambda f, title=hit.title: self._post(self._show_fragments, f, title))

    def _fragments_in_thread(self, path: str):
        if self._backend is None:
            self._backend = Backend(self.db_path)
        return self._backend.document_fragments(path)

    def _show_fragments(self, future, title: str) -> None:
        try:
            fragments = future.result()
        except Exception as exc:
            log.exception("expand failed title=%s", title)
            self._set_status(f"не вдалось розгорнути: {type(exc).__name__}: {exc} (деталі в лозі)")
            return
        log.info("expand done fragments=%d", len(fragments))
        lines = [f"[b]{escape(title)}[/b] · фрагментів у індексі: {len(fragments)}", ""]
        for locator, body in fragments:
            flat = " ".join(body.split())[:240]
            lines.append(f"[b]{escape(locator)}[/b]  {escape(flat)}")
        self.query_one("#detail", Static).update("\n".join(lines))
        self._set_status(f"розгорнуто: {title[:60]} · {len(fragments)} фрагментів")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a result row expands its document, as the plan asks."""
        self.action_expand_document()

    def action_index_now(self) -> None:
        import sys

        # Same interpreter as this app, so it works without `archive` being on PATH.
        try:
            subprocess.Popen([sys.executable, "-m", "archive_index.cli", "index"],
                             start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.exception("index start failed")
            self._set_status(f"не вдалось запустити індексацію: {exc}")
            return
        log.info("index started from the interface")
        self._set_status("індексація запущена у фоні (лог: ~/.local/share/archive-index/runs.jsonl)")

    # search -----------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("query", "project"):
            self._run_search()
            self.query_one("#results", DataTable).focus()

    def action_semantic_search(self) -> None:
        """Meaning-only results for the current query, when no word matched (explicit, never automatic)."""
        log.info("semantic search requested by the user")
        self._run_search(mode_override="semantic")

    def _run_search(self, mode_override: str | None = None) -> None:
        query = self.query_one("#query", Input).value.strip()
        if not query:
            return
        self._last_query = query
        project = self.query_one("#project", Input).value.strip()
        raw = self.query_one("#minsim", Input).value.strip().replace(",", ".")
        try:
            min_similarity = float(raw) if raw else 0.0
        except ValueError:
            self._set_status(f"мін. схожість має бути числом, а не «{raw}»")
            return
        raw_limit = self.query_one("#limit", Input).value.strip()
        try:
            limit = int(raw_limit) if raw_limit else DEFAULT_LIMIT
        except ValueError:
            self._set_status(f"ліміт має бути цілим числом, а не «{raw_limit}»")
            return
        if not 1 <= limit <= MAX_LIMIT:
            self._set_status(f"ліміт має бути від 1 до {MAX_LIMIT}")
            return
        collection = FILTERS[self.filter_index]
        self._set_status("шукаю…")
        self._fmt = FORMATS[self.format_index]
        mode = mode_override or self.mode
        self._searched_mode = mode
        log.info("search query=%r mode=%s collection=%s project=%r min_sim=%s limit=%d fmt=%s",
                 query, mode, collection, project, min_similarity, limit, self._fmt)
        future = self._exec.submit(self._search_in_thread, query, mode, collection, project,
                                   min_similarity, limit, self._fmt)
        future.add_done_callback(self._on_done)

    def _search_in_thread(self, query, mode, collection, project, min_similarity=0.0,
                          limit=DEFAULT_LIMIT, fmt="все"):
        if self._backend is None:
            self._backend = Backend(self.db_path)
        return self._backend.search(query, mode, collection, project, min_similarity, limit, fmt)

    def _post(self, fn, *args) -> None:
        """Run fn on the UI thread: directly if already there, otherwise via call_from_thread."""
        if threading.current_thread() is threading.main_thread():
            fn(*args)
        else:
            self.call_from_thread(fn, *args)

    def _on_done(self, future) -> None:
        try:
            hits, seconds, bodies, lexical_total = future.result()
        except Exception as exc:  # shown to the user and written to the log, never silently dropped
            log.exception("search failed")
            self._post(self._set_status, f"помилка пошуку: {type(exc).__name__}: {exc} (деталі в лозі)")
            return
        log.info("search done hits=%d seconds=%.2f lexical_total=%s", len(hits), seconds, lexical_total)
        self._post(self._show_results, hits, seconds, bodies, lexical_total)

    def _show_results(self, hits: list[Hit], seconds: float, bodies: dict[int, str],
                      lexical_total: int | None) -> None:
        self._hits = hits
        self._bodies = bodies
        table = self.query_one("#results", DataTable)
        table.clear()
        for hit in hits:
            similarity = "-" if hit.similarity is None else f"{hit.similarity:.3f}"
            more = f"  (+{hit.more_in_document})" if hit.more_in_document else ""
            table.add_row(
                hit.via, hit.collection, hit.project or "-", escape(hit.title[:60]) + more,
                escape(hit.locator), similarity, key=str(hit.chunk_id),
            )
        if hits:
            table.move_cursor(row=0)
            self._show_detail(hits[0])
        elif lexical_total == 0 and getattr(self, "_searched_mode", self.mode) == "hybrid":
            self.query_one("#detail", Static).update(
                f"[b]За словами нічого не знайдено[/b]\n\nУ архіві немає жодного фрагмента з цими словами. "
                "Натисни [b]s[/b], щоб показати семантично схожі фрагменти. Вони НЕ є відповіддю на запит, "
                "лише близькі за змістом.")
        else:
            self.query_one("#detail", Static).update("Нічого не знайдено.")
        total = "-" if lexical_total is None else str(lexical_total)
        shown = f"показано {len(hits)}"
        if lexical_total is not None and lexical_total > len(hits):
            shown += f" (ще {lexical_total - len(hits)} за словами: підніми ліміт)"
        mode_shown = getattr(self, "_searched_mode", self.mode)
        self._set_status(f"{shown} · за словами всього: {total} · {seconds:.1f} с · "
                         f"режим {mode_shown} · колекція {FILTERS[self.filter_index]}")

    # detail -----------------------------------------------------------------

    def _selected(self) -> Hit | None:
        table = self.query_one("#results", DataTable)
        if not self._hits or table.cursor_row is None or table.cursor_row >= len(self._hits):
            return None
        return self._hits[table.cursor_row]

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        hit = self._selected()
        if hit is not None:
            self._show_detail(hit)

    def _show_detail(self, hit: Hit) -> None:
        try:
            self._render_detail(hit)
        except Exception as exc:  # a broken fragment must not take the whole interface down
            log.exception("detail failed chunk=%s path=%s", hit.chunk_id, hit.path)
            self._set_status(f"не вдалось показати фрагмент: {type(exc).__name__} (деталі в лозі)")

    def _render_detail(self, hit: Hit) -> None:
        # Header is markup with every field escaped; the body is a Text object, so the query words are
        # highlighted as styles and never as markup (nested [reverse] tags broke the parser).
        header = (
            f"[b]{escape(hit.title)}[/b]\n"
            f"{escape(hit.collection)} · {escape(hit.project or '-')} · {escape(hit.locator)} · {hit.via}\n"
            f"[dim]{escape(hit.path)}[/dim]\n\n"
        )
        body = Text(self._bodies.get(hit.chunk_id, hit.snippet)[:4000])
        words = sorted({w for w in tokenize(self._last_query) if len(w) >= 3}, key=len, reverse=True)
        if words:
            body.highlight_words(words, style="reverse", case_sensitive=False)
        self.query_one("#detail", Static).update(Text.assemble(Text.from_markup(header), body))

    def _set_status(self, message: str) -> None:
        # Plain text: messages carry user input and paths, which must not be read as markup.
        self.query_one("#status", Static).update(Text(message))



def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="archive-tui", description="Search the archive in a terminal.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    ArchiveApp(args.db).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
