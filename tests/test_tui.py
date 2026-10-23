import asyncio
from pathlib import Path

from textual.widgets import DataTable, Input, Static

from archive_index.search import Hit
from archive_index.tui import ArchiveApp, Backend


class FakeBackend:
    def __init__(self):
        self.calls = []

    def search(self, query, mode, collection, project, min_similarity=0.0, limit=200, fmt="все"):
        self.calls.append((query, mode, collection, project, limit))
        hits = [
            Hit(chunk_id=1, collection="sessions", project="demo-radio", title="сесія реверсу",
                path="/x/a.jsonl", locator="line 4", heading="", snippet="реверс bladeRF",
                occurred_at="", score=0.9, via="both"),
            Hit(chunk_id=2, collection="wiki", project="llm", title="нотатки",
                path="/x/b.md", locator="section 1", heading="", snippet="точка доступу",
                occurred_at="", score=0.5, via="lexical"),
        ]
        return hits, 0.01, {1: "повний текст 1", 2: "повний текст 2"}, 57


def _run(coro):
    return asyncio.run(coro)


def test_search_fills_results_and_detail(tmp_path):
    backend = FakeBackend()

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=backend)
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "реверс"
            await pilot.press("enter")
            await pilot.pause(0.3)
            table = app.query_one("#results", DataTable)
            assert table.row_count == 2
            assert backend.calls[-1][:2] == ("реверс", "hybrid")
            status = str(app.query_one("#status", Static).render())
            assert "за словами всього: 57" in status
            assert "повний текст 1" in str(app.query_one("#detail", Static).render())

    _run(scenario())


def test_mode_and_collection_keys_rerun_search(tmp_path):
    backend = FakeBackend()

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=backend)
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "точка доступу"
            await pilot.press("enter")
            await pilot.pause(0.3)
            app.query_one("#results", DataTable).focus()
            await pilot.press("m")
            await pilot.pause(0.3)
            assert backend.calls[-1][1] == "lexical"
            await pilot.press("f")
            await pilot.pause(0.3)
            assert backend.calls[-1][2] == "wiki"  # first filter after "all" in FILTERS

    _run(scenario())


def test_empty_query_does_not_search(tmp_path):
    backend = FakeBackend()

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=backend)
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert backend.calls == []

    _run(scenario())


def test_project_filter_is_passed_to_search(tmp_path):
    backend = FakeBackend()

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=backend)
        async with app.run_test() as pilot:
            app.query_one("#project", Input).value = "llm"
            app.query_one("#query", Input).value = "нотатки"
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert backend.calls[-1][3] == "llm"

    _run(scenario())


def test_index_key_does_not_crash_when_process_cannot_start(tmp_path, monkeypatch):
    import subprocess

    def boom(*args, **kwargs):
        raise FileNotFoundError("archive")

    monkeypatch.setattr(subprocess, "Popen", boom)
    backend = FakeBackend()

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=backend)
        async with app.run_test() as pilot:
            app.action_index_now()
            await pilot.pause(0.1)
            assert "не вдалось" in str(app.query_one("#status", Static).render())

    _run(scenario())


def test_limit_field_is_passed_and_validated(tmp_path):
    backend = FakeBackend()

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=backend)
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "реверс"
            app.query_one("#limit", Input).value = "50"
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert backend.calls[-1][4] == 50
            app.query_one("#limit", Input).value = "abc"
            app.query_one("#query", Input).focus()
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert "цілим числом" in str(app.query_one("#status", Static).render())

    _run(scenario())


def test_open_command_goes_to_the_place_of_the_match(monkeypatch):
    from archive_index import tui
    from archive_index.tui import format_matches, open_command

    monkeypatch.setattr(tui.shutil, "which", lambda name: None)  # no positioned player installed
    assert open_command("/x/spec.pdf", "page 14") == ["evince", "--page-index=14", "/x/spec.pdf"]
    assert open_command("/x/a.py", "line 42") == ["code", "-g", "/x/a.py:42"]
    assert open_command("/x/rec.mp3", "0-20s") == ["xdg-open", "/x/rec.mp3"]
    assert format_matches("/x/a.PDF", "pdf") and not format_matches("/x/a.md", "pdf")
    assert format_matches("/x/r.opus", "audio")


def test_audio_match_opens_a_clip_that_starts_at_the_match(tmp_path, monkeypatch):
    import wave

    import numpy as np

    from archive_index import tui

    monkeypatch.setattr(tui.shutil, "which", lambda name: None)  # the clip path is the fallback
    monkeypatch.setattr(tui, "CLIP_CACHE", tmp_path / "clips")
    src = tmp_path / "rec.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.arange(16000 * 4) % 200 * 100).astype("<i2").tobytes())
    clip = tui.audio_clip(str(src), "1-3s")
    with wave.open(str(clip)) as w:
        assert w.getnframes() == 2 * 16000  # exactly the matched two seconds
    assert tui.open_target(str(src), "1-3s") == str(tui.audio_clip(str(src), "1-3s"))
    assert tui.open_target(str(src), "not-a-span") == str(src)


def test_audio_opens_at_the_second_with_the_positioned_player(monkeypatch):
    from archive_index import tui

    monkeypatch.setattr(tui.shutil, "which", lambda name: "/usr/bin/mpv")
    cmd = tui.open_command("/rec/call.wav", "42-62s")
    assert cmd == ["mpv", "--start=42", "/rec/call.wav"]
    assert tui.open_target("/rec/call.wav", "42-62s") == "/rec/call.wav"  # no clip needed


def test_audio_falls_back_to_a_clip_without_a_positioned_player(tmp_path, monkeypatch):
    from archive_index import tui

    monkeypatch.setattr(tui.shutil, "which", lambda name: None)
    monkeypatch.setattr(tui, "audio_clip", lambda path, loc: tmp_path / "clip.wav")
    assert tui.open_target("/rec/call.wav", "42-62s") == str(tmp_path / "clip.wav")


class AudioBackend:
    """One audio hit whose file exists, so the open action can run end to end."""

    def __init__(self, path):
        self.path = path

    def search(self, query, mode, collection, project, min_similarity=0.0, limit=200, fmt="все"):
        hit = Hit(chunk_id=9, collection="audio", project="demo-call", title="запис дзвінка",
                  path=str(self.path), locator="15-35s", heading="", snippet="ретранслятор",
                  occurred_at="", score=0.8, via="lexical")
        return [hit], 0.01, {9: "повний текст"}, 1


def test_open_key_on_audio_hit_starts_the_recording_at_the_second(tmp_path, monkeypatch):
    from archive_index import tui

    rec = tmp_path / "call.wav"
    rec.write_bytes(b"RIFF")  # the file must exist; the player itself is replaced below
    launched = []

    class Recorder:
        def __init__(self, command, **kwargs):
            launched.append(command)

    monkeypatch.setattr(tui.subprocess, "Popen", Recorder)
    monkeypatch.setattr(tui.shutil, "which", lambda name: "/usr/bin/gst-play-1.0" if name == tui.PLAYER else None)

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=AudioBackend(rec))
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "ретранслятор"
            await pilot.press("enter")
            await pilot.pause(0.3)
            table = app.query_one("#results", DataTable)
            assert table.row_count == 1
            await pilot.press("o")
            await pilot.pause(0.3)
            status = str(app.query_one("#status", Static).render())
            assert "відкрито на 15-35s: call.wav" in status

    _run(scenario())
    assert launched == [["mpv", "--start=15", str(rec)]]


def test_video_frame_and_sound_window_open_at_their_second(monkeypatch):
    from archive_index import tui

    monkeypatch.setattr(tui.shutil, "which", lambda name: "/usr/bin/mpv" if name == tui.PLAYER else None)
    assert tui.open_command("/v/notes.webm", "frame 01:25") == ["mpv", "--start=85", "/v/notes.webm"]
    assert tui.open_command("/v/notes.mp4", "20-35s") == ["mpv", "--start=20", "/v/notes.mp4"]
    # without mpv, gst-play is used with an exact seek
    monkeypatch.setattr(tui.shutil, "which", lambda name: "/usr/bin/gst-play-1.0" if name == tui.FALLBACK_PLAYER else None)
    assert tui.open_command("/v/notes.mp4", "20-35s") == ["gst-play-1.0", "-a", "-s", "20", "/v/notes.mp4"]
    assert tui.open_command("/x/spec.pdf", "frame 01:25") == ["xdg-open", "/x/spec.pdf"]  # not a video


class ExpandBackend(FakeBackend):
    def document_fragments(self, path, limit=200):
        return [("line 4", "реверс bladeRF"), ("line 9", "другий фрагмент сесії"), ("line 17", "третій")]


def test_enter_expands_the_selected_document_to_all_its_fragments(tmp_path):
    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=ExpandBackend())
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "реверс"
            await pilot.press("enter")
            await pilot.pause(0.3)
            table = app.query_one("#results", DataTable)
            table.focus()
            await pilot.press("enter")
            await pilot.pause(0.5)
            detail = str(app.query_one("#detail", Static).render())
            assert "фрагментів у індексі: 3" in detail
            assert "line 17" in detail and "другий фрагмент сесії" in detail

    _run(scenario())


def test_format_filter_reaches_rare_formats_below_the_top_results(tmp_path):
    """Regression: the filter used to act on the top results only, so a rare format could show nothing."""
    from archive_index.indexer import run_index
    from archive_index.store import Store
    from archive_index.search import hybrid, VectorCache
    import numpy as np
    from archive_index.lexical import query_expression

    projects = tmp_path / "Projects" / "p"
    projects.mkdir(parents=True)
    for i in range(30):  # many markdown notes that match the word more strongly
        (projects / f"note{i}.md").write_text("антена антена антена ретранслятор " * 3, encoding="utf-8")
    (projects / "spec.txt").write_text("антена ретранслятор один раз", encoding="utf-8")

    class Emb:
        def documents(self, texts, titles=None):
            return np.ones((len(texts), 4), dtype=np.float32) if texts else np.zeros((0, 0), np.float32)

        def query(self, text, prompt="SearchQuery"):
            return np.ones(4, dtype=np.float32)

    store = Store(tmp_path / "f.sqlite3")
    run_index(store, Emb(), projects_root=tmp_path / "Projects", home=tmp_path / "h", collections=("docs", "wiki"))
    cache = VectorCache(store)
    hits = hybrid(store, cache, "антена", None, mode="lexical", k=5, per_document=0,
                  suffixes=(".txt",))
    assert [h.path.rsplit("/", 1)[-1] for h in hits] == ["spec.txt"]
    assert store.count_lexical(query_expression("антена"), collections=None, project=None,
                               suffixes=(".txt",)) == 1
    store.close()


def test_query_words_are_highlighted_in_the_detail_text():
    from rich.text import Text

    from archive_index.tui import ArchiveApp

    body = Text("Антена працює, а антенна ні. Hailo verse rev [/] [b]")
    body.highlight_words(["антена", "verse", "rev"], style="reverse", case_sensitive=False)
    styled = [body.plain[span.start:span.end] for span in body.spans if span.style == "reverse"]
    assert styled == ["Антена", "verse", "rev"]  # no markup parsing: brackets in the text stay as they are
    assert body.plain.endswith("[/] [b]")


def test_detail_survives_brackets_and_nested_words_in_query(tmp_path):
    """Regression: highlight tags nested inside each other (a query word inside 'reverse') broke rendering."""
    class TrickyBackend(FakeBackend):
        def search(self, query, mode, collection, project, min_similarity=0.0, limit=200, fmt="все"):
            hits, sec, _, total = super().search(query, mode, collection, project, min_similarity, limit, fmt)
            bodies = {1: "reverse [/reverse] verse [/] [b] text", 2: "обычный фрагмент"}
            return hits, sec, bodies, total

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=TrickyBackend())
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "reverse verse rev [/]"
            await pilot.press("enter")
            await pilot.pause(0.4)
            await pilot.press("down")
            await pilot.pause(0.2)
            await pilot.press("up")
            await pilot.pause(0.3)
            assert "reverse [/reverse]" in str(app.query_one("#detail", Static).render())

    _run(scenario())

def test_unhandled_and_detail_errors_are_logged_with_traceback(tmp_path, caplog):
    import logging

    class BrokenBackend(FakeBackend):
        def search(self, query, mode, collection, project, min_similarity=0.0, limit=200, fmt="все"):
            hits, sec, _, total = super().search(query, mode, collection, project, min_similarity, limit, fmt)
            return hits, sec, {1: "фрагмент"}, total

    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=BrokenBackend())
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "реверс"
            await pilot.press("enter")
            await pilot.pause(0.3)
            def boom(hit):
                raise ValueError("сталася помилка відображення")
            app._render_detail = boom
            table = app.query_one("#results", DataTable)
            table.move_cursor(row=1)
            await pilot.pause(0.3)
            assert "не вдалось показати фрагмент" in str(app.query_one("#status", Static).render())

    with caplog.at_level(logging.DEBUG, logger="archive_index"):
        _run(scenario())
    records = [r for r in caplog.records if r.name == "archive_index.tui"]
    assert any("search query=" in r.getMessage() for r in records)
    assert any(r.exc_info and "сталася помилка відображення" in str(r.exc_info[1]) for r in records)


def test_slash_hotkey_does_not_leak_into_the_query(tmp_path):
    """Regression: pressing / to focus the query field typed a '/' into it."""
    async def scenario():
        app = ArchiveApp(tmp_path / "x.sqlite3", backend=FakeBackend())
        async with app.run_test() as pilot:
            await pilot.press("slash")
            await pilot.pause(0.2)
            await pilot.press("r", "e", "v")
            await pilot.pause(0.2)
            assert app.query_one("#query", Input).value == "rev"

    _run(scenario())


def test_hybrid_shows_nothing_when_no_word_matches_but_semantic_key_does(tmp_path):
    """No meaning-only padding by default: an unknown word gives an empty list, 's' gives meaning matches."""
    import numpy as np

    from archive_index.indexer import run_index  # noqa: F401  (import check: the indexer stays importable)
    from archive_index.search import VectorCache
    from archive_index.store import ChunkRecord, Store

    store = Store(tmp_path / "s.sqlite3")
    vec = np.ones(4, dtype=np.float32)
    store.replace_document(collection="docs", path="/p/antena.txt", size=1, mtime_ns=1, sha256="a",
                           title="antena", project="p",
                           chunks=[ChunkRecord(ord=0, heading="", locator="section 1",
                                               body="Антена ретранслятора працює на частоті 1800 МГц.",
                                               terms="антена ретранслятора", occurred_at="", vector=vec,
                                               body_sha="x")])

    class Emb:
        def query(self, text, prompt="SearchQuery"):
            return vec

    backend = Backend(tmp_path / "s.sqlite3", embedder=Emb())
    hits, _, _, total = backend.search("Монтана", "hybrid", "all", "", 0.0, 50, "все")
    assert hits == [] and total == 0
    hits, _, _, _ = backend.search("Монтана", "semantic", "all", "", 0.0, 50, "все")
    assert hits  # the explicit meaning-only search still returns the nearest fragments
    store.close()
    backend.store.close()


def test_g_key_builds_the_neuron_picture_and_opens_it(tmp_path, monkeypatch):
    import numpy as np

    from archive_index import tui
    from archive_index.store import ChunkRecord, Store

    store = Store(tmp_path / "g.sqlite3")
    rng = np.random.default_rng(1)
    chunks = [ChunkRecord(ord=i, heading="", locator=f"section {i}", body=f"текст фрагмента номер {i} про антену",
                          terms=f"антена {i}", occurred_at="", vector=rng.standard_normal(16).astype(np.float32),
                          body_sha=f"s{i}") for i in range(6)]
    store.replace_document(collection="docs", path="/p/g.txt", size=1, mtime_ns=1, sha256="g", title="g",
                           project="p", chunks=chunks)
    store.close()

    class Emb:
        def query(self, text, prompt="SearchQuery"):
            return rng.standard_normal(16).astype(np.float32)

    launched = []

    class Recorder:
        def __init__(self, command, **kwargs):
            launched.append(command)

    monkeypatch.setattr(tui.subprocess, "Popen", Recorder)
    import archive_index.embed as embed_module
    monkeypatch.setattr(embed_module, "make_embedder", lambda: Emb())  # the backend is created in the worker thread

    async def scenario():
        app = ArchiveApp(tmp_path / "g.sqlite3")
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "антена"
            await pilot.press("enter")
            await pilot.pause(0.5)
            await pilot.press("escape")
            await pilot.press("b")
            await pilot.pause(1.0)

    _run(scenario())
    opened = [c for c in launched if c and c[0] in ("xdg-open", "google-chrome")]  # other libraries may run commands too
    assert opened
    html_file = Path(opened[0][1])
    assert html_file.exists() and html_file.parent.name == "graphs"
    text = html_file.read_text(encoding="utf-8")
    assert "const DATA" in text and '"kind": "hit"' in text


def test_graph_opens_with_the_browser_named_in_the_environment(tmp_path, monkeypatch):
    from archive_index import tui

    monkeypatch.delenv("ARCHIVE_BROWSER", raising=False)
    assert tui.browser_command(tmp_path / "g.html") == ["xdg-open", str(tmp_path / "g.html")]
    monkeypatch.setenv("ARCHIVE_BROWSER", "google-chrome")
    assert tui.browser_command(tmp_path / "g.html") == ["google-chrome", str(tmp_path / "g.html")]


def test_g_opens_the_graph_in_the_terminal_and_escape_closes_it(tmp_path, monkeypatch):
    import numpy as np

    from archive_index import tui
    from archive_index.store import ChunkRecord, Store

    store = Store(tmp_path / "t.sqlite3")
    rng = np.random.default_rng(3)
    chunks = [ChunkRecord(ord=i, heading="", locator=f"section {i}", body=f"фрагмент {i} про антену",
                          terms=f"антена {i}", occurred_at="", vector=rng.standard_normal(16).astype(np.float32),
                          body_sha=f"t{i}") for i in range(5)]
    store.replace_document(collection="docs", path="/p/t.txt", size=1, mtime_ns=1, sha256="t", title="t",
                           project="p", chunks=chunks)
    store.close()

    class Emb:
        def query(self, text, prompt="SearchQuery"):
            return rng.standard_normal(16).astype(np.float32)

    import archive_index.embed as embed_module
    monkeypatch.setattr(embed_module, "make_embedder", lambda: Emb())

    async def scenario():
        app = ArchiveApp(tmp_path / "t.sqlite3")
        async with app.run_test() as pilot:
            app.query_one("#query", Input).value = "антена"
            await pilot.press("enter")
            await pilot.pause(0.8)
            await pilot.press("escape")
            await pilot.press("g")
            await pilot.pause(1.0)
            assert isinstance(app.screen, tui.GraphScreen)
            await pilot.press("escape")
            await pilot.pause(0.3)
            assert not isinstance(app.screen, tui.GraphScreen)

    _run(scenario())
