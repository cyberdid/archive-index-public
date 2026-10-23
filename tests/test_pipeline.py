import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from archive_index.cli import main as cli_main
from archive_index.indexer import run_index
from archive_index.search import VectorCache, hybrid
from archive_index.store import Store


class FakeEmbedder:
    """Bag-of-words hashing into 64 dims; no model, deterministic, similar texts score high."""

    DIM = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for word in text.lower().replace("|", " ").split():
            v[hash(word) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def documents(self, texts, titles=None):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return self._vec(text)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    projects = tmp_path / "Projects"
    home = tmp_path / "home"

    (projects / "llm" / "wiki").mkdir(parents=True)
    (projects / "llm" / "wiki" / "setup.md").write_text(
        "# Wi-Fi\n\nУ лабораторії ми налаштовуємо точку доступу на Jetson.\n\n"
        "# Ключі\n\nAPI_KEY=supersecretvalue123 лежить тут.\n",
        encoding="utf-8",
    )
    (projects / "demo-radio").mkdir(parents=True)
    (projects / "demo-radio" / "README.md").write_text("# demo-radio\n\nПасивний моніторинг радіо.\n", encoding="utf-8")

    repo = projects / "demo-radio"
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "Fix Wi-Fi scanner dwell time"], check=True)

    run_dir = projects / "llm" / ".antigravity_runs"
    run_dir.mkdir(parents=True)
    (run_dir / "run-1.jsonl").write_text(
        json.dumps({"kind": "event", "event_type": "subtask_failed", "subtask": "task_1",
                    "payload": {"error": "verification path mismatch"}}) + "\n",
        encoding="utf-8",
    )

    claude_dir = home / ".claude" / "projects" / "-home-x-Projects-llm"
    claude_dir.mkdir(parents=True)
    claude_lines = [
        {"type": "user", "sessionId": "abcd1234-0000", "cwd": str(projects / "llm"),
         "timestamp": "2026-10-01T10:00:00Z",
         "message": {"role": "user", "content": "як налаштувати точку доступу"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "внутрішнє міркування, не індексувати"},
            {"type": "text", "text": "Відкрий nmcli і створи точку доступу."}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "very long tool output"}]}},
    ]
    (claude_dir / "abcd1234-0000.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in claude_lines) + "\n", encoding="utf-8")

    codex_dir = home / ".codex" / "sessions" / "2026" / "01" / "01"
    codex_dir.mkdir(parents=True)
    codex_lines = [
        {"type": "session_meta", "payload": {"id": "cdx-0001", "cwd": str(projects / "demo-radio")}},
        {"type": "response_item", "timestamp": "2026-01-01T09:00:00Z", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "<environment_context>служебне</environment_context>"}]}},
        {"type": "response_item", "timestamp": "2026-01-01T09:01:00Z", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "знайди сесію про реверс bladeRF"}]}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Реверс bladeRF описано в demo-radio/research."}]}},
    ]
    (codex_dir / "rollout-cdx-0001.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in codex_lines) + "\n", encoding="utf-8")

    return projects, home, tmp_path / "archive.sqlite3"


def _index(world, **kwargs):
    projects, home, db = world
    store = Store(db)
    report = run_index(store, FakeEmbedder(), projects_root=projects, home=home, **kwargs)
    return store, report


def test_full_run_indexes_every_collection(world):
    store, report = _index(world)
    stats = store.stats()
    assert set(stats) == {"wiki", "git", "sessions", "run_logs"}
    assert report.errors == []
    assert stats["sessions"]["documents"] == 2
    store.close()


def test_secrets_never_reach_the_database(world):
    store, report = _index(world)
    rows = store.conn.execute("SELECT body FROM chunks").fetchall()
    joined = "\n".join(r[0] for r in rows)
    assert "supersecretvalue123" not in joined
    assert report.redactions.get("credential_assignment", 0) >= 1
    store.close()


def test_thinking_and_tool_results_are_not_indexed(world):
    store, _ = _index(world)
    joined = "\n".join(r[0] for r in store.conn.execute("SELECT body FROM chunks"))
    assert "внутрішнє міркування" not in joined
    assert "very long tool output" not in joined
    assert "служебне" not in joined  # environment_context noise is dropped
    store.close()


def test_lexical_search_matches_inflected_ukrainian(world):
    store, _ = _index(world)
    cache = VectorCache(store)
    hits = hybrid(store, cache, "точку доступу", None, mode="lexical", k=5)
    assert any("точку доступу" in h.snippet or "точка доступу" in h.snippet or h.collection == "wiki"
               for h in hits)
    assert any(h.collection == "wiki" and h.heading == "Wi-Fi" for h in hits)
    store.close()


def test_session_turns_carry_project_and_locator(world):
    store, _ = _index(world)
    cache = VectorCache(store)
    hits = hybrid(store, cache, "реверс bladeRF", None, collections=["sessions"], mode="lexical", k=3)
    assert hits and hits[0].project == "demo-radio"
    assert hits[0].locator.startswith("line ")
    store.close()


def test_project_filter_restricts_results(world):
    store, _ = _index(world)
    cache = VectorCache(store)
    hits = hybrid(store, cache, "точку доступу", None, project="demo-radio", mode="lexical", k=5)
    assert all(h.project == "demo-radio" for h in hits)
    store.close()


def test_git_history_is_searchable(world):
    store, _ = _index(world)
    cache = VectorCache(store)
    hits = hybrid(store, cache, "dwell time scanner", None, collections=["git"], mode="lexical", k=3)
    assert hits and "dwell time" in hits[0].snippet
    store.close()


def test_second_run_skips_unchanged_files(world):
    store, first = _index(world)
    store.close()
    store2, second = _index(world)
    assert second.chunks == 0
    assert second.skipped >= first.indexed.get("wiki", 0)
    store2.close()


def test_deleted_source_is_removed_from_index(world):
    projects, _, _ = world
    store, _ = _index(world)
    (projects / "demo-radio" / "README.md").unlink()
    store2, _ = _index(world)
    paths = [r[0] for r in store2.conn.execute("SELECT path FROM documents WHERE collection='wiki'")]
    assert not any(p.endswith("demo-radio/README.md") for p in paths)
    store.close()
    store2.close()


def test_semantic_mode_uses_vectors(world):
    store, _ = _index(world)
    cache = VectorCache(store)
    qv = FakeEmbedder().query("точку доступу Jetson")
    hits = hybrid(store, cache, "точку доступу Jetson", qv, mode="semantic", k=3)
    assert hits and all(h.via == "semantic" for h in hits)
    store.close()


def test_cli_index_and_search_round_trip(world, capsys):
    projects, home, db = world
    code = cli_main(
        ["--db", str(db), "index", "--projects-root", str(projects), "--home", str(home)],
        embedder=FakeEmbedder(),
    )
    assert code == 0
    capsys.readouterr()
    code = cli_main(["--db", str(db), "search", "точку доступу", "--mode", "lexical", "--json", "-k", "3"],
                    embedder=FakeEmbedder())
    assert code == 0
    out = capsys.readouterr().out
    assert '"collection": "wiki"' in out


def test_database_file_is_owner_only(world):
    _, _, db = world
    store, _ = _index(world)
    store.close()
    assert oct(os.stat(db).st_mode & 0o777) == "0o600"


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        self.embedded = 0

    def documents(self, texts, titles=None):
        self.embedded += len(texts)
        return super().documents(texts, titles)


def test_growing_session_embeds_only_the_new_turn(world):
    projects, home, db = world
    store = Store(db)
    run_index(store, FakeEmbedder(), projects_root=projects, home=home, collections=("sessions",))
    store.close()

    session = next((home / ".claude" / "projects").glob("*/*.jsonl"))
    with session.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "user", "sessionId": "abcd1234-0000",
                             "cwd": str(projects / "llm"), "timestamp": "2026-10-02T09:00:00Z",
                             "message": {"role": "user", "content": "нове питання про Jetson"}},
                            ensure_ascii=False) + "\n")

    store2 = Store(db)
    counter = CountingEmbedder()
    report = run_index(store2, counter, projects_root=projects, home=home, collections=("sessions",))
    assert report.reused_vectors >= 1
    assert counter.embedded == 1  # only the appended turn was embedded
    store2.close()


class MediaFakeEmbedder(FakeEmbedder):
    def image(self, path):
        return self._vec("image " + Path(path).name)

    def audio(self, samples, sample_rate=16000):
        return self._vec("audio " + str(len(samples)))


def test_rebuild_terms_reproduces_the_terms_of_indexing(tmp_path):
    import wave

    from PIL import Image

    from archive_index.lexical import index_terms

    projects = tmp_path / "Projects"
    (projects / "p" / "photo_set").mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 10, 10)).save(projects / "p" / "photo_set" / "антена_на_даху.png")
    (projects / "p" / "notes.md").write_text("# Антени\n\nНалаштування точки доступу на даху.", encoding="utf-8")
    rec = projects / "p" / "processed_recordings"
    rec.mkdir(parents=True)
    with wave.open(str(rec / "seg-001_enhanced.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000 * 25, dtype=np.int16).tobytes())
    (rec / "seg-001_transcript.txt").write_text(
        "[00:00:02.000 --> 00:00:08.000] Speaker 1: перевірка зв'язку на частоті\n", encoding="utf-8")

    store = Store(tmp_path / "t.sqlite3")
    try:
        report = run_index(store, MediaFakeEmbedder(), projects_root=projects, home=tmp_path / "h",
                           collections=("wiki", "images", "audio"))
        assert report.errors == []
        read = "SELECT c.collection, f.terms FROM chunks c JOIN chunks_fts f ON f.rowid = c.id ORDER BY c.id"
        indexed = store.conn.execute(read).fetchall()
        assert {collection for collection, _ in indexed} == {"wiki", "images", "audio"}
        store.conn.execute("UPDATE chunks_fts SET terms = ''")
        store.rebuild_terms(index_terms)
        assert store.conn.execute(read).fetchall() == indexed
    finally:
        store.close()


def test_images_and_audio_are_indexed_with_locators(tmp_path):
    import wave

    import numpy as np
    from PIL import Image

    projects = tmp_path / "Projects"
    (projects / "3d").mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 10, 10)).save(projects / "3d" / "bottom_profile.png")
    rec = projects / "demo-audio" / "recordings"
    rec.mkdir(parents=True)
    with wave.open(str(rec / "seg-001.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(16000 * 40, dtype=np.int16)).tobytes())  # 40 s of silence

    store = Store(tmp_path / "media.sqlite3")
    report = run_index(store, MediaFakeEmbedder(), projects_root=projects, home=tmp_path / "h",
                       collections=("images", "audio"))
    assert report.errors == []
    assert store.stats()["images"]["chunks"] == 1
    assert store.stats()["audio"]["chunks"] >= 2  # 40 s with 20 s windows and 15 s step
    locators = [r[0] for r in store.conn.execute("SELECT locator FROM chunks WHERE collection='audio'")]
    assert all(loc.endswith("s") and "-" in loc for loc in locators)
    store.close()
