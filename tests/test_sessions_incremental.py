import json

import numpy as np

from archive_index.indexer import run_index
from archive_index.search import VectorCache, hybrid
from archive_index.store import Store


class FakeEmbedder:
    DIM = 64

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype=np.float32)
        for word in text.lower().replace("|", " ").split():
            v[sum(map(ord, word)) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def documents(self, texts, titles=None):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, 0), np.float32)

    def query(self, text):
        return self._vec(text)


class Counting(FakeEmbedder):
    def __init__(self):
        self.calls = 0

    def documents(self, texts, titles=None):
        self.calls += len(texts)
        return super().documents(texts, titles)


def _claude_line(kind, text, lineno):
    return json.dumps({"type": kind, "sessionId": "inc-0001", "cwd": "/home/x/Projects/llm",
                       "timestamp": f"2026-10-01T10:{lineno:02d}:00Z",
                       "message": {"role": kind, "content": text}}, ensure_ascii=False) + "\n"


def _run(tmp_path, embedder):
    store = Store(tmp_path / "inc.sqlite3")
    report = run_index(store, embedder, projects_root=tmp_path / "Projects",
                       home=tmp_path / "home", collections=("sessions",))
    return store, report


def _session(tmp_path):
    folder = tmp_path / "home" / ".claude" / "projects" / "-x-llm"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "inc-0001.jsonl"


def _bodies(store):
    return [r[0] for r in store.conn.execute("SELECT body FROM chunks WHERE collection='sessions'")]


def test_reply_appended_after_first_run_is_attached_and_not_duplicated(tmp_path):
    path = _session(tmp_path)
    path.write_text(_claude_line("user", "первинне питання про реверс", 1), encoding="utf-8")
    store, _ = _run(tmp_path, FakeEmbedder())
    store.close()

    with path.open("a", encoding="utf-8") as fh:
        fh.write(_claude_line("assistant", "відповідь, що прийшла пізніше", 2))
    store, _ = _run(tmp_path, FakeEmbedder())

    bodies = _bodies(store)
    joined = "\n".join(bodies)
    assert "відповідь, що прийшла пізніше" in joined
    assert joined.count("первинне питання про реверс") == 1  # the re-read turn was trimmed, not doubled
    store.close()


def test_new_turns_embed_only_new_text(tmp_path):
    path = _session(tmp_path)
    path.write_text(_claude_line("user", "старе питання", 1) + _claude_line("assistant", "стара відповідь", 2),
                    encoding="utf-8")
    store, _ = _run(tmp_path, FakeEmbedder())
    store.close()

    with path.open("a", encoding="utf-8") as fh:
        fh.write(_claude_line("user", "новий хід про Jetson", 3))
    counter = Counting()
    store, report = _run(tmp_path, counter)
    assert counter.calls == 1  # the re-read old turn hits the cache; only the new turn is embedded
    assert "новий хід про Jetson" in "\n".join(_bodies(store))
    store.close()


def test_partial_last_line_is_not_consumed_until_complete(tmp_path):
    path = _session(tmp_path)
    path.write_text(_claude_line("user", "питання до неповного рядка", 1), encoding="utf-8")
    half = _claude_line("assistant", "відповідь дописується", 2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(half[:25])  # the model is still writing this line
    store, _ = _run(tmp_path, FakeEmbedder())
    store.close()

    with path.open("a", encoding="utf-8") as fh:
        fh.write(half[25:])
    store, _ = _run(tmp_path, FakeEmbedder())
    joined = "\n".join(_bodies(store))
    assert "відповідь дописується" in joined
    assert joined.count("питання до неповного рядка") == 1
    store.close()


def test_rewritten_file_is_fully_reindexed(tmp_path):
    path = _session(tmp_path)
    path.write_text(_claude_line("user", "версія один з унікальним словом alphaword", 1), encoding="utf-8")
    store, _ = _run(tmp_path, FakeEmbedder())
    store.close()

    path.write_text(_claude_line("user", "версія два з унікальним словом betaword", 1), encoding="utf-8")
    store, _ = _run(tmp_path, FakeEmbedder())
    joined = "\n".join(_bodies(store))
    assert "alphaword" not in joined  # stale text from the old version must be gone
    assert "betaword" in joined
    store.close()


def test_codex_session_grows_incrementally(tmp_path):
    folder = tmp_path / "home" / ".codex" / "sessions" / "2026" / "01" / "01"
    folder.mkdir(parents=True)
    path = folder / "rollout-cdx.jsonl"
    meta = {"type": "session_meta", "payload": {"id": "cdx-9", "cwd": str(tmp_path / "Projects" / "demo-radio")}}
    msg = lambda role, text: {"type": "response_item", "payload": {  # noqa: E731
        "type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}}
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [meta, msg("user", "реверс пристрою")])
                    + "\n", encoding="utf-8")
    store, _ = _run(tmp_path, FakeEmbedder())
    store.close()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(msg("assistant", "знайдено в каталозі research"), ensure_ascii=False) + "\n")
    store, _ = _run(tmp_path, FakeEmbedder())
    cache = VectorCache(store)
    hits = hybrid(store, cache, "каталозі research", None, mode="lexical", k=3)
    assert hits and hits[0].project == "demo-radio"
    assert "реверс пристрою" in "\n".join(_bodies(store))
    store.close()


def test_deleted_session_file_disappears_from_index(tmp_path):
    path = _session(tmp_path)
    path.write_text(_claude_line("user", "сесія яку видалять звичайним шляхом", 1), encoding="utf-8")
    store, _ = _run(tmp_path, FakeEmbedder())
    store.close()
    path.unlink()
    store, _ = _run(tmp_path, FakeEmbedder())
    assert not any("сесія яку видалять" in b for b in _bodies(store))
    store.close()
