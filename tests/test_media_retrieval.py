"""Retrieval on pictures and sound (plan C, D1, G3).

Two levels:
- pipeline tests with a deterministic embedder whose vectors come from the file bytes: they check that every
  picture and every 20-second window is stored, found by its own vector, and found by its name or locator;
- model tests, run only with ARCHIVE_TEST_MODEL=1, with the real EmbeddingGemma on synthetic files.
"""

import hashlib
import os
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from archive_index.indexer import run_index
from archive_index.search import VectorCache, hybrid
from archive_index.store import Store


def _unit(data: bytes) -> np.ndarray:
    digest = np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8).astype(np.float32) - 127.5
    return digest / np.linalg.norm(digest)


class ContentEmbedder:
    """Vectors come from the content: the same bytes give the same vector, different bytes differ."""

    def image(self, path):
        return _unit(Path(path).read_bytes())

    def audio(self, samples):
        return _unit(np.asarray(samples, dtype=np.float32).tobytes())

    def documents(self, texts, titles=None):
        return np.zeros((len(texts), 32), dtype=np.float32)

    def query(self, text, prompt="SearchQuery"):
        return _unit(text.encode("utf-8"))


def _write_tone(path: Path, hz: float, seconds: float = 45.0) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000.0
    samples = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((samples * 32767).astype("<i2").tobytes())
    return samples


def _make_pictures(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    pictures = {}
    for name, color in {"red": (220, 30, 30), "green": (30, 200, 60), "blue": (30, 60, 220)}.items():
        path = folder / f"{name}_square.png"
        Image.new("RGB", (96, 96), color).save(path)
        pictures[name] = path
    return pictures


@pytest.fixture()
def media_index(tmp_path):
    projects = tmp_path / "Projects"
    pictures = _make_pictures(projects / "p" / "pictures")
    sound = projects / "p" / "sound"
    sound.mkdir(parents=True)
    tone440 = sound / "tone440.wav"
    tone1200 = sound / "tone1200.wav"
    samples440 = _write_tone(tone440, 440.0)
    _write_tone(tone1200, 1200.0)
    store = Store(tmp_path / "m.sqlite3")
    run_index(store, ContentEmbedder(), projects_root=projects, home=tmp_path / "h",
              collections=("images", "audio"))
    yield store, pictures, samples440, projects
    store.close()


def _top_path(store, vector, collection, k=1):
    cache = VectorCache(store)
    ids = [cid for cid, _ in cache.semantic_scored(vector, collections=[collection], project=None, pool=k)]
    rows = store.chunk_rows(ids)
    return [rows[cid].path for cid in ids]


def test_every_picture_is_found_by_its_own_content(media_index):
    store, pictures, _, _ = media_index
    embedder = ContentEmbedder()
    for name, path in pictures.items():
        assert _top_path(store, embedder.image(str(path)), "images")[0] == str(path), name


def test_pictures_are_found_by_their_name_words(media_index):
    store, _, _, _ = media_index
    cache = VectorCache(store)
    hits = hybrid(store, cache, "green", None, mode="lexical", collections=["images"], k=5, per_document=0)
    assert [Path(h.path).name for h in hits] == ["green_square.png"]
    assert hits[0].locator == "image"


def test_a_sound_window_is_found_by_its_own_samples(media_index):
    from archive_index.audio_io import read_audio

    store, _, _, projects = media_index
    embedder = ContentEmbedder()
    # Decode the file the same way the indexer does, so the window is byte-identical to the indexed one.
    window = np.asarray(read_audio(projects / "p" / "sound" / "tone440.wav"), dtype=np.float32)[0:20 * 16000]
    top = _top_path(store, embedder.audio(window), "audio")[0]
    assert Path(top).name == "tone440.wav"
    cache = VectorCache(store)
    ids = [cid for cid, _ in cache.semantic_scored(embedder.audio(window), collections=["audio"], project=None,
                                                   pool=1)]
    assert store.chunk_rows(ids)[ids[0]].locator == "0-20s"


def test_sound_is_found_by_its_file_name(media_index):
    store, _, _, _ = media_index
    cache = VectorCache(store)
    hits = hybrid(store, cache, "tone1200", None, mode="lexical", collections=["audio"], k=10, per_document=0)
    assert hits and all(Path(h.path).name == "tone1200.wav" for h in hits)
    assert {h.locator for h in hits} >= {"0-20s"}


def test_audio_transcript_binds_to_matching_20s_window_and_reuses_vectors(tmp_path):
    projects = tmp_path / "Projects"
    sound = projects / "p" / "processed_recordings" / "session1"
    sound.mkdir(parents=True)
    wav_path = sound / "seg-001_enhanced.wav"
    _write_tone(wav_path, 440.0, seconds=45.0)

    class CountingEmbedder(ContentEmbedder):
        def __init__(self):
            self.audio_calls = 0

        def audio(self, samples):
            self.audio_calls += 1
            return super().audio(samples)

    store = Store(tmp_path / "t.sqlite3")
    embedder = CountingEmbedder()
    rep1 = run_index(store, embedder, projects_root=projects, home=tmp_path / "h", collections=("audio",))
    first_calls = embedder.audio_calls
    assert first_calls == 3
    assert rep1.reused_vectors == 0

    # Add companion transcript; re-indexing without --force must detect the new transcript
    # and reuse all 3 audio window vectors from SQLite without calling embedder.audio() again.
    transcript = sound / "seg-001_transcript.txt"
    transcript.write_text(
        "# Source: /tmp/seg-001.opus\n"
        "[00:00:02.000 --> 00:00:08.000] Speaker 1 [User] (91%): Перше вікно перевірка зв'язку\n"
        "[00:00:22.000 --> 00:00:28.000] Speaker 2 [Op] (85%): Друге вікно секретний маркер синхронізації\n"
        "[00:00:31.000 --> 00:00:34.000] Speaker 1: Субтитри зробив хтось\n",
        encoding="utf-8",
    )
    rep2 = run_index(store, embedder, projects_root=projects, home=tmp_path / "h", collections=("audio",))
    assert embedder.audio_calls == first_calls  # zero additional GPU calls!
    assert rep2.reused_vectors == 3

    cache = VectorCache(store)
    hits = hybrid(store, cache, "маркер синхронізації", None, mode="lexical", collections=["audio"], k=5, per_document=0)
    assert hits
    assert hits[0].locator == "15-35s"
    assert "маркер синхронізації" in hits[0].snippet

    # Non-speech marker ("Субтитри...") should be filtered out
    sub_hits = hybrid(store, cache, "субтитри", None, mode="lexical", collections=["audio"], k=5, per_document=0)
    assert not sub_hits
    store.close()


def test_a_different_picture_does_not_win(media_index):
    store, pictures, _, _ = media_index
    embedder = ContentEmbedder()
    assert _top_path(store, embedder.image(str(pictures["red"])), "images")[0] != str(pictures["green"])


@pytest.mark.skipif(os.environ.get("ARCHIVE_TEST_MODEL") != "1",
                    reason="real model test: set ARCHIVE_TEST_MODEL=1 (uses the GPU or CPU for a few minutes)")
def test_real_model_finds_each_synthetic_picture_and_sound_window(tmp_path):
    from archive_index.embed import Embedder

    embedder = Embedder()
    projects = tmp_path / "Projects"
    pictures = {}
    for i, color in enumerate([(220, 30, 30), (30, 200, 60), (30, 60, 220), (240, 220, 30), (120, 40, 160)]):
        pictures[i] = projects / "p" / f"pic{i}.png"
        pictures[i].parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), color).save(pictures[i])
    tones = []
    for i, hz in enumerate([220.0, 440.0, 880.0, 1760.0, 3520.0]):
        path = projects / "p" / f"tone{i}.wav"
        _write_tone(path, hz, seconds=25.0)
        tones.append(path)
    store = Store(tmp_path / "real.sqlite3")
    run_index(store, embedder, projects_root=projects, home=tmp_path / "h", collections=("images", "audio"))
    picture_hits = sum(_top_path(store, embedder.image(str(p)), "images")[0] == str(p) for p in pictures.values())
    sound_hits = 0
    for path in tones:
        samples = np.asarray(_read(path), dtype=np.float32)
        window = samples[: 20 * 16000]
        sound_hits += _top_path(store, embedder.audio(window), "audio")[0] == str(path)
    store.close()
    assert picture_hits == 5, f"pictures found by their own vector: {picture_hits}/5"
    assert sound_hits == 5, f"sound windows found by their own samples: {sound_hits}/5"


def _read(path: Path):
    from archive_index.audio_io import read_audio

    return read_audio(path)
