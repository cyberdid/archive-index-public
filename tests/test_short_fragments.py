from pathlib import Path

import numpy as np

from archive_index.indexer import MIN_LETTERS, drop_short_fragments, letter_count
from archive_index.store import ChunkRecord, Store


def _piece(body, locator="page 1", image="", audio=None):
    return ("", locator, "", body, image, audio)


def test_page_numbers_and_punctuation_are_dropped_from_documents():
    pieces = [_piece("3"), _piece("Антена ретранслятора працює на частоті 1800 МГц у вікні."), _piece(", ")]
    kept = drop_short_fragments("docs", pieces)
    assert [p[3] for p in kept] == ["Антена ретранслятора працює на частоті 1800 МГц у вікні."]
    assert letter_count("3") < MIN_LETTERS


def test_a_document_made_only_of_short_text_keeps_it():
    pieces = [_piece("Так."), _piece("Ні.")]
    assert drop_short_fragments("docs", pieces) == pieces


def test_images_and_sound_are_never_dropped_and_other_collections_are_untouched():
    image = _piece("solid.png", image="/x/solid.png")
    sound = _piece("0-20s", audio=np.zeros(4, dtype=np.float32))
    text = [_piece("3"), _piece("Довгий текст про антену і ретранслятор, достатньо літер для пошуку.")]
    assert drop_short_fragments("images", [image] + text) == [image] + text
    kept = drop_short_fragments("docs", [image, sound, _piece("3"), _piece("Довгий текст про антену.")])
    assert image in kept and sound in kept and _piece("3") not in kept


def test_existing_index_loses_only_short_fragments_of_mixed_documents(tmp_path):
    vec = np.ones(4, dtype=np.float32)
    store = Store(tmp_path / "s.sqlite3")
    store.replace_document(collection="docs", path="/p/mixed.pdf", size=1, mtime_ns=1, sha256="m", title="m",
                           project="p", chunks=[
                               ChunkRecord(ord=0, heading="", locator="page 1", body="7", terms="7",
                                           occurred_at="", vector=vec, body_sha="a"),
                               ChunkRecord(ord=1, heading="", locator="page 2",
                                           body="Довгий текст сторінки про антену і ретранслятор.",
                                           terms="t", occurred_at="", vector=vec, body_sha="b")])
    store.replace_document(collection="docs", path="/p/short.txt", size=1, mtime_ns=1, sha256="s", title="s",
                           project="p", chunks=[
                               ChunkRecord(ord=0, heading="", locator="text", body="Так.", terms="t",
                                           occurred_at="", vector=vec, body_sha="c")])
    assert store.remove_short_fragments(("docs",), MIN_LETTERS) == 1
    counts = dict(store.conn.execute("SELECT d.path, COUNT(*) FROM chunks c JOIN documents d ON d.id=c.doc_id "
                                     "GROUP BY d.path").fetchall())
    assert counts == {"/p/mixed.pdf": 1, "/p/short.txt": 1}
    store.close()


def test_a_reader_does_not_block_the_indexer_writing(tmp_path):
    """Regression: 'database is locked' while the interface read and the timer wrote at the same time."""
    from archive_index.store import ChunkRecord

    db = tmp_path / "shared.sqlite3"
    writer = Store(db)
    reader = Store(db)
    vec = np.ones(4, dtype=np.float32)
    writer.replace_document(collection="docs", path="/p/a.txt", size=1, mtime_ns=1, sha256="a", title="a",
                            project="p", chunks=[ChunkRecord(ord=0, heading="", locator="t", body="один два три четыре",
                                                             terms="t", occurred_at="", vector=vec, body_sha="a")])
    cur = reader.conn.execute("SELECT COUNT(*) FROM chunks")  # an open read while the writer commits
    writer.replace_document(collection="docs", path="/p/b.txt", size=1, mtime_ns=1, sha256="b", title="b",
                            project="p", chunks=[ChunkRecord(ord=0, heading="", locator="t", body="пять шість сім",
                                                             terms="t", occurred_at="", vector=vec, body_sha="b")])
    assert cur.fetchone()[0] == 1
    assert reader.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    writer.close()
    reader.close()
