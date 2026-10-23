import hashlib
import json

import pytest

from paper.corpus.fetch import ChecksumError, Source, copy_for_release, fetch, load_sources


def _file(tmp_path, name, data=b"hello"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_fetch_rejects_wrong_checksum(tmp_path):
    src = _file(tmp_path, "a.bin")
    source = Source(name="a", part="notes", url=src.as_uri(), licence="CC-BY-4.0", redistributable=True,
                    sha256="0" * 64)
    with pytest.raises(ChecksumError):
        fetch(source, tmp_path / "raw")


def test_fetch_accepts_the_right_checksum_and_resumes(tmp_path):
    data = b"payload"
    src = _file(tmp_path, "b.bin", data)
    source = Source(name="b", part="notes", url=src.as_uri(), licence="CC-BY-4.0", redistributable=True,
                    sha256=hashlib.sha256(data).hexdigest())
    first = fetch(source, tmp_path / "raw")
    assert first.read_bytes() == data
    src.unlink()                       # second call must not download again
    assert fetch(source, tmp_path / "raw") == first


def test_non_redistributable_source_is_never_copied_to_release(tmp_path):
    path = _file(tmp_path, "c.bin", b"closed")
    source = Source(name="c", part="documents", url="https://example.org/c.bin", licence="proprietary",
                    redistributable=False, sha256=hashlib.sha256(b"closed").hexdigest())
    release = tmp_path / "release"
    assert copy_for_release(source, path, release) is None
    assert not any(release.rglob("c.bin"))
    entry = json.loads((release / "download_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry == {"name": "c", "url": "https://example.org/c.bin", "sha256": source.sha256, "licence": "proprietary"}


def test_sources_file_lists_every_part_with_a_licence():
    sources = load_sources()
    parts = {s.part for s in sources}
    assert {"notes", "documents", "tables", "code", "images", "speech"} <= parts
    assert all(s.licence and s.url for s in sources)
