import io
import json
import tarfile
import wave

import numpy as np
from PIL import Image

from paper.corpus.build import build_corpus, language_shares, read_manifest


def _wiki(raw, lang, pageid, topic, words=400):
    d = raw / f"wiki_{lang}"
    d.mkdir(parents=True, exist_ok=True)
    text = ("Перший розділ про радіо і антени. " if lang == "uk" else "Первый раздел о радио и антеннах. ") * (words // 6)
    (d / f"{pageid}.json").write_text(json.dumps({"pageid": pageid, "title": f"Стаття password {pageid}",
                                                  "url": f"https://{lang}.wikipedia.org/?curid={pageid}",
                                                  "text": text, "topic": topic}, ensure_ascii=False), encoding="utf-8")


def _tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((np.zeros(int(16000 * seconds))).astype("<i2").tobytes())
    return buf.getvalue()


def _fixture(tmp_path):
    raw = tmp_path / "raw"
    for i in range(6):
        _wiki(raw, "uk", 100 + i, "radio")
    for i in range(3):
        _wiki(raw, "ru", 200 + i, "radio")
    t = raw / "datagovua"
    t.mkdir(parents=True)
    (t / "ds1.csv").write_text("назва;кількість\nантена;3\nкабель;7\n", encoding="utf-8")
    (t / "ds1.json").write_text(json.dumps({"id": "ds1", "title": "Обладнання", "url": "https://data.gov.ua/dataset/ds1",
                                            "licence": "cc-by"}), encoding="utf-8")
    repo = raw / "github" / "owner__tool"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("# tool\n\nУкраїнська документація інструмента.\n", encoding="utf-8")
    (repo / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (repo / "meta.json").write_text(json.dumps({"full_name": "owner/tool", "url": "https://github.com/owner/tool",
                                                "licence": "MIT"}), encoding="utf-8")
    img = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(img, "JPEG")
    _tar(raw / "xm3600_images" / "images.tgz", [("images/aaa111.jpg", img.getvalue()), ("images/bbb222.jpg", img.getvalue())])
    a = raw / "xm3600_attributions"
    a.mkdir(parents=True)
    (a / "image_attributions.csv").write_text(
        "ImageID,OriginalLandingURL,License,Author\naaa111,https://flickr/1,https://creativecommons.org/licenses/by/2.0/,A\n"
        "bbb222,https://flickr/2,https://creativecommons.org/licenses/by/2.0/,B\n", encoding="utf-8")
    for lang in ("uk", "ru"):
        _tar(raw / f"fleurs_{lang}_audio" / "test.tar.gz", [("test/1.wav", _wav_bytes()), ("test/2.wav", _wav_bytes())])
        d = raw / f"fleurs_{lang}_text"
        d.mkdir(parents=True)
        (d / "test.tsv").write_text("7\t1.wav\tРечення один.\tречення один\tр е\t16000\tMALE\n"
                                    "7\t2.wav\tРечення один.\tречення один\tр е\t16000\tFEMALE\n", encoding="utf-8")
    return raw


SIZE = {"notes": 6, "documents": 3, "tables": 1, "code": 1, "images": 2, "speech": 4}


def test_manifest_lists_every_file_once_with_sha256(tmp_path):
    raw = _fixture(tmp_path)
    out = tmp_path / "ua-par"
    build_corpus(raw, out, size=SIZE, render_pdf=False)
    entries = read_manifest(out / "manifest.jsonl")
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "manifest.jsonl")
    assert sorted(e["doc_id"] for e in entries) == files
    assert all(len(e["sha256"]) == 64 and e["source"] for e in entries)
    parts = {e["part"] for e in entries}
    assert parts == {"notes", "documents", "tables", "code", "images", "speech"}


def test_language_shares_meet_spec(tmp_path):
    raw = _fixture(tmp_path)
    out = tmp_path / "ua-par"
    build_corpus(raw, out, size=SIZE, render_pdf=False)
    shares = language_shares(read_manifest(out / "manifest.jsonl"))
    assert shares["uk"] >= 0.40 and shares["ru"] >= 0.20


def test_file_names_do_not_depend_on_titles(tmp_path):
    """Titles like 'password …' would hit the archive's sensitive-name filter and silently vanish."""
    raw = _fixture(tmp_path)
    out = tmp_path / "ua-par"
    build_corpus(raw, out, size=SIZE, render_pdf=False)
    assert not any("password" in p.name.lower() for p in out.rglob("*"))


def test_speech_entries_carry_their_transcript(tmp_path):
    raw = _fixture(tmp_path)
    out = tmp_path / "ua-par"
    build_corpus(raw, out, size=SIZE, render_pdf=False)
    speech = [e for e in read_manifest(out / "manifest.jsonl") if e["part"] == "speech"]
    assert speech and all(e["transcript"] == "Речення один." and e["sentence_id"] == "7" for e in speech)


def test_non_tabular_csv_resource_is_skipped_and_quota_is_filled(tmp_path):
    raw = _fixture(tmp_path)
    tables = raw / "datagovua"
    (tables / "000-link.csv").write_text("https://example.org/file\n", encoding="utf-8")
    (tables / "000-link.json").write_text(json.dumps({"id": "000-link", "title": "Link",
                                                       "url": "https://data.gov.ua/dataset/000-link",
                                                       "licence": "cc-by"}), encoding="utf-8")
    out = tmp_path / "ua-par"

    build_corpus(raw, out, size=SIZE, render_pdf=False)

    tables_in_manifest = [e for e in read_manifest(out / "manifest.jsonl") if e["part"] == "tables"]
    assert len(tables_in_manifest) == 1
    assert tables_in_manifest[0]["doc_id"].endswith("ds1.csv")


def test_utf16le_csv_is_decoded_without_losing_text(tmp_path):
    raw = _fixture(tmp_path)
    source = raw / "datagovua" / "ds1.csv"
    source.write_text("назва;кількість\nантена;3\nкабель;7\n", encoding="utf-16")
    out = tmp_path / "ua-par"

    build_corpus(raw, out, size=SIZE, render_pdf=False)

    table = next(out.rglob("ds1.csv"))
    assert "антена" in table.read_text(encoding="utf-8")
