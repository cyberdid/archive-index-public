"""Fetch the UA-PAR source files with checksum checks, and decide what may go into the release (plan Task 3)."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

SOURCES = Path(__file__).with_name("sources.yaml")


class ChecksumError(RuntimeError):
    """The downloaded file does not have the pinned sha256."""


@dataclass(frozen=True)
class Source:
    name: str
    part: str
    url: str
    licence: str
    redistributable: bool
    sha256: str | None = None
    note: str = ""


def load_sources(path: Path = SOURCES) -> list[Source]:
    return [Source(**item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source: Source, dest: Path) -> Path:
    """Download one file into dest/<name>/<basename>; skip it when a verified copy is already there."""
    target = dest / source.name / source.url.rstrip("/").rsplit("/", 1)[-1]
    if target.exists() and (source.sha256 is None or sha256_of(target) == source.sha256):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(source.url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if source.sha256 is not None and sha256_of(partial) != source.sha256:
        partial.unlink()
        raise ChecksumError(f"{source.name}: sha256 mismatch for {source.url}")
    partial.replace(target)
    return target


def copy_for_release(source: Source, path: Path, release_dir: Path) -> Path | None:
    """Copy a redistributable file into the release; for any other file write only its URL and sha256."""
    release_dir.mkdir(parents=True, exist_ok=True)
    if source.redistributable:
        target = release_dir / source.part / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return target
    entry = {"name": source.name, "url": source.url, "sha256": source.sha256 or sha256_of(path),
             "licence": source.licence}
    with (release_dir / "download_manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return None
