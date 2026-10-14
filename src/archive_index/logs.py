"""File logging for archive-index: the TUI, the CLI and the indexer write one rotating log.

Location: ~/.local/state/archive-index/archive.log (override with ARCHIVE_LOG_DIR, level with
ARCHIVE_LOG_LEVEL). Errors carry the full traceback and the action that caused them. The log holds
queries, paths and counts; it does not hold the text of fragments.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = "archive_index"


def log_dir() -> Path:
    return Path(os.environ.get("ARCHIVE_LOG_DIR", Path.home() / ".local" / "state" / "archive-index"))


def setup_logging() -> Path:
    """Attach the file handler once per process. Returns the log file path."""
    path = log_dir() / "archive.log"
    root = logging.getLogger(ROOT)
    if not any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path for h in root.handlers):
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [pid %(process)d]: %(message)s"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("ARCHIVE_LOG_LEVEL", "DEBUG").upper())
        root.propagate = False
    return path


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"{ROOT}.{name}")
