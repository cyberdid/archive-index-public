import os
import tempfile

# Tests write their log to a temporary folder, never to the user's real archive.log.
os.environ.setdefault("ARCHIVE_LOG_DIR", tempfile.mkdtemp(prefix="archive-log-test-"))
