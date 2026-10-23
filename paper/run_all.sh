#!/usr/bin/env bash
# quick: saved-result rendering; full: all frozen-input statistical analyses.
# Neither mode makes model calls or repeats physical measurements implicitly.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PAPER_PYTHON=${PAPER_PYTHON:-python3}
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PAPER_PYTHON" -m paper.reproduce "$@"
