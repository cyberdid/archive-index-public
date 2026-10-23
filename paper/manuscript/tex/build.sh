#!/usr/bin/env bash
# Build the archive-index paper: English and Ukrainian PDFs with Tectonic (XeTeX), as for the author's FFT paper.
# Output: paper/manuscript/article-en.pdf, paper/manuscript/article-uk.pdf. Usage: bash paper/manuscript/tex/build.sh [en|uk|all]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(cd "$HERE/.." && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PAPER_PYTHON=${PAPER_PYTHON:-python3}
TECTONIC=${TECTONIC:-$(command -v tectonic || echo "$HOME/.local/bin/tectonic")}
which=${1:-all}
case "$which" in en|uk|all) ;; *) echo 'Usage: build.sh [en|uk|all]' >&2; exit 2 ;; esac
# A standalone build must reject stale or hand-edited numbers, just as the
# orchestration command does. Select the research interpreter with PAPER_PYTHON.
(cd "$ROOT" && PYTHONPATH="src:.:${PYTHONPATH:-}" "$PAPER_PYTHON" -m paper.report.validate)
for lang in en uk; do
  [[ $which == all || $which == "$lang" ]] || continue
  (cd "$HERE/$lang" && "$TECTONIC" --keep-logs --outdir "$HERE/$lang" main.tex)
  cp "$HERE/$lang/main.pdf" "$OUT/article-$lang.pdf"
  echo "built $OUT/article-$lang.pdf"
done
