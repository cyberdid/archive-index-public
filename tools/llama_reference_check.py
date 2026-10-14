"""Check the llama-server backend (plan D3) against the SentenceTransformers reference vectors.

Start the server first, e.g.:
  llama-server -m embeddinggemma2-f16.gguf --embeddings --pooling mean -ngl 0 --port 8181
Then: python tools/llama_reference_check.py http://127.0.0.1:8181 [min_cosine]
Exit 1 if any text vector has cosine below min_cosine (default 0.999, for the F16 model).
Quantized models are checked with a lower bar, e.g. 0.985 for Q4_K_M (measured 0.9885 worst case).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_check import TEXTS  # the same fixed sentences as the reference file

from archive_index.embed import LlamaServerEmbedder


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8181"
    bar = float(sys.argv[2]) if len(sys.argv) > 2 else 0.999
    ref = np.load("evals/reference.npz")
    vectors = LlamaServerEmbedder(url).documents(TEXTS, ["ref"] * len(TEXTS))
    worst = 1.0
    for i, vector in enumerate(vectors):
        expected = ref[f"text{i}"]
        cos = float(np.dot(vector, expected) / (np.linalg.norm(vector) * np.linalg.norm(expected)))
        worst = min(worst, cos)
        print(f"text{i} dim={len(vector)} cos={cos:.5f}")
    print(f"worst cosine {worst:.5f} (bar {bar}) ->", "ok" if worst >= bar else "FAIL")
    return 0 if worst >= bar else 1


if __name__ == "__main__":
    raise SystemExit(main())
