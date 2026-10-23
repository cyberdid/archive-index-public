"""RQ4: text-to-image and text-to-audio retrieval in one embedding space (plan Task 9).

Gold labels are human: XM3600 captions for images, FLEURS transcripts for speech. Speech is scored in three modes:
'audio' (query vs. the clip's audio vector), 'text' (query vs. the clip's transcript embedded as a document, as when
the archive attaches transcripts), and 'both' (the higher of the two similarities).
"""

from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import numpy as np

from archive_index.audio_io import read_audio

KS = (1, 5, 10)


class _MemoizedEmbedder:
    """Reuse identical embeddings across language and modality conditions in the full evaluation."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.cache: dict[tuple, np.ndarray] = {}

    def query(self, text: str, prompt: str = "SearchQuery") -> np.ndarray:
        key = ("query", prompt, text)
        if key not in self.cache:
            self.cache[key] = self.embedder.query(text, prompt=prompt)
        return self.cache[key]

    def documents(self, texts: list[str], titles=None) -> np.ndarray:
        key = ("documents", tuple(texts), tuple(titles or [""] * len(texts)))
        if key not in self.cache:
            self.cache[key] = self.embedder.documents(texts, titles)
        return self.cache[key]

    def image(self, path: str) -> np.ndarray:
        key = ("image", path)
        if key not in self.cache:
            self.cache[key] = self.embedder.image(path)
        return self.cache[key]

    def audio(self, samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        digest = hashlib.sha256(samples.tobytes()).digest()
        key = ("audio", sample_rate, samples.shape, digest)
        if key not in self.cache:
            self.cache[key] = self.embedder.audio(samples, sample_rate=sample_rate)
        return self.cache[key]


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def retrieval_scores(sims: np.ndarray, gold: list[set[int]], ks=KS) -> dict[str, float]:
    """sims: queries × items. recall@k = share of queries with a gold item in the top k; MRR over the full ranking."""
    order = np.argsort(-sims, axis=1)
    ranks = []
    for row, targets in zip(order, gold):
        hit = next((r for r, item in enumerate(row, start=1) if int(item) in targets), None)
        ranks.append(hit)
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    out["queries"] = n
    out["items"] = int(sims.shape[1])
    out.update({f"chance@{k}": min(1.0, k / sims.shape[1]) for k in ks})
    return out


def _query_matrix(embedder, texts: list[str]) -> np.ndarray:
    return _unit(np.stack([embedder.query(t) for t in texts]))


def text_to_image(queries, images: dict[str, Path], embedder, ks=KS) -> dict[str, float]:
    """queries: (qid, text, gold image id)."""
    ids = sorted(images)
    item_vectors = _unit(np.stack([embedder.image(str(images[i])) for i in ids]))
    index = {image_id: n for n, image_id in enumerate(ids)}
    sims = _query_matrix(embedder, [q[1] for q in queries]) @ item_vectors.T
    return retrieval_scores(sims, [{index[q[2]]} for q in queries], ks)


def text_to_audio(queries, clips, embedder, mode: str = "audio", ks=KS) -> dict[str, float]:
    """queries: (qid, text, gold sentence id); clips: (clip id, path, sentence id, transcript)."""
    q = _query_matrix(embedder, [x[1] for x in queries])
    sims = None
    if mode in ("audio", "both"):
        audio = _unit(np.stack([embedder.audio(np.asarray(read_audio(c[1]), dtype=np.float32)) for c in clips]))
        sims = q @ audio.T
    if mode in ("text", "both"):
        text = _unit(embedder.documents([c[3] for c in clips], ["recording"] * len(clips)))
        text_sims = q @ text.T
        sims = text_sims if sims is None else np.maximum(sims, text_sims)
    gold = [{n for n, c in enumerate(clips) if c[2] == x[2]} for x in queries]
    return retrieval_scores(sims, gold, ks)


def _captions(zip_path: Path) -> dict[str, dict[str, list[str]]]:
    out = {}
    with zipfile.ZipFile(zip_path) as z, z.open("captions.jsonl") as fh:
        for line in fh:
            record = json.loads(line)
            out[record["image/key"]] = {lang: record[lang]["caption"] for lang in ("uk", "ru") if lang in record}
    return out


def run(manifest: list[dict], raw_dir: Path, embedder) -> dict:
    """Every condition of RQ4 on the corpus: images by uk/ru captions, speech monolingual and cross-lingual."""
    embedder = _MemoizedEmbedder(embedder)
    results: dict = {"images": {}, "speech": {}}
    images = {e["image_id"]: Path(e["path"]) for e in manifest if e["part"] == "images"}
    captions = _captions(raw_dir / "xm3600_captions" / "captions.zip")
    for lang in ("uk", "ru"):
        queries = [(f"{i}-{lang}", captions[i][lang][0], i) for i in sorted(images) if lang in captions.get(i, {})]
        results["images"][lang] = text_to_image(queries, images, embedder)
    speech = [e for e in manifest if e["part"] == "speech"]
    by_lang = {lang: [(e["doc_id"], e["path"], e["sentence_id"], e["transcript"]) for e in speech if e["lang"] == lang]
               for lang in ("uk", "ru")}
    for clip_lang in ("uk", "ru"):
        clips = by_lang[clip_lang]
        for query_lang in ("uk", "ru"):
            source = by_lang[query_lang]
            seen, queries = set(), []
            for clip_id, _path, sentence_id, transcript in source:
                if sentence_id not in seen and any(c[2] == sentence_id for c in clips):
                    seen.add(sentence_id)
                    queries.append((f"{query_lang}-{sentence_id}", transcript, sentence_id))
            key = f"{query_lang}->{clip_lang}"
            results["speech"][key] = {mode: text_to_audio(queries, clips, embedder, mode)
                                      for mode in ("audio", "text", "both")}
    return results


def main() -> int:
    from archive_index.embed import Embedder
    from paper.corpus.build import read_manifest

    manifest = read_manifest(Path("paper/data/ua-par/manifest.jsonl"))
    results = run(manifest, Path("paper/data/raw"), Embedder())
    out = Path("paper/results/multimodal.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
