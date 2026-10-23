import numpy as np
import pytest

from paper.eval.multimodal import _MemoizedEmbedder, retrieval_scores, text_to_audio, text_to_image


class Perfect:
    """Every item and its gold query map to the same one-hot vector."""

    def __init__(self, key):
        self.key = key
        self.labels = {}

    def _v(self, label):
        label = str(label)
        if label not in self.labels:
            self.labels[label] = len(self.labels)
        v = np.zeros(256, dtype=np.float32)
        v[self.labels[label]] = 1.0
        return v

    def query(self, text, prompt="SearchQuery"):
        return self._v(self.key(text))

    def image(self, path):
        return self._v(str(path).rsplit("/", 1)[-1].split(".")[0])

    def audio(self, samples, sample_rate=16000):
        return self._v(str(int(samples[0])))

    def documents(self, texts, titles=None):
        return np.stack([self._v(self.key(t)) for t in texts])


def test_retrieval_scores_recall_and_mrr():
    sims = np.array([[0.9, 0.1, 0.2], [0.1, 0.2, 0.8]])
    gold = [{0}, {0}]  # q2: item 0 is ranked third
    out = retrieval_scores(sims, gold, ks=(1, 2))
    assert out["recall@1"] == 0.5 and out["recall@2"] == 0.5 and out["mrr"] == pytest.approx((1 + 1 / 3) / 2)


def test_perfect_embedder_finds_every_image(tmp_path):
    images = {"a": tmp_path / "a.jpg", "b": tmp_path / "b.jpg"}
    queries = [("q1", "a", "a"), ("q2", "b", "b")]
    out = text_to_image(queries, images, Perfect(key=lambda t: t), ks=(1,))
    assert out["recall@1"] == 1.0


def test_audio_conditions(monkeypatch):
    import paper.eval.multimodal as mm

    monkeypatch.setattr(mm, "read_audio", lambda path: np.full(10, float(path.split("-")[1])))
    clips = [("c1", "clip-1", "7", "речення сім"), ("c2", "clip-2", "8", "речення вісім")]
    sentence_of = {"речення сім": "1", "речення вісім": "2", "предложение семь": "1"}
    emb = Perfect(key=lambda t: sentence_of.get(t, t))
    queries = [("q1", "речення сім", "7"), ("q2", "речення вісім", "8")]
    assert text_to_audio(queries, clips, emb, mode="audio", ks=(1,))["recall@1"] == 1.0
    assert text_to_audio(queries, clips, emb, mode="text", ks=(1,))["recall@1"] == 1.0
    assert text_to_audio(queries, clips, emb, mode="both", ks=(1,))["recall@1"] == 1.0
    cross = [("q3", "предложение семь", "7")]
    assert text_to_audio(cross, clips, emb, mode="text", ks=(1,))["recall@1"] == 1.0


def test_memoized_embedder_reuses_vectors_between_conditions():
    class Counting:
        def __init__(self):
            self.calls = {"query": 0, "documents": 0, "image": 0, "audio": 0}

        def query(self, text, prompt="SearchQuery"):
            self.calls["query"] += 1
            return np.ones(4, dtype=np.float32)

        def documents(self, texts, titles=None):
            self.calls["documents"] += 1
            return np.ones((len(texts), 4), dtype=np.float32)

        def image(self, path):
            self.calls["image"] += 1
            return np.ones(4, dtype=np.float32)

        def audio(self, samples, sample_rate=16000):
            self.calls["audio"] += 1
            return np.ones(4, dtype=np.float32)

    base = Counting()
    emb = _MemoizedEmbedder(base)
    samples = np.array([1, 2, 3], dtype=np.float32)
    emb.query("запит")
    emb.query("запит")
    emb.documents(["текст"], ["запис"])
    emb.documents(["текст"], ["запис"])
    emb.image("image.jpg")
    emb.image("image.jpg")
    emb.audio(samples)
    emb.audio(samples.copy())
    assert base.calls == {"query": 1, "documents": 1, "image": 1, "audio": 1}
