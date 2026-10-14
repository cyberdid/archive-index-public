"""EmbeddingGemma 2 wrapper: documents get 'title | text' prefixes, queries use SearchQuery."""

from __future__ import annotations

import numpy as np

MODEL_ID = "google/embeddinggemma-2"


class Embedder:
    """Loads the model lazily so that imports and tests stay cheap."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs = {}
            if device == "cuda":
                # The model card says float16 silently degrades these embeddings.
                kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                )
            self._model = SentenceTransformer(self.model_id, device=device, model_kwargs=kwargs)
        return self._model

    def documents(self, texts: list[str], titles: list[str] | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefixed = [
            f"title: {title or 'none'} | text: {text}"
            for title, text in zip(titles or [""] * len(texts), texts)
        ]
        vectors = self._load().encode(
            prefixed, batch_size=16, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def query(self, text: str, prompt: str = "SearchQuery") -> np.ndarray:
        """prompt: 'SearchQuery' for documents, 'CodeRetrieval' for the code collection (model prompt names)."""
        vector = self._load().encode(
            text, prompt_name=prompt, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vector, dtype=np.float32)

    def image(self, path: str) -> np.ndarray:
        vector = self._load().encode(
            {"image": str(path)}, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vector, dtype=np.float32).reshape(-1)

    def audio(self, samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        vector = self._load().encode(
            {"audio": {"array": samples, "sampling_rate": sample_rate}},
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32).reshape(-1)



# The same prefixes the model card defines for SearchQuery and CodeRetrieval.
QUERY_PREFIX = {"SearchQuery": "task: search result | query: ", "CodeRetrieval": "task: code retrieval | query: "}


def query_prompt(collections) -> str:
    """Code searches use the code-retrieval prompt; everything else the general search prompt (plan E)."""
    names = {c for c in (collections or []) if c}
    return "CodeRetrieval" if names == {"code"} else "SearchQuery"


class LlamaServerEmbedder:
    """Fallback backend: a llama-server (llama.cpp) running the GGUF build of EmbeddingGemma 2 (plan D3).

    Text only: the server is started with --embeddings --pooling mean. Prefixes are added here,
    exactly as in the SentenceTransformers path. Images and sound need the full model.
    """

    def __init__(self, url: str = "http://127.0.0.1:8080") -> None:
        self.url = url.rstrip("/")

    def _post(self, inputs: list[str]) -> np.ndarray:
        import json
        import urllib.request

        body = json.dumps({"input": inputs}).encode("utf-8")
        request = urllib.request.Request(self.url + "/v1/embeddings", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))["data"]
        vectors = np.asarray([item["embedding"] for item in sorted(data, key=lambda d: d["index"])],
                             dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)  # re-normalise, as the model card requires

    def documents(self, texts: list[str], titles: list[str] | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefixed = [f"title: {title or 'none'} | text: {text}"
                    for title, text in zip(titles or [""] * len(texts), texts)]
        return self._post(prefixed)

    def query(self, text: str, prompt: str = "SearchQuery") -> np.ndarray:
        return self._post([f"{QUERY_PREFIX[prompt]}{text}"])[0]

    def image(self, path: str) -> np.ndarray:
        raise NotImplementedError("images need the full model; llama-server backend is text only")

    def audio(self, samples, sample_rate: int = 16_000) -> np.ndarray:
        raise NotImplementedError("audio needs the full model; llama-server backend is text only")


def make_embedder():
    """Full model by default; ARCHIVE_EMBED_BACKEND=llama-server selects the GGUF server."""
    import os

    if os.environ.get("ARCHIVE_EMBED_BACKEND") == "llama-server":
        return LlamaServerEmbedder(os.environ.get("ARCHIVE_LLAMA_URL", "http://127.0.0.1:8080"))
    return Embedder()
