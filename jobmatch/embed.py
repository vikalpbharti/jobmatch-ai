"""Turn text into L2-normalised vectors, so a dot product is cosine similarity.

Two backends:
- "st":    sentence-transformers embeddings (semantic: matches meaning, not just words)
- "tfidf": TF-IDF keyword vectors, kept as the baseline the embeddings are evaluated against
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class Embedder:
    def __init__(self, backend: str = "st", model_name: str = DEFAULT_MODEL):
        if backend not in ("st", "tfidf"):
            raise ValueError("backend must be 'st' or 'tfidf'")
        self.backend = backend
        self.model_name = model_name if backend == "st" else "tfidf"
        self._model = None

    # --- setup -------------------------------------------------------------
    def fit(self, corpus: list[str]) -> "Embedder":
        """Only TF-IDF needs fitting on the corpus; the transformer model is pre-trained."""
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._model = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
            self._model.fit(corpus)
        return self

    def _load(self):
        if self._model is None and self.backend == "st":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        if self._model is None:
            raise RuntimeError("TF-IDF embedder must be fitted (or loaded) before encoding")
        return self._model

    # --- encoding ----------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        model = self._load()
        if self.backend == "st":
            vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True)
        else:
            vecs = model.transform(texts).toarray()
        return _normalise(np.asarray(vecs, dtype=np.float32))

    # --- persistence (TF-IDF vocabulary must be saved with the index) ------
    def save(self, directory: Path) -> None:
        if self.backend == "tfidf":
            joblib.dump(self._model, Path(directory) / "tfidf.joblib")

    @classmethod
    def load(cls, directory: Path, backend: str, model_name: str) -> "Embedder":
        emb = cls(backend=backend, model_name=model_name)
        if backend == "tfidf":
            emb._model = joblib.load(Path(directory) / "tfidf.joblib")
        return emb
