# ingest/indexer.py
from __future__ import annotations
from typing import List, Dict, Any
import os
import pickle
import faiss
import numpy as np

class FaissStore:
    """
    Minimal FAISS wrapper (cosine similarity via normalized inner product).
    Persists:
      - FAISS index (binary)
      - metadata/texts (pickle)
    """

    def __init__(self, index_path: str, store_path: str):
        self.index_path = index_path
        self.store_path = store_path
        self.index: faiss.Index | None = None
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def _ensure_index(self, dim: int):
        if self.index is None:
            # Inner product + L2 normalization -> cosine similarity
            self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict]):
        assert embeddings.ndim == 2, "embeddings must be [n, d]"
        assert len(texts) == embeddings.shape[0] == len(metadatas)
        self._ensure_index(embeddings.shape[1])

        # Normalize for cosine
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump(
                {"texts": self.texts, "metadatas": self.metadatas},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self):
        if not (os.path.exists(self.index_path) and os.path.exists(self.store_path)):
            raise FileNotFoundError("Index or store file not found.")
        self.index = faiss.read_index(self.index_path)
        with open(self.store_path, "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]

    def search(self, query_vec: np.ndarray, k: int = 5):
        assert self.index is not None, "Index not loaded"
        q = query_vec.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        results = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append(
                {"text": self.texts[i], "metadata": self.metadatas[i], "score": float(s)}
            )
        return results
