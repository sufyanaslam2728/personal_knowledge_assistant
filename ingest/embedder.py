# ingest/embedder.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        texts = [d.page_content for d in docs]
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
        # ensure float32 for FAISS
        return vecs.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
