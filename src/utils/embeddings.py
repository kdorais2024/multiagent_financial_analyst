"""
FAISS vector store utilities for caching and semantic search.

Used to:
- Cache previous research results for fast retrieval
- Enable semantic search over historical research memos
- Store document embeddings for the RAG pipeline
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

from configs.settings import settings


class FAISSStore:
    """
    Lightweight FAISS wrapper for embedding storage and retrieval.

    Supports:
    - Adding documents with metadata
    - Similarity search with score thresholds
    - Persistence to disk
    """

    def __init__(self, dimension: int = 1536, index_path: str | None = None):
        """
        Initialize the FAISS store.

        Args:
            dimension: Embedding dimension (1536 for text-embedding-3-small)
            index_path: Path to persist the index. Uses settings default if None.
        """
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            self._faiss = None

        self.dimension = dimension
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.index = None
        self.metadata: list[dict[str, Any]] = []

        self._initialize_index()

    def _initialize_index(self):
        """Create or load the FAISS index."""
        if self._faiss is None:
            return

        meta_path = f"{self.index_path}.meta"

        if os.path.exists(self.index_path) and os.path.exists(meta_path):
            self.index = self._faiss.read_index(self.index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = self._faiss.IndexFlatL2(self.dimension)

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]):
        """
        Add embeddings with associated metadata.

        Args:
            embeddings: numpy array of shape (n, dimension)
            metadata: list of metadata dicts, one per embedding
        """
        if self.index is None:
            return

        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector of shape (1, dimension)
            k: Number of results to return
            score_threshold: Maximum L2 distance (lower = more similar)

        Returns:
            List of metadata dicts with added 'score' field
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(
            query_embedding.astype(np.float32).reshape(1, -1), k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if score_threshold and dist > score_threshold:
                continue
            result = self.metadata[idx].copy()
            result["score"] = float(dist)
            results.append(result)

        return results

    def save(self):
        """Persist the index and metadata to disk."""
        if self.index is None:
            return

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self._faiss.write_index(self.index, self.index_path)
        with open(f"{self.index_path}.meta", "wb") as f:
            pickle.dump(self.metadata, f)

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self.index.ntotal if self.index else 0
