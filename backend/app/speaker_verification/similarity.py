"""
Similarity Computation for Speaker Verification.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSimilarityMetric(ABC):
    """
    Abstract interface for computing similarity between speaker embeddings.
    """
    @abstractmethod
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        pass


class CosineSimilarityMetric(BaseSimilarityMetric):
    """
    Computes Cosine Similarity between two speaker embeddings.
    """
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes cosine similarity in [-1.0, 1.0].
        Args:
            embedding1: 1D numpy array of shape (D,)
            embedding2: 1D numpy array of shape (D,)
        Returns:
            float similarity score
        """
        if embedding1 is None or embedding2 is None:
            raise ValueError("Embeddings cannot be None.")

        e1 = np.asarray(embedding1, dtype=np.float32).flatten()
        e2 = np.asarray(embedding2, dtype=np.float32).flatten()

        if e1.shape[0] != e2.shape[0]:
            raise ValueError(f"Embedding dimension mismatch: {e1.shape[0]} vs {e2.shape[0]}.")

        if not np.all(np.isfinite(e1)) or not np.all(np.isfinite(e2)):
            raise ValueError("Embeddings contain non-finite values (NaN or Inf).")

        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0

        dot_product = np.dot(e1, e2)
        cosine_sim = float(dot_product / (norm1 * norm2))

        # Clamp safely within mathematical [-1.0, 1.0] range
        return max(-1.0, min(1.0, cosine_sim))
