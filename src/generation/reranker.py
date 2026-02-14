"""
Cross-encoder reranker for improving retrieval precision.

Scores each (query, document) pair directly using cross-attention,
providing more accurate relevance ranking than bi-encoder similarity.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from src.utils.logger import get_logger
from src.indexing.vector_store import SearchResult
from config import load_config

config = load_config()
reranker_config = config["reranker"]
logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank search results by relevance to the query.

        Args:
            query: The user query
            results: Search results from vector store
            top_k: Number of top results to keep (None = use config default)

        Returns:
            Reranked list of SearchResult, sorted by reranker score (highest first)
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None, top_k: Optional[int] = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name or reranker_config["model"]
        self.top_k = top_k or reranker_config["top_k"]
        self.model = CrossEncoder(self.model_name)

        logger.info(f"Cross-encoder reranker initialized: {self.model_name} (top_k={self.top_k})")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder scores."""
        if not results:
            return []

        top_k = top_k or self.top_k

        # Score each (query, document) pair
        pairs = [(query, r.content) for r in results]
        scores = self.model.predict(pairs)

        # Pair results with their reranker scores and sort
        scored = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top K and update scores to reflect reranker ordering
        reranked = []
        for result, rerank_score in scored[:top_k]:
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=float(rerank_score),
                metadata=result.metadata,
            ))

        logger.info(
            f"Reranked {len(results)} → {len(reranked)} results "
            f"(top score: {reranked[0].score:.4f})"
        )
        return reranked


def get_reranker(provider: str = None, **kwargs) -> BaseReranker:
    """
    Factory function to create a reranker instance.

    Args:
        provider: Provider name ("cross-encoder")
        **kwargs: Override any constructor parameter

    Returns:
        BaseReranker instance
    """
    provider = provider or reranker_config["provider"]
    logger.info(f"Creating reranker: provider={provider}")

    if provider == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. "
            f"Supported: cross-encoder"
        )
