"""
Embedding abstraction layer.

Supports multiple embedding providers:
- SentenceTransformers (local, free)
- OpenAI (API, paid)
"""
from src.utils.logger import get_logger
from config import load_config

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

config = load_config()
embedding_config = config["embedding"]
logger = get_logger(__name__)

class BaseEmbedder(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Embed single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass

class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence transformers embedder (local, free)."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize sentence transformers embedder.

        Args:
            model_name: Model name (default: all-mpnet-base-v2)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = embedding_config["model"] or model_name
        logger.info(f"Loading sentence transformer model: {self.model_name}")

        self.model = SentenceTransformer(model_name)

        logger.info(f"Model loaded. Dimension: {self.dimension}")

    def embed(self, text: str) -> List[float]:
        """Embed single text."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """Embed batch of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return embeddings.tolist()
    
    # For quick testing
    @staticmethod
    def similarity(emb_a: List[float], emb_b: List[float]) -> float:
        """Cosine similarity between two embeddings."""
        from sentence_transformers import util
        return util.cos_sim(emb_a, emb_b).item()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    
def get_embedder(provider: str = None, model: str = None) -> BaseEmbedder:
    """
    Get embedder based on provider.

    Args:
        provider: Provider name ("sentence-transformers", "openai")
        model: Model name (provider-specific)

    Returns:
        Embedder instance
    """
    provider = provider or embedding_config["provider"]
    model = model or embedding_config["model"]

    logger.info(f"Creating embedder: provider={provider}, model={model}")

    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: sentence-transformers, openai"
        )


if __name__ == "__main__":
    embedder = get_embedder()

    # Single embedding
    text = "Microsoft reported strong earnings in Q4 2024."
    embedding = embedder.embed(text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")

    # Batch embedding
    texts = [
        "Apple's revenue grew 10% year-over-year.",
        "Apple's profit rose 8% compared to last year."
        "NVIDIA's GPU sales surged due to AI demand.",
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"\nBatch embeddings: {len(embeddings)} vectors")

    similarities = embedder.similarity(embeddings[0], embeddings[1])
    print(similarities)