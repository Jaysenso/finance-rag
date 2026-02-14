"""
Qdrant vector store for SEC financial document chunks.

Supports cloud and local modes.
"""
from src.utils.logger import get_logger
from config import load_config
from src.preprocessing.chunking import Chunk

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

config = load_config()
vs_config = config["vector_store"]
embedding_config = config["embedding"]
logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class QdrantVectorStore:
    """Qdrant-backed vector store for SEC filing chunks."""

    def __init__(
        self,
        collection_name: str = None,
        mode: str = None,
        url: str = None,
        api_key: str = None,
        path: str = None,
        distance: str = None,
        dimension: int = None,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client not installed. "
                "Install with: pip install qdrant-client"
            )

        self.collection_name = collection_name or vs_config["collection_name"]
        self.distance = distance or vs_config["distance"]
        self.dimension = dimension or embedding_config["dimension"]
        self.batch_size = vs_config["batch_size"]

        mode = mode or vs_config["mode"]

        if mode == "cloud":
            url = url or vs_config["url"]
            api_key = api_key or os.getenv("QDRANT_API_KEY") or vs_config["api_key"]
            if not url:
                raise ValueError("Qdrant Cloud URL is required in cloud mode. Set it in config.yaml or pass url=...")
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"Qdrant cloud client initialized: {url}")
        else:
            path = path or vs_config["path"]
            self.client = QdrantClient(path=path)
            logger.info(f"Qdrant local client initialized: {path}")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist, with payload indexes."""
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Collection '{self.collection_name}' exists "
                f"({info.points_count} points)"
            )
            return

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=distance_map[self.distance],
            ),
        )
        logger.info(
            f"Created collection '{self.collection_name}' "
            f"(dim={self.dimension}, distance={self.distance})"
        )

        # Create payload indexes for filterable fields
        index_fields = {
            "company": "keyword",
            "document_type": "keyword",
            "filing_date": "keyword",
            "has_table": "bool",
            "has_chart": "bool",
            "parent_doc_id": "keyword",
        }
        for field_name, field_type in index_fields.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
        logger.info(f"Created payload indexes: {list(index_fields.keys())}")

    @staticmethod
    def _chunk_to_payload(chunk: Chunk) -> Dict[str, Any]:
        """Convert a Chunk to a Qdrant payload dict."""
        return {
            "content": chunk.content,
            "company": chunk.company,
            "document_type": chunk.document_type,
            "filing_date": chunk.filing_date,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "parent_doc_id": chunk.parent_doc_id,
            "has_table": chunk.has_table,
            "has_chart": chunk.has_chart,
            "token_count": chunk.token_count,
        }

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        """
        Upsert chunks with their embeddings into the vector store.

        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (same order as chunks)
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=self._chunk_to_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        for i in range(0, len(points), self.batch_size):
            batch = points[i : i + self.batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} points to '{self.collection_name}'")

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        company: Optional[str | List[str]] = None,
        document_type: Optional[str | List[str]] = None,
        filing_date: Optional[str | List[str]] = None,
        has_table: Optional[bool] = None,
        has_chart: Optional[bool] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks with optional metadata filters.

        Args:
            query_embedding: Query vector
            limit: Max results to return
            company: Filter by ticker(s) - "AAPL" or ["AAPL", "MSFT"]
            document_type: Filter by filing type(s) - "10-K" or ["10-K", "10-Q"]
            filing_date: Filter by date(s)
            has_table: Filter for chunks containing tables
            has_chart: Filter for chunks containing charts
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult ordered by similarity (highest first)
        """
        query_filter = self._build_filter(
            company=company,
            document_type=document_type,
            filing_date=filing_date,
            has_table=has_table,
            has_chart=has_chart,
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        search_results = []
        for point in results.points:
            payload = point.payload
            search_results.append(SearchResult(
                chunk_id=point.id,
                content=payload["content"],
                score=point.score,
                metadata={k: v for k, v in payload.items() if k != "content"},
            ))

        logger.info(
            f"Search returned {len(search_results)} results "
            f"(limit={limit}, filtered={query_filter is not None})"
        )
        return search_results

    def _build_filter(
        self,
        company: Optional[str | List[str]] = None,
        document_type: Optional[str | List[str]] = None,
        filing_date: Optional[str | List[str]] = None,
        has_table: Optional[bool] = None,
        has_chart: Optional[bool] = None,
    ):
        """Build Qdrant Filter from search parameters."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        conditions = []

        for key, value in [
            ("company", company),
            ("document_type", document_type),
            ("filing_date", filing_date),
        ]:
            if value is None:
                continue

            if isinstance(value, str):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, list) and len(value) == 1:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value[0]))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )

        for key, value in [("has_table", has_table), ("has_chart", has_chart)]:
            if value is not None:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        if not conditions:
            return None

        return Filter(must=conditions)

    def retrieve_by_ids(self, chunk_ids: List[str]) -> List[SearchResult]:
        """
        Retrieve points by their chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of SearchResult objects (without scores since this isn't a search)
        """
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_ids,
            with_payload=True,
            with_vectors=False,
        )
        
        search_results = []
        for point in points:
            payload = point.payload
            search_results.append(SearchResult(
                chunk_id=point.id,
                content=payload["content"],
                score=0.0,  # No score for direct retrieval
                metadata={k: v for k, v in payload.items() if k != "content"},
            ))
        
        logger.info(f"Retrieved {len(search_results)} points by ID")
        return search_results


    def delete(self, chunk_ids: List[str]) -> None:
        """Delete points by chunk IDs."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=chunk_ids,
        )
        logger.info(f"Deleted {len(chunk_ids)} points from '{self.collection_name}'")

    def delete_by_filter(
        self,
        company: Optional[str] = None,
        document_type: Optional[str] = None,
        parent_doc_id: Optional[str] = None,
    ) -> None:
        """
        Delete points matching filter criteria.
        Useful for removing all chunks from a specific filing before reindexing.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        conditions = []
        if company:
            conditions.append(FieldCondition(key="company", match=MatchValue(value=company)))
        if document_type:
            conditions.append(FieldCondition(key="document_type", match=MatchValue(value=document_type)))
        if parent_doc_id:
            conditions.append(FieldCondition(key="parent_doc_id", match=MatchValue(value=parent_doc_id)))

        if not conditions:
            raise ValueError("At least one filter parameter is required")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=Filter(must=conditions)),
        )
        logger.info(
            f"Deleted points matching filter "
            f"(company={company}, doc_type={document_type}, doc_id={parent_doc_id})"
        )

    def count(self) -> int:
        """Get total number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count


def get_vector_store(**kwargs) -> QdrantVectorStore:
    """
    Factory function to create a vector store instance.

    Args:
        **kwargs: Override any QdrantVectorStore constructor parameter

    Returns:
        QdrantVectorStore instance
    """
    provider = vs_config.get("provider", "qdrant")
    logger.info(f"Creating vector store: provider={provider}")

    if provider == "qdrant":
        return QdrantVectorStore(**kwargs)
    else:
        raise ValueError(
            f"Unknown vector store provider: {provider}. "
            f"Supported: qdrant"
        )


if __name__ == "__main__":
    from src.indexing.embedder import get_embedder

    store = get_vector_store()
    embedder = get_embedder()

    test_chunks = [
        Chunk(
            chunk_id=1,
            content="Apple reported total revenue of $394.3 billion for fiscal year 2024.",
            company="AAPL", document_type="10-K", filing_date="2024-11-01",
            page_number=1, chunk_index=0, parent_doc_id="doc-001",
            has_table=False, has_chart=False, token_count=15,
        ),
        Chunk(
            chunk_id=2,
            content="Microsoft's total revenue was $245.1 billion in fiscal year 2024.",
            company="MSFT", document_type="10-K", filing_date="2024-07-30",
            page_number=1, chunk_index=0, parent_doc_id="doc-002",
            has_table=False, has_chart=False, token_count=14,
        ),
    ]

    embeddings = embedder.embed_batch([c.content for c in test_chunks])
    store.upsert(test_chunks, embeddings)
    print(f"Collection has {store.count()} points")

    query = "What was Microsoft's revenue in 2024?"
    query_emb = embedder.embed(query)
    results = store.search(query_emb, company=["AAPL","MSFT"], limit=5)
    for r in results:
        print(f"  [{r.score:.4f}] {r.content[:80]}...")
