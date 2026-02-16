"""
Question vector store for HyPE retrieval.

Manages a separate Qdrant collection for hypothetical question embeddings
that reference back to content chunks.
"""
from src.utils.logger import get_logger
from config import load_config
from src.preprocessing.models import GeneratedQuestions

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

config = load_config()
vs_config = config["vector_store"]
hype_config = config.get("hype", {})
embedding_config = config["embedding"]
logger = get_logger(__name__)


@dataclass
class QuestionSearchResult:
    """A single search result from the question store."""
    question_id: str
    question_text: str
    chunk_id: str  # Reference to content chunk
    score: float
    metadata: Dict[str, Any]


class QuestionVectorStore:
    """
    Qdrant-backed vector store for hypothetical question embeddings.
    
    Each point represents one hypothetical question that references
    a chunk in the content database.
    """
    
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
        
        self.collection_name = collection_name or hype_config.get(
            "question_collection_name", "sec_filings_questions"
        )
        self.distance = distance or vs_config["distance"]
        self.dimension = dimension or embedding_config["dimension"]
        self.batch_size = vs_config["batch_size"]
        
        mode = mode or vs_config["mode"]
        
        if mode == "cloud":
            url = url or vs_config["url"]
            api_key = api_key or os.getenv("QDRANT_API_KEY") or vs_config["api_key"]
            if not url:
                raise ValueError("Qdrant Cloud URL is required in cloud mode")
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"Question store cloud client initialized: {url}")
        else:
            path = path or vs_config["path"]
            self.client = QdrantClient(path=path)
            logger.info(f"Question store local client initialized: {path}")
        
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist, with payload indexes."""
        from qdrant_client.models import Distance, VectorParams
        
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Question collection '{self.collection_name}' exists "
                f"({info.points_count} questions)"
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
            f"Created question collection '{self.collection_name}' "
            f"(dim={self.dimension}, distance={self.distance})"
        )
        
        # Create payload indexes for filterable fields
        index_fields = {
            "chunk_id": "keyword",  # Reference to content chunk
            "company": "keyword",
            "document_type": "keyword",
            "filing_date": "keyword",
            "question_number": "integer",
        }
        for field_name, field_type in index_fields.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
        logger.info(f"Created question payload indexes: {list(index_fields.keys())}")
    
    def upsert(
        self,
        generated_questions_list: List[GeneratedQuestions],
        question_embeddings: List[List[List[float]]],  # [chunk][question][embedding_dim]
    ) -> None:
        """
        Upsert generated questions with their embeddings.
        
        Args:
            generated_questions_list: List of GeneratedQuestions objects
            question_embeddings: Nested list of embeddings
                - Outer list: one per chunk
                - Middle list: one per question (typically 3)
                - Inner list: embedding vector
        """
        from qdrant_client.models import PointStruct
        
        if len(generated_questions_list) != len(question_embeddings):
            raise ValueError(
                f"Mismatch: {len(generated_questions_list)} question sets but "
                f"{len(question_embeddings)} embedding sets"
            )
        
        points = []
        for gen_q, embeddings in zip(generated_questions_list, question_embeddings):
            if len(gen_q.questions) != len(embeddings):
                logger.warning(
                    f"Question/embedding count mismatch for chunk {gen_q.chunk_id}: "
                    f"{len(gen_q.questions)} questions, {len(embeddings)} embeddings"
                )
                continue
            
            for q_num, (question_text, embedding) in enumerate(zip(gen_q.questions, embeddings), 1):
                # Use deterministic UUID based on chunk_id and question number
                # This ensures we don't create duplicates if we re-run ingestion
                import uuid
                namespace = uuid.UUID(gen_q.chunk_id) if len(gen_q.chunk_id) == 36 else uuid.NAMESPACE_DNS
                question_id = str(uuid.uuid5(namespace, f"{gen_q.chunk_id}_{q_num}"))
                
                points.append(PointStruct(
                    id=question_id,
                    vector=embedding,
                    payload={
                        "question_text": question_text,
                        "chunk_id": gen_q.chunk_id,
                        "company": gen_q.company,
                        "document_type": gen_q.document_type,
                        "filing_date": gen_q.filing_date,
                        "page_number": gen_q.page_number,
                        "question_number": q_num,
                    },
                ))
        
        # Batch upsert
        for i in range(0, len(points), self.batch_size):
            batch = points[i : i + self.batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        
        logger.info(
            f"Upserted {len(points)} question points to '{self.collection_name}'"
        )
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        company: Optional[str | List[str]] = None,
        document_type: Optional[str | List[str]] = None,
        filing_date: Optional[str | List[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[QuestionSearchResult]:
        """
        Search for similar questions with optional metadata filters.
        
        Args:
            query_embedding: Query vector
            limit: Max results to return
            company: Filter by ticker(s)
            document_type: Filter by filing type(s)
            filing_date: Filter by date(s)
            score_threshold: Minimum similarity score
            
        Returns:
            List of QuestionSearchResult ordered by similarity
        """
        query_filter = self._build_filter(
            company=company,
            document_type=document_type,
            filing_date=filing_date,
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
            search_results.append(QuestionSearchResult(
                question_id=point.id,
                question_text=payload["question_text"],
                chunk_id=payload["chunk_id"],
                score=point.score,
                metadata={k: v for k, v in payload.items() 
                         if k not in ["question_text", "chunk_id"]},
            ))
        
        logger.info(
            f"Question search returned {len(search_results)} results "
            f"(limit={limit}, filtered={query_filter is not None})"
        )
        return search_results
    
    def _build_filter(
        self,
        company: Optional[str | List[str]] = None,
        document_type: Optional[str | List[str]] = None,
        filing_date: Optional[str | List[str]] = None,
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
        
        if not conditions:
            return None
        
        return Filter(must=conditions)
    
    def count(self) -> int:
        """Get total number of questions in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
    
    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> None:
        """Delete all questions associated with given chunk IDs."""
        from qdrant_client.models import Filter, FieldCondition, MatchAny, FilterSelector
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))]
                )
            ),
        )
        logger.info(f"Deleted questions for {len(chunk_ids)} chunks")


def get_question_store(**kwargs) -> QuestionVectorStore:
    """
    Factory function to create a question vector store instance.
    
    Args:
        **kwargs: Override any QuestionVectorStore constructor parameter
        
    Returns:
        QuestionVectorStore instance
    """
    logger.info("Creating question vector store")
    return QuestionVectorStore(**kwargs)
