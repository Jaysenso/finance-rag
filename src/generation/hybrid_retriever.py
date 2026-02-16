"""
Hybrid retriever for HyPE dual-database system.

Searches both content and question databases in parallel,
then fuses results using Reciprocal Rank Fusion (RRF).
"""
from typing import List, Optional, Tuple
from src.utils.logger import get_logger
from src.indexing.embedder import BaseEmbedder, get_embedder
from src.indexing.vector_store import QdrantVectorStore, SearchResult, get_vector_store
from src.indexing.question_store import QuestionVectorStore, get_question_store
from config import load_config

config = load_config()
hype_config = config.get("hype", {})
logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that searches both content and question databases.
    
    Workflow:
    1. Embed user query
    2. Search content DB (chunk embeddings)
    3. Search question DB (hypothetical question embeddings)
    4. Map question results to chunk IDs
    5. Fuse using Reciprocal Rank Fusion (RRF)
    6. Return top K unique chunks
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder = None,
        content_store: QdrantVectorStore = None,
        question_store: QuestionVectorStore = None,
        content_weight: float = None,
        question_weight: float = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedder: Embedder for query encoding
            content_store: Content vector store
            question_store: Question vector store
            content_weight: Weight for content DB results (0.0-1.0)
            question_weight: Weight for question DB results (0.0-1.0)
        """
        self.embedder = embedder or get_embedder()
        self.content_store = content_store or get_vector_store()
        self.question_store = question_store or get_question_store()
        
        self.content_weight = content_weight or hype_config.get("content_weight", 0.5)
        self.question_weight = question_weight or hype_config.get("question_weight", 0.5)
        
        logger.info(
            f"HybridRetriever initialized "
            f"(content_weight={self.content_weight}, "
            f"question_weight={self.question_weight})"
        )
    
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        company: Optional[str] = None,
        document_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Hybrid retrieval across content and question databases.
        
        Args:
            query: User query
            limit: Number of final results to return
            company: Optional company filter
            document_type: Optional document type filter
            score_threshold: Optional minimum score threshold
            
        Returns:
            List of SearchResult objects (fused and deduplicated)
        """
        # Step 1: Embed query
        query_embedding = self.embedder.embed(query)
        
        # Step 2: Search content DB
        # Fetch 2x limit to ensure we don't miss important contexts after deduplication
        content_limit = limit * 2
        content_results = self.content_store.search(
            query_embedding=query_embedding,
            limit=content_limit,
            company=company,
            document_type=document_type,
            score_threshold=score_threshold,
        )
        
        # Step 3: Search question DB
        question_limit = limit * 2
        question_results = self.question_store.search(
            query_embedding=query_embedding,
            limit=question_limit,
            company=company,
            document_type=document_type,
            score_threshold=score_threshold,
        )
        
        logger.info(
            f"Retrieved {len(content_results)} from content DB, "
            f"{len(question_results)} from question DB"
        )
        
        # Step 4: Map question results to chunk IDs and retrieve chunks
        question_chunk_ids = [q.chunk_id for q in question_results]
        question_chunks = self.content_store.retrieve_by_ids(question_chunk_ids)
        
        # Preserve question search scores by mapping chunk_id -> score
        question_score_map = {q.chunk_id: q.score for q in question_results}
        for chunk in question_chunks:
            chunk.score = question_score_map.get(chunk.chunk_id, chunk.score)
        
        # Step 5: Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(
            content_results=content_results,
            question_results=question_chunks,
            k=60,  # RRF parameter
        )
        
        # Step 6: Return top K
        final_results = fused_results[:limit]
        
        logger.info(
            f"Hybrid retrieval returned {len(final_results)} fused results "
            f"(limit={limit})"
        )
        
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        content_results: List[SearchResult],
        question_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Fuse results from content and question databases using RRF.
        
        RRF formula: score(chunk) = sum(1 / (k + rank_i))
        where rank_i is the rank of the chunk in result list i
        
        Args:
            content_results: Results from content DB
            question_results: Results from question DB (already mapped to chunks)
            k: RRF constant (typically 60)
            
        Returns:
            Fused and sorted list of SearchResult objects
        """
        # Build chunk_id -> SearchResult mapping for deduplication
        # Note: Scores don't matter here since RRF will overwrite them
        chunk_map = {}
        
        # Add content results
        for result in content_results:
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result
        
        # Add question results
        for result in question_results:
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Score from content DB
        for rank, result in enumerate(content_results):
            chunk_id = result.chunk_id
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += self.content_weight / (k + rank + 1)
        
        # Score from question DB
        for rank, result in enumerate(question_results):
            chunk_id = result.chunk_id
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += self.question_weight / (k + rank + 1)
        
        # Sort by RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )
        
        # Build final result list with RRF scores
        fused_results = []
        for chunk_id in sorted_chunk_ids:
            result = chunk_map[chunk_id]
            # Update score to RRF score
            result.score = rrf_scores[chunk_id]
            fused_results.append(result)
        
        logger.info(
            f"RRF fusion: {len(content_results)} content + "
            f"{len(question_results)} question → "
            f"{len(fused_results)} unique chunks"
        )
        
        return fused_results


def get_hybrid_retriever(**kwargs) -> HybridRetriever:
    """
    Factory function to create a hybrid retriever.
    
    Args:
        **kwargs: Override any HybridRetriever constructor parameter
        
    Returns:
        HybridRetriever instance
    """
    logger.info("Creating hybrid retriever")
    return HybridRetriever(**kwargs)
