
"""
Data models for Generation module.
"""
from typing import Optional, List
from dataclasses import dataclass, field
from src.indexing.vector_store import SearchResult   


# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class SubQuery:
    """A decomposed sub-query with retrieval parameters and results."""
    query: str
    company: Optional[str] = None
    document_type: Optional[str] = None
    time_hint: Optional[str] = None
    results: List[SearchResult] = field(default_factory=list)
    verification_score: float = 0.0
    verified: bool = False
    verification_reason: str = ""
    verification_missing: str = ""
    retry_count: int = 0
    original_query: str = ""
    reformulation_history: List[str] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Structured output from query analysis step."""
    intent: str
    companies: List[str]
    document_types: List[str]
    time_periods: List[str]
    needs_table: bool
    sub_queries: List[SubQuery]
    raw_query: str


@dataclass
class Source:
    """A cited source in the final answer."""
    source_number: int
    company: str
    document_type: str
    filing_date: str
    page_number: Optional[int]
    chunk_id: str
    content: str
    score: float


@dataclass
class RAGResponse:
    """Final response from the RAG agent."""
    answer: str
    sources: List[Source]
    sub_queries: List[SubQuery]
    query_analysis: QueryAnalysis
    total_chunks_retrieved: int
    total_retries: int

