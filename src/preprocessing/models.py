"""
Data models for preprocessing module.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
# ── Document Parser ────────────────────────────────────────

@dataclass
class DocumentElement:
    """Represents a single element from a document."""
    element_id: str
    type: str  
    text: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_description: Optional[str] = None

@dataclass
class ParsedDocument:
    """Represents a parsed document."""
    file_path: Path
    elements: List[DocumentElement]
    metadata: Dict[str, Any]

# ── Chunker  ──────────────────────────────────────────────
@dataclass
class Chunk:
    """Represents a text chunk for indexing."""
    chunk_id: str
    content: str
    company: str
    document_type: str
    filing_date: str
    page_number: Optional[int]
    chunk_index: int
    parent_doc_id: str
    has_table: bool
    has_chart: bool
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "company": self.company,
            "document_type": self.document_type,
            "filing_date": self.filing_date,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "parent_doc_id": self.parent_doc_id,
            "has_table": self.has_table,
            "has_chart": self.has_chart,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

# ── HyPE Dataclass  ──────────────────────────────────────────────
@dataclass
class GeneratedQuestions:
    """Container for generated questions with metadata."""
    chunk_id: str
    questions: List[str]
    company: str
    document_type: str
    filing_date: str
    page_number: int
