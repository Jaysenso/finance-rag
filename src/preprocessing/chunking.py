from config import load_config
from src.utils.logger import get_logger
from src.preprocessing.document_parser import DocumentParser, DocumentElement

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

config = load_config()
chunking_config = config["chunking"]
logger = get_logger(__name__)


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


class SemanticChunker:
    """Semantic chunker for financial documents."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
    ):
        self.chunk_size = chunk_size or chunking_config["chunk_size"]
        self.chunk_overlap = chunk_overlap or chunking_config["chunk_overlap"]
        self.min_chunk_size = min_chunk_size or chunking_config["min_chunk_size"]
        self.max_chunk_size = max_chunk_size or chunking_config["max_chunk_size"]

        self.parser = DocumentParser()

        logger.info(
            f"Semantic chunker initialized: "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    def chunk_document(
        self,
        file_path: str | Path,
        company: str,
        document_type: str,
        filing_date: str,
    ) -> List[Chunk]:
        """
        Chunk a parsed document. Tables and images become individual chunks.
        Text elements are merged with semantic overlap.
        """
        logger.info(f"Chunking document: {Path(file_path).name}")

        parsed_doc = self.parser.parse_document(file_path)
        parent_doc_id = str(uuid.uuid4())

        chunks: List[Chunk] = []
        text_buffer: List[DocumentElement] = []
        chunk_index = 0

        for element in parsed_doc.elements:
            if element.type in ("Table", "Image"):
                # Flush any accumulated text first
                if text_buffer:
                    text_chunks = self._chunk_text_elements(
                        text_buffer, company, document_type,
                        filing_date, parent_doc_id, chunk_index,
                    )
                    chunks.extend(text_chunks)
                    chunk_index += len(text_chunks)
                    text_buffer = []

                # Table/Image gets its own chunk
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    content=element.text,
                    company=company,
                    document_type=document_type,
                    filing_date=filing_date,
                    page_number=element.page_number,
                    chunk_index=chunk_index,
                    parent_doc_id=parent_doc_id,
                    has_table=element.type == "Table",
                    has_chart=element.type == "Image",
                    token_count=self._estimate_tokens(element.text),
                    metadata=element.metadata,
                ))
                chunk_index += 1
            else:
                text_buffer.append(element)

        # Flush remaining text
        if text_buffer:
            text_chunks = self._chunk_text_elements(
                text_buffer, company, document_type,
                filing_date, parent_doc_id, chunk_index,
            )
            chunks.extend(text_chunks)

        logger.info(f"Created {len(chunks)} chunks from {Path(file_path).name}")
        return chunks

    def _chunk_text_elements(
        self,
        elements: List[DocumentElement],
        company: str,
        document_type: str,
        filing_date: str,
        parent_doc_id: str,
        start_index: int,
    ) -> List[Chunk]:
        """
        Merge consecutive text elements into semantic chunks with overlap.
        Splits on paragraph boundaries (element boundaries) w.r.t min/max chunk sizes.
        """
        # Build list of (text, page_number) paragraphs
        paragraphs: List[tuple[str, Optional[int]]] = []

        for el in elements:
            if el.text.strip():
                paragraphs.append((el.text.strip(), el.page_number))

        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        chunk_index = start_index
        i = 0

        while i < len(paragraphs):
            # Accumulate paragraphs up to chunk_size
            current_texts: List[str] = []
            current_tokens = 0
            page_number = paragraphs[i][1]
            start_i = i

            while i < len(paragraphs):
                para_text, para_page = paragraphs[i]
                para_tokens = self._estimate_tokens(para_text)

                # If adding this paragraph would exceed max, stop
                # (unless the chunk is still empty — always take at least one)
                if current_texts and current_tokens + para_tokens > self.chunk_size:
                    break

                current_texts.append(para_text)
                current_tokens += para_tokens
                if page_number is None:
                    page_number = para_page
                i += 1

            # Too small and more paragraphs remain — merge into next iteration
            if current_tokens < self.min_chunk_size and i < len(paragraphs):
                continue

            # Too small and it's the last batch — append to previous chunk
            if current_tokens < self.min_chunk_size and chunks:
                prev = chunks[-1]
                prev.content += "\n\n" + "\n\n".join(current_texts)
                prev.token_count += current_tokens
                continue

            content = "\n\n".join(current_texts)

            # Truncate if over max chunk size (Fallback Mechanism - Docling docs are rarely > 2048 tokens)
            if current_tokens > self.max_chunk_size:
                max_chars = self.max_chunk_size * 4
                content = content[:max_chars]
                current_tokens = self.max_chunk_size

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                company=company,
                document_type=document_type,
                filing_date=filing_date,
                page_number=page_number,
                chunk_index=chunk_index,
                parent_doc_id=parent_doc_id,
                has_table=False,
                has_chart=False,
                token_count=current_tokens,
            ))
            chunk_index += 1

            # Overlap: rewind by overlap amount so next chunk starts earlier
            if i < len(paragraphs):
                overlap_tokens = 0
                rewind = i
                # Guard prevents rewinding into current chunk's start and skip entirely if its the last chunk
                while rewind > start_i and overlap_tokens < self.chunk_overlap:
                    rewind -= 1
                    overlap_tokens += self._estimate_tokens(paragraphs[rewind][0])
                i = rewind if rewind > start_i else i

        return chunks

if __name__ == "__main__":

    # Test chunking
    chunker = SemanticChunker()
    file_path = "./src/data/pdf/sample/sample-unstructured-paper.pdf"
    
    doc_chunks = chunker.chunk_document(
        file_path=file_path, company="APPL",
        document_type="10-K", filing_date="2025-04-03")
    
    print(doc_chunks[:10])