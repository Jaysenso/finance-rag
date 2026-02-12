from src.utils.logger import get_logger
from src.preprocessing.document_parser import DocumentParser
from src.preprocessing.chunking import SemanticChunker
from src.indexing.embedder import get_embedder
logger = get_logger(__name__)

def main():
    """Main entry point for testing modules."""

    logger.info("Application started")
    file_path = "./src/data/pdf/sample/sample-unstructured-paper.pdf"

    # # Test chunking
    chunker = SemanticChunker()

    doc_chunks = chunker.chunk_document(
        file_path=file_path, company="APPL",
        document_type="10-K", filing_date="2025-04-03")

    # --- Validate chunks ---
    print(f"\n=== Chunks ===")
    print(f"Total chunks: {len(doc_chunks)}")
    table_chunks = [c for c in doc_chunks if c.has_table]
    image_chunks = [c for c in doc_chunks if c.has_chart]
    text_chunks = [c for c in doc_chunks if not c.has_table and not c.has_chart]
    print(f"Text: {len(text_chunks)} | Tables: {len(table_chunks)} | Images: {len(image_chunks)}")
    for i, chunk in enumerate(doc_chunks[:5]):
        label = "TABLE" if chunk.has_table else "IMAGE" if chunk.has_chart else "TEXT"
        print(f"  [{i}] {label} | page {chunk.page_number} | ~{chunk.token_count} tokens | {chunk.content[:80]}...")

    # --- Validate embeddings ---
    embedder = get_embedder()
    embeddings = embedder.embed_batch([chunk.content for chunk in doc_chunks])
    print(f"\n=== Embeddings ===")
    print(f"Vectors: {len(embeddings)} | Dimension: {len(embeddings[0])}")

    # --- Validate similarity ---
    if len(embeddings) >= 3:
        sim_01 = embedder.similarity(embeddings[0], embeddings[1])
        sim_02 = embedder.similarity(embeddings[0], embeddings[2])
        print(f"\n=== Similarity ===")
        print(f"Chunk 0 vs 1: {sim_01:.4f}")
        print(f"Chunk 0 vs 2: {sim_02:.4f}")

    logger.info("Task Completed")


if __name__ == "__main__":
    main()
