"""
Document Ingestion Pipeline

This script handles the complete ingestion lifecycle:
1. Parse PDF documents (SEC filings)
2. Chunk documents into semantic units
3. Embed chunks
4. Index chunks to content vector database
5. Generate hypothetical questions (HyPE)
6. Embed and index questions to question database

Supports two metadata modes:
- Manual: Specify --company, --doc-type, and --date for all files
- Auto: Extract metadata from filenames (format: COMPANY_DOCTYPE_ACCESSION.pdf)

Usage:
    # Manual metadata (single file or batch with same metadata)
    python ingest.py --file path/to/document.pdf --company AAPL --doc-type 10-K --date 2023-12-31
    
    # Auto metadata extraction (batch with different companies/types)
    python ingest.py --directory path/to/pdfs/
    
    # Skip HyPE for faster ingestion
    python ingest.py --directory path/to/pdfs/ --skip-hype
    
Filename Format for Auto-Extraction:
    COMPANY_DOCTYPE_ACCESSION.pdf
    Examples:
        AAPL_10-K_0000320193-25-000079.pdf
        MSFT_10-Q_0001564590-24-000123.xlsx
        GOOGL_10-K_0001652044-25-000012.jpg
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict
from src.utils.logger import get_logger
from src.utils.utils import extract_metadata_from_filename
from src.preprocessing.document_parser import get_document_parser
from src.preprocessing.models import Chunk
from src.preprocessing.chunking import get_chunker
from src.indexing.embedder import get_embedder
from src.indexing.vector_store import get_vector_store
from src.indexing.hype_ingestion import index_questions_for_chunks
from config import load_config

logger = get_logger(__name__)
config = load_config()


def parse_documents(
    file_paths: List[Path],
    company: Optional[str] = None,
    document_type: Optional[str] = None,
    filing_date: Optional[str] = None,
) -> List[Chunk]:
    """
    Parse documents and chunk them.
    
    Supports PDF, Excel (.xlsx, .xls), and Images (.jpg, .jpeg, .png).
    
    Supports two modes:
    1. Manual metadata: All files use the same company/doc_type/date
    2. Auto metadata: Extract metadata from each filename
    
    Args:
        file_paths: List of file paths (PDF, Excel, or Image)
        company: Company ticker (optional if using filename extraction)
        document_type: Filing type (optional if using filename extraction)
        filing_date: Filing date (optional if using filename extraction)
        
    Returns:
        List of Chunk objects
    """
    logger.info(f"Parsing {len(file_paths)} document(s)...")
    
    # Determine if we're using automatic metadata extraction
    auto_extract = company is None or document_type is None or filing_date is None
    
    if auto_extract:
        logger.info("Using automatic metadata extraction from filenames")
        logger.info("Expected format: COMPANY_DOCTYPE_ACCESSION.pdf (e.g., AAPL_10-K_0000320193-25-000079.pdf)")
    
    # Prepare metadata for each file
    file_metadata = []
    valid_file_paths = []
    
    for file_path in file_paths:
        logger.info(f"Processing: {file_path.name}")
        
        # Extract or use provided metadata
        if auto_extract:
            try:
                metadata = extract_metadata_from_filename(str(file_path))
                doc_company = metadata["company"]
                doc_type = metadata["document_type"]
                doc_date = metadata["filing_date"]
                logger.info(f"  Extracted: {doc_company} | {doc_type} | {doc_date}")
            except ValueError as e:
                logger.error(f"  ✗ {e}")
                logger.warning(f"  Skipping {file_path.name}")
                continue
        else:
            doc_company = company
            doc_type = document_type
            doc_date = filing_date
        
        valid_file_paths.append(file_path)
        file_metadata.append({
            "company": doc_company,
            "document_type": doc_type,
            "filing_date": doc_date,
        })
    
    if not valid_file_paths:
        logger.error("No valid files to process")
        return []
    
    # Batch parse all documents
    parser = get_document_parser()
    logger.info(f"Batch parsing {len(valid_file_paths)} documents...")
    parsed_docs = parser.parse_documents_batch([str(fp) for fp in valid_file_paths])
    print(parsed_docs)
    # Chunk each parsed document
    chunker = get_chunker()
    all_chunks = []
    
    for parsed_doc, metadata in zip(parsed_docs, file_metadata):
        chunks = chunker.chunk_document(
            parsed_doc=parsed_doc,
            company=metadata["company"],
            document_type=metadata["document_type"],
            filing_date=metadata["filing_date"],
        )
        all_chunks.extend(chunks)
        
        logger.info(
            f"  ✓ {parsed_doc.file_path.name}: {parsed_doc.metadata['num_elements']} elements → {len(chunks)} chunks"
        )
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def index_content_chunks(chunks: List[Chunk]) -> None:
    """
    Embed and index content chunks to the vector database.
    
    Args:
        chunks: List of Chunk objects to index
    """
    logger.info(f"Indexing {len(chunks)} chunks to content database...")
    
    embedder = get_embedder()
    vector_store = get_vector_store()
    
    # Extract text for embedding
    chunk_texts = [chunk.content for chunk in chunks]
    
    # Embed chunks
    logger.info("Generating embeddings...")
    embeddings = embedder.embed_batch(chunk_texts, show_progress=True)
    
    # Index to vector store
    logger.info("Upserting to vector database...")
    vector_store.upsert(chunks, embeddings)
    
    logger.info(f"✓ Content indexing complete! Total points: {vector_store.count()}")


def index_hype_questions(chunks: List[Chunk]) -> None:
    """
    Generate and index hypothetical questions for HyPE retrieval.
    
    Args:
        chunks: List of Chunk objects to generate questions for
    """
    logger.info("Starting HyPE question generation and indexing...")
    
    index_questions_for_chunks(chunks)
    
    logger.info("✓ HyPE indexing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the finance RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single document with manual metadata
  python ingest.py --file data/AAPL_10K_2023.pdf --company AAPL --doc-type 10-K --date 2023-12-31
  
  # Batch ingest with automatic metadata extraction from filenames
  python ingest.py --directory data/filings/
  
  # Batch ingest with manual metadata (all files use same metadata)
  python ingest.py --directory data/filings/ --company AAPL --doc-type 10-K --date 2023-12-31
  
  # Skip HyPE question generation (faster, for testing)
  python ingest.py --directory data/filings/ --skip-hype

Filename Format for Auto-Extraction:
  COMPANY_DOCTYPE_ACCESSION.pdf
  Example: AAPL_10-K_0000320193-25-000079.pdf
           MSFT_10-Q_0001564590-24-000123.xlsx
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to a single PDF file to ingest"
    )
    input_group.add_argument(
        "--directory",
        type=str,
        help="Path to directory containing files (PDF, Excel, Images)"
    )
    
    # Metadata (optional for batch ingestion with proper filenames)
    parser.add_argument(
        "--company",
        type=str,
        required=False,
        help="Company ticker symbol (e.g., AAPL, MSFT). Optional if using filename extraction."
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        required=False,
        help="Document type (e.g., 10-K, 10-Q). Optional if using filename extraction."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=False,
        help="Filing date in YYYY-MM-DD format. Optional if using filename extraction."
    )
    
    # Options
    parser.add_argument(
        "--skip-hype",
        action="store_true",
        help="Skip HyPE question generation (faster ingestion)"
    )
    parser.add_argument(
        "--content-only",
        action="store_true",
        help="Only index content chunks, skip HyPE entirely"
    )
    
    args = parser.parse_args()
    
    # Validate and collect file paths
    file_paths = []
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
            
        accepted_extensions = {".pdf", ".xlsx", ".xls", ".jpg", ".jpeg", ".png"}
        if file_path.suffix.lower() not in accepted_extensions:
            logger.error(f"File type not supported: {args.file}")
            logger.error(f"Supported types: {', '.join(accepted_extensions)}")
            sys.exit(1)
        file_paths.append(file_path)
    
    elif args.directory:
        dir_path = Path(args.directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory not found: {args.directory}")
            sys.exit(1)
            
        # Scan for all supported extensions
        supported_patterns = ["*.pdf", "*.xlsx", "*.xls", "*.jpg", "*.jpeg", "*.png"]
        file_paths = []
        for pattern in supported_patterns:
            file_paths.extend(list(dir_path.glob(pattern)))
            
        if not file_paths:
            logger.error(f"No supported files found in: {args.directory}")
            sys.exit(1)
    
    # Display ingestion plan
    auto_extract = args.company is None or args.doc_type is None or args.date is None
    
    logger.info("=" * 70)
    logger.info("INGESTION PIPELINE")
    logger.info("=" * 70)
    
    if auto_extract:
        logger.info("Mode:          Auto-extract metadata from filenames")
    else:
        logger.info("Mode:          Manual metadata")
        logger.info(f"Company:       {args.company}")
        logger.info(f"Document Type: {args.doc_type}")
        logger.info(f"Filing Date:   {args.date}")
    
    logger.info(f"Files:         {len(file_paths)}")
    for fp in file_paths:
        logger.info(f"  - {fp.name}")
    logger.info(f"HyPE Enabled:  {not args.skip_hype and not args.content_only}")
    logger.info("=" * 70)
    
    try:
        # Step 1: Parse and chunk documents
        chunks = parse_documents(
            file_paths=file_paths,
            company=args.company,
            document_type=args.doc_type,
            filing_date=args.date,
        )
        
        if not chunks:
            logger.error("No chunks created, aborting ingestion")
            sys.exit(1)
        
        # Step 2: Index content chunks
        index_content_chunks(chunks)
        
        # Step 3: Generate and index HyPE questions (optional)
        if not args.skip_hype and not args.content_only:
            index_hype_questions(chunks)
        else:
            logger.info("Skipping HyPE question generation (--skip-hype or --content-only)")
        
        # Success summary
        logger.info("=" * 70)
        logger.info("✓ INGESTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total chunks indexed: {len(chunks)}")
        if not args.skip_hype and not args.content_only:
            questions_per_chunk = config.get("hype", {}).get("questions_per_chunk", 3)
            logger.info(f"Total questions indexed: {len(chunks) * questions_per_chunk}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
