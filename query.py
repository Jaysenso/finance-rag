"""
Query Interface for Finance RAG System

This script provides a production-ready query interface with multiple modes:
1. Interactive mode - Chat-like interface for multiple queries
2. Single query mode - One-off query execution
3. Batch mode - Process multiple queries from a file

Usage:
    # Interactive mode (default)
    python query.py
    
    # Single query mode
    python query.py --query "What was Apple's revenue in 2023?"
    
    # Batch mode from file
    python query.py --batch queries.txt
    
    # Filter by company/document type
    python query.py --query "What was the revenue?" --company AAPL --doc-type 10-K
    
    # Show detailed retrieval information
    python query.py --query "Revenue growth?" --verbose
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List
from src.utils.logger import get_logger
from src.generation.rag_agent import RAGAgent
from config import load_config

logger = get_logger(__name__)
config = load_config()


def format_response(response, verbose: bool = False) -> str:
    """
    Format RAG response for display.
    
    Args:
        response: RAGResponse object
        verbose: Show detailed retrieval information
        
    Returns:
        Formatted string
    """
    output = []
    
    # Header
    output.append("\n" + "=" * 70)
    output.append("ANSWER")
    output.append("=" * 70)
    output.append(f"\n{response.answer}\n")
    
    # Sources
    if response.sources:
        output.append(f"--- Sources ({len(response.sources)}) ---")
        for s in response.sources:
            output.append(
                f"  [{s.source_number}] {s.company} {s.document_type} "
                f"(p.{s.page_number}, score: {s.score:.3f})"
            )
    else:
        output.append("--- No sources retrieved ---")
    
    # Verbose information
    if verbose:
        output.append(f"\n--- Sub-queries ({len(response.sub_queries)}) ---")
        for sq in response.sub_queries:
            output.append(
                f"  • {sq.query}\n"
                f"    verified={sq.verified} | "
                f"score={sq.verification_score:.2f} | "
                f"retries={sq.retry_count}"
            )
        
        output.append(
            f"\n--- Retrieval Stats ---\n"
            f"  Total chunks retrieved: {response.total_chunks_retrieved}\n"
            f"  Total retries: {response.total_retries}"
        )
    
    output.append("=" * 70)
    
    return "\n".join(output)


def query_single(
    query: str,
    agent: RAGAgent,
    verbose: bool = False,
    company: Optional[str] = None,
    document_type: Optional[str] = None,
    filing_date: Optional[str] = None,
) -> None:
    """
    Execute a single query and display results.
    
    Args:
        query: User query string
        agent: Initialized RAGAgent
        verbose: Show detailed information
        company: Optional company filter
        document_type: Optional document type filter
        filing_date: Optional filing date filter
    """
    logger.info(f"Query: {query}")
    
    # Build filter kwargs
    filter_kwargs = {}
    if company:
        filter_kwargs["company"] = company
    if document_type:
        filter_kwargs["document_type"] = document_type
    if filing_date:
        filter_kwargs["filing_date"] = filing_date
    
    try:
        response = agent.query(query, **filter_kwargs)
        print(format_response(response, verbose=verbose))
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}\n")


def query_interactive(
    agent: RAGAgent,
    verbose: bool = False,
    company: Optional[str] = None,
    document_type: Optional[str] = None,
    filing_date: Optional[str] = None,
) -> None:
    """
    Interactive query mode - chat-like interface.
    
    Args:
        agent: Initialized RAGAgent
        verbose: Show detailed information
        company: Optional company filter
        document_type: Optional document type filter
        filing_date: Optional filing date filter
    """
    print("\n" + "=" * 70)
    print("FINANCE RAG - INTERACTIVE MODE")
    print("=" * 70)
    print("Enter your questions about SEC filings.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'verbose' to toggle detailed output")
    print("  - Type 'clear' to clear filters")
    
    if company or document_type or filing_date:
        print("\nActive Filters:")
        if company:
            print(f"  Company: {company}")
        if document_type:
            print(f"  Document Type: {document_type}")
        if filing_date:
            print(f"  Filing Date: {filing_date}")
    
    print("=" * 70 + "\n")
    
    current_verbose = verbose
    current_company = company
    current_doc_type = document_type
    current_date = filing_date
    
    while True:
        try:
            query = input("Query> ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!\n")
                break
            
            elif query.lower() == "verbose":
                current_verbose = not current_verbose
                print(f"Verbose mode: {'ON' if current_verbose else 'OFF'}\n")
                continue
            
            elif query.lower() == "clear":
                current_company = None
                current_doc_type = None
                current_date = None
                print("Filters cleared.\n")
                continue
            
            elif query.lower() == "help":
                print("\nCommands:")
                print("  quit/exit - Exit the program")
                print("  verbose   - Toggle detailed output")
                print("  clear     - Clear all filters")
                print("  help      - Show this help\n")
                continue
            
            # Execute query
            query_single(
                query=query,
                agent=agent,
                verbose=current_verbose,
                company=current_company,
                document_type=current_doc_type,
                filing_date=current_date,
            )
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except EOFError:
            print("\n\nGoodbye!\n")
            break


def query_batch(
    batch_file: Path,
    agent: RAGAgent,
    verbose: bool = False,
    company: Optional[str] = None,
    document_type: Optional[str] = None,
    filing_date: Optional[str] = None,
) -> None:
    """
    Batch query mode - process queries from a file.
    
    Args:
        batch_file: Path to file containing queries (one per line)
        agent: Initialized RAGAgent
        verbose: Show detailed information
        company: Optional company filter
        document_type: Optional document type filter
        filing_date: Optional filing date filter
    """
    if not batch_file.exists():
        logger.error(f"Batch file not found: {batch_file}")
        sys.exit(1)
    
    queries = []
    with open(batch_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                queries.append(line)
    
    if not queries:
        logger.error(f"No queries found in {batch_file}")
        sys.exit(1)
    
    logger.info(f"Processing {len(queries)} queries from {batch_file.name}")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(queries)}")
        print(f"{'='*70}")
        
        query_single(
            query=query,
            agent=agent,
            verbose=verbose,
            company=company,
            document_type=document_type,
            filing_date=filing_date,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Query the finance RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python query.py
  
  # Single query
  python query.py --query "What was Apple's revenue in 2023?"
  
  # Query with filters
  python query.py --query "What was the revenue?" --company AAPL --doc-type 10-K
  
  # Batch processing
  python query.py --batch queries.txt --verbose
  
  # Verbose output
  python query.py --query "Revenue growth trends?" --verbose
        """
    )
    
    # Query mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--query",
        type=str,
        help="Single query to execute"
    )
    mode_group.add_argument(
        "--batch",
        type=str,
        help="Path to file containing queries (one per line)"
    )
    
    # Filters
    parser.add_argument(
        "--company",
        type=str,
        help="Filter by company ticker (e.g., AAPL, MSFT)"
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        help="Filter by document type (e.g., 10-K, 10-Q)"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Filter by filing date (YYYY-MM-DD)"
    )
    
    # Options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed retrieval information"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG agent
    logger.info("Initializing RAG agent...")
    try:
        agent = RAGAgent()
    except Exception as e:
        logger.error(f"Failed to initialize RAG agent: {e}", exc_info=True)
        sys.exit(1)
    
    # Determine mode and execute
    try:
        if args.query:
            # Single query mode
            query_single(
                query=args.query,
                agent=agent,
                verbose=args.verbose,
                company=args.company,
                document_type=args.doc_type,
                filing_date=args.date,
            )
        
        elif args.batch:
            # Batch mode
            query_batch(
                batch_file=Path(args.batch),
                agent=agent,
                verbose=args.verbose,
                company=args.company,
                document_type=args.doc_type,
                filing_date=args.date,
            )
        
        else:
            # Interactive mode (default)
            query_interactive(
                agent=agent,
                verbose=args.verbose,
                company=args.company,
                document_type=args.doc_type,
                filing_date=args.date,
            )
    
    except KeyboardInterrupt:
        logger.info("\nQuery interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
