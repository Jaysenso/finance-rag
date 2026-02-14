from src.utils.logger import get_logger
from src.test.sample_query import rag_test_questions
from src.generation import RAGAgent

logger = get_logger(__name__)

def query_rag(query: str):
    """Handle query mode with a passed-in query string."""
    agent = RAGAgent()
    response = agent.query(query)

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    print(f"\n{response.answer}")

    print(f"\n--- Sources ({len(response.sources)}) ---")
    for s in response.sources:
        print(f"  [{s.source_number}] {s.company} {s.document_type} p.{s.page_number} (score: {s.score:.3f})")

    print(f"\n--- Sub-queries ({len(response.sub_queries)}) ---")
    for sq in response.sub_queries:
        print(f"  - {sq.query} | verified={sq.verified} | score={sq.verification_score:.2f} | retries={sq.retry_count}")

    print(f"\nTotal chunks: {response.total_chunks_retrieved} | Total retries: {response.total_retries}")


def main():

    logger.info("Application started")
    
    for query in rag_test_questions[-5:]:
        query_rag(query)

    logger.info("Task Completed")


if __name__ == "__main__":
    main()
