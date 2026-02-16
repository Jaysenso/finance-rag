"""
HyPE ingestion pipeline - Generate and index hypothetical questions.

This script processes existing chunks to generate hypothetical questions
and indexes them into the question database.
"""
from src.utils.logger import get_logger
from src.preprocessing.question_generator import get_question_generator
from src.indexing.embedder import get_embedder
from src.indexing.question_store import get_question_store
from src.preprocessing.models import Chunk
from typing import List

logger = get_logger(__name__)


def index_questions_for_chunks(chunks: List[Chunk]) -> None:
    """
    Generate hypothetical questions for chunks and index them.
    
    Args:
        chunks: List of Chunk objects to process
    """
    logger.info(f"Starting HyPE indexing for {len(chunks)} chunks")
    
    # Initialize components
    question_gen = get_question_generator()
    embedder = get_embedder()
    question_store = get_question_store()
    
    # Step 1: Generate questions
    logger.info("Generating hypothetical questions...")
    generated_questions_list = question_gen.generate_batch(chunks, show_progress=True)
    
    if not generated_questions_list:
        logger.warning("No questions generated, aborting")
        return
    
    # Step 2: Embed questions
    logger.info(f"Embedding {len(generated_questions_list)} question sets...")
    all_question_embeddings = []
    
    for i, gen_q in enumerate(generated_questions_list):
        if (i + 1) % 10 == 0:
            logger.info(f"Embedded questions for {i + 1}/{len(generated_questions_list)} chunks")
        
        # Embed all questions for this chunk
        question_embeddings = embedder.embed_batch(gen_q.questions)
        all_question_embeddings.append(question_embeddings)
    
    # Step 3: Index to question database
    logger.info("Indexing questions to question database...")
    question_store.upsert(generated_questions_list, all_question_embeddings)
    
    logger.info(
        f"HyPE indexing complete! "
        f"Indexed {len(generated_questions_list) * question_gen.questions_per_chunk} questions "
        f"for {len(generated_questions_list)} chunks"
    )


if __name__ == "__main__":
    # Example usage
    logger.info("HyPE Ingestion Pipeline")
    logger.info("=" * 60)
    
