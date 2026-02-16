"""
Hypothetical question generator for HyPE retrieval.

Generates time-aware hypothetical questions from financial document chunks
to improve question-to-question semantic matching.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.logger import get_logger
from src.generation.llm import BaseLLM, get_llm
from src.preprocessing.models import Chunk, GeneratedQuestions
from src.utils.prompts import QUESTION_GENERATION_SYSTEM, QUESTION_GENERATION_USER

logger = get_logger(__name__)



class QuestionGenerator:
    """
    Generates hypothetical questions from financial document chunks.
    
    Questions are time-aware and include company context to improve
    retrieval precision across similar documents from different periods.
    """
    
    def __init__(
        self,
        llm: BaseLLM = None,
        questions_per_chunk: int = 3,
        max_tokens: int = 150,
        num_threads: int = 4,
    ):
        """
        Initialize question generator.
        
        Args:
            llm: Language model for question generation
            questions_per_chunk: Number of questions to generate per chunk
            max_tokens: Max tokens for generation
            num_threads: Number of threads for parallel generation
        """
        self.llm = llm or get_llm()
        self.questions_per_chunk = questions_per_chunk
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        
        logger.info(
            f"QuestionGenerator initialized "
            f"(questions_per_chunk={questions_per_chunk}, threads={num_threads})"
        )
    
    def generate(self, chunk: Chunk) -> Optional[GeneratedQuestions]:
        """
        Generate hypothetical questions for a single chunk.
        
        Args:
            chunk: Chunk object to generate questions for
            
        Returns:
            GeneratedQuestions object or None if generation fails
        """
        try:
            messages = [
                {"role": "system", "content": QUESTION_GENERATION_SYSTEM.format(questions_per_chunk=self.questions_per_chunk)},
                {"role": "user", "content": QUESTION_GENERATION_USER.format(
                    company=chunk.company or "Unknown",
                    document_type=chunk.document_type or "Unknown",
                    filing_date=chunk.filing_date or "Unknown",
                    page_number=chunk.page_number or "?",
                    content=chunk.content,
                    questions_per_chunk=self.questions_per_chunk or 1,
                )},
            ]
            
            parsed = self.llm.generate_json(messages, max_tokens=self.max_tokens)
            
            if parsed is None or "questions" not in parsed:
                logger.warning(f"Question generation failed for chunk {chunk.chunk_id}")
                return None
            
            questions = parsed["questions"]
            
            # Validate we got the right number of questions
            if len(questions) != self.questions_per_chunk:
                logger.warning(
                    f"Expected {self.questions_per_chunk} questions, got {len(questions)} "
                    f"for chunk {chunk.chunk_id}"
                )
                # Pad or truncate to expected count
                if len(questions) < self.questions_per_chunk:
                    # Pad with generic question
                    while len(questions) < self.questions_per_chunk:
                        questions.append(
                            f"What information does {chunk.company} provide in their "
                            f"{chunk.document_type} filed on {chunk.filing_date}?"
                        )
                else:
                    questions = questions[:self.questions_per_chunk]
            
            return GeneratedQuestions(
                chunk_id=chunk.chunk_id,
                questions=questions,
                company=chunk.company,
                document_type=chunk.document_type,
                filing_date=chunk.filing_date,
                page_number=chunk.page_number,
            )
            
        except Exception as e:
            logger.error(f"Error generating questions for chunk {chunk.chunk_id}: {e}")
            return None
    
    def generate_batch(
        self,
        chunks: List[Chunk],
        show_progress: bool = True,
    ) -> List[GeneratedQuestions]:
        """
        Generate questions for a batch of chunks using parallel execution.
        Preserves the order of input chunks.
        
        Args:
            chunks: List of chunks to process
            show_progress: Whether to log progress
            
        Returns:
            List of GeneratedQuestions objects (excludes failed generations)
        """
        results = []
        total = len(chunks)
        completed = 0
        
        logger.info(f"Starting parallel question generation with {self.num_threads} threads for {total} chunks")
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # map preserves order
            futures = [executor.submit(self.generate, chunk) for chunk in chunks]
            
            for i, future in enumerate(futures):
                try:
                    generated = future.result()
                    completed += 1
                    if show_progress and completed % 10 == 0:
                        logger.info(f"Generated questions for {completed}/{total} chunks")
                        
                    if generated:
                        results.append(generated)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunks[i].chunk_id}: {e}")
        
        logger.info(
            f"Question generation complete: {len(results)}/{total} successful "
            f"({len(results) * self.questions_per_chunk} total questions)"
        )
        
        return results


def get_question_generator(**kwargs) -> QuestionGenerator:
    """
    Factory function to create a question generator.
    
    Args:
        **kwargs: Override any QuestionGenerator constructor parameter
        
    Returns:
        QuestionGenerator instance
    """
    from config import load_config
    config = load_config()
    hype_config = config.get("hype", {})
    
    questions_per_chunk = kwargs.pop(
        "questions_per_chunk",
        hype_config.get("questions_per_chunk", 1)
    )
    max_tokens = kwargs.pop(
        "max_tokens",
        hype_config.get("max_tokens", 150)
    )
    num_threads = kwargs.pop(
        "num_threads",
        hype_config.get("num_threads", 10)
    )
    
    return QuestionGenerator(
        questions_per_chunk=questions_per_chunk,
        max_tokens=max_tokens,
        num_threads=num_threads,
        **kwargs
    )
