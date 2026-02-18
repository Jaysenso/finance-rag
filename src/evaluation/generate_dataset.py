"""
Generate synthetic test set for RAG evaluation.

This script scans the Qdrant vector store for chunks and uses an LLM to generate
(Question, Ground Truth Answer, Context) triplets for evaluation.
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import argparse

from src.utils.logger import get_logger
from src.generation.llm import get_llm, BaseLLM
from src.indexing.vector_store import get_vector_store, QdrantVectorStore
from qdrant_client.models import ScrollRequest

logger = get_logger(__name__)

# System prompt for generating QA pairs
QA_GENERATION_SYSTEM = """You are an expert financial analyst creating a test dataset for a RAG system.
Your task is to generate {count} high-quality question-answer pairs based ONLY on the provided text chunk.

Guidelines:
1. Questions should be specific and answerable using the chunk.
2. Answers must be derived strictly from the chunk.
3. Include the specific sentences from the chunk that support the answer as "context".
4. Ensure questions include relevant entities (Company Name, Date) if mentioned in the text.

Return valid JSON list:
[
  {{
    "question": "What was Apple's revenue in Q3 2023?",
    "ground_truth": "Apple's revenue was $81.8 billion.",
    "context": "Apple posted quarterly revenue of $81.8 billion..."
  }}
]
"""

class TestSetGenerator:
    def __init__(self, llm: BaseLLM = None, store: QdrantVectorStore = None):
        self.llm = llm or get_llm()
        self.store = store or get_vector_store()

    def generate_for_chunk(self, chunk_content: str, metadata: Dict, count: int = 1) -> List[Dict]:
        """Generate QA pairs for a single chunk."""
        # Add metadata context to the chunk content for the LLM
        context_str = (
            f"Company: {metadata.get('company', 'Unknown')}\n"
            f"Date: {metadata.get('filing_date', 'Unknown')}\n"
            f"Type: {metadata.get('document_type', 'Unknown')}\n\n"
            f"Text:\n{chunk_content}"
        )

        messages = [
            {"role": "system", "content": QA_GENERATION_SYSTEM.format(count=count)},
            {"role": "user", "content": context_str}
        ]

        try:
            response = self.llm.generate_json(messages)

            if not response or not isinstance(response, list):
                logger.warning("Invalid JSON response from LLM")
                return []
            
            for item in response:
                item["metadata"] = metadata
                # Use the chunk content as the golden context
                item["ground_truth_context"] = [chunk_content] 
            
            return response
        except Exception as e:
            logger.error(f"Error generating QA: {e}")
            return []

    def generate_dataset(self, output_file: str, sample_size: int = 10, offset: int = 0):
        """
        Generate a dataset by fetching ALL IDs and sampling k chunks.
        
        Args:
            output_file: Path to save the dataset
            sample_size: Number of chunks to process
            offset: Ignored
        """
        logger.info(f"Fetching all chunk IDs from Qdrant...")
        logger.info(f"Starting to fetch IDs...")
        
        # 1. Fetch all IDs using scroll
        all_ids = []
        next_offset = None
        while True:
            try:
                points, next_offset = self.store.client.scroll(
                    collection_name=self.store.collection_name,
                    limit=1000,
                    offset=next_offset,
                    with_payload=False,
                    with_vectors=False
                )
                current_batch_ids = [p.id for p in points]
                all_ids.extend(current_batch_ids)
                logger.info(f"Fetched {len(current_batch_ids)} IDs. Total so far: {len(all_ids)}")
                if next_offset is None:
                    break
            except Exception as e:
                logger.error(f"Error fetching IDs: {e}")
                return
        
        total_points = len(all_ids)
        logger.info(f"Total chunks available: {total_points}")
        
        if total_points == 0:
            logger.error("No points in collection.")
            return

        # 2. Sample k IDs
        num_samples = min(sample_size, total_points)
        sampled_ids = random.sample(all_ids, num_samples)
        logger.info(f"Selected {len(sampled_ids)} random chunks")


        dataset = []
        
        # 3. Retrieve content for sampled IDs
        # Retrieve in batches to be efficient
        batch_size = 100
        logger.info(f"Retrieving content for {len(sampled_ids)} chunks...")
        for i in range(0, len(sampled_ids), batch_size):
            batch_ids = sampled_ids[i:i+batch_size]
            try:
                points = self.store.client.retrieve(
                    collection_name=self.store.collection_name,
                    ids=batch_ids,
                    with_payload=True,
                    with_vectors=False
                )
            except Exception as e:
                logger.error(f"Error retrieving batch: {e}")
                continue
            
            for point in points:
                payload = point.payload
                content = payload.get("content")
                metadata = {k: v for k, v in payload.items() if k != "content"}
                
                if not content:
                    continue

                logger.info(f"Generating QA for chunk {point.id}")
                qa_pairs = self.generate_for_chunk(content, metadata)
                dataset.extend(qa_pairs)

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved {len(dataset)} items to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate RAG test dataset")
    parser.add_argument("--output", default="src/evaluation/data/test_set.json", help="Output JSON file")
    parser.add_argument("--samples", type=int, default=10, help="Number of chunks to sample")
    parser.add_argument("--offset", type=int, default=0, help="Offset for scrolling")
    
    args = parser.parse_args()
    
    generator = TestSetGenerator()
    generator.generate_dataset(args.output, args.samples, args.offset)

if __name__ == "__main__":
    main()
