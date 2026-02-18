"""
Run RAG evaluation using RAGAS.

This script:
1. Loads the test dataset.
2. Runs the RAG pipeline for each question to generate answers and retrieve contexts.
3. Evaluates the results using RAGAS metrics:
   - Context Precision
   - Context Recall
   - Faithfulness
   - Answer Relevancy
"""
import json
import asyncio
import os
import pandas as pd
from pathlib import Path

try:
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    from ragas.metrics.collections import ContextPrecision, Faithfulness, ContextRecall
except ImportError:
    print("Ragas not installed. Please run: pip install ragas")
    exit(1)

from src.generation.rag_agent import RAGAgent
from src.utils.logger import get_logger
from config import load_config
from openai import AsyncOpenAI

config = load_config()
eval_config = config.get("evaluation", {})
logger = get_logger(__name__)


async def score_sample(sample: dict, metrics: list) -> dict:
    METRIC_FIELDS = {
    "context_precision": ["user_input", "retrieved_contexts", "reference"],
    "context_recall": ["user_input", "retrieved_contexts", "reference"],
    "faithfulness": ["user_input", "retrieved_contexts", "response"],
    "response_relevancy": ["user_input", "retrieved_contexts", "response"],
}
    """Score a single sample across all metrics concurrently."""    
    tasks = [
        metric.ascore(**{k: sample[k] for k in METRIC_FIELDS[metric.name]})
        for metric in metrics
    ]
    results = await asyncio.gather(*tasks)

    row = {
        "question": sample["user_input"],
        "response": sample["response"],
        "reference": sample["reference"],
        "retrieved_contexts": sample["retrieved_contexts"],
    }
    row.update({m.name: r.value for m, r in zip(metrics, results)})
    return row


async def run_evaluation(test_set_path: str, output_path: str):
    """
    Run evaluation across all test cases with multiple Ragas metrics.
    """
    test_set_path = Path(test_set_path)
    output_path = Path(output_path)

    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        return

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"Loaded {len(test_data)} test cases")

    # --- LLM setup ---
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    llm = llm_factory("arcee-ai/trinity-large-preview:free", client=client, max_tokens=30000)

    # --- Metrics ---
    metrics = [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        Faithfulness(llm=llm),
    ]

    # --- RAG pipeline ---
    logger.info("Initializing RAG Agent...")
    rag_agent = RAGAgent(verbose=False)

    samples = []

    logger.info("Running RAG pipeline to generate answers...")
    for i, item in enumerate(test_data[5:6]):
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        logger.info(f"Processing {i+1}/{len(test_data)}: {question}")

        try:
            response = rag_agent.query(question)
            retrieved_contexts = (
                [s.content for s in response.sources] if response.sources else []
            )
            samples.append(
                {
                    "user_input": question,
                    "response": response.answer,
                    "retrieved_contexts": retrieved_contexts,
                    "reference": ground_truth,
                }
            )
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            continue

    if not samples:
        logger.error("No samples collected. Exiting.")
        return

    # --- Evaluation ---
    logger.info(f"Starting Ragas evaluation on {len(samples)} samples...")

    all_results = []
    for i, sample in enumerate(samples):
        logger.info(f"Scoring sample {i+1}/{len(samples)}: {sample['user_input']}")
        try:
            row = await score_sample(sample, metrics)
            all_results.append(row)
        except Exception as e:
            logger.error(f"Error scoring sample {i+1}: {e}")
            continue

    if not all_results:
        logger.error("No results produced. Exiting.")
        return

    # --- Save results ---
    df = pd.DataFrame(all_results)
    metric_names = [m.name for m in metrics]

    # Build mean scores row and prepend it
    mean_row = {"question": "*** MEAN SCORES ***", "response": "", "reference": ""}
    mean_row.update(df[metric_names].mean().to_dict())
    df = pd.concat([pd.DataFrame([mean_row]), df], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")

    # --- Summary ---
    print("\n=== Mean Scores ===")
    print(df[metric_names].mean().to_string())
    print("\n=== Evaluation Summary ===")
    print(df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    asyncio.run(
        run_evaluation(
            test_set_path="./src/evaluation/data/test_set.json",
            output_path="./src/evaluation/data/eval_results.json",
        )
    )