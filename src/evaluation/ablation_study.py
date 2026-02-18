"""
Ablation Study Script ()

Compares three approaches for answering financial questions:
1. RAG Pipeline (Current System): Grounded generation using retrieved context.
2. LLM Only (Baseline): Direct query to the generation LLM without context.
3. Web-Search LLM (External Benchmark): Direct query to a model with web capabilities.

"""

import argparse
import json
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.generation.rag_agent import RAGAgent
from src.generation.llm import OpenRouterLLM, get_llm
from src.utils.logger import get_logger
from src.utils.prompts import LLM_AS_JUDGE_SYSTEM, LLM_AS_JUDGE_USER, LLM_AS_JUDGE_SYSTEM_NO_REF, LLM_AS_JUDGE_USER_NO_REF
from config import load_config


logger = get_logger(__name__)
config = load_config()

def llm_as_judge(input: List[Dict]) -> List[Dict]:

    judge_model_name = config["evaluation"].get("llm_as_judge_llm", None)
    if judge_model_name is None:
        logger.warning("Please define judge model name in config.yaml.")
        return []

    judge = OpenRouterLLM(model=judge_model_name)
    logger.info(f"Initializing LLM as Judge {judge_model_name}")

    results = []
    for i, item in enumerate(input):
        question = item["question"]
        ground_truth = item["ground_truth"]
        rag_answer = item["rag_answer"]
        llm_only_answer = item["llm_only_answer"]
        web_search_answer = item["web_search_answer"]

        # Evaluation with reference (ground truth)
        prompt = LLM_AS_JUDGE_USER.format(
            question=question,
            ground_truth=ground_truth,
            rag_answer=rag_answer,
            llm_only_answer=llm_only_answer,
            web_search_answer=web_search_answer
        )
        messages = [
            {"role": "system", "content": LLM_AS_JUDGE_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        response_w_ref = judge.generate(messages)

        # Evaluation without reference
        no_ref_prompt = LLM_AS_JUDGE_USER_NO_REF.format(
            question=question,
            rag_answer=rag_answer,
            llm_only_answer=llm_only_answer,
            web_search_answer=web_search_answer
        )
        messages = [
            {"role": "system", "content": LLM_AS_JUDGE_SYSTEM_NO_REF},
            {"role": "user", "content": no_ref_prompt}
        ]
        response_no_ref = judge.generate(messages)

        results.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "rag_answer": rag_answer,
            "llm_only_answer": llm_only_answer,
            "web_search_answer": web_search_answer,
            "judge_response_with_ref": response_w_ref,
            "judge_response_no_ref": response_no_ref
        })

    return results

def run_ablation(
    test_set_path: str,
    output_path: str,
    limit: int,
    web_model_name: str,
    llm_only_model_name: str = None
):
    """
    Run the ablation study comparing RAG, LLM-only, and Web-Search LLM.
    """
    test_set_path = Path(test_set_path)
    output_path = Path(output_path)
    
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        return

    # Load test set
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Slice data if limit is set
    if limit > 0:
        test_data = test_data[:limit]
        
    logger.info(f"Starting ablation study on {len(test_data)} test cases")

    # 1. Initialize RAG Agent
    # We use the default configuration from config.yaml
    logger.info("Initializing RAG Agent...")
    rag_agent = RAGAgent(verbose=False)
    
    # 2. Initialize LLM Only 
    # Use the same model as the RAG generation_llm for a fair comparison of "Context vs No Context"
    if llm_only_model_name:
        llm_model = llm_only_model_name
    else:
        llm_model = config.get("generation_llm", {}).get("model", "google/gemini-2.0-flash-001")
    
    logger.info(f"Initializing LLM-Only baseline with model: {llm_model}")
    llm_only = OpenRouterLLM(model=llm_model)

    # 3. Initialize Web Search LLM
    # This represents an "Oracle" or "External Tool" baseline
    logger.info(f"Initializing Web-Search Benchmark with model: {web_model_name}")
    try:
        web_llm = OpenRouterLLM(model=web_model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Web LLM: {e}")
        web_llm = None

    results = []

    for i, item in enumerate(test_data[5:]):
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        logger.info(f"Processing {i+1}/{len(test_data)}: {question}")

        row = {
            "question_id": i + 1,
            "question": question,
            "ground_truth": ground_truth
        }

        # --- Approach 1: RAG Pipeline ---
        try:
            rag_response = rag_agent.query(question)
            row["rag_answer"] = rag_response.answer
            # Store sources for inspection
            row["rag_sources"] = "\n".join([
                f"[{s.company} {s.document_type} p.{s.page_number}]: {s.content[:100]}..." 
                for s in rag_response.sources
            ])
            # Check if sources were actually used/retrieved
            row["rag_retrieved_count"] = len(rag_response.sources)
        except Exception as e:
            logger.error(f"RAG failed for q={i}: {e}")
            row["rag_answer"] = f"ERROR: {e}"
            row["rag_sources"] = ""
            row["rag_retrieved_count"] = 0

        # --- Approach 2: LLM Only (No Context) ---
        try:
            messages = [{"role": "user", "content": question}]
            row["llm_only_answer"] = llm_only.generate(messages)
        except Exception as e:
            logger.error(f"LLM-only failed for q={i}: {e}")
            row["llm_only_answer"] = f"ERROR: {e}"

        # --- Approach 3: Web Search LLM ---
        if web_llm:
            try:
                messages = [{"role": "user", "content": question}]
                row["web_search_answer"] = web_llm.web_search(messages)
            except Exception as e:
                logger.error(f"Web LLM failed for q={i}: {e}")
                row["web_search_answer"] = f"ERROR: {e}" 
        else:
            row["web_search_answer"] = "SKIPPED"

        results.append(row)

    # Save results to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        json.dump(results, f, indent=4) 

    logger.info(f"Ablation study completed. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run RAG Verification/Ablation Study")
    parser.add_argument("--test-set", default="src/evaluation/data/ablation_test_set.json", help="Path to test set JSON")
    parser.add_argument("--output", default="src/evaluation/data/ablation_results.json", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=20, help="Number of test cases to run (0 for all)")
    parser.add_argument("--web-model", default=config["evaluation"].get("web_llm"),
                      help="Model ID for Web Search comparison (e.g. perplexity/llama-3-sonar-large-32k-online or google/gemini-2.0-flash-001)")
    parser.add_argument("--llm-model", default=config["generation_llm"].get("model"), 
                      help="Model ID for LLM-only baseline (defaults to config.generation_llm)")
    
    args = parser.parse_args()
    
    run_ablation(
        args.test_set, 
        args.output, 
        args.limit, 
        args.web_model,
        args.llm_model
    )

if __name__ == "__main__":
    main()
