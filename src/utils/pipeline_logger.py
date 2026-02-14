"""
Pipeline logging utilities for RAG execution tracking.

Provides functions to log detailed pipeline execution state at each node,
including retrieved chunks, filtering decisions, model responses, and routing.
"""

import os
from datetime import datetime
from typing import TextIO


def setup_pipeline_logger() -> tuple[str, TextIO]:
    """
    Setup dedicated file logger for pipeline execution.
    
    Returns:
        Tuple of (log_file_path, log_file_handle)
    """
    log_dir = "./logs/pipeline"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    return log_file_path, log_file


def log_pipeline_header(log_file: TextIO, user_query: str):
    """Write pipeline execution header."""
    log_file.write(f"{'='*80}\n")
    log_file.write(f"RAG PIPELINE EXECUTION\n")
    log_file.write(f"{'='*80}\n")
    log_file.write(f"Query: {user_query}\n")
    log_file.flush()


def log_pipeline_footer(log_file: TextIO):
    """Write pipeline completion footer."""
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"PIPELINE COMPLETE\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()


def log_state(node_name: str, state: dict, log_file: TextIO):
    """
    Log state after each node execution.
    
    Args:
        node_name: Name of the executed node
        state: Current pipeline state dictionary
        log_file: File handle to write logs to
    """
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"NODE: {node_name.upper()}\n")
    log_file.write(f"{'='*80}\n")
    
    if node_name == "analyze":
        _log_analyze_node(state, log_file)
    elif node_name == "retrieve":
        _log_retrieve_node(state, log_file)
    elif node_name == "rerank":
        _log_rerank_node(state, log_file)
    elif node_name == "verify":
        _log_verify_node(state, log_file)
    elif node_name == "reformulate":
        _log_reformulate_node(state, log_file)
    elif node_name == "decide_next":
        _log_decide_next_node(state, log_file)
    elif node_name == "generate":
        _log_generate_node(state, log_file)
    
    log_file.flush()


def _log_analyze_node(state: dict, log_file: TextIO):
    """Log query analysis results."""
    log_file.write(f"User Query: {state['user_query']}\n")
    if state.get('query_analysis'):
        qa = state['query_analysis']
        log_file.write(f"Intent: {qa.intent}\n")
        log_file.write(f"Companies: {qa.companies}\n")
        log_file.write(f"Document Types: {qa.document_types}\n")
        log_file.write(f"Sub-queries: {len(qa.sub_queries)}\n")
        for i, sq in enumerate(qa.sub_queries, 1):
            log_file.write(f"  {i}. {sq.query} (company={sq.company}, doc_type={sq.document_type})\n")


def _log_retrieve_node(state: dict, log_file: TextIO):
    """Log retrieval results."""
    results = state.get('current_results', [])
    sq_idx = state.get('current_sq_idx', 0)
    if state.get('sub_queries') and sq_idx < len(state['sub_queries']):
        sq = state['sub_queries'][sq_idx]
        log_file.write(f"Query: {sq.query}\n")
    log_file.write(f"Filters: company={state.get('current_company')}, doc_type={state.get('current_doc_type')}\n")
    log_file.write(f"Retrieved: {len(results)} chunks\n")
    for i, r in enumerate(results[:10], 1):  # Show top 10
        log_file.write(
            f"  [{i}] Score: {r.score:.4f} | {r.metadata.get('company', '?')} "
            f"{r.metadata.get('document_type', '?')} p.{r.metadata.get('page_number', '?')}\n"
        )
        log_file.write(f"      Content: {r.content[:20]}...\n")


def _log_rerank_node(state: dict, log_file: TextIO):
    """Log reranking results."""
    results = state.get('current_results', [])
    log_file.write(f"After reranking: {len(results)} chunks\n")
    for i, r in enumerate(results, 1):
        log_file.write(
            f"  [{i}] Score: {r.score:.4f} | {r.metadata.get('company', '?')} "
            f"{r.metadata.get('document_type', '?')} p.{r.metadata.get('page_number', '?')}\n"
        )


def _log_verify_node(state: dict, log_file: TextIO):
    """Log verification results."""
    sq_idx = state.get('current_sq_idx', 0)
    if state.get('sub_queries') and sq_idx < len(state['sub_queries']):
        sq = state['sub_queries'][sq_idx]
        log_file.write(f"Verification Score: {sq.verification_score:.2f}\n")
        log_file.write(f"Verified: {sq.verified}\n")
        log_file.write(f"Reason: {sq.verification_reason}\n")
        log_file.write(f"Missing: {sq.verification_missing}\n")


def _log_reformulate_node(state: dict, log_file: TextIO):
    """Log query reformulation."""
    sq_idx = state.get('current_sq_idx', 0)
    if state.get('sub_queries') and sq_idx < len(state['sub_queries']):
        sq = state['sub_queries'][sq_idx]
        log_file.write(f"Original Query: {sq.reformulation_history[-1] if sq.reformulation_history else 'N/A'}\n")
        log_file.write(f"Reformulated Query: {sq.query}\n")


def _log_decide_next_node(state: dict, log_file: TextIO):
    """Log routing decision."""
    log_file.write(f"Next Action: {state.get('next_action')}\n")
    log_file.write(f"Retry Count: {state.get('retry_count')}/{state.get('total_retries')}\n")


def _log_generate_node(state: dict, log_file: TextIO):
    """Log generation results with full context."""
    log_file.write(f"Total Results: {len(state.get('all_results', []))}\n")
    log_file.write(f"Answer Length: {len(state.get('answer', ''))} chars\n")
    log_file.write(f"Sources: {len(state.get('sources', []))}\n")
    
    # Log full context passed to LLM
    log_file.write(f"\n--- Full Context Passed to Generation LLM ---\n")
    all_results = state.get('all_results', [])
    if all_results:
        for i, r in enumerate(all_results, 1):
            meta = r.metadata
            company = meta.get("company", "?")
            doc_type = meta.get("document_type", "?")
            page = meta.get("page_number", "?")
            filing_date = meta.get("filing_date", "")
            log_file.write(f"\n[Source {i}] ({company} {doc_type}, p.{page}, filed {filing_date}) chunk_id: {r.chunk_id}\n")
            log_file.write(f"{r.content[:200]}\n")
    else:
        log_file.write("(No context available)\n")
    
    log_file.write(f"\n--- Final Answer ---\n")
    log_file.write(f"{state.get('answer', '')}\n")
    log_file.write(f"\n--- Sources Referenced ---\n")
    for s in state.get('sources', []):
        log_file.write(
            f"  [{s.source_number}] {s.company} {s.document_type} "
            f"p.{s.page_number} (score: {s.score:.4f})\n"
        )
