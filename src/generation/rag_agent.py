"""
Agentic RAG orchestrator using LangGraph.

Pipeline: Analyze → Retrieve → Rerank → Verify → (Retry?) → Generate
Each step is a graph node. Self-correction is a conditional edge
"""
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.generation.models import SubQuery, QueryAnalysis, Source, RAGResponse

from src.utils.logger import get_logger
from src.utils.pipeline_logger import (
    setup_pipeline_logger,
    log_pipeline_header,
    log_pipeline_footer,
    log_state,
)
from src.generation.llm import BaseLLM, get_llm
from src.generation.reranker import BaseReranker, get_reranker
from src.utils.prompts import (
    QUERY_ANALYSIS_SYSTEM, QUERY_ANALYSIS_USER,
    VERIFICATION_SYSTEM, VERIFICATION_USER,
    REFORMULATION_SYSTEM, REFORMULATION_USER,
    GENERATION_SYSTEM, GENERATION_USER,
    VALID_DOC_TYPES,
    format_sources_for_prompt,
)
from src.indexing.embedder import BaseEmbedder, get_embedder
from src.indexing.vector_store import QdrantVectorStore, SearchResult, get_vector_store
from src.generation.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from config import load_config

config = load_config()
retrieval_config = config["retrieval"]
hype_config = config.get("hype", {})
logger = get_logger(__name__)


# ── Graph State ──────────────────────────────────────────────

class RAGState(TypedDict):
    user_query: str
    query_analysis: Optional[QueryAnalysis]
    sub_queries: List[SubQuery]
    current_sq_idx: int
    # Per sub-query retrieval state
    current_results: List[SearchResult]
    current_company: Optional[str]
    current_doc_type: Optional[str]
    current_limit: int
    retry_count: int
    # Accumulated across sub-queries
    all_results: List[SearchResult]
    total_retries: int
    # Output
    answer: str
    sources: List[Source]
    next_action: str


# ── RAGAgent ─────────────────────────────────────────────────

class RAGAgent:
    """
    Agentic RAG pipeline for SEC financial filings using LangGraph.

    Graph: analyze → retrieve → rerank → verify → decide_next
                                                      ├→ retrieve (retry/next sub-query)
                                                      └→ generate → END
    """

    def __init__(
        self,
        llm: BaseLLM = None,
        generation_llm: BaseLLM = None,
        embedder: BaseEmbedder = None,
        reranker: BaseReranker = None,
        vector_store: QdrantVectorStore = None,
        hybrid_retriever: HybridRetriever = None,
        max_retries: int = None,
        retrieval_limit: int = None,
        verification_threshold: float = None,
        score_threshold: float = None,
        verbose: bool = False,
    ):
        # LLM for analysis/verification/reformulation
        self.llm = llm or get_llm()
        self.verbose = verbose
        
        # LLM for final answer generation (supports tool calling)
        self.generation_llm = generation_llm or get_llm(config_key="generation_llm")
        
        self.embedder = embedder or get_embedder()
        self.reranker = reranker or get_reranker()
        self.vector_store = vector_store or get_vector_store()
        self.max_retries = max_retries or retrieval_config["max_retries"]
        self.retrieval_limit = retrieval_limit or retrieval_config["retrieval_limit"]
        self.verification_threshold = verification_threshold or retrieval_config["verification_threshold"]
        self.score_threshold = score_threshold or retrieval_config["score_threshold"]
        
        # HyPE hybrid retriever (optional)
        self.use_hype = hype_config.get("enabled", False)
        if self.use_hype:
            self.hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
            logger.info("HyPE dual-database retrieval enabled")
        else:
            self.hybrid_retriever = None

        self.graph = self._build_graph()

        if self.verbose:
            logger.info(
                f"RAGAgent initialized (retries={self.max_retries}, "
                f"limit={self.retrieval_limit}, threshold={self.verification_threshold}, "
                f"hype={self.use_hype})"
            )

    # ── Graph Construction ───────────────────────────────────

    def _build_graph(self):
        """Build the LangGraph state graph."""
        graph = StateGraph(RAGState)

        # Nodes
        graph.add_node("analyze", self._node_analyze)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("rerank", self._node_rerank)
        graph.add_node("verify", self._node_verify)
        graph.add_node("reformulate", self._node_reformulate)
        graph.add_node("decide_next", self._node_decide_next)
        graph.add_node("generate", self._node_generate)

        # Edges
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            lambda s: "rerank" if s["current_results"] else "decide_next",
        )
        graph.add_edge("rerank", "verify")
        graph.add_edge("verify", "decide_next")
        graph.add_edge("reformulate", "retrieve")
        graph.add_conditional_edges(
            "decide_next",
            lambda s: s["next_action"],
        )
        graph.add_edge("generate", END)

        return graph.compile()

    # ── Nodes Definition ───────────────────────────────────────────

    # Pipeline logging methods moved to src/utils/pipeline_logger.py

    def query(self, user_query: str) -> RAGResponse:
        """
        Process a user query through the LangGraph RAG pipeline.

        Args:
            user_query: Question about SEC filings / Financial Reports

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        logger.info(f"Processing query: {user_query}")

        # Setup pipeline logger (ALWAYS active)
        log_path, log_file = setup_pipeline_logger()
        # Only show log path if verbose, or maybe always? User asked for "basic prompt". 
        # "Processing query" is basic. "Log file created at..." might be useful but "basic".
        # Let's keep it simple: detailed logs show path.
        if self.verbose:
            logger.info(f"Pipeline log: {log_path}")
        
        log_pipeline_header(log_file, user_query)

        initial_state: RAGState = {
            "user_query": user_query,
            "query_analysis": None,
            "sub_queries": [],
            "current_sq_idx": 0,
            "current_results": [],
            "current_company": None,
            "current_doc_type": None,
            "current_limit": self.retrieval_limit,
            "retry_count": 0,
            "all_results": [],
            "total_retries": 0,
            "answer": "",
            "sources": [],
            "next_action": "",
        }

        # Stream through graph and log each node
        full_state = initial_state.copy()
        try:
            for chunk in self.graph.stream(initial_state):
                node_name = list(chunk.keys())[0]
                partial_state = chunk[node_name]
                # Accumulate state updates
                full_state.update(partial_state)
                
                # ALWAYS log state to file
                log_state(node_name, full_state, log_file)
                
                    
        finally:
            log_pipeline_footer(log_file)
            log_file.close()

        return RAGResponse(
            answer=full_state["answer"],
            sources=full_state["sources"],
            sub_queries=full_state["sub_queries"],
            query_analysis=full_state["query_analysis"],
            total_chunks_retrieved=len(full_state["sources"]),
            total_retries=full_state["total_retries"],
        )

    # ── Node: Analyze ────────────────────────────────────────

    def _node_analyze(self, state: RAGState) -> dict:
        """Decompose user query into sub-queries with filters."""
        user_query = state["user_query"]

        available_companies = self.vector_store.get_available_companies()
        available_companies_str = ", ".join(available_companies)

        messages = [
            {"role": "system", "content": QUERY_ANALYSIS_SYSTEM.format(companies_str=available_companies_str)},
            {"role": "user", "content": QUERY_ANALYSIS_USER.format(query=user_query)},
        ]

        parsed = self.llm.generate_json(messages, max_tokens=1024)

        # Fallback: unfiltered wide search if LLM fails
        if parsed is None:
            logger.warning("Query analysis failed, falling back to wide search")
            analysis = QueryAnalysis(
                intent="general", companies=[], document_types=[],
                time_periods=[], needs_table=False,
                sub_queries=[SubQuery(query=user_query)],
                raw_query=user_query,
            )
        else:
            analysis = self._parse_analysis(parsed, user_query)

        first_sq = analysis.sub_queries[0]
        if self.verbose:
            logger.info(
                f"Query analysis: intent={analysis.intent}, "
                f"companies={analysis.companies}, "
                f"sub_queries={len(analysis.sub_queries)}"
            )

        return {
            "query_analysis": analysis,
            "sub_queries": analysis.sub_queries,
            "current_sq_idx": 0,
            "current_company": first_sq.company,
            "current_doc_type": first_sq.document_type,
            "current_limit": self.retrieval_limit,
            "retry_count": 0,
        }

    def _parse_analysis(self, parsed: dict, user_query: str) -> QueryAnalysis:
        """Parse LLM JSON output into QueryAnalysis."""
        try:
            sub_queries = []
            for sq_data in parsed.get("sub_queries", []):
                doc_type = sq_data.get("document_type")
                if doc_type and doc_type not in VALID_DOC_TYPES:
                    doc_type = None

                query_text = sq_data.get("query", user_query)
                sub_queries.append(SubQuery(
                    query=query_text,
                    company=sq_data.get("company"),
                    document_type=doc_type,
                    time_hint=sq_data.get("time_hint"),
                    original_query=query_text,
                ))

            # FALLBACK if LLM returns with 0 response
            if not sub_queries:
                sub_queries = [SubQuery(query=user_query, original_query=user_query)]

            return QueryAnalysis(
                intent=parsed.get("intent", "general"),
                companies=parsed.get("companies", []),
                document_types=parsed.get("document_types", []),
                time_periods=parsed.get("time_periods", []),
                needs_table=parsed.get("needs_table", False),
                sub_queries=sub_queries,
                raw_query=user_query,
            )

        except (KeyError, TypeError) as e:
            logger.warning(f"Query analysis structure invalid: {e}, wide search fallback")
            return QueryAnalysis(
                intent="general", companies=[], document_types=[],
                time_periods=[], needs_table=False,
                sub_queries=[SubQuery(query=user_query, original_query=user_query)],
                raw_query=user_query,
            )

    # ── Node: Retrieve ───────────────────────────────────────

    def _node_retrieve(self, state: RAGState) -> dict:
        """Retrieve using hybrid or standard search."""
        current_sq = state["sub_queries"][state["current_sq_idx"]]
        
        # Use HyPE hybrid retrieval if enabled
        if self.use_hype and self.hybrid_retriever:
            logger.info(f"Using HyPE hybrid retrieval for: {current_sq.query}")
            results = self.hybrid_retriever.retrieve(
                query=current_sq.query,
                limit=state["current_limit"],
                company=state["current_company"],
                document_type=state["current_doc_type"],
                score_threshold=self.score_threshold,
            )
        else:
            # Standard content-only retrieval
            query_embedding = self.embedder.embed(current_sq.query)
            results = self.vector_store.search(
                query_embedding=query_embedding,
                limit=state["current_limit"],
                company=state["current_company"],
                document_type=state["current_doc_type"],
                score_threshold=self.score_threshold,
            )

        if self.verbose:
            logger.info(
                f"Retrieved {len(results)} chunks for: {current_sq.query} "
                f"(company={state['current_company']}, doc_type={state['current_doc_type']})"
            )
        return {"current_results": results}

    # ── Node: Rerank ─────────────────────────────────────────

    def _node_rerank(self, state: RAGState) -> dict:
        """Rerank retrieved chunks using cross-encoder."""
        sq = state["sub_queries"][state["current_sq_idx"]]
        reranked = self.reranker.rerank(sq.query, state["current_results"])
        return {"current_results": reranked}

    # ── Node: Verify ─────────────────────────────────────────

    def _node_verify(self, state: RAGState) -> dict:
        """LLM verifies context relevance to the query."""
        sq = state["sub_queries"][state["current_sq_idx"]]
        results = state["current_results"]

        context_text = "\n\n---\n\n".join(
            f"[Chunk {i+1}] ({r.metadata.get('company', '?')} "
            f"{r.metadata.get('document_type', '?')}, p.{r.metadata.get('page_number', '?')})\n"
            f"{r.content}"
            for i, r in enumerate(results)
        )

        messages = [
            {"role": "system", "content": VERIFICATION_SYSTEM},
            {"role": "user", "content": VERIFICATION_USER.format(
                query=sq.query, context=context_text,
            )},
        ]

        parsed = self.llm.generate_json(messages, max_tokens=256)

        # Fail fast if LLM returns unparseable JSON
        if parsed is None:
            raise RuntimeError(
                f"Verification failed: LLM returned invalid JSON for sub-query '{sq.query}'. "
                "Check LLM connectivity or prompt formatting."
            )

        # Fail fast if LLM omits required 'score' field
        if "score" not in parsed:
            raise RuntimeError(
                f"Verification failed: LLM response missing 'score' field for sub-query '{sq.query}'. "
                f"Response: {parsed}"
            )

        score = float(parsed["score"])
        is_relevant = score >= self.verification_threshold
        if self.verbose:
            logger.info(
                f"Verification: score={score:.2f}, relevant={is_relevant}, "
                f"reason={parsed.get('reason', 'N/A')}"
            )

        # Update sub-query via state return
        sub_queries = list(state["sub_queries"])
        idx = state["current_sq_idx"]
        sub_queries[idx].verification_score = score
        sub_queries[idx].verified = is_relevant
        sub_queries[idx].verification_reason = parsed.get("reason", "")
        sub_queries[idx].verification_missing = parsed.get("missing", "")

        return {"sub_queries": sub_queries}

    # ── Node: Reformulate ────────────────────────────────────

    def _node_reformulate(self, state: RAGState) -> dict:
        """Reformulate query based on verification feedback to improve retrieval."""
        sq = state["sub_queries"][state["current_sq_idx"]]
        
        messages = [
            {"role": "system", "content": REFORMULATION_SYSTEM},
            {"role": "user", "content": REFORMULATION_USER.format(
                query=sq.query,
                original_query=sq.original_query or sq.query,
                history=", ".join(sq.reformulation_history) if sq.reformulation_history else "None",
                reason=sq.verification_reason,
                missing=sq.verification_missing,
            )},
        ]
        
        parsed = self.llm.generate_json(messages, max_tokens=256)
        
        # Fail gracefully if LLM returns unparseable JSON
        if parsed is None or "reformulated_query" not in parsed:
            logger.warning(
                f"Reformulation failed for sub-query '{sq.query}'. "
                "Using original query for retry."
            )
            reformulated = sq.original_query or sq.query
        else:
            reformulated = parsed["reformulated_query"]
            logger.info(
                f"Reformulated query: '{sq.query}' → '{reformulated}'"
            )
        
        # Update sub-query with reformulated text
        sub_queries = list(state["sub_queries"])
        idx = state["current_sq_idx"]
        sub_queries[idx].reformulation_history.append(sub_queries[idx].query)
        sub_queries[idx].query = reformulated
        
        return {"sub_queries": sub_queries}

    # ── Node: Decide Next ────────────────────────────────────

    def _node_decide_next(self, state: RAGState) -> dict:
        """
        Route to next action: reformulate and retry, process next sub-query, or generate.

        Handles self-correction: if verification fails and retries remain,
        reformulate the query and loop back to retrieve.
        """
        sq_idx = state["current_sq_idx"]
        sub_queries = state["sub_queries"]
        current_sq = sub_queries[sq_idx]
        has_results = len(state["current_results"]) > 0
        verified = current_sq.verified
        retry_count = state["retry_count"]

        # Case 1: Not verified (or no results), retries remaining then reformulate and retry
        if (not has_results or not verified) and retry_count < self.max_retries:
            logger.info(
                f"Verification failed (score={current_sq.verification_score:.2f}). "
                f"Triggering reformulation (retry {retry_count + 1}/{self.max_retries})"
            )
            return {
                "retry_count": retry_count + 1,
                "total_retries": state["total_retries"] + 1,
                "current_results": [],
                "next_action": "reformulate",
            }

        # Case 2: Verified or retries exhausted then collect results and move on
        all_results = list(state["all_results"])
        if has_results:
            all_results.extend(state["current_results"])
            current_sq.results = state["current_results"]
        current_sq.retry_count = retry_count

        next_idx = sq_idx + 1

        # Case 2a: More sub-queries then set up next sub-query and retrieve
        if next_idx < len(sub_queries):
            next_sq = sub_queries[next_idx]
            return {
                "all_results": all_results,
                "current_sq_idx": next_idx,
                "current_company": next_sq.company,
                "current_doc_type": next_sq.document_type,
                "current_limit": self.retrieval_limit,
                "retry_count": 0,
                "current_results": [],
                "next_action": "retrieve",
            }

        # Case 2b: All sub-queries done then generate
        return {
            "all_results": all_results,
            "current_results": [],
            "next_action": "generate",
        }

    # ── Node: Generate ───────────────────────────────────────

    def _node_generate(self, state: RAGState) -> dict:
        """Generate grounded answer with citations from all collected results."""
        user_query = state["user_query"]
        all_results = state["all_results"]

        # No results then return fallback message
        if not all_results:
            logger.warning("No results to generate from")
            return {
                "answer": "I could not find relevant information in the SEC filings to answer this question. "
                          "Try rephrasing or specifying the company ticker and filing type.",
                "sources": [],
            }

        # Deduplicate by chunk_id, keeping highest score
        seen = {}
        for r in all_results:
            if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                seen[r.chunk_id] = r
        unique_results = sorted(seen.values(), key=lambda r: r.score, reverse=True)

        # Generate answer
        sources_text = format_sources_for_prompt(unique_results)
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM},
            {"role": "user", "content": GENERATION_USER.format(
                query=user_query, sources=sources_text,
            )},
        ]
        answer = self.generation_llm.generate(messages)

        # Build sources list
        sources = [
            Source(
                source_number=i + 1,
                company=r.metadata.get("company", ""),
                document_type=r.metadata.get("document_type", ""),
                filing_date=r.metadata.get("filing_date", ""),
                page_number=r.metadata.get("page_number"),
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
            )
            for i, r in enumerate(unique_results)
        ]
        if self.verbose:
            logger.info(f"Generated answer: {len(answer)} chars, {len(sources)} sources")
        return {"answer": answer, "sources": sources}



