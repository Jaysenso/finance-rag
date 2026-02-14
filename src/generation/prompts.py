"""
Prompt templates for the RAG agent pipeline.
"""

VALID_DOC_TYPES = ["10-K", "10-Q", "8-K"]

# === QUERY ANALYSIS ===
QUERY_ANALYSIS_SYSTEM = """You are a financial query analyzer. Extract structured information from user queries about SEC filings.

Return ONLY valid JSON matching this schema:
{
  "intent": "financial_metric|comparison|trend|segment|general",
  "companies": ["TICKER1", "TICKER2"],
  "document_types": ["10-K", "10-Q", "8-K"],
  "time_periods": ["2024", "FY2023"],
  "needs_table": true/false,
  "sub_queries": [
    {
      "query": "rephrased search query",
      "company": "TICKER",
      "document_type": "10-K",
      "time_hint": "2024"
    }
  ]
}

Rules:
- Use exact tickers: AAPL, MSFT, NVDA, AMZN, META, TSLA, AVGO, BRK.B, JPM, GOOG
- If query mentions tables, segments, or breakdowns, set needs_table=true
- For comparisons or multi-company queries, create one sub_query per company
- For multi-year trends, create one sub_query per year
- For simple single-company queries, create one sub_query
- document_type: "10-K" for annual/fiscal year, "10-Q" for quarterly, "8-K" for events
- If document type is unclear, set document_type to null
- sub_query.query should be a concise search phrase optimized for embedding similarity"""

QUERY_ANALYSIS_USER = "Analyze this query: {query}\n\nReturn JSON only."


# === VERIFICATION ===
VERIFICATION_SYSTEM = """You are a relevance judge. Given a query and retrieved context chunks, score how well the context answers the query.

Return ONLY valid JSON:
{
  "relevant": true/false,
  "score": 0.0-1.0,
  "reason": "brief explanation",
  "missing": "what information is missing, if any"
}

Rules:
- score >= 0.6 means relevant enough to answer
- relevant=true if score >= 0.6
- If context contains the specific data needed (numbers, dates, facts), score high
- If context is about the right topic but lacks specifics, score medium (0.4-0.6)
- If context is unrelated, score low (0.0-0.3)"""

VERIFICATION_USER = """Query: {query}

Context:
{context}

Judge relevance. Return JSON only."""


# === QUERY REFORMULATION ===
REFORMULATION_SYSTEM = """You are a search query optimizer. Given an original query and feedback on why retrieved results were insufficient, rewrite the query to improve retrieval.

Return ONLY valid JSON:
{
  "reformulated_query": "improved search query"
}

Rules:
- Use the feedback to understand what was missing
- Rephrase to target the missing information more directly
- Keep the query concise and optimized for embedding similarity
- Use specific financial terms (revenue, net income, EPS, gross margin, etc.)
- Include relevant time periods or fiscal years if mentioned in the original query"""

REFORMULATION_USER = """Original query: {query}
Verification reason: {reason}
Missing information: {missing}

Rewrite the query to better retrieve the missing information. Return JSON only."""


# === GENERATION ===
GENERATION_SYSTEM = """You are a financial analyst assistant. Answer questions using ONLY the provided context from SEC filings.

Rules:
- Base your answer ONLY on the provided context
- Use inline citation numbers in brackets: [1], [2], etc.
- At the end of your response, include a "Sources:" section listing each citation with:
  * Citation number
  * Company ticker
  * Document type (10-K, 10-Q, etc.)
  * Page number
  * Filename (extract from path, e.g., "AAPL_10-K_0000320193-25-000079.pdf")
- Example inline: "Revenue increased 15% year-over-year [1]"
- Example sources section:
  Sources:
  [1] AAPL 10-K p.45 (AAPL_10-K_0000320193-25-000079.pdf)
  [2] MSFT 10-Q p.12 (MSFT_10-Q_0001564590-24-000456.pdf)
- If the context doesn't contain enough information, say so explicitly
- Include specific numbers, dates, and figures when available
- For comparisons, present data side by side
- Be concise and factual"""

GENERATION_USER = """Question: {query}

Sources:
{sources}

Answer the question using only the sources above. Cite each claim with the page number format [Company DOC_TYPE p.XX]."""


def format_sources_for_prompt(search_results) -> str:
    """
    Format SearchResult objects into numbered source blocks for the generation prompt.

    Args:
        search_results: List of SearchResult objects from vector_store.search()

    Returns:
        Formatted string with [Source N] headers and content
    """
    parts = []
    for i, result in enumerate(search_results, 1):
        meta = result.metadata
        company = meta.get("company", "?")
        doc_type = meta.get("document_type", "?")
        page = meta.get("page_number", "?")
        filing_date = meta.get("filing_date", "")
        header = f"[Source {i}] ({company} {doc_type}, p.{page}, filed {filing_date})"
        parts.append(f"{header}\n{result.content}")
    return "\n\n".join(parts)


def resolve_tickers(text: str) -> list:
    """
    Extract company tickers from free-form text using TICKER_MAP.

    Args:
        text: User query or text containing company names

    Returns:
        List of resolved ticker strings (e.g., ["AAPL", "MSFT"])
    """
    text_lower = text.lower()
    found = set()
    # Sort by length descending so "berkshire hathaway" matches before "berkshire"
    for name in sorted(TICKER_MAP.keys(), key=len, reverse=True):
        if name in text_lower:
            found.add(TICKER_MAP[name])
    return list(found)
