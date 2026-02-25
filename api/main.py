"""
FastAPI backend for Finance RAG.

Exposes a single streaming endpoint:
  POST /query/stream  →  Server-Sent Events (SSE)

SSE event shapes:
  {"type": "status",  "message": "..."}
  {"type": "token",   "content": "..."}
  {"type": "sources", "sources": [...]}
  {"type": "error",   "message": "..."}
"""
import asyncio
import json
import queue as stdlib_queue
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── Path bootstrap ────────────────────────────────────────────────────────────
# api/ lives one level below the project root; src.* imports need the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.rag_agent import RAGAgent  # noqa: E402


# ── Lifespan: load heavy models once at startup ───────────────────────────────
_agent: Optional[RAGAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = RAGAgent()
    yield
    # nothing to teardown


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Finance RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response shapes ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    document_type: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sse(payload: dict) -> dict:
    """Wrap a dict as an SSE data event."""
    return {"data": json.dumps(payload)}


def _serialize_sources(sources) -> list[dict]:
    return [
        {
            "source_number": s.source_number,
            "company": s.company,
            "document_type": s.document_type,
            "filing_date": s.filing_date,
            "page_number": s.page_number,
            "score": round(s.score, 4),
        }
        for s in sources
    ]


# ── Streaming endpoint ────────────────────────────────────────────────────────
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream a RAG query response as Server-Sent Events.

    The event sequence is:
      1. status  – "Analyzing your query…"
      2. token   – answer words, one per event (or small chunks)
      3. sources – final JSON payload with cited sources
    """

    async def generate():
        # Queue bridges the blocking pipeline thread and this async generator.
        # Items are either:
        #   dict          → SSE event payload (status / token)
        #   RAGResponse   → pipeline finished successfully
        #   Exception     → pipeline raised an error
        event_q: stdlib_queue.SimpleQueue = stdlib_queue.SimpleQueue()

        def token_cb(chunk: str) -> None:
            event_q.put({"type": "token", "content": chunk})

        def status_cb(message: str) -> None:
            event_q.put({"type": "status", "message": message})

        def run_pipeline() -> None:
            try:
                result = _agent.query(
                    request.query,
                    token_callback=token_cb,
                    status_callback=status_cb,
                )
                event_q.put(result)
            except Exception as exc:
                event_q.put(exc)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, run_pipeline)

        # Drain the queue until we get the terminal RAGResponse or an Exception.
        while True:
            item = await loop.run_in_executor(None, event_q.get)

            if isinstance(item, dict):
                # Regular SSE event (status or token)
                yield _sse(item)
            elif isinstance(item, Exception):
                yield _sse({"type": "error", "message": str(item)})
                return
            else:
                # RAGResponse — pipeline is done, send sources and close.
                yield _sse(
                    {
                        "type": "sources",
                        "sources": _serialize_sources(item.sources),
                    }
                )
                return

    return EventSourceResponse(generate())


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}
