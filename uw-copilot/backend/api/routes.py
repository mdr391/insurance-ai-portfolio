"""
API Routes — Underwriting Copilot.
"""

from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Optional
import logging

from models.schemas import CopilotQuery, CopilotResponse, DocumentType
from rag.retrieval_engine import UnderwritingRAGEngine

logger = logging.getLogger(__name__)

copilot_router = APIRouter()
docs_router = APIRouter()
health_router = APIRouter()

_engine = UnderwritingRAGEngine()

ALLOWED_ROLES = {"underwriter", "senior_underwriter", "actuary", "supervisor", "system"}


def verify_role(x_user_role: Optional[str] = Header(None)) -> str:
    role = x_user_role or "underwriter"
    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=403, detail=f"Role '{role}' not authorized")
    return role


@health_router.get("")
async def health():
    return {"status": "healthy", "service": "underwriting-copilot", "version": "1.0.0"}


@copilot_router.post("/query", response_model=CopilotResponse)
async def query_copilot(
    query: CopilotQuery,
    role: str = Depends(verify_role),
):
    """
    Submit a natural-language question to the Underwriting Copilot.

    The RAG pipeline:
    1. Embeds the query (text-embedding-3-large)
    2. Retrieves top-K relevant chunks (Azure AI Search hybrid)
    3. Reranks with cross-encoder (Cohere Rerank v3)
    4. Generates grounded answer with inline citations (GPT-4o)

    Returns: answer, citations with source attribution, confidence level,
             retrieval metadata, and cost tracking.
    """
    logger.info(f"Query received | role={role} | q='{query.query_text[:80]}...'")
    try:
        result = await _engine.query(query)
        return result
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@copilot_router.get("/documents")
async def list_documents(role: str = Depends(verify_role)):
    """List all indexed documents with metadata."""
    seen = set()
    docs = []
    for chunk in _engine._chunks:
        if chunk.doc_id not in seen:
            seen.add(chunk.doc_id)
            docs.append({
                "doc_id": chunk.doc_id,
                "title": chunk.doc_title,
                "type": chunk.doc_type,
            })
    return {"documents": docs, "total": len(docs)}


@copilot_router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str, role: str = Depends(verify_role)):
    """Get all indexed chunks for a specific document."""
    chunks = [c for c in _engine._chunks if c.doc_id == doc_id]
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "chunks": len(chunks), "sections": list(set(c.section for c in chunks))}
