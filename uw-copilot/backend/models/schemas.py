"""
Data models for Underwriting Copilot RAG system.
All models use Pydantic v2.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum
import uuid


class DocumentType(str, Enum):
    UNDERWRITING_GUIDELINE = "underwriting_guideline"
    LOSS_RUN = "loss_run"
    COVERAGE_MANUAL = "coverage_manual"
    RATE_FILING = "rate_filing"
    REGULATORY = "regulatory"
    POLICY_FORM = "policy_form"
    REINSURANCE = "reinsurance"


class DocumentChunk(BaseModel):
    """A single indexed chunk from a source document."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    doc_title: str
    doc_type: DocumentType
    page_number: Optional[int] = None
    section: Optional[str] = None
    content: str
    token_count: int
    embedding_model: str = "text-embedding-3-large"
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


class RetrievedChunk(BaseModel):
    """A chunk returned from vector search, with relevance score."""
    chunk: DocumentChunk
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rerank_score: Optional[float] = None  # Cross-encoder rerank score


class Citation(BaseModel):
    """Inline citation linking answer text to source chunk."""
    citation_id: int
    doc_title: str
    doc_type: DocumentType
    section: Optional[str] = None
    page_number: Optional[int] = None
    excerpt: str  # Relevant passage from source (non-PII)
    similarity_score: float
    confidence: Literal["high", "medium", "low"]


class CopilotQuery(BaseModel):
    """Underwriter's question to the copilot."""
    query_text: str = Field(..., min_length=5, description="Natural language question")
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_type_filter: Optional[List[DocumentType]] = None  # Restrict to specific doc types
    top_k: int = Field(default=5, ge=1, le=20)
    user_role: str = "underwriter"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CopilotResponse(BaseModel):
    """Full RAG response with grounded answer and citations."""
    query_id: str
    query_text: str
    answer: str
    citations: List[Citation]
    confidence_level: Literal["high", "medium", "low"]
    confidence_rationale: str
    retrieved_chunks: int
    reranked_chunks: int
    answer_tokens: int
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    model_used: str
    embedding_model: str
    cost_usd: float
    disclaimer: str = (
        "This response is AI-generated from indexed underwriting documents. "
        "Always validate against primary sources before binding coverage."
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class IndexingJob(BaseModel):
    """Document ingestion + indexing job status."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_title: str
    doc_type: DocumentType
    status: Literal["queued", "chunking", "embedding", "indexing", "complete", "failed"]
    chunks_created: int = 0
    chunks_indexed: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class EvalResult(BaseModel):
    """Single RAG evaluation result for a test question."""
    question: str
    expected_answer_keywords: List[str]
    actual_answer: str
    citations_count: int
    answer_faithfulness: float  # 0-1, grounded in retrieved docs
    answer_relevance: float     # 0-1, answers the question
    citation_precision: float   # Correct citations / total citations
    latency_ms: int
    passed: bool


class EvalReport(BaseModel):
    """Aggregate evaluation report across test suite."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_questions: int
    passed: int
    avg_faithfulness: float
    avg_relevance: float
    avg_citation_precision: float
    avg_latency_ms: float
    hallucination_rate: float
    run_at: datetime = Field(default_factory=datetime.utcnow)
    results: List[EvalResult]
