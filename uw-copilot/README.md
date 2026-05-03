# Underwriting Copilot — RAG Knowledge System

> **Portfolio Project** — Senior Applied AI Engineer · Insurance Domain  
> Demonstrates enterprise-grade Retrieval-Augmented Generation for underwriting knowledge queries

---

## Overview

A production RAG system that lets underwriters query their internal knowledge base — underwriting guidelines, coverage manuals, rate filings, and loss run reports — in natural language. Every answer is **grounded in retrieved source documents** with **inline citations** pointing to the exact page and section. Unanswerable questions are clearly flagged rather than hallucinated.

**Live Demo**: Open `frontend/index.html` in a browser — no server required.

---

## Architecture

```
Underwriter Query (natural language)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              UNDERWRITING COPILOT RAG PIPELINE              │
│                                                             │
│  1. EMBED QUERY                                             │
│     Azure OpenAI text-embedding-3-large (3072-dim)         │
│                                                             │
│  2. HYBRID RETRIEVAL  ┌─────────────────────┐              │
│     Azure AI Search   │ BM25 full-text       │              │
│     (HNSW index)      │ + Vector HNSW        │ top-K=15     │
│                       └─────────────────────┘              │
│                                                             │
│  3. CROSS-ENCODER RERANK                                    │
│     Cohere Rerank v3 → top-5 most relevant chunks          │
│                                                             │
│  4. GENERATE (GPT-4o)                                       │
│     Citation-aware prompt · JSON structured output         │
│     Confidence scoring · Gap detection                      │
│                                                             │
│  5. CITATION ASSEMBLY                                       │
│     Map [N] markers → source doc, section, page, excerpt   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Grounded Answer + Citations + Confidence + Disclaimer
```

### Two-Stage Retrieval (Why it matters)

**Stage 1 — Dense + Sparse Hybrid Retrieval**  
BM25 catches exact keyword matches (policy numbers, NCCI codes, specific dollar limits). Vector search catches semantic matches (flood → water damage exclusion). Azure AI Search combines both with Reciprocal Rank Fusion for best recall.

**Stage 2 — Cross-Encoder Reranking**  
The vector retriever optimizes recall (broad net). The cross-encoder optimizes precision (right documents at the top). Cohere Rerank v3 reads the full query and each candidate chunk together to score true relevance. This two-stage approach significantly outperforms vector-only retrieval on underwriting content.

### Chunking Strategy

- **512 token chunks** with 51-token overlap (10%) to preserve context at boundaries
- **Section-header detection** keeps section context in chunk metadata
- **Semantic chunking** in production (LlamaIndex `SemanticSplitterNodeParser`) finds natural breaks using embedding similarity rather than fixed windows
- **Metadata preserved**: doc title, doc type, section name, page number → enables structured citations

---

## Document Types Supported

| Type | Examples | Relevance |
|------|----------|-----------|
| Underwriting Guideline | Appetite, TIV limits, eligibility rules | Primary underwriting decisions |
| Coverage Manual | ISO forms, coverage grants/exclusions | Coverage structure questions |
| Loss Run | Claim frequency, severity, driver analysis | Risk assessment, renewal |
| Rate Filing | Class codes, base rates, e-mod rules | Premium adequacy |
| Regulatory | State-specific compliance rules | Regulatory questions |

---

## Project Structure

```
uw-copilot/
├── backend/
│   ├── main.py                      # FastAPI app entry point
│   ├── requirements.txt
│   ├── models/
│   │   └── schemas.py               # Pydantic v2 models
│   ├── rag/
│   │   ├── ingestion.py             # Document chunking + embedding pipeline
│   │   └── retrieval_engine.py      # Retrieve → rerank → generate
│   ├── api/
│   │   └── routes.py                # REST API routes + RBAC
│   ├── eval/
│   │   └── ragas_eval.py            # RAGAS evaluation harness + CI gate
│   └── tests/
│       └── test_rag.py              # pytest suite (ingestion + retrieval + e2e)
└── frontend/
    └── index.html                   # Interactive demo UI
```

---

## Running the Demo

### Interactive UI (no dependencies)
```bash
open frontend/index.html
```

Try these queries:
- "Is flood covered under our standard commercial property policy?"
- "What are the driver tier definitions and what surcharge applies to Tier 3?"
- "What security controls must an account have before we can offer cyber coverage?"
- "What is the e-mod threshold that requires senior underwriter approval?"

### Backend API (Python 3.11+)
```bash
cd backend
pip install -r requirements.txt

# Configure Azure services
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_KEY="your-key"
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_KEY="your-search-key"
export COHERE_API_KEY="your-cohere-key"

uvicorn main:app --reload
# API docs: http://localhost:8000/docs
```

### Tests
```bash
cd backend
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Evaluation
```bash
cd backend
python eval/ragas_eval.py
# Runs 8 ground-truth questions, outputs pass/fail per metric
```

---

## API Reference

### POST `/api/v1/copilot/query`
Submit a question. Returns grounded answer with citations.

**Request**:
```json
{
  "query_text": "What is the maximum TIV for habitational risks without reinsurance?",
  "top_k": 5,
  "doc_type_filter": ["underwriting_guideline"]
}
```

**Response**:
```json
{
  "answer": "The maximum single-location TIV for habitational risks is **$50M** without reinsurance approval [1]...",
  "citations": [
    {
      "citation_id": 1,
      "doc_title": "Commercial Property Underwriting Guidelines v12.3",
      "section": "Coverage Limits & Sublimits",
      "page_number": 34,
      "excerpt": "Maximum single-location TIV for habitational risks is $50M...",
      "similarity_score": 0.89,
      "confidence": "high"
    }
  ],
  "confidence_level": "high",
  "confidence_rationale": "Direct match to TIV section with specific threshold.",
  "retrieval_latency_ms": 180,
  "generation_latency_ms": 820,
  "total_latency_ms": 1000,
  "cost_usd": 0.0074,
  "disclaimer": "This response is AI-generated... validate before binding."
}
```

### GET `/api/v1/copilot/documents`
List all indexed documents.

---

## Evaluation Framework

The eval harness runs 8 curated underwriting questions with known expected keywords and scores:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Faithfulness | ≥ 0.75 | Answer grounded in retrieved documents |
| Answer Relevance | ≥ 0.70 | Answer addresses the question |
| Citation Precision | ≥ 0.65 | Citations reference correct content |
| Hallucination Rate | ≤ 0.15 | Proportion of uncited claims |

Evaluation fails the CI gate if any threshold is breached, blocking deployment.

---

## Production Deployment (Azure)

```
Azure API Management         → Auth, rate limiting, role enforcement
Azure Container Apps         → FastAPI service (auto-scaling)
Azure OpenAI Service         → GPT-4o (generation) + text-embedding-3-large
Azure AI Search              → Hybrid BM25 + HNSW vector index
Cohere API                   → Rerank v3 cross-encoder
Azure Blob Storage           → Raw document storage (encrypted at rest)
Azure Key Vault              → API key management
Azure Functions              → Async document ingestion trigger
Azure Monitor                → Latency, cost, and quality telemetry
GitHub Actions               → CI/CD with RAGAS eval gate
```

**Cost at scale**: ~$0.006–$0.012 per query at GPT-4o pricing. Embedding is one-time per document at ~$0.0001 per 1K tokens.

---

## Design Decisions

**Why two-stage retrieval over vector-only?**  
Underwriting guidelines use precise numerical thresholds ($50M, 1.25 e-mod) and specific terms (FEMA Zone AE, ISO CP 10 30) that benefit from BM25 keyword matching. The cross-encoder reranker then ensures the most contextually relevant chunk surfaces first — reducing hallucination risk.

**Why GPT-4o over smaller models?**  
Insurance answers must be precise. Smaller models are more likely to conflate similar-sounding rules (e.g., mixing up Tier 2 and Tier 3 thresholds). GPT-4o's instruction-following is necessary for reliable JSON-structured output with correct citation numbering.

**Why explicit confidence scoring?**  
Underwriters need to know when to trust the AI. Low confidence → manual verification. The system is more valuable when it honestly flags uncertainty than when it confidently hallucinates.

**Why include a disclaimer on every response?**  
Regulatory requirement for AI-assisted underwriting decisions in enterprise deployments. The copilot is a tool, not a decision-maker.

---

*Built to demonstrate Senior Applied AI Engineer capabilities for enterprise insurance AI.*
