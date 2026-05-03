# Insurance AI Engineering Portfolio
### Senior Applied AI Engineer · Liberty Mutual Insurance

> Four production-grade AI systems demonstrating enterprise AI engineering across the full insurance technology stack — from claim intake to knowledge retrieval, governance, and systematic evaluation.

---

## Portfolio Overview

This portfolio was built to demonstrate the exact capabilities outlined in the **Senior Applied AI Engineer** job description at Liberty Mutual Insurance. Each project maps directly to a stated requirement.

| Project | Domain | JD Requirement |
|---------|--------|----------------|
| [Claims Triage & Fraud Signal Agent](#1-claims-triage--fraud-signal-agent) | Claims | Agentic workflow · LLM/ML hybrid · Audit logging |
| [Underwriting Copilot (RAG)](#2-underwriting-copilot--rag-knowledge-system) | Underwriting | Retrieval/knowledge grounding · Citation UI · Evaluation |
| [Governed AI Gateway](#3-governed-ai-gateway) | Operations | Access control · PII handling · Audit logs · HITL |
| [LLM Eval & Regression Framework](#4-llm-evaluation--regression-framework) | All domains | Evaluation/quality methods · CI/CD gate · Model governance |

**Tech stack across projects:**  
Python · FastAPI · Azure OpenAI · LangGraph · LlamaIndex · Azure AI Search · Azure Cosmos DB · Microsoft Presidio · XGBoost · RAGAS · Pydantic v2 · pytest · GitHub Actions · Terraform (Azure)

---

## Quick Start — Run All Projects

> **Prerequisites:** Python 3.11+, pip, a modern browser.  
> All four demos run **without API keys** using built-in simulation.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/insurance-ai-portfolio.git
cd insurance-ai-portfolio

# 2. Run the interactive launcher (guides you through each project)
python run_all.py

# ── OR run each project individually ──────────────────────────────────

# Open any frontend demo instantly (no server needed)
open claims-triage-agent/frontend/index.html
open uw-copilot/frontend/index.html
open ai-gateway/frontend/index.html
open llm-eval-framework/frontend/index.html

# Start any backend API
cd claims-triage-agent/backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8001
cd uw-copilot/backend         && pip install -r requirements.txt && uvicorn main:app --reload --port 8002
cd ai-gateway/backend         && pip install -r requirements.txt && uvicorn main:app --reload --port 8003
cd llm-eval-framework/backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8004
```

**API docs** (when backends are running):
- Claims Triage: http://localhost:8001/docs
- UW Copilot:    http://localhost:8002/docs
- AI Gateway:    http://localhost:8003/docs
- Eval Framework: http://localhost:8004/docs

---

## Projects

### 1. Claims Triage & Fraud Signal Agent

**What it does:** End-to-end agentic pipeline that processes a raw insurance claim through 7 stages: PII redaction → LLM entity extraction → rules-based fraud scoring → ML fraud scoring (XGBoost) → LLM contextual analysis → fraud aggregation → adjuster routing. Every stage is independently auditable.

**Demo:** Open `claims-triage-agent/frontend/index.html` → load any of the 4 pre-built scenarios (Low Risk through Critical) → watch the pipeline animate → inspect fraud signals, routing decision, analyst narrative, and PII-redacted audit trail.

**Key engineering decisions:**
- PII redaction (Microsoft Presidio) runs before any text reaches an LLM — hard architectural rule
- Three-signal fraud scoring (rules 30% + XGBoost 40% + GPT-4o 30%) prevents single-model dependence  
- Human-in-the-loop escalation is mandatory for High/Critical risk claims
- Audit trail uses SHA-256 hashing of inputs — raw PII never logged

**Files:** `claims-triage-agent/`
```
backend/
  agents/claims_pipeline.py   # LangGraph agentic orchestrator (7 stages)
  models/schemas.py            # Pydantic v2 — FraudSignal, RoutingDecision, AuditEntry
  services/llm_client.py       # Azure OpenAI integration + structured output
  services/fraud_rules.py      # Versioned rules engine (v3.2.1)
  services/ml_scorer.py        # XGBoost + SHAP explainability
  services/pii_redactor.py     # Presidio PII redaction
  api/routes.py                # FastAPI + RBAC enforcement
  tests/test_pipeline.py       # 20 pytest tests
frontend/index.html            # Dark operator dashboard UI
```

**Run the backend:**
```bash
cd claims-triage-agent/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
# POST http://localhost:8001/api/v1/claims/process
```

---

### 2. Underwriting Copilot — RAG Knowledge System

**What it does:** Retrieval-Augmented Generation system that lets underwriters query indexed guidelines, coverage manuals, and loss runs in natural language. Every answer is grounded in retrieved source documents with inline citations `[1]`, `[2]` pointing to the exact document, section, and page. Unanswerable questions are explicitly flagged.

**Demo:** Open `uw-copilot/frontend/index.html` → click any sample query (flood coverage, e-mod thresholds, cyber controls, driver tiers) → watch the 5-stage pipeline animate → read the grounded answer with citations, confidence level, retrieval metadata, and compliance disclaimer.

**Key engineering decisions:**
- Two-stage retrieval: BM25 + vector (Azure AI Search) → cross-encoder rerank (Cohere Rerank v3) — significantly outperforms vector-only on insurance content
- 512-token chunks with 10% overlap and section-header detection preserves context
- Confidence scoring is explicit — underwriters know when to verify manually
- Compliance disclaimer on every response (required for AI-assisted underwriting)

**Files:** `uw-copilot/`
```
backend/
  rag/retrieval_engine.py     # Hybrid retrieval → rerank → generate pipeline
  rag/ingestion.py             # Document chunking (512 tokens, 10% overlap)
  eval/ragas_eval.py           # RAGAS evaluation harness + CI gate
  models/schemas.py            # CopilotQuery, CopilotResponse, Citation
  api/routes.py                # Query endpoint + document management
  tests/test_rag.py            # 18 pytest tests
frontend/index.html            # Editorial light-theme research UI
```

**Run the backend:**
```bash
cd uw-copilot/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8002
# POST http://localhost:8002/api/v1/copilot/query
```

---

### 3. Governed AI Gateway

**What it does:** A thin API gateway between all internal teams and LLM providers. Every LLM call in the enterprise routes through this gateway, which enforces: JWT authentication (Azure AD), role-based model access control, bidirectional PII redaction, content policy, rate limiting, cost tracking, and an immutable PII-free audit trail.

**Demo:** Open `ai-gateway/frontend/index.html` → navigate to **Request Simulator** → try the 4 scenarios (RBAC Deny, PII Injection, Blocked Content, Clean Request) → watch the policy engine evaluate each rule live → switch to Dashboard, Audit Log, RBAC Matrix, Policy Rules, Team Usage.

**Key engineering decisions:**
- Bidirectional PII scanning — both prompt-in AND completion-out are scanned (LLMs can reconstruct PII from context)
- User identity stored as SHA-256 hash in audit log — never raw identity (GDPR/GLBA compliance)
- Policy engine follows OPA pattern — each rule is an isolated method, ready for Rego migration
- `X-Gateway-*` response headers carry governance metadata to downstream consumers

**Files:** `ai-gateway/`
```
backend/
  gateway/core.py              # 7-stage request lifecycle orchestrator
  middleware/auth.py           # JWT verification + RBAC model allowlist
  middleware/pii.py            # Bidirectional Presidio redaction
  middleware/policy.py         # 6-rule policy engine (RBAC, rate limit, content, PII, anomaly)
  services/audit.py            # Append-only audit logger + cost tracker
  models/schemas.py            # AuthenticatedCaller, AuditEvent, GatewayResponse
  api/routes.py                # Gateway, audit, usage, dashboard endpoints
  tests/test_gateway.py        # 35 pytest tests
infra/main_tf.py               # Terraform IaC (Container Apps, Cosmos DB, Redis)
frontend/index.html            # Control room dashboard UI
```

**Run the backend:**
```bash
cd ai-gateway/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8003

# Issue a demo token, then call the gateway:
curl -X POST "http://localhost:8003/api/v1/auth/token?user_id=sarah&role=underwriter&team=UW&display_name=Sarah"
curl -X POST "http://localhost:8003/api/v1/gateway/chat" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"What is the max TIV?"}]}'
```

---

### 4. LLM Evaluation & Regression Framework

**What it does:** A production evaluation harness for insurance AI. Every prompt change or model upgrade runs through 22 expert-curated test cases across 7 insurance domains. Eight metrics are evaluated (LLM-as-judge + rule-based). A CI gate blocks deployment if any metric drops below threshold vs the baseline run.

**Demo:** Open `llm-eval-framework/frontend/index.html` → explore 10 seeded historical runs → **Model Compare** to rank GPT-4o vs Claude vs GPT-3.5 → **Regression Report** for current vs baseline delta → **Run Eval** to trigger a new run with live progress → **Metric Trends** for any metric charted over time.

**Key engineering decisions:**
- PII leakage has 99% gate threshold — any output PII is a regulatory incident
- Two evaluator families: LLM judge (semantic accuracy) + rule-based (deterministic, free, instant)
- Regression threshold blocks deployment if factual_accuracy drops >3pp or PII leakage drops >1pp
- 22 cases chosen for quality, not quantity — authored by domain experts, versioned in DVC

**Files:** `llm-eval-framework/`
```
backend/
  evaluators/metrics.py        # 8 evaluators: LLM judge + rule-based
  runners/eval_runner.py       # Async orchestrator (asyncio.gather + semaphore)
  runners/regression.py        # Baseline comparison + deploy/block recommendation
  datasets/insurance_cases.py  # 22 ground-truth cases across 7 domains
  storage/run_store.py         # Run persistence (pre-seeded with 10 runs)
  models/schemas.py            # EvalCase, CaseResult, RegressionReport
  tests/test_eval_framework.py # 45 pytest tests
ci/eval.yml                    # GitHub Actions: smoke eval on PR, full regression on merge
frontend/index.html            # Scientific warm-theme dashboard
```

**Run the backend:**
```bash
cd llm-eval-framework/backend
pip install -r requirements.txt
uvicorn api.routes:app --reload --port 8004
# POST http://localhost:8004/api/v1/runs  (trigger eval)
# GET  http://localhost:8004/api/v1/runs  (list runs)
```

---

## Testing All Projects

Run all test suites from the repo root:

```bash
# Install test deps for all projects
pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests
pytest claims-triage-agent/backend/tests/ \
       uw-copilot/backend/tests/ \
       ai-gateway/backend/tests/ \
       llm-eval-framework/backend/tests/ \
       -v --tb=short

# Individual project test coverage
cd claims-triage-agent/backend  && pytest tests/ -v --cov=. --cov-report=term-missing
cd uw-copilot/backend           && pytest tests/ -v --cov=. --cov-report=term-missing
cd ai-gateway/backend           && pytest tests/ -v --cov=. --cov-report=term-missing
cd llm-eval-framework/backend   && pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Connecting Azure OpenAI (Optional)

The demos run fully without API keys using simulation. To use real Azure OpenAI:

```bash
# Create a .env file in each project's backend/ directory
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# For the AI Gateway demo token (dev only):
GATEWAY_JWT_SECRET=your-secret-key

# For the UW Copilot (optional reranking):
COHERE_API_KEY=your-cohere-key

# For the AI Gateway (production storage):
COSMOS_CONNECTION_STRING=AccountEndpoint=...
REDIS_CONNECTION_STRING=your-redis.redis.cache.windows.net:6380,...
```

---

## Architecture Across All Projects

```
                     ┌─────────────────────────────────────┐
                     │         AI GATEWAY (Project 3)       │
                     │  Auth · PII · Policy · Audit · Cost  │
                     └──────────────┬──────────────────────┘
                                    │ All LLM calls route through
                  ┌─────────────────┼──────────────────┐
                  │                 │                  │
         ┌────────▼───────┐ ┌───────▼──────┐ ┌────────▼──────┐
         │  Claims Triage  │ │  UW Copilot  │ │   (future     │
         │  Agent (Proj 1) │ │  RAG (Proj 2)│ │   projects)   │
         └────────┬────────┘ └──────┬───────┘ └───────────────┘
                  │                 │
                  └────────┬────────┘
                           │
              ┌────────────▼───────────────┐
              │   LLM Eval Framework (Proj 4)│
              │   Evaluates quality of ALL   │
              │   projects above on every    │
              │   model / prompt change      │
              └──────────────────────────────┘
```

---

## Contact

Built by Zahidur Rahman — mdr391@gmail.com — www.linkedin.com/in/zahidur-rahman-mdr391

*Targeting: Senior Applied AI Engineer *
