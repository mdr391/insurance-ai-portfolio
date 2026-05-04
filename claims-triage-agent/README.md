# Claims Triage & Fraud Signal Agent

![CI](https://github.com/mdr391/insurance-ai-portfolio/actions/workflows/ci.yml/badge.svg)

> **Portfolio Project** — Senior Applied AI Engineer · Insurance Domain  
> Demonstrates production-grade agentic AI for enterprise insurance use cases

---

## Overview

A fully agentic claims processing pipeline built for enterprise insurance operations. Ingests raw FNOL claim submissions, extracts structured data via LLM, scores fraud risk across three signal sources (rules + ML + LLM), and routes claims to the appropriate adjuster queue — with complete PII-redacted audit trail and human-in-the-loop escalation.

**Live Demo**: Open `frontend/index.html` in a browser — no server required.

---

## Architecture

```
Claim Submission (REST API)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLAIMS TRIAGE AGENT                          │
│                   (LangGraph StateGraph)                        │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │   PII    │──▶│ Entity   │──▶│  Rules   │──▶│  ML      │    │
│  │ Redact   │   │ Extract  │   │ Engine   │   │ Scorer   │    │
│  │(Presidio)│   │ (GPT-4o) │   │(v3.2.1)  │   │(XGBoost) │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                    │            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐       │            │
│  │ Routing  │◀──│Aggregate │◀──│LLM Fraud │◀──────┘            │
│  │Decision  │   │  Fraud   │   │ Analysis │                    │
│  └──────────┘   └──────────┘   │ (GPT-4o) │                    │
│        │                       └──────────┘                    │
│        │         Audit Log (every stage)                       │
└────────┼────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐         ┌─────────────────┐
  │ Adjuster    │────────▶│ HITL Escalation │ (if required)
  │ Queue       │         │ (Supervisor UI) │
  └─────────────┘         └─────────────────┘
```

### Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 1. Ingestion | `PIIRedactor` (Presidio) | Redact PII before any LLM call |
| 2. Extraction | Azure OpenAI GPT-4o | Extract structured entities from redacted text |
| 3. Rules | `FraudRulesEngine` | 15+ deterministic fraud signal rules |
| 4. ML Scoring | XGBoost (Azure ML) | 47-feature gradient boosting fraud model |
| 5. LLM Analysis | Azure OpenAI GPT-4o | Contextual fraud reasoning over all signals |
| 6. Aggregation | Weighted ensemble | Rules 30% + ML 40% + LLM 30% |
| 7. Routing | `RoutingEngine` | Queue assignment with explainable rationale |

---

## Key Engineering Decisions

### Governance & Compliance

**PII Redaction First** — All LLM calls operate on PII-redacted text only. Microsoft Presidio with custom insurance-domain recognizers handles names, policy numbers, VINs, SSNs, and addresses before text enters the LLM pipeline. This satisfies GLBA and enterprise data handling requirements.

**Immutable Audit Log** — Every pipeline stage writes an `AuditEntry` with SHA-256 hash of inputs, actor identity, latency, model/version used, and token cost. Entries are append-only (Cosmos DB with server-side timestamps in production). No PII in logs.

**Human-in-the-Loop** — High and Critical risk claims require supervisor/fraud investigator sign-off before any routing is finalized. The HITL endpoint enforces role-based access (supervisor or fraud_investigator roles only).

**Explainable Routing** — Every routing decision includes a plain-English `routing_rationale` combining fraud score, complexity assessment, and key signal summary. Adjusters can see exactly why a claim landed in their queue.

### Signal Architecture

Three independent scoring sources prevent any single model from being the sole gatekeeper:

- **Rules Engine** — deterministic, fully auditable, fast (<10ms). Version-controlled rule configs loaded from YAML.
- **ML Model** — XGBoost trained on historical claims. SHAP values provide per-feature attribution. Served via Azure ML endpoint.
- **LLM Analysis** — GPT-4o contextual reasoning catches narrative anomalies rules and ML miss. Structured output (JSON mode) ensures parseable responses.

Weighted composite: `0.30 × rules + 0.40 × ML + 0.30 × LLM`

### Cost Control

- PII redaction runs locally (no LLM cost)
- Rules engine runs locally
- ML inference is cheap (<$0.001/claim)
- Only 2 LLM calls per claim: extraction + fraud analysis
- Average cost: ~$0.008-0.015 per claim at GPT-4o pricing

---

## Project Structure

```
claims-triage-agent/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── requirements.txt
│   ├── agents/
│   │   └── claims_pipeline.py     # LangGraph agentic pipeline
│   ├── api/
│   │   └── routes.py              # REST endpoints + RBAC
│   ├── models/
│   │   └── schemas.py             # Pydantic v2 data models
│   ├── services/
│   │   ├── llm_client.py          # Azure OpenAI integration
│   │   ├── pii_redactor.py        # Presidio PII redaction
│   │   ├── fraud_rules.py         # Rules engine (versioned)
│   │   └── ml_scorer.py           # XGBoost + SHAP
│   ├── utils/
│   │   └── logging_config.py
│   └── tests/
│       └── test_pipeline.py       # pytest suite
└── frontend/
    └── index.html                 # Interactive demo UI
```

---

## Running the Demo

### Interactive UI (no dependencies)
```bash
open frontend/index.html
```

### Backend API (requires Python 3.11+)
```bash
cd backend
pip install -r requirements.txt
# Configure Azure OpenAI (optional — demo runs without it)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
uvicorn main:app --reload
# API docs: http://localhost:8000/docs
```

### Tests
```bash
cd backend
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## API Reference

### POST `/api/v1/claims/process`
Submit a claim for triage. Returns full `ClaimProcessingResult`.

**Headers**: `X-User-Role: adjuster` (or underwriter, fraud_investigator, supervisor)

**Request Body**:
```json
{
  "claim_text": "My vehicle was rear-ended...",
  "policy_number": "LM-4419823",
  "claimant_name": "Jane Smith",
  "incident_date": "2024-05-15",
  "claim_type": "auto"
}
```

**Response**: Full result with extracted data, fraud assessment (risk level, signals, narrative), routing decision, and audit trail.

### POST `/api/v1/claims/{claim_id}/hitl-review`
Submit human override. **Requires supervisor or fraud_investigator role.**

### GET `/api/v1/audit/{claim_id}`
Retrieve audit trail. **Requires supervisor/fraud_investigator/system role.**

---

## Production Deployment (Azure)

```
Azure API Management         → Rate limiting, auth, API versioning
Azure Container Apps         → FastAPI service (auto-scaling)
Azure OpenAI Service         → GPT-4o endpoint (PTU reserved)
Azure ML                     → XGBoost model serving endpoint
Azure Cosmos DB              → Append-only audit log
Azure Blob Storage           → Document storage (encrypted)
Azure Key Vault              → API keys, secrets
Azure Monitor + App Insights → Observability, latency tracking
Microsoft Presidio           → PII detection/anonymization
```

---

## Evaluation & Quality

The pipeline includes eval hooks for:
- **Extraction accuracy** — ground-truth comparison on labeled test claims
- **Fraud signal precision/recall** — against SIU-confirmed fraud cases
- **Routing agreement rate** — vs. senior adjuster decisions
- **Hallucination rate** — LLM outputs grounded in claim text only

---

*Built to demonstrate Senior Applied AI Engineer capabilities for enterprise insurance AI.*
