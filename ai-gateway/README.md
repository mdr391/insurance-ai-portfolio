# Governed AI Gateway

![CI](https://github.com/mdr391/insurance-ai-portfolio/actions/workflows/ci.yml/badge.svg)

> **Portfolio Project** — Senior Applied AI Engineer · Insurance Domain  
> Demonstrates enterprise-grade AI governance: RBAC, PII protection, audit logs, cost chargeback

---

## Overview

A production-grade API gateway that sits between all internal teams and external LLM providers (Azure OpenAI, Anthropic). Every LLM call in the enterprise routes through this gateway, which enforces authentication, role-based model access, PII redaction, content policy, rate limiting, cost tracking, and immutable audit logging — in a single, consistent control plane.

**Live Demo**: Open `frontend/index.html` — fully interactive governance dashboard. Try the Request Simulator with different roles and scenarios (PII injection, RBAC deny, content block, clean request).

---

## Architecture

```
Internal Teams                    AI Gateway                     LLM Providers
─────────────                     ──────────                     ─────────────
  Underwriting App ─┐
  Claims System    ──┤  POST /api/v1/gateway/chat
  Fraud Tool       ──┤  Authorization: Bearer <JWT>
  Data Science     ──┤
  Actuarial        ─┘
                      │
                      ▼
            ┌─────────────────────────────────────────────────────────┐
            │  1. JWT Verification (Azure AD RS256)                   │
            │     Extract: user_id, role, team, scopes                │
            ├─────────────────────────────────────────────────────────┤
            │  2. Bidirectional PII Redaction (Presidio)              │
            │     Inbound: SSN, VIN, phone, email → [REDACTED]        │
            │     Outbound: scan completion for reconstructed PII      │
            ├─────────────────────────────────────────────────────────┤
            │  3. Policy Engine (6 rules)                             │
            │     RBAC-001: model access allowlist                    │
            │     RL-002:   prompt token limit (per role)             │
            │     RL-003:   rate limit (sliding window, Redis)        │
            │     CM-001:   content moderation (block + log)          │
            │     PII-001:  redaction confirmation                    │
            │     AN-001:   anomaly detection (async escalation)      │
            │     Decision: ALLOW | DENY | REDACT_AND_ALLOW          │
            ├─────────────────────────────────────────────────────────┤
            │  4. LLM Proxy (ALLOW / REDACT_AND_ALLOW only)           │
            │     Route to Azure OpenAI or Anthropic                  │
            ├─────────────────────────────────────────────────────────┤
            │  5. Audit Log (append-only, PII-free)                   │
            │     Per-event: SHA-256(user_id), SHA-256(prompt),       │
            │     tokens, cost, latency, policy decision              │
            └─────────────────────────────────────────────────────────┘
                      │
                      ▼
            Enriched Response +
            X-Gateway-* headers (request_id, policy, cost, pii_redacted)
```

---

## Key Features

### 1. Role-Based Model Access Control (RBAC)
Every role has an explicit allowlist of permitted models. Compliance officers can only access `gpt-4o-mini`; data scientists can access all models; actuaries are restricted to cheaper models. Model upgrades require explicit policy updates.

| Role | GPT-4o | GPT-4o Mini | GPT-3.5 | Claude Sonnet | Claude Haiku |
|------|--------|-------------|---------|---------------|--------------|
| Underwriter | ✓ | ✓ | — | — | — |
| Claims Adjuster | ✓ | ✓ | — | — | — |
| Fraud Investigator | ✓ | ✓ | — | ✓ | — |
| Actuary | — | ✓ | ✓ | — | — |
| Compliance | — | ✓ | — | — | — |
| Data Scientist | ✓ | ✓ | ✓ | ✓ | ✓ |
| Admin | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2. Bidirectional PII Redaction
All prompts are scanned before reaching the LLM provider. LLM completions are scanned before delivery to the caller. Uses Microsoft Presidio with custom insurance-domain recognizers (policy numbers `LM-XXXXXXX`, VINs, claim numbers).

**Why bidirectional?** LLMs can reconstruct PII from context (e.g., given partial SSN + name → complete). Scanning the output catches this edge case.

### 3. Immutable Audit Log
Every request generates a chain of typed events. No raw PII enters the log — user identity is pseudonymized (SHA-256), prompt content is hashed. Events are append-only in Azure Cosmos DB with a 7-year TTL (insurance regulatory requirement). Enables:
- Compliance audits (who queried what, when)
- Cost attribution by team (chargeback)
- Incident investigation (fraud, data leaks)
- Capacity planning

### 4. Cost Tracking & Chargeback
Per-request cost is computed from model pricing × token counts and attributed to the caller's team. Real-time dashboard shows utilization vs. team budgets. Teams approaching budget trigger alerts.

### 5. Policy Engine
Rules are evaluated in sequence; DENY short-circuits remaining rules. The engine is designed for migration to Open Policy Agent (OPA) — rules are declarative and independently testable.

---

## Project Structure

```
ai-gateway/
├── backend/
│   ├── main.py                     # FastAPI app entry point
│   ├── requirements.txt
│   ├── gateway/
│   │   └── core.py                 # Request lifecycle orchestrator
│   ├── middleware/
│   │   ├── auth.py                 # JWT verification + RBAC enforcement
│   │   ├── pii.py                  # Bidirectional PII redaction (Presidio)
│   │   └── policy.py               # Policy engine (6 rules, OPA-ready)
│   ├── services/
│   │   └── audit.py                # Append-only audit logger + cost tracker
│   ├── models/
│   │   └── schemas.py              # Pydantic v2 data models
│   ├── api/
│   │   └── routes.py               # REST endpoints
│   └── tests/
│       └── test_gateway.py         # pytest suite (35+ tests)
├── frontend/
│   └── index.html                  # Interactive governance dashboard
└── infra/
    └── main_tf.py                  # Terraform IaC (Azure)
```

---

## Running the Demo

### Interactive Dashboard (no dependencies)
```bash
open frontend/index.html
```

**Try these scenarios in the Request Simulator:**
- **RBAC Deny**: Set Role = Actuary, Model = GPT-4o → 403 Denied
- **PII Injection**: Load "PII in Prompt" scenario → see redaction in audit log  
- **Blocked Content**: Load "Blocked Content" → content policy denial
- **Clean Request**: Normal underwriting query → full success path

### Backend API (Python 3.11+)
```bash
cd backend
pip install -r requirements.txt

# Azure configuration
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_KEY="your-key"
export COSMOS_CONNECTION_STRING="AccountEndpoint=..."
export REDIS_CONNECTION_STRING="your-redis.redis.cache.windows.net:6380,..."
export GATEWAY_JWT_SECRET="your-secret"  # dev only; production uses Azure AD

uvicorn main:app --reload
# API docs: http://localhost:8000/docs
```

### Issue a demo token and call the gateway:
```bash
# Get a token
curl -X POST "http://localhost:8000/api/v1/auth/token?user_id=sarah.chen&role=underwriter&team=UW&display_name=Sarah Chen"

# Use it to call the gateway
curl -X POST "http://localhost:8000/api/v1/gateway/chat" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"What is the max TIV for habitational risks?"}]}'
```

### Tests
```bash
cd backend
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## API Reference

### POST `/api/v1/gateway/chat`
Main LLM proxy. All governance rules enforced. Returns `GatewayResponse`.

**Response headers include:**
- `X-Gateway-Request-ID` — unique trace ID
- `X-Gateway-Policy` — ALLOW | DENY | REDACT_AND_ALLOW
- `X-Gateway-PII-Redacted` — count of redacted tokens
- `X-Gateway-Cost-USD` — request cost
- `X-Gateway-Team` — caller's team code (for chargeback)

### GET `/api/v1/audit/events` _(requires `audit:read` scope)_
Query the immutable audit log. Filterable by request_id and event_type.

### GET `/api/v1/usage/teams` _(requires `usage:read` scope)_
Cost and token breakdown by team with budget utilization.

### GET `/api/v1/usage/dashboard` _(requires `usage:read` scope)_
Real-time gateway health metrics and alerts.

---

## Production Deployment

```
Azure API Management         → External entry point, subscription keys, throttling
Azure Container Apps         → FastAPI gateway service (min 2 replicas, scale to 20)
Azure AD / Entra ID          → RS256 JWT issuance (replace demo HMAC)
Microsoft Presidio           → PII detection + anonymization
Azure Cosmos DB              → Append-only audit log (7yr TTL, Continuous backup)
Azure Cache for Redis        → Sliding window rate limiting (6379/TLS)
Azure Key Vault              → API keys, secrets, connection strings
Azure Container Registry     → Docker image registry
Azure Monitor + App Insights → Latency, cost, and quality telemetry
Azure Log Analytics          → SIEM integration, alerting
GitHub Actions               → CI/CD with test gate + Terraform apply
```

---

## Design Decisions

**Why a gateway rather than per-team libraries?**  
Per-team libraries drift over time — different versions, different PII handling, inconsistent audit formats. A central gateway guarantees consistent governance regardless of which team or framework calls the LLM.

**Why pseudonymize (not delete) user identity in audit logs?**  
Compliance requires the ability to investigate incidents. Full anonymization prevents that. Pseudonymization (SHA-256 + salt) allows correlation of events from the same user without storing the raw identity. The mapping key is in Key Vault, accessible only to authorized investigators.

**Why bidirectional PII scanning?**  
Outbound scanning is an uncommon but critical control. LLMs can reconstruct PII from partial context. Scanning completions before delivery closes this vector. The cost is ~3ms per response.

**Why OPA-ready policy engine?**  
Policy-as-code (OPA + Rego) allows non-engineers to update rules through Git without code deploys. The current Python engine is structurally identical — migrating means replacing `policy_engine.evaluate()` with an OPA HTTP call.

---

*Built to demonstrate Senior Applied AI Engineer capabilities for enterprise insurance AI.*
