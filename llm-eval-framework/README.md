# LLM Evaluation & Regression Framework

> **Portfolio Project** — Senior Applied AI Engineer · Insurance Domain  
> Demonstrates production-grade LLM evaluation with CI/CD regression gating

---

## Overview

A production evaluation harness for insurance AI systems. Every prompt change, model upgrade, or configuration update runs through a structured evaluation suite before deployment. The framework measures factual accuracy, hallucination rate, domain accuracy, regulatory safety, and PII leakage — then compares results against a baseline to detect regressions and block unsafe deployments.

**Live Demo**: Open `frontend/index.html` — fully interactive dashboard with 10 seeded historical runs, model comparison, metric trends, and regression analysis.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │           EVALUATION PIPELINE                   │
                    │                                                 │
  Dataset Store ──▶ │  1. LOAD CASES                                  │
  (DVC + Blob)      │     Insurance QA, Claims, Coverage, Fraud...    │
                    │                                                 │
                    │  2. CALL MODEL UNDER TEST (parallel)            │
                    │     GPT-4o / Claude / GPT-4o-mini              │
                    │     asyncio.gather + semaphore rate limiting    │
                    │                                                 │
                    │  3. EVALUATE EACH CASE (per metric)             │
                    │     ┌─────────────────────────────────────┐     │
                    │     │ LLM Judge (GPT-4o):                  │     │
                    │     │   factual_accuracy                   │     │
                    │     │   insurance_accuracy                 │     │
                    │     │   hallucination_rate                 │     │
                    │     │   regulatory_safety                  │     │
                    │     │                                      │     │
                    │     │ String / Rule-Based:                 │     │
                    │     │   answer_relevance (keyword)        │     │
                    │     │   pii_leakage (regex)               │     │
                    │     │   completeness (length ratio)       │     │
                    │     │   toxicity (pattern match)          │     │
                    │     └─────────────────────────────────────┘     │
                    │                                                 │
                    │  4. AGGREGATE                                    │
                    │     Weighted score · Pass rate · By difficulty  │
                    │                                                 │
                    │  5. GATE EVALUATION                             │
                    │     All critical metrics vs thresholds          │
                    │     PASS | FAIL | WARN                         │
                    │                                                 │
                    │  6. REGRESSION COMPARISON (if baseline set)    │
                    │     Delta per metric vs last PASS run           │
                    │     DEPLOY | BLOCK | INVESTIGATE               │
                    └─────────────────────────────────────────────────┘
                                        │
                            ┌───────────┴──────────┐
                            │    GitHub Actions     │
                            │  Post PR comment +   │
                            │  Block merge if FAIL │
                            └──────────────────────┘
```

---

## Evaluation Datasets (22 cases across 7 domains)

| Dataset | Cases | Focus |
|---------|-------|-------|
| Underwriting Q&A | 8 | TIV limits, flood zones, cyber controls, e-mod thresholds |
| Claims Triage | 3 | Fraud signal detection, routing decisions |
| Coverage Reasoning | 2 | Multi-condition coverage scenarios (PCO, liquor liability) |
| Fraud Signals | 2 | Staged accident indicators, threshold avoidance |
| Policy Extraction | 1 | Coverage comparison and endorsement analysis |
| Regulatory Compliance | 2 | Binding authority, fair lending rules |
| Customer Comms | 1 | Plain-language denial explanations |

All cases have: expected_answer, expected_keywords, forbidden_phrases, difficulty level, and citation requirements. Cases are authored by senior underwriters and compliance officers, versioned in DVC.

---

## Metric Suite

### LLM-as-Judge Metrics (GPT-4o evaluates the answer)
- **Factual Accuracy** (weight 25%, gate 80%) — Are coverage limits and exclusion rules stated correctly?
- **Hallucination Rate** (weight 20%, gate 85%) — Are claims grounded in the question/context?
- **Insurance Accuracy** (weight 20%, gate 78%) — Technical accuracy reviewable by a domain expert
- **Regulatory Safety** (weight 10%, gate 95%) — No illegal advice, prohibited commitments, or discrimination

### Rule / String-Based Metrics (fast, free, deterministic)
- **Answer Relevance** (weight 15%) — Fraction of expected keywords present
- **PII Leakage** (weight 10%, gate 99%) — Regex scan for SSN, VIN, email, phone in output
- **Completeness** — Answer length ratio vs reference
- **Toxicity** — Pattern-based inappropriate content detection

### Gate Thresholds
All critical metrics must meet thresholds for **PASS**. Any single metric failure = **FAIL** → deployment blocked.

```
factual_accuracy    ≥ 80%   (LLM judge)
hallucination_rate  ≥ 85%   (hybrid)
insurance_accuracy  ≥ 78%   (LLM judge)
regulatory_safety   ≥ 95%   (LLM judge)
pii_leakage         ≥ 99%   (regex — near-zero tolerance)
```

### Regression Thresholds
Deployment is blocked if any metric **drops more than**:
```
factual_accuracy    -3pp    hallucination_rate  -3pp
insurance_accuracy  -5pp    pii_leakage         -1pp (any PII regression = block)
regulatory_safety   -2pp
```

---

## Project Structure

```
llm-eval-framework/
├── backend/
│   ├── api/routes.py                # FastAPI REST API
│   ├── datasets/insurance_cases.py  # 22 ground-truth eval cases
│   ├── evaluators/metrics.py        # 8 metric evaluators (LLM + rule-based)
│   ├── runners/
│   │   ├── eval_runner.py           # Orchestrates full runs (async, concurrent)
│   │   └── regression.py           # Baseline comparison engine
│   ├── storage/run_store.py         # Run persistence (pre-seeded with 10 runs)
│   ├── models/schemas.py            # Pydantic v2 data models
│   ├── requirements.txt
│   └── tests/test_eval_framework.py # pytest suite (45+ tests)
├── ci/eval.yml                      # GitHub Actions CI workflow
└── frontend/index.html              # Interactive dashboard
```

---

## Running the Demo

### Interactive Dashboard (no dependencies)
```bash
open frontend/index.html
```

**What to explore:**
- **Overview** — Latest run metrics with gate status, trend chart across all models
- **Model Compare** — Side-by-side GPT-4o vs Claude vs GPT-3.5-turbo vs Haiku
- **Run Eval** — Trigger a new run, watch progress, see gate result
- **Metric Trends** — Historical chart for any metric across all runs
- **Regression Report** — Latest vs baseline delta with deploy/block recommendation
- **Gate Thresholds** — Full threshold configuration reference

### Backend API
```bash
cd backend
pip install -r requirements.txt
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_KEY="..."
python -m api.routes
# http://localhost:8000/docs
```

### CLI Evaluation Runner
```bash
cd backend
python -m eval_runner \
  --model gpt-4o \
  --datasets underwriting_qa claims_triage \
  --sample-size 10 \
  --baseline-run-id <run_id>
```

### Tests
```bash
cd backend
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## API Reference

### POST `/api/v1/runs`
Trigger a new eval run (async). Returns `run_id` immediately.

**Request:**
```json
{
  "model": "gpt-4o",
  "datasets": ["underwriting_qa", "claims_triage"],
  "metrics": ["factual_accuracy", "hallucination_rate", "insurance_accuracy"],
  "sample_size": 10,
  "baseline_run_id": "optional-run-id",
  "triggered_by": "ci"
}
```

### GET `/api/v1/runs`
List all runs with summary statistics.

### GET `/api/v1/runs/{run_id}`
Full run result with per-case breakdown.

### GET `/api/v1/runs/compare`
Latest results per model (comparison view).

### GET `/api/v1/runs/trend?metric=factual_accuracy&model=gpt-4o`
Historical trend data for charting.

---

## CI/CD Integration

The `ci/eval.yml` GitHub Actions workflow:

**On PR** → Smoke eval (10 sampled cases) → Post result as PR comment
**On merge to main** → Full eval (all 22 cases) → Regression vs baseline → Block merge if FAIL

```yaml
# Simplified gate check in CI
python -m eval_runner --model gpt-4o --datasets all --baseline-run-id $BASELINE
# Exit code 1 if gate FAIL → GitHub Actions marks step as failed → PR blocked
```

---

## Production Deployment

```
Azure Container Jobs         → Parallel eval execution (each case = separate task)
Azure OpenAI Service         → GPT-4o as model under test + judge model
Azure Blob Storage           → Run artifacts (JSON), dataset files (via DVC)
Azure PostgreSQL              → Run metadata, queryable history
Azure Monitor                → Metric telemetry, alerting on score drops
GitHub Actions               → CI/CD trigger, PR comment posting, merge gating
DVC + Azure Blob             → Dataset versioning (reproducible eval suites)
```

---

## Design Decisions

**Why two evaluator types (LLM judge + rule-based)?**  
LLM judges are powerful but expensive and non-deterministic. Rule-based evaluators (keyword coverage, PII regex) are free, instant, and fully reproducible. The hybrid approach uses each where it makes sense: LLM for semantic accuracy, rules for structural compliance.

**Why GPT-4o as judge even for GPT-4o answers?**  
The judge evaluates factual accuracy against a fixed reference answer — it's not a "what's better" comparison but a "does this match the ground truth" check. Temperature=0 maximizes reproducibility. In production, we'd consider a more capable future model as judge.

**Why 22 cases and not thousands?**  
22 carefully curated expert-authored cases run in <2 minutes and cost <$0.25. A larger LLM-generated dataset would have noise that obscures real signal. Quality over quantity for regulated domain evals. Cases grow through PR-gated expert review.

**Why PII leakage has 99% threshold?**  
Any PII in an LLM output is a regulatory incident. Near-zero tolerance reflects GLBA and CCPA requirements. A single PII leak detected by this evaluator in CI is worth far more than any other metric.

---

*Built to demonstrate Senior Applied AI Engineer capabilities for enterprise insurance AI.*
