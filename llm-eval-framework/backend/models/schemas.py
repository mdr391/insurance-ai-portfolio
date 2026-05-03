"""
Data models for the LLM Evaluation & Regression Framework.

Every evaluation concept is strongly typed:
  - EvalCase:      A single test question with expected answer
  - EvalRun:       A full evaluation suite run (model × dataset × timestamp)
  - MetricResult:  A single metric score with rationale
  - CaseResult:    All metrics for one eval case
  - RunResult:     Aggregated results for a full run
  - RegressionReport: Comparison of two runs (current vs baseline)
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


# ─── Enums ────────────────────────────────────────────────────────────────────

class EvalDataset(str, Enum):
    """Built-in insurance evaluation datasets."""
    UNDERWRITING_QA      = "underwriting_qa"
    CLAIMS_TRIAGE        = "claims_triage"
    FRAUD_SIGNALS        = "fraud_signals"
    COVERAGE_REASONING   = "coverage_reasoning"
    POLICY_EXTRACTION    = "policy_extraction"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    CUSTOMER_COMMS       = "customer_comms"


class MetricName(str, Enum):
    """All supported evaluation metrics."""
    # Correctness
    FACTUAL_ACCURACY     = "factual_accuracy"
    ANSWER_RELEVANCE     = "answer_relevance"
    COMPLETENESS         = "completeness"
    # Safety / Governance
    HALLUCINATION_RATE   = "hallucination_rate"
    PII_LEAKAGE          = "pii_leakage"
    TOXICITY             = "toxicity"
    # Domain-specific
    INSURANCE_ACCURACY   = "insurance_accuracy"   # Domain-expert checklist
    CITATION_PRECISION   = "citation_precision"   # For RAG outputs
    REGULATORY_SAFETY    = "regulatory_safety"    # No illegal advice
    # Performance
    LATENCY_P50_MS       = "latency_p50_ms"
    LATENCY_P99_MS       = "latency_p99_ms"
    COST_PER_1K_TOKENS   = "cost_per_1k_tokens"
    # Consistency
    CONSISTENCY          = "consistency"          # Same Q → same A across runs
    ROBUSTNESS           = "robustness"           # Paraphrase Q → same A


class RunStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETE   = "complete"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


class GateStatus(str, Enum):
    PASS  = "pass"
    FAIL  = "fail"
    WARN  = "warn"


# ─── Eval Case ────────────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    """A single test case in the evaluation dataset."""
    case_id: str                  = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset: EvalDataset
    question: str
    expected_answer: str
    expected_keywords: List[str]  = Field(default_factory=list)
    forbidden_phrases: List[str]  = Field(default_factory=list)  # Must not appear
    context_docs: List[str]       = Field(default_factory=list)  # For RAG evals
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    category: str                 = ""       # Sub-category within dataset
    requires_citation: bool       = False    # Whether answer must cite sources
    tags: List[str]               = Field(default_factory=list)


# ─── Metric Results ───────────────────────────────────────────────────────────

class MetricResult(BaseModel):
    """Score for a single metric on a single case."""
    metric: MetricName
    score: float               = Field(..., ge=0.0, le=1.0)
    passed: bool               # score >= threshold for this metric
    threshold: float           # Threshold used for pass/fail
    rationale: str             = ""    # LLM-generated explanation of the score
    scorer_type: Literal[      # How the score was computed
        "llm_judge",           # GPT-4o as judge
        "string_match",        # Exact/fuzzy keyword matching
        "rule_based",          # Deterministic rules
        "embedding_sim",       # Cosine similarity
        "human",               # Human-labeled ground truth
    ] = "llm_judge"


class CaseResult(BaseModel):
    """Full result for one eval case — all metrics."""
    case_id: str
    run_id: str
    question: str
    expected_answer: str
    actual_answer: str
    metrics: List[MetricResult]
    overall_score: float        # Weighted average across metrics
    passed: bool                # All required metrics passed
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str
    dataset: EvalDataset
    difficulty: str
    evaluated_at: datetime      = Field(default_factory=datetime.utcnow)

    @property
    def metric_dict(self) -> Dict[str, float]:
        return {m.metric.value: m.score for m in self.metrics}

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ─── Run ──────────────────────────────────────────────────────────────────────

class EvalRunConfig(BaseModel):
    """Configuration for an evaluation run."""
    run_id: str                  = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str                    = ""
    model: str                   = "gpt-4o"
    datasets: List[EvalDataset]  = Field(default_factory=list)
    metrics: List[MetricName]    = Field(default_factory=list)
    sample_size: Optional[int]   = None    # None = run all cases
    temperature: float           = 0.1     # Low for reproducibility
    judge_model: str             = "gpt-4o"   # Model used to score answers
    baseline_run_id: Optional[str] = None  # If set, run regression comparison
    tags: List[str]              = Field(default_factory=list)
    triggered_by: str            = "manual"  # "ci", "scheduled", "manual"


class RunSummary(BaseModel):
    """Aggregated metrics for a completed run."""
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    avg_overall_score: float
    metrics_summary: Dict[str, Dict]   # metric → {avg, min, max, pass_rate}
    by_dataset: Dict[str, Dict]        # dataset → {pass_rate, avg_score}
    by_difficulty: Dict[str, Dict]     # difficulty → {pass_rate, avg_score}
    total_cost_usd: float
    total_tokens: int
    avg_latency_ms: float
    p99_latency_ms: float
    gate_status: GateStatus
    gate_failures: List[str]           # Which thresholds failed


class EvalRunResult(BaseModel):
    """Complete result of an evaluation run."""
    config: EvalRunConfig
    status: RunStatus           = RunStatus.COMPLETE
    started_at: datetime        = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float     = 0.0
    case_results: List[CaseResult] = Field(default_factory=list)
    summary: Optional[RunSummary] = None
    error: Optional[str]        = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ─── Regression Report ────────────────────────────────────────────────────────

class MetricDelta(BaseModel):
    """Change in a metric between current and baseline run."""
    metric: str
    baseline_score: float
    current_score: float
    delta: float               # current - baseline
    delta_pct: float           # percentage change
    direction: Literal["improved", "regressed", "unchanged"]
    significant: bool          # |delta| > significance threshold


class RegressionReport(BaseModel):
    """Comparison of current run against a baseline."""
    report_id: str             = Field(default_factory=lambda: str(uuid.uuid4()))
    current_run_id: str
    baseline_run_id: str
    current_model: str
    baseline_model: str
    generated_at: datetime     = Field(default_factory=datetime.utcnow)
    metric_deltas: List[MetricDelta]
    regressions: List[str]     # Metric names that regressed significantly
    improvements: List[str]    # Metric names that improved significantly
    new_failures: List[str]    # Case IDs that now fail (were passing in baseline)
    new_passes: List[str]      # Case IDs that now pass (were failing in baseline)
    gate_status: GateStatus
    gate_verdict: str          # Human-readable verdict
    recommendation: str        # "deploy", "block", "investigate"

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ─── Gate Thresholds ──────────────────────────────────────────────────────────

# CI gate: run must meet ALL these thresholds to PASS
GATE_THRESHOLDS: Dict[MetricName, float] = {
    MetricName.FACTUAL_ACCURACY:    0.80,
    MetricName.HALLUCINATION_RATE:  0.85,   # Note: higher = less hallucination
    MetricName.ANSWER_RELEVANCE:    0.75,
    MetricName.INSURANCE_ACCURACY:  0.78,
    MetricName.PII_LEAKAGE:         0.99,   # Near-zero tolerance
    MetricName.REGULATORY_SAFETY:   0.95,
}

# Regression gate: block deploy if any metric drops more than this
REGRESSION_THRESHOLDS: Dict[str, float] = {
    "factual_accuracy":   0.03,   # -3% absolute triggers block
    "hallucination_rate": 0.03,
    "insurance_accuracy": 0.05,
    "pii_leakage":        0.01,   # Any PII regression = block
    "regulatory_safety":  0.02,
}

# Metric weights for overall score
METRIC_WEIGHTS: Dict[MetricName, float] = {
    MetricName.FACTUAL_ACCURACY:   0.25,
    MetricName.HALLUCINATION_RATE: 0.20,
    MetricName.INSURANCE_ACCURACY: 0.20,
    MetricName.ANSWER_RELEVANCE:   0.15,
    MetricName.REGULATORY_SAFETY:  0.10,
    MetricName.PII_LEAKAGE:        0.10,
}
