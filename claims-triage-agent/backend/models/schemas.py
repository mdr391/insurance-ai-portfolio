"""
Data models for Claims Triage Agent.
All models use Pydantic v2 for validation and serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum
import uuid


class ClaimType(str, Enum):
    AUTO = "auto"
    PROPERTY = "property"
    LIABILITY = "liability"
    WORKERS_COMP = "workers_comp"
    COMMERCIAL = "commercial"


class FraudRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingQueue(str, Enum):
    STANDARD = "standard_review"
    FAST_TRACK = "fast_track"          # Low complexity, low fraud risk
    SPECIALIST = "specialist_review"   # Complex claims
    FRAUD_INVESTIGATION = "fraud_investigation"
    HUMAN_ESCALATION = "human_escalation"  # HITL override


class ClaimSubmission(BaseModel):
    """Raw claim submission from intake channel."""
    claim_text: str = Field(..., min_length=10, description="Raw claim description from claimant")
    policy_number: str = Field(..., description="Policyholder's policy number")
    claimant_name: str = Field(..., description="Name of claimant")
    incident_date: str = Field(..., description="Date of incident (YYYY-MM-DD)")
    claim_type: ClaimType
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    supporting_docs: Optional[List[str]] = Field(default=[], description="Base64-encoded document strings")


class ExtractedClaimData(BaseModel):
    """Structured data extracted by LLM from raw claim text."""
    incident_description: str
    location: Optional[str] = None
    estimated_loss_amount: Optional[float] = None
    injured_parties: List[str] = []
    witnesses: List[str] = []
    third_parties_involved: List[str] = []
    vehicle_info: Optional[dict] = None
    property_info: Optional[dict] = None
    prior_claims_mentioned: bool = False
    timeline_consistency: str  # "consistent", "inconsistent", "unclear"
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)


class FraudSignal(BaseModel):
    """Individual fraud indicator."""
    signal_type: str
    description: str
    severity: Literal["low", "medium", "high"]
    source: Literal["rules_engine", "ml_model", "llm_analysis"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class FraudAssessment(BaseModel):
    """Aggregated fraud risk assessment."""
    risk_level: FraudRiskLevel
    composite_score: float = Field(..., ge=0.0, le=1.0)
    signals: List[FraudSignal]
    rules_score: float
    ml_score: float
    llm_score: float
    narrative: str  # LLM-generated explanation
    recommended_action: str


class RoutingDecision(BaseModel):
    """Adjuster routing decision with explainability."""
    queue: RoutingQueue
    assigned_adjuster_tier: Literal["junior", "senior", "specialist", "fraud_investigator"]
    priority: Literal["low", "normal", "high", "urgent"]
    estimated_complexity: Literal["simple", "moderate", "complex"]
    routing_rationale: str
    requires_human_review: bool
    sla_hours: int  # Expected response time


class AuditEntry(BaseModel):
    """Immutable audit log entry — PII redacted."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stage: str  # "ingestion", "extraction", "fraud_scoring", "routing", "hitl_review"
    action: str
    actor: str  # system component or human reviewer
    input_hash: str  # SHA-256 of input (not raw input — privacy)
    output_summary: str  # Non-PII summary
    latency_ms: int
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None


class ClaimProcessingResult(BaseModel):
    """Full result returned to caller after agentic pipeline completes."""
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    policy_number: str
    claim_type: ClaimType
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    extracted_data: ExtractedClaimData
    fraud_assessment: FraudAssessment
    routing_decision: RoutingDecision
    pipeline_latency_ms: int
    total_tokens_used: int
    estimated_cost_usd: float
    audit_trail: List[AuditEntry]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HITLReviewRequest(BaseModel):
    """Human-in-the-loop review submission."""
    claim_id: str
    reviewer_id: str
    reviewer_role: str
    decision: Literal["approve_routing", "override_routing", "escalate", "reject"]
    override_queue: Optional[RoutingQueue] = None
    notes: str
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)
