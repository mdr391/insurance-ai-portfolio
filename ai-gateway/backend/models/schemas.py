"""
Data models for the Governed AI Gateway.
Every request, response, policy decision, and audit event is typed.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# ─── Identity & Access ────────────────────────────────────────────────────────

class UserRole(str, Enum):
    UNDERWRITER        = "underwriter"
    CLAIMS_ADJUSTER    = "claims_adjuster"
    FRAUD_INVESTIGATOR = "fraud_investigator"
    ACTUARY            = "actuary"
    COMPLIANCE         = "compliance"
    DATA_SCIENTIST     = "data_scientist"
    ADMIN              = "admin"
    SERVICE_ACCOUNT    = "service_account"


class TeamCode(str, Enum):
    UNDERWRITING   = "UW"
    CLAIMS         = "CL"
    FRAUD          = "FR"
    ACTUARIAL      = "AC"
    COMPLIANCE     = "CO"
    DATA           = "DS"
    ENGINEERING    = "EN"


class AuthenticatedCaller(BaseModel):
    """Represents a validated caller after JWT verification."""
    user_id: str
    email: str                      # Stored hashed in audit log
    role: UserRole
    team: TeamCode
    display_name: str
    scopes: List[str]               # e.g. ["llm:query", "llm:admin"]
    token_issued_at: datetime
    token_expires_at: datetime


# ─── Policy Engine ────────────────────────────────────────────────────────────

class PolicyDecision(str, Enum):
    ALLOW   = "allow"
    DENY    = "deny"
    REDACT  = "redact_and_allow"    # PII found, redacted, then forwarded


class PolicyRule(BaseModel):
    rule_id: str
    rule_name: str
    triggered: bool
    decision: PolicyDecision
    reason: str


class GatewayPolicyResult(BaseModel):
    """Aggregate policy evaluation result."""
    final_decision: PolicyDecision
    rules_evaluated: int
    rules_triggered: List[PolicyRule]
    pii_tokens_found: int
    pii_tokens_redacted: int
    content_flags: List[str]        # e.g. ["profanity", "competitor_mention"]
    evaluation_ms: int


# ─── LLM Request / Response ───────────────────────────────────────────────────

class LLMModel(str, Enum):
    GPT4O          = "gpt-4o"
    GPT4O_MINI     = "gpt-4o-mini"
    GPT35_TURBO    = "gpt-3.5-turbo"
    CLAUDE_SONNET  = "claude-3-5-sonnet"
    CLAUDE_HAIKU   = "claude-3-haiku"


class GatewayRequest(BaseModel):
    """Inbound request from any internal consumer."""
    model: LLMModel = LLMModel.GPT4O
    messages: List[Dict[str, str]]
    max_tokens: int = Field(default=1000, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None  # caller-supplied context

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class GatewayResponse(BaseModel):
    """Enriched response returned to the caller — always includes governance metadata."""
    request_id: str
    model: str
    content: str
    usage: TokenUsage
    pii_was_redacted: bool
    redaction_count: int
    policy_decision: PolicyDecision
    gateway_latency_ms: int         # Total gateway overhead
    upstream_latency_ms: int        # Time spent at LLM provider
    total_latency_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    governance_headers: Dict[str, str]  # X-Gateway-* headers for downstream use

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ─── Audit Log ────────────────────────────────────────────────────────────────

class AuditEventType(str, Enum):
    REQUEST_RECEIVED  = "request_received"
    AUTH_VERIFIED     = "auth_verified"
    AUTH_FAILED       = "auth_failed"
    POLICY_EVALUATED  = "policy_evaluated"
    PII_REDACTED      = "pii_redacted"
    REQUEST_DENIED    = "request_denied"
    LLM_CALLED        = "llm_called"
    RESPONSE_RETURNED = "response_returned"
    RATE_LIMITED      = "rate_limited"
    COST_ALERT        = "cost_alert"
    ANOMALY_DETECTED  = "anomaly_detected"


class AuditEvent(BaseModel):
    """Immutable audit log entry. PII-free by design."""
    event_id: str            = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    timestamp: datetime      = Field(default_factory=datetime.utcnow)
    event_type: AuditEventType
    user_id: str             # Pseudonymized — hashed in storage
    user_id_hash: str        # SHA-256 of actual user_id
    role: UserRole
    team: TeamCode
    model: Optional[str]     = None
    prompt_hash: str         = ""   # SHA-256 of prompt (not raw text)
    tokens_used: int         = 0
    cost_usd: float          = 0.0
    pii_redacted: int        = 0
    policy_decision: Optional[PolicyDecision] = None
    latency_ms: int          = 0
    status_code: int         = 200
    error: Optional[str]     = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ─── Cost & Usage ─────────────────────────────────────────────────────────────

class CostSummary(BaseModel):
    """Aggregated cost and usage metrics for a team/time window."""
    team: TeamCode
    period: str                     # e.g. "2024-05" or "2024-W22"
    total_requests: int
    successful_requests: int
    denied_requests: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    cost_by_model: Dict[str, float]
    avg_latency_ms: float
    pii_incidents: int
    budget_usd: float
    budget_utilization_pct: float


# ─── Rate Limiting ────────────────────────────────────────────────────────────

class RateLimitPolicy(BaseModel):
    """Per-role rate limit configuration."""
    role: UserRole
    requests_per_minute: int
    requests_per_hour: int
    tokens_per_day: int
    max_prompt_tokens: int


RATE_LIMIT_POLICIES: Dict[UserRole, RateLimitPolicy] = {
    UserRole.ADMIN: RateLimitPolicy(
        role=UserRole.ADMIN,
        requests_per_minute=120, requests_per_hour=2000,
        tokens_per_day=5_000_000, max_prompt_tokens=32000,
    ),
    UserRole.DATA_SCIENTIST: RateLimitPolicy(
        role=UserRole.DATA_SCIENTIST,
        requests_per_minute=60, requests_per_hour=1000,
        tokens_per_day=2_000_000, max_prompt_tokens=16000,
    ),
    UserRole.UNDERWRITER: RateLimitPolicy(
        role=UserRole.UNDERWRITER,
        requests_per_minute=30, requests_per_hour=500,
        tokens_per_day=500_000, max_prompt_tokens=8000,
    ),
    UserRole.CLAIMS_ADJUSTER: RateLimitPolicy(
        role=UserRole.CLAIMS_ADJUSTER,
        requests_per_minute=30, requests_per_hour=500,
        tokens_per_day=500_000, max_prompt_tokens=8000,
    ),
    UserRole.FRAUD_INVESTIGATOR: RateLimitPolicy(
        role=UserRole.FRAUD_INVESTIGATOR,
        requests_per_minute=40, requests_per_hour=600,
        tokens_per_day=750_000, max_prompt_tokens=12000,
    ),
    UserRole.ACTUARY: RateLimitPolicy(
        role=UserRole.ACTUARY,
        requests_per_minute=20, requests_per_hour=300,
        tokens_per_day=250_000, max_prompt_tokens=8000,
    ),
    UserRole.COMPLIANCE: RateLimitPolicy(
        role=UserRole.COMPLIANCE,
        requests_per_minute=20, requests_per_hour=200,
        tokens_per_day=200_000, max_prompt_tokens=8000,
    ),
    UserRole.SERVICE_ACCOUNT: RateLimitPolicy(
        role=UserRole.SERVICE_ACCOUNT,
        requests_per_minute=200, requests_per_hour=5000,
        tokens_per_day=10_000_000, max_prompt_tokens=32000,
    ),
}


# ─── Dashboard ────────────────────────────────────────────────────────────────

class DashboardMetrics(BaseModel):
    """Real-time gateway health and usage metrics for the admin dashboard."""
    snapshot_at: datetime = Field(default_factory=datetime.utcnow)
    requests_last_hour: int
    requests_last_24h: int
    tokens_last_24h: int
    cost_last_24h: float
    cost_mtd: float
    budget_mtd: float
    denied_requests_24h: int
    pii_incidents_24h: int
    avg_latency_ms_1h: float
    p99_latency_ms_1h: float
    active_models: List[str]
    by_team: List[CostSummary]
    by_model: Dict[str, Dict]
    recent_events: List[AuditEvent]
    alerts: List[str]
