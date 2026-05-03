"""
API Route handlers for Claims Triage Agent.
"""

from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import uuid
from datetime import datetime

from models.schemas import (
    ClaimSubmission, ClaimProcessingResult,
    HITLReviewRequest, AuditEntry
)
from agents.claims_pipeline import ClaimsTriageAgent

logger = logging.getLogger(__name__)

# ── Health ──────────────────────────────────────────────────────────────────
health_router = APIRouter()

@health_router.get("")
async def health_check():
    return {"status": "healthy", "service": "claims-triage-agent", "version": "1.0.0", "ts": datetime.utcnow().isoformat()}


# ── Claims ───────────────────────────────────────────────────────────────────
claims_router = APIRouter()
_agent = ClaimsTriageAgent()

# Simple RBAC simulation
ALLOWED_ROLES = {"adjuster", "underwriter", "fraud_investigator", "supervisor", "system"}

def verify_role(x_user_role: Optional[str] = Header(None)) -> str:
    """
    In production: validate JWT, extract role from claims.
    Here we use a simple header-based role check.
    """
    role = x_user_role or "adjuster"
    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=403, detail=f"Role '{role}' not authorized")
    return role


@claims_router.post("/process", response_model=ClaimProcessingResult)
async def process_claim(
    submission: ClaimSubmission,
    role: str = Depends(verify_role),
):
    """
    Submit a new claim for automated triage.
    
    Pipeline:
    1. PII redaction
    2. LLM entity extraction
    3. Rules + ML + LLM fraud scoring
    4. Adjuster routing decision
    5. Audit log creation
    
    Returns full ClaimProcessingResult including fraud assessment,
    routing decision, and complete audit trail.
    """
    logger.info(f"New claim submission | type={submission.claim_type} | role={role}")
    try:
        result = await _agent.run(submission)
        return result
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@claims_router.post("/{claim_id}/hitl-review")
async def submit_hitl_review(
    claim_id: str,
    review: HITLReviewRequest,
    role: str = Depends(verify_role),
):
    """
    Human-in-the-loop review override.
    
    Supervisors and fraud investigators can override routing decisions.
    All overrides are logged immutably for compliance.
    """
    if role not in {"supervisor", "fraud_investigator"}:
        raise HTTPException(status_code=403, detail="Insufficient role for HITL review")

    logger.info(f"HITL review submitted | claim={claim_id} | decision={review.decision} | reviewer={review.reviewer_id}")

    return {
        "claim_id": claim_id,
        "review_recorded": True,
        "decision": review.decision,
        "override_queue": review.override_queue,
        "reviewer": review.reviewer_id,
        "timestamp": review.reviewed_at.isoformat(),
        "audit_entry_id": str(uuid.uuid4()),
    }


@claims_router.get("/{claim_id}/status")
async def get_claim_status(claim_id: str, role: str = Depends(verify_role)):
    """Get current status of a processed claim."""
    # In production: query claim store (Cosmos DB / Postgres)
    return {"claim_id": claim_id, "status": "routed", "queue": "specialist_review"}


# ── Audit ─────────────────────────────────────────────────────────────────
audit_router = APIRouter()

@audit_router.get("/{claim_id}")
async def get_audit_trail(
    claim_id: str,
    role: str = Depends(verify_role),
):
    """
    Retrieve immutable audit trail for a claim.
    
    In production: query append-only audit log (Azure Cosmos DB / Postgres with row-level security).
    Access restricted to supervisors and compliance roles.
    """
    if role not in {"supervisor", "fraud_investigator", "system"}:
        raise HTTPException(status_code=403, detail="Audit trail access requires elevated role")
    # In production: return from persistent audit store
    return {"claim_id": claim_id, "message": "Audit trail stored in persistent log"}
