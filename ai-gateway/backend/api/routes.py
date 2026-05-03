"""
API Routes — Governed AI Gateway.

Endpoints:
  POST /api/v1/gateway/chat          — Main LLM proxy (governed)
  POST /api/v1/auth/token            — Issue demo JWT (dev only)
  GET  /api/v1/audit/events          — Audit log query (compliance)
  GET  /api/v1/usage/teams           — Cost by team (admin/compliance)
  GET  /api/v1/usage/dashboard       — Real-time metrics dashboard
  GET  /health                       — Health probe
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from gateway.core import ai_gateway
from middleware.auth import auth_middleware
from models.schemas import (
    AuditEventType, AuthenticatedCaller, DashboardMetrics,
    GatewayRequest, LLMModel, TeamCode, UserRole,
)
from services.audit import audit_logger, compute_cost

logger = logging.getLogger(__name__)

gateway_router  = APIRouter()
auth_router     = APIRouter()
audit_router    = APIRouter()
usage_router    = APIRouter()
health_router   = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────

@health_router.get("")
async def health():
    return {
        "status": "healthy",
        "service": "ai-gateway",
        "version": "1.0.0",
        "total_requests": audit_logger.total_requests(),
        "total_cost_usd": audit_logger.total_cost(),
    }


# ── Auth (demo only) ────────────────────────────────────────────────────────

@auth_router.post("/token")
async def issue_token(
    user_id: str,
    role: UserRole,
    team: TeamCode,
    display_name: str = "Demo User",
):
    """
    Issue a demo JWT for testing. In production, users authenticate
    via Azure AD / SSO — this endpoint does not exist.
    """
    token = auth_middleware.issue_demo_token(user_id, role, team, display_name)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
        "scope": " ".join(auth_middleware._scopes_for_role(role)),
        "note": "Demo token only. In production, authenticate via Azure AD.",
    }


# ── Gateway ─────────────────────────────────────────────────────────────────

@gateway_router.post("/chat")
async def gateway_chat(
    request: GatewayRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Governed LLM proxy endpoint.

    All application teams call this instead of hitting LLM providers directly.
    The gateway enforces: auth, RBAC, PII redaction, rate limits, audit logging.

    Returns the LLM response enriched with governance metadata headers.
    """
    response = await ai_gateway.handle(request, authorization)
    return response


# ── Audit ───────────────────────────────────────────────────────────────────

@audit_router.get("/events")
async def get_audit_events(
    limit: int = Query(default=50, ge=1, le=500),
    request_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(None),
):
    """
    Query the immutable audit log.
    Requires 'audit:read' scope (compliance officers, senior management, fraud investigators).
    """
    caller = auth_middleware.verify_token(authorization)
    auth_middleware.enforce_scope(caller, "audit:read")

    events = audit_logger.get_events(
        limit=limit,
        request_id=request_id,
        event_type=event_type,
    )
    return {
        "events": [e.model_dump() for e in events],
        "total_returned": len(events),
        "queried_by": caller.user_id,
        "queried_at": datetime.utcnow().isoformat(),
    }


# ── Usage & Cost ────────────────────────────────────────────────────────────

@usage_router.get("/teams")
async def get_team_usage(authorization: Optional[str] = Header(None)):
    """Cost and token breakdown by team. Requires 'usage:read' scope."""
    caller = auth_middleware.verify_token(authorization)
    auth_middleware.enforce_scope(caller, "usage:read")

    cost_by_team  = audit_logger.get_cost_by_team()
    token_by_team = audit_logger.get_tokens_by_team()
    req_by_team   = audit_logger.get_requests_by_team()
    pii_by_team   = audit_logger.get_pii_by_team()

    from models.schemas import TEAM_DAILY_BUDGET_USD
    teams = []
    for tc in TeamCode:
        cost = cost_by_team.get(tc.value, 0.0)
        budget = TEAM_DAILY_BUDGET_USD.get(tc.value, 100.0)
        teams.append({
            "team": tc.value,
            "requests": req_by_team.get(tc.value, 0),
            "tokens": token_by_team.get(tc.value, 0),
            "cost_usd": round(cost, 4),
            "budget_usd": budget,
            "utilization_pct": round((cost / budget) * 100, 1) if budget > 0 else 0,
            "pii_incidents": pii_by_team.get(tc.value, 0),
        })

    return {
        "teams": teams,
        "total_cost_usd": audit_logger.total_cost(),
        "period": "today",
    }


@usage_router.get("/dashboard")
async def get_dashboard(authorization: Optional[str] = Header(None)):
    """Real-time dashboard metrics. Requires 'usage:read' scope."""
    caller = auth_middleware.verify_token(authorization)
    auth_middleware.enforce_scope(caller, "usage:read")

    events = audit_logger.get_events(limit=20)
    alerts = []
    if audit_logger.total_pii_incidents() > 10:
        alerts.append(f"⚠ {audit_logger.total_pii_incidents()} PII incidents today — review required")
    if audit_logger.total_denied() > 5:
        alerts.append(f"🚫 {audit_logger.total_denied()} denied requests — check RBAC policy")

    return {
        "snapshot_at": datetime.utcnow().isoformat(),
        "requests_total": audit_logger.total_requests(),
        "tokens_total": sum(audit_logger.get_tokens_by_team().values()),
        "cost_total_usd": audit_logger.total_cost(),
        "denied_total": audit_logger.total_denied(),
        "pii_incidents": audit_logger.total_pii_incidents(),
        "avg_latency_ms": audit_logger.get_avg_latency(),
        "p99_latency_ms": audit_logger.get_p99_latency(),
        "cost_by_team": audit_logger.get_cost_by_team(),
        "cost_by_model": audit_logger.get_cost_by_model(),
        "recent_events": [e.model_dump() for e in events[:10]],
        "alerts": alerts,
    }
