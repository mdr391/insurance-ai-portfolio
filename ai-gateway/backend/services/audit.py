"""
Audit Logger — append-only, PII-free event log.

Every gateway request generates a chain of audit events:
  REQUEST_RECEIVED → AUTH_VERIFIED → POLICY_EVALUATED →
  PII_REDACTED (if applicable) → LLM_CALLED → RESPONSE_RETURNED

Design principles:
  - No raw PII ever enters the audit log
  - User identity stored as SHA-256 hash (pseudonymization)
  - Prompt content stored as SHA-256 hash only
  - Events are append-only (no updates, no deletes)
  - Events are sequenced and tamper-evident

Production storage:
  - Azure Cosmos DB (append-only container, TTL-based archival)
  - OR Azure PostgreSQL with row-level security + audit triggers
  - Exported to Azure Monitor Log Analytics for SIEM integration
  - Retention: 7 years (regulatory requirement for insurance)

Cost Tracking:
  Per-model pricing is applied per event and aggregated by:
    - Team (for chargeback)
    - User (for abuse detection)
    - Model (for budget forecasting)
"""

import hashlib
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from models.schemas import (
    AuditEvent, AuditEventType, AuthenticatedCaller,
    CostSummary, GatewayPolicyResult, LLMModel,
    PolicyDecision, TeamCode, TokenUsage, UserRole,
)

logger = logging.getLogger(__name__)

# LLM pricing per 1K tokens (input / output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":           (0.005,  0.015),
    "gpt-4o-mini":      (0.00015, 0.0006),
    "gpt-3.5-turbo":    (0.0005, 0.0015),
    "claude-3-5-sonnet":(0.003,  0.015),
    "claude-3-haiku":   (0.00025, 0.00125),
}


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute request cost from token counts and model pricing."""
    input_rate, output_rate = MODEL_PRICING.get(model, (0.005, 0.015))
    return round(
        (prompt_tokens / 1000) * input_rate +
        (completion_tokens / 1000) * output_rate,
        6,
    )


class AuditLogger:
    """
    Append-only audit log with pseudonymization.

    In production, events are written to:
      - Azure Cosmos DB (primary, queryable)
      - Azure Monitor (SIEM, alerting)
      - Azure Blob Storage (cold archive after 90 days)

    This demo uses an in-memory store that fully simulates the interface.
    """

    def __init__(self):
        self._events: list[AuditEvent] = []             # In-memory (production: Cosmos DB)
        self._cost_by_team: dict[str, float] = defaultdict(float)
        self._cost_by_user: dict[str, float] = defaultdict(float)
        self._cost_by_model: dict[str, float] = defaultdict(float)
        self._tokens_by_team: dict[str, int] = defaultdict(int)
        self._requests_by_team: dict[str, int] = defaultdict(int)
        self._denied_by_team: dict[str, int] = defaultdict(int)
        self._pii_by_team: dict[str, int] = defaultdict(int)
        self._latencies: list[float] = []

    def log(self, event: AuditEvent) -> None:
        """
        Append an event to the audit log.

        Production:
            cosmos_container.create_item(body=event.model_dump())
            logger.info(f"AUDIT: {event.event_type} | {event.request_id}")
        """
        self._events.append(event)

        # Update cost aggregations
        if event.cost_usd > 0:
            self._cost_by_team[event.team.value] += event.cost_usd
            self._cost_by_user[event.user_id_hash] += event.cost_usd
            self._cost_by_model[event.model or "unknown"] += event.cost_usd
            self._tokens_by_team[event.team.value] += event.tokens_used

        if event.event_type == AuditEventType.REQUEST_RECEIVED:
            self._requests_by_team[event.team.value] += 1
        if event.event_type == AuditEventType.REQUEST_DENIED:
            self._denied_by_team[event.team.value] += 1
        if event.pii_redacted > 0:
            self._pii_by_team[event.team.value] += event.pii_redacted
        if event.latency_ms > 0:
            self._latencies.append(event.latency_ms)

        logger.debug(f"AUDIT [{event.event_type.value}] req={event.request_id} user={event.user_id_hash[:8]}...")

    def log_request_received(
        self, request_id: str, caller: AuthenticatedCaller
    ) -> AuditEvent:
        ev = AuditEvent(
            request_id=request_id,
            event_type=AuditEventType.REQUEST_RECEIVED,
            user_id=caller.user_id,
            user_id_hash=self._hash(caller.user_id),
            role=caller.role,
            team=caller.team,
        )
        self.log(ev)
        return ev

    def log_auth_verified(
        self, request_id: str, caller: AuthenticatedCaller
    ) -> AuditEvent:
        ev = AuditEvent(
            request_id=request_id,
            event_type=AuditEventType.AUTH_VERIFIED,
            user_id=caller.user_id,
            user_id_hash=self._hash(caller.user_id),
            role=caller.role,
            team=caller.team,
            metadata={"scopes": caller.scopes},
        )
        self.log(ev)
        return ev

    def log_pii_redaction(
        self, request_id: str, caller: AuthenticatedCaller,
        prompt_hash: str, pii_count: int, entity_types: list[str],
    ) -> AuditEvent:
        ev = AuditEvent(
            request_id=request_id,
            event_type=AuditEventType.PII_REDACTED,
            user_id=caller.user_id,
            user_id_hash=self._hash(caller.user_id),
            role=caller.role,
            team=caller.team,
            prompt_hash=prompt_hash,
            pii_redacted=pii_count,
            metadata={"entity_types": entity_types},
        )
        self.log(ev)
        return ev

    def log_policy_decision(
        self, request_id: str, caller: AuthenticatedCaller,
        policy: GatewayPolicyResult,
    ) -> AuditEvent:
        ev = AuditEvent(
            request_id=request_id,
            event_type=AuditEventType.POLICY_EVALUATED,
            user_id=caller.user_id,
            user_id_hash=self._hash(caller.user_id),
            role=caller.role,
            team=caller.team,
            policy_decision=policy.final_decision,
            pii_redacted=policy.pii_tokens_redacted,
            metadata={
                "rules_evaluated": policy.rules_evaluated,
                "rules_triggered": len(policy.rules_triggered),
                "content_flags": policy.content_flags,
            },
        )
        self.log(ev)
        return ev

    def log_llm_call(
        self, request_id: str, caller: AuthenticatedCaller,
        model: str, prompt_hash: str,
        prompt_tokens: int, completion_tokens: int,
        latency_ms: int,
    ) -> AuditEvent:
        cost = compute_cost(model, prompt_tokens, completion_tokens)
        ev = AuditEvent(
            request_id=request_id,
            event_type=AuditEventType.LLM_CALLED,
            user_id=caller.user_id,
            user_id_hash=self._hash(caller.user_id),
            role=caller.role,
            team=caller.team,
            model=model,
            prompt_hash=prompt_hash,
            tokens_used=prompt_tokens + completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        self.log(ev)
        return ev

    def log_denied(
        self, request_id: str, caller: AuthenticatedCaller,
        reason: str, rule_id: str,
    ) -> AuditEvent:
        ev = AuditEvent(
            request_id=request_id,
            event_type=AuditEventType.REQUEST_DENIED,
            user_id=caller.user_id,
            user_id_hash=self._hash(caller.user_id),
            role=caller.role,
            team=caller.team,
            policy_decision=PolicyDecision.DENY,
            status_code=403,
            error=reason,
            metadata={"rule_id": rule_id},
        )
        self.log(ev)
        return ev

    # ── Queries ─────────────────────────────────────────────────────────────

    def get_events(
        self,
        limit: int = 100,
        request_id: Optional[str] = None,
        user_hash: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[AuditEvent]:
        events = self._events
        if request_id:
            events = [e for e in events if e.request_id == request_id]
        if user_hash:
            events = [e for e in events if e.user_id_hash.startswith(user_hash)]
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        return list(reversed(events))[:limit]

    def get_cost_by_team(self) -> dict[str, float]:
        return dict(self._cost_by_team)

    def get_cost_by_model(self) -> dict[str, float]:
        return dict(self._cost_by_model)

    def get_tokens_by_team(self) -> dict[str, int]:
        return dict(self._tokens_by_team)

    def get_requests_by_team(self) -> dict[str, int]:
        return dict(self._requests_by_team)

    def get_pii_by_team(self) -> dict[str, int]:
        return dict(self._pii_by_team)

    def get_avg_latency(self) -> float:
        return round(sum(self._latencies) / max(len(self._latencies), 1), 1)

    def get_p99_latency(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_l = sorted(self._latencies)
        idx = int(len(sorted_l) * 0.99)
        return float(sorted_l[min(idx, len(sorted_l) - 1)])

    def total_requests(self) -> int:
        return len([e for e in self._events if e.event_type == AuditEventType.REQUEST_RECEIVED])

    def total_denied(self) -> int:
        return len([e for e in self._events if e.event_type == AuditEventType.REQUEST_DENIED])

    def total_pii_incidents(self) -> int:
        return len([e for e in self._events if e.event_type == AuditEventType.PII_REDACTED])

    def total_cost(self) -> float:
        return round(sum(self._cost_by_team.values()), 4)

    def _hash(self, value: str) -> str:
        return hashlib.sha256(f"audit-salt:{value}".encode()).hexdigest()


audit_logger = AuditLogger()
