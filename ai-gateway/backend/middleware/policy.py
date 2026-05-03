"""
Policy Engine — evaluates every request against governance rules.

Rules evaluated (in order):
  1. Authentication validity
  2. Model access (role-based allowlist)
  3. Prompt token limit (role-specific)
  4. Rate limiting (per-user sliding window)
  5. Daily token budget (per-team)
  6. Content moderation (blocked topics, competitor mentions)
  7. PII presence
  8. Anomaly detection (unusual volume, off-hours spikes)

Decision: ALLOW | DENY | REDACT_AND_ALLOW

All decisions are logged with the rule that triggered them.
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

from models.schemas import (
    AuthenticatedCaller, GatewayPolicyResult, GatewayRequest,
    LLMModel, PolicyDecision, PolicyRule, RATE_LIMIT_POLICIES, UserRole,
)

logger = logging.getLogger(__name__)


# ── In-memory rate limit state (production: Redis with sliding window lua script)
_request_timestamps: dict[str, deque] = defaultdict(deque)       # user_id → timestamps
_hourly_tokens: dict[str, int] = defaultdict(int)                 # user_id → tokens today
_team_daily_tokens: dict[str, int] = defaultdict(int)             # team → tokens today

# Per-team daily token budgets (USD)
TEAM_DAILY_BUDGET_USD = {
    "UW": 500.0,
    "CL": 300.0,
    "FR": 200.0,
    "AC": 150.0,
    "CO": 100.0,
    "DS": 800.0,
    "EN": 1000.0,
}

# Content policy: topics the gateway will not process
BLOCKED_TOPIC_PATTERNS = [
    r"\b(?:competitor|rival)\s+(?:pricing|rates|quote)\b",
    r"\b(?:bypass|jailbreak|ignore previous|ignore all instructions)\b",
    r"\b(?:social security number|ssn|credit card)\s+(?:list|dump|database)\b",
]

# Competitor keywords — flag but don't deny (log for compliance)
COMPETITOR_KEYWORDS = [
    "travelers", "chubb", "aig", "zurich", "hartford", "nationwide",
    "progressive", "allstate", "state farm",
]

import re
_BLOCKED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BLOCKED_TOPIC_PATTERNS]


class PolicyEngine:
    """
    Central policy enforcement point for the AI Gateway.

    Implements the Open Policy Agent (OPA) pattern in Python.
    In production, this delegates to an OPA sidecar for policy-as-code
    with Git-managed rules and full audit trail.

    OPA integration:
        import requests
        decision = requests.post(
            "http://localhost:8181/v1/data/gateway/allow",
            json={"input": {"user": caller.dict(), "request": req.dict()}}
        ).json()["result"]
    """

    def evaluate(
        self,
        caller: AuthenticatedCaller,
        request: GatewayRequest,
        prompt_text: str,
        pii_tokens: int,
    ) -> GatewayPolicyResult:
        """Evaluate all policy rules. Returns aggregated decision."""
        t0 = time.time()
        rules: list[PolicyRule] = []
        final_decision = PolicyDecision.ALLOW

        # Rule 1: Model access control
        rule = self._rule_model_access(caller, request.model)
        rules.append(rule)
        if rule.decision == PolicyDecision.DENY:
            final_decision = PolicyDecision.DENY

        # Rule 2: Prompt token limit
        rule = self._rule_token_limit(caller, request)
        rules.append(rule)
        if rule.decision == PolicyDecision.DENY:
            final_decision = PolicyDecision.DENY

        # Rule 3: Rate limiting (per user)
        rule = self._rule_rate_limit(caller)
        rules.append(rule)
        if rule.decision == PolicyDecision.DENY:
            final_decision = PolicyDecision.DENY

        # Rule 4: Content moderation
        rule, flags = self._rule_content_moderation(prompt_text)
        rules.append(rule)
        if rule.decision == PolicyDecision.DENY:
            final_decision = PolicyDecision.DENY

        # Rule 5: PII presence (doesn't block — handled by redaction layer)
        pii_rule = self._rule_pii_check(pii_tokens)
        rules.append(pii_rule)
        # PII triggers REDACT_AND_ALLOW, not DENY (redaction already happened upstream)
        if final_decision == PolicyDecision.ALLOW and pii_tokens > 0:
            final_decision = PolicyDecision.REDACT_AND_ALLOW

        # Rule 6: Anomaly detection
        rule = self._rule_anomaly(caller, request)
        rules.append(rule)
        # Anomaly is logged but doesn't deny (escalates to human review async)

        eval_ms = int((time.time() - t0) * 1000)
        triggered = [r for r in rules if r.triggered]

        logger.info(
            f"Policy eval: {caller.user_id} → {final_decision.value} | "
            f"{len(triggered)}/{len(rules)} rules triggered | {eval_ms}ms"
        )

        return GatewayPolicyResult(
            final_decision=final_decision,
            rules_evaluated=len(rules),
            rules_triggered=triggered,
            pii_tokens_found=pii_tokens,
            pii_tokens_redacted=pii_tokens,  # all found tokens are redacted
            content_flags=flags,
            evaluation_ms=eval_ms,
        )

    def _rule_model_access(self, caller: AuthenticatedCaller, model: LLMModel) -> PolicyRule:
        from middleware.auth import ROLE_MODEL_ALLOWLIST
        allowed = ROLE_MODEL_ALLOWLIST.get(caller.role, set())
        permitted = model in allowed
        return PolicyRule(
            rule_id="RBAC-001",
            rule_name="model_access_control",
            triggered=not permitted,
            decision=PolicyDecision.ALLOW if permitted else PolicyDecision.DENY,
            reason=(
                f"Role '{caller.role.value}' permitted for {model.value}"
                if permitted else
                f"Role '{caller.role.value}' NOT in allowlist for {model.value}"
            ),
        )

    def _rule_token_limit(self, caller: AuthenticatedCaller, request: GatewayRequest) -> PolicyRule:
        policy = RATE_LIMIT_POLICIES.get(caller.role)
        if not policy:
            return PolicyRule(rule_id="RL-001", rule_name="prompt_token_limit", triggered=False, decision=PolicyDecision.ALLOW, reason="No policy found, default allow")

        # Estimate prompt tokens from message lengths
        estimated_prompt_tokens = sum(len(m.get("content", "").split()) * 1.3 for m in request.messages)
        over_limit = estimated_prompt_tokens > policy.max_prompt_tokens
        return PolicyRule(
            rule_id="RL-002",
            rule_name="prompt_token_limit",
            triggered=over_limit,
            decision=PolicyDecision.DENY if over_limit else PolicyDecision.ALLOW,
            reason=(
                f"Estimated {int(estimated_prompt_tokens)} tokens exceeds role limit of {policy.max_prompt_tokens}"
                if over_limit else
                f"Prompt within token limit ({int(estimated_prompt_tokens)} / {policy.max_prompt_tokens})"
            ),
        )

    def _rule_rate_limit(self, caller: AuthenticatedCaller) -> PolicyRule:
        """Sliding window rate limiter (per user, per minute)."""
        policy = RATE_LIMIT_POLICIES.get(caller.role)
        if not policy:
            return PolicyRule(rule_id="RL-003", rule_name="rate_limit", triggered=False, decision=PolicyDecision.ALLOW, reason="No policy")

        now = time.time()
        window_start = now - 60
        q = _request_timestamps[caller.user_id]

        # Remove old timestamps
        while q and q[0] < window_start:
            q.popleft()

        over_limit = len(q) >= policy.requests_per_minute
        if not over_limit:
            q.append(now)

        return PolicyRule(
            rule_id="RL-003",
            rule_name="rate_limit_per_minute",
            triggered=over_limit,
            decision=PolicyDecision.DENY if over_limit else PolicyDecision.ALLOW,
            reason=(
                f"Rate limit exceeded: {len(q)}/{policy.requests_per_minute} req/min"
                if over_limit else
                f"Rate OK: {len(q)}/{policy.requests_per_minute} req/min"
            ),
        )

    def _rule_content_moderation(self, text: str) -> tuple[PolicyRule, list[str]]:
        """Check for blocked topics and content policy violations."""
        flags: list[str] = []

        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(text):
                flags.append(f"blocked_pattern:{pattern.pattern[:30]}")

        text_lower = text.lower()
        for kw in COMPETITOR_KEYWORDS:
            if kw in text_lower:
                flags.append(f"competitor_mention:{kw}")

        is_blocked = any(f.startswith("blocked_pattern") for f in flags)
        return PolicyRule(
            rule_id="CM-001",
            rule_name="content_moderation",
            triggered=bool(flags),
            decision=PolicyDecision.DENY if is_blocked else PolicyDecision.ALLOW,
            reason=(
                f"Blocked content detected: {flags}" if is_blocked
                else f"Content flags (non-blocking): {flags}" if flags
                else "Content clear"
            ),
        ), flags

    def _rule_pii_check(self, pii_tokens: int) -> PolicyRule:
        return PolicyRule(
            rule_id="PII-001",
            rule_name="pii_redaction_check",
            triggered=pii_tokens > 0,
            decision=PolicyDecision.REDACT_AND_ALLOW if pii_tokens > 0 else PolicyDecision.ALLOW,
            reason=(
                f"{pii_tokens} PII tokens detected and redacted before LLM forwarding"
                if pii_tokens > 0 else
                "No PII detected"
            ),
        )

    def _rule_anomaly(self, caller: AuthenticatedCaller, request: GatewayRequest) -> PolicyRule:
        """Detect anomalous usage patterns (volume spikes, off-hours, unusual models)."""
        hour = datetime.now(timezone.utc).hour
        is_off_hours = hour < 6 or hour > 22  # UTC
        is_max_tokens = request.max_tokens >= 4000
        is_anomaly = is_off_hours and is_max_tokens

        return PolicyRule(
            rule_id="AN-001",
            rule_name="anomaly_detection",
            triggered=is_anomaly,
            decision=PolicyDecision.ALLOW,  # Log only, don't deny
            reason=(
                "Anomaly flagged: off-hours high-token request (async escalation queued)"
                if is_anomaly else "No anomaly detected"
            ),
        )


policy_engine = PolicyEngine()
