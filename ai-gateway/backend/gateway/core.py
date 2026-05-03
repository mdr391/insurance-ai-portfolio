"""
Gateway Core — orchestrates the full request lifecycle.

Every LLM call through the gateway passes through this sequence:

  ┌─────────────────────────────────────────┐
  │  1. RECEIVE & LOG                        │
  │     Assign request_id, log arrival       │
  ├─────────────────────────────────────────┤
  │  2. AUTHENTICATE                         │
  │     Verify JWT, extract caller identity  │
  ├─────────────────────────────────────────┤
  │  3. REDACT PII                           │
  │     Scan prompt, replace PII tokens      │
  ├─────────────────────────────────────────┤
  │  4. EVALUATE POLICY                      │
  │     RBAC + rate limit + content check    │
  ├─────────────────────────────────────────┤
  │  5. FORWARD TO LLM (if ALLOW/REDACT)    │
  │     Call upstream provider               │
  ├─────────────────────────────────────────┤
  │  6. SCAN OUTPUT                          │
  │     Check completion for reconstructed PII│
  ├─────────────────────────────────────────┤
  │  7. AUDIT & RESPOND                      │
  │     Write cost/token event, return       │
  └─────────────────────────────────────────┘
"""

import hashlib
import logging
import random
import time
import uuid
from typing import Optional

from fastapi import HTTPException

from middleware.auth import auth_middleware
from middleware.pii import pii_service
from middleware.policy import policy_engine
from models.schemas import (
    AuditEventType, AuthenticatedCaller, GatewayPolicyResult,
    GatewayRequest, GatewayResponse, LLMModel,
    PolicyDecision, TokenUsage,
)
from services.audit import audit_logger, compute_cost

logger = logging.getLogger(__name__)


class AIGateway:
    """
    The governed AI gateway — single entry point for all LLM calls
    within the enterprise.

    Replaces: direct calls to OpenAI / Anthropic APIs from application code.
    All teams route through this gateway to get consistent:
      - Authentication & authorization
      - PII protection
      - Audit logging
      - Cost tracking & chargeback
      - Rate limiting
      - Content moderation
    """

    async def handle(
        self,
        request: GatewayRequest,
        authorization: Optional[str],
    ) -> GatewayResponse:
        request_id = str(uuid.uuid4())
        t_total_start = time.time()

        # ── 1. RECEIVE ────────────────────────────────────────────────────
        logger.info(f"[{request_id}] Request received | model={request.model}")

        # ── 2. AUTHENTICATE ───────────────────────────────────────────────
        try:
            caller = auth_middleware.verify_token(authorization)
        except HTTPException as e:
            logger.warning(f"[{request_id}] Auth failed: {e.detail}")
            raise

        audit_logger.log_request_received(request_id, caller)
        audit_logger.log_auth_verified(request_id, caller)

        # ── 3. REDACT PII ─────────────────────────────────────────────────
        # Extract full prompt text for scanning
        full_prompt = "\n".join(
            m.get("content", "") for m in request.messages
        )
        redaction = pii_service.redact_prompt(full_prompt)

        if redaction.was_redacted:
            # Rebuild messages with redacted content
            redacted_messages = []
            remaining = redaction.redacted_text
            for msg in request.messages:
                original = msg.get("content", "")
                redacted = pii_service.redact_prompt(original).redacted_text
                redacted_messages.append({**msg, "content": redacted})
            request = request.model_copy(update={"messages": redacted_messages})

            audit_logger.log_pii_redaction(
                request_id=request_id,
                caller=caller,
                prompt_hash=redaction.original_hash,
                pii_count=redaction.tokens_redacted,
                entity_types=redaction.entity_types_found,
            )

        # ── 4. EVALUATE POLICY ────────────────────────────────────────────
        policy_result = policy_engine.evaluate(
            caller=caller,
            request=request,
            prompt_text=full_prompt,
            pii_tokens=redaction.tokens_redacted,
        )
        audit_logger.log_policy_decision(request_id, caller, policy_result)

        if policy_result.final_decision == PolicyDecision.DENY:
            denied_rule = next(
                (r for r in policy_result.rules_triggered if r.decision == PolicyDecision.DENY),
                None,
            )
            reason = denied_rule.reason if denied_rule else "Policy denial"
            rule_id = denied_rule.rule_id if denied_rule else "UNKNOWN"
            audit_logger.log_denied(request_id, caller, reason, rule_id)
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "request_denied",
                    "reason": reason,
                    "rule_id": rule_id,
                    "request_id": request_id,
                },
            )

        # ── 5. FORWARD TO LLM ─────────────────────────────────────────────
        t_llm_start = time.time()
        llm_response = await self._call_llm(request)
        llm_latency_ms = int((time.time() - t_llm_start) * 1000)

        # ── 6. SCAN OUTPUT ────────────────────────────────────────────────
        cleaned_content, output_pii_count = pii_service.scan_completion(llm_response["content"])
        if output_pii_count > 0:
            logger.warning(f"[{request_id}] Output scan caught {output_pii_count} PII tokens")

        # ── 7. AUDIT & RESPOND ────────────────────────────────────────────
        prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()
        audit_logger.log_llm_call(
            request_id=request_id,
            caller=caller,
            model=request.model.value,
            prompt_hash=prompt_hash,
            prompt_tokens=llm_response["prompt_tokens"],
            completion_tokens=llm_response["completion_tokens"],
            latency_ms=llm_latency_ms,
        )

        total_ms = int((time.time() - t_total_start) * 1000)
        gateway_overhead_ms = total_ms - llm_latency_ms

        cost = compute_cost(
            request.model.value,
            llm_response["prompt_tokens"],
            llm_response["completion_tokens"],
        )

        usage = TokenUsage(
            prompt_tokens=llm_response["prompt_tokens"],
            completion_tokens=llm_response["completion_tokens"],
            total_tokens=llm_response["prompt_tokens"] + llm_response["completion_tokens"],
            cost_usd=cost,
        )

        logger.info(
            f"[{request_id}] Complete | {request.model.value} | "
            f"{usage.total_tokens} tokens | ${cost:.5f} | "
            f"{total_ms}ms total ({gateway_overhead_ms}ms gateway overhead)"
        )

        return GatewayResponse(
            request_id=request_id,
            model=request.model.value,
            content=cleaned_content,
            usage=usage,
            pii_was_redacted=redaction.was_redacted,
            redaction_count=redaction.tokens_redacted + output_pii_count,
            policy_decision=policy_result.final_decision,
            gateway_latency_ms=gateway_overhead_ms,
            upstream_latency_ms=llm_latency_ms,
            total_latency_ms=total_ms,
            governance_headers={
                "X-Gateway-Request-ID":    request_id,
                "X-Gateway-Policy":        policy_result.final_decision.value,
                "X-Gateway-PII-Redacted":  str(redaction.tokens_redacted),
                "X-Gateway-Model":         request.model.value,
                "X-Gateway-Cost-USD":      str(cost),
                "X-Gateway-Team":          caller.team.value,
            },
        )

    async def _call_llm(self, request: GatewayRequest) -> dict:
        """
        Call the upstream LLM provider.

        Production (Azure OpenAI):
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-02-01",
            )
            response = client.chat.completions.create(
                model=request.model.value,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return {
                "content": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        Production (Anthropic via gateway):
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            ...
        """
        await _simulate_latency(request.model)

        prompt_words = sum(len(m.get("content", "").split()) for m in request.messages)
        prompt_tokens = int(prompt_words * 1.35)

        simulated_responses = {
            LLMModel.GPT4O: "Based on the underwriting guidelines, I can confirm that the maximum TIV for this risk class is $50M without facultative reinsurance. The account's loss history shows favorable trends with a 3-year combined ratio of 87.4%. I recommend offering coverage with standard terms and a 5% wind/hail deductible given the coastal exposure.",
            LLMModel.GPT4O_MINI: "The claim appears to fall within standard parameters. The reported damage is consistent with the reported incident date and location. I recommend routing to the standard adjuster queue for processing.",
            LLMModel.GPT35_TURBO: "Based on the provided information, the classification code appears correct. The estimated premium at class rates is within acceptable bounds.",
            LLMModel.CLAUDE_SONNET: "After reviewing the submitted documentation, the exposure profile aligns with a preferred risk tier. The account's safety programs and loss control measures warrant a schedule credit of 8-12% off manual rates.",
            LLMModel.CLAUDE_HAIKU: "Acknowledged. The request has been processed and the relevant policy sections have been identified for your review.",
        }
        content = simulated_responses.get(request.model, "Response generated successfully.")
        completion_tokens = len(content.split()) * 2

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


async def _simulate_latency(model: LLMModel) -> None:
    """Simulate realistic LLM provider latency."""
    import asyncio
    latencies = {
        LLMModel.GPT4O: (600, 1400),
        LLMModel.GPT4O_MINI: (200, 600),
        LLMModel.GPT35_TURBO: (150, 450),
        LLMModel.CLAUDE_SONNET: (700, 1600),
        LLMModel.CLAUDE_HAIKU: (100, 350),
    }
    lo, hi = latencies.get(model, (400, 1000))
    await asyncio.sleep(random.uniform(lo, hi) / 1000)


ai_gateway = AIGateway()
