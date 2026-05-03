"""
Claims Triage Agentic Pipeline using LangGraph.

Pipeline stages:
  1. Document Ingestion & PII Redaction
  2. LLM Entity Extraction
  3. Rules-Based Fraud Signals
  4. ML Fraud Scoring (simulated)
  5. LLM Fraud Narrative
  6. Routing Decision
  7. Audit Logging
  8. HITL Escalation (if triggered)

Each node is independently testable and logged.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, TypedDict

from models.schemas import (
    AuditEntry, ClaimProcessingResult, ClaimSubmission,
    ClaimType, ExtractedClaimData, FraudAssessment,
    FraudRiskLevel, FraudSignal, RoutingDecision, RoutingQueue,
)
from services.llm_client import LLMClient
from services.pii_redactor import PIIRedactor
from services.fraud_rules import FraudRulesEngine
from services.ml_scorer import MLFraudScorer

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    submission: ClaimSubmission
    claim_id: str
    redacted_text: str
    extracted_data: ExtractedClaimData | None
    fraud_signals: list
    rules_score: float
    ml_score: float
    llm_fraud_score: float
    fraud_assessment: FraudAssessment | None
    routing_decision: RoutingDecision | None
    audit_trail: list
    pipeline_start_ms: float
    total_tokens: int
    total_cost: float
    errors: list


class ClaimsTriageAgent:
    """
    Orchestrates the end-to-end claims triage pipeline.
    
    In production this uses LangGraph's StateGraph for conditional
    branching, parallel node execution, and HITL interrupts.
    This implementation shows the full logic with the same interface.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.pii_redactor = PIIRedactor()
        self.rules_engine = FraudRulesEngine()
        self.ml_scorer = MLFraudScorer()

    async def run(self, submission: ClaimSubmission) -> ClaimProcessingResult:
        """Execute the full agentic pipeline."""
        state: PipelineState = {
            "submission": submission,
            "claim_id": str(uuid.uuid4()),
            "redacted_text": "",
            "extracted_data": None,
            "fraud_signals": [],
            "rules_score": 0.0,
            "ml_score": 0.0,
            "llm_fraud_score": 0.0,
            "fraud_assessment": None,
            "routing_decision": None,
            "audit_trail": [],
            "pipeline_start_ms": time.time() * 1000,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": [],
        }

        # Sequential pipeline (LangGraph would run parallelizable nodes concurrently)
        state = await self._node_ingest_and_redact(state)
        state = await self._node_extract_entities(state)
        state = await self._node_rules_fraud_check(state)
        state = await self._node_ml_fraud_score(state)
        state = await self._node_llm_fraud_analysis(state)
        state = await self._node_aggregate_fraud(state)
        state = await self._node_routing_decision(state)

        pipeline_latency = int(time.time() * 1000 - state["pipeline_start_ms"])

        return ClaimProcessingResult(
            claim_id=state["claim_id"],
            policy_number=submission.policy_number,
            claim_type=submission.claim_type,
            extracted_data=state["extracted_data"],
            fraud_assessment=state["fraud_assessment"],
            routing_decision=state["routing_decision"],
            pipeline_latency_ms=pipeline_latency,
            total_tokens_used=state["total_tokens"],
            estimated_cost_usd=state["total_cost"],
            audit_trail=state["audit_trail"],
        )

    async def _node_ingest_and_redact(self, state: PipelineState) -> PipelineState:
        """Stage 1: Ingest raw claim, redact PII before any LLM call."""
        t0 = time.time()
        raw_text = state["submission"].claim_text
        redacted = self.pii_redactor.redact(raw_text)
        state["redacted_text"] = redacted

        state["audit_trail"].append(AuditEntry(
            claim_id=state["claim_id"],
            stage="ingestion",
            action="pii_redaction_applied",
            actor="pii_redactor_service",
            input_hash=hashlib.sha256(raw_text.encode()).hexdigest(),
            output_summary=f"Redacted {self.pii_redactor.last_redaction_count} PII tokens. Text length: {len(redacted)} chars.",
            latency_ms=int((time.time() - t0) * 1000),
        ))
        logger.info(f"[{state['claim_id']}] Ingestion complete. PII tokens redacted: {self.pii_redactor.last_redaction_count}")
        return state

    async def _node_extract_entities(self, state: PipelineState) -> PipelineState:
        """Stage 2: LLM extracts structured entities from redacted claim text."""
        t0 = time.time()
        result = await self.llm.extract_claim_entities(
            redacted_text=state["redacted_text"],
            claim_type=state["submission"].claim_type,
            incident_date=state["submission"].incident_date,
        )
        state["extracted_data"] = result["data"]
        state["total_tokens"] += result["tokens"]
        state["total_cost"] += result["cost"]

        state["audit_trail"].append(AuditEntry(
            claim_id=state["claim_id"],
            stage="extraction",
            action="entity_extraction_completed",
            actor="azure_openai_gpt4o",
            input_hash=hashlib.sha256(state["redacted_text"].encode()).hexdigest(),
            output_summary=f"Extracted {len(state['extracted_data'].injured_parties)} injured parties, "
                           f"loss estimate: ${state['extracted_data'].estimated_loss_amount or 'unknown'}, "
                           f"timeline: {state['extracted_data'].timeline_consistency}",
            latency_ms=int((time.time() - t0) * 1000),
            model_used="gpt-4o",
            tokens_used=result["tokens"],
            cost_usd=result["cost"],
        ))
        logger.info(f"[{state['claim_id']}] Entity extraction complete. Confidence: {state['extracted_data'].extraction_confidence:.2f}")
        return state

    async def _node_rules_fraud_check(self, state: PipelineState) -> PipelineState:
        """Stage 3: Deterministic rules engine for known fraud patterns."""
        t0 = time.time()
        signals, score = self.rules_engine.evaluate(
            extracted=state["extracted_data"],
            submission=state["submission"],
        )
        state["fraud_signals"].extend(signals)
        state["rules_score"] = score

        state["audit_trail"].append(AuditEntry(
            claim_id=state["claim_id"],
            stage="fraud_scoring",
            action="rules_engine_evaluation",
            actor="fraud_rules_engine_v3",
            input_hash=hashlib.sha256(str(state["extracted_data"]).encode()).hexdigest(),
            output_summary=f"Rules score: {score:.3f}. Signals triggered: {len(signals)}. "
                           f"High-severity: {sum(1 for s in signals if s.severity == 'high')}",
            latency_ms=int((time.time() - t0) * 1000),
        ))
        return state

    async def _node_ml_fraud_score(self, state: PipelineState) -> PipelineState:
        """Stage 4: ML model fraud scoring (gradient boosting on claim features)."""
        t0 = time.time()
        ml_result = self.ml_scorer.score(
            extracted=state["extracted_data"],
            submission=state["submission"],
        )
        state["ml_score"] = ml_result["score"]
        state["fraud_signals"].extend(ml_result["signals"])

        state["audit_trail"].append(AuditEntry(
            claim_id=state["claim_id"],
            stage="fraud_scoring",
            action="ml_model_scoring",
            actor="xgboost_fraud_model_v2",
            input_hash=hashlib.sha256(str(state["extracted_data"]).encode()).hexdigest(),
            output_summary=f"ML fraud probability: {ml_result['score']:.3f}. "
                           f"Top feature: {ml_result.get('top_feature', 'n/a')}",
            latency_ms=int((time.time() - t0) * 1000),
            model_used="xgboost_fraud_v2",
        ))
        return state

    async def _node_llm_fraud_analysis(self, state: PipelineState) -> PipelineState:
        """Stage 5: LLM performs contextual fraud reasoning over all signals."""
        t0 = time.time()
        result = await self.llm.analyze_fraud_context(
            redacted_text=state["redacted_text"],
            extracted=state["extracted_data"],
            rules_signals=state["fraud_signals"],
            rules_score=state["rules_score"],
            ml_score=state["ml_score"],
        )
        state["llm_fraud_score"] = result["fraud_score"]
        state["fraud_signals"].extend(result["additional_signals"])
        state["total_tokens"] += result["tokens"]
        state["total_cost"] += result["cost"]

        state["audit_trail"].append(AuditEntry(
            claim_id=state["claim_id"],
            stage="fraud_scoring",
            action="llm_contextual_analysis",
            actor="azure_openai_gpt4o",
            input_hash=hashlib.sha256(state["redacted_text"].encode()).hexdigest(),
            output_summary=f"LLM fraud score: {result['fraud_score']:.3f}. "
                           f"Additional signals: {len(result['additional_signals'])}",
            latency_ms=int((time.time() - t0) * 1000),
            model_used="gpt-4o",
            tokens_used=result["tokens"],
            cost_usd=result["cost"],
        ))
        return state

    async def _node_aggregate_fraud(self, state: PipelineState) -> PipelineState:
        """Stage 6: Combine all fraud scores into final risk assessment."""
        # Weighted composite: rules 30%, ML 40%, LLM 30%
        composite = (
            state["rules_score"] * 0.30 +
            state["ml_score"] * 0.40 +
            state["llm_fraud_score"] * 0.30
        )

        if composite >= 0.75:
            risk_level = FraudRiskLevel.CRITICAL
        elif composite >= 0.50:
            risk_level = FraudRiskLevel.HIGH
        elif composite >= 0.25:
            risk_level = FraudRiskLevel.MEDIUM
        else:
            risk_level = FraudRiskLevel.LOW

        action_map = {
            FraudRiskLevel.LOW: "Proceed with standard review",
            FraudRiskLevel.MEDIUM: "Flag for specialist review with enhanced documentation",
            FraudRiskLevel.HIGH: "Route to fraud investigation unit; do not pay pending review",
            FraudRiskLevel.CRITICAL: "Immediate escalation to SIU; suspend claim processing",
        }

        narrative = await self.llm.generate_fraud_narrative(
            signals=state["fraud_signals"],
            composite_score=composite,
            risk_level=risk_level,
        )
        state["total_tokens"] += narrative["tokens"]
        state["total_cost"] += narrative["cost"]

        state["fraud_assessment"] = FraudAssessment(
            risk_level=risk_level,
            composite_score=round(composite, 4),
            signals=state["fraud_signals"],
            rules_score=round(state["rules_score"], 4),
            ml_score=round(state["ml_score"], 4),
            llm_score=round(state["llm_fraud_score"], 4),
            narrative=narrative["text"],
            recommended_action=action_map[risk_level],
        )

        state["audit_trail"].append(AuditEntry(
            claim_id=state["claim_id"],
            stage="fraud_scoring",
            action="fraud_assessment_finalized",
            actor="aggregation_service",
            input_hash=hashlib.sha256(str(composite).encode()).hexdigest(),
            output_summary=f"Final risk level: {risk_level.value}. Composite score: {composite:.3f}. "
                           f"Total signals: {len(state['fraud_signals'])}",
            latency_ms=0,
        ))
        return state

    async def _node_routing_decision(self, state: PipelineState) -> PipelineState:
        """Stage 7: Route to appropriate adjuster queue based on fraud + complexity."""
        t0 = time.time()
        assessment = state["fraud_assessment"]
        extracted = state["extracted_data"]

        # Determine complexity
        complexity_score = 0
        if extracted.injured_parties:
            complexity_score += len(extracted.injured_parties)
        if extracted.estimated_loss_amount and extracted.estimated_loss_amount > 50000:
            complexity_score += 2
        if extracted.third_parties_involved:
            complexity_score += 1
        if extracted.timeline_consistency == "inconsistent":
            complexity_score += 1

        if complexity_score >= 4:
            complexity = "complex"
        elif complexity_score >= 2:
            complexity = "moderate"
        else:
            complexity = "simple"

        # Routing logic
        if assessment.risk_level in [FraudRiskLevel.CRITICAL, FraudRiskLevel.HIGH]:
            queue = RoutingQueue.FRAUD_INVESTIGATION
            tier = "fraud_investigator"
            priority = "urgent"
            sla = 4
            requires_hitl = True
        elif assessment.risk_level == FraudRiskLevel.MEDIUM or complexity == "complex":
            queue = RoutingQueue.SPECIALIST
            tier = "specialist"
            priority = "high"
            sla = 24
            requires_hitl = complexity == "complex"
        elif complexity == "simple" and assessment.risk_level == FraudRiskLevel.LOW:
            queue = RoutingQueue.FAST_TRACK
            tier = "junior"
            priority = "normal"
            sla = 48
            requires_hitl = False
        else:
            queue = RoutingQueue.STANDARD
            tier = "senior"
            priority = "normal"
            sla = 72
            requires_hitl = False

        rationale = (
            f"Claim routed to {queue.value} based on fraud risk level '{assessment.risk_level.value}' "
            f"(composite score: {assessment.composite_score:.2f}) and complexity assessment '{complexity}'. "
            f"{'Human review required due to elevated fraud risk or claim complexity.' if requires_hitl else 'Eligible for automated processing.'}"
        )

        state["routing_decision"] = RoutingDecision(
            queue=queue,
            assigned_adjuster_tier=tier,
            priority=priority,
            estimated_complexity=complexity,
            routing_rationale=rationale,
            requires_human_review=requires_hitl,
            sla_hours=sla,
        )

        state["audit_trail"].append(AuditEntry(
            claim_id=state["claim_id"],
            stage="routing",
            action="routing_decision_made",
            actor="routing_engine",
            input_hash=hashlib.sha256(f"{assessment.risk_level}{complexity}".encode()).hexdigest(),
            output_summary=f"Routed to: {queue.value}. Tier: {tier}. Priority: {priority}. SLA: {sla}h. HITL: {requires_hitl}",
            latency_ms=int((time.time() - t0) * 1000),
        ))
        logger.info(f"[{state['claim_id']}] Routing complete → {queue.value} ({priority} priority)")
        return state
