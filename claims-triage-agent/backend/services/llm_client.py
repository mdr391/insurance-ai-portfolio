"""
LLM Client Service — Azure OpenAI integration.

All prompts use structured outputs (JSON mode) to ensure
parseable, schema-validated responses. PII-free text only
enters here (redaction happens upstream).
"""

import json
import logging
import os
import random
import time
from typing import Any

from models.schemas import (
    ClaimType, ExtractedClaimData, FraudAssessment,
    FraudRiskLevel, FraudSignal,
)

logger = logging.getLogger(__name__)

# In production: from openai import AzureOpenAI
# For demo: we simulate realistic LLM responses

EXTRACTION_SYSTEM_PROMPT = """You are an insurance claim analysis assistant working for an enterprise insurer.
Your task is to extract structured information from claim descriptions.
PII has already been redacted (names appear as [NAME], phone as [PHONE], etc.)

Respond ONLY with valid JSON matching this schema:
{
  "incident_description": str,
  "location": str | null,
  "estimated_loss_amount": float | null,
  "injured_parties": [str],
  "witnesses": [str],
  "third_parties_involved": [str],
  "vehicle_info": object | null,
  "property_info": object | null,
  "prior_claims_mentioned": bool,
  "timeline_consistency": "consistent" | "inconsistent" | "unclear",
  "extraction_confidence": float (0-1)
}

Be precise. If a value is unclear, use null. Flag timeline inconsistencies."""

FRAUD_ANALYSIS_PROMPT = """You are a senior insurance fraud analyst.
Review the claim data, rules engine signals, and ML score provided.
Identify any additional fraud indicators not caught by the rules engine.
Focus on: narrative inconsistencies, implausible details, suspicious patterns.

Respond ONLY with valid JSON:
{
  "fraud_score": float (0-1),
  "additional_signals": [
    {
      "signal_type": str,
      "description": str,
      "severity": "low"|"medium"|"high",
      "source": "llm_analysis",
      "confidence": float
    }
  ]
}"""

NARRATIVE_PROMPT = """You are a fraud assessment report writer for an insurance SIU team.
Write a concise, professional narrative (3-4 sentences) explaining the fraud risk assessment.
Be factual and evidence-based. Do not use PII. This will be read by adjusters and investigators."""


class LLMClient:
    """
    Wraps Azure OpenAI API calls with retry logic, cost tracking,
    and structured output parsing.
    
    In production, replace _simulate_* methods with actual API calls:
    
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01"
        )
    """

    # GPT-4o pricing (per 1K tokens, as of 2024)
    INPUT_COST_PER_1K = 0.005
    OUTPUT_COST_PER_1K = 0.015

    def __init__(self):
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
        # self.client = AzureOpenAI(...)  # production

    async def extract_claim_entities(
        self,
        redacted_text: str,
        claim_type: ClaimType,
        incident_date: str,
    ) -> dict:
        """Extract structured entities from redacted claim text."""
        tokens_used = len(redacted_text.split()) * 2  # approximation
        cost = (tokens_used / 1000) * self.INPUT_COST_PER_1K

        # Production call:
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        #         {"role": "user", "content": f"Claim type: {claim_type}\nDate: {incident_date}\n\n{redacted_text}"}
        #     ],
        #     response_format={"type": "json_object"},
        #     temperature=0.1,
        # )
        # extracted_json = json.loads(response.choices[0].message.content)

        # Simulated realistic extraction:
        extracted_json = self._simulate_extraction(redacted_text, claim_type)

        return {
            "data": ExtractedClaimData(**extracted_json),
            "tokens": tokens_used,
            "cost": cost,
        }

    async def analyze_fraud_context(
        self,
        redacted_text: str,
        extracted: ExtractedClaimData,
        rules_signals: list,
        rules_score: float,
        ml_score: float,
    ) -> dict:
        """LLM contextual fraud analysis."""
        context = json.dumps({
            "claim_summary": extracted.incident_description,
            "timeline_consistency": extracted.timeline_consistency,
            "prior_claims_mentioned": extracted.prior_claims_mentioned,
            "rules_score": rules_score,
            "ml_score": ml_score,
            "rules_signals": [s.model_dump() for s in rules_signals],
        }, indent=2)

        tokens_used = len(context.split()) * 3
        cost = (tokens_used / 1000) * (self.INPUT_COST_PER_1K + self.OUTPUT_COST_PER_1K)

        result = self._simulate_llm_fraud(rules_score, ml_score, extracted)

        additional_signals = [
            FraudSignal(**s) for s in result.get("additional_signals", [])
        ]

        return {
            "fraud_score": result["fraud_score"],
            "additional_signals": additional_signals,
            "tokens": tokens_used,
            "cost": cost,
        }

    async def generate_fraud_narrative(
        self,
        signals: list,
        composite_score: float,
        risk_level: FraudRiskLevel,
    ) -> dict:
        """Generate plain-English fraud assessment narrative."""
        tokens_used = 350
        cost = (tokens_used / 1000) * self.OUTPUT_COST_PER_1K

        narrative = self._simulate_narrative(composite_score, risk_level, signals)

        return {"text": narrative, "tokens": tokens_used, "cost": cost}

    # ─── Simulation helpers (replace with real API calls in production) ────

    def _simulate_extraction(self, text: str, claim_type: ClaimType) -> dict:
        text_lower = text.lower()
        has_injury = any(w in text_lower for w in ["injur", "hurt", "hospitali", "medical", "pain"])
        has_vehicle = any(w in text_lower for w in ["car", "truck", "vehicle", "collision", "accident", "rear"])
        has_property = any(w in text_lower for w in ["house", "roof", "water", "flood", "fire", "damage"])
        amount_hint = 45000 if "totaled" in text_lower else (8500 if has_injury else 3200)
        inconsistent = any(w in text_lower for w in ["but", "however", "although", "strange", "weird"])

        return {
            "incident_description": text[:300] + ("..." if len(text) > 300 else ""),
            "location": "Highway 95, Exit 12" if has_vehicle else "Residential property",
            "estimated_loss_amount": float(amount_hint),
            "injured_parties": ["[CLAIMANT]"] if has_injury else [],
            "witnesses": ["[WITNESS_1]"] if random.random() > 0.5 else [],
            "third_parties_involved": ["[THIRD_PARTY_1]"] if has_vehicle else [],
            "vehicle_info": {"make": "Honda", "year": 2019, "vin_partial": "***XY42"} if has_vehicle else None,
            "property_info": {"type": "single_family", "year_built": 1987} if has_property else None,
            "prior_claims_mentioned": "previous" in text_lower or "before" in text_lower,
            "timeline_consistency": "inconsistent" if inconsistent else "consistent",
            "extraction_confidence": round(random.uniform(0.78, 0.96), 3),
        }

    def _simulate_llm_fraud(
        self, rules_score: float, ml_score: float, extracted: ExtractedClaimData
    ) -> dict:
        base = (rules_score + ml_score) / 2
        noise = random.uniform(-0.05, 0.08)
        llm_score = min(1.0, max(0.0, base + noise))

        signals = []
        if extracted.timeline_consistency == "inconsistent":
            signals.append({
                "signal_type": "narrative_inconsistency",
                "description": "Claim narrative contains temporal contradictions that warrant closer review.",
                "severity": "high",
                "source": "llm_analysis",
                "confidence": 0.82,
            })
        if extracted.prior_claims_mentioned:
            signals.append({
                "signal_type": "prior_claims_pattern",
                "description": "Claimant references previous claims; frequency analysis recommended.",
                "severity": "medium",
                "source": "llm_analysis",
                "confidence": 0.71,
            })

        return {"fraud_score": round(llm_score, 4), "additional_signals": signals}

    def _simulate_narrative(
        self, score: float, risk_level: FraudRiskLevel, signals: list
    ) -> str:
        signal_count = len(signals)
        high_signals = [s for s in signals if hasattr(s, 'severity') and s.severity == 'high']

        narratives = {
            FraudRiskLevel.LOW: (
                f"This claim presents a low fraud risk profile with a composite score of {score:.2f}. "
                f"The claim narrative is internally consistent and aligns with the reported incident details. "
                f"Standard verification procedures are sufficient for processing. "
                f"No significant fraud indicators were identified across rules, ML, or contextual analysis."
            ),
            FraudRiskLevel.MEDIUM: (
                f"This claim exhibits moderate fraud indicators with a composite score of {score:.2f}, "
                f"triggering {signal_count} signals across the detection pipeline. "
                f"Enhanced documentation review and supervisor sign-off are recommended before settlement. "
                f"Key areas of concern include policy history patterns and claim circumstance details."
            ),
            FraudRiskLevel.HIGH: (
                f"This claim presents elevated fraud risk (score: {score:.2f}) with {signal_count} signals detected, "
                f"including {len(high_signals)} high-severity indicator(s). "
                f"Routing to the fraud investigation unit is required; payment should be suspended pending review. "
                f"A full investigation including recorded statement and field inspection is recommended."
            ),
            FraudRiskLevel.CRITICAL: (
                f"CRITICAL fraud risk identified (score: {score:.2f}). This claim exhibits {signal_count} fraud signals "
                f"with {len(high_signals)} classified as high-severity. "
                f"Immediate escalation to the Special Investigations Unit is mandatory. "
                f"Do not communicate settlement intent to claimant pending SIU review and potential law enforcement referral."
            ),
        }
        return narratives.get(risk_level, narratives[FraudRiskLevel.MEDIUM])
