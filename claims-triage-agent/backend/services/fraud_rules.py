"""
Fraud Rules Engine — deterministic signal detection.

Rules are versioned, auditable, and independently testable.
In production these are loaded from a rules config store
(YAML/DB) and can be updated without code deploys.
"""

import logging
from datetime import datetime, date
from typing import Tuple

from models.schemas import ClaimSubmission, ExtractedClaimData, FraudSignal

logger = logging.getLogger(__name__)

RULES_VERSION = "v3.2.1"


class FraudRulesEngine:
    """
    Deterministic rules-based fraud signal detection.
    
    Rules are organized by category:
    - Temporal rules (timing anomalies)
    - Claim pattern rules (history-based)
    - Value rules (amount anomalies)
    - Narrative rules (content flags)
    - Policy rules (coverage-related)
    """

    def evaluate(
        self,
        extracted: ExtractedClaimData,
        submission: ClaimSubmission,
    ) -> Tuple[list[FraudSignal], float]:
        """
        Run all rules. Returns (signals, normalized_score).
        Score is a weighted average of triggered signal severities.
        """
        signals = []
        weight_map = {"low": 0.1, "medium": 0.35, "high": 0.7}

        signals += self._temporal_rules(extracted, submission)
        signals += self._value_rules(extracted, submission)
        signals += self._narrative_rules(extracted, submission)
        signals += self._policy_rules(extracted, submission)

        if not signals:
            return [], 0.0

        total_weight = sum(weight_map[s.severity] * s.confidence for s in signals)
        max_possible = sum(weight_map[s.severity] for s in signals)
        score = min(1.0, total_weight / max(max_possible, 1e-9))

        logger.info(
            f"Rules engine: {len(signals)} signals, score={score:.3f} "
            f"(rules_version={RULES_VERSION})"
        )
        return signals, round(score, 4)

    def _temporal_rules(
        self, extracted: ExtractedClaimData, submission: ClaimSubmission
    ) -> list[FraudSignal]:
        signals = []
        try:
            incident_dt = datetime.strptime(submission.incident_date, "%Y-%m-%d").date()
            days_since = (date.today() - incident_dt).days

            # Rule T1: Late reporting (>30 days without explanation)
            if days_since > 30:
                signals.append(FraudSignal(
                    signal_type="late_reporting",
                    description=f"Claim reported {days_since} days after incident. Late reporting can indicate staging.",
                    severity="medium" if days_since < 90 else "high",
                    source="rules_engine",
                    confidence=0.75,
                ))

            # Rule T2: Future-dated incident
            if incident_dt > date.today():
                signals.append(FraudSignal(
                    signal_type="future_dated_incident",
                    description="Incident date is in the future. Likely data entry error or fraudulent claim.",
                    severity="high",
                    source="rules_engine",
                    confidence=0.99,
                ))

            # Rule T3: Weekend/holiday incident (staging pattern)
            if incident_dt.weekday() >= 5:  # Saturday=5, Sunday=6
                signals.append(FraudSignal(
                    signal_type="weekend_incident",
                    description="Incident occurred on a weekend. Elevated staging pattern in this claim type.",
                    severity="low",
                    source="rules_engine",
                    confidence=0.45,
                ))
        except (ValueError, TypeError):
            pass
        return signals

    def _value_rules(
        self, extracted: ExtractedClaimData, submission: ClaimSubmission
    ) -> list[FraudSignal]:
        signals = []
        amount = extracted.estimated_loss_amount

        if amount is None:
            return signals

        # Rule V1: Very high claim amount
        if amount > 100_000:
            signals.append(FraudSignal(
                signal_type="high_value_claim",
                description=f"Claimed loss of ${amount:,.0f} exceeds $100K threshold. Enhanced review required.",
                severity="medium",
                source="rules_engine",
                confidence=0.80,
            ))

        # Rule V2: Round-number amounts (common in staged claims)
        if amount % 5000 == 0 and amount > 10000:
            signals.append(FraudSignal(
                signal_type="round_number_claim",
                description=f"Claimed amount (${amount:,.0f}) is a suspiciously round number.",
                severity="low",
                source="rules_engine",
                confidence=0.55,
            ))

        # Rule V3: Amount just below reporting threshold ($9,999 patterns)
        if 9000 <= amount <= 9999 or 19000 <= amount <= 19999:
            signals.append(FraudSignal(
                signal_type="threshold_avoidance",
                description=f"Claimed amount (${amount:,.0f}) falls just below a common reporting threshold.",
                severity="medium",
                source="rules_engine",
                confidence=0.70,
            ))

        return signals

    def _narrative_rules(
        self, extracted: ExtractedClaimData, submission: ClaimSubmission
    ) -> list[FraudSignal]:
        signals = []

        # Rule N1: Timeline inconsistency flagged by extraction
        if extracted.timeline_consistency == "inconsistent":
            signals.append(FraudSignal(
                signal_type="timeline_inconsistency",
                description="LLM extraction flagged internal timeline inconsistencies in the claim narrative.",
                severity="high",
                source="rules_engine",
                confidence=0.85,
            ))

        # Rule N2: Prior claims mentioned
        if extracted.prior_claims_mentioned:
            signals.append(FraudSignal(
                signal_type="prior_claims_mentioned",
                description="Claimant references prior claims in narrative. Frequency analysis required.",
                severity="low",
                source="rules_engine",
                confidence=0.60,
            ))

        # Rule N3: No witnesses for high-value claim
        if (
            not extracted.witnesses
            and extracted.estimated_loss_amount
            and extracted.estimated_loss_amount > 25000
        ):
            signals.append(FraudSignal(
                signal_type="no_witnesses_high_value",
                description="High-value claim with no witnesses reported. Common in staged incidents.",
                severity="medium",
                source="rules_engine",
                confidence=0.65,
            ))

        # Rule N4: Low extraction confidence
        if extracted.extraction_confidence < 0.60:
            signals.append(FraudSignal(
                signal_type="low_narrative_clarity",
                description=f"LLM extraction confidence is low ({extracted.extraction_confidence:.2f}). "
                            "Narrative may be vague or evasive.",
                severity="low",
                source="rules_engine",
                confidence=0.50,
            ))

        return signals

    def _policy_rules(
        self, extracted: ExtractedClaimData, submission: ClaimSubmission
    ) -> list[FraudSignal]:
        """
        In production, these rules query the policy system of record
        (e.g., Duck Creek, Guidewire) via internal API to check:
        - Policy age at time of claim
        - Coverage gaps
        - Recent policy changes
        - Claims frequency
        
        Demo uses heuristics only.
        """
        signals = []

        # Rule P1: Policy number format anomaly
        if submission.policy_number and len(submission.policy_number) < 6:
            signals.append(FraudSignal(
                signal_type="invalid_policy_format",
                description="Policy number format does not match expected pattern.",
                severity="high",
                source="rules_engine",
                confidence=0.95,
            ))

        return signals
