"""
ML Fraud Scorer — XGBoost gradient boosting model.

In production:
    - Model artifact stored in Azure ML registry
    - Features computed from claim + policy + claims history
    - SHAP values provided for explainability
    - Model versioned and registered; inference via Azure ML endpoint

This demo shows the feature engineering logic and scoring interface,
with a deterministic simulation that produces realistic scores.
"""

import logging
import math
import random
from typing import Any

from models.schemas import ClaimSubmission, ExtractedClaimData, FraudSignal

logger = logging.getLogger(__name__)

MODEL_VERSION = "xgboost_fraud_v2.3"


class MLFraudScorer:
    """
    Gradient boosting fraud probability scorer.
    
    Features used (in production, 47 total):
    
    Claim features:
    - estimated_loss_amount (log-scaled)
    - claim_type_encoded
    - days_since_incident
    - has_witnesses (bool)
    - injured_count
    - third_party_count
    - timeline_consistency_score
    - extraction_confidence
    - prior_claims_flag
    
    Policy features (from policy system):
    - policy_age_days
    - coverage_limit
    - premium_paid_ratio
    - recent_endorsements
    
    Behavioral features (from claims history):
    - claimant_claim_count_12mo
    - claimant_claim_frequency
    - attorney_involvement_flag
    """

    def __init__(self):
        # In production: load from Azure ML model registry
        # self.model = joblib.load("models/xgboost_fraud_v2.pkl")
        logger.info(f"ML Fraud Scorer initialized (version={MODEL_VERSION})")

    def score(
        self,
        extracted: ExtractedClaimData,
        submission: ClaimSubmission,
    ) -> dict:
        """
        Compute fraud probability score.
        Returns score (0-1), signals, and top contributing feature.
        """
        features = self._engineer_features(extracted, submission)
        score = self._simulate_model_inference(features)
        signals = self._generate_shap_signals(features, score)
        top_feature = max(features, key=lambda k: abs(features[k]) if isinstance(features[k], (int, float)) else 0)

        logger.info(
            f"ML score: {score:.4f} | top feature: {top_feature} "
            f"| model: {MODEL_VERSION}"
        )

        return {
            "score": score,
            "signals": signals,
            "features": features,
            "top_feature": top_feature,
            "model_version": MODEL_VERSION,
        }

    def _engineer_features(
        self,
        extracted: ExtractedClaimData,
        submission: ClaimSubmission,
    ) -> dict[str, Any]:
        """Feature engineering pipeline."""
        from datetime import datetime, date
        try:
            incident_dt = datetime.strptime(submission.incident_date, "%Y-%m-%d").date()
            days_since = (date.today() - incident_dt).days
        except (ValueError, TypeError):
            days_since = 0

        amount = extracted.estimated_loss_amount or 0
        log_amount = math.log1p(amount)

        claim_type_risk = {
            "auto": 0.45,
            "property": 0.30,
            "liability": 0.55,
            "workers_comp": 0.60,
            "commercial": 0.35,
        }

        timeline_score = {
            "consistent": 0.0,
            "inconsistent": 1.0,
            "unclear": 0.5,
        }.get(extracted.timeline_consistency, 0.5)

        return {
            "log_loss_amount": round(log_amount, 4),
            "claim_type_risk": claim_type_risk.get(submission.claim_type.value, 0.4),
            "days_since_incident": min(days_since, 365),
            "has_witnesses": int(bool(extracted.witnesses)),
            "injured_count": len(extracted.injured_parties),
            "third_party_count": len(extracted.third_parties_involved),
            "timeline_score": timeline_score,
            "extraction_confidence": extracted.extraction_confidence,
            "prior_claims_flag": int(extracted.prior_claims_mentioned),
            "is_weekend": int(
                datetime.strptime(submission.incident_date, "%Y-%m-%d").weekday() >= 5
                if submission.incident_date else False
            ),
        }

    def _simulate_model_inference(self, features: dict) -> float:
        """
        Simulate XGBoost inference.
        In production: return float(self.model.predict_proba([feature_vector])[0][1])
        """
        # Weighted feature contribution (mimics learned weights)
        score = 0.0
        score += features.get("timeline_score", 0) * 0.28
        score += features.get("prior_claims_flag", 0) * 0.15
        score += features.get("claim_type_risk", 0.4) * 0.20
        score += (1 - features.get("has_witnesses", 1)) * 0.12
        score += min(features.get("days_since_incident", 0) / 365, 1) * 0.10
        score += (1 - features.get("extraction_confidence", 0.8)) * 0.10
        score += features.get("is_weekend", 0) * 0.05

        # Add small noise to simulate model variance
        noise = random.gauss(0, 0.03)
        return round(min(1.0, max(0.0, score + noise)), 4)

    def _generate_shap_signals(self, features: dict, score: float) -> list[FraudSignal]:
        """
        In production, use SHAP to explain model predictions.
        Here we simulate the top contributing SHAP signals.
        """
        signals = []

        if features.get("timeline_score", 0) > 0.5 and score > 0.4:
            signals.append(FraudSignal(
                signal_type="ml_timeline_anomaly",
                description="ML model flags timeline inconsistency as primary fraud predictor for this claim profile.",
                severity="high",
                source="ml_model",
                confidence=min(score + 0.1, 1.0),
            ))

        if features.get("days_since_incident", 0) > 45 and score > 0.3:
            signals.append(FraudSignal(
                signal_type="ml_late_reporting_pattern",
                description="ML model identifies late-reporting pattern consistent with staged claim profiles.",
                severity="medium",
                source="ml_model",
                confidence=score,
            ))

        if features.get("claim_type_risk", 0) > 0.5 and score > 0.35:
            signals.append(FraudSignal(
                signal_type="ml_high_risk_claim_type",
                description=f"Claim type has elevated base fraud rate in historical training data.",
                severity="low",
                source="ml_model",
                confidence=0.65,
            ))

        return signals
