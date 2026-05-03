"""
Regression Engine — compares current run against a baseline.

A regression is detected when a metric drops by more than
the configured threshold (e.g., factual_accuracy drops by >3%).

Output: RegressionReport with:
  - Per-metric deltas (improved / regressed / unchanged)
  - New failure cases (passing in baseline, failing now)
  - New pass cases (failing in baseline, passing now)
  - Deploy recommendation: "deploy" | "block" | "investigate"

In production:
  - Run on every PR merge to main
  - Baseline is the last PASS run on the main branch
  - Results posted as GitHub PR comment via Actions workflow
  - Block merge if gate_status == FAIL
"""

import logging
from datetime import datetime

from models.schemas import (
    EvalRunResult, GateStatus, MetricDelta,
    RegressionReport, REGRESSION_THRESHOLDS,
)

logger = logging.getLogger(__name__)

SIGNIFICANCE_THRESHOLD = 0.02  # Deltas below this are "unchanged"


class RegressionEngine:
    """Compares two eval runs and generates a regression report."""

    def compare(
        self,
        current: EvalRunResult,
        baseline: EvalRunResult,
    ) -> RegressionReport:
        """Generate a regression report comparing current to baseline."""
        if not current.summary or not baseline.summary:
            raise ValueError("Both runs must have summaries to compare")

        current_metrics = current.summary.metrics_summary
        baseline_metrics = baseline.summary.metrics_summary

        # Per-metric deltas
        all_metrics = set(current_metrics) | set(baseline_metrics)
        deltas: list[MetricDelta] = []
        regressions = []
        improvements = []

        for metric_name in sorted(all_metrics):
            curr_score = current_metrics.get(metric_name, {}).get("avg", 0.0)
            base_score = baseline_metrics.get(metric_name, {}).get("avg", 0.0)
            delta = curr_score - base_score
            delta_pct = (delta / max(base_score, 1e-6)) * 100

            if abs(delta) < SIGNIFICANCE_THRESHOLD:
                direction = "unchanged"
                significant = False
            elif delta > 0:
                direction = "improved"
                significant = True
                improvements.append(metric_name)
            else:
                direction = "regressed"
                regression_threshold = REGRESSION_THRESHOLDS.get(metric_name, 0.05)
                significant = abs(delta) >= regression_threshold
                if significant:
                    regressions.append(metric_name)

            deltas.append(MetricDelta(
                metric=metric_name,
                baseline_score=round(base_score, 4),
                current_score=round(curr_score, 4),
                delta=round(delta, 4),
                delta_pct=round(delta_pct, 2),
                direction=direction,
                significant=significant,
            ))

        # Case-level regression analysis
        baseline_passed = {r.case_id for r in baseline.case_results if r.passed}
        current_passed = {r.case_id for r in current.case_results if r.passed}

        new_failures = list(baseline_passed - current_passed)
        new_passes = list(current_passed - baseline_passed)

        # Gate and recommendation
        gate_status = GateStatus.FAIL if regressions else (
            GateStatus.WARN if new_failures else GateStatus.PASS
        )

        if gate_status == GateStatus.PASS:
            recommendation = "deploy"
            verdict = (
                f"✅ No regressions detected. Current run ({current.config.model}) "
                f"matches or improves on baseline ({baseline.config.model}). "
                f"Safe to deploy."
            )
        elif gate_status == GateStatus.WARN:
            recommendation = "investigate"
            verdict = (
                f"⚠️ No metric regressions but {len(new_failures)} test cases "
                f"that previously passed are now failing. Investigate before deploying."
            )
        else:
            recommendation = "block"
            verdict = (
                f"🚫 Regression detected in {len(regressions)} metric(s): "
                f"{', '.join(regressions)}. Deployment blocked. "
                f"Investigate prompt changes or model version differences."
            )

        logger.info(
            f"Regression: {gate_status.value} | "
            f"regressions={regressions} | "
            f"improvements={improvements} | "
            f"new_failures={len(new_failures)} | "
            f"new_passes={len(new_passes)} | "
            f"recommendation={recommendation}"
        )

        return RegressionReport(
            current_run_id=current.config.run_id,
            baseline_run_id=baseline.config.run_id,
            current_model=current.config.model,
            baseline_model=baseline.config.model,
            metric_deltas=deltas,
            regressions=regressions,
            improvements=improvements,
            new_failures=new_failures,
            new_passes=new_passes,
            gate_status=gate_status,
            gate_verdict=verdict,
            recommendation=recommendation,
        )
