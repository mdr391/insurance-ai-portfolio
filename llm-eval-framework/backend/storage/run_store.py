"""
Run Store — persistence layer for evaluation runs and regression reports.

In production:
  - Run metadata → Azure PostgreSQL (queryable, indexed)
  - Case results → Azure Blob Storage as JSON (large payloads)
  - Regression reports → PostgreSQL + Blob
  - Access via SQLAlchemy ORM + async driver (asyncpg)

This implementation uses in-memory storage that fully simulates
the interface, pre-seeded with realistic historical run data
so the dashboard has data to display immediately.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional
import random

from models.schemas import (
    CaseResult, EvalDataset, EvalRunConfig, EvalRunResult,
    GateStatus, MetricName, RegressionReport, RunStatus, RunSummary,
    GATE_THRESHOLDS,
)

logger = logging.getLogger(__name__)


def _make_seed_summary(
    pass_rate: float,
    model: str,
    n_cases: int = 22,
    gate: GateStatus = GateStatus.PASS,
    gate_failures: list[str] | None = None,
    cost_mult: float = 1.0,
) -> RunSummary:
    """Build a realistic RunSummary for seeded historical runs."""
    pricing = {"gpt-4o": 0.010, "gpt-4o-mini": 0.0008, "gpt-3.5-turbo": 0.002, "claude-3-5-sonnet": 0.009, "claude-3-haiku": 0.0006}
    cost_per = pricing.get(model, 0.010) * cost_mult

    def ms(base_score: float) -> dict:
        s = round(min(1.0, base_score + random.gauss(0, 0.02)), 4)
        return {"avg": s, "min": round(s-0.08, 4), "max": round(min(1.0, s+0.06), 4), "pass_rate": round(min(1.0, pass_rate + random.gauss(0, 0.05)), 4), "threshold": 0.75}

    base = pass_rate
    return RunSummary(
        total_cases=n_cases, passed_cases=int(n_cases * pass_rate),
        failed_cases=n_cases - int(n_cases * pass_rate),
        pass_rate=round(pass_rate, 4),
        avg_overall_score=round(base * 0.92 + 0.05, 4),
        metrics_summary={
            MetricName.FACTUAL_ACCURACY.value:   ms(base * 0.95 + 0.04),
            MetricName.HALLUCINATION_RATE.value: ms(base * 0.90 + 0.08),
            MetricName.ANSWER_RELEVANCE.value:   ms(base * 0.98 + 0.01),
            MetricName.INSURANCE_ACCURACY.value: ms(base * 0.92 + 0.05),
            MetricName.REGULATORY_SAFETY.value:  ms(base * 0.97 + 0.02),
            MetricName.PII_LEAKAGE.value:        ms(0.98),
            MetricName.COMPLETENESS.value:       ms(base * 0.88 + 0.08),
        },
        by_dataset={
            EvalDataset.UNDERWRITING_QA.value:      {"count": 8, "pass_rate": round(min(1, pass_rate+0.03), 3), "avg_score": round(base+0.02, 3)},
            EvalDataset.CLAIMS_TRIAGE.value:         {"count": 3, "pass_rate": round(min(1, pass_rate-0.04), 3), "avg_score": round(base-0.02, 3)},
            EvalDataset.COVERAGE_REASONING.value:    {"count": 2, "pass_rate": round(min(1, pass_rate+0.01), 3), "avg_score": round(base+0.01, 3)},
            EvalDataset.FRAUD_SIGNALS.value:         {"count": 2, "pass_rate": round(min(1, pass_rate-0.02), 3), "avg_score": round(base-0.01, 3)},
            EvalDataset.REGULATORY_COMPLIANCE.value: {"count": 2, "pass_rate": round(min(1, pass_rate+0.05), 3), "avg_score": round(base+0.04, 3)},
            EvalDataset.POLICY_EXTRACTION.value:     {"count": 1, "pass_rate": round(min(1, pass_rate), 3),    "avg_score": round(base, 3)},
            EvalDataset.CUSTOMER_COMMS.value:        {"count": 1, "pass_rate": round(min(1, pass_rate-0.01), 3),"avg_score": round(base-0.01, 3)},
        },
        by_difficulty={
            "easy":   {"count": 7, "pass_rate": round(min(1, pass_rate+0.12), 3), "avg_score": round(base+0.09, 3)},
            "medium": {"count": 11, "pass_rate": round(min(1, pass_rate), 3),     "avg_score": round(base, 3)},
            "hard":   {"count": 4, "pass_rate": round(max(0, pass_rate-0.15), 3), "avg_score": round(max(0, base-0.12), 3)},
        },
        total_cost_usd=round(n_cases * cost_per * random.uniform(0.9, 1.1), 6),
        total_tokens=n_cases * random.randint(800, 1400),
        avg_latency_ms=round(random.uniform(180, 650), 1),
        p99_latency_ms=round(random.uniform(800, 1800), 1),
        gate_status=gate,
        gate_failures=gate_failures or [],
    )


def _seed_runs() -> dict[str, EvalRunResult]:
    """Create realistic historical run data for demo."""
    runs = {}
    now = datetime.utcnow()

    seed_data = [
        # (model, days_ago, pass_rate, gate, name, triggered_by)
        ("gpt-4o",            14, 0.91, GateStatus.PASS, "Baseline — GPT-4o v1",   "manual"),
        ("gpt-4o",            10, 0.88, GateStatus.PASS, "Weekly scheduled run",   "scheduled"),
        ("gpt-4o-mini",        9, 0.76, GateStatus.PASS, "GPT-4o-mini evaluation", "manual"),
        ("gpt-3.5-turbo",      8, 0.64, GateStatus.FAIL, "GPT-3.5 regression test","ci"),
        ("claude-3-5-sonnet",  6, 0.89, GateStatus.PASS, "Claude Sonnet baseline", "manual"),
        ("gpt-4o",             5, 0.93, GateStatus.PASS, "Pre-release validation", "ci"),
        ("gpt-4o",             3, 0.87, GateStatus.PASS, "Weekly scheduled run",   "scheduled"),
        ("gpt-4o-mini",        2, 0.79, GateStatus.PASS, "Mini — cost experiment", "manual"),
        ("claude-3-haiku",     1, 0.68, GateStatus.FAIL, "Haiku — fast eval",      "ci"),
        ("gpt-4o",             0, 0.90, GateStatus.PASS, "Latest production run",  "ci"),
    ]

    run_ids = []
    for model, days_ago, pass_rate, gate, name, triggered in seed_data:
        rid = str(uuid.uuid4())
        run_ids.append(rid)
        gate_failures = []
        if gate == GateStatus.FAIL:
            gate_failures = [f"factual_accuracy: {pass_rate:.2f} < threshold 0.80"]

        started = now - timedelta(days=days_ago, hours=random.randint(0, 8))
        duration = random.uniform(45, 180)

        config = EvalRunConfig(
            run_id=rid,
            name=name,
            model=model,
            datasets=list(EvalDataset),
            metrics=list(MetricName)[:7],
            triggered_by=triggered,
        )
        summary = _make_seed_summary(pass_rate, model, gate=gate, gate_failures=gate_failures)
        result = EvalRunResult(
            config=config,
            status=RunStatus.COMPLETE,
            started_at=started,
            completed_at=started + timedelta(seconds=duration),
            duration_seconds=round(duration, 1),
            case_results=[],  # Omitted for seed data (too large to generate)
            summary=summary,
        )
        runs[rid] = result

    return runs


class RunStore:
    """In-memory run persistence with pre-seeded historical data."""

    def __init__(self):
        self._runs: dict[str, EvalRunResult] = _seed_runs()
        self._regression_reports: dict[str, RegressionReport] = {}

    def save_run(self, run: EvalRunResult) -> None:
        self._runs[run.config.run_id] = run
        logger.debug(f"Saved run {run.config.run_id} (status={run.status.value})")

    def get_run(self, run_id: str) -> Optional[EvalRunResult]:
        return self._runs.get(run_id)

    def list_runs(self, limit: int = 50) -> list[EvalRunResult]:
        runs = list(self._runs.values())
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    def save_regression_report(self, report: RegressionReport) -> None:
        self._regression_reports[report.report_id] = report

    def get_regression_report(self, report_id: str) -> Optional[RegressionReport]:
        return self._regression_reports.get(report_id)

    def list_regression_reports(self) -> list[RegressionReport]:
        return sorted(
            self._regression_reports.values(),
            key=lambda r: r.generated_at,
            reverse=True,
        )

    def get_trend_data(self, metric: str, model: str | None = None) -> list[dict]:
        """Historical trend for a metric over time."""
        runs = self.list_runs(limit=30)
        trend = []
        for r in reversed(runs):
            if r.status != RunStatus.COMPLETE or not r.summary:
                continue
            if model and r.config.model != model:
                continue
            ms = r.summary.metrics_summary.get(metric, {})
            if ms:
                trend.append({
                    "run_id": r.config.run_id,
                    "model": r.config.model,
                    "date": r.started_at.isoformat(),
                    "score": ms.get("avg", 0),
                    "gate": r.summary.gate_status.value,
                })
        return trend

    def model_comparison(self) -> list[dict]:
        """Latest run stats per model."""
        by_model: dict[str, EvalRunResult] = {}
        for run in self.list_runs():
            m = run.config.model
            if m not in by_model and run.status == RunStatus.COMPLETE:
                by_model[m] = run
        result = []
        for model, run in by_model.items():
            if not run.summary:
                continue
            result.append({
                "model": model,
                "pass_rate": run.summary.pass_rate,
                "avg_score": run.summary.avg_overall_score,
                "gate": run.summary.gate_status.value,
                "cost_usd": run.summary.total_cost_usd,
                "avg_latency_ms": run.summary.avg_latency_ms,
                "run_id": run.config.run_id,
                "date": run.started_at.isoformat(),
            })
        result.sort(key=lambda x: x["avg_score"], reverse=True)
        return result


run_store = RunStore()
