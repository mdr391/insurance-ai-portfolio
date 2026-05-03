"""
Evaluation Runner — orchestrates full eval runs.

Responsibilities:
  1. Load dataset cases (or sample subset)
  2. For each case: call the model under test → collect response
  3. Run all configured metrics against each response
  4. Aggregate results into RunSummary
  5. Evaluate against gate thresholds
  6. If baseline_run_id set → generate RegressionReport
  7. Persist results to RunStore

Concurrency:
  Cases run concurrently (asyncio.gather) with semaphore-based
  rate limiting to respect provider API limits.

In production:
  - Job is submitted to Azure Container Jobs (serverless, parallel)
  - Progress streamed via WebSocket or Server-Sent Events
  - Results stored in Azure Blob Storage (JSON) + Postgres (metadata)
  - GitHub Actions calls POST /api/v1/runs to trigger from CI
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Optional, AsyncIterator

from datasets.insurance_cases import load_all_cases
from evaluators.metrics import evaluate_case, compute_overall_score
from models.schemas import (
    CaseResult, EvalCase, EvalDataset, EvalRunConfig, EvalRunResult,
    GateStatus, MetricName, RunStatus, RunSummary,
    GATE_THRESHOLDS, METRIC_WEIGHTS,
)
from runners.regression import RegressionEngine
from storage.run_store import run_store

logger = logging.getLogger(__name__)

DEFAULT_METRICS = [
    MetricName.FACTUAL_ACCURACY,
    MetricName.HALLUCINATION_RATE,
    MetricName.ANSWER_RELEVANCE,
    MetricName.INSURANCE_ACCURACY,
    MetricName.REGULATORY_SAFETY,
    MetricName.PII_LEAKAGE,
    MetricName.COMPLETENESS,
]

# Semaphore limits concurrent LLM calls (avoid API rate limits)
_CONCURRENCY_LIMIT = 5


class EvalRunner:
    """
    Orchestrates a full evaluation run.

    Usage:
        runner = EvalRunner()
        result = await runner.run(config)
    """

    def __init__(self):
        self.regression_engine = RegressionEngine()

    async def run(self, config: EvalRunConfig) -> EvalRunResult:
        """Execute a full evaluation run and return results."""
        run_result = EvalRunResult(config=config, status=RunStatus.RUNNING)
        run_store.save_run(run_result)  # Persist initial state

        logger.info(
            f"[{config.run_id}] Starting eval run | model={config.model} | "
            f"datasets={[d.value for d in config.datasets]} | "
            f"metrics={len(config.metrics or DEFAULT_METRICS)}"
        )

        t_start = time.time()

        try:
            # Load cases
            cases = load_all_cases(
                datasets=config.datasets if config.datasets else list(EvalDataset),
                sample_size=config.sample_size,
            )
            logger.info(f"[{config.run_id}] Loaded {len(cases)} eval cases")

            # Run cases concurrently
            metrics = config.metrics or DEFAULT_METRICS
            semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)
            tasks = [
                self._run_case(case, config, metrics, semaphore)
                for case in cases
            ]
            case_results = await asyncio.gather(*tasks, return_exceptions=False)

            run_result.case_results = case_results
            run_result.duration_seconds = round(time.time() - t_start, 2)
            run_result.status = RunStatus.COMPLETE
            run_result.completed_at = datetime.utcnow()

            # Aggregate summary
            run_result.summary = self._aggregate(case_results, metrics)

            logger.info(
                f"[{config.run_id}] Run complete | "
                f"pass_rate={run_result.summary.pass_rate:.1%} | "
                f"gate={run_result.summary.gate_status.value} | "
                f"cost=${run_result.summary.total_cost_usd:.4f} | "
                f"duration={run_result.duration_seconds:.1f}s"
            )

        except Exception as e:
            run_result.status = RunStatus.FAILED
            run_result.error = str(e)
            run_result.duration_seconds = round(time.time() - t_start, 2)
            logger.error(f"[{config.run_id}] Run failed: {e}", exc_info=True)

        finally:
            run_store.save_run(run_result)

        # Regression comparison if baseline specified
        if config.baseline_run_id and run_result.status == RunStatus.COMPLETE:
            baseline = run_store.get_run(config.baseline_run_id)
            if baseline:
                report = self.regression_engine.compare(run_result, baseline)
                run_store.save_regression_report(report)
                logger.info(
                    f"[{config.run_id}] Regression report: {report.gate_status.value} | "
                    f"regressions={report.regressions} | "
                    f"recommendation={report.recommendation}"
                )

        return run_result

    async def _run_case(
        self,
        case: EvalCase,
        config: EvalRunConfig,
        metrics: list[MetricName],
        semaphore: asyncio.Semaphore,
    ) -> CaseResult:
        """Run a single eval case: call model + evaluate all metrics."""
        async with semaphore:
            t0 = time.time()

            # Call model under test
            llm_response = await self._call_model(case, config)
            latency_ms = int((time.time() - t0) * 1000)

            # Evaluate all metrics
            metric_results = evaluate_case(case, llm_response["content"], metrics)
            overall = compute_overall_score(metric_results)
            all_required_passed = all(
                mr.passed for mr in metric_results
                if mr.metric in GATE_THRESHOLDS
            )

            return CaseResult(
                case_id=case.case_id,
                run_id=config.run_id,
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer=llm_response["content"],
                metrics=metric_results,
                overall_score=overall,
                passed=all_required_passed,
                latency_ms=latency_ms,
                prompt_tokens=llm_response["prompt_tokens"],
                completion_tokens=llm_response["completion_tokens"],
                cost_usd=llm_response["cost_usd"],
                model=config.model,
                dataset=case.dataset,
                difficulty=case.difficulty,
            )

    async def _call_model(self, case: EvalCase, config: EvalRunConfig) -> dict:
        """
        Call the model under test.

        Production:
            from openai import AzureOpenAI
            client = AzureOpenAI(...)
            t0 = time.time()
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": case.question},
                ],
                temperature=config.temperature,
                max_tokens=800,
            )
            return {
                "content": response.choices[0].message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "cost_usd": compute_cost(config.model, ...),
            }
        """
        await asyncio.sleep(random.uniform(0.05, 0.25))  # simulate API latency

        # Simulate model response quality based on model tier
        quality_map = {
            "gpt-4o":           0.87,
            "gpt-4o-mini":      0.74,
            "gpt-3.5-turbo":    0.66,
            "claude-3-5-sonnet": 0.85,
            "claude-3-haiku":   0.71,
        }
        base_quality = quality_map.get(config.model, 0.75)

        # Build a realistic simulated response using expected keywords
        keywords_to_include = case.expected_keywords[: int(len(case.expected_keywords) * base_quality)]
        answer_parts = [case.expected_answer[:200]]
        if keywords_to_include:
            answer_parts.append(f"Key considerations include: {', '.join(keywords_to_include[:3])}.")
        content = " ".join(answer_parts)

        prompt_tokens = len(case.question.split()) * 2 + 50
        completion_tokens = len(content.split()) * 2

        pricing = {
            "gpt-4o":           (0.005, 0.015),
            "gpt-4o-mini":      (0.00015, 0.0006),
            "gpt-3.5-turbo":    (0.0005, 0.0015),
            "claude-3-5-sonnet": (0.003, 0.015),
            "claude-3-haiku":   (0.00025, 0.00125),
        }
        ir, or_ = pricing.get(config.model, (0.005, 0.015))
        cost = (prompt_tokens / 1000) * ir + (completion_tokens / 1000) * or_

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }

    def _aggregate(
        self, case_results: list[CaseResult], metrics: list[MetricName]
    ) -> RunSummary:
        """Aggregate case results into run-level summary with gate evaluation."""
        total = len(case_results)
        passed = sum(1 for r in case_results if r.passed)
        pass_rate = passed / max(total, 1)

        # Metric-level aggregation
        metrics_summary: dict[str, dict] = {}
        for metric in metrics:
            scores = [
                mr.score
                for r in case_results
                for mr in r.metrics
                if mr.metric == metric
            ]
            if not scores:
                continue
            metric_passed = [
                mr.passed
                for r in case_results
                for mr in r.metrics
                if mr.metric == metric
            ]
            metrics_summary[metric.value] = {
                "avg": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "pass_rate": round(sum(metric_passed) / len(metric_passed), 4),
                "threshold": GATE_THRESHOLDS.get(metric, 0.70),
            }

        # By dataset
        by_dataset: dict[str, dict] = {}
        for ds in set(r.dataset for r in case_results):
            ds_cases = [r for r in case_results if r.dataset == ds]
            by_dataset[ds.value] = {
                "count": len(ds_cases),
                "pass_rate": round(sum(1 for r in ds_cases if r.passed) / len(ds_cases), 4),
                "avg_score": round(sum(r.overall_score for r in ds_cases) / len(ds_cases), 4),
            }

        # By difficulty
        by_difficulty: dict[str, dict] = {}
        for diff in ["easy", "medium", "hard"]:
            dc = [r for r in case_results if r.difficulty == diff]
            if dc:
                by_difficulty[diff] = {
                    "count": len(dc),
                    "pass_rate": round(sum(1 for r in dc if r.passed) / len(dc), 4),
                    "avg_score": round(sum(r.overall_score for r in dc) / len(dc), 4),
                }

        # Costs and latency
        total_cost = sum(r.cost_usd for r in case_results)
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in case_results)
        latencies = sorted(r.latency_ms for r in case_results)
        avg_latency = sum(latencies) / max(len(latencies), 1)
        p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0

        # Gate evaluation
        gate_failures = []
        for metric, threshold in GATE_THRESHOLDS.items():
            summary = metrics_summary.get(metric.value, {})
            avg = summary.get("avg", 0)
            if avg < threshold:
                gate_failures.append(
                    f"{metric.value}: {avg:.3f} < threshold {threshold:.2f}"
                )

        gate_status = GateStatus.PASS if not gate_failures else GateStatus.FAIL

        return RunSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            pass_rate=round(pass_rate, 4),
            avg_overall_score=round(sum(r.overall_score for r in case_results) / max(total, 1), 4),
            metrics_summary=metrics_summary,
            by_dataset=by_dataset,
            by_difficulty=by_difficulty,
            total_cost_usd=round(total_cost, 6),
            total_tokens=total_tokens,
            avg_latency_ms=round(avg_latency, 1),
            p99_latency_ms=float(p99_latency),
            gate_status=gate_status,
            gate_failures=gate_failures,
        )


eval_runner = EvalRunner()
