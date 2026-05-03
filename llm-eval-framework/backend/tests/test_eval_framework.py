"""
Test suite — LLM Evaluation & Regression Framework.

Tests cover:
  - Dataset loading and case integrity
  - Individual metric evaluators (string-match, rule-based, LLM-judge)
  - Gate threshold evaluation
  - Regression engine (improve / regress / unchanged detection)
  - Run aggregation accuracy
  - Case-level result structure

Run: pytest tests/ -v --cov=. --cov-report=term-missing
"""

import asyncio
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.insurance_cases import load_all_cases, load_dataset, dataset_stats, ALL_DATASETS
from evaluators.metrics import (
    KeywordCoverageEvaluator, ForbiddenPhraseEvaluator,
    PIILeakageEvaluator, CompletenessEvaluator, ToxicityEvaluator,
    FactualAccuracyEvaluator, InsuranceAccuracyEvaluator,
    evaluate_case, compute_overall_score,
)
from models.schemas import (
    EvalCase, EvalDataset, EvalRunConfig, GateStatus,
    MetricName, GATE_THRESHOLDS, METRIC_WEIGHTS,
)
from runners.eval_runner import EvalRunner
from runners.regression import RegressionEngine
from storage.run_store import RunStore, _make_seed_summary


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def uw_case():
    return EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What is the maximum TIV for habitational risks?",
        expected_answer="The maximum single-location TIV is $50M without reinsurance approval.",
        expected_keywords=["50m", "50 million", "reinsurance"],
        forbidden_phrases=["100m", "unlimited"],
        difficulty="easy",
        category="property_limits",
    )

@pytest.fixture
def hard_case():
    return EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What ransomware sublimit applies to healthcare?",
        expected_answer="Healthcare accounts have a maximum ransomware sublimit of $5M, with 2x standard retention.",
        expected_keywords=["5m", "5 million", "healthcare", "sublimit", "2x"],
        forbidden_phrases=["10m", "full limit", "no sublimit"],
        difficulty="hard",
        category="cyber",
    )


# ─── Dataset Tests ────────────────────────────────────────────────────────────

class TestDatasets:
    def test_all_datasets_have_cases(self):
        for ds in EvalDataset:
            cases = load_dataset(ds)
            assert len(cases) > 0, f"Dataset {ds.value} has no cases"

    def test_cases_have_required_fields(self):
        cases = load_all_cases()
        for case in cases:
            assert case.case_id, "case_id missing"
            assert case.question, "question missing"
            assert case.expected_answer, "expected_answer missing"
            assert case.difficulty in ("easy", "medium", "hard")

    def test_load_all_returns_all_datasets(self):
        cases = load_all_cases()
        datasets_found = {c.dataset for c in cases}
        assert len(datasets_found) == len(ALL_DATASETS)

    def test_sample_size_respected(self):
        cases = load_dataset(EvalDataset.UNDERWRITING_QA, sample_size=2)
        assert len(cases) <= 2

    def test_dataset_stats_structure(self):
        stats = dataset_stats()
        for ds_name, info in stats.items():
            assert "count" in info
            assert "difficulties" in info
            assert info["count"] > 0

    def test_no_duplicate_expected_keywords(self):
        """No case should have duplicate keywords (quality check)."""
        for cases in ALL_DATASETS.values():
            for case in cases:
                assert len(case.expected_keywords) == len(set(case.expected_keywords)), \
                    f"Duplicate keywords in case {case.case_id}"

    def test_forbidden_not_in_expected_answer(self):
        """Expected answers should not contain forbidden phrases."""
        for cases in ALL_DATASETS.values():
            for case in cases:
                for phrase in case.forbidden_phrases:
                    assert phrase.lower() not in case.expected_answer.lower(), \
                        f"Forbidden phrase '{phrase}' found in expected_answer for {case.case_id}"


# ─── Metric Evaluator Tests ───────────────────────────────────────────────────

class TestKeywordCoverage:
    def setup_method(self):
        self.ev = KeywordCoverageEvaluator()

    def test_perfect_keyword_coverage(self, uw_case):
        answer = "The maximum single-location TIV is $50M (50 million) without reinsurance approval."
        result = self.ev.evaluate(uw_case, answer)
        assert result.score == 1.0
        assert result.passed

    def test_zero_keyword_coverage(self, uw_case):
        result = self.ev.evaluate(uw_case, "I don't know the answer to this question.")
        assert result.score == 0.0
        assert not result.passed

    def test_partial_coverage(self, uw_case):
        result = self.ev.evaluate(uw_case, "The limit involves reinsurance requirements.")
        assert 0.0 < result.score < 1.0

    def test_empty_keywords_returns_perfect(self):
        case = EvalCase(
            dataset=EvalDataset.UNDERWRITING_QA,
            question="Q", expected_answer="A",
            expected_keywords=[], difficulty="easy", category="test",
        )
        result = self.ev.evaluate(case, "any answer")
        assert result.score == 1.0

    def test_score_bounded_0_to_1(self, uw_case):
        for answer in ["", "a" * 1000, "50m reinsurance"]:
            result = self.ev.evaluate(uw_case, answer)
            assert 0.0 <= result.score <= 1.0


class TestForbiddenPhrases:
    def setup_method(self):
        self.ev = ForbiddenPhraseEvaluator()

    def test_no_violations_scores_1(self, uw_case):
        result = self.ev.evaluate(uw_case, "The limit is $50M with reinsurance.")
        assert result.score == 1.0

    def test_single_violation_penalizes(self, uw_case):
        result = self.ev.evaluate(uw_case, "The limit is 100m without any restrictions.")
        assert result.score < 1.0

    def test_multiple_violations_penalize_more(self, uw_case):
        one_violation = self.ev.evaluate(uw_case, "100m is the limit").score
        two_violations = self.ev.evaluate(uw_case, "100m and unlimited coverage").score
        assert two_violations <= one_violation


class TestPIILeakage:
    def setup_method(self):
        self.ev = PIILeakageEvaluator()

    def test_clean_output_passes(self, uw_case):
        result = self.ev.evaluate(uw_case, "The TIV limit is $50M per location.")
        assert result.score == 1.0
        assert result.passed

    def test_ssn_in_output_fails(self, uw_case):
        result = self.ev.evaluate(uw_case, "Claimant SSN: 123-45-6789")
        assert result.score == 0.0
        assert not result.passed

    def test_email_in_output_fails(self, uw_case):
        result = self.ev.evaluate(uw_case, "Contact john.doe@example.com")
        assert result.score == 0.0

    def test_phone_in_output_fails(self, uw_case):
        result = self.ev.evaluate(uw_case, "Call 617-555-0123")
        assert result.score == 0.0

    def test_pii_threshold_is_high(self):
        """PII leakage has near-zero tolerance."""
        threshold = GATE_THRESHOLDS.get(MetricName.PII_LEAKAGE, 0.99)
        assert threshold >= 0.95, "PII leakage threshold must be very high"


class TestCompleteness:
    def setup_method(self):
        self.ev = CompletenessEvaluator()

    def test_full_answer_scores_high(self, uw_case):
        result = self.ev.evaluate(uw_case, uw_case.expected_answer)
        assert result.score >= 0.8

    def test_very_short_answer_scores_low(self, uw_case):
        result = self.ev.evaluate(uw_case, "Yes.")
        assert result.score < 0.5

    def test_score_bounded(self, uw_case):
        for answer in ["", "short", uw_case.expected_answer * 5]:
            result = self.ev.evaluate(uw_case, answer)
            assert 0.0 <= result.score <= 1.0


# ─── Overall Score Tests ──────────────────────────────────────────────────────

class TestOverallScore:
    def test_weights_sum_to_meaningful_value(self):
        total = sum(METRIC_WEIGHTS.values())
        assert total > 0.5, "Metric weights should sum to at least 0.5"

    def test_all_1s_gives_1(self, uw_case):
        from evaluators.metrics import ForbiddenPhraseEvaluator
        from models.schemas import MetricResult
        perfect = [
            MetricResult(metric=m, score=1.0, passed=True, threshold=0.8, rationale="perfect", scorer_type="rule_based")
            for m in METRIC_WEIGHTS.keys()
        ]
        score = compute_overall_score(perfect)
        assert score == 1.0

    def test_all_0s_gives_0(self, uw_case):
        from models.schemas import MetricResult
        zeros = [
            MetricResult(metric=m, score=0.0, passed=False, threshold=0.8, rationale="zero", scorer_type="rule_based")
            for m in METRIC_WEIGHTS.keys()
        ]
        score = compute_overall_score(zeros)
        assert score == 0.0

    def test_overall_score_bounded(self, uw_case):
        metrics = evaluate_case(
            uw_case,
            "The maximum TIV is $50M (50 million) with reinsurance approval needed.",
            [MetricName.FACTUAL_ACCURACY, MetricName.ANSWER_RELEVANCE, MetricName.PII_LEAKAGE],
        )
        score = compute_overall_score(metrics)
        assert 0.0 <= score <= 1.0


# ─── Gate Threshold Tests ──────────────────────────────────────────────────────

class TestGateThresholds:
    def test_pii_threshold_is_highest(self):
        """PII leakage should have the strictest threshold."""
        pii_threshold = GATE_THRESHOLDS.get(MetricName.PII_LEAKAGE, 0)
        other_thresholds = [v for k, v in GATE_THRESHOLDS.items() if k != MetricName.PII_LEAKAGE]
        assert pii_threshold >= max(other_thresholds) * 0.95

    def test_all_thresholds_between_0_and_1(self):
        for metric, threshold in GATE_THRESHOLDS.items():
            assert 0.0 < threshold <= 1.0, f"{metric.value} threshold out of range: {threshold}"

    def test_regulatory_safety_high_threshold(self):
        """Regulatory safety must have a high threshold for compliance."""
        threshold = GATE_THRESHOLDS.get(MetricName.REGULATORY_SAFETY, 0)
        assert threshold >= 0.90


# ─── Regression Engine Tests ──────────────────────────────────────────────────

class TestRegressionEngine:
    def setup_method(self):
        self.engine = RegressionEngine()
        self.store = RunStore()

    def _make_run(self, run_id, model, pass_rate, gate):
        from models.schemas import EvalRunResult, RunStatus
        config = EvalRunConfig(run_id=run_id, model=model, datasets=[], metrics=[])
        summary = _make_seed_summary(pass_rate, model, gate=gate)
        return EvalRunResult(
            config=config, status=RunStatus.COMPLETE,
            summary=summary, case_results=[],
        )

    def test_no_regression_gives_pass(self):
        current  = self._make_run("c1", "gpt-4o", 0.92, GateStatus.PASS)
        baseline = self._make_run("b1", "gpt-4o", 0.88, GateStatus.PASS)
        report = self.engine.compare(current, baseline)
        # Improved run should not regress
        assert report.recommendation in ("deploy", "investigate")

    def test_significant_drop_gives_block(self):
        current  = self._make_run("c2", "gpt-4o", 0.55, GateStatus.FAIL)
        baseline = self._make_run("b2", "gpt-4o", 0.92, GateStatus.PASS)
        report = self.engine.compare(current, baseline)
        assert report.gate_status == GateStatus.FAIL
        assert report.recommendation == "block"

    def test_report_has_all_required_fields(self):
        current  = self._make_run("c3", "gpt-4o", 0.85, GateStatus.PASS)
        baseline = self._make_run("b3", "gpt-4o", 0.83, GateStatus.PASS)
        report = self.engine.compare(current, baseline)
        assert report.report_id
        assert report.current_run_id == "c3"
        assert report.baseline_run_id == "b3"
        assert len(report.metric_deltas) > 0
        assert report.recommendation in ("deploy", "block", "investigate")

    def test_metric_deltas_have_correct_direction(self):
        current  = self._make_run("c4", "gpt-4o", 0.92, GateStatus.PASS)
        baseline = self._make_run("b4", "gpt-4o", 0.88, GateStatus.PASS)
        report = self.engine.compare(current, baseline)
        for delta in report.metric_deltas:
            if delta.delta > 0.02:
                assert delta.direction == "improved"
            elif delta.delta < -0.02:
                assert delta.direction == "regressed"


# ─── Eval Runner Integration Tests ────────────────────────────────────────────

class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_returns_result(self):
        runner = EvalRunner()
        config = EvalRunConfig(
            model="gpt-4o-mini",
            datasets=[EvalDataset.UNDERWRITING_QA],
            metrics=[MetricName.FACTUAL_ACCURACY, MetricName.ANSWER_RELEVANCE],
            sample_size=3,
        )
        result = await runner.run(config)
        assert result.status.value == "complete"
        assert len(result.case_results) <= 3
        assert result.summary is not None

    @pytest.mark.asyncio
    async def test_summary_pass_rate_bounded(self):
        runner = EvalRunner()
        config = EvalRunConfig(
            model="gpt-4o",
            datasets=[EvalDataset.CLAIMS_TRIAGE],
            metrics=[MetricName.ANSWER_RELEVANCE, MetricName.PII_LEAKAGE],
            sample_size=2,
        )
        result = await runner.run(config)
        assert 0.0 <= result.summary.pass_rate <= 1.0

    @pytest.mark.asyncio
    async def test_cost_is_positive(self):
        runner = EvalRunner()
        config = EvalRunConfig(
            model="gpt-4o",
            datasets=[EvalDataset.FRAUD_SIGNALS],
            metrics=[MetricName.FACTUAL_ACCURACY],
            sample_size=2,
        )
        result = await runner.run(config)
        assert result.summary.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_gate_status_determined(self):
        runner = EvalRunner()
        config = EvalRunConfig(
            model="gpt-4o",
            datasets=[EvalDataset.UNDERWRITING_QA],
            sample_size=3,
        )
        result = await runner.run(config)
        assert result.summary.gate_status in (GateStatus.PASS, GateStatus.FAIL, GateStatus.WARN)
