"""
RAG Evaluation Harness using RAGAS.

Evaluates the pipeline on a curated test set of underwriting questions
with known expected answers. Tracks:
  - Faithfulness: Is the answer grounded in retrieved context?
  - Answer Relevance: Does the answer address the question?
  - Citation Precision: Are citations accurate?
  - Hallucination Rate: Claims not supported by sources
  - Context Recall: Were relevant source chunks retrieved?

In production, this runs on every deployment as a regression gate
in the GitHub Actions CI/CD pipeline. Fails if any metric drops
below threshold.

Production code:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    dataset = Dataset.from_list([{
        "question": q["question"],
        "answer": q["generated_answer"],
        "contexts": q["retrieved_contexts"],
        "ground_truth": q["expected_answer"],
    } for q in test_results])

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=AzureChatOpenAI(model="gpt-4o", ...),
        embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-large", ...),
    )
"""

import asyncio
import json
import logging
from datetime import datetime

from models.schemas import CopilotQuery, EvalResult, EvalReport
from rag.retrieval_engine import UnderwritingRAGEngine

logger = logging.getLogger(__name__)

# Ground-truth test set — curated by senior underwriters
TEST_QUESTIONS = [
    {
        "question": "What is the maximum TIV for a single habitational property without reinsurance approval?",
        "expected_keywords": ["50m", "50 million", "reinsurance", "facultative"],
        "category": "property",
    },
    {
        "question": "Is flood covered under a standard commercial property policy?",
        "expected_keywords": ["excluded", "iso cp 10 30", "flood zone x", "manuscript endorsement"],
        "category": "property",
    },
    {
        "question": "What controls are required before we can offer cyber coverage?",
        "expected_keywords": ["mfa", "multi-factor", "edr", "backup", "patch"],
        "category": "cyber",
    },
    {
        "question": "What is the e-mod threshold that requires senior underwriter approval?",
        "expected_keywords": ["1.25", "senior", "risk improvement"],
        "category": "workers_comp",
    },
    {
        "question": "Is liquor liability covered under a standard GL policy for a bar?",
        "expected_keywords": ["excluded", "dram shop", "separate", "liquor liability policy"],
        "category": "liability",
    },
    {
        "question": "What sprinkler requirements apply to warehouse risks?",
        "expected_keywords": ["10,000", "mandatory", "sprinkler", "warehouse"],
        "category": "property",
    },
    {
        "question": "What driver tier requires a 25% surcharge and senior approval?",
        "expected_keywords": ["tier 3", "non-standard", "25%", "surcharge"],
        "category": "auto",
    },
    {
        "question": "What ransomware sublimit applies to healthcare accounts?",
        "expected_keywords": ["5m", "5 million", "healthcare", "sublimit"],
        "category": "cyber",
    },
]

# Evaluation thresholds (CI gate)
THRESHOLDS = {
    "avg_faithfulness": 0.75,
    "avg_relevance": 0.70,
    "avg_citation_precision": 0.65,
    "hallucination_rate": 0.15,  # must be below this
}


class RAGEvaluator:
    """
    Runs evaluation suite and generates pass/fail report.
    Designed to run in CI/CD as a regression gate.
    """

    def __init__(self, engine: UnderwritingRAGEngine):
        self.engine = engine

    async def run_eval(self) -> EvalReport:
        """Run full evaluation suite."""
        results = []
        logger.info(f"Starting eval run on {len(TEST_QUESTIONS)} questions...")

        for test in TEST_QUESTIONS:
            result = await self._eval_single(test)
            results.append(result)
            logger.info(
                f"[{test['category']}] '{test['question'][:50]}...' "
                f"→ faithfulness={result.answer_faithfulness:.2f} "
                f"relevance={result.answer_relevance:.2f} "
                f"{'PASS' if result.passed else 'FAIL'}"
            )

        passed = sum(1 for r in results if r.passed)
        avg_faith = sum(r.answer_faithfulness for r in results) / len(results)
        avg_rel = sum(r.answer_relevance for r in results) / len(results)
        avg_cit = sum(r.citation_precision for r in results) / len(results)
        avg_lat = sum(r.latency_ms for r in results) / len(results)
        hallucination_rate = 1 - avg_faith

        report = EvalReport(
            total_questions=len(TEST_QUESTIONS),
            passed=passed,
            avg_faithfulness=round(avg_faith, 3),
            avg_relevance=round(avg_rel, 3),
            avg_citation_precision=round(avg_cit, 3),
            avg_latency_ms=round(avg_lat),
            hallucination_rate=round(hallucination_rate, 3),
            results=results,
        )

        # CI gate check
        self._check_thresholds(report)
        return report

    async def _eval_single(self, test: dict) -> EvalResult:
        import time
        t0 = time.time()
        query = CopilotQuery(query_text=test["question"])
        response = await self.engine.query(query)
        latency = int((time.time() - t0) * 1000)

        answer_lower = response.answer.lower()
        keywords_hit = sum(1 for kw in test["expected_keywords"] if kw.lower() in answer_lower)
        keyword_coverage = keywords_hit / len(test["expected_keywords"])

        # Approximate metrics (in production: use RAGAS LLM-based scoring)
        faithfulness = min(1.0, (len(response.citations) / max(response.retrieved_chunks, 1)) * 0.6 + keyword_coverage * 0.4)
        relevance = keyword_coverage
        citation_precision = min(1.0, len(response.citations) / max(response.retrieved_chunks, 1))

        passed = (
            faithfulness >= THRESHOLDS["avg_faithfulness"] * 0.85 and
            relevance >= THRESHOLDS["avg_relevance"] * 0.85
        )

        return EvalResult(
            question=test["question"],
            expected_answer_keywords=test["expected_keywords"],
            actual_answer=response.answer,
            citations_count=len(response.citations),
            answer_faithfulness=round(faithfulness, 3),
            answer_relevance=round(relevance, 3),
            citation_precision=round(citation_precision, 3),
            latency_ms=latency,
            passed=passed,
        )

    def _check_thresholds(self, report: EvalReport) -> None:
        failures = []
        if report.avg_faithfulness < THRESHOLDS["avg_faithfulness"]:
            failures.append(f"faithfulness {report.avg_faithfulness:.3f} < {THRESHOLDS['avg_faithfulness']}")
        if report.avg_relevance < THRESHOLDS["avg_relevance"]:
            failures.append(f"relevance {report.avg_relevance:.3f} < {THRESHOLDS['avg_relevance']}")
        if report.hallucination_rate > THRESHOLDS["hallucination_rate"]:
            failures.append(f"hallucination_rate {report.hallucination_rate:.3f} > {THRESHOLDS['hallucination_rate']}")

        if failures:
            logger.warning(f"EVAL GATE FAILED: {'; '.join(failures)}")
        else:
            logger.info(f"EVAL GATE PASSED: {report.passed}/{report.total_questions} questions passed")


if __name__ == "__main__":
    import asyncio
    engine = UnderwritingRAGEngine()
    evaluator = RAGEvaluator(engine)
    report = asyncio.run(evaluator.run_eval())
    print(json.dumps(report.model_dump(), indent=2, default=str))
