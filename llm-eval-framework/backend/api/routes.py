"""
API Routes — LLM Evaluation & Regression Framework.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from models.schemas import EvalDataset, EvalRunConfig, MetricName, RunStatus
from runners.eval_runner import eval_runner
from storage.run_store import run_store
from datasets.insurance_cases import dataset_stats

runs_router      = APIRouter()
datasets_router  = APIRouter()
metrics_router   = APIRouter()
health_router    = APIRouter()


@health_router.get("")
async def health():
    runs = run_store.list_runs(limit=5)
    return {"status": "healthy", "service": "llm-eval-framework", "version": "1.0.0", "total_runs": len(run_store._runs)}


# ── Runs ──────────────────────────────────────────────────────────────────────

@runs_router.post("")
async def create_run(config: EvalRunConfig, background_tasks: BackgroundTasks):
    """
    Trigger an evaluation run. Runs asynchronously — returns run_id immediately.
    Poll GET /runs/{run_id} for status and results.

    In production, this submits a job to Azure Container Jobs for parallel execution.
    """
    logger.info(f"New eval run triggered: model={config.model} datasets={config.datasets}")
    background_tasks.add_task(eval_runner.run, config)
    return {
        "run_id": config.run_id,
        "status": RunStatus.QUEUED.value,
        "model": config.model,
        "message": "Eval run queued. Poll GET /api/v1/runs/{run_id} for status.",
    }


@runs_router.get("")
async def list_runs(limit: int = Query(default=20, ge=1, le=100)):
    """List all evaluation runs, most recent first."""
    runs = run_store.list_runs(limit=limit)
    return {
        "runs": [
            {
                "run_id": r.config.run_id,
                "name": r.config.name,
                "model": r.config.model,
                "status": r.status.value,
                "pass_rate": r.summary.pass_rate if r.summary else None,
                "gate": r.summary.gate_status.value if r.summary else None,
                "started_at": r.started_at.isoformat(),
                "duration_s": r.duration_seconds,
                "cost_usd": r.summary.total_cost_usd if r.summary else None,
                "triggered_by": r.config.triggered_by,
            }
            for r in runs
        ],
        "total": len(runs),
    }


@runs_router.get("/compare")
async def model_comparison():
    """Latest results per model — for side-by-side comparison."""
    return {"models": run_store.model_comparison()}


@runs_router.get("/trend")
async def metric_trend(
    metric: str = Query(default="factual_accuracy"),
    model: Optional[str] = Query(default=None),
):
    """Historical trend for a metric over time."""
    return {"trend": run_store.get_trend_data(metric, model)}


@runs_router.get("/{run_id}")
async def get_run(run_id: str):
    """Get full result for a specific run."""
    run = run_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.model_dump()


@runs_router.get("/{run_id}/cases")
async def get_run_cases(run_id: str, failed_only: bool = Query(default=False)):
    """Get per-case results for a run (for drill-down analysis)."""
    run = run_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    cases = run.case_results
    if failed_only:
        cases = [c for c in cases if not c.passed]
    return {"run_id": run_id, "cases": [c.model_dump() for c in cases], "total": len(cases)}


# ── Datasets ──────────────────────────────────────────────────────────────────

@datasets_router.get("")
async def list_datasets():
    """List all available evaluation datasets with statistics."""
    return {"datasets": dataset_stats()}


# ── Metrics ───────────────────────────────────────────────────────────────────

@metrics_router.get("")
async def list_metrics():
    """List all supported metrics with thresholds."""
    from models.schemas import GATE_THRESHOLDS, METRIC_WEIGHTS
    return {
        "metrics": [
            {
                "name": m.value,
                "gate_threshold": GATE_THRESHOLDS.get(m),
                "weight": METRIC_WEIGHTS.get(m, 0.05),
            }
            for m in MetricName
        ]
    }


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Evaluation & Regression Framework",
    description="Production LLM evaluation harness for insurance AI — factual accuracy, hallucination, domain accuracy, CI/CD regression gating.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router,   prefix="/health",              tags=["Health"])
app.include_router(runs_router,     prefix="/api/v1/runs",         tags=["Runs"])
app.include_router(datasets_router, prefix="/api/v1/datasets",     tags=["Datasets"])
app.include_router(metrics_router,  prefix="/api/v1/metrics",      tags=["Metrics"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.routes:app", host="0.0.0.0", port=8000, reload=True)
