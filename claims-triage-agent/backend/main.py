"""
Claims Triage & Fraud Signal Agent
Liberty Mutual Insurance - Portfolio Project
Senior Applied AI Engineer Demo

FastAPI application entry point.
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

from api.routes import claims_router, audit_router, health_router
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Claims Triage & Fraud Signal Agent",
    description="""
    Enterprise-grade agentic pipeline for insurance claims processing.
    
    Features:
    - LLM-powered document ingestion & entity extraction
    - Multi-signal fraud scoring (rules + ML + LLM)
    - Adjuster routing with explainable decisions
    - Human-in-the-loop escalation
    - Full PII-redacted audit trail
    - Role-based access control
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(claims_router, prefix="/api/v1/claims", tags=["Claims"])
app.include_router(audit_router, prefix="/api/v1/audit", tags=["Audit"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
