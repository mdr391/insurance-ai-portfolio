"""Underwriting Copilot — FastAPI Application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

from api.routes import copilot_router, health_router

app = FastAPI(
    title="Underwriting Copilot RAG",
    description="Retrieval-Augmented Generation for underwriting knowledge queries. Grounded answers with inline citations from indexed guidelines, coverage manuals, and loss run reports.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(copilot_router, prefix="/api/v1/copilot", tags=["Copilot"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
