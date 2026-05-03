"""
Governed AI Gateway — FastAPI Application
Liberty Mutual Insurance · Portfolio Project
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from api.routes import gateway_router, auth_router, audit_router, usage_router, health_router

app = FastAPI(
    title="Governed AI Gateway",
    description="""
Enterprise AI Gateway enforcing:
- JWT authentication (Azure AD / Entra ID)
- Role-based model access control (RBAC)  
- Bidirectional PII redaction (Presidio)
- Policy engine (rate limits, content moderation, budget)
- Immutable audit log (PII-free, append-only)
- Cost tracking & chargeback by team
- Real-time governance dashboard
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(health_router,  prefix="/health",             tags=["Health"])
app.include_router(auth_router,    prefix="/api/v1/auth",        tags=["Auth (Demo)"])
app.include_router(gateway_router, prefix="/api/v1/gateway",     tags=["Gateway"])
app.include_router(audit_router,   prefix="/api/v1/audit",       tags=["Audit"])
app.include_router(usage_router,   prefix="/api/v1/usage",       tags=["Usage & Cost"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
