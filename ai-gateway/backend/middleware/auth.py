"""
Authentication & RBAC Middleware.

Every request to the gateway must carry a valid JWT issued by the
corporate identity provider (Azure AD / Entra ID in production).

The token is validated for:
  1. Signature (RS256, public key from JWKS endpoint)
  2. Expiry
  3. Issuer (must be corporate IdP)
  4. Audience (must be this gateway's app ID)
  5. Required claims: user_id, role, team, scopes

RBAC is then enforced based on the role claim:
  - Which LLM models the caller may access
  - Which API endpoints are reachable
  - Maximum token limits per request

In production:
  from azure.identity import DefaultAzureCredential
  from msal import ConfidentialClientApplication
  Uses python-jose or PyJWT with JWKS key rotation.
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

from fastapi import HTTPException, Header, Request
from models.schemas import (
    AuthenticatedCaller, LLMModel, TeamCode, UserRole,
)

logger = logging.getLogger(__name__)

# Models each role is permitted to call
ROLE_MODEL_ALLOWLIST: dict[UserRole, set[LLMModel]] = {
    UserRole.ADMIN:              {LLMModel.GPT4O, LLMModel.GPT4O_MINI, LLMModel.GPT35_TURBO, LLMModel.CLAUDE_SONNET, LLMModel.CLAUDE_HAIKU},
    UserRole.DATA_SCIENTIST:     {LLMModel.GPT4O, LLMModel.GPT4O_MINI, LLMModel.GPT35_TURBO, LLMModel.CLAUDE_SONNET, LLMModel.CLAUDE_HAIKU},
    UserRole.FRAUD_INVESTIGATOR: {LLMModel.GPT4O, LLMModel.GPT4O_MINI, LLMModel.CLAUDE_SONNET},
    UserRole.UNDERWRITER:        {LLMModel.GPT4O, LLMModel.GPT4O_MINI},
    UserRole.CLAIMS_ADJUSTER:    {LLMModel.GPT4O, LLMModel.GPT4O_MINI},
    UserRole.ACTUARY:            {LLMModel.GPT4O_MINI, LLMModel.GPT35_TURBO},
    UserRole.COMPLIANCE:         {LLMModel.GPT4O_MINI},
    UserRole.SERVICE_ACCOUNT:    {LLMModel.GPT4O, LLMModel.GPT4O_MINI, LLMModel.GPT35_TURBO, LLMModel.CLAUDE_SONNET, LLMModel.CLAUDE_HAIKU},
}

# Endpoint scope requirements
ENDPOINT_SCOPES: dict[str, str] = {
    "/api/v1/gateway/chat":          "llm:query",
    "/api/v1/gateway/admin":         "llm:admin",
    "/api/v1/audit/events":          "audit:read",
    "/api/v1/usage/teams":           "usage:read",
}

# Simulated token store (production: Azure AD JWKS + token introspection)
_DEMO_TOKENS: dict[str, dict] = {}


class AuthMiddleware:
    """
    Validates bearer tokens and extracts caller identity.

    Production implementation:
        from jose import jwt, JWTError
        from jose.backends import RSAKey

        jwks_client = PyJWKClient(
            f"https://login.microsoftonline.com/{TENANT_ID}/discovery/v2.0/keys"
        )

        def verify(token: str) -> dict:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=GATEWAY_APP_ID,
                issuer=f"https://sts.windows.net/{TENANT_ID}/",
            )
            return payload
    """

    GATEWAY_SECRET = os.getenv("GATEWAY_JWT_SECRET", "dev-secret-change-in-production")
    TOKEN_TTL_MINUTES = 60

    def issue_demo_token(
        self,
        user_id: str,
        role: UserRole,
        team: TeamCode,
        display_name: str,
    ) -> str:
        """
        Issue a demo JWT. In production, tokens come from Azure AD.
        This simulates the token payload that Azure AD would issue.
        """
        import base64, json, time
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "email": f"{user_id.lower().replace(' ', '.')}@libertymutual.com",
            "role": role.value,
            "team": team.value,
            "name": display_name,
            "scopes": self._scopes_for_role(role),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=self.TOKEN_TTL_MINUTES)).timestamp()),
            "iss": "https://sts.windows.net/libertymutual-tenant/",
            "aud": "ai-gateway-prod",
        }
        # Simple HMAC signature for demo (use RS256 in production)
        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
        body   = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig    = hmac.new(self.GATEWAY_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
        token  = f"{header}.{body}.{sig}"
        _DEMO_TOKENS[token] = payload
        logger.info(f"Demo token issued for {user_id} / {role.value}")
        return token

    def verify_token(self, authorization: Optional[str]) -> AuthenticatedCaller:
        """Verify bearer token → return AuthenticatedCaller or raise 401."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ", 1)[1]

        # In production: JWKS signature verification
        payload = _DEMO_TOKENS.get(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        exp = payload.get("exp", 0)
        if datetime.now(timezone.utc).timestamp() > exp:
            raise HTTPException(status_code=401, detail="Token expired")

        return AuthenticatedCaller(
            user_id=payload["sub"],
            email=payload["email"],
            role=UserRole(payload["role"]),
            team=TeamCode(payload["team"]),
            display_name=payload["name"],
            scopes=payload["scopes"],
            token_issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            token_expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )

    def enforce_model_access(self, caller: AuthenticatedCaller, model: LLMModel) -> None:
        """Raise 403 if caller's role is not permitted to use the requested model."""
        allowed = ROLE_MODEL_ALLOWLIST.get(caller.role, set())
        if model not in allowed:
            logger.warning(f"RBAC DENY: {caller.user_id} ({caller.role}) attempted {model}")
            raise HTTPException(
                status_code=403,
                detail=f"Role '{caller.role.value}' is not permitted to access model '{model.value}'. "
                       f"Permitted models: {[m.value for m in allowed]}",
            )

    def enforce_scope(self, caller: AuthenticatedCaller, required_scope: str) -> None:
        """Raise 403 if caller lacks required OAuth scope."""
        if required_scope not in caller.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Scope '{required_scope}' required but not present in token",
            )

    def pseudonymize(self, value: str) -> str:
        """One-way hash for storing user identity in audit logs (GDPR compliance)."""
        return hashlib.sha256(f"gateway-salt:{value}".encode()).hexdigest()[:16]

    def _scopes_for_role(self, role: UserRole) -> list[str]:
        scope_map = {
            UserRole.ADMIN:              ["llm:query", "llm:admin", "audit:read", "usage:read", "usage:admin"],
            UserRole.DATA_SCIENTIST:     ["llm:query", "audit:read", "usage:read"],
            UserRole.UNDERWRITER:        ["llm:query"],
            UserRole.CLAIMS_ADJUSTER:    ["llm:query"],
            UserRole.FRAUD_INVESTIGATOR: ["llm:query", "audit:read"],
            UserRole.ACTUARY:            ["llm:query"],
            UserRole.COMPLIANCE:         ["llm:query", "audit:read", "usage:read"],
            UserRole.SERVICE_ACCOUNT:    ["llm:query", "llm:admin", "audit:read", "usage:read"],
        }
        return scope_map.get(role, ["llm:query"])


auth_middleware = AuthMiddleware()
