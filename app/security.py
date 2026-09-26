"""Service-to-service auth for calls from doqseal-backend.

One scheme for every internal endpoint: an HS256 JWT signed with the shared
secret (AI_ENGINE_JWT_SECRET, or its alias AI_ENGINE_SERVICE_TOKEN).

Claims: iss="doqseal-backend", aud="doqseal-ai-engine", sub=<userId>,
org=<organisationId>, pid=<projectId|null>, scope, iat, exp (at most 5 minutes
after iat), jti. The organisation and user are taken only from verified claims;
a body or query organisationId that differs from `org` is refused with 403.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Request

from app.config import settings

logger = logging.getLogger("doqseal.security")

ISSUER = "doqseal-backend"
AUDIENCE = "doqseal-ai-engine"
ALGORITHM = "HS256"
MAX_LIFETIME_SECONDS = 300
LEEWAY_SECONDS = 30
SCOPES = frozenset({"chat", "rag:read", "rag:delete", "bundle:classify"})


@dataclass(frozen=True)
class ServiceClaims:
    user_id: str
    organisation_id: str
    project_id: str | None
    scope: str
    jti: str


def service_secret() -> str:
    return (settings.service_jwt_secret or "").strip()


def auth_enforced() -> bool:
    return bool(service_secret())


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def verify_token(token: str, *, scope: str) -> ServiceClaims:
    secret = service_secret()
    if not secret:
        raise HTTPException(status_code=503, detail="service auth is not configured")
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
            leeway=LEEWAY_SECONDS,
            options={"require": ["iss", "aud", "sub", "org", "scope", "iat", "exp", "jti"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("token expired") from None
    except jwt.InvalidTokenError:
        raise _unauthorized("invalid token") from None

    try:
        lifetime = int(payload["exp"]) - int(payload["iat"])
    except (TypeError, ValueError):
        raise _unauthorized("invalid token") from None
    if lifetime <= 0 or lifetime > MAX_LIFETIME_SECONDS:
        raise _unauthorized("token lifetime too long")

    org = payload.get("org")
    sub = payload.get("sub")
    if not isinstance(org, str) or not org.strip() or not isinstance(sub, str) or not sub.strip():
        raise _unauthorized("invalid token")

    token_scope = payload.get("scope")
    if token_scope not in SCOPES or token_scope != scope:
        raise HTTPException(status_code=403, detail="token scope does not allow this call")

    pid = payload.get("pid")
    return ServiceClaims(
        user_id=sub,
        organisation_id=org,
        project_id=pid if isinstance(pid, str) and pid else None,
        scope=token_scope,
        jti=str(payload.get("jti")),
    )


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def require_claims(request: Request, *, scope: str) -> ServiceClaims:
    """Always requires a valid token (503 when the secret is not configured)."""
    if not auth_enforced():
        raise HTTPException(status_code=503, detail="service auth is not configured")
    token = _bearer(request)
    if not token:
        raise _unauthorized("missing bearer token")
    return verify_token(token, scope=scope)


def optional_claims(request: Request, *, scope: str) -> ServiceClaims | None:
    """Legacy endpoints: enforce when the secret is set, otherwise warn and allow."""
    if not auth_enforced():
        logger.warning(
            "SECURITY: %s %s called without service auth. Set AI_ENGINE_JWT_SECRET "
            "on both doqseal-backend and doqseal-ai-engine to enforce it.",
            request.method,
            request.url.path,
        )
        return None
    token = _bearer(request)
    if not token:
        raise _unauthorized("missing bearer token")
    return verify_token(token, scope=scope)


def check_org(claims: ServiceClaims | None, requested_org: str | None) -> None:
    """403 when a caller-supplied organisation differs from the verified one."""
    if claims is None or requested_org is None:
        return
    if requested_org.strip() and requested_org.strip() != claims.organisation_id:
        logger.warning("organisation mismatch between token and request (scope=%s)", claims.scope)
        raise HTTPException(status_code=403, detail="organisation mismatch")
