"""JWT authentication for service-to-service calls from doqseal-backend."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("doqseal.auth")

JWT_SECRET_ENV = "AI_ENGINE_JWT_SECRET"
JWT_ISSUER = "doqseal-backend"
JWT_AUDIENCE = "doqseal-ai-engine"
JWT_ALGORITHM = "HS256"
JWT_MAX_AGE_SECONDS = 300  # 5 minutes max


@dataclass
class ServiceClaims:
    """Verified claims from a service JWT."""

    user_id: str
    organisation_id: str
    project_id: str | None
    scope: Literal["chat", "rag:delete", "rag:read"]
    jti: str


def _get_jwt_secret() -> str | None:
    """Return the JWT secret if configured, else None (legacy mode)."""
    return os.getenv(JWT_SECRET_ENV, "").strip() or None


def _log_legacy_warning(endpoint: str) -> None:
    """Log a loud warning when running without JWT enforcement."""
    logger.warning(
        "SECURITY: %s called without JWT enforcement. "
        "Set %s to enable authentication. "
        "This is acceptable only during rollout.",
        endpoint,
        JWT_SECRET_ENV,
    )


def verify_service_jwt(token: str) -> ServiceClaims:
    """Verify the service JWT and return claims.

    Raises:
        HTTPException: 401 for invalid/expired tokens.
    """
    secret = _get_jwt_secret()
    if not secret:
        raise HTTPException(
            status_code=500,
            detail="JWT verification requested but secret not configured",
        )

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "require": ["iss", "aud", "sub", "org", "scope", "iat", "exp", "jti"],
            },
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid token issuer")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.InvalidTokenError as e:
        logger.warning("JWT validation failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid token")

    iat = payload.get("iat", 0)
    exp = payload.get("exp", 0)
    if exp - iat > JWT_MAX_AGE_SECONDS:
        raise HTTPException(status_code=401, detail="Token lifetime exceeds maximum")

    return ServiceClaims(
        user_id=payload["sub"],
        organisation_id=payload["org"],
        project_id=payload.get("pid"),
        scope=payload["scope"],
        jti=payload["jti"],
    )


def check_org_match(claims: ServiceClaims, body_org: str | None) -> None:
    """Verify body organisationId matches token claim.

    Raises:
        HTTPException: 403 if mismatch.
    """
    if body_org and body_org != claims.organisation_id:
        logger.warning(
            "Organisation mismatch: token=%s body=%s",
            claims.organisation_id,
            body_org,
        )
        raise HTTPException(
            status_code=403,
            detail="Organisation mismatch between token and request",
        )


class OptionalJWTBearer(HTTPBearer):
    """Bearer auth that's optional when JWT_SECRET is not set (rollout mode)."""

    def __init__(self, *, scope: str):
        super().__init__(auto_error=False)
        self.required_scope = scope

    async def __call__(self, request: Request) -> ServiceClaims | None:
        secret = _get_jwt_secret()

        if not secret:
            _log_legacy_warning(request.url.path)
            return None

        credentials: HTTPAuthorizationCredentials | None = await super().__call__(request)

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Missing authorization header",
            )

        if credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication scheme",
            )

        claims = verify_service_jwt(credentials.credentials)

        if claims.scope != self.required_scope:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient scope: required {self.required_scope}",
            )

        return claims


def get_chat_claims(
    claims: ServiceClaims | None = Depends(OptionalJWTBearer(scope="chat")),
) -> ServiceClaims | None:
    """Dependency for chat endpoints requiring 'chat' scope."""
    return claims


def get_rag_read_claims(
    claims: ServiceClaims | None = Depends(OptionalJWTBearer(scope="rag:read")),
) -> ServiceClaims | None:
    """Dependency for RAG read endpoints requiring 'rag:read' scope."""
    return claims


def get_rag_delete_claims(
    claims: ServiceClaims | None = Depends(OptionalJWTBearer(scope="rag:delete")),
) -> ServiceClaims | None:
    """Dependency for RAG delete endpoints requiring 'rag:delete' scope."""
    return claims


def resolve_context(
    claims: ServiceClaims | None,
    body_org: str | None,
    body_user: str | None,
    body_project: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Resolve org/user/project from claims or body (legacy mode).

    Returns:
        (organisation_id, user_id, project_id)

    Raises:
        HTTPException: 400 if org missing in legacy mode
        HTTPException: 403 if org mismatch
    """
    if claims:
        check_org_match(claims, body_org)
        return claims.organisation_id, claims.user_id, claims.project_id

    if not body_org:
        raise HTTPException(status_code=400, detail="organisationId is required")

    return body_org, body_user, body_project
