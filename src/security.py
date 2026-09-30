"""JWT bearer authentication and scope checks."""

from __future__ import annotations

import logging
from typing import Any, Dict, Set

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# auto_error=False so we control the status code: a missing/invalid credential
# is 401 (with WWW-Authenticate), an authenticated-but-forbidden caller is 403.
_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def decode_token(token: str, settings: Settings) -> Dict[str, Any]:
    options = {"require": ["exp", "sub"]}
    if settings.JWT_AUDIENCE:
        options["require"].append("aud")
    if settings.JWT_ISSUER:
        options["require"].append("iss")
    return jwt.decode(
        token,
        settings.JWT_SECRET.get_secret_value(),
        algorithms=[settings.JWT_ALGORITHM],
        audience=settings.JWT_AUDIENCE,
        issuer=settings.JWT_ISSUER,
        leeway=settings.JWT_LEEWAY_SECONDS,
        options=options,
    )


def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Missing bearer token.")
    try:
        return decode_token(credentials.credentials, settings)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token signature has expired.") from None
    except jwt.InvalidTokenError as exc:
        # Log the reason server-side; don't leak validation details to clients.
        logger.info("JWT rejected", extra={"reason": type(exc).__name__})
        raise _unauthorized("Invalid authentication token.") from exc


def token_scopes(claims: Dict[str, Any]) -> Set[str]:
    scopes: Set[str] = set()
    scope = claims.get("scope")
    if isinstance(scope, str):
        scopes.update(scope.split())
    for key in ("scopes", "roles"):
        value = claims.get(key)
        if isinstance(value, (list, tuple)):
            scopes.update(str(v) for v in value)
    return scopes


def get_principal(
    claims: Dict[str, Any] = Depends(verify_jwt_token),
    settings: Settings = Depends(get_settings),
):
    """Authenticated caller with tenant, groups and admin flag resolved from the JWT."""
    from src.tenancy import TenancyError, principal_from_claims

    try:
        return principal_from_claims(claims, token_scopes(claims), settings)
    except TenancyError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def get_ingest_principal(
    principal=Depends(get_principal),
    settings: Settings = Depends(get_settings),
):
    """Principal that may upload / manage documents (ingest scope or admin)."""
    required = settings.JWT_INGEST_SCOPE
    if required and required not in principal.scopes and not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Token lacks required scope '{required}'.",
        )
    return principal
