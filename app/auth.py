"""API key authentication dependencies."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def _extract_token(
    authorization: Optional[str],
    x_api_key: Optional[str],
) -> Optional[str]:
    if authorization:
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    if x_api_key:
        return x_api_key.strip()
    return None


def verify_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate Bearer token or X-API-Key when API_KEYS is configured."""
    allowed = settings.api_key_set
    if not allowed:
        return

    token = _extract_token(authorization, x_api_key)
    if not token or not any(secrets.compare_digest(token, k) for k in allowed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
