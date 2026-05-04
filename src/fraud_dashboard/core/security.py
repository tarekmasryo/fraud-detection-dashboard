from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from fraud_dashboard.core.config import get_settings

bearer_scheme = HTTPBearer(auto_error=False)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def create_access_token(subject: str, *, role: str = "admin", expires_in_s: int = 3600) -> str:
    settings = get_settings()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": subject, "role": role, "exp": int(time.time() + expires_in_s)}
    signing_input = f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}.{_b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    sig = hmac.new(
        settings.jwt_secret_key.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64url(sig)}"


def verify_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "HS256" or header.get("typ") != "JWT":
            raise ValueError("unsupported token header")
        signing_input = f"{header_b64}.{payload_b64}"
        expected = hmac.new(
            settings.jwt_secret_key.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
            raise ValueError("invalid signature")
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired token")
        if not payload.get("sub"):
            raise ValueError("missing subject")
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token"
        ) from exc


def hash_api_key(api_key: str) -> str:
    settings = get_settings()
    return hmac.new(
        settings.api_key_hash_secret.encode(), api_key.encode(), hashlib.sha256
    ).hexdigest()


def _valid_api_key(api_key: str | None) -> bool:
    if not api_key:
        return False
    settings = get_settings()
    if settings.demo_api_key_hash:
        return hmac.compare_digest(hash_api_key(api_key), settings.demo_api_key_hash)
    return hmac.compare_digest(api_key, settings.demo_api_key)


def require_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.require_auth:
        return {"sub": "anonymous-dev", "role": "admin", "auth": "disabled"}
    if _valid_api_key(x_api_key):
        return {"sub": "service", "role": "service", "auth": "api_key"}
    if credentials and credentials.scheme.lower() == "bearer":
        principal = verify_access_token(credentials.credentials)
        principal["auth"] = "jwt"
        return principal
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def require_role(*allowed_roles: str):
    def dependency(
        principal: Annotated[dict[str, Any], Depends(require_principal)],
    ) -> dict[str, Any]:
        role = str(principal.get("role", ""))
        if role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return principal

    return dependency
