# -*- coding: utf-8 -*-
"""
GreenLang Factors API — authentication + tier access control.

Two authentication paths are supported:

1. **JWT Bearer** — the production path. Validated by the existing
   ``greenlang.integration.api.dependencies.get_current_user`` dependency.
2. **API-Key** — the developer-tier path. Validated in-process against a
   keyring that ships via environment variable (``GL_FACTORS_API_KEYS``),
   a JSON file (``GL_FACTORS_API_KEY_FILE``), or both.

Tier gating for routes is enforced by
:func:`require_tier_for_endpoint`, which is meant to be used as a
FastAPI dependency on the route.

The keyring format (JSON)::

    [
      {
        "key_id": "dev-001",
        "key": "gl_fac_plk_...",
        "tier": "community",
        "tenant_id": "dev-tenant",
        "user_id": "alice",
        "active": true,
        "description": "Developer tier key"
      }
    ]

In production, swap the JSON loader for a database-backed keyring (the
``APIKeyValidator`` class is the extension point).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier-based endpoint access control
# ---------------------------------------------------------------------------

#: Tier hierarchy (lower index = lower tier).  Enforcement compares index
#: order, not string equality, so ``"pro"`` is automatically allowed on
#: routes that require ``"community"``.
TIER_ORDER: List[str] = ["community", "pro", "enterprise", "internal"]


#: Per-endpoint minimum tier.  Keys are route-key fragments matched by
#: ``endpoint_key`` via substring containment; entries are consulted in
#: declaration order, so **put the most specific (longest) key first**.
#: Everything not listed defaults to ``"community"``.
ENDPOINT_MIN_TIER: Dict[str, str] = {
    # Most-specific suffixes first (longer matches before shorter ones).
    "/audit-bundle": "enterprise",
    "/search/facets": "community",
    "/search/v2": "community",
    "/coverage": "community",
    "/export": "pro",
    "/match": "pro",
    "/diff": "pro",
    "/search": "community",
}


def tier_rank(tier: Optional[str]) -> int:
    """Return the sort index of ``tier`` within :data:`TIER_ORDER`.

    Unknown tiers are treated as ``"community"`` (lowest) for safety.
    """
    if not tier:
        return 0
    t = tier.strip().lower()
    try:
        return TIER_ORDER.index(t)
    except ValueError:
        return 0


def min_tier_for_endpoint(endpoint_key: str) -> str:
    """Return the minimum allowed tier for a given route key."""
    for key, tier in ENDPOINT_MIN_TIER.items():
        if key in endpoint_key:
            return tier
    return "community"


def tier_allows_endpoint(tier: str, endpoint_key: str) -> bool:
    """True when ``tier`` meets the endpoint's minimum."""
    return tier_rank(tier) >= tier_rank(min_tier_for_endpoint(endpoint_key))


# ---------------------------------------------------------------------------
# API-Key validation
# ---------------------------------------------------------------------------


@dataclass
class APIKeyRecord:
    """In-memory representation of an API key and its tenant context."""

    key_id: str
    key_hash: str  # sha256(key) hex
    tier: str = "community"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    active: bool = True
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)

    def to_user(self) -> Dict[str, Any]:
        """Project the key record into a user-dict compatible with routes."""
        return {
            "user_id": self.user_id or f"api-key:{self.key_id}",
            "email": None,
            "tenant_id": self.tenant_id,
            "tier": self.tier,
            "roles": list(self.roles),
            "permissions": list(self.permissions),
            "auth_method": "api_key",
            "api_key_id": self.key_id,
        }


class APIKeyValidator:
    """Keyring-backed API-key validator.

    Thread-safe and cheap to call (constant-time hash compare).  The
    keyring is loaded once on first use; call :meth:`reload` to pick up
    newly added keys.
    """

    def __init__(
        self,
        keys: Optional[List[APIKeyRecord]] = None,
        *,
        env_var: str = "GL_FACTORS_API_KEYS",
        file_env_var: str = "GL_FACTORS_API_KEY_FILE",
    ) -> None:
        self._env_var = env_var
        self._file_env_var = file_env_var
        self._lock = threading.Lock()
        self._records: List[APIKeyRecord] = list(keys or [])
        if keys is None:
            self._load_from_environment()

    # ----- Loading -----

    def reload(self) -> None:
        """Re-read the keyring from environment sources."""
        with self._lock:
            self._records = []
            self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load API-key records from env var and/or file path."""
        # JSON blob in env var.
        raw = os.environ.get(self._env_var, "").strip()
        if raw:
            try:
                for entry in json.loads(raw):
                    record = _record_from_entry(entry)
                    if record is not None:
                        self._records.append(record)
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Failed to parse %s: %s", self._env_var, exc)

        # JSON file on disk.
        file_path = os.environ.get(self._file_env_var, "").strip()
        if file_path:
            try:
                entries = json.loads(Path(file_path).read_text(encoding="utf-8"))
                for entry in entries:
                    record = _record_from_entry(entry)
                    if record is not None:
                        self._records.append(record)
            except FileNotFoundError:
                logger.warning("API key file not found: %s", file_path)
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Failed to parse %s: %s", file_path, exc)

        logger.info(
            "APIKeyValidator loaded %d key(s) from env/file", len(self._records)
        )

    # ----- Validation -----

    def validate(self, api_key: str) -> Optional[APIKeyRecord]:
        """Return the matching record for ``api_key``, or ``None``."""
        if not api_key:
            return None
        candidate_hash = _hash_key(api_key)
        with self._lock:
            for record in self._records:
                if not record.active:
                    continue
                if hmac.compare_digest(record.key_hash, candidate_hash):
                    return record
        return None

    def list_records(self) -> List[APIKeyRecord]:
        with self._lock:
            return list(self._records)


def _record_from_entry(entry: Dict[str, Any]) -> Optional[APIKeyRecord]:
    """Build an ``APIKeyRecord`` from a keyring JSON entry."""
    key_id = entry.get("key_id")
    key = entry.get("key")
    if not key_id or not key:
        return None
    key_hash = entry.get("key_hash") or _hash_key(key)
    return APIKeyRecord(
        key_id=str(key_id),
        key_hash=key_hash,
        tier=str(entry.get("tier", "community")),
        tenant_id=entry.get("tenant_id"),
        user_id=entry.get("user_id"),
        description=str(entry.get("description", "")),
        active=bool(entry.get("active", True)),
        roles=list(entry.get("roles") or []),
        permissions=list(entry.get("permissions") or []),
    )


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Unified authenticator (JWT → API key fallback)
# ---------------------------------------------------------------------------


_default_validator: Optional[APIKeyValidator] = None


def default_validator() -> APIKeyValidator:
    """Return the process-wide default :class:`APIKeyValidator`."""
    global _default_validator
    if _default_validator is None:
        _default_validator = APIKeyValidator()
    return _default_validator


def authenticate_headers(
    authorization: Optional[str],
    api_key_header: Optional[str],
    *,
    jwt_decode: Optional[Callable[[str], Dict[str, Any]]] = None,
    validator: Optional[APIKeyValidator] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a user dict from raw HTTP headers.

    Tries ``Authorization: Bearer <jwt>`` first; if absent or invalid,
    falls back to ``X-API-Key: <key>``.  Returns ``None`` when neither
    authentication path yields a user.

    This is a *pure* function — no FastAPI coupling — so it can be
    unit-tested in isolation.
    """
    # Path 1: JWT Bearer.
    if authorization and jwt_decode is not None:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
            try:
                payload = jwt_decode(token)
                if payload:
                    user = {
                        "user_id": payload.get("sub"),
                        "email": payload.get("email"),
                        "tenant_id": payload.get("tenant_id"),
                        "tier": payload.get("tier", "community"),
                        "roles": payload.get("roles", []),
                        "permissions": payload.get("permissions", []),
                        "auth_method": "jwt",
                    }
                    if user["user_id"]:
                        return user
            except Exception as exc:  # noqa: BLE001 - propagate minimally
                logger.debug("JWT decode rejected: %s", exc)

    # Path 2: API key.
    if api_key_header:
        v = validator or default_validator()
        record = v.validate(api_key_header.strip())
        if record is not None:
            return record.to_user()

    return None
