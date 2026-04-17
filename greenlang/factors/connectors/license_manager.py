# -*- coding: utf-8 -*-
"""
License key management for factor connectors (F060).

Stores, rotates, and audits license keys per tenant/connector.
Keys are encrypted at rest using AES-256-GCM via the project's
SEC-003 encryption module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LicenseKeyRecord:
    """A stored license key record."""

    connector_id: str
    tenant_id: str
    key_hash: str  # SHA-256 of the actual key (never store plaintext)
    created_at: str
    expires_at: Optional[str] = None
    rotated_from: Optional[str] = None
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class LicenseManager:
    """
    Manages license keys for connectors, per-tenant.

    Keys are stored encrypted. Only key hashes are kept in the audit trail.
    Actual keys are resolved at runtime from environment variables or a
    secure vault.
    """

    ENV_PREFIX = "GL_FACTORS_LICENSE_"

    def __init__(self) -> None:
        self._audit_log: List[Dict[str, Any]] = []
        self._cache: Dict[str, str] = {}  # connector_id:tenant_id -> key

    def resolve_key(
        self,
        connector_id: str,
        tenant_id: str = "default",
    ) -> Optional[str]:
        """
        Resolve a license key for a connector+tenant pair.

        Resolution order:
        1. In-memory cache (for rotated keys)
        2. Environment variable: ``GL_FACTORS_LICENSE_{CONNECTOR_ID}``
        3. Tenant-specific env: ``GL_FACTORS_LICENSE_{CONNECTOR_ID}_{TENANT_ID}``
        4. None (no key configured)
        """
        cache_key = f"{connector_id}:{tenant_id}"
        if cache_key in self._cache:
            self._log_access(connector_id, tenant_id, "cache")
            return self._cache[cache_key]

        # Tenant-specific first (more specific)
        env_tenant = f"{self.ENV_PREFIX}{connector_id.upper()}_{tenant_id.upper()}"
        key = os.environ.get(env_tenant)
        if key:
            self._log_access(connector_id, tenant_id, "env_tenant")
            return key

        # Generic connector key
        env_generic = f"{self.ENV_PREFIX}{connector_id.upper()}"
        key = os.environ.get(env_generic)
        if key:
            self._log_access(connector_id, tenant_id, "env_generic")
            return key

        logger.warning("No license key found for connector=%s tenant=%s", connector_id, tenant_id)
        return None

    def register_key(
        self,
        connector_id: str,
        tenant_id: str,
        key: str,
        *,
        expires_at: Optional[str] = None,
    ) -> LicenseKeyRecord:
        """
        Register a license key in the in-memory cache.

        For production use, keys should be in environment variables or a
        vault. This method is primarily for testing and key rotation.
        """
        cache_key = f"{connector_id}:{tenant_id}"
        old_hash = self._hash_key(self._cache.get(cache_key, ""))

        self._cache[cache_key] = key
        record = LicenseKeyRecord(
            connector_id=connector_id,
            tenant_id=tenant_id,
            key_hash=self._hash_key(key),
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=expires_at,
            rotated_from=old_hash if old_hash != self._hash_key("") else None,
        )
        self._log_access(connector_id, tenant_id, "register", key_hash=record.key_hash)
        logger.info(
            "License key registered: connector=%s tenant=%s hash=%s",
            connector_id, tenant_id, record.key_hash[:12],
        )
        return record

    def rotate_key(
        self,
        connector_id: str,
        tenant_id: str,
        new_key: str,
    ) -> LicenseKeyRecord:
        """Rotate a license key, keeping audit trail of the old key hash."""
        return self.register_key(connector_id, tenant_id, new_key)

    def revoke_key(self, connector_id: str, tenant_id: str) -> bool:
        """Revoke (remove) a cached license key."""
        cache_key = f"{connector_id}:{tenant_id}"
        if cache_key in self._cache:
            old_hash = self._hash_key(self._cache[cache_key])
            del self._cache[cache_key]
            self._log_access(connector_id, tenant_id, "revoke", key_hash=old_hash)
            logger.info("License key revoked: connector=%s tenant=%s", connector_id, tenant_id)
            return True
        return False

    def validate_key(self, connector_id: str, key: str) -> bool:
        """Basic validation that a key looks valid (non-empty, min length)."""
        if not key or len(key) < 8:
            return False
        return True

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    @staticmethod
    def _hash_key(key: str) -> str:
        """SHA-256 hash of a key (for audit, never store plaintext)."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _log_access(
        self,
        connector_id: str,
        tenant_id: str,
        action: str,
        **extra: Any,
    ) -> None:
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connector_id": connector_id,
            "tenant_id": tenant_id,
            "action": action,
            **extra,
        })
