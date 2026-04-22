# -*- coding: utf-8 -*-
"""
Enterprise tenant overlay system (F064).

Allows Enterprise-tier customers to supply custom emission factors
(from internal energy audits, supplier-specific data) that override
catalog defaults for their tenant scope.

Features:
- Per-tenant factor overlays with date-range validity
- Resolution logic: overlay > catalog default (same factor_id)
- CRUD operations for overlay management
- Strict tenant isolation (Tenant A never sees Tenant B overlays)
- Immutable audit trail for all overlay changes
- Transparent per-tenant envelope encryption for ``customer_private``
  overrides via ``TenantVaultTransit`` (Vault Transit in prod, AES-256
  GCM dev fallback) — plaintext override values never touch the DB.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from greenlang.factors.security.tenant_vault_transit import (
    TenantKeyAccessError,
    TenantVaultTransit,
    default_transit,
)

logger = logging.getLogger(__name__)

# Redistribution class that triggers transparent encryption of the
# override value. Kept as a module-local constant so we don't create a
# circular import against greenlang.factors.billing.skus.
REDISTRIBUTION_CUSTOMER_PRIVATE = "customer_private"

# Marker used in the persisted ``override_value`` column for encrypted
# rows. Real override values are finite floats; NaN with a distinctive
# bit pattern is a safe sentinel. We store the ciphertext separately
# and set this sentinel on the numeric column so any legacy reader gets
# an obvious "do not use this number" signal.
_ENCRYPTED_SENTINEL: float = float("nan")


def _float_to_plaintext(value: float) -> bytes:
    """Encode a float into an 8-byte big-endian IEEE-754 double.

    Using a fixed-width binary encoding instead of ``repr(value)``
    keeps ciphertext length constant and avoids locale-sensitive
    round-tripping.
    """
    return struct.pack(">d", float(value))


def _plaintext_to_float(raw: bytes) -> float:
    """Inverse of :func:`_float_to_plaintext`."""
    if len(raw) != 8:
        raise ValueError(
            "private override plaintext must be 8 bytes, got %d" % len(raw)
        )
    return struct.unpack(">d", raw)[0]


def _is_finite(value: float) -> bool:
    """Return True only for regular finite floats (not NaN / infinity)."""
    return isinstance(value, float) and value == value and value not in (
        float("inf"), float("-inf")
    )


@dataclass
class TenantOverlay:
    """A single tenant factor overlay record.

    For overlays flagged ``redistribution_class == "customer_private"``
    the ``override_value`` column is an unreadable sentinel on disk;
    the real value is stored in ``override_value_ciphertext`` and only
    materialised at read time via ``TenantVaultTransit.decrypt``.
    """

    overlay_id: str
    tenant_id: str
    factor_id: str
    override_value: float
    override_unit: str = "kg_co2e"
    valid_from: str = ""  # ISO date YYYY-MM-DD
    valid_to: Optional[str] = None  # ISO date or None (open-ended)
    source: str = ""  # "supplier_audit", "internal_energy_audit", etc.
    notes: str = ""
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    active: bool = True
    redistribution_class: str = "internal_only"
    override_value_ciphertext: Optional[str] = None
    override_value_digest: Optional[str] = None

    def is_valid_on(self, check_date: Optional[str] = None) -> bool:
        """Check if this overlay is valid on the given date (YYYY-MM-DD)."""
        if not self.active:
            return False
        d = check_date or date.today().isoformat()
        if self.valid_from and d < self.valid_from:
            return False
        if self.valid_to and d > self.valid_to:
            return False
        return True

    def is_private(self) -> bool:
        """Return True if this overlay requires tenant-key decryption."""
        return self.redistribution_class == REDISTRIBUTION_CUSTOMER_PRIVATE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OverlayAuditEntry:
    """Immutable audit record for overlay changes."""

    timestamp: str
    tenant_id: str
    overlay_id: str
    action: str  # "create", "update", "delete"
    actor: str
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class TenantOverlayRegistry:
    """
    Production-grade tenant overlay management.

    Supports SQLite (local) or Postgres (production) backends.
    Enforces strict tenant isolation — all queries are scoped by tenant_id.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        transit: Optional[TenantVaultTransit] = None,
    ) -> None:
        self._db_path = db_path
        self._overlays: Dict[str, Dict[str, TenantOverlay]] = {}  # tenant_id -> {overlay_id -> overlay}
        self._audit: List[OverlayAuditEntry] = []
        self._transit = transit  # resolved lazily via _get_transit()
        if db_path:
            self._init_db(db_path)

    def _get_transit(self) -> TenantVaultTransit:
        """Return the configured transit client, defaulting to the singleton."""
        if self._transit is None:
            self._transit = default_transit()
        return self._transit

    def _init_db(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tenant_overlays (
                overlay_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                factor_id TEXT NOT NULL,
                override_value REAL,
                override_unit TEXT NOT NULL DEFAULT 'kg_co2e',
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                source TEXT,
                notes TEXT,
                created_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active BOOLEAN NOT NULL DEFAULT 1,
                redistribution_class TEXT NOT NULL DEFAULT 'internal_only',
                override_value_ciphertext TEXT,
                override_value_digest TEXT
            )
        """)
        # Best-effort idempotent migration for existing DBs.  SQLite
        # swallows errors on ALTER when the column already exists;
        # we catch per-statement so startup never fails.
        for ddl in (
            "ALTER TABLE tenant_overlays ADD COLUMN redistribution_class "
            "TEXT NOT NULL DEFAULT 'internal_only'",
            "ALTER TABLE tenant_overlays ADD COLUMN override_value_ciphertext TEXT",
            "ALTER TABLE tenant_overlays ADD COLUMN override_value_digest TEXT",
        ):
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError:
                pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_to_tenant_factor "
            "ON tenant_overlays (tenant_id, factor_id, valid_from)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_to_tenant_active "
            "ON tenant_overlays (tenant_id, active)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tenant_overlay_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                overlay_id TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                details_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_overlay(
        self,
        tenant_id: str,
        factor_id: str,
        override_value: float,
        *,
        override_unit: str = "kg_co2e",
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        source: str = "",
        notes: str = "",
        created_by: str = "",
        redistribution_class: str = "internal_only",
    ) -> TenantOverlay:
        """Create a new tenant overlay.

        When ``redistribution_class == "customer_private"``:
          * the plaintext override value is encrypted through
            ``TenantVaultTransit`` under the tenant's key,
          * the on-disk ``override_value`` column is written as a NaN
            sentinel (never the real number),
          * the in-memory object still carries the plaintext so the
            resolution path doesn't need a decrypt round-trip on the
            hot path after write.
        """
        ciphertext: Optional[str] = None
        digest: Optional[str] = None
        stored_value = override_value

        if redistribution_class == REDISTRIBUTION_CUSTOMER_PRIVATE:
            ciphertext, digest = self._encrypt_private_value(tenant_id, override_value)
            stored_value = _ENCRYPTED_SENTINEL

        overlay = TenantOverlay(
            overlay_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            factor_id=factor_id,
            override_value=override_value,
            override_unit=override_unit,
            valid_from=valid_from or date.today().isoformat(),
            valid_to=valid_to,
            source=source,
            notes=notes,
            created_by=created_by,
            created_at=self._now(),
            updated_at=self._now(),
            redistribution_class=redistribution_class,
            override_value_ciphertext=ciphertext,
            override_value_digest=digest,
        )

        # In-memory store
        self._overlays.setdefault(tenant_id, {})[overlay.overlay_id] = overlay

        # Persist — the DB row carries the sentinel, never the plaintext.
        if self._db_path:
            self._persist_overlay(overlay, on_disk_value=stored_value)

        # Audit: private overlays log only the digest + pointer, never
        # the plaintext float.
        if overlay.is_private():
            self._log_audit(
                tenant_id, overlay.overlay_id, "create", created_by,
                old_value=None, new_value=None,
                details={
                    "redistribution_class": redistribution_class,
                    "value_digest": digest,
                    "ciphertext_ptr": (ciphertext or "")[:16],
                },
            )
            logger.info(
                "Overlay created (private): tenant=%s factor=%s digest=%s id=%s",
                tenant_id, factor_id, digest, overlay.overlay_id,
            )
        else:
            self._log_audit(
                tenant_id, overlay.overlay_id, "create", created_by,
                new_value=override_value,
            )
            logger.info(
                "Overlay created: tenant=%s factor=%s value=%s id=%s",
                tenant_id, factor_id, override_value, overlay.overlay_id,
            )
        return overlay

    def update_overlay(
        self,
        tenant_id: str,
        overlay_id: str,
        *,
        override_value: Optional[float] = None,
        valid_to: Optional[str] = None,
        notes: Optional[str] = None,
        updated_by: str = "",
    ) -> Optional[TenantOverlay]:
        """Update an existing overlay (strict tenant isolation).

        Updating the numeric value of a ``customer_private`` overlay
        re-encrypts through Vault Transit and refreshes the stored
        ciphertext / digest. The audit entry for a private update logs
        digests only.
        """
        overlay = self._get_overlay(tenant_id, overlay_id)
        if not overlay:
            return None

        old_value = overlay.override_value
        is_private = overlay.is_private()

        if override_value is not None:
            overlay.override_value = override_value
            if is_private:
                ciphertext, digest = self._encrypt_private_value(
                    tenant_id, override_value
                )
                overlay.override_value_ciphertext = ciphertext
                overlay.override_value_digest = digest
        if valid_to is not None:
            overlay.valid_to = valid_to
        if notes is not None:
            overlay.notes = notes
        overlay.updated_at = self._now()

        if self._db_path:
            on_disk = (
                _ENCRYPTED_SENTINEL if is_private else overlay.override_value
            )
            self._update_overlay_db(overlay, on_disk_value=on_disk)

        if is_private:
            self._log_audit(
                tenant_id, overlay_id, "update", updated_by,
                old_value=None, new_value=None,
                details={
                    "redistribution_class": overlay.redistribution_class,
                    "value_digest": overlay.override_value_digest,
                    "ciphertext_ptr": (overlay.override_value_ciphertext or "")[:16],
                },
            )
        else:
            self._log_audit(
                tenant_id, overlay_id, "update", updated_by,
                old_value=old_value, new_value=overlay.override_value,
            )
        return overlay

    def delete_overlay(
        self,
        tenant_id: str,
        overlay_id: str,
        *,
        deleted_by: str = "",
    ) -> bool:
        """Soft-delete an overlay (sets active=False, strict tenant isolation)."""
        overlay = self._get_overlay(tenant_id, overlay_id)
        if not overlay:
            return False

        overlay.active = False
        overlay.updated_at = self._now()

        if self._db_path:
            on_disk = (
                _ENCRYPTED_SENTINEL if overlay.is_private() else overlay.override_value
            )
            self._update_overlay_db(overlay, on_disk_value=on_disk)

        if overlay.is_private():
            self._log_audit(
                tenant_id, overlay_id, "delete", deleted_by,
                details={
                    "redistribution_class": overlay.redistribution_class,
                    "value_digest": overlay.override_value_digest,
                },
            )
        else:
            self._log_audit(
                tenant_id, overlay_id, "delete", deleted_by,
                old_value=overlay.override_value,
            )
        logger.info("Overlay deleted: tenant=%s id=%s", tenant_id, overlay_id)
        return True

    # ------------------------------------------------------------------
    # Customer-private decrypt helper
    # ------------------------------------------------------------------

    def decrypt_private_value(
        self, tenant_id: str, overlay: TenantOverlay
    ) -> float:
        """Materialise the plaintext override value of a private overlay.

        Raises
        ------
        TenantKeyAccessError
            If ``tenant_id`` does not match the overlay's owner.
        ValueError
            If the overlay is not flagged ``customer_private`` or has
            no ciphertext on record.
        """
        if overlay.tenant_id != tenant_id:
            raise TenantKeyAccessError(
                "Cross-tenant decrypt denied: overlay owner=%r caller=%r"
                % (overlay.tenant_id, tenant_id)
            )
        if not overlay.is_private():
            raise ValueError("Overlay is not customer_private; read override_value directly.")
        if not overlay.override_value_ciphertext:
            raise ValueError(
                "Overlay %r has no ciphertext on record." % overlay.overlay_id
            )
        plaintext = self._get_transit().decrypt(
            tenant_id, overlay.override_value_ciphertext
        )
        return _plaintext_to_float(plaintext)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_overlay(self, tenant_id: str, overlay_id: str) -> Optional[TenantOverlay]:
        """Get a single overlay (strict tenant isolation)."""
        return self._get_overlay(tenant_id, overlay_id)

    def list_overlays(
        self,
        tenant_id: str,
        *,
        active_only: bool = True,
        factor_id: Optional[str] = None,
    ) -> List[TenantOverlay]:
        """List overlays for a tenant (strict tenant isolation)."""
        tenant_overlays = self._overlays.get(tenant_id, {})
        results = list(tenant_overlays.values())
        if active_only:
            results = [o for o in results if o.active]
        if factor_id:
            results = [o for o in results if o.factor_id == factor_id]
        return sorted(results, key=lambda o: o.valid_from)

    def resolve_factor(
        self,
        tenant_id: str,
        factor_id: str,
        *,
        check_date: Optional[str] = None,
    ) -> Optional[TenantOverlay]:
        """
        Resolve the active overlay for a factor_id + tenant_id pair.

        Returns the most recently created valid overlay, or None if
        no overlay exists (meaning the catalog default should be used).
        """
        overlays = self.list_overlays(tenant_id, factor_id=factor_id)
        valid = [o for o in overlays if o.is_valid_on(check_date)]
        if not valid:
            return None
        # Most recent valid_from wins
        return max(valid, key=lambda o: o.valid_from)

    def merge_search_results(
        self,
        tenant_id: str,
        catalog_results: List[Dict[str, Any]],
        *,
        check_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Merge tenant overlays into search results.

        For each catalog result, if a tenant overlay exists for that
        factor_id, replace the value with the overlay value.
        """
        merged = []
        for result in catalog_results:
            fid = result.get("factor_id", "")
            overlay = self.resolve_factor(tenant_id, fid, check_date=check_date)
            if overlay:
                merged_result = dict(result)
                # For customer_private rows the in-memory object still
                # carries the plaintext (populated at write time). When
                # the overlay was loaded from disk without a decrypt
                # step, override_value will be NaN and callers MUST go
                # through ``decrypt_private_value``.
                if overlay.is_private():
                    value = (
                        overlay.override_value
                        if _is_finite(overlay.override_value)
                        else self.decrypt_private_value(tenant_id, overlay)
                    )
                else:
                    value = overlay.override_value
                merged_result["co2e_total"] = value
                merged_result["unit"] = overlay.override_unit
                merged_result["_overlay_id"] = overlay.overlay_id
                merged_result["_overlay_source"] = overlay.source
                merged_result["_overlay_redistribution_class"] = (
                    overlay.redistribution_class
                )
                merged.append(merged_result)
            else:
                merged.append(result)
        return merged

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    @property
    def audit_log(self) -> List[OverlayAuditEntry]:
        return list(self._audit)

    def audit_for_tenant(self, tenant_id: str) -> List[OverlayAuditEntry]:
        return [e for e in self._audit if e.tenant_id == tenant_id]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_overlay(self, tenant_id: str, overlay_id: str) -> Optional[TenantOverlay]:
        tenant_overlays = self._overlays.get(tenant_id, {})
        return tenant_overlays.get(overlay_id)

    def _log_audit(
        self,
        tenant_id: str,
        overlay_id: str,
        action: str,
        actor: str,
        *,
        old_value: Optional[float] = None,
        new_value: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = OverlayAuditEntry(
            timestamp=self._now(),
            tenant_id=tenant_id,
            overlay_id=overlay_id,
            action=action,
            actor=actor,
            old_value=old_value,
            new_value=new_value,
            details=details or {},
        )
        self._audit.append(entry)
        if self._db_path:
            self._persist_audit(entry)

    def _encrypt_private_value(
        self, tenant_id: str, value: float
    ) -> tuple[str, str]:
        """Encrypt a private override value; return ``(ciphertext, digest)``."""
        plaintext = _float_to_plaintext(value)
        transit = self._get_transit()
        ciphertext = transit.encrypt(tenant_id, plaintext)
        digest = hashlib.sha256(plaintext).hexdigest()
        return ciphertext, digest

    def _persist_overlay(
        self,
        overlay: TenantOverlay,
        *,
        on_disk_value: Optional[float] = None,
    ) -> None:
        """Write the overlay to SQLite.

        ``on_disk_value`` lets callers override what goes into the
        ``override_value`` column (e.g., the NaN sentinel for
        ``customer_private`` rows).
        """
        stored = on_disk_value if on_disk_value is not None else overlay.override_value
        # NaN sentinels persist cleanly as SQL NULL — any legacy reader
        # treating NULL as "missing" gets an obvious error, which is
        # exactly the behaviour we want for encrypted rows.
        if isinstance(stored, float) and stored != stored:
            stored = None
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            INSERT OR REPLACE INTO tenant_overlays (
                overlay_id, tenant_id, factor_id, override_value, override_unit,
                valid_from, valid_to, source, notes, created_by, created_at,
                updated_at, active, redistribution_class,
                override_value_ciphertext, override_value_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                overlay.overlay_id, overlay.tenant_id, overlay.factor_id,
                stored, overlay.override_unit,
                overlay.valid_from, overlay.valid_to, overlay.source,
                overlay.notes, overlay.created_by, overlay.created_at,
                overlay.updated_at, int(overlay.active),
                overlay.redistribution_class,
                overlay.override_value_ciphertext,
                overlay.override_value_digest,
            ),
        )
        conn.commit()
        conn.close()

    def _update_overlay_db(
        self,
        overlay: TenantOverlay,
        *,
        on_disk_value: Optional[float] = None,
    ) -> None:
        self._persist_overlay(overlay, on_disk_value=on_disk_value)

    def _persist_audit(self, entry: OverlayAuditEntry) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            INSERT INTO tenant_overlay_audit (
                timestamp, tenant_id, overlay_id, action, actor,
                old_value, new_value, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp, entry.tenant_id, entry.overlay_id,
                entry.action, entry.actor, entry.old_value, entry.new_value,
                json.dumps(entry.details, default=str),
            ),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Legacy compat (from original stub)
    # ------------------------------------------------------------------

    def register_sqlite(self, tenant_id: str, sqlite_path: str) -> None:
        """Legacy: register an external SQLite overlay bundle path."""
        logger.info("Legacy register_sqlite: tenant=%s path=%s", tenant_id, sqlite_path)

    def list_paths(self, tenant_id: str) -> List[str]:
        """Legacy: list registered SQLite paths."""
        return []
