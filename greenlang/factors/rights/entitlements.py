# -*- coding: utf-8 -*-
"""Phase 1 source-level entitlement model.

This is **distinct** from `greenlang/factors/entitlements.py` which
holds *pack-SKU* entitlements (FREIGHT_PREMIUM, etc.) used by the
existing pack-level licensing middleware. The Phase 1 model here
operates at **source URN** granularity and is the unit the
SourceRightsService consults.

Storage: file-backed YAML for v0.1 alpha. v0.5+ may migrate to a
SQLite/Postgres-backed store; the public API in this module remains
stable across that migration.

Schema:

    entitlements:
      - tenant_id: <str>
        source_urn: <urn:gl:source:...>
        pack_urn:    <urn:gl:pack:...>      # optional
        entitlement_type: source_access | pack_access | private_owner
        status: active | expired | revoked
        valid_from: <ISO-8601>
        valid_until: <ISO-8601 | null>
        approved_by: <human:...>
        contract_ref: <internal contract id>
        notes: <freeform>
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntitlementType(str, enum.Enum):
    SOURCE_ACCESS = "source_access"
    PACK_ACCESS = "pack_access"
    PRIVATE_OWNER = "private_owner"


class EntitlementStatus(str, enum.Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitlementRecord:
    """One row of the source-level entitlement ledger."""

    tenant_id: str
    source_urn: Optional[str]
    pack_urn: Optional[str]
    entitlement_type: EntitlementType
    status: EntitlementStatus
    valid_from: str
    valid_until: Optional[str]
    approved_by: str
    contract_ref: Optional[str] = None
    notes: Optional[str] = None

    def is_live(self, at: Optional[datetime] = None) -> bool:
        """Return True iff status is active AND we're inside the validity window."""
        if self.status != EntitlementStatus.ACTIVE:
            return False
        now = at or datetime.now(timezone.utc)
        if self.valid_from:
            try:
                vf = _parse_iso(self.valid_from)
                if vf and now < vf:
                    return False
            except ValueError:
                logger.warning("invalid valid_from %r in entitlement", self.valid_from)
        if self.valid_until:
            try:
                vu = _parse_iso(self.valid_until)
                if vu and now > vu:
                    return False
            except ValueError:
                logger.warning("invalid valid_until %r in entitlement", self.valid_until)
        return True


def _parse_iso(s: str) -> Optional[datetime]:
    if not isinstance(s, str) or not s.strip():
        return None
    txt = s.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class EntitlementStore:
    """In-memory lookup over EntitlementRecord rows.

    Construct with ``EntitlementStore.from_records(...)`` or load from
    a YAML file via ``load_alpha_entitlements()``.
    """

    _by_tenant: Dict[str, List[EntitlementRecord]] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls, records: Iterable[EntitlementRecord]
    ) -> "EntitlementStore":
        store = cls()
        for r in records:
            store._by_tenant.setdefault(r.tenant_id, []).append(r)
        return store

    def list_for_tenant(self, tenant_id: str) -> List[EntitlementRecord]:
        return list(self._by_tenant.get(tenant_id, ()))

    def has_active_source_access(
        self, tenant_id: str, source_urn: str, at: Optional[datetime] = None
    ) -> bool:
        for r in self._by_tenant.get(tenant_id, ()):
            if r.entitlement_type not in (
                EntitlementType.SOURCE_ACCESS,
                EntitlementType.PACK_ACCESS,
                EntitlementType.PRIVATE_OWNER,
            ):
                continue
            if r.source_urn and r.source_urn != source_urn:
                continue
            if not r.source_urn and r.entitlement_type != EntitlementType.PACK_ACCESS:
                continue
            if r.is_live(at):
                return True
        return False

    def has_active_pack_access(
        self, tenant_id: str, pack_urn: str, at: Optional[datetime] = None
    ) -> bool:
        for r in self._by_tenant.get(tenant_id, ()):
            if r.entitlement_type not in (
                EntitlementType.PACK_ACCESS,
                EntitlementType.SOURCE_ACCESS,
                EntitlementType.PRIVATE_OWNER,
            ):
                continue
            if r.pack_urn and r.pack_urn != pack_urn:
                continue
            if not r.pack_urn:
                # SOURCE_ACCESS rows without pack_urn cover all packs
                # of that source; the caller already knows the source
                # match. We do NOT verify the source linkage here —
                # the SourceRightsService cross-checks if needed.
                if r.entitlement_type == EntitlementType.SOURCE_ACCESS and r.is_live(at):
                    return True
                continue
            if r.is_live(at):
                return True
        return False

    def is_private_owner(
        self, tenant_id: str, source_urn: str, at: Optional[datetime] = None
    ) -> bool:
        for r in self._by_tenant.get(tenant_id, ()):
            if r.entitlement_type != EntitlementType.PRIVATE_OWNER:
                continue
            if r.source_urn != source_urn:
                continue
            if r.is_live(at):
                return True
        return False


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


_DEFAULT_ALPHA_FILE = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "entitlements"
    / "alpha_v0_1.yaml"
)


def load_alpha_entitlements(
    path: Optional[Path] = None,
) -> EntitlementStore:
    """Load entitlements from the alpha YAML file.

    Returns an empty store if the file is missing — alpha is
    public-only by default; entitlements are only required when a
    `tenant_entitlement_required` source is queried.
    """
    p = path or _DEFAULT_ALPHA_FILE
    if not p.is_file():
        logger.info("entitlements file %s not found; loading empty store", p)
        return EntitlementStore()
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        raise RuntimeError("PyYAML required to load entitlements")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = raw.get("entitlements") or []
    out: List[EntitlementRecord] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            out.append(
                EntitlementRecord(
                    tenant_id=str(r["tenant_id"]),
                    source_urn=r.get("source_urn"),
                    pack_urn=r.get("pack_urn"),
                    entitlement_type=EntitlementType(r["entitlement_type"]),
                    status=EntitlementStatus(r.get("status", "active")),
                    valid_from=str(r["valid_from"]),
                    valid_until=r.get("valid_until"),
                    approved_by=str(r.get("approved_by", "")),
                    contract_ref=r.get("contract_ref"),
                    notes=r.get("notes"),
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("dropping malformed entitlement %r: %s", r, exc)
    return EntitlementStore.from_records(out)
