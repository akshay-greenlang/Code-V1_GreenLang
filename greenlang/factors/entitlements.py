# -*- coding: utf-8 -*-
"""
Per-pack entitlement enforcement (Phase F8).

Premium packs (Freight, Product-Carbon / LCI, CBAM-premium, Finance,
Land, Agrifood) are sold as separate SKUs on top of the Community /
Developer Pro / Consulting / Enterprise tiers.  This module answers:

    "Is tenant T allowed to read/resolve factors from pack P?"

Entitlements persist in a SQLite/Postgres table keyed by
``(tenant_id, pack_sku)`` with optional expiry + seat / volume caps.
OEM redistribution is a tri-state flag (forbidden / internal_only /
redistributable) captured alongside the entitlement.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

logger = logging.getLogger(__name__)


class PackSKU:
    """Known premium pack SKUs."""

    ELECTRICITY_PREMIUM = "electricity_premium"
    FREIGHT_PREMIUM = "freight_premium"
    PRODUCT_CARBON_PREMIUM = "product_carbon_premium"
    EPD_PREMIUM = "epd_premium"
    AGRIFOOD_PREMIUM = "agrifood_premium"
    FINANCE_PREMIUM = "finance_premium"
    CBAM_PREMIUM = "cbam_premium"
    LAND_PREMIUM = "land_premium"

    ALL = [
        ELECTRICITY_PREMIUM,
        FREIGHT_PREMIUM,
        PRODUCT_CARBON_PREMIUM,
        EPD_PREMIUM,
        AGRIFOOD_PREMIUM,
        FINANCE_PREMIUM,
        CBAM_PREMIUM,
        LAND_PREMIUM,
    ]


class OEMRights(str):
    """Tri-state OEM redistribution rights."""

    FORBIDDEN = "forbidden"
    INTERNAL_ONLY = "internal_only"
    REDISTRIBUTABLE = "redistributable"


@dataclass
class Entitlement:
    """One row of the entitlement ledger."""

    tenant_id: str
    pack_sku: str
    granted_at: str
    expires_at: Optional[str]
    oem_rights: str
    seat_cap: Optional[int]
    volume_cap_per_month: Optional[int]
    notes: Optional[str] = None
    active: bool = True

    def is_live(self, at: Optional[date] = None) -> bool:
        if not self.active:
            return False
        if self.expires_at:
            exp = date.fromisoformat(self.expires_at[:10])
            today = at or date.today()
            if exp < today:
                return False
        return True


class EntitlementError(RuntimeError):
    pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS factor_pack_entitlements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id         TEXT NOT NULL,
    pack_sku          TEXT NOT NULL,
    granted_at        TEXT NOT NULL,
    expires_at        TEXT,
    oem_rights        TEXT NOT NULL DEFAULT 'forbidden',
    seat_cap          INTEGER,
    volume_cap_per_month INTEGER,
    active            INTEGER NOT NULL DEFAULT 1,
    notes             TEXT,
    UNIQUE (tenant_id, pack_sku)
);
CREATE INDEX IF NOT EXISTS idx_ent_tenant ON factor_pack_entitlements (tenant_id);
CREATE INDEX IF NOT EXISTS idx_ent_sku    ON factor_pack_entitlements (pack_sku);
"""


class EntitlementRegistry:
    """Thread-safe SQLite-backed entitlement registry."""

    def __init__(self, sqlite_path: Union[str, Path]) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.sqlite_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def grant(
        self,
        *,
        tenant_id: str,
        pack_sku: str,
        expires_at: Optional[str] = None,
        oem_rights: str = OEMRights.FORBIDDEN,
        seat_cap: Optional[int] = None,
        volume_cap_per_month: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Entitlement:
        if pack_sku not in PackSKU.ALL:
            raise EntitlementError(
                "Unknown pack_sku %r; expected one of %s" % (pack_sku, PackSKU.ALL)
            )
        if oem_rights not in (OEMRights.FORBIDDEN, OEMRights.INTERNAL_ONLY, OEMRights.REDISTRIBUTABLE):
            raise EntitlementError("Invalid oem_rights %r" % oem_rights)
        granted_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO factor_pack_entitlements (
                    tenant_id, pack_sku, granted_at, expires_at, oem_rights,
                    seat_cap, volume_cap_per_month, active, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(tenant_id, pack_sku) DO UPDATE SET
                    granted_at = excluded.granted_at,
                    expires_at = excluded.expires_at,
                    oem_rights = excluded.oem_rights,
                    seat_cap = excluded.seat_cap,
                    volume_cap_per_month = excluded.volume_cap_per_month,
                    active = 1,
                    notes = excluded.notes
                """,
                (
                    tenant_id, pack_sku, granted_at, expires_at, oem_rights,
                    seat_cap, volume_cap_per_month, notes,
                ),
            )
        return Entitlement(
            tenant_id=tenant_id,
            pack_sku=pack_sku,
            granted_at=granted_at,
            expires_at=expires_at,
            oem_rights=oem_rights,
            seat_cap=seat_cap,
            volume_cap_per_month=volume_cap_per_month,
            notes=notes,
        )

    def revoke(self, *, tenant_id: str, pack_sku: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE factor_pack_entitlements SET active = 0 "
                "WHERE tenant_id = ? AND pack_sku = ?",
                (tenant_id, pack_sku),
            )
            return cur.rowcount > 0

    def is_entitled(
        self,
        *,
        tenant_id: str,
        pack_sku: str,
        at: Optional[date] = None,
    ) -> bool:
        ent = self.get(tenant_id=tenant_id, pack_sku=pack_sku)
        return ent is not None and ent.is_live(at=at)

    def get(
        self, *, tenant_id: str, pack_sku: str
    ) -> Optional[Entitlement]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT tenant_id, pack_sku, granted_at, expires_at, oem_rights,
                       seat_cap, volume_cap_per_month, active, notes
                FROM factor_pack_entitlements
                WHERE tenant_id = ? AND pack_sku = ?
                """,
                (tenant_id, pack_sku),
            ).fetchone()
        if row is None:
            return None
        return Entitlement(
            tenant_id=row[0], pack_sku=row[1], granted_at=row[2],
            expires_at=row[3], oem_rights=row[4], seat_cap=row[5],
            volume_cap_per_month=row[6], active=bool(row[7]), notes=row[8],
        )

    def list_for_tenant(self, tenant_id: str) -> List[Entitlement]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tenant_id, pack_sku, granted_at, expires_at, oem_rights,
                       seat_cap, volume_cap_per_month, active, notes
                FROM factor_pack_entitlements
                WHERE tenant_id = ?
                ORDER BY pack_sku ASC
                """,
                (tenant_id,),
            ).fetchall()
        return [
            Entitlement(
                tenant_id=r[0], pack_sku=r[1], granted_at=r[2],
                expires_at=r[3], oem_rights=r[4], seat_cap=r[5],
                volume_cap_per_month=r[6], active=bool(r[7]), notes=r[8],
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Method-pack → SKU mapping
# ---------------------------------------------------------------------------

PACK_SKU_FOR_METHOD_PROFILE: Dict[str, str] = {
    "freight_iso_14083": PackSKU.FREIGHT_PREMIUM,
    "product_carbon": PackSKU.PRODUCT_CARBON_PREMIUM,
    "land_removals": PackSKU.LAND_PREMIUM,
    "finance_proxy": PackSKU.FINANCE_PREMIUM,
    "eu_cbam": PackSKU.CBAM_PREMIUM,
}


def pack_sku_for_profile(method_profile: str) -> Optional[str]:
    """Return the premium SKU a method profile requires, or None if open-core."""
    return PACK_SKU_FOR_METHOD_PROFILE.get(method_profile)


# ---------------------------------------------------------------------------
# Track C-5: OEM redistribution guard
# ---------------------------------------------------------------------------
#
# Hooked into the licensing guard (see e.g.
# ``greenlang.factors.security.license_check``) so any factor served via
# an OEM key is gated by both:
#   1. the parent OEM's redistribution_grants (set at signup), and
#   2. the per-factor ``licensing.redistribution_class`` field on the
#      Canonical Factor Record.
#
# A factor is ONLY redistributable through an OEM when its license_class
# appears in the OEM's grant. The check is intentionally cheap and
# deterministic so it can sit in the request-time hot path.


def _factor_license_class(factor: Any) -> Optional[str]:
    """Pull the redistribution-relevant license class off a factor.

    Accepts a dict or any object with a ``license_class`` /
    ``licensing`` / ``source`` attribute - the canonical record changes
    shape across editions, so we probe a few common surfaces before
    giving up.
    """
    if factor is None:
        return None
    # Dict shape (most common in API serialisation path).
    if isinstance(factor, dict):
        # Direct field
        cls = factor.get("license_class") or factor.get("redistribution_class")
        if cls:
            return str(cls)
        # Nested licensing block
        lic = factor.get("licensing") or factor.get("license") or {}
        if isinstance(lic, dict):
            cls = (
                lic.get("license_class")
                or lic.get("redistribution")
                or lic.get("redistribution_class")
            )
            if cls:
                return str(cls)
        # Nested source block
        src = factor.get("source") or {}
        if isinstance(src, dict):
            cls = src.get("license_class")
            if cls:
                return str(cls)
        return None
    # Object shape - canonical_v2 record / dataclass.
    for attr in ("license_class", "redistribution_class"):
        val = getattr(factor, attr, None)
        if val:
            return str(val)
    lic = getattr(factor, "licensing", None) or getattr(factor, "license", None)
    if lic is not None:
        for attr in ("license_class", "redistribution", "redistribution_class"):
            val = getattr(lic, attr, None)
            if val:
                return str(val)
    src = getattr(factor, "source", None)
    if src is not None:
        val = getattr(src, "license_class", None)
        if val:
            return str(val)
    return None


def check_oem_redistribution(oem_id: str, factor: Any) -> bool:
    """Return True iff ``factor`` is within the OEM's redistribution grant.

    This is the load-bearing license guard the API layer hooks into.
    It deliberately fails closed: any non-trivial error (missing OEM,
    factor with no license class, lookup exception) returns False so a
    misconfigured request never silently leaks a licensed factor.

    Args:
        oem_id: OEM partner id presented by the caller's API key.
        factor: Canonical factor record / dict with a license_class.

    Returns:
        True when redistribution is allowed; False otherwise.
    """
    if not oem_id:
        return False
    try:
        # Local import avoids circular dependency at module load.
        from greenlang.factors.onboarding.partner_setup import (
            OemError,
            get_redistribution_grant,
        )
    except Exception:  # pragma: no cover - defensive
        logger.warning("OEM registry unavailable; denying redistribution by default")
        return False

    try:
        grant = get_redistribution_grant(oem_id)
    except OemError as exc:
        logger.warning("Unknown OEM in redistribution check: %s", exc)
        return False

    license_class = _factor_license_class(factor)
    if not license_class:
        # No license class on the factor = treat as licensed-by-default.
        # Refuse redistribution; the operator can fix the catalog row.
        logger.debug(
            "Factor missing license_class; denying OEM %s redistribution", oem_id
        )
        return False

    allowed = grant.covers(license_class)
    if not allowed:
        logger.info(
            "OEM %s denied redistribution: license_class=%s grants=%s",
            oem_id, license_class, grant.allowed_classes,
        )
    return allowed


__all__ = [
    "Entitlement",
    "EntitlementError",
    "EntitlementRegistry",
    "OEMRights",
    "PackSKU",
    "PACK_SKU_FOR_METHOD_PROFILE",
    "pack_sku_for_profile",
    "check_oem_redistribution",
]
