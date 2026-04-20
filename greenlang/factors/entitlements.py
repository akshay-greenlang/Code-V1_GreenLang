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


__all__ = [
    "Entitlement",
    "EntitlementError",
    "EntitlementRegistry",
    "OEMRights",
    "PackSKU",
    "PACK_SKU_FOR_METHOD_PROFILE",
    "pack_sku_for_profile",
]
