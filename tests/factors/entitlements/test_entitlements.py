# -*- coding: utf-8 -*-
"""Phase F8 — Premium Pack entitlements + Consulting tier tests."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from greenlang.factors.entitlements import (
    Entitlement,
    EntitlementError,
    EntitlementRegistry,
    OEMRights,
    PACK_SKU_FOR_METHOD_PROFILE,
    PackSKU,
    pack_sku_for_profile,
)
from greenlang.factors.tier_enforcement import Tier, TierVisibility


# --------------------------------------------------------------------------
# Consulting tier
# --------------------------------------------------------------------------


class TestConsultingTier:
    def test_consulting_is_registered_enum(self):
        assert Tier.CONSULTING.value == "consulting"

    def test_consulting_visibility(self):
        v = TierVisibility.from_tier("consulting")
        assert v.include_preview is True
        assert v.include_connector is False        # no connector-only
        assert v.audit_bundle_allowed is False     # audit = enterprise only
        assert v.bulk_export_allowed is True
        assert v.max_export_rows == 50_000

    def test_consulting_sits_between_pro_and_enterprise(self):
        pro = TierVisibility.from_tier("pro")
        cons = TierVisibility.from_tier("consulting")
        ent = TierVisibility.from_tier("enterprise")
        assert pro.max_export_rows < cons.max_export_rows < ent.max_export_rows

    def test_unknown_tier_defaults_to_community(self):
        v = TierVisibility.from_tier("nonsense_tier")
        assert v.max_export_rows == 1_000


# --------------------------------------------------------------------------
# Pack SKU registry
# --------------------------------------------------------------------------


class TestPackSKU:
    def test_all_skus_listed(self):
        assert len(PackSKU.ALL) == 8
        assert "freight_premium" in PackSKU.ALL
        assert "finance_premium" in PackSKU.ALL

    def test_method_profile_sku_mapping(self):
        assert pack_sku_for_profile("freight_iso_14083") == PackSKU.FREIGHT_PREMIUM
        assert pack_sku_for_profile("product_carbon") == PackSKU.PRODUCT_CARBON_PREMIUM
        assert pack_sku_for_profile("eu_cbam") == PackSKU.CBAM_PREMIUM

    def test_open_core_profiles_return_none(self):
        # Scope 1/2 are open-core, no SKU required.
        assert pack_sku_for_profile("corporate_scope1") is None
        assert pack_sku_for_profile("corporate_scope2_location_based") is None

    def test_mapping_covers_all_premium_profiles(self):
        # Every SKU has at least one method_profile mapping in production.
        mapped_skus = set(PACK_SKU_FOR_METHOD_PROFILE.values())
        # Electricity/EPD/Agrifood/Land are future work — not required yet.
        required = {
            PackSKU.FREIGHT_PREMIUM, PackSKU.PRODUCT_CARBON_PREMIUM,
            PackSKU.CBAM_PREMIUM, PackSKU.FINANCE_PREMIUM, PackSKU.LAND_PREMIUM,
        }
        assert required.issubset(mapped_skus)


# --------------------------------------------------------------------------
# EntitlementRegistry
# --------------------------------------------------------------------------


class TestEntitlementRegistry:
    def test_grant_and_check(self, tmp_path: Path):
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            reg.grant(tenant_id="acme", pack_sku=PackSKU.FREIGHT_PREMIUM)
            assert reg.is_entitled(tenant_id="acme", pack_sku=PackSKU.FREIGHT_PREMIUM)
            assert not reg.is_entitled(tenant_id="acme", pack_sku=PackSKU.LAND_PREMIUM)
        finally:
            reg.close()

    def test_unknown_sku_rejected(self, tmp_path: Path):
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            with pytest.raises(EntitlementError):
                reg.grant(tenant_id="acme", pack_sku="bogus_pack")
        finally:
            reg.close()

    def test_invalid_oem_rights_rejected(self, tmp_path: Path):
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            with pytest.raises(EntitlementError):
                reg.grant(
                    tenant_id="acme",
                    pack_sku=PackSKU.FREIGHT_PREMIUM,
                    oem_rights="sneaky",
                )
        finally:
            reg.close()

    def test_expired_entitlement_not_live(self, tmp_path: Path):
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            yesterday = (date.today() - timedelta(days=1)).isoformat()
            reg.grant(
                tenant_id="acme",
                pack_sku=PackSKU.CBAM_PREMIUM,
                expires_at=yesterday,
            )
            assert not reg.is_entitled(
                tenant_id="acme", pack_sku=PackSKU.CBAM_PREMIUM
            )
        finally:
            reg.close()

    def test_revoke_deactivates(self, tmp_path: Path):
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            reg.grant(tenant_id="acme", pack_sku=PackSKU.FREIGHT_PREMIUM)
            assert reg.revoke(tenant_id="acme", pack_sku=PackSKU.FREIGHT_PREMIUM)
            assert not reg.is_entitled(
                tenant_id="acme", pack_sku=PackSKU.FREIGHT_PREMIUM
            )
        finally:
            reg.close()

    def test_grant_upserts(self, tmp_path: Path):
        """Granting twice should update, not duplicate."""
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            reg.grant(tenant_id="acme", pack_sku=PackSKU.LAND_PREMIUM, seat_cap=5)
            reg.grant(tenant_id="acme", pack_sku=PackSKU.LAND_PREMIUM, seat_cap=10)
            ent = reg.get(tenant_id="acme", pack_sku=PackSKU.LAND_PREMIUM)
            assert ent is not None
            assert ent.seat_cap == 10
        finally:
            reg.close()

    def test_list_for_tenant(self, tmp_path: Path):
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            reg.grant(tenant_id="acme", pack_sku=PackSKU.FREIGHT_PREMIUM)
            reg.grant(tenant_id="acme", pack_sku=PackSKU.LAND_PREMIUM)
            ents = reg.list_for_tenant("acme")
            assert len(ents) == 2
            assert {e.pack_sku for e in ents} == {
                PackSKU.FREIGHT_PREMIUM, PackSKU.LAND_PREMIUM
            }
        finally:
            reg.close()

    def test_oem_rights_tri_state(self, tmp_path: Path):
        reg = EntitlementRegistry(tmp_path / "ent.sqlite")
        try:
            for rights in (
                OEMRights.FORBIDDEN,
                OEMRights.INTERNAL_ONLY,
                OEMRights.REDISTRIBUTABLE,
            ):
                reg.grant(
                    tenant_id=f"tenant_{rights}",
                    pack_sku=PackSKU.CBAM_PREMIUM,
                    oem_rights=rights,
                )
                ent = reg.get(
                    tenant_id=f"tenant_{rights}", pack_sku=PackSKU.CBAM_PREMIUM
                )
                assert ent.oem_rights == rights
        finally:
            reg.close()


# --------------------------------------------------------------------------
# Migration V442 check
# --------------------------------------------------------------------------


class TestMigrationFile:
    def test_v442_exists(self):
        p = Path("deployment/database/migrations/sql/V442__factor_pack_entitlements.sql")
        assert p.exists()
        sql = p.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS factor_pack_entitlements" in sql
        assert "chk_pack_sku" in sql
        assert "chk_oem_rights" in sql
