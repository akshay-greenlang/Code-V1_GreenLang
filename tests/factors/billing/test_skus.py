# -*- coding: utf-8 -*-
"""
Unit tests for :mod:`greenlang.factors.billing.skus`.

Covers:
    * Every tier has sane defaults (price, meters, entitlements).
    * Every premium pack is accessible under at least one tier.
    * Overage prices are non-negative ``Decimal`` amounts.
    * ``allowed_for()`` enforces the CTO spec (community cannot access
      ``customer_private``, platform can redistribute ``restricted``, etc.).
    * Round-trip: Python catalog ↔ declarative YAML mirror.

These tests are the contract between the billing module and the rest of
the stack; a failure here means the pricing / entitlement story has
drifted and must be reconciled before ship.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from greenlang.factors.billing import skus
from greenlang.factors.billing.skus import (
    ALL_METHOD_PROFILES,
    ALL_REDISTRIBUTION_CLASSES,
    CATALOG,
    METHOD_PROFILE_CORPORATE,
    METHOD_PROFILE_ELECTRICITY,
    OPEN_CORE_METHOD_PROFILES,
    REDISTRIBUTION_CONNECTOR_ONLY,
    REDISTRIBUTION_CUSTOMER_PRIVATE,
    REDISTRIBUTION_OPEN,
    REDISTRIBUTION_RESTRICTED,
    Catalog,
    Meter,
    MeterDefinition,
    PremiumPack,
    PremiumPackConfig,
    SLALevel,
    Tier,
    TierConfig,
    allowed_for,
    meter_price_id,
    overage_price,
    pack_from_price_id,
    pack_included,
    tier_entitlements,
    tier_from_price_id,
)
from greenlang.factors.entitlements import OEMRights
from greenlang.factors.entitlements import PackSKU as LegacyPackSKU


# ---------------------------------------------------------------------------
# Tier-level invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", list(Tier))
def test_every_tier_in_catalog(tier: Tier) -> None:
    """Every enum value must have a ``TierConfig`` entry."""
    cfg = CATALOG.tier(tier)
    assert isinstance(cfg, TierConfig)
    assert cfg.tier == tier


@pytest.mark.parametrize("tier", list(Tier))
def test_tier_has_sane_defaults(tier: Tier) -> None:
    """Sanity-check every tier: non-negative prices, well-formed Stripe IDs."""
    cfg = CATALOG.tier(tier)
    assert cfg.stripe_product_id.startswith("prod_factors_"), cfg
    assert cfg.stripe_price_monthly_id.startswith("price_factors_"), cfg
    assert cfg.stripe_price_annual_id.startswith("price_factors_"), cfg
    assert isinstance(cfg.monthly_price_usd, Decimal)
    assert isinstance(cfg.annual_price_usd, Decimal)
    assert cfg.monthly_price_usd >= Decimal("0")
    assert cfg.annual_price_usd >= Decimal("0")


def test_community_is_free() -> None:
    cfg = CATALOG.tier(Tier.COMMUNITY)
    assert cfg.monthly_price_usd == Decimal("0")
    assert cfg.annual_price_usd == Decimal("0")
    assert cfg.sla_level == SLALevel.NONE
    assert cfg.audit_bundle_allowed is False
    assert cfg.bulk_export_max_rows == 0
    assert cfg.oem_enabled is False
    assert cfg.included_packs == frozenset()


def test_pro_pricing_matches_prd() -> None:
    cfg = CATALOG.tier(Tier.PRO)
    assert cfg.monthly_price_usd == Decimal("299.00")
    # API-calls meter: 100k included, $0.002/call over
    api = cfg.meter(Meter.API_CALLS)
    assert api is not None
    assert api.included_per_month == 100_000
    assert api.overage_unit_price_usd == Decimal("0.002")


def test_consulting_pricing_and_bundles() -> None:
    cfg = CATALOG.tier(Tier.CONSULTING)
    assert cfg.monthly_price_usd == Decimal("1499.00")
    # 3 bundled packs per PRD 7.3
    assert len(cfg.included_packs) == 3
    assert PremiumPack.ELECTRICITY in cfg.included_packs
    assert PremiumPack.CBAM_EU_POLICY in cfg.included_packs


def test_platform_pricing_and_oem() -> None:
    cfg = CATALOG.tier(Tier.PLATFORM)
    assert cfg.monthly_price_usd == Decimal("4999.00")
    assert cfg.oem_enabled is True
    assert cfg.oem_rights == OEMRights.REDISTRIBUTABLE
    # Platform meters 5M API calls per PRD 7.2
    api = cfg.meter(Meter.API_CALLS)
    assert api is not None
    assert api.included_per_month == 5_000_000
    # OEM meter must exist on Platform
    oem = cfg.meter(Meter.OEM_SITES)
    assert oem is not None
    assert oem.included_per_month >= 1


def test_enterprise_pricing_acv_floor() -> None:
    """ACV starts at $75k/yr per the brief."""
    cfg = CATALOG.tier(Tier.ENTERPRISE)
    assert cfg.annual_price_usd == Decimal("75000.00")
    assert cfg.monthly_price_usd == Decimal("6250.00")  # 75k / 12
    assert cfg.sla_level == SLALevel.UPTIME_99_95
    assert cfg.sso_scim_included is True
    assert cfg.annual_contract_required is True


# ---------------------------------------------------------------------------
# Premium pack invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pack", list(PremiumPack))
def test_every_pack_has_config(pack: PremiumPack) -> None:
    cfg = CATALOG.pack(pack)
    assert isinstance(cfg, PremiumPackConfig)
    assert cfg.pack == pack
    assert cfg.stripe_product_id.startswith("prod_factors_pack_")
    assert cfg.pro_addon_monthly_usd > Decimal("0")
    assert cfg.enterprise_addon_annual_usd > Decimal("0")


@pytest.mark.parametrize("pack", list(PremiumPack))
def test_every_pack_accessible_under_some_tier(pack: PremiumPack) -> None:
    """Every pack must be usable by at least one tier (never orphaned)."""
    accessible = [t for t in Tier if allowed_for(t, pack)]
    assert accessible, f"pack {pack} is not accessible under any tier"
    # Community can never access premium.
    assert Tier.COMMUNITY not in accessible


def test_pack_values_match_legacy_entitlements_sku() -> None:
    """String values must stay byte-compatible with the legacy PackSKU."""
    assert PremiumPack.ELECTRICITY.value == LegacyPackSKU.ELECTRICITY_PREMIUM
    assert PremiumPack.FREIGHT.value == LegacyPackSKU.FREIGHT_PREMIUM
    assert PremiumPack.PRODUCT_LCI.value == LegacyPackSKU.PRODUCT_CARBON_PREMIUM
    assert PremiumPack.CBAM_EU_POLICY.value == LegacyPackSKU.CBAM_PREMIUM
    assert PremiumPack.LAND_REMOVALS.value == LegacyPackSKU.LAND_PREMIUM


def test_prd_pack_prices() -> None:
    """Spot-check pack prices against PRD 7.3 addon column."""
    assert CATALOG.pack(PremiumPack.ELECTRICITY).pro_addon_monthly_usd == Decimal("99.00")
    assert CATALOG.pack(PremiumPack.FREIGHT).pro_addon_monthly_usd == Decimal("199.00")
    assert CATALOG.pack(PremiumPack.PRODUCT_LCI).pro_addon_monthly_usd == Decimal("499.00")
    assert CATALOG.pack(PremiumPack.CONSTRUCTION_EPD).pro_addon_monthly_usd == Decimal("199.00")
    assert CATALOG.pack(PremiumPack.AGRIFOOD_LAND).pro_addon_monthly_usd == Decimal("199.00")
    assert CATALOG.pack(PremiumPack.FINANCE_PROXY).pro_addon_monthly_usd == Decimal("299.00")
    assert CATALOG.pack(PremiumPack.CBAM_EU_POLICY).pro_addon_monthly_usd == Decimal("299.00")
    assert CATALOG.pack(PremiumPack.LAND_REMOVALS).pro_addon_monthly_usd == Decimal("149.00")


def test_license_chain_flags() -> None:
    """Packs that redistribute third-party IP must flag it."""
    assert CATALOG.pack(PremiumPack.PRODUCT_LCI).requires_license_chain is True
    assert CATALOG.pack(PremiumPack.FINANCE_PROXY).requires_license_chain is True
    assert CATALOG.pack(PremiumPack.ELECTRICITY).requires_license_chain is False


# ---------------------------------------------------------------------------
# Overage pricing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier,meter",
    [
        (t, m)
        for t in Tier
        for m in Meter
        if CATALOG.tier(t).meter(m) is not None
    ],
)
def test_overage_prices_are_non_negative_decimal(tier: Tier, meter: Meter) -> None:
    md = CATALOG.tier(tier).meter(meter)
    assert md is not None
    assert isinstance(md.overage_unit_price_usd, Decimal)
    assert md.overage_unit_price_usd >= Decimal("0")


def test_overage_price_below_cap_is_zero() -> None:
    # 50k API calls on Pro (100k included) should be $0.
    assert overage_price(Tier.PRO, Meter.API_CALLS, 50_000) == Decimal("0.00")


def test_overage_price_above_cap() -> None:
    # 250k on Pro = 150k overage * $0.002 = $300.
    assert overage_price(Tier.PRO, Meter.API_CALLS, 250_000) == Decimal("300.00")


def test_overage_price_hard_stop_returns_zero() -> None:
    # Community API call overage is a hard stop, not a billable event.
    assert overage_price(Tier.COMMUNITY, Meter.API_CALLS, 10_000) == Decimal("0.00")


def test_overage_price_negative_quantity_raises() -> None:
    with pytest.raises(ValueError):
        overage_price(Tier.PRO, Meter.API_CALLS, -1)


def test_overage_price_unknown_meter_raises() -> None:
    # Community has no tenants meter.
    with pytest.raises(KeyError):
        overage_price(Tier.COMMUNITY, Meter.TENANTS, 10)


def test_enterprise_api_call_overage_cheaper_than_pro() -> None:
    pro = overage_price(Tier.PRO, Meter.API_CALLS, 200_000)
    ent = overage_price(Tier.ENTERPRISE, Meter.API_CALLS, 10_200_000)  # 200k over
    # Same overage volume, Enterprise price per unit must be lower.
    pro_unit = pro / Decimal("100000")
    ent_unit = ent / Decimal("200000")
    assert ent_unit < pro_unit


# ---------------------------------------------------------------------------
# Allowed-for / redistribution / method profile spec
# ---------------------------------------------------------------------------


def test_allowed_for_community_denies_premium() -> None:
    """CTO spec: Community can never access premium packs."""
    for pack in PremiumPack:
        assert allowed_for(Tier.COMMUNITY, pack) is False


def test_allowed_for_string_inputs_accepted() -> None:
    """Helper should accept raw string tier / pack identifiers."""
    assert allowed_for("pro", "freight_premium") is True  # type: ignore[arg-type]


def test_pack_included_respects_bundle() -> None:
    assert pack_included(Tier.CONSULTING, PremiumPack.ELECTRICITY) is True
    assert pack_included(Tier.CONSULTING, PremiumPack.LAND_REMOVALS) is False
    assert pack_included(Tier.COMMUNITY, PremiumPack.ELECTRICITY) is False


def test_community_has_only_open_redistribution() -> None:
    """CTO spec: Community can only see ``redistribute_open``."""
    classes = CATALOG.tier(Tier.COMMUNITY).allowed_redistribution_classes
    assert classes == frozenset({REDISTRIBUTION_OPEN})
    assert REDISTRIBUTION_CUSTOMER_PRIVATE not in classes
    assert REDISTRIBUTION_CONNECTOR_ONLY not in classes


def test_platform_can_redistribute_restricted() -> None:
    """CTO spec: Platform tier can redistribute ``restricted`` class."""
    classes = CATALOG.tier(Tier.PLATFORM).allowed_redistribution_classes
    assert REDISTRIBUTION_RESTRICTED in classes
    # ... but not connector-only (that's Enterprise-only).
    assert REDISTRIBUTION_CONNECTOR_ONLY not in classes


def test_enterprise_sees_all_redistribution_classes() -> None:
    classes = CATALOG.tier(Tier.ENTERPRISE).allowed_redistribution_classes
    assert classes == ALL_REDISTRIBUTION_CLASSES


def test_community_only_open_core_method_profiles() -> None:
    profiles = CATALOG.tier(Tier.COMMUNITY).allowed_method_profiles
    assert profiles == OPEN_CORE_METHOD_PROFILES
    assert METHOD_PROFILE_CORPORATE in profiles
    assert METHOD_PROFILE_ELECTRICITY in profiles


def test_enterprise_sees_all_method_profiles() -> None:
    assert CATALOG.tier(Tier.ENTERPRISE).allowed_method_profiles == ALL_METHOD_PROFILES


def test_max_api_calls_per_day_monotonic() -> None:
    """Higher tiers must have higher (or uncapped) daily throttles."""
    c = CATALOG.tier(Tier.COMMUNITY).max_api_calls_per_day
    p = CATALOG.tier(Tier.PRO).max_api_calls_per_day
    con = CATALOG.tier(Tier.CONSULTING).max_api_calls_per_day
    plat = CATALOG.tier(Tier.PLATFORM).max_api_calls_per_day
    ent = CATALOG.tier(Tier.ENTERPRISE).max_api_calls_per_day
    assert c and p and con and plat
    assert c < p < con < plat
    # Enterprise is uncapped (``None``) which ranks above everything.
    assert ent is None


# ---------------------------------------------------------------------------
# Entitlement snapshot
# ---------------------------------------------------------------------------


def test_tier_entitlements_for_platform() -> None:
    ent = tier_entitlements(Tier.PLATFORM)
    assert ent["tier"] == "platform"
    assert ent["oem_enabled"] is True
    assert ent["oem_rights"] == OEMRights.REDISTRIBUTABLE
    assert ent["audit_bundle_allowed"] is True
    assert "electricity_premium" in ent["included_packs"]


def test_tier_entitlements_json_safe() -> None:
    """Snapshot must be JSON-serialisable (no sets, no Decimals, no enums)."""
    import json

    payload = {t.value: tier_entitlements(t) for t in Tier}
    dumped = json.dumps(payload, sort_keys=True)
    assert isinstance(dumped, str)
    assert "\"tier\"" in dumped


# ---------------------------------------------------------------------------
# Price-id reverse lookup
# ---------------------------------------------------------------------------


def test_tier_from_price_id_round_trip() -> None:
    for tier in Tier:
        cfg = CATALOG.tier(tier)
        assert tier_from_price_id(cfg.stripe_price_monthly_id) == tier
        assert tier_from_price_id(cfg.stripe_price_annual_id) == tier


def test_pack_from_price_id_round_trip() -> None:
    for pack in PremiumPack:
        cfg = CATALOG.pack(pack)
        assert pack_from_price_id(cfg.stripe_price_monthly_id) == pack
        assert pack_from_price_id(cfg.stripe_price_annual_id) == pack


def test_meter_price_id_is_unique() -> None:
    seen = set()
    for tier in Tier:
        for m in Meter:
            if CATALOG.tier(tier).meter(m) is None:
                continue
            pid = meter_price_id(tier, m)
            assert pid not in seen, f"duplicate meter price id: {pid}"
            seen.add(pid)


def test_tier_from_price_id_meter_lookup() -> None:
    # Meter prices also resolve back to their tier.
    assert tier_from_price_id("price_factors_meter_api_calls_pro") == Tier.PRO
    assert tier_from_price_id("price_factors_meter_oem_sites_platform") == Tier.PLATFORM


def test_unknown_price_id_returns_none() -> None:
    assert tier_from_price_id("") is None
    assert tier_from_price_id("price_totally_made_up") is None
    assert pack_from_price_id("") is None


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def yaml_catalog() -> dict:
    """Load the declarative YAML mirror of the Python catalog."""
    path = (
        Path(__file__).resolve().parents[3]
        / "deployment"
        / "stripe"
        / "factors-products.yaml"
    )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_yaml_has_every_tier(yaml_catalog: dict) -> None:
    yaml_tiers = {t["tier"] for t in yaml_catalog["tiers"]}
    assert yaml_tiers == {t.value for t in Tier}


def test_yaml_has_every_pack(yaml_catalog: dict) -> None:
    yaml_packs = {p["pack"] for p in yaml_catalog["packs"]}
    assert yaml_packs == {p.value for p in PremiumPack}


def test_yaml_prices_match_python(yaml_catalog: dict) -> None:
    """Every tier's monthly + annual price matches ``skus.py``."""
    for tier_spec in yaml_catalog["tiers"]:
        tier = Tier(tier_spec["tier"])
        py_cfg = CATALOG.tier(tier)
        yaml_prices = {p["id"]: Decimal(p["amount_usd"]) for p in tier_spec["prices"]}
        assert yaml_prices[py_cfg.stripe_price_monthly_id] == py_cfg.monthly_price_usd
        assert yaml_prices[py_cfg.stripe_price_annual_id] == py_cfg.annual_price_usd


def test_yaml_pack_prices_match_python(yaml_catalog: dict) -> None:
    for pack_spec in yaml_catalog["packs"]:
        pack = PremiumPack(pack_spec["pack"])
        py_cfg = CATALOG.pack(pack)
        yaml_prices = {p["id"]: Decimal(p["amount_usd"]) for p in pack_spec["prices"]}
        assert (
            yaml_prices[py_cfg.stripe_price_monthly_id]
            == py_cfg.pro_addon_monthly_usd
        )
        assert (
            yaml_prices[py_cfg.stripe_price_annual_id]
            == py_cfg.enterprise_addon_annual_usd
        )


def test_yaml_meter_rates_match_python(yaml_catalog: dict) -> None:
    """Meter overage rates in YAML line up with the Python ``MeterDefinition``."""
    for tier_spec in yaml_catalog["tiers"]:
        tier = Tier(tier_spec["tier"])
        py_cfg = CATALOG.tier(tier)
        for meter_spec in tier_spec.get("meters", []):
            meter = Meter(meter_spec["meter"])
            md = py_cfg.meter(meter)
            assert md is not None
            assert meter_spec["included_per_month"] == md.included_per_month
            assert Decimal(meter_spec["overage_unit_price_usd"]) == md.overage_unit_price_usd


# ---------------------------------------------------------------------------
# Catalog container behaviour
# ---------------------------------------------------------------------------


def test_catalog_object_lists_all_products() -> None:
    ids = CATALOG.all_stripe_product_ids()
    assert all(pid.startswith("prod_factors_") for pid in ids)
    # 5 tiers + 8 packs = 13 products.
    assert len(ids) == 5 + 8


def test_catalog_object_lists_all_prices() -> None:
    ids = CATALOG.all_stripe_price_ids()
    assert all(pid.startswith("price_factors_") for pid in ids)
    # No duplicates.
    assert len(set(ids)) == len(ids)


def test_catalog_is_immutable() -> None:
    """The top-level ``Catalog`` should be a frozen dataclass."""
    # Attempting to rebind attributes raises ``FrozenInstanceError``.
    with pytest.raises(Exception):
        CATALOG.tiers = {}  # type: ignore[misc]
