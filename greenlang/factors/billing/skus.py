# -*- coding: utf-8 -*-
"""
Stripe SKU catalog for the GreenLang Factors FY27 launch.

This module is the **single source of truth** for pricing, meters, premium
packs, and per-tier entitlements used by the Stripe provisioning script,
the webhook handler, the tier enforcement layer, and the entitlement
registry. Every number is a ``Decimal`` (never ``float``) so monthly ACV
math does not drift due to binary floating-point rounding.

References:
    * docs/product/PRD-FY27-Factors.md sect. 7 (pricing architecture)
    * greenlang/factors/tier_enforcement.py (tier visibility)
    * greenlang/factors/entitlements.py (PackSKU + OEMRights)
    * greenlang/factors/billing/stripe_provider.py (legacy TierConfig)

Conventions:
    * Stripe product IDs follow the prefix ``prod_factors_<tier>`` or
      ``prod_factors_pack_<pack>``.
    * Stripe price IDs follow ``price_factors_<tier>_<cadence>`` or
      ``price_factors_pack_<pack>_<cadence>`` where ``cadence`` is one of
      ``monthly``, ``annual``, or ``overage``.
    * Meter price IDs follow ``price_factors_meter_<meter>_<tier>``.

Typical use:

    >>> from greenlang.factors.billing.skus import (
    ...     Tier, PremiumPack, CATALOG, allowed_for, overage_price,
    ... )
    >>> CATALOG.tier(Tier.PRO).monthly_price_usd
    Decimal('299.00')
    >>> allowed_for(Tier.COMMUNITY, PremiumPack.ELECTRICITY)
    False
    >>> overage_price(Tier.PRO, MeterDefinition.API_CALLS, 250_000)
    Decimal('300.00')
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from greenlang.factors.entitlements import OEMRights
from greenlang.factors.entitlements import PackSKU as _LegacyPackSKU

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core enums
# ---------------------------------------------------------------------------


class Tier(str, Enum):
    """Top-level subscription tier.

    Values intentionally line up with :class:`greenlang.factors.tier_enforcement.Tier`
    so the visibility enforcer and the billing SKU catalog speak the same
    string. ``PLATFORM`` is introduced in FY27 and corresponds to the upper
    slice of the Consulting/Platform band in PRD sect. 7.2.
    """

    COMMUNITY = "community"
    PRO = "pro"
    CONSULTING = "consulting"
    PLATFORM = "platform"
    ENTERPRISE = "enterprise"


class PremiumPack(str, Enum):
    """Premium data-pack SKUs sold on top of a tier.

    String values deliberately match the legacy ``PackSKU`` identifiers in
    :mod:`greenlang.factors.entitlements` so downstream entitlement rows
    stay byte-compatible.
    """

    ELECTRICITY = _LegacyPackSKU.ELECTRICITY_PREMIUM
    FREIGHT = _LegacyPackSKU.FREIGHT_PREMIUM
    PRODUCT_LCI = _LegacyPackSKU.PRODUCT_CARBON_PREMIUM
    CONSTRUCTION_EPD = _LegacyPackSKU.EPD_PREMIUM
    AGRIFOOD_LAND = _LegacyPackSKU.AGRIFOOD_PREMIUM
    FINANCE_PROXY = _LegacyPackSKU.FINANCE_PREMIUM
    CBAM_EU_POLICY = _LegacyPackSKU.CBAM_PREMIUM
    # ``land_premium`` is a separate SKU per PRD sect. 7.3 row 8.
    LAND_REMOVALS = _LegacyPackSKU.LAND_PREMIUM


class Meter(str, Enum):
    """Usage meter types reported to Stripe.

    * ``API_CALLS`` — single-record HTTP hits to ``/search``, ``/match``, etc.
    * ``BATCH_ROWS`` — rows inside a ``/export`` or ``/match_bulk`` call.
    * ``PRIVATE_REGISTRY_MB`` — overlay / private-registry storage.
    * ``TENANTS`` — multi-tenant seats (Consulting / Platform).
    * ``OEM_SITES`` — white-label embed sites (Platform uplift + Enterprise addon).
    """

    API_CALLS = "api_calls"
    BATCH_ROWS = "batch_rows"
    PRIVATE_REGISTRY_MB = "private_registry_mb"
    TENANTS = "tenants"
    OEM_SITES = "oem_sites"


class SLALevel(str, Enum):
    """SLA banding per PRD sect. 7.2."""

    NONE = "none"
    UPTIME_99_5 = "99.5"
    UPTIME_99_9 = "99.9"
    UPTIME_99_95 = "99.95"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def _d(value: str) -> Decimal:
    """Build a ``Decimal`` from a string literal.

    Using string input sidesteps binary ``float`` rounding entirely — the
    cardinal rule from the brief.
    """
    return Decimal(value)


@dataclass(frozen=True)
class MeterDefinition:
    """Stripe metered-price definition for a single usage dimension.

    Attributes:
        meter: The usage meter type (one of :class:`Meter`).
        display_name: Human-friendly label used in invoices + product pages.
        unit_label: Stripe ``unit_label`` shown on line items ("API call", etc.).
        included_per_month: Quantity included in the tier base fee.
        overage_unit_price_usd: Per-unit price charged **above** the included
            quota. ``Decimal('0')`` means "not billable past the cap —
            enforce a hard stop instead."
    """

    meter: Meter
    display_name: str
    unit_label: str
    included_per_month: int
    overage_unit_price_usd: Decimal


@dataclass(frozen=True)
class TierConfig:
    """Full economic + entitlement definition for a subscription tier.

    Holds everything the Stripe provisioner needs to create the product,
    the recurring price (monthly + annual), the metered overage prices,
    and everything the enforcement + entitlement layers need to decide
    whether a request is permitted.
    """

    tier: Tier
    display_name: str
    stripe_product_id: str
    stripe_price_monthly_id: str
    stripe_price_annual_id: str
    monthly_price_usd: Decimal
    annual_price_usd: Decimal
    meters: Tuple[MeterDefinition, ...]
    included_packs: FrozenSet[PremiumPack]
    allowed_redistribution_classes: FrozenSet[str]
    allowed_method_profiles: FrozenSet[str]
    max_api_calls_per_day: Optional[int]
    private_registry_enabled: bool
    private_registry_entries: Optional[int]
    oem_enabled: bool
    oem_rights: str  # OEMRights.*
    sla_level: SLALevel
    audit_bundle_allowed: bool
    bulk_export_max_rows: int
    sso_scim_included: bool
    annual_contract_required: bool

    def meter(self, meter: Meter) -> Optional[MeterDefinition]:
        """Look up a meter definition for this tier, or ``None`` if absent."""
        for m in self.meters:
            if m.meter == meter:
                return m
        return None


@dataclass(frozen=True)
class PremiumPackConfig:
    """Stripe product + price for a Premium Data Pack (PRD sect. 7.3)."""

    pack: PremiumPack
    display_name: str
    stripe_product_id: str
    stripe_price_monthly_id: str
    stripe_price_annual_id: str
    pro_addon_monthly_usd: Decimal
    consulting_addon_monthly_usd: Decimal  # "Include option" list price
    enterprise_addon_annual_usd: Decimal
    default_oem_rights: str  # OEMRights.*
    requires_license_chain: bool
    license_chain_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Redistribution classes + method profiles (matches CTO spec)
# ---------------------------------------------------------------------------
#
# These strings are the same as the ``factor_status`` / ``redistribution_class``
# values used by ``tier_enforcement.factor_visible_for_tier`` and the
# Canonical Factor Record's ``license.redistribution`` field. Keeping them
# co-located here makes the enforcement rules auditable in ONE place.

REDISTRIBUTION_OPEN = "redistribute_open"
REDISTRIBUTION_RESTRICTED = "redistribute_restricted"
REDISTRIBUTION_CONNECTOR_ONLY = "connector_only"
REDISTRIBUTION_CUSTOMER_PRIVATE = "customer_private"
REDISTRIBUTION_INTERNAL_ONLY = "internal_only"

ALL_REDISTRIBUTION_CLASSES: FrozenSet[str] = frozenset(
    {
        REDISTRIBUTION_OPEN,
        REDISTRIBUTION_RESTRICTED,
        REDISTRIBUTION_CONNECTOR_ONLY,
        REDISTRIBUTION_CUSTOMER_PRIVATE,
        REDISTRIBUTION_INTERNAL_ONLY,
    }
)

# Method profiles registered by ``greenlang/factors/method_packs/``.
METHOD_PROFILE_CORPORATE = "corporate"
METHOD_PROFILE_ELECTRICITY = "electricity"
METHOD_PROFILE_FREIGHT = "freight_iso_14083"
METHOD_PROFILE_EU_POLICY = "eu_cbam"
METHOD_PROFILE_LAND = "land_removals"
METHOD_PROFILE_PRODUCT_CARBON = "product_carbon"
METHOD_PROFILE_FINANCE = "finance_proxy"

OPEN_CORE_METHOD_PROFILES: FrozenSet[str] = frozenset(
    {METHOD_PROFILE_CORPORATE, METHOD_PROFILE_ELECTRICITY}
)
ALL_METHOD_PROFILES: FrozenSet[str] = frozenset(
    {
        METHOD_PROFILE_CORPORATE,
        METHOD_PROFILE_ELECTRICITY,
        METHOD_PROFILE_FREIGHT,
        METHOD_PROFILE_EU_POLICY,
        METHOD_PROFILE_LAND,
        METHOD_PROFILE_PRODUCT_CARBON,
        METHOD_PROFILE_FINANCE,
    }
)


# ---------------------------------------------------------------------------
# Meter rate cards
# ---------------------------------------------------------------------------
#
# Numbers below come from PRD sect. 7.2 (quota columns) + sect. 7.5 + the
# legacy stripe_provider.TIER_CONFIGS. Where the PRD is silent we pick a
# defensible value and flag it with TODO so product can ratify later.

_COMMUNITY_METERS: Tuple[MeterDefinition, ...] = (
    MeterDefinition(
        meter=Meter.API_CALLS,
        display_name="API calls",
        unit_label="API call",
        included_per_month=1_000,
        overage_unit_price_usd=_d("0"),  # hard stop -- no overage on free tier
    ),
    MeterDefinition(
        meter=Meter.BATCH_ROWS,
        display_name="Batch rows",
        unit_label="row",
        included_per_month=1_000,
        overage_unit_price_usd=_d("0"),
    ),
)

_PRO_METERS: Tuple[MeterDefinition, ...] = (
    MeterDefinition(
        meter=Meter.API_CALLS,
        display_name="API calls",
        unit_label="API call",
        included_per_month=100_000,  # Pro mid sub-tier; see PRD 7.2
        overage_unit_price_usd=_d("0.002"),  # $0.002 per call over 100k
    ),
    MeterDefinition(
        meter=Meter.BATCH_ROWS,
        display_name="Batch rows",
        unit_label="row",
        included_per_month=100_000,
        overage_unit_price_usd=_d("0.0005"),
    ),
    MeterDefinition(
        meter=Meter.PRIVATE_REGISTRY_MB,
        display_name="Private registry storage",
        unit_label="MB",
        # TODO(product): PRD says "50 entries per project"; storage MB cap
        # is derived. 50 entries x ~20 MB = 1 GB included.
        included_per_month=1_024,
        overage_unit_price_usd=_d("0.10"),
    ),
)

_CONSULTING_METERS: Tuple[MeterDefinition, ...] = (
    MeterDefinition(
        meter=Meter.API_CALLS,
        display_name="API calls",
        unit_label="API call",
        included_per_month=1_000_000,  # Consulting low end per PRD 7.2
        overage_unit_price_usd=_d("0.001"),
    ),
    MeterDefinition(
        meter=Meter.BATCH_ROWS,
        display_name="Batch rows",
        unit_label="row",
        included_per_month=2_000_000,
        overage_unit_price_usd=_d("0.0003"),
    ),
    MeterDefinition(
        meter=Meter.PRIVATE_REGISTRY_MB,
        display_name="Private registry storage",
        unit_label="MB",
        included_per_month=10_240,
        overage_unit_price_usd=_d("0.08"),
    ),
    MeterDefinition(
        meter=Meter.TENANTS,
        display_name="Multi-tenant sub-tenants",
        unit_label="tenant",
        # TODO(product): PRD says "multi-client sub-tenants"; cap 25 by default.
        included_per_month=25,
        overage_unit_price_usd=_d("50.00"),
    ),
)

_PLATFORM_METERS: Tuple[MeterDefinition, ...] = (
    MeterDefinition(
        meter=Meter.API_CALLS,
        display_name="API calls",
        unit_label="API call",
        included_per_month=5_000_000,  # Platform upper band per PRD 7.2
        overage_unit_price_usd=_d("0.0005"),
    ),
    MeterDefinition(
        meter=Meter.BATCH_ROWS,
        display_name="Batch rows",
        unit_label="row",
        included_per_month=5_000_000,
        overage_unit_price_usd=_d("0.0002"),
    ),
    MeterDefinition(
        meter=Meter.PRIVATE_REGISTRY_MB,
        display_name="Private registry storage",
        unit_label="MB",
        included_per_month=51_200,
        overage_unit_price_usd=_d("0.05"),
    ),
    MeterDefinition(
        meter=Meter.TENANTS,
        display_name="Multi-tenant sub-tenants",
        unit_label="tenant",
        included_per_month=100,
        overage_unit_price_usd=_d("35.00"),
    ),
    MeterDefinition(
        meter=Meter.OEM_SITES,
        display_name="OEM white-label sites",
        unit_label="site",
        # TODO(product): PRD 7.2 says "OEM white-label ... included in Platform
        # SKU"; we include 3 sites and meter beyond that.
        included_per_month=3,
        overage_unit_price_usd=_d("500.00"),
    ),
)

_ENTERPRISE_METERS: Tuple[MeterDefinition, ...] = (
    MeterDefinition(
        meter=Meter.API_CALLS,
        display_name="API calls",
        unit_label="API call",
        included_per_month=10_000_000,  # Enterprise floor per PRD 7.2
        overage_unit_price_usd=_d("0.0002"),
    ),
    MeterDefinition(
        meter=Meter.BATCH_ROWS,
        display_name="Batch rows",
        unit_label="row",
        included_per_month=10_000_000,
        overage_unit_price_usd=_d("0.0001"),
    ),
    MeterDefinition(
        meter=Meter.PRIVATE_REGISTRY_MB,
        display_name="Private registry storage",
        unit_label="MB",
        # Unlimited in contract; we still meter for visibility. Overage is
        # effectively bill-back at cost.
        included_per_month=1_048_576,  # 1 TB included
        overage_unit_price_usd=_d("0.02"),
    ),
    MeterDefinition(
        meter=Meter.TENANTS,
        display_name="Multi-tenant sub-tenants",
        unit_label="tenant",
        included_per_month=1_000,
        overage_unit_price_usd=_d("25.00"),
    ),
    MeterDefinition(
        meter=Meter.OEM_SITES,
        display_name="OEM white-label sites",
        unit_label="site",
        # OEM is a $50k+ addon per PRD 7.2; zero included by default.
        included_per_month=0,
        overage_unit_price_usd=_d("0"),  # negotiated via addon, not metered
    ),
)


# ---------------------------------------------------------------------------
# Tier catalog
# ---------------------------------------------------------------------------

_TIERS: Dict[Tier, TierConfig] = {
    Tier.COMMUNITY: TierConfig(
        tier=Tier.COMMUNITY,
        display_name="Community",
        stripe_product_id="prod_factors_community",
        stripe_price_monthly_id="price_factors_community_monthly",
        stripe_price_annual_id="price_factors_community_annual",
        monthly_price_usd=_d("0.00"),
        annual_price_usd=_d("0.00"),
        meters=_COMMUNITY_METERS,
        included_packs=frozenset(),
        allowed_redistribution_classes=frozenset({REDISTRIBUTION_OPEN}),
        allowed_method_profiles=OPEN_CORE_METHOD_PROFILES,
        max_api_calls_per_day=100,  # 1k/month throttled evenly
        private_registry_enabled=False,
        private_registry_entries=0,
        oem_enabled=False,
        oem_rights=OEMRights.FORBIDDEN,
        sla_level=SLALevel.NONE,
        audit_bundle_allowed=False,
        bulk_export_max_rows=0,
        sso_scim_included=False,
        annual_contract_required=False,
    ),
    Tier.PRO: TierConfig(
        tier=Tier.PRO,
        display_name="Developer Pro",
        stripe_product_id="prod_factors_pro",
        stripe_price_monthly_id="price_factors_pro_monthly",
        stripe_price_annual_id="price_factors_pro_annual",
        monthly_price_usd=_d("299.00"),
        # ~17% annual discount on the $299/mo headline rate per standard
        # SaaS discounting; 299 * 12 * 0.83 ~= 2977.
        annual_price_usd=_d("2988.00"),
        meters=_PRO_METERS,
        included_packs=frozenset(),  # zero packs bundled; buy as add-on
        allowed_redistribution_classes=frozenset(
            {REDISTRIBUTION_OPEN, REDISTRIBUTION_RESTRICTED}
        ),
        allowed_method_profiles=frozenset(
            OPEN_CORE_METHOD_PROFILES
            | {METHOD_PROFILE_FREIGHT, METHOD_PROFILE_LAND}
        ),
        max_api_calls_per_day=10_000,
        private_registry_enabled=True,
        private_registry_entries=50,
        oem_enabled=False,
        oem_rights=OEMRights.FORBIDDEN,
        sla_level=SLALevel.UPTIME_99_5,
        audit_bundle_allowed=False,
        bulk_export_max_rows=5_000,
        sso_scim_included=False,
        annual_contract_required=False,
    ),
    Tier.CONSULTING: TierConfig(
        tier=Tier.CONSULTING,
        display_name="Consulting",
        stripe_product_id="prod_factors_consulting",
        stripe_price_monthly_id="price_factors_consulting_monthly",
        stripe_price_annual_id="price_factors_consulting_annual",
        monthly_price_usd=_d("1499.00"),
        # $25k-$75k/yr band per PRD 7.2 -- we publish the entry rung.
        annual_price_usd=_d("25000.00"),
        meters=_CONSULTING_METERS,
        # "3 packs included" per PRD 7.3; we bundle electricity + CBAM
        # (matching the "Included (1 of 3)" rows) plus freight as the
        # common consulting ask. Customer can swap via custom contract.
        included_packs=frozenset(
            {
                PremiumPack.ELECTRICITY,
                PremiumPack.CBAM_EU_POLICY,
                PremiumPack.FREIGHT,
            }
        ),
        allowed_redistribution_classes=frozenset(
            {
                REDISTRIBUTION_OPEN,
                REDISTRIBUTION_RESTRICTED,
                REDISTRIBUTION_CUSTOMER_PRIVATE,
            }
        ),
        allowed_method_profiles=frozenset(
            OPEN_CORE_METHOD_PROFILES
            | {
                METHOD_PROFILE_FREIGHT,
                METHOD_PROFILE_EU_POLICY,
                METHOD_PROFILE_LAND,
            }
        ),
        max_api_calls_per_day=50_000,
        private_registry_enabled=True,
        private_registry_entries=None,  # multi-tenant; unlimited per sub-tenant
        oem_enabled=False,
        oem_rights=OEMRights.INTERNAL_ONLY,
        sla_level=SLALevel.UPTIME_99_9,
        audit_bundle_allowed=True,
        bulk_export_max_rows=50_000,
        sso_scim_included=False,
        annual_contract_required=True,
    ),
    Tier.PLATFORM: TierConfig(
        tier=Tier.PLATFORM,
        display_name="Platform",
        stripe_product_id="prod_factors_platform",
        stripe_price_monthly_id="price_factors_platform_monthly",
        stripe_price_annual_id="price_factors_platform_annual",
        monthly_price_usd=_d("4999.00"),
        annual_price_usd=_d("50000.00"),  # mid band + OEM uplift baked in
        meters=_PLATFORM_METERS,
        included_packs=frozenset(
            {
                PremiumPack.ELECTRICITY,
                PremiumPack.CBAM_EU_POLICY,
                PremiumPack.FREIGHT,
            }
        ),
        allowed_redistribution_classes=frozenset(
            {
                REDISTRIBUTION_OPEN,
                REDISTRIBUTION_RESTRICTED,
                REDISTRIBUTION_CUSTOMER_PRIVATE,
            }
        ),
        allowed_method_profiles=frozenset(
            OPEN_CORE_METHOD_PROFILES
            | {
                METHOD_PROFILE_FREIGHT,
                METHOD_PROFILE_EU_POLICY,
                METHOD_PROFILE_LAND,
                METHOD_PROFILE_PRODUCT_CARBON,
            }
        ),
        max_api_calls_per_day=250_000,
        private_registry_enabled=True,
        private_registry_entries=None,
        oem_enabled=True,
        oem_rights=OEMRights.REDISTRIBUTABLE,
        sla_level=SLALevel.UPTIME_99_9,
        audit_bundle_allowed=True,
        bulk_export_max_rows=100_000,
        sso_scim_included=False,
        annual_contract_required=True,
    ),
    Tier.ENTERPRISE: TierConfig(
        tier=Tier.ENTERPRISE,
        display_name="Enterprise",
        stripe_product_id="prod_factors_enterprise",
        stripe_price_monthly_id="price_factors_enterprise_monthly",
        stripe_price_annual_id="price_factors_enterprise_annual",
        # $100k-$300k ACV band per PRD 7.2; we publish the contractual
        # floor (ACV from $75k/yr per the brief).
        monthly_price_usd=_d("6250.00"),  # 75k / 12
        annual_price_usd=_d("75000.00"),
        meters=_ENTERPRISE_METERS,
        # Every pack is available; none are "bundled" by default so ACV
        # conversations can itemize them. Add-on pricing lives on
        # ``PremiumPackConfig.enterprise_addon_annual_usd``.
        included_packs=frozenset(),
        allowed_redistribution_classes=ALL_REDISTRIBUTION_CLASSES,
        allowed_method_profiles=ALL_METHOD_PROFILES,
        max_api_calls_per_day=None,  # no per-day throttle; monthly cap only
        private_registry_enabled=True,
        private_registry_entries=None,  # unlimited per contract
        oem_enabled=True,
        oem_rights=OEMRights.REDISTRIBUTABLE,
        sla_level=SLALevel.UPTIME_99_95,
        audit_bundle_allowed=True,
        bulk_export_max_rows=1_000_000,
        sso_scim_included=True,
        annual_contract_required=True,
    ),
}


# ---------------------------------------------------------------------------
# Premium pack catalog
# ---------------------------------------------------------------------------


def _pack_prod(pack: PremiumPack) -> str:
    return f"prod_factors_pack_{pack.value}"


def _pack_price(pack: PremiumPack, cadence: str) -> str:
    return f"price_factors_pack_{pack.value}_{cadence}"


_PACKS: Dict[PremiumPack, PremiumPackConfig] = {
    PremiumPack.ELECTRICITY: PremiumPackConfig(
        pack=PremiumPack.ELECTRICITY,
        display_name="Electricity Premium",
        stripe_product_id=_pack_prod(PremiumPack.ELECTRICITY),
        stripe_price_monthly_id=_pack_price(PremiumPack.ELECTRICITY, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.ELECTRICITY, "annual"),
        pro_addon_monthly_usd=_d("99.00"),
        consulting_addon_monthly_usd=_d("99.00"),
        enterprise_addon_annual_usd=_d("12000.00"),
        default_oem_rights=OEMRights.INTERNAL_ONLY,
        requires_license_chain=False,
    ),
    PremiumPack.FREIGHT: PremiumPackConfig(
        pack=PremiumPack.FREIGHT,
        display_name="Freight Premium (ISO 14083)",
        stripe_product_id=_pack_prod(PremiumPack.FREIGHT),
        stripe_price_monthly_id=_pack_price(PremiumPack.FREIGHT, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.FREIGHT, "annual"),
        pro_addon_monthly_usd=_d("199.00"),
        consulting_addon_monthly_usd=_d("199.00"),
        enterprise_addon_annual_usd=_d("18000.00"),
        default_oem_rights=OEMRights.INTERNAL_ONLY,
        requires_license_chain=False,
    ),
    PremiumPack.PRODUCT_LCI: PremiumPackConfig(
        pack=PremiumPack.PRODUCT_LCI,
        display_name="Product Carbon / LCI Premium",
        stripe_product_id=_pack_prod(PremiumPack.PRODUCT_LCI),
        stripe_price_monthly_id=_pack_price(PremiumPack.PRODUCT_LCI, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.PRODUCT_LCI, "annual"),
        pro_addon_monthly_usd=_d("499.00"),
        consulting_addon_monthly_usd=_d("499.00"),
        enterprise_addon_annual_usd=_d("40000.00"),
        default_oem_rights=OEMRights.FORBIDDEN,
        requires_license_chain=True,
        license_chain_notes=(
            "Customer must bring their own ecoinvent / Sphera / GaBi license; "
            "GreenLang ships resolution layer only."
        ),
    ),
    PremiumPack.CONSTRUCTION_EPD: PremiumPackConfig(
        pack=PremiumPack.CONSTRUCTION_EPD,
        display_name="Construction EPD Premium",
        stripe_product_id=_pack_prod(PremiumPack.CONSTRUCTION_EPD),
        stripe_price_monthly_id=_pack_price(PremiumPack.CONSTRUCTION_EPD, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.CONSTRUCTION_EPD, "annual"),
        pro_addon_monthly_usd=_d("199.00"),
        consulting_addon_monthly_usd=_d("199.00"),
        enterprise_addon_annual_usd=_d("18000.00"),
        default_oem_rights=OEMRights.INTERNAL_ONLY,
        requires_license_chain=False,
    ),
    PremiumPack.AGRIFOOD_LAND: PremiumPackConfig(
        pack=PremiumPack.AGRIFOOD_LAND,
        display_name="Agrifood Premium",
        stripe_product_id=_pack_prod(PremiumPack.AGRIFOOD_LAND),
        stripe_price_monthly_id=_pack_price(PremiumPack.AGRIFOOD_LAND, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.AGRIFOOD_LAND, "annual"),
        pro_addon_monthly_usd=_d("199.00"),
        consulting_addon_monthly_usd=_d("199.00"),
        enterprise_addon_annual_usd=_d("24000.00"),
        default_oem_rights=OEMRights.INTERNAL_ONLY,
        requires_license_chain=False,
    ),
    PremiumPack.FINANCE_PROXY: PremiumPackConfig(
        pack=PremiumPack.FINANCE_PROXY,
        display_name="Finance Proxy Premium (PCAF)",
        stripe_product_id=_pack_prod(PremiumPack.FINANCE_PROXY),
        stripe_price_monthly_id=_pack_price(PremiumPack.FINANCE_PROXY, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.FINANCE_PROXY, "annual"),
        pro_addon_monthly_usd=_d("299.00"),
        consulting_addon_monthly_usd=_d("299.00"),
        enterprise_addon_annual_usd=_d("36000.00"),
        default_oem_rights=OEMRights.FORBIDDEN,
        requires_license_chain=True,
        license_chain_notes=(
            "Requires PCAF attribution license chain; redistribution "
            "forbidden without customer-side agreement."
        ),
    ),
    PremiumPack.CBAM_EU_POLICY: PremiumPackConfig(
        pack=PremiumPack.CBAM_EU_POLICY,
        display_name="CBAM / EU Policy Premium",
        stripe_product_id=_pack_prod(PremiumPack.CBAM_EU_POLICY),
        stripe_price_monthly_id=_pack_price(PremiumPack.CBAM_EU_POLICY, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.CBAM_EU_POLICY, "annual"),
        pro_addon_monthly_usd=_d("299.00"),
        # TODO(product): PRD 7.3 lists CBAM under "Included (1 of 3)"; the
        # standalone monthly rate on Consulting matches the Pro addon.
        consulting_addon_monthly_usd=_d("299.00"),
        enterprise_addon_annual_usd=_d("36000.00"),
        default_oem_rights=OEMRights.INTERNAL_ONLY,
        requires_license_chain=False,
    ),
    PremiumPack.LAND_REMOVALS: PremiumPackConfig(
        pack=PremiumPack.LAND_REMOVALS,
        display_name="Land / Removals Premium",
        stripe_product_id=_pack_prod(PremiumPack.LAND_REMOVALS),
        stripe_price_monthly_id=_pack_price(PremiumPack.LAND_REMOVALS, "monthly"),
        stripe_price_annual_id=_pack_price(PremiumPack.LAND_REMOVALS, "annual"),
        pro_addon_monthly_usd=_d("149.00"),
        consulting_addon_monthly_usd=_d("149.00"),
        enterprise_addon_annual_usd=_d("18000.00"),
        default_oem_rights=OEMRights.INTERNAL_ONLY,
        requires_license_chain=False,
    ),
}

# Every pack must be accessible under SOME tier; this is asserted at import
# time so a future PRD edit can't silently orphan a SKU.
_PACK_COVERAGE: Dict[PremiumPack, FrozenSet[Tier]] = {
    pack: frozenset(
        {
            tier
            for tier, cfg in _TIERS.items()
            if tier != Tier.COMMUNITY  # community never entitles premium
        }
    )
    for pack in PremiumPack
}


# ---------------------------------------------------------------------------
# Public catalog object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Catalog:
    """Immutable container for the full SKU catalog."""

    tiers: Mapping[Tier, TierConfig]
    packs: Mapping[PremiumPack, PremiumPackConfig]

    def tier(self, tier: Tier) -> TierConfig:
        """Return the :class:`TierConfig` for ``tier``."""
        return self.tiers[tier]

    def pack(self, pack: PremiumPack) -> PremiumPackConfig:
        """Return the :class:`PremiumPackConfig` for ``pack``."""
        return self.packs[pack]

    def all_stripe_product_ids(self) -> List[str]:
        """Return every ``prod_factors_*`` id managed by this catalog."""
        ids = [t.stripe_product_id for t in self.tiers.values()]
        ids.extend(p.stripe_product_id for p in self.packs.values())
        return ids

    def all_stripe_price_ids(self) -> List[str]:
        """Return every ``price_factors_*`` id managed by this catalog."""
        ids: List[str] = []
        for t in self.tiers.values():
            ids.append(t.stripe_price_monthly_id)
            ids.append(t.stripe_price_annual_id)
            for meter in t.meters:
                ids.append(_meter_price_id(t.tier, meter.meter))
        for p in self.packs.values():
            ids.append(p.stripe_price_monthly_id)
            ids.append(p.stripe_price_annual_id)
        return ids


CATALOG: Catalog = Catalog(tiers=dict(_TIERS), packs=dict(_PACKS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meter_price_id(tier: Tier, meter: Meter) -> str:
    """Stripe price id for a metered overage line on ``tier``."""
    return f"price_factors_meter_{meter.value}_{tier.value}"


def meter_price_id(tier: Tier, meter: Meter) -> str:
    """Public accessor for the metered-overage Stripe price id."""
    return _meter_price_id(tier, meter)


def allowed_for(tier: Tier, pack: PremiumPack) -> bool:
    """Return ``True`` if ``tier`` is allowed to purchase or bundle ``pack``.

    Community can't buy premium packs. Every other tier can — either as an
    included bundle (see :attr:`TierConfig.included_packs`) or as a paid
    addon (see :class:`PremiumPackConfig`).
    """
    if not isinstance(tier, Tier):
        tier = Tier(str(tier).lower().strip())
    if not isinstance(pack, PremiumPack):
        pack = PremiumPack(str(pack).lower().strip())
    return tier in _PACK_COVERAGE[pack]


def pack_included(tier: Tier, pack: PremiumPack) -> bool:
    """Return ``True`` if ``pack`` is bundled free at ``tier``."""
    return pack in CATALOG.tier(tier).included_packs


def overage_price(tier: Tier, meter: Meter, quantity: int) -> Decimal:
    """Compute the overage charge (USD) for ``quantity`` units on ``tier``.

    If ``quantity`` does not exceed the included allowance the returned
    amount is ``Decimal('0.00')``. If the meter has no overage pricing
    (e.g. Community hard-stops instead of billing overage) we return
    ``Decimal('0.00')`` as well — the caller is expected to enforce a 402
    / 429 at request time rather than silently bill.

    Raises:
        KeyError: if the tier does not define ``meter``.
        ValueError: if ``quantity`` is negative.
    """
    if quantity < 0:
        raise ValueError("quantity must be non-negative")
    tier_cfg = CATALOG.tier(tier)
    meter_def = tier_cfg.meter(meter)
    if meter_def is None:
        raise KeyError(
            f"Tier {tier.value!r} does not define meter {meter.value!r}"
        )
    overage_qty = max(0, quantity - meter_def.included_per_month)
    if overage_qty == 0 or meter_def.overage_unit_price_usd == _d("0"):
        return _d("0.00")
    total = (Decimal(overage_qty) * meter_def.overage_unit_price_usd).quantize(
        _d("0.01")
    )
    return total


def tier_entitlements(tier: Tier) -> Dict[str, object]:
    """Return a serialisable entitlement snapshot for ``tier``.

    This is the dict that the webhook handler should persist against a
    tenant row after a ``customer.subscription.created|updated`` event.
    """
    cfg = CATALOG.tier(tier)
    return {
        "tier": cfg.tier.value,
        "allowed_redistribution_classes": sorted(cfg.allowed_redistribution_classes),
        "allowed_method_profiles": sorted(cfg.allowed_method_profiles),
        "max_api_calls_per_day": cfg.max_api_calls_per_day,
        "private_registry_enabled": cfg.private_registry_enabled,
        "private_registry_entries": cfg.private_registry_entries,
        "oem_enabled": cfg.oem_enabled,
        "oem_rights": cfg.oem_rights,
        "sla_level": cfg.sla_level.value,
        "audit_bundle_allowed": cfg.audit_bundle_allowed,
        "bulk_export_max_rows": cfg.bulk_export_max_rows,
        "sso_scim_included": cfg.sso_scim_included,
        "annual_contract_required": cfg.annual_contract_required,
        "included_packs": sorted(p.value for p in cfg.included_packs),
    }


def tier_from_price_id(price_id: str) -> Optional[Tier]:
    """Reverse-lookup a tier from a Stripe price id.

    Used by the webhook handler to convert ``price_factors_pro_monthly``
    back into :data:`Tier.PRO` so entitlements can be persisted.
    """
    if not price_id:
        return None
    for tier, cfg in _TIERS.items():
        if price_id in (cfg.stripe_price_monthly_id, cfg.stripe_price_annual_id):
            return tier
        for m in cfg.meters:
            if price_id == _meter_price_id(tier, m.meter):
                return tier
    return None


def pack_from_price_id(price_id: str) -> Optional[PremiumPack]:
    """Reverse-lookup a premium pack from a Stripe price id."""
    if not price_id:
        return None
    for pack, cfg in _PACKS.items():
        if price_id in (cfg.stripe_price_monthly_id, cfg.stripe_price_annual_id):
            return pack
    return None


# ---------------------------------------------------------------------------
# Sanity checks that fire at import
# ---------------------------------------------------------------------------


def _assert_catalog_invariants() -> None:
    """Runtime assertions — cheap, and fail loud if the catalog is broken.

    We run these at import so a typo in the pricing table breaks CI
    rather than production.
    """
    # Every tier has a product + a monthly price + an annual price.
    for tier, cfg in _TIERS.items():
        assert cfg.stripe_product_id.startswith(
            "prod_factors_"
        ), f"{tier} product id must start with prod_factors_"
        assert cfg.stripe_price_monthly_id.startswith(
            "price_factors_"
        ), f"{tier} monthly price id must start with price_factors_"
        assert cfg.stripe_price_annual_id.startswith(
            "price_factors_"
        ), f"{tier} annual price id must start with price_factors_"
        assert cfg.monthly_price_usd >= _d(
            "0"
        ), f"{tier} monthly price must be >= 0"
        assert cfg.annual_price_usd >= _d(
            "0"
        ), f"{tier} annual price must be >= 0"

    # Every premium pack is accessible under some tier.
    for pack in PremiumPack:
        tiers = _PACK_COVERAGE.get(pack, frozenset())
        assert tiers, f"pack {pack} has no tier coverage"

    # Included packs can only reference packs in the catalog.
    for tier, cfg in _TIERS.items():
        for pack in cfg.included_packs:
            assert pack in _PACKS, f"{tier} bundles unknown pack {pack}"


_assert_catalog_invariants()


# ---------------------------------------------------------------------------
# Public 4-SKU catalog (FY27 Pricing Page surface)
# ---------------------------------------------------------------------------
#
# The internal catalog above keeps Consulting and Platform separate so the
# sales motion can negotiate the upper rung without tipping its hand on
# the entry rung. The Pricing Page, however, surfaces a single combined
# "Consulting / Platform" SKU (per the founder decision documented in the
# FY27 launch plan), giving four buy-able SKUs:
#
#     1. community              free, open-class only, 60 req/min
#     2. developer_pro          $499/mo + $0.001/req over 100k, open + selected licensed
#     3. consulting_platform    $2500/mo annual + usage, 5 sub-tenants, open + most licensed
#     4. enterprise             contact-sales, all classes including OEM redistribution
#
# Each SKU maps onto one of the internal Tier enum members so downstream
# entitlement checks remain consistent. ``consulting_platform`` resolves
# to ``Tier.PLATFORM`` (the more permissive of the two) because the public
# SKU is the upper inclusive offer.

PUBLIC_SKUS: List[Dict[str, Any]] = [
    {
        "plan_id": "community",
        "tier_name": Tier.COMMUNITY.value,
        "display_name": "Community",
        "tagline": "Free open-source emission factors for individuals, students, and OSS maintainers.",
        "price_usd_monthly": "0.00",
        "price_usd_annual": "0.00",
        "contact_sales": False,
        "self_serve": True,
        "rate_limit": {
            "requests_per_minute": 60,
            "requests_per_month_included": 1_000,
        },
        "overage_unit_price_usd": None,
        "license_classes": [
            REDISTRIBUTION_OPEN,
        ],
        "included_premium_packs": [],
        "included_sub_tenants": 0,
        "oem_redistribution": False,
        "sla": None,
        "features": [
            "Open-class emission factors only",
            "Public catalog browser",
            "Community Slack support",
            "Read-only API (no overrides)",
        ],
    },
    {
        "plan_id": "developer_pro",
        "tier_name": Tier.PRO.value,
        "display_name": "Developer Pro",
        "tagline": "For startups and consultants shipping into production. Usage-based pricing.",
        "price_usd_monthly": "499.00",
        "price_usd_annual": "4990.00",
        "contact_sales": False,
        "self_serve": True,
        "rate_limit": {
            "requests_per_minute": 1_000,
            "requests_per_month_included": 100_000,
        },
        "overage_unit_price_usd": "0.001",
        "license_classes": [
            REDISTRIBUTION_OPEN,
            REDISTRIBUTION_RESTRICTED,
        ],
        "included_premium_packs": [],
        "included_sub_tenants": 0,
        "oem_redistribution": False,
        "sla": SLALevel.UPTIME_99_5.value,
        "features": [
            "100,000 requests / month included",
            "$0.001 / request over included quota",
            "Open + selected licensed packs",
            "Private overrides (50 entries / project)",
            "99.5% uptime SLA",
            "Email support",
        ],
    },
    {
        "plan_id": "consulting_platform",
        "tier_name": Tier.PLATFORM.value,
        "display_name": "Consulting / Platform",
        "tagline": "For consulting firms and ESG SaaS platforms with multi-client portfolios.",
        "price_usd_monthly": "2500.00",
        "price_usd_annual": "25000.00",
        "contact_sales": False,
        "self_serve": True,
        "rate_limit": {
            "requests_per_minute": 5_000,
            "requests_per_month_included": 1_000_000,
        },
        "overage_unit_price_usd": "0.0008",
        "license_classes": [
            REDISTRIBUTION_OPEN,
            REDISTRIBUTION_RESTRICTED,
            REDISTRIBUTION_CUSTOMER_PRIVATE,
        ],
        "included_premium_packs": [
            PremiumPack.ELECTRICITY.value,
            PremiumPack.CBAM_EU_POLICY.value,
            PremiumPack.FREIGHT.value,
        ],
        "included_sub_tenants": 5,
        "oem_redistribution": False,
        "sla": SLALevel.UPTIME_99_9.value,
        "features": [
            "1,000,000 requests / month included",
            "5 sub-tenants included (additional $50/mo each)",
            "Most licensed premium packs included",
            "Customer-private overrides (unlimited per sub-tenant)",
            "Audit bundles + signed receipts",
            "99.9% uptime SLA",
            "Slack-Connect support channel",
        ],
    },
    {
        "plan_id": "enterprise",
        "tier_name": Tier.ENTERPRISE.value,
        "display_name": "Enterprise",
        "tagline": "For Fortune 500 enterprises with regulated reporting and OEM redistribution needs.",
        "price_usd_monthly": None,
        "price_usd_annual": None,
        "contact_sales": True,
        "self_serve": False,
        "rate_limit": {
            "requests_per_minute": 0,  # unlimited; surface as zero in JSON to mean "no limit"
            "requests_per_month_included": 0,
        },
        "overage_unit_price_usd": None,
        "license_classes": sorted(ALL_REDISTRIBUTION_CLASSES),
        "included_premium_packs": [p.value for p in PremiumPack],
        "included_sub_tenants": 0,  # negotiated per contract; not metered
        "oem_redistribution": True,
        "sla": SLALevel.UPTIME_99_95.value,
        "features": [
            "Unlimited requests (fair-use)",
            "All licensed packs + OEM white-label rights",
            "Customer-private + connector-only license classes",
            "SSO / SCIM included",
            "Full audit bundles + reproducibility manifests",
            "99.95% uptime SLA + named TAM",
            "Custom contract + DPA",
        ],
    },
]


def get_skus() -> List[Dict[str, Any]]:
    """Return the four public SKUs that drive the FY27 Pricing Page.

    Each entry is a plain ``dict`` (not a dataclass) so the FastAPI route
    that surfaces this list does not need to import private types and can
    serialise straight to JSON. The fields match :class:`PlanView` in
    :mod:`greenlang.factors.billing.api`.
    """
    return [dict(sku) for sku in PUBLIC_SKUS]


def get_sku_by_id(plan_id: str) -> Optional[Dict[str, Any]]:
    """Look up one SKU by its public ``plan_id`` slug.

    Args:
        plan_id: One of ``community``, ``developer_pro``,
            ``consulting_platform``, ``enterprise``.

    Returns:
        The SKU dict (a fresh copy) or ``None`` if no match.
    """
    needle = (plan_id or "").lower().strip()
    for sku in PUBLIC_SKUS:
        if sku["plan_id"] == needle:
            return dict(sku)
    return None


# Sanity: every public SKU must reference an existing internal tier.
for _public_sku in PUBLIC_SKUS:
    _t = Tier(_public_sku["tier_name"])
    assert _t in _TIERS, f"public SKU {_public_sku['plan_id']!r} -> unknown tier {_t}"
del _public_sku, _t

# Sanity: there must be exactly four public SKUs (Community, Developer Pro,
# Consulting/Platform, Enterprise) per the FY27 launch decision.
assert len(PUBLIC_SKUS) == 4, "PUBLIC_SKUS must contain exactly 4 entries"


__all__ = [
    "Tier",
    "PremiumPack",
    "Meter",
    "SLALevel",
    "MeterDefinition",
    "TierConfig",
    "PremiumPackConfig",
    "Catalog",
    "CATALOG",
    "REDISTRIBUTION_OPEN",
    "REDISTRIBUTION_RESTRICTED",
    "REDISTRIBUTION_CONNECTOR_ONLY",
    "REDISTRIBUTION_CUSTOMER_PRIVATE",
    "REDISTRIBUTION_INTERNAL_ONLY",
    "ALL_REDISTRIBUTION_CLASSES",
    "METHOD_PROFILE_CORPORATE",
    "METHOD_PROFILE_ELECTRICITY",
    "METHOD_PROFILE_FREIGHT",
    "METHOD_PROFILE_EU_POLICY",
    "METHOD_PROFILE_LAND",
    "METHOD_PROFILE_PRODUCT_CARBON",
    "METHOD_PROFILE_FINANCE",
    "OPEN_CORE_METHOD_PROFILES",
    "ALL_METHOD_PROFILES",
    "allowed_for",
    "pack_included",
    "overage_price",
    "tier_entitlements",
    "tier_from_price_id",
    "pack_from_price_id",
    "meter_price_id",
    # Public 4-SKU catalog
    "PUBLIC_SKUS",
    "get_skus",
    "get_sku_by_id",
]
