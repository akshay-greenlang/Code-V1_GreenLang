# -*- coding: utf-8 -*-
"""PCAF financed-emissions method packs (GAP-8).

Implements the seven PCAF asset classes from the **PCAF Global GHG Accounting
and Reporting Standard for the Financial Industry v2.0 (2022)**:

    1. Listed Equity & Corporate Bonds (share split into two packs for
       distinct disclosure context)
    2. Corporate Bonds  (separate pack: different EVIC denominator semantics)
    3. Business Loans & Unlisted Equity
    4. Project Finance
    5. Commercial Real Estate
    6. Mortgages
    7. Motor Vehicle Loans

Each pack encodes:

- **Attribution-factor hierarchy** (customer_specific > supplier_specific
  > sector_regional > sector_global > asset_class_default)
- **Boundary rule** — minimum Scope 1+2 of the financed counterparty;
  Scope 3 required for fossil-fuel and high-emitting sectors per PCAF
  Chapter 5.5 guidance
- **PCAF Data Quality Score (DQS 1-5)** — score 1 = verified emissions,
  score 5 = asset-class-default proxy
- **AR6 100-yr GWP basis**   (CTO non-negotiable #1 — gas-level vectors
  stored separately, CO2e derived via GWP set)
- **Uncertainty-band disclosure** required when DQS >= 4
- **Intensity modes** — absolute (tCO2e), economic intensity
  (tCO2e / EUR M revenue or EVIC), physical intensity
  (tCO2e / m2 floor area, / MWh, / km driven) as applicable per class.
- **Reporting labels** — PCAF-compliant output format.

Each variant self-registers with the method-pack registry on import AND
with the named sub-registry so callers can resolve
``get_pack("pcaf_listed_equity")`` or ``get_pack("pcaf_mortgages")``.

Source citations
----------------
- PCAF, *Global GHG Accounting and Reporting Standard Part A: Financed
  Emissions*, Second Edition, December 2022.
- GHG Protocol, *Corporate Value Chain (Scope 3) Accounting and Reporting
  Standard*, Category 15 — Investments.
- IPCC AR6 (WG1, 2021) — GWP 100-year values.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    DEFAULT_FALLBACK,
    DeprecationRule,
    FallbackStep,
    MarketInstrumentTreatment,
    MethodPack,
    SelectionRule,
)
from greenlang.factors.method_packs.registry import register_pack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PCAF-specific enums
# ---------------------------------------------------------------------------


class PCAFAssetClass(str, Enum):
    """The 7 PCAF asset classes covered in the Global Standard Part A."""

    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    BUSINESS_LOANS = "business_loans_unlisted_equity"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    MOTOR_VEHICLE_LOANS = "motor_vehicle_loans"


class PCAFDataQualityScore(int, Enum):
    """PCAF Data Quality Score (1 = best, 5 = worst)."""

    SCORE_1_VERIFIED = 1                # Audited / verified counterparty emissions
    SCORE_2_UNVERIFIED = 2              # Unverified counterparty-reported data
    SCORE_3_PHYSICAL_PROXY = 3          # Sector + physical activity proxy
    SCORE_4_ECONOMIC_PROXY = 4          # Sector + economic activity proxy
    SCORE_5_ASSET_CLASS_DEFAULT = 5     # Asset-class-average proxy


class PCAFAttributionMethod(str, Enum):
    """Attribution-factor denominator per PCAF equations."""

    EVIC = "evic"                       # Enterprise Value Including Cash
    OUTSTANDING_AMOUNT_PLUS_EQUITY = "outstanding_amount_plus_equity"
    COMMITTED_CAPITAL = "committed_capital"
    PROPERTY_VALUE_AT_ORIGINATION = "property_value_at_origination"
    VEHICLE_VALUE_AT_ORIGINATION = "vehicle_value_at_origination"


class PCAFIntensityMode(str, Enum):
    """Intensity reporting modes per PCAF Chapter 5."""

    ABSOLUTE = "absolute_tco2e"
    ECONOMIC_INTENSITY = "tco2e_per_eur_million"
    PHYSICAL_INTENSITY_FLOOR_AREA = "tco2e_per_m2"
    PHYSICAL_INTENSITY_ENERGY = "tco2e_per_mwh"
    PHYSICAL_INTENSITY_DISTANCE = "tco2e_per_km_driven"


# ---------------------------------------------------------------------------
# PCAF attribution-factor hierarchy (5-step)
# ---------------------------------------------------------------------------


PCAF_ATTRIBUTION_HIERARCHY: Tuple[FallbackStep, ...] = (
    FallbackStep(1, "customer_specific",
                 "Counterparty-verified Scope 1+2(+3) emissions"),
    FallbackStep(2, "supplier_specific",
                 "Value-chain supplier-reported data for the counterparty"),
    FallbackStep(3, "sector_regional",
                 "Sector average for the counterparty's country / region"),
    FallbackStep(4, "sector_global",
                 "Global sector average (GICS / NACE / NAICS)"),
    FallbackStep(5, "asset_class_default",
                 "PCAF asset-class default for the country or global"),
)


# ---------------------------------------------------------------------------
# Metadata payload carried on each variant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PCAFPackMetadata:
    """Per-variant metadata that sits alongside the immutable MethodPack.

    We keep this separate from the base :class:`MethodPack` so we do not
    modify the frozen base class (out of scope per GAP-8 instructions).
    The metadata is looked up via :func:`get_pcaf_metadata(variant_name)`.
    """

    variant_name: str
    asset_class: PCAFAssetClass
    attribution_method: PCAFAttributionMethod
    attribution_hierarchy: Tuple[FallbackStep, ...]
    intensity_modes: Tuple[PCAFIntensityMode, ...]
    requires_scope3_for_sectors: Tuple[str, ...]
    uncertainty_band_required_dqs: int
    dqs_scale: Tuple[PCAFDataQualityScore, ...] = (
        PCAFDataQualityScore.SCORE_1_VERIFIED,
        PCAFDataQualityScore.SCORE_2_UNVERIFIED,
        PCAFDataQualityScore.SCORE_3_PHYSICAL_PROXY,
        PCAFDataQualityScore.SCORE_4_ECONOMIC_PROXY,
        PCAFDataQualityScore.SCORE_5_ASSET_CLASS_DEFAULT,
    )
    attribution_formula: str = ""
    proxy_hierarchy: Tuple[FallbackStep, ...] = PCAF_ATTRIBUTION_HIERARCHY


# ---------------------------------------------------------------------------
# Named variant registry — lives NEXT TO the profile registry.
# ---------------------------------------------------------------------------


_variant_lock = threading.Lock()
_variants: Dict[str, MethodPack] = {}
_metadata: Dict[str, PCAFPackMetadata] = {}


def register_pcaf_variant(
    variant_name: str, pack: MethodPack, metadata: PCAFPackMetadata
) -> None:
    """Register a PCAF variant pack by name, with sidecar metadata."""
    if not variant_name:
        raise ValueError("variant_name cannot be empty")
    with _variant_lock:
        existing = _variants.get(variant_name)
        if existing is not None and existing.pack_version != pack.pack_version:
            logger.warning(
                "PCAF variant %s version bumped from %s to %s",
                variant_name, existing.pack_version, pack.pack_version,
            )
        _variants[variant_name] = pack
        _metadata[variant_name] = metadata
        logger.info(
            "Registered PCAF variant %s v%s (%s)",
            variant_name, pack.pack_version, metadata.asset_class.value,
        )


def get_pcaf_variant(variant_name: str) -> MethodPack:
    """Retrieve a registered PCAF variant pack."""
    with _variant_lock:
        pack = _variants.get(variant_name)
    if pack is None:
        raise KeyError(
            "no PCAF variant %r; available: %s"
            % (variant_name, sorted(_variants))
        )
    return pack


def get_pcaf_metadata(variant_name: str) -> PCAFPackMetadata:
    """Retrieve per-variant metadata (attribution formula, DQS rubric, ...)."""
    with _variant_lock:
        meta = _metadata.get(variant_name)
    if meta is None:
        raise KeyError(
            "no PCAF metadata for %r; available: %s"
            % (variant_name, sorted(_metadata))
        )
    return meta


def list_pcaf_variants() -> List[str]:
    """Return the registered PCAF variant names, sorted."""
    with _variant_lock:
        return sorted(_variants)


# ---------------------------------------------------------------------------
# Shared defaults for all PCAF variants
# ---------------------------------------------------------------------------


# PCAF high-emitting sectors (per Chapter 5.5) that REQUIRE Scope 3
# disclosure as part of the financed-emissions calculation.
_PCAF_SCOPE3_SECTORS: Tuple[str, ...] = (
    "oil_and_gas",
    "coal_mining",
    "power_generation",
    "steel",
    "cement",
    "aluminum",
    "chemicals",
    "aviation",
    "shipping",
    "road_transportation",
    "automobile_manufacturing",
    "commercial_construction",
    "residential_construction",
    "agriculture",
    "paper_and_forest_products",
    "mining_and_metals",
)


_DEPRECATION = DeprecationRule(max_age_days=365 * 4, grace_period_days=365)


def _pcaf_selection_rule() -> SelectionRule:
    """Common selection rule — all PCAF variants accept FINANCE_PROXY +
    EMISSIONS families with SPEND_PROXY / DIRECT_FACTOR formulas."""
    return SelectionRule(
        allowed_families=(
            FactorFamily.FINANCE_PROXY,
            FactorFamily.EMISSIONS,
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.ENERGY_CONVERSION,
            FactorFamily.GRID_INTENSITY,
            FactorFamily.CLASSIFICATION_MAPPING,
        ),
        allowed_formula_types=(
            FormulaType.SPEND_PROXY,
            FormulaType.DIRECT_FACTOR,
            FormulaType.LCA,
        ),
        allowed_statuses=("certified", "preview"),
    )


def _pcaf_boundary(allowed_scopes: Tuple[str, ...] = ("1", "2", "3")) -> BoundaryRule:
    return BoundaryRule(
        allowed_scopes=allowed_scopes,
        allowed_boundaries=("cradle_to_grave", "cradle_to_gate"),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    )


def _build_pack(
    *,
    variant_name: str,
    display_name: str,
    description: str,
    audit_template: str,
    pack_version: str = "1.0.0",
    tags: Tuple[str, ...] = ("finance", "pcaf", "licensed"),
    allowed_scopes: Tuple[str, ...] = ("1", "2", "3"),
) -> MethodPack:
    """Construct a MethodPack configured for a PCAF variant.

    Every variant re-uses the MethodProfile.FINANCE_PROXY enum value —
    differentiation is by ``variant_name`` (string) and by the per-variant
    metadata registered alongside.
    """
    return MethodPack(
        profile=MethodProfile.FINANCE_PROXY,
        name=display_name,
        description=description,
        selection_rule=_pcaf_selection_rule(),
        boundary_rule=_pcaf_boundary(allowed_scopes),
        gwp_basis="IPCC_AR6_100",
        region_hierarchy=DEFAULT_FALLBACK,
        deprecation=_DEPRECATION,
        reporting_labels=(
            "PCAF",
            "PCAF_Part_A_v2.0",
            "GHG_Protocol_Scope3_Cat15",
            "IFRS_S2",
        ),
        audit_text_template=audit_template,
        pack_version=pack_version,
        tags=tags + (variant_name,),
    )


# ---------------------------------------------------------------------------
# 1. PCAF Listed Equity
# ---------------------------------------------------------------------------

PCAF_LISTED_EQUITY = _build_pack(
    variant_name="pcaf_listed_equity",
    display_name="PCAF — Listed Equity (Asset Class 1)",
    description=(
        "Financed emissions from publicly traded equity investments per "
        "PCAF Global Standard Ch. 5.1. Attribution factor = outstanding "
        "investment / EVIC (Enterprise Value Including Cash). Requires "
        "Scope 3 disclosure for fossil-fuel and high-emitting sectors."
    ),
    audit_template=(
        "PCAF Listed Equity. Counterparty: {counterparty}. "
        "Investment: {outstanding_amount} {currency}. EVIC: {evic}. "
        "Attribution factor = outstanding / EVIC = {attribution_factor}. "
        "Scope 1+2 emissions: {scope12_emissions} tCO2e. "
        "Data quality score: {pcaf_dqs}/5. GWP basis: {gwp_basis}."
    ),
)
_PCAF_LISTED_EQUITY_META = PCAFPackMetadata(
    variant_name="pcaf_listed_equity",
    asset_class=PCAFAssetClass.LISTED_EQUITY,
    attribution_method=PCAFAttributionMethod.EVIC,
    attribution_hierarchy=PCAF_ATTRIBUTION_HIERARCHY,
    intensity_modes=(
        PCAFIntensityMode.ABSOLUTE,
        PCAFIntensityMode.ECONOMIC_INTENSITY,
    ),
    requires_scope3_for_sectors=_PCAF_SCOPE3_SECTORS,
    uncertainty_band_required_dqs=4,
    attribution_formula="attribution_factor = outstanding_amount / EVIC",
)


# ---------------------------------------------------------------------------
# 2. PCAF Corporate Bonds
# ---------------------------------------------------------------------------

PCAF_CORPORATE_BONDS = _build_pack(
    variant_name="pcaf_corporate_bonds",
    display_name="PCAF — Corporate Bonds (Asset Class 2)",
    description=(
        "Financed emissions from corporate bond holdings per PCAF Global "
        "Standard Ch. 5.1. EVIC-based attribution identical mathematically "
        "to listed equity, but reporting labels distinguish the instrument "
        "for regulated disclosures (fixed-income vs. equity)."
    ),
    audit_template=(
        "PCAF Corporate Bonds. Issuer: {counterparty}. "
        "Bond outstanding: {outstanding_amount} {currency}. EVIC: {evic}. "
        "Attribution factor = {attribution_factor}. "
        "Scope 1+2 emissions: {scope12_emissions} tCO2e. "
        "Data quality score: {pcaf_dqs}/5. GWP basis: {gwp_basis}."
    ),
)
_PCAF_CORPORATE_BONDS_META = PCAFPackMetadata(
    variant_name="pcaf_corporate_bonds",
    asset_class=PCAFAssetClass.CORPORATE_BONDS,
    attribution_method=PCAFAttributionMethod.EVIC,
    attribution_hierarchy=PCAF_ATTRIBUTION_HIERARCHY,
    intensity_modes=(
        PCAFIntensityMode.ABSOLUTE,
        PCAFIntensityMode.ECONOMIC_INTENSITY,
    ),
    requires_scope3_for_sectors=_PCAF_SCOPE3_SECTORS,
    uncertainty_band_required_dqs=4,
    attribution_formula="attribution_factor = bond_outstanding_amount / EVIC",
)


# ---------------------------------------------------------------------------
# 3. PCAF Business Loans & Unlisted Equity
# ---------------------------------------------------------------------------

PCAF_BUSINESS_LOANS = _build_pack(
    variant_name="pcaf_business_loans",
    display_name="PCAF — Business Loans & Unlisted Equity (Asset Class 3)",
    description=(
        "Financed emissions from business loans, credit lines, and unlisted "
        "equity investments per PCAF Global Standard Ch. 5.2. Attribution "
        "uses (outstanding_amount / (total_equity + total_debt)). When the "
        "denominator is unavailable, revenue-based proxy is allowed per "
        "PCAF Chapter 5.2.3, raising the data quality score (DQS 4 or 5)."
    ),
    audit_template=(
        "PCAF Business Loans. Borrower: {counterparty}. "
        "Loan outstanding: {outstanding_amount} {currency}. "
        "Denominator ({attribution_method}): {denominator}. "
        "Attribution factor = {attribution_factor}. "
        "Emissions: {scope12_emissions} tCO2e. DQS: {pcaf_dqs}/5."
    ),
)
_PCAF_BUSINESS_LOANS_META = PCAFPackMetadata(
    variant_name="pcaf_business_loans",
    asset_class=PCAFAssetClass.BUSINESS_LOANS,
    attribution_method=PCAFAttributionMethod.OUTSTANDING_AMOUNT_PLUS_EQUITY,
    attribution_hierarchy=PCAF_ATTRIBUTION_HIERARCHY,
    intensity_modes=(
        PCAFIntensityMode.ABSOLUTE,
        PCAFIntensityMode.ECONOMIC_INTENSITY,
    ),
    requires_scope3_for_sectors=_PCAF_SCOPE3_SECTORS,
    uncertainty_band_required_dqs=4,
    attribution_formula=(
        "attribution_factor = outstanding_amount / (total_equity + total_debt); "
        "fallback: outstanding_amount / annual_revenue (DQS 4 / 5)"
    ),
)


# ---------------------------------------------------------------------------
# 4. PCAF Project Finance
# ---------------------------------------------------------------------------

PCAF_PROJECT_FINANCE = _build_pack(
    variant_name="pcaf_project_finance",
    display_name="PCAF — Project Finance (Asset Class 4)",
    description=(
        "Financed emissions from project finance exposures per PCAF Global "
        "Standard Ch. 5.3. Attribution factor = outstanding_amount / "
        "total_committed_capital of the project. Scope 1+2+3 all in scope "
        "for project-level emissions (no 'corporate' layer in between)."
    ),
    audit_template=(
        "PCAF Project Finance. Project: {project_id}. Sponsor: {counterparty}. "
        "Outstanding: {outstanding_amount} {currency}. "
        "Committed capital: {committed_capital}. "
        "Attribution factor = {attribution_factor}. "
        "Project lifecycle emissions: {project_emissions} tCO2e. DQS: {pcaf_dqs}/5."
    ),
)
_PCAF_PROJECT_FINANCE_META = PCAFPackMetadata(
    variant_name="pcaf_project_finance",
    asset_class=PCAFAssetClass.PROJECT_FINANCE,
    attribution_method=PCAFAttributionMethod.COMMITTED_CAPITAL,
    attribution_hierarchy=PCAF_ATTRIBUTION_HIERARCHY,
    intensity_modes=(
        PCAFIntensityMode.ABSOLUTE,
        PCAFIntensityMode.PHYSICAL_INTENSITY_ENERGY,
    ),
    requires_scope3_for_sectors=_PCAF_SCOPE3_SECTORS,
    uncertainty_band_required_dqs=4,
    attribution_formula=(
        "attribution_factor = outstanding_amount / total_committed_capital"
    ),
)


# ---------------------------------------------------------------------------
# 5. PCAF Commercial Real Estate
# ---------------------------------------------------------------------------

PCAF_COMMERCIAL_REAL_ESTATE = _build_pack(
    variant_name="pcaf_commercial_real_estate",
    display_name="PCAF — Commercial Real Estate (Asset Class 5)",
    description=(
        "Financed emissions from commercial real estate lending and equity "
        "per PCAF Global Standard Ch. 5.4. Emissions derived from building "
        "energy consumption (actual primary data > energy-labels / EPCs > "
        "statistical building-type averages). Attribution factor uses "
        "outstanding_amount / property_value_at_origination."
    ),
    audit_template=(
        "PCAF CRE. Property: {property_id}. Building type: {building_type}. "
        "Floor area: {floor_area_m2} m2. "
        "Energy intensity: {energy_intensity_kwh_per_m2} kWh/m2. "
        "Attribution factor = {attribution_factor}. "
        "Emissions: {emissions_tco2e} tCO2e. DQS: {pcaf_dqs}/5."
    ),
    allowed_scopes=("1", "2"),
)
_PCAF_CRE_META = PCAFPackMetadata(
    variant_name="pcaf_commercial_real_estate",
    asset_class=PCAFAssetClass.COMMERCIAL_REAL_ESTATE,
    attribution_method=PCAFAttributionMethod.PROPERTY_VALUE_AT_ORIGINATION,
    attribution_hierarchy=PCAF_ATTRIBUTION_HIERARCHY,
    intensity_modes=(
        PCAFIntensityMode.ABSOLUTE,
        PCAFIntensityMode.PHYSICAL_INTENSITY_FLOOR_AREA,
        PCAFIntensityMode.PHYSICAL_INTENSITY_ENERGY,
    ),
    requires_scope3_for_sectors=(),
    uncertainty_band_required_dqs=4,
    attribution_formula=(
        "emissions = floor_area_m2 * energy_intensity_kwh_per_m2 * "
        "grid_factor_kgco2e_per_kwh * attribution_factor; "
        "attribution_factor = outstanding_amount / property_value_at_origination"
    ),
)


# ---------------------------------------------------------------------------
# 6. PCAF Mortgages
# ---------------------------------------------------------------------------

PCAF_MORTGAGES = _build_pack(
    variant_name="pcaf_mortgages",
    display_name="PCAF — Mortgages (Asset Class 6)",
    description=(
        "Financed emissions from residential mortgage portfolios per PCAF "
        "Global Standard Ch. 5.5. Emissions derived from floor-area x "
        "energy-intensity x grid factor, attributed by outstanding loan / "
        "property value at origination. Scope 1 (on-site fossil heating) + "
        "Scope 2 (electricity) only. DQS 1 = metered energy data; "
        "DQS 5 = national building-stock average."
    ),
    audit_template=(
        "PCAF Mortgages. Mortgage ID: {mortgage_id}. "
        "Property: {property_type}, {floor_area_m2} m2. "
        "Outstanding: {outstanding_amount}. "
        "Property value at origination: {property_value_at_origination}. "
        "Attribution factor = {attribution_factor}. "
        "Attributed emissions: {attributed_emissions_tco2e} tCO2e. "
        "DQS: {pcaf_dqs}/5."
    ),
    allowed_scopes=("1", "2"),
)
_PCAF_MORTGAGES_META = PCAFPackMetadata(
    variant_name="pcaf_mortgages",
    asset_class=PCAFAssetClass.MORTGAGES,
    attribution_method=PCAFAttributionMethod.PROPERTY_VALUE_AT_ORIGINATION,
    attribution_hierarchy=PCAF_ATTRIBUTION_HIERARCHY,
    intensity_modes=(
        PCAFIntensityMode.ABSOLUTE,
        PCAFIntensityMode.PHYSICAL_INTENSITY_FLOOR_AREA,
    ),
    requires_scope3_for_sectors=(),
    uncertainty_band_required_dqs=4,
    attribution_formula=(
        "emissions = floor_area_m2 * energy_intensity * grid_factor * "
        "attribution_factor; "
        "attribution_factor = outstanding_mortgage / property_value_at_origination"
    ),
)


# ---------------------------------------------------------------------------
# 7. PCAF Motor Vehicle Loans
# ---------------------------------------------------------------------------

PCAF_MOTOR_VEHICLE_LOANS = _build_pack(
    variant_name="pcaf_motor_vehicle_loans",
    display_name="PCAF — Motor Vehicle Loans (Asset Class 7)",
    description=(
        "Financed emissions from motor vehicle loans and leases per PCAF "
        "Global Standard Ch. 5.6. Emissions derived from "
        "annual_km_driven x fuel/electricity intensity x CO2e factor, "
        "attributed by outstanding_loan / vehicle_value_at_origination. "
        "Tank-to-wheel minimum; well-to-wheel recommended when data exists."
    ),
    audit_template=(
        "PCAF Motor Vehicle Loans. Vehicle: {vehicle_id}. "
        "Type: {vehicle_type}, fuel: {fuel_type}. "
        "Annual distance: {km_driven_per_year} km. "
        "Attribution factor = {attribution_factor}. "
        "Attributed emissions: {attributed_emissions_tco2e} tCO2e. "
        "DQS: {pcaf_dqs}/5."
    ),
    allowed_scopes=("1", "2", "3"),
)
_PCAF_MOTOR_VEHICLE_META = PCAFPackMetadata(
    variant_name="pcaf_motor_vehicle_loans",
    asset_class=PCAFAssetClass.MOTOR_VEHICLE_LOANS,
    attribution_method=PCAFAttributionMethod.VEHICLE_VALUE_AT_ORIGINATION,
    attribution_hierarchy=PCAF_ATTRIBUTION_HIERARCHY,
    intensity_modes=(
        PCAFIntensityMode.ABSOLUTE,
        PCAFIntensityMode.PHYSICAL_INTENSITY_DISTANCE,
    ),
    requires_scope3_for_sectors=(),
    uncertainty_band_required_dqs=4,
    attribution_formula=(
        "emissions = km_driven_per_year * fuel_economy * emission_factor * "
        "attribution_factor; "
        "attribution_factor = outstanding_loan / vehicle_value_at_origination"
    ),
)


# ---------------------------------------------------------------------------
# Umbrella pack (back-compat) — keeps ``MethodProfile.FINANCE_PROXY`` valid
# ---------------------------------------------------------------------------

FINANCE_PROXY = _build_pack(
    variant_name="pcaf_umbrella",
    display_name="Financed Emissions (PCAF — all asset classes)",
    description=(
        "Umbrella PCAF pack covering all seven asset classes. Routes to a "
        "specific variant pack via the ``variant_name`` resolver: "
        "pcaf_listed_equity, pcaf_corporate_bonds, pcaf_business_loans, "
        "pcaf_project_finance, pcaf_commercial_real_estate, pcaf_mortgages, "
        "pcaf_motor_vehicle_loans."
    ),
    audit_template=(
        "PCAF asset class {asset_class}. Attribution factor: "
        "{attribution_factor}. DQS: {pcaf_dqs}/5. GWP: {gwp_basis}."
    ),
)


# ---------------------------------------------------------------------------
# DQS rubric (PCAF Table 5.3 / 5.5) — exported for config + tests
# ---------------------------------------------------------------------------


PCAF_DQS_RUBRIC: Dict[int, str] = {
    1: (
        "Verified/audited emissions data from the counterparty "
        "(Scope 1+2(+3)) - highest quality."
    ),
    2: (
        "Unverified counterparty-reported emissions data "
        "(CDP / sustainability report, not audited)."
    ),
    3: (
        "Sector emissions factor combined with PHYSICAL activity data "
        "(e.g. floor area, energy use, km driven)."
    ),
    4: (
        "Sector emissions factor combined with ECONOMIC activity data "
        "(e.g. revenue, asset turnover) - uncertainty-band disclosure required."
    ),
    5: (
        "Asset-class-average proxy - lowest quality; uncertainty-band "
        "disclosure AND coverage-ratio disclosure required."
    ),
}


# ---------------------------------------------------------------------------
# Module-level registration
# ---------------------------------------------------------------------------


_ALL_VARIANTS: Tuple[Tuple[MethodPack, PCAFPackMetadata], ...] = (
    (PCAF_LISTED_EQUITY, _PCAF_LISTED_EQUITY_META),
    (PCAF_CORPORATE_BONDS, _PCAF_CORPORATE_BONDS_META),
    (PCAF_BUSINESS_LOANS, _PCAF_BUSINESS_LOANS_META),
    (PCAF_PROJECT_FINANCE, _PCAF_PROJECT_FINANCE_META),
    (PCAF_COMMERCIAL_REAL_ESTATE, _PCAF_CRE_META),
    (PCAF_MORTGAGES, _PCAF_MORTGAGES_META),
    (PCAF_MOTOR_VEHICLE_LOANS, _PCAF_MOTOR_VEHICLE_META),
)


def _register_all() -> None:
    """Register every PCAF variant pack + umbrella with both registries."""
    for pack, meta in _ALL_VARIANTS:
        register_pcaf_variant(meta.variant_name, pack, meta)

    # Umbrella stays on MethodProfile.FINANCE_PROXY so legacy callers
    # using get_pack(MethodProfile.FINANCE_PROXY) keep working.
    register_pack(FINANCE_PROXY)


_register_all()


# ---------------------------------------------------------------------------
# Pydantic-style classes exposed by GAP-8 spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PCAFVariantHolder:
    """Lightweight class wrapper exposing the underlying MethodPack.

    GAP-8 expects classes named e.g. ``PCAFListedEquityPack``. Keeping the
    MethodPack dataclass immutable is a hard constraint, so we expose
    *variant holder* classes whose instances reference the registered
    :class:`MethodPack` + metadata. Callers access ``.pack`` and ``.metadata``
    directly or use the helper accessors below.
    """

    variant_name: str

    @property
    def pack(self) -> MethodPack:
        return get_pcaf_variant(self.variant_name)

    @property
    def metadata(self) -> PCAFPackMetadata:
        return get_pcaf_metadata(self.variant_name)


class PCAFListedEquityPack(_PCAFVariantHolder):
    """PCAF Asset Class 1 — Listed Equity (EVIC-based attribution)."""

    def __init__(self) -> None:
        super().__init__(variant_name="pcaf_listed_equity")


class PCAFCorporateBondsPack(_PCAFVariantHolder):
    """PCAF Asset Class 2 — Corporate Bonds (EVIC-based attribution)."""

    def __init__(self) -> None:
        super().__init__(variant_name="pcaf_corporate_bonds")


class PCAFBusinessLoansPack(_PCAFVariantHolder):
    """PCAF Asset Class 3 — Business Loans & Unlisted Equity."""

    def __init__(self) -> None:
        super().__init__(variant_name="pcaf_business_loans")


class PCAFProjectFinancePack(_PCAFVariantHolder):
    """PCAF Asset Class 4 — Project Finance (committed-capital attribution)."""

    def __init__(self) -> None:
        super().__init__(variant_name="pcaf_project_finance")


class PCAFCommercialRealEstatePack(_PCAFVariantHolder):
    """PCAF Asset Class 5 — Commercial Real Estate (floor-area x intensity)."""

    def __init__(self) -> None:
        super().__init__(variant_name="pcaf_commercial_real_estate")


class PCAFMortgagesPack(_PCAFVariantHolder):
    """PCAF Asset Class 6 — Residential Mortgages."""

    def __init__(self) -> None:
        super().__init__(variant_name="pcaf_mortgages")


class PCAFMotorVehicleLoansPack(_PCAFVariantHolder):
    """PCAF Asset Class 7 — Motor Vehicle Loans & Leases."""

    def __init__(self) -> None:
        super().__init__(variant_name="pcaf_motor_vehicle_loans")


__all__ = [
    # Enums
    "PCAFAssetClass",
    "PCAFDataQualityScore",
    "PCAFAttributionMethod",
    "PCAFIntensityMode",
    # Data classes
    "PCAFPackMetadata",
    # Class wrappers
    "PCAFListedEquityPack",
    "PCAFCorporateBondsPack",
    "PCAFBusinessLoansPack",
    "PCAFProjectFinancePack",
    "PCAFCommercialRealEstatePack",
    "PCAFMortgagesPack",
    "PCAFMotorVehicleLoansPack",
    # MethodPack instances
    "FINANCE_PROXY",
    "PCAF_LISTED_EQUITY",
    "PCAF_CORPORATE_BONDS",
    "PCAF_BUSINESS_LOANS",
    "PCAF_PROJECT_FINANCE",
    "PCAF_COMMERCIAL_REAL_ESTATE",
    "PCAF_MORTGAGES",
    "PCAF_MOTOR_VEHICLE_LOANS",
    # Registry helpers
    "register_pcaf_variant",
    "get_pcaf_variant",
    "get_pcaf_metadata",
    "list_pcaf_variants",
    # Reference data
    "PCAF_DQS_RUBRIC",
    "PCAF_ATTRIBUTION_HIERARCHY",
]
