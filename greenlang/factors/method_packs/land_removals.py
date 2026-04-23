# -*- coding: utf-8 -*-
"""GHG Protocol Land Sector & Removals (LSR) method packs (GAP-9).

Implements four variants of the LSR pack from the **GHG Protocol Land
Sector and Removals Guidance (2024)**:

    1. Land-Use Emissions       (deforestation, drainage, peatland oxidation)
    2. Land Management          (ongoing cropland / grassland management)
    3. Active Removals          (net-new carbon removed from atmosphere)
    4. Carbon Storage           (durable storage of previously removed carbon)

Each variant encodes:

- **Selection-rule hierarchy** — project-specific data > regional Tier 2 >
  IPCC Tier 1 defaults (IPCC 2006 GL + 2019 Refinement).
- **Boundary rule** — direct land use, indirect land-use change (iLUC),
  and soil-organic-carbon (SOC) changes.
- **Permanence class** — short (< 10 y), medium (10-100 y), long (> 100 y).
- **Reversal-risk flag** + buffer-pool % per class.
- **Biogenic accounting treatment** — carbon-neutral vs. sequestration-
  tracked, with explicit gas-level storage (CTO non-negotiable #1).
- **Removal-type categorisation** — nature-based (afforestation, peatland
  rewetting, soil carbon, blue carbon) or technology-based (biochar, BECCS,
  DACCS, enhanced weathering, mineralisation, ocean alkalinity).
- **Sequestration vs. storage** — distinguish active removal (net flux)
  from storage (lock of previously removed carbon).
- **MRV requirements** — measurement method, reporting frequency,
  verification standard (VCS / Gold Standard / Puro.earth / Isometric /
  Climeworks-verified).

Each variant self-registers with the method-pack registry on import AND
with the named sub-registry so callers can resolve
``get_pack("lsr_land_use_emissions")``, etc.

Source citations
----------------
- GHG Protocol, *Land Sector and Removals Guidance (Draft Public Comment
  Version)*, September 2024.
- IPCC, *2006 Guidelines for National Greenhouse Gas Inventories, Volume
  4: Agriculture, Forestry and Other Land Use*, with *2019 Refinement*.
- ICVCM, *Core Carbon Principles & Assessment Framework*, 2023.
- Verra VCS Standard v4.4; Gold Standard for the Global Goals v1.2;
  Puro.earth Supplier General Rules v2.0; Isometric Standard v1.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    CannotResolveAction,
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
# LSR-specific enums
# ---------------------------------------------------------------------------


class PermanenceClass(str, Enum):
    """Permanence tiers for stored / removed carbon (GHG LSR Ch. 9)."""

    SHORT = "short"                     # < 10 years
    MEDIUM = "medium"                   # 10 - 100 years
    LONG = "long"                       # > 100 years


class ReversalRiskLevel(str, Enum):
    """Reversal-risk flag — drives buffer-pool contribution."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BiogenicAccountingTreatment(str, Enum):
    """LSR biogenic CO2 treatment distinct from base BiogenicTreatment."""

    CARBON_NEUTRAL = "carbon_neutral"                # zero-rate biogenic CO2
    SEQUESTRATION_TRACKED = "sequestration_tracked"  # explicit flux tracking
    STORAGE_TRACKED = "storage_tracked"              # stock-based accounting


class RemovalCategory(str, Enum):
    """High-level classification of a removal / storage activity."""

    NATURE_BASED = "nature_based"
    TECHNOLOGY_BASED = "technology_based"
    HYBRID = "hybrid"                   # biochar / BECCS share both features


class RemovalType(str, Enum):
    """Specific removal-type taxonomy from LSR Annex + Puro.earth + ICVCM."""

    # Nature-based
    AFFORESTATION = "afforestation"
    REFORESTATION = "reforestation"
    FOREST_MANAGEMENT = "forest_management"
    PEATLAND_REWETTING = "peatland_rewetting"
    SOIL_CARBON_SEQUESTRATION = "soil_carbon_sequestration"
    BLUE_CARBON_MANGROVE = "blue_carbon_mangrove"
    BLUE_CARBON_SEAGRASS = "blue_carbon_seagrass"
    BLUE_CARBON_SALTMARSH = "blue_carbon_saltmarsh"
    BIOCHAR = "biochar"                                   # hybrid; often nature-adjacent
    # Technology-based
    BECCS = "beccs"                                       # bioenergy + CCS
    DACCS = "daccs"                                       # direct air capture + CCS
    ENHANCED_ROCK_WEATHERING = "enhanced_rock_weathering"
    MINERAL_CARBONATION = "mineral_carbonation"
    OCEAN_ALKALINITY_ENHANCEMENT = "ocean_alkalinity_enhancement"


class VerificationStandard(str, Enum):
    """Approved MRV / issuance standards for removal credits."""

    VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    PURO_EARTH = "puro_earth"
    ISOMETRIC = "isometric"
    CLIMEWORKS_VERIFIED = "climeworks_verified"
    ACR = "american_carbon_registry"
    CAR = "climate_action_reserve"
    ICVCM_CCP_APPROVED = "icvcm_ccp_approved"


class ReportingFrequency(str, Enum):
    """MRV reporting cadence."""

    ANNUAL = "annual"
    BIENNIAL = "biennial"
    QUINQUENNIAL = "quinquennial"
    PROJECT_LIFECYCLE = "project_lifecycle"


# ---------------------------------------------------------------------------
# v0.2 enums (Wave 4-G promotion) — standard-template alignment
# ---------------------------------------------------------------------------


class LandUseCategory(str, Enum):
    """IPCC 2006 GL / 2019 Refinement Volume 4 land-use categories."""

    FOREST = "forest"
    GRASSLAND = "grassland"
    CROPLAND = "cropland"
    WETLAND = "wetland"
    SETTLEMENT = "settlement"
    OTHER = "other"


class SequestrationBasis(str, Enum):
    """Basis on which sequestration / removals are accounted for."""

    BIOMASS = "biomass"                             # above + below-ground biomass
    SOIL = "soil"                                   # soil organic carbon
    HARVESTED_WOOD_PRODUCTS = "harvested_wood_products"
    MINERALIZATION = "mineralization"


class RemovalBasis(str, Enum):
    """GHG Protocol LSR Ch. 6 removal-basis classification."""

    NATURE_BASED = "nature_based"
    TECHNOLOGY_BASED = "technology_based"
    HYBRID = "hybrid"


class PermanenceClassV2(str, Enum):
    """Permanence class per GHG LSR Ch. 9 + ICVCM-CCP paragraph 8.

    Distinct from the legacy :class:`PermanenceClass` (short / medium /
    long) so the v0.2 surface aligns with the audit spec's exact labels.
    """

    LONG_TERM_100YR = "long_term_100yr"             # >= 100 years durable
    MEDIUM_25_100YR = "medium_25_100yr"             # 25 - 99 years
    SHORT_LT_25YR = "short_lt_25yr"                 # < 25 years


# ---------------------------------------------------------------------------
# Metadata payload carried on each LSR variant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LSRPackMetadata:
    """Per-variant metadata for Land Sector & Removals packs."""

    variant_name: str
    permanence_class: PermanenceClass
    reversal_risk: ReversalRiskLevel
    buffer_pool_pct: float                  # 0.0-1.0
    biogenic_treatment: BiogenicAccountingTreatment
    removal_category: Optional[RemovalCategory]
    allowed_removal_types: Tuple[RemovalType, ...]
    verification_standards: Tuple[VerificationStandard, ...]
    reporting_frequency: ReportingFrequency
    is_active_removal: bool                 # True = new flux, False = storage lock
    iluc_included: bool = False             # indirect land-use change in scope?
    soc_tracked: bool = False               # soil organic carbon changes tracked?
    direct_land_use_included: bool = True


# Permanence-class -> default buffer pool % (GHG LSR + ICVCM CCP guidance).
DEFAULT_BUFFER_POOL: Dict[PermanenceClass, float] = {
    PermanenceClass.SHORT: 0.35,    # 35% — <10y tenure has aggressive buffer
    PermanenceClass.MEDIUM: 0.20,   # 20% — 10-100y
    PermanenceClass.LONG: 0.10,     # 10% — >100y durable technology
}


# Risk-level -> buffer-pool multiplier applied on top of permanence default.
RISK_BUFFER_MULTIPLIER: Dict[ReversalRiskLevel, float] = {
    ReversalRiskLevel.LOW: 1.0,
    ReversalRiskLevel.MEDIUM: 1.5,
    ReversalRiskLevel.HIGH: 2.0,
}


# ---------------------------------------------------------------------------
# LSR-specific fallback chain (project primary > Tier 2 regional > Tier 1 IPCC)
# ---------------------------------------------------------------------------


LSR_FALLBACK_HIERARCHY: Tuple[FallbackStep, ...] = (
    FallbackStep(1, "project_specific",
                 "Project-site measured data (soil cores, LiDAR biomass, ...)"),
    FallbackStep(2, "regional_tier2",
                 "IPCC Tier 2 country / regional emission factor"),
    FallbackStep(3, "tier1_default",
                 "IPCC Tier 1 default factor (2006 GL + 2019 Refinement)"),
    FallbackStep(4, "global_default",
                 "Global average proxy (last resort - DQS reduced)"),
)


# ---------------------------------------------------------------------------
# v0.2 — LSR-specific fallback hierarchy (customer > supplier > project
# > national default). NO global default tier (cannot resolve safely is
# the contract when all four tiers fail).
# ---------------------------------------------------------------------------


LSR_V02_FALLBACK_HIERARCHY: Tuple[FallbackStep, ...] = (
    FallbackStep(1, "customer_removal",
                 "Tenant-supplied / customer-measured removal factor"),
    FallbackStep(2, "supplier_removal",
                 "Supplier-specific measured removal factor"),
    FallbackStep(3, "project_level",
                 "Project-level site-measured data (VCS / Puro.earth / etc.)"),
    FallbackStep(4, "national_default",
                 "National / IPCC Tier 2 default factor"),
    # NOTE: no global-default tier — cannot_resolve_safely fires instead.
)


def icvcm_ccp_compliant(verification_standards: Tuple[VerificationStandard, ...]) -> bool:
    """Check whether any of the given verification standards satisfies the
    ICVCM Core Carbon Principles compliance gate.

    The ICVCM-CCP designation itself is direct approval; VCS, Gold Standard,
    Puro.earth, and Isometric are eligible when the *specific methodology*
    has been ICVCM-CCP approved. TODO(methodology-review): we currently
    treat the presence of ``ICVCM_CCP_APPROVED`` OR any of the four
    top-tier standards as "compliant"; methodology lead must confirm
    whether we should gate on ICVCM approval exclusively.
    """
    if VerificationStandard.ICVCM_CCP_APPROVED in verification_standards:
        return True
    compliant_via_methodology = {
        VerificationStandard.VCS,
        VerificationStandard.GOLD_STANDARD,
        VerificationStandard.PURO_EARTH,
        VerificationStandard.ISOMETRIC,
    }
    return bool(set(verification_standards) & compliant_via_methodology)


# ---------------------------------------------------------------------------
# Named variant registry
# ---------------------------------------------------------------------------


_variant_lock = threading.Lock()
_variants: Dict[str, MethodPack] = {}
_metadata: Dict[str, LSRPackMetadata] = {}


def register_lsr_variant(
    variant_name: str, pack: MethodPack, metadata: LSRPackMetadata
) -> None:
    """Register an LSR variant pack by name, with sidecar metadata."""
    if not variant_name:
        raise ValueError("variant_name cannot be empty")
    with _variant_lock:
        existing = _variants.get(variant_name)
        if existing is not None and existing.pack_version != pack.pack_version:
            logger.warning(
                "LSR variant %s version bumped from %s to %s",
                variant_name, existing.pack_version, pack.pack_version,
            )
        _variants[variant_name] = pack
        _metadata[variant_name] = metadata
        logger.info(
            "Registered LSR variant %s v%s (permanence=%s, reversal=%s)",
            variant_name,
            pack.pack_version,
            metadata.permanence_class.value,
            metadata.reversal_risk.value,
        )


def get_lsr_variant(variant_name: str) -> MethodPack:
    """Retrieve a registered LSR variant pack."""
    with _variant_lock:
        pack = _variants.get(variant_name)
    if pack is None:
        raise KeyError(
            "no LSR variant %r; available: %s"
            % (variant_name, sorted(_variants))
        )
    return pack


def get_lsr_metadata(variant_name: str) -> LSRPackMetadata:
    """Retrieve per-variant LSR metadata."""
    with _variant_lock:
        meta = _metadata.get(variant_name)
    if meta is None:
        raise KeyError(
            "no LSR metadata for %r; available: %s"
            % (variant_name, sorted(_metadata))
        )
    return meta


def list_lsr_variants() -> List[str]:
    """Return the registered LSR variant names, sorted."""
    with _variant_lock:
        return sorted(_variants)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_DEPRECATION = DeprecationRule(max_age_days=365 * 5, grace_period_days=730)


def _lsr_selection_rule(
    allowed_formula_types: Tuple[FormulaType, ...] = (
        FormulaType.DIRECT_FACTOR,
        FormulaType.CARBON_BUDGET,
        FormulaType.LCA,
    ),
) -> SelectionRule:
    return SelectionRule(
        allowed_families=(
            FactorFamily.LAND_USE_REMOVALS,
            FactorFamily.EMISSIONS,
            FactorFamily.MATERIAL_EMBODIED,
        ),
        allowed_formula_types=allowed_formula_types,
        allowed_statuses=("certified", "preview"),
    )


def _build_pack(
    *,
    variant_name: str,
    display_name: str,
    description: str,
    audit_template: str,
    biogenic: BiogenicTreatment,
    market_instruments: MarketInstrumentTreatment,
    allowed_scopes: Tuple[str, ...] = ("1", "3"),
    pack_version: str = "0.2.0",
    tags: Tuple[str, ...] = ("land", "lsr", "licensed"),
) -> MethodPack:
    """Build an LSR MethodPack for a given variant.

    v0.2 (Wave 4-G): uses :data:`LSR_V02_FALLBACK_HIERARCHY` (customer ->
    supplier -> project -> national default, NO global default), strict
    ``RAISE_NO_SAFE_MATCH`` contract, and 180-day advance notice on
    deprecation.
    """
    return MethodPack(
        profile=MethodProfile.LAND_REMOVALS,
        name=display_name,
        description=description,
        selection_rule=_lsr_selection_rule(),
        boundary_rule=BoundaryRule(
            allowed_scopes=allowed_scopes,
            allowed_boundaries=("cradle_to_grave",),
            biogenic_treatment=biogenic,
            market_instruments=market_instruments,
        ),
        gwp_basis="IPCC_AR6_100",
        region_hierarchy=LSR_V02_FALLBACK_HIERARCHY,
        deprecation=_DEPRECATION,
        reporting_labels=(
            "GHG_Protocol_LSR",
            "GHG_Protocol_LSR_Standard_2024",
            "IPCC_2006_GL",
            "IPCC_2019_Refinement",
            "ICVCM_CCP",
        ),
        audit_text_template=audit_template,
        pack_version=pack_version,
        tags=tags + (variant_name,),
        cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
        global_default_tier_allowed=False,
        replacement_pack_id=None,
        deprecation_notice_days=180,
    )


# ---------------------------------------------------------------------------
# 1. Land-Use Emissions Pack
# ---------------------------------------------------------------------------

GHG_LSR_LAND_USE_EMISSIONS = _build_pack(
    variant_name="lsr_land_use_emissions",
    display_name="GHG LSR — Land-Use Emissions",
    description=(
        "Direct land-use emissions: deforestation-driven CO2, peatland "
        "drainage oxidation, and land-use conversion per GHG Protocol Land "
        "Sector and Removals Standard. Scope 1 for owned / controlled land; "
        "Scope 3 Cat 1 for supply-chain land footprint. Includes both direct "
        "land use AND indirect land-use change (iLUC) where data allows, "
        "plus soil organic carbon (SOC) flux."
    ),
    audit_template=(
        "LSR Land-Use Emissions. Site: {site_id}. "
        "Land-use transition: {from_class} -> {to_class}. "
        "Area: {area_ha} ha. AGB change: {agb_delta_tc} tC. "
        "SOC change: {soc_delta_tc} tC. iLUC allocated: {iluc_tco2e} tCO2e. "
        "Total: {total_emissions_tco2e} tCO2e. "
        "Tier: {ipcc_tier}. GWP: {gwp_basis}."
    ),
    biogenic=BiogenicTreatment.INCLUDED,
    market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    allowed_scopes=("1", "3"),
)
_LUE_META = LSRPackMetadata(
    variant_name="lsr_land_use_emissions",
    permanence_class=PermanenceClass.LONG,
    reversal_risk=ReversalRiskLevel.LOW,         # emissions already realised
    buffer_pool_pct=0.0,                          # buffer pool doesn't apply to emissions
    biogenic_treatment=BiogenicAccountingTreatment.SEQUESTRATION_TRACKED,
    removal_category=None,
    allowed_removal_types=(),
    verification_standards=(
        VerificationStandard.VCS,
        VerificationStandard.GOLD_STANDARD,
        VerificationStandard.ICVCM_CCP_APPROVED,
    ),
    reporting_frequency=ReportingFrequency.ANNUAL,
    is_active_removal=False,
    iluc_included=True,
    soc_tracked=True,
    direct_land_use_included=True,
)


# ---------------------------------------------------------------------------
# 2. Land Management Pack
# ---------------------------------------------------------------------------

GHG_LSR_LAND_MANAGEMENT = _build_pack(
    variant_name="lsr_land_management",
    display_name="GHG LSR — Land Management",
    description=(
        "Ongoing land-management emissions and removals on existing land: "
        "cropland management, grassland management, wetland management, "
        "and forest-management practices (fertiliser, tillage, cover crops, "
        "silvicultural regimes). Tracks both emissions and small-scale "
        "removals; distinct from active 'new' carbon removal projects."
    ),
    audit_template=(
        "LSR Land Management. Site: {site_id}. "
        "Practice: {management_practice}. Area: {area_ha} ha. "
        "Net flux: {net_flux_tco2e} tCO2e (positive = emission). "
        "SOC change: {soc_delta_tc} tC. Tier: {ipcc_tier}."
    ),
    biogenic=BiogenicTreatment.INCLUDED,
    market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    allowed_scopes=("1", "3"),
)
_LM_META = LSRPackMetadata(
    variant_name="lsr_land_management",
    permanence_class=PermanenceClass.MEDIUM,
    reversal_risk=ReversalRiskLevel.MEDIUM,
    buffer_pool_pct=DEFAULT_BUFFER_POOL[PermanenceClass.MEDIUM]
    * RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.MEDIUM],
    biogenic_treatment=BiogenicAccountingTreatment.SEQUESTRATION_TRACKED,
    removal_category=RemovalCategory.NATURE_BASED,
    allowed_removal_types=(
        RemovalType.SOIL_CARBON_SEQUESTRATION,
        RemovalType.FOREST_MANAGEMENT,
    ),
    verification_standards=(
        VerificationStandard.VCS,
        VerificationStandard.GOLD_STANDARD,
        VerificationStandard.CAR,
    ),
    reporting_frequency=ReportingFrequency.ANNUAL,
    is_active_removal=False,
    iluc_included=False,
    soc_tracked=True,
    direct_land_use_included=True,
)


# ---------------------------------------------------------------------------
# 3. Active Removals Pack
# ---------------------------------------------------------------------------

GHG_LSR_REMOVALS = _build_pack(
    variant_name="lsr_removals",
    display_name="GHG LSR — Active Removals",
    description=(
        "Active atmospheric removals: net-new carbon removed from the "
        "atmosphere via afforestation, reforestation, biochar, BECCS, "
        "DACCS, enhanced weathering, mineralisation, ocean alkalinity "
        "enhancement, peatland rewetting, and blue-carbon (mangrove / "
        "seagrass / saltmarsh) projects. Requires third-party MRV per "
        "VCS / Gold Standard / Puro.earth / Isometric and ICVCM-CCP "
        "alignment. Permanence + reversal-risk drive the buffer pool."
    ),
    audit_template=(
        "LSR Active Removal. Project: {project_id}. "
        "Removal type: {removal_type} ({removal_category}). "
        "Removed: {removed_tco2e} tCO2e. Buffer pool: {buffer_pct:.1%}. "
        "Net issuable: {net_issued_tco2e} tCO2e. "
        "Permanence: {permanence_class}. Reversal risk: {reversal_risk}. "
        "MRV standard: {verification_standard}."
    ),
    biogenic=BiogenicTreatment.INCLUDED,
    market_instruments=MarketInstrumentTreatment.REQUIRE_CERTIFICATE,
    allowed_scopes=("1", "3"),
)
_REMOVALS_META = LSRPackMetadata(
    variant_name="lsr_removals",
    permanence_class=PermanenceClass.MEDIUM,
    reversal_risk=ReversalRiskLevel.MEDIUM,
    buffer_pool_pct=DEFAULT_BUFFER_POOL[PermanenceClass.MEDIUM]
    * RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.MEDIUM],
    biogenic_treatment=BiogenicAccountingTreatment.SEQUESTRATION_TRACKED,
    removal_category=RemovalCategory.HYBRID,
    allowed_removal_types=(
        RemovalType.AFFORESTATION,
        RemovalType.REFORESTATION,
        RemovalType.PEATLAND_REWETTING,
        RemovalType.SOIL_CARBON_SEQUESTRATION,
        RemovalType.BLUE_CARBON_MANGROVE,
        RemovalType.BLUE_CARBON_SEAGRASS,
        RemovalType.BLUE_CARBON_SALTMARSH,
        RemovalType.BIOCHAR,
        RemovalType.BECCS,
        RemovalType.DACCS,
        RemovalType.ENHANCED_ROCK_WEATHERING,
        RemovalType.MINERAL_CARBONATION,
        RemovalType.OCEAN_ALKALINITY_ENHANCEMENT,
    ),
    verification_standards=(
        VerificationStandard.VCS,
        VerificationStandard.GOLD_STANDARD,
        VerificationStandard.PURO_EARTH,
        VerificationStandard.ISOMETRIC,
        VerificationStandard.CLIMEWORKS_VERIFIED,
        VerificationStandard.ICVCM_CCP_APPROVED,
    ),
    reporting_frequency=ReportingFrequency.ANNUAL,
    is_active_removal=True,
    iluc_included=True,
    soc_tracked=True,
    direct_land_use_included=True,
)


# ---------------------------------------------------------------------------
# 4. Carbon Storage Pack
# ---------------------------------------------------------------------------

GHG_LSR_STORAGE = _build_pack(
    variant_name="lsr_storage",
    display_name="GHG LSR — Carbon Storage",
    description=(
        "Durable storage of previously removed carbon: geological CO2 "
        "injection, long-lived wood products, embedded construction "
        "materials, mineralised carbon in concrete, and other storage "
        "pools. Distinct from active removal — tracks the stock lock "
        "rather than the net atmospheric flux."
    ),
    audit_template=(
        "LSR Carbon Storage. Asset: {asset_id}. "
        "Storage medium: {storage_medium}. "
        "Stored: {stored_tco2e} tCO2e. "
        "Expected lifetime: {lifetime_years} years. "
        "Permanence class: {permanence_class}. "
        "Reversal risk: {reversal_risk}. Leakage: {leakage_tco2e} tCO2e."
    ),
    biogenic=BiogenicTreatment.INCLUDED,
    market_instruments=MarketInstrumentTreatment.REQUIRE_CERTIFICATE,
    allowed_scopes=("1", "3"),
)
_STORAGE_META = LSRPackMetadata(
    variant_name="lsr_storage",
    permanence_class=PermanenceClass.LONG,
    reversal_risk=ReversalRiskLevel.LOW,
    buffer_pool_pct=DEFAULT_BUFFER_POOL[PermanenceClass.LONG]
    * RISK_BUFFER_MULTIPLIER[ReversalRiskLevel.LOW],
    biogenic_treatment=BiogenicAccountingTreatment.STORAGE_TRACKED,
    removal_category=RemovalCategory.TECHNOLOGY_BASED,
    allowed_removal_types=(
        RemovalType.BIOCHAR,
        RemovalType.MINERAL_CARBONATION,
        RemovalType.ENHANCED_ROCK_WEATHERING,
    ),
    verification_standards=(
        VerificationStandard.PURO_EARTH,
        VerificationStandard.ISOMETRIC,
        VerificationStandard.VCS,
    ),
    reporting_frequency=ReportingFrequency.ANNUAL,
    is_active_removal=False,
    iluc_included=False,
    soc_tracked=False,
    direct_land_use_included=False,
)


# ---------------------------------------------------------------------------
# Umbrella pack (back-compat) — MethodProfile.LAND_REMOVALS
# ---------------------------------------------------------------------------

LAND_REMOVALS = _build_pack(
    variant_name="lsr_umbrella",
    display_name="Land Sector & Removals (GHG Protocol LSR) — all variants",
    description=(
        "Umbrella LSR pack covering all four variants: "
        "lsr_land_use_emissions, lsr_land_management, lsr_removals, "
        "lsr_storage. Resolution engines should route to the specific "
        "variant via get_pack(variant_name)."
    ),
    audit_template=(
        "LSR umbrella. Activity: {activity}. Variant: {variant_name}. "
        "Permanence: {permanence_class}. Reversal risk: {reversal_risk}. "
        "GWP: {gwp_basis}."
    ),
    biogenic=BiogenicTreatment.INCLUDED,
    market_instruments=MarketInstrumentTreatment.REQUIRE_CERTIFICATE,
)


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


_ALL_VARIANTS: Tuple[Tuple[MethodPack, LSRPackMetadata], ...] = (
    (GHG_LSR_LAND_USE_EMISSIONS, _LUE_META),
    (GHG_LSR_LAND_MANAGEMENT, _LM_META),
    (GHG_LSR_REMOVALS, _REMOVALS_META),
    (GHG_LSR_STORAGE, _STORAGE_META),
)


def _register_all() -> None:
    for pack, meta in _ALL_VARIANTS:
        register_lsr_variant(meta.variant_name, pack, meta)
    register_pack(LAND_REMOVALS)


_register_all()


# ---------------------------------------------------------------------------
# Class wrappers exposed per GAP-9 spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LSRVariantHolder:
    """Wrapper exposing the underlying MethodPack + metadata by name."""

    variant_name: str

    @property
    def pack(self) -> MethodPack:
        return get_lsr_variant(self.variant_name)

    @property
    def metadata(self) -> LSRPackMetadata:
        return get_lsr_metadata(self.variant_name)


class GHGLSRLandUseEmissionsPack(_LSRVariantHolder):
    """GHG LSR — land-use emissions (deforestation, drainage, iLUC)."""

    def __init__(self) -> None:
        super().__init__(variant_name="lsr_land_use_emissions")


class GHGLSRLandManagementPack(_LSRVariantHolder):
    """GHG LSR — ongoing land management (cropland / grassland / forest)."""

    def __init__(self) -> None:
        super().__init__(variant_name="lsr_land_management")


class GHGLSRRemovalsPack(_LSRVariantHolder):
    """GHG LSR — active atmospheric removals (nature-based + technology)."""

    def __init__(self) -> None:
        super().__init__(variant_name="lsr_removals")


class GHGLSRStoragePack(_LSRVariantHolder):
    """GHG LSR — durable carbon storage (stock-based accounting)."""

    def __init__(self) -> None:
        super().__init__(variant_name="lsr_storage")


# ---------------------------------------------------------------------------
# Helper — buffer-pool computation
# ---------------------------------------------------------------------------


def compute_buffer_pool_pct(
    permanence: PermanenceClass, risk: ReversalRiskLevel
) -> float:
    """Compute required buffer-pool contribution (0.0-1.0).

    Example
    -------
    >>> compute_buffer_pool_pct(PermanenceClass.LONG, ReversalRiskLevel.LOW)
    0.1
    >>> compute_buffer_pool_pct(PermanenceClass.SHORT, ReversalRiskLevel.HIGH)
    0.7
    """
    base = DEFAULT_BUFFER_POOL[permanence]
    mult = RISK_BUFFER_MULTIPLIER[risk]
    return min(1.0, round(base * mult, 4))


# ---------------------------------------------------------------------------
# v0.2 sidecar metadata map (Wave 4-G scaffold).
# TODO(methodology-review): each entry's standards_alignment and
# reversal_risk_flag must be confirmed before the pack flips to certified.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LSRV02Metadata:
    """v0.2 scaffold metadata stored alongside the frozen MethodPack.

    Spec-driven fields (audit template §5). Not mutated at runtime.
    """

    pack_id: str
    standards_alignment: Tuple[str, ...]
    land_use_category: LandUseCategory
    sequestration_basis: SequestrationBasis
    removal_basis: Optional[RemovalBasis]
    permanence_class_v2: PermanenceClassV2
    reversal_risk_flag: bool
    biogenic_accounting_treatment: str   # "neutral" | "separate_report" | "include_co2"
    icvcm_ccp_required: bool = True
    deprecation_notice_days: int = 180


LSR_V02_METADATA: Dict[str, LSRV02Metadata] = {
    "lsr_land_use_emissions": LSRV02Metadata(
        pack_id="lsr_land_use_emissions",
        standards_alignment=(
            "GHG Protocol Land Sector and Removals Standard (2024)",
            "IPCC 2006 GL + 2019 Refinement",
        ),
        land_use_category=LandUseCategory.FOREST,        # primary (cross-category data allowed)
        sequestration_basis=SequestrationBasis.BIOMASS,
        removal_basis=None,                              # emissions, not removals
        permanence_class_v2=PermanenceClassV2.LONG_TERM_100YR,
        reversal_risk_flag=False,                        # emissions already realised
        biogenic_accounting_treatment="include_co2",
        icvcm_ccp_required=False,                        # applies only to project-based removals
    ),
    "lsr_land_management": LSRV02Metadata(
        pack_id="lsr_land_management",
        standards_alignment=(
            "GHG Protocol Land Sector and Removals Standard (2024)",
            "IPCC 2019 Refinement",
        ),
        land_use_category=LandUseCategory.CROPLAND,
        sequestration_basis=SequestrationBasis.SOIL,
        removal_basis=RemovalBasis.NATURE_BASED,
        permanence_class_v2=PermanenceClassV2.MEDIUM_25_100YR,
        reversal_risk_flag=True,
        biogenic_accounting_treatment="separate_report",
        icvcm_ccp_required=False,
    ),
    "lsr_removals": LSRV02Metadata(
        pack_id="lsr_removals",
        standards_alignment=(
            "GHG Protocol Land Sector and Removals Standard (2024)",
            "ICVCM Core Carbon Principles",
        ),
        land_use_category=LandUseCategory.FOREST,
        sequestration_basis=SequestrationBasis.BIOMASS,
        removal_basis=RemovalBasis.HYBRID,               # covers both nature + tech
        permanence_class_v2=PermanenceClassV2.MEDIUM_25_100YR,
        reversal_risk_flag=True,
        biogenic_accounting_treatment="separate_report",
        icvcm_ccp_required=True,                         # ICVCM-CCP enforced
    ),
    "lsr_storage": LSRV02Metadata(
        pack_id="lsr_storage",
        standards_alignment=(
            "GHG Protocol Land Sector and Removals Standard (2024)",
            "ICVCM Core Carbon Principles",
            "Puro.earth Supplier General Rules v2.0",
        ),
        land_use_category=LandUseCategory.OTHER,         # geologic / product
        sequestration_basis=SequestrationBasis.MINERALIZATION,
        removal_basis=RemovalBasis.TECHNOLOGY_BASED,
        permanence_class_v2=PermanenceClassV2.LONG_TERM_100YR,
        reversal_risk_flag=False,
        biogenic_accounting_treatment="include_co2",
        icvcm_ccp_required=True,
    ),
}


#: Per-variant changelog (MP12) — v0.1.0 baseline + v0.2.0 promotion.
LSR_CHANGELOG: Dict[str, Tuple[Dict[str, Any], ...]] = {
    variant: (
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                f"Initial LSR variant {variant} (MP9 scaffold)",
                "IPCC 2006 GL + 2019 Refinement selection rules",
                "Permanence + reversal-risk sidecar metadata",
            ],
            "impact": "none (new variant)",
            "migration_notes": "N/A — variant introduction.",
        },
        {
            "version": "0.2.0",
            "date": "2026-04-23",
            "changes": [
                "Added LandUseCategory / SequestrationBasis / RemovalBasis enums",
                "Replaced DEFAULT_FALLBACK with LSR_V02_FALLBACK_HIERARCHY "
                "(customer -> supplier -> project -> national, NO global default)",
                "cannot_resolve_action=RAISE_NO_SAFE_MATCH; "
                "global_default_tier_allowed=False",
                "ICVCM-CCP enforcement flag on project-based removal variants",
                "Added ICVCM_CCP reporting label",
                "deprecation_notice_days=180",
            ],
            "impact": (
                "Resolver now refuses tier-7 global default; unresolvable "
                "requests raise FactorCannotResolveSafelyError instead."
            ),
            "migration_notes": (
                "Callers that previously relied on silent global-default "
                "fall-through must switch to explicit region-specific lookups "
                "or populate project-level data. "
                "TODO(methodology-review) before promoting to `certified`."
            ),
        },
    )
    for variant in ("lsr_land_use_emissions", "lsr_land_management", "lsr_removals", "lsr_storage")
}


def get_lsr_v02_metadata(variant_name: str) -> LSRV02Metadata:
    """Retrieve v0.2 scaffold metadata for an LSR variant."""
    meta = LSR_V02_METADATA.get(variant_name)
    if meta is None:
        raise KeyError(
            "no v0.2 LSR metadata for %r; available: %s"
            % (variant_name, sorted(LSR_V02_METADATA))
        )
    return meta


__all__ = [
    # Enums
    "PermanenceClass",
    "ReversalRiskLevel",
    "BiogenicAccountingTreatment",
    "RemovalCategory",
    "RemovalType",
    "VerificationStandard",
    "ReportingFrequency",
    # Data classes
    "LSRPackMetadata",
    # Class wrappers
    "GHGLSRLandUseEmissionsPack",
    "GHGLSRLandManagementPack",
    "GHGLSRRemovalsPack",
    "GHGLSRStoragePack",
    # MethodPack instances
    "LAND_REMOVALS",
    "GHG_LSR_LAND_USE_EMISSIONS",
    "GHG_LSR_LAND_MANAGEMENT",
    "GHG_LSR_REMOVALS",
    "GHG_LSR_STORAGE",
    # Registry helpers
    "register_lsr_variant",
    "get_lsr_variant",
    "get_lsr_metadata",
    "list_lsr_variants",
    # Reference data
    "DEFAULT_BUFFER_POOL",
    "RISK_BUFFER_MULTIPLIER",
    "LSR_FALLBACK_HIERARCHY",
    # v0.2 (Wave 4-G)
    "LandUseCategory",
    "SequestrationBasis",
    "RemovalBasis",
    "PermanenceClassV2",
    "LSRV02Metadata",
    "LSR_V02_METADATA",
    "LSR_V02_FALLBACK_HIERARCHY",
    "LSR_CHANGELOG",
    "get_lsr_v02_metadata",
    "icvcm_ccp_compliant",
    # Utilities
    "compute_buffer_pool_pct",
]
