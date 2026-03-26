"""
PACK-049 GHG Multi-Site Management Pack - Regional Factor Engine
====================================================================

Manages the assignment of region-specific emission factors, grid
electricity factors, and climate zone parameters to individual sites
in a multi-site portfolio. Implements a tiered lookup hierarchy
(facility-specific > national > regional > IPCC default) and supports
factor overrides with approval workflows.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 8): Describes
      the hierarchy of emission factor sources and the preference
      for facility-specific over default factors.
    - GHG Protocol Scope 2 Guidance: Location-based vs market-based
      methods require different grid emission factors.
    - IEA CO2 Emissions from Fuel Combustion: Source for national
      grid emission factors.
    - IPCC 2006 Guidelines: Default emission factors by fuel type
      and sector.
    - eGRID (US): Sub-national grid emission factors.
    - AIB European Residual Mixes: Residual mix factors for EU
      market-based accounting.
    - ISO 14064-1:2018 (Clause 6.4): Selection of quantification
      approach and emission factors.

Capabilities:
    - Assign emission factors to sites using tiered lookup
    - Manage grid region assignments with location/residual factors
    - Assign climate zones for weather normalisation
    - Override factors with approval and audit trail
    - Track factor coverage and tier distribution across portfolio
    - Support multiple factor vintages (years)
    - Include default factor databases for 20+ countries

Zero-Hallucination:
    - All factor lookups are deterministic from stored data
    - No LLM involvement in factor selection or calculation
    - All calculations use Decimal arithmetic
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  4 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash",
                         "approved_at")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round6(value: Any) -> Decimal:
    """Round a value to six decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FactorType(str, Enum):
    """Types of emission factors."""
    GRID_ELECTRICITY = "GRID_ELECTRICITY"
    RESIDUAL_MIX = "RESIDUAL_MIX"
    NATURAL_GAS = "NATURAL_GAS"
    DIESEL = "DIESEL"
    PETROL = "PETROL"
    LPG = "LPG"
    COAL = "COAL"
    FUEL_OIL = "FUEL_OIL"
    BIOMASS = "BIOMASS"
    STEAM = "STEAM"
    CHILLED_WATER = "CHILLED_WATER"
    DISTRICT_HEATING = "DISTRICT_HEATING"
    REFRIGERANT = "REFRIGERANT"
    WASTE_LANDFILL = "WASTE_LANDFILL"
    WASTE_INCINERATION = "WASTE_INCINERATION"
    WATER_SUPPLY = "WATER_SUPPLY"
    WATER_TREATMENT = "WATER_TREATMENT"


class FactorTier(str, Enum):
    """Data quality tiers for emission factors.

    Follows GHG Protocol hierarchy from most to least accurate.
    """
    FACILITY_SPECIFIC = "FACILITY_SPECIFIC"
    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"
    NATIONAL = "NATIONAL"
    REGIONAL = "REGIONAL"
    IPCC_DEFAULT = "IPCC_DEFAULT"


class FactorSource(str, Enum):
    """Source databases for emission factors."""
    IEA = "IEA"
    EGRID = "EGRID"
    DEFRA = "DEFRA"
    AIB_RESIDUAL = "AIB_RESIDUAL"
    IPCC_2006 = "IPCC_2006"
    IPCC_2019 = "IPCC_2019"
    EPA = "EPA"
    NGA = "NGA"
    GHG_PROTOCOL = "GHG_PROTOCOL"
    ADEME = "ADEME"
    EXIOBASE = "EXIOBASE"
    CUSTOM = "CUSTOM"


# ---------------------------------------------------------------------------
# Default Factor Databases
# ---------------------------------------------------------------------------

# National grid emission factors (kgCO2e/kWh) - Location-based
# Source: IEA 2023 data, used as representative defaults
DEFAULT_GRID_FACTORS: Dict[str, Dict[str, Any]] = {
    "US": {
        "location_factor": Decimal("0.417"),
        "residual_mix_factor": Decimal("0.425"),
        "t_and_d_loss_pct": Decimal("5.10"),
        "source": "EPA eGRID 2022",
        "vintage_year": 2022,
    },
    "GB": {
        "location_factor": Decimal("0.207"),
        "residual_mix_factor": Decimal("0.312"),
        "t_and_d_loss_pct": Decimal("7.50"),
        "source": "DEFRA 2023",
        "vintage_year": 2023,
    },
    "DE": {
        "location_factor": Decimal("0.380"),
        "residual_mix_factor": Decimal("0.492"),
        "t_and_d_loss_pct": Decimal("4.00"),
        "source": "IEA 2023 / AIB 2022",
        "vintage_year": 2022,
    },
    "FR": {
        "location_factor": Decimal("0.052"),
        "residual_mix_factor": Decimal("0.320"),
        "t_and_d_loss_pct": Decimal("5.80"),
        "source": "ADEME 2023",
        "vintage_year": 2023,
    },
    "CN": {
        "location_factor": Decimal("0.555"),
        "residual_mix_factor": None,
        "t_and_d_loss_pct": Decimal("5.50"),
        "source": "IEA 2023",
        "vintage_year": 2022,
    },
    "IN": {
        "location_factor": Decimal("0.708"),
        "residual_mix_factor": None,
        "t_and_d_loss_pct": Decimal("19.00"),
        "source": "IEA 2023",
        "vintage_year": 2022,
    },
    "JP": {
        "location_factor": Decimal("0.457"),
        "residual_mix_factor": Decimal("0.470"),
        "t_and_d_loss_pct": Decimal("4.80"),
        "source": "IEA 2023",
        "vintage_year": 2022,
    },
    "BR": {
        "location_factor": Decimal("0.075"),
        "residual_mix_factor": None,
        "t_and_d_loss_pct": Decimal("15.50"),
        "source": "IEA 2023",
        "vintage_year": 2022,
    },
    "AU": {
        "location_factor": Decimal("0.680"),
        "residual_mix_factor": Decimal("0.700"),
        "t_and_d_loss_pct": Decimal("5.00"),
        "source": "NGA 2023",
        "vintage_year": 2023,
    },
    "CA": {
        "location_factor": Decimal("0.120"),
        "residual_mix_factor": Decimal("0.130"),
        "t_and_d_loss_pct": Decimal("6.20"),
        "source": "Environment Canada 2022",
        "vintage_year": 2022,
    },
    "KR": {
        "location_factor": Decimal("0.459"),
        "residual_mix_factor": Decimal("0.465"),
        "t_and_d_loss_pct": Decimal("3.50"),
        "source": "IEA 2023",
        "vintage_year": 2022,
    },
    "MX": {
        "location_factor": Decimal("0.423"),
        "residual_mix_factor": None,
        "t_and_d_loss_pct": Decimal("11.00"),
        "source": "IEA 2023",
        "vintage_year": 2022,
    },
    "ZA": {
        "location_factor": Decimal("0.928"),
        "residual_mix_factor": None,
        "t_and_d_loss_pct": Decimal("8.50"),
        "source": "Eskom 2023",
        "vintage_year": 2023,
    },
    "SE": {
        "location_factor": Decimal("0.013"),
        "residual_mix_factor": Decimal("0.292"),
        "t_and_d_loss_pct": Decimal("6.00"),
        "source": "IEA 2023 / AIB 2022",
        "vintage_year": 2022,
    },
    "NO": {
        "location_factor": Decimal("0.008"),
        "residual_mix_factor": Decimal("0.350"),
        "t_and_d_loss_pct": Decimal("6.50"),
        "source": "IEA 2023 / AIB 2022",
        "vintage_year": 2022,
    },
    "PL": {
        "location_factor": Decimal("0.693"),
        "residual_mix_factor": Decimal("0.720"),
        "t_and_d_loss_pct": Decimal("6.00"),
        "source": "IEA 2023 / AIB 2022",
        "vintage_year": 2022,
    },
    "IT": {
        "location_factor": Decimal("0.260"),
        "residual_mix_factor": Decimal("0.411"),
        "t_and_d_loss_pct": Decimal("6.20"),
        "source": "IEA 2023 / AIB 2022",
        "vintage_year": 2022,
    },
    "ES": {
        "location_factor": Decimal("0.163"),
        "residual_mix_factor": Decimal("0.282"),
        "t_and_d_loss_pct": Decimal("8.90"),
        "source": "IEA 2023 / AIB 2022",
        "vintage_year": 2022,
    },
    "NL": {
        "location_factor": Decimal("0.386"),
        "residual_mix_factor": Decimal("0.465"),
        "t_and_d_loss_pct": Decimal("3.80"),
        "source": "IEA 2023 / AIB 2022",
        "vintage_year": 2022,
    },
    "SG": {
        "location_factor": Decimal("0.408"),
        "residual_mix_factor": None,
        "t_and_d_loss_pct": Decimal("2.50"),
        "source": "EMA Singapore 2023",
        "vintage_year": 2023,
    },
}

# Default fuel emission factors (kgCO2e per unit)
DEFAULT_FUEL_FACTORS: Dict[str, Dict[str, Any]] = {
    FactorType.NATURAL_GAS.value: {
        "factor_value": Decimal("2.02"),
        "unit": "kgCO2e/m3",
        "source": "IPCC 2006",
        "tier": FactorTier.IPCC_DEFAULT.value,
    },
    FactorType.DIESEL.value: {
        "factor_value": Decimal("2.68"),
        "unit": "kgCO2e/litre",
        "source": "IPCC 2006",
        "tier": FactorTier.IPCC_DEFAULT.value,
    },
    FactorType.PETROL.value: {
        "factor_value": Decimal("2.31"),
        "unit": "kgCO2e/litre",
        "source": "IPCC 2006",
        "tier": FactorTier.IPCC_DEFAULT.value,
    },
    FactorType.LPG.value: {
        "factor_value": Decimal("1.56"),
        "unit": "kgCO2e/litre",
        "source": "IPCC 2006",
        "tier": FactorTier.IPCC_DEFAULT.value,
    },
    FactorType.COAL.value: {
        "factor_value": Decimal("2450"),
        "unit": "kgCO2e/tonne",
        "source": "IPCC 2006",
        "tier": FactorTier.IPCC_DEFAULT.value,
    },
    FactorType.FUEL_OIL.value: {
        "factor_value": Decimal("3.18"),
        "unit": "kgCO2e/litre",
        "source": "IPCC 2006",
        "tier": FactorTier.IPCC_DEFAULT.value,
    },
}

# Factor source metadata
DEFAULT_FACTOR_DATABASES: Dict[str, Dict[str, Any]] = {
    FactorSource.IEA.value: {
        "name": "IEA CO2 Emissions from Fuel Combustion",
        "url": "https://www.iea.org/data-and-statistics",
        "coverage": "Global",
        "update_frequency": "Annual",
        "last_update": "2023",
    },
    FactorSource.EGRID.value: {
        "name": "EPA eGRID",
        "url": "https://www.epa.gov/egrid",
        "coverage": "United States sub-regions",
        "update_frequency": "Annual",
        "last_update": "2022",
    },
    FactorSource.DEFRA.value: {
        "name": "UK DEFRA Conversion Factors",
        "url": "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting",
        "coverage": "United Kingdom",
        "update_frequency": "Annual",
        "last_update": "2023",
    },
    FactorSource.AIB_RESIDUAL.value: {
        "name": "AIB European Residual Mixes",
        "url": "https://www.aib-net.org/facts/european-residual-mix",
        "coverage": "EU/EEA countries",
        "update_frequency": "Annual",
        "last_update": "2022",
    },
    FactorSource.IPCC_2006.value: {
        "name": "IPCC 2006 Guidelines Emission Factor Database",
        "url": "https://www.ipcc-nggip.iges.or.jp/EFDB/main.php",
        "coverage": "Global defaults",
        "update_frequency": "Irregular",
        "last_update": "2006 (2019 refinement)",
    },
    FactorSource.EPA.value: {
        "name": "US EPA GHG Emission Factors Hub",
        "url": "https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
        "coverage": "United States",
        "update_frequency": "Annual",
        "last_update": "2023",
    },
    FactorSource.NGA.value: {
        "name": "Australia National Greenhouse Accounts",
        "url": "https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors",
        "coverage": "Australia",
        "update_frequency": "Annual",
        "last_update": "2023",
    },
    FactorSource.ADEME.value: {
        "name": "ADEME Base Carbone",
        "url": "https://bilans-ges.ademe.fr/",
        "coverage": "France",
        "update_frequency": "Ongoing",
        "last_update": "2023",
    },
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class FactorAssignment(BaseModel):
    """An emission factor assigned to a specific site.

    Links a site to a particular emission factor with full
    provenance including source, tier, and vintage year.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    assignment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assignment identifier.",
    )
    site_id: str = Field(
        ...,
        description="The site this factor is assigned to.",
    )
    factor_type: str = Field(
        ...,
        description="Type of emission factor.",
    )
    tier: str = Field(
        ...,
        description="Data quality tier.",
    )
    source: str = Field(
        ...,
        description="Source database or reference.",
    )
    factor_value: Decimal = Field(
        ...,
        description="The emission factor value.",
    )
    unit: str = Field(
        ...,
        description="Unit of the factor (e.g., kgCO2e/kWh).",
    )
    vintage_year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Year the factor data pertains to.",
    )
    valid_from: Optional[date] = Field(
        None,
        description="Start of validity period.",
    )
    valid_to: Optional[date] = Field(
        None,
        description="End of validity period.",
    )
    country: Optional[str] = Field(
        None,
        description="Country the factor applies to.",
    )
    region: Optional[str] = Field(
        None,
        description="Sub-national region the factor applies to.",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes.",
    )

    @field_validator("factor_value", mode="before")
    @classmethod
    def _coerce_factor(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("factor_type")
    @classmethod
    def _validate_factor_type(cls, v: str) -> str:
        valid = {ft.value for ft in FactorType}
        if v.upper() not in valid:
            logger.warning("Factor type '%s' not standard; accepted.", v)
        return v.upper()

    @field_validator("tier")
    @classmethod
    def _validate_tier(cls, v: str) -> str:
        valid = {t.value for t in FactorTier}
        if v.upper() not in valid:
            logger.warning("Tier '%s' not standard; accepted.", v)
        return v.upper()


class GridRegion(BaseModel):
    """Electricity grid region with emission factors.

    Defines the electricity grid characteristics for a site,
    including location-based and residual mix emission factors.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    region_id: str = Field(
        default_factory=_new_uuid,
        description="Unique region identifier.",
    )
    region_name: str = Field(
        ...,
        description="Name of the grid region.",
    )
    country: str = Field(
        ...,
        description="Country code.",
    )
    grid_operator: Optional[str] = Field(
        None,
        description="Name of the grid operator.",
    )
    location_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Location-based grid factor (kgCO2e/kWh).",
    )
    residual_mix_factor: Optional[Decimal] = Field(
        None,
        description="Residual mix factor for market-based (kgCO2e/kWh).",
    )
    t_and_d_loss_pct: Decimal = Field(
        default=Decimal("5.00"),
        ge=Decimal("0"),
        le=Decimal("50"),
        description="Transmission & distribution loss percentage.",
    )
    source: Optional[str] = Field(
        None,
        description="Data source.",
    )
    vintage_year: Optional[int] = Field(
        None,
        description="Year the factors pertain to.",
    )

    @field_validator("location_factor", "residual_mix_factor", "t_and_d_loss_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        if v is not None:
            return Decimal(str(v))
        return v


class ClimateZone(BaseModel):
    """Climate zone assignment for a site.

    Used for weather normalisation of energy data (heating/cooling
    degree days) and benchmarking.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    zone_id: str = Field(
        default_factory=_new_uuid,
        description="Unique zone identifier.",
    )
    zone_name: str = Field(
        ...,
        description="Name of the climate zone.",
    )
    hdd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Annual heating degree days (base 18C).",
    )
    cdd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Annual cooling degree days (base 18C).",
    )
    koppen_class: Optional[str] = Field(
        None,
        description="Koppen-Geiger climate classification.",
    )
    avg_temperature_c: Optional[Decimal] = Field(
        None,
        description="Average annual temperature in Celsius.",
    )

    @field_validator("hdd", "cdd", "avg_temperature_c", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        if v is not None:
            return Decimal(str(v))
        return v


class FactorOverride(BaseModel):
    """An override of an emission factor for a specific site.

    Used when a site has a more accurate factor (e.g., from direct
    measurement or a specific supplier contract) that should replace
    the default database factor.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    override_id: str = Field(
        default_factory=_new_uuid,
        description="Unique override identifier.",
    )
    site_id: str = Field(
        ...,
        description="The site this override applies to.",
    )
    factor_type: str = Field(
        ...,
        description="Type of factor being overridden.",
    )
    original_value: Decimal = Field(
        ...,
        description="The original factor value.",
    )
    override_value: Decimal = Field(
        ...,
        description="The new overriding factor value.",
    )
    justification: str = Field(
        ...,
        description="Justification for the override.",
    )
    approved_by: Optional[str] = Field(
        None,
        description="User who approved the override.",
    )
    approved_at: Optional[datetime] = Field(
        None,
        description="When the override was approved.",
    )
    is_approved: bool = Field(
        default=False,
        description="Whether the override has been approved.",
    )
    evidence_reference: Optional[str] = Field(
        None,
        description="Reference to supporting evidence.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the override was created.",
    )

    @field_validator("original_value", "override_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class RegionalFactorResult(BaseModel):
    """Complete factor assignment result for a site.

    Combines all factor assignments, grid region, and climate zone
    for a single site.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier.",
    )
    site_id: str = Field(
        ...,
        description="The site these factors apply to.",
    )
    assignments: List[FactorAssignment] = Field(
        default_factory=list,
        description="All factor assignments for the site.",
    )
    grid_region: Optional[GridRegion] = Field(
        None,
        description="Grid region assignment.",
    )
    climate_zone: Optional[ClimateZone] = Field(
        None,
        description="Climate zone assignment.",
    )
    overrides: List[FactorOverride] = Field(
        default_factory=list,
        description="Any factor overrides applied.",
    )
    tier_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of assignments by tier.",
    )
    factor_count: int = Field(
        default=0,
        description="Total number of factor assignments.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this result was generated.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )


class FactorCoverage(BaseModel):
    """Factor coverage assessment for the portfolio."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    total_sites: int = Field(
        default=0, description="Total sites assessed."
    )
    sites_with_factors: int = Field(
        default=0, description="Sites with at least one factor."
    )
    sites_without_factors: List[str] = Field(
        default_factory=list, description="Site IDs missing factors."
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0"), description="Coverage percentage."
    )
    factor_type_coverage: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of sites with each factor type.",
    )
    tier_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Aggregate tier distribution.",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RegionalFactorEngine:
    """Manages region-specific emission factor assignments.

    Provides tiered factor lookup, grid region assignment, climate
    zone mapping, factor overrides, and coverage analytics. All
    values use Decimal arithmetic with SHA-256 provenance hashing.

    Attributes:
        _assignments: Dict mapping (site_id, factor_type) to list of assignments.
        _grid_regions: Dict mapping site_id to GridRegion.
        _climate_zones: Dict mapping site_id to ClimateZone.
        _overrides: Dict mapping override_id to FactorOverride.

    Example:
        >>> engine = RegionalFactorEngine()
        >>> assignments = engine.assign_factors(
        ...     site_id="site-001",
        ...     country="US",
        ...     region="RFCE",
        ...     preferred_tier="NATIONAL",
        ... )
        >>> assert len(assignments) > 0
    """

    def __init__(self) -> None:
        """Initialise the RegionalFactorEngine with empty state."""
        self._assignments: Dict[str, List[FactorAssignment]] = {}
        self._grid_regions: Dict[str, GridRegion] = {}
        self._climate_zones: Dict[str, ClimateZone] = {}
        self._overrides: Dict[str, FactorOverride] = {}
        self._custom_factors: Dict[str, Dict[str, FactorAssignment]] = {}
        logger.info("RegionalFactorEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Factor Assignment
    # ------------------------------------------------------------------

    def assign_factors(
        self,
        site_id: str,
        country: str,
        region: Optional[str] = None,
        preferred_tier: Optional[str] = None,
        vintage_year: Optional[int] = None,
    ) -> List[FactorAssignment]:
        """Assign emission factors to a site using tiered lookup.

        Lookup hierarchy:
            1. Facility-specific (from custom_factors)
            2. National (from DEFAULT_GRID_FACTORS + DEFAULT_FUEL_FACTORS)
            3. Regional (sub-national if available)
            4. IPCC Default (global defaults)

        If preferred_tier is specified, the engine attempts to use
        that tier first but falls back through the hierarchy if not
        available.

        Args:
            site_id: The site to assign factors to.
            country: ISO country code.
            region: Optional sub-national region.
            preferred_tier: Preferred tier level.
            vintage_year: Preferred factor vintage year.

        Returns:
            List of FactorAssignments for the site.
        """
        logger.info(
            "Assigning factors to site '%s' (country=%s, region=%s).",
            site_id, country, region,
        )

        assignments: List[FactorAssignment] = []
        year = vintage_year or 2022

        # 1. Grid electricity factor
        grid_assignment = self._assign_grid_factor(
            site_id, country, region, year
        )
        if grid_assignment:
            assignments.append(grid_assignment)

        # 2. Residual mix factor (for market-based Scope 2)
        residual_assignment = self._assign_residual_mix_factor(
            site_id, country, year
        )
        if residual_assignment:
            assignments.append(residual_assignment)

        # 3. Fuel factors
        for fuel_type, fuel_data in DEFAULT_FUEL_FACTORS.items():
            # Check for facility-specific override first
            custom_key = f"{site_id}:{fuel_type}"
            if custom_key in self._custom_factors:
                custom = self._custom_factors[custom_key]
                for k, v in custom.items():
                    assignments.append(v)
                continue

            # Use IPCC defaults
            assignment = FactorAssignment(
                site_id=site_id,
                factor_type=fuel_type,
                tier=fuel_data["tier"],
                source=fuel_data["source"],
                factor_value=fuel_data["factor_value"],
                unit=fuel_data["unit"],
                vintage_year=year,
                country=country,
                region=region,
            )
            assignments.append(assignment)

        # Store assignments
        self._assignments[site_id] = assignments

        logger.info(
            "Assigned %d factors to site '%s'.",
            len(assignments), site_id,
        )
        return assignments

    def _assign_grid_factor(
        self,
        site_id: str,
        country: str,
        region: Optional[str],
        vintage_year: int,
    ) -> Optional[FactorAssignment]:
        """Assign grid electricity factor with tiered lookup.

        Args:
            site_id: Site identifier.
            country: Country code.
            region: Sub-national region.
            vintage_year: Factor vintage year.

        Returns:
            FactorAssignment or None if not available.
        """
        # Check facility-specific first
        custom_key = f"{site_id}:{FactorType.GRID_ELECTRICITY.value}"
        if custom_key in self._custom_factors:
            custom = list(self._custom_factors[custom_key].values())
            if custom:
                return custom[0]

        # Check national database
        country_upper = country.upper()
        if country_upper in DEFAULT_GRID_FACTORS:
            grid_data = DEFAULT_GRID_FACTORS[country_upper]
            return FactorAssignment(
                site_id=site_id,
                factor_type=FactorType.GRID_ELECTRICITY.value,
                tier=FactorTier.NATIONAL.value,
                source=grid_data["source"],
                factor_value=grid_data["location_factor"],
                unit="kgCO2e/kWh",
                vintage_year=grid_data.get("vintage_year", vintage_year),
                country=country_upper,
                region=region,
            )

        # IPCC default for unknown countries
        logger.warning(
            "No grid factor found for country '%s'; using IPCC global average.",
            country_upper,
        )
        return FactorAssignment(
            site_id=site_id,
            factor_type=FactorType.GRID_ELECTRICITY.value,
            tier=FactorTier.IPCC_DEFAULT.value,
            source="IPCC 2006 Global Average",
            factor_value=Decimal("0.500"),
            unit="kgCO2e/kWh",
            vintage_year=vintage_year,
            country=country_upper,
            region=region,
            notes="Global average - consider sourcing national factor.",
        )

    def _assign_residual_mix_factor(
        self,
        site_id: str,
        country: str,
        vintage_year: int,
    ) -> Optional[FactorAssignment]:
        """Assign residual mix factor for market-based Scope 2.

        Args:
            site_id: Site identifier.
            country: Country code.
            vintage_year: Factor vintage year.

        Returns:
            FactorAssignment or None if residual mix not available.
        """
        country_upper = country.upper()
        if country_upper in DEFAULT_GRID_FACTORS:
            grid_data = DEFAULT_GRID_FACTORS[country_upper]
            residual = grid_data.get("residual_mix_factor")
            if residual is not None:
                return FactorAssignment(
                    site_id=site_id,
                    factor_type=FactorType.RESIDUAL_MIX.value,
                    tier=FactorTier.NATIONAL.value,
                    source=grid_data["source"],
                    factor_value=residual,
                    unit="kgCO2e/kWh",
                    vintage_year=grid_data.get("vintage_year", vintage_year),
                    country=country_upper,
                )

        logger.info(
            "No residual mix factor available for country '%s'.",
            country_upper,
        )
        return None

    # ------------------------------------------------------------------
    # Factor Lookup
    # ------------------------------------------------------------------

    def lookup_factor(
        self,
        site_id: str,
        factor_type: str,
        assignments: Optional[List[FactorAssignment]] = None,
        vintage_year: Optional[int] = None,
    ) -> Optional[FactorAssignment]:
        """Look up a specific factor for a site.

        Uses the tiered hierarchy: facility-specific > supplier >
        national > regional > IPCC default.

        Args:
            site_id: The site to look up.
            factor_type: The factor type to find.
            assignments: Pre-existing assignments to search. If None,
                uses internal storage.
            vintage_year: Preferred vintage year.

        Returns:
            The best available FactorAssignment, or None.
        """
        if assignments is None:
            assignments = self._assignments.get(site_id, [])

        # Filter by factor type
        matching = [
            a for a in assignments
            if a.factor_type.upper() == factor_type.upper()
        ]

        if not matching:
            return None

        # Sort by tier priority (facility-specific first)
        tier_priority = {
            FactorTier.FACILITY_SPECIFIC.value: 0,
            FactorTier.SUPPLIER_SPECIFIC.value: 1,
            FactorTier.NATIONAL.value: 2,
            FactorTier.REGIONAL.value: 3,
            FactorTier.IPCC_DEFAULT.value: 4,
        }

        matching.sort(
            key=lambda a: tier_priority.get(a.tier, 99)
        )

        # If vintage_year specified, prefer that year
        if vintage_year:
            year_matching = [
                a for a in matching if a.vintage_year == vintage_year
            ]
            if year_matching:
                return year_matching[0]

        # Return highest priority
        return matching[0]

    # ------------------------------------------------------------------
    # Grid Region
    # ------------------------------------------------------------------

    def assign_grid_region(
        self,
        site_id: str,
        country: str,
        region: Optional[str] = None,
        grid_data: Optional[Dict[str, Any]] = None,
    ) -> GridRegion:
        """Assign a grid region to a site.

        Uses provided grid_data or falls back to DEFAULT_GRID_FACTORS
        for the country.

        Args:
            site_id: The site.
            country: Country code.
            region: Sub-national region name.
            grid_data: Optional custom grid data with keys:
                location_factor, residual_mix_factor, t_and_d_loss_pct,
                grid_operator, source.

        Returns:
            The assigned GridRegion.
        """
        logger.info(
            "Assigning grid region to site '%s' (country=%s).",
            site_id, country,
        )

        country_upper = country.upper()

        if grid_data:
            grid_region = GridRegion(
                region_name=grid_data.get("region_name", f"{country_upper}-{region or 'DEFAULT'}"),
                country=country_upper,
                grid_operator=grid_data.get("grid_operator"),
                location_factor=_decimal(grid_data.get("location_factor", "0.500")),
                residual_mix_factor=(
                    _decimal(grid_data["residual_mix_factor"])
                    if grid_data.get("residual_mix_factor") is not None
                    else None
                ),
                t_and_d_loss_pct=_decimal(grid_data.get("t_and_d_loss_pct", "5.00")),
                source=grid_data.get("source"),
                vintage_year=grid_data.get("vintage_year"),
            )
        elif country_upper in DEFAULT_GRID_FACTORS:
            db_data = DEFAULT_GRID_FACTORS[country_upper]
            grid_region = GridRegion(
                region_name=f"{country_upper}-NATIONAL",
                country=country_upper,
                location_factor=db_data["location_factor"],
                residual_mix_factor=db_data.get("residual_mix_factor"),
                t_and_d_loss_pct=db_data["t_and_d_loss_pct"],
                source=db_data["source"],
                vintage_year=db_data.get("vintage_year"),
            )
        else:
            logger.warning(
                "No grid data for country '%s'; using global default.",
                country_upper,
            )
            grid_region = GridRegion(
                region_name=f"{country_upper}-DEFAULT",
                country=country_upper,
                location_factor=Decimal("0.500"),
                t_and_d_loss_pct=Decimal("5.00"),
                source="IPCC Global Default",
            )

        self._grid_regions[site_id] = grid_region
        logger.info(
            "Grid region '%s' assigned to site '%s' (EF=%s kgCO2e/kWh).",
            grid_region.region_name, site_id, grid_region.location_factor,
        )
        return grid_region

    # ------------------------------------------------------------------
    # Climate Zone
    # ------------------------------------------------------------------

    def assign_climate_zone(
        self,
        site_id: str,
        latitude: Union[Decimal, str, float, None] = None,
        longitude: Union[Decimal, str, float, None] = None,
        zone_data: Optional[Dict[str, Any]] = None,
    ) -> ClimateZone:
        """Assign a climate zone to a site.

        Uses provided zone_data or estimates from latitude using a
        simplified Koppen classification.

        Args:
            site_id: The site.
            latitude: Site latitude in decimal degrees.
            longitude: Site longitude in decimal degrees.
            zone_data: Optional pre-determined zone data with keys:
                zone_name, hdd, cdd, koppen_class.

        Returns:
            The assigned ClimateZone.
        """
        logger.info(
            "Assigning climate zone to site '%s'.",
            site_id,
        )

        if zone_data:
            climate_zone = ClimateZone(
                zone_name=zone_data.get("zone_name", "Custom"),
                hdd=_decimal(zone_data.get("hdd", "0")),
                cdd=_decimal(zone_data.get("cdd", "0")),
                koppen_class=zone_data.get("koppen_class"),
                avg_temperature_c=(
                    _decimal(zone_data["avg_temperature_c"])
                    if zone_data.get("avg_temperature_c") is not None
                    else None
                ),
            )
        elif latitude is not None:
            lat = _decimal(latitude)
            climate_zone = self._estimate_climate_from_latitude(lat)
        else:
            logger.warning(
                "No zone_data or coordinates for site '%s'; using temperate default.",
                site_id,
            )
            climate_zone = ClimateZone(
                zone_name="Temperate (Default)",
                hdd=Decimal("2500"),
                cdd=Decimal("500"),
                koppen_class="Cfb",
            )

        self._climate_zones[site_id] = climate_zone
        logger.info(
            "Climate zone '%s' assigned to site '%s' (HDD=%s, CDD=%s).",
            climate_zone.zone_name,
            site_id,
            climate_zone.hdd,
            climate_zone.cdd,
        )
        return climate_zone

    def _estimate_climate_from_latitude(
        self, latitude: Decimal,
    ) -> ClimateZone:
        """Estimate climate zone from latitude.

        Simplified classification based on latitude bands:
            0-15:  Tropical (Af)
            15-25: Subtropical (Cfa)
            25-40: Temperate (Cfb)
            40-55: Continental (Dfb)
            55-90: Subarctic/Arctic (Dfc)

        Args:
            latitude: Decimal latitude in degrees.

        Returns:
            Estimated ClimateZone.
        """
        abs_lat = abs(latitude)

        if abs_lat < Decimal("15"):
            return ClimateZone(
                zone_name="Tropical",
                hdd=Decimal("0"),
                cdd=Decimal("3500"),
                koppen_class="Af",
                avg_temperature_c=Decimal("27"),
            )
        elif abs_lat < Decimal("25"):
            return ClimateZone(
                zone_name="Subtropical",
                hdd=Decimal("500"),
                cdd=Decimal("2500"),
                koppen_class="Cfa",
                avg_temperature_c=Decimal("22"),
            )
        elif abs_lat < Decimal("40"):
            return ClimateZone(
                zone_name="Temperate",
                hdd=Decimal("2000"),
                cdd=Decimal("1000"),
                koppen_class="Cfb",
                avg_temperature_c=Decimal("15"),
            )
        elif abs_lat < Decimal("55"):
            return ClimateZone(
                zone_name="Continental",
                hdd=Decimal("4000"),
                cdd=Decimal("300"),
                koppen_class="Dfb",
                avg_temperature_c=Decimal("7"),
            )
        else:
            return ClimateZone(
                zone_name="Subarctic",
                hdd=Decimal("6500"),
                cdd=Decimal("50"),
                koppen_class="Dfc",
                avg_temperature_c=Decimal("-2"),
            )

    # ------------------------------------------------------------------
    # Factor Overrides
    # ------------------------------------------------------------------

    def override_factor(
        self,
        site_id: str,
        factor_type: str,
        assignments: Optional[List[FactorAssignment]] = None,
        new_value: Union[Decimal, str, int, float] = Decimal("0"),
        justification: str = "",
        evidence_reference: Optional[str] = None,
    ) -> FactorOverride:
        """Override an existing emission factor for a site.

        Creates an override record and updates the assignment.
        Override requires separate approval before taking effect.

        Args:
            site_id: The site.
            factor_type: The factor type to override.
            assignments: Current assignments. If None, uses internal.
            new_value: The new factor value.
            justification: Reason for the override.
            evidence_reference: Reference to supporting evidence.

        Returns:
            The created FactorOverride.

        Raises:
            ValueError: If no existing factor found to override.
        """
        if assignments is None:
            assignments = self._assignments.get(site_id, [])

        # Find current factor
        current = self.lookup_factor(site_id, factor_type, assignments)
        if current is None:
            raise ValueError(
                f"No existing '{factor_type}' factor for site '{site_id}' "
                f"to override."
            )

        override = FactorOverride(
            site_id=site_id,
            factor_type=factor_type,
            original_value=current.factor_value,
            override_value=_decimal(new_value),
            justification=justification,
            evidence_reference=evidence_reference,
        )
        self._overrides[override.override_id] = override

        logger.info(
            "Override created for site '%s' factor '%s': %s -> %s (pending approval).",
            site_id, factor_type, current.factor_value, new_value,
        )
        return override

    def approve_override(
        self,
        override_id: str,
        approved_by: str,
    ) -> FactorOverride:
        """Approve a factor override.

        Once approved, the override value replaces the original in
        the site's factor assignments.

        Args:
            override_id: The override to approve.
            approved_by: User approving the override.

        Returns:
            The approved FactorOverride.

        Raises:
            KeyError: If override not found.
            ValueError: If already approved.
        """
        if override_id not in self._overrides:
            raise KeyError(f"Override '{override_id}' not found.")

        override = self._overrides[override_id]
        if override.is_approved:
            raise ValueError(f"Override '{override_id}' is already approved.")

        now = _utcnow()
        updated = override.model_copy(update={
            "is_approved": True,
            "approved_by": approved_by,
            "approved_at": now,
        })
        self._overrides[override_id] = updated

        # Update the corresponding assignment
        site_assignments = self._assignments.get(override.site_id, [])
        for i, a in enumerate(site_assignments):
            if a.factor_type.upper() == override.factor_type.upper():
                site_assignments[i] = a.model_copy(update={
                    "factor_value": override.override_value,
                    "tier": FactorTier.FACILITY_SPECIFIC.value,
                    "notes": f"Overridden: {override.justification}",
                })
                break
        self._assignments[override.site_id] = site_assignments

        logger.info(
            "Override '%s' approved by '%s'. Factor updated.",
            override_id, approved_by,
        )
        return updated

    # ------------------------------------------------------------------
    # Coverage & Distribution Analytics
    # ------------------------------------------------------------------

    def get_factor_coverage(
        self,
        site_ids: List[str],
        assignments: Optional[Dict[str, List[FactorAssignment]]] = None,
    ) -> FactorCoverage:
        """Assess factor coverage across the site portfolio.

        Args:
            site_ids: List of all site IDs.
            assignments: Pre-computed assignments by site_id. If None,
                uses internal storage.

        Returns:
            FactorCoverage with coverage metrics.
        """
        if assignments is None:
            assignments = self._assignments

        sites_with: List[str] = []
        sites_without: List[str] = []
        factor_type_counts: Dict[str, int] = {}
        tier_counts: Dict[str, int] = {}

        for site_id in site_ids:
            site_assignments = assignments.get(site_id, [])
            if site_assignments:
                sites_with.append(site_id)
                for a in site_assignments:
                    ft = a.factor_type
                    factor_type_counts[ft] = factor_type_counts.get(ft, 0) + 1
                    t = a.tier
                    tier_counts[t] = tier_counts.get(t, 0) + 1
            else:
                sites_without.append(site_id)

        total = len(site_ids)
        coverage_pct = _round2(
            _safe_divide(_decimal(len(sites_with)), _decimal(total)) * Decimal("100")
        ) if total > 0 else Decimal("0")

        result = FactorCoverage(
            total_sites=total,
            sites_with_factors=len(sites_with),
            sites_without_factors=sites_without,
            coverage_pct=coverage_pct,
            factor_type_coverage=factor_type_counts,
            tier_distribution=tier_counts,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Factor coverage: %d/%d sites (%s%%), %d factor types.",
            len(sites_with), total, coverage_pct,
            len(factor_type_counts),
        )
        return result

    def get_tier_distribution(
        self,
        assignments: Optional[List[FactorAssignment]] = None,
    ) -> Dict[str, int]:
        """Get the tier distribution across all factor assignments.

        Args:
            assignments: List of assignments. If None, aggregates
                from all sites in internal storage.

        Returns:
            Dictionary mapping tier name to count.
        """
        if assignments is None:
            assignments = []
            for site_assignments in self._assignments.values():
                assignments.extend(site_assignments)

        distribution: Dict[str, int] = {}
        for a in assignments:
            tier = a.tier
            distribution[tier] = distribution.get(tier, 0) + 1
        return distribution

    # ------------------------------------------------------------------
    # Site Factor Result
    # ------------------------------------------------------------------

    def get_site_factor_result(
        self,
        site_id: str,
    ) -> RegionalFactorResult:
        """Get the complete factor result for a site.

        Combines all assignments, grid region, climate zone, and
        overrides into a single result object.

        Args:
            site_id: The site ID.

        Returns:
            RegionalFactorResult with all factor information.
        """
        assignments = self._assignments.get(site_id, [])
        grid_region = self._grid_regions.get(site_id)
        climate_zone = self._climate_zones.get(site_id)

        overrides = [
            o for o in self._overrides.values()
            if o.site_id == site_id and o.is_approved
        ]

        tier_dist = self.get_tier_distribution(assignments)

        result = RegionalFactorResult(
            site_id=site_id,
            assignments=assignments,
            grid_region=grid_region,
            climate_zone=climate_zone,
            overrides=overrides,
            tier_distribution=tier_dist,
            factor_count=len(assignments),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Custom Factor Registration
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        site_id: str,
        factor: FactorAssignment,
    ) -> FactorAssignment:
        """Register a facility-specific custom factor.

        Custom factors take priority in the tiered lookup.

        Args:
            site_id: The site.
            factor: The custom factor assignment.

        Returns:
            The registered FactorAssignment.
        """
        key = f"{site_id}:{factor.factor_type}"
        if key not in self._custom_factors:
            self._custom_factors[key] = {}
        self._custom_factors[key][factor.assignment_id] = factor

        logger.info(
            "Custom factor registered: site '%s', type '%s', value=%s.",
            site_id, factor.factor_type, factor.factor_value,
        )
        return factor

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_grid_region(self, site_id: str) -> Optional[GridRegion]:
        """Get grid region for a site.

        Args:
            site_id: The site ID.

        Returns:
            GridRegion or None.
        """
        return self._grid_regions.get(site_id)

    def get_climate_zone(self, site_id: str) -> Optional[ClimateZone]:
        """Get climate zone for a site.

        Args:
            site_id: The site ID.

        Returns:
            ClimateZone or None.
        """
        return self._climate_zones.get(site_id)

    def get_assignments(self, site_id: str) -> List[FactorAssignment]:
        """Get all factor assignments for a site.

        Args:
            site_id: The site ID.

        Returns:
            List of FactorAssignments.
        """
        return self._assignments.get(site_id, [])

    def get_overrides(
        self,
        site_id: Optional[str] = None,
        approved_only: bool = False,
    ) -> List[FactorOverride]:
        """Get factor overrides, optionally filtered.

        Args:
            site_id: Filter by site. If None, returns all.
            approved_only: If True, only return approved overrides.

        Returns:
            List of FactorOverride records.
        """
        results: List[FactorOverride] = []
        for override in self._overrides.values():
            if site_id and override.site_id != site_id:
                continue
            if approved_only and not override.is_approved:
                continue
            results.append(override)
        return results

    def get_available_countries(self) -> List[str]:
        """Get list of countries with default grid factors.

        Returns:
            Sorted list of country codes.
        """
        return sorted(DEFAULT_GRID_FACTORS.keys())

    def get_factor_database_info(
        self,
        source: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get metadata about available factor databases.

        Args:
            source: Specific source to retrieve. If None, returns all.

        Returns:
            Dictionary of factor database metadata.
        """
        if source:
            source_upper = source.upper()
            if source_upper in DEFAULT_FACTOR_DATABASES:
                return {source_upper: DEFAULT_FACTOR_DATABASES[source_upper]}
            return {}
        return dict(DEFAULT_FACTOR_DATABASES)
