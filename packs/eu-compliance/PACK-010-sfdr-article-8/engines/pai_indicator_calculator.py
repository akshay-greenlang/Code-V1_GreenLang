# -*- coding: utf-8 -*-
"""
PAIIndicatorCalculatorEngine - PACK-010 SFDR Article 8 Engine 1
================================================================

Calculates all 18 mandatory Principal Adverse Impact (PAI) indicators defined
in SFDR Regulatory Technical Standards (RTS) Annex I, Table 1.

PAI indicators quantify the negative sustainability impacts of investment
decisions at the fund/portfolio level. Each indicator is computed as a
portfolio-weighted aggregate of investee-level data.

Indicator Groups:
    PAI 1-6   : Climate & GHG (Scope 1/2/3, carbon footprint, fossil fuel)
    PAI 7-9   : Biodiversity, Water, Waste (environmental)
    PAI 10-14 : Social & Governance (UNGC, gender pay, board diversity, weapons)
    PAI 15-16 : Sovereign (country GHG intensity, social violations)
    PAI 17-18 : Real Estate (fossil fuel exposure, energy efficiency)

Zero-Hallucination:
    - All calculations use deterministic Python arithmetic
    - Coverage ratios track data availability per indicator
    - Data quality flags distinguish REPORTED / ESTIMATED / MODELED
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _round_val(value: Any, places: int = 6) -> float:
    """Round a Decimal (or numeric) to specified places and return float."""
    if not isinstance(value, Decimal):
        value = _decimal(value)
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)


def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide safely, returning zero when denominator is zero."""
    if denominator == Decimal("0"):
        return Decimal("0")
    return numerator / denominator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PAIIndicatorId(str, Enum):
    """All 18 mandatory PAI indicators from SFDR RTS Table 1."""

    PAI_1 = "PAI_1"    # GHG emissions (Scope 1 + 2 + 3 + Total)
    PAI_2 = "PAI_2"    # Carbon footprint
    PAI_3 = "PAI_3"    # GHG intensity of investee companies
    PAI_4 = "PAI_4"    # Exposure to fossil fuel companies
    PAI_5 = "PAI_5"    # Non-renewable energy share
    PAI_6 = "PAI_6"    # Energy intensity per high-impact climate sector
    PAI_7 = "PAI_7"    # Biodiversity-sensitive areas
    PAI_8 = "PAI_8"    # Emissions to water
    PAI_9 = "PAI_9"    # Hazardous waste and radioactive waste ratio
    PAI_10 = "PAI_10"  # UNGC/OECD principles violations
    PAI_11 = "PAI_11"  # Lack of UNGC/OECD compliance processes
    PAI_12 = "PAI_12"  # Unadjusted gender pay gap
    PAI_13 = "PAI_13"  # Board gender diversity
    PAI_14 = "PAI_14"  # Exposure to controversial weapons
    PAI_15 = "PAI_15"  # GHG intensity of investee countries
    PAI_16 = "PAI_16"  # Investee countries subject to social violations
    PAI_17 = "PAI_17"  # Fossil fuels through real estate
    PAI_18 = "PAI_18"  # Energy-inefficient real estate


class PAICategory(str, Enum):
    """PAI indicator groupings per SFDR RTS."""

    CLIMATE_GHG = "climate_ghg"
    ENVIRONMENT = "environment"
    SOCIAL = "social"
    SOVEREIGN = "sovereign"
    REAL_ESTATE = "real_estate"


class DataQualityFlag(str, Enum):
    """Data quality classification for investee-level data."""

    REPORTED = "REPORTED"       # Company-disclosed, audited or verified
    ESTIMATED = "ESTIMATED"     # Proxy-based or model-estimated by data provider
    MODELED = "MODELED"         # In-house modeled from secondary inputs
    NOT_AVAILABLE = "NOT_AVAILABLE"


class NACESector(str, Enum):
    """NACE high-impact climate sectors per SFDR RTS (PAI 6)."""

    A = "A"   # Agriculture, forestry and fishing
    B = "B"   # Mining and quarrying
    C = "C"   # Manufacturing
    D = "D"   # Electricity, gas, steam and air conditioning supply
    E = "E"   # Water supply, sewerage, waste management
    F = "F"   # Construction
    G = "G"   # Wholesale and retail trade
    H = "H"   # Transportation and storage
    L = "L"   # Real estate activities


# ---------------------------------------------------------------------------
# Indicator Metadata Registry
# ---------------------------------------------------------------------------

PAI_METADATA: Dict[str, Dict[str, Any]] = {
    PAIIndicatorId.PAI_1: {
        "name": "GHG emissions",
        "category": PAICategory.CLIMATE_GHG,
        "metric": "Scope 1 + Scope 2 + Scope 3 GHG emissions (tCO2eq)",
        "unit": "tCO2eq",
        "description": "Scope 1, Scope 2, Scope 3 and Total GHG emissions",
        "aggregation": "attribution",
        "rts_table": "Table 1, Row 1",
    },
    PAIIndicatorId.PAI_2: {
        "name": "Carbon footprint",
        "category": PAICategory.CLIMATE_GHG,
        "metric": "Carbon footprint (tCO2eq / EUR M invested)",
        "unit": "tCO2eq / EUR M",
        "description": "Total GHG emissions per EUR million invested",
        "aggregation": "sum_over_nav",
        "rts_table": "Table 1, Row 2",
    },
    PAIIndicatorId.PAI_3: {
        "name": "GHG intensity of investee companies",
        "category": PAICategory.CLIMATE_GHG,
        "metric": "Weighted average GHG intensity (tCO2eq / EUR M revenue)",
        "unit": "tCO2eq / EUR M revenue",
        "description": "Portfolio-weighted average of investee GHG intensity",
        "aggregation": "weighted_average",
        "rts_table": "Table 1, Row 3",
    },
    PAIIndicatorId.PAI_4: {
        "name": "Exposure to companies active in the fossil fuel sector",
        "category": PAICategory.CLIMATE_GHG,
        "metric": "Share of investments in fossil fuel companies (%)",
        "unit": "%",
        "description": "Percentage of portfolio value in fossil fuel companies",
        "aggregation": "share",
        "rts_table": "Table 1, Row 4",
    },
    PAIIndicatorId.PAI_5: {
        "name": "Share of non-renewable energy consumption and production",
        "category": PAICategory.CLIMATE_GHG,
        "metric": "Weighted average non-renewable energy share (%)",
        "unit": "%",
        "description": "Portfolio-weighted share of non-renewable energy",
        "aggregation": "weighted_average",
        "rts_table": "Table 1, Row 5",
    },
    PAIIndicatorId.PAI_6: {
        "name": "Energy consumption intensity per high impact climate sector",
        "category": PAICategory.CLIMATE_GHG,
        "metric": "Energy intensity (GWh / EUR M revenue) by NACE sector",
        "unit": "GWh / EUR M revenue",
        "description": "Energy consumption intensity per NACE high-impact sector",
        "aggregation": "weighted_average_by_sector",
        "rts_table": "Table 1, Row 6",
    },
    PAIIndicatorId.PAI_7: {
        "name": "Activities negatively affecting biodiversity-sensitive areas",
        "category": PAICategory.ENVIRONMENT,
        "metric": "Share of investments in companies with biodiversity impact (%)",
        "unit": "%",
        "description": "Investments near or affecting biodiversity-sensitive areas",
        "aggregation": "share",
        "rts_table": "Table 1, Row 7",
    },
    PAIIndicatorId.PAI_8: {
        "name": "Emissions to water",
        "category": PAICategory.ENVIRONMENT,
        "metric": "Weighted average water pollutant emissions (tonnes)",
        "unit": "tonnes",
        "description": "Tonnes of emissions to water by portfolio companies",
        "aggregation": "attribution",
        "rts_table": "Table 1, Row 8",
    },
    PAIIndicatorId.PAI_9: {
        "name": "Hazardous waste and radioactive waste ratio",
        "category": PAICategory.ENVIRONMENT,
        "metric": "Weighted average hazardous + radioactive waste (tonnes)",
        "unit": "tonnes",
        "description": "Tonnes of hazardous and radioactive waste generated",
        "aggregation": "attribution",
        "rts_table": "Table 1, Row 9",
    },
    PAIIndicatorId.PAI_10: {
        "name": "Violations of UN Global Compact and OECD Guidelines",
        "category": PAICategory.SOCIAL,
        "metric": "Share with UNGC/OECD violations (%)",
        "unit": "%",
        "description": "Percentage of investments in companies with violations",
        "aggregation": "share",
        "rts_table": "Table 1, Row 10",
    },
    PAIIndicatorId.PAI_11: {
        "name": "Lack of processes and compliance mechanisms to monitor UNGC/OECD",
        "category": PAICategory.SOCIAL,
        "metric": "Share lacking compliance mechanisms (%)",
        "unit": "%",
        "description": "Investments in companies without UNGC/OECD compliance processes",
        "aggregation": "share",
        "rts_table": "Table 1, Row 11",
    },
    PAIIndicatorId.PAI_12: {
        "name": "Unadjusted gender pay gap",
        "category": PAICategory.SOCIAL,
        "metric": "Weighted average unadjusted gender pay gap (%)",
        "unit": "%",
        "description": "Portfolio-weighted average gender pay gap",
        "aggregation": "weighted_average",
        "rts_table": "Table 1, Row 12",
    },
    PAIIndicatorId.PAI_13: {
        "name": "Board gender diversity",
        "category": PAICategory.SOCIAL,
        "metric": "Weighted average female-to-male board ratio (%)",
        "unit": "%",
        "description": "Portfolio-weighted average ratio of female board members",
        "aggregation": "weighted_average",
        "rts_table": "Table 1, Row 13",
    },
    PAIIndicatorId.PAI_14: {
        "name": "Exposure to controversial weapons",
        "category": PAICategory.SOCIAL,
        "metric": "Share exposed to controversial weapons (%)",
        "unit": "%",
        "description": "Investments in companies involved with controversial weapons",
        "aggregation": "share",
        "rts_table": "Table 1, Row 14",
    },
    PAIIndicatorId.PAI_15: {
        "name": "GHG intensity of investee countries",
        "category": PAICategory.SOVEREIGN,
        "metric": "Weighted average country GHG intensity (tCO2eq / EUR M GDP)",
        "unit": "tCO2eq / EUR M GDP",
        "description": "GHG intensity of sovereign bond issuer countries",
        "aggregation": "weighted_average",
        "rts_table": "Table 1, Row 15",
    },
    PAIIndicatorId.PAI_16: {
        "name": "Investee countries subject to social violations",
        "category": PAICategory.SOVEREIGN,
        "metric": "Share in countries with social violations (%)",
        "unit": "%",
        "description": "Investments in sovereigns with social violations",
        "aggregation": "share",
        "rts_table": "Table 1, Row 16",
    },
    PAIIndicatorId.PAI_17: {
        "name": "Exposure to fossil fuels through real estate assets",
        "category": PAICategory.REAL_ESTATE,
        "metric": "Share of real estate exposed to fossil fuels (%)",
        "unit": "%",
        "description": "Real estate investments involved in fossil fuel extraction/storage/transport",
        "aggregation": "share",
        "rts_table": "Table 1, Row 17",
    },
    PAIIndicatorId.PAI_18: {
        "name": "Exposure to energy-inefficient real estate assets",
        "category": PAICategory.REAL_ESTATE,
        "metric": "Share of energy-inefficient real estate (%)",
        "unit": "%",
        "description": "Real estate investments that are energy-inefficient (below NZEB)",
        "aggregation": "share",
        "rts_table": "Table 1, Row 18",
    },
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class PAIIndicatorConfig(BaseModel):
    """Configuration for the PAI Indicator Calculator Engine.

    Attributes:
        reporting_period_start: Start of the PAI reporting period.
        reporting_period_end: End of the PAI reporting period.
        total_nav_eur: Total Net Asset Value of the fund in EUR.
        total_aum_eur: Total Assets Under Management in EUR.
        coverage_threshold_pct: Minimum data coverage to consider result valid.
        include_scope_3: Whether to include Scope 3 in GHG totals.
        high_impact_nace_sectors: NACE sectors considered high-impact for PAI 6.
    """

    reporting_period_start: datetime = Field(
        ..., description="PAI reporting period start date",
    )
    reporting_period_end: datetime = Field(
        ..., description="PAI reporting period end date",
    )
    total_nav_eur: float = Field(
        ..., gt=0,
        description="Total Net Asset Value of the fund (EUR)",
    )
    total_aum_eur: Optional[float] = Field(
        None, gt=0,
        description="Total Assets Under Management (EUR), defaults to NAV",
    )
    coverage_threshold_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum coverage percentage for valid results",
    )
    include_scope_3: bool = Field(
        default=True,
        description="Include Scope 3 emissions in PAI 1 total",
    )
    high_impact_nace_sectors: List[NACESector] = Field(
        default_factory=lambda: [
            NACESector.A, NACESector.B, NACESector.C, NACESector.D,
            NACESector.E, NACESector.F, NACESector.G, NACESector.H,
            NACESector.L,
        ],
        description="NACE sectors for PAI 6 energy intensity calculation",
    )

    @field_validator("reporting_period_end")
    @classmethod
    def end_after_start(cls, v: datetime, info: Any) -> datetime:
        """Ensure end date is after start date."""
        start = info.data.get("reporting_period_start")
        if start and v <= start:
            raise ValueError("reporting_period_end must be after reporting_period_start")
        return v


class InvesteeGHGData(BaseModel):
    """GHG emissions data for a single investee company."""

    scope_1_tco2eq: Optional[float] = Field(
        None, ge=0, description="Scope 1 GHG emissions (tCO2eq)",
    )
    scope_2_tco2eq: Optional[float] = Field(
        None, ge=0, description="Scope 2 GHG emissions (tCO2eq)",
    )
    scope_3_tco2eq: Optional[float] = Field(
        None, ge=0, description="Scope 3 GHG emissions (tCO2eq)",
    )
    total_ghg_tco2eq: Optional[float] = Field(
        None, ge=0,
        description="Total GHG emissions (auto-calculated if not provided)",
    )
    revenue_eur: Optional[float] = Field(
        None, gt=0, description="Company annual revenue in EUR",
    )
    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.NOT_AVAILABLE,
        description="Quality flag for the emissions data",
    )

    @model_validator(mode="after")
    def compute_total_if_missing(self) -> "InvesteeGHGData":
        """Auto-compute total GHG from scopes if not explicitly provided."""
        if self.total_ghg_tco2eq is None:
            s1 = self.scope_1_tco2eq or 0.0
            s2 = self.scope_2_tco2eq or 0.0
            s3 = self.scope_3_tco2eq or 0.0
            self.total_ghg_tco2eq = s1 + s2 + s3
        return self


class InvesteeEnvironmentalData(BaseModel):
    """Environmental data for PAI 7-9."""

    affects_biodiversity_sensitive_area: Optional[bool] = Field(
        None, description="Whether company activities affect biodiversity areas",
    )
    emissions_to_water_tonnes: Optional[float] = Field(
        None, ge=0, description="Tonnes of water pollutant emissions",
    )
    hazardous_waste_tonnes: Optional[float] = Field(
        None, ge=0, description="Tonnes of hazardous waste generated",
    )
    radioactive_waste_tonnes: Optional[float] = Field(
        None, ge=0, description="Tonnes of radioactive waste generated",
    )
    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.NOT_AVAILABLE,
    )


class InvesteeSocialData(BaseModel):
    """Social/governance data for PAI 10-14."""

    has_ungc_oecd_violations: Optional[bool] = Field(
        None, description="Whether company has UNGC/OECD principle violations",
    )
    has_compliance_mechanisms: Optional[bool] = Field(
        None, description="Whether company has UNGC/OECD compliance processes",
    )
    unadjusted_gender_pay_gap_pct: Optional[float] = Field(
        None, ge=-100.0, le=100.0,
        description="Unadjusted gender pay gap as percentage",
    )
    female_board_members_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Percentage of female board members",
    )
    involved_controversial_weapons: Optional[bool] = Field(
        None,
        description="Involved in anti-personnel mines, cluster munitions, "
                    "chemical/biological weapons, depleted uranium, "
                    "white phosphorus, or nuclear weapons",
    )
    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.NOT_AVAILABLE,
    )


class InvesteeEnergyData(BaseModel):
    """Energy data for PAI 5 and PAI 6."""

    non_renewable_energy_share_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Share of non-renewable energy consumption+production (%)",
    )
    energy_consumption_gwh: Optional[float] = Field(
        None, ge=0, description="Total energy consumption (GWh)",
    )
    nace_sector: Optional[NACESector] = Field(
        None, description="Primary NACE sector of the company",
    )
    is_fossil_fuel_company: Optional[bool] = Field(
        None,
        description="Whether company is active in fossil fuel sector "
                    "(exploration, processing, storage, transport of fossil fuels)",
    )
    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.NOT_AVAILABLE,
    )


class SovereignData(BaseModel):
    """Data for sovereign bond PAI indicators (15-16)."""

    country_code: str = Field(
        ..., min_length=2, max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    ghg_intensity_tco2eq_per_eur_m_gdp: Optional[float] = Field(
        None, ge=0,
        description="Country GHG intensity (tCO2eq per EUR M GDP)",
    )
    has_social_violations: Optional[bool] = Field(
        None,
        description="Whether country is subject to social violations "
                    "(per international treaties, UN sanctions)",
    )
    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.NOT_AVAILABLE,
    )


class RealEstateData(BaseModel):
    """Data for real estate PAI indicators (17-18)."""

    involved_fossil_fuels: Optional[bool] = Field(
        None,
        description="Whether real estate is involved in fossil fuel "
                    "extraction, storage, transport, or manufacture",
    )
    is_energy_inefficient: Optional[bool] = Field(
        None,
        description="Whether asset is below nearly zero-energy building "
                    "(NZEB) standard or applicable EPC threshold",
    )
    energy_performance_certificate: Optional[str] = Field(
        None,
        description="Energy Performance Certificate rating (A-G)",
    )
    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.NOT_AVAILABLE,
    )


class InvesteeData(BaseModel):
    """Complete data for a single investee holding.

    Combines financial exposure with all PAI-relevant sustainability metrics.
    The portfolio weight is calculated from value_eur / total_nav.

    Attributes:
        investee_id: Unique identifier (ISIN, LEI, or internal).
        investee_name: Human-readable name.
        investee_type: CORPORATE, SOVEREIGN, or REAL_ESTATE.
        value_eur: Current value of the holding in EUR.
        enterprise_value_eur: Enterprise value of the investee (for attribution).
        ghg_data: GHG emissions data.
        environmental_data: Environmental impact data.
        social_data: Social and governance data.
        energy_data: Energy consumption data.
        sovereign_data: Sovereign-specific data.
        real_estate_data: Real estate specific data.
    """

    investee_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Unique investee identifier (ISIN, LEI, or internal ID)",
    )
    investee_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Investee name",
    )
    investee_type: str = Field(
        ...,
        description="Type: CORPORATE, SOVEREIGN, or REAL_ESTATE",
    )
    value_eur: float = Field(
        ..., gt=0,
        description="Current value of the holding in EUR",
    )
    enterprise_value_eur: Optional[float] = Field(
        None, gt=0,
        description="Enterprise value including cash (EVIC) in EUR",
    )
    ghg_data: Optional[InvesteeGHGData] = Field(
        None, description="GHG emissions data",
    )
    environmental_data: Optional[InvesteeEnvironmentalData] = Field(
        None, description="Environmental impact data",
    )
    social_data: Optional[InvesteeSocialData] = Field(
        None, description="Social and governance data",
    )
    energy_data: Optional[InvesteeEnergyData] = Field(
        None, description="Energy consumption data",
    )
    sovereign_data: Optional[SovereignData] = Field(
        None, description="Sovereign bond specific data",
    )
    real_estate_data: Optional[RealEstateData] = Field(
        None, description="Real estate specific data",
    )

    @field_validator("investee_type")
    @classmethod
    def validate_investee_type(cls, v: str) -> str:
        """Validate investee type is one of the allowed values."""
        allowed = {"CORPORATE", "SOVEREIGN", "REAL_ESTATE"}
        upper = v.strip().upper()
        if upper not in allowed:
            raise ValueError(
                f"investee_type must be one of {allowed}, got '{v}'"
            )
        return upper


class PAICoverage(BaseModel):
    """Data coverage statistics for a single PAI indicator.

    Tracks what percentage of the portfolio has data for the indicator,
    broken down by data quality classification.
    """

    indicator_id: PAIIndicatorId = Field(
        ..., description="PAI indicator identifier",
    )
    total_holdings: int = Field(
        ..., ge=0, description="Total number of holdings",
    )
    holdings_with_data: int = Field(
        ..., ge=0, description="Holdings with data for this indicator",
    )
    holdings_without_data: int = Field(
        ..., ge=0, description="Holdings lacking data",
    )
    coverage_by_count_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Coverage by holding count (%)",
    )
    coverage_by_value_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Coverage by portfolio value (%)",
    )
    quality_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of holdings per DataQualityFlag",
    )
    is_sufficient: bool = Field(
        ..., description="Whether coverage meets configured threshold",
    )


class PAISingleResult(BaseModel):
    """Result for a single PAI indicator calculation."""

    indicator_id: PAIIndicatorId = Field(
        ..., description="PAI indicator identifier",
    )
    indicator_name: str = Field(
        ..., description="Human-readable indicator name",
    )
    category: PAICategory = Field(
        ..., description="PAI category",
    )
    value: Optional[float] = Field(
        None, description="Calculated indicator value",
    )
    unit: str = Field(
        ..., description="Unit of measurement",
    )
    sub_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Breakdown values (e.g., Scope 1/2/3 for PAI 1)",
    )
    coverage: PAICoverage = Field(
        ..., description="Data coverage for this indicator",
    )
    data_quality_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribution of data quality flags (pct)",
    )
    methodology_note: str = Field(
        default="", description="Note on calculation methodology",
    )
    rts_reference: str = Field(
        default="", description="RTS table reference",
    )


class PAIResult(BaseModel):
    """Complete PAI calculation result for a portfolio.

    Contains all 18 indicator results, coverage analysis, period-over-period
    comparison data, and provenance tracking.
    """

    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier",
    )
    fund_name: Optional[str] = Field(
        None, description="Fund or product name",
    )
    reporting_period_start: datetime = Field(
        ..., description="PAI reporting period start",
    )
    reporting_period_end: datetime = Field(
        ..., description="PAI reporting period end",
    )
    total_nav_eur: float = Field(
        ..., gt=0, description="Total NAV at calculation time",
    )
    total_holdings: int = Field(
        ..., ge=0, description="Total number of holdings assessed",
    )
    indicators: Dict[str, PAISingleResult] = Field(
        default_factory=dict,
        description="Results keyed by PAIIndicatorId value",
    )
    overall_coverage_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Average data coverage across all indicators",
    )
    calculation_timestamp: datetime = Field(
        default_factory=_utcnow,
        description="When the calculation was performed",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Total processing time in milliseconds",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class PAIPeriodComparison(BaseModel):
    """Period-over-period comparison for a single PAI indicator."""

    indicator_id: PAIIndicatorId = Field(
        ..., description="PAI indicator identifier",
    )
    current_value: Optional[float] = Field(
        None, description="Current period value",
    )
    previous_value: Optional[float] = Field(
        None, description="Previous period value",
    )
    absolute_change: Optional[float] = Field(
        None, description="Absolute change (current - previous)",
    )
    percentage_change: Optional[float] = Field(
        None, description="Percentage change",
    )
    direction: str = Field(
        default="unchanged",
        description="Direction of change: improved, worsened, unchanged, or insufficient_data",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PAIIndicatorCalculatorEngine:
    """PAI Indicator Calculator for all 18 mandatory SFDR indicators.

    Implements the full PAI calculation methodology from SFDR RTS Annex I,
    Table 1. Each indicator uses portfolio-weighted aggregation with
    appropriate formulas (attribution, weighted average, or share-based).

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Decimal arithmetic
        - Coverage ratios track data availability per indicator
        - Data quality flags distinguish REPORTED / ESTIMATED / MODELED
        - SHA-256 provenance hashing on every result
        - No LLM involvement in any numeric path

    Attributes:
        config: PAI calculation configuration.
        _calculation_count: Running count of calculations performed.

    Example:
        >>> config = PAIIndicatorConfig(
        ...     reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ...     reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        ...     total_nav_eur=100_000_000.0,
        ... )
        >>> engine = PAIIndicatorCalculatorEngine(config)
        >>> result = engine.calculate_all_pai(holdings)
        >>> assert len(result.indicators) == 18
    """

    def __init__(self, config: PAIIndicatorConfig) -> None:
        """Initialize the PAI Indicator Calculator Engine.

        Args:
            config: Configuration including reporting period, NAV, and thresholds.
        """
        self.config = config
        self._calculation_count: int = 0
        self._nav_decimal = _decimal(config.total_nav_eur)
        self._nav_millions = self._nav_decimal / _decimal("1000000")
        logger.info(
            "PAIIndicatorCalculatorEngine initialized (v%s, NAV=%.2f EUR, period=%s to %s)",
            _MODULE_VERSION,
            config.total_nav_eur,
            config.reporting_period_start.isoformat(),
            config.reporting_period_end.isoformat(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_all_pai(
        self,
        holdings: List[InvesteeData],
        fund_name: Optional[str] = None,
    ) -> PAIResult:
        """Calculate all 18 mandatory PAI indicators for the portfolio.

        Iterates through each PAI indicator, calculating the portfolio-level
        value using the appropriate aggregation methodology. Returns a
        complete PAIResult with all indicators, coverage analysis, and
        provenance tracking.

        Args:
            holdings: List of investee holding data.
            fund_name: Optional fund name for reporting.

        Returns:
            PAIResult containing all 18 indicator calculations.

        Raises:
            ValueError: If holdings list is empty.
        """
        start = _utcnow()
        self._calculation_count += 1

        if not holdings:
            raise ValueError("Holdings list cannot be empty")

        logger.info(
            "Calculating all 18 PAI indicators for %d holdings (NAV=%.2f EUR)",
            len(holdings), self.config.total_nav_eur,
        )

        indicators: Dict[str, PAISingleResult] = {}
        total_coverage = Decimal("0")

        for pai_id in PAIIndicatorId:
            single_result = self.calculate_single_pai(pai_id, holdings)
            indicators[pai_id.value] = single_result
            total_coverage += _decimal(single_result.coverage.coverage_by_value_pct)

        overall_coverage = _round_val(
            _safe_divide(total_coverage, _decimal(len(PAIIndicatorId))), 2
        )

        elapsed_ms = (
            _utcnow() - start
        ).total_seconds() * 1000

        result = PAIResult(
            fund_name=fund_name,
            reporting_period_start=self.config.reporting_period_start,
            reporting_period_end=self.config.reporting_period_end,
            total_nav_eur=self.config.total_nav_eur,
            total_holdings=len(holdings),
            indicators=indicators,
            overall_coverage_pct=overall_coverage,
            processing_time_ms=round(elapsed_ms, 2),
        )

        result.provenance_hash = _compute_hash({
            "config": self.config.model_dump(mode="json"),
            "holdings_count": len(holdings),
            "indicators": {k: v.value for k, v in indicators.items()},
            "overall_coverage": overall_coverage,
        })

        logger.info(
            "PAI calculation complete: %d indicators, overall coverage=%.1f%%, "
            "time=%.1fms",
            len(indicators), overall_coverage, elapsed_ms,
        )

        return result

    def calculate_single_pai(
        self,
        indicator_id: PAIIndicatorId,
        holdings: List[InvesteeData],
    ) -> PAISingleResult:
        """Calculate a single PAI indicator for the portfolio.

        Dispatches to the appropriate indicator-specific calculation method
        based on the indicator_id.

        Args:
            indicator_id: Which PAI indicator to calculate.
            holdings: List of investee holding data.

        Returns:
            PAISingleResult with value, coverage, and methodology note.

        Raises:
            ValueError: If indicator_id is not recognized.
        """
        metadata = PAI_METADATA.get(indicator_id)
        if not metadata:
            raise ValueError(f"Unknown PAI indicator: {indicator_id}")

        dispatch: Dict[PAIIndicatorId, Any] = {
            PAIIndicatorId.PAI_1: self._calc_pai_1,
            PAIIndicatorId.PAI_2: self._calc_pai_2,
            PAIIndicatorId.PAI_3: self._calc_pai_3,
            PAIIndicatorId.PAI_4: self._calc_pai_4,
            PAIIndicatorId.PAI_5: self._calc_pai_5,
            PAIIndicatorId.PAI_6: self._calc_pai_6,
            PAIIndicatorId.PAI_7: self._calc_pai_7,
            PAIIndicatorId.PAI_8: self._calc_pai_8,
            PAIIndicatorId.PAI_9: self._calc_pai_9,
            PAIIndicatorId.PAI_10: self._calc_pai_10,
            PAIIndicatorId.PAI_11: self._calc_pai_11,
            PAIIndicatorId.PAI_12: self._calc_pai_12,
            PAIIndicatorId.PAI_13: self._calc_pai_13,
            PAIIndicatorId.PAI_14: self._calc_pai_14,
            PAIIndicatorId.PAI_15: self._calc_pai_15,
            PAIIndicatorId.PAI_16: self._calc_pai_16,
            PAIIndicatorId.PAI_17: self._calc_pai_17,
            PAIIndicatorId.PAI_18: self._calc_pai_18,
        }

        calc_fn = dispatch.get(indicator_id)
        if calc_fn is None:
            raise ValueError(f"No calculation function for {indicator_id}")

        logger.debug("Calculating %s: %s", indicator_id.value, metadata["name"])
        return calc_fn(holdings, metadata)

    def get_coverage_ratios(
        self,
        holdings: List[InvesteeData],
    ) -> Dict[str, PAICoverage]:
        """Calculate data coverage ratios for all 18 PAI indicators.

        Determines what percentage of the portfolio (by count and by value)
        has available data for each indicator.

        Args:
            holdings: List of investee holding data.

        Returns:
            Dictionary keyed by PAI indicator ID with PAICoverage objects.
        """
        logger.info("Computing coverage ratios for %d holdings", len(holdings))
        coverages: Dict[str, PAICoverage] = {}

        for pai_id in PAIIndicatorId:
            coverages[pai_id.value] = self._compute_coverage(pai_id, holdings)

        return coverages

    def compare_periods(
        self,
        current_result: PAIResult,
        previous_result: PAIResult,
    ) -> List[PAIPeriodComparison]:
        """Compare PAI indicator values between two reporting periods.

        Calculates absolute and percentage changes for each indicator and
        classifies the direction (improved, worsened, unchanged).

        For indicators where higher values are worse (PAI 1-12, 14-18),
        a decrease is classified as 'improved'. For PAI 13 (board gender
        diversity), an increase is classified as 'improved'.

        Args:
            current_result: Current period PAI result.
            previous_result: Previous period PAI result.

        Returns:
            List of PAIPeriodComparison objects, one per indicator.
        """
        logger.info(
            "Comparing PAI periods: %s vs %s",
            current_result.reporting_period_end.isoformat(),
            previous_result.reporting_period_end.isoformat(),
        )

        # PAI 13 is the only indicator where higher = better
        higher_is_better = {PAIIndicatorId.PAI_13}

        comparisons: List[PAIPeriodComparison] = []

        for pai_id in PAIIndicatorId:
            current_ind = current_result.indicators.get(pai_id.value)
            previous_ind = previous_result.indicators.get(pai_id.value)

            current_val = current_ind.value if current_ind else None
            previous_val = previous_ind.value if previous_ind else None

            comparison = self._build_comparison(
                pai_id, current_val, previous_val,
                pai_id in higher_is_better,
            )
            comparisons.append(comparison)

        return comparisons

    # ------------------------------------------------------------------
    # PAI 1: GHG Emissions (Scope 1 + 2 + 3 + Total)
    # ------------------------------------------------------------------

    def _calc_pai_1(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 1: GHG emissions using attribution method.

        Formula (per investee):
            attributed_emissions = (value_invested / EVIC) * total_ghg

        Portfolio total = SUM(attributed_emissions_i)
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_1, holdings)

        scope_1_total = Decimal("0")
        scope_2_total = Decimal("0")
        scope_3_total = Decimal("0")
        total_ghg = Decimal("0")

        for h in corporates:
            if not h.ghg_data or h.ghg_data.total_ghg_tco2eq is None:
                continue

            attribution_factor = self._attribution_factor(h)

            s1 = _decimal(h.ghg_data.scope_1_tco2eq or 0.0) * attribution_factor
            s2 = _decimal(h.ghg_data.scope_2_tco2eq or 0.0) * attribution_factor
            s3 = _decimal(h.ghg_data.scope_3_tco2eq or 0.0) * attribution_factor

            scope_1_total += s1
            scope_2_total += s2
            scope_3_total += s3

            if self.config.include_scope_3:
                total_ghg += s1 + s2 + s3
            else:
                total_ghg += s1 + s2

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_1,
            indicator_name=metadata["name"],
            category=PAICategory.CLIMATE_GHG,
            value=_round_val(total_ghg, 2),
            unit=metadata["unit"],
            sub_values={
                "scope_1_tco2eq": _round_val(scope_1_total, 2),
                "scope_2_tco2eq": _round_val(scope_2_total, 2),
                "scope_3_tco2eq": _round_val(scope_3_total, 2),
                "total_tco2eq": _round_val(total_ghg, 2),
                "includes_scope_3": self.config.include_scope_3,
            },
            coverage=coverage,
            methodology_note=(
                "Attribution method: (invested_value / EVIC) * company_emissions. "
                f"Scope 3 {'included' if self.config.include_scope_3 else 'excluded'}."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 2: Carbon Footprint
    # ------------------------------------------------------------------

    def _calc_pai_2(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 2: Carbon footprint.

        Formula:
            carbon_footprint = SUM(attributed_ghg_i) / (current_NAV / 1,000,000)
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_2, holdings)

        total_attributed_ghg = Decimal("0")
        for h in corporates:
            if not h.ghg_data or h.ghg_data.total_ghg_tco2eq is None:
                continue
            attribution_factor = self._attribution_factor(h)
            total_attributed_ghg += _decimal(h.ghg_data.total_ghg_tco2eq) * attribution_factor

        carbon_footprint = _safe_divide(total_attributed_ghg, self._nav_millions)

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_2,
            indicator_name=metadata["name"],
            category=PAICategory.CLIMATE_GHG,
            value=_round_val(carbon_footprint, 2),
            unit=metadata["unit"],
            sub_values={
                "total_attributed_ghg_tco2eq": _round_val(total_attributed_ghg, 2),
                "nav_eur_millions": _round_val(self._nav_millions, 2),
            },
            coverage=coverage,
            methodology_note=(
                "Carbon footprint = SUM(attributed_emissions) / NAV_EUR_millions. "
                "Attribution = (invested_value / EVIC) * company_total_GHG."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 3: GHG Intensity of Investee Companies
    # ------------------------------------------------------------------

    def _calc_pai_3(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 3: Weighted average GHG intensity.

        Formula:
            PAI3 = SUM(weight_i * (company_ghg_i / company_revenue_i))
            where weight_i = value_i / total_portfolio_value_of_corporates
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_3, holdings)

        total_corp_value = sum(_decimal(h.value_eur) for h in corporates)
        weighted_intensity = Decimal("0")
        contributing_count = 0

        for h in corporates:
            if (
                not h.ghg_data
                or h.ghg_data.total_ghg_tco2eq is None
                or not h.ghg_data.revenue_eur
            ):
                continue

            weight = _safe_divide(_decimal(h.value_eur), total_corp_value)
            revenue_m = _decimal(h.ghg_data.revenue_eur) / _decimal("1000000")
            intensity = _safe_divide(
                _decimal(h.ghg_data.total_ghg_tco2eq), revenue_m
            )
            weighted_intensity += weight * intensity
            contributing_count += 1

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_3,
            indicator_name=metadata["name"],
            category=PAICategory.CLIMATE_GHG,
            value=_round_val(weighted_intensity, 2),
            unit=metadata["unit"],
            sub_values={
                "contributing_companies": contributing_count,
                "total_corporate_value_eur": _round_val(total_corp_value, 2),
            },
            coverage=coverage,
            methodology_note=(
                "Weighted average: SUM(portfolio_weight * (company_GHG / company_revenue_EUR_M)). "
                "Weight = holding_value / total_corporate_portfolio_value."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 4: Fossil Fuel Exposure
    # ------------------------------------------------------------------

    def _calc_pai_4(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 4: Percentage of portfolio in fossil fuel companies.

        Formula:
            PAI4 = SUM(value_i where is_fossil_fuel=True) / total_NAV * 100
        """
        coverage = self._compute_coverage(PAIIndicatorId.PAI_4, holdings)

        fossil_value = Decimal("0")
        fossil_count = 0

        for h in holdings:
            if h.energy_data and h.energy_data.is_fossil_fuel_company is True:
                fossil_value += _decimal(h.value_eur)
                fossil_count += 1

        share_pct = _safe_divide(fossil_value, self._nav_decimal) * _decimal("100")

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_4,
            indicator_name=metadata["name"],
            category=PAICategory.CLIMATE_GHG,
            value=_round_val(share_pct, 4),
            unit=metadata["unit"],
            sub_values={
                "fossil_fuel_value_eur": _round_val(fossil_value, 2),
                "fossil_fuel_company_count": fossil_count,
            },
            coverage=coverage,
            methodology_note=(
                "Share = SUM(value of fossil fuel companies) / total_NAV * 100."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 5: Non-Renewable Energy Share
    # ------------------------------------------------------------------

    def _calc_pai_5(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 5: Weighted average non-renewable energy share.

        Formula:
            PAI5 = SUM(weight_i * non_renewable_pct_i)
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_5, holdings)

        total_corp_value = sum(_decimal(h.value_eur) for h in corporates)
        weighted_nre = Decimal("0")
        contributing_count = 0

        for h in corporates:
            if (
                not h.energy_data
                or h.energy_data.non_renewable_energy_share_pct is None
            ):
                continue

            weight = _safe_divide(_decimal(h.value_eur), total_corp_value)
            weighted_nre += weight * _decimal(
                h.energy_data.non_renewable_energy_share_pct
            )
            contributing_count += 1

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_5,
            indicator_name=metadata["name"],
            category=PAICategory.CLIMATE_GHG,
            value=_round_val(weighted_nre, 4),
            unit=metadata["unit"],
            sub_values={
                "contributing_companies": contributing_count,
            },
            coverage=coverage,
            methodology_note=(
                "Weighted average: SUM(portfolio_weight * company_non_renewable_pct)."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 6: Energy Intensity per High-Impact Climate Sector
    # ------------------------------------------------------------------

    def _calc_pai_6(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 6: Energy intensity per NACE high-impact sector.

        Formula per sector:
            sector_intensity = SUM(weight_i * (energy_gwh_i / revenue_M_i))
            where weight_i = value_i / total_sector_value
        """
        coverage = self._compute_coverage(PAIIndicatorId.PAI_6, holdings)
        target_sectors = set(self.config.high_impact_nace_sectors)

        # Group holdings by NACE sector
        sector_holdings: Dict[str, List[InvesteeData]] = defaultdict(list)
        for h in holdings:
            if (
                h.investee_type == "CORPORATE"
                and h.energy_data
                and h.energy_data.nace_sector
                and h.energy_data.nace_sector in target_sectors
            ):
                sector_holdings[h.energy_data.nace_sector.value].append(h)

        sector_intensities: Dict[str, float] = {}
        total_weighted = Decimal("0")
        total_sectors_with_data = 0

        for sector_code, sector_list in sector_holdings.items():
            sector_total_value = sum(_decimal(h.value_eur) for h in sector_list)
            sector_intensity = Decimal("0")
            has_data = False

            for h in sector_list:
                if (
                    h.energy_data
                    and h.energy_data.energy_consumption_gwh is not None
                    and h.ghg_data
                    and h.ghg_data.revenue_eur
                ):
                    weight = _safe_divide(_decimal(h.value_eur), sector_total_value)
                    revenue_m = _decimal(h.ghg_data.revenue_eur) / _decimal("1000000")
                    intensity = _safe_divide(
                        _decimal(h.energy_data.energy_consumption_gwh), revenue_m
                    )
                    sector_intensity += weight * intensity
                    has_data = True

            if has_data:
                sector_intensities[sector_code] = _round_val(sector_intensity, 4)
                total_weighted += sector_intensity
                total_sectors_with_data += 1

        avg_intensity = _safe_divide(
            total_weighted, _decimal(max(total_sectors_with_data, 1))
        )

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_6,
            indicator_name=metadata["name"],
            category=PAICategory.CLIMATE_GHG,
            value=_round_val(avg_intensity, 4),
            unit=metadata["unit"],
            sub_values={
                "sector_intensities": sector_intensities,
                "sectors_with_data": total_sectors_with_data,
            },
            coverage=coverage,
            methodology_note=(
                "Per-sector weighted average: SUM(weight * (energy_GWh / revenue_EUR_M)). "
                "Overall value is average across sectors with data."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 7: Biodiversity-Sensitive Areas
    # ------------------------------------------------------------------

    def _calc_pai_7(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 7: Share of investments affecting biodiversity areas.

        Formula:
            PAI7 = SUM(value_i where affects_biodiversity=True) / total_NAV * 100
        """
        coverage = self._compute_coverage(PAIIndicatorId.PAI_7, holdings)

        affected_value = Decimal("0")
        affected_count = 0

        for h in holdings:
            if (
                h.environmental_data
                and h.environmental_data.affects_biodiversity_sensitive_area is True
            ):
                affected_value += _decimal(h.value_eur)
                affected_count += 1

        share_pct = _safe_divide(affected_value, self._nav_decimal) * _decimal("100")

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_7,
            indicator_name=metadata["name"],
            category=PAICategory.ENVIRONMENT,
            value=_round_val(share_pct, 4),
            unit=metadata["unit"],
            sub_values={
                "affected_value_eur": _round_val(affected_value, 2),
                "affected_count": affected_count,
            },
            coverage=coverage,
            methodology_note=(
                "Share = SUM(value of companies affecting biodiversity areas) "
                "/ total_NAV * 100."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 8: Emissions to Water
    # ------------------------------------------------------------------

    def _calc_pai_8(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 8: Water pollutant emissions via attribution.

        Formula:
            PAI8 = SUM((value_i / EVIC_i) * emissions_to_water_i)
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_8, holdings)

        total_water_emissions = Decimal("0")
        contributing_count = 0

        for h in corporates:
            if (
                not h.environmental_data
                or h.environmental_data.emissions_to_water_tonnes is None
            ):
                continue

            attribution_factor = self._attribution_factor(h)
            total_water_emissions += (
                _decimal(h.environmental_data.emissions_to_water_tonnes)
                * attribution_factor
            )
            contributing_count += 1

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_8,
            indicator_name=metadata["name"],
            category=PAICategory.ENVIRONMENT,
            value=_round_val(total_water_emissions, 4),
            unit=metadata["unit"],
            sub_values={
                "contributing_companies": contributing_count,
            },
            coverage=coverage,
            methodology_note=(
                "Attribution method: SUM((invested_value / EVIC) * "
                "company_water_emissions_tonnes)."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 9: Hazardous Waste and Radioactive Waste
    # ------------------------------------------------------------------

    def _calc_pai_9(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 9: Hazardous + radioactive waste via attribution.

        Formula:
            PAI9 = SUM((value_i / EVIC_i) * (hazardous_waste_i + radioactive_waste_i))
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_9, holdings)

        total_haz_waste = Decimal("0")
        total_rad_waste = Decimal("0")
        contributing_count = 0

        for h in corporates:
            if not h.environmental_data:
                continue

            haz = h.environmental_data.hazardous_waste_tonnes
            rad = h.environmental_data.radioactive_waste_tonnes
            if haz is None and rad is None:
                continue

            attribution_factor = self._attribution_factor(h)
            if haz is not None:
                total_haz_waste += _decimal(haz) * attribution_factor
            if rad is not None:
                total_rad_waste += _decimal(rad) * attribution_factor
            contributing_count += 1

        total_waste = total_haz_waste + total_rad_waste

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_9,
            indicator_name=metadata["name"],
            category=PAICategory.ENVIRONMENT,
            value=_round_val(total_waste, 4),
            unit=metadata["unit"],
            sub_values={
                "hazardous_waste_tonnes": _round_val(total_haz_waste, 4),
                "radioactive_waste_tonnes": _round_val(total_rad_waste, 4),
                "contributing_companies": contributing_count,
            },
            coverage=coverage,
            methodology_note=(
                "Attribution: SUM((invested / EVIC) * (hazardous + radioactive waste))."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 10: UNGC/OECD Principles Violations
    # ------------------------------------------------------------------

    def _calc_pai_10(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 10: Share of investments with UNGC/OECD violations.

        Formula:
            PAI10 = SUM(value_i where has_violations=True) / total_NAV * 100
        """
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_10, holdings, metadata,
            PAICategory.SOCIAL,
            lambda h: (
                h.social_data is not None
                and h.social_data.has_ungc_oecd_violations is True
            ),
            "Share = SUM(value of companies with UNGC/OECD violations) / NAV * 100.",
        )

    # ------------------------------------------------------------------
    # PAI 11: Lack of UNGC/OECD Compliance Processes
    # ------------------------------------------------------------------

    def _calc_pai_11(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 11: Share lacking compliance mechanisms.

        Formula:
            PAI11 = SUM(value_i where has_compliance_mechanisms=False) / NAV * 100
        """
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_11, holdings, metadata,
            PAICategory.SOCIAL,
            lambda h: (
                h.social_data is not None
                and h.social_data.has_compliance_mechanisms is False
            ),
            "Share = SUM(value of companies lacking UNGC/OECD mechanisms) / NAV * 100.",
        )

    # ------------------------------------------------------------------
    # PAI 12: Unadjusted Gender Pay Gap
    # ------------------------------------------------------------------

    def _calc_pai_12(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 12: Weighted average unadjusted gender pay gap.

        Formula:
            PAI12 = SUM(weight_i * gender_pay_gap_pct_i)
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_12, holdings)

        total_corp_value = sum(_decimal(h.value_eur) for h in corporates)
        weighted_gap = Decimal("0")
        contributing_count = 0

        for h in corporates:
            if (
                not h.social_data
                or h.social_data.unadjusted_gender_pay_gap_pct is None
            ):
                continue

            weight = _safe_divide(_decimal(h.value_eur), total_corp_value)
            weighted_gap += weight * _decimal(
                h.social_data.unadjusted_gender_pay_gap_pct
            )
            contributing_count += 1

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_12,
            indicator_name=metadata["name"],
            category=PAICategory.SOCIAL,
            value=_round_val(weighted_gap, 4),
            unit=metadata["unit"],
            sub_values={"contributing_companies": contributing_count},
            coverage=coverage,
            methodology_note=(
                "Weighted average: SUM(portfolio_weight * company_gender_pay_gap_pct)."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 13: Board Gender Diversity
    # ------------------------------------------------------------------

    def _calc_pai_13(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 13: Weighted average female board member percentage.

        Formula:
            PAI13 = SUM(weight_i * female_board_pct_i)
        """
        corporates = [h for h in holdings if h.investee_type == "CORPORATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_13, holdings)

        total_corp_value = sum(_decimal(h.value_eur) for h in corporates)
        weighted_diversity = Decimal("0")
        contributing_count = 0

        for h in corporates:
            if (
                not h.social_data
                or h.social_data.female_board_members_pct is None
            ):
                continue

            weight = _safe_divide(_decimal(h.value_eur), total_corp_value)
            weighted_diversity += weight * _decimal(
                h.social_data.female_board_members_pct
            )
            contributing_count += 1

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_13,
            indicator_name=metadata["name"],
            category=PAICategory.SOCIAL,
            value=_round_val(weighted_diversity, 4),
            unit=metadata["unit"],
            sub_values={"contributing_companies": contributing_count},
            coverage=coverage,
            methodology_note=(
                "Weighted average: SUM(portfolio_weight * company_female_board_pct). "
                "Higher values are better for this indicator."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 14: Controversial Weapons Exposure
    # ------------------------------------------------------------------

    def _calc_pai_14(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 14: Share exposed to controversial weapons.

        Controversial weapons include: anti-personnel mines, cluster
        munitions, chemical/biological weapons, depleted uranium,
        white phosphorus, and nuclear weapons.

        Formula:
            PAI14 = SUM(value_i where controversial_weapons=True) / NAV * 100
        """
        return self._calc_share_indicator(
            PAIIndicatorId.PAI_14, holdings, metadata,
            PAICategory.SOCIAL,
            lambda h: (
                h.social_data is not None
                and h.social_data.involved_controversial_weapons is True
            ),
            "Share = SUM(value of companies with controversial weapons) / NAV * 100.",
        )

    # ------------------------------------------------------------------
    # PAI 15: GHG Intensity of Investee Countries
    # ------------------------------------------------------------------

    def _calc_pai_15(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 15: Weighted average country GHG intensity.

        Applies to sovereign bond holdings. Formula:
            PAI15 = SUM(weight_i * country_ghg_intensity_i)
        """
        sovereigns = [h for h in holdings if h.investee_type == "SOVEREIGN"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_15, holdings)

        total_sov_value = sum(_decimal(h.value_eur) for h in sovereigns)
        weighted_intensity = Decimal("0")
        contributing_count = 0
        country_breakdown: Dict[str, float] = {}

        for h in sovereigns:
            if (
                not h.sovereign_data
                or h.sovereign_data.ghg_intensity_tco2eq_per_eur_m_gdp is None
            ):
                continue

            weight = _safe_divide(_decimal(h.value_eur), total_sov_value)
            intensity = _decimal(
                h.sovereign_data.ghg_intensity_tco2eq_per_eur_m_gdp
            )
            weighted_intensity += weight * intensity
            contributing_count += 1

            cc = h.sovereign_data.country_code
            country_breakdown[cc] = _round_val(intensity, 2)

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_15,
            indicator_name=metadata["name"],
            category=PAICategory.SOVEREIGN,
            value=_round_val(weighted_intensity, 2),
            unit=metadata["unit"],
            sub_values={
                "contributing_countries": contributing_count,
                "country_breakdown": country_breakdown,
            },
            coverage=coverage,
            methodology_note=(
                "Weighted average: SUM(sovereign_weight * country_ghg_intensity). "
                "Weight = holding_value / total_sovereign_value."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 16: Investee Countries Subject to Social Violations
    # ------------------------------------------------------------------

    def _calc_pai_16(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 16: Share in countries with social violations.

        Formula:
            PAI16 = SUM(value_i where has_social_violations=True) / total_sov_value * 100
        """
        sovereigns = [h for h in holdings if h.investee_type == "SOVEREIGN"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_16, holdings)

        total_sov_value = sum(_decimal(h.value_eur) for h in sovereigns)
        violation_value = Decimal("0")
        violation_count = 0

        for h in sovereigns:
            if (
                h.sovereign_data
                and h.sovereign_data.has_social_violations is True
            ):
                violation_value += _decimal(h.value_eur)
                violation_count += 1

        share_pct = (
            _safe_divide(violation_value, total_sov_value) * _decimal("100")
            if total_sov_value > 0
            else Decimal("0")
        )

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_16,
            indicator_name=metadata["name"],
            category=PAICategory.SOVEREIGN,
            value=_round_val(share_pct, 4),
            unit=metadata["unit"],
            sub_values={
                "violation_value_eur": _round_val(violation_value, 2),
                "violation_count": violation_count,
                "total_sovereign_value_eur": _round_val(total_sov_value, 2),
            },
            coverage=coverage,
            methodology_note=(
                "Share = SUM(value of sovereigns with social violations) "
                "/ total_sovereign_value * 100."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 17: Fossil Fuels through Real Estate
    # ------------------------------------------------------------------

    def _calc_pai_17(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 17: Real estate exposure to fossil fuels.

        Formula:
            PAI17 = SUM(value_i where re_fossil_fuel=True) / total_RE_value * 100
        """
        re_holdings = [h for h in holdings if h.investee_type == "REAL_ESTATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_17, holdings)

        total_re_value = sum(_decimal(h.value_eur) for h in re_holdings)
        fossil_value = Decimal("0")
        fossil_count = 0

        for h in re_holdings:
            if (
                h.real_estate_data
                and h.real_estate_data.involved_fossil_fuels is True
            ):
                fossil_value += _decimal(h.value_eur)
                fossil_count += 1

        share_pct = (
            _safe_divide(fossil_value, total_re_value) * _decimal("100")
            if total_re_value > 0
            else Decimal("0")
        )

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_17,
            indicator_name=metadata["name"],
            category=PAICategory.REAL_ESTATE,
            value=_round_val(share_pct, 4),
            unit=metadata["unit"],
            sub_values={
                "fossil_fuel_re_value_eur": _round_val(fossil_value, 2),
                "fossil_fuel_re_count": fossil_count,
                "total_re_value_eur": _round_val(total_re_value, 2),
            },
            coverage=coverage,
            methodology_note=(
                "Share = SUM(value of RE with fossil fuel involvement) "
                "/ total_RE_value * 100."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # PAI 18: Energy-Inefficient Real Estate
    # ------------------------------------------------------------------

    def _calc_pai_18(
        self,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
    ) -> PAISingleResult:
        """Calculate PAI 18: Real estate energy inefficiency exposure.

        Formula:
            PAI18 = SUM(value_i where energy_inefficient=True) / total_RE_value * 100
        """
        re_holdings = [h for h in holdings if h.investee_type == "REAL_ESTATE"]
        coverage = self._compute_coverage(PAIIndicatorId.PAI_18, holdings)

        total_re_value = sum(_decimal(h.value_eur) for h in re_holdings)
        inefficient_value = Decimal("0")
        inefficient_count = 0

        for h in re_holdings:
            if (
                h.real_estate_data
                and h.real_estate_data.is_energy_inefficient is True
            ):
                inefficient_value += _decimal(h.value_eur)
                inefficient_count += 1

        share_pct = (
            _safe_divide(inefficient_value, total_re_value) * _decimal("100")
            if total_re_value > 0
            else Decimal("0")
        )

        return PAISingleResult(
            indicator_id=PAIIndicatorId.PAI_18,
            indicator_name=metadata["name"],
            category=PAICategory.REAL_ESTATE,
            value=_round_val(share_pct, 4),
            unit=metadata["unit"],
            sub_values={
                "inefficient_re_value_eur": _round_val(inefficient_value, 2),
                "inefficient_re_count": inefficient_count,
                "total_re_value_eur": _round_val(total_re_value, 2),
            },
            coverage=coverage,
            methodology_note=(
                "Share = SUM(value of energy-inefficient RE) "
                "/ total_RE_value * 100. "
                "Inefficient = below NZEB or EPC threshold."
            ),
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # Shared Helper: Share-Based Indicator
    # ------------------------------------------------------------------

    def _calc_share_indicator(
        self,
        indicator_id: PAIIndicatorId,
        holdings: List[InvesteeData],
        metadata: Dict[str, Any],
        category: PAICategory,
        condition_fn: Any,
        methodology_note: str,
    ) -> PAISingleResult:
        """Generic share-based PAI indicator calculation.

        Used for PAI 10, 11, 14 where the formula is:
            share = SUM(value where condition=True) / total_NAV * 100

        Args:
            indicator_id: PAI indicator to calculate.
            holdings: All portfolio holdings.
            metadata: Indicator metadata dict.
            category: PAI category for the result.
            condition_fn: Lambda returning True if holding meets the condition.
            methodology_note: Description of methodology.

        Returns:
            PAISingleResult with share percentage.
        """
        coverage = self._compute_coverage(indicator_id, holdings)

        flagged_value = Decimal("0")
        flagged_count = 0

        for h in holdings:
            if condition_fn(h):
                flagged_value += _decimal(h.value_eur)
                flagged_count += 1

        share_pct = _safe_divide(flagged_value, self._nav_decimal) * _decimal("100")

        return PAISingleResult(
            indicator_id=indicator_id,
            indicator_name=metadata["name"],
            category=category,
            value=_round_val(share_pct, 4),
            unit=metadata["unit"],
            sub_values={
                "flagged_value_eur": _round_val(flagged_value, 2),
                "flagged_count": flagged_count,
            },
            coverage=coverage,
            methodology_note=methodology_note,
            rts_reference=metadata["rts_table"],
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _attribution_factor(self, holding: InvesteeData) -> Decimal:
        """Calculate the attribution factor for a holding.

        Attribution factor = invested_value / enterprise_value (EVIC).
        Falls back to 1.0 if EVIC is not available.

        Args:
            holding: The investee holding.

        Returns:
            Attribution factor as Decimal.
        """
        if holding.enterprise_value_eur and holding.enterprise_value_eur > 0:
            return _safe_divide(
                _decimal(holding.value_eur),
                _decimal(holding.enterprise_value_eur),
            )
        # Fallback: full attribution (conservative)
        return Decimal("1")

    def _compute_coverage(
        self,
        indicator_id: PAIIndicatorId,
        holdings: List[InvesteeData],
    ) -> PAICoverage:
        """Compute data coverage for a specific PAI indicator.

        Determines how many holdings have the relevant data fields populated
        for the given indicator.

        Args:
            indicator_id: Which PAI indicator to check coverage for.
            holdings: List of all holdings.

        Returns:
            PAICoverage with count and value-based coverage percentages.
        """
        total = len(holdings)
        if total == 0:
            return PAICoverage(
                indicator_id=indicator_id,
                total_holdings=0,
                holdings_with_data=0,
                holdings_without_data=0,
                coverage_by_count_pct=0.0,
                coverage_by_value_pct=0.0,
                is_sufficient=False,
            )

        total_value = sum(_decimal(h.value_eur) for h in holdings)
        with_data = 0
        value_with_data = Decimal("0")
        quality_counts: Dict[str, int] = defaultdict(int)

        for h in holdings:
            has_data, quality = self._check_data_availability(indicator_id, h)
            if has_data:
                with_data += 1
                value_with_data += _decimal(h.value_eur)
            quality_counts[quality.value] += 1

        count_pct = _round_val(
            _safe_divide(_decimal(with_data), _decimal(total)) * _decimal("100"), 2
        )
        value_pct = _round_val(
            _safe_divide(value_with_data, total_value) * _decimal("100"), 2
        )

        return PAICoverage(
            indicator_id=indicator_id,
            total_holdings=total,
            holdings_with_data=with_data,
            holdings_without_data=total - with_data,
            coverage_by_count_pct=count_pct,
            coverage_by_value_pct=value_pct,
            quality_breakdown=dict(quality_counts),
            is_sufficient=value_pct >= self.config.coverage_threshold_pct,
        )

    def _check_data_availability(
        self,
        indicator_id: PAIIndicatorId,
        holding: InvesteeData,
    ) -> Tuple[bool, DataQualityFlag]:
        """Check whether a holding has data for a specific PAI indicator.

        Returns a tuple of (has_data, data_quality_flag).

        Args:
            indicator_id: Which PAI indicator.
            holding: The investee holding.

        Returns:
            Tuple of (bool, DataQualityFlag).
        """
        checks: Dict[PAIIndicatorId, Any] = {
            PAIIndicatorId.PAI_1: lambda h: self._has_ghg(h),
            PAIIndicatorId.PAI_2: lambda h: self._has_ghg(h),
            PAIIndicatorId.PAI_3: lambda h: self._has_ghg_with_revenue(h),
            PAIIndicatorId.PAI_4: lambda h: self._has_energy_fossil(h),
            PAIIndicatorId.PAI_5: lambda h: self._has_nre_share(h),
            PAIIndicatorId.PAI_6: lambda h: self._has_energy_intensity(h),
            PAIIndicatorId.PAI_7: lambda h: self._has_biodiversity(h),
            PAIIndicatorId.PAI_8: lambda h: self._has_water_emissions(h),
            PAIIndicatorId.PAI_9: lambda h: self._has_waste(h),
            PAIIndicatorId.PAI_10: lambda h: self._has_ungc_violations(h),
            PAIIndicatorId.PAI_11: lambda h: self._has_compliance_mechanisms(h),
            PAIIndicatorId.PAI_12: lambda h: self._has_gender_pay(h),
            PAIIndicatorId.PAI_13: lambda h: self._has_board_diversity(h),
            PAIIndicatorId.PAI_14: lambda h: self._has_weapons(h),
            PAIIndicatorId.PAI_15: lambda h: self._has_sovereign_ghg(h),
            PAIIndicatorId.PAI_16: lambda h: self._has_sovereign_violations(h),
            PAIIndicatorId.PAI_17: lambda h: self._has_re_fossil(h),
            PAIIndicatorId.PAI_18: lambda h: self._has_re_efficiency(h),
        }

        check_fn = checks.get(indicator_id)
        if check_fn is None:
            return False, DataQualityFlag.NOT_AVAILABLE

        return check_fn(holding)

    # --- Data availability check helpers ---

    def _has_ghg(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check GHG data availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if h.ghg_data and h.ghg_data.total_ghg_tco2eq is not None:
            return True, h.ghg_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_ghg_with_revenue(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check GHG + revenue data availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if (
            h.ghg_data
            and h.ghg_data.total_ghg_tco2eq is not None
            and h.ghg_data.revenue_eur is not None
        ):
            return True, h.ghg_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_energy_fossil(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check fossil fuel flag availability."""
        if h.energy_data and h.energy_data.is_fossil_fuel_company is not None:
            return True, h.energy_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_nre_share(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check non-renewable energy share availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if (
            h.energy_data
            and h.energy_data.non_renewable_energy_share_pct is not None
        ):
            return True, h.energy_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_energy_intensity(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check energy intensity data availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if (
            h.energy_data
            and h.energy_data.energy_consumption_gwh is not None
            and h.energy_data.nace_sector is not None
            and h.ghg_data
            and h.ghg_data.revenue_eur is not None
        ):
            return True, h.energy_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_biodiversity(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check biodiversity impact flag availability."""
        if (
            h.environmental_data
            and h.environmental_data.affects_biodiversity_sensitive_area is not None
        ):
            return True, h.environmental_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_water_emissions(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check water emissions data availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if (
            h.environmental_data
            and h.environmental_data.emissions_to_water_tonnes is not None
        ):
            return True, h.environmental_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_waste(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check waste data availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if h.environmental_data and (
            h.environmental_data.hazardous_waste_tonnes is not None
            or h.environmental_data.radioactive_waste_tonnes is not None
        ):
            return True, h.environmental_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_ungc_violations(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check UNGC/OECD violations flag availability."""
        if h.social_data and h.social_data.has_ungc_oecd_violations is not None:
            return True, h.social_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_compliance_mechanisms(
        self, h: InvesteeData
    ) -> Tuple[bool, DataQualityFlag]:
        """Check compliance mechanisms flag availability."""
        if h.social_data and h.social_data.has_compliance_mechanisms is not None:
            return True, h.social_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_gender_pay(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check gender pay gap data availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if (
            h.social_data
            and h.social_data.unadjusted_gender_pay_gap_pct is not None
        ):
            return True, h.social_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_board_diversity(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check board diversity data availability."""
        if h.investee_type != "CORPORATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if h.social_data and h.social_data.female_board_members_pct is not None:
            return True, h.social_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_weapons(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check controversial weapons flag availability."""
        if (
            h.social_data
            and h.social_data.involved_controversial_weapons is not None
        ):
            return True, h.social_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_sovereign_ghg(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check sovereign GHG intensity availability."""
        if h.investee_type != "SOVEREIGN":
            return False, DataQualityFlag.NOT_AVAILABLE
        if (
            h.sovereign_data
            and h.sovereign_data.ghg_intensity_tco2eq_per_eur_m_gdp is not None
        ):
            return True, h.sovereign_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_sovereign_violations(
        self, h: InvesteeData
    ) -> Tuple[bool, DataQualityFlag]:
        """Check sovereign social violations flag availability."""
        if h.investee_type != "SOVEREIGN":
            return False, DataQualityFlag.NOT_AVAILABLE
        if h.sovereign_data and h.sovereign_data.has_social_violations is not None:
            return True, h.sovereign_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_re_fossil(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check real estate fossil fuel involvement availability."""
        if h.investee_type != "REAL_ESTATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if h.real_estate_data and h.real_estate_data.involved_fossil_fuels is not None:
            return True, h.real_estate_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    def _has_re_efficiency(self, h: InvesteeData) -> Tuple[bool, DataQualityFlag]:
        """Check real estate energy efficiency availability."""
        if h.investee_type != "REAL_ESTATE":
            return False, DataQualityFlag.NOT_AVAILABLE
        if (
            h.real_estate_data
            and h.real_estate_data.is_energy_inefficient is not None
        ):
            return True, h.real_estate_data.data_quality
        return False, DataQualityFlag.NOT_AVAILABLE

    # ------------------------------------------------------------------
    # Period Comparison Helper
    # ------------------------------------------------------------------

    def _build_comparison(
        self,
        indicator_id: PAIIndicatorId,
        current_val: Optional[float],
        previous_val: Optional[float],
        higher_is_better: bool,
    ) -> PAIPeriodComparison:
        """Build a period-over-period comparison for a single indicator.

        Args:
            indicator_id: PAI indicator.
            current_val: Current period value.
            previous_val: Previous period value.
            higher_is_better: Whether higher values are better (True for PAI 13).

        Returns:
            PAIPeriodComparison with change direction classified.
        """
        if current_val is None or previous_val is None:
            return PAIPeriodComparison(
                indicator_id=indicator_id,
                current_value=current_val,
                previous_value=previous_val,
                direction="insufficient_data",
                provenance_hash=_compute_hash({
                    "indicator": indicator_id.value,
                    "current": current_val,
                    "previous": previous_val,
                }),
            )

        abs_change = current_val - previous_val
        pct_change = (
            (abs_change / previous_val * 100.0)
            if previous_val != 0.0
            else None
        )

        if abs(abs_change) < 1e-10:
            direction = "unchanged"
        elif higher_is_better:
            direction = "improved" if abs_change > 0 else "worsened"
        else:
            direction = "improved" if abs_change < 0 else "worsened"

        return PAIPeriodComparison(
            indicator_id=indicator_id,
            current_value=current_val,
            previous_value=previous_val,
            absolute_change=round(abs_change, 6),
            percentage_change=round(pct_change, 4) if pct_change is not None else None,
            direction=direction,
            provenance_hash=_compute_hash({
                "indicator": indicator_id.value,
                "current": current_val,
                "previous": previous_val,
                "change": abs_change,
            }),
        )

    # ------------------------------------------------------------------
    # Read-only Properties
    # ------------------------------------------------------------------

    @property
    def calculation_count(self) -> int:
        """Number of full PAI calculations performed since initialization."""
        return self._calculation_count

    @property
    def supported_indicators(self) -> List[str]:
        """List of all supported PAI indicator IDs."""
        return [p.value for p in PAIIndicatorId]

    @property
    def indicator_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Full metadata registry for all PAI indicators."""
        return {k.value: v for k, v in PAI_METADATA.items()}
