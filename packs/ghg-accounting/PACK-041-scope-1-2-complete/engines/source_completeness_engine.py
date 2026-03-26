# -*- coding: utf-8 -*-
"""
SourceCompletenessEngine - PACK-041 Scope 1-2 Complete Engine 2
================================================================

Source completeness and materiality assessment engine for GHG inventories.
Scans all applicable Scope 1 and Scope 2 emission source categories for a
given set of facilities, assesses materiality, checks data availability,
and identifies gaps to ensure the inventory meets GHG Protocol completeness
requirements (Chapter 1, Principle of Completeness).

Determines which of the 13 Scope 1-2 source categories apply to each
sector, benchmarks expected emissions, and flags missing or incomplete data
that could undermine the inventory's credibility.

Calculation Methodology:
    Category Applicability:
        For each sector, look up SECTOR_CATEGORY_MAP to determine which
        of the 13 source categories are applicable.

    Materiality Assessment:
        materiality_pct = category_emissions / total_estimated_emissions * 100
        If materiality_pct >= 1.0% -> MATERIAL
        If materiality_pct < 1.0% and not regulatory-required -> IMMATERIAL
        If category is legally required regardless of materiality -> REGULATORY_REQUIRED

    Completeness Score:
        completeness_pct = (categories_with_data / applicable_categories) * 100

    Coverage Ratio:
        coverage_ratio = reported_emissions / estimated_total_emissions

    Data Quality Score:
        quality_score = sum(category_quality_weight * category_completeness) /
                       sum(category_quality_weight)

Regulatory References:
    - GHG Protocol Corporate Standard (Revised), Chapter 1 (Completeness Principle)
    - GHG Protocol Corporate Standard, Chapter 4 (Setting Operational Boundaries)
    - GHG Protocol Scope 2 Guidance (2015)
    - ISO 14064-1:2018, Clause 5.2 (Quantification of GHG Emissions)
    - SEC Climate Disclosure Rule (2024) - Material emission sources
    - CSRD/ESRS E1 - Climate change disclosure requirements

Zero-Hallucination:
    - Sector-category mapping from published GHG Protocol guidance
    - Benchmark emission estimates from published sector averages
    - Deterministic arithmetic for all scoring and thresholds
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-041 Scope 1-2 Complete
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
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


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceCategory(str, Enum):
    """GHG Protocol Scope 1 and 2 source categories.

    STATIONARY_COMBUSTION: Boilers, furnaces, turbines, heaters, incinerators.
    MOBILE_COMBUSTION:     Company-owned vehicles and mobile equipment.
    PROCESS_EMISSIONS:     Chemical/physical transformations (cement, steel, etc.).
    FUGITIVE_EMISSIONS:    Leaks from equipment (valves, seals, pipelines).
    REFRIGERANT_LEAKAGE:   HFC/PFC leakage from HVAC and refrigeration.
    LAND_USE:              Land use and land use change emissions.
    WASTE_TREATMENT:       On-site waste treatment and disposal.
    AGRICULTURAL:          Agricultural emissions (enteric fermentation, soils).
    ELECTRICITY:           Purchased electricity (Scope 2).
    STEAM:                 Purchased steam (Scope 2).
    HEATING:               Purchased heating (Scope 2).
    COOLING:               Purchased cooling (Scope 2).
    DUAL_REPORTING:        Scope 2 dual reporting (location + market-based).
    """
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"
    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEATING = "heating"
    COOLING = "cooling"
    DUAL_REPORTING = "dual_reporting"


class MaterialityLevel(str, Enum):
    """Materiality classification for a source category.

    MATERIAL:            Contributes >= 1% of total emissions.
    IMMATERIAL:          Contributes < 1% and not regulatory-required.
    REGULATORY_REQUIRED: Must be reported regardless of materiality.
    UNKNOWN:             Insufficient data to assess materiality.
    """
    MATERIAL = "material"
    IMMATERIAL = "immaterial"
    REGULATORY_REQUIRED = "regulatory_required"
    UNKNOWN = "unknown"


class DataAvailability(str, Enum):
    """Data availability status for a source category.

    AVAILABLE:      Complete data available.
    PARTIAL:        Some data available but gaps exist.
    MISSING:        No data available.
    NOT_APPLICABLE: Category does not apply to this sector/facility.
    """
    AVAILABLE = "available"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


class GapSeverity(str, Enum):
    """Severity classification of a data gap.

    CRITICAL:  Gap in a material category; must be resolved for credible inventory.
    HIGH:      Gap in a regulatory-required category.
    MEDIUM:    Gap in a non-material but applicable category.
    LOW:       Gap in an immaterial or non-applicable category.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapResolutionAction(str, Enum):
    """Recommended action to resolve a data gap.

    COLLECT_PRIMARY:   Collect primary activity data from facilities.
    USE_PROXY:         Use proxy or estimated data.
    USE_BENCHMARK:     Use sector benchmark data.
    CONTACT_SUPPLIER:  Request data from energy supplier.
    INSTALL_METERING:  Install metering or monitoring equipment.
    REVIEW_RECORDS:    Review historical records (invoices, logs).
    NOT_REQUIRED:      No action needed.
    """
    COLLECT_PRIMARY = "collect_primary"
    USE_PROXY = "use_proxy"
    USE_BENCHMARK = "use_benchmark"
    CONTACT_SUPPLIER = "contact_supplier"
    INSTALL_METERING = "install_metering"
    REVIEW_RECORDS = "review_records"
    NOT_REQUIRED = "not_required"


class SectorType(str, Enum):
    """Industry sector classification for category mapping.

    These map to the standard sectors used in GHG Protocol guidance
    for determining applicable emission source categories.
    """
    MANUFACTURING = "manufacturing"
    ENERGY_POWER = "energy_power"
    OIL_GAS = "oil_gas"
    CHEMICALS = "chemicals"
    CEMENT = "cement"
    STEEL = "steel"
    MINING = "mining"
    TRANSPORT = "transport"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    REAL_ESTATE = "real_estate"
    COMMERCIAL_OFFICE = "commercial_office"
    RETAIL = "retail"
    HOSPITALITY = "hospitality"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    AGRICULTURE = "agriculture"
    FOOD_BEVERAGE = "food_beverage"
    FINANCIAL_SERVICES = "financial_services"
    TECHNOLOGY = "technology"
    TELECOMMUNICATIONS = "telecommunications"
    WASTE_MANAGEMENT = "waste_management"
    WATER_UTILITY = "water_utility"
    GOVERNMENT = "government"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Materiality threshold (percentage of total emissions).
MATERIALITY_THRESHOLD_PCT: Decimal = Decimal("1.0")

# Minimum completeness score to pass (percentage).
MINIMUM_COMPLETENESS_PCT: Decimal = Decimal("95.0")

# Sector-to-category applicability map.
# True = category is applicable/expected for that sector.
SECTOR_CATEGORY_MAP: Dict[str, Dict[str, bool]] = {
    SectorType.MANUFACTURING.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.ENERGY_POWER.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: False,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.OIL_GAS.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: False,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.CHEMICALS.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.CEMENT.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: False,
        SourceCategory.REFRIGERANT_LEAKAGE.value: False,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: False,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: False,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.STEEL.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: False,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: False,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.COMMERCIAL_OFFICE.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: False,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: False,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.RETAIL.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: False,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: False,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.HOSPITALITY.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: False,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.HEALTHCARE.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.TRANSPORT.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: False,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: False,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.AGRICULTURE.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: True,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: True,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.FOOD_BEVERAGE.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: True,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.FINANCIAL_SERVICES.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: False,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: False,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.TECHNOLOGY.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: False,
        SourceCategory.FUGITIVE_EMISSIONS.value: False,
        SourceCategory.REFRIGERANT_LEAKAGE.value: True,
        SourceCategory.LAND_USE.value: False,
        SourceCategory.WASTE_TREATMENT.value: False,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: True,
        SourceCategory.COOLING.value: True,
        SourceCategory.DUAL_REPORTING.value: True,
    },
    SectorType.WASTE_MANAGEMENT.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: True,
        SourceCategory.MOBILE_COMBUSTION.value: True,
        SourceCategory.PROCESS_EMISSIONS.value: True,
        SourceCategory.FUGITIVE_EMISSIONS.value: True,
        SourceCategory.REFRIGERANT_LEAKAGE.value: False,
        SourceCategory.LAND_USE.value: True,
        SourceCategory.WASTE_TREATMENT.value: True,
        SourceCategory.AGRICULTURAL.value: False,
        SourceCategory.ELECTRICITY.value: True,
        SourceCategory.STEAM.value: False,
        SourceCategory.HEATING.value: False,
        SourceCategory.COOLING.value: False,
        SourceCategory.DUAL_REPORTING.value: True,
    },
}

# Default category map for unlisted sectors -- include the most common categories.
DEFAULT_SECTOR_CATEGORIES: Dict[str, bool] = {
    SourceCategory.STATIONARY_COMBUSTION.value: True,
    SourceCategory.MOBILE_COMBUSTION.value: True,
    SourceCategory.PROCESS_EMISSIONS.value: False,
    SourceCategory.FUGITIVE_EMISSIONS.value: False,
    SourceCategory.REFRIGERANT_LEAKAGE.value: True,
    SourceCategory.LAND_USE.value: False,
    SourceCategory.WASTE_TREATMENT.value: False,
    SourceCategory.AGRICULTURAL.value: False,
    SourceCategory.ELECTRICITY.value: True,
    SourceCategory.STEAM.value: False,
    SourceCategory.HEATING.value: True,
    SourceCategory.COOLING.value: True,
    SourceCategory.DUAL_REPORTING.value: True,
}

# Sector benchmark emission intensity estimates (tCO2e per million USD revenue).
# Source: GHG Protocol and CDP sector benchmarks (representative averages).
SECTOR_BENCHMARK_INTENSITIES: Dict[str, Dict[str, Decimal]] = {
    SectorType.MANUFACTURING.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("85"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("12"),
        SourceCategory.PROCESS_EMISSIONS.value: Decimal("40"),
        SourceCategory.FUGITIVE_EMISSIONS.value: Decimal("5"),
        SourceCategory.REFRIGERANT_LEAKAGE.value: Decimal("3"),
        SourceCategory.WASTE_TREATMENT.value: Decimal("2"),
        SourceCategory.ELECTRICITY.value: Decimal("60"),
        SourceCategory.STEAM.value: Decimal("15"),
        SourceCategory.COOLING.value: Decimal("5"),
    },
    SectorType.COMMERCIAL_OFFICE.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("15"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("5"),
        SourceCategory.REFRIGERANT_LEAKAGE.value: Decimal("2"),
        SourceCategory.ELECTRICITY.value: Decimal("35"),
        SourceCategory.HEATING.value: Decimal("10"),
        SourceCategory.COOLING.value: Decimal("8"),
    },
    SectorType.TECHNOLOGY.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("8"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("3"),
        SourceCategory.REFRIGERANT_LEAKAGE.value: Decimal("4"),
        SourceCategory.ELECTRICITY.value: Decimal("55"),
        SourceCategory.HEATING.value: Decimal("5"),
        SourceCategory.COOLING.value: Decimal("12"),
    },
    SectorType.OIL_GAS.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("200"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("30"),
        SourceCategory.PROCESS_EMISSIONS.value: Decimal("120"),
        SourceCategory.FUGITIVE_EMISSIONS.value: Decimal("80"),
        SourceCategory.REFRIGERANT_LEAKAGE.value: Decimal("2"),
        SourceCategory.WASTE_TREATMENT.value: Decimal("5"),
        SourceCategory.ELECTRICITY.value: Decimal("45"),
        SourceCategory.STEAM.value: Decimal("20"),
    },
    SectorType.CEMENT.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("150"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("10"),
        SourceCategory.PROCESS_EMISSIONS.value: Decimal("350"),
        SourceCategory.ELECTRICITY.value: Decimal("70"),
    },
    SectorType.STEEL.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("180"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("8"),
        SourceCategory.PROCESS_EMISSIONS.value: Decimal("250"),
        SourceCategory.FUGITIVE_EMISSIONS.value: Decimal("10"),
        SourceCategory.WASTE_TREATMENT.value: Decimal("5"),
        SourceCategory.ELECTRICITY.value: Decimal("90"),
        SourceCategory.STEAM.value: Decimal("25"),
    },
    SectorType.AGRICULTURE.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("20"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("25"),
        SourceCategory.FUGITIVE_EMISSIONS.value: Decimal("5"),
        SourceCategory.REFRIGERANT_LEAKAGE.value: Decimal("3"),
        SourceCategory.LAND_USE.value: Decimal("30"),
        SourceCategory.WASTE_TREATMENT.value: Decimal("10"),
        SourceCategory.AGRICULTURAL.value: Decimal("120"),
        SourceCategory.ELECTRICITY.value: Decimal("18"),
        SourceCategory.COOLING.value: Decimal("5"),
    },
    SectorType.TRANSPORT.value: {
        SourceCategory.STATIONARY_COMBUSTION.value: Decimal("10"),
        SourceCategory.MOBILE_COMBUSTION.value: Decimal("180"),
        SourceCategory.FUGITIVE_EMISSIONS.value: Decimal("8"),
        SourceCategory.REFRIGERANT_LEAKAGE.value: Decimal("5"),
        SourceCategory.ELECTRICITY.value: Decimal("25"),
    },
}

# Categories that are always regulatory-required regardless of materiality
# (e.g. for SEC, CSRD, or ISO 14064-1).
REGULATORY_REQUIRED_CATEGORIES: set = {
    SourceCategory.STATIONARY_COMBUSTION.value,
    SourceCategory.ELECTRICITY.value,
    SourceCategory.DUAL_REPORTING.value,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class FacilityInfo(BaseModel):
    """Basic facility information for completeness scanning.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Human-readable name.
        sector: Industry sector.
        country: ISO 3166-1 alpha-2 country code.
        annual_revenue_usd: Annual revenue in USD (for intensity estimates).
        employee_count: Number of employees.
        floor_area_m2: Floor area in square metres.
        has_onsite_combustion: Whether facility has on-site boilers/furnaces.
        has_fleet_vehicles: Whether facility has owned vehicles.
        has_process_operations: Whether facility has process emissions.
        has_refrigeration: Whether facility has HVAC/refrigeration.
        has_onsite_waste_treatment: Whether facility treats waste on-site.
        has_agricultural_operations: Whether facility has agricultural ops.
        purchases_electricity: Whether facility purchases electricity.
        purchases_steam: Whether facility purchases steam.
        purchases_heating: Whether facility purchases district heating.
        purchases_cooling: Whether facility purchases district cooling.
        reported_emissions: Dict of category -> reported tCO2e (if available).
        data_sources: Dict of category -> list of data source names.
    """
    facility_id: str = Field(default_factory=_new_uuid, description="Facility ID")
    facility_name: str = Field(default="", max_length=300, description="Facility name")
    sector: str = Field(default=SectorType.OTHER.value, description="Industry sector")
    country: str = Field(default="", max_length=2, description="Country code")
    annual_revenue_usd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual revenue (USD)"
    )
    employee_count: int = Field(default=0, ge=0, description="Employee count")
    floor_area_m2: Decimal = Field(
        default=Decimal("0"), ge=0, description="Floor area (m2)"
    )
    has_onsite_combustion: bool = Field(default=False, description="On-site combustion")
    has_fleet_vehicles: bool = Field(default=False, description="Fleet vehicles")
    has_process_operations: bool = Field(default=False, description="Process operations")
    has_refrigeration: bool = Field(default=False, description="HVAC/refrigeration")
    has_onsite_waste_treatment: bool = Field(default=False, description="On-site waste")
    has_agricultural_operations: bool = Field(default=False, description="Agricultural ops")
    purchases_electricity: bool = Field(default=True, description="Purchases electricity")
    purchases_steam: bool = Field(default=False, description="Purchases steam")
    purchases_heating: bool = Field(default=False, description="Purchases heating")
    purchases_cooling: bool = Field(default=False, description="Purchases cooling")
    reported_emissions: Dict[str, Decimal] = Field(
        default_factory=dict, description="Category -> reported tCO2e"
    )
    data_sources: Dict[str, List[str]] = Field(
        default_factory=dict, description="Category -> data source names"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class CategoryAssessment(BaseModel):
    """Assessment of a single source category for one or more facilities.

    Attributes:
        category: The source category.
        is_applicable: Whether this category applies.
        estimated_emissions_tco2e: Benchmark estimated emissions.
        reported_emissions_tco2e: Actual reported emissions (if available).
        data_availability: Data availability status.
        data_sources: Names of available data sources.
        facility_count: Number of facilities where this category applies.
        notes: Assessment notes.
    """
    category: SourceCategory = Field(..., description="Source category")
    is_applicable: bool = Field(default=False, description="Category applicable")
    estimated_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Benchmark estimated emissions"
    )
    reported_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Reported emissions"
    )
    data_availability: DataAvailability = Field(
        default=DataAvailability.MISSING, description="Data availability"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data source names"
    )
    facility_count: int = Field(default=0, ge=0, description="Applicable facilities")
    notes: str = Field(default="", description="Notes")


class MaterialityAssessment(BaseModel):
    """Materiality assessment for a source category.

    Attributes:
        category: The source category.
        materiality_level: Materiality classification.
        materiality_pct: Percentage of total estimated emissions.
        estimated_emissions_tco2e: Estimated emissions for this category.
        total_estimated_emissions_tco2e: Total estimated emissions.
        is_regulatory_required: Whether this is regulatory-required.
        rationale: Explanation of materiality assessment.
    """
    category: SourceCategory = Field(..., description="Source category")
    materiality_level: MaterialityLevel = Field(
        default=MaterialityLevel.UNKNOWN, description="Materiality level"
    )
    materiality_pct: Decimal = Field(
        default=Decimal("0"), description="Materiality percentage"
    )
    estimated_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Estimated emissions"
    )
    total_estimated_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total estimated emissions"
    )
    is_regulatory_required: bool = Field(
        default=False, description="Regulatory-required"
    )
    rationale: str = Field(default="", description="Rationale")


class DataGap(BaseModel):
    """A data gap identified during completeness assessment.

    Attributes:
        gap_id: Unique gap ID.
        category: The source category with the gap.
        severity: Gap severity.
        description: Description of the gap.
        affected_facilities: List of affected facility IDs.
        estimated_missing_emissions_tco2e: Estimated emissions not captured.
        recommended_action: Recommended resolution action.
        resolution_deadline_days: Suggested deadline in days.
        notes: Additional notes.
    """
    gap_id: str = Field(default_factory=_new_uuid, description="Gap ID")
    category: SourceCategory = Field(..., description="Source category")
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM, description="Severity")
    description: str = Field(default="", description="Gap description")
    affected_facilities: List[str] = Field(
        default_factory=list, description="Affected facility IDs"
    )
    estimated_missing_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Estimated missing emissions"
    )
    recommended_action: GapResolutionAction = Field(
        default=GapResolutionAction.COLLECT_PRIMARY,
        description="Recommended action",
    )
    resolution_deadline_days: int = Field(
        default=30, ge=0, description="Resolution deadline (days)"
    )
    notes: str = Field(default="", description="Notes")


class CompletenessResult(BaseModel):
    """Overall completeness assessment result.

    Attributes:
        result_id: Unique result ID.
        category_assessments: Per-category assessments.
        materiality_assessments: Per-category materiality assessments.
        gaps: Identified data gaps.
        total_categories: Total categories evaluated.
        applicable_categories: Number of applicable categories.
        categories_with_data: Categories that have data.
        completeness_pct: Completeness percentage.
        coverage_ratio: Reported/estimated emissions ratio.
        total_estimated_emissions_tco2e: Total estimated emissions.
        total_reported_emissions_tco2e: Total reported emissions.
        passes_completeness: Whether completeness meets threshold.
        warnings: List of warnings.
        calculated_at: Timestamp.
        processing_time_ms: Processing time (ms).
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    category_assessments: List[CategoryAssessment] = Field(
        default_factory=list, description="Category assessments"
    )
    materiality_assessments: List[MaterialityAssessment] = Field(
        default_factory=list, description="Materiality assessments"
    )
    gaps: List[DataGap] = Field(default_factory=list, description="Data gaps")
    total_categories: int = Field(default=0, description="Total categories")
    applicable_categories: int = Field(default=0, description="Applicable categories")
    categories_with_data: int = Field(default=0, description="Categories with data")
    completeness_pct: Decimal = Field(
        default=Decimal("0"), description="Completeness %"
    )
    coverage_ratio: Decimal = Field(
        default=Decimal("0"), description="Coverage ratio"
    )
    total_estimated_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total estimated"
    )
    total_reported_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total reported"
    )
    passes_completeness: bool = Field(default=False, description="Passes threshold")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CompletenessReport(BaseModel):
    """Full completeness report combining all assessment outputs.

    Attributes:
        report_id: Unique report ID.
        completeness_result: The completeness result.
        summary_text: Human-readable summary.
        recommendations: List of recommendations.
        calculated_at: Timestamp.
        provenance_hash: SHA-256 hash.
    """
    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    completeness_result: Optional[CompletenessResult] = Field(
        default=None, description="Completeness result"
    )
    summary_text: str = Field(default="", description="Summary text")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

FacilityInfo.model_rebuild()
CategoryAssessment.model_rebuild()
MaterialityAssessment.model_rebuild()
DataGap.model_rebuild()
CompletenessResult.model_rebuild()
CompletenessReport.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SourceCompletenessEngine:
    """Source completeness and materiality assessment engine.

    Scans facilities against the 13 Scope 1-2 source categories, assesses
    materiality, checks data availability, identifies gaps, and produces a
    completeness report.

    Attributes:
        _assessments: Most recent category assessments.
        _materiality: Most recent materiality assessments.
        _gaps: Most recent gaps.
        _result: Most recent completeness result.

    Example:
        >>> engine = SourceCompletenessEngine()
        >>> facilities = [FacilityInfo(sector="manufacturing", ...)]
        >>> assessments = engine.scan_categories(facilities, "manufacturing")
        >>> result = engine.generate_completeness_report()
    """

    def __init__(self) -> None:
        """Initialise SourceCompletenessEngine."""
        self._assessments: List[CategoryAssessment] = []
        self._materiality: List[MaterialityAssessment] = []
        self._gaps: List[DataGap] = []
        self._result: Optional[CompletenessResult] = None
        self._warnings: List[str] = []
        logger.info(
            "SourceCompletenessEngine v%s initialised", _MODULE_VERSION
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_categories(
        self,
        facilities: List[FacilityInfo],
        sector: str,
    ) -> List[CategoryAssessment]:
        """Scan all source categories for applicability across facilities.

        Determines which of the 13 categories are applicable based on
        sector mapping and facility-level flags, estimates emissions from
        benchmarks, and checks data availability.

        Args:
            facilities: List of facility information.
            sector: Primary sector for the organisation.

        Returns:
            List of CategoryAssessment objects.

        Raises:
            ValueError: If no facilities are provided.
        """
        t0 = time.perf_counter()
        self._warnings = []

        if not facilities:
            raise ValueError("At least one facility must be provided")

        logger.info(
            "Scanning %d facilities for sector '%s'",
            len(facilities), sector,
        )

        # Get sector category map.
        cat_map = SECTOR_CATEGORY_MAP.get(sector, DEFAULT_SECTOR_CATEGORIES)

        assessments: List[CategoryAssessment] = []
        for cat in SourceCategory:
            is_applicable = cat_map.get(cat.value, False)

            # Override based on facility-level flags.
            is_applicable = self._refine_applicability(
                cat, is_applicable, facilities
            )

            # Estimate emissions from benchmarks.
            estimated = self._estimate_category_emissions(
                cat, sector, facilities
            )

            # Check reported data.
            reported = Decimal("0")
            sources: List[str] = []
            applicable_fac_count = 0
            for fac in facilities:
                if cat.value in fac.reported_emissions:
                    reported += _decimal(fac.reported_emissions[cat.value])
                if cat.value in fac.data_sources:
                    sources.extend(fac.data_sources[cat.value])
                if self._facility_has_category(cat, fac):
                    applicable_fac_count += 1

            # Determine data availability.
            if not is_applicable:
                availability = DataAvailability.NOT_APPLICABLE
            elif reported > Decimal("0") and sources:
                availability = DataAvailability.AVAILABLE
            elif reported > Decimal("0") or sources:
                availability = DataAvailability.PARTIAL
            else:
                availability = DataAvailability.MISSING

            assessment = CategoryAssessment(
                category=cat,
                is_applicable=is_applicable,
                estimated_emissions_tco2e=_round_val(estimated, 2),
                reported_emissions_tco2e=_round_val(reported, 2),
                data_availability=availability,
                data_sources=list(set(sources)),
                facility_count=applicable_fac_count,
                notes=self._build_category_notes(cat, is_applicable, availability),
            )
            assessments.append(assessment)

        self._assessments = assessments
        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        logger.info(
            "Category scan complete: %d applicable of %d total (%.1f ms)",
            sum(1 for a in assessments if a.is_applicable),
            len(assessments),
            float(elapsed),
        )
        return assessments

    def assess_materiality(
        self,
        categories: Optional[List[CategoryAssessment]] = None,
        estimated_emissions: Optional[Dict[str, Decimal]] = None,
    ) -> List[MaterialityAssessment]:
        """Assess materiality of each source category.

        Categories contributing >= 1% of total estimated emissions are
        classified as MATERIAL. Categories that are regulatory-required
        are classified as REGULATORY_REQUIRED regardless of materiality.

        Args:
            categories: Category assessments. Uses internal state if None.
            estimated_emissions: Optional override for estimated emissions.

        Returns:
            List of MaterialityAssessment objects.
        """
        cats = categories or self._assessments
        if not cats:
            raise ValueError(
                "No category assessments available. Call scan_categories() first."
            )

        logger.info("Assessing materiality for %d categories", len(cats))

        # Calculate total estimated emissions.
        total_estimated = Decimal("0")
        for cat in cats:
            if cat.is_applicable:
                override = (
                    _decimal(estimated_emissions.get(cat.category.value, 0))
                    if estimated_emissions else Decimal("0")
                )
                total_estimated += (
                    override if override > 0 else cat.estimated_emissions_tco2e
                )

        results: List[MaterialityAssessment] = []
        for cat in cats:
            if not cat.is_applicable:
                results.append(MaterialityAssessment(
                    category=cat.category,
                    materiality_level=MaterialityLevel.UNKNOWN,
                    materiality_pct=Decimal("0"),
                    estimated_emissions_tco2e=Decimal("0"),
                    total_estimated_emissions_tco2e=_round_val(total_estimated, 2),
                    is_regulatory_required=False,
                    rationale="Category not applicable to this sector.",
                ))
                continue

            override = (
                _decimal(estimated_emissions.get(cat.category.value, 0))
                if estimated_emissions else Decimal("0")
            )
            cat_emissions = (
                override if override > 0 else cat.estimated_emissions_tco2e
            )
            pct = _safe_pct(cat_emissions, total_estimated)
            is_regulatory = cat.category.value in REGULATORY_REQUIRED_CATEGORIES

            if is_regulatory:
                level = MaterialityLevel.REGULATORY_REQUIRED
            elif pct >= MATERIALITY_THRESHOLD_PCT:
                level = MaterialityLevel.MATERIAL
            elif total_estimated == Decimal("0"):
                level = MaterialityLevel.UNKNOWN
            else:
                level = MaterialityLevel.IMMATERIAL

            rationale = self._build_materiality_rationale(
                cat.category, level, pct, is_regulatory
            )

            results.append(MaterialityAssessment(
                category=cat.category,
                materiality_level=level,
                materiality_pct=_round_val(pct, 4),
                estimated_emissions_tco2e=_round_val(cat_emissions, 2),
                total_estimated_emissions_tco2e=_round_val(total_estimated, 2),
                is_regulatory_required=is_regulatory,
                rationale=rationale,
            ))

        self._materiality = results
        logger.info(
            "Materiality assessed: %d material, %d immaterial, %d regulatory",
            sum(1 for r in results if r.materiality_level == MaterialityLevel.MATERIAL),
            sum(1 for r in results if r.materiality_level == MaterialityLevel.IMMATERIAL),
            sum(1 for r in results
                if r.materiality_level == MaterialityLevel.REGULATORY_REQUIRED),
        )
        return results

    def check_data_availability(
        self,
        categories: Optional[List[CategoryAssessment]] = None,
        data_sources: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, DataAvailability]:
        """Check data availability for each category.

        Args:
            categories: Category assessments. Uses internal state if None.
            data_sources: Optional additional data sources per category.

        Returns:
            Dict mapping category name to DataAvailability.
        """
        cats = categories or self._assessments
        if not cats:
            raise ValueError("No category assessments available.")

        logger.info("Checking data availability for %d categories", len(cats))

        result: Dict[str, DataAvailability] = {}
        for cat in cats:
            if not cat.is_applicable:
                result[cat.category.value] = DataAvailability.NOT_APPLICABLE
                continue

            # Check if additional sources override.
            additional = data_sources.get(cat.category.value, []) if data_sources else []
            all_sources = cat.data_sources + additional

            if cat.reported_emissions_tco2e > Decimal("0") and all_sources:
                result[cat.category.value] = DataAvailability.AVAILABLE
            elif cat.reported_emissions_tco2e > Decimal("0") or all_sources:
                result[cat.category.value] = DataAvailability.PARTIAL
            else:
                result[cat.category.value] = DataAvailability.MISSING

        return result

    def identify_gaps(
        self,
        assessments: Optional[List[CategoryAssessment]] = None,
        materiality_results: Optional[List[MaterialityAssessment]] = None,
    ) -> List[DataGap]:
        """Identify data gaps in the inventory.

        A gap exists where an applicable category has MISSING or PARTIAL
        data availability.

        Args:
            assessments: Category assessments.  Uses internal state if None.
            materiality_results: Materiality assessments.  Uses internal if None.

        Returns:
            List of DataGap objects sorted by severity.
        """
        cats = assessments or self._assessments
        mats = materiality_results or self._materiality
        if not cats:
            raise ValueError("No category assessments available.")

        logger.info("Identifying data gaps")

        # Build materiality lookup.
        mat_lookup: Dict[str, MaterialityAssessment] = {}
        for m in mats:
            mat_lookup[m.category.value] = m

        gaps: List[DataGap] = []
        for cat in cats:
            if not cat.is_applicable:
                continue
            if cat.data_availability in (
                DataAvailability.AVAILABLE, DataAvailability.NOT_APPLICABLE
            ):
                continue

            # Determine severity.
            mat = mat_lookup.get(cat.category.value)
            severity = self._determine_gap_severity(cat, mat)

            # Determine recommended action.
            action = self._recommend_gap_action(cat)

            # Determine deadline.
            deadline = self._determine_deadline(severity)

            description = self._build_gap_description(cat, severity)

            gap = DataGap(
                category=cat.category,
                severity=severity,
                description=description,
                affected_facilities=[],  # Could be populated with facility IDs
                estimated_missing_emissions_tco2e=_round_val(
                    cat.estimated_emissions_tco2e - cat.reported_emissions_tco2e
                    if cat.estimated_emissions_tco2e > cat.reported_emissions_tco2e
                    else Decimal("0"), 2
                ),
                recommended_action=action,
                resolution_deadline_days=deadline,
            )
            gaps.append(gap)

        # Sort by severity: CRITICAL > HIGH > MEDIUM > LOW.
        severity_order = {
            GapSeverity.CRITICAL: 0,
            GapSeverity.HIGH: 1,
            GapSeverity.MEDIUM: 2,
            GapSeverity.LOW: 3,
        }
        gaps.sort(key=lambda g: severity_order.get(g.severity, 99))
        self._gaps = gaps

        logger.info(
            "Identified %d data gaps (%d critical, %d high)",
            len(gaps),
            sum(1 for g in gaps if g.severity == GapSeverity.CRITICAL),
            sum(1 for g in gaps if g.severity == GapSeverity.HIGH),
        )
        return gaps

    def generate_completeness_report(self) -> CompletenessReport:
        """Generate a full completeness report.

        Combines category assessments, materiality assessments, and
        data gaps into a comprehensive report.

        Returns:
            CompletenessReport with summary and recommendations.

        Raises:
            ValueError: If scan_categories has not been called.
        """
        if not self._assessments:
            raise ValueError(
                "No assessments available. Call scan_categories() first."
            )

        t0 = time.perf_counter()
        logger.info("Generating completeness report")

        # Ensure materiality and gaps have been computed.
        if not self._materiality:
            self.assess_materiality()
        if not self._gaps:
            self.identify_gaps()

        # Compute summary metrics.
        applicable = [a for a in self._assessments if a.is_applicable]
        with_data = [
            a for a in applicable
            if a.data_availability in (
                DataAvailability.AVAILABLE, DataAvailability.PARTIAL
            )
        ]
        total_estimated = sum(
            (a.estimated_emissions_tco2e for a in applicable), Decimal("0")
        )
        total_reported = sum(
            (a.reported_emissions_tco2e for a in applicable), Decimal("0")
        )
        completeness_pct = _safe_pct(
            _decimal(len(with_data)), _decimal(len(applicable))
        )
        coverage = _safe_divide(total_reported, total_estimated)
        passes = completeness_pct >= MINIMUM_COMPLETENESS_PCT

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))

        result = CompletenessResult(
            category_assessments=self._assessments,
            materiality_assessments=self._materiality,
            gaps=self._gaps,
            total_categories=len(self._assessments),
            applicable_categories=len(applicable),
            categories_with_data=len(with_data),
            completeness_pct=_round_val(completeness_pct, 2),
            coverage_ratio=_round_val(coverage, 4),
            total_estimated_emissions_tco2e=_round_val(total_estimated, 2),
            total_reported_emissions_tco2e=_round_val(total_reported, 2),
            passes_completeness=passes,
            warnings=list(self._warnings),
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result)
        self._result = result

        # Build report.
        summary_parts: List[str] = [
            "Source Completeness Assessment Report",
            f"Total categories: {result.total_categories}",
            f"Applicable categories: {result.applicable_categories}",
            f"Categories with data: {result.categories_with_data}",
            f"Completeness: {result.completeness_pct}%",
            f"Coverage ratio: {result.coverage_ratio}",
            f"Estimated total: {result.total_estimated_emissions_tco2e} tCO2e",
            f"Reported total: {result.total_reported_emissions_tco2e} tCO2e",
            f"Data gaps: {len(self._gaps)}",
            f"Passes completeness threshold ({MINIMUM_COMPLETENESS_PCT}%): "
            f"{'Yes' if passes else 'No'}",
        ]
        recommendations = self._build_recommendations()

        report = CompletenessReport(
            completeness_result=result,
            summary_text="\n".join(summary_parts),
            recommendations=recommendations,
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Completeness report generated: %.1f%% complete, %d gaps",
            float(completeness_pct), len(self._gaps),
        )
        return report

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _refine_applicability(
        self,
        category: SourceCategory,
        sector_applicable: bool,
        facilities: List[FacilityInfo],
    ) -> bool:
        """Refine category applicability using facility-level flags.

        Facility flags can override sector defaults when specific
        operational characteristics are known.

        Args:
            category: The source category.
            sector_applicable: Whether the sector map says applicable.
            facilities: List of facilities.

        Returns:
            Refined applicability.
        """
        if category == SourceCategory.STATIONARY_COMBUSTION:
            if any(f.has_onsite_combustion for f in facilities):
                return True
        elif category == SourceCategory.MOBILE_COMBUSTION:
            if any(f.has_fleet_vehicles for f in facilities):
                return True
        elif category == SourceCategory.PROCESS_EMISSIONS:
            if any(f.has_process_operations for f in facilities):
                return True
        elif category == SourceCategory.REFRIGERANT_LEAKAGE:
            if any(f.has_refrigeration for f in facilities):
                return True
        elif category == SourceCategory.WASTE_TREATMENT:
            if any(f.has_onsite_waste_treatment for f in facilities):
                return True
        elif category == SourceCategory.AGRICULTURAL:
            if any(f.has_agricultural_operations for f in facilities):
                return True
        elif category == SourceCategory.ELECTRICITY:
            if any(f.purchases_electricity for f in facilities):
                return True
        elif category == SourceCategory.STEAM:
            if any(f.purchases_steam for f in facilities):
                return True
        elif category == SourceCategory.HEATING:
            if any(f.purchases_heating for f in facilities):
                return True
        elif category == SourceCategory.COOLING:
            if any(f.purchases_cooling for f in facilities):
                return True
        elif category == SourceCategory.DUAL_REPORTING:
            # Dual reporting applies whenever electricity is purchased.
            if any(f.purchases_electricity for f in facilities):
                return True

        return sector_applicable

    def _facility_has_category(
        self, category: SourceCategory, facility: FacilityInfo
    ) -> bool:
        """Check if a specific facility has a given category.

        Args:
            category: The source category.
            facility: The facility.

        Returns:
            True if the facility has the category.
        """
        mapping: Dict[str, str] = {
            SourceCategory.STATIONARY_COMBUSTION.value: "has_onsite_combustion",
            SourceCategory.MOBILE_COMBUSTION.value: "has_fleet_vehicles",
            SourceCategory.PROCESS_EMISSIONS.value: "has_process_operations",
            SourceCategory.REFRIGERANT_LEAKAGE.value: "has_refrigeration",
            SourceCategory.WASTE_TREATMENT.value: "has_onsite_waste_treatment",
            SourceCategory.AGRICULTURAL.value: "has_agricultural_operations",
            SourceCategory.ELECTRICITY.value: "purchases_electricity",
            SourceCategory.STEAM.value: "purchases_steam",
            SourceCategory.HEATING.value: "purchases_heating",
            SourceCategory.COOLING.value: "purchases_cooling",
        }
        attr = mapping.get(category.value)
        if attr and hasattr(facility, attr):
            return getattr(facility, attr, False)
        return False

    def _estimate_category_emissions(
        self,
        category: SourceCategory,
        sector: str,
        facilities: List[FacilityInfo],
    ) -> Decimal:
        """Estimate emissions for a category using sector benchmarks.

        Uses the sector benchmark intensity (tCO2e per million USD revenue)
        multiplied by aggregate revenue.

        Args:
            category: The source category.
            sector: Industry sector.
            facilities: List of facilities.

        Returns:
            Estimated emissions (tCO2e).
        """
        benchmarks = SECTOR_BENCHMARK_INTENSITIES.get(sector, {})
        intensity = benchmarks.get(category.value, Decimal("0"))
        if intensity == Decimal("0"):
            return Decimal("0")

        total_revenue_m = _safe_divide(
            sum((f.annual_revenue_usd for f in facilities), Decimal("0")),
            Decimal("1000000"),
        )
        return intensity * total_revenue_m

    def _build_category_notes(
        self,
        category: SourceCategory,
        is_applicable: bool,
        availability: DataAvailability,
    ) -> str:
        """Build notes for a category assessment.

        Args:
            category: The source category.
            is_applicable: Whether applicable.
            availability: Data availability.

        Returns:
            Notes string.
        """
        if not is_applicable:
            return f"{category.value} is not applicable to this sector/facilities."
        if availability == DataAvailability.AVAILABLE:
            return f"{category.value}: complete data available."
        if availability == DataAvailability.PARTIAL:
            return f"{category.value}: partial data available; gaps exist."
        return f"{category.value}: no data available; collection required."

    def _build_materiality_rationale(
        self,
        category: SourceCategory,
        level: MaterialityLevel,
        pct: Decimal,
        is_regulatory: bool,
    ) -> str:
        """Build rationale text for a materiality assessment.

        Args:
            category: The source category.
            level: Materiality level.
            pct: Materiality percentage.
            is_regulatory: Whether regulatory-required.

        Returns:
            Rationale string.
        """
        if is_regulatory:
            return (
                f"{category.value} is regulatory-required and must be reported "
                f"regardless of materiality ({pct}% of estimated total)."
            )
        if level == MaterialityLevel.MATERIAL:
            return (
                f"{category.value} is material at {pct}% of estimated total "
                f"(threshold: {MATERIALITY_THRESHOLD_PCT}%)."
            )
        if level == MaterialityLevel.IMMATERIAL:
            return (
                f"{category.value} is immaterial at {pct}% of estimated total "
                f"(below {MATERIALITY_THRESHOLD_PCT}% threshold). "
                f"Disclosure recommended but not required for completeness."
            )
        return f"{category.value}: insufficient data to assess materiality."

    def _determine_gap_severity(
        self,
        cat: CategoryAssessment,
        mat: Optional[MaterialityAssessment],
    ) -> GapSeverity:
        """Determine severity of a data gap.

        Args:
            cat: The category assessment.
            mat: The materiality assessment (if available).

        Returns:
            Gap severity.
        """
        if mat and mat.is_regulatory_required:
            return GapSeverity.HIGH
        if mat and mat.materiality_level == MaterialityLevel.MATERIAL:
            return GapSeverity.CRITICAL
        if cat.data_availability == DataAvailability.MISSING:
            return GapSeverity.MEDIUM
        return GapSeverity.LOW

    def _recommend_gap_action(self, cat: CategoryAssessment) -> GapResolutionAction:
        """Recommend an action to resolve a data gap.

        Args:
            cat: The category assessment.

        Returns:
            Recommended action.
        """
        scope2_cats = {
            SourceCategory.ELECTRICITY.value,
            SourceCategory.STEAM.value,
            SourceCategory.HEATING.value,
            SourceCategory.COOLING.value,
            SourceCategory.DUAL_REPORTING.value,
        }
        if cat.category.value in scope2_cats:
            if cat.data_availability == DataAvailability.MISSING:
                return GapResolutionAction.CONTACT_SUPPLIER
            return GapResolutionAction.REVIEW_RECORDS

        combustion_cats = {
            SourceCategory.STATIONARY_COMBUSTION.value,
            SourceCategory.MOBILE_COMBUSTION.value,
        }
        if cat.category.value in combustion_cats:
            return GapResolutionAction.REVIEW_RECORDS

        if cat.category.value == SourceCategory.REFRIGERANT_LEAKAGE.value:
            return GapResolutionAction.COLLECT_PRIMARY

        if cat.category.value == SourceCategory.FUGITIVE_EMISSIONS.value:
            return GapResolutionAction.INSTALL_METERING

        return GapResolutionAction.COLLECT_PRIMARY

    def _determine_deadline(self, severity: GapSeverity) -> int:
        """Determine resolution deadline based on severity.

        Args:
            severity: Gap severity.

        Returns:
            Deadline in days.
        """
        deadlines: Dict[GapSeverity, int] = {
            GapSeverity.CRITICAL: 14,
            GapSeverity.HIGH: 30,
            GapSeverity.MEDIUM: 60,
            GapSeverity.LOW: 90,
        }
        return deadlines.get(severity, 60)

    def _build_gap_description(
        self, cat: CategoryAssessment, severity: GapSeverity
    ) -> str:
        """Build description for a data gap.

        Args:
            cat: The category assessment.
            severity: Gap severity.

        Returns:
            Description string.
        """
        status = (
            "No data reported" if cat.data_availability == DataAvailability.MISSING
            else "Partial data only"
        )
        return (
            f"{status} for {cat.category.value}. "
            f"Estimated emissions: {cat.estimated_emissions_tco2e} tCO2e. "
            f"Severity: {severity.value}. "
            f"Applicable to {cat.facility_count} facility(ies)."
        )

    def _build_recommendations(self) -> List[str]:
        """Build a list of recommendations based on current state.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        critical_gaps = [
            g for g in self._gaps if g.severity == GapSeverity.CRITICAL
        ]
        if critical_gaps:
            recs.append(
                f"CRITICAL: {len(critical_gaps)} material source category(ies) "
                f"have missing or incomplete data. Resolve within 14 days to "
                f"ensure inventory credibility."
            )
            for gap in critical_gaps:
                recs.append(
                    f"  - {gap.category.value}: {gap.recommended_action.value}"
                )

        high_gaps = [g for g in self._gaps if g.severity == GapSeverity.HIGH]
        if high_gaps:
            recs.append(
                f"HIGH: {len(high_gaps)} regulatory-required category(ies) "
                f"have data gaps. Resolve within 30 days."
            )

        if self._result and not self._result.passes_completeness:
            recs.append(
                f"Completeness score ({self._result.completeness_pct}%) is "
                f"below the {MINIMUM_COMPLETENESS_PCT}% threshold. "
                f"Address data gaps to meet GHG Protocol completeness principle."
            )

        if not recs:
            recs.append(
                "No critical or high-severity gaps identified. "
                "Inventory meets completeness requirements."
            )

        return recs
