# -*- coding: utf-8 -*-
"""
IndirectEmissionsCalculatorEngine - Engine 3: Use of Sold Products (AGENT-MRV-024)

This module implements the IndirectEmissionsCalculatorEngine for AGENT-MRV-024
(Use of Sold Products, GHG Protocol Scope 3 Category 11). It provides thread-safe
singleton calculations for indirect use-phase emissions from electricity consumption,
heating fuel, and steam/cooling consumed by products during their use phase.

Calculation Formulae (GHG Protocol Scope 3, Chapter 6):

    Formula D -- Electricity Consumption:
        E_indirect_elec = SUM_i( Q_sold_i x L_i x AE_i x EF_grid )
        where:
            Q_sold_i  = units of product i sold in reporting period
            L_i       = expected lifetime of product i (years)
            AE_i      = annual electricity consumption (kWh/year)
            EF_grid   = grid emission factor for use-region (kgCO2e/kWh)

    Formula E -- Heating Fuel:
        E_indirect_heat = SUM_i( Q_sold_i x L_i x AF_heat_i x EF_fuel )
        where:
            AF_heat_i = annual heating fuel consumption (liters/year or m3/year)
            EF_fuel   = combustion emission factor for fuel type (kgCO2e/unit)

    Formula F -- Steam/Cooling:
        E_indirect_steam = SUM_i( Q_sold_i x L_i x AS_i x EF_steam )
        where:
            AS_i      = annual steam/cooling consumption (MJ/year)
            EF_steam  = emission factor for steam/cooling (kgCO2e/MJ)

Energy degradation is applied year-by-year using the formula:
    consumption_year_t = base_consumption x (1 - degradation_rate) ^ t

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal with ROUND_HALF_UP to 8 decimal places
    - No LLM calls anywhere in the numeric calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from IEA, DEFRA 2024, EPA eGRID

Supports:
    - 16 regional grid emission factors (US, GB, DE, FR, CN, IN, JP, KR, BR, CA,
      AU, MX, IT, ES, PL, GLOBAL)
    - 15 heating fuel types with combustion emission factors
    - 4 steam/cooling source types with factors
    - Energy degradation curves per product category (0.0-2.0% per year)
    - Year-by-year lifetime energy consumption modeling
    - 5-dimension Data Quality Indicators (DQI)
    - Uncertainty quantification per method
    - Full SHA-256 provenance tracking

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation. All mutable state is protected by dedicated locks.

Example:
    >>> from greenlang.agents.mrv.use_of_sold_products.indirect_emissions_calculator import (
    ...     IndirectEmissionsCalculatorEngine,
    ... )
    >>> engine = IndirectEmissionsCalculatorEngine()
    >>> products = [IndirectProductInput(
    ...     product_id="REF-001",
    ...     product_type="refrigerator",
    ...     category="APPLIANCES",
    ...     units_sold=5000,
    ...     lifetime_years=15,
    ...     annual_energy_kwh=Decimal("400"),
    ...     use_region="US",
    ... )]
    >>> result = engine.calculate_electricity(products, "ORG-001", 2024)
    >>> assert result["total_co2e"] > Decimal("0")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-011
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_usp_"
ENGINE_NAME: str = "IndirectEmissionsCalculatorEngine"
ENGINE_NUMBER: int = 3

# ==============================================================================
# DECIMAL PRECISION CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_QUANT_8DP = _PRECISION


def _q(value: Decimal) -> Decimal:
    """
    Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with exactly 8 decimal places.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class IndirectEmissionMethod(str, Enum):
    """Calculation methods for indirect use-phase emissions."""

    ELECTRICITY = "electricity"
    HEATING = "heating"
    STEAM_COOLING = "steam_cooling"


class GridRegion(str, Enum):
    """Electricity grid regions for emission factor lookup."""

    US = "US"
    GB = "GB"
    DE = "DE"
    FR = "FR"
    CN = "CN"
    IN = "IN"
    JP = "JP"
    KR = "KR"
    BR = "BR"
    CA = "CA"
    AU = "AU"
    MX = "MX"
    IT = "IT"
    ES = "ES"
    PL = "PL"
    GLOBAL = "GLOBAL"


class HeatingFuelType(str, Enum):
    """Heating fuel types with combustion emission factors."""

    NATURAL_GAS = "NATURAL_GAS"
    LPG = "LPG"
    KEROSENE = "KEROSENE"
    HFO = "HFO"
    DIESEL = "DIESEL"
    PROPANE = "PROPANE"
    COAL = "COAL"
    WOOD_PELLETS = "WOOD_PELLETS"
    ETHANOL = "ETHANOL"
    BIODIESEL = "BIODIESEL"
    CNG = "CNG"
    LNG = "LNG"
    HYDROGEN = "HYDROGEN"
    GASOLINE = "GASOLINE"
    JET_FUEL = "JET_FUEL"


class SteamCoolingSource(str, Enum):
    """Steam and cooling energy source types."""

    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    INDUSTRIAL_STEAM = "industrial_steam"
    CHP_STEAM = "chp_steam"


class ProductCategory(str, Enum):
    """Product categories for use-phase calculations."""

    VEHICLES = "VEHICLES"
    APPLIANCES = "APPLIANCES"
    HVAC = "HVAC"
    LIGHTING = "LIGHTING"
    IT_EQUIPMENT = "IT_EQUIPMENT"
    INDUSTRIAL_EQUIPMENT = "INDUSTRIAL_EQUIPMENT"
    BUILDING_PRODUCTS = "BUILDING_PRODUCTS"
    CONSUMER_PRODUCTS = "CONSUMER_PRODUCTS"
    MEDICAL_DEVICES = "MEDICAL_DEVICES"
    FUELS_FEEDSTOCKS = "FUELS_FEEDSTOCKS"


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges."""

    TIER_1 = "tier_1"  # Product-specific, primary data
    TIER_2 = "tier_2"  # Category average, secondary data
    TIER_3 = "tier_3"  # Global average, estimated data


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method."""

    ANALYTICAL = "analytical"
    MONTE_CARLO = "monte_carlo"
    IPCC_TIER_2 = "ipcc_tier_2"


# ==============================================================================
# EMISSION FACTOR TABLES
# ==============================================================================

# Grid emission factors (kgCO2e/kWh) by region -- IEA 2023 / EPA eGRID 2022
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.417"),
    "GB": Decimal("0.233"),
    "DE": Decimal("0.348"),
    "FR": Decimal("0.052"),
    "CN": Decimal("0.555"),
    "IN": Decimal("0.708"),
    "JP": Decimal("0.462"),
    "KR": Decimal("0.424"),
    "BR": Decimal("0.075"),
    "CA": Decimal("0.120"),
    "AU": Decimal("0.656"),
    "MX": Decimal("0.431"),
    "IT": Decimal("0.256"),
    "ES": Decimal("0.175"),
    "PL": Decimal("0.635"),
    "GLOBAL": Decimal("0.475"),
}

# Heating fuel combustion emission factors (kgCO2e per unit -- liters, m3, or kg)
HEATING_FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "NATURAL_GAS": {
        "ef": Decimal("2.024"),
        "unit": Decimal("1"),  # per m3
        "ncv_mj": Decimal("38.3"),
    },
    "LPG": {
        "ef": Decimal("1.557"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("26.1"),
    },
    "KEROSENE": {
        "ef": Decimal("2.541"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("37.0"),
    },
    "HFO": {
        "ef": Decimal("3.114"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("40.4"),
    },
    "DIESEL": {
        "ef": Decimal("2.706"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("38.6"),
    },
    "PROPANE": {
        "ef": Decimal("1.530"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("25.3"),
    },
    "COAL": {
        "ef": Decimal("2.883"),
        "unit": Decimal("1"),  # per kg
        "ncv_mj": Decimal("25.8"),
    },
    "WOOD_PELLETS": {
        "ef": Decimal("0.015"),
        "unit": Decimal("1"),  # per kg
        "ncv_mj": Decimal("17.0"),
    },
    "ETHANOL": {
        "ef": Decimal("0.020"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("26.7"),
    },
    "BIODIESEL": {
        "ef": Decimal("0.015"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("37.0"),
    },
    "CNG": {
        "ef": Decimal("2.024"),
        "unit": Decimal("1"),  # per m3
        "ncv_mj": Decimal("38.3"),
    },
    "LNG": {
        "ef": Decimal("2.750"),
        "unit": Decimal("1"),  # per kg
        "ncv_mj": Decimal("49.5"),
    },
    "HYDROGEN": {
        "ef": Decimal("0.000"),
        "unit": Decimal("1"),  # per kg (zero direct combustion emissions)
        "ncv_mj": Decimal("120.0"),
    },
    "GASOLINE": {
        "ef": Decimal("2.315"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("34.2"),
    },
    "JET_FUEL": {
        "ef": Decimal("2.548"),
        "unit": Decimal("1"),  # per liter
        "ncv_mj": Decimal("37.4"),
    },
}

# Steam and cooling emission factors (kgCO2e/MJ)
STEAM_COOLING_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "district_heating": {
        "ef": Decimal("0.0650"),
        "description_factor": Decimal("1"),
    },
    "district_cooling": {
        "ef": Decimal("0.0480"),
        "description_factor": Decimal("1"),
    },
    "industrial_steam": {
        "ef": Decimal("0.0720"),
        "description_factor": Decimal("1"),
    },
    "chp_steam": {
        "ef": Decimal("0.0550"),
        "description_factor": Decimal("1"),
    },
}

# Energy degradation rates by product category (annual % efficiency loss)
DEGRADATION_RATES: Dict[str, Decimal] = {
    "VEHICLES": Decimal("0.015"),
    "APPLIANCES": Decimal("0.005"),
    "HVAC": Decimal("0.010"),
    "LIGHTING": Decimal("0.020"),
    "IT_EQUIPMENT": Decimal("0.000"),
    "INDUSTRIAL_EQUIPMENT": Decimal("0.010"),
    "BUILDING_PRODUCTS": Decimal("0.003"),
    "CONSUMER_PRODUCTS": Decimal("0.000"),
    "MEDICAL_DEVICES": Decimal("0.002"),
    "FUELS_FEEDSTOCKS": Decimal("0.000"),
}

# DQI base scores by method (out of 100)
DQI_BASE_SCORES: Dict[str, Decimal] = {
    "electricity": Decimal("80"),
    "heating": Decimal("75"),
    "steam_cooling": Decimal("70"),
}

# Uncertainty half-width (95% CI) by method
UNCERTAINTY_HALFWIDTHS: Dict[str, Decimal] = {
    "electricity": Decimal("0.20"),     # +/- 20%
    "heating": Decimal("0.25"),         # +/- 25%
    "steam_cooling": Decimal("0.30"),   # +/- 30%
}

# DQI dimension weights (sum to 1.0)
DQI_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    "reliability": Decimal("0.25"),
    "completeness": Decimal("0.25"),
    "temporal": Decimal("0.20"),
    "geographical": Decimal("0.15"),
    "technological": Decimal("0.15"),
}

# DQI tier scoring multipliers
DQI_TIER_MULTIPLIERS: Dict[str, Decimal] = {
    "tier_1": Decimal("1.00"),
    "tier_2": Decimal("0.80"),
    "tier_3": Decimal("0.60"),
}


# ==============================================================================
# INPUT / OUTPUT DATA STRUCTURES
# ==============================================================================


class IndirectProductInput:
    """
    Input data for a single product indirect emissions calculation.

    Represents one product line (or SKU) with its sales volume, lifetime,
    and annual energy consumption characteristics. All numeric fields use
    Decimal for regulatory precision.

    Attributes:
        product_id: Unique product identifier.
        product_type: Specific product type (e.g., "refrigerator", "laptop").
        category: Product category from ProductCategory enum.
        units_sold: Number of units sold in reporting period.
        lifetime_years: Expected product lifetime in years.
        annual_energy_kwh: Annual electricity consumption (kWh/year). Used
            for electricity method.
        annual_fuel_consumption: Annual heating fuel consumption (liters/year
            or m3/year). Used for heating method.
        fuel_type: Heating fuel type code. Required for heating method.
        annual_steam_mj: Annual steam/cooling consumption (MJ/year). Used
            for steam/cooling method.
        steam_source: Steam/cooling source type. Required for steam method.
        use_region: Grid region code for electricity EF lookup.
        degradation_rate: Optional override for annual efficiency degradation.
        data_quality_tier: Data quality tier for DQI scoring.
        tenant_id: Tenant identifier for multi-tenancy.
    """

    __slots__ = (
        "product_id",
        "product_type",
        "category",
        "units_sold",
        "lifetime_years",
        "annual_energy_kwh",
        "annual_fuel_consumption",
        "fuel_type",
        "annual_steam_mj",
        "steam_source",
        "use_region",
        "degradation_rate",
        "data_quality_tier",
        "tenant_id",
    )

    def __init__(
        self,
        product_id: str,
        product_type: str,
        category: str,
        units_sold: int,
        lifetime_years: int,
        annual_energy_kwh: Optional[Decimal] = None,
        annual_fuel_consumption: Optional[Decimal] = None,
        fuel_type: Optional[str] = None,
        annual_steam_mj: Optional[Decimal] = None,
        steam_source: Optional[str] = None,
        use_region: str = "GLOBAL",
        degradation_rate: Optional[Decimal] = None,
        data_quality_tier: str = "tier_2",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize IndirectProductInput with validated fields."""
        self.product_id = product_id
        self.product_type = product_type
        self.category = category
        self.units_sold = units_sold
        self.lifetime_years = lifetime_years
        self.annual_energy_kwh = annual_energy_kwh
        self.annual_fuel_consumption = annual_fuel_consumption
        self.fuel_type = fuel_type
        self.annual_steam_mj = annual_steam_mj
        self.steam_source = steam_source
        self.use_region = use_region
        self.degradation_rate = degradation_rate
        self.data_quality_tier = data_quality_tier
        self.tenant_id = tenant_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for hashing and provenance."""
        return {
            "product_id": self.product_id,
            "product_type": self.product_type,
            "category": self.category,
            "units_sold": self.units_sold,
            "lifetime_years": self.lifetime_years,
            "annual_energy_kwh": str(self.annual_energy_kwh) if self.annual_energy_kwh is not None else None,
            "annual_fuel_consumption": str(self.annual_fuel_consumption) if self.annual_fuel_consumption is not None else None,
            "fuel_type": self.fuel_type,
            "annual_steam_mj": str(self.annual_steam_mj) if self.annual_steam_mj is not None else None,
            "steam_source": self.steam_source,
            "use_region": self.use_region,
            "degradation_rate": str(self.degradation_rate) if self.degradation_rate is not None else None,
            "data_quality_tier": self.data_quality_tier,
            "tenant_id": self.tenant_id,
        }


class IndirectEmissionResult:
    """
    Result from an indirect emissions calculation for a single product.

    All numeric values are Decimal with 8-decimal-place precision.

    Attributes:
        product_id: Product identifier.
        product_type: Product type string.
        category: Product category code.
        method: Calculation method used (electricity, heating, steam_cooling).
        units_sold: Units sold in reporting period.
        lifetime_years: Lifetime used in calculation.
        annual_consumption: Annual energy consumption (kWh, liters, or MJ).
        total_lifetime_consumption: Total consumption over lifetime with degradation.
        emission_factor: Emission factor applied.
        ef_unit: Emission factor unit description.
        co2e_per_unit: Emissions per product unit (kgCO2e).
        total_co2e: Total emissions for all units (kgCO2e).
        degradation_rate: Degradation rate applied.
        dqi_score: Data quality indicator score (0-100).
        uncertainty_lower: Lower bound of 95% confidence interval.
        uncertainty_upper: Upper bound of 95% confidence interval.
        provenance_hash: SHA-256 provenance hash.
    """

    __slots__ = (
        "product_id",
        "product_type",
        "category",
        "method",
        "units_sold",
        "lifetime_years",
        "annual_consumption",
        "total_lifetime_consumption",
        "emission_factor",
        "ef_unit",
        "co2e_per_unit",
        "total_co2e",
        "degradation_rate",
        "dqi_score",
        "uncertainty_lower",
        "uncertainty_upper",
        "provenance_hash",
    )

    def __init__(
        self,
        product_id: str,
        product_type: str,
        category: str,
        method: str,
        units_sold: int,
        lifetime_years: int,
        annual_consumption: Decimal,
        total_lifetime_consumption: Decimal,
        emission_factor: Decimal,
        ef_unit: str,
        co2e_per_unit: Decimal,
        total_co2e: Decimal,
        degradation_rate: Decimal,
        dqi_score: Decimal,
        uncertainty_lower: Decimal,
        uncertainty_upper: Decimal,
        provenance_hash: str,
    ) -> None:
        """Initialize IndirectEmissionResult."""
        self.product_id = product_id
        self.product_type = product_type
        self.category = category
        self.method = method
        self.units_sold = units_sold
        self.lifetime_years = lifetime_years
        self.annual_consumption = annual_consumption
        self.total_lifetime_consumption = total_lifetime_consumption
        self.emission_factor = emission_factor
        self.ef_unit = ef_unit
        self.co2e_per_unit = co2e_per_unit
        self.total_co2e = total_co2e
        self.degradation_rate = degradation_rate
        self.dqi_score = dqi_score
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "product_id": self.product_id,
            "product_type": self.product_type,
            "category": self.category,
            "method": self.method,
            "units_sold": self.units_sold,
            "lifetime_years": self.lifetime_years,
            "annual_consumption": str(self.annual_consumption),
            "total_lifetime_consumption": str(self.total_lifetime_consumption),
            "emission_factor": str(self.emission_factor),
            "ef_unit": self.ef_unit,
            "co2e_per_unit": str(self.co2e_per_unit),
            "total_co2e": str(self.total_co2e),
            "degradation_rate": str(self.degradation_rate),
            "dqi_score": str(self.dqi_score),
            "uncertainty_lower": str(self.uncertainty_lower),
            "uncertainty_upper": str(self.uncertainty_upper),
            "provenance_hash": self.provenance_hash,
        }


# ==============================================================================
# HASH UTILITY
# ==============================================================================


def _compute_provenance_hash(*parts: Any) -> str:
    """
    Compute SHA-256 provenance hash from variable inputs.

    Serializes each part to a deterministic string representation and
    concatenates them before hashing. Handles Decimal, dict, list,
    and arbitrary objects.

    Args:
        *parts: Variable number of inputs to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).
    """
    hash_input = ""
    for part in parts:
        if isinstance(part, Decimal):
            hash_input += str(part.quantize(_PRECISION, rounding=ROUND_HALF_UP))
        elif isinstance(part, dict):
            hash_input += json.dumps(part, sort_keys=True, default=str)
        elif isinstance(part, (list, tuple)):
            hash_input += json.dumps(
                [str(x) if isinstance(x, Decimal) else x for x in part],
                sort_keys=True,
                default=str,
            )
        elif hasattr(part, "to_dict"):
            hash_input += json.dumps(part.to_dict(), sort_keys=True, default=str)
        else:
            hash_input += str(part)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class IndirectEmissionsCalculatorEngine:
    """
    Thread-safe singleton engine for indirect use-phase emissions calculations.

    Implements the complete indirect emissions calculation pipeline per GHG
    Protocol Scope 3 Category 11 for three energy types: electricity, heating
    fuel, and steam/cooling. All arithmetic uses Python Decimal with
    ROUND_HALF_UP quantization to 8 decimal places for regulatory precision.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic. No LLM calls are used for any numeric computation.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The
        _calculation_count attribute is protected by a dedicated lock.

    Attributes:
        _calculation_count: Total number of calculations performed.
        _count_lock: Lock protecting the calculation counter.

    Example:
        >>> engine = IndirectEmissionsCalculatorEngine()
        >>> products = [IndirectProductInput(
        ...     product_id="APP-001",
        ...     product_type="refrigerator",
        ...     category="APPLIANCES",
        ...     units_sold=10000,
        ...     lifetime_years=15,
        ...     annual_energy_kwh=Decimal("400"),
        ...     use_region="US",
        ... )]
        >>> result = engine.calculate_electricity(products, "ORG-001", 2024)
        >>> assert result["total_co2e"] > Decimal("0")
    """

    _instance: Optional["IndirectEmissionsCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "IndirectEmissionsCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the indirect emissions calculator engine (once)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._calculation_count: int = 0
        self._count_lock: threading.Lock = threading.Lock()

        logger.info(
            "%s initialized: agent=%s, version=%s, "
            "grid_regions=%d, heating_fuels=%d, steam_sources=%d",
            ENGINE_NAME,
            AGENT_ID,
            VERSION,
            len(GRID_EMISSION_FACTORS),
            len(HEATING_FUEL_EMISSION_FACTORS),
            len(STEAM_COOLING_EMISSION_FACTORS),
        )

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _increment_count(self) -> int:
        """
        Increment and return the calculation counter thread-safely.

        Returns:
            Updated calculation count.
        """
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _get_degradation_rate(
        self,
        category: str,
        override: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Get the annual energy degradation rate for a product category.

        Args:
            category: Product category code.
            override: Optional explicit degradation rate override.

        Returns:
            Degradation rate as Decimal (0.0-1.0 fraction, e.g. 0.015 = 1.5%).
        """
        if override is not None:
            return override
        return DEGRADATION_RATES.get(category, Decimal("0.005"))

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def validate_inputs(
        self,
        products: List[IndirectProductInput],
    ) -> List[str]:
        """
        Validate a list of product inputs for indirect emissions calculation.

        Checks:
        - Product list is non-empty
        - Each product has a valid product_id and product_type
        - units_sold is positive
        - lifetime_years is positive and <= 50
        - At least one energy consumption field is provided
        - Category is a recognized product category

        Args:
            products: List of IndirectProductInput objects to validate.

        Returns:
            List of validation error messages. Empty list means all valid.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> errors = engine.validate_inputs([])
            >>> assert "Product list must not be empty" in errors
        """
        errors: List[str] = []

        if not products:
            errors.append("Product list must not be empty")
            return errors

        valid_categories = {c.value for c in ProductCategory}

        for i, product in enumerate(products):
            prefix = f"Product[{i}] (id={product.product_id})"

            if not product.product_id or not product.product_id.strip():
                errors.append(f"{prefix}: product_id must be non-empty")

            if not product.product_type or not product.product_type.strip():
                errors.append(f"{prefix}: product_type must be non-empty")

            if product.units_sold <= 0:
                errors.append(
                    f"{prefix}: units_sold must be positive, got {product.units_sold}"
                )

            if product.lifetime_years <= 0:
                errors.append(
                    f"{prefix}: lifetime_years must be positive, got {product.lifetime_years}"
                )
            elif product.lifetime_years > 50:
                errors.append(
                    f"{prefix}: lifetime_years exceeds maximum (50), got {product.lifetime_years}"
                )

            if product.category not in valid_categories:
                errors.append(
                    f"{prefix}: unrecognized category '{product.category}'. "
                    f"Valid categories: {sorted(valid_categories)}"
                )

            has_energy = (
                product.annual_energy_kwh is not None
                or product.annual_fuel_consumption is not None
                or product.annual_steam_mj is not None
            )
            if not has_energy:
                errors.append(
                    f"{prefix}: at least one energy consumption field must be "
                    "provided (annual_energy_kwh, annual_fuel_consumption, "
                    "or annual_steam_mj)"
                )

            if product.annual_energy_kwh is not None and product.annual_energy_kwh < _ZERO:
                errors.append(
                    f"{prefix}: annual_energy_kwh must be non-negative, "
                    f"got {product.annual_energy_kwh}"
                )

            if product.annual_fuel_consumption is not None and product.annual_fuel_consumption < _ZERO:
                errors.append(
                    f"{prefix}: annual_fuel_consumption must be non-negative, "
                    f"got {product.annual_fuel_consumption}"
                )

            if product.annual_steam_mj is not None and product.annual_steam_mj < _ZERO:
                errors.append(
                    f"{prefix}: annual_steam_mj must be non-negative, "
                    f"got {product.annual_steam_mj}"
                )

            if product.annual_fuel_consumption is not None and product.fuel_type is None:
                errors.append(
                    f"{prefix}: fuel_type is required when annual_fuel_consumption is provided"
                )

            if product.annual_steam_mj is not None and product.steam_source is None:
                errors.append(
                    f"{prefix}: steam_source is required when annual_steam_mj is provided"
                )

            if product.degradation_rate is not None:
                if product.degradation_rate < _ZERO or product.degradation_rate > _ONE:
                    errors.append(
                        f"{prefix}: degradation_rate must be between 0 and 1, "
                        f"got {product.degradation_rate}"
                    )

        if errors:
            logger.warning(
                "Input validation found %d error(s) across %d product(s)",
                len(errors),
                len(products),
            )

        return errors

    # ==========================================================================
    # EMISSION FACTOR RESOLUTION
    # ==========================================================================

    def _resolve_grid_ef(self, region: str) -> Decimal:
        """
        Resolve grid emission factor for a region.

        Falls back to GLOBAL if region is not found.

        Args:
            region: Grid region code (e.g., "US", "GB", "GLOBAL").

        Returns:
            Grid emission factor in kgCO2e/kWh.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> engine._resolve_grid_ef("US")
            Decimal('0.417')
        """
        region_upper = region.upper()
        ef = GRID_EMISSION_FACTORS.get(region_upper)
        if ef is None:
            logger.warning(
                "Grid region '%s' not found, falling back to GLOBAL (%s kgCO2e/kWh)",
                region_upper,
                GRID_EMISSION_FACTORS["GLOBAL"],
            )
            ef = GRID_EMISSION_FACTORS["GLOBAL"]
        return ef

    def _resolve_fuel_ef(self, fuel_type: str) -> Decimal:
        """
        Resolve heating fuel emission factor by fuel type code.

        Args:
            fuel_type: Fuel type code (e.g., "NATURAL_GAS", "DIESEL").

        Returns:
            Combustion emission factor in kgCO2e per unit.

        Raises:
            ValueError: If fuel type is not recognized.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> engine._resolve_fuel_ef("NATURAL_GAS")
            Decimal('2.024')
        """
        fuel_upper = fuel_type.upper()
        fuel_data = HEATING_FUEL_EMISSION_FACTORS.get(fuel_upper)
        if fuel_data is None:
            available = sorted(HEATING_FUEL_EMISSION_FACTORS.keys())
            raise ValueError(
                f"Fuel type '{fuel_upper}' not found. "
                f"Available types: {available}"
            )
        return fuel_data["ef"]

    def _resolve_steam_ef(self, source: str) -> Decimal:
        """
        Resolve steam/cooling emission factor by source type.

        Args:
            source: Steam/cooling source type.

        Returns:
            Emission factor in kgCO2e/MJ.

        Raises:
            ValueError: If source type is not recognized.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> engine._resolve_steam_ef("district_heating")
            Decimal('0.0650')
        """
        source_lower = source.lower()
        source_data = STEAM_COOLING_EMISSION_FACTORS.get(source_lower)
        if source_data is None:
            available = sorted(STEAM_COOLING_EMISSION_FACTORS.keys())
            raise ValueError(
                f"Steam/cooling source '{source_lower}' not found. "
                f"Available sources: {available}"
            )
        return source_data["ef"]

    # ==========================================================================
    # DEGRADATION MODELING
    # ==========================================================================

    def apply_degradation(
        self,
        base_annual_consumption: Decimal,
        lifetime_years: int,
        degradation_rate: Decimal,
    ) -> Decimal:
        """
        Compute total lifetime consumption with year-by-year degradation.

        Applies an annual efficiency degradation model where each year's
        consumption is reduced by the degradation rate:
            consumption_year_t = base x (1 - rate) ^ t

        The total is the sum over all years from t=0 to t=(lifetime-1).

        Args:
            base_annual_consumption: Base annual energy consumption (year 0).
            lifetime_years: Product lifetime in years.
            degradation_rate: Annual degradation rate as a fraction (e.g., 0.015).

        Returns:
            Total lifetime consumption, quantized to 8 decimal places.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> total = engine.apply_degradation(
            ...     Decimal("400"), 15, Decimal("0.005")
            ... )
            >>> total > Decimal("0")
            True
        """
        if degradation_rate == _ZERO:
            # No degradation: simple multiplication
            return _q(base_annual_consumption * Decimal(str(lifetime_years)))

        total = _ZERO
        decay_factor = _ONE - degradation_rate

        for year in range(lifetime_years):
            year_consumption = _q(
                base_annual_consumption * _q(decay_factor ** year)
            )
            total = total + year_consumption

        return _q(total)

    def compute_degradation_curve(
        self,
        base_annual_consumption: Decimal,
        lifetime_years: int,
        degradation_rate: Decimal,
    ) -> List[Decimal]:
        """
        Compute the year-by-year energy consumption with degradation.

        Returns a list of annual consumption values, one per year of the
        product lifetime, each reduced by the cumulative degradation.

        Args:
            base_annual_consumption: Base annual energy consumption (year 0).
            lifetime_years: Product lifetime in years.
            degradation_rate: Annual degradation rate.

        Returns:
            List of Decimal values, one per year, quantized to 8 dp.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> curve = engine.compute_degradation_curve(
            ...     Decimal("1000"), 5, Decimal("0.01")
            ... )
            >>> len(curve) == 5
            True
            >>> curve[0] > curve[4]
            True
        """
        curve: List[Decimal] = []
        decay_factor = _ONE - degradation_rate

        for year in range(lifetime_years):
            if degradation_rate == _ZERO:
                year_value = _q(base_annual_consumption)
            else:
                year_value = _q(
                    base_annual_consumption * _q(decay_factor ** year)
                )
            curve.append(year_value)

        return curve

    # ==========================================================================
    # SINGLE-PRODUCT CALCULATIONS
    # ==========================================================================

    def calculate_product_electricity(
        self,
        product: IndirectProductInput,
        grid_ef: Decimal,
    ) -> IndirectEmissionResult:
        """
        Calculate indirect electricity emissions for a single product.

        Formula: E = units_sold x total_lifetime_consumption x grid_EF
        where total_lifetime_consumption accounts for degradation.

        Args:
            product: Product input data.
            grid_ef: Grid emission factor (kgCO2e/kWh).

        Returns:
            IndirectEmissionResult with emissions breakdown.

        Raises:
            ValueError: If annual_energy_kwh is None.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> product = IndirectProductInput(
            ...     product_id="REF-001",
            ...     product_type="refrigerator",
            ...     category="APPLIANCES",
            ...     units_sold=5000,
            ...     lifetime_years=15,
            ...     annual_energy_kwh=Decimal("400"),
            ...     use_region="US",
            ... )
            >>> result = engine.calculate_product_electricity(
            ...     product, Decimal("0.417")
            ... )
            >>> result.total_co2e > Decimal("0")
            True
        """
        if product.annual_energy_kwh is None:
            raise ValueError(
                f"Product '{product.product_id}': annual_energy_kwh is required "
                "for electricity calculation"
            )

        degradation_rate = self._get_degradation_rate(
            product.category, product.degradation_rate
        )

        # Total lifetime kWh with degradation
        total_lifetime_kwh = self.apply_degradation(
            product.annual_energy_kwh,
            product.lifetime_years,
            degradation_rate,
        )

        # Emissions per single unit over its lifetime
        co2e_per_unit = _q(total_lifetime_kwh * grid_ef)

        # Total emissions for all units sold
        units_dec = Decimal(str(product.units_sold))
        total_co2e = _q(co2e_per_unit * units_dec)

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(product, "electricity")
        uncertainty = self.compute_uncertainty(total_co2e, "electricity")

        # Provenance
        provenance_hash = self._build_provenance(
            "electricity",
            product.to_dict(),
            {
                "grid_ef": str(grid_ef),
                "degradation_rate": str(degradation_rate),
                "total_lifetime_kwh": str(total_lifetime_kwh),
                "co2e_per_unit": str(co2e_per_unit),
                "total_co2e": str(total_co2e),
            },
        )

        return IndirectEmissionResult(
            product_id=product.product_id,
            product_type=product.product_type,
            category=product.category,
            method="electricity",
            units_sold=product.units_sold,
            lifetime_years=product.lifetime_years,
            annual_consumption=product.annual_energy_kwh,
            total_lifetime_consumption=total_lifetime_kwh,
            emission_factor=grid_ef,
            ef_unit="kgCO2e/kWh",
            co2e_per_unit=co2e_per_unit,
            total_co2e=total_co2e,
            degradation_rate=degradation_rate,
            dqi_score=dqi_score,
            uncertainty_lower=uncertainty["lower"],
            uncertainty_upper=uncertainty["upper"],
            provenance_hash=provenance_hash,
        )

    def calculate_product_heating(
        self,
        product: IndirectProductInput,
        fuel_ef: Decimal,
    ) -> IndirectEmissionResult:
        """
        Calculate indirect heating fuel emissions for a single product.

        Formula: E = units_sold x total_lifetime_fuel x fuel_EF
        where total_lifetime_fuel accounts for degradation.

        Args:
            product: Product input data.
            fuel_ef: Fuel combustion emission factor (kgCO2e/unit).

        Returns:
            IndirectEmissionResult with emissions breakdown.

        Raises:
            ValueError: If annual_fuel_consumption is None.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> product = IndirectProductInput(
            ...     product_id="HVAC-001",
            ...     product_type="gas_furnace",
            ...     category="HVAC",
            ...     units_sold=2000,
            ...     lifetime_years=20,
            ...     annual_fuel_consumption=Decimal("1500"),
            ...     fuel_type="NATURAL_GAS",
            ... )
            >>> result = engine.calculate_product_heating(
            ...     product, Decimal("2.024")
            ... )
            >>> result.total_co2e > Decimal("0")
            True
        """
        if product.annual_fuel_consumption is None:
            raise ValueError(
                f"Product '{product.product_id}': annual_fuel_consumption is required "
                "for heating calculation"
            )

        degradation_rate = self._get_degradation_rate(
            product.category, product.degradation_rate
        )

        # Total lifetime fuel consumption with degradation
        total_lifetime_fuel = self.apply_degradation(
            product.annual_fuel_consumption,
            product.lifetime_years,
            degradation_rate,
        )

        # Emissions per unit
        co2e_per_unit = _q(total_lifetime_fuel * fuel_ef)

        # Total emissions for all units
        units_dec = Decimal(str(product.units_sold))
        total_co2e = _q(co2e_per_unit * units_dec)

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(product, "heating")
        uncertainty = self.compute_uncertainty(total_co2e, "heating")

        # Provenance
        provenance_hash = self._build_provenance(
            "heating",
            product.to_dict(),
            {
                "fuel_type": product.fuel_type,
                "fuel_ef": str(fuel_ef),
                "degradation_rate": str(degradation_rate),
                "total_lifetime_fuel": str(total_lifetime_fuel),
                "co2e_per_unit": str(co2e_per_unit),
                "total_co2e": str(total_co2e),
            },
        )

        fuel_unit = "kgCO2e/L"
        if product.fuel_type and product.fuel_type.upper() in ("NATURAL_GAS", "CNG"):
            fuel_unit = "kgCO2e/m3"
        elif product.fuel_type and product.fuel_type.upper() in (
            "COAL", "WOOD_PELLETS", "LNG", "HYDROGEN",
        ):
            fuel_unit = "kgCO2e/kg"

        return IndirectEmissionResult(
            product_id=product.product_id,
            product_type=product.product_type,
            category=product.category,
            method="heating",
            units_sold=product.units_sold,
            lifetime_years=product.lifetime_years,
            annual_consumption=product.annual_fuel_consumption,
            total_lifetime_consumption=total_lifetime_fuel,
            emission_factor=fuel_ef,
            ef_unit=fuel_unit,
            co2e_per_unit=co2e_per_unit,
            total_co2e=total_co2e,
            degradation_rate=degradation_rate,
            dqi_score=dqi_score,
            uncertainty_lower=uncertainty["lower"],
            uncertainty_upper=uncertainty["upper"],
            provenance_hash=provenance_hash,
        )

    def calculate_product_steam(
        self,
        product: IndirectProductInput,
        steam_ef: Decimal,
    ) -> IndirectEmissionResult:
        """
        Calculate indirect steam/cooling emissions for a single product.

        Formula: E = units_sold x total_lifetime_steam x steam_EF
        where total_lifetime_steam accounts for degradation.

        Args:
            product: Product input data.
            steam_ef: Steam/cooling emission factor (kgCO2e/MJ).

        Returns:
            IndirectEmissionResult with emissions breakdown.

        Raises:
            ValueError: If annual_steam_mj is None.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> product = IndirectProductInput(
            ...     product_id="BLDG-001",
            ...     product_type="industrial_heater",
            ...     category="BUILDING_PRODUCTS",
            ...     units_sold=500,
            ...     lifetime_years=20,
            ...     annual_steam_mj=Decimal("5000"),
            ...     steam_source="district_heating",
            ... )
            >>> result = engine.calculate_product_steam(
            ...     product, Decimal("0.065")
            ... )
            >>> result.total_co2e > Decimal("0")
            True
        """
        if product.annual_steam_mj is None:
            raise ValueError(
                f"Product '{product.product_id}': annual_steam_mj is required "
                "for steam/cooling calculation"
            )

        degradation_rate = self._get_degradation_rate(
            product.category, product.degradation_rate
        )

        # Total lifetime MJ with degradation
        total_lifetime_mj = self.apply_degradation(
            product.annual_steam_mj,
            product.lifetime_years,
            degradation_rate,
        )

        # Emissions per unit
        co2e_per_unit = _q(total_lifetime_mj * steam_ef)

        # Total emissions for all units
        units_dec = Decimal(str(product.units_sold))
        total_co2e = _q(co2e_per_unit * units_dec)

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(product, "steam_cooling")
        uncertainty = self.compute_uncertainty(total_co2e, "steam_cooling")

        # Provenance
        provenance_hash = self._build_provenance(
            "steam_cooling",
            product.to_dict(),
            {
                "steam_source": product.steam_source,
                "steam_ef": str(steam_ef),
                "degradation_rate": str(degradation_rate),
                "total_lifetime_mj": str(total_lifetime_mj),
                "co2e_per_unit": str(co2e_per_unit),
                "total_co2e": str(total_co2e),
            },
        )

        return IndirectEmissionResult(
            product_id=product.product_id,
            product_type=product.product_type,
            category=product.category,
            method="steam_cooling",
            units_sold=product.units_sold,
            lifetime_years=product.lifetime_years,
            annual_consumption=product.annual_steam_mj,
            total_lifetime_consumption=total_lifetime_mj,
            emission_factor=steam_ef,
            ef_unit="kgCO2e/MJ",
            co2e_per_unit=co2e_per_unit,
            total_co2e=total_co2e,
            degradation_rate=degradation_rate,
            dqi_score=dqi_score,
            uncertainty_lower=uncertainty["lower"],
            uncertainty_upper=uncertainty["upper"],
            provenance_hash=provenance_hash,
        )

    # ==========================================================================
    # BATCH CALCULATION METHODS
    # ==========================================================================

    def calculate_electricity(
        self,
        products: List[IndirectProductInput],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate indirect electricity emissions for multiple products.

        Formula per product:
            E = units_sold x lifetime x annual_kWh x grid_EF
            (with year-by-year degradation applied to annual_kWh)

        Args:
            products: List of product inputs with annual_energy_kwh.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with product results, total_co2e, and provenance.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> products = [IndirectProductInput(
            ...     product_id="REF-001", product_type="refrigerator",
            ...     category="APPLIANCES", units_sold=5000,
            ...     lifetime_years=15, annual_energy_kwh=Decimal("400"),
            ...     use_region="US",
            ... )]
            >>> result = engine.calculate_electricity(products, "ORG-001", 2024)
            >>> result["total_co2e"] > Decimal("0")
            True
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "%s electricity calculation #%d: org=%s, year=%d, products=%d",
            ENGINE_NAME, calc_number, org_id, year, len(products),
        )

        # Validate
        errors = self.validate_inputs(products)
        if errors:
            raise ValueError(
                f"Input validation failed with {len(errors)} error(s): "
                + "; ".join(errors[:5])
            )

        # Calculate per product
        results: List[IndirectEmissionResult] = []
        total_co2e = _ZERO
        error_list: List[Dict[str, str]] = []

        for product in products:
            try:
                if product.annual_energy_kwh is None:
                    logger.debug(
                        "Skipping product '%s': no annual_energy_kwh",
                        product.product_id,
                    )
                    continue

                grid_ef = self._resolve_grid_ef(product.use_region)
                result = self.calculate_product_electricity(product, grid_ef)
                results.append(result)
                total_co2e = _q(total_co2e + result.total_co2e)

            except Exception as exc:
                logger.error(
                    "Electricity calculation failed for product '%s': %s",
                    product.product_id, exc, exc_info=True,
                )
                error_list.append({
                    "product_id": product.product_id,
                    "error": str(exc),
                })

        # Aggregate DQI
        avg_dqi = _ZERO
        if results:
            dqi_sum = sum(r.dqi_score for r in results)
            avg_dqi = _q(dqi_sum / Decimal(str(len(results))))

        # Aggregate uncertainty
        agg_uncertainty = self.compute_uncertainty(total_co2e, "electricity")

        # Provenance for the batch
        batch_provenance = self._build_provenance(
            "electricity_batch",
            {
                "org_id": org_id,
                "year": year,
                "product_count": len(products),
                "method": "electricity",
            },
            {
                "total_co2e": str(total_co2e),
                "result_count": len(results),
                "avg_dqi": str(avg_dqi),
            },
        )

        duration = time.monotonic() - start_time

        logger.info(
            "%s electricity calculation #%d complete: "
            "total_co2e=%s kgCO2e, products=%d, duration=%.3fs",
            ENGINE_NAME, calc_number, total_co2e, len(results), duration,
        )

        return {
            "method": "electricity",
            "org_id": org_id,
            "reporting_year": year,
            "product_results": [r.to_dict() for r in results],
            "total_co2e": total_co2e,
            "total_tco2e": _q(total_co2e / Decimal("1000")),
            "product_count": len(results),
            "avg_dqi_score": avg_dqi,
            "uncertainty_lower": agg_uncertainty["lower"],
            "uncertainty_upper": agg_uncertainty["upper"],
            "errors": error_list,
            "provenance_hash": batch_provenance,
            "processing_time_ms": _q(Decimal(str(duration * 1000))),
            "engine": ENGINE_NAME,
            "agent_id": AGENT_ID,
            "version": VERSION,
        }

    def calculate_heating(
        self,
        products: List[IndirectProductInput],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate indirect heating fuel emissions for multiple products.

        Formula per product:
            E = units_sold x lifetime x annual_fuel x fuel_EF
            (with year-by-year degradation applied to annual_fuel)

        Args:
            products: List of product inputs with annual_fuel_consumption.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with product results, total_co2e, and provenance.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> products = [IndirectProductInput(
            ...     product_id="HVAC-001", product_type="gas_furnace",
            ...     category="HVAC", units_sold=2000,
            ...     lifetime_years=20, annual_fuel_consumption=Decimal("1500"),
            ...     fuel_type="NATURAL_GAS",
            ... )]
            >>> result = engine.calculate_heating(products, "ORG-001", 2024)
            >>> result["total_co2e"] > Decimal("0")
            True
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "%s heating calculation #%d: org=%s, year=%d, products=%d",
            ENGINE_NAME, calc_number, org_id, year, len(products),
        )

        # Validate
        errors = self.validate_inputs(products)
        if errors:
            raise ValueError(
                f"Input validation failed with {len(errors)} error(s): "
                + "; ".join(errors[:5])
            )

        # Calculate per product
        results: List[IndirectEmissionResult] = []
        total_co2e = _ZERO
        error_list: List[Dict[str, str]] = []

        for product in products:
            try:
                if product.annual_fuel_consumption is None:
                    logger.debug(
                        "Skipping product '%s': no annual_fuel_consumption",
                        product.product_id,
                    )
                    continue

                fuel_ef = self._resolve_fuel_ef(product.fuel_type or "NATURAL_GAS")
                result = self.calculate_product_heating(product, fuel_ef)
                results.append(result)
                total_co2e = _q(total_co2e + result.total_co2e)

            except Exception as exc:
                logger.error(
                    "Heating calculation failed for product '%s': %s",
                    product.product_id, exc, exc_info=True,
                )
                error_list.append({
                    "product_id": product.product_id,
                    "error": str(exc),
                })

        # Aggregate DQI
        avg_dqi = _ZERO
        if results:
            dqi_sum = sum(r.dqi_score for r in results)
            avg_dqi = _q(dqi_sum / Decimal(str(len(results))))

        # Aggregate uncertainty
        agg_uncertainty = self.compute_uncertainty(total_co2e, "heating")

        # Provenance for the batch
        batch_provenance = self._build_provenance(
            "heating_batch",
            {
                "org_id": org_id,
                "year": year,
                "product_count": len(products),
                "method": "heating",
            },
            {
                "total_co2e": str(total_co2e),
                "result_count": len(results),
                "avg_dqi": str(avg_dqi),
            },
        )

        duration = time.monotonic() - start_time

        logger.info(
            "%s heating calculation #%d complete: "
            "total_co2e=%s kgCO2e, products=%d, duration=%.3fs",
            ENGINE_NAME, calc_number, total_co2e, len(results), duration,
        )

        return {
            "method": "heating",
            "org_id": org_id,
            "reporting_year": year,
            "product_results": [r.to_dict() for r in results],
            "total_co2e": total_co2e,
            "total_tco2e": _q(total_co2e / Decimal("1000")),
            "product_count": len(results),
            "avg_dqi_score": avg_dqi,
            "uncertainty_lower": agg_uncertainty["lower"],
            "uncertainty_upper": agg_uncertainty["upper"],
            "errors": error_list,
            "provenance_hash": batch_provenance,
            "processing_time_ms": _q(Decimal(str(duration * 1000))),
            "engine": ENGINE_NAME,
            "agent_id": AGENT_ID,
            "version": VERSION,
        }

    def calculate_steam_cooling(
        self,
        products: List[IndirectProductInput],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate indirect steam/cooling emissions for multiple products.

        Formula per product:
            E = units_sold x lifetime x annual_MJ x steam_EF
            (with year-by-year degradation applied to annual_MJ)

        Args:
            products: List of product inputs with annual_steam_mj.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with product results, total_co2e, and provenance.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> products = [IndirectProductInput(
            ...     product_id="BLDG-001", product_type="industrial_heater",
            ...     category="BUILDING_PRODUCTS", units_sold=500,
            ...     lifetime_years=20, annual_steam_mj=Decimal("5000"),
            ...     steam_source="district_heating",
            ... )]
            >>> result = engine.calculate_steam_cooling(products, "ORG-001", 2024)
            >>> result["total_co2e"] > Decimal("0")
            True
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "%s steam/cooling calculation #%d: org=%s, year=%d, products=%d",
            ENGINE_NAME, calc_number, org_id, year, len(products),
        )

        # Validate
        errors = self.validate_inputs(products)
        if errors:
            raise ValueError(
                f"Input validation failed with {len(errors)} error(s): "
                + "; ".join(errors[:5])
            )

        # Calculate per product
        results: List[IndirectEmissionResult] = []
        total_co2e = _ZERO
        error_list: List[Dict[str, str]] = []

        for product in products:
            try:
                if product.annual_steam_mj is None:
                    logger.debug(
                        "Skipping product '%s': no annual_steam_mj",
                        product.product_id,
                    )
                    continue

                steam_ef = self._resolve_steam_ef(
                    product.steam_source or "district_heating"
                )
                result = self.calculate_product_steam(product, steam_ef)
                results.append(result)
                total_co2e = _q(total_co2e + result.total_co2e)

            except Exception as exc:
                logger.error(
                    "Steam/cooling calculation failed for product '%s': %s",
                    product.product_id, exc, exc_info=True,
                )
                error_list.append({
                    "product_id": product.product_id,
                    "error": str(exc),
                })

        # Aggregate DQI
        avg_dqi = _ZERO
        if results:
            dqi_sum = sum(r.dqi_score for r in results)
            avg_dqi = _q(dqi_sum / Decimal(str(len(results))))

        # Aggregate uncertainty
        agg_uncertainty = self.compute_uncertainty(total_co2e, "steam_cooling")

        # Provenance for the batch
        batch_provenance = self._build_provenance(
            "steam_cooling_batch",
            {
                "org_id": org_id,
                "year": year,
                "product_count": len(products),
                "method": "steam_cooling",
            },
            {
                "total_co2e": str(total_co2e),
                "result_count": len(results),
                "avg_dqi": str(avg_dqi),
            },
        )

        duration = time.monotonic() - start_time

        logger.info(
            "%s steam/cooling calculation #%d complete: "
            "total_co2e=%s kgCO2e, products=%d, duration=%.3fs",
            ENGINE_NAME, calc_number, total_co2e, len(results), duration,
        )

        return {
            "method": "steam_cooling",
            "org_id": org_id,
            "reporting_year": year,
            "product_results": [r.to_dict() for r in results],
            "total_co2e": total_co2e,
            "total_tco2e": _q(total_co2e / Decimal("1000")),
            "product_count": len(results),
            "avg_dqi_score": avg_dqi,
            "uncertainty_lower": agg_uncertainty["lower"],
            "uncertainty_upper": agg_uncertainty["upper"],
            "errors": error_list,
            "provenance_hash": batch_provenance,
            "processing_time_ms": _q(Decimal(str(duration * 1000))),
            "engine": ENGINE_NAME,
            "agent_id": AGENT_ID,
            "version": VERSION,
        }

    # ==========================================================================
    # DISPATCHER
    # ==========================================================================

    def calculate(
        self,
        products: List[IndirectProductInput],
        method: str,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Dispatch to the appropriate calculation method.

        Routes to calculate_electricity, calculate_heating, or
        calculate_steam_cooling based on the method parameter.

        Args:
            products: List of product inputs.
            method: Calculation method ("electricity", "heating", "steam_cooling").
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with calculation results.

        Raises:
            ValueError: If method is not recognized or inputs are invalid.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> result = engine.calculate(products, "electricity", "ORG-001", 2024)
        """
        method_lower = method.lower().strip()

        dispatch_map = {
            "electricity": self.calculate_electricity,
            "heating": self.calculate_heating,
            "steam_cooling": self.calculate_steam_cooling,
        }

        calculator = dispatch_map.get(method_lower)
        if calculator is None:
            raise ValueError(
                f"Unknown indirect emission method '{method}'. "
                f"Valid methods: {sorted(dispatch_map.keys())}"
            )

        logger.info(
            "%s dispatching to %s method for org=%s, year=%d",
            ENGINE_NAME, method_lower, org_id, year,
        )

        return calculator(products, org_id, year)

    # ==========================================================================
    # DATA QUALITY & UNCERTAINTY
    # ==========================================================================

    def compute_dqi_score(
        self,
        product: IndirectProductInput,
        method: str,
    ) -> Decimal:
        """
        Compute the Data Quality Indicator score for a product calculation.

        Base scores by method:
        - electricity: 80
        - heating: 75
        - steam_cooling: 70

        Adjusted by data quality tier:
        - tier_1: x 1.00 (product-specific, primary data)
        - tier_2: x 0.80 (category average)
        - tier_3: x 0.60 (global average)

        Additional adjustments:
        - Product-specific degradation rate provided: +5
        - Recognized category: +3

        Args:
            product: Product input data.
            method: Calculation method string.

        Returns:
            DQI score as Decimal (0-100).

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> product = IndirectProductInput(
            ...     product_id="REF-001", product_type="refrigerator",
            ...     category="APPLIANCES", units_sold=5000,
            ...     lifetime_years=15, annual_energy_kwh=Decimal("400"),
            ...     data_quality_tier="tier_1",
            ... )
            >>> score = engine.compute_dqi_score(product, "electricity")
            >>> score >= Decimal("80")
            True
        """
        base = DQI_BASE_SCORES.get(method, Decimal("70"))
        tier_mult = DQI_TIER_MULTIPLIERS.get(
            product.data_quality_tier, Decimal("0.80")
        )

        score = _q(base * tier_mult)

        # Bonus for product-specific degradation rate
        if product.degradation_rate is not None:
            score = _q(score + Decimal("5"))

        # Bonus for recognized category
        valid_categories = {c.value for c in ProductCategory}
        if product.category in valid_categories:
            score = _q(score + Decimal("3"))

        # Cap at 100
        if score > Decimal("100"):
            score = Decimal("100")

        return _q(score)

    def compute_uncertainty(
        self,
        emissions: Decimal,
        method: str,
    ) -> Dict[str, Decimal]:
        """
        Compute uncertainty bounds for emissions.

        Uses method-specific half-widths for the 95% confidence interval:
        - electricity: +/- 20%
        - heating: +/- 25%
        - steam_cooling: +/- 30%

        Args:
            emissions: Total emissions value (kgCO2e).
            method: Calculation method string.

        Returns:
            Dictionary with "lower" and "upper" bounds as Decimal.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> bounds = engine.compute_uncertainty(Decimal("1000"), "electricity")
            >>> bounds["lower"]
            Decimal('800.00000000')
            >>> bounds["upper"]
            Decimal('1200.00000000')
        """
        halfwidth = UNCERTAINTY_HALFWIDTHS.get(method, Decimal("0.25"))

        lower = _q(emissions * (_ONE - halfwidth))
        upper = _q(emissions * (_ONE + halfwidth))

        # Ensure lower bound is non-negative
        if lower < _ZERO:
            lower = _ZERO

        return {
            "lower": lower,
            "upper": upper,
            "halfwidth_pct": _q(halfwidth * Decimal("100")),
            "method": method,
        }

    # ==========================================================================
    # PROVENANCE
    # ==========================================================================

    def _build_provenance(
        self,
        method: str,
        inputs: Any,
        result: Any,
    ) -> str:
        """
        Build SHA-256 provenance hash for a calculation.

        Combines the method identifier, serialized inputs, and serialized
        results into a deterministic hash.

        Args:
            method: Calculation method string.
            inputs: Input data (dict or object with to_dict).
            result: Result data (dict or object with to_dict).

        Returns:
            SHA-256 provenance hash string (64 hex characters).
        """
        return _compute_provenance_hash(
            ENGINE_NAME,
            AGENT_ID,
            VERSION,
            method,
            inputs,
            result,
        )

    # ==========================================================================
    # SUMMARY & STATE
    # ==========================================================================

    def get_calculation_count(self) -> int:
        """
        Get the total number of calculations performed.

        Returns:
            Integer count of calculations.
        """
        with self._count_lock:
            return self._calculation_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the engine state and configuration.

        Returns:
            Dict with engine metadata and statistics.

        Example:
            >>> engine = IndirectEmissionsCalculatorEngine()
            >>> summary = engine.get_engine_summary()
            >>> summary["engine"] == "IndirectEmissionsCalculatorEngine"
            True
        """
        return {
            "engine": ENGINE_NAME,
            "engine_number": ENGINE_NUMBER,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "calculation_count": self.get_calculation_count(),
            "grid_regions": len(GRID_EMISSION_FACTORS),
            "heating_fuels": len(HEATING_FUEL_EMISSION_FACTORS),
            "steam_sources": len(STEAM_COOLING_EMISSION_FACTORS),
            "product_categories": len(DEGRADATION_RATES),
            "methods": ["electricity", "heating", "steam_cooling"],
        }

    def get_supported_grid_regions(self) -> List[str]:
        """
        Get list of supported grid regions.

        Returns:
            Sorted list of region codes.
        """
        return sorted(GRID_EMISSION_FACTORS.keys())

    def get_supported_fuel_types(self) -> List[str]:
        """
        Get list of supported heating fuel types.

        Returns:
            Sorted list of fuel type codes.
        """
        return sorted(HEATING_FUEL_EMISSION_FACTORS.keys())

    def get_supported_steam_sources(self) -> List[str]:
        """
        Get list of supported steam/cooling source types.

        Returns:
            Sorted list of source type codes.
        """
        return sorted(STEAM_COOLING_EMISSION_FACTORS.keys())

    def get_grid_ef(self, region: str) -> Optional[Decimal]:
        """
        Get grid emission factor for a specific region.

        Args:
            region: Grid region code.

        Returns:
            Emission factor in kgCO2e/kWh, or None if not found.
        """
        return GRID_EMISSION_FACTORS.get(region.upper())

    def get_fuel_ef(self, fuel_type: str) -> Optional[Decimal]:
        """
        Get heating fuel emission factor.

        Args:
            fuel_type: Fuel type code.

        Returns:
            Emission factor in kgCO2e/unit, or None if not found.
        """
        data = HEATING_FUEL_EMISSION_FACTORS.get(fuel_type.upper())
        if data is not None:
            return data["ef"]
        return None

    def get_steam_ef(self, source: str) -> Optional[Decimal]:
        """
        Get steam/cooling emission factor.

        Args:
            source: Source type code.

        Returns:
            Emission factor in kgCO2e/MJ, or None if not found.
        """
        data = STEAM_COOLING_EMISSION_FACTORS.get(source.lower())
        if data is not None:
            return data["ef"]
        return None

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: This method is intended for use in test fixtures only.
        Do not call in production code.
        """
        with cls._lock:
            cls._instance = None


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================

_engine_instance: Optional[IndirectEmissionsCalculatorEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_indirect_emissions_calculator() -> IndirectEmissionsCalculatorEngine:
    """
    Get the singleton IndirectEmissionsCalculatorEngine instance.

    Thread-safe accessor for the global engine instance.

    Returns:
        IndirectEmissionsCalculatorEngine singleton instance.

    Example:
        >>> calculator = get_indirect_emissions_calculator()
        >>> result = calculator.calculate_electricity(products, "ORG-001", 2024)
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = IndirectEmissionsCalculatorEngine()
        return _engine_instance


def reset_indirect_emissions_calculator() -> None:
    """
    Reset the module-level calculator instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    IndirectEmissionsCalculatorEngine.reset()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "ENGINE_NAME",
    "ENGINE_NUMBER",
    # Enums
    "IndirectEmissionMethod",
    "GridRegion",
    "HeatingFuelType",
    "SteamCoolingSource",
    "ProductCategory",
    "DataQualityTier",
    "UncertaintyMethod",
    # EF Tables
    "GRID_EMISSION_FACTORS",
    "HEATING_FUEL_EMISSION_FACTORS",
    "STEAM_COOLING_EMISSION_FACTORS",
    "DEGRADATION_RATES",
    "DQI_BASE_SCORES",
    "UNCERTAINTY_HALFWIDTHS",
    "DQI_DIMENSION_WEIGHTS",
    "DQI_TIER_MULTIPLIERS",
    # Data Models
    "IndirectProductInput",
    "IndirectEmissionResult",
    # Engine
    "IndirectEmissionsCalculatorEngine",
    "get_indirect_emissions_calculator",
    "reset_indirect_emissions_calculator",
    # Utility
    "_compute_provenance_hash",
]
