# -*- coding: utf-8 -*-
"""
FuelsAndFeedstocksCalculatorEngine - Engine 4: Use of Sold Products (AGENT-MRV-024)

This module implements the FuelsAndFeedstocksCalculatorEngine for AGENT-MRV-024
(Use of Sold Products, GHG Protocol Scope 3 Category 11). It provides thread-safe
singleton calculations for emissions from fuels and feedstocks sold by the reporting
company that are combusted or oxidized by downstream end users.

Calculation Formulae (GHG Protocol Scope 3, Chapter 6):

    Formula G -- Fuel Combustion by End Users:
        E_fuels = SUM_j( V_sold_j x EF_combustion_j )
        where:
            V_sold_j         = volume of fuel j sold (liters, m3, or tonnes)
            EF_combustion_j  = combustion emission factor for fuel j (kgCO2e/unit)

    Formula H -- Feedstock Oxidation:
        E_feedstock = SUM_j( M_sold_j x C_content_j x OF_j x 44/12 )
        where:
            M_sold_j     = mass of feedstock j sold (tonnes)
            C_content_j  = carbon content fraction (0-1)
            OF_j         = oxidation factor (fraction combusted, typically 0.95-1.0)
            44/12        = molecular weight ratio of CO2 to C (3.6667)

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal with ROUND_HALF_UP to 8 decimal places
    - No LLM calls anywhere in the numeric calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from IPCC 2006 GL, DEFRA 2024, EPA

Supports:
    - 15 fuel types with combustion emission factors and net calorific values
    - 5 feedstock types with carbon content and oxidation factors
    - Volume unit conversions (liters, m3, kg, tonnes, gallons)
    - Molecular weight ratio CO2/C = 44/12 for feedstock carbon oxidation
    - 5-dimension Data Quality Indicators (DQI)
    - Uncertainty quantification (+/- 10%)
    - Full SHA-256 provenance tracking

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation. All mutable state is protected by dedicated locks.

Example:
    >>> from greenlang.agents.mrv.use_of_sold_products.fuels_feedstocks_calculator import (
    ...     FuelsAndFeedstocksCalculatorEngine,
    ...     FuelSaleInput,
    ... )
    >>> engine = FuelsAndFeedstocksCalculatorEngine()
    >>> fuels = [FuelSaleInput(
    ...     fuel_id="SALE-001",
    ...     fuel_type="GASOLINE",
    ...     volume_sold=Decimal("1000000"),
    ...     unit="liters",
    ... )]
    >>> result = engine.calculate_fuel_sales(fuels, "ORG-001", 2024)
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
ENGINE_NAME: str = "FuelsAndFeedstocksCalculatorEngine"
ENGINE_NUMBER: int = 4

# ==============================================================================
# DECIMAL PRECISION CONSTANTS
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_QUANT_8DP = _PRECISION

# Molecular weight ratio CO2/C = 44/12 = 3.666666...
_CO2_C_RATIO = Decimal("44") / Decimal("12")


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


class FuelType(str, Enum):
    """Fuel types sold by the reporting company."""

    GASOLINE = "GASOLINE"
    DIESEL = "DIESEL"
    NATURAL_GAS = "NATURAL_GAS"
    LPG = "LPG"
    KEROSENE = "KEROSENE"
    HFO = "HFO"
    JET_FUEL = "JET_FUEL"
    ETHANOL = "ETHANOL"
    BIODIESEL = "BIODIESEL"
    COAL = "COAL"
    WOOD_PELLETS = "WOOD_PELLETS"
    PROPANE = "PROPANE"
    HYDROGEN = "HYDROGEN"
    CNG = "CNG"
    LNG = "LNG"


class FeedstockType(str, Enum):
    """Feedstock types sold that may be oxidized downstream."""

    PETROLEUM_COKE = "PETROLEUM_COKE"
    NAPHTHA = "NAPHTHA"
    NATURAL_GAS_LIQUIDS = "NATURAL_GAS_LIQUIDS"
    COAL_TAR = "COAL_TAR"
    CARBON_BLACK = "CARBON_BLACK"


class VolumeUnit(str, Enum):
    """Volume/mass units for fuel and feedstock quantities."""

    LITERS = "liters"
    CUBIC_METERS = "m3"
    KILOGRAMS = "kg"
    TONNES = "tonnes"
    GALLONS = "gallons"


class FuelFeedstockMethod(str, Enum):
    """Calculation methods for fuels and feedstocks."""

    FUEL_SALES = "fuel_sales"
    FEEDSTOCK = "feedstock"


class DataQualityTier(str, Enum):
    """Data quality tiers affecting DQI and uncertainty."""

    TIER_1 = "tier_1"  # Primary data, product-specific
    TIER_2 = "tier_2"  # Industry average, secondary data
    TIER_3 = "tier_3"  # Global average, estimated data


# ==============================================================================
# EMISSION FACTOR TABLES
# ==============================================================================

# Fuel combustion emission factors -- IPCC 2006 GL / DEFRA 2024 / EPA
# ef: kgCO2e per native unit (liters for liquids, m3 for gases, kg for solids)
# ncv: net calorific value in MJ per native unit
# native_unit: the unit that ef and ncv refer to
# density: kg per liter (for liquid fuels) or kg per m3 (for gases)
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "GASOLINE": {
        "ef": Decimal("2.315"),
        "ncv_mj": Decimal("34.2"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.745"),
        "description": "Motor gasoline / petrol",
    },
    "DIESEL": {
        "ef": Decimal("2.706"),
        "ncv_mj": Decimal("38.6"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.832"),
        "description": "Automotive diesel / gas oil",
    },
    "NATURAL_GAS": {
        "ef": Decimal("2.024"),
        "ncv_mj": Decimal("38.3"),
        "native_unit": "m3",
        "density_kg_per_m3": Decimal("0.717"),
        "description": "Pipeline natural gas",
    },
    "LPG": {
        "ef": Decimal("1.557"),
        "ncv_mj": Decimal("26.1"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.510"),
        "description": "Liquefied petroleum gas",
    },
    "KEROSENE": {
        "ef": Decimal("2.541"),
        "ncv_mj": Decimal("37.0"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.800"),
        "description": "Kerosene / paraffin",
    },
    "HFO": {
        "ef": Decimal("3.114"),
        "ncv_mj": Decimal("40.4"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.960"),
        "description": "Heavy fuel oil / residual fuel oil",
    },
    "JET_FUEL": {
        "ef": Decimal("2.548"),
        "ncv_mj": Decimal("37.4"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.804"),
        "description": "Aviation turbine fuel / Jet A-1",
    },
    "ETHANOL": {
        "ef": Decimal("0.020"),
        "ncv_mj": Decimal("26.7"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.789"),
        "description": "Fuel ethanol (biogenic, near-zero fossil)",
    },
    "BIODIESEL": {
        "ef": Decimal("0.015"),
        "ncv_mj": Decimal("37.0"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.880"),
        "description": "Biodiesel / FAME (biogenic, near-zero fossil)",
    },
    "COAL": {
        "ef": Decimal("2.883"),
        "ncv_mj": Decimal("25.8"),
        "native_unit": "kg",
        "density_kg_per_l": None,
        "description": "Bituminous coal",
    },
    "WOOD_PELLETS": {
        "ef": Decimal("0.015"),
        "ncv_mj": Decimal("17.0"),
        "native_unit": "kg",
        "density_kg_per_l": None,
        "description": "Wood pellets (biogenic, near-zero fossil)",
    },
    "PROPANE": {
        "ef": Decimal("1.530"),
        "ncv_mj": Decimal("25.3"),
        "native_unit": "liters",
        "density_kg_per_l": Decimal("0.493"),
        "description": "Propane gas (liquefied)",
    },
    "HYDROGEN": {
        "ef": Decimal("0.000"),
        "ncv_mj": Decimal("120.0"),
        "native_unit": "kg",
        "density_kg_per_l": None,
        "description": "Hydrogen fuel (zero direct combustion CO2)",
    },
    "CNG": {
        "ef": Decimal("2.024"),
        "ncv_mj": Decimal("38.3"),
        "native_unit": "m3",
        "density_kg_per_m3": Decimal("0.717"),
        "description": "Compressed natural gas",
    },
    "LNG": {
        "ef": Decimal("2.750"),
        "ncv_mj": Decimal("49.5"),
        "native_unit": "kg",
        "density_kg_per_l": None,
        "description": "Liquefied natural gas",
    },
}

# Feedstock properties -- IPCC 2006 GL
# carbon_content: mass fraction of carbon in feedstock (0-1)
# oxidation_factor: fraction that is oxidized to CO2 downstream (0-1)
# native_unit: unit the feedstock is measured in
FEEDSTOCK_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "PETROLEUM_COKE": {
        "carbon_content": Decimal("0.870"),
        "oxidation_factor": Decimal("0.980"),
        "native_unit": "tonnes",
        "description": "Petroleum coke (calcined and raw)",
    },
    "NAPHTHA": {
        "carbon_content": Decimal("0.835"),
        "oxidation_factor": Decimal("0.950"),
        "native_unit": "tonnes",
        "description": "Naphtha (light distillate feedstock)",
    },
    "NATURAL_GAS_LIQUIDS": {
        "carbon_content": Decimal("0.830"),
        "oxidation_factor": Decimal("0.960"),
        "native_unit": "tonnes",
        "description": "Natural gas liquids (NGLs/condensate)",
    },
    "COAL_TAR": {
        "carbon_content": Decimal("0.900"),
        "oxidation_factor": Decimal("0.970"),
        "native_unit": "tonnes",
        "description": "Coal tar / coal tar pitch",
    },
    "CARBON_BLACK": {
        "carbon_content": Decimal("0.970"),
        "oxidation_factor": Decimal("0.990"),
        "native_unit": "tonnes",
        "description": "Carbon black feedstock",
    },
}

# Unit conversion factors to native units
# Keyed by (from_unit, to_unit) for common conversions
UNIT_CONVERSIONS: Dict[Tuple[str, str], Decimal] = {
    # Volume conversions
    ("liters", "liters"): Decimal("1"),
    ("m3", "m3"): Decimal("1"),
    ("m3", "liters"): Decimal("1000"),
    ("liters", "m3"): Decimal("0.001"),
    ("gallons", "liters"): Decimal("3.78541"),
    ("liters", "gallons"): Decimal("0.264172"),
    ("gallons", "m3"): Decimal("0.003785"),
    # Mass conversions
    ("kg", "kg"): Decimal("1"),
    ("tonnes", "tonnes"): Decimal("1"),
    ("tonnes", "kg"): Decimal("1000"),
    ("kg", "tonnes"): Decimal("0.001"),
}

# DQI base scores by method -- fuels are well-characterized
DQI_BASE_SCORES: Dict[str, Decimal] = {
    "fuel_sales": Decimal("90"),
    "feedstock": Decimal("85"),
}

# DQI tier multipliers
DQI_TIER_MULTIPLIERS: Dict[str, Decimal] = {
    "tier_1": Decimal("1.00"),
    "tier_2": Decimal("0.90"),
    "tier_3": Decimal("0.75"),
}

# Uncertainty half-width (95% CI) -- fuels well-characterized at +/- 10%
UNCERTAINTY_HALFWIDTHS: Dict[str, Decimal] = {
    "fuel_sales": Decimal("0.10"),
    "feedstock": Decimal("0.10"),
}


# ==============================================================================
# HASH UTILITY
# ==============================================================================


def _compute_provenance_hash(*parts: Any) -> str:
    """
    Compute SHA-256 provenance hash from variable inputs.

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
# INPUT / OUTPUT DATA STRUCTURES
# ==============================================================================


class FuelSaleInput:
    """
    Input data for a single fuel sale line item.

    Attributes:
        fuel_id: Unique identifier for this fuel sale record.
        fuel_type: Fuel type code (e.g., "GASOLINE", "DIESEL").
        volume_sold: Volume or mass sold.
        unit: Unit of volume_sold (liters, m3, kg, tonnes, gallons).
        region: Optional region for reporting purposes.
        data_quality_tier: Data quality tier.
        tenant_id: Tenant identifier for multi-tenancy.
    """

    __slots__ = (
        "fuel_id",
        "fuel_type",
        "volume_sold",
        "unit",
        "region",
        "data_quality_tier",
        "tenant_id",
    )

    def __init__(
        self,
        fuel_id: str,
        fuel_type: str,
        volume_sold: Decimal,
        unit: str = "liters",
        region: Optional[str] = None,
        data_quality_tier: str = "tier_1",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize FuelSaleInput."""
        self.fuel_id = fuel_id
        self.fuel_type = fuel_type
        self.volume_sold = volume_sold
        self.unit = unit
        self.region = region
        self.data_quality_tier = data_quality_tier
        self.tenant_id = tenant_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for hashing and provenance."""
        return {
            "fuel_id": self.fuel_id,
            "fuel_type": self.fuel_type,
            "volume_sold": str(self.volume_sold),
            "unit": self.unit,
            "region": self.region,
            "data_quality_tier": self.data_quality_tier,
            "tenant_id": self.tenant_id,
        }


class FeedstockInput:
    """
    Input data for a single feedstock sale.

    Attributes:
        feedstock_id: Unique identifier for this feedstock record.
        feedstock_type: Feedstock type code.
        mass_sold: Mass sold in the native unit (typically tonnes).
        unit: Unit for mass_sold (kg or tonnes).
        carbon_content_override: Optional override for carbon content fraction.
        oxidation_factor_override: Optional override for oxidation factor.
        region: Optional region for reporting.
        data_quality_tier: Data quality tier.
        tenant_id: Tenant identifier.
    """

    __slots__ = (
        "feedstock_id",
        "feedstock_type",
        "mass_sold",
        "unit",
        "carbon_content_override",
        "oxidation_factor_override",
        "region",
        "data_quality_tier",
        "tenant_id",
    )

    def __init__(
        self,
        feedstock_id: str,
        feedstock_type: str,
        mass_sold: Decimal,
        unit: str = "tonnes",
        carbon_content_override: Optional[Decimal] = None,
        oxidation_factor_override: Optional[Decimal] = None,
        region: Optional[str] = None,
        data_quality_tier: str = "tier_1",
        tenant_id: Optional[str] = None,
    ) -> None:
        """Initialize FeedstockInput."""
        self.feedstock_id = feedstock_id
        self.feedstock_type = feedstock_type
        self.mass_sold = mass_sold
        self.unit = unit
        self.carbon_content_override = carbon_content_override
        self.oxidation_factor_override = oxidation_factor_override
        self.region = region
        self.data_quality_tier = data_quality_tier
        self.tenant_id = tenant_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for hashing and provenance."""
        return {
            "feedstock_id": self.feedstock_id,
            "feedstock_type": self.feedstock_type,
            "mass_sold": str(self.mass_sold),
            "unit": self.unit,
            "carbon_content_override": str(self.carbon_content_override) if self.carbon_content_override is not None else None,
            "oxidation_factor_override": str(self.oxidation_factor_override) if self.oxidation_factor_override is not None else None,
            "region": self.region,
            "data_quality_tier": self.data_quality_tier,
            "tenant_id": self.tenant_id,
        }


class FuelEmissionResult:
    """
    Result from a single fuel combustion emissions calculation.

    Attributes:
        fuel_id: Fuel sale record identifier.
        fuel_type: Fuel type code.
        volume_sold: Volume/mass sold in native units.
        volume_native_unit: Native unit used for the EF.
        emission_factor: Combustion emission factor applied.
        ef_unit: Emission factor unit description.
        total_co2e: Total emissions (kgCO2e).
        ncv_mj: Net calorific value (MJ per native unit).
        dqi_score: Data quality indicator score (0-100).
        uncertainty_lower: Lower bound of 95% CI.
        uncertainty_upper: Upper bound of 95% CI.
        provenance_hash: SHA-256 provenance hash.
    """

    __slots__ = (
        "fuel_id",
        "fuel_type",
        "volume_sold",
        "volume_native_unit",
        "emission_factor",
        "ef_unit",
        "total_co2e",
        "ncv_mj",
        "dqi_score",
        "uncertainty_lower",
        "uncertainty_upper",
        "provenance_hash",
    )

    def __init__(
        self,
        fuel_id: str,
        fuel_type: str,
        volume_sold: Decimal,
        volume_native_unit: str,
        emission_factor: Decimal,
        ef_unit: str,
        total_co2e: Decimal,
        ncv_mj: Decimal,
        dqi_score: Decimal,
        uncertainty_lower: Decimal,
        uncertainty_upper: Decimal,
        provenance_hash: str,
    ) -> None:
        """Initialize FuelEmissionResult."""
        self.fuel_id = fuel_id
        self.fuel_type = fuel_type
        self.volume_sold = volume_sold
        self.volume_native_unit = volume_native_unit
        self.emission_factor = emission_factor
        self.ef_unit = ef_unit
        self.total_co2e = total_co2e
        self.ncv_mj = ncv_mj
        self.dqi_score = dqi_score
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "fuel_id": self.fuel_id,
            "fuel_type": self.fuel_type,
            "volume_sold": str(self.volume_sold),
            "volume_native_unit": self.volume_native_unit,
            "emission_factor": str(self.emission_factor),
            "ef_unit": self.ef_unit,
            "total_co2e": str(self.total_co2e),
            "ncv_mj": str(self.ncv_mj),
            "dqi_score": str(self.dqi_score),
            "uncertainty_lower": str(self.uncertainty_lower),
            "uncertainty_upper": str(self.uncertainty_upper),
            "provenance_hash": self.provenance_hash,
        }


class FeedstockEmissionResult:
    """
    Result from a single feedstock oxidation emissions calculation.

    Attributes:
        feedstock_id: Feedstock record identifier.
        feedstock_type: Feedstock type code.
        mass_sold_tonnes: Mass sold in tonnes.
        carbon_content: Carbon content fraction used.
        oxidation_factor: Oxidation factor used.
        co2_c_ratio: Molecular weight ratio (44/12) used.
        total_co2e: Total emissions (kgCO2e).
        dqi_score: Data quality indicator score (0-100).
        uncertainty_lower: Lower bound of 95% CI.
        uncertainty_upper: Upper bound of 95% CI.
        provenance_hash: SHA-256 provenance hash.
    """

    __slots__ = (
        "feedstock_id",
        "feedstock_type",
        "mass_sold_tonnes",
        "carbon_content",
        "oxidation_factor",
        "co2_c_ratio",
        "total_co2e",
        "dqi_score",
        "uncertainty_lower",
        "uncertainty_upper",
        "provenance_hash",
    )

    def __init__(
        self,
        feedstock_id: str,
        feedstock_type: str,
        mass_sold_tonnes: Decimal,
        carbon_content: Decimal,
        oxidation_factor: Decimal,
        co2_c_ratio: Decimal,
        total_co2e: Decimal,
        dqi_score: Decimal,
        uncertainty_lower: Decimal,
        uncertainty_upper: Decimal,
        provenance_hash: str,
    ) -> None:
        """Initialize FeedstockEmissionResult."""
        self.feedstock_id = feedstock_id
        self.feedstock_type = feedstock_type
        self.mass_sold_tonnes = mass_sold_tonnes
        self.carbon_content = carbon_content
        self.oxidation_factor = oxidation_factor
        self.co2_c_ratio = co2_c_ratio
        self.total_co2e = total_co2e
        self.dqi_score = dqi_score
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "feedstock_id": self.feedstock_id,
            "feedstock_type": self.feedstock_type,
            "mass_sold_tonnes": str(self.mass_sold_tonnes),
            "carbon_content": str(self.carbon_content),
            "oxidation_factor": str(self.oxidation_factor),
            "co2_c_ratio": str(self.co2_c_ratio),
            "total_co2e": str(self.total_co2e),
            "dqi_score": str(self.dqi_score),
            "uncertainty_lower": str(self.uncertainty_lower),
            "uncertainty_upper": str(self.uncertainty_upper),
            "provenance_hash": self.provenance_hash,
        }


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class FuelsAndFeedstocksCalculatorEngine:
    """
    Thread-safe singleton engine for fuel sales and feedstock emissions.

    Implements the complete fuels-and-feedstocks calculation pipeline per GHG
    Protocol Scope 3 Category 11, covering:
    - Fuel combustion by end users (Formula G)
    - Feedstock oxidation by downstream processes (Formula H)

    All arithmetic uses Python Decimal with ROUND_HALF_UP quantization to
    8 decimal places for regulatory precision.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic. No LLM calls are used for any numeric computation.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The
        _calculation_count is protected by a dedicated lock.

    Attributes:
        _calculation_count: Total number of calculations performed.
        _count_lock: Lock protecting the calculation counter.

    Example:
        >>> engine = FuelsAndFeedstocksCalculatorEngine()
        >>> fuels = [FuelSaleInput(
        ...     fuel_id="SALE-001", fuel_type="DIESEL",
        ...     volume_sold=Decimal("500000"), unit="liters",
        ... )]
        >>> result = engine.calculate_fuel_sales(fuels, "ORG-001", 2024)
        >>> assert result["total_co2e"] > Decimal("0")
    """

    _instance: Optional["FuelsAndFeedstocksCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FuelsAndFeedstocksCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the fuels and feedstocks calculator engine (once)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._calculation_count: int = 0
        self._count_lock: threading.Lock = threading.Lock()

        logger.info(
            "%s initialized: agent=%s, version=%s, "
            "fuel_types=%d, feedstock_types=%d, CO2_C_ratio=%s",
            ENGINE_NAME,
            AGENT_ID,
            VERSION,
            len(FUEL_EMISSION_FACTORS),
            len(FEEDSTOCK_PROPERTIES),
            str(_q(_CO2_C_RATIO)),
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

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def validate_inputs(
        self,
        items: List[Union[FuelSaleInput, FeedstockInput]],
    ) -> List[str]:
        """
        Validate a list of fuel sale or feedstock inputs.

        Checks:
        - List is non-empty
        - Each item has a valid ID
        - Volumes/masses are positive
        - Fuel/feedstock types are recognized
        - Units are recognized

        Args:
            items: List of FuelSaleInput or FeedstockInput objects.

        Returns:
            List of validation error messages. Empty means all valid.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> errors = engine.validate_inputs([])
            >>> assert "Input list must not be empty" in errors
        """
        errors: List[str] = []

        if not items:
            errors.append("Input list must not be empty")
            return errors

        valid_fuel_types = set(FUEL_EMISSION_FACTORS.keys())
        valid_feedstock_types = set(FEEDSTOCK_PROPERTIES.keys())
        valid_volume_units = {u.value for u in VolumeUnit}

        for i, item in enumerate(items):
            if isinstance(item, FuelSaleInput):
                prefix = f"FuelSale[{i}] (id={item.fuel_id})"

                if not item.fuel_id or not item.fuel_id.strip():
                    errors.append(f"{prefix}: fuel_id must be non-empty")

                if item.volume_sold <= _ZERO:
                    errors.append(
                        f"{prefix}: volume_sold must be positive, got {item.volume_sold}"
                    )

                if item.fuel_type.upper() not in valid_fuel_types:
                    errors.append(
                        f"{prefix}: unrecognized fuel_type '{item.fuel_type}'. "
                        f"Valid types: {sorted(valid_fuel_types)}"
                    )

                if item.unit.lower() not in valid_volume_units:
                    errors.append(
                        f"{prefix}: unrecognized unit '{item.unit}'. "
                        f"Valid units: {sorted(valid_volume_units)}"
                    )

            elif isinstance(item, FeedstockInput):
                prefix = f"Feedstock[{i}] (id={item.feedstock_id})"

                if not item.feedstock_id or not item.feedstock_id.strip():
                    errors.append(f"{prefix}: feedstock_id must be non-empty")

                if item.mass_sold <= _ZERO:
                    errors.append(
                        f"{prefix}: mass_sold must be positive, got {item.mass_sold}"
                    )

                if item.feedstock_type.upper() not in valid_feedstock_types:
                    errors.append(
                        f"{prefix}: unrecognized feedstock_type '{item.feedstock_type}'. "
                        f"Valid types: {sorted(valid_feedstock_types)}"
                    )

                if item.carbon_content_override is not None:
                    if item.carbon_content_override <= _ZERO or item.carbon_content_override > _ONE:
                        errors.append(
                            f"{prefix}: carbon_content must be in (0, 1], "
                            f"got {item.carbon_content_override}"
                        )

                if item.oxidation_factor_override is not None:
                    if item.oxidation_factor_override <= _ZERO or item.oxidation_factor_override > _ONE:
                        errors.append(
                            f"{prefix}: oxidation_factor must be in (0, 1], "
                            f"got {item.oxidation_factor_override}"
                        )
            else:
                errors.append(
                    f"Item[{i}]: unrecognized input type '{type(item).__name__}'. "
                    "Expected FuelSaleInput or FeedstockInput."
                )

        if errors:
            logger.warning(
                "Input validation found %d error(s) across %d item(s)",
                len(errors), len(items),
            )

        return errors

    # ==========================================================================
    # UNIT CONVERSION
    # ==========================================================================

    def convert_units(
        self,
        amount: Decimal,
        from_unit: str,
        to_unit: str,
        fuel_type: Optional[str] = None,
    ) -> Decimal:
        """
        Convert between volume/mass units for fuels and feedstocks.

        Supports direct conversions (liters<->m3, kg<->tonnes, gallons->liters)
        and density-based conversions (liters<->kg) when fuel_type is provided.

        Args:
            amount: Quantity to convert.
            from_unit: Source unit (liters, m3, kg, tonnes, gallons).
            to_unit: Target unit.
            fuel_type: Optional fuel type code for density-based conversions.

        Returns:
            Converted amount, quantized to 8 decimal places.

        Raises:
            ValueError: If conversion is not supported.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> engine.convert_units(Decimal("1000"), "gallons", "liters")
            Decimal('3785.41000000')
        """
        from_lower = from_unit.lower()
        to_lower = to_unit.lower()

        # Identity conversion
        if from_lower == to_lower:
            return _q(amount)

        # Direct conversion from table
        key = (from_lower, to_lower)
        if key in UNIT_CONVERSIONS:
            return _q(amount * UNIT_CONVERSIONS[key])

        # Density-based conversion: liters <-> kg
        if fuel_type is not None:
            fuel_data = FUEL_EMISSION_FACTORS.get(fuel_type.upper())
            if fuel_data is not None:
                density = fuel_data.get("density_kg_per_l")
                if density is not None:
                    if from_lower == "liters" and to_lower == "kg":
                        return _q(amount * density)
                    if from_lower == "kg" and to_lower == "liters":
                        return _q(amount / density)
                    if from_lower == "kg" and to_lower == "tonnes":
                        return _q(amount * Decimal("0.001"))
                    if from_lower == "tonnes" and to_lower == "kg":
                        return _q(amount * Decimal("1000"))
                    if from_lower == "liters" and to_lower == "tonnes":
                        kg = amount * density
                        return _q(kg * Decimal("0.001"))
                    if from_lower == "tonnes" and to_lower == "liters":
                        kg = amount * Decimal("1000")
                        return _q(kg / density)

                density_m3 = fuel_data.get("density_kg_per_m3")
                if density_m3 is not None:
                    if from_lower == "m3" and to_lower == "kg":
                        return _q(amount * density_m3)
                    if from_lower == "kg" and to_lower == "m3":
                        return _q(amount / density_m3)

        raise ValueError(
            f"Cannot convert from '{from_unit}' to '{to_unit}'"
            + (f" for fuel_type '{fuel_type}'" if fuel_type else "")
            + ". Provide fuel_type for density-based conversions."
        )

    # ==========================================================================
    # EMISSION FACTOR RESOLUTION
    # ==========================================================================

    def _resolve_fuel_ef(self, fuel_type: str) -> Dict[str, Any]:
        """
        Resolve fuel emission factor data by fuel type code.

        Args:
            fuel_type: Fuel type code.

        Returns:
            Dictionary with ef, ncv_mj, native_unit, etc.

        Raises:
            ValueError: If fuel type is not recognized.
        """
        fuel_upper = fuel_type.upper()
        fuel_data = FUEL_EMISSION_FACTORS.get(fuel_upper)
        if fuel_data is None:
            available = sorted(FUEL_EMISSION_FACTORS.keys())
            raise ValueError(
                f"Fuel type '{fuel_upper}' not found. Available: {available}"
            )
        return fuel_data

    def _resolve_feedstock_props(self, feedstock_type: str) -> Dict[str, Any]:
        """
        Resolve feedstock properties by feedstock type code.

        Args:
            feedstock_type: Feedstock type code.

        Returns:
            Dictionary with carbon_content, oxidation_factor, etc.

        Raises:
            ValueError: If feedstock type is not recognized.
        """
        fs_upper = feedstock_type.upper()
        fs_data = FEEDSTOCK_PROPERTIES.get(fs_upper)
        if fs_data is None:
            available = sorted(FEEDSTOCK_PROPERTIES.keys())
            raise ValueError(
                f"Feedstock type '{fs_upper}' not found. Available: {available}"
            )
        return fs_data

    # ==========================================================================
    # SINGLE-ITEM CALCULATIONS
    # ==========================================================================

    def calculate_single_fuel(
        self,
        fuel_sale: FuelSaleInput,
    ) -> FuelEmissionResult:
        """
        Calculate emissions for a single fuel sale line item.

        Formula: E = volume_in_native_units x EF_combustion

        Converts the input volume to the native unit of the emission factor
        before multiplying.

        Args:
            fuel_sale: Fuel sale input data.

        Returns:
            FuelEmissionResult with emissions and provenance.

        Raises:
            ValueError: If fuel type is not recognized.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> fuel = FuelSaleInput(
            ...     fuel_id="SALE-001", fuel_type="DIESEL",
            ...     volume_sold=Decimal("100000"), unit="liters",
            ... )
            >>> result = engine.calculate_single_fuel(fuel)
            >>> result.total_co2e > Decimal("0")
            True
        """
        fuel_data = self._resolve_fuel_ef(fuel_sale.fuel_type)
        native_unit = fuel_data["native_unit"]
        ef = fuel_data["ef"]
        ncv = fuel_data["ncv_mj"]

        # Convert to native units if necessary
        volume_native = self._convert_to_native(
            fuel_sale.volume_sold,
            fuel_sale.unit,
            native_unit,
            fuel_sale.fuel_type,
        )

        # E = volume x EF (ZERO-HALLUCINATION deterministic arithmetic)
        total_co2e = _q(volume_native * ef)

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(
            fuel_sale.data_quality_tier, "fuel_sales"
        )
        uncertainty = self.compute_uncertainty(total_co2e, "fuel_sales")

        # EF unit description
        ef_unit = f"kgCO2e/{native_unit}"
        if native_unit == "liters":
            ef_unit = "kgCO2e/L"
        elif native_unit == "m3":
            ef_unit = "kgCO2e/m3"
        elif native_unit == "kg":
            ef_unit = "kgCO2e/kg"

        # Provenance
        provenance_hash = self._build_provenance(
            "fuel_combustion",
            fuel_sale.to_dict(),
            {
                "volume_native": str(volume_native),
                "native_unit": native_unit,
                "ef": str(ef),
                "total_co2e": str(total_co2e),
            },
        )

        return FuelEmissionResult(
            fuel_id=fuel_sale.fuel_id,
            fuel_type=fuel_sale.fuel_type.upper(),
            volume_sold=volume_native,
            volume_native_unit=native_unit,
            emission_factor=ef,
            ef_unit=ef_unit,
            total_co2e=total_co2e,
            ncv_mj=ncv,
            dqi_score=dqi_score,
            uncertainty_lower=uncertainty["lower"],
            uncertainty_upper=uncertainty["upper"],
            provenance_hash=provenance_hash,
        )

    def calculate_single_feedstock(
        self,
        feedstock: FeedstockInput,
    ) -> FeedstockEmissionResult:
        """
        Calculate emissions for a single feedstock sale.

        Formula: E = M_sold x C_content x OF x (44/12) x 1000
        The result is in kgCO2e (M_sold in tonnes, so multiply by 1000 to get kg,
        then the carbon fraction and oxidation give kgC, and 44/12 converts to kgCO2).

        Args:
            feedstock: Feedstock input data.

        Returns:
            FeedstockEmissionResult with emissions and provenance.

        Raises:
            ValueError: If feedstock type is not recognized.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> fs = FeedstockInput(
            ...     feedstock_id="FS-001", feedstock_type="PETROLEUM_COKE",
            ...     mass_sold=Decimal("1000"), unit="tonnes",
            ... )
            >>> result = engine.calculate_single_feedstock(fs)
            >>> result.total_co2e > Decimal("0")
            True
        """
        fs_props = self._resolve_feedstock_props(feedstock.feedstock_type)

        # Use overrides if provided, otherwise use default values
        carbon_content = (
            feedstock.carbon_content_override
            if feedstock.carbon_content_override is not None
            else fs_props["carbon_content"]
        )
        oxidation_factor = (
            feedstock.oxidation_factor_override
            if feedstock.oxidation_factor_override is not None
            else fs_props["oxidation_factor"]
        )

        # Convert mass to tonnes if not already
        mass_tonnes = self._convert_mass_to_tonnes(
            feedstock.mass_sold, feedstock.unit
        )

        # E = M_sold(tonnes) x C_content x OF x (44/12) x 1000(kg/tonne)
        # Result is in kgCO2
        co2_c_ratio = _q(_CO2_C_RATIO)
        tonnes_to_kg = Decimal("1000")

        total_co2e = _q(
            mass_tonnes
            * carbon_content
            * oxidation_factor
            * co2_c_ratio
            * tonnes_to_kg
        )

        # DQI and uncertainty
        dqi_score = self.compute_dqi_score(
            feedstock.data_quality_tier, "feedstock"
        )
        uncertainty = self.compute_uncertainty(total_co2e, "feedstock")

        # Provenance
        provenance_hash = self._build_provenance(
            "feedstock_oxidation",
            feedstock.to_dict(),
            {
                "mass_tonnes": str(mass_tonnes),
                "carbon_content": str(carbon_content),
                "oxidation_factor": str(oxidation_factor),
                "co2_c_ratio": str(co2_c_ratio),
                "total_co2e": str(total_co2e),
            },
        )

        return FeedstockEmissionResult(
            feedstock_id=feedstock.feedstock_id,
            feedstock_type=feedstock.feedstock_type.upper(),
            mass_sold_tonnes=mass_tonnes,
            carbon_content=carbon_content,
            oxidation_factor=oxidation_factor,
            co2_c_ratio=co2_c_ratio,
            total_co2e=total_co2e,
            dqi_score=dqi_score,
            uncertainty_lower=uncertainty["lower"],
            uncertainty_upper=uncertainty["upper"],
            provenance_hash=provenance_hash,
        )

    # ==========================================================================
    # BATCH CALCULATION METHODS
    # ==========================================================================

    def calculate_fuel_sales(
        self,
        fuel_sales: List[FuelSaleInput],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for multiple fuel sales.

        Formula per fuel: E = volume x EF_combustion

        Args:
            fuel_sales: List of fuel sale inputs.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with fuel results, total_co2e, and provenance.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> fuels = [FuelSaleInput(
            ...     fuel_id="SALE-001", fuel_type="GASOLINE",
            ...     volume_sold=Decimal("1000000"), unit="liters",
            ... )]
            >>> result = engine.calculate_fuel_sales(fuels, "ORG-001", 2024)
            >>> result["total_co2e"] > Decimal("0")
            True
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "%s fuel_sales calculation #%d: org=%s, year=%d, items=%d",
            ENGINE_NAME, calc_number, org_id, year, len(fuel_sales),
        )

        # Validate
        errors = self.validate_inputs(fuel_sales)
        if errors:
            raise ValueError(
                f"Input validation failed with {len(errors)} error(s): "
                + "; ".join(errors[:5])
            )

        results: List[FuelEmissionResult] = []
        total_co2e = _ZERO
        error_list: List[Dict[str, str]] = []

        for fuel_sale in fuel_sales:
            try:
                result = self.calculate_single_fuel(fuel_sale)
                results.append(result)
                total_co2e = _q(total_co2e + result.total_co2e)
            except Exception as exc:
                logger.error(
                    "Fuel calculation failed for '%s': %s",
                    fuel_sale.fuel_id, exc, exc_info=True,
                )
                error_list.append({
                    "fuel_id": fuel_sale.fuel_id,
                    "error": str(exc),
                })

        # Aggregate DQI
        avg_dqi = _ZERO
        if results:
            dqi_sum = sum(r.dqi_score for r in results)
            avg_dqi = _q(dqi_sum / Decimal(str(len(results))))

        # Aggregate uncertainty
        agg_uncertainty = self.compute_uncertainty(total_co2e, "fuel_sales")

        # Provenance
        batch_provenance = self._build_provenance(
            "fuel_sales_batch",
            {"org_id": org_id, "year": year, "item_count": len(fuel_sales)},
            {"total_co2e": str(total_co2e), "result_count": len(results)},
        )

        duration = time.monotonic() - start_time

        logger.info(
            "%s fuel_sales calculation #%d complete: "
            "total_co2e=%s kgCO2e, items=%d, duration=%.3fs",
            ENGINE_NAME, calc_number, total_co2e, len(results), duration,
        )

        return {
            "method": "fuel_sales",
            "org_id": org_id,
            "reporting_year": year,
            "fuel_results": [r.to_dict() for r in results],
            "total_co2e": total_co2e,
            "total_tco2e": _q(total_co2e / Decimal("1000")),
            "item_count": len(results),
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

    def calculate_feedstock(
        self,
        feedstocks: List[FeedstockInput],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for multiple feedstock sales.

        Formula per feedstock: E = mass x C_content x OF x (44/12) x 1000

        Args:
            feedstocks: List of feedstock inputs.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with feedstock results, total_co2e, and provenance.

        Raises:
            ValueError: If inputs are invalid.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> feedstocks = [FeedstockInput(
            ...     feedstock_id="FS-001", feedstock_type="PETROLEUM_COKE",
            ...     mass_sold=Decimal("5000"), unit="tonnes",
            ... )]
            >>> result = engine.calculate_feedstock(feedstocks, "ORG-001", 2024)
            >>> result["total_co2e"] > Decimal("0")
            True
        """
        start_time = time.monotonic()
        calc_number = self._increment_count()

        logger.info(
            "%s feedstock calculation #%d: org=%s, year=%d, items=%d",
            ENGINE_NAME, calc_number, org_id, year, len(feedstocks),
        )

        # Validate
        errors = self.validate_inputs(feedstocks)
        if errors:
            raise ValueError(
                f"Input validation failed with {len(errors)} error(s): "
                + "; ".join(errors[:5])
            )

        results: List[FeedstockEmissionResult] = []
        total_co2e = _ZERO
        error_list: List[Dict[str, str]] = []

        for feedstock in feedstocks:
            try:
                result = self.calculate_single_feedstock(feedstock)
                results.append(result)
                total_co2e = _q(total_co2e + result.total_co2e)
            except Exception as exc:
                logger.error(
                    "Feedstock calculation failed for '%s': %s",
                    feedstock.feedstock_id, exc, exc_info=True,
                )
                error_list.append({
                    "feedstock_id": feedstock.feedstock_id,
                    "error": str(exc),
                })

        # Aggregate DQI
        avg_dqi = _ZERO
        if results:
            dqi_sum = sum(r.dqi_score for r in results)
            avg_dqi = _q(dqi_sum / Decimal(str(len(results))))

        # Aggregate uncertainty
        agg_uncertainty = self.compute_uncertainty(total_co2e, "feedstock")

        # Provenance
        batch_provenance = self._build_provenance(
            "feedstock_batch",
            {"org_id": org_id, "year": year, "item_count": len(feedstocks)},
            {"total_co2e": str(total_co2e), "result_count": len(results)},
        )

        duration = time.monotonic() - start_time

        logger.info(
            "%s feedstock calculation #%d complete: "
            "total_co2e=%s kgCO2e, items=%d, duration=%.3fs",
            ENGINE_NAME, calc_number, total_co2e, len(results), duration,
        )

        return {
            "method": "feedstock",
            "org_id": org_id,
            "reporting_year": year,
            "feedstock_results": [r.to_dict() for r in results],
            "total_co2e": total_co2e,
            "total_tco2e": _q(total_co2e / Decimal("1000")),
            "item_count": len(results),
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
        items: List[Union[FuelSaleInput, FeedstockInput]],
        method: str,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Dispatch to the appropriate calculation method.

        Routes to calculate_fuel_sales or calculate_feedstock based on
        the method parameter.

        Args:
            items: List of FuelSaleInput or FeedstockInput.
            method: Calculation method ("fuel_sales" or "feedstock").
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with calculation results.

        Raises:
            ValueError: If method is not recognized.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> result = engine.calculate(fuel_inputs, "fuel_sales", "ORG-001", 2024)
        """
        method_lower = method.lower().strip()

        if method_lower == "fuel_sales":
            return self.calculate_fuel_sales(items, org_id, year)
        elif method_lower == "feedstock":
            return self.calculate_feedstock(items, org_id, year)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Valid methods: ['fuel_sales', 'feedstock']"
            )

    # ==========================================================================
    # DATA QUALITY & UNCERTAINTY
    # ==========================================================================

    def compute_dqi_score(
        self,
        data_quality_tier: str,
        method: str,
    ) -> Decimal:
        """
        Compute the Data Quality Indicator score.

        Base scores (fuels are well-characterized):
        - fuel_sales: 90
        - feedstock: 85

        Adjusted by tier:
        - tier_1: x 1.00
        - tier_2: x 0.90
        - tier_3: x 0.75

        Args:
            data_quality_tier: Data quality tier string.
            method: Calculation method.

        Returns:
            DQI score as Decimal (0-100).

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> engine.compute_dqi_score("tier_1", "fuel_sales")
            Decimal('90.00000000')
        """
        base = DQI_BASE_SCORES.get(method, Decimal("85"))
        tier_mult = DQI_TIER_MULTIPLIERS.get(
            data_quality_tier, Decimal("0.90")
        )
        score = _q(base * tier_mult)

        # Cap at 100
        if score > Decimal("100"):
            score = Decimal("100.00000000")

        return score

    def compute_uncertainty(
        self,
        emissions: Decimal,
        method: str,
    ) -> Dict[str, Decimal]:
        """
        Compute uncertainty bounds for emissions.

        Fuels and feedstocks are well-characterized with +/- 10% uncertainty.

        Args:
            emissions: Total emissions value (kgCO2e).
            method: Calculation method string.

        Returns:
            Dictionary with "lower" and "upper" bounds.

        Example:
            >>> engine = FuelsAndFeedstocksCalculatorEngine()
            >>> bounds = engine.compute_uncertainty(Decimal("1000"), "fuel_sales")
            >>> bounds["lower"]
            Decimal('900.00000000')
        """
        halfwidth = UNCERTAINTY_HALFWIDTHS.get(method, Decimal("0.10"))

        lower = _q(emissions * (_ONE - halfwidth))
        upper = _q(emissions * (_ONE + halfwidth))

        if lower < _ZERO:
            lower = _ZERO

        return {
            "lower": lower,
            "upper": upper,
            "halfwidth_pct": _q(halfwidth * Decimal("100")),
            "method": method,
        }

    # ==========================================================================
    # UNIT CONVERSION HELPERS
    # ==========================================================================

    def _convert_to_native(
        self,
        amount: Decimal,
        from_unit: str,
        native_unit: str,
        fuel_type: str,
    ) -> Decimal:
        """
        Convert an amount to the native unit of the emission factor.

        Args:
            amount: Quantity to convert.
            from_unit: Source unit.
            native_unit: Target native unit.
            fuel_type: Fuel type code for density lookups.

        Returns:
            Converted amount in native units.
        """
        from_lower = from_unit.lower()
        native_lower = native_unit.lower()

        if from_lower == native_lower:
            return _q(amount)

        return self.convert_units(amount, from_unit, native_unit, fuel_type)

    def _convert_mass_to_tonnes(
        self,
        mass: Decimal,
        unit: str,
    ) -> Decimal:
        """
        Convert a mass value to tonnes.

        Args:
            mass: Mass value.
            unit: Source unit (kg or tonnes).

        Returns:
            Mass in tonnes, quantized.
        """
        unit_lower = unit.lower()

        if unit_lower == "tonnes":
            return _q(mass)
        elif unit_lower == "kg":
            return _q(mass * Decimal("0.001"))
        else:
            raise ValueError(
                f"Cannot convert mass from '{unit}' to tonnes. "
                "Supported: kg, tonnes."
            )

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

        Args:
            method: Calculation method string.
            inputs: Input data (dict or object with to_dict).
            result: Result data (dict or object with to_dict).

        Returns:
            SHA-256 provenance hash (64 hex characters).
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
            Integer count.
        """
        with self._count_lock:
            return self._calculation_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the engine state and configuration.

        Returns:
            Dictionary with engine metadata and statistics.
        """
        return {
            "engine": ENGINE_NAME,
            "engine_number": ENGINE_NUMBER,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "calculation_count": self.get_calculation_count(),
            "fuel_types": len(FUEL_EMISSION_FACTORS),
            "feedstock_types": len(FEEDSTOCK_PROPERTIES),
            "co2_c_ratio": str(_q(_CO2_C_RATIO)),
            "methods": ["fuel_sales", "feedstock"],
        }

    def get_supported_fuel_types(self) -> List[str]:
        """Get sorted list of supported fuel type codes."""
        return sorted(FUEL_EMISSION_FACTORS.keys())

    def get_supported_feedstock_types(self) -> List[str]:
        """Get sorted list of supported feedstock type codes."""
        return sorted(FEEDSTOCK_PROPERTIES.keys())

    def get_fuel_ef(self, fuel_type: str) -> Optional[Dict[str, Any]]:
        """
        Get fuel emission factor data for a specific fuel type.

        Args:
            fuel_type: Fuel type code.

        Returns:
            Fuel data dict or None if not found.
        """
        return FUEL_EMISSION_FACTORS.get(fuel_type.upper())

    def get_feedstock_props(self, feedstock_type: str) -> Optional[Dict[str, Any]]:
        """
        Get feedstock properties for a specific feedstock type.

        Args:
            feedstock_type: Feedstock type code.

        Returns:
            Feedstock properties dict or None if not found.
        """
        return FEEDSTOCK_PROPERTIES.get(feedstock_type.upper())

    def get_co2_c_ratio(self) -> Decimal:
        """
        Get the molecular weight ratio CO2/C = 44/12.

        Returns:
            The ratio as Decimal, quantized to 8 dp.
        """
        return _q(_CO2_C_RATIO)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: For use in test fixtures only. Do not call in production.
        """
        with cls._lock:
            cls._instance = None


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================

_engine_instance: Optional[FuelsAndFeedstocksCalculatorEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_fuels_feedstocks_calculator() -> FuelsAndFeedstocksCalculatorEngine:
    """
    Get the singleton FuelsAndFeedstocksCalculatorEngine instance.

    Thread-safe accessor.

    Returns:
        FuelsAndFeedstocksCalculatorEngine singleton instance.
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = FuelsAndFeedstocksCalculatorEngine()
        return _engine_instance


def reset_fuels_feedstocks_calculator() -> None:
    """
    Reset the module-level calculator instance (for testing only).

    Warning: For use in test fixtures only. Do not call in production.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    FuelsAndFeedstocksCalculatorEngine.reset()


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
    "FuelType",
    "FeedstockType",
    "VolumeUnit",
    "FuelFeedstockMethod",
    "DataQualityTier",
    # EF Tables
    "FUEL_EMISSION_FACTORS",
    "FEEDSTOCK_PROPERTIES",
    "UNIT_CONVERSIONS",
    "DQI_BASE_SCORES",
    "DQI_TIER_MULTIPLIERS",
    "UNCERTAINTY_HALFWIDTHS",
    # Data Models
    "FuelSaleInput",
    "FeedstockInput",
    "FuelEmissionResult",
    "FeedstockEmissionResult",
    # Engine
    "FuelsAndFeedstocksCalculatorEngine",
    "get_fuels_feedstocks_calculator",
    "reset_fuels_feedstocks_calculator",
    # Utility
    "_compute_provenance_hash",
]
