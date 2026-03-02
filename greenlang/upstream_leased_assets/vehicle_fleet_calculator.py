# -*- coding: utf-8 -*-
"""
VehicleFleetCalculatorEngine - Engine 3: Upstream Leased Assets Agent (AGENT-MRV-021)

Core calculation engine for vehicle fleet emissions covering leased cars, SUVs,
vans, and trucks. Supports distance-based, fuel-based, electric vehicle,
fleet aggregate, spend-based, batch, and annual estimation methods.

This engine implements deterministic Decimal-based emissions calculations
for all leased vehicle types, following DEFRA 2024 emission factors and
the GHG Protocol Scope 3 Category 8 methodology.

Primary Formulae:
    Distance-Based (per-vkm):
        ttw_co2e = annual_km x count x ef_per_km x age_factor
        wtt_co2e = annual_km x count x wtt_per_km x age_factor
        total    = ttw_co2e + wtt_co2e

    Fuel-Based (per-litre):
        ttw_co2e = fuel_litres x ef_per_litre
        wtt_co2e = fuel_litres x wtt_per_litre
        total    = ttw_co2e + wtt_co2e

    Electric Vehicle (per-km via kWh):
        energy_kwh = annual_km x consumption_per_km x count
        co2e       = energy_kwh x grid_ef(country_code)

    Fleet Aggregate:
        Iterate per-vehicle, sum to fleet totals

    Spend-Based (EEIO):
        co2e = deflated_usd x eeio_factor

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from DEFRA 2024 conversion factors

Supports:
    - 8 vehicle types (small/medium/large car, SUV, light/heavy van,
      light/heavy truck)
    - 4 fuel types per car (petrol, diesel, hybrid, bev)
    - 2 fuel types per van (diesel, bev)
    - 1 fuel type per truck (diesel)
    - Vehicle age degradation (new, mid, old, vintage)
    - Well-to-tank (WTT) upstream emissions
    - Electric vehicle grid-based calculations with eGRID subregion support
    - Fleet-level aggregation with per-vehicle breakdown
    - Spend-based fallback using EEIO factors with CPI deflation
    - Batch processing for multiple vehicle calculations
    - Quick annual emission estimation
    - Input validation with detailed error messages
    - SHA-256 provenance hash integration for audit trails

Example:
    >>> from greenlang.upstream_leased_assets.vehicle_fleet_calculator import (
    ...     get_vehicle_fleet_calculator,
    ... )
    >>> from decimal import Decimal
    >>> engine = get_vehicle_fleet_calculator()
    >>> result = engine.calculate_distance_based(
    ...     vehicle_type="medium_car",
    ...     fuel_type="diesel",
    ...     annual_km=Decimal("20000"),
    ...     count=5,
    ... )
    >>> assert result["co2e_kg"] > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-021 Upstream Leased Assets (GL-MRV-S3-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# ENGINE METADATA
# ==============================================================================

ENGINE_ID: str = "vehicle_fleet_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-008"
AGENT_COMPONENT: str = "AGENT-MRV-021"
VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_THOUSAND = Decimal("1000")
ROUNDING = ROUND_HALF_UP

# Unit conversion constants
_KM_PER_MILE = Decimal("1.60934")
_LITRES_PER_GALLON = Decimal("3.78541")

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# Hours in a year (used for EV annual estimates)
_HOURS_PER_YEAR = Decimal("8760")

# ==============================================================================
# VEHICLE EMISSION FACTORS (kgCO2e per km, TTW) - DEFRA 2024
# ==============================================================================

VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # small_car
    "small_car": {
        "petrol": Decimal("0.14930"),
        "diesel": Decimal("0.13105"),
        "hybrid": Decimal("0.09814"),
        "bev": Decimal("0.04350"),
    },
    # medium_car
    "medium_car": {
        "petrol": Decimal("0.18210"),
        "diesel": Decimal("0.16192"),
        "hybrid": Decimal("0.12050"),
        "bev": Decimal("0.05020"),
    },
    # large_car
    "large_car": {
        "petrol": Decimal("0.22180"),
        "diesel": Decimal("0.20867"),
        "hybrid": Decimal("0.16270"),
        "bev": Decimal("0.06730"),
    },
    # suv
    "suv": {
        "petrol": Decimal("0.20980"),
        "diesel": Decimal("0.18790"),
        "hybrid": Decimal("0.15030"),
        "bev": Decimal("0.06200"),
    },
    # light_van
    "light_van": {
        "diesel": Decimal("0.22430"),
        "bev": Decimal("0.07100"),
    },
    # heavy_van
    "heavy_van": {
        "diesel": Decimal("0.31200"),
        "bev": Decimal("0.09800"),
    },
    # light_truck
    "light_truck": {
        "diesel": Decimal("0.31200"),
    },
    # heavy_truck
    "heavy_truck": {
        "diesel": Decimal("0.58600"),
    },
}

# ==============================================================================
# VALID VEHICLE TYPES AND FUEL TYPES
# ==============================================================================

VALID_VEHICLE_TYPES: List[str] = [
    "small_car",
    "medium_car",
    "large_car",
    "suv",
    "light_van",
    "heavy_van",
    "light_truck",
    "heavy_truck",
]

VALID_FUEL_TYPES: List[str] = [
    "petrol",
    "diesel",
    "hybrid",
    "bev",
]

# ==============================================================================
# EV ENERGY CONSUMPTION (kWh per km) by vehicle type
# Source: DEFRA 2024 / EPA 2024 average BEV consumption
# ==============================================================================

EV_CONSUMPTION: Dict[str, Decimal] = {
    "small_car": Decimal("0.148"),
    "small": Decimal("0.148"),
    "medium_car": Decimal("0.171"),
    "medium": Decimal("0.171"),
    "large_car": Decimal("0.226"),
    "large": Decimal("0.226"),
    "suv": Decimal("0.209"),
    "light_van": Decimal("0.253"),
    "heavy_van": Decimal("0.349"),
}

# ==============================================================================
# FUEL EMISSION FACTORS (kgCO2e per litre) - DEFRA 2024
# For fuel-based calculation method
# ==============================================================================

FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "petrol": {
        "ef_per_litre": Decimal("2.31484"),
        "wtt_per_litre": Decimal("0.58549"),
    },
    "diesel": {
        "ef_per_litre": Decimal("2.68787"),
        "wtt_per_litre": Decimal("0.60927"),
    },
    "lpg": {
        "ef_per_litre": Decimal("1.55363"),
        "wtt_per_litre": Decimal("0.32149"),
    },
    "cng": {
        "ef_per_litre": Decimal("2.02130"),
        "wtt_per_litre": Decimal("0.46490"),
    },
    "e85": {
        "ef_per_litre": Decimal("1.61039"),
        "wtt_per_litre": Decimal("0.40260"),
    },
}

# ==============================================================================
# WTT RATIOS by fuel type
# Ratio of upstream (well-to-tank) to tailpipe (tank-to-wheel) emissions
# Source: DEFRA 2024
# ==============================================================================

WTT_RATIOS: Dict[str, Decimal] = {
    "petrol": Decimal("0.253"),
    "diesel": Decimal("0.225"),
    "hybrid": Decimal("0.253"),
    "bev": Decimal("0.000"),
}

# ==============================================================================
# VEHICLE AGE FACTORS
# Multiplier applied to per-km emission factors to account for age degradation
# Source: DEFRA 2024 / EMEP/EEA Emission Inventory Guidebook
# ==============================================================================

VEHICLE_AGE_FACTORS: Dict[str, Decimal] = {
    "new_0_3yr": Decimal("0.95"),
    "mid_4_7yr": Decimal("1.00"),
    "old_8_12yr": Decimal("1.08"),
    "vintage_13plus": Decimal("1.15"),
}

# ==============================================================================
# GRID EMISSION FACTORS (kgCO2e per kWh) by country
# Source: IEA 2024, EPA eGRID 2024
# ==============================================================================

GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.37120"),
    "GB": Decimal("0.20700"),
    "DE": Decimal("0.33800"),
    "FR": Decimal("0.05200"),
    "JP": Decimal("0.47100"),
    "CN": Decimal("0.55500"),
    "IN": Decimal("0.71600"),
    "AU": Decimal("0.65600"),
    "CA": Decimal("0.12000"),
    "BR": Decimal("0.07500"),
    "KR": Decimal("0.42200"),
    "IT": Decimal("0.25800"),
    "ES": Decimal("0.18100"),
    "NL": Decimal("0.33000"),
    "SE": Decimal("0.01200"),
    "NO": Decimal("0.00800"),
    "DK": Decimal("0.14000"),
    "FI": Decimal("0.07300"),
    "PL": Decimal("0.63500"),
    "AT": Decimal("0.09400"),
    "BE": Decimal("0.16200"),
    "CH": Decimal("0.01100"),
    "IE": Decimal("0.29600"),
    "PT": Decimal("0.18400"),
    "NZ": Decimal("0.08700"),
    "SG": Decimal("0.40800"),
    "ZA": Decimal("0.92800"),
    "MX": Decimal("0.42300"),
    "GLOBAL": Decimal("0.43200"),
}

# ==============================================================================
# eGRID SUBREGION EMISSION FACTORS (kgCO2e per kWh) - EPA eGRID 2024
# ==============================================================================

EGRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "AKGD": Decimal("0.41100"),
    "AKMS": Decimal("0.20300"),
    "AZNM": Decimal("0.36300"),
    "CAMX": Decimal("0.21500"),
    "ERCT": Decimal("0.36200"),
    "FRCC": Decimal("0.36200"),
    "HIMS": Decimal("0.55400"),
    "HIOA": Decimal("0.62100"),
    "MROE": Decimal("0.51700"),
    "MROW": Decimal("0.40500"),
    "NEWE": Decimal("0.18400"),
    "NWPP": Decimal("0.25100"),
    "NYCW": Decimal("0.22100"),
    "NYLI": Decimal("0.38300"),
    "NYUP": Decimal("0.09200"),
    "PRMS": Decimal("0.68200"),
    "RFCE": Decimal("0.30300"),
    "RFCM": Decimal("0.48600"),
    "RFCW": Decimal("0.44500"),
    "RMPA": Decimal("0.47700"),
    "SPNO": Decimal("0.42300"),
    "SPSO": Decimal("0.39100"),
    "SRMV": Decimal("0.33800"),
    "SRMW": Decimal("0.62100"),
    "SRSO": Decimal("0.38000"),
    "SRTV": Decimal("0.37900"),
    "SRVC": Decimal("0.27500"),
}

# ==============================================================================
# SPEND-BASED EEIO FACTORS (kgCO2e per USD) - EPA USEEIO v2.0
# ==============================================================================

EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "441100": {
        "label": "Automobile Dealers",
        "factor": Decimal("0.19830"),
        "sector": "vehicle_leasing",
    },
    "441200": {
        "label": "Other Motor Vehicle Dealers",
        "factor": Decimal("0.18540"),
        "sector": "vehicle_leasing",
    },
    "532100": {
        "label": "Automotive Equipment Rental and Leasing",
        "factor": Decimal("0.32450"),
        "sector": "vehicle_leasing",
    },
    "532400": {
        "label": "Commercial/Industrial Equipment Rental and Leasing",
        "factor": Decimal("0.28760"),
        "sector": "equipment_leasing",
    },
    "811100": {
        "label": "Automotive Repair and Maintenance",
        "factor": Decimal("0.15620"),
        "sector": "vehicle_maintenance",
    },
    "336100": {
        "label": "Motor Vehicle Manufacturing",
        "factor": Decimal("0.35280"),
        "sector": "vehicle_manufacturing",
    },
    "324100": {
        "label": "Petroleum Refineries",
        "factor": Decimal("0.82340"),
        "sector": "fuel_production",
    },
    "447100": {
        "label": "Gasoline Stations",
        "factor": Decimal("0.41553"),
        "sector": "fuel_retail",
    },
}

# ==============================================================================
# CURRENCY EXCHANGE RATES (to USD) - Mid-market rates 2024
# ==============================================================================

CURRENCY_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1.00000"),
    "EUR": Decimal("1.08500"),
    "GBP": Decimal("1.26700"),
    "JPY": Decimal("0.00667"),
    "CNY": Decimal("0.13830"),
    "CAD": Decimal("0.74100"),
    "AUD": Decimal("0.65200"),
    "INR": Decimal("0.01198"),
    "KRW": Decimal("0.00075"),
    "BRL": Decimal("0.20100"),
    "MXN": Decimal("0.05880"),
    "CHF": Decimal("1.12800"),
    "SEK": Decimal("0.09400"),
    "NOK": Decimal("0.09300"),
    "DKK": Decimal("0.14500"),
    "SGD": Decimal("0.74500"),
    "NZD": Decimal("0.60800"),
    "ZAR": Decimal("0.05300"),
}

# ==============================================================================
# CPI DEFLATORS (base year 2021 = 1.000) - US BLS CPI-U
# ==============================================================================

CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.84710"),
    2016: Decimal("0.85790"),
    2017: Decimal("0.87610"),
    2018: Decimal("0.89760"),
    2019: Decimal("0.91380"),
    2020: Decimal("0.92530"),
    2021: Decimal("1.00000"),
    2022: Decimal("1.07990"),
    2023: Decimal("1.11450"),
    2024: Decimal("1.14220"),
    2025: Decimal("1.16960"),
}

# ==============================================================================
# DEFAULT ANNUAL KM BY VEHICLE TYPE
# Source: DEFRA 2024 / Fleet industry benchmarks
# ==============================================================================

DEFAULT_ANNUAL_KM: Dict[str, Decimal] = {
    "small_car": Decimal("15000"),
    "medium_car": Decimal("20000"),
    "large_car": Decimal("25000"),
    "suv": Decimal("22000"),
    "light_van": Decimal("25000"),
    "heavy_van": Decimal("30000"),
    "light_truck": Decimal("40000"),
    "heavy_truck": Decimal("60000"),
}

# ==============================================================================
# DQI SCORE BY METHOD
# GHG Protocol data quality indicators (1=best, 5=worst)
# ==============================================================================

DQI_SCORES: Dict[str, Decimal] = {
    "distance_based": Decimal("2.0"),
    "fuel_based": Decimal("1.5"),
    "electric_vehicle": Decimal("2.0"),
    "fleet_aggregate": Decimal("2.0"),
    "spend_based": Decimal("4.0"),
    "average_data": Decimal("3.5"),
    "estimate": Decimal("4.0"),
}

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["VehicleFleetCalculatorEngine"] = None
_instance_lock: threading.Lock = threading.Lock()


# ==============================================================================
# HELPER: Quantize a Decimal to 8 decimal places
# ==============================================================================

def _q(value: Decimal) -> Decimal:
    """
    Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with exactly 8 decimal places.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _safe_decimal(value: Any) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert (str, int, float, or Decimal).

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If value cannot be converted to Decimal.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert '{value}' (type={type(value).__name__}) to Decimal"
        ) from exc


def _calculate_provenance_hash(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    Args:
        inputs: Input parameters as a dict.
        outputs: Output values as a dict.

    Returns:
        SHA-256 hex digest string.
    """
    payload = json.dumps(
        {"inputs": inputs, "outputs": outputs, "engine": ENGINE_ID, "version": ENGINE_VERSION},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSORS
# ==============================================================================

def get_vehicle_fleet_calculator() -> "VehicleFleetCalculatorEngine":
    """
    Get the singleton VehicleFleetCalculatorEngine instance.

    Returns:
        VehicleFleetCalculatorEngine singleton.
    """
    return VehicleFleetCalculatorEngine.get_instance()


def reset_vehicle_fleet_calculator() -> None:
    """
    Reset the singleton VehicleFleetCalculatorEngine instance (testing only).
    """
    VehicleFleetCalculatorEngine.reset_instance()


# ==============================================================================
# VehicleFleetCalculatorEngine
# ==============================================================================


class VehicleFleetCalculatorEngine:
    """
    Engine 3: Vehicle fleet emissions calculator for upstream leased assets.

    Implements deterministic emissions calculations for leased vehicles including
    cars (by size and fuel type), SUVs, vans, and trucks using DEFRA 2024
    emission factors aligned with GHG Protocol Scope 3 Category 8 methodology.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with DEFRA/EPA-sourced parameters. No LLM
    calls are made anywhere in the calculation pipeline.

    Thread Safety:
        This engine is fully thread-safe. A reentrant lock protects shared
        state during calculations. The singleton instance is created lazily
        with double-checked locking.

    Attributes:
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.
        _initialized_at: Timestamp when the engine was initialized.

    Example:
        >>> engine = get_vehicle_fleet_calculator()
        >>> result = engine.calculate_distance_based(
        ...     vehicle_type="medium_car",
        ...     fuel_type="diesel",
        ...     annual_km=Decimal("20000"),
        ...     count=5,
        ... )
        >>> assert result["co2e_kg"] > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance() -> "VehicleFleetCalculatorEngine":
        """
        Get or create the singleton VehicleFleetCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Returns:
            Singleton VehicleFleetCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = VehicleFleetCalculatorEngine()
        return _instance

    @staticmethod
    def reset_instance() -> None:
        """
        Reset the singleton instance (for testing only).

        This method is intended exclusively for unit tests that need
        a fresh engine instance. It should never be called in production.
        """
        global _instance
        with _instance_lock:
            _instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the VehicleFleetCalculatorEngine."""
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0
        self._initialized_at: str = datetime.now(timezone.utc).isoformat()

        logger.info(
            "VehicleFleetCalculatorEngine initialized: "
            "engine=%s, version=%s, agent=%s",
            ENGINE_ID,
            ENGINE_VERSION,
            AGENT_ID,
        )

    # ==================================================================
    # PROPERTY: calculation_count
    # ==================================================================

    @property
    def calculation_count(self) -> int:
        """Return the total number of calculations performed by this engine."""
        return self._calculation_count

    @property
    def engine_id(self) -> str:
        """Return the engine identifier."""
        return ENGINE_ID

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        return ENGINE_VERSION

    # ==================================================================
    # VALIDATION HELPERS
    # ==================================================================

    def _validate_vehicle_type(self, vehicle_type: str) -> str:
        """
        Validate and normalize vehicle type string.

        Args:
            vehicle_type: Vehicle type key to validate.

        Returns:
            Normalized vehicle type string (lowercase, stripped).

        Raises:
            ValueError: If vehicle_type is not recognized.
        """
        normalized = vehicle_type.lower().strip()
        if normalized not in VEHICLE_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown vehicle_type '{vehicle_type}'. "
                f"Available: {VALID_VEHICLE_TYPES}"
            )
        return normalized

    def _validate_fuel_type(self, vehicle_type: str, fuel_type: str) -> str:
        """
        Validate and normalize fuel type for a given vehicle type.

        Args:
            vehicle_type: Normalized vehicle type key.
            fuel_type: Fuel type key to validate.

        Returns:
            Normalized fuel type string (lowercase, stripped).

        Raises:
            ValueError: If fuel_type is not valid for the vehicle_type.
        """
        normalized = fuel_type.lower().strip()
        available_fuels = VEHICLE_EMISSION_FACTORS.get(vehicle_type, {})
        if normalized not in available_fuels:
            raise ValueError(
                f"Fuel type '{fuel_type}' not available for vehicle type "
                f"'{vehicle_type}'. Available fuels: {list(available_fuels.keys())}"
            )
        return normalized

    def _validate_positive_decimal(
        self, value: Any, field_name: str
    ) -> Decimal:
        """
        Validate that a value is a positive Decimal.

        Args:
            value: Value to validate and convert.
            field_name: Name of the field for error messages.

        Returns:
            Validated Decimal value.

        Raises:
            ValueError: If value is not positive.
        """
        dec_val = _safe_decimal(value)
        if dec_val <= _ZERO:
            raise ValueError(
                f"{field_name} must be positive, got {dec_val}"
            )
        return dec_val

    def _validate_non_negative_decimal(
        self, value: Any, field_name: str
    ) -> Decimal:
        """
        Validate that a value is a non-negative Decimal.

        Args:
            value: Value to validate and convert.
            field_name: Name of the field for error messages.

        Returns:
            Validated Decimal value.

        Raises:
            ValueError: If value is negative.
        """
        dec_val = _safe_decimal(value)
        if dec_val < _ZERO:
            raise ValueError(
                f"{field_name} must be non-negative, got {dec_val}"
            )
        return dec_val

    def _validate_count(self, count: int) -> int:
        """
        Validate vehicle count is a positive integer.

        Args:
            count: Number of vehicles.

        Returns:
            Validated count.

        Raises:
            ValueError: If count is not a positive integer.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError(
                f"Vehicle count must be a positive integer, got {count}"
            )
        return count

    def _validate_vehicle_age(self, vehicle_age: str) -> str:
        """
        Validate and normalize vehicle age category.

        Args:
            vehicle_age: Vehicle age category key.

        Returns:
            Normalized vehicle age string.

        Raises:
            ValueError: If vehicle_age is not recognized.
        """
        normalized = vehicle_age.lower().strip()
        if normalized not in VEHICLE_AGE_FACTORS:
            raise ValueError(
                f"Unknown vehicle_age '{vehicle_age}'. "
                f"Available: {list(VEHICLE_AGE_FACTORS.keys())}"
            )
        return normalized

    def _validate_country_code(self, country_code: str) -> str:
        """
        Validate and normalize country code.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Normalized country code string (uppercase, stripped).

        Raises:
            ValueError: If country_code is not recognized.
        """
        normalized = country_code.upper().strip()
        if normalized not in GRID_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown country_code '{country_code}'. "
                f"Available: {list(GRID_EMISSION_FACTORS.keys())}"
            )
        return normalized

    def _get_grid_ef(
        self, country_code: str, egrid_subregion: Optional[str] = None
    ) -> Decimal:
        """
        Get grid emission factor for a country or eGRID subregion.

        eGRID subregion takes precedence for US-based calculations when
        provided. Otherwise falls back to national grid factor.

        Args:
            country_code: ISO country code (uppercase).
            egrid_subregion: Optional EPA eGRID subregion code.

        Returns:
            Grid emission factor in kgCO2e/kWh.

        Raises:
            ValueError: If egrid_subregion is provided but not recognized.
        """
        if egrid_subregion is not None:
            subregion = egrid_subregion.upper().strip()
            if subregion not in EGRID_EMISSION_FACTORS:
                raise ValueError(
                    f"Unknown eGRID subregion '{egrid_subregion}'. "
                    f"Available: {list(EGRID_EMISSION_FACTORS.keys())}"
                )
            return EGRID_EMISSION_FACTORS[subregion]
        return GRID_EMISSION_FACTORS[country_code]

    # ==================================================================
    # 1. calculate_distance_based
    # ==================================================================

    def calculate_distance_based(
        self,
        vehicle_type: str,
        fuel_type: str,
        annual_km: Union[Decimal, int, float, str],
        count: int = 1,
        vehicle_age: str = "mid_4_7yr",
        country_code: str = "US",
        include_wtt: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate vehicle fleet emissions using distance-based method.

        Computes annual fleet emissions by multiplying annual distance per
        vehicle by per-km emission factors, adjusting for vehicle age and
        fleet size (count), with optional WTT upstream emissions.

        Formula:
            age_factor  = VEHICLE_AGE_FACTORS[vehicle_age]
            ef_per_km   = VEHICLE_EMISSION_FACTORS[vehicle_type][fuel_type]
            wtt_per_km  = ef_per_km x WTT_RATIOS[fuel_type]

            ttw_co2e = annual_km x count x ef_per_km x age_factor
            wtt_co2e = annual_km x count x wtt_per_km x age_factor  (if include_wtt)
            co2e_kg  = ttw_co2e + wtt_co2e

        Args:
            vehicle_type: Vehicle type key (e.g., "medium_car", "heavy_truck").
            fuel_type: Fuel type key (e.g., "petrol", "diesel", "hybrid", "bev").
            annual_km: Annual kilometres driven per vehicle.
            count: Number of vehicles in this fleet segment (default 1).
            vehicle_age: Vehicle age category (default "mid_4_7yr").
            country_code: ISO country code for grid EF context (default "US").
            include_wtt: Include well-to-tank upstream emissions (default True).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg, method,
            vehicle_type, fuel_type, annual_km, count, vehicle_age,
            ef_per_km, wtt_per_km, age_factor, ef_source, dqi_score,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_distance_based(
            ...     vehicle_type="medium_car",
            ...     fuel_type="diesel",
            ...     annual_km=Decimal("20000"),
            ...     count=5,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            v_type = self._validate_vehicle_type(vehicle_type)
            f_type = self._validate_fuel_type(v_type, fuel_type)
            km = self._validate_positive_decimal(annual_km, "annual_km")
            cnt = self._validate_count(count)
            v_age = self._validate_vehicle_age(vehicle_age)
            c_code = self._validate_country_code(country_code)

            # Step 2: Resolve emission factors (ZERO HALLUCINATION)
            ef_per_km = VEHICLE_EMISSION_FACTORS[v_type][f_type]
            wtt_ratio = WTT_RATIOS.get(f_type, _ZERO)
            wtt_per_km = _q(ef_per_km * wtt_ratio)
            age_factor = VEHICLE_AGE_FACTORS[v_age]

            # Step 3: Calculate emissions (Decimal only)
            count_dec = _safe_decimal(cnt)

            ttw_co2e = _q(km * count_dec * ef_per_km * age_factor)
            wtt_co2e = _ZERO
            if include_wtt:
                wtt_co2e = _q(km * count_dec * wtt_per_km * age_factor)
            co2e_kg = _q(ttw_co2e + wtt_co2e)

            # Step 4: Build provenance hash
            input_data = {
                "vehicle_type": v_type,
                "fuel_type": f_type,
                "annual_km": str(km),
                "count": cnt,
                "vehicle_age": v_age,
                "country_code": c_code,
                "include_wtt": include_wtt,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "ttw_co2e_kg": str(ttw_co2e),
                "wtt_co2e_kg": str(wtt_co2e),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "method": "distance_based",
                "vehicle_type": v_type,
                "fuel_type": f_type,
                "annual_km": km,
                "count": cnt,
                "vehicle_age": v_age,
                "ef_per_km": ef_per_km,
                "wtt_per_km": wtt_per_km,
                "age_factor": age_factor,
                "ef_source": "DEFRA_2024",
                "dqi_score": DQI_SCORES["distance_based"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Distance-based calculation complete: type=%s, fuel=%s, "
                "km=%s, count=%d, co2e=%s kg, ttw=%s kg, wtt=%s kg, "
                "age=%s, duration=%.4fs",
                v_type, f_type, km, cnt, co2e_kg,
                ttw_co2e, wtt_co2e, v_age, duration,
            )

            return result

    # ==================================================================
    # 2. calculate_fuel_based
    # ==================================================================

    def calculate_fuel_based(
        self,
        fuel_type: str,
        fuel_litres: Union[Decimal, int, float, str],
        count: int = 1,
        include_wtt: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate vehicle fleet emissions using fuel-based method.

        Uses fuel consumption volume and per-litre emission factors for
        direct measurement scenarios where fuel records are available.

        Formula:
            total_litres = fuel_litres x count
            ttw_co2e = total_litres x ef_per_litre
            wtt_co2e = total_litres x wtt_per_litre  (if include_wtt)
            co2e_kg  = ttw_co2e + wtt_co2e

        Args:
            fuel_type: Fuel type key (petrol, diesel, lpg, cng, e85).
            fuel_litres: Total fuel consumed in litres per vehicle.
            count: Number of vehicles consuming this fuel (default 1).
            include_wtt: Include well-to-tank upstream emissions (default True).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg, method,
            fuel_type, fuel_litres, total_fuel_litres, count,
            ef_per_litre, wtt_per_litre, ef_source, dqi_score,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If fuel_litres <= 0 or fuel_type is unknown.

        Example:
            >>> result = engine.calculate_fuel_based(
            ...     fuel_type="diesel",
            ...     fuel_litres=Decimal("2500"),
            ...     count=3,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            f_type = fuel_type.lower().strip()
            if f_type not in FUEL_EMISSION_FACTORS:
                raise ValueError(
                    f"Unknown fuel_type '{fuel_type}'. "
                    f"Available: {list(FUEL_EMISSION_FACTORS.keys())}"
                )
            litres = self._validate_positive_decimal(fuel_litres, "fuel_litres")
            cnt = self._validate_count(count)

            # Step 2: Resolve emission factors (ZERO HALLUCINATION)
            fuel_ef = FUEL_EMISSION_FACTORS[f_type]
            ef_per_litre = fuel_ef["ef_per_litre"]
            wtt_per_litre = fuel_ef["wtt_per_litre"]

            # Step 3: Calculate emissions (Decimal only)
            count_dec = _safe_decimal(cnt)
            total_litres = _q(litres * count_dec)

            ttw_co2e = _q(total_litres * ef_per_litre)
            wtt_co2e = _ZERO
            if include_wtt:
                wtt_co2e = _q(total_litres * wtt_per_litre)
            co2e_kg = _q(ttw_co2e + wtt_co2e)

            # Step 4: Build provenance hash
            input_data = {
                "fuel_type": f_type,
                "fuel_litres": str(litres),
                "count": cnt,
                "include_wtt": include_wtt,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "ttw_co2e_kg": str(ttw_co2e),
                "wtt_co2e_kg": str(wtt_co2e),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "method": "fuel_based",
                "fuel_type": f_type,
                "fuel_litres": litres,
                "total_fuel_litres": total_litres,
                "count": cnt,
                "ef_per_litre": ef_per_litre,
                "wtt_per_litre": wtt_per_litre,
                "ef_source": "DEFRA_2024",
                "dqi_score": DQI_SCORES["fuel_based"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Fuel-based calculation complete: fuel=%s, litres=%s, "
                "count=%d, co2e=%s kg, ttw=%s kg, wtt=%s kg, duration=%.4fs",
                f_type, litres, cnt, co2e_kg,
                ttw_co2e, wtt_co2e, duration,
            )

            return result

    # ==================================================================
    # 3. calculate_electric_vehicle
    # ==================================================================

    def calculate_electric_vehicle(
        self,
        vehicle_type: str,
        annual_km: Union[Decimal, int, float, str],
        count: int = 1,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate electric vehicle fleet emissions using grid-based method.

        Computes annual EV fleet emissions from electricity consumption per
        km multiplied by grid emission factor for the specified location.

        Formula:
            consumption_per_km = EV_CONSUMPTION[vehicle_type]
            grid_ef            = grid_factor(country_code or egrid_subregion)

            energy_kwh = annual_km x consumption_per_km x count
            co2e_kg    = energy_kwh x grid_ef

        Args:
            vehicle_type: Vehicle type key for EV consumption lookup
                          (e.g., "medium_car", "suv", "light_van").
            annual_km: Annual kilometres driven per vehicle.
            count: Number of EVs in this fleet segment (default 1).
            country_code: ISO country code for grid EF (default "US").
            egrid_subregion: Optional EPA eGRID subregion code (overrides
                             country_code for US-based calculations).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            energy_kwh, consumption_per_km, grid_ef, method,
            vehicle_type, fuel_type, annual_km, count,
            ef_source, dqi_score, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If vehicle_type has no EV consumption data, or if
                        annual_km <= 0, or if country/subregion is unknown.

        Example:
            >>> result = engine.calculate_electric_vehicle(
            ...     vehicle_type="medium_car",
            ...     annual_km=Decimal("20000"),
            ...     count=3,
            ...     country_code="US",
            ...     egrid_subregion="CAMX",
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            v_type = vehicle_type.lower().strip()
            if v_type not in EV_CONSUMPTION:
                raise ValueError(
                    f"No EV consumption data for vehicle_type '{vehicle_type}'. "
                    f"Available: {list(EV_CONSUMPTION.keys())}"
                )
            km = self._validate_positive_decimal(annual_km, "annual_km")
            cnt = self._validate_count(count)
            c_code = self._validate_country_code(country_code)

            # Step 2: Resolve factors (ZERO HALLUCINATION)
            consumption_per_km = EV_CONSUMPTION[v_type]
            grid_ef = self._get_grid_ef(c_code, egrid_subregion)

            # Step 3: Calculate emissions (Decimal only)
            count_dec = _safe_decimal(cnt)
            energy_kwh = _q(km * consumption_per_km * count_dec)
            co2e_kg = _q(energy_kwh * grid_ef)

            # EVs have zero TTW tailpipe; all is upstream grid
            ttw_co2e = _ZERO
            wtt_co2e = co2e_kg

            # Step 4: Build provenance hash
            input_data = {
                "vehicle_type": v_type,
                "annual_km": str(km),
                "count": cnt,
                "country_code": c_code,
                "egrid_subregion": egrid_subregion,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "energy_kwh": str(energy_kwh),
                "grid_ef": str(grid_ef),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Determine EF source label
            ef_source = "EPA_eGRID_2024" if egrid_subregion else "IEA_2024"

            # Step 6: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "energy_kwh": energy_kwh,
                "consumption_per_km": consumption_per_km,
                "grid_ef": grid_ef,
                "method": "electric_vehicle",
                "vehicle_type": v_type,
                "fuel_type": "bev",
                "annual_km": km,
                "count": cnt,
                "ef_source": ef_source,
                "dqi_score": DQI_SCORES["electric_vehicle"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 7: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "EV calculation complete: type=%s, km=%s, count=%d, "
                "kWh=%s, grid_ef=%s, co2e=%s kg, duration=%.4fs",
                v_type, km, cnt, energy_kwh, grid_ef, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # 4. calculate_fleet
    # ==================================================================

    def calculate_fleet(
        self,
        vehicles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate aggregate fleet emissions from multiple vehicle segments.

        Iterates over a list of vehicle specifications, calculates emissions
        for each segment using the appropriate method (distance, fuel, or EV),
        and aggregates totals for the entire fleet.

        Each vehicle dict should contain:
            - vehicle_type (str): Vehicle type key.
            - fuel_type (str): Fuel type key (or "bev" for electric).
            - annual_km (Decimal/int/float/str): Annual km per vehicle.
            - count (int, optional): Number of vehicles (default 1).
            - vehicle_age (str, optional): Age category (default "mid_4_7yr").
            - country_code (str, optional): ISO country code (default "US").
            - egrid_subregion (str, optional): eGRID subregion for EVs.
            - include_wtt (bool, optional): Include WTT (default True).
            - method (str, optional): "distance_based" | "fuel_based" | "ev".

        If method is "fuel_based", requires:
            - fuel_litres (Decimal/int/float/str): Fuel consumed per vehicle.

        Args:
            vehicles: List of vehicle specification dicts.

        Returns:
            Dict with keys: fleet_co2e_kg, fleet_ttw_co2e_kg,
            fleet_wtt_co2e_kg, total_vehicles, vehicle_results (list
            of per-segment results), method, provenance_hash,
            calculation_timestamp.

        Raises:
            ValueError: If vehicles list is empty or exceeds _MAX_BATCH_SIZE.

        Example:
            >>> result = engine.calculate_fleet([
            ...     {"vehicle_type": "medium_car", "fuel_type": "diesel",
            ...      "annual_km": 20000, "count": 10},
            ...     {"vehicle_type": "suv", "fuel_type": "bev",
            ...      "annual_km": 18000, "count": 5, "method": "ev"},
            ... ])
            >>> assert result["fleet_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        if not vehicles:
            raise ValueError("Vehicles list must not be empty")
        if len(vehicles) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Fleet size {len(vehicles)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        vehicle_results: List[Dict[str, Any]] = []
        fleet_co2e = _ZERO
        fleet_ttw = _ZERO
        fleet_wtt = _ZERO
        total_vehicles = 0
        errors: List[Dict[str, Any]] = []

        for idx, spec in enumerate(vehicles):
            try:
                seg_result = self._calculate_fleet_segment(idx, spec)
                vehicle_results.append(seg_result)
                fleet_co2e = _q(fleet_co2e + seg_result["co2e_kg"])
                fleet_ttw = _q(fleet_ttw + seg_result["ttw_co2e_kg"])
                fleet_wtt = _q(fleet_wtt + seg_result["wtt_co2e_kg"])
                total_vehicles += seg_result.get("count", 1)
            except Exception as exc:
                logger.warning(
                    "Fleet segment %d failed: %s", idx, str(exc)
                )
                errors.append({
                    "index": idx,
                    "error": str(exc),
                    "vehicle_spec": spec,
                })

        # Build provenance hash for fleet
        input_data = {
            "vehicle_count": len(vehicles),
            "total_vehicles": total_vehicles,
        }
        output_data = {
            "fleet_co2e_kg": str(fleet_co2e),
            "fleet_ttw_co2e_kg": str(fleet_ttw),
            "fleet_wtt_co2e_kg": str(fleet_wtt),
        }
        provenance_hash = _calculate_provenance_hash(input_data, output_data)

        timestamp = datetime.now(timezone.utc).isoformat()
        result: Dict[str, Any] = {
            "fleet_co2e_kg": fleet_co2e,
            "fleet_ttw_co2e_kg": fleet_ttw,
            "fleet_wtt_co2e_kg": fleet_wtt,
            "total_vehicles": total_vehicles,
            "segments_processed": len(vehicle_results),
            "segments_failed": len(errors),
            "vehicle_results": vehicle_results,
            "errors": errors,
            "method": "fleet_aggregate",
            "dqi_score": DQI_SCORES["fleet_aggregate"],
            "provenance_hash": provenance_hash,
            "calculation_timestamp": timestamp,
        }

        duration = time.monotonic() - start_time
        logger.info(
            "Fleet calculation complete: segments=%d, vehicles=%d, "
            "co2e=%s kg, errors=%d, duration=%.4fs",
            len(vehicle_results), total_vehicles,
            fleet_co2e, len(errors), duration,
        )

        return result

    def _calculate_fleet_segment(
        self, idx: int, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single fleet segment.

        Routes to the appropriate calculation method based on the 'method'
        key in the specification dict.

        Args:
            idx: Segment index for logging.
            spec: Vehicle specification dict.

        Returns:
            Calculation result dict from the appropriate method.

        Raises:
            ValueError: If required keys are missing or method is unknown.
        """
        method = spec.get("method", "distance_based").lower().strip()

        if method == "fuel_based":
            return self.calculate_fuel_based(
                fuel_type=spec["fuel_type"],
                fuel_litres=spec["fuel_litres"],
                count=spec.get("count", 1),
                include_wtt=spec.get("include_wtt", True),
            )
        elif method in ("ev", "electric_vehicle", "electric"):
            return self.calculate_electric_vehicle(
                vehicle_type=spec["vehicle_type"],
                annual_km=spec["annual_km"],
                count=spec.get("count", 1),
                country_code=spec.get("country_code", "US"),
                egrid_subregion=spec.get("egrid_subregion"),
            )
        elif method == "distance_based":
            return self.calculate_distance_based(
                vehicle_type=spec["vehicle_type"],
                fuel_type=spec["fuel_type"],
                annual_km=spec["annual_km"],
                count=spec.get("count", 1),
                vehicle_age=spec.get("vehicle_age", "mid_4_7yr"),
                country_code=spec.get("country_code", "US"),
                include_wtt=spec.get("include_wtt", True),
            )
        else:
            raise ValueError(
                f"Unknown fleet segment method '{method}' at index {idx}. "
                f"Supported: distance_based, fuel_based, ev"
            )

    # ==================================================================
    # 5. calculate_spend_based
    # ==================================================================

    def calculate_spend_based(
        self,
        naics_code: str,
        amount: Union[Decimal, int, float, str],
        currency: str = "USD",
        reporting_year: int = 2024,
    ) -> Dict[str, Any]:
        """
        Calculate vehicle fleet emissions using spend-based EEIO method.

        Fallback method when activity data is unavailable. Converts currency
        to USD, applies CPI deflation to base year 2021, then multiplies by
        the appropriate EEIO factor.

        Formula:
            usd_amount      = amount x currency_rate
            deflated_amount = usd_amount / cpi_deflator
            co2e_kg         = deflated_amount x eeio_factor

        Args:
            naics_code: NAICS industry code for EEIO factor lookup.
            amount: Expenditure amount in the specified currency.
            currency: Currency code (default "USD").
            reporting_year: Reporting year for CPI deflation (default 2024).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg, method,
            naics_code, naics_label, original_amount, currency, usd_amount,
            deflated_amount, eeio_factor, reporting_year, ef_source,
            dqi_score, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If NAICS code unknown, amount <= 0, currency unknown,
                        or reporting_year not in CPI table.

        Example:
            >>> result = engine.calculate_spend_based(
            ...     naics_code="532100",
            ...     amount=Decimal("50000"),
            ...     currency="USD",
            ...     reporting_year=2024,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            if naics_code not in EEIO_FACTORS:
                raise ValueError(
                    f"Unknown NAICS code '{naics_code}'. "
                    f"Available: {list(EEIO_FACTORS.keys())}"
                )
            amt = self._validate_positive_decimal(amount, "amount")
            cur = currency.upper().strip()
            if cur not in CURRENCY_RATES:
                raise ValueError(
                    f"Unknown currency '{currency}'. "
                    f"Available: {list(CURRENCY_RATES.keys())}"
                )
            if reporting_year not in CPI_DEFLATORS:
                raise ValueError(
                    f"No CPI deflator for year {reporting_year}. "
                    f"Available: {list(CPI_DEFLATORS.keys())}"
                )

            # Step 2: Convert to USD (ZERO HALLUCINATION)
            currency_rate = CURRENCY_RATES[cur]
            usd_amount = _q(amt * currency_rate)

            # Step 3: Apply CPI deflation to base year 2021
            cpi_deflator = CPI_DEFLATORS[reporting_year]
            deflated_amount = _q(usd_amount / cpi_deflator)

            # Step 4: Apply EEIO factor
            eeio_info = EEIO_FACTORS[naics_code]
            eeio_factor = eeio_info["factor"]
            co2e_kg = _q(deflated_amount * eeio_factor)

            # Spend-based has no TTW/WTT split
            ttw_co2e = co2e_kg
            wtt_co2e = _ZERO

            # Step 5: Build provenance hash
            input_data = {
                "naics_code": naics_code,
                "amount": str(amt),
                "currency": cur,
                "reporting_year": reporting_year,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "usd_amount": str(usd_amount),
                "deflated_amount": str(deflated_amount),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 6: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "method": "spend_based",
                "naics_code": naics_code,
                "naics_label": eeio_info["label"],
                "original_amount": amt,
                "currency": cur,
                "usd_amount": usd_amount,
                "deflated_amount": deflated_amount,
                "eeio_factor": eeio_factor,
                "reporting_year": reporting_year,
                "ef_source": "EPA_USEEIO_v2.0",
                "dqi_score": DQI_SCORES["spend_based"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 7: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Spend-based calculation complete: naics=%s, amount=%s %s, "
                "usd=%s, deflated=%s, co2e=%s kg, duration=%.4fs",
                naics_code, amt, cur, usd_amount,
                deflated_amount, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # 6. calculate_batch
    # ==================================================================

    def calculate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple vehicle calculations in a single batch.

        Each item dict must contain a 'method' key indicating the calculation
        type (distance_based, fuel_based, electric_vehicle, spend_based,
        estimate) plus the method-specific parameters.

        Failed calculations do not halt the batch; they are recorded in
        the errors list.

        Args:
            items: List of dicts, each with 'method' and method-specific params.

        Returns:
            Dict with keys: total_co2e_kg, total_ttw_co2e_kg,
            total_wtt_co2e_kg, items_processed, items_failed,
            results (list), errors (list), provenance_hash,
            calculation_timestamp.

        Raises:
            ValueError: If items list exceeds _MAX_BATCH_SIZE.

        Example:
            >>> batch_result = engine.calculate_batch([
            ...     {"method": "distance_based", "vehicle_type": "medium_car",
            ...      "fuel_type": "diesel", "annual_km": 20000, "count": 5},
            ...     {"method": "fuel_based", "fuel_type": "diesel",
            ...      "fuel_litres": 3000, "count": 2},
            ... ])
            >>> assert batch_result["total_co2e_kg"] > Decimal("0")
        """
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = _ZERO
        total_ttw = _ZERO
        total_wtt = _ZERO

        for idx, item in enumerate(items):
            try:
                calc_result = self._dispatch_batch_item(idx, item)
                results.append({
                    "index": idx,
                    "status": "success",
                    "result": calc_result,
                })
                total_co2e = _q(total_co2e + calc_result.get("co2e_kg", _ZERO))
                total_ttw = _q(total_ttw + calc_result.get("ttw_co2e_kg", _ZERO))
                total_wtt = _q(total_wtt + calc_result.get("wtt_co2e_kg", _ZERO))
            except Exception as exc:
                logger.warning(
                    "Batch item %d failed: %s", idx, str(exc)
                )
                errors.append({
                    "index": idx,
                    "status": "error",
                    "error": str(exc),
                    "item": item,
                })

        # Build provenance hash
        input_data = {
            "batch_size": len(items),
            "items_processed": len(results),
            "items_failed": len(errors),
        }
        output_data = {
            "total_co2e_kg": str(total_co2e),
            "total_ttw_co2e_kg": str(total_ttw),
            "total_wtt_co2e_kg": str(total_wtt),
        }
        provenance_hash = _calculate_provenance_hash(input_data, output_data)

        timestamp = datetime.now(timezone.utc).isoformat()
        batch_result: Dict[str, Any] = {
            "total_co2e_kg": total_co2e,
            "total_ttw_co2e_kg": total_ttw,
            "total_wtt_co2e_kg": total_wtt,
            "items_processed": len(results),
            "items_failed": len(errors),
            "results": results,
            "errors": errors,
            "provenance_hash": provenance_hash,
            "calculation_timestamp": timestamp,
        }

        duration = time.monotonic() - start_time
        logger.info(
            "Batch calculation complete: total=%d, success=%d, "
            "errors=%d, co2e=%s kg, duration=%.4fs",
            len(items), len(results), len(errors),
            total_co2e, duration,
        )

        return batch_result

    def _dispatch_batch_item(
        self, idx: int, item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dispatch a single batch item to the appropriate calculation method.

        Args:
            idx: Item index for logging.
            item: Dict with 'method' and method-specific parameters.

        Returns:
            Calculation result dict.

        Raises:
            ValueError: If method is unknown or required parameters missing.
        """
        method = item.get("method", "distance_based").lower().strip()

        if method == "distance_based":
            return self.calculate_distance_based(
                vehicle_type=item["vehicle_type"],
                fuel_type=item["fuel_type"],
                annual_km=item["annual_km"],
                count=item.get("count", 1),
                vehicle_age=item.get("vehicle_age", "mid_4_7yr"),
                country_code=item.get("country_code", "US"),
                include_wtt=item.get("include_wtt", True),
            )
        elif method == "fuel_based":
            return self.calculate_fuel_based(
                fuel_type=item["fuel_type"],
                fuel_litres=item["fuel_litres"],
                count=item.get("count", 1),
                include_wtt=item.get("include_wtt", True),
            )
        elif method in ("electric_vehicle", "ev", "electric"):
            return self.calculate_electric_vehicle(
                vehicle_type=item["vehicle_type"],
                annual_km=item["annual_km"],
                count=item.get("count", 1),
                country_code=item.get("country_code", "US"),
                egrid_subregion=item.get("egrid_subregion"),
            )
        elif method == "spend_based":
            return self.calculate_spend_based(
                naics_code=item["naics_code"],
                amount=item["amount"],
                currency=item.get("currency", "USD"),
                reporting_year=item.get("reporting_year", 2024),
            )
        elif method == "estimate":
            return self.estimate_annual_emissions(
                vehicle_type=item["vehicle_type"],
                fuel_type=item.get("fuel_type", "diesel"),
                annual_km=item.get("annual_km", 15000),
            )
        else:
            raise ValueError(
                f"Unknown batch method '{method}' at index {idx}. "
                f"Supported: distance_based, fuel_based, electric_vehicle, "
                f"spend_based, estimate"
            )

    # ==================================================================
    # 7. estimate_annual_emissions
    # ==================================================================

    def estimate_annual_emissions(
        self,
        vehicle_type: str,
        fuel_type: str = "diesel",
        annual_km: Union[Decimal, int, float, str, None] = None,
    ) -> Dict[str, Any]:
        """
        Quick annual emission estimation using default parameters.

        Provides a rapid estimate for screening-level assessments when
        detailed activity data is not yet available. Uses default annual
        mileage from fleet benchmarks if not provided.

        Args:
            vehicle_type: Vehicle type key (e.g., "medium_car", "heavy_truck").
            fuel_type: Fuel type key (default "diesel").
            annual_km: Optional annual km override. If None, uses default
                       from DEFAULT_ANNUAL_KM for the vehicle type.

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg, method,
            vehicle_type, fuel_type, annual_km, count, ef_source,
            dqi_score, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If vehicle_type or fuel_type is invalid.

        Example:
            >>> result = engine.estimate_annual_emissions(
            ...     vehicle_type="heavy_truck",
            ...     fuel_type="diesel",
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate vehicle type
            v_type = self._validate_vehicle_type(vehicle_type)

            # Step 2: Determine annual km
            if annual_km is not None:
                km = _safe_decimal(annual_km)
                if km <= _ZERO:
                    raise ValueError(
                        f"annual_km must be positive, got {km}"
                    )
            else:
                km = DEFAULT_ANNUAL_KM.get(v_type, Decimal("15000"))

            # Step 3: Determine fuel type and check compatibility
            f_type = fuel_type.lower().strip()

            # For BEV, use the electric vehicle path
            if f_type == "bev":
                return self.calculate_electric_vehicle(
                    vehicle_type=v_type,
                    annual_km=km,
                    count=1,
                    country_code="US",
                )

            # Check fuel type is valid for vehicle
            available_fuels = VEHICLE_EMISSION_FACTORS.get(v_type, {})
            if f_type not in available_fuels:
                # Fall back to first available fuel
                if available_fuels:
                    f_type = list(available_fuels.keys())[0]
                    logger.warning(
                        "Fuel type '%s' not available for '%s', "
                        "falling back to '%s'",
                        fuel_type, v_type, f_type,
                    )
                else:
                    raise ValueError(
                        f"No emission factors available for vehicle type '{v_type}'"
                    )

            # Step 4: Calculate using distance-based method with defaults
            ef_per_km = VEHICLE_EMISSION_FACTORS[v_type][f_type]
            wtt_ratio = WTT_RATIOS.get(f_type, _ZERO)
            wtt_per_km = _q(ef_per_km * wtt_ratio)

            ttw_co2e = _q(km * ef_per_km)
            wtt_co2e = _q(km * wtt_per_km)
            co2e_kg = _q(ttw_co2e + wtt_co2e)

            # Step 5: Build provenance hash
            input_data = {
                "vehicle_type": v_type,
                "fuel_type": f_type,
                "annual_km": str(km),
                "method": "estimate",
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "ttw_co2e_kg": str(ttw_co2e),
                "wtt_co2e_kg": str(wtt_co2e),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 6: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "method": "estimate",
                "vehicle_type": v_type,
                "fuel_type": f_type,
                "annual_km": km,
                "count": 1,
                "ef_per_km": ef_per_km,
                "wtt_per_km": wtt_per_km,
                "ef_source": "DEFRA_2024",
                "dqi_score": DQI_SCORES["estimate"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 7: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Estimate complete: type=%s, fuel=%s, km=%s, "
                "co2e=%s kg, duration=%.4fs",
                v_type, f_type, km, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # UTILITY METHODS
    # ==================================================================

    @staticmethod
    def convert_miles_to_km(miles: Union[Decimal, int, float, str]) -> Decimal:
        """
        Convert miles to kilometres.

        Uses the standard conversion factor 1 mile = 1.60934 km.

        Args:
            miles: Distance in miles.

        Returns:
            Distance in kilometres, quantized to 8 decimal places.

        Raises:
            ValueError: If miles is negative.
        """
        dec_miles = _safe_decimal(miles)
        if dec_miles < _ZERO:
            raise ValueError(f"Miles must be non-negative, got {dec_miles}")
        return _q(dec_miles * _KM_PER_MILE)

    @staticmethod
    def convert_gallons_to_litres(
        gallons: Union[Decimal, int, float, str],
    ) -> Decimal:
        """
        Convert US gallons to litres.

        Uses the standard conversion factor 1 US gallon = 3.78541 litres.

        Args:
            gallons: Volume in US gallons.

        Returns:
            Volume in litres, quantized to 8 decimal places.

        Raises:
            ValueError: If gallons is negative.
        """
        dec_gallons = _safe_decimal(gallons)
        if dec_gallons < _ZERO:
            raise ValueError(f"Gallons must be non-negative, got {dec_gallons}")
        return _q(dec_gallons * _LITRES_PER_GALLON)

    # ==================================================================
    # EMISSION FACTOR ACCESSORS (read-only)
    # ==================================================================

    @staticmethod
    def get_vehicle_emission_factors() -> Dict[str, Dict[str, str]]:
        """
        Return all vehicle emission factors as a serializable dict.

        Returns:
            Dict mapping vehicle type to fuel type to EF string.
        """
        return {
            vtype: {ftype: str(ef) for ftype, ef in fuels.items()}
            for vtype, fuels in VEHICLE_EMISSION_FACTORS.items()
        }

    @staticmethod
    def get_fuel_emission_factors() -> Dict[str, Dict[str, str]]:
        """
        Return all fuel emission factors as a serializable dict.

        Returns:
            Dict mapping fuel type to ef_per_litre and wtt_per_litre strings.
        """
        return {
            ftype: {k: str(v) for k, v in ef.items()}
            for ftype, ef in FUEL_EMISSION_FACTORS.items()
        }

    @staticmethod
    def get_ev_consumption_factors() -> Dict[str, str]:
        """
        Return all EV consumption factors as a serializable dict.

        Returns:
            Dict mapping vehicle type to kWh/km string.
        """
        return {vtype: str(kwh) for vtype, kwh in EV_CONSUMPTION.items()}

    @staticmethod
    def get_grid_emission_factors() -> Dict[str, str]:
        """
        Return all grid emission factors as a serializable dict.

        Returns:
            Dict mapping country code to kgCO2e/kWh string.
        """
        return {code: str(ef) for code, ef in GRID_EMISSION_FACTORS.items()}

    @staticmethod
    def get_egrid_emission_factors() -> Dict[str, str]:
        """
        Return all eGRID subregion emission factors as a serializable dict.

        Returns:
            Dict mapping subregion code to kgCO2e/kWh string.
        """
        return {code: str(ef) for code, ef in EGRID_EMISSION_FACTORS.items()}

    @staticmethod
    def get_vehicle_age_factors() -> Dict[str, str]:
        """
        Return all vehicle age factors as a serializable dict.

        Returns:
            Dict mapping age category to multiplier string.
        """
        return {age: str(factor) for age, factor in VEHICLE_AGE_FACTORS.items()}

    @staticmethod
    def get_wtt_ratios() -> Dict[str, str]:
        """
        Return all WTT ratios as a serializable dict.

        Returns:
            Dict mapping fuel type to WTT ratio string.
        """
        return {ftype: str(ratio) for ftype, ratio in WTT_RATIOS.items()}

    @staticmethod
    def get_default_annual_km() -> Dict[str, str]:
        """
        Return default annual km by vehicle type as a serializable dict.

        Returns:
            Dict mapping vehicle type to default annual km string.
        """
        return {vtype: str(km) for vtype, km in DEFAULT_ANNUAL_KM.items()}

    @staticmethod
    def get_supported_vehicle_types() -> List[str]:
        """
        Return list of supported vehicle types.

        Returns:
            List of vehicle type key strings.
        """
        return list(VALID_VEHICLE_TYPES)

    @staticmethod
    def get_supported_fuel_types() -> List[str]:
        """
        Return list of supported fuel types.

        Returns:
            List of fuel type key strings.
        """
        return list(VALID_FUEL_TYPES)

    @staticmethod
    def get_eeio_factors() -> Dict[str, Dict[str, str]]:
        """
        Return all EEIO factors as a serializable dict.

        Returns:
            Dict mapping NAICS code to factor info dict.
        """
        return {
            code: {
                "label": info["label"],
                "factor": str(info["factor"]),
                "sector": info["sector"],
            }
            for code, info in EEIO_FACTORS.items()
        }

    @staticmethod
    def get_supported_currencies() -> List[str]:
        """
        Return list of supported currency codes.

        Returns:
            List of ISO currency code strings.
        """
        return list(CURRENCY_RATES.keys())

    @staticmethod
    def get_supported_reporting_years() -> List[int]:
        """
        Return list of supported reporting years for CPI deflation.

        Returns:
            List of year integers.
        """
        return list(CPI_DEFLATORS.keys())

    # ==================================================================
    # ENGINE INFO
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Return engine metadata and status information.

        Returns:
            Dict with engine_id, version, agent_id, calculation_count,
            initialized_at, vehicle_types, fuel_types.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "calculation_count": self._calculation_count,
            "initialized_at": self._initialized_at,
            "vehicle_types": VALID_VEHICLE_TYPES,
            "fuel_types": VALID_FUEL_TYPES,
            "grid_countries": list(GRID_EMISSION_FACTORS.keys()),
            "egrid_subregions": list(EGRID_EMISSION_FACTORS.keys()),
            "supported_currencies": list(CURRENCY_RATES.keys()),
            "supported_years": list(CPI_DEFLATORS.keys()),
        }


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "VehicleFleetCalculatorEngine",
    "get_vehicle_fleet_calculator",
    "reset_vehicle_fleet_calculator",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "VEHICLE_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "EV_CONSUMPTION",
    "WTT_RATIOS",
    "VEHICLE_AGE_FACTORS",
    "GRID_EMISSION_FACTORS",
    "EGRID_EMISSION_FACTORS",
    "EEIO_FACTORS",
    "CURRENCY_RATES",
    "CPI_DEFLATORS",
    "DEFAULT_ANNUAL_KM",
    "DQI_SCORES",
]
