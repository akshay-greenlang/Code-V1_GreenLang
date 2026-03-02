# -*- coding: utf-8 -*-
"""
Personal Vehicle Calculator Engine - Engine 2: Employee Commuting Agent (AGENT-MRV-020)

Calculates GHG emissions from personal vehicle commuting including cars (by size,
fuel type, age) and motorcycles. Supports distance-based and fuel-based methods
with WTT (well-to-tank) and TTW (tank-to-wheel) breakdown.

Primary Formulae:
    Distance-Based (per-vkm):
        annual_distance = one_way_distance x multiplier x working_days x (1 - wfh_fraction)
        ttw_co2e        = annual_distance x ef.co2e_per_km
        wtt_co2e        = ttw_co2e x wtt_factor  (if include_wtt)
        total_co2e      = (ttw_co2e + wtt_co2e) x age_factor x (1 + cold_start) x (1 + urban_driving)

    Fuel-Based (per-litre):
        co2e            = annual_fuel x fuel_ef_per_litre

    Electric Vehicle (per-km via kWh):
        energy_kwh      = annual_distance x consumption_per_km
        co2e            = energy_kwh x grid_factor(country_code)

    Plugin Hybrid (split):
        electric_co2e   = electric_distance x ev_consumption x grid_factor
        ice_co2e        = ice_distance x ice_ef
        total_co2e      = electric_co2e + ice_co2e

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from DEFRA 2024 / EPA 2024

Supports:
    - 12 vehicle types (average, small/medium/large petrol/diesel, hybrid,
      plugin hybrid, BEV, van, motorcycle)
    - 5 fuel types (petrol, diesel, LPG, E10, B7)
    - 4 motorcycle types (small, medium, large, average)
    - 4 EV types (small, medium, large, SUV)
    - Distance-based and fuel-based calculation methods
    - WTT (well-to-tank) upstream emissions
    - Vehicle age degradation (new, mid, old, vintage)
    - Cold start and urban driving uplift factors
    - Plugin hybrid electric/ICE split calculation
    - Miles-to-km and gallons-to-litres unit conversion
    - Batch processing for multiple vehicle calculations
    - Input validation with detailed error messages
    - Provenance hash integration for audit trails
    - Prometheus metrics integration

Example:
    >>> from greenlang.employee_commuting.personal_vehicle_calculator import (
    ...     get_personal_vehicle_calculator,
    ... )
    >>> from decimal import Decimal
    >>> engine = get_personal_vehicle_calculator()
    >>> result = engine.calculate_distance_based(
    ...     vehicle_type="car_medium_petrol",
    ...     fuel_type="petrol",
    ...     one_way_distance_km=Decimal("15.0"),
    ...     working_days=225,
    ...     wfh_fraction=Decimal("0.20"),
    ... )
    >>> assert result["co2e_kg"] > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-020 Employee Commuting (GL-MRV-S3-007)
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

from greenlang.employee_commuting.models import (
    AGENT_COMPONENT,
    AGENT_ID,
    VERSION,
    VehicleType,
    FuelType,
    RegionCode,
    EFSource,
    GWPVersion,
    VEHICLE_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    GRID_EMISSION_FACTORS,
    WORKING_DAYS_DEFAULTS,
    GWP_VALUES,
    calculate_provenance_hash,
)
from greenlang.employee_commuting.metrics import EmployeeCommutingMetrics, get_metrics
from greenlang.employee_commuting.config import get_config
from greenlang.employee_commuting.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ==============================================================================
# ENGINE METADATA
# ==============================================================================

ENGINE_ID: str = "personal_vehicle_calculator_engine"
ENGINE_VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
ROUNDING = ROUND_HALF_UP

# Unit conversion constants
_KM_PER_MILE = Decimal("1.60934")
_LITRES_PER_GALLON = Decimal("3.78541")

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# ==============================================================================
# VEHICLE AGE CATEGORIES
# ==============================================================================

VEHICLE_AGE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "new_0_3yr": {
        "label": "New (0-3 years)",
        "degradation_years": 0,
        "degradation_rate": Decimal("0.005"),
    },
    "mid_4_7yr": {
        "label": "Mid-life (4-7 years)",
        "degradation_years": 5,
        "degradation_rate": Decimal("0.005"),
    },
    "old_8_12yr": {
        "label": "Old (8-12 years)",
        "degradation_years": 10,
        "degradation_rate": Decimal("0.005"),
    },
    "vintage_13plus": {
        "label": "Vintage (13+ years)",
        "degradation_years": 15,
        "degradation_rate": Decimal("0.005"),
    },
}

# ==============================================================================
# MOTORCYCLE EMISSION FACTORS (kgCO2e per vkm) - DEFRA 2024
# ==============================================================================

MOTORCYCLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "small": {
        "ef_per_vkm": Decimal("0.08346"),
        "wtt_per_vkm": Decimal("0.02112"),
        "co2_per_vkm": Decimal("0.08270"),
        "ch4_per_vkm": Decimal("0.00003"),
        "n2o_per_vkm": Decimal("0.00010"),
    },
    "medium": {
        "ef_per_vkm": Decimal("0.10083"),
        "wtt_per_vkm": Decimal("0.02551"),
        "co2_per_vkm": Decimal("0.09990"),
        "ch4_per_vkm": Decimal("0.00003"),
        "n2o_per_vkm": Decimal("0.00012"),
    },
    "large": {
        "ef_per_vkm": Decimal("0.13240"),
        "wtt_per_vkm": Decimal("0.03350"),
        "co2_per_vkm": Decimal("0.13120"),
        "ch4_per_vkm": Decimal("0.00004"),
        "n2o_per_vkm": Decimal("0.00015"),
    },
    "average": {
        "ef_per_vkm": Decimal("0.11337"),
        "wtt_per_vkm": Decimal("0.02867"),
        "co2_per_vkm": Decimal("0.11230"),
        "ch4_per_vkm": Decimal("0.00004"),
        "n2o_per_vkm": Decimal("0.00013"),
    },
}

# ==============================================================================
# EV ENERGY CONSUMPTION (kWh per km) by vehicle size
# Source: DEFRA 2024 / EPA 2024 average BEV consumption
# ==============================================================================

EV_CONSUMPTION: Dict[str, Decimal] = {
    "small_car": Decimal("0.14800"),
    "medium_car": Decimal("0.17200"),
    "large_car": Decimal("0.21200"),
    "suv": Decimal("0.20000"),
}

# ==============================================================================
# FUEL EMISSION FACTORS (kgCO2e per litre) - DEFRA 2024
# Includes individual gas breakdown for detailed reporting
# ==============================================================================

FUEL_EFS: Dict[str, Dict[str, Decimal]] = {
    "gasoline": {
        "co2e_per_litre": Decimal("2.31484"),
        "co2_per_litre": Decimal("2.30220"),
        "ch4_per_litre": Decimal("0.00058"),
        "n2o_per_litre": Decimal("0.00590"),
        "wtt_per_litre": Decimal("0.58549"),
    },
    "petrol": {
        "co2e_per_litre": Decimal("2.31484"),
        "co2_per_litre": Decimal("2.30220"),
        "ch4_per_litre": Decimal("0.00058"),
        "n2o_per_litre": Decimal("0.00590"),
        "wtt_per_litre": Decimal("0.58549"),
    },
    "diesel": {
        "co2e_per_litre": Decimal("2.68787"),
        "co2_per_litre": Decimal("2.67560"),
        "ch4_per_litre": Decimal("0.00010"),
        "n2o_per_litre": Decimal("0.00474"),
        "wtt_per_litre": Decimal("0.60927"),
    },
    "e85": {
        "co2e_per_litre": Decimal("1.61039"),
        "co2_per_litre": Decimal("1.60120"),
        "ch4_per_litre": Decimal("0.00042"),
        "n2o_per_litre": Decimal("0.00354"),
        "wtt_per_litre": Decimal("0.40260"),
    },
    "lpg": {
        "co2e_per_litre": Decimal("1.55363"),
        "co2_per_litre": Decimal("1.54480"),
        "ch4_per_litre": Decimal("0.00045"),
        "n2o_per_litre": Decimal("0.00328"),
        "wtt_per_litre": Decimal("0.32149"),
    },
    "cng": {
        "co2e_per_litre": Decimal("2.02130"),
        "co2_per_litre": Decimal("2.00910"),
        "ch4_per_litre": Decimal("0.00098"),
        "n2o_per_litre": Decimal("0.00244"),
        "wtt_per_litre": Decimal("0.46490"),
    },
}

# ==============================================================================
# VEHICLE-TYPE TO GAS BREAKDOWN FACTORS (kgCO2e components per vkm) - DEFRA 2024
# Provides CO2, CH4, N2O split for per-km EFs
# ==============================================================================

VEHICLE_GAS_BREAKDOWN: Dict[str, Dict[str, Decimal]] = {
    "car_average": {
        "co2_fraction": Decimal("0.99150"),
        "ch4_fraction": Decimal("0.00015"),
        "n2o_fraction": Decimal("0.00835"),
    },
    "car_small_petrol": {
        "co2_fraction": Decimal("0.99200"),
        "ch4_fraction": Decimal("0.00020"),
        "n2o_fraction": Decimal("0.00780"),
    },
    "car_medium_petrol": {
        "co2_fraction": Decimal("0.99180"),
        "ch4_fraction": Decimal("0.00018"),
        "n2o_fraction": Decimal("0.00802"),
    },
    "car_large_petrol": {
        "co2_fraction": Decimal("0.99160"),
        "ch4_fraction": Decimal("0.00016"),
        "n2o_fraction": Decimal("0.00824"),
    },
    "car_small_diesel": {
        "co2_fraction": Decimal("0.99300"),
        "ch4_fraction": Decimal("0.00005"),
        "n2o_fraction": Decimal("0.00695"),
    },
    "car_medium_diesel": {
        "co2_fraction": Decimal("0.99280"),
        "ch4_fraction": Decimal("0.00005"),
        "n2o_fraction": Decimal("0.00715"),
    },
    "car_large_diesel": {
        "co2_fraction": Decimal("0.99260"),
        "ch4_fraction": Decimal("0.00006"),
        "n2o_fraction": Decimal("0.00734"),
    },
    "hybrid": {
        "co2_fraction": Decimal("0.99350"),
        "ch4_fraction": Decimal("0.00012"),
        "n2o_fraction": Decimal("0.00638"),
    },
    "plugin_hybrid": {
        "co2_fraction": Decimal("0.99400"),
        "ch4_fraction": Decimal("0.00010"),
        "n2o_fraction": Decimal("0.00590"),
    },
    "bev": {
        "co2_fraction": Decimal("1.00000"),
        "ch4_fraction": Decimal("0.00000"),
        "n2o_fraction": Decimal("0.00000"),
    },
    "van_average": {
        "co2_fraction": Decimal("0.99100"),
        "ch4_fraction": Decimal("0.00012"),
        "n2o_fraction": Decimal("0.00888"),
    },
    "motorcycle": {
        "co2_fraction": Decimal("0.99060"),
        "ch4_fraction": Decimal("0.00035"),
        "n2o_fraction": Decimal("0.00905"),
    },
}

# ==============================================================================
# WTT RATIO TABLE
# Ratio of WTT emissions to TTW emissions for each vehicle type
# Derived from DEFRA 2024: wtt_per_vkm / ef_per_vkm
# ==============================================================================

WTT_RATIOS: Dict[str, Decimal] = {
    "car_average": Decimal("0.14604"),
    "car_small_petrol": Decimal("0.15905"),
    "car_medium_petrol": Decimal("0.15917"),
    "car_large_petrol": Decimal("0.15907"),
    "car_small_diesel": Decimal("0.14175"),
    "car_medium_diesel": Decimal("0.14168"),
    "car_large_diesel": Decimal("0.14177"),
    "hybrid": Decimal("0.15928"),
    "plugin_hybrid": Decimal("0.13298"),
    "bev": Decimal("0.21114"),
    "van_average": Decimal("0.22536"),
    "motorcycle": Decimal("0.25290"),
}

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["PersonalVehicleCalculatorEngine"] = None
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


# ==============================================================================
# PersonalVehicleCalculatorEngine
# ==============================================================================


class PersonalVehicleCalculatorEngine:
    """
    Engine 2: Personal vehicle emissions calculator for employee commuting.

    Implements deterministic emissions calculations for personal cars (by size,
    fuel type, and age) and motorcycles used in daily commuting. Supports
    distance-based, fuel-based, electric vehicle, and plugin hybrid methods
    using DEFRA 2024 / EPA 2024 emission factors.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with DEFRA/EPA-sourced parameters. No
    LLM calls are made anywhere in the calculation pipeline.

    Thread Safety:
        This engine is fully thread-safe. A reentrant lock protects shared
        state during calculations. The singleton instance is created lazily
        with double-checked locking.

    Attributes:
        _config: Employee commuting configuration singleton.
        _metrics: Prometheus metrics collector for monitoring.
        _provenance: SHA-256 provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.

    Example:
        >>> engine = PersonalVehicleCalculatorEngine.get_instance()
        >>> result = engine.calculate_distance_based(
        ...     vehicle_type="car_medium_petrol",
        ...     fuel_type="petrol",
        ...     one_way_distance_km=Decimal("15.0"),
        ...     working_days=225,
        ...     wfh_fraction=Decimal("0.20"),
        ... )
        >>> assert result["co2e_kg"] > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance(
        metrics: Optional[EmployeeCommutingMetrics] = None,
    ) -> "PersonalVehicleCalculatorEngine":
        """
        Get or create the singleton PersonalVehicleCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Args:
            metrics: Optional Prometheus metrics collector. A default
                     instance is obtained via get_metrics() if None.

        Returns:
            Singleton PersonalVehicleCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = PersonalVehicleCalculatorEngine(
                        metrics=metrics,
                    )
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

    def __init__(
        self,
        metrics: Optional[EmployeeCommutingMetrics] = None,
    ) -> None:
        """
        Initialise the PersonalVehicleCalculatorEngine.

        Args:
            metrics: Optional Prometheus metrics collector. A default
                     instance is obtained via get_metrics() if None.
        """
        self._config = get_config()
        self._metrics: EmployeeCommutingMetrics = metrics or get_metrics()
        self._provenance = get_provenance_tracker()
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0

        logger.info(
            "PersonalVehicleCalculatorEngine initialised: "
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

    # ==================================================================
    # 1. calculate_distance_based
    # ==================================================================

    def calculate_distance_based(
        self,
        vehicle_type: str,
        fuel_type: str,
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal = _ZERO,
        round_trip: bool = True,
        vehicle_age: str = "mid_4_7yr",
        include_wtt: bool = True,
        cold_start_uplift: Decimal = _ZERO,
        urban_driving_uplift: Decimal = _ZERO,
    ) -> Dict[str, Any]:
        """
        Calculate personal vehicle emissions using distance-based method.

        Computes annual commute emissions by multiplying the annual distance
        driven by per-km emission factors, then applying age degradation and
        optional cold start / urban driving uplifts.

        Formula:
            annual_distance = one_way_distance x multiplier x working_days x (1 - wfh_fraction)
            where multiplier = 2 if round_trip else 1

            ttw_co2e = annual_distance x ef.co2e_per_km
            wtt_co2e = ttw_co2e x wtt_factor (if include_wtt, else 0)

            age_factor = 1 + (degradation_years x 0.005)
            total_co2e = (ttw_co2e + wtt_co2e) x age_factor x (1 + cold_start) x (1 + urban_driving)

        Args:
            vehicle_type: Vehicle type key from VEHICLE_EMISSION_FACTORS
                          (e.g., "car_medium_petrol", "car_large_diesel").
            fuel_type: Fuel type for gas breakdown reference
                       (e.g., "petrol", "diesel").
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of working days per year.
            wfh_fraction: Fraction of days working from home (0.0-1.0).
            round_trip: If True, multiply distance by 2 for return journey.
            vehicle_age: Vehicle age category key from VEHICLE_AGE_CATEGORIES.
            include_wtt: If True, include well-to-tank upstream emissions.
            cold_start_uplift: Fractional uplift for cold start emissions (0.0-0.10).
            urban_driving_uplift: Fractional uplift for urban driving conditions (0.0-0.15).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg, co2_kg, ch4_kg,
            n2o_kg, annual_distance_km, ef_used, ef_source, vehicle_type,
            fuel_type, vehicle_age, method, provenance_hash,
            calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.
            KeyError: If vehicle_type not found in emission factors.

        Example:
            >>> result = engine.calculate_distance_based(
            ...     vehicle_type="car_medium_petrol",
            ...     fuel_type="petrol",
            ...     one_way_distance_km=Decimal("15.0"),
            ...     working_days=225,
            ...     wfh_fraction=Decimal("0.20"),
            ... )
            >>> assert result["method"] == "distance_based"
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            errors = self._validate_distance_inputs(
                vehicle_type=vehicle_type,
                fuel_type=fuel_type,
                one_way_distance_km=one_way_distance_km,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                vehicle_age=vehicle_age,
                cold_start_uplift=cold_start_uplift,
                urban_driving_uplift=urban_driving_uplift,
            )
            if errors:
                raise ValueError(
                    f"Distance-based input validation failed: {'; '.join(errors)}"
                )

            # Step 2: Resolve emission factor
            ef = self._get_ef(vehicle_type, fuel_type, vehicle_age)

            # Step 3: Calculate annual distance (ZERO HALLUCINATION - Decimal only)
            multiplier = _TWO if round_trip else _ONE
            one_way = _safe_decimal(one_way_distance_km)
            days = _safe_decimal(working_days)
            wfh = _safe_decimal(wfh_fraction)

            annual_distance = _q(one_way * multiplier * days * (_ONE - wfh))

            # Step 4: Calculate TTW emissions
            ef_per_vkm = ef["ef_per_vkm"]
            ttw_co2e = _q(annual_distance * ef_per_vkm)

            # Step 5: Calculate WTT emissions
            wtt_co2e = _ZERO
            if include_wtt:
                wtt_ratio = ef["wtt_ratio"]
                wtt_co2e = _q(ttw_co2e * wtt_ratio)

            # Step 6: Apply age degradation
            base_co2e = _q(ttw_co2e + wtt_co2e)
            age_factor = self._get_age_factor(vehicle_age)
            aged_co2e = _q(base_co2e * age_factor)

            # Step 7: Apply cold start and urban driving uplifts
            total_co2e = self._apply_uplifts(
                aged_co2e, cold_start_uplift, urban_driving_uplift,
            )

            # Step 8: Calculate gas breakdown
            gas_breakdown = self._calculate_gas_breakdown(
                vehicle_type, ttw_co2e, total_co2e,
            )

            # Step 9: Build provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "vehicle_type": vehicle_type,
                    "fuel_type": fuel_type,
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "round_trip": round_trip,
                    "vehicle_age": vehicle_age,
                    "include_wtt": include_wtt,
                    "cold_start_uplift": str(cold_start_uplift),
                    "urban_driving_uplift": str(urban_driving_uplift),
                },
                {
                    "co2e_kg": str(total_co2e),
                    "ttw_co2e_kg": str(ttw_co2e),
                    "wtt_co2e_kg": str(wtt_co2e),
                    "annual_distance_km": str(annual_distance),
                },
            )

            # Step 10: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": total_co2e,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "co2_kg": gas_breakdown["co2_kg"],
                "ch4_kg": gas_breakdown["ch4_kg"],
                "n2o_kg": gas_breakdown["n2o_kg"],
                "annual_distance_km": annual_distance,
                "ef_used": str(ef_per_vkm),
                "ef_source": EFSource.DEFRA.value,
                "vehicle_type": vehicle_type,
                "fuel_type": fuel_type,
                "vehicle_age": vehicle_age,
                "method": "distance_based",
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 11: Record metrics
            duration = time.monotonic() - start_time
            self._record_vehicle_metrics(
                method="distance_based",
                mode="car",
                vehicle_type=vehicle_type,
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Distance-based calculation complete: type=%s, fuel=%s, "
                "dist=%s km, co2e=%s kg, ttw=%s kg, wtt=%s kg, age=%s",
                vehicle_type,
                fuel_type,
                annual_distance,
                total_co2e,
                ttw_co2e,
                wtt_co2e,
                vehicle_age,
            )

            return result

    # ==================================================================
    # 2. calculate_fuel_based
    # ==================================================================

    def calculate_fuel_based(
        self,
        fuel_type: str,
        fuel_consumed_litres: Decimal,
        period_days: Optional[int] = None,
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
    ) -> Dict[str, Any]:
        """
        Calculate personal vehicle emissions using fuel-based method.

        Computes annual commute emissions from reported fuel consumption,
        optionally annualising from a shorter measurement period.

        Formula:
            If period_days is provided:
                annual_fuel = fuel_consumed x (working_days x (1 - wfh_fraction)) / period_days
            Else:
                annual_fuel = fuel_consumed

            co2e = annual_fuel x fuel_ef_per_litre

        Args:
            fuel_type: Fuel type key (gasoline, petrol, diesel, e85, lpg, cng).
            fuel_consumed_litres: Fuel consumed in litres during measurement
                                  period (or annual total if period_days=None).
            period_days: Number of days in the measurement period. If None,
                         fuel_consumed_litres is treated as annual total.
            working_days: Number of working days per year (default 240).
            wfh_fraction: Fraction of days working from home (0.0-1.0).

        Returns:
            Dict with keys: co2e_kg, wtt_co2e_kg, total_co2e_kg,
            co2_kg, ch4_kg, n2o_kg, fuel_consumed_l, annual_fuel_l,
            fuel_type, ef_used, ef_source, method, provenance_hash,
            calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.
            KeyError: If fuel_type not found in FUEL_EFS.

        Example:
            >>> result = engine.calculate_fuel_based(
            ...     fuel_type="gasoline",
            ...     fuel_consumed_litres=Decimal("50.0"),
            ...     period_days=30,
            ...     working_days=225,
            ...     wfh_fraction=Decimal("0.10"),
            ... )
            >>> assert result["method"] == "fuel_based"
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            errors = self._validate_fuel_inputs(
                fuel_type=fuel_type,
                fuel_consumed_litres=fuel_consumed_litres,
                period_days=period_days,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
            )
            if errors:
                raise ValueError(
                    f"Fuel-based input validation failed: {'; '.join(errors)}"
                )

            # Step 2: Resolve fuel emission factor
            fuel_key = fuel_type.lower().strip()
            if fuel_key not in FUEL_EFS:
                raise KeyError(
                    f"Fuel type '{fuel_type}' not found. "
                    f"Available: {list(FUEL_EFS.keys())}"
                )
            fuel_ef = FUEL_EFS[fuel_key]

            # Step 3: Calculate annual fuel (ZERO HALLUCINATION - Decimal only)
            fuel = _safe_decimal(fuel_consumed_litres)
            wfh = _safe_decimal(wfh_fraction)
            days = _safe_decimal(working_days)

            if period_days is not None:
                period = _safe_decimal(period_days)
                commute_days = _q(days * (_ONE - wfh))
                annual_fuel = _q(fuel * commute_days / period)
            else:
                annual_fuel = fuel

            # Step 4: Calculate emissions
            co2e_per_litre = fuel_ef["co2e_per_litre"]
            co2e = _q(annual_fuel * co2e_per_litre)

            # Step 5: Calculate WTT emissions
            wtt_per_litre = fuel_ef["wtt_per_litre"]
            wtt_co2e = _q(annual_fuel * wtt_per_litre)

            total_co2e = _q(co2e + wtt_co2e)

            # Step 6: Calculate gas breakdown
            co2_kg = _q(annual_fuel * fuel_ef["co2_per_litre"])
            ch4_kg = _q(annual_fuel * fuel_ef["ch4_per_litre"])
            n2o_kg = _q(annual_fuel * fuel_ef["n2o_per_litre"])

            # Step 7: Build provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "fuel_type": fuel_type,
                    "fuel_consumed_litres": str(fuel_consumed_litres),
                    "period_days": period_days,
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                },
                {
                    "co2e_kg": str(co2e),
                    "wtt_co2e_kg": str(wtt_co2e),
                    "annual_fuel_l": str(annual_fuel),
                },
            )

            # Step 8: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e,
                "wtt_co2e_kg": wtt_co2e,
                "total_co2e_kg": total_co2e,
                "co2_kg": co2_kg,
                "ch4_kg": ch4_kg,
                "n2o_kg": n2o_kg,
                "fuel_consumed_l": fuel,
                "annual_fuel_l": annual_fuel,
                "fuel_type": fuel_key,
                "ef_used": str(co2e_per_litre),
                "ef_source": EFSource.DEFRA.value,
                "method": "fuel_based",
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 9: Record metrics
            duration = time.monotonic() - start_time
            self._record_vehicle_metrics(
                method="distance_based",
                mode="car",
                vehicle_type=f"fuel_{fuel_key}",
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Fuel-based calculation complete: fuel=%s, consumed=%s L, "
                "annual=%s L, co2e=%s kg, wtt=%s kg",
                fuel_key,
                fuel,
                annual_fuel,
                co2e,
                wtt_co2e,
            )

            return result

    # ==================================================================
    # 3. calculate_electric_vehicle
    # ==================================================================

    def calculate_electric_vehicle(
        self,
        vehicle_type: str,
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal,
        country_code: str = "US",
        round_trip: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate BEV (battery electric vehicle) commuting emissions.

        Computes emissions from electricity consumed for EV charging,
        using per-km energy consumption and regional grid emission factors.

        Formula:
            annual_distance = one_way_distance x multiplier x working_days x (1 - wfh_fraction)
            energy_kwh = annual_distance x consumption_per_km
            co2e = energy_kwh x grid_factor(country_code)

        Args:
            vehicle_type: EV size category (small_car, medium_car, large_car, suv).
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of working days per year.
            wfh_fraction: Fraction of days working from home (0.0-1.0).
            country_code: ISO 3166-1 alpha-2 country code for grid factor lookup.
            round_trip: If True, multiply distance by 2 for return journey.

        Returns:
            Dict with keys: co2e_kg, energy_kwh, annual_distance_km,
            consumption_per_km, grid_factor, grid_source, vehicle_type,
            country_code, method, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.
            KeyError: If vehicle_type or country_code not found.

        Example:
            >>> result = engine.calculate_electric_vehicle(
            ...     vehicle_type="medium_car",
            ...     one_way_distance_km=Decimal("20.0"),
            ...     working_days=225,
            ...     wfh_fraction=Decimal("0.40"),
            ...     country_code="GB",
            ... )
            >>> assert result["energy_kwh"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            errors = self._validate_ev_inputs(
                vehicle_type=vehicle_type,
                one_way_distance_km=one_way_distance_km,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                country_code=country_code,
            )
            if errors:
                raise ValueError(
                    f"EV input validation failed: {'; '.join(errors)}"
                )

            # Step 2: Resolve EV consumption factor
            ev_key = vehicle_type.lower().strip()
            if ev_key not in EV_CONSUMPTION:
                raise KeyError(
                    f"EV vehicle type '{vehicle_type}' not found. "
                    f"Available: {list(EV_CONSUMPTION.keys())}"
                )
            consumption_per_km = EV_CONSUMPTION[ev_key]

            # Step 3: Resolve grid emission factor
            grid_factor = self._resolve_grid_factor(country_code)

            # Step 4: Calculate annual distance (ZERO HALLUCINATION - Decimal only)
            multiplier = _TWO if round_trip else _ONE
            one_way = _safe_decimal(one_way_distance_km)
            days = _safe_decimal(working_days)
            wfh = _safe_decimal(wfh_fraction)

            annual_distance = _q(one_way * multiplier * days * (_ONE - wfh))

            # Step 5: Calculate energy consumption
            energy_kwh = _q(annual_distance * consumption_per_km)

            # Step 6: Calculate emissions
            co2e = _q(energy_kwh * grid_factor)

            # Step 7: Build provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "vehicle_type": vehicle_type,
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "country_code": country_code,
                    "round_trip": round_trip,
                },
                {
                    "co2e_kg": str(co2e),
                    "energy_kwh": str(energy_kwh),
                    "annual_distance_km": str(annual_distance),
                },
            )

            # Step 8: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e,
                "energy_kwh": energy_kwh,
                "annual_distance_km": annual_distance,
                "consumption_per_km": consumption_per_km,
                "grid_factor": grid_factor,
                "grid_source": EFSource.IEA.value,
                "vehicle_type": ev_key,
                "country_code": country_code,
                "method": "electric_vehicle",
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 9: Record metrics
            duration = time.monotonic() - start_time
            self._record_vehicle_metrics(
                method="distance_based",
                mode="electric_vehicle",
                vehicle_type=ev_key,
                co2e=float(co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "EV calculation complete: type=%s, country=%s, "
                "dist=%s km, kwh=%s, co2e=%s kg",
                ev_key,
                country_code,
                annual_distance,
                energy_kwh,
                co2e,
            )

            return result

    # ==================================================================
    # 4. calculate_motorcycle
    # ==================================================================

    def calculate_motorcycle(
        self,
        motorcycle_type: str,
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal,
        round_trip: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate motorcycle commuting emissions using distance-based method.

        Uses per-vehicle-km emission factors from MOTORCYCLE_EMISSION_FACTORS.
        Motorcycles are always single-occupancy (occupancy = 1.0).

        Formula:
            annual_distance = one_way_distance x multiplier x working_days x (1 - wfh_fraction)
            ttw_co2e = annual_distance x ef_per_vkm
            wtt_co2e = annual_distance x wtt_per_vkm
            total_co2e = ttw_co2e + wtt_co2e

        Args:
            motorcycle_type: Motorcycle size category (small, medium, large, average).
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of working days per year.
            wfh_fraction: Fraction of days working from home (0.0-1.0).
            round_trip: If True, multiply distance by 2 for return journey.

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg, co2_kg, ch4_kg,
            n2o_kg, annual_distance_km, ef_used, ef_source, motorcycle_type,
            method, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.
            KeyError: If motorcycle_type not found in MOTORCYCLE_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_motorcycle(
            ...     motorcycle_type="medium",
            ...     one_way_distance_km=Decimal("12.0"),
            ...     working_days=225,
            ...     wfh_fraction=Decimal("0.0"),
            ... )
            >>> assert result["motorcycle_type"] == "medium"
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            errors: List[str] = []
            moto_key = motorcycle_type.lower().strip()
            if moto_key not in MOTORCYCLE_EMISSION_FACTORS:
                errors.append(
                    f"Unknown motorcycle_type '{motorcycle_type}'. "
                    f"Available: {list(MOTORCYCLE_EMISSION_FACTORS.keys())}"
                )
            self._validate_common_distance_params(
                errors, one_way_distance_km, working_days, wfh_fraction,
            )
            if errors:
                raise ValueError(
                    f"Motorcycle input validation failed: {'; '.join(errors)}"
                )

            # Step 2: Resolve emission factors
            moto_ef = MOTORCYCLE_EMISSION_FACTORS[moto_key]
            ef_per_vkm = moto_ef["ef_per_vkm"]
            wtt_per_vkm = moto_ef["wtt_per_vkm"]

            # Step 3: Calculate annual distance (ZERO HALLUCINATION - Decimal only)
            multiplier = _TWO if round_trip else _ONE
            one_way = _safe_decimal(one_way_distance_km)
            days = _safe_decimal(working_days)
            wfh = _safe_decimal(wfh_fraction)

            annual_distance = _q(one_way * multiplier * days * (_ONE - wfh))

            # Step 4: Calculate emissions
            ttw_co2e = _q(annual_distance * ef_per_vkm)
            wtt_co2e = _q(annual_distance * wtt_per_vkm)
            total_co2e = _q(ttw_co2e + wtt_co2e)

            # Step 5: Calculate gas breakdown
            co2_kg = _q(annual_distance * moto_ef["co2_per_vkm"])
            ch4_kg = _q(annual_distance * moto_ef["ch4_per_vkm"])
            n2o_kg = _q(annual_distance * moto_ef["n2o_per_vkm"])

            # Step 6: Build provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "motorcycle_type": motorcycle_type,
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "round_trip": round_trip,
                },
                {
                    "co2e_kg": str(total_co2e),
                    "ttw_co2e_kg": str(ttw_co2e),
                    "wtt_co2e_kg": str(wtt_co2e),
                    "annual_distance_km": str(annual_distance),
                },
            )

            # Step 7: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": total_co2e,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "co2_kg": co2_kg,
                "ch4_kg": ch4_kg,
                "n2o_kg": n2o_kg,
                "annual_distance_km": annual_distance,
                "ef_used": str(ef_per_vkm),
                "ef_source": EFSource.DEFRA.value,
                "motorcycle_type": moto_key,
                "method": "motorcycle_distance_based",
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 8: Record metrics
            duration = time.monotonic() - start_time
            self._record_vehicle_metrics(
                method="distance_based",
                mode="motorcycle",
                vehicle_type=moto_key,
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "Motorcycle calculation complete: type=%s, dist=%s km, "
                "co2e=%s kg, ttw=%s kg, wtt=%s kg",
                moto_key,
                annual_distance,
                total_co2e,
                ttw_co2e,
                wtt_co2e,
            )

            return result

    # ==================================================================
    # 5. calculate_plugin_hybrid
    # ==================================================================

    def calculate_plugin_hybrid(
        self,
        vehicle_type: str,
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal,
        electric_fraction: Decimal = Decimal("0.50"),
        country_code: str = "US",
        round_trip: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate plugin hybrid electric vehicle (PHEV) commuting emissions.

        Splits driving between electric mode and ICE mode based on the
        electric_fraction parameter. Electric portion uses grid factor;
        ICE portion uses per-km vehicle emission factors.

        Formula:
            annual_distance = one_way_distance x multiplier x working_days x (1 - wfh_fraction)
            electric_distance = annual_distance x electric_fraction
            ice_distance = annual_distance x (1 - electric_fraction)

            electric_co2e = electric_distance x ev_consumption_per_km x grid_factor
            ice_co2e = ice_distance x ice_ef_per_vkm
            wtt_co2e = ice_distance x wtt_per_vkm
            total_co2e = electric_co2e + ice_co2e + wtt_co2e

        Args:
            vehicle_type: Vehicle size for EV consumption lookup
                          (small_car, medium_car, large_car, suv).
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of working days per year.
            wfh_fraction: Fraction of days working from home (0.0-1.0).
            electric_fraction: Fraction of distance driven on electric (0.0-1.0).
            country_code: ISO 3166-1 alpha-2 code for grid factor lookup.
            round_trip: If True, multiply distance by 2 for return journey.

        Returns:
            Dict with keys: co2e_kg, electric_co2e_kg, ice_co2e_kg,
            wtt_co2e_kg, total_co2e_kg, annual_distance_km,
            electric_distance_km, ice_distance_km, electric_fraction,
            vehicle_type, country_code, method, provenance_hash,
            calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.
            KeyError: If vehicle_type or country_code not found.

        Example:
            >>> result = engine.calculate_plugin_hybrid(
            ...     vehicle_type="medium_car",
            ...     one_way_distance_km=Decimal("18.0"),
            ...     working_days=225,
            ...     wfh_fraction=Decimal("0.20"),
            ...     electric_fraction=Decimal("0.60"),
            ...     country_code="GB",
            ... )
            >>> assert result["electric_fraction"] == Decimal("0.60")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            errors: List[str] = []
            ev_key = vehicle_type.lower().strip()
            if ev_key not in EV_CONSUMPTION:
                errors.append(
                    f"Unknown vehicle_type '{vehicle_type}' for PHEV. "
                    f"Available: {list(EV_CONSUMPTION.keys())}"
                )
            self._validate_common_distance_params(
                errors, one_way_distance_km, working_days, wfh_fraction,
            )
            e_frac = _safe_decimal(electric_fraction)
            if e_frac < _ZERO or e_frac > _ONE:
                errors.append(
                    f"electric_fraction must be 0.0-1.0, got {electric_fraction}"
                )
            if errors:
                raise ValueError(
                    f"PHEV input validation failed: {'; '.join(errors)}"
                )

            # Step 2: Resolve factors
            consumption_per_km = EV_CONSUMPTION[ev_key]
            grid_factor = self._resolve_grid_factor(country_code)

            # ICE emission factor for PHEV (use plugin_hybrid from VEHICLE_EMISSION_FACTORS)
            phev_ef = VEHICLE_EMISSION_FACTORS.get(VehicleType.PLUGIN_HYBRID)
            if phev_ef is None:
                raise KeyError("PLUGIN_HYBRID not found in VEHICLE_EMISSION_FACTORS")
            ice_ef_per_vkm = phev_ef["ef_per_vkm"]
            wtt_per_vkm = phev_ef["wtt_per_vkm"]

            # Step 3: Calculate annual distance (ZERO HALLUCINATION - Decimal only)
            multiplier = _TWO if round_trip else _ONE
            one_way = _safe_decimal(one_way_distance_km)
            days = _safe_decimal(working_days)
            wfh = _safe_decimal(wfh_fraction)

            annual_distance = _q(one_way * multiplier * days * (_ONE - wfh))

            # Step 4: Split distance
            electric_distance = _q(annual_distance * e_frac)
            ice_distance = _q(annual_distance * (_ONE - e_frac))

            # Step 5: Calculate electric portion emissions
            electric_kwh = _q(electric_distance * consumption_per_km)
            electric_co2e = _q(electric_kwh * grid_factor)

            # Step 6: Calculate ICE portion emissions
            ice_co2e = _q(ice_distance * ice_ef_per_vkm)
            wtt_co2e = _q(ice_distance * wtt_per_vkm)

            # Step 7: Total
            total_co2e = _q(electric_co2e + ice_co2e + wtt_co2e)

            # Step 8: Build provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "vehicle_type": vehicle_type,
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "electric_fraction": str(electric_fraction),
                    "country_code": country_code,
                    "round_trip": round_trip,
                },
                {
                    "total_co2e_kg": str(total_co2e),
                    "electric_co2e_kg": str(electric_co2e),
                    "ice_co2e_kg": str(ice_co2e),
                    "wtt_co2e_kg": str(wtt_co2e),
                    "annual_distance_km": str(annual_distance),
                },
            )

            # Step 9: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": total_co2e,
                "electric_co2e_kg": electric_co2e,
                "ice_co2e_kg": ice_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "total_co2e_kg": total_co2e,
                "energy_kwh": electric_kwh,
                "annual_distance_km": annual_distance,
                "electric_distance_km": electric_distance,
                "ice_distance_km": ice_distance,
                "electric_fraction": e_frac,
                "consumption_per_km": consumption_per_km,
                "grid_factor": grid_factor,
                "ice_ef_per_vkm": ice_ef_per_vkm,
                "vehicle_type": ev_key,
                "country_code": country_code,
                "method": "plugin_hybrid",
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 10: Record metrics
            duration = time.monotonic() - start_time
            self._record_vehicle_metrics(
                method="distance_based",
                mode="car",
                vehicle_type=f"phev_{ev_key}",
                co2e=float(total_co2e),
                duration=duration,
            )
            self._calculation_count += 1

            logger.debug(
                "PHEV calculation complete: type=%s, e_frac=%.2f, "
                "dist=%s km, e_co2e=%s kg, ice_co2e=%s kg, total=%s kg",
                ev_key,
                float(e_frac),
                annual_distance,
                electric_co2e,
                ice_co2e,
                total_co2e,
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
        Process multiple personal vehicle calculations in a single batch.

        Each item dict must contain a 'method' key indicating the calculation
        type plus the method-specific parameters.

        Supported method values:
            - distance_based
            - fuel_based
            - electric_vehicle
            - motorcycle
            - plugin_hybrid

        Args:
            items: List of dicts, each containing 'method' and method-specific
                   parameters. Maximum batch size is 10000.

        Returns:
            Dict with keys:
                - results: List of individual result dicts (each with 'index',
                  'status', 'method', and either 'result' or 'error').
                - totals: Aggregated co2e_kg across all successful calculations.
                - batch_size: Number of items processed.
                - success_count: Number of successful calculations.
                - error_count: Number of failed calculations.
                - processing_time_ms: Total processing duration.

        Raises:
            ValueError: If items list exceeds _MAX_BATCH_SIZE.

        Example:
            >>> batch = engine.calculate_batch([
            ...     {"method": "distance_based", "vehicle_type": "car_medium_petrol",
            ...      "fuel_type": "petrol", "one_way_distance_km": "15.0",
            ...      "working_days": 225, "wfh_fraction": "0.2"},
            ...     {"method": "motorcycle", "motorcycle_type": "medium",
            ...      "one_way_distance_km": "10.0", "working_days": 225,
            ...      "wfh_fraction": "0.0"},
            ... ])
            >>> assert batch["success_count"] == 2
        """
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        results: List[Dict[str, Any]] = []
        total_co2e = _ZERO
        success_count = 0
        error_count = 0
        start_time = time.monotonic()

        for idx, item in enumerate(items):
            method = item.get("method", "unknown")
            try:
                calc_result = self._dispatch_calculation(item)
                results.append({
                    "index": idx,
                    "method": method,
                    "status": "success",
                    "result": calc_result,
                })
                # Accumulate co2e from the result
                result_co2e = calc_result.get(
                    "co2e_kg",
                    calc_result.get("total_co2e_kg", _ZERO),
                )
                if isinstance(result_co2e, Decimal):
                    total_co2e = _q(total_co2e + result_co2e)
                success_count += 1

            except Exception as exc:
                logger.warning(
                    "Batch item %d (%s) failed: %s",
                    idx, method, str(exc),
                )
                results.append({
                    "index": idx,
                    "method": method,
                    "status": "error",
                    "error": str(exc),
                })
                error_count += 1

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Batch personal vehicle calculation complete: "
            "total=%d, success=%d, errors=%d, co2e=%s kg, duration=%.2fms",
            len(items),
            success_count,
            error_count,
            total_co2e,
            duration_ms,
        )

        return {
            "results": results,
            "totals": {
                "co2e_kg": total_co2e,
            },
            "batch_size": len(items),
            "success_count": success_count,
            "error_count": error_count,
            "processing_time_ms": _q(_safe_decimal(duration_ms)),
        }

    # ==================================================================
    # 7. estimate_annual_emissions
    # ==================================================================

    def estimate_annual_emissions(
        self,
        vehicle_type: str,
        fuel_type: str,
        one_way_distance_km: Decimal,
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
    ) -> Dict[str, Any]:
        """
        Quick estimate of annual commuting emissions with sensible defaults.

        Convenience method that calls calculate_distance_based with default
        settings (round_trip=True, mid_4_7yr age, WTT included, no uplifts).

        Args:
            vehicle_type: Vehicle type key from VEHICLE_EMISSION_FACTORS.
            fuel_type: Fuel type for gas breakdown reference.
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of working days per year (default 240).
            wfh_fraction: Fraction of days working from home (default 0.0).

        Returns:
            Same Dict structure as calculate_distance_based.

        Raises:
            ValueError: If any input parameter is invalid.
            KeyError: If vehicle_type not found.

        Example:
            >>> result = engine.estimate_annual_emissions(
            ...     vehicle_type="car_average",
            ...     fuel_type="petrol",
            ...     one_way_distance_km=Decimal("15.0"),
            ... )
            >>> assert result["method"] == "distance_based"
        """
        return self.calculate_distance_based(
            vehicle_type=vehicle_type,
            fuel_type=fuel_type,
            one_way_distance_km=one_way_distance_km,
            working_days=working_days,
            wfh_fraction=wfh_fraction,
            round_trip=True,
            vehicle_age="mid_4_7yr",
            include_wtt=True,
            cold_start_uplift=_ZERO,
            urban_driving_uplift=_ZERO,
        )

    # ==================================================================
    # Emission Factor Resolution
    # ==================================================================

    def _get_ef(
        self,
        vehicle_type: str,
        fuel_type: str,
        vehicle_age: str,
    ) -> Dict[str, Decimal]:
        """
        Resolve emission factors for a given vehicle type.

        Looks up per-vkm emission factors from VEHICLE_EMISSION_FACTORS and
        derives the WTT ratio from WTT_RATIOS. Falls back to average car
        factors if the specific vehicle type is not found.

        Args:
            vehicle_type: Vehicle type key (e.g., "car_medium_petrol").
            fuel_type: Fuel type for reference logging.
            vehicle_age: Vehicle age category for reference logging.

        Returns:
            Dict with 'ef_per_vkm', 'wtt_per_vkm', 'wtt_ratio', 'occupancy'.

        Raises:
            KeyError: If vehicle_type not found in any factor table.
        """
        vt_key = vehicle_type.lower().strip()

        # Try to find by VehicleType enum
        vt_enum = None
        for member in VehicleType:
            if member.value == vt_key:
                vt_enum = member
                break

        if vt_enum is None:
            raise KeyError(
                f"Vehicle type '{vehicle_type}' not found in VehicleType enum. "
                f"Available: {[m.value for m in VehicleType]}"
            )

        ef_entry = VEHICLE_EMISSION_FACTORS.get(vt_enum)
        if ef_entry is None:
            raise KeyError(
                f"No emission factors for vehicle type '{vehicle_type}'"
            )

        ef_per_vkm = ef_entry["ef_per_vkm"]
        wtt_per_vkm = ef_entry["wtt_per_vkm"]

        # Derive WTT ratio
        wtt_ratio = WTT_RATIOS.get(vt_key)
        if wtt_ratio is None:
            # Calculate from raw values if not in ratio table
            if ef_per_vkm > _ZERO:
                wtt_ratio = _q(wtt_per_vkm / ef_per_vkm)
            else:
                wtt_ratio = _ZERO

        occupancy = ef_entry.get("occupancy", _ONE)

        logger.debug(
            "Resolved vehicle EF: type=%s, fuel=%s, age=%s, "
            "ef_vkm=%s, wtt_vkm=%s, wtt_ratio=%s, occ=%s",
            vt_key,
            fuel_type,
            vehicle_age,
            ef_per_vkm,
            wtt_per_vkm,
            wtt_ratio,
            occupancy,
        )

        return {
            "ef_per_vkm": ef_per_vkm,
            "wtt_per_vkm": wtt_per_vkm,
            "wtt_ratio": wtt_ratio,
            "occupancy": occupancy if occupancy is not None else _ONE,
        }

    # ==================================================================
    # Age Degradation
    # ==================================================================

    def _apply_age_degradation(
        self,
        co2e: Decimal,
        vehicle_age: str,
    ) -> Decimal:
        """
        Apply vehicle age degradation factor to emissions.

        Older vehicles emit more due to engine wear, catalytic converter
        degradation, and reduced fuel efficiency. The degradation rate is
        0.5% per year of age.

        Args:
            co2e: Base emissions in kgCO2e.
            vehicle_age: Vehicle age category key.

        Returns:
            Emissions with age degradation applied.
        """
        age_factor = self._get_age_factor(vehicle_age)
        return _q(co2e * age_factor)

    def _get_age_factor(self, vehicle_age: str) -> Decimal:
        """
        Calculate the multiplicative age degradation factor.

        Formula:
            age_factor = 1 + (degradation_years x degradation_rate)
            where degradation_rate = 0.005 (0.5% per year)

        Args:
            vehicle_age: Vehicle age category key from VEHICLE_AGE_CATEGORIES.

        Returns:
            Age factor as Decimal (>= 1.0).
        """
        age_key = vehicle_age.lower().strip()
        if age_key not in VEHICLE_AGE_CATEGORIES:
            logger.warning(
                "Unknown vehicle_age '%s', defaulting to mid_4_7yr", vehicle_age,
            )
            age_key = "mid_4_7yr"

        age_cat = VEHICLE_AGE_CATEGORIES[age_key]
        degradation_years = _safe_decimal(age_cat["degradation_years"])
        degradation_rate = age_cat["degradation_rate"]

        age_factor = _q(_ONE + (degradation_years * degradation_rate))

        logger.debug(
            "Age factor: age=%s, years=%s, rate=%s, factor=%s",
            age_key,
            degradation_years,
            degradation_rate,
            age_factor,
        )

        return age_factor

    # ==================================================================
    # Uplift Application
    # ==================================================================

    def _apply_uplifts(
        self,
        co2e: Decimal,
        cold_start_uplift: Decimal,
        urban_driving_uplift: Decimal,
    ) -> Decimal:
        """
        Apply cold start and urban driving uplift factors.

        Cold start uplift accounts for higher emissions during engine warm-up
        on short trips. Urban driving uplift accounts for stop-start traffic
        conditions that reduce fuel efficiency.

        Formula:
            adjusted = co2e x (1 + cold_start) x (1 + urban_driving)

        Args:
            co2e: Base emissions in kgCO2e.
            cold_start_uplift: Fractional cold start uplift (0.0-0.10).
            urban_driving_uplift: Fractional urban driving uplift (0.0-0.15).

        Returns:
            Emissions with uplifts applied.
        """
        cold = _safe_decimal(cold_start_uplift)
        urban = _safe_decimal(urban_driving_uplift)

        adjusted = _q(co2e * (_ONE + cold) * (_ONE + urban))

        if cold > _ZERO or urban > _ZERO:
            logger.debug(
                "Applied uplifts: cold_start=%s, urban=%s, "
                "before=%s, after=%s",
                cold,
                urban,
                co2e,
                adjusted,
            )

        return adjusted

    # ==================================================================
    # Gas Breakdown
    # ==================================================================

    def _calculate_gas_breakdown(
        self,
        vehicle_type: str,
        ttw_co2e: Decimal,
        total_co2e: Decimal,
    ) -> Dict[str, Decimal]:
        """
        Calculate individual gas breakdown (CO2, CH4, N2O) from total CO2e.

        Uses predefined fraction tables from VEHICLE_GAS_BREAKDOWN to split
        the TTW CO2e into individual gas components.

        Args:
            vehicle_type: Vehicle type key for fraction lookup.
            ttw_co2e: Tank-to-wheel CO2e to split.
            total_co2e: Total CO2e (for reference; breakdown is from TTW).

        Returns:
            Dict with co2_kg, ch4_kg, n2o_kg (all Decimal).
        """
        vt_key = vehicle_type.lower().strip()
        breakdown = VEHICLE_GAS_BREAKDOWN.get(vt_key)

        if breakdown is None:
            # Default: attribute all to CO2
            logger.debug(
                "No gas breakdown for '%s', attributing all to CO2",
                vehicle_type,
            )
            return {
                "co2_kg": ttw_co2e,
                "ch4_kg": _ZERO,
                "n2o_kg": _ZERO,
            }

        co2_kg = _q(ttw_co2e * breakdown["co2_fraction"])
        ch4_kg = _q(ttw_co2e * breakdown["ch4_fraction"])
        n2o_kg = _q(ttw_co2e * breakdown["n2o_fraction"])

        return {
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
        }

    # ==================================================================
    # Grid Factor Resolution
    # ==================================================================

    def _resolve_grid_factor(self, country_code: str) -> Decimal:
        """
        Resolve grid emission factor for a country/region.

        Looks up the regional grid emission factor from GRID_EMISSION_FACTORS.
        Falls back to GLOBAL if the specific country is not found.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., "US", "GB").

        Returns:
            Grid emission factor in kgCO2e per kWh.
        """
        code = country_code.upper().strip()

        # Try to find by RegionCode enum
        for member in RegionCode:
            if member.value == code:
                factor = GRID_EMISSION_FACTORS.get(member)
                if factor is not None:
                    logger.debug(
                        "Resolved grid factor: country=%s, factor=%s kgCO2e/kWh",
                        code,
                        factor,
                    )
                    return factor

        # Fall back to GLOBAL
        global_factor = GRID_EMISSION_FACTORS[RegionCode.GLOBAL]
        logger.warning(
            "Grid factor not found for '%s', using GLOBAL default: %s kgCO2e/kWh",
            country_code,
            global_factor,
        )
        return global_factor

    # ==================================================================
    # Provenance Hash
    # ==================================================================

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Creates a deterministic hash from the input parameters and result
        values to enable verification that results were not tampered with.

        Args:
            input_data: Dictionary of input parameters.
            result_data: Dictionary of key result values.

        Returns:
            Hexadecimal SHA-256 hash string (64 characters).
        """
        combined = {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "input": input_data,
            "output": result_data,
        }
        hash_str = json.dumps(combined, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

    # ==================================================================
    # Input Validation
    # ==================================================================

    def _validate_inputs(self, params: Dict[str, Any]) -> List[str]:
        """
        General-purpose input validation.

        Validates common parameters used across multiple calculation methods.

        Args:
            params: Dictionary of parameters to validate.

        Returns:
            List of error messages (empty if all valid).
        """
        errors: List[str] = []

        if "one_way_distance_km" in params:
            dist = params["one_way_distance_km"]
            try:
                d = _safe_decimal(dist)
                if d <= _ZERO:
                    errors.append(
                        f"one_way_distance_km must be positive, got {dist}"
                    )
                if d > Decimal("500"):
                    errors.append(
                        f"one_way_distance_km exceeds 500 km: {dist}. "
                        "This may not be a daily commute."
                    )
            except ValueError:
                errors.append(
                    f"one_way_distance_km is not a valid number: {dist}"
                )

        if "working_days" in params:
            wd = params["working_days"]
            if not isinstance(wd, int) or wd < 1 or wd > 366:
                errors.append(
                    f"working_days must be integer 1-366, got {wd}"
                )

        if "wfh_fraction" in params:
            wfh = params["wfh_fraction"]
            try:
                w = _safe_decimal(wfh)
                if w < _ZERO or w > _ONE:
                    errors.append(
                        f"wfh_fraction must be 0.0-1.0, got {wfh}"
                    )
            except ValueError:
                errors.append(
                    f"wfh_fraction is not a valid number: {wfh}"
                )

        return errors

    def _validate_distance_inputs(
        self,
        vehicle_type: str,
        fuel_type: str,
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal,
        vehicle_age: str,
        cold_start_uplift: Decimal,
        urban_driving_uplift: Decimal,
    ) -> List[str]:
        """
        Validate inputs for distance-based calculation.

        Args:
            vehicle_type: Vehicle type key.
            fuel_type: Fuel type key.
            one_way_distance_km: One-way distance.
            working_days: Annual working days.
            wfh_fraction: WFH fraction.
            vehicle_age: Vehicle age category.
            cold_start_uplift: Cold start uplift.
            urban_driving_uplift: Urban driving uplift.

        Returns:
            List of error messages (empty if all valid).
        """
        errors: List[str] = []

        # Validate vehicle type
        vt_key = vehicle_type.lower().strip()
        valid_types = [m.value for m in VehicleType]
        if vt_key not in valid_types:
            errors.append(
                f"Unknown vehicle_type '{vehicle_type}'. "
                f"Available: {valid_types}"
            )

        # Validate vehicle age
        age_key = vehicle_age.lower().strip()
        if age_key not in VEHICLE_AGE_CATEGORIES:
            errors.append(
                f"Unknown vehicle_age '{vehicle_age}'. "
                f"Available: {list(VEHICLE_AGE_CATEGORIES.keys())}"
            )

        # Validate common distance params
        self._validate_common_distance_params(
            errors, one_way_distance_km, working_days, wfh_fraction,
        )

        # Validate uplifts
        try:
            cs = _safe_decimal(cold_start_uplift)
            if cs < _ZERO or cs > Decimal("0.10"):
                errors.append(
                    f"cold_start_uplift must be 0.0-0.10, got {cold_start_uplift}"
                )
        except ValueError:
            errors.append(
                f"cold_start_uplift is not a valid number: {cold_start_uplift}"
            )

        try:
            ud = _safe_decimal(urban_driving_uplift)
            if ud < _ZERO or ud > Decimal("0.15"):
                errors.append(
                    f"urban_driving_uplift must be 0.0-0.15, got {urban_driving_uplift}"
                )
        except ValueError:
            errors.append(
                f"urban_driving_uplift is not a valid number: {urban_driving_uplift}"
            )

        return errors

    def _validate_fuel_inputs(
        self,
        fuel_type: str,
        fuel_consumed_litres: Decimal,
        period_days: Optional[int],
        working_days: int,
        wfh_fraction: Decimal,
    ) -> List[str]:
        """
        Validate inputs for fuel-based calculation.

        Args:
            fuel_type: Fuel type key.
            fuel_consumed_litres: Fuel consumed.
            period_days: Measurement period days.
            working_days: Annual working days.
            wfh_fraction: WFH fraction.

        Returns:
            List of error messages (empty if all valid).
        """
        errors: List[str] = []

        # Validate fuel type
        fuel_key = fuel_type.lower().strip()
        if fuel_key not in FUEL_EFS:
            errors.append(
                f"Unknown fuel_type '{fuel_type}'. "
                f"Available: {list(FUEL_EFS.keys())}"
            )

        # Validate fuel quantity
        try:
            fuel = _safe_decimal(fuel_consumed_litres)
            if fuel <= _ZERO:
                errors.append(
                    f"fuel_consumed_litres must be positive, got {fuel_consumed_litres}"
                )
        except ValueError:
            errors.append(
                f"fuel_consumed_litres is not a valid number: {fuel_consumed_litres}"
            )

        # Validate period days
        if period_days is not None:
            if not isinstance(period_days, int) or period_days < 1:
                errors.append(
                    f"period_days must be positive integer, got {period_days}"
                )

        # Validate working days
        if not isinstance(working_days, int) or working_days < 1 or working_days > 366:
            errors.append(
                f"working_days must be integer 1-366, got {working_days}"
            )

        # Validate WFH fraction
        try:
            w = _safe_decimal(wfh_fraction)
            if w < _ZERO or w > _ONE:
                errors.append(
                    f"wfh_fraction must be 0.0-1.0, got {wfh_fraction}"
                )
        except ValueError:
            errors.append(
                f"wfh_fraction is not a valid number: {wfh_fraction}"
            )

        return errors

    def _validate_ev_inputs(
        self,
        vehicle_type: str,
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal,
        country_code: str,
    ) -> List[str]:
        """
        Validate inputs for electric vehicle calculation.

        Args:
            vehicle_type: EV size category.
            one_way_distance_km: One-way distance.
            working_days: Annual working days.
            wfh_fraction: WFH fraction.
            country_code: Country code for grid factor.

        Returns:
            List of error messages (empty if all valid).
        """
        errors: List[str] = []

        # Validate EV type
        ev_key = vehicle_type.lower().strip()
        if ev_key not in EV_CONSUMPTION:
            errors.append(
                f"Unknown EV vehicle_type '{vehicle_type}'. "
                f"Available: {list(EV_CONSUMPTION.keys())}"
            )

        # Validate common distance params
        self._validate_common_distance_params(
            errors, one_way_distance_km, working_days, wfh_fraction,
        )

        # Validate country code (non-empty string)
        if not country_code or not isinstance(country_code, str):
            errors.append("country_code must be a non-empty string")

        return errors

    def _validate_common_distance_params(
        self,
        errors: List[str],
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal,
    ) -> None:
        """
        Validate common parameters shared by distance-based calculations.

        Appends error messages to the provided errors list.

        Args:
            errors: List to append error messages to.
            one_way_distance_km: One-way distance.
            working_days: Annual working days.
            wfh_fraction: WFH fraction.
        """
        # Validate distance
        try:
            d = _safe_decimal(one_way_distance_km)
            if d <= _ZERO:
                errors.append(
                    f"one_way_distance_km must be positive, got {one_way_distance_km}"
                )
            if d > Decimal("500"):
                errors.append(
                    f"one_way_distance_km exceeds 500 km: {one_way_distance_km}. "
                    "This may not be a daily commute."
                )
        except ValueError:
            errors.append(
                f"one_way_distance_km is not a valid number: {one_way_distance_km}"
            )

        # Validate working days
        if not isinstance(working_days, int) or working_days < 1 or working_days > 366:
            errors.append(
                f"working_days must be integer 1-366, got {working_days}"
            )

        # Validate WFH fraction
        try:
            w = _safe_decimal(wfh_fraction)
            if w < _ZERO or w > _ONE:
                errors.append(
                    f"wfh_fraction must be 0.0-1.0, got {wfh_fraction}"
                )
        except ValueError:
            errors.append(
                f"wfh_fraction is not a valid number: {wfh_fraction}"
            )

    # ==================================================================
    # Batch Dispatch
    # ==================================================================

    def _dispatch_calculation(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dispatch a single calculation based on the method field.

        Args:
            item: Dict with 'method' and method-specific parameters.

        Returns:
            Method-specific result dict.

        Raises:
            ValueError: If method is unknown or parameters are invalid.
        """
        method = item.get("method", "").lower().strip()

        if method == "distance_based":
            return self.calculate_distance_based(
                vehicle_type=item["vehicle_type"],
                fuel_type=item.get("fuel_type", "petrol"),
                one_way_distance_km=_safe_decimal(item["one_way_distance_km"]),
                working_days=int(item.get("working_days", 240)),
                wfh_fraction=_safe_decimal(item.get("wfh_fraction", "0")),
                round_trip=item.get("round_trip", True),
                vehicle_age=item.get("vehicle_age", "mid_4_7yr"),
                include_wtt=item.get("include_wtt", True),
                cold_start_uplift=_safe_decimal(
                    item.get("cold_start_uplift", "0")
                ),
                urban_driving_uplift=_safe_decimal(
                    item.get("urban_driving_uplift", "0")
                ),
            )

        elif method == "fuel_based":
            return self.calculate_fuel_based(
                fuel_type=item["fuel_type"],
                fuel_consumed_litres=_safe_decimal(item["fuel_consumed_litres"]),
                period_days=item.get("period_days"),
                working_days=int(item.get("working_days", 240)),
                wfh_fraction=_safe_decimal(item.get("wfh_fraction", "0")),
            )

        elif method == "electric_vehicle":
            return self.calculate_electric_vehicle(
                vehicle_type=item["vehicle_type"],
                one_way_distance_km=_safe_decimal(item["one_way_distance_km"]),
                working_days=int(item.get("working_days", 240)),
                wfh_fraction=_safe_decimal(item.get("wfh_fraction", "0")),
                country_code=item.get("country_code", "US"),
                round_trip=item.get("round_trip", True),
            )

        elif method == "motorcycle":
            return self.calculate_motorcycle(
                motorcycle_type=item.get("motorcycle_type", "average"),
                one_way_distance_km=_safe_decimal(item["one_way_distance_km"]),
                working_days=int(item.get("working_days", 240)),
                wfh_fraction=_safe_decimal(item.get("wfh_fraction", "0")),
                round_trip=item.get("round_trip", True),
            )

        elif method == "plugin_hybrid":
            return self.calculate_plugin_hybrid(
                vehicle_type=item.get("vehicle_type", "medium_car"),
                one_way_distance_km=_safe_decimal(item["one_way_distance_km"]),
                working_days=int(item.get("working_days", 240)),
                wfh_fraction=_safe_decimal(item.get("wfh_fraction", "0")),
                electric_fraction=_safe_decimal(
                    item.get("electric_fraction", "0.50")
                ),
                country_code=item.get("country_code", "US"),
                round_trip=item.get("round_trip", True),
            )

        else:
            raise ValueError(
                f"Unknown calculation method '{method}'. "
                "Supported: distance_based, fuel_based, electric_vehicle, "
                "motorcycle, plugin_hybrid"
            )

    # ==================================================================
    # Emission Factor & Vehicle Accessors
    # ==================================================================

    @staticmethod
    def get_supported_vehicles() -> List[Dict[str, Any]]:
        """
        Return all supported vehicle types with their emission factors.

        Returns:
            List of dicts with vehicle_type, ef_per_vkm, wtt_per_vkm,
            occupancy for each supported vehicle.
        """
        vehicles: List[Dict[str, Any]] = []

        # Cars from VEHICLE_EMISSION_FACTORS
        for vt, ef in VEHICLE_EMISSION_FACTORS.items():
            vehicles.append({
                "vehicle_type": vt.value,
                "category": "car" if "car" in vt.value else (
                    "motorcycle" if vt.value == "motorcycle" else (
                        "van" if "van" in vt.value else "other"
                    )
                ),
                "ef_per_vkm": ef["ef_per_vkm"],
                "wtt_per_vkm": ef["wtt_per_vkm"],
                "occupancy": ef.get("occupancy"),
                "ef_source": EFSource.DEFRA.value,
            })

        # Motorcycles from MOTORCYCLE_EMISSION_FACTORS
        for mtype, mef in MOTORCYCLE_EMISSION_FACTORS.items():
            vehicles.append({
                "vehicle_type": f"motorcycle_{mtype}",
                "category": "motorcycle",
                "ef_per_vkm": mef["ef_per_vkm"],
                "wtt_per_vkm": mef["wtt_per_vkm"],
                "occupancy": _ONE,
                "ef_source": EFSource.DEFRA.value,
            })

        # EVs from EV_CONSUMPTION
        for ev_type, consumption in EV_CONSUMPTION.items():
            vehicles.append({
                "vehicle_type": f"ev_{ev_type}",
                "category": "electric_vehicle",
                "consumption_per_km_kwh": consumption,
                "ef_per_vkm": None,
                "wtt_per_vkm": None,
                "occupancy": _ONE,
                "ef_source": EFSource.IEA.value,
            })

        return vehicles

    @staticmethod
    def get_supported_fuels() -> List[str]:
        """
        Return all supported fuel types.

        Returns:
            List of fuel type keys from FUEL_EFS.
        """
        return list(FUEL_EFS.keys())

    @staticmethod
    def get_supported_motorcycle_types() -> List[str]:
        """
        Return all supported motorcycle types.

        Returns:
            List of motorcycle type keys.
        """
        return list(MOTORCYCLE_EMISSION_FACTORS.keys())

    @staticmethod
    def get_supported_ev_types() -> List[str]:
        """
        Return all supported EV types.

        Returns:
            List of EV type keys from EV_CONSUMPTION.
        """
        return list(EV_CONSUMPTION.keys())

    @staticmethod
    def get_vehicle_age_categories() -> Dict[str, Dict[str, Any]]:
        """
        Return all vehicle age categories with degradation parameters.

        Returns:
            Dict of age category key -> {label, degradation_years, degradation_rate}.
        """
        return dict(VEHICLE_AGE_CATEGORIES)

    @staticmethod
    def get_fuel_emission_factors() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all fuel emission factors.

        Returns:
            Dict mapping fuel type to factor dict.
        """
        return dict(FUEL_EFS)

    @staticmethod
    def get_grid_emission_factors() -> Dict[str, Decimal]:
        """
        Return all grid emission factors by region.

        Returns:
            Dict mapping region code to grid factor (kgCO2e/kWh).
        """
        return {rc.value: factor for rc, factor in GRID_EMISSION_FACTORS.items()}

    # ==================================================================
    # Unit Conversion Helpers
    # ==================================================================

    @staticmethod
    def convert_miles_to_km(miles: Decimal) -> Decimal:
        """
        Convert miles to kilometres.

        Uses the standard conversion factor: 1 mile = 1.60934 km.

        Args:
            miles: Distance in miles. Must be non-negative.

        Returns:
            Distance in kilometres, quantized to 8 decimal places.

        Raises:
            ValueError: If miles is negative.

        Example:
            >>> PersonalVehicleCalculatorEngine.convert_miles_to_km(Decimal("10"))
            Decimal('16.09340000')
        """
        if miles < _ZERO:
            raise ValueError(f"Miles must be non-negative, got {miles}")
        return _q(miles * _KM_PER_MILE)

    @staticmethod
    def convert_gallons_to_litres(gallons: Decimal) -> Decimal:
        """
        Convert US gallons to litres.

        Uses the standard conversion factor: 1 US gallon = 3.78541 litres.

        Args:
            gallons: Volume in US gallons. Must be non-negative.

        Returns:
            Volume in litres, quantized to 8 decimal places.

        Raises:
            ValueError: If gallons is negative.

        Example:
            >>> PersonalVehicleCalculatorEngine.convert_gallons_to_litres(Decimal("10"))
            Decimal('37.85410000')
        """
        if gallons < _ZERO:
            raise ValueError(f"Gallons must be non-negative, got {gallons}")
        return _q(gallons * _LITRES_PER_GALLON)

    # ==================================================================
    # Internal: Metrics Recording
    # ==================================================================

    def _record_vehicle_metrics(
        self,
        method: str,
        mode: str,
        vehicle_type: str,
        co2e: float,
        duration: float,
    ) -> None:
        """
        Record personal vehicle calculation metrics to Prometheus.

        Wraps calls to the EmployeeCommutingMetrics singleton to record
        calculation throughput, emissions, duration, and commute trip
        counters. All calls are wrapped in try/except to prevent
        metrics failures from disrupting calculations.

        Args:
            method: Calculation method label (distance_based / fuel_based).
            mode: Transport mode label (car / motorcycle / electric_vehicle).
            vehicle_type: Vehicle type label for commute counter.
            co2e: Emissions in kgCO2e for emissions counter.
            duration: Calculation duration in seconds.
        """
        try:
            self._metrics.record_calculation(
                method=method,
                mode=mode,
                status="success",
                duration=duration,
                co2e=co2e,
                category="transport",
            )
            self._metrics.record_commute(
                mode=mode,
                vehicle_type=vehicle_type,
            )
        except Exception as exc:
            logger.warning(
                "Failed to record personal vehicle metrics: %s", str(exc),
            )


# ==============================================================================
# MODULE-LEVEL ACCESSORS
# ==============================================================================


def get_personal_vehicle_calculator() -> PersonalVehicleCalculatorEngine:
    """
    Get the singleton PersonalVehicleCalculatorEngine instance.

    Convenience function for module-level access to the engine singleton.
    Thread-safe; uses double-checked locking internally.

    Returns:
        Singleton PersonalVehicleCalculatorEngine instance.

    Example:
        >>> engine = get_personal_vehicle_calculator()
        >>> result = engine.calculate_distance_based(...)
    """
    return PersonalVehicleCalculatorEngine.get_instance()


def reset_personal_vehicle_calculator() -> None:
    """
    Reset the singleton PersonalVehicleCalculatorEngine instance.

    Intended exclusively for unit tests that need a fresh engine instance.
    Should never be called in production code.

    Example:
        >>> reset_personal_vehicle_calculator()
        >>> engine = get_personal_vehicle_calculator()
        >>> assert engine.calculation_count == 0
    """
    PersonalVehicleCalculatorEngine.reset_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "PersonalVehicleCalculatorEngine",
    "get_personal_vehicle_calculator",
    "reset_personal_vehicle_calculator",
    "MOTORCYCLE_EMISSION_FACTORS",
    "EV_CONSUMPTION",
    "FUEL_EFS",
    "VEHICLE_AGE_CATEGORIES",
    "VEHICLE_GAS_BREAKDOWN",
    "WTT_RATIOS",
]
