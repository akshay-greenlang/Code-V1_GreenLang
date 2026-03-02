# -*- coding: utf-8 -*-
"""
Active Transport Calculator Engine - Engine 5: Employee Commuting Agent (AGENT-MRV-020)

Calculates GHG emissions from active and micro-mobility commuting modes including
cycling, walking, e-bikes, and e-scooters. Active modes (cycling/walking) have zero
operational emissions but optional lifecycle emissions. E-bikes and e-scooters have
electricity-based operational emissions plus lifecycle manufacturing emissions.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Lifecycle factors from peer-reviewed LCA studies

Supported Modes (6):
    - Cycling (pedal bicycle) - zero operational, optional lifecycle
    - Walking (pedestrian) - zero operational, optional lifecycle
    - E-bike standard (250W pedelec) - electricity + lifecycle
    - E-bike cargo (cargo pedelec) - electricity + lifecycle
    - E-bike speed pedelec (45 km/h S-pedelec) - electricity + lifecycle
    - E-scooter personal (privately owned kick-scooter) - electricity + lifecycle
    - E-scooter shared (dockless rental kick-scooter) - electricity + lifecycle

Formulae:
    Active Transport (cycling / walking):
        annual_distance = one_way_km x multiplier x working_days x (1 - wfh_fraction)
        operational_co2e = 0  (human-powered)
        lifecycle_co2e = annual_distance x lifecycle_ef_per_km  (if included)
        total_co2e = operational_co2e + lifecycle_co2e

    Electric Micro-Mobility (e-bike / e-scooter):
        annual_distance = one_way_km x multiplier x working_days x (1 - wfh_fraction)
        energy_kwh = annual_distance x energy_kwh_per_km
        operational_co2e = energy_kwh x grid_factor
        manufacturing_co2e = annual_distance x manufacturing_per_km
        battery_co2e = annual_distance x (battery_replacement_co2e / battery_life_km)
        maintenance_co2e = annual_distance x maintenance_per_km
        lifecycle_co2e = manufacturing_co2e + battery_co2e + maintenance_co2e
        total_co2e = operational_co2e + (lifecycle_co2e if include_lifecycle else 0)

    Avoided Emissions:
        avoided_co2e = alternative_mode_co2e - active_mode_co2e
        avoided_pct = avoided_co2e / alternative_mode_co2e x 100

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
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    RegionCode,
    GRID_EMISSION_FACTORS,
    VEHICLE_EMISSION_FACTORS,
    VehicleType,
    calculate_provenance_hash,
)
from greenlang.employee_commuting.metrics import EmployeeCommutingMetrics, get_metrics
from greenlang.employee_commuting.config import get_config
from greenlang.employee_commuting.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ==============================================================================
# ENGINE METADATA
# ==============================================================================

ENGINE_ID: str = "active_transport_calculator_engine"
ENGINE_VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION & CONSTANTS
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")
ROUNDING = ROUND_HALF_UP

# Batch processing limits
_MAX_BATCH_SIZE: int = 10000

# Round-trip multiplier
_ROUND_TRIP_MULTIPLIER = _TWO
_ONE_WAY_MULTIPLIER = _ONE


def _q(value: Decimal) -> Decimal:
    """
    Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with exactly 8 decimal places.
    """
    return value.quantize(_QUANT_8DP, rounding=ROUNDING)


# ==============================================================================
# ACTIVE TRANSPORT OPERATIONAL EMISSION FACTORS
# ==============================================================================

# Active transport operational emissions (kg CO2e per km) - Zero for human-powered
ACTIVE_OPERATIONAL_EFS: Dict[str, Decimal] = {
    "cycling": _ZERO,
    "walking": _ZERO,
}

# Active transport lifecycle emissions (kg CO2e per km) - manufacturing, maintenance, disposal
# Source: European Cyclists' Federation LCA, peer-reviewed studies
ACTIVE_LIFECYCLE_EFS: Dict[str, Dict[str, Any]] = {
    "cycling": {
        "manufacturing_per_km": Decimal("0.00500"),   # ~75 kg CO2e / 15,000 km lifecycle
        "maintenance_per_km": Decimal("0.00200"),     # tires, chain, parts
        "total_per_km": Decimal("0.00700"),
        "source": "ECF_LCA_2024",
    },
    "walking": {
        "manufacturing_per_km": Decimal("0.00300"),   # footwear lifecycle
        "maintenance_per_km": Decimal("0.00000"),
        "total_per_km": Decimal("0.00300"),
        "source": "MIT_LCA_2024",
    },
}

# ==============================================================================
# E-BIKE EMISSION FACTORS
# ==============================================================================

E_BIKE_FACTORS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "energy_kwh_per_km": Decimal("0.01100"),      # 0.011 kWh/km average
        "manufacturing_co2e": Decimal("134.00000"),    # kg CO2e total manufacturing
        "useful_life_km": Decimal("30000.00000"),      # 30,000 km useful life
        "manufacturing_per_km": Decimal("0.00447"),    # amortized
        "battery_replacement_co2e": Decimal("45.00000"),
        "battery_life_km": Decimal("15000.00000"),
        "maintenance_per_km": Decimal("0.00150"),
        "source": "DEFRA_2024_LCA",
    },
    "cargo": {
        "energy_kwh_per_km": Decimal("0.01800"),
        "manufacturing_co2e": Decimal("210.00000"),
        "useful_life_km": Decimal("25000.00000"),
        "manufacturing_per_km": Decimal("0.00840"),
        "battery_replacement_co2e": Decimal("65.00000"),
        "battery_life_km": Decimal("12000.00000"),
        "maintenance_per_km": Decimal("0.00200"),
        "source": "DEFRA_2024_LCA",
    },
    "speed_pedelec": {
        "energy_kwh_per_km": Decimal("0.01500"),
        "manufacturing_co2e": Decimal("165.00000"),
        "useful_life_km": Decimal("25000.00000"),
        "manufacturing_per_km": Decimal("0.00660"),
        "battery_replacement_co2e": Decimal("55.00000"),
        "battery_life_km": Decimal("12000.00000"),
        "maintenance_per_km": Decimal("0.00180"),
        "source": "DEFRA_2024_LCA",
    },
}

# ==============================================================================
# E-SCOOTER EMISSION FACTORS
# ==============================================================================

E_SCOOTER_FACTORS: Dict[str, Dict[str, Any]] = {
    "personal": {
        "energy_kwh_per_km": Decimal("0.02000"),
        "manufacturing_co2e": Decimal("85.00000"),
        "useful_life_km": Decimal("10000.00000"),
        "manufacturing_per_km": Decimal("0.00850"),
        "battery_replacement_co2e": Decimal("25.00000"),
        "battery_life_km": Decimal("5000.00000"),
        "maintenance_per_km": Decimal("0.00100"),
        "source": "IEA_LCA_2024",
    },
    "shared": {
        "energy_kwh_per_km": Decimal("0.02500"),
        "manufacturing_co2e": Decimal("85.00000"),
        "useful_life_km": Decimal("5000.00000"),        # shorter due to shared use
        "manufacturing_per_km": Decimal("0.01700"),
        "battery_replacement_co2e": Decimal("25.00000"),
        "battery_life_km": Decimal("2500.00000"),
        "maintenance_per_km": Decimal("0.00300"),        # higher due to shared abuse
        "source": "IEA_LCA_2024",
    },
}

# ==============================================================================
# eGRID SUBREGION FACTORS (kg CO2e per kWh) - for US sub-national resolution
# Source: EPA eGRID 2022, 26 subregions. Converted from kg/MWh to kg/kWh.
# ==============================================================================

EGRID_SUBREGION_KWH_FACTORS: Dict[str, Decimal] = {
    "AKGD": Decimal("0.46452"),
    "AKMS": Decimal("0.20538"),
    "AZNM": Decimal("0.37089"),
    "CAMX": Decimal("0.22530"),
    "ERCT": Decimal("0.38010"),
    "FRCC": Decimal("0.39244"),
    "HIMS": Decimal("0.52807"),
    "HIOA": Decimal("0.66180"),
    "MROE": Decimal("0.48233"),
    "MROW": Decimal("0.46824"),
    "NEWE": Decimal("0.21364"),
    "NWPP": Decimal("0.26585"),
    "NYCW": Decimal("0.23277"),
    "NYLI": Decimal("0.45439"),
    "NYUP": Decimal("0.11540"),
    "PRMS": Decimal("0.64933"),
    "RFCE": Decimal("0.28690"),
    "RFCM": Decimal("0.54474"),
    "RFCW": Decimal("0.46572"),
    "RMPA": Decimal("0.52845"),
    "SPNO": Decimal("0.43881"),
    "SPSO": Decimal("0.42286"),
    "SRMV": Decimal("0.34868"),
    "SRMW": Decimal("0.61428"),
    "SRSO": Decimal("0.36722"),
    "SRTV": Decimal("0.37658"),
    "SRVC": Decimal("0.28554"),
}

# ==============================================================================
# ALTERNATIVE MODE EMISSION FACTORS (for avoided emissions comparison)
# Simplified per-km factors for common car alternatives (kg CO2e per km)
# Source: DEFRA 2024 (TTW + WTT combined)
# ==============================================================================

ALTERNATIVE_MODE_EFS: Dict[str, Dict[str, Decimal]] = {
    "small_car_petrol": {
        "ef_per_km": Decimal("0.24056"),   # ef_per_vkm + wtt_per_vkm
        "source_desc": "Small petrol car, DEFRA 2024",
    },
    "medium_car_petrol": {
        "ef_per_km": Decimal("0.29668"),   # ef_per_vkm + wtt_per_vkm
        "source_desc": "Medium petrol car, DEFRA 2024",
    },
    "large_car_petrol": {
        "ef_per_km": Decimal("0.41019"),   # ef_per_vkm + wtt_per_vkm
        "source_desc": "Large petrol car, DEFRA 2024",
    },
    "medium_car_diesel": {
        "ef_per_km": Decimal("0.26579"),   # ef_per_vkm + wtt_per_vkm
        "source_desc": "Medium diesel car, DEFRA 2024",
    },
    "average_car": {
        "ef_per_km": Decimal("0.31110"),   # ef_per_vkm + wtt_per_vkm
        "source_desc": "Average car, DEFRA 2024",
    },
    "hybrid": {
        "ef_per_km": Decimal("0.20668"),   # ef_per_vkm + wtt_per_vkm
        "source_desc": "Hybrid car, DEFRA 2024",
    },
    "motorcycle": {
        "ef_per_km": Decimal("0.14204"),   # ef_per_vkm + wtt_per_vkm
        "source_desc": "Motorcycle, DEFRA 2024",
    },
    "bus_local": {
        "ef_per_km": Decimal("0.10312"),   # per pkm
        "source_desc": "Local bus, DEFRA 2024",
    },
}

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["ActiveTransportCalculatorEngine"] = None
_instance_lock: threading.Lock = threading.Lock()


# ==============================================================================
# ActiveTransportCalculatorEngine
# ==============================================================================


class ActiveTransportCalculatorEngine:
    """
    Engine 5: Active transport and micro-mobility emissions calculator.

    Calculates GHG emissions for active commuting modes (cycling, walking) and
    electric micro-mobility modes (e-bikes, e-scooters). Active human-powered
    modes have zero operational emissions with optional lifecycle (manufacturing,
    maintenance) emissions. Electric modes compute operational emissions from
    electricity consumption multiplied by grid emission factors, plus optional
    lifecycle emissions.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with peer-reviewed LCA factors. No LLM
    calls are made anywhere in the calculation pipeline.

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
        >>> engine = ActiveTransportCalculatorEngine.get_instance()
        >>> result = engine.calculate_cycling(one_way_distance_km=Decimal("5.0"))
        >>> assert result["total_co2e_kg"] >= Decimal("0")
        >>> result = engine.calculate_e_bike(
        ...     one_way_distance_km=Decimal("8.0"),
        ...     country_code="US",
        ... )
        >>> assert result["total_co2e_kg"] > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance(
        metrics: Optional[EmployeeCommutingMetrics] = None,
    ) -> "ActiveTransportCalculatorEngine":
        """
        Get or create the singleton ActiveTransportCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Args:
            metrics: Optional Prometheus metrics collector. A default
                     instance is obtained via get_metrics() if None.

        Returns:
            Singleton ActiveTransportCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = ActiveTransportCalculatorEngine(
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
        Initialise the ActiveTransportCalculatorEngine.

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
            "ActiveTransportCalculatorEngine initialised: engine=%s, version=%s, agent=%s",
            ENGINE_ID,
            ENGINE_VERSION,
            AGENT_ID,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    # 1. calculate_cycling
    # ==================================================================

    def calculate_cycling(
        self,
        one_way_distance_km: Decimal,
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
        round_trip: bool = True,
        include_lifecycle: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for pedal bicycle commuting.

        Cycling has zero operational emissions. When include_lifecycle is True,
        manufacturing and maintenance emissions are included based on European
        Cyclists' Federation LCA data.

        Formula:
            annual_distance = one_way_km x multiplier x working_days x (1 - wfh)
            operational_co2e = 0
            lifecycle_co2e = annual_distance x 0.00700  (if included)
            total_co2e = operational_co2e + lifecycle_co2e

        Args:
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of commute days per year (default 240).
            wfh_fraction: Fraction of days worked from home, 0.0 to 1.0.
            round_trip: If True, multiply distance by 2 for round trip.
            include_lifecycle: Include manufacturing/maintenance emissions.

        Returns:
            Dictionary with calculation results including:
            - mode, annual_distance_km, operational_co2e_kg, lifecycle_co2e_kg,
              total_co2e_kg, ef_source, provenance_hash, metadata.

        Raises:
            ValueError: If inputs fail validation.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> result = engine.calculate_cycling(Decimal("5.0"))
            >>> result["operational_co2e_kg"]
            Decimal('0.00000000')
        """
        start_time = time.monotonic()

        with self._lock:
            # Validate inputs
            errors = self._validate_inputs({
                "one_way_distance_km": one_way_distance_km,
                "working_days": working_days,
                "wfh_fraction": wfh_fraction,
                "mode": "cycling",
            })
            if errors:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")

            # Compute annual distance
            annual_distance = self._compute_annual_distance(
                one_way_distance_km, working_days, wfh_fraction, round_trip,
            )

            # Operational emissions: always zero for cycling
            operational_co2e = _ZERO

            # Lifecycle emissions
            lifecycle_co2e = _ZERO
            lifecycle_detail = {}
            if include_lifecycle:
                lf = ACTIVE_LIFECYCLE_EFS["cycling"]
                manufacturing_co2e = _q(annual_distance * lf["manufacturing_per_km"])
                maintenance_co2e = _q(annual_distance * lf["maintenance_per_km"])
                lifecycle_co2e = _q(manufacturing_co2e + maintenance_co2e)
                lifecycle_detail = {
                    "manufacturing_co2e_kg": str(manufacturing_co2e),
                    "maintenance_co2e_kg": str(maintenance_co2e),
                    "lifecycle_ef_source": lf["source"],
                }

            total_co2e = _q(operational_co2e + lifecycle_co2e)

            # Provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "mode": "cycling",
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "round_trip": round_trip,
                    "include_lifecycle": include_lifecycle,
                },
                {
                    "annual_distance_km": str(annual_distance),
                    "operational_co2e_kg": str(operational_co2e),
                    "lifecycle_co2e_kg": str(lifecycle_co2e),
                    "total_co2e_kg": str(total_co2e),
                },
            )

            self._calculation_count += 1
            duration = time.monotonic() - start_time

            result = {
                "mode": "cycling",
                "mode_category": "active_transport",
                "one_way_distance_km": str(one_way_distance_km),
                "annual_distance_km": str(annual_distance),
                "working_days": working_days,
                "wfh_fraction": str(wfh_fraction),
                "round_trip": round_trip,
                "include_lifecycle": include_lifecycle,
                "operational_co2e_kg": operational_co2e,
                "lifecycle_co2e_kg": lifecycle_co2e,
                "total_co2e_kg": total_co2e,
                "ef_source": "ECF_LCA_2024",
                "provenance_hash": provenance_hash,
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": round(duration * 1000, 3),
                "metadata": {
                    "zero_operational_emissions": True,
                    **lifecycle_detail,
                },
            }

            self._record_metrics("cycling", duration, total_co2e)
            logger.debug(
                "calculate_cycling completed: distance=%.2f km, co2e=%.8f kg, "
                "duration=%.3f ms",
                float(one_way_distance_km),
                float(total_co2e),
                duration * 1000,
            )

            return result

    # ==================================================================
    # 2. calculate_walking
    # ==================================================================

    def calculate_walking(
        self,
        one_way_distance_km: Decimal,
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
        round_trip: bool = True,
        include_lifecycle: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for walking commute.

        Walking has zero operational emissions. When include_lifecycle is True,
        footwear manufacturing emissions are included based on MIT LCA data.

        Formula:
            annual_distance = one_way_km x multiplier x working_days x (1 - wfh)
            operational_co2e = 0
            lifecycle_co2e = annual_distance x 0.00300  (if included)
            total_co2e = operational_co2e + lifecycle_co2e

        Args:
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of commute days per year (default 240).
            wfh_fraction: Fraction of days worked from home, 0.0 to 1.0.
            round_trip: If True, multiply distance by 2 for round trip.
            include_lifecycle: Include footwear manufacturing emissions.

        Returns:
            Dictionary with calculation results including:
            - mode, annual_distance_km, operational_co2e_kg, lifecycle_co2e_kg,
              total_co2e_kg, ef_source, provenance_hash, metadata.

        Raises:
            ValueError: If inputs fail validation.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> result = engine.calculate_walking(Decimal("2.0"))
            >>> result["operational_co2e_kg"]
            Decimal('0.00000000')
        """
        start_time = time.monotonic()

        with self._lock:
            # Validate inputs
            errors = self._validate_inputs({
                "one_way_distance_km": one_way_distance_km,
                "working_days": working_days,
                "wfh_fraction": wfh_fraction,
                "mode": "walking",
            })
            if errors:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")

            # Compute annual distance
            annual_distance = self._compute_annual_distance(
                one_way_distance_km, working_days, wfh_fraction, round_trip,
            )

            # Operational emissions: always zero for walking
            operational_co2e = _ZERO

            # Lifecycle emissions
            lifecycle_co2e = _ZERO
            lifecycle_detail = {}
            if include_lifecycle:
                lf = ACTIVE_LIFECYCLE_EFS["walking"]
                manufacturing_co2e = _q(annual_distance * lf["manufacturing_per_km"])
                maintenance_co2e = _q(annual_distance * lf["maintenance_per_km"])
                lifecycle_co2e = _q(manufacturing_co2e + maintenance_co2e)
                lifecycle_detail = {
                    "manufacturing_co2e_kg": str(manufacturing_co2e),
                    "maintenance_co2e_kg": str(maintenance_co2e),
                    "lifecycle_ef_source": lf["source"],
                }

            total_co2e = _q(operational_co2e + lifecycle_co2e)

            # Provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "mode": "walking",
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "round_trip": round_trip,
                    "include_lifecycle": include_lifecycle,
                },
                {
                    "annual_distance_km": str(annual_distance),
                    "operational_co2e_kg": str(operational_co2e),
                    "lifecycle_co2e_kg": str(lifecycle_co2e),
                    "total_co2e_kg": str(total_co2e),
                },
            )

            self._calculation_count += 1
            duration = time.monotonic() - start_time

            result = {
                "mode": "walking",
                "mode_category": "active_transport",
                "one_way_distance_km": str(one_way_distance_km),
                "annual_distance_km": str(annual_distance),
                "working_days": working_days,
                "wfh_fraction": str(wfh_fraction),
                "round_trip": round_trip,
                "include_lifecycle": include_lifecycle,
                "operational_co2e_kg": operational_co2e,
                "lifecycle_co2e_kg": lifecycle_co2e,
                "total_co2e_kg": total_co2e,
                "ef_source": "MIT_LCA_2024",
                "provenance_hash": provenance_hash,
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": round(duration * 1000, 3),
                "metadata": {
                    "zero_operational_emissions": True,
                    **lifecycle_detail,
                },
            }

            self._record_metrics("walking", duration, total_co2e)
            logger.debug(
                "calculate_walking completed: distance=%.2f km, co2e=%.8f kg, "
                "duration=%.3f ms",
                float(one_way_distance_km),
                float(total_co2e),
                duration * 1000,
            )

            return result

    # ==================================================================
    # 3. calculate_e_bike
    # ==================================================================

    def calculate_e_bike(
        self,
        one_way_distance_km: Decimal,
        e_bike_type: str = "standard",
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
        round_trip: bool = True,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        include_lifecycle: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for e-bike commuting.

        E-bikes have electricity-based operational emissions (energy consumed
        per km multiplied by grid emission factor) plus optional lifecycle
        emissions covering manufacturing, battery replacement, and maintenance.

        Formula:
            annual_distance = one_way_km x multiplier x working_days x (1 - wfh)
            energy_kwh = annual_distance x energy_kwh_per_km
            grid_factor = get_grid_factor(country_code, egrid_subregion)
            operational_co2e = energy_kwh x grid_factor
            manufacturing_co2e = annual_distance x manufacturing_per_km
            battery_co2e = annual_distance x (battery_replacement_co2e / battery_life_km)
            maintenance_co2e = annual_distance x maintenance_per_km
            lifecycle_co2e = manufacturing_co2e + battery_co2e + maintenance_co2e
            total_co2e = operational_co2e + (lifecycle_co2e if include_lifecycle else 0)

        Args:
            one_way_distance_km: One-way commute distance in kilometres.
            e_bike_type: Type of e-bike: "standard", "cargo", "speed_pedelec".
            working_days: Number of commute days per year (default 240).
            wfh_fraction: Fraction of days worked from home, 0.0 to 1.0.
            round_trip: If True, multiply distance by 2 for round trip.
            country_code: ISO 3166-1 alpha-2 country code for grid factor.
            egrid_subregion: Optional EPA eGRID subregion (US only, 26 subregions).
            include_lifecycle: Include manufacturing/battery/maintenance emissions.

        Returns:
            Dictionary with calculation results including:
            - mode, e_bike_type, annual_distance_km, energy_kwh,
              grid_factor_kg_per_kwh, operational_co2e_kg, lifecycle_co2e_kg,
              total_co2e_kg, ef_source, provenance_hash, metadata.

        Raises:
            ValueError: If inputs fail validation or e_bike_type is unknown.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> result = engine.calculate_e_bike(
            ...     one_way_distance_km=Decimal("8.0"),
            ...     e_bike_type="standard",
            ...     country_code="US",
            ... )
            >>> assert result["total_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Validate e_bike_type
            if e_bike_type not in E_BIKE_FACTORS:
                raise ValueError(
                    f"Unknown e_bike_type '{e_bike_type}'. "
                    f"Valid types: {sorted(E_BIKE_FACTORS.keys())}"
                )

            # Validate inputs
            errors = self._validate_inputs({
                "one_way_distance_km": one_way_distance_km,
                "working_days": working_days,
                "wfh_fraction": wfh_fraction,
                "mode": "e_bike",
                "country_code": country_code,
                "egrid_subregion": egrid_subregion,
            })
            if errors:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")

            factors = E_BIKE_FACTORS[e_bike_type]

            # Compute annual distance
            annual_distance = self._compute_annual_distance(
                one_way_distance_km, working_days, wfh_fraction, round_trip,
            )

            # Operational emissions (electricity)
            energy_kwh = _q(annual_distance * factors["energy_kwh_per_km"])
            grid_factor = self._get_grid_factor(country_code, egrid_subregion)
            operational_co2e = _q(energy_kwh * grid_factor)

            # Lifecycle emissions
            lifecycle_co2e = _ZERO
            manufacturing_co2e = _ZERO
            battery_co2e = _ZERO
            maintenance_co2e = _ZERO
            lifecycle_detail = {}

            if include_lifecycle:
                manufacturing_co2e = _q(
                    annual_distance * factors["manufacturing_per_km"]
                )
                battery_per_km = _q(
                    factors["battery_replacement_co2e"] / factors["battery_life_km"]
                )
                battery_co2e = _q(annual_distance * battery_per_km)
                maintenance_co2e = _q(
                    annual_distance * factors["maintenance_per_km"]
                )
                lifecycle_co2e = _q(
                    manufacturing_co2e + battery_co2e + maintenance_co2e
                )
                lifecycle_detail = {
                    "manufacturing_co2e_kg": str(manufacturing_co2e),
                    "battery_co2e_kg": str(battery_co2e),
                    "battery_per_km": str(battery_per_km),
                    "maintenance_co2e_kg": str(maintenance_co2e),
                    "lifecycle_ef_source": factors["source"],
                }

            total_co2e = _q(operational_co2e + lifecycle_co2e)

            # Provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "mode": "e_bike",
                    "e_bike_type": e_bike_type,
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "round_trip": round_trip,
                    "country_code": country_code,
                    "egrid_subregion": egrid_subregion,
                    "include_lifecycle": include_lifecycle,
                },
                {
                    "annual_distance_km": str(annual_distance),
                    "energy_kwh": str(energy_kwh),
                    "grid_factor_kg_per_kwh": str(grid_factor),
                    "operational_co2e_kg": str(operational_co2e),
                    "lifecycle_co2e_kg": str(lifecycle_co2e),
                    "total_co2e_kg": str(total_co2e),
                },
            )

            self._calculation_count += 1
            duration = time.monotonic() - start_time

            result = {
                "mode": "e_bike",
                "mode_category": "micro_mobility",
                "e_bike_type": e_bike_type,
                "one_way_distance_km": str(one_way_distance_km),
                "annual_distance_km": str(annual_distance),
                "working_days": working_days,
                "wfh_fraction": str(wfh_fraction),
                "round_trip": round_trip,
                "country_code": country_code,
                "egrid_subregion": egrid_subregion,
                "include_lifecycle": include_lifecycle,
                "energy_kwh": energy_kwh,
                "grid_factor_kg_per_kwh": grid_factor,
                "operational_co2e_kg": operational_co2e,
                "lifecycle_co2e_kg": lifecycle_co2e,
                "total_co2e_kg": total_co2e,
                "ef_source": factors["source"],
                "provenance_hash": provenance_hash,
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": round(duration * 1000, 3),
                "metadata": {
                    "energy_kwh_per_km": str(factors["energy_kwh_per_km"]),
                    "manufacturing_total_co2e": str(factors["manufacturing_co2e"]),
                    "useful_life_km": str(factors["useful_life_km"]),
                    **lifecycle_detail,
                },
            }

            self._record_metrics("e_bike", duration, total_co2e)
            logger.debug(
                "calculate_e_bike completed: type=%s, distance=%.2f km, "
                "co2e=%.8f kg, duration=%.3f ms",
                e_bike_type,
                float(one_way_distance_km),
                float(total_co2e),
                duration * 1000,
            )

            return result

    # ==================================================================
    # 4. calculate_e_scooter
    # ==================================================================

    def calculate_e_scooter(
        self,
        one_way_distance_km: Decimal,
        scooter_type: str = "personal",
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
        round_trip: bool = True,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        include_lifecycle: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for e-scooter commuting.

        E-scooters have electricity-based operational emissions plus optional
        lifecycle emissions. Shared e-scooters have higher lifecycle emissions
        per km due to shorter useful life and higher maintenance needs.

        Formula:
            annual_distance = one_way_km x multiplier x working_days x (1 - wfh)
            energy_kwh = annual_distance x energy_kwh_per_km
            grid_factor = get_grid_factor(country_code, egrid_subregion)
            operational_co2e = energy_kwh x grid_factor
            manufacturing_co2e = annual_distance x manufacturing_per_km
            battery_co2e = annual_distance x (battery_replacement_co2e / battery_life_km)
            maintenance_co2e = annual_distance x maintenance_per_km
            lifecycle_co2e = manufacturing_co2e + battery_co2e + maintenance_co2e
            total_co2e = operational_co2e + (lifecycle_co2e if include_lifecycle else 0)

        Args:
            one_way_distance_km: One-way commute distance in kilometres.
            scooter_type: Type of e-scooter: "personal" or "shared".
            working_days: Number of commute days per year (default 240).
            wfh_fraction: Fraction of days worked from home, 0.0 to 1.0.
            round_trip: If True, multiply distance by 2 for round trip.
            country_code: ISO 3166-1 alpha-2 country code for grid factor.
            egrid_subregion: Optional EPA eGRID subregion (US only, 26 subregions).
            include_lifecycle: Include manufacturing/battery/maintenance emissions.

        Returns:
            Dictionary with calculation results including:
            - mode, scooter_type, annual_distance_km, energy_kwh,
              grid_factor_kg_per_kwh, operational_co2e_kg, lifecycle_co2e_kg,
              total_co2e_kg, ef_source, provenance_hash, metadata.

        Raises:
            ValueError: If inputs fail validation or scooter_type is unknown.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> result = engine.calculate_e_scooter(
            ...     one_way_distance_km=Decimal("3.5"),
            ...     scooter_type="personal",
            ...     country_code="GB",
            ... )
            >>> assert result["total_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Validate scooter_type
            if scooter_type not in E_SCOOTER_FACTORS:
                raise ValueError(
                    f"Unknown scooter_type '{scooter_type}'. "
                    f"Valid types: {sorted(E_SCOOTER_FACTORS.keys())}"
                )

            # Validate inputs
            errors = self._validate_inputs({
                "one_way_distance_km": one_way_distance_km,
                "working_days": working_days,
                "wfh_fraction": wfh_fraction,
                "mode": "e_scooter",
                "country_code": country_code,
                "egrid_subregion": egrid_subregion,
            })
            if errors:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")

            factors = E_SCOOTER_FACTORS[scooter_type]

            # Compute annual distance
            annual_distance = self._compute_annual_distance(
                one_way_distance_km, working_days, wfh_fraction, round_trip,
            )

            # Operational emissions (electricity)
            energy_kwh = _q(annual_distance * factors["energy_kwh_per_km"])
            grid_factor = self._get_grid_factor(country_code, egrid_subregion)
            operational_co2e = _q(energy_kwh * grid_factor)

            # Lifecycle emissions
            lifecycle_co2e = _ZERO
            manufacturing_co2e = _ZERO
            battery_co2e = _ZERO
            maintenance_co2e = _ZERO
            lifecycle_detail = {}

            if include_lifecycle:
                manufacturing_co2e = _q(
                    annual_distance * factors["manufacturing_per_km"]
                )
                battery_per_km = _q(
                    factors["battery_replacement_co2e"] / factors["battery_life_km"]
                )
                battery_co2e = _q(annual_distance * battery_per_km)
                maintenance_co2e = _q(
                    annual_distance * factors["maintenance_per_km"]
                )
                lifecycle_co2e = _q(
                    manufacturing_co2e + battery_co2e + maintenance_co2e
                )
                lifecycle_detail = {
                    "manufacturing_co2e_kg": str(manufacturing_co2e),
                    "battery_co2e_kg": str(battery_co2e),
                    "battery_per_km": str(battery_per_km),
                    "maintenance_co2e_kg": str(maintenance_co2e),
                    "lifecycle_ef_source": factors["source"],
                }

            total_co2e = _q(operational_co2e + lifecycle_co2e)

            # Provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "mode": "e_scooter",
                    "scooter_type": scooter_type,
                    "one_way_distance_km": str(one_way_distance_km),
                    "working_days": working_days,
                    "wfh_fraction": str(wfh_fraction),
                    "round_trip": round_trip,
                    "country_code": country_code,
                    "egrid_subregion": egrid_subregion,
                    "include_lifecycle": include_lifecycle,
                },
                {
                    "annual_distance_km": str(annual_distance),
                    "energy_kwh": str(energy_kwh),
                    "grid_factor_kg_per_kwh": str(grid_factor),
                    "operational_co2e_kg": str(operational_co2e),
                    "lifecycle_co2e_kg": str(lifecycle_co2e),
                    "total_co2e_kg": str(total_co2e),
                },
            )

            self._calculation_count += 1
            duration = time.monotonic() - start_time

            result = {
                "mode": "e_scooter",
                "mode_category": "micro_mobility",
                "scooter_type": scooter_type,
                "one_way_distance_km": str(one_way_distance_km),
                "annual_distance_km": str(annual_distance),
                "working_days": working_days,
                "wfh_fraction": str(wfh_fraction),
                "round_trip": round_trip,
                "country_code": country_code,
                "egrid_subregion": egrid_subregion,
                "include_lifecycle": include_lifecycle,
                "energy_kwh": energy_kwh,
                "grid_factor_kg_per_kwh": grid_factor,
                "operational_co2e_kg": operational_co2e,
                "lifecycle_co2e_kg": lifecycle_co2e,
                "total_co2e_kg": total_co2e,
                "ef_source": factors["source"],
                "provenance_hash": provenance_hash,
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": round(duration * 1000, 3),
                "metadata": {
                    "energy_kwh_per_km": str(factors["energy_kwh_per_km"]),
                    "manufacturing_total_co2e": str(factors["manufacturing_co2e"]),
                    "useful_life_km": str(factors["useful_life_km"]),
                    **lifecycle_detail,
                },
            }

            self._record_metrics("e_scooter", duration, total_co2e)
            logger.debug(
                "calculate_e_scooter completed: type=%s, distance=%.2f km, "
                "co2e=%.8f kg, duration=%.3f ms",
                scooter_type,
                float(one_way_distance_km),
                float(total_co2e),
                duration * 1000,
            )

            return result

    # ==================================================================
    # 5. calculate_active_commute (universal dispatcher)
    # ==================================================================

    def calculate_active_commute(
        self,
        mode: str,
        one_way_distance_km: Decimal,
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
        round_trip: bool = True,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        include_lifecycle: bool = True,
        sub_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Universal dispatcher for active and micro-mobility mode calculations.

        Routes the request to the appropriate mode-specific calculator based
        on the mode parameter.

        Args:
            mode: Transport mode - "cycling", "walking", "e_bike", "e_scooter".
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of commute days per year (default 240).
            wfh_fraction: Fraction of days worked from home, 0.0 to 1.0.
            round_trip: If True, multiply distance by 2 for round trip.
            country_code: ISO 3166-1 alpha-2 country code for grid factor.
            egrid_subregion: Optional EPA eGRID subregion (US only).
            include_lifecycle: Include lifecycle emissions.
            sub_type: Sub-type for e-bike ("standard"/"cargo"/"speed_pedelec")
                      or e-scooter ("personal"/"shared"). Defaults to "standard"
                      for e-bike, "personal" for e-scooter.

        Returns:
            Dictionary with mode-specific calculation results.

        Raises:
            ValueError: If mode is not supported or inputs fail validation.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> result = engine.calculate_active_commute(
            ...     mode="cycling",
            ...     one_way_distance_km=Decimal("5.0"),
            ... )
            >>> assert result["mode"] == "cycling"
        """
        supported = self.get_supported_modes()
        if mode not in supported:
            raise ValueError(
                f"Unsupported mode '{mode}'. Supported modes: {supported}"
            )

        if mode == "cycling":
            return self.calculate_cycling(
                one_way_distance_km=one_way_distance_km,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                round_trip=round_trip,
                include_lifecycle=include_lifecycle,
            )

        if mode == "walking":
            return self.calculate_walking(
                one_way_distance_km=one_way_distance_km,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                round_trip=round_trip,
                include_lifecycle=include_lifecycle,
            )

        if mode == "e_bike":
            e_bike_type = sub_type or "standard"
            return self.calculate_e_bike(
                one_way_distance_km=one_way_distance_km,
                e_bike_type=e_bike_type,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                round_trip=round_trip,
                country_code=country_code,
                egrid_subregion=egrid_subregion,
                include_lifecycle=include_lifecycle,
            )

        if mode == "e_scooter":
            scooter_type = sub_type or "personal"
            return self.calculate_e_scooter(
                one_way_distance_km=one_way_distance_km,
                scooter_type=scooter_type,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                round_trip=round_trip,
                country_code=country_code,
                egrid_subregion=egrid_subregion,
                include_lifecycle=include_lifecycle,
            )

        # Defensive - should not reach here due to check above
        raise ValueError(f"Unhandled mode '{mode}'")

    # ==================================================================
    # 6. calculate_batch
    # ==================================================================

    def calculate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Batch-process multiple active transport calculations.

        Processes each item through calculate_active_commute and collects
        results, errors, and aggregate statistics.

        Args:
            items: List of dictionaries, each containing parameters for
                   calculate_active_commute. Required key: "mode". Other
                   keys map to calculate_active_commute parameters.

        Returns:
            Dictionary with:
            - results: List of successful calculation results.
            - errors: List of error dictionaries for failed calculations.
            - summary: Aggregate statistics (total_co2e, count, by_mode).
            - provenance_hash: Batch-level provenance hash.
            - processing_time_ms: Total batch processing time.

        Raises:
            ValueError: If items list is empty or exceeds _MAX_BATCH_SIZE.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> batch = [
            ...     {"mode": "cycling", "one_way_distance_km": Decimal("5.0")},
            ...     {"mode": "e_bike", "one_way_distance_km": Decimal("8.0")},
            ... ]
            >>> result = engine.calculate_batch(batch)
            >>> assert len(result["results"]) == 2
        """
        start_time = time.monotonic()

        if not items:
            raise ValueError("Batch items list cannot be empty")

        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = _ZERO
        by_mode: Dict[str, Dict[str, Any]] = {}

        for idx, item in enumerate(items):
            try:
                # Extract parameters
                mode = item.get("mode")
                if mode is None:
                    raise ValueError("Missing required key 'mode'")

                # Convert one_way_distance_km to Decimal if needed
                distance_raw = item.get("one_way_distance_km", Decimal("0"))
                if not isinstance(distance_raw, Decimal):
                    distance_raw = Decimal(str(distance_raw))

                wfh_raw = item.get("wfh_fraction", _ZERO)
                if not isinstance(wfh_raw, Decimal):
                    wfh_raw = Decimal(str(wfh_raw))

                calc_result = self.calculate_active_commute(
                    mode=mode,
                    one_way_distance_km=distance_raw,
                    working_days=item.get("working_days", 240),
                    wfh_fraction=wfh_raw,
                    round_trip=item.get("round_trip", True),
                    country_code=item.get("country_code", "US"),
                    egrid_subregion=item.get("egrid_subregion"),
                    include_lifecycle=item.get("include_lifecycle", True),
                    sub_type=item.get("sub_type"),
                )

                results.append(calc_result)
                item_co2e = calc_result["total_co2e_kg"]
                total_co2e = _q(total_co2e + item_co2e)

                # Aggregate by mode
                if mode not in by_mode:
                    by_mode[mode] = {
                        "count": 0,
                        "total_co2e_kg": _ZERO,
                    }
                by_mode[mode]["count"] += 1
                by_mode[mode]["total_co2e_kg"] = _q(
                    by_mode[mode]["total_co2e_kg"] + item_co2e
                )

            except Exception as e:
                logger.warning(
                    "Batch item %d failed: %s", idx, str(e),
                )
                errors.append({
                    "index": idx,
                    "item": {k: str(v) for k, v in item.items()},
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

        # Serialize by_mode for provenance
        by_mode_serializable = {
            k: {
                "count": v["count"],
                "total_co2e_kg": str(v["total_co2e_kg"]),
            }
            for k, v in by_mode.items()
        }

        # Batch provenance hash
        batch_provenance_hash = self._calculate_provenance_hash(
            {
                "batch_size": len(items),
                "successful": len(results),
                "failed": len(errors),
            },
            {
                "total_co2e_kg": str(total_co2e),
                "by_mode": by_mode_serializable,
            },
        )

        duration = time.monotonic() - start_time

        batch_result = {
            "results": results,
            "errors": errors,
            "summary": {
                "total_items": len(items),
                "successful": len(results),
                "failed": len(errors),
                "total_co2e_kg": total_co2e,
                "by_mode": by_mode_serializable,
            },
            "provenance_hash": batch_provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration * 1000, 3),
        }

        logger.info(
            "calculate_batch completed: items=%d, success=%d, failed=%d, "
            "total_co2e=%.8f kg, duration=%.3f ms",
            len(items),
            len(results),
            len(errors),
            float(total_co2e),
            duration * 1000,
        )

        return batch_result

    # ==================================================================
    # 7. compare_active_modes
    # ==================================================================

    def compare_active_modes(
        self,
        one_way_distance_km: Decimal,
        working_days: int = 240,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        wfh_fraction: Decimal = _ZERO,
        include_lifecycle: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare emissions across all active and micro-mobility modes.

        Runs calculations for cycling, walking, all e-bike types, and all
        e-scooter types with identical parameters and returns a ranked
        comparison.

        Args:
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of commute days per year (default 240).
            country_code: ISO 3166-1 alpha-2 country code for grid factor.
            egrid_subregion: Optional EPA eGRID subregion (US only).
            wfh_fraction: Fraction of days worked from home, 0.0 to 1.0.
            include_lifecycle: Include lifecycle emissions.

        Returns:
            Dictionary with:
            - modes: Dict mapping mode name to calculation result.
            - ranking: List of mode names sorted by total_co2e ascending.
            - lowest_co2e: Mode with the lowest total emissions.
            - highest_co2e: Mode with the highest total emissions.
            - annual_distance_km: Common annual distance for all modes.
            - provenance_hash: Comparison-level provenance hash.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> comparison = engine.compare_active_modes(
            ...     one_way_distance_km=Decimal("5.0"),
            ... )
            >>> assert comparison["lowest_co2e"]["mode"] in ["cycling", "walking"]
        """
        start_time = time.monotonic()

        modes_results: Dict[str, Dict[str, Any]] = {}

        # Active transport modes
        for active_mode in ["cycling", "walking"]:
            result = self.calculate_active_commute(
                mode=active_mode,
                one_way_distance_km=one_way_distance_km,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                include_lifecycle=include_lifecycle,
            )
            modes_results[active_mode] = result

        # E-bike types
        for e_bike_type in sorted(E_BIKE_FACTORS.keys()):
            label = f"e_bike_{e_bike_type}"
            result = self.calculate_e_bike(
                one_way_distance_km=one_way_distance_km,
                e_bike_type=e_bike_type,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                country_code=country_code,
                egrid_subregion=egrid_subregion,
                include_lifecycle=include_lifecycle,
            )
            modes_results[label] = result

        # E-scooter types
        for scooter_type in sorted(E_SCOOTER_FACTORS.keys()):
            label = f"e_scooter_{scooter_type}"
            result = self.calculate_e_scooter(
                one_way_distance_km=one_way_distance_km,
                scooter_type=scooter_type,
                working_days=working_days,
                wfh_fraction=wfh_fraction,
                country_code=country_code,
                egrid_subregion=egrid_subregion,
                include_lifecycle=include_lifecycle,
            )
            modes_results[label] = result

        # Sort by total_co2e ascending
        ranking = sorted(
            modes_results.keys(),
            key=lambda m: modes_results[m]["total_co2e_kg"],
        )

        lowest = modes_results[ranking[0]]
        highest = modes_results[ranking[-1]]

        # Provenance hash
        comparison_provenance = self._calculate_provenance_hash(
            {
                "comparison": True,
                "one_way_distance_km": str(one_way_distance_km),
                "working_days": working_days,
                "country_code": country_code,
                "modes_count": len(modes_results),
            },
            {
                "ranking": ranking,
                "lowest_mode": ranking[0],
                "lowest_co2e_kg": str(lowest["total_co2e_kg"]),
                "highest_mode": ranking[-1],
                "highest_co2e_kg": str(highest["total_co2e_kg"]),
            },
        )

        duration = time.monotonic() - start_time

        # Build concise ranking summary
        ranking_summary = [
            {
                "rank": i + 1,
                "mode": mode_name,
                "total_co2e_kg": modes_results[mode_name]["total_co2e_kg"],
                "operational_co2e_kg": modes_results[mode_name]["operational_co2e_kg"],
                "lifecycle_co2e_kg": modes_results[mode_name]["lifecycle_co2e_kg"],
            }
            for i, mode_name in enumerate(ranking)
        ]

        comparison_result = {
            "modes": modes_results,
            "ranking": ranking,
            "ranking_summary": ranking_summary,
            "lowest_co2e": {
                "mode": ranking[0],
                "total_co2e_kg": lowest["total_co2e_kg"],
            },
            "highest_co2e": {
                "mode": ranking[-1],
                "total_co2e_kg": highest["total_co2e_kg"],
            },
            "one_way_distance_km": str(one_way_distance_km),
            "annual_distance_km": lowest["annual_distance_km"],
            "working_days": working_days,
            "country_code": country_code,
            "include_lifecycle": include_lifecycle,
            "provenance_hash": comparison_provenance,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration * 1000, 3),
        }

        logger.info(
            "compare_active_modes completed: modes=%d, lowest=%s (%.8f kg), "
            "highest=%s (%.8f kg), duration=%.3f ms",
            len(modes_results),
            ranking[0],
            float(lowest["total_co2e_kg"]),
            ranking[-1],
            float(highest["total_co2e_kg"]),
            duration * 1000,
        )

        return comparison_result

    # ==================================================================
    # 8. calculate_avoided_emissions
    # ==================================================================

    def calculate_avoided_emissions(
        self,
        mode: str,
        one_way_distance_km: Decimal,
        working_days: int = 240,
        wfh_fraction: Decimal = _ZERO,
        round_trip: bool = True,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        include_lifecycle: bool = True,
        sub_type: Optional[str] = None,
        alternative_mode: str = "average_car",
        alternative_ef_per_km: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Calculate emissions avoided by using active transport vs a car alternative.

        Computes the difference between the alternative mode's emissions
        (e.g., driving a medium car) and the active transport mode's emissions
        for the same commute distance and schedule.

        Formula:
            active_co2e = calculate for chosen active mode
            alternative_co2e = annual_distance x alternative_ef_per_km
            avoided_co2e = alternative_co2e - active_co2e
            avoided_pct = (avoided_co2e / alternative_co2e) x 100

        Args:
            mode: Active transport mode ("cycling"/"walking"/"e_bike"/"e_scooter").
            one_way_distance_km: One-way commute distance in kilometres.
            working_days: Number of commute days per year (default 240).
            wfh_fraction: Fraction of days worked from home.
            round_trip: If True, multiply distance by 2 for round trip.
            country_code: ISO 3166-1 alpha-2 country code for grid factor.
            egrid_subregion: Optional EPA eGRID subregion (US only).
            include_lifecycle: Include lifecycle emissions for active mode.
            sub_type: Sub-type for e-bike or e-scooter.
            alternative_mode: Alternative mode for comparison. Options:
                "small_car_petrol", "medium_car_petrol", "large_car_petrol",
                "medium_car_diesel", "average_car", "hybrid", "motorcycle",
                "bus_local". Default: "average_car".
            alternative_ef_per_km: Optional custom emission factor for the
                alternative mode (kg CO2e per km). Overrides built-in factors.

        Returns:
            Dictionary with:
            - active_mode_result: Full result from active mode calculation.
            - alternative_mode: Name of alternative mode.
            - alternative_ef_per_km: Emission factor used for alternative.
            - alternative_co2e_kg: Total alternative mode emissions.
            - active_co2e_kg: Total active mode emissions.
            - avoided_co2e_kg: Emissions avoided (positive = savings).
            - avoided_percentage: Percentage reduction.
            - provenance_hash: Result provenance hash.

        Raises:
            ValueError: If alternative_mode is unknown and no custom EF provided.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> result = engine.calculate_avoided_emissions(
            ...     mode="cycling",
            ...     one_way_distance_km=Decimal("10.0"),
            ...     alternative_mode="average_car",
            ... )
            >>> assert result["avoided_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        # Resolve alternative emission factor
        if alternative_ef_per_km is not None:
            if not isinstance(alternative_ef_per_km, Decimal):
                alternative_ef_per_km = Decimal(str(alternative_ef_per_km))
            if alternative_ef_per_km < _ZERO:
                raise ValueError(
                    "alternative_ef_per_km must be non-negative"
                )
            alt_ef = alternative_ef_per_km
            alt_source = "custom"
        else:
            if alternative_mode not in ALTERNATIVE_MODE_EFS:
                raise ValueError(
                    f"Unknown alternative_mode '{alternative_mode}'. "
                    f"Valid: {sorted(ALTERNATIVE_MODE_EFS.keys())}. "
                    f"Or provide alternative_ef_per_km."
                )
            alt_data = ALTERNATIVE_MODE_EFS[alternative_mode]
            alt_ef = alt_data["ef_per_km"]
            alt_source = alt_data["source_desc"]

        # Calculate active mode emissions
        active_result = self.calculate_active_commute(
            mode=mode,
            one_way_distance_km=one_way_distance_km,
            working_days=working_days,
            wfh_fraction=wfh_fraction,
            round_trip=round_trip,
            country_code=country_code,
            egrid_subregion=egrid_subregion,
            include_lifecycle=include_lifecycle,
            sub_type=sub_type,
        )

        active_co2e = active_result["total_co2e_kg"]

        # Calculate alternative mode emissions using same annual distance
        annual_distance = Decimal(active_result["annual_distance_km"])
        alternative_co2e = _q(annual_distance * alt_ef)

        # Avoided emissions
        avoided_co2e = _q(alternative_co2e - active_co2e)

        # Avoided percentage (handle division by zero if alternative is zero)
        if alternative_co2e > _ZERO:
            avoided_pct = _q((avoided_co2e / alternative_co2e) * _HUNDRED)
        else:
            avoided_pct = _ZERO

        # Provenance hash
        provenance_hash = self._calculate_provenance_hash(
            {
                "mode": mode,
                "alternative_mode": alternative_mode,
                "one_way_distance_km": str(one_way_distance_km),
                "alternative_ef_per_km": str(alt_ef),
            },
            {
                "active_co2e_kg": str(active_co2e),
                "alternative_co2e_kg": str(alternative_co2e),
                "avoided_co2e_kg": str(avoided_co2e),
                "avoided_percentage": str(avoided_pct),
            },
        )

        duration = time.monotonic() - start_time

        result = {
            "active_mode": mode,
            "active_mode_result": active_result,
            "alternative_mode": alternative_mode,
            "alternative_ef_per_km": alt_ef,
            "alternative_ef_source": alt_source,
            "annual_distance_km": str(annual_distance),
            "active_co2e_kg": active_co2e,
            "alternative_co2e_kg": alternative_co2e,
            "avoided_co2e_kg": avoided_co2e,
            "avoided_percentage": avoided_pct,
            "provenance_hash": provenance_hash,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(duration * 1000, 3),
        }

        logger.info(
            "calculate_avoided_emissions: mode=%s vs %s, avoided=%.8f kg (%.2f%%), "
            "duration=%.3f ms",
            mode,
            alternative_mode,
            float(avoided_co2e),
            float(avoided_pct),
            duration * 1000,
        )

        return result

    # ==================================================================
    # HELPER: _compute_annual_distance
    # ==================================================================

    def _compute_annual_distance(
        self,
        one_way_distance_km: Decimal,
        working_days: int,
        wfh_fraction: Decimal,
        round_trip: bool,
    ) -> Decimal:
        """
        Compute annual commute distance from input parameters.

        Formula:
            multiplier = 2 if round_trip else 1
            annual_distance = one_way_km x multiplier x working_days x (1 - wfh)

        Args:
            one_way_distance_km: One-way commute distance in km.
            working_days: Number of commute days per year.
            wfh_fraction: Fraction of working days spent at home.
            round_trip: Whether to double the distance for round trip.

        Returns:
            Annual commute distance in km, quantized to 8 decimal places.
        """
        multiplier = _ROUND_TRIP_MULTIPLIER if round_trip else _ONE_WAY_MULTIPLIER
        office_fraction = _q(_ONE - wfh_fraction)
        working_days_dec = Decimal(str(working_days))

        annual_distance = _q(
            one_way_distance_km * multiplier * working_days_dec * office_fraction
        )

        return annual_distance

    # ==================================================================
    # HELPER: _get_grid_factor
    # ==================================================================

    def _get_grid_factor(
        self,
        country_code: str,
        egrid_subregion: Optional[str] = None,
    ) -> Decimal:
        """
        Resolve grid emission factor for electricity consumption.

        Priority order:
        1. eGRID subregion factor (if US and subregion provided)
        2. Country-level factor from GRID_EMISSION_FACTORS via RegionCode
        3. Global default (0.43600 kg CO2e/kWh)

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            egrid_subregion: Optional EPA eGRID subregion (US only).

        Returns:
            Grid emission factor in kg CO2e per kWh.
        """
        # If US and eGRID subregion specified, use subregion factor
        if country_code == "US" and egrid_subregion is not None:
            subregion_upper = egrid_subregion.upper()
            if subregion_upper in EGRID_SUBREGION_KWH_FACTORS:
                return EGRID_SUBREGION_KWH_FACTORS[subregion_upper]
            else:
                logger.warning(
                    "Unknown eGRID subregion '%s', falling back to US national "
                    "average. Valid subregions: %s",
                    egrid_subregion,
                    sorted(EGRID_SUBREGION_KWH_FACTORS.keys()),
                )

        # Map country code to RegionCode enum
        try:
            region = RegionCode(country_code)
        except ValueError:
            logger.warning(
                "Unknown country_code '%s', using GLOBAL grid factor",
                country_code,
            )
            region = RegionCode.GLOBAL

        return GRID_EMISSION_FACTORS.get(
            region, GRID_EMISSION_FACTORS[RegionCode.GLOBAL]
        )

    # ==================================================================
    # HELPER: _calculate_provenance_hash
    # ==================================================================

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Creates a deterministic hash from input parameters and calculation
        results, including engine identity for traceability.

        Args:
            input_data: Dictionary of input parameters.
            result_data: Dictionary of calculation results.

        Returns:
            Hexadecimal SHA-256 hash string (64 characters).
        """
        hash_payload = json.dumps(
            {
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
                "input": input_data,
                "result": result_data,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

    # ==================================================================
    # HELPER: _validate_inputs
    # ==================================================================

    def _validate_inputs(
        self,
        params: Dict[str, Any],
    ) -> List[str]:
        """
        Validate calculation input parameters.

        Checks:
        - one_way_distance_km > 0 and <= 100 km (active transport limit)
        - working_days >= 1 and <= 366
        - wfh_fraction >= 0 and < 1
        - country_code is a non-empty string
        - egrid_subregion is valid if provided

        Args:
            params: Dictionary of parameters to validate.

        Returns:
            List of error messages. Empty list means all valid.
        """
        errors: List[str] = []

        # Distance validation
        distance = params.get("one_way_distance_km")
        if distance is not None:
            if not isinstance(distance, Decimal):
                try:
                    distance = Decimal(str(distance))
                except (InvalidOperation, TypeError, ValueError):
                    errors.append(
                        f"one_way_distance_km must be a valid number, "
                        f"got '{distance}'"
                    )
                    return errors

            if distance <= _ZERO:
                errors.append(
                    "one_way_distance_km must be positive, "
                    f"got {distance}"
                )
            elif distance > Decimal("100"):
                mode = params.get("mode", "unknown")
                errors.append(
                    f"one_way_distance_km={distance} exceeds 100 km limit "
                    f"for active transport mode '{mode}'. "
                    f"Consider a different transport mode."
                )

        # Working days validation
        working_days = params.get("working_days")
        if working_days is not None:
            if not isinstance(working_days, int):
                errors.append(
                    f"working_days must be an integer, got {type(working_days).__name__}"
                )
            elif working_days < 1 or working_days > 366:
                errors.append(
                    f"working_days must be between 1 and 366, got {working_days}"
                )

        # WFH fraction validation
        wfh_fraction = params.get("wfh_fraction")
        if wfh_fraction is not None:
            if not isinstance(wfh_fraction, Decimal):
                try:
                    wfh_fraction = Decimal(str(wfh_fraction))
                except (InvalidOperation, TypeError, ValueError):
                    errors.append(
                        f"wfh_fraction must be a valid number, got '{wfh_fraction}'"
                    )
                    return errors

            if wfh_fraction < _ZERO or wfh_fraction >= _ONE:
                errors.append(
                    f"wfh_fraction must be >= 0 and < 1, got {wfh_fraction}"
                )

        # Country code validation
        country_code = params.get("country_code")
        if country_code is not None:
            if not isinstance(country_code, str) or not country_code.strip():
                errors.append(
                    f"country_code must be a non-empty string, got '{country_code}'"
                )

        # eGRID subregion validation
        egrid_subregion = params.get("egrid_subregion")
        if egrid_subregion is not None:
            subregion_upper = egrid_subregion.upper()
            if subregion_upper not in EGRID_SUBREGION_KWH_FACTORS:
                # Warning only - fallback to country level
                logger.warning(
                    "eGRID subregion '%s' not recognized; will fall back to "
                    "country-level grid factor",
                    egrid_subregion,
                )

        return errors

    # ==================================================================
    # HELPER: _record_metrics
    # ==================================================================

    def _record_metrics(
        self,
        mode: str,
        duration: float,
        co2e: Decimal,
    ) -> None:
        """
        Record Prometheus metrics for a calculation.

        Args:
            mode: Transport mode name.
            duration: Calculation duration in seconds.
            co2e: Total CO2e in kg.
        """
        try:
            self._metrics.record_calculation(
                mode=mode,
                method="active_transport",
                status="success",
                duration=duration,
                co2e=float(co2e),
            )
        except Exception as e:
            # Metrics recording must never break calculation
            logger.debug("Metrics recording failed (non-fatal): %s", str(e))

    # ==================================================================
    # INFO: get_supported_modes
    # ==================================================================

    def get_supported_modes(self) -> List[str]:
        """
        Return list of all supported active and micro-mobility modes.

        Returns:
            Sorted list of mode identifiers:
            ["cycling", "e_bike", "e_scooter", "walking"]
        """
        return sorted(["cycling", "walking", "e_bike", "e_scooter"])

    # ==================================================================
    # INFO: get_mode_details
    # ==================================================================

    def get_mode_details(self, mode: str) -> Dict[str, Any]:
        """
        Return detailed information about a supported mode.

        Args:
            mode: Transport mode identifier.

        Returns:
            Dictionary with mode name, category, sub_types, emission factor
            sources, and configuration details.

        Raises:
            ValueError: If mode is not supported.

        Example:
            >>> engine = ActiveTransportCalculatorEngine.get_instance()
            >>> details = engine.get_mode_details("e_bike")
            >>> assert details["sub_types"] == ["cargo", "speed_pedelec", "standard"]
        """
        if mode not in self.get_supported_modes():
            raise ValueError(
                f"Unsupported mode '{mode}'. "
                f"Supported: {self.get_supported_modes()}"
            )

        if mode == "cycling":
            return {
                "mode": "cycling",
                "category": "active_transport",
                "description": "Pedal bicycle commuting. Zero operational emissions.",
                "sub_types": [],
                "operational_ef_per_km": str(ACTIVE_OPERATIONAL_EFS["cycling"]),
                "lifecycle_ef_per_km": str(ACTIVE_LIFECYCLE_EFS["cycling"]["total_per_km"]),
                "lifecycle_source": ACTIVE_LIFECYCLE_EFS["cycling"]["source"],
                "has_operational_emissions": False,
                "has_lifecycle_emissions": True,
                "requires_grid_factor": False,
                "max_typical_distance_km": "20",
            }

        if mode == "walking":
            return {
                "mode": "walking",
                "category": "active_transport",
                "description": "Walking commute. Zero operational emissions.",
                "sub_types": [],
                "operational_ef_per_km": str(ACTIVE_OPERATIONAL_EFS["walking"]),
                "lifecycle_ef_per_km": str(ACTIVE_LIFECYCLE_EFS["walking"]["total_per_km"]),
                "lifecycle_source": ACTIVE_LIFECYCLE_EFS["walking"]["source"],
                "has_operational_emissions": False,
                "has_lifecycle_emissions": True,
                "requires_grid_factor": False,
                "max_typical_distance_km": "5",
            }

        if mode == "e_bike":
            sub_types = sorted(E_BIKE_FACTORS.keys())
            return {
                "mode": "e_bike",
                "category": "micro_mobility",
                "description": (
                    "Electric bicycle (pedelec) commuting. Electricity-based "
                    "operational emissions plus lifecycle manufacturing."
                ),
                "sub_types": sub_types,
                "factors_by_sub_type": {
                    st: {
                        "energy_kwh_per_km": str(E_BIKE_FACTORS[st]["energy_kwh_per_km"]),
                        "manufacturing_per_km": str(E_BIKE_FACTORS[st]["manufacturing_per_km"]),
                        "useful_life_km": str(E_BIKE_FACTORS[st]["useful_life_km"]),
                        "source": E_BIKE_FACTORS[st]["source"],
                    }
                    for st in sub_types
                },
                "has_operational_emissions": True,
                "has_lifecycle_emissions": True,
                "requires_grid_factor": True,
                "max_typical_distance_km": "30",
            }

        if mode == "e_scooter":
            sub_types = sorted(E_SCOOTER_FACTORS.keys())
            return {
                "mode": "e_scooter",
                "category": "micro_mobility",
                "description": (
                    "Electric kick-scooter commuting. Electricity-based "
                    "operational emissions plus lifecycle manufacturing."
                ),
                "sub_types": sub_types,
                "factors_by_sub_type": {
                    st: {
                        "energy_kwh_per_km": str(E_SCOOTER_FACTORS[st]["energy_kwh_per_km"]),
                        "manufacturing_per_km": str(E_SCOOTER_FACTORS[st]["manufacturing_per_km"]),
                        "useful_life_km": str(E_SCOOTER_FACTORS[st]["useful_life_km"]),
                        "source": E_SCOOTER_FACTORS[st]["source"],
                    }
                    for st in sub_types
                },
                "has_operational_emissions": True,
                "has_lifecycle_emissions": True,
                "requires_grid_factor": True,
                "max_typical_distance_km": "15",
            }

        # Defensive - should not reach here
        raise ValueError(f"Unhandled mode '{mode}'")

    # ==================================================================
    # INFO: get_engine_info
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Return engine metadata and configuration summary.

        Returns:
            Dictionary with engine_id, version, supported modes, emission
            factor sources, calculation count, and configuration state.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "agent_version": VERSION,
            "supported_modes": self.get_supported_modes(),
            "e_bike_types": sorted(E_BIKE_FACTORS.keys()),
            "e_scooter_types": sorted(E_SCOOTER_FACTORS.keys()),
            "egrid_subregions": sorted(EGRID_SUBREGION_KWH_FACTORS.keys()),
            "alternative_modes": sorted(ALTERNATIVE_MODE_EFS.keys()),
            "calculation_count": self._calculation_count,
            "decimal_precision": "8 decimal places, ROUND_HALF_UP",
            "ef_sources": [
                "ECF_LCA_2024 (cycling lifecycle)",
                "MIT_LCA_2024 (walking lifecycle)",
                "DEFRA_2024_LCA (e-bike lifecycle)",
                "IEA_LCA_2024 (e-scooter lifecycle)",
                "GRID_EMISSION_FACTORS (11 regions, IEA 2024)",
                "EPA eGRID 2022 (26 US subregions)",
                "DEFRA 2024 (alternative mode comparisons)",
            ],
            "zero_hallucination": True,
            "thread_safe": True,
        }

    # ==================================================================
    # INFO: get_egrid_subregions
    # ==================================================================

    def get_egrid_subregions(self) -> List[str]:
        """
        Return list of all supported EPA eGRID subregion codes.

        Returns:
            Sorted list of 26 eGRID subregion codes (e.g., "CAMX", "ERCT").
        """
        return sorted(EGRID_SUBREGION_KWH_FACTORS.keys())

    # ==================================================================
    # INFO: get_alternative_modes
    # ==================================================================

    def get_alternative_modes(self) -> Dict[str, Dict[str, str]]:
        """
        Return all available alternative modes for avoided-emissions comparison.

        Returns:
            Dictionary mapping mode name to emission factor and source description.
        """
        return {
            mode: {
                "ef_per_km": str(data["ef_per_km"]),
                "source_desc": data["source_desc"],
            }
            for mode, data in sorted(ALTERNATIVE_MODE_EFS.items())
        }


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def get_active_transport_calculator() -> ActiveTransportCalculatorEngine:
    """
    Get the singleton ActiveTransportCalculatorEngine instance.

    Convenience function that delegates to
    ActiveTransportCalculatorEngine.get_instance().

    Returns:
        Singleton ActiveTransportCalculatorEngine instance.

    Example:
        >>> engine = get_active_transport_calculator()
        >>> result = engine.calculate_cycling(Decimal("5.0"))
    """
    return ActiveTransportCalculatorEngine.get_instance()


def reset_active_transport_calculator() -> None:
    """
    Reset the singleton ActiveTransportCalculatorEngine instance (testing only).

    Convenience function that delegates to
    ActiveTransportCalculatorEngine.reset_instance().

    This function is intended exclusively for unit tests that need
    a fresh engine instance. It should never be called in production.
    """
    ActiveTransportCalculatorEngine.reset_instance()
