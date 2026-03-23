# -*- coding: utf-8 -*-
"""
EquipmentCalculatorEngine - Engine 4: Upstream Leased Assets Agent (AGENT-MRV-021)

Core calculation engine for leased equipment emissions covering manufacturing,
construction, generator, agricultural, mining, and HVAC equipment. Supports
energy-based, fuel-based, output-based, average-data, batch, and annual
estimation methods.

This engine implements deterministic Decimal-based emissions calculations
for all leased equipment categories, following DEFRA 2024 / EPA 2024
emission factors and the GHG Protocol Scope 3 Category 8 methodology.

Primary Formulae:
    Energy-Based (grid electricity):
        co2e = power_kw x operating_hours x load_factor x count x grid_ef

    Fuel-Based (per-litre):
        total_litres = fuel_litres x count
        ttw_co2e     = total_litres x ef_per_litre
        wtt_co2e     = total_litres x wtt_per_litre
        co2e         = ttw_co2e + wtt_co2e

    Output-Based (generator efficiency):
        fuel_kwh = kwh_generated / efficiency
        co2e     = fuel_kwh x fuel_ef_per_kwh x count

    Average-Data (benchmarks):
        Uses default hours, load factor, fuel consumption from industry
        benchmarks when metered data is unavailable.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Emission factors sourced from DEFRA 2024 / EPA 2024

Supports:
    - 6 equipment types (manufacturing, construction, generator,
      agricultural, mining, hvac)
    - 5 fuel types (diesel, petrol, lpg, cng, natural_gas)
    - Load factor adjustment by equipment type
    - Grid-based electricity calculations for electric equipment
    - Output-based generator efficiency calculations
    - Average-data benchmarks for screening-level assessments
    - Batch processing for multiple equipment calculations
    - Quick annual emission estimation
    - Input validation with detailed error messages
    - SHA-256 provenance hash integration for audit trails

Example:
    >>> from greenlang.agents.mrv.upstream_leased_assets.equipment_calculator import (
    ...     get_equipment_calculator,
    ... )
    >>> from decimal import Decimal
    >>> engine = get_equipment_calculator()
    >>> result = engine.calculate_energy_based(
    ...     equipment_type="manufacturing",
    ...     power_kw=Decimal("75"),
    ...     operating_hours=Decimal("4000"),
    ...     load_factor=Decimal("0.65"),
    ...     count=3,
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

ENGINE_ID: str = "equipment_calculator_engine"
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
ROUNDING = ROUND_HALF_UP

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# Energy conversion: 1 litre diesel = 10.21 kWh (LHV)
_DIESEL_KWH_PER_LITRE = Decimal("10.21")
_PETROL_KWH_PER_LITRE = Decimal("9.10")
_LPG_KWH_PER_LITRE = Decimal("6.98")
_CNG_KWH_PER_KG = Decimal("13.10")
_NATURAL_GAS_KWH_PER_M3 = Decimal("10.55")

# ==============================================================================
# EQUIPMENT BENCHMARKS
# Default operating parameters when metered data is unavailable
# Source: Industry benchmarks, DEFRA 2024, EPA NONROAD model
# ==============================================================================

EQUIPMENT_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "label": "Manufacturing Equipment",
        "default_hours": Decimal("4000"),
        "default_load_factor": Decimal("0.65"),
        "primary_fuel": "diesel",
        "fuel_consumption_lph": Decimal("8.5"),
        "typical_power_kw": Decimal("75"),
        "description": "General manufacturing machinery (CNC, presses, lathes)",
    },
    "construction": {
        "label": "Construction Equipment",
        "default_hours": Decimal("1500"),
        "default_load_factor": Decimal("0.55"),
        "primary_fuel": "diesel",
        "fuel_consumption_lph": Decimal("12.0"),
        "typical_power_kw": Decimal("130"),
        "description": "Excavators, loaders, dozers, cranes",
    },
    "generator": {
        "label": "Diesel Generator",
        "default_hours": Decimal("500"),
        "default_load_factor": Decimal("0.70"),
        "primary_fuel": "diesel",
        "fuel_consumption_lph": Decimal("15.0"),
        "typical_power_kw": Decimal("200"),
        "description": "Standby and prime power diesel generators",
    },
    "agricultural": {
        "label": "Agricultural Equipment",
        "default_hours": Decimal("1200"),
        "default_load_factor": Decimal("0.60"),
        "primary_fuel": "diesel",
        "fuel_consumption_lph": Decimal("6.0"),
        "typical_power_kw": Decimal("85"),
        "description": "Tractors, harvesters, irrigation pumps",
    },
    "mining": {
        "label": "Mining Equipment",
        "default_hours": Decimal("3000"),
        "default_load_factor": Decimal("0.70"),
        "primary_fuel": "diesel",
        "fuel_consumption_lph": Decimal("20.0"),
        "typical_power_kw": Decimal("250"),
        "description": "Haul trucks, drilling rigs, crushers",
    },
    "hvac": {
        "label": "HVAC Equipment",
        "default_hours": Decimal("4380"),
        "default_load_factor": Decimal("0.50"),
        "primary_fuel": "electricity",
        "fuel_consumption_lph": Decimal("0"),
        "typical_power_kw": Decimal("50"),
        "description": "Heating, ventilation, and air conditioning systems",
    },
}

# ==============================================================================
# VALID EQUIPMENT TYPES
# ==============================================================================

VALID_EQUIPMENT_TYPES: List[str] = [
    "manufacturing",
    "construction",
    "generator",
    "agricultural",
    "mining",
    "hvac",
]

# ==============================================================================
# FUEL EMISSION FACTORS (kgCO2e per litre) - DEFRA 2024
# ==============================================================================

FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "diesel": {
        "ef_per_litre": Decimal("2.68787"),
        "wtt_per_litre": Decimal("0.60927"),
        "ef_per_kwh": Decimal("0.26326"),
    },
    "petrol": {
        "ef_per_litre": Decimal("2.31484"),
        "wtt_per_litre": Decimal("0.58549"),
        "ef_per_kwh": Decimal("0.25438"),
    },
    "lpg": {
        "ef_per_litre": Decimal("1.55363"),
        "wtt_per_litre": Decimal("0.32149"),
        "ef_per_kwh": Decimal("0.22258"),
    },
    "cng": {
        "ef_per_litre": Decimal("2.02130"),
        "wtt_per_litre": Decimal("0.46490"),
        "ef_per_kwh": Decimal("0.15430"),
    },
    "natural_gas": {
        "ef_per_litre": Decimal("2.02130"),
        "wtt_per_litre": Decimal("0.46490"),
        "ef_per_kwh": Decimal("0.18400"),
    },
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
# DQI SCORE BY METHOD
# GHG Protocol data quality indicators (1=best, 5=worst)
# ==============================================================================

DQI_SCORES: Dict[str, Decimal] = {
    "energy_based": Decimal("2.0"),
    "fuel_based": Decimal("1.5"),
    "output_based": Decimal("2.5"),
    "average_data": Decimal("3.5"),
    "estimate": Decimal("4.0"),
}

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["EquipmentCalculatorEngine"] = None
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

def get_equipment_calculator() -> "EquipmentCalculatorEngine":
    """
    Get the singleton EquipmentCalculatorEngine instance.

    Returns:
        EquipmentCalculatorEngine singleton.
    """
    return EquipmentCalculatorEngine.get_instance()


def reset_equipment_calculator() -> None:
    """
    Reset the singleton EquipmentCalculatorEngine instance (testing only).
    """
    EquipmentCalculatorEngine.reset_instance()


# ==============================================================================
# EquipmentCalculatorEngine
# ==============================================================================


class EquipmentCalculatorEngine:
    """
    Engine 4: Equipment emissions calculator for upstream leased assets.

    Implements deterministic emissions calculations for leased equipment
    including manufacturing machinery, construction equipment, generators,
    agricultural equipment, mining equipment, and HVAC systems using DEFRA
    2024 / EPA 2024 emission factors aligned with GHG Protocol Scope 3
    Category 8 methodology.

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
        >>> engine = get_equipment_calculator()
        >>> result = engine.calculate_energy_based(
        ...     equipment_type="manufacturing",
        ...     power_kw=Decimal("75"),
        ...     operating_hours=Decimal("4000"),
        ...     load_factor=Decimal("0.65"),
        ...     count=3,
        ... )
        >>> assert result["co2e_kg"] > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance() -> "EquipmentCalculatorEngine":
        """
        Get or create the singleton EquipmentCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Returns:
            Singleton EquipmentCalculatorEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = EquipmentCalculatorEngine()
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
        """Initialize the EquipmentCalculatorEngine."""
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0
        self._initialized_at: str = datetime.now(timezone.utc).isoformat()

        logger.info(
            "EquipmentCalculatorEngine initialized: "
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

    def _validate_equipment_type(self, equipment_type: str) -> str:
        """
        Validate and normalize equipment type string.

        Args:
            equipment_type: Equipment type key to validate.

        Returns:
            Normalized equipment type string (lowercase, stripped).

        Raises:
            ValueError: If equipment_type is not recognized.
        """
        normalized = equipment_type.lower().strip()
        if normalized not in EQUIPMENT_BENCHMARKS:
            raise ValueError(
                f"Unknown equipment_type '{equipment_type}'. "
                f"Available: {VALID_EQUIPMENT_TYPES}"
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
        Validate equipment count is a positive integer.

        Args:
            count: Number of equipment units.

        Returns:
            Validated count.

        Raises:
            ValueError: If count is not a positive integer.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError(
                f"Equipment count must be a positive integer, got {count}"
            )
        return count

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

    def _validate_load_factor(self, load_factor: Any) -> Decimal:
        """
        Validate load factor is between 0 (exclusive) and 1 (inclusive).

        Args:
            load_factor: Load factor value to validate.

        Returns:
            Validated Decimal load factor.

        Raises:
            ValueError: If load_factor is out of range.
        """
        lf = _safe_decimal(load_factor)
        if lf <= _ZERO or lf > _ONE:
            raise ValueError(
                f"Load factor must be in range (0.0, 1.0], got {lf}"
            )
        return lf

    def _validate_efficiency(self, efficiency: Any) -> Decimal:
        """
        Validate generator efficiency is between 0 (exclusive) and 1 (exclusive).

        Args:
            efficiency: Efficiency value to validate.

        Returns:
            Validated Decimal efficiency.

        Raises:
            ValueError: If efficiency is out of range.
        """
        eff = _safe_decimal(efficiency)
        if eff <= _ZERO or eff >= _ONE:
            raise ValueError(
                f"Efficiency must be in range (0.0, 1.0), got {eff}"
            )
        return eff

    def _validate_fuel_type(self, fuel_type: str) -> str:
        """
        Validate and normalize fuel type string.

        Args:
            fuel_type: Fuel type key to validate.

        Returns:
            Normalized fuel type string (lowercase, stripped).

        Raises:
            ValueError: If fuel_type is not recognized.
        """
        normalized = fuel_type.lower().strip()
        if normalized not in FUEL_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown fuel_type '{fuel_type}'. "
                f"Available: {list(FUEL_EMISSION_FACTORS.keys())}"
            )
        return normalized

    # ==================================================================
    # 1. calculate_energy_based
    # ==================================================================

    def calculate_energy_based(
        self,
        equipment_type: str,
        power_kw: Union[Decimal, int, float, str],
        operating_hours: Union[Decimal, int, float, str],
        load_factor: Optional[Union[Decimal, int, float, str]] = None,
        country_code: str = "US",
        count: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate equipment emissions using energy-based method.

        Computes emissions from the electrical energy consumption of leased
        equipment using power rating, operating hours, load factor, and
        the grid emission factor for the specified country.

        Formula:
            effective_load = load_factor or BENCHMARKS[equipment_type].default_load_factor
            energy_kwh     = power_kw x operating_hours x effective_load x count
            co2e_kg        = energy_kwh x grid_ef(country_code)

        Args:
            equipment_type: Equipment category key (e.g., "manufacturing", "hvac").
            power_kw: Rated power in kilowatts per unit.
            operating_hours: Annual operating hours per unit.
            load_factor: Average load factor (0-1]. If None, uses benchmark default.
            country_code: ISO country code for grid EF (default "US").
            count: Number of equipment units (default 1).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            energy_kwh, power_kw, operating_hours, load_factor,
            grid_ef, method, equipment_type, count, country_code,
            ef_source, dqi_score, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If any input parameter is invalid.

        Example:
            >>> result = engine.calculate_energy_based(
            ...     equipment_type="manufacturing",
            ...     power_kw=Decimal("75"),
            ...     operating_hours=Decimal("4000"),
            ...     load_factor=Decimal("0.65"),
            ...     count=3,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            eq_type = self._validate_equipment_type(equipment_type)
            pwr = self._validate_positive_decimal(power_kw, "power_kw")
            hours = self._validate_positive_decimal(operating_hours, "operating_hours")
            c_code = self._validate_country_code(country_code)
            cnt = self._validate_count(count)

            # Resolve load factor
            if load_factor is not None:
                lf = self._validate_load_factor(load_factor)
            else:
                lf = EQUIPMENT_BENCHMARKS[eq_type]["default_load_factor"]

            # Step 2: Resolve grid emission factor (ZERO HALLUCINATION)
            grid_ef = GRID_EMISSION_FACTORS[c_code]

            # Step 3: Calculate energy consumption and emissions (Decimal only)
            count_dec = _safe_decimal(cnt)
            energy_kwh = _q(pwr * hours * lf * count_dec)
            co2e_kg = _q(energy_kwh * grid_ef)

            # Energy-based grid emissions: all upstream (no tailpipe)
            ttw_co2e = _ZERO
            wtt_co2e = co2e_kg

            # Step 4: Build provenance hash
            input_data = {
                "equipment_type": eq_type,
                "power_kw": str(pwr),
                "operating_hours": str(hours),
                "load_factor": str(lf),
                "country_code": c_code,
                "count": cnt,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "energy_kwh": str(energy_kwh),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "energy_kwh": energy_kwh,
                "power_kw": pwr,
                "operating_hours": hours,
                "load_factor": lf,
                "grid_ef": grid_ef,
                "method": "energy_based",
                "equipment_type": eq_type,
                "count": cnt,
                "country_code": c_code,
                "ef_source": "IEA_2024",
                "dqi_score": DQI_SCORES["energy_based"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Energy-based calculation complete: type=%s, power=%s kW, "
                "hours=%s, lf=%s, count=%d, kWh=%s, co2e=%s kg, "
                "duration=%.4fs",
                eq_type, pwr, hours, lf, cnt,
                energy_kwh, co2e_kg, duration,
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
        Calculate equipment emissions using fuel-based method.

        Uses fuel consumption volume and per-litre emission factors for
        scenarios where fuel records are available (e.g., diesel generators,
        construction equipment fuel logs).

        Formula:
            total_litres = fuel_litres x count
            ttw_co2e     = total_litres x ef_per_litre
            wtt_co2e     = total_litres x wtt_per_litre  (if include_wtt)
            co2e_kg      = ttw_co2e + wtt_co2e

        Args:
            fuel_type: Fuel type key (diesel, petrol, lpg, cng, natural_gas).
            fuel_litres: Total fuel consumed in litres per unit.
            count: Number of equipment units consuming this fuel (default 1).
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
            ...     fuel_litres=Decimal("5000"),
            ...     count=2,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            f_type = self._validate_fuel_type(fuel_type)
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
                "count=%d, co2e=%s kg, ttw=%s kg, wtt=%s kg, "
                "duration=%.4fs",
                f_type, litres, cnt, co2e_kg,
                ttw_co2e, wtt_co2e, duration,
            )

            return result

    # ==================================================================
    # 3. calculate_output_based
    # ==================================================================

    def calculate_output_based(
        self,
        fuel_type: str,
        kwh_generated: Union[Decimal, int, float, str],
        efficiency: Union[Decimal, int, float, str] = Decimal("0.35"),
        count: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate generator emissions using output-based method.

        Computes emissions from the electricity generated by fuel-burning
        equipment (typically diesel/gas generators) by back-calculating
        fuel input from electrical output and thermal efficiency.

        Formula:
            fuel_kwh     = kwh_generated / efficiency
            total_fuel   = fuel_kwh x count
            fuel_ef_kwh  = FUEL_EMISSION_FACTORS[fuel_type]["ef_per_kwh"]
            ttw_co2e     = total_fuel x fuel_ef_kwh
            wtt_ratio    = wtt_per_litre / ef_per_litre
            wtt_co2e     = ttw_co2e x wtt_ratio
            co2e_kg      = ttw_co2e + wtt_co2e

        Args:
            fuel_type: Fuel type key (diesel, petrol, lpg, cng, natural_gas).
            kwh_generated: Electrical energy output in kWh per unit.
            efficiency: Generator thermal efficiency (default 0.35 = 35%).
            count: Number of generators (default 1).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            kwh_generated, fuel_kwh_input, efficiency, method,
            fuel_type, count, ef_source, dqi_score,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If kwh_generated <= 0, efficiency out of range,
                        or fuel_type unknown.

        Example:
            >>> result = engine.calculate_output_based(
            ...     fuel_type="diesel",
            ...     kwh_generated=Decimal("10000"),
            ...     efficiency=Decimal("0.35"),
            ...     count=2,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            f_type = self._validate_fuel_type(fuel_type)
            kwh_out = self._validate_positive_decimal(kwh_generated, "kwh_generated")
            eff = self._validate_efficiency(efficiency)
            cnt = self._validate_count(count)

            # Step 2: Resolve emission factors (ZERO HALLUCINATION)
            fuel_ef = FUEL_EMISSION_FACTORS[f_type]
            ef_per_kwh = fuel_ef["ef_per_kwh"]
            ef_per_litre = fuel_ef["ef_per_litre"]
            wtt_per_litre = fuel_ef["wtt_per_litre"]

            # WTT ratio for the fuel
            wtt_ratio = _q(wtt_per_litre / ef_per_litre)

            # Step 3: Calculate fuel input and emissions (Decimal only)
            count_dec = _safe_decimal(cnt)
            fuel_kwh_input = _q(kwh_out / eff)
            total_fuel_kwh = _q(fuel_kwh_input * count_dec)

            ttw_co2e = _q(total_fuel_kwh * ef_per_kwh)
            wtt_co2e = _q(ttw_co2e * wtt_ratio)
            co2e_kg = _q(ttw_co2e + wtt_co2e)

            # Step 4: Build provenance hash
            input_data = {
                "fuel_type": f_type,
                "kwh_generated": str(kwh_out),
                "efficiency": str(eff),
                "count": cnt,
            }
            output_data = {
                "co2e_kg": str(co2e_kg),
                "ttw_co2e_kg": str(ttw_co2e),
                "wtt_co2e_kg": str(wtt_co2e),
                "fuel_kwh_input": str(fuel_kwh_input),
            }
            provenance_hash = _calculate_provenance_hash(input_data, output_data)

            # Step 5: Build result
            timestamp = datetime.now(timezone.utc).isoformat()
            result: Dict[str, Any] = {
                "co2e_kg": co2e_kg,
                "ttw_co2e_kg": ttw_co2e,
                "wtt_co2e_kg": wtt_co2e,
                "kwh_generated": kwh_out,
                "fuel_kwh_input": fuel_kwh_input,
                "total_fuel_kwh": total_fuel_kwh,
                "efficiency": eff,
                "method": "output_based",
                "fuel_type": f_type,
                "count": cnt,
                "ef_per_kwh": ef_per_kwh,
                "wtt_ratio": wtt_ratio,
                "ef_source": "DEFRA_2024",
                "dqi_score": DQI_SCORES["output_based"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Output-based calculation complete: fuel=%s, kwh_out=%s, "
                "eff=%s, count=%d, fuel_kwh=%s, co2e=%s kg, "
                "duration=%.4fs",
                f_type, kwh_out, eff, cnt,
                fuel_kwh_input, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # 4. calculate_average_data
    # ==================================================================

    def calculate_average_data(
        self,
        equipment_type: str,
        count: int = 1,
        country_code: str = "US",
    ) -> Dict[str, Any]:
        """
        Calculate equipment emissions using average-data benchmarks.

        Fallback method when metered data is unavailable. Uses industry
        benchmark parameters (default hours, load factor, fuel consumption)
        for the specified equipment type.

        For diesel/petrol/lpg/cng equipment:
            fuel_litres = default_hours x fuel_consumption_lph x count
            co2e        = fuel_litres x ef_per_litre

        For electric equipment (HVAC):
            energy_kwh = typical_power_kw x default_hours x default_load_factor x count
            co2e       = energy_kwh x grid_ef

        Args:
            equipment_type: Equipment category key (e.g., "manufacturing").
            count: Number of equipment units (default 1).
            country_code: ISO country code for grid EF (default "US").

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg,
            method, equipment_type, count, benchmark_hours,
            benchmark_load_factor, benchmark_fuel_lph, primary_fuel,
            ef_source, dqi_score, provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If equipment_type or country_code is invalid.

        Example:
            >>> result = engine.calculate_average_data(
            ...     equipment_type="construction",
            ...     count=5,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            eq_type = self._validate_equipment_type(equipment_type)
            cnt = self._validate_count(count)
            c_code = self._validate_country_code(country_code)

            # Step 2: Resolve benchmarks (ZERO HALLUCINATION)
            benchmark = EQUIPMENT_BENCHMARKS[eq_type]
            default_hours = benchmark["default_hours"]
            default_lf = benchmark["default_load_factor"]
            primary_fuel = benchmark["primary_fuel"]
            fuel_lph = benchmark["fuel_consumption_lph"]
            typical_power = benchmark["typical_power_kw"]

            count_dec = _safe_decimal(cnt)

            # Step 3: Calculate based on fuel type
            if primary_fuel == "electricity":
                # Electric equipment: use grid emissions
                energy_kwh = _q(typical_power * default_hours * default_lf * count_dec)
                grid_ef = GRID_EMISSION_FACTORS[c_code]
                co2e_kg = _q(energy_kwh * grid_ef)
                ttw_co2e = _ZERO
                wtt_co2e = co2e_kg
                total_fuel_litres = _ZERO
                ef_source = "IEA_2024"
            else:
                # Fuel-burning equipment: use fuel consumption benchmarks
                total_fuel_litres = _q(default_hours * fuel_lph * count_dec)
                fuel_ef = FUEL_EMISSION_FACTORS.get(primary_fuel, FUEL_EMISSION_FACTORS["diesel"])
                ef_per_litre = fuel_ef["ef_per_litre"]
                wtt_per_litre = fuel_ef["wtt_per_litre"]

                ttw_co2e = _q(total_fuel_litres * ef_per_litre)
                wtt_co2e = _q(total_fuel_litres * wtt_per_litre)
                co2e_kg = _q(ttw_co2e + wtt_co2e)
                energy_kwh = _ZERO
                ef_source = "DEFRA_2024"

            # Step 4: Build provenance hash
            input_data = {
                "equipment_type": eq_type,
                "count": cnt,
                "country_code": c_code,
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
                "method": "average_data",
                "equipment_type": eq_type,
                "equipment_label": benchmark["label"],
                "count": cnt,
                "country_code": c_code,
                "benchmark_hours": default_hours,
                "benchmark_load_factor": default_lf,
                "benchmark_fuel_lph": fuel_lph,
                "total_fuel_litres": total_fuel_litres,
                "energy_kwh": energy_kwh,
                "primary_fuel": primary_fuel,
                "typical_power_kw": typical_power,
                "ef_source": ef_source,
                "dqi_score": DQI_SCORES["average_data"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Average-data calculation complete: type=%s, count=%d, "
                "fuel=%s, co2e=%s kg, duration=%.4fs",
                eq_type, cnt, primary_fuel, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # 5. calculate_batch
    # ==================================================================

    def calculate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple equipment calculations in a single batch.

        Each item dict must contain a 'method' key indicating the calculation
        type (energy_based, fuel_based, output_based, average_data, estimate)
        plus the method-specific parameters.

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
            ...     {"method": "energy_based", "equipment_type": "manufacturing",
            ...      "power_kw": 75, "operating_hours": 4000, "count": 3},
            ...     {"method": "fuel_based", "fuel_type": "diesel",
            ...      "fuel_litres": 5000, "count": 2},
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
        method = item.get("method", "energy_based").lower().strip()

        if method == "energy_based":
            return self.calculate_energy_based(
                equipment_type=item["equipment_type"],
                power_kw=item["power_kw"],
                operating_hours=item["operating_hours"],
                load_factor=item.get("load_factor"),
                country_code=item.get("country_code", "US"),
                count=item.get("count", 1),
            )
        elif method == "fuel_based":
            return self.calculate_fuel_based(
                fuel_type=item["fuel_type"],
                fuel_litres=item["fuel_litres"],
                count=item.get("count", 1),
                include_wtt=item.get("include_wtt", True),
            )
        elif method == "output_based":
            return self.calculate_output_based(
                fuel_type=item["fuel_type"],
                kwh_generated=item["kwh_generated"],
                efficiency=item.get("efficiency", Decimal("0.35")),
                count=item.get("count", 1),
            )
        elif method == "average_data":
            return self.calculate_average_data(
                equipment_type=item["equipment_type"],
                count=item.get("count", 1),
                country_code=item.get("country_code", "US"),
            )
        elif method == "estimate":
            return self.estimate_annual_emissions(
                equipment_type=item["equipment_type"],
                power_kw=item.get("power_kw"),
                count=item.get("count", 1),
            )
        else:
            raise ValueError(
                f"Unknown batch method '{method}' at index {idx}. "
                f"Supported: energy_based, fuel_based, output_based, "
                f"average_data, estimate"
            )

    # ==================================================================
    # 6. estimate_annual_emissions
    # ==================================================================

    def estimate_annual_emissions(
        self,
        equipment_type: str,
        power_kw: Optional[Union[Decimal, int, float, str]] = None,
        count: int = 1,
    ) -> Dict[str, Any]:
        """
        Quick annual emission estimation using benchmark parameters.

        Provides a rapid estimate for screening-level assessments when
        detailed operating data is not available. Uses industry benchmark
        hours, load factors, and fuel consumption rates.

        For fuel-burning equipment:
            fuel_litres = benchmark_hours x fuel_lph x count
            co2e        = fuel_litres x (ef_per_litre + wtt_per_litre)

        For electric equipment:
            If power_kw provided, use it; otherwise use typical_power_kw.
            energy_kwh = power x hours x load_factor x count
            co2e       = energy_kwh x grid_ef("US")

        Args:
            equipment_type: Equipment category key.
            power_kw: Optional power rating override in kW. If None, uses
                      benchmark typical_power_kw.
            count: Number of equipment units (default 1).

        Returns:
            Dict with keys: co2e_kg, ttw_co2e_kg, wtt_co2e_kg, method,
            equipment_type, count, ef_source, dqi_score,
            provenance_hash, calculation_timestamp.

        Raises:
            ValueError: If equipment_type is invalid.

        Example:
            >>> result = engine.estimate_annual_emissions(
            ...     equipment_type="generator",
            ...     count=3,
            ... )
            >>> assert result["co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate inputs
            eq_type = self._validate_equipment_type(equipment_type)
            cnt = self._validate_count(count)

            # Step 2: Resolve benchmarks (ZERO HALLUCINATION)
            benchmark = EQUIPMENT_BENCHMARKS[eq_type]
            default_hours = benchmark["default_hours"]
            default_lf = benchmark["default_load_factor"]
            primary_fuel = benchmark["primary_fuel"]
            fuel_lph = benchmark["fuel_consumption_lph"]
            typical_power = benchmark["typical_power_kw"]

            # Use provided power or benchmark default
            if power_kw is not None:
                pwr = self._validate_positive_decimal(power_kw, "power_kw")
            else:
                pwr = typical_power

            count_dec = _safe_decimal(cnt)

            # Step 3: Calculate based on fuel type
            if primary_fuel == "electricity":
                energy_kwh = _q(pwr * default_hours * default_lf * count_dec)
                grid_ef = GRID_EMISSION_FACTORS["US"]
                co2e_kg = _q(energy_kwh * grid_ef)
                ttw_co2e = _ZERO
                wtt_co2e = co2e_kg
                total_fuel_litres = _ZERO
                ef_source = "IEA_2024"
            else:
                total_fuel_litres = _q(default_hours * fuel_lph * count_dec)
                fuel_ef = FUEL_EMISSION_FACTORS.get(
                    primary_fuel, FUEL_EMISSION_FACTORS["diesel"]
                )
                ef_per_litre = fuel_ef["ef_per_litre"]
                wtt_per_litre = fuel_ef["wtt_per_litre"]

                ttw_co2e = _q(total_fuel_litres * ef_per_litre)
                wtt_co2e = _q(total_fuel_litres * wtt_per_litre)
                co2e_kg = _q(ttw_co2e + wtt_co2e)
                energy_kwh = _ZERO
                ef_source = "DEFRA_2024"

            # Step 4: Build provenance hash
            input_data = {
                "equipment_type": eq_type,
                "power_kw": str(pwr),
                "count": cnt,
                "method": "estimate",
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
                "method": "estimate",
                "equipment_type": eq_type,
                "equipment_label": benchmark["label"],
                "power_kw": pwr,
                "count": cnt,
                "benchmark_hours": default_hours,
                "benchmark_load_factor": default_lf,
                "benchmark_fuel_lph": fuel_lph,
                "total_fuel_litres": total_fuel_litres,
                "energy_kwh": energy_kwh,
                "primary_fuel": primary_fuel,
                "ef_source": ef_source,
                "dqi_score": DQI_SCORES["estimate"],
                "provenance_hash": provenance_hash,
                "calculation_timestamp": timestamp,
            }

            # Step 6: Record metrics
            duration = time.monotonic() - start_time
            self._calculation_count += 1

            logger.debug(
                "Estimate complete: type=%s, power=%s kW, count=%d, "
                "co2e=%s kg, duration=%.4fs",
                eq_type, pwr, cnt, co2e_kg, duration,
            )

            return result

    # ==================================================================
    # EMISSION FACTOR ACCESSORS (read-only)
    # ==================================================================

    @staticmethod
    def get_equipment_benchmarks() -> Dict[str, Dict[str, str]]:
        """
        Return all equipment benchmarks as a serializable dict.

        Returns:
            Dict mapping equipment type to benchmark parameters.
        """
        return {
            eq_type: {
                "label": bench["label"],
                "default_hours": str(bench["default_hours"]),
                "default_load_factor": str(bench["default_load_factor"]),
                "primary_fuel": bench["primary_fuel"],
                "fuel_consumption_lph": str(bench["fuel_consumption_lph"]),
                "typical_power_kw": str(bench["typical_power_kw"]),
                "description": bench["description"],
            }
            for eq_type, bench in EQUIPMENT_BENCHMARKS.items()
        }

    @staticmethod
    def get_fuel_emission_factors() -> Dict[str, Dict[str, str]]:
        """
        Return all fuel emission factors as a serializable dict.

        Returns:
            Dict mapping fuel type to EF parameter strings.
        """
        return {
            ftype: {k: str(v) for k, v in ef.items()}
            for ftype, ef in FUEL_EMISSION_FACTORS.items()
        }

    @staticmethod
    def get_grid_emission_factors() -> Dict[str, str]:
        """
        Return all grid emission factors as a serializable dict.

        Returns:
            Dict mapping country code to kgCO2e/kWh string.
        """
        return {code: str(ef) for code, ef in GRID_EMISSION_FACTORS.items()}

    @staticmethod
    def get_supported_equipment_types() -> List[str]:
        """
        Return list of supported equipment types.

        Returns:
            List of equipment type key strings.
        """
        return list(VALID_EQUIPMENT_TYPES)

    @staticmethod
    def get_supported_fuel_types() -> List[str]:
        """
        Return list of supported fuel types for equipment.

        Returns:
            List of fuel type key strings.
        """
        return list(FUEL_EMISSION_FACTORS.keys())

    # ==================================================================
    # ENGINE INFO
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Return engine metadata and status information.

        Returns:
            Dict with engine_id, version, agent_id, calculation_count,
            initialized_at, equipment_types, fuel_types.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "calculation_count": self._calculation_count,
            "initialized_at": self._initialized_at,
            "equipment_types": VALID_EQUIPMENT_TYPES,
            "fuel_types": list(FUEL_EMISSION_FACTORS.keys()),
            "grid_countries": list(GRID_EMISSION_FACTORS.keys()),
        }


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "EquipmentCalculatorEngine",
    "get_equipment_calculator",
    "reset_equipment_calculator",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "EQUIPMENT_BENCHMARKS",
    "FUEL_EMISSION_FACTORS",
    "GRID_EMISSION_FACTORS",
    "DQI_SCORES",
    "VALID_EQUIPMENT_TYPES",
]
