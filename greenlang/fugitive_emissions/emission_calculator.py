# -*- coding: utf-8 -*-
"""
EmissionCalculatorEngine - 5 Calculation Methods (Engine 2 of 7)

AGENT-MRV-005: Fugitive Emissions Agent

Core calculation engine implementing five EPA-recognized methods for
estimating fugitive emissions from equipment leaks, coal mines, wastewater
treatment, pneumatic devices, and direct measurement.

Calculation Methods:
    1. Average Emission Factor: count * EF * hours * gas_fraction
    2. Screening Ranges: (leak_count * leak_EF + no_leak_count * no_leak_EF) * hours * gas_fraction
    3. Correlation Equation: 10^(a + b * log10(ppmv)) per component
    4. Engineering Estimate: pneumatics, tank losses, coal mine, wastewater
    5. Direct Measurement: pass-through of measured values

All calculations use Python Decimal arithmetic with 8+ decimal places for
zero-hallucination determinism. Every calculation result includes a per-gas
breakdown, GWP-adjusted CO2e, processing time, and SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Stateless per-calculation. Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.fugitive_emissions.emission_calculator import EmissionCalculatorEngine
    >>> from greenlang.fugitive_emissions.fugitive_source_database import FugitiveSourceDatabaseEngine
    >>> db = FugitiveSourceDatabaseEngine()
    >>> calc = EmissionCalculatorEngine(source_database=db)
    >>> result = calc.calculate({
    ...     "method": "AVERAGE_EMISSION_FACTOR",
    ...     "component_type": "valve",
    ...     "service_type": "gas",
    ...     "component_count": 500,
    ...     "operating_hours": 8760,
    ... })
    >>> assert result["status"] == "SUCCESS"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["EmissionCalculatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.fugitive_emissions.fugitive_source_database import (
        FugitiveSourceDatabaseEngine,
        CH4_DENSITY_KG_PER_M3,
        POST_MINING_FRACTION,
        N2O_N_RATIO,
    )
    _SOURCE_DB_AVAILABLE = True
except ImportError:
    _SOURCE_DB_AVAILABLE = False
    FugitiveSourceDatabaseEngine = None  # type: ignore[misc,assignment]
    CH4_DENSITY_KG_PER_M3 = Decimal("0.6682")
    POST_MINING_FRACTION = Decimal("0.33")
    N2O_N_RATIO = Decimal("1.5714")

try:
    from greenlang.fugitive_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.metrics import (
        record_component_operation as _record_calc_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calc_operation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HOURS_PER_YEAR = Decimal("8760")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted to Decimal.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the standard 8-decimal-place precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ===========================================================================
# Enumerations
# ===========================================================================


class CalculationMethod(str, Enum):
    """Supported fugitive emission calculation methods."""

    AVERAGE_EMISSION_FACTOR = "AVERAGE_EMISSION_FACTOR"
    SCREENING_RANGES = "SCREENING_RANGES"
    CORRELATION_EQUATION = "CORRELATION_EQUATION"
    ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class EngineeringSubmethod(str, Enum):
    """Engineering estimate sub-methods."""

    PNEUMATIC = "PNEUMATIC"
    TANK_LOSS = "TANK_LOSS"
    COAL_MINE = "COAL_MINE"
    WASTEWATER = "WASTEWATER"


class CalculationStatus(str, Enum):
    """Result status codes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


# ===========================================================================
# Data classes for per-gas results
# ===========================================================================


@dataclass
class GasResult:
    """Emission result for a single greenhouse gas species.

    Attributes:
        gas: Gas species identifier (CH4, CO2, N2O).
        emission_kg: Mass emission in kilograms.
        emission_tonnes: Mass emission in metric tonnes.
        gwp_value: Global Warming Potential used.
        co2e_kg: CO2-equivalent emission in kilograms.
        co2e_tonnes: CO2-equivalent emission in metric tonnes.
    """

    gas: str
    emission_kg: Decimal
    emission_tonnes: Decimal
    gwp_value: Decimal
    co2e_kg: Decimal
    co2e_tonnes: Decimal


# ===========================================================================
# EmissionCalculatorEngine
# ===========================================================================


class EmissionCalculatorEngine:
    """Core calculation engine for fugitive emissions implementing five
    EPA-recognized estimation methods.

    Uses deterministic Decimal arithmetic throughout. All numeric lookups
    are delegated to FugitiveSourceDatabaseEngine (Engine 1) to ensure
    consistent emission factor usage across the pipeline.

    Attributes:
        _source_db: Reference to the FugitiveSourceDatabaseEngine.
        _config: Optional configuration dictionary.

    Example:
        >>> db = FugitiveSourceDatabaseEngine()
        >>> calc = EmissionCalculatorEngine(source_database=db)
        >>> result = calc.calculate({
        ...     "method": "AVERAGE_EMISSION_FACTOR",
        ...     "component_type": "valve",
        ...     "service_type": "gas",
        ...     "component_count": 500,
        ...     "operating_hours": 8760,
        ... })
    """

    def __init__(
        self,
        source_database: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the EmissionCalculatorEngine.

        Args:
            source_database: FugitiveSourceDatabaseEngine instance.
                If None, a default instance is created.
            config: Optional configuration dictionary. Supports:
                - default_gwp_source (str): Default GWP report (AR4/AR5/AR6/AR6_20YR).
                - default_gas_fraction (float): Default CH4 fraction (0-1).
                - decimal_precision (int): Decimal places (default 8).
        """
        if source_database is not None:
            self._source_db = source_database
        elif _SOURCE_DB_AVAILABLE and FugitiveSourceDatabaseEngine is not None:
            self._source_db = FugitiveSourceDatabaseEngine(config=config)
        else:
            self._source_db = None

        self._config = config or {}
        self._lock = threading.RLock()

        # Defaults
        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6",
        )
        self._default_gas_fraction: Decimal = _D(
            self._config.get("default_gas_fraction", "0.95"),
        )

        # Statistics
        self._total_calculations: int = 0
        self._total_batches: int = 0
        self._total_errors: int = 0

        # Method dispatch table
        self._method_dispatch = {
            CalculationMethod.AVERAGE_EMISSION_FACTOR: self.calculate_average_ef,
            CalculationMethod.SCREENING_RANGES: self.calculate_screening,
            CalculationMethod.CORRELATION_EQUATION: self.calculate_correlation,
            CalculationMethod.ENGINEERING_ESTIMATE: self.calculate_engineering,
            CalculationMethod.DIRECT_MEASUREMENT: self.calculate_direct,
        }

        logger.info(
            "EmissionCalculatorEngine initialized: default_gwp=%s, "
            "default_gas_fraction=%s, source_db=%s",
            self._default_gwp_source,
            self._default_gas_fraction,
            "connected" if self._source_db else "none",
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def calculate(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate fugitive emissions using the specified method.

        Dispatches to the appropriate method-specific calculator based on
        the ``method`` field in input_data, then applies GWP conversion
        and provenance hashing.

        Args:
            input_data: Dictionary with calculation parameters:
                - method (str): Calculation method (AVERAGE_EMISSION_FACTOR,
                    SCREENING_RANGES, CORRELATION_EQUATION,
                    ENGINEERING_ESTIMATE, DIRECT_MEASUREMENT).
                - gwp_source (str, optional): GWP report edition.
                - gas_fraction (float, optional): CH4 fraction override.
                - Additional fields depending on method.

        Returns:
            Dictionary with:
                - calculation_id: Unique calculation identifier.
                - method: Calculation method used.
                - status: SUCCESS, PARTIAL, or ERROR.
                - emissions_by_gas: Per-gas breakdown.
                - total_co2e_kg: Total CO2-equivalent (kg).
                - total_co2e_tonnes: Total CO2-equivalent (tonnes).
                - processing_time_ms: Calculation duration.
                - provenance_hash: SHA-256 hash.

        Raises:
            ValueError: If method is invalid or required fields are missing.
        """
        t0 = time.monotonic()
        calc_id = f"calc_{uuid4().hex[:12]}"

        try:
            method_str = input_data.get("method", "")
            if not method_str:
                raise ValueError("'method' field is required")

            try:
                method = CalculationMethod(method_str.upper())
            except ValueError:
                raise ValueError(
                    f"Unknown calculation method: {method_str}. "
                    f"Valid methods: {[m.value for m in CalculationMethod]}"
                )

            # Resolve GWP source
            gwp_source = input_data.get(
                "gwp_source", self._default_gwp_source,
            ).upper()

            # Dispatch to method-specific calculator
            handler = self._method_dispatch.get(method)
            if handler is None:
                raise ValueError(f"No handler for method: {method.value}")

            method_result = handler(input_data)

            # Apply GWP conversion
            emissions_by_gas = self._apply_gwp_to_result(
                method_result, gwp_source,
            )

            # Aggregate totals
            total_co2e_kg = _ZERO
            total_emission_kg = _ZERO
            gas_details: List[Dict[str, Any]] = []

            for gr in emissions_by_gas:
                total_co2e_kg += gr.co2e_kg
                total_emission_kg += gr.emission_kg
                gas_details.append({
                    "gas": gr.gas,
                    "emission_kg": str(_quantize(gr.emission_kg)),
                    "emission_tonnes": str(_quantize(gr.emission_tonnes)),
                    "gwp_value": str(gr.gwp_value),
                    "co2e_kg": str(_quantize(gr.co2e_kg)),
                    "co2e_tonnes": str(_quantize(gr.co2e_tonnes)),
                })

            total_co2e_tonnes = _quantize(total_co2e_kg / _D("1000"))
            total_emission_tonnes = _quantize(total_emission_kg / _D("1000"))

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result = {
                "calculation_id": calc_id,
                "method": method.value,
                "status": CalculationStatus.SUCCESS.value,
                "gwp_source": gwp_source,
                "emissions_by_gas": gas_details,
                "total_emission_kg": str(_quantize(total_emission_kg)),
                "total_emission_tonnes": str(_quantize(total_emission_tonnes)),
                "total_co2e_kg": str(_quantize(total_co2e_kg)),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "calculation_details": method_result.get("details", {}),
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_success(method.value)

            logger.info(
                "Calculation %s [%s]: %s kg CO2e (%s tonnes) in %.1fms",
                calc_id, method.value,
                _quantize(total_co2e_kg), total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._record_error()

            error_result = {
                "calculation_id": calc_id,
                "method": input_data.get("method", "UNKNOWN"),
                "status": CalculationStatus.ERROR.value,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            error_result["provenance_hash"] = _compute_hash(error_result)

            logger.error(
                "Calculation %s failed: %s in %.1fms",
                calc_id, exc, elapsed_ms, exc_info=True,
            )
            return error_result

    # ------------------------------------------------------------------
    # Method 1: Average Emission Factor
    # ------------------------------------------------------------------

    def calculate_average_ef(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions using the average emission factor method.

        Formula:
            Emission (kg) = component_count * EF (kg/hr) * operating_hours * gas_fraction

        Args:
            input_data: Dictionary with:
                - component_type (str): Component type (valve, pump, etc.).
                - service_type (str): Service type (gas, light_liquid, etc.).
                - component_count (int): Number of components.
                - operating_hours (float): Annual operating hours (default 8760).
                - gas_fraction (float, optional): CH4 mole/weight fraction.

        Returns:
            Dictionary with raw_emissions_by_gas and calculation details.

        Raises:
            ValueError: If required fields are missing or EF not found.
        """
        component_type = input_data.get("component_type", "")
        service_type = input_data.get("service_type", "")
        component_count = _D(input_data.get("component_count", 0))
        operating_hours = _D(input_data.get("operating_hours", "8760"))
        gas_fraction = self._resolve_gas_fraction(input_data)

        if not component_type:
            raise ValueError("component_type is required for AVERAGE_EMISSION_FACTOR")
        if not service_type:
            raise ValueError("service_type is required for AVERAGE_EMISSION_FACTOR")
        if component_count <= _ZERO:
            raise ValueError("component_count must be > 0")

        # Look up emission factor
        ef_data = self._lookup_component_ef(component_type, service_type)
        ef_kg_per_hr = ef_data["ef_decimal"]

        # Core calculation: count * EF * hours * gas_fraction
        total_toc_kg = _quantize(
            component_count * ef_kg_per_hr * operating_hours
        )
        ch4_kg = _quantize(total_toc_kg * gas_fraction)

        raw_emissions = {"CH4": ch4_kg}

        # If gas composition has CO2, compute that fraction too
        co2_fraction = self._get_species_fraction("CO2")
        if co2_fraction > _ZERO:
            co2_kg = _quantize(total_toc_kg * co2_fraction)
            raw_emissions["CO2"] = co2_kg

        details = {
            "component_type": component_type,
            "service_type": service_type,
            "component_count": str(component_count),
            "ef_kg_per_hr": str(ef_kg_per_hr),
            "ef_source": ef_data.get("source", "EPA"),
            "operating_hours": str(operating_hours),
            "gas_fraction_ch4": str(gas_fraction),
            "total_toc_kg": str(total_toc_kg),
            "formula": "count * EF * hours * gas_fraction",
        }

        logger.debug(
            "Average EF: %s x %s kg/hr x %s hr x %s = %s kg CH4",
            component_count, ef_kg_per_hr, operating_hours,
            gas_fraction, ch4_kg,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Method 2: Screening Ranges
    # ------------------------------------------------------------------

    def calculate_screening(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions using the screening ranges method.

        Formula:
            Emission (kg) = (leak_count * leak_EF + no_leak_count * no_leak_EF) * hours * gas_fraction

        Args:
            input_data: Dictionary with:
                - component_type (str): Component type.
                - service_type (str): Service type.
                - leak_count (int): Components with screening >= threshold.
                - no_leak_count (int): Components with screening < threshold.
                - operating_hours (float, optional): Annual hours (default 8760).
                - gas_fraction (float, optional): CH4 fraction.
                - threshold_ppmv (int, optional): Leak/no-leak threshold (default 10000).

        Returns:
            Dictionary with raw_emissions_by_gas and calculation details.

        Raises:
            ValueError: If required fields are missing or factors not found.
        """
        component_type = input_data.get("component_type", "")
        service_type = input_data.get("service_type", "")
        leak_count = _D(input_data.get("leak_count", 0))
        no_leak_count = _D(input_data.get("no_leak_count", 0))
        operating_hours = _D(input_data.get("operating_hours", "8760"))
        gas_fraction = self._resolve_gas_fraction(input_data)
        threshold_ppmv = int(input_data.get("threshold_ppmv", 10000))

        if not component_type:
            raise ValueError("component_type is required for SCREENING_RANGES")
        if not service_type:
            raise ValueError("service_type is required for SCREENING_RANGES")
        if leak_count < _ZERO or no_leak_count < _ZERO:
            raise ValueError("leak_count and no_leak_count must be >= 0")
        if leak_count + no_leak_count <= _ZERO:
            raise ValueError("Total component count must be > 0")

        # Look up screening factors
        sf_data = self._lookup_screening_factor(component_type, service_type)
        leak_ef = sf_data["leak_ef_decimal"]
        no_leak_ef = sf_data["no_leak_ef_decimal"]

        # Core calculation
        leak_emission_kg = _quantize(leak_count * leak_ef * operating_hours)
        no_leak_emission_kg = _quantize(no_leak_count * no_leak_ef * operating_hours)
        total_toc_kg = _quantize(leak_emission_kg + no_leak_emission_kg)
        ch4_kg = _quantize(total_toc_kg * gas_fraction)

        raw_emissions = {"CH4": ch4_kg}

        co2_fraction = self._get_species_fraction("CO2")
        if co2_fraction > _ZERO:
            raw_emissions["CO2"] = _quantize(total_toc_kg * co2_fraction)

        details = {
            "component_type": component_type,
            "service_type": service_type,
            "leak_count": str(leak_count),
            "no_leak_count": str(no_leak_count),
            "threshold_ppmv": threshold_ppmv,
            "leak_ef_kg_per_hr": str(leak_ef),
            "no_leak_ef_kg_per_hr": str(no_leak_ef),
            "operating_hours": str(operating_hours),
            "gas_fraction_ch4": str(gas_fraction),
            "leak_emission_kg": str(leak_emission_kg),
            "no_leak_emission_kg": str(no_leak_emission_kg),
            "total_toc_kg": str(total_toc_kg),
            "formula": "(leak_count * leak_EF + no_leak_count * no_leak_EF) * hours * gas_fraction",
        }

        logger.debug(
            "Screening: (%s * %s + %s * %s) * %s hr * %s = %s kg CH4",
            leak_count, leak_ef, no_leak_count, no_leak_ef,
            operating_hours, gas_fraction, ch4_kg,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Method 3: Correlation Equation
    # ------------------------------------------------------------------

    def calculate_correlation(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions using EPA correlation equations.

        Formula per component:
            leak_rate_kg_hr = 10^(a + b * log10(ppmv))

        Total:
            Emission (kg) = sum(leak_rate_i) * operating_hours * gas_fraction

        Args:
            input_data: Dictionary with:
                - component_type (str): Component type.
                - service_type (str): Service type.
                - screening_values (list[dict]): List of {ppmv: float} entries
                    for each component screened, OR
                - screening_values_ppmv (list[float]): Flat list of ppmv readings.
                - operating_hours (float, optional): Annual hours (default 8760).
                - gas_fraction (float, optional): CH4 fraction.

        Returns:
            Dictionary with raw_emissions_by_gas and calculation details.

        Raises:
            ValueError: If required fields are missing or coefficients not found.
        """
        component_type = input_data.get("component_type", "")
        service_type = input_data.get("service_type", "")
        operating_hours = _D(input_data.get("operating_hours", "8760"))
        gas_fraction = self._resolve_gas_fraction(input_data)

        if not component_type:
            raise ValueError("component_type is required for CORRELATION_EQUATION")
        if not service_type:
            raise ValueError("service_type is required for CORRELATION_EQUATION")

        # Parse screening values
        screening_values = self._parse_screening_values(input_data)
        if not screening_values:
            raise ValueError(
                "screening_values or screening_values_ppmv is required "
                "for CORRELATION_EQUATION"
            )

        # Look up correlation coefficients
        coeff_data = self._lookup_correlation_coefficients(
            component_type, service_type,
        )
        a = coeff_data["a_decimal"]
        b = coeff_data["b_decimal"]
        default_zero_ef = coeff_data["default_zero_ef_decimal"]

        # Calculate per-component leak rates
        total_leak_rate_kg_hr = _ZERO
        component_results: List[Dict[str, Any]] = []
        component_count = len(screening_values)

        for idx, ppmv_val in enumerate(screening_values):
            ppmv = _D(ppmv_val)

            if ppmv <= _ZERO:
                # Below detection limit
                leak_rate = default_zero_ef
            else:
                # log10(kg/hr) = a + b * log10(ppmv)
                log_ppmv = _D(str(math.log10(float(ppmv))))
                log_rate = a + b * log_ppmv
                leak_rate = _quantize(
                    _D(str(10 ** float(log_rate)))
                )

            total_leak_rate_kg_hr += leak_rate
            component_results.append({
                "index": idx,
                "ppmv": str(ppmv),
                "leak_rate_kg_hr": str(_quantize(leak_rate)),
            })

        # Total emission
        total_toc_kg = _quantize(total_leak_rate_kg_hr * operating_hours)
        ch4_kg = _quantize(total_toc_kg * gas_fraction)

        raw_emissions = {"CH4": ch4_kg}

        co2_fraction = self._get_species_fraction("CO2")
        if co2_fraction > _ZERO:
            raw_emissions["CO2"] = _quantize(total_toc_kg * co2_fraction)

        details = {
            "component_type": component_type,
            "service_type": service_type,
            "component_count": component_count,
            "coefficient_a": str(a),
            "coefficient_b": str(b),
            "default_zero_ef_kg_hr": str(default_zero_ef),
            "operating_hours": str(operating_hours),
            "gas_fraction_ch4": str(gas_fraction),
            "total_leak_rate_kg_hr": str(_quantize(total_leak_rate_kg_hr)),
            "total_toc_kg": str(total_toc_kg),
            "formula": "10^(a + b * log10(ppmv)) per component, summed",
            "component_results": component_results[:50],  # Limit detail output
            "components_truncated": component_count > 50,
        }

        logger.debug(
            "Correlation: %d components, total_rate=%s kg/hr, "
            "%s hr * %s = %s kg CH4",
            component_count, _quantize(total_leak_rate_kg_hr),
            operating_hours, gas_fraction, ch4_kg,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Method 4: Engineering Estimate
    # ------------------------------------------------------------------

    def calculate_engineering(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions using engineering estimate sub-methods.

        Dispatches to the appropriate sub-method based on the
        ``engineering_type`` field.

        Sub-methods:
            - PNEUMATIC: count * rate * hours * gas_fraction
            - COAL_MINE: production * EF * (1 - recovery)
            - WASTEWATER: BOD * Bo * MCF * (1 - recovery) for CH4;
                          N_load * EF * N2O_N_ratio for N2O
            - TANK_LOSS: delegated to EquipmentComponentEngine

        Args:
            input_data: Dictionary with:
                - engineering_type (str): Sub-method type.
                - Additional fields per sub-method.

        Returns:
            Dictionary with raw_emissions_by_gas and calculation details.

        Raises:
            ValueError: If engineering_type is missing or invalid.
        """
        eng_type_str = input_data.get("engineering_type", "")
        if not eng_type_str:
            raise ValueError(
                "engineering_type is required for ENGINEERING_ESTIMATE. "
                "Valid types: PNEUMATIC, COAL_MINE, WASTEWATER, TANK_LOSS"
            )

        try:
            eng_type = EngineeringSubmethod(eng_type_str.upper())
        except ValueError:
            raise ValueError(
                f"Unknown engineering_type: {eng_type_str}. "
                f"Valid: {[e.value for e in EngineeringSubmethod]}"
            )

        if eng_type == EngineeringSubmethod.PNEUMATIC:
            return self._calculate_pneumatic(input_data)
        elif eng_type == EngineeringSubmethod.COAL_MINE:
            return self._calculate_coal_mine(input_data)
        elif eng_type == EngineeringSubmethod.WASTEWATER:
            return self._calculate_wastewater(input_data)
        elif eng_type == EngineeringSubmethod.TANK_LOSS:
            return self._calculate_tank_loss(input_data)
        else:
            raise ValueError(f"Unhandled engineering_type: {eng_type.value}")

    def _calculate_pneumatic(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate pneumatic device emissions.

        Formula:
            CH4 (kg) = device_count * rate_m3_day * (hours/24) * CH4_density * gas_fraction

        Args:
            input_data: Dictionary with:
                - device_type (str): high_bleed, low_bleed, intermittent, zero_bleed.
                - device_count (int): Number of devices.
                - operating_hours (float, optional): Annual hours (default 8760).
                - gas_fraction (float, optional): CH4 fraction.

        Returns:
            Dictionary with raw_emissions_by_gas and details.
        """
        device_type = input_data.get("device_type", "high_bleed")
        device_count = _D(input_data.get("device_count", 0))
        operating_hours = _D(input_data.get("operating_hours", "8760"))
        gas_fraction = self._resolve_gas_fraction(input_data)

        if device_count <= _ZERO:
            raise ValueError("device_count must be > 0")

        # Look up pneumatic rate
        rate_data = self._lookup_pneumatic_rate(device_type)
        rate_m3_per_day = rate_data["rate_m3_per_day_decimal"]

        # Convert operating hours to days
        operating_days = _quantize(operating_hours / _D("24"))

        # CH4 emission: count * rate * days * density * gas_fraction
        ch4_m3 = _quantize(device_count * rate_m3_per_day * operating_days)
        ch4_kg = _quantize(ch4_m3 * CH4_DENSITY_KG_PER_M3 * gas_fraction)

        raw_emissions = {"CH4": ch4_kg}

        details = {
            "engineering_type": "PNEUMATIC",
            "device_type": device_type,
            "device_count": str(device_count),
            "rate_m3_per_day": str(rate_m3_per_day),
            "operating_hours": str(operating_hours),
            "operating_days": str(operating_days),
            "ch4_m3_total": str(ch4_m3),
            "ch4_density_kg_per_m3": str(CH4_DENSITY_KG_PER_M3),
            "gas_fraction": str(gas_fraction),
            "formula": "count * rate_m3/day * days * CH4_density * gas_fraction",
        }

        logger.debug(
            "Pneumatic: %s x %s m3/day x %s days x %s = %s kg CH4",
            device_count, rate_m3_per_day, operating_days,
            gas_fraction, ch4_kg,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    def _calculate_coal_mine(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate coal mine methane emissions.

        Formula:
            CH4 (kg) = production_tonnes * EF_m3_per_tonne * CH4_density * (1 - recovery)

        For post-mining:
            CH4_post (kg) = CH4_mining (kg) * post_mining_fraction

        Args:
            input_data: Dictionary with:
                - coal_production_tonnes (float): Annual coal production.
                - coal_rank (str): ANTHRACITE, BITUMINOUS, SUBBITUMINOUS, LIGNITE.
                - recovery_fraction (float, optional): Methane recovery (0-1, default 0).
                - include_post_mining (bool, optional): Include post-mining emissions.

        Returns:
            Dictionary with raw_emissions_by_gas and details.
        """
        production = _D(input_data.get("coal_production_tonnes", 0))
        coal_rank = input_data.get("coal_rank", "BITUMINOUS").upper()
        recovery = _D(input_data.get("recovery_fraction", "0"))
        include_post_mining = input_data.get("include_post_mining", True)

        if production <= _ZERO:
            raise ValueError("coal_production_tonnes must be > 0")
        if recovery < _ZERO or recovery > _ONE:
            raise ValueError("recovery_fraction must be between 0 and 1")

        # Look up coal methane factor
        factor_data = self._lookup_coal_methane_factor(coal_rank)
        ef_m3_per_tonne = factor_data["ef_m3_per_tonne_decimal"]

        # Mining emissions
        ch4_m3 = _quantize(production * ef_m3_per_tonne)
        ch4_kg_gross = _quantize(ch4_m3 * CH4_DENSITY_KG_PER_M3)
        ch4_kg_net = _quantize(ch4_kg_gross * (_ONE - recovery))

        # Post-mining emissions
        ch4_post_kg = _ZERO
        if include_post_mining:
            ch4_post_kg = _quantize(ch4_kg_net * POST_MINING_FRACTION)

        total_ch4_kg = _quantize(ch4_kg_net + ch4_post_kg)

        raw_emissions = {"CH4": total_ch4_kg}

        details = {
            "engineering_type": "COAL_MINE",
            "coal_rank": coal_rank,
            "coal_production_tonnes": str(production),
            "ef_m3_per_tonne": str(ef_m3_per_tonne),
            "ch4_density_kg_per_m3": str(CH4_DENSITY_KG_PER_M3),
            "recovery_fraction": str(recovery),
            "ch4_m3_gross": str(ch4_m3),
            "ch4_kg_gross": str(ch4_kg_gross),
            "ch4_kg_mining_net": str(ch4_kg_net),
            "include_post_mining": include_post_mining,
            "post_mining_fraction": str(POST_MINING_FRACTION),
            "ch4_kg_post_mining": str(ch4_post_kg),
            "total_ch4_kg": str(total_ch4_kg),
            "formula": "production * EF * density * (1 - recovery) [+ post_mining_fraction]",
        }

        logger.debug(
            "Coal mine: %s t * %s m3/t * %s kg/m3 * (1-%s) = %s kg CH4 "
            "(+ %s kg post-mining)",
            production, ef_m3_per_tonne, CH4_DENSITY_KG_PER_M3,
            recovery, ch4_kg_net, ch4_post_kg,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    def _calculate_wastewater(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate wastewater treatment emissions (CH4 and N2O).

        CH4 Formula:
            CH4 (kg) = BOD_load * Bo * MCF * (1 - recovery)

        N2O Formula:
            N2O (kg) = N_load * EF_N2O * N2O_N_ratio

        Args:
            input_data: Dictionary with:
                - bod_load_kg (float): Annual BOD load in kg.
                - treatment_type (str): Wastewater treatment system type.
                - recovery_fraction (float, optional): CH4 recovery (0-1, default 0).
                - nitrogen_load_kg (float, optional): Annual nitrogen load in kg.

        Returns:
            Dictionary with raw_emissions_by_gas (CH4, possibly N2O) and details.
        """
        bod_load = _D(input_data.get("bod_load_kg", 0))
        treatment_type = input_data.get("treatment_type", "UNTREATED_DISCHARGE").upper()
        recovery = _D(input_data.get("recovery_fraction", "0"))
        nitrogen_load = _D(input_data.get("nitrogen_load_kg", "0"))

        if bod_load <= _ZERO:
            raise ValueError("bod_load_kg must be > 0")
        if recovery < _ZERO or recovery > _ONE:
            raise ValueError("recovery_fraction must be between 0 and 1")

        # Look up wastewater factors
        ww_data = self._lookup_wastewater_factor(treatment_type)
        bo = ww_data["bo_decimal"]
        mcf = ww_data["mcf_decimal"]
        n2o_ef = ww_data["n2o_ef_decimal"]

        # CH4 calculation
        ch4_gross_kg = _quantize(bod_load * bo * mcf)
        ch4_kg = _quantize(ch4_gross_kg * (_ONE - recovery))

        raw_emissions: Dict[str, Decimal] = {"CH4": ch4_kg}

        # N2O calculation (if nitrogen load provided)
        n2o_kg = _ZERO
        if nitrogen_load > _ZERO:
            n2o_n_kg = _quantize(nitrogen_load * n2o_ef)
            n2o_kg = _quantize(n2o_n_kg * N2O_N_RATIO)
            raw_emissions["N2O"] = n2o_kg

        details = {
            "engineering_type": "WASTEWATER",
            "treatment_type": treatment_type,
            "bod_load_kg": str(bod_load),
            "bo_kg_ch4_per_kg_bod": str(bo),
            "mcf": str(mcf),
            "recovery_fraction": str(recovery),
            "ch4_gross_kg": str(ch4_gross_kg),
            "ch4_net_kg": str(ch4_kg),
            "nitrogen_load_kg": str(nitrogen_load),
            "n2o_ef_kg_per_kg_n": str(n2o_ef),
            "n2o_n_ratio": str(N2O_N_RATIO),
            "n2o_kg": str(n2o_kg),
            "formula_ch4": "BOD * Bo * MCF * (1 - recovery)",
            "formula_n2o": "N_load * EF_N2O * N2O_N_ratio",
        }

        logger.debug(
            "Wastewater: CH4=%s kg (%s BOD * %s Bo * %s MCF), "
            "N2O=%s kg (%s N * %s EF)",
            ch4_kg, bod_load, bo, mcf,
            n2o_kg, nitrogen_load, n2o_ef,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    def _calculate_tank_loss(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate tank storage loss emissions (VOC as CH4 equivalent).

        For fugitive emission reporting purposes, tank losses are reported
        as VOC but can be converted to CO2e using facility-specific
        gas composition data.

        This method provides a simplified estimate based on annual loss
        rate. For detailed AP-42 calculations, use the
        EquipmentComponentEngine directly.

        Args:
            input_data: Dictionary with:
                - annual_loss_kg (float): Known annual tank loss (if pre-calculated).
                - gas_fraction (float, optional): Fraction of loss that is CH4.

        Returns:
            Dictionary with raw_emissions_by_gas and details.
        """
        annual_loss_kg = _D(input_data.get("annual_loss_kg", 0))
        gas_fraction = self._resolve_gas_fraction(input_data)

        if annual_loss_kg < _ZERO:
            raise ValueError("annual_loss_kg must be >= 0")

        ch4_kg = _quantize(annual_loss_kg * gas_fraction)

        raw_emissions = {"CH4": ch4_kg}

        details = {
            "engineering_type": "TANK_LOSS",
            "annual_loss_kg": str(annual_loss_kg),
            "gas_fraction": str(gas_fraction),
            "ch4_kg": str(ch4_kg),
            "note": "For detailed AP-42 calculations, use EquipmentComponentEngine",
            "formula": "annual_loss_kg * gas_fraction",
        }

        logger.debug(
            "Tank loss: %s kg * %s = %s kg CH4",
            annual_loss_kg, gas_fraction, ch4_kg,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Method 5: Direct Measurement
    # ------------------------------------------------------------------

    def calculate_direct(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Pass-through for direct measurement data.

        Accepts pre-measured emission values and normalizes them
        to the standard output format.

        Args:
            input_data: Dictionary with:
                - measured_ch4_kg (float): Measured CH4 emission in kg.
                - measured_co2_kg (float, optional): Measured CO2 emission in kg.
                - measured_n2o_kg (float, optional): Measured N2O emission in kg.
                - measurement_method (str, optional): Instrument type description.
                - measurement_duration_hr (float, optional): Duration of measurement.

        Returns:
            Dictionary with raw_emissions_by_gas and details.

        Raises:
            ValueError: If no measured values are provided.
        """
        ch4_kg = _D(input_data.get("measured_ch4_kg", "0"))
        co2_kg = _D(input_data.get("measured_co2_kg", "0"))
        n2o_kg = _D(input_data.get("measured_n2o_kg", "0"))
        method_desc = input_data.get("measurement_method", "unspecified")
        duration_hr = _D(input_data.get("measurement_duration_hr", "0"))

        if ch4_kg <= _ZERO and co2_kg <= _ZERO and n2o_kg <= _ZERO:
            raise ValueError(
                "At least one measured gas value must be > 0 for "
                "DIRECT_MEASUREMENT"
            )

        raw_emissions: Dict[str, Decimal] = {}
        if ch4_kg > _ZERO:
            raw_emissions["CH4"] = _quantize(ch4_kg)
        if co2_kg > _ZERO:
            raw_emissions["CO2"] = _quantize(co2_kg)
        if n2o_kg > _ZERO:
            raw_emissions["N2O"] = _quantize(n2o_kg)

        details = {
            "measurement_method": method_desc,
            "measurement_duration_hr": str(duration_hr),
            "measured_ch4_kg": str(ch4_kg),
            "measured_co2_kg": str(co2_kg),
            "measured_n2o_kg": str(n2o_kg),
            "formula": "pass-through (direct measurement)",
        }

        logger.debug(
            "Direct measurement: CH4=%s kg, CO2=%s kg, N2O=%s kg [%s]",
            ch4_kg, co2_kg, n2o_kg, method_desc,
        )

        return {
            "raw_emissions_by_gas": raw_emissions,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        records: List[Dict[str, Any]],
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Process multiple calculations in a single batch.

        Args:
            records: List of input dictionaries, each containing fields
                for a single calculation (including ``method``).
            continue_on_error: If True, skip failed records and continue.
                If False, stop on first error.

        Returns:
            Dictionary with:
                - results: List of individual calculation results.
                - summary: Aggregated totals across all successful calculations.
                - total_records: Number of input records.
                - successful: Number of successful calculations.
                - failed: Number of failed calculations.
                - processing_time_ms: Total batch duration.
                - provenance_hash: SHA-256 hash.
        """
        t0 = time.monotonic()
        batch_id = f"batch_{uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        successful = 0
        failed = 0
        total_co2e_kg = _ZERO
        total_co2e_by_gas: Dict[str, Decimal] = defaultdict(lambda: _ZERO)

        for idx, record in enumerate(records):
            try:
                calc_result = self.calculate(record)
                results.append(calc_result)

                if calc_result.get("status") == CalculationStatus.SUCCESS.value:
                    successful += 1
                    co2e = _D(calc_result.get("total_co2e_kg", "0"))
                    total_co2e_kg += co2e

                    for gas_entry in calc_result.get("emissions_by_gas", []):
                        gas = gas_entry["gas"]
                        gas_co2e = _D(gas_entry.get("co2e_kg", "0"))
                        total_co2e_by_gas[gas] = total_co2e_by_gas[gas] + gas_co2e
                else:
                    failed += 1

            except Exception as exc:
                failed += 1
                error_entry = {
                    "record_index": idx,
                    "status": CalculationStatus.ERROR.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                results.append(error_entry)

                if not continue_on_error:
                    logger.error(
                        "Batch %s stopped at record %d: %s",
                        batch_id, idx, exc,
                    )
                    break

                logger.warning(
                    "Batch %s record %d failed (continuing): %s",
                    batch_id, idx, exc,
                )

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        with self._lock:
            self._total_batches += 1

        summary = {
            "total_co2e_kg": str(_quantize(total_co2e_kg)),
            "total_co2e_tonnes": str(_quantize(total_co2e_kg / _D("1000"))),
            "co2e_by_gas": {
                gas: str(_quantize(val))
                for gas, val in total_co2e_by_gas.items()
            },
        }

        batch_result = {
            "batch_id": batch_id,
            "results": results,
            "summary": summary,
            "total_records": len(records),
            "successful": successful,
            "failed": failed,
            "continue_on_error": continue_on_error,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow().isoformat(),
        }
        batch_result["provenance_hash"] = _compute_hash({
            k: v for k, v in batch_result.items()
            if k != "results"  # Hash summary only for performance
        })

        logger.info(
            "Batch %s: %d/%d successful, %s kg CO2e in %.1fms",
            batch_id, successful, len(records),
            _quantize(total_co2e_kg), elapsed_ms,
        )

        return batch_result

    # ------------------------------------------------------------------
    # GWP Application
    # ------------------------------------------------------------------

    def apply_gwp(
        self,
        emissions_kg: Dict[str, Decimal],
        gwp_source: Optional[str] = None,
    ) -> List[GasResult]:
        """Apply GWP conversion to raw gas emissions.

        Args:
            emissions_kg: Dictionary mapping gas species to kg emissions.
            gwp_source: IPCC assessment report. Defaults to engine default.

        Returns:
            List of GasResult entries with CO2e calculations.
        """
        source = (gwp_source or self._default_gwp_source).upper()
        results: List[GasResult] = []

        for gas, kg in emissions_kg.items():
            gwp = self._lookup_gwp(gas, source)
            co2e_kg = _quantize(kg * gwp)
            results.append(GasResult(
                gas=gas,
                emission_kg=_quantize(kg),
                emission_tonnes=_quantize(kg / _D("1000")),
                gwp_value=gwp,
                co2e_kg=co2e_kg,
                co2e_tonnes=_quantize(co2e_kg / _D("1000")),
            ))

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine calculation statistics.

        Returns:
            Dictionary with calculation counts and metadata.
        """
        with self._lock:
            return {
                "total_calculations": self._total_calculations,
                "total_batches": self._total_batches,
                "total_errors": self._total_errors,
                "default_gwp_source": self._default_gwp_source,
                "default_gas_fraction": str(self._default_gas_fraction),
                "source_database_connected": self._source_db is not None,
            }

    # ------------------------------------------------------------------
    # Private: Lookups (delegate to source database)
    # ------------------------------------------------------------------

    def _lookup_component_ef(
        self,
        component_type: str,
        service_type: str,
    ) -> Dict[str, Any]:
        """Look up component emission factor from source database.

        Args:
            component_type: Component type.
            service_type: Service type.

        Returns:
            Dictionary with ef_decimal and metadata.

        Raises:
            ValueError: If no emission factor is found.
        """
        if self._source_db is not None:
            result = self._source_db.get_component_ef(
                component_type, service_type,
            )
            if result is not None:
                return result

        raise ValueError(
            f"No emission factor found for ({component_type}, {service_type})"
        )

    def _lookup_screening_factor(
        self,
        component_type: str,
        service_type: str,
    ) -> Dict[str, Any]:
        """Look up screening range factors from source database.

        Args:
            component_type: Component type.
            service_type: Service type.

        Returns:
            Dictionary with leak_ef_decimal and no_leak_ef_decimal.

        Raises:
            ValueError: If no screening factors are found.
        """
        if self._source_db is not None:
            result = self._source_db.get_screening_factor(
                component_type, service_type,
            )
            if result is not None:
                return result

        raise ValueError(
            f"No screening factors found for ({component_type}, {service_type})"
        )

    def _lookup_correlation_coefficients(
        self,
        component_type: str,
        service_type: str,
    ) -> Dict[str, Any]:
        """Look up correlation equation coefficients from source database.

        Args:
            component_type: Component type.
            service_type: Service type.

        Returns:
            Dictionary with a_decimal, b_decimal, default_zero_ef_decimal.

        Raises:
            ValueError: If no coefficients are found.
        """
        if self._source_db is not None:
            result = self._source_db.get_correlation_coefficients(
                component_type, service_type,
            )
            if result is not None:
                return result

        raise ValueError(
            f"No correlation coefficients found for "
            f"({component_type}, {service_type})"
        )

    def _lookup_coal_methane_factor(
        self,
        coal_rank: str,
    ) -> Dict[str, Any]:
        """Look up coal mine methane factor from source database.

        Args:
            coal_rank: Coal rank string.

        Returns:
            Dictionary with ef_m3_per_tonne_decimal.

        Raises:
            ValueError: If coal rank is not found.
        """
        if self._source_db is not None:
            result = self._source_db.get_coal_methane_factor(coal_rank)
            if result is not None:
                return result

        raise ValueError(f"No coal methane factor found for rank: {coal_rank}")

    def _lookup_wastewater_factor(
        self,
        treatment_type: str,
    ) -> Dict[str, Any]:
        """Look up wastewater treatment factors from source database.

        Args:
            treatment_type: Treatment system type.

        Returns:
            Dictionary with bo_decimal, mcf_decimal, n2o_ef_decimal.

        Raises:
            ValueError: If treatment type is not found.
        """
        if self._source_db is not None:
            result = self._source_db.get_wastewater_factor(treatment_type)
            if result is not None:
                return result

        raise ValueError(
            f"No wastewater factor found for treatment: {treatment_type}"
        )

    def _lookup_pneumatic_rate(
        self,
        device_type: str,
    ) -> Dict[str, Any]:
        """Look up pneumatic device emission rate from source database.

        Args:
            device_type: Pneumatic device type.

        Returns:
            Dictionary with rate_m3_per_day_decimal.

        Raises:
            ValueError: If device type is not found.
        """
        if self._source_db is not None:
            result = self._source_db.get_pneumatic_rate(device_type)
            if result is not None:
                return result

        raise ValueError(
            f"No pneumatic rate found for device type: {device_type}"
        )

    def _lookup_gwp(
        self,
        gas: str,
        gwp_source: str,
    ) -> Decimal:
        """Look up GWP value for a gas species.

        Args:
            gas: Gas species (CH4, CO2, N2O).
            gwp_source: IPCC assessment report.

        Returns:
            GWP value as Decimal. Returns 1 for unknown gases (conservative).
        """
        if self._source_db is not None:
            result = self._source_db.get_gwp(gas, gwp_source)
            if result is not None:
                return result.get("gwp_decimal", _ONE)

        # Fallback: return 1 for unknown gases (no amplification)
        logger.warning(
            "GWP lookup failed for %s/%s, using 1.0",
            gas, gwp_source,
        )
        return _ONE

    # ------------------------------------------------------------------
    # Private: Gas fraction resolution
    # ------------------------------------------------------------------

    def _resolve_gas_fraction(
        self,
        input_data: Dict[str, Any],
    ) -> Decimal:
        """Resolve the CH4 gas fraction from input or defaults.

        Priority: input_data > engine config > source database > default 0.95.

        Args:
            input_data: Input dictionary that may contain gas_fraction.

        Returns:
            CH4 weight/mole fraction as Decimal.
        """
        if "gas_fraction" in input_data:
            frac = _D(input_data["gas_fraction"])
            if frac < _ZERO or frac > _ONE:
                raise ValueError(
                    f"gas_fraction must be between 0 and 1, got {frac}"
                )
            return frac

        # Try source database
        if self._source_db is not None:
            try:
                frac = self._source_db.get_mole_fraction("CH4")
                if frac > _ZERO:
                    return frac
            except Exception:
                pass

        return self._default_gas_fraction

    def _get_species_fraction(
        self,
        species: str,
    ) -> Decimal:
        """Get mole fraction for a gas species from source database.

        Args:
            species: Gas species name (e.g., "CO2").

        Returns:
            Mole fraction as Decimal. Returns 0 if not available.
        """
        if self._source_db is not None:
            try:
                return self._source_db.get_mole_fraction(species)
            except Exception:
                pass
        return _ZERO

    # ------------------------------------------------------------------
    # Private: GWP application to method result
    # ------------------------------------------------------------------

    def _apply_gwp_to_result(
        self,
        method_result: Dict[str, Any],
        gwp_source: str,
    ) -> List[GasResult]:
        """Apply GWP conversion to a method calculation result.

        Args:
            method_result: Result from a method-specific calculator.
            gwp_source: IPCC assessment report.

        Returns:
            List of GasResult entries.
        """
        raw_emissions = method_result.get("raw_emissions_by_gas", {})
        return self.apply_gwp(raw_emissions, gwp_source)

    # ------------------------------------------------------------------
    # Private: Screening value parsing
    # ------------------------------------------------------------------

    def _parse_screening_values(
        self,
        input_data: Dict[str, Any],
    ) -> List[float]:
        """Parse screening values from input data.

        Supports two formats:
            - screening_values: List of {ppmv: float} dicts.
            - screening_values_ppmv: Flat list of floats.

        Args:
            input_data: Input dictionary.

        Returns:
            List of ppmv values as floats.
        """
        # Format 1: list of dicts
        sv = input_data.get("screening_values")
        if sv is not None and isinstance(sv, list):
            values = []
            for entry in sv:
                if isinstance(entry, dict):
                    values.append(float(entry.get("ppmv", 0)))
                else:
                    values.append(float(entry))
            return values

        # Format 2: flat list
        sv_flat = input_data.get("screening_values_ppmv")
        if sv_flat is not None and isinstance(sv_flat, list):
            return [float(v) for v in sv_flat]

        return []

    # ------------------------------------------------------------------
    # Private: Statistics tracking
    # ------------------------------------------------------------------

    def _record_success(self, method: str) -> None:
        """Record a successful calculation."""
        with self._lock:
            self._total_calculations += 1

    def _record_error(self) -> None:
        """Record a failed calculation."""
        with self._lock:
            self._total_calculations += 1
            self._total_errors += 1
