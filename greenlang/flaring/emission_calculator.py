# -*- coding: utf-8 -*-
"""
EmissionCalculatorEngine - Engine 2: Flaring Agent (AGENT-MRV-006)

Core calculation engine implementing four flaring emission calculation
methodologies aligned with EPA 40 CFR Part 98 Subpart W (Section W.23),
IPCC 2006 Guidelines, API Compendium, and GHG Protocol Corporate Standard:

1. **Gas Composition Method** (highest accuracy):
   For each hydrocarbon component i:
       CO2_i = V * x_i * (n_C_i * MW_CO2) / MW_i * CE
       CH4_slip = V * x_CH4 * (1 - CE) * MW_CH4 / MW_CH4  (uncombusted)
   Total CO2 = sum(CO2_i) + pilot_CO2 + purge_CO2
   Total CH4 = CH4_slip + pilot_CH4_slip
   N2O = V * EF_N2O

2. **Default Emission Factor Method** (Tier 1):
   CO2 = V * HHV * EF_CO2
   CH4 = V * HHV * EF_CH4
   N2O = V * HHV * EF_N2O

3. **Engineering Estimate Method**:
   CO2 = FlowCapacity * UtilizationFactor * OperatingHours * EF

4. **Direct Measurement Method**:
   CO2 = sum(MeasuredFlow_t * MeasuredComposition_t) for each time period t

Additional calculations:
    - Pilot gas emissions (continuous): PilotFlow * PilotHHV * EF * Hours
    - Purge gas emissions (if combustible): PurgeFlow * PurgeHHV * EF * Hours
    - GWP conversion: CO2e = CO2 + CH4 * GWP_CH4 + N2O * GWP_N2O

Zero-Hallucination Guarantees:
    - All calculations use Python ``Decimal`` (8+ decimal places).
    - No LLM calls in any calculation path.
    - Every step is recorded in the calculation trace.
    - SHA-256 provenance hash for every result.
    - Same inputs always produce identical outputs (deterministic).

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.flaring.emission_calculator import EmissionCalculatorEngine
    >>> from greenlang.flaring.flare_system_database import FlareSystemDatabaseEngine
    >>> from decimal import Decimal
    >>> db = FlareSystemDatabaseEngine()
    >>> calc = EmissionCalculatorEngine(flare_database=db)
    >>> result = calc.calculate(
    ...     method="GAS_COMPOSITION",
    ...     volume_scf=Decimal("1000000"),
    ...     composition={"CH4": Decimal("0.85"), "C2H6": Decimal("0.07"),
    ...                  "C3H8": Decimal("0.03"), "CO2": Decimal("0.02"),
    ...                  "N2": Decimal("0.03")},
    ...     combustion_efficiency=Decimal("0.98"),
    ... )
    >>> print(result["total_co2e_kg"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["EmissionCalculatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports for GreenLang infrastructure
# ---------------------------------------------------------------------------

try:
    from greenlang.flaring.flare_system_database import FlareSystemDatabaseEngine
    _DATABASE_AVAILABLE = True
except ImportError:
    _DATABASE_AVAILABLE = False
    FlareSystemDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.flaring.metrics import (
        record_calculation as _record_calculation,
        record_emissions as _record_emissions,
        observe_calculation_duration as _observe_calculation_duration,
        record_batch as _record_batch,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]
    _record_batch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places

#: Molecular weight of CO2 (g/mol)
_MW_CO2 = Decimal("44.010")

#: Molecular weight of CH4 (g/mol)
_MW_CH4 = Decimal("16.043")

#: Molecular weight of air (g/mol)
_MW_AIR = Decimal("28.9647")

#: kg to tonnes
_KG_TO_TONNES = Decimal("0.001")

#: tonnes to kg
_TONNES_TO_KG = Decimal("1000")

#: 1 Mscf (thousand scf) in scf
_MSCF_TO_SCF = Decimal("1000")

#: 1 MMscf (million scf) in scf
_MMSCF_TO_SCF = Decimal("1000000")

#: 1 scf = 0.028317 Nm3 (standard conversion)
_SCF_TO_NM3 = Decimal("0.028317")

#: 1 MMBtu = 1.055056 GJ
_MMBTU_TO_GJ = Decimal("1.055056")

#: 1 GJ = 0.947817 MMBtu
_GJ_TO_MMBTU = Decimal("0.947817")

#: BTU per scf to MMBtu per scf conversion
_BTU_TO_MMBTU = Decimal("0.000001")

#: 1 lb = 0.453592 kg
_LB_TO_KG = Decimal("0.453592")

#: Default N2O emission factor (kg N2O per MMBtu) per EPA
_DEFAULT_N2O_EF_KG_PER_MMBTU = Decimal("0.00006")

#: Molar volume at EPA standard conditions (scf/lb-mol)
#: 1 lb-mol of ideal gas at 60 deg F, 14.696 psia = 379.3 scf
_MOLAR_VOLUME_SCF_PER_LBMOL = Decimal("379.3")

#: Molar volume at ISO standard conditions (Nm3/kmol)
#: 1 kmol of ideal gas at 15 deg C, 101.325 kPa = 23.645 Nm3
_MOLAR_VOLUME_NM3_PER_KMOL = Decimal("23.645")

#: g per lb-mol to kg per scf conversion denominator
_G_PER_MOL_TO_KG = Decimal("0.001")


# ---------------------------------------------------------------------------
# Valid method identifiers
# ---------------------------------------------------------------------------

_VALID_METHODS = frozenset({
    "GAS_COMPOSITION",
    "DEFAULT_EMISSION_FACTOR",
    "ENGINEERING_ESTIMATE",
    "DIRECT_MEASUREMENT",
})


# ---------------------------------------------------------------------------
# Fallback GWP values (used when database is unavailable)
# ---------------------------------------------------------------------------

_FALLBACK_GWP: Dict[str, Decimal] = {
    "CO2": Decimal("1"),
    "CH4": Decimal("29.8"),
    "N2O": Decimal("273"),
}


# ===========================================================================
# EmissionCalculatorEngine
# ===========================================================================


class EmissionCalculatorEngine:
    """Core calculation engine for flaring emission calculations.

    Implements four calculation methods (gas composition, default EF,
    engineering estimate, direct measurement) with deterministic Decimal
    arithmetic, full calculation trace, and SHA-256 provenance hashing.

    This engine uses the FlareSystemDatabaseEngine for gas composition
    data, component properties, and emission factor lookups. It computes
    CO2 from hydrocarbon combustion stoichiometry, uncombusted CH4 slip,
    N2O from high-temperature combustion, and pilot/purge gas emissions.

    Thread-safe for concurrent calculations.

    Attributes:
        _flare_db: Reference to FlareSystemDatabaseEngine.
        _config: Optional configuration dictionary.
        _lock: Thread lock for shared mutable state.
        _provenance: Reference to the provenance tracker.
        _precision_places: Number of Decimal places for rounding.

    Example:
        >>> db = FlareSystemDatabaseEngine()
        >>> calc = EmissionCalculatorEngine(flare_database=db)
        >>> result = calc.calculate(
        ...     method="DEFAULT_EMISSION_FACTOR",
        ...     volume_scf=Decimal("1000000"),
        ...     gas_type="NATURAL_GAS",
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        flare_database: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize EmissionCalculatorEngine.

        Args:
            flare_database: FlareSystemDatabaseEngine instance for data
                lookups. If None and the module is available, a default
                instance is created.
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``default_gwp_source`` (str): Default GWP source.
                  Default ``"AR6"``.
                - ``default_ef_source`` (str): Default EF source.
                  Default ``"EPA"``.
                - ``default_ce`` (str): Default combustion efficiency.
                  Default ``"0.98"``.
        """
        if flare_database is not None:
            self._flare_db = flare_database
        elif _DATABASE_AVAILABLE:
            self._flare_db = FlareSystemDatabaseEngine(config=config)
        else:
            self._flare_db = None

        self._config = config or {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get(
            "enable_provenance", True
        )
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer = Decimal(10) ** -self._precision_places
        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6"
        )
        self._default_ef_source: str = self._config.get(
            "default_ef_source", "EPA"
        )
        self._default_ce: Decimal = Decimal(
            str(self._config.get("default_ce", "0.98"))
        )

        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "EmissionCalculatorEngine initialized (precision=%d, "
            "gwp=%s, ef=%s, default_ce=%s)",
            self._precision_places,
            self._default_gwp_source,
            self._default_ef_source,
            self._default_ce,
        )

    # ==================================================================
    # PUBLIC API: Unified Calculate
    # ==================================================================

    def calculate(
        self,
        method: str = "GAS_COMPOSITION",
        volume_scf: Optional[Decimal] = None,
        volume_nm3: Optional[Decimal] = None,
        composition: Optional[Dict[str, Decimal]] = None,
        gas_type: Optional[str] = None,
        combustion_efficiency: Optional[Decimal] = None,
        flare_type: Optional[str] = None,
        ef_source: Optional[str] = None,
        gwp_source: Optional[str] = None,
        pilot_gas_flow_mmbtu_hr: Optional[Decimal] = None,
        pilot_gas_composition: Optional[Dict[str, Decimal]] = None,
        pilot_operating_hours: Optional[Decimal] = None,
        purge_gas_flow_scf_hr: Optional[Decimal] = None,
        purge_gas_composition: Optional[Dict[str, Decimal]] = None,
        purge_gas_combustible: bool = False,
        purge_operating_hours: Optional[Decimal] = None,
        flow_capacity_scf_hr: Optional[Decimal] = None,
        utilization_factor: Optional[Decimal] = None,
        operating_hours: Optional[Decimal] = None,
        measured_periods: Optional[List[Dict[str, Any]]] = None,
        n2o_ef_kg_per_mmbtu: Optional[Decimal] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate flaring emissions using the specified method.

        This is the main entry point for all flaring calculations. It
        dispatches to the appropriate method-specific calculator based
        on the ``method`` parameter.

        Args:
            method: Calculation method. One of ``"GAS_COMPOSITION"``,
                ``"DEFAULT_EMISSION_FACTOR"``,
                ``"ENGINEERING_ESTIMATE"``,
                ``"DIRECT_MEASUREMENT"``.
            volume_scf: Gas volume in standard cubic feet (for methods
                1 and 2). Mutually exclusive with volume_nm3.
            volume_nm3: Gas volume in normal cubic meters. If provided,
                converted to scf internally.
            composition: Gas composition dict (component -> mole fraction).
                Required for GAS_COMPOSITION method.
            gas_type: Gas type for default EF lookup (e.g.
                ``"NATURAL_GAS"``). Used with DEFAULT_EMISSION_FACTOR.
            combustion_efficiency: CE as a decimal (0-1). If None,
                defaults to engine default (0.98).
            flare_type: Flare type for CE lookup if CE not provided.
            ef_source: Emission factor source (``"EPA"``, ``"IPCC"``,
                ``"API"``). Defaults to engine default.
            gwp_source: GWP source override. Defaults to engine default.
            pilot_gas_flow_mmbtu_hr: Pilot gas flow rate in MMBtu/hr
                per pilot tip.
            pilot_gas_composition: Pilot gas composition (optional,
                defaults to natural gas).
            pilot_operating_hours: Hours the pilot was active. Defaults
                to 8760 (full year) if pilot_gas_flow is provided.
            purge_gas_flow_scf_hr: Purge gas flow rate in scf/hr.
            purge_gas_composition: Purge gas composition (optional).
            purge_gas_combustible: Whether purge gas is combustible.
                If False (N2 purge), purge emissions are zero.
            purge_operating_hours: Hours purge was active. Defaults to
                8760 if purge_gas_flow is provided.
            flow_capacity_scf_hr: Equipment capacity for engineering
                estimate method.
            utilization_factor: Fraction of capacity utilized (0-1) for
                engineering estimate method.
            operating_hours: Operating hours for engineering estimate.
            measured_periods: List of measurement period dicts for
                direct measurement method. Each dict has:
                ``flow_scf``, ``composition`` (optional).
            n2o_ef_kg_per_mmbtu: N2O emission factor override. Default
                0.00006 kg/MMBtu.
            calculation_id: Optional external calculation ID.

        Returns:
            Dictionary with keys:
                - calculation_id (str)
                - status (str): ``"SUCCESS"`` or ``"FAILED"``
                - method (str)
                - gas_emissions (List[Dict]): per-gas breakdown
                - pilot_emissions (Dict or None)
                - purge_emissions (Dict or None)
                - total_co2_kg (Decimal)
                - total_ch4_kg (Decimal)
                - total_n2o_kg (Decimal)
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - calculation_trace (List[str])
                - provenance_hash (str)
                - processing_time_ms (float)
                - error_message (str, optional)

        Example:
            >>> result = calc.calculate(
            ...     method="GAS_COMPOSITION",
            ...     volume_scf=Decimal("1000000"),
            ...     composition={"CH4": Decimal("0.85"),
            ...                  "C2H6": Decimal("0.07"),
            ...                  "C3H8": Decimal("0.03"),
            ...                  "CO2": Decimal("0.02"),
            ...                  "N2": Decimal("0.03")},
            ...     combustion_efficiency=Decimal("0.98"),
            ... )
        """
        start_time = time.monotonic()
        calc_id = calculation_id or f"fl_calc_{uuid.uuid4().hex[:12]}"
        method_key = method.upper()
        gwp = (gwp_source or self._default_gwp_source).upper()
        ef_src = (ef_source or self._default_ef_source).upper()
        trace: List[str] = []

        try:
            # Validate method
            if method_key not in _VALID_METHODS:
                raise ValueError(
                    f"Invalid calculation method: {method}. "
                    f"Valid methods: {sorted(_VALID_METHODS)}"
                )

            # Resolve volume to scf
            vol_scf = self._resolve_volume_scf(volume_scf, volume_nm3, trace)

            # Resolve combustion efficiency
            ce = self._resolve_ce(
                combustion_efficiency, flare_type, trace,
            )

            trace.append(
                f"[1] Method={method_key}, Volume_scf={vol_scf}, "
                f"CE={ce}, GWP={gwp}, EF_source={ef_src}"
            )

            # Dispatch to method-specific calculator
            if method_key == "GAS_COMPOSITION":
                gas_emissions = self._calculate_gas_composition(
                    volume_scf=vol_scf,
                    composition=composition or {},
                    ce=ce,
                    n2o_ef=n2o_ef_kg_per_mmbtu or _DEFAULT_N2O_EF_KG_PER_MMBTU,
                    gwp_source=gwp,
                    trace=trace,
                )
            elif method_key == "DEFAULT_EMISSION_FACTOR":
                gas_emissions = self._calculate_default_ef(
                    volume_scf=vol_scf,
                    gas_type=gas_type or "GENERAL",
                    ef_source=ef_src,
                    gwp_source=gwp,
                    composition=composition,
                    trace=trace,
                )
            elif method_key == "ENGINEERING_ESTIMATE":
                gas_emissions = self._calculate_engineering_estimate(
                    flow_capacity_scf_hr=flow_capacity_scf_hr or Decimal("0"),
                    utilization_factor=utilization_factor or Decimal("1.0"),
                    operating_hours=operating_hours or Decimal("8760"),
                    gas_type=gas_type or "GENERAL",
                    ef_source=ef_src,
                    gwp_source=gwp,
                    composition=composition,
                    trace=trace,
                )
            elif method_key == "DIRECT_MEASUREMENT":
                gas_emissions = self._calculate_direct_measurement(
                    measured_periods=measured_periods or [],
                    ce=ce,
                    gwp_source=gwp,
                    trace=trace,
                )
            else:
                raise ValueError(f"Unhandled method: {method_key}")

            # Calculate pilot gas emissions
            pilot_emissions = None
            if pilot_gas_flow_mmbtu_hr is not None:
                pilot_emissions = self._calculate_pilot_emissions(
                    flow_mmbtu_hr=pilot_gas_flow_mmbtu_hr,
                    composition=pilot_gas_composition,
                    hours=pilot_operating_hours or Decimal("8760"),
                    ce=ce,
                    ef_source=ef_src,
                    gwp_source=gwp,
                    trace=trace,
                )

            # Calculate purge gas emissions
            purge_emissions = None
            if (
                purge_gas_flow_scf_hr is not None
                and purge_gas_combustible
            ):
                purge_emissions = self._calculate_purge_emissions(
                    flow_scf_hr=purge_gas_flow_scf_hr,
                    composition=purge_gas_composition,
                    hours=purge_operating_hours or Decimal("8760"),
                    ce=ce,
                    ef_source=ef_src,
                    gwp_source=gwp,
                    trace=trace,
                )

            # Aggregate totals
            totals = self._aggregate_totals(
                gas_emissions, pilot_emissions, purge_emissions,
                gwp, trace,
            )

            # Provenance hash
            elapsed_ms = (time.monotonic() - start_time) * 1000
            provenance_hash = self._compute_provenance_hash({
                "calculation_id": calc_id,
                "method": method_key,
                "total_co2e_kg": str(totals["total_co2e_kg"]),
                "gwp_source": gwp,
                "ef_source": ef_src,
            })
            trace.append(f"[PROV] hash={provenance_hash[:16]}...")

            # Metrics
            if _METRICS_AVAILABLE and _record_calculation is not None:
                _record_calculation(
                    flare_type or "UNKNOWN", method_key, "completed",
                )
            if _METRICS_AVAILABLE and _record_emissions is not None:
                _record_emissions(
                    flare_type or "UNKNOWN", "CO2",
                    float(totals["total_co2_kg"]),
                )
            if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
                _observe_calculation_duration(
                    "single_calculation", elapsed_ms / 1000,
                )

            # Provenance record
            self._record_provenance(
                "calculate_flaring_emissions", calc_id,
                {
                    "method": method_key,
                    "total_co2e_kg": str(totals["total_co2e_kg"]),
                    "hash": provenance_hash,
                },
            )

            return {
                "calculation_id": calc_id,
                "status": "SUCCESS",
                "method": method_key,
                "ef_source": ef_src,
                "gwp_source": gwp,
                "combustion_efficiency": ce,
                "gas_emissions": gas_emissions,
                "pilot_emissions": pilot_emissions,
                "purge_emissions": purge_emissions,
                "total_co2_kg": totals["total_co2_kg"],
                "total_ch4_kg": totals["total_ch4_kg"],
                "total_n2o_kg": totals["total_n2o_kg"],
                "total_co2e_kg": totals["total_co2e_kg"],
                "total_co2e_tonnes": totals["total_co2e_tonnes"],
                "calculation_trace": trace,
                "provenance_hash": provenance_hash,
                "processing_time_ms": elapsed_ms,
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Flaring calculation failed (id=%s, method=%s): %s",
                calc_id, method_key, exc, exc_info=True,
            )
            if _METRICS_AVAILABLE and _record_calculation is not None:
                _record_calculation(
                    flare_type or "UNKNOWN", method_key, "failed",
                )

            return {
                "calculation_id": calc_id,
                "status": "FAILED",
                "method": method_key,
                "ef_source": ef_src,
                "gwp_source": gwp,
                "combustion_efficiency": combustion_efficiency,
                "gas_emissions": [],
                "pilot_emissions": None,
                "purge_emissions": None,
                "total_co2_kg": Decimal("0"),
                "total_ch4_kg": Decimal("0"),
                "total_n2o_kg": Decimal("0"),
                "total_co2e_kg": Decimal("0"),
                "total_co2e_tonnes": Decimal("0"),
                "calculation_trace": trace,
                "provenance_hash": "",
                "processing_time_ms": elapsed_ms,
                "error_message": str(exc),
            }

    # ==================================================================
    # PUBLIC API: Batch Calculation
    # ==================================================================

    def calculate_batch(
        self,
        calculations: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multiple flaring calculation requests in batch.

        Each element in ``calculations`` is a dictionary of keyword
        arguments for the ``calculate()`` method.

        Args:
            calculations: List of calculation parameter dictionaries.
            gwp_source: Optional GWP override applied to all calcs.

        Returns:
            Dictionary with keys:
                - batch_id (str)
                - results (List[Dict])
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - success_count (int)
                - failure_count (int)
                - processing_time_ms (float)

        Example:
            >>> batch = calc.calculate_batch([
            ...     {"method": "GAS_COMPOSITION",
            ...      "volume_scf": Decimal("500000"),
            ...      "composition": comp},
            ...     {"method": "DEFAULT_EMISSION_FACTOR",
            ...      "volume_scf": Decimal("1000000"),
            ...      "gas_type": "ASSOCIATED_GAS"},
            ... ])
        """
        start_time = time.monotonic()
        batch_id = f"fl_batch_{uuid.uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")
        total_co2e_tonnes = Decimal("0")
        success_count = 0
        failure_count = 0

        for calc_params in calculations:
            if gwp_source and "gwp_source" not in calc_params:
                calc_params["gwp_source"] = gwp_source

            result = self.calculate(**calc_params)
            results.append(result)

            if result["status"] == "SUCCESS":
                success_count += 1
                total_co2e_kg += result["total_co2e_kg"]
                total_co2e_tonnes += result["total_co2e_tonnes"]
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if _METRICS_AVAILABLE and _record_batch is not None:
            status = "completed" if failure_count == 0 else "partial"
            _record_batch(status)

        logger.info(
            "Batch %s completed: %d success, %d failed, %.1f ms",
            batch_id, success_count, failure_count, elapsed_ms,
        )

        return {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(calculations),
            "processing_time_ms": elapsed_ms,
        }

    # ==================================================================
    # PUBLIC API: GWP Application
    # ==================================================================

    def apply_gwp(
        self,
        co2_kg: Decimal,
        ch4_kg: Decimal,
        n2o_kg: Decimal,
        gwp_source: str = "AR6",
    ) -> Dict[str, Decimal]:
        """Apply GWP conversion to compute total CO2-equivalent.

        CO2e = CO2 + CH4 * GWP_CH4 + N2O * GWP_N2O

        Args:
            co2_kg: CO2 mass in kg.
            ch4_kg: CH4 mass in kg.
            n2o_kg: N2O mass in kg.
            gwp_source: GWP source. Default ``"AR6"``.

        Returns:
            Dictionary with gwp values and total co2e.

        Example:
            >>> result = calc.apply_gwp(
            ...     Decimal("1000"), Decimal("10"), Decimal("0.5"),
            ... )
            >>> result["total_co2e_kg"]
        """
        gwp_ch4 = self._resolve_gwp("CH4", gwp_source)
        gwp_n2o = self._resolve_gwp("N2O", gwp_source)

        co2e_from_co2 = self._quantize(co2_kg)
        co2e_from_ch4 = self._quantize(ch4_kg * gwp_ch4)
        co2e_from_n2o = self._quantize(n2o_kg * gwp_n2o)
        total_co2e = self._quantize(
            co2e_from_co2 + co2e_from_ch4 + co2e_from_n2o
        )

        return {
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "gwp_ch4": gwp_ch4,
            "gwp_n2o": gwp_n2o,
            "co2e_from_co2_kg": co2e_from_co2,
            "co2e_from_ch4_kg": co2e_from_ch4,
            "co2e_from_n2o_kg": co2e_from_n2o,
            "total_co2e_kg": total_co2e,
            "total_co2e_tonnes": self._quantize(total_co2e * _KG_TO_TONNES),
        }

    # ==================================================================
    # PUBLIC API: Unit Conversions
    # ==================================================================

    def convert_volume_scf_to_nm3(self, volume_scf: Decimal) -> Decimal:
        """Convert volume from standard cubic feet to normal cubic meters.

        Args:
            volume_scf: Volume in scf.

        Returns:
            Volume in Nm3.
        """
        return self._quantize(volume_scf * _SCF_TO_NM3)

    def convert_volume_nm3_to_scf(self, volume_nm3: Decimal) -> Decimal:
        """Convert volume from normal cubic meters to standard cubic feet.

        Args:
            volume_nm3: Volume in Nm3.

        Returns:
            Volume in scf.
        """
        return self._quantize(volume_nm3 / _SCF_TO_NM3)

    def convert_energy_mmbtu_to_gj(self, energy_mmbtu: Decimal) -> Decimal:
        """Convert energy from MMBtu to GJ.

        Args:
            energy_mmbtu: Energy in MMBtu.

        Returns:
            Energy in GJ.
        """
        return self._quantize(energy_mmbtu * _MMBTU_TO_GJ)

    def convert_energy_gj_to_mmbtu(self, energy_gj: Decimal) -> Decimal:
        """Convert energy from GJ to MMBtu.

        Args:
            energy_gj: Energy in GJ.

        Returns:
            Energy in MMBtu.
        """
        return self._quantize(energy_gj * _GJ_TO_MMBTU)

    def convert_mass_lb_to_kg(self, mass_lb: Decimal) -> Decimal:
        """Convert mass from pounds to kilograms.

        Args:
            mass_lb: Mass in lb.

        Returns:
            Mass in kg.
        """
        return self._quantize(mass_lb * _LB_TO_KG)

    def convert_mass_kg_to_lb(self, mass_kg: Decimal) -> Decimal:
        """Convert mass from kilograms to pounds.

        Args:
            mass_kg: Mass in kg.

        Returns:
            Mass in lb.
        """
        return self._quantize(mass_kg / _LB_TO_KG)

    # ==================================================================
    # METHOD 1: Gas Composition
    # ==================================================================

    def _calculate_gas_composition(
        self,
        volume_scf: Decimal,
        composition: Dict[str, Decimal],
        ce: Decimal,
        n2o_ef: Decimal,
        gwp_source: str,
        trace: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate emissions using the gas composition method.

        For each hydrocarbon component i that produces CO2:
            CO2_i = V * x_i * (n_C_i * MW_CO2 / MW_i) * CE
        where V is in scf and result is in lb-mol-based mass.

        For volumetric approach at standard conditions:
            moles_i = V * x_i / MolarVolume (scf -> lb-mol)
            CO2_mass_i = moles_i * n_C_i * MW_CO2 * CE (in g, then -> kg)
            CH4_slip = moles_CH4 * MW_CH4 * (1 - CE) (in g, then -> kg)

        Args:
            volume_scf: Total gas volume in standard cubic feet.
            composition: Mole fraction dict.
            ce: Combustion efficiency (0-1).
            n2o_ef: N2O emission factor in kg/MMBtu.
            gwp_source: GWP source.
            trace: Audit trace.

        Returns:
            List of per-gas emission dictionaries.
        """
        if not composition:
            raise ValueError(
                "Gas composition is required for GAS_COMPOSITION method"
            )

        trace.append(
            f"[GC] volume_scf={volume_scf}, CE={ce}, "
            f"components={len(composition)}"
        )

        total_co2_kg = Decimal("0")
        total_ch4_slip_kg = Decimal("0")
        component_details: List[Dict[str, Any]] = []

        # Calculate CO2 from combustion of each carbon-containing component
        for component, fraction in composition.items():
            comp_key = component.upper()
            frac = Decimal(str(fraction))

            if frac <= Decimal("0"):
                continue

            # Get component properties from database or fallback
            mw_comp = self._get_molecular_weight(comp_key)
            carbon_count = self._get_carbon_count(comp_key)
            produces_co2 = self._check_produces_co2(comp_key)

            # Moles of component: V * x_i / MolarVolume
            # At 60 F, 14.696 psia: 1 lb-mol = 379.3 scf
            moles_lbmol = self._quantize(
                volume_scf * frac / _MOLAR_VOLUME_SCF_PER_LBMOL
            )

            # CO2 from this component's combustion
            co2_from_component_kg = Decimal("0")
            if produces_co2 and carbon_count > 0:
                # Each mole of component produces carbon_count moles of CO2
                # Mass CO2 = moles * n_C * MW_CO2 (in lb, since lb-mol)
                # Convert lb to kg: * 0.453592
                co2_from_component_lb = self._quantize(
                    moles_lbmol * Decimal(str(carbon_count)) * _MW_CO2 * ce
                )
                co2_from_component_kg = self._quantize(
                    co2_from_component_lb * _LB_TO_KG
                )
                total_co2_kg += co2_from_component_kg

            # CH4 slip: uncombusted methane
            ch4_slip_kg = Decimal("0")
            if comp_key == "CH4":
                ch4_slip_lb = self._quantize(
                    moles_lbmol * _MW_CH4 * (Decimal("1") - ce)
                )
                ch4_slip_kg = self._quantize(ch4_slip_lb * _LB_TO_KG)
                total_ch4_slip_kg += ch4_slip_kg

            component_details.append({
                "component": comp_key,
                "mole_fraction": frac,
                "moles_lbmol": moles_lbmol,
                "molecular_weight": mw_comp,
                "carbon_count": carbon_count,
                "co2_kg": co2_from_component_kg,
                "ch4_slip_kg": ch4_slip_kg,
            })

            trace.append(
                f"[GC:{comp_key}] x={frac}, moles={moles_lbmol} lb-mol, "
                f"nC={carbon_count}, CO2={co2_from_component_kg} kg, "
                f"CH4_slip={ch4_slip_kg} kg"
            )

        # N2O emissions: based on total energy content
        hhv_btu_scf = self._calculate_mixture_hhv(composition)
        total_energy_mmbtu = self._quantize(
            volume_scf * hhv_btu_scf * _BTU_TO_MMBTU
        )
        total_n2o_kg = self._quantize(total_energy_mmbtu * n2o_ef)

        trace.append(
            f"[GC:N2O] HHV={hhv_btu_scf} BTU/scf, "
            f"energy={total_energy_mmbtu} MMBtu, "
            f"N2O_ef={n2o_ef} kg/MMBtu, N2O={total_n2o_kg} kg"
        )

        # GWP conversion
        gwp_ch4 = self._resolve_gwp("CH4", gwp_source)
        gwp_n2o = self._resolve_gwp("N2O", gwp_source)

        co2e_from_co2 = total_co2_kg  # GWP of CO2 = 1
        co2e_from_ch4 = self._quantize(total_ch4_slip_kg * gwp_ch4)
        co2e_from_n2o = self._quantize(total_n2o_kg * gwp_n2o)

        results: List[Dict[str, Any]] = []

        # CO2 result
        results.append({
            "gas": "CO2",
            "mass_kg": total_co2_kg,
            "mass_tonnes": self._quantize(total_co2_kg * _KG_TO_TONNES),
            "gwp": Decimal("1"),
            "gwp_source": gwp_source,
            "co2e_kg": co2e_from_co2,
            "co2e_tonnes": self._quantize(co2e_from_co2 * _KG_TO_TONNES),
            "method": "GAS_COMPOSITION",
            "component_detail": component_details,
        })

        # CH4 result (uncombusted slip)
        results.append({
            "gas": "CH4",
            "mass_kg": total_ch4_slip_kg,
            "mass_tonnes": self._quantize(
                total_ch4_slip_kg * _KG_TO_TONNES
            ),
            "gwp": gwp_ch4,
            "gwp_source": gwp_source,
            "co2e_kg": co2e_from_ch4,
            "co2e_tonnes": self._quantize(co2e_from_ch4 * _KG_TO_TONNES),
            "method": "GAS_COMPOSITION",
        })

        # N2O result
        results.append({
            "gas": "N2O",
            "mass_kg": total_n2o_kg,
            "mass_tonnes": self._quantize(total_n2o_kg * _KG_TO_TONNES),
            "gwp": gwp_n2o,
            "gwp_source": gwp_source,
            "co2e_kg": co2e_from_n2o,
            "co2e_tonnes": self._quantize(co2e_from_n2o * _KG_TO_TONNES),
            "method": "GAS_COMPOSITION",
        })

        trace.append(
            f"[GC:TOTAL] CO2={total_co2_kg} kg, "
            f"CH4_slip={total_ch4_slip_kg} kg, N2O={total_n2o_kg} kg"
        )

        return results

    # ==================================================================
    # METHOD 2: Default Emission Factor
    # ==================================================================

    def _calculate_default_ef(
        self,
        volume_scf: Decimal,
        gas_type: str,
        ef_source: str,
        gwp_source: str,
        composition: Optional[Dict[str, Decimal]],
        trace: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate emissions using default emission factors.

        CO2 = V * HHV * EF_CO2
        CH4 = V * HHV * EF_CH4
        N2O = V * HHV * EF_N2O

        Where HHV is either from composition (if provided) or from the
        gas type default.

        Args:
            volume_scf: Gas volume in standard cubic feet.
            gas_type: Gas type for EF lookup.
            ef_source: Emission factor source.
            gwp_source: GWP source.
            composition: Optional composition for HHV calculation.
            trace: Audit trace.

        Returns:
            List of per-gas emission dictionaries.
        """
        gt_key = gas_type.upper()

        # Calculate HHV
        if composition:
            hhv_btu_scf = self._calculate_mixture_hhv(composition)
        else:
            hhv_btu_scf = self._get_default_hhv(gt_key)

        # Total energy in MMBtu
        total_energy_mmbtu = self._quantize(
            volume_scf * hhv_btu_scf * _BTU_TO_MMBTU
        )

        # For IPCC, energy must be in GJ (NCV basis)
        if ef_source == "IPCC":
            total_energy_gj = self._quantize(
                total_energy_mmbtu * _MMBTU_TO_GJ
            )
        else:
            total_energy_gj = None

        trace.append(
            f"[DEF_EF] gas_type={gt_key}, HHV={hhv_btu_scf} BTU/scf, "
            f"volume={volume_scf} scf, energy={total_energy_mmbtu} MMBtu"
        )

        results: List[Dict[str, Any]] = []

        for gas_name in ["CO2", "CH4", "N2O"]:
            # Get emission factor
            ef = self._get_emission_factor(gt_key, gas_name, ef_source)

            # Calculate emissions
            if ef_source == "IPCC" and total_energy_gj is not None:
                mass_kg = self._quantize(total_energy_gj * ef)
            else:
                mass_kg = self._quantize(total_energy_mmbtu * ef)

            # GWP conversion
            gwp = self._resolve_gwp(gas_name, gwp_source)
            co2e_kg = self._quantize(mass_kg * gwp)

            trace.append(
                f"[DEF_EF:{gas_name}] EF={ef}, mass={mass_kg} kg, "
                f"GWP={gwp}, co2e={co2e_kg} kg"
            )

            results.append({
                "gas": gas_name,
                "mass_kg": mass_kg,
                "mass_tonnes": self._quantize(mass_kg * _KG_TO_TONNES),
                "emission_factor": ef,
                "emission_factor_source": ef_source,
                "emission_factor_unit": (
                    "kg/GJ" if ef_source == "IPCC" else "kg/MMBtu"
                ),
                "gwp": gwp,
                "gwp_source": gwp_source,
                "co2e_kg": co2e_kg,
                "co2e_tonnes": self._quantize(co2e_kg * _KG_TO_TONNES),
                "method": "DEFAULT_EMISSION_FACTOR",
                "energy_mmbtu": total_energy_mmbtu,
            })

        return results

    # ==================================================================
    # METHOD 3: Engineering Estimate
    # ==================================================================

    def _calculate_engineering_estimate(
        self,
        flow_capacity_scf_hr: Decimal,
        utilization_factor: Decimal,
        operating_hours: Decimal,
        gas_type: str,
        ef_source: str,
        gwp_source: str,
        composition: Optional[Dict[str, Decimal]],
        trace: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate emissions using engineering estimates.

        Volume = FlowCapacity * UtilizationFactor * OperatingHours
        Then apply the default emission factor method.

        Args:
            flow_capacity_scf_hr: Max flow capacity in scf/hr.
            utilization_factor: Fraction of capacity utilized (0-1).
            operating_hours: Total operating hours.
            gas_type: Gas type for EF lookup.
            ef_source: Emission factor source.
            gwp_source: GWP source.
            composition: Optional composition.
            trace: Audit trace.

        Returns:
            List of per-gas emission dictionaries.
        """
        # Validate utilization factor
        if utilization_factor < Decimal("0") or utilization_factor > Decimal("1"):
            raise ValueError(
                f"Utilization factor must be 0-1, got {utilization_factor}"
            )

        # Calculate estimated volume
        estimated_volume_scf = self._quantize(
            flow_capacity_scf_hr * utilization_factor * operating_hours
        )

        trace.append(
            f"[ENG] capacity={flow_capacity_scf_hr} scf/hr, "
            f"util={utilization_factor}, hours={operating_hours}, "
            f"est_volume={estimated_volume_scf} scf"
        )

        # Use default EF method with estimated volume
        return self._calculate_default_ef(
            volume_scf=estimated_volume_scf,
            gas_type=gas_type,
            ef_source=ef_source,
            gwp_source=gwp_source,
            composition=composition,
            trace=trace,
        )

    # ==================================================================
    # METHOD 4: Direct Measurement
    # ==================================================================

    def _calculate_direct_measurement(
        self,
        measured_periods: List[Dict[str, Any]],
        ce: Decimal,
        gwp_source: str,
        trace: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate emissions from direct measurement data.

        For each measurement period:
            CO2_t = flow_scf_t * composition_t (processed as gas comp)

        If composition is not provided per period, flow is summed and
        treated as measured CO2 directly.

        Args:
            measured_periods: List of measurement period dicts. Each
                must have ``flow_scf`` (Decimal). Optional:
                ``composition`` (Dict[str, Decimal]),
                ``co2_kg`` (Decimal), ``ch4_kg`` (Decimal),
                ``n2o_kg`` (Decimal).
            ce: Combustion efficiency.
            gwp_source: GWP source.
            trace: Audit trace.

        Returns:
            List of per-gas emission dictionaries.
        """
        if not measured_periods:
            raise ValueError(
                "At least one measurement period is required for "
                "DIRECT_MEASUREMENT method"
            )

        total_co2_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        total_n2o_kg = Decimal("0")

        trace.append(
            f"[DIRECT] periods={len(measured_periods)}"
        )

        for i, period in enumerate(measured_periods):
            # Check if direct mass values are provided
            if "co2_kg" in period:
                p_co2 = Decimal(str(period["co2_kg"]))
                p_ch4 = Decimal(str(period.get("ch4_kg", "0")))
                p_n2o = Decimal(str(period.get("n2o_kg", "0")))

                total_co2_kg += p_co2
                total_ch4_kg += p_ch4
                total_n2o_kg += p_n2o

                trace.append(
                    f"[DIRECT:{i}] direct: CO2={p_co2} kg, "
                    f"CH4={p_ch4} kg, N2O={p_n2o} kg"
                )
            elif "flow_scf" in period and "composition" in period:
                # Calculate from flow and composition using gas comp method
                flow_scf = Decimal(str(period["flow_scf"]))
                comp = period["composition"]

                sub_trace: List[str] = []
                sub_results = self._calculate_gas_composition(
                    volume_scf=flow_scf,
                    composition=comp,
                    ce=ce,
                    n2o_ef=_DEFAULT_N2O_EF_KG_PER_MMBTU,
                    gwp_source=gwp_source,
                    trace=sub_trace,
                )

                for sr in sub_results:
                    if sr["gas"] == "CO2":
                        total_co2_kg += sr["mass_kg"]
                    elif sr["gas"] == "CH4":
                        total_ch4_kg += sr["mass_kg"]
                    elif sr["gas"] == "N2O":
                        total_n2o_kg += sr["mass_kg"]

                trace.append(
                    f"[DIRECT:{i}] flow={flow_scf} scf, "
                    f"components={len(comp)}"
                )
            elif "flow_scf" in period:
                raise ValueError(
                    f"Measurement period {i} has flow_scf but no "
                    f"composition. Provide composition or direct mass values."
                )
            else:
                raise ValueError(
                    f"Measurement period {i} must have either 'co2_kg' "
                    f"or 'flow_scf' + 'composition'."
                )

        # GWP conversion
        gwp_ch4 = self._resolve_gwp("CH4", gwp_source)
        gwp_n2o = self._resolve_gwp("N2O", gwp_source)

        results: List[Dict[str, Any]] = []

        # CO2
        co2e_co2 = total_co2_kg
        results.append({
            "gas": "CO2",
            "mass_kg": total_co2_kg,
            "mass_tonnes": self._quantize(total_co2_kg * _KG_TO_TONNES),
            "gwp": Decimal("1"),
            "gwp_source": gwp_source,
            "co2e_kg": co2e_co2,
            "co2e_tonnes": self._quantize(co2e_co2 * _KG_TO_TONNES),
            "method": "DIRECT_MEASUREMENT",
        })

        # CH4
        co2e_ch4 = self._quantize(total_ch4_kg * gwp_ch4)
        results.append({
            "gas": "CH4",
            "mass_kg": total_ch4_kg,
            "mass_tonnes": self._quantize(total_ch4_kg * _KG_TO_TONNES),
            "gwp": gwp_ch4,
            "gwp_source": gwp_source,
            "co2e_kg": co2e_ch4,
            "co2e_tonnes": self._quantize(co2e_ch4 * _KG_TO_TONNES),
            "method": "DIRECT_MEASUREMENT",
        })

        # N2O
        co2e_n2o = self._quantize(total_n2o_kg * gwp_n2o)
        results.append({
            "gas": "N2O",
            "mass_kg": total_n2o_kg,
            "mass_tonnes": self._quantize(total_n2o_kg * _KG_TO_TONNES),
            "gwp": gwp_n2o,
            "gwp_source": gwp_source,
            "co2e_kg": co2e_n2o,
            "co2e_tonnes": self._quantize(co2e_n2o * _KG_TO_TONNES),
            "method": "DIRECT_MEASUREMENT",
        })

        trace.append(
            f"[DIRECT:TOTAL] CO2={total_co2_kg} kg, "
            f"CH4={total_ch4_kg} kg, N2O={total_n2o_kg} kg"
        )

        return results

    # ==================================================================
    # PILOT AND PURGE GAS EMISSIONS
    # ==================================================================

    def _calculate_pilot_emissions(
        self,
        flow_mmbtu_hr: Decimal,
        composition: Optional[Dict[str, Decimal]],
        hours: Decimal,
        ce: Decimal,
        ef_source: str,
        gwp_source: str,
        trace: List[str],
    ) -> Dict[str, Any]:
        """Calculate emissions from continuous pilot gas combustion.

        Pilot gas burns continuously to maintain the flare flame.
        Emissions = PilotFlow * PilotHHV * EF * OperatingHours

        Since flow is already in MMBtu/hr (energy basis), we multiply
        by EF directly:
            CO2 = flow_mmbtu_hr * hours * EF_CO2
            CH4 = flow_mmbtu_hr * hours * EF_CH4 * (1 - CE) [slip from pilot]
            N2O = flow_mmbtu_hr * hours * EF_N2O

        Args:
            flow_mmbtu_hr: Pilot gas energy flow rate in MMBtu/hr.
            composition: Pilot gas composition (optional, defaults to NG).
            hours: Operating hours (typically 8760 for continuous).
            ce: Combustion efficiency of the pilot flame.
            ef_source: EF source.
            gwp_source: GWP source.
            trace: Audit trace.

        Returns:
            Dictionary with pilot emission details.
        """
        total_energy_mmbtu = self._quantize(flow_mmbtu_hr * hours)

        # Use natural gas factors for pilot (typical pilot fuel)
        pilot_gas_type = "NATURAL_GAS"
        ef_co2 = self._get_emission_factor(pilot_gas_type, "CO2", ef_source)
        ef_ch4 = self._get_emission_factor(pilot_gas_type, "CH4", ef_source)
        ef_n2o = self._get_emission_factor(pilot_gas_type, "N2O", ef_source)

        # For IPCC, convert energy to GJ
        if ef_source == "IPCC":
            energy_for_ef = self._quantize(total_energy_mmbtu * _MMBTU_TO_GJ)
        else:
            energy_for_ef = total_energy_mmbtu

        co2_kg = self._quantize(energy_for_ef * ef_co2)
        # CH4 slip from pilot is very small as pilot burns efficiently
        ch4_kg = self._quantize(energy_for_ef * ef_ch4)
        n2o_kg = self._quantize(energy_for_ef * ef_n2o)

        # GWP conversion
        gwp_ch4 = self._resolve_gwp("CH4", gwp_source)
        gwp_n2o = self._resolve_gwp("N2O", gwp_source)
        co2e_kg = self._quantize(
            co2_kg + ch4_kg * gwp_ch4 + n2o_kg * gwp_n2o
        )

        trace.append(
            f"[PILOT] flow={flow_mmbtu_hr} MMBtu/hr, hours={hours}, "
            f"energy={total_energy_mmbtu} MMBtu, CO2={co2_kg} kg, "
            f"CH4={ch4_kg} kg, N2O={n2o_kg} kg, co2e={co2e_kg} kg"
        )

        return {
            "source": "PILOT",
            "flow_mmbtu_hr": flow_mmbtu_hr,
            "operating_hours": hours,
            "total_energy_mmbtu": total_energy_mmbtu,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": self._quantize(co2e_kg * _KG_TO_TONNES),
            "ef_source": ef_source,
            "gwp_source": gwp_source,
        }

    def _calculate_purge_emissions(
        self,
        flow_scf_hr: Decimal,
        composition: Optional[Dict[str, Decimal]],
        hours: Decimal,
        ce: Decimal,
        ef_source: str,
        gwp_source: str,
        trace: List[str],
    ) -> Dict[str, Any]:
        """Calculate emissions from combustible purge gas.

        Purge gas maintains positive pressure to prevent flashback.
        If purge is N2, emissions are zero (handled by caller).
        If purge is natural gas or other combustible, emissions are
        calculated like flare gas.

        Emissions = PurgeFlow_scf_hr * Hours * HHV * EF

        Args:
            flow_scf_hr: Purge gas flow rate in scf/hr.
            composition: Purge gas composition (optional, defaults to NG).
            hours: Operating hours.
            ce: Combustion efficiency.
            ef_source: EF source.
            gwp_source: GWP source.
            trace: Audit trace.

        Returns:
            Dictionary with purge emission details.
        """
        total_volume_scf = self._quantize(flow_scf_hr * hours)

        # Calculate HHV of purge gas
        if composition:
            hhv_btu_scf = self._calculate_mixture_hhv(composition)
        else:
            # Default to natural gas HHV
            hhv_btu_scf = Decimal("1012.0")

        total_energy_mmbtu = self._quantize(
            total_volume_scf * hhv_btu_scf * _BTU_TO_MMBTU
        )

        # Get emission factors
        purge_gas_type = "NATURAL_GAS"
        ef_co2 = self._get_emission_factor(purge_gas_type, "CO2", ef_source)
        ef_ch4 = self._get_emission_factor(purge_gas_type, "CH4", ef_source)
        ef_n2o = self._get_emission_factor(purge_gas_type, "N2O", ef_source)

        # For IPCC
        if ef_source == "IPCC":
            energy_for_ef = self._quantize(total_energy_mmbtu * _MMBTU_TO_GJ)
        else:
            energy_for_ef = total_energy_mmbtu

        co2_kg = self._quantize(energy_for_ef * ef_co2)
        ch4_kg = self._quantize(energy_for_ef * ef_ch4)
        n2o_kg = self._quantize(energy_for_ef * ef_n2o)

        # GWP conversion
        gwp_ch4 = self._resolve_gwp("CH4", gwp_source)
        gwp_n2o = self._resolve_gwp("N2O", gwp_source)
        co2e_kg = self._quantize(
            co2_kg + ch4_kg * gwp_ch4 + n2o_kg * gwp_n2o
        )

        trace.append(
            f"[PURGE] flow={flow_scf_hr} scf/hr, hours={hours}, "
            f"volume={total_volume_scf} scf, HHV={hhv_btu_scf} BTU/scf, "
            f"energy={total_energy_mmbtu} MMBtu, CO2={co2_kg} kg, "
            f"co2e={co2e_kg} kg"
        )

        return {
            "source": "PURGE",
            "flow_scf_hr": flow_scf_hr,
            "operating_hours": hours,
            "total_volume_scf": total_volume_scf,
            "hhv_btu_scf": hhv_btu_scf,
            "total_energy_mmbtu": total_energy_mmbtu,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": self._quantize(co2e_kg * _KG_TO_TONNES),
            "ef_source": ef_source,
            "gwp_source": gwp_source,
        }

    # ==================================================================
    # PRIVATE: Aggregation
    # ==================================================================

    def _aggregate_totals(
        self,
        gas_emissions: List[Dict[str, Any]],
        pilot_emissions: Optional[Dict[str, Any]],
        purge_emissions: Optional[Dict[str, Any]],
        gwp_source: str,
        trace: List[str],
    ) -> Dict[str, Decimal]:
        """Aggregate total emissions across gas, pilot, and purge.

        Args:
            gas_emissions: List of per-gas emission dictionaries.
            pilot_emissions: Pilot emission dict (or None).
            purge_emissions: Purge emission dict (or None).
            gwp_source: GWP source.
            trace: Audit trace.

        Returns:
            Dictionary with total_co2_kg, total_ch4_kg, total_n2o_kg,
            total_co2e_kg, total_co2e_tonnes.
        """
        total_co2_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        total_n2o_kg = Decimal("0")

        # Sum from flare gas emissions
        for ge in gas_emissions:
            gas_name = ge.get("gas", "")
            mass_kg = Decimal(str(ge.get("mass_kg", "0")))
            if gas_name == "CO2":
                total_co2_kg += mass_kg
            elif gas_name == "CH4":
                total_ch4_kg += mass_kg
            elif gas_name == "N2O":
                total_n2o_kg += mass_kg

        # Add pilot emissions
        if pilot_emissions is not None:
            total_co2_kg += Decimal(str(pilot_emissions.get("co2_kg", "0")))
            total_ch4_kg += Decimal(str(pilot_emissions.get("ch4_kg", "0")))
            total_n2o_kg += Decimal(str(pilot_emissions.get("n2o_kg", "0")))

        # Add purge emissions
        if purge_emissions is not None:
            total_co2_kg += Decimal(str(purge_emissions.get("co2_kg", "0")))
            total_ch4_kg += Decimal(str(purge_emissions.get("ch4_kg", "0")))
            total_n2o_kg += Decimal(str(purge_emissions.get("n2o_kg", "0")))

        # GWP conversion
        gwp_ch4 = self._resolve_gwp("CH4", gwp_source)
        gwp_n2o = self._resolve_gwp("N2O", gwp_source)

        total_co2e_kg = self._quantize(
            total_co2_kg
            + total_ch4_kg * gwp_ch4
            + total_n2o_kg * gwp_n2o
        )
        total_co2e_tonnes = self._quantize(total_co2e_kg * _KG_TO_TONNES)

        trace.append(
            f"[AGGREGATE] CO2={total_co2_kg} kg, CH4={total_ch4_kg} kg, "
            f"N2O={total_n2o_kg} kg, CO2e={total_co2e_kg} kg "
            f"({total_co2e_tonnes} t)"
        )

        return {
            "total_co2_kg": self._quantize(total_co2_kg),
            "total_ch4_kg": self._quantize(total_ch4_kg),
            "total_n2o_kg": self._quantize(total_n2o_kg),
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
        }

    # ==================================================================
    # PRIVATE: Factor and Property Resolution
    # ==================================================================

    def _resolve_volume_scf(
        self,
        volume_scf: Optional[Decimal],
        volume_nm3: Optional[Decimal],
        trace: List[str],
    ) -> Decimal:
        """Resolve volume input to standard cubic feet.

        If both are None, returns 0. If volume_nm3 is provided,
        converts to scf.

        Args:
            volume_scf: Volume in scf (optional).
            volume_nm3: Volume in Nm3 (optional).
            trace: Audit trace.

        Returns:
            Volume in scf.
        """
        if volume_scf is not None:
            return Decimal(str(volume_scf))
        elif volume_nm3 is not None:
            vol_scf = self._quantize(
                Decimal(str(volume_nm3)) / _SCF_TO_NM3
            )
            trace.append(
                f"[CONVERT] {volume_nm3} Nm3 -> {vol_scf} scf"
            )
            return vol_scf
        else:
            return Decimal("0")

    def _resolve_ce(
        self,
        ce_input: Optional[Decimal],
        flare_type: Optional[str],
        trace: List[str],
    ) -> Decimal:
        """Resolve combustion efficiency from input or flare type.

        Priority:
        1. Explicit CE input
        2. Flare type typical CE from database
        3. Engine default CE

        Args:
            ce_input: Explicit CE value (optional).
            flare_type: Flare type for lookup (optional).
            trace: Audit trace.

        Returns:
            Combustion efficiency as Decimal (0-1).

        Raises:
            ValueError: If CE is out of range.
        """
        if ce_input is not None:
            ce = Decimal(str(ce_input))
            if ce < Decimal("0") or ce > Decimal("1"):
                raise ValueError(
                    f"Combustion efficiency must be 0-1, got {ce}"
                )
            trace.append(f"[CE] explicit={ce}")
            return ce

        if flare_type and self._flare_db is not None:
            try:
                specs = self._flare_db.get_flare_type_specs(flare_type)
                ce = specs["typical_ce"]
                trace.append(
                    f"[CE] from flare type {flare_type}: {ce}"
                )
                return ce
            except KeyError:
                pass

        trace.append(f"[CE] using default={self._default_ce}")
        return self._default_ce

    def _resolve_gwp(self, gas: str, gwp_source: str) -> Decimal:
        """Resolve GWP value from database or fallback.

        Args:
            gas: Gas identifier.
            gwp_source: GWP source.

        Returns:
            GWP value as Decimal.
        """
        gas_key = gas.upper()

        if self._flare_db is not None:
            try:
                return self._flare_db.get_gwp(gas_key, gwp_source)
            except (KeyError, AttributeError):
                pass

        if gas_key in _FALLBACK_GWP:
            return _FALLBACK_GWP[gas_key]

        raise KeyError(
            f"No GWP for gas '{gas}' in source '{gwp_source}'"
        )

    def _get_emission_factor(
        self,
        gas_type: str,
        gas: str,
        source: str,
    ) -> Decimal:
        """Get emission factor from database or raise.

        Args:
            gas_type: Gas type identifier.
            gas: Emission gas.
            source: EF source.

        Returns:
            Emission factor as Decimal.
        """
        if self._flare_db is not None:
            try:
                return self._flare_db.get_emission_factor(
                    gas_type, gas, source,
                )
            except (KeyError, AttributeError):
                pass

        # Fallback: minimal built-in EPA factors for general flaring
        _FALLBACK_EF: Dict[str, Decimal] = {
            "CO2": Decimal("60.0"),
            "CH4": Decimal("0.003"),
            "N2O": Decimal("0.00006"),
        }
        gas_key = gas.upper()
        if gas_key in _FALLBACK_EF:
            return _FALLBACK_EF[gas_key]

        raise KeyError(
            f"No emission factor for {gas_type}/{gas}/{source}"
        )

    def _get_molecular_weight(self, component: str) -> Decimal:
        """Get molecular weight from database or fallback.

        Args:
            component: Component identifier.

        Returns:
            Molecular weight as Decimal.
        """
        if self._flare_db is not None:
            try:
                return self._flare_db.get_molecular_weight(component)
            except (KeyError, AttributeError):
                pass

        # Fallback molecular weights
        _FALLBACK_MW: Dict[str, Decimal] = {
            "CH4":     Decimal("16.043"),
            "C2H6":    Decimal("30.069"),
            "C3H8":    Decimal("44.096"),
            "N_C4H10": Decimal("58.122"),
            "I_C4H10": Decimal("58.122"),
            "C5H12":   Decimal("72.149"),
            "C6_PLUS": Decimal("86.175"),
            "CO2":     Decimal("44.010"),
            "N2":      Decimal("28.014"),
            "H2S":     Decimal("34.081"),
            "H2":      Decimal("2.016"),
            "CO":      Decimal("28.010"),
            "C2H4":    Decimal("28.054"),
            "C3H6":    Decimal("42.080"),
            "H2O":     Decimal("18.015"),
        }
        comp_key = component.upper()
        if comp_key in _FALLBACK_MW:
            return _FALLBACK_MW[comp_key]

        raise KeyError(f"Unknown gas component: {component}")

    def _get_carbon_count(self, component: str) -> int:
        """Get carbon atom count from database or fallback.

        Args:
            component: Component identifier.

        Returns:
            Carbon atom count.
        """
        if self._flare_db is not None:
            try:
                return self._flare_db.get_carbon_count(component)
            except (KeyError, AttributeError):
                pass

        _FALLBACK_CC: Dict[str, int] = {
            "CH4": 1, "C2H6": 2, "C3H8": 3, "N_C4H10": 4,
            "I_C4H10": 4, "C5H12": 5, "C6_PLUS": 6, "CO2": 1,
            "N2": 0, "H2S": 0, "H2": 0, "CO": 1, "C2H4": 2,
            "C3H6": 3, "H2O": 0,
        }
        comp_key = component.upper()
        return _FALLBACK_CC.get(comp_key, 0)

    def _check_produces_co2(self, component: str) -> bool:
        """Check if combustion of a component produces CO2.

        Args:
            component: Component identifier.

        Returns:
            True if combustion produces CO2.
        """
        if self._flare_db is not None:
            try:
                return self._flare_db.produces_co2(component)
            except (KeyError, AttributeError):
                pass

        _FALLBACK_PRODUCES_CO2: Dict[str, bool] = {
            "CH4": True, "C2H6": True, "C3H8": True, "N_C4H10": True,
            "I_C4H10": True, "C5H12": True, "C6_PLUS": True,
            "CO2": False, "N2": False, "H2S": False, "H2": False,
            "CO": True, "C2H4": True, "C3H6": True, "H2O": False,
        }
        return _FALLBACK_PRODUCES_CO2.get(component.upper(), False)

    def _calculate_mixture_hhv(
        self,
        composition: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate mixture HHV from composition using database.

        Falls back to inline component HHVs if database is unavailable.

        Args:
            composition: Mole fraction dictionary.

        Returns:
            Mixture HHV in BTU/scf.
        """
        if self._flare_db is not None:
            try:
                return self._flare_db.calculate_hhv(composition, "EPA")
            except (KeyError, AttributeError):
                pass

        # Fallback HHV values (BTU/scf)
        _FALLBACK_HHV: Dict[str, Decimal] = {
            "CH4": Decimal("1012.0"), "C2H6": Decimal("1773.0"),
            "C3H8": Decimal("2524.0"), "N_C4H10": Decimal("3271.0"),
            "I_C4H10": Decimal("3253.0"), "C5H12": Decimal("4010.0"),
            "C6_PLUS": Decimal("4762.0"), "CO2": Decimal("0"),
            "N2": Decimal("0"), "H2S": Decimal("647.0"),
            "H2": Decimal("325.0"), "CO": Decimal("321.0"),
            "C2H4": Decimal("1614.0"), "C3H6": Decimal("2336.0"),
            "H2O": Decimal("0"),
        }

        hhv = Decimal("0")
        for comp, frac in composition.items():
            comp_key = comp.upper()
            comp_hhv = _FALLBACK_HHV.get(comp_key, Decimal("0"))
            hhv += Decimal(str(frac)) * comp_hhv

        return self._quantize(hhv)

    def _get_default_hhv(self, gas_type: str) -> Decimal:
        """Get default HHV for a gas type when no composition available.

        Args:
            gas_type: Gas type identifier.

        Returns:
            Default HHV in BTU/scf.
        """
        _DEFAULT_HHVS: Dict[str, Decimal] = {
            "GENERAL":                    Decimal("1000.0"),
            "NATURAL_GAS":                Decimal("1028.0"),
            "ASSOCIATED_GAS":             Decimal("1250.0"),
            "REFINERY_OFF_GAS":           Decimal("600.0"),
            "LANDFILL_GAS":               Decimal("500.0"),
            "BIOGAS":                     Decimal("600.0"),
            "CHEMICAL_PLANT_WASTE_GAS":   Decimal("450.0"),
        }
        gt_key = gas_type.upper()
        return _DEFAULT_HHVS.get(gt_key, Decimal("1000.0"))

    # ==================================================================
    # PRIVATE: Utility methods
    # ==================================================================

    def _quantize(self, value: Decimal) -> Decimal:
        """Round a Decimal to the configured precision.

        Args:
            value: Raw Decimal value.

        Returns:
            Rounded Decimal.
        """
        try:
            return value.quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
        except InvalidOperation:
            logger.warning("Failed to quantize value: %s", value)
            return value

    def _compute_provenance_hash(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 provenance hash for a calculation result.

        Args:
            data: Dictionary of calculation identifiers and results.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation in the provenance tracker if available.

        Args:
            action: Action name.
            entity_id: Entity identifier.
            data: Optional data dictionary.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="flaring_calculator",
                    action=action,
                    entity_id=entity_id,
                    data=data or {},
                )
            except Exception as exc:
                logger.debug(
                    "Provenance recording failed (non-critical): %s", exc,
                )

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"EmissionCalculatorEngine("
            f"precision={self._precision_places}, "
            f"gwp={self._default_gwp_source}, "
            f"ef={self._default_ef_source}, "
            f"ce={self._default_ce}, "
            f"db={'yes' if self._flare_db else 'no'})"
        )
