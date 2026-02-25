# -*- coding: utf-8 -*-
"""
Engine 7: Steam/Heat Purchase Pipeline Engine for AGENT-MRV-011.

13-stage orchestrated calculation pipeline for Scope 2 purchased steam,
district heating, and district cooling emission calculations:

  Stage 1:  Validate input (facility, tenant, energy type, quantities)
  Stage 2:  Resolve facility information
  Stage 3:  Resolve supplier data (steam only)
  Stage 4:  Load emission factors (fuel-based, regional, or COP)
  Stage 5:  Convert units to GJ (standard basis)
  Stage 6:  Calculate emissions (dispatch to Engine 2 or 3)
  Stage 7:  Apply CHP allocation (Engine 4, if applicable)
  Stage 8:  Separate biogenic CO2 from fossil CO2
  Stage 9:  Compute per-gas breakdown with GWP
  Stage 10: Quantify uncertainty (Engine 5)
  Stage 11: Check regulatory compliance (Engine 6)
  Stage 12: Assemble final result
  Stage 13: Seal provenance chain (SHA-256)

This pipeline orchestrates all six upstream engines:
  - Engine 1: SteamHeatDatabaseEngine (EF lookups, fuel data, regional factors)
  - Engine 2: SteamEmissionsCalculatorEngine (steam emission calculations)
  - Engine 3: HeatCoolingCalculatorEngine (heating and cooling calculations)
  - Engine 4: CHPAllocationEngine (CHP emission allocation)
  - Engine 5: UncertaintyQuantifierEngine (Monte Carlo / analytical uncertainty)
  - Engine 6: ComplianceCheckerEngine (multi-framework regulatory compliance)

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Decimal arithmetic
    - Emission factors sourced from database or validated fallback tables
    - No LLM calls in any numeric calculation path
    - Provenance chain hashes every stage output with SHA-256
    - Unit conversions use exact GJ-based conversion factors

Thread Safety:
    Thread-safe singleton using ``threading.RLock``. All mutable state
    (counters, calculation store, batch jobs) is protected by the lock.
    Concurrent calls to ``run_pipeline`` are safe; each call creates
    its own provenance chain for per-invocation audit trails.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
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
# Conditional imports for upstream engines (Engines 1-6 of MRV-011)
# ---------------------------------------------------------------------------

try:
    from greenlang.steam_heat_purchase.steam_heat_database import (
        SteamHeatDatabaseEngine,
    )
    _DB_ENGINE_AVAILABLE = True
except ImportError:
    _DB_ENGINE_AVAILABLE = False

try:
    from greenlang.steam_heat_purchase.steam_emissions_calculator import (
        SteamEmissionsCalculatorEngine,
    )
    _STEAM_ENGINE_AVAILABLE = True
except ImportError:
    _STEAM_ENGINE_AVAILABLE = False

try:
    from greenlang.steam_heat_purchase.heat_cooling_calculator import (
        HeatCoolingCalculatorEngine,
    )
    _HEAT_COOL_ENGINE_AVAILABLE = True
except ImportError:
    _HEAT_COOL_ENGINE_AVAILABLE = False

try:
    from greenlang.steam_heat_purchase.chp_allocation import (
        CHPAllocationEngine,
    )
    _CHP_ENGINE_AVAILABLE = True
except ImportError:
    _CHP_ENGINE_AVAILABLE = False

try:
    from greenlang.steam_heat_purchase.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
    _UNC_ENGINE_AVAILABLE = True
except ImportError:
    _UNC_ENGINE_AVAILABLE = False

try:
    from greenlang.steam_heat_purchase.compliance_checker import (
        ComplianceCheckerEngine,
    )
    _COMP_ENGINE_AVAILABLE = True
except ImportError:
    _COMP_ENGINE_AVAILABLE = False

try:
    from greenlang.steam_heat_purchase.provenance import get_provenance
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False

try:
    from greenlang.steam_heat_purchase.metrics import (
        get_metrics as _get_metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")
_QUANT_6 = Decimal("0.000001")

#: Pipeline version identifier.
PIPELINE_VERSION: str = "1.0.0"

#: Supported energy types for this pipeline.
SUPPORTED_ENERGY_TYPES: List[str] = [
    "steam",
    "district_heating",
    "district_cooling",
    "hot_water",
    "chilled_water",
    "process_heat",
    "chp",
]

#: Pipeline stage names in execution order.
PIPELINE_STAGES: List[str] = [
    "validate_request",
    "resolve_facility",
    "resolve_supplier",
    "load_emission_factors",
    "convert_units",
    "calculate_emissions",
    "chp_allocation",
    "separate_biogenic",
    "gas_breakdown",
    "quantify_uncertainty",
    "check_compliance",
    "assemble_result",
    "seal_provenance",
]

#: IPCC GWP values by assessment report.
GWP_TABLE: Dict[str, Dict[str, Decimal]] = {
    "AR4": {"CO2": Decimal("1"), "CH4": Decimal("25"), "N2O": Decimal("298")},
    "AR5": {"CO2": Decimal("1"), "CH4": Decimal("28"), "N2O": Decimal("265")},
    "AR6": {"CO2": Decimal("1"), "CH4": Decimal("27.9"), "N2O": Decimal("273")},
    "AR6_20YR": {"CO2": Decimal("1"), "CH4": Decimal("81.2"), "N2O": Decimal("273")},
}

VALID_GWP_SOURCES: set = {"AR4", "AR5", "AR6", "AR6_20YR"}

#: Energy unit to GJ conversion factors (multiply by factor to get GJ).
UNIT_TO_GJ: Dict[str, Decimal] = {
    "gj": Decimal("1.0"),
    "mwh": Decimal("3.6"),
    "kwh": Decimal("0.0036"),
    "mmbtu": Decimal("1.055056"),
    "therm": Decimal("0.105506"),
    "mj": Decimal("0.001"),
}

#: Fallback fuel emission factors (kgCO2, kgCH4, kgN2O per GJ of fuel input).
FALLBACK_FUEL_EFS: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "co2_ef": Decimal("56.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.85"),
        "is_biogenic": _ZERO,
    },
    "fuel_oil_2": {
        "co2_ef": Decimal("74.100"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.82"),
        "is_biogenic": _ZERO,
    },
    "fuel_oil_6": {
        "co2_ef": Decimal("77.400"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.80"),
        "is_biogenic": _ZERO,
    },
    "coal_bituminous": {
        "co2_ef": Decimal("94.600"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.78"),
        "is_biogenic": _ZERO,
    },
    "biomass": {
        "co2_ef": Decimal("0.0"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.75"),
        "is_biogenic": _ONE,
        "biogenic_co2_ef": Decimal("112.000"),
    },
    "biogas": {
        "co2_ef": Decimal("0.0"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.80"),
        "is_biogenic": _ONE,
        "biogenic_co2_ef": Decimal("54.600"),
    },
    "waste": {
        "co2_ef": Decimal("91.700"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.70"),
        "is_biogenic": _ZERO,
    },
}

#: Fallback district heating factors by region (kgCO2e/GJ, distribution loss).
FALLBACK_DH_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "denmark": {"ef_kgco2e_per_gj": Decimal("36.0"), "loss_pct": Decimal("0.10")},
    "sweden": {"ef_kgco2e_per_gj": Decimal("18.0"), "loss_pct": Decimal("0.08")},
    "finland": {"ef_kgco2e_per_gj": Decimal("55.0"), "loss_pct": Decimal("0.09")},
    "germany": {"ef_kgco2e_per_gj": Decimal("72.0"), "loss_pct": Decimal("0.12")},
    "poland": {"ef_kgco2e_per_gj": Decimal("105.0"), "loss_pct": Decimal("0.15")},
    "netherlands": {"ef_kgco2e_per_gj": Decimal("58.0"), "loss_pct": Decimal("0.10")},
    "france": {"ef_kgco2e_per_gj": Decimal("42.0"), "loss_pct": Decimal("0.10")},
    "uk": {"ef_kgco2e_per_gj": Decimal("65.0"), "loss_pct": Decimal("0.11")},
    "us": {"ef_kgco2e_per_gj": Decimal("70.0"), "loss_pct": Decimal("0.12")},
    "japan": {"ef_kgco2e_per_gj": Decimal("62.0"), "loss_pct": Decimal("0.10")},
    "china": {"ef_kgco2e_per_gj": Decimal("95.0"), "loss_pct": Decimal("0.18")},
    "russia": {"ef_kgco2e_per_gj": Decimal("88.0"), "loss_pct": Decimal("0.20")},
    "global": {"ef_kgco2e_per_gj": Decimal("70.0"), "loss_pct": Decimal("0.12")},
}

#: Fallback cooling system COP values by technology.
FALLBACK_COOLING_COP: Dict[str, Decimal] = {
    "centrifugal_chiller": Decimal("6.0"),
    "screw_chiller": Decimal("4.5"),
    "reciprocating_chiller": Decimal("4.0"),
    "absorption_single": Decimal("0.7"),
    "absorption_double": Decimal("1.2"),
    "absorption_triple": Decimal("1.6"),
    "heat_pump": Decimal("3.5"),
    "free_cooling": Decimal("20.0"),
    "thermal_storage": Decimal("5.0"),
}

#: Default grid emission factor for cooling (kgCO2e/kWh).
DEFAULT_GRID_EF_KWH: Decimal = Decimal("0.450")

#: CHP default efficiencies by fuel type.
FALLBACK_CHP_EFFICIENCIES: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "electrical_efficiency": Decimal("0.35"),
        "thermal_efficiency": Decimal("0.45"),
        "overall_efficiency": Decimal("0.80"),
    },
    "coal": {
        "electrical_efficiency": Decimal("0.30"),
        "thermal_efficiency": Decimal("0.40"),
        "overall_efficiency": Decimal("0.70"),
    },
    "biomass": {
        "electrical_efficiency": Decimal("0.25"),
        "thermal_efficiency": Decimal("0.50"),
        "overall_efficiency": Decimal("0.75"),
    },
    "fuel_oil": {
        "electrical_efficiency": Decimal("0.32"),
        "thermal_efficiency": Decimal("0.43"),
        "overall_efficiency": Decimal("0.75"),
    },
    "municipal_waste": {
        "electrical_efficiency": Decimal("0.20"),
        "thermal_efficiency": Decimal("0.45"),
        "overall_efficiency": Decimal("0.65"),
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Convert a value to Decimal safely.

    Args:
        value: Value to convert.
        default: Fallback if conversion fails.

    Returns:
        Decimal representation of value or default.
    """
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to 6 decimal places with half-up rounding.

    Args:
        value: Decimal to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_QUANT_6, ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Pipeline Engine
# ---------------------------------------------------------------------------


class SteamHeatPipelineEngine:
    """Engine 7: 13-stage orchestrated Steam/Heat Purchase pipeline.

    Ties together all six upstream engines (database, steam calculator,
    heat/cooling calculator, CHP allocation, uncertainty, compliance) to
    provide end-to-end Scope 2 purchased steam, district heating, and
    district cooling emission calculations with full provenance tracking.

    The pipeline supports four primary calculation modes:
      - **Steam**: Fuel-based or supplier-specific emission factors for
        purchased steam, including boiler efficiency and condensate return.
      - **Heating**: Regional emission factors for district heating networks
        with distribution loss adjustments.
      - **Cooling**: COP-based calculations for district cooling systems
        converting cooling output to electrical input.
      - **CHP**: Combined heat and power allocation splitting total plant
        emissions between electrical and thermal outputs.

    Thread Safety:
        Thread-safe singleton using ``threading.RLock``. All mutable
        pipeline state (counters, calculation store, batch jobs) is
        protected by the lock. Concurrent calls to ``run_pipeline``
        are safe; each creates its own provenance chain.

    Attributes:
        _db_engine: SteamHeatDatabaseEngine for EF lookups.
        _steam_engine: SteamEmissionsCalculatorEngine for steam calcs.
        _heat_cool_engine: HeatCoolingCalculatorEngine for heating/cooling.
        _chp_engine: CHPAllocationEngine for CHP allocation.
        _uncertainty_engine: UncertaintyQuantifierEngine for uncertainty.
        _compliance_engine: ComplianceCheckerEngine for compliance.
    """

    _instance: Optional["SteamHeatPipelineEngine"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "SteamHeatPipelineEngine":
        """Ensure singleton pattern with thread-safe initialization."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False  # type: ignore[attr-defined]
        return cls._instance

    def __init__(
        self,
        db_engine: Any = None,
        steam_engine: Any = None,
        heat_cool_engine: Any = None,
        chp_engine: Any = None,
        uncertainty_engine: Any = None,
        compliance_engine: Any = None,
        config: Any = None,
        metrics: Any = None,
    ) -> None:
        """Initialize the SteamHeatPipelineEngine.

        Args:
            db_engine: SteamHeatDatabaseEngine instance or None.
            steam_engine: SteamEmissionsCalculatorEngine instance or None.
            heat_cool_engine: HeatCoolingCalculatorEngine instance or None.
            chp_engine: CHPAllocationEngine instance or None.
            uncertainty_engine: UncertaintyQuantifierEngine instance or None.
            compliance_engine: ComplianceCheckerEngine instance or None.
            config: SteamHeatPurchaseConfig instance or None.
            metrics: SteamHeatPurchaseMetrics instance or None.
        """
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._config = config
        self._metrics = metrics
        self._lock = threading.RLock()

        # Pipeline counters
        self._pipeline_runs: int = 0
        self._successful_runs: int = 0
        self._failed_runs: int = 0
        self._total_co2e_kg: Decimal = _ZERO
        self._total_biogenic_co2_kg: Decimal = _ZERO
        self._total_energy_gj: Decimal = _ZERO
        self._total_duration_ms: float = 0.0

        # In-memory calculation store
        self._calculations: Dict[str, Dict[str, Any]] = {}
        self._batch_jobs: Dict[str, Dict[str, Any]] = {}

        # Initialize upstream engines
        self._db_engine = self._init_engine(
            db_engine,
            SteamHeatDatabaseEngine if _DB_ENGINE_AVAILABLE else None,
            "SteamHeatDatabaseEngine",
        )
        self._steam_engine = self._init_engine(
            steam_engine,
            SteamEmissionsCalculatorEngine if _STEAM_ENGINE_AVAILABLE else None,
            "SteamEmissionsCalculatorEngine",
        )
        self._heat_cool_engine = self._init_engine(
            heat_cool_engine,
            HeatCoolingCalculatorEngine if _HEAT_COOL_ENGINE_AVAILABLE else None,
            "HeatCoolingCalculatorEngine",
        )
        self._chp_engine = self._init_engine(
            chp_engine,
            CHPAllocationEngine if _CHP_ENGINE_AVAILABLE else None,
            "CHPAllocationEngine",
        )
        self._uncertainty_engine = self._init_engine(
            uncertainty_engine,
            UncertaintyQuantifierEngine if _UNC_ENGINE_AVAILABLE else None,
            "UncertaintyQuantifierEngine",
        )
        self._compliance_engine = self._init_engine(
            compliance_engine,
            ComplianceCheckerEngine if _COMP_ENGINE_AVAILABLE else None,
            "ComplianceCheckerEngine",
        )

        self._created_at = _utcnow()

        engine_count = sum(
            1 for e in [
                self._db_engine, self._steam_engine, self._heat_cool_engine,
                self._chp_engine, self._uncertainty_engine, self._compliance_engine,
            ] if e is not None
        )
        logger.info(
            "SteamHeatPipelineEngine initialized with %d/6 engines", engine_count,
        )

    def _init_engine(
        self,
        provided: Any,
        engine_cls: Any,
        engine_name: str,
    ) -> Any:
        """Initialize an engine: use provided, create from class, or None.

        Args:
            provided: Engine instance supplied by caller.
            engine_cls: Engine class (may be None if import failed).
            engine_name: Human-readable engine name for logging.

        Returns:
            Engine instance or None.
        """
        if provided is not None:
            return provided
        if engine_cls is not None:
            try:
                return engine_cls()
            except Exception as exc:
                logger.warning(
                    "Failed to create %s: %s", engine_name, exc,
                )
        return None

    # ==================================================================
    # Public Methods - Main Pipeline
    # ==================================================================

    def run_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full 13-stage Steam/Heat Purchase pipeline.

        Dispatches to ``calculate_steam``, ``calculate_heating``,
        ``calculate_cooling``, or ``calculate_chp`` based on the
        ``energy_type`` field in the request.

        Args:
            request: Calculation request dict with keys:
                - calculation_id (optional, auto-generated UUID)
                - tenant_id: str
                - facility_id: str
                - energy_type: str (steam, district_heating, district_cooling,
                  hot_water, chilled_water, process_heat, chp)
                - quantity: Decimal or float (energy consumed)
                - unit: str (gj, mwh, kwh, mmbtu, therm, mj)
                - fuel_type: str (optional, for fuel-based calcs)
                - supplier_id: str (optional, for steam)
                - region: str (optional, for district heating)
                - boiler_efficiency: Decimal (optional, 0-1)
                - distribution_loss_pct: Decimal (optional, 0-1)
                - condensate_return_pct: Decimal (optional, 0-1)
                - cooling_technology: str (optional, for cooling)
                - cop: Decimal (optional, for cooling)
                - grid_ef_kwh: Decimal (optional, kgCO2e/kWh)
                - gwp_source: str (AR4/AR5/AR6/AR6_20YR, default AR5)
                - chp_params: dict (optional, for CHP allocation)
                - include_compliance: bool (default False)
                - compliance_frameworks: list of str (optional)
                - include_uncertainty: bool (default False)
                - mc_iterations: int (optional, default 5000)

        Returns:
            Complete calculation result dict with provenance hash.

        Raises:
            ValueError: If input validation fails.
        """
        energy_type = str(request.get("energy_type", "steam")).lower()

        if energy_type in ("steam", "process_heat"):
            return self.calculate_steam(request)
        elif energy_type in ("district_heating", "hot_water"):
            return self.calculate_heating(request)
        elif energy_type in ("district_cooling", "chilled_water"):
            return self.calculate_cooling(request)
        elif energy_type == "chp":
            return self.calculate_chp(request)
        else:
            raise ValueError(
                f"Unsupported energy_type '{energy_type}'. "
                f"Must be one of: {SUPPORTED_ENERGY_TYPES}"
            )

    def calculate_steam(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full steam emission calculation pipeline.

        Runs all 13 stages for purchased steam, including fuel-based
        emission factor lookup, boiler efficiency adjustment, condensate
        return correction, biogenic separation, per-gas GWP breakdown,
        uncertainty quantification, and compliance checks.

        Args:
            request: Steam calculation request dict.

        Returns:
            Complete steam calculation result with provenance.

        Raises:
            ValueError: If validation fails.
        """
        return self._execute_pipeline(request, "steam")

    def calculate_heating(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full district heating emission calculation pipeline.

        Runs all 13 stages for district heating, using regional emission
        factors with distribution loss adjustments.

        Args:
            request: Heating calculation request dict.

        Returns:
            Complete heating calculation result with provenance.

        Raises:
            ValueError: If validation fails.
        """
        return self._execute_pipeline(request, "district_heating")

    def calculate_cooling(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full district cooling emission calculation pipeline.

        Runs all 13 stages for district cooling, using COP-based
        calculations to convert cooling output to electrical input,
        then applying grid emission factors.

        Args:
            request: Cooling calculation request dict.

        Returns:
            Complete cooling calculation result with provenance.

        Raises:
            ValueError: If validation fails.
        """
        return self._execute_pipeline(request, "district_cooling")

    def calculate_chp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full CHP emission allocation pipeline.

        Runs all 13 stages for combined heat and power, allocating
        total plant emissions between electrical and thermal outputs
        using the specified allocation method (efficiency, energy,
        or exergy).

        Args:
            request: CHP calculation request dict.

        Returns:
            Complete CHP allocation result with provenance.

        Raises:
            ValueError: If validation fails.
        """
        return self._execute_pipeline(request, "chp")

    # ==================================================================
    # Public Methods - Batch Pipeline
    # ==================================================================

    def run_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run pipeline for a batch of calculation requests.

        Iterates through each request, calling ``run_pipeline`` for each.
        Failed requests are captured in the errors list without aborting
        the batch. Results are aggregated with totals.

        Args:
            requests: List of individual calculation request dicts.

        Returns:
            Batch result dict with individual results, errors, and
            aggregated totals.
        """
        batch_start = time.monotonic()
        batch_id = str(uuid.uuid4())
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        total_biogenic_kg = _ZERO
        total_energy_gj = _ZERO

        for idx, req in enumerate(requests):
            try:
                result = self.run_pipeline(req)
                results.append(result)
                total_co2e_kg += _safe_decimal(result.get("total_co2e_kg"))
                total_biogenic_kg += _safe_decimal(result.get("biogenic_co2_kg"))
                total_energy_gj += _safe_decimal(result.get("energy_gj"))
            except Exception as exc:
                errors.append({
                    "index": idx,
                    "calculation_id": req.get("calculation_id", ""),
                    "facility_id": req.get("facility_id", ""),
                    "energy_type": req.get("energy_type", ""),
                    "error": str(exc),
                })
                logger.error("Batch item %d failed: %s", idx, exc)

        batch_duration = time.monotonic() - batch_start

        # Store batch job
        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "total_requests": len(requests),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "total_co2e_kg": _quantize(total_co2e_kg),
            "total_co2e_tonnes": _quantize(total_co2e_kg / _THOUSAND),
            "total_biogenic_co2_kg": _quantize(total_biogenic_kg),
            "total_energy_gj": _quantize(total_energy_gj),
            "facility_count": len(set(
                r.get("facility_id", "") for r in results
            )),
            "duration_ms": round(batch_duration * 1000, 3),
            "provenance_hash": self._compute_batch_hash(results),
            "created_at": _utcnow().isoformat(),
        }

        with self._lock:
            self._batch_jobs[batch_id] = batch_result

        if self._metrics and _METRICS_AVAILABLE:
            try:
                tenant_id = requests[0].get("tenant_id", "") if requests else ""
                status = "success" if not errors else ("partial" if results else "failure")
                _get_metrics().record_batch(status, len(requests), tenant_id)
            except Exception:
                pass

        logger.info(
            "Batch %s completed: %d/%d successful, %.3fs",
            batch_id, len(results), len(requests), batch_duration,
        )
        return batch_result

    # ==================================================================
    # Public Methods - Aggregation
    # ==================================================================

    def aggregate_results(
        self,
        calc_ids: List[str],
        aggregation_type: str = "by_facility",
    ) -> Dict[str, Any]:
        """Aggregate stored calculation results by a specified dimension.

        Groups stored calculations by facility, fuel type, energy type,
        supplier, or period and produces summary totals for each group.

        Args:
            calc_ids: List of calculation IDs to aggregate.
            aggregation_type: Dimension (by_facility, by_fuel,
                by_energy_type, by_supplier, by_period).

        Returns:
            Aggregation result dict with grouped totals.
        """
        with self._lock:
            calcs = [
                self._calculations[cid]
                for cid in calc_ids
                if cid in self._calculations
            ]

        if not calcs:
            return {
                "aggregation_type": aggregation_type,
                "groups": [],
                "total_co2e_kg": _ZERO,
                "total_co2e_tonnes": _ZERO,
                "calculation_count": 0,
            }

        group_key = self._aggregation_key(aggregation_type)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for calc in calcs:
            key = str(calc.get(group_key, "unknown"))
            groups.setdefault(key, []).append(calc)

        group_summaries: List[Dict[str, Any]] = []
        grand_total_co2e = _ZERO

        for group_name, group_calcs in sorted(groups.items()):
            group_co2e = sum(
                _safe_decimal(c.get("total_co2e_kg")) for c in group_calcs
            )
            group_biogenic = sum(
                _safe_decimal(c.get("biogenic_co2_kg")) for c in group_calcs
            )
            group_energy = sum(
                _safe_decimal(c.get("energy_gj")) for c in group_calcs
            )
            grand_total_co2e += group_co2e

            group_summaries.append({
                "group": group_name,
                "calculation_count": len(group_calcs),
                "total_co2e_kg": _quantize(group_co2e),
                "total_co2e_tonnes": _quantize(group_co2e / _THOUSAND),
                "total_biogenic_co2_kg": _quantize(group_biogenic),
                "total_energy_gj": _quantize(group_energy),
            })

        return {
            "aggregation_type": aggregation_type,
            "groups": group_summaries,
            "total_co2e_kg": _quantize(grand_total_co2e),
            "total_co2e_tonnes": _quantize(grand_total_co2e / _THOUSAND),
            "calculation_count": len(calcs),
            "provenance_hash": self._compute_aggregation_hash(
                calc_ids, aggregation_type
            ),
        }

    # ==================================================================
    # Public Methods - Validation and Query
    # ==================================================================

    def validate_pipeline_request(
        self, request: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate a pipeline request without executing it.

        Args:
            request: Calculation request dict.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        validation = self._validate_request(request)
        return (
            validation.get("valid", False),
            validation.get("errors", []),
        )

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Return current pipeline operational status.

        Returns:
            Dictionary with pipeline status, engine availability, and
            readiness information.
        """
        with self._lock:
            return {
                "status": "ready",
                "version": PIPELINE_VERSION,
                "engines": {
                    "database": self._db_engine is not None,
                    "steam_calculator": self._steam_engine is not None,
                    "heat_cool_calculator": self._heat_cool_engine is not None,
                    "chp_allocation": self._chp_engine is not None,
                    "uncertainty": self._uncertainty_engine is not None,
                    "compliance": self._compliance_engine is not None,
                },
                "engines_available": sum(
                    1 for e in [
                        self._db_engine, self._steam_engine,
                        self._heat_cool_engine, self._chp_engine,
                        self._uncertainty_engine, self._compliance_engine,
                    ] if e is not None
                ),
                "stages_count": len(PIPELINE_STAGES),
                "supported_energy_types": list(SUPPORTED_ENERGY_TYPES),
                "pipeline_runs": self._pipeline_runs,
                "created_at": self._created_at.isoformat(),
            }

    def get_supported_energy_types(self) -> List[str]:
        """Return the list of supported energy types.

        Returns:
            Ordered list of energy type strings.
        """
        return list(SUPPORTED_ENERGY_TYPES)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Return pipeline-level aggregate statistics.

        Returns:
            Dictionary with run counts, success rate, cumulative emissions,
            and average duration.
        """
        with self._lock:
            avg_duration = (
                self._total_duration_ms / self._pipeline_runs
                if self._pipeline_runs > 0
                else 0.0
            )
            success_rate = (
                self._successful_runs / self._pipeline_runs * 100.0
                if self._pipeline_runs > 0
                else 0.0
            )
            return {
                "pipeline_runs": self._pipeline_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "success_rate_pct": round(success_rate, 2),
                "avg_duration_ms": round(avg_duration, 3),
                "total_co2e_kg": self._total_co2e_kg,
                "total_co2e_tonnes": _quantize(
                    self._total_co2e_kg / _THOUSAND
                ),
                "total_biogenic_co2_kg": self._total_biogenic_co2_kg,
                "total_energy_gj": self._total_energy_gj,
                "stored_calculations": len(self._calculations),
                "stored_batch_jobs": len(self._batch_jobs),
                "engines_available": sum(
                    1 for e in [
                        self._db_engine, self._steam_engine,
                        self._heat_cool_engine, self._chp_engine,
                        self._uncertainty_engine, self._compliance_engine,
                    ] if e is not None
                ),
            }

    def health_check(self) -> Dict[str, Any]:
        """Run a health check on the pipeline and all engines.

        Returns:
            Dictionary with overall health status and per-engine status.
        """
        engine_health: Dict[str, str] = {}
        engine_names = {
            "database": self._db_engine,
            "steam_calculator": self._steam_engine,
            "heat_cool_calculator": self._heat_cool_engine,
            "chp_allocation": self._chp_engine,
            "uncertainty": self._uncertainty_engine,
            "compliance": self._compliance_engine,
        }

        for name, engine in engine_names.items():
            if engine is None:
                engine_health[name] = "unavailable"
            else:
                try:
                    if hasattr(engine, "health_check"):
                        hc = engine.health_check()
                        engine_health[name] = hc.get("status", "healthy")
                    else:
                        engine_health[name] = "healthy"
                except Exception as exc:
                    engine_health[name] = f"unhealthy: {exc}"

        all_healthy = all(
            s in ("healthy", "unavailable") for s in engine_health.values()
        )

        return {
            "status": "healthy" if all_healthy else "degraded",
            "version": PIPELINE_VERSION,
            "engines": engine_health,
            "pipeline_runs": self._pipeline_runs,
            "uptime_since": self._created_at.isoformat(),
            "checked_at": _utcnow().isoformat(),
        }

    def compare_energy_sources(
        self, requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare emission results across different energy source scenarios.

        Runs the pipeline for each request and produces a side-by-side
        comparison of emission intensities, total emissions, and cost
        implications for each scenario.

        Args:
            requests: List of calculation requests representing different
                energy source scenarios.

        Returns:
            Comparison result dict with per-scenario results and ranking.
        """
        scenario_results: List[Dict[str, Any]] = []
        for idx, req in enumerate(requests):
            scenario_name = req.get("scenario_name", f"scenario_{idx + 1}")
            try:
                result = self.run_pipeline(req)
                scenario_results.append({
                    "scenario_name": scenario_name,
                    "energy_type": req.get("energy_type", "unknown"),
                    "fuel_type": req.get("fuel_type", ""),
                    "total_co2e_kg": result.get("total_co2e_kg", _ZERO),
                    "total_co2e_tonnes": result.get("total_co2e_tonnes", _ZERO),
                    "biogenic_co2_kg": result.get("biogenic_co2_kg", _ZERO),
                    "energy_gj": result.get("energy_gj", _ZERO),
                    "emission_intensity_kgco2e_per_gj": result.get(
                        "emission_intensity_kgco2e_per_gj", _ZERO
                    ),
                    "calculation_id": result.get("calculation_id", ""),
                    "status": "success",
                })
            except Exception as exc:
                scenario_results.append({
                    "scenario_name": scenario_name,
                    "energy_type": req.get("energy_type", "unknown"),
                    "status": "error",
                    "error": str(exc),
                })

        # Rank scenarios by total CO2e (lowest first)
        successful = [s for s in scenario_results if s["status"] == "success"]
        ranked = sorted(
            successful,
            key=lambda s: _safe_decimal(s.get("total_co2e_kg")),
        )
        for rank, scenario in enumerate(ranked, 1):
            scenario["rank"] = rank

        return {
            "comparison_id": str(uuid.uuid4()),
            "total_scenarios": len(requests),
            "successful_scenarios": len(successful),
            "scenarios": scenario_results,
            "best_scenario": ranked[0]["scenario_name"] if ranked else None,
            "worst_scenario": ranked[-1]["scenario_name"] if ranked else None,
            "created_at": _utcnow().isoformat(),
        }

    def export_results(
        self,
        calc_ids: List[str],
        format: str = "json",
    ) -> Dict[str, Any]:
        """Export stored calculation results in the specified format.

        Args:
            calc_ids: List of calculation IDs to export.
            format: Export format (json, csv_summary, ghg_protocol).

        Returns:
            Export result dict with data in the specified format.
        """
        with self._lock:
            calcs = [
                self._calculations[cid]
                for cid in calc_ids
                if cid in self._calculations
            ]

        if not calcs:
            return {
                "format": format,
                "record_count": 0,
                "data": [],
                "exported_at": _utcnow().isoformat(),
            }

        if format == "csv_summary":
            return self._export_csv_summary(calcs)
        elif format == "ghg_protocol":
            return self._export_ghg_protocol(calcs)
        else:
            return self._export_json(calcs)

    def get_calculation(self, calc_id: str) -> Dict[str, Any]:
        """Retrieve a stored calculation result by its ID.

        Args:
            calc_id: Calculation identifier.

        Returns:
            Calculation result dict.

        Raises:
            KeyError: If calculation ID is not found.
        """
        with self._lock:
            if calc_id not in self._calculations:
                raise KeyError(f"Calculation '{calc_id}' not found")
            return dict(self._calculations[calc_id])

    # ==================================================================
    # Public Methods - Lifecycle
    # ==================================================================

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance and all state.

        Intended for testing teardown and re-initialization.
        """
        if cls._instance is not None:
            instance = cls._instance
            with instance._lock:
                instance._pipeline_runs = 0
                instance._successful_runs = 0
                instance._failed_runs = 0
                instance._total_co2e_kg = _ZERO
                instance._total_biogenic_co2_kg = _ZERO
                instance._total_energy_gj = _ZERO
                instance._total_duration_ms = 0.0
                instance._calculations.clear()
                instance._batch_jobs.clear()

            # Reset upstream engines
            for engine in [
                instance._db_engine, instance._steam_engine,
                instance._heat_cool_engine, instance._chp_engine,
                instance._uncertainty_engine, instance._compliance_engine,
            ]:
                if engine is not None and hasattr(engine, "reset"):
                    try:
                        engine.reset()
                    except Exception as exc:
                        logger.warning("Engine reset failed: %s", exc)

            cls._instance = None
            logger.info("SteamHeatPipelineEngine singleton reset")

    # ==================================================================
    # Core Pipeline Execution
    # ==================================================================

    def _execute_pipeline(
        self,
        request: Dict[str, Any],
        energy_type: str,
    ) -> Dict[str, Any]:
        """Execute the 13-stage pipeline for a given energy type.

        This is the core orchestration method that all public calculation
        methods delegate to. It runs each stage sequentially, recording
        provenance and timing for each step.

        Args:
            request: Calculation request dict.
            energy_type: Normalized energy type string.

        Returns:
            Complete calculation result dict.

        Raises:
            ValueError: If validation fails.
        """
        pipeline_start = time.monotonic()
        calc_id = request.get("calculation_id", str(uuid.uuid4()))
        tenant_id = request.get("tenant_id", "")
        facility_id = request.get("facility_id", "")
        gwp_source = request.get("gwp_source", "AR5")
        include_compliance = request.get("include_compliance", False)
        compliance_frameworks = request.get("compliance_frameworks")
        include_uncertainty = request.get("include_uncertainty", False)
        mc_iterations = request.get("mc_iterations", 5000)

        # Ensure energy_type is set in request
        request["energy_type"] = energy_type

        # Provenance chain
        prov = None
        chain_id = None
        if _PROVENANCE_AVAILABLE:
            try:
                prov = get_provenance()
                chain_id = prov.create_chain(calc_id)
            except Exception as exc:
                logger.warning("Provenance chain creation failed: %s", exc)
                prov = None

        calculation_trace: List[Dict[str, Any]] = []
        ctx: Dict[str, Any] = {
            "calc_id": calc_id,
            "energy_type": energy_type,
            "request": request,
        }

        try:
            # ---- Stage 1: Validate Request ----
            stage_start = time.monotonic()
            validation = self._validate_request(request)
            if not validation.get("valid", False):
                self._record_failure(pipeline_start, energy_type, tenant_id)
                raise ValueError(
                    f"Input validation failed: {validation.get('errors', [])}"
                )
            ctx["validation"] = validation
            calculation_trace.append(
                self._trace_entry("validate_request", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "REQUEST_RECEIVED", {
                    "calc_id": calc_id, "energy_type": energy_type,
                })
                prov.add_stage(chain_id, "INPUT_VALIDATED", {
                    "valid": True,
                    "warnings": validation.get("warnings", []),
                })

            # ---- Stage 2: Resolve Facility ----
            stage_start = time.monotonic()
            facility_info = self._resolve_facility(facility_id)
            ctx["facility"] = facility_info
            calculation_trace.append(
                self._trace_entry("resolve_facility", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "FACILITY_RESOLVED", {
                    "facility_id": facility_id,
                    "resolved": facility_info is not None,
                })

            # ---- Stage 3: Resolve Supplier (steam only) ----
            stage_start = time.monotonic()
            supplier_info = None
            if energy_type in ("steam", "process_heat"):
                supplier_id = request.get("supplier_id")
                if supplier_id:
                    supplier_info = self._resolve_supplier(supplier_id)
            ctx["supplier"] = supplier_info
            calculation_trace.append(
                self._trace_entry("resolve_supplier", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "SUPPLIER_RESOLVED", {
                    "supplier_id": request.get("supplier_id", ""),
                    "resolved": supplier_info is not None,
                })

            # ---- Stage 4: Load Emission Factors ----
            stage_start = time.monotonic()
            emission_factors = self._load_emission_factors(energy_type, request)
            ctx["emission_factors"] = emission_factors
            calculation_trace.append(
                self._trace_entry("load_emission_factors", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                ef_stage = "FUEL_EF_RETRIEVED" if energy_type in ("steam", "process_heat", "chp") else (
                    "DH_EF_RETRIEVED" if energy_type in ("district_heating", "hot_water") else
                    "COOLING_PARAMS_RETRIEVED"
                )
                prov.add_stage(chain_id, ef_stage, {
                    "source": emission_factors.get("source", "fallback"),
                    "factor_count": len(emission_factors.get("factors", {})),
                })

            # ---- Stage 5: Convert Units ----
            stage_start = time.monotonic()
            unit_result = self._convert_units(request)
            energy_gj = unit_result["energy_gj"]
            ctx["energy_gj"] = energy_gj
            ctx["unit_conversion"] = unit_result
            calculation_trace.append(
                self._trace_entry("convert_units", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "UNIT_CONVERTED", {
                    "original_quantity": str(request.get("quantity", _ZERO)),
                    "original_unit": request.get("unit", "gj"),
                    "energy_gj": str(energy_gj),
                })

            # ---- Stage 6: Calculate Emissions ----
            stage_start = time.monotonic()
            calc_result = self._calculate_emissions(
                energy_type, request, emission_factors, energy_gj,
            )
            ctx["calc_result"] = calc_result
            calculation_trace.append(
                self._trace_entry("calculate_emissions", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                calc_stage = "STEAM_CALCULATED" if energy_type in ("steam", "process_heat") else (
                    "HEATING_CALCULATED" if energy_type in ("district_heating", "hot_water") else
                    "COOLING_CALCULATED"
                )
                prov.add_stage(chain_id, calc_stage, {
                    "co2_kg": str(calc_result.get("co2_kg", _ZERO)),
                    "ch4_kg": str(calc_result.get("ch4_kg", _ZERO)),
                    "n2o_kg": str(calc_result.get("n2o_kg", _ZERO)),
                    "total_co2e_kg": str(calc_result.get("total_co2e_kg", _ZERO)),
                })

            # ---- Stage 7: CHP Allocation (if applicable) ----
            stage_start = time.monotonic()
            chp_result = None
            chp_params = request.get("chp_params")
            if energy_type == "chp" or chp_params:
                chp_result = self._apply_chp_allocation(calc_result, chp_params or {})
                if chp_result:
                    calc_result = self._merge_chp_into_result(calc_result, chp_result)
            ctx["chp_result"] = chp_result
            calculation_trace.append(
                self._trace_entry("chp_allocation", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "CHP_ALLOCATED", {
                    "applied": chp_result is not None,
                    "method": chp_params.get("allocation_method", "none") if chp_params else "none",
                })

            # ---- Stage 8: Separate Biogenic CO2 ----
            stage_start = time.monotonic()
            biogenic_result = self._separate_biogenic(
                calc_result, emission_factors,
            )
            ctx["biogenic"] = biogenic_result
            calculation_trace.append(
                self._trace_entry("separate_biogenic", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "BIOGENIC_SEPARATED", {
                    "biogenic_co2_kg": str(biogenic_result.get("biogenic_co2_kg", _ZERO)),
                    "fossil_co2_kg": str(biogenic_result.get("fossil_co2_kg", _ZERO)),
                })

            # ---- Stage 9: Gas Breakdown with GWP ----
            stage_start = time.monotonic()
            gas_breakdown = self._compute_gas_breakdown(
                calc_result, gwp_source,
            )
            ctx["gas_breakdown"] = gas_breakdown
            calculation_trace.append(
                self._trace_entry("gas_breakdown", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "GAS_BREAKDOWN_COMPUTED", {
                    "gwp_source": gwp_source,
                    "gas_count": len(gas_breakdown),
                })

            # ---- Stage 10: Uncertainty Quantification ----
            stage_start = time.monotonic()
            uncertainty_result = None
            if include_uncertainty:
                uncertainty_result = self._quantify_uncertainty(
                    calc_result, energy_type, mc_iterations,
                )
            ctx["uncertainty"] = uncertainty_result
            calculation_trace.append(
                self._trace_entry("quantify_uncertainty", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "UNCERTAINTY_QUANTIFIED", {
                    "performed": uncertainty_result is not None,
                    "method": uncertainty_result.get("method", "none") if uncertainty_result else "none",
                })

            # ---- Stage 11: Compliance Checks ----
            stage_start = time.monotonic()
            compliance_results: List[Dict[str, Any]] = []
            if include_compliance:
                compliance_results = self._check_compliance(
                    calc_result, energy_type, request, compliance_frameworks,
                )
            ctx["compliance"] = compliance_results
            calculation_trace.append(
                self._trace_entry("check_compliance", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "COMPLIANCE_CHECKED", {
                    "performed": len(compliance_results) > 0,
                    "frameworks_checked": len(compliance_results),
                })

            # ---- Stage 12: Assemble Result ----
            stage_start = time.monotonic()
            total_co2e_kg = _safe_decimal(calc_result.get("total_co2e_kg"))
            emission_intensity = _ZERO
            if energy_gj > _ZERO:
                emission_intensity = _quantize(total_co2e_kg / energy_gj)

            result = self._assemble_result(
                calc_id=calc_id,
                tenant_id=tenant_id,
                facility_id=facility_id,
                energy_type=energy_type,
                gwp_source=gwp_source,
                energy_gj=energy_gj,
                unit_result=unit_result,
                calc_result=calc_result,
                biogenic_result=biogenic_result,
                gas_breakdown=gas_breakdown,
                chp_result=chp_result,
                uncertainty_result=uncertainty_result,
                compliance_results=compliance_results,
                emission_intensity=emission_intensity,
                request=request,
                calculation_trace=calculation_trace,
            )
            calculation_trace.append(
                self._trace_entry("assemble_result", time.monotonic() - stage_start)
            )
            if prov and chain_id:
                prov.add_stage(chain_id, "RESULT_ASSEMBLED", {
                    "total_co2e_kg": str(total_co2e_kg),
                    "total_co2e_tonnes": str(result.get("total_co2e_tonnes", _ZERO)),
                })

            # ---- Stage 13: Seal Provenance ----
            stage_start = time.monotonic()
            provenance_hash = self._seal_provenance(
                prov, chain_id, calc_id, result,
            )
            result["provenance_hash"] = provenance_hash
            calculation_trace.append(
                self._trace_entry("seal_provenance", time.monotonic() - stage_start)
            )

            # Store result
            with self._lock:
                self._calculations[calc_id] = result

            # Record metrics and counters
            pipeline_duration = time.monotonic() - pipeline_start
            duration_ms = pipeline_duration * 1000
            self._record_success(
                duration_ms, total_co2e_kg,
                _safe_decimal(biogenic_result.get("biogenic_co2_kg")),
                energy_gj, energy_type, request,
            )

            logger.info(
                "Pipeline completed: calc=%s energy_type=%s co2e=%.6f kg "
                "(%.3f ms)",
                calc_id, energy_type, total_co2e_kg, duration_ms,
            )
            return result

        except ValueError:
            raise
        except Exception as exc:
            self._record_failure(pipeline_start, energy_type, tenant_id)
            logger.error(
                "Pipeline failed: calc=%s energy_type=%s error=%s",
                calc_id, energy_type, exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Internal Pipeline Steps
    # ==================================================================

    def _validate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Validate pipeline request input fields.

        Args:
            request: Raw calculation request dict.

        Returns:
            Validation result with valid (bool), errors, warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Required: facility_id
        facility_id = request.get("facility_id")
        if not facility_id:
            errors.append("facility_id is required")

        # Required: energy_type
        energy_type = str(request.get("energy_type", "")).lower()
        if not energy_type:
            errors.append("energy_type is required")
        elif energy_type not in SUPPORTED_ENERGY_TYPES:
            errors.append(
                f"energy_type '{energy_type}' is not supported. "
                f"Must be one of: {SUPPORTED_ENERGY_TYPES}"
            )

        # Required: quantity
        quantity = request.get("quantity")
        if quantity is None:
            errors.append("quantity is required")
        else:
            qty = _safe_decimal(quantity, Decimal("-1"))
            if qty < _ZERO:
                errors.append("quantity must be a non-negative number")

        # Unit validation
        unit = str(request.get("unit", "gj")).lower()
        if unit not in UNIT_TO_GJ:
            errors.append(
                f"unit '{unit}' is not supported. "
                f"Must be one of: {sorted(UNIT_TO_GJ.keys())}"
            )

        # GWP source validation
        gwp_source = request.get("gwp_source", "AR5")
        if gwp_source not in VALID_GWP_SOURCES:
            errors.append(
                f"gwp_source '{gwp_source}' is invalid. "
                f"Must be one of: {sorted(VALID_GWP_SOURCES)}"
            )

        # Tenant ID
        if not request.get("tenant_id"):
            warnings.append("tenant_id not specified; defaulting to empty")

        # Energy-type-specific validation
        if energy_type in ("steam", "process_heat"):
            self._validate_steam_fields(request, errors, warnings)
        elif energy_type in ("district_heating", "hot_water"):
            self._validate_heating_fields(request, errors, warnings)
        elif energy_type in ("district_cooling", "chilled_water"):
            self._validate_cooling_fields(request, errors, warnings)
        elif energy_type == "chp":
            self._validate_chp_fields(request, errors, warnings)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _validate_steam_fields(
        self,
        request: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate steam-specific request fields.

        Args:
            request: Request dict.
            errors: Error list to append to.
            warnings: Warning list to append to.
        """
        boiler_eff = request.get("boiler_efficiency")
        if boiler_eff is not None:
            eff = _safe_decimal(boiler_eff, Decimal("-1"))
            if eff <= _ZERO or eff > _ONE:
                errors.append("boiler_efficiency must be between 0 and 1")

        condensate = request.get("condensate_return_pct")
        if condensate is not None:
            cond = _safe_decimal(condensate, Decimal("-1"))
            if cond < _ZERO or cond > _ONE:
                errors.append("condensate_return_pct must be between 0 and 1")

        if not request.get("fuel_type") and not request.get("supplier_id"):
            warnings.append(
                "Neither fuel_type nor supplier_id provided for steam; "
                "will use default natural gas emission factors"
            )

    def _validate_heating_fields(
        self,
        request: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate district heating-specific request fields.

        Args:
            request: Request dict.
            errors: Error list to append to.
            warnings: Warning list to append to.
        """
        dist_loss = request.get("distribution_loss_pct")
        if dist_loss is not None:
            loss = _safe_decimal(dist_loss, Decimal("-1"))
            if loss < _ZERO or loss > Decimal("0.50"):
                errors.append("distribution_loss_pct must be between 0 and 0.50")

        if not request.get("region"):
            warnings.append(
                "region not specified for district heating; "
                "will use global default emission factor"
            )

    def _validate_cooling_fields(
        self,
        request: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate district cooling-specific request fields.

        Args:
            request: Request dict.
            errors: Error list to append to.
            warnings: Warning list to append to.
        """
        cop = request.get("cop")
        if cop is not None:
            cop_val = _safe_decimal(cop, Decimal("-1"))
            if cop_val <= _ZERO:
                errors.append("cop must be a positive number")
            elif cop_val > Decimal("25"):
                warnings.append("cop > 25 is unusually high; verify input")

        grid_ef = request.get("grid_ef_kwh")
        if grid_ef is not None:
            gef = _safe_decimal(grid_ef, Decimal("-1"))
            if gef < _ZERO:
                errors.append("grid_ef_kwh must be non-negative")

    def _validate_chp_fields(
        self,
        request: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate CHP-specific request fields.

        Args:
            request: Request dict.
            errors: Error list to append to.
            warnings: Warning list to append to.
        """
        chp_params = request.get("chp_params", {})
        if not chp_params:
            warnings.append(
                "chp_params not provided; will use default CHP settings"
            )
            return

        alloc_method = chp_params.get("allocation_method", "efficiency")
        valid_methods = {"efficiency", "energy", "exergy"}
        if alloc_method not in valid_methods:
            errors.append(
                f"CHP allocation_method '{alloc_method}' is invalid. "
                f"Must be one of: {sorted(valid_methods)}"
            )

        for eff_key in ("electrical_efficiency", "thermal_efficiency"):
            eff_val = chp_params.get(eff_key)
            if eff_val is not None:
                eff = _safe_decimal(eff_val, Decimal("-1"))
                if eff <= _ZERO or eff >= _ONE:
                    errors.append(
                        f"CHP {eff_key} must be between 0 and 1 (exclusive)"
                    )

    def _resolve_facility(
        self, facility_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Stage 2: Resolve facility information from database.

        Args:
            facility_id: Facility identifier.

        Returns:
            Facility info dict or None if not found.
        """
        if self._db_engine and hasattr(self._db_engine, "get_facility"):
            try:
                return self._db_engine.get_facility(facility_id)
            except Exception as exc:
                logger.warning(
                    "Facility lookup failed for %s: %s", facility_id, exc,
                )
        return {"facility_id": facility_id, "source": "passthrough"}

    def _resolve_supplier(
        self, supplier_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Stage 3: Resolve steam supplier information.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Supplier info dict or None if not found.
        """
        if self._db_engine and hasattr(self._db_engine, "get_supplier"):
            try:
                return self._db_engine.get_supplier(supplier_id)
            except Exception as exc:
                logger.warning(
                    "Supplier lookup failed for %s: %s", supplier_id, exc,
                )
        return None

    def _load_emission_factors(
        self,
        energy_type: str,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 4: Load emission factors appropriate for the energy type.

        Dispatches to fuel-based, regional, or COP-based EF loading
        depending on the energy type.

        Args:
            energy_type: Normalized energy type.
            request: Full request dict for parameter extraction.

        Returns:
            Emission factor data dict with source and factors.
        """
        if energy_type in ("steam", "process_heat", "chp"):
            return self._load_fuel_emission_factors(request)
        elif energy_type in ("district_heating", "hot_water"):
            return self._load_heating_emission_factors(request)
        elif energy_type in ("district_cooling", "chilled_water"):
            return self._load_cooling_parameters(request)
        return {"source": "none", "factors": {}}

    def _load_fuel_emission_factors(
        self, request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load fuel-based emission factors for steam/CHP.

        Args:
            request: Request dict with fuel_type.

        Returns:
            Emission factor dict with CO2, CH4, N2O per GJ.
        """
        fuel_type = str(request.get("fuel_type", "natural_gas")).lower()

        # Try database engine
        if self._db_engine and hasattr(self._db_engine, "get_fuel_emission_factor"):
            try:
                db_result = self._db_engine.get_fuel_emission_factor(fuel_type)
                if db_result:
                    return {
                        "source": "database",
                        "fuel_type": fuel_type,
                        "factors": db_result,
                    }
            except Exception as exc:
                logger.warning("Fuel EF DB lookup failed: %s", exc)

        # Fallback to built-in table
        fallback = FALLBACK_FUEL_EFS.get(
            fuel_type, FALLBACK_FUEL_EFS["natural_gas"]
        )
        return {
            "source": "fallback",
            "fuel_type": fuel_type,
            "factors": dict(fallback),
        }

    def _load_heating_emission_factors(
        self, request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load regional emission factors for district heating.

        Args:
            request: Request dict with region.

        Returns:
            Emission factor dict with kgCO2e/GJ and loss percentage.
        """
        region = str(request.get("region", "global")).lower()

        # Try database engine
        if self._db_engine and hasattr(self._db_engine, "get_district_heating_factor"):
            try:
                db_result = self._db_engine.get_district_heating_factor(region)
                if db_result:
                    return {
                        "source": "database",
                        "region": region,
                        "factors": db_result,
                    }
            except Exception as exc:
                logger.warning("DH EF DB lookup failed: %s", exc)

        # Fallback to built-in table
        fallback = FALLBACK_DH_FACTORS.get(
            region, FALLBACK_DH_FACTORS["global"]
        )
        return {
            "source": "fallback",
            "region": region,
            "factors": dict(fallback),
        }

    def _load_cooling_parameters(
        self, request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load cooling system parameters (COP, grid EF).

        Args:
            request: Request dict with cooling_technology and cop.

        Returns:
            Cooling parameter dict with COP and grid EF.
        """
        technology = str(request.get("cooling_technology", "centrifugal_chiller")).lower()
        cop = request.get("cop")
        grid_ef = request.get("grid_ef_kwh")

        # Try database engine
        if self._db_engine and hasattr(self._db_engine, "get_cooling_parameters"):
            try:
                db_result = self._db_engine.get_cooling_parameters(technology)
                if db_result:
                    return {
                        "source": "database",
                        "technology": technology,
                        "factors": db_result,
                    }
            except Exception as exc:
                logger.warning("Cooling params DB lookup failed: %s", exc)

        # Fallback
        resolved_cop = _safe_decimal(cop) if cop is not None else (
            FALLBACK_COOLING_COP.get(technology, Decimal("5.0"))
        )
        resolved_grid_ef = _safe_decimal(grid_ef) if grid_ef is not None else DEFAULT_GRID_EF_KWH

        return {
            "source": "fallback",
            "technology": technology,
            "factors": {
                "cop": resolved_cop,
                "grid_ef_kwh": resolved_grid_ef,
            },
        }

    def _convert_units(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: Convert energy quantity to GJ (standard basis).

        Args:
            request: Request dict with quantity and unit.

        Returns:
            Unit conversion result with energy_gj.
        """
        quantity = _safe_decimal(request.get("quantity"))
        unit = str(request.get("unit", "gj")).lower()

        conversion_factor = UNIT_TO_GJ.get(unit, _ONE)
        energy_gj = _quantize(quantity * conversion_factor)

        return {
            "original_quantity": quantity,
            "original_unit": unit,
            "conversion_factor": conversion_factor,
            "energy_gj": energy_gj,
        }

    def _calculate_emissions(
        self,
        energy_type: str,
        request: Dict[str, Any],
        emission_factors: Dict[str, Any],
        energy_gj: Decimal,
    ) -> Dict[str, Any]:
        """Stage 6: Calculate emissions by dispatching to Engine 2 or 3.

        Args:
            energy_type: Normalized energy type.
            request: Full request dict.
            emission_factors: Loaded emission factors from stage 4.
            energy_gj: Converted energy in GJ from stage 5.

        Returns:
            Calculation result dict with per-gas and total CO2e.
        """
        if energy_type in ("steam", "process_heat", "chp"):
            return self._calculate_steam_emissions(
                request, emission_factors, energy_gj,
            )
        elif energy_type in ("district_heating", "hot_water"):
            return self._calculate_heating_emissions(
                request, emission_factors, energy_gj,
            )
        elif energy_type in ("district_cooling", "chilled_water"):
            return self._calculate_cooling_emissions(
                request, emission_factors, energy_gj,
            )
        return self._zero_result()

    def _calculate_steam_emissions(
        self,
        request: Dict[str, Any],
        emission_factors: Dict[str, Any],
        energy_gj: Decimal,
    ) -> Dict[str, Any]:
        """Calculate steam emissions using Engine 2 or fallback.

        Implements the fuel-based method:
          fuel_input_gj = energy_gj / boiler_efficiency
          CO2 = fuel_input_gj * co2_ef
          CH4 = fuel_input_gj * ch4_ef
          N2O = fuel_input_gj * n2o_ef

        Adjusts for condensate return percentage.

        Args:
            request: Request dict.
            emission_factors: Fuel EF dict.
            energy_gj: Delivered thermal energy in GJ.

        Returns:
            Per-gas emission result dict in kg.
        """
        # Try Engine 2
        if self._steam_engine and hasattr(self._steam_engine, "calculate"):
            try:
                return self._steam_engine.calculate(request, emission_factors, energy_gj)
            except Exception as exc:
                logger.warning("SteamEmissionsCalculatorEngine failed: %s", exc)

        # Fallback calculation
        factors = emission_factors.get("factors", {})
        boiler_eff = _safe_decimal(
            request.get("boiler_efficiency"),
            _safe_decimal(factors.get("default_efficiency"), Decimal("0.85")),
        )
        condensate_pct = _safe_decimal(request.get("condensate_return_pct"))

        # Adjust for condensate return (reduces effective steam demand)
        effective_energy_gj = energy_gj
        if condensate_pct > _ZERO:
            effective_energy_gj = _quantize(energy_gj * (_ONE - condensate_pct * Decimal("0.15")))

        # Calculate fuel input
        if boiler_eff <= _ZERO:
            boiler_eff = Decimal("0.85")
        fuel_input_gj = _quantize(effective_energy_gj / boiler_eff)

        # Per-gas emissions (kg)
        co2_ef = _safe_decimal(factors.get("co2_ef"), Decimal("56.100"))
        ch4_ef = _safe_decimal(factors.get("ch4_ef"), Decimal("0.001"))
        n2o_ef = _safe_decimal(factors.get("n2o_ef"), Decimal("0.0001"))

        co2_kg = _quantize(fuel_input_gj * co2_ef)
        ch4_kg = _quantize(fuel_input_gj * ch4_ef)
        n2o_kg = _quantize(fuel_input_gj * n2o_ef)

        return {
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "total_co2e_kg": _ZERO,  # Computed in gas_breakdown stage
            "fuel_input_gj": fuel_input_gj,
            "boiler_efficiency": boiler_eff,
            "condensate_return_pct": condensate_pct,
            "method": "fuel_based",
        }

    def _calculate_heating_emissions(
        self,
        request: Dict[str, Any],
        emission_factors: Dict[str, Any],
        energy_gj: Decimal,
    ) -> Dict[str, Any]:
        """Calculate district heating emissions using Engine 3 or fallback.

        Implements: total_co2e = energy_gj / (1 - dist_loss) * ef_per_gj

        Args:
            request: Request dict.
            emission_factors: Regional EF dict.
            energy_gj: Delivered thermal energy in GJ.

        Returns:
            Emission result dict.
        """
        # Try Engine 3
        if self._heat_cool_engine and hasattr(self._heat_cool_engine, "calculate_heating"):
            try:
                return self._heat_cool_engine.calculate_heating(
                    request, emission_factors, energy_gj,
                )
            except Exception as exc:
                logger.warning("HeatCoolingCalculatorEngine.calculate_heating failed: %s", exc)

        # Fallback calculation
        factors = emission_factors.get("factors", {})
        ef_per_gj = _safe_decimal(factors.get("ef_kgco2e_per_gj"), Decimal("70.0"))
        default_loss = _safe_decimal(factors.get("loss_pct"), Decimal("0.12"))
        dist_loss = _safe_decimal(request.get("distribution_loss_pct"), default_loss)

        # Adjust for distribution loss
        denominator = _ONE - dist_loss
        if denominator <= _ZERO:
            denominator = Decimal("0.88")
        adjusted_gj = _quantize(energy_gj / denominator)

        # Total emissions (kgCO2e) -- composite EF already in CO2e
        total_co2e_kg = _quantize(adjusted_gj * ef_per_gj)

        # Approximate per-gas (97% CO2, 2% CH4, 1% N2O of CO2e)
        co2_kg = _quantize(total_co2e_kg * Decimal("0.97"))
        ch4_kg_co2e = _quantize(total_co2e_kg * Decimal("0.02"))
        n2o_kg_co2e = _quantize(total_co2e_kg * Decimal("0.01"))

        return {
            "co2_kg": co2_kg,
            "ch4_kg": _ZERO,
            "n2o_kg": _ZERO,
            "ch4_co2e_kg": ch4_kg_co2e,
            "n2o_co2e_kg": n2o_kg_co2e,
            "total_co2e_kg": total_co2e_kg,
            "adjusted_energy_gj": adjusted_gj,
            "distribution_loss_pct": dist_loss,
            "ef_kgco2e_per_gj": ef_per_gj,
            "method": "direct_ef",
        }

    def _calculate_cooling_emissions(
        self,
        request: Dict[str, Any],
        emission_factors: Dict[str, Any],
        energy_gj: Decimal,
    ) -> Dict[str, Any]:
        """Calculate district cooling emissions using Engine 3 or fallback.

        Implements COP-based method:
          electrical_input_kwh = cooling_gj * 277.778 / COP
          total_co2e = electrical_input_kwh * grid_ef_kwh

        Args:
            request: Request dict.
            emission_factors: Cooling parameter dict.
            energy_gj: Cooling output in GJ.

        Returns:
            Emission result dict.
        """
        # Try Engine 3
        if self._heat_cool_engine and hasattr(self._heat_cool_engine, "calculate_cooling"):
            try:
                return self._heat_cool_engine.calculate_cooling(
                    request, emission_factors, energy_gj,
                )
            except Exception as exc:
                logger.warning("HeatCoolingCalculatorEngine.calculate_cooling failed: %s", exc)

        # Fallback calculation
        factors = emission_factors.get("factors", {})
        cop = _safe_decimal(factors.get("cop"), Decimal("5.0"))
        grid_ef_kwh = _safe_decimal(factors.get("grid_ef_kwh"), DEFAULT_GRID_EF_KWH)

        if cop <= _ZERO:
            cop = Decimal("5.0")

        # Convert cooling output to electrical input
        cooling_kwh = _quantize(energy_gj * Decimal("277.778"))
        electrical_input_kwh = _quantize(cooling_kwh / cop)

        # Total emissions from grid electricity
        total_co2e_kg = _quantize(electrical_input_kwh * grid_ef_kwh)

        # Approximate per-gas (same as grid electricity breakdown)
        co2_kg = _quantize(total_co2e_kg * Decimal("0.97"))
        ch4_kg_co2e = _quantize(total_co2e_kg * Decimal("0.02"))
        n2o_kg_co2e = _quantize(total_co2e_kg * Decimal("0.01"))

        return {
            "co2_kg": co2_kg,
            "ch4_kg": _ZERO,
            "n2o_kg": _ZERO,
            "ch4_co2e_kg": ch4_kg_co2e,
            "n2o_co2e_kg": n2o_kg_co2e,
            "total_co2e_kg": total_co2e_kg,
            "cooling_kwh": cooling_kwh,
            "electrical_input_kwh": electrical_input_kwh,
            "cop": cop,
            "grid_ef_kwh": grid_ef_kwh,
            "method": "cop_based",
        }

    def _apply_chp_allocation(
        self,
        calc_result: Dict[str, Any],
        chp_params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Stage 7: Apply CHP allocation via Engine 4 or fallback.

        Allocates total CHP plant emissions between electrical and
        thermal outputs using the specified method.

        Args:
            calc_result: Raw emission calculation result.
            chp_params: CHP allocation parameters.

        Returns:
            CHP allocation result dict or None.
        """
        # Try Engine 4
        if self._chp_engine and hasattr(self._chp_engine, "allocate"):
            try:
                return self._chp_engine.allocate(calc_result, chp_params)
            except Exception as exc:
                logger.warning("CHPAllocationEngine failed: %s", exc)

        # Fallback CHP allocation
        method = chp_params.get("allocation_method", "efficiency")
        fuel_type = chp_params.get("fuel_type", "natural_gas")
        defaults = FALLBACK_CHP_EFFICIENCIES.get(
            fuel_type, FALLBACK_CHP_EFFICIENCIES["natural_gas"]
        )

        elec_eff = _safe_decimal(
            chp_params.get("electrical_efficiency"),
            defaults["electrical_efficiency"],
        )
        thermal_eff = _safe_decimal(
            chp_params.get("thermal_efficiency"),
            defaults["thermal_efficiency"],
        )

        total_co2e_kg = _safe_decimal(calc_result.get("total_co2e_kg"))
        if total_co2e_kg <= _ZERO:
            # Recalculate from per-gas
            co2_kg = _safe_decimal(calc_result.get("co2_kg"))
            ch4_kg = _safe_decimal(calc_result.get("ch4_kg"))
            n2o_kg = _safe_decimal(calc_result.get("n2o_kg"))
            gwp = GWP_TABLE.get("AR5", GWP_TABLE["AR5"])
            total_co2e_kg = (
                co2_kg * gwp["CO2"]
                + ch4_kg * gwp["CH4"]
                + n2o_kg * gwp["N2O"]
            )

        # Calculate thermal share based on allocation method
        if method == "efficiency":
            thermal_share = self._chp_efficiency_allocation(
                elec_eff, thermal_eff,
            )
        elif method == "energy":
            thermal_share = self._chp_energy_allocation(chp_params)
        elif method == "exergy":
            thermal_share = self._chp_exergy_allocation(chp_params)
        else:
            thermal_share = self._chp_efficiency_allocation(
                elec_eff, thermal_eff,
            )

        thermal_co2e_kg = _quantize(total_co2e_kg * thermal_share)
        electrical_co2e_kg = _quantize(total_co2e_kg * (_ONE - thermal_share))

        if self._metrics and _METRICS_AVAILABLE:
            try:
                tenant_id = chp_params.get("tenant_id", "")
                _get_metrics().record_chp_allocation(method, fuel_type, tenant_id)
            except Exception:
                pass

        return {
            "allocation_method": method,
            "thermal_share": _quantize(thermal_share),
            "electrical_share": _quantize(_ONE - thermal_share),
            "total_co2e_kg": _quantize(total_co2e_kg),
            "thermal_co2e_kg": thermal_co2e_kg,
            "electrical_co2e_kg": electrical_co2e_kg,
            "electrical_efficiency": elec_eff,
            "thermal_efficiency": thermal_eff,
            "fuel_type": fuel_type,
        }

    def _chp_efficiency_allocation(
        self,
        elec_eff: Decimal,
        thermal_eff: Decimal,
    ) -> Decimal:
        """Compute thermal share using efficiency-based allocation.

        Formula: thermal_share = (Q_heat/eta_thermal) /
            ((Q_heat/eta_thermal) + (Q_elec/eta_elec))

        For simplicity using unit outputs:
            thermal_share = (1/eta_thermal) / ((1/eta_thermal) + (1/eta_elec))

        Args:
            elec_eff: Electrical efficiency (0-1).
            thermal_eff: Thermal efficiency (0-1).

        Returns:
            Thermal share as Decimal (0-1).
        """
        if elec_eff <= _ZERO or thermal_eff <= _ZERO:
            return Decimal("0.50")

        thermal_input = _ONE / thermal_eff
        electrical_input = _ONE / elec_eff
        total_input = thermal_input + electrical_input

        if total_input <= _ZERO:
            return Decimal("0.50")

        return _quantize(thermal_input / total_input)

    def _chp_energy_allocation(
        self, chp_params: Dict[str, Any],
    ) -> Decimal:
        """Compute thermal share using energy-based allocation.

        Simply proportional to thermal output / total output (in GJ).

        Args:
            chp_params: CHP parameters with thermal_output_gj and
                electrical_output_gj.

        Returns:
            Thermal share as Decimal (0-1).
        """
        thermal_gj = _safe_decimal(chp_params.get("thermal_output_gj"), _ONE)
        electrical_gj = _safe_decimal(chp_params.get("electrical_output_gj"), _ONE)
        total = thermal_gj + electrical_gj

        if total <= _ZERO:
            return Decimal("0.50")

        return _quantize(thermal_gj / total)

    def _chp_exergy_allocation(
        self, chp_params: Dict[str, Any],
    ) -> Decimal:
        """Compute thermal share using exergy-based allocation.

        Exergy of heat = Q * (1 - T_ambient / T_supply) (Carnot factor).
        Exergy of electricity = 1.0 (by definition).

        Args:
            chp_params: CHP parameters with temperatures.

        Returns:
            Thermal share as Decimal (0-1).
        """
        t_supply_k = _safe_decimal(
            chp_params.get("supply_temperature_k"), Decimal("423.15")
        )
        t_ambient_k = _safe_decimal(
            chp_params.get("ambient_temperature_k"), Decimal("288.15")
        )
        thermal_gj = _safe_decimal(chp_params.get("thermal_output_gj"), _ONE)
        electrical_gj = _safe_decimal(chp_params.get("electrical_output_gj"), _ONE)

        if t_supply_k <= _ZERO:
            t_supply_k = Decimal("423.15")

        carnot_factor = _quantize(_ONE - (t_ambient_k / t_supply_k))
        if carnot_factor < _ZERO:
            carnot_factor = _ZERO

        thermal_exergy = _quantize(thermal_gj * carnot_factor)
        electrical_exergy = electrical_gj  # Exergy of electricity = 1:1
        total_exergy = thermal_exergy + electrical_exergy

        if total_exergy <= _ZERO:
            return Decimal("0.50")

        return _quantize(thermal_exergy / total_exergy)

    def _merge_chp_into_result(
        self,
        calc_result: Dict[str, Any],
        chp_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge CHP allocation into the calculation result.

        Replaces total_co2e_kg with the thermal-only portion for
        Scope 2 reporting (thermal output is the purchased heat).

        Args:
            calc_result: Original calculation result.
            chp_result: CHP allocation result.

        Returns:
            Updated calculation result.
        """
        merged = dict(calc_result)
        merged["total_co2e_kg"] = chp_result["thermal_co2e_kg"]
        merged["chp_total_co2e_kg"] = chp_result["total_co2e_kg"]
        merged["chp_electrical_co2e_kg"] = chp_result["electrical_co2e_kg"]
        merged["chp_thermal_share"] = chp_result["thermal_share"]
        merged["chp_allocation_method"] = chp_result["allocation_method"]
        return merged

    def _separate_biogenic(
        self,
        calc_result: Dict[str, Any],
        emission_factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 8: Separate biogenic CO2 from fossil CO2.

        Per GHG Protocol guidance, biogenic CO2 from sustainably sourced
        biomass is reported separately and not included in Scope 2 totals.

        Args:
            calc_result: Emission calculation result.
            emission_factors: Loaded emission factors.

        Returns:
            Biogenic separation result dict.
        """
        factors = emission_factors.get("factors", {})
        is_biogenic = _safe_decimal(factors.get("is_biogenic")) > _ZERO

        if not is_biogenic:
            co2_kg = _safe_decimal(calc_result.get("co2_kg"))
            return {
                "biogenic_co2_kg": _ZERO,
                "fossil_co2_kg": co2_kg,
                "is_biogenic_fuel": False,
                "biogenic_fraction": _ZERO,
            }

        # For biogenic fuels, CO2 is biogenic; only CH4 and N2O are fossil
        co2_kg = _safe_decimal(calc_result.get("co2_kg"))
        biogenic_co2_ef = _safe_decimal(factors.get("biogenic_co2_ef"))
        fuel_input_gj = _safe_decimal(calc_result.get("fuel_input_gj"))

        biogenic_co2_kg = _ZERO
        if biogenic_co2_ef > _ZERO and fuel_input_gj > _ZERO:
            biogenic_co2_kg = _quantize(fuel_input_gj * biogenic_co2_ef)
        elif co2_kg > _ZERO:
            biogenic_co2_kg = co2_kg

        return {
            "biogenic_co2_kg": biogenic_co2_kg,
            "fossil_co2_kg": _ZERO,
            "is_biogenic_fuel": True,
            "biogenic_fraction": _ONE,
        }

    def _compute_gas_breakdown(
        self,
        calc_result: Dict[str, Any],
        gwp_source: str,
    ) -> List[Dict[str, Any]]:
        """Stage 9: Compute per-gas breakdown with GWP conversion.

        Applies GWP factors to convert CH4 and N2O mass emissions
        to CO2e for aggregated reporting.

        Args:
            calc_result: Emission calculation result.
            gwp_source: IPCC GWP assessment report.

        Returns:
            List of per-gas breakdown dicts.
        """
        gwp = GWP_TABLE.get(gwp_source, GWP_TABLE["AR5"])

        co2_kg = _safe_decimal(calc_result.get("co2_kg"))
        ch4_kg = _safe_decimal(calc_result.get("ch4_kg"))
        n2o_kg = _safe_decimal(calc_result.get("n2o_kg"))

        co2_co2e = _quantize(co2_kg * gwp["CO2"])
        ch4_co2e = _quantize(ch4_kg * gwp["CH4"])
        n2o_co2e = _quantize(n2o_kg * gwp["N2O"])

        # If the calculator already produced total_co2e_kg (e.g. heating/cooling),
        # honour it and use pre-computed CO2e
        precomputed_co2e = _safe_decimal(calc_result.get("total_co2e_kg"))
        gas_total = co2_co2e + ch4_co2e + n2o_co2e

        # Use the larger of gas-sum or precomputed (precomputed may include
        # additional adjustments like distribution loss)
        if precomputed_co2e > gas_total:
            total_co2e = precomputed_co2e
        else:
            total_co2e = gas_total

        # Update calc_result in place for downstream use
        calc_result["total_co2e_kg"] = total_co2e

        # Also set pre-computed CO2e from ch4/n2o if present
        ch4_co2e_pre = _safe_decimal(calc_result.get("ch4_co2e_kg"))
        n2o_co2e_pre = _safe_decimal(calc_result.get("n2o_co2e_kg"))
        if ch4_co2e_pre > _ZERO:
            ch4_co2e = ch4_co2e_pre
        if n2o_co2e_pre > _ZERO:
            n2o_co2e = n2o_co2e_pre

        breakdown: List[Dict[str, Any]] = [
            {
                "gas": "CO2",
                "emission_kg": co2_kg,
                "gwp_factor": gwp["CO2"],
                "co2e_kg": co2_co2e,
            },
            {
                "gas": "CH4",
                "emission_kg": ch4_kg,
                "gwp_factor": gwp["CH4"],
                "co2e_kg": ch4_co2e,
            },
            {
                "gas": "N2O",
                "emission_kg": n2o_kg,
                "gwp_factor": gwp["N2O"],
                "co2e_kg": n2o_co2e,
            },
        ]

        return breakdown

    def _quantify_uncertainty(
        self,
        calc_result: Dict[str, Any],
        energy_type: str,
        mc_iterations: int,
    ) -> Dict[str, Any]:
        """Stage 10: Quantify uncertainty via Engine 5 or fallback.

        Args:
            calc_result: Emission calculation result.
            energy_type: Energy type for uncertainty parameter selection.
            mc_iterations: Number of Monte Carlo iterations.

        Returns:
            Uncertainty result dict.
        """
        base_co2e_kg = _safe_decimal(calc_result.get("total_co2e_kg"))

        # Try Engine 5
        if self._uncertainty_engine and hasattr(self._uncertainty_engine, "quantify"):
            try:
                result = self._uncertainty_engine.quantify(
                    calc_result, energy_type, mc_iterations,
                )
                if self._metrics and _METRICS_AVAILABLE:
                    try:
                        _get_metrics().record_uncertainty(
                            result.get("method", "monte_carlo"), "",
                        )
                    except Exception:
                        pass
                return result
            except Exception as exc:
                logger.warning("UncertaintyQuantifierEngine failed: %s", exc)

        # Fallback: IPCC default uncertainty ranges
        return self._fallback_uncertainty(base_co2e_kg, energy_type, mc_iterations)

    def _fallback_uncertainty(
        self,
        base_co2e_kg: Decimal,
        energy_type: str,
        iterations: int,
    ) -> Dict[str, Any]:
        """Provide conservative uncertainty estimate using IPCC defaults.

        Uncertainty ranges by energy type:
        - Steam (supplier-specific): EF +/-5%, AD +/-3%
        - Steam (default EF): EF +/-10%, AD +/-3%
        - District heating: EF +/-15%, AD +/-5%
        - District cooling: COP +/-10%, grid EF +/-10%, AD +/-3%

        Args:
            base_co2e_kg: Base emissions in kg CO2e.
            energy_type: Energy type for parameter selection.
            iterations: Requested iterations (informational).

        Returns:
            Uncertainty result dict.
        """
        uncertainty_params = {
            "steam": {"ef_pct": Decimal("0.10"), "ad_pct": Decimal("0.03")},
            "process_heat": {"ef_pct": Decimal("0.10"), "ad_pct": Decimal("0.03")},
            "district_heating": {"ef_pct": Decimal("0.15"), "ad_pct": Decimal("0.05")},
            "hot_water": {"ef_pct": Decimal("0.15"), "ad_pct": Decimal("0.05")},
            "district_cooling": {"ef_pct": Decimal("0.12"), "ad_pct": Decimal("0.05")},
            "chilled_water": {"ef_pct": Decimal("0.12"), "ad_pct": Decimal("0.05")},
            "chp": {"ef_pct": Decimal("0.12"), "ad_pct": Decimal("0.05")},
        }

        params = uncertainty_params.get(energy_type, {
            "ef_pct": Decimal("0.10"), "ad_pct": Decimal("0.05"),
        })

        ef_pct = params["ef_pct"]
        ad_pct = params["ad_pct"]
        combined_pct = _quantize(
            (ef_pct ** 2 + ad_pct ** 2).sqrt()
        )

        z_score = Decimal("1.96")  # 95% confidence
        lower = _quantize(base_co2e_kg * (_ONE - combined_pct * z_score))
        upper = _quantize(base_co2e_kg * (_ONE + combined_pct * z_score))

        return {
            "method": "ipcc_default_uncertainty",
            "iterations": iterations,
            "base_co2e_kg": base_co2e_kg,
            "ef_uncertainty_pct": ef_pct,
            "activity_data_uncertainty_pct": ad_pct,
            "combined_uncertainty_pct": combined_pct,
            "confidence_interval": "95%",
            "lower_bound_kg": max(lower, _ZERO),
            "upper_bound_kg": upper,
            "note": (
                f"IPCC default uncertainty for {energy_type}. "
                "Use Engine 5 for Monte Carlo analysis."
            ),
        }

    def _check_compliance(
        self,
        calc_result: Dict[str, Any],
        energy_type: str,
        request: Dict[str, Any],
        frameworks: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Stage 11: Run regulatory compliance checks via Engine 6.

        Args:
            calc_result: Emission calculation result.
            energy_type: Energy type.
            request: Original request dict.
            frameworks: List of framework names or None.

        Returns:
            List of compliance check result dicts.
        """
        check_data = {
            "energy_type": energy_type,
            "total_co2e_kg": calc_result.get("total_co2e_kg", _ZERO),
            "method": calc_result.get("method", ""),
            "request": request,
            "calc_result": calc_result,
        }

        # Try Engine 6
        if self._compliance_engine and hasattr(self._compliance_engine, "check_compliance"):
            try:
                checks = self._compliance_engine.check_compliance(check_data, frameworks)
                if self._metrics and _METRICS_AVAILABLE:
                    for c in checks:
                        try:
                            _get_metrics().record_compliance_check(
                                c.get("framework", "UNKNOWN"),
                                c.get("status", "not_assessed"),
                                request.get("tenant_id", ""),
                            )
                        except Exception:
                            pass
                return checks
            except Exception as exc:
                logger.warning("ComplianceCheckerEngine failed: %s", exc)

        # Fallback: basic compliance checks
        return self._fallback_compliance_checks(
            calc_result, energy_type, request, frameworks,
        )

    def _fallback_compliance_checks(
        self,
        calc_result: Dict[str, Any],
        energy_type: str,
        request: Dict[str, Any],
        frameworks: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Run basic built-in compliance checks as fallback.

        Args:
            calc_result: Calculation result.
            energy_type: Energy type.
            request: Original request.
            frameworks: Requested frameworks.

        Returns:
            List of compliance check dicts.
        """
        checks: List[Dict[str, Any]] = []
        requested = set(frameworks) if frameworks else {"GHG_PROTOCOL"}

        if "GHG_PROTOCOL" in requested:
            findings: List[str] = []
            status = "compliant"

            method = calc_result.get("method", "")
            if not method:
                findings.append("Calculation method not specified")
                status = "partial"

            total_co2e = _safe_decimal(calc_result.get("total_co2e_kg"))
            if total_co2e <= _ZERO:
                findings.append("Total CO2e is zero; verify input data")
                status = "partial"

            if energy_type in ("steam", "process_heat") and not request.get("supplier_id"):
                findings.append(
                    "Supplier-specific data preferred over default EFs "
                    "for GHG Protocol Scope 2 steam reporting"
                )

            checks.append({
                "framework": "GHG_PROTOCOL",
                "status": status,
                "findings": findings,
                "checked_at": _utcnow().isoformat(),
            })

        if "ISO_14064" in requested:
            findings_iso: List[str] = []
            status_iso = "compliant"

            if not request.get("facility_id"):
                findings_iso.append("Facility identification required")
                status_iso = "non_compliant"

            checks.append({
                "framework": "ISO_14064",
                "status": status_iso,
                "findings": findings_iso,
                "checked_at": _utcnow().isoformat(),
            })

        if "CSRD_ESRS_E1" in requested:
            findings_csrd: List[str] = []
            status_csrd = "compliant"

            if energy_type == "chp" and not request.get("chp_params"):
                findings_csrd.append(
                    "CHP allocation methodology must be disclosed under ESRS E1"
                )
                status_csrd = "partial"

            checks.append({
                "framework": "CSRD_ESRS_E1",
                "status": status_csrd,
                "findings": findings_csrd,
                "checked_at": _utcnow().isoformat(),
            })

        return checks

    def _assemble_result(
        self,
        calc_id: str,
        tenant_id: str,
        facility_id: str,
        energy_type: str,
        gwp_source: str,
        energy_gj: Decimal,
        unit_result: Dict[str, Any],
        calc_result: Dict[str, Any],
        biogenic_result: Dict[str, Any],
        gas_breakdown: List[Dict[str, Any]],
        chp_result: Optional[Dict[str, Any]],
        uncertainty_result: Optional[Dict[str, Any]],
        compliance_results: List[Dict[str, Any]],
        emission_intensity: Decimal,
        request: Dict[str, Any],
        calculation_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Stage 12: Assemble the final calculation result.

        Combines all stage outputs into a single comprehensive result
        dict suitable for storage, API response, and audit purposes.

        Args:
            calc_id: Calculation identifier.
            tenant_id: Tenant identifier.
            facility_id: Facility identifier.
            energy_type: Normalized energy type.
            gwp_source: GWP assessment report used.
            energy_gj: Energy in GJ.
            unit_result: Unit conversion result.
            calc_result: Emission calculation result.
            biogenic_result: Biogenic separation result.
            gas_breakdown: Per-gas breakdown.
            chp_result: CHP allocation result or None.
            uncertainty_result: Uncertainty result or None.
            compliance_results: List of compliance results.
            emission_intensity: kgCO2e per GJ.
            request: Original request.
            calculation_trace: Stage timing trace.

        Returns:
            Complete result dict.
        """
        total_co2e_kg = _safe_decimal(calc_result.get("total_co2e_kg"))
        total_co2e_tonnes = _quantize(total_co2e_kg / _THOUSAND)

        return {
            "calculation_id": calc_id,
            "tenant_id": tenant_id,
            "facility_id": facility_id,
            "energy_type": energy_type,
            "method": calc_result.get("method", ""),
            "gwp_source": gwp_source,
            # Energy data
            "original_quantity": unit_result.get("original_quantity", _ZERO),
            "original_unit": unit_result.get("original_unit", "gj"),
            "energy_gj": energy_gj,
            # Emissions
            "co2_kg": _safe_decimal(calc_result.get("co2_kg")),
            "ch4_kg": _safe_decimal(calc_result.get("ch4_kg")),
            "n2o_kg": _safe_decimal(calc_result.get("n2o_kg")),
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            # Biogenic
            "biogenic_co2_kg": biogenic_result.get("biogenic_co2_kg", _ZERO),
            "fossil_co2_kg": biogenic_result.get("fossil_co2_kg", _ZERO),
            "is_biogenic_fuel": biogenic_result.get("is_biogenic_fuel", False),
            # Gas breakdown
            "gas_breakdown": gas_breakdown,
            # Intensity
            "emission_intensity_kgco2e_per_gj": emission_intensity,
            # CHP
            "chp_allocation": chp_result,
            # Method-specific fields
            "boiler_efficiency": calc_result.get("boiler_efficiency"),
            "condensate_return_pct": calc_result.get("condensate_return_pct"),
            "fuel_input_gj": calc_result.get("fuel_input_gj"),
            "distribution_loss_pct": calc_result.get("distribution_loss_pct"),
            "cop": calc_result.get("cop"),
            "grid_ef_kwh": calc_result.get("grid_ef_kwh"),
            "electrical_input_kwh": calc_result.get("electrical_input_kwh"),
            # Uncertainty
            "uncertainty": uncertainty_result,
            # Compliance
            "compliance_results": compliance_results,
            # Provenance (set in stage 13)
            "provenance_hash": "",
            # Metadata
            "calculated_at": _utcnow().isoformat(),
            "calculation_trace": calculation_trace,
            "metadata": {
                "pipeline_version": PIPELINE_VERSION,
                "stages_completed": len(calculation_trace),
                "fuel_type": request.get("fuel_type", ""),
                "supplier_id": request.get("supplier_id", ""),
                "region": request.get("region", ""),
                "cooling_technology": request.get("cooling_technology", ""),
            },
        }

    def _seal_provenance(
        self,
        prov: Any,
        chain_id: Optional[str],
        calc_id: str,
        result: Dict[str, Any],
    ) -> str:
        """Stage 13: Seal the provenance chain with final SHA-256 hash.

        Args:
            prov: SteamHeatPurchaseProvenance instance or None.
            chain_id: Provenance chain ID or None.
            calc_id: Calculation identifier.
            result: Assembled result dict.

        Returns:
            SHA-256 hex digest string.
        """
        if prov and chain_id:
            try:
                return prov.seal_chain(chain_id)
            except Exception as exc:
                logger.warning("Provenance seal failed: %s", exc)

        # Fallback: compute hash from result
        return self._compute_result_hash(calc_id, result)

    # ==================================================================
    # Private Helpers
    # ==================================================================

    @staticmethod
    def _zero_result() -> Dict[str, Any]:
        """Return a zeroed calculation result.

        Returns:
            Dict with all emission values set to zero.
        """
        return {
            "co2_kg": _ZERO,
            "ch4_kg": _ZERO,
            "n2o_kg": _ZERO,
            "total_co2e_kg": _ZERO,
            "method": "none",
        }

    @staticmethod
    def _trace_entry(stage: str, duration_s: float) -> Dict[str, Any]:
        """Create a calculation trace entry for timing.

        Args:
            stage: Pipeline stage name.
            duration_s: Duration in seconds.

        Returns:
            Trace entry dict with stage, duration_ms, and timestamp.
        """
        return {
            "stage": stage,
            "duration_ms": round(duration_s * 1000, 3),
            "timestamp": _utcnow().isoformat(),
        }

    @staticmethod
    def _compute_result_hash(
        calc_id: str, result: Dict[str, Any],
    ) -> str:
        """Compute a fallback provenance hash from the result.

        Args:
            calc_id: Calculation identifier.
            result: Result dict to hash.

        Returns:
            SHA-256 hex digest string.
        """
        payload = json.dumps(
            {
                "calculation_id": calc_id,
                "total_co2e_kg": str(result.get("total_co2e_kg", _ZERO)),
                "energy_type": result.get("energy_type", ""),
                "energy_gj": str(result.get("energy_gj", _ZERO)),
                "calculated_at": result.get("calculated_at", ""),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_batch_hash(
        results: List[Dict[str, Any]],
    ) -> str:
        """Compute a combined provenance hash for a batch of results.

        Args:
            results: List of individual pipeline result dicts.

        Returns:
            SHA-256 hex digest string.
        """
        hashes = [r.get("provenance_hash", "") for r in results]
        combined = "|".join(hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_aggregation_hash(
        calc_ids: List[str], aggregation_type: str,
    ) -> str:
        """Compute provenance hash for an aggregation operation.

        Args:
            calc_ids: List of calculation IDs aggregated.
            aggregation_type: Aggregation dimension.

        Returns:
            SHA-256 hex digest string.
        """
        payload = json.dumps({
            "calc_ids": sorted(calc_ids),
            "aggregation_type": aggregation_type,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _aggregation_key(aggregation_type: str) -> str:
        """Map aggregation type to the result dict key to group by.

        Args:
            aggregation_type: Aggregation dimension.

        Returns:
            Result dict key name.
        """
        mapping = {
            "by_facility": "facility_id",
            "by_fuel": "fuel_type",
            "by_energy_type": "energy_type",
            "by_supplier": "supplier_id",
            "by_period": "calculated_at",
        }
        return mapping.get(aggregation_type, "facility_id")

    def _record_success(
        self,
        duration_ms: float,
        total_co2e_kg: Decimal,
        biogenic_co2_kg: Decimal,
        energy_gj: Decimal,
        energy_type: str,
        request: Dict[str, Any],
    ) -> None:
        """Record successful pipeline execution metrics.

        Args:
            duration_ms: Pipeline duration in milliseconds.
            total_co2e_kg: Total CO2e in kg.
            biogenic_co2_kg: Biogenic CO2 in kg.
            energy_gj: Energy consumed in GJ.
            energy_type: Energy type.
            request: Original request dict.
        """
        with self._lock:
            self._pipeline_runs += 1
            self._successful_runs += 1
            self._total_co2e_kg += total_co2e_kg
            self._total_biogenic_co2_kg += biogenic_co2_kg
            self._total_energy_gj += energy_gj
            self._total_duration_ms += duration_ms

        if self._metrics and _METRICS_AVAILABLE:
            try:
                tenant_id = request.get("tenant_id", "")
                fuel_type = request.get("fuel_type", "unknown")
                method = request.get("calculation_method", "default_emission_factor")
                _get_metrics().record_calculation(
                    energy_type, method, "success",
                    duration_ms / 1000, float(total_co2e_kg),
                    float(biogenic_co2_kg), fuel_type, tenant_id,
                )
            except Exception:
                pass

    def _record_failure(
        self,
        pipeline_start: float,
        energy_type: str,
        tenant_id: str,
    ) -> None:
        """Record failed pipeline execution metrics.

        Args:
            pipeline_start: Pipeline start monotonic time.
            energy_type: Energy type.
            tenant_id: Tenant identifier.
        """
        duration_ms = (time.monotonic() - pipeline_start) * 1000
        with self._lock:
            self._pipeline_runs += 1
            self._failed_runs += 1
            self._total_duration_ms += duration_ms

        if self._metrics and _METRICS_AVAILABLE:
            try:
                _get_metrics().record_error("pipeline", "calculation_error", tenant_id)
            except Exception:
                pass

    def _export_json(
        self, calcs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Export calculations as JSON format.

        Args:
            calcs: List of calculation result dicts.

        Returns:
            Export result dict.
        """
        return {
            "format": "json",
            "record_count": len(calcs),
            "data": calcs,
            "exported_at": _utcnow().isoformat(),
        }

    def _export_csv_summary(
        self, calcs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Export calculations as CSV-style summary rows.

        Args:
            calcs: List of calculation result dicts.

        Returns:
            Export result dict with header and rows.
        """
        header = [
            "calculation_id", "facility_id", "energy_type", "method",
            "energy_gj", "total_co2e_kg", "total_co2e_tonnes",
            "biogenic_co2_kg", "gwp_source", "calculated_at",
        ]
        rows: List[List[Any]] = []
        for c in calcs:
            rows.append([
                c.get("calculation_id", ""),
                c.get("facility_id", ""),
                c.get("energy_type", ""),
                c.get("method", ""),
                str(c.get("energy_gj", _ZERO)),
                str(c.get("total_co2e_kg", _ZERO)),
                str(c.get("total_co2e_tonnes", _ZERO)),
                str(c.get("biogenic_co2_kg", _ZERO)),
                c.get("gwp_source", ""),
                c.get("calculated_at", ""),
            ])

        return {
            "format": "csv_summary",
            "record_count": len(calcs),
            "header": header,
            "rows": rows,
            "exported_at": _utcnow().isoformat(),
        }

    def _export_ghg_protocol(
        self, calcs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Export calculations in GHG Protocol reporting format.

        Groups calculations by energy type and produces totals
        suitable for Scope 2 reporting templates.

        Args:
            calcs: List of calculation result dicts.

        Returns:
            Export result dict with GHG Protocol structure.
        """
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for c in calcs:
            etype = c.get("energy_type", "unknown")
            by_type.setdefault(etype, []).append(c)

        scope2_categories: List[Dict[str, Any]] = []
        grand_total = _ZERO

        for etype, group in sorted(by_type.items()):
            group_co2e = sum(_safe_decimal(c.get("total_co2e_kg")) for c in group)
            group_biogenic = sum(_safe_decimal(c.get("biogenic_co2_kg")) for c in group)
            group_energy = sum(_safe_decimal(c.get("energy_gj")) for c in group)
            grand_total += group_co2e

            scope2_categories.append({
                "category": etype,
                "facility_count": len(set(c.get("facility_id", "") for c in group)),
                "total_co2e_kg": _quantize(group_co2e),
                "total_co2e_tonnes": _quantize(group_co2e / _THOUSAND),
                "biogenic_co2_kg": _quantize(group_biogenic),
                "total_energy_gj": _quantize(group_energy),
            })

        return {
            "format": "ghg_protocol",
            "reporting_scope": "scope_2",
            "subcategory": "purchased_steam_heat_cooling",
            "categories": scope2_categories,
            "grand_total_co2e_kg": _quantize(grand_total),
            "grand_total_co2e_tonnes": _quantize(grand_total / _THOUSAND),
            "record_count": len(calcs),
            "exported_at": _utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[SteamHeatPipelineEngine] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> SteamHeatPipelineEngine:
    """Return the module-level singleton SteamHeatPipelineEngine.

    This is the recommended entry point for obtaining the pipeline
    engine in production code. The singleton is created lazily on
    first call and reused for all subsequent calls.

    Returns:
        The singleton SteamHeatPipelineEngine instance.

    Example:
        >>> from greenlang.steam_heat_purchase.steam_heat_pipeline import (
        ...     get_pipeline,
        ... )
        >>> pipeline = get_pipeline()
        >>> result = pipeline.run_pipeline({
        ...     "facility_id": "FAC-001",
        ...     "energy_type": "steam",
        ...     "quantity": 1000,
        ...     "unit": "gj",
        ...     "fuel_type": "natural_gas",
        ... })
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = SteamHeatPipelineEngine()
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["SteamHeatPipelineEngine"]
