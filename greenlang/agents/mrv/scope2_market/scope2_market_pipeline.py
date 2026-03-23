# -*- coding: utf-8 -*-
"""
Engine 7: Scope 2 Market-Based Pipeline Engine for AGENT-MRV-010.

8-stage orchestrated calculation pipeline for market-based Scope 2 emissions:
  Stage 1: Validate input (facility, tenant, energy purchases, instruments)
  Stage 2: Resolve instrument data (EFs, quality, vintage, supplier-specific)
  Stage 3: Allocate instruments (GHG Protocol hierarchy, certificate retirement)
  Stage 4: Calculate covered emissions (instrument EF x allocated MWh)
  Stage 5: Calculate uncovered emissions (residual mix x uncovered MWh)
  Stage 6: Apply GWP conversion (AR4/AR5/AR6/AR6_20YR)
  Stage 7: Run compliance checks (optional, multi-framework)
  Stage 8: Assemble results with provenance chain

GHG Protocol Scope 2 Guidance (2015) specifies a quality hierarchy for
contractual instruments used in the market-based method.  This pipeline
enforces that hierarchy during allocation (Stage 3) and tracks certificate
retirement to prevent double-counting.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Decimal arithmetic
    - Emission factors sourced from database or validated fallback tables
    - No LLM calls in any numeric calculation path
    - Provenance chain hashes every stage output with SHA-256

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for upstream engines (Engines 1-6 of MRV-010)
# ---------------------------------------------------------------------------
_InstrumentDatabaseEngine = None
_InstrumentAllocationEngine = None
_CoveredEmissionsEngine = None
_ResidualMixEngine = None
_UncertaintyQuantifierEngine = None
_ComplianceCheckerEngine = None
_Scope2MarketProvenance = None


def _lazy_import_engines() -> None:
    """Lazily import upstream engine classes to avoid circular dependencies."""
    global _InstrumentDatabaseEngine, _InstrumentAllocationEngine
    global _CoveredEmissionsEngine, _ResidualMixEngine
    global _UncertaintyQuantifierEngine, _ComplianceCheckerEngine
    global _Scope2MarketProvenance

    if _InstrumentDatabaseEngine is None:
        try:
            from greenlang.agents.mrv.scope2_market.instrument_database import (
                InstrumentDatabaseEngine,
            )
            _InstrumentDatabaseEngine = InstrumentDatabaseEngine
        except ImportError:
            pass

    if _InstrumentAllocationEngine is None:
        try:
            from greenlang.agents.mrv.scope2_market.instrument_allocation import (
                InstrumentAllocationEngine,
            )
            _InstrumentAllocationEngine = InstrumentAllocationEngine
        except ImportError:
            pass

    if _CoveredEmissionsEngine is None:
        try:
            from greenlang.agents.mrv.scope2_market.covered_emissions import (
                CoveredEmissionsEngine,
            )
            _CoveredEmissionsEngine = CoveredEmissionsEngine
        except ImportError:
            pass

    if _ResidualMixEngine is None:
        try:
            from greenlang.agents.mrv.scope2_market.residual_mix import (
                ResidualMixEngine,
            )
            _ResidualMixEngine = ResidualMixEngine
        except ImportError:
            pass

    if _UncertaintyQuantifierEngine is None:
        try:
            from greenlang.agents.mrv.scope2_market.uncertainty_quantifier import (
                UncertaintyQuantifierEngine,
            )
            _UncertaintyQuantifierEngine = UncertaintyQuantifierEngine
        except ImportError:
            pass

    if _ComplianceCheckerEngine is None:
        try:
            from greenlang.agents.mrv.scope2_market.compliance_checker import (
                ComplianceCheckerEngine,
            )
            _ComplianceCheckerEngine = ComplianceCheckerEngine
        except ImportError:
            pass

    if _Scope2MarketProvenance is None:
        try:
            from greenlang.agents.mrv.scope2_market.provenance import (
                Scope2MarketProvenance,
            )
            _Scope2MarketProvenance = Scope2MarketProvenance
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Built-in fallback data
# ---------------------------------------------------------------------------

GWP_TABLE: Dict[str, Dict[str, Decimal]] = {
    "AR4": {"co2": Decimal("1"), "ch4": Decimal("25"), "n2o": Decimal("298")},
    "AR5": {"co2": Decimal("1"), "ch4": Decimal("28"), "n2o": Decimal("265")},
    "AR6": {"co2": Decimal("1"), "ch4": Decimal("27.9"), "n2o": Decimal("273")},
    "AR6_20YR": {"co2": Decimal("1"), "ch4": Decimal("81.2"), "n2o": Decimal("273")},
}

# Global residual mix fallback when region-specific data is unavailable
# (tCO2e/MWh -- conservative world average)
DEFAULT_RESIDUAL_MIX: Decimal = Decimal("0.500")

# Renewable energy instruments produce zero emissions
RENEWABLE_EF: Decimal = Decimal("0.000")

# GHG Protocol Scope 2 Guidance instrument quality hierarchy (highest first)
INSTRUMENT_HIERARCHY: List[str] = [
    "energy_attribute_certificate",
    "direct_contract",
    "supplier_specific",
    "green_tariff",
    "default_delivered",
    "residual_mix",
]

VALID_INSTRUMENT_TYPES: set = {
    "eac", "rec", "go", "i_rec", "rego", "lgc", "t_rec",
    "ppa", "vppa", "green_tariff", "direct_line", "self_generated",
    "bundled", "unbundled", "supplier_specific",
}

RENEWABLE_INSTRUMENT_TYPES: set = {
    "rec", "go", "i_rec", "rego", "lgc", "t_rec",
    "ppa", "vppa", "self_generated",
}

VALID_GWP_SOURCES: set = {"AR4", "AR5", "AR6", "AR6_20YR"}

PIPELINE_STAGES: List[str] = [
    "validate_input",
    "resolve_instruments",
    "allocate_instruments",
    "calculate_covered",
    "calculate_uncovered",
    "apply_gwp_conversion",
    "compliance_checks",
    "assemble_results",
]

# Default residual mix factors by region (tCO2e/MWh)
# These are conservative approximations; production deployments should use
# the InstrumentDatabaseEngine for authoritative region-specific values.
DEFAULT_RESIDUAL_MIX_BY_REGION: Dict[str, Decimal] = {
    "US": Decimal("0.425"),
    "US-WECC": Decimal("0.351"),
    "US-RFCE": Decimal("0.406"),
    "US-SRMV": Decimal("0.477"),
    "US-CAMX": Decimal("0.302"),
    "US-ERCT": Decimal("0.403"),
    "US-NYUP": Decimal("0.213"),
    "EU": Decimal("0.327"),
    "EU-DE": Decimal("0.450"),
    "EU-FR": Decimal("0.060"),
    "EU-PL": Decimal("0.750"),
    "EU-ES": Decimal("0.240"),
    "EU-IT": Decimal("0.350"),
    "EU-NL": Decimal("0.480"),
    "EU-SE": Decimal("0.040"),
    "UK": Decimal("0.233"),
    "JP": Decimal("0.470"),
    "AU": Decimal("0.680"),
    "IN": Decimal("0.820"),
    "CN": Decimal("0.610"),
    "BR": Decimal("0.074"),
    "CA": Decimal("0.120"),
    "GLOBAL": DEFAULT_RESIDUAL_MIX,
}

_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")
_QUANT_6 = Decimal("0.000001")
_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# Pipeline Engine
# ---------------------------------------------------------------------------


class Scope2MarketPipelineEngine:
    """Engine 7: 8-stage orchestrated Scope 2 market-based calculation pipeline.

    Ties together all upstream engines (instrument database, allocation,
    covered emissions, residual mix, uncertainty, compliance) to provide
    end-to-end market-based Scope 2 emission calculations with full
    provenance tracking.

    The GHG Protocol market-based method uses contractual instruments
    (RECs, GOs, PPAs, supplier-specific EFs) to determine the emission
    factor applied to electricity consumption.  Consumption not covered
    by any instrument uses the grid residual mix factor.

    Thread Safety:
        All mutable state is protected by a threading.Lock.  Concurrent
        calls to run_pipeline are safe; however, each call creates its
        own provenance chain, so provenance is per-invocation.

    Attributes:
        _instrument_db: Engine for instrument EF lookups.
        _allocation: Engine for GHG Protocol hierarchy allocation.
        _covered: Engine for covered emissions calculations.
        _residual_mix: Engine for residual mix factor resolution.
        _uncertainty: Engine for Monte Carlo uncertainty analysis.
        _compliance: Engine for regulatory compliance checks.
    """

    def __init__(
        self,
        instrument_db: Any = None,
        allocation_engine: Any = None,
        covered_engine: Any = None,
        residual_mix_engine: Any = None,
        uncertainty_engine: Any = None,
        compliance_engine: Any = None,
        config: Any = None,
        metrics: Any = None,
    ) -> None:
        """Initialize the Scope2MarketPipelineEngine.

        Args:
            instrument_db: InstrumentDatabaseEngine instance or None.
            allocation_engine: InstrumentAllocationEngine instance or None.
            covered_engine: CoveredEmissionsEngine instance or None.
            residual_mix_engine: ResidualMixEngine instance or None.
            uncertainty_engine: UncertaintyQuantifierEngine instance or None.
            compliance_engine: ComplianceCheckerEngine instance or None.
            config: Application configuration object.
            metrics: Scope2MarketMetrics instance for telemetry.
        """
        _lazy_import_engines()
        self._config = config
        self._metrics = metrics
        self._lock = threading.Lock()
        self._pipeline_runs: int = 0
        self._total_co2e_processed: Decimal = _ZERO
        self._total_covered_mwh: Decimal = _ZERO
        self._total_uncovered_mwh: Decimal = _ZERO
        self._instruments_retired: int = 0

        # Initialize upstream engines (use provided or create defaults)
        self._instrument_db = self._init_engine(
            instrument_db, _InstrumentDatabaseEngine, config, metrics
        )
        self._allocation = self._init_engine(
            allocation_engine, _InstrumentAllocationEngine, config, metrics
        )
        self._covered = self._init_engine(
            covered_engine, _CoveredEmissionsEngine, config, metrics
        )
        self._residual_mix = self._init_engine(
            residual_mix_engine, _ResidualMixEngine, config, metrics
        )
        self._uncertainty = self._init_engine(
            uncertainty_engine, _UncertaintyQuantifierEngine, config, metrics
        )
        self._compliance = self._init_engine(
            compliance_engine, _ComplianceCheckerEngine, config, metrics
        )

        engine_count = sum(
            1 for e in [
                self._instrument_db, self._allocation, self._covered,
                self._residual_mix, self._uncertainty, self._compliance,
            ] if e is not None
        )
        logger.info(
            "Scope2MarketPipelineEngine initialized with %d engines",
            engine_count,
        )

    @staticmethod
    def _init_engine(
        provided: Any,
        engine_cls: Any,
        config: Any,
        metrics: Any,
    ) -> Any:
        """Initialize an engine: use provided, create from class, or None.

        Args:
            provided: Engine instance supplied by caller.
            engine_cls: Lazily-imported engine class (may be None).
            config: Configuration object.
            metrics: Metrics instance.

        Returns:
            Engine instance or None.
        """
        if provided is not None:
            return provided
        if engine_cls is not None:
            try:
                return engine_cls(config, metrics)
            except Exception as exc:
                logger.warning(
                    "Failed to create %s: %s",
                    engine_cls.__name__ if engine_cls else "engine",
                    exc,
                )
        return None

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full 8-stage Scope 2 market-based calculation pipeline.

        Args:
            request: Calculation request dict with keys:
                - calculation_id (optional, auto-generated)
                - tenant_id: str
                - facility_id: str
                - region: str (grid region for residual mix lookup)
                - gwp_source: 'AR4'|'AR5'|'AR6'|'AR6_20YR' (default 'AR5')
                - purchases: list of energy purchase dicts, each with:
                    - purchase_id: str
                    - mwh: Decimal (energy quantity)
                    - energy_type: str (default 'electricity')
                - instruments: list of instrument dicts, each with:
                    - instrument_id: str
                    - instrument_type: str (rec, go, ppa, ...)
                    - mwh: Decimal (quantity covered)
                    - emission_factor: Decimal (tCO2e/MWh, optional)
                    - vintage_year: int (optional)
                    - supplier_id: str (optional)
                    - is_renewable: bool (optional)
                    - priority: int (optional, for custom ordering)
                    - quality_tier: str (optional)
                - include_compliance: bool (default False)
                - compliance_frameworks: list of str (optional)

        Returns:
            MarketBasedResult dict with full results and provenance.

        Raises:
            ValueError: If input validation fails (Stage 1).
        """
        start = time.monotonic()

        prov = None
        if _Scope2MarketProvenance:
            prov = _Scope2MarketProvenance()

        calc_id = request.get("calculation_id", str(uuid.uuid4()))
        tenant_id = request.get("tenant_id", "")
        facility_id = request.get("facility_id", "")
        region = request.get("region", "GLOBAL")
        gwp_source = request.get("gwp_source", "AR5")
        purchases = request.get("purchases", [])
        instruments = request.get("instruments", [])
        include_compliance = request.get("include_compliance", False)
        compliance_frameworks = request.get("compliance_frameworks")

        stages_data: Dict[str, Any] = {}
        calculation_trace: List[Dict[str, Any]] = []

        # ---- Stage 1: Validate Input ----
        stage_start = time.monotonic()
        validation = self._stage_validate_input(
            facility_id, tenant_id, purchases, instruments, region, gwp_source
        )
        if not validation.get("valid", False):
            if self._metrics:
                try:
                    self._metrics.record_error("validation_error")
                except Exception:
                    pass
            raise ValueError(
                f"Input validation failed: {validation.get('errors', [])}"
            )
        stages_data["validation"] = validation
        calculation_trace.append(
            self._trace_entry("validate_input", time.monotonic() - stage_start)
        )
        if prov:
            prov.hash_input(request)

        # ---- Stage 2: Resolve Instrument Data ----
        stage_start = time.monotonic()
        resolved_instruments = self._stage_resolve_instruments(
            instruments, region
        )
        stages_data["resolved_instruments"] = resolved_instruments
        calculation_trace.append(
            self._trace_entry("resolve_instruments", time.monotonic() - stage_start)
        )
        if prov:
            prov.hash_custom(
                "resolve_instruments",
                {"count": len(resolved_instruments), "region": region},
            )

        # ---- Stage 3: Allocate Instruments ----
        stage_start = time.monotonic()
        total_purchase_mwh = self._sum_purchase_mwh(purchases)
        allocation = self._stage_allocate_instruments(
            total_purchase_mwh, resolved_instruments
        )
        stages_data["allocation"] = allocation
        calculation_trace.append(
            self._trace_entry("allocate_instruments", time.monotonic() - stage_start)
        )
        if prov:
            prov.hash_custom(
                "allocate_instruments",
                {
                    "total_mwh": str(total_purchase_mwh),
                    "covered_mwh": str(allocation["covered_mwh"]),
                    "uncovered_mwh": str(allocation["uncovered_mwh"]),
                    "instruments_allocated": len(allocation["allocations"]),
                },
            )

        # ---- Stage 4: Calculate Covered Emissions ----
        stage_start = time.monotonic()
        covered_result = self._stage_calculate_covered(
            allocation["allocations"], gwp_source
        )
        stages_data["covered_emissions"] = covered_result
        calculation_trace.append(
            self._trace_entry("calculate_covered", time.monotonic() - stage_start)
        )
        if prov:
            prov.hash_custom(
                "covered_emissions",
                {
                    "covered_co2e_kg": str(covered_result["total_covered_co2e_kg"]),
                    "instrument_count": len(covered_result["details"]),
                },
            )

        # ---- Stage 5: Calculate Uncovered Emissions ----
        stage_start = time.monotonic()
        uncovered_result = self._stage_calculate_uncovered(
            allocation["uncovered_mwh"], region, gwp_source
        )
        stages_data["uncovered_emissions"] = uncovered_result
        calculation_trace.append(
            self._trace_entry("calculate_uncovered", time.monotonic() - stage_start)
        )
        if prov:
            prov.hash_custom(
                "uncovered_emissions",
                {
                    "uncovered_mwh": str(allocation["uncovered_mwh"]),
                    "residual_mix_factor": str(
                        uncovered_result["residual_mix_factor"]
                    ),
                    "uncovered_co2e_kg": str(
                        uncovered_result["total_uncovered_co2e_kg"]
                    ),
                },
            )

        # ---- Stage 6: Apply GWP Conversion ----
        stage_start = time.monotonic()
        gwp_result = self._stage_apply_gwp(
            covered_result, uncovered_result, gwp_source
        )
        stages_data["gwp_conversion"] = gwp_result
        calculation_trace.append(
            self._trace_entry("apply_gwp_conversion", time.monotonic() - stage_start)
        )
        if prov:
            prov.hash_custom(
                "gwp_conversion",
                {
                    "gwp_source": gwp_source,
                    "total_co2e_kg": str(gwp_result["total_co2e_kg"]),
                    "total_co2e_tonnes": str(gwp_result["total_co2e_tonnes"]),
                },
            )

        # ---- Stage 7: Compliance Checks (optional) ----
        compliance_results: List[Dict[str, Any]] = []
        if include_compliance:
            stage_start = time.monotonic()
            compliance_results = self._stage_compliance_check(
                {
                    "calculation_id": calc_id,
                    "facility_id": facility_id,
                    "region": region,
                    "gwp_source": gwp_source,
                    "total_purchase_mwh": total_purchase_mwh,
                    "covered_mwh": allocation["covered_mwh"],
                    "uncovered_mwh": allocation["uncovered_mwh"],
                    "coverage_pct": allocation["coverage_pct"],
                    "total_co2e_kg": gwp_result["total_co2e_kg"],
                    "total_co2e_tonnes": gwp_result["total_co2e_tonnes"],
                    "gas_breakdown": gwp_result.get("gas_breakdown", []),
                    "instruments_used": len(allocation["allocations"]),
                },
                compliance_frameworks,
            )
            stages_data["compliance"] = compliance_results
            calculation_trace.append(
                self._trace_entry("compliance_checks", time.monotonic() - stage_start)
            )
            if prov:
                for cr in compliance_results:
                    prov.hash_custom(
                        "compliance_check",
                        {
                            "framework": cr.get("framework", ""),
                            "status": cr.get("status", "not_assessed"),
                            "findings_count": len(cr.get("findings", [])),
                        },
                    )

        # ---- Stage 8: Assemble Results ----
        stage_start = time.monotonic()
        provenance_hash = prov.get_chain_hash() if prov else self._compute_fallback_hash(
            calc_id, stages_data
        )

        total_co2e_kg = gwp_result["total_co2e_kg"]
        total_co2e_tonnes = gwp_result["total_co2e_tonnes"]

        result = {
            "calculation_id": calc_id,
            "tenant_id": tenant_id,
            "facility_id": facility_id,
            "method": "market_based",
            "region": region,
            "gwp_source": gwp_source,
            "total_purchase_mwh": total_purchase_mwh,
            "covered_mwh": allocation["covered_mwh"],
            "uncovered_mwh": allocation["uncovered_mwh"],
            "coverage_pct": allocation["coverage_pct"],
            "covered_co2e_kg": covered_result["total_covered_co2e_kg"],
            "uncovered_co2e_kg": uncovered_result["total_uncovered_co2e_kg"],
            "residual_mix_factor": uncovered_result["residual_mix_factor"],
            "residual_mix_source": uncovered_result.get("source", "fallback"),
            "instrument_allocations": allocation["allocations"],
            "covered_details": covered_result["details"],
            "gas_breakdown": gwp_result.get("gas_breakdown", []),
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "provenance_hash": provenance_hash,
            "compliance_results": compliance_results,
            "calculated_at": datetime.utcnow().isoformat(),
            "calculation_trace": calculation_trace,
            "metadata": {
                "pipeline_version": "1.0.0",
                "stages_completed": len(stages_data),
                "validation_warnings": validation.get("warnings", []),
                "instruments_retired": len(allocation["allocations"]),
            },
        }

        calculation_trace.append(
            self._trace_entry("assemble_results", time.monotonic() - stage_start)
        )

        # Record metrics
        duration = time.monotonic() - start
        with self._lock:
            self._pipeline_runs += 1
            self._total_co2e_processed += total_co2e_tonnes
            self._total_covered_mwh += allocation["covered_mwh"]
            self._total_uncovered_mwh += allocation["uncovered_mwh"]
            self._instruments_retired += len(allocation["allocations"])

        if self._metrics:
            try:
                primary_type = self._primary_instrument_type(
                    allocation["allocations"]
                )
                self._metrics.record_calculation(
                    primary_type,
                    "market_based",
                    duration,
                    float(total_co2e_tonnes),
                )
                self._metrics.set_coverage_percentage(
                    facility_id,
                    float(allocation["coverage_pct"]),
                )
            except Exception:
                pass

        logger.info(
            "Pipeline completed: calc=%s facility=%s co2e=%.6f tonnes "
            "coverage=%.1f%% (%.3fs)",
            calc_id, facility_id, total_co2e_tonnes,
            allocation["coverage_pct"], duration,
        )
        return result

    # ------------------------------------------------------------------
    # Batch Pipeline
    # ------------------------------------------------------------------

    def run_batch_pipeline(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run pipeline for a batch of calculation requests.

        Args:
            batch: Dict with keys:
                - batch_id: str (optional, auto-generated)
                - tenant_id: str
                - requests: list of individual request dicts

        Returns:
            BatchCalculationResult dict with individual and aggregate results.
        """
        batch_id = batch.get("batch_id", str(uuid.uuid4()))
        requests = batch.get("requests", [])
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = _ZERO

        for i, req in enumerate(requests):
            try:
                result = self.run_pipeline(req)
                results.append(result)
                total_co2e += result.get("total_co2e_tonnes", _ZERO)
            except Exception as exc:
                errors.append({
                    "index": i,
                    "error": str(exc),
                    "calculation_id": req.get("calculation_id", ""),
                    "facility_id": req.get("facility_id", ""),
                })
                logger.error("Batch item %d failed: %s", i, exc)

        total_covered = sum(
            r.get("covered_mwh", _ZERO) for r in results
        )
        total_uncovered = sum(
            r.get("uncovered_mwh", _ZERO) for r in results
        )

        return {
            "batch_id": batch_id,
            "total_requests": len(requests),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "total_co2e_tonnes": total_co2e.quantize(_QUANT_6, ROUND_HALF_UP),
            "total_covered_mwh": total_covered.quantize(_QUANT_6, ROUND_HALF_UP),
            "total_uncovered_mwh": total_uncovered.quantize(
                _QUANT_6, ROUND_HALF_UP
            ),
            "facility_count": len(
                set(r.get("facility_id", "") for r in results)
            ),
            "provenance_hash": self._compute_batch_hash(results),
        }

    # ------------------------------------------------------------------
    # Facility Pipeline
    # ------------------------------------------------------------------

    def run_facility_pipeline(
        self,
        facility_id: str,
        purchases: List[Dict[str, Any]],
        instruments: List[Dict[str, Any]],
        region: str = "GLOBAL",
        gwp_source: str = "AR5",
        include_compliance: bool = False,
        compliance_frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calculate market-based Scope 2 for a single facility.

        Convenience method that packages purchases and instruments into a
        single pipeline request.

        Args:
            facility_id: Facility identifier.
            purchases: List of energy purchase dicts.
            instruments: List of instrument dicts.
            region: Grid region for residual mix lookup.
            gwp_source: GWP assessment report.
            include_compliance: Whether to run compliance checks.
            compliance_frameworks: List of framework names.

        Returns:
            MarketBasedResult dict.
        """
        req = {
            "facility_id": facility_id,
            "purchases": purchases,
            "instruments": instruments,
            "region": region,
            "gwp_source": gwp_source,
            "include_compliance": include_compliance,
            "compliance_frameworks": compliance_frameworks,
        }
        return self.run_pipeline(req)

    # ------------------------------------------------------------------
    # Total Scope 2 Market Calculation
    # ------------------------------------------------------------------

    def calculate_total_scope2_market(
        self,
        purchases_with_instruments: List[Dict[str, Any]],
        region: str = "GLOBAL",
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """Calculate total market-based Scope 2 across multiple facilities.

        Args:
            purchases_with_instruments: List of dicts, each containing:
                - facility_id: str
                - purchases: list of purchase dicts
                - instruments: list of instrument dicts
                - region: str (optional, defaults to param)
            region: Default region if not specified per-facility.
            gwp_source: GWP assessment report.

        Returns:
            Aggregated market-based Scope 2 results.
        """
        facility_results: List[Dict[str, Any]] = []
        grand_total_co2e = _ZERO
        grand_covered_mwh = _ZERO
        grand_uncovered_mwh = _ZERO
        grand_total_mwh = _ZERO

        for entry in purchases_with_instruments:
            fid = entry.get("facility_id", str(uuid.uuid4()))
            fac_region = entry.get("region", region)
            fac_purchases = entry.get("purchases", [])
            fac_instruments = entry.get("instruments", [])

            result = self.run_facility_pipeline(
                facility_id=fid,
                purchases=fac_purchases,
                instruments=fac_instruments,
                region=fac_region,
                gwp_source=gwp_source,
            )
            facility_results.append(result)
            grand_total_co2e += result.get("total_co2e_tonnes", _ZERO)
            grand_covered_mwh += result.get("covered_mwh", _ZERO)
            grand_uncovered_mwh += result.get("uncovered_mwh", _ZERO)
            grand_total_mwh += result.get("total_purchase_mwh", _ZERO)

        weighted_coverage = _ZERO
        if grand_total_mwh > _ZERO:
            weighted_coverage = (
                (grand_covered_mwh / grand_total_mwh) * _HUNDRED
            ).quantize(_QUANT_6, ROUND_HALF_UP)

        return {
            "facility_count": len(facility_results),
            "facility_results": facility_results,
            "grand_total_co2e_tonnes": grand_total_co2e.quantize(
                _QUANT_6, ROUND_HALF_UP
            ),
            "grand_total_mwh": grand_total_mwh.quantize(_QUANT_6, ROUND_HALF_UP),
            "grand_covered_mwh": grand_covered_mwh.quantize(
                _QUANT_6, ROUND_HALF_UP
            ),
            "grand_uncovered_mwh": grand_uncovered_mwh.quantize(
                _QUANT_6, ROUND_HALF_UP
            ),
            "weighted_coverage_pct": weighted_coverage,
            "gwp_source": gwp_source,
        }

    # ------------------------------------------------------------------
    # Uncertainty Integration
    # ------------------------------------------------------------------

    def run_with_uncertainty(
        self,
        request: Dict[str, Any],
        mc_iterations: int = 5000,
    ) -> Dict[str, Any]:
        """Run pipeline with Monte Carlo uncertainty analysis.

        Args:
            request: Calculation request.
            mc_iterations: Number of Monte Carlo iterations.

        Returns:
            Dict with 'result' and 'uncertainty' keys.
        """
        result = self.run_pipeline(request)

        uncertainty = None
        if self._uncertainty:
            try:
                uncertainty = self._uncertainty.run_monte_carlo(
                    base_emissions_kg=result.get("total_co2e_kg", _ZERO),
                    covered_emissions_kg=result.get("covered_co2e_kg", _ZERO),
                    uncovered_emissions_kg=result.get("uncovered_co2e_kg", _ZERO),
                    coverage_pct=result.get("coverage_pct", _ZERO),
                    iterations=mc_iterations,
                )
            except Exception as exc:
                logger.warning("Uncertainty analysis failed: %s", exc)
                uncertainty = self._fallback_uncertainty(
                    result.get("total_co2e_kg", _ZERO), mc_iterations
                )
        else:
            uncertainty = self._fallback_uncertainty(
                result.get("total_co2e_kg", _ZERO), mc_iterations
            )

        if self._metrics:
            try:
                self._metrics.record_uncertainty_run("monte_carlo")
            except Exception:
                pass

        return {
            "result": result,
            "uncertainty": uncertainty,
        }

    # ------------------------------------------------------------------
    # Dual Reporting
    # ------------------------------------------------------------------

    def run_with_dual_reporting(
        self,
        request: Dict[str, Any],
        location_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run market-based pipeline and combine with location-based result.

        GHG Protocol Scope 2 Guidance (2015) requires organizations to report
        both location-based and market-based Scope 2 emissions.  This method
        produces the combined dual report.

        Args:
            request: Market-based calculation request.
            location_result: Pre-computed location-based result dict, expected
                to contain at minimum 'total_co2e_tonnes' and 'total_co2e_kg'.

        Returns:
            Dict with market_result, location_result, and comparison metrics.
        """
        market_result = self.run_pipeline(request)

        location_co2e_tonnes = Decimal(
            str(location_result.get("total_co2e_tonnes", _ZERO))
        )
        market_co2e_tonnes = market_result.get("total_co2e_tonnes", _ZERO)

        difference_tonnes = (market_co2e_tonnes - location_co2e_tonnes).quantize(
            _QUANT_6, ROUND_HALF_UP
        )
        difference_pct = _ZERO
        if location_co2e_tonnes > _ZERO:
            difference_pct = (
                (difference_tonnes / location_co2e_tonnes) * _HUNDRED
            ).quantize(_QUANT_6, ROUND_HALF_UP)

        status = "complete"
        if market_co2e_tonnes == _ZERO and location_co2e_tonnes == _ZERO:
            status = "partial"

        if self._metrics:
            try:
                self._metrics.record_dual_report(status)
            except Exception:
                pass

        return {
            "dual_report": True,
            "market_result": market_result,
            "location_result": location_result,
            "comparison": {
                "location_co2e_tonnes": location_co2e_tonnes,
                "market_co2e_tonnes": market_co2e_tonnes,
                "difference_tonnes": difference_tonnes,
                "difference_pct": difference_pct,
                "market_lower": market_co2e_tonnes < location_co2e_tonnes,
                "coverage_pct": market_result.get("coverage_pct", _ZERO),
            },
            "status": status,
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Pipeline Control
    # ------------------------------------------------------------------

    def get_pipeline_stages(self) -> List[str]:
        """List the 8 pipeline stage names.

        Returns:
            Ordered list of stage name strings.
        """
        return list(PIPELINE_STAGES)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics.

        Returns:
            Dict with run counts, cumulative emissions, engine availability.
        """
        with self._lock:
            return {
                "pipeline_runs": self._pipeline_runs,
                "total_co2e_processed_tonnes": self._total_co2e_processed,
                "total_covered_mwh": self._total_covered_mwh,
                "total_uncovered_mwh": self._total_uncovered_mwh,
                "instruments_retired": self._instruments_retired,
                "engines_available": sum(
                    1 for e in [
                        self._instrument_db, self._allocation,
                        self._covered, self._residual_mix,
                        self._uncertainty, self._compliance,
                    ] if e is not None
                ),
                "stages_count": len(PIPELINE_STAGES),
                "engines": {
                    "instrument_db": self._instrument_db is not None,
                    "allocation": self._allocation is not None,
                    "covered": self._covered is not None,
                    "residual_mix": self._residual_mix is not None,
                    "uncertainty": self._uncertainty is not None,
                    "compliance": self._compliance is not None,
                },
            }

    def reset(self) -> None:
        """Reset pipeline counters to zero."""
        with self._lock:
            self._pipeline_runs = 0
            self._total_co2e_processed = _ZERO
            self._total_covered_mwh = _ZERO
            self._total_uncovered_mwh = _ZERO
            self._instruments_retired = 0
        logger.info("Scope2MarketPipelineEngine counters reset")

    # ==================================================================
    # Stage Implementations
    # ==================================================================

    def _stage_validate_input(
        self,
        facility_id: str,
        tenant_id: str,
        purchases: List[Dict[str, Any]],
        instruments: List[Dict[str, Any]],
        region: str,
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 1: Validate all input fields.

        Args:
            facility_id: Facility identifier.
            tenant_id: Tenant identifier.
            purchases: Energy purchase list.
            instruments: Instrument list.
            region: Grid region.
            gwp_source: GWP assessment report source.

        Returns:
            Dict with valid (bool), errors (list), warnings (list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Required identifiers
        if not facility_id:
            errors.append("facility_id is required")
        if not tenant_id:
            warnings.append(
                "tenant_id not specified; defaulting to empty string"
            )

        # Purchases validation
        if not purchases:
            errors.append("At least one energy purchase is required")
        else:
            for idx, p in enumerate(purchases):
                p_errors = self._validate_purchase(p, idx)
                errors.extend(p_errors)

        # Instruments validation
        if not instruments:
            warnings.append(
                "No instruments provided; all consumption will use "
                "residual mix factor"
            )
        else:
            for idx, inst in enumerate(instruments):
                i_errors, i_warnings = self._validate_instrument(inst, idx)
                errors.extend(i_errors)
                warnings.extend(i_warnings)

        # Region validation
        if not region:
            warnings.append(
                "region not specified; defaulting to 'GLOBAL'"
            )

        # GWP source validation
        if gwp_source not in VALID_GWP_SOURCES:
            errors.append(
                f"Invalid gwp_source '{gwp_source}'. "
                f"Must be one of: {sorted(VALID_GWP_SOURCES)}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _validate_purchase(
        self, purchase: Dict[str, Any], idx: int
    ) -> List[str]:
        """Validate a single energy purchase entry.

        Args:
            purchase: Purchase dict.
            idx: Index in the purchases list.

        Returns:
            List of error strings.
        """
        errors: List[str] = []
        prefix = f"purchases[{idx}]"

        mwh = purchase.get("mwh")
        if mwh is None:
            errors.append(f"{prefix}.mwh is required")
        else:
            try:
                mwh_dec = Decimal(str(mwh))
                if mwh_dec < _ZERO:
                    errors.append(f"{prefix}.mwh cannot be negative")
            except (InvalidOperation, TypeError, ValueError):
                errors.append(f"{prefix}.mwh must be a valid number")

        return errors

    def _validate_instrument(
        self, instrument: Dict[str, Any], idx: int
    ) -> Tuple[List[str], List[str]]:
        """Validate a single instrument entry.

        Args:
            instrument: Instrument dict.
            idx: Index in the instruments list.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []
        prefix = f"instruments[{idx}]"

        # Instrument type
        itype = instrument.get("instrument_type", "")
        if not itype:
            errors.append(f"{prefix}.instrument_type is required")
        elif itype not in VALID_INSTRUMENT_TYPES:
            errors.append(
                f"{prefix}.instrument_type '{itype}' is invalid. "
                f"Must be one of: {sorted(VALID_INSTRUMENT_TYPES)}"
            )

        # MWh quantity
        mwh = instrument.get("mwh")
        if mwh is None:
            errors.append(f"{prefix}.mwh is required")
        else:
            try:
                mwh_dec = Decimal(str(mwh))
                if mwh_dec <= _ZERO:
                    errors.append(f"{prefix}.mwh must be positive")
            except (InvalidOperation, TypeError, ValueError):
                errors.append(f"{prefix}.mwh must be a valid number")

        # Vintage year
        vintage = instrument.get("vintage_year")
        if vintage is not None:
            current_year = datetime.utcnow().year
            if not isinstance(vintage, int) or vintage < 2000:
                errors.append(
                    f"{prefix}.vintage_year must be an integer >= 2000"
                )
            elif vintage > current_year:
                warnings.append(
                    f"{prefix}.vintage_year {vintage} is in the future"
                )
            elif vintage < current_year - 5:
                warnings.append(
                    f"{prefix}.vintage_year {vintage} is older than 5 years; "
                    "some frameworks may reject it"
                )

        # Emission factor
        ef = instrument.get("emission_factor")
        if ef is not None:
            try:
                ef_dec = Decimal(str(ef))
                if ef_dec < _ZERO:
                    errors.append(
                        f"{prefix}.emission_factor cannot be negative"
                    )
            except (InvalidOperation, TypeError, ValueError):
                errors.append(
                    f"{prefix}.emission_factor must be a valid number"
                )

        return errors, warnings

    # ------------------------------------------------------------------
    # Stage 2: Resolve Instrument Data
    # ------------------------------------------------------------------

    def _stage_resolve_instruments(
        self,
        instruments: List[Dict[str, Any]],
        region: str,
    ) -> List[Dict[str, Any]]:
        """Stage 2: Resolve instrument emission factors and quality data.

        For each instrument, look up or confirm:
        - Emission factor (from DB or provided value)
        - Quality tier
        - Vintage validity
        - Supplier-specific EFs if applicable

        Args:
            instruments: Raw instrument list from request.
            region: Grid region for context.

        Returns:
            List of resolved instrument dicts with confirmed EFs.
        """
        resolved: List[Dict[str, Any]] = []

        for inst in instruments:
            resolved_inst = self._resolve_single_instrument(inst, region)
            resolved.append(resolved_inst)

        # Sort by priority (GHG Protocol hierarchy: highest quality first)
        resolved.sort(key=lambda x: x.get("priority", 99))

        return resolved

    def _resolve_single_instrument(
        self,
        instrument: Dict[str, Any],
        region: str,
    ) -> Dict[str, Any]:
        """Resolve a single instrument's emission factor and metadata.

        Args:
            instrument: Raw instrument dict.
            region: Grid region.

        Returns:
            Resolved instrument dict.
        """
        itype = instrument.get("instrument_type", "")
        mwh = Decimal(str(instrument.get("mwh", _ZERO)))
        instrument_id = instrument.get("instrument_id", str(uuid.uuid4()))
        vintage = instrument.get("vintage_year")
        supplier_id = instrument.get("supplier_id")
        provided_ef = instrument.get("emission_factor")
        is_renewable = instrument.get("is_renewable")
        custom_priority = instrument.get("priority")
        quality_tier = instrument.get("quality_tier")

        # Determine if instrument is renewable.
        # If caller provides an explicit emission_factor > 0 without setting
        # is_renewable, treat the instrument as non-renewable (the EF implies
        # non-zero emissions, e.g. a PPA with a fossil-fuel supplier).
        if is_renewable is None:
            if provided_ef is not None and Decimal(str(provided_ef)) > _ZERO:
                is_renewable = False
            else:
                is_renewable = itype in RENEWABLE_INSTRUMENT_TYPES

        # Resolve emission factor
        ef = self._resolve_emission_factor(
            itype, provided_ef, supplier_id, is_renewable, region
        )

        # Determine priority (lower = higher priority)
        priority = self._determine_priority(itype, custom_priority)

        # Determine quality tier
        if quality_tier is None:
            quality_tier = self._determine_quality_tier(itype)

        # Check vintage validity
        vintage_valid = True
        if vintage is not None:
            current_year = datetime.utcnow().year
            vintage_valid = (current_year - vintage) <= 5

        return {
            "instrument_id": instrument_id,
            "instrument_type": itype,
            "mwh": mwh,
            "emission_factor": ef,
            "is_renewable": is_renewable,
            "vintage_year": vintage,
            "vintage_valid": vintage_valid,
            "supplier_id": supplier_id,
            "priority": priority,
            "quality_tier": quality_tier,
            "ef_source": "provided" if provided_ef is not None else (
                "database" if self._instrument_db else "fallback"
            ),
        }

    def _resolve_emission_factor(
        self,
        instrument_type: str,
        provided_ef: Any,
        supplier_id: Optional[str],
        is_renewable: bool,
        region: str,
    ) -> Decimal:
        """Resolve the emission factor for an instrument.

        Args:
            instrument_type: Type of instrument.
            provided_ef: Caller-supplied EF (may be None).
            supplier_id: Supplier ID for supplier-specific lookup.
            is_renewable: Whether instrument represents renewable energy.
            region: Grid region.

        Returns:
            Resolved emission factor in tCO2e/MWh.
        """
        # Use provided EF if available (caller override takes precedence)
        if provided_ef is not None:
            return Decimal(str(provided_ef))

        # If renewable and no explicit EF, EF is always zero
        if is_renewable:
            return RENEWABLE_EF

        # Try database lookup
        if self._instrument_db:
            try:
                if supplier_id:
                    db_ef = self._instrument_db.get_supplier_ef(
                        supplier_id, region
                    )
                else:
                    db_ef = self._instrument_db.get_instrument_ef(
                        instrument_type, region
                    )
                if db_ef is not None:
                    return Decimal(str(db_ef))
            except Exception as exc:
                logger.warning(
                    "Instrument DB lookup failed for %s: %s",
                    instrument_type, exc,
                )

        # Fallback: use residual mix as conservative estimate for
        # non-renewable instruments without specific EF
        return self._get_residual_mix_factor(region)

    def _determine_priority(
        self, instrument_type: str, custom_priority: Optional[int]
    ) -> int:
        """Determine allocation priority for an instrument type.

        Lower values mean higher priority.  Follows GHH Protocol
        Scope 2 Guidance hierarchy.

        Args:
            instrument_type: Instrument type string.
            custom_priority: Caller-supplied priority override.

        Returns:
            Integer priority (1 = highest).
        """
        if custom_priority is not None:
            return custom_priority

        # GHG Protocol quality hierarchy
        priority_map: Dict[str, int] = {
            # Tier 1: Energy attribute certificates (direct ownership)
            "self_generated": 1,
            "direct_line": 2,
            # Tier 2: Contractual instruments (direct purchase)
            "ppa": 3,
            "vppa": 4,
            "green_tariff": 5,
            # Tier 3: Certificates (unbundled)
            "rec": 6,
            "go": 7,
            "i_rec": 8,
            "rego": 9,
            "lgc": 10,
            "t_rec": 11,
            # Tier 4: Bundled/unbundled generic
            "bundled": 12,
            "unbundled": 13,
            # Tier 5: Supplier-specific
            "supplier_specific": 14,
            "eac": 15,
        }
        return priority_map.get(instrument_type, 50)

    @staticmethod
    def _determine_quality_tier(instrument_type: str) -> str:
        """Determine quality tier based on instrument type.

        Args:
            instrument_type: Instrument type string.

        Returns:
            Quality tier label.
        """
        tier_map: Dict[str, str] = {
            "self_generated": "tier_1",
            "direct_line": "tier_1",
            "ppa": "tier_2",
            "vppa": "tier_2",
            "green_tariff": "tier_2",
            "rec": "tier_3",
            "go": "tier_3",
            "i_rec": "tier_3",
            "rego": "tier_3",
            "lgc": "tier_3",
            "t_rec": "tier_3",
            "bundled": "tier_3",
            "unbundled": "tier_4",
            "supplier_specific": "tier_4",
            "eac": "tier_3",
        }
        return tier_map.get(instrument_type, "tier_5")

    # ------------------------------------------------------------------
    # Stage 3: Allocate Instruments
    # ------------------------------------------------------------------

    def _stage_allocate_instruments(
        self,
        total_mwh: Decimal,
        resolved_instruments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Stage 3: Allocate instruments to consumption using GHH Protocol hierarchy.

        Instruments are allocated in priority order (lowest priority number
        first).  Each instrument covers up to its stated MWh.  Any consumption
        remaining after all instruments are exhausted is uncovered and will
        use the residual mix factor.

        Args:
            total_mwh: Total energy purchased (MWh).
            resolved_instruments: Priority-sorted instrument list from Stage 2.

        Returns:
            Allocation result with covered_mwh, uncovered_mwh, allocations list.
        """
        if self._allocation:
            try:
                return self._allocation.allocate(total_mwh, resolved_instruments)
            except Exception as exc:
                logger.warning(
                    "AllocationEngine failed, using fallback: %s", exc
                )

        # Built-in fallback allocation
        remaining_mwh = total_mwh
        allocations: List[Dict[str, Any]] = []

        for inst in resolved_instruments:
            if remaining_mwh <= _ZERO:
                break

            inst_mwh = inst.get("mwh", _ZERO)
            allocated = min(inst_mwh, remaining_mwh)
            remaining_mwh -= allocated

            allocations.append({
                "instrument_id": inst["instrument_id"],
                "instrument_type": inst["instrument_type"],
                "requested_mwh": inst_mwh,
                "allocated_mwh": allocated,
                "emission_factor": inst["emission_factor"],
                "is_renewable": inst.get("is_renewable", False),
                "quality_tier": inst.get("quality_tier", ""),
                "vintage_year": inst.get("vintage_year"),
                "priority": inst.get("priority", 99),
                "status": "allocated",
            })

            if self._metrics:
                try:
                    self._metrics.record_instrument_retired(
                        inst["instrument_type"]
                    )
                except Exception:
                    pass

        covered_mwh = (total_mwh - remaining_mwh).quantize(
            _QUANT_6, ROUND_HALF_UP
        )
        uncovered_mwh = remaining_mwh.quantize(_QUANT_6, ROUND_HALF_UP)

        coverage_pct = _ZERO
        if total_mwh > _ZERO:
            coverage_pct = (
                (covered_mwh / total_mwh) * _HUNDRED
            ).quantize(_QUANT_6, ROUND_HALF_UP)

        return {
            "total_mwh": total_mwh,
            "covered_mwh": covered_mwh,
            "uncovered_mwh": uncovered_mwh,
            "coverage_pct": coverage_pct,
            "allocations": allocations,
            "instruments_used": len(allocations),
        }

    # ------------------------------------------------------------------
    # Stage 4: Calculate Covered Emissions
    # ------------------------------------------------------------------

    def _stage_calculate_covered(
        self,
        allocations: List[Dict[str, Any]],
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 4: Calculate emissions for instrument-covered consumption.

        For each allocated instrument:
        - Renewable instruments produce zero emissions
        - Non-renewable: allocated_mwh x instrument emission_factor
        - Supplier-specific uses supplier EF

        Args:
            allocations: List of allocation dicts from Stage 3.
            gwp_source: GWP source for consistency tracking.

        Returns:
            Dict with total_covered_co2e_kg, details list.
        """
        if self._covered:
            try:
                return self._covered.calculate(allocations, gwp_source)
            except Exception as exc:
                logger.warning(
                    "CoveredEmissionsEngine failed, using fallback: %s", exc
                )

        # Built-in fallback calculation
        details: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO

        for alloc in allocations:
            allocated_mwh = alloc.get("allocated_mwh", _ZERO)
            ef = alloc.get("emission_factor", _ZERO)
            is_renewable = alloc.get("is_renewable", False)

            if is_renewable:
                co2e_tonnes = _ZERO
            else:
                # EF is in tCO2e/MWh, result is in tCO2e
                co2e_tonnes = (allocated_mwh * ef).quantize(
                    _QUANT_6, ROUND_HALF_UP
                )

            co2e_kg = (co2e_tonnes * _THOUSAND).quantize(
                _QUANT_6, ROUND_HALF_UP
            )
            total_co2e_kg += co2e_kg

            details.append({
                "instrument_id": alloc.get("instrument_id", ""),
                "instrument_type": alloc.get("instrument_type", ""),
                "allocated_mwh": allocated_mwh,
                "emission_factor": ef,
                "is_renewable": is_renewable,
                "co2e_tonnes": co2e_tonnes,
                "co2e_kg": co2e_kg,
                "quality_tier": alloc.get("quality_tier", ""),
            })

        return {
            "total_covered_co2e_kg": total_co2e_kg.quantize(
                _QUANT_6, ROUND_HALF_UP
            ),
            "total_covered_co2e_tonnes": (total_co2e_kg / _THOUSAND).quantize(
                _QUANT_6, ROUND_HALF_UP
            ),
            "details": details,
            "instruments_calculated": len(details),
        }

    # ------------------------------------------------------------------
    # Stage 5: Calculate Uncovered Emissions
    # ------------------------------------------------------------------

    def _stage_calculate_uncovered(
        self,
        uncovered_mwh: Decimal,
        region: str,
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 5: Calculate emissions for uncovered consumption using residual mix.

        Args:
            uncovered_mwh: MWh not covered by any instrument.
            region: Grid region for residual mix lookup.
            gwp_source: GWP assessment report source.

        Returns:
            Dict with total_uncovered_co2e_kg, gas breakdown, residual mix details.
        """
        if uncovered_mwh <= _ZERO:
            return {
                "uncovered_mwh": _ZERO,
                "residual_mix_factor": _ZERO,
                "source": "n/a",
                "total_uncovered_co2e_kg": _ZERO,
                "total_uncovered_co2e_tonnes": _ZERO,
                "gas_breakdown": [],
            }

        # Resolve residual mix factor
        rm_factor, rm_source = self._resolve_residual_mix(region)

        if self._metrics:
            try:
                self._metrics.record_residual_mix_lookup(rm_source)
            except Exception:
                pass

        # Calculate total uncovered emissions (tCO2e)
        uncovered_co2e_tonnes = (uncovered_mwh * rm_factor).quantize(
            _QUANT_6, ROUND_HALF_UP
        )
        uncovered_co2e_kg = (uncovered_co2e_tonnes * _THOUSAND).quantize(
            _QUANT_6, ROUND_HALF_UP
        )

        # Approximate per-gas breakdown for uncovered portion
        # Typical grid mix ratios: CO2 ~97%, CH4 ~2%, N2O ~1%
        gas_breakdown = self._approximate_gas_breakdown(
            uncovered_co2e_kg, gwp_source
        )

        return {
            "uncovered_mwh": uncovered_mwh,
            "residual_mix_factor": rm_factor,
            "source": rm_source,
            "total_uncovered_co2e_kg": uncovered_co2e_kg,
            "total_uncovered_co2e_tonnes": uncovered_co2e_tonnes,
            "gas_breakdown": gas_breakdown,
        }

    def _resolve_residual_mix(
        self, region: str
    ) -> Tuple[Decimal, str]:
        """Resolve the residual mix emission factor for a region.

        Args:
            region: Grid region identifier.

        Returns:
            Tuple of (factor_tCO2e_per_MWh, source_label).
        """
        # Try database engine first
        if self._residual_mix:
            try:
                result = self._residual_mix.get_residual_mix_factor(region)
                if result is not None:
                    factor = Decimal(str(result.get("factor", DEFAULT_RESIDUAL_MIX)))
                    source = result.get("source", "database")
                    return factor, source
            except Exception as exc:
                logger.warning(
                    "ResidualMixEngine lookup failed for %s: %s", region, exc
                )

        # Fallback to built-in table
        factor = DEFAULT_RESIDUAL_MIX_BY_REGION.get(
            region, DEFAULT_RESIDUAL_MIX
        )
        return factor, "fallback"

    def _approximate_gas_breakdown(
        self,
        total_co2e_kg: Decimal,
        gwp_source: str,
    ) -> List[Dict[str, Any]]:
        """Approximate per-gas breakdown for grid electricity.

        Uses typical grid mix ratios: CO2 ~97%, CH4 ~2%, N2O ~1% of
        the total CO2e, then back-calculates mass emissions using GWP.

        Args:
            total_co2e_kg: Total CO2e in kg.
            gwp_source: GWP assessment report.

        Returns:
            List of gas breakdown dicts.
        """
        gwp = GWP_TABLE.get(gwp_source, GWP_TABLE["AR5"])

        # CO2e contributions by gas (approximate grid average ratios)
        co2_co2e_kg = (total_co2e_kg * Decimal("0.97")).quantize(
            _QUANT_6, ROUND_HALF_UP
        )
        ch4_co2e_kg = (total_co2e_kg * Decimal("0.02")).quantize(
            _QUANT_6, ROUND_HALF_UP
        )
        n2o_co2e_kg = (total_co2e_kg * Decimal("0.01")).quantize(
            _QUANT_6, ROUND_HALF_UP
        )

        # Back-calculate mass emissions
        co2_kg = co2_co2e_kg  # GWP of CO2 is 1
        ch4_kg = _ZERO
        if gwp["ch4"] > _ZERO:
            ch4_kg = (ch4_co2e_kg / gwp["ch4"]).quantize(
                _QUANT_6, ROUND_HALF_UP
            )
        n2o_kg = _ZERO
        if gwp["n2o"] > _ZERO:
            n2o_kg = (n2o_co2e_kg / gwp["n2o"]).quantize(
                _QUANT_6, ROUND_HALF_UP
            )

        return [
            {
                "gas": "co2",
                "emission_kg": co2_kg,
                "gwp_factor": gwp["co2"],
                "co2e_kg": co2_co2e_kg,
            },
            {
                "gas": "ch4",
                "emission_kg": ch4_kg,
                "gwp_factor": gwp["ch4"],
                "co2e_kg": ch4_co2e_kg,
            },
            {
                "gas": "n2o",
                "emission_kg": n2o_kg,
                "gwp_factor": gwp["n2o"],
                "co2e_kg": n2o_co2e_kg,
            },
        ]

    # ------------------------------------------------------------------
    # Stage 6: Apply GWP Conversion
    # ------------------------------------------------------------------

    def _stage_apply_gwp(
        self,
        covered_result: Dict[str, Any],
        uncovered_result: Dict[str, Any],
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 6: Combine covered and uncovered emissions with GWP conversion.

        Covered emissions are already in CO2e (instrument EFs are CO2e-based).
        Uncovered emissions have an approximate gas breakdown.  This stage
        assembles the total and ensures GWP consistency.

        Args:
            covered_result: Stage 4 output.
            uncovered_result: Stage 5 output.
            gwp_source: GWP assessment report used.

        Returns:
            Dict with total CO2e and combined gas breakdown.
        """
        covered_co2e_kg = covered_result.get(
            "total_covered_co2e_kg", _ZERO
        )
        uncovered_co2e_kg = uncovered_result.get(
            "total_uncovered_co2e_kg", _ZERO
        )
        total_co2e_kg = (covered_co2e_kg + uncovered_co2e_kg).quantize(
            _QUANT_6, ROUND_HALF_UP
        )
        total_co2e_tonnes = (total_co2e_kg / _THOUSAND).quantize(
            _QUANT_6, ROUND_HALF_UP
        )

        # Merge gas breakdowns
        # Covered emissions are aggregated as CO2e (no per-gas split from
        # instruments); uncovered has approximate per-gas breakdown.
        gas_breakdown = uncovered_result.get("gas_breakdown", [])

        # Add covered emissions as a CO2e-only entry if non-zero
        if covered_co2e_kg > _ZERO:
            gas_breakdown = list(gas_breakdown)  # copy
            gas_breakdown.append({
                "gas": "co2e_covered",
                "emission_kg": covered_co2e_kg,
                "gwp_factor": _ONE,
                "co2e_kg": covered_co2e_kg,
                "note": "Covered emissions from contractual instruments",
            })

        return {
            "gwp_source": gwp_source,
            "covered_co2e_kg": covered_co2e_kg,
            "uncovered_co2e_kg": uncovered_co2e_kg,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "gas_breakdown": gas_breakdown,
        }

    # ------------------------------------------------------------------
    # Stage 7: Compliance Checks
    # ------------------------------------------------------------------

    def _stage_compliance_check(
        self,
        result_data: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Stage 7: Run regulatory compliance checks.

        Args:
            result_data: Calculation data to check.
            frameworks: List of framework names (optional).

        Returns:
            List of compliance check result dicts.
        """
        if self._compliance:
            try:
                checks = self._compliance.check_compliance(
                    result_data, frameworks
                )
                if self._metrics:
                    for c in checks:
                        try:
                            self._metrics.record_compliance_check(
                                c.get("framework", "UNKNOWN"),
                                c.get("status", "not_assessed"),
                            )
                        except Exception:
                            pass
                return checks
            except Exception as exc:
                logger.warning("Compliance check failed: %s", exc)

        # Fallback: basic GHH Protocol checks
        return self._fallback_compliance_checks(result_data, frameworks)

    def _fallback_compliance_checks(
        self,
        result_data: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run basic built-in compliance checks when engine is unavailable.

        Args:
            result_data: Calculation data.
            frameworks: Requested frameworks.

        Returns:
            List of compliance check dicts.
        """
        checks: List[Dict[str, Any]] = []
        requested = set(frameworks) if frameworks else {"GHG_PROTOCOL"}

        if "GHG_PROTOCOL" in requested:
            findings: List[str] = []
            status = "compliant"

            coverage = result_data.get("coverage_pct", _ZERO)
            if coverage < _HUNDRED:
                findings.append(
                    f"Coverage is {coverage}%; uncovered consumption "
                    f"uses residual mix factor"
                )
            instruments_used = result_data.get("instruments_used", 0)
            if instruments_used == 0:
                findings.append(
                    "No contractual instruments applied; "
                    "all emissions use residual mix"
                )
                status = "partial"

            checks.append({
                "framework": "GHG_PROTOCOL",
                "status": status,
                "findings": findings,
                "checked_at": datetime.utcnow().isoformat(),
            })

        if "RE100" in requested:
            coverage = result_data.get("coverage_pct", _ZERO)
            re100_status = "compliant" if coverage >= _HUNDRED else "non_compliant"
            checks.append({
                "framework": "RE100",
                "status": re100_status,
                "findings": [
                    f"RE100 requires 100% renewable coverage; "
                    f"current coverage is {coverage}%"
                ] if coverage < _HUNDRED else [],
                "checked_at": datetime.utcnow().isoformat(),
            })

        return checks

    # ==================================================================
    # Private Helpers
    # ==================================================================

    @staticmethod
    def _sum_purchase_mwh(purchases: List[Dict[str, Any]]) -> Decimal:
        """Sum total MWh from purchase list.

        Args:
            purchases: List of purchase dicts.

        Returns:
            Total MWh as Decimal.
        """
        total = _ZERO
        for p in purchases:
            try:
                total += Decimal(str(p.get("mwh", _ZERO)))
            except (InvalidOperation, TypeError, ValueError):
                continue
        return total.quantize(_QUANT_6, ROUND_HALF_UP)

    def _get_residual_mix_factor(self, region: str) -> Decimal:
        """Get residual mix factor for a region (fallback table).

        Args:
            region: Grid region identifier.

        Returns:
            Residual mix factor in tCO2e/MWh.
        """
        return DEFAULT_RESIDUAL_MIX_BY_REGION.get(region, DEFAULT_RESIDUAL_MIX)

    @staticmethod
    def _primary_instrument_type(
        allocations: List[Dict[str, Any]],
    ) -> str:
        """Determine the primary instrument type from allocations.

        Returns the instrument type that covers the most MWh.

        Args:
            allocations: List of allocation dicts.

        Returns:
            Primary instrument type string.
        """
        if not allocations:
            return "residual_mix"

        best = max(
            allocations,
            key=lambda a: a.get("allocated_mwh", _ZERO),
        )
        return best.get("instrument_type", "unknown")

    @staticmethod
    def _trace_entry(stage: str, duration_s: float) -> Dict[str, Any]:
        """Create a calculation trace entry.

        Args:
            stage: Stage name.
            duration_s: Stage duration in seconds.

        Returns:
            Trace entry dict.
        """
        return {
            "stage": stage,
            "duration_ms": round(duration_s * 1000, 3),
            "timestamp": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _compute_fallback_hash(
        calc_id: str,
        stages_data: Dict[str, Any],
    ) -> str:
        """Compute a fallback provenance hash when provenance engine is unavailable.

        Args:
            calc_id: Calculation identifier.
            stages_data: Dict of stage results.

        Returns:
            SHA-256 hex digest string.
        """
        payload = json.dumps(
            {"calculation_id": calc_id, "stages": str(stages_data)},
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
            results: List of individual pipeline results.

        Returns:
            SHA-256 hex digest string.
        """
        hashes = [
            r.get("provenance_hash", "") for r in results
        ]
        combined = "|".join(hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def _fallback_uncertainty(
        base_co2e_kg: Decimal,
        iterations: int,
    ) -> Dict[str, Any]:
        """Provide a conservative uncertainty estimate when engine is unavailable.

        Uses IPCC default uncertainty ranges for purchased electricity:
        - Emission factor uncertainty: +/- 10%
        - Activity data uncertainty: +/- 3%
        - Combined (root sum of squares): ~10.4%

        Args:
            base_co2e_kg: Base emissions in kg CO2e.
            iterations: Number of iterations requested (informational).

        Returns:
            Uncertainty result dict.
        """
        ef_pct = Decimal("0.10")
        ad_pct = Decimal("0.03")
        # Combined using error propagation (root sum of squares)
        combined_pct = (
            (ef_pct ** 2 + ad_pct ** 2).sqrt()
        ).quantize(_QUANT_6, ROUND_HALF_UP)

        lower = (base_co2e_kg * (_ONE - combined_pct * Decimal("1.96"))).quantize(
            _QUANT_6, ROUND_HALF_UP
        )
        upper = (base_co2e_kg * (_ONE + combined_pct * Decimal("1.96"))).quantize(
            _QUANT_6, ROUND_HALF_UP
        )

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
            "note": "Fallback estimate using IPCC default uncertainty ranges",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Scope2MarketPipelineEngine",
    "GWP_TABLE",
    "DEFAULT_RESIDUAL_MIX",
    "RENEWABLE_EF",
    "VALID_INSTRUMENT_TYPES",
    "RENEWABLE_INSTRUMENT_TYPES",
    "VALID_GWP_SOURCES",
    "PIPELINE_STAGES",
    "DEFAULT_RESIDUAL_MIX_BY_REGION",
    "INSTRUMENT_HIERARCHY",
]
