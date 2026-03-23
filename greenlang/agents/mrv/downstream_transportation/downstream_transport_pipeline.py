# -*- coding: utf-8 -*-
"""
DownstreamTransportPipelineEngine - AGENT-MRV-022 Engine 7

This module implements the DownstreamTransportPipelineEngine for Downstream
Transportation & Distribution (GHG Protocol Scope 3 Category 9). It orchestrates
a 10-stage pipeline for complete downstream transport emissions calculation from
raw input to compliant output with full audit trail.

The 10 stages are:
1. VALIDATE: Input validation (required fields, types, ranges)
2. CLASSIFY: Incoterm classification (Cat 4 vs Cat 9), mode identification
3. NORMALIZE: Unit conversion (mass->tonnes, distance->km, currency->USD)
4. RESOLVE_EFS: Lookup emission factors from reference tables by mode/vehicle
5. CALCULATE: Route to appropriate calculator (distance/spend/average/warehouse/last-mile)
6. ALLOCATE: Product-level allocation by mass/volume/revenue
7. AGGREGATE: Sum by mode, channel, destination, method
8. COMPLIANCE: Run compliance checker (7 frameworks)
9. PROVENANCE: Build provenance chain with all 10 stage hashes
10. SEAL: Final SHA-256 seal + Merkle root

Sub-Activities:
    9a: Outbound transportation (post-sale transport per Incoterms)
    9b: Outbound distribution (DC / warehouse operations)
    9c: Retail storage (third-party retail energy)
    9d: Last-mile delivery (final delivery to end consumer)

Example:
    >>> from greenlang.agents.mrv.downstream_transportation.downstream_transport_pipeline import (
    ...     get_pipeline_engine,
    ... )
    >>> engine = get_pipeline_engine()
    >>> result = engine.calculate({
    ...     "calculation_id": "dto-001",
    ...     "shipments": [{"mode": "road", "distance_km": 500, "mass_kg": 10000}],
    ... })
    >>> print(f"Total emissions: {result['total_co2e']} kgCO2e")

Module: greenlang.agents.mrv.downstream_transportation.downstream_transport_pipeline
Agent: AGENT-MRV-022
Version: 1.0.0
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "dto_pipeline_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_8DP: Decimal = Decimal("0.00000001")
_QUANT_2DP: Decimal = Decimal("0.01")
ROUNDING: str = ROUND_HALF_UP


# ==============================================================================
# ENUMS
# ==============================================================================


class PipelineStage(str, Enum):
    """Pipeline stage enumeration."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE = "CALCULATE"
    ALLOCATE = "ALLOCATE"
    AGGREGATE = "AGGREGATE"
    COMPLIANCE = "COMPLIANCE"
    PROVENANCE = "PROVENANCE"
    SEAL = "SEAL"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class TransportMode(str, Enum):
    """Transport mode classification."""

    ROAD = "road"
    RAIL = "rail"
    SEA = "sea"
    AIR = "air"
    INLAND_WATERWAY = "inland_waterway"
    PIPELINE = "pipeline"
    MULTIMODAL = "multimodal"
    LAST_MILE = "last_mile"


class CalculationMethod(str, Enum):
    """Calculation method hierarchy."""

    SUPPLIER_SPECIFIC = "supplier_specific"
    DISTANCE_BASED = "distance_based"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    WAREHOUSE = "warehouse"
    LAST_MILE = "last_mile"


class AllocationMethod(str, Enum):
    """Product allocation method."""

    MASS = "mass"
    VOLUME = "volume"
    REVENUE = "revenue"
    UNITS = "units"
    ECONOMIC_VALUE = "economic_value"


class DistributionChannel(str, Enum):
    """Distribution channel classification."""

    DIRECT_TO_CONSUMER = "direct_to_consumer"
    WHOLESALE = "wholesale"
    RETAIL = "retail"
    E_COMMERCE = "e_commerce"
    THIRD_PARTY_LOGISTICS = "3pl"


class IncotermCategory(str, Enum):
    """Incoterm-based category classification."""

    CATEGORY_4 = "CATEGORY_4"
    CATEGORY_9 = "CATEGORY_9"
    AMBIGUOUS = "AMBIGUOUS"


# ==============================================================================
# INCOTERM CLASSIFICATION TABLE
# ==============================================================================

INCOTERM_TO_CATEGORY: Dict[str, IncotermCategory] = {
    # Seller pays downstream -> Category 9
    "DDP": IncotermCategory.CATEGORY_9,
    "DAP": IncotermCategory.CATEGORY_9,
    "DPU": IncotermCategory.CATEGORY_9,
    "CIF": IncotermCategory.CATEGORY_9,
    "CIP": IncotermCategory.CATEGORY_9,
    "CPT": IncotermCategory.CATEGORY_9,
    "CFR": IncotermCategory.CATEGORY_9,
    # Buyer pays downstream -> Category 4 (not Cat 9)
    "EXW": IncotermCategory.CATEGORY_4,
    "FCA": IncotermCategory.CATEGORY_4,
    "FAS": IncotermCategory.CATEGORY_4,
    "FOB": IncotermCategory.CATEGORY_4,
}


# ==============================================================================
# EMISSION FACTOR REFERENCE TABLES (embedded for pipeline self-sufficiency)
# ==============================================================================

# Distance-based emission factors: kgCO2e per tonne-km (DEFRA 2024 / GLEC v3.2)
DISTANCE_EF_BY_MODE: Dict[str, Dict[str, Decimal]] = {
    "road": {
        "articulated_diesel": Decimal("0.10544"),
        "rigid_diesel": Decimal("0.24910"),
        "van_diesel": Decimal("0.59540"),
        "van_petrol": Decimal("0.61239"),
        "average": Decimal("0.11501"),
        "wtt": Decimal("0.01460"),
    },
    "rail": {
        "freight_diesel": Decimal("0.02544"),
        "freight_electric": Decimal("0.00648"),
        "average": Decimal("0.01596"),
        "wtt": Decimal("0.00380"),
    },
    "sea": {
        "container_average": Decimal("0.01601"),
        "bulk_carrier": Decimal("0.00395"),
        "tanker": Decimal("0.00505"),
        "ro_ro": Decimal("0.04073"),
        "average": Decimal("0.01601"),
        "wtt": Decimal("0.00335"),
    },
    "air": {
        "freight_long_haul": Decimal("0.60290"),
        "freight_short_haul": Decimal("1.12850"),
        "belly_freight": Decimal("0.43980"),
        "average": Decimal("0.60290"),
        "wtt": Decimal("0.11000"),
    },
    "inland_waterway": {
        "barge": Decimal("0.03116"),
        "average": Decimal("0.03116"),
        "wtt": Decimal("0.00598"),
    },
    "pipeline": {
        "oil": Decimal("0.00437"),
        "gas": Decimal("0.01200"),
        "average": Decimal("0.00800"),
        "wtt": Decimal("0.00150"),
    },
    "last_mile": {
        "van_diesel": Decimal("0.59540"),
        "van_electric": Decimal("0.08950"),
        "cargo_bike": Decimal("0.00100"),
        "drone": Decimal("0.02500"),
        "average": Decimal("0.35000"),
        "wtt": Decimal("0.05200"),
    },
}

# Warehouse / distribution center emission factors: kgCO2e per tonne per day
WAREHOUSE_EF: Dict[str, Decimal] = {
    "ambient": Decimal("0.01850"),
    "chilled": Decimal("0.06520"),
    "frozen": Decimal("0.11340"),
    "cold_chain": Decimal("0.07800"),
    "cross_dock": Decimal("0.00520"),
    "average": Decimal("0.02500"),
}

# Retail storage emission factors: kgCO2e per tonne per day
RETAIL_STORAGE_EF: Dict[str, Decimal] = {
    "ambient_shelf": Decimal("0.00980"),
    "refrigerated_display": Decimal("0.04850"),
    "frozen_cabinet": Decimal("0.08700"),
    "average": Decimal("0.02100"),
}

# Spend-based EEIO factors: kgCO2e per USD (deflated to base year 2021)
SPEND_EEIO_FACTORS: Dict[str, Decimal] = {
    "road_freight": Decimal("0.42000"),
    "rail_freight": Decimal("0.18000"),
    "water_freight": Decimal("0.21000"),
    "air_freight": Decimal("1.87000"),
    "warehousing": Decimal("0.28000"),
    "courier_delivery": Decimal("0.55000"),
    "general_freight": Decimal("0.38000"),
    "logistics_services": Decimal("0.33000"),
}

# Average-data emission factors by distribution channel: kgCO2e per tonne
AVERAGE_CHANNEL_EF: Dict[str, Decimal] = {
    "direct_to_consumer": Decimal("45.00"),
    "wholesale": Decimal("25.00"),
    "retail": Decimal("35.00"),
    "e_commerce": Decimal("65.00"),
    "3pl": Decimal("30.00"),
    "average": Decimal("38.00"),
}

# Currency conversion rates to USD (simplified, would use live rates in production)
CURRENCY_TO_USD: Dict[str, Decimal] = {
    "USD": Decimal("1.000000"),
    "EUR": Decimal("1.089000"),
    "GBP": Decimal("1.264000"),
    "JPY": Decimal("0.006700"),
    "CNY": Decimal("0.138000"),
    "INR": Decimal("0.011950"),
    "CAD": Decimal("0.738000"),
    "AUD": Decimal("0.652000"),
    "BRL": Decimal("0.197000"),
    "KRW": Decimal("0.000750"),
    "CHF": Decimal("1.129000"),
    "SEK": Decimal("0.095000"),
    "NOK": Decimal("0.092000"),
    "DKK": Decimal("0.146000"),
    "MXN": Decimal("0.058000"),
    "ZAR": Decimal("0.055000"),
    "SGD": Decimal("0.746000"),
    "HKD": Decimal("0.128000"),
    "NZD": Decimal("0.607000"),
    "THB": Decimal("0.028000"),
}

# CPI deflators indexed to 2021 base year
CPI_DEFLATORS: Dict[int, Decimal] = {
    2018: Decimal("0.8920"),
    2019: Decimal("0.9100"),
    2020: Decimal("0.9220"),
    2021: Decimal("1.0000"),
    2022: Decimal("1.0800"),
    2023: Decimal("1.1150"),
    2024: Decimal("1.1480"),
    2025: Decimal("1.1780"),
    2026: Decimal("1.2050"),
}

# GWP values (AR5) for gas breakdown
GWP_CH4: Decimal = Decimal("28")
GWP_N2O: Decimal = Decimal("265")


# ==============================================================================
# DownstreamTransportPipelineEngine
# ==============================================================================


class DownstreamTransportPipelineEngine:
    """
    DownstreamTransportPipelineEngine - Orchestrated 10-stage pipeline for
    downstream transportation & distribution emissions (Category 9).

    This engine coordinates the complete calculation workflow through 10
    sequential stages, from input validation to sealed audit trail with
    Merkle root provenance. It supports all downstream sub-activities
    (9a outbound transport, 9b warehousing, 9c retail storage, 9d last-mile)
    and all four calculation methods (distance-based, spend-based,
    average-data, supplier-specific).

    Thread Safety:
        Singleton pattern with threading.RLock for concurrent access.
        Provenance chains are tracked per-calculation to avoid cross-talk.

    Attributes:
        _compliance_engine: Lazy-loaded ComplianceCheckerEngine
        _provenance_chains: Per-calculation provenance stage hashes
        _calculation_count: Running count of calculations performed

    Example:
        >>> engine = get_pipeline_engine()
        >>> result = engine.calculate({
        ...     "calculation_id": "dto-001",
        ...     "shipments": [
        ...         {"mode": "road", "distance_km": 500, "mass_kg": 10000}
        ...     ],
        ... })
        >>> result["status"]
        'SUCCESS'
    """

    _instance: Optional["DownstreamTransportPipelineEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "DownstreamTransportPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize DownstreamTransportPipelineEngine.

        Prevents re-initialization of the singleton. All sub-engines are
        lazy-loaded on first use.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._compliance_engine: Optional[Any] = None
        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}
        self._calculation_count: int = 0

        self._initialized = True
        logger.info(
            "DownstreamTransportPipelineEngine initialized (version %s, agent GL-MRV-S3-009)",
            ENGINE_VERSION,
        )

    # ==========================================================================
    # PUBLIC API: CORE PIPELINE
    # ==========================================================================

    def calculate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the 10-stage downstream transportation emissions pipeline.

        Args:
            input_data: Dictionary with calculation parameters. Expected keys:
                - calculation_id (str): Unique identifier
                - shipments (list): List of shipment dicts with mode, distance_km, mass_kg, etc.
                - warehouses (list, optional): List of warehouse dicts
                - last_mile (list, optional): List of last-mile delivery dicts
                - method (str, optional): Preferred calculation method
                - allocation_method (str, optional): mass/volume/revenue
                - incoterms (list, optional): Incoterm codes
                - reporting_period (str, optional): Reporting period
                - currency (str, optional): Currency for spend data
                - reporting_year (int, optional): Year for CPI deflation

        Returns:
            Pipeline result dictionary with:
                - status: SUCCESS / PARTIAL_SUCCESS / FAILED
                - total_co2e: Total emissions (kg CO2e)
                - total_co2: CO2 component (kg)
                - total_ch4: CH4 component (kg)
                - total_n2o: N2O component (kg)
                - wtt_co2e: Well-to-Tank emissions (kg CO2e)
                - by_mode: Emissions by transport mode
                - by_channel: Emissions by distribution channel
                - by_destination: Emissions by destination
                - by_method: Emissions by calculation method
                - compliance: Framework compliance results
                - provenance_hash: SHA-256 provenance seal
                - merkle_root: Merkle root of all stage hashes
                - stage_durations_ms: Duration per stage

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.
        """
        calc_id = input_data.get("calculation_id", f"dto-{int(time.time() * 1000)}")
        self._provenance_chains[calc_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # ------------------------------------------------------------------
            # Stage 1: VALIDATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            validated = self._stage_validate(input_data)
            duration_ms = self._elapsed_ms(start)
            stage_durations["VALIDATE"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.VALIDATE, input_data, validated)
            logger.info("[%s] Stage VALIDATE completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 2: CLASSIFY
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            classified = self._stage_classify(validated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CLASSIFY"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.CLASSIFY, validated, classified)
            logger.info("[%s] Stage CLASSIFY completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 3: NORMALIZE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            normalized = self._stage_normalize(classified)
            duration_ms = self._elapsed_ms(start)
            stage_durations["NORMALIZE"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.NORMALIZE, classified, normalized)
            logger.info("[%s] Stage NORMALIZE completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 4: RESOLVE_EFS
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            ef_resolved = self._stage_resolve_efs(normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations["RESOLVE_EFS"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.RESOLVE_EFS, normalized, ef_resolved)
            logger.info("[%s] Stage RESOLVE_EFS completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 5: CALCULATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            calculated = self._stage_calculate(ef_resolved)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CALCULATE"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.CALCULATE, ef_resolved, calculated)
            logger.info(
                "[%s] Stage CALCULATE completed in %.2fms (%d shipment results)",
                calc_id, duration_ms, len(calculated.get("shipment_results", [])),
            )

            # ------------------------------------------------------------------
            # Stage 6: ALLOCATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            allocated = self._stage_allocate(calculated, normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations["ALLOCATE"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.ALLOCATE, calculated, allocated)
            logger.info("[%s] Stage ALLOCATE completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 7: AGGREGATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            aggregated = self._stage_aggregate(allocated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["AGGREGATE"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.AGGREGATE, allocated, aggregated)
            logger.info("[%s] Stage AGGREGATE completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 8: COMPLIANCE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            compliance = self._stage_compliance(aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["COMPLIANCE"] = duration_ms
            self._record_provenance(calc_id, PipelineStage.COMPLIANCE, aggregated, compliance)
            logger.info("[%s] Stage COMPLIANCE completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 9: PROVENANCE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            provenance_data = self._stage_provenance(calc_id)
            duration_ms = self._elapsed_ms(start)
            stage_durations["PROVENANCE"] = duration_ms
            logger.info("[%s] Stage PROVENANCE completed in %.2fms", calc_id, duration_ms)

            # ------------------------------------------------------------------
            # Stage 10: SEAL
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            seal_data = self._stage_seal(calc_id, aggregated, provenance_data)
            duration_ms = self._elapsed_ms(start)
            stage_durations["SEAL"] = duration_ms
            logger.info("[%s] Stage SEAL completed in %.2fms", calc_id, duration_ms)

            # Build final result
            total_dur = sum(stage_durations.values())
            self._calculation_count += 1

            result = {
                "calculation_id": calc_id,
                "status": PipelineStatus.SUCCESS.value,
                "total_co2e": str(aggregated["total_co2e"]),
                "total_co2": str(aggregated.get("total_co2", Decimal("0"))),
                "total_ch4": str(aggregated.get("total_ch4", Decimal("0"))),
                "total_n2o": str(aggregated.get("total_n2o", Decimal("0"))),
                "wtt_co2e": str(aggregated.get("wtt_co2e", Decimal("0"))),
                "emission_scope": aggregated.get("emission_scope", "WTW"),
                "by_mode": aggregated.get("by_mode", {}),
                "by_channel": aggregated.get("by_channel", {}),
                "by_destination": aggregated.get("by_destination", {}),
                "by_method": aggregated.get("by_method", {}),
                "shipment_results": allocated.get("shipment_results", []),
                "warehouse_results": allocated.get("warehouse_results", []),
                "last_mile_results": allocated.get("last_mile_results", []),
                "allocation_method": normalized.get("allocation_method", "mass"),
                "compliance": compliance,
                "provenance_hash": seal_data["provenance_hash"],
                "merkle_root": seal_data["merkle_root"],
                "stage_durations_ms": stage_durations,
                "total_duration_ms": total_dur,
                "method": normalized.get("method", "distance_based"),
                "reporting_period": normalized.get("reporting_period"),
                "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "[%s] Pipeline completed successfully in %.2fms. "
                "Total emissions: %s kgCO2e",
                calc_id, total_dur, aggregated["total_co2e"],
            )

            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error("[%s] Pipeline execution failed: %s", calc_id, e, exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {e}") from e
        finally:
            self._provenance_chains.pop(calc_id, None)

    # ==========================================================================
    # PUBLIC API: BATCH PROCESSING
    # ==========================================================================

    def calculate_batch(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a list of calculation inputs with error isolation.

        Each input is independently processed through the full 10-stage
        pipeline. Errors in one input do not affect others.

        Args:
            inputs: List of calculation input dictionaries.

        Returns:
            Batch result with individual results, totals, and error details.
        """
        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        logger.info("Starting batch calculation (%d inputs)", len(inputs))

        for idx, inp in enumerate(inputs):
            try:
                result = self.calculate(inp)
                results.append(result)
            except Exception as e:
                logger.error("Batch item %d failed: %s", idx, e)
                errors.append({
                    "index": idx,
                    "calculation_id": inp.get("calculation_id", f"batch-{idx}"),
                    "error": str(e),
                })

        # Aggregate totals
        total_co2e = sum(
            (Decimal(str(r.get("total_co2e", "0"))) for r in results),
            Decimal("0"),
        )
        successful = len(results)
        failed = len(errors)
        duration_ms = (time.monotonic() - start_time) * 1000.0

        # Batch provenance hash
        batch_data = {
            "calculation_hashes": [r.get("provenance_hash", "") for r in results],
            "total_co2e": str(total_co2e),
            "count": successful,
        }
        batch_hash = hashlib.sha256(
            json.dumps(batch_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        logger.info(
            "Batch completed in %.2fms. Success=%d, Failed=%d, Total=%s kgCO2e",
            duration_ms, successful, failed, total_co2e,
        )

        return {
            "status": PipelineStatus.SUCCESS.value if failed == 0 else PipelineStatus.PARTIAL_SUCCESS.value,
            "results": results,
            "errors": errors,
            "total_co2e": str(total_co2e),
            "successful_count": successful,
            "failed_count": failed,
            "batch_duration_ms": duration_ms,
            "provenance_hash": batch_hash,
        }

    # ==========================================================================
    # PUBLIC API: SHORTCUT METHODS
    # ==========================================================================

    def calculate_distance_method(
        self, shipments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Distance-based calculation shortcut for a list of shipments.

        Args:
            shipments: List of shipment dicts with mode, distance_km, mass_kg.

        Returns:
            Pipeline result using distance-based method.

        Example:
            >>> result = engine.calculate_distance_method([
            ...     {"mode": "road", "distance_km": 500, "mass_kg": 10000},
            ...     {"mode": "rail", "distance_km": 1200, "mass_kg": 25000},
            ... ])
        """
        return self.calculate({
            "calculation_id": f"dto-dist-{int(time.time() * 1000)}",
            "shipments": shipments,
            "method": CalculationMethod.DISTANCE_BASED.value,
        })

    def calculate_spend_method(
        self, spend_inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Spend-based calculation shortcut.

        Args:
            spend_inputs: List of spend dicts with category, amount, currency.

        Returns:
            Pipeline result using spend-based method.

        Example:
            >>> result = engine.calculate_spend_method([
            ...     {"category": "road_freight", "amount": 50000, "currency": "USD"},
            ... ])
        """
        return self.calculate({
            "calculation_id": f"dto-spend-{int(time.time() * 1000)}",
            "spend_data": spend_inputs,
            "method": CalculationMethod.SPEND_BASED.value,
        })

    def calculate_average_method(
        self, avg_inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Average-data calculation shortcut.

        Args:
            avg_inputs: List of average-data dicts with channel, mass_tonnes.

        Returns:
            Pipeline result using average-data method.

        Example:
            >>> result = engine.calculate_average_method([
            ...     {"channel": "e_commerce", "mass_tonnes": 100},
            ... ])
        """
        return self.calculate({
            "calculation_id": f"dto-avg-{int(time.time() * 1000)}",
            "average_data": avg_inputs,
            "method": CalculationMethod.AVERAGE_DATA.value,
        })

    def calculate_distribution_chain(
        self,
        warehouses: List[Dict[str, Any]],
        last_mile: List[Dict[str, Any]],
        shipments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Combined distribution chain calculation (9b + 9d + optional 9a).

        Combines warehouse/DC emissions, last-mile delivery, and optionally
        outbound transportation into a single pipeline run.

        Args:
            warehouses: List of warehouse dicts with type, mass_tonnes, storage_days.
            last_mile: List of last-mile dicts with mode, distance_km, mass_kg.
            shipments: Optional list of outbound transport shipments.

        Returns:
            Pipeline result covering the full distribution chain.

        Example:
            >>> result = engine.calculate_distribution_chain(
            ...     warehouses=[{"type": "chilled", "mass_tonnes": 50, "storage_days": 3}],
            ...     last_mile=[{"mode": "van_diesel", "distance_km": 15, "mass_kg": 500}],
            ...     shipments=[{"mode": "road", "distance_km": 200, "mass_kg": 50000}],
            ... )
        """
        return self.calculate({
            "calculation_id": f"dto-chain-{int(time.time() * 1000)}",
            "shipments": shipments or [],
            "warehouses": warehouses,
            "last_mile": last_mile,
            "method": CalculationMethod.DISTANCE_BASED.value,
        })

    # ==========================================================================
    # STAGE 1: VALIDATE
    # ==========================================================================

    def _stage_validate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: VALIDATE - Input validation.

        Validates required fields, types, ranges, and data consistency.

        Args:
            input_data: Raw calculation input.

        Returns:
            Validated input (pass-through if valid).

        Raises:
            ValueError: If validation fails with detailed error messages.
        """
        errors: List[str] = []

        # At least one data source required
        has_shipments = bool(input_data.get("shipments"))
        has_spend = bool(input_data.get("spend_data"))
        has_average = bool(input_data.get("average_data"))
        has_warehouses = bool(input_data.get("warehouses"))
        has_last_mile = bool(input_data.get("last_mile"))

        if not any([has_shipments, has_spend, has_average, has_warehouses, has_last_mile]):
            errors.append(
                "At least one data source required: shipments, spend_data, "
                "average_data, warehouses, or last_mile."
            )

        # Validate shipments
        for idx, shipment in enumerate(input_data.get("shipments") or []):
            if not shipment.get("mode"):
                errors.append(f"Shipment {idx}: 'mode' is required")
            if shipment.get("distance_km") is not None:
                try:
                    dist = Decimal(str(shipment["distance_km"]))
                    if dist < 0:
                        errors.append(f"Shipment {idx}: distance_km must be >= 0")
                except (InvalidOperation, ValueError):
                    errors.append(f"Shipment {idx}: distance_km must be numeric")
            if shipment.get("mass_kg") is not None:
                try:
                    mass = Decimal(str(shipment["mass_kg"]))
                    if mass < 0:
                        errors.append(f"Shipment {idx}: mass_kg must be >= 0")
                except (InvalidOperation, ValueError):
                    errors.append(f"Shipment {idx}: mass_kg must be numeric")

        # Validate spend data
        for idx, spend in enumerate(input_data.get("spend_data") or []):
            if not spend.get("category"):
                errors.append(f"Spend {idx}: 'category' is required")
            if spend.get("amount") is None:
                errors.append(f"Spend {idx}: 'amount' is required")
            elif Decimal(str(spend["amount"])) < 0:
                errors.append(f"Spend {idx}: amount must be >= 0")

        # Validate warehouses
        for idx, wh in enumerate(input_data.get("warehouses") or []):
            if wh.get("mass_tonnes") is None and wh.get("mass_kg") is None:
                errors.append(f"Warehouse {idx}: mass_tonnes or mass_kg is required")
            if wh.get("storage_days") is not None:
                try:
                    days = Decimal(str(wh["storage_days"]))
                    if days < 0:
                        errors.append(f"Warehouse {idx}: storage_days must be >= 0")
                except (InvalidOperation, ValueError):
                    errors.append(f"Warehouse {idx}: storage_days must be numeric")

        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        return input_data

    # ==========================================================================
    # STAGE 2: CLASSIFY
    # ==========================================================================

    def _stage_classify(self, validated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: CLASSIFY - Incoterm classification and mode identification.

        Classifies each shipment by Incoterm (Cat 4 vs Cat 9), identifies
        transport modes, and selects calculation method.

        Args:
            validated: Validated input data.

        Returns:
            Enriched data with classification metadata.
        """
        data = dict(validated)

        # Classify Incoterms
        incoterms = data.get("incoterms") or []
        cat_9_incoterms: List[str] = []
        cat_4_incoterms: List[str] = []

        for ic in incoterms:
            ic_upper = str(ic).upper()
            category = INCOTERM_TO_CATEGORY.get(ic_upper, IncotermCategory.AMBIGUOUS)
            if category == IncotermCategory.CATEGORY_9:
                cat_9_incoterms.append(ic_upper)
            elif category == IncotermCategory.CATEGORY_4:
                cat_4_incoterms.append(ic_upper)

        data["_cat_9_incoterms"] = cat_9_incoterms
        data["_cat_4_incoterms"] = cat_4_incoterms

        # Classify shipment modes
        modes_found: List[str] = []
        for shipment in data.get("shipments") or []:
            mode = str(shipment.get("mode", "road")).lower()
            if mode not in modes_found:
                modes_found.append(mode)

        data["_modes_found"] = modes_found

        # Determine primary calculation method
        method = data.get("method")
        if not method:
            if data.get("shipments"):
                has_distance = any(
                    s.get("distance_km") for s in data.get("shipments", [])
                )
                if has_distance:
                    method = CalculationMethod.DISTANCE_BASED.value
                else:
                    method = CalculationMethod.AVERAGE_DATA.value
            elif data.get("spend_data"):
                method = CalculationMethod.SPEND_BASED.value
            elif data.get("average_data"):
                method = CalculationMethod.AVERAGE_DATA.value
            elif data.get("warehouses") or data.get("last_mile"):
                method = CalculationMethod.WAREHOUSE.value
            else:
                method = CalculationMethod.AVERAGE_DATA.value

        data["method"] = method

        logger.debug(
            "Classification: modes=%s, method=%s, cat9=%d, cat4=%d",
            modes_found, method, len(cat_9_incoterms), len(cat_4_incoterms),
        )

        return data

    # ==========================================================================
    # STAGE 3: NORMALIZE
    # ==========================================================================

    def _stage_normalize(self, classified: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: NORMALIZE - Unit conversion and currency normalization.

        Conversions:
        - Mass: kg -> tonnes (divide by 1000)
        - Distance: miles -> km (multiply by 1.60934)
        - Currency: any -> USD using exchange rates
        - CPI: deflate spend to base year 2021

        Args:
            classified: Classified input data.

        Returns:
            Data with normalized units.
        """
        data = dict(classified)

        # Normalize shipments
        for shipment in data.get("shipments") or []:
            # Mass normalization: kg -> tonnes
            if "mass_kg" in shipment and "mass_tonnes" not in shipment:
                mass_kg = Decimal(str(shipment["mass_kg"]))
                shipment["mass_tonnes"] = str(
                    (mass_kg / Decimal("1000")).quantize(_QUANT_8DP, rounding=ROUNDING)
                )

            # Distance normalization: miles -> km
            if "distance_miles" in shipment and "distance_km" not in shipment:
                miles = Decimal(str(shipment["distance_miles"]))
                shipment["distance_km"] = str(
                    (miles * Decimal("1.60934")).quantize(_QUANT_8DP, rounding=ROUNDING)
                )

        # Normalize warehouse mass
        for wh in data.get("warehouses") or []:
            if "mass_kg" in wh and "mass_tonnes" not in wh:
                mass_kg = Decimal(str(wh["mass_kg"]))
                wh["mass_tonnes"] = str(
                    (mass_kg / Decimal("1000")).quantize(_QUANT_8DP, rounding=ROUNDING)
                )

        # Normalize last-mile mass
        for lm in data.get("last_mile") or []:
            if "mass_kg" in lm and "mass_tonnes" not in lm:
                mass_kg = Decimal(str(lm["mass_kg"]))
                lm["mass_tonnes"] = str(
                    (mass_kg / Decimal("1000")).quantize(_QUANT_8DP, rounding=ROUNDING)
                )
            if "distance_miles" in lm and "distance_km" not in lm:
                miles = Decimal(str(lm["distance_miles"]))
                lm["distance_km"] = str(
                    (miles * Decimal("1.60934")).quantize(_QUANT_8DP, rounding=ROUNDING)
                )

        # Normalize spend data (currency conversion + CPI deflation)
        reporting_year = data.get("reporting_year", 2024)
        currency = data.get("currency", "USD").upper()

        for spend in data.get("spend_data") or []:
            spend_currency = str(spend.get("currency", currency)).upper()
            amount = Decimal(str(spend["amount"]))

            # Currency conversion
            rate = CURRENCY_TO_USD.get(spend_currency, Decimal("1.0"))
            amount_usd = (amount * rate).quantize(_QUANT_8DP, rounding=ROUNDING)
            spend["amount_usd"] = str(amount_usd)

            # CPI deflation to base year 2021
            deflator = CPI_DEFLATORS.get(reporting_year, Decimal("1.0"))
            amount_deflated = (amount_usd / deflator).quantize(_QUANT_8DP, rounding=ROUNDING)
            spend["amount_usd_deflated"] = str(amount_deflated)

        return data

    # ==========================================================================
    # STAGE 4: RESOLVE_EFS
    # ==========================================================================

    def _stage_resolve_efs(self, normalized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 4: RESOLVE_EFS - Lookup emission factors from reference tables.

        Resolves EFs by mode and vehicle/vessel type from the embedded
        reference tables. Uses the hierarchy:
        1. Supplier-specific (if provided in shipment)
        2. Vehicle-specific DEFRA/GLEC
        3. Mode-average DEFRA/GLEC

        Args:
            normalized: Normalized input data.

        Returns:
            Data enriched with resolved emission factors.
        """
        data = dict(normalized)

        # Resolve EFs for shipments
        for shipment in data.get("shipments") or []:
            mode = str(shipment.get("mode", "road")).lower()
            vehicle_type = str(shipment.get("vehicle_type", "average")).lower()

            # Check for supplier-specific EF
            if shipment.get("supplier_ef"):
                shipment["_ef"] = str(shipment["supplier_ef"])
                shipment["_ef_source"] = "supplier_specific"
                shipment["_wtt_ef"] = str(shipment.get("supplier_wtt_ef", "0"))
                continue

            # Lookup from reference table
            mode_efs = DISTANCE_EF_BY_MODE.get(mode, DISTANCE_EF_BY_MODE["road"])
            ef = mode_efs.get(vehicle_type, mode_efs.get("average", Decimal("0.11501")))
            wtt_ef = mode_efs.get("wtt", Decimal("0"))

            shipment["_ef"] = str(ef)
            shipment["_wtt_ef"] = str(wtt_ef)
            shipment["_ef_source"] = "DEFRA_GLEC"

        # Resolve EFs for warehouses
        for wh in data.get("warehouses") or []:
            wh_type = str(wh.get("type", "average")).lower()
            ef = WAREHOUSE_EF.get(wh_type, WAREHOUSE_EF["average"])
            wh["_ef"] = str(ef)
            wh["_ef_source"] = "DEFRA_GLEC"

        # Resolve EFs for last-mile
        for lm in data.get("last_mile") or []:
            mode = str(lm.get("mode", "average")).lower()
            lm_efs = DISTANCE_EF_BY_MODE.get("last_mile", {})
            ef = lm_efs.get(mode, lm_efs.get("average", Decimal("0.35000")))
            wtt_ef = lm_efs.get("wtt", Decimal("0"))
            lm["_ef"] = str(ef)
            lm["_wtt_ef"] = str(wtt_ef)
            lm["_ef_source"] = "DEFRA_GLEC"

        # Resolve EEIO factors for spend-based
        for spend in data.get("spend_data") or []:
            category = str(spend.get("category", "general_freight")).lower()
            eeio_ef = SPEND_EEIO_FACTORS.get(category, SPEND_EEIO_FACTORS["general_freight"])
            spend["_eeio_ef"] = str(eeio_ef)
            spend["_ef_source"] = "EEIO"

        # Resolve average-data factors
        for avg in data.get("average_data") or []:
            channel = str(avg.get("channel", "average")).lower()
            ef = AVERAGE_CHANNEL_EF.get(channel, AVERAGE_CHANNEL_EF["average"])
            avg["_ef"] = str(ef)
            avg["_ef_source"] = "industry_average"

        return data

    # ==========================================================================
    # STAGE 5: CALCULATE
    # ==========================================================================

    def _stage_calculate(self, ef_resolved: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 5: CALCULATE - Execute emissions calculations.

        Routes to the appropriate calculation engine based on method:
        - Distance-based: tonne-km x mode-specific EF
        - Spend-based: deflated USD x EEIO factor
        - Average-data: tonnes x channel-specific factor
        - Warehouse: tonnes x days x warehouse EF
        - Last-mile: tonne-km x last-mile EF

        All arithmetic uses Decimal with ROUND_HALF_UP 8dp.

        Args:
            ef_resolved: Data with resolved emission factors.

        Returns:
            Data with calculation results for each item.
        """
        data = dict(ef_resolved)
        shipment_results: List[Dict[str, Any]] = []
        warehouse_results: List[Dict[str, Any]] = []
        last_mile_results: List[Dict[str, Any]] = []

        # Calculate distance-based shipments
        for idx, shipment in enumerate(data.get("shipments") or []):
            result = self._calc_distance_shipment(idx, shipment)
            shipment_results.append(result)

        # Calculate spend-based
        for idx, spend in enumerate(data.get("spend_data") or []):
            result = self._calc_spend_item(idx, spend)
            shipment_results.append(result)

        # Calculate average-data
        for idx, avg in enumerate(data.get("average_data") or []):
            result = self._calc_average_item(idx, avg)
            shipment_results.append(result)

        # Calculate warehouse emissions
        for idx, wh in enumerate(data.get("warehouses") or []):
            result = self._calc_warehouse_item(idx, wh)
            warehouse_results.append(result)

        # Calculate last-mile emissions
        for idx, lm in enumerate(data.get("last_mile") or []):
            result = self._calc_last_mile_item(idx, lm)
            last_mile_results.append(result)

        data["shipment_results"] = shipment_results
        data["warehouse_results"] = warehouse_results
        data["last_mile_results"] = last_mile_results

        return data

    def _calc_distance_shipment(
        self, idx: int, shipment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate emissions for a single distance-based shipment."""
        distance_km = Decimal(str(shipment.get("distance_km", "0")))
        mass_tonnes = Decimal(str(shipment.get("mass_tonnes", "0")))
        ef = Decimal(str(shipment.get("_ef", "0.11501")))
        wtt_ef = Decimal(str(shipment.get("_wtt_ef", "0")))

        # tonne-km x EF
        tonne_km = (distance_km * mass_tonnes).quantize(_QUANT_8DP, rounding=ROUNDING)
        co2e = (tonne_km * ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        wtt_co2e = (tonne_km * wtt_ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        total_co2e = (co2e + wtt_co2e).quantize(_QUANT_8DP, rounding=ROUNDING)

        # Approximate gas breakdown (transport typical: 99% CO2, 0.5% CH4, 0.5% N2O)
        co2 = (co2e * Decimal("0.99")).quantize(_QUANT_8DP, rounding=ROUNDING)
        ch4_co2e = (co2e * Decimal("0.005")).quantize(_QUANT_8DP, rounding=ROUNDING)
        n2o_co2e = (co2e * Decimal("0.005")).quantize(_QUANT_8DP, rounding=ROUNDING)
        ch4 = (ch4_co2e / GWP_CH4).quantize(_QUANT_8DP, rounding=ROUNDING)
        n2o = (n2o_co2e / GWP_N2O).quantize(_QUANT_8DP, rounding=ROUNDING)

        return {
            "index": idx,
            "type": "shipment",
            "sub_activity": "9a",
            "mode": shipment.get("mode", "road"),
            "method": CalculationMethod.DISTANCE_BASED.value,
            "distance_km": str(distance_km),
            "mass_tonnes": str(mass_tonnes),
            "tonne_km": str(tonne_km),
            "ef": str(ef),
            "wtt_ef": str(wtt_ef),
            "ef_source": shipment.get("_ef_source", "DEFRA_GLEC"),
            "co2e": str(co2e),
            "wtt_co2e": str(wtt_co2e),
            "total_co2e": str(total_co2e),
            "co2": str(co2),
            "ch4": str(ch4),
            "n2o": str(n2o),
            "destination": shipment.get("destination", ""),
            "channel": shipment.get("channel", ""),
        }

    def _calc_spend_item(
        self, idx: int, spend: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate emissions for a single spend-based item."""
        amount_deflated = Decimal(str(spend.get("amount_usd_deflated", spend.get("amount", "0"))))
        eeio_ef = Decimal(str(spend.get("_eeio_ef", "0.38")))

        co2e = (amount_deflated * eeio_ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        total_co2e = co2e  # No separate WTT for spend-based

        return {
            "index": idx,
            "type": "spend",
            "sub_activity": "9a",
            "mode": spend.get("category", "general_freight"),
            "method": CalculationMethod.SPEND_BASED.value,
            "amount_usd_deflated": str(amount_deflated),
            "eeio_ef": str(eeio_ef),
            "ef_source": "EEIO",
            "co2e": str(co2e),
            "wtt_co2e": "0",
            "total_co2e": str(total_co2e),
            "co2": str(co2e),
            "ch4": "0",
            "n2o": "0",
            "destination": spend.get("destination", ""),
            "channel": spend.get("channel", ""),
        }

    def _calc_average_item(
        self, idx: int, avg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate emissions for a single average-data item."""
        mass_tonnes = Decimal(str(avg.get("mass_tonnes", "0")))
        ef = Decimal(str(avg.get("_ef", "38.00")))

        co2e = (mass_tonnes * ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        total_co2e = co2e

        return {
            "index": idx,
            "type": "average",
            "sub_activity": "9a",
            "mode": avg.get("channel", "average"),
            "method": CalculationMethod.AVERAGE_DATA.value,
            "mass_tonnes": str(mass_tonnes),
            "ef": str(ef),
            "ef_source": "industry_average",
            "co2e": str(co2e),
            "wtt_co2e": "0",
            "total_co2e": str(total_co2e),
            "co2": str(co2e),
            "ch4": "0",
            "n2o": "0",
            "destination": avg.get("destination", ""),
            "channel": avg.get("channel", ""),
        }

    def _calc_warehouse_item(
        self, idx: int, wh: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate emissions for a single warehouse/DC."""
        mass_tonnes = Decimal(str(wh.get("mass_tonnes", "0")))
        storage_days = Decimal(str(wh.get("storage_days", "1")))
        ef = Decimal(str(wh.get("_ef", "0.025")))

        # kgCO2e = mass_tonnes x storage_days x ef_per_tonne_day
        co2e = (mass_tonnes * storage_days * ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        total_co2e = co2e

        return {
            "index": idx,
            "type": "warehouse",
            "sub_activity": "9b",
            "warehouse_type": wh.get("type", "average"),
            "method": CalculationMethod.WAREHOUSE.value,
            "mass_tonnes": str(mass_tonnes),
            "storage_days": str(storage_days),
            "ef": str(ef),
            "ef_source": wh.get("_ef_source", "DEFRA_GLEC"),
            "co2e": str(co2e),
            "wtt_co2e": "0",
            "total_co2e": str(total_co2e),
        }

    def _calc_last_mile_item(
        self, idx: int, lm: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate emissions for a single last-mile delivery."""
        distance_km = Decimal(str(lm.get("distance_km", "0")))
        mass_tonnes = Decimal(str(lm.get("mass_tonnes", "0")))
        ef = Decimal(str(lm.get("_ef", "0.35")))
        wtt_ef = Decimal(str(lm.get("_wtt_ef", "0")))

        tonne_km = (distance_km * mass_tonnes).quantize(_QUANT_8DP, rounding=ROUNDING)
        co2e = (tonne_km * ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        wtt_co2e = (tonne_km * wtt_ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        total_co2e = (co2e + wtt_co2e).quantize(_QUANT_8DP, rounding=ROUNDING)

        return {
            "index": idx,
            "type": "last_mile",
            "sub_activity": "9d",
            "mode": lm.get("mode", "van_diesel"),
            "method": CalculationMethod.LAST_MILE.value,
            "distance_km": str(distance_km),
            "mass_tonnes": str(mass_tonnes),
            "tonne_km": str(tonne_km),
            "ef": str(ef),
            "wtt_ef": str(wtt_ef),
            "ef_source": lm.get("_ef_source", "DEFRA_GLEC"),
            "co2e": str(co2e),
            "wtt_co2e": str(wtt_co2e),
            "total_co2e": str(total_co2e),
        }

    # ==========================================================================
    # STAGE 6: ALLOCATE
    # ==========================================================================

    def _stage_allocate(
        self, calculated: Dict[str, Any], normalized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 6: ALLOCATE - Product-level allocation.

        Applies allocation factors to distribute shared transport emissions
        across products using mass, volume, or revenue allocation.

        Args:
            calculated: Data with raw calculation results.
            normalized: Normalized input (for allocation parameters).

        Returns:
            Data with allocated emissions per product/shipment.
        """
        data = dict(calculated)
        allocation_method = normalized.get("allocation_method", "mass")
        allocation_factor = Decimal(str(normalized.get("allocation_factor", "1.0")))

        # Apply allocation factor to shipment results
        for result in data.get("shipment_results", []):
            factor = allocation_factor
            # Allow shipment-level override
            if "allocation_factor" in result:
                factor = Decimal(str(result["allocation_factor"]))

            original_co2e = Decimal(str(result["total_co2e"]))
            allocated_co2e = (original_co2e * factor).quantize(_QUANT_8DP, rounding=ROUNDING)
            result["allocated_co2e"] = str(allocated_co2e)
            result["allocation_method"] = allocation_method
            result["allocation_factor"] = str(factor)

        # Apply to warehouse results
        for result in data.get("warehouse_results", []):
            original_co2e = Decimal(str(result["total_co2e"]))
            allocated_co2e = (original_co2e * allocation_factor).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
            result["allocated_co2e"] = str(allocated_co2e)
            result["allocation_method"] = allocation_method
            result["allocation_factor"] = str(allocation_factor)

        # Apply to last-mile results
        for result in data.get("last_mile_results", []):
            original_co2e = Decimal(str(result["total_co2e"]))
            allocated_co2e = (original_co2e * allocation_factor).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )
            result["allocated_co2e"] = str(allocated_co2e)
            result["allocation_method"] = allocation_method
            result["allocation_factor"] = str(allocation_factor)

        data["allocation_method"] = allocation_method
        return data

    # ==========================================================================
    # STAGE 7: AGGREGATE
    # ==========================================================================

    def _stage_aggregate(self, allocated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 7: AGGREGATE - Sum emissions by mode, channel, destination, method.

        Args:
            allocated: Data with allocated emissions.

        Returns:
            Aggregated totals across all dimensions.
        """
        total_co2e = Decimal("0")
        total_co2 = Decimal("0")
        total_ch4 = Decimal("0")
        total_n2o = Decimal("0")
        wtt_co2e = Decimal("0")

        by_mode: Dict[str, str] = {}
        by_channel: Dict[str, str] = {}
        by_destination: Dict[str, str] = {}
        by_method: Dict[str, str] = {}

        all_results = (
            allocated.get("shipment_results", [])
            + allocated.get("warehouse_results", [])
            + allocated.get("last_mile_results", [])
        )

        for result in all_results:
            emissions = Decimal(str(result.get("allocated_co2e", result.get("total_co2e", "0"))))
            total_co2e += emissions

            result_co2 = Decimal(str(result.get("co2", "0")))
            result_ch4 = Decimal(str(result.get("ch4", "0")))
            result_n2o = Decimal(str(result.get("n2o", "0")))
            result_wtt = Decimal(str(result.get("wtt_co2e", "0")))

            total_co2 += result_co2
            total_ch4 += result_ch4
            total_n2o += result_n2o
            wtt_co2e += result_wtt

            # By mode
            mode = result.get("mode", result.get("warehouse_type", "other"))
            mode_total = Decimal(str(by_mode.get(mode, "0")))
            by_mode[mode] = str(mode_total + emissions)

            # By channel
            channel = result.get("channel")
            if channel:
                ch_total = Decimal(str(by_channel.get(channel, "0")))
                by_channel[channel] = str(ch_total + emissions)

            # By destination
            destination = result.get("destination")
            if destination:
                dest_total = Decimal(str(by_destination.get(destination, "0")))
                by_destination[destination] = str(dest_total + emissions)

            # By method
            method = result.get("method", "unknown")
            method_total = Decimal(str(by_method.get(method, "0")))
            by_method[method] = str(method_total + emissions)

        return {
            "total_co2e": str(total_co2e.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "total_co2": str(total_co2.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "total_ch4": str(total_ch4.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "total_n2o": str(total_n2o.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "wtt_co2e": str(wtt_co2e.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "emission_scope": "WTW",
            "by_mode": by_mode,
            "by_channel": by_channel,
            "by_destination": by_destination,
            "by_method": by_method,
            "item_count": len(all_results),
            "shipment_results": allocated.get("shipment_results", []),
            "warehouse_results": allocated.get("warehouse_results", []),
            "last_mile_results": allocated.get("last_mile_results", []),
            "method": allocated.get("method", "distance_based"),
            "reporting_period": allocated.get("reporting_period"),
        }

    # ==========================================================================
    # STAGE 8: COMPLIANCE
    # ==========================================================================

    def _stage_compliance(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 8: COMPLIANCE - Run compliance checker.

        Delegates to the ComplianceCheckerEngine for all 7 frameworks.
        Falls back to lightweight inline check if engine is unavailable.

        Args:
            aggregated: Aggregated calculation results.

        Returns:
            Compliance check results.
        """
        engine = self._get_compliance_engine()
        if engine is not None:
            try:
                compliance_input = {
                    "total_co2e": aggregated["total_co2e"],
                    "method": aggregated.get("method"),
                    "calculation_method": aggregated.get("method"),
                    "mode_breakdown": aggregated.get("by_mode"),
                    "by_mode": aggregated.get("by_mode"),
                    "emission_scope": aggregated.get("emission_scope", "WTW"),
                    "reporting_period": aggregated.get("reporting_period"),
                }
                return engine.check_all_frameworks(compliance_input)
            except Exception as e:
                logger.warning("ComplianceCheckerEngine failed, using inline: %s", e)

        # Inline lightweight compliance check
        findings: List[str] = []
        status = "pass"

        total_co2e = Decimal(str(aggregated.get("total_co2e", "0")))
        if total_co2e <= 0:
            findings.append("GHG Protocol: total_co2e is zero or missing")
            status = "fail"

        if not aggregated.get("method"):
            findings.append("GHG Protocol: calculation method not specified")
            status = "warning"

        return {
            "status": status,
            "findings": findings,
            "frameworks_checked": ["ghg_protocol"],
            "inline": True,
        }

    # ==========================================================================
    # STAGE 9: PROVENANCE
    # ==========================================================================

    def _stage_provenance(self, calc_id: str) -> Dict[str, Any]:
        """
        Stage 9: PROVENANCE - Build provenance chain with all stage hashes.

        Collects the hashes recorded during each of the 8 previous stages
        and creates a chain data structure suitable for Merkle root calculation.

        Args:
            calc_id: Calculation identifier.

        Returns:
            Provenance data with stage hashes and chain hash.
        """
        chain = self._provenance_chains.get(calc_id, [])

        stage_hashes = [entry.get("output_hash", "") for entry in chain]
        chain_hash = hashlib.sha256(
            "|".join(stage_hashes).encode("utf-8")
        ).hexdigest()

        return {
            "calc_id": calc_id,
            "stage_count": len(chain),
            "stage_hashes": stage_hashes,
            "chain_hash": chain_hash,
            "stages": [
                {
                    "stage": entry.get("stage", ""),
                    "input_hash": entry.get("input_hash", ""),
                    "output_hash": entry.get("output_hash", ""),
                    "timestamp": entry.get("timestamp", ""),
                }
                for entry in chain
            ],
        }

    # ==========================================================================
    # STAGE 10: SEAL
    # ==========================================================================

    def _stage_seal(
        self,
        calc_id: str,
        aggregated: Dict[str, Any],
        provenance_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Stage 10: SEAL - Final SHA-256 seal + Merkle root.

        Creates an immutable audit fingerprint by:
        1. Combining the provenance chain hash with final aggregated results
        2. Computing the final SHA-256 seal
        3. Computing a Merkle root over all individual stage hashes

        Args:
            calc_id: Calculation identifier.
            aggregated: Aggregated results.
            provenance_data: Provenance chain data from Stage 9.

        Returns:
            Seal data with provenance_hash and merkle_root.
        """
        # Final provenance hash: chain_hash + total_co2e
        seal_input = (
            f"{provenance_data['chain_hash']}|"
            f"{aggregated.get('total_co2e', '0')}|"
            f"{calc_id}"
        )
        provenance_hash = hashlib.sha256(seal_input.encode("utf-8")).hexdigest()

        # Merkle root computation
        merkle_root = self._compute_merkle_root(provenance_data.get("stage_hashes", []))

        logger.info(
            "[%s] Sealed: provenance=%s, merkle=%s",
            calc_id, provenance_hash[:16], merkle_root[:16],
        )

        return {
            "provenance_hash": provenance_hash,
            "merkle_root": merkle_root,
            "seal_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _compute_merkle_root(hashes: List[str]) -> str:
        """
        Compute Merkle root from a list of hashes.

        Uses SHA-256 pair-wise hashing with duplication for odd counts.

        Args:
            hashes: List of hex-encoded hash strings.

        Returns:
            Merkle root as hex-encoded SHA-256 hash.
        """
        if not hashes:
            return hashlib.sha256(b"empty").hexdigest()

        current_level = list(hashes)

        while len(current_level) > 1:
            next_level: List[str] = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # Duplicate last hash if odd number of elements
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256(
                    (left + right).encode("utf-8")
                ).hexdigest()
                next_level.append(combined)

            current_level = next_level

        return current_level[0]

    # ==========================================================================
    # PROVENANCE HELPERS
    # ==========================================================================

    def _record_provenance(
        self,
        calc_id: str,
        stage: PipelineStage,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """
        Record a provenance entry for a pipeline stage.

        Args:
            calc_id: Calculation identifier.
            stage: Pipeline stage enum.
            input_data: Input to this stage.
            output_data: Output from this stage.
        """
        input_str = json.dumps(input_data, sort_keys=True, default=str) if not isinstance(input_data, str) else input_data
        output_str = json.dumps(output_data, sort_keys=True, default=str) if not isinstance(output_data, str) else output_data

        entry: Dict[str, Any] = {
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": hashlib.sha256(input_str.encode("utf-8")).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode("utf-8")).hexdigest(),
        }

        chain = self._provenance_chains.get(calc_id)
        if chain is not None:
            # Chain link: hash of previous entry
            if chain:
                entry["chain_hash"] = hashlib.sha256(
                    json.dumps(chain[-1], sort_keys=True).encode("utf-8")
                ).hexdigest()
            chain.append(entry)

    @staticmethod
    def _elapsed_ms(start: datetime) -> float:
        """Calculate milliseconds elapsed since start."""
        return (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

    # ==========================================================================
    # LAZY ENGINE LOADING
    # ==========================================================================

    def _get_compliance_engine(self) -> Optional[Any]:
        """Get or create ComplianceCheckerEngine (lazy loading)."""
        if self._compliance_engine is None:
            try:
                from greenlang.agents.mrv.downstream_transportation.compliance_checker import (
                    get_compliance_checker,
                )
                self._compliance_engine = get_compliance_checker()
            except ImportError:
                logger.debug("ComplianceCheckerEngine not available, using inline compliance")
                self._compliance_engine = None
        return self._compliance_engine

    # ==========================================================================
    # STATUS AND MANAGEMENT
    # ==========================================================================

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status including loaded engines.

        Returns:
            Dictionary with pipeline status information.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": "GL-MRV-S3-009",
            "agent_component": "AGENT-MRV-022",
            "calculation_count": self._calculation_count,
            "engines_loaded": {
                "compliance": self._compliance_engine is not None,
            },
            "active_chains": len(self._provenance_chains),
            "reference_tables": {
                "distance_modes": len(DISTANCE_EF_BY_MODE),
                "warehouse_types": len(WAREHOUSE_EF),
                "retail_types": len(RETAIL_STORAGE_EF),
                "eeio_categories": len(SPEND_EEIO_FACTORS),
                "channel_averages": len(AVERAGE_CHANNEL_EF),
                "currencies": len(CURRENCY_TO_USD),
            },
        }

    def reset_pipeline(self) -> None:
        """
        Reset pipeline state (clear provenance chains).

        Used for testing or periodic cleanup.
        """
        with self._lock:
            self._provenance_chains.clear()
            logger.info("Pipeline state reset")

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        This forces re-initialization on next instantiation.
        """
        with cls._lock:
            cls._instance = None
            logger.info("DownstreamTransportPipelineEngine singleton reset")


# ==============================================================================
# MODULE-LEVEL ACCESSORS (thread-safe)
# ==============================================================================


def get_pipeline_engine() -> DownstreamTransportPipelineEngine:
    """
    Get singleton pipeline engine instance.

    Returns:
        DownstreamTransportPipelineEngine singleton instance.

    Example:
        >>> engine = get_pipeline_engine()
        >>> engine.get_pipeline_status()["engine_id"]
        'dto_pipeline_engine'
    """
    return DownstreamTransportPipelineEngine()


def reset_pipeline_engine() -> None:
    """
    Reset the pipeline engine singleton instance.

    Used for testing or re-initialization.

    Example:
        >>> reset_pipeline_engine()
        >>> engine = get_pipeline_engine()  # Fresh instance
    """
    DownstreamTransportPipelineEngine.reset_singleton()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "PipelineStage",
    "PipelineStatus",
    "TransportMode",
    "CalculationMethod",
    "AllocationMethod",
    "DistributionChannel",
    "IncotermCategory",
    "DISTANCE_EF_BY_MODE",
    "WAREHOUSE_EF",
    "RETAIL_STORAGE_EF",
    "SPEND_EEIO_FACTORS",
    "AVERAGE_CHANNEL_EF",
    "CURRENCY_TO_USD",
    "CPI_DEFLATORS",
    "DownstreamTransportPipelineEngine",
    "get_pipeline_engine",
    "reset_pipeline_engine",
]
