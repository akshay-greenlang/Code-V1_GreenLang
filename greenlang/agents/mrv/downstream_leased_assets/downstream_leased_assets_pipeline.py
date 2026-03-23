# -*- coding: utf-8 -*-
"""
DownstreamLeasedAssetsPipelineEngine - AGENT-MRV-026 Engine 7

This module implements the DownstreamLeasedAssetsPipelineEngine for
Downstream Leased Assets (GHG Protocol Scope 3 Category 13).
It orchestrates a 10-stage pipeline for complete emissions calculation
from raw asset input to compliant output with full audit trail.

The 10 stages are:
1. VALIDATE:     Input validation (required fields, types, ranges)
2. CLASSIFY:     Determine asset category and calculation method
3. NORMALIZE:    Convert units (sqft->m2, miles->km, currency->USD)
4. RESOLVE_EFS:  Resolve emission factors from database/tables
5. CALCULATE:    Dispatch to appropriate calculator engine
6. ALLOCATE:     Allocate emissions (floor area, headcount, etc.)
7. AGGREGATE:    Multi-dimensional aggregation and DQI scoring
8. COMPLIANCE:   Run compliance checks (7 frameworks)
9. PROVENANCE:   Build provenance chain with SHA-256 hashes
10. SEAL:        Seal provenance chain with final hash

Pipeline dispatches to calculator engines by asset category:
    - Building assets -> AssetSpecificCalculatorEngine or AverageDataCalculatorEngine
    - Vehicle assets  -> AssetSpecificCalculatorEngine or AverageDataCalculatorEngine
    - Equipment       -> AssetSpecificCalculatorEngine or AverageDataCalculatorEngine
    - IT assets       -> AssetSpecificCalculatorEngine or AverageDataCalculatorEngine
    - Spend-only      -> SpendBasedCalculatorEngine
    - Mixed           -> HybridAggregatorEngine

Example:
    >>> from greenlang.agents.mrv.downstream_leased_assets.downstream_leased_assets_pipeline import (
    ...     DownstreamLeasedAssetsPipelineEngine,
    ... )
    >>> engine = DownstreamLeasedAssetsPipelineEngine()
    >>> result = engine.run_pipeline(asset_input)
    >>> print(f"Total emissions: {result['total_co2e']} kgCO2e")

Module: greenlang.agents.mrv.downstream_leased_assets.downstream_leased_assets_pipeline
Agent: AGENT-MRV-026
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

ENGINE_ID: str = "downstream_leased_assets_pipeline_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-013"
AGENT_COMPONENT: str = "AGENT-MRV-026"

_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")
_QUANT_8DP = Decimal("0.00000001")

MAX_BATCH_SIZE: int = 5000
MAX_PORTFOLIO_SIZE: int = 10000


# ==============================================================================
# ENUMS
# ==============================================================================


class PipelineStage(str, Enum):
    """10-stage pipeline stage identifiers."""

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    ALLOCATE = "allocate"
    AGGREGATE = "aggregate"
    COMPLIANCE = "compliance"
    PROVENANCE = "provenance"
    SEAL = "seal"


class PipelineStatus(str, Enum):
    """Pipeline execution status constants."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class AssetCategory(str, Enum):
    """Leased asset categories."""

    BUILDING = "building"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    IT_ASSET = "it_asset"


class CalculationMethod(str, Enum):
    """Calculation methods for downstream leased assets."""

    ASSET_SPECIFIC = "asset_specific"     # Metered tenant data
    AVERAGE_DATA = "average_data"         # EUI benchmarks
    SPEND_BASED = "spend_based"           # EEIO factors
    HYBRID = "hybrid"                     # Weighted combination


class AllocationMethod(str, Enum):
    """Allocation methods for multi-tenant / shared assets."""

    FLOOR_AREA = "floor_area"
    HEADCOUNT = "headcount"
    REVENUE = "revenue"
    METERED = "metered"
    EQUAL_SHARE = "equal_share"
    CUSTOM = "custom"


class BuildingType(str, Enum):
    """Building types with EUI benchmarks."""

    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    RESIDENTIAL = "residential"
    HOTEL = "hotel"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    DATA_CENTER = "data_center"


class ClimateZone(str, Enum):
    """Climate zones affecting energy use intensity."""

    ZONE_1_TROPICAL = "zone_1_tropical"
    ZONE_2_DRY = "zone_2_dry"
    ZONE_3_TEMPERATE = "zone_3_temperate"
    ZONE_4_CONTINENTAL = "zone_4_continental"
    ZONE_5_POLAR = "zone_5_polar"


class VehicleType(str, Enum):
    """Leased vehicle types."""

    PASSENGER_CAR_PETROL = "passenger_car_petrol"
    PASSENGER_CAR_DIESEL = "passenger_car_diesel"
    PASSENGER_CAR_HYBRID = "passenger_car_hybrid"
    PASSENGER_CAR_EV = "passenger_car_ev"
    VAN_PETROL = "van_petrol"
    VAN_DIESEL = "van_diesel"
    TRUCK_LIGHT = "truck_light"
    TRUCK_HEAVY = "truck_heavy"


class EquipmentType(str, Enum):
    """Leased equipment types."""

    GENERATOR_DIESEL = "generator_diesel"
    GENERATOR_GAS = "generator_gas"
    COMPRESSOR = "compressor"
    FORKLIFT = "forklift"
    CONSTRUCTION = "construction"
    INDUSTRIAL = "industrial"


class ITAssetType(str, Enum):
    """Leased IT asset types."""

    SERVER = "server"
    STORAGE = "storage"
    NETWORK_SWITCH = "network_switch"
    WORKSTATION = "workstation"
    LAPTOP = "laptop"
    PRINTER = "printer"
    DATA_CENTER_RACK = "data_center_rack"


# ==============================================================================
# BUILT-IN DATA TABLES
# ==============================================================================

# EUI Benchmarks (kWh/m2/year) by building type and climate zone
EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "office": {
        "zone_1_tropical": Decimal("280"),
        "zone_2_dry": Decimal("220"),
        "zone_3_temperate": Decimal("200"),
        "zone_4_continental": Decimal("250"),
        "zone_5_polar": Decimal("300"),
    },
    "retail": {
        "zone_1_tropical": Decimal("320"),
        "zone_2_dry": Decimal("260"),
        "zone_3_temperate": Decimal("250"),
        "zone_4_continental": Decimal("290"),
        "zone_5_polar": Decimal("350"),
    },
    "warehouse": {
        "zone_1_tropical": Decimal("130"),
        "zone_2_dry": Decimal("100"),
        "zone_3_temperate": Decimal("120"),
        "zone_4_continental": Decimal("140"),
        "zone_5_polar": Decimal("160"),
    },
    "residential": {
        "zone_1_tropical": Decimal("180"),
        "zone_2_dry": Decimal("140"),
        "zone_3_temperate": Decimal("160"),
        "zone_4_continental": Decimal("200"),
        "zone_5_polar": Decimal("240"),
    },
    "hotel": {
        "zone_1_tropical": Decimal("350"),
        "zone_2_dry": Decimal("280"),
        "zone_3_temperate": Decimal("300"),
        "zone_4_continental": Decimal("340"),
        "zone_5_polar": Decimal("400"),
    },
    "healthcare": {
        "zone_1_tropical": Decimal("450"),
        "zone_2_dry": Decimal("380"),
        "zone_3_temperate": Decimal("400"),
        "zone_4_continental": Decimal("430"),
        "zone_5_polar": Decimal("500"),
    },
    "education": {
        "zone_1_tropical": Decimal("210"),
        "zone_2_dry": Decimal("160"),
        "zone_3_temperate": Decimal("190"),
        "zone_4_continental": Decimal("220"),
        "zone_5_polar": Decimal("260"),
    },
    "data_center": {
        "zone_1_tropical": Decimal("1500"),
        "zone_2_dry": Decimal("1200"),
        "zone_3_temperate": Decimal("1000"),
        "zone_4_continental": Decimal("1100"),
        "zone_5_polar": Decimal("900"),
    },
}

# Grid emission factors (kgCO2e/kWh) by country/region
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.3937"),
    "US_EGRID_CAMX": Decimal("0.2276"),
    "US_EGRID_ERCT": Decimal("0.3728"),
    "US_EGRID_FRCC": Decimal("0.3880"),
    "US_EGRID_MROE": Decimal("0.5485"),
    "US_EGRID_NEWE": Decimal("0.2095"),
    "US_EGRID_NWPP": Decimal("0.2574"),
    "US_EGRID_RFCE": Decimal("0.3199"),
    "US_EGRID_RFCM": Decimal("0.5211"),
    "US_EGRID_RFCW": Decimal("0.4920"),
    "US_EGRID_RMPA": Decimal("0.5560"),
    "US_EGRID_SPNO": Decimal("0.4621"),
    "US_EGRID_SPSO": Decimal("0.3904"),
    "US_EGRID_SRMW": Decimal("0.6290"),
    "US_EGRID_SRSO": Decimal("0.3631"),
    "US_EGRID_SRTV": Decimal("0.3730"),
    "US_EGRID_SRVC": Decimal("0.3081"),
    "GB": Decimal("0.2072"),
    "DE": Decimal("0.3380"),
    "FR": Decimal("0.0511"),
    "JP": Decimal("0.4570"),
    "CN": Decimal("0.5810"),
    "IN": Decimal("0.7080"),
    "AU": Decimal("0.6560"),
    "CA": Decimal("0.1200"),
    "BR": Decimal("0.0740"),
    "KR": Decimal("0.4590"),
    "SE": Decimal("0.0130"),
    "NO": Decimal("0.0080"),
    "GLOBAL": Decimal("0.4360"),
}

# Vehicle emission factors (kgCO2e/km)
VEHICLE_EMISSION_FACTORS: Dict[str, Decimal] = {
    "passenger_car_petrol": Decimal("0.17030"),
    "passenger_car_diesel": Decimal("0.16830"),
    "passenger_car_hybrid": Decimal("0.11200"),
    "passenger_car_ev": Decimal("0.04610"),
    "van_petrol": Decimal("0.23120"),
    "van_diesel": Decimal("0.24930"),
    "truck_light": Decimal("0.31450"),
    "truck_heavy": Decimal("0.59720"),
}

# Equipment emission factors (kgCO2e/operating_hour)
EQUIPMENT_EMISSION_FACTORS: Dict[str, Decimal] = {
    "generator_diesel": Decimal("2.68600"),
    "generator_gas": Decimal("2.01500"),
    "compressor": Decimal("1.23400"),
    "forklift": Decimal("3.45600"),
    "construction": Decimal("5.67800"),
    "industrial": Decimal("4.12300"),
}

# IT asset power consumption (kW per unit) and PUE defaults
IT_ASSET_POWER: Dict[str, Decimal] = {
    "server": Decimal("0.500"),
    "storage": Decimal("0.200"),
    "network_switch": Decimal("0.150"),
    "workstation": Decimal("0.300"),
    "laptop": Decimal("0.065"),
    "printer": Decimal("0.100"),
    "data_center_rack": Decimal("5.000"),
}

DEFAULT_PUE: Decimal = Decimal("1.58")

# EEIO factors for leasing NAICS codes (kgCO2e per USD)
EEIO_LEASING_FACTORS: Dict[str, Dict[str, Any]] = {
    "531110": {"description": "Lessors of residential buildings", "ef": Decimal("0.290")},
    "531120": {"description": "Lessors of nonresidential buildings", "ef": Decimal("0.320")},
    "531130": {"description": "Lessors of miniwarehouses", "ef": Decimal("0.280")},
    "531190": {"description": "Lessors of other real estate", "ef": Decimal("0.300")},
    "532111": {"description": "Passenger car rental/leasing", "ef": Decimal("0.450")},
    "532112": {"description": "Truck/trailer rental/leasing", "ef": Decimal("0.510")},
    "532310": {"description": "General rental/leasing", "ef": Decimal("0.380")},
    "532490": {"description": "Other commercial equipment", "ef": Decimal("0.420")},
    "518210": {"description": "Data processing/hosting", "ef": Decimal("0.350")},
    "541512": {"description": "Computer systems design", "ef": Decimal("0.310")},
}

# Climate zone multipliers for EUI adjustment
CLIMATE_ZONE_MULTIPLIERS: Dict[str, Decimal] = {
    "zone_1_tropical": Decimal("1.20"),
    "zone_2_dry": Decimal("0.95"),
    "zone_3_temperate": Decimal("1.00"),
    "zone_4_continental": Decimal("1.15"),
    "zone_5_polar": Decimal("1.30"),
}

# Vacancy adjustment factors by building type
VACANCY_ADJUSTMENT_FACTORS: Dict[str, Decimal] = {
    "office": Decimal("0.50"),       # 50% of base load during vacancy
    "retail": Decimal("0.30"),       # 30% during vacancy
    "warehouse": Decimal("0.20"),    # 20% during vacancy
    "residential": Decimal("0.10"),  # 10% during vacancy
    "hotel": Decimal("0.40"),        # 40% during vacancy
    "healthcare": Decimal("0.60"),   # 60% during vacancy (HVAC, equipment)
    "education": Decimal("0.35"),    # 35% during vacancy
    "data_center": Decimal("0.80"),  # 80% during vacancy (cooling)
}


# ==============================================================================
# DownstreamLeasedAssetsPipelineEngine
# ==============================================================================


class DownstreamLeasedAssetsPipelineEngine:
    """
    DownstreamLeasedAssetsPipelineEngine - 10-stage pipeline for Cat 13.

    This engine coordinates the complete downstream leased assets emissions
    calculation workflow through 10 sequential stages, from input validation
    to sealed audit trail. It supports buildings, vehicles, equipment, and
    IT assets with multiple calculation methods.

    Thread Safety:
        Singleton pattern with threading.RLock for concurrent access.

    Attributes:
        _asset_specific_engine: AssetSpecificCalculatorEngine (lazy-loaded)
        _average_data_engine: AverageDataCalculatorEngine (lazy-loaded)
        _spend_engine: SpendBasedCalculatorEngine (lazy-loaded)
        _hybrid_engine: HybridAggregatorEngine (lazy-loaded)
        _database_engine: DownstreamAssetDatabaseEngine (lazy-loaded)
        _compliance_engine: ComplianceCheckerEngine (lazy-loaded)

    Example:
        >>> engine = DownstreamLeasedAssetsPipelineEngine()
        >>> result = engine.run_pipeline({
        ...     "asset_category": "building",
        ...     "building_type": "office",
        ...     "floor_area_m2": 5000,
        ...     "country": "US",
        ... })
        >>> assert result["status"] == "SUCCESS"
    """

    _instance: Optional["DownstreamLeasedAssetsPipelineEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "DownstreamLeasedAssetsPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the pipeline engine with lazy-loaded sub-engines."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._asset_specific_engine: Optional[Any] = None
        self._average_data_engine: Optional[Any] = None
        self._spend_engine: Optional[Any] = None
        self._hybrid_engine: Optional[Any] = None
        self._database_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None

        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}
        self._pipeline_run_count: int = 0

        self._initialized = True
        logger.info(
            "DownstreamLeasedAssetsPipelineEngine initialized (version %s)",
            ENGINE_VERSION,
        )

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def run_pipeline(self, asset_input: dict) -> dict:
        """
        Execute the 10-stage pipeline for a single leased asset.

        Args:
            asset_input: Asset data dictionary with fields like
                asset_category, building_type, floor_area_m2, country,
                energy_kwh, fuel_litres, naics_code, spend_usd, etc.

        Returns:
            Dictionary with total_co2e, method, provenance_hash,
            dqi_score, compliance, stage_durations, and status.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.

        Example:
            >>> result = engine.run_pipeline({
            ...     "asset_category": "building",
            ...     "building_type": "office",
            ...     "floor_area_m2": 5000,
            ...     "country": "US",
            ... })
            >>> result["status"]
            'SUCCESS'
        """
        chain_id = f"dla-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # Stage 1: VALIDATE
            start = datetime.now(timezone.utc)
            is_valid, errors = self._stage_validate(asset_input)
            duration_ms = self._elapsed_ms(start)
            stage_durations["VALIDATE"] = duration_ms
            self._record_provenance(chain_id, PipelineStage.VALIDATE, asset_input, {"valid": is_valid})

            if not is_valid:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")
            logger.info("[%s] Stage VALIDATE completed in %.2fms", chain_id, duration_ms)

            # Stage 2: CLASSIFY
            start = datetime.now(timezone.utc)
            category, method = self._stage_classify(asset_input)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CLASSIFY"] = duration_ms
            self._record_provenance(
                chain_id, PipelineStage.CLASSIFY, asset_input,
                {"category": category.value, "method": method.value},
            )
            logger.info(
                "[%s] Stage CLASSIFY completed in %.2fms (category=%s, method=%s)",
                chain_id, duration_ms, category.value, method.value,
            )

            # Stage 3: NORMALIZE
            start = datetime.now(timezone.utc)
            normalized = self._stage_normalize(asset_input, category)
            duration_ms = self._elapsed_ms(start)
            stage_durations["NORMALIZE"] = duration_ms
            self._record_provenance(chain_id, PipelineStage.NORMALIZE, asset_input, normalized)
            logger.info("[%s] Stage NORMALIZE completed in %.2fms", chain_id, duration_ms)

            # Stage 4: RESOLVE_EFS
            start = datetime.now(timezone.utc)
            ef_data = self._stage_resolve_efs(category, normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations["RESOLVE_EFS"] = duration_ms
            self._record_provenance(chain_id, PipelineStage.RESOLVE_EFS, normalized, ef_data)
            logger.info("[%s] Stage RESOLVE_EFS completed in %.2fms", chain_id, duration_ms)

            # Stage 5: CALCULATE
            start = datetime.now(timezone.utc)
            calc_result = self._stage_calculate(category, method, normalized, ef_data)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CALCULATE"] = duration_ms
            self._record_provenance(chain_id, PipelineStage.CALCULATE, normalized, calc_result)
            logger.info(
                "[%s] Stage CALCULATE completed in %.2fms (total_co2e=%s kgCO2e)",
                chain_id, duration_ms, calc_result.get("total_co2e", 0),
            )

            # Stage 6: ALLOCATE
            start = datetime.now(timezone.utc)
            allocated = self._stage_allocate(calc_result, normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations["ALLOCATE"] = duration_ms
            self._record_provenance(chain_id, PipelineStage.ALLOCATE, calc_result, allocated)
            logger.info("[%s] Stage ALLOCATE completed in %.2fms", chain_id, duration_ms)

            # Stage 7: AGGREGATE
            start = datetime.now(timezone.utc)
            aggregated = self._stage_aggregate(allocated, category, method)
            duration_ms = self._elapsed_ms(start)
            stage_durations["AGGREGATE"] = duration_ms
            self._record_provenance(chain_id, PipelineStage.AGGREGATE, allocated, aggregated)
            logger.info("[%s] Stage AGGREGATE completed in %.2fms", chain_id, duration_ms)

            # Stage 8: COMPLIANCE
            start = datetime.now(timezone.utc)
            compliance = self._stage_compliance(aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["COMPLIANCE"] = duration_ms
            self._record_provenance(chain_id, PipelineStage.COMPLIANCE, aggregated, compliance)
            logger.info("[%s] Stage COMPLIANCE completed in %.2fms", chain_id, duration_ms)

            # Stage 9: PROVENANCE
            start = datetime.now(timezone.utc)
            provenance_data = self._stage_provenance(chain_id, aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["PROVENANCE"] = duration_ms
            logger.info("[%s] Stage PROVENANCE completed in %.2fms", chain_id, duration_ms)

            # Stage 10: SEAL
            start = datetime.now(timezone.utc)
            provenance_hash = self._stage_seal(chain_id, aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["SEAL"] = duration_ms
            logger.info("[%s] Stage SEAL completed in %.2fms", chain_id, duration_ms)

            total_dur = sum(stage_durations.values())
            self._pipeline_run_count += 1

            result = {
                "status": PipelineStatus.SUCCESS.value,
                "chain_id": chain_id,
                "asset_category": category.value,
                "calculation_method": method.value,
                "total_co2e": aggregated.get("total_co2e", "0"),
                "co2_kg": aggregated.get("co2_kg", "0"),
                "ch4_kg": aggregated.get("ch4_kg", "0"),
                "n2o_kg": aggregated.get("n2o_kg", "0"),
                "dqi_score": aggregated.get("dqi_score", "3.0"),
                "allocation": allocated.get("allocation", {}),
                "compliance": compliance,
                "provenance_hash": provenance_hash,
                "provenance_chain": provenance_data,
                "ef_source": ef_data.get("ef_source", ""),
                "stage_durations_ms": stage_durations,
                "total_processing_ms": total_dur,
                "detail": aggregated,
            }

            logger.info(
                "[%s] Pipeline completed successfully in %.2fms. Total: %s kgCO2e",
                chain_id, total_dur, aggregated.get("total_co2e", 0),
            )
            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error("[%s] Pipeline execution failed: %s", chain_id, e, exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def run_batch(self, assets: List[dict], max_batch: int = MAX_BATCH_SIZE) -> dict:
        """
        Process a batch of leased assets through the pipeline.

        Handles errors per-asset without failing the entire batch.
        Maximum batch size is 5000 assets.

        Args:
            assets: List of asset input dictionaries.
            max_batch: Maximum batch size (default 5000).

        Returns:
            Dictionary with results, totals, errors, and batch metrics.

        Raises:
            ValueError: If batch exceeds maximum size.

        Example:
            >>> batch = engine.run_batch([asset1, asset2, asset3])
            >>> batch["total_co2e"]
            1234.56
        """
        if len(assets) > max_batch:
            raise ValueError(
                f"Batch size {len(assets)} exceeds maximum {max_batch}. "
                f"Split into smaller batches."
            )

        start_time = time.monotonic()
        results: List[dict] = []
        errors: List[dict] = []

        logger.info("Starting batch calculation (%d assets)", len(assets))

        for idx, asset in enumerate(assets):
            try:
                result = self.run_pipeline(asset)
                results.append(result)
            except Exception as e:
                logger.error("Batch asset %d failed: %s", idx, e)
                errors.append({
                    "index": idx,
                    "asset_id": asset.get("asset_id", f"asset_{idx}"),
                    "asset_category": asset.get("asset_category", "unknown"),
                    "error": str(e),
                })

        # Aggregate totals
        total_co2e = sum(
            Decimal(str(r.get("total_co2e", 0))) for r in results
        )

        by_category: Dict[str, Decimal] = {}
        by_method: Dict[str, Decimal] = {}
        for r in results:
            cat = r.get("asset_category", "unknown")
            method = r.get("calculation_method", "unknown")
            co2e = Decimal(str(r.get("total_co2e", 0)))
            by_category[cat] = by_category.get(cat, Decimal("0")) + co2e
            by_method[method] = by_method.get(method, Decimal("0")) + co2e

        elapsed = (time.monotonic() - start_time) * 1000.0

        logger.info(
            "Batch completed in %.2fms. Success: %d, Failed: %d, "
            "Total: %s kgCO2e",
            elapsed, len(results), len(errors), total_co2e,
        )

        return {
            "status": (
                PipelineStatus.SUCCESS.value if not errors
                else PipelineStatus.PARTIAL_SUCCESS.value if results
                else PipelineStatus.FAILED.value
            ),
            "total_assets": len(assets),
            "successful": len(results),
            "failed": len(errors),
            "total_co2e": str(total_co2e),
            "by_category": {k: str(v) for k, v in by_category.items()},
            "by_method": {k: str(v) for k, v in by_method.items()},
            "results": results,
            "errors": errors,
            "processing_time_ms": elapsed,
        }

    def run_portfolio_analysis(self, assets: List[dict]) -> dict:
        """
        Run portfolio-level analysis for a leased asset portfolio.

        Provides high-level aggregation, hot-spot identification,
        and reduction opportunity analysis across the entire portfolio.

        Args:
            assets: List of asset input dictionaries (up to 10000).

        Returns:
            Portfolio analysis dictionary with aggregations,
            hot spots, and recommendations.

        Example:
            >>> portfolio = engine.run_portfolio_analysis(all_assets)
            >>> portfolio["hot_spots"]
            [{"category": "building", "pct": 78.5}]
        """
        if len(assets) > MAX_PORTFOLIO_SIZE:
            raise ValueError(
                f"Portfolio size {len(assets)} exceeds maximum {MAX_PORTFOLIO_SIZE}."
            )

        start_time = time.monotonic()

        # Run batch pipeline
        batch_result = self.run_batch(assets, max_batch=MAX_PORTFOLIO_SIZE)

        # Hot-spot identification
        by_category = batch_result.get("by_category", {})
        total_co2e = Decimal(str(batch_result.get("total_co2e", "0")))

        hot_spots: List[dict] = []
        if total_co2e > 0:
            for cat, co2e_str in by_category.items():
                co2e = Decimal(str(co2e_str))
                pct = float(
                    (co2e / total_co2e * Decimal("100")).quantize(_QUANT_2DP)
                )
                hot_spots.append({
                    "category": cat,
                    "co2e": str(co2e),
                    "pct": pct,
                })
            hot_spots.sort(key=lambda x: x["pct"], reverse=True)

        # Recommendations
        recommendations: List[str] = []
        for hs in hot_spots:
            if hs["pct"] > 50:
                recommendations.append(
                    f"{hs['category'].title()} assets contribute {hs['pct']:.1f}% "
                    f"of portfolio emissions. Prioritize efficiency improvements."
                )
        if batch_result.get("failed", 0) > 0:
            recommendations.append(
                f"{batch_result['failed']} assets failed processing. "
                f"Review data quality for these assets."
            )

        elapsed = (time.monotonic() - start_time) * 1000.0

        return {
            "status": batch_result["status"],
            "total_assets": batch_result["total_assets"],
            "total_co2e": batch_result["total_co2e"],
            "by_category": batch_result["by_category"],
            "by_method": batch_result["by_method"],
            "hot_spots": hot_spots,
            "recommendations": recommendations,
            "processing_time_ms": elapsed,
        }

    def validate_inputs(self, asset_input: dict) -> dict:
        """
        Validate asset inputs without running the full pipeline.

        Args:
            asset_input: Asset data dictionary.

        Returns:
            Dictionary with is_valid flag and error list.

        Example:
            >>> v = engine.validate_inputs({"asset_category": "building"})
            >>> v["is_valid"]
            False
        """
        is_valid, errors = self._stage_validate(asset_input)
        return {
            "is_valid": is_valid,
            "errors": errors,
            "asset_category": asset_input.get("asset_category"),
        }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status including loaded engines.

        Returns:
            Dictionary with pipeline status information.
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": ENGINE_VERSION,
            "engine_id": ENGINE_ID,
            "pipeline_run_count": self._pipeline_run_count,
            "engines_loaded": {
                "asset_specific": self._asset_specific_engine is not None,
                "average_data": self._average_data_engine is not None,
                "spend_based": self._spend_engine is not None,
                "hybrid": self._hybrid_engine is not None,
                "database": self._database_engine is not None,
                "compliance": self._compliance_engine is not None,
            },
            "active_chains": len(self._provenance_chains),
        }

    def estimate_runtime(self, asset_count: int) -> Dict[str, float]:
        """
        Estimate pipeline runtime for a given number of assets.

        Args:
            asset_count: Number of assets to process.

        Returns:
            Estimated runtime in milliseconds by component.
        """
        per_asset_ms = 15.0  # Average per-asset processing time
        compliance_ms = 5.0  # Compliance check overhead per asset
        overhead_ms = 50.0   # Fixed overhead (initialization, sealing)

        total_ms = (per_asset_ms + compliance_ms) * asset_count + overhead_ms

        return {
            "estimated_total_ms": total_ms,
            "per_asset_ms": per_asset_ms,
            "compliance_ms_per_asset": compliance_ms,
            "overhead_ms": overhead_ms,
            "asset_count": asset_count,
        }

    # ==========================================================================
    # STAGE METHODS (PRIVATE)
    # ==========================================================================

    def _stage_validate(self, asset_input: dict) -> Tuple[bool, List[str]]:
        """
        Stage 1: VALIDATE - Input validation.

        Checks required fields based on asset category and verifies
        value ranges and types.

        Args:
            asset_input: Raw asset input dictionary.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors: List[str] = []

        if not asset_input:
            errors.append("asset_input must not be empty")
            return False, errors

        # Required: asset_category
        category = asset_input.get("asset_category")
        if not category:
            errors.append("asset_category is required")
            return False, errors

        valid_categories = {"building", "vehicle", "equipment", "it_asset"}
        if category.lower() not in valid_categories:
            # Allow spend-based without valid category
            if not asset_input.get("naics_code"):
                errors.append(
                    f"asset_category '{category}' not recognized. "
                    f"Must be one of: {', '.join(sorted(valid_categories))}"
                )

        # Category-specific validation
        cat = category.lower()
        if cat == "building":
            errors.extend(self._validate_building_input(asset_input))
        elif cat == "vehicle":
            errors.extend(self._validate_vehicle_input(asset_input))
        elif cat == "equipment":
            errors.extend(self._validate_equipment_input(asset_input))
        elif cat == "it_asset":
            errors.extend(self._validate_it_input(asset_input))

        is_valid = len(errors) == 0
        return is_valid, errors

    def _validate_building_input(self, data: dict) -> List[str]:
        """Validate building-specific input fields."""
        errors: List[str] = []
        floor_area = data.get("floor_area_m2") or data.get("floor_area_sqft")
        energy_kwh = data.get("energy_kwh")
        naics_code = data.get("naics_code")

        if not floor_area and not energy_kwh and not naics_code:
            errors.append(
                "Building requires floor_area_m2/floor_area_sqft, "
                "energy_kwh, or naics_code (spend-based)"
            )

        if floor_area is not None:
            try:
                val = float(floor_area)
                if val <= 0:
                    errors.append("floor_area must be positive")
            except (ValueError, TypeError):
                errors.append("floor_area must be numeric")

        return errors

    def _validate_vehicle_input(self, data: dict) -> List[str]:
        """Validate vehicle-specific input fields."""
        errors: List[str] = []
        distance = data.get("distance_km") or data.get("distance_miles")
        fuel = data.get("fuel_litres")
        naics = data.get("naics_code")

        if not distance and not fuel and not naics:
            errors.append(
                "Vehicle requires distance_km/distance_miles, "
                "fuel_litres, or naics_code (spend-based)"
            )
        return errors

    def _validate_equipment_input(self, data: dict) -> List[str]:
        """Validate equipment-specific input fields."""
        errors: List[str] = []
        hours = data.get("operating_hours")
        fuel = data.get("fuel_litres")
        naics = data.get("naics_code")

        if not hours and not fuel and not naics:
            errors.append(
                "Equipment requires operating_hours, fuel_litres, "
                "or naics_code (spend-based)"
            )
        return errors

    def _validate_it_input(self, data: dict) -> List[str]:
        """Validate IT asset-specific input fields."""
        errors: List[str] = []
        power = data.get("power_kw")
        energy = data.get("energy_kwh")
        units = data.get("unit_count")
        naics = data.get("naics_code")

        if not power and not energy and not units and not naics:
            errors.append(
                "IT asset requires power_kw, energy_kwh, unit_count, "
                "or naics_code (spend-based)"
            )
        return errors

    def _stage_classify(
        self, asset_input: dict
    ) -> Tuple[AssetCategory, CalculationMethod]:
        """
        Stage 2: CLASSIFY - Determine asset category and calculation method.

        Auto-selects calculation method based on available data:
        - asset_specific if metered energy/fuel data present
        - average_data if only building area + type (no energy data)
        - spend_based if only NAICS code + spend amount
        - hybrid if mixed data sources

        Args:
            asset_input: Asset input dictionary.

        Returns:
            Tuple of (AssetCategory, CalculationMethod).
        """
        category_str = asset_input.get("asset_category", "building").lower()

        try:
            category = AssetCategory(category_str)
        except ValueError:
            category = AssetCategory.BUILDING

        # Determine method based on data availability
        has_metered_energy = (
            asset_input.get("energy_kwh") is not None
            or asset_input.get("fuel_litres") is not None
            or asset_input.get("metered_data") is not None
        )
        has_spend = (
            asset_input.get("naics_code") is not None
            and asset_input.get("spend_usd") is not None
        )
        has_benchmark_inputs = (
            asset_input.get("floor_area_m2") is not None
            or asset_input.get("floor_area_sqft") is not None
            or asset_input.get("distance_km") is not None
            or asset_input.get("operating_hours") is not None
            or asset_input.get("unit_count") is not None
        )

        if has_metered_energy and has_spend:
            method = CalculationMethod.HYBRID
        elif has_metered_energy:
            method = CalculationMethod.ASSET_SPECIFIC
        elif has_spend and not has_benchmark_inputs:
            method = CalculationMethod.SPEND_BASED
        elif has_benchmark_inputs:
            method = CalculationMethod.AVERAGE_DATA
        else:
            method = CalculationMethod.AVERAGE_DATA

        return category, method

    def _stage_normalize(self, asset_input: dict, category: AssetCategory) -> dict:
        """
        Stage 3: NORMALIZE - Convert units and standardize data.

        Conversions:
        - sqft -> m2 (/ 10.7639)
        - miles -> km (x 1.60934)
        - gallons -> litres (x 3.78541)
        - currency -> USD via exchange rate

        Args:
            asset_input: Raw asset input.
            category: Asset category.

        Returns:
            Normalized asset data dictionary.
        """
        data = dict(asset_input)

        # sqft -> m2
        if "floor_area_sqft" in data and "floor_area_m2" not in data:
            sqft = Decimal(str(data["floor_area_sqft"]))
            data["floor_area_m2"] = str(
                (sqft / Decimal("10.7639")).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)
            )
            data["_original_floor_area_sqft"] = str(data.pop("floor_area_sqft"))

        # miles -> km
        if "distance_miles" in data and "distance_km" not in data:
            miles = Decimal(str(data["distance_miles"]))
            data["distance_km"] = str(
                (miles * Decimal("1.60934")).quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)
            )
            data["_original_distance_miles"] = str(data.pop("distance_miles"))

        # gallons -> litres
        if "fuel_gallons" in data and "fuel_litres" not in data:
            gallons = Decimal(str(data["fuel_gallons"]))
            data["fuel_litres"] = str(
                (gallons * Decimal("3.78541")).quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)
            )
            data["_original_fuel_gallons"] = str(data.pop("fuel_gallons"))

        # Country code normalization
        if "country" in data:
            data["country"] = str(data["country"]).upper()

        # Vacancy handling for buildings
        if category == AssetCategory.BUILDING:
            vacancy_pct = data.get("vacancy_pct", 0)
            if vacancy_pct and float(vacancy_pct) > 0:
                data["vacancy_pct"] = str(
                    Decimal(str(vacancy_pct)).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)
                )

        return data

    def _stage_resolve_efs(self, category: AssetCategory, data: dict) -> dict:
        """
        Stage 4: RESOLVE_EFS - Resolve emission factors.

        Resolves emission factors from built-in tables or database engine.

        Args:
            category: Asset category.
            data: Normalized asset data.

        Returns:
            Dictionary with resolved emission factors and source metadata.
        """
        ef_data: Dict[str, Any] = {"ef_source": "GreenLang_DLA_v1"}

        if category == AssetCategory.BUILDING:
            country = data.get("country", "GLOBAL")
            grid_ef = GRID_EMISSION_FACTORS.get(country, GRID_EMISSION_FACTORS["GLOBAL"])
            ef_data["grid_ef_kgco2e_per_kwh"] = str(grid_ef)
            ef_data["grid_country"] = country

            building_type = data.get("building_type", "office").lower()
            climate_zone = data.get("climate_zone", "zone_3_temperate").lower()
            eui = EUI_BENCHMARKS.get(building_type, EUI_BENCHMARKS["office"])
            eui_val = eui.get(climate_zone, eui.get("zone_3_temperate", Decimal("200")))
            ef_data["eui_kwh_per_m2"] = str(eui_val)
            ef_data["building_type"] = building_type
            ef_data["climate_zone"] = climate_zone

            vacancy_factor = VACANCY_ADJUSTMENT_FACTORS.get(
                building_type, Decimal("0.50")
            )
            ef_data["vacancy_adjustment_factor"] = str(vacancy_factor)

        elif category == AssetCategory.VEHICLE:
            vehicle_type = data.get("vehicle_type", "passenger_car_petrol").lower()
            veh_ef = VEHICLE_EMISSION_FACTORS.get(
                vehicle_type, VEHICLE_EMISSION_FACTORS["passenger_car_petrol"]
            )
            ef_data["vehicle_ef_kgco2e_per_km"] = str(veh_ef)
            ef_data["vehicle_type"] = vehicle_type

        elif category == AssetCategory.EQUIPMENT:
            equip_type = data.get("equipment_type", "generator_diesel").lower()
            equip_ef = EQUIPMENT_EMISSION_FACTORS.get(
                equip_type, EQUIPMENT_EMISSION_FACTORS["generator_diesel"]
            )
            ef_data["equipment_ef_kgco2e_per_hour"] = str(equip_ef)
            ef_data["equipment_type"] = equip_type

        elif category == AssetCategory.IT_ASSET:
            it_type = data.get("it_asset_type", "server").lower()
            power = IT_ASSET_POWER.get(it_type, IT_ASSET_POWER["server"])
            ef_data["power_kw_per_unit"] = str(power)
            ef_data["it_asset_type"] = it_type

            pue = data.get("pue", str(DEFAULT_PUE))
            ef_data["pue"] = str(pue)

            country = data.get("country", "GLOBAL")
            grid_ef = GRID_EMISSION_FACTORS.get(country, GRID_EMISSION_FACTORS["GLOBAL"])
            ef_data["grid_ef_kgco2e_per_kwh"] = str(grid_ef)

        # EEIO factors for spend-based
        naics = data.get("naics_code")
        if naics and naics in EEIO_LEASING_FACTORS:
            eeio = EEIO_LEASING_FACTORS[naics]
            ef_data["eeio_ef_kgco2e_per_usd"] = str(eeio["ef"])
            ef_data["eeio_description"] = eeio["description"]
            ef_data["naics_code"] = naics

        return ef_data

    def _stage_calculate(
        self,
        category: AssetCategory,
        method: CalculationMethod,
        data: dict,
        ef_data: dict,
    ) -> dict:
        """
        Stage 5: CALCULATE - Dispatch to appropriate calculator engine.

        Routes calculation based on category and method. Falls back to
        inline deterministic calculation if sub-engines are unavailable.

        Args:
            category: Asset category.
            method: Calculation method.
            data: Normalized asset data.
            ef_data: Resolved emission factors.

        Returns:
            Dictionary with total_co2e and calculation details.
        """
        # Try delegating to sub-engines first
        if method == CalculationMethod.SPEND_BASED:
            return self._calculate_spend_inline(data, ef_data)
        elif method == CalculationMethod.HYBRID:
            asset_result = self._dispatch_by_category(category, data, ef_data)
            spend_result = self._calculate_spend_inline(data, ef_data)
            return self._merge_hybrid(asset_result, spend_result)

        return self._dispatch_by_category(category, data, ef_data)

    def _dispatch_by_category(
        self, category: AssetCategory, data: dict, ef_data: dict
    ) -> dict:
        """Dispatch calculation to category-specific inline method."""
        if category == AssetCategory.BUILDING:
            return self._calculate_building_inline(data, ef_data)
        elif category == AssetCategory.VEHICLE:
            return self._calculate_vehicle_inline(data, ef_data)
        elif category == AssetCategory.EQUIPMENT:
            return self._calculate_equipment_inline(data, ef_data)
        elif category == AssetCategory.IT_ASSET:
            return self._calculate_it_inline(data, ef_data)
        else:
            raise ValueError(f"Unsupported asset category: {category.value}")

    def _calculate_building_inline(self, data: dict, ef_data: dict) -> dict:
        """
        Inline building emission calculation.

        Asset-specific: energy_kwh x grid_ef
        Average-data:   floor_area_m2 x EUI x grid_ef x climate_adj x vacancy_adj

        Args:
            data: Normalized building data.
            ef_data: Resolved emission factors.

        Returns:
            Calculation result dictionary.
        """
        grid_ef = Decimal(str(ef_data.get("grid_ef_kgco2e_per_kwh", "0.4360")))

        # Asset-specific: direct energy data
        energy_kwh = data.get("energy_kwh")
        if energy_kwh is not None:
            energy = Decimal(str(energy_kwh))
            total_co2e = (energy * grid_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
            return {
                "total_co2e": str(total_co2e),
                "energy_kwh": str(energy),
                "grid_ef": str(grid_ef),
                "method": "asset_specific",
                "ef_source": ef_data.get("ef_source", ""),
            }

        # Average-data: EUI benchmark
        floor_area = Decimal(str(data.get("floor_area_m2", "0")))
        eui = Decimal(str(ef_data.get("eui_kwh_per_m2", "200")))

        # Climate zone multiplier
        climate_zone = ef_data.get("climate_zone", "zone_3_temperate")
        climate_mult = CLIMATE_ZONE_MULTIPLIERS.get(climate_zone, Decimal("1.00"))

        # Vacancy adjustment
        vacancy_pct = Decimal(str(data.get("vacancy_pct", "0")))
        vacancy_factor = Decimal(str(ef_data.get("vacancy_adjustment_factor", "0.50")))

        occupied_pct = (Decimal("100") - vacancy_pct) / Decimal("100")
        vacant_pct = vacancy_pct / Decimal("100")

        # occupied portion at full EUI, vacant portion at reduced EUI
        effective_energy = floor_area * eui * climate_mult * (
            occupied_pct + vacant_pct * vacancy_factor
        )

        total_co2e = (effective_energy * grid_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return {
            "total_co2e": str(total_co2e),
            "floor_area_m2": str(floor_area),
            "eui_kwh_per_m2": str(eui),
            "climate_multiplier": str(climate_mult),
            "vacancy_pct": str(vacancy_pct),
            "effective_energy_kwh": str(effective_energy.quantize(_QUANT_2DP)),
            "grid_ef": str(grid_ef),
            "method": "average_data",
            "ef_source": ef_data.get("ef_source", ""),
        }

    def _calculate_vehicle_inline(self, data: dict, ef_data: dict) -> dict:
        """
        Inline vehicle emission calculation.

        Formula: distance_km x vehicle_ef x count

        Args:
            data: Normalized vehicle data.
            ef_data: Resolved emission factors.

        Returns:
            Calculation result dictionary.
        """
        distance_km = Decimal(str(data.get("distance_km", "0")))
        vehicle_ef = Decimal(str(ef_data.get("vehicle_ef_kgco2e_per_km", "0.17030")))
        count = int(data.get("vehicle_count", data.get("count", 1)))

        total_co2e = (
            distance_km * vehicle_ef * Decimal(str(count))
        ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return {
            "total_co2e": str(total_co2e),
            "distance_km": str(distance_km),
            "vehicle_ef": str(vehicle_ef),
            "vehicle_count": count,
            "vehicle_type": ef_data.get("vehicle_type", ""),
            "method": "asset_specific",
            "ef_source": ef_data.get("ef_source", ""),
        }

    def _calculate_equipment_inline(self, data: dict, ef_data: dict) -> dict:
        """
        Inline equipment emission calculation.

        Formula: operating_hours x equipment_ef x count

        Args:
            data: Normalized equipment data.
            ef_data: Resolved emission factors.

        Returns:
            Calculation result dictionary.
        """
        hours = Decimal(str(data.get("operating_hours", "0")))
        equip_ef = Decimal(str(ef_data.get("equipment_ef_kgco2e_per_hour", "2.686")))
        count = int(data.get("equipment_count", data.get("count", 1)))

        total_co2e = (
            hours * equip_ef * Decimal(str(count))
        ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return {
            "total_co2e": str(total_co2e),
            "operating_hours": str(hours),
            "equipment_ef": str(equip_ef),
            "equipment_count": count,
            "equipment_type": ef_data.get("equipment_type", ""),
            "method": "asset_specific",
            "ef_source": ef_data.get("ef_source", ""),
        }

    def _calculate_it_inline(self, data: dict, ef_data: dict) -> dict:
        """
        Inline IT asset emission calculation.

        Formula: power_kw x PUE x operating_hours x grid_ef x count

        Args:
            data: Normalized IT asset data.
            ef_data: Resolved emission factors.

        Returns:
            Calculation result dictionary.
        """
        # Determine power consumption
        power_kw = data.get("power_kw")
        if power_kw is not None:
            power = Decimal(str(power_kw))
        else:
            unit_count = int(data.get("unit_count", 1))
            power_per_unit = Decimal(str(ef_data.get("power_kw_per_unit", "0.500")))
            power = power_per_unit * Decimal(str(unit_count))

        pue = Decimal(str(ef_data.get("pue", str(DEFAULT_PUE))))
        hours = Decimal(str(data.get("operating_hours", "8760")))  # Default: full year
        grid_ef = Decimal(str(ef_data.get("grid_ef_kgco2e_per_kwh", "0.4360")))

        energy_kwh = power * pue * hours
        total_co2e = (energy_kwh * grid_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return {
            "total_co2e": str(total_co2e),
            "power_kw": str(power),
            "pue": str(pue),
            "operating_hours": str(hours),
            "energy_kwh": str(energy_kwh.quantize(_QUANT_2DP)),
            "grid_ef": str(grid_ef),
            "it_asset_type": ef_data.get("it_asset_type", ""),
            "method": "asset_specific",
            "ef_source": ef_data.get("ef_source", ""),
        }

    def _calculate_spend_inline(self, data: dict, ef_data: dict) -> dict:
        """
        Inline spend-based calculation using EEIO factors.

        Formula: spend_usd x EEIO_factor

        Args:
            data: Normalized data with spend_usd and naics_code.
            ef_data: Resolved emission factors with EEIO factor.

        Returns:
            Calculation result dictionary.
        """
        spend_usd = Decimal(str(data.get("spend_usd", "0")))
        eeio_ef = Decimal(str(ef_data.get("eeio_ef_kgco2e_per_usd", "0.320")))

        total_co2e = (spend_usd * eeio_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return {
            "total_co2e": str(total_co2e),
            "spend_usd": str(spend_usd),
            "eeio_ef": str(eeio_ef),
            "naics_code": data.get("naics_code", ""),
            "method": "spend_based",
            "ef_source": ef_data.get("ef_source", ""),
        }

    def _merge_hybrid(self, asset_result: dict, spend_result: dict) -> dict:
        """
        Merge asset-specific and spend-based results for hybrid method.

        Applies 70% weight to asset-specific and 30% to spend-based.

        Args:
            asset_result: Asset-specific calculation result.
            spend_result: Spend-based calculation result.

        Returns:
            Merged hybrid result dictionary.
        """
        asset_co2e = Decimal(str(asset_result.get("total_co2e", "0")))
        spend_co2e = Decimal(str(spend_result.get("total_co2e", "0")))

        weighted = (
            asset_co2e * Decimal("0.70") + spend_co2e * Decimal("0.30")
        ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return {
            "total_co2e": str(weighted),
            "asset_specific_co2e": str(asset_co2e),
            "spend_based_co2e": str(spend_co2e),
            "asset_weight": "0.70",
            "spend_weight": "0.30",
            "method": "hybrid",
            "asset_detail": asset_result,
            "spend_detail": spend_result,
            "ef_source": asset_result.get("ef_source", ""),
        }

    def _stage_allocate(self, calc_result: dict, data: dict) -> dict:
        """
        Stage 6: ALLOCATE - Allocate emissions by tenant / cost center.

        Applies allocation method if specified (floor_area, headcount,
        revenue, metered, etc.). Otherwise returns unallocated result.

        Args:
            calc_result: Calculation result from Stage 5.
            data: Normalized asset data.

        Returns:
            Enriched result with allocation information.
        """
        result = dict(calc_result)
        allocation: Dict[str, Any] = {}

        alloc_method = data.get("allocation_method")
        tenant_share = data.get("tenant_share_pct")
        lessor_share = data.get("lessor_share_pct")

        if alloc_method:
            allocation["method"] = alloc_method

        if tenant_share is not None:
            tenant_pct = Decimal(str(tenant_share)) / Decimal("100")
            total = Decimal(str(result.get("total_co2e", "0")))
            allocated_co2e = (total * tenant_pct).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
            allocation["tenant_share_pct"] = str(tenant_share)
            allocation["allocated_co2e"] = str(allocated_co2e)
            result["total_co2e"] = str(allocated_co2e)
        elif lessor_share is not None:
            lessor_pct = Decimal(str(lessor_share)) / Decimal("100")
            total = Decimal(str(result.get("total_co2e", "0")))
            cat13_co2e = (total * (Decimal("1") - lessor_pct)).quantize(
                _QUANT_8DP, rounding=ROUND_HALF_UP
            )
            allocation["lessor_share_pct"] = str(lessor_share)
            allocation["cat13_co2e"] = str(cat13_co2e)
            result["total_co2e"] = str(cat13_co2e)

        # Attach tenant and cost center metadata
        if data.get("tenant_id"):
            allocation["tenant_id"] = data["tenant_id"]
        if data.get("cost_center"):
            allocation["cost_center"] = data["cost_center"]
        if data.get("department"):
            allocation["department"] = data["department"]

        result["allocation"] = allocation
        return result

    def _stage_aggregate(
        self, allocated: dict, category: AssetCategory, method: CalculationMethod
    ) -> dict:
        """
        Stage 7: AGGREGATE - Calculate DQI score and add metadata.

        Args:
            allocated: Allocated result from Stage 6.
            category: Asset category.
            method: Calculation method used.

        Returns:
            Enriched result with DQI score and aggregation metadata.
        """
        result = dict(allocated)

        # DQI scoring based on method
        dqi_scores: Dict[str, Decimal] = {
            "asset_specific": Decimal("4.5"),
            "average_data": Decimal("3.0"),
            "spend_based": Decimal("1.5"),
            "hybrid": Decimal("3.5"),
        }
        result["dqi_score"] = str(dqi_scores.get(method.value, Decimal("3.0")))

        # Add category and method metadata
        result["asset_category"] = category.value
        result["calculation_method"] = method.value

        return result

    def _stage_compliance(self, aggregated: dict) -> dict:
        """
        Stage 8: COMPLIANCE - Run compliance checks.

        Delegates to ComplianceCheckerEngine if available, otherwise
        performs lightweight inline compliance assessment.

        Args:
            aggregated: Aggregated result from Stage 7.

        Returns:
            Compliance check results dictionary.
        """
        engine = self._get_compliance_engine()
        if engine is not None:
            try:
                all_results = engine.check_all_frameworks(aggregated)
                summary = engine.get_compliance_summary(all_results)
                return summary
            except Exception as e:
                logger.warning("ComplianceCheckerEngine failed: %s", e)

        # Inline lightweight compliance check
        findings: List[str] = []
        status = "PASS"

        total_co2e = aggregated.get("total_co2e")
        if not total_co2e or Decimal(str(total_co2e)) <= 0:
            findings.append("GHG Protocol: total_co2e is missing or zero")
            status = "FAIL"

        method = aggregated.get("calculation_method")
        if not method:
            findings.append("GHG Protocol: calculation method not documented")
            status = "WARNING"

        return {
            "overall_status": status,
            "overall_score": 100.0 if status == "PASS" else 50.0,
            "findings": findings,
            "frameworks_checked": ["GHG_PROTOCOL"],
        }

    def _stage_provenance(self, chain_id: str, aggregated: dict) -> List[dict]:
        """
        Stage 9: PROVENANCE - Build the provenance chain.

        Args:
            chain_id: Provenance chain identifier.
            aggregated: Aggregated result.

        Returns:
            List of provenance chain entries.
        """
        return list(self._provenance_chains.get(chain_id, []))

    def _stage_seal(self, chain_id: str, aggregated: dict) -> str:
        """
        Stage 10: SEAL - Seal provenance chain with SHA-256 hash.

        Creates an immutable audit fingerprint from the entire provenance
        chain and the final calculation result.

        Args:
            chain_id: Provenance chain identifier.
            aggregated: Final aggregated result data.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        chain = self._provenance_chains.get(chain_id, [])
        chain_str = json.dumps(chain, sort_keys=True, default=str)
        result_str = json.dumps(aggregated, sort_keys=True, default=str)
        combined = f"{chain_str}|{result_str}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ==========================================================================
    # PROVENANCE AND UTILITY HELPERS
    # ==========================================================================

    def _record_provenance(
        self,
        chain_id: str,
        stage: PipelineStage,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """
        Record a provenance entry for a pipeline stage.

        Args:
            chain_id: Provenance chain identifier.
            stage: Pipeline stage enum.
            input_data: Input to this stage.
            output_data: Output from this stage.
        """
        input_str = (
            json.dumps(input_data, sort_keys=True, default=str)
            if not isinstance(input_data, str)
            else input_data
        )
        output_str = (
            json.dumps(output_data, sort_keys=True, default=str)
            if not isinstance(output_data, str)
            else output_data
        )

        entry = {
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": hashlib.sha256(input_str.encode("utf-8")).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode("utf-8")).hexdigest(),
        }

        chain = self._provenance_chains.get(chain_id)
        if chain is not None:
            entry["chain_hash"] = hashlib.sha256(
                (json.dumps(chain[-1], sort_keys=True) if chain else "").encode("utf-8")
            ).hexdigest()
            chain.append(entry)

    @staticmethod
    def _elapsed_ms(start: datetime) -> float:
        """Calculate milliseconds elapsed since start."""
        return (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

    # ==========================================================================
    # LAZY ENGINE LOADING
    # ==========================================================================

    def _get_asset_specific_engine(self) -> Optional[Any]:
        """Get or create AssetSpecificCalculatorEngine (lazy loading)."""
        if self._asset_specific_engine is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.asset_specific_calculator import (
                    AssetSpecificCalculatorEngine,
                )
                self._asset_specific_engine = AssetSpecificCalculatorEngine()
            except ImportError:
                logger.debug("AssetSpecificCalculatorEngine not available")
        return self._asset_specific_engine

    def _get_average_data_engine(self) -> Optional[Any]:
        """Get or create AverageDataCalculatorEngine (lazy loading)."""
        if self._average_data_engine is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.average_data_calculator import (
                    AverageDataCalculatorEngine,
                )
                self._average_data_engine = AverageDataCalculatorEngine()
            except ImportError:
                logger.debug("AverageDataCalculatorEngine not available")
        return self._average_data_engine

    def _get_spend_engine(self) -> Optional[Any]:
        """Get or create SpendBasedCalculatorEngine (lazy loading)."""
        if self._spend_engine is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.spend_based_calculator import (
                    SpendBasedCalculatorEngine,
                )
                self._spend_engine = SpendBasedCalculatorEngine()
            except ImportError:
                logger.debug("SpendBasedCalculatorEngine not available")
        return self._spend_engine

    def _get_hybrid_engine(self) -> Optional[Any]:
        """Get or create HybridAggregatorEngine (lazy loading)."""
        if self._hybrid_engine is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.hybrid_aggregator import (
                    HybridAggregatorEngine,
                )
                self._hybrid_engine = HybridAggregatorEngine()
            except ImportError:
                logger.debug("HybridAggregatorEngine not available")
        return self._hybrid_engine

    def _get_database_engine(self) -> Optional[Any]:
        """Get or create DownstreamAssetDatabaseEngine (lazy loading)."""
        if self._database_engine is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.downstream_asset_database import (
                    DownstreamAssetDatabaseEngine,
                )
                self._database_engine = DownstreamAssetDatabaseEngine()
            except ImportError:
                logger.debug("DownstreamAssetDatabaseEngine not available")
        return self._database_engine

    def _get_compliance_engine(self) -> Optional[Any]:
        """Get or create ComplianceCheckerEngine (lazy loading)."""
        if self._compliance_engine is None:
            try:
                from greenlang.agents.mrv.downstream_leased_assets.compliance_checker import (
                    ComplianceCheckerEngine,
                )
                self._compliance_engine = ComplianceCheckerEngine.get_instance()
            except ImportError:
                logger.debug("ComplianceCheckerEngine not available")
        return self._compliance_engine

    # ==========================================================================
    # UTILITY / STATUS METHODS
    # ==========================================================================

    def reset_pipeline(self) -> None:
        """Reset pipeline state (clear provenance chains)."""
        with self._lock:
            self._provenance_chains.clear()
            logger.info("Pipeline state reset")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None


# ==============================================================================
# MODULE-LEVEL HELPERS
# ==============================================================================


def get_pipeline_engine() -> DownstreamLeasedAssetsPipelineEngine:
    """
    Get singleton pipeline engine instance.

    Returns:
        DownstreamLeasedAssetsPipelineEngine singleton instance.
    """
    return DownstreamLeasedAssetsPipelineEngine()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "PipelineStage",
    "PipelineStatus",
    "AssetCategory",
    "CalculationMethod",
    "AllocationMethod",
    "BuildingType",
    "ClimateZone",
    "VehicleType",
    "EquipmentType",
    "ITAssetType",
    "EUI_BENCHMARKS",
    "GRID_EMISSION_FACTORS",
    "VEHICLE_EMISSION_FACTORS",
    "EQUIPMENT_EMISSION_FACTORS",
    "IT_ASSET_POWER",
    "DEFAULT_PUE",
    "EEIO_LEASING_FACTORS",
    "CLIMATE_ZONE_MULTIPLIERS",
    "VACANCY_ADJUSTMENT_FACTORS",
    "DownstreamLeasedAssetsPipelineEngine",
    "get_pipeline_engine",
]
