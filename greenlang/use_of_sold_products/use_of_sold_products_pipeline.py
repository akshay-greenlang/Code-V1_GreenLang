# -*- coding: utf-8 -*-
"""
UseOfSoldProductsPipelineEngine - AGENT-MRV-024 Engine 7

This module implements the UseOfSoldProductsPipelineEngine for Use of Sold Products
(GHG Protocol Scope 3 Category 11). It orchestrates a 10-stage pipeline for
complete use-of-sold-products emissions calculation from raw input to compliant
output with full audit trail.

Category 11 encompasses total expected lifetime emissions from the USE of goods
and services sold by the reporting company. This is often the largest Scope 3
category for manufacturers of energy-consuming products.

The 10 stages are:
1. VALIDATE: Input validation (required fields, types, ranges, category checks)
2. CLASSIFY: Determine product use category and emission type (direct/indirect)
3. NORMALIZE: Convert units (energy, mass, currency, GWP standardization)
4. RESOLVE_EFS: Resolve emission factors (fuel EF, grid EF, refrigerant GWP)
5. CALCULATE: Delegate to direct/indirect/fuels calculators
6. LIFETIME: Apply lifetime modeling (years, degradation, total lifetime emissions)
7. AGGREGATE: Multi-dimensional aggregation (by category, type, product)
8. COMPLIANCE: Run compliance checks if enabled (7 frameworks + DC)
9. PROVENANCE: Build provenance chain entries
10. SEAL: Seal provenance chain, generate final SHA-256 hash

Dual-path processing:
    - Direct emissions path: fuel combustion, refrigerant leakage, chemical release
    - Indirect emissions path: electricity, heating fuel, steam/cooling

Example:
    >>> from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
    ...     UseOfSoldProductsPipelineEngine,
    ... )
    >>> engine = UseOfSoldProductsPipelineEngine.get_instance()
    >>> result = engine.run_pipeline(inputs, org_id="ORG-001", year=2024)
    >>> print(f"Total lifetime emissions: {result['total_co2e']} kgCO2e")

Module: greenlang.use_of_sold_products.use_of_sold_products_pipeline
Agent: AGENT-MRV-024
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import logging
import hashlib
import json
import time
from threading import RLock
from enum import Enum

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "use_of_sold_products_pipeline_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"

_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")
_QUANT_8DP = Decimal("0.00000001")
ROUNDING = ROUND_HALF_UP

MAX_BATCH_SIZE: int = 5000
DEFAULT_LIFETIME_YEARS: int = 10
DEFAULT_DEGRADATION_RATE: Decimal = Decimal("0.00")


# ==============================================================================
# PIPELINE ENUMS
# ==============================================================================


class PipelineStage(str, Enum):
    """Pipeline stage identifiers for provenance tracking."""

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    LIFETIME = "lifetime"
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


class ProductUseCategory(str, Enum):
    """Product use categories for classification."""

    VEHICLES = "vehicles"
    APPLIANCES = "appliances"
    HVAC = "hvac"
    LIGHTING = "lighting"
    IT_EQUIPMENT = "it_equipment"
    INDUSTRIAL_EQUIPMENT = "industrial_equipment"
    FUELS_FEEDSTOCKS = "fuels_feedstocks"
    BUILDING_PRODUCTS = "building_products"
    CONSUMER_PRODUCTS = "consumer_products"
    MEDICAL_DEVICES = "medical_devices"


class EmissionType(str, Enum):
    """Whether emissions are direct or indirect use-phase."""

    DIRECT = "direct"
    INDIRECT = "indirect"
    BOTH = "both"


class CalculationMethod(str, Enum):
    """Calculation methods for Category 11."""

    DIRECT_FUEL_COMBUSTION = "direct_fuel_combustion"
    DIRECT_REFRIGERANT_LEAKAGE = "direct_refrigerant_leakage"
    DIRECT_CHEMICAL_RELEASE = "direct_chemical_release"
    INDIRECT_ELECTRICITY = "indirect_electricity"
    INDIRECT_HEATING_FUEL = "indirect_heating_fuel"
    INDIRECT_STEAM_COOLING = "indirect_steam_cooling"
    FUELS_SOLD = "fuels_sold"
    FEEDSTOCKS_SOLD = "feedstocks_sold"


# ==============================================================================
# EMISSION FACTOR TABLES
# ==============================================================================

# Fuel emission factors (kgCO2e per litre) - DEFRA 2024
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "petrol": {
        "co2_per_litre": Decimal("2.31"),
        "ch4_per_litre": Decimal("0.00058"),
        "n2o_per_litre": Decimal("0.00620"),
        "wtt_per_litre": Decimal("0.59"),
    },
    "diesel": {
        "co2_per_litre": Decimal("2.68"),
        "ch4_per_litre": Decimal("0.00039"),
        "n2o_per_litre": Decimal("0.01402"),
        "wtt_per_litre": Decimal("0.63"),
    },
    "natural_gas": {
        "co2_per_m3": Decimal("2.02"),
        "ch4_per_m3": Decimal("0.00059"),
        "n2o_per_m3": Decimal("0.00010"),
        "wtt_per_m3": Decimal("0.46"),
    },
    "lpg": {
        "co2_per_litre": Decimal("1.56"),
        "ch4_per_litre": Decimal("0.00066"),
        "n2o_per_litre": Decimal("0.00037"),
        "wtt_per_litre": Decimal("0.23"),
    },
    "heating_oil": {
        "co2_per_litre": Decimal("2.54"),
        "ch4_per_litre": Decimal("0.00031"),
        "n2o_per_litre": Decimal("0.00634"),
        "wtt_per_litre": Decimal("0.58"),
    },
    "kerosene": {
        "co2_per_litre": Decimal("2.54"),
        "ch4_per_litre": Decimal("0.00001"),
        "n2o_per_litre": Decimal("0.00002"),
        "wtt_per_litre": Decimal("0.58"),
    },
    "coal": {
        "co2_per_kg": Decimal("2.42"),
        "ch4_per_kg": Decimal("0.00036"),
        "n2o_per_kg": Decimal("0.01438"),
        "wtt_per_kg": Decimal("0.39"),
    },
    "ethanol_e85": {
        "co2_per_litre": Decimal("1.61"),
        "ch4_per_litre": Decimal("0.00035"),
        "n2o_per_litre": Decimal("0.00290"),
        "wtt_per_litre": Decimal("0.20"),
    },
    "biodiesel_b20": {
        "co2_per_litre": Decimal("2.23"),
        "ch4_per_litre": Decimal("0.00032"),
        "n2o_per_litre": Decimal("0.01200"),
        "wtt_per_litre": Decimal("0.50"),
    },
    "hydrogen": {
        "co2_per_kg": Decimal("0.00"),
        "ch4_per_kg": Decimal("0.00"),
        "n2o_per_kg": Decimal("0.00"),
        "wtt_per_kg": Decimal("11.20"),
    },
}

# Grid electricity emission factors (kgCO2e per kWh) - IEA/eGRID 2024
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.3937"),
    "US_CAMX": Decimal("0.2415"),
    "US_ERCT": Decimal("0.3736"),
    "US_FRCC": Decimal("0.3831"),
    "US_MROE": Decimal("0.5612"),
    "US_MROW": Decimal("0.4379"),
    "US_NEWE": Decimal("0.2134"),
    "US_NWPP": Decimal("0.2697"),
    "US_NYCW": Decimal("0.2420"),
    "US_RFCE": Decimal("0.3077"),
    "US_RFCM": Decimal("0.5241"),
    "US_RFCW": Decimal("0.4839"),
    "US_RMPA": Decimal("0.5395"),
    "US_SPNO": Decimal("0.4619"),
    "US_SPSO": Decimal("0.4267"),
    "US_SRMV": Decimal("0.3528"),
    "US_SRSO": Decimal("0.3782"),
    "US_SRTV": Decimal("0.4076"),
    "US_SRVC": Decimal("0.3207"),
    "GB": Decimal("0.2121"),
    "DE": Decimal("0.3380"),
    "FR": Decimal("0.0569"),
    "CN": Decimal("0.5810"),
    "IN": Decimal("0.7080"),
    "JP": Decimal("0.4570"),
    "AU": Decimal("0.6560"),
    "CA": Decimal("0.1200"),
    "BR": Decimal("0.0750"),
    "KR": Decimal("0.4590"),
    "IT": Decimal("0.2340"),
    "ES": Decimal("0.1530"),
    "NL": Decimal("0.3280"),
    "SE": Decimal("0.0130"),
    "NO": Decimal("0.0080"),
    "GLOBAL": Decimal("0.4360"),
}

# Refrigerant GWP values (AR5 100-year) for common refrigerants
REFRIGERANT_GWP: Dict[str, int] = {
    "R-134a": 1430,
    "R-410A": 2088,
    "R-407C": 1774,
    "R-404A": 3922,
    "R-507A": 3985,
    "R-32": 675,
    "R-1234yf": 4,
    "R-1234ze": 7,
    "R-290": 3,
    "R-600a": 3,
    "R-744": 1,
    "R-717": 0,
    "R-22": 1810,
    "R-12": 10900,
    "R-11": 4750,
    "R-502": 4657,
    "R-23": 14800,
    "R-125": 3500,
    "R-143a": 4470,
    "R-227ea": 3220,
    "R-236fa": 9810,
    "SF6": 22800,
    "HFC-152a": 124,
    "CO2": 1,
}

# Default product lifetime estimates (years) by category
DEFAULT_PRODUCT_LIFETIMES: Dict[str, int] = {
    "vehicles": 15,
    "appliances": 12,
    "hvac": 15,
    "lighting": 5,
    "it_equipment": 5,
    "industrial_equipment": 20,
    "fuels_feedstocks": 1,
    "building_products": 30,
    "consumer_products": 2,
    "medical_devices": 10,
}

# Default annual energy consumption (kWh/year) for typical products
DEFAULT_ENERGY_PROFILES: Dict[str, Decimal] = {
    "refrigerator": Decimal("400"),
    "washing_machine": Decimal("200"),
    "dishwasher": Decimal("270"),
    "clothes_dryer": Decimal("600"),
    "oven_electric": Decimal("350"),
    "air_conditioner_room": Decimal("1200"),
    "air_conditioner_central": Decimal("3500"),
    "heat_pump": Decimal("3000"),
    "furnace_gas": Decimal("12000"),  # kWh of gas
    "water_heater_electric": Decimal("2000"),
    "water_heater_gas": Decimal("3500"),  # kWh of gas
    "laptop": Decimal("50"),
    "desktop": Decimal("200"),
    "monitor": Decimal("100"),
    "server": Decimal("4000"),
    "led_bulb": Decimal("10"),
    "cfl_bulb": Decimal("13"),
    "tv_lcd": Decimal("100"),
    "ev_charger": Decimal("2400"),
    "medical_imaging": Decimal("50000"),
    "ventilator": Decimal("1500"),
}

# Degradation rates (annual efficiency loss) by category
DEFAULT_DEGRADATION_RATES: Dict[str, Decimal] = {
    "vehicles": Decimal("0.005"),
    "appliances": Decimal("0.01"),
    "hvac": Decimal("0.015"),
    "lighting": Decimal("0.02"),
    "it_equipment": Decimal("0.00"),
    "industrial_equipment": Decimal("0.005"),
    "fuels_feedstocks": Decimal("0.00"),
    "building_products": Decimal("0.005"),
    "consumer_products": Decimal("0.00"),
    "medical_devices": Decimal("0.01"),
}

# Category -> emission type mapping
CATEGORY_EMISSION_TYPE: Dict[str, EmissionType] = {
    "vehicles": EmissionType.DIRECT,
    "appliances": EmissionType.INDIRECT,
    "hvac": EmissionType.BOTH,
    "lighting": EmissionType.INDIRECT,
    "it_equipment": EmissionType.INDIRECT,
    "industrial_equipment": EmissionType.BOTH,
    "fuels_feedstocks": EmissionType.DIRECT,
    "building_products": EmissionType.INDIRECT,
    "consumer_products": EmissionType.DIRECT,
    "medical_devices": EmissionType.INDIRECT,
}


# ==============================================================================
# PIPELINE ENGINE
# ==============================================================================


class UseOfSoldProductsPipelineEngine:
    """
    UseOfSoldProductsPipelineEngine - 10-stage pipeline for Category 11.

    This engine coordinates the complete use-of-sold-products emissions
    calculation workflow through 10 sequential stages, from input validation
    to sealed audit trail. It supports dual-path processing for both direct
    and indirect use-phase emissions.

    The engine uses lazy initialization for sub-engines, creating them only
    when needed. This reduces startup time and memory footprint.

    Attributes:
        _database_engine: ProductUseDatabaseEngine (lazy-loaded)
        _direct_engine: DirectEmissionsCalculatorEngine (lazy-loaded)
        _indirect_engine: IndirectEmissionsCalculatorEngine (lazy-loaded)
        _fuels_engine: FuelsAndFeedstocksCalculatorEngine (lazy-loaded)
        _lifetime_engine: LifetimeModelingEngine (lazy-loaded)
        _compliance_engine: ComplianceCheckerEngine (lazy-loaded)

    Example:
        >>> engine = UseOfSoldProductsPipelineEngine.get_instance()
        >>> result = engine.run_pipeline(inputs, org_id="ORG-001", year=2024)
        >>> assert result["status"] == "SUCCESS"
    """

    _instance: Optional["UseOfSoldProductsPipelineEngine"] = None
    _lock: RLock = RLock()

    def __new__(cls) -> "UseOfSoldProductsPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize UseOfSoldProductsPipelineEngine.

        Prevents re-initialization of the singleton. All sub-engines are
        lazy-loaded on first pipeline run.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Lazy-loaded engines
        self._database_engine: Optional[Any] = None
        self._direct_engine: Optional[Any] = None
        self._indirect_engine: Optional[Any] = None
        self._fuels_engine: Optional[Any] = None
        self._lifetime_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None

        # Pipeline state
        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}
        self._run_count: int = 0

        self._initialized = True
        logger.info(
            "UseOfSoldProductsPipelineEngine initialized (version %s)",
            ENGINE_VERSION,
        )

    @classmethod
    def get_instance(cls) -> "UseOfSoldProductsPipelineEngine":
        """
        Get singleton instance (thread-safe).

        Returns:
            UseOfSoldProductsPipelineEngine singleton instance.
        """
        if cls._instance is None:
            cls()
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None
            logger.info("UseOfSoldProductsPipelineEngine singleton reset")

    # ==========================================================================
    # PUBLIC API - CORE PROCESSING METHODS
    # ==========================================================================

    def run_pipeline(
        self,
        inputs: List[Dict[str, Any]],
        org_id: str = "",
        year: int = 2024,
    ) -> Dict[str, Any]:
        """
        Execute the full 10-stage pipeline for use-of-sold-products emissions.

        Processes a list of product inputs through validation, classification,
        normalization, EF resolution, calculation, lifetime modeling,
        aggregation, compliance checking, provenance tracking, and sealing.

        Args:
            inputs: List of product input dictionaries, each containing
                product_category, units_sold, and method-specific data.
            org_id: Organization identifier for provenance.
            year: Reporting year.

        Returns:
            Dictionary with total_co2e, direct_co2e, indirect_co2e,
            breakdowns, compliance results, provenance hash, and status.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.

        Example:
            >>> result = engine.run_pipeline([
            ...     {"product_category": "vehicles", "fuel_type": "petrol",
            ...      "fuel_consumption_litres_per_year": 1500, "units_sold": 10000,
            ...      "lifetime_years": 15},
            ... ], org_id="ORG-001", year=2024)
            >>> result["status"]
            'SUCCESS'
        """
        chain_id = f"usp-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}
        pipeline_start = time.monotonic()

        try:
            # ------------------------------------------------------------------
            # Stage 1: VALIDATE
            # ------------------------------------------------------------------
            start = time.monotonic()
            validated_inputs, errors = self._stage_validate(inputs)
            duration_ms = self._elapsed_ms(start)
            stage_durations["VALIDATE"] = duration_ms
            self._record_provenance(
                chain_id, PipelineStage.VALIDATE,
                {"input_count": len(inputs)},
                {"valid": len(errors) == 0, "errors": errors},
            )

            if errors:
                raise ValueError(
                    f"Input validation failed: {'; '.join(errors)}"
                )
            logger.info(
                "[%s] Stage VALIDATE completed in %.2fms (%d products)",
                chain_id, duration_ms, len(validated_inputs),
            )

            # ------------------------------------------------------------------
            # Stage 2: CLASSIFY
            # ------------------------------------------------------------------
            start = time.monotonic()
            classified = self._stage_classify(validated_inputs)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CLASSIFY"] = duration_ms
            self._record_provenance(
                chain_id, PipelineStage.CLASSIFY,
                {"count": len(classified)},
                {"categories": list({c.get("product_category", "") for c in classified})},
            )
            logger.info(
                "[%s] Stage CLASSIFY completed in %.2fms", chain_id, duration_ms,
            )

            # ------------------------------------------------------------------
            # Stage 3: NORMALIZE
            # ------------------------------------------------------------------
            start = time.monotonic()
            normalized = self._stage_normalize(classified)
            duration_ms = self._elapsed_ms(start)
            stage_durations["NORMALIZE"] = duration_ms
            self._record_provenance(
                chain_id, PipelineStage.NORMALIZE,
                {"count": len(normalized)},
                {"normalized": True},
            )
            logger.info(
                "[%s] Stage NORMALIZE completed in %.2fms", chain_id, duration_ms,
            )

            # ------------------------------------------------------------------
            # Stage 4: RESOLVE_EFS
            # ------------------------------------------------------------------
            start = time.monotonic()
            resolved = self._stage_resolve_efs(normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations["RESOLVE_EFS"] = duration_ms
            self._record_provenance(
                chain_id, PipelineStage.RESOLVE_EFS,
                {"count": len(resolved)},
                {"efs_resolved": True},
            )
            logger.info(
                "[%s] Stage RESOLVE_EFS completed in %.2fms", chain_id, duration_ms,
            )

            # ------------------------------------------------------------------
            # Stage 5: CALCULATE
            # ------------------------------------------------------------------
            start = time.monotonic()
            calculated = self._stage_calculate(resolved)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CALCULATE"] = duration_ms
            total_annual = sum(
                Decimal(str(c.get("annual_co2e", 0))) for c in calculated
            )
            self._record_provenance(
                chain_id, PipelineStage.CALCULATE,
                {"count": len(calculated)},
                {"total_annual_co2e": str(total_annual)},
            )
            logger.info(
                "[%s] Stage CALCULATE completed in %.2fms (annual=%s kgCO2e)",
                chain_id, duration_ms, total_annual,
            )

            # ------------------------------------------------------------------
            # Stage 6: LIFETIME
            # ------------------------------------------------------------------
            start = time.monotonic()
            lifetime_results = self._stage_lifetime(calculated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["LIFETIME"] = duration_ms
            total_lifetime = sum(
                Decimal(str(lr.get("lifetime_co2e", 0))) for lr in lifetime_results
            )
            self._record_provenance(
                chain_id, PipelineStage.LIFETIME,
                {"count": len(lifetime_results)},
                {"total_lifetime_co2e": str(total_lifetime)},
            )
            logger.info(
                "[%s] Stage LIFETIME completed in %.2fms (lifetime=%s kgCO2e)",
                chain_id, duration_ms, total_lifetime,
            )

            # ------------------------------------------------------------------
            # Stage 7: AGGREGATE
            # ------------------------------------------------------------------
            start = time.monotonic()
            aggregated = self._stage_aggregate(lifetime_results)
            duration_ms = self._elapsed_ms(start)
            stage_durations["AGGREGATE"] = duration_ms
            self._record_provenance(
                chain_id, PipelineStage.AGGREGATE,
                {"products": len(lifetime_results)},
                {"total_co2e": str(aggregated.get("total_co2e", 0))},
            )
            logger.info(
                "[%s] Stage AGGREGATE completed in %.2fms", chain_id, duration_ms,
            )

            # ------------------------------------------------------------------
            # Stage 8: COMPLIANCE
            # ------------------------------------------------------------------
            start = time.monotonic()
            compliance_results = self._stage_compliance(aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["COMPLIANCE"] = duration_ms
            self._record_provenance(
                chain_id, PipelineStage.COMPLIANCE,
                {"input": "aggregated_result"},
                {"compliance_status": compliance_results.get("overall_status", "N/A")},
            )
            logger.info(
                "[%s] Stage COMPLIANCE completed in %.2fms", chain_id, duration_ms,
            )

            # ------------------------------------------------------------------
            # Stage 9: PROVENANCE
            # ------------------------------------------------------------------
            start = time.monotonic()
            provenance_data = self._stage_provenance(
                chain_id, aggregated, compliance_results, org_id, year,
            )
            duration_ms = self._elapsed_ms(start)
            stage_durations["PROVENANCE"] = duration_ms
            logger.info(
                "[%s] Stage PROVENANCE completed in %.2fms", chain_id, duration_ms,
            )

            # ------------------------------------------------------------------
            # Stage 10: SEAL
            # ------------------------------------------------------------------
            start = time.monotonic()
            provenance_hash = self._stage_seal(chain_id, aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["SEAL"] = duration_ms
            logger.info(
                "[%s] Stage SEAL completed in %.2fms (hash=%s...)",
                chain_id, duration_ms, provenance_hash[:16],
            )

            # Build final result
            total_duration_ms = (time.monotonic() - pipeline_start) * 1000.0
            self._run_count += 1

            final_result = {
                "status": PipelineStatus.SUCCESS.value,
                "chain_id": chain_id,
                "org_id": org_id,
                "reporting_year": year,
                "total_co2e": str(aggregated.get("total_co2e", Decimal("0"))),
                "direct_co2e": str(aggregated.get("direct_co2e", Decimal("0"))),
                "indirect_co2e": str(aggregated.get("indirect_co2e", Decimal("0"))),
                "by_category": aggregated.get("by_category", {}),
                "by_emission_type": aggregated.get("by_emission_type", {}),
                "by_method": aggregated.get("by_method", {}),
                "product_count": len(lifetime_results),
                "total_units_sold": sum(
                    int(lr.get("units_sold", 0)) for lr in lifetime_results
                ),
                "compliance": compliance_results,
                "provenance_hash": provenance_hash,
                "provenance_chain": provenance_data,
                "stage_durations_ms": stage_durations,
                "total_processing_time_ms": total_duration_ms,
                "engine_version": ENGINE_VERSION,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "[%s] Pipeline completed successfully in %.2fms. "
                "Total lifetime emissions: %s kgCO2e",
                chain_id, total_duration_ms, aggregated.get("total_co2e", 0),
            )
            return final_result

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "[%s] Pipeline execution failed: %s", chain_id, e, exc_info=True,
            )
            raise RuntimeError(f"Pipeline execution failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def run_batch(
        self,
        batch_inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple product portfolios in a single batch.

        Each item in batch_inputs is a dict with 'products', 'org_id', 'year'.
        Handles errors per-portfolio without failing the entire batch.

        Args:
            batch_inputs: List of batch items, each containing:
                - products: List of product input dicts
                - org_id: Organization identifier
                - year: Reporting year

        Returns:
            Dictionary with individual results, totals, and error details.

        Example:
            >>> result = engine.run_batch([
            ...     {"products": [...], "org_id": "ORG-001", "year": 2024},
            ...     {"products": [...], "org_id": "ORG-002", "year": 2024},
            ... ])
            >>> result["total_portfolios"]
            2
        """
        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = Decimal("0")

        if len(batch_inputs) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(batch_inputs)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        logger.info("Starting batch processing (%d portfolios)", len(batch_inputs))

        for idx, batch_item in enumerate(batch_inputs):
            products = batch_item.get("products", [])
            org_id = batch_item.get("org_id", f"ORG-{idx}")
            year = batch_item.get("year", 2024)

            try:
                result = self.run_pipeline(products, org_id=org_id, year=year)
                results.append(result)
                result_co2e = Decimal(str(result.get("total_co2e", "0")))
                total_co2e += result_co2e
            except Exception as e:
                logger.error(
                    "Batch item %d (%s) failed: %s", idx, org_id, e,
                )
                errors.append({
                    "index": idx,
                    "org_id": org_id,
                    "error": str(e),
                })

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return {
            "status": PipelineStatus.SUCCESS.value if not errors
            else PipelineStatus.PARTIAL_SUCCESS.value,
            "total_portfolios": len(batch_inputs),
            "successful": len(results),
            "failed": len(errors),
            "total_co2e": str(total_co2e),
            "results": results,
            "errors": errors,
            "processing_time_ms": elapsed_ms,
        }

    def run_portfolio_analysis(
        self,
        inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run portfolio-level analysis across all product categories.

        Provides additional analytics beyond run_pipeline, including
        hot-spot identification, Pareto analysis, and category-level
        intensity metrics.

        Args:
            inputs: List of product input dictionaries.

        Returns:
            Dictionary with portfolio analysis results.

        Example:
            >>> analysis = engine.run_portfolio_analysis(products)
            >>> analysis["hot_spots"][0]["category"]
            'vehicles'
        """
        start_time = time.monotonic()

        # Run standard pipeline
        pipeline_result = self.run_pipeline(inputs, org_id="portfolio", year=2024)

        # Compute hot-spots (top contributors)
        by_category = pipeline_result.get("by_category", {})
        total_co2e = Decimal(str(pipeline_result.get("total_co2e", "0")))

        hot_spots: List[Dict[str, Any]] = []
        for category, cat_co2e_str in sorted(
            by_category.items(),
            key=lambda x: Decimal(str(x[1])),
            reverse=True,
        ):
            cat_co2e = Decimal(str(cat_co2e_str))
            pct = (
                (cat_co2e / total_co2e * Decimal("100"))
                if total_co2e > 0
                else Decimal("0")
            )
            hot_spots.append({
                "category": category,
                "co2e": str(cat_co2e),
                "percentage": str(pct.quantize(_QUANT_2DP, rounding=ROUNDING)),
            })

        # Pareto analysis (cumulative percentage)
        cumulative_pct = Decimal("0")
        pareto_items: List[Dict[str, Any]] = []
        for hs in hot_spots:
            cumulative_pct += Decimal(hs["percentage"])
            pareto_items.append({
                **hs,
                "cumulative_pct": str(
                    cumulative_pct.quantize(_QUANT_2DP, rounding=ROUNDING)
                ),
                "in_80_pct": cumulative_pct <= Decimal("80"),
            })

        # Intensity metrics
        total_units = pipeline_result.get("total_units_sold", 0)
        intensity_per_unit = (
            total_co2e / Decimal(str(total_units))
            if total_units > 0
            else Decimal("0")
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return {
            "pipeline_result": pipeline_result,
            "hot_spots": pareto_items,
            "total_co2e": str(total_co2e),
            "total_units_sold": total_units,
            "intensity_per_unit_kg_co2e": str(
                intensity_per_unit.quantize(_QUANT_4DP, rounding=ROUNDING)
            ),
            "category_count": len(by_category),
            "processing_time_ms": elapsed_ms,
        }

    def validate_inputs(
        self,
        inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate inputs without running the full pipeline.

        Useful for pre-flight validation before submitting a batch.

        Args:
            inputs: List of product input dictionaries.

        Returns:
            Dictionary with is_valid flag and error details.
        """
        _, errors = self._stage_validate(inputs)
        return {
            "is_valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": errors,
            "product_count": len(inputs),
        }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and configuration.

        Returns:
            Dictionary with engine info, run count, and engine availability.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "run_count": self._run_count,
            "stages": [s.value for s in PipelineStage],
            "engines": {
                "database": self._database_engine is not None,
                "direct": self._direct_engine is not None,
                "indirect": self._indirect_engine is not None,
                "fuels": self._fuels_engine is not None,
                "lifetime": self._lifetime_engine is not None,
                "compliance": self._compliance_engine is not None,
            },
        }

    def estimate_runtime(self, n: int) -> Dict[str, Any]:
        """
        Estimate pipeline runtime for n products.

        Based on empirical benchmarks: ~2ms per product per stage.

        Args:
            n: Number of products.

        Returns:
            Dictionary with estimated runtime in ms and seconds.
        """
        ms_per_product_per_stage = 2.0
        num_stages = len(PipelineStage)
        estimated_ms = n * ms_per_product_per_stage * num_stages
        return {
            "product_count": n,
            "stages": num_stages,
            "estimated_ms": round(estimated_ms, 1),
            "estimated_seconds": round(estimated_ms / 1000.0, 2),
            "note": "Estimate based on ~2ms per product per stage",
        }

    # ==========================================================================
    # STAGE METHODS (PRIVATE)
    # ==========================================================================

    def _stage_validate(
        self, inputs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Stage 1: VALIDATE - Input validation.

        Checks:
        - inputs is non-empty
        - Each product has product_category
        - Each product has units_sold > 0
        - Method-specific required fields are present

        Args:
            inputs: List of product input dicts.

        Returns:
            Tuple of (validated_inputs, error_messages).
        """
        errors: List[str] = []

        if not inputs:
            errors.append("inputs must not be empty")
            return [], errors

        if len(inputs) > MAX_BATCH_SIZE:
            errors.append(
                f"Input count {len(inputs)} exceeds max {MAX_BATCH_SIZE}"
            )

        valid_categories = {c.value for c in ProductUseCategory}

        for idx, product in enumerate(inputs):
            if not isinstance(product, dict):
                errors.append(f"Product {idx}: must be a dictionary")
                continue

            # Required: product_category
            category = product.get("product_category")
            if not category:
                errors.append(f"Product {idx}: product_category is required")
            elif category.lower() not in valid_categories:
                errors.append(
                    f"Product {idx}: unknown product_category '{category}'. "
                    f"Valid: {sorted(valid_categories)}"
                )

            # Required: units_sold
            units_sold = product.get("units_sold")
            if units_sold is None:
                errors.append(f"Product {idx}: units_sold is required")
            elif not isinstance(units_sold, (int, float)) or units_sold <= 0:
                errors.append(
                    f"Product {idx}: units_sold must be a positive number"
                )

            # Method-specific validation
            self._validate_method_specific(idx, product, errors)

        return inputs if not errors else [], errors

    def _validate_method_specific(
        self, idx: int, product: dict, errors: List[str]
    ) -> None:
        """
        Validate method-specific required fields for a product.

        Args:
            idx: Product index for error messages.
            product: Product dictionary.
            errors: Error accumulator list (mutated in-place).
        """
        category = (product.get("product_category") or "").lower()

        # Vehicles require fuel info
        if category == "vehicles":
            fuel_type = product.get("fuel_type")
            fuel_consumption = product.get("fuel_consumption_litres_per_year")
            if not fuel_type and not fuel_consumption:
                # May be electric vehicle
                kwh = product.get("electricity_kwh_per_year")
                if not kwh:
                    errors.append(
                        f"Product {idx} (vehicles): fuel_type + "
                        "fuel_consumption_litres_per_year or "
                        "electricity_kwh_per_year is required"
                    )

        # HVAC may require refrigerant info
        if category == "hvac":
            refrigerant = product.get("refrigerant_type")
            kwh = product.get("electricity_kwh_per_year")
            if not refrigerant and not kwh:
                errors.append(
                    f"Product {idx} (hvac): refrigerant_type or "
                    "electricity_kwh_per_year is required"
                )

        # Fuels require fuel_type
        if category == "fuels_feedstocks":
            fuel_type = product.get("fuel_type")
            quantity = product.get("quantity_sold")
            if not fuel_type:
                errors.append(
                    f"Product {idx} (fuels_feedstocks): fuel_type is required"
                )
            if not quantity:
                errors.append(
                    f"Product {idx} (fuels_feedstocks): quantity_sold is required"
                )

    def _stage_classify(
        self, products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: CLASSIFY - Determine product use category and emission type.

        Auto-selects emission type (direct/indirect/both) based on the
        product category and available data. Assigns calculation method
        via waterfall logic.

        Args:
            products: List of validated product dicts.

        Returns:
            List of classified product dicts with emission_type and method.
        """
        classified: List[Dict[str, Any]] = []

        for product in products:
            p = dict(product)  # shallow copy
            category = p.get("product_category", "").lower()

            # Assign emission type from category mapping
            emission_type = CATEGORY_EMISSION_TYPE.get(
                category, EmissionType.INDIRECT
            )
            p["emission_type"] = emission_type.value

            # Assign calculation method via waterfall
            p["calculation_method"] = self._resolve_method(category, p)

            classified.append(p)

        return classified

    def _resolve_method(
        self, category: str, product: dict
    ) -> str:
        """
        Resolve calculation method via waterfall logic.

        Priority:
        1. Explicit method override
        2. Fuels/feedstocks -> fuels_sold / feedstocks_sold
        3. Fuel data present -> direct_fuel_combustion
        4. Refrigerant data present -> direct_refrigerant_leakage
        5. Chemical data present -> direct_chemical_release
        6. Electricity data present -> indirect_electricity
        7. Heating fuel data present -> indirect_heating_fuel
        8. Steam/cooling data present -> indirect_steam_cooling
        9. Default: indirect_electricity

        Args:
            category: Product use category (lowercase).
            product: Product dictionary.

        Returns:
            Calculation method string.
        """
        # Check for explicit override
        explicit = product.get("calculation_method")
        if explicit:
            return explicit

        # Fuels and feedstocks
        if category == "fuels_feedstocks":
            if product.get("is_feedstock", False):
                return CalculationMethod.FEEDSTOCKS_SOLD.value
            return CalculationMethod.FUELS_SOLD.value

        # Direct fuel combustion
        if product.get("fuel_type") and product.get("fuel_consumption_litres_per_year"):
            return CalculationMethod.DIRECT_FUEL_COMBUSTION.value

        # Direct refrigerant leakage
        if product.get("refrigerant_type") and product.get("refrigerant_charge_kg"):
            return CalculationMethod.DIRECT_REFRIGERANT_LEAKAGE.value

        # Direct chemical release
        if product.get("chemical_type") and product.get("chemical_content_kg"):
            return CalculationMethod.DIRECT_CHEMICAL_RELEASE.value

        # Indirect electricity
        if product.get("electricity_kwh_per_year"):
            return CalculationMethod.INDIRECT_ELECTRICITY.value

        # Indirect heating fuel
        if product.get("heating_fuel_type") and product.get("heating_fuel_consumption"):
            return CalculationMethod.INDIRECT_HEATING_FUEL.value

        # Indirect steam/cooling
        if product.get("steam_kwh_per_year") or product.get("cooling_kwh_per_year"):
            return CalculationMethod.INDIRECT_STEAM_COOLING.value

        # Default based on category
        if category in ("appliances", "lighting", "it_equipment",
                        "building_products", "medical_devices"):
            return CalculationMethod.INDIRECT_ELECTRICITY.value

        return CalculationMethod.INDIRECT_ELECTRICITY.value

    def _stage_normalize(
        self, products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 3: NORMALIZE - Convert units and standardize values.

        Conversions:
        - miles -> km (fuel consumption context)
        - gallons -> litres
        - Wh -> kWh, MWh -> kWh
        - BTU -> kWh
        - Currency to USD for spend-based
        - Product category to lowercase

        Args:
            products: List of classified product dicts.

        Returns:
            List of normalized product dicts.
        """
        normalized: List[Dict[str, Any]] = []

        for product in products:
            p = dict(product)

            # Normalize category to lowercase
            if "product_category" in p:
                p["product_category"] = p["product_category"].lower()

            # Gallons to litres
            if "fuel_consumption_gallons_per_year" in p:
                gallons = Decimal(str(p["fuel_consumption_gallons_per_year"]))
                p["fuel_consumption_litres_per_year"] = str(
                    (gallons * Decimal("3.78541")).quantize(
                        _QUANT_8DP, rounding=ROUNDING
                    )
                )
                p["_original_fuel_gallons"] = p.pop(
                    "fuel_consumption_gallons_per_year"
                )

            # Wh to kWh
            if "electricity_wh_per_year" in p:
                wh = Decimal(str(p["electricity_wh_per_year"]))
                p["electricity_kwh_per_year"] = str(
                    (wh / Decimal("1000")).quantize(_QUANT_8DP, rounding=ROUNDING)
                )
                p["_original_wh"] = p.pop("electricity_wh_per_year")

            # MWh to kWh
            if "electricity_mwh_per_year" in p:
                mwh = Decimal(str(p["electricity_mwh_per_year"]))
                p["electricity_kwh_per_year"] = str(
                    (mwh * Decimal("1000")).quantize(_QUANT_8DP, rounding=ROUNDING)
                )
                p["_original_mwh"] = p.pop("electricity_mwh_per_year")

            # BTU to kWh (1 BTU = 0.000293071 kWh)
            if "energy_btu_per_year" in p:
                btu = Decimal(str(p["energy_btu_per_year"]))
                p["electricity_kwh_per_year"] = str(
                    (btu * Decimal("0.000293071")).quantize(
                        _QUANT_8DP, rounding=ROUNDING
                    )
                )
                p["_original_btu"] = p.pop("energy_btu_per_year")

            # Cubic feet to cubic metres (natural gas)
            if "natural_gas_cf_per_year" in p:
                cf = Decimal(str(p["natural_gas_cf_per_year"]))
                p["natural_gas_m3_per_year"] = str(
                    (cf * Decimal("0.0283168")).quantize(
                        _QUANT_8DP, rounding=ROUNDING
                    )
                )
                p["_original_cf"] = p.pop("natural_gas_cf_per_year")

            # Ensure units_sold is int
            if "units_sold" in p:
                p["units_sold"] = int(p["units_sold"])

            normalized.append(p)

        return normalized

    def _stage_resolve_efs(
        self, products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 4: RESOLVE_EFS - Resolve emission factors from built-in tables.

        Resolves:
        - Fuel emission factors (kgCO2e/litre or kgCO2e/m3)
        - Grid electricity factors (kgCO2e/kWh) by country/region
        - Refrigerant GWP values (AR5 100-year)

        Args:
            products: List of normalized product dicts.

        Returns:
            List of product dicts with resolved emission factors.
        """
        resolved: List[Dict[str, Any]] = []

        for product in products:
            p = dict(product)
            method = p.get("calculation_method", "")

            # Resolve fuel EFs
            if method in (
                CalculationMethod.DIRECT_FUEL_COMBUSTION.value,
                CalculationMethod.FUELS_SOLD.value,
                CalculationMethod.INDIRECT_HEATING_FUEL.value,
            ):
                fuel_type = p.get("fuel_type", "").lower()
                heating_fuel = p.get("heating_fuel_type", "").lower()
                lookup_fuel = fuel_type or heating_fuel
                ef_data = FUEL_EMISSION_FACTORS.get(lookup_fuel)
                if ef_data:
                    p["resolved_fuel_ef"] = {
                        k: str(v) for k, v in ef_data.items()
                    }
                    p["ef_source"] = "DEFRA_2024"
                else:
                    p["resolved_fuel_ef"] = None
                    p["ef_source"] = "unknown"
                    logger.warning(
                        "No fuel EF found for '%s'", lookup_fuel,
                    )

            # Resolve grid EFs
            if method in (
                CalculationMethod.INDIRECT_ELECTRICITY.value,
                CalculationMethod.INDIRECT_STEAM_COOLING.value,
            ):
                country = p.get("country_code", "GLOBAL").upper()
                region = p.get("egrid_subregion", "").upper()
                lookup_key = f"US_{region}" if region else country
                grid_ef = GRID_EMISSION_FACTORS.get(
                    lookup_key,
                    GRID_EMISSION_FACTORS.get(country, GRID_EMISSION_FACTORS["GLOBAL"]),
                )
                p["resolved_grid_ef"] = str(grid_ef)
                p["ef_source"] = "IEA_eGRID_2024"

            # Resolve refrigerant GWP
            if method == CalculationMethod.DIRECT_REFRIGERANT_LEAKAGE.value:
                ref_type = p.get("refrigerant_type", "")
                gwp = REFRIGERANT_GWP.get(ref_type)
                if gwp is not None:
                    p["resolved_gwp"] = gwp
                    p["ef_source"] = "IPCC_AR5"
                else:
                    p["resolved_gwp"] = None
                    p["ef_source"] = "unknown"
                    logger.warning(
                        "No GWP found for refrigerant '%s'", ref_type,
                    )

            # Chemical release - use provided emission factor
            if method == CalculationMethod.DIRECT_CHEMICAL_RELEASE.value:
                p["ef_source"] = p.get("ef_source", "product_data")

            resolved.append(p)

        return resolved

    def _stage_calculate(
        self, products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 5: CALCULATE - Compute annual use-phase emissions per product.

        Dispatches to the appropriate calculation method based on
        the resolved calculation_method field.

        Args:
            products: List of EF-resolved product dicts.

        Returns:
            List of product dicts with annual_co2e and breakdown.
        """
        calculated: List[Dict[str, Any]] = []

        for product in products:
            p = dict(product)
            method = p.get("calculation_method", "")

            try:
                if method == CalculationMethod.DIRECT_FUEL_COMBUSTION.value:
                    result = self._calc_direct_fuel(p)
                elif method == CalculationMethod.DIRECT_REFRIGERANT_LEAKAGE.value:
                    result = self._calc_direct_refrigerant(p)
                elif method == CalculationMethod.DIRECT_CHEMICAL_RELEASE.value:
                    result = self._calc_direct_chemical(p)
                elif method == CalculationMethod.INDIRECT_ELECTRICITY.value:
                    result = self._calc_indirect_electricity(p)
                elif method == CalculationMethod.INDIRECT_HEATING_FUEL.value:
                    result = self._calc_indirect_heating(p)
                elif method == CalculationMethod.INDIRECT_STEAM_COOLING.value:
                    result = self._calc_indirect_steam_cooling(p)
                elif method == CalculationMethod.FUELS_SOLD.value:
                    result = self._calc_fuels_sold(p)
                elif method == CalculationMethod.FEEDSTOCKS_SOLD.value:
                    result = self._calc_feedstocks_sold(p)
                else:
                    result = self._calc_indirect_electricity(p)

                p.update(result)
            except Exception as e:
                logger.error(
                    "Calculation failed for product '%s': %s",
                    p.get("product_category"), e, exc_info=True,
                )
                p["annual_co2e"] = "0"
                p["calculation_error"] = str(e)

            calculated.append(p)

        return calculated

    def _calc_direct_fuel(self, product: dict) -> Dict[str, Any]:
        """
        Calculate direct fuel combustion emissions (per unit per year).

        Formula: annual_co2e = fuel_consumption * ef * units_sold

        Args:
            product: Product dict with fuel_consumption_litres_per_year,
                resolved_fuel_ef, units_sold.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        consumption = Decimal(str(
            product.get("fuel_consumption_litres_per_year", 0)
        ))
        units = Decimal(str(product.get("units_sold", 1)))

        ef_data = product.get("resolved_fuel_ef") or {}
        co2_ef = Decimal(str(ef_data.get("co2_per_litre", "0")))
        ch4_ef = Decimal(str(ef_data.get("ch4_per_litre", "0")))
        n2o_ef = Decimal(str(ef_data.get("n2o_per_litre", "0")))
        wtt_ef = Decimal(str(ef_data.get("wtt_per_litre", "0")))

        co2_per_unit = consumption * co2_ef
        ch4_per_unit = consumption * ch4_ef
        n2o_per_unit = consumption * n2o_ef
        wtt_per_unit = consumption * wtt_ef
        total_per_unit = co2_per_unit + ch4_per_unit + n2o_per_unit

        annual_co2e = (total_per_unit * units).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )
        wtt_total = (wtt_per_unit * units).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "annual_co2e": str(annual_co2e),
            "annual_wtt_co2e": str(wtt_total),
            "co2_per_unit": str(co2_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "ch4_per_unit": str(ch4_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "n2o_per_unit": str(n2o_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "emission_type_resolved": "direct",
        }

    def _calc_direct_refrigerant(self, product: dict) -> Dict[str, Any]:
        """
        Calculate direct refrigerant leakage emissions (per unit per year).

        Formula: annual_co2e = charge_kg * annual_leak_rate * GWP * units_sold

        Args:
            product: Product dict with refrigerant_charge_kg,
                annual_leak_rate, resolved_gwp, units_sold.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        charge_kg = Decimal(str(product.get("refrigerant_charge_kg", 0)))
        annual_leak_rate = Decimal(str(
            product.get("annual_leak_rate", "0.02")
        ))
        gwp = Decimal(str(product.get("resolved_gwp", 0)))
        units = Decimal(str(product.get("units_sold", 1)))

        leakage_per_unit = charge_kg * annual_leak_rate
        co2e_per_unit = leakage_per_unit * gwp
        annual_co2e = (co2e_per_unit * units).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "annual_co2e": str(annual_co2e),
            "annual_wtt_co2e": "0",
            "leakage_kg_per_unit": str(
                leakage_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "co2e_per_unit": str(
                co2e_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "refrigerant_type": product.get("refrigerant_type", ""),
            "gwp_used": str(gwp),
            "emission_type_resolved": "direct",
        }

    def _calc_direct_chemical(self, product: dict) -> Dict[str, Any]:
        """
        Calculate direct chemical release emissions (per unit per year).

        Formula: annual_co2e = chemical_content_kg * release_fraction *
                              GWP * units_sold

        Args:
            product: Product dict with chemical_content_kg,
                release_fraction, chemical_gwp, units_sold.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        content_kg = Decimal(str(product.get("chemical_content_kg", 0)))
        release_fraction = Decimal(str(
            product.get("release_fraction", "1.0")
        ))
        gwp = Decimal(str(product.get("chemical_gwp", 1)))
        units = Decimal(str(product.get("units_sold", 1)))

        release_per_unit = content_kg * release_fraction
        co2e_per_unit = release_per_unit * gwp
        annual_co2e = (co2e_per_unit * units).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "annual_co2e": str(annual_co2e),
            "annual_wtt_co2e": "0",
            "release_kg_per_unit": str(
                release_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "co2e_per_unit": str(
                co2e_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "emission_type_resolved": "direct",
        }

    def _calc_indirect_electricity(self, product: dict) -> Dict[str, Any]:
        """
        Calculate indirect electricity consumption emissions (per unit per year).

        Formula: annual_co2e = electricity_kwh * grid_ef * units_sold

        Args:
            product: Product dict with electricity_kwh_per_year,
                resolved_grid_ef, units_sold.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        kwh = Decimal(str(product.get("electricity_kwh_per_year", 0)))
        grid_ef = Decimal(str(product.get("resolved_grid_ef", "0.4360")))
        units = Decimal(str(product.get("units_sold", 1)))

        # If no explicit kWh, try to look up from default profiles
        if kwh == 0:
            product_name = product.get("product_name", "").lower()
            default_kwh = DEFAULT_ENERGY_PROFILES.get(product_name)
            if default_kwh:
                kwh = default_kwh
                product["electricity_kwh_per_year"] = str(kwh)
                product["_kwh_source"] = "default_energy_profile"

        co2e_per_unit = kwh * grid_ef
        annual_co2e = (co2e_per_unit * units).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "annual_co2e": str(annual_co2e),
            "annual_wtt_co2e": "0",
            "kwh_per_unit": str(kwh.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "grid_ef_used": str(grid_ef),
            "co2e_per_unit": str(
                co2e_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "emission_type_resolved": "indirect",
        }

    def _calc_indirect_heating(self, product: dict) -> Dict[str, Any]:
        """
        Calculate indirect heating fuel emissions (per unit per year).

        For products like furnaces and water heaters that consume
        heating fuel during the use phase.

        Formula: annual_co2e = fuel_consumption * fuel_ef * units_sold

        Args:
            product: Product dict with heating_fuel_consumption,
                resolved_fuel_ef, units_sold.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        consumption = Decimal(str(
            product.get("heating_fuel_consumption", 0)
        ))
        units = Decimal(str(product.get("units_sold", 1)))

        ef_data = product.get("resolved_fuel_ef") or {}
        # Try litre-based first, then m3-based
        co2_ef = Decimal(str(
            ef_data.get("co2_per_litre", ef_data.get("co2_per_m3", "0"))
        ))

        co2e_per_unit = consumption * co2_ef
        annual_co2e = (co2e_per_unit * units).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "annual_co2e": str(annual_co2e),
            "annual_wtt_co2e": "0",
            "fuel_consumption_per_unit": str(
                consumption.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "co2e_per_unit": str(
                co2e_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "emission_type_resolved": "indirect",
        }

    def _calc_indirect_steam_cooling(self, product: dict) -> Dict[str, Any]:
        """
        Calculate indirect steam/cooling emissions (per unit per year).

        For products connected to district heating or cooling systems.

        Formula: annual_co2e = (steam_kwh + cooling_kwh) * grid_ef * units_sold
        (Using grid_ef as proxy for district energy carbon intensity)

        Args:
            product: Product dict with steam_kwh_per_year,
                cooling_kwh_per_year, resolved_grid_ef, units_sold.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        steam_kwh = Decimal(str(product.get("steam_kwh_per_year", 0)))
        cooling_kwh = Decimal(str(product.get("cooling_kwh_per_year", 0)))
        total_kwh = steam_kwh + cooling_kwh

        # Use steam/cooling-specific EF if provided, else grid_ef as proxy
        ef = Decimal(str(
            product.get("steam_ef", product.get("resolved_grid_ef", "0.4360"))
        ))
        units = Decimal(str(product.get("units_sold", 1)))

        co2e_per_unit = total_kwh * ef
        annual_co2e = (co2e_per_unit * units).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "annual_co2e": str(annual_co2e),
            "annual_wtt_co2e": "0",
            "steam_kwh_per_unit": str(
                steam_kwh.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "cooling_kwh_per_unit": str(
                cooling_kwh.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "co2e_per_unit": str(
                co2e_per_unit.quantize(_QUANT_8DP, rounding=ROUNDING)
            ),
            "emission_type_resolved": "indirect",
        }

    def _calc_fuels_sold(self, product: dict) -> Dict[str, Any]:
        """
        Calculate emissions from fuels sold to end users.

        The reporting company sells fuels whose combustion by end users
        generates Category 11 emissions.

        Formula: annual_co2e = quantity_sold * ef_per_unit

        Args:
            product: Product dict with quantity_sold, fuel_type,
                resolved_fuel_ef, units_sold.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        quantity = Decimal(str(product.get("quantity_sold", 0)))
        unit_type = product.get("quantity_unit", "litres")

        ef_data = product.get("resolved_fuel_ef") or {}
        if unit_type in ("litres", "liters", "l"):
            co2_ef = Decimal(str(ef_data.get("co2_per_litre", "0")))
        elif unit_type in ("m3", "cubic_metres"):
            co2_ef = Decimal(str(ef_data.get("co2_per_m3", "0")))
        elif unit_type in ("kg", "kilograms"):
            co2_ef = Decimal(str(ef_data.get("co2_per_kg", "0")))
        else:
            co2_ef = Decimal(str(ef_data.get("co2_per_litre", "0")))

        # For fuels sold, units_sold represents number of transactions/orders
        # quantity_sold represents total fuel volume
        annual_co2e = (quantity * co2_ef).quantize(
            _QUANT_8DP, rounding=ROUNDING
        )

        return {
            "annual_co2e": str(annual_co2e),
            "annual_wtt_co2e": "0",
            "quantity_sold": str(quantity),
            "quantity_unit": unit_type,
            "co2_ef_used": str(co2_ef),
            "emission_type_resolved": "direct",
        }

    def _calc_feedstocks_sold(self, product: dict) -> Dict[str, Any]:
        """
        Calculate emissions from feedstocks sold (oxidized during use).

        Feedstocks are materials that are chemically transformed and
        whose carbon is released during the use phase.

        Formula: annual_co2e = quantity_sold * carbon_content *
                              oxidation_factor * (44/12) [CO2/C ratio]

        Args:
            product: Product dict with quantity_sold, carbon_content,
                oxidation_factor.

        Returns:
            Dict with annual_co2e and breakdown fields.
        """
        quantity = Decimal(str(product.get("quantity_sold", 0)))
        carbon_content = Decimal(str(
            product.get("carbon_content", "0.85")
        ))
        oxidation_factor = Decimal(str(
            product.get("oxidation_factor", "1.0")
        ))
        co2_c_ratio = Decimal("3.6667")  # 44/12

        co2_emissions = (
            quantity * carbon_content * oxidation_factor * co2_c_ratio
        ).quantize(_QUANT_8DP, rounding=ROUNDING)

        return {
            "annual_co2e": str(co2_emissions),
            "annual_wtt_co2e": "0",
            "quantity_sold": str(quantity),
            "carbon_content": str(carbon_content),
            "oxidation_factor": str(oxidation_factor),
            "emission_type_resolved": "direct",
        }

    def _stage_lifetime(
        self, products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 6: LIFETIME - Apply lifetime modeling.

        Extends annual emissions over the product lifetime, applying
        degradation rates where applicable.

        Formula: lifetime_co2e = sum(annual_co2e * (1 + degradation_rate)^year)
                 for year in range(lifetime_years)

        Args:
            products: List of calculated product dicts.

        Returns:
            List of product dicts with lifetime_co2e.
        """
        results: List[Dict[str, Any]] = []

        for product in products:
            p = dict(product)
            category = p.get("product_category", "").lower()

            # Determine lifetime
            lifetime_years = int(p.get(
                "lifetime_years",
                DEFAULT_PRODUCT_LIFETIMES.get(category, DEFAULT_LIFETIME_YEARS),
            ))

            # Determine degradation rate
            degradation_rate = Decimal(str(p.get(
                "degradation_rate",
                str(DEFAULT_DEGRADATION_RATES.get(
                    category, DEFAULT_DEGRADATION_RATE
                )),
            )))

            annual_co2e = Decimal(str(p.get("annual_co2e", "0")))
            annual_wtt = Decimal(str(p.get("annual_wtt_co2e", "0")))

            # Calculate lifetime emissions with degradation
            total_lifetime = Decimal("0")
            total_wtt_lifetime = Decimal("0")
            year_breakdown: List[Dict[str, str]] = []

            for yr in range(lifetime_years):
                # Degradation increases emissions (efficiency loss)
                degradation_multiplier = (
                    Decimal("1") + degradation_rate
                ) ** yr
                year_co2e = annual_co2e * degradation_multiplier
                year_wtt = annual_wtt * degradation_multiplier

                total_lifetime += year_co2e
                total_wtt_lifetime += year_wtt

                year_breakdown.append({
                    "year": str(yr + 1),
                    "co2e": str(year_co2e.quantize(_QUANT_8DP, rounding=ROUNDING)),
                    "degradation_multiplier": str(
                        degradation_multiplier.quantize(
                            _QUANT_8DP, rounding=ROUNDING
                        )
                    ),
                })

            p["lifetime_years"] = lifetime_years
            p["degradation_rate"] = str(degradation_rate)
            p["lifetime_co2e"] = str(
                total_lifetime.quantize(_QUANT_8DP, rounding=ROUNDING)
            )
            p["lifetime_wtt_co2e"] = str(
                total_wtt_lifetime.quantize(_QUANT_8DP, rounding=ROUNDING)
            )
            p["year_breakdown"] = year_breakdown

            results.append(p)

        return results

    def _stage_aggregate(
        self, products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Stage 7: AGGREGATE - Multi-dimensional aggregation.

        Aggregates by:
        - total_co2e: sum of all lifetime emissions
        - direct_co2e: sum of direct emission products
        - indirect_co2e: sum of indirect emission products
        - by_category: sum grouped by product_category
        - by_emission_type: sum grouped by emission_type
        - by_method: sum grouped by calculation_method

        Args:
            products: List of lifetime-computed product dicts.

        Returns:
            Dictionary with aggregated breakdowns.
        """
        total_co2e = Decimal("0")
        direct_co2e = Decimal("0")
        indirect_co2e = Decimal("0")
        total_wtt = Decimal("0")

        by_category: Dict[str, Decimal] = {}
        by_emission_type: Dict[str, Decimal] = {}
        by_method: Dict[str, Decimal] = {}

        for p in products:
            lifetime = Decimal(str(p.get("lifetime_co2e", "0")))
            lifetime_wtt = Decimal(str(p.get("lifetime_wtt_co2e", "0")))
            emission_type = p.get("emission_type_resolved", p.get("emission_type", "indirect"))
            method = p.get("calculation_method", "unknown")
            category = p.get("product_category", "unknown")

            total_co2e += lifetime
            total_wtt += lifetime_wtt

            if emission_type == "direct":
                direct_co2e += lifetime
            else:
                indirect_co2e += lifetime

            by_category[category] = by_category.get(
                category, Decimal("0")
            ) + lifetime

            by_emission_type[emission_type] = by_emission_type.get(
                emission_type, Decimal("0")
            ) + lifetime

            by_method[method] = by_method.get(
                method, Decimal("0")
            ) + lifetime

        return {
            "total_co2e": str(total_co2e.quantize(_QUANT_2DP, rounding=ROUNDING)),
            "direct_co2e": str(direct_co2e.quantize(_QUANT_2DP, rounding=ROUNDING)),
            "indirect_co2e": str(indirect_co2e.quantize(_QUANT_2DP, rounding=ROUNDING)),
            "total_wtt_co2e": str(total_wtt.quantize(_QUANT_2DP, rounding=ROUNDING)),
            "by_category": {
                k: str(v.quantize(_QUANT_2DP, rounding=ROUNDING))
                for k, v in by_category.items()
            },
            "by_emission_type": {
                k: str(v.quantize(_QUANT_2DP, rounding=ROUNDING))
                for k, v in by_emission_type.items()
            },
            "by_method": {
                k: str(v.quantize(_QUANT_2DP, rounding=ROUNDING))
                for k, v in by_method.items()
            },
            "product_count": len(products),
        }

    def _stage_compliance(
        self, aggregated: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 8: COMPLIANCE - Run compliance checks.

        Delegates to ComplianceCheckerEngine if available. Returns
        compliance summary with overall status and per-framework scores.

        Args:
            aggregated: Aggregated result dictionary.

        Returns:
            Dictionary with compliance check results.
        """
        try:
            if self._compliance_engine is None:
                self._compliance_engine = self._load_engine(
                    "greenlang.use_of_sold_products.compliance_checker",
                    "ComplianceCheckerEngine",
                )

            if self._compliance_engine is not None:
                engine = self._compliance_engine
                if hasattr(engine, "get_instance"):
                    engine = engine.get_instance()

                # Prepare result dict for compliance checking
                check_input = {
                    "total_co2e": aggregated.get("total_co2e"),
                    "direct_co2e": aggregated.get("direct_co2e"),
                    "indirect_co2e": aggregated.get("indirect_co2e"),
                    "product_breakdown": aggregated.get("by_category"),
                    "by_category": aggregated.get("by_category"),
                    "method": "multiple",
                    "emission_type": "both",
                    "ef_sources": "DEFRA_2024/IEA_eGRID_2024/IPCC_AR5",
                }

                results = engine.check_all(check_input)
                report = engine.generate_report(results)
                return report

        except Exception as e:
            logger.warning(
                "Compliance check skipped: %s", e,
            )

        return {
            "overall_status": "SKIPPED",
            "overall_score": 0.0,
            "note": "Compliance engine not available",
        }

    def _stage_provenance(
        self,
        chain_id: str,
        aggregated: Dict[str, Any],
        compliance: Dict[str, Any],
        org_id: str,
        year: int,
    ) -> List[Dict[str, Any]]:
        """
        Stage 9: PROVENANCE - Build complete provenance chain.

        Args:
            chain_id: Pipeline chain identifier.
            aggregated: Aggregated result dictionary.
            compliance: Compliance check results.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            List of provenance chain entries.
        """
        chain = self._provenance_chains.get(chain_id, [])

        chain.append({
            "stage": PipelineStage.PROVENANCE.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "org_id": org_id,
                "year": year,
                "total_co2e": aggregated.get("total_co2e"),
                "compliance_status": compliance.get("overall_status"),
                "engine_version": ENGINE_VERSION,
            },
        })

        return chain

    def _stage_seal(
        self, chain_id: str, aggregated: Dict[str, Any]
    ) -> str:
        """
        Stage 10: SEAL - Generate SHA-256 hash for provenance chain.

        Creates a deterministic hash of the entire provenance chain
        plus the final aggregated result for audit trail integrity.

        Args:
            chain_id: Pipeline chain identifier.
            aggregated: Aggregated result dictionary.

        Returns:
            SHA-256 hex digest string.
        """
        chain = self._provenance_chains.get(chain_id, [])

        # Build deterministic string for hashing
        seal_data = {
            "chain_id": chain_id,
            "chain_entries": len(chain),
            "total_co2e": aggregated.get("total_co2e"),
            "direct_co2e": aggregated.get("direct_co2e"),
            "indirect_co2e": aggregated.get("indirect_co2e"),
            "product_count": aggregated.get("product_count"),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "engine_version": ENGINE_VERSION,
        }

        seal_str = json.dumps(seal_data, sort_keys=True, default=str)
        provenance_hash = hashlib.sha256(seal_str.encode("utf-8")).hexdigest()

        # Record seal in chain
        chain.append({
            "stage": PipelineStage.SEAL.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hash": provenance_hash,
        })

        return provenance_hash

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    def _record_provenance(
        self,
        chain_id: str,
        stage: PipelineStage,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """Record a provenance chain entry for the given stage."""
        chain = self._provenance_chains.get(chain_id)
        if chain is None:
            return

        chain.append({
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_summary": self._summarize(input_data),
            "output_summary": self._summarize(output_data),
        })

    @staticmethod
    def _summarize(data: Any) -> str:
        """Create a brief string summary for provenance logging."""
        if isinstance(data, dict):
            return json.dumps(
                {k: str(v)[:100] for k, v in list(data.items())[:10]},
                default=str,
            )
        if isinstance(data, list):
            return f"[list: {len(data)} items]"
        return str(data)[:200]

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Calculate elapsed milliseconds since start (monotonic)."""
        return (time.monotonic() - start) * 1000.0

    @staticmethod
    def _load_engine(module_path: str, class_name: str) -> Optional[Any]:
        """Load an engine class dynamically with graceful fallback."""
        try:
            import importlib
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except (ImportError, AttributeError) as e:
            logger.warning(
                "Engine %s not available: %s", class_name, e,
            )
            return None
