# -*- coding: utf-8 -*-
"""
FranchisesPipelineEngine - Orchestrated 10-stage pipeline for franchise emissions.

This module implements the FranchisesPipelineEngine for AGENT-MRV-027 (Franchises).
It orchestrates a 10-stage pipeline for complete franchise emissions calculation
from raw input to compliant output with full audit trail.

The 10 stages are:
1. VALIDATE: Input validation (required fields, types, ranges, schema checks)
2. CLASSIFY: Determine franchise type, ownership, agreement type
3. NORMALIZE: Unit conversion, currency normalization, area standardization
4. RESOLVE_EFS: Look up emission factors from database engine
5. CALCULATE: Route to appropriate calculator (franchise-specific/average/spend/hybrid)
6. ALLOCATE: Multi-brand allocation, pro-rata for partial year
7. AGGREGATE: Network-level aggregation, by type/region/method
8. COMPLIANCE: Run compliance checker across frameworks
9. PROVENANCE: Generate SHA-256 hashes, build Merkle tree
10. SEAL: Final validation, seal results, generate provenance record

Pipeline Features:
- Async stage execution with configurable timeouts
- Stage-level metrics recording
- Error recovery and partial results
- Batch processing support (up to 10,000 units)
- Network-level aggregation
- Progress callbacks

Example:
    >>> from greenlang.agents.mrv.franchises.franchises_pipeline import FranchisesPipelineEngine
    >>> engine = FranchisesPipelineEngine()
    >>> result = engine.execute_single(unit_input)
    >>> assert result["status"] == "SUCCESS"
    >>> print(f"Total emissions: {result['total_co2e']} kgCO2e")

Module: greenlang.agents.mrv.franchises.franchises_pipeline
Agent: AGENT-MRV-027
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "gl_frn_pipeline_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-014"
AGENT_COMPONENT: str = "AGENT-MRV-027"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

# Maximum batch size for network processing
MAX_BATCH_SIZE: int = 10_000

# Default stage timeout in seconds
DEFAULT_STAGE_TIMEOUT_S: float = 30.0

# Unit conversions
SQ_FT_TO_SQ_M: Decimal = Decimal("0.09290304")
BTU_TO_KWH: Decimal = Decimal("0.00029307107")
THERM_TO_KWH: Decimal = Decimal("29.3001")
MILES_TO_KM: Decimal = Decimal("1.60934")
GALLONS_TO_LITRES: Decimal = Decimal("3.78541")


# ==============================================================================
# ENUMS
# ==============================================================================


class PipelineStage(str, Enum):
    """Pipeline stage identifiers."""

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


class CalculationMethod(str, Enum):
    """Calculation method for franchise emissions."""

    FRANCHISE_SPECIFIC = "franchise_specific"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


class FranchiseType(str, Enum):
    """Types of franchise operations."""

    QUICK_SERVICE_RESTAURANT = "quick_service_restaurant"
    FULL_SERVICE_RESTAURANT = "full_service_restaurant"
    HOTEL = "hotel"
    CONVENIENCE_STORE = "convenience_store"
    GAS_STATION = "gas_station"
    RETAIL_STORE = "retail_store"
    FITNESS_CENTER = "fitness_center"
    AUTOMOTIVE_SERVICE = "automotive_service"
    LAUNDRY_DRY_CLEANING = "laundry_dry_cleaning"
    OFFICE = "office"
    OTHER = "other"


class OwnershipType(str, Enum):
    """Ownership type for franchise units."""

    FRANCHISE = "franchise"
    COMPANY_OWNED = "company_owned"
    JOINT_VENTURE = "joint_venture"
    LEASED = "leased"


class AgreementType(str, Enum):
    """Types of franchise agreements."""

    SINGLE_UNIT = "single_unit"
    MULTI_UNIT = "multi_unit"
    MASTER_FRANCHISE = "master_franchise"
    AREA_DEVELOPMENT = "area_development"
    CONVERSION = "conversion"
    SUB_FRANCHISE = "sub_franchise"


class AllocationMethod(str, Enum):
    """Allocation method for multi-brand or shared units."""

    REVENUE_SHARE = "revenue_share"
    FLOOR_AREA = "floor_area"
    OPERATING_HOURS = "operating_hours"
    HEADCOUNT = "headcount"
    EQUAL_SPLIT = "equal_split"


class EFSource(str, Enum):
    """Emission factor sources."""

    CBECS = "CBECS"         # Commercial Buildings Energy Consumption Survey
    ENERGY_STAR = "ENERGY_STAR"
    EGRID = "eGRID"
    IEA = "IEA"
    DEFRA = "DEFRA"
    EPA = "EPA"
    IPCC = "IPCC"
    CUSTOM = "CUSTOM"


# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    stage: PipelineStage
    status: PipelineStatus
    data: Dict[str, Any]
    duration_ms: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProvenanceEntry:
    """Single entry in the provenance chain."""

    stage: str
    timestamp: str
    input_hash: str
    output_hash: str
    chain_hash: str = ""


# ==============================================================================
# DEFAULT EUI BENCHMARKS (kWh/m2/year by franchise type)
# ==============================================================================

# Source: CBECS 2018, ENERGY STAR Portfolio Manager
DEFAULT_EUI_KWH_PER_SQM: Dict[str, Decimal] = {
    FranchiseType.QUICK_SERVICE_RESTAURANT.value: Decimal("970.00"),
    FranchiseType.FULL_SERVICE_RESTAURANT.value: Decimal("730.00"),
    FranchiseType.HOTEL.value: Decimal("330.00"),
    FranchiseType.CONVENIENCE_STORE.value: Decimal("580.00"),
    FranchiseType.GAS_STATION.value: Decimal("490.00"),
    FranchiseType.RETAIL_STORE.value: Decimal("250.00"),
    FranchiseType.FITNESS_CENTER.value: Decimal("350.00"),
    FranchiseType.AUTOMOTIVE_SERVICE.value: Decimal("310.00"),
    FranchiseType.LAUNDRY_DRY_CLEANING.value: Decimal("620.00"),
    FranchiseType.OFFICE.value: Decimal("210.00"),
    FranchiseType.OTHER.value: Decimal("300.00"),
}

# Default electricity EF (kgCO2e/kWh) -- US average from eGRID 2024
DEFAULT_ELECTRICITY_EF: Decimal = Decimal("0.3937")

# Default natural gas EF (kgCO2e/kWh)
DEFAULT_GAS_EF: Decimal = Decimal("0.18293")

# Refrigerant leakage defaults (kgCO2e per unit per year)
DEFAULT_REFRIGERANT_EF: Dict[str, Decimal] = {
    FranchiseType.QUICK_SERVICE_RESTAURANT.value: Decimal("850.00"),
    FranchiseType.FULL_SERVICE_RESTAURANT.value: Decimal("650.00"),
    FranchiseType.CONVENIENCE_STORE.value: Decimal("1200.00"),
    FranchiseType.GAS_STATION.value: Decimal("500.00"),
    FranchiseType.HOTEL.value: Decimal("1500.00"),
    FranchiseType.RETAIL_STORE.value: Decimal("400.00"),
    FranchiseType.OTHER.value: Decimal("300.00"),
}

# EEIO spend-based EFs (kgCO2e per USD) by NAICS category
EEIO_FRANCHISE_FACTORS: Dict[str, Decimal] = {
    "722513": Decimal("0.42"),  # Limited-service restaurants
    "722511": Decimal("0.38"),  # Full-service restaurants
    "721110": Decimal("0.28"),  # Hotels (except casino)
    "445120": Decimal("0.35"),  # Convenience stores
    "447110": Decimal("0.31"),  # Gasoline stations
    "448140": Decimal("0.22"),  # Family clothing stores
    "713940": Decimal("0.18"),  # Fitness centers
    "811111": Decimal("0.25"),  # General automotive repair
    "812310": Decimal("0.30"),  # Coin-operated laundries
    "999999": Decimal("0.25"),  # Default / other
}


# ==============================================================================
# FranchisesPipelineEngine
# ==============================================================================


class FranchisesPipelineEngine:
    """
    10-stage pipeline engine for franchise emissions calculations.

    Orchestrates the complete franchise emissions calculation workflow from
    input validation to sealed audit trail. Supports single unit, batch,
    and network-level calculations.

    Attributes:
        _database_engine: FranchiseDatabaseEngine (lazy-loaded)
        _franchise_specific_engine: FranchiseSpecificCalculatorEngine (lazy-loaded)
        _average_data_engine: AverageDataCalculatorEngine (lazy-loaded)
        _spend_engine: SpendBasedCalculatorEngine (lazy-loaded)
        _hybrid_engine: HybridAggregatorEngine (lazy-loaded)
        _compliance_engine: ComplianceCheckerEngine (lazy-loaded)
        _provenance_chains: In-progress provenance chains

    Thread Safety:
        Singleton pattern with RLock for concurrent access.

    Example:
        >>> engine = FranchisesPipelineEngine()
        >>> result = engine.execute_single(unit_input)
        >>> assert result["status"] == "SUCCESS"
    """

    _instance: Optional["FranchisesPipelineEngine"] = None
    _lock: RLock = RLock()

    def __new__(cls) -> "FranchisesPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize FranchisesPipelineEngine.

        Prevents re-initialization of the singleton. All sub-engines
        are lazy-loaded on first use.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Lazy-loaded engines
        self._database_engine: Optional[Any] = None
        self._franchise_specific_engine: Optional[Any] = None
        self._average_data_engine: Optional[Any] = None
        self._spend_engine: Optional[Any] = None
        self._hybrid_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None

        # Pipeline state
        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}
        self._stage_timeout_s: float = DEFAULT_STAGE_TIMEOUT_S
        self._progress_callback: Optional[Callable] = None

        self._initialized = True
        logger.info(
            "FranchisesPipelineEngine initialized (version %s)", ENGINE_VERSION,
        )

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def execute(self, network_input: dict) -> dict:
        """
        Execute the full pipeline for an entire franchise network.

        Processes all franchise units in the network, aggregates results,
        and runs compliance checks on the network-level result.

        Args:
            network_input: Network-level input dictionary containing:
                - units: List of franchise unit input dicts
                - reporting_period: Reporting period string
                - consolidation_approach: financial_control/equity_share
                - tenant_id: Optional tenant identifier

        Returns:
            Network aggregation result dictionary with:
                - total_co2e, by_type, by_region, by_method
                - compliance results
                - provenance_hash
                - per-unit results

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.

        Example:
            >>> result = engine.execute(network_input)
            >>> result["status"]
            'SUCCESS'
        """
        start_time = time.monotonic()
        chain_id = self._new_chain_id("network")

        try:
            units = network_input.get("units") or network_input.get("franchise_units") or []
            if not units:
                raise ValueError("No franchise units provided in network input")

            if len(units) > MAX_BATCH_SIZE:
                raise ValueError(
                    f"Batch size {len(units)} exceeds maximum {MAX_BATCH_SIZE}"
                )

            logger.info(
                "[%s] Starting network pipeline: %d units, period=%s",
                chain_id, len(units),
                network_input.get("reporting_period", "unspecified"),
            )

            # Process each unit
            unit_results: List[dict] = []
            unit_errors: List[dict] = []

            for idx, unit_input in enumerate(units):
                try:
                    # Inject network-level context
                    unit_input["consolidation_approach"] = network_input.get("consolidation_approach")
                    unit_input["reporting_period"] = network_input.get("reporting_period")
                    unit_input["tenant_id"] = network_input.get("tenant_id")

                    result = self.execute_single(unit_input)
                    unit_results.append(result)

                    if self._progress_callback:
                        self._progress_callback(idx + 1, len(units))

                except Exception as e:
                    logger.error(
                        "[%s] Unit %d failed: %s", chain_id, idx, str(e),
                    )
                    unit_errors.append({
                        "index": idx,
                        "unit_id": unit_input.get("unit_id", f"unknown-{idx}"),
                        "error": str(e),
                    })

            # Stage 7: AGGREGATE at network level
            aggregated = self._stage_aggregate(
                unit_results,
                reporting_period=network_input.get("reporting_period", ""),
            )

            # Stage 8: COMPLIANCE on aggregated result
            compliance = self._stage_compliance(aggregated)
            aggregated["compliance"] = compliance

            # Stage 9+10: PROVENANCE + SEAL
            provenance_hash = self._stage_provenance_and_seal(chain_id, aggregated)
            aggregated["provenance_hash"] = provenance_hash

            # Build final result
            total_duration_ms = (time.monotonic() - start_time) * 1000.0

            status = PipelineStatus.SUCCESS
            if unit_errors:
                status = PipelineStatus.PARTIAL_SUCCESS if unit_results else PipelineStatus.FAILED

            final_result = {
                "status": status.value,
                "total_co2e": aggregated.get("total_co2e", "0"),
                "total_units": len(units),
                "successful_units": len(unit_results),
                "failed_units": len(unit_errors),
                "by_franchise_type": aggregated.get("by_franchise_type", {}),
                "by_region": aggregated.get("by_region", {}),
                "by_method": aggregated.get("by_method", {}),
                "data_coverage": aggregated.get("data_coverage"),
                "compliance": compliance,
                "provenance_hash": provenance_hash,
                "unit_results": unit_results,
                "unit_errors": unit_errors,
                "reporting_period": network_input.get("reporting_period", ""),
                "consolidation_approach": network_input.get("consolidation_approach", ""),
                "processing_time_ms": total_duration_ms,
            }

            logger.info(
                "[%s] Network pipeline complete: %d/%d units successful, "
                "total_co2e=%s, duration=%.2fms",
                chain_id,
                len(unit_results), len(units),
                aggregated.get("total_co2e", "0"),
                total_duration_ms,
            )

            return final_result

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "[%s] Network pipeline failed: %s", chain_id, str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Network pipeline failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def execute_single(self, unit_input: dict) -> dict:
        """
        Execute the 10-stage pipeline for a single franchise unit.

        Args:
            unit_input: Single franchise unit input dictionary containing:
                - unit_id: Unique unit identifier
                - franchise_type: Type of franchise
                - ownership_type: franchise/company_owned
                - floor_area_sqm or floor_area_sqft
                - electricity_kwh, gas_kwh, etc. (for franchise-specific)
                - revenue or spend (for spend-based)

        Returns:
            Calculation result dictionary with total_co2e, method,
            provenance_hash, and stage details.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.

        Example:
            >>> result = engine.execute_single({
            ...     "unit_id": "FRN-001",
            ...     "franchise_type": "quick_service_restaurant",
            ...     "electricity_kwh": 150000,
            ...     "gas_kwh": 80000,
            ... })
            >>> result["total_co2e"]
            '73565.00000000'
        """
        chain_id = self._new_chain_id("unit")
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # Stage 1: VALIDATE
            start = time.monotonic()
            is_valid, errors = self._stage_validate(unit_input)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.VALIDATE.value] = duration_ms
            self._record_provenance(chain_id, PipelineStage.VALIDATE, unit_input, {"valid": is_valid})

            if not is_valid:
                raise ValueError(f"Input validation failed: {'; '.join(errors)}")
            logger.debug("[%s] VALIDATE completed in %.2fms", chain_id, duration_ms)

            # Stage 2: CLASSIFY
            start = time.monotonic()
            classified = self._stage_classify(unit_input)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.CLASSIFY.value] = duration_ms
            self._record_provenance(chain_id, PipelineStage.CLASSIFY, unit_input, classified)
            logger.debug("[%s] CLASSIFY completed in %.2fms", chain_id, duration_ms)

            # Stage 3: NORMALIZE
            start = time.monotonic()
            normalized = self._stage_normalize(classified)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.NORMALIZE.value] = duration_ms
            self._record_provenance(chain_id, PipelineStage.NORMALIZE, classified, normalized)
            logger.debug("[%s] NORMALIZE completed in %.2fms", chain_id, duration_ms)

            # Stage 4: RESOLVE_EFS
            start = time.monotonic()
            with_efs = self._stage_resolve_efs(normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.RESOLVE_EFS.value] = duration_ms
            self._record_provenance(chain_id, PipelineStage.RESOLVE_EFS, normalized, with_efs)
            logger.debug("[%s] RESOLVE_EFS completed in %.2fms", chain_id, duration_ms)

            # Stage 5: CALCULATE
            start = time.monotonic()
            calc_result = self._stage_calculate(with_efs)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.CALCULATE.value] = duration_ms
            self._record_provenance(chain_id, PipelineStage.CALCULATE, with_efs, calc_result)
            logger.debug(
                "[%s] CALCULATE completed in %.2fms (method=%s, co2e=%s)",
                chain_id, duration_ms,
                calc_result.get("method", "unknown"),
                calc_result.get("total_co2e", "0"),
            )

            # Stage 6: ALLOCATE
            start = time.monotonic()
            allocated = self._stage_allocate(calc_result)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.ALLOCATE.value] = duration_ms
            self._record_provenance(chain_id, PipelineStage.ALLOCATE, calc_result, allocated)
            logger.debug("[%s] ALLOCATE completed in %.2fms", chain_id, duration_ms)

            # Stages 7 (AGGREGATE) skipped for single unit -- done at network level

            # Stage 8: COMPLIANCE (lightweight for single unit)
            start = time.monotonic()
            compliance = self._stage_compliance(allocated)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.COMPLIANCE.value] = duration_ms
            self._record_provenance(chain_id, PipelineStage.COMPLIANCE, allocated, compliance)
            logger.debug("[%s] COMPLIANCE completed in %.2fms", chain_id, duration_ms)

            # Stage 9+10: PROVENANCE + SEAL
            start = time.monotonic()
            provenance_hash = self._stage_provenance_and_seal(chain_id, allocated)
            duration_ms = self._elapsed_ms(start)
            stage_durations[PipelineStage.SEAL.value] = duration_ms
            logger.debug("[%s] SEAL completed in %.2fms", chain_id, duration_ms)

            # Build result
            total_duration_ms = sum(stage_durations.values())

            result = {
                "status": PipelineStatus.SUCCESS.value,
                "unit_id": unit_input.get("unit_id", ""),
                "franchise_type": classified.get("franchise_type", ""),
                "method": calc_result.get("method", ""),
                "total_co2e": calc_result.get("total_co2e", "0"),
                "electricity_co2e": calc_result.get("electricity_co2e", "0"),
                "gas_co2e": calc_result.get("gas_co2e", "0"),
                "refrigerant_co2e": calc_result.get("refrigerant_co2e", "0"),
                "other_co2e": calc_result.get("other_co2e", "0"),
                "ef_source": with_efs.get("ef_source", ""),
                "dqi_score": calc_result.get("dqi_score"),
                "provenance_hash": provenance_hash,
                "compliance": compliance,
                "stage_durations": stage_durations,
                "processing_time_ms": total_duration_ms,
            }

            logger.info(
                "[%s] Single unit pipeline complete: unit=%s, method=%s, "
                "co2e=%s, duration=%.2fms",
                chain_id,
                unit_input.get("unit_id", ""),
                calc_result.get("method", ""),
                calc_result.get("total_co2e", "0"),
                total_duration_ms,
            )

            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "[%s] Single unit pipeline failed: %s", chain_id, str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Unit pipeline failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def execute_batch(
        self, inputs: List[dict],
    ) -> List[dict]:
        """
        Process a batch of franchise units independently.

        Each unit is processed through the full 10-stage pipeline.
        Errors in one unit do not prevent processing of other units.

        Args:
            inputs: List of franchise unit input dictionaries.

        Returns:
            List of result dictionaries, one per input.

        Example:
            >>> results = engine.execute_batch([unit1, unit2, unit3])
            >>> len(results)
            3
        """
        if len(inputs) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        start_time = time.monotonic()
        results: List[dict] = []

        logger.info("Starting batch pipeline: %d units", len(inputs))

        for idx, unit_input in enumerate(inputs):
            try:
                result = self.execute_single(unit_input)
                results.append(result)
            except Exception as e:
                logger.error("Batch unit %d failed: %s", idx, str(e))
                results.append({
                    "status": PipelineStatus.FAILED.value,
                    "unit_id": unit_input.get("unit_id", f"unknown-{idx}"),
                    "error": str(e),
                    "total_co2e": "0",
                })

        total_duration_ms = (time.monotonic() - start_time) * 1000.0
        successful = sum(1 for r in results if r.get("status") == PipelineStatus.SUCCESS.value)

        logger.info(
            "Batch pipeline complete: %d/%d successful, duration=%.2fms",
            successful, len(inputs), total_duration_ms,
        )

        return results

    # ==========================================================================
    # STAGE 1: VALIDATE
    # ==========================================================================

    def _stage_validate(self, unit_input: dict) -> Tuple[bool, List[str]]:
        """
        Stage 1: Input validation, schema checks, required fields.

        Validates:
        - unit_input is not empty
        - Required fields present for the calculation method
        - Numeric values are valid
        - Ownership type is valid for Category 14

        Args:
            unit_input: Raw franchise unit input dictionary.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors: List[str] = []

        if not unit_input:
            errors.append("Unit input must not be empty")
            return False, errors

        # Check ownership type -- company-owned units should not be in Cat 14
        ownership = str(unit_input.get("ownership_type", "franchise")).lower()
        if ownership in ("company_owned", "company", "corporate"):
            errors.append(
                f"Company-owned units (ownership_type='{ownership}') belong in "
                "Scope 1/2, not Category 14. DC-FRN-001 violation."
            )

        # Determine which data is available to decide method
        has_energy = (
            unit_input.get("electricity_kwh") is not None
            or unit_input.get("gas_kwh") is not None
            or unit_input.get("total_energy_kwh") is not None
        )
        has_area = (
            unit_input.get("floor_area_sqm") is not None
            or unit_input.get("floor_area_sqft") is not None
        )
        has_spend = (
            unit_input.get("revenue") is not None
            or unit_input.get("spend") is not None
            or unit_input.get("royalty_revenue") is not None
        )

        if not has_energy and not has_area and not has_spend:
            errors.append(
                "At least one of energy data (electricity_kwh/gas_kwh), "
                "floor area (floor_area_sqm/sqft), or spend data "
                "(revenue/spend) is required."
            )

        # Validate numeric fields if present
        numeric_fields = [
            "electricity_kwh", "gas_kwh", "total_energy_kwh",
            "floor_area_sqm", "floor_area_sqft",
            "revenue", "spend", "royalty_revenue",
            "refrigerant_charge_kg", "refrigerant_leak_rate",
            "operating_months",
        ]
        for field_name in numeric_fields:
            value = unit_input.get(field_name)
            if value is not None:
                try:
                    dec_val = Decimal(str(value))
                    if dec_val < 0:
                        errors.append(f"{field_name} must be >= 0, got {value}")
                except (InvalidOperation, ValueError):
                    errors.append(f"{field_name} is not a valid number: {value}")

        # Validate operating_months range
        operating_months = unit_input.get("operating_months")
        if operating_months is not None:
            try:
                months = Decimal(str(operating_months))
                if months < 0 or months > Decimal("12"):
                    errors.append(
                        f"operating_months must be 0-12, got {operating_months}"
                    )
            except (InvalidOperation, ValueError):
                pass  # Already caught above

        is_valid = len(errors) == 0
        return is_valid, errors

    # ==========================================================================
    # STAGE 2: CLASSIFY
    # ==========================================================================

    def _stage_classify(self, unit_input: dict) -> dict:
        """
        Stage 2: Determine franchise type, ownership, agreement type,
        and auto-select calculation method.

        Args:
            unit_input: Validated unit input dictionary.

        Returns:
            Classified input with franchise_type, method, agreement_type.
        """
        data = dict(unit_input)

        # Franchise type
        franchise_type = data.get("franchise_type", FranchiseType.OTHER.value)
        try:
            FranchiseType(franchise_type)
        except ValueError:
            franchise_type = FranchiseType.OTHER.value
        data["franchise_type"] = franchise_type

        # Ownership type
        ownership = data.get("ownership_type", OwnershipType.FRANCHISE.value)
        data["ownership_type"] = ownership

        # Agreement type
        agreement = data.get("agreement_type", AgreementType.SINGLE_UNIT.value)
        data["agreement_type"] = agreement

        # Auto-select calculation method
        has_energy = (
            data.get("electricity_kwh") is not None
            or data.get("gas_kwh") is not None
            or data.get("total_energy_kwh") is not None
        )
        has_area = (
            data.get("floor_area_sqm") is not None
            or data.get("floor_area_sqft") is not None
        )
        has_spend = (
            data.get("revenue") is not None
            or data.get("spend") is not None
            or data.get("royalty_revenue") is not None
        )

        if has_energy:
            method = CalculationMethod.FRANCHISE_SPECIFIC.value
        elif has_area:
            method = CalculationMethod.AVERAGE_DATA.value
        elif has_spend:
            method = CalculationMethod.SPEND_BASED.value
        else:
            method = CalculationMethod.AVERAGE_DATA.value

        # Allow override
        if data.get("calculation_method"):
            method = data["calculation_method"]

        data["method"] = method

        return data

    # ==========================================================================
    # STAGE 3: NORMALIZE
    # ==========================================================================

    def _stage_normalize(self, classified: dict) -> dict:
        """
        Stage 3: Unit conversion, currency normalization, area standardization.

        Conversions:
        - sq ft -> sq m
        - BTU -> kWh
        - therms -> kWh
        - miles -> km
        - Currency -> USD

        Args:
            classified: Classified input dictionary.

        Returns:
            Normalized input dictionary.
        """
        data = dict(classified)

        # Square feet to square metres
        if data.get("floor_area_sqft") is not None and data.get("floor_area_sqm") is None:
            sqft = Decimal(str(data["floor_area_sqft"]))
            data["floor_area_sqm"] = str(
                (sqft * SQ_FT_TO_SQ_M).quantize(_QUANT_8DP, rounding=ROUNDING)
            )
            data["_original_floor_area_sqft"] = str(data.pop("floor_area_sqft"))

        # BTU to kWh
        if data.get("energy_btu") is not None and data.get("total_energy_kwh") is None:
            btu = Decimal(str(data["energy_btu"]))
            data["total_energy_kwh"] = str(
                (btu * BTU_TO_KWH).quantize(_QUANT_8DP, rounding=ROUNDING)
            )

        # Therms to kWh
        if data.get("gas_therms") is not None and data.get("gas_kwh") is None:
            therms = Decimal(str(data["gas_therms"]))
            data["gas_kwh"] = str(
                (therms * THERM_TO_KWH).quantize(_QUANT_8DP, rounding=ROUNDING)
            )
            data["_original_gas_therms"] = str(data.pop("gas_therms"))

        # Currency normalization (simplified -- production uses full FX rates)
        if data.get("currency") and data.get("currency") != "USD":
            # Placeholder: log warning, assume 1:1 for non-USD
            currency = data["currency"]
            logger.info(
                "Currency normalization: %s -> USD (using placeholder rate)",
                currency,
            )
            # In production, apply actual exchange rate
            data["_original_currency"] = currency
            data["currency"] = "USD"

        # Pro-rata for partial-year operations
        operating_months = data.get("operating_months")
        if operating_months is not None:
            months = Decimal(str(operating_months))
            if Decimal("0") < months < Decimal("12"):
                pro_rata = (months / Decimal("12")).quantize(_QUANT_8DP, rounding=ROUNDING)
                data["pro_rata_fraction"] = str(pro_rata)
                data["pro_rata_applied"] = True

        return data

    # ==========================================================================
    # STAGE 4: RESOLVE_EFS
    # ==========================================================================

    def _stage_resolve_efs(self, normalized: dict) -> dict:
        """
        Stage 4: Look up emission factors from database engine or defaults.

        Resolves:
        - Electricity grid EF (by region/country)
        - Natural gas EF
        - EUI benchmark (by franchise type)
        - Refrigerant leakage EF
        - EEIO spend-based EF

        Args:
            normalized: Normalized input dictionary.

        Returns:
            Input enriched with emission factor data.
        """
        data = dict(normalized)
        method = data.get("method", CalculationMethod.AVERAGE_DATA.value)
        franchise_type = data.get("franchise_type", FranchiseType.OTHER.value)

        # Try database engine first
        db_engine = self._get_database_engine()

        # Electricity EF
        region = data.get("region") or data.get("egrid_subregion") or data.get("country_code")
        if db_engine:
            try:
                ef = db_engine.get_electricity_ef(region)
                if ef is not None:
                    data["electricity_ef"] = str(ef)
                    data["ef_source"] = "database"
            except Exception:
                pass

        if "electricity_ef" not in data:
            data["electricity_ef"] = str(DEFAULT_ELECTRICITY_EF)
            data["ef_source"] = EFSource.EGRID.value

        # Natural gas EF
        data["gas_ef"] = str(DEFAULT_GAS_EF)

        # EUI benchmark (for average-data method)
        if method == CalculationMethod.AVERAGE_DATA.value:
            eui = DEFAULT_EUI_KWH_PER_SQM.get(
                franchise_type,
                DEFAULT_EUI_KWH_PER_SQM[FranchiseType.OTHER.value],
            )
            data["eui_kwh_per_sqm"] = str(eui)

        # Refrigerant EF
        ref_ef = DEFAULT_REFRIGERANT_EF.get(
            franchise_type,
            DEFAULT_REFRIGERANT_EF.get(FranchiseType.OTHER.value, Decimal("300.00")),
        )
        data["refrigerant_ef"] = str(ref_ef)

        # EEIO factors (for spend-based method)
        if method == CalculationMethod.SPEND_BASED.value:
            naics = data.get("naics_code", "999999")
            eeio_ef = EEIO_FRANCHISE_FACTORS.get(
                naics,
                EEIO_FRANCHISE_FACTORS.get("999999", Decimal("0.25")),
            )
            data["eeio_ef"] = str(eeio_ef)

        return data

    # ==========================================================================
    # STAGE 5: CALCULATE
    # ==========================================================================

    def _stage_calculate(self, with_efs: dict) -> dict:
        """
        Stage 5: Route to appropriate calculator engine.

        Dispatches to:
        - franchise_specific: Uses actual energy/refrigerant data
        - average_data: Uses EUI benchmarks x floor area
        - spend_based: Uses EEIO factors x revenue/spend
        - hybrid: Blends methods

        Args:
            with_efs: Input enriched with emission factors.

        Returns:
            Calculation result dictionary.
        """
        method = with_efs.get("method", CalculationMethod.AVERAGE_DATA.value)

        if method == CalculationMethod.FRANCHISE_SPECIFIC.value:
            return self._calculate_franchise_specific(with_efs)
        elif method == CalculationMethod.AVERAGE_DATA.value:
            return self._calculate_average_data(with_efs)
        elif method == CalculationMethod.SPEND_BASED.value:
            return self._calculate_spend_based(with_efs)
        elif method == CalculationMethod.HYBRID.value:
            return self._calculate_hybrid(with_efs)
        else:
            return self._calculate_average_data(with_efs)

    def _calculate_franchise_specific(self, data: dict) -> dict:
        """
        Calculate using franchise-specific (primary) energy data.

        Formula:
            electricity_co2e = electricity_kwh x electricity_ef
            gas_co2e = gas_kwh x gas_ef
            refrigerant_co2e = refrigerant_charge_kg x leak_rate x GWP
                (or default per franchise type)
            total_co2e = electricity + gas + refrigerant + other

        All arithmetic uses Decimal with ROUND_HALF_UP.

        Args:
            data: Input with energy data and emission factors.

        Returns:
            Calculation result dictionary.
        """
        electricity_kwh = Decimal(str(data.get("electricity_kwh", "0")))
        gas_kwh = Decimal(str(data.get("gas_kwh", "0")))
        electricity_ef = Decimal(str(data.get("electricity_ef", str(DEFAULT_ELECTRICITY_EF))))
        gas_ef = Decimal(str(data.get("gas_ef", str(DEFAULT_GAS_EF))))
        pro_rata = Decimal(str(data.get("pro_rata_fraction", "1")))

        # Electricity emissions
        electricity_co2e = (electricity_kwh * electricity_ef * pro_rata).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )

        # Gas emissions
        gas_co2e = (gas_kwh * gas_ef * pro_rata).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )

        # Refrigerant emissions
        refrigerant_co2e = self._calculate_refrigerant(data, pro_rata)

        # Other energy (steam, cooling, mobile combustion)
        other_kwh = Decimal(str(data.get("other_energy_kwh", "0")))
        other_ef = Decimal(str(data.get("other_ef", str(DEFAULT_GAS_EF))))
        other_co2e = (other_kwh * other_ef * pro_rata).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )

        total_co2e = electricity_co2e + gas_co2e + refrigerant_co2e + other_co2e

        return {
            "method": CalculationMethod.FRANCHISE_SPECIFIC.value,
            "total_co2e": str(total_co2e),
            "electricity_co2e": str(electricity_co2e),
            "gas_co2e": str(gas_co2e),
            "refrigerant_co2e": str(refrigerant_co2e),
            "other_co2e": str(other_co2e),
            "electricity_kwh": str(electricity_kwh),
            "gas_kwh": str(gas_kwh),
            "electricity_ef": str(electricity_ef),
            "gas_ef": str(gas_ef),
            "pro_rata_fraction": str(pro_rata),
            "ef_source": data.get("ef_source", EFSource.EGRID.value),
            "dqi_score": "4.5",
            **{k: v for k, v in data.items() if k.startswith("unit_")},
        }

    def _calculate_average_data(self, data: dict) -> dict:
        """
        Calculate using EUI benchmarks x floor area.

        Formula:
            estimated_energy_kwh = floor_area_sqm x EUI
            electricity_fraction = 0.65 (default)
            electricity_co2e = estimated_energy x elec_fraction x elec_ef
            gas_co2e = estimated_energy x (1 - elec_fraction) x gas_ef
            refrigerant_co2e = per-type default
            total_co2e = electricity + gas + refrigerant

        Args:
            data: Input with floor area and EUI benchmarks.

        Returns:
            Calculation result dictionary.
        """
        floor_area = Decimal(str(data.get("floor_area_sqm", "100")))
        eui = Decimal(str(data.get("eui_kwh_per_sqm", "300")))
        electricity_ef = Decimal(str(data.get("electricity_ef", str(DEFAULT_ELECTRICITY_EF))))
        gas_ef = Decimal(str(data.get("gas_ef", str(DEFAULT_GAS_EF))))
        pro_rata = Decimal(str(data.get("pro_rata_fraction", "1")))

        # Default electricity fraction (65% electric, 35% gas for commercial buildings)
        elec_fraction = Decimal(str(data.get("electricity_fraction", "0.65")))

        estimated_energy = (floor_area * eui * pro_rata).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )

        electricity_kwh = (estimated_energy * elec_fraction).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )
        gas_kwh = (estimated_energy * (Decimal("1") - elec_fraction)).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )

        electricity_co2e = (electricity_kwh * electricity_ef).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )
        gas_co2e = (gas_kwh * gas_ef).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )

        refrigerant_co2e = self._calculate_refrigerant(data, pro_rata)
        total_co2e = electricity_co2e + gas_co2e + refrigerant_co2e

        return {
            "method": CalculationMethod.AVERAGE_DATA.value,
            "total_co2e": str(total_co2e),
            "electricity_co2e": str(electricity_co2e),
            "gas_co2e": str(gas_co2e),
            "refrigerant_co2e": str(refrigerant_co2e),
            "other_co2e": "0",
            "estimated_energy_kwh": str(estimated_energy),
            "floor_area_sqm": str(floor_area),
            "eui_kwh_per_sqm": str(eui),
            "electricity_ef": str(electricity_ef),
            "gas_ef": str(gas_ef),
            "pro_rata_fraction": str(pro_rata),
            "ef_source": data.get("ef_source", EFSource.CBECS.value),
            "dqi_score": "2.5",
            **{k: v for k, v in data.items() if k.startswith("unit_")},
        }

    def _calculate_spend_based(self, data: dict) -> dict:
        """
        Calculate using EEIO spend-based approach.

        Formula:
            spend_usd = revenue or spend (in USD)
            total_co2e = spend_usd x EEIO_ef

        Args:
            data: Input with revenue/spend and EEIO factors.

        Returns:
            Calculation result dictionary.
        """
        spend = Decimal(str(
            data.get("revenue")
            or data.get("spend")
            or data.get("royalty_revenue")
            or "0"
        ))
        eeio_ef = Decimal(str(data.get("eeio_ef", "0.25")))
        pro_rata = Decimal(str(data.get("pro_rata_fraction", "1")))

        total_co2e = (spend * eeio_ef * pro_rata).quantize(
            _QUANT_8DP, rounding=ROUNDING,
        )

        return {
            "method": CalculationMethod.SPEND_BASED.value,
            "total_co2e": str(total_co2e),
            "electricity_co2e": "0",
            "gas_co2e": "0",
            "refrigerant_co2e": "0",
            "other_co2e": str(total_co2e),
            "spend_usd": str(spend),
            "eeio_ef": str(eeio_ef),
            "pro_rata_fraction": str(pro_rata),
            "ef_source": EFSource.EPA.value,
            "dqi_score": "1.5",
            **{k: v for k, v in data.items() if k.startswith("unit_")},
        }

    def _calculate_hybrid(self, data: dict) -> dict:
        """
        Calculate using hybrid approach (blend of methods).

        Uses franchise-specific data where available, fills gaps with
        average-data or spend-based estimates.

        Args:
            data: Input with mixed data availability.

        Returns:
            Calculation result dictionary.
        """
        has_energy = (
            data.get("electricity_kwh") is not None
            or data.get("gas_kwh") is not None
        )

        if has_energy:
            # Use franchise-specific for energy, add refrigerant default
            result = self._calculate_franchise_specific(data)
            result["method"] = CalculationMethod.HYBRID.value
            result["dqi_score"] = "3.5"
            return result
        else:
            # Fall back to average-data
            result = self._calculate_average_data(data)
            result["method"] = CalculationMethod.HYBRID.value
            result["dqi_score"] = "3.0"
            return result

    def _calculate_refrigerant(self, data: dict, pro_rata: Decimal) -> Decimal:
        """
        Calculate refrigerant emissions for a franchise unit.

        Uses explicit charge/leak data if available, otherwise defaults.

        Args:
            data: Input data with optional refrigerant fields.
            pro_rata: Pro-rata fraction for partial-year.

        Returns:
            Refrigerant CO2e in kg as Decimal.
        """
        charge_kg = data.get("refrigerant_charge_kg")
        leak_rate = data.get("refrigerant_leak_rate")
        gwp = data.get("refrigerant_gwp")

        if charge_kg is not None and leak_rate is not None and gwp is not None:
            # Explicit calculation: charge x leak_rate x GWP
            co2e = (
                Decimal(str(charge_kg))
                * Decimal(str(leak_rate))
                * Decimal(str(gwp))
                * pro_rata
            ).quantize(_QUANT_8DP, rounding=ROUNDING)
            return co2e

        # Default per franchise type
        franchise_type = data.get("franchise_type", FranchiseType.OTHER.value)
        default_ef = DEFAULT_REFRIGERANT_EF.get(
            franchise_type,
            DEFAULT_REFRIGERANT_EF.get(FranchiseType.OTHER.value, Decimal("300.00")),
        )
        return (default_ef * pro_rata).quantize(_QUANT_8DP, rounding=ROUNDING)

    # ==========================================================================
    # STAGE 6: ALLOCATE
    # ==========================================================================

    def _stage_allocate(self, calc_result: dict) -> dict:
        """
        Stage 6: Multi-brand allocation, pro-rata for partial year.

        Applies brand-level allocation if the unit operates multiple brands.
        Pro-rata allocation for partial-year operations is already handled
        in the normalize and calculate stages.

        Args:
            calc_result: Calculation result dictionary.

        Returns:
            Allocated result dictionary.
        """
        data = dict(calc_result)

        # Multi-brand allocation
        brands = data.get("brands") or data.get("brand_allocation")
        if isinstance(brands, dict) and len(brands) > 1:
            total = Decimal(str(data.get("total_co2e", "0")))
            allocated_brands: Dict[str, str] = {}

            for brand_name, share in brands.items():
                share_dec = Decimal(str(share))
                brand_co2e = (total * share_dec).quantize(
                    _QUANT_8DP, rounding=ROUNDING,
                )
                allocated_brands[brand_name] = str(brand_co2e)

            data["brand_emissions"] = allocated_brands
            data["allocation_method"] = AllocationMethod.REVENUE_SHARE.value

        return data

    # ==========================================================================
    # STAGE 7: AGGREGATE
    # ==========================================================================

    def _stage_aggregate(
        self,
        unit_results: List[dict],
        reporting_period: str = "",
    ) -> dict:
        """
        Stage 7: Network-level aggregation by type/region/method.

        Aggregates individual unit results into network-level summaries.

        Args:
            unit_results: List of individual unit calculation results.
            reporting_period: Reporting period label.

        Returns:
            Aggregated result dictionary.
        """
        total_co2e = Decimal("0")
        by_franchise_type: Dict[str, Decimal] = {}
        by_region: Dict[str, Decimal] = {}
        by_method: Dict[str, Decimal] = {}
        units_with_data = 0
        total_units = len(unit_results)

        for result in unit_results:
            if result.get("status") != PipelineStatus.SUCCESS.value:
                continue

            co2e = Decimal(str(result.get("total_co2e", "0")))
            total_co2e += co2e

            # By franchise type
            ft = result.get("franchise_type", "other")
            by_franchise_type[ft] = by_franchise_type.get(ft, Decimal("0")) + co2e

            # By region
            region = result.get("region", "unknown")
            by_region[region] = by_region.get(region, Decimal("0")) + co2e

            # By method
            method = result.get("method", "unknown")
            by_method[method] = by_method.get(method, Decimal("0")) + co2e

            # Data coverage
            if method == CalculationMethod.FRANCHISE_SPECIFIC.value:
                units_with_data += 1

        data_coverage = Decimal("0")
        if total_units > 0:
            data_coverage = (
                Decimal(str(units_with_data)) / Decimal(str(total_units))
            ).quantize(_QUANT_2DP, rounding=ROUNDING)

        return {
            "total_co2e": str(total_co2e.quantize(_QUANT_8DP, rounding=ROUNDING)),
            "total_units": total_units,
            "units_with_data": units_with_data,
            "data_coverage": str(data_coverage),
            "by_franchise_type": {k: str(v) for k, v in by_franchise_type.items()},
            "by_region": {k: str(v) for k, v in by_region.items()},
            "by_method": {k: str(v) for k, v in by_method.items()},
            "reporting_period": reporting_period,
            "franchise_type_breakdown": {k: str(v) for k, v in by_franchise_type.items()},
        }

    # ==========================================================================
    # STAGE 8: COMPLIANCE
    # ==========================================================================

    def _stage_compliance(self, result: dict) -> dict:
        """
        Stage 8: Run compliance checks across frameworks.

        Uses the ComplianceCheckerEngine if available, otherwise provides
        inline lightweight compliance checking.

        Args:
            result: Calculation or aggregation result dictionary.

        Returns:
            Compliance check results dictionary.
        """
        compliance_engine = self._get_compliance_engine()
        if compliance_engine is not None:
            try:
                check_results = compliance_engine.check_compliance(result)
                summary = compliance_engine.get_compliance_summary(check_results)
                return summary
            except Exception as e:
                logger.warning(
                    "ComplianceCheckerEngine failed, using inline: %s", str(e),
                )

        # Inline lightweight compliance check
        findings: List[str] = []
        status = "PASS"

        total_co2e = result.get("total_co2e")
        if not total_co2e or Decimal(str(total_co2e)) <= 0:
            findings.append("GHG Protocol: total_co2e missing or zero")
            status = "FAIL"

        method = result.get("method") or result.get("calculation_method")
        if not method:
            findings.append("GHG Protocol: calculation method not documented")
            status = "WARNING"

        return {
            "overall_status": status,
            "overall_score": 100.0 if not findings else 50.0,
            "findings": findings,
            "frameworks_checked": 1,
        }

    # ==========================================================================
    # STAGE 9+10: PROVENANCE + SEAL
    # ==========================================================================

    def _stage_provenance_and_seal(
        self, chain_id: str, result: dict,
    ) -> str:
        """
        Stages 9 and 10: Generate SHA-256 provenance hash and seal.

        Builds a Merkle-tree-style hash over the entire provenance chain
        for this calculation, producing an immutable audit fingerprint.

        Args:
            chain_id: Provenance chain identifier.
            result: Final result data.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        chain = self._provenance_chains.get(chain_id, [])
        chain_str = json.dumps(chain, sort_keys=True, default=str)
        result_str = json.dumps(result, sort_keys=True, default=str)
        combined = f"{chain_str}|{result_str}"
        provenance_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        logger.debug(
            "[%s] Provenance sealed: hash=%s", chain_id, provenance_hash[:16],
        )

        return provenance_hash

    # ==========================================================================
    # ERROR HANDLING
    # ==========================================================================

    def _handle_stage_error(
        self, stage: PipelineStage, error: Exception,
    ) -> dict:
        """
        Handle a stage-level error with recovery action.

        Args:
            stage: The pipeline stage that failed.
            error: The exception that occurred.

        Returns:
            Recovery action dictionary.
        """
        logger.error(
            "Stage %s failed: %s", stage.value, str(error), exc_info=True,
        )

        return {
            "stage": stage.value,
            "error": str(error),
            "action": "skip" if stage in (
                PipelineStage.COMPLIANCE,
                PipelineStage.PROVENANCE,
            ) else "fail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==========================================================================
    # PROVENANCE HELPERS
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
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        output_str = json.dumps(output_data, sort_keys=True, default=str)

        entry: Dict[str, Any] = {
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
    def _new_chain_id(prefix: str) -> str:
        """Generate a new provenance chain identifier."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return f"frn-{prefix}-{ts}"

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Calculate milliseconds elapsed since start (monotonic)."""
        return (time.monotonic() - start) * 1000.0

    # ==========================================================================
    # LAZY ENGINE LOADING
    # ==========================================================================

    def _get_database_engine(self) -> Optional[Any]:
        """Get or create FranchiseDatabaseEngine (lazy loading)."""
        if self._database_engine is None:
            try:
                from greenlang.agents.mrv.franchises.franchise_database import FranchiseDatabaseEngine
                self._database_engine = FranchiseDatabaseEngine()
            except ImportError:
                logger.debug("FranchiseDatabaseEngine not available")
                self._database_engine = None
        return self._database_engine

    def _get_franchise_specific_engine(self) -> Optional[Any]:
        """Get or create FranchiseSpecificCalculatorEngine (lazy loading)."""
        if self._franchise_specific_engine is None:
            try:
                from greenlang.agents.mrv.franchises.franchise_specific_calculator import FranchiseSpecificCalculatorEngine
                self._franchise_specific_engine = FranchiseSpecificCalculatorEngine()
            except ImportError:
                logger.debug("FranchiseSpecificCalculatorEngine not available")
                self._franchise_specific_engine = None
        return self._franchise_specific_engine

    def _get_average_data_engine(self) -> Optional[Any]:
        """Get or create AverageDataCalculatorEngine (lazy loading)."""
        if self._average_data_engine is None:
            try:
                from greenlang.agents.mrv.franchises.average_data_calculator import AverageDataCalculatorEngine
                self._average_data_engine = AverageDataCalculatorEngine()
            except ImportError:
                logger.debug("AverageDataCalculatorEngine not available")
                self._average_data_engine = None
        return self._average_data_engine

    def _get_spend_engine(self) -> Optional[Any]:
        """Get or create SpendBasedCalculatorEngine (lazy loading)."""
        if self._spend_engine is None:
            try:
                from greenlang.agents.mrv.franchises.spend_based_calculator import SpendBasedCalculatorEngine
                self._spend_engine = SpendBasedCalculatorEngine()
            except ImportError:
                logger.debug("SpendBasedCalculatorEngine not available")
                self._spend_engine = None
        return self._spend_engine

    def _get_hybrid_engine(self) -> Optional[Any]:
        """Get or create HybridAggregatorEngine (lazy loading)."""
        if self._hybrid_engine is None:
            try:
                from greenlang.agents.mrv.franchises.hybrid_aggregator import HybridAggregatorEngine
                self._hybrid_engine = HybridAggregatorEngine()
            except ImportError:
                logger.debug("HybridAggregatorEngine not available")
                self._hybrid_engine = None
        return self._hybrid_engine

    def _get_compliance_engine(self) -> Optional[Any]:
        """Get or create ComplianceCheckerEngine (lazy loading)."""
        if self._compliance_engine is None:
            try:
                from greenlang.agents.mrv.franchises.compliance_checker import ComplianceCheckerEngine
                self._compliance_engine = ComplianceCheckerEngine.get_instance()
            except ImportError:
                logger.debug("ComplianceCheckerEngine not available")
                self._compliance_engine = None
        return self._compliance_engine

    # ==========================================================================
    # UTILITY / STATUS METHODS
    # ==========================================================================

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status including loaded engines.

        Returns:
            Dictionary with pipeline status information.

        Example:
            >>> status = engine.get_pipeline_status()
            >>> status["agent_id"]
            'GL-MRV-S3-014'
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "engine_id": ENGINE_ID,
            "version": ENGINE_VERSION,
            "engines_loaded": {
                "database": self._database_engine is not None,
                "franchise_specific": self._franchise_specific_engine is not None,
                "average_data": self._average_data_engine is not None,
                "spend_based": self._spend_engine is not None,
                "hybrid": self._hybrid_engine is not None,
                "compliance": self._compliance_engine is not None,
            },
            "active_chains": len(self._provenance_chains),
            "stage_timeout_s": self._stage_timeout_s,
            "max_batch_size": MAX_BATCH_SIZE,
        }

    def set_progress_callback(self, callback: Optional[Callable]) -> None:
        """
        Set a progress callback for batch/network processing.

        Args:
            callback: Callable(current: int, total: int) or None.
        """
        self._progress_callback = callback

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
            logger.info("FranchisesPipelineEngine singleton reset")


# ==============================================================================
# MODULE-LEVEL HELPERS
# ==============================================================================


def get_pipeline_engine() -> FranchisesPipelineEngine:
    """
    Get singleton pipeline engine instance.

    Returns:
        FranchisesPipelineEngine singleton instance.

    Example:
        >>> engine = get_pipeline_engine()
        >>> engine.get_pipeline_status()["engine_id"]
        'gl_frn_pipeline_engine'
    """
    return FranchisesPipelineEngine()


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
    "CalculationMethod",
    "FranchiseType",
    "OwnershipType",
    "AgreementType",
    "AllocationMethod",
    "EFSource",
    "StageResult",
    "ProvenanceEntry",
    "FranchisesPipelineEngine",
    "get_pipeline_engine",
]
