"""
TransportPipelineEngine - Orchestrated 10-stage pipeline for upstream transportation emissions.

This module implements the TransportPipelineEngine for AGENT-MRV-017 (Upstream Transportation).
It orchestrates a 10-stage pipeline for complete transport emissions calculation from raw input
to compliant output with full audit trail.

The 10 stages are:
1. VALIDATE: Input validation (required fields, types, ranges)
2. CLASSIFY: Determine transport mode, vehicle/vessel type, method selection, Incoterm mapping
3. NORMALIZE: Unit conversion (mass→tonnes, distance→km, currency→USD)
4. RESOLVE_EFS: Select emission factors from hierarchy (supplier→DEFRA→EPA→GLEC→EEIO)
5. CALCULATE_LEGS: Per-leg emissions using selected method (distance/fuel/spend/supplier)
6. CALCULATE_HUBS: Hub/transshipment/warehousing emissions
7. ALLOCATE: Apply allocation method for shared transport
8. COMPLIANCE: Framework compliance checking (7 frameworks)
9. AGGREGATE: Sum by mode, carrier, route, period
10. SEAL: Provenance hash, audit trail

Example:
    >>> from greenlang.agents.mrv.upstream_transportation.transport_pipeline import TransportPipelineEngine
    >>> engine = TransportPipelineEngine(config)
    >>> result = engine.execute(calculation_request)
    >>> assert result.status == "SUCCESS"
    >>> print(f"Total emissions: {result.total_emissions_kg_co2e} kg CO2e")

Module: greenlang.agents.mrv.upstream_transportation.transport_pipeline
Agent: AGENT-MRV-017
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from decimal import Decimal
import logging
import hashlib
import json
from threading import RLock
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator

from greenlang.agents.mrv.upstream_transportation.transport_database import (
    TransportDatabaseEngine,
    TransportMode,
    VehicleType,
    EmissionFactorSource,
    TransportMethodType,
    AllocationMethod,
    IncotermCategory,
)
from greenlang.agents.mrv.upstream_transportation.distance_calculator import (
    DistanceBasedCalculatorEngine,
    DistanceCalculationRequest,
)
from greenlang.agents.mrv.upstream_transportation.fuel_calculator import (
    FuelBasedCalculatorEngine,
    FuelCalculationRequest,
)
from greenlang.agents.mrv.upstream_transportation.spend_calculator import (
    SpendBasedCalculatorEngine,
    SpendCalculationRequest,
)
from greenlang.agents.mrv.upstream_transportation.multileg_calculator import (
    MultiLegCalculatorEngine,
    MultiLegRequest,
    LegDefinition,
)
from greenlang.agents.mrv.upstream_transportation.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceCheckRequest,
    ComplianceCheckResult,
    RegulatoryFramework,
)


logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline stage enumeration."""
    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE_LEGS = "CALCULATE_LEGS"
    CALCULATE_HUBS = "CALCULATE_HUBS"
    ALLOCATE = "ALLOCATE"
    COMPLIANCE = "COMPLIANCE"
    AGGREGATE = "AGGREGATE"
    SEAL = "SEAL"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class LegCalculationResult(BaseModel):
    """Result for a single transport leg calculation."""
    leg_id: str = Field(..., description="Unique leg identifier")
    mode: TransportMode = Field(..., description="Transport mode")
    vehicle_type: Optional[str] = Field(None, description="Vehicle/vessel type")
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    distance_km: Decimal = Field(..., ge=0, description="Distance in km")
    mass_tonnes: Decimal = Field(..., ge=0, description="Mass transported in tonnes")
    emissions_kg_co2: Decimal = Field(..., ge=0, description="CO2 emissions (kg)")
    emissions_kg_ch4: Decimal = Field(..., ge=0, description="CH4 emissions (kg)")
    emissions_kg_n2o: Decimal = Field(..., ge=0, description="N2O emissions (kg)")
    emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total CO2e emissions (kg)")
    emission_factor_source: EmissionFactorSource = Field(..., description="EF source")
    method_used: TransportMethodType = Field(..., description="Calculation method")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HubCalculationResult(BaseModel):
    """Result for hub/warehouse/transshipment emissions."""
    hub_id: str = Field(..., description="Unique hub identifier")
    hub_type: str = Field(..., description="Hub type (warehouse/transshipment/terminal)")
    location: str = Field(..., description="Hub location")
    throughput_tonnes: Decimal = Field(..., ge=0, description="Throughput in tonnes")
    storage_days: Optional[Decimal] = Field(None, ge=0, description="Storage duration (days)")
    emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total CO2e emissions (kg)")
    emission_factor: Decimal = Field(..., description="EF used (kg CO2e/tonne or kg CO2e/tonne-day)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AggregationResult(BaseModel):
    """Aggregated emissions across multiple dimensions."""
    by_mode: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by transport mode")
    by_carrier: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by carrier")
    by_route: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by route")
    by_period: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by time period")
    by_supplier: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by supplier")
    total_distance_km: Decimal = Field(..., ge=0, description="Total distance")
    total_mass_tonnes: Decimal = Field(..., ge=0, description="Total mass transported")
    total_emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total emissions")


class CalculationRequest(BaseModel):
    """Request model for transport emissions calculation."""
    calculation_id: str = Field(..., description="Unique calculation identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    reporting_period_start: datetime = Field(..., description="Reporting period start")
    reporting_period_end: datetime = Field(..., description="Reporting period end")

    # Transport details
    mode: Optional[TransportMode] = Field(None, description="Transport mode (auto-detected if not provided)")
    vehicle_type: Optional[str] = Field(None, description="Vehicle/vessel type")
    incoterm: Optional[str] = Field(None, description="Incoterm (e.g., FOB, CIF)")

    # Shipment details
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    mass_kg: Optional[Decimal] = Field(None, ge=0, description="Mass in kg")
    distance_km: Optional[Decimal] = Field(None, ge=0, description="Distance in km")
    fuel_liters: Optional[Decimal] = Field(None, ge=0, description="Fuel consumption in liters")
    fuel_type: Optional[str] = Field(None, description="Fuel type")
    spend_amount: Optional[Decimal] = Field(None, ge=0, description="Transportation spend")
    spend_currency: Optional[str] = Field(None, description="Spend currency")

    # Multi-leg support
    legs: Optional[List[Dict[str, Any]]] = Field(None, description="Multi-leg journey definitions")
    hubs: Optional[List[Dict[str, Any]]] = Field(None, description="Hub/warehouse/transshipment points")

    # Allocation (for shared transport)
    allocation_method: Optional[AllocationMethod] = Field(None, description="Allocation method")
    allocation_factor: Optional[Decimal] = Field(None, ge=0, le=1, description="Allocation factor (0-1)")

    # Compliance
    frameworks: List[RegulatoryFramework] = Field(
        default_factory=lambda: [RegulatoryFramework.GHG_PROTOCOL],
        description="Regulatory frameworks to check"
    )

    # Metadata
    supplier_id: Optional[str] = Field(None, description="Supplier identifier")
    carrier_id: Optional[str] = Field(None, description="Carrier identifier")
    route_id: Optional[str] = Field(None, description="Route identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CalculationResult(BaseModel):
    """Result model for transport emissions calculation."""
    calculation_id: str = Field(..., description="Calculation identifier")
    status: PipelineStatus = Field(..., description="Pipeline execution status")
    tenant_id: str = Field(..., description="Tenant identifier")

    # Results
    leg_results: List[LegCalculationResult] = Field(default_factory=list, description="Per-leg results")
    hub_results: List[HubCalculationResult] = Field(default_factory=list, description="Hub results")
    aggregation: Optional[AggregationResult] = Field(None, description="Aggregated results")

    # Totals
    total_emissions_kg_co2: Decimal = Field(..., ge=0, description="Total CO2 emissions (kg)")
    total_emissions_kg_ch4: Decimal = Field(..., ge=0, description="Total CH4 emissions (kg)")
    total_emissions_kg_n2o: Decimal = Field(..., ge=0, description="Total N2O emissions (kg)")
    total_emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total CO2e emissions (kg)")

    # Compliance
    compliance_results: Dict[str, ComplianceCheckResult] = Field(
        default_factory=dict,
        description="Compliance check results by framework"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_durations_ms: Dict[str, float] = Field(default_factory=dict, description="Duration per stage (ms)")

    # Errors/Warnings
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchCalculationRequest(BaseModel):
    """Batch calculation request for multiple shipments."""
    batch_id: str = Field(..., description="Unique batch identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    calculations: List[CalculationRequest] = Field(..., min_items=1, description="Individual calculations")
    parallel: bool = Field(default=True, description="Execute in parallel if True")
    stop_on_error: bool = Field(default=False, description="Stop batch on first error if True")


class BatchCalculationResult(BaseModel):
    """Result model for batch calculation."""
    batch_id: str = Field(..., description="Batch identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    results: List[CalculationResult] = Field(default_factory=list, description="Individual results")
    total_emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total batch emissions")
    successful_count: int = Field(..., ge=0, description="Number of successful calculations")
    failed_count: int = Field(..., ge=0, description="Number of failed calculations")
    batch_duration_ms: float = Field(..., ge=0, description="Total batch duration (ms)")
    provenance_hash: str = Field(..., description="Batch-level provenance hash")


@dataclass
class ProvenanceEntry:
    """Provenance entry for a pipeline stage."""
    stage: PipelineStage
    timestamp: datetime
    input_hash: str
    output_hash: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProvenanceTracker:
    """Tracks provenance chain for audit trail."""

    def __init__(self):
        self.chains: Dict[str, List[ProvenanceEntry]] = {}
        self._lock = RLock()

    def start_chain(self, chain_id: str) -> str:
        """Start a new provenance chain."""
        with self._lock:
            self.chains[chain_id] = []
            return chain_id

    def record_stage(
        self,
        chain_id: str,
        stage: PipelineStage,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a pipeline stage execution."""
        with self._lock:
            if chain_id not in self.chains:
                raise ValueError(f"Chain {chain_id} not started")

            input_hash = self._hash_data(input_data)
            output_hash = self._hash_data(output_data)

            entry = ProvenanceEntry(
                stage=stage,
                timestamp=datetime.now(timezone.utc),
                input_hash=input_hash,
                output_hash=output_hash,
                duration_ms=duration_ms,
                metadata=metadata or {}
            )

            self.chains[chain_id].append(entry)

    def seal_chain(self, chain_id: str) -> str:
        """Seal the provenance chain and return final hash."""
        with self._lock:
            if chain_id not in self.chains:
                raise ValueError(f"Chain {chain_id} not found")

            chain = self.chains[chain_id]
            chain_data = {
                "chain_id": chain_id,
                "stages": [
                    {
                        "stage": entry.stage.value,
                        "timestamp": entry.timestamp.isoformat(),
                        "input_hash": entry.input_hash,
                        "output_hash": entry.output_hash,
                        "duration_ms": entry.duration_ms,
                    }
                    for entry in chain
                ]
            }

            return self._hash_data(chain_data)

    def get_chain(self, chain_id: str) -> List[ProvenanceEntry]:
        """Get provenance chain entries."""
        with self._lock:
            return self.chains.get(chain_id, [])

    def _hash_data(self, data: Any) -> str:
        """Hash arbitrary data for provenance."""
        if isinstance(data, BaseModel):
            data_str = data.json(sort_keys=True)
        else:
            data_str = json.dumps(data, sort_keys=True, default=str)

        return hashlib.sha256(data_str.encode()).hexdigest()


class TransportPipelineEngine:
    """
    TransportPipelineEngine - Orchestrated 10-stage pipeline for transport emissions.

    This engine coordinates the complete transport emissions calculation workflow
    through 10 sequential stages, from input validation to sealed audit trail.

    Attributes:
        db_engine: Database engine for emission factor lookups
        distance_engine: Distance-based calculation engine
        fuel_engine: Fuel-based calculation engine
        spend_engine: Spend-based calculation engine
        multileg_engine: Multi-leg calculation engine
        compliance_engine: Compliance checking engine
        provenance: Provenance tracker for audit trail

    Example:
        >>> engine = TransportPipelineEngine(config)
        >>> request = CalculationRequest(
        ...     calculation_id="calc-001",
        ...     tenant_id="tenant-123",
        ...     origin="Shanghai",
        ...     destination="Los Angeles",
        ...     mass_kg=Decimal("25000"),
        ...     mode=TransportMode.SEA_FREIGHT,
        ... )
        >>> result = engine.execute(request)
        >>> print(f"Emissions: {result.total_emissions_kg_co2e} kg CO2e")
    """

    def __init__(
        self,
        db_engine: TransportDatabaseEngine,
        distance_engine: DistanceBasedCalculatorEngine,
        fuel_engine: FuelBasedCalculatorEngine,
        spend_engine: SpendBasedCalculatorEngine,
        multileg_engine: MultiLegCalculatorEngine,
        compliance_engine: ComplianceCheckerEngine,
    ):
        """
        Initialize TransportPipelineEngine.

        Args:
            db_engine: Database engine for EF lookups
            distance_engine: Distance-based calculator
            fuel_engine: Fuel-based calculator
            spend_engine: Spend-based calculator
            multileg_engine: Multi-leg calculator
            compliance_engine: Compliance checker
        """
        self.db_engine = db_engine
        self.distance_engine = distance_engine
        self.fuel_engine = fuel_engine
        self.spend_engine = spend_engine
        self.multileg_engine = multileg_engine
        self.compliance_engine = compliance_engine
        self.provenance = ProvenanceTracker()

        self._lock = RLock()
        self._stage_cache: Dict[str, Dict[PipelineStage, Any]] = {}

        logger.info("TransportPipelineEngine initialized")

    def execute(self, request: CalculationRequest) -> CalculationResult:
        """
        Execute the 10-stage transport emissions calculation pipeline.

        Args:
            request: Calculation request with transport details

        Returns:
            CalculationResult with emissions, compliance, and provenance

        Raises:
            ValueError: If request validation fails
            RuntimeError: If pipeline execution fails
        """
        chain_id = request.calculation_id
        self.provenance.start_chain(chain_id)

        stage_durations: Dict[str, float] = {}
        errors: List[str] = []
        warnings: List[str] = []

        try:
            # Stage 1: VALIDATE
            start_time = datetime.now()
            validated_request = self._validate(request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["VALIDATE"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.VALIDATE, request, validated_request, duration_ms
            )
            logger.info("[%s] Stage VALIDATE completed in %.2fms", chain_id, duration_ms)

            # Stage 2: CLASSIFY
            start_time = datetime.now()
            classified_request = self._classify(validated_request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["CLASSIFY"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.CLASSIFY, validated_request, classified_request, duration_ms
            )
            logger.info("[%s] Stage CLASSIFY completed in %.2fms", chain_id, duration_ms)

            # Stage 3: NORMALIZE
            start_time = datetime.now()
            normalized_request = self._normalize(classified_request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["NORMALIZE"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.NORMALIZE, classified_request, normalized_request, duration_ms
            )
            logger.info("[%s] Stage NORMALIZE completed in %.2fms", chain_id, duration_ms)

            # Stage 4: RESOLVE_EFS
            start_time = datetime.now()
            ef_resolved_request = self._resolve_efs(normalized_request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["RESOLVE_EFS"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.RESOLVE_EFS, normalized_request, ef_resolved_request, duration_ms
            )
            logger.info("[%s] Stage RESOLVE_EFS completed in %.2fms", chain_id, duration_ms)

            # Stage 5: CALCULATE_LEGS
            start_time = datetime.now()
            leg_results = self._calculate_legs(ef_resolved_request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["CALCULATE_LEGS"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.CALCULATE_LEGS, ef_resolved_request, leg_results, duration_ms
            )
            logger.info("[%s] Stage CALCULATE_LEGS completed in %.2fms (%s legs)", chain_id, duration_ms, len(leg_results))

            # Stage 6: CALCULATE_HUBS
            start_time = datetime.now()
            hub_results = self._calculate_hubs(ef_resolved_request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["CALCULATE_HUBS"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.CALCULATE_HUBS, ef_resolved_request, hub_results, duration_ms
            )
            logger.info("[%s] Stage CALCULATE_HUBS completed in %.2fms (%s hubs)", chain_id, duration_ms, len(hub_results))

            # Stage 7: ALLOCATE
            start_time = datetime.now()
            allocated_legs, allocated_hubs = self._allocate(
                leg_results, hub_results, ef_resolved_request
            )
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["ALLOCATE"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.ALLOCATE,
                {"legs": leg_results, "hubs": hub_results},
                {"legs": allocated_legs, "hubs": allocated_hubs},
                duration_ms
            )
            logger.info("[%s] Stage ALLOCATE completed in %.2fms", chain_id, duration_ms)

            # Stage 8: COMPLIANCE
            start_time = datetime.now()
            compliance_results = self._check_compliance(
                allocated_legs, allocated_hubs, ef_resolved_request
            )
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["COMPLIANCE"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.COMPLIANCE,
                {"legs": allocated_legs, "hubs": allocated_hubs},
                compliance_results,
                duration_ms
            )
            logger.info("[%s] Stage COMPLIANCE completed in %.2fms", chain_id, duration_ms)

            # Stage 9: AGGREGATE
            start_time = datetime.now()
            aggregation = self._aggregate(allocated_legs, allocated_hubs, ef_resolved_request)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["AGGREGATE"] = duration_ms
            self.provenance.record_stage(
                chain_id, PipelineStage.AGGREGATE,
                {"legs": allocated_legs, "hubs": allocated_hubs},
                aggregation,
                duration_ms
            )
            logger.info("[%s] Stage AGGREGATE completed in %.2fms", chain_id, duration_ms)

            # Calculate totals
            total_co2 = sum(leg.emissions_kg_co2 for leg in allocated_legs)
            total_ch4 = sum(leg.emissions_kg_ch4 for leg in allocated_legs)
            total_n2o = sum(leg.emissions_kg_n2o for leg in allocated_legs)
            total_co2e = sum(leg.emissions_kg_co2e for leg in allocated_legs)
            total_co2e += sum(hub.emissions_kg_co2e for hub in allocated_hubs)

            # Create result
            result = CalculationResult(
                calculation_id=chain_id,
                status=PipelineStatus.SUCCESS,
                tenant_id=request.tenant_id,
                leg_results=allocated_legs,
                hub_results=allocated_hubs,
                aggregation=aggregation,
                total_emissions_kg_co2=total_co2,
                total_emissions_kg_ch4=total_ch4,
                total_emissions_kg_n2o=total_n2o,
                total_emissions_kg_co2e=total_co2e,
                compliance_results=compliance_results,
                provenance_hash="",  # Will be set in Stage 10
                stage_durations_ms=stage_durations,
                errors=errors,
                warnings=warnings,
            )

            # Stage 10: SEAL
            start_time = datetime.now()
            provenance_hash = self._seal(result)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            stage_durations["SEAL"] = duration_ms
            result.provenance_hash = provenance_hash
            logger.info("[%s] Stage SEAL completed in %.2fms", chain_id, duration_ms)

            logger.info(
                f"[{chain_id}] Pipeline completed successfully. "
                f"Total emissions: {total_co2e} kg CO2e"
            )

            return result

        except Exception as e:
            logger.error("[%s] Pipeline execution failed: %s", chain_id, e, exc_info=True)
            errors.append(f"Pipeline execution failed: {str(e)}")

            # Return partial result
            return CalculationResult(
                calculation_id=chain_id,
                status=PipelineStatus.FAILED,
                tenant_id=request.tenant_id,
                leg_results=[],
                hub_results=[],
                total_emissions_kg_co2=Decimal("0"),
                total_emissions_kg_ch4=Decimal("0"),
                total_emissions_kg_n2o=Decimal("0"),
                total_emissions_kg_co2e=Decimal("0"),
                compliance_results={},
                provenance_hash="",
                stage_durations_ms=stage_durations,
                errors=errors,
                warnings=warnings,
            )

    def execute_batch(self, batch_request: BatchCalculationRequest) -> BatchCalculationResult:
        """
        Execute batch calculation for multiple shipments.

        Args:
            batch_request: Batch request with multiple calculations

        Returns:
            BatchCalculationResult with aggregated results
        """
        start_time = datetime.now()
        results: List[CalculationResult] = []

        logger.info(
            f"[{batch_request.batch_id}] Starting batch calculation "
            f"({len(batch_request.calculations)} shipments, parallel={batch_request.parallel})"
        )

        for calc_request in batch_request.calculations:
            try:
                result = self.execute(calc_request)
                results.append(result)

                if batch_request.stop_on_error and result.status == PipelineStatus.FAILED:
                    logger.warning(
                        f"[{batch_request.batch_id}] Stopping batch on error "
                        f"(calculation {calc_request.calculation_id} failed)"
                    )
                    break

            except Exception as e:
                logger.error(
                    f"[{batch_request.batch_id}] Calculation {calc_request.calculation_id} "
                    f"failed: {str(e)}"
                )

                if batch_request.stop_on_error:
                    break

        # Aggregate results
        total_emissions = sum(r.total_emissions_kg_co2e for r in results)
        successful_count = sum(1 for r in results if r.status == PipelineStatus.SUCCESS)
        failed_count = len(results) - successful_count

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Create batch provenance hash
        batch_data = {
            "batch_id": batch_request.batch_id,
            "tenant_id": batch_request.tenant_id,
            "calculation_hashes": [r.provenance_hash for r in results],
            "total_emissions_kg_co2e": str(total_emissions),
        }
        batch_hash = hashlib.sha256(
            json.dumps(batch_data, sort_keys=True).encode()
        ).hexdigest()

        logger.info(
            f"[{batch_request.batch_id}] Batch calculation completed in {duration_ms:.2f}ms. "
            f"Success: {successful_count}, Failed: {failed_count}, "
            f"Total emissions: {total_emissions} kg CO2e"
        )

        return BatchCalculationResult(
            batch_id=batch_request.batch_id,
            tenant_id=batch_request.tenant_id,
            results=results,
            total_emissions_kg_co2e=total_emissions,
            successful_count=successful_count,
            failed_count=failed_count,
            batch_duration_ms=duration_ms,
            provenance_hash=batch_hash,
        )

    def _validate(self, request: CalculationRequest) -> CalculationRequest:
        """
        Stage 1: VALIDATE - Input validation.

        Validates:
        - Required fields present
        - Type checking
        - Range validation
        - Data consistency

        Args:
            request: Input calculation request

        Returns:
            Validated request

        Raises:
            ValueError: If validation fails
        """
        errors = self.validate_request(request)

        if errors:
            raise ValueError(f"Request validation failed: {'; '.join(errors)}")

        return request

    def _classify(self, request: CalculationRequest) -> CalculationRequest:
        """
        Stage 2: CLASSIFY - Determine transport mode, vehicle type, method.

        Classification logic:
        - If mode not provided, infer from distance/location
        - If vehicle_type not provided, use default for mode
        - Select calculation method based on available data
        - Map Incoterm to category

        Args:
            request: Validated request

        Returns:
            Request with classifications added to metadata
        """
        metadata = request.metadata.copy()

        # Classify transport mode
        if not request.mode:
            classified_mode = self._classify_mode(request)
            metadata["classified_mode"] = classified_mode.value
        else:
            metadata["classified_mode"] = request.mode.value

        # Classify vehicle type
        if not request.vehicle_type:
            mode = request.mode or TransportMode(metadata["classified_mode"])
            default_vehicle = self._get_default_vehicle_type(mode)
            metadata["classified_vehicle_type"] = default_vehicle
        else:
            metadata["classified_vehicle_type"] = request.vehicle_type

        # Select calculation method
        method = self._select_calculation_method(request)
        metadata["calculation_method"] = method.value

        # Classify Incoterm
        if request.incoterm:
            incoterm_category = self._classify_incoterm(request.incoterm)
            metadata["incoterm_category"] = incoterm_category.value

        request.metadata = metadata
        return request

    def _normalize(self, request: CalculationRequest) -> CalculationRequest:
        """
        Stage 3: NORMALIZE - Unit conversion.

        Conversions:
        - Mass: kg → tonnes
        - Distance: miles → km (if needed)
        - Currency: any → USD (base year)

        Args:
            request: Classified request

        Returns:
            Request with normalized values in metadata
        """
        metadata = request.metadata.copy()

        # Normalize mass to tonnes
        if request.mass_kg:
            mass_tonnes = request.mass_kg / Decimal("1000")
            metadata["mass_tonnes"] = mass_tonnes

        # Normalize distance (already in km)
        if request.distance_km:
            metadata["distance_km"] = request.distance_km

        # Normalize currency (simplified - would use exchange rates in production)
        if request.spend_amount and request.spend_currency:
            # In production, this would call a currency conversion service
            # For now, assume USD or apply fixed conversion rate
            if request.spend_currency == "USD":
                metadata["spend_usd"] = request.spend_amount
            else:
                # Placeholder conversion (would be real exchange rate)
                conversion_rate = Decimal("1.0")  # Replace with actual rate
                metadata["spend_usd"] = request.spend_amount * conversion_rate
                metadata["currency_conversion_rate"] = conversion_rate

        request.metadata = metadata
        return request

    def _resolve_efs(self, request: CalculationRequest) -> CalculationRequest:
        """
        Stage 4: RESOLVE_EFS - Select emission factors from hierarchy.

        Hierarchy:
        1. Supplier-specific EF (if available)
        2. DEFRA EF
        3. EPA EF
        4. GLEC EF
        5. EEIO EF (fallback)

        Args:
            request: Normalized request

        Returns:
            Request with emission factors in metadata
        """
        metadata = request.metadata.copy()

        mode = TransportMode(metadata.get("classified_mode", request.mode.value))
        vehicle_type = metadata.get("classified_vehicle_type", request.vehicle_type)

        # Try to get supplier-specific EF first
        ef_source = EmissionFactorSource.DEFRA  # Default

        if request.supplier_id:
            supplier_ef = self.db_engine.get_supplier_emission_factor(
                request.tenant_id,
                request.supplier_id,
                mode,
                vehicle_type
            )
            if supplier_ef:
                metadata["emission_factor"] = supplier_ef
                metadata["emission_factor_source"] = EmissionFactorSource.SUPPLIER_SPECIFIC.value
                ef_source = EmissionFactorSource.SUPPLIER_SPECIFIC

        # If no supplier EF, try hierarchy
        if "emission_factor" not in metadata:
            for source in [
                EmissionFactorSource.DEFRA,
                EmissionFactorSource.EPA,
                EmissionFactorSource.GLEC,
                EmissionFactorSource.EEIO,
            ]:
                ef = self.db_engine.get_emission_factor(
                    mode=mode,
                    vehicle_type=vehicle_type,
                    source=source,
                    region="Global",  # Could be parameterized
                )

                if ef:
                    metadata["emission_factor"] = ef
                    metadata["emission_factor_source"] = source.value
                    ef_source = source
                    break

        if "emission_factor" not in metadata:
            raise ValueError(
                f"No emission factor found for mode={mode}, vehicle_type={vehicle_type}"
            )

        logger.debug(
            f"Resolved EF: {metadata['emission_factor']} "
            f"(source: {metadata['emission_factor_source']})"
        )

        request.metadata = metadata
        return request

    def _calculate_legs(self, request: CalculationRequest) -> List[LegCalculationResult]:
        """
        Stage 5: CALCULATE_LEGS - Per-leg emissions calculation.

        Uses appropriate calculator based on method:
        - DISTANCE_BASED → DistanceBasedCalculatorEngine
        - FUEL_BASED → FuelBasedCalculatorEngine
        - SPEND_BASED → SpendBasedCalculatorEngine
        - SUPPLIER_SPECIFIC → DistanceBasedCalculatorEngine with supplier EF

        Args:
            request: Request with resolved EFs

        Returns:
            List of leg calculation results
        """
        method = TransportMethodType(request.metadata["calculation_method"])

        # If multi-leg request
        if request.legs:
            return self._calculate_multileg(request)

        # Single leg calculation
        mode = TransportMode(request.metadata.get("classified_mode", request.mode.value))
        vehicle_type = request.metadata.get("classified_vehicle_type", request.vehicle_type)
        mass_tonnes = request.metadata.get("mass_tonnes", Decimal("0"))

        if method == TransportMethodType.DISTANCE_BASED:
            # Use distance-based calculator
            calc_request = DistanceCalculationRequest(
                tenant_id=request.tenant_id,
                mode=mode,
                vehicle_type=vehicle_type,
                origin=request.origin,
                destination=request.destination,
                distance_km=request.distance_km,
                mass_tonnes=mass_tonnes,
                emission_factor_source=EmissionFactorSource(
                    request.metadata["emission_factor_source"]
                ),
            )

            calc_result = self.distance_engine.calculate(calc_request)

            return [
                LegCalculationResult(
                    leg_id=f"{request.calculation_id}-leg-1",
                    mode=mode,
                    vehicle_type=vehicle_type,
                    origin=request.origin,
                    destination=request.destination,
                    distance_km=calc_result.distance_km,
                    mass_tonnes=mass_tonnes,
                    emissions_kg_co2=calc_result.emissions_kg_co2,
                    emissions_kg_ch4=calc_result.emissions_kg_ch4,
                    emissions_kg_n2o=calc_result.emissions_kg_n2o,
                    emissions_kg_co2e=calc_result.emissions_kg_co2e,
                    emission_factor_source=calc_result.emission_factor_source,
                    method_used=TransportMethodType.DISTANCE_BASED,
                    metadata=calc_result.metadata,
                )
            ]

        elif method == TransportMethodType.FUEL_BASED:
            # Use fuel-based calculator
            calc_request = FuelCalculationRequest(
                tenant_id=request.tenant_id,
                mode=mode,
                vehicle_type=vehicle_type,
                fuel_type=request.fuel_type,
                fuel_liters=request.fuel_liters,
                mass_tonnes=mass_tonnes,
                distance_km=request.distance_km,
            )

            calc_result = self.fuel_engine.calculate(calc_request)

            return [
                LegCalculationResult(
                    leg_id=f"{request.calculation_id}-leg-1",
                    mode=mode,
                    vehicle_type=vehicle_type,
                    origin=request.origin,
                    destination=request.destination,
                    distance_km=request.distance_km or Decimal("0"),
                    mass_tonnes=mass_tonnes,
                    emissions_kg_co2=calc_result.emissions_kg_co2,
                    emissions_kg_ch4=calc_result.emissions_kg_ch4,
                    emissions_kg_n2o=calc_result.emissions_kg_n2o,
                    emissions_kg_co2e=calc_result.emissions_kg_co2e,
                    emission_factor_source=EmissionFactorSource.EPA,  # Fuel-based uses EPA
                    method_used=TransportMethodType.FUEL_BASED,
                    metadata=calc_result.metadata,
                )
            ]

        elif method == TransportMethodType.SPEND_BASED:
            # Use spend-based calculator
            calc_request = SpendCalculationRequest(
                tenant_id=request.tenant_id,
                mode=mode,
                spend_amount_usd=request.metadata.get("spend_usd", request.spend_amount),
                region="Global",
                eeio_sector=self._map_mode_to_eeio_sector(mode),
            )

            calc_result = self.spend_engine.calculate(calc_request)

            return [
                LegCalculationResult(
                    leg_id=f"{request.calculation_id}-leg-1",
                    mode=mode,
                    vehicle_type=vehicle_type,
                    origin=request.origin,
                    destination=request.destination,
                    distance_km=Decimal("0"),  # Unknown for spend-based
                    mass_tonnes=mass_tonnes,
                    emissions_kg_co2=Decimal("0"),  # Spend-based only gives CO2e
                    emissions_kg_ch4=Decimal("0"),
                    emissions_kg_n2o=Decimal("0"),
                    emissions_kg_co2e=calc_result.emissions_kg_co2e,
                    emission_factor_source=EmissionFactorSource.EEIO,
                    method_used=TransportMethodType.SPEND_BASED,
                    metadata=calc_result.metadata,
                )
            ]

        else:
            raise ValueError(f"Unsupported calculation method: {method}")

    def _calculate_hubs(self, request: CalculationRequest) -> List[HubCalculationResult]:
        """
        Stage 6: CALCULATE_HUBS - Hub/transshipment/warehouse emissions.

        Hub emissions calculated based on:
        - Throughput (tonnes)
        - Storage duration (days) for warehouses
        - Hub type-specific emission factors

        Args:
            request: Request with hub definitions

        Returns:
            List of hub calculation results
        """
        if not request.hubs:
            return []

        hub_results: List[HubCalculationResult] = []

        for i, hub_def in enumerate(request.hubs):
            hub_type = hub_def.get("hub_type", "transshipment")
            location = hub_def.get("location", "Unknown")
            throughput_tonnes = Decimal(str(hub_def.get("throughput_tonnes", 0)))
            storage_days = hub_def.get("storage_days")

            # Get hub emission factor (kg CO2e per tonne or kg CO2e per tonne-day)
            if hub_type == "warehouse" and storage_days:
                # Warehousing: kg CO2e per tonne-day
                ef = self.db_engine.get_warehouse_emission_factor(location)
                emissions_kg_co2e = throughput_tonnes * Decimal(str(storage_days)) * ef
            else:
                # Transshipment/terminal: kg CO2e per tonne
                ef = self.db_engine.get_hub_emission_factor(hub_type, location)
                emissions_kg_co2e = throughput_tonnes * ef

            hub_results.append(
                HubCalculationResult(
                    hub_id=f"{request.calculation_id}-hub-{i+1}",
                    hub_type=hub_type,
                    location=location,
                    throughput_tonnes=throughput_tonnes,
                    storage_days=Decimal(str(storage_days)) if storage_days else None,
                    emissions_kg_co2e=emissions_kg_co2e,
                    emission_factor=ef,
                    metadata=hub_def.get("metadata", {}),
                )
            )

        return hub_results

    def _allocate(
        self,
        leg_results: List[LegCalculationResult],
        hub_results: List[HubCalculationResult],
        request: CalculationRequest,
    ) -> Tuple[List[LegCalculationResult], List[HubCalculationResult]]:
        """
        Stage 7: ALLOCATE - Apply allocation method for shared transport.

        Allocation methods:
        - MASS: Allocate by mass proportion
        - VOLUME: Allocate by volume proportion
        - TEU: Allocate by container count
        - REVENUE: Allocate by revenue share
        - PALLET: Allocate by pallet count
        - CHARGEABLE_WEIGHT: Allocate by chargeable weight

        Args:
            leg_results: Calculated leg results
            hub_results: Calculated hub results
            request: Request with allocation configuration

        Returns:
            Tuple of (allocated leg results, allocated hub results)
        """
        if not request.allocation_method or not request.allocation_factor:
            # No allocation needed
            return leg_results, hub_results

        allocation_factor = request.allocation_factor

        logger.debug(
            f"Applying allocation: method={request.allocation_method}, "
            f"factor={allocation_factor}"
        )

        # Allocate leg emissions
        allocated_legs: List[LegCalculationResult] = []
        for leg in leg_results:
            allocated_leg = leg.copy(deep=True)
            allocated_leg.emissions_kg_co2 *= allocation_factor
            allocated_leg.emissions_kg_ch4 *= allocation_factor
            allocated_leg.emissions_kg_n2o *= allocation_factor
            allocated_leg.emissions_kg_co2e *= allocation_factor
            allocated_leg.metadata["allocation_method"] = request.allocation_method.value
            allocated_leg.metadata["allocation_factor"] = str(allocation_factor)
            allocated_legs.append(allocated_leg)

        # Allocate hub emissions
        allocated_hubs: List[HubCalculationResult] = []
        for hub in hub_results:
            allocated_hub = hub.copy(deep=True)
            allocated_hub.emissions_kg_co2e *= allocation_factor
            allocated_hub.metadata["allocation_method"] = request.allocation_method.value
            allocated_hub.metadata["allocation_factor"] = str(allocation_factor)
            allocated_hubs.append(allocated_hub)

        return allocated_legs, allocated_hubs

    def _check_compliance(
        self,
        leg_results: List[LegCalculationResult],
        hub_results: List[HubCalculationResult],
        request: CalculationRequest,
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Stage 8: COMPLIANCE - Framework compliance checking.

        Checks compliance with:
        - GHG Protocol Scope 3 Category 4 & 9
        - ISO 14064-1:2018
        - CDP Supply Chain
        - CSRD ESRS E1-6
        - SBTi Transport Guidance
        - GLEC Framework
        - EU CBAM (indirect emissions)

        Args:
            leg_results: Allocated leg results
            hub_results: Allocated hub results
            request: Request with framework list

        Returns:
            Dict of framework → compliance check result
        """
        total_emissions_kg_co2e = sum(leg.emissions_kg_co2e for leg in leg_results)
        total_emissions_kg_co2e += sum(hub.emissions_kg_co2e for hub in hub_results)

        compliance_results: Dict[str, ComplianceCheckResult] = {}

        for framework in request.frameworks:
            check_request = ComplianceCheckRequest(
                tenant_id=request.tenant_id,
                framework=framework,
                transport_mode=TransportMode(request.metadata.get("classified_mode")),
                calculation_method=TransportMethodType(request.metadata["calculation_method"]),
                emission_factor_source=EmissionFactorSource(
                    request.metadata["emission_factor_source"]
                ),
                total_emissions_kg_co2e=total_emissions_kg_co2e,
                has_supplier_specific_data=bool(request.supplier_id),
                has_primary_data=(
                    TransportMethodType(request.metadata["calculation_method"])
                    in [TransportMethodType.FUEL_BASED, TransportMethodType.DISTANCE_BASED]
                ),
                metadata=request.metadata,
            )

            result = self.compliance_engine.check_compliance(check_request)
            compliance_results[framework.value] = result

        return compliance_results

    def _aggregate(
        self,
        leg_results: List[LegCalculationResult],
        hub_results: List[HubCalculationResult],
        request: CalculationRequest,
    ) -> AggregationResult:
        """
        Stage 9: AGGREGATE - Sum by mode, carrier, route, period.

        Aggregation dimensions:
        - by_mode: Total emissions per transport mode
        - by_carrier: Total emissions per carrier
        - by_route: Total emissions per route (origin-destination pair)
        - by_period: Total emissions per time period
        - by_supplier: Total emissions per supplier

        Args:
            leg_results: Allocated leg results
            hub_results: Allocated hub results
            request: Request for metadata

        Returns:
            AggregationResult with multi-dimensional aggregations
        """
        by_mode: Dict[str, Decimal] = {}
        by_carrier: Dict[str, Decimal] = {}
        by_route: Dict[str, Decimal] = {}
        by_supplier: Dict[str, Decimal] = {}

        total_distance_km = Decimal("0")
        total_mass_tonnes = Decimal("0")
        total_emissions_kg_co2e = Decimal("0")

        for leg in leg_results:
            mode_key = leg.mode.value
            by_mode[mode_key] = by_mode.get(mode_key, Decimal("0")) + leg.emissions_kg_co2e

            carrier_key = request.carrier_id or "Unknown"
            by_carrier[carrier_key] = by_carrier.get(carrier_key, Decimal("0")) + leg.emissions_kg_co2e

            route_key = f"{leg.origin}-{leg.destination}"
            by_route[route_key] = by_route.get(route_key, Decimal("0")) + leg.emissions_kg_co2e

            if request.supplier_id:
                by_supplier[request.supplier_id] = (
                    by_supplier.get(request.supplier_id, Decimal("0")) + leg.emissions_kg_co2e
                )

            total_distance_km += leg.distance_km
            total_mass_tonnes += leg.mass_tonnes
            total_emissions_kg_co2e += leg.emissions_kg_co2e

        # Add hub emissions
        for hub in hub_results:
            total_emissions_kg_co2e += hub.emissions_kg_co2e

        # By period (simplified - could be more granular)
        period_key = f"{request.reporting_period_start.date()}_to_{request.reporting_period_end.date()}"
        by_period = {period_key: total_emissions_kg_co2e}

        return AggregationResult(
            by_mode=by_mode,
            by_carrier=by_carrier,
            by_route=by_route,
            by_period=by_period,
            by_supplier=by_supplier,
            total_distance_km=total_distance_km,
            total_mass_tonnes=total_mass_tonnes,
            total_emissions_kg_co2e=total_emissions_kg_co2e,
        )

    def _seal(self, result: CalculationResult) -> str:
        """
        Stage 10: SEAL - Provenance hash, audit trail.

        Creates final provenance hash by sealing the chain.

        Args:
            result: Calculation result before sealing

        Returns:
            Provenance hash (SHA-256)
        """
        return self.provenance.seal_chain(result.calculation_id)

    def _calculate_multileg(self, request: CalculationRequest) -> List[LegCalculationResult]:
        """Calculate emissions for multi-leg journey."""
        if not request.legs:
            return []

        leg_definitions = [
            LegDefinition(
                leg_id=leg_def.get("leg_id", f"leg-{i+1}"),
                mode=TransportMode(leg_def["mode"]),
                vehicle_type=leg_def.get("vehicle_type"),
                origin=leg_def["origin"],
                destination=leg_def["destination"],
                distance_km=Decimal(str(leg_def.get("distance_km", 0))),
                fuel_liters=Decimal(str(leg_def.get("fuel_liters", 0))) if leg_def.get("fuel_liters") else None,
                fuel_type=leg_def.get("fuel_type"),
            )
            for i, leg_def in enumerate(request.legs)
        ]

        multileg_request = MultiLegRequest(
            tenant_id=request.tenant_id,
            journey_id=request.calculation_id,
            legs=leg_definitions,
            total_mass_tonnes=request.metadata.get("mass_tonnes", Decimal("0")),
        )

        multileg_result = self.multileg_engine.calculate(multileg_request)

        # Convert to LegCalculationResult format
        leg_results: List[LegCalculationResult] = []
        for leg_res in multileg_result.leg_results:
            leg_results.append(
                LegCalculationResult(
                    leg_id=leg_res.leg_id,
                    mode=leg_res.mode,
                    vehicle_type=leg_res.vehicle_type,
                    origin=leg_res.origin,
                    destination=leg_res.destination,
                    distance_km=leg_res.distance_km,
                    mass_tonnes=leg_res.mass_tonnes,
                    emissions_kg_co2=leg_res.emissions_kg_co2,
                    emissions_kg_ch4=leg_res.emissions_kg_ch4,
                    emissions_kg_n2o=leg_res.emissions_kg_n2o,
                    emissions_kg_co2e=leg_res.emissions_kg_co2e,
                    emission_factor_source=leg_res.emission_factor_source,
                    method_used=leg_res.method_used,
                    metadata=leg_res.metadata,
                )
            )

        return leg_results

    def _classify_mode(self, request: CalculationRequest) -> TransportMode:
        """Classify transport mode based on available data."""
        # Simple heuristic - could be more sophisticated
        if request.distance_km:
            if request.distance_km > Decimal("5000"):
                return TransportMode.AIR_FREIGHT
            elif request.distance_km > Decimal("500"):
                return TransportMode.SEA_FREIGHT
            else:
                return TransportMode.ROAD_FREIGHT

        # Default
        return TransportMode.ROAD_FREIGHT

    def _get_default_vehicle_type(self, mode: TransportMode) -> str:
        """Get default vehicle type for a transport mode."""
        defaults = {
            TransportMode.ROAD_FREIGHT: "HGV_Average",
            TransportMode.RAIL_FREIGHT: "Freight_Train_Average",
            TransportMode.SEA_FREIGHT: "Container_Ship_Average",
            TransportMode.AIR_FREIGHT: "Cargo_Aircraft_Average",
            TransportMode.INLAND_WATERWAY: "Barge_Average",
        }
        return defaults.get(mode, "Average")

    def _select_calculation_method(self, request: CalculationRequest) -> TransportMethodType:
        """Select calculation method based on available data."""
        if request.fuel_liters and request.fuel_type:
            return TransportMethodType.FUEL_BASED
        elif request.distance_km:
            return TransportMethodType.DISTANCE_BASED
        elif request.spend_amount:
            return TransportMethodType.SPEND_BASED
        elif request.supplier_id:
            return TransportMethodType.SUPPLIER_SPECIFIC
        else:
            # Default to distance-based (will need distance lookup)
            return TransportMethodType.DISTANCE_BASED

    def _classify_incoterm(self, incoterm: str) -> IncotermCategory:
        """Classify Incoterm into category."""
        incoterm_upper = incoterm.upper()

        if incoterm_upper in ["EXW"]:
            return IncotermCategory.EXW
        elif incoterm_upper in ["FCA", "FAS", "FOB"]:
            return IncotermCategory.F_GROUP
        elif incoterm_upper in ["CFR", "CIF", "CPT", "CIP"]:
            return IncotermCategory.C_GROUP
        elif incoterm_upper in ["DAP", "DPU", "DDP"]:
            return IncotermCategory.D_GROUP
        else:
            return IncotermCategory.OTHER

    def _map_mode_to_eeio_sector(self, mode: TransportMode) -> str:
        """Map transport mode to EEIO sector code."""
        mapping = {
            TransportMode.ROAD_FREIGHT: "484000",  # Truck transportation
            TransportMode.RAIL_FREIGHT: "482000",  # Rail transportation
            TransportMode.SEA_FREIGHT: "483000",   # Water transportation
            TransportMode.AIR_FREIGHT: "481000",   # Air transportation
            TransportMode.INLAND_WATERWAY: "483000",
        }
        return mapping.get(mode, "484000")

    def validate_request(self, request: CalculationRequest) -> List[str]:
        """
        Validate calculation request and return list of errors.

        Args:
            request: Request to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Required fields
        if not request.calculation_id:
            errors.append("calculation_id is required")
        if not request.tenant_id:
            errors.append("tenant_id is required")
        if not request.origin:
            errors.append("origin is required")
        if not request.destination:
            errors.append("destination is required")

        # At least one data source required
        has_distance = request.distance_km is not None
        has_fuel = request.fuel_liters is not None and request.fuel_type is not None
        has_spend = request.spend_amount is not None
        has_legs = request.legs is not None and len(request.legs) > 0

        if not (has_distance or has_fuel or has_spend or has_legs):
            errors.append(
                "At least one of distance_km, fuel data, spend data, or legs must be provided"
            )

        # Range validation
        if request.mass_kg is not None and request.mass_kg < 0:
            errors.append("mass_kg must be non-negative")
        if request.distance_km is not None and request.distance_km < 0:
            errors.append("distance_km must be non-negative")
        if request.fuel_liters is not None and request.fuel_liters < 0:
            errors.append("fuel_liters must be non-negative")
        if request.spend_amount is not None and request.spend_amount < 0:
            errors.append("spend_amount must be non-negative")

        # Allocation validation
        if request.allocation_factor is not None:
            if request.allocation_factor < 0 or request.allocation_factor > 1:
                errors.append("allocation_factor must be between 0 and 1")
            if not request.allocation_method:
                errors.append("allocation_method required when allocation_factor provided")

        return errors

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get pipeline status and metrics.

        Returns:
            Dict with pipeline status information
        """
        return {
            "total_chains": len(self.provenance.chains),
            "active_calculations": len(self._stage_cache),
        }

    def reset_pipeline(self) -> None:
        """Reset pipeline state (for testing)."""
        with self._lock:
            self._stage_cache.clear()
            self.provenance.chains.clear()
        logger.info("Pipeline reset complete")

    def get_stage_result(self, calculation_id: str, stage: PipelineStage) -> Optional[Any]:
        """
        Get cached result for a specific stage.

        Args:
            calculation_id: Calculation identifier
            stage: Pipeline stage

        Returns:
            Cached stage result or None
        """
        with self._lock:
            if calculation_id in self._stage_cache:
                return self._stage_cache[calculation_id].get(stage)
            return None

    def retry_from_stage(
        self,
        calculation_id: str,
        stage: PipelineStage
    ) -> CalculationResult:
        """
        Retry pipeline execution from a specific stage.

        Args:
            calculation_id: Calculation identifier
            stage: Stage to retry from

        Returns:
            New calculation result

        Raises:
            ValueError: If calculation not found or stage invalid
        """
        # This would require storing intermediate results in _stage_cache
        # Implementation would retrieve cached results up to the retry stage
        # and re-execute from there
        raise NotImplementedError("Stage retry not yet implemented")

    def get_supported_modes(self) -> List[TransportMode]:
        """
        Get list of supported transport modes.

        Returns:
            List of supported TransportMode enums
        """
        return [
            TransportMode.ROAD_FREIGHT,
            TransportMode.RAIL_FREIGHT,
            TransportMode.SEA_FREIGHT,
            TransportMode.AIR_FREIGHT,
            TransportMode.INLAND_WATERWAY,
        ]
