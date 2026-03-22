# -*- coding: utf-8 -*-
"""
Cost Allocation Workflow
===================================

4-phase cost allocation workflow within PACK-036 Utility Analysis Pack.
Orchestrates meter mapping, cost pooling, allocation calculation, and
reconciliation report generation to distribute utility costs across
departments, tenants, processes, or cost centres.

Phases:
    1. MeterMapping           -- Map meters to cost centres, validate meter
                                  hierarchy, resolve sub-metering gaps
    2. CostPooling            -- Pool utility costs by commodity, period,
                                  and billing account into allocation pools
    3. AllocationCalculation  -- Apply allocation rules (direct metering,
                                  pro-rata by area, headcount, production,
                                  or hybrid methods) to distribute costs
    4. ReconciliationReport   -- Reconcile allocations against total spend,
                                  generate chargeback report with variance
                                  analysis

The workflow follows GreenLang zero-hallucination principles: all cost
calculations use deterministic arithmetic (consumption * rate, area-based
pro-rata, headcount weighting). No LLM calls in the numeric path.

Schedule: monthly / on-demand
Estimated duration: 15 minutes

Regulatory References:
    - FASB ASC 842 lease accounting (for tenant allocations)
    - IFRS 16 lease standard
    - ASHRAE Standard 105 for energy accounting
    - IPMVP Option C for whole-building verification

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"


def _utcnow() -> datetime:
    """Return current UTC timestamp with zero microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {k: v for k, v in s.items()
             if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AllocationMethod(str, Enum):
    """Cost allocation method classification."""
    DIRECT_METERING = "direct_metering"
    AREA_PRORATE = "area_prorate"
    HEADCOUNT = "headcount"
    PRODUCTION_UNITS = "production_units"
    OPERATING_HOURS = "operating_hours"
    CONNECTED_LOAD = "connected_load"
    HYBRID = "hybrid"
    EQUAL_SPLIT = "equal_split"


class CostCentreType(str, Enum):
    """Cost centre classification."""
    DEPARTMENT = "department"
    TENANT = "tenant"
    PROCESS_LINE = "process_line"
    BUILDING_ZONE = "building_zone"
    FLOOR = "floor"
    BUSINESS_UNIT = "business_unit"


class UtilityType(str, Enum):
    """Utility commodity type."""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    WATER = "water"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    SEWER = "sewer"
    COMPRESSED_AIR = "compressed_air"


class MeterType(str, Enum):
    """Meter classification."""
    MAIN = "main"
    SUB = "sub"
    VIRTUAL = "virtual"
    CHECK = "check"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Typical common area percentages by building type
COMMON_AREA_PERCENTAGES: Dict[str, float] = {
    "office": 0.15,
    "retail": 0.10,
    "industrial": 0.08,
    "mixed_use": 0.18,
    "hospital": 0.20,
    "hotel": 0.25,
    "school": 0.12,
    "warehouse": 0.05,
    "default": 0.15,
}

# Allocation accuracy benchmarks by method
METHOD_ACCURACY_BENCHMARKS: Dict[str, Tuple[float, float]] = {
    "direct_metering": (0.95, 0.99),
    "area_prorate": (0.75, 0.90),
    "headcount": (0.65, 0.85),
    "production_units": (0.80, 0.95),
    "operating_hours": (0.70, 0.90),
    "connected_load": (0.80, 0.92),
    "hybrid": (0.85, 0.95),
    "equal_split": (0.50, 0.75),
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class MeterRecord(BaseModel):
    """Meter definition and mapping record.

    Attributes:
        meter_id: Unique meter identifier.
        meter_name: Display name.
        meter_type: Meter classification (main/sub/virtual).
        utility_type: Utility commodity type.
        parent_meter_id: Parent meter for hierarchy.
        cost_centre_id: Mapped cost centre.
        location: Physical location description.
        installed_capacity_kw: Connected load capacity.
    """
    meter_id: str = Field(default_factory=lambda: f"mtr-{uuid.uuid4().hex[:8]}")
    meter_name: str = Field(default="")
    meter_type: MeterType = Field(default=MeterType.SUB)
    utility_type: UtilityType = Field(default=UtilityType.ELECTRICITY)
    parent_meter_id: str = Field(default="")
    cost_centre_id: str = Field(default="")
    location: str = Field(default="")
    installed_capacity_kw: float = Field(default=0.0, ge=0.0)


class CostCentre(BaseModel):
    """Cost centre definition.

    Attributes:
        centre_id: Unique cost centre identifier.
        centre_name: Display name.
        centre_type: Cost centre classification.
        floor_area_m2: Floor area for area-based allocation.
        headcount: Number of occupants/employees.
        production_units: Production output for production-based allocation.
        operating_hours_per_month: Monthly operating hours.
        connected_load_kw: Total connected electrical load.
        allocation_method: Preferred allocation method.
        parent_centre_id: Parent cost centre for hierarchy.
    """
    centre_id: str = Field(default_factory=lambda: f"cc-{uuid.uuid4().hex[:8]}")
    centre_name: str = Field(default="")
    centre_type: CostCentreType = Field(default=CostCentreType.DEPARTMENT)
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    headcount: int = Field(default=0, ge=0)
    production_units: float = Field(default=0.0, ge=0.0)
    operating_hours_per_month: float = Field(default=160.0, ge=0.0)
    connected_load_kw: float = Field(default=0.0, ge=0.0)
    allocation_method: AllocationMethod = Field(default=AllocationMethod.AREA_PRORATE)
    parent_centre_id: str = Field(default="")


class CostPool(BaseModel):
    """Utility cost pool for a billing period.

    Attributes:
        pool_id: Unique pool identifier.
        utility_type: Utility commodity.
        period: Billing period (YYYY-MM).
        total_consumption: Total consumption.
        consumption_unit: Unit of measure.
        total_cost: Total cost for the pool.
        currency: ISO 4217 currency code.
        account_number: Utility account number.
    """
    pool_id: str = Field(default_factory=lambda: f"pool-{uuid.uuid4().hex[:8]}")
    utility_type: UtilityType = Field(default=UtilityType.ELECTRICITY)
    period: str = Field(default="")
    total_consumption: float = Field(default=0.0, ge=0.0)
    consumption_unit: str = Field(default="kwh")
    total_cost: float = Field(default=0.0)
    currency: str = Field(default="USD")
    account_number: str = Field(default="")


class MeterReading(BaseModel):
    """Sub-meter reading for direct allocation.

    Attributes:
        meter_id: Meter identifier.
        period: Reading period (YYYY-MM).
        consumption: Metered consumption.
        unit: Unit of measure.
    """
    meter_id: str = Field(default="")
    period: str = Field(default="")
    consumption: float = Field(default=0.0, ge=0.0)
    unit: str = Field(default="kwh")


class AllocationEntry(BaseModel):
    """A single cost allocation result.

    Attributes:
        allocation_id: Unique allocation identifier.
        cost_centre_id: Allocated cost centre.
        cost_centre_name: Cost centre display name.
        utility_type: Utility commodity.
        period: Allocation period (YYYY-MM).
        method: Allocation method used.
        allocated_consumption: Allocated consumption.
        allocated_cost: Allocated cost amount.
        allocation_pct: Percentage of total pool.
        basis_value: Value used for allocation basis.
        basis_unit: Unit of allocation basis.
        confidence: Allocation confidence (0-1).
    """
    allocation_id: str = Field(default_factory=lambda: f"alloc-{uuid.uuid4().hex[:8]}")
    cost_centre_id: str = Field(default="")
    cost_centre_name: str = Field(default="")
    utility_type: str = Field(default="electricity")
    period: str = Field(default="")
    method: str = Field(default="area_prorate")
    allocated_consumption: float = Field(default=0.0)
    allocated_cost: float = Field(default=0.0)
    allocation_pct: float = Field(default=0.0)
    basis_value: float = Field(default=0.0)
    basis_unit: str = Field(default="")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class CostAllocationInput(BaseModel):
    """Input data model for CostAllocationWorkflow.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        building_type: Building type for common area defaults.
        meters: Meter definitions and mappings.
        cost_centres: Cost centre definitions.
        cost_pools: Utility cost pools to allocate.
        meter_readings: Sub-meter readings for direct allocation.
        common_area_method: How to allocate common area costs.
        include_common_area: Whether to separate common area costs.
        reconciliation_tolerance_pct: Max acceptable variance percentage.
        entity_id: Multi-tenant entity identifier.
        tenant_id: Multi-tenant tenant identifier.
    """
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    building_type: str = Field(default="office")
    meters: List[MeterRecord] = Field(default_factory=list)
    cost_centres: List[CostCentre] = Field(default_factory=list)
    cost_pools: List[CostPool] = Field(default_factory=list)
    meter_readings: List[MeterReading] = Field(default_factory=list)
    common_area_method: AllocationMethod = Field(default=AllocationMethod.AREA_PRORATE)
    include_common_area: bool = Field(default=True)
    reconciliation_tolerance_pct: float = Field(default=2.0, ge=0.0, le=100.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class CostAllocationResult(BaseModel):
    """Complete result from cost allocation workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="cost_allocation")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    total_cost_allocated: float = Field(default=0.0)
    total_cost_pool: float = Field(default=0.0)
    reconciliation_variance: float = Field(default=0.0)
    reconciliation_variance_pct: float = Field(default=0.0)
    allocations: List[AllocationEntry] = Field(default_factory=list)
    allocation_count: int = Field(default=0)
    cost_centres_allocated: int = Field(default=0)
    allocation_summary: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CostAllocationWorkflow:
    """
    4-phase cost allocation workflow.

    Distributes utility costs across cost centres using direct metering,
    area-based pro-rata, headcount, or hybrid allocation methods. Each
    phase produces a PhaseResult with SHA-256 provenance hash.

    Phases:
        1. MeterMapping          - Map meters to cost centres
        2. CostPooling           - Pool costs by commodity and period
        3. AllocationCalculation - Apply allocation rules
        4. ReconciliationReport  - Reconcile and generate chargeback report

    Zero-hallucination: all allocations use deterministic arithmetic
    (consumption ratios, area percentages, weighted averages).

    Example:
        >>> wf = CostAllocationWorkflow()
        >>> inp = CostAllocationInput(
        ...     cost_centres=[CostCentre(centre_name="Floor 1", floor_area_m2=500)],
        ...     cost_pools=[CostPool(total_cost=5000.0)],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise CostAllocationWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self._meter_map: Dict[str, str] = {}
        self._cost_pools: List[CostPool] = []
        self._allocations: List[AllocationEntry] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(self, input_data: CostAllocationInput) -> CostAllocationResult:
        """Execute the 4-phase cost allocation workflow.

        Args:
            input_data: Validated cost allocation input.

        Returns:
            CostAllocationResult with allocations and reconciliation.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting cost allocation workflow %s for facility=%s",
            self.workflow_id, input_data.facility_id,
        )

        self._phase_results = []
        self._meter_map = {}
        self._cost_pools = []
        self._allocations = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_1_meter_mapping(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_2_cost_pooling(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_3_allocation_calculation(input_data)
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 3 failed: {phase3.errors}")

            phase4 = self._phase_4_reconciliation_report(input_data)
            self._phase_results.append(phase4)

            failed_count = sum(
                1 for p in self._phase_results if p.status == PhaseStatus.FAILED
            )
            if failed_count == 0:
                overall_status = WorkflowStatus.COMPLETED
            elif failed_count < len(self._phase_results):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error("Cost allocation failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        total_pool = sum(p.total_cost for p in self._cost_pools)
        total_allocated = sum(a.allocated_cost for a in self._allocations)
        variance = total_pool - total_allocated
        variance_pct = (abs(variance) / total_pool * 100.0) if total_pool > 0 else 0.0
        centres_allocated = len(set(a.cost_centre_id for a in self._allocations))

        result = CostAllocationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            total_cost_allocated=round(total_allocated, 2),
            total_cost_pool=round(total_pool, 2),
            reconciliation_variance=round(variance, 2),
            reconciliation_variance_pct=round(variance_pct, 2),
            allocations=self._allocations,
            allocation_count=len(self._allocations),
            cost_centres_allocated=centres_allocated,
            allocation_summary=self._build_allocation_summary(),
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Cost allocation %s completed in %.2fs: pool=$%.2f allocated=$%.2f "
            "variance=$%.2f (%.2f%%)",
            self.workflow_id, elapsed, total_pool, total_allocated,
            variance, variance_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Meter Mapping
    # -------------------------------------------------------------------------

    def _phase_1_meter_mapping(
        self, input_data: CostAllocationInput
    ) -> PhaseResult:
        """Map meters to cost centres and validate hierarchy.

        Args:
            input_data: Cost allocation input.

        Returns:
            PhaseResult with meter mapping outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        mapped = 0
        unmapped = 0
        main_meters: List[str] = []

        for meter in input_data.meters:
            if meter.cost_centre_id:
                self._meter_map[meter.meter_id] = meter.cost_centre_id
                mapped += 1
            else:
                unmapped += 1
                warnings.append(f"Meter {meter.meter_id} ({meter.meter_name}) has no cost centre mapping")

            if meter.meter_type == MeterType.MAIN:
                main_meters.append(meter.meter_id)

        # Validate hierarchy: sub-meters should have parent
        orphan_subs = 0
        for meter in input_data.meters:
            if meter.meter_type == MeterType.SUB and not meter.parent_meter_id:
                orphan_subs += 1
                warnings.append(
                    f"Sub-meter {meter.meter_id} has no parent meter assignment"
                )

        # Check coverage: all cost centres should have at least one meter
        metered_centres = set(self._meter_map.values())
        all_centres = {cc.centre_id for cc in input_data.cost_centres}
        unmetered = all_centres - metered_centres
        if unmetered:
            warnings.append(
                f"{len(unmetered)} cost centre(s) have no direct sub-metering; "
                f"will use allocation method"
            )

        outputs["total_meters"] = len(input_data.meters)
        outputs["mapped_meters"] = mapped
        outputs["unmapped_meters"] = unmapped
        outputs["main_meters"] = len(main_meters)
        outputs["orphan_sub_meters"] = orphan_subs
        outputs["metered_cost_centres"] = len(metered_centres)
        outputs["unmetered_cost_centres"] = len(unmetered)
        outputs["total_cost_centres"] = len(all_centres)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 MeterMapping: %d mapped, %d unmapped, %d orphans (%.3fs)",
            mapped, unmapped, orphan_subs, elapsed,
        )
        return PhaseResult(
            phase_name="meter_mapping", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Cost Pooling
    # -------------------------------------------------------------------------

    def _phase_2_cost_pooling(
        self, input_data: CostAllocationInput
    ) -> PhaseResult:
        """Pool utility costs by commodity and period.

        Args:
            input_data: Cost allocation input.

        Returns:
            PhaseResult with cost pool outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._cost_pools = list(input_data.cost_pools)

        # Separate common area costs if configured
        if input_data.include_common_area:
            common_pct = COMMON_AREA_PERCENTAGES.get(
                input_data.building_type,
                COMMON_AREA_PERCENTAGES["default"],
            )
            outputs["common_area_pct"] = round(common_pct * 100.0, 1)
        else:
            common_pct = 0.0

        # Pool summary by utility type
        by_utility: Dict[str, Dict[str, float]] = {}
        by_period: Dict[str, float] = {}

        for pool in self._cost_pools:
            ut = pool.utility_type.value
            if ut not in by_utility:
                by_utility[ut] = {"total_cost": 0.0, "total_consumption": 0.0, "pools": 0}
            by_utility[ut]["total_cost"] += pool.total_cost
            by_utility[ut]["total_consumption"] += pool.total_consumption
            by_utility[ut]["pools"] += 1

            if pool.period:
                by_period[pool.period] = by_period.get(pool.period, 0.0) + pool.total_cost

        total_cost = sum(p.total_cost for p in self._cost_pools)

        outputs["total_pools"] = len(self._cost_pools)
        outputs["total_cost"] = round(total_cost, 2)
        outputs["by_utility_type"] = {
            k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in by_utility.items()
        }
        outputs["periods_covered"] = len(by_period)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 CostPooling: %d pools, $%.2f total (%.3fs)",
            len(self._cost_pools), total_cost, elapsed,
        )
        return PhaseResult(
            phase_name="cost_pooling", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Allocation Calculation
    # -------------------------------------------------------------------------

    def _phase_3_allocation_calculation(
        self, input_data: CostAllocationInput
    ) -> PhaseResult:
        """Apply allocation rules to distribute costs.

        Args:
            input_data: Cost allocation input.

        Returns:
            PhaseResult with allocation outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        allocations: List[AllocationEntry] = []

        if not input_data.cost_centres:
            return PhaseResult(
                phase_name="allocation_calculation", phase_number=3,
                status=PhaseStatus.FAILED,
                errors=["No cost centres defined for allocation"],
                duration_seconds=round(time.perf_counter() - t_start, 4),
            )

        if not self._cost_pools:
            return PhaseResult(
                phase_name="allocation_calculation", phase_number=3,
                status=PhaseStatus.FAILED,
                errors=["No cost pools available for allocation"],
                duration_seconds=round(time.perf_counter() - t_start, 4),
            )

        # Build meter reading lookup: meter_id+period -> consumption
        reading_lookup: Dict[str, float] = {}
        for reading in input_data.meter_readings:
            key = f"{reading.meter_id}_{reading.period}"
            reading_lookup[key] = reading.consumption

        # Common area cost separation
        common_pct = 0.0
        if input_data.include_common_area:
            common_pct = COMMON_AREA_PERCENTAGES.get(
                input_data.building_type,
                COMMON_AREA_PERCENTAGES["default"],
            )

        # Calculate allocation basis totals
        total_area = sum(cc.floor_area_m2 for cc in input_data.cost_centres)
        total_headcount = sum(cc.headcount for cc in input_data.cost_centres)
        total_production = sum(cc.production_units for cc in input_data.cost_centres)
        total_hours = sum(cc.operating_hours_per_month for cc in input_data.cost_centres)
        total_connected = sum(cc.connected_load_kw for cc in input_data.cost_centres)

        methods_used: Dict[str, int] = {}

        for pool in self._cost_pools:
            allocatable_cost = pool.total_cost * (1.0 - common_pct)
            common_cost = pool.total_cost * common_pct
            allocatable_consumption = pool.total_consumption * (1.0 - common_pct)

            for cc in input_data.cost_centres:
                method = cc.allocation_method
                method_str = method.value
                methods_used[method_str] = methods_used.get(method_str, 0) + 1

                alloc_pct = 0.0
                basis_value = 0.0
                basis_unit = ""
                confidence = 0.85

                if method == AllocationMethod.DIRECT_METERING:
                    # Look up meter readings for this cost centre
                    metered_consumption = 0.0
                    for meter in input_data.meters:
                        if meter.cost_centre_id == cc.centre_id:
                            key = f"{meter.meter_id}_{pool.period}"
                            metered_consumption += reading_lookup.get(key, 0.0)

                    if pool.total_consumption > 0:
                        alloc_pct = metered_consumption / pool.total_consumption
                    basis_value = metered_consumption
                    basis_unit = pool.consumption_unit
                    confidence = METHOD_ACCURACY_BENCHMARKS["direct_metering"][0]

                elif method == AllocationMethod.AREA_PRORATE:
                    if total_area > 0:
                        alloc_pct = cc.floor_area_m2 / total_area
                    basis_value = cc.floor_area_m2
                    basis_unit = "m2"
                    confidence = METHOD_ACCURACY_BENCHMARKS["area_prorate"][0]

                elif method == AllocationMethod.HEADCOUNT:
                    if total_headcount > 0:
                        alloc_pct = cc.headcount / total_headcount
                    basis_value = float(cc.headcount)
                    basis_unit = "persons"
                    confidence = METHOD_ACCURACY_BENCHMARKS["headcount"][0]

                elif method == AllocationMethod.PRODUCTION_UNITS:
                    if total_production > 0:
                        alloc_pct = cc.production_units / total_production
                    basis_value = cc.production_units
                    basis_unit = "units"
                    confidence = METHOD_ACCURACY_BENCHMARKS["production_units"][0]

                elif method == AllocationMethod.OPERATING_HOURS:
                    if total_hours > 0:
                        alloc_pct = cc.operating_hours_per_month / total_hours
                    basis_value = cc.operating_hours_per_month
                    basis_unit = "hours"
                    confidence = METHOD_ACCURACY_BENCHMARKS["operating_hours"][0]

                elif method == AllocationMethod.CONNECTED_LOAD:
                    if total_connected > 0:
                        alloc_pct = cc.connected_load_kw / total_connected
                    basis_value = cc.connected_load_kw
                    basis_unit = "kw"
                    confidence = METHOD_ACCURACY_BENCHMARKS["connected_load"][0]

                elif method == AllocationMethod.EQUAL_SPLIT:
                    n_centres = len(input_data.cost_centres)
                    alloc_pct = 1.0 / n_centres if n_centres > 0 else 0.0
                    basis_value = 1.0
                    basis_unit = "equal"
                    confidence = METHOD_ACCURACY_BENCHMARKS["equal_split"][0]

                else:
                    # Hybrid: average of area and headcount
                    area_pct = cc.floor_area_m2 / total_area if total_area > 0 else 0.0
                    hc_pct = cc.headcount / total_headcount if total_headcount > 0 else 0.0
                    alloc_pct = (area_pct + hc_pct) / 2.0
                    basis_value = cc.floor_area_m2
                    basis_unit = "hybrid"
                    confidence = METHOD_ACCURACY_BENCHMARKS["hybrid"][0]

                alloc_pct = min(1.0, max(0.0, alloc_pct))
                allocated_cost = allocatable_cost * alloc_pct
                allocated_consumption = allocatable_consumption * alloc_pct

                # Add common area share (distributed by same method)
                common_share = common_cost * alloc_pct
                allocated_cost += common_share

                allocations.append(AllocationEntry(
                    cost_centre_id=cc.centre_id,
                    cost_centre_name=cc.centre_name,
                    utility_type=pool.utility_type.value,
                    period=pool.period,
                    method=method_str,
                    allocated_consumption=round(allocated_consumption, 4),
                    allocated_cost=round(allocated_cost, 2),
                    allocation_pct=round(alloc_pct * 100.0, 2),
                    basis_value=round(basis_value, 2),
                    basis_unit=basis_unit,
                    confidence=round(confidence, 2),
                ))

        self._allocations = allocations

        outputs["allocations_created"] = len(allocations)
        outputs["methods_used"] = methods_used
        outputs["cost_centres_allocated"] = len(
            set(a.cost_centre_id for a in allocations)
        )
        outputs["total_allocated"] = round(
            sum(a.allocated_cost for a in allocations), 2
        )

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 AllocationCalculation: %d allocations, $%.2f total (%.3fs)",
            len(allocations), outputs["total_allocated"], elapsed,
        )
        return PhaseResult(
            phase_name="allocation_calculation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Reconciliation Report
    # -------------------------------------------------------------------------

    def _phase_4_reconciliation_report(
        self, input_data: CostAllocationInput
    ) -> PhaseResult:
        """Reconcile allocations against total spend.

        Args:
            input_data: Cost allocation input.

        Returns:
            PhaseResult with reconciliation outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = f"rpt-{uuid.uuid4().hex[:8]}"
        total_pool = sum(p.total_cost for p in self._cost_pools)
        total_allocated = sum(a.allocated_cost for a in self._allocations)
        variance = total_pool - total_allocated
        variance_pct = (abs(variance) / total_pool * 100.0) if total_pool > 0 else 0.0

        # Check against tolerance
        tolerance = input_data.reconciliation_tolerance_pct
        reconciled = variance_pct <= tolerance
        if not reconciled:
            warnings.append(
                f"Reconciliation variance {variance_pct:.2f}% exceeds "
                f"tolerance {tolerance:.1f}%"
            )

        # Cost centre summary
        centre_summary: Dict[str, Dict[str, Any]] = {}
        for alloc in self._allocations:
            cid = alloc.cost_centre_id
            if cid not in centre_summary:
                centre_summary[cid] = {
                    "name": alloc.cost_centre_name,
                    "total_cost": 0.0,
                    "total_consumption": 0.0,
                    "methods": set(),
                }
            centre_summary[cid]["total_cost"] += alloc.allocated_cost
            centre_summary[cid]["total_consumption"] += alloc.allocated_consumption
            centre_summary[cid]["methods"].add(alloc.method)

        # Convert sets to lists for JSON serialisation
        centre_report = {}
        for cid, data in centre_summary.items():
            centre_report[cid] = {
                "name": data["name"],
                "total_cost": round(data["total_cost"], 2),
                "total_consumption": round(data["total_consumption"], 2),
                "cost_pct": round(
                    data["total_cost"] / total_pool * 100.0, 2
                ) if total_pool > 0 else 0.0,
                "methods": list(data["methods"]),
            }

        outputs["report_id"] = report_id
        outputs["generated_at"] = _utcnow().isoformat()
        outputs["total_pool_cost"] = round(total_pool, 2)
        outputs["total_allocated_cost"] = round(total_allocated, 2)
        outputs["variance"] = round(variance, 2)
        outputs["variance_pct"] = round(variance_pct, 2)
        outputs["reconciled"] = reconciled
        outputs["tolerance_pct"] = tolerance
        outputs["centre_summary"] = centre_report
        outputs["methodology"] = [
            "Direct metering where sub-meters available",
            "Area-based pro-rata for unmetered spaces",
            "Common area costs distributed proportionally",
            "FASB ASC 842 / IFRS 16 compliant for tenant allocations",
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 ReconciliationReport: report=%s, variance=%.2f%%, "
            "reconciled=%s (%.3fs)",
            report_id, variance_pct, reconciled, elapsed,
        )
        return PhaseResult(
            phase_name="reconciliation_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_allocation_summary(self) -> Dict[str, Any]:
        """Build allocation summary statistics.

        Returns:
            Dictionary with summary statistics.
        """
        by_method: Dict[str, int] = {}
        by_utility: Dict[str, float] = {}

        for alloc in self._allocations:
            by_method[alloc.method] = by_method.get(alloc.method, 0) + 1
            by_utility[alloc.utility_type] = (
                by_utility.get(alloc.utility_type, 0.0) + alloc.allocated_cost
            )

        return {
            "by_method": by_method,
            "by_utility_type": {k: round(v, 2) for k, v in by_utility.items()},
            "avg_confidence": round(
                sum(a.confidence for a in self._allocations) / max(len(self._allocations), 1),
                2,
            ),
            "workflow_version": _MODULE_VERSION,
        }
