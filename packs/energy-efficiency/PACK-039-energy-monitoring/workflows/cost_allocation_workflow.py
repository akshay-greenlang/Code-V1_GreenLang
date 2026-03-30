# -*- coding: utf-8 -*-
"""
Cost Allocation Workflow
===================================

3-phase workflow for collecting meter readings, calculating energy costs
using tariff structures, and generating internal cost allocation bills
within PACK-039 Energy Monitoring Pack.

Phases:
    1. MeterReadingCollection  -- Aggregate meter readings by cost centre
    2. CostCalculation         -- Apply tariff rates and demand charges
    3. BillGeneration          -- Generate internal bills and cost reports

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - ISO 50001:2018 Clause 6.5 (energy data collection)
    - ISO 50006:2014 (energy performance using baselines)
    - IPMVP 2022 (cost savings measurement)
    - ASHRAE Standard 105 (standard methods for energy cost)

Schedule: monthly / billing cycle
Estimated duration: 10 minutes

Author: GreenLang Team
Version: 39.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

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

class AllocationBasis(str, Enum):
    """Basis for cost allocation."""

    METERED = "metered"
    FLOOR_AREA = "floor_area"
    HEADCOUNT = "headcount"
    PRODUCTION = "production"
    FIXED_SPLIT = "fixed_split"
    HYBRID = "hybrid"

class BillStatus(str, Enum):
    """Internal bill status."""

    DRAFT = "draft"
    ISSUED = "issued"
    DISPUTED = "disputed"
    PAID = "paid"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

ALLOCATION_METHODS: Dict[str, Dict[str, Any]] = {
    "direct_metering": {
        "description": "Cost allocated based on direct sub-meter readings",
        "accuracy_pct": 98.0,
        "typical_use_case": "Sub-metered tenants or departments",
        "requires_meters": True,
        "fairness_rating": "excellent",
        "implementation_cost": "high",
    },
    "floor_area_pro_rata": {
        "description": "Cost allocated proportional to occupied floor area",
        "accuracy_pct": 70.0,
        "typical_use_case": "Multi-tenant buildings without sub-meters",
        "requires_meters": False,
        "fairness_rating": "moderate",
        "implementation_cost": "low",
    },
    "headcount_pro_rata": {
        "description": "Cost allocated proportional to employee headcount",
        "accuracy_pct": 60.0,
        "typical_use_case": "Office environments with similar usage patterns",
        "requires_meters": False,
        "fairness_rating": "moderate",
        "implementation_cost": "low",
    },
    "production_based": {
        "description": "Cost allocated based on production output or throughput",
        "accuracy_pct": 85.0,
        "typical_use_case": "Manufacturing with shared utilities",
        "requires_meters": False,
        "fairness_rating": "good",
        "implementation_cost": "medium",
    },
    "hybrid_metered_area": {
        "description": "Metered loads direct + common area by floor area",
        "accuracy_pct": 90.0,
        "typical_use_case": "Partially sub-metered buildings",
        "requires_meters": True,
        "fairness_rating": "good",
        "implementation_cost": "medium",
    },
    "time_of_use_metered": {
        "description": "Metered allocation with TOU rate differentiation",
        "accuracy_pct": 95.0,
        "typical_use_case": "TOU tariffs with significant peak/off-peak differential",
        "requires_meters": True,
        "fairness_rating": "excellent",
        "implementation_cost": "high",
    },
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class CostCentre(BaseModel):
    """A cost centre for allocation."""

    cost_centre_id: str = Field(default_factory=lambda: f"cc-{uuid.uuid4().hex[:8]}")
    cost_centre_name: str = Field(..., min_length=1, description="Cost centre name")
    meter_ids: List[str] = Field(default_factory=list, description="Assigned meter IDs")
    floor_area_m2: Decimal = Field(default=Decimal("0"), ge=0, description="Floor area")
    headcount: int = Field(default=0, ge=0, description="Number of occupants")
    production_units: Decimal = Field(default=Decimal("0"), ge=0, description="Production output")
    allocation_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100, description="Fixed allocation %")
    department: str = Field(default="", description="Department or business unit")

class CostAllocationInput(BaseModel):
    """Input data model for CostAllocationWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    billing_period: str = Field(default="2026-01", description="Billing period label")
    allocation_method: str = Field(
        default="direct_metering",
        description="Allocation method key from reference data",
    )
    cost_centres: List[CostCentre] = Field(
        default_factory=list,
        description="Cost centres to allocate costs to",
    )
    meter_readings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Meter readings: meter_id, energy_kwh, peak_kw",
    )
    tariff: Dict[str, Any] = Field(
        default_factory=lambda: {
            "energy_rate_per_kwh": 0.12,
            "demand_rate_per_kw": 15.00,
            "fixed_charge_per_month": 250.00,
            "peak_energy_rate": 0.18,
            "off_peak_energy_rate": 0.08,
            "tax_rate_pct": 5.0,
        },
        description="Tariff structure for cost calculation",
    )
    total_facility_kwh: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total facility energy from main meter",
    )
    total_facility_peak_kw: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total facility peak demand from main meter",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_name")
    @classmethod
    def validate_facility_name(cls, v: str) -> str:
        """Ensure facility name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("facility_name must not be blank")
        return stripped

class CostAllocationResult(BaseModel):
    """Complete result from cost allocation workflow."""

    allocation_id: str = Field(..., description="Unique allocation execution ID")
    facility_id: str = Field(default="", description="Facility identifier")
    billing_period: str = Field(default="", description="Billing period")
    allocation_method: str = Field(default="", description="Method used")
    cost_centres_billed: int = Field(default=0, ge=0)
    total_facility_cost: Decimal = Field(default=Decimal("0"), ge=0)
    total_energy_cost: Decimal = Field(default=Decimal("0"), ge=0)
    total_demand_cost: Decimal = Field(default=Decimal("0"), ge=0)
    total_fixed_cost: Decimal = Field(default=Decimal("0"), ge=0)
    total_tax: Decimal = Field(default=Decimal("0"), ge=0)
    allocated_cost: Decimal = Field(default=Decimal("0"), ge=0)
    unallocated_cost: Decimal = Field(default=Decimal("0"), ge=0)
    allocation_accuracy_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    bills: List[Dict[str, Any]] = Field(default_factory=list)
    allocation_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class CostAllocationWorkflow:
    """
    3-phase cost allocation workflow for internal energy billing.

    Collects meter readings per cost centre, applies tariff calculations,
    and generates internal allocation bills with full auditability.

    Zero-hallucination: all cost calculations use deterministic tariff
    formulas. Allocation methods are sourced from validated reference data.
    No LLM calls in the billing computation path.

    Attributes:
        allocation_id: Unique allocation execution identifier.
        _readings_by_cc: Aggregated readings per cost centre.
        _costs: Calculated costs per cost centre.
        _bills: Generated internal bills.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = CostAllocationWorkflow()
        >>> cc = CostCentre(cost_centre_name="IT Department", floor_area_m2=Decimal("500"))
        >>> inp = CostAllocationInput(
        ...     facility_name="HQ",
        ...     cost_centres=[cc],
        ...     total_facility_kwh=Decimal("100000"),
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_facility_cost > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CostAllocationWorkflow."""
        self.allocation_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._readings_by_cc: Dict[str, Dict[str, Any]] = {}
        self._costs: Dict[str, Dict[str, Any]] = {}
        self._bills: List[Dict[str, Any]] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: CostAllocationInput) -> CostAllocationResult:
        """
        Execute the 3-phase cost allocation workflow.

        Args:
            input_data: Validated cost allocation input.

        Returns:
            CostAllocationResult with readings, costs, and bills.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        self.logger.info(
            "Starting cost allocation workflow %s for facility=%s period=%s method=%s",
            self.allocation_id, input_data.facility_name,
            input_data.billing_period, input_data.allocation_method,
        )

        self._phase_results = []
        self._readings_by_cc = {}
        self._costs = {}
        self._bills = []

        try:
            phase1 = self._phase_meter_reading_collection(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_cost_calculation(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_bill_generation(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error("Cost allocation workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Aggregate totals
        total_energy_cost = sum(
            Decimal(str(c.get("energy_cost", 0))) for c in self._costs.values()
        )
        total_demand_cost = sum(
            Decimal(str(c.get("demand_cost", 0))) for c in self._costs.values()
        )
        total_fixed = sum(
            Decimal(str(c.get("fixed_cost", 0))) for c in self._costs.values()
        )
        total_tax = sum(
            Decimal(str(c.get("tax", 0))) for c in self._costs.values()
        )
        total_facility = total_energy_cost + total_demand_cost + total_fixed + total_tax

        allocated = sum(
            Decimal(str(c.get("total_cost", 0))) for c in self._costs.values()
        )
        unallocated = max(Decimal("0"), total_facility - allocated)

        method_info = ALLOCATION_METHODS.get(
            input_data.allocation_method,
            ALLOCATION_METHODS["direct_metering"],
        )
        accuracy = Decimal(str(method_info["accuracy_pct"]))

        result = CostAllocationResult(
            allocation_id=self.allocation_id,
            facility_id=input_data.facility_id,
            billing_period=input_data.billing_period,
            allocation_method=input_data.allocation_method,
            cost_centres_billed=len(self._bills),
            total_facility_cost=total_facility,
            total_energy_cost=total_energy_cost,
            total_demand_cost=total_demand_cost,
            total_fixed_cost=total_fixed,
            total_tax=total_tax,
            allocated_cost=allocated,
            unallocated_cost=unallocated,
            allocation_accuracy_pct=accuracy,
            bills=self._bills,
            allocation_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Cost allocation workflow %s completed in %dms total=$%.2f "
            "allocated=$%.2f centres=%d",
            self.allocation_id, int(elapsed_ms), float(total_facility),
            float(allocated), len(self._bills),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Meter Reading Collection
    # -------------------------------------------------------------------------

    def _phase_meter_reading_collection(
        self, input_data: CostAllocationInput
    ) -> PhaseResult:
        """Aggregate meter readings by cost centre."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Index meter readings by meter_id
        readings_index: Dict[str, Dict[str, Any]] = {}
        for reading in input_data.meter_readings:
            mid = reading.get("meter_id", "")
            if mid:
                readings_index[mid] = reading

        total_metered_kwh = Decimal("0")
        method = input_data.allocation_method

        for cc in input_data.cost_centres:
            cc_energy = Decimal("0")
            cc_peak = Decimal("0")

            if method in ("direct_metering", "hybrid_metered_area", "time_of_use_metered"):
                # Use meter readings directly
                for mid in cc.meter_ids:
                    reading = readings_index.get(mid, {})
                    cc_energy += Decimal(str(reading.get("energy_kwh", 0)))
                    cc_peak = max(cc_peak, Decimal(str(reading.get("peak_kw", 0))))

            elif method == "floor_area_pro_rata":
                # Allocate by floor area proportion
                total_area = sum(
                    float(c.floor_area_m2) for c in input_data.cost_centres
                )
                if total_area > 0 and input_data.total_facility_kwh > 0:
                    proportion = float(cc.floor_area_m2) / total_area
                    cc_energy = (input_data.total_facility_kwh * Decimal(str(proportion))).quantize(Decimal("0.01"))
                    cc_peak = (input_data.total_facility_peak_kw * Decimal(str(proportion))).quantize(Decimal("0.01"))

            elif method == "headcount_pro_rata":
                total_hc = sum(c.headcount for c in input_data.cost_centres)
                if total_hc > 0 and input_data.total_facility_kwh > 0:
                    proportion = cc.headcount / total_hc
                    cc_energy = (input_data.total_facility_kwh * Decimal(str(proportion))).quantize(Decimal("0.01"))
                    cc_peak = (input_data.total_facility_peak_kw * Decimal(str(proportion))).quantize(Decimal("0.01"))

            elif method == "production_based":
                total_prod = sum(
                    float(c.production_units) for c in input_data.cost_centres
                )
                if total_prod > 0 and input_data.total_facility_kwh > 0:
                    proportion = float(cc.production_units) / total_prod
                    cc_energy = (input_data.total_facility_kwh * Decimal(str(proportion))).quantize(Decimal("0.01"))
                    cc_peak = (input_data.total_facility_peak_kw * Decimal(str(proportion))).quantize(Decimal("0.01"))

            else:
                # Fixed split by allocation_pct
                if cc.allocation_pct > 0 and input_data.total_facility_kwh > 0:
                    proportion = float(cc.allocation_pct) / 100.0
                    cc_energy = (input_data.total_facility_kwh * Decimal(str(proportion))).quantize(Decimal("0.01"))
                    cc_peak = (input_data.total_facility_peak_kw * Decimal(str(proportion))).quantize(Decimal("0.01"))

            total_metered_kwh += cc_energy

            self._readings_by_cc[cc.cost_centre_id] = {
                "cost_centre_id": cc.cost_centre_id,
                "cost_centre_name": cc.cost_centre_name,
                "energy_kwh": str(cc_energy),
                "peak_kw": str(cc_peak),
                "meters_used": cc.meter_ids,
                "allocation_basis": method,
            }

        # Check for unmetered energy
        if input_data.total_facility_kwh > 0:
            unmetered_pct = round(
                (1 - float(total_metered_kwh) / float(input_data.total_facility_kwh)) * 100, 1
            )
            if unmetered_pct > 5:
                warnings.append(
                    f"{unmetered_pct}% of facility energy unallocated; "
                    f"consider adding common area meters"
                )

        outputs["cost_centres_collected"] = len(self._readings_by_cc)
        outputs["total_metered_kwh"] = str(total_metered_kwh)
        outputs["allocation_method"] = method

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 MeterReadingCollection: %d cost centres, total=%.0f kWh",
            len(self._readings_by_cc), float(total_metered_kwh),
        )
        return PhaseResult(
            phase_name="meter_reading_collection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Cost Calculation
    # -------------------------------------------------------------------------

    def _phase_cost_calculation(
        self, input_data: CostAllocationInput
    ) -> PhaseResult:
        """Apply tariff rates and demand charges."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        tariff = input_data.tariff
        energy_rate = Decimal(str(tariff.get("energy_rate_per_kwh", 0.12)))
        demand_rate = Decimal(str(tariff.get("demand_rate_per_kw", 15.00)))
        fixed_charge = Decimal(str(tariff.get("fixed_charge_per_month", 250.00)))
        tax_rate = Decimal(str(tariff.get("tax_rate_pct", 5.0))) / Decimal("100")

        num_centres = len(self._readings_by_cc)
        fixed_per_cc = (fixed_charge / max(Decimal(str(num_centres)), Decimal("1"))).quantize(
            Decimal("0.01")
        )

        for cc_id, reading in self._readings_by_cc.items():
            cc_energy = Decimal(str(reading["energy_kwh"]))
            cc_peak = Decimal(str(reading["peak_kw"]))

            energy_cost = (cc_energy * energy_rate).quantize(Decimal("0.01"))
            demand_cost = (cc_peak * demand_rate).quantize(Decimal("0.01"))
            subtotal = energy_cost + demand_cost + fixed_per_cc
            tax = (subtotal * tax_rate).quantize(Decimal("0.01"))
            total = subtotal + tax

            self._costs[cc_id] = {
                "cost_centre_id": cc_id,
                "cost_centre_name": reading["cost_centre_name"],
                "energy_kwh": str(cc_energy),
                "peak_kw": str(cc_peak),
                "energy_cost": str(energy_cost),
                "demand_cost": str(demand_cost),
                "fixed_cost": str(fixed_per_cc),
                "subtotal": str(subtotal),
                "tax": str(tax),
                "total_cost": str(total),
                "energy_rate": str(energy_rate),
                "demand_rate": str(demand_rate),
                "unit_cost_per_kwh": str(
                    (total / cc_energy).quantize(Decimal("0.0001"))
                    if cc_energy > 0 else Decimal("0")
                ),
            }

        outputs["cost_centres_calculated"] = len(self._costs)
        outputs["total_cost"] = str(sum(
            Decimal(str(c["total_cost"])) for c in self._costs.values()
        ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 CostCalculation: %d centres, total=$%s",
            len(self._costs), outputs["total_cost"],
        )
        return PhaseResult(
            phase_name="cost_calculation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Bill Generation
    # -------------------------------------------------------------------------

    def _phase_bill_generation(
        self, input_data: CostAllocationInput
    ) -> PhaseResult:
        """Generate internal bills and cost reports."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        bill_date = utcnow().isoformat() + "Z"

        for cc_id, cost_data in self._costs.items():
            bill = {
                "bill_id": f"bill-{_new_uuid()[:8]}",
                "allocation_id": self.allocation_id,
                "cost_centre_id": cc_id,
                "cost_centre_name": cost_data["cost_centre_name"],
                "billing_period": input_data.billing_period,
                "facility_name": input_data.facility_name,
                "allocation_method": input_data.allocation_method,
                "line_items": [
                    {
                        "description": f"Energy consumption ({cost_data['energy_kwh']} kWh)",
                        "quantity": cost_data["energy_kwh"],
                        "unit": "kWh",
                        "rate": cost_data["energy_rate"],
                        "amount": cost_data["energy_cost"],
                    },
                    {
                        "description": f"Demand charge ({cost_data['peak_kw']} kW)",
                        "quantity": cost_data["peak_kw"],
                        "unit": "kW",
                        "rate": cost_data["demand_rate"],
                        "amount": cost_data["demand_cost"],
                    },
                    {
                        "description": "Fixed/service charge",
                        "quantity": "1",
                        "unit": "month",
                        "rate": cost_data["fixed_cost"],
                        "amount": cost_data["fixed_cost"],
                    },
                ],
                "subtotal": cost_data["subtotal"],
                "tax": cost_data["tax"],
                "total": cost_data["total_cost"],
                "unit_cost_per_kwh": cost_data["unit_cost_per_kwh"],
                "status": BillStatus.DRAFT.value,
                "generated_at": bill_date,
                "provenance_hash": _compute_hash(
                    json.dumps(cost_data, sort_keys=True, default=str)
                ),
            }
            self._bills.append(bill)

        outputs["bills_generated"] = len(self._bills)
        outputs["billing_period"] = input_data.billing_period
        outputs["total_billed"] = str(sum(
            Decimal(str(b["total"])) for b in self._bills
        ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 BillGeneration: %d bills generated, total=$%s",
            len(self._bills), outputs["total_billed"],
        )
        return PhaseResult(
            phase_name="bill_generation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: CostAllocationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
