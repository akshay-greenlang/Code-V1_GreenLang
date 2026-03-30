# -*- coding: utf-8 -*-
"""
Load Shifting Workflow
===================================

3-phase workflow for identifying shiftable loads, mapping operational
constraints, and optimising multi-load schedules to minimise peak demand
within PACK-038 Peak Shaving Pack.

Phases:
    1. LoadInventory        -- Identify shiftable loads with kW capacity
    2. ConstraintMapping    -- Map operational constraints (comfort, production)
    3. ScheduleOptimization -- Multi-load coordination with aggregate peak limit

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - ASHRAE 55 (thermal comfort constraints)
    - OSHA workplace environment standards
    - ISO 50001 energy management best practices

Schedule: on-demand / quarterly
Estimated duration: 15 minutes

Author: GreenLang Team
Version: 38.0.0
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

class ShiftDirection(str, Enum):
    """Direction of load shift."""

    PRE_COOL = "pre_cool"
    DEFER = "defer"
    SPLIT = "split"
    ADVANCE = "advance"

class ConstraintType(str, Enum):
    """Operational constraint type."""

    COMFORT = "comfort"
    PRODUCTION = "production"
    EQUIPMENT = "equipment"
    SAFETY = "safety"
    OCCUPANCY = "occupancy"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

SHIFTABLE_LOAD_PARAMS: Dict[str, Dict[str, Any]] = {
    "hvac_cooling": {
        "description": "HVAC cooling / pre-cooling capable",
        "typical_rated_kw_pct": 0.30,
        "max_shift_hours": 3,
        "shift_direction": "pre_cool",
        "thermal_mass_hours": 2.0,
        "comfort_impact": "low",
        "automation_level": "high",
        "response_time_min": 5,
    },
    "water_heating": {
        "description": "Domestic / process water heating",
        "typical_rated_kw_pct": 0.08,
        "max_shift_hours": 6,
        "shift_direction": "defer",
        "thermal_mass_hours": 4.0,
        "comfort_impact": "low",
        "automation_level": "high",
        "response_time_min": 1,
    },
    "ev_charging": {
        "description": "Electric vehicle charging stations",
        "typical_rated_kw_pct": 0.10,
        "max_shift_hours": 8,
        "shift_direction": "defer",
        "thermal_mass_hours": 0,
        "comfort_impact": "none",
        "automation_level": "high",
        "response_time_min": 1,
    },
    "process_batch": {
        "description": "Batch process loads (mixers, kilns, ovens)",
        "typical_rated_kw_pct": 0.15,
        "max_shift_hours": 4,
        "shift_direction": "advance",
        "thermal_mass_hours": 0,
        "comfort_impact": "none",
        "automation_level": "medium",
        "response_time_min": 30,
    },
    "laundry_dryers": {
        "description": "Commercial laundry / dryer loads",
        "typical_rated_kw_pct": 0.05,
        "max_shift_hours": 4,
        "shift_direction": "defer",
        "thermal_mass_hours": 0,
        "comfort_impact": "low",
        "automation_level": "medium",
        "response_time_min": 5,
    },
    "pumping": {
        "description": "Water pumping / irrigation",
        "typical_rated_kw_pct": 0.08,
        "max_shift_hours": 6,
        "shift_direction": "advance",
        "thermal_mass_hours": 0,
        "comfort_impact": "none",
        "automation_level": "high",
        "response_time_min": 2,
    },
    "compressed_air": {
        "description": "Compressed air (with storage tank)",
        "typical_rated_kw_pct": 0.10,
        "max_shift_hours": 2,
        "shift_direction": "pre_cool",
        "thermal_mass_hours": 1.5,
        "comfort_impact": "none",
        "automation_level": "high",
        "response_time_min": 3,
    },
    "refrigeration": {
        "description": "Refrigeration / cold storage with thermal mass",
        "typical_rated_kw_pct": 0.12,
        "max_shift_hours": 3,
        "shift_direction": "pre_cool",
        "thermal_mass_hours": 3.0,
        "comfort_impact": "none",
        "automation_level": "medium",
        "response_time_min": 5,
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

class ShiftableLoad(BaseModel):
    """Individual shiftable load record."""

    load_id: str = Field(default_factory=lambda: f"sl-{uuid.uuid4().hex[:8]}")
    load_type: str = Field(default="", description="Load type key from SHIFTABLE_LOAD_PARAMS")
    name: str = Field(default="", description="Human-readable load name")
    rated_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Rated power kW")
    shiftable_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Shiftable capacity kW")
    max_shift_hours: Decimal = Field(default=Decimal("0"), ge=0, description="Max shift window hours")
    shift_direction: str = Field(default="defer", description="pre_cool|defer|split|advance")
    constraint_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    priority: int = Field(default=0, ge=0, description="Shift priority rank")
    optimal_start_hour: int = Field(default=0, ge=0, le=23)
    optimal_end_hour: int = Field(default=0, ge=0, le=23)

class LoadShiftInput(BaseModel):
    """Input data model for LoadShiftWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    peak_demand_kw: Decimal = Field(..., gt=0, description="Current peak demand kW")
    target_peak_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Target peak kW")
    peak_window_start_hour: int = Field(default=12, ge=0, le=23, description="Peak window start hour")
    peak_window_end_hour: int = Field(default=18, ge=0, le=23, description="Peak window end hour")
    operating_hours_start: int = Field(default=6, ge=0, le=23, description="Facility operating start")
    operating_hours_end: int = Field(default=22, ge=0, le=23, description="Facility operating end")
    loads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Load data: load_type, rated_kw, name, constraints",
    )
    constraints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Operational constraints: type, load_id, description, severity",
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

class LoadShiftResult(BaseModel):
    """Complete result from load shifting workflow."""

    shift_id: str = Field(..., description="Unique load shift execution ID")
    facility_id: str = Field(default="", description="Facility identifier")
    total_shiftable_kw: Decimal = Field(default=Decimal("0"), ge=0)
    total_shifted_kw: Decimal = Field(default=Decimal("0"), ge=0)
    peak_reduction_kw: Decimal = Field(default=Decimal("0"), ge=0)
    new_peak_kw: Decimal = Field(default=Decimal("0"), ge=0)
    loads_assessed: int = Field(default=0, ge=0)
    loads_shifted: int = Field(default=0, ge=0)
    shiftable_loads: List[ShiftableLoad] = Field(default_factory=list)
    schedule: Dict[str, Any] = Field(default_factory=dict)
    constraint_violations: int = Field(default=0, ge=0)
    shift_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class LoadShiftWorkflow:
    """
    3-phase load shifting workflow for peak demand reduction.

    Identifies shiftable loads with capacity ratings, maps operational
    constraints, and optimises multi-load schedules to stay within
    aggregate peak limits.

    Zero-hallucination: all scheduling uses deterministic constraint
    satisfaction and priority-based allocation. No LLM calls in the
    numeric computation path.

    Attributes:
        shift_id: Unique load shift execution identifier.
        _loads: Identified shiftable loads.
        _constraints: Mapped operational constraints.
        _schedule: Optimised schedule.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = LoadShiftWorkflow()
        >>> inp = LoadShiftInput(
        ...     facility_name="Warehouse D",
        ...     peak_demand_kw=Decimal("1800"),
        ... )
        >>> result = wf.run(inp)
        >>> assert result.peak_reduction_kw >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LoadShiftWorkflow."""
        self.shift_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._loads: List[ShiftableLoad] = []
        self._constraints: List[Dict[str, Any]] = []
        self._schedule: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: LoadShiftInput) -> LoadShiftResult:
        """
        Execute the 3-phase load shifting workflow.

        Args:
            input_data: Validated load shift input.

        Returns:
            LoadShiftResult with shiftable loads, schedule, and peak reduction.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        self.logger.info(
            "Starting load shift workflow %s for facility=%s peak=%s kW",
            self.shift_id, input_data.facility_name, input_data.peak_demand_kw,
        )

        self._phase_results = []
        self._loads = []
        self._constraints = []
        self._schedule = {}

        try:
            phase1 = self._phase_load_inventory(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_constraint_mapping(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_schedule_optimization(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error("Load shift workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        total_shiftable = sum(ld.shiftable_kw for ld in self._loads)
        shifted_loads = [ld for ld in self._loads if ld.priority > 0]
        total_shifted = sum(ld.shiftable_kw for ld in shifted_loads)
        peak_reduction = Decimal(str(self._schedule.get("peak_reduction_kw", 0)))
        new_peak = input_data.peak_demand_kw - peak_reduction

        result = LoadShiftResult(
            shift_id=self.shift_id,
            facility_id=input_data.facility_id,
            total_shiftable_kw=total_shiftable,
            total_shifted_kw=total_shifted,
            peak_reduction_kw=peak_reduction,
            new_peak_kw=max(Decimal("0"), new_peak),
            loads_assessed=len(self._loads),
            loads_shifted=len(shifted_loads),
            shiftable_loads=self._loads,
            schedule=self._schedule,
            constraint_violations=self._schedule.get("constraint_violations", 0),
            shift_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Load shift workflow %s completed in %dms shiftable=%.0f kW "
            "shifted=%.0f kW reduction=%.0f kW",
            self.shift_id, int(elapsed_ms), float(total_shiftable),
            float(total_shifted), float(peak_reduction),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Load Inventory
    # -------------------------------------------------------------------------

    def _phase_load_inventory(
        self, input_data: LoadShiftInput
    ) -> PhaseResult:
        """Identify shiftable loads with kW capacity."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if input_data.loads:
            for load_dict in input_data.loads:
                load_type = load_dict.get("load_type", "")
                params = SHIFTABLE_LOAD_PARAMS.get(load_type, {})
                rated_kw = Decimal(str(load_dict.get("rated_kw", 0)))

                if not params:
                    warnings.append(f"Load type '{load_type}' not in shiftable parameters")
                    shiftable_pct = Decimal("0.10")
                else:
                    shiftable_pct = Decimal(str(params.get("typical_rated_kw_pct", 0.10)))

                # For shiftable loads, the shiftable kW is the full rated kW
                # (unlike flexibility which is a percentage)
                shiftable_kw = rated_kw  # Full load can be shifted in time

                self._loads.append(ShiftableLoad(
                    load_id=load_dict.get("load_id", f"sl-{uuid.uuid4().hex[:8]}"),
                    load_type=load_type,
                    name=load_dict.get("name", params.get("description", load_type)),
                    rated_kw=rated_kw,
                    shiftable_kw=shiftable_kw,
                    max_shift_hours=Decimal(str(params.get("max_shift_hours", 2))),
                    shift_direction=params.get("shift_direction", "defer"),
                ))
        else:
            # Generate from benchmarks
            warnings.append("No load data provided; generating from benchmarks")
            peak_kw = float(input_data.peak_demand_kw)
            for lt_key, params in SHIFTABLE_LOAD_PARAMS.items():
                rated_kw = Decimal(str(round(peak_kw * params["typical_rated_kw_pct"], 1)))
                if rated_kw <= 0:
                    continue
                self._loads.append(ShiftableLoad(
                    load_type=lt_key,
                    name=params["description"],
                    rated_kw=rated_kw,
                    shiftable_kw=rated_kw,
                    max_shift_hours=Decimal(str(params["max_shift_hours"])),
                    shift_direction=params["shift_direction"],
                ))

        outputs["loads_identified"] = len(self._loads)
        outputs["total_shiftable_kw"] = str(sum(ld.shiftable_kw for ld in self._loads))
        outputs["load_types"] = list(set(ld.load_type for ld in self._loads))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 LoadInventory: %d shiftable loads identified, total=%.0f kW",
            len(self._loads), float(sum(ld.shiftable_kw for ld in self._loads)),
        )
        return PhaseResult(
            phase_name="load_inventory", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Constraint Mapping
    # -------------------------------------------------------------------------

    def _phase_constraint_mapping(
        self, input_data: LoadShiftInput
    ) -> PhaseResult:
        """Map operational constraints (comfort, production, equipment)."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Apply provided constraints
        if input_data.constraints:
            self._constraints = list(input_data.constraints)
        else:
            # Generate default constraints from load parameters
            for load_item in self._loads:
                params = SHIFTABLE_LOAD_PARAMS.get(load_item.load_type, {})
                comfort = params.get("comfort_impact", "low")

                if comfort == "none":
                    score = Decimal("95")
                elif comfort == "low":
                    score = Decimal("80")
                elif comfort == "medium":
                    score = Decimal("60")
                else:
                    score = Decimal("30")

                # Automation level affects constraint score
                auto = params.get("automation_level", "medium")
                if auto == "high":
                    score = min(Decimal("100"), score + Decimal("10"))
                elif auto == "low":
                    score = max(Decimal("0"), score - Decimal("15"))

                load_item.constraint_score = score

                self._constraints.append({
                    "load_id": load_item.load_id,
                    "load_type": load_item.load_type,
                    "constraint_type": "comfort" if comfort != "none" else "equipment",
                    "comfort_impact": comfort,
                    "automation_level": auto,
                    "constraint_score": str(score),
                    "response_time_min": params.get("response_time_min", 5),
                })

        # Update constraint scores for loads with explicit constraints
        for constraint in input_data.constraints or []:
            load_id = constraint.get("load_id", "")
            severity = constraint.get("severity", "medium")
            for load_item in self._loads:
                if load_item.load_id == load_id:
                    if severity == "critical":
                        load_item.constraint_score = Decimal("10")
                    elif severity == "high":
                        load_item.constraint_score = Decimal("30")
                    elif severity == "medium":
                        load_item.constraint_score = Decimal("60")
                    else:
                        load_item.constraint_score = Decimal("85")

        outputs["constraints_mapped"] = len(self._constraints)
        outputs["avg_constraint_score"] = str(round(
            float(sum(ld.constraint_score for ld in self._loads))
            / max(len(self._loads), 1), 1
        ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 ConstraintMapping: %d constraints mapped, avg_score=%.1f",
            len(self._constraints),
            float(sum(ld.constraint_score for ld in self._loads)) / max(len(self._loads), 1),
        )
        return PhaseResult(
            phase_name="constraint_mapping", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Schedule Optimization
    # -------------------------------------------------------------------------

    def _phase_schedule_optimization(
        self, input_data: LoadShiftInput
    ) -> PhaseResult:
        """Multi-load coordination with aggregate peak constraint."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        peak_kw = float(input_data.peak_demand_kw)
        target_kw = float(input_data.target_peak_kw) if input_data.target_peak_kw > 0 else peak_kw * 0.85
        peak_start = input_data.peak_window_start_hour
        peak_end = input_data.peak_window_end_hour
        op_start = input_data.operating_hours_start
        op_end = input_data.operating_hours_end

        # Sort loads by constraint score (highest = easiest to shift first)
        sorted_loads = sorted(
            self._loads,
            key=lambda x: float(x.constraint_score),
            reverse=True,
        )

        cumulative_reduction = Decimal("0")
        reduction_target = Decimal(str(round(peak_kw - target_kw, 1)))
        shift_schedule: List[Dict[str, Any]] = []
        violations = 0
        priority = 0

        for load_item in sorted_loads:
            if cumulative_reduction >= reduction_target:
                break

            # Skip loads with very low constraint scores (too risky)
            if load_item.constraint_score < Decimal("20"):
                warnings.append(f"Skipping {load_item.name}: constraint score too low")
                continue

            priority += 1
            load_item.priority = priority

            # Determine optimal shift window
            max_shift = int(load_item.max_shift_hours)
            direction = load_item.shift_direction

            if direction in ("pre_cool", "advance"):
                # Shift to before peak window
                optimal_start = max(op_start, peak_start - max_shift)
                optimal_end = peak_start
            elif direction == "defer":
                # Shift to after peak window
                optimal_start = peak_end
                optimal_end = min(op_end, peak_end + max_shift)
            else:
                # Split: half before, half after
                optimal_start = max(op_start, peak_start - max_shift // 2)
                optimal_end = min(op_end, peak_end + max_shift // 2)

            load_item.optimal_start_hour = optimal_start
            load_item.optimal_end_hour = optimal_end

            # Check for schedule conflicts
            if optimal_start >= optimal_end:
                violations += 1
                warnings.append(f"Schedule conflict for {load_item.name}")
                continue

            cumulative_reduction += load_item.shiftable_kw

            shift_schedule.append({
                "load_id": load_item.load_id,
                "load_name": load_item.name,
                "shiftable_kw": str(load_item.shiftable_kw),
                "direction": direction,
                "from_window": f"{peak_start:02d}:00-{peak_end:02d}:00",
                "to_window": f"{optimal_start:02d}:00-{optimal_end:02d}:00",
                "priority": priority,
                "constraint_score": str(load_item.constraint_score),
            })

        # Calculate actual peak reduction (capped at target)
        actual_reduction = min(cumulative_reduction, reduction_target)

        self._schedule = {
            "shift_schedule": shift_schedule,
            "peak_reduction_kw": str(actual_reduction),
            "new_peak_kw": str(Decimal(str(peak_kw)) - actual_reduction),
            "loads_shifted": len(shift_schedule),
            "constraint_violations": violations,
            "target_met": float(cumulative_reduction) >= float(reduction_target),
        }

        outputs["loads_shifted"] = len(shift_schedule)
        outputs["peak_reduction_kw"] = str(actual_reduction)
        outputs["new_peak_kw"] = str(Decimal(str(peak_kw)) - actual_reduction)
        outputs["target_met"] = float(cumulative_reduction) >= float(reduction_target)
        outputs["constraint_violations"] = violations

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 ScheduleOptimization: %d loads shifted, reduction=%.0f kW, "
            "target_met=%s",
            len(shift_schedule), float(actual_reduction),
            outputs["target_met"],
        )
        return PhaseResult(
            phase_name="schedule_optimization", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: LoadShiftResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
