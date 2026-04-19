# -*- coding: utf-8 -*-
"""
Flexibility Assessment Workflow
===================================

4-phase workflow for assessing load flexibility potential and curtailment
capacity within PACK-037 Demand Response Pack.

Phases:
    1. LoadInventory         -- Import and classify all facility loads
    2. FlexibilityScoring    -- Score each load for DR flexibility
    3. CurtailmentCapacity   -- Calculate aggregate curtailment capacity (kW)
    4. FlexibilityReport     -- Compile flexibility assessment report

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - FERC Order 2222 (DER aggregation)
    - NAESB WEQ Business Practice Standards
    - ISO/RTO demand response programme rules

Schedule: on-demand / quarterly
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 37.0.0
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

class LoadCriticality(str, Enum):
    """Criticality classification for facility loads."""

    CRITICAL = "critical"
    ESSENTIAL = "essential"
    NON_ESSENTIAL = "non_essential"
    SHEDDABLE = "sheddable"

class FlexibilityTier(str, Enum):
    """Flexibility tier classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Typical load categories with flexibility characteristics
LOAD_FLEXIBILITY_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "hvac_cooling": {
        "category": "hvac",
        "description": "Central cooling / chiller systems",
        "typical_flexibility_pct": 0.30,
        "max_curtail_duration_min": 120,
        "ramp_time_min": 5,
        "criticality_default": "non_essential",
        "recovery_time_min": 30,
    },
    "hvac_heating": {
        "category": "hvac",
        "description": "Heating systems",
        "typical_flexibility_pct": 0.20,
        "max_curtail_duration_min": 90,
        "ramp_time_min": 10,
        "criticality_default": "essential",
        "recovery_time_min": 30,
    },
    "lighting_interior": {
        "category": "lighting",
        "description": "Interior lighting",
        "typical_flexibility_pct": 0.25,
        "max_curtail_duration_min": 240,
        "ramp_time_min": 1,
        "criticality_default": "non_essential",
        "recovery_time_min": 1,
    },
    "lighting_exterior": {
        "category": "lighting",
        "description": "Exterior / parking lighting",
        "typical_flexibility_pct": 0.50,
        "max_curtail_duration_min": 480,
        "ramp_time_min": 1,
        "criticality_default": "sheddable",
        "recovery_time_min": 1,
    },
    "process_motors": {
        "category": "motors",
        "description": "Industrial process motors",
        "typical_flexibility_pct": 0.15,
        "max_curtail_duration_min": 60,
        "ramp_time_min": 15,
        "criticality_default": "essential",
        "recovery_time_min": 20,
    },
    "compressed_air": {
        "category": "motors",
        "description": "Compressed air systems",
        "typical_flexibility_pct": 0.20,
        "max_curtail_duration_min": 90,
        "ramp_time_min": 5,
        "criticality_default": "non_essential",
        "recovery_time_min": 10,
    },
    "refrigeration": {
        "category": "refrigeration",
        "description": "Refrigeration / cold storage",
        "typical_flexibility_pct": 0.25,
        "max_curtail_duration_min": 180,
        "ramp_time_min": 2,
        "criticality_default": "essential",
        "recovery_time_min": 15,
    },
    "ev_charging": {
        "category": "ev",
        "description": "Electric vehicle charging stations",
        "typical_flexibility_pct": 0.80,
        "max_curtail_duration_min": 480,
        "ramp_time_min": 1,
        "criticality_default": "sheddable",
        "recovery_time_min": 1,
    },
    "data_center_cooling": {
        "category": "data_center",
        "description": "Data centre cooling infrastructure",
        "typical_flexibility_pct": 0.10,
        "max_curtail_duration_min": 30,
        "ramp_time_min": 5,
        "criticality_default": "critical",
        "recovery_time_min": 10,
    },
    "water_heating": {
        "category": "water_heating",
        "description": "Domestic / process water heating",
        "typical_flexibility_pct": 0.40,
        "max_curtail_duration_min": 240,
        "ramp_time_min": 1,
        "criticality_default": "sheddable",
        "recovery_time_min": 5,
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

class LoadItem(BaseModel):
    """Individual load within a facility."""

    load_id: str = Field(default_factory=lambda: f"load-{uuid.uuid4().hex[:8]}")
    load_type: str = Field(default="", description="Load type key from benchmarks")
    name: str = Field(default="", description="Human-readable load name")
    rated_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Rated power kW")
    average_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Average demand kW")
    criticality: str = Field(default="non_essential", description="Criticality classification")
    flexibility_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    curtailable_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Curtailable capacity kW")
    flexibility_tier: str = Field(default="none", description="high|medium|low|none")
    max_curtail_duration_min: int = Field(default=0, ge=0)
    ramp_time_min: int = Field(default=0, ge=0)

class FlexibilityAssessmentInput(BaseModel):
    """Input data model for FlexibilityAssessmentWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    facility_type: str = Field(default="commercial", description="commercial|industrial|institutional|retail|warehouse")
    peak_demand_kw: Decimal = Field(..., gt=0, description="Facility peak demand in kW")
    annual_energy_kwh: Decimal = Field(..., ge=0, description="Annual energy consumption kWh")
    loads: List[Dict[str, Any]] = Field(default_factory=list, description="Load inventory data")
    operating_hours: int = Field(default=2500, ge=0, le=8760, description="Annual operating hours")
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

class FlexibilityAssessmentResult(BaseModel):
    """Complete result from flexibility assessment workflow."""

    assessment_id: str = Field(..., description="Unique assessment execution ID")
    facility_id: str = Field(default="", description="Assessed facility ID")
    total_loads_assessed: int = Field(default=0, ge=0)
    loads: List[LoadItem] = Field(default_factory=list)
    total_rated_kw: Decimal = Field(default=Decimal("0"), ge=0)
    total_curtailable_kw: Decimal = Field(default=Decimal("0"), ge=0)
    flexibility_ratio_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    high_flex_loads: int = Field(default=0, ge=0)
    medium_flex_loads: int = Field(default=0, ge=0)
    low_flex_loads: int = Field(default=0, ge=0)
    assessment_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class FlexibilityAssessmentWorkflow:
    """
    4-phase flexibility assessment workflow for demand response readiness.

    Performs load inventory import, criticality classification, flexibility
    scoring, curtailment capacity calculation, and report generation.

    Zero-hallucination: all flexibility scores use deterministic benchmark
    percentages and engineering parameters. No LLM calls in the numeric
    computation path.

    Attributes:
        assessment_id: Unique assessment execution identifier.
        _loads: Assessed load items.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = FlexibilityAssessmentWorkflow()
        >>> inp = FlexibilityAssessmentInput(
        ...     facility_name="Warehouse A",
        ...     peak_demand_kw=Decimal("2000"),
        ...     annual_energy_kwh=Decimal("4000000"),
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_curtailable_kw > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FlexibilityAssessmentWorkflow."""
        self.assessment_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._loads: List[LoadItem] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: FlexibilityAssessmentInput) -> FlexibilityAssessmentResult:
        """
        Execute the 4-phase flexibility assessment workflow.

        Args:
            input_data: Validated flexibility assessment input.

        Returns:
            FlexibilityAssessmentResult with load inventory and curtailment capacity.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        self.logger.info(
            "Starting flexibility assessment workflow %s for facility=%s type=%s",
            self.assessment_id, input_data.facility_name, input_data.facility_type,
        )

        self._phase_results = []
        self._loads = []

        try:
            # Phase 1: Load Inventory
            phase1 = self._phase_load_inventory(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Flexibility Scoring
            phase2 = self._phase_flexibility_scoring(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Curtailment Capacity
            phase3 = self._phase_curtailment_capacity(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Flexibility Report
            phase4 = self._phase_flexibility_report(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("Flexibility assessment workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_rated_kw = sum(ld.rated_kw for ld in self._loads)
        total_curtailable_kw = sum(ld.curtailable_kw for ld in self._loads)
        flexibility_ratio = (
            Decimal(str(round(float(total_curtailable_kw) / float(total_rated_kw) * 100, 2)))
            if total_rated_kw > 0 else Decimal("0")
        )
        high_flex = sum(1 for ld in self._loads if ld.flexibility_tier == "high")
        medium_flex = sum(1 for ld in self._loads if ld.flexibility_tier == "medium")
        low_flex = sum(1 for ld in self._loads if ld.flexibility_tier == "low")
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = FlexibilityAssessmentResult(
            assessment_id=self.assessment_id,
            facility_id=input_data.facility_id,
            total_loads_assessed=len(self._loads),
            loads=self._loads,
            total_rated_kw=total_rated_kw,
            total_curtailable_kw=total_curtailable_kw,
            flexibility_ratio_pct=flexibility_ratio,
            high_flex_loads=high_flex,
            medium_flex_loads=medium_flex,
            low_flex_loads=low_flex,
            assessment_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Flexibility assessment workflow %s completed in %dms loads=%d "
            "curtailable=%.1f kW ratio=%.1f%%",
            self.assessment_id, int(elapsed_ms), len(self._loads),
            float(total_curtailable_kw), float(flexibility_ratio),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Load Inventory
    # -------------------------------------------------------------------------

    def _phase_load_inventory(
        self, input_data: FlexibilityAssessmentInput
    ) -> PhaseResult:
        """Import and classify all facility loads."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if input_data.loads:
            # Use provided load data
            for load_dict in input_data.loads:
                load_type = load_dict.get("load_type", "")
                benchmark = LOAD_FLEXIBILITY_BENCHMARKS.get(load_type, {})

                load_item = LoadItem(
                    load_id=load_dict.get("load_id", f"load-{uuid.uuid4().hex[:8]}"),
                    load_type=load_type,
                    name=load_dict.get("name", benchmark.get("description", load_type)),
                    rated_kw=Decimal(str(load_dict.get("rated_kw", 0))),
                    average_kw=Decimal(str(load_dict.get("average_kw", 0))),
                    criticality=load_dict.get(
                        "criticality", benchmark.get("criticality_default", "non_essential")
                    ),
                    max_curtail_duration_min=int(
                        load_dict.get("max_curtail_duration_min",
                                      benchmark.get("max_curtail_duration_min", 0))
                    ),
                    ramp_time_min=int(
                        load_dict.get("ramp_time_min", benchmark.get("ramp_time_min", 0))
                    ),
                )
                self._loads.append(load_item)
        else:
            # Auto-generate from benchmarks using peak demand allocation
            warnings.append("No load inventory provided; generating from benchmarks")
            self._generate_benchmark_loads(input_data)

        outputs["loads_imported"] = len(self._loads)
        outputs["facility_id"] = input_data.facility_id
        outputs["peak_demand_kw"] = str(input_data.peak_demand_kw)
        outputs["load_types"] = list(set(ld.load_type for ld in self._loads))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 LoadInventory: %d loads imported for facility=%s",
            len(self._loads), input_data.facility_name,
        )
        return PhaseResult(
            phase_name="load_inventory", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_benchmark_loads(
        self, input_data: FlexibilityAssessmentInput
    ) -> None:
        """Generate benchmark loads from peak demand allocation."""
        peak_kw = float(input_data.peak_demand_kw)
        facility_type = input_data.facility_type

        # Allocation percentages by facility type
        allocations: Dict[str, Dict[str, float]] = {
            "commercial": {
                "hvac_cooling": 0.35, "lighting_interior": 0.25,
                "lighting_exterior": 0.05, "water_heating": 0.10,
                "compressed_air": 0.05, "ev_charging": 0.05,
            },
            "industrial": {
                "hvac_cooling": 0.15, "process_motors": 0.35,
                "compressed_air": 0.15, "lighting_interior": 0.10,
                "refrigeration": 0.10, "water_heating": 0.05,
            },
            "warehouse": {
                "hvac_cooling": 0.20, "lighting_interior": 0.20,
                "lighting_exterior": 0.10, "refrigeration": 0.20,
                "ev_charging": 0.10, "compressed_air": 0.10,
            },
        }

        alloc = allocations.get(facility_type, allocations["commercial"])
        for load_type, pct in alloc.items():
            benchmark = LOAD_FLEXIBILITY_BENCHMARKS.get(load_type, {})
            rated_kw = round(peak_kw * pct, 1)
            if rated_kw <= 0:
                continue

            load_item = LoadItem(
                load_type=load_type,
                name=benchmark.get("description", load_type),
                rated_kw=Decimal(str(rated_kw)),
                average_kw=Decimal(str(round(rated_kw * 0.65, 1))),
                criticality=benchmark.get("criticality_default", "non_essential"),
                max_curtail_duration_min=benchmark.get("max_curtail_duration_min", 0),
                ramp_time_min=benchmark.get("ramp_time_min", 0),
            )
            self._loads.append(load_item)

    # -------------------------------------------------------------------------
    # Phase 2: Flexibility Scoring
    # -------------------------------------------------------------------------

    def _phase_flexibility_scoring(
        self, input_data: FlexibilityAssessmentInput
    ) -> PhaseResult:
        """Score each load for DR flexibility potential."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for load_item in self._loads:
            benchmark = LOAD_FLEXIBILITY_BENCHMARKS.get(load_item.load_type, {})

            # Flexibility score: composite of criticality, duration, and benchmark pct
            criticality_score = self._criticality_to_score(load_item.criticality)
            duration_score = min(100.0, load_item.max_curtail_duration_min / 4.8)
            benchmark_pct = benchmark.get("typical_flexibility_pct", 0.10)
            benchmark_score = benchmark_pct * 100.0

            # Weighted composite: criticality 40%, duration 30%, benchmark 30%
            composite = (
                0.40 * criticality_score
                + 0.30 * duration_score
                + 0.30 * benchmark_score
            )
            load_item.flexibility_score = Decimal(str(round(composite, 2)))

            # Assign tier
            if composite >= 60:
                load_item.flexibility_tier = "high"
            elif composite >= 35:
                load_item.flexibility_tier = "medium"
            elif composite > 10:
                load_item.flexibility_tier = "low"
            else:
                load_item.flexibility_tier = "none"

        tier_counts = {}
        for ld in self._loads:
            tier_counts[ld.flexibility_tier] = tier_counts.get(ld.flexibility_tier, 0) + 1

        outputs["loads_scored"] = len(self._loads)
        outputs["tier_distribution"] = tier_counts
        outputs["avg_flexibility_score"] = str(round(
            float(sum(ld.flexibility_score for ld in self._loads)) / max(len(self._loads), 1), 2
        ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 FlexibilityScoring: %d loads scored, tiers=%s",
            len(self._loads), tier_counts,
        )
        return PhaseResult(
            phase_name="flexibility_scoring", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _criticality_to_score(self, criticality: str) -> float:
        """Convert criticality classification to a numeric score (0-100)."""
        mapping = {
            "sheddable": 100.0,
            "non_essential": 70.0,
            "essential": 30.0,
            "critical": 5.0,
        }
        return mapping.get(criticality, 50.0)

    # -------------------------------------------------------------------------
    # Phase 3: Curtailment Capacity
    # -------------------------------------------------------------------------

    def _phase_curtailment_capacity(
        self, input_data: FlexibilityAssessmentInput
    ) -> PhaseResult:
        """Calculate aggregate curtailment capacity in kW for each load."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_curtailable = Decimal("0")
        for load_item in self._loads:
            benchmark = LOAD_FLEXIBILITY_BENCHMARKS.get(load_item.load_type, {})
            flex_pct = Decimal(str(benchmark.get("typical_flexibility_pct", 0.10)))

            # Curtailable kW = rated_kw * benchmark flexibility percentage
            # Adjusted by criticality (critical loads get 50% reduction)
            adjustment = Decimal("1.0")
            if load_item.criticality == "critical":
                adjustment = Decimal("0.50")
            elif load_item.criticality == "essential":
                adjustment = Decimal("0.75")

            curtailable = (load_item.rated_kw * flex_pct * adjustment).quantize(
                Decimal("0.1")
            )
            load_item.curtailable_kw = curtailable
            total_curtailable += curtailable

        flexibility_ratio = (
            Decimal(str(round(
                float(total_curtailable) / float(input_data.peak_demand_kw) * 100, 2
            )))
            if input_data.peak_demand_kw > 0 else Decimal("0")
        )

        outputs["total_curtailable_kw"] = str(total_curtailable)
        outputs["peak_demand_kw"] = str(input_data.peak_demand_kw)
        outputs["flexibility_ratio_pct"] = str(flexibility_ratio)
        outputs["loads_with_capacity"] = sum(
            1 for ld in self._loads if ld.curtailable_kw > 0
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 CurtailmentCapacity: total=%.1f kW, ratio=%.1f%%",
            float(total_curtailable), float(flexibility_ratio),
        )
        return PhaseResult(
            phase_name="curtailment_capacity", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Flexibility Report
    # -------------------------------------------------------------------------

    def _phase_flexibility_report(
        self, input_data: FlexibilityAssessmentInput
    ) -> PhaseResult:
        """Compile flexibility assessment report."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_rated = sum(ld.rated_kw for ld in self._loads)
        total_curtailable = sum(ld.curtailable_kw for ld in self._loads)
        high_flex = [ld for ld in self._loads if ld.flexibility_tier == "high"]
        by_category: Dict[str, Decimal] = {}
        for ld in self._loads:
            cat = ld.load_type.split("_")[0] if ld.load_type else "other"
            by_category[cat] = by_category.get(cat, Decimal("0")) + ld.curtailable_kw

        outputs["total_loads"] = len(self._loads)
        outputs["total_rated_kw"] = str(total_rated)
        outputs["total_curtailable_kw"] = str(total_curtailable)
        outputs["high_flexibility_loads"] = len(high_flex)
        outputs["top_5_curtailable"] = [
            {"name": ld.name, "curtailable_kw": str(ld.curtailable_kw),
             "tier": ld.flexibility_tier}
            for ld in sorted(self._loads, key=lambda x: x.curtailable_kw, reverse=True)[:5]
        ]
        outputs["curtailable_by_category"] = {k: str(v) for k, v in by_category.items()}
        outputs["report_generated_at"] = utcnow().isoformat() + "Z"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 FlexibilityReport: %d loads, %.1f kW curtailable",
            len(self._loads), float(total_curtailable),
        )
        return PhaseResult(
            phase_name="flexibility_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: FlexibilityAssessmentResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
