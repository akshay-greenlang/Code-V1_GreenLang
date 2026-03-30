# -*- coding: utf-8 -*-
"""
Implementation Planning Workflow
===================================

3-phase workflow for detailed engineering design, procurement, and
commissioning of peak shaving equipment within PACK-038 Peak Shaving Pack.

Phases:
    1. SolutionDesign    -- Detailed engineering specifications
    2. Procurement       -- Vendor comparison and selection criteria
    3. Commissioning     -- Installation verification and commissioning tests

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - NEC Article 706 (Energy Storage Systems)
    - NFPA 855 (Standard for BESS Installation)
    - UL 9540 / IEC 62619 (Battery Safety)
    - IEEE 1547 (Interconnection Standards)

Schedule: on-demand / project-based
Estimated duration: 20 minutes

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

class MilestoneStatus(str, Enum):
    """Implementation milestone status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    BLOCKED = "blocked"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

IMPLEMENTATION_MILESTONES: Dict[str, Dict[str, Any]] = {
    "site_assessment": {
        "name": "Site Assessment & Survey",
        "typical_duration_days": 5,
        "predecessor": None,
        "deliverables": ["site_survey_report", "electrical_single_line_diagram"],
        "responsible": "engineering",
    },
    "utility_interconnection": {
        "name": "Utility Interconnection Application",
        "typical_duration_days": 30,
        "predecessor": "site_assessment",
        "deliverables": ["interconnection_agreement", "utility_approval"],
        "responsible": "engineering",
    },
    "permitting": {
        "name": "Permitting & Code Compliance",
        "typical_duration_days": 45,
        "predecessor": "site_assessment",
        "deliverables": ["building_permit", "electrical_permit", "fire_permit"],
        "responsible": "engineering",
    },
    "equipment_procurement": {
        "name": "Equipment Procurement & Ordering",
        "typical_duration_days": 14,
        "predecessor": "site_assessment",
        "deliverables": ["purchase_orders", "delivery_schedule"],
        "responsible": "procurement",
    },
    "equipment_delivery": {
        "name": "Equipment Delivery & Inspection",
        "typical_duration_days": 60,
        "predecessor": "equipment_procurement",
        "deliverables": ["delivery_receipt", "inspection_report"],
        "responsible": "procurement",
    },
    "civil_works": {
        "name": "Civil Works & Foundation",
        "typical_duration_days": 15,
        "predecessor": "permitting",
        "deliverables": ["foundation_certification", "drainage_plan"],
        "responsible": "construction",
    },
    "electrical_installation": {
        "name": "Electrical Installation",
        "typical_duration_days": 20,
        "predecessor": "equipment_delivery",
        "deliverables": ["wiring_diagrams", "grounding_report"],
        "responsible": "construction",
    },
    "bms_integration": {
        "name": "BMS / EMS Integration & Programming",
        "typical_duration_days": 10,
        "predecessor": "electrical_installation",
        "deliverables": ["controls_schematic", "setpoint_schedule"],
        "responsible": "controls",
    },
    "safety_systems": {
        "name": "Safety Systems Installation (Fire, HVAC)",
        "typical_duration_days": 7,
        "predecessor": "electrical_installation",
        "deliverables": ["fire_suppression_cert", "ventilation_report"],
        "responsible": "construction",
    },
    "functional_testing": {
        "name": "Functional Testing & Pre-commissioning",
        "typical_duration_days": 5,
        "predecessor": "bms_integration",
        "deliverables": ["functional_test_report", "punch_list"],
        "responsible": "commissioning",
    },
    "commissioning": {
        "name": "Commissioning & Performance Verification",
        "typical_duration_days": 7,
        "predecessor": "functional_testing",
        "deliverables": ["commissioning_report", "performance_certificate"],
        "responsible": "commissioning",
    },
    "handover": {
        "name": "Handover & Operator Training",
        "typical_duration_days": 3,
        "predecessor": "commissioning",
        "deliverables": ["O&M_manual", "training_records", "warranty_docs"],
        "responsible": "commissioning",
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

class ImplementationInput(BaseModel):
    """Input data model for ImplementationWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    project_name: str = Field(default="Peak Shaving Implementation", description="Project name")
    solution_type: str = Field(default="bess", description="bess|load_shift|thermal|hybrid")
    power_kw: Decimal = Field(default=Decimal("0"), gt=0, description="System power rating kW")
    energy_kwh: Decimal = Field(default=Decimal("0"), ge=0, description="System energy capacity kWh")
    total_budget: Decimal = Field(default=Decimal("0"), ge=0, description="Total project budget $")
    target_completion_days: int = Field(default=180, ge=30, le=730, description="Target completion days")
    vendor_quotes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Vendor quote data: vendor_name, total_cost, warranty_years, lead_time_days",
    )
    site_constraints: Dict[str, Any] = Field(
        default_factory=lambda: {
            "available_space_sqft": 500,
            "electrical_capacity_kva": 2000,
            "structural_capacity_psf": 150,
        },
        description="Site physical constraints",
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

class ImplementationResult(BaseModel):
    """Complete result from implementation planning workflow."""

    implementation_id: str = Field(..., description="Unique implementation execution ID")
    facility_id: str = Field(default="", description="Facility identifier")
    project_name: str = Field(default="", description="Project name")
    solution_specs: Dict[str, Any] = Field(default_factory=dict)
    milestone_schedule: List[Dict[str, Any]] = Field(default_factory=list)
    total_duration_days: int = Field(default=0, ge=0)
    critical_path_days: int = Field(default=0, ge=0)
    vendor_recommendation: Dict[str, Any] = Field(default_factory=dict)
    commissioning_checklist: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_total_cost: Decimal = Field(default=Decimal("0"), ge=0)
    implementation_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ImplementationWorkflow:
    """
    3-phase implementation planning workflow for peak shaving projects.

    Generates detailed engineering specifications, evaluates vendor quotes,
    and produces commissioning checklists with milestone schedules.

    Zero-hallucination: all cost estimates use vendor-provided data and
    published cost benchmarks. No LLM calls in the numeric computation path.

    Attributes:
        implementation_id: Unique implementation execution identifier.
        _specs: Solution design specifications.
        _procurement: Procurement evaluation results.
        _commissioning: Commissioning plan data.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ImplementationWorkflow()
        >>> inp = ImplementationInput(
        ...     facility_name="Building F",
        ...     power_kw=Decimal("500"),
        ...     energy_kwh=Decimal("2000"),
        ... )
        >>> result = wf.run(inp)
        >>> assert len(result.milestone_schedule) > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ImplementationWorkflow."""
        self.implementation_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._specs: Dict[str, Any] = {}
        self._procurement: Dict[str, Any] = {}
        self._commissioning: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: ImplementationInput) -> ImplementationResult:
        """
        Execute the 3-phase implementation planning workflow.

        Args:
            input_data: Validated implementation input.

        Returns:
            ImplementationResult with specs, schedule, and commissioning plan.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        self.logger.info(
            "Starting implementation workflow %s for facility=%s solution=%s",
            self.implementation_id, input_data.facility_name, input_data.solution_type,
        )

        self._phase_results = []
        self._specs = {}
        self._procurement = {}
        self._commissioning = {}

        try:
            phase1 = self._phase_solution_design(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_procurement(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_commissioning(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error("Implementation workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Build milestone schedule
        milestones = self._build_milestone_schedule(input_data)
        critical_path = self._calculate_critical_path(milestones)

        result = ImplementationResult(
            implementation_id=self.implementation_id,
            facility_id=input_data.facility_id,
            project_name=input_data.project_name,
            solution_specs=self._specs,
            milestone_schedule=milestones,
            total_duration_days=sum(m.get("duration_days", 0) for m in milestones),
            critical_path_days=critical_path,
            vendor_recommendation=self._procurement.get("recommendation", {}),
            commissioning_checklist=self._commissioning.get("checklist", []),
            estimated_total_cost=Decimal(str(self._procurement.get("estimated_cost", 0))),
            implementation_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Implementation workflow %s completed in %dms milestones=%d "
            "critical_path=%d days",
            self.implementation_id, int(elapsed_ms), len(milestones),
            critical_path,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Solution Design
    # -------------------------------------------------------------------------

    def _phase_solution_design(
        self, input_data: ImplementationInput
    ) -> PhaseResult:
        """Detailed engineering specifications for peak shaving equipment."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        power_kw = input_data.power_kw
        energy_kwh = input_data.energy_kwh
        solution = input_data.solution_type
        site = input_data.site_constraints

        # Space requirements (sqft per kW for different solutions)
        space_req: Dict[str, float] = {
            "bess": 0.8,
            "load_shift": 0.1,
            "thermal": 1.5,
            "hybrid": 1.0,
        }
        required_sqft = round(float(power_kw) * space_req.get(solution, 1.0), 0)
        available_sqft = site.get("available_space_sqft", 500)
        space_adequate = required_sqft <= available_sqft

        # Electrical requirements
        apparent_power_kva = (power_kw / Decimal("0.95")).quantize(Decimal("0.1"))
        available_kva = Decimal(str(site.get("electrical_capacity_kva", 2000)))
        electrical_adequate = apparent_power_kva <= available_kva

        # Weight / structural (BESS specific)
        if solution in ("bess", "hybrid"):
            weight_kg_per_kwh = 12  # Average for containerised BESS
            total_weight_kg = float(energy_kwh) * weight_kg_per_kwh
            weight_psf = total_weight_kg * 2.205 / max(required_sqft, 1)
            structural_adequate = weight_psf <= site.get("structural_capacity_psf", 150)
        else:
            total_weight_kg = 0
            weight_psf = 0
            structural_adequate = True

        self._specs = {
            "solution_type": solution,
            "power_kw": str(power_kw),
            "energy_kwh": str(energy_kwh),
            "apparent_power_kva": str(apparent_power_kva),
            "required_sqft": required_sqft,
            "space_adequate": space_adequate,
            "electrical_adequate": electrical_adequate,
            "structural_adequate": structural_adequate,
            "total_weight_kg": round(total_weight_kg, 0),
            "interconnection_voltage": "480V 3-phase" if float(power_kw) < 1000 else "4.16kV 3-phase",
            "protection_requirements": [
                "AC disconnect switch",
                "DC disconnect switch",
                "Ground fault protection",
                "Arc flash protection",
                "Fire suppression (clean agent)",
            ],
            "code_compliance": [
                "NEC Article 706",
                "NFPA 855",
                "UL 9540 / IEC 62619",
                "IEEE 1547",
            ],
        }

        if not space_adequate:
            warnings.append(f"Space shortfall: {required_sqft} sqft required vs {available_sqft} available")
        if not electrical_adequate:
            warnings.append(f"Electrical shortfall: {apparent_power_kva} kVA required vs {available_kva} available")
        if not structural_adequate:
            warnings.append(f"Structural concern: {round(weight_psf, 1)} psf load")

        outputs["solution_type"] = solution
        outputs["power_kw"] = str(power_kw)
        outputs["space_adequate"] = space_adequate
        outputs["electrical_adequate"] = electrical_adequate
        outputs["structural_adequate"] = structural_adequate

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 SolutionDesign: %s %s kW space=%s electrical=%s structural=%s",
            solution, power_kw, space_adequate, electrical_adequate, structural_adequate,
        )
        return PhaseResult(
            phase_name="solution_design", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Procurement
    # -------------------------------------------------------------------------

    def _phase_procurement(
        self, input_data: ImplementationInput
    ) -> PhaseResult:
        """Vendor comparison and selection criteria."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        power_kw = input_data.power_kw
        energy_kwh = input_data.energy_kwh

        if input_data.vendor_quotes:
            # Score vendors
            scored_vendors: List[Dict[str, Any]] = []
            for quote in input_data.vendor_quotes:
                cost = Decimal(str(quote.get("total_cost", 0)))
                warranty = int(quote.get("warranty_years", 5))
                lead_time = int(quote.get("lead_time_days", 90))

                # Scoring: lower cost, longer warranty, shorter lead time = better
                cost_per_kw = float(cost) / max(float(power_kw), 1)
                cost_score = max(0, 100 - cost_per_kw / 10)
                warranty_score = min(100, warranty * 8)
                lead_score = max(0, 100 - lead_time / 2)

                composite = 0.45 * cost_score + 0.30 * warranty_score + 0.25 * lead_score

                scored_vendors.append({
                    "vendor_name": quote.get("vendor_name", "Unknown"),
                    "total_cost": str(cost),
                    "cost_per_kw": round(cost_per_kw, 2),
                    "warranty_years": warranty,
                    "lead_time_days": lead_time,
                    "composite_score": round(composite, 1),
                })

            scored_vendors.sort(key=lambda x: x["composite_score"], reverse=True)
            recommendation = scored_vendors[0] if scored_vendors else {}
            estimated_cost = float(recommendation.get("total_cost", 0))

        else:
            # Estimate from benchmark costs
            warnings.append("No vendor quotes provided; using benchmark cost estimates")
            benchmark_per_kwh = Decimal("350")
            benchmark_per_kw = Decimal("250")
            equipment = (energy_kwh * benchmark_per_kwh + power_kw * benchmark_per_kw)
            bos = equipment * Decimal("0.15")
            install = equipment * Decimal("0.12")
            estimated_cost = float(equipment + bos + install)
            scored_vendors = []
            recommendation = {
                "vendor_name": "Benchmark Estimate",
                "total_cost": str(round(estimated_cost, 2)),
                "cost_per_kw": round(estimated_cost / max(float(power_kw), 1), 2),
                "note": "Based on industry benchmark pricing",
            }

        self._procurement = {
            "vendors_evaluated": len(scored_vendors),
            "scored_vendors": scored_vendors,
            "recommendation": recommendation,
            "estimated_cost": str(round(estimated_cost, 2)),
        }

        outputs["vendors_evaluated"] = len(scored_vendors)
        outputs["recommended_vendor"] = recommendation.get("vendor_name", "none")
        outputs["estimated_cost"] = str(round(estimated_cost, 2))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 Procurement: %d vendors, recommended=%s cost=$%.0f",
            len(scored_vendors), recommendation.get("vendor_name"),
            estimated_cost,
        )
        return PhaseResult(
            phase_name="procurement", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Commissioning
    # -------------------------------------------------------------------------

    def _phase_commissioning(
        self, input_data: ImplementationInput
    ) -> PhaseResult:
        """Installation verification and commissioning tests."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        checklist: List[Dict[str, Any]] = [
            {
                "test_id": "CT-001",
                "test_name": "Visual Inspection & Documentation Review",
                "category": "pre_energisation",
                "acceptance_criteria": "All documentation complete, no visible defects",
                "duration_hours": 4,
            },
            {
                "test_id": "CT-002",
                "test_name": "Electrical Continuity & Insulation Test",
                "category": "pre_energisation",
                "acceptance_criteria": "Insulation resistance > 1 MOhm",
                "duration_hours": 2,
            },
            {
                "test_id": "CT-003",
                "test_name": "Protection System Verification",
                "category": "pre_energisation",
                "acceptance_criteria": "All protection relays trip within spec",
                "duration_hours": 4,
            },
            {
                "test_id": "CT-004",
                "test_name": "Initial Energisation & SOC Calibration",
                "category": "energisation",
                "acceptance_criteria": "System energises without fault, SOC reads correctly",
                "duration_hours": 6,
            },
            {
                "test_id": "CT-005",
                "test_name": "Charge/Discharge Cycle Test",
                "category": "functional",
                "acceptance_criteria": "Round-trip efficiency within 2% of spec",
                "duration_hours": 8,
            },
            {
                "test_id": "CT-006",
                "test_name": "Peak Shaving Dispatch Test",
                "category": "functional",
                "acceptance_criteria": "Demand reduced by target kW within 1 minute",
                "duration_hours": 4,
            },
            {
                "test_id": "CT-007",
                "test_name": "BMS/EMS Communication Test",
                "category": "integration",
                "acceptance_criteria": "All signals read/write correctly, latency < 500ms",
                "duration_hours": 4,
            },
            {
                "test_id": "CT-008",
                "test_name": "Fire Suppression System Test",
                "category": "safety",
                "acceptance_criteria": "System activates within 30 seconds of trigger",
                "duration_hours": 2,
            },
            {
                "test_id": "CT-009",
                "test_name": "Thermal Management Verification",
                "category": "safety",
                "acceptance_criteria": "Operating temp within spec under full load",
                "duration_hours": 8,
            },
            {
                "test_id": "CT-010",
                "test_name": "72-Hour Reliability Run",
                "category": "performance",
                "acceptance_criteria": "No faults, availability > 99%, performance > 95%",
                "duration_hours": 72,
            },
        ]

        total_test_hours = sum(t["duration_hours"] for t in checklist)

        self._commissioning = {
            "checklist": checklist,
            "total_tests": len(checklist),
            "total_test_hours": total_test_hours,
            "estimated_commissioning_days": max(1, total_test_hours // 10),
        }

        outputs["total_tests"] = len(checklist)
        outputs["total_test_hours"] = total_test_hours
        outputs["estimated_commissioning_days"] = max(1, total_test_hours // 10)
        outputs["categories"] = list(set(t["category"] for t in checklist))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 Commissioning: %d tests, %d hours total",
            len(checklist), total_test_hours,
        )
        return PhaseResult(
            phase_name="commissioning", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Schedule Helpers
    # -------------------------------------------------------------------------

    def _build_milestone_schedule(
        self, input_data: ImplementationInput
    ) -> List[Dict[str, Any]]:
        """Build project milestone schedule from reference data."""
        milestones: List[Dict[str, Any]] = []
        for ms_key, ms_data in IMPLEMENTATION_MILESTONES.items():
            milestones.append({
                "milestone_key": ms_key,
                "name": ms_data["name"],
                "duration_days": ms_data["typical_duration_days"],
                "predecessor": ms_data["predecessor"],
                "deliverables": ms_data["deliverables"],
                "responsible": ms_data["responsible"],
                "status": "not_started",
            })
        return milestones

    def _calculate_critical_path(
        self, milestones: List[Dict[str, Any]]
    ) -> int:
        """Calculate critical path duration in days."""
        # Build dependency graph and find longest path
        by_key: Dict[str, Dict[str, Any]] = {
            m["milestone_key"]: m for m in milestones
        }

        def _earliest_finish(key: str, memo: Dict[str, int]) -> int:
            """Recursive earliest finish calculation."""
            if key in memo:
                return memo[key]
            ms = by_key.get(key)
            if not ms:
                return 0
            pred = ms.get("predecessor")
            if pred and pred in by_key:
                pred_finish = _earliest_finish(pred, memo)
            else:
                pred_finish = 0
            finish = pred_finish + ms["duration_days"]
            memo[key] = finish
            return finish

        memo: Dict[str, int] = {}
        for key in by_key:
            _earliest_finish(key, memo)

        return max(memo.values()) if memo else 0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ImplementationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
