# -*- coding: utf-8 -*-
"""
M&V Plan Workflow
===================================

4-phase workflow for developing a comprehensive Measurement & Verification
plan compliant with IPMVP Core Concepts and ISO 50015.

Phases:
    1. ECMReview            -- Review energy conservation measures and scope
    2. OptionSelection      -- Select IPMVP option (A/B/C/D) per ECM
    3. BoundaryDefinition   -- Define measurement boundaries
    4. MeteringPlan         -- Develop metering and data collection plan

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - IPMVP Core Concepts (EVO 10000-1:2022)
    - ISO 50015:2014 (M&V of energy performance)
    - FEMP M&V Guidelines 4.0
    - ASHRAE Guideline 14-2014

Schedule: on-demand / project initiation
Estimated duration: 20 minutes

Author: GreenLang Platform Team
Version: 40.0.0
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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


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


class IPMVPOption(str, Enum):
    """IPMVP measurement and verification options."""

    OPTION_A = "A"
    OPTION_B = "B"
    OPTION_C = "C"
    OPTION_D = "D"


class BoundaryType(str, Enum):
    """Measurement boundary types."""

    RETROFIT_ISOLATION = "retrofit_isolation"
    WHOLE_FACILITY = "whole_facility"
    SUB_METERED = "sub_metered"
    CALIBRATED_SIMULATION = "calibrated_simulation"


class ECMComplexity(str, Enum):
    """Energy conservation measure complexity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

IPMVP_OPTIONS: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Retrofit Isolation: Key Parameter Measurement",
        "description": (
            "Savings determined by field measurement of the key performance "
            "parameter(s) which define the ECM's energy use. Estimation of "
            "remaining parameters allowed."
        ),
        "boundary": "retrofit_isolation",
        "metering_scope": "key_parameters_only",
        "typical_accuracy_pct": 15.0,
        "cost_relative": "low",
        "complexity": "low",
        "best_for": [
            "lighting_retrofits",
            "constant_load_equipment",
            "vfd_installations",
        ],
        "data_requirements": [
            "key_parameter_spot_measurements",
            "operating_hours",
            "nameplate_data",
        ],
        "metering_duration": "short_term",
        "savings_calculation": "engineering_estimate_with_key_measurement",
        "typical_cost_pct_of_ecm": 3.0,
    },
    "B": {
        "name": "Retrofit Isolation: All Parameter Measurement",
        "description": (
            "Savings determined by field measurement of ALL energy use "
            "parameters of the ECM. Continuous metering required."
        ),
        "boundary": "retrofit_isolation",
        "metering_scope": "all_parameters",
        "typical_accuracy_pct": 10.0,
        "cost_relative": "medium",
        "complexity": "medium",
        "best_for": [
            "hvac_upgrades",
            "variable_speed_drives",
            "boiler_replacements",
            "chiller_replacements",
        ],
        "data_requirements": [
            "continuous_energy_metering",
            "operating_schedules",
            "weather_data",
        ],
        "metering_duration": "continuous",
        "savings_calculation": "metered_energy_use_with_regression",
        "typical_cost_pct_of_ecm": 5.0,
    },
    "C": {
        "name": "Whole Facility",
        "description": (
            "Savings determined by measuring energy use at the whole "
            "facility level. Utility meter data with regression analysis."
        ),
        "boundary": "whole_facility",
        "metering_scope": "utility_meters",
        "typical_accuracy_pct": 20.0,
        "cost_relative": "low",
        "complexity": "medium",
        "best_for": [
            "multiple_ecm_projects",
            "whole_building_retrofits",
            "energy_management_programs",
        ],
        "data_requirements": [
            "utility_billing_data",
            "weather_data",
            "occupancy_data",
            "production_data",
        ],
        "metering_duration": "continuous",
        "savings_calculation": "whole_facility_regression",
        "typical_cost_pct_of_ecm": 1.0,
    },
    "D": {
        "name": "Calibrated Simulation",
        "description": (
            "Savings determined by calibrated building energy simulation "
            "model. Model must meet ASHRAE 14 calibration criteria."
        ),
        "boundary": "calibrated_simulation",
        "metering_scope": "simulation_plus_spot",
        "typical_accuracy_pct": 25.0,
        "cost_relative": "high",
        "complexity": "high",
        "best_for": [
            "new_construction",
            "major_renovation",
            "no_baseline_data_available",
            "complex_interactive_systems",
        ],
        "data_requirements": [
            "building_plans",
            "equipment_schedules",
            "weather_data",
            "calibration_measurements",
        ],
        "metering_duration": "spot_and_short_term",
        "savings_calculation": "calibrated_simulation_model",
        "typical_cost_pct_of_ecm": 10.0,
    },
}

BOUNDARY_TYPES: Dict[str, Dict[str, Any]] = {
    "retrofit_isolation": {
        "description": "Boundary around individual ECM equipment",
        "meter_location": "at_equipment",
        "energy_streams": ["electricity", "gas", "steam", "chilled_water"],
        "includes_interactive_effects": False,
        "applicable_options": ["A", "B"],
    },
    "whole_facility": {
        "description": "Boundary around entire facility utility meters",
        "meter_location": "utility_meter",
        "energy_streams": ["electricity", "gas", "fuel_oil", "steam", "district"],
        "includes_interactive_effects": True,
        "applicable_options": ["C"],
    },
    "sub_metered": {
        "description": "Boundary around sub-metered building section",
        "meter_location": "sub_panel",
        "energy_streams": ["electricity", "gas"],
        "includes_interactive_effects": True,
        "applicable_options": ["B", "C"],
    },
    "calibrated_simulation": {
        "description": "Virtual boundary defined by simulation model",
        "meter_location": "model_based",
        "energy_streams": ["all"],
        "includes_interactive_effects": True,
        "applicable_options": ["D"],
    },
}

METERING_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "electricity": {
        "meter_type": "electrical_energy",
        "typical_accuracy_pct": 1.0,
        "calibration_interval_months": 12,
        "data_interval_minutes": 15,
        "parameters": ["kWh", "kW", "kVAR", "PF", "V", "A"],
    },
    "gas": {
        "meter_type": "gas_flow",
        "typical_accuracy_pct": 2.0,
        "calibration_interval_months": 12,
        "data_interval_minutes": 60,
        "parameters": ["m3", "therms", "MJ"],
    },
    "steam": {
        "meter_type": "steam_flow",
        "typical_accuracy_pct": 3.0,
        "calibration_interval_months": 6,
        "data_interval_minutes": 15,
        "parameters": ["kg/h", "GJ", "pressure", "temperature"],
    },
    "chilled_water": {
        "meter_type": "thermal_energy",
        "typical_accuracy_pct": 5.0,
        "calibration_interval_months": 12,
        "data_interval_minutes": 15,
        "parameters": ["kWh_th", "flow_rate", "supply_temp", "return_temp"],
    },
    "temperature": {
        "meter_type": "weather_sensor",
        "typical_accuracy_pct": 0.5,
        "calibration_interval_months": 24,
        "data_interval_minutes": 60,
        "parameters": ["dry_bulb_C", "wet_bulb_C", "RH_pct"],
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


class ECMDefinition(BaseModel):
    """Energy Conservation Measure definition."""

    ecm_id: str = Field(default_factory=lambda: f"ecm-{uuid.uuid4().hex[:8]}")
    ecm_name: str = Field(..., min_length=1, description="ECM display name")
    ecm_type: str = Field(default="lighting", description="ECM category")
    description: str = Field(default="", description="ECM description")
    estimated_savings_pct: float = Field(
        default=10.0, ge=0, le=100,
        description="Estimated savings percentage",
    )
    estimated_savings_kwh: float = Field(
        default=0.0, ge=0, description="Estimated annual savings kWh",
    )
    estimated_cost: float = Field(
        default=0.0, ge=0, description="Implementation cost",
    )
    energy_streams: List[str] = Field(
        default_factory=lambda: ["electricity"],
        description="Affected energy streams",
    )
    interactive_effects: bool = Field(
        default=False, description="Has interactive effects with other systems",
    )
    complexity: str = Field(default="low", description="ECM complexity: low/medium/high")
    preferred_option: str = Field(default="", description="Preferred IPMVP option if any")


class MVPlanInput(BaseModel):
    """Input data model for MVPlanWorkflow."""

    project_id: str = Field(default_factory=lambda: f"proj-{uuid.uuid4().hex[:8]}")
    project_name: str = Field(..., min_length=1, description="Project name")
    facility_id: str = Field(default="", description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    ecm_list: List[ECMDefinition] = Field(
        default_factory=list, description="Energy conservation measures",
    )
    total_project_cost: float = Field(
        default=0.0, ge=0, description="Total project implementation cost",
    )
    mv_budget_pct: float = Field(
        default=5.0, ge=0, le=100,
        description="M&V budget as percentage of project cost",
    )
    baseline_data_available: bool = Field(
        default=True, description="Whether baseline energy data is available",
    )
    contract_type: str = Field(
        default="performance_contract",
        description="Contract type: performance_contract, utility_incentive, internal",
    )
    reporting_frequency: str = Field(
        default="monthly", description="M&V reporting frequency",
    )
    mv_duration_years: int = Field(
        default=3, ge=1, le=20, description="M&V period duration in years",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        """Ensure project name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("project_name must not be blank")
        return stripped


class MVPlanResult(BaseModel):
    """Complete result from M&V plan workflow."""

    plan_id: str = Field(..., description="Unique M&V plan ID")
    project_id: str = Field(default="", description="Project identifier")
    project_name: str = Field(default="", description="Project name")
    ecm_count: int = Field(default=0, ge=0, description="Number of ECMs reviewed")
    option_assignments: List[Dict[str, Any]] = Field(
        default_factory=list, description="IPMVP option per ECM",
    )
    boundary_definitions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Measurement boundary per ECM",
    )
    metering_plan: Dict[str, Any] = Field(
        default_factory=dict, description="Metering and data collection plan",
    )
    total_meters_required: int = Field(default=0, ge=0)
    estimated_mv_cost: Decimal = Field(default=Decimal("0"), ge=0)
    mv_cost_pct_of_project: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    mv_duration_years: int = Field(default=3, ge=1)
    reporting_frequency: str = Field(default="monthly")
    compliance_frameworks: List[str] = Field(default_factory=list)
    phases_completed: List[str] = Field(default_factory=list)
    workflow_duration_ms: int = Field(default=0, ge=0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MVPlanWorkflow:
    """
    4-phase M&V plan development workflow.

    Develops a comprehensive M&V plan by reviewing ECMs, selecting IPMVP
    options, defining measurement boundaries, and creating metering plans.

    Zero-hallucination: all option selection criteria and cost estimates are
    derived from IPMVP reference data. No LLM calls in the decision path.

    Attributes:
        plan_id: Unique plan execution identifier.
        _ecm_reviews: ECM review results.
        _option_assignments: IPMVP option assignments per ECM.
        _boundaries: Measurement boundary definitions.
        _metering_plan: Metering plan details.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = MVPlanWorkflow()
        >>> ecm = ECMDefinition(ecm_name="LED Lighting Retrofit")
        >>> inp = MVPlanInput(project_name="HQ Retrofit", ecm_list=[ecm])
        >>> result = wf.run(inp)
        >>> assert result.ecm_count > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MVPlanWorkflow."""
        self.plan_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._ecm_reviews: List[Dict[str, Any]] = []
        self._option_assignments: List[Dict[str, Any]] = []
        self._boundaries: List[Dict[str, Any]] = []
        self._metering_plan: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: MVPlanInput) -> MVPlanResult:
        """
        Execute the 4-phase M&V plan workflow.

        Args:
            input_data: Validated M&V plan input.

        Returns:
            MVPlanResult with option assignments, boundaries, and metering plan.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting M&V plan workflow %s for project=%s ecms=%d",
            self.plan_id, input_data.project_name, len(input_data.ecm_list),
        )

        self._phase_results = []
        self._ecm_reviews = []
        self._option_assignments = []
        self._boundaries = []
        self._metering_plan = {}

        try:
            # Phase 1: ECM Review
            phase1 = self._phase_ecm_review(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Option Selection
            phase2 = self._phase_option_selection(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Boundary Definition
            phase3 = self._phase_boundary_definition(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Metering Plan
            phase4 = self._phase_metering_plan(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error(
                "M&V plan workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Calculate total meters and cost
        total_meters = self._metering_plan.get("total_meters", 0)
        mv_cost = Decimal(str(self._metering_plan.get("estimated_cost", 0)))
        project_cost = max(input_data.total_project_cost, 1.0)
        mv_cost_pct = Decimal(str(round(float(mv_cost) / project_cost * 100, 2)))

        result = MVPlanResult(
            plan_id=self.plan_id,
            project_id=input_data.project_id,
            project_name=input_data.project_name,
            ecm_count=len(input_data.ecm_list),
            option_assignments=self._option_assignments,
            boundary_definitions=self._boundaries,
            metering_plan=self._metering_plan,
            total_meters_required=total_meters,
            estimated_mv_cost=mv_cost,
            mv_cost_pct_of_project=mv_cost_pct,
            mv_duration_years=input_data.mv_duration_years,
            reporting_frequency=input_data.reporting_frequency,
            compliance_frameworks=["IPMVP", "ISO 50015", "FEMP 4.0"],
            phases_completed=completed_phases,
            workflow_duration_ms=int(elapsed_ms),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "M&V plan workflow %s completed in %dms ecms=%d meters=%d cost=$%.0f",
            self.plan_id, int(elapsed_ms), len(input_data.ecm_list),
            total_meters, float(mv_cost),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: ECM Review
    # -------------------------------------------------------------------------

    def _phase_ecm_review(self, input_data: MVPlanInput) -> PhaseResult:
        """Review energy conservation measures and scope."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.ecm_list:
            warnings.append("No ECMs provided; creating placeholder ECM")
            input_data.ecm_list.append(ECMDefinition(
                ecm_name="General Energy Efficiency",
                ecm_type="general",
                estimated_savings_pct=10.0,
            ))

        reviews: List[Dict[str, Any]] = []
        total_savings_kwh = 0.0
        total_cost = 0.0

        for ecm in input_data.ecm_list:
            review = {
                "ecm_id": ecm.ecm_id,
                "ecm_name": ecm.ecm_name,
                "ecm_type": ecm.ecm_type,
                "complexity": ecm.complexity,
                "estimated_savings_pct": ecm.estimated_savings_pct,
                "estimated_savings_kwh": ecm.estimated_savings_kwh,
                "estimated_cost": ecm.estimated_cost,
                "energy_streams": ecm.energy_streams,
                "interactive_effects": ecm.interactive_effects,
                "review_status": "reviewed",
                "reviewed_at": _utcnow().isoformat() + "Z",
            }

            # Validate savings estimates
            if ecm.estimated_savings_pct > 50.0:
                warnings.append(
                    f"ECM '{ecm.ecm_name}' claims {ecm.estimated_savings_pct}% savings "
                    f"which is unusually high; verify estimate"
                )

            total_savings_kwh += ecm.estimated_savings_kwh
            total_cost += ecm.estimated_cost
            reviews.append(review)

        self._ecm_reviews = reviews

        outputs["ecms_reviewed"] = len(reviews)
        outputs["total_estimated_savings_kwh"] = round(total_savings_kwh, 2)
        outputs["total_estimated_cost"] = round(total_cost, 2)
        outputs["ecm_types"] = list(set(e["ecm_type"] for e in reviews))
        outputs["has_interactive_effects"] = any(
            e["interactive_effects"] for e in reviews
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 ECMReview: %d ECMs reviewed, total savings=%,.0f kWh",
            len(reviews), total_savings_kwh,
        )
        return PhaseResult(
            phase_name="ecm_review", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Option Selection
    # -------------------------------------------------------------------------

    def _phase_option_selection(self, input_data: MVPlanInput) -> PhaseResult:
        """Select IPMVP option (A/B/C/D) per ECM."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        assignments: List[Dict[str, Any]] = []
        option_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}

        for ecm in input_data.ecm_list:
            selected_option = self._select_option_for_ecm(ecm, input_data)
            option_spec = IPMVP_OPTIONS[selected_option]

            assignment = {
                "ecm_id": ecm.ecm_id,
                "ecm_name": ecm.ecm_name,
                "selected_option": selected_option,
                "option_name": option_spec["name"],
                "rationale": self._build_option_rationale(ecm, selected_option),
                "typical_accuracy_pct": option_spec["typical_accuracy_pct"],
                "typical_cost_pct_of_ecm": option_spec["typical_cost_pct_of_ecm"],
                "metering_scope": option_spec["metering_scope"],
                "metering_duration": option_spec["metering_duration"],
                "data_requirements": option_spec["data_requirements"],
                "assigned_at": _utcnow().isoformat() + "Z",
            }
            assignments.append(assignment)
            option_counts[selected_option] = option_counts.get(selected_option, 0) + 1

        self._option_assignments = assignments

        outputs["assignments"] = len(assignments)
        outputs["option_distribution"] = {
            k: v for k, v in option_counts.items() if v > 0
        }
        outputs["avg_accuracy_pct"] = round(
            sum(a["typical_accuracy_pct"] for a in assignments) /
            max(len(assignments), 1), 1,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 OptionSelection: %d assignments, distribution=%s",
            len(assignments), outputs["option_distribution"],
        )
        return PhaseResult(
            phase_name="option_selection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Boundary Definition
    # -------------------------------------------------------------------------

    def _phase_boundary_definition(self, input_data: MVPlanInput) -> PhaseResult:
        """Define measurement boundaries for each ECM."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        boundaries: List[Dict[str, Any]] = []

        for assignment in self._option_assignments:
            option = assignment["selected_option"]
            option_spec = IPMVP_OPTIONS[option]
            boundary_key = option_spec["boundary"]
            boundary_spec = BOUNDARY_TYPES.get(
                boundary_key, BOUNDARY_TYPES["retrofit_isolation"]
            )

            ecm = self._find_ecm(input_data.ecm_list, assignment["ecm_id"])
            energy_streams = ecm.energy_streams if ecm else ["electricity"]

            boundary = {
                "ecm_id": assignment["ecm_id"],
                "ecm_name": assignment["ecm_name"],
                "boundary_type": boundary_key,
                "boundary_description": boundary_spec["description"],
                "meter_location": boundary_spec["meter_location"],
                "energy_streams": energy_streams,
                "includes_interactive_effects": boundary_spec[
                    "includes_interactive_effects"
                ],
                "measurement_points": self._determine_measurement_points(
                    energy_streams, boundary_key,
                ),
                "defined_at": _utcnow().isoformat() + "Z",
            }
            boundaries.append(boundary)

        self._boundaries = boundaries

        outputs["boundaries_defined"] = len(boundaries)
        outputs["boundary_types_used"] = list(set(
            b["boundary_type"] for b in boundaries
        ))
        outputs["total_measurement_points"] = sum(
            len(b.get("measurement_points", [])) for b in boundaries
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 BoundaryDefinition: %d boundaries, types=%s",
            len(boundaries), outputs["boundary_types_used"],
        )
        return PhaseResult(
            phase_name="boundary_definition", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Metering Plan
    # -------------------------------------------------------------------------

    def _phase_metering_plan(self, input_data: MVPlanInput) -> PhaseResult:
        """Develop metering and data collection plan."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        meter_list: List[Dict[str, Any]] = []
        total_cost = 0.0

        all_streams: List[str] = []
        for boundary in self._boundaries:
            all_streams.extend(boundary.get("energy_streams", []))
        unique_streams = list(set(all_streams))

        for stream in unique_streams:
            meter_spec = METERING_REQUIREMENTS.get(stream)
            if not meter_spec:
                warnings.append(f"No metering spec for stream '{stream}'")
                continue

            meter_entry = {
                "meter_id": f"mtr-{_new_uuid()[:8]}",
                "energy_stream": stream,
                "meter_type": meter_spec["meter_type"],
                "accuracy_pct": meter_spec["typical_accuracy_pct"],
                "calibration_interval_months": meter_spec["calibration_interval_months"],
                "data_interval_minutes": meter_spec["data_interval_minutes"],
                "parameters": meter_spec["parameters"],
                "estimated_install_cost": self._estimate_meter_cost(stream),
                "annual_maintenance_cost": self._estimate_annual_maintenance(stream),
            }
            total_cost += meter_entry["estimated_install_cost"]
            total_cost += (
                meter_entry["annual_maintenance_cost"]
                * input_data.mv_duration_years
            )
            meter_list.append(meter_entry)

        # Add weather station if temperature data needed
        if any(
            a["selected_option"] in ("B", "C")
            for a in self._option_assignments
        ):
            if "temperature" not in unique_streams:
                t_spec = METERING_REQUIREMENTS["temperature"]
                meter_list.append({
                    "meter_id": f"mtr-{_new_uuid()[:8]}",
                    "energy_stream": "temperature",
                    "meter_type": t_spec["meter_type"],
                    "accuracy_pct": t_spec["typical_accuracy_pct"],
                    "calibration_interval_months": t_spec["calibration_interval_months"],
                    "data_interval_minutes": t_spec["data_interval_minutes"],
                    "parameters": t_spec["parameters"],
                    "estimated_install_cost": 500.0,
                    "annual_maintenance_cost": 100.0,
                })
                total_cost += 500.0 + 100.0 * input_data.mv_duration_years

        self._metering_plan = {
            "meters": meter_list,
            "total_meters": len(meter_list),
            "estimated_cost": round(total_cost, 2),
            "data_collection_frequency": input_data.reporting_frequency,
            "mv_duration_years": input_data.mv_duration_years,
            "calibration_schedule": self._build_calibration_schedule(meter_list),
            "data_management": {
                "storage": "greenlang_timeseries_db",
                "backup_frequency": "daily",
                "retention_years": input_data.mv_duration_years + 2,
            },
        }

        outputs["total_meters"] = len(meter_list)
        outputs["estimated_cost"] = round(total_cost, 2)
        outputs["energy_streams_metered"] = unique_streams
        outputs["calibration_events_per_year"] = sum(
            12 / m.get("calibration_interval_months", 12) for m in meter_list
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 MeteringPlan: %d meters, cost=$%.0f, duration=%d years",
            len(meter_list), total_cost, input_data.mv_duration_years,
        )
        return PhaseResult(
            phase_name="metering_plan", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _select_option_for_ecm(
        self, ecm: ECMDefinition, input_data: MVPlanInput,
    ) -> str:
        """Deterministically select IPMVP option for an ECM."""
        # Use preferred option if specified and valid
        if ecm.preferred_option and ecm.preferred_option in IPMVP_OPTIONS:
            return ecm.preferred_option

        # No baseline data => Option D
        if not input_data.baseline_data_available:
            return "D"

        # Multiple ECMs with interactive effects => Option C
        if len(input_data.ecm_list) > 3 and ecm.interactive_effects:
            return "C"

        # Complex ECMs with variable loads => Option B
        if ecm.complexity == "high":
            return "B"

        # Simple, constant-load ECMs => Option A
        if ecm.complexity == "low" and not ecm.interactive_effects:
            return "A"

        # Check ECM type against IPMVP best-for lists
        for option_key, option_spec in IPMVP_OPTIONS.items():
            best_for = option_spec.get("best_for", [])
            if ecm.ecm_type in best_for:
                return option_key

        # Default to Option B for medium complexity
        return "B"

    def _build_option_rationale(self, ecm: ECMDefinition, option: str) -> str:
        """Build rationale string for option selection."""
        spec = IPMVP_OPTIONS[option]
        rationale_parts = [
            f"Option {option} ({spec['name']}) selected for '{ecm.ecm_name}'.",
            f"ECM complexity: {ecm.complexity}.",
            f"Interactive effects: {'yes' if ecm.interactive_effects else 'no'}.",
            f"Typical accuracy: +/-{spec['typical_accuracy_pct']}%.",
            f"Metering scope: {spec['metering_scope']}.",
        ]
        return " ".join(rationale_parts)

    def _find_ecm(
        self, ecm_list: List[ECMDefinition], ecm_id: str,
    ) -> Optional[ECMDefinition]:
        """Find ECM by ID."""
        for ecm in ecm_list:
            if ecm.ecm_id == ecm_id:
                return ecm
        return None

    def _determine_measurement_points(
        self, energy_streams: List[str], boundary_type: str,
    ) -> List[Dict[str, str]]:
        """Determine measurement points for a boundary."""
        points = []
        for stream in energy_streams:
            spec = METERING_REQUIREMENTS.get(stream)
            if spec:
                points.append({
                    "stream": stream,
                    "meter_type": spec["meter_type"],
                    "location": boundary_type,
                    "accuracy_pct": str(spec["typical_accuracy_pct"]),
                })
        return points

    def _estimate_meter_cost(self, stream: str) -> float:
        """Estimate meter installation cost by energy stream."""
        cost_map = {
            "electricity": 2500.0,
            "gas": 3000.0,
            "steam": 5000.0,
            "chilled_water": 4000.0,
            "temperature": 500.0,
        }
        return cost_map.get(stream, 2000.0)

    def _estimate_annual_maintenance(self, stream: str) -> float:
        """Estimate annual maintenance cost by energy stream."""
        maint_map = {
            "electricity": 300.0,
            "gas": 400.0,
            "steam": 600.0,
            "chilled_water": 500.0,
            "temperature": 100.0,
        }
        return maint_map.get(stream, 250.0)

    def _build_calibration_schedule(
        self, meters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build calibration schedule for all meters."""
        schedule = []
        for meter in meters:
            interval = meter.get("calibration_interval_months", 12)
            schedule.append({
                "meter_id": meter["meter_id"],
                "energy_stream": meter["energy_stream"],
                "interval_months": interval,
                "next_calibration": "baseline_period_start",
            })
        return schedule

    # -------------------------------------------------------------------------
    # Provenance Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: MVPlanResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
