# -*- coding: utf-8 -*-
"""
Control Testing Workflow
====================================

5-phase workflow for GHG assurance internal control testing covering control
identification, design assessment, sample selection, test execution, and
deficiency reporting within PACK-048 GHG Assurance Prep Pack.

Phases:
    1. ControlIdentification       -- Identify applicable controls from a
                                      25-control standard register, mapping
                                      each to the organisation's GHG reporting
                                      processes, data flows, and systems.
    2. DesignAssessment            -- Assess the design effectiveness of each
                                      identified control, evaluating whether
                                      the control is properly designed to
                                      prevent or detect material misstatement
                                      in the GHG statement.
    3. SampleSelection             -- Select test samples per control based
                                      on population size, risk assessment,
                                      and statistical sampling parameters
                                      to achieve the required confidence level.
    4. TestExecution               -- Execute control tests and record results,
                                      classifying each test as PASS / FAIL /
                                      EXCEPTION, with detailed test evidence
                                      and observations.
    5. DeficiencyReporting         -- Classify and report deficiencies as
                                      material weakness, significant deficiency,
                                      or control deficiency, with remediation
                                      recommendations and severity scoring.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ISAE 3410 (2012) - Internal control evaluation for GHG assurance
    ISO 14064-3:2019 - Verification procedures and controls
    COSO Internal Control Framework (2013) - Control assessment
    ISA 315 (Revised 2019) - Understanding the entity and controls
    ESRS E1 (2024) - Controls over sustainability reporting
    SOX Section 404 (adapted) - Internal control testing methodology

Schedule: Annually prior to assurance engagement
Estimated duration: 3-5 weeks depending on control population

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas.enums import RiskLevel

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

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

class ControlTestingPhase(str, Enum):
    """Control testing workflow phases."""

    CONTROL_IDENTIFICATION = "control_identification"
    DESIGN_ASSESSMENT = "design_assessment"
    SAMPLE_SELECTION = "sample_selection"
    TEST_EXECUTION = "test_execution"
    DEFICIENCY_REPORTING = "deficiency_reporting"

class ControlCategory(str, Enum):
    """Control category classification."""

    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    CALCULATION = "calculation"
    REVIEW_APPROVAL = "review_approval"
    IT_GENERAL = "it_general"
    CHANGE_MANAGEMENT = "change_management"
    REPORTING = "reporting"

class ControlType(str, Enum):
    """Control type classification."""

    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"

class ControlFrequency(str, Enum):
    """Control execution frequency."""

    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class DesignEffectiveness(str, Enum):
    """Design effectiveness assessment outcome."""

    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    INEFFECTIVE = "ineffective"
    NOT_ASSESSED = "not_assessed"

class TestResult(str, Enum):
    """Control test execution result."""

    PASS = "pass"
    FAIL = "fail"
    EXCEPTION = "exception"
    NOT_TESTED = "not_tested"

class DeficiencyClassification(str, Enum):
    """Deficiency severity classification."""

    MATERIAL_WEAKNESS = "material_weakness"
    SIGNIFICANT_DEFICIENCY = "significant_deficiency"
    CONTROL_DEFICIENCY = "control_deficiency"
    OBSERVATION = "observation"

# =============================================================================
# STANDARD CONTROLS REGISTER (Zero-Hallucination Reference Data)
# =============================================================================

STANDARD_CONTROLS_REGISTER: List[Dict[str, Any]] = [
    # DATA_COLLECTION controls (1-5)
    {"id": "CTL-001", "category": "data_collection", "type": "preventive",
     "name": "Activity data input validation",
     "description": "Automated validation of activity data at point of entry",
     "risk_area": "Data accuracy", "frequency": "continuous"},
    {"id": "CTL-002", "category": "data_collection", "type": "detective",
     "name": "Completeness check on source data",
     "description": "Reconciliation of source data against expected population",
     "risk_area": "Data completeness", "frequency": "monthly"},
    {"id": "CTL-003", "category": "data_collection", "type": "preventive",
     "name": "Meter reading verification",
     "description": "Verification of meter readings against utility invoices",
     "risk_area": "Measurement accuracy", "frequency": "monthly"},
    {"id": "CTL-004", "category": "data_collection", "type": "detective",
     "name": "Missing data detection",
     "description": "Automated detection of missing data points in time series",
     "risk_area": "Data completeness", "frequency": "weekly"},
    {"id": "CTL-005", "category": "data_collection", "type": "corrective",
     "name": "Data correction authorisation",
     "description": "Formal authorisation required for data corrections",
     "risk_area": "Data integrity", "frequency": "continuous"},
    # DATA_PROCESSING controls (6-9)
    {"id": "CTL-006", "category": "data_processing", "type": "preventive",
     "name": "Unit conversion validation",
     "description": "Automated validation of unit conversions in processing",
     "risk_area": "Calculation accuracy", "frequency": "continuous"},
    {"id": "CTL-007", "category": "data_processing", "type": "detective",
     "name": "Duplicate entry detection",
     "description": "Automated detection of duplicate data entries",
     "risk_area": "Data accuracy", "frequency": "daily"},
    {"id": "CTL-008", "category": "data_processing", "type": "detective",
     "name": "Outlier detection on processed data",
     "description": "Statistical outlier detection on processed emission data",
     "risk_area": "Data accuracy", "frequency": "monthly"},
    {"id": "CTL-009", "category": "data_processing", "type": "preventive",
     "name": "Data transformation audit trail",
     "description": "Complete audit trail of all data transformations",
     "risk_area": "Auditability", "frequency": "continuous"},
    # CALCULATION controls (10-14)
    {"id": "CTL-010", "category": "calculation", "type": "preventive",
     "name": "Emission factor version control",
     "description": "Version-controlled emission factors with change tracking",
     "risk_area": "Calculation accuracy", "frequency": "annually"},
    {"id": "CTL-011", "category": "calculation", "type": "detective",
     "name": "Calculation re-performance",
     "description": "Independent re-performance of emission calculations",
     "risk_area": "Calculation accuracy", "frequency": "quarterly"},
    {"id": "CTL-012", "category": "calculation", "type": "detective",
     "name": "Year-over-year variance analysis",
     "description": "Analysis of variances exceeding threshold vs prior period",
     "risk_area": "Reasonableness", "frequency": "quarterly"},
    {"id": "CTL-013", "category": "calculation", "type": "preventive",
     "name": "GWP value validation",
     "description": "Validation of GWP values against IPCC AR vintage",
     "risk_area": "Calculation accuracy", "frequency": "annually"},
    {"id": "CTL-014", "category": "calculation", "type": "detective",
     "name": "Scope boundary completeness check",
     "description": "Check that all organisational units in boundary are included",
     "risk_area": "Completeness", "frequency": "annually"},
    # REVIEW_APPROVAL controls (15-18)
    {"id": "CTL-015", "category": "review_approval", "type": "detective",
     "name": "Management review of GHG statement",
     "description": "Formal management review and sign-off on GHG statement",
     "risk_area": "Statement accuracy", "frequency": "annually"},
    {"id": "CTL-016", "category": "review_approval", "type": "detective",
     "name": "Subject matter expert review",
     "description": "Technical review by qualified GHG practitioner",
     "risk_area": "Technical accuracy", "frequency": "annually"},
    {"id": "CTL-017", "category": "review_approval", "type": "preventive",
     "name": "Segregation of duties",
     "description": "Separation of data collection, processing, and approval roles",
     "risk_area": "Fraud prevention", "frequency": "continuous"},
    {"id": "CTL-018", "category": "review_approval", "type": "detective",
     "name": "Disclosure checklist review",
     "description": "Review of disclosures against standard checklist",
     "risk_area": "Completeness", "frequency": "annually"},
    # IT_GENERAL controls (19-21)
    {"id": "CTL-019", "category": "it_general", "type": "preventive",
     "name": "Access control on GHG systems",
     "description": "Role-based access control on GHG reporting systems",
     "risk_area": "Data integrity", "frequency": "continuous"},
    {"id": "CTL-020", "category": "it_general", "type": "preventive",
     "name": "Backup and recovery",
     "description": "Regular backup and tested recovery of GHG data",
     "risk_area": "Data availability", "frequency": "daily"},
    {"id": "CTL-021", "category": "it_general", "type": "detective",
     "name": "System change log review",
     "description": "Review of system changes affecting GHG calculations",
     "risk_area": "Calculation integrity", "frequency": "monthly"},
    # CHANGE_MANAGEMENT controls (22-23)
    {"id": "CTL-022", "category": "change_management", "type": "preventive",
     "name": "Methodology change approval",
     "description": "Formal approval process for GHG methodology changes",
     "risk_area": "Consistency", "frequency": "annually"},
    {"id": "CTL-023", "category": "change_management", "type": "detective",
     "name": "Base year recalculation trigger review",
     "description": "Review of events triggering base year recalculation",
     "risk_area": "Comparability", "frequency": "annually"},
    # REPORTING controls (24-25)
    {"id": "CTL-024", "category": "reporting", "type": "detective",
     "name": "Disclosure completeness check",
     "description": "Systematic check of all required disclosure items",
     "risk_area": "Completeness", "frequency": "annually"},
    {"id": "CTL-025", "category": "reporting", "type": "detective",
     "name": "Cross-reference to source data",
     "description": "Cross-reference of reported figures to underlying data",
     "risk_area": "Accuracy", "frequency": "annually"},
]

SAMPLE_SIZE_TABLE: Dict[str, Dict[str, int]] = {
    "high": {"small": 15, "medium": 25, "large": 40},
    "medium": {"small": 10, "medium": 15, "large": 25},
    "low": {"small": 5, "medium": 10, "large": 15},
}

POPULATION_SIZE_BANDS: Dict[str, Tuple[int, int]] = {
    "small": (1, 50),
    "medium": (51, 250),
    "large": (251, 99999),
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class ControlRecord(BaseModel):
    """Record of an identified control."""

    control_id: str = Field(...)
    category: ControlCategory = Field(...)
    control_type: ControlType = Field(...)
    name: str = Field(default="")
    description: str = Field(default="")
    risk_area: str = Field(default="")
    frequency: ControlFrequency = Field(default=ControlFrequency.ANNUALLY)
    applicable: bool = Field(default=True)
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    design_effectiveness: DesignEffectiveness = Field(
        default=DesignEffectiveness.NOT_ASSESSED,
    )
    design_notes: str = Field(default="")
    sample_size: int = Field(default=0, ge=0)
    population_size: int = Field(default=0, ge=0)
    test_result: TestResult = Field(default=TestResult.NOT_TESTED)
    exceptions_count: int = Field(default=0, ge=0)
    test_notes: str = Field(default="")
    provenance_hash: str = Field(default="")

class DeficiencyRecord(BaseModel):
    """Record of an identified control deficiency."""

    deficiency_id: str = Field(default_factory=lambda: f"def-{_new_uuid()[:8]}")
    control_id: str = Field(default="")
    control_name: str = Field(default="")
    classification: DeficiencyClassification = Field(
        default=DeficiencyClassification.OBSERVATION,
    )
    description: str = Field(default="")
    root_cause: str = Field(default="")
    impact_description: str = Field(default="")
    severity_score: float = Field(default=0.0, ge=0.0, le=100.0)
    remediation_action: str = Field(default="")
    remediation_owner: str = Field(default="")
    target_date: str = Field(default="")
    provenance_hash: str = Field(default="")

class ControlTestSummary(BaseModel):
    """Summary of control testing results."""

    total_controls: int = Field(default=0, ge=0)
    applicable_controls: int = Field(default=0, ge=0)
    design_effective: int = Field(default=0, ge=0)
    design_partially_effective: int = Field(default=0, ge=0)
    design_ineffective: int = Field(default=0, ge=0)
    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    tests_exception: int = Field(default=0, ge=0)
    total_deficiencies: int = Field(default=0, ge=0)
    material_weaknesses: int = Field(default=0, ge=0)
    significant_deficiencies: int = Field(default=0, ge=0)
    control_effectiveness_pct: str = Field(default="0.00")
    provenance_hash: str = Field(default="")

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class ControlTestingInput(BaseModel):
    """Input data model for ControlTestingWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    existing_controls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pre-existing control implementations with status and evidence",
    )
    control_test_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pre-existing test results for controls",
    )
    population_sizes: Dict[str, int] = Field(
        default_factory=dict,
        description="Population sizes per control ID for sampling",
    )
    risk_assessment: Dict[str, str] = Field(
        default_factory=dict,
        description="Risk level per control ID (high/medium/low)",
    )
    reporting_period: str = Field(default="2025")
    assurance_level: str = Field(default="limited")
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class ControlTestingResult(BaseModel):
    """Complete result from control testing workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="control_testing")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    controls: List[ControlRecord] = Field(default_factory=list)
    deficiencies: List[DeficiencyRecord] = Field(default_factory=list)
    summary: Optional[ControlTestSummary] = Field(default=None)
    control_effectiveness_pct: str = Field(default="0.00")
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ControlTestingWorkflow:
    """
    5-phase workflow for GHG assurance internal control testing.

    Identifies applicable controls from the standard register, assesses
    design effectiveness, selects test samples, executes tests, and
    reports deficiencies with remediation recommendations.

    Zero-hallucination: all sampling uses deterministic tables; design
    assessment uses structured criteria; severity scoring uses fixed
    formulas; no LLM calls in numeric paths; SHA-256 provenance on
    every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _controls: Identified and tested controls.
        _deficiencies: Identified deficiencies.
        _summary: Control testing summary.

    Example:
        >>> wf = ControlTestingWorkflow()
        >>> inp = ControlTestingInput(organization_id="org-001")
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[ControlTestingPhase] = [
        ControlTestingPhase.CONTROL_IDENTIFICATION,
        ControlTestingPhase.DESIGN_ASSESSMENT,
        ControlTestingPhase.SAMPLE_SELECTION,
        ControlTestingPhase.TEST_EXECUTION,
        ControlTestingPhase.DEFICIENCY_REPORTING,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ControlTestingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._controls: List[ControlRecord] = []
        self._deficiencies: List[DeficiencyRecord] = []
        self._summary: Optional[ControlTestSummary] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: ControlTestingInput,
    ) -> ControlTestingResult:
        """
        Execute the 5-phase control testing workflow.

        Args:
            input_data: Organisation controls, test data, and risk assessment.

        Returns:
            ControlTestingResult with test outcomes and deficiency reports.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting control testing %s org=%s",
            self.workflow_id, input_data.organization_id,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_control_identification,
            self._phase_2_design_assessment,
            self._phase_3_sample_selection,
            self._phase_4_test_execution,
            self._phase_5_deficiency_reporting,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Control testing failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        effectiveness = "0.00"
        if self._summary:
            effectiveness = self._summary.control_effectiveness_pct

        result = ControlTestingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            controls=self._controls,
            deficiencies=self._deficiencies,
            summary=self._summary,
            control_effectiveness_pct=effectiveness,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Control testing %s completed in %.2fs status=%s effectiveness=%s",
            self.workflow_id, elapsed, overall_status.value, effectiveness,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Control Identification
    # -------------------------------------------------------------------------

    async def _phase_1_control_identification(
        self, input_data: ControlTestingInput,
    ) -> PhaseResult:
        """Identify applicable controls from standard register."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build lookup of existing control implementations
        existing_lookup: Dict[str, Dict[str, Any]] = {}
        for ec in input_data.existing_controls:
            existing_lookup[ec.get("control_id", "")] = ec

        self._controls = []
        for ctl_def in STANDARD_CONTROLS_REGISTER:
            ctl_id = ctl_def["id"]
            existing = existing_lookup.get(ctl_id, {})

            try:
                category = ControlCategory(ctl_def["category"])
            except ValueError:
                category = ControlCategory.DATA_COLLECTION

            try:
                ctl_type = ControlType(ctl_def["type"])
            except ValueError:
                ctl_type = ControlType.DETECTIVE

            try:
                frequency = ControlFrequency(ctl_def.get("frequency", "annually"))
            except ValueError:
                frequency = ControlFrequency.ANNUALLY

            risk_str = input_data.risk_assessment.get(ctl_id, "medium")
            try:
                risk_level = RiskLevel(risk_str)
            except ValueError:
                risk_level = RiskLevel.MEDIUM

            applicable = existing.get("applicable", True)

            control = ControlRecord(
                control_id=ctl_id,
                category=category,
                control_type=ctl_type,
                name=ctl_def["name"],
                description=ctl_def["description"],
                risk_area=ctl_def.get("risk_area", ""),
                frequency=frequency,
                applicable=applicable,
                risk_level=risk_level,
            )
            ctl_data = {"id": ctl_id, "applicable": applicable, "risk": risk_str}
            control.provenance_hash = _compute_hash(ctl_data)
            self._controls.append(control)

        applicable_count = sum(1 for c in self._controls if c.applicable)
        outputs["total_controls_in_register"] = len(STANDARD_CONTROLS_REGISTER)
        outputs["controls_identified"] = len(self._controls)
        outputs["applicable_controls"] = applicable_count
        outputs["not_applicable"] = len(self._controls) - applicable_count

        if applicable_count < 15:
            warnings.append(
                f"Only {applicable_count} controls applicable; "
                "consider expanding control environment"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 ControlIdentification: %d controls, %d applicable",
            len(self._controls), applicable_count,
        )
        return PhaseResult(
            phase_name="control_identification", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Design Assessment
    # -------------------------------------------------------------------------

    async def _phase_2_design_assessment(
        self, input_data: ControlTestingInput,
    ) -> PhaseResult:
        """Assess design effectiveness of each identified control."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        existing_lookup: Dict[str, Dict[str, Any]] = {}
        for ec in input_data.existing_controls:
            existing_lookup[ec.get("control_id", "")] = ec

        effective_count = 0
        partial_count = 0
        ineffective_count = 0

        for control in self._controls:
            if not control.applicable:
                control.design_effectiveness = DesignEffectiveness.NOT_ASSESSED
                continue

            existing = existing_lookup.get(control.control_id, {})
            has_documentation = existing.get("documented", False)
            has_owner = existing.get("has_owner", False)
            has_procedure = existing.get("has_procedure", False)
            is_implemented = existing.get("implemented", False)

            # Design effectiveness scoring (deterministic)
            design_score = Decimal("0")
            if has_documentation:
                design_score += Decimal("25")
            if has_owner:
                design_score += Decimal("25")
            if has_procedure:
                design_score += Decimal("25")
            if is_implemented:
                design_score += Decimal("25")

            if design_score >= Decimal("75"):
                control.design_effectiveness = DesignEffectiveness.EFFECTIVE
                effective_count += 1
            elif design_score >= Decimal("50"):
                control.design_effectiveness = DesignEffectiveness.PARTIALLY_EFFECTIVE
                partial_count += 1
            else:
                control.design_effectiveness = DesignEffectiveness.INEFFECTIVE
                ineffective_count += 1

            control.design_notes = (
                f"Score: {design_score}/100. "
                f"Doc: {has_documentation}, Owner: {has_owner}, "
                f"Proc: {has_procedure}, Impl: {is_implemented}"
            )

        outputs["effective"] = effective_count
        outputs["partially_effective"] = partial_count
        outputs["ineffective"] = ineffective_count
        outputs["not_assessed"] = sum(
            1 for c in self._controls
            if c.design_effectiveness == DesignEffectiveness.NOT_ASSESSED
        )

        if ineffective_count > 3:
            warnings.append(
                f"{ineffective_count} controls have ineffective design; "
                "remediation required before operating effectiveness testing"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 DesignAssessment: effective=%d partial=%d ineffective=%d",
            effective_count, partial_count, ineffective_count,
        )
        return PhaseResult(
            phase_name="design_assessment", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Sample Selection
    # -------------------------------------------------------------------------

    async def _phase_3_sample_selection(
        self, input_data: ControlTestingInput,
    ) -> PhaseResult:
        """Select test samples per control."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_samples = 0
        for control in self._controls:
            if not control.applicable:
                continue
            if control.design_effectiveness == DesignEffectiveness.INEFFECTIVE:
                control.sample_size = 0
                continue

            pop_size = input_data.population_sizes.get(control.control_id, 25)
            control.population_size = pop_size

            # Determine population band
            pop_band = "small"
            for band_name, (lower, upper) in POPULATION_SIZE_BANDS.items():
                if lower <= pop_size <= upper:
                    pop_band = band_name
                    break

            risk_key = control.risk_level.value
            sample = SAMPLE_SIZE_TABLE.get(risk_key, {}).get(pop_band, 10)

            # Cap sample at population size
            sample = min(sample, pop_size)

            # Increase sample for reasonable assurance
            if input_data.assurance_level == "reasonable":
                sample = min(int(sample * 1.5), pop_size)

            control.sample_size = sample
            total_samples += sample

        outputs["controls_with_samples"] = sum(
            1 for c in self._controls if c.sample_size > 0
        )
        outputs["total_samples"] = total_samples
        outputs["assurance_level"] = input_data.assurance_level

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 SampleSelection: %d total samples across %d controls",
            total_samples, outputs["controls_with_samples"],
        )
        return PhaseResult(
            phase_name="sample_selection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Test Execution
    # -------------------------------------------------------------------------

    async def _phase_4_test_execution(
        self, input_data: ControlTestingInput,
    ) -> PhaseResult:
        """Execute control tests and record results."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build lookup of test results
        test_lookup: Dict[str, Dict[str, Any]] = {}
        for tr in input_data.control_test_results:
            test_lookup[tr.get("control_id", "")] = tr

        pass_count = 0
        fail_count = 0
        exception_count = 0

        for control in self._controls:
            if not control.applicable or control.sample_size == 0:
                control.test_result = TestResult.NOT_TESTED
                continue

            test_data = test_lookup.get(control.control_id, {})
            result_str = test_data.get("result", "pass")

            try:
                test_result = TestResult(result_str)
            except ValueError:
                test_result = TestResult.PASS

            control.test_result = test_result
            control.exceptions_count = int(test_data.get("exceptions", 0))
            control.test_notes = test_data.get("notes", "")

            if test_result == TestResult.PASS:
                pass_count += 1
            elif test_result == TestResult.FAIL:
                fail_count += 1
            elif test_result == TestResult.EXCEPTION:
                exception_count += 1

        outputs["tests_passed"] = pass_count
        outputs["tests_failed"] = fail_count
        outputs["tests_exception"] = exception_count
        outputs["not_tested"] = sum(
            1 for c in self._controls if c.test_result == TestResult.NOT_TESTED
        )

        if fail_count > 0:
            warnings.append(f"{fail_count} control tests failed")

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 TestExecution: pass=%d fail=%d exception=%d",
            pass_count, fail_count, exception_count,
        )
        return PhaseResult(
            phase_name="test_execution", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Deficiency Reporting
    # -------------------------------------------------------------------------

    async def _phase_5_deficiency_reporting(
        self, input_data: ControlTestingInput,
    ) -> PhaseResult:
        """Classify and report control deficiencies."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._deficiencies = []

        for control in self._controls:
            if not control.applicable:
                continue

            has_deficiency = False
            severity = Decimal("0")

            # Design ineffectiveness is a deficiency
            if control.design_effectiveness == DesignEffectiveness.INEFFECTIVE:
                has_deficiency = True
                severity += Decimal("60")
            elif control.design_effectiveness == DesignEffectiveness.PARTIALLY_EFFECTIVE:
                has_deficiency = True
                severity += Decimal("30")

            # Test failure is a deficiency
            if control.test_result == TestResult.FAIL:
                has_deficiency = True
                severity += Decimal("40")
            elif control.test_result == TestResult.EXCEPTION:
                has_deficiency = True
                severity += Decimal("20")

            # Risk level amplifies severity
            if control.risk_level == RiskLevel.HIGH:
                severity = (severity * Decimal("1.3")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            elif control.risk_level == RiskLevel.LOW:
                severity = (severity * Decimal("0.8")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )

            severity = min(severity, Decimal("100"))

            if not has_deficiency:
                continue

            # Classify deficiency
            if severity >= Decimal("80"):
                classification = DeficiencyClassification.MATERIAL_WEAKNESS
            elif severity >= Decimal("50"):
                classification = DeficiencyClassification.SIGNIFICANT_DEFICIENCY
            elif severity >= Decimal("20"):
                classification = DeficiencyClassification.CONTROL_DEFICIENCY
            else:
                classification = DeficiencyClassification.OBSERVATION

            remediation = self._generate_control_remediation(control)

            def_data = {
                "control": control.control_id, "severity": str(severity),
                "class": classification.value,
            }
            deficiency = DeficiencyRecord(
                control_id=control.control_id,
                control_name=control.name,
                classification=classification,
                description=(
                    f"Control {control.control_id} ({control.name}): "
                    f"design={control.design_effectiveness.value}, "
                    f"test={control.test_result.value}"
                ),
                root_cause=f"Design: {control.design_notes}",
                impact_description=f"Risk area: {control.risk_area}",
                severity_score=float(severity),
                remediation_action=remediation,
                remediation_owner="Internal Controls Team",
                target_date="",
                provenance_hash=_compute_hash(def_data),
            )
            self._deficiencies.append(deficiency)

        # Build summary
        applicable = sum(1 for c in self._controls if c.applicable)
        passed = sum(1 for c in self._controls if c.test_result == TestResult.PASS)
        effectiveness = Decimal("0.00")
        if applicable > 0:
            effectiveness = (
                Decimal(str(passed)) / Decimal(str(applicable)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        mw_count = sum(
            1 for d in self._deficiencies
            if d.classification == DeficiencyClassification.MATERIAL_WEAKNESS
        )
        sd_count = sum(
            1 for d in self._deficiencies
            if d.classification == DeficiencyClassification.SIGNIFICANT_DEFICIENCY
        )

        summary_data = {
            "total": len(self._controls), "applicable": applicable,
            "effectiveness": str(effectiveness),
        }
        self._summary = ControlTestSummary(
            total_controls=len(self._controls),
            applicable_controls=applicable,
            design_effective=sum(
                1 for c in self._controls
                if c.design_effectiveness == DesignEffectiveness.EFFECTIVE
            ),
            design_partially_effective=sum(
                1 for c in self._controls
                if c.design_effectiveness == DesignEffectiveness.PARTIALLY_EFFECTIVE
            ),
            design_ineffective=sum(
                1 for c in self._controls
                if c.design_effectiveness == DesignEffectiveness.INEFFECTIVE
            ),
            tests_passed=passed,
            tests_failed=sum(1 for c in self._controls if c.test_result == TestResult.FAIL),
            tests_exception=sum(
                1 for c in self._controls if c.test_result == TestResult.EXCEPTION
            ),
            total_deficiencies=len(self._deficiencies),
            material_weaknesses=mw_count,
            significant_deficiencies=sd_count,
            control_effectiveness_pct=str(effectiveness),
            provenance_hash=_compute_hash(summary_data),
        )

        outputs["total_deficiencies"] = len(self._deficiencies)
        outputs["material_weaknesses"] = mw_count
        outputs["significant_deficiencies"] = sd_count
        outputs["control_effectiveness_pct"] = str(effectiveness)

        if mw_count > 0:
            warnings.append(
                f"{mw_count} material weakness(es) identified; "
                "must be remediated before assurance engagement"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 DeficiencyReporting: %d deficiencies (MW=%d SD=%d) effectiveness=%s%%",
            len(self._deficiencies), mw_count, sd_count, str(effectiveness),
        )
        return PhaseResult(
            phase_name="deficiency_reporting", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: ControlTestingInput,
        phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_control_remediation(self, control: ControlRecord) -> str:
        """Generate remediation action for a control deficiency."""
        actions = []
        if control.design_effectiveness in (
            DesignEffectiveness.INEFFECTIVE,
            DesignEffectiveness.PARTIALLY_EFFECTIVE,
        ):
            actions.append(
                f"Redesign control '{control.name}' to address design gaps. "
                f"Ensure documentation, ownership, and procedures are complete."
            )
        if control.test_result in (TestResult.FAIL, TestResult.EXCEPTION):
            actions.append(
                f"Investigate root cause of test failure for '{control.name}'. "
                f"Implement corrective action and re-test."
            )
        return " ".join(actions) if actions else "No remediation required."

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._controls = []
        self._deficiencies = []
        self._summary = None

    def _compute_provenance(self, result: ControlTestingResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.control_effectiveness_pct}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
