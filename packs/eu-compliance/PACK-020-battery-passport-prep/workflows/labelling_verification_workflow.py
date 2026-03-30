# -*- coding: utf-8 -*-
"""
Labelling Verification Workflow
=====================================

4-phase workflow for verifying battery labelling compliance per EU Battery
Regulation 2023/1542, Articles 13-14 and Annex VI. Implements requirement
mapping, label review, compliance checking, and corrective action planning.

Phases:
    1. RequirementMapping    -- Identify applicable labelling requirements
    2. LabelReview           -- Review label content against requirements
    3. ComplianceCheck       -- Check conformity of each label element
    4. CorrectiveActions     -- Plan corrective actions for non-conformities

Regulatory references:
    - EU Regulation 2023/1542 Art. 13 (labelling of batteries)
    - EU Regulation 2023/1542 Art. 14 (marking requirements)
    - EU Regulation 2023/1542 Annex VI (labelling requirements)
    - EN IEC 62902 (secondary cell and battery marking symbols)
    - ISO 7000 / IEC 60417 (graphical symbols)

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class WorkflowPhase(str, Enum):
    """Phases of the labelling verification workflow."""
    REQUIREMENT_MAPPING = "requirement_mapping"
    LABEL_REVIEW = "label_review"
    COMPLIANCE_CHECK = "compliance_check"
    CORRECTIVE_ACTIONS = "corrective_actions"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class LabelElementType(str, Enum):
    """Types of label elements per Annex VI."""
    CROSSED_OUT_WHEELIE_BIN = "crossed_out_wheelie_bin"
    CHEMICAL_SYMBOLS = "chemical_symbols"
    CAPACITY_MARKING = "capacity_marking"
    QR_CODE = "qr_code"
    CE_MARKING = "ce_marking"
    MANUFACTURER_INFO = "manufacturer_info"
    MANUFACTURING_DATE = "manufacturing_date"
    BATTERY_TYPE = "battery_type"
    HAZARD_SYMBOLS = "hazard_symbols"
    SEPARATE_COLLECTION = "separate_collection"
    CARBON_FOOTPRINT_CLASS = "carbon_footprint_class"
    RECYCLED_CONTENT_INFO = "recycled_content_info"
    MATERIAL_COMPOSITION = "material_composition"
    WEIGHT = "weight"
    VOLTAGE_CAPACITY = "voltage_capacity"

class ConformityStatus(str, Enum):
    """Conformity status for a label element."""
    CONFORMANT = "conformant"
    NON_CONFORMANT = "non_conformant"
    PARTIALLY_CONFORMANT = "partially_conformant"
    NOT_APPLICABLE = "not_applicable"
    NOT_REVIEWED = "not_reviewed"

class Severity(str, Enum):
    """Non-conformity severity level."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"

# =============================================================================
# LABEL REQUIREMENTS BY BATTERY CATEGORY (Art. 13/14, Annex VI)
# =============================================================================

LABEL_REQUIREMENTS: Dict[str, List[str]] = {
    "ev_battery": [
        LabelElementType.CROSSED_OUT_WHEELIE_BIN.value,
        LabelElementType.CHEMICAL_SYMBOLS.value,
        LabelElementType.CAPACITY_MARKING.value,
        LabelElementType.QR_CODE.value,
        LabelElementType.CE_MARKING.value,
        LabelElementType.MANUFACTURER_INFO.value,
        LabelElementType.MANUFACTURING_DATE.value,
        LabelElementType.BATTERY_TYPE.value,
        LabelElementType.HAZARD_SYMBOLS.value,
        LabelElementType.SEPARATE_COLLECTION.value,
        LabelElementType.CARBON_FOOTPRINT_CLASS.value,
        LabelElementType.WEIGHT.value,
        LabelElementType.VOLTAGE_CAPACITY.value,
    ],
    "industrial_battery": [
        LabelElementType.CROSSED_OUT_WHEELIE_BIN.value,
        LabelElementType.CHEMICAL_SYMBOLS.value,
        LabelElementType.CAPACITY_MARKING.value,
        LabelElementType.QR_CODE.value,
        LabelElementType.CE_MARKING.value,
        LabelElementType.MANUFACTURER_INFO.value,
        LabelElementType.MANUFACTURING_DATE.value,
        LabelElementType.BATTERY_TYPE.value,
        LabelElementType.HAZARD_SYMBOLS.value,
        LabelElementType.SEPARATE_COLLECTION.value,
        LabelElementType.CARBON_FOOTPRINT_CLASS.value,
        LabelElementType.WEIGHT.value,
        LabelElementType.VOLTAGE_CAPACITY.value,
    ],
    "lmt_battery": [
        LabelElementType.CROSSED_OUT_WHEELIE_BIN.value,
        LabelElementType.CHEMICAL_SYMBOLS.value,
        LabelElementType.CAPACITY_MARKING.value,
        LabelElementType.QR_CODE.value,
        LabelElementType.CE_MARKING.value,
        LabelElementType.MANUFACTURER_INFO.value,
        LabelElementType.MANUFACTURING_DATE.value,
        LabelElementType.BATTERY_TYPE.value,
        LabelElementType.SEPARATE_COLLECTION.value,
        LabelElementType.WEIGHT.value,
        LabelElementType.VOLTAGE_CAPACITY.value,
    ],
    "portable_battery": [
        LabelElementType.CROSSED_OUT_WHEELIE_BIN.value,
        LabelElementType.CHEMICAL_SYMBOLS.value,
        LabelElementType.CAPACITY_MARKING.value,
        LabelElementType.CE_MARKING.value,
        LabelElementType.MANUFACTURER_INFO.value,
        LabelElementType.SEPARATE_COLLECTION.value,
        LabelElementType.VOLTAGE_CAPACITY.value,
    ],
    "sli_battery": [
        LabelElementType.CROSSED_OUT_WHEELIE_BIN.value,
        LabelElementType.CHEMICAL_SYMBOLS.value,
        LabelElementType.CAPACITY_MARKING.value,
        LabelElementType.CE_MARKING.value,
        LabelElementType.MANUFACTURER_INFO.value,
        LabelElementType.SEPARATE_COLLECTION.value,
        LabelElementType.WEIGHT.value,
        LabelElementType.VOLTAGE_CAPACITY.value,
    ],
}

# Minimum size requirements for label elements (mm)
ELEMENT_SIZE_REQUIREMENTS: Dict[str, Dict[str, float]] = {
    LabelElementType.CROSSED_OUT_WHEELIE_BIN.value: {"min_height_mm": 3.0, "min_width_mm": 3.0},
    LabelElementType.CE_MARKING.value: {"min_height_mm": 5.0, "min_width_mm": 5.0},
    LabelElementType.QR_CODE.value: {"min_height_mm": 15.0, "min_width_mm": 15.0},
    LabelElementType.CHEMICAL_SYMBOLS.value: {"min_height_mm": 2.0, "min_width_mm": 2.0},
    LabelElementType.HAZARD_SYMBOLS.value: {"min_height_mm": 10.0, "min_width_mm": 10.0},
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class LabelElement(BaseModel):
    """Individual label element submitted for review."""
    element_id: str = Field(default_factory=lambda: f"le-{_new_uuid()[:8]}")
    element_type: str = Field(..., description="Label element type")
    present: bool = Field(default=False, description="Element present on label")
    content: str = Field(default="", description="Content or description")
    height_mm: float = Field(default=0.0, ge=0.0, description="Height in mm")
    width_mm: float = Field(default=0.0, ge=0.0, description="Width in mm")
    legible: bool = Field(default=True, description="Content is legible")
    indelible: bool = Field(default=True, description="Marking is indelible")
    language_codes: List[str] = Field(
        default_factory=list, description="Languages present"
    )
    position: str = Field(default="", description="Position on battery")
    notes: str = Field(default="")

class ConformityResult(BaseModel):
    """Conformity check result for a label element."""
    element_type: str = Field(..., description="Label element type")
    status: ConformityStatus = Field(default=ConformityStatus.NOT_REVIEWED)
    is_required: bool = Field(default=True)
    checks_passed: int = Field(default=0, ge=0)
    checks_total: int = Field(default=0, ge=0)
    issues: List[str] = Field(default_factory=list)
    severity: Optional[Severity] = Field(default=None)

class CorrectiveAction(BaseModel):
    """Corrective action for a non-conformity."""
    action_id: str = Field(default_factory=lambda: f"ca-{_new_uuid()[:8]}")
    element_type: str = Field(default="")
    issue_description: str = Field(default="")
    corrective_measure: str = Field(default="")
    severity: Severity = Field(default=Severity.MINOR)
    deadline: str = Field(default="")
    responsible_party: str = Field(default="")

class LabellingVerificationInput(BaseModel):
    """Input data model for LabellingVerificationWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    battery_model: str = Field(default="", description="Battery model identifier")
    battery_category: str = Field(default="ev_battery")
    label_elements: List[LabelElement] = Field(default_factory=list)
    market_countries: List[str] = Field(
        default_factory=list,
        description="Target market country codes for language requirements"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class LabellingVerificationResult(BaseModel):
    """Complete result from labelling verification workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="labelling_verification")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    battery_id: str = Field(default="")
    conformity_results: List[ConformityResult] = Field(default_factory=list)
    corrective_actions: List[CorrectiveAction] = Field(default_factory=list)
    elements_conformant: int = Field(default=0, ge=0)
    elements_non_conformant: int = Field(default=0, ge=0)
    overall_conformant: bool = Field(default=False)
    label_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class LabellingVerificationWorkflow:
    """
    4-phase labelling verification workflow per EU Battery Regulation.

    Implements label compliance checking following EU Regulation 2023/1542
    Art. 13-14 and Annex VI. Maps applicable requirements by battery
    category, reviews submitted label elements, checks conformity of each
    element, and generates corrective action plans for non-conformities.

    Zero-hallucination: all conformity checks use deterministic presence/size
    comparisons against documented requirements. No LLM in conformity paths.

    Example:
        >>> wf = LabellingVerificationWorkflow()
        >>> inp = LabellingVerificationInput(label_elements=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.overall_conformant in (True, False)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize LabellingVerificationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._elements: List[LabelElement] = []
        self._conformity: List[ConformityResult] = []
        self._corrective: List[CorrectiveAction] = []
        self._required_types: List[str] = []
        self._overall_conformant: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.REQUIREMENT_MAPPING.value, "description": "Identify applicable labelling requirements"},
            {"name": WorkflowPhase.LABEL_REVIEW.value, "description": "Review label content against requirements"},
            {"name": WorkflowPhase.COMPLIANCE_CHECK.value, "description": "Check conformity of each label element"},
            {"name": WorkflowPhase.CORRECTIVE_ACTIONS.value, "description": "Plan corrective actions for non-conformities"},
        ]

    def validate_inputs(self, input_data: LabellingVerificationInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.label_elements:
            issues.append("No label elements provided for review")
        if input_data.battery_category not in LABEL_REQUIREMENTS:
            issues.append(f"Unknown battery category: {input_data.battery_category}")
        return issues

    async def execute(
        self,
        input_data: Optional[LabellingVerificationInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> LabellingVerificationResult:
        """
        Execute the 4-phase labelling verification workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            LabellingVerificationResult with conformity checks and corrective actions.
        """
        if input_data is None:
            input_data = LabellingVerificationInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting labelling verification workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_requirement_mapping(input_data))
            phases_done += 1
            phase_results.append(await self._phase_label_review(input_data))
            phases_done += 1
            phase_results.append(await self._phase_compliance_check(input_data))
            phases_done += 1
            phase_results.append(await self._phase_corrective_actions(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Labelling verification workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        conformant_count = sum(
            1 for c in self._conformity if c.status == ConformityStatus.CONFORMANT
        )
        non_conformant_count = sum(
            1 for c in self._conformity
            if c.status in (ConformityStatus.NON_CONFORMANT, ConformityStatus.PARTIALLY_CONFORMANT)
        )
        completeness = round(
            (conformant_count / len(self._conformity) * 100)
            if self._conformity else 0.0, 1
        )

        result = LabellingVerificationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            battery_id=input_data.battery_id,
            conformity_results=self._conformity,
            corrective_actions=self._corrective,
            elements_conformant=conformant_count,
            elements_non_conformant=non_conformant_count,
            overall_conformant=self._overall_conformant,
            label_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
            executed_at=utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Labelling verification %s completed in %.2fs: %d conformant, %d non-conformant",
            self.workflow_id, elapsed, conformant_count, non_conformant_count,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Requirement Mapping
    # -------------------------------------------------------------------------

    async def _phase_requirement_mapping(
        self, input_data: LabellingVerificationInput,
    ) -> PhaseResult:
        """Identify applicable labelling requirements for battery category."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._required_types = LABEL_REQUIREMENTS.get(
            input_data.battery_category, []
        )

        if not self._required_types:
            self._required_types = LABEL_REQUIREMENTS.get("ev_battery", [])
            warnings.append(
                f"Unknown battery category '{input_data.battery_category}'; "
                f"defaulting to ev_battery requirements"
            )

        outputs["battery_category"] = input_data.battery_category
        outputs["required_elements"] = self._required_types
        outputs["required_element_count"] = len(self._required_types)
        outputs["regulation_reference"] = "EU Regulation 2023/1542 Art. 13-14, Annex VI"

        # Language requirements
        outputs["target_markets"] = input_data.market_countries
        if input_data.market_countries:
            outputs["language_note"] = (
                "Labels must include languages of target market Member States"
            )
        else:
            warnings.append(
                "No target market countries specified; language compliance cannot be verified"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 RequirementMapping: %d required elements for %s",
            len(self._required_types), input_data.battery_category,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REQUIREMENT_MAPPING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Label Review
    # -------------------------------------------------------------------------

    async def _phase_label_review(
        self, input_data: LabellingVerificationInput,
    ) -> PhaseResult:
        """Review submitted label elements against requirements."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._elements = list(input_data.label_elements)
        submitted_types = set(e.element_type for e in self._elements)
        required_set = set(self._required_types)

        present_and_required = submitted_types & required_set
        missing_required = required_set - submitted_types
        extra_submitted = submitted_types - required_set

        outputs["elements_submitted"] = len(self._elements)
        outputs["elements_present_count"] = sum(1 for e in self._elements if e.present)
        outputs["required_elements_covered"] = len(present_and_required)
        outputs["missing_required_elements"] = sorted(missing_required)
        outputs["extra_elements_submitted"] = sorted(extra_submitted)

        if missing_required:
            warnings.append(
                f"Missing required label elements: {', '.join(sorted(missing_required))}"
            )

        # Check legibility
        illegible = [e for e in self._elements if not e.legible and e.present]
        if illegible:
            warnings.append(
                f"{len(illegible)} elements flagged as not legible"
            )

        # Check indelibility
        non_indelible = [e for e in self._elements if not e.indelible and e.present]
        if non_indelible:
            warnings.append(
                f"{len(non_indelible)} elements flagged as not indelible"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 LabelReview: %d submitted, %d required covered, %d missing",
            len(self._elements), len(present_and_required), len(missing_required),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.LABEL_REVIEW.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Compliance Check
    # -------------------------------------------------------------------------

    async def _phase_compliance_check(
        self, input_data: LabellingVerificationInput,
    ) -> PhaseResult:
        """Check conformity of each label element."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._conformity = []

        element_map: Dict[str, LabelElement] = {
            e.element_type: e for e in self._elements
        }

        for req_type in self._required_types:
            element = element_map.get(req_type)
            conformity = self._check_element_conformity(req_type, element)
            self._conformity.append(conformity)

        conformant = sum(1 for c in self._conformity if c.status == ConformityStatus.CONFORMANT)
        non_conformant = sum(
            1 for c in self._conformity
            if c.status == ConformityStatus.NON_CONFORMANT
        )
        partial = sum(
            1 for c in self._conformity
            if c.status == ConformityStatus.PARTIALLY_CONFORMANT
        )

        self._overall_conformant = non_conformant == 0 and partial == 0

        outputs["conformant_count"] = conformant
        outputs["non_conformant_count"] = non_conformant
        outputs["partially_conformant_count"] = partial
        outputs["overall_conformant"] = self._overall_conformant
        outputs["conformity_rate_pct"] = round(
            (conformant / len(self._conformity) * 100)
            if self._conformity else 0.0, 1
        )

        if non_conformant > 0:
            nc_types = [
                c.element_type for c in self._conformity
                if c.status == ConformityStatus.NON_CONFORMANT
            ]
            warnings.append(
                f"Non-conformant elements: {', '.join(nc_types)}"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ComplianceCheck: %d conformant, %d non-conformant, overall=%s",
            conformant, non_conformant, self._overall_conformant,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.COMPLIANCE_CHECK.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _check_element_conformity(
        self, element_type: str, element: Optional[LabelElement],
    ) -> ConformityResult:
        """Deterministic conformity check for a label element."""
        if element is None:
            return ConformityResult(
                element_type=element_type,
                status=ConformityStatus.NON_CONFORMANT,
                is_required=True,
                checks_passed=0,
                checks_total=1,
                issues=[f"{element_type}: required element not submitted"],
                severity=Severity.MAJOR,
            )

        issues: List[str] = []
        checks_total = 0
        checks_passed = 0

        # Check 1: Element present
        checks_total += 1
        if element.present:
            checks_passed += 1
        else:
            issues.append(f"{element_type}: element not present on label")

        # Check 2: Legibility
        checks_total += 1
        if element.legible:
            checks_passed += 1
        else:
            issues.append(f"{element_type}: not legible")

        # Check 3: Indelibility
        checks_total += 1
        if element.indelible:
            checks_passed += 1
        else:
            issues.append(f"{element_type}: not indelible")

        # Check 4: Size requirements
        size_req = ELEMENT_SIZE_REQUIREMENTS.get(element_type)
        if size_req:
            checks_total += 1
            min_h = size_req.get("min_height_mm", 0.0)
            min_w = size_req.get("min_width_mm", 0.0)
            if element.height_mm >= min_h and element.width_mm >= min_w:
                checks_passed += 1
            else:
                issues.append(
                    f"{element_type}: size {element.height_mm}x{element.width_mm}mm "
                    f"below minimum {min_h}x{min_w}mm"
                )

        # Check 5: Content not empty (for elements that need content)
        content_required_types = {
            LabelElementType.MANUFACTURER_INFO.value,
            LabelElementType.CAPACITY_MARKING.value,
            LabelElementType.CHEMICAL_SYMBOLS.value,
            LabelElementType.MANUFACTURING_DATE.value,
            LabelElementType.BATTERY_TYPE.value,
            LabelElementType.VOLTAGE_CAPACITY.value,
            LabelElementType.WEIGHT.value,
        }
        if element_type in content_required_types:
            checks_total += 1
            if element.content:
                checks_passed += 1
            else:
                issues.append(f"{element_type}: content is empty")

        # Determine conformity
        if checks_passed == checks_total:
            status = ConformityStatus.CONFORMANT
            severity = None
        elif checks_passed >= checks_total * 0.7:
            status = ConformityStatus.PARTIALLY_CONFORMANT
            severity = Severity.MINOR
        else:
            status = ConformityStatus.NON_CONFORMANT
            severity = Severity.MAJOR if element.present else Severity.CRITICAL

        return ConformityResult(
            element_type=element_type,
            status=status,
            is_required=True,
            checks_passed=checks_passed,
            checks_total=checks_total,
            issues=issues,
            severity=severity,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Corrective Actions
    # -------------------------------------------------------------------------

    async def _phase_corrective_actions(
        self, input_data: LabellingVerificationInput,
    ) -> PhaseResult:
        """Plan corrective actions for non-conformities."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._corrective = []

        for conf in self._conformity:
            if conf.status in (
                ConformityStatus.NON_CONFORMANT,
                ConformityStatus.PARTIALLY_CONFORMANT,
            ):
                for issue in conf.issues:
                    measure = self._determine_corrective_measure(
                        conf.element_type, issue
                    )
                    self._corrective.append(CorrectiveAction(
                        element_type=conf.element_type,
                        issue_description=issue,
                        corrective_measure=measure,
                        severity=conf.severity or Severity.MINOR,
                        deadline=f"{input_data.reporting_year}-12-31",
                        responsible_party="Product Compliance Team",
                    ))

        severity_counts: Dict[str, int] = {}
        for ca in self._corrective:
            severity_counts[ca.severity.value] = (
                severity_counts.get(ca.severity.value, 0) + 1
            )

        outputs["corrective_actions_created"] = len(self._corrective)
        outputs["severity_distribution"] = severity_counts
        outputs["elements_requiring_action"] = len(set(
            ca.element_type for ca in self._corrective
        ))

        if not self._corrective:
            outputs["note"] = "No corrective actions required; all elements conformant"
        else:
            critical = severity_counts.get("critical", 0)
            if critical > 0:
                warnings.append(
                    f"{critical} critical corrective actions required"
                )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 CorrectiveActions: %d actions created",
            len(self._corrective),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.CORRECTIVE_ACTIONS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _determine_corrective_measure(
        self, element_type: str, issue: str,
    ) -> str:
        """Determine corrective measure based on element type and issue."""
        if "not present" in issue or "not submitted" in issue:
            return f"Add {element_type} to battery label per Annex VI requirements."
        if "not legible" in issue:
            return f"Increase font size or contrast of {element_type} marking."
        if "not indelible" in issue:
            return f"Apply {element_type} using indelible marking method (laser engraving or permanent adhesive)."
        if "size" in issue and "below minimum" in issue:
            return f"Increase {element_type} dimensions to meet minimum size requirements."
        if "content is empty" in issue:
            return f"Populate {element_type} field with required content per Annex VI."
        return f"Review and correct {element_type} to meet Annex VI requirements."

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: LabellingVerificationResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
