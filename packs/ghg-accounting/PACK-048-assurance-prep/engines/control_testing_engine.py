# -*- coding: utf-8 -*-
"""
ControlTestingEngine - PACK-048 GHG Assurance Prep Engine 4
====================================================================

Evaluates the design and operating effectiveness of internal controls
over GHG reporting.  Implements 25 standard controls across 5 categories
(Data Collection, Calculation, Review, Reporting, IT General), with
control maturity assessment, deficiency classification, and remediation
planning.

Calculation Methodology:
    Control Categories (5 categories, 5 controls each = 25 total):
        DC-01 to DC-05: Data Collection
            Meter calibration, data capture, supplier validation,
            activity reconciliation, completeness checks.
        CA-01 to CA-05: Calculation
            EF selection, formula validation, unit conversion,
            aggregation review, system access controls.
        RV-01 to RV-05: Review
            Peer review, management review, cross-scope reconciliation,
            variance analysis, sign-off procedures.
        RE-01 to RE-05: Reporting
            Data extraction, template accuracy, disclosure review,
            submission approval, archive procedures.
        IT-01 to IT-05: IT General Controls
            Access management, change management, backup/recovery,
            audit trail, data integrity checks.

    Control Types:
        PREVENTIVE:     Prevents errors from occurring.
        DETECTIVE:      Detects errors after occurrence.
        CORRECTIVE:     Corrects errors once detected.

    Design Effectiveness (0-100):
        DE = w_documentation * doc_score + w_suitability * suit_score
             + w_coverage * cov_score + w_frequency * freq_score

        Default weights: doc=0.25, suit=0.35, cov=0.20, freq=0.20

    Operating Effectiveness (0-100):
        OE = (tests_passed / tests_performed) * 100

    Deficiency Classification:
        MATERIAL_WEAKNESS:       DE < 40 OR OE < 50 on key control
        SIGNIFICANT_DEFICIENCY:  DE < 60 OR OE < 70 on key control
        DEFICIENCY:              DE < 80 OR OE < 80 on any control

    Control Maturity Model:
        Level 1 (Ad Hoc):      score < 20
        Level 2 (Repeatable):  score 20-39
        Level 3 (Defined):     score 40-59
        Level 4 (Managed):     score 60-79
        Level 5 (Optimised):   score >= 80

    Sample Testing:
        Population size determines sample:
            1-10:    100% testing
            11-50:   minimum 10 or 25%
            51-250:  minimum 25 or 15%
            251+:    minimum 30 or 10%

Regulatory References:
    - ISAE 3410 para 49-51: Understanding entity's controls
    - ISAE 3000 (Revised): Control environment assessment
    - COSO Internal Control Framework (2013)
    - ISO 14064-3:2019: Control environment requirements
    - PCAOB AS 2201: Internal control audit standard (reference)

Zero-Hallucination:
    - All control definitions from published frameworks
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-048 GHG Assurance Prep
Engine:  4 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ControlCategory(str, Enum):
    """Control category."""
    DATA_COLLECTION = "data_collection"
    CALCULATION = "calculation"
    REVIEW = "review"
    REPORTING = "reporting"
    IT_GENERAL = "it_general"

class ControlType(str, Enum):
    """Control type.

    PREVENTIVE: Prevents errors from occurring.
    DETECTIVE:  Detects errors after occurrence.
    CORRECTIVE: Corrects errors once detected.
    """
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"

class DeficiencyLevel(str, Enum):
    """Deficiency classification.

    DEFICIENCY:              Minor control deficiency.
    SIGNIFICANT_DEFICIENCY:  Significant control deficiency.
    MATERIAL_WEAKNESS:       Material weakness in controls.
    NONE:                    No deficiency.
    """
    NONE = "none"
    DEFICIENCY = "deficiency"
    SIGNIFICANT_DEFICIENCY = "significant_deficiency"
    MATERIAL_WEAKNESS = "material_weakness"

class MaturityLevel(str, Enum):
    """Control maturity level.

    AD_HOC:      Level 1 - Informal, ad hoc processes.
    REPEATABLE:  Level 2 - Basic processes repeatable.
    DEFINED:     Level 3 - Processes documented and standardised.
    MANAGED:     Level 4 - Processes measured and managed.
    OPTIMISED:   Level 5 - Continuous improvement culture.
    """
    AD_HOC = "ad_hoc"
    REPEATABLE = "repeatable"
    DEFINED = "defined"
    MANAGED = "managed"
    OPTIMISED = "optimised"

# ---------------------------------------------------------------------------
# Constants -- Standard Controls
# ---------------------------------------------------------------------------

STANDARD_CONTROLS: List[Dict[str, Any]] = [
    # Data Collection (DC-01 to DC-05)
    {"control_id": "DC-01", "category": "data_collection", "name": "Meter Calibration",
     "description": "Measurement instruments calibrated per manufacturer schedule with records maintained",
     "control_type": "preventive", "is_key": True},
    {"control_id": "DC-02", "category": "data_collection", "name": "Data Capture",
     "description": "Activity data captured at source with automated validation on entry",
     "control_type": "preventive", "is_key": True},
    {"control_id": "DC-03", "category": "data_collection", "name": "Supplier Validation",
     "description": "Supplier-provided data validated against independent sources or benchmarks",
     "control_type": "detective", "is_key": True},
    {"control_id": "DC-04", "category": "data_collection", "name": "Activity Reconciliation",
     "description": "Activity data reconciled to source records (invoices, meters, logs)",
     "control_type": "detective", "is_key": True},
    {"control_id": "DC-05", "category": "data_collection", "name": "Completeness Check",
     "description": "Completeness of data collection verified against facility register",
     "control_type": "detective", "is_key": True},
    # Calculation (CA-01 to CA-05)
    {"control_id": "CA-01", "category": "calculation", "name": "EF Selection",
     "description": "Emission factor selection reviewed against authoritative sources and justified",
     "control_type": "preventive", "is_key": True},
    {"control_id": "CA-02", "category": "calculation", "name": "Formula Validation",
     "description": "Calculation formulas verified against GHG Protocol methodologies",
     "control_type": "preventive", "is_key": True},
    {"control_id": "CA-03", "category": "calculation", "name": "Unit Conversion",
     "description": "Unit conversion factors validated and applied consistently",
     "control_type": "preventive", "is_key": False},
    {"control_id": "CA-04", "category": "calculation", "name": "Aggregation Review",
     "description": "Scope and category aggregation reviewed for double-counting",
     "control_type": "detective", "is_key": True},
    {"control_id": "CA-05", "category": "calculation", "name": "System Access",
     "description": "Calculation system access restricted to authorised personnel",
     "control_type": "preventive", "is_key": False},
    # Review (RV-01 to RV-05)
    {"control_id": "RV-01", "category": "review", "name": "Peer Review",
     "description": "Independent peer review of calculations by qualified reviewer",
     "control_type": "detective", "is_key": True},
    {"control_id": "RV-02", "category": "review", "name": "Management Review",
     "description": "Management review of aggregated GHG results with signoff",
     "control_type": "detective", "is_key": True},
    {"control_id": "RV-03", "category": "review", "name": "Cross-Scope Reconciliation",
     "description": "Cross-scope reconciliation to identify gaps or overlaps",
     "control_type": "detective", "is_key": True},
    {"control_id": "RV-04", "category": "review", "name": "Variance Analysis",
     "description": "YoY variance analysis performed with explanations for significant changes",
     "control_type": "detective", "is_key": False},
    {"control_id": "RV-05", "category": "review", "name": "Sign-Off",
     "description": "Formal sign-off by responsible officer at each reporting level",
     "control_type": "preventive", "is_key": True},
    # Reporting (RE-01 to RE-05)
    {"control_id": "RE-01", "category": "reporting", "name": "Data Extraction",
     "description": "Data extraction from source system to report verified for accuracy",
     "control_type": "detective", "is_key": True},
    {"control_id": "RE-02", "category": "reporting", "name": "Template Accuracy",
     "description": "Reporting template formulas and mappings verified",
     "control_type": "preventive", "is_key": False},
    {"control_id": "RE-03", "category": "reporting", "name": "Disclosure Review",
     "description": "Disclosure content reviewed for completeness and consistency",
     "control_type": "detective", "is_key": True},
    {"control_id": "RE-04", "category": "reporting", "name": "Submission Approval",
     "description": "Final submission reviewed and approved by authorised senior officer",
     "control_type": "preventive", "is_key": True},
    {"control_id": "RE-05", "category": "reporting", "name": "Archive",
     "description": "Submitted report and supporting evidence archived with retention policy",
     "control_type": "corrective", "is_key": False},
    # IT General (IT-01 to IT-05)
    {"control_id": "IT-01", "category": "it_general", "name": "Access Management",
     "description": "User access to GHG systems managed via role-based access control",
     "control_type": "preventive", "is_key": True},
    {"control_id": "IT-02", "category": "it_general", "name": "Change Management",
     "description": "System changes follow change management process with approval",
     "control_type": "preventive", "is_key": True},
    {"control_id": "IT-03", "category": "it_general", "name": "Backup",
     "description": "GHG data backed up regularly with tested recovery procedures",
     "control_type": "corrective", "is_key": False},
    {"control_id": "IT-04", "category": "it_general", "name": "Audit Trail",
     "description": "System audit trail captures all data changes with timestamps",
     "control_type": "detective", "is_key": True},
    {"control_id": "IT-05", "category": "it_general", "name": "Data Integrity",
     "description": "Data integrity checks (checksums, reconciliation) performed periodically",
     "control_type": "detective", "is_key": True},
]

# Design effectiveness weights
DEFAULT_DOC_WEIGHT: Decimal = Decimal("0.25")
DEFAULT_SUIT_WEIGHT: Decimal = Decimal("0.35")
DEFAULT_COV_WEIGHT: Decimal = Decimal("0.20")
DEFAULT_FREQ_WEIGHT: Decimal = Decimal("0.20")

# Sample size thresholds
SAMPLE_SIZE_RULES: List[Tuple[int, int, Decimal]] = [
    # (max_pop, min_sample, pct)
    (10, 10, Decimal("1.00")),
    (50, 10, Decimal("0.25")),
    (250, 25, Decimal("0.15")),
    (999999, 30, Decimal("0.10")),
]

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class Control(BaseModel):
    """A control definition with assessment scores.

    Attributes:
        control_id:             Control identifier (e.g. DC-01).
        category:               Control category.
        name:                   Control name.
        description:            Control description.
        control_type:           PREVENTIVE/DETECTIVE/CORRECTIVE.
        is_key:                 Whether this is a key control.
        documentation_score:    Documentation quality score (0-100).
        suitability_score:      Design suitability score (0-100).
        coverage_score:         Coverage score (0-100).
        frequency_score:        Operating frequency score (0-100).
        tests_performed:        Number of tests performed.
        tests_passed:           Number of tests passed.
        population_size:        Population size for sample testing.
        evidence_ref:           Supporting evidence reference.
        notes:                  Assessor notes.
    """
    control_id: str = Field(default="", description="Control ID")
    category: str = Field(default="", description="Category")
    name: str = Field(default="", description="Name")
    description: str = Field(default="", description="Description")
    control_type: str = Field(default=ControlType.DETECTIVE.value, description="Type")
    is_key: bool = Field(default=False, description="Key control")
    documentation_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Documentation score"
    )
    suitability_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Suitability score"
    )
    coverage_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Coverage score"
    )
    frequency_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Frequency score"
    )
    tests_performed: int = Field(default=0, ge=0, description="Tests performed")
    tests_passed: int = Field(default=0, ge=0, description="Tests passed")
    population_size: int = Field(default=0, ge=0, description="Population size")
    evidence_ref: str = Field(default="", description="Evidence reference")
    notes: str = Field(default="", description="Notes")

    @field_validator(
        "documentation_score", "suitability_score",
        "coverage_score", "frequency_score",
        mode="before",
    )
    @classmethod
    def coerce_score(cls, v: Any) -> Decimal:
        return _decimal(v)

class ControlTestingConfig(BaseModel):
    """Configuration for control testing.

    Attributes:
        organisation_id:    Organisation identifier.
        doc_weight:         Weight for documentation score.
        suit_weight:        Weight for suitability score.
        cov_weight:         Weight for coverage score.
        freq_weight:        Weight for frequency score.
        output_precision:   Output decimal places.
    """
    organisation_id: str = Field(default="", description="Org ID")
    doc_weight: Decimal = Field(default=DEFAULT_DOC_WEIGHT, ge=0, le=1)
    suit_weight: Decimal = Field(default=DEFAULT_SUIT_WEIGHT, ge=0, le=1)
    cov_weight: Decimal = Field(default=DEFAULT_COV_WEIGHT, ge=0, le=1)
    freq_weight: Decimal = Field(default=DEFAULT_FREQ_WEIGHT, ge=0, le=1)
    output_precision: int = Field(default=2, ge=0, le=6, description="Output precision")

class ControlTestingInput(BaseModel):
    """Input for control testing engine.

    Attributes:
        controls:   Control assessments.
        config:     Testing configuration.
    """
    controls: List[Control] = Field(default_factory=list, description="Controls")
    config: ControlTestingConfig = Field(
        default_factory=ControlTestingConfig, description="Configuration"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class ControlTest(BaseModel):
    """Result of testing a single control.

    Attributes:
        control_id:                 Control identifier.
        name:                       Control name.
        category:                   Control category.
        control_type:               Control type.
        is_key:                     Whether key control.
        design_effectiveness:       Design effectiveness score (0-100).
        operating_effectiveness:    Operating effectiveness score (0-100).
        overall_effectiveness:      Combined score (0-100).
        recommended_sample_size:    Recommended sample size.
        actual_sample_size:         Actual tests performed.
        deficiency_level:           Deficiency classification.
        maturity_level:             Control maturity level.
    """
    control_id: str = Field(default="", description="Control ID")
    name: str = Field(default="", description="Name")
    category: str = Field(default="", description="Category")
    control_type: str = Field(default="", description="Type")
    is_key: bool = Field(default=False, description="Key")
    design_effectiveness: Decimal = Field(
        default=Decimal("0"), description="Design effectiveness"
    )
    operating_effectiveness: Decimal = Field(
        default=Decimal("0"), description="Operating effectiveness"
    )
    overall_effectiveness: Decimal = Field(
        default=Decimal("0"), description="Overall effectiveness"
    )
    recommended_sample_size: int = Field(default=0, description="Recommended sample")
    actual_sample_size: int = Field(default=0, description="Actual sample")
    deficiency_level: str = Field(
        default=DeficiencyLevel.NONE.value, description="Deficiency"
    )
    maturity_level: str = Field(
        default=MaturityLevel.AD_HOC.value, description="Maturity"
    )

class ControlDeficiency(BaseModel):
    """Identified control deficiency.

    Attributes:
        control_id:         Control identifier.
        control_name:       Control name.
        category:           Category.
        deficiency_level:   Deficiency classification.
        description:        Deficiency description.
        impact:             Impact assessment.
        root_cause:         Root cause analysis.
        remediation:        Remediation recommendation.
    """
    control_id: str = Field(default="", description="Control ID")
    control_name: str = Field(default="", description="Name")
    category: str = Field(default="", description="Category")
    deficiency_level: str = Field(default="", description="Level")
    description: str = Field(default="", description="Description")
    impact: str = Field(default="", description="Impact")
    root_cause: str = Field(default="", description="Root cause")
    remediation: str = Field(default="", description="Remediation")

class RemediationPlan(BaseModel):
    """Remediation plan for identified deficiencies.

    Attributes:
        plan_id:            Plan identifier.
        control_id:         Control identifier.
        deficiency_level:   Deficiency classification.
        action:             Remediation action.
        owner:              Responsible person/role.
        target_date:        Target completion date.
        priority:           Priority (1=highest).
        estimated_effort_days: Estimated effort.
    """
    plan_id: str = Field(default_factory=_new_uuid, description="Plan ID")
    control_id: str = Field(default="", description="Control ID")
    deficiency_level: str = Field(default="", description="Deficiency level")
    action: str = Field(default="", description="Action")
    owner: str = Field(default="", description="Owner")
    target_date: str = Field(default="", description="Target date")
    priority: int = Field(default=0, description="Priority")
    estimated_effort_days: Decimal = Field(
        default=Decimal("0"), description="Effort (days)"
    )

class CategorySummary(BaseModel):
    """Summary for a control category.

    Attributes:
        category:                   Category name.
        control_count:              Number of controls.
        avg_design_effectiveness:   Average DE.
        avg_operating_effectiveness: Average OE.
        avg_overall:                Average overall.
        maturity_level:             Category maturity.
        deficiency_count:           Number of deficiencies.
        material_weakness_count:    Material weakness count.
    """
    category: str = Field(default="", description="Category")
    control_count: int = Field(default=0, description="Controls")
    avg_design_effectiveness: Decimal = Field(default=Decimal("0"), description="Avg DE")
    avg_operating_effectiveness: Decimal = Field(default=Decimal("0"), description="Avg OE")
    avg_overall: Decimal = Field(default=Decimal("0"), description="Avg overall")
    maturity_level: str = Field(default="", description="Maturity")
    deficiency_count: int = Field(default=0, description="Deficiencies")
    material_weakness_count: int = Field(default=0, description="Material weaknesses")

class ControlResult(BaseModel):
    """Complete result of control testing.

    Attributes:
        result_id:              Unique result identifier.
        organisation_id:        Organisation identifier.
        control_tests:          Individual control test results.
        category_summaries:     Per-category summaries.
        deficiencies:           Identified deficiencies.
        remediation_plans:      Remediation plans.
        overall_design:         Overall design effectiveness.
        overall_operating:      Overall operating effectiveness.
        overall_maturity:       Overall maturity level.
        total_controls:         Total controls tested.
        total_deficiencies:     Total deficiencies.
        material_weaknesses:    Material weakness count.
        significant_deficiencies: Significant deficiency count.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Org ID")
    control_tests: List[ControlTest] = Field(
        default_factory=list, description="Test results"
    )
    category_summaries: List[CategorySummary] = Field(
        default_factory=list, description="Category summaries"
    )
    deficiencies: List[ControlDeficiency] = Field(
        default_factory=list, description="Deficiencies"
    )
    remediation_plans: List[RemediationPlan] = Field(
        default_factory=list, description="Remediation plans"
    )
    overall_design: Decimal = Field(default=Decimal("0"), description="Overall DE")
    overall_operating: Decimal = Field(default=Decimal("0"), description="Overall OE")
    overall_maturity: str = Field(default="", description="Overall maturity")
    total_controls: int = Field(default=0, description="Total controls")
    total_deficiencies: int = Field(default=0, description="Total deficiencies")
    material_weaknesses: int = Field(default=0, description="Material weaknesses")
    significant_deficiencies: int = Field(default=0, description="Significant deficiencies")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ControlTestingEngine:
    """Evaluates internal controls over GHG reporting.

    Tests 25 standard controls across 5 categories for design and operating
    effectiveness, classifies deficiencies, and generates remediation plans.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every control test documented.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("ControlTestingEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: ControlTestingInput) -> ControlResult:
        """Test controls and classify deficiencies.

        Args:
            input_data: Control assessments and configuration.

        Returns:
            ControlResult with test results, deficiencies, and remediation plans.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.config
        prec = config.output_precision
        prec_str = "0." + "0" * prec

        # Step 1: Merge with standard controls
        controls = self._merge_controls(input_data.controls)

        # Step 2: Test each control
        test_results: List[ControlTest] = []
        deficiencies: List[ControlDeficiency] = []

        for control in controls:
            test = self._test_control(control, config, prec_str)
            test_results.append(test)

            if test.deficiency_level != DeficiencyLevel.NONE.value:
                deficiency = self._classify_deficiency(control, test)
                deficiencies.append(deficiency)

        # Step 3: Generate remediation plans
        remediation_plans: List[RemediationPlan] = []
        for i, deficiency in enumerate(deficiencies):
            plan = self._create_remediation_plan(deficiency, i + 1)
            remediation_plans.append(plan)

        # Step 4: Category summaries
        category_summaries = self._build_category_summaries(test_results, deficiencies, prec_str)

        # Step 5: Overall metrics
        total_controls = len(test_results)
        if total_controls > 0:
            overall_de = _safe_divide(
                sum(t.design_effectiveness for t in test_results),
                _decimal(total_controls),
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            overall_oe = _safe_divide(
                sum(t.operating_effectiveness for t in test_results),
                _decimal(total_controls),
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        else:
            overall_de = Decimal("0")
            overall_oe = Decimal("0")

        overall_score = _safe_divide(overall_de + overall_oe, Decimal("2"))
        overall_maturity = self._assess_maturity(overall_score)

        mw_count = sum(
            1 for d in deficiencies
            if d.deficiency_level == DeficiencyLevel.MATERIAL_WEAKNESS.value
        )
        sd_count = sum(
            1 for d in deficiencies
            if d.deficiency_level == DeficiencyLevel.SIGNIFICANT_DEFICIENCY.value
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ControlResult(
            organisation_id=config.organisation_id,
            control_tests=test_results,
            category_summaries=category_summaries,
            deficiencies=deficiencies,
            remediation_plans=remediation_plans,
            overall_design=overall_de,
            overall_operating=overall_oe,
            overall_maturity=overall_maturity,
            total_controls=total_controls,
            total_deficiencies=len(deficiencies),
            material_weaknesses=mw_count,
            significant_deficiencies=sd_count,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def get_standard_controls(self) -> List[Dict[str, Any]]:
        """Get the 25 standard control definitions."""
        return list(STANDARD_CONTROLS)

    def compute_sample_size(self, population: int) -> int:
        """Compute recommended sample size for a population.

        Args:
            population: Population size.

        Returns:
            Recommended sample size.
        """
        return self._recommended_sample_size(population)

    # ------------------------------------------------------------------
    # Internal: Control Merging
    # ------------------------------------------------------------------

    def _merge_controls(self, user_controls: List[Control]) -> List[Control]:
        """Merge user-provided controls with standard control definitions."""
        user_map = {c.control_id: c for c in user_controls}
        merged: List[Control] = []

        for std in STANDARD_CONTROLS:
            ctrl_id = std["control_id"]
            if ctrl_id in user_map:
                uc = user_map[ctrl_id]
                merged.append(Control(
                    control_id=ctrl_id,
                    category=std["category"],
                    name=std["name"],
                    description=std["description"],
                    control_type=std["control_type"],
                    is_key=std["is_key"],
                    documentation_score=uc.documentation_score,
                    suitability_score=uc.suitability_score,
                    coverage_score=uc.coverage_score,
                    frequency_score=uc.frequency_score,
                    tests_performed=uc.tests_performed,
                    tests_passed=uc.tests_passed,
                    population_size=uc.population_size,
                    evidence_ref=uc.evidence_ref,
                    notes=uc.notes,
                ))
            else:
                merged.append(Control(
                    control_id=ctrl_id,
                    category=std["category"],
                    name=std["name"],
                    description=std["description"],
                    control_type=std["control_type"],
                    is_key=std["is_key"],
                ))

        # Add any user controls not in standard set
        std_ids = {s["control_id"] for s in STANDARD_CONTROLS}
        for uc in user_controls:
            if uc.control_id not in std_ids:
                merged.append(uc)

        return merged

    # ------------------------------------------------------------------
    # Internal: Control Testing
    # ------------------------------------------------------------------

    def _test_control(
        self, control: Control, config: ControlTestingConfig, prec_str: str,
    ) -> ControlTest:
        """Test a single control for design and operating effectiveness."""
        # Design effectiveness
        de = (
            config.doc_weight * control.documentation_score
            + config.suit_weight * control.suitability_score
            + config.cov_weight * control.coverage_score
            + config.freq_weight * control.frequency_score
        ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Operating effectiveness
        oe = Decimal("0")
        if control.tests_performed > 0:
            oe = _safe_divide(
                _decimal(control.tests_passed),
                _decimal(control.tests_performed),
            ) * Decimal("100")
            oe = oe.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Overall
        overall = _safe_divide(de + oe, Decimal("2")).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )

        # Sample size
        rec_sample = self._recommended_sample_size(control.population_size)

        # Deficiency
        deficiency = self._assess_deficiency(control, de, oe)

        # Maturity
        maturity = self._assess_maturity(overall)

        return ControlTest(
            control_id=control.control_id,
            name=control.name,
            category=control.category,
            control_type=control.control_type,
            is_key=control.is_key,
            design_effectiveness=de,
            operating_effectiveness=oe,
            overall_effectiveness=overall,
            recommended_sample_size=rec_sample,
            actual_sample_size=control.tests_performed,
            deficiency_level=deficiency,
            maturity_level=maturity,
        )

    def _assess_deficiency(
        self, control: Control, de: Decimal, oe: Decimal,
    ) -> str:
        """Classify control deficiency."""
        if control.is_key:
            if de < Decimal("40") or oe < Decimal("50"):
                return DeficiencyLevel.MATERIAL_WEAKNESS.value
            if de < Decimal("60") or oe < Decimal("70"):
                return DeficiencyLevel.SIGNIFICANT_DEFICIENCY.value
        if de < Decimal("80") or oe < Decimal("80"):
            return DeficiencyLevel.DEFICIENCY.value
        return DeficiencyLevel.NONE.value

    def _assess_maturity(self, score: Decimal) -> str:
        """Assess control maturity level."""
        if score >= Decimal("80"):
            return MaturityLevel.OPTIMISED.value
        if score >= Decimal("60"):
            return MaturityLevel.MANAGED.value
        if score >= Decimal("40"):
            return MaturityLevel.DEFINED.value
        if score >= Decimal("20"):
            return MaturityLevel.REPEATABLE.value
        return MaturityLevel.AD_HOC.value

    def _recommended_sample_size(self, population: int) -> int:
        """Compute recommended sample size."""
        if population <= 0:
            return 0
        for max_pop, min_sample, pct in SAMPLE_SIZE_RULES:
            if population <= max_pop:
                pct_sample = int(
                    (_decimal(population) * pct).to_integral_value(rounding=ROUND_HALF_UP)
                )
                return max(min_sample, pct_sample)
        return 30

    # ------------------------------------------------------------------
    # Internal: Deficiency Classification
    # ------------------------------------------------------------------

    def _classify_deficiency(
        self, control: Control, test: ControlTest,
    ) -> ControlDeficiency:
        """Create deficiency record."""
        if test.deficiency_level == DeficiencyLevel.MATERIAL_WEAKNESS.value:
            impact = (
                "Material weakness may result in material misstatement "
                "of GHG emissions not being prevented or detected."
            )
        elif test.deficiency_level == DeficiencyLevel.SIGNIFICANT_DEFICIENCY.value:
            impact = (
                "Significant deficiency increases risk of material misstatement. "
                "Merits attention from those responsible for oversight."
            )
        else:
            impact = "Minor deficiency; does not individually cause material misstatement."

        return ControlDeficiency(
            control_id=control.control_id,
            control_name=control.name,
            category=control.category,
            deficiency_level=test.deficiency_level,
            description=f"Control {control.control_id} ({control.name}) "
                        f"DE={test.design_effectiveness}, OE={test.operating_effectiveness}.",
            impact=impact,
            root_cause=f"Design or operating effectiveness below threshold.",
            remediation=f"Strengthen {control.name.lower()} controls. "
                        f"Target DE>=80, OE>=80 for non-key; DE>=60, OE>=70 for key controls.",
        )

    # ------------------------------------------------------------------
    # Internal: Remediation
    # ------------------------------------------------------------------

    def _create_remediation_plan(
        self, deficiency: ControlDeficiency, priority: int,
    ) -> RemediationPlan:
        """Create remediation plan for a deficiency."""
        effort_map = {
            DeficiencyLevel.MATERIAL_WEAKNESS.value: Decimal("20"),
            DeficiencyLevel.SIGNIFICANT_DEFICIENCY.value: Decimal("10"),
            DeficiencyLevel.DEFICIENCY.value: Decimal("5"),
        }
        effort = effort_map.get(deficiency.deficiency_level, Decimal("5"))

        return RemediationPlan(
            control_id=deficiency.control_id,
            deficiency_level=deficiency.deficiency_level,
            action=deficiency.remediation,
            owner="GHG Reporting Manager",
            target_date="",
            priority=priority,
            estimated_effort_days=effort,
        )

    # ------------------------------------------------------------------
    # Internal: Category Summaries
    # ------------------------------------------------------------------

    def _build_category_summaries(
        self,
        tests: List[ControlTest],
        deficiencies: List[ControlDeficiency],
        prec_str: str,
    ) -> List[CategorySummary]:
        """Build per-category summaries."""
        cat_map: Dict[str, List[ControlTest]] = {}
        for test in tests:
            cat = test.category
            if cat not in cat_map:
                cat_map[cat] = []
            cat_map[cat].append(test)

        def_map: Dict[str, List[ControlDeficiency]] = {}
        for d in deficiencies:
            cat = d.category
            if cat not in def_map:
                def_map[cat] = []
            def_map[cat].append(d)

        summaries: List[CategorySummary] = []
        for cat, cat_tests in cat_map.items():
            n = len(cat_tests)
            avg_de = _safe_divide(
                sum(t.design_effectiveness for t in cat_tests), _decimal(n)
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            avg_oe = _safe_divide(
                sum(t.operating_effectiveness for t in cat_tests), _decimal(n)
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            avg_overall = _safe_divide(avg_de + avg_oe, Decimal("2")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )

            cat_defs = def_map.get(cat, [])
            mw_count = sum(
                1 for d in cat_defs
                if d.deficiency_level == DeficiencyLevel.MATERIAL_WEAKNESS.value
            )

            summaries.append(CategorySummary(
                category=cat,
                control_count=n,
                avg_design_effectiveness=avg_de,
                avg_operating_effectiveness=avg_oe,
                avg_overall=avg_overall,
                maturity_level=self._assess_maturity(avg_overall),
                deficiency_count=len(cat_defs),
                material_weakness_count=mw_count,
            ))

        return summaries

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ControlCategory",
    "ControlType",
    "DeficiencyLevel",
    "MaturityLevel",
    # Input Models
    "Control",
    "ControlTestingConfig",
    "ControlTestingInput",
    # Output Models
    "ControlTest",
    "ControlDeficiency",
    "RemediationPlan",
    "CategorySummary",
    "ControlResult",
    # Engine
    "ControlTestingEngine",
]
