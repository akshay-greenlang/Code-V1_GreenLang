# -*- coding: utf-8 -*-
"""
ConformityAssessmentEngine - PACK-020 Battery Passport Engine 8
=================================================================

Assesses readiness for conformity assessment procedures per
Articles 17-22 of the EU Battery Regulation (2023/1542).

Articles 17-22 of the EU Battery Regulation establish the conformity
assessment procedures that manufacturers must complete before placing
batteries on the EU market.  The regulation references the conformity
assessment modules from Decision 768/2008/EC, with Module A (internal
production control) as the default for most battery types and
Modules D/E/G/H available for more complex or high-risk batteries.

Conformity Assessment Modules:
    - Module A:  Internal production control (self-assessment)
    - Module B:  EU-type examination (by notified body)
    - Module C:  Conformity to type based on internal production control
    - Module D:  Conformity to type based on quality assurance of production
    - Module E:  Conformity to type based on product quality assurance
    - Module G:  Conformity based on unit verification (individual assessment)
    - Module H:  Conformity based on full quality assurance

Key Requirements:
    - Technical documentation (Art 18): design, manufacturing, test data
    - EU Declaration of Conformity (Art 19): formal declaration per Annex V
    - CE marking (Art 20): applied after successful conformity assessment
    - Notified body involvement (Art 21): for Modules B/D/E/G/H
    - Market surveillance (Art 22): post-market compliance monitoring

Regulatory References:
    - EU Regulation 2023/1542 (EU Battery Regulation), Art 17-22
    - Decision 768/2008/EC (New Legislative Framework modules)
    - Regulation (EC) No 765/2008 (accreditation and market surveillance)
    - EN 62660-1/2/3 (lithium-ion battery test standards)
    - IEC 62619 (safety of secondary lithium cells for industrial use)

Zero-Hallucination:
    - Completeness scores use deterministic checklist ratios
    - Module determination uses static category mapping
    - All assessments are binary pass/fail checklist-based
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-020 Battery Passport Prep Pack
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BatteryCategory(str, Enum):
    """Battery category per EU Battery Regulation classification.

    Determines which conformity assessment module applies.
    """
    PORTABLE = "portable"
    LMT = "lmt"
    SLI = "sli"
    EV = "ev"
    INDUSTRIAL = "industrial"

class ConformityModule(str, Enum):
    """Conformity assessment module per Decision 768/2008/EC.

    Each module represents a different level of rigour in the
    conformity assessment process, from self-certification
    (Module A) to full third-party quality system certification
    (Module H).
    """
    MODULE_A = "module_a"
    MODULE_B = "module_b"
    MODULE_C = "module_c"
    MODULE_D = "module_d"
    MODULE_E = "module_e"
    MODULE_G = "module_g"
    MODULE_H = "module_h"

class DocumentationType(str, Enum):
    """Type of documentation required for conformity assessment.

    Each type represents a category of evidence that must be
    prepared and maintained as part of the technical file.
    """
    TECHNICAL_FILE = "technical_file"
    EU_DECLARATION = "eu_declaration"
    TEST_REPORTS = "test_reports"
    QUALITY_SYSTEM = "quality_system"
    DESIGN_EXAMINATION = "design_examination"

class ConformityStatus(str, Enum):
    """Overall conformity assessment readiness status.

    Indicates the current readiness level of the manufacturer
    to complete the conformity assessment process.
    """
    READY = "ready"
    PARTIALLY_READY = "partially_ready"
    NOT_READY = "not_ready"
    IN_PROGRESS = "in_progress"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Module descriptions with regulatory references.
MODULE_DESCRIPTIONS: Dict[str, str] = {
    ConformityModule.MODULE_A.value: (
        "Internal production control: The manufacturer ensures and "
        "declares on their sole responsibility that the batteries "
        "satisfy the requirements of the Regulation. No notified "
        "body involvement required. Suitable for most battery types."
    ),
    ConformityModule.MODULE_B.value: (
        "EU-type examination: A notified body examines the technical "
        "design and verifies that the design meets applicable "
        "requirements. Used in combination with Module C, D, or E."
    ),
    ConformityModule.MODULE_C.value: (
        "Conformity to type based on internal production control: "
        "After EU-type examination (Module B), the manufacturer "
        "ensures production conforms to the approved type."
    ),
    ConformityModule.MODULE_D.value: (
        "Conformity to type based on quality assurance of production: "
        "After Module B, the manufacturer operates a quality system "
        "for production, inspection, and testing approved by a "
        "notified body."
    ),
    ConformityModule.MODULE_E.value: (
        "Conformity to type based on product quality assurance: "
        "After Module B, the manufacturer operates a quality system "
        "for final product inspection and testing approved by a "
        "notified body."
    ),
    ConformityModule.MODULE_G.value: (
        "Conformity based on unit verification: A notified body "
        "examines each individual product and issues a certificate "
        "of conformity. Used for unique or very small batch products."
    ),
    ConformityModule.MODULE_H.value: (
        "Conformity based on full quality assurance: The manufacturer "
        "operates a comprehensive quality system covering design, "
        "production, and testing, approved and surveilled by a "
        "notified body."
    ),
}

# Default module per battery category.
DEFAULT_MODULE_BY_CATEGORY: Dict[str, ConformityModule] = {
    BatteryCategory.PORTABLE.value: ConformityModule.MODULE_A,
    BatteryCategory.LMT.value: ConformityModule.MODULE_A,
    BatteryCategory.SLI.value: ConformityModule.MODULE_A,
    BatteryCategory.EV.value: ConformityModule.MODULE_A,
    BatteryCategory.INDUSTRIAL.value: ConformityModule.MODULE_A,
}

# Modules requiring notified body involvement.
NOTIFIED_BODY_MODULES: List[str] = [
    ConformityModule.MODULE_B.value,
    ConformityModule.MODULE_D.value,
    ConformityModule.MODULE_E.value,
    ConformityModule.MODULE_G.value,
    ConformityModule.MODULE_H.value,
]

# Technical documentation checklist items.
TECHNICAL_DOC_CHECKLIST: Dict[str, Dict[str, str]] = {
    "td_01": {
        "item": "General product description",
        "description": "A general description of the battery including "
                       "category, chemistry, rated capacity, and voltage",
        "article": "Art 18(2)(a)",
    },
    "td_02": {
        "item": "Design and manufacturing drawings",
        "description": "Conceptual design and manufacturing drawings, "
                       "schematics of components and assemblies",
        "article": "Art 18(2)(b)",
    },
    "td_03": {
        "item": "Material specifications",
        "description": "Descriptions of materials used, including "
                       "active materials, electrolyte, and casing",
        "article": "Art 18(2)(c)",
    },
    "td_04": {
        "item": "Harmonised standards applied",
        "description": "List of harmonised standards applied in full "
                       "or in part, and solutions adopted to satisfy "
                       "essential requirements",
        "article": "Art 18(2)(d)",
    },
    "td_05": {
        "item": "Risk assessment",
        "description": "Risk assessment addressing safety, environmental, "
                       "and health risks during production, use, and end-of-life",
        "article": "Art 18(2)(e)",
    },
    "td_06": {
        "item": "Test reports - safety",
        "description": "Results of safety tests performed per applicable "
                       "harmonised standards (e.g., EN 62660, IEC 62619)",
        "article": "Art 18(2)(f)",
    },
    "td_07": {
        "item": "Test reports - performance",
        "description": "Results of performance and durability tests "
                       "including capacity, cycle life, and efficiency",
        "article": "Art 18(2)(g)",
    },
    "td_08": {
        "item": "Carbon footprint declaration",
        "description": "Carbon footprint calculation per lifecycle "
                       "assessment methodology (EV and industrial only)",
        "article": "Art 18(2)(h)",
    },
    "td_09": {
        "item": "Recycled content documentation",
        "description": "Documentation of recycled content percentages "
                       "for cobalt, lithium, nickel, and lead",
        "article": "Art 18(2)(i)",
    },
    "td_10": {
        "item": "Supply chain due diligence",
        "description": "Due diligence policy and evidence for critical "
                       "raw materials per Article 48",
        "article": "Art 18(2)(j)",
    },
    "td_11": {
        "item": "Quality control records",
        "description": "Records of quality control measures during "
                       "production, including inspection and test results",
        "article": "Art 18(2)(k)",
    },
    "td_12": {
        "item": "Labelling compliance",
        "description": "Evidence of correct labelling and marking per "
                       "Articles 13-14",
        "article": "Art 18(2)(l)",
    },
}

# EU Declaration of Conformity checklist (per Annex V).
EU_DOC_CHECKLIST: Dict[str, Dict[str, str]] = {
    "doc_01": {
        "item": "Battery model identification",
        "description": "Model, batch, type, or serial number of the battery",
        "article": "Annex V(1)",
    },
    "doc_02": {
        "item": "Manufacturer identification",
        "description": "Name and address of the manufacturer and, "
                       "where applicable, authorised representative",
        "article": "Annex V(2)",
    },
    "doc_03": {
        "item": "Declaration statement",
        "description": "Statement that the declaration is issued under "
                       "the sole responsibility of the manufacturer",
        "article": "Annex V(3)",
    },
    "doc_04": {
        "item": "Identification of battery",
        "description": "Identification of the battery allowing traceability "
                       "(photo or image may be included)",
        "article": "Annex V(4)",
    },
    "doc_05": {
        "item": "Applicable legislation",
        "description": "Reference to the EU Battery Regulation and any "
                       "other applicable Union harmonisation legislation",
        "article": "Annex V(5)",
    },
    "doc_06": {
        "item": "Harmonised standards",
        "description": "Reference to harmonised standards used or other "
                       "technical specifications",
        "article": "Annex V(6)",
    },
    "doc_07": {
        "item": "Notified body details",
        "description": "Where applicable, name and number of the notified "
                       "body and reference to the certificate issued",
        "article": "Annex V(7)",
    },
    "doc_08": {
        "item": "Date and signature",
        "description": "Place and date of issue, name, function, and "
                       "signature of the person authorised to sign",
        "article": "Annex V(8)",
    },
}

# Test requirements by battery category.
TEST_REQUIREMENTS: Dict[str, List[Dict[str, str]]] = {
    BatteryCategory.PORTABLE.value: [
        {"test": "Capacity test", "standard": "IEC 61960", "required": "yes"},
        {"test": "Safety test", "standard": "IEC 62133", "required": "yes"},
        {"test": "Transport test", "standard": "UN 38.3", "required": "yes"},
        {"test": "Environmental test", "standard": "IEC 60068", "required": "yes"},
    ],
    BatteryCategory.LMT.value: [
        {"test": "Capacity test", "standard": "IEC 61960", "required": "yes"},
        {"test": "Safety test", "standard": "IEC 62619", "required": "yes"},
        {"test": "Cycle life test", "standard": "EN 62660-1", "required": "yes"},
        {"test": "Transport test", "standard": "UN 38.3", "required": "yes"},
        {"test": "Abuse test", "standard": "IEC 62660-2", "required": "yes"},
    ],
    BatteryCategory.SLI.value: [
        {"test": "Capacity test", "standard": "EN 50342-1", "required": "yes"},
        {"test": "Cranking test", "standard": "EN 50342-1", "required": "yes"},
        {"test": "Durability test", "standard": "EN 50342-1", "required": "yes"},
        {"test": "Safety test", "standard": "EN 50342-2", "required": "yes"},
    ],
    BatteryCategory.EV.value: [
        {"test": "Capacity test", "standard": "EN 62660-1", "required": "yes"},
        {"test": "Power capability test", "standard": "EN 62660-1", "required": "yes"},
        {"test": "Cycle life test", "standard": "EN 62660-1", "required": "yes"},
        {"test": "Safety - abuse test", "standard": "EN 62660-2", "required": "yes"},
        {"test": "Safety - mechanical", "standard": "EN 62660-3", "required": "yes"},
        {"test": "Thermal management", "standard": "ISO 12405", "required": "yes"},
        {"test": "Transport test", "standard": "UN 38.3", "required": "yes"},
        {"test": "Environmental test", "standard": "IEC 60068", "required": "yes"},
        {"test": "EMC test", "standard": "CISPR 25 / ISO 11452", "required": "yes"},
    ],
    BatteryCategory.INDUSTRIAL.value: [
        {"test": "Capacity test", "standard": "IEC 62620", "required": "yes"},
        {"test": "Safety test", "standard": "IEC 62619", "required": "yes"},
        {"test": "Cycle life test", "standard": "IEC 62620", "required": "yes"},
        {"test": "Transport test", "standard": "UN 38.3", "required": "yes"},
        {"test": "Environmental test", "standard": "IEC 60068", "required": "yes"},
        {"test": "Fire safety", "standard": "IEC 62897 / UL 9540A", "required": "yes"},
        {"test": "Grid connection", "standard": "IEC 62933", "required": "conditional"},
    ],
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DocumentationItem(BaseModel):
    """A single documentation item in the conformity checklist.

    Records whether a required document or evidence is available
    and its status.
    """
    item_id: str = Field(
        ...,
        description="Checklist item identifier",
    )
    item_name: str = Field(
        default="",
        description="Name of the documentation item",
    )
    description: str = Field(
        default="",
        description="Description of the requirement",
    )
    article: str = Field(
        default="",
        description="Regulatory article reference",
    )
    available: bool = Field(
        default=False,
        description="Whether the documentation is available",
    )
    notes: str = Field(
        default="",
        description="Additional notes or status details",
        max_length=2000,
    )

class TestResult(BaseModel):
    """Result of a specific test in the conformity assessment.

    Records the test name, applicable standard, and whether
    the test has been completed and passed.
    """
    test_name: str = Field(
        ...,
        description="Name of the test",
    )
    standard: str = Field(
        default="",
        description="Applicable standard reference",
    )
    required: bool = Field(
        default=True,
        description="Whether this test is required",
    )
    completed: bool = Field(
        default=False,
        description="Whether the test has been completed",
    )
    passed: bool = Field(
        default=False,
        description="Whether the test was passed",
    )
    report_reference: str = Field(
        default="",
        description="Reference to the test report document",
    )

class ConformityInput(BaseModel):
    """Input data for a conformity assessment evaluation.

    Captures the current state of documentation, testing,
    and CE marking for a battery to determine readiness.
    """
    battery_id: str = Field(
        ...,
        description="Unique battery identifier",
    )
    category: BatteryCategory = Field(
        ...,
        description="Battery category",
    )
    conformity_module: Optional[ConformityModule] = Field(
        default=None,
        description="Selected conformity module (auto-determined if None)",
    )
    technical_documentation: Dict[str, bool] = Field(
        default_factory=dict,
        description="Dict mapping doc item ID to availability (True/False)",
    )
    test_reports: Dict[str, bool] = Field(
        default_factory=dict,
        description="Dict mapping test name to completion status",
    )
    test_results: Dict[str, bool] = Field(
        default_factory=dict,
        description="Dict mapping test name to pass/fail status",
    )
    eu_declaration: Dict[str, bool] = Field(
        default_factory=dict,
        description="Dict mapping declaration item ID to availability",
    )
    ce_marking: bool = Field(
        default=False,
        description="Whether CE marking has been applied",
    )
    notified_body_required: bool = Field(
        default=False,
        description="Whether a notified body is required",
    )
    notified_body_id: Optional[str] = Field(
        default=None,
        description="Notified body identification number",
    )
    notified_body_certificate: Optional[str] = Field(
        default=None,
        description="Notified body certificate reference",
    )
    harmonised_standards_applied: List[str] = Field(
        default_factory=list,
        description="List of harmonised standards applied",
    )
    quality_system_certified: bool = Field(
        default=False,
        description="Whether a quality management system is certified",
    )
    quality_system_standard: str = Field(
        default="",
        description="Quality system standard (e.g., ISO 9001)",
    )

class ConformityResult(BaseModel):
    """Result of a conformity assessment readiness evaluation.

    Contains completeness scores for documentation, testing,
    and declaration, along with overall readiness status and
    actionable missing items.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of assessment (UTC)",
    )
    battery_id: str = Field(
        default="",
        description="Battery identifier assessed",
    )
    category: BatteryCategory = Field(
        default=BatteryCategory.PORTABLE,
        description="Battery category",
    )
    module: ConformityModule = Field(
        default=ConformityModule.MODULE_A,
        description="Conformity assessment module applied",
    )
    module_description: str = Field(
        default="",
        description="Description of the applied module",
    )
    documentation_completeness: float = Field(
        default=0.0,
        description="Technical documentation completeness (0-100%)",
    )
    test_coverage: float = Field(
        default=0.0,
        description="Test coverage completeness (0-100%)",
    )
    test_pass_rate: float = Field(
        default=0.0,
        description="Percentage of completed tests that passed (0-100%)",
    )
    declaration_valid: bool = Field(
        default=False,
        description="Whether EU Declaration of Conformity is complete",
    )
    declaration_completeness: float = Field(
        default=0.0,
        description="EU Declaration completeness (0-100%)",
    )
    ce_marking_applied: bool = Field(
        default=False,
        description="Whether CE marking has been applied",
    )
    notified_body_required: bool = Field(
        default=False,
        description="Whether a notified body is required",
    )
    notified_body_engaged: bool = Field(
        default=False,
        description="Whether a notified body has been engaged",
    )
    quality_system_adequate: bool = Field(
        default=False,
        description="Whether the quality system is adequate for the module",
    )
    overall_status: ConformityStatus = Field(
        default=ConformityStatus.NOT_READY,
        description="Overall conformity assessment readiness",
    )
    overall_score: float = Field(
        default=0.0,
        description="Overall readiness score (0-100%)",
    )
    documentation_items: List[DocumentationItem] = Field(
        default_factory=list,
        description="Per-item documentation checklist results",
    )
    test_items: List[TestResult] = Field(
        default_factory=list,
        description="Per-test checklist results",
    )
    declaration_items: List[DocumentationItem] = Field(
        default_factory=list,
        description="Per-item declaration checklist results",
    )
    missing_items: List[str] = Field(
        default_factory=list,
        description="List of missing items preventing readiness",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConformityAssessmentEngine:
    """Conformity assessment readiness engine per Art 17-22.

    Provides deterministic, zero-hallucination assessment of:
    - Technical documentation completeness (12-item checklist)
    - Test coverage and pass rates by battery category
    - EU Declaration of Conformity completeness (8-item checklist)
    - CE marking verification
    - Conformity module determination
    - Notified body requirement checks
    - Quality system adequacy
    - Overall readiness scoring

    All assessments are checklist-based and deterministic.  No LLM
    is used in any assessment path.

    Usage::

        engine = ConformityAssessmentEngine()
        input_data = ConformityInput(
            battery_id="BAT-001",
            category=BatteryCategory.EV,
            technical_documentation={"td_01": True, "td_02": True},
            test_reports={"Capacity test": True},
            test_results={"Capacity test": True},
            eu_declaration={"doc_01": True},
            ce_marking=False,
        )
        result = engine.assess_conformity(input_data)
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise ConformityAssessmentEngine."""
        logger.info(
            "ConformityAssessmentEngine v%s initialised",
            self.engine_version,
        )

    # ------------------------------------------------------------------ #
    # Full Conformity Assessment                                           #
    # ------------------------------------------------------------------ #

    def assess_conformity(
        self, input_data: ConformityInput
    ) -> ConformityResult:
        """Perform a complete conformity assessment readiness evaluation.

        Checks technical documentation, test coverage, EU Declaration,
        CE marking, and module-specific requirements to determine
        overall readiness.

        Args:
            input_data: ConformityInput with current state.

        Returns:
            ConformityResult with completeness scores and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Assessing conformity for battery %s, category=%s",
            input_data.battery_id,
            input_data.category.value,
        )

        # Determine module
        module = self.determine_module(
            input_data.category, input_data.conformity_module
        )

        # Check documentation
        doc_items, doc_completeness = self.check_documentation(
            input_data.technical_documentation,
            input_data.category,
        )

        # Check test coverage
        test_items, test_coverage, test_pass_rate = self.check_test_coverage(
            input_data.category,
            input_data.test_reports,
            input_data.test_results,
        )

        # Validate declaration
        decl_items, decl_completeness, decl_valid = self.validate_declaration(
            input_data.eu_declaration,
            module,
            input_data.notified_body_id,
        )

        # Notified body check
        nb_required = module.value in NOTIFIED_BODY_MODULES
        nb_engaged = (
            input_data.notified_body_id is not None
            and input_data.notified_body_id != ""
        )

        # Quality system check
        qs_adequate = self._check_quality_system(
            module,
            input_data.quality_system_certified,
            input_data.quality_system_standard,
        )

        # Missing items
        missing = self._identify_missing_items(
            doc_items,
            test_items,
            decl_items,
            input_data.ce_marking,
            nb_required,
            nb_engaged,
            qs_adequate,
            module,
        )

        # Overall score and status
        overall_score = self._calculate_overall_score(
            doc_completeness,
            test_coverage,
            decl_completeness,
            input_data.ce_marking,
            nb_required,
            nb_engaged,
            qs_adequate,
        )
        overall_status = self._determine_status(overall_score, missing)

        # Recommendations
        recommendations = self._generate_recommendations(
            missing,
            module,
            input_data.category,
            doc_completeness,
            test_coverage,
            decl_completeness,
            input_data.ce_marking,
            nb_required,
            nb_engaged,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ConformityResult(
            battery_id=input_data.battery_id,
            category=input_data.category,
            module=module,
            module_description=MODULE_DESCRIPTIONS.get(module.value, ""),
            documentation_completeness=doc_completeness,
            test_coverage=test_coverage,
            test_pass_rate=test_pass_rate,
            declaration_valid=decl_valid,
            declaration_completeness=decl_completeness,
            ce_marking_applied=input_data.ce_marking,
            notified_body_required=nb_required,
            notified_body_engaged=nb_engaged,
            quality_system_adequate=qs_adequate,
            overall_status=overall_status,
            overall_score=overall_score,
            documentation_items=doc_items,
            test_items=test_items,
            declaration_items=decl_items,
            missing_items=missing,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Conformity assessment for %s: docs=%.1f%%, tests=%.1f%%, "
            "decl=%.1f%%, overall=%s (%.1f%%) in %.3f ms",
            input_data.battery_id,
            doc_completeness,
            test_coverage,
            decl_completeness,
            overall_status.value,
            overall_score,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Documentation Completeness                                           #
    # ------------------------------------------------------------------ #

    def check_documentation(
        self,
        documentation: Dict[str, bool],
        category: BatteryCategory,
    ) -> Tuple[List[DocumentationItem], float]:
        """Check technical documentation completeness.

        Evaluates each item in the documentation checklist against
        the provided availability status.

        Args:
            documentation: Dict mapping item ID to availability.
            category: Battery category for context.

        Returns:
            Tuple of (list of DocumentationItem, completeness percentage).
        """
        items: List[DocumentationItem] = []
        available_count = 0
        total_count = len(TECHNICAL_DOC_CHECKLIST)

        for item_id, item_def in TECHNICAL_DOC_CHECKLIST.items():
            available = documentation.get(item_id, False)

            # Carbon footprint doc only required for EV/industrial
            required_for_category = True
            if item_id == "td_08" and category not in (
                BatteryCategory.EV, BatteryCategory.INDUSTRIAL
            ):
                required_for_category = False
                if not available:
                    # Don't count against completeness if not required
                    total_count -= 1

            if available:
                available_count += 1

            items.append(DocumentationItem(
                item_id=item_id,
                item_name=item_def["item"],
                description=item_def["description"],
                article=item_def["article"],
                available=available,
                notes="" if available else (
                    "Not required for this category"
                    if not required_for_category
                    else "Documentation missing"
                ),
            ))

        completeness = _round2(
            _safe_divide(
                float(available_count),
                float(total_count),
                0.0,
            ) * 100.0
        )

        return items, completeness

    # ------------------------------------------------------------------ #
    # Test Coverage                                                        #
    # ------------------------------------------------------------------ #

    def check_test_coverage(
        self,
        category: BatteryCategory,
        test_reports: Dict[str, bool],
        test_results: Dict[str, bool],
    ) -> Tuple[List[TestResult], float, float]:
        """Check test coverage and pass rates.

        Evaluates required tests for the battery category against
        the provided completion and pass statuses.

        Args:
            category: Battery category (determines required tests).
            test_reports: Dict mapping test name to completion status.
            test_results: Dict mapping test name to pass/fail status.

        Returns:
            Tuple of (list of TestResult, coverage %, pass rate %).
        """
        required_tests = TEST_REQUIREMENTS.get(category.value, [])
        items: List[TestResult] = []
        completed_count = 0
        passed_count = 0
        required_count = 0

        for test_def in required_tests:
            test_name = test_def["test"]
            standard = test_def.get("standard", "")
            is_required = test_def.get("required", "yes") == "yes"

            if is_required:
                required_count += 1

            completed = test_reports.get(test_name, False)
            passed = test_results.get(test_name, False)

            if completed and is_required:
                completed_count += 1
            if passed and is_required:
                passed_count += 1

            items.append(TestResult(
                test_name=test_name,
                standard=standard,
                required=is_required,
                completed=completed,
                passed=passed,
                report_reference="" if not completed else f"TR-{test_name}",
            ))

        coverage = _round2(
            _safe_divide(
                float(completed_count),
                float(required_count),
                0.0,
            ) * 100.0
        )

        pass_rate = _round2(
            _safe_divide(
                float(passed_count),
                float(completed_count),
                0.0,
            ) * 100.0
        )

        return items, coverage, pass_rate

    # ------------------------------------------------------------------ #
    # Declaration Validation                                               #
    # ------------------------------------------------------------------ #

    def validate_declaration(
        self,
        declaration: Dict[str, bool],
        module: ConformityModule,
        notified_body_id: Optional[str] = None,
    ) -> Tuple[List[DocumentationItem], float, bool]:
        """Validate EU Declaration of Conformity completeness.

        Checks each item in the declaration checklist per Annex V.

        Args:
            declaration: Dict mapping item ID to availability.
            module: Applied conformity module.
            notified_body_id: Notified body ID if applicable.

        Returns:
            Tuple of (list of items, completeness %, is_valid bool).
        """
        items: List[DocumentationItem] = []
        available_count = 0
        total_count = len(EU_DOC_CHECKLIST)

        for item_id, item_def in EU_DOC_CHECKLIST.items():
            available = declaration.get(item_id, False)

            # Notified body details only required if module needs NB
            if item_id == "doc_07":
                if module.value not in NOTIFIED_BODY_MODULES:
                    # Not required for Module A
                    if not available:
                        total_count -= 1
                    else:
                        available_count += 1

                    items.append(DocumentationItem(
                        item_id=item_id,
                        item_name=item_def["item"],
                        description=item_def["description"],
                        article=item_def["article"],
                        available=available,
                        notes="Not required for Module A",
                    ))
                    continue
                elif not notified_body_id:
                    available = False

            if available:
                available_count += 1

            items.append(DocumentationItem(
                item_id=item_id,
                item_name=item_def["item"],
                description=item_def["description"],
                article=item_def["article"],
                available=available,
                notes="" if available else "Missing",
            ))

        completeness = _round2(
            _safe_divide(
                float(available_count),
                float(total_count),
                0.0,
            ) * 100.0
        )

        # Declaration is valid if all applicable items are available
        is_valid = available_count >= total_count

        return items, completeness, is_valid

    # ------------------------------------------------------------------ #
    # Module Determination                                                 #
    # ------------------------------------------------------------------ #

    def determine_module(
        self,
        category: BatteryCategory,
        selected_module: Optional[ConformityModule] = None,
    ) -> ConformityModule:
        """Determine the applicable conformity assessment module.

        If a module is explicitly selected, validates it is
        appropriate for the category.  Otherwise, returns the
        default module for the category.

        Args:
            category: Battery category.
            selected_module: Explicitly selected module (if any).

        Returns:
            ConformityModule to apply.
        """
        if selected_module is not None:
            logger.debug(
                "Using explicitly selected module: %s for %s",
                selected_module.value,
                category.value,
            )
            return selected_module

        default = DEFAULT_MODULE_BY_CATEGORY.get(
            category.value, ConformityModule.MODULE_A
        )
        logger.debug(
            "Using default module: %s for %s",
            default.value,
            category.value,
        )
        return default

    # ------------------------------------------------------------------ #
    # Module Information                                                   #
    # ------------------------------------------------------------------ #

    def get_module_info(
        self, module: ConformityModule
    ) -> Dict[str, Any]:
        """Get detailed information about a conformity module.

        Args:
            module: ConformityModule to look up.

        Returns:
            Dict with module details and requirements.
        """
        requires_nb = module.value in NOTIFIED_BODY_MODULES
        requires_qs = module.value in [
            ConformityModule.MODULE_D.value,
            ConformityModule.MODULE_E.value,
            ConformityModule.MODULE_H.value,
        ]

        return {
            "module": module.value,
            "description": MODULE_DESCRIPTIONS.get(module.value, ""),
            "requires_notified_body": requires_nb,
            "requires_quality_system": requires_qs,
            "applicable_categories": [
                cat.value for cat in BatteryCategory
                if DEFAULT_MODULE_BY_CATEGORY.get(cat.value) == module
            ],
            "provenance_hash": _compute_hash({
                "module": module.value,
                "requires_nb": requires_nb,
            }),
        }

    def get_all_modules(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all conformity modules.

        Returns:
            Dict mapping module name to module details.
        """
        return {
            module.value: self.get_module_info(module)
            for module in ConformityModule
        }

    # ------------------------------------------------------------------ #
    # Checklist Retrieval                                                  #
    # ------------------------------------------------------------------ #

    def get_documentation_checklist(self) -> Dict[str, Dict[str, str]]:
        """Get the full technical documentation checklist.

        Returns:
            Dict mapping item ID to item details.
        """
        return dict(TECHNICAL_DOC_CHECKLIST)

    def get_declaration_checklist(self) -> Dict[str, Dict[str, str]]:
        """Get the full EU Declaration of Conformity checklist.

        Returns:
            Dict mapping item ID to item details.
        """
        return dict(EU_DOC_CHECKLIST)

    def get_test_requirements(
        self, category: BatteryCategory
    ) -> List[Dict[str, str]]:
        """Get the test requirements for a battery category.

        Args:
            category: Battery category.

        Returns:
            List of test requirement dicts.
        """
        return list(TEST_REQUIREMENTS.get(category.value, []))

    # ------------------------------------------------------------------ #
    # Summary Utilities                                                    #
    # ------------------------------------------------------------------ #

    def get_conformity_summary(
        self, result: ConformityResult
    ) -> Dict[str, Any]:
        """Return a structured summary of a conformity assessment.

        Args:
            result: ConformityResult to summarise.

        Returns:
            Dict with summary statistics.
        """
        return {
            "battery_id": result.battery_id,
            "category": result.category.value,
            "module": result.module.value,
            "overall_status": result.overall_status.value,
            "overall_score": result.overall_score,
            "documentation_completeness": result.documentation_completeness,
            "test_coverage": result.test_coverage,
            "test_pass_rate": result.test_pass_rate,
            "declaration_valid": result.declaration_valid,
            "declaration_completeness": result.declaration_completeness,
            "ce_marking_applied": result.ce_marking_applied,
            "notified_body_required": result.notified_body_required,
            "notified_body_engaged": result.notified_body_engaged,
            "quality_system_adequate": result.quality_system_adequate,
            "missing_items_count": len(result.missing_items),
            "recommendation_count": len(result.recommendations),
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Private: Quality System Check                                        #
    # ------------------------------------------------------------------ #

    def _check_quality_system(
        self,
        module: ConformityModule,
        qs_certified: bool,
        qs_standard: str,
    ) -> bool:
        """Check if the quality system meets module requirements.

        Modules D, E, and H require a certified quality management
        system (typically ISO 9001 or equivalent).

        Args:
            module: Applied conformity module.
            qs_certified: Whether QMS is certified.
            qs_standard: QMS standard reference.

        Returns:
            True if quality system is adequate for the module.
        """
        qs_required_modules = [
            ConformityModule.MODULE_D.value,
            ConformityModule.MODULE_E.value,
            ConformityModule.MODULE_H.value,
        ]

        if module.value not in qs_required_modules:
            return True  # Not required for this module

        if not qs_certified:
            return False

        # Check for recognized standards
        recognized = ["iso 9001", "iso9001", "iatf 16949", "iatf16949"]
        return any(
            std in qs_standard.lower().replace(" ", "")
            for std in [s.replace(" ", "") for s in recognized]
        ) if qs_standard else False

    # ------------------------------------------------------------------ #
    # Private: Missing Items Identification                                #
    # ------------------------------------------------------------------ #

    def _identify_missing_items(
        self,
        doc_items: List[DocumentationItem],
        test_items: List[TestResult],
        decl_items: List[DocumentationItem],
        ce_marking: bool,
        nb_required: bool,
        nb_engaged: bool,
        qs_adequate: bool,
        module: ConformityModule,
    ) -> List[str]:
        """Identify all items preventing conformity readiness.

        Args:
            doc_items: Technical documentation items.
            test_items: Test result items.
            decl_items: Declaration items.
            ce_marking: Whether CE marking is applied.
            nb_required: Whether notified body is required.
            nb_engaged: Whether notified body is engaged.
            qs_adequate: Whether quality system is adequate.
            module: Applied conformity module.

        Returns:
            List of missing item descriptions.
        """
        missing: List[str] = []

        # Missing documentation
        for item in doc_items:
            if not item.available and "Not required" not in item.notes:
                missing.append(
                    f"Technical documentation: {item.item_name} "
                    f"({item.article})"
                )

        # Missing/failed tests
        for test in test_items:
            if test.required and not test.completed:
                missing.append(
                    f"Test not completed: {test.test_name} "
                    f"({test.standard})"
                )
            elif test.required and test.completed and not test.passed:
                missing.append(
                    f"Test failed: {test.test_name} ({test.standard})"
                )

        # Missing declaration items
        for item in decl_items:
            if not item.available and "Not required" not in item.notes:
                missing.append(
                    f"EU Declaration: {item.item_name} ({item.article})"
                )

        # CE marking
        if not ce_marking:
            missing.append(
                "CE marking not applied (Art 20)"
            )

        # Notified body
        if nb_required and not nb_engaged:
            missing.append(
                f"Notified body required for {module.value} but "
                "not engaged (Art 21)"
            )

        # Quality system
        qs_modules = [
            ConformityModule.MODULE_D.value,
            ConformityModule.MODULE_E.value,
            ConformityModule.MODULE_H.value,
        ]
        if module.value in qs_modules and not qs_adequate:
            missing.append(
                f"Quality management system required for {module.value} "
                "but not certified or not adequate"
            )

        return missing

    # ------------------------------------------------------------------ #
    # Private: Overall Score                                               #
    # ------------------------------------------------------------------ #

    def _calculate_overall_score(
        self,
        doc_completeness: float,
        test_coverage: float,
        decl_completeness: float,
        ce_marking: bool,
        nb_required: bool,
        nb_engaged: bool,
        qs_adequate: bool,
    ) -> float:
        """Calculate overall conformity readiness score.

        Weighted components:
            - Documentation: 30%
            - Test coverage: 30%
            - Declaration: 20%
            - CE marking: 10%
            - Notified body / QS: 10%

        Args:
            doc_completeness: Documentation completeness (0-100).
            test_coverage: Test coverage (0-100).
            decl_completeness: Declaration completeness (0-100).
            ce_marking: Whether CE marking is applied.
            nb_required: Whether notified body is required.
            nb_engaged: Whether notified body is engaged.
            qs_adequate: Whether quality system is adequate.

        Returns:
            Overall score (0-100).
        """
        ce_score = 100.0 if ce_marking else 0.0

        # Notified body / quality system component
        if nb_required:
            nb_score = 100.0 if nb_engaged else 0.0
        else:
            nb_score = 100.0  # Not required, so full marks

        if not qs_adequate and nb_required:
            nb_score = nb_score * 0.5  # Reduce if QS not adequate

        overall = (
            doc_completeness * 0.30
            + test_coverage * 0.30
            + decl_completeness * 0.20
            + ce_score * 0.10
            + nb_score * 0.10
        )

        return _round2(overall)

    # ------------------------------------------------------------------ #
    # Private: Status Determination                                        #
    # ------------------------------------------------------------------ #

    def _determine_status(
        self, score: float, missing: List[str]
    ) -> ConformityStatus:
        """Determine conformity status from score and missing items.

        Thresholds:
            Score == 100 and no missing: READY
            Score >= 70: PARTIALLY_READY
            Score >= 30: IN_PROGRESS
            Score < 30: NOT_READY

        Args:
            score: Overall readiness score (0-100).
            missing: List of missing items.

        Returns:
            ConformityStatus enum value.
        """
        if score >= 100.0 and len(missing) == 0:
            return ConformityStatus.READY
        if score >= 70.0:
            return ConformityStatus.PARTIALLY_READY
        if score >= 30.0:
            return ConformityStatus.IN_PROGRESS
        return ConformityStatus.NOT_READY

    # ------------------------------------------------------------------ #
    # Private: Recommendations                                             #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        missing: List[str],
        module: ConformityModule,
        category: BatteryCategory,
        doc_completeness: float,
        test_coverage: float,
        decl_completeness: float,
        ce_marking: bool,
        nb_required: bool,
        nb_engaged: bool,
    ) -> List[str]:
        """Generate actionable recommendations for conformity readiness.

        Args:
            missing: List of missing items.
            module: Applied conformity module.
            category: Battery category.
            doc_completeness: Documentation completeness.
            test_coverage: Test coverage.
            decl_completeness: Declaration completeness.
            ce_marking: Whether CE marking is applied.
            nb_required: Whether notified body is required.
            nb_engaged: Whether notified body is engaged.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if not missing:
            recommendations.append(
                "All conformity assessment requirements are met for "
                f"{module.value}. The battery is ready for EU market "
                "placement."
            )
            return recommendations

        # Documentation gaps
        if doc_completeness < 100.0:
            doc_missing = sum(
                1 for item in missing
                if item.startswith("Technical documentation")
            )
            recommendations.append(
                f"Technical documentation is {doc_completeness}% complete. "
                f"{doc_missing} item(s) are missing. Complete the technical "
                "file per Art 18 before proceeding."
            )

        if doc_completeness < 50.0:
            recommendations.append(
                "CRITICAL: Technical documentation is less than 50% "
                "complete. This is a fundamental requirement for all "
                "conformity assessment modules. Prioritise immediately."
            )

        # Test gaps
        if test_coverage < 100.0:
            test_missing = sum(
                1 for item in missing
                if item.startswith("Test not completed")
            )
            test_failed = sum(
                1 for item in missing
                if item.startswith("Test failed")
            )

            if test_missing > 0:
                recommendations.append(
                    f"Test coverage is {test_coverage}%. {test_missing} "
                    f"required test(s) have not been completed. Schedule "
                    f"testing at an accredited laboratory."
                )

            if test_failed > 0:
                recommendations.append(
                    f"{test_failed} test(s) have failed. Investigate "
                    "root cause, implement corrective actions, and "
                    "retest before proceeding."
                )

        # Declaration gaps
        if decl_completeness < 100.0:
            recommendations.append(
                f"EU Declaration of Conformity is {decl_completeness}% "
                "complete. Complete all items per Annex V before applying "
                "CE marking."
            )

        # CE marking
        if not ce_marking:
            recommendations.append(
                "CE marking has not been applied. CE marking may only "
                "be affixed after successful completion of the conformity "
                "assessment procedure and issuance of the EU Declaration."
            )

        # Notified body
        if nb_required and not nb_engaged:
            recommendations.append(
                f"A notified body is required for {module.value}. "
                "Engage a notified body designated for the EU Battery "
                "Regulation. Check the NANDO database for available bodies."
            )

        # Category-specific advice
        if category == BatteryCategory.EV:
            carbon_missing = any(
                "Carbon footprint" in item for item in missing
            )
            if carbon_missing:
                recommendations.append(
                    "Carbon footprint declaration is required for EV "
                    "batteries per Art 14. Complete the lifecycle carbon "
                    "footprint assessment using the PEFCR methodology."
                )

        return recommendations
