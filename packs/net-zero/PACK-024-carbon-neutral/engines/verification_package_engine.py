# -*- coding: utf-8 -*-
"""
VerificationPackageEngine - PACK-024 Carbon Neutral Engine 8
=============================================================

10-component evidence assembly engine for carbon neutral verification
packages, covering footprint report, management plan, credit portfolio,
retirement certificates, reduction evidence, baseline verification,
claims documentation, quality assessment, temporal alignment, and
assurance statement.

This engine assembles all necessary evidence into a structured
verification package suitable for third-party assurance under
ISO 14068-1:2023 Section 11 and ISO 14064-3:2019.

Calculation Methodology:
    Package Completeness:
        Each component scored:
            COMPLETE:   Component present with all required elements.
            PARTIAL:    Component present with some gaps.
            MISSING:    Component not provided.
            NOT_REQUIRED: Not applicable to this claim type.

        completeness_score = complete_count / required_count * 100

    Verification Readiness:
        ready = completeness_score >= 80% AND all_critical_complete

    Evidence Quality:
        quality_score = sum(component_quality * component_weight) / total_weight
        Each component quality: 0-10 scale

    10 Components (ISO 14068-1:2023, Section 11):
        1. Footprint Report: Quantified GHG inventory
        2. Management Plan: Carbon management plan with targets
        3. Credit Portfolio: Details of carbon credits procured
        4. Retirement Certificates: Registry retirement confirmations
        5. Reduction Evidence: Evidence of emission reductions achieved
        6. Baseline Verification: Base year verification
        7. Claims Documentation: Claim substantiation documents
        8. Quality Assessment: Credit and data quality assessments
        9. Temporal Alignment: Vintage-footprint alignment evidence
       10. Assurance Statement: Third-party assurance/verification statement

Regulatory References:
    - ISO 14068-1:2023 - Section 11: Verification of carbon neutrality
    - ISO 14064-3:2019 - Specification for verification of GHG assertions
    - PAS 2060:2014 - Section 5.6: Verification
    - ISAE 3410 (2012) - Assurance engagements on GHG statements
    - IAF MD 6:2014 - Application of ISO 14065 for GHG validation/verification

Zero-Hallucination:
    - All 10 components from ISO 14068-1:2023 Section 11
    - Verification requirements from ISO 14064-3:2019
    - No LLM involvement in any calculation path
    - Deterministic scoring throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-024 Carbon Neutral
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ComponentId(str, Enum):
    """Verification package component identifiers.

    10 components per ISO 14068-1:2023 Section 11.
    """
    FOOTPRINT_REPORT = "footprint_report"
    MANAGEMENT_PLAN = "management_plan"
    CREDIT_PORTFOLIO = "credit_portfolio"
    RETIREMENT_CERTIFICATES = "retirement_certificates"
    REDUCTION_EVIDENCE = "reduction_evidence"
    BASELINE_VERIFICATION = "baseline_verification"
    CLAIMS_DOCUMENTATION = "claims_documentation"
    QUALITY_ASSESSMENT = "quality_assessment"
    TEMPORAL_ALIGNMENT = "temporal_alignment"
    ASSURANCE_STATEMENT = "assurance_statement"


class ComponentStatus(str, Enum):
    """Component status in the verification package.

    COMPLETE: All required elements present.
    PARTIAL: Some elements present, gaps identified.
    MISSING: Component not provided.
    NOT_REQUIRED: Not applicable to this claim type.
    """
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_REQUIRED = "not_required"


class AssuranceLevel(str, Enum):
    """Level of assurance per ISAE 3410.

    LIMITED: Limited assurance engagement.
    REASONABLE: Reasonable assurance engagement.
    NO_ASSURANCE: No third-party assurance.
    """
    LIMITED = "limited"
    REASONABLE = "reasonable"
    NO_ASSURANCE = "no_assurance"


class VerificationReadiness(str, Enum):
    """Verification package readiness.

    READY: Package is complete and ready for verification.
    CONDITIONALLY_READY: Minor gaps, may proceed with conditions.
    NOT_READY: Significant gaps, not ready for verification.
    """
    READY = "ready"
    CONDITIONALLY_READY = "conditionally_ready"
    NOT_READY = "not_ready"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Component definitions with requirements.
COMPONENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    ComponentId.FOOTPRINT_REPORT.value: {
        "name": "GHG Footprint Report",
        "description": "Quantified organisational GHG inventory per ISO 14064-1",
        "standard_ref": "ISO 14068-1:2023 Section 6, ISO 14064-1:2018",
        "is_critical": True,
        "weight": Decimal("0.15"),
        "required_elements": [
            "Organisational boundary",
            "Operational boundary (scopes)",
            "Quantified emissions by scope",
            "Emission factors and sources",
            "Data quality assessment",
            "Base year definition",
            "Methodology description",
        ],
    },
    ComponentId.MANAGEMENT_PLAN.value: {
        "name": "Carbon Management Plan",
        "description": "Reduction-first management plan per ISO 14068-1 Section 9",
        "standard_ref": "ISO 14068-1:2023 Section 9",
        "is_critical": True,
        "weight": Decimal("0.12"),
        "required_elements": [
            "Reduction targets",
            "Mitigation hierarchy evidence",
            "Timeline and milestones",
            "Measure descriptions",
            "Financial projections",
        ],
    },
    ComponentId.CREDIT_PORTFOLIO.value: {
        "name": "Credit Portfolio Documentation",
        "description": "Details of all carbon credits procured",
        "standard_ref": "ISO 14068-1:2023 Section 8",
        "is_critical": True,
        "weight": Decimal("0.12"),
        "required_elements": [
            "Credit standard and methodology",
            "Project descriptions",
            "Quantities and vintage years",
            "Quality assessment scores",
            "Additionality evidence",
            "Permanence assessment",
        ],
    },
    ComponentId.RETIREMENT_CERTIFICATES.value: {
        "name": "Retirement Certificates",
        "description": "Registry retirement confirmations for all credits",
        "standard_ref": "ISO 14068-1:2023 Section 8.5, PAS 2060:2014 Section 5.4.3",
        "is_critical": True,
        "weight": Decimal("0.12"),
        "required_elements": [
            "Registry confirmation references",
            "Serial number tracking",
            "Beneficiary designation",
            "Retirement dates",
            "Quantities confirmed",
        ],
    },
    ComponentId.REDUCTION_EVIDENCE.value: {
        "name": "Reduction Evidence",
        "description": "Evidence of emission reductions achieved",
        "standard_ref": "ISO 14068-1:2023 Section 9.4",
        "is_critical": True,
        "weight": Decimal("0.10"),
        "required_elements": [
            "Year-on-year emissions comparison",
            "Measure implementation evidence",
            "Reduction quantification",
            "Progress against targets",
        ],
    },
    ComponentId.BASELINE_VERIFICATION.value: {
        "name": "Baseline Verification",
        "description": "Base year emissions verification",
        "standard_ref": "ISO 14068-1:2023 Section 6.4, GHG Protocol Ch. 5",
        "is_critical": False,
        "weight": Decimal("0.08"),
        "required_elements": [
            "Base year selection justification",
            "Base year emissions quantification",
            "Recalculation policy",
            "Structural changes adjustment",
        ],
    },
    ComponentId.CLAIMS_DOCUMENTATION.value: {
        "name": "Claims Documentation",
        "description": "Claim substantiation documents and QES",
        "standard_ref": "ISO 14068-1:2023 Section 10, PAS 2060:2014 Section 5.5",
        "is_critical": True,
        "weight": Decimal("0.08"),
        "required_elements": [
            "Qualifying Explanatory Statement (QES)",
            "Claim wording and scope",
            "Disclosure documents",
            "Standard compliance mapping",
        ],
    },
    ComponentId.QUALITY_ASSESSMENT.value: {
        "name": "Quality Assessment",
        "description": "Credit and data quality assessments",
        "standard_ref": "ICVCM CCP V1.0, ISO 14064-1:2018 Annex A",
        "is_critical": False,
        "weight": Decimal("0.08"),
        "required_elements": [
            "Credit quality scoring methodology",
            "Per-credit quality assessments",
            "Data quality evaluation",
            "Uncertainty assessment",
        ],
    },
    ComponentId.TEMPORAL_ALIGNMENT.value: {
        "name": "Temporal Alignment Evidence",
        "description": "Vintage-footprint year alignment documentation",
        "standard_ref": "ISO 14068-1:2023 Section 8.3",
        "is_critical": True,
        "weight": Decimal("0.07"),
        "required_elements": [
            "Vintage year documentation",
            "Footprint period coverage",
            "Temporal match validation",
            "Carryforward documentation",
        ],
    },
    ComponentId.ASSURANCE_STATEMENT.value: {
        "name": "Assurance Statement",
        "description": "Third-party verification/assurance statement",
        "standard_ref": "ISO 14068-1:2023 Section 11, ISO 14064-3:2019, ISAE 3410",
        "is_critical": False,
        "weight": Decimal("0.08"),
        "required_elements": [
            "Verification body identification",
            "Scope of verification",
            "Assurance level",
            "Opinion/conclusion",
            "Methodology used",
        ],
    },
}

# Minimum completeness for verification readiness.
MIN_COMPLETENESS_READY: Decimal = Decimal("80")
MIN_COMPLETENESS_CONDITIONAL: Decimal = Decimal("60")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class ComponentInput(BaseModel):
    """Input for a single verification package component.

    Attributes:
        component_id: Component identifier.
        status: Component status.
        quality_score: Quality of the component (0-10).
        elements_present: Which required elements are present.
        elements_missing: Which required elements are missing.
        document_references: References to supporting documents.
        last_updated: Date of last update.
        reviewer: Reviewer name.
        notes: Additional notes.
    """
    component_id: str = Field(..., description="Component ID")
    status: str = Field(
        default=ComponentStatus.MISSING.value, description="Status"
    )
    quality_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("10"),
        description="Quality (0-10)"
    )
    elements_present: List[str] = Field(
        default_factory=list, description="Elements present"
    )
    elements_missing: List[str] = Field(
        default_factory=list, description="Elements missing"
    )
    document_references: List[str] = Field(
        default_factory=list, description="Document references"
    )
    last_updated: Optional[str] = Field(default=None, description="Last updated")
    reviewer: str = Field(default="", description="Reviewer")
    notes: str = Field(default="", description="Notes")

    @field_validator("component_id")
    @classmethod
    def validate_component(cls, v: str) -> str:
        valid = {c.value for c in ComponentId}
        if v not in valid:
            raise ValueError(f"Unknown component '{v}'.")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {s.value for s in ComponentStatus}
        if v not in valid:
            raise ValueError(f"Unknown status '{v}'.")
        return v


class VerificationPackageInput(BaseModel):
    """Complete input for verification package assembly.

    Attributes:
        entity_name: Reporting entity name.
        assessment_year: Year of assessment.
        claim_type: Type of claim being verified.
        assurance_level: Desired assurance level.
        components: Component input data.
        verifier_name: Third-party verifier name.
        target_standard: Target standard.
        include_gap_analysis: Whether to include gap analysis.
        include_recommendations: Whether to include recommendations.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    assessment_year: int = Field(
        ..., ge=2015, le=2060, description="Assessment year"
    )
    claim_type: str = Field(
        default="carbon_neutral", description="Claim type"
    )
    assurance_level: str = Field(
        default=AssuranceLevel.LIMITED.value, description="Assurance level"
    )
    components: List[ComponentInput] = Field(
        default_factory=list, description="Components"
    )
    verifier_name: str = Field(default="", description="Verifier name")
    target_standard: str = Field(
        default="iso_14068_1", description="Target standard"
    )
    include_gap_analysis: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class ComponentResult(BaseModel):
    """Assessment result for a single component.

    Attributes:
        component_id: Component identifier.
        component_name: Human-readable name.
        description: Component description.
        standard_ref: Standard reference.
        is_critical: Whether critical for verification.
        status: Component status.
        quality_score: Quality score (0-10).
        weight: Component weight.
        weighted_score: quality * weight.
        required_elements_count: Total required elements.
        elements_present_count: Elements present.
        elements_missing_count: Elements missing.
        elements_missing: List of missing elements.
        completeness_pct: Completeness percentage.
        document_count: Number of supporting documents.
        issues: Issues identified.
        recommendations: Recommendations.
    """
    component_id: str = Field(default="")
    component_name: str = Field(default="")
    description: str = Field(default="")
    standard_ref: str = Field(default="")
    is_critical: bool = Field(default=False)
    status: str = Field(default=ComponentStatus.MISSING.value)
    quality_score: Decimal = Field(default=Decimal("0"))
    weight: Decimal = Field(default=Decimal("0"))
    weighted_score: Decimal = Field(default=Decimal("0"))
    required_elements_count: int = Field(default=0)
    elements_present_count: int = Field(default=0)
    elements_missing_count: int = Field(default=0)
    elements_missing: List[str] = Field(default_factory=list)
    completeness_pct: Decimal = Field(default=Decimal("0"))
    document_count: int = Field(default=0)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class GapAnalysis(BaseModel):
    """Gap analysis for the verification package.

    Attributes:
        critical_gaps: Gaps in critical components.
        non_critical_gaps: Gaps in non-critical components.
        element_gaps: Total missing elements.
        estimated_effort_days: Estimated effort to close gaps.
        priority_actions: Prioritised actions to close gaps.
    """
    critical_gaps: List[str] = Field(default_factory=list)
    non_critical_gaps: List[str] = Field(default_factory=list)
    element_gaps: int = Field(default=0)
    estimated_effort_days: int = Field(default=0)
    priority_actions: List[str] = Field(default_factory=list)


class VerificationPackageResult(BaseModel):
    """Complete verification package result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        assessment_year: Assessment year.
        claim_type: Claim type.
        assurance_level: Assurance level.
        component_results: Per-component results.
        gap_analysis: Gap analysis.
        completeness_pct: Overall completeness.
        quality_weighted_score: Quality-weighted score.
        components_total: Total components.
        components_complete: Complete components.
        components_partial: Partial components.
        components_missing: Missing components.
        all_critical_complete: Whether all critical components complete.
        readiness: Verification readiness.
        package_hash: SHA-256 hash of the package.
        recommendations: Overall recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    assessment_year: int = Field(default=0)
    claim_type: str = Field(default="")
    assurance_level: str = Field(default="")
    component_results: List[ComponentResult] = Field(default_factory=list)
    gap_analysis: Optional[GapAnalysis] = Field(default=None)
    completeness_pct: Decimal = Field(default=Decimal("0"))
    quality_weighted_score: Decimal = Field(default=Decimal("0"))
    components_total: int = Field(default=0)
    components_complete: int = Field(default=0)
    components_partial: int = Field(default=0)
    components_missing: int = Field(default=0)
    all_critical_complete: bool = Field(default=False)
    readiness: str = Field(default=VerificationReadiness.NOT_READY.value)
    package_hash: str = Field(default="")
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VerificationPackageEngine:
    """10-component verification package assembly engine.

    Assembles and assesses verification evidence packages for carbon
    neutral claims per ISO 14068-1:2023 Section 11.

    Usage::

        engine = VerificationPackageEngine()
        result = engine.assemble(input_data)
        print(f"Readiness: {result.readiness}")
        print(f"Completeness: {result.completeness_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        logger.info("VerificationPackageEngine v%s initialised", self.engine_version)

    def assemble(
        self, data: VerificationPackageInput,
    ) -> VerificationPackageResult:
        """Assemble and assess verification package.

        Args:
            data: Validated package input.

        Returns:
            VerificationPackageResult with assessment.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        errors: List[str] = []

        # Build input lookup
        input_map = {c.component_id: c for c in data.components}

        # Step 1: Assess each component
        component_results: List[ComponentResult] = []
        for comp_id in ComponentId:
            comp_def = COMPONENT_DEFINITIONS[comp_id.value]
            inp = input_map.get(comp_id.value)
            cr = self._assess_component(comp_def, inp, comp_id.value)
            component_results.append(cr)

        # Step 2: Calculate totals
        required = [c for c in component_results if c.status != ComponentStatus.NOT_REQUIRED.value]
        complete = sum(1 for c in required if c.status == ComponentStatus.COMPLETE.value)
        partial = sum(1 for c in required if c.status == ComponentStatus.PARTIAL.value)
        missing = sum(1 for c in required if c.status == ComponentStatus.MISSING.value)
        total_required = len(required)

        completeness = Decimal("0")
        if total_required > 0:
            score_sum = Decimal("0")
            for c in required:
                if c.status == ComponentStatus.COMPLETE.value:
                    score_sum += Decimal("1")
                elif c.status == ComponentStatus.PARTIAL.value:
                    score_sum += Decimal("0.5")
            completeness = _safe_pct(score_sum, _decimal(total_required))

        # Quality weighted score
        total_weight = sum((c.weight for c in component_results), Decimal("0"))
        quality_weighted = Decimal("0")
        if total_weight > Decimal("0"):
            quality_weighted = sum(
                (c.weighted_score for c in component_results), Decimal("0")
            ) / total_weight * Decimal("10")

        # Critical components
        all_critical = all(
            c.status == ComponentStatus.COMPLETE.value
            for c in component_results if c.is_critical
        )

        # Readiness
        if completeness >= MIN_COMPLETENESS_READY and all_critical:
            readiness = VerificationReadiness.READY.value
        elif completeness >= MIN_COMPLETENESS_CONDITIONAL:
            readiness = VerificationReadiness.CONDITIONALLY_READY.value
        else:
            readiness = VerificationReadiness.NOT_READY.value

        # Step 3: Gap analysis
        gap: Optional[GapAnalysis] = None
        if data.include_gap_analysis:
            gap = self._gap_analysis(component_results)

        # Step 4: Package hash
        pkg_data = {
            c.component_id: {
                "status": c.status,
                "quality": str(c.quality_score),
                "elements_present": c.elements_present_count,
            }
            for c in component_results
        }
        package_hash = hashlib.sha256(
            json.dumps(pkg_data, sort_keys=True).encode()
        ).hexdigest()

        # Step 5: Recommendations
        recommendations: List[str] = []
        if data.include_recommendations:
            for c in component_results:
                recommendations.extend(c.recommendations)

        if not all_critical:
            warnings.append("Not all critical components are complete.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = VerificationPackageResult(
            entity_name=data.entity_name,
            assessment_year=data.assessment_year,
            claim_type=data.claim_type,
            assurance_level=data.assurance_level,
            component_results=component_results,
            gap_analysis=gap,
            completeness_pct=_round_val(completeness, 2),
            quality_weighted_score=_round_val(quality_weighted, 2),
            components_total=total_required,
            components_complete=complete,
            components_partial=partial,
            components_missing=missing,
            all_critical_complete=all_critical,
            readiness=readiness,
            package_hash=package_hash,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_component(
        self,
        comp_def: Dict[str, Any],
        inp: Optional[ComponentInput],
        comp_id: str,
    ) -> ComponentResult:
        """Assess a single component."""
        name = comp_def["name"]
        required_elements = comp_def["required_elements"]
        is_critical = comp_def["is_critical"]
        weight = comp_def["weight"]

        if inp:
            status = inp.status
            quality = inp.quality_score
            present = len(inp.elements_present)
            elements_missing = inp.elements_missing or [
                e for e in required_elements
                if e not in inp.elements_present
            ]
            doc_count = len(inp.document_references)
        else:
            status = ComponentStatus.MISSING.value
            quality = Decimal("0")
            present = 0
            elements_missing = list(required_elements)
            doc_count = 0

        total_req = len(required_elements)
        missing_count = len(elements_missing)
        completeness = _safe_pct(_decimal(present), _decimal(total_req))
        weighted = quality * weight

        issues: List[str] = []
        recs: List[str] = []

        if status == ComponentStatus.MISSING.value:
            issues.append(f"Component '{name}' is missing from the package.")
            if is_critical:
                recs.append(f"CRITICAL: Provide '{name}' -- required for verification.")
            else:
                recs.append(f"Provide '{name}' to improve package completeness.")
        elif status == ComponentStatus.PARTIAL.value:
            issues.append(
                f"Component '{name}' is incomplete. Missing {missing_count} elements."
            )
            recs.append(f"Complete missing elements for '{name}': {', '.join(elements_missing[:3])}")

        return ComponentResult(
            component_id=comp_id,
            component_name=name,
            description=comp_def["description"],
            standard_ref=comp_def["standard_ref"],
            is_critical=is_critical,
            status=status,
            quality_score=quality,
            weight=weight,
            weighted_score=_round_val(weighted, 4),
            required_elements_count=total_req,
            elements_present_count=present,
            elements_missing_count=missing_count,
            elements_missing=elements_missing,
            completeness_pct=_round_val(completeness, 2),
            document_count=doc_count,
            issues=issues,
            recommendations=recs,
        )

    def _gap_analysis(
        self, components: List[ComponentResult],
    ) -> GapAnalysis:
        """Perform gap analysis on the package."""
        critical: List[str] = []
        non_critical: List[str] = []
        total_missing = 0
        effort = 0

        for c in components:
            if c.status in (ComponentStatus.MISSING.value, ComponentStatus.PARTIAL.value):
                gap_desc = f"{c.component_name}: {c.elements_missing_count} elements missing"
                if c.is_critical:
                    critical.append(gap_desc)
                    effort += c.elements_missing_count * 2
                else:
                    non_critical.append(gap_desc)
                    effort += c.elements_missing_count
                total_missing += c.elements_missing_count

        actions: List[str] = []
        for g in critical:
            actions.append(f"PRIORITY: {g}")
        for g in non_critical[:5]:
            actions.append(g)

        return GapAnalysis(
            critical_gaps=critical,
            non_critical_gaps=non_critical,
            element_gaps=total_missing,
            estimated_effort_days=max(1, effort),
            priority_actions=actions,
        )
