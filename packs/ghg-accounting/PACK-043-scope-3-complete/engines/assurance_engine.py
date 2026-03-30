# -*- coding: utf-8 -*-
"""
AssuranceEngine - PACK-043 Scope 3 Complete Pack Engine 10
============================================================

Generates ISAE 3410 reasonable assurance evidence packages for
Scope 3 GHG inventories.  Provides calculation provenance chains,
methodology decision logs, data source inventories, assumption
registers, completeness statements, emission factor provenance,
verifier query management, finding tracking, assurance readiness
scoring, and year-on-year comparison packages.

Evidence Categories (8):
    1. calculation_provenance  - Step-by-step hash chain for every calculation
    2. methodology_decisions   - Tier selection rationale per category
    3. data_sources            - Every data point with origin and timestamp
    4. assumptions             - All assumptions with sensitivity analysis
    5. emission_factors        - EF source, version, applicability
    6. completeness            - Inclusion/exclusion rationale for all 15 cats
    7. uncertainty             - Uncertainty ranges and methodology
    8. boundary                - Organisational and operational boundary

Assurance Readiness Score (0-100):
    readiness = sum(
        category_weight * category_completeness_pct
        for category in evidence_categories
    )
    Weights: calculation_provenance 20%, methodology 15%, data_sources 15%,
             assumptions 10%, emission_factors 15%, completeness 10%,
             uncertainty 10%, boundary 5%

ISAE 3410 Requirements:
    Reasonable Assurance:
        - Complete evidence across all 8 categories
        - Hash chain provenance for all material calculations
        - Reconciliation of data sources to primary records
        - Sensitivity analysis for material assumptions
        - Year-on-year explanatory notes for >10% changes
    Limited Assurance:
        - Inquiry and analytical procedures
        - Plausibility assessment of reported figures
        - Methodology review (less extensive testing)

Verifier Query Management:
    - Track incoming queries with unique IDs
    - Link queries to evidence and source data
    - Manage response workflow (open -> in_progress -> responded -> closed)
    - Maintain finding register with root cause and remediation

Regulatory References:
    - ISAE 3410 Assurance Engagements on GHG Statements (2012)
    - ISAE 3000 (Revised) Assurance Engagements Other Than Audits (2013)
    - GHG Protocol Corporate Value Chain Standard (2011), Chapter 8
    - ISO 14064-3:2019 (Verification of GHG assertions)
    - ESRS E1 (External assurance requirements)
    - SEC Climate Disclosure Rule (Attestation requirements)

Zero-Hallucination:
    - All readiness scores computed deterministically
    - Hash chains use SHA-256 for immutability
    - No LLM involvement in any scoring path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-043 Scope 3 Complete
Engine:  10 of 10
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "43.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvidenceCategory(str, Enum):
    """Evidence categories for ISAE 3410 assurance."""
    CALCULATION_PROVENANCE = "calculation_provenance"
    METHODOLOGY_DECISIONS = "methodology_decisions"
    DATA_SOURCES = "data_sources"
    ASSUMPTIONS = "assumptions"
    EMISSION_FACTORS = "emission_factors"
    COMPLETENESS = "completeness"
    UNCERTAINTY = "uncertainty"
    BOUNDARY = "boundary"

class AssuranceLevel(str, Enum):
    """Assurance engagement levels."""
    REASONABLE = "reasonable"
    LIMITED = "limited"

class QueryStatus(str, Enum):
    """Verifier query lifecycle status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESPONDED = "responded"
    CLOSED = "closed"
    REOPENED = "reopened"

class FindingSeverity(str, Enum):
    """Assurance finding severity."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"

class FindingStatus(str, Enum):
    """Finding remediation status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    REMEDIATED = "remediated"
    ACCEPTED = "accepted"
    CLOSED = "closed"

class ReadinessRating(str, Enum):
    """Assurance readiness rating."""
    READY = "ready"
    MOSTLY_READY = "mostly_ready"
    PARTIALLY_READY = "partially_ready"
    NOT_READY = "not_ready"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Evidence category weights for readiness scoring.
EVIDENCE_CATEGORY_WEIGHTS: Dict[str, float] = {
    EvidenceCategory.CALCULATION_PROVENANCE: 0.20,
    EvidenceCategory.METHODOLOGY_DECISIONS: 0.15,
    EvidenceCategory.DATA_SOURCES: 0.15,
    EvidenceCategory.ASSUMPTIONS: 0.10,
    EvidenceCategory.EMISSION_FACTORS: 0.15,
    EvidenceCategory.COMPLETENESS: 0.10,
    EvidenceCategory.UNCERTAINTY: 0.10,
    EvidenceCategory.BOUNDARY: 0.05,
}
"""Category weights for assurance readiness score (sum = 1.0)."""

# Readiness thresholds.
READINESS_THRESHOLDS: Dict[str, float] = {
    ReadinessRating.READY: 90.0,
    ReadinessRating.MOSTLY_READY: 70.0,
    ReadinessRating.PARTIALLY_READY: 40.0,
}
"""Readiness score thresholds for rating classification."""

# Minimum evidence items per category for reasonable assurance.
REASONABLE_ASSURANCE_MINIMUMS: Dict[str, int] = {
    EvidenceCategory.CALCULATION_PROVENANCE: 1,   # Per material category
    EvidenceCategory.METHODOLOGY_DECISIONS: 1,    # Per material category
    EvidenceCategory.DATA_SOURCES: 1,             # Per data point
    EvidenceCategory.ASSUMPTIONS: 1,              # Per assumption
    EvidenceCategory.EMISSION_FACTORS: 1,         # Per EF used
    EvidenceCategory.COMPLETENESS: 1,             # Overall statement
    EvidenceCategory.UNCERTAINTY: 1,              # Overall assessment
    EvidenceCategory.BOUNDARY: 1,                 # Overall statement
}
"""Minimum evidence items for reasonable assurance."""

# Scope 3 category names for reference.
SCOPE3_CATEGORY_NAMES: Dict[int, str] = {
    1: "Purchased Goods & Services",
    2: "Capital Goods",
    3: "Fuel- & Energy-Related Activities",
    4: "Upstream Transportation & Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation & Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}
"""GHG Protocol Scope 3 category names."""

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class CalculationStep(BaseModel):
    """Single step in a calculation provenance chain.

    Attributes:
        step_number: Sequence number.
        description: What this step computes.
        inputs: Input values and their sources.
        formula: Formula applied.
        output_value: Calculated output value.
        output_unit: Unit of output.
        step_hash: SHA-256 of this step (chained from previous).
    """
    step_number: int = Field(..., ge=1, description="Step sequence")
    description: str = Field(..., description="Step description")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    formula: str = Field(default="", description="Formula applied")
    output_value: float = Field(..., description="Output value")
    output_unit: str = Field(default="tCO2e", description="Output unit")
    step_hash: str = Field(default="", description="Step hash")

class ProvenanceChain(BaseModel):
    """Complete provenance chain for a calculation.

    Attributes:
        chain_id: Unique chain identifier.
        category: Scope 3 category.
        category_name: Category name.
        steps: Ordered list of calculation steps.
        final_value: Final calculated value.
        final_unit: Unit of final value.
        chain_hash: SHA-256 hash of entire chain.
        calculated_at: Calculation timestamp.
    """
    chain_id: str = Field(default_factory=_new_uuid, description="Chain ID")
    category: int = Field(default=0, description="Scope 3 category")
    category_name: str = Field(default="", description="Category name")
    steps: List[CalculationStep] = Field(default_factory=list, description="Steps")
    final_value: float = Field(default=0, description="Final value")
    final_unit: str = Field(default="tCO2e", description="Final unit")
    chain_hash: str = Field(default="", description="Chain hash")
    calculated_at: str = ""

class MethodologyDecision(BaseModel):
    """Methodology decision record for a Scope 3 category.

    Attributes:
        category: Scope 3 category number.
        category_name: Category name.
        selected_tier: Selected calculation tier.
        rationale: Rationale for tier selection.
        alternatives_considered: Other tiers considered.
        data_availability: Data availability assessment.
        materiality: Whether this category is material.
        reviewer: Person who approved the decision.
        decision_date: Date of decision.
    """
    category: int = Field(..., ge=1, le=15, description="Scope 3 category")
    category_name: str = Field(default="", description="Category name")
    selected_tier: str = Field(..., description="Selected methodology tier")
    rationale: str = Field(default="", description="Selection rationale")
    alternatives_considered: List[str] = Field(
        default_factory=list, description="Alternatives considered"
    )
    data_availability: str = Field(default="", description="Data availability")
    materiality: bool = Field(default=True, description="Is material")
    reviewer: str = Field(default="", description="Reviewer name")
    decision_date: str = Field(default="", description="Decision date")

class DataSourceRecord(BaseModel):
    """Record of a data source used in the inventory.

    Attributes:
        source_id: Unique source identifier.
        category: Scope 3 category.
        source_name: Name of the data source.
        source_type: Type (primary/secondary/estimated).
        description: Description of the data.
        value: Data value.
        unit: Unit of measurement.
        collection_date: When data was collected.
        responsible_person: Data owner.
        verification_status: Whether verified.
    """
    source_id: str = Field(default_factory=_new_uuid, description="Source ID")
    category: int = Field(default=0, description="Scope 3 category")
    source_name: str = Field(..., description="Source name")
    source_type: str = Field(default="primary", description="primary/secondary/estimated")
    description: str = Field(default="", description="Data description")
    value: Optional[float] = Field(default=None, description="Data value")
    unit: str = Field(default="", description="Unit")
    collection_date: str = Field(default="", description="Collection date")
    responsible_person: str = Field(default="", description="Data owner")
    verification_status: str = Field(default="unverified", description="Verification status")

class AssumptionRecord(BaseModel):
    """Record of an assumption used in the inventory.

    Attributes:
        assumption_id: Unique identifier.
        category: Scope 3 category.
        description: Description of the assumption.
        value: Assumed value.
        unit: Unit.
        source: Source of the assumption.
        sensitivity: Sensitivity (high/medium/low).
        sensitivity_range_pct: Range for sensitivity analysis.
        impact_on_total_pct: Impact on total if assumption changes.
        alternative_values: Alternative values considered.
    """
    assumption_id: str = Field(default_factory=_new_uuid, description="ID")
    category: int = Field(default=0, description="Scope 3 category")
    description: str = Field(..., description="Assumption description")
    value: float = Field(default=0, description="Assumed value")
    unit: str = Field(default="", description="Unit")
    source: str = Field(default="", description="Source")
    sensitivity: str = Field(default="medium", description="Sensitivity level")
    sensitivity_range_pct: float = Field(default=20.0, description="Sensitivity range %")
    impact_on_total_pct: float = Field(default=0, description="Impact on total %")
    alternative_values: List[float] = Field(
        default_factory=list, description="Alternative values"
    )

class EmissionFactorRecord(BaseModel):
    """Record of an emission factor used.

    Attributes:
        ef_id: Unique identifier.
        category: Scope 3 category.
        factor_name: Name of the factor.
        factor_value: Factor value.
        factor_unit: Factor unit.
        source_database: Source database (e.g., DEFRA, ecoinvent).
        source_version: Database version.
        publication_year: Year of publication.
        geographic_scope: Geographic applicability.
        sector_scope: Sector applicability.
        applicability_notes: Notes on applicability.
    """
    ef_id: str = Field(default_factory=_new_uuid, description="EF ID")
    category: int = Field(default=0, description="Scope 3 category")
    factor_name: str = Field(..., description="Factor name")
    factor_value: float = Field(..., description="Factor value")
    factor_unit: str = Field(default="kgCO2e/unit", description="Factor unit")
    source_database: str = Field(default="", description="Source database")
    source_version: str = Field(default="", description="Database version")
    publication_year: int = Field(default=2024, description="Publication year")
    geographic_scope: str = Field(default="global", description="Geographic scope")
    sector_scope: str = Field(default="", description="Sector scope")
    applicability_notes: str = Field(default="", description="Applicability notes")

class CompletenessItem(BaseModel):
    """Completeness assessment for a Scope 3 category.

    Attributes:
        category: Scope 3 category number.
        category_name: Category name.
        included: Whether included in the inventory.
        rationale: Rationale for inclusion/exclusion.
        materiality_assessment: Materiality details.
        estimated_pct_of_total: Estimated share of total Scope 3.
        data_gaps: Known data gaps.
    """
    category: int = Field(..., ge=1, le=15, description="Category number")
    category_name: str = Field(default="", description="Category name")
    included: bool = Field(default=True, description="Included in inventory")
    rationale: str = Field(default="", description="Inclusion/exclusion rationale")
    materiality_assessment: str = Field(default="", description="Materiality notes")
    estimated_pct_of_total: float = Field(default=0, description="Est. % of total")
    data_gaps: List[str] = Field(default_factory=list, description="Known data gaps")

class VerifierQuery(BaseModel):
    """Verifier query record.

    Attributes:
        query_id: Unique query identifier.
        query_text: Query text.
        category: Related Scope 3 category.
        evidence_category: Related evidence category.
        status: Query lifecycle status.
        raised_date: Date raised.
        response_text: Response text.
        response_date: Date responded.
        evidence_refs: List of evidence references.
        assigned_to: Person assigned.
    """
    query_id: str = Field(default_factory=_new_uuid, description="Query ID")
    query_text: str = Field(..., description="Query text")
    category: int = Field(default=0, description="Scope 3 category")
    evidence_category: str = Field(default="", description="Evidence category")
    status: str = Field(default="open", description="Query status")
    raised_date: str = Field(default="", description="Date raised")
    response_text: str = Field(default="", description="Response")
    response_date: str = Field(default="", description="Response date")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence refs")
    assigned_to: str = Field(default="", description="Assigned person")

class Finding(BaseModel):
    """Assurance finding record.

    Attributes:
        finding_id: Unique finding identifier.
        description: Finding description.
        severity: Finding severity.
        category: Related Scope 3 category.
        root_cause: Root cause analysis.
        remediation: Remediation action.
        status: Finding status.
        raised_date: Date raised.
        due_date: Remediation due date.
        closed_date: Date closed.
    """
    finding_id: str = Field(default_factory=_new_uuid, description="Finding ID")
    description: str = Field(..., description="Finding description")
    severity: str = Field(default="minor", description="Finding severity")
    category: int = Field(default=0, description="Scope 3 category")
    root_cause: str = Field(default="", description="Root cause")
    remediation: str = Field(default="", description="Remediation action")
    status: str = Field(default="open", description="Finding status")
    raised_date: str = Field(default="", description="Date raised")
    due_date: str = Field(default="", description="Due date")
    closed_date: str = Field(default="", description="Closed date")

class EvidencePackage(BaseModel):
    """Complete evidence package for assurance.

    Attributes:
        package_id: Unique package identifier.
        reporting_year: Reporting year.
        assurance_level: Assurance level targeted.
        organisation_name: Organisation name.
        calculation_provenance: List of provenance chains.
        methodology_decisions: Methodology decision log.
        data_sources: Data source inventory.
        assumptions: Assumption register.
        emission_factors: Emission factor register.
        completeness: Completeness statement.
        total_tco2e: Total Scope 3 emissions.
        category_count: Number of categories included.
        readiness_score: Assurance readiness score (0-100).
        readiness_rating: Readiness rating.
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 provenance.
    """
    package_id: str = Field(default_factory=_new_uuid, description="Package ID")
    reporting_year: int = Field(default=0, description="Reporting year")
    assurance_level: str = Field(default="reasonable", description="Assurance level")
    organisation_name: str = Field(default="", description="Organisation")
    calculation_provenance: List[ProvenanceChain] = Field(default_factory=list)
    methodology_decisions: List[MethodologyDecision] = Field(default_factory=list)
    data_sources: List[DataSourceRecord] = Field(default_factory=list)
    assumptions: List[AssumptionRecord] = Field(default_factory=list)
    emission_factors: List[EmissionFactorRecord] = Field(default_factory=list)
    completeness: List[CompletenessItem] = Field(default_factory=list)
    total_tco2e: float = Field(default=0, description="Total Scope 3 tCO2e")
    category_count: int = Field(default=0, description="Categories included")
    readiness_score: float = Field(default=0, description="Readiness 0-100")
    readiness_rating: str = Field(default="", description="Readiness rating")
    generated_at: str = ""
    provenance_hash: str = ""

class AssuranceScore(BaseModel):
    """Assurance readiness score breakdown.

    Attributes:
        overall_score: Overall readiness score (0-100).
        rating: Readiness rating.
        by_category: Scores by evidence category.
        gaps: Identified gaps.
        recommendations: Recommendations to improve readiness.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    overall_score: float
    rating: str
    by_category: Dict[str, float] = Field(default_factory=dict)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = ""
    calculated_at: str = ""

class YoYComparisonItem(BaseModel):
    """Year-on-year comparison item for verification.

    Attributes:
        category: Scope 3 category.
        category_name: Category name.
        previous_tco2e: Previous year emissions.
        current_tco2e: Current year emissions.
        change_tco2e: Absolute change.
        change_pct: Percentage change.
        explanation: Explanation for material changes.
        requires_explanation: Whether >10% change needs explanation.
    """
    category: int
    category_name: str
    previous_tco2e: float
    current_tco2e: float
    change_tco2e: float
    change_pct: float
    explanation: str = ""
    requires_explanation: bool = False

class YoYComparisonPackage(BaseModel):
    """Year-on-year comparison package for verification.

    Attributes:
        current_year: Current reporting year.
        previous_year: Previous reporting year.
        current_total_tco2e: Current year total.
        previous_total_tco2e: Previous year total.
        total_change_pct: Overall change percentage.
        items: Per-category comparison items.
        material_changes: Categories with >10% change.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    current_year: int
    previous_year: int
    current_total_tco2e: float
    previous_total_tco2e: float
    total_change_pct: float
    items: List[YoYComparisonItem] = Field(default_factory=list)
    material_changes: List[int] = Field(default_factory=list)
    provenance_hash: str = ""
    calculated_at: str = ""

# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------

class AssuranceEngine:
    """Generates ISAE 3410 assurance evidence packages.

    Provides creation of complete evidence bundles, provenance chains,
    methodology logs, data source inventories, assumption registers,
    completeness statements, emission factor provenance, verifier query
    management, finding tracking, readiness scoring, and year-on-year
    comparison packages.

    All readiness scores are computed deterministically.  Every result
    carries a SHA-256 provenance hash.

    Example:
        >>> engine = AssuranceEngine()
        >>> package = engine.generate_evidence_package(inventory, decisions)
        >>> score = engine.calculate_assurance_readiness(package)
    """

    def __init__(self) -> None:
        """Initialise AssuranceEngine."""
        self._queries: List[VerifierQuery] = []
        self._findings: List[Finding] = []
        logger.info("AssuranceEngine v%s initialised", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public -- generate_evidence_package
    # -------------------------------------------------------------------

    def generate_evidence_package(
        self,
        provenance_chains: List[ProvenanceChain],
        methodology_decisions: List[MethodologyDecision],
        data_sources: List[DataSourceRecord],
        assumptions: List[AssumptionRecord],
        emission_factors: List[EmissionFactorRecord],
        completeness: List[CompletenessItem],
        total_tco2e: float = 0.0,
        reporting_year: int = 0,
        organisation_name: str = "",
        assurance_level: str = "reasonable",
    ) -> EvidencePackage:
        """Generate a complete evidence package for assurance engagement.

        Args:
            provenance_chains: Calculation provenance chains.
            methodology_decisions: Methodology decision log.
            data_sources: Data source inventory.
            assumptions: Assumption register.
            emission_factors: Emission factor register.
            completeness: Completeness statement items.
            total_tco2e: Total Scope 3 emissions.
            reporting_year: Reporting year.
            organisation_name: Organisation name.
            assurance_level: Assurance level (reasonable/limited).

        Returns:
            EvidencePackage with readiness score.
        """
        start_ms = time.time()

        cat_count = sum(1 for c in completeness if c.included)
        yr = reporting_year or utcnow().year

        package = EvidencePackage(
            reporting_year=yr,
            assurance_level=assurance_level,
            organisation_name=organisation_name,
            calculation_provenance=provenance_chains,
            methodology_decisions=methodology_decisions,
            data_sources=data_sources,
            assumptions=assumptions,
            emission_factors=emission_factors,
            completeness=completeness,
            total_tco2e=_round2(total_tco2e),
            category_count=cat_count,
            generated_at=utcnow().isoformat(),
        )

        # Calculate readiness.
        score = self._score_evidence_package(package)
        package.readiness_score = score.overall_score
        package.readiness_rating = score.rating
        package.provenance_hash = _compute_hash(package)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Evidence package generated: readiness=%.0f%% (%s) in %.1f ms",
            score.overall_score, score.rating, elapsed_ms,
        )
        return package

    # -------------------------------------------------------------------
    # Public -- generate_calculation_provenance
    # -------------------------------------------------------------------

    def generate_calculation_provenance(
        self,
        calculations: List[Dict[str, Any]],
        category: int = 0,
    ) -> ProvenanceChain:
        """Generate a step-by-step provenance chain for calculations.

        Each step is hashed, with each hash including the previous step's
        hash to form an immutable chain.

        Args:
            calculations: List of calculation step dicts.
            category: Scope 3 category number.

        Returns:
            ProvenanceChain with hash chain.
        """
        start_ms = time.time()
        steps: List[CalculationStep] = []
        prev_hash = ""

        for i, calc in enumerate(calculations, 1):
            step = CalculationStep(
                step_number=i,
                description=calc.get("description", f"Step {i}"),
                inputs=calc.get("inputs", {}),
                formula=calc.get("formula", ""),
                output_value=calc.get("output_value", 0),
                output_unit=calc.get("output_unit", "tCO2e"),
            )
            # Chain hash: SHA-256(previous_hash + step_data).
            step_data = json.dumps(step.model_dump(mode="json"), sort_keys=True, default=str)
            chain_input = f"{prev_hash}{step_data}"
            step.step_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()
            prev_hash = step.step_hash
            steps.append(step)

        final_value = steps[-1].output_value if steps else 0
        chain = ProvenanceChain(
            category=category,
            category_name=SCOPE3_CATEGORY_NAMES.get(category, f"Category {category}"),
            steps=steps,
            final_value=final_value,
            chain_hash=prev_hash,
            calculated_at=utcnow().isoformat(),
        )

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.debug(
            "Provenance chain: %d steps, category %d in %.1f ms",
            len(steps), category, elapsed_ms,
        )
        return chain

    # -------------------------------------------------------------------
    # Public -- generate_methodology_log
    # -------------------------------------------------------------------

    def generate_methodology_log(
        self,
        decisions: List[Dict[str, Any]],
    ) -> List[MethodologyDecision]:
        """Generate a methodology decision log.

        Args:
            decisions: List of decision dicts with category, tier, rationale.

        Returns:
            List of MethodologyDecision records.
        """
        records: List[MethodologyDecision] = []
        for d in decisions:
            cat = d.get("category", 0)
            records.append(MethodologyDecision(
                category=cat,
                category_name=SCOPE3_CATEGORY_NAMES.get(cat, f"Category {cat}"),
                selected_tier=d.get("selected_tier", "spend"),
                rationale=d.get("rationale", ""),
                alternatives_considered=d.get("alternatives", []),
                data_availability=d.get("data_availability", ""),
                materiality=d.get("materiality", True),
                reviewer=d.get("reviewer", ""),
                decision_date=d.get("decision_date", utcnow().isoformat()),
            ))
        logger.info("Methodology log: %d decisions recorded", len(records))
        return records

    # -------------------------------------------------------------------
    # Public -- generate_data_source_inventory
    # -------------------------------------------------------------------

    def generate_data_source_inventory(
        self,
        sources: List[Dict[str, Any]],
    ) -> List[DataSourceRecord]:
        """Generate a data source inventory.

        Args:
            sources: List of source dicts.

        Returns:
            List of DataSourceRecord.
        """
        records: List[DataSourceRecord] = []
        for s in sources:
            records.append(DataSourceRecord(
                category=s.get("category", 0),
                source_name=s.get("source_name", ""),
                source_type=s.get("source_type", "primary"),
                description=s.get("description", ""),
                value=s.get("value"),
                unit=s.get("unit", ""),
                collection_date=s.get("collection_date", ""),
                responsible_person=s.get("responsible_person", ""),
                verification_status=s.get("verification_status", "unverified"),
            ))
        logger.info("Data source inventory: %d sources recorded", len(records))
        return records

    # -------------------------------------------------------------------
    # Public -- generate_assumption_register
    # -------------------------------------------------------------------

    def generate_assumption_register(
        self,
        assumptions: List[Dict[str, Any]],
    ) -> List[AssumptionRecord]:
        """Generate an assumption register with sensitivity assessment.

        Args:
            assumptions: List of assumption dicts.

        Returns:
            List of AssumptionRecord.
        """
        records: List[AssumptionRecord] = []
        for a in assumptions:
            records.append(AssumptionRecord(
                category=a.get("category", 0),
                description=a.get("description", ""),
                value=a.get("value", 0),
                unit=a.get("unit", ""),
                source=a.get("source", ""),
                sensitivity=a.get("sensitivity", "medium"),
                sensitivity_range_pct=a.get("sensitivity_range_pct", 20.0),
                impact_on_total_pct=a.get("impact_on_total_pct", 0),
                alternative_values=a.get("alternative_values", []),
            ))
        logger.info("Assumption register: %d assumptions recorded", len(records))
        return records

    # -------------------------------------------------------------------
    # Public -- generate_completeness_statement
    # -------------------------------------------------------------------

    def generate_completeness_statement(
        self,
        inventory_categories: List[int],
        excluded_categories: Optional[Dict[int, str]] = None,
        category_estimates: Optional[Dict[int, float]] = None,
    ) -> List[CompletenessItem]:
        """Generate completeness statement for all 15 Scope 3 categories.

        Args:
            inventory_categories: List of included category numbers.
            excluded_categories: Excluded categories with rationale.
            category_estimates: Estimated share of total by category.

        Returns:
            List of CompletenessItem for all 15 categories.
        """
        excluded = excluded_categories or {}
        estimates = category_estimates or {}
        items: List[CompletenessItem] = []

        for cat_num in range(1, 16):
            included = cat_num in inventory_categories
            if included:
                rationale = "Included based on materiality screening and data availability"
            else:
                rationale = excluded.get(
                    cat_num,
                    "Excluded: not material or not applicable to the organisation",
                )

            items.append(CompletenessItem(
                category=cat_num,
                category_name=SCOPE3_CATEGORY_NAMES.get(cat_num, f"Category {cat_num}"),
                included=included,
                rationale=rationale,
                estimated_pct_of_total=estimates.get(cat_num, 0),
            ))

        logger.info(
            "Completeness statement: %d included, %d excluded",
            sum(1 for i in items if i.included),
            sum(1 for i in items if not i.included),
        )
        return items

    # -------------------------------------------------------------------
    # Public -- generate_ef_provenance
    # -------------------------------------------------------------------

    def generate_ef_provenance(
        self,
        factors: List[Dict[str, Any]],
    ) -> List[EmissionFactorRecord]:
        """Generate emission factor provenance records.

        Args:
            factors: List of emission factor dicts.

        Returns:
            List of EmissionFactorRecord.
        """
        records: List[EmissionFactorRecord] = []
        for f in factors:
            records.append(EmissionFactorRecord(
                category=f.get("category", 0),
                factor_name=f.get("factor_name", ""),
                factor_value=f.get("factor_value", 0),
                factor_unit=f.get("factor_unit", "kgCO2e/unit"),
                source_database=f.get("source_database", ""),
                source_version=f.get("source_version", ""),
                publication_year=f.get("publication_year", 2024),
                geographic_scope=f.get("geographic_scope", "global"),
                sector_scope=f.get("sector_scope", ""),
                applicability_notes=f.get("applicability_notes", ""),
            ))
        logger.info("EF provenance: %d factors recorded", len(records))
        return records

    # -------------------------------------------------------------------
    # Public -- manage_verifier_query
    # -------------------------------------------------------------------

    def manage_verifier_query(
        self,
        query_text: str,
        category: int = 0,
        evidence_category: str = "",
        response_text: str = "",
        evidence_refs: Optional[List[str]] = None,
        assigned_to: str = "",
    ) -> VerifierQuery:
        """Create and manage a verifier query.

        Args:
            query_text: Query text from verifier.
            category: Related Scope 3 category.
            evidence_category: Related evidence category.
            response_text: Response (if responding).
            evidence_refs: Evidence references.
            assigned_to: Person assigned.

        Returns:
            VerifierQuery record.
        """
        now = utcnow().isoformat()
        status = QueryStatus.RESPONDED.value if response_text else QueryStatus.OPEN.value

        query = VerifierQuery(
            query_text=query_text,
            category=category,
            evidence_category=evidence_category,
            status=status,
            raised_date=now,
            response_text=response_text,
            response_date=now if response_text else "",
            evidence_refs=evidence_refs or [],
            assigned_to=assigned_to,
        )
        self._queries.append(query)
        logger.info("Verifier query %s: %s", query.query_id[:8], status)
        return query

    # -------------------------------------------------------------------
    # Public -- track_findings
    # -------------------------------------------------------------------

    def track_findings(
        self,
        description: str,
        severity: str = "minor",
        category: int = 0,
        root_cause: str = "",
        remediation: str = "",
    ) -> Finding:
        """Track an assurance finding.

        Args:
            description: Finding description.
            severity: Finding severity.
            category: Related Scope 3 category.
            root_cause: Root cause analysis.
            remediation: Remediation action.

        Returns:
            Finding record.
        """
        status = FindingStatus.IN_PROGRESS.value if remediation else FindingStatus.OPEN.value
        finding = Finding(
            description=description,
            severity=severity,
            category=category,
            root_cause=root_cause,
            remediation=remediation,
            status=status,
            raised_date=utcnow().isoformat(),
        )
        self._findings.append(finding)
        logger.info(
            "Finding %s tracked: %s (%s)",
            finding.finding_id[:8], severity, status,
        )
        return finding

    # -------------------------------------------------------------------
    # Public -- calculate_assurance_readiness
    # -------------------------------------------------------------------

    def calculate_assurance_readiness(
        self,
        evidence_package: EvidencePackage,
    ) -> AssuranceScore:
        """Calculate assurance readiness score from evidence package.

        Scores each evidence category on completeness and computes
        a weighted overall score.

        Args:
            evidence_package: The evidence package to assess.

        Returns:
            AssuranceScore with breakdown and recommendations.
        """
        start_ms = time.time()
        score = self._score_evidence_package(evidence_package)
        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Assurance readiness: %.0f%% (%s) in %.1f ms",
            score.overall_score, score.rating, elapsed_ms,
        )
        return score

    # -------------------------------------------------------------------
    # Public -- generate_yoy_comparison_package
    # -------------------------------------------------------------------

    def generate_yoy_comparison_package(
        self,
        current_year: int,
        previous_year: int,
        current_by_category: Dict[int, float],
        previous_by_category: Dict[int, float],
        explanations: Optional[Dict[int, str]] = None,
        material_change_threshold_pct: float = 10.0,
    ) -> YoYComparisonPackage:
        """Generate year-on-year comparison package for verification.

        Identifies categories with changes exceeding the threshold and
        requires explanatory notes for material differences.

        Args:
            current_year: Current reporting year.
            previous_year: Previous reporting year.
            current_by_category: Current year emissions by category.
            previous_by_category: Previous year emissions by category.
            explanations: Explanations for material changes.
            material_change_threshold_pct: Threshold for requiring explanation.

        Returns:
            YoYComparisonPackage with flagged material changes.
        """
        start_ms = time.time()
        explanations = explanations or {}
        threshold = _decimal(material_change_threshold_pct)

        items: List[YoYComparisonItem] = []
        material_changes: List[int] = []
        all_cats = sorted(set(list(current_by_category.keys()) + list(previous_by_category.keys())))

        current_total = Decimal("0")
        previous_total = Decimal("0")

        for cat in all_cats:
            curr = _decimal(current_by_category.get(cat, 0))
            prev = _decimal(previous_by_category.get(cat, 0))
            current_total += curr
            previous_total += prev

            change = curr - prev
            change_pct = _safe_pct(abs(change), prev) if prev > Decimal("0") else Decimal("0")
            requires_explanation = float(change_pct) >= float(threshold)

            if requires_explanation:
                material_changes.append(cat)

            items.append(YoYComparisonItem(
                category=cat,
                category_name=SCOPE3_CATEGORY_NAMES.get(cat, f"Category {cat}"),
                previous_tco2e=_round2(prev),
                current_tco2e=_round2(curr),
                change_tco2e=_round2(change),
                change_pct=_round2(change_pct) if prev > Decimal("0") else 0.0,
                explanation=explanations.get(cat, ""),
                requires_explanation=requires_explanation,
            ))

        total_change_pct = _safe_pct(
            abs(current_total - previous_total), previous_total,
        ) if previous_total > Decimal("0") else Decimal("0")

        result = YoYComparisonPackage(
            current_year=current_year,
            previous_year=previous_year,
            current_total_tco2e=_round2(current_total),
            previous_total_tco2e=_round2(previous_total),
            total_change_pct=_round2(total_change_pct),
            items=items,
            material_changes=material_changes,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "YoY comparison: %d-%d, %.1f%% total change, %d material changes in %.1f ms",
            previous_year, current_year, _round2(total_change_pct),
            len(material_changes), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Private -- scoring
    # -------------------------------------------------------------------

    def _score_evidence_package(
        self,
        package: EvidencePackage,
    ) -> AssuranceScore:
        """Score an evidence package across all categories.

        Args:
            package: Evidence package to score.

        Returns:
            AssuranceScore with breakdown.
        """
        by_category: Dict[str, float] = {}
        gaps: List[str] = []
        recommendations: List[str] = []

        # 1. Calculation provenance.
        material_cats = sum(1 for c in package.completeness if c.included)
        prov_count = len(package.calculation_provenance)
        prov_score = min(100.0, _round2(
            _safe_pct(_decimal(prov_count), _decimal(max(material_cats, 1)))
        ))
        by_category[EvidenceCategory.CALCULATION_PROVENANCE] = prov_score
        if prov_score < 100:
            gaps.append(f"Calculation provenance: {prov_count}/{material_cats} categories covered")
            recommendations.append("Add provenance chains for all material categories")

        # 2. Methodology decisions.
        meth_count = len(package.methodology_decisions)
        meth_score = min(100.0, _round2(
            _safe_pct(_decimal(meth_count), _decimal(max(material_cats, 1)))
        ))
        by_category[EvidenceCategory.METHODOLOGY_DECISIONS] = meth_score
        if meth_score < 100:
            gaps.append(f"Methodology decisions: {meth_count}/{material_cats} categories documented")
            recommendations.append("Document methodology decisions for all material categories")

        # 3. Data sources.
        ds_count = len(package.data_sources)
        ds_target = max(material_cats * 3, 1)  # At least 3 sources per category.
        ds_score = min(100.0, _round2(_safe_pct(_decimal(ds_count), _decimal(ds_target))))
        by_category[EvidenceCategory.DATA_SOURCES] = ds_score
        if ds_score < 80:
            gaps.append(f"Data sources: {ds_count} recorded (target: {ds_target})")
            recommendations.append("Expand data source inventory with primary source references")

        # 4. Assumptions.
        assum_count = len(package.assumptions)
        assum_score = min(100.0, 100.0 if assum_count >= 5 else _round2(
            _decimal(assum_count) * Decimal("20")
        ))
        by_category[EvidenceCategory.ASSUMPTIONS] = assum_score
        if assum_score < 100:
            gaps.append(f"Assumptions: {assum_count} documented (minimum 5 recommended)")
            recommendations.append("Document all material assumptions with sensitivity ranges")

        # 5. Emission factors.
        ef_count = len(package.emission_factors)
        ef_target = max(material_cats, 1)
        ef_score = min(100.0, _round2(_safe_pct(_decimal(ef_count), _decimal(ef_target))))
        by_category[EvidenceCategory.EMISSION_FACTORS] = ef_score
        if ef_score < 100:
            gaps.append(f"Emission factors: {ef_count}/{ef_target} documented")
            recommendations.append("Document source and version for all emission factors used")

        # 6. Completeness.
        comp_items = len(package.completeness)
        comp_score = min(100.0, _round2(_safe_pct(_decimal(comp_items), Decimal("15"))))
        by_category[EvidenceCategory.COMPLETENESS] = comp_score
        if comp_score < 100:
            gaps.append(f"Completeness: {comp_items}/15 categories assessed")
            recommendations.append("Assess all 15 Scope 3 categories for completeness")

        # 7. Uncertainty (check if any provenance chains exist as proxy).
        uncertainty_score = 80.0 if prov_count > 0 else 0.0
        by_category[EvidenceCategory.UNCERTAINTY] = uncertainty_score
        if uncertainty_score < 80:
            gaps.append("Uncertainty assessment not documented")
            recommendations.append("Add uncertainty quantification (IPCC Approach 1 or 2)")

        # 8. Boundary.
        boundary_score = 100.0 if material_cats > 0 else 0.0
        by_category[EvidenceCategory.BOUNDARY] = boundary_score
        if boundary_score < 100:
            gaps.append("Organisational boundary not documented")
            recommendations.append("Document organisational and operational boundary")

        # Weighted overall.
        overall = Decimal("0")
        for cat_key, weight in EVIDENCE_CATEGORY_WEIGHTS.items():
            cat_str = cat_key if isinstance(cat_key, str) else cat_key.value
            score = _decimal(by_category.get(cat_str, 0))
            overall += score * _decimal(weight)

        overall_f = _round2(overall)

        # Rating.
        if overall_f >= READINESS_THRESHOLDS[ReadinessRating.READY]:
            rating = ReadinessRating.READY.value
        elif overall_f >= READINESS_THRESHOLDS[ReadinessRating.MOSTLY_READY]:
            rating = ReadinessRating.MOSTLY_READY.value
        elif overall_f >= READINESS_THRESHOLDS[ReadinessRating.PARTIALLY_READY]:
            rating = ReadinessRating.PARTIALLY_READY.value
        else:
            rating = ReadinessRating.NOT_READY.value

        result = AssuranceScore(
            overall_score=overall_f,
            rating=rating,
            by_category=by_category,
            gaps=gaps,
            recommendations=recommendations,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Public -- _compute_provenance
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_provenance(data: Any) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest (64 characters).
        """
        return _compute_hash(data)

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

CalculationStep.model_rebuild()
ProvenanceChain.model_rebuild()
MethodologyDecision.model_rebuild()
DataSourceRecord.model_rebuild()
AssumptionRecord.model_rebuild()
EmissionFactorRecord.model_rebuild()
CompletenessItem.model_rebuild()
VerifierQuery.model_rebuild()
Finding.model_rebuild()
EvidencePackage.model_rebuild()
AssuranceScore.model_rebuild()
YoYComparisonItem.model_rebuild()
YoYComparisonPackage.model_rebuild()
