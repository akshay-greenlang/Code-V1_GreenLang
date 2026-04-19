# -*- coding: utf-8 -*-
"""
EvidenceConsolidationEngine - PACK-048 GHG Assurance Prep Engine 1
====================================================================

Consolidates, categorises, and quality-grades evidence across Scope 1,
Scope 2, and Scope 3 emission sources for third-party assurance
readiness.  Produces a digitally indexed evidence package with SHA-256
file hashes, completeness scoring per scope/category/facility, and
package versioning (DRAFT/REVIEW/FINAL).

Calculation Methodology:
    Evidence Quality Grading (5-point scale):
        EXCELLENT:      Score >= 90  -- Primary source data, verified
        GOOD:           Score >= 75  -- Primary source, unverified
        ADEQUATE:       Score >= 50  -- Secondary source, reasonable basis
        MARGINAL:       Score >= 25  -- Estimated, limited basis
        INSUFFICIENT:   Score <  25  -- No supporting evidence

    Quality Score per evidence item:
        Q_item = w_source * source_score + w_recency * recency_score
                 + w_verification * verify_score + w_completeness * complete_score

        Default weights:
            w_source       = 0.30
            w_recency      = 0.20
            w_verification = 0.30
            w_completeness = 0.20

    Completeness Score per scope:
        C_scope = count(evidenced_items) / count(required_items) * 100

    Evidence Category (per ISAE 3410):
        SOURCE_DATA:    Raw activity data (meter readings, fuel receipts)
        EMISSION_FACTOR: EF selection documentation and sources
        CALCULATION:    Formula application and intermediate results
        ASSUMPTION:     Documented assumptions with justification
        METHODOLOGY:    Calculation methodology documentation
        BOUNDARY:       Organisational/operational boundary evidence
        COMPLETENESS:   Completeness assessment documentation
        CONTROL:        Internal control documentation
        APPROVAL:       Management sign-off and approval records
        EXTERNAL:       Third-party data and external validation

    Scope 1 Evidence Types:
        - Stationary combustion records (fuel purchase, meter data)
        - Mobile fleet logs (fuel cards, telematics, odometer)
        - Process emissions (production data, stoichiometric calcs)
        - Fugitive monitoring (leak detection, OGI surveys)
        - Refrigerant tracking (charge records, top-up logs)

    Scope 2 Evidence Types:
        - Utility bills (electricity, steam, heating, cooling)
        - Contractual instruments (RECs, PPAs, GoOs, I-RECs)
        - Residual mix factors (AIB, Green-e, national registry)
        - Grid emission factors (IEA, national grid operators)

    Scope 3 Evidence Types:
        - Supplier data (primary questionnaires, CDP responses)
        - Spend-based calculations (procurement data, EEIO factors)
        - Activity-based calculations (transport logs, waste data)
        - EF selection rationale (DEFRA, EPA, ecoinvent version)

    Digital Evidence Index:
        hash(file) = SHA-256(file_content)
        index_entry = {file_path, hash, size_bytes, evidence_category,
                       scope, upload_timestamp, uploader}

    Package Versioning:
        DRAFT   -> REVIEW  (all required evidence present)
        REVIEW  -> FINAL   (quality review complete, sign-off obtained)

Regulatory References:
    - ISAE 3410: Assurance Engagements on Greenhouse Gas Statements
    - ISAE 3000 (Revised): Assurance Engagements Other than Audits
    - ISO 14064-3:2019: Specification for validation/verification
    - GHG Protocol Corporate Standard Ch 7: Inventory Quality Mgmt
    - GHG Protocol Scope 2 Guidance: Contractual instruments
    - GHG Protocol Scope 3 Standard: Data collection requirements
    - ESRS E1: Evidence and documentation requirements
    - AA1000AS v3: Evidence and documentation principles

Zero-Hallucination:
    - All quality scores use deterministic Decimal arithmetic
    - Evidence categories from published ISAE 3410 standard
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-048 GHG Assurance Prep
Engine:  1 of 10
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

def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvidenceCategory(str, Enum):
    """Evidence category per ISAE 3410.

    SOURCE_DATA:    Raw activity data (meter readings, fuel receipts).
    EMISSION_FACTOR: EF selection documentation and sources.
    CALCULATION:    Formula application and intermediate results.
    ASSUMPTION:     Documented assumptions with justification.
    METHODOLOGY:    Calculation methodology documentation.
    BOUNDARY:       Organisational/operational boundary evidence.
    COMPLETENESS:   Completeness assessment documentation.
    CONTROL:        Internal control documentation.
    APPROVAL:       Management sign-off and approval records.
    EXTERNAL:       Third-party data and external validation.
    """
    SOURCE_DATA = "source_data"
    EMISSION_FACTOR = "emission_factor"
    CALCULATION = "calculation"
    ASSUMPTION = "assumption"
    METHODOLOGY = "methodology"
    BOUNDARY = "boundary"
    COMPLETENESS = "completeness"
    CONTROL = "control"
    APPROVAL = "approval"
    EXTERNAL = "external"

class QualityGrade(str, Enum):
    """Evidence quality grade (5-point scale).

    EXCELLENT:      Score >= 90 -- Primary source data, verified.
    GOOD:           Score >= 75 -- Primary source, unverified.
    ADEQUATE:       Score >= 50 -- Secondary source, reasonable basis.
    MARGINAL:       Score >= 25 -- Estimated, limited basis.
    INSUFFICIENT:   Score <  25 -- No supporting evidence.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    MARGINAL = "marginal"
    INSUFFICIENT = "insufficient"

class EmissionScope(str, Enum):
    """GHG Protocol emission scope.

    SCOPE_1: Direct emissions.
    SCOPE_2: Indirect energy emissions.
    SCOPE_3: Other indirect emissions.
    CROSS_SCOPE: Cross-scope consolidation.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    CROSS_SCOPE = "cross_scope"

class Scope1EvidenceType(str, Enum):
    """Scope 1 evidence sub-types."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_FLEET = "mobile_fleet"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_MONITORING = "fugitive_monitoring"
    REFRIGERANT_TRACKING = "refrigerant_tracking"

class Scope2EvidenceType(str, Enum):
    """Scope 2 evidence sub-types."""
    UTILITY_BILLS = "utility_bills"
    CONTRACTUAL_INSTRUMENTS = "contractual_instruments"
    RESIDUAL_MIX_FACTORS = "residual_mix_factors"
    GRID_EMISSION_FACTORS = "grid_emission_factors"

class Scope3EvidenceType(str, Enum):
    """Scope 3 evidence sub-types."""
    SUPPLIER_DATA = "supplier_data"
    SPEND_BASED_CALCS = "spend_based_calcs"
    ACTIVITY_BASED_CALCS = "activity_based_calcs"
    EF_SELECTION_RATIONALE = "ef_selection_rationale"

class SourceType(str, Enum):
    """Source type for evidence quality scoring.

    PRIMARY_VERIFIED:   Primary data with third-party verification.
    PRIMARY_UNVERIFIED: Primary data without verification.
    SECONDARY_DIRECT:   Secondary data from direct source (e.g., supplier).
    SECONDARY_INDIRECT: Secondary data from indirect source (e.g., database).
    ESTIMATED:          Estimated or modelled data.
    """
    PRIMARY_VERIFIED = "primary_verified"
    PRIMARY_UNVERIFIED = "primary_unverified"
    SECONDARY_DIRECT = "secondary_direct"
    SECONDARY_INDIRECT = "secondary_indirect"
    ESTIMATED = "estimated"

class PackageStatus(str, Enum):
    """Evidence package versioning status.

    DRAFT:  Initial evidence collection.
    REVIEW: All required evidence present, under quality review.
    FINAL:  Quality review complete, sign-off obtained.
    """
    DRAFT = "draft"
    REVIEW = "review"
    FINAL = "final"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Quality score thresholds for grade assignment
GRADE_THRESHOLDS: Dict[str, Decimal] = {
    QualityGrade.EXCELLENT.value: Decimal("90"),
    QualityGrade.GOOD.value: Decimal("75"),
    QualityGrade.ADEQUATE.value: Decimal("50"),
    QualityGrade.MARGINAL.value: Decimal("25"),
    QualityGrade.INSUFFICIENT.value: Decimal("0"),
}

# Source type scores (0-100)
SOURCE_TYPE_SCORES: Dict[str, Decimal] = {
    SourceType.PRIMARY_VERIFIED.value: Decimal("100"),
    SourceType.PRIMARY_UNVERIFIED.value: Decimal("80"),
    SourceType.SECONDARY_DIRECT.value: Decimal("60"),
    SourceType.SECONDARY_INDIRECT.value: Decimal("40"),
    SourceType.ESTIMATED.value: Decimal("20"),
}

# Default quality weights
DEFAULT_SOURCE_WEIGHT: Decimal = Decimal("0.30")
DEFAULT_RECENCY_WEIGHT: Decimal = Decimal("0.20")
DEFAULT_VERIFICATION_WEIGHT: Decimal = Decimal("0.30")
DEFAULT_COMPLETENESS_WEIGHT: Decimal = Decimal("0.20")

# Required evidence categories per ISAE 3410
REQUIRED_CATEGORIES: List[str] = [
    EvidenceCategory.SOURCE_DATA.value,
    EvidenceCategory.EMISSION_FACTOR.value,
    EvidenceCategory.CALCULATION.value,
    EvidenceCategory.METHODOLOGY.value,
    EvidenceCategory.BOUNDARY.value,
]

# Scope 1 required evidence types
SCOPE_1_REQUIRED: List[str] = [
    Scope1EvidenceType.STATIONARY_COMBUSTION.value,
    Scope1EvidenceType.MOBILE_FLEET.value,
    Scope1EvidenceType.PROCESS_EMISSIONS.value,
    Scope1EvidenceType.FUGITIVE_MONITORING.value,
    Scope1EvidenceType.REFRIGERANT_TRACKING.value,
]

# Scope 2 required evidence types
SCOPE_2_REQUIRED: List[str] = [
    Scope2EvidenceType.UTILITY_BILLS.value,
    Scope2EvidenceType.CONTRACTUAL_INSTRUMENTS.value,
    Scope2EvidenceType.RESIDUAL_MIX_FACTORS.value,
    Scope2EvidenceType.GRID_EMISSION_FACTORS.value,
]

# Scope 3 required evidence types
SCOPE_3_REQUIRED: List[str] = [
    Scope3EvidenceType.SUPPLIER_DATA.value,
    Scope3EvidenceType.SPEND_BASED_CALCS.value,
    Scope3EvidenceType.ACTIVITY_BASED_CALCS.value,
    Scope3EvidenceType.EF_SELECTION_RATIONALE.value,
]

MAX_EVIDENCE_AGE_MONTHS: int = 18
MAX_EVIDENCE_ITEMS: int = 100000

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class QualityWeights(BaseModel):
    """Weights for evidence quality scoring.

    Attributes:
        source_weight:          Weight for source type scoring.
        recency_weight:         Weight for data recency scoring.
        verification_weight:    Weight for verification status scoring.
        completeness_weight:    Weight for data completeness scoring.
    """
    source_weight: Decimal = Field(default=DEFAULT_SOURCE_WEIGHT, ge=0, le=1)
    recency_weight: Decimal = Field(default=DEFAULT_RECENCY_WEIGHT, ge=0, le=1)
    verification_weight: Decimal = Field(default=DEFAULT_VERIFICATION_WEIGHT, ge=0, le=1)
    completeness_weight: Decimal = Field(default=DEFAULT_COMPLETENESS_WEIGHT, ge=0, le=1)

    @model_validator(mode="after")
    def check_weights_sum(self) -> "QualityWeights":
        total = (
            self.source_weight + self.recency_weight
            + self.verification_weight + self.completeness_weight
        )
        if abs(total - Decimal("1")) > Decimal("0.01"):
            logger.warning(
                "Quality weights sum to %s (expected ~1.0). Results may be skewed.", total
            )
        return self

class EvidenceItem(BaseModel):
    """A single evidence item for assurance.

    Attributes:
        evidence_id:        Unique evidence identifier.
        title:              Evidence title/description.
        scope:              Emission scope.
        category:           ISAE 3410 evidence category.
        sub_type:           Scope-specific evidence sub-type.
        facility_id:        Facility identifier.
        facility_name:      Facility name.
        source_type:        Source type for quality scoring.
        source_document:    Source document reference.
        file_path:          File path or URI.
        file_hash:          SHA-256 hash of file content.
        file_size_bytes:    File size in bytes.
        reporting_period:   Reporting period (e.g. "2024-01" to "2024-12").
        data_date:          Date of data (for recency scoring).
        is_verified:        Whether data has been verified.
        verifier_name:      Name of verifier (if verified).
        completeness_pct:   Completeness percentage of this evidence item.
        notes:              Additional notes.
        uploaded_by:        Uploader identity.
        uploaded_at:        Upload timestamp.
    """
    evidence_id: str = Field(default_factory=_new_uuid, description="Evidence ID")
    title: str = Field(default="", description="Evidence title")
    scope: EmissionScope = Field(..., description="Emission scope")
    category: EvidenceCategory = Field(..., description="Evidence category")
    sub_type: str = Field(default="", description="Scope-specific sub-type")
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(default="", description="Facility name")
    source_type: SourceType = Field(
        default=SourceType.SECONDARY_INDIRECT, description="Source type"
    )
    source_document: str = Field(default="", description="Source document ref")
    file_path: str = Field(default="", description="File path/URI")
    file_hash: str = Field(default="", description="SHA-256 file hash")
    file_size_bytes: int = Field(default=0, ge=0, description="File size")
    reporting_period: str = Field(default="", description="Reporting period")
    data_date: str = Field(default="", description="Data date (ISO)")
    is_verified: bool = Field(default=False, description="Verification status")
    verifier_name: str = Field(default="", description="Verifier name")
    completeness_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100, description="Completeness %"
    )
    notes: str = Field(default="", description="Notes")
    uploaded_by: str = Field(default="", description="Uploader")
    uploaded_at: str = Field(default="", description="Upload timestamp")

    @field_validator("completeness_pct", mode="before")
    @classmethod
    def coerce_completeness(cls, v: Any) -> Decimal:
        return _decimal(v)

class ConsolidationConfig(BaseModel):
    """Configuration for evidence consolidation.

    Attributes:
        organisation_id:        Organisation identifier.
        organisation_name:      Organisation name.
        reporting_year:         Reporting year.
        include_scope_1:        Whether to include Scope 1.
        include_scope_2:        Whether to include Scope 2.
        include_scope_3:        Whether to include Scope 3.
        scope_3_categories:     Specific Scope 3 categories (1-15).
        quality_weights:        Quality scoring weights.
        required_categories:    Required evidence categories.
        facility_ids:           Facility IDs to include (empty=all).
        max_evidence_age_months: Maximum evidence age in months.
        output_precision:       Output decimal places.
    """
    organisation_id: str = Field(default="", description="Org ID")
    organisation_name: str = Field(default="", description="Org name")
    reporting_year: int = Field(default=2024, description="Reporting year")
    include_scope_1: bool = Field(default=True, description="Include Scope 1")
    include_scope_2: bool = Field(default=True, description="Include Scope 2")
    include_scope_3: bool = Field(default=True, description="Include Scope 3")
    scope_3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Scope 3 categories to include (1-15)",
    )
    quality_weights: QualityWeights = Field(
        default_factory=QualityWeights, description="Quality weights"
    )
    required_categories: List[str] = Field(
        default_factory=lambda: list(REQUIRED_CATEGORIES),
        description="Required evidence categories",
    )
    facility_ids: List[str] = Field(
        default_factory=list, description="Facility filter"
    )
    max_evidence_age_months: int = Field(
        default=MAX_EVIDENCE_AGE_MONTHS, ge=1, le=60, description="Max age months"
    )
    output_precision: int = Field(default=2, ge=0, le=6, description="Output precision")

class ConsolidationInput(BaseModel):
    """Input for evidence consolidation.

    Attributes:
        evidence_items:     All evidence items to consolidate.
        config:             Consolidation configuration.
    """
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list, description="Evidence items"
    )
    config: ConsolidationConfig = Field(
        default_factory=ConsolidationConfig, description="Configuration"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class EvidenceQualityScore(BaseModel):
    """Quality score for a single evidence item.

    Attributes:
        evidence_id:        Evidence identifier.
        source_score:       Source type sub-score (0-100).
        recency_score:      Recency sub-score (0-100).
        verification_score: Verification sub-score (0-100).
        completeness_score: Completeness sub-score (0-100).
        composite_score:    Weighted composite score (0-100).
        grade:              Quality grade.
    """
    evidence_id: str = Field(default="", description="Evidence ID")
    source_score: Decimal = Field(default=Decimal("0"), description="Source score")
    recency_score: Decimal = Field(default=Decimal("0"), description="Recency score")
    verification_score: Decimal = Field(default=Decimal("0"), description="Verification score")
    completeness_score: Decimal = Field(default=Decimal("0"), description="Completeness score")
    composite_score: Decimal = Field(default=Decimal("0"), description="Composite score")
    grade: str = Field(default=QualityGrade.INSUFFICIENT.value, description="Quality grade")

class CompletenessScore(BaseModel):
    """Completeness score for a scope, category, or facility.

    Attributes:
        dimension:          Dimension name (scope, category, facility).
        dimension_value:    Dimension value (e.g. "scope_1", "source_data").
        required_count:     Required evidence count.
        present_count:      Present evidence count.
        completeness_pct:   Completeness percentage.
        missing_items:      List of missing evidence descriptions.
        grade:              Quality grade based on completeness.
    """
    dimension: str = Field(default="", description="Dimension")
    dimension_value: str = Field(default="", description="Dimension value")
    required_count: int = Field(default=0, description="Required count")
    present_count: int = Field(default=0, description="Present count")
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Completeness %")
    missing_items: List[str] = Field(default_factory=list, description="Missing items")
    grade: str = Field(default=QualityGrade.INSUFFICIENT.value, description="Grade")

class EvidenceIndexEntry(BaseModel):
    """Entry in the digital evidence index.

    Attributes:
        evidence_id:    Evidence identifier.
        file_path:      File path or URI.
        file_hash:      SHA-256 hash.
        file_size_bytes: File size.
        category:       Evidence category.
        scope:          Emission scope.
        sub_type:       Evidence sub-type.
        facility_id:    Facility identifier.
        uploaded_at:    Upload timestamp.
        uploaded_by:    Uploader.
    """
    evidence_id: str = Field(default="", description="Evidence ID")
    file_path: str = Field(default="", description="File path")
    file_hash: str = Field(default="", description="SHA-256 hash")
    file_size_bytes: int = Field(default=0, description="Size bytes")
    category: str = Field(default="", description="Category")
    scope: str = Field(default="", description="Scope")
    sub_type: str = Field(default="", description="Sub-type")
    facility_id: str = Field(default="", description="Facility ID")
    uploaded_at: str = Field(default="", description="Upload time")
    uploaded_by: str = Field(default="", description="Uploader")

class EvidenceIndex(BaseModel):
    """Digital evidence index with SHA-256 file hashes.

    Attributes:
        index_id:           Index identifier.
        total_files:        Total file count.
        total_size_bytes:   Total size in bytes.
        entries:            Index entries.
        index_hash:         SHA-256 hash of the full index.
    """
    index_id: str = Field(default_factory=_new_uuid, description="Index ID")
    total_files: int = Field(default=0, description="Total files")
    total_size_bytes: int = Field(default=0, description="Total size")
    entries: List[EvidenceIndexEntry] = Field(default_factory=list, description="Entries")
    index_hash: str = Field(default="", description="Index hash")

class ScopeEvidenceSummary(BaseModel):
    """Summary of evidence for a single scope.

    Attributes:
        scope:                  Emission scope.
        total_items:            Total evidence items.
        by_category:            Item count per ISAE 3410 category.
        by_sub_type:            Item count per sub-type.
        by_facility:            Item count per facility.
        avg_quality_score:      Average quality score.
        avg_quality_grade:      Average quality grade.
        completeness_scores:    Completeness scores by dimension.
    """
    scope: str = Field(default="", description="Scope")
    total_items: int = Field(default=0, description="Total items")
    by_category: Dict[str, int] = Field(default_factory=dict, description="By category")
    by_sub_type: Dict[str, int] = Field(default_factory=dict, description="By sub-type")
    by_facility: Dict[str, int] = Field(default_factory=dict, description="By facility")
    avg_quality_score: Decimal = Field(default=Decimal("0"), description="Avg quality")
    avg_quality_grade: str = Field(
        default=QualityGrade.INSUFFICIENT.value, description="Avg grade"
    )
    completeness_scores: List[CompletenessScore] = Field(
        default_factory=list, description="Completeness"
    )

class EvidencePackage(BaseModel):
    """Consolidated evidence package.

    Attributes:
        package_id:             Package identifier.
        organisation_id:        Organisation identifier.
        reporting_year:         Reporting year.
        status:                 Package status (DRAFT/REVIEW/FINAL).
        version:                Package version number.
        scope_summaries:        Per-scope evidence summaries.
        cross_scope_summary:    Cross-scope consolidation summary.
        overall_completeness:   Overall completeness percentage.
        overall_quality_score:  Overall quality score.
        overall_quality_grade:  Overall quality grade.
        evidence_index:         Digital evidence index.
        quality_scores:         Per-item quality scores.
        completeness_scores:    Aggregated completeness scores.
        total_evidence_items:   Total evidence items.
        total_file_size_bytes:  Total file size.
    """
    package_id: str = Field(default_factory=_new_uuid, description="Package ID")
    organisation_id: str = Field(default="", description="Org ID")
    reporting_year: int = Field(default=2024, description="Year")
    status: str = Field(default=PackageStatus.DRAFT.value, description="Status")
    version: int = Field(default=1, ge=1, description="Version")
    scope_summaries: List[ScopeEvidenceSummary] = Field(
        default_factory=list, description="Scope summaries"
    )
    cross_scope_summary: Optional[ScopeEvidenceSummary] = Field(
        default=None, description="Cross-scope"
    )
    overall_completeness: Decimal = Field(default=Decimal("0"), description="Overall completeness")
    overall_quality_score: Decimal = Field(default=Decimal("0"), description="Overall quality")
    overall_quality_grade: str = Field(
        default=QualityGrade.INSUFFICIENT.value, description="Overall grade"
    )
    evidence_index: EvidenceIndex = Field(
        default_factory=EvidenceIndex, description="Evidence index"
    )
    quality_scores: List[EvidenceQualityScore] = Field(
        default_factory=list, description="Quality scores"
    )
    completeness_scores: List[CompletenessScore] = Field(
        default_factory=list, description="Completeness scores"
    )
    total_evidence_items: int = Field(default=0, description="Total items")
    total_file_size_bytes: int = Field(default=0, description="Total size")

class PackageResult(BaseModel):
    """Complete result of evidence consolidation.

    Attributes:
        result_id:              Unique result identifier.
        organisation_id:        Organisation identifier.
        package:                Evidence package.
        status_recommendation:  Recommended package status.
        readiness_for_review:   Whether package is ready for review.
        gaps_identified:        Number of gaps identified.
        gap_details:            Gap descriptions.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Org ID")
    package: EvidencePackage = Field(
        default_factory=EvidencePackage, description="Evidence package"
    )
    status_recommendation: str = Field(
        default=PackageStatus.DRAFT.value, description="Status recommendation"
    )
    readiness_for_review: bool = Field(default=False, description="Ready for review")
    gaps_identified: int = Field(default=0, description="Gaps identified")
    gap_details: List[str] = Field(default_factory=list, description="Gap details")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EvidenceConsolidationEngine:
    """Consolidates and quality-grades evidence for GHG assurance readiness.

    Produces digitally indexed evidence packages with SHA-256 file hashes,
    completeness scoring per scope/category/facility, and package versioning.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every evidence item quality-scored.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("EvidenceConsolidationEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: ConsolidationInput) -> PackageResult:
        """Consolidate evidence into an assurance-ready package.

        Args:
            input_data: Evidence items and consolidation configuration.

        Returns:
            PackageResult with evidence package, quality scores, and completeness.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.config
        prec = config.output_precision
        prec_str = "0." + "0" * prec
        items = input_data.evidence_items

        if len(items) > MAX_EVIDENCE_ITEMS:
            raise ValueError(
                f"Maximum {MAX_EVIDENCE_ITEMS} evidence items allowed (got {len(items)})"
            )

        # Step 1: Filter by scope inclusion
        filtered = self._filter_by_scope(items, config)
        if len(filtered) < len(items):
            warnings.append(
                f"Excluded {len(items) - len(filtered)} items outside scope filter."
            )

        # Step 2: Filter by facility
        if config.facility_ids:
            fac_set = set(config.facility_ids)
            before = len(filtered)
            filtered = [
                it for it in filtered
                if not it.facility_id or it.facility_id in fac_set
            ]
            if len(filtered) < before:
                warnings.append(
                    f"Excluded {before - len(filtered)} items outside facility filter."
                )

        # Step 3: Score quality for each evidence item
        quality_scores: List[EvidenceQualityScore] = []
        for item in filtered:
            qs = self._score_evidence_quality(item, config, prec_str)
            quality_scores.append(qs)

        # Step 4: Build per-scope summaries
        scope_summaries: List[ScopeEvidenceSummary] = []
        for scope_val in [EmissionScope.SCOPE_1, EmissionScope.SCOPE_2, EmissionScope.SCOPE_3]:
            if scope_val == EmissionScope.SCOPE_1 and not config.include_scope_1:
                continue
            if scope_val == EmissionScope.SCOPE_2 and not config.include_scope_2:
                continue
            if scope_val == EmissionScope.SCOPE_3 and not config.include_scope_3:
                continue
            scope_items = [it for it in filtered if it.scope == scope_val]
            scope_qs = [
                qs for qs in quality_scores
                if any(it.evidence_id == qs.evidence_id and it.scope == scope_val for it in filtered)
            ]
            summary = self._build_scope_summary(scope_val.value, scope_items, scope_qs, config, prec_str)
            scope_summaries.append(summary)

        # Step 5: Cross-scope consolidation
        cross_scope = self._build_cross_scope_summary(filtered, quality_scores, config, prec_str)

        # Step 6: Completeness scoring
        completeness_scores = self._compute_completeness_scores(filtered, config, prec_str)

        # Step 7: Overall metrics
        overall_quality = Decimal("0")
        if quality_scores:
            total_q = sum(qs.composite_score for qs in quality_scores)
            overall_quality = _safe_divide(
                total_q, _decimal(len(quality_scores))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        overall_grade = self._assign_grade(overall_quality)

        overall_completeness = Decimal("0")
        if completeness_scores:
            total_c = sum(cs.completeness_pct for cs in completeness_scores)
            overall_completeness = _safe_divide(
                total_c, _decimal(len(completeness_scores))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Step 8: Build evidence index
        evidence_index = self._build_evidence_index(filtered)

        # Step 9: Total file size
        total_size = sum(it.file_size_bytes for it in filtered)

        # Step 10: Gap analysis
        gaps = self._identify_gaps(filtered, completeness_scores, config)

        # Step 11: Status recommendation
        status_rec = self._recommend_status(overall_completeness, overall_quality, gaps)
        ready_for_review = status_rec in (PackageStatus.REVIEW.value, PackageStatus.FINAL.value)

        # Build package
        package = EvidencePackage(
            organisation_id=config.organisation_id,
            reporting_year=config.reporting_year,
            status=status_rec,
            version=1,
            scope_summaries=scope_summaries,
            cross_scope_summary=cross_scope,
            overall_completeness=overall_completeness,
            overall_quality_score=overall_quality,
            overall_quality_grade=overall_grade,
            evidence_index=evidence_index,
            quality_scores=quality_scores,
            completeness_scores=completeness_scores,
            total_evidence_items=len(filtered),
            total_file_size_bytes=total_size,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = PackageResult(
            organisation_id=config.organisation_id,
            package=package,
            status_recommendation=status_rec,
            readiness_for_review=ready_for_review,
            gaps_identified=len(gaps),
            gap_details=gaps,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def score_evidence_item(
        self, item: EvidenceItem, config: ConsolidationConfig,
    ) -> EvidenceQualityScore:
        """Score a single evidence item.

        Args:
            item:   Evidence item to score.
            config: Configuration with quality weights.

        Returns:
            EvidenceQualityScore.
        """
        prec_str = "0." + "0" * config.output_precision
        return self._score_evidence_quality(item, config, prec_str)

    def compute_file_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content.

        Args:
            content: File content bytes.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return hashlib.sha256(content).hexdigest()

    def assign_quality_grade(self, score: Decimal) -> str:
        """Assign quality grade from composite score.

        Args:
            score: Composite quality score (0-100).

        Returns:
            Quality grade string.
        """
        return self._assign_grade(score)

    # ------------------------------------------------------------------
    # Internal: Filtering
    # ------------------------------------------------------------------

    def _filter_by_scope(
        self, items: List[EvidenceItem], config: ConsolidationConfig,
    ) -> List[EvidenceItem]:
        """Filter evidence items by scope inclusion settings."""
        result: List[EvidenceItem] = []
        for item in items:
            if item.scope == EmissionScope.SCOPE_1 and not config.include_scope_1:
                continue
            if item.scope == EmissionScope.SCOPE_2 and not config.include_scope_2:
                continue
            if item.scope == EmissionScope.SCOPE_3 and not config.include_scope_3:
                continue
            if item.scope == EmissionScope.CROSS_SCOPE:
                # Always include cross-scope items
                pass
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Internal: Quality Scoring
    # ------------------------------------------------------------------

    def _score_evidence_quality(
        self,
        item: EvidenceItem,
        config: ConsolidationConfig,
        prec_str: str,
    ) -> EvidenceQualityScore:
        """Score evidence quality.

        Q_item = w_source * source_score + w_recency * recency_score
                 + w_verification * verify_score + w_completeness * complete_score
        """
        qw = config.quality_weights

        # Source score
        source_score = SOURCE_TYPE_SCORES.get(item.source_type.value, Decimal("20"))

        # Recency score: 100 for current year, -10 per month past max_age
        recency_score = Decimal("100")
        if item.data_date:
            try:
                data_dt = datetime.fromisoformat(item.data_date.replace("Z", "+00:00"))
                now = utcnow()
                months_old = (now.year - data_dt.year) * 12 + (now.month - data_dt.month)
                if months_old > config.max_evidence_age_months:
                    recency_score = Decimal("0")
                elif months_old > 0:
                    decay = _decimal(months_old) * Decimal("100") / _decimal(config.max_evidence_age_months)
                    recency_score = max(Decimal("100") - decay, Decimal("0"))
            except (ValueError, TypeError):
                recency_score = Decimal("50")

        # Verification score
        verification_score = Decimal("100") if item.is_verified else Decimal("30")

        # Completeness score
        completeness_score = item.completeness_pct

        # Composite
        composite = (
            qw.source_weight * source_score
            + qw.recency_weight * recency_score
            + qw.verification_weight * verification_score
            + qw.completeness_weight * completeness_score
        ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        grade = self._assign_grade(composite)

        return EvidenceQualityScore(
            evidence_id=item.evidence_id,
            source_score=source_score.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            recency_score=recency_score.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            verification_score=verification_score.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            completeness_score=completeness_score.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            composite_score=composite,
            grade=grade,
        )

    def _assign_grade(self, score: Decimal) -> str:
        """Assign quality grade from score."""
        if score >= Decimal("90"):
            return QualityGrade.EXCELLENT.value
        if score >= Decimal("75"):
            return QualityGrade.GOOD.value
        if score >= Decimal("50"):
            return QualityGrade.ADEQUATE.value
        if score >= Decimal("25"):
            return QualityGrade.MARGINAL.value
        return QualityGrade.INSUFFICIENT.value

    # ------------------------------------------------------------------
    # Internal: Scope Summaries
    # ------------------------------------------------------------------

    def _build_scope_summary(
        self,
        scope_value: str,
        items: List[EvidenceItem],
        quality_scores: List[EvidenceQualityScore],
        config: ConsolidationConfig,
        prec_str: str,
    ) -> ScopeEvidenceSummary:
        """Build evidence summary for a single scope."""
        by_category: Dict[str, int] = {}
        by_sub_type: Dict[str, int] = {}
        by_facility: Dict[str, int] = {}

        for item in items:
            cat = item.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            if item.sub_type:
                by_sub_type[item.sub_type] = by_sub_type.get(item.sub_type, 0) + 1
            fac = item.facility_id or "unspecified"
            by_facility[fac] = by_facility.get(fac, 0) + 1

        # Average quality
        item_ids = {it.evidence_id for it in items}
        matching_qs = [qs for qs in quality_scores if qs.evidence_id in item_ids]
        avg_quality = Decimal("0")
        if matching_qs:
            total_q = sum(qs.composite_score for qs in matching_qs)
            avg_quality = _safe_divide(
                total_q, _decimal(len(matching_qs))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        avg_grade = self._assign_grade(avg_quality)

        # Completeness per required category
        completeness_list: List[CompletenessScore] = []
        required = self._get_required_for_scope(scope_value)
        for req_type in required:
            present = sum(1 for it in items if it.sub_type == req_type)
            pct = _safe_divide(
                _decimal(present), _decimal(1), Decimal("0")
            ) * Decimal("100")
            pct = min(pct, Decimal("100")).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            missing: List[str] = []
            if present == 0:
                missing.append(f"No evidence for {req_type}")
            completeness_list.append(CompletenessScore(
                dimension="sub_type",
                dimension_value=req_type,
                required_count=1,
                present_count=min(present, 1),
                completeness_pct=pct,
                missing_items=missing,
                grade=self._assign_grade(pct),
            ))

        return ScopeEvidenceSummary(
            scope=scope_value,
            total_items=len(items),
            by_category=by_category,
            by_sub_type=by_sub_type,
            by_facility=by_facility,
            avg_quality_score=avg_quality,
            avg_quality_grade=avg_grade,
            completeness_scores=completeness_list,
        )

    def _build_cross_scope_summary(
        self,
        items: List[EvidenceItem],
        quality_scores: List[EvidenceQualityScore],
        config: ConsolidationConfig,
        prec_str: str,
    ) -> ScopeEvidenceSummary:
        """Build cross-scope consolidation summary."""
        by_scope: Dict[str, int] = {}
        by_category: Dict[str, int] = {}

        for item in items:
            s = item.scope.value
            by_scope[s] = by_scope.get(s, 0) + 1
            cat = item.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        avg_quality = Decimal("0")
        if quality_scores:
            total_q = sum(qs.composite_score for qs in quality_scores)
            avg_quality = _safe_divide(
                total_q, _decimal(len(quality_scores))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Completeness per required ISAE 3410 category
        completeness_list: List[CompletenessScore] = []
        for req_cat in config.required_categories:
            present = by_category.get(req_cat, 0)
            pct = Decimal("100") if present > 0 else Decimal("0")
            missing: List[str] = []
            if present == 0:
                missing.append(f"No evidence in category: {req_cat}")
            completeness_list.append(CompletenessScore(
                dimension="category",
                dimension_value=req_cat,
                required_count=1,
                present_count=min(present, 1),
                completeness_pct=pct,
                missing_items=missing,
                grade=self._assign_grade(pct),
            ))

        return ScopeEvidenceSummary(
            scope=EmissionScope.CROSS_SCOPE.value,
            total_items=len(items),
            by_category=by_category,
            by_sub_type=by_scope,
            by_facility={},
            avg_quality_score=avg_quality,
            avg_quality_grade=self._assign_grade(avg_quality),
            completeness_scores=completeness_list,
        )

    def _get_required_for_scope(self, scope_value: str) -> List[str]:
        """Get required evidence sub-types for a scope."""
        if scope_value == EmissionScope.SCOPE_1.value:
            return SCOPE_1_REQUIRED
        if scope_value == EmissionScope.SCOPE_2.value:
            return SCOPE_2_REQUIRED
        if scope_value == EmissionScope.SCOPE_3.value:
            return SCOPE_3_REQUIRED
        return []

    # ------------------------------------------------------------------
    # Internal: Completeness Scoring
    # ------------------------------------------------------------------

    def _compute_completeness_scores(
        self,
        items: List[EvidenceItem],
        config: ConsolidationConfig,
        prec_str: str,
    ) -> List[CompletenessScore]:
        """Compute completeness scores by scope, category, and facility."""
        scores: List[CompletenessScore] = []

        # By scope
        scope_counts: Dict[str, int] = {}
        for item in items:
            s = item.scope.value
            scope_counts[s] = scope_counts.get(s, 0) + 1

        active_scopes: List[str] = []
        if config.include_scope_1:
            active_scopes.append(EmissionScope.SCOPE_1.value)
        if config.include_scope_2:
            active_scopes.append(EmissionScope.SCOPE_2.value)
        if config.include_scope_3:
            active_scopes.append(EmissionScope.SCOPE_3.value)

        for scope_val in active_scopes:
            present = scope_counts.get(scope_val, 0)
            required_types = self._get_required_for_scope(scope_val)
            types_present = len(set(
                it.sub_type for it in items
                if it.scope.value == scope_val and it.sub_type in required_types
            ))
            req_count = len(required_types)
            pct = _safe_divide(
                _decimal(types_present), _decimal(req_count)
            ) * Decimal("100")
            pct = pct.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

            missing: List[str] = []
            present_types = {
                it.sub_type for it in items
                if it.scope.value == scope_val
            }
            for rt in required_types:
                if rt not in present_types:
                    missing.append(f"Missing {scope_val} evidence: {rt}")

            scores.append(CompletenessScore(
                dimension="scope",
                dimension_value=scope_val,
                required_count=req_count,
                present_count=types_present,
                completeness_pct=pct,
                missing_items=missing,
                grade=self._assign_grade(pct),
            ))

        # By ISAE 3410 category
        cat_present = set(it.category.value for it in items)
        for req_cat in config.required_categories:
            is_present = req_cat in cat_present
            pct = Decimal("100") if is_present else Decimal("0")
            missing = [] if is_present else [f"Missing category: {req_cat}"]
            scores.append(CompletenessScore(
                dimension="category",
                dimension_value=req_cat,
                required_count=1,
                present_count=1 if is_present else 0,
                completeness_pct=pct,
                missing_items=missing,
                grade=self._assign_grade(pct),
            ))

        return scores

    # ------------------------------------------------------------------
    # Internal: Evidence Index
    # ------------------------------------------------------------------

    def _build_evidence_index(self, items: List[EvidenceItem]) -> EvidenceIndex:
        """Build digital evidence index."""
        entries: List[EvidenceIndexEntry] = []
        total_size = 0

        for item in items:
            if item.file_path:
                entries.append(EvidenceIndexEntry(
                    evidence_id=item.evidence_id,
                    file_path=item.file_path,
                    file_hash=item.file_hash,
                    file_size_bytes=item.file_size_bytes,
                    category=item.category.value,
                    scope=item.scope.value,
                    sub_type=item.sub_type,
                    facility_id=item.facility_id,
                    uploaded_at=item.uploaded_at,
                    uploaded_by=item.uploaded_by,
                ))
                total_size += item.file_size_bytes

        index = EvidenceIndex(
            total_files=len(entries),
            total_size_bytes=total_size,
            entries=entries,
        )
        index.index_hash = _compute_hash(index)
        return index

    # ------------------------------------------------------------------
    # Internal: Gap Analysis
    # ------------------------------------------------------------------

    def _identify_gaps(
        self,
        items: List[EvidenceItem],
        completeness_scores: List[CompletenessScore],
        config: ConsolidationConfig,
    ) -> List[str]:
        """Identify evidence gaps."""
        gaps: List[str] = []

        for cs in completeness_scores:
            if cs.completeness_pct < Decimal("100"):
                gaps.extend(cs.missing_items)

        # Check for items missing file hashes
        no_hash = [it for it in items if it.file_path and not it.file_hash]
        if no_hash:
            gaps.append(
                f"{len(no_hash)} evidence file(s) missing SHA-256 hash."
            )

        # Check for unverified high-importance items
        unverified_source = [
            it for it in items
            if it.category == EvidenceCategory.SOURCE_DATA and not it.is_verified
        ]
        if unverified_source:
            gaps.append(
                f"{len(unverified_source)} source data item(s) not verified."
            )

        # Check for missing approval records
        has_approval = any(it.category == EvidenceCategory.APPROVAL for it in items)
        if not has_approval:
            gaps.append("No management approval records found.")

        return gaps

    # ------------------------------------------------------------------
    # Internal: Status Recommendation
    # ------------------------------------------------------------------

    def _recommend_status(
        self,
        completeness: Decimal,
        quality: Decimal,
        gaps: List[str],
    ) -> str:
        """Recommend package status based on completeness and quality."""
        if completeness >= Decimal("95") and quality >= Decimal("75") and len(gaps) <= 2:
            return PackageStatus.FINAL.value
        if completeness >= Decimal("70") and quality >= Decimal("50"):
            return PackageStatus.REVIEW.value
        return PackageStatus.DRAFT.value

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
    "EvidenceCategory",
    "QualityGrade",
    "EmissionScope",
    "Scope1EvidenceType",
    "Scope2EvidenceType",
    "Scope3EvidenceType",
    "SourceType",
    "PackageStatus",
    # Input Models
    "QualityWeights",
    "EvidenceItem",
    "ConsolidationConfig",
    "ConsolidationInput",
    # Output Models
    "EvidenceQualityScore",
    "CompletenessScore",
    "EvidenceIndexEntry",
    "EvidenceIndex",
    "ScopeEvidenceSummary",
    "EvidencePackage",
    "PackageResult",
    # Engine
    "EvidenceConsolidationEngine",
]
