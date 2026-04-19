# -*- coding: utf-8 -*-
"""
Scope 3 Assessment Workflow
================================

4-phase workflow for Scope 3 target design within PACK-023 SBTi
Alignment Pack.  The workflow screens all 15 categories with data
quality scoring, calculates materiality using Pareto analysis,
checks coverage against 67%/90% requirements, and designs
category-specific targets (absolute/intensity/engagement).

Phases:
    1. CategoryScreen    -- Screen all 15 categories with data quality scoring
    2. MaterialityCalc   -- Calculate materiality (% of total, Pareto analysis)
    3. CoverageCheck     -- Check coverage against 67%/90% requirements
    4. TargetDesign      -- Design category-specific targets

Regulatory references:
    - SBTi Corporate Manual V5.3 (2024): Scope 3 criteria C17-C20
    - GHG Protocol Scope 3 Standard (2011): 15 categories
    - SBTi Scope 3 Optional Guidance (2024)

Zero-hallucination: all materiality calculations, coverage checks,
and Pareto analysis use deterministic arithmetic.  No LLM calls
in the numeric computation path.

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"

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

class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

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

class DataQuality(str, Enum):
    """Data quality tier for Scope 3 categories."""

    PRIMARY = "primary"         # Supplier-specific data
    SECONDARY = "secondary"     # Industry average data
    PROXY = "proxy"             # Proxy or estimated data
    SPEND = "spend"             # Spend-based estimates
    NOT_ESTIMATED = "not_estimated"

class Scope3TargetType(str, Enum):
    """Target type for individual Scope 3 categories."""

    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    ENGAGEMENT = "engagement"
    NOT_TARGETED = "not_targeted"

# =============================================================================
# REFERENCE DATA
# =============================================================================

SCOPE3_CATEGORIES: Dict[int, str] = {
    1: "Purchased Goods & Services",
    2: "Capital Goods",
    3: "Fuel & Energy Activities",
    4: "Upstream Transportation",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

# Recommended target approach by category (SBTi guidance)
CATEGORY_TARGET_APPROACH: Dict[int, str] = {
    1: "engagement",    # Supplier engagement for purchased goods
    2: "absolute",      # Absolute for capital goods
    3: "absolute",      # Absolute for fuel & energy
    4: "intensity",     # Intensity for transportation (tCO2e/tkm)
    5: "absolute",      # Absolute for waste
    6: "intensity",     # Intensity for business travel (tCO2e/FTE)
    7: "intensity",     # Intensity for commuting (tCO2e/FTE)
    8: "absolute",      # Absolute for leased assets
    9: "intensity",     # Intensity for downstream transport
    10: "absolute",     # Absolute for processing
    11: "absolute",     # Absolute for use of sold products
    12: "absolute",     # Absolute for end-of-life
    13: "absolute",     # Absolute for downstream leased
    14: "engagement",   # Engagement for franchises
    15: "engagement",   # Engagement for investments
}

NEAR_TERM_COVERAGE_MIN = 67.0   # % of Scope 3
LONG_TERM_COVERAGE_MIN = 90.0   # % of Scope 3
MATERIALITY_TRIGGER = 40.0      # S3 as % of total
ENGAGEMENT_MIN_COVERAGE = 80.0  # % of suppliers for engagement targets

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

class CategoryData(BaseModel):
    """Input data for a single Scope 3 category."""

    category_id: int = Field(..., ge=1, le=15)
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    data_quality: DataQuality = Field(default=DataQuality.NOT_ESTIMATED)
    supplier_count: int = Field(default=0, ge=0)
    data_source: str = Field(default="")
    notes: str = Field(default="")

class CategoryScreenResult(BaseModel):
    """Screening result for a single Scope 3 category."""

    category_id: int = Field(..., ge=1, le=15)
    category_name: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0)
    pct_of_scope3: float = Field(default=0.0)
    pct_of_total: float = Field(default=0.0)
    cumulative_pct: float = Field(default=0.0, description="Cumulative % for Pareto")
    data_quality: DataQuality = Field(default=DataQuality.NOT_ESTIMATED)
    data_quality_score: int = Field(default=0, ge=0, le=5)
    is_material: bool = Field(default=False)
    pareto_rank: int = Field(default=0)
    recommended_target_type: str = Field(default="")

class MaterialityResult(BaseModel):
    """Materiality analysis summary."""

    scope3_total_tco2e: float = Field(default=0.0)
    scope3_pct_of_total: float = Field(default=0.0)
    scope3_target_required: bool = Field(default=False)
    material_categories: List[int] = Field(default_factory=list)
    material_emissions_tco2e: float = Field(default=0.0)
    pareto_80_categories: List[int] = Field(default_factory=list)
    top_3_categories: List[int] = Field(default_factory=list)

class CoverageResult(BaseModel):
    """Coverage assessment against SBTi requirements."""

    near_term_coverage_pct: float = Field(default=0.0)
    near_term_meets_min: bool = Field(default=False)
    long_term_coverage_pct: float = Field(default=0.0)
    long_term_meets_min: bool = Field(default=False)
    categories_in_target: List[int] = Field(default_factory=list)
    additional_needed_for_67: List[int] = Field(default_factory=list)
    additional_needed_for_90: List[int] = Field(default_factory=list)

class CategoryTarget(BaseModel):
    """Target design for a single Scope 3 category."""

    category_id: int = Field(..., ge=1, le=15)
    category_name: str = Field(default="")
    target_type: Scope3TargetType = Field(default=Scope3TargetType.NOT_TARGETED)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_year: int = Field(default=2030, ge=2025)
    engagement_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    base_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    rationale: str = Field(default="")

class Scope3AssessmentConfig(BaseModel):
    """Configuration for the Scope 3 assessment workflow."""

    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    categories: List[CategoryData] = Field(default_factory=list)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    long_term_target_year: int = Field(default=2040, ge=2035, le=2060)
    annual_reduction_rate: float = Field(default=0.025, ge=0.0, le=0.20)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("categories")
    @classmethod
    def _validate_categories(cls, v: List[CategoryData]) -> List[CategoryData]:
        seen: set = set()
        for cat in v:
            if cat.category_id in seen:
                raise ValueError(f"Duplicate category_id: {cat.category_id}")
            seen.add(cat.category_id)
        return v

class Scope3AssessmentResult(BaseModel):
    """Complete result from the Scope 3 assessment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="scope3_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    screening_results: List[CategoryScreenResult] = Field(default_factory=list)
    materiality: Optional[MaterialityResult] = Field(None)
    coverage: Optional[CoverageResult] = Field(None)
    category_targets: List[CategoryTarget] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class Scope3AssessmentWorkflow:
    """
    4-phase Scope 3 assessment workflow for SBTi target design.

    Screens all 15 Scope 3 categories with data quality scoring,
    calculates materiality using Pareto analysis, checks coverage
    against 67%/90% requirements, and designs category-specific
    targets.

    Zero-hallucination: all materiality calculations, Pareto
    analysis, and coverage checks use deterministic arithmetic.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = Scope3AssessmentWorkflow()
        >>> config = Scope3AssessmentConfig(
        ...     scope1_tco2e=5000, scope2_tco2e=3000,
        ...     categories=[CategoryData(category_id=1, emissions_tco2e=20000)],
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.materiality is not None
    """

    def __init__(self) -> None:
        """Initialise Scope3AssessmentWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._screening: List[CategoryScreenResult] = []
        self._materiality: Optional[MaterialityResult] = None
        self._coverage: Optional[CoverageResult] = None
        self._targets: List[CategoryTarget] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: Scope3AssessmentConfig) -> Scope3AssessmentResult:
        """
        Execute the 4-phase Scope 3 assessment workflow.

        Args:
            config: Scope 3 assessment configuration with category data.

        Returns:
            Scope3AssessmentResult with screening, materiality, coverage,
            and category targets.
        """
        started_at = utcnow()
        self.logger.info(
            "Starting Scope 3 assessment workflow %s, categories=%d",
            self.workflow_id, len(config.categories),
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_category_screen(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_materiality_calc(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_coverage_check(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_target_design(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Scope 3 assessment failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = Scope3AssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            screening_results=self._screening,
            materiality=self._materiality,
            coverage=self._coverage,
            category_targets=self._targets,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Scope 3 assessment %s completed in %.2fs, targets=%d",
            self.workflow_id, elapsed, len(self._targets),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Category Screen
    # -------------------------------------------------------------------------

    async def _phase_category_screen(self, config: Scope3AssessmentConfig) -> PhaseResult:
        """Screen all 15 categories with data quality scoring."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._screening = []

        # Build lookup from provided categories
        cat_map: Dict[int, CategoryData] = {c.category_id: c for c in config.categories}

        scope3_total = sum(c.emissions_tco2e for c in config.categories)
        scope12_total = config.scope1_tco2e + config.scope2_tco2e
        total_emissions = scope12_total + scope3_total

        for cat_id in range(1, 16):
            cat_data = cat_map.get(cat_id)
            emissions = cat_data.emissions_tco2e if cat_data else 0.0
            dq = cat_data.data_quality if cat_data else DataQuality.NOT_ESTIMATED

            pct_of_scope3 = (emissions / scope3_total * 100.0) if scope3_total > 0 else 0.0
            pct_of_total = (emissions / total_emissions * 100.0) if total_emissions > 0 else 0.0

            # Data quality score (1-5 scale, 5 = best)
            dq_score = self._quality_score(dq)

            # Material if >1% of total emissions or >5% of Scope 3
            is_material = pct_of_total > 1.0 or pct_of_scope3 > 5.0

            recommended = CATEGORY_TARGET_APPROACH.get(cat_id, "absolute")

            self._screening.append(CategoryScreenResult(
                category_id=cat_id,
                category_name=SCOPE3_CATEGORIES.get(cat_id, f"Category {cat_id}"),
                emissions_tco2e=round(emissions, 2),
                pct_of_scope3=round(pct_of_scope3, 2),
                pct_of_total=round(pct_of_total, 2),
                data_quality=dq,
                data_quality_score=dq_score,
                is_material=is_material,
                recommended_target_type=recommended,
            ))

        # Sort by emissions descending for Pareto ranking
        self._screening.sort(key=lambda c: c.emissions_tco2e, reverse=True)

        cumulative = 0.0
        for rank, cat in enumerate(self._screening, 1):
            cumulative += cat.pct_of_scope3
            cat.cumulative_pct = round(cumulative, 2)
            cat.pareto_rank = rank

        # Identify categories not estimated
        not_estimated = [c.category_id for c in self._screening if c.data_quality == DataQuality.NOT_ESTIMATED]
        if not_estimated:
            warnings.append(f"Categories not estimated: {not_estimated}")

        outputs["categories_screened"] = 15
        outputs["categories_with_data"] = 15 - len(not_estimated)
        outputs["scope3_total_tco2e"] = round(scope3_total, 2)
        outputs["total_emissions_tco2e"] = round(total_emissions, 2)
        outputs["material_count"] = sum(1 for c in self._screening if c.is_material)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Category screen: %d categories, S3=%.2f tCO2e", 15, scope3_total)
        return PhaseResult(
            phase_name="category_screen",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _quality_score(self, dq: DataQuality) -> int:
        """Map data quality tier to numeric score."""
        mapping = {
            DataQuality.PRIMARY: 5,
            DataQuality.SECONDARY: 4,
            DataQuality.PROXY: 3,
            DataQuality.SPEND: 2,
            DataQuality.NOT_ESTIMATED: 1,
        }
        return mapping.get(dq, 1)

    # -------------------------------------------------------------------------
    # Phase 2: Materiality Calculation
    # -------------------------------------------------------------------------

    async def _phase_materiality_calc(self, config: Scope3AssessmentConfig) -> PhaseResult:
        """Calculate materiality (% of total, Pareto analysis)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        scope3_total = sum(c.emissions_tco2e for c in self._screening)
        scope12_total = config.scope1_tco2e + config.scope2_tco2e
        total_emissions = scope12_total + scope3_total

        scope3_pct = (scope3_total / total_emissions * 100.0) if total_emissions > 0 else 0.0
        scope3_required = scope3_pct >= MATERIALITY_TRIGGER

        material_cats = [c.category_id for c in self._screening if c.is_material]
        material_emissions = sum(c.emissions_tco2e for c in self._screening if c.is_material)

        # Pareto 80/20: categories covering 80% of Scope 3
        pareto_cats: List[int] = []
        for cat in self._screening:
            pareto_cats.append(cat.category_id)
            if cat.cumulative_pct >= 80.0:
                break

        # Top 3 by emissions
        top_3 = [c.category_id for c in self._screening[:3]]

        self._materiality = MaterialityResult(
            scope3_total_tco2e=round(scope3_total, 2),
            scope3_pct_of_total=round(scope3_pct, 2),
            scope3_target_required=scope3_required,
            material_categories=material_cats,
            material_emissions_tco2e=round(material_emissions, 2),
            pareto_80_categories=pareto_cats,
            top_3_categories=top_3,
        )

        outputs["scope3_pct_of_total"] = round(scope3_pct, 2)
        outputs["scope3_target_required"] = scope3_required
        outputs["material_categories"] = len(material_cats)
        outputs["pareto_80_categories"] = len(pareto_cats)
        outputs["material_emissions_tco2e"] = round(material_emissions, 2)
        outputs["top_3_categories"] = top_3

        if not scope3_required:
            warnings.append(
                f"Scope 3 is {scope3_pct:.1f}% of total (<{MATERIALITY_TRIGGER}%); "
                "target not required but recommended"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Materiality: S3=%.1f%%, required=%s, material=%d, Pareto80=%d",
            scope3_pct, scope3_required, len(material_cats), len(pareto_cats),
        )
        return PhaseResult(
            phase_name="materiality_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Coverage Check
    # -------------------------------------------------------------------------

    async def _phase_coverage_check(self, config: Scope3AssessmentConfig) -> PhaseResult:
        """Check coverage against 67%/90% requirements."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        scope3_total = sum(c.emissions_tco2e for c in self._screening)

        # Start with material categories as included in target
        included_cats = [c.category_id for c in self._screening if c.is_material]
        included_emissions = sum(
            c.emissions_tco2e for c in self._screening if c.is_material
        )
        coverage_pct = (included_emissions / scope3_total * 100.0) if scope3_total > 0 else 0.0

        # Check near-term (67%)
        nt_meets = coverage_pct >= NEAR_TERM_COVERAGE_MIN
        additional_67: List[int] = []
        if not nt_meets:
            # Add more categories by Pareto order
            temp_coverage = coverage_pct
            temp_emissions = included_emissions
            for cat in self._screening:
                if cat.category_id not in included_cats and cat.emissions_tco2e > 0:
                    temp_emissions += cat.emissions_tco2e
                    temp_coverage = (temp_emissions / scope3_total * 100.0) if scope3_total > 0 else 0.0
                    additional_67.append(cat.category_id)
                    if temp_coverage >= NEAR_TERM_COVERAGE_MIN:
                        break

        # Check long-term (90%)
        lt_coverage_emissions = included_emissions + sum(
            c.emissions_tco2e for c in self._screening
            if c.category_id in additional_67
        )
        lt_coverage_pct = (lt_coverage_emissions / scope3_total * 100.0) if scope3_total > 0 else 0.0
        lt_meets = lt_coverage_pct >= LONG_TERM_COVERAGE_MIN

        additional_90: List[int] = []
        if not lt_meets:
            temp_emissions_lt = lt_coverage_emissions
            included_plus_67 = set(included_cats) | set(additional_67)
            for cat in self._screening:
                if cat.category_id not in included_plus_67 and cat.emissions_tco2e > 0:
                    temp_emissions_lt += cat.emissions_tco2e
                    additional_90.append(cat.category_id)
                    temp_cov = (temp_emissions_lt / scope3_total * 100.0) if scope3_total > 0 else 0.0
                    if temp_cov >= LONG_TERM_COVERAGE_MIN:
                        break

        self._coverage = CoverageResult(
            near_term_coverage_pct=round(coverage_pct, 2),
            near_term_meets_min=nt_meets,
            long_term_coverage_pct=round(lt_coverage_pct, 2),
            long_term_meets_min=lt_meets,
            categories_in_target=included_cats,
            additional_needed_for_67=additional_67,
            additional_needed_for_90=additional_90,
        )

        outputs["near_term_coverage_pct"] = round(coverage_pct, 2)
        outputs["near_term_meets_67"] = nt_meets
        outputs["long_term_coverage_pct"] = round(lt_coverage_pct, 2)
        outputs["long_term_meets_90"] = lt_meets
        outputs["categories_in_target"] = len(included_cats)
        outputs["additional_for_67"] = len(additional_67)
        outputs["additional_for_90"] = len(additional_90)

        if not nt_meets:
            warnings.append(
                f"Near-term coverage {coverage_pct:.1f}% < {NEAR_TERM_COVERAGE_MIN}%; "
                f"add categories {additional_67}"
            )
        if not lt_meets:
            warnings.append(
                f"Long-term coverage {lt_coverage_pct:.1f}% < {LONG_TERM_COVERAGE_MIN}%; "
                f"add categories {additional_90}"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Coverage: NT=%.1f%% (meets=%s), LT=%.1f%% (meets=%s)",
            coverage_pct, nt_meets, lt_coverage_pct, lt_meets,
        )
        return PhaseResult(
            phase_name="coverage_check",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Target Design
    # -------------------------------------------------------------------------

    async def _phase_target_design(self, config: Scope3AssessmentConfig) -> PhaseResult:
        """Design category-specific targets (absolute/intensity/engagement)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._targets = []

        # Categories to include in target
        all_included = set()
        if self._coverage:
            all_included.update(self._coverage.categories_in_target)
            all_included.update(self._coverage.additional_needed_for_67)

        rate = config.annual_reduction_rate
        nt_years = config.near_term_target_year - utcnow().year

        for cat in self._screening:
            if cat.category_id not in all_included:
                self._targets.append(CategoryTarget(
                    category_id=cat.category_id,
                    category_name=cat.category_name,
                    target_type=Scope3TargetType.NOT_TARGETED,
                    base_emissions_tco2e=round(cat.emissions_tco2e, 2),
                    rationale="Category not material or below coverage threshold",
                ))
                continue

            # Determine target type
            recommended = CATEGORY_TARGET_APPROACH.get(cat.category_id, "absolute")
            if recommended == "engagement":
                target_type = Scope3TargetType.ENGAGEMENT
            elif recommended == "intensity":
                target_type = Scope3TargetType.INTENSITY
            else:
                target_type = Scope3TargetType.ABSOLUTE

            # Calculate reduction
            if nt_years > 0:
                reduction_pct = min(1.0 - (1.0 - rate) ** nt_years, 0.90) * 100.0
            else:
                reduction_pct = rate * 100.0

            target_emissions = cat.emissions_tco2e * (1.0 - reduction_pct / 100.0)

            # Engagement coverage
            engagement_cov = ENGAGEMENT_MIN_COVERAGE if target_type == Scope3TargetType.ENGAGEMENT else 0.0

            rationale = self._build_rationale(cat, target_type, reduction_pct)

            self._targets.append(CategoryTarget(
                category_id=cat.category_id,
                category_name=cat.category_name,
                target_type=target_type,
                reduction_pct=round(reduction_pct, 2),
                target_year=config.near_term_target_year,
                engagement_coverage_pct=engagement_cov,
                base_emissions_tco2e=round(cat.emissions_tco2e, 2),
                target_emissions_tco2e=round(target_emissions, 2),
                rationale=rationale,
            ))

        targeted_count = sum(1 for t in self._targets if t.target_type != Scope3TargetType.NOT_TARGETED)
        total_reduction = sum(
            t.base_emissions_tco2e - t.target_emissions_tco2e
            for t in self._targets if t.target_type != Scope3TargetType.NOT_TARGETED
        )

        outputs["categories_targeted"] = targeted_count
        outputs["categories_not_targeted"] = len(self._targets) - targeted_count
        outputs["total_targeted_reduction_tco2e"] = round(total_reduction, 2)
        outputs["absolute_targets"] = sum(1 for t in self._targets if t.target_type == Scope3TargetType.ABSOLUTE)
        outputs["intensity_targets"] = sum(1 for t in self._targets if t.target_type == Scope3TargetType.INTENSITY)
        outputs["engagement_targets"] = sum(1 for t in self._targets if t.target_type == Scope3TargetType.ENGAGEMENT)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Target design: %d targeted, reduction=%.2f tCO2e",
            targeted_count, total_reduction,
        )
        return PhaseResult(
            phase_name="target_design",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _build_rationale(
        self, cat: CategoryScreenResult, target_type: Scope3TargetType,
        reduction_pct: float,
    ) -> str:
        """Build rationale string for category target."""
        parts = [
            f"Cat {cat.category_id} ({cat.category_name}): "
            f"{cat.pct_of_scope3:.1f}% of S3, Pareto rank #{cat.pareto_rank}.",
        ]
        if target_type == Scope3TargetType.ENGAGEMENT:
            parts.append(
                f"Engagement target: {ENGAGEMENT_MIN_COVERAGE}% supplier coverage "
                f"with {reduction_pct:.1f}% reduction."
            )
        elif target_type == Scope3TargetType.INTENSITY:
            parts.append(f"Intensity target: {reduction_pct:.1f}% reduction.")
        else:
            parts.append(f"Absolute target: {reduction_pct:.1f}% reduction.")
        return " ".join(parts)
