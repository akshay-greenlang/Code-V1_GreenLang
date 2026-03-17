# -*- coding: utf-8 -*-
"""
Taxonomy Alignment Workflow
================================

Four-phase workflow for calculating EU Taxonomy alignment ratios for
SFDR Article 8 financial products. Orchestrates holdings analysis,
alignment assessment, portfolio aggregation, and commitment tracking
into a single auditable pipeline.

Regulatory Context:
    Per EU Taxonomy Regulation 2020/852 and SFDR RTS 2022/1288:
    - Article 5/6 of Taxonomy Regulation: Disclosure of taxonomy alignment.
    - SFDR Article 8 products must disclose taxonomy alignment where they
      commit to a minimum proportion of sustainable investments.
    - Alignment measured on revenue, CapEx, and OpEx basis.
    - Six environmental objectives: climate change mitigation, climate
      change adaptation, water/marine resources, circular economy,
      pollution prevention, biodiversity/ecosystems.
    - Alignment requires: substantial contribution to at least one objective,
      DNSH for all other objectives, minimum social safeguards compliance.
    - Double-counting prevention across environmental objectives.

    Alignment Ratios (per RTS):
    - Revenue-based: Share of investee revenue from taxonomy-aligned activities
    - CapEx-based: Share of investee capital expenditure in taxonomy-aligned activities
    - OpEx-based: Share of investee operational expenditure in taxonomy-aligned activities

Phases:
    1. HoldingsAnalysis - Map portfolio holdings to taxonomy-eligible
       activities, collect investee taxonomy data, identify data gaps
    2. AlignmentAssessment - Calculate alignment ratios per holding
       (revenue/CapEx/OpEx), apply DNSH and minimum safeguards checks
    3. Aggregation - Portfolio-level alignment ratio, breakdown by
       environmental objective, double-counting prevention
    4. CommitmentTracking - Compare actual alignment to pre-contractual
       minimum commitment, trend analysis, adherence reporting

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITIES
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class AlignmentBasis(str, Enum):
    """Basis for taxonomy alignment measurement."""
    REVENUE = "REVENUE"
    CAPEX = "CAPEX"
    OPEX = "OPEX"


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy environmental objectives."""
    CLIMATE_MITIGATION = "CLIMATE_MITIGATION"
    CLIMATE_ADAPTATION = "CLIMATE_ADAPTATION"
    WATER_MARINE = "WATER_MARINE"
    CIRCULAR_ECONOMY = "CIRCULAR_ECONOMY"
    POLLUTION_PREVENTION = "POLLUTION_PREVENTION"
    BIODIVERSITY = "BIODIVERSITY"


class EligibilityStatus(str, Enum):
    """Taxonomy eligibility status."""
    ELIGIBLE_ALIGNED = "ELIGIBLE_ALIGNED"
    ELIGIBLE_NOT_ALIGNED = "ELIGIBLE_NOT_ALIGNED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=_utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - TAXONOMY ALIGNMENT
# =============================================================================


class TaxonomyHolding(BaseModel):
    """A holding with taxonomy alignment data."""
    holding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issuer_name: str = Field(..., description="Issuer name")
    isin: Optional[str] = Field(None)
    sector: str = Field(default="")
    nace_code: str = Field(default="", description="NACE activity code")
    country: str = Field(default="")
    portfolio_weight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    market_value_eur: float = Field(default=0.0, ge=0.0)
    taxonomy_eligible_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="% of activities taxonomy-eligible"
    )
    revenue_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    capex_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    opex_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    substantial_contribution_objectives: List[str] = Field(
        default_factory=list,
        description="Environmental objectives with substantial contribution"
    )
    dnsh_compliant: bool = Field(
        default=False, description="Passes DNSH for all other objectives"
    )
    minimum_safeguards_compliant: bool = Field(
        default=False, description="Meets minimum social safeguards"
    )
    data_source: str = Field(default="company_reported")
    reporting_year: Optional[int] = Field(None)


class TaxonomyAlignmentInput(BaseModel):
    """Input configuration for the taxonomy alignment workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    reporting_date: str = Field(..., description="Reporting date YYYY-MM-DD")
    holdings: List[TaxonomyHolding] = Field(
        default_factory=list, description="Holdings with taxonomy data"
    )
    primary_alignment_basis: AlignmentBasis = Field(
        default=AlignmentBasis.REVENUE,
        description="Primary basis for alignment calculation"
    )
    precontractual_minimum_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Pre-contractual minimum taxonomy alignment commitment"
    )
    previous_period_alignment: Optional[Dict[str, Any]] = Field(
        None, description="Previous period alignment data for trending"
    )
    include_sovereign: bool = Field(
        default=False,
        description="Include sovereign bonds in alignment calculation"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate reporting date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("reporting_date must be YYYY-MM-DD format")
        return v


class TaxonomyAlignmentResult(WorkflowResult):
    """Complete result from the taxonomy alignment workflow."""
    product_name: str = Field(default="")
    total_holdings_analyzed: int = Field(default=0)
    taxonomy_eligible_pct: float = Field(default=0.0)
    revenue_aligned_pct: float = Field(default=0.0)
    capex_aligned_pct: float = Field(default=0.0)
    opex_aligned_pct: float = Field(default=0.0)
    primary_alignment_pct: float = Field(default=0.0)
    precontractual_commitment_pct: float = Field(default=0.0)
    commitment_met: bool = Field(default=False)
    data_coverage_pct: float = Field(default=0.0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class HoldingsAnalysisPhase:
    """
    Phase 1: Holdings Analysis.

    Maps portfolio holdings to taxonomy-eligible activities, collects
    investee taxonomy data, and identifies data gaps.
    """

    PHASE_NAME = "holdings_analysis"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute holdings analysis phase."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("holdings", [])

            outputs["total_holdings"] = len(holdings)

            # Eligibility assessment
            eligible_count = 0
            aligned_count = 0
            not_eligible_count = 0
            data_gap_count = 0
            total_eligible_weight = 0.0

            holding_assessments: List[Dict[str, Any]] = []

            for h in holdings:
                eligible_pct = h.get("taxonomy_eligible_pct", 0.0)
                revenue_aligned = h.get("revenue_aligned_pct", 0.0)
                weight = h.get("portfolio_weight_pct", 0.0)
                has_data = h.get("data_source", "") != ""

                if eligible_pct > 0:
                    eligible_count += 1
                    total_eligible_weight += weight * eligible_pct / 100.0
                    if revenue_aligned > 0:
                        aligned_count += 1
                        eligibility = EligibilityStatus.ELIGIBLE_ALIGNED.value
                    else:
                        eligibility = (
                            EligibilityStatus.ELIGIBLE_NOT_ALIGNED.value
                        )
                elif not has_data:
                    data_gap_count += 1
                    eligibility = EligibilityStatus.DATA_UNAVAILABLE.value
                else:
                    not_eligible_count += 1
                    eligibility = EligibilityStatus.NOT_ELIGIBLE.value

                holding_assessments.append({
                    "holding_id": h.get("holding_id", ""),
                    "issuer_name": h.get("issuer_name", ""),
                    "nace_code": h.get("nace_code", ""),
                    "portfolio_weight_pct": weight,
                    "taxonomy_eligible_pct": eligible_pct,
                    "eligibility_status": eligibility,
                    "data_source": h.get("data_source", ""),
                })

            outputs["holding_assessments"] = holding_assessments
            outputs["eligible_count"] = eligible_count
            outputs["aligned_count"] = aligned_count
            outputs["not_eligible_count"] = not_eligible_count
            outputs["data_gap_count"] = data_gap_count
            outputs["total_eligible_weight_pct"] = round(
                total_eligible_weight, 2
            )

            # Data coverage
            total = max(len(holdings), 1)
            data_coverage = round(
                (total - data_gap_count) / total * 100, 1
            )
            outputs["data_coverage_pct"] = data_coverage

            if data_gap_count > 0:
                warnings.append(
                    f"{data_gap_count} holding(s) lack taxonomy data"
                )

            # NACE sector distribution
            nace_dist: Dict[str, int] = {}
            for h in holdings:
                nace = h.get("nace_code", "Unknown")
                nace_dist[nace] = nace_dist.get(nace, 0) + 1
            outputs["nace_distribution"] = nace_dist

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error("HoldingsAnalysis failed: %s", exc, exc_info=True)
            errors.append(f"Holdings analysis failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class AlignmentAssessmentPhase:
    """
    Phase 2: Alignment Assessment.

    Calculates alignment ratios per holding on revenue, CapEx, and OpEx
    basis, and applies DNSH and minimum safeguards checks.
    """

    PHASE_NAME = "alignment_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute alignment assessment phase."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("holdings", [])

            alignment_results: List[Dict[str, Any]] = []
            dnsh_failures = 0
            safeguards_failures = 0

            for h in holdings:
                revenue_pct = h.get("revenue_aligned_pct", 0.0)
                capex_pct = h.get("capex_aligned_pct", 0.0)
                opex_pct = h.get("opex_aligned_pct", 0.0)
                dnsh = h.get("dnsh_compliant", False)
                safeguards = h.get("minimum_safeguards_compliant", False)
                objectives = h.get(
                    "substantial_contribution_objectives", []
                )
                weight = h.get("portfolio_weight_pct", 0.0)

                # Full alignment requires DNSH and safeguards
                fully_aligned = dnsh and safeguards
                if not dnsh:
                    dnsh_failures += 1
                if not safeguards:
                    safeguards_failures += 1

                # Effective alignment (zero if DNSH/safeguards fail)
                effective_revenue = revenue_pct if fully_aligned else 0.0
                effective_capex = capex_pct if fully_aligned else 0.0
                effective_opex = opex_pct if fully_aligned else 0.0

                alignment_results.append({
                    "holding_id": h.get("holding_id", ""),
                    "issuer_name": h.get("issuer_name", ""),
                    "portfolio_weight_pct": weight,
                    "revenue_aligned_pct": revenue_pct,
                    "capex_aligned_pct": capex_pct,
                    "opex_aligned_pct": opex_pct,
                    "dnsh_compliant": dnsh,
                    "minimum_safeguards_compliant": safeguards,
                    "fully_aligned": fully_aligned,
                    "effective_revenue_pct": effective_revenue,
                    "effective_capex_pct": effective_capex,
                    "effective_opex_pct": effective_opex,
                    "substantial_contribution_objectives": objectives,
                })

            outputs["alignment_results"] = alignment_results
            outputs["dnsh_failure_count"] = dnsh_failures
            outputs["safeguards_failure_count"] = safeguards_failures

            if dnsh_failures > 0:
                warnings.append(
                    f"{dnsh_failures} holding(s) fail DNSH assessment"
                )
            if safeguards_failures > 0:
                warnings.append(
                    f"{safeguards_failures} holding(s) fail minimum "
                    f"safeguards compliance"
                )

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error(
                "AlignmentAssessment failed: %s", exc, exc_info=True
            )
            errors.append(f"Alignment assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class AggregationPhase:
    """
    Phase 3: Aggregation.

    Computes portfolio-level alignment ratios, breaks down by environmental
    objective, and applies double-counting prevention.
    """

    PHASE_NAME = "aggregation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute aggregation phase."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            assessment_output = context.get_phase_output(
                "alignment_assessment"
            )
            results = assessment_output.get("alignment_results", [])

            # Portfolio-weighted alignment ratios
            total_weight = sum(
                r.get("portfolio_weight_pct", 0.0) for r in results
            )
            if total_weight <= 0:
                total_weight = 100.0

            revenue_aligned = 0.0
            capex_aligned = 0.0
            opex_aligned = 0.0

            objective_breakdown: Dict[str, float] = {
                obj.value: 0.0 for obj in EnvironmentalObjective
            }

            for r in results:
                weight = r.get("portfolio_weight_pct", 0.0) / total_weight
                revenue_aligned += (
                    r.get("effective_revenue_pct", 0.0) * weight
                )
                capex_aligned += (
                    r.get("effective_capex_pct", 0.0) * weight
                )
                opex_aligned += (
                    r.get("effective_opex_pct", 0.0) * weight
                )

                # Breakdown by objective (avoid double-counting)
                objectives = r.get(
                    "substantial_contribution_objectives", []
                )
                if objectives and r.get("fully_aligned", False):
                    # Attribute to primary objective only
                    primary = objectives[0] if objectives else None
                    if primary and primary in objective_breakdown:
                        objective_breakdown[primary] += (
                            r.get("effective_revenue_pct", 0.0) * weight
                        )

            outputs["portfolio_revenue_aligned_pct"] = round(
                revenue_aligned, 2
            )
            outputs["portfolio_capex_aligned_pct"] = round(
                capex_aligned, 2
            )
            outputs["portfolio_opex_aligned_pct"] = round(
                opex_aligned, 2
            )

            # Round objective breakdown
            outputs["objective_breakdown"] = {
                k: round(v, 2) for k, v in objective_breakdown.items()
            }

            # Alignment pie chart data
            not_eligible_pct = 100.0 - sum(
                r.get("portfolio_weight_pct", 0.0) / total_weight * 100
                for r in results
                if r.get("effective_revenue_pct", 0) > 0
            )
            outputs["pie_chart_data"] = {
                "taxonomy_aligned_pct": round(revenue_aligned, 2),
                "eligible_not_aligned_pct": round(
                    max(0, 100.0 - revenue_aligned - not_eligible_pct), 2
                ),
                "not_taxonomy_eligible_pct": round(
                    max(0, not_eligible_pct), 2
                ),
            }

            # Double-counting check
            total_objective_sum = sum(objective_breakdown.values())
            if total_objective_sum > revenue_aligned * 1.01:
                warnings.append(
                    f"Potential double-counting detected: objective sum "
                    f"({total_objective_sum:.2f}%) exceeds portfolio "
                    f"alignment ({revenue_aligned:.2f}%)"
                )
                outputs["double_counting_flag"] = True
            else:
                outputs["double_counting_flag"] = False

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Aggregation failed: %s", exc, exc_info=True)
            errors.append(f"Aggregation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


class CommitmentTrackingPhase:
    """
    Phase 4: Commitment Tracking.

    Compares actual alignment to the pre-contractual minimum commitment,
    performs trend analysis, and generates adherence reporting.
    """

    PHASE_NAME = "commitment_tracking"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute commitment tracking phase."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            aggregation_output = context.get_phase_output("aggregation")
            analysis_output = context.get_phase_output("holdings_analysis")

            commitment_pct = config.get("precontractual_minimum_pct", 0.0)
            primary_basis = config.get(
                "primary_alignment_basis", AlignmentBasis.REVENUE.value
            )
            previous_data = config.get("previous_period_alignment")

            # Get actual alignment based on primary basis
            if primary_basis == AlignmentBasis.REVENUE.value:
                actual_pct = aggregation_output.get(
                    "portfolio_revenue_aligned_pct", 0.0
                )
            elif primary_basis == AlignmentBasis.CAPEX.value:
                actual_pct = aggregation_output.get(
                    "portfolio_capex_aligned_pct", 0.0
                )
            else:
                actual_pct = aggregation_output.get(
                    "portfolio_opex_aligned_pct", 0.0
                )

            outputs["primary_alignment_basis"] = primary_basis
            outputs["actual_alignment_pct"] = actual_pct
            outputs["precontractual_commitment_pct"] = commitment_pct
            outputs["variance_pct"] = round(actual_pct - commitment_pct, 2)
            outputs["commitment_met"] = actual_pct >= commitment_pct

            if actual_pct < commitment_pct:
                warnings.append(
                    f"Taxonomy alignment ({actual_pct:.2f}%) is below "
                    f"pre-contractual commitment ({commitment_pct:.2f}%)"
                )

            # Trend analysis
            if previous_data:
                prev_actual = previous_data.get("actual_alignment_pct", 0.0)
                trend_change = actual_pct - prev_actual
                outputs["trend_analysis"] = {
                    "previous_period_pct": prev_actual,
                    "current_period_pct": actual_pct,
                    "change_pct": round(trend_change, 2),
                    "direction": (
                        "improving" if trend_change > 0
                        else "declining" if trend_change < 0
                        else "stable"
                    ),
                }
            else:
                outputs["trend_analysis"] = {
                    "previous_period_pct": None,
                    "current_period_pct": actual_pct,
                    "change_pct": None,
                    "direction": "first_period",
                }

            # Adherence report
            outputs["adherence_report"] = {
                "report_id": str(uuid.uuid4()),
                "generated_at": _utcnow().isoformat(),
                "product_name": config.get("product_name", ""),
                "reporting_date": config.get("reporting_date", ""),
                "alignment_ratios": {
                    "revenue_pct": aggregation_output.get(
                        "portfolio_revenue_aligned_pct", 0.0
                    ),
                    "capex_pct": aggregation_output.get(
                        "portfolio_capex_aligned_pct", 0.0
                    ),
                    "opex_pct": aggregation_output.get(
                        "portfolio_opex_aligned_pct", 0.0
                    ),
                },
                "commitment": {
                    "committed_pct": commitment_pct,
                    "actual_pct": actual_pct,
                    "met": actual_pct >= commitment_pct,
                    "basis": primary_basis,
                },
                "objective_breakdown": aggregation_output.get(
                    "objective_breakdown", {}
                ),
                "data_coverage_pct": analysis_output.get(
                    "data_coverage_pct", 0.0
                ),
                "double_counting_flag": aggregation_output.get(
                    "double_counting_flag", False
                ),
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "CommitmentTracking failed: %s", exc, exc_info=True
            )
            errors.append(f"Commitment tracking failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class TaxonomyAlignmentWorkflow:
    """
    Four-phase taxonomy alignment workflow for SFDR Article 8.

    Orchestrates holdings analysis through commitment tracking for
    EU Taxonomy alignment ratio calculation.

    Example:
        >>> wf = TaxonomyAlignmentWorkflow()
        >>> input_data = TaxonomyAlignmentInput(
        ...     organization_id="org-123",
        ...     product_name="Green Bond Fund",
        ...     reporting_date="2026-01-15",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "taxonomy_alignment"

    PHASE_ORDER = [
        "holdings_analysis",
        "alignment_assessment",
        "aggregation",
        "commitment_tracking",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize the taxonomy alignment workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "holdings_analysis": HoldingsAnalysisPhase(),
            "alignment_assessment": AlignmentAssessmentPhase(),
            "aggregation": AggregationPhase(),
            "commitment_tracking": CommitmentTrackingPhase(),
        }

    async def run(
        self, input_data: TaxonomyAlignmentInput
    ) -> TaxonomyAlignmentResult:
        """Execute the complete 4-phase taxonomy alignment workflow."""
        started_at = _utcnow()
        logger.info(
            "Starting taxonomy alignment workflow %s for org=%s product=%s",
            self.workflow_id, input_data.organization_id,
            input_data.product_name,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name == "holdings_analysis":
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=_utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = _utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )

        return TaxonomyAlignmentResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            total_holdings_analyzed=summary.get(
                "total_holdings_analyzed", 0
            ),
            taxonomy_eligible_pct=summary.get(
                "taxonomy_eligible_pct", 0.0
            ),
            revenue_aligned_pct=summary.get("revenue_aligned_pct", 0.0),
            capex_aligned_pct=summary.get("capex_aligned_pct", 0.0),
            opex_aligned_pct=summary.get("opex_aligned_pct", 0.0),
            primary_alignment_pct=summary.get(
                "primary_alignment_pct", 0.0
            ),
            precontractual_commitment_pct=summary.get(
                "precontractual_commitment_pct", 0.0
            ),
            commitment_met=summary.get("commitment_met", False),
            data_coverage_pct=summary.get("data_coverage_pct", 0.0),
        )

    def _build_config(
        self, input_data: TaxonomyAlignmentInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        config["primary_alignment_basis"] = (
            input_data.primary_alignment_basis.value
        )
        if input_data.holdings:
            config["holdings"] = [
                h.model_dump() for h in input_data.holdings
            ]
        return config

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        analysis = context.get_phase_output("holdings_analysis")
        aggregation = context.get_phase_output("aggregation")
        tracking = context.get_phase_output("commitment_tracking")

        return {
            "product_name": config.get("product_name", ""),
            "total_holdings_analyzed": analysis.get("total_holdings", 0),
            "taxonomy_eligible_pct": analysis.get(
                "total_eligible_weight_pct", 0.0
            ),
            "revenue_aligned_pct": aggregation.get(
                "portfolio_revenue_aligned_pct", 0.0
            ),
            "capex_aligned_pct": aggregation.get(
                "portfolio_capex_aligned_pct", 0.0
            ),
            "opex_aligned_pct": aggregation.get(
                "portfolio_opex_aligned_pct", 0.0
            ),
            "primary_alignment_pct": tracking.get(
                "actual_alignment_pct", 0.0
            ),
            "precontractual_commitment_pct": tracking.get(
                "precontractual_commitment_pct", 0.0
            ),
            "commitment_met": tracking.get("commitment_met", False),
            "data_coverage_pct": analysis.get("data_coverage_pct", 0.0),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
