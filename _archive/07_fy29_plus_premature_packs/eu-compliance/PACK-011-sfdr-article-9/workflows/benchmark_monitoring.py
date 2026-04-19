# -*- coding: utf-8 -*-
"""
Benchmark Monitoring Workflow
================================================

Four-phase workflow for EU Climate Benchmark monitoring under SFDR Article 9.
Orchestrates benchmark selection, alignment assessment, deviation analysis,
and trajectory projection into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 9 products designating an EU Climate Benchmark (CTB or PAB)
      must continuously monitor alignment with the benchmark methodology.
    - EU Climate Transition Benchmarks (CTB) require a minimum 30% GHG
      intensity reduction relative to the investable universe, plus 7%
      year-on-year decarbonization trajectory.
    - Paris-Aligned Benchmarks (PAB) require a minimum 50% GHG intensity
      reduction, plus 7% year-on-year decarbonization, plus sector-specific
      exclusions (fossil fuels, controversial weapons, tobacco).
    - Deviations from benchmark alignment must be documented with root cause
      analysis and remediation plans.
    - Forward-looking trajectory projections must demonstrate continued
      alignment with temperature pathways.

Phases:
    1. BenchmarkSelection - Identify and validate designated EU Climate Benchmark
    2. AlignmentAssessment - Measure portfolio alignment against benchmark criteria
    3. DeviationAnalysis - Identify and analyze deviations from benchmark targets
    4. TrajectoryProjection - Project forward-looking alignment trajectory

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# UTILITIES
# =============================================================================

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

class BenchmarkType(str, Enum):
    """EU Climate Benchmark type."""
    CTB = "CTB"
    PAB = "PAB"
    CUSTOM = "CUSTOM"
    NONE = "NONE"

class AlignmentStatus(str, Enum):
    """Portfolio alignment status relative to benchmark."""
    ALIGNED = "ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    MISALIGNED = "MISALIGNED"
    NOT_ASSESSED = "NOT_ASSESSED"

class DeviationSeverity(str, Enum):
    """Severity classification for benchmark deviations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

# =============================================================================
# DATA MODELS - SHARED
# =============================================================================

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
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
# DATA MODELS - BENCHMARK MONITORING
# =============================================================================

class BenchmarkMonitoringInput(BaseModel):
    """Input configuration for the benchmark monitoring workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None, description="ISIN if applicable")
    reporting_date: str = Field(
        ..., description="Assessment date YYYY-MM-DD"
    )
    benchmark_type: str = Field(
        default=BenchmarkType.CTB.value,
        description="EU Climate Benchmark type (CTB or PAB)"
    )
    benchmark_name: Optional[str] = Field(
        None, description="Designated benchmark name"
    )
    benchmark_provider: Optional[str] = Field(
        None, description="Benchmark administrator/provider"
    )
    portfolio_carbon_intensity: float = Field(
        default=0.0, ge=0.0,
        description="Current portfolio weighted average carbon intensity"
    )
    universe_carbon_intensity: float = Field(
        default=0.0, ge=0.0,
        description="Investable universe carbon intensity"
    )
    portfolio_ghg_emissions: float = Field(
        default=0.0, ge=0.0,
        description="Total portfolio GHG emissions (tCO2e)"
    )
    prior_year_carbon_intensity: Optional[float] = Field(
        None, ge=0.0,
        description="Prior year carbon intensity for trajectory"
    )
    decarbonization_target_pct: float = Field(
        default=7.0, ge=0.0, le=100.0,
        description="Annual decarbonization rate target (default 7%)"
    )
    sector_exposures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Portfolio sector exposure breakdown"
    )
    exclusion_compliance: Dict[str, bool] = Field(
        default_factory=dict,
        description="Compliance with benchmark exclusion criteria"
    )
    projection_years: int = Field(
        default=5, ge=1, le=30,
        description="Number of years for trajectory projection"
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

class BenchmarkMonitoringResult(WorkflowResult):
    """Complete result from the benchmark monitoring workflow."""
    product_name: str = Field(default="")
    benchmark_type: str = Field(default="CTB")
    benchmark_name: str = Field(default="")
    alignment_status: str = Field(default="NOT_ASSESSED")
    ghg_reduction_pct: float = Field(default=0.0)
    required_reduction_pct: float = Field(default=30.0)
    alignment_gap_pct: float = Field(default=0.0)
    deviations_found: int = Field(default=0)
    critical_deviations: int = Field(default=0)
    yoy_decarbonization_pct: float = Field(default=0.0)
    projected_alignment_year: Optional[int] = Field(None)
    on_track: bool = Field(default=False)

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class BenchmarkSelectionPhase:
    """
    Phase 1: Benchmark Selection.

    Identifies and validates the designated EU Climate Benchmark,
    confirms benchmark type requirements (CTB vs PAB), and establishes
    the applicable reduction thresholds and exclusion criteria.
    """

    PHASE_NAME = "benchmark_selection"

    # Regulatory thresholds per benchmark type
    BENCHMARK_REQUIREMENTS = {
        BenchmarkType.CTB.value: {
            "minimum_ghg_reduction_pct": 30.0,
            "annual_decarbonization_pct": 7.0,
            "description": "EU Climate Transition Benchmark",
            "exclusions_required": [
                "controversial_weapons",
            ],
        },
        BenchmarkType.PAB.value: {
            "minimum_ghg_reduction_pct": 50.0,
            "annual_decarbonization_pct": 7.0,
            "description": "EU Paris-Aligned Benchmark",
            "exclusions_required": [
                "controversial_weapons",
                "tobacco",
                "coal_revenue_1pct",
                "oil_revenue_10pct",
                "gas_revenue_50pct",
                "high_intensity_electricity",
            ],
        },
    }

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute benchmark selection phase.

        Args:
            context: Workflow context with benchmark configuration.

        Returns:
            PhaseResult with validated benchmark parameters.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            benchmark_type = config.get(
                "benchmark_type", BenchmarkType.CTB.value
            )
            benchmark_name = config.get("benchmark_name", "")
            benchmark_provider = config.get("benchmark_provider", "")

            outputs["product_name"] = config.get("product_name", "")
            outputs["benchmark_type"] = benchmark_type
            outputs["benchmark_name"] = benchmark_name
            outputs["benchmark_provider"] = benchmark_provider

            # Validate benchmark type
            requirements = self.BENCHMARK_REQUIREMENTS.get(
                benchmark_type, {}
            )

            if not requirements:
                if benchmark_type == BenchmarkType.NONE.value:
                    warnings.append(
                        "No EU Climate Benchmark designated; Article 9 "
                        "products should explain methodology for "
                        "attaining sustainable objective without a benchmark"
                    )
                    outputs["has_valid_benchmark"] = False
                    outputs["requirements"] = {}
                else:
                    warnings.append(
                        f"Custom benchmark type '{benchmark_type}' "
                        f"specified; standard CTB/PAB requirements "
                        f"will not be applied"
                    )
                    outputs["has_valid_benchmark"] = True
                    outputs["requirements"] = {
                        "minimum_ghg_reduction_pct": 0.0,
                        "annual_decarbonization_pct": config.get(
                            "decarbonization_target_pct", 7.0
                        ),
                        "description": "Custom benchmark",
                        "exclusions_required": [],
                    }
            else:
                outputs["has_valid_benchmark"] = True
                outputs["requirements"] = requirements

            # Validate benchmark provider
            if not benchmark_provider and benchmark_type in (
                BenchmarkType.CTB.value, BenchmarkType.PAB.value
            ):
                warnings.append(
                    "Benchmark provider/administrator not specified"
                )

            outputs["minimum_ghg_reduction_pct"] = requirements.get(
                "minimum_ghg_reduction_pct", 0.0
            )
            outputs["annual_decarbonization_pct"] = requirements.get(
                "annual_decarbonization_pct", 7.0
            )
            outputs["exclusions_required"] = requirements.get(
                "exclusions_required", []
            )

            status = PhaseStatus.COMPLETED
            records = 1

        except Exception as exc:
            logger.error(
                "BenchmarkSelection failed: %s", exc, exc_info=True
            )
            errors.append(f"Benchmark selection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
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

    Measures the portfolio's alignment against benchmark criteria including
    GHG intensity reduction, exclusion compliance, and sector exposure
    requirements. Produces a quantified alignment status.
    """

    PHASE_NAME = "alignment_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute alignment assessment phase.

        Args:
            context: Workflow context with benchmark requirements.

        Returns:
            PhaseResult with alignment assessment results.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            selection_output = context.get_phase_output(
                "benchmark_selection"
            )
            requirements = selection_output.get("requirements", {})
            has_benchmark = selection_output.get(
                "has_valid_benchmark", False
            )

            portfolio_ci = config.get("portfolio_carbon_intensity", 0.0)
            universe_ci = config.get("universe_carbon_intensity", 0.0)
            exclusion_compliance = config.get(
                "exclusion_compliance", {}
            )

            alignment_checks = []

            # Check 1: GHG intensity reduction vs universe
            min_reduction = requirements.get(
                "minimum_ghg_reduction_pct", 0.0
            )

            if universe_ci > 0:
                actual_reduction = (
                    (universe_ci - portfolio_ci) / universe_ci * 100.0
                )
            else:
                actual_reduction = 0.0

            actual_reduction = round(actual_reduction, 2)
            ghg_aligned = actual_reduction >= min_reduction

            alignment_checks.append({
                "check": "ghg_intensity_reduction",
                "passed": ghg_aligned,
                "required": min_reduction,
                "actual": actual_reduction,
                "detail": (
                    f"Portfolio CI: {portfolio_ci:.2f}, "
                    f"Universe CI: {universe_ci:.2f}, "
                    f"Reduction: {actual_reduction:.1f}% "
                    f"(required: {min_reduction:.1f}%)"
                ),
            })

            # Check 2: Exclusion criteria compliance
            required_exclusions = requirements.get(
                "exclusions_required", []
            )
            exclusion_results = []
            all_exclusions_met = True

            for exclusion in required_exclusions:
                is_compliant = exclusion_compliance.get(exclusion, False)
                exclusion_results.append({
                    "criterion": exclusion,
                    "compliant": is_compliant,
                })
                if not is_compliant:
                    all_exclusions_met = False

            alignment_checks.append({
                "check": "exclusion_criteria",
                "passed": all_exclusions_met,
                "required_count": len(required_exclusions),
                "compliant_count": sum(
                    1 for e in exclusion_results if e["compliant"]
                ),
                "detail": exclusion_results,
            })

            # Check 3: Year-on-year decarbonization
            prior_ci = config.get("prior_year_carbon_intensity")
            required_decarb = requirements.get(
                "annual_decarbonization_pct", 7.0
            )
            yoy_decarb = 0.0

            if prior_ci is not None and prior_ci > 0:
                yoy_decarb = round(
                    (prior_ci - portfolio_ci) / prior_ci * 100.0, 2
                )
                decarb_aligned = yoy_decarb >= required_decarb
            else:
                decarb_aligned = True
                warnings.append(
                    "Prior year carbon intensity not available; "
                    "year-on-year decarbonization cannot be assessed"
                )

            alignment_checks.append({
                "check": "yoy_decarbonization",
                "passed": decarb_aligned,
                "required": required_decarb,
                "actual": yoy_decarb,
                "detail": (
                    f"YoY decarbonization: {yoy_decarb:.1f}% "
                    f"(required: {required_decarb:.1f}%)"
                ),
            })

            # Overall alignment determination
            all_passed = all(
                c["passed"] for c in alignment_checks
            )

            if not has_benchmark:
                alignment_status = AlignmentStatus.NOT_ASSESSED.value
            elif all_passed:
                alignment_status = AlignmentStatus.ALIGNED.value
            elif any(c["passed"] for c in alignment_checks):
                alignment_status = AlignmentStatus.PARTIALLY_ALIGNED.value
            else:
                alignment_status = AlignmentStatus.MISALIGNED.value

            outputs["alignment_checks"] = alignment_checks
            outputs["alignment_status"] = alignment_status
            outputs["ghg_reduction_pct"] = actual_reduction
            outputs["required_reduction_pct"] = min_reduction
            outputs["alignment_gap_pct"] = round(
                max(min_reduction - actual_reduction, 0.0), 2
            )
            outputs["exclusion_compliance"] = exclusion_results
            outputs["yoy_decarbonization_pct"] = yoy_decarb
            outputs["all_checks_passed"] = all_passed

            if not all_passed:
                failed = [
                    c["check"] for c in alignment_checks
                    if not c["passed"]
                ]
                warnings.append(
                    f"Benchmark alignment checks failed: "
                    f"{', '.join(failed)}"
                )

            status = PhaseStatus.COMPLETED
            records = len(alignment_checks)

        except Exception as exc:
            logger.error(
                "AlignmentAssessment failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Alignment assessment failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
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

class DeviationAnalysisPhase:
    """
    Phase 3: Deviation Analysis.

    Identifies deviations from benchmark alignment targets, performs
    root cause analysis, assesses severity, and generates remediation
    recommendations for each deviation.
    """

    PHASE_NAME = "deviation_analysis"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute deviation analysis phase.

        Args:
            context: Workflow context with alignment assessment.

        Returns:
            PhaseResult with deviation analysis and remediation plans.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            alignment_output = context.get_phase_output(
                "alignment_assessment"
            )
            alignment_checks = alignment_output.get(
                "alignment_checks", []
            )
            sector_exposures = config.get("sector_exposures", [])

            deviations: List[Dict[str, Any]] = []

            # Analyze each failed alignment check
            for check in alignment_checks:
                if check.get("passed", True):
                    continue

                check_name = check.get("check", "")

                if check_name == "ghg_intensity_reduction":
                    gap = check.get("required", 0) - check.get("actual", 0)
                    severity = self._classify_severity(gap, 10.0, 20.0)
                    deviations.append({
                        "deviation_id": str(uuid.uuid4()),
                        "check": check_name,
                        "severity": severity,
                        "gap_value": round(gap, 2),
                        "gap_unit": "percentage_points",
                        "root_cause": (
                            "Portfolio carbon intensity exceeds "
                            "benchmark reduction requirement"
                        ),
                        "remediation": [
                            "Increase allocation to low-carbon assets",
                            "Engage high-carbon investees on "
                            "decarbonization plans",
                            "Review sector allocation for carbon "
                            "hotspots",
                        ],
                        "timeline": "3-6 months",
                    })

                elif check_name == "exclusion_criteria":
                    detail = check.get("detail", [])
                    non_compliant = [
                        e for e in detail if not e.get("compliant")
                    ]
                    for exc_item in non_compliant:
                        criterion = exc_item.get("criterion", "")
                        deviations.append({
                            "deviation_id": str(uuid.uuid4()),
                            "check": check_name,
                            "severity": DeviationSeverity.CRITICAL.value,
                            "criterion": criterion,
                            "root_cause": (
                                f"Exclusion criterion '{criterion}' "
                                f"not met"
                            ),
                            "remediation": [
                                f"Divest from holdings violating "
                                f"'{criterion}' criterion",
                                "Update screening process to prevent "
                                "future violations",
                            ],
                            "timeline": "Immediate (30 days)",
                        })

                elif check_name == "yoy_decarbonization":
                    gap = check.get("required", 0) - check.get("actual", 0)
                    severity = self._classify_severity(gap, 3.0, 5.0)
                    deviations.append({
                        "deviation_id": str(uuid.uuid4()),
                        "check": check_name,
                        "severity": severity,
                        "gap_value": round(gap, 2),
                        "gap_unit": "percentage_points",
                        "root_cause": (
                            "Year-on-year decarbonization rate below "
                            "required trajectory"
                        ),
                        "remediation": [
                            "Accelerate portfolio decarbonization",
                            "Increase engagement intensity with "
                            "high-emitting investees",
                            "Consider portfolio rebalancing toward "
                            "faster-decarbonizing sectors",
                        ],
                        "timeline": "6-12 months",
                    })

            # Sector-level deviation analysis
            sector_deviations = []
            for sector in sector_exposures:
                sector_name = sector.get("sector", "")
                sector_ci = sector.get("carbon_intensity", 0.0)
                sector_weight = sector.get("weight_pct", 0.0)

                # Flag sectors contributing disproportionately
                contribution = sector_ci * sector_weight / 100.0
                if contribution > 0 and sector_weight > 5.0:
                    sector_deviations.append({
                        "sector": sector_name,
                        "weight_pct": sector_weight,
                        "carbon_intensity": sector_ci,
                        "carbon_contribution": round(contribution, 2),
                    })

            # Sort by carbon contribution descending
            sector_deviations.sort(
                key=lambda s: s["carbon_contribution"], reverse=True
            )

            outputs["deviations"] = deviations
            outputs["deviations_count"] = len(deviations)
            outputs["critical_deviations"] = sum(
                1 for d in deviations
                if d.get("severity") == DeviationSeverity.CRITICAL.value
            )
            outputs["sector_analysis"] = sector_deviations[:10]
            outputs["top_carbon_contributors"] = [
                s["sector"] for s in sector_deviations[:5]
            ]

            if outputs["critical_deviations"] > 0:
                warnings.append(
                    f"{outputs['critical_deviations']} critical "
                    f"deviation(s) require immediate attention"
                )

            status = PhaseStatus.COMPLETED
            records = len(deviations)

        except Exception as exc:
            logger.error(
                "DeviationAnalysis failed: %s", exc, exc_info=True
            )
            errors.append(f"Deviation analysis failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
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

    def _classify_severity(
        self, gap: float, medium_threshold: float, high_threshold: float
    ) -> str:
        """Classify deviation severity based on gap magnitude."""
        if gap >= high_threshold:
            return DeviationSeverity.CRITICAL.value
        elif gap >= medium_threshold:
            return DeviationSeverity.HIGH.value
        elif gap > 0:
            return DeviationSeverity.MEDIUM.value
        return DeviationSeverity.LOW.value

class TrajectoryProjectionPhase:
    """
    Phase 4: Trajectory Projection.

    Projects the portfolio's forward-looking decarbonization trajectory
    and assesses whether the portfolio will remain aligned with the
    benchmark over the projection horizon using deterministic formulas.
    """

    PHASE_NAME = "trajectory_projection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute trajectory projection phase.

        Args:
            context: Workflow context with alignment and deviation data.

        Returns:
            PhaseResult with projected trajectory and alignment forecast.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            selection_output = context.get_phase_output(
                "benchmark_selection"
            )
            alignment_output = context.get_phase_output(
                "alignment_assessment"
            )

            portfolio_ci = config.get("portfolio_carbon_intensity", 0.0)
            universe_ci = config.get("universe_carbon_intensity", 0.0)
            required_decarb = selection_output.get(
                "requirements", {}
            ).get("annual_decarbonization_pct", 7.0)
            projection_years = config.get("projection_years", 5)
            current_reduction = alignment_output.get(
                "ghg_reduction_pct", 0.0
            )
            required_reduction = alignment_output.get(
                "required_reduction_pct", 30.0
            )

            # Project portfolio CI trajectory (compound decarbonization)
            trajectory_points: List[Dict[str, Any]] = []
            projected_ci = portfolio_ci

            for year in range(1, projection_years + 1):
                projected_ci = projected_ci * (1.0 - required_decarb / 100.0)
                projected_ci = round(projected_ci, 4)

                # Also project universe CI (assumed static or slight decline)
                projected_universe = universe_ci * (1.0 - 0.02) ** year
                projected_universe = round(projected_universe, 4)

                if projected_universe > 0:
                    projected_reduction = round(
                        (projected_universe - projected_ci)
                        / projected_universe * 100.0, 2
                    )
                else:
                    projected_reduction = 0.0

                trajectory_points.append({
                    "year": year,
                    "projected_portfolio_ci": projected_ci,
                    "projected_universe_ci": projected_universe,
                    "projected_reduction_pct": projected_reduction,
                    "meets_threshold": (
                        projected_reduction >= required_reduction
                    ),
                })

            # Determine projected alignment year
            alignment_year = None
            if current_reduction < required_reduction:
                for point in trajectory_points:
                    if point["meets_threshold"]:
                        alignment_year = point["year"]
                        break
            else:
                alignment_year = 0  # Already aligned

            # On-track determination
            on_track = (
                alignment_year is not None
                and alignment_year <= projection_years
            )

            # Temperature pathway alignment (simplified 2-degree check)
            # 7% annual decarbonization aligns with ~1.5C pathway
            actual_decarb = alignment_output.get(
                "yoy_decarbonization_pct", 0.0
            )
            if actual_decarb >= 7.0:
                temperature_pathway = "1.5C"
            elif actual_decarb >= 4.0:
                temperature_pathway = "Below 2C"
            elif actual_decarb >= 2.0:
                temperature_pathway = "2C"
            else:
                temperature_pathway = "Above 2C"

            outputs["trajectory_points"] = trajectory_points
            outputs["projection_years"] = projection_years
            outputs["current_portfolio_ci"] = portfolio_ci
            outputs["projected_final_ci"] = (
                trajectory_points[-1]["projected_portfolio_ci"]
                if trajectory_points else portfolio_ci
            )
            outputs["projected_alignment_year"] = alignment_year
            outputs["on_track"] = on_track
            outputs["temperature_pathway"] = temperature_pathway
            outputs["required_annual_decarbonization_pct"] = required_decarb
            outputs["actual_annual_decarbonization_pct"] = actual_decarb

            if not on_track:
                warnings.append(
                    f"Portfolio is not on track to meet benchmark "
                    f"alignment within {projection_years} years"
                )

            if temperature_pathway in ("2C", "Above 2C"):
                warnings.append(
                    f"Current trajectory aligns with {temperature_pathway} "
                    f"pathway; Article 9 products should target 1.5C"
                )

            status = PhaseStatus.COMPLETED
            records = len(trajectory_points)

        except Exception as exc:
            logger.error(
                "TrajectoryProjection failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Trajectory projection failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
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

# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================

class BenchmarkMonitoringWorkflow:
    """
    Four-phase EU Climate Benchmark monitoring workflow for Article 9.

    Orchestrates the complete benchmark monitoring pipeline from benchmark
    selection through alignment assessment, deviation analysis, and
    trajectory projection. Supports checkpoint/resume and phase skipping.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = BenchmarkMonitoringWorkflow()
        >>> input_data = BenchmarkMonitoringInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     reporting_date="2026-01-01",
        ...     benchmark_type="PAB",
        ...     portfolio_carbon_intensity=50.0,
        ...     universe_carbon_intensity=120.0,
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "benchmark_monitoring"

    PHASE_ORDER = [
        "benchmark_selection",
        "alignment_assessment",
        "deviation_analysis",
        "trajectory_projection",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the benchmark monitoring workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "benchmark_selection": BenchmarkSelectionPhase(),
            "alignment_assessment": AlignmentAssessmentPhase(),
            "deviation_analysis": DeviationAnalysisPhase(),
            "trajectory_projection": TrajectoryProjectionPhase(),
        }

    async def run(
        self, input_data: BenchmarkMonitoringInput
    ) -> BenchmarkMonitoringResult:
        """
        Execute the complete 4-phase benchmark monitoring workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            BenchmarkMonitoringResult with per-phase details and summary.
        """
        started_at = utcnow()
        logger.info(
            "Starting benchmark monitoring workflow %s for org=%s product=%s",
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
                logger.info(
                    "Phase '%s' already completed, skipping",
                    phase_name,
                )
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(
                phase_name, f"Starting: {phase_name}", pct
            )
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(
                        phase_name, phase_result.outputs
                    )
                    context.mark_phase(
                        phase_name, PhaseStatus.COMPLETED
                    )
                else:
                    context.mark_phase(
                        phase_name, phase_result.status
                    )
                    if phase_name == "benchmark_selection":
                        overall_status = WorkflowStatus.FAILED
                        logger.error(
                            "Critical phase '%s' failed, aborting",
                            phase_name,
                        )
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
                    started_at=utcnow(),
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
                WorkflowStatus.COMPLETED if all_ok
                else WorkflowStatus.PARTIAL
            )

        completed_at = utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )
        logger.info(
            "Benchmark monitoring workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return BenchmarkMonitoringResult(
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
            benchmark_type=summary.get("benchmark_type", "CTB"),
            benchmark_name=summary.get("benchmark_name", ""),
            alignment_status=summary.get(
                "alignment_status", "NOT_ASSESSED"
            ),
            ghg_reduction_pct=summary.get("ghg_reduction_pct", 0.0),
            required_reduction_pct=summary.get(
                "required_reduction_pct", 30.0
            ),
            alignment_gap_pct=summary.get("alignment_gap_pct", 0.0),
            deviations_found=summary.get("deviations_found", 0),
            critical_deviations=summary.get("critical_deviations", 0),
            yoy_decarbonization_pct=summary.get(
                "yoy_decarbonization_pct", 0.0
            ),
            projected_alignment_year=summary.get(
                "projected_alignment_year"
            ),
            on_track=summary.get("on_track", False),
        )

    def _build_config(
        self, input_data: BenchmarkMonitoringInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return input_data.model_dump()

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        selection = context.get_phase_output("benchmark_selection")
        alignment = context.get_phase_output("alignment_assessment")
        deviation = context.get_phase_output("deviation_analysis")
        trajectory = context.get_phase_output("trajectory_projection")

        return {
            "product_name": selection.get("product_name", ""),
            "benchmark_type": selection.get("benchmark_type", "CTB"),
            "benchmark_name": selection.get("benchmark_name", ""),
            "alignment_status": alignment.get(
                "alignment_status", "NOT_ASSESSED"
            ),
            "ghg_reduction_pct": alignment.get(
                "ghg_reduction_pct", 0.0
            ),
            "required_reduction_pct": alignment.get(
                "required_reduction_pct", 30.0
            ),
            "alignment_gap_pct": alignment.get(
                "alignment_gap_pct", 0.0
            ),
            "deviations_found": deviation.get(
                "deviations_count", 0
            ),
            "critical_deviations": deviation.get(
                "critical_deviations", 0
            ),
            "yoy_decarbonization_pct": alignment.get(
                "yoy_decarbonization_pct", 0.0
            ),
            "projected_alignment_year": trajectory.get(
                "projected_alignment_year"
            ),
            "on_track": trajectory.get("on_track", False),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug(
                    "Progress callback failed for phase=%s", phase
                )
