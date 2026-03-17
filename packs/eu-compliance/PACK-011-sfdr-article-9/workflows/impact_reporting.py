# -*- coding: utf-8 -*-
"""
Impact Reporting Workflow
================================================

Four-phase workflow for SFDR Article 9 impact measurement and reporting.
Orchestrates KPI definition, data collection, impact calculation, and
report generation into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 9 products must demonstrate measurable positive impact aligned
      with the declared sustainable investment objective.
    - Impact must be measured using clearly defined KPIs that track attainment
      of the sustainable objective.
    - Data collection must follow verifiable methodologies and be sourced from
      reliable providers.
    - Impact calculations must use deterministic formulas with full provenance.
    - Reports must be generated in structured formats suitable for regulatory
      submission and investor communication.

Phases:
    1. KPIDefinition - Define impact KPIs aligned with sustainable objective
    2. DataCollection - Source, validate, and normalize impact data
    3. ImpactCalculation - Calculate impact metrics using deterministic formulas
    4. ReportGeneration - Generate structured impact report

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


class KPICategory(str, Enum):
    """KPI category classification."""
    ENVIRONMENTAL = "ENVIRONMENTAL"
    SOCIAL = "SOCIAL"
    GOVERNANCE = "GOVERNANCE"
    CLIMATE = "CLIMATE"
    BIODIVERSITY = "BIODIVERSITY"
    CIRCULAR_ECONOMY = "CIRCULAR_ECONOMY"


class MeasurementFrequency(str, Enum):
    """KPI measurement frequency."""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"


class DataQualityTier(str, Enum):
    """Data quality tier classification."""
    TIER_1_REPORTED = "TIER_1_REPORTED"
    TIER_2_ESTIMATED = "TIER_2_ESTIMATED"
    TIER_3_PROXY = "TIER_3_PROXY"


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
# DATA MODELS - IMPACT REPORTING
# =============================================================================


class ImpactReportingInput(BaseModel):
    """Input configuration for the impact reporting workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None, description="ISIN if applicable")
    reporting_period_start: str = Field(
        ..., description="Period start date YYYY-MM-DD"
    )
    reporting_period_end: str = Field(
        ..., description="Period end date YYYY-MM-DD"
    )
    sustainable_objective: str = Field(
        ..., description="Declared sustainable objective description"
    )
    kpi_definitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pre-defined KPI definitions (optional)"
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data source identifiers for collection"
    )
    measurement_frequency: str = Field(
        default=MeasurementFrequency.QUARTERLY.value,
        description="KPI measurement frequency"
    )
    benchmark_index: Optional[str] = Field(
        None, description="Benchmark index for comparison"
    )
    target_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Target values for each KPI"
    )
    portfolio_holdings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Portfolio holdings for impact calculation"
    )
    previous_report_id: Optional[str] = Field(
        None, description="Previous report for trend comparison"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_period_start", "reporting_period_end")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be YYYY-MM-DD format")
        return v


class ImpactReportingResult(WorkflowResult):
    """Complete result from the impact reporting workflow."""
    product_name: str = Field(default="")
    reporting_period: str = Field(default="")
    kpis_defined: int = Field(default=0)
    data_points_collected: int = Field(default=0)
    data_quality_score: float = Field(default=0.0)
    impact_metrics_calculated: int = Field(default=0)
    positive_impact_confirmed: bool = Field(default=False)
    report_format: str = Field(default="structured_json")
    report_sections_completed: int = Field(default=0)
    report_sections_total: int = Field(default=6)
    overall_impact_score: float = Field(default=0.0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class KPIDefinitionPhase:
    """
    Phase 1: KPI Definition.

    Defines impact KPIs aligned with the product's sustainable investment
    objective. Each KPI includes category, unit, target, measurement
    frequency, and data source mapping.
    """

    PHASE_NAME = "kpi_definition"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute KPI definition phase.

        Args:
            context: Workflow context with product configuration.

        Returns:
            PhaseResult with defined KPIs.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            product_name = config.get("product_name", "")
            objective = config.get("sustainable_objective", "")
            user_kpis = config.get("kpi_definitions", [])
            targets = config.get("target_values", {})
            frequency = config.get(
                "measurement_frequency",
                MeasurementFrequency.QUARTERLY.value,
            )

            outputs["product_name"] = product_name
            outputs["sustainable_objective"] = objective

            # Build KPI registry from user-defined or defaults
            kpis = []

            if user_kpis:
                for idx, kpi_def in enumerate(user_kpis):
                    kpi = {
                        "kpi_id": kpi_def.get(
                            "kpi_id", f"KPI-{idx + 1:03d}"
                        ),
                        "name": kpi_def.get("name", f"KPI {idx + 1}"),
                        "category": kpi_def.get(
                            "category",
                            KPICategory.ENVIRONMENTAL.value,
                        ),
                        "unit": kpi_def.get("unit", ""),
                        "description": kpi_def.get("description", ""),
                        "measurement_frequency": kpi_def.get(
                            "measurement_frequency", frequency
                        ),
                        "target_value": targets.get(
                            kpi_def.get("name", ""), None
                        ),
                        "data_source": kpi_def.get("data_source", ""),
                        "calculation_method": kpi_def.get(
                            "calculation_method", "weighted_average"
                        ),
                        "is_mandatory": kpi_def.get("is_mandatory", True),
                    }
                    kpis.append(kpi)
            else:
                # Default KPIs for Article 9 products
                default_kpis = [
                    {
                        "kpi_id": "KPI-001",
                        "name": "Carbon Intensity",
                        "category": KPICategory.CLIMATE.value,
                        "unit": "tCO2e/EUR million revenue",
                        "description": (
                            "Weighted average carbon intensity of portfolio"
                        ),
                        "is_mandatory": True,
                    },
                    {
                        "kpi_id": "KPI-002",
                        "name": "GHG Emissions Scope 1+2",
                        "category": KPICategory.CLIMATE.value,
                        "unit": "tCO2e",
                        "description": (
                            "Total Scope 1 and 2 GHG emissions"
                        ),
                        "is_mandatory": True,
                    },
                    {
                        "kpi_id": "KPI-003",
                        "name": "Renewable Energy Share",
                        "category": KPICategory.ENVIRONMENTAL.value,
                        "unit": "percentage",
                        "description": (
                            "Share of renewable energy in investee revenue"
                        ),
                        "is_mandatory": False,
                    },
                    {
                        "kpi_id": "KPI-004",
                        "name": "Gender Pay Gap",
                        "category": KPICategory.SOCIAL.value,
                        "unit": "percentage",
                        "description": (
                            "Average unadjusted gender pay gap"
                        ),
                        "is_mandatory": False,
                    },
                    {
                        "kpi_id": "KPI-005",
                        "name": "Taxonomy Alignment",
                        "category": KPICategory.ENVIRONMENTAL.value,
                        "unit": "percentage",
                        "description": (
                            "Share of taxonomy-aligned investments"
                        ),
                        "is_mandatory": True,
                    },
                ]
                for kpi in default_kpis:
                    kpi["measurement_frequency"] = frequency
                    kpi["target_value"] = targets.get(kpi["name"], None)
                    kpi["data_source"] = ""
                    kpi["calculation_method"] = "weighted_average"
                kpis = default_kpis

            # Validate: at least one mandatory KPI required
            mandatory_count = sum(
                1 for k in kpis if k.get("is_mandatory", False)
            )
            if mandatory_count == 0:
                warnings.append(
                    "No mandatory KPIs defined; recommend at least one"
                )

            # Validate: KPIs must have units
            for kpi in kpis:
                if not kpi.get("unit"):
                    warnings.append(
                        f"KPI '{kpi['name']}' has no measurement unit"
                    )

            outputs["kpis"] = kpis
            outputs["kpi_count"] = len(kpis)
            outputs["mandatory_kpi_count"] = mandatory_count
            outputs["measurement_frequency"] = frequency
            outputs["categories_covered"] = list(set(
                k["category"] for k in kpis
            ))

            status = PhaseStatus.COMPLETED
            records = len(kpis)

        except Exception as exc:
            logger.error(
                "KPIDefinition failed: %s", exc, exc_info=True
            )
            errors.append(f"KPI definition failed: {str(exc)}")
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


class DataCollectionPhase:
    """
    Phase 2: Data Collection.

    Sources, validates, and normalizes impact data from configured
    data providers. Assigns data quality tiers and calculates an
    overall data quality score for the reporting period.
    """

    PHASE_NAME = "data_collection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute data collection phase.

        Args:
            context: Workflow context with KPI definitions.

        Returns:
            PhaseResult with collected and validated data.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            kpi_output = context.get_phase_output("kpi_definition")
            kpis = kpi_output.get("kpis", [])
            data_sources = config.get("data_sources", [])
            holdings = config.get("portfolio_holdings", [])

            outputs["data_sources_configured"] = len(data_sources)
            outputs["holdings_count"] = len(holdings)

            # Collect data points per KPI
            collected_data: List[Dict[str, Any]] = []
            quality_scores: List[float] = []

            for kpi in kpis:
                kpi_id = kpi.get("kpi_id", "")
                kpi_name = kpi.get("name", "")

                # Collect data per holding
                kpi_data_points = []
                for holding in holdings:
                    holding_id = holding.get("holding_id", "")
                    holding_name = holding.get("name", "")
                    weight = holding.get("weight_pct", 0.0)

                    # Check if holding has reported data for this KPI
                    reported_values = holding.get("reported_data", {})
                    kpi_value = reported_values.get(kpi_name)

                    if kpi_value is not None:
                        quality_tier = DataQualityTier.TIER_1_REPORTED.value
                        quality_score = 1.0
                    elif holding.get("estimated_data", {}).get(kpi_name):
                        kpi_value = holding["estimated_data"][kpi_name]
                        quality_tier = DataQualityTier.TIER_2_ESTIMATED.value
                        quality_score = 0.7
                    else:
                        kpi_value = None
                        quality_tier = DataQualityTier.TIER_3_PROXY.value
                        quality_score = 0.3

                    data_point = {
                        "kpi_id": kpi_id,
                        "kpi_name": kpi_name,
                        "holding_id": holding_id,
                        "holding_name": holding_name,
                        "weight_pct": weight,
                        "value": kpi_value,
                        "quality_tier": quality_tier,
                        "quality_score": quality_score,
                        "collected_at": _utcnow().isoformat(),
                    }
                    kpi_data_points.append(data_point)

                    if kpi_value is not None:
                        quality_scores.append(quality_score)

                collected_data.extend(kpi_data_points)

                # Check coverage for mandatory KPIs
                coverage = sum(
                    1 for dp in kpi_data_points
                    if dp["value"] is not None
                )
                total = len(kpi_data_points)
                coverage_pct = (
                    coverage / total * 100.0 if total > 0 else 0.0
                )

                if kpi.get("is_mandatory") and coverage_pct < 70.0:
                    warnings.append(
                        f"Mandatory KPI '{kpi_name}' has low data "
                        f"coverage: {coverage_pct:.1f}%"
                    )

            # Calculate overall data quality
            overall_quality = (
                sum(quality_scores) / len(quality_scores)
                if quality_scores else 0.0
            )

            outputs["collected_data"] = collected_data
            outputs["data_points_count"] = len(collected_data)
            outputs["data_points_with_values"] = sum(
                1 for d in collected_data if d["value"] is not None
            )
            outputs["overall_data_quality_score"] = round(
                overall_quality, 3
            )
            outputs["quality_tier_distribution"] = {
                DataQualityTier.TIER_1_REPORTED.value: sum(
                    1 for d in collected_data
                    if d["quality_tier"] == DataQualityTier.TIER_1_REPORTED.value
                ),
                DataQualityTier.TIER_2_ESTIMATED.value: sum(
                    1 for d in collected_data
                    if d["quality_tier"] == DataQualityTier.TIER_2_ESTIMATED.value
                ),
                DataQualityTier.TIER_3_PROXY.value: sum(
                    1 for d in collected_data
                    if d["quality_tier"] == DataQualityTier.TIER_3_PROXY.value
                ),
            }

            if overall_quality < 0.5:
                warnings.append(
                    f"Overall data quality score is low: "
                    f"{overall_quality:.2f}"
                )

            status = PhaseStatus.COMPLETED
            records = len(collected_data)

        except Exception as exc:
            logger.error(
                "DataCollection failed: %s", exc, exc_info=True
            )
            errors.append(f"Data collection failed: {str(exc)}")
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


class ImpactCalculationPhase:
    """
    Phase 3: Impact Calculation.

    Calculates portfolio-level impact metrics using deterministic
    weighted-average formulas. Compares against targets and benchmarks.
    All calculations use zero-hallucination arithmetic only.
    """

    PHASE_NAME = "impact_calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute impact calculation phase.

        Args:
            context: Workflow context with collected data.

        Returns:
            PhaseResult with calculated impact metrics.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            kpi_output = context.get_phase_output("kpi_definition")
            data_output = context.get_phase_output("data_collection")
            kpis = kpi_output.get("kpis", [])
            collected_data = data_output.get("collected_data", [])
            targets = config.get("target_values", {})
            benchmark_index = config.get("benchmark_index")

            impact_metrics: List[Dict[str, Any]] = []
            positive_impacts = 0
            negative_impacts = 0

            for kpi in kpis:
                kpi_id = kpi.get("kpi_id", "")
                kpi_name = kpi.get("name", "")
                target_value = targets.get(kpi_name)
                calc_method = kpi.get(
                    "calculation_method", "weighted_average"
                )

                # Filter data for this KPI
                kpi_data = [
                    d for d in collected_data
                    if d["kpi_id"] == kpi_id and d["value"] is not None
                ]

                if not kpi_data:
                    warnings.append(
                        f"No data available for KPI '{kpi_name}'"
                    )
                    impact_metrics.append({
                        "kpi_id": kpi_id,
                        "kpi_name": kpi_name,
                        "calculated_value": None,
                        "target_value": target_value,
                        "target_met": False,
                        "data_coverage_pct": 0.0,
                        "calculation_method": calc_method,
                    })
                    continue

                # Deterministic weighted average calculation
                if calc_method == "weighted_average":
                    total_weight = sum(
                        d["weight_pct"] for d in kpi_data
                    )
                    if total_weight > 0:
                        calculated = sum(
                            d["value"] * d["weight_pct"]
                            for d in kpi_data
                        ) / total_weight
                    else:
                        calculated = sum(
                            d["value"] for d in kpi_data
                        ) / len(kpi_data)
                elif calc_method == "sum":
                    calculated = sum(d["value"] for d in kpi_data)
                elif calc_method == "average":
                    calculated = (
                        sum(d["value"] for d in kpi_data) / len(kpi_data)
                    )
                else:
                    calculated = sum(
                        d["value"] * d["weight_pct"]
                        for d in kpi_data
                    ) / max(
                        sum(d["weight_pct"] for d in kpi_data), 1.0
                    )

                calculated = round(calculated, 4)

                # Target comparison
                target_met = False
                if target_value is not None:
                    # For intensity metrics, lower is better
                    category = kpi.get("category", "")
                    if category in (
                        KPICategory.CLIMATE.value,
                    ):
                        target_met = calculated <= target_value
                    else:
                        target_met = calculated >= target_value

                if target_met:
                    positive_impacts += 1
                elif target_value is not None:
                    negative_impacts += 1

                # Data coverage
                total_holdings = len(
                    config.get("portfolio_holdings", [])
                )
                coverage_pct = (
                    len(kpi_data) / total_holdings * 100.0
                    if total_holdings > 0 else 0.0
                )

                metric = {
                    "kpi_id": kpi_id,
                    "kpi_name": kpi_name,
                    "calculated_value": calculated,
                    "target_value": target_value,
                    "target_met": target_met,
                    "data_coverage_pct": round(coverage_pct, 1),
                    "calculation_method": calc_method,
                    "data_points_used": len(kpi_data),
                    "unit": kpi.get("unit", ""),
                }
                impact_metrics.append(metric)

            # Overall impact assessment
            total_with_targets = positive_impacts + negative_impacts
            impact_score = (
                positive_impacts / total_with_targets * 100.0
                if total_with_targets > 0 else 0.0
            )

            outputs["impact_metrics"] = impact_metrics
            outputs["metrics_calculated"] = len(
                [m for m in impact_metrics if m["calculated_value"] is not None]
            )
            outputs["positive_impacts"] = positive_impacts
            outputs["negative_impacts"] = negative_impacts
            outputs["overall_impact_score"] = round(impact_score, 1)
            outputs["positive_impact_confirmed"] = impact_score >= 50.0
            outputs["benchmark_index"] = benchmark_index

            if impact_score < 50.0 and total_with_targets > 0:
                warnings.append(
                    f"Impact score {impact_score:.1f}% is below 50% "
                    f"threshold; product may not be meeting its "
                    f"sustainable objective"
                )

            status = PhaseStatus.COMPLETED
            records = len(impact_metrics)

        except Exception as exc:
            logger.error(
                "ImpactCalculation failed: %s", exc, exc_info=True
            )
            errors.append(f"Impact calculation failed: {str(exc)}")
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


class ReportGenerationPhase:
    """
    Phase 4: Report Generation.

    Generates a structured impact report containing all impact metrics,
    trend analysis (if prior reports exist), methodology description,
    and regulatory compliance status sections.
    """

    PHASE_NAME = "report_generation"

    REPORT_SECTIONS = [
        "executive_summary",
        "kpi_overview",
        "impact_metrics_detail",
        "data_quality_assessment",
        "target_performance",
        "methodology_and_limitations",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute report generation phase.

        Args:
            context: Workflow context with calculated impact metrics.

        Returns:
            PhaseResult with generated report structure.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            kpi_output = context.get_phase_output("kpi_definition")
            data_output = context.get_phase_output("data_collection")
            calc_output = context.get_phase_output("impact_calculation")

            product_name = config.get("product_name", "")
            period_start = config.get("reporting_period_start", "")
            period_end = config.get("reporting_period_end", "")
            impact_metrics = calc_output.get("impact_metrics", [])

            sections: Dict[str, Any] = {}
            completed_sections = 0

            # Section 1: Executive summary
            sections["executive_summary"] = {
                "product_name": product_name,
                "reporting_period": f"{period_start} to {period_end}",
                "sustainable_objective": config.get(
                    "sustainable_objective", ""
                ),
                "kpis_measured": kpi_output.get("kpi_count", 0),
                "overall_impact_score": calc_output.get(
                    "overall_impact_score", 0.0
                ),
                "positive_impact_confirmed": calc_output.get(
                    "positive_impact_confirmed", False
                ),
                "data_quality_score": data_output.get(
                    "overall_data_quality_score", 0.0
                ),
            }
            completed_sections += 1

            # Section 2: KPI overview
            sections["kpi_overview"] = {
                "total_kpis": kpi_output.get("kpi_count", 0),
                "mandatory_kpis": kpi_output.get(
                    "mandatory_kpi_count", 0
                ),
                "categories_covered": kpi_output.get(
                    "categories_covered", []
                ),
                "measurement_frequency": kpi_output.get(
                    "measurement_frequency", ""
                ),
                "kpi_definitions": kpi_output.get("kpis", []),
            }
            completed_sections += 1

            # Section 3: Impact metrics detail
            sections["impact_metrics_detail"] = {
                "metrics": impact_metrics,
                "metrics_with_values": calc_output.get(
                    "metrics_calculated", 0
                ),
                "positive_impacts": calc_output.get(
                    "positive_impacts", 0
                ),
                "negative_impacts": calc_output.get(
                    "negative_impacts", 0
                ),
            }
            completed_sections += 1

            # Section 4: Data quality assessment
            sections["data_quality_assessment"] = {
                "overall_score": data_output.get(
                    "overall_data_quality_score", 0.0
                ),
                "data_points_collected": data_output.get(
                    "data_points_count", 0
                ),
                "data_points_with_values": data_output.get(
                    "data_points_with_values", 0
                ),
                "tier_distribution": data_output.get(
                    "quality_tier_distribution", {}
                ),
                "data_sources_count": data_output.get(
                    "data_sources_configured", 0
                ),
            }
            completed_sections += 1

            # Section 5: Target performance
            target_metrics = [
                m for m in impact_metrics
                if m.get("target_value") is not None
            ]
            targets_met = sum(
                1 for m in target_metrics if m.get("target_met", False)
            )
            sections["target_performance"] = {
                "metrics_with_targets": len(target_metrics),
                "targets_met": targets_met,
                "targets_missed": len(target_metrics) - targets_met,
                "target_achievement_rate": round(
                    targets_met / len(target_metrics) * 100.0
                    if target_metrics else 0.0, 1
                ),
                "detail": target_metrics,
            }
            completed_sections += 1

            # Section 6: Methodology and limitations
            sections["methodology_and_limitations"] = {
                "calculation_methods_used": list(set(
                    m.get("calculation_method", "")
                    for m in impact_metrics
                )),
                "data_sources": config.get("data_sources", []),
                "limitations": [
                    "Data availability may vary across investees",
                    "Estimated and proxy data used where reported "
                    "data unavailable",
                    "Calculation methodologies may differ from "
                    "investee-reported figures",
                ],
                "methodology_description": (
                    "Impact metrics are calculated using weighted "
                    "average methodology based on portfolio weights. "
                    "Data quality is assessed using a three-tier "
                    "system (reported, estimated, proxy)."
                ),
            }
            completed_sections += 1

            outputs["report_sections"] = sections
            outputs["sections_completed"] = completed_sections
            outputs["sections_total"] = len(self.REPORT_SECTIONS)
            outputs["report_format"] = "structured_json"
            outputs["report_version"] = "1.0"
            outputs["generated_at"] = _utcnow().isoformat()

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "ReportGeneration failed: %s", exc, exc_info=True
            )
            errors.append(f"Report generation failed: {str(exc)}")
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


class ImpactReportingWorkflow:
    """
    Four-phase impact measurement and reporting workflow for Article 9.

    Orchestrates the complete impact reporting pipeline from KPI definition
    through data collection, impact calculation, and report generation.
    Supports checkpoint/resume and phase skipping.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = ImpactReportingWorkflow()
        >>> input_data = ImpactReportingInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     reporting_period_start="2025-01-01",
        ...     reporting_period_end="2025-12-31",
        ...     sustainable_objective="Carbon reduction",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "impact_reporting"

    PHASE_ORDER = [
        "kpi_definition",
        "data_collection",
        "impact_calculation",
        "report_generation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the impact reporting workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "kpi_definition": KPIDefinitionPhase(),
            "data_collection": DataCollectionPhase(),
            "impact_calculation": ImpactCalculationPhase(),
            "report_generation": ReportGenerationPhase(),
        }

    async def run(
        self, input_data: ImpactReportingInput
    ) -> ImpactReportingResult:
        """
        Execute the complete 4-phase impact reporting workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            ImpactReportingResult with per-phase details and summary.
        """
        started_at = _utcnow()
        logger.info(
            "Starting impact reporting workflow %s for org=%s product=%s",
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
                    if phase_name == "kpi_definition":
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
                WorkflowStatus.COMPLETED if all_ok
                else WorkflowStatus.PARTIAL
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
        logger.info(
            "Impact reporting workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return ImpactReportingResult(
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
            reporting_period=summary.get("reporting_period", ""),
            kpis_defined=summary.get("kpis_defined", 0),
            data_points_collected=summary.get(
                "data_points_collected", 0
            ),
            data_quality_score=summary.get("data_quality_score", 0.0),
            impact_metrics_calculated=summary.get(
                "impact_metrics_calculated", 0
            ),
            positive_impact_confirmed=summary.get(
                "positive_impact_confirmed", False
            ),
            report_sections_completed=summary.get(
                "report_sections_completed", 0
            ),
            report_sections_total=summary.get(
                "report_sections_total", 6
            ),
            overall_impact_score=summary.get(
                "overall_impact_score", 0.0
            ),
        )

    def _build_config(
        self, input_data: ImpactReportingInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return input_data.model_dump()

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        kpi_out = context.get_phase_output("kpi_definition")
        data_out = context.get_phase_output("data_collection")
        calc_out = context.get_phase_output("impact_calculation")
        report_out = context.get_phase_output("report_generation")

        return {
            "product_name": kpi_out.get("product_name", ""),
            "reporting_period": (
                f"{context.config.get('reporting_period_start', '')} to "
                f"{context.config.get('reporting_period_end', '')}"
            ),
            "kpis_defined": kpi_out.get("kpi_count", 0),
            "data_points_collected": data_out.get(
                "data_points_count", 0
            ),
            "data_quality_score": data_out.get(
                "overall_data_quality_score", 0.0
            ),
            "impact_metrics_calculated": calc_out.get(
                "metrics_calculated", 0
            ),
            "positive_impact_confirmed": calc_out.get(
                "positive_impact_confirmed", False
            ),
            "overall_impact_score": calc_out.get(
                "overall_impact_score", 0.0
            ),
            "report_sections_completed": report_out.get(
                "sections_completed", 0
            ),
            "report_sections_total": report_out.get(
                "sections_total", 6
            ),
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
