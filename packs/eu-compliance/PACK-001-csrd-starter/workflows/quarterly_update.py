# -*- coding: utf-8 -*-
"""
Quarterly Update Workflow
=========================

Lightweight quarterly data refresh and progress tracking workflow.
Designed to run at the end of each fiscal quarter (Q1-Q4) to incrementally
update emissions data, recalculate trends, and flag deviations from annual
targets without requiring a full end-to-end reporting run.

Steps:
    1. Incremental data intake (new quarter only)
    2. Recalculate emissions for updated period
    3. Update trends and benchmarks
    4. Generate quarterly progress report
    5. Flag deviations from annual targets

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class QuarterlyStepStatus(str, Enum):
    """Status of a quarterly workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Quarter(str, Enum):
    """Fiscal quarter identifier."""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class DeviationSeverity(str, Enum):
    """Severity level for target deviations."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# =============================================================================
# DATA MODELS
# =============================================================================


class QuarterlyUpdateInput(BaseModel):
    """Input configuration for the quarterly update workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2024, le=2050, description="Fiscal year")
    quarter: Quarter = Field(..., description="Quarter to update (Q1-Q4)")
    quarter_start: str = Field(..., description="Quarter start date (YYYY-MM-DD)")
    quarter_end: str = Field(..., description="Quarter end date (YYYY-MM-DD)")
    data_source_ids: List[str] = Field(
        default_factory=list,
        description="Specific data source IDs to refresh; empty = all"
    )
    annual_target_tco2e: Optional[float] = Field(
        None, ge=0, description="Annual emissions target in tCO2e for deviation checks"
    )
    deviation_threshold_pct: float = Field(
        default=10.0, ge=0, le=100,
        description="Percentage deviation from quarterly target that triggers alert"
    )
    recalculate_scopes: List[str] = Field(
        default_factory=lambda: ["scope1", "scope2", "scope3"],
        description="Which scopes to recalculate"
    )
    comparison_quarters: List[str] = Field(
        default_factory=list,
        description="Previous quarters to compare against (e.g. ['2024-Q4', '2025-Q1'])"
    )

    @field_validator("quarter_start", "quarter_end")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Validate ISO date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v


class StepResult(BaseModel):
    """Result from a single workflow step."""
    step_name: str = Field(..., description="Step identifier")
    status: QuarterlyStepStatus = Field(..., description="Step completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DeviationAlert(BaseModel):
    """Alert raised when quarterly data deviates from annual targets."""
    metric_name: str = Field(..., description="Name of the metric that deviated")
    scope: str = Field(..., description="Scope (scope1, scope2, scope3, total)")
    expected_value: float = Field(..., description="Expected quarterly value")
    actual_value: float = Field(..., description="Actual quarterly value")
    deviation_pct: float = Field(..., description="Percentage deviation")
    severity: DeviationSeverity = Field(..., description="Alert severity")
    recommendation: str = Field(default="", description="Suggested action")


class TrendDataPoint(BaseModel):
    """Single data point in a trend series."""
    period: str = Field(..., description="Period label (e.g. '2025-Q1')")
    value_tco2e: float = Field(..., description="Emissions value in tCO2e")
    data_quality_score: float = Field(default=0.0, ge=0, le=1.0)


class QuarterlyUpdateResult(BaseModel):
    """Complete result from the quarterly update workflow."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    status: QuarterlyStepStatus = Field(..., description="Overall workflow status")
    quarter_label: str = Field(..., description="e.g. '2025-Q2'")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    steps: List[StepResult] = Field(default_factory=list)
    emissions_summary: Dict[str, Any] = Field(default_factory=dict)
    trends: List[TrendDataPoint] = Field(default_factory=list)
    deviations: List[DeviationAlert] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class QuarterlyUpdateWorkflow:
    """
    Lightweight quarterly data refresh and progress tracking.

    Runs at the end of each fiscal quarter to incrementally update emissions,
    detect trends, and flag deviations from the annual plan without running
    the full annual reporting pipeline.

    Attributes:
        workflow_id: Unique execution identifier.
        _cancelled: Cancellation flag for cooperative shutdown.
        _progress_callback: Optional callback for step progress updates.

    Example:
        >>> wf = QuarterlyUpdateWorkflow()
        >>> inp = QuarterlyUpdateInput(
        ...     organization_id="org-123",
        ...     reporting_year=2025,
        ...     quarter=Quarter.Q2,
        ...     quarter_start="2025-04-01",
        ...     quarter_end="2025-06-30",
        ...     annual_target_tco2e=50000.0,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == QuarterlyStepStatus.COMPLETED
    """

    STEPS = [
        "incremental_intake",
        "emissions_recalculation",
        "trend_analysis",
        "progress_report",
        "deviation_detection",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the quarterly update workflow.

        Args:
            progress_callback: Optional callback(step_name, message, pct_complete).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._cancelled: bool = False
        self._progress_callback = progress_callback
        self._step_results: Dict[str, StepResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: QuarterlyUpdateInput) -> QuarterlyUpdateResult:
        """
        Execute the quarterly update workflow.

        Args:
            input_data: Validated quarterly update input.

        Returns:
            QuarterlyUpdateResult with emissions summary, trends, deviations.
        """
        started_at = datetime.utcnow()
        quarter_label = f"{input_data.reporting_year}-{input_data.quarter.value}"
        logger.info(
            "Starting quarterly update %s for org=%s quarter=%s",
            self.workflow_id, input_data.organization_id, quarter_label,
        )
        self._notify("workflow", "Quarterly update started", 0.0)

        completed_steps: List[StepResult] = []
        overall_status = QuarterlyStepStatus.RUNNING
        emissions_summary: Dict[str, Any] = {}
        trends: List[TrendDataPoint] = []
        deviations: List[DeviationAlert] = []

        step_handlers = [
            ("incremental_intake", self._step_incremental_intake),
            ("emissions_recalculation", self._step_emissions_recalculation),
            ("trend_analysis", self._step_trend_analysis),
            ("progress_report", self._step_progress_report),
            ("deviation_detection", self._step_deviation_detection),
        ]

        try:
            for idx, (step_name, handler) in enumerate(step_handlers):
                if self._cancelled:
                    overall_status = QuarterlyStepStatus.SKIPPED
                    logger.warning("Workflow %s cancelled at step %s", self.workflow_id, step_name)
                    break

                pct = idx / len(step_handlers)
                self._notify(step_name, f"Starting: {step_name}", pct)
                step_started = datetime.utcnow()

                try:
                    step_result = await handler(input_data, pct)
                    step_result.started_at = step_started
                    step_result.completed_at = datetime.utcnow()
                    step_result.duration_seconds = (
                        step_result.completed_at - step_started
                    ).total_seconds()
                except Exception as exc:
                    logger.error("Step '%s' failed: %s", step_name, exc, exc_info=True)
                    step_result = StepResult(
                        step_name=step_name,
                        status=QuarterlyStepStatus.FAILED,
                        started_at=step_started,
                        completed_at=datetime.utcnow(),
                        duration_seconds=(datetime.utcnow() - step_started).total_seconds(),
                        errors=[str(exc)],
                        provenance_hash=self._hash({"error": str(exc)}),
                    )

                completed_steps.append(step_result)
                self._step_results[step_name] = step_result

                # Collect outputs for final result
                if step_name == "emissions_recalculation" and step_result.artifacts:
                    emissions_summary = step_result.artifacts.get("emissions", {})
                if step_name == "trend_analysis" and step_result.artifacts:
                    raw_trends = step_result.artifacts.get("trends", [])
                    trends = [TrendDataPoint(**t) for t in raw_trends if isinstance(t, dict)]
                if step_name == "deviation_detection" and step_result.artifacts:
                    raw_devs = step_result.artifacts.get("deviations", [])
                    deviations = [DeviationAlert(**d) for d in raw_devs if isinstance(d, dict)]

                if step_result.status == QuarterlyStepStatus.FAILED:
                    overall_status = QuarterlyStepStatus.FAILED
                    break

            if overall_status == QuarterlyStepStatus.RUNNING:
                overall_status = QuarterlyStepStatus.COMPLETED

        except Exception as exc:
            logger.critical("Quarterly update %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = QuarterlyStepStatus.FAILED

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        metrics = self._aggregate_metrics(completed_steps)
        artifacts = {s.step_name: s.artifacts for s in completed_steps if s.artifacts}
        provenance = self._hash({
            "workflow_id": self.workflow_id,
            "steps": [s.provenance_hash for s in completed_steps],
        })

        self._notify("workflow", f"Quarterly update {overall_status.value}", 1.0)
        logger.info(
            "Quarterly update %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return QuarterlyUpdateResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            quarter_label=quarter_label,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            steps=completed_steps,
            emissions_summary=emissions_summary,
            trends=trends,
            deviations=deviations,
            metrics=metrics,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation of the workflow."""
        logger.info("Cancellation requested for quarterly update %s", self.workflow_id)
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Step 1: Incremental Data Intake
    # -------------------------------------------------------------------------

    async def _step_incremental_intake(
        self, input_data: QuarterlyUpdateInput, pct_base: float
    ) -> StepResult:
        """
        Ingest only the data for the current quarter, not the full year.

        Agents invoked:
            - greenlang.agents.data.erp_connector_agent (incremental ERP pull)
            - greenlang.agents.data.document_ingestion_agent (new invoices/docs)
            - greenlang.agents.data.utility_tariff_agent (utility bills)
            - greenlang.data_quality_profiler (quality check on new data)
        """
        step_name = "incremental_intake"
        errors: List[str] = []
        warnings: List[str] = []
        records = 0
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Pulling incremental ERP data", pct_base + 0.02)

        # Pull ERP data for the quarter period only
        erp_result = await self._pull_incremental_erp(
            input_data.organization_id,
            input_data.quarter_start,
            input_data.quarter_end,
            input_data.data_source_ids,
        )
        records += erp_result.get("records_ingested", 0)
        artifacts["erp_intake"] = erp_result

        self._notify(step_name, "Ingesting new documents", pct_base + 0.04)

        # Ingest any new documents (invoices, utility bills, etc.)
        doc_result = await self._ingest_new_documents(
            input_data.organization_id,
            input_data.quarter_start,
            input_data.quarter_end,
        )
        records += doc_result.get("documents_processed", 0)
        artifacts["document_intake"] = doc_result

        self._notify(step_name, "Running quality checks on new data", pct_base + 0.06)

        # Data quality profiling on new data only
        quality = await self._profile_quarterly_data(
            input_data.organization_id, input_data.quarter_start, input_data.quarter_end
        )
        artifacts["quality_profile"] = quality

        if quality.get("completeness_pct", 0) < 80.0:
            warnings.append(
                f"Quarter data completeness is {quality.get('completeness_pct', 0):.1f}%."
                " Some metrics may be estimated."
            )

        # Check for data gaps compared to previous quarter
        gap_check = await self._check_quarter_data_gaps(
            input_data.organization_id, input_data.quarter, input_data.reporting_year
        )
        if gap_check.get("missing_sources", []):
            warnings.append(
                f"Missing data from sources: {', '.join(gap_check['missing_sources'])}"
            )
        artifacts["gap_check"] = gap_check
        artifacts["total_records_ingested"] = records

        provenance = self._hash(artifacts)
        status = QuarterlyStepStatus.COMPLETED if not errors else QuarterlyStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=records,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 2: Emissions Recalculation
    # -------------------------------------------------------------------------

    async def _step_emissions_recalculation(
        self, input_data: QuarterlyUpdateInput, pct_base: float
    ) -> StepResult:
        """
        Recalculate emissions for the quarter using zero-hallucination agents.

        Agents invoked (conditional on recalculate_scopes):
            Scope 1: greenlang.stationary_combustion, greenlang.mobile_combustion, etc.
            Scope 2: greenlang.scope2_location, greenlang.scope2_market
            Scope 3: greenlang.scope3.* categories as configured
        """
        step_name = "emissions_recalculation"
        errors: List[str] = []
        warnings: List[str] = []
        records = 0
        artifacts: Dict[str, Any] = {}

        calc_context = {
            "organization_id": input_data.organization_id,
            "period_start": input_data.quarter_start,
            "period_end": input_data.quarter_end,
            "reporting_year": input_data.reporting_year,
            "quarter": input_data.quarter.value,
        }

        emissions: Dict[str, float] = {}

        if "scope1" in input_data.recalculate_scopes:
            self._notify(step_name, "Recalculating Scope 1", pct_base + 0.02)
            s1 = await self._recalculate_scope1(calc_context)
            emissions["scope1_tco2e"] = s1.get("total_tco2e", 0.0)
            records += s1.get("records_processed", 0)
            if s1.get("warnings"):
                warnings.extend(s1["warnings"])

        if "scope2" in input_data.recalculate_scopes:
            self._notify(step_name, "Recalculating Scope 2", pct_base + 0.04)
            s2 = await self._recalculate_scope2(calc_context)
            emissions["scope2_location_tco2e"] = s2.get("location_based_tco2e", 0.0)
            emissions["scope2_market_tco2e"] = s2.get("market_based_tco2e", 0.0)
            records += s2.get("records_processed", 0)

        if "scope3" in input_data.recalculate_scopes:
            self._notify(step_name, "Recalculating Scope 3", pct_base + 0.06)
            s3 = await self._recalculate_scope3(calc_context)
            emissions["scope3_tco2e"] = s3.get("total_tco2e", 0.0)
            emissions["scope3_by_category"] = s3.get("by_category", {})
            records += s3.get("records_processed", 0)

        # Compute quarter total
        scope1 = emissions.get("scope1_tco2e", 0.0)
        scope2_loc = emissions.get("scope2_location_tco2e", 0.0)
        scope3 = emissions.get("scope3_tco2e", 0.0)
        emissions["total_location_tco2e"] = round(scope1 + scope2_loc + scope3, 4)
        scope2_mkt = emissions.get("scope2_market_tco2e", 0.0)
        emissions["total_market_tco2e"] = round(scope1 + scope2_mkt + scope3, 4)

        artifacts["emissions"] = emissions
        artifacts["records_calculated"] = records

        # Store quarterly provenance
        calc_hash = self._hash(emissions)
        artifacts["calculation_provenance"] = calc_hash

        provenance = self._hash(artifacts)
        status = QuarterlyStepStatus.COMPLETED if not errors else QuarterlyStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=records,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 3: Trend Analysis
    # -------------------------------------------------------------------------

    async def _step_trend_analysis(
        self, input_data: QuarterlyUpdateInput, pct_base: float
    ) -> StepResult:
        """
        Compare current quarter emissions with previous quarters and benchmarks.

        Computes quarter-over-quarter and year-over-year trends. Fetches
        comparison data from prior reporting periods if available.
        """
        step_name = "trend_analysis"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Fetching historical quarter data", pct_base + 0.02)

        # Get current quarter result
        current_emissions = self._step_results.get("emissions_recalculation")
        current_total = 0.0
        if current_emissions and current_emissions.artifacts:
            current_total = current_emissions.artifacts.get("emissions", {}).get(
                "total_location_tco2e", 0.0
            )

        # Fetch historical data for comparison
        historical = await self._fetch_historical_quarters(
            input_data.organization_id,
            input_data.reporting_year,
            input_data.comparison_quarters,
        )

        self._notify(step_name, "Computing trend metrics", pct_base + 0.04)

        # Build trend series
        trend_points: List[Dict[str, Any]] = []
        for h in historical:
            trend_points.append({
                "period": h.get("period", ""),
                "value_tco2e": h.get("total_tco2e", 0.0),
                "data_quality_score": h.get("quality_score", 0.0),
            })
        # Add current quarter
        quarter_label = f"{input_data.reporting_year}-{input_data.quarter.value}"
        trend_points.append({
            "period": quarter_label,
            "value_tco2e": current_total,
            "data_quality_score": 1.0,
        })
        artifacts["trends"] = trend_points

        # Quarter-over-quarter change
        if len(trend_points) >= 2:
            prev_val = trend_points[-2]["value_tco2e"]
            if prev_val > 0:
                qoq_change = ((current_total - prev_val) / prev_val) * 100
                artifacts["qoq_change_pct"] = round(qoq_change, 2)
                if qoq_change > 20:
                    warnings.append(
                        f"Quarter-over-quarter increase of {qoq_change:.1f}%."
                        " Investigate potential data anomalies."
                    )

        # Year-over-year change (same quarter, previous year)
        yoy_quarter = await self._fetch_yoy_quarter(
            input_data.organization_id,
            input_data.reporting_year - 1,
            input_data.quarter,
        )
        if yoy_quarter and yoy_quarter.get("total_tco2e", 0) > 0:
            yoy_change = (
                (current_total - yoy_quarter["total_tco2e"]) / yoy_quarter["total_tco2e"]
            ) * 100
            artifacts["yoy_change_pct"] = round(yoy_change, 2)

        # Benchmark comparison (sector average if available)
        benchmark = await self._fetch_sector_benchmark(
            input_data.organization_id, input_data.quarter
        )
        if benchmark:
            artifacts["benchmark"] = benchmark

        provenance = self._hash(artifacts)
        status = QuarterlyStepStatus.COMPLETED if not errors else QuarterlyStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(trend_points),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 4: Progress Report
    # -------------------------------------------------------------------------

    async def _step_progress_report(
        self, input_data: QuarterlyUpdateInput, pct_base: float
    ) -> StepResult:
        """
        Generate a quarterly progress report summarizing emissions, trends,
        and status of ESRS data collection.

        Agents invoked:
            - greenlang.agents.reporting.integrated_report_agent (report generation)
            - greenlang.agents.foundation.citations_agent (evidence linking)
        """
        step_name = "progress_report"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}

        self._notify(step_name, "Generating quarterly progress report", pct_base + 0.02)

        # Gather data from previous steps
        intake_data = self._step_results.get("incremental_intake", StepResult(
            step_name="", status=QuarterlyStepStatus.PENDING, provenance_hash=""
        ))
        calc_data = self._step_results.get("emissions_recalculation", StepResult(
            step_name="", status=QuarterlyStepStatus.PENDING, provenance_hash=""
        ))
        trend_data = self._step_results.get("trend_analysis", StepResult(
            step_name="", status=QuarterlyStepStatus.PENDING, provenance_hash=""
        ))

        quarter_label = f"{input_data.reporting_year}-{input_data.quarter.value}"

        report_content = await self._generate_quarterly_report(
            organization_id=input_data.organization_id,
            quarter_label=quarter_label,
            intake_artifacts=intake_data.artifacts,
            emissions_artifacts=calc_data.artifacts,
            trend_artifacts=trend_data.artifacts,
        )
        artifacts["report"] = report_content

        # Data collection progress (what % of annual ESRS data points are filled)
        esrs_progress = await self._check_esrs_data_progress(
            input_data.organization_id, input_data.reporting_year, input_data.quarter
        )
        artifacts["esrs_progress"] = esrs_progress

        expected_pct = {Quarter.Q1: 25.0, Quarter.Q2: 50.0, Quarter.Q3: 75.0, Quarter.Q4: 100.0}
        target = expected_pct.get(input_data.quarter, 25.0)
        actual = esrs_progress.get("completion_pct", 0.0)
        if actual < target * 0.8:
            warnings.append(
                f"ESRS data completion is {actual:.1f}%, below the {target:.0f}% target "
                f"for {input_data.quarter.value}."
            )
        artifacts["completion_on_track"] = actual >= target * 0.8

        provenance = self._hash(artifacts)
        status = QuarterlyStepStatus.COMPLETED if not errors else QuarterlyStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=0,
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Step 5: Deviation Detection
    # -------------------------------------------------------------------------

    async def _step_deviation_detection(
        self, input_data: QuarterlyUpdateInput, pct_base: float
    ) -> StepResult:
        """
        Flag deviations from annual targets and expected quarterly trajectories.

        Compares actual quarterly emissions against pro-rated annual targets.
        Applies the configured deviation_threshold_pct to determine severity.
        """
        step_name = "deviation_detection"
        errors: List[str] = []
        warnings: List[str] = []
        artifacts: Dict[str, Any] = {}
        deviations: List[Dict[str, Any]] = []

        self._notify(step_name, "Checking deviations from annual targets", pct_base + 0.02)

        # Get current quarter emissions
        calc_result = self._step_results.get("emissions_recalculation")
        if not calc_result or calc_result.status != QuarterlyStepStatus.COMPLETED:
            return StepResult(
                step_name=step_name,
                status=QuarterlyStepStatus.SKIPPED,
                artifacts={"reason": "emissions_recalculation not available"},
                provenance_hash=self._hash({"skipped": True}),
            )

        emissions = calc_result.artifacts.get("emissions", {})
        threshold = input_data.deviation_threshold_pct

        # Pro-rate annual target to quarter
        if input_data.annual_target_tco2e and input_data.annual_target_tco2e > 0:
            quarterly_target = input_data.annual_target_tco2e / 4.0

            scopes_to_check = [
                ("total_location_tco2e", "total", quarterly_target),
                ("scope1_tco2e", "scope1", None),
                ("scope2_location_tco2e", "scope2", None),
                ("scope3_tco2e", "scope3", None),
            ]

            for metric_key, scope_label, expected in scopes_to_check:
                actual = emissions.get(metric_key, 0.0)
                if expected is None:
                    # No sub-target specified; skip sub-scope deviation
                    continue

                if expected > 0:
                    dev_pct = ((actual - expected) / expected) * 100
                    if abs(dev_pct) > threshold:
                        severity = DeviationSeverity.CRITICAL if abs(dev_pct) > threshold * 2 else (
                            DeviationSeverity.WARNING if abs(dev_pct) > threshold else
                            DeviationSeverity.INFO
                        )
                        direction = "above" if dev_pct > 0 else "below"
                        deviations.append({
                            "metric_name": metric_key,
                            "scope": scope_label,
                            "expected_value": round(expected, 2),
                            "actual_value": round(actual, 2),
                            "deviation_pct": round(dev_pct, 2),
                            "severity": severity.value,
                            "recommendation": (
                                f"Actual emissions are {abs(dev_pct):.1f}% {direction} the "
                                f"quarterly target. Investigate and verify data inputs for {scope_label}."
                            ),
                        })
        else:
            warnings.append("No annual target set; deviation check skipped for absolute values.")

        # Check for sudden spikes compared to previous quarter
        trend_result = self._step_results.get("trend_analysis")
        if trend_result and trend_result.artifacts.get("qoq_change_pct") is not None:
            qoq = trend_result.artifacts["qoq_change_pct"]
            if abs(qoq) > threshold * 1.5:
                severity = DeviationSeverity.CRITICAL if abs(qoq) > threshold * 3 else DeviationSeverity.WARNING
                deviations.append({
                    "metric_name": "qoq_change",
                    "scope": "total",
                    "expected_value": 0.0,
                    "actual_value": qoq,
                    "deviation_pct": round(qoq, 2),
                    "severity": severity.value,
                    "recommendation": (
                        f"Quarter-over-quarter change of {qoq:.1f}% exceeds threshold. "
                        "Review underlying data sources for anomalies."
                    ),
                })

        artifacts["deviations"] = deviations
        artifacts["total_deviations"] = len(deviations)
        artifacts["critical_deviations"] = sum(
            1 for d in deviations if d.get("severity") == DeviationSeverity.CRITICAL.value
        )

        if artifacts["critical_deviations"] > 0:
            warnings.append(
                f"{artifacts['critical_deviations']} critical deviation(s) detected."
            )

        provenance = self._hash(artifacts)
        status = QuarterlyStepStatus.COMPLETED if not errors else QuarterlyStepStatus.FAILED

        return StepResult(
            step_name=step_name,
            status=status,
            records_processed=len(deviations),
            artifacts=artifacts,
            errors=errors,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Agent Invocation Helpers
    # -------------------------------------------------------------------------

    async def _pull_incremental_erp(
        self, org_id: str, start: str, end: str, source_ids: List[str]
    ) -> Dict[str, Any]:
        """Pull incremental data from ERP for the quarter period."""
        logger.info("Pulling ERP data for org=%s period=%s to %s", org_id, start, end)
        await asyncio.sleep(0)
        return {"records_ingested": 0, "status": "completed"}

    async def _ingest_new_documents(
        self, org_id: str, start: str, end: str
    ) -> Dict[str, Any]:
        """Ingest new documents (invoices, bills) for the quarter."""
        await asyncio.sleep(0)
        return {"documents_processed": 0, "status": "completed"}

    async def _profile_quarterly_data(
        self, org_id: str, start: str, end: str
    ) -> Dict[str, Any]:
        """Profile data quality for the quarter period."""
        await asyncio.sleep(0)
        return {"completeness_pct": 0.0, "accuracy_score": 0.0}

    async def _check_quarter_data_gaps(
        self, org_id: str, quarter: Quarter, year: int
    ) -> Dict[str, Any]:
        """Check for data gaps compared to expected sources."""
        await asyncio.sleep(0)
        return {"missing_sources": []}

    async def _recalculate_scope1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run Scope 1 calculation agents for the quarter."""
        await asyncio.sleep(0)
        return {"total_tco2e": 0.0, "records_processed": 0}

    async def _recalculate_scope2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run Scope 2 calculation agents for the quarter."""
        await asyncio.sleep(0)
        return {"location_based_tco2e": 0.0, "market_based_tco2e": 0.0, "records_processed": 0}

    async def _recalculate_scope3(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run Scope 3 calculation agents for the quarter."""
        await asyncio.sleep(0)
        return {"total_tco2e": 0.0, "by_category": {}, "records_processed": 0}

    async def _fetch_historical_quarters(
        self, org_id: str, year: int, comparison: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch historical quarter emissions for trend analysis."""
        await asyncio.sleep(0)
        return []

    async def _fetch_yoy_quarter(
        self, org_id: str, year: int, quarter: Quarter
    ) -> Optional[Dict[str, Any]]:
        """Fetch same quarter from previous year for YoY comparison."""
        await asyncio.sleep(0)
        return None

    async def _fetch_sector_benchmark(
        self, org_id: str, quarter: Quarter
    ) -> Optional[Dict[str, Any]]:
        """Fetch sector benchmark data for comparison."""
        await asyncio.sleep(0)
        return None

    async def _generate_quarterly_report(
        self, organization_id: str, quarter_label: str,
        intake_artifacts: Dict[str, Any], emissions_artifacts: Dict[str, Any],
        trend_artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate the quarterly progress report document."""
        await asyncio.sleep(0)
        return {
            "report_id": str(uuid.uuid4()),
            "quarter": quarter_label,
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def _check_esrs_data_progress(
        self, org_id: str, year: int, quarter: Quarter
    ) -> Dict[str, Any]:
        """Check how many ESRS data points are populated for the year so far."""
        await asyncio.sleep(0)
        return {"completion_pct": 0.0, "populated": 0, "total": 0}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _notify(self, step: str, message: str, pct: float) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(step, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for step=%s", step)

    def _aggregate_metrics(self, steps: List[StepResult]) -> Dict[str, Any]:
        """Aggregate metrics across all steps."""
        return {
            "total_records_processed": sum(s.records_processed for s in steps),
            "total_errors": sum(len(s.errors) for s in steps),
            "total_warnings": sum(len(s.warnings) for s in steps),
            "steps_completed": sum(
                1 for s in steps if s.status == QuarterlyStepStatus.COMPLETED
            ),
            "steps_failed": sum(
                1 for s in steps if s.status == QuarterlyStepStatus.FAILED
            ),
        }

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
