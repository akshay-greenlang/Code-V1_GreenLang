"""
DualReportingPipelineEngine - Orchestrates 10-stage reconciliation pipeline.

This module implements the complete pipeline for reconciling dual Scope 2 reporting
(location-based and market-based methods) across multiple regulatory frameworks.

Pipeline Stages:
1. COLLECT_RESULTS - Gather upstream results from location/market engines
2. ALIGN_BOUNDARIES - Verify reporting boundary alignment
3. MAP_ENERGY_TYPES - Normalize energy type classifications
4. ANALYZE_DISCREPANCIES - Identify and explain differences
5. SCORE_QUALITY - Assess data quality across methods
6. GENERATE_TABLES - Generate framework-specific reporting tables
7. ANALYZE_TRENDS - Analyze temporal trends (optional)
8. CHECK_COMPLIANCE - Validate regulatory compliance
9. ASSEMBLE_REPORT - Combine all outputs into final report
10. SEAL_PROVENANCE - Calculate SHA-256 provenance hash

Example:
    >>> engine = DualReportingPipelineEngine()
    >>> request = ReconciliationRequest(
    ...     tenant_id="acme-corp",
    ...     period_start=date(2024, 1, 1),
    ...     period_end=date(2024, 12, 31),
    ...     upstream_results=[location_result, market_result],
    ...     frameworks=[ReportingFramework.GHG_PROTOCOL]
    ... )
    >>> report = engine.run_pipeline(request)
    >>> assert report.status == ReconciliationStatus.COMPLETED
"""

from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
import hashlib
import json
import threading
import logging
import csv
from io import StringIO

from pydantic import BaseModel, Field, validator

from greenlang.exceptions import WorkflowException
from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
    EnergyType,
    Scope2Method,
    PipelineStage,
    ReconciliationStatus,
    ReportingFramework,
    ExportFormat,
    BatchStatus,
    UpstreamResult,
    ReconciliationWorkspace,
    ReconciliationRequest,
    ReconciliationReport,
    DiscrepancyReport,
    DiscrepancyItem,
    DiscrepancyType,
    QualityAssessment,
    ReportingTableSet,
    TrendReport,
    TrendDataPoint,
    BatchReconciliationRequest,
    BatchReconciliationResult,
    AggregationResult,
    ExportRequest,
    ComplianceCheckResult,
    ComplianceIssue,
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    MAX_UPSTREAM_RESULTS,
    MAX_BATCH_PERIODS,
    DECIMAL_PLACES,
    ZERO,
    ONE_HUNDRED,
)
from greenlang.agents.mrv.dual_reporting_reconciliation.config import DualReportingReconciliationConfig
from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import DualReportingReconciliationMetrics
from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import DualReportingReconciliationProvenance

logger = logging.getLogger(__name__)

__all__ = [
    "DualReportingPipelineEngine",
    "PipelineExecutionError",
    "StageExecutionError",
]


class PipelineExecutionError(WorkflowException):
    """Raised when pipeline execution fails."""

    def __init__(self, message: str, stage: Optional[PipelineStage] = None, details: Optional[Dict] = None):
        """Initialize pipeline execution error."""
        self.stage = stage
        self.details = details or {}
        super().__init__(message)


class StageExecutionError(WorkflowException):
    """Raised when a specific stage fails."""

    def __init__(self, stage: PipelineStage, message: str, details: Optional[Dict] = None):
        """Initialize stage execution error."""
        self.stage = stage
        self.details = details or {}
        super().__init__(f"Stage {stage.value} failed: {message}")


class PipelineContext(BaseModel):
    """Pipeline execution context."""

    request: ReconciliationRequest
    workspace: Optional[ReconciliationWorkspace] = None
    discrepancy_report: Optional[DiscrepancyReport] = None
    quality_assessment: Optional[QualityAssessment] = None
    reporting_tables: Optional[ReportingTableSet] = None
    trend_report: Optional[TrendReport] = None
    compliance_results: Dict[str, ComplianceCheckResult] = {}
    completed_stages: List[str] = []
    stage_timings_ms: Dict[str, Decimal] = {}
    errors: List[str] = []

    class Config:
        arbitrary_types_allowed = True


class DualReportingPipelineEngine:
    """
    Pipeline orchestrator for dual Scope 2 reporting reconciliation.

    This engine coordinates the 10-stage reconciliation pipeline, managing
    data flow between specialized engines and ensuring complete provenance.

    Thread-safe singleton implementation with lazy engine initialization.

    Attributes:
        config: Engine configuration
        metrics: Metrics tracker
        provenance: Provenance tracker
    """

    _instance = None
    _initialized = False
    _lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize pipeline engine (once)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.config = DualReportingReconciliationConfig()
            self.metrics = DualReportingReconciliationMetrics()
            self.provenance = DualReportingReconciliationProvenance()

            # Lazy-loaded engines (initialized on first use)
            self._collector = None
            self._discrepancy_analyzer = None
            self._quality_scorer = None
            self._table_generator = None
            self._trend_analyzer = None
            self._compliance_checker = None

            # Pipeline state
            self._current_pipeline_id: Optional[str] = None
            self._current_stage: Optional[PipelineStage] = None
            self._pipeline_start_time: Optional[datetime] = None

            self._initialized = True
            logger.info("%s-DualReportingPipelineEngine initialized (v%s)", AGENT_COMPONENT, VERSION)

    def _get_collector(self):
        """Lazy load result collector engine."""
        if self._collector is None:
            from greenlang.agents.mrv.dual_reporting_reconciliation.dual_result_collector import (
                DualResultCollectorEngine,
            )
            self._collector = DualResultCollectorEngine()
        return self._collector

    def _get_discrepancy_analyzer(self):
        """Lazy load discrepancy analyzer engine."""
        if self._discrepancy_analyzer is None:
            from greenlang.agents.mrv.dual_reporting_reconciliation.discrepancy_analyzer import (
                DiscrepancyAnalyzerEngine,
            )
            self._discrepancy_analyzer = DiscrepancyAnalyzerEngine()
        return self._discrepancy_analyzer

    def _get_quality_scorer(self):
        """Lazy load quality scorer engine."""
        if self._quality_scorer is None:
            from greenlang.agents.mrv.dual_reporting_reconciliation.quality_scorer import QualityScorerEngine
            self._quality_scorer = QualityScorerEngine()
        return self._quality_scorer

    def _get_table_generator(self):
        """Lazy load reporting table generator engine."""
        if self._table_generator is None:
            from greenlang.agents.mrv.dual_reporting_reconciliation.reporting_table_generator import (
                ReportingTableGeneratorEngine,
            )
            self._table_generator = ReportingTableGeneratorEngine()
        return self._table_generator

    def _get_trend_analyzer(self):
        """Lazy load trend analyzer engine."""
        if self._trend_analyzer is None:
            from greenlang.agents.mrv.dual_reporting_reconciliation.trend_analyzer import TrendAnalysisEngine
            self._trend_analyzer = TrendAnalysisEngine()
        return self._trend_analyzer

    def _get_compliance_checker(self):
        """Lazy load compliance checker engine."""
        if self._compliance_checker is None:
            from greenlang.agents.mrv.dual_reporting_reconciliation.compliance_checker import (
                ComplianceCheckerEngine,
            )
            self._compliance_checker = ComplianceCheckerEngine()
        return self._compliance_checker

    def run_pipeline(self, request: ReconciliationRequest) -> ReconciliationReport:
        """
        Execute complete 10-stage reconciliation pipeline.

        Args:
            request: Reconciliation request with upstream results

        Returns:
            Complete reconciliation report with provenance

        Raises:
            PipelineExecutionError: If pipeline execution fails
            ValueError: If request validation fails
        """
        pipeline_id = str(uuid4())
        self._current_pipeline_id = pipeline_id
        self._pipeline_start_time = datetime.utcnow()

        logger.info(
            f"Starting pipeline {pipeline_id} for tenant {request.tenant_id}, "
            f"period {request.period_start} to {request.period_end}"
        )

        try:
            # Validate request
            self._validate_request(request)

            # Initialize context
            context = PipelineContext(request=request)

            # Execute all 10 stages
            stages = list(PipelineStage)
            for stage in stages:
                stage_start = datetime.utcnow()
                self._current_stage = stage

                logger.info("Pipeline %s: Executing stage %s", pipeline_id, stage.value)

                try:
                    context = self._execute_stage(stage, context)
                    context.completed_stages.append(stage.value)

                    # Calculate stage timing
                    stage_duration = (datetime.utcnow() - stage_start).total_seconds() * 1000
                    context.stage_timings_ms[stage.value] = Decimal(str(stage_duration)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

                    # Record metrics
                    self.metrics.record_stage_execution(
                        stage=stage.value,
                        duration_ms=float(context.stage_timings_ms[stage.value]),
                        success=True,
                    )

                except Exception as e:
                    logger.error("Pipeline %s: Stage %s failed: %s", pipeline_id, stage.value, e, exc_info=True)
                    self.metrics.record_stage_execution(
                        stage=stage.value,
                        duration_ms=0.0,
                        success=False,
                    )
                    raise StageExecutionError(stage=stage, message=str(e))

            # Pipeline complete
            total_duration = (datetime.utcnow() - self._pipeline_start_time).total_seconds() * 1000
            total_duration_decimal = Decimal(str(total_duration)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            # Final report should be in context after stage 10
            if not hasattr(context, "final_report") or context.final_report is None:
                raise PipelineExecutionError("Pipeline completed but no final report generated")

            report = context.final_report
            report.pipeline_duration_ms = total_duration_decimal

            # Record pipeline metrics
            self.metrics.record_pipeline_execution(
                tenant_id=request.tenant_id,
                duration_ms=float(total_duration_decimal),
                success=True,
                upstream_count=len(request.upstream_results),
            )

            logger.info(
                f"Pipeline {pipeline_id} completed successfully in {total_duration_decimal}ms, "
                f"status={report.status.value}"
            )

            return report

        except Exception as e:
            logger.error("Pipeline %s failed: %s", pipeline_id, e, exc_info=True)

            if self._pipeline_start_time:
                total_duration = (datetime.utcnow() - self._pipeline_start_time).total_seconds() * 1000
                self.metrics.record_pipeline_execution(
                    tenant_id=request.tenant_id,
                    duration_ms=total_duration,
                    success=False,
                    upstream_count=len(request.upstream_results),
                )

            raise PipelineExecutionError(
                message=f"Pipeline execution failed: {str(e)}",
                stage=self._current_stage,
                details={"pipeline_id": pipeline_id},
            ) from e

        finally:
            self._current_pipeline_id = None
            self._current_stage = None
            self._pipeline_start_time = None

    def _validate_request(self, request: ReconciliationRequest) -> None:
        """
        Validate reconciliation request.

        Args:
            request: Request to validate

        Raises:
            ValueError: If validation fails
        """
        # Check upstream results count
        if len(request.upstream_results) > MAX_UPSTREAM_RESULTS:
            raise ValueError(
                f"Too many upstream results: {len(request.upstream_results)} "
                f"(max {MAX_UPSTREAM_RESULTS})"
            )

        if len(request.upstream_results) == 0:
            raise ValueError("At least one upstream result required")

        # Check for location and market methods
        methods = {result.method for result in request.upstream_results}
        if Scope2Method.LOCATION_BASED not in methods:
            raise ValueError("Location-based result required")
        if Scope2Method.MARKET_BASED not in methods:
            raise ValueError("Market-based result required")

        # Validate period
        if request.period_end < request.period_start:
            raise ValueError(f"Invalid period: end {request.period_end} < start {request.period_start}")

        # Validate frameworks
        if not request.frameworks:
            raise ValueError("At least one reporting framework required")

        # Validate trend data if trends requested
        if request.include_trends and not request.trend_data:
            logger.warning("Trends requested but no historical data provided, will skip trend analysis")

    def _execute_stage(self, stage: PipelineStage, context: PipelineContext) -> PipelineContext:
        """
        Execute single pipeline stage.

        Args:
            stage: Stage to execute
            context: Current pipeline context

        Returns:
            Updated pipeline context

        Raises:
            StageExecutionError: If stage execution fails
        """
        try:
            if stage == PipelineStage.COLLECT_RESULTS:
                context.workspace = self._stage_collect_results(context.request)

            elif stage == PipelineStage.ALIGN_BOUNDARIES:
                context.workspace = self._stage_align_boundaries(context.workspace)

            elif stage == PipelineStage.MAP_ENERGY_TYPES:
                context.workspace = self._stage_map_energy_types(context.workspace)

            elif stage == PipelineStage.ANALYZE_DISCREPANCIES:
                context.discrepancy_report = self._stage_analyze_discrepancies(context.workspace)

            elif stage == PipelineStage.SCORE_QUALITY:
                context.quality_assessment = self._stage_score_quality(context.workspace)

            elif stage == PipelineStage.GENERATE_TABLES:
                context.reporting_tables = self._stage_generate_tables(
                    workspace=context.workspace,
                    discrepancy_report=context.discrepancy_report,
                    quality_assessment=context.quality_assessment,
                    frameworks=context.request.frameworks,
                )

            elif stage == PipelineStage.ANALYZE_TRENDS:
                context.trend_report = self._stage_analyze_trends(
                    trend_data=context.request.trend_data if context.request.include_trends else [],
                    config_params=context.request.metadata,
                )

            elif stage == PipelineStage.CHECK_COMPLIANCE:
                context.compliance_results = self._stage_check_compliance(
                    workspace=context.workspace,
                    discrepancy_report=context.discrepancy_report,
                    quality_assessment=context.quality_assessment,
                    frameworks=context.request.frameworks,
                )

            elif stage == PipelineStage.ASSEMBLE_REPORT:
                report = self._stage_assemble_report(
                    workspace=context.workspace,
                    discrepancy_report=context.discrepancy_report,
                    quality_assessment=context.quality_assessment,
                    reporting_tables=context.reporting_tables,
                    trend_report=context.trend_report,
                    compliance_results=context.compliance_results,
                    request=context.request,
                    completed_stages=context.completed_stages,
                )
                context.final_report = report

            elif stage == PipelineStage.SEAL_PROVENANCE:
                context.final_report = self._stage_seal_provenance(context.final_report)

            else:
                raise StageExecutionError(stage=stage, message=f"Unknown stage: {stage.value}")

            return context

        except Exception as e:
            raise StageExecutionError(
                stage=stage,
                message=str(e),
                details={"context": context.dict() if hasattr(context, "dict") else str(context)},
            ) from e

    def _stage_collect_results(self, request: ReconciliationRequest) -> ReconciliationWorkspace:
        """
        Stage 1: Collect and organize upstream results.

        Args:
            request: Reconciliation request

        Returns:
            Workspace with collected results
        """
        collector = self._get_collector()
        workspace = collector.collect_results(request.upstream_results)

        logger.info(
            f"Collected {len(workspace.location_results)} location-based and "
            f"{len(workspace.market_results)} market-based results"
        )

        return workspace

    def _stage_align_boundaries(self, workspace: ReconciliationWorkspace) -> ReconciliationWorkspace:
        """
        Stage 2: Verify boundary alignment.

        Args:
            workspace: Current workspace

        Returns:
            Updated workspace with alignment verification
        """
        collector = self._get_collector()

        # Check boundary alignment
        alignment_issues = collector.verify_boundary_alignment(workspace)

        if alignment_issues:
            logger.warning("Found %s boundary alignment issues", len(alignment_issues))
            # Store issues in metadata
            workspace.metadata["boundary_alignment_issues"] = [
                {
                    "location_boundary": issue.get("location_boundary"),
                    "market_boundary": issue.get("market_boundary"),
                    "mismatch_type": issue.get("mismatch_type"),
                }
                for issue in alignment_issues
            ]
        else:
            logger.info("All boundaries aligned successfully")

        return workspace

    def _stage_map_energy_types(self, workspace: ReconciliationWorkspace) -> ReconciliationWorkspace:
        """
        Stage 3: Map and normalize energy types.

        Args:
            workspace: Current workspace

        Returns:
            Updated workspace with mapped energy types
        """
        collector = self._get_collector()

        # Map energy types
        energy_mapping = collector.map_energy_types(workspace)

        logger.info("Mapped %s energy type classifications", len(energy_mapping))

        # Store mapping in metadata
        workspace.metadata["energy_type_mapping"] = {
            energy_type.value: mapping.dict() for energy_type, mapping in energy_mapping.items()
        }

        return workspace

    def _stage_analyze_discrepancies(self, workspace: ReconciliationWorkspace) -> DiscrepancyReport:
        """
        Stage 4: Analyze discrepancies between methods.

        Args:
            workspace: Current workspace

        Returns:
            Discrepancy analysis report
        """
        analyzer = self._get_discrepancy_analyzer()

        # Run discrepancy analysis
        discrepancy_report = analyzer.analyze_discrepancies(workspace)

        logger.info(
            f"Identified {len(discrepancy_report.items)} discrepancies, "
            f"total delta={discrepancy_report.total_delta_tco2e} tCO2e"
        )

        return discrepancy_report

    def _stage_score_quality(self, workspace: ReconciliationWorkspace) -> QualityAssessment:
        """
        Stage 5: Score data quality.

        Args:
            workspace: Current workspace

        Returns:
            Quality assessment
        """
        scorer = self._get_quality_scorer()

        # Score quality
        quality_assessment = scorer.score_quality(workspace)

        logger.info(
            f"Quality assessment complete: overall_score={quality_assessment.overall_score}, "
            f"location_score={quality_assessment.location_based_score}, "
            f"market_score={quality_assessment.market_based_score}"
        )

        return quality_assessment

    def _stage_generate_tables(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: DiscrepancyReport,
        quality_assessment: QualityAssessment,
        frameworks: List[ReportingFramework],
    ) -> ReportingTableSet:
        """
        Stage 6: Generate framework-specific reporting tables.

        Args:
            workspace: Current workspace
            discrepancy_report: Discrepancy analysis
            quality_assessment: Quality assessment
            frameworks: Frameworks to generate tables for

        Returns:
            Set of reporting tables
        """
        generator = self._get_table_generator()

        # Generate tables for each framework
        table_set = generator.generate_table_set(
            workspace=workspace,
            discrepancy_report=discrepancy_report,
            quality_assessment=quality_assessment,
            frameworks=frameworks,
        )

        logger.info("Generated reporting tables for %s frameworks", len(frameworks))

        return table_set

    def _stage_analyze_trends(
        self,
        trend_data: List[TrendDataPoint],
        config_params: Dict[str, Any],
    ) -> Optional[TrendReport]:
        """
        Stage 7: Analyze temporal trends (optional).

        Args:
            trend_data: Historical trend data points
            config_params: Configuration parameters

        Returns:
            Trend report if data available, None otherwise
        """
        if not trend_data:
            logger.info("No trend data provided, skipping trend analysis")
            return None

        analyzer = self._get_trend_analyzer()

        # Run trend analysis
        trend_report = analyzer.analyze_trends(
            trend_data=trend_data,
            config_params=config_params,
        )

        logger.info(
            f"Trend analysis complete: analyzed {len(trend_data)} periods, "
            f"forecast {len(trend_report.forecast_periods) if trend_report and trend_report.forecast_periods else 0} periods"
        )

        return trend_report

    def _stage_check_compliance(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: DiscrepancyReport,
        quality_assessment: QualityAssessment,
        frameworks: List[ReportingFramework],
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Stage 8: Check regulatory compliance.

        Args:
            workspace: Current workspace
            discrepancy_report: Discrepancy analysis
            quality_assessment: Quality assessment
            frameworks: Frameworks to check compliance for

        Returns:
            Compliance check results by framework
        """
        checker = self._get_compliance_checker()

        compliance_results = {}
        for framework in frameworks:
            result = checker.check_compliance(
                workspace=workspace,
                discrepancy_report=discrepancy_report,
                quality_assessment=quality_assessment,
                framework=framework,
            )
            compliance_results[framework.value] = result

        total_issues = sum(len(result.issues) for result in compliance_results.values())
        logger.info("Compliance check complete: %s frameworks, %s total issues", len(frameworks), total_issues)

        return compliance_results

    def _stage_assemble_report(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: DiscrepancyReport,
        quality_assessment: QualityAssessment,
        reporting_tables: ReportingTableSet,
        trend_report: Optional[TrendReport],
        compliance_results: Dict[str, ComplianceCheckResult],
        request: ReconciliationRequest,
        completed_stages: List[str],
    ) -> ReconciliationReport:
        """
        Stage 9: Assemble final reconciliation report.

        Args:
            workspace: Current workspace
            discrepancy_report: Discrepancy analysis
            quality_assessment: Quality assessment
            reporting_tables: Framework-specific tables
            trend_report: Trend analysis (optional)
            compliance_results: Compliance check results
            request: Original request
            completed_stages: List of completed stage names

        Returns:
            Complete reconciliation report (without provenance hash)
        """
        # Determine overall status
        status = self._determine_report_status(
            discrepancy_report=discrepancy_report,
            quality_assessment=quality_assessment,
            compliance_results=compliance_results,
        )

        report = ReconciliationReport(
            reconciliation_id=self._current_pipeline_id or str(uuid4()),
            tenant_id=request.tenant_id,
            status=status,
            period_start=request.period_start,
            period_end=request.period_end,
            workspace=workspace,
            discrepancy_report=discrepancy_report,
            quality_assessment=quality_assessment,
            reporting_tables=reporting_tables,
            trend_report=trend_report,
            compliance_results=compliance_results,
            pipeline_stages_completed=completed_stages + [PipelineStage.ASSEMBLE_REPORT.value],
            pipeline_duration_ms=ZERO,  # Will be updated after stage 10
            provenance_hash="",  # Will be calculated in stage 10
            created_at=datetime.utcnow(),
        )

        logger.info("Report assembled with status=%s", status.value)

        return report

    def _determine_report_status(
        self,
        discrepancy_report: DiscrepancyReport,
        quality_assessment: QualityAssessment,
        compliance_results: Dict[str, ComplianceCheckResult],
    ) -> ReconciliationStatus:
        """
        Determine overall report status based on analysis results.

        Args:
            discrepancy_report: Discrepancy analysis
            quality_assessment: Quality assessment
            compliance_results: Compliance check results

        Returns:
            Overall reconciliation status
        """
        # Check for critical compliance issues
        critical_issues = []
        for result in compliance_results.values():
            critical_issues.extend([issue for issue in result.issues if issue.severity == "CRITICAL"])

        if critical_issues:
            return ReconciliationStatus.FAILED

        # Check for warnings
        warning_issues = []
        for result in compliance_results.values():
            warning_issues.extend([issue for issue in result.issues if issue.severity == "WARNING"])

        # Check quality score
        min_quality_score = self.config.get("quality", {}).get("min_acceptable_score", 70.0)
        if quality_assessment.overall_score < Decimal(str(min_quality_score)):
            return ReconciliationStatus.COMPLETED_WITH_WARNINGS

        # Check discrepancy threshold
        max_discrepancy_pct = self.config.get("discrepancy", {}).get("max_total_delta_pct", 10.0)
        if abs(discrepancy_report.total_delta_pct) > Decimal(str(max_discrepancy_pct)):
            return ReconciliationStatus.COMPLETED_WITH_WARNINGS

        if warning_issues:
            return ReconciliationStatus.COMPLETED_WITH_WARNINGS

        return ReconciliationStatus.COMPLETED

    def _stage_seal_provenance(self, report: ReconciliationReport) -> ReconciliationReport:
        """
        Stage 10: Calculate final provenance hash.

        Args:
            report: Report to seal

        Returns:
            Report with provenance hash
        """
        # Calculate hash of complete report (excluding provenance_hash field)
        report_dict = report.dict(exclude={"provenance_hash"})
        report_json = json.dumps(report_dict, sort_keys=True, default=str)
        provenance_hash = hashlib.sha256(report_json.encode()).hexdigest()

        report.provenance_hash = provenance_hash

        # Record in provenance tracker
        self.provenance.record_hash(
            component="pipeline",
            operation="seal_report",
            data_hash=provenance_hash,
            metadata={
                "reconciliation_id": report.reconciliation_id,
                "tenant_id": report.tenant_id,
                "period_start": str(report.period_start),
                "period_end": str(report.period_end),
                "status": report.status.value,
            },
        )

        logger.info("Report sealed with provenance hash: %s", provenance_hash)

        return report

    def run_batch(self, request: BatchReconciliationRequest) -> BatchReconciliationResult:
        """
        Process multiple reconciliation periods in batch.

        Args:
            request: Batch reconciliation request

        Returns:
            Batch reconciliation result with aggregation

        Raises:
            ValueError: If batch request validation fails
        """
        batch_id = str(uuid4())
        logger.info("Starting batch reconciliation %s with %s periods", batch_id, len(request.periods))

        # Validate batch size
        if len(request.periods) > MAX_BATCH_PERIODS:
            raise ValueError(
                f"Too many periods in batch: {len(request.periods)} (max {MAX_BATCH_PERIODS})"
            )

        reports: List[ReconciliationReport] = []
        completed_count = 0
        failed_count = 0

        for period_request in request.periods:
            try:
                report = self.run_pipeline(period_request)
                reports.append(report)
                completed_count += 1
            except Exception as e:
                logger.error(
                    f"Batch {batch_id}: Period {period_request.period_start} to "
                    f"{period_request.period_end} failed: {str(e)}"
                )
                failed_count += 1

                if request.fail_fast:
                    logger.error("Batch %s: Fail-fast enabled, stopping batch", batch_id)
                    break

        # Determine batch status
        if failed_count == 0:
            status = BatchStatus.COMPLETED
        elif completed_count == 0:
            status = BatchStatus.FAILED
        else:
            status = BatchStatus.PARTIAL

        # Aggregate results if requested and we have completed reports
        aggregation = None
        if request.include_aggregation and reports:
            aggregation = self.aggregate_results(reports)

        # Calculate batch provenance hash
        batch_data = {
            "batch_id": batch_id,
            "total_periods": len(request.periods),
            "completed_periods": completed_count,
            "failed_periods": failed_count,
            "report_ids": [r.reconciliation_id for r in reports],
        }
        batch_json = json.dumps(batch_data, sort_keys=True)
        batch_hash = hashlib.sha256(batch_json.encode()).hexdigest()

        result = BatchReconciliationResult(
            batch_id=batch_id,
            status=status,
            total_periods=len(request.periods),
            completed_periods=completed_count,
            failed_periods=failed_count,
            reports=reports,
            aggregation=aggregation,
            provenance_hash=batch_hash,
        )

        logger.info(
            f"Batch {batch_id} complete: status={status.value}, "
            f"completed={completed_count}, failed={failed_count}"
        )

        return result

    def aggregate_results(self, reports: List[ReconciliationReport]) -> AggregationResult:
        """
        Aggregate multiple reconciliation reports.

        Args:
            reports: Reports to aggregate

        Returns:
            Aggregation result with totals and averages

        Raises:
            ValueError: If reports list is empty
        """
        if not reports:
            raise ValueError("Cannot aggregate empty report list")

        # Sum total emissions
        total_location_tco2e = sum(
            (r.workspace.total_location_tco2e for r in reports),
            start=ZERO,
        )
        total_market_tco2e = sum(
            (r.workspace.total_market_tco2e for r in reports),
            start=ZERO,
        )

        # Calculate average PIF
        pifs = [r.workspace.pif for r in reports if r.workspace.pif is not None]
        average_pif = (
            sum(pifs, start=ZERO) / Decimal(str(len(pifs)))
            if pifs
            else ZERO
        )

        # Calculate average quality score
        quality_scores = [r.quality_assessment.overall_score for r in reports]
        average_quality_score = sum(quality_scores, start=ZERO) / Decimal(str(len(quality_scores)))

        # Calculate aggregation provenance hash
        agg_data = {
            "total_location_tco2e": str(total_location_tco2e),
            "total_market_tco2e": str(total_market_tco2e),
            "average_pif": str(average_pif),
            "average_quality_score": str(average_quality_score),
            "period_count": len(reports),
        }
        agg_json = json.dumps(agg_data, sort_keys=True)
        agg_hash = hashlib.sha256(agg_json.encode()).hexdigest()

        return AggregationResult(
            total_location_tco2e=total_location_tco2e,
            total_market_tco2e=total_market_tco2e,
            average_pif=average_pif,
            average_quality_score=average_quality_score,
            period_count=len(reports),
            provenance_hash=agg_hash,
        )

    def export_report(self, report: ReconciliationReport, export_format: ExportFormat) -> str:
        """
        Export reconciliation report to specified format.

        Args:
            report: Report to export
            export_format: Output format (JSON or CSV)

        Returns:
            Exported report as string

        Raises:
            ValueError: If export format not supported
        """
        if export_format == ExportFormat.JSON:
            return self._export_json(report)
        elif export_format == ExportFormat.CSV:
            return self._export_csv(report)
        else:
            raise ValueError(f"Unsupported export format: {export_format.value}")

    def _export_json(self, report: ReconciliationReport) -> str:
        """
        Export report as JSON.

        Args:
            report: Report to export

        Returns:
            JSON string
        """
        report_dict = report.dict()
        return json.dumps(report_dict, indent=2, default=str)

    def _export_csv(self, report: ReconciliationReport) -> str:
        """
        Export report as CSV (summary + discrepancies).

        Args:
            report: Report to export

        Returns:
            CSV string
        """
        output = StringIO()
        writer = csv.writer(output)

        # Summary section
        writer.writerow(["RECONCILIATION SUMMARY"])
        writer.writerow(["Reconciliation ID", report.reconciliation_id])
        writer.writerow(["Tenant ID", report.tenant_id])
        writer.writerow(["Status", report.status.value])
        writer.writerow(["Period Start", str(report.period_start)])
        writer.writerow(["Period End", str(report.period_end)])
        writer.writerow([])

        # Totals section
        writer.writerow(["EMISSION TOTALS"])
        writer.writerow(["Method", "Total tCO2e"])
        writer.writerow(["Location-Based", str(report.workspace.total_location_tco2e)])
        writer.writerow(["Market-Based", str(report.workspace.total_market_tco2e)])
        writer.writerow(["PIF", str(report.workspace.pif) if report.workspace.pif else "N/A"])
        writer.writerow([])

        # Quality section
        writer.writerow(["QUALITY ASSESSMENT"])
        writer.writerow(["Metric", "Score"])
        writer.writerow(["Overall", str(report.quality_assessment.overall_score)])
        writer.writerow(["Location-Based", str(report.quality_assessment.location_based_score)])
        writer.writerow(["Market-Based", str(report.quality_assessment.market_based_score)])
        writer.writerow([])

        # Discrepancies section
        writer.writerow(["DISCREPANCIES"])
        writer.writerow([
            "Energy Type",
            "Type",
            "Location tCO2e",
            "Market tCO2e",
            "Delta tCO2e",
            "Delta %",
            "Explanation",
        ])
        for item in report.discrepancy_report.items:
            writer.writerow([
                item.energy_type.value if item.energy_type else "TOTAL",
                item.discrepancy_type.value,
                str(item.location_tco2e),
                str(item.market_tco2e),
                str(item.delta_tco2e),
                str(item.delta_pct),
                item.explanation,
            ])

        return output.getvalue()

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline execution status.

        Returns:
            Pipeline status information
        """
        return {
            "pipeline_id": self._current_pipeline_id,
            "current_stage": self._current_stage.value if self._current_stage else None,
            "start_time": self._pipeline_start_time.isoformat() if self._pipeline_start_time else None,
            "elapsed_ms": (
                (datetime.utcnow() - self._pipeline_start_time).total_seconds() * 1000
                if self._pipeline_start_time
                else 0.0
            ),
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on pipeline engine.

        Returns:
            Health status with component checks
        """
        health = {
            "status": "healthy",
            "agent_id": AGENT_ID,
            "component": AGENT_COMPONENT,
            "version": VERSION,
            "initialized": self._initialized,
            "current_pipeline": self._current_pipeline_id,
            "engines": {},
        }

        # Check lazy-loaded engines
        engine_checks = {
            "collector": self._collector,
            "discrepancy_analyzer": self._discrepancy_analyzer,
            "quality_scorer": self._quality_scorer,
            "table_generator": self._table_generator,
            "trend_analyzer": self._trend_analyzer,
            "compliance_checker": self._compliance_checker,
        }

        for engine_name, engine in engine_checks.items():
            if engine is not None:
                try:
                    engine_health = engine.health_check()
                    health["engines"][engine_name] = engine_health.get("status", "unknown")
                except Exception as e:
                    health["engines"][engine_name] = f"error: {str(e)}"
                    health["status"] = "degraded"
            else:
                health["engines"][engine_name] = "not_loaded"

        return health

    def reset(self) -> None:
        """
        Reset pipeline state (for testing).

        WARNING: This clears all lazy-loaded engines and pipeline state.
        """
        with self._lock:
            self._collector = None
            self._discrepancy_analyzer = None
            self._quality_scorer = None
            self._table_generator = None
            self._trend_analyzer = None
            self._compliance_checker = None

            self._current_pipeline_id = None
            self._current_stage = None
            self._pipeline_start_time = None

            logger.warning("Pipeline engine reset - all engines unloaded")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DualReportingPipelineEngine(version={VERSION}, "
            f"initialized={self._initialized}, "
            f"current_pipeline={self._current_pipeline_id})"
        )
