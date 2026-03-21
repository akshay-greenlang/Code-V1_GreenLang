# -*- coding: utf-8 -*-
"""
Reporting Workflow
===================================

3-phase workflow for generating quick-win energy efficiency reports
within PACK-033 Quick Wins Identifier Pack.

Phases:
    1. DataAggregation     -- Compile results from scans, priorities, progress
    2. ReportGeneration    -- Generate specified report types via engines
    3. Distribution        -- Format and package reports for delivery

The workflow follows GreenLang zero-hallucination principles: all
aggregated metrics use deterministic arithmetic on verified input data.
SHA-256 provenance hashes guarantee auditability.

Schedule: on-demand / periodic
Estimated duration: 10 minutes

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

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


class ReportType(str, Enum):
    """Available report types."""

    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_SCAN = "detailed_scan"
    FINANCIAL_ANALYSIS = "financial_analysis"
    CARBON_IMPACT = "carbon_impact"
    IMPLEMENTATION_PLAN = "implementation_plan"
    PROGRESS_REPORT = "progress_report"
    BENCHMARK_COMPARISON = "benchmark_comparison"


class ReportFormat(str, Enum):
    """Output format for reports."""

    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class GeneratedReport(BaseModel):
    """A single generated report."""

    report_type: str = Field(default="", description="Type of report")
    content: Dict[str, Any] = Field(default_factory=dict, description="Report content")
    format: str = Field(default="json", description="Output format")
    generated_at: str = Field(default="", description="ISO 8601 timestamp")
    page_count: int = Field(default=0, ge=0, description="Estimated page count")
    provenance_hash: str = Field(default="", description="SHA-256 of report content")


class ReportingInput(BaseModel):
    """Input data model for ReportingWorkflow."""

    scan_ids: List[str] = Field(
        default_factory=list,
        description="Scan IDs to include in reports",
    )
    scan_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Scan result data for aggregation",
    )
    prioritization_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prioritization result data",
    )
    progress_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Progress tracking result data",
    )
    report_types: List[str] = Field(
        default_factory=lambda: ["executive_summary"],
        description="Report types to generate",
    )
    format: str = Field(default="json", description="Output format")
    period: str = Field(default="", description="Reporting period (YYYY-QN or YYYY-MM)")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ReportingResult(BaseModel):
    """Complete result from reporting workflow."""

    report_id: str = Field(..., description="Unique report batch ID")
    reports_generated: List[GeneratedReport] = Field(default_factory=list)
    executive_summary: str = Field(default="", description="Text executive summary")
    total_reports: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ReportingWorkflow:
    """
    3-phase reporting workflow for quick-win energy efficiency programs.

    Performs data aggregation across scans/priorities/progress, generates
    specified report types, and formats/packages for distribution.

    Zero-hallucination: all aggregated metrics are deterministic sums,
    averages, and percentages of verified input data. No LLM calls
    in the numeric computation path.

    Attributes:
        report_id: Unique report batch identifier.
        _aggregated_data: Compiled data from all sources.
        _generated_reports: List of generated report objects.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ReportingWorkflow()
        >>> inp = ReportingInput(
        ...     scan_results=[{...}],
        ...     report_types=["executive_summary", "financial_analysis"],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_reports >= 1
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ReportingWorkflow."""
        self.report_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._aggregated_data: Dict[str, Any] = {}
        self._generated_reports: List[GeneratedReport] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: ReportingInput) -> ReportingResult:
        """
        Execute the 3-phase reporting workflow.

        Args:
            input_data: Validated reporting input.

        Returns:
            ReportingResult with generated reports and executive summary.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting reporting workflow %s types=%s period=%s",
            self.report_id, input_data.report_types, input_data.period,
        )

        self._phase_results = []
        self._aggregated_data = {}
        self._generated_reports = []

        try:
            # Phase 1: Data Aggregation
            phase1 = self._phase_data_aggregation(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Report Generation
            phase2 = self._phase_report_generation(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Distribution
            phase3 = self._phase_distribution(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error("Reporting workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        executive_summary = self._build_executive_summary()
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = ReportingResult(
            report_id=self.report_id,
            reports_generated=self._generated_reports,
            executive_summary=executive_summary,
            total_reports=len(self._generated_reports),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Reporting workflow %s completed in %.0fms reports=%d",
            self.report_id, elapsed_ms, len(self._generated_reports),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Aggregation
    # -------------------------------------------------------------------------

    def _phase_data_aggregation(
        self, input_data: ReportingInput
    ) -> PhaseResult:
        """Compile all results from scans, priorities, and progress."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Aggregate scan results
        total_wins = 0
        total_scan_savings_kwh = Decimal("0")
        total_scan_savings_cost = Decimal("0")
        facilities_scanned = 0

        for scan in input_data.scan_results:
            total_wins += scan.get("total_quick_wins_found", 0)
            total_scan_savings_kwh += Decimal(str(scan.get("estimated_total_savings_kwh", 0)))
            total_scan_savings_cost += Decimal(str(scan.get("estimated_total_savings_cost", 0)))
            facilities_scanned += 1

        # Aggregate prioritization results
        total_npv = Decimal("0")
        total_investment = Decimal("0")
        total_co2 = Decimal("0")
        for prio in input_data.prioritization_results:
            total_npv += Decimal(str(prio.get("total_npv", 0)))
            total_investment += Decimal(str(prio.get("total_investment", 0)))
            total_co2 += Decimal(str(prio.get("total_co2e_reduction", 0)))

        # Aggregate progress results
        measures_completed = 0
        verified_savings_kwh = Decimal("0")
        for progress in input_data.progress_results:
            measures_completed += progress.get("measures_completed", 0)
            verified_savings_kwh += Decimal(str(progress.get("total_verified_savings_kwh", 0)))

        self._aggregated_data = {
            "facilities_scanned": facilities_scanned,
            "total_quick_wins": total_wins,
            "total_scan_savings_kwh": str(total_scan_savings_kwh),
            "total_scan_savings_cost": str(total_scan_savings_cost),
            "total_npv": str(total_npv),
            "total_investment": str(total_investment),
            "total_co2e_reduction": str(total_co2),
            "measures_completed": measures_completed,
            "verified_savings_kwh": str(verified_savings_kwh),
            "period": input_data.period,
        }

        if not input_data.scan_results:
            warnings.append("No scan results provided for aggregation")

        outputs.update(self._aggregated_data)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 DataAggregation: %d facilities, %d wins, savings=%.0f kWh",
            facilities_scanned, total_wins, float(total_scan_savings_kwh),
        )
        return PhaseResult(
            phase_name="data_aggregation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Report Generation
    # -------------------------------------------------------------------------

    def _phase_report_generation(
        self, input_data: ReportingInput
    ) -> PhaseResult:
        """Generate specified report types."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        now_iso = datetime.utcnow().isoformat() + "Z"

        for report_type_str in input_data.report_types:
            content = self._generate_report_content(report_type_str)

            report = GeneratedReport(
                report_type=report_type_str,
                content=content,
                format=input_data.format,
                generated_at=now_iso,
                page_count=self._estimate_page_count(content),
                provenance_hash=self._hash_dict(content),
            )
            self._generated_reports.append(report)

        outputs["reports_generated"] = len(self._generated_reports)
        outputs["report_types"] = input_data.report_types
        outputs["output_format"] = input_data.format

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 ReportGeneration: %d reports generated",
            len(self._generated_reports),
        )
        return PhaseResult(
            phase_name="report_generation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_report_content(self, report_type: str) -> Dict[str, Any]:
        """Generate content for a specific report type."""
        agg = self._aggregated_data

        if report_type == "executive_summary":
            return {
                "title": "Quick Wins Energy Efficiency - Executive Summary",
                "period": agg.get("period", ""),
                "facilities_scanned": agg.get("facilities_scanned", 0),
                "total_quick_wins": agg.get("total_quick_wins", 0),
                "total_savings_kwh": agg.get("total_scan_savings_kwh", "0"),
                "total_savings_cost": agg.get("total_scan_savings_cost", "0"),
                "total_npv": agg.get("total_npv", "0"),
                "total_co2e_reduction": agg.get("total_co2e_reduction", "0"),
                "measures_completed": agg.get("measures_completed", 0),
                "verified_savings_kwh": agg.get("verified_savings_kwh", "0"),
            }

        if report_type == "financial_analysis":
            return {
                "title": "Quick Wins - Financial Analysis Report",
                "period": agg.get("period", ""),
                "total_investment": agg.get("total_investment", "0"),
                "total_npv": agg.get("total_npv", "0"),
                "total_savings_cost": agg.get("total_scan_savings_cost", "0"),
                "roi_pct": self._calculate_roi(agg),
            }

        if report_type == "carbon_impact":
            return {
                "title": "Quick Wins - Carbon Impact Report",
                "period": agg.get("period", ""),
                "total_co2e_reduction_tonnes": agg.get("total_co2e_reduction", "0"),
                "total_savings_kwh": agg.get("total_scan_savings_kwh", "0"),
            }

        if report_type == "progress_report":
            return {
                "title": "Quick Wins - Progress Report",
                "period": agg.get("period", ""),
                "measures_completed": agg.get("measures_completed", 0),
                "verified_savings_kwh": agg.get("verified_savings_kwh", "0"),
            }

        # Default: detailed scan
        return {
            "title": f"Quick Wins - {report_type.replace('_', ' ').title()} Report",
            "period": agg.get("period", ""),
            "data": agg,
        }

    def _calculate_roi(self, agg: Dict[str, Any]) -> str:
        """Calculate ROI percentage from aggregated data."""
        investment = float(agg.get("total_investment", 0))
        savings = float(agg.get("total_scan_savings_cost", 0))
        if investment > 0:
            return str(round((savings - investment) / investment * 100.0, 2))
        return "0"

    def _estimate_page_count(self, content: Dict[str, Any]) -> int:
        """Estimate page count based on content size."""
        content_str = json.dumps(content, default=str)
        # Rough estimate: ~3000 chars per page
        return max(1, len(content_str) // 3000 + 1)

    # -------------------------------------------------------------------------
    # Phase 3: Distribution
    # -------------------------------------------------------------------------

    def _phase_distribution(
        self, input_data: ReportingInput
    ) -> PhaseResult:
        """Format and package reports for delivery."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Package metadata for each report
        packages: List[Dict[str, str]] = []
        for report in self._generated_reports:
            package = {
                "report_type": report.report_type,
                "format": report.format,
                "generated_at": report.generated_at,
                "page_count": str(report.page_count),
                "provenance_hash": report.provenance_hash,
            }
            packages.append(package)

        outputs["packages"] = packages
        outputs["total_packages"] = len(packages)
        outputs["distribution_format"] = input_data.format

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 Distribution: %d reports packaged in %s format",
            len(packages), input_data.format,
        )
        return PhaseResult(
            phase_name="distribution", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _build_executive_summary(self) -> str:
        """Build a text executive summary from aggregated data."""
        agg = self._aggregated_data
        facilities = agg.get("facilities_scanned", 0)
        wins = agg.get("total_quick_wins", 0)
        savings_kwh = agg.get("total_scan_savings_kwh", "0")
        savings_cost = agg.get("total_scan_savings_cost", "0")
        co2 = agg.get("total_co2e_reduction", "0")
        completed = agg.get("measures_completed", 0)
        period = agg.get("period", "current period")

        return (
            f"Quick Wins Energy Efficiency Report - {period}. "
            f"{facilities} facilities scanned, {wins} quick wins identified. "
            f"Estimated annual savings: {savings_kwh} kWh ({savings_cost} cost). "
            f"CO2e reduction potential: {co2} tonnes. "
            f"{completed} measures implemented to date."
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ReportingResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
