# -*- coding: utf-8 -*-
"""
Reporting Workflow
===================================

3-phase workflow for generating demand response performance reports
within PACK-037 Demand Response Pack.

Phases:
    1. DataAggregation       -- Compile results from events, settlements, programs
    2. ReportGeneration      -- Generate specified report types via engines
    3. DistributionDelivery  -- Format and package reports for distribution

The workflow follows GreenLang zero-hallucination principles: all
aggregated metrics use deterministic arithmetic on verified input data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - FERC reporting requirements for demand response
    - ISO/RTO performance reporting standards
    - NAESB WEQ data exchange formats

Schedule: on-demand / monthly / quarterly
Estimated duration: 10 minutes

Author: GreenLang Team
Version: 37.0.0
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

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

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

class DRReportType(str, Enum):
    """Available demand response report types."""

    EXECUTIVE_SUMMARY = "executive_summary"
    EVENT_PERFORMANCE = "event_performance"
    REVENUE_SUMMARY = "revenue_summary"
    PROGRAM_COMPLIANCE = "program_compliance"
    DER_UTILIZATION = "der_utilization"
    CARBON_IMPACT = "carbon_impact"
    FLEXIBILITY_ASSESSMENT = "flexibility_assessment"
    SETTLEMENT_DETAIL = "settlement_detail"

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

class DRReportingInput(BaseModel):
    """Input data model for DRReportingWorkflow."""

    facility_ids: List[str] = Field(
        default_factory=list,
        description="Facility IDs to include in reports",
    )
    event_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Event execution result data",
    )
    settlement_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Settlement result data",
    )
    program_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Program enrollment data",
    )
    der_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="DER dispatch and utilization data",
    )
    report_types: List[str] = Field(
        default_factory=lambda: ["executive_summary"],
        description="Report types to generate",
    )
    format: str = Field(default="json", description="Output format")
    period: str = Field(default="", description="Reporting period (YYYY-QN or YYYY-MM)")
    grid_emission_factor: Decimal = Field(
        default=Decimal("0.400"), ge=0,
        description="Grid emission factor kgCO2/kWh",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class DRReportingResult(BaseModel):
    """Complete result from DR reporting workflow."""

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

class DRReportingWorkflow:
    """
    3-phase reporting workflow for demand response programs.

    Performs data aggregation across events, settlements, and programs,
    generates specified report types, and formats for distribution.

    Zero-hallucination: all aggregated metrics are deterministic sums,
    averages, and percentages of verified input data. No LLM calls
    in the numeric computation path.

    Attributes:
        report_id: Unique report batch identifier.
        _aggregated_data: Compiled data from all sources.
        _generated_reports: List of generated report objects.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = DRReportingWorkflow()
        >>> inp = DRReportingInput(
        ...     event_results=[{"event_id": "evt-1", "performance_pct": 95}],
        ...     report_types=["executive_summary", "revenue_summary"],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_reports >= 1
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DRReportingWorkflow."""
        self.report_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._aggregated_data: Dict[str, Any] = {}
        self._generated_reports: List[GeneratedReport] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: DRReportingInput) -> DRReportingResult:
        """
        Execute the 3-phase DR reporting workflow.

        Args:
            input_data: Validated reporting input.

        Returns:
            DRReportingResult with generated reports and executive summary.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        self.logger.info(
            "Starting DR reporting workflow %s types=%s period=%s",
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

            # Phase 3: Distribution Delivery
            phase3 = self._phase_distribution_delivery(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error("DR reporting workflow failed: %s", exc, exc_info=True)
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

        result = DRReportingResult(
            report_id=self.report_id,
            reports_generated=self._generated_reports,
            executive_summary=executive_summary,
            total_reports=len(self._generated_reports),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "DR reporting workflow %s completed in %.0fms reports=%d",
            self.report_id, elapsed_ms, len(self._generated_reports),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Aggregation
    # -------------------------------------------------------------------------

    def _phase_data_aggregation(
        self, input_data: DRReportingInput
    ) -> PhaseResult:
        """Compile results from events, settlements, and programs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Aggregate event performance
        total_events = len(input_data.event_results)
        total_curtailed_kw = Decimal("0")
        total_performance_pct = Decimal("0")
        events_met_target = 0

        for event in input_data.event_results:
            total_curtailed_kw += Decimal(str(event.get("average_curtailed_kw", 0)))
            perf = Decimal(str(event.get("performance_pct", 0)))
            total_performance_pct += perf
            if perf >= Decimal("90"):
                events_met_target += 1

        avg_performance = (
            total_performance_pct / Decimal(str(max(total_events, 1)))
        ).quantize(Decimal("0.1"))

        # Aggregate settlement revenue
        total_gross_revenue = Decimal("0")
        total_penalties = Decimal("0")
        total_net_revenue = Decimal("0")

        for settlement in input_data.settlement_results:
            total_gross_revenue += Decimal(str(settlement.get("gross_revenue", 0)))
            total_penalties += Decimal(str(settlement.get("penalty_amount", 0)))
            total_net_revenue += Decimal(str(settlement.get("net_settlement", 0)))

        # Aggregate program data
        programs_enrolled = len(input_data.program_data)
        total_committed_kw = Decimal("0")
        for prog in input_data.program_data:
            total_committed_kw += Decimal(str(prog.get("committed_kw", 0)))

        # Calculate carbon avoided
        avoided_kwh = total_curtailed_kw * Decimal("4")  # assume 4-hour avg events
        carbon_avoided = (
            avoided_kwh * input_data.grid_emission_factor / Decimal("1000")
        ).quantize(Decimal("0.01"))

        self._aggregated_data = {
            "period": input_data.period,
            "total_events": total_events,
            "events_met_target": events_met_target,
            "avg_performance_pct": str(avg_performance),
            "total_curtailed_kw": str(total_curtailed_kw),
            "total_gross_revenue": str(total_gross_revenue),
            "total_penalties": str(total_penalties),
            "total_net_revenue": str(total_net_revenue),
            "programs_enrolled": programs_enrolled,
            "total_committed_kw": str(total_committed_kw),
            "carbon_avoided_tonnes": str(carbon_avoided),
            "facilities_count": len(input_data.facility_ids),
        }

        if not input_data.event_results:
            warnings.append("No event results provided for aggregation")

        outputs.update(self._aggregated_data)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 DataAggregation: %d events, revenue=%.0f, perf=%.1f%%",
            total_events, float(total_net_revenue), float(avg_performance),
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
        self, input_data: DRReportingInput
    ) -> PhaseResult:
        """Generate specified report types."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        now_iso = utcnow().isoformat() + "Z"

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
                "title": "Demand Response Program - Executive Summary",
                "period": agg.get("period", ""),
                "total_events": agg.get("total_events", 0),
                "events_met_target": agg.get("events_met_target", 0),
                "avg_performance_pct": agg.get("avg_performance_pct", "0"),
                "total_net_revenue": agg.get("total_net_revenue", "0"),
                "programs_enrolled": agg.get("programs_enrolled", 0),
                "carbon_avoided_tonnes": agg.get("carbon_avoided_tonnes", "0"),
            }

        if report_type == "revenue_summary":
            return {
                "title": "Demand Response - Revenue Summary",
                "period": agg.get("period", ""),
                "total_gross_revenue": agg.get("total_gross_revenue", "0"),
                "total_penalties": agg.get("total_penalties", "0"),
                "total_net_revenue": agg.get("total_net_revenue", "0"),
                "programs_enrolled": agg.get("programs_enrolled", 0),
                "total_committed_kw": agg.get("total_committed_kw", "0"),
            }

        if report_type == "event_performance":
            return {
                "title": "Demand Response - Event Performance Report",
                "period": agg.get("period", ""),
                "total_events": agg.get("total_events", 0),
                "events_met_target": agg.get("events_met_target", 0),
                "avg_performance_pct": agg.get("avg_performance_pct", "0"),
                "total_curtailed_kw": agg.get("total_curtailed_kw", "0"),
            }

        if report_type == "carbon_impact":
            return {
                "title": "Demand Response - Carbon Impact Report",
                "period": agg.get("period", ""),
                "carbon_avoided_tonnes": agg.get("carbon_avoided_tonnes", "0"),
                "total_curtailed_kw": agg.get("total_curtailed_kw", "0"),
            }

        if report_type == "program_compliance":
            return {
                "title": "Demand Response - Program Compliance Report",
                "period": agg.get("period", ""),
                "programs_enrolled": agg.get("programs_enrolled", 0),
                "total_committed_kw": agg.get("total_committed_kw", "0"),
                "total_events": agg.get("total_events", 0),
                "events_met_target": agg.get("events_met_target", 0),
                "compliance_rate_pct": str(
                    round(
                        int(agg.get("events_met_target", 0))
                        / max(int(agg.get("total_events", 0)), 1) * 100, 1
                    )
                ),
            }

        # Default: generic report
        return {
            "title": f"Demand Response - {report_type.replace('_', ' ').title()} Report",
            "period": agg.get("period", ""),
            "data": agg,
        }

    def _estimate_page_count(self, content: Dict[str, Any]) -> int:
        """Estimate page count based on content size."""
        content_str = json.dumps(content, default=str)
        return max(1, len(content_str) // 3000 + 1)

    # -------------------------------------------------------------------------
    # Phase 3: Distribution Delivery
    # -------------------------------------------------------------------------

    def _phase_distribution_delivery(
        self, input_data: DRReportingInput
    ) -> PhaseResult:
        """Format and package reports for distribution."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

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
            "Phase 3 DistributionDelivery: %d reports packaged in %s format",
            len(packages), input_data.format,
        )
        return PhaseResult(
            phase_name="distribution_delivery", phase_number=3,
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
        events = agg.get("total_events", 0)
        met_target = agg.get("events_met_target", 0)
        perf = agg.get("avg_performance_pct", "0")
        revenue = agg.get("total_net_revenue", "0")
        co2 = agg.get("carbon_avoided_tonnes", "0")
        programs = agg.get("programs_enrolled", 0)
        period = agg.get("period", "current period")

        return (
            f"Demand Response Program Report - {period}. "
            f"{events} DR events dispatched, {met_target} met performance target. "
            f"Average performance: {perf}%. "
            f"Net revenue: ${revenue}. "
            f"Carbon avoided: {co2} tonnes CO2e. "
            f"{programs} programs enrolled."
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: DRReportingResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
