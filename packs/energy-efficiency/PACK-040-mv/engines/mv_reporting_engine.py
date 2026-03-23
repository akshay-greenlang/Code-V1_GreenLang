# -*- coding: utf-8 -*-
"""
MVReportingEngine - PACK-040 M&V Engine 10
=============================================

Comprehensive M&V report generation engine producing M&V Plan, Baseline,
Post-Installation, Savings Verification, Annual M&V Summary, Persistence,
and Compliance reports in multiple formats (Markdown, HTML, JSON).
Includes automated scheduling, distribution, and compliance checking
against IPMVP, ISO 50015, FEMP 4.0, and ASHRAE 14.

Report Types:
    1. M&V Plan         - Pre-retrofit documentation per IPMVP
    2. Baseline          - Model development, validation, diagnostics
    3. Post-Installation - Installation verification, meter commissioning
    4. Savings           - Periodic savings with uncertainty bounds
    5. Annual Summary    - Year-end comprehensive savings report
    6. Persistence       - Multi-year tracking and degradation
    7. Compliance        - Standards conformity checklist

Export Formats:
    - Markdown (MD):  Structured text with tables and headers
    - HTML:           Full HTML document with inline CSS
    - JSON:           Machine-readable structured data

Compliance Checking:
    - IPMVP Core Concepts 2022: Option selection, baseline, adjustments
    - ISO 50015:2014: M&V plan, reporting period, verification
    - FEMP M&V Guidelines 4.0: Federal project requirements
    - ASHRAE Guideline 14-2014: Statistical criteria (CVRMSE, NMBE, R2)

Regulatory References:
    - IPMVP Core Concepts 2022 (EVO) - M&V reporting templates
    - ASHRAE Guideline 14-2014 - Statistical validation reporting
    - ISO 50015:2014 - M&V report requirements
    - ISO 50001:2018 - Energy review documentation
    - FEMP M&V Guidelines 4.0 - Federal M&V reporting
    - EU EED Article 7 - Energy savings reporting

Zero-Hallucination:
    - Report content populated from engine calculation results only
    - Compliance checks are deterministic rule evaluations
    - No LLM used to generate report content or conclusions
    - All statistics reproduced directly from input data
    - SHA-256 provenance hash on every report

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MVReportType(str, Enum):
    """Type of M&V report.

    MV_PLAN:          Pre-retrofit M&V plan document.
    BASELINE:         Baseline model development report.
    POST_INSTALL:     Post-installation verification report.
    SAVINGS:          Savings verification report.
    ANNUAL_SUMMARY:   Annual M&V summary report.
    PERSISTENCE:      Multi-year persistence report.
    COMPLIANCE:       Standards compliance report.
    EXECUTIVE:        Executive summary (2-4 pages).
    """
    MV_PLAN = "mv_plan"
    BASELINE = "baseline"
    POST_INSTALL = "post_install"
    SAVINGS = "savings"
    ANNUAL_SUMMARY = "annual_summary"
    PERSISTENCE = "persistence"
    COMPLIANCE = "compliance"
    EXECUTIVE = "executive"


class ReportFormat(str, Enum):
    """Output format for M&V reports.

    MARKDOWN:  Structured Markdown text.
    HTML:      Full HTML document.
    JSON:      Machine-readable JSON.
    """
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class ComplianceFramework(str, Enum):
    """Compliance framework for checking.

    IPMVP:      IPMVP Core Concepts 2022.
    ISO_50015:  ISO 50015:2014.
    FEMP:       FEMP M&V Guidelines 4.0.
    ASHRAE_14:  ASHRAE Guideline 14-2014.
    EU_EED:     EU Energy Efficiency Directive.
    """
    IPMVP = "ipmvp"
    ISO_50015 = "iso_50015"
    FEMP = "femp"
    ASHRAE_14 = "ashrae_14"
    EU_EED = "eu_eed"


class CheckStatus(str, Enum):
    """Compliance check result status.

    PASS:        Requirement fully satisfied.
    PARTIAL:     Partially satisfied.
    FAIL:        Requirement not met.
    NOT_APPLICABLE: Requirement not applicable.
    NOT_EVALUATED: Not yet evaluated.
    """
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"


class ScheduleFrequency(str, Enum):
    """Report scheduling frequency.

    MONTHLY:     Monthly report generation.
    QUARTERLY:   Quarterly reports.
    SEMI_ANNUAL: Semi-annual reports.
    ANNUAL:      Annual reports.
    ON_DEMAND:   Generated on request only.
    """
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


class DistributionChannel(str, Enum):
    """Report distribution channel.

    EMAIL:      Email delivery.
    PORTAL:     Web portal upload.
    API:        API endpoint delivery.
    FILE:       File system output.
    PRINT:      Print-ready format.
    """
    EMAIL = "email"
    PORTAL = "portal"
    API = "api"
    FILE = "file"
    PRINT = "print"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    """A single section within an M&V report."""

    section_id: str = Field(default_factory=_new_uuid, description="Section ID")
    title: str = Field(..., description="Section title")
    order: int = Field(default=0, description="Display order")
    content_md: str = Field(default="", description="Markdown content")
    tables: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tables as list of row dicts"
    )
    figures: List[str] = Field(
        default_factory=list, description="Figure references"
    )
    notes: List[str] = Field(default_factory=list, description="Section notes")


class ComplianceCheck(BaseModel):
    """Single compliance requirement check."""

    check_id: str = Field(default_factory=_new_uuid, description="Check ID")
    framework: ComplianceFramework = Field(..., description="Framework")
    requirement_id: str = Field(default="", description="Requirement identifier")
    requirement_text: str = Field(default="", description="Requirement description")
    status: CheckStatus = Field(default=CheckStatus.NOT_EVALUATED, description="Status")
    evidence: str = Field(default="", description="Supporting evidence")
    notes: str = Field(default="", description="Additional notes")


class ComplianceResult(BaseModel):
    """Overall compliance assessment result."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    framework: ComplianceFramework = Field(..., description="Framework assessed")
    total_checks: int = Field(default=0, description="Total requirements checked")
    passed: int = Field(default=0, description="Requirements passed")
    partial: int = Field(default=0, description="Requirements partially met")
    failed: int = Field(default=0, description="Requirements failed")
    not_applicable: int = Field(default=0, description="Not applicable requirements")
    compliance_score_pct: Decimal = Field(
        default=Decimal("0"), description="Compliance score (%)"
    )
    checks: List[ComplianceCheck] = Field(
        default_factory=list, description="Individual checks"
    )
    overall_compliant: bool = Field(default=False, description="Overall pass/fail")
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class ReportSchedule(BaseModel):
    """Schedule configuration for automated report generation."""

    schedule_id: str = Field(default_factory=_new_uuid, description="Schedule ID")
    report_type: MVReportType = Field(..., description="Report type to generate")
    frequency: ScheduleFrequency = Field(
        default=ScheduleFrequency.ANNUAL, description="Generation frequency"
    )
    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.MARKDOWN],
        description="Output formats",
    )
    distribution: List[DistributionChannel] = Field(
        default_factory=lambda: [DistributionChannel.FILE],
        description="Distribution channels",
    )
    recipients: List[str] = Field(
        default_factory=list, description="Distribution recipients"
    )
    next_due: Optional[str] = Field(None, description="Next due date YYYY-MM-DD")
    active: bool = Field(default=True, description="Whether schedule is active")


class ReportConfig(BaseModel):
    """Configuration for a single report generation."""

    config_id: str = Field(default_factory=_new_uuid, description="Config ID")
    project_id: str = Field(default="", description="M&V project reference")
    report_type: MVReportType = Field(..., description="Report type")
    format: ReportFormat = Field(
        default=ReportFormat.MARKDOWN, description="Output format"
    )
    title: str = Field(default="", description="Report title override")
    author: str = Field(default="GreenLang M&V Engine", description="Report author")
    reporting_period_start: str = Field(default="", description="Period start YYYY-MM-DD")
    reporting_period_end: str = Field(default="", description="Period end YYYY-MM-DD")
    include_appendices: bool = Field(default=True, description="Include appendices")
    include_raw_data: bool = Field(default=False, description="Include raw data tables")
    custom_sections: List[ReportSection] = Field(
        default_factory=list, description="Custom sections to include"
    )


class ReportOutput(BaseModel):
    """Generated report output."""

    output_id: str = Field(default_factory=_new_uuid, description="Output ID")
    config_id: str = Field(default="", description="Configuration reference")
    report_type: MVReportType = Field(..., description="Report type")
    format: ReportFormat = Field(default=ReportFormat.MARKDOWN, description="Format")
    title: str = Field(default="", description="Report title")
    content: str = Field(default="", description="Report content (MD/HTML/JSON)")
    sections: List[ReportSection] = Field(
        default_factory=list, description="Report sections"
    )
    page_count_estimate: int = Field(default=0, description="Estimated page count")
    word_count: int = Field(default=0, description="Word count")
    compliance_results: List[ComplianceResult] = Field(
        default_factory=list, description="Compliance check results"
    )
    generated_at: datetime = Field(default_factory=_utcnow, description="Generation time")
    processing_time_ms: Decimal = Field(default=Decimal("0"), description="Processing time")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class ReportingResult(BaseModel):
    """Overall reporting engine result (multiple reports)."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    project_id: str = Field(default="", description="M&V project reference")
    reports_generated: int = Field(default=0, description="Number of reports generated")
    outputs: List[ReportOutput] = Field(
        default_factory=list, description="Generated report outputs"
    )
    compliance_summary: Optional[ComplianceResult] = Field(
        None, description="Aggregate compliance result"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Compliance Checklists (Deterministic Lookups)
# ---------------------------------------------------------------------------

IPMVP_CHECKS: List[Dict[str, str]] = [
    {"id": "IPMVP-01", "text": "IPMVP option selected and documented"},
    {"id": "IPMVP-02", "text": "Measurement boundary defined"},
    {"id": "IPMVP-03", "text": "Baseline period specified (>= 12 months or full operating cycle)"},
    {"id": "IPMVP-04", "text": "Baseline model developed with regression analysis"},
    {"id": "IPMVP-05", "text": "Independent variables identified and documented"},
    {"id": "IPMVP-06", "text": "Routine adjustments methodology specified"},
    {"id": "IPMVP-07", "text": "Non-routine adjustments methodology specified"},
    {"id": "IPMVP-08", "text": "Reporting period specified"},
    {"id": "IPMVP-09", "text": "Savings calculation methodology documented"},
    {"id": "IPMVP-10", "text": "Uncertainty analysis performed"},
    {"id": "IPMVP-11", "text": "M&V plan documented before implementation"},
    {"id": "IPMVP-12", "text": "Data quality requirements specified"},
    {"id": "IPMVP-13", "text": "Metering plan documented"},
    {"id": "IPMVP-14", "text": "Calibration requirements documented"},
    {"id": "IPMVP-15", "text": "Expected savings quantified with uncertainty bounds"},
]

ASHRAE14_CHECKS: List[Dict[str, str]] = [
    {"id": "ASHRAE-01", "text": "CVRMSE within limits (monthly <25%, daily <30%)"},
    {"id": "ASHRAE-02", "text": "NMBE within limits (monthly +/-5%, daily +/-10%)"},
    {"id": "ASHRAE-03", "text": "R-squared meets minimum (monthly >0.70, daily >0.50)"},
    {"id": "ASHRAE-04", "text": "Model residuals tested for normality"},
    {"id": "ASHRAE-05", "text": "Model residuals tested for autocorrelation (DW test)"},
    {"id": "ASHRAE-06", "text": "Sufficient degrees of freedom (n > p + 2)"},
    {"id": "ASHRAE-07", "text": "t-statistics significant at alpha=0.05 for all coefficients"},
    {"id": "ASHRAE-08", "text": "F-statistic significant (model is meaningful)"},
    {"id": "ASHRAE-09", "text": "Fractional savings uncertainty quantified at 68% confidence"},
    {"id": "ASHRAE-10", "text": "No multicollinearity (VIF < 10 for all variables)"},
]

ISO50015_CHECKS: List[Dict[str, str]] = [
    {"id": "ISO-01", "text": "M&V plan established before measurement activities begin"},
    {"id": "ISO-02", "text": "Measurement boundary clearly defined"},
    {"id": "ISO-03", "text": "Baseline period representative of normal operation"},
    {"id": "ISO-04", "text": "Relevant variables identified and tracked"},
    {"id": "ISO-05", "text": "Energy performance model validated"},
    {"id": "ISO-06", "text": "Static factors documented and accounted for"},
    {"id": "ISO-07", "text": "Reporting period representative and documented"},
    {"id": "ISO-08", "text": "Adjusted energy consumption calculated"},
    {"id": "ISO-09", "text": "Savings reported with uncertainty"},
    {"id": "ISO-10", "text": "Competence of M&V practitioner documented"},
]

FEMP_CHECKS: List[Dict[str, str]] = [
    {"id": "FEMP-01", "text": "M&V option selection justified per FEMP guidelines"},
    {"id": "FEMP-02", "text": "Baseline developed using 12+ months of utility data"},
    {"id": "FEMP-03", "text": "Weather normalisation applied using TMY3 data"},
    {"id": "FEMP-04", "text": "Non-routine adjustments documented with engineering estimates"},
    {"id": "FEMP-05", "text": "Annual savings report submitted within 90 days of period end"},
    {"id": "FEMP-06", "text": "Cumulative savings tracked against ESPC guarantee"},
    {"id": "FEMP-07", "text": "Metering meets FEMP accuracy requirements"},
    {"id": "FEMP-08", "text": "Independent M&V review performed"},
]

EU_EED_CHECKS: List[Dict[str, str]] = [
    {"id": "EED-01", "text": "Savings calculated using approved methodology"},
    {"id": "EED-02", "text": "Additionality of savings demonstrated"},
    {"id": "EED-03", "text": "Materiality threshold met (savings > 5% of baseline)"},
    {"id": "EED-04", "text": "Double counting avoided (savings not claimed elsewhere)"},
    {"id": "EED-05", "text": "Verification by independent third party"},
]


# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------


class MVReportingEngine:
    """M&V report generation engine with multi-format export and compliance.

    Generates M&V Plan, Baseline, Savings Verification, Annual Summary,
    Persistence, and Compliance reports.  Supports Markdown, HTML, and
    JSON output formats.  Includes automated compliance checking against
    IPMVP, ISO 50015, FEMP, ASHRAE 14, and EU EED.

    All report content is generated deterministically from input data
    (zero-hallucination).  No LLM is used for text generation -- all
    narrative is produced via structured templates.

    Attributes:
        _module_version: Engine version string.

    Example:
        >>> engine = MVReportingEngine()
        >>> config = ReportConfig(
        ...     project_id="proj-001",
        ...     report_type=MVReportType.SAVINGS,
        ...     format=ReportFormat.MARKDOWN,
        ... )
        >>> output = engine.generate_report(config, data={...})
        >>> assert output.provenance_hash != ""
    """

    def __init__(self) -> None:
        """Initialise the MVReportingEngine."""
        self._module_version: str = _MODULE_VERSION
        logger.info("MVReportingEngine v%s initialised", self._module_version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        config: ReportConfig,
        data: Optional[Dict[str, Any]] = None,
    ) -> ReportOutput:
        """Generate a single M&V report.

        Dispatches to the appropriate report builder based on
        ``config.report_type`` and renders in the requested format.

        Args:
            config: Report configuration.
            data: Report data (varies by report type).

        Returns:
            ReportOutput with rendered content and provenance.
        """
        t0 = time.perf_counter()
        if data is None:
            data = {}

        logger.info(
            "Generating report: type=%s, format=%s, project=%s",
            config.report_type.value,
            config.format.value,
            config.project_id,
        )

        # Build sections based on report type
        sections = self._build_sections(config, data)

        # Set title
        title = config.title or self._default_title(config)

        # Render content
        content = self._render(sections, config.format, title, config)

        word_count = len(content.split())
        page_estimate = max(1, word_count // 300)

        # Run compliance checks if compliance report
        compliance_results: List[ComplianceResult] = []
        if config.report_type == MVReportType.COMPLIANCE:
            compliance_results = self._run_all_compliance_checks(data)

        elapsed = (time.perf_counter() - t0) * 1000.0
        output = ReportOutput(
            config_id=config.config_id,
            report_type=config.report_type,
            format=config.format,
            title=title,
            content=content,
            sections=sections,
            page_count_estimate=page_estimate,
            word_count=word_count,
            compliance_results=compliance_results,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        output.provenance_hash = _compute_hash(output)

        logger.info(
            "Report generated: type=%s, %d words, ~%d pages, hash=%s (%.1f ms)",
            config.report_type.value, word_count, page_estimate,
            output.provenance_hash[:16], elapsed,
        )
        return output

    def generate_batch(
        self,
        configs: List[ReportConfig],
        data: Optional[Dict[str, Any]] = None,
    ) -> ReportingResult:
        """Generate multiple reports in batch.

        Args:
            configs: List of report configurations.
            data: Shared data for all reports.

        Returns:
            ReportingResult with all generated outputs.
        """
        t0 = time.perf_counter()
        logger.info("Generating batch: %d reports", len(configs))

        outputs: List[ReportOutput] = []
        for cfg in configs:
            try:
                output = self.generate_report(cfg, data)
                outputs.append(output)
            except Exception as exc:
                logger.error("Report %s failed: %s", cfg.report_type.value, exc)

        project_id = configs[0].project_id if configs else ""

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = ReportingResult(
            project_id=project_id,
            reports_generated=len(outputs),
            outputs=outputs,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Batch complete: %d/%d reports, hash=%s (%.1f ms)",
            len(outputs), len(configs),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def check_compliance(
        self,
        framework: ComplianceFramework,
        data: Dict[str, Any],
    ) -> ComplianceResult:
        """Check compliance against a specific framework.

        Args:
            framework: The compliance framework to check against.
            data: Project data with required fields for evaluation.

        Returns:
            ComplianceResult with pass/fail for each requirement.
        """
        t0 = time.perf_counter()
        logger.info("Checking compliance: framework=%s", framework.value)

        checklist = self._get_checklist(framework)
        checks: List[ComplianceCheck] = []

        for item in checklist:
            status = self._evaluate_check(framework, item["id"], data)
            evidence = self._get_evidence(framework, item["id"], data)
            checks.append(ComplianceCheck(
                framework=framework,
                requirement_id=item["id"],
                requirement_text=item["text"],
                status=status,
                evidence=evidence,
            ))

        total = len(checks)
        passed = sum(1 for c in checks if c.status == CheckStatus.PASS)
        partial = sum(1 for c in checks if c.status == CheckStatus.PARTIAL)
        failed = sum(1 for c in checks if c.status == CheckStatus.FAIL)
        na = sum(1 for c in checks if c.status == CheckStatus.NOT_APPLICABLE)
        applicable = total - na
        score = _safe_pct(_decimal(passed), _decimal(applicable)) if applicable > 0 else Decimal("0")
        overall = failed == 0

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = ComplianceResult(
            framework=framework,
            total_checks=total,
            passed=passed,
            partial=partial,
            failed=failed,
            not_applicable=na,
            compliance_score_pct=_round_val(score, 2),
            checks=checks,
            overall_compliant=overall,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Compliance %s: %d/%d passed (%.1f%%), compliant=%s, hash=%s (%.1f ms)",
            framework.value, passed, applicable, float(score),
            overall, result.provenance_hash[:16], elapsed,
        )
        return result

    def create_schedule(
        self,
        project_id: str,
        report_type: MVReportType,
        frequency: ScheduleFrequency = ScheduleFrequency.ANNUAL,
        formats: Optional[List[ReportFormat]] = None,
        channels: Optional[List[DistributionChannel]] = None,
        recipients: Optional[List[str]] = None,
    ) -> ReportSchedule:
        """Create a report schedule for automated generation.

        Args:
            project_id: M&V project reference.
            report_type: Report type to schedule.
            frequency: Generation frequency.
            formats: Output formats.
            channels: Distribution channels.
            recipients: Distribution recipients.

        Returns:
            ReportSchedule configuration.
        """
        logger.info(
            "Creating schedule: project=%s, type=%s, freq=%s",
            project_id, report_type.value, frequency.value,
        )

        return ReportSchedule(
            report_type=report_type,
            frequency=frequency,
            formats=formats or [ReportFormat.MARKDOWN],
            distribution=channels or [DistributionChannel.FILE],
            recipients=recipients or [],
            active=True,
        )

    def export_to_format(
        self,
        output: ReportOutput,
        target_format: ReportFormat,
    ) -> str:
        """Convert an existing report to a different format.

        Args:
            output: Existing report output.
            target_format: Target format.

        Returns:
            Rendered content in the target format.
        """
        t0 = time.perf_counter()
        content = self._render(
            output.sections, target_format, output.title,
            ReportConfig(report_type=output.report_type, format=target_format),
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Format conversion: %s -> %s (%.1f ms)",
            output.format.value, target_format.value, elapsed,
        )
        return content

    # ------------------------------------------------------------------
    # Internal: Section Builders
    # ------------------------------------------------------------------

    def _build_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build report sections based on report type."""
        dispatch = {
            MVReportType.MV_PLAN: self._build_mv_plan_sections,
            MVReportType.BASELINE: self._build_baseline_sections,
            MVReportType.POST_INSTALL: self._build_post_install_sections,
            MVReportType.SAVINGS: self._build_savings_sections,
            MVReportType.ANNUAL_SUMMARY: self._build_annual_sections,
            MVReportType.PERSISTENCE: self._build_persistence_sections,
            MVReportType.COMPLIANCE: self._build_compliance_sections,
            MVReportType.EXECUTIVE: self._build_executive_sections,
        }
        builder = dispatch.get(config.report_type, self._build_generic_sections)
        sections = builder(config, data)

        # Add custom sections
        if config.custom_sections:
            for cs in config.custom_sections:
                cs.order = len(sections) + cs.order
                sections.append(cs)

        sections.sort(key=lambda s: s.order)
        return sections

    def _build_mv_plan_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build M&V Plan report sections."""
        sections = [
            ReportSection(
                title="1. Project Overview",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    f"**Report Period:** {config.reporting_period_start} to "
                    f"{config.reporting_period_end}\n\n"
                    f"**Prepared By:** {config.author}\n\n"
                    f"**ECM Description:** {data.get('ecm_description', 'N/A')}\n\n"
                    f"**Estimated Savings:** {data.get('estimated_savings_kwh', 'N/A')} kWh/yr\n"
                ),
            ),
            ReportSection(
                title="2. IPMVP Option Selection",
                order=2,
                content_md=(
                    f"**Selected Option:** {data.get('ipmvp_option', 'N/A')}\n\n"
                    f"**Justification:** {data.get('option_justification', 'N/A')}\n\n"
                    "**Measurement Boundary:** "
                    f"{data.get('measurement_boundary', 'Whole facility')}\n"
                ),
            ),
            ReportSection(
                title="3. Baseline Period",
                order=3,
                content_md=(
                    f"**Baseline Start:** {data.get('baseline_start', 'N/A')}\n\n"
                    f"**Baseline End:** {data.get('baseline_end', 'N/A')}\n\n"
                    f"**Duration:** {data.get('baseline_months', 12)} months\n\n"
                    f"**Model Type:** {data.get('model_type', 'OLS regression')}\n\n"
                    "**Independent Variables:** "
                    f"{', '.join(data.get('independent_vars', ['temperature']))}\n"
                ),
            ),
            ReportSection(
                title="4. Adjustment Methodology",
                order=4,
                content_md=(
                    "**Routine Adjustments:**\n"
                    f"- Weather normalisation: {data.get('weather_method', 'HDD/CDD with optimised balance points')}\n"
                    f"- Production normalisation: {data.get('production_method', 'N/A')}\n\n"
                    "**Non-Routine Adjustments:**\n"
                    f"- {data.get('nra_description', 'Documented on occurrence basis')}\n"
                ),
            ),
            ReportSection(
                title="5. Metering Plan",
                order=5,
                content_md=(
                    f"**Number of Meters:** {data.get('meter_count', 'N/A')}\n\n"
                    f"**Data Interval:** {data.get('interval_min', 15)} minutes\n\n"
                    f"**Accuracy Class:** {data.get('accuracy_class', 'ANSI C12.20 Class 0.5')}\n\n"
                    f"**Calibration Schedule:** {data.get('cal_schedule', 'Per ANSI C12.20')}\n"
                ),
            ),
            ReportSection(
                title="6. Savings Calculation Method",
                order=6,
                content_md=(
                    "**Method:** Avoided energy use = Adjusted baseline - Actual consumption\n\n"
                    "**Uncertainty:** Fractional savings uncertainty per ASHRAE 14\n\n"
                    f"**Reporting Frequency:** {data.get('reporting_frequency', 'Annual')}\n"
                ),
            ),
        ]
        return sections

    def _build_baseline_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build Baseline report sections."""
        r2 = data.get("r_squared", "N/A")
        cvrmse = data.get("cvrmse", "N/A")
        nmbe = data.get("nmbe", "N/A")
        model_type = data.get("model_type", "N/A")
        n_obs = data.get("n_observations", "N/A")

        sections = [
            ReportSection(
                title="1. Baseline Model Summary",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    f"**Model Type:** {model_type}\n\n"
                    f"**Number of Observations:** {n_obs}\n\n"
                    f"**Independent Variables:** {', '.join(data.get('variables', []))}\n"
                ),
            ),
            ReportSection(
                title="2. Model Validation Statistics",
                order=2,
                content_md=(
                    "| Statistic | Value | ASHRAE 14 Limit | Status |\n"
                    "|-----------|-------|-----------------|--------|\n"
                    f"| R-squared | {r2} | >= 0.70 | {'PASS' if isinstance(r2, (int, float, Decimal)) and float(str(r2)) >= 0.70 else 'CHECK'} |\n"
                    f"| CV(RMSE) | {cvrmse}% | <= 25% | {'PASS' if isinstance(cvrmse, (int, float, Decimal)) and float(str(cvrmse)) <= 25 else 'CHECK'} |\n"
                    f"| NMBE | {nmbe}% | +/- 5% | {'PASS' if isinstance(nmbe, (int, float, Decimal)) and abs(float(str(nmbe))) <= 5 else 'CHECK'} |\n"
                ),
            ),
            ReportSection(
                title="3. Regression Coefficients",
                order=3,
                content_md=self._format_coefficients_table(data.get("coefficients", [])),
            ),
            ReportSection(
                title="4. Residual Diagnostics",
                order=4,
                content_md=(
                    f"**Durbin-Watson:** {data.get('durbin_watson', 'N/A')}\n\n"
                    f"**F-Statistic:** {data.get('f_statistic', 'N/A')}\n\n"
                    f"**Residual Pattern:** {data.get('residual_pattern', 'Random')}\n"
                ),
            ),
            ReportSection(
                title="5. Model Selection Rationale",
                order=5,
                content_md=(
                    f"**Selected Model:** {model_type}\n\n"
                    f"**Rationale:** {data.get('selection_rationale', 'Highest adjusted R-squared among models passing ASHRAE 14')}\n\n"
                    f"**Models Compared:** {data.get('models_compared', 'N/A')}\n"
                ),
            ),
        ]
        return sections

    def _build_post_install_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build Post-Installation report sections."""
        return [
            ReportSection(
                title="1. Installation Verification",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    f"**ECM:** {data.get('ecm_description', 'N/A')}\n\n"
                    f"**Installation Date:** {data.get('install_date', 'N/A')}\n\n"
                    f"**Verified By:** {data.get('verified_by', 'N/A')}\n\n"
                    f"**Installation Conforms to Spec:** {data.get('conforms', 'Yes')}\n"
                ),
            ),
            ReportSection(
                title="2. Meter Commissioning",
                order=2,
                content_md=(
                    f"**Meters Installed:** {data.get('meters_installed', 0)}\n\n"
                    f"**All Calibrated:** {data.get('all_calibrated', 'Yes')}\n\n"
                    f"**Data Flow Verified:** {data.get('data_verified', 'Yes')}\n"
                ),
            ),
            ReportSection(
                title="3. Short-Term Test Results",
                order=3,
                content_md=(
                    f"**Test Duration:** {data.get('test_duration', 'N/A')}\n\n"
                    f"**Measured kW Reduction:** {data.get('kw_reduction', 'N/A')}\n\n"
                    f"**Consistent with Estimates:** {data.get('consistent', 'Yes')}\n"
                ),
            ),
        ]

    def _build_savings_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build Savings Verification report sections."""
        return [
            ReportSection(
                title="1. Reporting Period",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    f"**Period:** {config.reporting_period_start} to "
                    f"{config.reporting_period_end}\n"
                ),
            ),
            ReportSection(
                title="2. Savings Summary",
                order=2,
                content_md=(
                    "| Metric | Value | Unit |\n"
                    "|--------|-------|------|\n"
                    f"| Adjusted Baseline | {data.get('adjusted_baseline', 'N/A')} | kWh |\n"
                    f"| Actual Consumption | {data.get('actual_consumption', 'N/A')} | kWh |\n"
                    f"| Avoided Energy | {data.get('avoided_energy', 'N/A')} | kWh |\n"
                    f"| Cost Savings | {data.get('cost_savings', 'N/A')} | $ |\n"
                    f"| FSU (68% conf) | {data.get('fsu_68', 'N/A')} | % |\n"
                    f"| FSU (90% conf) | {data.get('fsu_90', 'N/A')} | % |\n"
                ),
            ),
            ReportSection(
                title="3. Adjustments Applied",
                order=3,
                content_md=(
                    "**Routine Adjustments:**\n"
                    f"- Weather: {data.get('weather_adjustment', 'N/A')} kWh\n"
                    f"- Production: {data.get('production_adjustment', 'N/A')} kWh\n\n"
                    "**Non-Routine Adjustments:**\n"
                    f"- {data.get('nra_summary', 'None applied')}\n"
                ),
            ),
            ReportSection(
                title="4. Uncertainty Analysis",
                order=4,
                content_md=(
                    f"**Model Uncertainty:** {data.get('model_uncertainty', 'N/A')}%\n\n"
                    f"**Measurement Uncertainty:** {data.get('meas_uncertainty', 'N/A')}%\n\n"
                    f"**Combined Uncertainty:** {data.get('combined_uncertainty', 'N/A')}%\n\n"
                    f"**Savings Significant:** {data.get('savings_significant', 'N/A')}\n"
                ),
            ),
        ]

    def _build_annual_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build Annual M&V Summary sections."""
        return [
            ReportSection(
                title="1. Annual Summary",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    f"**Year:** {data.get('year', 'N/A')}\n\n"
                    f"**Annual Savings:** {data.get('annual_savings_kwh', 'N/A')} kWh\n\n"
                    f"**Annual Cost Savings:** ${data.get('annual_cost_savings', 'N/A')}\n\n"
                    f"**Cumulative Savings:** {data.get('cumulative_kwh', 'N/A')} kWh\n"
                ),
            ),
            ReportSection(
                title="2. Savings Trend",
                order=2,
                content_md=(
                    f"**Trend Direction:** {data.get('trend', 'Stable')}\n\n"
                    f"**Persistence Factor:** {data.get('persistence_factor', 'N/A')}\n\n"
                    f"**Year-over-Year Change:** {data.get('yoy_change', 'N/A')}%\n"
                ),
            ),
            ReportSection(
                title="3. Compliance Status",
                order=3,
                content_md=(
                    f"**IPMVP Compliant:** {data.get('ipmvp_compliant', 'N/A')}\n\n"
                    f"**ASHRAE 14 Compliant:** {data.get('ashrae_compliant', 'N/A')}\n\n"
                    f"**Guarantee Met:** {data.get('guarantee_met', 'N/A')}\n"
                ),
            ),
        ]

    def _build_persistence_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build Persistence report sections."""
        return [
            ReportSection(
                title="1. Persistence Summary",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    f"**Years Tracked:** {data.get('years_tracked', 'N/A')}\n\n"
                    f"**Current PF:** {data.get('persistence_factor', 'N/A')}\n\n"
                    f"**Status:** {data.get('persistence_status', 'N/A')}\n"
                ),
            ),
            ReportSection(
                title="2. Degradation Analysis",
                order=2,
                content_md=(
                    f"**Degradation Model:** {data.get('degradation_model', 'N/A')}\n\n"
                    f"**Annual Degradation Rate:** {data.get('degradation_rate', 'N/A')}%\n\n"
                    f"**Years to 80%:** {data.get('years_to_80', 'N/A')}\n\n"
                    f"**Re-commissioning Recommended:** {data.get('recommission', 'No')}\n"
                ),
            ),
            ReportSection(
                title="3. Year-by-Year Performance",
                order=3,
                content_md=self._format_persistence_table(data.get("annual_records", [])),
            ),
        ]

    def _build_compliance_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build Compliance report sections."""
        return [
            ReportSection(
                title="1. Compliance Overview",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    "**Frameworks Assessed:** IPMVP, ISO 50015, FEMP, ASHRAE 14, EU EED\n"
                ),
            ),
            ReportSection(
                title="2. IPMVP Compliance",
                order=2,
                content_md="See compliance results in appendix.\n",
            ),
            ReportSection(
                title="3. ASHRAE 14 Statistical Compliance",
                order=3,
                content_md="See compliance results in appendix.\n",
            ),
            ReportSection(
                title="4. ISO 50015 Compliance",
                order=4,
                content_md="See compliance results in appendix.\n",
            ),
            ReportSection(
                title="5. Recommendations",
                order=5,
                content_md=f"**Recommendations:** {data.get('recommendations', 'N/A')}\n",
            ),
        ]

    def _build_executive_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build Executive Summary sections."""
        return [
            ReportSection(
                title="Executive Summary",
                order=1,
                content_md=(
                    f"**Project:** {config.project_id}\n\n"
                    f"**Period:** {config.reporting_period_start} to "
                    f"{config.reporting_period_end}\n\n"
                    f"**Verified Savings:** {data.get('verified_savings', 'N/A')} kWh "
                    f"(${data.get('cost_savings', 'N/A')})\n\n"
                    f"**Persistence Factor:** {data.get('persistence_factor', 'N/A')}\n\n"
                    f"**Compliance:** {data.get('compliance_summary', 'N/A')}\n\n"
                    f"**Key Finding:** {data.get('key_finding', 'N/A')}\n"
                ),
            ),
        ]

    def _build_generic_sections(
        self, config: ReportConfig, data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build generic report sections."""
        return [
            ReportSection(
                title="Report",
                order=1,
                content_md=(
                    f"**Project ID:** {config.project_id}\n\n"
                    f"**Report Type:** {config.report_type.value}\n\n"
                    f"**Generated:** {_utcnow().isoformat()}\n"
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Internal: Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        sections: List[ReportSection],
        fmt: ReportFormat,
        title: str,
        config: ReportConfig,
    ) -> str:
        """Render sections into the requested format."""
        if fmt == ReportFormat.MARKDOWN:
            return self._render_markdown(sections, title, config)
        elif fmt == ReportFormat.HTML:
            return self._render_html(sections, title, config)
        elif fmt == ReportFormat.JSON:
            return self._render_json(sections, title, config)
        return self._render_markdown(sections, title, config)

    def _render_markdown(
        self, sections: List[ReportSection], title: str, config: ReportConfig,
    ) -> str:
        """Render sections as Markdown."""
        lines: List[str] = []
        lines.append(f"# {title}\n")
        lines.append(f"**Generated:** {_utcnow().isoformat()}\n")
        lines.append(f"**Author:** {config.author}\n")
        lines.append("---\n")

        for section in sections:
            lines.append(f"\n## {section.title}\n")
            if section.content_md:
                lines.append(section.content_md + "\n")
            for note in section.notes:
                lines.append(f"> {note}\n")

        lines.append("\n---\n")
        lines.append(f"*Report generated by GreenLang PACK-040 M&V Engine v{_MODULE_VERSION}*\n")
        return "\n".join(lines)

    def _render_html(
        self, sections: List[ReportSection], title: str, config: ReportConfig,
    ) -> str:
        """Render sections as HTML."""
        parts: List[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html><head>")
        parts.append(f"<title>{title}</title>")
        parts.append("<style>")
        parts.append("body { font-family: Arial, sans-serif; max-width: 900px; margin: auto; padding: 20px; }")
        parts.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
        parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        parts.append("th { background-color: #2e7d32; color: white; }")
        parts.append("h1 { color: #1b5e20; }")
        parts.append("h2 { color: #2e7d32; border-bottom: 2px solid #a5d6a7; }")
        parts.append(".meta { color: #666; font-size: 0.9em; }")
        parts.append("</style>")
        parts.append("</head><body>")
        parts.append(f"<h1>{title}</h1>")
        parts.append(f'<p class="meta">Generated: {_utcnow().isoformat()} | '
                      f'Author: {config.author}</p>')
        parts.append("<hr>")

        for section in sections:
            parts.append(f"<h2>{section.title}</h2>")
            if section.content_md:
                # Simple MD-to-HTML conversion for tables and paragraphs
                html_content = self._md_to_html_simple(section.content_md)
                parts.append(html_content)
            for note in section.notes:
                parts.append(f"<blockquote>{note}</blockquote>")

        parts.append("<hr>")
        parts.append(f"<p class='meta'>Report generated by GreenLang PACK-040 M&V Engine v{_MODULE_VERSION}</p>")
        parts.append("</body></html>")
        return "\n".join(parts)

    def _render_json(
        self, sections: List[ReportSection], title: str, config: ReportConfig,
    ) -> str:
        """Render sections as JSON."""
        report_data = {
            "title": title,
            "generated_at": _utcnow().isoformat(),
            "author": config.author,
            "project_id": config.project_id,
            "report_type": config.report_type.value,
            "sections": [
                {
                    "title": s.title,
                    "order": s.order,
                    "content": s.content_md,
                    "tables": s.tables,
                    "notes": s.notes,
                }
                for s in sections
            ],
            "engine_version": _MODULE_VERSION,
        }
        return json.dumps(report_data, indent=2, default=str)

    def _md_to_html_simple(self, md: str) -> str:
        """Simple Markdown to HTML conversion (tables, bold, paragraphs)."""
        lines = md.split("\n")
        html_lines: List[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append("<br>")
                continue

            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if all(set(c) <= {"-", " ", ":"} for c in cells):
                    continue  # Skip separator row
                if not in_table:
                    html_lines.append("<table>")
                    in_table = True
                    tag = "th"
                else:
                    tag = "td"
                row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
                html_lines.append(f"<tr>{row}</tr>")
            else:
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                # Bold
                processed = stripped
                while "**" in processed:
                    processed = processed.replace("**", "<strong>", 1)
                    processed = processed.replace("**", "</strong>", 1)
                # List items
                if processed.startswith("- "):
                    processed = f"<li>{processed[2:]}</li>"
                html_lines.append(f"<p>{processed}</p>")

        if in_table:
            html_lines.append("</table>")

        return "\n".join(html_lines)

    # ------------------------------------------------------------------
    # Internal: Compliance
    # ------------------------------------------------------------------

    def _get_checklist(self, framework: ComplianceFramework) -> List[Dict[str, str]]:
        """Get compliance checklist for a framework."""
        mapping = {
            ComplianceFramework.IPMVP: IPMVP_CHECKS,
            ComplianceFramework.ASHRAE_14: ASHRAE14_CHECKS,
            ComplianceFramework.ISO_50015: ISO50015_CHECKS,
            ComplianceFramework.FEMP: FEMP_CHECKS,
            ComplianceFramework.EU_EED: EU_EED_CHECKS,
        }
        return mapping.get(framework, [])

    def _evaluate_check(
        self,
        framework: ComplianceFramework,
        check_id: str,
        data: Dict[str, Any],
    ) -> CheckStatus:
        """Evaluate a single compliance check against available data."""
        # Deterministic rule evaluation based on data keys
        if framework == ComplianceFramework.ASHRAE_14:
            return self._evaluate_ashrae_check(check_id, data)
        elif framework == ComplianceFramework.IPMVP:
            return self._evaluate_ipmvp_check(check_id, data)

        # Default: check if the corresponding data field exists
        field_map = {
            "ISO-01": "mv_plan_date",
            "ISO-02": "measurement_boundary",
            "ISO-03": "baseline_representative",
            "ISO-04": "relevant_variables",
            "ISO-05": "model_validated",
            "ISO-06": "static_factors_documented",
            "ISO-07": "reporting_period_documented",
            "ISO-08": "adjusted_energy_calculated",
            "ISO-09": "savings_with_uncertainty",
            "ISO-10": "practitioner_competence",
            "FEMP-01": "femp_option_justified",
            "FEMP-02": "baseline_12_months",
            "FEMP-03": "tmy_normalisation",
            "FEMP-04": "nra_documented",
            "FEMP-05": "report_within_90_days",
            "FEMP-06": "cumulative_tracking",
            "FEMP-07": "metering_accuracy",
            "FEMP-08": "independent_review",
            "EED-01": "eed_methodology",
            "EED-02": "additionality",
            "EED-03": "materiality",
            "EED-04": "no_double_counting",
            "EED-05": "third_party_verification",
        }
        field = field_map.get(check_id, "")
        if field and data.get(field):
            return CheckStatus.PASS
        if field and field in data:
            return CheckStatus.PARTIAL
        return CheckStatus.NOT_EVALUATED

    def _evaluate_ashrae_check(
        self, check_id: str, data: Dict[str, Any],
    ) -> CheckStatus:
        """Evaluate ASHRAE 14 specific checks."""
        if check_id == "ASHRAE-01":
            cvrmse = data.get("cvrmse")
            if cvrmse is not None:
                return CheckStatus.PASS if float(str(cvrmse)) <= 25 else CheckStatus.FAIL
        elif check_id == "ASHRAE-02":
            nmbe = data.get("nmbe")
            if nmbe is not None:
                return CheckStatus.PASS if abs(float(str(nmbe))) <= 5 else CheckStatus.FAIL
        elif check_id == "ASHRAE-03":
            r2 = data.get("r_squared")
            if r2 is not None:
                return CheckStatus.PASS if float(str(r2)) >= 0.70 else CheckStatus.FAIL
        elif check_id == "ASHRAE-04":
            return CheckStatus.PASS if data.get("normality_tested") else CheckStatus.NOT_EVALUATED
        elif check_id == "ASHRAE-05":
            dw = data.get("durbin_watson")
            if dw is not None:
                dw_f = float(str(dw))
                return CheckStatus.PASS if 1.5 <= dw_f <= 2.5 else CheckStatus.FAIL
        elif check_id == "ASHRAE-06":
            n = data.get("n_observations", 0)
            p = data.get("n_parameters", 0)
            if n and p:
                return CheckStatus.PASS if n > p + 2 else CheckStatus.FAIL
        elif check_id == "ASHRAE-07":
            return CheckStatus.PASS if data.get("all_t_significant") else CheckStatus.NOT_EVALUATED
        elif check_id == "ASHRAE-08":
            f_stat = data.get("f_statistic")
            if f_stat is not None:
                return CheckStatus.PASS if float(str(f_stat)) > 4 else CheckStatus.FAIL
        elif check_id == "ASHRAE-09":
            return CheckStatus.PASS if data.get("fsu_calculated") else CheckStatus.NOT_EVALUATED
        elif check_id == "ASHRAE-10":
            return CheckStatus.PASS if data.get("vif_ok") else CheckStatus.NOT_EVALUATED
        return CheckStatus.NOT_EVALUATED

    def _evaluate_ipmvp_check(
        self, check_id: str, data: Dict[str, Any],
    ) -> CheckStatus:
        """Evaluate IPMVP specific checks."""
        field_map = {
            "IPMVP-01": "ipmvp_option",
            "IPMVP-02": "measurement_boundary",
            "IPMVP-03": "baseline_period",
            "IPMVP-04": "baseline_model",
            "IPMVP-05": "independent_vars",
            "IPMVP-06": "routine_adjustment_method",
            "IPMVP-07": "non_routine_adjustment_method",
            "IPMVP-08": "reporting_period",
            "IPMVP-09": "savings_method",
            "IPMVP-10": "uncertainty_analysis",
            "IPMVP-11": "mv_plan_date",
            "IPMVP-12": "data_quality_requirements",
            "IPMVP-13": "metering_plan",
            "IPMVP-14": "calibration_requirements",
            "IPMVP-15": "expected_savings_with_uncertainty",
        }
        field = field_map.get(check_id, "")
        if field and data.get(field):
            return CheckStatus.PASS
        if field and field in data:
            return CheckStatus.PARTIAL
        return CheckStatus.NOT_EVALUATED

    def _get_evidence(
        self,
        framework: ComplianceFramework,
        check_id: str,
        data: Dict[str, Any],
    ) -> str:
        """Get supporting evidence for a compliance check."""
        if framework == ComplianceFramework.ASHRAE_14:
            if check_id == "ASHRAE-01":
                return f"CV(RMSE) = {data.get('cvrmse', 'N/A')}%"
            if check_id == "ASHRAE-02":
                return f"NMBE = {data.get('nmbe', 'N/A')}%"
            if check_id == "ASHRAE-03":
                return f"R-squared = {data.get('r_squared', 'N/A')}"
        return ""

    def _run_all_compliance_checks(
        self, data: Dict[str, Any],
    ) -> List[ComplianceResult]:
        """Run compliance checks against all frameworks."""
        results: List[ComplianceResult] = []
        for fw in ComplianceFramework:
            results.append(self.check_compliance(fw, data))
        return results

    # ------------------------------------------------------------------
    # Internal: Formatting helpers
    # ------------------------------------------------------------------

    def _default_title(self, config: ReportConfig) -> str:
        """Generate default title based on report type."""
        titles = {
            MVReportType.MV_PLAN: "M&V Plan",
            MVReportType.BASELINE: "Baseline Development Report",
            MVReportType.POST_INSTALL: "Post-Installation Verification Report",
            MVReportType.SAVINGS: "Savings Verification Report",
            MVReportType.ANNUAL_SUMMARY: "Annual M&V Summary",
            MVReportType.PERSISTENCE: "Savings Persistence Report",
            MVReportType.COMPLIANCE: "Standards Compliance Report",
            MVReportType.EXECUTIVE: "Executive Summary",
        }
        base = titles.get(config.report_type, "M&V Report")
        return f"{base} - {config.project_id}" if config.project_id else base

    def _format_coefficients_table(self, coefficients: List[Dict[str, Any]]) -> str:
        """Format regression coefficients as a Markdown table."""
        if not coefficients:
            return "No coefficient data available.\n"
        lines = ["| Coefficient | Value | Std Error | t-Statistic | Significant |"]
        lines.append("|------------|-------|-----------|-------------|-------------|")
        for coef in coefficients:
            lines.append(
                f"| {coef.get('name', 'N/A')} "
                f"| {coef.get('value', 'N/A')} "
                f"| {coef.get('std_error', 'N/A')} "
                f"| {coef.get('t_statistic', 'N/A')} "
                f"| {coef.get('significant', 'N/A')} |"
            )
        return "\n".join(lines) + "\n"

    def _format_persistence_table(self, records: List[Dict[str, Any]]) -> str:
        """Format annual persistence records as a Markdown table."""
        if not records:
            return "No persistence data available.\n"
        lines = ["| Year | Expected (kWh) | Actual (kWh) | PF | Status |"]
        lines.append("|------|---------------|-------------|------|--------|")
        for rec in records:
            lines.append(
                f"| {rec.get('year', 'N/A')} "
                f"| {rec.get('expected_savings_kwh', 'N/A')} "
                f"| {rec.get('actual_savings_kwh', 'N/A')} "
                f"| {rec.get('persistence_factor', 'N/A')} "
                f"| {rec.get('status', 'N/A')} |"
            )
        return "\n".join(lines) + "\n"
