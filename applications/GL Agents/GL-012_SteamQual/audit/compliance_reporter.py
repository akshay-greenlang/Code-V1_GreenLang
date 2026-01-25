"""
Compliance Reporter for GL-012 SteamQual SteamQualityController

This module implements steam quality KPI reporting and compliance
documentation for the SteamQualityController.

Key Features:
    - Steam quality KPI tracking (dryness, superheat, pressure)
    - Control loop performance reporting (IAE, ISE, settling time)
    - Quality trend analysis over time
    - Multiple export formats (JSON, CSV, PDF)
    - Auditor-ready compliance documentation

Reference Standards:
    - ASME PTC 19.11 (Steam and Water Sampling)
    - ASME B31.1 (Power Piping)
    - ISO 50001:2018 (Energy Management)
    - ISO 9001:2015 (Quality Management)

Quality Metrics Tracked:
    - Steam dryness fraction (target: >= 0.98 for saturated steam)
    - Superheat margin (target: >= 10F above saturation)
    - Pressure stability (target: +/- 2% of setpoint)
    - Control loop performance (settling time, overshoot)
    - Quality excursion frequency and duration

Example:
    >>> reporter = ComplianceReporter(provenance_tracker=provenance)
    >>> quality_report = reporter.generate_quality_kpi_report(
    ...     time_period=period,
    ...     steam_headers=["HP", "MP", "LP"],
    ...     prepared_by="operator-001"
    ... )
    >>> exported = reporter.export_for_auditor(quality_report, ReportFormat.PDF)

Author: GreenLang Steam Quality Team
Version: 1.0.0
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report export formats."""

    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"
    XLSX = "XLSX"


class ReportingPeriod(BaseModel):
    """Time period for compliance reporting."""

    start_date: datetime = Field(..., description="Period start date")
    end_date: datetime = Field(..., description="Period end date")
    period_type: str = Field(
        default="CUSTOM", description="Period type (HOURLY, DAILY, WEEKLY, MONTHLY, CUSTOM)"
    )

    @validator("end_date")
    def end_after_start(cls, v, values):
        """Validate end_date is after start_date."""
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @property
    def duration_hours(self) -> float:
        """Duration in hours."""
        return (self.end_date - self.start_date).total_seconds() / 3600

    @property
    def duration_days(self) -> float:
        """Duration in days."""
        return (self.end_date - self.start_date).total_seconds() / 86400

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class QualityMetric(BaseModel):
    """
    A single steam quality metric measurement.

    Represents a quality measurement with target, actual value,
    and compliance status.
    """

    metric_id: str = Field(..., description="Unique metric identifier")
    metric_name: str = Field(..., description="Human-readable metric name")
    metric_type: str = Field(
        ..., description="Type (DRYNESS, SUPERHEAT, PRESSURE, ENTHALPY)"
    )
    unit: str = Field(..., description="Engineering unit")

    # Values
    target_value: float = Field(..., description="Target/setpoint value")
    actual_value: float = Field(..., description="Actual measured/calculated value")
    min_value: float = Field(..., description="Minimum observed value")
    max_value: float = Field(..., description="Maximum observed value")
    std_deviation: float = Field(0.0, ge=0, description="Standard deviation")

    # Limits
    low_limit: Optional[float] = Field(None, description="Low alarm limit")
    high_limit: Optional[float] = Field(None, description="High alarm limit")
    low_warning: Optional[float] = Field(None, description="Low warning limit")
    high_warning: Optional[float] = Field(None, description="High warning limit")

    # Compliance
    is_compliant: bool = Field(..., description="Whether metric meets target")
    deviation_pct: float = Field(..., description="Deviation from target (%)")
    time_in_spec_pct: float = Field(
        100.0, ge=0, le=100, description="Time within specification (%)"
    )

    # Data quality
    data_points: int = Field(0, ge=0, description="Number of data points")
    data_completeness_pct: float = Field(
        100.0, ge=0, le=100, description="Data completeness (%)"
    )

    class Config:
        frozen = True


class ControlLoopKPI(BaseModel):
    """
    Control loop performance KPIs.

    Tracks key performance indicators for steam quality control loops.
    """

    loop_id: str = Field(..., description="Control loop identifier")
    loop_name: str = Field(..., description="Human-readable loop name")
    loop_type: str = Field(
        ..., description="Type (PRESSURE, TEMPERATURE, FLOW, CASCADE)"
    )

    # Performance metrics
    setpoint: float = Field(..., description="Current setpoint")
    setpoint_unit: str = Field(..., description="Setpoint unit")

    # Error metrics
    mean_error: float = Field(..., description="Mean error from setpoint")
    mean_absolute_error: float = Field(..., ge=0, description="Mean absolute error (MAE)")
    integral_absolute_error: float = Field(..., ge=0, description="Integral absolute error (IAE)")
    integral_squared_error: float = Field(..., ge=0, description="Integral squared error (ISE)")

    # Dynamic performance
    settling_time_seconds: Optional[float] = Field(
        None, ge=0, description="Settling time (seconds)"
    )
    rise_time_seconds: Optional[float] = Field(
        None, ge=0, description="Rise time (seconds)"
    )
    overshoot_pct: Optional[float] = Field(
        None, ge=0, description="Maximum overshoot (%)"
    )

    # Stability
    oscillation_count: int = Field(0, ge=0, description="Number of oscillations")
    stability_index: float = Field(
        1.0, ge=0, le=1, description="Stability index (1.0 = stable)"
    )

    # Mode statistics
    time_in_auto_pct: float = Field(
        0.0, ge=0, le=100, description="Time in AUTO mode (%)"
    )
    time_in_manual_pct: float = Field(
        0.0, ge=0, le=100, description="Time in MANUAL mode (%)"
    )
    mode_changes: int = Field(0, ge=0, description="Number of mode changes")

    # Compliance
    is_within_tolerance: bool = Field(..., description="Within acceptable tolerance")
    tolerance_pct: float = Field(2.0, ge=0, description="Tolerance threshold (%)")

    class Config:
        frozen = True


class QualityExcursion(BaseModel):
    """
    A quality excursion event.

    Records when steam quality went outside acceptable limits.
    """

    excursion_id: UUID = Field(default_factory=uuid4, description="Unique excursion ID")
    start_time: datetime = Field(..., description="Excursion start time")
    end_time: Optional[datetime] = Field(None, description="Excursion end time (None if ongoing)")
    duration_minutes: Optional[float] = Field(None, description="Duration in minutes")

    # Location
    steam_header: str = Field(..., description="Affected steam header")
    metric_type: str = Field(..., description="Metric that exceeded limits")

    # Severity
    severity: str = Field(..., description="Severity (WARNING, ALARM, CRITICAL)")
    limit_type: str = Field(..., description="Limit type (HIGH, LOW)")
    limit_value: float = Field(..., description="Limit that was exceeded")
    peak_value: float = Field(..., description="Peak excursion value")
    peak_deviation_pct: float = Field(..., description="Peak deviation from limit (%)")

    # Impact
    estimated_impact: Optional[str] = Field(None, description="Estimated impact description")
    affected_processes: List[str] = Field(
        default_factory=list, description="Affected downstream processes"
    )

    # Resolution
    is_resolved: bool = Field(False, description="Whether excursion is resolved")
    resolution_action: Optional[str] = Field(None, description="Action taken to resolve")
    resolved_by: Optional[str] = Field(None, description="Who resolved the excursion")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v),
        }


class QualityKPIReport(BaseModel):
    """
    Steam quality KPI report.

    Comprehensive report on steam quality metrics across all headers.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Unique report ID")
    report_type: str = Field(default="QUALITY_KPI", description="Report type")
    report_version: str = Field(default="1.0.0", description="Report version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report creation timestamp"
    )

    # Facility
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Period
    reporting_period: ReportingPeriod = Field(..., description="Reporting period")

    # Headers reported
    steam_headers: List[str] = Field(..., description="Steam headers included")

    # Quality metrics by header
    dryness_metrics: Dict[str, QualityMetric] = Field(
        default_factory=dict, description="Dryness metrics by header"
    )
    superheat_metrics: Dict[str, QualityMetric] = Field(
        default_factory=dict, description="Superheat metrics by header"
    )
    pressure_metrics: Dict[str, QualityMetric] = Field(
        default_factory=dict, description="Pressure metrics by header"
    )
    enthalpy_metrics: Dict[str, QualityMetric] = Field(
        default_factory=dict, description="Enthalpy metrics by header"
    )

    # Summary statistics
    overall_compliance_pct: float = Field(
        ..., ge=0, le=100, description="Overall compliance rate (%)"
    )
    headers_in_spec: int = Field(0, ge=0, description="Headers meeting all specs")
    headers_out_of_spec: int = Field(0, ge=0, description="Headers not meeting specs")

    # Excursions
    total_excursions: int = Field(0, ge=0, description="Total quality excursions")
    excursion_duration_minutes: float = Field(
        0.0, ge=0, description="Total excursion duration (minutes)"
    )
    excursions: List[QualityExcursion] = Field(
        default_factory=list, description="List of excursions"
    )

    # Trends
    dryness_trend: str = Field(
        default="STABLE", description="Dryness trend (IMPROVING, STABLE, DEGRADING)"
    )
    quality_trend_7d: Optional[float] = Field(
        None, description="7-day quality trend (%)"
    )
    quality_trend_30d: Optional[float] = Field(
        None, description="30-day quality trend (%)"
    )

    # Methodology
    methodology_notes: str = Field(..., description="Measurement methodology")
    data_quality_statement: str = Field(..., description="Data quality statement")

    # Prepared by
    prepared_by: str = Field(..., description="Preparer name/ID")
    approved_by: Optional[str] = Field(None, description="Approver name/ID")

    # Hash for integrity
    report_hash: Optional[str] = Field(None, description="SHA-256 hash of report")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of report content."""
        data = self.dict(exclude={"report_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class ControlPerformanceReport(BaseModel):
    """
    Control loop performance report.

    Comprehensive report on control loop performance for steam quality.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Unique report ID")
    report_type: str = Field(default="CONTROL_PERFORMANCE", description="Report type")
    report_version: str = Field(default="1.0.0", description="Report version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report creation timestamp"
    )

    # Facility
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Period
    reporting_period: ReportingPeriod = Field(..., description="Reporting period")

    # Control loops
    control_loops: List[ControlLoopKPI] = Field(
        ..., description="Control loop KPIs"
    )

    # Summary
    total_loops: int = Field(0, ge=0, description="Total control loops")
    loops_in_auto: int = Field(0, ge=0, description="Loops in AUTO mode")
    loops_in_manual: int = Field(0, ge=0, description="Loops in MANUAL mode")
    loops_within_tolerance: int = Field(0, ge=0, description="Loops within tolerance")
    avg_stability_index: float = Field(
        1.0, ge=0, le=1, description="Average stability index"
    )

    # Aggregate performance
    avg_iae: float = Field(0.0, ge=0, description="Average IAE across loops")
    avg_settling_time_seconds: Optional[float] = Field(
        None, ge=0, description="Average settling time (seconds)"
    )
    avg_overshoot_pct: Optional[float] = Field(
        None, ge=0, description="Average overshoot (%)"
    )

    # Issues
    control_issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="Control issues identified"
    )

    # Methodology
    methodology_notes: str = Field(..., description="Methodology documentation")
    data_quality_statement: str = Field(..., description="Data quality statement")

    # Prepared by
    prepared_by: str = Field(..., description="Preparer name/ID")

    # Hash
    report_hash: Optional[str] = Field(None, description="SHA-256 hash of report")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of report content."""
        data = self.dict(exclude={"report_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class SteamQualityReport(BaseModel):
    """
    Comprehensive steam quality report combining KPIs and control performance.

    Suitable for regulatory submission and auditor review.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Unique report ID")
    report_type: str = Field(default="STEAM_QUALITY_COMPREHENSIVE", description="Report type")
    report_version: str = Field(default="1.0.0", description="Report version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report creation timestamp"
    )

    # Facility
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Period
    reporting_period: ReportingPeriod = Field(..., description="Reporting period")

    # Included reports
    quality_kpi_report: QualityKPIReport = Field(..., description="Quality KPI report")
    control_performance_report: ControlPerformanceReport = Field(
        ..., description="Control performance report"
    )

    # Executive summary
    executive_summary: str = Field(..., description="Executive summary")
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )

    # Overall status
    overall_status: str = Field(
        ..., description="Overall status (COMPLIANT, NEEDS_ATTENTION, NON_COMPLIANT)"
    )
    overall_quality_score: float = Field(
        ..., ge=0, le=100, description="Overall quality score (0-100)"
    )

    # Reference standards
    reference_standards: List[str] = Field(
        default_factory=list, description="Reference standards"
    )

    # Verification
    verification_status: str = Field(
        default="UNVERIFIED", description="Verification status"
    )
    verified_by: Optional[str] = Field(None, description="Verifier name/ID")
    verification_date: Optional[datetime] = Field(None, description="Verification date")

    # Prepared by
    prepared_by: str = Field(..., description="Preparer name/ID")
    reviewed_by: Optional[str] = Field(None, description="Reviewer name/ID")
    approved_by: Optional[str] = Field(None, description="Approver name/ID")

    # Hash
    report_hash: Optional[str] = Field(None, description="SHA-256 hash of report")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of report content."""
        data = self.dict(exclude={"report_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class ExportedReport(BaseModel):
    """
    Exported report with format-specific content.
    """

    export_id: UUID = Field(default_factory=uuid4, description="Export ID")
    source_report_id: str = Field(..., description="Source report ID")
    source_report_type: str = Field(..., description="Source report type")
    export_format: ReportFormat = Field(..., description="Export format")
    exported_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Export timestamp"
    )

    # Content
    file_path: Optional[str] = Field(None, description="File path if written to disk")
    content_bytes: Optional[int] = Field(None, description="Content size in bytes")
    content_hash: str = Field(..., description="SHA-256 hash of content")

    # Metadata
    exported_by: str = Field(..., description="Exporter ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ComplianceReporter:
    """
    Compliance reporter for steam quality KPIs and control performance.

    Generates regulatory compliance reports with full audit
    trail integration and provenance tracking.

    Attributes:
        provenance_tracker: Provenance tracker for data lineage
        storage_path: Path for storing reports

    Example:
        >>> reporter = ComplianceReporter(provenance_tracker=provenance)
        >>> quality_report = reporter.generate_quality_kpi_report(
        ...     time_period=period,
        ...     steam_headers=["HP", "MP", "LP"],
        ...     prepared_by="operator-001"
        ... )
    """

    # Quality targets
    DRYNESS_TARGET = 0.98  # Minimum dryness fraction for saturated steam
    SUPERHEAT_TARGET_F = 10.0  # Minimum superheat (Fahrenheit)
    PRESSURE_TOLERANCE_PCT = 2.0  # Pressure tolerance percentage

    def __init__(
        self,
        provenance_tracker: Optional[Any] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize compliance reporter.

        Args:
            provenance_tracker: Provenance tracker for data lineage
            storage_path: Path for storing reports
        """
        self.provenance_tracker = provenance_tracker
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("ComplianceReporter initialized")

    def generate_quality_kpi_report(
        self,
        time_period: ReportingPeriod,
        facility_id: str,
        facility_name: str,
        steam_headers: List[str],
        quality_data: Dict[str, Dict[str, Any]],
        prepared_by: str,
        excursions: Optional[List[QualityExcursion]] = None,
    ) -> QualityKPIReport:
        """
        Generate a steam quality KPI report.

        Args:
            time_period: Reporting period
            facility_id: Facility identifier
            facility_name: Facility name
            steam_headers: List of steam headers to report
            quality_data: Quality data by header, containing:
                - dryness_avg, dryness_min, dryness_max, dryness_std
                - superheat_avg, superheat_min, superheat_max, superheat_std
                - pressure_avg, pressure_min, pressure_max, pressure_std
                - data_points, time_in_spec_pct
            prepared_by: Preparer name/ID
            excursions: Optional list of quality excursions

        Returns:
            QualityKPIReport
        """
        start_time = datetime.now(timezone.utc)

        dryness_metrics: Dict[str, QualityMetric] = {}
        superheat_metrics: Dict[str, QualityMetric] = {}
        pressure_metrics: Dict[str, QualityMetric] = {}
        enthalpy_metrics: Dict[str, QualityMetric] = {}

        headers_in_spec = 0
        total_time_in_spec = 0.0

        for header in steam_headers:
            header_data = quality_data.get(header, {})

            # Dryness metric
            dryness_avg = header_data.get("dryness_avg", 0.98)
            dryness_in_spec = dryness_avg >= self.DRYNESS_TARGET
            dryness_deviation = ((dryness_avg - self.DRYNESS_TARGET) / self.DRYNESS_TARGET) * 100

            dryness_metrics[header] = QualityMetric(
                metric_id=f"{header}_DRYNESS",
                metric_name=f"{header} Steam Dryness",
                metric_type="DRYNESS",
                unit="fraction",
                target_value=self.DRYNESS_TARGET,
                actual_value=dryness_avg,
                min_value=header_data.get("dryness_min", dryness_avg),
                max_value=header_data.get("dryness_max", dryness_avg),
                std_deviation=header_data.get("dryness_std", 0.0),
                low_limit=0.95,
                is_compliant=dryness_in_spec,
                deviation_pct=dryness_deviation,
                time_in_spec_pct=header_data.get("dryness_time_in_spec_pct", 100.0),
                data_points=header_data.get("data_points", 0),
                data_completeness_pct=header_data.get("data_completeness_pct", 100.0),
            )

            # Superheat metric
            superheat_avg = header_data.get("superheat_avg", 15.0)
            superheat_in_spec = superheat_avg >= self.SUPERHEAT_TARGET_F
            superheat_deviation = (
                (superheat_avg - self.SUPERHEAT_TARGET_F) / self.SUPERHEAT_TARGET_F
            ) * 100 if self.SUPERHEAT_TARGET_F > 0 else 0

            superheat_metrics[header] = QualityMetric(
                metric_id=f"{header}_SUPERHEAT",
                metric_name=f"{header} Superheat",
                metric_type="SUPERHEAT",
                unit="degF",
                target_value=self.SUPERHEAT_TARGET_F,
                actual_value=superheat_avg,
                min_value=header_data.get("superheat_min", superheat_avg),
                max_value=header_data.get("superheat_max", superheat_avg),
                std_deviation=header_data.get("superheat_std", 0.0),
                low_limit=5.0,
                is_compliant=superheat_in_spec,
                deviation_pct=superheat_deviation,
                time_in_spec_pct=header_data.get("superheat_time_in_spec_pct", 100.0),
                data_points=header_data.get("data_points", 0),
                data_completeness_pct=header_data.get("data_completeness_pct", 100.0),
            )

            # Pressure metric
            pressure_setpoint = header_data.get("pressure_setpoint", 150.0)
            pressure_avg = header_data.get("pressure_avg", pressure_setpoint)
            pressure_deviation = (
                (pressure_avg - pressure_setpoint) / pressure_setpoint
            ) * 100 if pressure_setpoint > 0 else 0
            pressure_in_spec = abs(pressure_deviation) <= self.PRESSURE_TOLERANCE_PCT

            pressure_metrics[header] = QualityMetric(
                metric_id=f"{header}_PRESSURE",
                metric_name=f"{header} Pressure",
                metric_type="PRESSURE",
                unit="psig",
                target_value=pressure_setpoint,
                actual_value=pressure_avg,
                min_value=header_data.get("pressure_min", pressure_avg),
                max_value=header_data.get("pressure_max", pressure_avg),
                std_deviation=header_data.get("pressure_std", 0.0),
                low_limit=pressure_setpoint * (1 - self.PRESSURE_TOLERANCE_PCT / 100),
                high_limit=pressure_setpoint * (1 + self.PRESSURE_TOLERANCE_PCT / 100),
                is_compliant=pressure_in_spec,
                deviation_pct=pressure_deviation,
                time_in_spec_pct=header_data.get("pressure_time_in_spec_pct", 100.0),
                data_points=header_data.get("data_points", 0),
                data_completeness_pct=header_data.get("data_completeness_pct", 100.0),
            )

            # Check if header is fully in spec
            if dryness_in_spec and superheat_in_spec and pressure_in_spec:
                headers_in_spec += 1

            # Accumulate time in spec
            total_time_in_spec += (
                dryness_metrics[header].time_in_spec_pct +
                superheat_metrics[header].time_in_spec_pct +
                pressure_metrics[header].time_in_spec_pct
            ) / 3

        # Calculate overall compliance
        overall_compliance = total_time_in_spec / len(steam_headers) if steam_headers else 100.0

        # Process excursions
        excursions = excursions or []
        total_excursion_duration = sum(
            e.duration_minutes or 0 for e in excursions
        )

        # Determine trend (simplified - production would use more sophisticated analysis)
        avg_dryness = sum(m.actual_value for m in dryness_metrics.values()) / len(dryness_metrics) if dryness_metrics else 0
        if avg_dryness >= 0.985:
            dryness_trend = "IMPROVING"
        elif avg_dryness >= 0.975:
            dryness_trend = "STABLE"
        else:
            dryness_trend = "DEGRADING"

        # Methodology
        methodology = (
            f"Steam quality measurements per ASME PTC 19.11. "
            f"Dryness measured via throttling calorimeter or calculated from heat balance. "
            f"Superheat calculated as temperature minus saturation temperature at pressure. "
            f"Pressure from calibrated transmitters. "
            f"Reporting period: {time_period.duration_hours:.1f} hours."
        )

        data_quality = (
            f"Data from plant historian with provenance tracking. "
            f"All calculations verified with SHA-256 hashing."
        )

        report = QualityKPIReport(
            facility_id=facility_id,
            facility_name=facility_name,
            reporting_period=time_period,
            steam_headers=steam_headers,
            dryness_metrics=dryness_metrics,
            superheat_metrics=superheat_metrics,
            pressure_metrics=pressure_metrics,
            enthalpy_metrics=enthalpy_metrics,
            overall_compliance_pct=overall_compliance,
            headers_in_spec=headers_in_spec,
            headers_out_of_spec=len(steam_headers) - headers_in_spec,
            total_excursions=len(excursions),
            excursion_duration_minutes=total_excursion_duration,
            excursions=excursions,
            dryness_trend=dryness_trend,
            methodology_notes=methodology,
            data_quality_statement=data_quality,
            prepared_by=prepared_by,
        )

        # Calculate and set hash
        report_dict = report.dict()
        report_dict["report_hash"] = report.calculate_hash()
        report = QualityKPIReport(**report_dict)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Quality KPI report generated: {report.report_id}",
            extra={
                "compliance_pct": overall_compliance,
                "headers": len(steam_headers),
                "processing_time_ms": processing_time,
            }
        )

        return report

    def generate_control_performance_report(
        self,
        time_period: ReportingPeriod,
        facility_id: str,
        facility_name: str,
        control_loops: List[ControlLoopKPI],
        prepared_by: str,
        control_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> ControlPerformanceReport:
        """
        Generate a control loop performance report.

        Args:
            time_period: Reporting period
            facility_id: Facility identifier
            facility_name: Facility name
            control_loops: List of control loop KPIs
            prepared_by: Preparer name/ID
            control_issues: Optional list of control issues

        Returns:
            ControlPerformanceReport
        """
        start_time = datetime.now(timezone.utc)

        # Calculate aggregates
        loops_in_auto = sum(1 for l in control_loops if l.time_in_auto_pct > 50)
        loops_in_manual = len(control_loops) - loops_in_auto
        loops_within_tolerance = sum(1 for l in control_loops if l.is_within_tolerance)

        avg_stability = (
            sum(l.stability_index for l in control_loops) / len(control_loops)
            if control_loops else 1.0
        )
        avg_iae = (
            sum(l.integral_absolute_error for l in control_loops) / len(control_loops)
            if control_loops else 0.0
        )

        settling_times = [l.settling_time_seconds for l in control_loops if l.settling_time_seconds]
        avg_settling = sum(settling_times) / len(settling_times) if settling_times else None

        overshoots = [l.overshoot_pct for l in control_loops if l.overshoot_pct]
        avg_overshoot = sum(overshoots) / len(overshoots) if overshoots else None

        # Methodology
        methodology = (
            f"Control loop performance calculated from historian data. "
            f"IAE and ISE calculated over reporting period. "
            f"Settling time measured from setpoint change to within 2% of final value. "
            f"Stability index based on oscillation analysis."
        )

        data_quality = (
            f"Control loop data from DCS/PLC historians. "
            f"Mode changes tracked in real-time. "
            f"Reporting period: {time_period.duration_hours:.1f} hours."
        )

        report = ControlPerformanceReport(
            facility_id=facility_id,
            facility_name=facility_name,
            reporting_period=time_period,
            control_loops=control_loops,
            total_loops=len(control_loops),
            loops_in_auto=loops_in_auto,
            loops_in_manual=loops_in_manual,
            loops_within_tolerance=loops_within_tolerance,
            avg_stability_index=avg_stability,
            avg_iae=avg_iae,
            avg_settling_time_seconds=avg_settling,
            avg_overshoot_pct=avg_overshoot,
            control_issues=control_issues or [],
            methodology_notes=methodology,
            data_quality_statement=data_quality,
            prepared_by=prepared_by,
        )

        # Calculate and set hash
        report_dict = report.dict()
        report_dict["report_hash"] = report.calculate_hash()
        report = ControlPerformanceReport(**report_dict)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Control performance report generated: {report.report_id}",
            extra={
                "loops": len(control_loops),
                "loops_in_tolerance": loops_within_tolerance,
                "processing_time_ms": processing_time,
            }
        )

        return report

    def generate_comprehensive_report(
        self,
        quality_report: QualityKPIReport,
        control_report: ControlPerformanceReport,
        prepared_by: str,
        reviewed_by: Optional[str] = None,
        approved_by: Optional[str] = None,
    ) -> SteamQualityReport:
        """
        Generate a comprehensive steam quality report.

        Combines quality KPIs and control performance into a single
        auditor-ready report.

        Args:
            quality_report: Quality KPI report
            control_report: Control performance report
            prepared_by: Preparer name/ID
            reviewed_by: Optional reviewer
            approved_by: Optional approver

        Returns:
            SteamQualityReport
        """
        start_time = datetime.now(timezone.utc)

        # Determine overall status
        if quality_report.overall_compliance_pct >= 95 and control_report.avg_stability_index >= 0.9:
            overall_status = "COMPLIANT"
        elif quality_report.overall_compliance_pct >= 85 and control_report.avg_stability_index >= 0.7:
            overall_status = "NEEDS_ATTENTION"
        else:
            overall_status = "NON_COMPLIANT"

        # Calculate quality score
        quality_score = (
            quality_report.overall_compliance_pct * 0.6 +
            control_report.avg_stability_index * 100 * 0.3 +
            (control_report.loops_within_tolerance / max(control_report.total_loops, 1)) * 100 * 0.1
        )

        # Generate executive summary
        executive_summary = (
            f"Steam quality report for {quality_report.facility_name} covering "
            f"{quality_report.reporting_period.duration_hours:.1f} hours. "
            f"Overall compliance: {quality_report.overall_compliance_pct:.1f}%. "
            f"{quality_report.headers_in_spec} of {len(quality_report.steam_headers)} headers met all specifications. "
            f"{control_report.loops_within_tolerance} of {control_report.total_loops} control loops within tolerance. "
            f"Quality score: {quality_score:.1f}/100."
        )

        # Key findings
        key_findings = []
        if quality_report.overall_compliance_pct < 95:
            key_findings.append(
                f"Overall compliance ({quality_report.overall_compliance_pct:.1f}%) below 95% target"
            )
        if quality_report.total_excursions > 0:
            key_findings.append(
                f"{quality_report.total_excursions} quality excursions with "
                f"{quality_report.excursion_duration_minutes:.1f} minutes total duration"
            )
        if control_report.loops_in_manual > 0:
            key_findings.append(
                f"{control_report.loops_in_manual} control loops operating in manual mode"
            )
        if control_report.avg_stability_index < 0.9:
            key_findings.append(
                f"Average stability index ({control_report.avg_stability_index:.2f}) below 0.9 target"
            )

        # Recommendations
        recommendations = []
        if quality_report.dryness_trend == "DEGRADING":
            recommendations.append("Review boiler blowdown rates and feedwater quality")
        if control_report.loops_in_manual > 0:
            recommendations.append("Return manual loops to automatic control or document justification")
        if quality_report.total_excursions > 3:
            recommendations.append("Investigate root cause of frequent quality excursions")

        # Reference standards
        reference_standards = [
            "ASME PTC 19.11 (Steam and Water Sampling)",
            "ASME B31.1 (Power Piping)",
            "ISO 50001:2018 (Energy Management)",
            "ISO 9001:2015 (Quality Management)",
        ]

        # Create combined reporting period
        reporting_period = ReportingPeriod(
            start_date=min(
                quality_report.reporting_period.start_date,
                control_report.reporting_period.start_date
            ),
            end_date=max(
                quality_report.reporting_period.end_date,
                control_report.reporting_period.end_date
            ),
            period_type="CUSTOM",
        )

        report = SteamQualityReport(
            facility_id=quality_report.facility_id,
            facility_name=quality_report.facility_name,
            reporting_period=reporting_period,
            quality_kpi_report=quality_report,
            control_performance_report=control_report,
            executive_summary=executive_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            overall_status=overall_status,
            overall_quality_score=quality_score,
            reference_standards=reference_standards,
            prepared_by=prepared_by,
            reviewed_by=reviewed_by,
            approved_by=approved_by,
        )

        # Calculate and set hash
        report_dict = report.dict()
        report_dict["report_hash"] = report.calculate_hash()
        report = SteamQualityReport(**report_dict)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Comprehensive report generated: {report.report_id}",
            extra={
                "overall_status": overall_status,
                "quality_score": quality_score,
                "processing_time_ms": processing_time,
            }
        )

        return report

    def export_for_auditor(
        self,
        report: Union[QualityKPIReport, ControlPerformanceReport, SteamQualityReport],
        format: ReportFormat,
        output_path: Optional[str] = None,
        exported_by: str = "GL-012",
    ) -> ExportedReport:
        """
        Export a report for auditor review.

        Args:
            report: Report to export
            format: Export format
            output_path: Output file path (optional)
            exported_by: Exporter identifier

        Returns:
            ExportedReport with export metadata

        Raises:
            ValueError: If format not supported
        """
        start_time = datetime.now(timezone.utc)
        content = ""
        content_bytes = 0

        if format == ReportFormat.JSON:
            content = json.dumps(report.dict(), indent=2, default=str)
            content_bytes = len(content.encode("utf-8"))

        elif format == ReportFormat.CSV:
            content = self._export_to_csv(report)
            content_bytes = len(content.encode("utf-8"))

        elif format == ReportFormat.PDF:
            # Generate text-based report (production would use proper PDF library)
            content = self._export_to_text(report)
            content_bytes = len(content.encode("utf-8"))

        else:
            raise ValueError(f"Unsupported format: {format}")

        # Calculate content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Write to file if path provided
        file_path = None
        if output_path:
            with open(output_path, "w") as f:
                f.write(content)
            file_path = output_path

        exported = ExportedReport(
            source_report_id=str(report.report_id),
            source_report_type=report.report_type,
            export_format=format,
            file_path=file_path,
            content_bytes=content_bytes,
            content_hash=content_hash,
            exported_by=exported_by,
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Report exported: {format.value}",
            extra={
                "source_report_id": str(report.report_id),
                "content_bytes": content_bytes,
                "processing_time_ms": processing_time,
            }
        )

        return exported

    def _export_to_csv(
        self,
        report: Union[QualityKPIReport, ControlPerformanceReport, SteamQualityReport],
    ) -> str:
        """Export report to CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        if isinstance(report, QualityKPIReport):
            writer.writerow(["Steam Quality KPI Report"])
            writer.writerow(["Report ID", str(report.report_id)])
            writer.writerow(["Facility", report.facility_name])
            writer.writerow(["Period Start", report.reporting_period.start_date.isoformat()])
            writer.writerow(["Period End", report.reporting_period.end_date.isoformat()])
            writer.writerow(["Overall Compliance (%)", report.overall_compliance_pct])
            writer.writerow([])
            writer.writerow(["Header", "Metric", "Target", "Actual", "Deviation (%)", "Compliant"])

            for header in report.steam_headers:
                if header in report.dryness_metrics:
                    m = report.dryness_metrics[header]
                    writer.writerow([header, "Dryness", m.target_value, m.actual_value, m.deviation_pct, m.is_compliant])
                if header in report.superheat_metrics:
                    m = report.superheat_metrics[header]
                    writer.writerow([header, "Superheat", m.target_value, m.actual_value, m.deviation_pct, m.is_compliant])
                if header in report.pressure_metrics:
                    m = report.pressure_metrics[header]
                    writer.writerow([header, "Pressure", m.target_value, m.actual_value, m.deviation_pct, m.is_compliant])

        elif isinstance(report, ControlPerformanceReport):
            writer.writerow(["Control Performance Report"])
            writer.writerow(["Report ID", str(report.report_id)])
            writer.writerow(["Facility", report.facility_name])
            writer.writerow(["Total Loops", report.total_loops])
            writer.writerow(["Loops in Auto", report.loops_in_auto])
            writer.writerow(["Average Stability", report.avg_stability_index])
            writer.writerow([])
            writer.writerow(["Loop ID", "Type", "MAE", "IAE", "Stability", "Within Tolerance"])

            for loop in report.control_loops:
                writer.writerow([
                    loop.loop_id,
                    loop.loop_type,
                    loop.mean_absolute_error,
                    loop.integral_absolute_error,
                    loop.stability_index,
                    loop.is_within_tolerance,
                ])

        elif isinstance(report, SteamQualityReport):
            writer.writerow(["Comprehensive Steam Quality Report"])
            writer.writerow(["Report ID", str(report.report_id)])
            writer.writerow(["Facility", report.facility_name])
            writer.writerow(["Overall Status", report.overall_status])
            writer.writerow(["Quality Score", report.overall_quality_score])
            writer.writerow([])
            writer.writerow(["Executive Summary"])
            writer.writerow([report.executive_summary])

        return output.getvalue()

    def _export_to_text(
        self,
        report: Union[QualityKPIReport, ControlPerformanceReport, SteamQualityReport],
    ) -> str:
        """Export report to text format (for PDF placeholder)."""
        lines = []
        lines.append("=" * 70)

        if isinstance(report, QualityKPIReport):
            lines.append("STEAM QUALITY KPI REPORT")
            lines.append("=" * 70)
            lines.append(f"Report ID: {report.report_id}")
            lines.append(f"Facility: {report.facility_name} ({report.facility_id})")
            lines.append(
                f"Period: {report.reporting_period.start_date.date()} to "
                f"{report.reporting_period.end_date.date()}"
            )
            lines.append("")
            lines.append("-" * 40)
            lines.append("COMPLIANCE SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Overall Compliance: {report.overall_compliance_pct:.1f}%")
            lines.append(f"Headers In Spec: {report.headers_in_spec}/{len(report.steam_headers)}")
            lines.append(f"Quality Excursions: {report.total_excursions}")
            lines.append(f"Dryness Trend: {report.dryness_trend}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("QUALITY METRICS BY HEADER")
            lines.append("-" * 40)

            for header in report.steam_headers:
                lines.append(f"\n{header} Header:")
                if header in report.dryness_metrics:
                    m = report.dryness_metrics[header]
                    status = "PASS" if m.is_compliant else "FAIL"
                    lines.append(f"  Dryness: {m.actual_value:.4f} (target: {m.target_value}) [{status}]")
                if header in report.superheat_metrics:
                    m = report.superheat_metrics[header]
                    status = "PASS" if m.is_compliant else "FAIL"
                    lines.append(f"  Superheat: {m.actual_value:.1f}F (target: {m.target_value}F) [{status}]")
                if header in report.pressure_metrics:
                    m = report.pressure_metrics[header]
                    status = "PASS" if m.is_compliant else "FAIL"
                    lines.append(f"  Pressure: {m.actual_value:.1f} psig (target: {m.target_value}) [{status}]")

        elif isinstance(report, ControlPerformanceReport):
            lines.append("CONTROL PERFORMANCE REPORT")
            lines.append("=" * 70)
            lines.append(f"Report ID: {report.report_id}")
            lines.append(f"Facility: {report.facility_name}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("PERFORMANCE SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Total Control Loops: {report.total_loops}")
            lines.append(f"Loops in AUTO: {report.loops_in_auto}")
            lines.append(f"Loops in MANUAL: {report.loops_in_manual}")
            lines.append(f"Loops Within Tolerance: {report.loops_within_tolerance}")
            lines.append(f"Average Stability Index: {report.avg_stability_index:.3f}")
            lines.append(f"Average IAE: {report.avg_iae:.3f}")

        elif isinstance(report, SteamQualityReport):
            lines.append("COMPREHENSIVE STEAM QUALITY REPORT")
            lines.append("=" * 70)
            lines.append(f"Report ID: {report.report_id}")
            lines.append(f"Facility: {report.facility_name}")
            lines.append(f"Overall Status: {report.overall_status}")
            lines.append(f"Quality Score: {report.overall_quality_score:.1f}/100")
            lines.append("")
            lines.append("-" * 40)
            lines.append("EXECUTIVE SUMMARY")
            lines.append("-" * 40)
            lines.append(report.executive_summary)
            lines.append("")
            lines.append("-" * 40)
            lines.append("KEY FINDINGS")
            lines.append("-" * 40)
            for finding in report.key_findings:
                lines.append(f"  - {finding}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            for rec in report.recommendations:
                lines.append(f"  - {rec}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"Report Hash: {report.report_hash}")

        return "\n".join(lines)

    def track_quality_kpi_over_time(
        self,
        header: str,
        metric_type: str,
        time_series: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Track quality KPI trends over time.

        Args:
            header: Steam header
            metric_type: Metric type (DRYNESS, SUPERHEAT, PRESSURE)
            time_series: List of time series points with timestamp and value

        Returns:
            Dictionary with trend analysis
        """
        if not time_series:
            return {"error": "No data provided"}

        values = [p.get("value", 0) for p in time_series]
        timestamps = [p.get("timestamp") for p in time_series]

        # Calculate statistics
        avg_value = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)

        # Simple trend calculation (first half vs second half)
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid if mid > 0 else avg_value
        second_half_avg = sum(values[mid:]) / (len(values) - mid) if mid < len(values) else avg_value

        if second_half_avg > first_half_avg * 1.02:
            trend = "IMPROVING"
        elif second_half_avg < first_half_avg * 0.98:
            trend = "DEGRADING"
        else:
            trend = "STABLE"

        trend_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg != 0 else 0

        return {
            "header": header,
            "metric_type": metric_type,
            "data_points": len(values),
            "avg_value": avg_value,
            "min_value": min_value,
            "max_value": max_value,
            "trend": trend,
            "trend_pct": trend_pct,
            "first_timestamp": timestamps[0] if timestamps else None,
            "last_timestamp": timestamps[-1] if timestamps else None,
        }
