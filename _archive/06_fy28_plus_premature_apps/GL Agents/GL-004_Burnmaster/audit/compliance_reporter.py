"""
ComplianceReporter - Regulatory compliance reporting for BURNMASTER.

This module implements the ComplianceReporter for GL-004 BURNMASTER, generating
emissions reports, permit compliance reports, optimization audit reports,
and safety compliance reports.

Supports regulatory requirements for industrial combustion systems with
full audit trail and export capabilities.

Example:
    >>> reporter = ComplianceReporter(config)
    >>> emissions_report = reporter.generate_emissions_report(period)
    >>> export = reporter.export_report(emissions_report, "pdf")
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, date, timezone, timedelta
from enum import Enum
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ReportType(str, Enum):
    """Types of compliance reports."""
    EMISSIONS = "emissions"
    PERMIT = "permit"
    OPTIMIZATION_AUDIT = "optimization_audit"
    SAFETY = "safety"
    OPERATIONAL = "operational"


class ReportStatus(str, Enum):
    """Status of compliance reports."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    REJECTED = "rejected"


class ExportFormat(str, Enum):
    """Export formats for reports."""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XML = "xml"
    EXCEL = "excel"


class ComplianceStatus(str, Enum):
    """Compliance status indicators."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"


# =============================================================================
# Input Models
# =============================================================================

class DateRange(BaseModel):
    """Date range for report period."""

    start_date: datetime = Field(..., description="Start of reporting period")
    end_date: datetime = Field(..., description="End of reporting period")

    @validator('end_date')
    def end_after_start(cls, v, values):
        """Validate end date is after start date."""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v

    @property
    def duration_days(self) -> int:
        """Get duration in days."""
        return (self.end_date - self.start_date).days


# =============================================================================
# Report Component Models
# =============================================================================

class EmissionsDataPoint(BaseModel):
    """Single emissions measurement."""

    timestamp: datetime = Field(..., description="Measurement timestamp")
    boiler_id: str = Field(..., description="Boiler identifier")
    nox_ppm: float = Field(..., ge=0, description="NOx concentration (ppm)")
    co_ppm: float = Field(..., ge=0, description="CO concentration (ppm)")
    so2_ppm: float = Field(0, ge=0, description="SO2 concentration (ppm)")
    particulate_mg_m3: float = Field(0, ge=0, description="Particulate matter (mg/m3)")
    co2_percent: float = Field(0, ge=0, description="CO2 percentage")
    o2_percent: float = Field(..., ge=0, le=21, description="O2 percentage")
    flue_gas_flow_m3_h: float = Field(..., gt=0, description="Flue gas flow (m3/h)")
    stack_temperature_c: float = Field(..., description="Stack temperature (C)")


class EmissionsSummary(BaseModel):
    """Summary of emissions for a period."""

    total_nox_kg: float = Field(..., ge=0, description="Total NOx emissions (kg)")
    total_co_kg: float = Field(..., ge=0, description="Total CO emissions (kg)")
    total_so2_kg: float = Field(0, ge=0, description="Total SO2 emissions (kg)")
    total_particulate_kg: float = Field(0, ge=0, description="Total particulate (kg)")
    total_co2_tonnes: float = Field(0, ge=0, description="Total CO2 emissions (tonnes)")
    avg_nox_ppm: float = Field(..., ge=0, description="Average NOx (ppm)")
    avg_co_ppm: float = Field(..., ge=0, description="Average CO (ppm)")
    max_nox_ppm: float = Field(..., ge=0, description="Maximum NOx (ppm)")
    max_co_ppm: float = Field(..., ge=0, description="Maximum CO (ppm)")
    measurement_count: int = Field(..., ge=0, description="Number of measurements")
    operating_hours: float = Field(..., ge=0, description="Total operating hours")


class PermitLimit(BaseModel):
    """Permit limit definition."""

    parameter: str = Field(..., description="Parameter name")
    limit_value: float = Field(..., description="Limit value")
    unit: str = Field(..., description="Unit of measurement")
    averaging_period: str = Field(..., description="Averaging period")
    actual_value: float = Field(..., description="Actual measured value")
    compliance_status: ComplianceStatus = Field(..., description="Compliance status")
    margin_percent: float = Field(..., description="Margin to limit (%)")


class OptimizationEvent(BaseModel):
    """Record of an optimization event."""

    event_id: str = Field(..., description="Event identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    optimization_type: str = Field(..., description="Type of optimization")
    before_state: Dict[str, float] = Field(..., description="State before optimization")
    after_state: Dict[str, float] = Field(..., description="State after optimization")
    improvement_achieved: Dict[str, float] = Field(..., description="Improvements achieved")
    model_used: str = Field(..., description="Model used for optimization")
    constraints_satisfied: bool = Field(..., description="All constraints satisfied")


class SafetyIncident(BaseModel):
    """Record of a safety incident."""

    incident_id: str = Field(..., description="Incident identifier")
    timestamp: datetime = Field(..., description="Incident timestamp")
    incident_type: str = Field(..., description="Type of incident")
    severity: str = Field(..., description="Incident severity")
    description: str = Field(..., description="Incident description")
    root_cause: Optional[str] = Field(None, description="Root cause if determined")
    corrective_action: Optional[str] = Field(None, description="Corrective action taken")
    resolution_status: str = Field(..., description="Resolution status")


# =============================================================================
# Report Output Models
# =============================================================================

class ComplianceReport(BaseModel):
    """Base class for compliance reports."""

    report_id: str = Field(..., description="Unique report identifier")
    report_type: ReportType = Field(..., description="Type of report")
    title: str = Field(..., description="Report title")
    period: DateRange = Field(..., description="Reporting period")
    generated_at: datetime = Field(..., description="Generation timestamp")
    generated_by: str = Field(..., description="Generator system ID")
    status: ReportStatus = Field(ReportStatus.DRAFT, description="Report status")
    report_hash: str = Field(..., description="SHA-256 hash of report content")

    # Metadata
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")
    regulatory_authority: str = Field(..., description="Regulatory authority")
    permit_number: Optional[str] = Field(None, description="Permit number")

    class Config:
        """Pydantic configuration."""
        frozen = True


class EmissionsReport(ComplianceReport):
    """Emissions compliance report."""

    report_type: ReportType = Field(ReportType.EMISSIONS, const=True)

    # Emissions data
    emissions_summary: EmissionsSummary = Field(..., description="Emissions summary")
    emissions_by_boiler: Dict[str, EmissionsSummary] = Field(
        default_factory=dict,
        description="Emissions by boiler"
    )
    daily_emissions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Daily emissions data"
    )

    # Compliance
    permit_limits: List[PermitLimit] = Field(
        default_factory=list,
        description="Permit limits and compliance"
    )
    overall_compliance: ComplianceStatus = Field(..., description="Overall compliance status")
    exceedances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Permit exceedances"
    )

    # Supporting data
    data_completeness_percent: float = Field(..., ge=0, le=100, description="Data completeness")
    measurement_methodology: str = Field(..., description="Measurement methodology")


class PermitReport(ComplianceReport):
    """Permit compliance report."""

    report_type: ReportType = Field(ReportType.PERMIT, const=True)

    # Permit details
    permit_conditions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Permit conditions"
    )
    condition_compliance: List[PermitLimit] = Field(
        default_factory=list,
        description="Compliance with each condition"
    )

    # Compliance summary
    overall_compliance: ComplianceStatus = Field(..., description="Overall compliance status")
    conditions_met: int = Field(..., ge=0, description="Number of conditions met")
    conditions_total: int = Field(..., ge=0, description="Total number of conditions")
    compliance_percentage: float = Field(..., ge=0, le=100, description="Compliance percentage")

    # Deviations
    deviations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Permit deviations"
    )
    corrective_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Corrective actions"
    )


class OptAuditReport(ComplianceReport):
    """Optimization audit report."""

    report_type: ReportType = Field(ReportType.OPTIMIZATION_AUDIT, const=True)

    # Optimization summary
    total_optimizations: int = Field(..., ge=0, description="Total optimizations")
    successful_optimizations: int = Field(..., ge=0, description="Successful optimizations")
    optimization_success_rate: float = Field(..., ge=0, le=100, description="Success rate")

    # Performance improvements
    efficiency_improvement_percent: float = Field(
        ...,
        description="Average efficiency improvement"
    )
    emissions_reduction_percent: float = Field(
        ...,
        description="Average emissions reduction"
    )
    fuel_savings_percent: float = Field(..., description="Average fuel savings")

    # Optimization events
    optimization_events: List[OptimizationEvent] = Field(
        default_factory=list,
        description="Optimization events"
    )

    # Model performance
    model_accuracy_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Model accuracy metrics"
    )
    model_versions_used: List[str] = Field(
        default_factory=list,
        description="Model versions used"
    )

    # Constraints
    constraint_violations: int = Field(0, ge=0, description="Constraint violations")
    constraint_compliance_rate: float = Field(..., ge=0, le=100, description="Constraint compliance")


class SafetyReport(ComplianceReport):
    """Safety compliance report."""

    report_type: ReportType = Field(ReportType.SAFETY, const=True)

    # Safety summary
    total_incidents: int = Field(..., ge=0, description="Total incidents")
    incidents_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Incidents by severity"
    )
    incidents_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Incidents by type"
    )

    # Safety metrics
    days_without_incident: int = Field(..., ge=0, description="Days without incident")
    safety_score: float = Field(..., ge=0, le=100, description="Safety score")
    near_misses: int = Field(0, ge=0, description="Near miss events")

    # Incidents
    incidents: List[SafetyIncident] = Field(
        default_factory=list,
        description="Safety incidents"
    )

    # Interlocks and alarms
    interlock_activations: int = Field(0, ge=0, description="Interlock activations")
    alarm_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Alarm counts by type"
    )

    # Compliance
    safety_compliance_status: ComplianceStatus = Field(..., description="Safety compliance")
    regulatory_findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Regulatory findings"
    )


# =============================================================================
# Configuration
# =============================================================================

class ComplianceReporterConfig(BaseModel):
    """Configuration for ComplianceReporter."""

    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")
    regulatory_authority: str = Field("EPA", description="Regulatory authority")
    permit_number: Optional[str] = Field(None, description="Permit number")
    generator_id: str = Field("burnmaster-reporter-001", description="Generator system ID")
    default_export_format: ExportFormat = Field(ExportFormat.JSON, description="Default export format")


# =============================================================================
# ComplianceReporter Implementation
# =============================================================================

class ComplianceReporter:
    """
    ComplianceReporter implementation for BURNMASTER.

    This class generates regulatory compliance reports including emissions,
    permit compliance, optimization audit, and safety reports.

    Attributes:
        config: Reporter configuration
        _reports: Storage for generated reports

    Example:
        >>> config = ComplianceReporterConfig(facility_id="FAC-001", facility_name="Plant A")
        >>> reporter = ComplianceReporter(config)
        >>> report = reporter.generate_emissions_report(period)
    """

    def __init__(self, config: ComplianceReporterConfig):
        """
        Initialize ComplianceReporter.

        Args:
            config: Reporter configuration
        """
        self.config = config
        self._reports: Dict[str, ComplianceReport] = {}

        logger.info(
            f"ComplianceReporter initialized for facility {config.facility_id} "
            f"({config.facility_name})"
        )

    def generate_emissions_report(
        self,
        period: DateRange,
        emissions_data: Optional[List[EmissionsDataPoint]] = None,
        permit_limits: Optional[List[PermitLimit]] = None
    ) -> EmissionsReport:
        """
        Generate an emissions compliance report.

        Args:
            period: Reporting period
            emissions_data: Emissions measurements
            permit_limits: Permit limits for comparison

        Returns:
            Emissions compliance report

        Raises:
            ValueError: If period is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            report_id = str(uuid.uuid4())

            # Calculate emissions summary
            emissions_summary = self._calculate_emissions_summary(
                emissions_data or [],
                period
            )

            # Calculate emissions by boiler
            emissions_by_boiler = self._calculate_emissions_by_boiler(
                emissions_data or []
            )

            # Calculate daily emissions
            daily_emissions = self._calculate_daily_emissions(
                emissions_data or [],
                period
            )

            # Evaluate permit compliance
            evaluated_limits = self._evaluate_permit_limits(
                emissions_summary,
                permit_limits or []
            )

            # Determine overall compliance
            overall_compliance = self._determine_overall_compliance(evaluated_limits)

            # Identify exceedances
            exceedances = [
                limit.dict() for limit in evaluated_limits
                if limit.compliance_status == ComplianceStatus.NON_COMPLIANT
            ]

            # Calculate data completeness
            expected_measurements = period.duration_days * 24  # Hourly data
            actual_measurements = len(emissions_data or [])
            data_completeness = min(100.0, (actual_measurements / expected_measurements) * 100) if expected_measurements > 0 else 0.0

            # Create report content for hashing
            report_content = {
                "report_id": report_id,
                "period": period.dict(),
                "emissions_summary": emissions_summary.dict(),
                "permit_limits": [l.dict() for l in evaluated_limits]
            }
            report_hash = hashlib.sha256(
                json.dumps(report_content, sort_keys=True, default=str).encode()
            ).hexdigest()

            report = EmissionsReport(
                report_id=report_id,
                report_type=ReportType.EMISSIONS,
                title=f"Emissions Compliance Report - {period.start_date.strftime('%Y-%m')}",
                period=period,
                generated_at=start_time,
                generated_by=self.config.generator_id,
                status=ReportStatus.DRAFT,
                report_hash=report_hash,
                facility_id=self.config.facility_id,
                facility_name=self.config.facility_name,
                regulatory_authority=self.config.regulatory_authority,
                permit_number=self.config.permit_number,
                emissions_summary=emissions_summary,
                emissions_by_boiler=emissions_by_boiler,
                daily_emissions=daily_emissions,
                permit_limits=evaluated_limits,
                overall_compliance=overall_compliance,
                exceedances=exceedances,
                data_completeness_percent=data_completeness,
                measurement_methodology="Continuous Emissions Monitoring System (CEMS)"
            )

            self._reports[report_id] = report

            logger.info(
                f"Generated emissions report {report_id}: "
                f"compliance={overall_compliance.value}, exceedances={len(exceedances)}"
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate emissions report: {str(e)}", exc_info=True)
            raise

    def generate_permit_compliance_report(
        self,
        period: DateRange,
        permit_conditions: Optional[List[Dict[str, Any]]] = None
    ) -> PermitReport:
        """
        Generate a permit compliance report.

        Args:
            period: Reporting period
            permit_conditions: Permit conditions to evaluate

        Returns:
            Permit compliance report
        """
        start_time = datetime.now(timezone.utc)

        try:
            report_id = str(uuid.uuid4())

            # Evaluate each permit condition
            conditions = permit_conditions or []
            condition_compliance = self._evaluate_permit_conditions(conditions)

            # Calculate compliance metrics
            conditions_met = sum(
                1 for c in condition_compliance
                if c.compliance_status == ComplianceStatus.COMPLIANT
            )
            conditions_total = len(condition_compliance)
            compliance_percentage = (
                (conditions_met / conditions_total * 100) if conditions_total > 0 else 100.0
            )

            # Determine overall compliance
            if compliance_percentage >= 100:
                overall_compliance = ComplianceStatus.COMPLIANT
            elif compliance_percentage >= 90:
                overall_compliance = ComplianceStatus.WARNING
            else:
                overall_compliance = ComplianceStatus.NON_COMPLIANT

            # Identify deviations
            deviations = [
                c.dict() for c in condition_compliance
                if c.compliance_status != ComplianceStatus.COMPLIANT
            ]

            # Create report hash
            report_content = {
                "report_id": report_id,
                "period": period.dict(),
                "conditions_met": conditions_met,
                "conditions_total": conditions_total
            }
            report_hash = hashlib.sha256(
                json.dumps(report_content, sort_keys=True, default=str).encode()
            ).hexdigest()

            report = PermitReport(
                report_id=report_id,
                report_type=ReportType.PERMIT,
                title=f"Permit Compliance Report - {period.start_date.strftime('%Y-%m')}",
                period=period,
                generated_at=start_time,
                generated_by=self.config.generator_id,
                status=ReportStatus.DRAFT,
                report_hash=report_hash,
                facility_id=self.config.facility_id,
                facility_name=self.config.facility_name,
                regulatory_authority=self.config.regulatory_authority,
                permit_number=self.config.permit_number,
                permit_conditions=conditions,
                condition_compliance=condition_compliance,
                overall_compliance=overall_compliance,
                conditions_met=conditions_met,
                conditions_total=conditions_total,
                compliance_percentage=compliance_percentage,
                deviations=deviations,
                corrective_actions=[]
            )

            self._reports[report_id] = report

            logger.info(
                f"Generated permit report {report_id}: "
                f"{conditions_met}/{conditions_total} conditions met"
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate permit report: {str(e)}", exc_info=True)
            raise

    def generate_optimization_audit_report(
        self,
        period: DateRange,
        optimization_events: Optional[List[OptimizationEvent]] = None
    ) -> OptAuditReport:
        """
        Generate an optimization audit report.

        Args:
            period: Reporting period
            optimization_events: Optimization events to include

        Returns:
            Optimization audit report
        """
        start_time = datetime.now(timezone.utc)

        try:
            report_id = str(uuid.uuid4())

            events = optimization_events or []

            # Calculate optimization metrics
            total_optimizations = len(events)
            successful = [e for e in events if e.constraints_satisfied]
            successful_optimizations = len(successful)
            success_rate = (
                (successful_optimizations / total_optimizations * 100)
                if total_optimizations > 0 else 100.0
            )

            # Calculate average improvements
            efficiency_improvements = []
            emissions_reductions = []
            fuel_savings = []

            for event in successful:
                if 'efficiency' in event.improvement_achieved:
                    efficiency_improvements.append(event.improvement_achieved['efficiency'])
                if 'emissions' in event.improvement_achieved:
                    emissions_reductions.append(event.improvement_achieved['emissions'])
                if 'fuel' in event.improvement_achieved:
                    fuel_savings.append(event.improvement_achieved['fuel'])

            avg_efficiency = sum(efficiency_improvements) / len(efficiency_improvements) if efficiency_improvements else 0.0
            avg_emissions = sum(emissions_reductions) / len(emissions_reductions) if emissions_reductions else 0.0
            avg_fuel = sum(fuel_savings) / len(fuel_savings) if fuel_savings else 0.0

            # Get unique model versions
            model_versions = list(set(e.model_used for e in events))

            # Count constraint violations
            constraint_violations = sum(1 for e in events if not e.constraints_satisfied)
            constraint_compliance = (
                ((total_optimizations - constraint_violations) / total_optimizations * 100)
                if total_optimizations > 0 else 100.0
            )

            # Create report hash
            report_content = {
                "report_id": report_id,
                "period": period.dict(),
                "total_optimizations": total_optimizations,
                "success_rate": success_rate
            }
            report_hash = hashlib.sha256(
                json.dumps(report_content, sort_keys=True, default=str).encode()
            ).hexdigest()

            report = OptAuditReport(
                report_id=report_id,
                report_type=ReportType.OPTIMIZATION_AUDIT,
                title=f"Optimization Audit Report - {period.start_date.strftime('%Y-%m')}",
                period=period,
                generated_at=start_time,
                generated_by=self.config.generator_id,
                status=ReportStatus.DRAFT,
                report_hash=report_hash,
                facility_id=self.config.facility_id,
                facility_name=self.config.facility_name,
                regulatory_authority=self.config.regulatory_authority,
                permit_number=self.config.permit_number,
                total_optimizations=total_optimizations,
                successful_optimizations=successful_optimizations,
                optimization_success_rate=success_rate,
                efficiency_improvement_percent=avg_efficiency,
                emissions_reduction_percent=avg_emissions,
                fuel_savings_percent=avg_fuel,
                optimization_events=events,
                model_accuracy_metrics={},
                model_versions_used=model_versions,
                constraint_violations=constraint_violations,
                constraint_compliance_rate=constraint_compliance
            )

            self._reports[report_id] = report

            logger.info(
                f"Generated optimization audit report {report_id}: "
                f"{successful_optimizations}/{total_optimizations} successful"
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate optimization audit report: {str(e)}", exc_info=True)
            raise

    def generate_safety_compliance_report(
        self,
        period: DateRange,
        incidents: Optional[List[SafetyIncident]] = None,
        alarm_data: Optional[Dict[str, int]] = None
    ) -> SafetyReport:
        """
        Generate a safety compliance report.

        Args:
            period: Reporting period
            incidents: Safety incidents
            alarm_data: Alarm counts by type

        Returns:
            Safety compliance report
        """
        start_time = datetime.now(timezone.utc)

        try:
            report_id = str(uuid.uuid4())

            incident_list = incidents or []
            alarms = alarm_data or {}

            # Calculate incident metrics
            total_incidents = len(incident_list)

            # Group by severity
            incidents_by_severity: Dict[str, int] = {}
            for incident in incident_list:
                severity = incident.severity
                incidents_by_severity[severity] = incidents_by_severity.get(severity, 0) + 1

            # Group by type
            incidents_by_type: Dict[str, int] = {}
            for incident in incident_list:
                inc_type = incident.incident_type
                incidents_by_type[inc_type] = incidents_by_type.get(inc_type, 0) + 1

            # Calculate days without incident
            if incident_list:
                last_incident = max(incident_list, key=lambda i: i.timestamp)
                days_since = (start_time - last_incident.timestamp).days
            else:
                days_since = period.duration_days

            # Calculate safety score (simple formula)
            critical_incidents = incidents_by_severity.get('critical', 0)
            major_incidents = incidents_by_severity.get('major', 0)
            minor_incidents = incidents_by_severity.get('minor', 0)

            # Score: 100 - (10 * critical) - (5 * major) - (1 * minor)
            safety_score = max(0, 100 - (10 * critical_incidents) - (5 * major_incidents) - (1 * minor_incidents))

            # Determine compliance status
            if critical_incidents > 0:
                safety_compliance = ComplianceStatus.NON_COMPLIANT
            elif major_incidents > 2:
                safety_compliance = ComplianceStatus.WARNING
            else:
                safety_compliance = ComplianceStatus.COMPLIANT

            # Create report hash
            report_content = {
                "report_id": report_id,
                "period": period.dict(),
                "total_incidents": total_incidents,
                "safety_score": safety_score
            }
            report_hash = hashlib.sha256(
                json.dumps(report_content, sort_keys=True, default=str).encode()
            ).hexdigest()

            report = SafetyReport(
                report_id=report_id,
                report_type=ReportType.SAFETY,
                title=f"Safety Compliance Report - {period.start_date.strftime('%Y-%m')}",
                period=period,
                generated_at=start_time,
                generated_by=self.config.generator_id,
                status=ReportStatus.DRAFT,
                report_hash=report_hash,
                facility_id=self.config.facility_id,
                facility_name=self.config.facility_name,
                regulatory_authority=self.config.regulatory_authority,
                permit_number=self.config.permit_number,
                total_incidents=total_incidents,
                incidents_by_severity=incidents_by_severity,
                incidents_by_type=incidents_by_type,
                days_without_incident=days_since,
                safety_score=safety_score,
                near_misses=0,
                incidents=incident_list,
                interlock_activations=0,
                alarm_counts=alarms,
                safety_compliance_status=safety_compliance,
                regulatory_findings=[]
            )

            self._reports[report_id] = report

            logger.info(
                f"Generated safety report {report_id}: "
                f"{total_incidents} incidents, score={safety_score}"
            )

            return report

        except Exception as e:
            logger.error(f"Failed to generate safety report: {str(e)}", exc_info=True)
            raise

    def export_report(
        self,
        report: ComplianceReport,
        format: ExportFormat = ExportFormat.JSON
    ) -> bytes:
        """
        Export a compliance report.

        Args:
            report: Report to export
            format: Export format

        Returns:
            Exported report as bytes

        Raises:
            ValueError: If format not supported
        """
        start_time = datetime.now(timezone.utc)

        try:
            if format == ExportFormat.JSON:
                export_data = report.dict()
                export_data['exported_at'] = start_time.isoformat()
                result = json.dumps(export_data, indent=2, default=str).encode('utf-8')

            elif format == ExportFormat.CSV:
                result = self._export_to_csv(report)

            elif format == ExportFormat.XML:
                result = self._export_to_xml(report)

            elif format == ExportFormat.PDF:
                # PDF export would require additional library
                raise NotImplementedError("PDF export not yet implemented")

            elif format == ExportFormat.EXCEL:
                # Excel export would require additional library
                raise NotImplementedError("Excel export not yet implemented")

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(
                f"Exported report {report.report_id} as {format.value}: "
                f"{len(result)} bytes"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to export report: {str(e)}", exc_info=True)
            raise

    def get_report(self, report_id: str) -> Optional[ComplianceReport]:
        """Get a report by ID."""
        return self._reports.get(report_id)

    def _calculate_emissions_summary(
        self,
        data: List[EmissionsDataPoint],
        period: DateRange
    ) -> EmissionsSummary:
        """Calculate emissions summary from data points."""
        if not data:
            return EmissionsSummary(
                total_nox_kg=0,
                total_co_kg=0,
                total_so2_kg=0,
                total_particulate_kg=0,
                total_co2_tonnes=0,
                avg_nox_ppm=0,
                avg_co_ppm=0,
                max_nox_ppm=0,
                max_co_ppm=0,
                measurement_count=0,
                operating_hours=0
            )

        # Calculate totals (simplified - actual would use mass flow calculations)
        total_nox = sum(d.nox_ppm * d.flue_gas_flow_m3_h / 1000000 for d in data)
        total_co = sum(d.co_ppm * d.flue_gas_flow_m3_h / 1000000 for d in data)
        total_so2 = sum(d.so2_ppm * d.flue_gas_flow_m3_h / 1000000 for d in data)
        total_particulate = sum(d.particulate_mg_m3 * d.flue_gas_flow_m3_h / 1000000 for d in data)
        total_co2 = sum(d.co2_percent * d.flue_gas_flow_m3_h / 100 for d in data) / 1000

        return EmissionsSummary(
            total_nox_kg=total_nox,
            total_co_kg=total_co,
            total_so2_kg=total_so2,
            total_particulate_kg=total_particulate,
            total_co2_tonnes=total_co2,
            avg_nox_ppm=sum(d.nox_ppm for d in data) / len(data),
            avg_co_ppm=sum(d.co_ppm for d in data) / len(data),
            max_nox_ppm=max(d.nox_ppm for d in data),
            max_co_ppm=max(d.co_ppm for d in data),
            measurement_count=len(data),
            operating_hours=len(data)  # Assuming hourly data
        )

    def _calculate_emissions_by_boiler(
        self,
        data: List[EmissionsDataPoint]
    ) -> Dict[str, EmissionsSummary]:
        """Calculate emissions grouped by boiler."""
        boilers: Dict[str, List[EmissionsDataPoint]] = {}
        for d in data:
            if d.boiler_id not in boilers:
                boilers[d.boiler_id] = []
            boilers[d.boiler_id].append(d)

        return {
            boiler_id: self._calculate_emissions_summary(
                points,
                DateRange(
                    start_date=min(p.timestamp for p in points),
                    end_date=max(p.timestamp for p in points)
                )
            )
            for boiler_id, points in boilers.items()
        }

    def _calculate_daily_emissions(
        self,
        data: List[EmissionsDataPoint],
        period: DateRange
    ) -> List[Dict[str, Any]]:
        """Calculate daily emissions totals."""
        daily: Dict[date, List[EmissionsDataPoint]] = {}
        for d in data:
            day = d.timestamp.date()
            if day not in daily:
                daily[day] = []
            daily[day].append(d)

        return [
            {
                "date": day.isoformat(),
                "nox_ppm_avg": sum(p.nox_ppm for p in points) / len(points),
                "co_ppm_avg": sum(p.co_ppm for p in points) / len(points),
                "measurements": len(points)
            }
            for day, points in sorted(daily.items())
        ]

    def _evaluate_permit_limits(
        self,
        summary: EmissionsSummary,
        limits: List[PermitLimit]
    ) -> List[PermitLimit]:
        """Evaluate compliance with permit limits."""
        evaluated = []
        for limit in limits:
            actual = 0.0
            if limit.parameter == "nox_ppm":
                actual = summary.avg_nox_ppm
            elif limit.parameter == "co_ppm":
                actual = summary.avg_co_ppm
            elif limit.parameter == "max_nox_ppm":
                actual = summary.max_nox_ppm
            elif limit.parameter == "max_co_ppm":
                actual = summary.max_co_ppm

            margin = ((limit.limit_value - actual) / limit.limit_value * 100) if limit.limit_value > 0 else 0

            if actual <= limit.limit_value:
                status = ComplianceStatus.COMPLIANT
            else:
                status = ComplianceStatus.NON_COMPLIANT

            evaluated.append(PermitLimit(
                parameter=limit.parameter,
                limit_value=limit.limit_value,
                unit=limit.unit,
                averaging_period=limit.averaging_period,
                actual_value=actual,
                compliance_status=status,
                margin_percent=margin
            ))

        return evaluated

    def _evaluate_permit_conditions(
        self,
        conditions: List[Dict[str, Any]]
    ) -> List[PermitLimit]:
        """Evaluate permit conditions."""
        # Simplified - would need actual condition evaluation logic
        return [
            PermitLimit(
                parameter=c.get('parameter', 'unknown'),
                limit_value=c.get('limit', 0),
                unit=c.get('unit', ''),
                averaging_period=c.get('period', ''),
                actual_value=c.get('actual', 0),
                compliance_status=ComplianceStatus.COMPLIANT,
                margin_percent=10.0
            )
            for c in conditions
        ]

    def _determine_overall_compliance(
        self,
        limits: List[PermitLimit]
    ) -> ComplianceStatus:
        """Determine overall compliance from individual limits."""
        if not limits:
            return ComplianceStatus.UNKNOWN

        non_compliant = sum(
            1 for l in limits
            if l.compliance_status == ComplianceStatus.NON_COMPLIANT
        )
        warnings = sum(
            1 for l in limits
            if l.compliance_status == ComplianceStatus.WARNING
        )

        if non_compliant > 0:
            return ComplianceStatus.NON_COMPLIANT
        elif warnings > 0:
            return ComplianceStatus.WARNING
        else:
            return ComplianceStatus.COMPLIANT

    def _export_to_csv(self, report: ComplianceReport) -> bytes:
        """Export report to CSV format."""
        lines = []
        lines.append("Report ID,Type,Period Start,Period End,Generated At,Status")
        lines.append(
            f"{report.report_id},{report.report_type.value},"
            f"{report.period.start_date.isoformat()},{report.period.end_date.isoformat()},"
            f"{report.generated_at.isoformat()},{report.status.value}"
        )
        return '\n'.join(lines).encode('utf-8')

    def _export_to_xml(self, report: ComplianceReport) -> bytes:
        """Export report to XML format."""
        def dict_to_xml(d: Any, name: str) -> str:
            if isinstance(d, dict):
                inner = ''.join(dict_to_xml(v, k) for k, v in d.items())
                return f"<{name}>{inner}</{name}>"
            elif isinstance(d, list):
                inner = ''.join(dict_to_xml(item, 'item') for item in d)
                return f"<{name}>{inner}</{name}>"
            else:
                escaped = str(d).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return f"<{name}>{escaped}</{name}>"

        xml_content = dict_to_xml(report.dict(), 'compliance_report')
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_content}'.encode('utf-8')
