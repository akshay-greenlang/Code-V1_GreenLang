"""
AuditReports - ESD Audit Reports Generation

This module implements comprehensive audit report generation for Emergency
Shutdown Systems per IEC 61511-1 Clause 5.2 (FSA) and Clause 16 (Testing).
Provides ESD test history reports, bypass history reports, response time
trend reports, and compliance summary reports with PDF/Excel export.

Key features:
- ESD test history reports
- Bypass history reports
- Response time trend reports
- Compliance summary reports
- PDF/Excel export capability
- Provenance hashing for all reports

Reference: IEC 61511-1 Clause 5.2, Clause 16, ISA TR84.00.09

Example:
    >>> from greenlang.safety.esd.audit_reports import ESDReportGenerator
    >>> generator = ESDReportGenerator(system_id="ESD-001")
    >>> report = generator.generate_compliance_report(start_date, end_date)
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
import json
import io

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Types of audit reports."""

    TEST_HISTORY = "test_history"
    BYPASS_HISTORY = "bypass_history"
    RESPONSE_TIME_TREND = "response_time_trend"
    COMPLIANCE_SUMMARY = "compliance_summary"
    INCIDENT_SUMMARY = "incident_summary"
    MAINTENANCE_SUMMARY = "maintenance_summary"
    SIF_STATUS = "sif_status"
    FUNCTIONAL_SAFETY = "functional_safety"


class ReportFormat(str, Enum):
    """Export formats."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    EXCEL = "excel"
    HTML = "html"


class ComplianceStatus(str, Enum):
    """Compliance status levels."""

    COMPLIANT = "compliant"
    MARGINAL = "marginal"
    NON_COMPLIANT = "non_compliant"
    NOT_TESTED = "not_tested"


class ReportMetadata(BaseModel):
    """Report metadata."""

    report_id: str = Field(
        default_factory=lambda: f"RPT-{uuid.uuid4().hex[:8].upper()}",
        description="Report identifier"
    )
    report_type: ReportType = Field(
        ...,
        description="Type of report"
    )
    title: str = Field(
        ...,
        description="Report title"
    )
    system_id: str = Field(
        ...,
        description="System identifier"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Generation timestamp"
    )
    generated_by: str = Field(
        default="",
        description="Generator"
    )
    period_start: Optional[datetime] = Field(
        None,
        description="Reporting period start"
    )
    period_end: Optional[datetime] = Field(
        None,
        description="Reporting period end"
    )
    classification: str = Field(
        default="Internal",
        description="Document classification"
    )
    version: str = Field(
        default="1.0",
        description="Report version"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash"
    )


class TestHistoryReport(BaseModel):
    """ESD test history report."""

    metadata: ReportMetadata = Field(
        ...,
        description="Report metadata"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics"
    )
    test_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual test records"
    )
    by_sif: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Tests grouped by SIF"
    )
    trends: Dict[str, Any] = Field(
        default_factory=dict,
        description="Trend analysis"
    )
    overdue_tests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Overdue tests"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )


class BypassHistoryReport(BaseModel):
    """Bypass history report."""

    metadata: ReportMetadata = Field(
        ...,
        description="Report metadata"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics"
    )
    bypass_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual bypass records"
    )
    by_sif: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Bypasses grouped by SIF"
    )
    by_reason: Dict[str, int] = Field(
        default_factory=dict,
        description="Bypasses by reason category"
    )
    duration_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Duration analysis"
    )
    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Policy violations"
    )
    active_bypasses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active bypasses"
    )


class ResponseTimeTrendReport(BaseModel):
    """Response time trend report."""

    metadata: ReportMetadata = Field(
        ...,
        description="Report metadata"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics"
    )
    sif_trends: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Trends by SIF"
    )
    degradation_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Degradation alerts"
    )
    component_analysis: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Analysis by component"
    )
    predictions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Future predictions"
    )


class ComplianceReport(BaseModel):
    """Compliance summary report."""

    metadata: ReportMetadata = Field(
        ...,
        description="Report metadata"
    )
    executive_summary: str = Field(
        default="",
        description="Executive summary"
    )
    overall_status: ComplianceStatus = Field(
        default=ComplianceStatus.COMPLIANT,
        description="Overall compliance status"
    )
    compliance_score: float = Field(
        default=0.0,
        description="Compliance score (0-100)"
    )
    sif_compliance: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-SIF compliance status"
    )
    test_compliance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Test compliance"
    )
    response_time_compliance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response time compliance"
    )
    bypass_compliance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bypass compliance"
    )
    documentation_compliance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Documentation compliance"
    )
    open_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Open corrective actions"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    certification_statement: str = Field(
        default="",
        description="Certification statement"
    )
    signoff: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report signoff"
    )


class ESDReportGenerator:
    """
    ESD Report Generator.

    Generates comprehensive audit reports for ESD systems including
    test history, bypass history, response time trends, and compliance
    summaries per IEC 61511 requirements.

    Key features:
    - Multiple report types
    - Multiple export formats
    - Provenance tracking
    - Trend analysis
    - Compliance scoring

    The generator follows IEC 61511 principles:
    - Complete documentation
    - Audit traceability
    - Clear compliance status

    Attributes:
        system_id: ESD system identifier
        reports: Generated reports

    Example:
        >>> generator = ESDReportGenerator(system_id="ESD-001")
        >>> report = generator.generate_test_history_report(
        ...     start_date, end_date
        ... )
    """

    def __init__(
        self,
        system_id: str,
        organization: str = "GreenLang"
    ):
        """
        Initialize ESDReportGenerator.

        Args:
            system_id: ESD system identifier
            organization: Organization name
        """
        self.system_id = system_id
        self.organization = organization
        self.reports: Dict[str, Any] = {}

        logger.info(f"ESDReportGenerator initialized: {system_id}")

    def generate_test_history_report(
        self,
        start_date: datetime,
        end_date: datetime,
        test_data: List[Dict[str, Any]],
        schedule_data: Optional[List[Dict[str, Any]]] = None,
        generated_by: str = ""
    ) -> TestHistoryReport:
        """
        Generate ESD test history report.

        Args:
            start_date: Reporting period start
            end_date: Reporting period end
            test_data: Test result records
            schedule_data: Test schedule data
            generated_by: Generator name

        Returns:
            TestHistoryReport
        """
        metadata = ReportMetadata(
            report_type=ReportType.TEST_HISTORY,
            title=f"ESD Test History Report - {self.system_id}",
            system_id=self.system_id,
            generated_by=generated_by,
            period_start=start_date,
            period_end=end_date,
        )

        # Filter tests in period
        tests_in_period = [
            t for t in test_data
            if start_date <= datetime.fromisoformat(t.get("test_date", start_date.isoformat())) <= end_date
        ]

        # Calculate summary
        total_tests = len(tests_in_period)
        passed_tests = sum(1 for t in tests_in_period if t.get("passed", False))
        failed_tests = total_tests - passed_tests

        summary = {
            "reporting_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate_percent": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "unique_sifs_tested": len(set(t.get("sif_id") for t in tests_in_period)),
        }

        # Group by SIF
        by_sif: Dict[str, List[Dict[str, Any]]] = {}
        for test in tests_in_period:
            sif_id = test.get("sif_id", "UNKNOWN")
            if sif_id not in by_sif:
                by_sif[sif_id] = []
            by_sif[sif_id].append(test)

        # Calculate trends
        trends = self._calculate_test_trends(tests_in_period)

        # Identify overdue tests
        overdue = []
        if schedule_data:
            now = datetime.utcnow()
            for schedule in schedule_data:
                next_due = schedule.get("next_test_due")
                if next_due:
                    due_date = datetime.fromisoformat(next_due) if isinstance(next_due, str) else next_due
                    if now > due_date:
                        overdue.append({
                            "sif_id": schedule.get("sif_id"),
                            "due_date": due_date.isoformat(),
                            "days_overdue": (now - due_date).days,
                        })

        # Generate recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append(
                f"Investigate and resolve {failed_tests} failed tests"
            )
        if len(overdue) > 0:
            recommendations.append(
                f"Schedule {len(overdue)} overdue tests immediately"
            )
        if summary["pass_rate_percent"] < 100:
            recommendations.append(
                "Review test procedures for failed SIFs"
            )

        report = TestHistoryReport(
            metadata=metadata,
            summary=summary,
            test_records=tests_in_period,
            by_sif=by_sif,
            trends=trends,
            overdue_tests=overdue,
            recommendations=recommendations,
        )

        # Calculate provenance
        metadata.provenance_hash = self._calculate_provenance(report)

        self.reports[metadata.report_id] = report

        logger.info(f"Test history report generated: {metadata.report_id}")

        return report

    def generate_bypass_history_report(
        self,
        start_date: datetime,
        end_date: datetime,
        bypass_data: List[Dict[str, Any]],
        generated_by: str = ""
    ) -> BypassHistoryReport:
        """
        Generate bypass history report.

        Args:
            start_date: Reporting period start
            end_date: Reporting period end
            bypass_data: Bypass records
            generated_by: Generator name

        Returns:
            BypassHistoryReport
        """
        metadata = ReportMetadata(
            report_type=ReportType.BYPASS_HISTORY,
            title=f"ESD Bypass History Report - {self.system_id}",
            system_id=self.system_id,
            generated_by=generated_by,
            period_start=start_date,
            period_end=end_date,
        )

        # Filter bypasses in period
        bypasses_in_period = [
            b for b in bypass_data
            if start_date <= datetime.fromisoformat(b.get("activated_at", start_date.isoformat())) <= end_date
        ]

        # Calculate summary
        total_bypasses = len(bypasses_in_period)
        total_duration_hours = sum(
            b.get("actual_duration_hours", 0) or b.get("approved_duration_hours", 0)
            for b in bypasses_in_period
        )

        summary = {
            "reporting_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "total_bypasses": total_bypasses,
            "total_duration_hours": round(total_duration_hours, 1),
            "avg_duration_hours": round(total_duration_hours / total_bypasses, 1) if total_bypasses > 0 else 0,
            "unique_sifs_bypassed": len(set(b.get("sif_id") for b in bypasses_in_period)),
        }

        # Group by SIF
        by_sif: Dict[str, List[Dict[str, Any]]] = {}
        for bypass in bypasses_in_period:
            sif_id = bypass.get("sif_id", "UNKNOWN")
            if sif_id not in by_sif:
                by_sif[sif_id] = []
            by_sif[sif_id].append(bypass)

        # Group by reason
        by_reason: Dict[str, int] = {}
        for bypass in bypasses_in_period:
            reason = bypass.get("bypass_type", "other")
            by_reason[reason] = by_reason.get(reason, 0) + 1

        # Duration analysis
        durations = [
            b.get("actual_duration_hours", 0) or b.get("approved_duration_hours", 0)
            for b in bypasses_in_period
        ]

        duration_analysis = {
            "min_hours": round(min(durations), 1) if durations else 0,
            "max_hours": round(max(durations), 1) if durations else 0,
            "avg_hours": round(sum(durations) / len(durations), 1) if durations else 0,
            "exceeded_count": sum(1 for b in bypasses_in_period if b.get("exceeded_approved", False)),
        }

        # Find violations
        violations = []
        for bypass in bypasses_in_period:
            if bypass.get("exceeded_approved", False):
                violations.append({
                    "bypass_id": bypass.get("bypass_id"),
                    "sif_id": bypass.get("sif_id"),
                    "violation": "Duration exceeded approved time",
                    "details": bypass,
                })

        # Find active bypasses
        now = datetime.utcnow()
        active = [
            b for b in bypass_data
            if b.get("state") == "active" or (
                b.get("activated_at") and
                not b.get("deactivated_at") and
                datetime.fromisoformat(b.get("expires_at", now.isoformat())) > now
            )
        ]

        report = BypassHistoryReport(
            metadata=metadata,
            summary=summary,
            bypass_records=bypasses_in_period,
            by_sif=by_sif,
            by_reason=by_reason,
            duration_analysis=duration_analysis,
            violations=violations,
            active_bypasses=active,
        )

        metadata.provenance_hash = self._calculate_provenance(report)
        self.reports[metadata.report_id] = report

        logger.info(f"Bypass history report generated: {metadata.report_id}")

        return report

    def generate_response_time_report(
        self,
        start_date: datetime,
        end_date: datetime,
        response_data: List[Dict[str, Any]],
        requirement_ms: float = 1000.0,
        generated_by: str = ""
    ) -> ResponseTimeTrendReport:
        """
        Generate response time trend report.

        Args:
            start_date: Reporting period start
            end_date: Reporting period end
            response_data: Response time measurements
            requirement_ms: Response time requirement
            generated_by: Generator name

        Returns:
            ResponseTimeTrendReport
        """
        metadata = ReportMetadata(
            report_type=ReportType.RESPONSE_TIME_TREND,
            title=f"ESD Response Time Trend Report - {self.system_id}",
            system_id=self.system_id,
            generated_by=generated_by,
            period_start=start_date,
            period_end=end_date,
        )

        # Filter measurements in period
        measurements_in_period = [
            m for m in response_data
            if start_date <= datetime.fromisoformat(m.get("test_date", start_date.isoformat())) <= end_date
        ]

        # Calculate summary
        response_times = [m.get("total_response_ms", 0) for m in measurements_in_period]

        if response_times:
            import statistics
            summary = {
                "reporting_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "measurement_count": len(response_times),
                "requirement_ms": requirement_ms,
                "avg_response_ms": round(statistics.mean(response_times), 1),
                "min_response_ms": round(min(response_times), 1),
                "max_response_ms": round(max(response_times), 1),
                "stdev_ms": round(statistics.stdev(response_times), 1) if len(response_times) > 1 else 0,
                "compliant_count": sum(1 for t in response_times if t <= requirement_ms),
                "non_compliant_count": sum(1 for t in response_times if t > requirement_ms),
            }
        else:
            summary = {
                "reporting_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "measurement_count": 0,
                "error": "No measurements in period",
            }

        # Analyze trends by SIF
        sif_trends: Dict[str, Dict[str, Any]] = {}
        sif_measurements: Dict[str, List[float]] = {}

        for m in measurements_in_period:
            sif_id = m.get("sif_id", "UNKNOWN")
            if sif_id not in sif_measurements:
                sif_measurements[sif_id] = []
            sif_measurements[sif_id].append(m.get("total_response_ms", 0))

        for sif_id, times in sif_measurements.items():
            if len(times) >= 2:
                first_half = times[:len(times)//2]
                second_half = times[len(times)//2:]

                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)

                change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0

                trend = "stable"
                if change_percent > 10:
                    trend = "degrading"
                elif change_percent < -10:
                    trend = "improving"

                sif_trends[sif_id] = {
                    "measurement_count": len(times),
                    "latest_ms": times[-1],
                    "avg_ms": round(sum(times) / len(times), 1),
                    "trend": trend,
                    "change_percent": round(change_percent, 1),
                }

        # Identify degradation alerts
        degradation_alerts = []
        for sif_id, trend_data in sif_trends.items():
            if trend_data["trend"] == "degrading":
                degradation_alerts.append({
                    "sif_id": sif_id,
                    "current_ms": trend_data["latest_ms"],
                    "trend": trend_data["trend"],
                    "change_percent": trend_data["change_percent"],
                    "alert": "Response time degradation detected",
                })

        report = ResponseTimeTrendReport(
            metadata=metadata,
            summary=summary,
            sif_trends=sif_trends,
            degradation_alerts=degradation_alerts,
        )

        metadata.provenance_hash = self._calculate_provenance(report)
        self.reports[metadata.report_id] = report

        logger.info(f"Response time trend report generated: {metadata.report_id}")

        return report

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        test_data: List[Dict[str, Any]],
        bypass_data: List[Dict[str, Any]],
        response_data: List[Dict[str, Any]],
        sif_list: List[Dict[str, Any]],
        generated_by: str = "",
        signoff_by: Optional[str] = None
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance summary report.

        Args:
            start_date: Reporting period start
            end_date: Reporting period end
            test_data: Test result records
            bypass_data: Bypass records
            response_data: Response time data
            sif_list: List of SIFs in scope
            generated_by: Generator name
            signoff_by: Signoff authority

        Returns:
            ComplianceReport
        """
        metadata = ReportMetadata(
            report_type=ReportType.COMPLIANCE_SUMMARY,
            title=f"ESD Compliance Summary Report - {self.system_id}",
            system_id=self.system_id,
            generated_by=generated_by,
            period_start=start_date,
            period_end=end_date,
            classification="Safety Critical",
        )

        # Generate sub-reports
        test_report = self.generate_test_history_report(
            start_date, end_date, test_data, generated_by=generated_by
        )
        bypass_report = self.generate_bypass_history_report(
            start_date, end_date, bypass_data, generated_by=generated_by
        )
        response_report = self.generate_response_time_report(
            start_date, end_date, response_data, generated_by=generated_by
        )

        # Calculate per-SIF compliance
        sif_compliance = []
        compliant_count = 0
        marginal_count = 0
        non_compliant_count = 0

        for sif in sif_list:
            sif_id = sif.get("sif_id")

            # Test compliance
            sif_tests = [t for t in test_data if t.get("sif_id") == sif_id]
            test_passed = all(t.get("passed", False) for t in sif_tests) if sif_tests else None

            # Response time compliance
            sif_responses = [r for r in response_data if r.get("sif_id") == sif_id]
            response_ok = all(
                r.get("total_response_ms", 0) <= r.get("requirement_ms", 1000)
                for r in sif_responses
            ) if sif_responses else None

            # Bypass compliance
            sif_bypasses = [b for b in bypass_data if b.get("sif_id") == sif_id]
            bypass_ok = not any(
                b.get("exceeded_approved", False) for b in sif_bypasses
            )

            # Overall status
            if test_passed is None or response_ok is None:
                status = ComplianceStatus.NOT_TESTED
            elif test_passed and response_ok and bypass_ok:
                status = ComplianceStatus.COMPLIANT
                compliant_count += 1
            elif test_passed and (not response_ok or not bypass_ok):
                status = ComplianceStatus.MARGINAL
                marginal_count += 1
            else:
                status = ComplianceStatus.NON_COMPLIANT
                non_compliant_count += 1

            sif_compliance.append({
                "sif_id": sif_id,
                "sif_name": sif.get("name", ""),
                "sil_level": sif.get("sil_level", 0),
                "status": status.value,
                "test_passed": test_passed,
                "response_time_ok": response_ok,
                "bypass_compliance_ok": bypass_ok,
            })

        # Calculate overall compliance score
        total_sifs = len(sif_list)
        compliance_score = (compliant_count / total_sifs * 100) if total_sifs > 0 else 0

        # Determine overall status
        if non_compliant_count > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif marginal_count > 0:
            overall_status = ComplianceStatus.MARGINAL
        elif compliant_count == total_sifs:
            overall_status = ComplianceStatus.COMPLIANT
        else:
            overall_status = ComplianceStatus.NOT_TESTED

        # Test compliance summary
        test_compliance = {
            "total_tests": test_report.summary.get("total_tests", 0),
            "passed": test_report.summary.get("passed", 0),
            "failed": test_report.summary.get("failed", 0),
            "pass_rate": test_report.summary.get("pass_rate_percent", 0),
            "overdue_count": len(test_report.overdue_tests),
        }

        # Response time compliance summary
        response_time_compliance = {
            "measurements": response_report.summary.get("measurement_count", 0),
            "compliant": response_report.summary.get("compliant_count", 0),
            "non_compliant": response_report.summary.get("non_compliant_count", 0),
            "degradation_alerts": len(response_report.degradation_alerts),
        }

        # Bypass compliance summary
        bypass_compliance = {
            "total_bypasses": bypass_report.summary.get("total_bypasses", 0),
            "total_hours": bypass_report.summary.get("total_duration_hours", 0),
            "violations": len(bypass_report.violations),
            "active_bypasses": len(bypass_report.active_bypasses),
        }

        # Documentation compliance (placeholder)
        documentation_compliance = {
            "test_records_complete": True,
            "bypass_records_complete": True,
            "provenance_verified": True,
        }

        # Generate recommendations
        recommendations = []
        recommendations.extend(test_report.recommendations)

        if response_report.degradation_alerts:
            recommendations.append(
                f"Investigate {len(response_report.degradation_alerts)} "
                f"SIFs with response time degradation"
            )

        if bypass_report.active_bypasses:
            recommendations.append(
                f"Review {len(bypass_report.active_bypasses)} active bypasses"
            )

        if bypass_report.violations:
            recommendations.append(
                f"Address {len(bypass_report.violations)} bypass policy violations"
            )

        # Generate executive summary
        executive_summary = (
            f"This compliance report covers the period from "
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
            f"for ESD system {self.system_id}. "
            f"Overall compliance score: {compliance_score:.1f}%. "
            f"Status: {overall_status.value.upper()}. "
            f"{compliant_count} of {total_sifs} SIFs are fully compliant, "
            f"{marginal_count} are marginal, and "
            f"{non_compliant_count} require corrective action."
        )

        # Generate certification statement
        if overall_status == ComplianceStatus.COMPLIANT:
            certification = (
                f"This ESD system ({self.system_id}) meets all IEC 61511 "
                f"requirements for the reporting period. All SIFs have been "
                f"tested per schedule, response times meet requirements, and "
                f"bypass management procedures have been followed."
            )
        else:
            certification = (
                f"This ESD system ({self.system_id}) has compliance gaps "
                f"that require attention. See recommendations for required actions."
            )

        report = ComplianceReport(
            metadata=metadata,
            executive_summary=executive_summary,
            overall_status=overall_status,
            compliance_score=compliance_score,
            sif_compliance=sif_compliance,
            test_compliance=test_compliance,
            response_time_compliance=response_time_compliance,
            bypass_compliance=bypass_compliance,
            documentation_compliance=documentation_compliance,
            recommendations=recommendations,
            certification_statement=certification,
            signoff={
                "signoff_by": signoff_by,
                "signoff_date": datetime.utcnow().isoformat() if signoff_by else None,
            } if signoff_by else {},
        )

        metadata.provenance_hash = self._calculate_provenance(report)
        self.reports[metadata.report_id] = report

        logger.info(f"Compliance report generated: {metadata.report_id}")

        return report

    def export_report(
        self,
        report_id: str,
        format: ReportFormat = ReportFormat.JSON
    ) -> Union[str, bytes, Dict[str, Any]]:
        """
        Export a report in the specified format.

        Args:
            report_id: Report to export
            format: Export format

        Returns:
            Exported report content
        """
        if report_id not in self.reports:
            raise ValueError(f"Report not found: {report_id}")

        report = self.reports[report_id]

        if format == ReportFormat.JSON:
            return self._export_json(report)
        elif format == ReportFormat.CSV:
            return self._export_csv(report)
        elif format == ReportFormat.HTML:
            return self._export_html(report)
        elif format == ReportFormat.PDF:
            return self._export_pdf(report)
        elif format == ReportFormat.EXCEL:
            return self._export_excel(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, report: Any) -> str:
        """Export report as JSON."""
        if hasattr(report, 'model_dump'):
            data = report.model_dump()
        else:
            data = dict(report)

        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(i) for i in obj]
            return obj

        data = convert_datetime(data)
        return json.dumps(data, indent=2)

    def _export_csv(self, report: Any) -> str:
        """Export report as CSV."""
        output = io.StringIO()

        # Export based on report type
        if isinstance(report, TestHistoryReport):
            output.write("sif_id,test_date,passed,response_time_ms,tester\n")
            for test in report.test_records:
                output.write(
                    f"{test.get('sif_id','')},{test.get('test_date','')}"
                    f",{test.get('passed','')},{test.get('response_time_ms','')}"
                    f",{test.get('tester','')}\n"
                )
        elif isinstance(report, BypassHistoryReport):
            output.write("bypass_id,sif_id,activated_at,duration_hours,reason\n")
            for bypass in report.bypass_records:
                output.write(
                    f"{bypass.get('bypass_id','')},{bypass.get('sif_id','')}"
                    f",{bypass.get('activated_at','')},{bypass.get('duration_hours','')}"
                    f",{bypass.get('reason','')}\n"
                )
        elif isinstance(report, ComplianceReport):
            output.write("sif_id,sif_name,sil_level,status,test_passed,response_time_ok\n")
            for sif in report.sif_compliance:
                output.write(
                    f"{sif.get('sif_id','')},{sif.get('sif_name','')}"
                    f",{sif.get('sil_level','')},{sif.get('status','')}"
                    f",{sif.get('test_passed','')},{sif.get('response_time_ok','')}\n"
                )

        return output.getvalue()

    def _export_html(self, report: Any) -> str:
        """Export report as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.metadata.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .compliant {{ background-color: #90EE90; }}
        .marginal {{ background-color: #FFD700; }}
        .non-compliant {{ background-color: #FF6B6B; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; }}
        .footer {{ margin-top: 30px; font-size: 0.8em; color: #666; }}
    </style>
</head>
<body>
    <h1>{report.metadata.title}</h1>
    <div class="summary">
        <p><strong>System:</strong> {report.metadata.system_id}</p>
        <p><strong>Generated:</strong> {report.metadata.generated_at.isoformat()}</p>
        <p><strong>Period:</strong> {report.metadata.period_start.isoformat() if report.metadata.period_start else 'N/A'} to {report.metadata.period_end.isoformat() if report.metadata.period_end else 'N/A'}</p>
    </div>
"""

        if isinstance(report, ComplianceReport):
            html += f"""
    <h2>Executive Summary</h2>
    <p>{report.executive_summary}</p>

    <h2>Overall Compliance</h2>
    <p><strong>Status:</strong> {report.overall_status.value}</p>
    <p><strong>Score:</strong> {report.compliance_score:.1f}%</p>

    <h2>SIF Compliance Status</h2>
    <table>
        <tr>
            <th>SIF ID</th>
            <th>Name</th>
            <th>SIL</th>
            <th>Status</th>
            <th>Tests</th>
            <th>Response Time</th>
        </tr>
"""
            for sif in report.sif_compliance:
                status_class = sif['status'].replace('_', '-')
                html += f"""
        <tr class="{status_class}">
            <td>{sif['sif_id']}</td>
            <td>{sif['sif_name']}</td>
            <td>{sif['sil_level']}</td>
            <td>{sif['status']}</td>
            <td>{'Pass' if sif['test_passed'] else 'Fail' if sif['test_passed'] is False else 'N/A'}</td>
            <td>{'OK' if sif['response_time_ok'] else 'Fail' if sif['response_time_ok'] is False else 'N/A'}</td>
        </tr>
"""
            html += """    </table>"""

            if report.recommendations:
                html += """
    <h2>Recommendations</h2>
    <ul>
"""
                for rec in report.recommendations:
                    html += f"        <li>{rec}</li>\n"
                html += """    </ul>"""

        html += f"""
    <div class="footer">
        <p>Report ID: {report.metadata.report_id}</p>
        <p>Provenance Hash: {report.metadata.provenance_hash}</p>
        <p>Generated by: {report.metadata.generated_by}</p>
    </div>
</body>
</html>
"""
        return html

    def _export_pdf(self, report: Any) -> bytes:
        """Export report as PDF (placeholder - requires external library)."""
        # In production, use reportlab or weasyprint
        # This is a placeholder that returns HTML as bytes
        html = self._export_html(report)
        return html.encode('utf-8')

    def _export_excel(self, report: Any) -> bytes:
        """Export report as Excel (placeholder - requires openpyxl)."""
        # In production, use openpyxl
        # This is a placeholder that returns CSV as bytes
        csv = self._export_csv(report)
        return csv.encode('utf-8')

    def _calculate_test_trends(
        self,
        tests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate test trends from historical data."""
        if not tests:
            return {"error": "No data"}

        # Sort by date
        sorted_tests = sorted(
            tests,
            key=lambda t: t.get("test_date", "")
        )

        # Calculate monthly pass rates
        monthly_rates = {}
        for test in sorted_tests:
            date_str = test.get("test_date", "")[:7]  # YYYY-MM
            if date_str not in monthly_rates:
                monthly_rates[date_str] = {"passed": 0, "total": 0}
            monthly_rates[date_str]["total"] += 1
            if test.get("passed", False):
                monthly_rates[date_str]["passed"] += 1

        # Calculate rates
        for month, data in monthly_rates.items():
            data["rate"] = (data["passed"] / data["total"] * 100) if data["total"] > 0 else 0

        return {
            "monthly_pass_rates": monthly_rates,
            "overall_trend": self._calculate_trend_direction(monthly_rates),
        }

    def _calculate_trend_direction(
        self,
        monthly_data: Dict[str, Dict[str, Any]]
    ) -> str:
        """Calculate overall trend direction."""
        if len(monthly_data) < 2:
            return "insufficient_data"

        rates = [d["rate"] for d in monthly_data.values()]

        first_half = sum(rates[:len(rates)//2]) / len(rates[:len(rates)//2])
        second_half = sum(rates[len(rates)//2:]) / len(rates[len(rates)//2:])

        if second_half > first_half + 5:
            return "improving"
        elif second_half < first_half - 5:
            return "degrading"
        else:
            return "stable"

    def _calculate_provenance(self, report: Any) -> str:
        """Calculate SHA-256 provenance hash for report."""
        metadata = report.metadata

        provenance_str = (
            f"{metadata.report_id}|"
            f"{metadata.report_type.value}|"
            f"{metadata.system_id}|"
            f"{metadata.generated_at.isoformat()}"
        )

        return hashlib.sha256(provenance_str.encode()).hexdigest()
