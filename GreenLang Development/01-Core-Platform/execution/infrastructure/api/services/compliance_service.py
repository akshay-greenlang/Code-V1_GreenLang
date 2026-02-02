"""
Compliance Service for GreenLang GraphQL API

This module provides the ComplianceService class that manages regulatory
compliance report generation, findings tracking, and audit trail.

Features:
    - Multi-framework compliance reporting (GHG Protocol, ISO 14064, etc.)
    - Compliance finding categorization and severity
    - Remediation action tracking
    - Report generation with provenance hashes
    - Historical report access

Example:
    >>> service = ComplianceService()
    >>> params = ReportParams(
    ...     facility_ids=["facility-001"],
    ...     start_date=date(2025, 1, 1),
    ...     end_date=date(2025, 3, 31)
    ... )
    >>> report = await service.generate_report("ghg_emissions", params)
    >>> reports = await service.list_reports(report_type="ghg_emissions")
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportTypeEnum(str, Enum):
    """Compliance report type."""
    GHG_EMISSIONS = "ghg_emissions"
    ENERGY_AUDIT = "energy_audit"
    EFFICIENCY_ANALYSIS = "efficiency_analysis"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ENVIRONMENTAL_IMPACT = "environmental_impact"


class ComplianceStatusEnum(str, Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    PENDING_REMEDIATION = "pending_remediation"
    PARTIALLY_COMPLIANT = "partially_compliant"


class FindingSeverityEnum(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategoryEnum(str, Enum):
    """Finding categories."""
    EMISSIONS_TRACKING = "emissions_tracking"
    DATA_QUALITY = "data_quality"
    CALCULATION_METHOD = "calculation_method"
    DOCUMENTATION = "documentation"
    REPORTING_DEADLINE = "reporting_deadline"
    VERIFICATION = "verification"
    SCOPE_COVERAGE = "scope_coverage"


@dataclass
class ComplianceFinding:
    """Single compliance finding."""
    id: str
    category: FindingCategoryEnum
    severity: FindingSeverityEnum
    description: str
    remediation_action: Optional[str]
    deadline: Optional[date]
    assigned_to: Optional[str]
    status: str
    created_at: datetime


@dataclass
class ComplianceReport:
    """Compliance report record."""
    id: str
    report_type: ReportTypeEnum
    status: ComplianceStatusEnum
    period_start: date
    period_end: date
    facility_ids: List[str]
    framework: str
    findings: List[ComplianceFinding]
    summary: str
    action_items_count: int
    compliance_score: float
    generated_at: datetime
    generated_by: str
    provenance_hash: str
    metadata: Dict[str, Any]


class ReportParams(BaseModel):
    """Report generation parameters."""
    facility_ids: List[str] = Field(..., description="Target facility IDs")
    start_date: date = Field(..., description="Report period start")
    end_date: date = Field(..., description="Report period end")
    framework: str = Field(
        default="GHG Protocol",
        description="Compliance framework"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include remediation recommendations"
    )
    include_scope3: bool = Field(
        default=False,
        description="Include Scope 3 emissions analysis"
    )

    @validator('end_date')
    def validate_date_range(cls, v: date, values: Dict[str, Any]) -> date:
        """Validate end_date is after start_date."""
        if 'start_date' in values and v < values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v

    @validator('facility_ids')
    def validate_facility_ids(cls, v: List[str]) -> List[str]:
        """Validate at least one facility is specified."""
        if not v:
            raise ValueError("At least one facility_id is required")
        return v


class ComplianceServiceError(Exception):
    """Base exception for compliance service errors."""
    pass


class ReportNotFoundError(ComplianceServiceError):
    """Raised when a report is not found."""
    pass


class ReportGenerationError(ComplianceServiceError):
    """Raised when report generation fails."""
    pass


class ComplianceService:
    """
    Compliance Service for managing regulatory compliance reports.

    Provides report generation, finding tracking, and compliance
    monitoring with full audit trail.

    Attributes:
        _reports: In-memory report registry
        _lock: Asyncio lock for thread-safe operations

    Example:
        >>> service = ComplianceService()
        >>> params = ReportParams(
        ...     facility_ids=["facility-001"],
        ...     start_date=date(2025, 1, 1),
        ...     end_date=date(2025, 3, 31)
        ... )
        >>> report = await service.generate_report("ghg_emissions", params)
        >>> print(f"Report {report.id}: {report.status}")
    """

    def __init__(self) -> None:
        """Initialize ComplianceService."""
        self._reports: Dict[str, ComplianceReport] = {}
        self._lock = asyncio.Lock()
        self._initialize_sample_reports()
        logger.info("ComplianceService initialized")

    def _initialize_sample_reports(self) -> None:
        """Initialize sample reports for demonstration."""
        now = datetime.utcnow()
        today = now.date()

        sample_reports = [
            ComplianceReport(
                id="report-001",
                report_type=ReportTypeEnum.GHG_EMISSIONS,
                status=ComplianceStatusEnum.COMPLIANT,
                period_start=date(2025, 1, 1),
                period_end=date(2025, 3, 31),
                facility_ids=["facility-001", "facility-002"],
                framework="GHG Protocol",
                findings=[
                    ComplianceFinding(
                        id="finding-001",
                        category=FindingCategoryEnum.DATA_QUALITY,
                        severity=FindingSeverityEnum.LOW,
                        description="Minor gaps in Scope 2 activity data",
                        remediation_action="Implement automated meter reading",
                        deadline=date(2025, 6, 30),
                        assigned_to="operations@company.com",
                        status="in_progress",
                        created_at=now
                    )
                ],
                summary=(
                    "Q1 2025 GHG emissions report shows 98% compliance with "
                    "GHG Protocol Corporate Standard. Total emissions: 12,456 tCO2e "
                    "(5% reduction YoY). One minor finding related to data quality."
                ),
                action_items_count=1,
                compliance_score=0.98,
                generated_at=now - timedelta(days=5),
                generated_by="system",
                provenance_hash=hashlib.sha256(b"report-001").hexdigest(),
                metadata={"version": "1.0", "reviewer": "compliance_team"}
            ),
            ComplianceReport(
                id="report-002",
                report_type=ReportTypeEnum.ENERGY_AUDIT,
                status=ComplianceStatusEnum.PARTIALLY_COMPLIANT,
                period_start=date(2024, 10, 1),
                period_end=date(2024, 12, 31),
                facility_ids=["facility-003"],
                framework="ISO 50001",
                findings=[
                    ComplianceFinding(
                        id="finding-002",
                        category=FindingCategoryEnum.CALCULATION_METHOD,
                        severity=FindingSeverityEnum.MEDIUM,
                        description="Energy baseline requires recalculation",
                        remediation_action="Update baseline with 2024 data",
                        deadline=date(2025, 3, 31),
                        assigned_to="energy_manager@company.com",
                        status="pending",
                        created_at=now - timedelta(days=30)
                    ),
                    ComplianceFinding(
                        id="finding-003",
                        category=FindingCategoryEnum.DOCUMENTATION,
                        severity=FindingSeverityEnum.HIGH,
                        description="Missing EnPI tracking documentation",
                        remediation_action="Establish EnPI tracking system",
                        deadline=date(2025, 2, 28),
                        assigned_to="energy_manager@company.com",
                        status="in_progress",
                        created_at=now - timedelta(days=30)
                    )
                ],
                summary=(
                    "Q4 2024 energy audit identifies two areas requiring attention. "
                    "Energy performance improvement target: 5% by end of 2025."
                ),
                action_items_count=2,
                compliance_score=0.75,
                generated_at=now - timedelta(days=15),
                generated_by="system",
                provenance_hash=hashlib.sha256(b"report-002").hexdigest(),
                metadata={"version": "1.0", "audit_type": "internal"}
            ),
            ComplianceReport(
                id="report-003",
                report_type=ReportTypeEnum.REGULATORY_COMPLIANCE,
                status=ComplianceStatusEnum.UNDER_REVIEW,
                period_start=date(2024, 1, 1),
                period_end=date(2024, 12, 31),
                facility_ids=["facility-001"],
                framework="EPA GHGRP",
                findings=[],
                summary=(
                    "Annual EPA GHG Reporting Program submission under review. "
                    "Total reported emissions: 48,234 tCO2e. Awaiting verification."
                ),
                action_items_count=0,
                compliance_score=1.0,
                generated_at=now - timedelta(days=2),
                generated_by="system",
                provenance_hash=hashlib.sha256(b"report-003").hexdigest(),
                metadata={"version": "1.0", "submission_id": "EPA-2024-12345"}
            ),
        ]

        for report in sample_reports:
            self._reports[report.id] = report

    async def generate_report(
        self,
        report_type: str,
        params: ReportParams
    ) -> ComplianceReport:
        """
        Generate a new compliance report.

        Args:
            report_type: Type of report to generate
            params: Report generation parameters

        Returns:
            Generated ComplianceReport

        Raises:
            ReportGenerationError: If report generation fails
        """
        try:
            # Validate report type
            try:
                report_type_enum = ReportTypeEnum(report_type.lower())
            except ValueError:
                raise ReportGenerationError(
                    f"Invalid report type: {report_type}. "
                    f"Valid types: {[t.value for t in ReportTypeEnum]}"
                )

            report_id = str(uuid4())
            now = datetime.utcnow()

            # Generate findings based on simulated analysis
            findings = await self._analyze_compliance(
                report_type_enum,
                params.facility_ids,
                params.start_date,
                params.end_date
            )

            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(findings)

            # Determine status
            status = self._determine_status(compliance_score, findings)

            # Generate summary
            summary = self._generate_summary(
                report_type_enum,
                params,
                findings,
                compliance_score
            )

            # Calculate provenance hash
            provenance_hash = self._calculate_report_hash(
                report_id, report_type_enum, params
            )

            report = ComplianceReport(
                id=report_id,
                report_type=report_type_enum,
                status=status,
                period_start=params.start_date,
                period_end=params.end_date,
                facility_ids=params.facility_ids,
                framework=params.framework,
                findings=findings,
                summary=summary,
                action_items_count=len([f for f in findings if f.deadline]),
                compliance_score=compliance_score,
                generated_at=now,
                generated_by="system",
                provenance_hash=provenance_hash,
                metadata={
                    "include_recommendations": params.include_recommendations,
                    "include_scope3": params.include_scope3,
                    "version": "1.0"
                }
            )

            async with self._lock:
                self._reports[report_id] = report

            logger.info(
                f"Generated {report_type} report {report_id} for "
                f"{len(params.facility_ids)} facilities"
            )
            return report

        except ReportGenerationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            raise ReportGenerationError(
                f"Failed to generate report: {str(e)}"
            ) from e

    async def list_reports(
        self,
        report_type: Optional[str] = None,
        status: Optional[str] = None,
        facility_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ComplianceReport]:
        """
        List compliance reports with optional filters.

        Args:
            report_type: Filter by report type
            status: Filter by compliance status
            facility_id: Filter by facility ID
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching ComplianceReport records
        """
        async with self._lock:
            reports = list(self._reports.values())

        # Apply filters
        if report_type:
            try:
                type_enum = ReportTypeEnum(report_type.lower())
                reports = [r for r in reports if r.report_type == type_enum]
            except ValueError:
                logger.warning(f"Invalid report_type filter: {report_type}")
                return []

        if status:
            try:
                status_enum = ComplianceStatusEnum(status.lower())
                reports = [r for r in reports if r.status == status_enum]
            except ValueError:
                logger.warning(f"Invalid status filter: {status}")
                return []

        if facility_id:
            reports = [r for r in reports if facility_id in r.facility_ids]

        # Sort by generation time (newest first)
        reports.sort(key=lambda r: r.generated_at, reverse=True)

        # Apply pagination
        reports = reports[offset:offset + limit]

        logger.debug(f"Listed {len(reports)} reports")
        return reports

    async def get_report(self, report_id: str) -> Optional[ComplianceReport]:
        """
        Get a specific report by ID.

        Args:
            report_id: Report identifier

        Returns:
            ComplianceReport if found, None otherwise
        """
        async with self._lock:
            report = self._reports.get(report_id)

        if report:
            logger.debug(f"Retrieved report {report_id}")
        else:
            logger.debug(f"Report not found: {report_id}")

        return report

    async def get_report_findings(
        self,
        report_id: str,
        severity: Optional[str] = None
    ) -> List[ComplianceFinding]:
        """
        Get findings for a specific report.

        Args:
            report_id: Report identifier
            severity: Optional severity filter

        Returns:
            List of ComplianceFinding records
        """
        report = await self.get_report(report_id)
        if not report:
            return []

        findings = report.findings

        if severity:
            try:
                severity_enum = FindingSeverityEnum(severity.lower())
                findings = [f for f in findings if f.severity == severity_enum]
            except ValueError:
                logger.warning(f"Invalid severity filter: {severity}")
                return []

        return findings

    async def update_finding_status(
        self,
        report_id: str,
        finding_id: str,
        status: str
    ) -> Optional[ComplianceFinding]:
        """
        Update the status of a finding.

        Args:
            report_id: Report identifier
            finding_id: Finding identifier
            status: New status

        Returns:
            Updated ComplianceFinding if found, None otherwise
        """
        async with self._lock:
            report = self._reports.get(report_id)
            if not report:
                return None

            for finding in report.findings:
                if finding.id == finding_id:
                    finding.status = status
                    logger.info(
                        f"Updated finding {finding_id} status to {status}"
                    )
                    return finding

        return None

    async def _analyze_compliance(
        self,
        report_type: ReportTypeEnum,
        facility_ids: List[str],
        start_date: date,
        end_date: date
    ) -> List[ComplianceFinding]:
        """
        Analyze compliance and generate findings.

        This is a simulation - in production, this would integrate
        with actual data sources and validation rules.
        """
        now = datetime.utcnow()
        findings = []

        # Simulate finding generation based on report type
        if report_type == ReportTypeEnum.GHG_EMISSIONS:
            # Simulate a data quality finding
            if len(facility_ids) > 1:
                findings.append(ComplianceFinding(
                    id=str(uuid4()),
                    category=FindingCategoryEnum.DATA_QUALITY,
                    severity=FindingSeverityEnum.LOW,
                    description=(
                        "Minor discrepancies in meter readings detected "
                        "across facilities"
                    ),
                    remediation_action=(
                        "Implement cross-facility data validation checks"
                    ),
                    deadline=end_date + timedelta(days=90),
                    assigned_to=None,
                    status="pending",
                    created_at=now
                ))

        elif report_type == ReportTypeEnum.ENERGY_AUDIT:
            findings.append(ComplianceFinding(
                id=str(uuid4()),
                category=FindingCategoryEnum.DOCUMENTATION,
                severity=FindingSeverityEnum.MEDIUM,
                description="Energy performance indicators require documentation",
                remediation_action="Create EnPI tracking dashboard",
                deadline=end_date + timedelta(days=60),
                assigned_to=None,
                status="pending",
                created_at=now
            ))

        return findings

    def _calculate_compliance_score(
        self,
        findings: List[ComplianceFinding]
    ) -> float:
        """Calculate overall compliance score based on findings."""
        if not findings:
            return 1.0

        # Weight by severity
        severity_weights = {
            FindingSeverityEnum.CRITICAL: 0.25,
            FindingSeverityEnum.HIGH: 0.15,
            FindingSeverityEnum.MEDIUM: 0.08,
            FindingSeverityEnum.LOW: 0.03,
            FindingSeverityEnum.INFO: 0.01,
        }

        total_deduction = sum(
            severity_weights.get(f.severity, 0.05)
            for f in findings
        )

        return max(0.0, min(1.0, 1.0 - total_deduction))

    def _determine_status(
        self,
        score: float,
        findings: List[ComplianceFinding]
    ) -> ComplianceStatusEnum:
        """Determine compliance status based on score and findings."""
        # Check for critical findings
        critical_findings = [
            f for f in findings
            if f.severity == FindingSeverityEnum.CRITICAL
        ]
        if critical_findings:
            return ComplianceStatusEnum.NON_COMPLIANT

        # Check for high-severity findings
        high_findings = [
            f for f in findings
            if f.severity == FindingSeverityEnum.HIGH
        ]
        if high_findings:
            return ComplianceStatusEnum.PENDING_REMEDIATION

        # Score-based determination
        if score >= 0.95:
            return ComplianceStatusEnum.COMPLIANT
        elif score >= 0.80:
            return ComplianceStatusEnum.PARTIALLY_COMPLIANT
        elif score >= 0.60:
            return ComplianceStatusEnum.UNDER_REVIEW
        else:
            return ComplianceStatusEnum.NON_COMPLIANT

    def _generate_summary(
        self,
        report_type: ReportTypeEnum,
        params: ReportParams,
        findings: List[ComplianceFinding],
        score: float
    ) -> str:
        """Generate executive summary for report."""
        period = f"{params.start_date.strftime('%b %Y')} - {params.end_date.strftime('%b %Y')}"
        facility_count = len(params.facility_ids)

        summaries = {
            ReportTypeEnum.GHG_EMISSIONS: (
                f"GHG emissions report for {facility_count} facilities covering {period}. "
                f"Compliance score: {score:.0%}. "
                f"{len(findings)} finding(s) identified."
            ),
            ReportTypeEnum.ENERGY_AUDIT: (
                f"Energy audit for {facility_count} facilities covering {period}. "
                f"Overall compliance: {score:.0%}. "
                f"{len(findings)} area(s) require attention."
            ),
            ReportTypeEnum.REGULATORY_COMPLIANCE: (
                f"Regulatory compliance assessment for {period}. "
                f"Compliance score: {score:.0%}. "
                f"Framework: {params.framework}."
            ),
            ReportTypeEnum.EFFICIENCY_ANALYSIS: (
                f"Efficiency analysis for {facility_count} facilities covering {period}. "
                f"Performance score: {score:.0%}."
            ),
            ReportTypeEnum.PREDICTIVE_MAINTENANCE: (
                f"Predictive maintenance report for {period}. "
                f"System health score: {score:.0%}."
            ),
            ReportTypeEnum.ENVIRONMENTAL_IMPACT: (
                f"Environmental impact assessment for {period}. "
                f"Sustainability score: {score:.0%}."
            ),
        }

        return summaries.get(
            report_type,
            f"Compliance report for {period}. Score: {score:.0%}."
        )

    def _calculate_report_hash(
        self,
        report_id: str,
        report_type: ReportTypeEnum,
        params: ReportParams
    ) -> str:
        """Calculate SHA-256 hash for report provenance."""
        content = json.dumps({
            "report_id": report_id,
            "report_type": report_type.value,
            "facility_ids": sorted(params.facility_ids),
            "start_date": params.start_date.isoformat(),
            "end_date": params.end_date.isoformat(),
            "framework": params.framework,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# Singleton instance
_compliance_service_instance: Optional[ComplianceService] = None


def get_compliance_service() -> ComplianceService:
    """
    Get the global ComplianceService instance.

    Returns:
        ComplianceService singleton instance
    """
    global _compliance_service_instance
    if _compliance_service_instance is None:
        _compliance_service_instance = ComplianceService()
    return _compliance_service_instance


__all__ = [
    "ComplianceService",
    "ComplianceReport",
    "ComplianceFinding",
    "ReportParams",
    "ReportTypeEnum",
    "ComplianceStatusEnum",
    "FindingSeverityEnum",
    "FindingCategoryEnum",
    "ComplianceServiceError",
    "ReportNotFoundError",
    "ReportGenerationError",
    "get_compliance_service",
]
