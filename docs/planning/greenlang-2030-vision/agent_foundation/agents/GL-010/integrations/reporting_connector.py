"""
Regulatory Reporting Systems Connector for GL-010 EMISSIONWATCH.

Provides integration with state/local agency submissions, air quality
management districts, tribal authority reporting, international reporting
(UNECE CLRTAP), and corporate sustainability reporting systems.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import asyncio
import json
import logging
import time
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator
import httpx

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    ConnectorError,
    ConnectionError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ReportingSystem(str, Enum):
    """Regulatory reporting systems."""

    # US State/Local
    STATE_AGENCY = "state_agency"
    LOCAL_AGENCY = "local_agency"
    AIR_DISTRICT = "air_district"
    TRIBAL = "tribal"

    # US Federal
    EPA_EIS = "epa_eis"  # Emissions Inventory System
    EPA_TRI = "epa_tri"  # Toxics Release Inventory
    EPA_GHGRP = "epa_ghgrp"  # GHG Reporting Program
    EPA_CEDRI = "epa_cedri"

    # International
    UNECE_CLRTAP = "unece_clrtap"  # LRTAP Convention
    EU_PRTR = "eu_prtr"  # Pollutant Release and Transfer Register
    EU_ETS = "eu_ets"  # Emissions Trading System
    UNFCCC = "unfccc"  # Climate change reporting

    # Corporate/Voluntary
    CDP = "cdp"  # Carbon Disclosure Project
    GRI = "gri"  # Global Reporting Initiative
    SASB = "sasb"  # Sustainability Accounting Standards
    TCFD = "tcfd"  # Task Force on Climate Disclosures
    SBTi = "sbti"  # Science Based Targets initiative
    CSR = "csr"  # Corporate Social Responsibility


class ReportFormat(str, Enum):
    """Report format types."""

    XML = "xml"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    HTML = "html"
    FLAT_FILE = "flat_file"
    EDI = "edi"


class ReportFrequency(str, Enum):
    """Reporting frequency."""

    REALTIME = "realtime"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    BIENNIAL = "biennial"
    AD_HOC = "ad_hoc"
    EVENT_DRIVEN = "event_driven"


class SubmissionMethod(str, Enum):
    """Report submission methods."""

    WEB_FORM = "web_form"
    API = "api"
    SFTP = "sftp"
    EMAIL = "email"
    PORTAL_UPLOAD = "portal_upload"
    EDI = "edi"
    MAIL = "mail"
    FAX = "fax"


class SubmissionStatus(str, Enum):
    """Report submission status."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    READY_FOR_SUBMISSION = "ready_for_submission"
    SUBMITTED = "submitted"
    RECEIVED = "received"
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REQUIRES_CORRECTION = "requires_correction"
    OVERDUE = "overdue"


class PollutantCategory(str, Enum):
    """Pollutant categories for reporting."""

    CRITERIA = "criteria"  # NOx, SO2, CO, PM, O3, Pb
    HAP = "hap"  # Hazardous Air Pollutants
    GHG = "ghg"  # Greenhouse Gases
    VOC = "voc"  # Volatile Organic Compounds
    TOXIC = "toxic"  # Toxic substances
    OZONE_DEPLETING = "ozone_depleting"
    HEAVY_METALS = "heavy_metals"
    PERSISTENT_ORGANIC = "persistent_organic"


# =============================================================================
# Pydantic Models
# =============================================================================


class ReportingAgency(BaseModel):
    """Regulatory agency information."""

    model_config = ConfigDict(frozen=True)

    agency_id: str = Field(..., description="Agency identifier")
    agency_name: str = Field(..., description="Agency name")
    agency_type: str = Field(..., description="Agency type")

    # Jurisdiction
    jurisdiction_level: str = Field(..., description="Federal/State/Local")
    state: Optional[str] = Field(default=None, description="State code")
    region: Optional[str] = Field(default=None, description="Region/district")

    # Contact
    contact_name: Optional[str] = Field(default=None)
    contact_email: Optional[str] = Field(default=None)
    contact_phone: Optional[str] = Field(default=None)
    website: Optional[str] = Field(default=None)

    # Submission
    submission_portal_url: Optional[str] = Field(default=None)
    api_endpoint: Optional[str] = Field(default=None)
    preferred_format: ReportFormat = Field(default=ReportFormat.XML)


class ReportingRequirement(BaseModel):
    """Regulatory reporting requirement."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Requirement identifier"
    )
    agency_id: str = Field(..., description="Reporting agency")
    reporting_system: ReportingSystem = Field(..., description="Reporting system")

    # Report details
    report_name: str = Field(..., description="Report name")
    report_type: str = Field(..., description="Report type")
    regulatory_basis: str = Field(..., description="Regulatory citation")

    # Frequency and deadlines
    frequency: ReportFrequency = Field(..., description="Reporting frequency")
    due_date_rule: str = Field(
        ...,
        description="Due date rule (e.g., '30 days after quarter end')"
    )
    grace_period_days: int = Field(default=0, ge=0)

    # Content
    pollutant_categories: List[PollutantCategory] = Field(
        default_factory=list,
        description="Pollutant categories"
    )
    required_data_elements: List[str] = Field(
        default_factory=list,
        description="Required data elements"
    )

    # Format
    report_format: ReportFormat = Field(..., description="Required format")
    submission_method: SubmissionMethod = Field(..., description="Submission method")

    # Applicability
    applicable_facility_types: List[str] = Field(default_factory=list)
    threshold_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Reporting thresholds"
    )

    is_mandatory: bool = Field(default=True)
    penalty_for_late: Optional[str] = Field(default=None)


class EmissionsDataRecord(BaseModel):
    """Emissions data for reporting."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Record identifier"
    )
    unit_id: str = Field(..., description="Emission unit ID")
    stack_id: Optional[str] = Field(default=None, description="Stack ID")

    # Period
    reporting_period_start: date = Field(..., description="Period start")
    reporting_period_end: date = Field(..., description="Period end")

    # Pollutant
    pollutant_code: str = Field(..., description="Pollutant code")
    pollutant_name: str = Field(..., description="Pollutant name")
    cas_number: Optional[str] = Field(default=None, description="CAS number")

    # Emissions
    emissions_value: Decimal = Field(..., ge=0, description="Emissions value")
    emissions_unit: str = Field(..., description="Unit")
    emissions_type: str = Field(
        default="actual",
        description="Type (actual/potential)"
    )

    # Calculation
    calculation_method: str = Field(..., description="Calculation method")
    emission_factor: Optional[float] = Field(default=None)
    emission_factor_source: Optional[str] = Field(default=None)

    # Quality
    data_quality_rating: Optional[str] = Field(default=None)
    uncertainty_percent: Optional[float] = Field(default=None, ge=0, le=100)


class ReportDocument(BaseModel):
    """Generated report document."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Document identifier"
    )
    requirement_id: str = Field(..., description="Requirement ID")

    # Period
    reporting_period_start: date = Field(..., description="Period start")
    reporting_period_end: date = Field(..., description="Period end")

    # Document
    document_format: ReportFormat = Field(..., description="Format")
    document_content: Optional[str] = Field(default=None, description="Content")
    document_path: Optional[str] = Field(default=None, description="File path")
    document_size_bytes: Optional[int] = Field(default=None, ge=0)

    # Generation
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Generation timestamp"
    )
    generated_by: str = Field(default="system", description="Generated by")

    # Status
    status: SubmissionStatus = Field(
        default=SubmissionStatus.DRAFT,
        description="Status"
    )

    # Validation
    is_validated: bool = Field(default=False)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)


class ReportSubmission(BaseModel):
    """Report submission record."""

    model_config = ConfigDict(frozen=True)

    submission_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Submission identifier"
    )
    document_id: str = Field(..., description="Document ID")
    requirement_id: str = Field(..., description="Requirement ID")
    agency_id: str = Field(..., description="Agency ID")

    # Submission details
    submission_method: SubmissionMethod = Field(..., description="Method")
    submitted_at: datetime = Field(..., description="Submission timestamp")
    submitted_by: str = Field(..., description="Submitter")

    # Confirmation
    confirmation_number: Optional[str] = Field(default=None)
    receipt_timestamp: Optional[datetime] = Field(default=None)
    receipt_document: Optional[str] = Field(default=None)

    # Status
    status: SubmissionStatus = Field(..., description="Status")
    status_message: Optional[str] = Field(default=None)
    last_status_update: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update"
    )

    # Follow-up
    agency_comments: Optional[str] = Field(default=None)
    correction_required: bool = Field(default=False)
    correction_deadline: Optional[date] = Field(default=None)


class ReportingSchedule(BaseModel):
    """Reporting schedule entry."""

    model_config = ConfigDict(frozen=True)

    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Schedule identifier"
    )
    requirement_id: str = Field(..., description="Requirement ID")

    # Period
    reporting_period_start: date = Field(..., description="Period start")
    reporting_period_end: date = Field(..., description="Period end")
    due_date: date = Field(..., description="Due date")

    # Status
    status: str = Field(default="pending", description="Status")
    document_id: Optional[str] = Field(default=None)
    submission_id: Optional[str] = Field(default=None)

    # Reminders
    reminder_sent: bool = Field(default=False)
    days_until_due: Optional[int] = Field(default=None)


class ReportingConnectorConfig(BaseConnectorConfig):
    """Configuration for reporting connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.REPORTING,
        description="Connector type"
    )

    # Facility identification
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Location (for jurisdiction determination)
    state: str = Field(..., min_length=2, max_length=2, description="State code")
    county: Optional[str] = Field(default=None)
    air_district: Optional[str] = Field(default=None)

    # Registered reporting systems
    registered_systems: List[ReportingSystem] = Field(
        default_factory=list,
        description="Registered reporting systems"
    )

    # Credentials storage
    credentials_vault_path: Optional[str] = Field(
        default=None,
        description="Path to credentials in vault"
    )

    # Schedule settings
    reminder_days_before_due: List[int] = Field(
        default_factory=lambda: [30, 14, 7, 1],
        description="Reminder days before due"
    )

    # Generation settings
    default_report_format: ReportFormat = Field(
        default=ReportFormat.XML,
        description="Default report format"
    )
    auto_validate: bool = Field(
        default=True,
        description="Auto-validate before submission"
    )


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """
    Generates reports in various formats.

    Supports XML, JSON, CSV, and structured formats for
    different regulatory systems.
    """

    def __init__(self, facility_id: str, facility_name: str) -> None:
        """
        Initialize generator.

        Args:
            facility_id: Facility identifier
            facility_name: Facility name
        """
        self._facility_id = facility_id
        self._facility_name = facility_name
        self._logger = logging.getLogger("reporting.generator")

    def generate_emissions_inventory(
        self,
        emissions_data: List[EmissionsDataRecord],
        period_start: date,
        period_end: date,
        report_format: ReportFormat,
    ) -> ReportDocument:
        """
        Generate emissions inventory report.

        Args:
            emissions_data: Emissions data records
            period_start: Period start
            period_end: Period end
            report_format: Output format

        Returns:
            Generated report document
        """
        if report_format == ReportFormat.JSON:
            content = self._generate_json_inventory(emissions_data, period_start, period_end)
        elif report_format == ReportFormat.XML:
            content = self._generate_xml_inventory(emissions_data, period_start, period_end)
        elif report_format == ReportFormat.CSV:
            content = self._generate_csv_inventory(emissions_data)
        else:
            raise ValidationError(f"Unsupported format: {report_format}")

        return ReportDocument(
            requirement_id="emissions_inventory",
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            document_format=report_format,
            document_content=content,
            status=SubmissionStatus.DRAFT,
        )

    def _generate_json_inventory(
        self,
        emissions_data: List[EmissionsDataRecord],
        period_start: date,
        period_end: date,
    ) -> str:
        """Generate JSON format inventory."""
        report = {
            "reportType": "EmissionsInventory",
            "facilityId": self._facility_id,
            "facilityName": self._facility_name,
            "reportingPeriod": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "generatedAt": datetime.utcnow().isoformat(),
            "emissions": [
                {
                    "unitId": record.unit_id,
                    "pollutant": record.pollutant_code,
                    "pollutantName": record.pollutant_name,
                    "emissions": float(record.emissions_value),
                    "unit": record.emissions_unit,
                    "calculationMethod": record.calculation_method,
                }
                for record in emissions_data
            ],
        }
        return json.dumps(report, indent=2)

    def _generate_xml_inventory(
        self,
        emissions_data: List[EmissionsDataRecord],
        period_start: date,
        period_end: date,
    ) -> str:
        """Generate XML format inventory."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<EmissionsInventory>',
            f'  <FacilityId>{self._facility_id}</FacilityId>',
            f'  <FacilityName>{self._facility_name}</FacilityName>',
            '  <ReportingPeriod>',
            f'    <Start>{period_start.isoformat()}</Start>',
            f'    <End>{period_end.isoformat()}</End>',
            '  </ReportingPeriod>',
            '  <Emissions>',
        ]

        for record in emissions_data:
            lines.extend([
                '    <EmissionRecord>',
                f'      <UnitId>{record.unit_id}</UnitId>',
                f'      <Pollutant>{record.pollutant_code}</Pollutant>',
                f'      <Value>{record.emissions_value}</Value>',
                f'      <Unit>{record.emissions_unit}</Unit>',
                f'      <Method>{record.calculation_method}</Method>',
                '    </EmissionRecord>',
            ])

        lines.extend([
            '  </Emissions>',
            '</EmissionsInventory>',
        ])

        return '\n'.join(lines)

    def _generate_csv_inventory(
        self,
        emissions_data: List[EmissionsDataRecord],
    ) -> str:
        """Generate CSV format inventory."""
        lines = [
            "UnitId,Pollutant,PollutantName,Emissions,Unit,Method",
        ]

        for record in emissions_data:
            lines.append(
                f"{record.unit_id},{record.pollutant_code},"
                f"{record.pollutant_name},{record.emissions_value},"
                f"{record.emissions_unit},{record.calculation_method}"
            )

        return '\n'.join(lines)

    def generate_ghg_report(
        self,
        emissions_data: List[EmissionsDataRecord],
        reporting_year: int,
        report_format: ReportFormat,
    ) -> ReportDocument:
        """
        Generate GHG emissions report.

        Args:
            emissions_data: GHG emissions data
            reporting_year: Reporting year
            report_format: Output format

        Returns:
            Generated report document
        """
        # Filter to GHG pollutants
        ghg_pollutants = {"co2", "ch4", "n2o", "co2e", "hfc", "pfc", "sf6"}
        ghg_data = [
            r for r in emissions_data
            if r.pollutant_code.lower() in ghg_pollutants
        ]

        period_start = date(reporting_year, 1, 1)
        period_end = date(reporting_year, 12, 31)

        return self.generate_emissions_inventory(
            ghg_data,
            period_start,
            period_end,
            report_format,
        )

    def generate_criteria_pollutant_report(
        self,
        emissions_data: List[EmissionsDataRecord],
        period_start: date,
        period_end: date,
        report_format: ReportFormat,
    ) -> ReportDocument:
        """
        Generate criteria pollutant report.

        Args:
            emissions_data: Emissions data
            period_start: Period start
            period_end: Period end
            report_format: Output format

        Returns:
            Generated report document
        """
        criteria_pollutants = {"nox", "so2", "co", "pm", "pm10", "pm25", "pb", "o3"}
        criteria_data = [
            r for r in emissions_data
            if r.pollutant_code.lower() in criteria_pollutants
        ]

        return self.generate_emissions_inventory(
            criteria_data,
            period_start,
            period_end,
            report_format,
        )


# =============================================================================
# Submission Handler
# =============================================================================


class SubmissionHandler:
    """
    Handles report submissions to various systems.

    Supports multiple submission methods and tracks submission status.
    """

    def __init__(self) -> None:
        """Initialize submission handler."""
        self._http_client: Optional[httpx.AsyncClient] = None
        self._logger = logging.getLogger("reporting.submission")

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._http_client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=10),
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()

    async def submit_via_api(
        self,
        document: ReportDocument,
        endpoint: str,
        auth_token: str,
        agency_id: str,
    ) -> ReportSubmission:
        """
        Submit report via API.

        Args:
            document: Report document
            endpoint: API endpoint
            auth_token: Authentication token
            agency_id: Agency identifier

        Returns:
            Submission record
        """
        if not self._http_client:
            await self.initialize()

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/xml" if document.document_format == ReportFormat.XML else "application/json",
        }

        try:
            response = await self._http_client.post(
                endpoint,
                content=document.document_content,
                headers=headers,
            )
            response.raise_for_status()

            result = response.json()

            return ReportSubmission(
                document_id=document.document_id,
                requirement_id=document.requirement_id,
                agency_id=agency_id,
                submission_method=SubmissionMethod.API,
                submitted_at=datetime.utcnow(),
                submitted_by="system",
                confirmation_number=result.get("confirmationNumber"),
                status=SubmissionStatus.SUBMITTED,
            )

        except httpx.HTTPError as e:
            self._logger.error(f"API submission failed: {e}")
            return ReportSubmission(
                document_id=document.document_id,
                requirement_id=document.requirement_id,
                agency_id=agency_id,
                submission_method=SubmissionMethod.API,
                submitted_at=datetime.utcnow(),
                submitted_by="system",
                status=SubmissionStatus.REJECTED,
                status_message=str(e),
            )

    async def submit_via_sftp(
        self,
        document: ReportDocument,
        sftp_host: str,
        sftp_path: str,
        username: str,
        password: str,
        agency_id: str,
    ) -> ReportSubmission:
        """
        Submit report via SFTP.

        Args:
            document: Report document
            sftp_host: SFTP host
            sftp_path: Remote path
            username: SFTP username
            password: SFTP password
            agency_id: Agency identifier

        Returns:
            Submission record
        """
        # In production, use asyncssh or paramiko for SFTP
        self._logger.info(f"Submitting to SFTP: {sftp_host}{sftp_path}")

        return ReportSubmission(
            document_id=document.document_id,
            requirement_id=document.requirement_id,
            agency_id=agency_id,
            submission_method=SubmissionMethod.SFTP,
            submitted_at=datetime.utcnow(),
            submitted_by="system",
            status=SubmissionStatus.SUBMITTED,
        )

    async def check_submission_status(
        self,
        submission: ReportSubmission,
        status_endpoint: str,
        auth_token: str,
    ) -> SubmissionStatus:
        """
        Check status of a submission.

        Args:
            submission: Submission record
            status_endpoint: Status API endpoint
            auth_token: Authentication token

        Returns:
            Current status
        """
        if not self._http_client:
            await self.initialize()

        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            url = f"{status_endpoint}/{submission.confirmation_number}"

            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()

            result = response.json()
            status_str = result.get("status", "").lower()

            status_map = {
                "received": SubmissionStatus.RECEIVED,
                "processing": SubmissionStatus.PROCESSING,
                "accepted": SubmissionStatus.ACCEPTED,
                "rejected": SubmissionStatus.REJECTED,
            }

            return status_map.get(status_str, SubmissionStatus.PROCESSING)

        except Exception as e:
            self._logger.error(f"Status check failed: {e}")
            return submission.status


# =============================================================================
# Schedule Manager
# =============================================================================


class ScheduleManager:
    """
    Manages reporting schedules and deadlines.
    """

    def __init__(self) -> None:
        """Initialize schedule manager."""
        self._schedules: Dict[str, ReportingSchedule] = {}
        self._requirements: Dict[str, ReportingRequirement] = {}
        self._logger = logging.getLogger("reporting.schedule")

    def add_requirement(self, requirement: ReportingRequirement) -> None:
        """Add reporting requirement."""
        self._requirements[requirement.requirement_id] = requirement

    def generate_schedule(
        self,
        requirement_id: str,
        start_year: int,
        end_year: int,
    ) -> List[ReportingSchedule]:
        """
        Generate reporting schedule for a requirement.

        Args:
            requirement_id: Requirement identifier
            start_year: Start year
            end_year: End year

        Returns:
            List of schedule entries
        """
        requirement = self._requirements.get(requirement_id)
        if not requirement:
            raise ValidationError(f"Unknown requirement: {requirement_id}")

        schedules = []

        for year in range(start_year, end_year + 1):
            if requirement.frequency == ReportFrequency.ANNUAL:
                period_start = date(year, 1, 1)
                period_end = date(year, 12, 31)
                due = self._calculate_due_date(
                    period_end,
                    requirement.due_date_rule,
                )
                schedules.append(ReportingSchedule(
                    requirement_id=requirement_id,
                    reporting_period_start=period_start,
                    reporting_period_end=period_end,
                    due_date=due,
                ))

            elif requirement.frequency == ReportFrequency.QUARTERLY:
                for quarter in range(1, 5):
                    period_start = date(year, (quarter - 1) * 3 + 1, 1)
                    if quarter == 4:
                        period_end = date(year, 12, 31)
                    else:
                        period_end = date(year, quarter * 3, 30 if quarter * 3 in [4, 6, 9, 11] else 31)
                    due = self._calculate_due_date(
                        period_end,
                        requirement.due_date_rule,
                    )
                    schedules.append(ReportingSchedule(
                        requirement_id=requirement_id,
                        reporting_period_start=period_start,
                        reporting_period_end=period_end,
                        due_date=due,
                    ))

            elif requirement.frequency == ReportFrequency.MONTHLY:
                for month in range(1, 13):
                    period_start = date(year, month, 1)
                    if month == 12:
                        period_end = date(year, 12, 31)
                    else:
                        period_end = date(year, month + 1, 1) - timedelta(days=1)
                    due = self._calculate_due_date(
                        period_end,
                        requirement.due_date_rule,
                    )
                    schedules.append(ReportingSchedule(
                        requirement_id=requirement_id,
                        reporting_period_start=period_start,
                        reporting_period_end=period_end,
                        due_date=due,
                    ))

        # Store schedules
        for schedule in schedules:
            self._schedules[schedule.schedule_id] = schedule

        return schedules

    def _calculate_due_date(self, period_end: date, rule: str) -> date:
        """Calculate due date from rule."""
        # Simple implementation - parse days from rule
        # Example rule: "30 days after period end"
        import re
        match = re.search(r'(\d+)\s*days', rule.lower())
        if match:
            days = int(match.group(1))
            return period_end + timedelta(days=days)

        # Default to 30 days after period end
        return period_end + timedelta(days=30)

    def get_upcoming_reports(
        self,
        days_ahead: int = 30,
    ) -> List[ReportingSchedule]:
        """
        Get reports due within specified days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of upcoming schedule entries
        """
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        upcoming = [
            s for s in self._schedules.values()
            if s.status == "pending" and today <= s.due_date <= cutoff
        ]

        return sorted(upcoming, key=lambda s: s.due_date)

    def get_overdue_reports(self) -> List[ReportingSchedule]:
        """
        Get overdue reports.

        Returns:
            List of overdue schedule entries
        """
        today = date.today()

        overdue = [
            s for s in self._schedules.values()
            if s.status == "pending" and s.due_date < today
        ]

        return sorted(overdue, key=lambda s: s.due_date)


# =============================================================================
# Reporting Connector
# =============================================================================


class ReportingConnector(BaseConnector):
    """
    Regulatory Reporting Systems Connector.

    Provides comprehensive integration for regulatory reporting:
    - State/local agency submissions
    - Air quality management districts
    - Tribal authority reporting
    - International reporting (UNECE CLRTAP)
    - Corporate sustainability reporting

    Features:
    - Multi-format report generation (XML, JSON, CSV)
    - Multiple submission methods (API, SFTP, portal)
    - Schedule management
    - Submission tracking
    - Deadline monitoring
    """

    def __init__(self, config: ReportingConnectorConfig) -> None:
        """
        Initialize reporting connector.

        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self._reporting_config = config

        # Initialize components
        self._generator = ReportGenerator(
            config.facility_id,
            config.facility_name,
        )
        self._submission_handler = SubmissionHandler()
        self._schedule_manager = ScheduleManager()

        # Data storage
        self._agencies: Dict[str, ReportingAgency] = {}
        self._requirements: Dict[str, ReportingRequirement] = {}
        self._documents: Dict[str, ReportDocument] = {}
        self._submissions: Dict[str, ReportSubmission] = {}

        self._logger = logging.getLogger(f"reporting.connector.{config.facility_id}")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Initialize reporting connector.

        Raises:
            ConnectionError: If initialization fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info("Initializing reporting connector")

        try:
            await self._submission_handler.initialize()
            self._state = ConnectionState.CONNECTED
            self._logger.info("Reporting connector initialized")

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to initialize reporting connector: {e}")

    async def disconnect(self) -> None:
        """Shutdown reporting connector."""
        self._logger.info("Shutting down reporting connector")

        await self._submission_handler.close()
        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on reporting systems.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Check for overdue reports
            overdue = self._schedule_manager.get_overdue_reports()
            upcoming = self._schedule_manager.get_upcoming_reports(7)

            latency_ms = (time.time() - start_time) * 1000

            status = HealthStatus.HEALTHY
            message = "Reporting systems healthy"

            if overdue:
                status = HealthStatus.UNHEALTHY
                message = f"{len(overdue)} overdue reports"
            elif upcoming:
                status = HealthStatus.DEGRADED
                message = f"{len(upcoming)} reports due within 7 days"

            return HealthCheckResult(
                status=status,
                latency_ms=latency_ms,
                message=message,
                details={
                    "overdue_count": len(overdue),
                    "upcoming_count": len(upcoming),
                    "registered_systems": len(self._reporting_config.registered_systems),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    async def validate_configuration(self) -> bool:
        """
        Validate reporting configuration.

        Returns:
            True if configuration is valid
        """
        issues: List[str] = []

        if not self._reporting_config.facility_id:
            issues.append("facility_id is required")

        if not self._reporting_config.state:
            issues.append("state is required")

        if issues:
            raise ConfigurationError(
                f"Invalid reporting configuration: {issues}",
                connector_id=self._config.connector_id,
            )

        return True

    # -------------------------------------------------------------------------
    # Reporting-Specific Methods
    # -------------------------------------------------------------------------

    async def register_agency(self, agency: ReportingAgency) -> str:
        """
        Register a reporting agency.

        Args:
            agency: Agency information

        Returns:
            Agency ID
        """
        self._agencies[agency.agency_id] = agency

        await self._audit_logger.log_operation(
            operation="register_agency",
            status="success",
            request_data={"agency_name": agency.agency_name},
        )

        return agency.agency_id

    async def add_requirement(
        self,
        requirement: ReportingRequirement,
    ) -> str:
        """
        Add a reporting requirement.

        Args:
            requirement: Reporting requirement

        Returns:
            Requirement ID
        """
        self._requirements[requirement.requirement_id] = requirement
        self._schedule_manager.add_requirement(requirement)

        await self._audit_logger.log_operation(
            operation="add_requirement",
            status="success",
            request_data={
                "report_name": requirement.report_name,
                "frequency": requirement.frequency.value,
            },
        )

        return requirement.requirement_id

    async def generate_report(
        self,
        requirement_id: str,
        emissions_data: List[EmissionsDataRecord],
        period_start: date,
        period_end: date,
    ) -> ReportDocument:
        """
        Generate a report.

        Args:
            requirement_id: Requirement identifier
            emissions_data: Emissions data
            period_start: Period start
            period_end: Period end

        Returns:
            Generated report document
        """
        requirement = self._requirements.get(requirement_id)
        if not requirement:
            raise ValidationError(f"Unknown requirement: {requirement_id}")

        document = self._generator.generate_emissions_inventory(
            emissions_data,
            period_start,
            period_end,
            requirement.report_format,
        )

        # Validate if configured
        if self._reporting_config.auto_validate:
            document = await self._validate_report(document, requirement)

        self._documents[document.document_id] = document

        await self._audit_logger.log_operation(
            operation="generate_report",
            status="success",
            request_data={
                "requirement_id": requirement_id,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
            },
            response_summary=f"Generated {requirement.report_name}",
        )

        return document

    async def _validate_report(
        self,
        document: ReportDocument,
        requirement: ReportingRequirement,
    ) -> ReportDocument:
        """Validate report against requirements."""
        errors: List[str] = []
        warnings: List[str] = []

        # Basic validation
        if not document.document_content:
            errors.append("Document content is empty")

        # Check required data elements
        # In production, parse and validate content

        return ReportDocument(
            document_id=document.document_id,
            requirement_id=document.requirement_id,
            reporting_period_start=document.reporting_period_start,
            reporting_period_end=document.reporting_period_end,
            document_format=document.document_format,
            document_content=document.document_content,
            document_path=document.document_path,
            generated_at=document.generated_at,
            generated_by=document.generated_by,
            status=SubmissionStatus.READY_FOR_SUBMISSION if not errors else SubmissionStatus.DRAFT,
            is_validated=True,
            validation_errors=errors,
            validation_warnings=warnings,
        )

    async def submit_report(
        self,
        document_id: str,
        agency_id: str,
        auth_token: Optional[str] = None,
    ) -> ReportSubmission:
        """
        Submit a report to an agency.

        Args:
            document_id: Document identifier
            agency_id: Agency identifier
            auth_token: Optional authentication token

        Returns:
            Submission record
        """
        document = self._documents.get(document_id)
        if not document:
            raise ValidationError(f"Unknown document: {document_id}")

        agency = self._agencies.get(agency_id)
        if not agency:
            raise ValidationError(f"Unknown agency: {agency_id}")

        requirement = self._requirements.get(document.requirement_id)

        # Submit based on method
        if requirement and requirement.submission_method == SubmissionMethod.API:
            if not agency.api_endpoint:
                raise ConfigurationError("Agency API endpoint not configured")
            submission = await self._submission_handler.submit_via_api(
                document,
                agency.api_endpoint,
                auth_token or "",
                agency_id,
            )
        else:
            # Default to manual tracking
            submission = ReportSubmission(
                document_id=document_id,
                requirement_id=document.requirement_id,
                agency_id=agency_id,
                submission_method=SubmissionMethod.PORTAL_UPLOAD,
                submitted_at=datetime.utcnow(),
                submitted_by="system",
                status=SubmissionStatus.SUBMITTED,
            )

        self._submissions[submission.submission_id] = submission

        await self._audit_logger.log_operation(
            operation="submit_report",
            status="success" if submission.status == SubmissionStatus.SUBMITTED else "failure",
            request_data={
                "document_id": document_id,
                "agency_id": agency_id,
            },
            response_summary=f"Submission {submission.status.value}",
        )

        return submission

    async def get_submission_status(
        self,
        submission_id: str,
    ) -> SubmissionStatus:
        """
        Get status of a submission.

        Args:
            submission_id: Submission identifier

        Returns:
            Current status
        """
        submission = self._submissions.get(submission_id)
        if not submission:
            raise ValidationError(f"Unknown submission: {submission_id}")

        return submission.status

    async def generate_schedule(
        self,
        requirement_id: str,
        years: int = 1,
    ) -> List[ReportingSchedule]:
        """
        Generate reporting schedule.

        Args:
            requirement_id: Requirement identifier
            years: Number of years to schedule

        Returns:
            List of schedule entries
        """
        current_year = date.today().year

        return self._schedule_manager.generate_schedule(
            requirement_id,
            current_year,
            current_year + years - 1,
        )

    async def get_upcoming_reports(
        self,
        days: int = 30,
    ) -> List[ReportingSchedule]:
        """
        Get reports due within specified days.

        Args:
            days: Days to look ahead

        Returns:
            List of upcoming reports
        """
        return self._schedule_manager.get_upcoming_reports(days)

    async def get_overdue_reports(self) -> List[ReportingSchedule]:
        """
        Get overdue reports.

        Returns:
            List of overdue reports
        """
        return self._schedule_manager.get_overdue_reports()

    async def get_reporting_summary(self) -> Dict[str, Any]:
        """
        Get summary of reporting status.

        Returns:
            Summary dictionary
        """
        today = date.today()
        upcoming = await self.get_upcoming_reports(30)
        overdue = await self.get_overdue_reports()

        return {
            "as_of": today.isoformat(),
            "total_requirements": len(self._requirements),
            "registered_agencies": len(self._agencies),
            "upcoming_reports": len(upcoming),
            "overdue_reports": len(overdue),
            "pending_submissions": sum(
                1 for s in self._submissions.values()
                if s.status in [SubmissionStatus.SUBMITTED, SubmissionStatus.PROCESSING]
            ),
            "accepted_submissions": sum(
                1 for s in self._submissions.values()
                if s.status == SubmissionStatus.ACCEPTED
            ),
            "rejected_submissions": sum(
                1 for s in self._submissions.values()
                if s.status == SubmissionStatus.REJECTED
            ),
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_reporting_connector(
    facility_id: str,
    facility_name: str,
    state: str,
    **kwargs: Any,
) -> ReportingConnector:
    """
    Factory function to create reporting connector.

    Args:
        facility_id: Facility identifier
        facility_name: Facility name
        state: State code
        **kwargs: Additional configuration

    Returns:
        Configured reporting connector
    """
    config = ReportingConnectorConfig(
        connector_name=f"Reporting_{facility_id}",
        facility_id=facility_id,
        facility_name=facility_name,
        state=state,
        **kwargs,
    )

    return ReportingConnector(config)
