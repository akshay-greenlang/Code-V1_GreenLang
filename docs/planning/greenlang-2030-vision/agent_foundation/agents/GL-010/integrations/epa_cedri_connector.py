"""
EPA CEDRI (Compliance and Emissions Data Reporting Interface) Connector for GL-010.

Provides integration with EPA's CEDRI system for electronic submission of
compliance and emissions reports. Supports XML report generation, digital
signatures, and submission tracking per EPA CDX requirements.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom

from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl
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


class ReportType(str, Enum):
    """EPA CEDRI report types."""

    QUARTERLY_EXCESS_EMISSIONS = "quarterly_excess_emissions"
    ANNUAL_EMISSIONS = "annual_emissions"
    SEMI_ANNUAL_REPORT = "semi_annual_report"
    DEVIATION_REPORT = "deviation_report"
    EXCESS_EMISSIONS = "excess_emissions"
    STARTUP_SHUTDOWN = "startup_shutdown"
    MALFUNCTION = "malfunction"
    PERFORMANCE_TEST = "performance_test"
    MONITORING_PLAN = "monitoring_plan"
    CERTIFICATION_TEST = "certification_test"
    ANNUAL_COMPLIANCE_CERT = "annual_compliance_certification"
    NSPS_REPORT = "nsps_report"
    NESHAP_REPORT = "neshap_report"
    MACT_REPORT = "mact_report"
    TITLE_V_REPORT = "title_v_report"


class SubmissionStatus(str, Enum):
    """Submission status codes."""

    DRAFT = "draft"
    PENDING_SIGNATURE = "pending_signature"
    SIGNED = "signed"
    SUBMITTED = "submitted"
    RECEIVED = "received"
    PROCESSING = "processing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REQUIRES_CORRECTION = "requires_correction"
    CORRECTED = "corrected"
    WITHDRAWN = "withdrawn"


class CDXEnvironment(str, Enum):
    """EPA CDX environment types."""

    PRODUCTION = "production"
    TEST = "test"
    DEVELOPMENT = "development"


class PollutantCode(str, Enum):
    """EPA pollutant codes."""

    NOX = "NOX"  # Nitrogen Oxides
    SO2 = "SO2"  # Sulfur Dioxide
    CO = "CO"  # Carbon Monoxide
    CO2 = "CO2"  # Carbon Dioxide
    PM = "PM"  # Particulate Matter
    PM10 = "PM10"  # PM10
    PM25 = "PM2.5"  # PM2.5
    VOC = "VOC"  # Volatile Organic Compounds
    HAP = "HAP"  # Hazardous Air Pollutants
    NH3 = "NH3"  # Ammonia
    HCL = "HCL"  # Hydrogen Chloride
    HF = "HF"  # Hydrogen Fluoride
    PB = "PB"  # Lead
    HG = "HG"  # Mercury
    O3 = "O3"  # Ozone


class UnitMeasureCode(str, Enum):
    """EPA unit of measure codes."""

    LBS = "LBS"  # Pounds
    TONS = "TONS"  # Short tons
    LBS_HR = "LBS/HR"  # Pounds per hour
    LBS_MMBTU = "LBS/MMBTU"  # Pounds per million BTU
    PPM = "PPM"  # Parts per million
    PERCENT = "PCT"  # Percent
    GR_DSCF = "GR/DSCF"  # Grains per dry standard cubic foot
    MG_M3 = "MG/M3"  # Milligrams per cubic meter


class RegulatoryProgram(str, Enum):
    """EPA regulatory programs."""

    NSPS = "nsps"  # New Source Performance Standards
    NESHAP = "neshap"  # National Emission Standards for HAPs
    MACT = "mact"  # Maximum Achievable Control Technology
    TITLE_V = "title_v"  # Title V Operating Permits
    PSD = "psd"  # Prevention of Significant Deterioration
    RACT = "ract"  # Reasonably Available Control Technology
    BACT = "bact"  # Best Available Control Technology
    LAER = "laer"  # Lowest Achievable Emission Rate
    ACID_RAIN = "acid_rain"  # Acid Rain Program
    CROSS_STATE = "cross_state"  # Cross-State Air Pollution Rule


# =============================================================================
# Pydantic Models
# =============================================================================


class FacilityIdentification(BaseModel):
    """EPA facility identification information."""

    model_config = ConfigDict(frozen=True)

    facility_name: str = Field(..., max_length=200, description="Facility name")
    facility_id: str = Field(..., description="EPA facility registry ID")
    eis_facility_id: Optional[str] = Field(
        default=None,
        description="EIS facility site ID"
    )
    oris_code: Optional[str] = Field(
        default=None,
        description="DOE/EIA ORIS plant code"
    )
    airs_id: Optional[str] = Field(
        default=None,
        description="AIRS facility system ID"
    )
    state_facility_id: Optional[str] = Field(
        default=None,
        description="State facility ID"
    )
    naics_code: str = Field(..., description="NAICS code")
    sic_code: Optional[str] = Field(default=None, description="SIC code")

    # Address
    street_address: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., min_length=2, max_length=2, description="State code")
    zip_code: str = Field(..., description="ZIP code")
    county: Optional[str] = Field(default=None, description="County")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)


class EmissionUnit(BaseModel):
    """Emission unit information."""

    model_config = ConfigDict(frozen=True)

    unit_id: str = Field(..., description="Unit identifier")
    unit_name: str = Field(..., description="Unit name")
    unit_type: str = Field(..., description="Unit type code")
    unit_description: Optional[str] = Field(default=None, description="Description")

    design_capacity: Optional[float] = Field(default=None, ge=0)
    capacity_unit: Optional[str] = Field(default=None, description="Capacity unit")

    fuel_type: Optional[str] = Field(default=None, description="Primary fuel type")
    operating_status: str = Field(default="operating", description="Operating status")

    control_devices: List[str] = Field(
        default_factory=list,
        description="Control device IDs"
    )


class EmissionsRecord(BaseModel):
    """Single emissions data record."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Record identifier"
    )
    unit_id: str = Field(..., description="Emission unit ID")
    pollutant_code: PollutantCode = Field(..., description="Pollutant code")

    # Time period
    reporting_period_start: date = Field(..., description="Period start date")
    reporting_period_end: date = Field(..., description="Period end date")

    # Emissions values
    total_emissions: float = Field(..., ge=0, description="Total emissions")
    emissions_unit: UnitMeasureCode = Field(..., description="Emissions unit")
    emission_rate: Optional[float] = Field(default=None, ge=0)
    rate_unit: Optional[UnitMeasureCode] = Field(default=None)

    # Operating data
    operating_hours: Optional[float] = Field(default=None, ge=0, le=8784)
    heat_input: Optional[float] = Field(default=None, ge=0)
    fuel_consumption: Optional[float] = Field(default=None, ge=0)
    fuel_consumption_unit: Optional[str] = Field(default=None)

    # Quality indicators
    data_quality_flag: Optional[str] = Field(default=None)
    percent_monitored: Optional[float] = Field(default=None, ge=0, le=100)
    calculation_method: Optional[str] = Field(default=None)


class DeviationRecord(BaseModel):
    """Deviation from permit conditions record."""

    model_config = ConfigDict(frozen=True)

    deviation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Deviation identifier"
    )
    unit_id: str = Field(..., description="Emission unit ID")

    deviation_type: str = Field(..., description="Type of deviation")
    deviation_start: datetime = Field(..., description="Start of deviation")
    deviation_end: Optional[datetime] = Field(default=None, description="End of deviation")
    duration_hours: Optional[float] = Field(default=None, ge=0)

    pollutant_code: Optional[PollutantCode] = Field(default=None)
    permit_limit: Optional[float] = Field(default=None)
    actual_value: Optional[float] = Field(default=None)
    excess_emissions: Optional[float] = Field(default=None, ge=0)

    cause: str = Field(..., description="Cause of deviation")
    corrective_action: str = Field(..., description="Corrective action taken")
    prevention_measures: Optional[str] = Field(default=None)

    reported_to_agency: bool = Field(default=False)
    agency_notification_date: Optional[datetime] = Field(default=None)


class ExcessEmissionsRecord(BaseModel):
    """Excess emissions event record."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event identifier"
    )
    unit_id: str = Field(..., description="Emission unit ID")

    event_type: str = Field(..., description="Event type")
    event_start: datetime = Field(..., description="Event start")
    event_end: Optional[datetime] = Field(default=None, description="Event end")
    duration_minutes: Optional[float] = Field(default=None, ge=0)

    pollutant_code: PollutantCode = Field(..., description="Pollutant")
    emission_limit: float = Field(..., description="Applicable limit")
    emission_limit_unit: UnitMeasureCode = Field(..., description="Limit unit")
    measured_value: float = Field(..., description="Measured value")
    excess_amount: float = Field(..., ge=0, description="Excess amount")

    cause_category: str = Field(..., description="Cause category")
    cause_description: str = Field(..., description="Detailed cause")
    corrective_action: str = Field(..., description="Corrective action")

    is_startup_shutdown: bool = Field(default=False)
    is_malfunction: bool = Field(default=False)
    affirmative_defense_claimed: bool = Field(default=False)


class CEDRIReport(BaseModel):
    """CEDRI report document."""

    model_config = ConfigDict(frozen=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier"
    )
    report_type: ReportType = Field(..., description="Report type")
    regulatory_program: RegulatoryProgram = Field(..., description="Regulatory program")

    # Facility information
    facility: FacilityIdentification = Field(..., description="Facility info")
    emission_units: List[EmissionUnit] = Field(
        default_factory=list,
        description="Emission units"
    )

    # Report period
    reporting_period_start: date = Field(..., description="Period start")
    reporting_period_end: date = Field(..., description="Period end")
    submission_date: Optional[datetime] = Field(default=None)

    # Report content
    emissions_records: List[EmissionsRecord] = Field(
        default_factory=list,
        description="Emissions data"
    )
    deviations: List[DeviationRecord] = Field(
        default_factory=list,
        description="Deviations"
    )
    excess_emissions: List[ExcessEmissionsRecord] = Field(
        default_factory=list,
        description="Excess emissions events"
    )

    # Certification
    responsible_official: str = Field(..., description="Responsible official name")
    responsible_official_title: str = Field(..., description="Official title")
    certification_statement: Optional[str] = Field(default=None)
    certification_date: Optional[datetime] = Field(default=None)

    # Submission tracking
    status: SubmissionStatus = Field(
        default=SubmissionStatus.DRAFT,
        description="Report status"
    )
    cdx_transaction_id: Optional[str] = Field(default=None)
    cdx_submission_id: Optional[str] = Field(default=None)

    # Comments and attachments
    comments: Optional[str] = Field(default=None)
    attachments: List[str] = Field(default_factory=list, description="Attachment paths")


class SubmissionResult(BaseModel):
    """Result of CEDRI submission."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(..., description="Submission successful")
    transaction_id: Optional[str] = Field(default=None, description="CDX transaction ID")
    submission_id: Optional[str] = Field(default=None, description="CEDRI submission ID")
    status: SubmissionStatus = Field(..., description="Submission status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    confirmation_number: Optional[str] = Field(default=None)
    receipt_url: Optional[str] = Field(default=None)

    errors: List[str] = Field(default_factory=list, description="Submission errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    validation_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="XML validation results"
    )


class CDXCredentials(BaseModel):
    """EPA CDX API credentials."""

    model_config = ConfigDict(frozen=True)

    username: str = Field(..., description="CDX username")
    password: str = Field(..., description="CDX password")  # Retrieved from vault
    api_key: Optional[str] = Field(default=None, description="API key if required")
    certificate_path: Optional[str] = Field(default=None, description="Client cert path")
    private_key_path: Optional[str] = Field(default=None, description="Private key path")


class EPACEDRIConnectorConfig(BaseConnectorConfig):
    """Configuration for EPA CEDRI connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.EPA_CEDRI,
        description="Connector type"
    )

    # CDX environment
    environment: CDXEnvironment = Field(
        default=CDXEnvironment.PRODUCTION,
        description="CDX environment"
    )

    # API endpoints
    cdx_base_url: str = Field(
        default="https://cdx.epa.gov",
        description="CDX base URL"
    )
    cedri_api_path: str = Field(
        default="/cedri/api/v1",
        description="CEDRI API path"
    )

    # Authentication
    auth_method: str = Field(
        default="naas",
        description="Authentication method (naas, cromerr)"
    )
    naas_token_url: str = Field(
        default="https://naas.epa.gov/token",
        description="NAAS token URL"
    )

    # Facility information
    facility_id: str = Field(..., description="EPA facility registry ID")
    program_id: str = Field(..., description="Program identifier")

    # Schema settings
    schema_version: str = Field(
        default="3.0",
        description="CEDRI XML schema version"
    )

    # Submission settings
    auto_validate: bool = Field(
        default=True,
        description="Auto-validate before submission"
    )
    require_signature: bool = Field(
        default=True,
        description="Require digital signature"
    )

    # Rate limiting
    requests_per_minute: int = Field(
        default=10,
        ge=1,
        le=60,
        description="API rate limit"
    )


# =============================================================================
# XML Report Generator
# =============================================================================


class CEDRIXMLGenerator:
    """
    Generates EPA CEDRI-compliant XML reports.

    Supports multiple report types and schema versions.
    """

    # XML namespaces
    NAMESPACES = {
        "cedri": "http://www.epa.gov/cedri",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }

    def __init__(self, schema_version: str = "3.0") -> None:
        """
        Initialize XML generator.

        Args:
            schema_version: CEDRI schema version
        """
        self._schema_version = schema_version
        self._logger = logging.getLogger("cedri.xml_generator")

    def generate_report_xml(self, report: CEDRIReport) -> str:
        """
        Generate XML document for CEDRI report.

        Args:
            report: CEDRI report data

        Returns:
            XML string
        """
        # Create root element with namespaces
        root = ET.Element("CEDRIReport")
        root.set("xmlns", self.NAMESPACES["cedri"])
        root.set("xmlns:xsi", self.NAMESPACES["xsi"])
        root.set("schemaVersion", self._schema_version)

        # Add header information
        self._add_header(root, report)

        # Add facility information
        self._add_facility(root, report.facility)

        # Add emission units
        units_elem = ET.SubElement(root, "EmissionUnits")
        for unit in report.emission_units:
            self._add_emission_unit(units_elem, unit)

        # Add report content based on type
        if report.report_type in [ReportType.QUARTERLY_EXCESS_EMISSIONS, ReportType.ANNUAL_EMISSIONS]:
            self._add_emissions_data(root, report)

        if report.deviations:
            self._add_deviations(root, report)

        if report.excess_emissions:
            self._add_excess_emissions(root, report)

        # Add certification
        self._add_certification(root, report)

        # Generate XML string
        xml_str = ET.tostring(root, encoding="unicode")

        # Pretty print
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split("\n") if line.strip()]
        return "\n".join(lines)

    def _add_header(self, root: ET.Element, report: CEDRIReport) -> None:
        """Add report header information."""
        header = ET.SubElement(root, "ReportHeader")

        ET.SubElement(header, "ReportID").text = report.report_id
        ET.SubElement(header, "ReportType").text = report.report_type.value
        ET.SubElement(header, "RegulatoryProgram").text = report.regulatory_program.value
        ET.SubElement(header, "ReportingPeriodStart").text = (
            report.reporting_period_start.isoformat()
        )
        ET.SubElement(header, "ReportingPeriodEnd").text = (
            report.reporting_period_end.isoformat()
        )

        if report.submission_date:
            ET.SubElement(header, "SubmissionDate").text = (
                report.submission_date.isoformat()
            )

        ET.SubElement(header, "Status").text = report.status.value

    def _add_facility(self, root: ET.Element, facility: FacilityIdentification) -> None:
        """Add facility identification information."""
        fac_elem = ET.SubElement(root, "FacilityIdentification")

        ET.SubElement(fac_elem, "FacilityName").text = facility.facility_name
        ET.SubElement(fac_elem, "FacilityID").text = facility.facility_id

        if facility.eis_facility_id:
            ET.SubElement(fac_elem, "EISFacilityID").text = facility.eis_facility_id
        if facility.oris_code:
            ET.SubElement(fac_elem, "ORISCode").text = facility.oris_code

        ET.SubElement(fac_elem, "NAICSCode").text = facility.naics_code

        # Address
        addr_elem = ET.SubElement(fac_elem, "Address")
        ET.SubElement(addr_elem, "StreetAddress").text = facility.street_address
        ET.SubElement(addr_elem, "City").text = facility.city
        ET.SubElement(addr_elem, "State").text = facility.state
        ET.SubElement(addr_elem, "ZipCode").text = facility.zip_code

        if facility.latitude and facility.longitude:
            loc_elem = ET.SubElement(fac_elem, "Location")
            ET.SubElement(loc_elem, "Latitude").text = str(facility.latitude)
            ET.SubElement(loc_elem, "Longitude").text = str(facility.longitude)

    def _add_emission_unit(self, parent: ET.Element, unit: EmissionUnit) -> None:
        """Add emission unit information."""
        unit_elem = ET.SubElement(parent, "EmissionUnit")

        ET.SubElement(unit_elem, "UnitID").text = unit.unit_id
        ET.SubElement(unit_elem, "UnitName").text = unit.unit_name
        ET.SubElement(unit_elem, "UnitType").text = unit.unit_type
        ET.SubElement(unit_elem, "OperatingStatus").text = unit.operating_status

        if unit.design_capacity:
            cap_elem = ET.SubElement(unit_elem, "DesignCapacity")
            ET.SubElement(cap_elem, "Value").text = str(unit.design_capacity)
            if unit.capacity_unit:
                ET.SubElement(cap_elem, "Unit").text = unit.capacity_unit

        if unit.fuel_type:
            ET.SubElement(unit_elem, "PrimaryFuelType").text = unit.fuel_type

        if unit.control_devices:
            ctrl_elem = ET.SubElement(unit_elem, "ControlDevices")
            for device_id in unit.control_devices:
                ET.SubElement(ctrl_elem, "ControlDeviceID").text = device_id

    def _add_emissions_data(self, root: ET.Element, report: CEDRIReport) -> None:
        """Add emissions data records."""
        if not report.emissions_records:
            return

        emissions_elem = ET.SubElement(root, "EmissionsData")

        for record in report.emissions_records:
            rec_elem = ET.SubElement(emissions_elem, "EmissionsRecord")

            ET.SubElement(rec_elem, "RecordID").text = record.record_id
            ET.SubElement(rec_elem, "UnitID").text = record.unit_id
            ET.SubElement(rec_elem, "PollutantCode").text = record.pollutant_code.value
            ET.SubElement(rec_elem, "PeriodStart").text = (
                record.reporting_period_start.isoformat()
            )
            ET.SubElement(rec_elem, "PeriodEnd").text = (
                record.reporting_period_end.isoformat()
            )

            emis_val = ET.SubElement(rec_elem, "TotalEmissions")
            ET.SubElement(emis_val, "Value").text = str(record.total_emissions)
            ET.SubElement(emis_val, "Unit").text = record.emissions_unit.value

            if record.emission_rate is not None:
                rate_elem = ET.SubElement(rec_elem, "EmissionRate")
                ET.SubElement(rate_elem, "Value").text = str(record.emission_rate)
                if record.rate_unit:
                    ET.SubElement(rate_elem, "Unit").text = record.rate_unit.value

            if record.operating_hours is not None:
                ET.SubElement(rec_elem, "OperatingHours").text = str(record.operating_hours)

            if record.heat_input is not None:
                ET.SubElement(rec_elem, "HeatInput").text = str(record.heat_input)

            if record.percent_monitored is not None:
                ET.SubElement(rec_elem, "PercentMonitored").text = (
                    str(record.percent_monitored)
                )

            if record.calculation_method:
                ET.SubElement(rec_elem, "CalculationMethod").text = record.calculation_method

    def _add_deviations(self, root: ET.Element, report: CEDRIReport) -> None:
        """Add deviation records."""
        dev_elem = ET.SubElement(root, "Deviations")

        for deviation in report.deviations:
            rec_elem = ET.SubElement(dev_elem, "DeviationRecord")

            ET.SubElement(rec_elem, "DeviationID").text = deviation.deviation_id
            ET.SubElement(rec_elem, "UnitID").text = deviation.unit_id
            ET.SubElement(rec_elem, "DeviationType").text = deviation.deviation_type
            ET.SubElement(rec_elem, "StartDateTime").text = (
                deviation.deviation_start.isoformat()
            )

            if deviation.deviation_end:
                ET.SubElement(rec_elem, "EndDateTime").text = (
                    deviation.deviation_end.isoformat()
                )

            if deviation.duration_hours is not None:
                ET.SubElement(rec_elem, "DurationHours").text = str(deviation.duration_hours)

            if deviation.pollutant_code:
                ET.SubElement(rec_elem, "PollutantCode").text = deviation.pollutant_code.value

            if deviation.permit_limit is not None:
                ET.SubElement(rec_elem, "PermitLimit").text = str(deviation.permit_limit)

            if deviation.actual_value is not None:
                ET.SubElement(rec_elem, "ActualValue").text = str(deviation.actual_value)

            if deviation.excess_emissions is not None:
                ET.SubElement(rec_elem, "ExcessEmissions").text = (
                    str(deviation.excess_emissions)
                )

            ET.SubElement(rec_elem, "Cause").text = deviation.cause
            ET.SubElement(rec_elem, "CorrectiveAction").text = deviation.corrective_action

            if deviation.prevention_measures:
                ET.SubElement(rec_elem, "PreventionMeasures").text = (
                    deviation.prevention_measures
                )

    def _add_excess_emissions(self, root: ET.Element, report: CEDRIReport) -> None:
        """Add excess emissions event records."""
        excess_elem = ET.SubElement(root, "ExcessEmissionsEvents")

        for event in report.excess_emissions:
            rec_elem = ET.SubElement(excess_elem, "ExcessEmissionsRecord")

            ET.SubElement(rec_elem, "EventID").text = event.event_id
            ET.SubElement(rec_elem, "UnitID").text = event.unit_id
            ET.SubElement(rec_elem, "EventType").text = event.event_type
            ET.SubElement(rec_elem, "StartDateTime").text = event.event_start.isoformat()

            if event.event_end:
                ET.SubElement(rec_elem, "EndDateTime").text = event.event_end.isoformat()

            if event.duration_minutes is not None:
                ET.SubElement(rec_elem, "DurationMinutes").text = str(event.duration_minutes)

            ET.SubElement(rec_elem, "PollutantCode").text = event.pollutant_code.value

            limit_elem = ET.SubElement(rec_elem, "EmissionLimit")
            ET.SubElement(limit_elem, "Value").text = str(event.emission_limit)
            ET.SubElement(limit_elem, "Unit").text = event.emission_limit_unit.value

            ET.SubElement(rec_elem, "MeasuredValue").text = str(event.measured_value)
            ET.SubElement(rec_elem, "ExcessAmount").text = str(event.excess_amount)

            ET.SubElement(rec_elem, "CauseCategory").text = event.cause_category
            ET.SubElement(rec_elem, "CauseDescription").text = event.cause_description
            ET.SubElement(rec_elem, "CorrectiveAction").text = event.corrective_action

            flags_elem = ET.SubElement(rec_elem, "EventFlags")
            ET.SubElement(flags_elem, "IsStartupShutdown").text = (
                str(event.is_startup_shutdown).lower()
            )
            ET.SubElement(flags_elem, "IsMalfunction").text = (
                str(event.is_malfunction).lower()
            )
            ET.SubElement(flags_elem, "AffirmativeDefenseClaimed").text = (
                str(event.affirmative_defense_claimed).lower()
            )

    def _add_certification(self, root: ET.Element, report: CEDRIReport) -> None:
        """Add certification information."""
        cert_elem = ET.SubElement(root, "Certification")

        ET.SubElement(cert_elem, "ResponsibleOfficial").text = report.responsible_official
        ET.SubElement(cert_elem, "Title").text = report.responsible_official_title

        if report.certification_statement:
            ET.SubElement(cert_elem, "CertificationStatement").text = (
                report.certification_statement
            )

        if report.certification_date:
            ET.SubElement(cert_elem, "CertificationDate").text = (
                report.certification_date.isoformat()
            )

    def validate_xml(self, xml_content: str) -> Tuple[bool, List[str]]:
        """
        Validate XML against CEDRI schema.

        Args:
            xml_content: XML string to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: List[str] = []

        try:
            # Parse XML
            root = ET.fromstring(xml_content)

            # Basic structure validation
            required_elements = ["ReportHeader", "FacilityIdentification", "Certification"]
            for elem_name in required_elements:
                if root.find(elem_name) is None:
                    errors.append(f"Missing required element: {elem_name}")

            # Validate header
            header = root.find("ReportHeader")
            if header is not None:
                required_header = ["ReportID", "ReportType", "ReportingPeriodStart", "ReportingPeriodEnd"]
                for elem_name in required_header:
                    if header.find(elem_name) is None:
                        errors.append(f"Missing required header element: {elem_name}")

            # In production, would use lxml to validate against XSD schema

            return len(errors) == 0, errors

        except ET.ParseError as e:
            errors.append(f"XML parse error: {e}")
            return False, errors


# =============================================================================
# CDX API Client
# =============================================================================


class CDXAPIClient:
    """
    EPA CDX API client for CEDRI submissions.

    Handles authentication, submission, and status tracking.
    """

    def __init__(
        self,
        config: EPACEDRIConnectorConfig,
        credentials: CDXCredentials,
    ) -> None:
        """
        Initialize CDX API client.

        Args:
            config: Connector configuration
            credentials: CDX credentials
        """
        self._config = config
        self._credentials = credentials
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._logger = logging.getLogger("cedri.cdx_client")

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=self._config.connection_timeout_seconds,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def authenticate(self) -> str:
        """
        Authenticate with EPA CDX using NAAS.

        Returns:
            Access token

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check if existing token is still valid
        if self._access_token and self._token_expires_at:
            if datetime.utcnow() < self._token_expires_at - timedelta(minutes=5):
                return self._access_token

        if not self._client:
            await self.initialize()

        try:
            # NAAS token request
            token_data = {
                "grant_type": "password",
                "username": self._credentials.username,
                "password": self._credentials.password,
            }

            response = await self._client.post(
                self._config.naas_token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()

            token_response = response.json()
            self._access_token = token_response["access_token"]
            expires_in = token_response.get("expires_in", 3600)
            self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            self._logger.info("CDX authentication successful")
            return self._access_token

        except httpx.HTTPStatusError as e:
            self._logger.error(f"CDX authentication failed: {e}")
            raise AuthenticationError(f"CDX authentication failed: {e.response.status_code}")
        except Exception as e:
            self._logger.error(f"CDX authentication error: {e}")
            raise AuthenticationError(f"CDX authentication error: {e}")

    async def submit_report(
        self,
        xml_content: str,
        report_type: ReportType,
        facility_id: str,
    ) -> SubmissionResult:
        """
        Submit CEDRI report to EPA.

        Args:
            xml_content: Report XML content
            report_type: Report type
            facility_id: Facility ID

        Returns:
            Submission result
        """
        await self.authenticate()

        api_url = f"{self._config.cdx_base_url}{self._config.cedri_api_path}/submit"

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/xml",
            "Accept": "application/json",
        }

        params = {
            "reportType": report_type.value,
            "facilityId": facility_id,
            "programId": self._config.program_id,
        }

        try:
            response = await self._client.post(
                api_url,
                content=xml_content.encode("utf-8"),
                headers=headers,
                params=params,
            )
            response.raise_for_status()

            result = response.json()

            return SubmissionResult(
                success=True,
                transaction_id=result.get("transactionId"),
                submission_id=result.get("submissionId"),
                status=SubmissionStatus.SUBMITTED,
                confirmation_number=result.get("confirmationNumber"),
                receipt_url=result.get("receiptUrl"),
            )

        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except Exception:
                pass

            return SubmissionResult(
                success=False,
                status=SubmissionStatus.REJECTED,
                errors=[f"HTTP {e.response.status_code}: {error_data.get('message', 'Unknown error')}"],
                validation_results=error_data.get("validationErrors", {}),
            )

        except Exception as e:
            return SubmissionResult(
                success=False,
                status=SubmissionStatus.REJECTED,
                errors=[str(e)],
            )

    async def get_submission_status(
        self,
        submission_id: str,
    ) -> SubmissionStatus:
        """
        Get status of a submission.

        Args:
            submission_id: Submission ID

        Returns:
            Current submission status
        """
        await self.authenticate()

        api_url = f"{self._config.cdx_base_url}{self._config.cedri_api_path}/status/{submission_id}"

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

        try:
            response = await self._client.get(api_url, headers=headers)
            response.raise_for_status()

            result = response.json()
            status_str = result.get("status", "unknown")

            # Map to enum
            status_map = {
                "submitted": SubmissionStatus.SUBMITTED,
                "received": SubmissionStatus.RECEIVED,
                "processing": SubmissionStatus.PROCESSING,
                "accepted": SubmissionStatus.ACCEPTED,
                "rejected": SubmissionStatus.REJECTED,
            }

            return status_map.get(status_str.lower(), SubmissionStatus.PROCESSING)

        except Exception as e:
            self._logger.error(f"Failed to get submission status: {e}")
            raise

    async def withdraw_submission(self, submission_id: str) -> bool:
        """
        Withdraw a pending submission.

        Args:
            submission_id: Submission ID

        Returns:
            True if withdrawal successful
        """
        await self.authenticate()

        api_url = f"{self._config.cdx_base_url}{self._config.cedri_api_path}/withdraw/{submission_id}"

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }

        try:
            response = await self._client.post(api_url, headers=headers)
            response.raise_for_status()
            return True

        except Exception as e:
            self._logger.error(f"Failed to withdraw submission: {e}")
            return False


# =============================================================================
# Digital Signature Handler
# =============================================================================


class DigitalSignatureHandler:
    """
    Handles digital signatures for EPA CROMERR compliance.

    Implements cross-media electronic reporting requirements.
    """

    def __init__(self, certificate_path: Optional[str] = None) -> None:
        """
        Initialize signature handler.

        Args:
            certificate_path: Path to signing certificate
        """
        self._certificate_path = certificate_path
        self._logger = logging.getLogger("cedri.signature")

    def sign_report(
        self,
        xml_content: str,
        signer_id: str,
    ) -> str:
        """
        Apply digital signature to report.

        Args:
            xml_content: XML content to sign
            signer_id: Signer identifier

        Returns:
            Signed XML content
        """
        # Generate hash of content
        content_hash = hashlib.sha256(xml_content.encode("utf-8")).hexdigest()

        # In production, would use xmlsec or similar library
        # to apply XML Digital Signature (XMLDSig)

        # Add signature element to XML
        root = ET.fromstring(xml_content)

        sig_elem = ET.SubElement(root, "DigitalSignature")
        ET.SubElement(sig_elem, "SignerID").text = signer_id
        ET.SubElement(sig_elem, "SignatureTimestamp").text = datetime.utcnow().isoformat()
        ET.SubElement(sig_elem, "ContentHash").text = content_hash
        ET.SubElement(sig_elem, "HashAlgorithm").text = "SHA-256"

        return ET.tostring(root, encoding="unicode")

    def verify_signature(self, signed_xml: str) -> bool:
        """
        Verify digital signature on report.

        Args:
            signed_xml: Signed XML content

        Returns:
            True if signature is valid
        """
        try:
            root = ET.fromstring(signed_xml)
            sig_elem = root.find("DigitalSignature")

            if sig_elem is None:
                return False

            # Verify hash
            stored_hash = sig_elem.find("ContentHash").text

            # Remove signature element and recalculate hash
            root.remove(sig_elem)
            content_without_sig = ET.tostring(root, encoding="unicode")
            calculated_hash = hashlib.sha256(content_without_sig.encode("utf-8")).hexdigest()

            return stored_hash == calculated_hash

        except Exception as e:
            self._logger.error(f"Signature verification failed: {e}")
            return False


# =============================================================================
# EPA CEDRI Connector
# =============================================================================


class EPACEDRIConnector(BaseConnector):
    """
    EPA CEDRI (Compliance and Emissions Data Reporting Interface) Connector.

    Provides integration with EPA's electronic reporting system for:
    - XML report generation (EPA schema compliant)
    - Electronic submission via EPA CDX
    - Digital signature support
    - Submission tracking and confirmation
    - Multiple report types (quarterly, annual, deviation, etc.)

    Compliance:
    - EPA CEDRI reporting requirements
    - CROMERR (Cross-Media Electronic Reporting Rule)
    - 40 CFR Part 75 and other applicable regulations
    """

    def __init__(
        self,
        config: EPACEDRIConnectorConfig,
        credentials: CDXCredentials,
    ) -> None:
        """
        Initialize EPA CEDRI connector.

        Args:
            config: Connector configuration
            credentials: CDX credentials
        """
        super().__init__(config)
        self._cedri_config = config
        self._credentials = credentials

        # Initialize components
        self._xml_generator = CEDRIXMLGenerator(config.schema_version)
        self._cdx_client = CDXAPIClient(config, credentials)
        self._signature_handler = DigitalSignatureHandler(credentials.certificate_path)

        # Submission tracking
        self._pending_submissions: Dict[str, CEDRIReport] = {}
        self._submission_history: List[SubmissionResult] = []

        self._logger = logging.getLogger(f"cedri.connector.{config.facility_id}")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connection to EPA CDX.

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info("Connecting to EPA CDX")

        try:
            await self._cdx_client.initialize()
            await self._cdx_client.authenticate()

            self._state = ConnectionState.CONNECTED
            self._logger.info("EPA CDX connection established")

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary="Connected to EPA CDX",
            )

        except AuthenticationError as e:
            self._state = ConnectionState.ERROR
            await self._audit_logger.log_operation(
                operation="connect",
                status="failure",
                error_message=str(e),
            )
            raise

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect to EPA CDX: {e}")

    async def disconnect(self) -> None:
        """Disconnect from EPA CDX."""
        self._logger.info("Disconnecting from EPA CDX")

        await self._cdx_client.close()
        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on CDX connection.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Test authentication
            await self._cdx_client.authenticate()

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="EPA CDX connection healthy",
                details={
                    "environment": self._cedri_config.environment.value,
                    "pending_submissions": len(self._pending_submissions),
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
        Validate CEDRI connector configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        issues: List[str] = []

        if not self._cedri_config.facility_id:
            issues.append("facility_id is required")

        if not self._cedri_config.program_id:
            issues.append("program_id is required")

        if not self._credentials.username or not self._credentials.password:
            issues.append("CDX credentials are required")

        if issues:
            raise ConfigurationError(
                f"Invalid CEDRI configuration: {issues}",
                connector_id=self._config.connector_id,
            )

        return True

    # -------------------------------------------------------------------------
    # CEDRI-Specific Methods
    # -------------------------------------------------------------------------

    async def generate_report(
        self,
        report_type: ReportType,
        facility: FacilityIdentification,
        period_start: date,
        period_end: date,
        emissions_data: Optional[List[EmissionsRecord]] = None,
        deviations: Optional[List[DeviationRecord]] = None,
        excess_emissions: Optional[List[ExcessEmissionsRecord]] = None,
        responsible_official: str = "",
        official_title: str = "",
        regulatory_program: RegulatoryProgram = RegulatoryProgram.TITLE_V,
        emission_units: Optional[List[EmissionUnit]] = None,
    ) -> CEDRIReport:
        """
        Generate a CEDRI report.

        Args:
            report_type: Type of report
            facility: Facility identification
            period_start: Reporting period start
            period_end: Reporting period end
            emissions_data: Emissions records
            deviations: Deviation records
            excess_emissions: Excess emissions events
            responsible_official: Official name
            official_title: Official title
            regulatory_program: Regulatory program
            emission_units: Emission unit information

        Returns:
            Generated CEDRI report
        """
        report = CEDRIReport(
            report_type=report_type,
            regulatory_program=regulatory_program,
            facility=facility,
            emission_units=emission_units or [],
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            emissions_records=emissions_data or [],
            deviations=deviations or [],
            excess_emissions=excess_emissions or [],
            responsible_official=responsible_official,
            responsible_official_title=official_title,
            status=SubmissionStatus.DRAFT,
        )

        await self._audit_logger.log_operation(
            operation="generate_report",
            status="success",
            request_data={
                "report_type": report_type.value,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
            },
            response_summary=f"Generated {report_type.value} report",
        )

        return report

    async def generate_xml(self, report: CEDRIReport) -> str:
        """
        Generate XML document for report.

        Args:
            report: CEDRI report

        Returns:
            XML string
        """
        xml_content = self._xml_generator.generate_report_xml(report)

        # Validate if configured
        if self._cedri_config.auto_validate:
            is_valid, errors = self._xml_generator.validate_xml(xml_content)
            if not is_valid:
                raise ValidationError(
                    f"XML validation failed: {errors}",
                    connector_id=self._config.connector_id,
                    details={"errors": errors},
                )

        return xml_content

    async def submit_report(
        self,
        report: CEDRIReport,
        sign: bool = True,
    ) -> SubmissionResult:
        """
        Submit report to EPA CEDRI.

        Args:
            report: Report to submit
            sign: Whether to apply digital signature

        Returns:
            Submission result

        Raises:
            ValidationError: If report validation fails
            ConnectionError: If submission fails
        """
        start_time = time.time()

        try:
            # Generate XML
            xml_content = await self.generate_xml(report)

            # Apply signature if required
            if sign and self._cedri_config.require_signature:
                xml_content = self._signature_handler.sign_report(
                    xml_content,
                    signer_id=report.responsible_official,
                )

            # Submit to CDX
            result = await self._cdx_client.submit_report(
                xml_content=xml_content,
                report_type=report.report_type,
                facility_id=report.facility.facility_id,
            )

            # Track submission
            if result.success:
                self._pending_submissions[result.submission_id] = report

            self._submission_history.append(result)

            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=result.success,
                latency_ms=duration_ms,
                error="; ".join(result.errors) if result.errors else None,
            )

            await self._audit_logger.log_operation(
                operation="submit_report",
                status="success" if result.success else "failure",
                request_data={
                    "report_id": report.report_id,
                    "report_type": report.report_type.value,
                },
                response_summary=f"Submission {result.status.value}",
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_logger.log_operation(
                operation="submit_report",
                status="failure",
                error_message=str(e),
                duration_ms=duration_ms,
            )
            raise

    async def get_submission_status(
        self,
        submission_id: str,
    ) -> SubmissionStatus:
        """
        Get current status of a submission.

        Args:
            submission_id: Submission ID

        Returns:
            Current status
        """
        status = await self._cdx_client.get_submission_status(submission_id)

        await self._audit_logger.log_operation(
            operation="get_submission_status",
            status="success",
            request_data={"submission_id": submission_id},
            response_summary=f"Status: {status.value}",
        )

        return status

    async def track_pending_submissions(self) -> Dict[str, SubmissionStatus]:
        """
        Check status of all pending submissions.

        Returns:
            Dictionary of submission ID to status
        """
        statuses: Dict[str, SubmissionStatus] = {}

        for submission_id in list(self._pending_submissions.keys()):
            try:
                status = await self.get_submission_status(submission_id)
                statuses[submission_id] = status

                # Remove from pending if final status
                if status in [SubmissionStatus.ACCEPTED, SubmissionStatus.REJECTED]:
                    del self._pending_submissions[submission_id]

            except Exception as e:
                self._logger.error(f"Failed to check status for {submission_id}: {e}")

        return statuses

    async def withdraw_submission(self, submission_id: str) -> bool:
        """
        Withdraw a pending submission.

        Args:
            submission_id: Submission ID

        Returns:
            True if withdrawal successful
        """
        result = await self._cdx_client.withdraw_submission(submission_id)

        if result and submission_id in self._pending_submissions:
            del self._pending_submissions[submission_id]

        await self._audit_logger.log_operation(
            operation="withdraw_submission",
            status="success" if result else "failure",
            request_data={"submission_id": submission_id},
        )

        return result

    async def get_submission_history(
        self,
        limit: int = 100,
    ) -> List[SubmissionResult]:
        """
        Get submission history.

        Args:
            limit: Maximum number of records

        Returns:
            List of submission results
        """
        return self._submission_history[-limit:]

    def validate_report_data(
        self,
        report: CEDRIReport,
    ) -> Tuple[bool, List[str]]:
        """
        Validate report data before submission.

        Args:
            report: Report to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues: List[str] = []

        # Check required fields
        if not report.facility.facility_id:
            issues.append("Facility ID is required")

        if not report.responsible_official:
            issues.append("Responsible official is required")

        # Check reporting period
        if report.reporting_period_end < report.reporting_period_start:
            issues.append("Reporting period end must be after start")

        # Check emissions data for quarterly/annual reports
        if report.report_type in [
            ReportType.QUARTERLY_EXCESS_EMISSIONS,
            ReportType.ANNUAL_EMISSIONS,
        ]:
            if not report.emissions_records and not report.excess_emissions:
                issues.append("Emissions data required for this report type")

        # Validate emissions records
        for record in report.emissions_records:
            if record.total_emissions < 0:
                issues.append(f"Negative emissions value in record {record.record_id}")

        return len(issues) == 0, issues


# =============================================================================
# Factory Function
# =============================================================================


def create_cedri_connector(
    facility_id: str,
    program_id: str,
    username: str,
    password: str,
    environment: CDXEnvironment = CDXEnvironment.PRODUCTION,
    **kwargs: Any,
) -> EPACEDRIConnector:
    """
    Factory function to create EPA CEDRI connector.

    Args:
        facility_id: EPA facility registry ID
        program_id: Program identifier
        username: CDX username
        password: CDX password
        environment: CDX environment
        **kwargs: Additional configuration

    Returns:
        Configured CEDRI connector
    """
    config = EPACEDRIConnectorConfig(
        connector_name=f"CEDRI_{facility_id}",
        facility_id=facility_id,
        program_id=program_id,
        environment=environment,
        **kwargs,
    )

    credentials = CDXCredentials(
        username=username,
        password=password,
    )

    return EPACEDRIConnector(config, credentials)
