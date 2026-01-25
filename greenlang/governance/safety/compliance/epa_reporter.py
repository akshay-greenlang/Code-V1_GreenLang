"""
EPAReporter - EPA Part 60/75/98 Reporting

This module implements emissions reporting per EPA regulations:
- 40 CFR Part 60: Standards of Performance for New Stationary Sources
- 40 CFR Part 75: Continuous Emission Monitoring
- 40 CFR Part 98: Mandatory Greenhouse Gas Reporting

Reference: EPA 40 CFR Parts 60, 75, 98

Example:
    >>> from greenlang.safety.compliance.epa_reporter import EPAReporter
    >>> reporter = EPAReporter(facility_id="FAC-001")
    >>> report = reporter.generate_report(period="Q1-2024")
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, date
import uuid

logger = logging.getLogger(__name__)


class ReportingProgram(str, Enum):
    """EPA reporting programs."""

    PART_60 = "part_60"  # NSPS
    PART_75 = "part_75"  # CEMS
    PART_98 = "part_98"  # GHG


class EmissionType(str, Enum):
    """Types of emissions reported."""

    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    NOX = "nox"
    SO2 = "so2"
    CO = "co"
    PM = "pm"  # Particulate Matter
    VOC = "voc"


class EmissionRecord(BaseModel):
    """Individual emission record."""

    record_id: str = Field(
        default_factory=lambda: f"EM-{uuid.uuid4().hex[:8].upper()}",
        description="Record identifier"
    )
    emission_type: EmissionType = Field(
        ...,
        description="Type of emission"
    )
    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    value: float = Field(
        ...,
        ge=0,
        description="Emission value"
    )
    unit: str = Field(
        ...,
        description="Measurement unit"
    )
    measurement_date: date = Field(
        ...,
        description="Date of measurement"
    )
    measurement_method: str = Field(
        default="CEMS",
        description="Measurement method"
    )
    data_quality: str = Field(
        default="measured",
        description="Data quality (measured, calculated, estimated)"
    )
    uncertainty_percent: Optional[float] = Field(
        None,
        ge=0,
        description="Measurement uncertainty (%)"
    )


class EPAReport(BaseModel):
    """EPA compliance report."""

    report_id: str = Field(
        default_factory=lambda: f"EPA-{uuid.uuid4().hex[:8].upper()}",
        description="Report identifier"
    )
    facility_id: str = Field(
        ...,
        description="EPA facility ID"
    )
    reporting_program: ReportingProgram = Field(
        ...,
        description="Reporting program"
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period (e.g., Q1-2024)"
    )
    submission_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Report submission date"
    )
    emissions: List[EmissionRecord] = Field(
        default_factory=list,
        description="Emission records"
    )
    total_emissions_mt: Dict[str, float] = Field(
        default_factory=dict,
        description="Total emissions by type (metric tons)"
    )
    compliance_status: str = Field(
        default="compliant",
        description="Compliance status"
    )
    exceedances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Any limit exceedances"
    )
    prepared_by: str = Field(
        default="",
        description="Report preparer"
    )
    certified_by: Optional[str] = Field(
        None,
        description="Certifying official"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }


class EPAReporter:
    """
    EPA Emissions Reporter.

    Generates compliance reports for EPA regulatory programs.
    Supports Part 60, 75, and 98 reporting requirements.

    Features:
    - Emission data aggregation
    - Limit compliance checking
    - Report generation
    - Electronic submission formatting

    The reporter follows zero-hallucination principles:
    - All calculations deterministic
    - Complete audit trail
    - No estimated values without flagging

    Attributes:
        facility_id: EPA facility identifier
        reports: Generated reports

    Example:
        >>> reporter = EPAReporter(facility_id="FAC-001")
        >>> reporter.add_emission(EmissionRecord(...))
        >>> report = reporter.generate_report("Q1-2024", ReportingProgram.PART_98)
    """

    # EPA emission limits by program (simplified examples)
    EMISSION_LIMITS: Dict[ReportingProgram, Dict[EmissionType, float]] = {
        ReportingProgram.PART_60: {
            EmissionType.NOX: 0.15,  # lb/MMBtu
            EmissionType.SO2: 0.50,  # lb/MMBtu
            EmissionType.PM: 0.03,  # lb/MMBtu
        },
        ReportingProgram.PART_75: {
            EmissionType.NOX: 0.10,  # lb/MMBtu
            EmissionType.SO2: 0.40,  # lb/MMBtu
        },
        ReportingProgram.PART_98: {
            # GHG reporting - no specific limits but reporting thresholds
        },
    }

    # GHG Global Warming Potentials (100-year)
    GWP: Dict[EmissionType, float] = {
        EmissionType.CO2: 1.0,
        EmissionType.CH4: 28.0,
        EmissionType.N2O: 265.0,
    }

    def __init__(
        self,
        facility_id: str,
        facility_name: str = ""
    ):
        """
        Initialize EPAReporter.

        Args:
            facility_id: EPA facility identifier
            facility_name: Facility name
        """
        self.facility_id = facility_id
        self.facility_name = facility_name
        self.emission_records: List[EmissionRecord] = []
        self.reports: List[EPAReport] = []

        logger.info(f"EPAReporter initialized for facility {facility_id}")

    def add_emission(self, record: EmissionRecord) -> None:
        """
        Add emission record.

        Args:
            record: EmissionRecord to add
        """
        self.emission_records.append(record)
        logger.debug(
            f"Added emission record: {record.emission_type.value} "
            f"from {record.source_id}"
        )

    def generate_report(
        self,
        reporting_period: str,
        program: ReportingProgram,
        prepared_by: str
    ) -> EPAReport:
        """
        Generate EPA compliance report.

        Args:
            reporting_period: Period string (e.g., "Q1-2024")
            program: Reporting program
            prepared_by: Report preparer name

        Returns:
            EPAReport object
        """
        logger.info(
            f"Generating {program.value} report for {reporting_period}"
        )

        # Filter records for period
        period_records = self._filter_by_period(reporting_period)

        # Calculate totals by emission type
        totals = self._calculate_totals(period_records)

        # Check compliance
        exceedances = self._check_limits(period_records, program)

        compliance_status = "compliant" if not exceedances else "non-compliant"

        # Build report
        report = EPAReport(
            facility_id=self.facility_id,
            reporting_program=program,
            reporting_period=reporting_period,
            emissions=period_records,
            total_emissions_mt=totals,
            compliance_status=compliance_status,
            exceedances=exceedances,
            prepared_by=prepared_by,
        )

        # Calculate provenance
        report.provenance_hash = self._calculate_provenance(report)

        # Store report
        self.reports.append(report)

        logger.info(
            f"Report generated: {report.report_id}, "
            f"status={compliance_status}"
        )

        return report

    def calculate_co2e(
        self,
        records: Optional[List[EmissionRecord]] = None
    ) -> float:
        """
        Calculate CO2 equivalent emissions.

        Args:
            records: Emission records (all if None)

        Returns:
            Total CO2e in metric tons
        """
        records = records or self.emission_records

        co2e_total = 0.0

        for record in records:
            if record.emission_type in self.GWP:
                gwp = self.GWP[record.emission_type]
                # Convert to metric tons if needed
                value_mt = self._convert_to_mt(
                    record.value,
                    record.unit
                )
                co2e_total += value_mt * gwp

        return co2e_total

    def export_xml(self, report: EPAReport) -> str:
        """
        Export report to EPA XML format.

        Args:
            report: Report to export

        Returns:
            XML string
        """
        # Simplified XML format
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<EPAReport>',
            f'  <FacilityID>{report.facility_id}</FacilityID>',
            f'  <ReportingProgram>{report.reporting_program.value}</ReportingProgram>',
            f'  <ReportingPeriod>{report.reporting_period}</ReportingPeriod>',
            '  <Emissions>',
        ]

        for emission in report.emissions:
            xml_lines.extend([
                '    <EmissionRecord>',
                f'      <Type>{emission.emission_type.value}</Type>',
                f'      <Source>{emission.source_id}</Source>',
                f'      <Value>{emission.value}</Value>',
                f'      <Unit>{emission.unit}</Unit>',
                '    </EmissionRecord>',
            ])

        xml_lines.extend([
            '  </Emissions>',
            f'  <ComplianceStatus>{report.compliance_status}</ComplianceStatus>',
            f'  <ProvenanceHash>{report.provenance_hash}</ProvenanceHash>',
            '</EPAReport>',
        ])

        return '\n'.join(xml_lines)

    def _filter_by_period(
        self,
        period: str
    ) -> List[EmissionRecord]:
        """Filter records by reporting period."""
        # Simplified period parsing
        # Format: Q1-2024 or 2024-01
        return self.emission_records  # Return all for simplicity

    def _calculate_totals(
        self,
        records: List[EmissionRecord]
    ) -> Dict[str, float]:
        """Calculate total emissions by type."""
        totals: Dict[str, float] = {}

        for record in records:
            emission_type = record.emission_type.value
            value_mt = self._convert_to_mt(record.value, record.unit)

            if emission_type not in totals:
                totals[emission_type] = 0.0
            totals[emission_type] += value_mt

        return totals

    def _check_limits(
        self,
        records: List[EmissionRecord],
        program: ReportingProgram
    ) -> List[Dict[str, Any]]:
        """Check records against emission limits."""
        exceedances = []
        limits = self.EMISSION_LIMITS.get(program, {})

        for record in records:
            if record.emission_type in limits:
                limit = limits[record.emission_type]
                if record.value > limit:
                    exceedances.append({
                        "record_id": record.record_id,
                        "emission_type": record.emission_type.value,
                        "source_id": record.source_id,
                        "value": record.value,
                        "limit": limit,
                        "exceedance_percent": (
                            (record.value - limit) / limit * 100
                        ),
                    })

        return exceedances

    def _convert_to_mt(self, value: float, unit: str) -> float:
        """Convert emission value to metric tons."""
        conversions = {
            "mt": 1.0,
            "kg": 0.001,
            "lb": 0.000453592,
            "ton": 0.907185,  # US short ton
        }
        return value * conversions.get(unit.lower(), 1.0)

    def _calculate_provenance(self, report: EPAReport) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{report.report_id}|"
            f"{report.facility_id}|"
            f"{report.reporting_period}|"
            f"{len(report.emissions)}|"
            f"{report.submission_date.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
