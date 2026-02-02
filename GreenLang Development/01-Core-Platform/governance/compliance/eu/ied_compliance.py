"""
IEDCompliance - EU Industrial Emissions Directive Compliance Module

This module implements comprehensive compliance support for the EU Industrial
Emissions Directive (2010/75/EU) including:
- Best Available Techniques (BAT) reference and assessment
- Emission Limit Value (ELV) tracking and monitoring
- Permit condition monitoring and compliance
- Annual reporting templates and data collection
- BREF (BAT Reference Document) integration
- Derogation management

The Industrial Emissions Directive is the main EU instrument regulating
pollutant emissions from industrial installations.

Reference: Directive 2010/75/EU of the European Parliament and of the Council

Example:
    >>> from greenlang.compliance.eu.ied_compliance import IEDComplianceManager
    >>> manager = IEDComplianceManager(installation_id="INST-001")
    >>> result = manager.assess_compliance(current_emissions)
    >>> print(f"Compliance Status: {result.compliance_status}")

Author: GreenLang Regulatory Intelligence Team
Version: 1.0
Date: 2025-12-07
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import logging
from datetime import datetime, date
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class IEDAnnexIActivity(str, Enum):
    """IED Annex I activity categories."""
    ENERGY_1_1 = "1.1"  # Combustion > 50 MW
    ENERGY_1_2 = "1.2"  # Refining
    ENERGY_1_3 = "1.3"  # Coke production
    ENERGY_1_4 = "1.4"  # Gasification/liquefaction
    METALS_2_1 = "2.1"  # Metal ore roasting
    METALS_2_2 = "2.2"  # Pig iron or steel production
    METALS_2_3 = "2.3"  # Ferrous metals processing
    METALS_2_4 = "2.4"  # Ferrous metal foundries
    METALS_2_5 = "2.5"  # Non-ferrous metals
    METALS_2_6 = "2.6"  # Surface treatment of metals
    MINERALS_3_1 = "3.1"  # Cement, lime, magnesium oxide
    MINERALS_3_3 = "3.3"  # Glass manufacturing
    MINERALS_3_4 = "3.4"  # Mineral fibres
    MINERALS_3_5 = "3.5"  # Ceramics
    CHEMICALS_4_1 = "4.1"  # Organic chemicals
    CHEMICALS_4_2 = "4.2"  # Inorganic chemicals
    CHEMICALS_4_3 = "4.3"  # Phosphorous fertilizers
    CHEMICALS_4_4 = "4.4"  # Plant protection products
    CHEMICALS_4_5 = "4.5"  # Pharmaceutical products
    CHEMICALS_4_6 = "4.6"  # Explosives
    WASTE_5_1 = "5.1"  # Hazardous waste disposal
    WASTE_5_2 = "5.2"  # Non-hazardous waste disposal
    WASTE_5_3 = "5.3"  # Non-hazardous waste incineration
    WASTE_5_4 = "5.4"  # Landfills
    OTHER_6_1 = "6.1"  # Pulp production
    OTHER_6_2 = "6.2"  # Paper production
    OTHER_6_4 = "6.4"  # Food and drink
    OTHER_6_5 = "6.5"  # Animal carcasses
    OTHER_6_6 = "6.6"  # Intensive poultry/pig
    OTHER_6_7 = "6.7"  # Surface treatment using solvents
    OTHER_6_8 = "6.8"  # Carbon or electrographite
    OTHER_6_9 = "6.9"  # CO2 capture
    OTHER_6_10 = "6.10"  # Wood preservation
    OTHER_6_11 = "6.11"  # Independently operated treatment


class ComplianceStatus(str, Enum):
    """Installation compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_ASSESSMENT = "pending_assessment"
    UNDER_DEROGATION = "under_derogation"
    TRANSITIONAL = "transitional"


class MonitoringFrequency(str, Enum):
    """Emission monitoring frequency."""
    CONTINUOUS = "continuous"
    PERIODIC_MONTHLY = "monthly"
    PERIODIC_QUARTERLY = "quarterly"
    PERIODIC_ANNUAL = "annual"


class PollutantCategory(str, Enum):
    """Pollutant categories."""
    AIR = "air"
    WATER = "water"
    SOIL = "soil"


# =============================================================================
# Data Models
# =============================================================================

class BATConclusion(BaseModel):
    """BAT Conclusion specification per IED Article 13."""

    bat_id: str = Field(
        default_factory=lambda: f"BAT-{uuid.uuid4().hex[:6].upper()}",
        description="BAT identifier"
    )
    bref_document: str = Field(
        ...,
        description="BREF document reference"
    )
    technique_number: str = Field(
        ...,
        description="BAT technique number"
    )
    technique_description: str = Field(
        ...,
        description="Description of the technique"
    )
    applicability: str = Field(
        default="General",
        description="Applicability conditions"
    )
    environmental_performance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental performance levels"
    )
    is_mandatory: bool = Field(
        default=True,
        description="Is this BAT mandatory"
    )


class BATAEL(BaseModel):
    """BAT Associated Emission Level (BAT-AEL) per IED Article 15."""

    ael_id: str = Field(
        default_factory=lambda: f"AEL-{uuid.uuid4().hex[:6].upper()}",
        description="AEL identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant name"
    )
    category: PollutantCategory = Field(
        ...,
        description="Pollutant category (air/water/soil)"
    )
    lower_limit: float = Field(
        ...,
        description="Lower BAT-AEL value"
    )
    upper_limit: float = Field(
        ...,
        description="Upper BAT-AEL value"
    )
    unit: str = Field(
        default="mg/Nm3",
        description="Unit of measurement"
    )
    reference_conditions: str = Field(
        default="Dry gas, 273K, 101.3kPa, 3% O2",
        description="Reference conditions"
    )
    averaging_period: str = Field(
        default="Daily average",
        description="Averaging period"
    )
    bref_reference: str = Field(
        default="",
        description="BREF document reference"
    )


class EmissionLimitValue(BaseModel):
    """Emission Limit Value (ELV) from permit."""

    elv_id: str = Field(
        default_factory=lambda: f"ELV-{uuid.uuid4().hex[:6].upper()}",
        description="ELV identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant name"
    )
    limit_value: float = Field(
        ...,
        description="Emission limit value"
    )
    unit: str = Field(
        default="mg/Nm3",
        description="Unit of measurement"
    )
    source: str = Field(
        default="Permit",
        description="Source of limit (Permit, BAT-AEL, National)"
    )
    monitoring_frequency: MonitoringFrequency = Field(
        default=MonitoringFrequency.CONTINUOUS,
        description="Required monitoring frequency"
    )
    reference_conditions: str = Field(
        default="",
        description="Reference conditions"
    )
    effective_date: Optional[date] = Field(
        None,
        description="Date limit becomes effective"
    )


class EmissionMeasurement(BaseModel):
    """Individual emission measurement record."""

    measurement_id: str = Field(
        default_factory=lambda: f"MEAS-{uuid.uuid4().hex[:8].upper()}",
        description="Measurement identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant measured"
    )
    measured_value: float = Field(
        ...,
        description="Measured emission value"
    )
    unit: str = Field(
        default="mg/Nm3",
        description="Unit of measurement"
    )
    measurement_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Measurement timestamp"
    )
    source_point: str = Field(
        default="",
        description="Emission source point"
    )
    averaging_period: str = Field(
        default="",
        description="Averaging period"
    )
    reference_conditions: str = Field(
        default="",
        description="Reference conditions applied"
    )
    measurement_method: str = Field(
        default="",
        description="Measurement method used"
    )
    uncertainty: Optional[float] = Field(
        None,
        description="Measurement uncertainty (%)"
    )


class PermitCondition(BaseModel):
    """Permit condition specification."""

    condition_id: str = Field(
        default_factory=lambda: f"COND-{uuid.uuid4().hex[:6].upper()}",
        description="Condition identifier"
    )
    condition_type: str = Field(
        ...,
        description="Type of condition"
    )
    description: str = Field(
        ...,
        description="Condition description"
    )
    ied_article: str = Field(
        default="",
        description="IED Article reference"
    )
    compliance_requirement: str = Field(
        default="",
        description="Compliance requirement"
    )
    verification_method: str = Field(
        default="",
        description="How compliance is verified"
    )
    frequency: str = Field(
        default="",
        description="Compliance check frequency"
    )
    is_compliant: Optional[bool] = Field(
        None,
        description="Current compliance status"
    )
    last_verified: Optional[datetime] = Field(
        None,
        description="Last verification date"
    )


class DerogationRequest(BaseModel):
    """Derogation request per IED Article 15(4)."""

    derogation_id: str = Field(
        default_factory=lambda: f"DEROG-{uuid.uuid4().hex[:6].upper()}",
        description="Derogation identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant subject to derogation"
    )
    bat_ael_upper: float = Field(
        ...,
        description="BAT-AEL upper limit"
    )
    requested_limit: float = Field(
        ...,
        description="Requested derogation limit"
    )
    unit: str = Field(
        default="mg/Nm3",
        description="Unit of measurement"
    )
    justification: str = Field(
        ...,
        description="Justification for derogation"
    )
    geographic_location: str = Field(
        default="",
        description="Geographic location considerations"
    )
    technical_characteristics: str = Field(
        default="",
        description="Technical characteristics justification"
    )
    environmental_impact: str = Field(
        default="",
        description="Environmental impact assessment"
    )
    status: str = Field(
        default="pending",
        description="Derogation status"
    )
    approval_date: Optional[date] = Field(
        None,
        description="Approval date if granted"
    )
    expiry_date: Optional[date] = Field(
        None,
        description="Expiry date"
    )


class ComplianceAssessment(BaseModel):
    """Compliance assessment result."""

    assessment_id: str = Field(
        default_factory=lambda: f"ASSESS-{uuid.uuid4().hex[:8].upper()}",
        description="Assessment identifier"
    )
    installation_id: str = Field(
        ...,
        description="Installation identifier"
    )
    assessment_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Assessment date"
    )
    compliance_status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status"
    )
    pollutants_assessed: int = Field(
        default=0,
        description="Number of pollutants assessed"
    )
    pollutants_compliant: int = Field(
        default=0,
        description="Number compliant"
    )
    pollutants_non_compliant: int = Field(
        default=0,
        description="Number non-compliant"
    )
    elv_compliance: Dict[str, bool] = Field(
        default_factory=dict,
        description="ELV compliance by pollutant"
    )
    bat_ael_compliance: Dict[str, bool] = Field(
        default_factory=dict,
        description="BAT-AEL compliance by pollutant"
    )
    permit_conditions_met: int = Field(
        default=0,
        description="Permit conditions met"
    )
    permit_conditions_total: int = Field(
        default=0,
        description="Total permit conditions"
    )
    derogations_active: int = Field(
        default=0,
        description="Active derogations"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Assessment findings"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    next_assessment_due: Optional[date] = Field(
        None,
        description="Next assessment due date"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }


class AnnualReport(BaseModel):
    """Annual emissions report per IED Article 14."""

    report_id: str = Field(
        default_factory=lambda: f"RPT-{uuid.uuid4().hex[:8].upper()}",
        description="Report identifier"
    )
    installation_id: str = Field(
        ...,
        description="Installation identifier"
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year"
    )
    operator_name: str = Field(
        default="",
        description="Operator name"
    )
    installation_address: str = Field(
        default="",
        description="Installation address"
    )
    activity_category: IEDAnnexIActivity = Field(
        ...,
        description="IED Annex I activity"
    )
    operating_hours: float = Field(
        default=0,
        ge=0,
        description="Annual operating hours"
    )
    production_capacity: float = Field(
        default=0,
        description="Production capacity"
    )
    production_capacity_unit: str = Field(
        default="",
        description="Capacity unit"
    )
    actual_production: float = Field(
        default=0,
        description="Actual production"
    )
    emissions_summary: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Emissions summary by pollutant"
    )
    compliance_summary: Dict[str, bool] = Field(
        default_factory=dict,
        description="Compliance summary"
    )
    incidents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Environmental incidents"
    )
    improvement_actions: List[str] = Field(
        default_factory=list,
        description="Improvement actions taken"
    )
    report_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Report generation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )


# =============================================================================
# IED Compliance Manager
# =============================================================================

class IEDComplianceManager:
    """
    EU Industrial Emissions Directive Compliance Manager.

    Implements comprehensive IED compliance tracking including:
    - BAT conclusions and BAT-AEL monitoring
    - Emission limit value tracking
    - Permit condition monitoring
    - Annual reporting generation
    - BREF document integration

    The manager follows zero-hallucination principles:
    - All assessments are deterministic
    - No LLM involvement in compliance determinations
    - Complete audit trail with provenance hashing

    Attributes:
        installation_id: Installation identifier
        operator: Operator name
        activity: IED Annex I activity category

    Example:
        >>> manager = IEDComplianceManager("INST-001", "Operator Ltd")
        >>> result = manager.assess_compliance(emissions)
    """

    # Default BAT-AELs for common activities (simplified)
    DEFAULT_BAT_AELS: Dict[IEDAnnexIActivity, Dict[str, BATAEL]] = {
        IEDAnnexIActivity.ENERGY_1_1: {
            "NOx": BATAEL(
                pollutant="NOx",
                category=PollutantCategory.AIR,
                lower_limit=50,
                upper_limit=85,
                unit="mg/Nm3",
                reference_conditions="Dry gas, 273K, 101.3kPa, 3% O2",
                bref_reference="LCP BREF 2017"
            ),
            "SO2": BATAEL(
                pollutant="SO2",
                category=PollutantCategory.AIR,
                lower_limit=10,
                upper_limit=35,
                unit="mg/Nm3",
                reference_conditions="Dry gas, 273K, 101.3kPa, 3% O2",
                bref_reference="LCP BREF 2017"
            ),
            "Dust": BATAEL(
                pollutant="Dust",
                category=PollutantCategory.AIR,
                lower_limit=2,
                upper_limit=5,
                unit="mg/Nm3",
                reference_conditions="Dry gas, 273K, 101.3kPa, 3% O2",
                bref_reference="LCP BREF 2017"
            ),
            "CO": BATAEL(
                pollutant="CO",
                category=PollutantCategory.AIR,
                lower_limit=10,
                upper_limit=30,
                unit="mg/Nm3",
                reference_conditions="Dry gas, 273K, 101.3kPa, 3% O2",
                bref_reference="LCP BREF 2017"
            ),
        },
        IEDAnnexIActivity.CHEMICALS_4_1: {
            "VOC": BATAEL(
                pollutant="VOC",
                category=PollutantCategory.AIR,
                lower_limit=5,
                upper_limit=20,
                unit="mg/Nm3",
                bref_reference="LVOC BREF"
            ),
            "NOx": BATAEL(
                pollutant="NOx",
                category=PollutantCategory.AIR,
                lower_limit=50,
                upper_limit=100,
                unit="mg/Nm3",
                bref_reference="LVOC BREF"
            ),
        },
    }

    def __init__(
        self,
        installation_id: str,
        operator: str = "",
        activity: Optional[IEDAnnexIActivity] = None
    ):
        """
        Initialize IED Compliance Manager.

        Args:
            installation_id: Unique installation identifier
            operator: Operator name
            activity: IED Annex I activity category
        """
        self.installation_id = installation_id
        self.operator = operator
        self.activity = activity

        self.bat_conclusions: List[BATConclusion] = []
        self.bat_aels: Dict[str, BATAEL] = {}
        self.elvs: Dict[str, EmissionLimitValue] = {}
        self.permit_conditions: List[PermitCondition] = []
        self.measurements: List[EmissionMeasurement] = []
        self.derogations: List[DerogationRequest] = []
        self.assessments: List[ComplianceAssessment] = []

        # Load default BAT-AELs for activity
        if activity and activity in self.DEFAULT_BAT_AELS:
            self.bat_aels = self.DEFAULT_BAT_AELS[activity].copy()

        logger.info(f"IEDComplianceManager initialized for {installation_id}")

    def set_activity(self, activity: IEDAnnexIActivity) -> None:
        """
        Set or update the IED activity category.

        Args:
            activity: IED Annex I activity
        """
        self.activity = activity
        if activity in self.DEFAULT_BAT_AELS:
            self.bat_aels = self.DEFAULT_BAT_AELS[activity].copy()
        logger.info(f"Activity set to {activity.value}")

    def add_bat_ael(self, bat_ael: BATAEL) -> None:
        """
        Add or update a BAT-AEL.

        Args:
            bat_ael: BATAEL to add
        """
        self.bat_aels[bat_ael.pollutant] = bat_ael
        logger.info(f"Added BAT-AEL for {bat_ael.pollutant}")

    def add_elv(self, elv: EmissionLimitValue) -> None:
        """
        Add or update an ELV.

        Args:
            elv: EmissionLimitValue to add
        """
        self.elvs[elv.pollutant] = elv
        logger.info(f"Added ELV for {elv.pollutant}")

    def add_permit_condition(self, condition: PermitCondition) -> None:
        """
        Add a permit condition.

        Args:
            condition: PermitCondition to add
        """
        self.permit_conditions.append(condition)
        logger.info(f"Added permit condition {condition.condition_id}")

    def record_measurement(self, measurement: EmissionMeasurement) -> None:
        """
        Record an emission measurement.

        Args:
            measurement: EmissionMeasurement to record
        """
        self.measurements.append(measurement)
        logger.info(
            f"Recorded measurement: {measurement.pollutant} = "
            f"{measurement.measured_value} {measurement.unit}"
        )

    def add_derogation(self, derogation: DerogationRequest) -> None:
        """
        Add a derogation request.

        Args:
            derogation: DerogationRequest to add
        """
        self.derogations.append(derogation)
        logger.info(f"Added derogation request {derogation.derogation_id}")

    def assess_compliance(
        self,
        current_emissions: Dict[str, float]
    ) -> ComplianceAssessment:
        """
        Assess compliance against BAT-AELs and ELVs.

        Args:
            current_emissions: Dict of pollutant to emission value

        Returns:
            ComplianceAssessment with results
        """
        logger.info(f"Assessing compliance for {self.installation_id}")

        elv_compliance: Dict[str, bool] = {}
        bat_ael_compliance: Dict[str, bool] = {}
        findings: List[str] = []
        recommendations: List[str] = []

        pollutants_assessed = 0
        pollutants_compliant = 0
        pollutants_non_compliant = 0

        # Check against BAT-AELs
        for pollutant, emission in current_emissions.items():
            pollutants_assessed += 1

            # Check BAT-AEL
            if pollutant in self.bat_aels:
                bat_ael = self.bat_aels[pollutant]
                is_compliant = emission <= bat_ael.upper_limit
                bat_ael_compliance[pollutant] = is_compliant

                if is_compliant:
                    if emission <= bat_ael.lower_limit:
                        findings.append(
                            f"{pollutant}: {emission} {bat_ael.unit} - "
                            f"Below BAT-AEL lower limit ({bat_ael.lower_limit})"
                        )
                    else:
                        findings.append(
                            f"{pollutant}: {emission} {bat_ael.unit} - "
                            f"Within BAT-AEL range ({bat_ael.lower_limit}-{bat_ael.upper_limit})"
                        )
                else:
                    findings.append(
                        f"{pollutant}: {emission} {bat_ael.unit} - "
                        f"EXCEEDS BAT-AEL upper limit ({bat_ael.upper_limit})"
                    )
                    recommendations.append(
                        f"Reduce {pollutant} emissions to below {bat_ael.upper_limit} {bat_ael.unit}"
                    )

            # Check ELV
            if pollutant in self.elvs:
                elv = self.elvs[pollutant]
                is_compliant = emission <= elv.limit_value
                elv_compliance[pollutant] = is_compliant

                if not is_compliant:
                    findings.append(
                        f"{pollutant}: {emission} {elv.unit} - "
                        f"EXCEEDS permit ELV ({elv.limit_value})"
                    )
                    recommendations.append(
                        f"URGENT: Reduce {pollutant} to below permit limit {elv.limit_value} {elv.unit}"
                    )

            # Aggregate compliance
            is_pollutant_compliant = (
                bat_ael_compliance.get(pollutant, True) and
                elv_compliance.get(pollutant, True)
            )
            if is_pollutant_compliant:
                pollutants_compliant += 1
            else:
                pollutants_non_compliant += 1

        # Check permit conditions
        conditions_met = sum(
            1 for c in self.permit_conditions
            if c.is_compliant is True
        )

        # Check derogations
        active_derogations = sum(
            1 for d in self.derogations
            if d.status == "approved"
        )

        # Determine overall status
        if pollutants_non_compliant == 0 and conditions_met == len(self.permit_conditions):
            compliance_status = ComplianceStatus.COMPLIANT
        elif active_derogations > 0 and pollutants_non_compliant <= active_derogations:
            compliance_status = ComplianceStatus.UNDER_DEROGATION
        else:
            compliance_status = ComplianceStatus.NON_COMPLIANT

        assessment = ComplianceAssessment(
            installation_id=self.installation_id,
            compliance_status=compliance_status,
            pollutants_assessed=pollutants_assessed,
            pollutants_compliant=pollutants_compliant,
            pollutants_non_compliant=pollutants_non_compliant,
            elv_compliance=elv_compliance,
            bat_ael_compliance=bat_ael_compliance,
            permit_conditions_met=conditions_met,
            permit_conditions_total=len(self.permit_conditions),
            derogations_active=active_derogations,
            findings=findings,
            recommendations=recommendations,
        )

        assessment.provenance_hash = self._calculate_provenance(assessment)
        self.assessments.append(assessment)

        logger.info(
            f"Compliance assessment complete: {compliance_status.value} "
            f"({pollutants_compliant}/{pollutants_assessed} compliant)"
        )

        return assessment

    def generate_annual_report(
        self,
        reporting_year: int,
        operating_hours: float,
        production_data: Dict[str, float]
    ) -> AnnualReport:
        """
        Generate annual emissions report per IED Article 14.

        Args:
            reporting_year: Year being reported
            operating_hours: Annual operating hours
            production_data: Production data dictionary

        Returns:
            AnnualReport
        """
        logger.info(f"Generating annual report for {reporting_year}")

        # Filter measurements for reporting year
        year_measurements = [
            m for m in self.measurements
            if m.measurement_timestamp.year == reporting_year
        ]

        # Summarize emissions
        emissions_summary: Dict[str, Dict[str, Any]] = {}
        for measurement in year_measurements:
            pollutant = measurement.pollutant
            if pollutant not in emissions_summary:
                emissions_summary[pollutant] = {
                    "measurements_count": 0,
                    "total": 0,
                    "average": 0,
                    "maximum": 0,
                    "minimum": float('inf'),
                    "unit": measurement.unit,
                }

            summary = emissions_summary[pollutant]
            summary["measurements_count"] += 1
            summary["total"] += measurement.measured_value
            summary["maximum"] = max(summary["maximum"], measurement.measured_value)
            summary["minimum"] = min(summary["minimum"], measurement.measured_value)

        # Calculate averages
        for pollutant, summary in emissions_summary.items():
            if summary["measurements_count"] > 0:
                summary["average"] = summary["total"] / summary["measurements_count"]
            if summary["minimum"] == float('inf'):
                summary["minimum"] = 0

        # Compliance summary based on latest assessment
        compliance_summary: Dict[str, bool] = {}
        if self.assessments:
            latest = self.assessments[-1]
            compliance_summary = {
                **latest.elv_compliance,
                **latest.bat_ael_compliance,
            }

        report = AnnualReport(
            installation_id=self.installation_id,
            reporting_year=reporting_year,
            operator_name=self.operator,
            activity_category=self.activity or IEDAnnexIActivity.ENERGY_1_1,
            operating_hours=operating_hours,
            production_capacity=production_data.get("capacity", 0),
            production_capacity_unit=production_data.get("unit", ""),
            actual_production=production_data.get("actual", 0),
            emissions_summary=emissions_summary,
            compliance_summary=compliance_summary,
        )

        report.provenance_hash = self._calculate_report_provenance(report)

        logger.info(f"Annual report generated: {report.report_id}")

        return report

    def get_monitoring_requirements(self) -> List[Dict[str, Any]]:
        """
        Get monitoring requirements per IED Article 14.

        Returns:
            List of monitoring requirements
        """
        requirements = []

        for pollutant, bat_ael in self.bat_aels.items():
            requirements.append({
                "pollutant": pollutant,
                "category": bat_ael.category.value,
                "bat_ael_range": f"{bat_ael.lower_limit}-{bat_ael.upper_limit} {bat_ael.unit}",
                "monitoring_frequency": MonitoringFrequency.CONTINUOUS.value,
                "reference_conditions": bat_ael.reference_conditions,
                "ied_article": "Article 14",
                "requirement": f"Monitor {pollutant} emissions continuously or periodically",
            })

        return requirements

    def check_bat_conclusions_applicability(
        self,
        installation_characteristics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check applicability of BAT conclusions to installation.

        Args:
            installation_characteristics: Installation characteristics

        Returns:
            List of applicable BAT conclusions
        """
        applicable = []

        for bat in self.bat_conclusions:
            # Simplified applicability check
            is_applicable = True  # Would involve detailed checks in practice

            applicable.append({
                "bat_id": bat.bat_id,
                "technique": bat.technique_description,
                "bref": bat.bref_document,
                "is_applicable": is_applicable,
                "is_mandatory": bat.is_mandatory,
                "applicability_notes": bat.applicability,
            })

        return applicable

    def get_bref_references(self) -> List[Dict[str, str]]:
        """
        Get relevant BREF document references.

        Returns:
            List of BREF references
        """
        brefs = {
            IEDAnnexIActivity.ENERGY_1_1: {
                "name": "Large Combustion Plants BREF",
                "code": "LCP BREF",
                "year": "2017",
                "url": "https://eippcb.jrc.ec.europa.eu/reference/large-combustion-plants-0"
            },
            IEDAnnexIActivity.CHEMICALS_4_1: {
                "name": "Large Volume Organic Chemicals BREF",
                "code": "LVOC BREF",
                "year": "2017",
                "url": "https://eippcb.jrc.ec.europa.eu/reference/production-large-volume-organic-chemicals"
            },
            IEDAnnexIActivity.METALS_2_2: {
                "name": "Iron and Steel Production BREF",
                "code": "IS BREF",
                "year": "2012",
                "url": "https://eippcb.jrc.ec.europa.eu/reference/iron-and-steel-production"
            },
        }

        if self.activity and self.activity in brefs:
            return [brefs[self.activity]]

        return list(brefs.values())

    def request_derogation(
        self,
        pollutant: str,
        requested_limit: float,
        justification: str
    ) -> DerogationRequest:
        """
        Create a derogation request per IED Article 15(4).

        Args:
            pollutant: Pollutant for derogation
            requested_limit: Requested limit value
            justification: Justification text

        Returns:
            DerogationRequest
        """
        bat_ael = self.bat_aels.get(pollutant)
        if not bat_ael:
            raise ValueError(f"No BAT-AEL found for {pollutant}")

        if requested_limit <= bat_ael.upper_limit:
            raise ValueError(
                f"Requested limit {requested_limit} is within BAT-AEL range. "
                "Derogation not required."
            )

        derogation = DerogationRequest(
            pollutant=pollutant,
            bat_ael_upper=bat_ael.upper_limit,
            requested_limit=requested_limit,
            unit=bat_ael.unit,
            justification=justification,
        )

        self.derogations.append(derogation)
        logger.info(f"Derogation request created: {derogation.derogation_id}")

        return derogation

    def _calculate_provenance(self, assessment: ComplianceAssessment) -> str:
        """Calculate SHA-256 provenance hash for assessment."""
        provenance_str = (
            f"{assessment.assessment_id}|"
            f"{assessment.installation_id}|"
            f"{assessment.compliance_status.value}|"
            f"{assessment.pollutants_compliant}/{assessment.pollutants_assessed}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_report_provenance(self, report: AnnualReport) -> str:
        """Calculate SHA-256 provenance hash for report."""
        provenance_str = (
            f"{report.report_id}|"
            f"{report.installation_id}|"
            f"{report.reporting_year}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def export_compliance_summary(self) -> Dict[str, Any]:
        """
        Export compliance summary for reporting.

        Returns:
            Compliance summary dictionary
        """
        latest_assessment = self.assessments[-1] if self.assessments else None

        return {
            "installation_id": self.installation_id,
            "operator": self.operator,
            "activity": self.activity.value if self.activity else None,
            "compliance_status": (
                latest_assessment.compliance_status.value
                if latest_assessment else "not_assessed"
            ),
            "bat_aels_tracked": len(self.bat_aels),
            "elvs_tracked": len(self.elvs),
            "permit_conditions": len(self.permit_conditions),
            "measurements_recorded": len(self.measurements),
            "derogations_active": sum(
                1 for d in self.derogations if d.status == "approved"
            ),
            "assessments_performed": len(self.assessments),
            "last_assessment_date": (
                latest_assessment.assessment_date.isoformat()
                if latest_assessment else None
            ),
            "export_timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "IEDComplianceManager",
    "IEDAnnexIActivity",
    "ComplianceStatus",
    "MonitoringFrequency",
    "PollutantCategory",
    "BATConclusion",
    "BATAEL",
    "EmissionLimitValue",
    "EmissionMeasurement",
    "PermitCondition",
    "DerogationRequest",
    "ComplianceAssessment",
    "AnnualReport",
]
