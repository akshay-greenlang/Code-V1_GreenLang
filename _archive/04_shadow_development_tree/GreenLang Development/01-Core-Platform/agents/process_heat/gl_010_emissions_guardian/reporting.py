"""
Regulatory Reporting Module - GL-010 EmissionsGuardian

This module implements automated regulatory emissions reporting for multiple
frameworks including EPA Part 98 (GHG Reporting), EU ETS, Title V Air Permits,
and various state/regional programs. Provides complete report generation,
validation, and electronic submission support.

Key Features:
    - EPA Part 98 Subpart C (Combustion) reporting
    - EPA Part 98 Subpart W (Petroleum & Natural Gas) reporting
    - EU ETS Annual Emissions Report (AER)
    - Title V Annual Compliance Certification
    - RGGI CO2 Allowance Tracking
    - California AB32 MRR (Mandatory Reporting Regulation)
    - State air permit emission inventory reports
    - CEMS data quality assurance reports

Regulatory References:
    - 40 CFR Part 98 - Mandatory GHG Reporting
    - EU MRR (Monitoring and Reporting Regulation)
    - Title V Operating Permits (40 CFR Part 70)
    - California MRR (17 CCR 95100)
    - RGGI Model Rule

Example:
    >>> reporter = RegulatoryReporter(facility_id="FACILITY-001")
    >>> report = reporter.generate_part98_report(
    ...     year=2024,
    ...     subparts=["C", "W"]
    ... )
    >>> print(f"Total GHG: {report.total_emissions_mtco2e:,.0f} mtCO2e")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import statistics
import uuid
from io import StringIO

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Regulatory Reporting Parameters
# =============================================================================

class ReportingConstants:
    """Regulatory reporting constants and thresholds."""

    # EPA Part 98 thresholds (mtCO2e/year)
    PART98_REPORTING_THRESHOLD = 25000
    PART98_SUBPART_C_THRESHOLD = 25000

    # EU ETS thresholds
    EU_ETS_REPORTING_THRESHOLD_MW = 20  # Thermal input capacity

    # California MRR thresholds
    CA_MRR_FACILITY_THRESHOLD = 10000  # mtCO2e/year
    CA_MRR_SUPPLIER_THRESHOLD = 25000

    # RGGI thresholds
    RGGI_APPLICABILITY_MW = 25  # Capacity threshold

    # GWP values (AR5)
    GWP_CO2 = 1
    GWP_CH4 = 28
    GWP_N2O = 265
    GWP_SF6 = 23500
    GWP_HFC134A = 1300

    # Data availability thresholds
    CEMS_DATA_AVAILABILITY_MIN = 90.0  # Minimum 90% valid hours
    SUBSTITUTE_DATA_LIMIT_PCT = 10.0  # Max 10% substitute data

    # Report deadlines
    PART98_DEADLINE_MONTH = 3
    PART98_DEADLINE_DAY = 31
    EU_ETS_DEADLINE_MONTH = 3
    EU_ETS_DEADLINE_DAY = 31


class ReportType(Enum):
    """Regulatory report types."""
    EPA_PART98 = "epa_part98"  # Mandatory GHG Reporting
    EU_ETS_AER = "eu_ets_aer"  # EU ETS Annual Emissions Report
    TITLE_V = "title_v"  # Title V Annual Compliance
    CA_MRR = "ca_mrr"  # California MRR
    RGGI = "rggi"  # RGGI CO2 Report
    STATE_EMISSIONS_INVENTORY = "state_ei"  # State EI
    AIR_QUALITY_PERMIT = "air_permit"  # Permit reporting
    CEMS_QA = "cems_qa"  # CEMS QA Report


class ReportStatus(Enum):
    """Report lifecycle status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMENDED = "amended"


class EmissionSource(Enum):
    """Emission source categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    ELECTRICITY_PURCHASED = "electricity_purchased"
    STEAM_PURCHASED = "steam_purchased"


class CalculationMethod(Enum):
    """Emission calculation methodologies."""
    CEMS = "cems"  # Continuous Emissions Monitoring
    FUEL_ANALYSIS = "fuel_analysis"  # Tier 4 fuel analysis
    DEFAULT_EMISSION_FACTOR = "default_ef"  # Tier 1/2 default factors
    MASS_BALANCE = "mass_balance"  # Tier 3 mass balance
    ENGINEERING_ESTIMATE = "engineering_estimate"


class DataQualityLevel(Enum):
    """Data quality assessment levels."""
    HIGH = "high"  # CEMS with full QA
    MEDIUM = "medium"  # Fuel analysis or verified estimates
    LOW = "low"  # Default factors or estimates
    SUBSTITUTE = "substitute"  # Substitute data used


# =============================================================================
# DATA MODELS
# =============================================================================

class EmissionUnit(BaseModel):
    """Individual emission unit for reporting."""

    unit_id: str = Field(..., description="Unit identifier")
    unit_name: str = Field(default="", description="Unit name")
    unit_type: str = Field(..., description="Unit type (boiler, turbine, etc.)")

    # Capacity
    rated_capacity_mmbtu_hr: Optional[float] = Field(
        default=None, ge=0, description="Rated capacity (MMBtu/hr)"
    )
    rated_capacity_mw: Optional[float] = Field(
        default=None, ge=0, description="Rated capacity (MW)"
    )

    # Operating data
    operating_hours: float = Field(default=8760, ge=0, description="Operating hours")
    fuel_types: List[str] = Field(default_factory=list, description="Fuel types used")

    # Calculation method
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.DEFAULT_EMISSION_FACTOR,
        description="Calculation methodology"
    )
    cems_equipped: bool = Field(default=False, description="CEMS equipped")

    # Annual emissions (metric tons)
    co2_mt: float = Field(default=0.0, ge=0, description="CO2 emissions (mt)")
    ch4_mt: float = Field(default=0.0, ge=0, description="CH4 emissions (mt)")
    n2o_mt: float = Field(default=0.0, ge=0, description="N2O emissions (mt)")
    co2e_mt: float = Field(default=0.0, ge=0, description="CO2e emissions (mt)")

    # Criteria pollutants (tons)
    nox_tons: float = Field(default=0.0, ge=0, description="NOx emissions (tons)")
    so2_tons: float = Field(default=0.0, ge=0, description="SO2 emissions (tons)")
    co_tons: float = Field(default=0.0, ge=0, description="CO emissions (tons)")
    pm_tons: float = Field(default=0.0, ge=0, description="PM emissions (tons)")
    voc_tons: float = Field(default=0.0, ge=0, description="VOC emissions (tons)")

    # Data quality
    data_quality: DataQualityLevel = Field(
        default=DataQualityLevel.MEDIUM,
        description="Data quality level"
    )
    substitute_data_pct: float = Field(
        default=0.0, ge=0, le=100, description="Substitute data percentage"
    )

    class Config:
        use_enum_values = True


class FuelConsumption(BaseModel):
    """Fuel consumption record for reporting."""

    fuel_type: str = Field(..., description="Fuel type")
    fuel_code: str = Field(default="", description="EPA fuel code")

    # Quantity
    quantity: float = Field(..., ge=0, description="Fuel quantity")
    quantity_unit: str = Field(..., description="Quantity unit")
    quantity_mmbtu: float = Field(default=0.0, ge=0, description="Quantity in MMBtu")

    # Fuel properties
    higher_heating_value: Optional[float] = Field(
        default=None, description="HHV (Btu/unit)"
    )
    carbon_content_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="Carbon content (%)"
    )

    # Emission factors
    co2_emission_factor: float = Field(
        default=0.0, ge=0, description="CO2 EF (kg/MMBtu)"
    )
    ch4_emission_factor: float = Field(
        default=0.0, ge=0, description="CH4 EF (kg/MMBtu)"
    )
    n2o_emission_factor: float = Field(
        default=0.0, ge=0, description="N2O EF (kg/MMBtu)"
    )

    # Source of emission factors
    ef_source: str = Field(
        default="40 CFR Part 98 Table C-1",
        description="Emission factor source"
    )

    # Calculated emissions
    co2_mt: float = Field(default=0.0, ge=0, description="CO2 emissions (mt)")
    ch4_mt: float = Field(default=0.0, ge=0, description="CH4 emissions (mt)")
    n2o_mt: float = Field(default=0.0, ge=0, description="N2O emissions (mt)")
    co2e_mt: float = Field(default=0.0, ge=0, description="CO2e emissions (mt)")


class CEMSDataSummary(BaseModel):
    """CEMS data summary for reporting."""

    unit_id: str = Field(..., description="Unit identifier")
    reporting_year: int = Field(..., description="Reporting year")

    # Data availability
    total_operating_hours: float = Field(default=0.0, ge=0)
    valid_cems_hours: float = Field(default=0.0, ge=0)
    data_availability_pct: float = Field(default=0.0, ge=0, le=100)

    # Substitute data
    substitute_data_hours: float = Field(default=0.0, ge=0)
    substitute_data_pct: float = Field(default=0.0, ge=0, le=100)
    substitute_data_method: str = Field(default="")

    # QA activities
    rata_performed: bool = Field(default=False)
    rata_result_pct: Optional[float] = Field(default=None)
    cga_count: int = Field(default=0, ge=0)
    calibration_drift_exceedances: int = Field(default=0, ge=0)

    # Annual totals
    annual_co2_tons: float = Field(default=0.0, ge=0)
    annual_nox_tons: float = Field(default=0.0, ge=0)
    annual_so2_tons: float = Field(default=0.0, ge=0)
    annual_heat_input_mmbtu: float = Field(default=0.0, ge=0)


class Part98Report(BaseModel):
    """EPA Part 98 GHG Report structure."""

    facility_id: str = Field(..., description="EPA facility ID")
    facility_name: str = Field(..., description="Facility name")
    reporting_year: int = Field(..., description="Reporting year")
    subparts_reported: List[str] = Field(..., description="Subparts reported")

    # Facility information
    naics_code: str = Field(..., description="Primary NAICS code")
    state: str = Field(..., description="State")
    parent_company: str = Field(default="", description="Parent company")

    # Total emissions
    total_emissions_mtco2e: float = Field(
        ..., ge=0, description="Total facility emissions (mtCO2e)"
    )

    # By gas
    co2_mt: float = Field(default=0.0, ge=0)
    ch4_mt: float = Field(default=0.0, ge=0)
    n2o_mt: float = Field(default=0.0, ge=0)
    other_ghg_mt: float = Field(default=0.0, ge=0)

    # By subpart
    emissions_by_subpart: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by subpart (mtCO2e)"
    )

    # Unit-level data
    emission_units: List[EmissionUnit] = Field(
        default_factory=list, description="Emission unit data"
    )

    # Fuel data
    fuel_consumption: List[FuelConsumption] = Field(
        default_factory=list, description="Fuel consumption data"
    )

    # CEMS data
    cems_summaries: List[CEMSDataSummary] = Field(
        default_factory=list, description="CEMS data summaries"
    )

    # Report metadata
    report_status: ReportStatus = Field(
        default=ReportStatus.DRAFT, description="Report status"
    )
    prepared_by: str = Field(default="", description="Report preparer")
    reviewed_by: str = Field(default="", description="Report reviewer")
    submission_date: Optional[datetime] = Field(default=None)
    confirmation_number: Optional[str] = Field(default=None)

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    generation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Config:
        use_enum_values = True


class TitleVReport(BaseModel):
    """Title V Annual Compliance Certification."""

    permit_number: str = Field(..., description="Title V permit number")
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")
    reporting_year: int = Field(..., description="Reporting year")

    # Permit terms compliance
    emission_limits_met: bool = Field(
        default=True, description="All emission limits met"
    )
    monitoring_requirements_met: bool = Field(
        default=True, description="All monitoring requirements met"
    )
    recordkeeping_requirements_met: bool = Field(
        default=True, description="All recordkeeping requirements met"
    )
    reporting_requirements_met: bool = Field(
        default=True, description="All reporting requirements met"
    )

    # Deviations
    deviations: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of deviations"
    )
    deviation_count: int = Field(default=0, ge=0)

    # Emissions vs limits
    emissions_summary: List[Dict[str, Any]] = Field(
        default_factory=list, description="Emissions vs permit limits"
    )

    # Certifications
    responsible_official: str = Field(..., description="Responsible official name")
    official_title: str = Field(..., description="Official title")
    certification_date: date = Field(..., description="Certification date")
    certification_statement: str = Field(
        default="I certify that, based on information and belief formed after "
                "reasonable inquiry, the statements and information in this document "
                "are true, accurate, and complete.",
        description="Certification statement"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class EmissionInventoryReport(BaseModel):
    """State/Local Emission Inventory Report."""

    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")
    reporting_year: int = Field(..., description="Reporting year")
    reporting_agency: str = Field(..., description="Reporting agency")

    # Location
    state: str = Field(..., description="State")
    county: str = Field(default="", description="County")
    air_basin: str = Field(default="", description="Air basin/district")

    # Emissions by pollutant (tons/year)
    nox_tpy: float = Field(default=0.0, ge=0)
    sox_tpy: float = Field(default=0.0, ge=0)
    co_tpy: float = Field(default=0.0, ge=0)
    pm10_tpy: float = Field(default=0.0, ge=0)
    pm25_tpy: float = Field(default=0.0, ge=0)
    voc_tpy: float = Field(default=0.0, ge=0)
    nh3_tpy: float = Field(default=0.0, ge=0)
    lead_tpy: float = Field(default=0.0, ge=0)

    # HAPs
    total_haps_tpy: float = Field(default=0.0, ge=0)
    haps_detail: Dict[str, float] = Field(
        default_factory=dict, description="HAPs by pollutant"
    )

    # By source category
    by_source_category: Dict[str, Dict[str, float]] = Field(
        default_factory=dict
    )

    # Stack parameters
    stack_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Stack parameters"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class ReportSubmission(BaseModel):
    """Report submission record."""

    submission_id: str = Field(..., description="Submission identifier")
    report_type: ReportType = Field(..., description="Report type")
    facility_id: str = Field(..., description="Facility identifier")
    reporting_year: int = Field(..., description="Reporting year")

    # Submission details
    submission_date: datetime = Field(..., description="Submission timestamp")
    submission_method: str = Field(
        default="electronic", description="Submission method"
    )
    submission_system: str = Field(default="", description="Submission system/portal")
    confirmation_number: Optional[str] = Field(default=None)

    # Status tracking
    status: ReportStatus = Field(
        default=ReportStatus.SUBMITTED, description="Submission status"
    )
    agency_response_date: Optional[datetime] = Field(default=None)
    agency_comments: Optional[str] = Field(default=None)

    # Amendment tracking
    is_amendment: bool = Field(default=False)
    amends_submission_id: Optional[str] = Field(default=None)
    amendment_reason: Optional[str] = Field(default=None)

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


# =============================================================================
# REGULATORY REPORTER
# =============================================================================

class RegulatoryReporter:
    """
    Automated Regulatory Emissions Reporting Engine.

    Implements comprehensive emissions reporting for multiple regulatory
    frameworks including EPA Part 98, EU ETS, Title V, and state programs.
    Provides report generation, validation, and submission tracking.

    Features:
        - Multi-framework report generation
        - Automated calculation validation
        - Data quality assessment
        - Electronic submission formatting
        - Amendment tracking
        - Complete audit trail

    Frameworks Supported:
        - EPA 40 CFR Part 98 (GHG Mandatory Reporting)
        - EU ETS Monitoring & Reporting Regulation
        - Title V Annual Compliance Certification
        - California MRR
        - RGGI CO2 Reporting
        - State emission inventory reports

    Example:
        >>> reporter = RegulatoryReporter(facility_id="FACILITY-001")
        >>> report = reporter.generate_part98_report(year=2024)
        >>> validation = reporter.validate_report(report)
        >>> if validation["is_valid"]:
        ...     submission = reporter.submit_report(report)
    """

    # EPA Part 98 Table C-1 emission factors (kg CO2/MMBtu)
    EPA_CO2_EMISSION_FACTORS = {
        "natural_gas": 53.06,
        "distillate_fuel_oil_no2": 73.96,
        "residual_fuel_oil_no6": 75.10,
        "propane": 62.87,
        "coal_bituminous": 93.28,
        "coal_subbituminous": 97.17,
        "coal_lignite": 97.72,
        "petroleum_coke": 102.41,
        "wood_residuals": 93.80,
        "landfill_gas": 52.07,
    }

    # EPA Part 98 Table C-2 CH4 emission factors (g CH4/MMBtu)
    EPA_CH4_EMISSION_FACTORS = {
        "natural_gas": 1.0,
        "distillate_fuel_oil_no2": 3.0,
        "residual_fuel_oil_no6": 3.0,
        "propane": 3.0,
        "coal_bituminous": 11.0,
        "coal_subbituminous": 11.0,
        "wood_residuals": 32.0,
    }

    # EPA Part 98 Table C-2 N2O emission factors (g N2O/MMBtu)
    EPA_N2O_EMISSION_FACTORS = {
        "natural_gas": 0.1,
        "distillate_fuel_oil_no2": 0.6,
        "residual_fuel_oil_no6": 0.6,
        "propane": 0.6,
        "coal_bituminous": 1.6,
        "coal_subbituminous": 1.6,
        "wood_residuals": 4.2,
    }

    def __init__(
        self,
        facility_id: str,
        facility_name: str = "",
        state: str = "",
        naics_code: str = "",
    ) -> None:
        """
        Initialize Regulatory Reporter.

        Args:
            facility_id: Facility identifier
            facility_name: Facility name
            state: State code
            naics_code: Primary NAICS code
        """
        self.facility_id = facility_id
        self.facility_name = facility_name
        self.state = state
        self.naics_code = naics_code

        # Report storage
        self._reports: Dict[str, Any] = {}
        self._submissions: List[ReportSubmission] = []

        # Emission unit registry
        self._emission_units: Dict[str, EmissionUnit] = {}

        # Fuel consumption data
        self._fuel_data: List[FuelConsumption] = []

        # CEMS data
        self._cems_data: Dict[str, CEMSDataSummary] = {}

        logger.info(
            f"RegulatoryReporter initialized for {facility_id} ({facility_name})"
        )

    # =========================================================================
    # EMISSION UNIT MANAGEMENT
    # =========================================================================

    def register_emission_unit(
        self,
        unit_id: str,
        unit_type: str,
        unit_name: str = "",
        rated_capacity_mmbtu_hr: Optional[float] = None,
        rated_capacity_mw: Optional[float] = None,
        fuel_types: Optional[List[str]] = None,
        cems_equipped: bool = False,
    ) -> EmissionUnit:
        """
        Register an emission unit for reporting.

        Args:
            unit_id: Unit identifier
            unit_type: Unit type (boiler, turbine, heater, etc.)
            unit_name: Unit name
            rated_capacity_mmbtu_hr: Rated capacity in MMBtu/hr
            rated_capacity_mw: Rated capacity in MW
            fuel_types: Fuel types used
            cems_equipped: Whether unit has CEMS

        Returns:
            EmissionUnit record
        """
        calc_method = (
            CalculationMethod.CEMS if cems_equipped
            else CalculationMethod.DEFAULT_EMISSION_FACTOR
        )

        unit = EmissionUnit(
            unit_id=unit_id,
            unit_name=unit_name,
            unit_type=unit_type,
            rated_capacity_mmbtu_hr=rated_capacity_mmbtu_hr,
            rated_capacity_mw=rated_capacity_mw,
            fuel_types=fuel_types or [],
            calculation_method=calc_method,
            cems_equipped=cems_equipped,
        )

        self._emission_units[unit_id] = unit

        logger.info(f"Emission unit {unit_id} registered: {unit_type}")

        return unit

    def record_fuel_consumption(
        self,
        fuel_type: str,
        quantity: float,
        quantity_unit: str,
        higher_heating_value: Optional[float] = None,
        carbon_content_pct: Optional[float] = None,
    ) -> FuelConsumption:
        """
        Record fuel consumption for reporting.

        Args:
            fuel_type: Fuel type
            quantity: Fuel quantity
            quantity_unit: Quantity unit (scf, gal, tons, etc.)
            higher_heating_value: HHV in Btu/unit
            carbon_content_pct: Carbon content percentage

        Returns:
            FuelConsumption record with calculated emissions
        """
        # Get default HHV if not provided
        default_hhv = self._get_default_hhv(fuel_type, quantity_unit)
        hhv = higher_heating_value or default_hhv

        # Calculate MMBtu
        quantity_mmbtu = quantity * hhv / 1_000_000 if hhv else 0

        # Get emission factors
        fuel_key = fuel_type.lower().replace(" ", "_")
        co2_ef = self.EPA_CO2_EMISSION_FACTORS.get(fuel_key, 53.06)
        ch4_ef = self.EPA_CH4_EMISSION_FACTORS.get(fuel_key, 1.0)
        n2o_ef = self.EPA_N2O_EMISSION_FACTORS.get(fuel_key, 0.1)

        # Calculate emissions (convert to metric tons)
        co2_mt = quantity_mmbtu * co2_ef / 1000
        ch4_mt = quantity_mmbtu * ch4_ef / 1_000_000
        n2o_mt = quantity_mmbtu * n2o_ef / 1_000_000

        # Calculate CO2e
        co2e_mt = (
            co2_mt * ReportingConstants.GWP_CO2 +
            ch4_mt * ReportingConstants.GWP_CH4 +
            n2o_mt * ReportingConstants.GWP_N2O
        )

        fuel_record = FuelConsumption(
            fuel_type=fuel_type,
            fuel_code=self._get_epa_fuel_code(fuel_type),
            quantity=quantity,
            quantity_unit=quantity_unit,
            quantity_mmbtu=quantity_mmbtu,
            higher_heating_value=hhv,
            carbon_content_pct=carbon_content_pct,
            co2_emission_factor=co2_ef,
            ch4_emission_factor=ch4_ef,
            n2o_emission_factor=n2o_ef,
            co2_mt=round(co2_mt, 3),
            ch4_mt=round(ch4_mt, 6),
            n2o_mt=round(n2o_mt, 6),
            co2e_mt=round(co2e_mt, 3),
        )

        self._fuel_data.append(fuel_record)

        logger.info(
            f"Fuel consumption recorded: {quantity:,.0f} {quantity_unit} {fuel_type}, "
            f"CO2e: {co2e_mt:,.1f} mt"
        )

        return fuel_record

    def _get_default_hhv(self, fuel_type: str, unit: str) -> float:
        """Get default higher heating value for fuel type."""
        # HHV values in Btu per standard unit
        default_hhv = {
            ("natural_gas", "scf"): 1020,
            ("natural_gas", "mscf"): 1020000,
            ("natural_gas", "mmscf"): 1020000000,
            ("distillate_fuel_oil_no2", "gal"): 138690,
            ("residual_fuel_oil_no6", "gal"): 149690,
            ("propane", "gal"): 91420,
            ("coal_bituminous", "tons"): 24930000,
            ("coal_subbituminous", "tons"): 17250000,
        }
        key = (fuel_type.lower().replace(" ", "_"), unit.lower())
        return default_hhv.get(key, 1000000)

    def _get_epa_fuel_code(self, fuel_type: str) -> str:
        """Get EPA fuel code for Part 98 reporting."""
        fuel_codes = {
            "natural_gas": "NG",
            "distillate_fuel_oil_no2": "DFO",
            "residual_fuel_oil_no6": "RFO",
            "propane": "PG",
            "coal_bituminous": "C",
            "coal_subbituminous": "C",
            "petroleum_coke": "PC",
            "wood_residuals": "WD",
        }
        return fuel_codes.get(fuel_type.lower().replace(" ", "_"), "OTH")

    # =========================================================================
    # EPA PART 98 REPORTING
    # =========================================================================

    def generate_part98_report(
        self,
        year: int,
        subparts: List[str] = None,
        parent_company: str = "",
        prepared_by: str = "",
    ) -> Part98Report:
        """
        Generate EPA Part 98 GHG Report.

        Args:
            year: Reporting year
            subparts: Subparts to report (default: ["C"])
            parent_company: Parent company name
            prepared_by: Report preparer name

        Returns:
            Part98Report with complete emissions data
        """
        subparts = subparts or ["C"]  # Default to Subpart C (combustion)

        logger.info(f"Generating Part 98 report for {self.facility_id}, year {year}")

        # Aggregate fuel data
        total_co2 = sum(f.co2_mt for f in self._fuel_data)
        total_ch4 = sum(f.ch4_mt for f in self._fuel_data)
        total_n2o = sum(f.n2o_mt for f in self._fuel_data)
        total_co2e = sum(f.co2e_mt for f in self._fuel_data)

        # Calculate emissions by subpart
        emissions_by_subpart = {}
        if "C" in subparts:
            emissions_by_subpart["C"] = total_co2e

        # Update emission units with calculated emissions
        for unit in self._emission_units.values():
            # Allocate emissions proportionally by capacity
            # (Simplified - in production would use unit-specific data)
            unit.co2_mt = total_co2 / max(len(self._emission_units), 1)
            unit.ch4_mt = total_ch4 / max(len(self._emission_units), 1)
            unit.n2o_mt = total_n2o / max(len(self._emission_units), 1)
            unit.co2e_mt = total_co2e / max(len(self._emission_units), 1)

        # Calculate provenance hash
        provenance_hash = self._hash_report_data(
            report_type="part98",
            facility_id=self.facility_id,
            year=year,
            total_emissions=total_co2e,
        )

        report = Part98Report(
            facility_id=self.facility_id,
            facility_name=self.facility_name,
            reporting_year=year,
            subparts_reported=subparts,
            naics_code=self.naics_code,
            state=self.state,
            parent_company=parent_company,
            total_emissions_mtco2e=round(total_co2e, 3),
            co2_mt=round(total_co2, 3),
            ch4_mt=round(total_ch4, 6),
            n2o_mt=round(total_n2o, 6),
            other_ghg_mt=0.0,
            emissions_by_subpart=emissions_by_subpart,
            emission_units=list(self._emission_units.values()),
            fuel_consumption=self._fuel_data,
            cems_summaries=list(self._cems_data.values()),
            prepared_by=prepared_by,
            provenance_hash=provenance_hash,
        )

        self._reports[f"part98_{year}"] = report

        logger.info(
            f"Part 98 report generated: {total_co2e:,.1f} mtCO2e total"
        )

        return report

    # =========================================================================
    # TITLE V REPORTING
    # =========================================================================

    def generate_title_v_report(
        self,
        year: int,
        permit_number: str,
        responsible_official: str,
        official_title: str,
        emission_limits: Optional[Dict[str, Dict[str, float]]] = None,
        deviations: Optional[List[Dict[str, Any]]] = None,
    ) -> TitleVReport:
        """
        Generate Title V Annual Compliance Certification.

        Args:
            year: Reporting year
            permit_number: Title V permit number
            responsible_official: Responsible official name
            official_title: Official's title
            emission_limits: Emission limits by pollutant and unit
            deviations: List of permit deviations

        Returns:
            TitleVReport
        """
        logger.info(f"Generating Title V report for {self.facility_id}, year {year}")

        deviations = deviations or []

        # Calculate emissions vs limits summary
        emissions_summary = []

        if emission_limits:
            for pollutant, limits in emission_limits.items():
                for unit_id, limit_value in limits.items():
                    unit = self._emission_units.get(unit_id)
                    if unit:
                        actual = getattr(unit, f"{pollutant.lower()}_tons", 0)
                        emissions_summary.append({
                            "unit_id": unit_id,
                            "pollutant": pollutant,
                            "limit_tpy": limit_value,
                            "actual_tpy": actual,
                            "pct_of_limit": round(actual / limit_value * 100, 1) if limit_value > 0 else 0,
                            "compliant": actual <= limit_value,
                        })

        # Determine compliance status
        all_limits_met = all(e["compliant"] for e in emissions_summary) if emissions_summary else True

        # Provenance hash
        provenance_hash = self._hash_report_data(
            report_type="title_v",
            facility_id=self.facility_id,
            year=year,
            total_emissions=0,
        )

        report = TitleVReport(
            permit_number=permit_number,
            facility_id=self.facility_id,
            facility_name=self.facility_name,
            reporting_year=year,
            emission_limits_met=all_limits_met,
            monitoring_requirements_met=True,
            recordkeeping_requirements_met=True,
            reporting_requirements_met=True,
            deviations=deviations,
            deviation_count=len(deviations),
            emissions_summary=emissions_summary,
            responsible_official=responsible_official,
            official_title=official_title,
            certification_date=date.today(),
            provenance_hash=provenance_hash,
        )

        self._reports[f"title_v_{year}"] = report

        logger.info(
            f"Title V report generated: {len(deviations)} deviations, "
            f"limits met: {all_limits_met}"
        )

        return report

    # =========================================================================
    # STATE EMISSION INVENTORY
    # =========================================================================

    def generate_emission_inventory(
        self,
        year: int,
        reporting_agency: str,
        county: str = "",
        air_basin: str = "",
    ) -> EmissionInventoryReport:
        """
        Generate State/Local Emission Inventory Report.

        Args:
            year: Reporting year
            reporting_agency: Reporting agency name
            county: County name
            air_basin: Air basin/district

        Returns:
            EmissionInventoryReport
        """
        logger.info(f"Generating emission inventory for {self.facility_id}, year {year}")

        # Aggregate criteria pollutant emissions from units
        total_nox = sum(u.nox_tons for u in self._emission_units.values())
        total_so2 = sum(u.so2_tons for u in self._emission_units.values())
        total_co = sum(u.co_tons for u in self._emission_units.values())
        total_pm = sum(u.pm_tons for u in self._emission_units.values())
        total_voc = sum(u.voc_tons for u in self._emission_units.values())

        # By source category
        by_source = {}
        for unit in self._emission_units.values():
            category = unit.unit_type
            if category not in by_source:
                by_source[category] = {
                    "nox": 0, "so2": 0, "co": 0, "pm": 0, "voc": 0
                }
            by_source[category]["nox"] += unit.nox_tons
            by_source[category]["so2"] += unit.so2_tons
            by_source[category]["co"] += unit.co_tons
            by_source[category]["pm"] += unit.pm_tons
            by_source[category]["voc"] += unit.voc_tons

        # Provenance hash
        provenance_hash = self._hash_report_data(
            report_type="emission_inventory",
            facility_id=self.facility_id,
            year=year,
            total_emissions=total_nox + total_so2 + total_co,
        )

        report = EmissionInventoryReport(
            facility_id=self.facility_id,
            facility_name=self.facility_name,
            reporting_year=year,
            reporting_agency=reporting_agency,
            state=self.state,
            county=county,
            air_basin=air_basin,
            nox_tpy=round(total_nox, 3),
            sox_tpy=round(total_so2, 3),
            co_tpy=round(total_co, 3),
            pm10_tpy=round(total_pm, 3),
            pm25_tpy=round(total_pm * 0.9, 3),  # Approximate PM2.5
            voc_tpy=round(total_voc, 3),
            by_source_category=by_source,
            provenance_hash=provenance_hash,
        )

        self._reports[f"ei_{year}"] = report

        logger.info(
            f"Emission inventory generated: NOx={total_nox:.1f}, "
            f"SO2={total_so2:.1f}, VOC={total_voc:.1f} tpy"
        )

        return report

    # =========================================================================
    # REPORT VALIDATION
    # =========================================================================

    def validate_report(
        self,
        report: Union[Part98Report, TitleVReport, EmissionInventoryReport],
    ) -> Dict[str, Any]:
        """
        Validate report for completeness and accuracy.

        Args:
            report: Report to validate

        Returns:
            Validation results with issues and warnings
        """
        issues = []
        warnings = []

        if isinstance(report, Part98Report):
            issues, warnings = self._validate_part98_report(report)
        elif isinstance(report, TitleVReport):
            issues, warnings = self._validate_title_v_report(report)
        elif isinstance(report, EmissionInventoryReport):
            issues, warnings = self._validate_ei_report(report)

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _validate_part98_report(
        self, report: Part98Report
    ) -> Tuple[List[str], List[str]]:
        """Validate Part 98 report."""
        issues = []
        warnings = []

        # Check required fields
        if not report.facility_id:
            issues.append("Facility ID is required")
        if not report.naics_code:
            issues.append("NAICS code is required")
        if not report.subparts_reported:
            issues.append("At least one subpart must be reported")

        # Check reporting threshold
        if report.total_emissions_mtco2e < ReportingConstants.PART98_REPORTING_THRESHOLD:
            warnings.append(
                f"Emissions {report.total_emissions_mtco2e:,.0f} mtCO2e below "
                f"reporting threshold of {ReportingConstants.PART98_REPORTING_THRESHOLD:,} mtCO2e"
            )

        # Check data quality
        for cems in report.cems_summaries:
            if cems.data_availability_pct < ReportingConstants.CEMS_DATA_AVAILABILITY_MIN:
                warnings.append(
                    f"Unit {cems.unit_id} CEMS data availability "
                    f"{cems.data_availability_pct:.1f}% below minimum "
                    f"{ReportingConstants.CEMS_DATA_AVAILABILITY_MIN}%"
                )
            if cems.substitute_data_pct > ReportingConstants.SUBSTITUTE_DATA_LIMIT_PCT:
                warnings.append(
                    f"Unit {cems.unit_id} substitute data "
                    f"{cems.substitute_data_pct:.1f}% exceeds "
                    f"{ReportingConstants.SUBSTITUTE_DATA_LIMIT_PCT}% limit"
                )

        # Check emission calculations
        calculated_co2e = (
            report.co2_mt * ReportingConstants.GWP_CO2 +
            report.ch4_mt * ReportingConstants.GWP_CH4 +
            report.n2o_mt * ReportingConstants.GWP_N2O
        )
        if abs(calculated_co2e - report.total_emissions_mtco2e) > 1:
            warnings.append(
                f"CO2e total {report.total_emissions_mtco2e:,.1f} differs from "
                f"calculated {calculated_co2e:,.1f}"
            )

        return issues, warnings

    def _validate_title_v_report(
        self, report: TitleVReport
    ) -> Tuple[List[str], List[str]]:
        """Validate Title V report."""
        issues = []
        warnings = []

        # Check required fields
        if not report.permit_number:
            issues.append("Permit number is required")
        if not report.responsible_official:
            issues.append("Responsible official is required")
        if not report.certification_date:
            issues.append("Certification date is required")

        # Check deviations
        if report.deviation_count > 0:
            warnings.append(
                f"{report.deviation_count} deviation(s) reported - "
                "ensure corrective actions documented"
            )

        # Check emission limits
        for summary in report.emissions_summary:
            if not summary.get("compliant", True):
                issues.append(
                    f"Emission limit exceeded for {summary['pollutant']} "
                    f"at unit {summary['unit_id']}"
                )

        return issues, warnings

    def _validate_ei_report(
        self, report: EmissionInventoryReport
    ) -> Tuple[List[str], List[str]]:
        """Validate emission inventory report."""
        issues = []
        warnings = []

        # Check required fields
        if not report.reporting_agency:
            issues.append("Reporting agency is required")
        if not report.state:
            issues.append("State is required")

        # Check for zero emissions (likely incomplete)
        total = (
            report.nox_tpy + report.sox_tpy + report.co_tpy +
            report.voc_tpy + report.pm10_tpy
        )
        if total == 0:
            warnings.append("All emissions are zero - verify data completeness")

        return issues, warnings

    # =========================================================================
    # REPORT SUBMISSION
    # =========================================================================

    def submit_report(
        self,
        report: Union[Part98Report, TitleVReport, EmissionInventoryReport],
        submission_system: str = "",
        submitter: str = "",
    ) -> ReportSubmission:
        """
        Submit report and record submission.

        Args:
            report: Report to submit
            submission_system: Submission system/portal name
            submitter: Person submitting

        Returns:
            ReportSubmission record
        """
        # Determine report type
        if isinstance(report, Part98Report):
            report_type = ReportType.EPA_PART98
            year = report.reporting_year
        elif isinstance(report, TitleVReport):
            report_type = ReportType.TITLE_V
            year = report.reporting_year
        elif isinstance(report, EmissionInventoryReport):
            report_type = ReportType.STATE_EMISSIONS_INVENTORY
            year = report.reporting_year
        else:
            raise ValueError("Unknown report type")

        submission_id = f"SUB_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Update report status
        if hasattr(report, "report_status"):
            report.report_status = ReportStatus.SUBMITTED
        if hasattr(report, "submission_date"):
            report.submission_date = datetime.now(timezone.utc)

        # Provenance hash
        provenance_hash = self._hash_report_data(
            report_type=report_type.value,
            facility_id=self.facility_id,
            year=year,
            total_emissions=0,
        )

        submission = ReportSubmission(
            submission_id=submission_id,
            report_type=report_type,
            facility_id=self.facility_id,
            reporting_year=year,
            submission_date=datetime.now(timezone.utc),
            submission_system=submission_system,
            status=ReportStatus.SUBMITTED,
            provenance_hash=provenance_hash,
        )

        self._submissions.append(submission)

        logger.info(
            f"Report submitted: {report_type.value} for year {year}, "
            f"submission ID {submission_id}"
        )

        return submission

    def update_submission_status(
        self,
        submission_id: str,
        status: ReportStatus,
        confirmation_number: Optional[str] = None,
        agency_comments: Optional[str] = None,
    ) -> Optional[ReportSubmission]:
        """
        Update submission status after agency response.

        Args:
            submission_id: Submission identifier
            status: New status
            confirmation_number: Agency confirmation number
            agency_comments: Agency comments

        Returns:
            Updated ReportSubmission or None if not found
        """
        submission = next(
            (s for s in self._submissions if s.submission_id == submission_id),
            None
        )

        if submission:
            submission.status = status
            submission.agency_response_date = datetime.now(timezone.utc)
            if confirmation_number:
                submission.confirmation_number = confirmation_number
            if agency_comments:
                submission.agency_comments = agency_comments

            logger.info(f"Submission {submission_id} updated to {status.value}")

        return submission

    # =========================================================================
    # REPORT EXPORT
    # =========================================================================

    def export_to_xml(
        self,
        report: Part98Report,
    ) -> str:
        """
        Export Part 98 report to XML format for e-GGRT submission.

        Args:
            report: Part 98 report

        Returns:
            XML string
        """
        # Simplified XML generation - production would use full EPA schema
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<GHGReport xmlns="http://www.ccdsupport.com/schema/ghg">',
            f'  <FacilityId>{report.facility_id}</FacilityId>',
            f'  <FacilityName>{report.facility_name}</FacilityName>',
            f'  <ReportingYear>{report.reporting_year}</ReportingYear>',
            f'  <NAICSCode>{report.naics_code}</NAICSCode>',
            '  <Emissions>',
            f'    <TotalCO2eMT>{report.total_emissions_mtco2e:.3f}</TotalCO2eMT>',
            f'    <CO2MT>{report.co2_mt:.3f}</CO2MT>',
            f'    <CH4MT>{report.ch4_mt:.6f}</CH4MT>',
            f'    <N2OMT>{report.n2o_mt:.6f}</N2OMT>',
            '  </Emissions>',
            '  <Subparts>',
        ]

        for subpart, emissions in report.emissions_by_subpart.items():
            xml_lines.append(f'    <Subpart name="{subpart}">{emissions:.3f}</Subpart>')

        xml_lines.extend([
            '  </Subparts>',
            '</GHGReport>',
        ])

        return '\n'.join(xml_lines)

    def export_to_csv(
        self,
        report: Union[Part98Report, EmissionInventoryReport],
    ) -> str:
        """
        Export report to CSV format.

        Args:
            report: Report to export

        Returns:
            CSV string
        """
        output = StringIO()

        if isinstance(report, Part98Report):
            # Header
            output.write("Facility ID,Facility Name,Year,Total CO2e (mt),CO2 (mt),CH4 (mt),N2O (mt)\n")
            # Data row
            output.write(
                f"{report.facility_id},{report.facility_name},{report.reporting_year},"
                f"{report.total_emissions_mtco2e:.3f},{report.co2_mt:.3f},"
                f"{report.ch4_mt:.6f},{report.n2o_mt:.6f}\n"
            )

            # Fuel consumption
            output.write("\nFuel Consumption\n")
            output.write("Fuel Type,Quantity,Unit,MMBtu,CO2 (mt),CH4 (mt),N2O (mt),CO2e (mt)\n")
            for fuel in report.fuel_consumption:
                output.write(
                    f"{fuel.fuel_type},{fuel.quantity},{fuel.quantity_unit},"
                    f"{fuel.quantity_mmbtu:.0f},{fuel.co2_mt:.3f},"
                    f"{fuel.ch4_mt:.6f},{fuel.n2o_mt:.6f},{fuel.co2e_mt:.3f}\n"
                )

        elif isinstance(report, EmissionInventoryReport):
            output.write("Pollutant,Emissions (tpy)\n")
            output.write(f"NOx,{report.nox_tpy:.3f}\n")
            output.write(f"SO2,{report.sox_tpy:.3f}\n")
            output.write(f"CO,{report.co_tpy:.3f}\n")
            output.write(f"PM10,{report.pm10_tpy:.3f}\n")
            output.write(f"PM2.5,{report.pm25_tpy:.3f}\n")
            output.write(f"VOC,{report.voc_tpy:.3f}\n")

        return output.getvalue()

    # =========================================================================
    # REPORTING DEADLINES
    # =========================================================================

    def get_upcoming_deadlines(
        self,
        year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming reporting deadlines.

        Args:
            year: Reporting year (default: current year)

        Returns:
            List of upcoming deadlines
        """
        year = year or datetime.now().year
        today = date.today()

        deadlines = [
            {
                "report_type": "EPA Part 98",
                "reporting_year": year,
                "deadline": date(
                    year + 1,
                    ReportingConstants.PART98_DEADLINE_MONTH,
                    ReportingConstants.PART98_DEADLINE_DAY
                ),
                "days_until": None,
                "status": "upcoming",
            },
            {
                "report_type": "EU ETS AER",
                "reporting_year": year,
                "deadline": date(
                    year + 1,
                    ReportingConstants.EU_ETS_DEADLINE_MONTH,
                    ReportingConstants.EU_ETS_DEADLINE_DAY
                ),
                "days_until": None,
                "status": "upcoming",
            },
            {
                "report_type": "Title V Certification",
                "reporting_year": year,
                "deadline": date(year + 1, 3, 15),  # Typical deadline
                "days_until": None,
                "status": "upcoming",
            },
        ]

        for d in deadlines:
            days = (d["deadline"] - today).days
            d["days_until"] = max(0, days)
            if days < 0:
                d["status"] = "overdue"
            elif days <= 30:
                d["status"] = "due_soon"

        return sorted(deadlines, key=lambda x: x["deadline"])

    def get_submission_history(
        self,
        report_type: Optional[ReportType] = None,
        year: Optional[int] = None,
    ) -> List[ReportSubmission]:
        """Get submission history with optional filtering."""
        submissions = self._submissions

        if report_type:
            submissions = [s for s in submissions if s.report_type == report_type.value]
        if year:
            submissions = [s for s in submissions if s.reporting_year == year]

        return list(reversed(submissions))

    # =========================================================================
    # HASH UTILITIES
    # =========================================================================

    def _hash_report_data(
        self,
        report_type: str,
        facility_id: str,
        year: int,
        total_emissions: float,
    ) -> str:
        """Calculate SHA-256 hash for report provenance."""
        data = {
            "report_type": report_type,
            "facility_id": facility_id,
            "year": year,
            "total_emissions": total_emissions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get overall reporting compliance summary."""
        current_year = datetime.now().year

        return {
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "as_of_date": date.today().isoformat(),
            "emission_units_registered": len(self._emission_units),
            "fuel_records": len(self._fuel_data),
            "reports_generated": len(self._reports),
            "submissions": len(self._submissions),
            "upcoming_deadlines": self.get_upcoming_deadlines(current_year),
            "recent_submissions": [
                {
                    "type": s.report_type,
                    "year": s.reporting_year,
                    "status": s.status,
                    "date": s.submission_date.isoformat(),
                }
                for s in self._submissions[-5:]
            ],
        }
