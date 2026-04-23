"""
Permit Limits Manager for GL-004 BURNMASTER

Comprehensive permit management system for air quality compliance tracking.
Implements Title V, PSD, NSR, State, and Minor Source permit handling with
full provenance tracking for regulatory audits.

This module implements:
- Permit data models with regulatory basis tracking
- Compliance status determination with averaging period calculations
- Exceedance management and root cause classification
- Automated reporting for excess emissions and deviation reports
- Unit conversions between lb/MMBtu, ppm, mg/m3

Regulatory References:
- 40 CFR Part 70 (Title V Operating Permits)
- 40 CFR Part 52 (Prevention of Significant Deterioration)
- 40 CFR Part 51 (New Source Review)
- 40 CFR Part 60 (NSPS)
- 40 CFR Part 63 (NESHAP)

Author: GL-RegulatoryIntelligence
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import hashlib
import json
import logging
import uuid
from collections import defaultdict

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class Pollutant(str, Enum):
    """Regulated air pollutants."""
    NOX = "NOx"
    SO2 = "SO2"
    PM = "PM"
    PM10 = "PM10"
    PM25 = "PM2.5"
    CO = "CO"
    VOC = "VOC"
    HAP = "HAP"
    NH3 = "NH3"
    H2S = "H2S"
    HCL = "HCl"
    HF = "HF"
    LEAD = "Lead"


class LimitUnits(str, Enum):
    """Emission limit unit types."""
    LB_MMBTU = "lb/MMBtu"
    LB_HR = "lb/hr"
    TONS_YR = "tons/yr"
    PPM = "ppm"
    PPMVD = "ppmvd"
    MG_M3 = "mg/m3"
    UG_M3 = "ug/m3"
    PERCENT = "%"
    GR_DSCF = "gr/dscf"  # Grains per dry standard cubic foot


class AveragingPeriod(str, Enum):
    """Emission averaging periods."""
    INSTANTANEOUS = "instantaneous"
    ONE_HOUR = "1-hr"
    THREE_HOUR = "3-hr"
    EIGHT_HOUR = "8-hr"
    TWENTY_FOUR_HOUR = "24-hr"
    THIRTY_DAY = "30-day"
    ROLLING_30_DAY = "30-day-rolling"
    ROLLING_12_MONTH = "12-month-rolling"
    ANNUAL = "annual"
    CALENDAR_YEAR = "calendar-year"


class RegulatoryBasis(str, Enum):
    """Regulatory basis for permit limits."""
    NSPS = "NSPS"              # New Source Performance Standards
    BACT = "BACT"              # Best Available Control Technology
    LAER = "LAER"              # Lowest Achievable Emission Rate
    RACT = "RACT"              # Reasonably Available Control Technology
    MACT = "MACT"              # Maximum Achievable Control Technology
    NESHAP = "NESHAP"          # National Emission Standards for HAPs
    STATE = "State"            # State-specific requirements
    LOCAL = "Local"            # Local air district requirements
    CONSENT_DECREE = "Consent" # Consent decree / settlement
    VOLUNTARY = "Voluntary"    # Voluntary commitment


class PermitType(str, Enum):
    """Types of air quality permits."""
    TITLE_V = "Title V Operating Permit"
    PSD = "PSD Permit"
    NSR = "NSR Permit"
    STATE_OPERATING = "State Operating Permit"
    MINOR_SOURCE = "Minor Source Permit"
    GENERAL = "General Permit"
    CONSTRUCTION = "Construction Permit"
    TEMPORARY = "Temporary Permit"


class ComplianceStatus(str, Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    APPROACHING = "approaching"       # Within 80-90% of limit
    WARNING = "warning"               # Within 90-100% of limit
    EXCEEDANCE = "exceedance"         # Above limit
    CRITICAL = "critical"             # Significantly above limit (>120%)
    EXEMPT = "exempt"                 # Exempt period (startup/shutdown)
    UNKNOWN = "unknown"               # Insufficient data


class DeviationType(str, Enum):
    """Types of permit deviations."""
    EXCEEDANCE = "exceedance"
    MONITORING_GAP = "monitoring_gap"
    REPORTING_DELAY = "reporting_delay"
    EQUIPMENT_MALFUNCTION = "equipment_malfunction"
    STARTUP_SHUTDOWN = "startup_shutdown"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class RootCauseCategory(str, Enum):
    """Root cause categories for deviations."""
    EQUIPMENT_FAILURE = "equipment_failure"
    PROCESS_UPSET = "process_upset"
    FUEL_QUALITY = "fuel_quality"
    OPERATOR_ERROR = "operator_error"
    DESIGN_LIMITATION = "design_limitation"
    WEATHER = "weather"
    UTILITY_FAILURE = "utility_failure"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class CorrectiveActionStatus(str, Enum):
    """Status of corrective actions."""
    IDENTIFIED = "identified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class PermitLimit:
    """
    Represents a single permit limit condition.

    Attributes:
        pollutant: Pollutant being limited (NOx, SO2, PM, CO, VOC, etc.)
        limit_value: Numerical limit value
        limit_units: Units for the limit (lb/MMBtu, lb/hr, tons/yr, ppm, mg/m3)
        averaging_period: Time period for averaging (1-hr, 3-hr, 24-hr, 30-day, annual)
        emission_point_id: Unique identifier for the emission point
        regulatory_basis: Source of the limit (NSPS, BACT, NESHAP, State)
        effective_date: Date when limit became effective
        expiration_date: Date when limit expires
        permit_number: Associated permit number
        condition_number: Specific permit condition reference
        reference_conditions: Reference conditions for measurement
        compliance_method: Method for demonstrating compliance
        notes: Additional notes or context
    """
    pollutant: str
    limit_value: float
    limit_units: str
    averaging_period: str
    emission_point_id: str
    regulatory_basis: str
    effective_date: datetime
    expiration_date: datetime
    permit_number: str = ""
    condition_number: str = ""
    reference_conditions: Dict[str, Any] = field(default_factory=dict)
    compliance_method: str = "CEMS"
    notes: str = ""
    limit_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Validate and set defaults for reference conditions."""
        if not self.reference_conditions:
            self.reference_conditions = {
                "o2_percent": 3.0,
                "moisture_basis": "dry",
                "temperature_f": 68,
                "pressure_inhg": 29.92
            }

    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """Check if limit is currently active."""
        check_time = at_time or datetime.now(timezone.utc)
        return self.effective_date <= check_time <= self.expiration_date


@dataclass
class EmissionMeasurement:
    """
    Single emission measurement record.

    Attributes:
        emission_point_id: Emission point where measurement was taken
        pollutant: Measured pollutant
        measured_value: Measured concentration/rate
        measurement_units: Units of measurement
        measurement_time: Timestamp of measurement
        measurement_method: Method used (CEMS, stack test, etc.)
        data_quality: Quality flag for the measurement
        o2_measured: Measured oxygen percentage (for normalization)
        is_valid: Whether measurement passes QA/QC
    """
    emission_point_id: str
    pollutant: str
    measured_value: float
    measurement_units: str
    measurement_time: datetime
    measurement_method: str = "CEMS"
    data_quality: str = "valid"
    o2_measured: float = 3.0
    is_valid: bool = True
    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ExemptPeriod:
    """
    Period exempt from certain compliance requirements.

    Used for startup, shutdown, malfunction, and maintenance periods.
    """
    emission_point_id: str
    start_time: datetime
    end_time: Optional[datetime]
    exemption_type: str  # startup, shutdown, malfunction, maintenance
    affected_pollutants: List[str]
    regulatory_basis: str
    notification_sent: bool = False
    notification_time: Optional[datetime] = None
    notes: str = ""
    exemption_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """Check if exemption is active at given time."""
        check_time = at_time or datetime.now(timezone.utc)
        if self.end_time is None:
            return check_time >= self.start_time
        return self.start_time <= check_time <= self.end_time


@dataclass
class Deviation:
    """
    Record of a permit deviation or exceedance.

    Tracks the deviation, root cause analysis, and corrective actions.
    """
    deviation_id: str
    emission_point_id: str
    pollutant: str
    limit_id: str
    deviation_type: str
    start_time: datetime
    end_time: Optional[datetime]
    measured_value: float
    limit_value: float
    limit_units: str
    percent_above_limit: float
    root_cause: str = ""
    root_cause_category: str = RootCauseCategory.UNKNOWN.value
    corrective_actions: List[Dict[str, Any]] = field(default_factory=list)
    preventive_actions: List[Dict[str, Any]] = field(default_factory=list)
    reported_to_agency: bool = False
    report_date: Optional[datetime] = None
    notes: str = ""

    def duration_hours(self) -> float:
        """Calculate duration of deviation in hours."""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 3600


@dataclass
class ComplianceCheckResult:
    """
    Result of a compliance check against permit limits.
    """
    check_id: str
    check_time: datetime
    emission_point_id: str
    pollutant: str
    limit_id: str
    limit_value: float
    limit_units: str
    averaging_period: str
    calculated_value: float
    compliance_status: str
    percent_of_limit: float
    margin_to_limit: float
    is_exempt: bool
    exemption_reason: str
    data_points_used: int
    data_coverage_percent: float
    provenance_hash: str
    calculation_details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Pydantic Models for API
# =============================================================================

class PermitLimitInput(BaseModel):
    """Input model for creating permit limits."""
    pollutant: str = Field(..., description="Pollutant being limited")
    limit_value: float = Field(..., gt=0, description="Limit value")
    limit_units: str = Field(..., description="Units for the limit")
    averaging_period: str = Field(..., description="Averaging period")
    emission_point_id: str = Field(..., description="Emission point ID")
    regulatory_basis: str = Field(..., description="Regulatory basis")
    effective_date: datetime = Field(..., description="Effective date")
    expiration_date: datetime = Field(..., description="Expiration date")
    permit_number: str = Field(default="", description="Permit number")
    condition_number: str = Field(default="", description="Condition number")
    reference_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reference conditions"
    )


class ComplianceCheckInput(BaseModel):
    """Input model for compliance checks."""
    emission_point_id: str = Field(..., description="Emission point ID")
    pollutant: str = Field(..., description="Pollutant to check")
    measured_value: float = Field(..., ge=0, description="Measured value")
    measurement_units: str = Field(..., description="Measurement units")
    measurement_time: datetime = Field(..., description="Measurement time")
    o2_measured: float = Field(default=3.0, ge=0, le=21, description="Measured O2%")


class DeviationReportOutput(BaseModel):
    """Output model for deviation reports."""
    report_id: str = Field(..., description="Report identifier")
    report_type: str = Field(..., description="Type of report")
    reporting_period_start: datetime = Field(..., description="Period start")
    reporting_period_end: datetime = Field(..., description="Period end")
    facility_name: str = Field(..., description="Facility name")
    permit_number: str = Field(..., description="Permit number")
    total_deviations: int = Field(..., description="Total deviation count")
    deviations: List[Dict[str, Any]] = Field(..., description="Deviation details")
    certification: str = Field(..., description="Compliance certification")
    generated_time: datetime = Field(..., description="Report generation time")
    provenance_hash: str = Field(..., description="Provenance hash")


# =============================================================================
# Unit Conversion Engine
# =============================================================================

class EmissionUnitConverter:
    """
    Deterministic unit conversion for emission measurements.

    Supports conversions between:
    - lb/MMBtu <-> ppm <-> mg/m3
    - Reference condition adjustments (dry, 3% O2, etc.)
    - Stack test correlation factors

    All conversions are traceable with provenance hashing.
    """

    # Molecular weights (g/mol) - NO2 basis for NOx
    MOLECULAR_WEIGHTS: Dict[str, float] = {
        "NOx": 46.0,
        "SO2": 64.066,
        "CO": 28.01,
        "PM": 1.0,  # Not applicable for mass conversion
        "VOC": 78.0,  # Approximation, varies by compound
        "NH3": 17.031,
        "H2S": 34.08,
        "HCl": 36.46,
    }

    # Standard conditions
    STANDARD_TEMP_K = 273.15  # 0 deg C
    STANDARD_PRESSURE_KPA = 101.325
    MOLAR_VOLUME_NM3 = 22.414  # L/mol at STP

    # Fuel heating values (MMBtu/1000 scf for gas, MMBtu/gal for oil)
    FUEL_HEATING_VALUES: Dict[str, float] = {
        "natural_gas": 1.02,  # MMBtu/1000 scf
        "fuel_oil_2": 0.138,  # MMBtu/gal
        "fuel_oil_6": 0.150,  # MMBtu/gal
    }

    # F-factors for combustion (dscf/MMBtu)
    F_FACTORS: Dict[str, float] = {
        "natural_gas": 8710,
        "fuel_oil_2": 9190,
        "fuel_oil_6": 9220,
        "coal_bituminous": 9780,
        "coal_subbituminous": 9820,
    }

    def __init__(self, precision: int = 4):
        """Initialize converter with precision settings."""
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def normalize_to_reference_o2(
        self,
        concentration: float,
        measured_o2: float,
        reference_o2: float = 3.0
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Normalize concentration to reference O2 level.

        Formula: C_ref = C_meas * (21 - O2_ref) / (21 - O2_meas)

        Args:
            concentration: Measured concentration
            measured_o2: Measured O2 percentage
            reference_o2: Reference O2 percentage

        Returns:
            Tuple of (normalized_value, calculation_details)
        """
        if measured_o2 >= 21.0:
            return Decimal('0'), {"error": "measured_o2 >= 21%"}

        factor = (21.0 - reference_o2) / (21.0 - measured_o2)
        normalized = concentration * factor

        details = {
            "input_concentration": concentration,
            "measured_o2": measured_o2,
            "reference_o2": reference_o2,
            "correction_factor": round(factor, 6),
            "normalized_concentration": round(normalized, self.precision),
            "formula": "C_meas * (21 - O2_ref) / (21 - O2_meas)"
        }

        return self._quantize(Decimal(str(normalized))), details

    def ppm_to_lb_mmbtu(
        self,
        ppm: float,
        pollutant: str,
        fuel_type: str = "natural_gas",
        measured_o2: float = 3.0,
        reference_o2: float = 3.0
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Convert ppm to lb/MMBtu.

        Formula: lb/MMBtu = ppm * MW * Fd / (385.5 * 10^6)
        Where: Fd = F-factor (dscf/MMBtu)
               MW = molecular weight (lb/lb-mol)
               385.5 = molar volume at std conditions (scf/lb-mol)

        Args:
            ppm: Concentration in ppm (volumetric, dry)
            pollutant: Pollutant type
            fuel_type: Fuel type for F-factor
            measured_o2: Measured O2 percentage
            reference_o2: Reference O2 percentage

        Returns:
            Tuple of (lb_mmbtu, calculation_details)
        """
        # Get molecular weight and F-factor
        mw = self.MOLECULAR_WEIGHTS.get(pollutant, 46.0)
        fd = self.F_FACTORS.get(fuel_type, 8710)

        # Normalize to reference O2 if needed
        if measured_o2 != reference_o2:
            ppm_normalized, norm_details = self.normalize_to_reference_o2(
                ppm, measured_o2, reference_o2
            )
            ppm = float(ppm_normalized)
        else:
            norm_details = {"normalization": "not_required"}

        # Convert: lb/MMBtu = ppm * MW * Fd / (385.5 * 10^6)
        # Note: MW in g/mol, need to convert to lb/lb-mol (same value)
        molar_volume_scf = 385.5  # scf/lb-mol at 68F, 29.92 inHg

        lb_mmbtu = ppm * mw * fd / (molar_volume_scf * 1e6)

        details = {
            "input_ppm": ppm,
            "pollutant": pollutant,
            "molecular_weight": mw,
            "fuel_type": fuel_type,
            "f_factor_dscf_mmbtu": fd,
            "molar_volume_scf": molar_volume_scf,
            "lb_mmbtu": round(lb_mmbtu, self.precision),
            "formula": "ppm * MW * Fd / (385.5 * 10^6)",
            "normalization": norm_details
        }

        return self._quantize(Decimal(str(lb_mmbtu))), details

    def lb_mmbtu_to_ppm(
        self,
        lb_mmbtu: float,
        pollutant: str,
        fuel_type: str = "natural_gas"
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Convert lb/MMBtu to ppm.

        Inverse of ppm_to_lb_mmbtu.

        Args:
            lb_mmbtu: Emission rate in lb/MMBtu
            pollutant: Pollutant type
            fuel_type: Fuel type for F-factor

        Returns:
            Tuple of (ppm, calculation_details)
        """
        mw = self.MOLECULAR_WEIGHTS.get(pollutant, 46.0)
        fd = self.F_FACTORS.get(fuel_type, 8710)
        molar_volume_scf = 385.5

        ppm = lb_mmbtu * molar_volume_scf * 1e6 / (mw * fd)

        details = {
            "input_lb_mmbtu": lb_mmbtu,
            "pollutant": pollutant,
            "molecular_weight": mw,
            "fuel_type": fuel_type,
            "f_factor_dscf_mmbtu": fd,
            "ppm": round(ppm, self.precision),
            "formula": "lb_mmbtu * 385.5 * 10^6 / (MW * Fd)"
        }

        return self._quantize(Decimal(str(ppm))), details

    def ppm_to_mg_m3(
        self,
        ppm: float,
        pollutant: str,
        temperature_c: float = 0.0,
        pressure_kpa: float = 101.325
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Convert ppm to mg/m3 at specified conditions.

        Formula: mg/m3 = ppm * MW * P / (R * T)
        Where: R = 8.314 J/(mol*K)

        Args:
            ppm: Concentration in ppm (volumetric)
            pollutant: Pollutant type
            temperature_c: Temperature in Celsius
            pressure_kpa: Pressure in kPa

        Returns:
            Tuple of (mg_m3, calculation_details)
        """
        mw = self.MOLECULAR_WEIGHTS.get(pollutant, 46.0)
        temp_k = temperature_c + 273.15

        # Ideal gas law: mg/m3 = ppm * MW * P / (R * T) * (1000 mg/g)
        # Simplified at STP: mg/m3 = ppm * MW / 22.414
        # With temp/pressure correction:
        mg_m3 = ppm * mw / 22.414 * (273.15 / temp_k) * (pressure_kpa / 101.325)

        details = {
            "input_ppm": ppm,
            "pollutant": pollutant,
            "molecular_weight": mw,
            "temperature_c": temperature_c,
            "temperature_k": temp_k,
            "pressure_kpa": pressure_kpa,
            "mg_m3": round(mg_m3, self.precision),
            "formula": "ppm * MW / 22.414 * (273.15/T) * (P/101.325)"
        }

        return self._quantize(Decimal(str(mg_m3))), details

    def mg_m3_to_ppm(
        self,
        mg_m3: float,
        pollutant: str,
        temperature_c: float = 0.0,
        pressure_kpa: float = 101.325
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Convert mg/m3 to ppm at specified conditions.

        Inverse of ppm_to_mg_m3.
        """
        mw = self.MOLECULAR_WEIGHTS.get(pollutant, 46.0)
        temp_k = temperature_c + 273.15

        ppm = mg_m3 * 22.414 / mw * (temp_k / 273.15) * (101.325 / pressure_kpa)

        details = {
            "input_mg_m3": mg_m3,
            "pollutant": pollutant,
            "molecular_weight": mw,
            "temperature_c": temperature_c,
            "pressure_kpa": pressure_kpa,
            "ppm": round(ppm, self.precision),
            "formula": "mg_m3 * 22.414 / MW * (T/273.15) * (101.325/P)"
        }

        return self._quantize(Decimal(str(ppm))), details

    def convert_units(
        self,
        value: float,
        from_units: str,
        to_units: str,
        pollutant: str,
        fuel_type: str = "natural_gas",
        measured_o2: float = 3.0,
        reference_o2: float = 3.0,
        temperature_c: float = 0.0,
        pressure_kpa: float = 101.325
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Convert between any supported unit types.

        Args:
            value: Input value
            from_units: Source units
            to_units: Target units
            pollutant: Pollutant type
            fuel_type: Fuel type (for lb/MMBtu conversions)
            measured_o2: Measured O2 for normalization
            reference_o2: Reference O2 for normalization
            temperature_c: Temperature for mg/m3 conversions
            pressure_kpa: Pressure for mg/m3 conversions

        Returns:
            Tuple of (converted_value, calculation_details)
        """
        # Normalize unit strings
        from_u = from_units.lower().replace(" ", "")
        to_u = to_units.lower().replace(" ", "")

        # Same units - no conversion needed
        if from_u == to_u:
            return self._quantize(Decimal(str(value))), {"conversion": "none_required"}

        details: Dict[str, Any] = {
            "input_value": value,
            "from_units": from_units,
            "to_units": to_units,
            "pollutant": pollutant,
            "steps": []
        }

        # Convert to ppm first (common intermediate)
        ppm_value = value

        if from_u in ["lb/mmbtu", "lbmmbtu"]:
            ppm_value, step = self.lb_mmbtu_to_ppm(value, pollutant, fuel_type)
            ppm_value = float(ppm_value)
            details["steps"].append({"lb_mmbtu_to_ppm": step})
        elif from_u in ["mg/m3", "mgm3", "mg/nm3"]:
            ppm_value, step = self.mg_m3_to_ppm(value, pollutant, temperature_c, pressure_kpa)
            ppm_value = float(ppm_value)
            details["steps"].append({"mg_m3_to_ppm": step})
        elif from_u not in ["ppm", "ppmvd", "ppmv"]:
            return Decimal('0'), {"error": f"Unsupported from_units: {from_units}"}

        # Now convert from ppm to target
        result = Decimal(str(ppm_value))

        if to_u in ["lb/mmbtu", "lbmmbtu"]:
            result, step = self.ppm_to_lb_mmbtu(
                ppm_value, pollutant, fuel_type, measured_o2, reference_o2
            )
            details["steps"].append({"ppm_to_lb_mmbtu": step})
        elif to_u in ["mg/m3", "mgm3", "mg/nm3"]:
            result, step = self.ppm_to_mg_m3(ppm_value, pollutant, temperature_c, pressure_kpa)
            details["steps"].append({"ppm_to_mg_m3": step})
        elif to_u not in ["ppm", "ppmvd", "ppmv"]:
            return Decimal('0'), {"error": f"Unsupported to_units: {to_units}"}

        details["output_value"] = float(result)
        details["provenance_hash"] = self._compute_hash(details)

        return result, details


# =============================================================================
# Averaging Period Calculator
# =============================================================================

class AveragingCalculator:
    """
    Calculates emissions averages for various regulatory periods.

    Supports:
    - Block averages (24-hr, calendar day)
    - Rolling averages (30-day, 12-month)
    - Annual totals
    - Startup/shutdown exclusions
    """

    def __init__(self, precision: int = 2):
        """Initialize calculator."""
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def calculate_block_average(
        self,
        measurements: List[EmissionMeasurement],
        block_hours: float = 24.0,
        reference_time: Optional[datetime] = None,
        exclude_exemptions: Optional[List[ExemptPeriod]] = None
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Calculate block average for a specified period.

        Args:
            measurements: List of measurements
            block_hours: Block period in hours
            reference_time: End time for the block
            exclude_exemptions: Exempt periods to exclude

        Returns:
            Tuple of (average_value, calculation_details)
        """
        ref_time = reference_time or datetime.now(timezone.utc)
        block_start = ref_time - timedelta(hours=block_hours)

        # Filter measurements within block period
        valid_measurements = []
        excluded_count = 0

        for m in measurements:
            if block_start <= m.measurement_time <= ref_time:
                if not m.is_valid:
                    excluded_count += 1
                    continue

                # Check exemptions
                is_exempt = False
                if exclude_exemptions:
                    for exemption in exclude_exemptions:
                        if (exemption.is_active(m.measurement_time) and
                            m.pollutant in exemption.affected_pollutants):
                            is_exempt = True
                            excluded_count += 1
                            break

                if not is_exempt:
                    valid_measurements.append(m)

        # Calculate average
        if not valid_measurements:
            return Decimal('0'), {
                "error": "no_valid_measurements",
                "block_start": block_start.isoformat(),
                "block_end": ref_time.isoformat()
            }

        total = sum(m.measured_value for m in valid_measurements)
        average = total / len(valid_measurements)

        details = {
            "block_start": block_start.isoformat(),
            "block_end": ref_time.isoformat(),
            "block_hours": block_hours,
            "total_measurements": len(measurements),
            "valid_measurements": len(valid_measurements),
            "excluded_measurements": excluded_count,
            "average_value": round(average, self.precision),
            "min_value": round(min(m.measured_value for m in valid_measurements), self.precision),
            "max_value": round(max(m.measured_value for m in valid_measurements), self.precision),
            "data_coverage_percent": round(len(valid_measurements) / max(1, len(measurements)) * 100, 1)
        }
        details["provenance_hash"] = self._compute_hash(details)

        return self._quantize(Decimal(str(average))), details

    def calculate_rolling_average(
        self,
        measurements: List[EmissionMeasurement],
        window_days: int = 30,
        reference_time: Optional[datetime] = None,
        minimum_data_coverage: float = 0.75,
        exclude_exemptions: Optional[List[ExemptPeriod]] = None
    ) -> Tuple[Optional[Decimal], Dict[str, Any]]:
        """
        Calculate rolling average over specified window.

        Args:
            measurements: List of measurements (should span at least window_days)
            window_days: Rolling window in days
            reference_time: End time for the window
            minimum_data_coverage: Minimum required data coverage (0-1)
            exclude_exemptions: Exempt periods to exclude

        Returns:
            Tuple of (average_value or None, calculation_details)
        """
        ref_time = reference_time or datetime.now(timezone.utc)
        window_start = ref_time - timedelta(days=window_days)

        # Group measurements by day
        daily_averages: Dict[str, List[float]] = defaultdict(list)
        excluded_hours = 0

        for m in measurements:
            if window_start <= m.measurement_time <= ref_time:
                if not m.is_valid:
                    continue

                # Check exemptions
                is_exempt = False
                if exclude_exemptions:
                    for exemption in exclude_exemptions:
                        if (exemption.is_active(m.measurement_time) and
                            m.pollutant in exemption.affected_pollutants):
                            is_exempt = True
                            excluded_hours += 1
                            break

                if not is_exempt:
                    day_key = m.measurement_time.strftime("%Y-%m-%d")
                    daily_averages[day_key].append(m.measured_value)

        # Calculate daily averages, then overall average
        days_with_data = len(daily_averages)
        required_days = int(window_days * minimum_data_coverage)

        if days_with_data < required_days:
            return None, {
                "error": "insufficient_data_coverage",
                "days_with_data": days_with_data,
                "required_days": required_days,
                "window_days": window_days,
                "data_coverage_percent": round(days_with_data / window_days * 100, 1)
            }

        # Calculate rolling average
        daily_means = [sum(vals) / len(vals) for vals in daily_averages.values()]
        rolling_avg = sum(daily_means) / len(daily_means)

        details = {
            "window_start": window_start.isoformat(),
            "window_end": ref_time.isoformat(),
            "window_days": window_days,
            "days_with_data": days_with_data,
            "excluded_exempt_hours": excluded_hours,
            "rolling_average": round(rolling_avg, self.precision),
            "daily_min": round(min(daily_means), self.precision),
            "daily_max": round(max(daily_means), self.precision),
            "data_coverage_percent": round(days_with_data / window_days * 100, 1),
            "minimum_required_coverage": minimum_data_coverage * 100
        }
        details["provenance_hash"] = self._compute_hash(details)

        return self._quantize(Decimal(str(rolling_avg))), details

    def calculate_annual_total(
        self,
        measurements: List[EmissionMeasurement],
        year: Optional[int] = None,
        emission_rate_units: str = "lb/hr",
        exclude_exemptions: Optional[List[ExemptPeriod]] = None
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Calculate annual emission total in tons/year.

        Args:
            measurements: List of hourly emission rate measurements
            year: Calendar year (defaults to current year)
            emission_rate_units: Units of input measurements
            exclude_exemptions: Exempt periods to exclude

        Returns:
            Tuple of (annual_tons, calculation_details)
        """
        target_year = year or datetime.now().year
        year_start = datetime(target_year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(target_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        # Sum hourly emissions
        total_lb = 0.0
        hours_counted = 0
        hours_excluded = 0

        for m in measurements:
            if year_start <= m.measurement_time <= year_end:
                if not m.is_valid:
                    hours_excluded += 1
                    continue

                # Check exemptions
                is_exempt = False
                if exclude_exemptions:
                    for exemption in exclude_exemptions:
                        if (exemption.is_active(m.measurement_time) and
                            m.pollutant in exemption.affected_pollutants):
                            is_exempt = True
                            hours_excluded += 1
                            break

                if not is_exempt:
                    # Assume each measurement represents 1 hour
                    if emission_rate_units.lower() == "lb/hr":
                        total_lb += m.measured_value
                    elif emission_rate_units.lower() == "kg/hr":
                        total_lb += m.measured_value * 2.20462
                    hours_counted += 1

        # Convert lb to tons
        annual_tons = total_lb / 2000

        details = {
            "year": target_year,
            "hours_counted": hours_counted,
            "hours_excluded": hours_excluded,
            "total_lb": round(total_lb, 2),
            "annual_tons": round(annual_tons, 2),
            "average_lb_hr": round(total_lb / max(1, hours_counted), 2),
            "data_coverage_percent": round(hours_counted / 8760 * 100, 1)
        }
        details["provenance_hash"] = self._compute_hash(details)

        return self._quantize(Decimal(str(annual_tons))), details


# =============================================================================
# Permit Limits Manager
# =============================================================================

class PermitLimitsManager:
    """
    Comprehensive permit limits management system.

    Manages permit conditions, compliance checking, exceedance tracking,
    and regulatory reporting with full provenance tracking.

    Example:
        >>> manager = PermitLimitsManager(facility_name="Power Plant A")
        >>> manager.add_permit_limit(limit)
        >>> result = manager.check_compliance("EP-001", "NOx", 0.05, datetime.now())
        >>> report = manager.generate_deviation_report(start, end)
    """

    def __init__(
        self,
        facility_name: str,
        facility_id: str = "",
        permit_number: str = ""
    ):
        """
        Initialize permit limits manager.

        Args:
            facility_name: Name of the facility
            facility_id: Unique facility identifier
            permit_number: Primary permit number
        """
        self.facility_name = facility_name
        self.facility_id = facility_id or str(uuid.uuid4())
        self.permit_number = permit_number

        # Storage
        self._limits: Dict[str, PermitLimit] = {}
        self._measurements: List[EmissionMeasurement] = []
        self._exemptions: List[ExemptPeriod] = []
        self._deviations: Dict[str, Deviation] = {}
        self._compliance_history: List[ComplianceCheckResult] = []

        # Calculators
        self._unit_converter = EmissionUnitConverter()
        self._averaging_calculator = AveragingCalculator()

        logger.info(
            f"PermitLimitsManager initialized for {facility_name} "
            f"(ID: {self.facility_id})"
        )

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Permit Limit Management
    # -------------------------------------------------------------------------

    def add_permit_limit(self, limit: PermitLimit) -> str:
        """
        Add a permit limit to the manager.

        Args:
            limit: PermitLimit to add

        Returns:
            Limit ID
        """
        self._limits[limit.limit_id] = limit
        logger.info(
            f"Added permit limit {limit.limit_id}: {limit.pollutant} "
            f"{limit.limit_value} {limit.limit_units} ({limit.averaging_period})"
        )
        return limit.limit_id

    def get_limit(self, limit_id: str) -> Optional[PermitLimit]:
        """Get a permit limit by ID."""
        return self._limits.get(limit_id)

    def get_limits_for_emission_point(
        self,
        emission_point_id: str,
        pollutant: Optional[str] = None,
        active_only: bool = True
    ) -> List[PermitLimit]:
        """
        Get all limits applicable to an emission point.

        Args:
            emission_point_id: Emission point identifier
            pollutant: Optional pollutant filter
            active_only: Only return active limits

        Returns:
            List of applicable PermitLimit objects
        """
        limits = []
        for limit in self._limits.values():
            if limit.emission_point_id != emission_point_id:
                continue
            if pollutant and limit.pollutant != pollutant:
                continue
            if active_only and not limit.is_active():
                continue
            limits.append(limit)
        return limits

    def remove_permit_limit(self, limit_id: str) -> bool:
        """Remove a permit limit."""
        if limit_id in self._limits:
            del self._limits[limit_id]
            logger.info(f"Removed permit limit {limit_id}")
            return True
        return False

    # -------------------------------------------------------------------------
    # Measurement Recording
    # -------------------------------------------------------------------------

    def record_measurement(self, measurement: EmissionMeasurement) -> str:
        """
        Record an emission measurement.

        Args:
            measurement: Measurement to record

        Returns:
            Measurement ID
        """
        self._measurements.append(measurement)
        return measurement.measurement_id

    def get_measurements(
        self,
        emission_point_id: Optional[str] = None,
        pollutant: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[EmissionMeasurement]:
        """
        Get measurements with optional filters.

        Args:
            emission_point_id: Filter by emission point
            pollutant: Filter by pollutant
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of matching measurements
        """
        results = []
        for m in self._measurements:
            if emission_point_id and m.emission_point_id != emission_point_id:
                continue
            if pollutant and m.pollutant != pollutant:
                continue
            if start_time and m.measurement_time < start_time:
                continue
            if end_time and m.measurement_time > end_time:
                continue
            results.append(m)
        return results

    # -------------------------------------------------------------------------
    # Exemption Management
    # -------------------------------------------------------------------------

    def add_exemption(self, exemption: ExemptPeriod) -> str:
        """
        Add an exempt period (startup, shutdown, malfunction).

        Args:
            exemption: ExemptPeriod to add

        Returns:
            Exemption ID
        """
        self._exemptions.append(exemption)
        logger.info(
            f"Added exemption {exemption.exemption_id}: {exemption.exemption_type} "
            f"for {exemption.emission_point_id}"
        )
        return exemption.exemption_id

    def end_exemption(
        self,
        exemption_id: str,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        End an active exemption period.

        Args:
            exemption_id: ID of exemption to end
            end_time: End timestamp (defaults to now)

        Returns:
            True if exemption was found and ended
        """
        for exemption in self._exemptions:
            if exemption.exemption_id == exemption_id:
                exemption.end_time = end_time or datetime.now(timezone.utc)
                logger.info(f"Ended exemption {exemption_id}")
                return True
        return False

    def get_active_exemptions(
        self,
        emission_point_id: Optional[str] = None,
        at_time: Optional[datetime] = None
    ) -> List[ExemptPeriod]:
        """Get active exemptions at specified time."""
        check_time = at_time or datetime.now(timezone.utc)
        active = []
        for exemption in self._exemptions:
            if emission_point_id and exemption.emission_point_id != emission_point_id:
                continue
            if exemption.is_active(check_time):
                active.append(exemption)
        return active

    # -------------------------------------------------------------------------
    # Compliance Checking
    # -------------------------------------------------------------------------

    def check_compliance(
        self,
        emission_point_id: str,
        pollutant: str,
        measured_value: float,
        measurement_time: datetime,
        measurement_units: str = "ppm",
        o2_measured: float = 3.0
    ) -> ComplianceCheckResult:
        """
        Check compliance against applicable permit limits.

        Determines compliance status for a single measurement point,
        considering all applicable limits and exemptions.

        Args:
            emission_point_id: Emission point identifier
            pollutant: Pollutant being checked
            measured_value: Measured concentration/rate
            measurement_time: Time of measurement
            measurement_units: Units of measurement
            o2_measured: Measured O2 percentage (for normalization)

        Returns:
            ComplianceCheckResult with status and details
        """
        check_id = str(uuid.uuid4())

        # Get applicable limits
        limits = self.get_limits_for_emission_point(
            emission_point_id, pollutant, active_only=True
        )

        if not limits:
            return ComplianceCheckResult(
                check_id=check_id,
                check_time=datetime.now(timezone.utc),
                emission_point_id=emission_point_id,
                pollutant=pollutant,
                limit_id="",
                limit_value=0.0,
                limit_units="",
                averaging_period="",
                calculated_value=measured_value,
                compliance_status=ComplianceStatus.UNKNOWN.value,
                percent_of_limit=0.0,
                margin_to_limit=0.0,
                is_exempt=False,
                exemption_reason="",
                data_points_used=1,
                data_coverage_percent=100.0,
                provenance_hash=self._compute_hash({"error": "no_applicable_limits"})
            )

        # Check exemptions
        exemptions = self.get_active_exemptions(emission_point_id, measurement_time)
        is_exempt = False
        exemption_reason = ""
        for exemption in exemptions:
            if pollutant in exemption.affected_pollutants:
                is_exempt = True
                exemption_reason = exemption.exemption_type
                break

        # Use the most stringent limit
        most_stringent_limit = min(limits, key=lambda l: l.limit_value)
        limit = most_stringent_limit

        # Convert units if necessary
        if measurement_units.lower() != limit.limit_units.lower():
            converted_value, _ = self._unit_converter.convert_units(
                measured_value,
                measurement_units,
                limit.limit_units,
                pollutant,
                measured_o2=o2_measured,
                reference_o2=limit.reference_conditions.get("o2_percent", 3.0)
            )
            calculated_value = float(converted_value)
        else:
            # Normalize O2 if needed
            ref_o2 = limit.reference_conditions.get("o2_percent", 3.0)
            if o2_measured != ref_o2:
                normalized, _ = self._unit_converter.normalize_to_reference_o2(
                    measured_value, o2_measured, ref_o2
                )
                calculated_value = float(normalized)
            else:
                calculated_value = measured_value

        # Calculate percent of limit and margin
        percent_of_limit = (calculated_value / limit.limit_value) * 100 if limit.limit_value > 0 else 0
        margin_to_limit = limit.limit_value - calculated_value

        # Determine compliance status
        if is_exempt:
            status = ComplianceStatus.EXEMPT
        elif percent_of_limit < 80:
            status = ComplianceStatus.COMPLIANT
        elif percent_of_limit < 90:
            status = ComplianceStatus.APPROACHING
        elif percent_of_limit < 100:
            status = ComplianceStatus.WARNING
        elif percent_of_limit < 120:
            status = ComplianceStatus.EXCEEDANCE
        else:
            status = ComplianceStatus.CRITICAL

        # Create compliance check result
        calculation_details = {
            "original_value": measured_value,
            "original_units": measurement_units,
            "calculated_value": round(calculated_value, 4),
            "limit_value": limit.limit_value,
            "limit_units": limit.limit_units,
            "percent_of_limit": round(percent_of_limit, 2),
            "o2_measured": o2_measured,
            "o2_reference": limit.reference_conditions.get("o2_percent", 3.0)
        }

        result = ComplianceCheckResult(
            check_id=check_id,
            check_time=datetime.now(timezone.utc),
            emission_point_id=emission_point_id,
            pollutant=pollutant,
            limit_id=limit.limit_id,
            limit_value=limit.limit_value,
            limit_units=limit.limit_units,
            averaging_period=limit.averaging_period,
            calculated_value=calculated_value,
            compliance_status=status.value,
            percent_of_limit=round(percent_of_limit, 2),
            margin_to_limit=round(margin_to_limit, 4),
            is_exempt=is_exempt,
            exemption_reason=exemption_reason,
            data_points_used=1,
            data_coverage_percent=100.0,
            provenance_hash=self._compute_hash(calculation_details),
            calculation_details=calculation_details
        )

        # Store in compliance history
        self._compliance_history.append(result)

        # Detect and record deviation if exceedance
        if status in [ComplianceStatus.EXCEEDANCE, ComplianceStatus.CRITICAL]:
            self._record_deviation(result, limit, measurement_time)

        logger.info(
            f"Compliance check {check_id}: {pollutant} at {emission_point_id} = "
            f"{calculated_value:.4f} {limit.limit_units} ({percent_of_limit:.1f}% of limit) - {status.value}"
        )

        return result

    def check_averaging_period_compliance(
        self,
        emission_point_id: str,
        pollutant: str,
        end_time: Optional[datetime] = None
    ) -> List[ComplianceCheckResult]:
        """
        Check compliance for all averaging periods for an emission point.

        Calculates block averages, rolling averages, and annual totals
        as required by applicable limits.

        Args:
            emission_point_id: Emission point identifier
            pollutant: Pollutant to check
            end_time: End time for averaging periods

        Returns:
            List of ComplianceCheckResult for each applicable period
        """
        ref_time = end_time or datetime.now(timezone.utc)
        results = []

        # Get applicable limits
        limits = self.get_limits_for_emission_point(
            emission_point_id, pollutant, active_only=True
        )

        # Get measurements
        measurements = self.get_measurements(
            emission_point_id=emission_point_id,
            pollutant=pollutant,
            start_time=ref_time - timedelta(days=365)  # Get a year of data
        )

        # Get exemptions
        exemptions = self._exemptions

        for limit in limits:
            check_id = str(uuid.uuid4())

            # Calculate appropriate average based on averaging period
            if limit.averaging_period == AveragingPeriod.ONE_HOUR.value:
                avg, details = self._averaging_calculator.calculate_block_average(
                    measurements, 1.0, ref_time, exemptions
                )
            elif limit.averaging_period == AveragingPeriod.THREE_HOUR.value:
                avg, details = self._averaging_calculator.calculate_block_average(
                    measurements, 3.0, ref_time, exemptions
                )
            elif limit.averaging_period == AveragingPeriod.TWENTY_FOUR_HOUR.value:
                avg, details = self._averaging_calculator.calculate_block_average(
                    measurements, 24.0, ref_time, exemptions
                )
            elif limit.averaging_period == AveragingPeriod.THIRTY_DAY.value:
                avg, details = self._averaging_calculator.calculate_rolling_average(
                    measurements, 30, ref_time, 0.75, exemptions
                )
            elif limit.averaging_period == AveragingPeriod.ROLLING_30_DAY.value:
                avg, details = self._averaging_calculator.calculate_rolling_average(
                    measurements, 30, ref_time, 0.75, exemptions
                )
            elif limit.averaging_period == AveragingPeriod.ROLLING_12_MONTH.value:
                avg, details = self._averaging_calculator.calculate_rolling_average(
                    measurements, 365, ref_time, 0.75, exemptions
                )
            elif limit.averaging_period in [
                AveragingPeriod.ANNUAL.value,
                AveragingPeriod.CALENDAR_YEAR.value
            ]:
                avg, details = self._averaging_calculator.calculate_annual_total(
                    measurements, ref_time.year, "lb/hr", exemptions
                )
            else:
                continue

            if avg is None:
                continue

            calculated_value = float(avg)
            percent_of_limit = (calculated_value / limit.limit_value) * 100 if limit.limit_value > 0 else 0
            margin_to_limit = limit.limit_value - calculated_value

            # Determine status
            if percent_of_limit < 80:
                status = ComplianceStatus.COMPLIANT
            elif percent_of_limit < 90:
                status = ComplianceStatus.APPROACHING
            elif percent_of_limit < 100:
                status = ComplianceStatus.WARNING
            elif percent_of_limit < 120:
                status = ComplianceStatus.EXCEEDANCE
            else:
                status = ComplianceStatus.CRITICAL

            result = ComplianceCheckResult(
                check_id=check_id,
                check_time=datetime.now(timezone.utc),
                emission_point_id=emission_point_id,
                pollutant=pollutant,
                limit_id=limit.limit_id,
                limit_value=limit.limit_value,
                limit_units=limit.limit_units,
                averaging_period=limit.averaging_period,
                calculated_value=calculated_value,
                compliance_status=status.value,
                percent_of_limit=round(percent_of_limit, 2),
                margin_to_limit=round(margin_to_limit, 4),
                is_exempt=False,
                exemption_reason="",
                data_points_used=details.get("valid_measurements", details.get("days_with_data", 0)),
                data_coverage_percent=details.get("data_coverage_percent", 0.0),
                provenance_hash=details.get("provenance_hash", ""),
                calculation_details=details
            )

            results.append(result)
            self._compliance_history.append(result)

        return results

    # -------------------------------------------------------------------------
    # Deviation Management
    # -------------------------------------------------------------------------

    def _record_deviation(
        self,
        compliance_result: ComplianceCheckResult,
        limit: PermitLimit,
        event_time: datetime
    ) -> str:
        """Record a deviation based on compliance check result."""
        deviation_id = str(uuid.uuid4())

        deviation = Deviation(
            deviation_id=deviation_id,
            emission_point_id=compliance_result.emission_point_id,
            pollutant=compliance_result.pollutant,
            limit_id=limit.limit_id,
            deviation_type=DeviationType.EXCEEDANCE.value,
            start_time=event_time,
            end_time=None,
            measured_value=compliance_result.calculated_value,
            limit_value=limit.limit_value,
            limit_units=limit.limit_units,
            percent_above_limit=max(0, compliance_result.percent_of_limit - 100)
        )

        self._deviations[deviation_id] = deviation
        logger.warning(
            f"Deviation recorded: {deviation_id} - {compliance_result.pollutant} "
            f"at {compliance_result.calculated_value:.4f} {limit.limit_units} "
            f"({compliance_result.percent_of_limit:.1f}% of limit)"
        )
        return deviation_id

    def update_deviation_root_cause(
        self,
        deviation_id: str,
        root_cause: str,
        root_cause_category: str,
        corrective_actions: Optional[List[Dict[str, Any]]] = None,
        preventive_actions: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Update root cause analysis for a deviation.

        Args:
            deviation_id: Deviation to update
            root_cause: Root cause description
            root_cause_category: Category of root cause
            corrective_actions: List of corrective actions taken
            preventive_actions: List of preventive actions planned

        Returns:
            True if deviation was updated
        """
        if deviation_id not in self._deviations:
            return False

        deviation = self._deviations[deviation_id]
        deviation.root_cause = root_cause
        deviation.root_cause_category = root_cause_category

        if corrective_actions:
            deviation.corrective_actions = corrective_actions
        if preventive_actions:
            deviation.preventive_actions = preventive_actions

        logger.info(f"Updated root cause for deviation {deviation_id}: {root_cause_category}")
        return True

    def close_deviation(
        self,
        deviation_id: str,
        end_time: Optional[datetime] = None
    ) -> bool:
        """Close a deviation by setting end time."""
        if deviation_id not in self._deviations:
            return False

        self._deviations[deviation_id].end_time = end_time or datetime.now(timezone.utc)
        logger.info(f"Closed deviation {deviation_id}")
        return True

    def get_deviations(
        self,
        emission_point_id: Optional[str] = None,
        pollutant: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        open_only: bool = False
    ) -> List[Deviation]:
        """Get deviations with optional filters."""
        results = []
        for deviation in self._deviations.values():
            if emission_point_id and deviation.emission_point_id != emission_point_id:
                continue
            if pollutant and deviation.pollutant != pollutant:
                continue
            if start_time and deviation.start_time < start_time:
                continue
            if end_time and deviation.start_time > end_time:
                continue
            if open_only and deviation.end_time is not None:
                continue
            results.append(deviation)
        return results

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def generate_excess_emissions_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Generate excess emissions report for regulatory submission.

        Args:
            start_time: Report period start
            end_time: Report period end

        Returns:
            Dict containing report data
        """
        report_id = str(uuid.uuid4())
        deviations = self.get_deviations(start_time=start_time, end_time=end_time)

        # Group deviations by pollutant and emission point
        summary: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

        for deviation in deviations:
            dev_data = {
                "deviation_id": deviation.deviation_id,
                "start_time": deviation.start_time.isoformat(),
                "end_time": deviation.end_time.isoformat() if deviation.end_time else "ongoing",
                "duration_hours": deviation.duration_hours(),
                "measured_value": deviation.measured_value,
                "limit_value": deviation.limit_value,
                "limit_units": deviation.limit_units,
                "percent_above_limit": deviation.percent_above_limit,
                "root_cause": deviation.root_cause,
                "root_cause_category": deviation.root_cause_category,
                "corrective_actions": deviation.corrective_actions
            }
            summary[deviation.emission_point_id][deviation.pollutant].append(dev_data)

        report = {
            "report_id": report_id,
            "report_type": "Excess Emissions Report",
            "facility_name": self.facility_name,
            "facility_id": self.facility_id,
            "permit_number": self.permit_number,
            "reporting_period_start": start_time.isoformat(),
            "reporting_period_end": end_time.isoformat(),
            "total_deviations": len(deviations),
            "deviations_by_emission_point": dict(summary),
            "generated_time": datetime.now(timezone.utc).isoformat()
        }
        report["provenance_hash"] = self._compute_hash(report)

        logger.info(
            f"Generated excess emissions report {report_id}: "
            f"{len(deviations)} deviations in period"
        )
        return report

    def generate_deviation_report(
        self,
        start_time: datetime,
        end_time: datetime,
        report_type: str = "semi-annual"
    ) -> DeviationReportOutput:
        """
        Generate semi-annual or annual deviation report.

        Args:
            start_time: Report period start
            end_time: Report period end
            report_type: Type of report (semi-annual, annual)

        Returns:
            DeviationReportOutput containing formatted report
        """
        report_id = str(uuid.uuid4())
        deviations = self.get_deviations(start_time=start_time, end_time=end_time)

        deviation_details = []
        for deviation in deviations:
            deviation_details.append({
                "deviation_id": deviation.deviation_id,
                "emission_point_id": deviation.emission_point_id,
                "pollutant": deviation.pollutant,
                "deviation_type": deviation.deviation_type,
                "start_time": deviation.start_time.isoformat(),
                "end_time": deviation.end_time.isoformat() if deviation.end_time else None,
                "duration_hours": deviation.duration_hours(),
                "measured_value": deviation.measured_value,
                "limit_value": deviation.limit_value,
                "limit_units": deviation.limit_units,
                "percent_above_limit": deviation.percent_above_limit,
                "root_cause": deviation.root_cause,
                "root_cause_category": deviation.root_cause_category,
                "corrective_actions": deviation.corrective_actions,
                "preventive_actions": deviation.preventive_actions,
                "reported_to_agency": deviation.reported_to_agency
            })

        certification = (
            f"I certify that, based on information and belief formed after "
            f"reasonable inquiry, the statements and information in this deviation "
            f"report are true, accurate, and complete for {self.facility_name} "
            f"during the reporting period from {start_time.date()} to {end_time.date()}."
        )

        report = DeviationReportOutput(
            report_id=report_id,
            report_type=f"{report_type} Deviation Report",
            reporting_period_start=start_time,
            reporting_period_end=end_time,
            facility_name=self.facility_name,
            permit_number=self.permit_number,
            total_deviations=len(deviations),
            deviations=deviation_details,
            certification=certification,
            generated_time=datetime.now(timezone.utc),
            provenance_hash=self._compute_hash({
                "report_id": report_id,
                "deviations": len(deviations),
                "period": f"{start_time.isoformat()}-{end_time.isoformat()}"
            })
        )

        logger.info(
            f"Generated {report_type} deviation report {report_id}: "
            f"{len(deviations)} deviations"
        )
        return report

    def generate_annual_compliance_certification(
        self,
        year: int
    ) -> Dict[str, Any]:
        """
        Generate annual compliance certification for Title V permit.

        Args:
            year: Calendar year for certification

        Returns:
            Dict containing certification data
        """
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        cert_id = str(uuid.uuid4())
        deviations = self.get_deviations(start_time=year_start, end_time=year_end)

        # Summarize compliance by condition
        condition_compliance: Dict[str, Dict] = {}
        for limit in self._limits.values():
            if not limit.is_active(year_end):
                continue

            limit_deviations = [d for d in deviations if d.limit_id == limit.limit_id]

            status = "compliant" if not limit_deviations else "deviation"
            condition_compliance[limit.limit_id] = {
                "condition_number": limit.condition_number,
                "pollutant": limit.pollutant,
                "emission_point_id": limit.emission_point_id,
                "limit_value": limit.limit_value,
                "limit_units": limit.limit_units,
                "averaging_period": limit.averaging_period,
                "regulatory_basis": limit.regulatory_basis,
                "compliance_status": status,
                "deviation_count": len(limit_deviations),
                "deviation_ids": [d.deviation_id for d in limit_deviations]
            }

        overall_compliance = all(
            c["compliance_status"] == "compliant"
            for c in condition_compliance.values()
        )

        certification = {
            "certification_id": cert_id,
            "certification_type": "Annual Compliance Certification",
            "facility_name": self.facility_name,
            "facility_id": self.facility_id,
            "permit_number": self.permit_number,
            "certification_year": year,
            "period_start": year_start.isoformat(),
            "period_end": year_end.isoformat(),
            "overall_compliance": "compliant" if overall_compliance else "deviation",
            "total_conditions_evaluated": len(condition_compliance),
            "total_deviations": len(deviations),
            "condition_compliance": condition_compliance,
            "certification_statement": (
                f"I certify, based on information and belief formed after "
                f"reasonable inquiry, that {self.facility_name} was in "
                f"{'continuous compliance' if overall_compliance else 'compliance except for the identified deviations'} "
                f"with all applicable requirements of the Title V permit "
                f"during calendar year {year}."
            ),
            "generated_time": datetime.now(timezone.utc).isoformat()
        }
        certification["provenance_hash"] = self._compute_hash(certification)

        logger.info(
            f"Generated annual compliance certification {cert_id} for {year}: "
            f"{'COMPLIANT' if overall_compliance else 'DEVIATIONS NOTED'}"
        )
        return certification

    def generate_emission_inventory(
        self,
        year: int
    ) -> Dict[str, Any]:
        """
        Generate annual emission inventory submission.

        Args:
            year: Calendar year for inventory

        Returns:
            Dict containing inventory data
        """
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        inventory_id = str(uuid.uuid4())

        # Calculate annual emissions by pollutant and emission point
        emission_summaries: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))

        # Group measurements by emission point and pollutant
        measurements_grouped: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        for m in self._measurements:
            if year_start <= m.measurement_time <= year_end:
                measurements_grouped[m.emission_point_id][m.pollutant].append(m)

        for ep_id, pollutant_data in measurements_grouped.items():
            for pollutant, measurements in pollutant_data.items():
                annual_tons, details = self._averaging_calculator.calculate_annual_total(
                    measurements, year, "lb/hr"
                )

                emission_summaries[ep_id][pollutant] = {
                    "annual_emissions_tons": float(annual_tons),
                    "hours_of_operation": details.get("hours_counted", 0),
                    "average_emission_rate_lb_hr": details.get("average_lb_hr", 0),
                    "data_coverage_percent": details.get("data_coverage_percent", 0),
                    "calculation_method": "CEMS continuous monitoring",
                    "provenance_hash": details.get("provenance_hash", "")
                }

        inventory = {
            "inventory_id": inventory_id,
            "inventory_type": "Annual Emission Inventory",
            "facility_name": self.facility_name,
            "facility_id": self.facility_id,
            "permit_number": self.permit_number,
            "inventory_year": year,
            "emissions_by_emission_point": dict(emission_summaries),
            "generated_time": datetime.now(timezone.utc).isoformat()
        }
        inventory["provenance_hash"] = self._compute_hash(inventory)

        logger.info(f"Generated emission inventory {inventory_id} for {year}")
        return inventory

    # -------------------------------------------------------------------------
    # Summary and Statistics
    # -------------------------------------------------------------------------

    def get_compliance_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get summary of compliance status across all emission points.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Dict containing compliance summary statistics
        """
        # Filter compliance history
        history = self._compliance_history
        if start_time:
            history = [h for h in history if h.check_time >= start_time]
        if end_time:
            history = [h for h in history if h.check_time <= end_time]

        # Count by status
        status_counts = defaultdict(int)
        for check in history:
            status_counts[check.compliance_status] += 1

        # Count deviations
        deviations = self.get_deviations(start_time=start_time, end_time=end_time)
        open_deviations = [d for d in deviations if d.end_time is None]

        summary = {
            "period_start": start_time.isoformat() if start_time else "all_time",
            "period_end": end_time.isoformat() if end_time else "current",
            "total_compliance_checks": len(history),
            "status_breakdown": dict(status_counts),
            "compliance_rate_percent": round(
                status_counts.get(ComplianceStatus.COMPLIANT.value, 0) / max(1, len(history)) * 100, 1
            ),
            "total_deviations": len(deviations),
            "open_deviations": len(open_deviations),
            "active_permit_limits": len([l for l in self._limits.values() if l.is_active()]),
            "active_exemptions": len(self.get_active_exemptions())
        }
        summary["provenance_hash"] = self._compute_hash(summary)

        return summary
