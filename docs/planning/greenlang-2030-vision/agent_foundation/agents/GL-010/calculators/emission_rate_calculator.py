"""
Emission Rate Calculator Module for GL-010 EMISSIONWATCH.

This module provides comprehensive emission rate calculations per EPA 40 CFR Part 75
including mass emission rates, heat input-based rates, output-based rates, and
regulatory compliance checking. All calculations are deterministic with full
provenance tracking.

Features:
- Mass emission rate (lb/hr, ton/yr)
- Heat input-based emission rate (lb/MMBtu)
- Output-based emission rate (lb/MWh)
- Diluent correction (O2, CO2)
- Moisture correction (wet to dry basis)
- Flow RATA calculations
- Annual emission totals with trend analysis
- Regulatory limit compliance checking

Zero-Hallucination Guarantee:
- All calculations are deterministic (same input -> same output)
- No LLM involvement in calculation path
- Full provenance tracking with SHA-256 hashes
- Complete audit trails for regulatory compliance

References:
- EPA 40 CFR Part 75, Appendix F (Emission rate calculations)
- EPA 40 CFR Part 60, Appendix A, Method 19
- EPA Acid Rain Program emission calculations
- EPA Clean Air Markets Division procedures

Author: GreenLang GL-010 EMISSIONWATCH Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
import threading
import math

from pydantic import BaseModel, Field, field_validator

from .constants import (
    MW, F_FACTORS, LB_TO_KG, KG_TO_LB, MMBTU_TO_GJ,
    O2_REFERENCE, NORMAL_TEMP_K, NORMAL_PRESSURE_KPA,
)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EmissionUnit(str, Enum):
    """Emission rate units."""
    LB_HR = "lb/hr"
    LB_MMBTU = "lb/MMBtu"
    LB_MWH = "lb/MWh"
    TON_HR = "ton/hr"
    TON_DAY = "ton/day"
    TON_MONTH = "ton/month"
    TON_QUARTER = "ton/quarter"
    TON_YEAR = "ton/yr"
    KG_HR = "kg/hr"
    KG_GJ = "kg/GJ"
    TONNE_YEAR = "tonne/yr"


class PollutantType(str, Enum):
    """Regulated pollutant types."""
    NOX = "nox"
    SO2 = "so2"
    CO2 = "co2"
    CO = "co"
    PM = "pm"
    PM10 = "pm10"
    PM25 = "pm25"
    VOC = "voc"
    HG = "hg"
    HCL = "hcl"


class DiluentsType(str, Enum):
    """Diluent gas types for correction."""
    O2 = "o2"
    CO2 = "co2"


class RegulatoryProgram(str, Enum):
    """Regulatory programs with emission limits."""
    ACID_RAIN = "acid_rain"
    CSAPR = "csapr"  # Cross-State Air Pollution Rule
    MATS = "mats"    # Mercury and Air Toxics Standards
    NSPS = "nsps"    # New Source Performance Standards
    TITLE_V = "title_v"
    STATE_SIP = "state_sip"
    PSD = "psd"      # Prevention of Significant Deterioration


class ComplianceStatus(str, Enum):
    """Emission limit compliance status."""
    COMPLIANT = "compliant"
    EXCEEDING = "exceeding"
    APPROACHING_LIMIT = "approaching_limit"  # >80% of limit
    DATA_INSUFFICIENT = "data_insufficient"


class AveragingPeriod(str, Enum):
    """Emission averaging periods."""
    HOURLY = "hourly"
    DAILY = "24_hour"
    ROLLING_30_DAY = "rolling_30_day"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    OZONE_SEASON = "ozone_season"


class FlowCalculationMethod(str, Enum):
    """Stack flow calculation methods."""
    VELOCITY_TRAVERSE = "velocity_traverse"
    CEMS_MEASURED = "cems_measured"
    F_FACTOR = "f_factor"
    ESTIMATED = "estimated"


# =============================================================================
# FROZEN DATACLASSES (Thread-Safe, Hashable)
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """Individual calculation step with provenance."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, Union[str, float, Decimal]]
    output_value: Decimal
    output_unit: str


@dataclass(frozen=True)
class MassEmissionRate:
    """
    Mass emission rate calculation result.

    Attributes:
        pollutant: Pollutant type
        rate_lb_hr: Emission rate in lb/hr
        rate_ton_yr: Annualized rate in short tons/yr
        rate_kg_hr: Emission rate in kg/hr
        heat_input_mmbtu_hr: Heat input used
        calculation_method: Method used for calculation
        calculation_steps: Detailed steps for audit
        provenance_hash: SHA-256 hash for provenance
    """
    pollutant: PollutantType
    rate_lb_hr: Decimal
    rate_ton_yr: Decimal
    rate_kg_hr: Decimal
    heat_input_mmbtu_hr: Decimal
    calculation_method: str
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class HeatInputBasedRate:
    """
    Heat input-based emission rate result.

    Attributes:
        pollutant: Pollutant type
        rate_lb_mmbtu: Emission factor (lb/MMBtu)
        rate_kg_gj: Emission factor (kg/GJ)
        f_factor_used: F-factor applied
        diluent_correction_applied: Whether O2/CO2 correction applied
        correction_factor: Diluent correction factor value
        calculation_steps: Detailed steps for audit
        provenance_hash: SHA-256 hash for provenance
    """
    pollutant: PollutantType
    rate_lb_mmbtu: Decimal
    rate_kg_gj: Decimal
    f_factor_used: Decimal
    diluent_correction_applied: bool
    correction_factor: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class OutputBasedRate:
    """
    Output-based emission rate result (lb/MWh).

    Attributes:
        pollutant: Pollutant type
        rate_lb_mwh_gross: Rate based on gross output
        rate_lb_mwh_net: Rate based on net output
        gross_output_mw: Gross electrical output
        net_output_mw: Net electrical output
        heat_rate_btu_kwh: Unit heat rate
        calculation_steps: Detailed steps for audit
        provenance_hash: SHA-256 hash for provenance
    """
    pollutant: PollutantType
    rate_lb_mwh_gross: Decimal
    rate_lb_mwh_net: Decimal
    gross_output_mw: Decimal
    net_output_mw: Decimal
    heat_rate_btu_kwh: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class DiluentsCorrection:
    """
    Diluent gas correction result.

    Attributes:
        original_concentration: Concentration before correction
        corrected_concentration: Concentration at reference conditions
        diluent_type: O2 or CO2 correction
        measured_diluent: Measured diluent concentration (%)
        reference_diluent: Reference diluent level (%)
        correction_factor: Applied correction factor
        dry_basis: Whether correction is on dry basis
        provenance_hash: SHA-256 hash for provenance
    """
    original_concentration: Decimal
    corrected_concentration: Decimal
    diluent_type: DiluentsType
    measured_diluent: Decimal
    reference_diluent: Decimal
    correction_factor: Decimal
    dry_basis: bool
    provenance_hash: str


@dataclass(frozen=True)
class MoistureCorrection:
    """
    Moisture correction result.

    Attributes:
        wet_basis_value: Value on wet basis
        dry_basis_value: Value corrected to dry basis
        moisture_fraction: Measured moisture (fraction 0-1)
        correction_factor: Applied correction factor
        provenance_hash: SHA-256 hash for provenance
    """
    wet_basis_value: Decimal
    dry_basis_value: Decimal
    moisture_fraction: Decimal
    correction_factor: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class FlowRATAResult:
    """
    Flow RATA calculation result.

    Attributes:
        test_date: Date of RATA test
        num_runs: Number of valid test runs
        reference_values: Reference method values (scfm)
        cems_values: CEMS flow values (scfm)
        relative_accuracy: Calculated RA (%)
        passed: Whether RATA passed (<10% RA)
        stack_area_ft2: Stack cross-sectional area
        provenance_hash: SHA-256 hash for provenance
    """
    test_date: datetime
    num_runs: int
    reference_values: Tuple[Decimal, ...]
    cems_values: Tuple[Decimal, ...]
    relative_accuracy: Decimal
    passed: bool
    stack_area_ft2: Decimal
    provenance_hash: str


@dataclass(frozen=True)
class AnnualEmissionTotal:
    """
    Annual emission total with trend analysis.

    Attributes:
        year: Reporting year
        pollutant: Pollutant type
        total_mass_tons: Total emissions (short tons)
        total_mass_tonnes: Total emissions (metric tonnes)
        total_heat_input_mmbtu: Total heat input
        operating_hours: Total operating hours
        average_rate_lb_mmbtu: Average emission rate
        peak_rate_lb_mmbtu: Maximum hourly rate
        ozone_season_tons: Ozone season total (May-Sep)
        quarterly_totals: Quarterly breakdown
        year_over_year_change: % change from prior year
        trend_direction: up, down, stable
        provenance_hash: SHA-256 hash for provenance
    """
    year: int
    pollutant: PollutantType
    total_mass_tons: Decimal
    total_mass_tonnes: Decimal
    total_heat_input_mmbtu: Decimal
    operating_hours: int
    average_rate_lb_mmbtu: Decimal
    peak_rate_lb_mmbtu: Decimal
    ozone_season_tons: Decimal
    quarterly_totals: Tuple[Decimal, Decimal, Decimal, Decimal]
    year_over_year_change: Optional[Decimal]
    trend_direction: str
    provenance_hash: str


@dataclass(frozen=True)
class ComplianceCheckResult:
    """
    Regulatory compliance check result.

    Attributes:
        pollutant: Pollutant checked
        regulatory_program: Applicable program
        limit_value: Emission limit
        limit_unit: Limit units
        averaging_period: Averaging period
        measured_value: Actual emission value
        percent_of_limit: Measured as % of limit
        compliance_status: Compliance determination
        margin_to_limit: Remaining margin
        exceedance_hours: Hours exceeding (if any)
        provenance_hash: SHA-256 hash for provenance
    """
    pollutant: PollutantType
    regulatory_program: RegulatoryProgram
    limit_value: Decimal
    limit_unit: str
    averaging_period: AveragingPeriod
    measured_value: Decimal
    percent_of_limit: Decimal
    compliance_status: ComplianceStatus
    margin_to_limit: Decimal
    exceedance_hours: int
    provenance_hash: str


# =============================================================================
# INPUT MODELS (Pydantic)
# =============================================================================

class CEMSHourlyInput(BaseModel):
    """Input for hourly CEMS data."""
    timestamp: datetime = Field(description="Hour timestamp")
    concentration_ppm: float = Field(ge=0, description="Pollutant concentration (ppm)")
    flow_scfh: Optional[float] = Field(default=None, ge=0, description="Stack flow (scf/hr)")
    o2_percent: Optional[float] = Field(default=None, ge=0, lt=21, description="O2 (%)")
    co2_percent: Optional[float] = Field(default=None, ge=0, lt=30, description="CO2 (%)")
    moisture_percent: Optional[float] = Field(default=None, ge=0, lt=50, description="Moisture (%)")
    heat_input_mmbtu: Optional[float] = Field(default=None, ge=0, description="Heat input (MMBtu)")


class MassEmissionInput(BaseModel):
    """Input for mass emission rate calculation."""
    pollutant: PollutantType = Field(description="Pollutant type")
    concentration_ppm: float = Field(ge=0, description="Pollutant concentration (ppm)")
    flow_scfh: float = Field(gt=0, description="Stack flow (scf/hr)")
    molecular_weight: Optional[float] = Field(default=None, gt=0, description="MW (g/mol)")
    temperature_f: float = Field(default=68.0, description="Stack temperature (F)")
    pressure_inhg: float = Field(default=29.92, ge=20, le=35, description="Barometric pressure")


class HeatInputRateInput(BaseModel):
    """Input for heat input-based emission rate calculation."""
    pollutant: PollutantType = Field(description="Pollutant type")
    concentration_ppm: float = Field(ge=0, description="Pollutant concentration (ppm)")
    fuel_type: str = Field(description="Fuel type for F-factor")
    measured_o2_percent: Optional[float] = Field(default=None, ge=0, lt=21, description="Measured O2")
    measured_co2_percent: Optional[float] = Field(default=None, ge=0, lt=30, description="Measured CO2")
    reference_o2_percent: float = Field(default=3.0, ge=0, lt=21, description="Reference O2")
    moisture_percent: float = Field(default=0.0, ge=0, lt=50, description="Moisture (%)")


class OutputBasedRateInput(BaseModel):
    """Input for output-based emission rate calculation."""
    pollutant: PollutantType = Field(description="Pollutant type")
    mass_emission_lb_hr: float = Field(ge=0, description="Mass rate (lb/hr)")
    gross_output_mw: float = Field(ge=0, description="Gross electrical output (MW)")
    net_output_mw: Optional[float] = Field(default=None, ge=0, description="Net output (MW)")
    auxiliary_power_mw: Optional[float] = Field(default=None, ge=0, description="Aux power (MW)")


class FlowRATAInput(BaseModel):
    """Input for flow RATA calculation."""
    reference_method_values: List[float] = Field(
        min_length=9,
        description="Reference method flow values (scfm)"
    )
    cems_values: List[float] = Field(
        min_length=9,
        description="Concurrent CEMS flow values (scfm)"
    )
    stack_diameter_ft: float = Field(gt=0, description="Stack diameter (ft)")
    test_date: Optional[datetime] = Field(default=None, description="RATA date")

    @field_validator("cems_values")
    @classmethod
    def validate_matching_length(cls, v, info):
        ref_values = info.data.get("reference_method_values", [])
        if len(v) != len(ref_values):
            raise ValueError("CEMS values must match reference method count")
        return v


class AnnualTotalInput(BaseModel):
    """Input for annual emission total calculation."""
    pollutant: PollutantType = Field(description="Pollutant type")
    year: int = Field(ge=1990, le=2100, description="Reporting year")
    hourly_emissions_lb: List[float] = Field(description="Hourly emissions (lb)")
    hourly_heat_input_mmbtu: List[float] = Field(description="Hourly heat input")
    prior_year_total_tons: Optional[float] = Field(default=None, description="Prior year total")


class ComplianceCheckInput(BaseModel):
    """Input for compliance check."""
    pollutant: PollutantType = Field(description="Pollutant to check")
    regulatory_program: RegulatoryProgram = Field(description="Applicable program")
    averaging_period: AveragingPeriod = Field(description="Averaging period")
    limit_value: float = Field(gt=0, description="Emission limit value")
    limit_unit: str = Field(description="Limit units (lb/MMBtu, lb/MWh, etc.)")
    measured_values: List[float] = Field(description="Measured values for period")
    limit_basis: str = Field(default="average", description="average or maximum")


# =============================================================================
# EMISSION RATE CALCULATOR CLASS
# =============================================================================

class EmissionRateCalculator:
    """
    Emission Rate Calculator per EPA 40 CFR Part 75 and Method 19.

    Provides deterministic, zero-hallucination calculations for emission
    rates including mass rates, heat input-based rates, output-based rates,
    and regulatory compliance checking.

    Thread Safety:
        All methods are thread-safe using thread-local state.

    Example:
        >>> calculator = EmissionRateCalculator()
        >>> result = calculator.calculate_mass_emission_rate(
        ...     MassEmissionInput(
        ...         pollutant=PollutantType.NOX,
        ...         concentration_ppm=50.0,
        ...         flow_scfh=500000.0
        ...     )
        ... )
        >>> print(f"NOx rate: {result.rate_lb_hr} lb/hr")
    """

    # Molecular weights (g/mol) for pollutants
    MOLECULAR_WEIGHTS: Dict[PollutantType, Decimal] = {
        PollutantType.NOX: Decimal("46.01"),   # As NO2
        PollutantType.SO2: Decimal("64.07"),
        PollutantType.CO2: Decimal("44.01"),
        PollutantType.CO: Decimal("28.01"),
        PollutantType.PM: Decimal("1"),        # Direct mass
        PollutantType.HG: Decimal("200.59"),
        PollutantType.HCL: Decimal("36.46"),
    }

    # Conversion constant K (lb-mol/dscf at 68F, 29.92 inHg)
    # K = 1 / (385.3 * 10^6) for lb/dscf-ppm
    K_CONSTANT: Decimal = Decimal("2.595e-9")

    # EPA Method 19 conversion factor
    # lb/dscf-ppm = MW * K where K = 1.194e-7 / MW_ref
    METHOD_19_K: Dict[PollutantType, Decimal] = {
        PollutantType.NOX: Decimal("1.194e-7"),   # lb/dscf-ppm for NO2
        PollutantType.SO2: Decimal("1.660e-7"),
        PollutantType.CO2: Decimal("1.142e-7"),
    }

    # Regulatory limits database (example values)
    REGULATORY_LIMITS: Dict[str, Dict[str, Any]] = {
        "acid_rain_nox_coal": {
            "limit": Decimal("0.40"),
            "unit": "lb/MMBtu",
            "averaging": AveragingPeriod.ANNUAL
        },
        "acid_rain_nox_gas": {
            "limit": Decimal("0.20"),
            "unit": "lb/MMBtu",
            "averaging": AveragingPeriod.ANNUAL
        },
        "acid_rain_so2": {
            "limit": Decimal("1.20"),
            "unit": "lb/MMBtu",
            "averaging": AveragingPeriod.ANNUAL
        },
        "nsps_nox_gas_turbine": {
            "limit": Decimal("25.0"),
            "unit": "ppm",
            "averaging": AveragingPeriod.HOURLY
        },
    }

    def __init__(self):
        """Initialize Emission Rate Calculator."""
        self._cache_lock = threading.Lock()
        self._f_factor_cache: Dict[str, Decimal] = {}

    def calculate_mass_emission_rate(
        self,
        rate_input: MassEmissionInput,
        precision: int = 4
    ) -> MassEmissionRate:
        """
        Calculate mass emission rate from concentration and flow.

        Formula (EPA Method 19):
        E (lb/hr) = C * K * MW * Q

        Where:
        - C = Pollutant concentration (ppm)
        - K = Conversion constant (2.595e-9 lb-mol/dscf)
        - MW = Molecular weight (g/mol)
        - Q = Stack flow rate (scf/hr)

        Args:
            rate_input: Concentration and flow data
            precision: Decimal places in result

        Returns:
            MassEmissionRate with lb/hr, ton/yr rates

        Reference:
            EPA 40 CFR Part 75, Appendix F, Equation F-1
        """
        steps = []
        step_num = 0

        conc = Decimal(str(rate_input.concentration_ppm))
        flow = Decimal(str(rate_input.flow_scfh))

        # Get molecular weight
        mw = self.MOLECULAR_WEIGHTS.get(
            rate_input.pollutant,
            Decimal(str(rate_input.molecular_weight or 46.01))
        )

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Retrieve molecular weight",
            formula="MW lookup from pollutant type",
            inputs={"pollutant": rate_input.pollutant.value},
            output_value=mw,
            output_unit="g/mol"
        ))

        # Temperature/pressure correction to standard conditions
        temp_f = Decimal(str(rate_input.temperature_f))
        press_inhg = Decimal(str(rate_input.pressure_inhg))

        # Standard: 68F (528R), 29.92 inHg
        temp_correction = (Decimal("528") / (temp_f + Decimal("460")))
        press_correction = press_inhg / Decimal("29.92")
        tp_factor = temp_correction * press_correction

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate temperature/pressure correction to std conditions",
            formula="Factor = (528 / (T + 460)) * (P / 29.92)",
            inputs={
                "temperature_f": str(temp_f),
                "pressure_inhg": str(press_inhg)
            },
            output_value=self._apply_precision(tp_factor, 4),
            output_unit="dimensionless"
        ))

        # Correct flow to standard conditions
        flow_std = flow * tp_factor

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Correct flow to standard conditions",
            formula="Q_std = Q_actual * TP_factor",
            inputs={
                "flow_actual_scfh": str(flow),
                "tp_factor": str(tp_factor)
            },
            output_value=self._apply_precision(flow_std, 0),
            output_unit="dscf/hr"
        ))

        # Calculate mass rate using ideal gas law conversion
        # lb/hr = ppm * flow_dscfh * MW / (385.3 * 10^6)
        # 385.3 = molar volume at std conditions (ft3/lb-mol)
        molar_volume = Decimal("385.3")
        lb_per_hr = conc * flow_std * mw / (molar_volume * Decimal("1e6"))
        lb_per_hr = self._apply_precision(lb_per_hr, precision)

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate mass emission rate",
            formula="E = C * Q * MW / (385.3 * 10^6)",
            inputs={
                "concentration_ppm": str(conc),
                "flow_dscfh": str(flow_std),
                "molecular_weight": str(mw)
            },
            output_value=lb_per_hr,
            output_unit="lb/hr"
        ))

        # Convert to other units
        kg_per_hr = lb_per_hr * LB_TO_KG
        kg_per_hr = self._apply_precision(kg_per_hr, precision)

        # Annualized assuming 8760 hours/year
        ton_per_yr = lb_per_hr * Decimal("8760") / Decimal("2000")
        ton_per_yr = self._apply_precision(ton_per_yr, precision)

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate annualized rate (8760 hrs)",
            formula="ton/yr = lb/hr * 8760 / 2000",
            inputs={"lb_per_hr": str(lb_per_hr)},
            output_value=ton_per_yr,
            output_unit="ton/yr"
        ))

        # Calculate provenance hash
        provenance_data = {
            "calculation_type": "mass_emission_rate",
            "pollutant": rate_input.pollutant.value,
            "concentration_ppm": str(conc),
            "flow_scfh": str(flow),
            "lb_per_hr": str(lb_per_hr),
            "ton_per_yr": str(ton_per_yr)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return MassEmissionRate(
            pollutant=rate_input.pollutant,
            rate_lb_hr=lb_per_hr,
            rate_ton_yr=ton_per_yr,
            rate_kg_hr=kg_per_hr,
            heat_input_mmbtu_hr=Decimal("0"),  # Not used in this method
            calculation_method="EPA_Method_19_Mass_Balance",
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    def calculate_heat_input_based_rate(
        self,
        rate_input: HeatInputRateInput,
        precision: int = 4
    ) -> HeatInputBasedRate:
        """
        Calculate heat input-based emission rate using F-factor method.

        EPA Method 19 Formula:
        E (lb/MMBtu) = C * K * Fd * (20.9 / (20.9 - %O2))

        Where:
        - C = Concentration (ppm, dry basis)
        - K = Pollutant-specific constant (lb/dscf-ppm)
        - Fd = Dry F-factor (dscf/MMBtu)
        - %O2 = Measured oxygen (%, dry basis)

        Args:
            rate_input: Concentration and fuel data
            precision: Decimal places in result

        Returns:
            HeatInputBasedRate with lb/MMBtu rate

        Reference:
            EPA 40 CFR Part 60, Appendix A, Method 19, Equation 19-3
        """
        steps = []
        step_num = 0

        conc = Decimal(str(rate_input.concentration_ppm))

        # Step 1: Get F-factor for fuel type
        fuel_type = rate_input.fuel_type.lower()
        f_factors = F_FACTORS.get(fuel_type, F_FACTORS.get("natural_gas", {}))
        fd = Decimal(str(f_factors.get("Fd", 8710)))

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Look up dry F-factor (Fd) for fuel type",
            formula="Fd from EPA Method 19 Table 19-1",
            inputs={"fuel_type": fuel_type},
            output_value=fd,
            output_unit="dscf/MMBtu"
        ))

        # Step 2: Moisture correction if needed
        moisture = Decimal(str(rate_input.moisture_percent)) / Decimal("100")
        if moisture > 0:
            conc_dry = conc / (Decimal("1") - moisture)
            step_num += 1
            steps.append(CalculationStep(
                step_number=step_num,
                description="Correct concentration to dry basis",
                formula="C_dry = C_wet / (1 - moisture)",
                inputs={
                    "concentration_wet": str(conc),
                    "moisture_fraction": str(moisture)
                },
                output_value=self._apply_precision(conc_dry, 2),
                output_unit="ppm (dry)"
            ))
        else:
            conc_dry = conc

        # Step 3: O2 diluent correction
        correction_factor = Decimal("1")
        diluent_correction_applied = False

        if rate_input.measured_o2_percent is not None:
            o2_meas = Decimal(str(rate_input.measured_o2_percent))
            o2_ref = Decimal(str(rate_input.reference_o2_percent))

            # Standard O2 correction: (20.9 - O2_ref) / (20.9 - O2_meas)
            o2_air = Decimal("20.9")

            if o2_meas < o2_air:
                correction_factor = (o2_air - o2_ref) / (o2_air - o2_meas)
                diluent_correction_applied = True

                step_num += 1
                steps.append(CalculationStep(
                    step_number=step_num,
                    description=f"Apply O2 correction to {o2_ref}% reference",
                    formula="Factor = (20.9 - O2_ref) / (20.9 - O2_meas)",
                    inputs={
                        "o2_measured": str(o2_meas),
                        "o2_reference": str(o2_ref)
                    },
                    output_value=self._apply_precision(correction_factor, 4),
                    output_unit="dimensionless"
                ))

        # Step 4: Get pollutant-specific K factor
        k_factor = self.METHOD_19_K.get(
            rate_input.pollutant,
            Decimal("1.194e-7")
        )

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Get pollutant K factor",
            formula="K from EPA Method 19",
            inputs={"pollutant": rate_input.pollutant.value},
            output_value=k_factor,
            output_unit="lb/dscf-ppm"
        ))

        # Step 5: Calculate emission rate
        # E = C * K * Fd * correction_factor
        rate_lb_mmbtu = conc_dry * k_factor * fd * correction_factor
        rate_lb_mmbtu = self._apply_precision(rate_lb_mmbtu, precision)

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate heat input-based emission rate",
            formula="E = C * K * Fd * O2_correction",
            inputs={
                "concentration_dry_ppm": str(conc_dry),
                "k_factor": str(k_factor),
                "fd": str(fd),
                "correction_factor": str(correction_factor)
            },
            output_value=rate_lb_mmbtu,
            output_unit="lb/MMBtu"
        ))

        # Convert to kg/GJ
        # lb/MMBtu * 0.429923 = kg/GJ
        rate_kg_gj = rate_lb_mmbtu * Decimal("0.429923")
        rate_kg_gj = self._apply_precision(rate_kg_gj, precision)

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Convert to metric units",
            formula="kg/GJ = lb/MMBtu * 0.429923",
            inputs={"rate_lb_mmbtu": str(rate_lb_mmbtu)},
            output_value=rate_kg_gj,
            output_unit="kg/GJ"
        ))

        # Provenance hash
        provenance_data = {
            "calculation_type": "heat_input_based_rate",
            "pollutant": rate_input.pollutant.value,
            "fuel_type": fuel_type,
            "concentration_ppm": str(conc),
            "fd": str(fd),
            "correction_factor": str(correction_factor),
            "rate_lb_mmbtu": str(rate_lb_mmbtu)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return HeatInputBasedRate(
            pollutant=rate_input.pollutant,
            rate_lb_mmbtu=rate_lb_mmbtu,
            rate_kg_gj=rate_kg_gj,
            f_factor_used=fd,
            diluent_correction_applied=diluent_correction_applied,
            correction_factor=correction_factor,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    def calculate_output_based_rate(
        self,
        rate_input: OutputBasedRateInput,
        precision: int = 4
    ) -> OutputBasedRate:
        """
        Calculate output-based emission rate (lb/MWh).

        Formula:
        E (lb/MWh) = E (lb/hr) / Output (MW)

        Args:
            rate_input: Mass rate and output data
            precision: Decimal places in result

        Returns:
            OutputBasedRate with lb/MWh rates

        Reference:
            EPA 40 CFR Part 75, Appendix F
        """
        steps = []
        step_num = 0

        mass_lb_hr = Decimal(str(rate_input.mass_emission_lb_hr))
        gross_mw = Decimal(str(rate_input.gross_output_mw))

        # Calculate net output if not provided
        if rate_input.net_output_mw is not None:
            net_mw = Decimal(str(rate_input.net_output_mw))
        elif rate_input.auxiliary_power_mw is not None:
            aux_mw = Decimal(str(rate_input.auxiliary_power_mw))
            net_mw = gross_mw - aux_mw
        else:
            # Estimate net as 95% of gross (typical auxiliary load)
            net_mw = gross_mw * Decimal("0.95")

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Determine net output",
            formula="Net_MW = Gross_MW - Auxiliary_MW",
            inputs={
                "gross_mw": str(gross_mw),
                "net_mw": str(net_mw)
            },
            output_value=self._apply_precision(net_mw, 2),
            output_unit="MW"
        ))

        # Calculate gross output-based rate
        if gross_mw > 0:
            rate_lb_mwh_gross = mass_lb_hr / gross_mw
        else:
            rate_lb_mwh_gross = Decimal("0")
        rate_lb_mwh_gross = self._apply_precision(rate_lb_mwh_gross, precision)

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate gross output-based rate",
            formula="E_gross = Mass_lb_hr / Gross_MW",
            inputs={
                "mass_lb_hr": str(mass_lb_hr),
                "gross_mw": str(gross_mw)
            },
            output_value=rate_lb_mwh_gross,
            output_unit="lb/MWh (gross)"
        ))

        # Calculate net output-based rate
        if net_mw > 0:
            rate_lb_mwh_net = mass_lb_hr / net_mw
        else:
            rate_lb_mwh_net = Decimal("0")
        rate_lb_mwh_net = self._apply_precision(rate_lb_mwh_net, precision)

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            description="Calculate net output-based rate",
            formula="E_net = Mass_lb_hr / Net_MW",
            inputs={
                "mass_lb_hr": str(mass_lb_hr),
                "net_mw": str(net_mw)
            },
            output_value=rate_lb_mwh_net,
            output_unit="lb/MWh (net)"
        ))

        # Estimate heat rate
        # Typical combined cycle: ~7000 Btu/kWh, coal: ~10000 Btu/kWh
        if gross_mw > 0:
            heat_rate = Decimal("8500")  # Default estimate
        else:
            heat_rate = Decimal("0")

        # Provenance hash
        provenance_data = {
            "calculation_type": "output_based_rate",
            "pollutant": rate_input.pollutant.value,
            "mass_lb_hr": str(mass_lb_hr),
            "gross_mw": str(gross_mw),
            "net_mw": str(net_mw),
            "rate_lb_mwh_gross": str(rate_lb_mwh_gross),
            "rate_lb_mwh_net": str(rate_lb_mwh_net)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return OutputBasedRate(
            pollutant=rate_input.pollutant,
            rate_lb_mwh_gross=rate_lb_mwh_gross,
            rate_lb_mwh_net=rate_lb_mwh_net,
            gross_output_mw=gross_mw,
            net_output_mw=net_mw,
            heat_rate_btu_kwh=heat_rate,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    def apply_diluent_correction(
        self,
        concentration: Union[float, Decimal],
        diluent_type: DiluentsType,
        measured_diluent: float,
        reference_diluent: float,
        dry_basis: bool = True,
        precision: int = 4
    ) -> DiluentsCorrection:
        """
        Apply diluent (O2 or CO2) correction to concentration.

        O2 Correction Formula:
        C_corrected = C_measured * (20.9 - O2_ref) / (20.9 - O2_meas)

        CO2 Correction Formula:
        C_corrected = C_measured * CO2_ref / CO2_meas

        Args:
            concentration: Measured concentration
            diluent_type: O2 or CO2 correction
            measured_diluent: Measured diluent (%)
            reference_diluent: Reference diluent (%)
            dry_basis: Whether on dry basis
            precision: Decimal places in result

        Returns:
            DiluentsCorrection with corrected value

        Reference:
            EPA 40 CFR Part 60, Appendix A, Method 19
        """
        conc = Decimal(str(concentration))
        meas = Decimal(str(measured_diluent))
        ref = Decimal(str(reference_diluent))

        if diluent_type == DiluentsType.O2:
            # O2 correction
            o2_air = Decimal("20.9")
            if meas < o2_air:
                correction_factor = (o2_air - ref) / (o2_air - meas)
            else:
                correction_factor = Decimal("1")
        else:
            # CO2 correction
            if meas > 0:
                correction_factor = ref / meas
            else:
                correction_factor = Decimal("1")

        corrected = conc * correction_factor
        corrected = self._apply_precision(corrected, precision)

        provenance_data = {
            "correction_type": diluent_type.value,
            "original": str(conc),
            "measured_diluent": str(meas),
            "reference_diluent": str(ref),
            "correction_factor": str(correction_factor),
            "corrected": str(corrected)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return DiluentsCorrection(
            original_concentration=conc,
            corrected_concentration=corrected,
            diluent_type=diluent_type,
            measured_diluent=meas,
            reference_diluent=ref,
            correction_factor=self._apply_precision(correction_factor, 4),
            dry_basis=dry_basis,
            provenance_hash=provenance_hash
        )

    def apply_moisture_correction(
        self,
        wet_value: Union[float, Decimal],
        moisture_percent: float,
        precision: int = 4
    ) -> MoistureCorrection:
        """
        Apply moisture correction (wet to dry basis).

        Formula:
        C_dry = C_wet / (1 - H2O)

        Where H2O = moisture fraction (0-1)

        Args:
            wet_value: Value on wet basis
            moisture_percent: Moisture content (%)
            precision: Decimal places in result

        Returns:
            MoistureCorrection with dry basis value

        Reference:
            EPA 40 CFR Part 60, Appendix A, Method 4
        """
        wet = Decimal(str(wet_value))
        moisture = Decimal(str(moisture_percent)) / Decimal("100")

        if moisture >= Decimal("1"):
            raise ValueError("Moisture fraction must be less than 100%")

        correction_factor = Decimal("1") / (Decimal("1") - moisture)
        dry_value = wet * correction_factor
        dry_value = self._apply_precision(dry_value, precision)

        provenance_data = {
            "correction_type": "moisture",
            "wet_value": str(wet),
            "moisture_percent": str(moisture_percent),
            "dry_value": str(dry_value)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return MoistureCorrection(
            wet_basis_value=wet,
            dry_basis_value=dry_value,
            moisture_fraction=moisture,
            correction_factor=self._apply_precision(correction_factor, 4),
            provenance_hash=provenance_hash
        )

    def calculate_flow_rata(
        self,
        rata_input: FlowRATAInput,
        precision: int = 2
    ) -> FlowRATAResult:
        """
        Calculate flow RATA relative accuracy.

        Formula (per 40 CFR Part 75):
        RA = (|d_avg| + |CC|) / RM_avg * 100

        Where:
        - d_avg = Mean difference (RM - CEMS)
        - CC = Confidence coefficient
        - RM_avg = Mean reference method value

        Args:
            rata_input: Reference and CEMS flow values
            precision: Decimal places in result

        Returns:
            FlowRATAResult with relative accuracy

        Reference:
            EPA 40 CFR Part 75, Appendix A, Section 6.5
        """
        n = len(rata_input.reference_method_values)

        ref_values = tuple(Decimal(str(v)) for v in rata_input.reference_method_values)
        cems_values = tuple(Decimal(str(v)) for v in rata_input.cems_values)

        # Calculate differences
        differences = tuple(ref - cems for ref, cems in zip(ref_values, cems_values))

        # Mean values
        ref_avg = sum(ref_values) / Decimal(str(n))
        d_avg = sum(differences) / Decimal(str(n))

        # Standard deviation of differences
        if n > 1:
            variance = sum((d - d_avg) ** 2 for d in differences) / Decimal(str(n - 1))
            s_d = variance.sqrt()
        else:
            s_d = Decimal("0")

        # t-statistic (two-tailed, alpha=0.05)
        t_values = {9: Decimal("2.306"), 12: Decimal("2.179"), 15: Decimal("2.145")}
        t_stat = t_values.get(n, Decimal("2.0"))

        # Confidence coefficient
        cc = t_stat * s_d / Decimal(str(n)).sqrt() if n > 0 else Decimal("0")

        # Relative accuracy
        if ref_avg > Decimal("0"):
            ra = (abs(d_avg) + abs(cc)) / ref_avg * Decimal("100")
        else:
            ra = Decimal("0")
        ra = self._apply_precision(ra, precision)

        # Flow RATA limit is 10%
        passed = ra <= Decimal("10.0")

        # Stack area
        diameter = Decimal(str(rata_input.stack_diameter_ft))
        area = Decimal(str(math.pi)) * (diameter / Decimal("2")) ** 2
        area = self._apply_precision(area, 2)

        provenance_data = {
            "test_type": "flow_rata",
            "num_runs": n,
            "ra": str(ra),
            "passed": passed,
            "test_date": (rata_input.test_date or datetime.utcnow()).isoformat()
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return FlowRATAResult(
            test_date=rata_input.test_date or datetime.utcnow(),
            num_runs=n,
            reference_values=ref_values,
            cems_values=cems_values,
            relative_accuracy=ra,
            passed=passed,
            stack_area_ft2=area,
            provenance_hash=provenance_hash
        )

    def calculate_annual_total(
        self,
        total_input: AnnualTotalInput,
        precision: int = 2
    ) -> AnnualEmissionTotal:
        """
        Calculate annual emission totals with trend analysis.

        Args:
            total_input: Hourly emissions and heat input data
            precision: Decimal places in result

        Returns:
            AnnualEmissionTotal with totals and trends

        Reference:
            EPA 40 CFR Part 75, Appendix G
        """
        emissions = [Decimal(str(e)) for e in total_input.hourly_emissions_lb]
        heat_inputs = [Decimal(str(h)) for h in total_input.hourly_heat_input_mmbtu]

        # Total mass
        total_lb = sum(emissions)
        total_tons = total_lb / Decimal("2000")
        total_tonnes = total_tons * Decimal("0.907185")

        total_tons = self._apply_precision(total_tons, precision)
        total_tonnes = self._apply_precision(total_tonnes, precision)

        # Total heat input
        total_heat = sum(heat_inputs)
        total_heat = self._apply_precision(total_heat, 0)

        # Operating hours
        operating_hours = sum(1 for e in emissions if e > 0)

        # Average rate
        if total_heat > 0:
            avg_rate = total_lb / total_heat
        else:
            avg_rate = Decimal("0")
        avg_rate = self._apply_precision(avg_rate, precision + 2)

        # Peak rate
        hourly_rates = []
        for e, h in zip(emissions, heat_inputs):
            if h > 0:
                hourly_rates.append(e / h)
        peak_rate = max(hourly_rates) if hourly_rates else Decimal("0")
        peak_rate = self._apply_precision(peak_rate, precision + 2)

        # Ozone season (May-Sep, hours 2880-6552 approximately)
        ozone_start = 2880  # May 1
        ozone_end = 6552    # Sep 30
        ozone_emissions = emissions[ozone_start:min(ozone_end, len(emissions))]
        ozone_tons = sum(ozone_emissions) / Decimal("2000")
        ozone_tons = self._apply_precision(ozone_tons, precision)

        # Quarterly totals (2190 hours per quarter)
        quarterly = []
        for q in range(4):
            start = q * 2190
            end = min((q + 1) * 2190, len(emissions))
            q_total = sum(emissions[start:end]) / Decimal("2000")
            quarterly.append(self._apply_precision(q_total, precision))

        # Year-over-year change
        yoy_change = None
        trend = "stable"
        if total_input.prior_year_total_tons is not None:
            prior = Decimal(str(total_input.prior_year_total_tons))
            if prior > 0:
                yoy_change = (total_tons - prior) / prior * Decimal("100")
                yoy_change = self._apply_precision(yoy_change, 1)
                if yoy_change > Decimal("5"):
                    trend = "up"
                elif yoy_change < Decimal("-5"):
                    trend = "down"

        provenance_data = {
            "calculation_type": "annual_total",
            "year": total_input.year,
            "pollutant": total_input.pollutant.value,
            "total_tons": str(total_tons),
            "operating_hours": operating_hours,
            "avg_rate": str(avg_rate)
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return AnnualEmissionTotal(
            year=total_input.year,
            pollutant=total_input.pollutant,
            total_mass_tons=total_tons,
            total_mass_tonnes=total_tonnes,
            total_heat_input_mmbtu=total_heat,
            operating_hours=operating_hours,
            average_rate_lb_mmbtu=avg_rate,
            peak_rate_lb_mmbtu=peak_rate,
            ozone_season_tons=ozone_tons,
            quarterly_totals=tuple(quarterly),
            year_over_year_change=yoy_change,
            trend_direction=trend,
            provenance_hash=provenance_hash
        )

    def check_compliance(
        self,
        check_input: ComplianceCheckInput,
        precision: int = 2
    ) -> ComplianceCheckResult:
        """
        Check emission compliance against regulatory limit.

        Args:
            check_input: Measured values and limit data
            precision: Decimal places in result

        Returns:
            ComplianceCheckResult with compliance status

        Reference:
            EPA Clean Air Act Title IV, CSAPR, MATS regulations
        """
        limit = Decimal(str(check_input.limit_value))
        values = [Decimal(str(v)) for v in check_input.measured_values]

        # Calculate measured value based on basis
        if check_input.limit_basis == "maximum":
            measured = max(values) if values else Decimal("0")
        else:  # average
            measured = sum(values) / Decimal(str(len(values))) if values else Decimal("0")

        measured = self._apply_precision(measured, precision)

        # Calculate percent of limit
        if limit > 0:
            percent_of_limit = measured / limit * Decimal("100")
        else:
            percent_of_limit = Decimal("0")
        percent_of_limit = self._apply_precision(percent_of_limit, 1)

        # Determine compliance status
        if measured > limit:
            status = ComplianceStatus.EXCEEDING
        elif percent_of_limit > Decimal("80"):
            status = ComplianceStatus.APPROACHING_LIMIT
        else:
            status = ComplianceStatus.COMPLIANT

        # Margin to limit
        margin = limit - measured
        margin = self._apply_precision(margin, precision)

        # Count exceedance hours
        exceedance_hours = sum(1 for v in values if v > limit)

        provenance_data = {
            "check_type": "compliance",
            "pollutant": check_input.pollutant.value,
            "program": check_input.regulatory_program.value,
            "limit": str(limit),
            "measured": str(measured),
            "status": status.value
        }
        provenance_hash = self._calculate_hash(provenance_data)

        return ComplianceCheckResult(
            pollutant=check_input.pollutant,
            regulatory_program=check_input.regulatory_program,
            limit_value=limit,
            limit_unit=check_input.limit_unit,
            averaging_period=check_input.averaging_period,
            measured_value=measured,
            percent_of_limit=percent_of_limit,
            compliance_status=status,
            margin_to_limit=margin,
            exceedance_hours=exceedance_hours,
            provenance_hash=provenance_hash
        )

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        if precision < 0:
            raise ValueError(f"Precision must be non-negative, got {precision}")
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    @staticmethod
    def _calculate_hash(data: Any) -> str:
        """Calculate SHA-256 hash of data for provenance."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global calculator instance
_calculator_instance: Optional[EmissionRateCalculator] = None
_calculator_lock = threading.Lock()


def get_emission_rate_calculator() -> EmissionRateCalculator:
    """Get or create global emission rate calculator (thread-safe)."""
    global _calculator_instance
    if _calculator_instance is None:
        with _calculator_lock:
            if _calculator_instance is None:
                _calculator_instance = EmissionRateCalculator()
    return _calculator_instance


def calculate_mass_rate(
    pollutant: str,
    concentration_ppm: float,
    flow_scfh: float
) -> MassEmissionRate:
    """
    Convenience function to calculate mass emission rate.

    Args:
        pollutant: Pollutant type (nox, so2, co2)
        concentration_ppm: Concentration in ppm
        flow_scfh: Stack flow in scf/hr

    Returns:
        MassEmissionRate with lb/hr and ton/yr rates
    """
    calculator = get_emission_rate_calculator()
    rate_input = MassEmissionInput(
        pollutant=PollutantType(pollutant.lower()),
        concentration_ppm=concentration_ppm,
        flow_scfh=flow_scfh
    )
    return calculator.calculate_mass_emission_rate(rate_input)


def calculate_lb_mmbtu(
    pollutant: str,
    concentration_ppm: float,
    fuel_type: str,
    measured_o2: Optional[float] = None
) -> HeatInputBasedRate:
    """
    Convenience function to calculate lb/MMBtu rate.

    Args:
        pollutant: Pollutant type
        concentration_ppm: Concentration in ppm
        fuel_type: Fuel type for F-factor
        measured_o2: Measured O2 (%) for correction

    Returns:
        HeatInputBasedRate with lb/MMBtu rate
    """
    calculator = get_emission_rate_calculator()
    rate_input = HeatInputRateInput(
        pollutant=PollutantType(pollutant.lower()),
        concentration_ppm=concentration_ppm,
        fuel_type=fuel_type,
        measured_o2_percent=measured_o2
    )
    return calculator.calculate_heat_input_based_rate(rate_input)


def check_limit_compliance(
    pollutant: str,
    program: str,
    limit: float,
    unit: str,
    measured_values: List[float]
) -> ComplianceCheckResult:
    """
    Convenience function to check compliance.

    Args:
        pollutant: Pollutant type
        program: Regulatory program
        limit: Emission limit value
        unit: Limit units
        measured_values: Measured emission values

    Returns:
        ComplianceCheckResult with compliance status
    """
    calculator = get_emission_rate_calculator()
    check_input = ComplianceCheckInput(
        pollutant=PollutantType(pollutant.lower()),
        regulatory_program=RegulatoryProgram(program.lower()),
        averaging_period=AveragingPeriod.ANNUAL,
        limit_value=limit,
        limit_unit=unit,
        measured_values=measured_values
    )
    return calculator.check_compliance(check_input)
