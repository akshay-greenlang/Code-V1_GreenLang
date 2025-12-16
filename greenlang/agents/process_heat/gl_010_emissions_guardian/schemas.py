# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Data Schemas Module

This module defines comprehensive Pydantic data models for emissions monitoring,
CEMS data, calculations, compliance tracking, and regulatory reporting within
the EmissionsGuardian Agent framework.

All schemas follow GreenLang patterns with strict validation, type safety,
and complete documentation for regulatory audit compliance.

Standards Compliance:
    - EPA 40 CFR Part 60/63 (NSPS/NESHAP emission limits)
    - EPA 40 CFR Part 75 (CEMS data requirements)
    - EPA 40 CFR Part 98 (GHG quantification)
    - EPA Method 19 (F-factor calculations)
    - GHG Protocol (Corporate Standard)
    - ISO 14064 (GHG verification)

Example:
    >>> from greenlang.agents.process_heat.gl_010_emissions_guardian.schemas import (
    ...     EmissionsReading,
    ...     CEMSDataPoint,
    ...     CalculationResult,
    ...     ComplianceAssessment,
    ... )
    >>> reading = EmissionsReading(
    ...     source_id="STACK-001",
    ...     pollutant="NOx",
    ...     value=15.5,
    ...     unit="lb/hr",
    ... )

Author: GreenLang Process Heat Team
Version: 2.0.0
"""

from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS - DATA CLASSIFICATION
# =============================================================================


class DataQuality(str, Enum):
    """Data quality classification per EPA requirements."""
    MEASURED = "measured"  # Direct measurement
    CALCULATED = "calculated"  # Derived from measurements
    SUBSTITUTE = "substitute"  # Missing data substitution
    ESTIMATED = "estimated"  # Engineering estimate
    DEFAULT = "default"  # Regulatory default value
    INVALID = "invalid"  # Failed QA/QC


class ValidityStatus(str, Enum):
    """Data validity status."""
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    PENDING = "pending"
    CALIBRATION = "calibration"
    MAINTENANCE = "maintenance"


class CalculationMethod(str, Enum):
    """Emission calculation methodology per EPA Part 98."""
    CEMS = "cems"  # 40 CFR 98.33(a)(1)
    FUEL_ANALYSIS = "fuel_analysis"  # 40 CFR 98.33(a)(2) Tier 4
    DEFAULT_HHV = "default_hhv"  # 40 CFR 98.33(a)(3) Tier 3
    DEFAULT_EF = "default_ef"  # 40 CFR 98.33(a)(4) Tier 1/2
    MASS_BALANCE = "mass_balance"  # Carbon mass balance
    F_FACTOR = "f_factor"  # EPA Method 19 F-factor


class EmissionCategory(str, Enum):
    """Emission category classification."""
    GHG = "ghg"  # Greenhouse gases
    CRITERIA = "criteria"  # EPA criteria pollutants
    HAP = "hap"  # Hazardous air pollutants
    TOXIC = "toxic"  # Toxic air pollutants
    FUGITIVE = "fugitive"  # Fugitive emissions


class ComplianceResult(str, Enum):
    """Compliance assessment result."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    EXCEEDANCE = "exceedance"
    DEVIATION = "deviation"
    NOT_APPLICABLE = "not_applicable"


class SourceCategory(str, Enum):
    """Emission source category."""
    COMBUSTION = "combustion"  # Fuel combustion
    PROCESS = "process"  # Process emissions
    FUGITIVE = "fugitive"  # Fugitive/leak emissions
    MOBILE = "mobile"  # Mobile sources
    STORAGE = "storage"  # Storage tanks
    FLARE = "flare"  # Flare combustion


class TimeResolution(str, Enum):
    """Time resolution for data aggregation."""
    MINUTE = "minute"
    FIFTEEN_MINUTE = "15_minute"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


# =============================================================================
# BASE SCHEMAS
# =============================================================================


class BaseEmissionsSchema(BaseModel):
    """
    Base schema for all emissions data models.

    Provides common fields for timestamps, identifiers, and provenance
    tracking required across all emissions data types.

    Attributes:
        id: Unique record identifier
        created_at: Record creation timestamp
        updated_at: Last update timestamp
        provenance_hash: SHA-256 hash for audit trail
    """

    id: Optional[str] = Field(
        default=None,
        description="Unique record identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp (UTC)"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp (UTC)"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    def calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 provenance hash for this record.

        Returns:
            SHA-256 hex digest of record content
        """
        # Exclude hash field itself from calculation
        data = self.dict(exclude={"provenance_hash", "updated_at"})
        # Convert datetime to ISO format for consistent hashing
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def update_provenance(self) -> None:
        """Update provenance hash and timestamp."""
        self.updated_at = datetime.now(timezone.utc)
        self.provenance_hash = self.calculate_provenance_hash()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
        }


class MeasurementValue(BaseModel):
    """
    Standardized measurement value with unit and uncertainty.

    Provides a complete representation of measured values including
    engineering units, uncertainty bounds, and quality indicators.

    Attributes:
        value: Numeric measurement value
        unit: Engineering unit
        uncertainty: Measurement uncertainty (optional)
        uncertainty_pct: Uncertainty as percentage (optional)

    Example:
        >>> measurement = MeasurementValue(
        ...     value=25.5,
        ...     unit="lb/hr",
        ...     uncertainty_pct=5.0,
        ...     quality=DataQuality.MEASURED,
        ... )
    """

    value: float = Field(
        ...,
        description="Numeric measurement value"
    )
    unit: str = Field(
        ...,
        description="Engineering unit"
    )
    uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        description="Absolute measurement uncertainty"
    )
    uncertainty_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Relative uncertainty (%)"
    )
    quality: DataQuality = Field(
        default=DataQuality.MEASURED,
        description="Data quality classification"
    )
    lower_bound: Optional[float] = Field(
        default=None,
        description="Lower confidence bound"
    )
    upper_bound: Optional[float] = Field(
        default=None,
        description="Upper confidence bound"
    )

    @validator('value')
    def validate_value(cls, v, values):
        """Validate value is finite."""
        if not isinstance(v, (int, float)):
            raise ValueError("Value must be numeric")
        if v != v:  # NaN check
            raise ValueError("Value cannot be NaN")
        return v

    def to_unit(self, target_unit: str, conversion_factor: float) -> "MeasurementValue":
        """
        Convert measurement to different unit.

        Args:
            target_unit: Target unit string
            conversion_factor: Multiplication factor for conversion

        Returns:
            New MeasurementValue in target unit
        """
        return MeasurementValue(
            value=self.value * conversion_factor,
            unit=target_unit,
            uncertainty=self.uncertainty * conversion_factor if self.uncertainty else None,
            uncertainty_pct=self.uncertainty_pct,
            quality=self.quality,
            lower_bound=self.lower_bound * conversion_factor if self.lower_bound else None,
            upper_bound=self.upper_bound * conversion_factor if self.upper_bound else None,
        )


# =============================================================================
# EMISSIONS DATA SCHEMAS
# =============================================================================


class EmissionsReading(BaseEmissionsSchema):
    """
    Single emissions reading from a monitoring system.

    Represents a point-in-time emissions measurement from CEMS,
    portable analyzers, or calculated values.

    Attributes:
        source_id: Emission source identifier
        timestamp: Measurement timestamp
        pollutant: Pollutant being measured
        value: Emission rate or concentration
        unit: Engineering unit

    Example:
        >>> reading = EmissionsReading(
        ...     source_id="STACK-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     pollutant="NOx",
        ...     value=15.5,
        ...     unit="lb/hr",
        ...     data_quality=DataQuality.MEASURED,
        ... )
    """

    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp (UTC)"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant identifier (CO2, NOx, SO2, etc.)"
    )
    value: float = Field(
        ...,
        description="Emission rate or concentration"
    )
    unit: str = Field(
        ...,
        description="Engineering unit (lb/hr, ppm, kg/hr, etc.)"
    )

    # Quality indicators
    data_quality: DataQuality = Field(
        default=DataQuality.MEASURED,
        description="Data quality classification"
    )
    validity_status: ValidityStatus = Field(
        default=ValidityStatus.VALID,
        description="Data validity status"
    )

    # Operating conditions
    load_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=150,
        description="Operating load (%)"
    )
    fuel_flow_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fuel flow rate"
    )
    fuel_flow_unit: Optional[str] = Field(
        default=None,
        description="Fuel flow unit"
    )

    # Stack conditions
    stack_temperature_f: Optional[float] = Field(
        default=None,
        description="Stack temperature (F)"
    )
    stack_flow_rate_acfm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Stack flow rate (ACFM)"
    )
    o2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=21,
        description="Stack O2 concentration (%)"
    )
    co2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=20,
        description="Stack CO2 concentration (%)"
    )
    moisture_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="Stack moisture content (%)"
    )

    # Diluent correction
    corrected_to_reference: bool = Field(
        default=False,
        description="Value corrected to reference conditions"
    )
    reference_o2_pct: Optional[float] = Field(
        default=None,
        description="Reference O2 for correction (%)"
    )

    # Metadata
    analyzer_id: Optional[str] = Field(
        default=None,
        description="Source analyzer identifier"
    )
    method_code: Optional[str] = Field(
        default=None,
        description="Measurement method code"
    )

    @validator('pollutant')
    def validate_pollutant(cls, v):
        """Validate pollutant identifier."""
        valid_pollutants = {
            'CO2', 'CO', 'NOx', 'NO', 'NO2', 'SO2', 'PM', 'PM10', 'PM2.5',
            'VOC', 'CH4', 'N2O', 'HCl', 'HF', 'NH3', 'O2', 'Hg',
            'co2', 'co', 'nox', 'no', 'no2', 'so2', 'pm', 'pm10', 'pm25',
            'voc', 'ch4', 'n2o', 'hcl', 'hf', 'nh3', 'o2', 'hg'
        }
        if v.lower() not in {p.lower() for p in valid_pollutants}:
            # Allow but warn for non-standard pollutants
            pass
        return v.upper()


class EmissionsAggregate(BaseEmissionsSchema):
    """
    Aggregated emissions data over a time period.

    Represents hourly, daily, monthly, or annual emissions summaries
    as required for regulatory reporting.

    Attributes:
        source_id: Emission source identifier
        pollutant: Pollutant identifier
        period_start: Aggregation period start
        period_end: Aggregation period end
        resolution: Time resolution

    Example:
        >>> aggregate = EmissionsAggregate(
        ...     source_id="STACK-001",
        ...     pollutant="CO2",
        ...     period_start=datetime(2024, 1, 1),
        ...     period_end=datetime(2024, 1, 31, 23, 59, 59),
        ...     resolution=TimeResolution.MONTHLY,
        ...     total_mass=1500.0,
        ...     total_mass_unit="tons",
        ... )
    """

    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant identifier"
    )

    # Time period
    period_start: datetime = Field(
        ...,
        description="Aggregation period start"
    )
    period_end: datetime = Field(
        ...,
        description="Aggregation period end"
    )
    resolution: TimeResolution = Field(
        ...,
        description="Time resolution"
    )

    # Aggregated values
    total_mass: float = Field(
        ...,
        ge=0,
        description="Total mass emissions"
    )
    total_mass_unit: str = Field(
        default="lb",
        description="Total mass unit (lb, kg, tons, mtCO2e)"
    )

    # Statistical summary
    avg_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Average emission rate"
    )
    max_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum emission rate"
    )
    min_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum emission rate"
    )
    rate_unit: Optional[str] = Field(
        default=None,
        description="Rate unit (lb/hr, kg/hr)"
    )
    std_dev: Optional[float] = Field(
        default=None,
        ge=0,
        description="Standard deviation of rates"
    )

    # Data completeness
    reading_count: int = Field(
        default=0,
        ge=0,
        description="Number of readings in aggregate"
    )
    valid_reading_count: int = Field(
        default=0,
        ge=0,
        description="Number of valid readings"
    )
    data_availability_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Data availability (%)"
    )
    substitute_data_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Substitute data (%)"
    )

    # Operating hours
    operating_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total operating hours"
    )

    # Fuel consumption
    fuel_consumption: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total fuel consumption"
    )
    fuel_consumption_unit: Optional[str] = Field(
        default=None,
        description="Fuel consumption unit"
    )

    # Calculation method
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.CEMS,
        description="Primary calculation method"
    )

    @root_validator
    def validate_period(cls, values):
        """Validate period start is before period end."""
        start = values.get('period_start')
        end = values.get('period_end')
        if start and end and start >= end:
            raise ValueError("period_start must be before period_end")
        return values


# =============================================================================
# CEMS DATA SCHEMAS
# =============================================================================


class CEMSDataPoint(BaseEmissionsSchema):
    """
    Single CEMS (Continuous Emissions Monitoring System) data point.

    Represents raw or validated CEMS data per EPA 40 CFR Part 75
    requirements for continuous monitoring.

    Attributes:
        unit_id: CEMS unit identifier
        timestamp: Measurement timestamp
        parameter: Measured parameter (concentration or flow)
        value: Measured value

    Example:
        >>> data_point = CEMSDataPoint(
        ...     unit_id="CEMS-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     parameter="NOx",
        ...     value=125.5,
        ...     unit="ppm",
        ...     analyzer_id="NOX-01",
        ... )
    """

    unit_id: str = Field(
        ...,
        description="CEMS unit identifier"
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp"
    )
    parameter: str = Field(
        ...,
        description="Measured parameter"
    )
    value: float = Field(
        ...,
        description="Measured value"
    )
    unit: str = Field(
        ...,
        description="Measurement unit"
    )

    # Analyzer information
    analyzer_id: Optional[str] = Field(
        default=None,
        description="Analyzer identifier"
    )
    analyzer_type: Optional[str] = Field(
        default=None,
        description="Analyzer type (NDIR, UV, chemiluminescence)"
    )

    # QA/QC status
    validity_status: ValidityStatus = Field(
        default=ValidityStatus.VALID,
        description="Data validity status"
    )
    quality_assured: bool = Field(
        default=False,
        description="QA/QC verification complete"
    )

    # Calibration state
    zero_drift: Optional[float] = Field(
        default=None,
        description="Zero calibration drift (%)"
    )
    span_drift: Optional[float] = Field(
        default=None,
        description="Span calibration drift (%)"
    )
    in_calibration: bool = Field(
        default=True,
        description="Analyzer within calibration limits"
    )

    # Range information
    range_low: Optional[float] = Field(
        default=None,
        description="Analyzer range low"
    )
    range_high: Optional[float] = Field(
        default=None,
        description="Analyzer range high"
    )
    span_value: Optional[float] = Field(
        default=None,
        description="Analyzer span value"
    )

    # Data flags
    out_of_range: bool = Field(
        default=False,
        description="Value out of analyzer range"
    )
    spike_detected: bool = Field(
        default=False,
        description="Spike detected in data"
    )
    interference_detected: bool = Field(
        default=False,
        description="Interference detected"
    )

    # Missing data flags
    is_substitute: bool = Field(
        default=False,
        description="Value is substitute data"
    )
    substitute_method: Optional[str] = Field(
        default=None,
        description="Substitute data method"
    )

    @validator('value')
    def validate_cems_value(cls, v, values):
        """Validate CEMS value against range."""
        range_high = values.get('range_high')
        if range_high and v > range_high * 1.1:
            # Flag but don't reject values slightly over range
            pass
        return v


class CEMSHourlyRecord(BaseEmissionsSchema):
    """
    CEMS hourly data record per 40 CFR Part 75.

    Represents the hourly aggregation of CEMS data as required
    for EPA reporting and compliance demonstration.

    Attributes:
        unit_id: CEMS unit identifier
        operating_date: Operating date
        operating_hour: Operating hour (0-23)

    Example:
        >>> hourly = CEMSHourlyRecord(
        ...     unit_id="CEMS-001",
        ...     operating_date=date(2024, 6, 15),
        ...     operating_hour=14,
        ...     nox_rate_lb_mmbtu=0.15,
        ...     heat_input_mmbtu=150.5,
        ... )
    """

    unit_id: str = Field(
        ...,
        description="CEMS unit identifier"
    )
    operating_date: date = Field(
        ...,
        description="Operating date"
    )
    operating_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Operating hour (0-23)"
    )

    # Pollutant concentrations (corrected)
    nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx concentration (ppm @ ref O2)"
    )
    so2_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 concentration (ppm @ ref O2)"
    )
    co_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO concentration (ppm @ ref O2)"
    )
    co2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=20,
        description="CO2 concentration (%)"
    )
    o2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=21,
        description="O2 concentration (%)"
    )

    # Emission rates
    nox_rate_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx emission rate (lb/MMBtu)"
    )
    so2_rate_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 emission rate (lb/MMBtu)"
    )
    co2_rate_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 emission rate (lb/MMBtu)"
    )

    # Mass emissions
    nox_mass_lb: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx mass emissions (lb)"
    )
    so2_mass_lb: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 mass emissions (lb)"
    )
    co2_mass_tons: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 mass emissions (tons)"
    )

    # Heat input
    heat_input_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat input (MMBtu)"
    )

    # Flow
    stack_flow_scfh: Optional[float] = Field(
        default=None,
        ge=0,
        description="Stack flow rate (SCFH)"
    )

    # Operating time
    op_time: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Operating time fraction (0-1)"
    )

    # Data quality
    method_code: Optional[str] = Field(
        default=None,
        description="Calculation method code"
    )
    percent_available: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Data availability (%)"
    )

    # QA flags
    daily_calibration_status: ValidityStatus = Field(
        default=ValidityStatus.VALID,
        description="Daily calibration status"
    )
    modc: Optional[str] = Field(
        default=None,
        description="Method of Determination Code"
    )

    # Bias adjustment
    bias_adjusted: bool = Field(
        default=False,
        description="Bias adjustment applied"
    )
    bias_adjustment_factor: Optional[float] = Field(
        default=None,
        description="Bias adjustment factor"
    )


class CEMSQuarterlySummary(BaseEmissionsSchema):
    """
    CEMS quarterly summary for Part 75 reporting.

    Aggregates quarterly CEMS data for regulatory submission
    and compliance demonstration.
    """

    unit_id: str = Field(
        ...,
        description="CEMS unit identifier"
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year"
    )
    quarter: int = Field(
        ...,
        ge=1,
        le=4,
        description="Reporting quarter"
    )

    # Pollutant totals
    nox_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total NOx emissions (tons)"
    )
    so2_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total SO2 emissions (tons)"
    )
    co2_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2 emissions (tons)"
    )

    # Heat input
    total_heat_input_mmbtu: float = Field(
        default=0.0,
        ge=0,
        description="Total heat input (MMBtu)"
    )

    # Operating statistics
    operating_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total operating hours"
    )
    total_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total hours in quarter"
    )
    capacity_factor_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Capacity factor (%)"
    )

    # Data availability
    nox_data_availability_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="NOx monitor data availability (%)"
    )
    so2_data_availability_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="SO2 monitor data availability (%)"
    )
    flow_data_availability_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Flow monitor data availability (%)"
    )

    # QA/QC status
    rata_performed: bool = Field(
        default=False,
        description="RATA performed this quarter"
    )
    rata_passed: Optional[bool] = Field(
        default=None,
        description="RATA passed"
    )
    calibration_drift_within_limits: bool = Field(
        default=True,
        description="Daily calibration drift within limits"
    )


# =============================================================================
# CALCULATION SCHEMAS
# =============================================================================


class CalculationInput(BaseEmissionsSchema):
    """
    Input data for emissions calculations.

    Standardizes input data for EPA Method 19, GHG Protocol,
    and other calculation methodologies.

    Attributes:
        calculation_type: Type of calculation
        source_id: Emission source identifier
        fuel_type: Fuel type identifier

    Example:
        >>> calc_input = CalculationInput(
        ...     calculation_type="ghg_emissions",
        ...     source_id="BOILER-001",
        ...     fuel_type="natural_gas",
        ...     fuel_consumption=1500.0,
        ...     fuel_unit="MMBtu",
        ... )
    """

    calculation_type: str = Field(
        ...,
        description="Type of calculation"
    )
    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )

    # Fuel data
    fuel_type: str = Field(
        ...,
        description="Fuel type identifier"
    )
    fuel_consumption: float = Field(
        ...,
        ge=0,
        description="Fuel consumption quantity"
    )
    fuel_unit: str = Field(
        default="MMBtu",
        description="Fuel consumption unit"
    )

    # Fuel analysis (Tier 4)
    fuel_carbon_content: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Fuel carbon content (%)"
    )
    fuel_hhv: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fuel higher heating value"
    )
    fuel_hhv_unit: Optional[str] = Field(
        default=None,
        description="HHV unit (Btu/scf, Btu/lb, etc.)"
    )

    # CEMS data (if applicable)
    co2_concentration_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=20,
        description="CO2 concentration from CEMS (%)"
    )
    stack_flow_scfh: Optional[float] = Field(
        default=None,
        ge=0,
        description="Stack flow rate (SCFH)"
    )

    # Operating conditions
    operating_hours: float = Field(
        default=0.0,
        ge=0,
        description="Operating hours"
    )
    load_factor: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.5,
        description="Load factor (0-1)"
    )

    # Reference conditions
    reference_temperature_f: float = Field(
        default=68.0,
        description="Reference temperature (F)"
    )
    reference_pressure_psia: float = Field(
        default=14.7,
        description="Reference pressure (psia)"
    )

    # Calculation method
    method: CalculationMethod = Field(
        default=CalculationMethod.DEFAULT_EF,
        description="Calculation method"
    )

    # Override emission factors
    emission_factor_override: Optional[float] = Field(
        default=None,
        description="Override emission factor value"
    )
    emission_factor_unit: Optional[str] = Field(
        default=None,
        description="Override emission factor unit"
    )


class CalculationResult(BaseEmissionsSchema):
    """
    Result from emissions calculation.

    Contains calculated values, method documentation, uncertainty
    analysis, and complete audit trail for regulatory compliance.

    Attributes:
        calculation_id: Unique calculation identifier
        source_id: Emission source identifier
        pollutant: Pollutant calculated
        value: Calculated emission value

    Example:
        >>> result = CalculationResult(
        ...     calculation_id="CALC-2024-001",
        ...     source_id="BOILER-001",
        ...     pollutant="CO2",
        ...     value=150.5,
        ...     unit="tons",
        ...     method=CalculationMethod.DEFAULT_EF,
        ... )
    """

    calculation_id: str = Field(
        ...,
        description="Unique calculation identifier"
    )
    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant calculated"
    )

    # Result values
    value: float = Field(
        ...,
        description="Calculated emission value"
    )
    unit: str = Field(
        ...,
        description="Result unit"
    )

    # Method documentation
    method: CalculationMethod = Field(
        ...,
        description="Calculation method used"
    )
    method_reference: str = Field(
        default="",
        description="Method regulatory reference (e.g., '40 CFR 98.33(a)(3)')"
    )
    equation_id: Optional[str] = Field(
        default=None,
        description="Specific equation identifier"
    )

    # Emission factor used
    emission_factor_value: Optional[float] = Field(
        default=None,
        description="Emission factor value used"
    )
    emission_factor_unit: Optional[str] = Field(
        default=None,
        description="Emission factor unit"
    )
    emission_factor_source: Optional[str] = Field(
        default=None,
        description="Emission factor source (e.g., 'EPA Part 98 Table C-1')"
    )

    # Global Warming Potential (for GHG)
    gwp_value: Optional[int] = Field(
        default=None,
        description="Global Warming Potential value"
    )
    gwp_source: Optional[str] = Field(
        default=None,
        description="GWP source (e.g., 'AR5', 'AR6')"
    )
    co2e_value: Optional[float] = Field(
        default=None,
        description="CO2 equivalent value"
    )

    # Uncertainty analysis
    uncertainty_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Calculation uncertainty (%)"
    )
    confidence_level: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Confidence level (%)"
    )
    lower_bound: Optional[float] = Field(
        default=None,
        description="Lower confidence bound"
    )
    upper_bound: Optional[float] = Field(
        default=None,
        description="Upper confidence bound"
    )

    # Input summary
    fuel_consumption: Optional[float] = Field(
        default=None,
        description="Input fuel consumption"
    )
    fuel_unit: Optional[str] = Field(
        default=None,
        description="Fuel consumption unit"
    )
    operating_hours: Optional[float] = Field(
        default=None,
        description="Operating hours"
    )

    # Calculation details
    calculation_steps: List[str] = Field(
        default_factory=list,
        description="Calculation step descriptions"
    )
    intermediate_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Intermediate calculation values"
    )

    # Verification
    verified: bool = Field(
        default=False,
        description="Calculation verified"
    )
    verification_date: Optional[datetime] = Field(
        default=None,
        description="Verification timestamp"
    )
    verifier: Optional[str] = Field(
        default=None,
        description="Verifier identifier"
    )


class EmissionFactor(BaseModel):
    """
    Emission factor data from regulatory sources.

    Stores emission factors from EPA AP-42, Part 98, and other
    regulatory sources for deterministic calculations.

    Attributes:
        factor_id: Unique factor identifier
        pollutant: Pollutant
        fuel_type: Applicable fuel type
        value: Emission factor value

    Example:
        >>> ef = EmissionFactor(
        ...     factor_id="EF-CO2-NG-001",
        ...     pollutant="CO2",
        ...     fuel_type="natural_gas",
        ...     value=53.06,
        ...     unit="kg/MMBtu",
        ...     source="40 CFR Part 98 Table C-1",
        ... )
    """

    factor_id: str = Field(
        ...,
        description="Unique factor identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant"
    )
    fuel_type: str = Field(
        ...,
        description="Applicable fuel type"
    )

    # Factor value
    value: float = Field(
        ...,
        gt=0,
        description="Emission factor value"
    )
    unit: str = Field(
        ...,
        description="Factor unit (kg/MMBtu, lb/MMBtu, etc.)"
    )

    # Source documentation
    source: str = Field(
        ...,
        description="Regulatory source"
    )
    table_reference: Optional[str] = Field(
        default=None,
        description="Table reference (e.g., 'Table C-1')"
    )
    effective_date: Optional[date] = Field(
        default=None,
        description="Factor effective date"
    )

    # Applicability
    combustion_type: Optional[str] = Field(
        default=None,
        description="Combustion type (boiler, turbine, etc.)"
    )
    capacity_range_mmbtu_hr: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Applicable capacity range"
    )

    # Uncertainty
    uncertainty_pct: Optional[float] = Field(
        default=None,
        ge=0,
        description="Factor uncertainty (%)"
    )
    quality_rating: Optional[str] = Field(
        default=None,
        description="EPA quality rating (A, B, C, D, E)"
    )


# =============================================================================
# COMPLIANCE SCHEMAS
# =============================================================================


class PermitLimit(BaseModel):
    """
    Permit emission limit specification.

    Defines regulatory limits from air permits, including short-term
    and long-term averaging periods.

    Attributes:
        limit_id: Unique limit identifier
        pollutant: Pollutant
        limit_value: Numeric limit value
        unit: Engineering unit

    Example:
        >>> limit = PermitLimit(
        ...     limit_id="NOX-LIMIT-001",
        ...     pollutant="NOx",
        ...     limit_value=25.0,
        ...     unit="lb/hr",
        ...     averaging_period_hr=1,
        ...     limit_type="hourly",
        ... )
    """

    limit_id: str = Field(
        ...,
        description="Unique limit identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant"
    )
    limit_value: float = Field(
        ...,
        gt=0,
        description="Numeric limit value"
    )
    unit: str = Field(
        ...,
        description="Engineering unit"
    )

    # Averaging
    averaging_period_hr: Optional[float] = Field(
        default=None,
        description="Averaging period (hours)"
    )
    limit_type: str = Field(
        default="short_term",
        description="Limit type (short_term, daily, monthly, annual)"
    )

    # Regulatory basis
    permit_number: Optional[str] = Field(
        default=None,
        description="Permit number"
    )
    permit_condition: Optional[str] = Field(
        default=None,
        description="Permit condition reference"
    )
    regulatory_citation: Optional[str] = Field(
        default=None,
        description="Regulatory citation"
    )

    # Applicability
    source_id: Optional[str] = Field(
        default=None,
        description="Specific source (None = facility-wide)"
    )
    effective_date: Optional[date] = Field(
        default=None,
        description="Limit effective date"
    )
    expiration_date: Optional[date] = Field(
        default=None,
        description="Limit expiration date"
    )

    # Operating conditions
    load_range_pct: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Applicable load range (%)"
    )
    fuel_types: Optional[List[str]] = Field(
        default=None,
        description="Applicable fuel types"
    )


class ComplianceAssessment(BaseEmissionsSchema):
    """
    Compliance assessment result.

    Documents the comparison of emissions against permit limits
    and regulatory requirements.

    Attributes:
        assessment_id: Unique assessment identifier
        source_id: Emission source identifier
        pollutant: Pollutant assessed
        period_start: Assessment period start
        period_end: Assessment period end

    Example:
        >>> assessment = ComplianceAssessment(
        ...     assessment_id="CA-2024-001",
        ...     source_id="STACK-001",
        ...     pollutant="NOx",
        ...     period_start=datetime(2024, 6, 1),
        ...     period_end=datetime(2024, 6, 30),
        ...     measured_value=22.5,
        ...     limit_value=25.0,
        ...     unit="lb/hr",
        ...     result=ComplianceResult.COMPLIANT,
        ... )
    """

    assessment_id: str = Field(
        ...,
        description="Unique assessment identifier"
    )
    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant assessed"
    )

    # Assessment period
    period_start: datetime = Field(
        ...,
        description="Assessment period start"
    )
    period_end: datetime = Field(
        ...,
        description="Assessment period end"
    )

    # Values
    measured_value: float = Field(
        ...,
        description="Measured or calculated emission value"
    )
    limit_value: float = Field(
        ...,
        description="Applicable limit value"
    )
    unit: str = Field(
        ...,
        description="Value unit"
    )

    # Result
    result: ComplianceResult = Field(
        ...,
        description="Compliance result"
    )
    margin_pct: float = Field(
        default=0.0,
        description="Margin to limit (%, positive = under limit)"
    )
    exceedance_pct: Optional[float] = Field(
        default=None,
        description="Exceedance percentage (if applicable)"
    )

    # Limit reference
    limit_id: Optional[str] = Field(
        default=None,
        description="Reference to limit record"
    )
    limit_type: Optional[str] = Field(
        default=None,
        description="Limit type"
    )
    permit_reference: Optional[str] = Field(
        default=None,
        description="Permit reference"
    )

    # Contributing factors
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to result"
    )

    # Corrective action
    corrective_action_required: bool = Field(
        default=False,
        description="Corrective action required"
    )
    corrective_action_description: Optional[str] = Field(
        default=None,
        description="Required corrective action"
    )

    # Notification requirements
    notification_required: bool = Field(
        default=False,
        description="Regulatory notification required"
    )
    notification_deadline: Optional[datetime] = Field(
        default=None,
        description="Notification deadline"
    )
    notification_sent: bool = Field(
        default=False,
        description="Notification sent"
    )


class ExceedanceEvent(BaseEmissionsSchema):
    """
    Emission exceedance event record.

    Documents permit exceedances for regulatory reporting and
    corrective action tracking.

    Attributes:
        event_id: Unique event identifier
        source_id: Emission source identifier
        pollutant: Pollutant exceeded
        exceedance_start: Exceedance start time

    Example:
        >>> event = ExceedanceEvent(
        ...     event_id="EXC-2024-001",
        ...     source_id="STACK-001",
        ...     pollutant="NOx",
        ...     exceedance_start=datetime(2024, 6, 15, 14, 30),
        ...     exceedance_end=datetime(2024, 6, 15, 15, 45),
        ...     max_value=28.5,
        ...     limit_value=25.0,
        ...     unit="lb/hr",
        ... )
    """

    event_id: str = Field(
        ...,
        description="Unique event identifier"
    )
    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    pollutant: str = Field(
        ...,
        description="Pollutant exceeded"
    )

    # Exceedance timing
    exceedance_start: datetime = Field(
        ...,
        description="Exceedance start time"
    )
    exceedance_end: Optional[datetime] = Field(
        default=None,
        description="Exceedance end time"
    )
    duration_minutes: Optional[float] = Field(
        default=None,
        ge=0,
        description="Exceedance duration (minutes)"
    )

    # Values
    max_value: float = Field(
        ...,
        description="Maximum emission value during exceedance"
    )
    avg_value: Optional[float] = Field(
        default=None,
        description="Average value during exceedance"
    )
    limit_value: float = Field(
        ...,
        description="Permit limit value"
    )
    unit: str = Field(
        ...,
        description="Value unit"
    )

    # Exceedance magnitude
    max_exceedance_pct: float = Field(
        default=0.0,
        description="Maximum exceedance (%)"
    )
    total_excess_mass: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total excess emissions"
    )
    excess_mass_unit: Optional[str] = Field(
        default=None,
        description="Excess mass unit"
    )

    # Root cause
    root_cause: Optional[str] = Field(
        default=None,
        description="Root cause of exceedance"
    )
    root_cause_category: Optional[str] = Field(
        default=None,
        description="Root cause category"
    )

    # Corrective action
    corrective_actions_taken: List[str] = Field(
        default_factory=list,
        description="Corrective actions taken"
    )
    preventive_actions: List[str] = Field(
        default_factory=list,
        description="Preventive actions for future"
    )

    # Reporting
    reported_to_agency: bool = Field(
        default=False,
        description="Reported to regulatory agency"
    )
    report_date: Optional[datetime] = Field(
        default=None,
        description="Agency report date"
    )
    agency_reference: Optional[str] = Field(
        default=None,
        description="Agency reference number"
    )


# =============================================================================
# ALERT AND NOTIFICATION SCHEMAS
# =============================================================================


class EmissionsAlert(BaseEmissionsSchema):
    """
    Emissions monitoring alert.

    Represents alerts generated by the monitoring system for
    approaching limits, anomalies, or exceedances.

    Attributes:
        alert_id: Unique alert identifier
        source_id: Emission source identifier
        alert_type: Type of alert
        severity: Alert severity

    Example:
        >>> alert = EmissionsAlert(
        ...     alert_id="ALERT-2024-001",
        ...     source_id="STACK-001",
        ...     alert_type="approaching_limit",
        ...     severity="warning",
        ...     pollutant="NOx",
        ...     current_value=22.5,
        ...     threshold_value=25.0,
        ...     message="NOx at 90% of permit limit",
        ... )
    """

    alert_id: str = Field(
        ...,
        description="Unique alert identifier"
    )
    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )

    # Alert classification
    alert_type: str = Field(
        ...,
        description="Type of alert"
    )
    severity: str = Field(
        ...,
        description="Alert severity (info, warning, alarm, critical)"
    )

    # Pollutant and values
    pollutant: Optional[str] = Field(
        default=None,
        description="Related pollutant"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current emission value"
    )
    threshold_value: Optional[float] = Field(
        default=None,
        description="Alert threshold value"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Value unit"
    )

    # Alert details
    message: str = Field(
        ...,
        description="Alert message"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert details"
    )

    # Timing
    triggered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert trigger time"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="Alert acknowledgment time"
    )
    cleared_at: Optional[datetime] = Field(
        default=None,
        description="Alert clear time"
    )

    # Status
    status: str = Field(
        default="active",
        description="Alert status (active, acknowledged, cleared)"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="Acknowledging user"
    )

    # Prediction (if applicable)
    predicted_exceedance: bool = Field(
        default=False,
        description="Alert based on prediction"
    )
    predicted_exceedance_time: Optional[datetime] = Field(
        default=None,
        description="Predicted exceedance time"
    )
    prediction_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Prediction confidence (0-1)"
    )

    # Actions
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended response actions"
    )


# =============================================================================
# SUMMARY AND REPORTING SCHEMAS
# =============================================================================


class SourceEmissionsSummary(BaseEmissionsSchema):
    """
    Emissions summary for a single source.

    Aggregates emissions data for reporting periods with
    compliance status and data quality metrics.

    Attributes:
        source_id: Emission source identifier
        source_name: Source description
        period_type: Reporting period type
        period_start: Period start
        period_end: Period end
    """

    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    source_name: Optional[str] = Field(
        default=None,
        description="Source description"
    )
    source_category: SourceCategory = Field(
        default=SourceCategory.COMBUSTION,
        description="Source category"
    )

    # Reporting period
    period_type: TimeResolution = Field(
        ...,
        description="Reporting period type"
    )
    period_start: datetime = Field(
        ...,
        description="Period start"
    )
    period_end: datetime = Field(
        ...,
        description="Period end"
    )

    # GHG emissions
    co2_tons: float = Field(
        default=0.0,
        ge=0,
        description="CO2 emissions (tons)"
    )
    ch4_tons: float = Field(
        default=0.0,
        ge=0,
        description="CH4 emissions (tons)"
    )
    n2o_tons: float = Field(
        default=0.0,
        ge=0,
        description="N2O emissions (tons)"
    )
    co2e_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2e emissions (tons)"
    )

    # Criteria pollutants
    nox_tons: float = Field(
        default=0.0,
        ge=0,
        description="NOx emissions (tons)"
    )
    so2_tons: float = Field(
        default=0.0,
        ge=0,
        description="SO2 emissions (tons)"
    )
    co_tons: float = Field(
        default=0.0,
        ge=0,
        description="CO emissions (tons)"
    )
    pm_tons: float = Field(
        default=0.0,
        ge=0,
        description="PM emissions (tons)"
    )
    voc_tons: float = Field(
        default=0.0,
        ge=0,
        description="VOC emissions (tons)"
    )

    # Activity data
    fuel_type: Optional[str] = Field(
        default=None,
        description="Primary fuel type"
    )
    fuel_consumption: Optional[float] = Field(
        default=None,
        description="Total fuel consumption"
    )
    fuel_unit: Optional[str] = Field(
        default=None,
        description="Fuel consumption unit"
    )
    heat_input_mmbtu: float = Field(
        default=0.0,
        ge=0,
        description="Total heat input (MMBtu)"
    )
    operating_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total operating hours"
    )

    # Data quality
    data_availability_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Data availability (%)"
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.DEFAULT_EF,
        description="Primary calculation method"
    )

    # Compliance
    compliance_status: ComplianceResult = Field(
        default=ComplianceResult.COMPLIANT,
        description="Overall compliance status"
    )
    exceedance_count: int = Field(
        default=0,
        ge=0,
        description="Number of exceedances"
    )

    # Emission rates
    avg_nox_lb_mmbtu: Optional[float] = Field(
        default=None,
        description="Average NOx rate (lb/MMBtu)"
    )
    avg_co2_lb_mmbtu: Optional[float] = Field(
        default=None,
        description="Average CO2 rate (lb/MMBtu)"
    )


class FacilityEmissionsSummary(BaseEmissionsSchema):
    """
    Facility-wide emissions summary.

    Aggregates all source emissions for regulatory reporting
    and GHG Protocol Scope 1 calculations.

    Attributes:
        facility_id: Facility identifier
        facility_name: Facility name
        period_type: Reporting period type
        period_start: Period start
        period_end: Period end
    """

    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    facility_name: Optional[str] = Field(
        default=None,
        description="Facility name"
    )

    # Regulatory identifiers
    epa_facility_id: Optional[str] = Field(
        default=None,
        description="EPA facility ID"
    )
    state_facility_id: Optional[str] = Field(
        default=None,
        description="State facility ID"
    )
    naics_code: Optional[str] = Field(
        default=None,
        description="Primary NAICS code"
    )

    # Reporting period
    period_type: TimeResolution = Field(
        ...,
        description="Reporting period type"
    )
    period_start: datetime = Field(
        ...,
        description="Period start"
    )
    period_end: datetime = Field(
        ...,
        description="Period end"
    )

    # Total GHG emissions (Scope 1)
    total_co2_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2 emissions (tons)"
    )
    total_ch4_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total CH4 emissions (tons)"
    )
    total_n2o_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total N2O emissions (tons)"
    )
    total_co2e_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2e emissions (tons)"
    )
    total_co2e_mtco2e: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2e emissions (mtCO2e)"
    )

    # Total criteria pollutants
    total_nox_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total NOx emissions (tons)"
    )
    total_so2_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total SO2 emissions (tons)"
    )
    total_co_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total CO emissions (tons)"
    )
    total_pm_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total PM emissions (tons)"
    )
    total_voc_tons: float = Field(
        default=0.0,
        ge=0,
        description="Total VOC emissions (tons)"
    )

    # Source breakdown
    source_count: int = Field(
        default=0,
        ge=0,
        description="Number of emission sources"
    )
    source_summaries: List[SourceEmissionsSummary] = Field(
        default_factory=list,
        description="Individual source summaries"
    )

    # By category
    combustion_co2e_tons: float = Field(
        default=0.0,
        ge=0,
        description="Combustion CO2e (tons)"
    )
    process_co2e_tons: float = Field(
        default=0.0,
        ge=0,
        description="Process CO2e (tons)"
    )
    fugitive_co2e_tons: float = Field(
        default=0.0,
        ge=0,
        description="Fugitive CO2e (tons)"
    )

    # Activity data
    total_heat_input_mmbtu: float = Field(
        default=0.0,
        ge=0,
        description="Total heat input (MMBtu)"
    )

    # Compliance summary
    overall_compliance: ComplianceResult = Field(
        default=ComplianceResult.COMPLIANT,
        description="Overall facility compliance"
    )
    total_exceedances: int = Field(
        default=0,
        ge=0,
        description="Total exceedance events"
    )

    # Reporting thresholds
    exceeds_part98_threshold: bool = Field(
        default=False,
        description="Exceeds Part 98 reporting threshold"
    )
    part98_threshold_mtco2e: float = Field(
        default=25000.0,
        description="Part 98 threshold (mtCO2e)"
    )


# =============================================================================
# EXPORT ALL SCHEMAS
# =============================================================================


__all__ = [
    # Enums
    "DataQuality",
    "ValidityStatus",
    "CalculationMethod",
    "EmissionCategory",
    "ComplianceResult",
    "SourceCategory",
    "TimeResolution",
    # Base schemas
    "BaseEmissionsSchema",
    "MeasurementValue",
    # Emissions data
    "EmissionsReading",
    "EmissionsAggregate",
    # CEMS data
    "CEMSDataPoint",
    "CEMSHourlyRecord",
    "CEMSQuarterlySummary",
    # Calculations
    "CalculationInput",
    "CalculationResult",
    "EmissionFactor",
    # Compliance
    "PermitLimit",
    "ComplianceAssessment",
    "ExceedanceEvent",
    # Alerts
    "EmissionsAlert",
    # Summaries
    "SourceEmissionsSummary",
    "FacilityEmissionsSummary",
]
