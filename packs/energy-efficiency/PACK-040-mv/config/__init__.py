"""
PACK-040 Measurement & Verification (M&V) Pack - Configuration Module

This module implements the MVConfig and PackConfig classes that load, merge,
and validate all configuration for the Measurement & Verification Pack. It
provides comprehensive Pydantic v2 models for baseline development with
multivariate regression and change-point models, routine and non-routine
adjustments per IPMVP, ASHRAE Guideline 14 statistical validation, all four
IPMVP options, fractional savings uncertainty quantification, metering plan
management, multi-year persistence tracking, and automated M&V reporting.

Facility Types:
    - COMMERCIAL_OFFICE: Weather-dependent offices (HVAC 40-55%), Option C, 3P/4P models
    - MANUFACTURING: Production-dependent plants, Option B, production normalization
    - RETAIL_PORTFOLIO: Multi-site retail, Option C, weather + sales normalization
    - HOSPITAL: 24/7 healthcare, Option B, steam/CHW metering, high stability
    - UNIVERSITY_CAMPUS: Multi-building campuses, Option C per building, academic calendar
    - GOVERNMENT_FEMP: Federal facilities, FEMP 4.0 compliance, ESPC verification
    - ESCO_PERFORMANCE_CONTRACT: Performance guarantees, multi-year, shared savings
    - PORTFOLIO: Statistical sampling across sites, portfolio-level uncertainty

IPMVP Options:
    - OPTION_A: Retrofit Isolation - Key Parameter Measurement (stipulated non-measured)
    - OPTION_B: Retrofit Isolation - All Parameter Measurement (short-term or continuous)
    - OPTION_C: Whole Facility comparison using utility meters
    - OPTION_D: Calibrated Simulation using energy models (DOE-2, EnergyPlus)

Baseline Model Types:
    - OLS: Ordinary Least Squares (single or multivariate linear regression)
    - THREE_PARAMETER_COOLING: E = a + b*max(0, T - Tcp) for cooling-dominant
    - THREE_PARAMETER_HEATING: E = a + b*max(0, Thp - T) for heating-dominant
    - FOUR_PARAMETER: E = a + bh*max(0, Thp - T) + bc*max(0, T - Tcp)
    - FIVE_PARAMETER: Separate heating and cooling change-points
    - TOWT: Time-of-Week and Temperature (hourly/daily schedules)
    - MVR: Multivariate Regression (multiple independent variables)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (commercial_office / manufacturing / retail_portfolio /
       hospital / university_campus / government_femp /
       esco_performance_contract / portfolio_mv)
    3. Environment overrides (MV_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - IPMVP Core Concepts 2022 (Efficiency Valuation Organization)
    - ASHRAE Guideline 14-2014 (Measurement of Energy, Demand, Water Savings)
    - ISO 50015:2014 (Measurement and Verification of Energy Performance)
    - ISO 50001:2018 (Energy Management Systems)
    - ISO 50006:2014 (Energy Baselines and Energy Performance Indicators)
    - FEMP M&V Guidelines 4.0 (Federal Energy Management Program)
    - EU EED Article 7 (Directive 2023/1791 energy savings verification)
    - EU EPC Directive 2012/27/EU Article 18 (Performance contract verification)

Example:
    >>> from packs.energy_efficiency.PACK_040_mv.config import (
    ...     PackConfig,
    ...     MVConfig,
    ...     FacilityType,
    ...     IPMVPOption,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("commercial_office")
    >>> print(config.pack.facility_type)
    FacilityType.COMMERCIAL_OFFICE
    >>> print(config.pack.ipmvp.default_option)
    IPMVPOption.OPTION_C
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - Measurement & Verification configuration enumeration types
# =============================================================================


class FacilityType(str, Enum):
    """Facility type classification for M&V scoping and preset selection."""

    COMMERCIAL_OFFICE = "COMMERCIAL_OFFICE"
    MANUFACTURING = "MANUFACTURING"
    RETAIL_PORTFOLIO = "RETAIL_PORTFOLIO"
    HOSPITAL = "HOSPITAL"
    UNIVERSITY_CAMPUS = "UNIVERSITY_CAMPUS"
    GOVERNMENT_FEMP = "GOVERNMENT_FEMP"
    ESCO_PERFORMANCE_CONTRACT = "ESCO_PERFORMANCE_CONTRACT"
    PORTFOLIO = "PORTFOLIO"


class IPMVPOption(str, Enum):
    """IPMVP option selection per EVO Core Concepts 2022.

    Option A: Retrofit Isolation - Key Parameter Measurement.
    Option B: Retrofit Isolation - All Parameter Measurement.
    Option C: Whole Facility comparison using utility meters.
    Option D: Calibrated Simulation using energy models.
    """

    OPTION_A = "OPTION_A"
    OPTION_B = "OPTION_B"
    OPTION_C = "OPTION_C"
    OPTION_D = "OPTION_D"


class BaselineModelType(str, Enum):
    """Regression model type for energy baseline development.

    OLS: Ordinary Least Squares linear regression.
    THREE_PARAMETER_COOLING: Flat + slope above cooling change-point.
    THREE_PARAMETER_HEATING: Flat + slope below heating change-point.
    FOUR_PARAMETER: Heating slope + flat + cooling slope (single change-point).
    FIVE_PARAMETER: Heating slope + flat + cooling slope (two change-points).
    TOWT: Time-of-Week and Temperature model for hourly/daily data.
    MVR: Multivariate regression with multiple independent variables.
    """

    OLS = "OLS"
    THREE_PARAMETER_COOLING = "THREE_PARAMETER_COOLING"
    THREE_PARAMETER_HEATING = "THREE_PARAMETER_HEATING"
    FOUR_PARAMETER = "FOUR_PARAMETER"
    FIVE_PARAMETER = "FIVE_PARAMETER"
    TOWT = "TOWT"
    MVR = "MVR"


class AdjustmentType(str, Enum):
    """Types of routine and non-routine adjustments per IPMVP.

    Routine adjustments account for expected operating condition changes.
    Non-routine adjustments account for unexpected or one-time changes.
    """

    WEATHER = "WEATHER"
    PRODUCTION = "PRODUCTION"
    OCCUPANCY = "OCCUPANCY"
    OPERATING_HOURS = "OPERATING_HOURS"
    FLOOR_AREA = "FLOOR_AREA"
    EQUIPMENT_CHANGE = "EQUIPMENT_CHANGE"
    SCHEDULE_CHANGE = "SCHEDULE_CHANGE"
    STATIC_FACTOR = "STATIC_FACTOR"


class SavingsType(str, Enum):
    """Savings calculation methodology.

    AVOIDED: Adjusted baseline prediction minus reporting period actual.
    NORMALIZED: Baseline at standard conditions minus reporting at standard.
    COST: Energy savings times blended rate plus demand savings times demand rate.
    CUMULATIVE: Sum of periodic savings over tracking period.
    ANNUALIZED: Periodic savings scaled to full year.
    """

    AVOIDED = "AVOIDED"
    NORMALIZED = "NORMALIZED"
    COST = "COST"
    CUMULATIVE = "CUMULATIVE"
    ANNUALIZED = "ANNUALIZED"


class UncertaintyComponent(str, Enum):
    """Components of savings uncertainty per ASHRAE Guideline 14.

    MEASUREMENT: Meter accuracy, CT/PT errors, calibration drift.
    MODEL: Regression standard error, prediction interval width.
    SAMPLING: Sample size, coefficient of variation, t-distribution.
    COMBINED: Root-sum-square of independent components.
    """

    MEASUREMENT = "MEASUREMENT"
    MODEL = "MODEL"
    SAMPLING = "SAMPLING"
    COMBINED = "COMBINED"


class MeterAccuracy(str, Enum):
    """Meter accuracy class per ANSI C12.20 and IEC 62053.

    CLASS_01: Revenue-grade accuracy (0.1% error).
    CLASS_02: Revenue-grade accuracy (0.2% error).
    CLASS_05: Standard commercial accuracy (0.5% error).
    CLASS_10: Sub-metering accuracy (1.0% error).
    CLASS_20: Monitoring-grade accuracy (2.0% error).
    """

    CLASS_01 = "CLASS_01"
    CLASS_02 = "CLASS_02"
    CLASS_05 = "CLASS_05"
    CLASS_10 = "CLASS_10"
    CLASS_20 = "CLASS_20"


class PersistenceModel(str, Enum):
    """Savings persistence degradation model type.

    LINEAR: Constant annual degradation rate.
    EXPONENTIAL: Degradation accelerates over time.
    STEP: Sudden step-change losses (e.g., control reset, equipment failure).
    """

    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"
    STEP = "STEP"


class ComplianceFramework(str, Enum):
    """M&V compliance framework for standards conformity checking.

    IPMVP: International Performance Measurement and Verification Protocol.
    ISO_50015: ISO 50015:2014 Measurement and verification of energy performance.
    FEMP: Federal Energy Management Program M&V Guidelines 4.0.
    ASHRAE_14: ASHRAE Guideline 14-2014 statistical criteria.
    EU_EED: EU Energy Efficiency Directive Article 7 savings verification.
    """

    IPMVP = "IPMVP"
    ISO_50015 = "ISO_50015"
    FEMP = "FEMP"
    ASHRAE_14 = "ASHRAE_14"
    EU_EED = "EU_EED"


class OutputFormat(str, Enum):
    """Output format for M&V reports and deliverables."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"


class ReportingFrequency(str, Enum):
    """Reporting and verification frequency for M&V analysis."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


# =============================================================================
# Reference Data Constants
# =============================================================================


FACILITY_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "COMMERCIAL_OFFICE": {
        "name": "Commercial Office",
        "typical_ipmvp_option": "OPTION_C",
        "typical_baseline_model": "FOUR_PARAMETER",
        "key_independent_variables": ["outdoor_temperature", "occupancy"],
        "hvac_share_pct": "40-55%",
        "baseline_granularity": "MONTHLY",
        "typical_ecms": [
            "HVAC upgrades", "LED lighting", "Building envelope",
            "Controls optimization", "VFD installations",
        ],
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_ipmvp_option": "OPTION_B",
        "typical_baseline_model": "MVR",
        "key_independent_variables": ["production_volume", "outdoor_temperature", "shift_count"],
        "hvac_share_pct": "10-20%",
        "baseline_granularity": "DAILY",
        "typical_ecms": [
            "Motor replacements", "Compressed air optimization",
            "Process heat recovery", "VFD installations", "Boiler upgrades",
        ],
    },
    "RETAIL_PORTFOLIO": {
        "name": "Retail Chain / Store Portfolio",
        "typical_ipmvp_option": "OPTION_C",
        "typical_baseline_model": "THREE_PARAMETER_COOLING",
        "key_independent_variables": ["outdoor_temperature", "sales_volume"],
        "hvac_share_pct": "25-35%",
        "baseline_granularity": "MONTHLY",
        "typical_ecms": [
            "LED lighting", "HVAC replacements", "Refrigeration upgrades",
            "EMS installation", "Building envelope",
        ],
    },
    "HOSPITAL": {
        "name": "Hospital / Healthcare Facility",
        "typical_ipmvp_option": "OPTION_B",
        "typical_baseline_model": "FIVE_PARAMETER",
        "key_independent_variables": ["outdoor_temperature", "patient_days", "steam_load"],
        "hvac_share_pct": "35-45%",
        "baseline_granularity": "WEEKLY",
        "typical_ecms": [
            "Chiller plant optimization", "Steam system upgrades",
            "LED lighting", "HVAC retro-commissioning", "Building envelope",
        ],
    },
    "UNIVERSITY_CAMPUS": {
        "name": "University / Campus",
        "typical_ipmvp_option": "OPTION_C",
        "typical_baseline_model": "FIVE_PARAMETER",
        "key_independent_variables": ["outdoor_temperature", "student_fte", "academic_schedule"],
        "hvac_share_pct": "35-45%",
        "baseline_granularity": "MONTHLY",
        "typical_ecms": [
            "Central plant optimization", "Building HVAC upgrades",
            "LED lighting campus-wide", "Lab ventilation optimization",
            "Building automation upgrades",
        ],
    },
    "GOVERNMENT_FEMP": {
        "name": "Federal Government (FEMP)",
        "typical_ipmvp_option": "OPTION_C",
        "typical_baseline_model": "FOUR_PARAMETER",
        "key_independent_variables": ["outdoor_temperature", "occupancy"],
        "hvac_share_pct": "35-50%",
        "baseline_granularity": "MONTHLY",
        "typical_ecms": [
            "ESPC measures", "UESC measures", "LED lighting",
            "HVAC modernization", "Building automation",
        ],
    },
    "ESCO_PERFORMANCE_CONTRACT": {
        "name": "ESCO / Performance Contract",
        "typical_ipmvp_option": "OPTION_C",
        "typical_baseline_model": "FOUR_PARAMETER",
        "key_independent_variables": ["outdoor_temperature", "production_volume"],
        "hvac_share_pct": "varies",
        "baseline_granularity": "MONTHLY",
        "typical_ecms": [
            "Bundled ECMs per contract", "Guaranteed savings measures",
            "Capital equipment replacements", "Controls optimization",
        ],
    },
    "PORTFOLIO": {
        "name": "Multi-Site Portfolio",
        "typical_ipmvp_option": "OPTION_C",
        "typical_baseline_model": "OLS",
        "key_independent_variables": ["outdoor_temperature", "floor_area"],
        "hvac_share_pct": "varies",
        "baseline_granularity": "MONTHLY",
        "typical_ecms": [
            "Portfolio-wide LED lighting", "HVAC standardization",
            "Building automation rollout", "Envelope improvements",
        ],
    },
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "commercial_office": (
        "Weather-dependent office buildings, Option C whole-facility, "
        "3P/4P change-point models, HDD/CDD normalization, monthly granularity"
    ),
    "manufacturing": (
        "Production-dependent manufacturing, Option B retrofit isolation, "
        "production normalization, multi-shift scheduling, daily granularity"
    ),
    "retail_portfolio": (
        "Multi-site retail portfolio, Option C, weather + sales normalization, "
        "portfolio-level sampling, monthly granularity"
    ),
    "hospital": (
        "24/7 healthcare facility, Option B specific systems, steam/CHW metering, "
        "high baseline stability requirement, weekly granularity"
    ),
    "university_campus": (
        "Multi-building university campus, Option C per building, "
        "academic calendar adjustments, central plant M&V, monthly granularity"
    ),
    "government_femp": (
        "Federal government FEMP 4.0 compliance, ESPC contract verification, "
        "federal reporting requirements, annual granularity"
    ),
    "esco_performance_contract": (
        "ESCO/EPC performance guarantee verification, multi-year tracking, "
        "shared savings calculations, dispute resolution, monthly granularity"
    ),
    "portfolio_mv": (
        "Statistical sampling across sites, portfolio-level uncertainty, "
        "aggregated savings reporting, quarterly granularity"
    ),
}

IPMVP_OPTION_INFO: Dict[str, Dict[str, Any]] = {
    "OPTION_A": {
        "name": "Retrofit Isolation - Key Parameter Measurement",
        "description": (
            "Savings are determined by field measurement of the key parameter(s) "
            "that define the energy use of the affected system(s). Parameters not "
            "selected for field measurement are estimated."
        ),
        "typical_ecms": [
            "Lighting retrofits (measure wattage, stipulate hours)",
            "Motor replacements (measure kW, stipulate hours)",
            "Insulation (measure heat loss, stipulate conditions)",
        ],
        "accuracy": "Low to Medium",
        "cost": "Low",
        "boundary": "Retrofit isolation (equipment or system level)",
        "measurement": "Key parameters only; others stipulated",
    },
    "OPTION_B": {
        "name": "Retrofit Isolation - All Parameter Measurement",
        "description": (
            "Savings are determined by field measurement of all parameters "
            "needed to determine the energy use of the systems affected by the ECM."
        ),
        "typical_ecms": [
            "Chiller plant optimization (continuous metering)",
            "VFD installations (metered motor input)",
            "Boiler replacements (fuel meter + output meter)",
        ],
        "accuracy": "Medium to High",
        "cost": "Medium",
        "boundary": "Retrofit isolation (equipment or system level)",
        "measurement": "All energy-related parameters measured",
    },
    "OPTION_C": {
        "name": "Whole Facility",
        "description": (
            "Savings are determined by measuring energy use at the whole facility "
            "or sub-facility level. Regression analysis of whole-building data is "
            "the most common approach."
        ),
        "typical_ecms": [
            "Multiple bundled ECMs affecting >10% of whole-facility energy",
            "Building-wide HVAC upgrades",
            "Comprehensive energy retrofits",
        ],
        "accuracy": "Medium (depends on ECM share of total)",
        "cost": "Low (uses existing utility meters)",
        "boundary": "Whole facility or sub-facility",
        "measurement": "Utility meters (existing or upgraded)",
    },
    "OPTION_D": {
        "name": "Calibrated Simulation",
        "description": (
            "Savings are determined through simulation of the energy use of the "
            "facility or affected systems. The simulation model must be calibrated "
            "so that it predicts energy use that closely matches actual metered data."
        ),
        "typical_ecms": [
            "New construction (no pre-retrofit baseline available)",
            "Complex interactive ECMs",
            "Deep energy retrofits with major envelope changes",
        ],
        "accuracy": "Medium to High (model-dependent)",
        "cost": "High (modeling effort)",
        "boundary": "Whole facility or system (simulated)",
        "measurement": "Calibration data from utility meters + spot measurements",
    },
}

ASHRAE_14_CRITERIA: Dict[str, Dict[str, float]] = {
    "MONTHLY": {
        "max_cvrmse_pct": 25.0,
        "max_nmbe_pct": 5.0,
        "min_r_squared": 0.70,
    },
    "DAILY": {
        "max_cvrmse_pct": 30.0,
        "max_nmbe_pct": 10.0,
        "min_r_squared": 0.50,
    },
    "HOURLY": {
        "max_cvrmse_pct": 30.0,
        "max_nmbe_pct": 10.0,
        "min_r_squared": 0.0,
    },
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class BaselineConfig(BaseModel):
    """Configuration for energy baseline development.

    Defines the baseline period, independent variables, regression model
    preferences, model validation criteria (ASHRAE 14), and automated
    model selection parameters used by the Baseline Engine.
    """

    baseline_start_date: Optional[str] = Field(
        None,
        description="Baseline period start date (ISO 8601, YYYY-MM-DD)",
    )
    baseline_end_date: Optional[str] = Field(
        None,
        description="Baseline period end date (ISO 8601, YYYY-MM-DD)",
    )
    baseline_months: int = Field(
        12,
        ge=6,
        le=36,
        description="Number of months in baseline period (minimum 12 recommended by IPMVP)",
    )
    granularity: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Baseline data granularity: DAILY, WEEKLY, or MONTHLY",
    )
    preferred_model_type: BaselineModelType = Field(
        BaselineModelType.FOUR_PARAMETER,
        description="Preferred regression model type for baseline development",
    )
    auto_model_selection: bool = Field(
        True,
        description="Enable automatic model selection comparing OLS, change-point, and TOWT",
    )
    model_selection_criterion: str = Field(
        "CVRMSE",
        description="Criterion for automatic model selection: CVRMSE, AIC, BIC, ADJUSTED_R2",
    )
    independent_variables: List[str] = Field(
        default_factory=lambda: ["outdoor_temperature"],
        description="Independent variables for regression (temperature, production, occupancy, etc.)",
    )
    max_cvrmse_pct: float = Field(
        25.0,
        ge=5.0,
        le=50.0,
        description="Maximum CVRMSE for model acceptance (ASHRAE 14: 25% monthly, 30% daily)",
    )
    max_nmbe_pct: float = Field(
        5.0,
        ge=1.0,
        le=15.0,
        description="Maximum NMBE for model acceptance (ASHRAE 14: 5% monthly, 10% daily)",
    )
    min_r_squared: float = Field(
        0.70,
        ge=0.0,
        le=0.99,
        description="Minimum R-squared for model acceptance (ASHRAE 14: 0.70 monthly)",
    )
    min_data_coverage_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum data coverage required in baseline period (%)",
    )
    outlier_removal_method: str = Field(
        "STUDENTIZED_RESIDUAL",
        description="Outlier removal method: STUDENTIZED_RESIDUAL, COOKS_DISTANCE, IQR, NONE",
    )
    outlier_threshold: float = Field(
        3.0,
        ge=1.5,
        le=5.0,
        description="Outlier threshold (standard deviations or Cook's distance multiplier)",
    )
    balance_point_optimization: bool = Field(
        True,
        description="Enable iterative balance point optimization for HDD/CDD regression",
    )
    heating_balance_point_c: float = Field(
        15.5,
        ge=5.0,
        le=25.0,
        description="Initial heating balance point temperature (Celsius)",
    )
    cooling_balance_point_c: float = Field(
        18.0,
        ge=10.0,
        le=30.0,
        description="Initial cooling balance point temperature (Celsius)",
    )
    min_observations: int = Field(
        12,
        ge=6,
        le=60,
        description="Minimum number of observations for regression fitting",
    )
    durbin_watson_check: bool = Field(
        True,
        description="Check Durbin-Watson statistic for residual autocorrelation",
    )
    f_test_significance: float = Field(
        0.05,
        ge=0.01,
        le=0.10,
        description="Significance level for F-test of overall model fit",
    )
    t_test_significance: float = Field(
        0.05,
        ge=0.01,
        le=0.10,
        description="Significance level for t-test of individual coefficients",
    )

    @model_validator(mode="after")
    def validate_balance_points(self) -> "BaselineConfig":
        """Cooling balance point must be >= heating balance point."""
        if self.cooling_balance_point_c < self.heating_balance_point_c:
            logger.warning(
                f"Cooling balance point ({self.cooling_balance_point_c} C) is below "
                f"heating balance point ({self.heating_balance_point_c} C). "
                "These should not overlap for change-point models."
            )
        return self


class AdjustmentConfig(BaseModel):
    """Configuration for routine and non-routine adjustments per IPMVP.

    Defines which adjustment types are enabled, their calculation methods,
    documentation requirements, and uncertainty attribution used by the
    Adjustment Engine.
    """

    routine_adjustments_enabled: bool = Field(
        True,
        description="Enable routine adjustments (weather, production, occupancy)",
    )
    non_routine_adjustments_enabled: bool = Field(
        True,
        description="Enable non-routine adjustments (floor area, equipment, schedule)",
    )
    weather_normalization: bool = Field(
        True,
        description="Apply HDD/CDD weather normalization to savings calculations",
    )
    production_normalization: bool = Field(
        False,
        description="Apply production volume normalization to savings calculations",
    )
    occupancy_adjustment: bool = Field(
        False,
        description="Apply occupancy-weighted adjustment to savings calculations",
    )
    operating_hours_adjustment: bool = Field(
        False,
        description="Apply operating hours correction to savings calculations",
    )
    floor_area_adjustment: bool = Field(
        False,
        description="Enable non-routine floor area change adjustment",
    )
    equipment_change_adjustment: bool = Field(
        False,
        description="Enable non-routine equipment addition/removal adjustment",
    )
    schedule_change_adjustment: bool = Field(
        False,
        description="Enable non-routine schedule change adjustment",
    )
    static_factor_adjustment: bool = Field(
        False,
        description="Enable non-routine static factor correction",
    )
    normalization_method: str = Field(
        "TMY",
        description="Weather normalization method: TMY (Typical Meteorological Year), ACTUAL, LONG_TERM_AVERAGE",
    )
    adjustment_documentation_required: bool = Field(
        True,
        description="Require documentation and justification for all non-routine adjustments",
    )
    adjustment_uncertainty_pct: float = Field(
        10.0,
        ge=0.0,
        le=50.0,
        description="Default uncertainty attributed to non-routine adjustments (%)",
    )
    max_non_routine_adjustments: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of non-routine adjustments per reporting period",
    )


class SavingsConfig(BaseModel):
    """Configuration for energy savings calculation methodology.

    Defines the primary and secondary savings types, cost parameters,
    demand charge handling, and cumulative tracking parameters used by
    the Savings Engine.
    """

    primary_savings_type: SavingsType = Field(
        SavingsType.AVOIDED,
        description="Primary savings calculation method",
    )
    secondary_savings_types: List[SavingsType] = Field(
        default_factory=lambda: [SavingsType.NORMALIZED, SavingsType.COST],
        description="Additional savings calculations to perform",
    )
    electricity_rate_per_kwh: float = Field(
        0.12,
        ge=0.01,
        le=1.0,
        description="Blended electricity rate (local currency/kWh)",
    )
    gas_rate_per_kwh: float = Field(
        0.04,
        ge=0.005,
        le=0.5,
        description="Natural gas rate (local currency/kWh thermal)",
    )
    steam_rate_per_kwh: float = Field(
        0.06,
        ge=0.01,
        le=0.5,
        description="Steam rate (local currency/kWh thermal)",
    )
    chilled_water_rate_per_kwh: float = Field(
        0.08,
        ge=0.01,
        le=0.5,
        description="Chilled water rate (local currency/kWh thermal)",
    )
    demand_charge_per_kw: float = Field(
        15.0,
        ge=0.0,
        le=100.0,
        description="Monthly demand charge (local currency/kW peak)",
    )
    include_demand_savings: bool = Field(
        True,
        description="Include peak demand reduction in cost savings calculation",
    )
    currency: str = Field(
        "USD",
        description="Currency for cost calculations (ISO 4217)",
    )
    escalation_rate_pct: float = Field(
        3.0,
        ge=0.0,
        le=15.0,
        description="Annual energy cost escalation rate for multi-year projections (%)",
    )
    discount_rate_pct: float = Field(
        8.0,
        ge=0.0,
        le=20.0,
        description="Discount rate for NPV calculations (%)",
    )
    cumulative_tracking_enabled: bool = Field(
        True,
        description="Track cumulative savings over the M&V contract period",
    )
    annualization_method: str = Field(
        "PROPORTIONAL",
        description="Method for annualizing partial-year savings: PROPORTIONAL, CALENDAR_MONTH",
    )
    negative_savings_handling: str = Field(
        "REPORT",
        description="How to handle negative savings: REPORT, ZERO_FLOOR, ALERT",
    )
    carbon_savings_tracking: bool = Field(
        True,
        description="Calculate carbon savings from energy savings using emission factors",
    )
    electricity_ef_kgco2_per_kwh: float = Field(
        0.40,
        ge=0.0,
        le=1.5,
        description="Electricity grid emission factor (kgCO2e/kWh) for carbon savings",
    )
    gas_ef_kgco2_per_kwh: float = Field(
        0.20,
        ge=0.0,
        le=0.5,
        description="Natural gas emission factor (kgCO2e/kWh) for carbon savings",
    )


class UncertaintyConfig(BaseModel):
    """Configuration for savings uncertainty quantification per ASHRAE 14.

    Defines confidence levels, uncertainty component settings, fractional
    savings uncertainty thresholds, and minimum detectable savings parameters
    used by the Uncertainty Engine.
    """

    enabled: bool = Field(
        True,
        description="Enable uncertainty quantification for savings calculations",
    )
    confidence_level_pct: float = Field(
        90.0,
        ge=68.0,
        le=99.0,
        description="Confidence level for uncertainty bounds (68% or 90% typical)",
    )
    measurement_uncertainty_enabled: bool = Field(
        True,
        description="Include meter measurement uncertainty in combined uncertainty",
    )
    model_uncertainty_enabled: bool = Field(
        True,
        description="Include regression model uncertainty in combined uncertainty",
    )
    sampling_uncertainty_enabled: bool = Field(
        False,
        description="Include sampling uncertainty for Option A key parameter measurement",
    )
    max_fractional_savings_uncertainty_pct: float = Field(
        50.0,
        ge=10.0,
        le=100.0,
        description="Maximum acceptable FSU at 68% confidence (ASHRAE 14 recommends <50%)",
    )
    minimum_detectable_savings_pct: float = Field(
        10.0,
        ge=1.0,
        le=50.0,
        description="Minimum savings that can be detected above model noise (%)",
    )
    default_meter_accuracy_pct: float = Field(
        1.0,
        ge=0.1,
        le=5.0,
        description="Default meter accuracy class for measurement uncertainty (%)",
    )
    ct_pt_error_pct: float = Field(
        0.5,
        ge=0.0,
        le=3.0,
        description="Current/potential transformer error contribution (%)",
    )
    calibration_drift_pct_per_year: float = Field(
        0.2,
        ge=0.0,
        le=2.0,
        description="Annual calibration drift rate for meters (%/year)",
    )
    propagation_method: str = Field(
        "ROOT_SUM_SQUARE",
        description="Uncertainty propagation method: ROOT_SUM_SQUARE, MONTE_CARLO",
    )
    monte_carlo_iterations: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo iterations if propagation_method is MONTE_CARLO",
    )


class IPMVPConfig(BaseModel):
    """Configuration for IPMVP option implementation.

    Defines the default IPMVP option, option selection criteria,
    measurement boundary rules, and compliance checking parameters
    used by the IPMVP Option Engine.
    """

    default_option: IPMVPOption = Field(
        IPMVPOption.OPTION_C,
        description="Default IPMVP option for new ECMs",
    )
    auto_option_selection: bool = Field(
        True,
        description="Enable automated IPMVP option recommendation based on ECM characteristics",
    )
    option_a_stipulated_values_review_months: int = Field(
        12,
        ge=6,
        le=36,
        description="Review interval for Option A stipulated parameter values (months)",
    )
    option_b_metering_duration_weeks: int = Field(
        4,
        ge=1,
        le=52,
        description="Default short-term metering duration for Option B (weeks)",
    )
    option_c_min_ecm_share_pct: float = Field(
        10.0,
        ge=5.0,
        le=50.0,
        description="Minimum ECM share of whole-facility energy for Option C viability (%)",
    )
    option_d_calibration_cvrmse_pct: float = Field(
        15.0,
        ge=5.0,
        le=30.0,
        description="Maximum CVRMSE for Option D simulation model calibration (%)",
    )
    option_d_calibration_nmbe_pct: float = Field(
        5.0,
        ge=1.0,
        le=15.0,
        description="Maximum NMBE for Option D simulation model calibration (%)",
    )
    measurement_boundary_documentation: bool = Field(
        True,
        description="Require documented measurement boundary for each ECM",
    )
    interactive_effects_assessment: bool = Field(
        True,
        description="Assess interactive effects between ECMs within the measurement boundary",
    )
    compliance_frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: [ComplianceFramework.IPMVP, ComplianceFramework.ASHRAE_14],
        description="M&V compliance frameworks to check against",
    )
    ipmvp_version: str = Field(
        "2022",
        description="IPMVP Core Concepts version for compliance checking",
    )


class RegressionConfig(BaseModel):
    """Configuration for statistical regression modeling.

    Defines the regression fitting parameters, diagnostic tests,
    residual analysis settings, and model comparison criteria used
    by the Regression Engine.
    """

    fitting_method: str = Field(
        "ORDINARY_LEAST_SQUARES",
        description="Regression fitting method: ORDINARY_LEAST_SQUARES, WEIGHTED_LEAST_SQUARES",
    )
    change_point_search_method: str = Field(
        "GRID_SEARCH",
        description="Change-point optimization: GRID_SEARCH, GOLDEN_SECTION, BRENT",
    )
    change_point_search_resolution_c: float = Field(
        0.5,
        ge=0.1,
        le=2.0,
        description="Temperature resolution for change-point grid search (Celsius)",
    )
    towt_time_bins: int = Field(
        12,
        ge=4,
        le=168,
        description="Number of time-of-week bins for TOWT model (12 = 2-hour bins)",
    )
    towt_temperature_bins: int = Field(
        6,
        ge=3,
        le=20,
        description="Number of temperature bins for TOWT model",
    )
    residual_normality_test: bool = Field(
        True,
        description="Perform Shapiro-Wilk normality test on regression residuals",
    )
    heteroscedasticity_test: bool = Field(
        True,
        description="Perform Breusch-Pagan test for heteroscedasticity",
    )
    influential_observation_detection: bool = Field(
        True,
        description="Detect influential observations using Cook's distance",
    )
    cooks_distance_threshold: float = Field(
        1.0,
        ge=0.5,
        le=4.0,
        description="Cook's distance threshold for influential observation flagging",
    )
    multicollinearity_check: bool = Field(
        True,
        description="Check VIF (Variance Inflation Factor) for multicollinearity in MVR",
    )
    max_vif: float = Field(
        10.0,
        ge=2.0,
        le=50.0,
        description="Maximum acceptable VIF for independent variables",
    )
    cross_validation_enabled: bool = Field(
        False,
        description="Enable k-fold cross-validation for model selection",
    )
    cross_validation_folds: int = Field(
        5,
        ge=3,
        le=10,
        description="Number of folds for cross-validation",
    )


class WeatherConfig(BaseModel):
    """Configuration for weather data and normalization.

    Defines weather data sources, degree-day calculation parameters,
    TMY normalization settings, and data quality thresholds used by
    the Weather Engine.
    """

    weather_source: str = Field(
        "LOCAL_STATION",
        description="Weather data source: LOCAL_STATION, TMY3, NOAA_ISD, CUSTOM_API",
    )
    weather_station_id: Optional[str] = Field(
        None,
        description="Weather station identifier (USAF-WBAN, ICAO, or custom ID)",
    )
    max_station_distance_km: float = Field(
        25.0,
        ge=1.0,
        le=100.0,
        description="Maximum acceptable distance from weather station to facility (km)",
    )
    degree_day_method: str = Field(
        "MEAN_TEMPERATURE",
        description="Degree-day calculation method: MEAN_TEMPERATURE, INTEGRATION, ASHRAE",
    )
    hdd_balance_point_c: float = Field(
        15.5,
        ge=5.0,
        le=25.0,
        description="Heating Degree Day balance point temperature (Celsius)",
    )
    cdd_balance_point_c: float = Field(
        18.0,
        ge=10.0,
        le=30.0,
        description="Cooling Degree Day balance point temperature (Celsius)",
    )
    balance_point_optimization: bool = Field(
        True,
        description="Enable iterative balance point optimization via regression",
    )
    tmy_source: str = Field(
        "TMY3",
        description="Typical Meteorological Year source: TMY3, TMY2, IWEC, CUSTOM",
    )
    min_weather_data_coverage_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum weather data completeness for analysis (%)",
    )
    temperature_unit: str = Field(
        "CELSIUS",
        description="Temperature unit: CELSIUS, FAHRENHEIT",
    )
    gap_fill_method: str = Field(
        "LINEAR_INTERPOLATION",
        description="Weather data gap-filling method: LINEAR_INTERPOLATION, NEARBY_STATION, NONE",
    )
    max_gap_hours: int = Field(
        6,
        ge=1,
        le=48,
        description="Maximum gap size to fill in weather data (hours)",
    )


class MeteringConfig(BaseModel):
    """Configuration for M&V metering plan and data management.

    Defines meter accuracy requirements, calibration schedules,
    sampling protocol parameters, data quality thresholds, and
    gap-handling rules used by the Metering Engine.
    """

    default_meter_accuracy: MeterAccuracy = Field(
        MeterAccuracy.CLASS_10,
        description="Default meter accuracy class for M&V measurements",
    )
    calibration_interval_months: int = Field(
        12,
        ge=3,
        le=60,
        description="Meter calibration interval (months)",
    )
    calibration_standard: str = Field(
        "ANSI_C12_20",
        description="Calibration standard: ANSI_C12_20, IEC_62053, FACILITY_SPECIFIC",
    )
    sampling_enabled: bool = Field(
        False,
        description="Enable statistical sampling for Option A key parameter measurement",
    )
    sampling_confidence_pct: float = Field(
        90.0,
        ge=80.0,
        le=99.0,
        description="Sampling confidence level for sample size calculation (%)",
    )
    sampling_precision_pct: float = Field(
        10.0,
        ge=5.0,
        le=20.0,
        description="Sampling precision (desired relative precision, %)",
    )
    sampling_cv_pct: float = Field(
        25.0,
        ge=5.0,
        le=100.0,
        description="Expected coefficient of variation for sampling population (%)",
    )
    min_data_completeness_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum meter data completeness threshold (%)",
    )
    max_gap_intervals: int = Field(
        4,
        ge=1,
        le=96,
        description="Maximum consecutive missing intervals before flagging data gap",
    )
    gap_fill_method: str = Field(
        "LINEAR_INTERPOLATION",
        description="Gap-filling method: LINEAR_INTERPOLATION, PREVIOUS_VALUE, AVERAGE, NONE",
    )
    data_validation_enabled: bool = Field(
        True,
        description="Enable automated data validation checks (range, consistency, completeness)",
    )
    negative_value_handling: str = Field(
        "FLAG",
        description="Handle negative meter readings: FLAG, ZERO, REJECT",
    )
    demand_window_minutes: int = Field(
        15,
        ge=5,
        le=60,
        description="Demand averaging window for peak demand calculation (minutes)",
    )


class PersistenceConfig(BaseModel):
    """Configuration for multi-year savings persistence tracking.

    Defines degradation model type, persistence thresholds,
    re-commissioning triggers, and performance guarantee tracking
    parameters used by the Persistence Engine.
    """

    enabled: bool = Field(
        True,
        description="Enable multi-year savings persistence tracking",
    )
    tracking_years: int = Field(
        5,
        ge=1,
        le=25,
        description="Number of years to track savings persistence",
    )
    degradation_model: PersistenceModel = Field(
        PersistenceModel.LINEAR,
        description="Savings degradation model type",
    )
    expected_annual_degradation_pct: float = Field(
        5.0,
        ge=0.0,
        le=25.0,
        description="Expected annual savings degradation rate (%)",
    )
    alert_threshold_pct: float = Field(
        20.0,
        ge=5.0,
        le=50.0,
        description="Savings degradation threshold for alert generation (%)",
    )
    recommissioning_trigger_pct: float = Field(
        30.0,
        ge=10.0,
        le=60.0,
        description="Savings degradation threshold for re-commissioning recommendation (%)",
    )
    seasonal_pattern_analysis: bool = Field(
        True,
        description="Analyze seasonal savings patterns for degradation detection",
    )
    persistence_factor_tracking: bool = Field(
        True,
        description="Calculate persistence factor (actual savings / expected savings ratio)",
    )
    minimum_persistence_factor: float = Field(
        0.70,
        ge=0.3,
        le=1.0,
        description="Minimum acceptable persistence factor before alert",
    )
    trend_analysis_method: str = Field(
        "LINEAR_REGRESSION",
        description="Trend analysis method: LINEAR_REGRESSION, MOVING_AVERAGE, EXPONENTIAL_SMOOTHING",
    )
    performance_guarantee_tracking: bool = Field(
        False,
        description="Track savings against performance guarantee in ESCO/EPC contracts",
    )
    guarantee_shortfall_alert: bool = Field(
        False,
        description="Alert when savings fall below guaranteed levels",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "PersistenceConfig":
        """Alert threshold must be below re-commissioning trigger threshold."""
        if self.alert_threshold_pct >= self.recommissioning_trigger_pct:
            logger.warning(
                f"Persistence alert threshold ({self.alert_threshold_pct}%) >= "
                f"re-commissioning trigger ({self.recommissioning_trigger_pct}%). "
                "Alert should fire before re-commissioning recommendation."
            )
        return self


class ReportingConfig(BaseModel):
    """Configuration for M&V report generation and scheduling.

    Defines reporting frequency, output formats, report content flags,
    compliance checking, and distribution parameters used by the
    MV Reporting Engine.
    """

    frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="M&V reporting frequency",
    )
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML],
        description="Output formats for M&V reports",
    )
    mv_plan_report: bool = Field(
        True,
        description="Generate M&V plan document per IPMVP",
    )
    baseline_report: bool = Field(
        True,
        description="Generate baseline analysis report with regression results",
    )
    savings_report: bool = Field(
        True,
        description="Generate savings verification report with uncertainty bounds",
    )
    uncertainty_report: bool = Field(
        True,
        description="Generate uncertainty analysis report",
    )
    annual_summary: bool = Field(
        True,
        description="Generate annual M&V summary with cumulative savings",
    )
    persistence_report: bool = Field(
        True,
        description="Generate persistence tracking report",
    )
    compliance_report: bool = Field(
        True,
        description="Generate standards compliance report",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary report (2-4 pages)",
    )
    option_comparison_report: bool = Field(
        True,
        description="Generate IPMVP option comparison report",
    )
    metering_plan_report: bool = Field(
        True,
        description="Generate metering plan document",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )
    include_regression_diagnostics: bool = Field(
        True,
        description="Include regression diagnostic plots in baseline reports",
    )
    include_residual_analysis: bool = Field(
        True,
        description="Include residual analysis charts in baseline reports",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration for M&V pack."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "mv_manager",
            "mv_practitioner",
            "energy_manager",
            "esco_manager",
            "facility_manager",
            "compliance_officer",
            "viewer",
            "admin",
        ],
        description="Available RBAC roles for the M&V pack",
    )
    data_classification: str = Field(
        "CONFIDENTIAL",
        description="Default data classification: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED",
    )
    audit_logging: bool = Field(
        True,
        description="Enable security audit logging for all M&V operations",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored M&V data",
    )
    tenant_data_isolation: bool = Field(
        True,
        description="Enforce tenant data isolation in multi-tenant deployments",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for M&V pack execution."""

    max_ecms_per_project: int = Field(
        50,
        ge=1,
        le=500,
        description="Maximum number of ECMs per M&V project",
    )
    max_baseline_data_points: int = Field(
        105120,
        ge=365,
        le=525600,
        description="Maximum data points per baseline regression (default: 3 years daily)",
    )
    regression_timeout_seconds: int = Field(
        15,
        ge=5,
        le=120,
        description="Timeout for individual regression fitting (seconds)",
    )
    savings_calculation_timeout_seconds: int = Field(
        5,
        ge=2,
        le=60,
        description="Timeout for savings calculation per ECM per period (seconds)",
    )
    uncertainty_timeout_seconds: int = Field(
        10,
        ge=5,
        le=120,
        description="Timeout for full ASHRAE 14 uncertainty propagation (seconds)",
    )
    report_generation_timeout_seconds: int = Field(
        10,
        ge=5,
        le=120,
        description="Timeout for individual report generation (seconds)",
    )
    cache_ttl_seconds: int = Field(
        600,
        ge=30,
        le=86400,
        description="Cache TTL for regression results and calculated values (seconds)",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=5000,
        description="Batch size for bulk data processing",
    )
    parallel_engines: int = Field(
        4,
        ge=1,
        le=8,
        description="Maximum number of engines running in parallel",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for M&V calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all M&V calculations",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    calculation_logging: bool = Field(
        True,
        description="Log all intermediate calculation steps (regression, adjustments, savings)",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track all assumptions, stipulated values, and default values used",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from raw meter data through verified savings",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=25,
        description="Audit trail retention period in years (FEMP requires 7 years)",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class MVConfig(BaseModel):
    """Main configuration for PACK-040 Measurement & Verification Pack.

    This is the root configuration model that contains all sub-configurations
    for M&V analysis. The facility_type field drives which IPMVP option,
    baseline model, adjustment types, and reporting parameters are
    prioritised for the deployment.

    The configuration supports the complete M&V lifecycle from baseline
    development through multi-year persistence tracking, implementing
    IPMVP Core Concepts 2022, ASHRAE Guideline 14-2014, ISO 50015:2014,
    FEMP M&V Guidelines 4.0, and EU EED Article 7.
    """

    # Project identification
    project_name: str = Field(
        "",
        description="M&V project name or identifier",
    )
    facility_name: str = Field(
        "",
        description="Facility name or site identifier",
    )
    company_name: str = Field(
        "",
        description="Legal entity name of the company",
    )
    facility_type: FacilityType = Field(
        FacilityType.COMMERCIAL_OFFICE,
        description="Primary facility type for M&V scoping and preset selection",
    )
    country: str = Field(
        "US",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Current reporting year for M&V analysis",
    )

    # Facility characteristics
    floor_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Total facility floor area in square metres",
    )
    annual_electricity_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual electricity consumption (kWh)",
    )
    annual_gas_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual natural gas consumption (kWh thermal)",
    )
    annual_steam_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual steam consumption (kWh thermal)",
    )
    annual_chilled_water_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual chilled water consumption (kWh thermal)",
    )
    operating_hours_per_year: int = Field(
        2500,
        ge=500,
        le=8760,
        description="Annual operating hours",
    )
    ecm_count: int = Field(
        1,
        ge=1,
        le=100,
        description="Number of Energy Conservation Measures in the M&V project",
    )
    contract_term_years: Optional[int] = Field(
        None,
        ge=1,
        le=25,
        description="Performance contract term in years (ESCO/EPC projects)",
    )

    # Sub-configurations
    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig,
        description="Baseline development configuration",
    )
    adjustments: AdjustmentConfig = Field(
        default_factory=AdjustmentConfig,
        description="Routine and non-routine adjustment configuration",
    )
    savings: SavingsConfig = Field(
        default_factory=SavingsConfig,
        description="Savings calculation configuration",
    )
    uncertainty: UncertaintyConfig = Field(
        default_factory=UncertaintyConfig,
        description="Uncertainty quantification configuration",
    )
    ipmvp: IPMVPConfig = Field(
        default_factory=IPMVPConfig,
        description="IPMVP option implementation configuration",
    )
    regression: RegressionConfig = Field(
        default_factory=RegressionConfig,
        description="Statistical regression modeling configuration",
    )
    weather: WeatherConfig = Field(
        default_factory=WeatherConfig,
        description="Weather data and normalization configuration",
    )
    metering: MeteringConfig = Field(
        default_factory=MeteringConfig,
        description="Metering plan and data management configuration",
    )
    persistence: PersistenceConfig = Field(
        default_factory=PersistenceConfig,
        description="Multi-year savings persistence tracking configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="M&V report generation configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and access control configuration",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and resource limits configuration",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )

    @model_validator(mode="after")
    def validate_option_c_ecm_share(self) -> "MVConfig":
        """Option C requires ECMs affecting a significant share of whole-facility energy."""
        if self.ipmvp.default_option == IPMVPOption.OPTION_C:
            if self.ecm_count == 1:
                logger.info(
                    "Single ECM with Option C: ensure ECM affects >10% of "
                    "whole-facility energy for statistically significant savings."
                )
        return self

    @model_validator(mode="after")
    def validate_hospital_stability(self) -> "MVConfig":
        """Healthcare facilities require high baseline stability for M&V accuracy."""
        if self.facility_type == FacilityType.HOSPITAL:
            if self.baseline.max_cvrmse_pct > 20.0:
                logger.warning(
                    "Hospital facility: CVRMSE threshold exceeds 20%. "
                    "24/7 facilities typically achieve tighter baseline fits."
                )
        return self

    @model_validator(mode="after")
    def validate_femp_compliance(self) -> "MVConfig":
        """FEMP facilities must have FEMP in compliance frameworks."""
        if self.facility_type == FacilityType.GOVERNMENT_FEMP:
            if ComplianceFramework.FEMP not in self.ipmvp.compliance_frameworks:
                logger.warning(
                    "Government FEMP facility: FEMP compliance framework is not enabled. "
                    "Federal ESPC/UESC projects require FEMP M&V Guidelines conformity."
                )
            if self.audit_trail.retention_years < 7:
                logger.warning(
                    "Government FEMP facility: audit trail retention "
                    f"({self.audit_trail.retention_years} years) is below the 7-year "
                    "FEMP requirement."
                )
        return self

    @model_validator(mode="after")
    def validate_esco_contract(self) -> "MVConfig":
        """ESCO/EPC projects must have persistence tracking and contract term configured."""
        if self.facility_type == FacilityType.ESCO_PERFORMANCE_CONTRACT:
            if not self.persistence.performance_guarantee_tracking:
                logger.warning(
                    "ESCO/EPC project: performance guarantee tracking is disabled. "
                    "Performance contracts require savings guarantee verification."
                )
            if self.contract_term_years is None:
                logger.warning(
                    "ESCO/EPC project: no contract_term_years configured. "
                    "Set the performance contract term for multi-year tracking."
                )
        return self

    @model_validator(mode="after")
    def validate_manufacturing_production(self) -> "MVConfig":
        """Manufacturing facilities should use production normalization."""
        if self.facility_type == FacilityType.MANUFACTURING:
            if not self.adjustments.production_normalization:
                logger.warning(
                    "Manufacturing facility: production normalization is disabled. "
                    "Production-dependent energy use requires production-normalised baselines."
                )
        return self

    @model_validator(mode="after")
    def validate_portfolio_sampling(self) -> "MVConfig":
        """Portfolio M&V should have sampling enabled for site selection."""
        if self.facility_type == FacilityType.PORTFOLIO:
            if not self.metering.sampling_enabled:
                logger.warning(
                    "Portfolio M&V: sampling is disabled. "
                    "Multi-site portfolios typically use statistical sampling for site selection."
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-040 M&V Pack.

    Handles preset loading, environment variable overrides, and
    configuration merging. Follows the standard GreenLang pack config
    pattern with from_preset(), from_yaml(), and merge() support.
    """

    pack: MVConfig = Field(
        default_factory=MVConfig,
        description="Main M&V configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-040-mv",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (commercial_office, manufacturing, etc.)
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in AVAILABLE_PRESETS.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(AVAILABLE_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)

        pack_config = MVConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = MVConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(
        cls,
        base: "PackConfig",
        overrides: Dict[str, Any],
    ) -> "PackConfig":
        """Create a new PackConfig by merging overrides into an existing config.

        Args:
            base: Base PackConfig instance.
            overrides: Dictionary of configuration overrides.

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = MVConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with MV_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: MV_PACK_BASELINE__MAX_CVRMSE_PCT=30
                 MV_PACK_SAVINGS__ELECTRICITY_RATE_PER_KWH=0.18
        """
        overrides: Dict[str, Any] = {}
        prefix = "MV_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value type
                if value.lower() in ("true", "yes", "1"):
                    current[parts[-1]] = True
                elif value.lower() in ("false", "no", "0"):
                    current[parts[-1]] = False
                else:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value
        return overrides

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_completeness(self) -> List[str]:
        """Validate configuration completeness and return warnings.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str, overrides: Optional[Dict[str, Any]] = None
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: MVConfig) -> List[str]:
    """Validate an M&V configuration and return any warnings.

    Performs comprehensive validation of the M&V configuration including
    project identification, ASHRAE 14 criteria consistency, IPMVP option
    compatibility, metering requirements, and compliance framework alignment.

    Args:
        config: MVConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check project identification
    if not config.project_name:
        warnings.append(
            "No project_name configured. Add a project name for report identification."
        )

    if not config.facility_name:
        warnings.append(
            "No facility_name configured. Add a facility name for report identification."
        )

    # Check baseline period configuration
    if config.baseline.baseline_months < 12:
        warnings.append(
            f"Baseline period ({config.baseline.baseline_months} months) is less than "
            "12 months. IPMVP recommends a minimum 12-month baseline to capture "
            "seasonal variation."
        )

    # Check ASHRAE 14 criteria consistency with granularity
    granularity = config.baseline.granularity.value
    ashrae_criteria = ASHRAE_14_CRITERIA.get(granularity.upper(), {})
    if ashrae_criteria:
        if config.baseline.max_cvrmse_pct > ashrae_criteria.get("max_cvrmse_pct", 100):
            warnings.append(
                f"CVRMSE threshold ({config.baseline.max_cvrmse_pct}%) exceeds "
                f"ASHRAE 14 {granularity} limit ({ashrae_criteria['max_cvrmse_pct']}%)."
            )
        if config.baseline.max_nmbe_pct > ashrae_criteria.get("max_nmbe_pct", 100):
            warnings.append(
                f"NMBE threshold ({config.baseline.max_nmbe_pct}%) exceeds "
                f"ASHRAE 14 {granularity} limit ({ashrae_criteria['max_nmbe_pct']}%)."
            )

    # Check Option C ECM significance
    if config.ipmvp.default_option == IPMVPOption.OPTION_C:
        if config.ipmvp.option_c_min_ecm_share_pct < 10:
            warnings.append(
                f"Option C minimum ECM share ({config.ipmvp.option_c_min_ecm_share_pct}%) "
                "is below 10%. ECMs affecting less than 10% of whole-facility energy "
                "may not produce statistically significant savings with Option C."
            )

    # Check Option A sampling configuration
    if config.ipmvp.default_option == IPMVPOption.OPTION_A:
        if not config.metering.sampling_enabled:
            warnings.append(
                "Option A selected but sampling is disabled. Option A key parameter "
                "measurement often requires sampling protocols for large populations."
            )

    # Check Option B metering duration
    if config.ipmvp.default_option == IPMVPOption.OPTION_B:
        if config.ipmvp.option_b_metering_duration_weeks < 2:
            warnings.append(
                f"Option B metering duration ({config.ipmvp.option_b_metering_duration_weeks} weeks) "
                "is very short. Minimum 2-4 weeks recommended for representative measurement."
            )

    # Check uncertainty configuration
    if config.uncertainty.enabled:
        if config.uncertainty.max_fractional_savings_uncertainty_pct > 50:
            warnings.append(
                f"FSU threshold ({config.uncertainty.max_fractional_savings_uncertainty_pct}%) "
                "exceeds ASHRAE 14 recommendation of 50% at 68% confidence."
            )

    # Check persistence tracking for multi-year projects
    if config.contract_term_years and config.contract_term_years > 1:
        if not config.persistence.enabled:
            warnings.append(
                f"Multi-year project ({config.contract_term_years} years) but persistence "
                "tracking is disabled. Enable persistence tracking for multi-year M&V."
            )

    # Check FEMP compliance requirements
    if ComplianceFramework.FEMP in config.ipmvp.compliance_frameworks:
        if config.audit_trail.retention_years < 7:
            warnings.append(
                f"FEMP compliance requires 7-year audit trail retention. "
                f"Current retention: {config.audit_trail.retention_years} years."
            )

    # Check weather normalization for weather-dependent facilities
    if config.facility_type in (
        FacilityType.COMMERCIAL_OFFICE, FacilityType.HOSPITAL,
        FacilityType.UNIVERSITY_CAMPUS, FacilityType.GOVERNMENT_FEMP,
    ):
        if not config.adjustments.weather_normalization:
            warnings.append(
                f"Weather normalization disabled for {config.facility_type.value}. "
                "Weather-dependent facilities require HDD/CDD normalization for accurate M&V."
            )

    # Check production normalization for manufacturing
    if config.facility_type == FacilityType.MANUFACTURING:
        if not config.adjustments.production_normalization:
            warnings.append(
                "Manufacturing facility: production normalization is disabled. "
                "Production-dependent energy use requires production adjustment."
            )

    # Check energy rate data
    if config.savings.carbon_savings_tracking:
        if config.savings.electricity_ef_kgco2_per_kwh <= 0:
            warnings.append(
                "Carbon savings tracking enabled but electricity emission factor is zero. "
                "Verify emission factor configuration."
            )

    # Check savings rate data
    if config.savings.electricity_rate_per_kwh <= 0:
        warnings.append(
            "Electricity rate is zero or negative. "
            "Cost savings calculations require valid energy rates."
        )

    # Check metering data quality
    if config.metering.min_data_completeness_pct < 80:
        warnings.append(
            f"Meter data completeness threshold ({config.metering.min_data_completeness_pct}%) "
            "is below 80%. M&V accuracy may be reduced."
        )

    # Check floor area for area-normalised savings
    if config.floor_area_m2 is None and config.facility_type in (
        FacilityType.COMMERCIAL_OFFICE, FacilityType.HOSPITAL,
        FacilityType.UNIVERSITY_CAMPUS, FacilityType.GOVERNMENT_FEMP,
    ):
        warnings.append(
            "No floor_area_m2 configured for a building-type facility. "
            "Floor area is needed for EUI-normalised savings reporting."
        )

    return warnings


def get_default_config(
    facility_type: FacilityType = FacilityType.COMMERCIAL_OFFICE,
) -> MVConfig:
    """Get default configuration for a given facility type.

    Args:
        facility_type: Facility type to configure for.

    Returns:
        MVConfig instance with facility-appropriate defaults.
    """
    return MVConfig(facility_type=facility_type)
