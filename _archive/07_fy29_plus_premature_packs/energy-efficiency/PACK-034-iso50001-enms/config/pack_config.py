"""
PACK-034 ISO 50001 Energy Management System Pack - Configuration Manager

This module implements the EnMSConfig and PackConfig classes that load,
merge, and validate all configuration for the ISO 50001 EnMS Pack.
It provides comprehensive Pydantic v2 models for implementing, operating,
and maintaining an ISO 50001:2018-compliant Energy Management System (EnMS)
including Significant Energy Use (SEU) identification, energy baseline
development, Energy Performance Indicator (EnPI) tracking, CUSUM analysis,
degree-day normalisation, compliance gap assessment, monitoring plans,
and action plan management.

Facility Types:
    - MANUFACTURING: Heavy/discrete/batch manufacturing (process heat, motors, compressed air)
    - COMMERCIAL_OFFICE: Commercial offices (HVAC, lighting, plug loads, IT)
    - DATA_CENTER: Data centres (cooling, UPS, IT load, PUE tracking)
    - HEALTHCARE: Hospitals and clinics (24/7 HVAC, sterilisation, medical equipment)
    - RETAIL: Retail chains (lighting, HVAC, refrigeration, multi-site)
    - LOGISTICS: Logistics and warehousing (lighting, dock ops, MHE, HVAC)
    - FOOD_PROCESSING: Food/beverage processing (process heat, refrigeration, CIP)
    - SME_MULTI_SITE: SME portfolios with 2-5 sites (simplified scope, budget-conscious)
    - EDUCATION: Schools and universities (HVAC, lighting, behavioural, scheduling)
    - HOTEL: Hotels and hospitality (HVAC, laundry, kitchen, lighting, pool)

EnMS Maturity Levels:
    - PLANNING: Organisation is planning EnMS implementation (gap analysis phase)
    - IMPLEMENTING: EnMS being implemented (policy, team, SEU, baseline, targets)
    - OPERATIONAL: EnMS operational but not yet certified (internal audit, review)
    - CERTIFIED: EnMS certified to ISO 50001:2018 (surveillance audits ongoing)
    - RECERTIFYING: EnMS due for recertification (3-year cycle renewal)

Baseline Model Preferences:
    - AUTO: Automatically select best-fit model based on data characteristics
    - SIMPLE_MEAN: Simple mean baseline (no relevant variables identified)
    - SINGLE_VARIABLE: Single-variable linear regression (e.g., HDD or production)
    - MULTI_VARIABLE: Multi-variable regression (e.g., HDD + CDD + production)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (manufacturing_facility / commercial_office / data_center /
       healthcare_facility / retail_chain / logistics_warehouse / food_processing /
       sme_multi_site)
    3. Environment overrides (ENMS_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - ISO 50001:2018: Energy Management Systems - Requirements with guidance
    - ISO 50006:2014: Measuring energy performance using energy baselines and EnPIs
    - ISO 50015:2014: Measurement and verification of energy performance
    - EU EED: Directive (EU) 2023/1791 (Energy Efficiency Directive - recast)
    - IPMVP: International Performance Measurement and Verification Protocol
    - ASHRAE 14: Measurement of Energy, Demand, and Water Savings

Example:
    >>> config = PackConfig.from_preset("manufacturing_facility")
    >>> print(config.pack.facility_type)
    FacilityType.MANUFACTURING
    >>> print(config.pack.enms_maturity)
    EnMSMaturity.PLANNING
    >>> print(config.pack.baseline.model_preference)
    BaselineModelPreference.AUTO
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - ISO 50001 EnMS enumeration types
# =============================================================================


class FacilityType(str, Enum):
    """Facility type classification for EnMS scoping."""

    MANUFACTURING = "MANUFACTURING"
    COMMERCIAL_OFFICE = "COMMERCIAL_OFFICE"
    DATA_CENTER = "DATA_CENTER"
    HEALTHCARE = "HEALTHCARE"
    RETAIL = "RETAIL"
    LOGISTICS = "LOGISTICS"
    FOOD_PROCESSING = "FOOD_PROCESSING"
    SME_MULTI_SITE = "SME_MULTI_SITE"
    EDUCATION = "EDUCATION"
    HOTEL = "HOTEL"


class EnMSMaturity(str, Enum):
    """EnMS implementation maturity level per ISO 50001 lifecycle."""

    PLANNING = "PLANNING"
    IMPLEMENTING = "IMPLEMENTING"
    OPERATIONAL = "OPERATIONAL"
    CERTIFIED = "CERTIFIED"
    RECERTIFYING = "RECERTIFYING"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency for energy performance tracking."""

    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"


class OutputFormat(str, Enum):
    """Output format for EnMS reports and documentation."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    PDF = "PDF"
    EXCEL = "EXCEL"


class BaselineModelPreference(str, Enum):
    """Preferred energy baseline regression model type per ISO 50006."""

    AUTO = "AUTO"
    SIMPLE_MEAN = "SIMPLE_MEAN"
    SINGLE_VARIABLE = "SINGLE_VARIABLE"
    MULTI_VARIABLE = "MULTI_VARIABLE"


class MonitoringGranularity(str, Enum):
    """Monitoring data granularity for energy performance tracking."""

    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class OutlierMethod(str, Enum):
    """Outlier detection method for baseline data cleaning."""

    IQR = "IQR"
    ZSCORE = "ZSCORE"


class CUSUMMethod(str, Enum):
    """CUSUM analysis method for energy performance monitoring."""

    TABULAR = "TABULAR"
    V_MASK = "V_MASK"


class SeasonalAdjustment(str, Enum):
    """Seasonal adjustment method for CUSUM analysis."""

    NONE = "NONE"
    DEGREE_DAY = "DEGREE_DAY"
    MONTHLY_FACTOR = "MONTHLY_FACTOR"


class NormalizationBasis(str, Enum):
    """Weather normalisation basis for degree-day calculations."""

    TMY3 = "TMY3"
    LONG_TERM_AVERAGE = "LONG_TERM_AVERAGE"
    REPORTING_PERIOD = "REPORTING_PERIOD"


class NormalizationMethod(str, Enum):
    """EnPI normalisation method for comparing performance across periods."""

    PRODUCTION = "PRODUCTION"
    AREA = "AREA"
    DEGREE_DAY = "DEGREE_DAY"
    OCCUPANCY = "OCCUPANCY"
    COMPOSITE = "COMPOSITE"


class VerificationMethod(str, Enum):
    """M&V verification method per IPMVP options."""

    OPTION_A = "OPTION_A"
    OPTION_B = "OPTION_B"
    OPTION_C = "OPTION_C"
    OPTION_D = "OPTION_D"


class EnPIType(str, Enum):
    """Types of Energy Performance Indicators per ISO 50006."""

    ABSOLUTE = "ABSOLUTE"
    INTENSITY = "INTENSITY"
    REGRESSION_BASED = "REGRESSION_BASED"
    STATISTICAL = "STATISTICAL"
    CUSUM = "CUSUM"


class SEUCategory(str, Enum):
    """Significant Energy Use categories for ISO 50001 SEU identification."""

    HVAC = "HVAC"
    LIGHTING = "LIGHTING"
    PROCESS_HEAT = "PROCESS_HEAT"
    COMPRESSED_AIR = "COMPRESSED_AIR"
    MOTORS_DRIVES = "MOTORS_DRIVES"
    REFRIGERATION = "REFRIGERATION"
    IT_EQUIPMENT = "IT_EQUIPMENT"
    WATER_HEATING = "WATER_HEATING"
    STEAM = "STEAM"
    COOLING = "COOLING"
    TRANSPORT = "TRANSPORT"
    BUILDING_ENVELOPE = "BUILDING_ENVELOPE"
    COOKING_KITCHEN = "COOKING_KITCHEN"
    LAUNDRY = "LAUNDRY"
    POOL_SPA = "POOL_SPA"


class ComplianceStatus(str, Enum):
    """ISO 50001 compliance requirement status."""

    CONFORMING = "CONFORMING"
    NON_CONFORMING = "NON_CONFORMING"
    OPPORTUNITY_FOR_IMPROVEMENT = "OPPORTUNITY_FOR_IMPROVEMENT"
    NOT_ASSESSED = "NOT_ASSESSED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ActionPlanStatus(str, Enum):
    """Status of energy management action plan items."""

    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    VERIFIED = "VERIFIED"
    ON_HOLD = "ON_HOLD"
    CANCELLED = "CANCELLED"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Facility type display information and typical characteristics
FACILITY_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_eui_kwh_per_m2": "200-1000",
        "typical_energy_split": "Electricity 50-70%, Gas 25-45%",
        "typical_seus": [
            "Process heat (boilers, furnaces, dryers)",
            "Compressed air systems",
            "Motor-driven systems (pumps, fans, conveyors)",
            "Lighting (high-bay, task, exterior)",
            "HVAC (process and comfort)",
        ],
        "iso50001_focus": "SEU-driven with production-normalised EnPIs",
        "typical_enpi": "kWh per unit of production or kWh per tonne",
        "certification_timeline_months": "12-18",
    },
    "COMMERCIAL_OFFICE": {
        "name": "Commercial Office Building",
        "typical_eui_kwh_per_m2": "150-300",
        "typical_energy_split": "Electricity 70-85%, Gas 15-30%",
        "typical_seus": [
            "HVAC (heating, cooling, ventilation)",
            "Lighting (general, task, exterior)",
            "IT equipment and data rooms",
            "Plug loads (kitchen, vending, desk equipment)",
            "Lifts and escalators",
        ],
        "iso50001_focus": "Occupancy and weather-normalised EnPIs",
        "typical_enpi": "kWh per m2 per year or kWh per occupant",
        "certification_timeline_months": "9-14",
    },
    "DATA_CENTER": {
        "name": "Data Centre / Server Room",
        "typical_eui_kwh_per_m2": "1000-5000",
        "typical_energy_split": "Electricity 95-100%",
        "typical_seus": [
            "Cooling systems (CRAC, chillers, free cooling)",
            "IT load (servers, storage, networking)",
            "UPS and power distribution",
            "Lighting and security",
            "Building HVAC (offices and control rooms)",
        ],
        "iso50001_focus": "PUE-based EnPIs with IT load normalisation",
        "typical_enpi": "PUE (Power Usage Effectiveness) and kWh per IT kW",
        "certification_timeline_months": "10-15",
    },
    "HEALTHCARE": {
        "name": "Hospital / Healthcare Facility",
        "typical_eui_kwh_per_m2": "300-700",
        "typical_energy_split": "Electricity 50-65%, Gas 30-45%",
        "typical_seus": [
            "HVAC (24/7 operation, clean rooms, theatres)",
            "Sterilisation and autoclaves",
            "Hot water systems (high demand)",
            "Medical equipment",
            "Lighting (wards, theatres, corridors, car parks)",
        ],
        "iso50001_focus": "Bed-day or patient-normalised EnPIs, 24/7 baseline",
        "typical_enpi": "kWh per bed-day or kWh per patient",
        "certification_timeline_months": "14-20",
    },
    "RETAIL": {
        "name": "Retail Chain / Store",
        "typical_eui_kwh_per_m2": "200-500",
        "typical_energy_split": "Electricity 80-95%, Gas 5-20%",
        "typical_seus": [
            "Lighting (display, accent, exterior signage)",
            "HVAC (comfort, aligned to trading hours)",
            "Refrigeration (display cases, cold rooms)",
            "IT and POS systems",
            "Water heating",
        ],
        "iso50001_focus": "Multi-site EnMS with store-level EnPIs",
        "typical_enpi": "kWh per m2 sales area or kWh per transaction",
        "certification_timeline_months": "12-18",
    },
    "LOGISTICS": {
        "name": "Logistics / Warehouse Facility",
        "typical_eui_kwh_per_m2": "50-200",
        "typical_energy_split": "Electricity 60-80%, Gas 15-35%",
        "typical_seus": [
            "Lighting (high-bay, loading docks, exterior)",
            "HVAC (destratification, dock seals, office areas)",
            "Material handling equipment (MHE) charging",
            "Dock door and conveyor systems",
            "Refrigeration (cold chain operations)",
        ],
        "iso50001_focus": "Throughput-normalised EnPIs, shift patterns",
        "typical_enpi": "kWh per pallet moved or kWh per m3 storage",
        "certification_timeline_months": "10-14",
    },
    "FOOD_PROCESSING": {
        "name": "Food & Beverage Processing Plant",
        "typical_eui_kwh_per_m2": "300-1500",
        "typical_energy_split": "Electricity 40-60%, Gas 35-55%",
        "typical_seus": [
            "Process heat (boilers, CHP, direct firing)",
            "Refrigeration (blast chillers, cold storage, freezing)",
            "Compressed air (packaging, pneumatic controls)",
            "Cleaning-in-place (CIP) and hot water",
            "Motors and drives (mixing, conveying, pumping)",
        ],
        "iso50001_focus": "Production-normalised EnPIs, seasonal product mix",
        "typical_enpi": "kWh per tonne of product or kWh per litre",
        "certification_timeline_months": "12-18",
    },
    "SME_MULTI_SITE": {
        "name": "SME Multi-Site Portfolio",
        "typical_eui_kwh_per_m2": "100-400",
        "typical_energy_split": "Electricity 60-80%, Gas 20-40%",
        "typical_seus": [
            "HVAC (heating, cooling per site)",
            "Lighting (general, exterior)",
            "IT and office equipment",
            "Process loads (varies by SME type)",
            "Water heating",
        ],
        "iso50001_focus": "Simplified EnMS with portfolio-level reporting",
        "typical_enpi": "kWh per m2 per year (portfolio average)",
        "certification_timeline_months": "8-12",
    },
    "EDUCATION": {
        "name": "School / University Campus",
        "typical_eui_kwh_per_m2": "100-350",
        "typical_energy_split": "Electricity 45-65%, Gas 30-50%",
        "typical_seus": [
            "HVAC (heating aligned to timetable, cooling in labs)",
            "Lighting (classrooms, corridors, sports halls)",
            "IT equipment (labs, lecture theatres, offices)",
            "Catering and kitchen equipment",
            "Sports and leisure facilities",
        ],
        "iso50001_focus": "Occupancy-normalised EnPIs, term-time vs holiday",
        "typical_enpi": "kWh per student FTE or kWh per m2",
        "certification_timeline_months": "10-14",
    },
    "HOTEL": {
        "name": "Hotel / Hospitality Venue",
        "typical_eui_kwh_per_m2": "200-500",
        "typical_energy_split": "Electricity 55-70%, Gas 25-40%",
        "typical_seus": [
            "HVAC (guest rooms, lobbies, conference areas)",
            "Hot water (guest rooms, kitchen, laundry)",
            "Laundry operations",
            "Kitchen and food service equipment",
            "Lighting (guest rooms, public areas, exterior)",
            "Pool and spa heating",
        ],
        "iso50001_focus": "Guest-night normalised EnPIs, occupancy-driven",
        "typical_enpi": "kWh per guest-night or kWh per occupied room",
        "certification_timeline_months": "12-16",
    },
}

# Available preset configurations
AVAILABLE_PRESETS: Dict[str, str] = {
    "manufacturing_facility": (
        "Heavy manufacturing with process heat, compressed air, motors, "
        "and production-normalised EnPIs"
    ),
    "commercial_office": (
        "HVAC-dominated commercial office with lighting, plug loads, "
        "and occupancy-driven EnPIs"
    ),
    "data_center": (
        "Cooling-dominated data centre with PUE focus, UPS efficiency, "
        "and IT load normalisation"
    ),
    "healthcare_facility": (
        "24/7 healthcare facility with HVAC, sterilisation, medical equipment, "
        "and bed-day normalised EnPIs"
    ),
    "retail_chain": (
        "Multi-site retail chain with lighting, HVAC, refrigeration, "
        "and store-level EnPIs"
    ),
    "logistics_warehouse": (
        "Logistics warehouse with high-bay lighting, dock operations, "
        "MHE charging, and throughput EnPIs"
    ),
    "food_processing": (
        "Food processing plant with process heat, refrigeration, compressed air, "
        "CIP, and production-normalised EnPIs"
    ),
    "sme_multi_site": (
        "Simplified SME scope across 2-5 sites with budget-conscious implementation "
        "and portfolio-level reporting"
    ),
}

# Default EnMS parameters
DEFAULT_ENMS_PARAMS: Dict[str, Any] = {
    "seu_cumulative_threshold_pct": 80.0,
    "seu_individual_threshold_pct": 5.0,
    "baseline_min_data_months": 12,
    "baseline_max_cv_rmse": 25.0,
    "baseline_min_r_squared": 0.75,
    "enpi_target_improvement_pct": 3.0,
    "cusum_alert_threshold_sigma": 2.0,
    "hdd_base_celsius": 15.5,
    "cdd_base_celsius": 18.3,
    "compliance_nc_closure_days": 90,
    "monitoring_data_retention_months": 60,
    "action_plan_max_payback_years": 5.0,
    "audit_trail_retention_days": 2555,
}

# ISO 50001 clause reference mapping
ISO_50001_CLAUSES: Dict[str, str] = {
    "4.1": "Understanding the organisation and its context",
    "4.2": "Understanding the needs and expectations of interested parties",
    "4.3": "Determining the scope of the EnMS",
    "4.4": "Energy management system",
    "5.1": "Leadership and commitment",
    "5.2": "Energy policy",
    "5.3": "Organisational roles, responsibilities and authorities",
    "6.1": "Actions to address risks and opportunities",
    "6.2": "Objectives, energy targets and planning to achieve them",
    "6.3": "Energy review",
    "6.4": "Energy performance indicators",
    "6.5": "Energy baseline",
    "6.6": "Planning for collection of energy data",
    "7.1": "Resources",
    "7.2": "Competence",
    "7.3": "Awareness",
    "7.4": "Communication",
    "7.5": "Documented information",
    "8.1": "Operational planning and control",
    "8.2": "Design",
    "8.3": "Procurement",
    "9.1": "Monitoring, measurement, analysis and evaluation of EnP and EnMS",
    "9.2": "Internal audit",
    "9.3": "Management review",
    "10.1": "Nonconformity and corrective action",
    "10.2": "Continual improvement",
}

# Regional energy tariff defaults
DEFAULT_ENERGY_TARIFFS: Dict[str, Dict[str, float]] = {
    "EU_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.22,
        "gas_rate_eur_per_kwh": 0.08,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "DE": {
        "electricity_rate_eur_per_kwh": 0.30,
        "gas_rate_eur_per_kwh": 0.10,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "UK": {
        "electricity_rate_eur_per_kwh": 0.28,
        "gas_rate_eur_per_kwh": 0.07,
        "carbon_price_eur_per_tco2": 55.0,
    },
    "FR": {
        "electricity_rate_eur_per_kwh": 0.20,
        "gas_rate_eur_per_kwh": 0.08,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "US": {
        "electricity_rate_eur_per_kwh": 0.12,
        "gas_rate_eur_per_kwh": 0.04,
        "carbon_price_eur_per_tco2": 30.0,
    },
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class SEUConfig(BaseModel):
    """Configuration for Significant Energy Use (SEU) identification.

    Per ISO 50001 Clause 6.3, the organisation shall identify SEUs by
    analysing energy use and consumption data to determine which energy
    uses account for substantial consumption and offer considerable
    potential for improvement. This configuration controls the thresholds
    and categories used for SEU identification.
    """

    cumulative_threshold_pct: float = Field(
        80.0,
        ge=50.0,
        le=100.0,
        description=(
            "Cumulative energy consumption threshold (%) for SEU identification. "
            "Energy uses are ranked by consumption and SEUs are identified until "
            "this cumulative percentage is reached (Pareto analysis)."
        ),
    )
    individual_threshold_pct: float = Field(
        5.0,
        ge=1.0,
        le=50.0,
        description=(
            "Individual energy use threshold (%). Any single energy use "
            "accounting for more than this percentage of total consumption "
            "is automatically classified as an SEU."
        ),
    )
    min_consumers: int = Field(
        3,
        ge=1,
        le=20,
        description="Minimum number of SEUs to identify regardless of threshold",
    )
    include_categories: List[SEUCategory] = Field(
        default_factory=lambda: [
            SEUCategory.HVAC,
            SEUCategory.LIGHTING,
            SEUCategory.PROCESS_HEAT,
            SEUCategory.COMPRESSED_AIR,
            SEUCategory.MOTORS_DRIVES,
            SEUCategory.REFRIGERATION,
        ],
        description="SEU categories to evaluate during energy review",
    )
    review_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="Frequency of SEU re-evaluation and energy review",
    )
    include_relevant_variables: bool = Field(
        True,
        description=(
            "Identify relevant variables (production, weather, occupancy) "
            "affecting each SEU per ISO 50001 Clause 6.3"
        ),
    )
    track_personnel: bool = Field(
        True,
        description=(
            "Track personnel whose work affects SEUs per ISO 50001 "
            "Clause 6.3 requirement for competence and awareness"
        ),
    )

    @field_validator("include_categories")
    @classmethod
    def validate_categories_not_empty(
        cls, v: List[SEUCategory]
    ) -> List[SEUCategory]:
        """At least one SEU category must be included."""
        if not v:
            raise ValueError(
                "At least one SEU category must be included for energy review."
            )
        return v


class BaselineConfig(BaseModel):
    """Configuration for energy baseline development per ISO 50006.

    The energy baseline is the quantitative reference providing a basis
    for comparison of energy performance. This configuration controls
    the regression modelling approach, statistical validity criteria,
    and data quality requirements for baseline development.
    """

    model_preference: BaselineModelPreference = Field(
        BaselineModelPreference.AUTO,
        description=(
            "Preferred baseline regression model type. AUTO selects "
            "the best-fit model based on data characteristics and "
            "statistical significance tests."
        ),
    )
    min_data_months: int = Field(
        12,
        ge=6,
        le=60,
        description=(
            "Minimum months of data required for baseline development. "
            "ISO 50006 recommends at least 12 months to capture seasonal "
            "variation."
        ),
    )
    max_cv_rmse: float = Field(
        25.0,
        ge=5.0,
        le=50.0,
        description=(
            "Maximum coefficient of variation of RMSE (%) for baseline "
            "model acceptance. ASHRAE Guideline 14 recommends CV(RMSE) "
            "<= 25% for monthly data, <= 30% for daily data."
        ),
    )
    min_r_squared: float = Field(
        0.75,
        ge=0.50,
        le=0.99,
        description=(
            "Minimum coefficient of determination (R-squared) for baseline "
            "model acceptance. Values >= 0.75 indicate acceptable fit."
        ),
    )
    significance_level: float = Field(
        0.05,
        ge=0.001,
        le=0.20,
        description=(
            "Statistical significance level (alpha) for regression "
            "coefficient t-tests and F-test model significance."
        ),
    )
    outlier_method: OutlierMethod = Field(
        OutlierMethod.IQR,
        description=(
            "Outlier detection method for baseline data cleaning. "
            "IQR: interquartile range method. ZSCORE: z-score method."
        ),
    )
    outlier_threshold: float = Field(
        1.5,
        ge=1.0,
        le=4.0,
        description=(
            "Outlier threshold multiplier. For IQR: multiplier of IQR "
            "beyond Q1/Q3. For ZSCORE: number of standard deviations."
        ),
    )
    auto_adjust: bool = Field(
        True,
        description=(
            "Automatically adjust baseline for non-routine events "
            "(facility changes, equipment additions/removals) and "
            "static factor changes per ISO 50006."
        ),
    )
    include_confidence_intervals: bool = Field(
        True,
        description=(
            "Calculate and report confidence intervals for baseline "
            "predictions using t-distribution."
        ),
    )
    normalisation_variables: List[str] = Field(
        default_factory=lambda: ["hdd", "cdd", "production"],
        description=(
            "Default relevant variables to test for baseline normalisation. "
            "Common variables: hdd, cdd, production, occupancy, shift_hours."
        ),
    )


class EnPIConfig(BaseModel):
    """Configuration for Energy Performance Indicators per ISO 50006.

    EnPIs are quantitative values or measures of energy performance
    defined by the organisation. This configuration controls the types
    of EnPIs calculated, normalisation methods, target improvement
    rates, and statistical testing levels.
    """

    enpi_types: List[EnPIType] = Field(
        default_factory=lambda: [
            EnPIType.ABSOLUTE,
            EnPIType.INTENSITY,
            EnPIType.REGRESSION_BASED,
        ],
        description=(
            "Types of EnPIs to calculate. ABSOLUTE: total consumption. "
            "INTENSITY: consumption per normalisation factor. "
            "REGRESSION_BASED: model-predicted vs actual comparison."
        ),
    )
    normalization_method: NormalizationMethod = Field(
        NormalizationMethod.PRODUCTION,
        description=(
            "Primary normalisation method for intensity EnPIs. "
            "PRODUCTION: per unit of output. AREA: per m2. "
            "DEGREE_DAY: weather normalised. OCCUPANCY: per occupant."
        ),
    )
    target_improvement_pct: float = Field(
        3.0,
        ge=0.5,
        le=20.0,
        description=(
            "Annual target improvement percentage for energy performance. "
            "Typical range: 2-5% for mature organisations, 5-10% for "
            "initial EnMS implementation."
        ),
    )
    statistical_test_level: float = Field(
        0.05,
        ge=0.001,
        le=0.20,
        description=(
            "Statistical significance level for EnPI trend analysis "
            "and performance change detection."
        ),
    )
    track_by_seu: bool = Field(
        True,
        description=(
            "Track EnPIs for each identified SEU in addition to "
            "facility-level EnPIs per ISO 50006 recommendations."
        ),
    )
    rolling_period_months: int = Field(
        12,
        ge=3,
        le=36,
        description=(
            "Rolling period for EnPI trend calculation (months). "
            "A 12-month rolling window smooths seasonal variation."
        ),
    )
    benchmark_enabled: bool = Field(
        True,
        description=(
            "Enable benchmarking of EnPIs against sector averages "
            "and best-in-class values from published data."
        ),
    )

    @field_validator("enpi_types")
    @classmethod
    def validate_enpi_types_not_empty(
        cls, v: List[EnPIType]
    ) -> List[EnPIType]:
        """At least one EnPI type must be selected."""
        if not v:
            raise ValueError("At least one EnPI type must be selected.")
        return v


class CUSUMConfig(BaseModel):
    """Configuration for CUSUM (Cumulative Sum) analysis.

    CUSUM analysis tracks the cumulative difference between actual and
    baseline-predicted energy consumption to detect shifts in energy
    performance over time. A downward trend indicates savings; an
    upward trend indicates increased consumption.
    """

    method: CUSUMMethod = Field(
        CUSUMMethod.TABULAR,
        description=(
            "CUSUM analysis method. TABULAR: standard tabular CUSUM "
            "with threshold alerts. V_MASK: V-mask overlay for visual "
            "detection of performance shifts."
        ),
    )
    alert_threshold: float = Field(
        2.0,
        ge=0.5,
        le=5.0,
        description=(
            "Alert threshold in standard deviations (sigma). Alerts "
            "trigger when cumulative sum exceeds this threshold, "
            "indicating a statistically significant performance shift."
        ),
    )
    monitoring_interval: MonitoringGranularity = Field(
        MonitoringGranularity.MONTHLY,
        description="Interval for CUSUM monitoring data points",
    )
    seasonal_adjustment: SeasonalAdjustment = Field(
        SeasonalAdjustment.NONE,
        description=(
            "Seasonal adjustment applied before CUSUM analysis. "
            "DEGREE_DAY: adjust using HDD/CDD regression. "
            "MONTHLY_FACTOR: adjust using monthly seasonal indices."
        ),
    )
    reset_on_baseline_change: bool = Field(
        True,
        description="Reset CUSUM accumulation when baseline is adjusted",
    )
    generate_charts: bool = Field(
        True,
        description="Generate CUSUM charts with trend lines and alert markers",
    )


class DegreeDayConfig(BaseModel):
    """Configuration for degree-day weather normalisation.

    Degree-day normalisation adjusts energy consumption for weather
    effects, enabling fair comparison across reporting periods with
    different weather conditions. Supports heating degree-days (HDD)
    and cooling degree-days (CDD) per variable-base methodology.
    """

    heating_base_celsius: float = Field(
        15.5,
        ge=10.0,
        le=22.0,
        description=(
            "Heating base temperature (Celsius). Heating degree-days "
            "accumulate when outdoor temperature falls below this value. "
            "Standard values: 15.5C (UK/EU), 18.3C (US ASHRAE)."
        ),
    )
    cooling_base_celsius: float = Field(
        18.3,
        ge=15.0,
        le=30.0,
        description=(
            "Cooling base temperature (Celsius). Cooling degree-days "
            "accumulate when outdoor temperature rises above this value."
        ),
    )
    optimize_base_temp: bool = Field(
        True,
        description=(
            "Automatically optimise base temperatures by testing a range "
            "of values and selecting the temperature that maximises the "
            "regression R-squared value."
        ),
    )
    normalization_basis: NormalizationBasis = Field(
        NormalizationBasis.TMY3,
        description=(
            "Reference weather data for normalisation. TMY3: Typical "
            "Meteorological Year data. LONG_TERM_AVERAGE: 20-year average. "
            "REPORTING_PERIOD: actual weather in reporting period."
        ),
    )
    weather_station_id: Optional[str] = Field(
        None,
        description=(
            "Weather station identifier for degree-day data retrieval. "
            "If None, the nearest station is selected automatically."
        ),
    )
    include_solar_gain: bool = Field(
        False,
        description=(
            "Include solar radiation data in weather model. "
            "Relevant for buildings with significant glazing area."
        ),
    )


class ComplianceConfig(BaseModel):
    """Configuration for ISO 50001 compliance assessment and gap analysis.

    Manages the frequency of compliance checks, automatic gap analysis
    generation, nonconformity closure timelines, and surveillance
    audit scheduling reminders.
    """

    assessment_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description=(
            "Frequency of compliance assessment against ISO 50001 "
            "clauses. Annual assessment recommended as minimum; "
            "quarterly for organisations in implementation phase."
        ),
    )
    auto_gap_analysis: bool = Field(
        True,
        description=(
            "Automatically generate gap analysis report mapping current "
            "EnMS status against all ISO 50001:2018 clauses."
        ),
    )
    nc_closure_days: int = Field(
        90,
        ge=30,
        le=365,
        description=(
            "Maximum days allowed for nonconformity closure. "
            "ISO 50001 requires corrective action without undue delay."
        ),
    )
    surveillance_reminder_days: int = Field(
        30,
        ge=7,
        le=90,
        description=(
            "Days before surveillance audit to send preparation reminder. "
            "Surveillance audits occur annually after initial certification."
        ),
    )
    track_legal_requirements: bool = Field(
        True,
        description=(
            "Track applicable legal and other requirements related to "
            "energy use per ISO 50001 Clause 4.2."
        ),
    )
    internal_audit_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description=(
            "Frequency of internal EnMS audits per ISO 50001 Clause 9.2."
        ),
    )
    management_review_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description=(
            "Frequency of management reviews per ISO 50001 Clause 9.3."
        ),
    )
    clauses_to_assess: List[str] = Field(
        default_factory=lambda: sorted(ISO_50001_CLAUSES.keys()),
        description=(
            "ISO 50001 clauses to include in compliance assessment. "
            "Default: all clauses (4.1 through 10.2)."
        ),
    )


class MonitoringConfig(BaseModel):
    """Configuration for energy monitoring, measurement, and data collection.

    Per ISO 50001 Clause 6.6, the organisation shall define an energy
    data collection plan. This configuration controls the granularity,
    retention, quality thresholds, and alerting for energy data.
    """

    granularity: MonitoringGranularity = Field(
        MonitoringGranularity.MONTHLY,
        description=(
            "Default monitoring data granularity. MONTHLY for billing "
            "data, DAILY for sub-metered data, HOURLY for real-time "
            "monitoring and advanced analysis."
        ),
    )
    data_retention_months: int = Field(
        60,
        ge=12,
        le=120,
        description=(
            "Data retention period in months. 60 months (5 years) "
            "recommended for trend analysis and baseline comparison "
            "across multiple certification cycles."
        ),
    )
    quality_threshold: float = Field(
        0.95,
        ge=0.80,
        le=1.0,
        description=(
            "Data quality threshold (0-1). Minimum proportion of "
            "valid, non-missing data points required for a period "
            "to be included in analysis."
        ),
    )
    auto_alerts: bool = Field(
        True,
        description=(
            "Enable automatic alerts for data quality issues, missing "
            "readings, unusual consumption patterns, and meter faults."
        ),
    )
    meter_hierarchy_enabled: bool = Field(
        True,
        description=(
            "Enable meter hierarchy for sub-meter disaggregation. "
            "Validates that sub-meter totals reconcile with main meters."
        ),
    )
    alert_threshold_pct: float = Field(
        15.0,
        ge=5.0,
        le=50.0,
        description=(
            "Percentage deviation from baseline prediction that "
            "triggers an automatic energy consumption alert."
        ),
    )
    collection_plan_template: bool = Field(
        True,
        description=(
            "Generate energy data collection plan template per "
            "ISO 50001 Clause 6.6 requirements."
        ),
    )


class ActionPlanConfig(BaseModel):
    """Configuration for energy management action plans.

    Per ISO 50001 Clause 6.2, the organisation shall establish action
    plans for achieving energy objectives and targets. This configuration
    controls the financial analysis parameters, verification methods,
    and planning criteria for action items.
    """

    max_payback_years: float = Field(
        5.0,
        ge=0.5,
        le=15.0,
        description=(
            "Maximum simple payback period (years) for action plan "
            "items to be included in recommended actions."
        ),
    )
    discount_rate: float = Field(
        0.08,
        ge=0.0,
        le=0.25,
        description="Discount rate for NPV calculations on action plan items",
    )
    analysis_period_years: int = Field(
        10,
        ge=1,
        le=25,
        description="Analysis period for lifecycle cost assessment (years)",
    )
    verification_method: VerificationMethod = Field(
        VerificationMethod.OPTION_B,
        description=(
            "Default M&V verification method per IPMVP. "
            "OPTION_A: retrofit isolation, key parameter measurement. "
            "OPTION_B: retrofit isolation, all parameter measurement. "
            "OPTION_C: whole facility. OPTION_D: calibrated simulation."
        ),
    )
    energy_price_escalation_pct: float = Field(
        3.0,
        ge=0.0,
        le=15.0,
        description="Annual energy price escalation rate (%) for NPV analysis",
    )
    include_non_energy_benefits: bool = Field(
        True,
        description=(
            "Include non-energy benefits (maintenance, productivity, "
            "comfort, safety) in action plan evaluation."
        ),
    )
    electricity_rate_eur_per_kwh: float = Field(
        0.22,
        ge=0.01,
        le=1.0,
        description="Electricity unit rate (EUR/kWh) for savings calculation",
    )
    gas_rate_eur_per_kwh: float = Field(
        0.08,
        ge=0.01,
        le=0.50,
        description="Natural gas unit rate (EUR/kWh) for savings calculation",
    )
    carbon_price_eur_per_tco2: float = Field(
        85.0,
        ge=0.0,
        le=500.0,
        description="Carbon price (EUR/tCO2e) for monetised carbon benefit",
    )
    currency: str = Field(
        "EUR",
        description="Currency for financial calculations (ISO 4217)",
    )


class ReportingConfig(BaseModel):
    """Configuration for EnMS report generation and documentation."""

    frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description=(
            "Reporting frequency for energy performance updates. "
            "Monthly recommended for operational tracking; quarterly "
            "for management review evidence."
        ),
    )
    output_formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML],
        description="Output formats for EnMS reports",
    )
    include_charts: bool = Field(
        True,
        description=(
            "Include charts and visualisations (EnPI trends, CUSUM, "
            "Sankey diagrams, regression plots) in reports."
        ),
    )
    include_recommendations: bool = Field(
        True,
        description="Include actionable recommendations in reports",
    )
    management_review_template: bool = Field(
        True,
        description=(
            "Generate management review report template per "
            "ISO 50001 Clause 9.3 input requirements."
        ),
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
    )
    include_executive_summary: bool = Field(
        True,
        description="Generate executive summary for senior management",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration for EnMS data."""

    enable_rls: bool = Field(
        True,
        description=(
            "Enable row-level security for multi-tenant and "
            "multi-site data isolation."
        ),
    )
    audit_trail: bool = Field(
        True,
        description="Enable security audit logging for all data access",
    )
    data_encryption: bool = Field(
        True,
        description="Require encryption at rest for stored energy data",
    )
    roles: List[str] = Field(
        default_factory=lambda: [
            "energy_manager",
            "enms_coordinator",
            "facility_manager",
            "sustainability_officer",
            "internal_auditor",
            "top_management",
            "engineer",
            "viewer",
            "admin",
        ],
        description="Available RBAC roles for the pack",
    )
    data_classification: str = Field(
        "INTERNAL",
        description=(
            "Default data classification: PUBLIC, INTERNAL, "
            "CONFIDENTIAL, RESTRICTED"
        ),
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_calculation_time_ms: int = Field(
        30000,
        ge=5000,
        le=120000,
        description="Maximum calculation time per engine (milliseconds)",
    )
    cache_results: bool = Field(
        True,
        description=(
            "Cache intermediate results (emission factors, degree-day "
            "data, regression models) for performance."
        ),
    )
    parallel_processing: bool = Field(
        True,
        description="Enable parallel engine execution where independent",
    )
    max_facilities: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of facilities per analysis run",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=5000,
        description="Batch size for bulk data processing",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for reference data (seconds)",
    )
    memory_ceiling_mb: int = Field(
        4096,
        ge=512,
        le=16384,
        description="Memory ceiling for pack execution (MB)",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for calculation audit trail and provenance tracking."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all calculations and data changes",
    )
    retention_days: int = Field(
        2555,
        ge=365,
        le=3650,
        description=(
            "Audit trail retention period in days. Default 2555 days "
            "(~7 years) covers two ISO 50001 certification cycles."
        ),
    )
    hash_algorithm: str = Field(
        "sha256",
        description="Hash algorithm for provenance tracking",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    calculation_logging: bool = Field(
        True,
        description="Log all intermediate calculation steps",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track all assumptions and default values used",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source to output",
    )
    baseline_change_log: bool = Field(
        True,
        description=(
            "Maintain detailed log of all baseline adjustments "
            "with justification per ISO 50006."
        ),
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class EnMSConfig(BaseModel):
    """Main configuration for PACK-034 ISO 50001 Energy Management System Pack.

    This is the root configuration model that contains all sub-configurations
    for ISO 50001 EnMS implementation, operation, and certification. The
    facility_type and enms_maturity fields drive which features are
    prioritised and which depth of analysis is performed.

    The configuration is structured around the ISO 50001 Plan-Do-Check-Act
    cycle:
        - Plan: SEU identification, baseline, EnPIs, targets, action plans
        - Do: Operational control, monitoring, data collection
        - Check: EnPI evaluation, CUSUM analysis, internal audit, compliance
        - Act: Management review, corrective action, continual improvement
    """

    # Facility identification
    facility_name: str = Field(
        "",
        description="Facility name or site identifier",
    )
    company_name: str = Field(
        "",
        description="Legal entity name of the organisation",
    )
    facility_type: FacilityType = Field(
        FacilityType.MANUFACTURING,
        description="Primary facility type for EnMS scoping",
    )
    enms_maturity: EnMSMaturity = Field(
        EnMSMaturity.PLANNING,
        description=(
            "Current EnMS maturity level. Determines which features "
            "and depth of analysis are prioritised."
        ),
    )
    country: str = Field(
        "DE",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for the EnMS assessment",
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
        description="Annual natural gas consumption (kWh)",
    )
    annual_other_energy_kwh: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Annual other energy consumption in kWh (steam, district "
            "heating, oil, biomass, etc.)"
        ),
    )
    employees: Optional[int] = Field(
        None,
        ge=0,
        description="Number of employees or regular occupants",
    )
    operating_hours_per_year: int = Field(
        4000,
        ge=500,
        le=8760,
        description="Annual operating hours",
    )
    production_units: Optional[str] = Field(
        None,
        description=(
            "Production output unit description (e.g., 'tonnes', 'units', "
            "'litres') for production-normalised EnPIs"
        ),
    )
    annual_production_volume: Optional[float] = Field(
        None,
        ge=0,
        description="Annual production volume in production_units",
    )

    # EnMS scope
    enms_scope: str = Field(
        "",
        description=(
            "Scope of the EnMS (buildings, processes, and energy types "
            "included) per ISO 50001 Clause 4.3."
        ),
    )
    enms_boundary: str = Field(
        "",
        description=(
            "Organisational and physical boundaries of the EnMS "
            "per ISO 50001 Clause 4.3."
        ),
    )

    # Sub-configurations
    seu: SEUConfig = Field(
        default_factory=SEUConfig,
        description="Significant Energy Use identification configuration",
    )
    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig,
        description="Energy baseline development configuration per ISO 50006",
    )
    enpi: EnPIConfig = Field(
        default_factory=EnPIConfig,
        description="Energy Performance Indicator configuration per ISO 50006",
    )
    cusum: CUSUMConfig = Field(
        default_factory=CUSUMConfig,
        description="CUSUM analysis configuration for performance monitoring",
    )
    degree_day: DegreeDayConfig = Field(
        default_factory=DegreeDayConfig,
        description="Degree-day weather normalisation configuration",
    )
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="ISO 50001 compliance assessment configuration",
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Energy monitoring and data collection configuration",
    )
    action_plan: ActionPlanConfig = Field(
        default_factory=ActionPlanConfig,
        description="Energy action plan management configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Report generation configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and access control",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and resource limits",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance tracking",
    )

    @model_validator(mode="after")
    def validate_manufacturing_seus(self) -> "EnMSConfig":
        """Manufacturing facilities should include process-related SEUs."""
        if self.facility_type == FacilityType.MANUFACTURING:
            mfg_seus = {
                SEUCategory.PROCESS_HEAT,
                SEUCategory.COMPRESSED_AIR,
                SEUCategory.MOTORS_DRIVES,
            }
            missing = mfg_seus - set(self.seu.include_categories)
            if missing:
                logger.info(
                    f"Manufacturing facility: adding "
                    f"{[c.value for c in missing]} to SEU categories."
                )
                self.seu.include_categories.extend(missing)
        return self

    @model_validator(mode="after")
    def validate_data_center_seus(self) -> "EnMSConfig":
        """Data centres should include IT and cooling SEU categories."""
        if self.facility_type == FacilityType.DATA_CENTER:
            dc_seus = {SEUCategory.IT_EQUIPMENT, SEUCategory.COOLING}
            missing = dc_seus - set(self.seu.include_categories)
            if missing:
                logger.info(
                    f"Data centre facility: adding "
                    f"{[c.value for c in missing]} to SEU categories."
                )
                self.seu.include_categories.extend(missing)
        return self

    @model_validator(mode="after")
    def validate_food_processing_seus(self) -> "EnMSConfig":
        """Food processing should include refrigeration and process heat."""
        if self.facility_type == FacilityType.FOOD_PROCESSING:
            food_seus = {
                SEUCategory.PROCESS_HEAT,
                SEUCategory.REFRIGERATION,
                SEUCategory.COMPRESSED_AIR,
            }
            missing = food_seus - set(self.seu.include_categories)
            if missing:
                logger.info(
                    f"Food processing facility: adding "
                    f"{[c.value for c in missing]} to SEU categories."
                )
                self.seu.include_categories.extend(missing)
        return self

    @model_validator(mode="after")
    def validate_hotel_seus(self) -> "EnMSConfig":
        """Hotels should include water heating, laundry, and kitchen."""
        if self.facility_type == FacilityType.HOTEL:
            hotel_seus = {
                SEUCategory.WATER_HEATING,
                SEUCategory.LAUNDRY,
                SEUCategory.COOKING_KITCHEN,
            }
            missing = hotel_seus - set(self.seu.include_categories)
            if missing:
                logger.info(
                    f"Hotel facility: adding "
                    f"{[c.value for c in missing]} to SEU categories."
                )
                self.seu.include_categories.extend(missing)
        return self

    @model_validator(mode="after")
    def validate_sme_simplified(self) -> "EnMSConfig":
        """SME multi-site should use simplified monitoring if not set."""
        if self.facility_type == FacilityType.SME_MULTI_SITE:
            if self.monitoring.granularity == MonitoringGranularity.HOURLY:
                logger.info(
                    "SME multi-site: HOURLY monitoring may be excessive. "
                    "Consider MONTHLY or WEEKLY granularity."
                )
        return self

    @model_validator(mode="after")
    def validate_maturity_compliance(self) -> "EnMSConfig":
        """Certified organisations should have annual internal audits."""
        if self.enms_maturity in (
            EnMSMaturity.CERTIFIED,
            EnMSMaturity.RECERTIFYING,
        ):
            if self.compliance.internal_audit_frequency == ReportingFrequency.SEMI_ANNUAL:
                logger.info(
                    "Certified EnMS: semi-annual internal audits configured."
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging. Follows the standard GreenLang pack config
    pattern with from_preset(), from_yaml(), and merge() support.
    """

    pack: EnMSConfig = Field(
        default_factory=EnMSConfig,
        description="Main ISO 50001 EnMS configuration",
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
        "PACK-034-iso50001-enms",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (manufacturing_facility, etc.)
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

        pack_config = EnMSConfig(**preset_data)
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
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}"
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = EnMSConfig(**config_data)
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
        pack_config = EnMSConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with ENMS_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: ENMS_PACK_BASELINE__MODEL_PREFERENCE=MULTI_VARIABLE
                 ENMS_PACK_ACTION_PLAN__DISCOUNT_RATE=0.06
        """
        overrides: Dict[str, Any] = {}
        prefix = "ENMS_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
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
    def _deep_merge(
        base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
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

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as a plain dictionary.

        Returns:
            Dictionary representation of the full configuration.
        """
        return self.model_dump()


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


def validate_config(config: EnMSConfig) -> List[str]:
    """Validate an EnMS configuration and return any warnings.

    Performs comprehensive validation covering facility identification,
    energy data availability, SEU categories, baseline parameters,
    EnPI configuration, and compliance settings.

    Args:
        config: EnMSConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check facility identification
    if not config.facility_name:
        warnings.append(
            "No facility_name configured. Add a facility name for "
            "report identification and ISO 50001 documentation."
        )

    # Check EnMS scope definition
    if not config.enms_scope:
        warnings.append(
            "No enms_scope configured. ISO 50001 Clause 4.3 requires "
            "a defined scope covering buildings, processes, and energy types."
        )

    # Check energy data availability
    if (
        config.annual_electricity_kwh is None
        and config.annual_gas_kwh is None
    ):
        warnings.append(
            "No annual energy consumption configured. SEU analysis and "
            "baseline development require annual_electricity_kwh and/or "
            "annual_gas_kwh."
        )

    # Check floor area for intensity EnPIs
    if config.floor_area_m2 is None:
        if config.enpi.normalization_method == NormalizationMethod.AREA:
            warnings.append(
                "EnPI normalisation set to AREA but no floor_area_m2 "
                "configured. Area-based EnPIs cannot be calculated."
            )

    # Check production data for production-normalised EnPIs
    if config.enpi.normalization_method == NormalizationMethod.PRODUCTION:
        if config.annual_production_volume is None:
            warnings.append(
                "EnPI normalisation set to PRODUCTION but no "
                "annual_production_volume configured. Production-normalised "
                "EnPIs cannot be calculated."
            )
        if not config.production_units:
            warnings.append(
                "EnPI normalisation set to PRODUCTION but no "
                "production_units configured. Add units for clarity."
            )

    # Check SEU category relevance for facility type
    if config.facility_type == FacilityType.DATA_CENTER:
        dc_cats = {SEUCategory.IT_EQUIPMENT, SEUCategory.COOLING}
        missing = dc_cats - set(config.seu.include_categories)
        if missing:
            warnings.append(
                f"Data centre facilities should include "
                f"{[c.value for c in missing]} in SEU categories."
            )

    if config.facility_type == FacilityType.MANUFACTURING:
        mfg_cats = {
            SEUCategory.PROCESS_HEAT,
            SEUCategory.COMPRESSED_AIR,
            SEUCategory.MOTORS_DRIVES,
        }
        missing = mfg_cats - set(config.seu.include_categories)
        if missing:
            warnings.append(
                f"Manufacturing facilities should include "
                f"{[c.value for c in missing]} in SEU categories."
            )

    if config.facility_type == FacilityType.FOOD_PROCESSING:
        food_cats = {SEUCategory.PROCESS_HEAT, SEUCategory.REFRIGERATION}
        missing = food_cats - set(config.seu.include_categories)
        if missing:
            warnings.append(
                f"Food processing facilities should include "
                f"{[c.value for c in missing]} in SEU categories."
            )

    # Check baseline parameters
    if config.baseline.min_data_months < 12:
        warnings.append(
            f"Baseline min_data_months is {config.baseline.min_data_months}. "
            "ISO 50006 recommends at least 12 months for seasonal coverage."
        )

    if config.baseline.max_cv_rmse > 30.0:
        warnings.append(
            f"Baseline max_cv_rmse is {config.baseline.max_cv_rmse}%. "
            "ASHRAE Guideline 14 recommends CV(RMSE) <= 25% for monthly data."
        )

    # Check compliance settings for certified organisations
    if config.enms_maturity in (
        EnMSMaturity.CERTIFIED,
        EnMSMaturity.RECERTIFYING,
    ):
        if config.compliance.nc_closure_days > 120:
            warnings.append(
                f"NC closure period of {config.compliance.nc_closure_days} "
                "days is long for a certified organisation. Certification "
                "bodies typically expect 90 days maximum."
            )

    # Check financial parameters
    if config.action_plan.electricity_rate_eur_per_kwh <= 0:
        warnings.append(
            "Electricity rate is zero or negative. Savings calculations "
            "for action plan items will be invalid."
        )

    # Check monitoring data quality threshold
    if config.monitoring.quality_threshold < 0.90:
        warnings.append(
            f"Data quality threshold is {config.monitoring.quality_threshold}. "
            "Values below 0.90 may yield unreliable EnPI calculations."
        )

    return warnings


def get_default_config(
    facility_type: FacilityType = FacilityType.MANUFACTURING,
) -> EnMSConfig:
    """Get default configuration for a given facility type.

    Args:
        facility_type: Facility type to configure for.

    Returns:
        EnMSConfig instance with facility-appropriate defaults.
    """
    return EnMSConfig(facility_type=facility_type)


def get_facility_info(
    facility_type: Union[str, FacilityType],
) -> Dict[str, Any]:
    """Get detailed information about a facility type.

    Args:
        facility_type: Facility type enum or string value.

    Returns:
        Dictionary with name, typical EUI, energy split, SEUs, and EnPI info.
    """
    key = (
        facility_type.value
        if isinstance(facility_type, FacilityType)
        else facility_type
    )
    return FACILITY_TYPE_INFO.get(
        key,
        {
            "name": key,
            "typical_eui_kwh_per_m2": "Varies",
            "typical_energy_split": "Varies",
            "typical_seus": [
                "HVAC",
                "Lighting",
                "Process loads",
            ],
            "iso50001_focus": "General EnMS implementation",
            "typical_enpi": "kWh per m2 per year",
            "certification_timeline_months": "12-18",
        },
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()


def get_iso50001_clause_info(clause_id: str) -> Optional[str]:
    """Get description for an ISO 50001 clause.

    Args:
        clause_id: ISO 50001 clause number (e.g., '6.3', '9.1').

    Returns:
        Clause description string, or None if clause not found.
    """
    return ISO_50001_CLAUSES.get(clause_id)


def list_iso50001_clauses() -> Dict[str, str]:
    """List all ISO 50001:2018 clauses.

    Returns:
        Dictionary mapping clause IDs to descriptions.
    """
    return ISO_50001_CLAUSES.copy()
