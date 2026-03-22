# -*- coding: utf-8 -*-
"""
PACK-036 Utility Analysis Pack - Configuration
================================================
Pydantic v2 configuration models for utility analysis including
facility profiles, commodity settings, analysis parameters,
reporting preferences, and facility-type presets.

Facility Types:
    - OFFICE: Office buildings (TOU optimization, HVAC-dominated load)
    - MANUFACTURING: Manufacturing plants (demand charges, power factor, process loads)
    - RETAIL: Retail stores (multi-site portfolio, lighting-dominated, extended hours)
    - WAREHOUSE: Warehouses (lighting-dominated, seasonal demand, large area)
    - HEALTHCARE: Hospitals and clinics (24/7 operations, critical loads)
    - EDUCATION: Schools and universities (seasonal occupancy, term-time patterns)
    - DATA_CENTER: Data centres (high-density loads, PUE tracking, cooling efficiency)
    - MULTI_SITE: Multi-site portfolios (cross-facility benchmarking, consolidated procurement)

Analysis Depth Levels:
    - QUICK: Rapid bill review with anomaly flagging (30-60 min)
    - STANDARD: Full bill audit with rate optimization analysis (2-4 hours)
    - COMPREHENSIVE: Complete utility analysis with demand profiling and budgeting (1-2 days)

Budget Methods:
    - HISTORICAL_TREND: Rolling average with seasonal adjustment
    - REGRESSION: Weather-regression (HDD/CDD) based forecast
    - MONTE_CARLO: Stochastic simulation with rate and weather uncertainty
    - ENSEMBLE: Weighted blend of all methods for robust forecast

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (office_building / manufacturing / retail_store /
       warehouse / healthcare / education / data_center / multi_site_portfolio)
    3. Environment overrides (UTILITY_ANALYSIS_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - EED: EU Energy Efficiency Directive 2023/1791 (Article 8 energy audits)
    - ISO 50001: Energy management systems (utility data requirements)
    - ASHRAE 14: Measurement of Energy, Demand, and Water Savings
    - IPMVP: Measurement and verification of energy savings
    - GHG Protocol: Corporate Standard (Scope 2 purchased energy)
    - ENERGY STAR: Portfolio Manager benchmarking integration
    - EU ETS: Emissions Trading System cost pass-through tracking

Example:
    >>> config = PackConfig.from_preset("office_building")
    >>> print(config.pack.facility.facility_type)
    FacilityType.OFFICE
    >>> print(config.pack.analysis_depth)
    AnalysisDepth.STANDARD
    >>> print(config.pack.budget.method)
    BudgetMethod.REGRESSION
"""

from __future__ import annotations

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
# Enums - Utility analysis enumeration types
# =============================================================================


class FacilityType(str, Enum):
    """Facility type classification for utility analysis scoping."""

    OFFICE = "OFFICE"
    MANUFACTURING = "MANUFACTURING"
    RETAIL = "RETAIL"
    WAREHOUSE = "WAREHOUSE"
    HEALTHCARE = "HEALTHCARE"
    EDUCATION = "EDUCATION"
    DATA_CENTER = "DATA_CENTER"
    MULTI_SITE = "MULTI_SITE"


class CommodityType(str, Enum):
    """Utility commodity types tracked in analysis."""

    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    WATER = "WATER"
    STEAM = "STEAM"
    CHILLED_WATER = "CHILLED_WATER"


class AnalysisDepth(str, Enum):
    """Analysis depth level for utility analysis scope."""

    QUICK = "QUICK"
    STANDARD = "STANDARD"
    COMPREHENSIVE = "COMPREHENSIVE"


class CurrencyCode(str, Enum):
    """Supported currency codes (ISO 4217)."""

    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    PLN = "PLN"
    CZK = "CZK"
    AUD = "AUD"
    NZD = "NZD"
    CAD = "CAD"
    JPY = "JPY"


class ReportingFrequency(str, Enum):
    """Reporting and tracking frequency for utility analysis."""

    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


class BudgetMethod(str, Enum):
    """Budget forecasting methodology."""

    HISTORICAL_TREND = "HISTORICAL_TREND"
    REGRESSION = "REGRESSION"
    MONTE_CARLO = "MONTE_CARLO"
    ENSEMBLE = "ENSEMBLE"


class OutputFormat(str, Enum):
    """Output format for utility analysis reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"
    XLSX = "XLSX"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Facility type display names, typical energy use, and utility analysis focus
FACILITY_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "OFFICE": {
        "name": "Office Building",
        "typical_eui_electricity_kwh_per_m2": "150-300",
        "typical_eui_gas_kwh_per_m2": "50-150",
        "typical_energy_split": "Electricity 70-85%, Gas 15-30%",
        "utility_focus": [
            "TOU rate optimization",
            "Demand charge management",
            "HVAC-dominated load profiling",
            "Occupancy-driven consumption patterns",
            "Weekend/holiday baseload analysis",
        ],
        "demand_charge_pct_of_bill": "30-50",
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_eui_electricity_kwh_per_m2": "200-500",
        "typical_eui_gas_kwh_per_m2": "100-400",
        "typical_energy_split": "Electricity 50-70%, Gas 25-45%",
        "utility_focus": [
            "Demand charge optimization",
            "Power factor correction",
            "Shift-pattern load scheduling",
            "Process load profiling",
            "Demand ratchet avoidance",
        ],
        "demand_charge_pct_of_bill": "40-60",
    },
    "RETAIL": {
        "name": "Retail Store",
        "typical_eui_electricity_kwh_per_m2": "200-400",
        "typical_eui_gas_kwh_per_m2": "20-80",
        "typical_energy_split": "Electricity 80-95%, Gas 5-20%",
        "utility_focus": [
            "Multi-site rate comparison",
            "Extended-hours load profiling",
            "Lighting-dominated consumption",
            "Portfolio demand consistency",
            "Seasonal promotion impacts",
        ],
        "demand_charge_pct_of_bill": "20-35",
    },
    "WAREHOUSE": {
        "name": "Warehouse & Distribution Centre",
        "typical_eui_electricity_kwh_per_m2": "50-150",
        "typical_eui_gas_kwh_per_m2": "20-80",
        "typical_energy_split": "Electricity 60-80%, Gas 15-35%",
        "utility_focus": [
            "Simple rate structure optimization",
            "Lighting load timing analysis",
            "Seasonal demand variation",
            "Dock door energy loss quantification",
            "Forklift charging load scheduling",
        ],
        "demand_charge_pct_of_bill": "15-30",
    },
    "HEALTHCARE": {
        "name": "Hospital / Clinic",
        "typical_eui_electricity_kwh_per_m2": "300-600",
        "typical_eui_gas_kwh_per_m2": "200-400",
        "typical_energy_split": "Electricity 50-65%, Gas 30-45%",
        "utility_focus": [
            "Budget accuracy (critical for planning)",
            "24/7 baseload analysis",
            "Demand consistency profiling",
            "Redundancy load quantification",
            "Steam and chilled water sub-metering",
        ],
        "demand_charge_pct_of_bill": "25-40",
    },
    "EDUCATION": {
        "name": "School / University",
        "typical_eui_electricity_kwh_per_m2": "100-250",
        "typical_eui_gas_kwh_per_m2": "80-200",
        "typical_energy_split": "Electricity 45-65%, Gas 30-50%",
        "utility_focus": [
            "Seasonal rate optimization (term vs holiday)",
            "Weather normalization for comparisons",
            "Campus-wide portfolio management",
            "Vacation shutdown verification",
            "Sports facility usage patterns",
        ],
        "demand_charge_pct_of_bill": "20-35",
    },
    "DATA_CENTER": {
        "name": "Data Centre / Server Room",
        "typical_eui_electricity_kwh_per_m2": "500-2000",
        "typical_eui_gas_kwh_per_m2": "0-10",
        "typical_energy_split": "Electricity 95-100%",
        "utility_focus": [
            "Flat demand profile procurement optimization",
            "PUE-correlated utility tracking",
            "Cooling efficiency vs ambient temperature",
            "Renewable energy procurement",
            "UPS efficiency monitoring",
        ],
        "demand_charge_pct_of_bill": "15-25",
    },
    "MULTI_SITE": {
        "name": "Multi-Site Portfolio",
        "typical_eui_electricity_kwh_per_m2": "Varies by site",
        "typical_eui_gas_kwh_per_m2": "Varies by site",
        "typical_energy_split": "Varies by site",
        "utility_focus": [
            "Cross-facility benchmarking",
            "Consolidated procurement negotiation",
            "Portfolio-level demand aggregation",
            "Best-practice sharing across sites",
            "Normalized comparison analytics",
        ],
        "demand_charge_pct_of_bill": "Varies",
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "office_building": "Office buildings with TOU optimization, demand management, HVAC-dominated load",
    "manufacturing": "Manufacturing facilities with demand charges, power factor, process loads",
    "retail_store": "Retail stores with multi-site portfolio, lighting-dominated, extended hours",
    "warehouse": "Warehouses with lighting-dominated, seasonal demand, large area",
    "healthcare": "Hospitals and clinics with 24/7 operations, critical loads, high reliability",
    "education": "Schools and universities with seasonal occupancy, term-time patterns",
    "data_center": "Data centres with high-density loads, PUE tracking, cooling efficiency",
    "multi_site_portfolio": "Multi-site portfolios with cross-facility benchmarking, consolidated procurement",
}

# Default commodity units by type
DEFAULT_COMMODITY_UNITS: Dict[str, str] = {
    "ELECTRICITY": "kWh",
    "NATURAL_GAS": "therms",
    "WATER": "m3",
    "STEAM": "klb",
    "CHILLED_WATER": "ton-hr",
}

# Default emission factors by commodity
DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "ELECTRICITY": 0.40,       # kgCO2e/kWh (EU average grid)
    "NATURAL_GAS": 0.20,       # kgCO2e/kWh
    "STEAM": 0.25,             # kgCO2e/kWh (district steam)
    "CHILLED_WATER": 0.15,     # kgCO2e/kWh (district cooling)
    "WATER": 0.0003,           # kgCO2e/litre (water treatment)
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class FacilityConfig(BaseModel):
    """Configuration for the facility being analysed.

    Identifies the physical facility and its operational characteristics
    used for normalization, benchmarking, and weather correction of
    utility consumption data.
    """

    facility_id: str = Field(
        "",
        description="Unique facility identifier (internal reference code)",
    )
    name: str = Field(
        "",
        description="Facility name or site identifier",
    )
    facility_type: FacilityType = Field(
        FacilityType.OFFICE,
        description="Primary facility type for utility analysis scoping",
    )
    address: str = Field(
        "",
        description="Full street address for weather station mapping",
    )
    gross_floor_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Gross floor area in square metres for EUI calculations",
    )
    occupancy_pct: float = Field(
        100.0,
        ge=0.0,
        le=100.0,
        description="Average occupancy percentage during operating hours",
    )
    operating_hours_per_week: float = Field(
        50.0,
        ge=0.0,
        le=168.0,
        description="Typical weekly operating hours",
    )
    num_workers: Optional[int] = Field(
        None,
        ge=0,
        description="Number of regular occupants or workers",
    )
    year_built: Optional[int] = Field(
        None,
        ge=1800,
        le=2030,
        description="Year the facility was constructed",
    )
    climate_zone: str = Field(
        "",
        description="ASHRAE climate zone (e.g. 4A, 5B) or Koppen classification",
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Facility latitude for weather data mapping",
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Facility longitude for weather data mapping",
    )
    utility_territory: str = Field(
        "",
        description="Utility service territory or distribution zone name",
    )
    currency: CurrencyCode = Field(
        CurrencyCode.EUR,
        description="Currency for all financial calculations (ISO 4217)",
    )

    @field_validator("facility_id")
    @classmethod
    def validate_facility_id_format(cls, v: str) -> str:
        """Facility ID should not contain whitespace if provided."""
        if v and " " in v:
            raise ValueError(
                "facility_id must not contain spaces. Use hyphens or underscores."
            )
        return v


class CommodityConfig(BaseModel):
    """Configuration for a single utility commodity.

    Defines the commodity type, metering details, rate structure,
    and consumption baseline used for bill auditing and analysis.
    """

    commodity_type: CommodityType = Field(
        ...,
        description="Type of utility commodity",
    )
    enabled: bool = Field(
        True,
        description="Whether this commodity is active for analysis",
    )
    unit: str = Field(
        "kWh",
        description="Measurement unit (kWh, therms, m3, klb, ton-hr)",
    )
    rate_per_unit: Optional[float] = Field(
        None,
        ge=0.0,
        description="Average volumetric rate per unit of consumption",
    )
    demand_rate_per_kw: Optional[float] = Field(
        None,
        ge=0.0,
        description="Demand charge rate per kW of peak demand (electricity only)",
    )
    annual_consumption: Optional[float] = Field(
        None,
        ge=0.0,
        description="Baseline annual consumption in the configured unit",
    )
    utility_account_number: str = Field(
        "",
        description="Utility account number for bill matching",
    )
    meter_ids: List[str] = Field(
        default_factory=list,
        description="List of meter identifiers associated with this commodity",
    )

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        """Validate unit is a recognized measurement unit."""
        allowed_units = {"kWh", "MWh", "therms", "m3", "ccf", "MCF", "klb", "ton-hr", "GJ", "MMBtu"}
        if v not in allowed_units:
            logger.warning(
                f"Unit '{v}' is not in standard list {sorted(allowed_units)}. "
                "Ensure custom unit has a valid conversion factor configured."
            )
        return v


class BillAuditConfig(BaseModel):
    """Configuration for utility bill auditing and anomaly detection.

    Controls thresholds for identifying billing errors, estimated reads,
    period gaps, and tax discrepancies in historical utility bills.
    """

    enabled: bool = Field(
        True,
        description="Enable bill auditing engine",
    )
    anomaly_threshold_std_dev: float = Field(
        2.0,
        ge=1.0,
        le=5.0,
        description="Standard deviation threshold for anomaly detection on consumption",
    )
    max_estimated_reads: int = Field(
        2,
        ge=0,
        le=12,
        description="Maximum consecutive estimated reads before flagging",
    )
    period_gap_days: int = Field(
        35,
        ge=28,
        le=90,
        description="Maximum billing period length (days) before flagging a gap",
    )
    tax_tolerance_pct: float = Field(
        0.5,
        ge=0.0,
        le=5.0,
        description="Tolerance for tax/surcharge calculation discrepancy (%)",
    )
    check_rate_consistency: bool = Field(
        True,
        description="Verify applied rates match tariff schedule",
    )
    check_meter_read_sequence: bool = Field(
        True,
        description="Verify meter reads are sequential (no rollbacks or skips)",
    )
    lookback_months: int = Field(
        24,
        ge=6,
        le=60,
        description="Number of months of bill history to audit",
    )

    @field_validator("anomaly_threshold_std_dev")
    @classmethod
    def validate_anomaly_threshold(cls, v: float) -> float:
        """Warn if threshold is very loose."""
        if v > 3.0:
            logger.warning(
                f"Anomaly threshold {v} sigma is very loose. "
                "Many billing errors may go undetected."
            )
        return v


class RateOptimizationConfig(BaseModel):
    """Configuration for rate structure and tariff optimization.

    Controls which rate structures are evaluated and the minimum
    savings threshold for recommending a rate change.
    """

    enabled: bool = Field(
        True,
        description="Enable rate optimization analysis",
    )
    include_tou: bool = Field(
        True,
        description="Evaluate time-of-use (TOU) rate structures",
    )
    include_demand_ratchet: bool = Field(
        True,
        description="Evaluate demand ratchet clauses and peak shaving potential",
    )
    include_real_time_pricing: bool = Field(
        False,
        description="Evaluate real-time pricing / spot market options",
    )
    include_green_tariffs: bool = Field(
        False,
        description="Evaluate renewable energy tariff options",
    )
    min_savings_threshold_eur: float = Field(
        500.0,
        ge=0.0,
        description="Minimum annual savings (EUR) to recommend a rate change",
    )
    compare_max_tariffs: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of alternative tariffs to evaluate",
    )
    include_contract_terms: bool = Field(
        True,
        description="Consider contract length and termination clauses in recommendation",
    )
    peak_shaving_analysis: bool = Field(
        True,
        description="Evaluate peak demand shaving through load shifting or storage",
    )


class DemandConfig(BaseModel):
    """Configuration for demand analysis and load profiling.

    Controls interval data resolution, peak identification thresholds,
    and power factor targets for demand charge optimization.
    """

    enabled: bool = Field(
        True,
        description="Enable demand analysis engine",
    )
    interval_resolution_minutes: int = Field(
        15,
        ge=1,
        le=60,
        description="Interval data resolution in minutes (1, 5, 15, 30, or 60)",
    )
    peak_threshold_pct: float = Field(
        90.0,
        ge=50.0,
        le=99.0,
        description="Percentile threshold for identifying peak demand events (%)",
    )
    power_factor_target: float = Field(
        0.95,
        ge=0.80,
        le=1.00,
        description="Target power factor for power factor correction analysis",
    )
    load_factor_tracking: bool = Field(
        True,
        description="Track and report load factor (avg demand / peak demand)",
    )
    coincident_peak_analysis: bool = Field(
        True,
        description="Analyse coincident peak contribution for demand charges",
    )
    ratchet_period_months: int = Field(
        12,
        ge=1,
        le=12,
        description="Demand ratchet lookback period in months",
    )
    peak_event_count: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of top peak events to identify and report",
    )

    @field_validator("interval_resolution_minutes")
    @classmethod
    def validate_interval_resolution(cls, v: int) -> int:
        """Validate interval resolution is a standard value."""
        allowed = {1, 5, 15, 30, 60}
        if v not in allowed:
            raise ValueError(
                f"interval_resolution_minutes must be one of {sorted(allowed)}, got {v}"
            )
        return v


class BudgetConfig(BaseModel):
    """Configuration for utility budget forecasting.

    Defines the forecasting methodology, horizon, rate escalation,
    and confidence intervals for utility cost budgeting.
    """

    enabled: bool = Field(
        True,
        description="Enable budget forecasting engine",
    )
    forecast_horizon_months: int = Field(
        12,
        ge=1,
        le=60,
        description="Budget forecast horizon in months",
    )
    method: BudgetMethod = Field(
        BudgetMethod.REGRESSION,
        description="Primary budget forecasting methodology",
    )
    monte_carlo_iterations: int = Field(
        1000,
        ge=100,
        le=50000,
        description="Number of Monte Carlo simulation iterations (if method is MONTE_CARLO or ENSEMBLE)",
    )
    rate_escalation_pct: float = Field(
        3.5,
        ge=0.0,
        le=20.0,
        description="Annual rate escalation assumption for budget projections (%)",
    )
    confidence_levels: List[float] = Field(
        default_factory=lambda: [0.80, 0.90, 0.95],
        description="Confidence levels for budget range estimation",
    )
    weather_normalize: bool = Field(
        True,
        description="Apply weather normalization (HDD/CDD) to baseline and forecast",
    )
    base_year_months: int = Field(
        12,
        ge=6,
        le=36,
        description="Number of months of historical data for baseline calculation",
    )
    include_demand_budget: bool = Field(
        True,
        description="Include demand charge budget separately from consumption",
    )

    @field_validator("confidence_levels")
    @classmethod
    def validate_confidence_levels(cls, v: List[float]) -> List[float]:
        """Validate all confidence levels are between 0 and 1."""
        for level in v:
            if not 0.0 < level < 1.0:
                raise ValueError(
                    f"Confidence level {level} must be between 0.0 and 1.0 (exclusive)."
                )
        return sorted(v)


class BenchmarkConfig(BaseModel):
    """Configuration for utility benchmarking against peers and standards.

    Defines which benchmarking standards to compare against, the peer
    group for relative performance, and weather normalization settings.
    """

    enabled: bool = Field(
        True,
        description="Enable benchmarking analysis",
    )
    standards: List[str] = Field(
        default_factory=lambda: ["ENERGY_STAR", "CIBSE_TM46", "ASHRAE_100"],
        description="Benchmarking standards to compare against",
    )
    peer_group_building_type: str = Field(
        "",
        description="Building type for peer group comparison (matches standard taxonomy)",
    )
    weather_normalize: bool = Field(
        True,
        description="Apply weather normalization before benchmarking comparisons",
    )
    include_eui: bool = Field(
        True,
        description="Calculate and compare Energy Use Intensity (kWh/m2/yr)",
    )
    include_cost_intensity: bool = Field(
        True,
        description="Calculate and compare cost intensity (EUR/m2/yr)",
    )
    include_carbon_intensity: bool = Field(
        True,
        description="Calculate and compare carbon intensity (kgCO2e/m2/yr)",
    )
    percentile_target: int = Field(
        25,
        ge=1,
        le=99,
        description="Target percentile rank for improvement planning (lower = better)",
    )
    normalize_by_occupancy: bool = Field(
        False,
        description="Normalize consumption by occupancy level for fairer comparison",
    )


class ReportingConfig(BaseModel):
    """Configuration for utility analysis report generation and distribution."""

    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML],
        description="Output formats for utility analysis reports",
    )
    frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Reporting frequency for utility tracking updates",
    )
    recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses for automated report distribution",
    )
    include_executive_summary: bool = Field(
        True,
        description="Generate executive summary with key metrics and trends",
    )
    include_benchmarks: bool = Field(
        True,
        description="Include benchmarking comparisons in reports",
    )
    include_carbon: bool = Field(
        True,
        description="Include Scope 2 carbon emissions analysis in reports",
    )
    include_budget_variance: bool = Field(
        True,
        description="Include budget vs actual variance analysis",
    )
    include_rate_analysis: bool = Field(
        True,
        description="Include rate optimization findings in reports",
    )
    include_demand_profile: bool = Field(
        True,
        description="Include demand profile and load factor charts",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )

    @field_validator("recipients")
    @classmethod
    def validate_recipients(cls, v: List[str]) -> List[str]:
        """Validate email addresses in recipients list."""
        for email in v:
            if email and "@" not in email:
                raise ValueError(f"Invalid email address: {email}")
        return v


class CarbonConfig(BaseModel):
    """Configuration for carbon emissions tracking linked to utility consumption.

    Calculates Scope 1 and Scope 2 emissions from utility data using
    grid and fuel emission factors per GHG Protocol Corporate Standard.
    """

    enabled: bool = Field(
        True,
        description="Enable carbon emissions tracking from utility data",
    )
    grid_emission_factor_kgco2_per_kwh: float = Field(
        0.40,
        ge=0.0,
        le=1.5,
        description="Grid electricity emission factor (kgCO2e/kWh)",
    )
    gas_emission_factor_kgco2_per_kwh: float = Field(
        0.20,
        ge=0.0,
        le=0.50,
        description="Natural gas emission factor (kgCO2e/kWh)",
    )
    carbon_price_eur_per_tco2: float = Field(
        85.0,
        ge=0.0,
        le=500.0,
        description="Carbon price for monetised carbon cost (EUR/tCO2e)",
    )
    include_scope_1: bool = Field(
        True,
        description="Include Scope 1 (direct combustion) carbon tracking",
    )
    include_scope_2: bool = Field(
        True,
        description="Include Scope 2 (purchased electricity) carbon tracking",
    )
    market_based_accounting: bool = Field(
        False,
        description="Use market-based emission factors (RECs, green tariffs)",
    )
    track_carbon_trend: bool = Field(
        True,
        description="Track month-over-month and year-over-year carbon trends",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "energy_manager",
            "facility_manager",
            "finance_analyst",
            "sustainability_officer",
            "procurement_officer",
            "viewer",
            "admin",
        ],
        description="Available RBAC roles for the pack",
    )
    data_classification: str = Field(
        "CONFIDENTIAL",
        description="Default data classification: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED",
    )
    audit_logging: bool = Field(
        True,
        description="Enable security audit logging for all data access",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports (account numbers, addresses)",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored utility data",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_facilities: int = Field(
        200,
        ge=1,
        le=2000,
        description="Maximum number of facilities per analysis run",
    )
    max_bills_per_facility: int = Field(
        600,
        ge=12,
        le=3000,
        description="Maximum bills to process per facility (across all commodities)",
    )
    max_interval_records: int = Field(
        500000,
        ge=1000,
        le=5000000,
        description="Maximum interval data records to load per facility",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for rate schedules and emission factors (seconds)",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=5000,
        description="Batch size for bulk bill processing",
    )
    calculation_timeout_seconds: int = Field(
        180,
        ge=15,
        le=600,
        description="Timeout for individual engine calculations (seconds)",
    )
    parallel_engines: int = Field(
        4,
        ge=1,
        le=8,
        description="Maximum number of engines running in parallel",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all calculations",
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
        description="Track full data lineage from source bill to output metric",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    bill_image_retention: bool = Field(
        True,
        description="Retain links to original bill images/PDFs for audit reference",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class UtilityAnalysisConfig(BaseModel):
    """Main configuration for PACK-036 Utility Analysis Pack.

    This is the root configuration model that contains all sub-configurations
    for utility bill auditing, rate optimization, demand analysis, budget
    forecasting, benchmarking, and reporting. The facility configuration
    drives which analysis profiles and benchmarks are applied.
    """

    # Facility identification
    facility: FacilityConfig = Field(
        default_factory=FacilityConfig,
        description="Facility identification and characteristics",
    )

    # Commodity definitions
    commodities: List[CommodityConfig] = Field(
        default_factory=lambda: [
            CommodityConfig(
                commodity_type=CommodityType.ELECTRICITY,
                enabled=True,
                unit="kWh",
            ),
            CommodityConfig(
                commodity_type=CommodityType.NATURAL_GAS,
                enabled=True,
                unit="therms",
            ),
        ],
        description="List of utility commodities to analyse",
    )

    # Analysis depth
    analysis_depth: AnalysisDepth = Field(
        AnalysisDepth.STANDARD,
        description="Overall analysis depth: QUICK, STANDARD, or COMPREHENSIVE",
    )

    # Reporting year
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for utility analysis",
    )

    # Sub-configurations
    bill_audit: BillAuditConfig = Field(
        default_factory=BillAuditConfig,
        description="Bill auditing and anomaly detection configuration",
    )
    rate_optimization: RateOptimizationConfig = Field(
        default_factory=RateOptimizationConfig,
        description="Rate structure and tariff optimization configuration",
    )
    demand: DemandConfig = Field(
        default_factory=DemandConfig,
        description="Demand analysis and load profiling configuration",
    )
    budget: BudgetConfig = Field(
        default_factory=BudgetConfig,
        description="Budget forecasting configuration",
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Benchmarking configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Report generation and distribution configuration",
    )
    carbon: CarbonConfig = Field(
        default_factory=CarbonConfig,
        description="Carbon emissions tracking configuration",
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
        description="Audit trail and provenance",
    )

    @model_validator(mode="after")
    def validate_commodities_not_empty(self) -> "UtilityAnalysisConfig":
        """At least one commodity must be enabled for analysis."""
        enabled = [c for c in self.commodities if c.enabled]
        if not enabled:
            raise ValueError(
                "At least one commodity must be enabled for utility analysis."
            )
        return self

    @model_validator(mode="after")
    def validate_demand_requires_electricity(self) -> "UtilityAnalysisConfig":
        """Demand analysis requires electricity commodity to be enabled."""
        if self.demand.enabled:
            has_electricity = any(
                c.commodity_type == CommodityType.ELECTRICITY and c.enabled
                for c in self.commodities
            )
            if not has_electricity:
                logger.warning(
                    "Demand analysis is enabled but electricity commodity is not active. "
                    "Demand analysis will be skipped."
                )
        return self

    @model_validator(mode="after")
    def validate_data_center_demand(self) -> "UtilityAnalysisConfig":
        """Data centres typically have flat demand -- adjust defaults."""
        if self.facility.facility_type == FacilityType.DATA_CENTER:
            if self.demand.enabled and self.demand.peak_threshold_pct < 95.0:
                logger.info(
                    "Data centre facility: demand is typically flat. Consider raising "
                    "peak_threshold_pct to 95+ to focus on true anomalies."
                )
        return self

    @model_validator(mode="after")
    def validate_multi_site_benchmarking(self) -> "UtilityAnalysisConfig":
        """Multi-site facilities should have benchmarking enabled."""
        if self.facility.facility_type == FacilityType.MULTI_SITE:
            if not self.benchmark.enabled:
                logger.info(
                    "Multi-site portfolio: benchmarking is recommended for "
                    "cross-facility comparison. Consider enabling benchmark."
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

    pack: UtilityAnalysisConfig = Field(
        default_factory=UtilityAnalysisConfig,
        description="Main Utility Analysis configuration",
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
        "PACK-036-utility-analysis",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (office_building, manufacturing, etc.)
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

        pack_config = UtilityAnalysisConfig(**preset_data)
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

        pack_config = UtilityAnalysisConfig(**config_data)
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
        pack_config = UtilityAnalysisConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with UTILITY_ANALYSIS_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: UTILITY_ANALYSIS_PACK_ANALYSIS_DEPTH=COMPREHENSIVE
                 UTILITY_ANALYSIS_PACK_BUDGET__METHOD=MONTE_CARLO
        """
        overrides: Dict[str, Any] = {}
        prefix = "UTILITY_ANALYSIS_PACK_"
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


def load_config(path: str) -> PackConfig:
    """Load configuration from a YAML file path.

    Convenience wrapper around PackConfig.from_yaml().

    Args:
        path: Path to YAML configuration file.

    Returns:
        PackConfig instance with YAML values applied.
    """
    return PackConfig.from_yaml(path)


def load_preset(
    facility_type: FacilityType,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a preset configuration by facility type.

    Maps FacilityType enum values to preset names and loads
    the corresponding preset YAML file.

    Args:
        facility_type: FacilityType enum for preset selection.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    preset_map: Dict[FacilityType, str] = {
        FacilityType.OFFICE: "office_building",
        FacilityType.MANUFACTURING: "manufacturing",
        FacilityType.RETAIL: "retail_store",
        FacilityType.WAREHOUSE: "warehouse",
        FacilityType.HEALTHCARE: "healthcare",
        FacilityType.EDUCATION: "education",
        FacilityType.DATA_CENTER: "data_center",
        FacilityType.MULTI_SITE: "multi_site_portfolio",
    }
    preset_name = preset_map.get(facility_type)
    if not preset_name:
        raise ValueError(f"No preset available for facility type: {facility_type}")
    return PackConfig.from_preset(preset_name, overrides)


def merge_configs(base: PackConfig, override: dict) -> PackConfig:
    """Merge overrides into a base configuration.

    Convenience wrapper around PackConfig.merge().

    Args:
        base: Base PackConfig instance.
        override: Dictionary of configuration overrides.

    Returns:
        New PackConfig with merged values.
    """
    return PackConfig.merge(base, override)


def validate_config(config: UtilityAnalysisConfig) -> List[str]:
    """Validate a utility analysis configuration and return any warnings.

    Args:
        config: UtilityAnalysisConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check facility identification
    if not config.facility.name:
        warnings.append(
            "No facility name configured. Add a name for report identification."
        )

    # Check floor area for intensity metrics
    if config.facility.gross_floor_area_m2 is None:
        warnings.append(
            "No gross_floor_area_m2 configured. Energy Use Intensity (EUI) "
            "benchmarking cannot be calculated without floor area."
        )

    # Check at least one commodity has consumption data
    has_consumption = any(
        c.annual_consumption is not None and c.annual_consumption > 0
        for c in config.commodities
        if c.enabled
    )
    if not has_consumption:
        warnings.append(
            "No annual_consumption configured for any enabled commodity. "
            "Bill audit and budget forecasting require baseline consumption data."
        )

    # Check rate data for rate optimization
    if config.rate_optimization.enabled:
        has_rates = any(
            c.rate_per_unit is not None and c.rate_per_unit > 0
            for c in config.commodities
            if c.enabled
        )
        if not has_rates:
            warnings.append(
                "Rate optimization is enabled but no rate_per_unit is configured "
                "for any commodity. Rate comparison requires current rate data."
            )

    # Check demand analysis prerequisites
    if config.demand.enabled:
        has_electricity = any(
            c.commodity_type == CommodityType.ELECTRICITY and c.enabled
            for c in config.commodities
        )
        if not has_electricity:
            warnings.append(
                "Demand analysis is enabled but electricity commodity is not active. "
                "Demand analysis requires electricity interval data."
            )

    # Check budget weather normalization prerequisites
    if config.budget.enabled and config.budget.weather_normalize:
        if not config.facility.latitude or not config.facility.longitude:
            warnings.append(
                "Budget weather normalization is enabled but facility latitude/longitude "
                "are not set. Weather data mapping requires geographic coordinates."
            )

    # Check benchmark configuration
    if config.benchmark.enabled:
        if not config.benchmark.standards:
            warnings.append(
                "Benchmarking is enabled but no standards are configured. "
                "Add at least one standard (e.g. ENERGY_STAR, CIBSE_TM46)."
            )

    # Check carbon emission factors
    if config.carbon.enabled:
        if config.carbon.grid_emission_factor_kgco2_per_kwh <= 0:
            warnings.append(
                "Carbon tracking enabled but grid emission factor is zero or negative."
            )

    # Check multi-site specific configuration
    if config.facility.facility_type == FacilityType.MULTI_SITE:
        if not config.benchmark.enabled:
            warnings.append(
                "Multi-site portfolio analysis benefits significantly from benchmarking. "
                "Consider enabling the benchmark configuration."
            )

    # Check reporting recipients
    if config.reporting.recipients:
        for email in config.reporting.recipients:
            if "@" not in email:
                warnings.append(f"Invalid email address in recipients: {email}")

    return warnings


def get_facility_info(facility_type: Union[str, FacilityType]) -> Dict[str, Any]:
    """Get detailed information about a facility type.

    Args:
        facility_type: Facility type enum or string value.

    Returns:
        Dictionary with name, typical EUI, energy split, and utility focus areas.
    """
    key = facility_type.value if isinstance(facility_type, FacilityType) else facility_type
    return FACILITY_TYPE_INFO.get(
        key,
        {
            "name": key,
            "typical_eui_electricity_kwh_per_m2": "Varies",
            "typical_eui_gas_kwh_per_m2": "Varies",
            "typical_energy_split": "Varies",
            "utility_focus": ["Bill auditing", "Rate optimization", "Budget forecasting"],
            "demand_charge_pct_of_bill": "Varies",
        },
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()


def get_default_config(
    facility_type: FacilityType = FacilityType.OFFICE,
) -> UtilityAnalysisConfig:
    """Get default configuration for a given facility type.

    Args:
        facility_type: Facility type to configure for.

    Returns:
        UtilityAnalysisConfig instance with facility-appropriate defaults.
    """
    return UtilityAnalysisConfig(
        facility=FacilityConfig(facility_type=facility_type),
    )
