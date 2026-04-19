"""
PACK-039 Energy Monitoring Pack - Configuration Module

This module implements the EnergyMonitoringConfig and PackConfig classes that
load, merge, and validate all configuration for the Energy Monitoring Pack. It
provides comprehensive Pydantic v2 models for real-time and interval energy
monitoring including meter management, energy performance indicator (EnPI)
tracking, anomaly detection, cost allocation, budget and target management,
alarming, carbon tracking, tenant billing, and audit trail provenance.

Facility Types:
    - COMMERCIAL_OFFICE: Office buildings (5-20 submeters, HVAC/lighting/plug EnPIs)
    - MANUFACTURING: Industrial plants (20-100 submeters, production-normalised EnPIs)
    - RETAIL_CHAIN: Retail stores (5-15 meters per store, multi-site portfolio)
    - HOSPITAL: Healthcare facilities (15-50 submeters, 24/7 critical systems)
    - UNIVERSITY_CAMPUS: University campuses (50-200 meters, multi-building)
    - DATA_CENTER: Data centres (20-80 submeters, PUE monitoring)
    - MULTI_TENANT: Multi-tenant buildings (per-tenant submetering, billing)
    - INDUSTRIAL_PROCESS: Process industry (30-150 meters, batch tracking)

Meter Protocols:
    - MODBUS_TCP: Modbus TCP/IP (most common BMS/submeter protocol)
    - MODBUS_RTU: Modbus RTU serial (legacy meters)
    - BACNET_IP: BACnet/IP (building automation systems)
    - MQTT: MQTT publish/subscribe (IoT meters, sensors)
    - OPC_UA: OPC Unified Architecture (industrial SCADA/DCS)
    - PULSE_OUTPUT: Pulse output contact closure (utility meters)
    - DLMS_COSEM: DLMS/COSEM IEC 62056 (smart utility meters)
    - API_REST: RESTful API integration (cloud meters, virtual meters)

Energy Types:
    - ELECTRICITY: Electrical energy (kWh)
    - NATURAL_GAS: Natural gas (kWh thermal / therms / m3)
    - STEAM: District steam (kWh thermal / GJ / klb)
    - CHILLED_WATER: District chilled water (kWh thermal / ton-hr)
    - HOT_WATER: District hot water (kWh thermal / GJ)
    - COMPRESSED_AIR: Compressed air (m3 / cfm)
    - FUEL_OIL: Fuel oil (litres / gallons)
    - LPG: Liquefied petroleum gas (litres / kg)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (commercial_office / manufacturing / retail_chain /
       hospital / university_campus / data_center / multi_tenant /
       industrial_process)
    3. Environment overrides (EM_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - ISO 50001:2018 Energy Management Systems (EnPIs, baselines)
    - ISO 50006:2014 Measuring Energy Performance Using EnPIs and EnBs
    - EU Energy Efficiency Directive (EU) 2023/1791
    - EU Energy Performance of Buildings Directive (EU) 2024/1275 (recast)
    - ASHRAE Guideline 14 - Measurement of Energy, Demand, and Water Savings
    - IPMVP (International Performance Measurement and Verification Protocol)
    - GHG Protocol Corporate Standard (Scope 1 and 2 emissions)
    - IEC 61968/61970 CIM (Common Information Model for meter data)

Example:
    >>> from packs.energy_efficiency.PACK_039_energy_monitoring.config import (
    ...     PackConfig,
    ...     EnergyMonitoringConfig,
    ...     FacilityType,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("commercial_office")
    >>> print(config.pack.facility_type)
    FacilityType.COMMERCIAL_OFFICE
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
# Enums - Energy monitoring configuration enumeration types
# =============================================================================


class FacilityType(str, Enum):
    """Facility type classification for energy monitoring scoping."""

    COMMERCIAL_OFFICE = "COMMERCIAL_OFFICE"
    MANUFACTURING = "MANUFACTURING"
    RETAIL_CHAIN = "RETAIL_CHAIN"
    HOSPITAL = "HOSPITAL"
    UNIVERSITY_CAMPUS = "UNIVERSITY_CAMPUS"
    DATA_CENTER = "DATA_CENTER"
    MULTI_TENANT = "MULTI_TENANT"
    INDUSTRIAL_PROCESS = "INDUSTRIAL_PROCESS"


class MeterProtocol(str, Enum):
    """Meter communication protocol for data acquisition."""

    MODBUS_TCP = "MODBUS_TCP"
    MODBUS_RTU = "MODBUS_RTU"
    BACNET_IP = "BACNET_IP"
    MQTT = "MQTT"
    OPC_UA = "OPC_UA"
    PULSE_OUTPUT = "PULSE_OUTPUT"
    DLMS_COSEM = "DLMS_COSEM"
    API_REST = "API_REST"


class EnergyType(str, Enum):
    """Energy carrier type for metered consumption."""

    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    STEAM = "STEAM"
    CHILLED_WATER = "CHILLED_WATER"
    HOT_WATER = "HOT_WATER"
    COMPRESSED_AIR = "COMPRESSED_AIR"
    FUEL_OIL = "FUEL_OIL"
    LPG = "LPG"


class IntervalLength(str, Enum):
    """Interval metering data granularity."""

    ONE_MINUTE = "1_MIN"
    FIVE_MINUTE = "5_MIN"
    FIFTEEN_MINUTE = "15_MIN"
    THIRTY_MINUTE = "30_MIN"
    SIXTY_MINUTE = "60_MIN"


class AnomalyMethod(str, Enum):
    """Anomaly detection methodology for energy consumption patterns."""

    STATISTICAL_ZSCORE = "STATISTICAL_ZSCORE"
    IQR_OUTLIER = "IQR_OUTLIER"
    CUSUM = "CUSUM"
    REGRESSION_RESIDUAL = "REGRESSION_RESIDUAL"
    ROLLING_BASELINE = "ROLLING_BASELINE"
    MACHINE_LEARNING = "MACHINE_LEARNING"


class EnPIType(str, Enum):
    """Energy Performance Indicator (EnPI) methodology per ISO 50006."""

    ABSOLUTE = "ABSOLUTE"
    INTENSITY_AREA = "INTENSITY_AREA"
    INTENSITY_PRODUCTION = "INTENSITY_PRODUCTION"
    INTENSITY_OCCUPANCY = "INTENSITY_OCCUPANCY"
    REGRESSION_MODEL = "REGRESSION_MODEL"
    CUSUM_BASELINE = "CUSUM_BASELINE"
    DEGREE_DAY_NORMALISED = "DEGREE_DAY_NORMALISED"


class AllocationMethod(str, Enum):
    """Energy cost and consumption allocation methodology."""

    DIRECT_SUBMETER = "DIRECT_SUBMETER"
    PROPORTIONAL_AREA = "PROPORTIONAL_AREA"
    PROPORTIONAL_HEADCOUNT = "PROPORTIONAL_HEADCOUNT"
    PRODUCTION_BASED = "PRODUCTION_BASED"
    OPERATING_HOURS = "OPERATING_HOURS"
    REGRESSION_DISAGGREGATION = "REGRESSION_DISAGGREGATION"
    HYBRID = "HYBRID"


class BudgetMethod(str, Enum):
    """Energy budget and target-setting methodology."""

    HISTORICAL_AVERAGE = "HISTORICAL_AVERAGE"
    WEATHER_NORMALISED = "WEATHER_NORMALISED"
    PRODUCTION_NORMALISED = "PRODUCTION_NORMALISED"
    TOP_DOWN_TARGET = "TOP_DOWN_TARGET"
    BOTTOM_UP_ENGINEERING = "BOTTOM_UP_ENGINEERING"
    REGRESSION_FORECAST = "REGRESSION_FORECAST"


class AlarmPriority(str, Enum):
    """Alarm priority classification for energy monitoring alerts."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class OutputFormat(str, Enum):
    """Output format for energy monitoring reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency for energy performance."""

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
        "typical_meter_count": "5-20",
        "typical_interval": "15_MIN",
        "energy_types": ["ELECTRICITY", "NATURAL_GAS"],
        "key_enpi": ["kWh/m2/yr", "kWh/occupant/yr", "W/m2 peak"],
        "monitoring_priorities": [
            "HVAC (40-60%)", "Lighting (15-25%)", "Plug loads (15-20%)",
            "Elevators/escalators (5-10%)"
        ],
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_meter_count": "20-100",
        "typical_interval": "5_MIN",
        "energy_types": ["ELECTRICITY", "NATURAL_GAS", "COMPRESSED_AIR", "STEAM"],
        "key_enpi": ["kWh/unit", "kWh/tonne", "SEC (specific energy consumption)"],
        "monitoring_priorities": [
            "Production lines (40-60%)", "Compressed air (10-20%)",
            "HVAC (10-15%)", "Lighting (5-10%)", "Auxiliaries (5-10%)"
        ],
    },
    "RETAIL_CHAIN": {
        "name": "Retail Chain / Store Portfolio",
        "typical_meter_count": "5-15 per store",
        "typical_interval": "15_MIN",
        "energy_types": ["ELECTRICITY", "NATURAL_GAS"],
        "key_enpi": ["kWh/m2/yr", "kWh/trading_hour", "kWh/revenue"],
        "monitoring_priorities": [
            "Refrigeration (35-45%)", "HVAC (25-35%)", "Lighting (15-20%)",
            "Cooking/baking (5-10%)"
        ],
    },
    "HOSPITAL": {
        "name": "Hospital / Healthcare Facility",
        "typical_meter_count": "15-50",
        "typical_interval": "15_MIN",
        "energy_types": ["ELECTRICITY", "NATURAL_GAS", "STEAM", "CHILLED_WATER"],
        "key_enpi": ["kWh/m2/yr", "kWh/bed-day", "kWh/patient-day"],
        "monitoring_priorities": [
            "HVAC/ventilation (35-45%)", "Medical equipment (15-25%)",
            "Lighting (10-15%)", "Kitchen/laundry (10-15%)",
            "Critical systems 24/7 (10-15%)"
        ],
    },
    "UNIVERSITY_CAMPUS": {
        "name": "University / Campus",
        "typical_meter_count": "50-200",
        "typical_interval": "15_MIN",
        "energy_types": ["ELECTRICITY", "NATURAL_GAS", "STEAM", "CHILLED_WATER"],
        "key_enpi": ["kWh/m2/yr", "kWh/student-FTE", "kWh/building"],
        "monitoring_priorities": [
            "HVAC (35-45%)", "Research labs (15-25%)", "Lighting (10-15%)",
            "Dining halls (5-10%)", "Data centres (5-10%)", "Residence halls (10-15%)"
        ],
    },
    "DATA_CENTER": {
        "name": "Data Centre / Server Facility",
        "typical_meter_count": "20-80",
        "typical_interval": "1_MIN",
        "energy_types": ["ELECTRICITY", "CHILLED_WATER"],
        "key_enpi": ["PUE", "DCiE", "kWh/rack", "kW/kW-IT"],
        "monitoring_priorities": [
            "IT compute (45-55%)", "Cooling (30-40%)",
            "UPS/power distribution (8-12%)", "Lighting/misc (2-5%)"
        ],
    },
    "MULTI_TENANT": {
        "name": "Multi-Tenant Building",
        "typical_meter_count": "10-100+",
        "typical_interval": "15_MIN",
        "energy_types": ["ELECTRICITY", "NATURAL_GAS", "CHILLED_WATER"],
        "key_enpi": ["kWh/m2/yr per tenant", "kWh/occupant", "common_area kWh/m2"],
        "monitoring_priorities": [
            "Tenant spaces (50-70%)", "Common areas (10-15%)",
            "HVAC central plant (15-25%)", "Elevators/parking (5-10%)"
        ],
    },
    "INDUSTRIAL_PROCESS": {
        "name": "Industrial Process Facility",
        "typical_meter_count": "30-150",
        "typical_interval": "1_MIN",
        "energy_types": [
            "ELECTRICITY", "NATURAL_GAS", "STEAM",
            "COMPRESSED_AIR", "FUEL_OIL"
        ],
        "key_enpi": ["kWh/tonne", "SEC", "GJ/batch", "kWh/unit"],
        "monitoring_priorities": [
            "Process heating/cooling (30-50%)", "Motors/drives (20-30%)",
            "Compressed air (10-15%)", "Pumps/fans (10-15%)",
            "Auxiliaries (5-10%)"
        ],
    },
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "commercial_office": "Office buildings with 5-20 submeters, HVAC/lighting/plug EnPIs, tenant allocation",
    "manufacturing": "Manufacturing with 20-100 submeters, production-normalised EnPIs, process metering",
    "retail_chain": "Retail stores with 5-15 meters per store, multi-site portfolio, refrigeration monitoring",
    "hospital": "Healthcare with 15-50 submeters, critical systems monitoring, department allocation",
    "university_campus": "University campuses with 50-200 meters, multi-building, departmental allocation",
    "data_center": "Data centres with 20-80 submeters, PUE monitoring, IT vs cooling allocation",
    "multi_tenant": "Multi-tenant with per-tenant submetering, interval cost allocation, tenant billing",
    "industrial_process": "Industrial process with 30-150 meters, batch tracking, specific energy consumption",
}

DEFAULT_FINANCIAL_PARAMS: Dict[str, Dict[str, float]] = {
    "US_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.12,
        "gas_rate_eur_per_kwh": 0.04,
        "steam_rate_eur_per_kwh": 0.06,
        "chilled_water_rate_eur_per_kwh": 0.08,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 30.0,
    },
    "EU_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.22,
        "gas_rate_eur_per_kwh": 0.07,
        "steam_rate_eur_per_kwh": 0.09,
        "chilled_water_rate_eur_per_kwh": 0.10,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "UK": {
        "electricity_rate_eur_per_kwh": 0.28,
        "gas_rate_eur_per_kwh": 0.08,
        "steam_rate_eur_per_kwh": 0.10,
        "chilled_water_rate_eur_per_kwh": 0.11,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 55.0,
    },
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class MeterConfig(BaseModel):
    """Configuration for meter data acquisition and management.

    Defines the meter inventory parameters, communication protocols,
    polling intervals, data quality thresholds, and virtual meter
    configuration used by the Meter Management Engine.
    """

    total_meter_count: int = Field(
        10,
        ge=1,
        le=1000,
        description="Total number of physical and virtual meters",
    )
    primary_protocol: MeterProtocol = Field(
        MeterProtocol.MODBUS_TCP,
        description="Primary meter communication protocol",
    )
    secondary_protocols: List[MeterProtocol] = Field(
        default_factory=list,
        description="Additional meter communication protocols in use",
    )
    polling_interval_seconds: int = Field(
        60,
        ge=1,
        le=3600,
        description="Meter polling interval in seconds",
    )
    energy_types_monitored: List[EnergyType] = Field(
        default_factory=lambda: [EnergyType.ELECTRICITY],
        description="Energy types monitored across all meters",
    )
    virtual_meters_enabled: bool = Field(
        True,
        description="Enable virtual meters (calculated from physical meter algebra)",
    )
    virtual_meter_count: int = Field(
        0,
        ge=0,
        le=500,
        description="Number of virtual/calculated meters",
    )
    meter_hierarchy_depth: int = Field(
        3,
        ge=1,
        le=10,
        description="Depth of meter hierarchy tree (main > sub > sub-sub)",
    )
    ct_ratio_validation: bool = Field(
        True,
        description="Validate CT ratio configuration against nameplate",
    )
    automatic_gap_detection: bool = Field(
        True,
        description="Automatically detect and flag data gaps in meter streams",
    )

    @model_validator(mode="after")
    def validate_virtual_meters(self) -> "MeterConfig":
        """Virtual meter count should not exceed total meter count."""
        if self.virtual_meter_count > self.total_meter_count:
            logger.warning(
                f"Virtual meter count ({self.virtual_meter_count}) exceeds "
                f"total meter count ({self.total_meter_count}). Virtual meters "
                "should be a subset of the total meter inventory."
            )
        return self


class IntervalDataConfig(BaseModel):
    """Configuration for interval meter data ingestion and analysis.

    Defines the expected interval data granularity, data quality thresholds,
    gap-filling strategies, and baseline period parameters used by the
    Interval Data Engine to process raw meter readings into clean time series.
    """

    interval_length: IntervalLength = Field(
        IntervalLength.FIFTEEN_MINUTE,
        description="Default interval data granularity for analysis",
    )
    minimum_data_coverage_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum data coverage required for analysis (%)",
    )
    gap_fill_method: str = Field(
        "LINEAR_INTERPOLATION",
        description="Gap-filling method: LINEAR_INTERPOLATION, PREVIOUS_VALUE, AVERAGE, NONE",
    )
    max_gap_fill_intervals: int = Field(
        4,
        ge=1,
        le=96,
        description="Maximum consecutive intervals to gap-fill before flagging",
    )
    lookback_months: int = Field(
        12,
        ge=1,
        le=36,
        description="Number of months of historical data for baseline analysis",
    )
    outlier_threshold_sigma: float = Field(
        3.0,
        ge=1.5,
        le=5.0,
        description="Standard deviation threshold for outlier detection",
    )
    weather_normalization: bool = Field(
        True,
        description="Apply weather normalisation (degree-day adjustment) to data",
    )
    heating_base_temp_c: float = Field(
        15.5,
        ge=5.0,
        le=25.0,
        description="Heating degree-day base temperature (Celsius)",
    )
    cooling_base_temp_c: float = Field(
        18.0,
        ge=10.0,
        le=30.0,
        description="Cooling degree-day base temperature (Celsius)",
    )
    demand_window_minutes: int = Field(
        15,
        ge=5,
        le=60,
        description="Demand averaging window for peak demand calculation (minutes)",
    )


class EnPIConfig(BaseModel):
    """Configuration for Energy Performance Indicators per ISO 50006.

    Defines the EnPI methodologies, baseline periods, normalisation
    variables, significance testing parameters, and tracking thresholds
    used by the EnPI Tracking Engine.
    """

    primary_enpi_type: EnPIType = Field(
        EnPIType.INTENSITY_AREA,
        description="Primary EnPI methodology for performance tracking",
    )
    secondary_enpi_types: List[EnPIType] = Field(
        default_factory=list,
        description="Additional EnPI methodologies to calculate",
    )
    baseline_year: int = Field(
        2024,
        ge=2015,
        le=2035,
        description="Baseline year for EnPI comparison (ISO 50001 EnB)",
    )
    baseline_months: int = Field(
        12,
        ge=6,
        le=36,
        description="Number of months in baseline period",
    )
    normalisation_variables: List[str] = Field(
        default_factory=lambda: ["floor_area_m2", "heating_degree_days", "cooling_degree_days"],
        description="Variables used for EnPI normalisation",
    )
    significance_level: float = Field(
        0.05,
        ge=0.01,
        le=0.10,
        description="Statistical significance level for EnPI change detection",
    )
    improvement_target_pct: float = Field(
        3.0,
        ge=0.5,
        le=20.0,
        description="Annual EnPI improvement target (%)",
    )
    cusum_tracking_enabled: bool = Field(
        True,
        description="Enable CUSUM (cumulative sum) chart for ongoing performance tracking",
    )
    regression_min_r_squared: float = Field(
        0.75,
        ge=0.5,
        le=0.99,
        description="Minimum R-squared for regression-based EnPI model acceptance",
    )
    auto_baseline_adjustment: bool = Field(
        False,
        description="Automatically adjust baseline when significant static factors change",
    )


class AnomalyDetectionConfig(BaseModel):
    """Configuration for energy consumption anomaly detection.

    Defines the anomaly detection methodologies, sensitivity thresholds,
    suppression windows, and notification rules used by the Anomaly
    Detection Engine to identify unusual consumption patterns.
    """

    enabled: bool = Field(
        True,
        description="Enable anomaly detection on meter data streams",
    )
    primary_method: AnomalyMethod = Field(
        AnomalyMethod.STATISTICAL_ZSCORE,
        description="Primary anomaly detection methodology",
    )
    secondary_methods: List[AnomalyMethod] = Field(
        default_factory=list,
        description="Additional anomaly detection methods to run in parallel",
    )
    sensitivity_sigma: float = Field(
        2.5,
        ge=1.0,
        le=5.0,
        description="Anomaly detection sensitivity (standard deviations from expected)",
    )
    minimum_deviation_pct: float = Field(
        15.0,
        ge=5.0,
        le=50.0,
        description="Minimum percentage deviation from expected to trigger anomaly (%)",
    )
    evaluation_window_hours: int = Field(
        24,
        ge=1,
        le=168,
        description="Window for evaluating consumption patterns (hours)",
    )
    suppression_window_hours: int = Field(
        4,
        ge=1,
        le=48,
        description="Minimum hours between repeated anomaly alerts for same meter",
    )
    weekday_weekend_split: bool = Field(
        True,
        description="Separate weekday and weekend baselines for anomaly detection",
    )
    holiday_calendar_enabled: bool = Field(
        True,
        description="Use holiday calendar to adjust expected consumption baselines",
    )
    after_hours_monitoring: bool = Field(
        True,
        description="Enhanced anomaly detection during non-occupied hours",
    )
    baseload_deviation_pct: float = Field(
        20.0,
        ge=5.0,
        le=50.0,
        description="Maximum acceptable deviation from expected baseload (%)",
    )


class AllocationConfig(BaseModel):
    """Configuration for energy cost and consumption allocation.

    Defines the allocation methodology, cost split ratios, tenant
    configuration, departmental allocation rules, and common area
    treatment used by the Allocation Engine.
    """

    enabled: bool = Field(
        True,
        description="Enable energy cost/consumption allocation",
    )
    primary_method: AllocationMethod = Field(
        AllocationMethod.DIRECT_SUBMETER,
        description="Primary allocation methodology",
    )
    fallback_method: AllocationMethod = Field(
        AllocationMethod.PROPORTIONAL_AREA,
        description="Fallback allocation method when submeter data unavailable",
    )
    common_area_split_method: str = Field(
        "PROPORTIONAL_AREA",
        description="Method for splitting common area energy: PROPORTIONAL_AREA, EQUAL, HEADCOUNT",
    )
    common_area_pct_cap: float = Field(
        30.0,
        ge=5.0,
        le=60.0,
        description="Maximum common area energy as percentage of total (%)",
    )
    allocation_frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Frequency of cost allocation calculations",
    )
    include_demand_charges: bool = Field(
        True,
        description="Include demand charge allocation in cost splits",
    )
    demand_charge_method: str = Field(
        "COINCIDENT_PEAK",
        description="Demand charge allocation: COINCIDENT_PEAK, NON_COINCIDENT, PROPORTIONAL",
    )
    loss_factor_pct: float = Field(
        3.0,
        ge=0.0,
        le=15.0,
        description="Distribution loss factor applied to submeter readings (%)",
    )
    reconciliation_enabled: bool = Field(
        True,
        description="Enable meter-to-billing reconciliation to balance main vs submeters",
    )
    reconciliation_tolerance_pct: float = Field(
        5.0,
        ge=1.0,
        le=15.0,
        description="Acceptable tolerance for main vs submeter reconciliation (%)",
    )


class BudgetTargetConfig(BaseModel):
    """Configuration for energy budgets and consumption targets.

    Defines the budget methodology, target levels, variance thresholds,
    and escalation rules used by the Budget and Target Engine to track
    energy performance against organisational goals.
    """

    enabled: bool = Field(
        True,
        description="Enable energy budget and target tracking",
    )
    budget_method: BudgetMethod = Field(
        BudgetMethod.WEATHER_NORMALISED,
        description="Budget-setting methodology",
    )
    annual_budget_kwh: Optional[float] = Field(
        None,
        ge=0.0,
        description="Annual energy budget (kWh) -- set per facility",
    )
    monthly_distribution: str = Field(
        "WEATHER_WEIGHTED",
        description="Monthly budget distribution: EQUAL, WEATHER_WEIGHTED, HISTORICAL_PATTERN",
    )
    warning_variance_pct: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Budget variance threshold for warning alert (%)",
    )
    critical_variance_pct: float = Field(
        10.0,
        ge=5.0,
        le=30.0,
        description="Budget variance threshold for critical alert (%)",
    )
    year_on_year_reduction_target_pct: float = Field(
        3.0,
        ge=0.0,
        le=15.0,
        description="Year-on-year energy reduction target (%)",
    )
    rolling_forecast_enabled: bool = Field(
        True,
        description="Enable rolling forecast of year-end energy consumption",
    )
    forecast_horizon_months: int = Field(
        3,
        ge=1,
        le=12,
        description="Forward-looking forecast horizon (months)",
    )
    degree_day_adjustment: bool = Field(
        True,
        description="Apply degree-day weather adjustment to budget tracking",
    )

    @model_validator(mode="after")
    def validate_variance_thresholds(self) -> "BudgetTargetConfig":
        """Warning threshold must be below critical threshold."""
        if self.warning_variance_pct >= self.critical_variance_pct:
            logger.warning(
                f"Budget warning variance ({self.warning_variance_pct}%) >= "
                f"critical variance ({self.critical_variance_pct}%). "
                "Warning should be below critical."
            )
        return self


class AlarmConfig(BaseModel):
    """Configuration for energy monitoring alarm and notification system.

    Defines alarm priorities, escalation rules, notification channels,
    suppression windows, and acknowledgement requirements used by the
    Alarm Management Engine.
    """

    enabled: bool = Field(
        True,
        description="Enable energy monitoring alarm system",
    )
    channels: List[str] = Field(
        default_factory=lambda: ["EMAIL", "SMS", "BMS_SIGNAL"],
        description="Notification channels for alarms",
    )
    default_priority: AlarmPriority = Field(
        AlarmPriority.MEDIUM,
        description="Default alarm priority for unclassified events",
    )
    critical_response_minutes: int = Field(
        15,
        ge=1,
        le=60,
        description="Required response time for critical alarms (minutes)",
    )
    high_response_minutes: int = Field(
        60,
        ge=5,
        le=240,
        description="Required response time for high-priority alarms (minutes)",
    )
    escalation_enabled: bool = Field(
        True,
        description="Enable escalation for unacknowledged alarms",
    )
    escalation_delay_minutes: int = Field(
        30,
        ge=5,
        le=120,
        description="Delay before escalation to next tier (minutes)",
    )
    suppression_window_minutes: int = Field(
        30,
        ge=5,
        le=120,
        description="Minimum minutes between repeated alarms of same type",
    )
    after_hours_critical_only: bool = Field(
        True,
        description="Only send critical alarms during after-hours periods",
    )
    meter_offline_alarm_minutes: int = Field(
        30,
        ge=5,
        le=240,
        description="Minutes of meter silence before triggering offline alarm",
    )
    acknowledgement_required: bool = Field(
        True,
        description="Require acknowledgement for high and critical alarms",
    )


class CostTrackingConfig(BaseModel):
    """Configuration for energy cost monitoring and tracking.

    Defines tariff rates, demand charges, time-of-use periods, and
    cost analysis parameters used by the Cost Tracking Engine to
    calculate real-time and historical energy costs.
    """

    enabled: bool = Field(
        True,
        description="Enable energy cost tracking and analysis",
    )
    electricity_rate_eur_per_kwh: float = Field(
        0.12,
        ge=0.01,
        le=1.0,
        description="Average electricity rate (EUR/kWh)",
    )
    gas_rate_eur_per_kwh: float = Field(
        0.04,
        ge=0.005,
        le=0.5,
        description="Natural gas rate (EUR/kWh thermal)",
    )
    steam_rate_eur_per_kwh: float = Field(
        0.06,
        ge=0.01,
        le=0.5,
        description="Steam rate (EUR/kWh thermal)",
    )
    chilled_water_rate_eur_per_kwh: float = Field(
        0.08,
        ge=0.01,
        le=0.5,
        description="Chilled water rate (EUR/kWh thermal)",
    )
    demand_charge_eur_per_kw: float = Field(
        15.00,
        ge=0.0,
        le=100.0,
        description="Monthly demand charge (EUR/kW of peak demand)",
    )
    time_of_use_enabled: bool = Field(
        False,
        description="Enable time-of-use rate differentiation",
    )
    peak_rate_multiplier: float = Field(
        1.5,
        ge=1.0,
        le=5.0,
        description="Peak period rate multiplier relative to average",
    )
    off_peak_rate_multiplier: float = Field(
        0.7,
        ge=0.1,
        le=1.0,
        description="Off-peak period rate multiplier relative to average",
    )
    currency: str = Field(
        "EUR",
        description="Currency for cost calculations (ISO 4217)",
    )
    cost_avoidance_tracking: bool = Field(
        True,
        description="Track cost avoidance from energy efficiency measures",
    )


class CarbonConfig(BaseModel):
    """Configuration for carbon impact tracking of energy consumption.

    Defines emission factors for each energy type, carbon accounting
    methodology (location-based vs market-based), and carbon intensity
    tracking parameters used by the Carbon Tracking Engine.
    """

    enabled: bool = Field(
        True,
        description="Enable carbon tracking for monitored energy consumption",
    )
    electricity_ef_kgco2_per_kwh: float = Field(
        0.40,
        ge=0.0,
        le=1.5,
        description="Electricity grid emission factor (kgCO2e/kWh)",
    )
    gas_ef_kgco2_per_kwh: float = Field(
        0.20,
        ge=0.0,
        le=0.5,
        description="Natural gas emission factor (kgCO2e/kWh)",
    )
    steam_ef_kgco2_per_kwh: float = Field(
        0.25,
        ge=0.0,
        le=0.8,
        description="Steam emission factor (kgCO2e/kWh)",
    )
    chilled_water_ef_kgco2_per_kwh: float = Field(
        0.15,
        ge=0.0,
        le=0.6,
        description="Chilled water emission factor (kgCO2e/kWh)",
    )
    carbon_price_eur_per_tco2: float = Field(
        85.0,
        ge=0.0,
        le=500.0,
        description="Carbon price for monetised carbon impact (EUR/tCO2e)",
    )
    market_based_accounting: bool = Field(
        False,
        description="Use market-based emission factors (RECs, green tariffs)",
    )
    scope1_tracking: bool = Field(
        True,
        description="Track Scope 1 emissions from on-site combustion",
    )
    scope2_tracking: bool = Field(
        True,
        description="Track Scope 2 emissions from purchased energy",
    )
    real_time_carbon_intensity: bool = Field(
        False,
        description="Use real-time grid carbon intensity data (where available)",
    )
    sbti_alignment_tracking: bool = Field(
        False,
        description="Track energy carbon reductions against SBTi pathway",
    )


class ReportingConfig(BaseModel):
    """Configuration for energy monitoring report generation and dashboards."""

    frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Reporting frequency for energy performance reports",
    )
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML],
        description="Output formats for energy monitoring reports",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary with key EnPIs and costs",
    )
    meter_health_report: bool = Field(
        True,
        description="Generate meter health and data quality report",
    )
    anomaly_report: bool = Field(
        True,
        description="Generate anomaly detection summary report",
    )
    budget_variance_report: bool = Field(
        True,
        description="Generate budget vs actual variance report",
    )
    cost_allocation_report: bool = Field(
        True,
        description="Generate cost allocation breakdown report",
    )
    carbon_report: bool = Field(
        True,
        description="Generate carbon emissions tracking report",
    )
    enpi_dashboard: bool = Field(
        True,
        description="Generate EnPI trend dashboard data",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration for energy monitoring."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "energy_monitoring_manager",
            "energy_manager",
            "facility_manager",
            "meter_technician",
            "tenant_manager",
            "sustainability_officer",
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
        description="Enable security audit logging for all monitoring operations",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored meter data",
    )
    tenant_data_isolation: bool = Field(
        True,
        description="Enforce tenant data isolation in multi-tenant deployments",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_meters: int = Field(
        500,
        ge=1,
        le=5000,
        description="Maximum number of meters per monitoring deployment",
    )
    max_intervals_per_query: int = Field(
        105120,
        ge=1000,
        le=525600,
        description="Maximum interval data points per query (default: 1 year of 5-min)",
    )
    cache_ttl_seconds: int = Field(
        300,
        ge=30,
        le=86400,
        description="Cache TTL for meter data and calculated values (seconds)",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=5000,
        description="Batch size for bulk meter data processing",
    )
    calculation_timeout_seconds: int = Field(
        120,
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
    real_time_buffer_seconds: int = Field(
        300,
        ge=30,
        le=3600,
        description="Buffer window for real-time data processing (seconds)",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all energy monitoring calculations",
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
        description="Track full data lineage from meter reading through report",
    )
    retention_years: int = Field(
        5,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class EnergyMonitoringConfig(BaseModel):
    """Main configuration for PACK-039 Energy Monitoring Pack.

    This is the root configuration model that contains all sub-configurations
    for energy monitoring. The facility_type field drives which meter
    configurations, EnPI methodologies, allocation methods, and anomaly
    detection parameters are prioritised for the deployment.
    """

    # Facility identification
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
        description="Primary facility type for energy monitoring scoping",
    )
    country: str = Field(
        "US",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for energy monitoring performance tracking",
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
    operating_hours_per_year: int = Field(
        2500,
        ge=500,
        le=8760,
        description="Annual operating hours",
    )
    occupant_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of building occupants or FTE",
    )
    production_units_per_year: Optional[float] = Field(
        None,
        ge=0,
        description="Annual production output (units, tonnes, etc.)",
    )

    # Sub-configurations
    meters: MeterConfig = Field(
        default_factory=MeterConfig,
        description="Meter data acquisition and management configuration",
    )
    interval_data: IntervalDataConfig = Field(
        default_factory=IntervalDataConfig,
        description="Interval data ingestion and analysis configuration",
    )
    enpi: EnPIConfig = Field(
        default_factory=EnPIConfig,
        description="Energy Performance Indicator configuration",
    )
    anomaly_detection: AnomalyDetectionConfig = Field(
        default_factory=AnomalyDetectionConfig,
        description="Anomaly detection configuration",
    )
    allocation: AllocationConfig = Field(
        default_factory=AllocationConfig,
        description="Energy cost/consumption allocation configuration",
    )
    budget_target: BudgetTargetConfig = Field(
        default_factory=BudgetTargetConfig,
        description="Budget and target tracking configuration",
    )
    alarms: AlarmConfig = Field(
        default_factory=AlarmConfig,
        description="Alarm and notification configuration",
    )
    cost_tracking: CostTrackingConfig = Field(
        default_factory=CostTrackingConfig,
        description="Energy cost tracking configuration",
    )
    carbon: CarbonConfig = Field(
        default_factory=CarbonConfig,
        description="Carbon impact tracking configuration",
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
        description="Audit trail and provenance",
    )

    @model_validator(mode="after")
    def validate_hospital_constraints(self) -> "EnergyMonitoringConfig":
        """Healthcare facilities must maintain 24/7 monitoring on critical systems."""
        if self.facility_type == FacilityType.HOSPITAL:
            if self.meters.polling_interval_seconds > 60:
                logger.warning(
                    "Hospital facility: polling interval exceeds 60 seconds. "
                    "Critical systems require frequent monitoring."
                )
            if not self.alarms.enabled:
                logger.warning(
                    "Hospital facility: alarms are disabled. 24/7 critical "
                    "system monitoring requires active alarming."
                )
        return self

    @model_validator(mode="after")
    def validate_data_center_constraints(self) -> "EnergyMonitoringConfig":
        """Data centres must track PUE and maintain high-frequency monitoring."""
        if self.facility_type == FacilityType.DATA_CENTER:
            if self.interval_data.interval_length not in (
                IntervalLength.ONE_MINUTE, IntervalLength.FIVE_MINUTE
            ):
                logger.warning(
                    "Data centre: interval length is not 1-min or 5-min. "
                    "PUE monitoring requires high-frequency data."
                )
        return self

    @model_validator(mode="after")
    def validate_multi_tenant_constraints(self) -> "EnergyMonitoringConfig":
        """Multi-tenant buildings must have allocation and tenant isolation enabled."""
        if self.facility_type == FacilityType.MULTI_TENANT:
            if not self.allocation.enabled:
                logger.warning(
                    "Multi-tenant facility: allocation is disabled. "
                    "Tenant billing requires active cost allocation."
                )
            if not self.security.tenant_data_isolation:
                logger.warning(
                    "Multi-tenant facility: tenant data isolation is disabled. "
                    "Tenant privacy requires data isolation."
                )
        return self

    @model_validator(mode="after")
    def validate_manufacturing_enpi(self) -> "EnergyMonitoringConfig":
        """Manufacturing facilities should use production-normalised EnPIs."""
        if self.facility_type in (
            FacilityType.MANUFACTURING, FacilityType.INDUSTRIAL_PROCESS
        ):
            if self.enpi.primary_enpi_type == EnPIType.INTENSITY_AREA:
                logger.warning(
                    "Manufacturing/process facility: using area-based EnPI. "
                    "Production-normalised EnPI (kWh/unit) is recommended."
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

    pack: EnergyMonitoringConfig = Field(
        default_factory=EnergyMonitoringConfig,
        description="Main Energy Monitoring configuration",
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
        "PACK-039-energy-monitoring",
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

        pack_config = EnergyMonitoringConfig(**preset_data)
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

        pack_config = EnergyMonitoringConfig(**config_data)
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
        pack_config = EnergyMonitoringConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with EM_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: EM_PACK_ENPI__IMPROVEMENT_TARGET_PCT=5
                 EM_PACK_COST_TRACKING__ELECTRICITY_RATE_EUR_PER_KWH=0.18
        """
        overrides: Dict[str, Any] = {}
        prefix = "EM_PACK_"
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


def validate_config(config: EnergyMonitoringConfig) -> List[str]:
    """Validate an energy monitoring configuration and return any warnings.

    Args:
        config: EnergyMonitoringConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check facility identification
    if not config.facility_name:
        warnings.append(
            "No facility_name configured. Add a facility name for report identification."
        )

    # Check floor area for area-based EnPIs
    if config.floor_area_m2 is None and config.enpi.primary_enpi_type in (
        EnPIType.INTENSITY_AREA, EnPIType.DEGREE_DAY_NORMALISED
    ):
        warnings.append(
            "No floor_area_m2 configured but area-based EnPI selected. "
            "Floor area is required for kWh/m2 calculations."
        )

    # Check production data for production-based EnPIs
    if config.production_units_per_year is None and config.enpi.primary_enpi_type == (
        EnPIType.INTENSITY_PRODUCTION
    ):
        warnings.append(
            "No production_units_per_year configured but production-based EnPI "
            "selected. Production data is required for kWh/unit calculations."
        )

    # Check annual energy data
    if config.annual_electricity_kwh is None:
        warnings.append(
            "No annual_electricity_kwh configured. Energy budgeting and EnPI "
            "baseline calculations require annual consumption data."
        )

    # Check meter count vs facility type
    facility_info = FACILITY_TYPE_INFO.get(config.facility_type.value, {})
    expected_range = facility_info.get("typical_meter_count", "")
    if expected_range and config.meters.total_meter_count < 2:
        warnings.append(
            f"Only {config.meters.total_meter_count} meter(s) configured for "
            f"{config.facility_type.value} (typical: {expected_range}). "
            "Submetering is essential for effective energy monitoring."
        )

    # Check interval data coverage
    if config.interval_data.minimum_data_coverage_pct < 80:
        warnings.append(
            f"Interval data coverage threshold ({config.interval_data.minimum_data_coverage_pct}%) "
            "is below 80%. EnPI accuracy may be reduced."
        )

    # Check EnPI baseline year
    if config.enpi.baseline_year >= config.reporting_year:
        warnings.append(
            f"EnPI baseline year ({config.enpi.baseline_year}) is not before "
            f"reporting year ({config.reporting_year}). Baseline must precede "
            "the reporting period."
        )

    # Check anomaly detection sensitivity
    if config.anomaly_detection.enabled:
        if config.anomaly_detection.sensitivity_sigma < 2.0:
            warnings.append(
                f"Anomaly detection sensitivity ({config.anomaly_detection.sensitivity_sigma} sigma) "
                "is very high. This may generate excessive false positive alerts."
            )

    # Check allocation for multi-tenant
    if config.facility_type == FacilityType.MULTI_TENANT:
        if not config.allocation.enabled:
            warnings.append(
                "Multi-tenant facility requires allocation enabled for tenant billing."
            )
        if config.allocation.primary_method != AllocationMethod.DIRECT_SUBMETER:
            warnings.append(
                "Multi-tenant facility: DIRECT_SUBMETER allocation is recommended "
                "for accurate tenant billing."
            )

    # Check budget configuration
    if config.budget_target.enabled and config.budget_target.annual_budget_kwh is None:
        warnings.append(
            "Budget tracking enabled but no annual_budget_kwh configured. "
            "Set an annual budget for variance tracking."
        )

    # Check carbon configuration
    if config.carbon.enabled:
        if config.carbon.electricity_ef_kgco2_per_kwh <= 0:
            warnings.append(
                "Carbon tracking enabled but electricity emission factor is zero."
            )

    # Check cost tracking rates
    if config.cost_tracking.enabled:
        if config.cost_tracking.electricity_rate_eur_per_kwh <= 0:
            warnings.append(
                "Cost tracking enabled but electricity rate is zero. "
                "Verify tariff configuration."
            )

    # Check hospital 24/7 monitoring
    if config.facility_type == FacilityType.HOSPITAL:
        if config.meters.polling_interval_seconds > 60:
            warnings.append(
                "Hospital facility: polling interval exceeds 60 seconds. "
                "Critical systems require frequent polling."
            )

    # Check data center PUE monitoring
    if config.facility_type == FacilityType.DATA_CENTER:
        if config.interval_data.interval_length not in (
            IntervalLength.ONE_MINUTE, IntervalLength.FIVE_MINUTE
        ):
            warnings.append(
                "Data centre: interval length should be 1-min or 5-min for PUE monitoring."
            )

    return warnings


def get_default_config(
    facility_type: FacilityType = FacilityType.COMMERCIAL_OFFICE,
) -> EnergyMonitoringConfig:
    """Get default configuration for a given facility type.

    Args:
        facility_type: Facility type to configure for.

    Returns:
        EnergyMonitoringConfig instance with facility-appropriate defaults.
    """
    return EnergyMonitoringConfig(facility_type=facility_type)


def get_facility_info(facility_type: Union[str, FacilityType]) -> Dict[str, Any]:
    """Get detailed information about a facility type.

    Args:
        facility_type: Facility type enum or string value.

    Returns:
        Dictionary with name, typical meters, energy types, and monitoring priorities.
    """
    key = facility_type.value if isinstance(facility_type, FacilityType) else facility_type
    return FACILITY_TYPE_INFO.get(
        key,
        {
            "name": key,
            "typical_meter_count": "Varies",
            "typical_interval": "15_MIN",
            "energy_types": ["ELECTRICITY"],
            "key_enpi": ["kWh/m2/yr"],
            "monitoring_priorities": ["HVAC", "Lighting"],
        },
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()


__all__ = [
    # Enums
    "AlarmPriority",
    "AllocationMethod",
    "AnomalyMethod",
    "BudgetMethod",
    "EnergyType",
    "EnPIType",
    "FacilityType",
    "IntervalLength",
    "MeterProtocol",
    "OutputFormat",
    "ReportingFrequency",
    # Sub-config models
    "AlarmConfig",
    "AllocationConfig",
    "AnomalyDetectionConfig",
    "AuditTrailConfig",
    "BudgetTargetConfig",
    "CarbonConfig",
    "CostTrackingConfig",
    "EnPIConfig",
    "IntervalDataConfig",
    "MeterConfig",
    "PerformanceConfig",
    "ReportingConfig",
    "SecurityConfig",
    # Main config models
    "EnergyMonitoringConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "CONFIG_DIR",
    "DEFAULT_FINANCIAL_PARAMS",
    "FACILITY_TYPE_INFO",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_facility_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
