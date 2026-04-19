"""
PACK-037 Demand Response Pack - Configuration Module

This module implements the DemandResponseConfig and PackConfig classes that load,
merge, and validate all configuration for the Demand Response Pack. It provides
comprehensive Pydantic v2 models for demand response management including load
flexibility assessment, DR program matching, customer baseline load simulation,
dispatch optimization, DER coordination, performance tracking, revenue
optimization, and carbon impact quantification.

Facility Types:
    - COMMERCIAL_OFFICE: Office buildings (HVAC, lighting, plug loads, EV)
    - MANUFACTURING: Industrial plants (motors, compressed air, process, HVAC)
    - RETAIL_GROCERY: Retail stores (HVAC, refrigeration thermal mass, lighting)
    - WAREHOUSE_COLD: Cold storage (refrigeration ride-through, lighting, dock)
    - HEALTHCARE: Hospitals (non-clinical HVAC, corridors, admin, kitchen)
    - EDUCATION_CAMPUS: Schools/universities (scheduling-based, summer DR)
    - DATA_CENTER: Data centres (cooling setpoint, UPS eco-mode, EV)
    - MICROGRID_DER: DER-rich sites (battery, solar, EV V2B, thermal, genset)

Baseline Methodologies:
    - HIGH_4_OF_5: PJM-style high 4 of 5 non-event days
    - 10_OF_10: Average of last 10 non-event weekdays
    - HIGH_5_SIMILAR: Highest 5 similar-day averages
    - 10CP: 10 coincident peak hours methodology
    - DEEMED_PROFILE: Pre-agreed deemed savings profile
    - TYPE_I_REGRESSION: Temperature-regression baseline
    - EU_STANDARD: EU standard baseline methodology
    - CUSTOM_REGRESSION: Custom multivariate regression

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (commercial_office / manufacturing / retail_grocery /
       warehouse_cold / healthcare / education_campus / data_center / microgrid_der)
    3. Environment overrides (DR_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - EU Electricity Market Design Regulation (EU) 2024/1747
    - EU Energy Efficiency Directive (EU) 2023/1791
    - FERC Order 2222 (DER aggregation in wholesale markets)
    - FERC Order 745 (DR compensation at LMP)
    - OpenADR 2.0b (Automated Demand Response protocol)
    - ISO 50001:2018 (Energy management systems)
    - GHG Protocol Corporate Standard (Scope 2 emissions)
    - SBTi Corporate Net-Zero Standard

Example:
    >>> from packs.energy_efficiency.PACK_037_demand_response.config import (
    ...     PackConfig,
    ...     DemandResponseConfig,
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
# Enums - Demand response configuration enumeration types
# =============================================================================


class FacilityType(str, Enum):
    """Facility type classification for DR program scoping."""

    COMMERCIAL_OFFICE = "COMMERCIAL_OFFICE"
    MANUFACTURING = "MANUFACTURING"
    RETAIL_GROCERY = "RETAIL_GROCERY"
    WAREHOUSE_COLD = "WAREHOUSE_COLD"
    HEALTHCARE = "HEALTHCARE"
    EDUCATION_CAMPUS = "EDUCATION_CAMPUS"
    DATA_CENTER = "DATA_CENTER"
    MICROGRID_DER = "MICROGRID_DER"


class BaselineMethodology(str, Enum):
    """Customer baseline load calculation methodology."""

    HIGH_4_OF_5 = "HIGH_4_OF_5"
    TEN_OF_10 = "10_OF_10"
    HIGH_5_SIMILAR = "HIGH_5_SIMILAR"
    TEN_CP = "10CP"
    DEEMED_PROFILE = "DEEMED_PROFILE"
    TYPE_I_REGRESSION = "TYPE_I_REGRESSION"
    EU_STANDARD = "EU_STANDARD"
    CUSTOM_REGRESSION = "CUSTOM_REGRESSION"


class DRProgramType(str, Enum):
    """Demand response program classification."""

    CAPACITY = "CAPACITY"
    ECONOMIC = "ECONOMIC"
    EMERGENCY = "EMERGENCY"
    ANCILLARY = "ANCILLARY"
    FREQUENCY_REGULATION = "FREQUENCY_REGULATION"
    DEMAND_CHARGE_MANAGEMENT = "DEMAND_CHARGE_MANAGEMENT"
    PRICE_RESPONSE = "PRICE_RESPONSE"


class LoadCriticality(str, Enum):
    """Load criticality classification for DR dispatch priority."""

    CRITICAL = "CRITICAL"
    ESSENTIAL = "ESSENTIAL"
    DEFERRABLE = "DEFERRABLE"
    SHEDDABLE = "SHEDDABLE"
    FLEXIBLE = "FLEXIBLE"


class CurtailmentStrategy(str, Enum):
    """Load curtailment strategy during DR events."""

    SETPOINT_RAISE = "SETPOINT_RAISE"
    DIMMING = "DIMMING"
    ZONE_SHED = "ZONE_SHED"
    CIRCUIT_SHED = "CIRCUIT_SHED"
    DEFERRAL = "DEFERRAL"
    VSD_REDUCTION = "VSD_REDUCTION"
    SCHEDULE_SHIFT = "SCHEDULE_SHIFT"
    PRE_COOL_COAST = "PRE_COOL_COAST"
    DUTY_CYCLE = "DUTY_CYCLE"
    ECO_MODE = "ECO_MODE"
    V2B_DISCHARGE = "V2B_DISCHARGE"
    FULL_SHED = "FULL_SHED"
    PRESSURE_REDUCTION = "PRESSURE_REDUCTION"
    LAB_SHUTDOWN = "LAB_SHUTDOWN"


class RiskTolerance(str, Enum):
    """Risk tolerance level for DR program enrollment."""

    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class OptimizationAggressiveness(str, Enum):
    """Baseline optimization aggressiveness level."""

    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class OutputFormat(str, Enum):
    """Output format for DR reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency for DR performance."""

    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


class DERType(str, Enum):
    """Distributed energy resource type classification."""

    BATTERY_BESS = "BATTERY_BESS"
    SOLAR_PV = "SOLAR_PV"
    BACKUP_GENERATOR = "BACKUP_GENERATOR"
    EV_V2B = "EV_V2B"
    THERMAL_STORAGE = "THERMAL_STORAGE"
    CHP = "CHP"


class ISORegion(str, Enum):
    """US ISO/RTO and international market regions."""

    PJM = "PJM"
    ERCOT = "ERCOT"
    CAISO = "CAISO"
    ISO_NE = "ISO_NE"
    NYISO = "NYISO"
    MISO = "MISO"
    SPP = "SPP"
    EU_AGGREGATOR = "EU_AGGREGATOR"
    UK_FLEX = "UK_FLEX"


# =============================================================================
# Reference Data Constants
# =============================================================================


FACILITY_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "COMMERCIAL_OFFICE": {
        "name": "Commercial Office",
        "typical_peak_kw": "200-2000",
        "typical_curtailment_pct": "15-35",
        "primary_loads": ["HVAC", "Lighting", "Plug loads", "EV charging"],
        "key_constraints": ["Occupant comfort", "Meeting schedules"],
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_peak_kw": "500-10000",
        "typical_curtailment_pct": "10-30",
        "primary_loads": ["Motors", "Compressed air", "Process", "HVAC"],
        "key_constraints": ["Production throughput", "Product quality"],
    },
    "RETAIL_GROCERY": {
        "name": "Retail / Grocery Store",
        "typical_peak_kw": "100-1500",
        "typical_curtailment_pct": "10-25",
        "primary_loads": ["Refrigeration", "HVAC", "Lighting"],
        "key_constraints": ["Food safety temperatures", "Customer comfort"],
    },
    "WAREHOUSE_COLD": {
        "name": "Warehouse / Cold Storage",
        "typical_peak_kw": "200-5000",
        "typical_curtailment_pct": "15-35",
        "primary_loads": ["Refrigeration", "Lighting", "Dock equipment"],
        "key_constraints": ["Cold chain integrity", "HACCP limits"],
    },
    "HEALTHCARE": {
        "name": "Hospital / Clinic",
        "typical_peak_kw": "500-5000",
        "typical_curtailment_pct": "5-15",
        "primary_loads": ["Non-clinical HVAC", "Admin lighting", "Kitchen"],
        "key_constraints": ["Patient safety", "Clinical exclusions"],
    },
    "EDUCATION_CAMPUS": {
        "name": "School / University Campus",
        "typical_peak_kw": "200-5000",
        "typical_curtailment_pct": "15-30",
        "primary_loads": ["HVAC", "Lighting", "IT labs", "EV"],
        "key_constraints": ["Academic schedule", "Research labs"],
    },
    "DATA_CENTER": {
        "name": "Data Centre / Server Room",
        "typical_peak_kw": "1000-50000",
        "typical_curtailment_pct": "5-15",
        "primary_loads": ["Cooling (CRAH/CRAC)", "UPS", "Lighting"],
        "key_constraints": ["IT uptime SLA (99.99%)", "PUE impact"],
    },
    "MICROGRID_DER": {
        "name": "Microgrid / DER-Rich Site",
        "typical_peak_kw": "500-10000",
        "typical_curtailment_pct": "30-80",
        "primary_loads": ["Battery BESS", "Solar PV", "EV V2B", "Thermal"],
        "key_constraints": ["SOC management", "Islanding capability"],
    },
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "commercial_office": "Corporate offices with HVAC, lighting, plug load, and EV DR",
    "manufacturing": "Manufacturing with motor VSD, compressed air, and process DR",
    "retail_grocery": "Retail/grocery with refrigeration thermal mass pre-cool DR",
    "warehouse_cold": "Cold storage with thermal mass ride-through and dock DR",
    "healthcare": "Healthcare with conservative non-clinical-only DR",
    "education_campus": "Education with scheduling-based and summer-enhanced DR",
    "data_center": "Data centres with cooling setpoint and UPS eco-mode DR",
    "microgrid_der": "DER-rich sites with full battery/solar/EV/thermal orchestration",
}

DEFAULT_FINANCIAL_PARAMS: Dict[str, Dict[str, float]] = {
    "US_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.12,
        "demand_charge_eur_per_kw": 12.00,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 30.0,
    },
    "EU_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.22,
        "demand_charge_eur_per_kw": 8.00,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "UK": {
        "electricity_rate_eur_per_kwh": 0.28,
        "demand_charge_eur_per_kw": 10.00,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 55.0,
    },
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class NotificationConfig(BaseModel):
    """Configuration for DR event notification delivery."""

    channels: List[str] = Field(
        default_factory=lambda: ["EMAIL", "SMS", "BMS_SIGNAL"],
        description="Notification channels for DR event alerts",
    )
    escalation_minutes: int = Field(
        15,
        ge=1,
        le=60,
        description="Minutes before escalation to next notification tier",
    )


class ComfortConstraintConfig(BaseModel):
    """Configuration for occupant comfort constraints during DR events."""

    max_temperature_rise_c: float = Field(
        2.0,
        ge=0.0,
        le=10.0,
        description="Maximum allowable temperature rise during event (Celsius)",
    )
    max_humidity_deviation_pct: float = Field(
        10.0,
        ge=0.0,
        le=30.0,
        description="Maximum allowable humidity deviation (%)",
    )
    min_lighting_level_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Minimum lighting level during curtailment (%)",
    )
    noise_limit_db: float = Field(
        55.0,
        ge=20.0,
        le=100.0,
        description="Maximum acceptable noise level during event (dB)",
    )


class OperationalConstraintConfig(BaseModel):
    """Configuration for operational constraints during DR events.

    These constraints define hard limits that must not be violated
    during any DR event dispatch. Facility-type-specific fields
    are optional and only relevant for matching facility types.
    """

    exclude_critical_areas: bool = Field(
        True,
        description="Exclude all critical areas from DR participation",
    )
    require_approval: bool = Field(
        False,
        description="Require manual approval before event execution",
    )
    extra_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Facility-type-specific operational constraint overrides",
    )


class DRParametersConfig(BaseModel):
    """Configuration for core demand response parameters.

    Defines curtailment capacity, event duration limits, notification
    preferences, and comfort/operational constraint boundaries.
    """

    min_curtailment_kw: float = Field(
        50.0,
        ge=0.0,
        le=100000.0,
        description="Minimum curtailment capacity to participate in DR (kW)",
    )
    max_event_duration_hours: float = Field(
        4.0,
        ge=0.5,
        le=24.0,
        description="Maximum DR event duration the facility can sustain (hours)",
    )
    max_events_per_season: int = Field(
        30,
        ge=1,
        le=200,
        description="Maximum number of DR events per season",
    )
    min_notification_minutes: int = Field(
        30,
        ge=0,
        le=1440,
        description="Minimum advance notification required for event dispatch (minutes)",
    )
    notification_preferences: NotificationConfig = Field(
        default_factory=NotificationConfig,
        description="DR event notification delivery configuration",
    )
    comfort_constraints: ComfortConstraintConfig = Field(
        default_factory=ComfortConstraintConfig,
        description="Occupant comfort constraints during DR events",
    )
    operational_constraints: OperationalConstraintConfig = Field(
        default_factory=OperationalConstraintConfig,
        description="Operational constraints during DR events",
    )


class LoadCategoryConfig(BaseModel):
    """Configuration for a single load category in the DR flexibility register."""

    included: bool = Field(
        False,
        description="Whether this load category participates in DR",
    )
    default_criticality: str = Field(
        "ESSENTIAL",
        description="Default criticality assignment for loads in this category",
    )
    curtailment_strategy: Optional[str] = Field(
        None,
        description="Default curtailment strategy for loads in this category",
    )
    scope: Optional[str] = Field(
        None,
        description="Scope limitation for load inclusion (e.g., NON_CLINICAL_ONLY)",
    )


class LoadCategoriesConfig(BaseModel):
    """Configuration for all load categories in the DR flexibility register.

    Defines which load types participate in DR, their default criticality
    assignment, and the curtailment strategy for each type.
    """

    hvac: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=True, default_criticality="DEFERRABLE",
            curtailment_strategy="SETPOINT_RAISE",
        ),
        description="HVAC loads (chillers, AHUs, RTUs, boilers)",
    )
    lighting: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=True, default_criticality="FLEXIBLE",
            curtailment_strategy="DIMMING",
        ),
        description="Lighting loads (interior, exterior, signage)",
    )
    plug_loads: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=False, default_criticality="SHEDDABLE",
        ),
        description="Plug loads (receptacles, non-essential equipment)",
    )
    ev_charging: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=False, default_criticality="DEFERRABLE",
            curtailment_strategy="DEFERRAL",
        ),
        description="EV charging stations",
    )
    it_equipment: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=False, default_criticality="CRITICAL",
        ),
        description="IT and server equipment",
    )
    refrigeration: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=False, default_criticality="ESSENTIAL",
        ),
        description="Refrigeration (walk-in coolers, display cases)",
    )
    motors_drives: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=False, default_criticality="ESSENTIAL",
        ),
        description="Motors and variable speed drives",
    )
    process_loads: LoadCategoryConfig = Field(
        default_factory=lambda: LoadCategoryConfig(
            included=False, default_criticality="CRITICAL",
        ),
        description="Process and production loads",
    )
    extra_categories: Dict[str, LoadCategoryConfig] = Field(
        default_factory=dict,
        description="Facility-type-specific additional load categories",
    )


class ProgramPreferencesConfig(BaseModel):
    """Configuration for DR program selection preferences.

    Defines which program types the facility prefers to enroll in,
    risk tolerance for penalty exposure, and minimum revenue thresholds.
    """

    preferred_types: List[str] = Field(
        default_factory=lambda: ["CAPACITY", "ECONOMIC"],
        description="Preferred DR program types for enrollment",
    )
    risk_tolerance: RiskTolerance = Field(
        RiskTolerance.MODERATE,
        description="Risk tolerance for penalty exposure in DR programs",
    )
    minimum_revenue_eur_per_year: float = Field(
        5000.0,
        ge=0.0,
        description="Minimum annual revenue threshold for program enrollment (EUR)",
    )
    preferred_isos: List[str] = Field(
        default_factory=lambda: ["PJM"],
        description="Preferred ISO/RTO regions for program matching",
    )


class BaselinePreferencesConfig(BaseModel):
    """Configuration for customer baseline load calculation preferences.

    Controls which CBL methodology is used, how aggressively the
    baseline is optimized within program rules, and whether
    adjustment factors are applied.
    """

    preferred_methodology: str = Field(
        "HIGH_4_OF_5",
        description="Preferred baseline calculation methodology",
    )
    optimization_aggressiveness: OptimizationAggressiveness = Field(
        OptimizationAggressiveness.MODERATE,
        description="How aggressively to optimize baseline within program rules",
    )
    adjustment_factor_enabled: bool = Field(
        True,
        description="Apply day-of adjustment factor to baseline",
    )


class DispatchWeightsConfig(BaseModel):
    """Configuration for multi-objective dispatch optimization weights.

    Defines the relative importance of each objective in the dispatch
    optimization linear program. Weights must sum to approximately 1.0.
    """

    operational_disruption: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Weight for minimizing operational disruption",
    )
    comfort_deviation: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Weight for minimizing occupant comfort deviation",
    )
    rebound_effect: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Weight for minimizing post-event rebound peak",
    )
    target_shortfall: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight for meeting curtailment target",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "DispatchWeightsConfig":
        """Validate that dispatch weights approximately sum to 1.0."""
        total = (
            self.operational_disruption
            + self.comfort_deviation
            + self.rebound_effect
            + self.target_shortfall
        )
        if abs(total - 1.0) > 0.05:
            logger.warning(
                f"Dispatch weights sum to {total:.2f}, expected ~1.0. "
                "Optimization results may not reflect intended balance."
            )
        return self


class BatteryBESSConfig(BaseModel):
    """Configuration for battery energy storage system (BESS) in DR."""

    enabled: bool = Field(False, description="Enable BESS for DR dispatch")
    capacity_kwh: float = Field(0.0, ge=0.0, description="Total battery capacity (kWh)")
    max_discharge_kw: float = Field(0.0, ge=0.0, description="Maximum discharge rate (kW)")
    min_soc_pct: float = Field(10.0, ge=0.0, le=100.0, description="Minimum state of charge (%)")
    round_trip_efficiency: float = Field(0.90, ge=0.5, le=1.0, description="Round-trip efficiency")
    cycles_per_day_limit: int = Field(2, ge=1, le=10, description="Max charge/discharge cycles per day")


class SolarPVConfig(BaseModel):
    """Configuration for solar PV assets in DR."""

    enabled: bool = Field(False, description="Enable solar PV for DR dispatch")
    capacity_kw: float = Field(0.0, ge=0.0, description="Installed PV capacity (kW)")
    behind_the_meter: bool = Field(True, description="PV is behind-the-meter")
    curtailment_for_grid: bool = Field(False, description="Allow PV curtailment for grid services")


class BackupGeneratorConfig(BaseModel):
    """Configuration for backup generator in DR."""

    enabled: bool = Field(False, description="Enable backup generator for DR dispatch")
    max_runtime_hours: float = Field(4.0, ge=0.0, le=24.0, description="Maximum runtime per event (hours)")
    fuel_type: str = Field("DIESEL", description="Fuel type (DIESEL, NATURAL_GAS, PROPANE)")
    max_capacity_kw: float = Field(0.0, ge=0.0, description="Generator nameplate capacity (kW)")
    emergency_only: bool = Field(False, description="Restrict to emergency DR programs only")


class EVV2BConfig(BaseModel):
    """Configuration for EV vehicle-to-building discharge in DR."""

    enabled: bool = Field(False, description="Enable EV V2B for DR dispatch")
    max_vehicles: int = Field(0, ge=0, le=500, description="Maximum number of V2B-capable vehicles")
    min_departure_soc_pct: float = Field(50.0, ge=0.0, le=100.0, description="Minimum SOC at departure (%)")
    max_discharge_kw_per_vehicle: float = Field(10.0, ge=0.0, le=50.0, description="Max discharge per vehicle (kW)")


class ThermalStorageConfig(BaseModel):
    """Configuration for thermal energy storage in DR."""

    enabled: bool = Field(False, description="Enable thermal storage for DR dispatch")
    type: str = Field("ICE_STORAGE", description="Thermal storage type (ICE_STORAGE, CHILLED_WATER, REFRIGERATION_MASS)")
    capacity_kwh_thermal: float = Field(0.0, ge=0.0, description="Thermal storage capacity (kWh thermal)")
    charge_duration_hours: float = Field(8.0, ge=0.0, le=24.0, description="Full charge duration (hours)")
    discharge_duration_hours: float = Field(6.0, ge=0.0, le=24.0, description="Full discharge duration (hours)")
    pre_cool_duration_minutes: int = Field(60, ge=0, le=240, description="Pre-cooling duration before event (minutes)")
    coast_duration_minutes: int = Field(120, ge=0, le=480, description="Thermal coast duration during event (minutes)")


class DERConfig(BaseModel):
    """Configuration for all distributed energy resource assets.

    Defines which DER types are enabled and their operational constraints
    for coordinated DR dispatch. Only the DER Coordinator Engine uses
    this configuration; facilities without DER assets leave all
    sub-configs disabled.
    """

    battery_bess: BatteryBESSConfig = Field(
        default_factory=BatteryBESSConfig,
        description="Battery energy storage system configuration",
    )
    solar_pv: SolarPVConfig = Field(
        default_factory=SolarPVConfig,
        description="Solar PV asset configuration",
    )
    backup_generator: BackupGeneratorConfig = Field(
        default_factory=BackupGeneratorConfig,
        description="Backup generator configuration",
    )
    ev_v2b: EVV2BConfig = Field(
        default_factory=EVV2BConfig,
        description="EV vehicle-to-building configuration",
    )
    thermal_storage: ThermalStorageConfig = Field(
        default_factory=ThermalStorageConfig,
        description="Thermal energy storage configuration",
    )

    def has_any_der(self) -> bool:
        """Check if any DER asset is enabled."""
        return (
            self.battery_bess.enabled
            or self.solar_pv.enabled
            or self.backup_generator.enabled
            or self.ev_v2b.enabled
            or self.thermal_storage.enabled
        )


class FinancialConfig(BaseModel):
    """Configuration for financial analysis of DR participation.

    Defines energy tariffs, demand charges, discount rates, and
    analysis parameters used to calculate DR revenue, ROI, and
    payback period.
    """

    electricity_rate_eur_per_kwh: float = Field(
        0.12,
        ge=0.01,
        le=1.0,
        description="Electricity volumetric rate (EUR/kWh)",
    )
    demand_charge_eur_per_kw: float = Field(
        12.00,
        ge=0.0,
        le=100.0,
        description="Monthly demand charge (EUR/kW of peak demand)",
    )
    discount_rate: float = Field(
        0.08,
        ge=0.0,
        le=0.25,
        description="Discount rate for NPV calculations",
    )
    analysis_period_years: int = Field(
        10,
        ge=1,
        le=25,
        description="Analysis period for lifecycle cost assessment (years)",
    )
    energy_price_escalation_pct: float = Field(
        3.0,
        ge=0.0,
        le=15.0,
        description="Annual energy price escalation rate (%)",
    )
    currency: str = Field(
        "EUR",
        description="Currency for financial calculations (ISO 4217)",
    )


class CarbonConfig(BaseModel):
    """Configuration for carbon impact assessment of DR events.

    Calculates tCO2e avoided using marginal emission factors during
    peak periods. Supports both location-based and market-based
    Scope 2 accounting per GHG Protocol.
    """

    enabled: bool = Field(
        True,
        description="Enable carbon impact assessment for DR events",
    )
    grid_emission_factor_kgco2_per_kwh: float = Field(
        0.40,
        ge=0.0,
        le=1.5,
        description="Average grid electricity emission factor (kgCO2e/kWh)",
    )
    marginal_emission_factor_kgco2_per_kwh: float = Field(
        0.65,
        ge=0.0,
        le=2.0,
        description="Marginal emission factor during peak periods (kgCO2e/kWh)",
    )
    carbon_price_eur_per_tco2: float = Field(
        85.0,
        ge=0.0,
        le=500.0,
        description="Carbon price for monetised carbon benefit (EUR/tCO2e)",
    )
    sbti_alignment_tracking: bool = Field(
        False,
        description="Track DR carbon reductions against SBTi reduction pathway",
    )
    generator_emission_factor_kgco2_per_kwh: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Emission factor for backup generator fuel (kgCO2e/kWh)",
    )
    market_based_accounting: bool = Field(
        False,
        description="Use market-based emission factors (RECs, green tariffs)",
    )


class ReportingConfig(BaseModel):
    """Configuration for DR report generation and performance dashboards."""

    frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Reporting frequency for DR performance updates",
    )
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML],
        description="Output formats for DR reports",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary with revenue and carbon impact",
    )
    settlement_reports: bool = Field(
        True,
        description="Generate settlement-grade verification reports",
    )
    der_performance_reports: bool = Field(
        False,
        description="Generate DER asset performance reports",
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
    """Security and access control configuration for DR operations."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "dr_manager",
            "energy_manager",
            "facility_manager",
            "der_operator",
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
        description="Enable security audit logging for all DR operations",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored DR data",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_facilities: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of facilities per DR portfolio",
    )
    max_loads_per_facility: int = Field(
        500,
        ge=10,
        le=5000,
        description="Maximum loads to evaluate per facility",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for program data and emission factors (seconds)",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=5000,
        description="Batch size for bulk facility processing",
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


class AuditTrailConfig(BaseModel):
    """Configuration for calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all DR calculations",
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
        description="Track full data lineage from meter through settlement",
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


class DemandResponseConfig(BaseModel):
    """Main configuration for PACK-037 Demand Response Pack.

    This is the root configuration model that contains all sub-configurations
    for demand response management. The facility_type field drives which load
    categories are prioritized, which DR programs are matched, and which DER
    assets are coordinated.
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
        description="Primary facility type for DR scoping",
    )
    country: str = Field(
        "US",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for DR performance tracking",
    )

    # Facility characteristics
    floor_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Total facility floor area in square metres",
    )
    peak_demand_kw: Optional[float] = Field(
        None,
        ge=0,
        description="Annual peak demand (kW)",
    )
    annual_electricity_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual electricity consumption (kWh)",
    )
    operating_hours_per_year: int = Field(
        2500,
        ge=500,
        le=8760,
        description="Annual operating hours",
    )

    # Sub-configurations
    dr_parameters: DRParametersConfig = Field(
        default_factory=DRParametersConfig,
        description="Core demand response parameters",
    )
    load_categories: LoadCategoriesConfig = Field(
        default_factory=LoadCategoriesConfig,
        description="Load category DR participation configuration",
    )
    program_preferences: ProgramPreferencesConfig = Field(
        default_factory=ProgramPreferencesConfig,
        description="DR program selection preferences",
    )
    baseline_preferences: BaselinePreferencesConfig = Field(
        default_factory=BaselinePreferencesConfig,
        description="Customer baseline load calculation preferences",
    )
    dispatch_weights: DispatchWeightsConfig = Field(
        default_factory=DispatchWeightsConfig,
        description="Multi-objective dispatch optimization weights",
    )
    der_configuration: DERConfig = Field(
        default_factory=DERConfig,
        description="Distributed energy resource configuration",
    )
    financial: FinancialConfig = Field(
        default_factory=FinancialConfig,
        description="Financial analysis configuration",
    )
    carbon: CarbonConfig = Field(
        default_factory=CarbonConfig,
        description="Carbon impact assessment configuration",
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
    def validate_healthcare_constraints(self) -> "DemandResponseConfig":
        """Healthcare facilities must use conservative DR parameters."""
        if self.facility_type == FacilityType.HEALTHCARE:
            if self.dr_parameters.max_event_duration_hours > 4:
                logger.warning(
                    "Healthcare facility: max_event_duration_hours exceeds 4. "
                    "Consider reducing for patient safety."
                )
            if self.dr_parameters.comfort_constraints.max_temperature_rise_c > 2.0:
                logger.warning(
                    "Healthcare facility: temperature rise exceeds 2.0C. "
                    "Non-clinical areas should maintain patient comfort."
                )
        return self

    @model_validator(mode="after")
    def validate_data_center_constraints(self) -> "DemandResponseConfig":
        """Data centres must enforce IT uptime constraints."""
        if self.facility_type == FacilityType.DATA_CENTER:
            if self.load_categories.it_equipment.included:
                logger.warning(
                    "Data centre: IT equipment should NOT be included in DR. "
                    "Setting it_equipment.included to False."
                )
                self.load_categories.it_equipment.included = False
        return self

    @model_validator(mode="after")
    def validate_microgrid_der(self) -> "DemandResponseConfig":
        """Microgrid facilities should have at least one DER asset enabled."""
        if self.facility_type == FacilityType.MICROGRID_DER:
            if not self.der_configuration.has_any_der():
                logger.warning(
                    "Microgrid/DER facility type selected but no DER assets enabled. "
                    "Enable at least one DER asset for DER Coordinator Engine."
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

    pack: DemandResponseConfig = Field(
        default_factory=DemandResponseConfig,
        description="Main Demand Response configuration",
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
        "PACK-037-demand-response",
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

        pack_config = DemandResponseConfig(**preset_data)
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

        pack_config = DemandResponseConfig(**config_data)
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
        pack_config = DemandResponseConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with DR_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: DR_PACK_DR_PARAMETERS__MIN_CURTAILMENT_KW=100
                 DR_PACK_FINANCIAL__ELECTRICITY_RATE_EUR_PER_KWH=0.15
        """
        overrides: Dict[str, Any] = {}
        prefix = "DR_PACK_"
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


def validate_config(config: DemandResponseConfig) -> List[str]:
    """Validate a demand response configuration and return any warnings.

    Args:
        config: DemandResponseConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check facility identification
    if not config.facility_name:
        warnings.append(
            "No facility_name configured. Add a facility name for report identification."
        )

    # Check peak demand data
    if config.peak_demand_kw is None:
        warnings.append(
            "No peak_demand_kw configured. Program matching and curtailment "
            "calculations require peak demand data."
        )

    # Check energy data availability
    if config.annual_electricity_kwh is None:
        warnings.append(
            "No annual_electricity_kwh configured. Baseline calculations "
            "require annual electricity consumption."
        )

    # Check curtailment minimum
    if config.dr_parameters.min_curtailment_kw <= 0:
        warnings.append(
            "min_curtailment_kw is zero or negative. DR programs require "
            "positive curtailment capacity."
        )

    # Check dispatch weights
    total_weight = (
        config.dispatch_weights.operational_disruption
        + config.dispatch_weights.comfort_deviation
        + config.dispatch_weights.rebound_effect
        + config.dispatch_weights.target_shortfall
    )
    if abs(total_weight - 1.0) > 0.05:
        warnings.append(
            f"Dispatch weights sum to {total_weight:.2f}, expected ~1.0."
        )

    # Check healthcare constraints
    if config.facility_type == FacilityType.HEALTHCARE:
        if config.program_preferences.risk_tolerance not in (
            RiskTolerance.VERY_LOW, RiskTolerance.LOW
        ):
            warnings.append(
                "Healthcare facilities should use VERY_LOW or LOW risk tolerance."
            )

    # Check data center IT exclusion
    if config.facility_type == FacilityType.DATA_CENTER:
        if config.load_categories.it_equipment.included:
            warnings.append(
                "Data centre: IT equipment should be excluded from DR participation."
            )

    # Check microgrid DER assets
    if config.facility_type == FacilityType.MICROGRID_DER:
        if not config.der_configuration.has_any_der():
            warnings.append(
                "Microgrid/DER facility type but no DER assets enabled."
            )

    # Check carbon configuration
    if config.carbon.enabled and config.carbon.marginal_emission_factor_kgco2_per_kwh <= 0:
        warnings.append(
            "Carbon assessment enabled but marginal emission factor is zero."
        )

    # Check at least one load category is included
    cats = config.load_categories
    any_included = (
        cats.hvac.included or cats.lighting.included or cats.plug_loads.included
        or cats.ev_charging.included or cats.refrigeration.included
        or cats.motors_drives.included or cats.process_loads.included
    )
    if not any_included:
        warnings.append(
            "No load categories are included for DR participation. "
            "At least one load category must be enabled."
        )

    return warnings


def get_default_config(
    facility_type: FacilityType = FacilityType.COMMERCIAL_OFFICE,
) -> DemandResponseConfig:
    """Get default configuration for a given facility type.

    Args:
        facility_type: Facility type to configure for.

    Returns:
        DemandResponseConfig instance with facility-appropriate defaults.
    """
    return DemandResponseConfig(facility_type=facility_type)


def get_facility_info(facility_type: Union[str, FacilityType]) -> Dict[str, Any]:
    """Get detailed information about a facility type.

    Args:
        facility_type: Facility type enum or string value.

    Returns:
        Dictionary with name, typical peak, curtailment potential, and constraints.
    """
    key = facility_type.value if isinstance(facility_type, FacilityType) else facility_type
    return FACILITY_TYPE_INFO.get(
        key,
        {
            "name": key,
            "typical_peak_kw": "Varies",
            "typical_curtailment_pct": "10-25",
            "primary_loads": ["HVAC", "Lighting"],
            "key_constraints": ["Operational continuity"],
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
    "BaselineMethodology",
    "CurtailmentStrategy",
    "DERType",
    "DRProgramType",
    "FacilityType",
    "ISORegion",
    "LoadCriticality",
    "OptimizationAggressiveness",
    "OutputFormat",
    "ReportingFrequency",
    "RiskTolerance",
    # Sub-config models
    "AuditTrailConfig",
    "BackupGeneratorConfig",
    "BaselinePreferencesConfig",
    "BatteryBESSConfig",
    "CarbonConfig",
    "ComfortConstraintConfig",
    "DERConfig",
    "DRParametersConfig",
    "DispatchWeightsConfig",
    "EVV2BConfig",
    "FinancialConfig",
    "LoadCategoriesConfig",
    "LoadCategoryConfig",
    "NotificationConfig",
    "OperationalConstraintConfig",
    "PerformanceConfig",
    "ProgramPreferencesConfig",
    "ReportingConfig",
    "SecurityConfig",
    "SolarPVConfig",
    "ThermalStorageConfig",
    # Main config models
    "DemandResponseConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "DEFAULT_FINANCIAL_PARAMS",
    "FACILITY_TYPE_INFO",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_facility_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
