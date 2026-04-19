"""
PACK-031 Industrial Energy Audit Pack - Configuration Manager

This module implements the IndustrialEnergyAuditConfig and PackConfig classes
that load, merge, and validate all configuration for the Industrial Energy Audit
Pack. It provides comprehensive Pydantic v2 models for every aspect of industrial
energy auditing: baseline modeling, equipment-level analysis, steam systems,
compressed air, waste heat recovery, lighting, HVAC, benchmarking, and
ISO 50001 alignment.

Industry Sectors:
    - MANUFACTURING: Discrete manufacturing (automotive parts, electronics, machining)
    - PROCESS_INDUSTRY: Chemical, petrochemical, pharmaceutical continuous processes
    - FOOD_BEVERAGE: Food processing, breweries, dairies
    - DATA_CENTER: Data centers, server farms, colocation facilities
    - WAREHOUSE_LOGISTICS: Warehouses, distribution centers, cold storage
    - AUTOMOTIVE: Automotive OEM and Tier 1 suppliers (paint, body, assembly, press)
    - STEEL_METALS: Steel mills, foundries, aluminum smelting, metal fabrication
    - PHARMACEUTICAL: Pharmaceutical manufacturing and cleanroom facilities
    - CHEMICAL: Chemical processing and specialty chemicals
    - CEMENT: Cement and clinker production
    - GLASS: Glass manufacturing (flat, container, specialty)
    - PAPER_PULP: Paper, pulp, and board manufacturing
    - TEXTILE: Textile manufacturing and dyeing
    - PLASTICS: Plastics processing (injection molding, extrusion, blow molding)
    - OTHER: Other industrial sectors

Facility Tiers:
    - LARGE_ENTERPRISE: Large industrial facilities (>250 employees or >EUR 50M revenue)
    - MID_MARKET: Mid-sized industrial facilities (50-250 employees)
    - SME: Small and medium industrial enterprises (<50 employees)

Audit Levels:
    - WALK_THROUGH: Preliminary visual inspection and quick assessment (ASHRAE Level I)
    - DETAILED: Comprehensive data collection and analysis (ASHRAE Level II / EN 16247)
    - INVESTMENT_GRADE: Rigorous engineering analysis for capital projects (ASHRAE Level III)

Energy Carriers:
    - ELECTRICITY, NATURAL_GAS, FUEL_OIL, LPG, COAL, BIOMASS, STEAM
    - COMPRESSED_AIR, DISTRICT_HEATING, DISTRICT_COOLING, SOLAR_THERMAL, BIOGAS

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (manufacturing_plant / process_industry / food_beverage /
       data_center / warehouse_logistics / automotive_manufacturing /
       steel_metals / sme_industrial)
    3. Environment overrides (ENERGY_AUDIT_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - EN 16247: Energy audits (Parts 1-5)
    - ISO 50001: Energy management systems
    - EED: EU Energy Efficiency Directive 2023/1791 (Article 8)
    - EU ETS: Emissions Trading System (for heavy industry)
    - BAT/BREF: Best Available Techniques Reference Documents
    - IEC 60034-30-1: Motor efficiency classes (IE1-IE5)
    - Ecodesign Regulation (EU) 2019/1781: Electric motors and VSD

Example:
    >>> config = PackConfig.from_preset("manufacturing_plant")
    >>> print(config.pack.industry_sector)
    IndustrySector.MANUFACTURING
    >>> print(config.pack.compressed_air.enabled)
    True
    >>> print(config.pack.baseline.r_squared_threshold)
    0.75
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
# Enums - Industrial energy audit enumeration types
# =============================================================================


class IndustrySector(str, Enum):
    """Industrial sector classification for energy audit scoping."""

    MANUFACTURING = "MANUFACTURING"
    PROCESS_INDUSTRY = "PROCESS_INDUSTRY"
    FOOD_BEVERAGE = "FOOD_BEVERAGE"
    DATA_CENTER = "DATA_CENTER"
    WAREHOUSE_LOGISTICS = "WAREHOUSE_LOGISTICS"
    AUTOMOTIVE = "AUTOMOTIVE"
    STEEL_METALS = "STEEL_METALS"
    PHARMACEUTICAL = "PHARMACEUTICAL"
    CHEMICAL = "CHEMICAL"
    CEMENT = "CEMENT"
    GLASS = "GLASS"
    PAPER_PULP = "PAPER_PULP"
    TEXTILE = "TEXTILE"
    PLASTICS = "PLASTICS"
    OTHER = "OTHER"


class FacilityTier(str, Enum):
    """Facility size tier classification."""

    LARGE_ENTERPRISE = "LARGE_ENTERPRISE"
    MID_MARKET = "MID_MARKET"
    SME = "SME"


class EnergyCarrier(str, Enum):
    """Energy carrier types tracked in the audit."""

    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    FUEL_OIL = "FUEL_OIL"
    LPG = "LPG"
    COAL = "COAL"
    BIOMASS = "BIOMASS"
    STEAM = "STEAM"
    COMPRESSED_AIR = "COMPRESSED_AIR"
    DISTRICT_HEATING = "DISTRICT_HEATING"
    DISTRICT_COOLING = "DISTRICT_COOLING"
    SOLAR_THERMAL = "SOLAR_THERMAL"
    BIOGAS = "BIOGAS"


class AuditLevel(str, Enum):
    """Energy audit depth level per EN 16247 / ASHRAE classification."""

    WALK_THROUGH = "WALK_THROUGH"
    DETAILED = "DETAILED"
    INVESTMENT_GRADE = "INVESTMENT_GRADE"


class MotorEfficiencyClass(str, Enum):
    """IEC 60034-30-1 motor efficiency classes."""

    IE1 = "IE1"
    IE2 = "IE2"
    IE3 = "IE3"
    IE4 = "IE4"
    IE5 = "IE5"


class NormalizationMethod(str, Enum):
    """Baseline normalization methods for energy performance."""

    DEGREE_DAYS = "DEGREE_DAYS"
    PRODUCTION_VOLUME = "PRODUCTION_VOLUME"
    MULTI_VARIABLE = "MULTI_VARIABLE"
    NONE = "NONE"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency."""

    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    QUARTERLY = "QUARTERLY"
    MONTHLY = "MONTHLY"
    WEEKLY = "WEEKLY"
    DAILY = "DAILY"
    REAL_TIME = "REAL_TIME"


class ComplianceStatus(str, Enum):
    """Overall compliance status for energy audit requirements."""

    COMPLIANT = "COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_ASSESSED = "NOT_ASSESSED"
    EXEMPT = "EXEMPT"


class OutputFormat(str, Enum):
    """Output format for audit reports."""

    PDF = "PDF"
    XLSX = "XLSX"
    JSON = "JSON"
    HTML = "HTML"


class EnPIType(str, Enum):
    """Energy Performance Indicator types for ISO 50001."""

    SEC = "SEC"  # Specific Energy Consumption (kWh/unit)
    PUE = "PUE"  # Power Usage Effectiveness (data centers)
    COP = "COP"  # Coefficient of Performance (cooling/heat pumps)
    SPECIFIC_POWER = "SPECIFIC_POWER"  # kW per unit output (compressors)
    THERMAL_EFFICIENCY = "THERMAL_EFFICIENCY"  # Boiler/furnace efficiency
    LPD = "LPD"  # Lighting Power Density (W/m2)
    EUI = "EUI"  # Energy Use Intensity (kWh/m2/yr)
    CUSTOM = "CUSTOM"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Industry sector display names, NACE codes, and typical energy profiles
SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "MANUFACTURING": {
        "name": "Discrete Manufacturing",
        "nace": "C25-C28",
        "typical_energy_split": "Electricity 60-70%, Gas 20-30%, Other 5-10%",
        "key_systems": ["Motors", "Compressed Air", "Process Heat", "HVAC", "Lighting"],
        "typical_sec_range_kwh_per_tonne": "500-2000",
        "bat_bref": "Smitheries and Foundries BREF",
    },
    "PROCESS_INDUSTRY": {
        "name": "Process Industry (Chemical/Petrochem)",
        "nace": "C20-C21",
        "typical_energy_split": "Gas 40-60%, Electricity 25-40%, Steam 10-20%",
        "key_systems": ["Steam", "Process Heat", "Waste Heat Recovery", "Motors", "Cooling"],
        "typical_sec_range_kwh_per_tonne": "Varies by product (200-10000)",
        "bat_bref": "Common Waste Water and Waste Gas BREF, Large Volume Organic Chemicals BREF",
    },
    "FOOD_BEVERAGE": {
        "name": "Food & Beverage Processing",
        "nace": "C10-C11",
        "typical_energy_split": "Gas 40-55%, Electricity 35-50%, Steam 5-15%",
        "key_systems": ["Refrigeration", "Steam", "Compressed Air", "Process Heat", "Drying"],
        "typical_sec_range_kwh_per_tonne": "200-1500",
        "bat_bref": "Food, Drink and Milk Industries BREF",
    },
    "DATA_CENTER": {
        "name": "Data Centers & Server Farms",
        "nace": "J63.11",
        "typical_energy_split": "Electricity 95-100%, District Cooling 0-5%",
        "key_systems": ["IT Load", "Cooling", "UPS", "Lighting", "Power Distribution"],
        "typical_sec_range_kwh_per_tonne": "N/A (PUE metric: 1.1-2.0)",
        "bat_bref": "EN 50600 Data Centre Series",
    },
    "WAREHOUSE_LOGISTICS": {
        "name": "Warehouses & Distribution Centers",
        "nace": "H52.10",
        "typical_energy_split": "Electricity 60-80%, Gas 15-30%, Diesel 5-10%",
        "key_systems": ["Lighting", "HVAC", "Refrigeration", "Dock Doors", "MHE"],
        "typical_sec_range_kwh_per_tonne": "10-80 per m2/yr (EUI metric)",
        "bat_bref": "N/A",
    },
    "AUTOMOTIVE": {
        "name": "Automotive Manufacturing",
        "nace": "C29",
        "typical_energy_split": "Gas 45-55%, Electricity 40-50%, Steam 5-10%",
        "key_systems": ["Paint Shop", "Body Shop", "Assembly", "Press Shop", "Compressed Air"],
        "typical_sec_range_kwh_per_tonne": "800-3000 per vehicle",
        "bat_bref": "Smitheries and Foundries BREF, Surface Treatment BREF",
    },
    "STEEL_METALS": {
        "name": "Steel Mills & Metal Fabrication",
        "nace": "C24",
        "typical_energy_split": "Coal/Coke 40-60%, Electricity 25-40%, Gas 10-20%",
        "key_systems": ["EAF/BF", "Rolling Mill", "Heat Treatment", "Compressed Air", "Cooling"],
        "typical_sec_range_kwh_per_tonne": "400-6000",
        "bat_bref": "Iron and Steel Production BREF, Ferrous Metals Processing BREF",
    },
    "PHARMACEUTICAL": {
        "name": "Pharmaceutical Manufacturing",
        "nace": "C21",
        "typical_energy_split": "Electricity 50-65%, Gas 25-35%, Steam 5-15%",
        "key_systems": ["HVAC/Cleanroom", "Steam", "Compressed Air", "Cooling", "Process"],
        "typical_sec_range_kwh_per_tonne": "5000-50000 (high value, low mass)",
        "bat_bref": "Common Waste Water and Waste Gas BREF",
    },
    "CHEMICAL": {
        "name": "Chemical Processing",
        "nace": "C20",
        "typical_energy_split": "Gas 35-55%, Electricity 30-45%, Steam 10-20%",
        "key_systems": ["Steam", "Reactors", "Distillation", "Waste Heat Recovery", "Cooling"],
        "typical_sec_range_kwh_per_tonne": "500-8000",
        "bat_bref": "Large Volume Organic Chemicals BREF, Large Volume Inorganic Chemicals BREF",
    },
    "CEMENT": {
        "name": "Cement & Clinker Production",
        "nace": "C23.51",
        "typical_energy_split": "Coal/Pet Coke 70-80%, Electricity 15-25%, AFR 5-15%",
        "key_systems": ["Kiln", "Raw Mill", "Cement Mill", "Clinker Cooler", "Fans"],
        "typical_sec_range_kwh_per_tonne": "3000-5000 (thermal) + 90-130 (electrical)",
        "bat_bref": "Cement, Lime and Magnesium Oxide BREF",
    },
    "GLASS": {
        "name": "Glass Manufacturing",
        "nace": "C23.1",
        "typical_energy_split": "Gas 70-80%, Electricity 15-25%, Fuel Oil 0-5%",
        "key_systems": ["Melting Furnace", "Forming", "Annealing", "Batch Plant", "Cullet"],
        "typical_sec_range_kwh_per_tonne": "4000-8000",
        "bat_bref": "Manufacture of Glass BREF",
    },
    "PAPER_PULP": {
        "name": "Paper, Pulp & Board",
        "nace": "C17",
        "typical_energy_split": "Gas 35-50%, Biomass 20-35%, Electricity 20-30%",
        "key_systems": ["Paper Machine", "Drying", "Pulping", "Steam", "Compressed Air"],
        "typical_sec_range_kwh_per_tonne": "1500-6000",
        "bat_bref": "Production of Pulp, Paper and Board BREF",
    },
    "TEXTILE": {
        "name": "Textile Manufacturing",
        "nace": "C13",
        "typical_energy_split": "Gas 40-55%, Electricity 35-50%, Steam 5-15%",
        "key_systems": ["Dyeing/Finishing", "Drying", "Steam", "Compressed Air", "HVAC"],
        "typical_sec_range_kwh_per_tonne": "1000-5000",
        "bat_bref": "Textiles Industry BREF",
    },
    "PLASTICS": {
        "name": "Plastics Processing",
        "nace": "C22",
        "typical_energy_split": "Electricity 70-85%, Gas 10-20%, Other 5-10%",
        "key_systems": ["Injection Molding", "Extrusion", "Cooling", "Compressed Air", "Drying"],
        "typical_sec_range_kwh_per_tonne": "500-3000",
        "bat_bref": "Polymers BREF",
    },
    "OTHER": {
        "name": "Other Industrial Sector",
        "nace": "C",
        "typical_energy_split": "Varies",
        "key_systems": ["Motors", "HVAC", "Lighting", "Compressed Air"],
        "typical_sec_range_kwh_per_tonne": "Varies",
        "bat_bref": "Energy Efficiency BREF",
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "manufacturing_plant": "Discrete manufacturing (automotive parts, electronics, machining)",
    "process_industry": "Chemical, petrochemical, pharmaceutical continuous processes",
    "food_beverage": "Food processing, breweries, dairies with refrigeration and steam",
    "data_center": "Data centers and server farms with PUE optimization",
    "warehouse_logistics": "Warehouses, distribution centers, cold storage facilities",
    "automotive_manufacturing": "Automotive OEM and Tier 1 suppliers (paint, body, assembly)",
    "steel_metals": "Steel mills, foundries, aluminum smelting, metal fabrication",
    "sme_industrial": "Simplified SME configuration focused on quick wins",
}

# Motor efficiency minimum requirements by region and date
MOTOR_EFFICIENCY_REQUIREMENTS: Dict[str, Dict[str, str]] = {
    "EU": {
        "0.75_to_375kW_2021": "IE3",
        "0.12_to_0.75kW_2023": "IE2",
        "75_to_200kW_2023": "IE4",
    },
    "US": {
        "general_2023": "NEMA_PREMIUM",
    },
}

# Compressed air specific power benchmarks (kW per m3/min at 7 bar)
COMPRESSED_AIR_BENCHMARKS: Dict[str, float] = {
    "best_practice": 5.5,
    "good": 6.5,
    "average": 7.5,
    "poor": 9.0,
}

# Steam system efficiency benchmarks
STEAM_BENCHMARKS: Dict[str, float] = {
    "boiler_efficiency_best": 0.92,
    "boiler_efficiency_good": 0.87,
    "boiler_efficiency_average": 0.82,
    "condensate_return_target_pct": 85.0,
    "blowdown_target_pct": 5.0,
    "steam_trap_failure_rate_target_pct": 5.0,
}

# Lighting Power Density standards (W/m2) per EN 12464-1
LPD_STANDARDS: Dict[str, float] = {
    "office": 8.0,
    "warehouse_low_bay": 6.0,
    "warehouse_high_bay": 8.0,
    "manufacturing_fine": 15.0,
    "manufacturing_general": 10.0,
    "cleanroom": 18.0,
    "loading_dock": 5.0,
    "outdoor_yard": 3.0,
}

# PUE benchmarks for data centers
PUE_BENCHMARKS: Dict[str, float] = {
    "world_class": 1.1,
    "efficient": 1.3,
    "average": 1.6,
    "inefficient": 2.0,
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class BaselineConfig(BaseModel):
    """Configuration for energy baseline modeling.

    Baselines are modeled using regression analysis against key driving
    variables (production volume, degree days, occupancy). The model must
    meet minimum statistical quality thresholds (R-squared, CV-RMSE) to be
    accepted for measurement and verification (M&V) per IPMVP.
    """

    min_months: int = Field(
        12,
        ge=6,
        le=60,
        description="Minimum months of historical data for baseline model",
    )
    r_squared_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Minimum R-squared for baseline regression acceptance",
    )
    cv_rmse_threshold: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Maximum CV(RMSE) for baseline regression acceptance",
    )
    normalization_method: NormalizationMethod = Field(
        NormalizationMethod.PRODUCTION_VOLUME,
        description="Primary normalization variable for baseline model",
    )
    secondary_variables: List[str] = Field(
        default_factory=lambda: ["degree_days"],
        description="Secondary normalization variables (degree_days, occupancy, etc.)",
    )
    granularity: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Data granularity for baseline model (monthly recommended)",
    )
    baseline_year: Optional[int] = Field(
        None,
        ge=2015,
        le=2030,
        description="Baseline reference year (auto-detected if None)",
    )
    energy_carriers: List[EnergyCarrier] = Field(
        default_factory=lambda: [
            EnergyCarrier.ELECTRICITY,
            EnergyCarrier.NATURAL_GAS,
        ],
        description="Energy carriers included in baseline model",
    )
    ipmvp_option: str = Field(
        "C",
        description="IPMVP M&V option: A (partially measured), B (retrofit isolation), "
        "C (whole facility), D (calibrated simulation)",
    )

    @field_validator("ipmvp_option")
    @classmethod
    def validate_ipmvp_option(cls, v: str) -> str:
        """Validate IPMVP option is A, B, C, or D."""
        valid = {"A", "B", "C", "D"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid IPMVP option: {v}. Must be one of {valid}.")
        return v.upper()


class AuditConfig(BaseModel):
    """Configuration for energy audit execution parameters.

    Governs the depth and scope of the energy audit, including EN 16247
    compliance and EED Article 8 requirements for large enterprises.
    """

    default_audit_level: AuditLevel = Field(
        AuditLevel.DETAILED,
        description="Default audit depth level",
    )
    en16247_compliance: bool = Field(
        True,
        description="Require EN 16247 compliance in audit methodology",
    )
    en16247_parts: List[int] = Field(
        default_factory=lambda: [1, 2, 3],
        description="EN 16247 parts to comply with (1=General, 2=Buildings, "
        "3=Processes, 4=Transport, 5=Competence)",
    )
    eed_article_8: bool = Field(
        True,
        description="Enable EED Article 8 large enterprise audit requirement",
    )
    schedule_months: int = Field(
        48,
        ge=12,
        le=60,
        description="Audit recurrence interval in months (EED requires max 48 months)",
    )
    significant_energy_use_threshold_pct: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Threshold (%) of total energy to classify as Significant Energy Use (SEU)",
    )
    minimum_coverage_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum percentage of total energy consumption covered by audit",
    )
    measure_payback_threshold_years: float = Field(
        5.0,
        ge=0.5,
        le=20.0,
        description="Maximum simple payback period (years) for recommended measures",
    )
    include_no_cost_measures: bool = Field(
        True,
        description="Include zero/low-cost operational measures in recommendations",
    )
    include_behavioral_measures: bool = Field(
        True,
        description="Include behavioral and awareness measures",
    )

    @field_validator("en16247_parts")
    @classmethod
    def validate_en16247_parts(cls, v: List[int]) -> List[int]:
        """Validate EN 16247 part numbers."""
        invalid = [p for p in v if p < 1 or p > 5]
        if invalid:
            raise ValueError(f"Invalid EN 16247 parts: {invalid}. Must be 1-5.")
        return sorted(set(v))


class EquipmentConfig(BaseModel):
    """Configuration for equipment-level energy analysis.

    Covers motor systems, pumps, fans, and general rotating equipment
    analysis per IEC 60034-30-1 and Ecodesign Regulation (EU) 2019/1781.
    """

    enabled: bool = Field(
        True,
        description="Enable equipment-level energy analysis",
    )
    motor_efficiency_min_class: MotorEfficiencyClass = Field(
        MotorEfficiencyClass.IE3,
        description="Minimum motor efficiency class for replacement recommendations",
    )
    motor_inventory_enabled: bool = Field(
        True,
        description="Build complete motor inventory (>0.75 kW)",
    )
    motor_load_profiling: bool = Field(
        True,
        description="Perform motor load profiling to identify oversized motors",
    )
    pump_benchmark_enabled: bool = Field(
        True,
        description="Enable pump system benchmarking per Europump/HI methodology",
    )
    pump_system_assessment: bool = Field(
        True,
        description="Assess complete pump systems (motor + pump + piping + controls)",
    )
    fan_benchmark_enabled: bool = Field(
        True,
        description="Enable fan system benchmarking",
    )
    vsd_retrofit_analysis: bool = Field(
        True,
        description="Analyse Variable Speed Drive retrofit opportunities",
    )
    vsd_min_savings_threshold_pct: float = Field(
        15.0,
        ge=5.0,
        le=50.0,
        description="Minimum energy savings (%) to recommend VSD retrofit",
    )
    power_quality_monitoring: bool = Field(
        False,
        description="Include power quality analysis (harmonics, power factor)",
    )
    equipment_age_threshold_years: int = Field(
        15,
        ge=5,
        le=40,
        description="Equipment age threshold (years) to flag for replacement assessment",
    )


class SteamConfig(BaseModel):
    """Configuration for steam system energy analysis.

    Covers boilers, distribution, condensate return, steam traps, and
    insulation assessment per best practice guidelines.
    """

    enabled: bool = Field(
        True,
        description="Enable steam system analysis",
    )
    boiler_efficiency_assessment: bool = Field(
        True,
        description="Assess boiler combustion and thermal efficiency",
    )
    boiler_efficiency_target_pct: float = Field(
        87.0,
        ge=70.0,
        le=98.0,
        description="Target boiler efficiency (%)",
    )
    trap_survey_enabled: bool = Field(
        True,
        description="Enable steam trap survey",
    )
    trap_survey_interval_months: int = Field(
        12,
        ge=3,
        le=36,
        description="Steam trap survey interval in months",
    )
    insulation_assessment_enabled: bool = Field(
        True,
        description="Assess insulation condition on pipes, valves, and flanges",
    )
    condensate_return_tracking: bool = Field(
        True,
        description="Track condensate return rate and identify improvement opportunities",
    )
    condensate_return_target_pct: float = Field(
        85.0,
        ge=50.0,
        le=100.0,
        description="Target condensate return rate (%)",
    )
    blowdown_optimization: bool = Field(
        True,
        description="Assess boiler blowdown rate and heat recovery potential",
    )
    blowdown_target_pct: float = Field(
        5.0,
        ge=1.0,
        le=15.0,
        description="Target blowdown rate (%)",
    )
    flash_steam_recovery: bool = Field(
        True,
        description="Assess flash steam recovery from condensate",
    )
    steam_pressure_optimization: bool = Field(
        False,
        description="Analyse steam header pressure optimization potential",
    )
    chp_assessment: bool = Field(
        False,
        description="Assess Combined Heat and Power (CHP/cogeneration) potential",
    )


class CompressedAirConfig(BaseModel):
    """Configuration for compressed air system analysis.

    Covers compressors, distribution, leak detection, pressure optimization,
    and specific power benchmarking.
    """

    enabled: bool = Field(
        True,
        description="Enable compressed air system analysis",
    )
    leak_survey_enabled: bool = Field(
        True,
        description="Enable ultrasonic leak detection survey",
    )
    leak_survey_interval_months: int = Field(
        6,
        ge=3,
        le=24,
        description="Leak survey interval in months",
    )
    leak_rate_target_pct: float = Field(
        10.0,
        ge=2.0,
        le=30.0,
        description="Target air leak rate as percentage of total generation",
    )
    specific_power_target_kw_per_m3_min: float = Field(
        6.5,
        ge=4.0,
        le=12.0,
        description="Target specific power (kW per m3/min at 7 bar reference)",
    )
    pressure_optimization: bool = Field(
        True,
        description="Analyse system pressure reduction potential",
    )
    pressure_band_bar: Tuple[float, float] = Field(
        default=(6.0, 7.0),
        description="Target pressure band (min, max) in bar gauge",
    )
    heat_recovery_assessment: bool = Field(
        True,
        description="Assess compressor waste heat recovery for space/process heating",
    )
    control_optimization: bool = Field(
        True,
        description="Analyse compressor sequencing and control strategy",
    )
    demand_profiling: bool = Field(
        True,
        description="Profile compressed air demand patterns (24h, weekly)",
    )
    inappropriate_use_audit: bool = Field(
        True,
        description="Audit inappropriate uses of compressed air (cooling, cleaning)",
    )
    dryer_efficiency: bool = Field(
        True,
        description="Assess air dryer type and efficiency",
    )


class WasteHeatConfig(BaseModel):
    """Configuration for waste heat recovery analysis.

    Covers identification and quantification of waste heat sources,
    pinch analysis, and recovery technology matching.
    """

    enabled: bool = Field(
        True,
        description="Enable waste heat recovery analysis",
    )
    min_temperature_c: float = Field(
        60.0,
        ge=25.0,
        le=500.0,
        description="Minimum temperature (deg C) for waste heat source identification",
    )
    pinch_analysis_enabled: bool = Field(
        False,
        description="Enable thermal pinch analysis for heat integration",
    )
    heat_exchanger_assessment: bool = Field(
        True,
        description="Assess heat exchanger effectiveness and fouling",
    )
    orc_screening: bool = Field(
        False,
        description="Screen for Organic Rankine Cycle (ORC) waste heat to power potential",
    )
    orc_min_temperature_c: float = Field(
        150.0,
        ge=80.0,
        le=600.0,
        description="Minimum source temperature (deg C) for ORC screening",
    )
    heat_pump_assessment: bool = Field(
        True,
        description="Assess industrial heat pump upgrade opportunities",
    )
    heat_pump_max_sink_temperature_c: float = Field(
        90.0,
        ge=40.0,
        le=160.0,
        description="Maximum heat pump sink temperature (deg C)",
    )
    cascade_analysis: bool = Field(
        True,
        description="Analyse heat cascade opportunities between processes",
    )
    flue_gas_recovery: bool = Field(
        True,
        description="Assess flue gas heat recovery potential (economizer/condensing)",
    )
    thermal_storage_screening: bool = Field(
        False,
        description="Screen for thermal energy storage opportunities",
    )


class LightingConfig(BaseModel):
    """Configuration for lighting system analysis.

    Covers lighting power density assessment, LED retrofit analysis,
    controls optimization, and daylighting integration.
    """

    enabled: bool = Field(
        True,
        description="Enable lighting system analysis",
    )
    lpd_standard: str = Field(
        "EN_12464_1",
        description="Lighting standard for LPD benchmarking: EN_12464_1, ASHRAE_90_1",
    )
    lpd_target_w_per_m2: Optional[float] = Field(
        None,
        ge=1.0,
        le=30.0,
        description="Custom LPD target (W/m2); uses standard defaults if None",
    )
    led_retrofit_analysis: bool = Field(
        True,
        description="Analyse LED retrofit potential for non-LED luminaires",
    )
    controls_assessment: bool = Field(
        True,
        description="Assess lighting controls (occupancy sensors, daylight dimming, scheduling)",
    )
    daylighting_integration: bool = Field(
        False,
        description="Assess daylighting integration potential (skylights, light shelves)",
    )
    emergency_lighting_audit: bool = Field(
        False,
        description="Include emergency lighting in energy audit",
    )
    outdoor_lighting: bool = Field(
        True,
        description="Include outdoor/yard lighting in analysis",
    )
    high_bay_assessment: bool = Field(
        True,
        description="Detailed assessment for high-bay industrial lighting",
    )
    lux_level_measurement: bool = Field(
        True,
        description="Perform lux level measurements to verify over/under-illumination",
    )


class HVACConfig(BaseModel):
    """Configuration for HVAC system analysis.

    Covers heating, ventilation, air conditioning, and air handling
    systems including VSD analysis, heat recovery, and economizers.
    """

    enabled: bool = Field(
        True,
        description="Enable HVAC system analysis",
    )
    vsd_analysis: bool = Field(
        True,
        description="Analyse Variable Speed Drive potential for fans and pumps",
    )
    economizer_analysis: bool = Field(
        True,
        description="Assess air-side and water-side economizer potential",
    )
    heat_recovery: bool = Field(
        True,
        description="Assess heat recovery from exhaust air and process ventilation",
    )
    ahu_optimization: bool = Field(
        True,
        description="Analyse Air Handling Unit scheduling and setpoint optimization",
    )
    chiller_efficiency: bool = Field(
        True,
        description="Assess chiller COP and capacity optimization",
    )
    free_cooling_assessment: bool = Field(
        True,
        description="Assess free cooling / economizer hours potential",
    )
    building_envelope: bool = Field(
        False,
        description="Include building envelope assessment (insulation, glazing, air tightness)",
    )
    bms_optimization: bool = Field(
        True,
        description="Assess Building Management System optimization opportunities",
    )
    setpoint_review: bool = Field(
        True,
        description="Review temperature and humidity setpoints vs. process requirements",
    )
    cleanroom_hvac: bool = Field(
        False,
        description="Include cleanroom HVAC analysis (ISO class-specific)",
    )
    process_ventilation: bool = Field(
        True,
        description="Include process ventilation / local exhaust ventilation",
    )


class BenchmarkConfig(BaseModel):
    """Configuration for energy performance benchmarking.

    Benchmarks facility against sector peers, Best Available Techniques
    (BAT-AEL), and internal site-to-site comparison.
    """

    enabled: bool = Field(
        True,
        description="Enable energy benchmarking",
    )
    peer_group_size_min: int = Field(
        5,
        ge=3,
        le=50,
        description="Minimum peer group size for meaningful benchmark",
    )
    bat_ael_comparison: bool = Field(
        True,
        description="Compare against BAT-AEL (Best Available Techniques Associated "
        "Emission/Energy Levels) from BREF documents",
    )
    kpi_set: List[str] = Field(
        default_factory=lambda: [
            "sec_kwh_per_tonne",
            "electricity_intensity",
            "thermal_intensity",
            "compressed_air_specific_power",
            "steam_system_efficiency",
            "lighting_power_density",
            "enpi_trend",
        ],
        description="KPI set for benchmarking",
    )
    site_to_site_comparison: bool = Field(
        True,
        description="Enable internal site-to-site comparison for multi-facility operations",
    )
    sector_benchmark_source: str = Field(
        "ODYSSEE_MURE",
        description="Sector benchmark data source: ODYSSEE_MURE, IEA, ENERGY_STAR, NATIONAL",
    )
    percentile_tracking: bool = Field(
        True,
        description="Track percentile ranking within peer group",
    )
    gap_to_best_practice: bool = Field(
        True,
        description="Calculate gap to sector best practice",
    )
    trend_analysis_years: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of years for trend analysis",
    )


class ISO50001Config(BaseModel):
    """Configuration for ISO 50001 Energy Management System alignment.

    Tracks EnPI definitions, energy baselines, SEU identification, and
    continuous improvement per Plan-Do-Check-Act.
    """

    enabled: bool = Field(
        False,
        description="Enable ISO 50001 alignment tracking",
    )
    certification_target_date: Optional[str] = Field(
        None,
        description="Target date for ISO 50001 certification (ISO 8601 format)",
    )
    enpi_tracking_frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Frequency for Energy Performance Indicator (EnPI) tracking",
    )
    enpi_types: List[EnPIType] = Field(
        default_factory=lambda: [EnPIType.SEC, EnPIType.THERMAL_EFFICIENCY],
        description="EnPI types to track",
    )
    energy_review_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="Frequency for comprehensive energy review",
    )
    management_review_frequency: ReportingFrequency = Field(
        ReportingFrequency.SEMI_ANNUAL,
        description="Frequency for management review meetings",
    )
    seu_register_enabled: bool = Field(
        True,
        description="Maintain Significant Energy Use (SEU) register",
    )
    action_plan_tracking: bool = Field(
        True,
        description="Track energy improvement action plan progress",
    )
    competence_matrix_enabled: bool = Field(
        False,
        description="Track energy team competence matrix per Clause 7.2",
    )
    internal_audit_enabled: bool = Field(
        True,
        description="Enable ISO 50001 internal audit scheduling",
    )
    eed_exemption_tracking: bool = Field(
        True,
        description="Track EED Article 8 exemption status (ISO 50001 certified = exempt)",
    )


class EEDConfig(BaseModel):
    """Configuration for EU Energy Efficiency Directive compliance.

    Tracks EED Article 8 mandatory audit requirements for large enterprises,
    including threshold assessment and audit scheduling.
    """

    enabled: bool = Field(
        True,
        description="Enable EED compliance tracking",
    )
    large_enterprise_threshold_employees: int = Field(
        250,
        ge=50,
        le=1000,
        description="Employee count threshold for large enterprise classification",
    )
    large_enterprise_threshold_revenue_eur: float = Field(
        50000000.0,
        ge=1000000.0,
        description="Annual revenue (EUR) threshold for large enterprise classification",
    )
    large_enterprise_threshold_balance_sheet_eur: float = Field(
        43000000.0,
        ge=1000000.0,
        description="Balance sheet total (EUR) threshold for large enterprise classification",
    )
    mandatory_audit: bool = Field(
        True,
        description="Facility meets EED mandatory audit threshold",
    )
    audit_interval_months: int = Field(
        48,
        ge=12,
        le=60,
        description="Maximum interval between mandatory audits (months)",
    )
    minimum_energy_covered_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum percentage of energy consumption covered by audit",
    )
    member_state: str = Field(
        "DE",
        description="EU member state for national transposition requirements (ISO 3166-1 alpha-2)",
    )
    national_registry_submission: bool = Field(
        False,
        description="Enable submission to national energy audit registry (BAFA for DE, etc.)",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_facilities: int = Field(
        50,
        ge=1,
        le=500,
        description="Maximum number of facilities per audit run",
    )
    max_equipment_items: int = Field(
        10000,
        ge=100,
        le=100000,
        description="Maximum equipment items per facility",
    )
    max_meters: int = Field(
        500,
        ge=10,
        le=10000,
        description="Maximum sub-meters per facility",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for emission factors and reference data (seconds)",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Batch size for bulk data processing",
    )
    calculation_timeout_seconds: int = Field(
        300,
        ge=30,
        le=3600,
        description="Timeout for individual engine calculations (seconds)",
    )
    parallel_engines: int = Field(
        4,
        ge=1,
        le=16,
        description="Maximum number of engines running in parallel",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "energy_auditor",
            "facility_manager",
            "energy_manager",
            "engineer",
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
        description="Enable PII redaction in exported reports",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored data",
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
        description="Track full data lineage from source to output",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for third-party energy auditors",
    )
    measure_tracking: bool = Field(
        True,
        description="Track energy conservation measure (ECM) implementation status",
    )


class ReportingConfig(BaseModel):
    """Configuration for audit report generation."""

    executive_summary_enabled: bool = Field(
        True,
        description="Generate executive summary with key findings and quick wins",
    )
    detailed_report_enabled: bool = Field(
        True,
        description="Generate detailed audit report per EN 16247",
    )
    equipment_register_enabled: bool = Field(
        True,
        description="Generate equipment energy register",
    )
    ecm_register_enabled: bool = Field(
        True,
        description="Generate Energy Conservation Measures (ECM) register",
    )
    energy_balance_enabled: bool = Field(
        True,
        description="Generate Sankey diagram / energy balance",
    )
    benchmark_report_enabled: bool = Field(
        True,
        description="Generate benchmarking report",
    )
    iso50001_gap_report_enabled: bool = Field(
        False,
        description="Generate ISO 50001 gap analysis report",
    )
    output_formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.PDF, OutputFormat.XLSX],
        description="Output formats for reports",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class IndustrialEnergyAuditConfig(BaseModel):
    """Main configuration for PACK-031 Industrial Energy Audit Pack.

    This is the root configuration model that contains all sub-configurations
    for industrial energy auditing. The industry_sector field drives which
    engines are prioritized and which benchmarks and BAT references are used.
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
    industry_sector: IndustrySector = Field(
        IndustrySector.MANUFACTURING,
        description="Primary industry sector of the facility",
    )
    facility_tier: FacilityTier = Field(
        FacilityTier.LARGE_ENTERPRISE,
        description="Facility size tier",
    )
    country: str = Field(
        "DE",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2025,
        ge=2020,
        le=2035,
        description="Reporting year for the energy audit",
    )

    # Facility characteristics
    floor_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Total facility floor area in square meters",
    )
    production_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Production/process area in square meters",
    )
    annual_production_tonnes: Optional[float] = Field(
        None,
        ge=0,
        description="Annual production output in tonnes (for SEC calculation)",
    )
    annual_production_units: Optional[float] = Field(
        None,
        ge=0,
        description="Annual production output in units (alternative to tonnes)",
    )
    production_unit_name: str = Field(
        "tonne",
        description="Name of the production unit (tonne, unit, vehicle, wafer, etc.)",
    )
    employees: Optional[int] = Field(
        None,
        ge=0,
        description="Number of employees at the facility",
    )
    operating_hours_per_year: int = Field(
        6000,
        ge=1000,
        le=8760,
        description="Annual operating hours",
    )
    number_of_shifts: int = Field(
        2,
        ge=1,
        le=4,
        description="Number of production shifts",
    )

    # Energy carriers in use
    energy_carriers: List[EnergyCarrier] = Field(
        default_factory=lambda: [
            EnergyCarrier.ELECTRICITY,
            EnergyCarrier.NATURAL_GAS,
        ],
        description="Energy carriers consumed at the facility",
    )

    # Sub-configurations for each engine
    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig,
        description="Energy baseline modeling configuration",
    )
    audit: AuditConfig = Field(
        default_factory=AuditConfig,
        description="Energy audit execution configuration",
    )
    equipment: EquipmentConfig = Field(
        default_factory=EquipmentConfig,
        description="Equipment-level analysis configuration",
    )
    steam: SteamConfig = Field(
        default_factory=SteamConfig,
        description="Steam system analysis configuration",
    )
    compressed_air: CompressedAirConfig = Field(
        default_factory=CompressedAirConfig,
        description="Compressed air system analysis configuration",
    )
    waste_heat: WasteHeatConfig = Field(
        default_factory=WasteHeatConfig,
        description="Waste heat recovery analysis configuration",
    )
    lighting: LightingConfig = Field(
        default_factory=LightingConfig,
        description="Lighting system analysis configuration",
    )
    hvac: HVACConfig = Field(
        default_factory=HVACConfig,
        description="HVAC system analysis configuration",
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Energy benchmarking configuration",
    )
    iso50001: ISO50001Config = Field(
        default_factory=ISO50001Config,
        description="ISO 50001 alignment configuration",
    )
    eed: EEDConfig = Field(
        default_factory=EEDConfig,
        description="EED compliance configuration",
    )

    # Supporting configurations
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and resource limits",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and access control",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Report generation configuration",
    )

    @model_validator(mode="after")
    def validate_data_center_disables_irrelevant(self) -> "IndustrialEnergyAuditConfig":
        """Data centers do not need steam or compressed air analysis."""
        if self.industry_sector == IndustrySector.DATA_CENTER:
            if self.steam.enabled:
                logger.info(
                    "Data center sector: disabling steam system analysis (not applicable)."
                )
                object.__setattr__(self.steam, "enabled", False)
            if self.compressed_air.enabled:
                logger.info(
                    "Data center sector: disabling compressed air analysis (not applicable)."
                )
                object.__setattr__(self.compressed_air, "enabled", False)
        return self

    @model_validator(mode="after")
    def validate_process_industry_requires_waste_heat(self) -> "IndustrialEnergyAuditConfig":
        """Process industries should have waste heat recovery enabled."""
        process_sectors = {
            IndustrySector.PROCESS_INDUSTRY,
            IndustrySector.CHEMICAL,
            IndustrySector.STEEL_METALS,
            IndustrySector.CEMENT,
            IndustrySector.GLASS,
        }
        if self.industry_sector in process_sectors and not self.waste_heat.enabled:
            logger.warning(
                f"Sector {self.industry_sector.value} has significant waste heat potential. "
                "Enabling waste heat recovery analysis."
            )
            object.__setattr__(self.waste_heat, "enabled", True)
        return self

    @model_validator(mode="after")
    def validate_large_enterprise_eed(self) -> "IndustrialEnergyAuditConfig":
        """Large enterprises must comply with EED Article 8."""
        if self.facility_tier == FacilityTier.LARGE_ENTERPRISE and not self.eed.enabled:
            logger.warning(
                "Large enterprise facilities are subject to EED Article 8 mandatory audits. "
                "Enabling EED compliance tracking."
            )
            object.__setattr__(self.eed, "enabled", True)
            object.__setattr__(self.eed, "mandatory_audit", True)
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

    pack: IndustrialEnergyAuditConfig = Field(
        default_factory=IndustrialEnergyAuditConfig,
        description="Main Industrial Energy Audit configuration",
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
        "PACK-031-industrial-energy-audit",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (manufacturing_plant, data_center, etc.)
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

        pack_config = IndustrialEnergyAuditConfig(**preset_data)
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

        pack_config = IndustrialEnergyAuditConfig(**config_data)
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
        pack_config = IndustrialEnergyAuditConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with ENERGY_AUDIT_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: ENERGY_AUDIT_PACK_STEAM__ENABLED=true
                 ENERGY_AUDIT_PACK_BASELINE__MIN_MONTHS=24
        """
        overrides: Dict[str, Any] = {}
        prefix = "ENERGY_AUDIT_PACK_"
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


def validate_config(config: IndustrialEnergyAuditConfig) -> List[str]:
    """Validate an energy audit configuration and return any warnings.

    Args:
        config: IndustrialEnergyAuditConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check facility identification
    if not config.facility_name:
        warnings.append(
            "No facility_name configured. Add a facility name for report identification."
        )

    # Check baseline data sufficiency
    if config.baseline.min_months < 12:
        warnings.append(
            f"Baseline min_months is {config.baseline.min_months}. "
            "12 months recommended for seasonal coverage per IPMVP."
        )

    # Check production data for SEC calculation
    if config.annual_production_tonnes is None and config.annual_production_units is None:
        warnings.append(
            "No production output configured. SEC (Specific Energy Consumption) "
            "cannot be calculated without annual production data."
        )

    # Check energy carriers
    if len(config.energy_carriers) < 1:
        warnings.append(
            "No energy carriers configured. At least one energy carrier is required."
        )

    # Check steam relevance
    steam_sectors = {
        IndustrySector.PROCESS_INDUSTRY,
        IndustrySector.CHEMICAL,
        IndustrySector.FOOD_BEVERAGE,
        IndustrySector.PHARMACEUTICAL,
        IndustrySector.PAPER_PULP,
        IndustrySector.TEXTILE,
        IndustrySector.AUTOMOTIVE,
    }
    if config.industry_sector in steam_sectors and not config.steam.enabled:
        warnings.append(
            f"Steam analysis is recommended for {config.industry_sector.value} sector."
        )

    # Check compressed air relevance
    ca_sectors = {
        IndustrySector.MANUFACTURING,
        IndustrySector.AUTOMOTIVE,
        IndustrySector.FOOD_BEVERAGE,
        IndustrySector.PHARMACEUTICAL,
        IndustrySector.PLASTICS,
        IndustrySector.TEXTILE,
    }
    if config.industry_sector in ca_sectors and not config.compressed_air.enabled:
        warnings.append(
            f"Compressed air analysis is recommended for {config.industry_sector.value} sector."
        )

    # Check EED compliance for large enterprises
    if config.facility_tier == FacilityTier.LARGE_ENTERPRISE:
        if not config.eed.enabled:
            warnings.append(
                "Large enterprise facilities are subject to EED Article 8 mandatory audits."
            )
        if not config.eed.mandatory_audit:
            warnings.append(
                "EED mandatory_audit should be true for large enterprise facilities."
            )

    # Check ISO 50001 and EED interaction
    if config.iso50001.enabled and config.iso50001.eed_exemption_tracking:
        if config.eed.mandatory_audit:
            warnings.append(
                "ISO 50001 certification exempts from EED mandatory audit. "
                "If certified, set eed.mandatory_audit to false."
            )

    # Check data center PUE tracking
    if config.industry_sector == IndustrySector.DATA_CENTER:
        has_pue = EnPIType.PUE in config.iso50001.enpi_types
        if config.iso50001.enabled and not has_pue:
            warnings.append(
                "Data centers should track PUE as a primary EnPI."
            )

    return warnings


def get_default_config(
    sector: IndustrySector = IndustrySector.MANUFACTURING,
) -> IndustrialEnergyAuditConfig:
    """Get default configuration for a given industry sector.

    Args:
        sector: Industry sector to configure for.

    Returns:
        IndustrialEnergyAuditConfig instance with sector-appropriate defaults.
    """
    return IndustrialEnergyAuditConfig(industry_sector=sector)


def get_sector_info(sector: Union[str, IndustrySector]) -> Dict[str, Any]:
    """Get detailed information about an industry sector.

    Args:
        sector: Industry sector enum or string value.

    Returns:
        Dictionary with name, NACE code, energy profile, and key systems.
    """
    key = sector.value if isinstance(sector, IndustrySector) else sector
    return SECTOR_INFO.get(
        key,
        {
            "name": key,
            "nace": "C",
            "typical_energy_split": "Varies",
            "key_systems": ["Motors", "HVAC", "Lighting"],
            "typical_sec_range_kwh_per_tonne": "Varies",
            "bat_bref": "Energy Efficiency BREF",
        },
    )


def get_compressed_air_benchmark(level: str = "good") -> float:
    """Get compressed air specific power benchmark.

    Args:
        level: Benchmark level (best_practice, good, average, poor).

    Returns:
        Specific power in kW per m3/min at 7 bar.
    """
    return COMPRESSED_AIR_BENCHMARKS.get(level, COMPRESSED_AIR_BENCHMARKS["average"])


def get_lpd_standard(area_type: str = "manufacturing_general") -> float:
    """Get Lighting Power Density standard for an area type.

    Args:
        area_type: Area type (office, warehouse_low_bay, manufacturing_general, etc.).

    Returns:
        LPD target in W/m2.
    """
    return LPD_STANDARDS.get(area_type, LPD_STANDARDS["manufacturing_general"])


def get_pue_benchmark(level: str = "efficient") -> float:
    """Get PUE benchmark value for data centers.

    Args:
        level: Benchmark level (world_class, efficient, average, inefficient).

    Returns:
        PUE value.
    """
    return PUE_BENCHMARKS.get(level, PUE_BENCHMARKS["average"])


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
