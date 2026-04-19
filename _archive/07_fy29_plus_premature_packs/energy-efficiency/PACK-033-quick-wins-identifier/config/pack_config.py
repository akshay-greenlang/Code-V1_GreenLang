"""
PACK-033 Quick Wins Identifier Pack - Configuration Manager

This module implements the QuickWinsConfig and PackConfig classes that load,
merge, and validate all configuration for the Quick Wins Identifier Pack.
It provides comprehensive Pydantic v2 models for rapid identification and
prioritization of low-cost, high-impact energy efficiency measures across
commercial and industrial facilities.

Facility Types:
    - OFFICE: Office buildings (lighting, HVAC, plug loads, IT, controls)
    - MANUFACTURING: Manufacturing plants (motors, compressed air, process, lighting)
    - RETAIL: Retail stores and shopping centres (lighting, HVAC, refrigeration)
    - WAREHOUSE: Warehouses and distribution centres (lighting, HVAC, dock seals)
    - HEALTHCARE: Hospitals and clinics (HVAC, lighting, water heating, sterilisation)
    - EDUCATION: Schools and universities (HVAC, lighting, scheduling, behavioural)
    - DATA_CENTER: Data centres and server rooms (cooling, UPS, airflow, IT load)
    - SME: Small and medium enterprises (simplified scan, top-5 quick wins)

Scan Depth Levels:
    - QUICK: Rapid walkthrough with utility bill analysis (30-60 min)
    - STANDARD: Systematic area-by-area scan with spot measurements (2-4 hours)
    - COMPREHENSIVE: Full facility scan with data logging and sub-metering (1-2 days)

Priority Profiles:
    - COST_FOCUSED: Minimize implementation cost, shortest payback first
    - SAVINGS_FOCUSED: Maximize annual savings regardless of upfront cost
    - BALANCED: Weighted balance of cost, savings, and implementation ease
    - CARBON_FOCUSED: Maximize tCO2e reduction per measure
    - QUICK_IMPLEMENTATION: Measures implementable within 30 days first

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (office_building / manufacturing / retail_store /
       warehouse / healthcare / education / data_center / sme_simplified)
    3. Environment overrides (QUICK_WINS_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - EED: EU Energy Efficiency Directive 2023/1791 (Article 8)
    - ISO 50001: Energy management systems
    - ASHRAE 14: Measurement of Energy, Demand, and Water Savings
    - IPMVP: Measurement and verification of energy savings
    - GHG Protocol: Corporate Standard (energy-related emissions)
    - SBTi: Science Based Targets initiative

Example:
    >>> config = PackConfig.from_preset("office_building")
    >>> print(config.pack.facility_type)
    FacilityType.OFFICE
    >>> print(config.pack.scan.depth)
    ScanDepth.STANDARD
    >>> print(config.pack.prioritization.profile)
    PriorityProfile.BALANCED
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
# Enums - Quick wins identifier enumeration types
# =============================================================================


class FacilityType(str, Enum):
    """Facility type classification for quick wins scoping."""

    OFFICE = "OFFICE"
    MANUFACTURING = "MANUFACTURING"
    RETAIL = "RETAIL"
    WAREHOUSE = "WAREHOUSE"
    HEALTHCARE = "HEALTHCARE"
    EDUCATION = "EDUCATION"
    DATA_CENTER = "DATA_CENTER"
    SME = "SME"


class ScanDepth(str, Enum):
    """Scan depth level for quick wins identification."""

    QUICK = "QUICK"
    STANDARD = "STANDARD"
    COMPREHENSIVE = "COMPREHENSIVE"


class OutputFormat(str, Enum):
    """Output format for quick wins reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency for savings tracking."""

    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


class PriorityProfile(str, Enum):
    """Prioritization profile for ranking quick win measures."""

    COST_FOCUSED = "COST_FOCUSED"
    SAVINGS_FOCUSED = "SAVINGS_FOCUSED"
    BALANCED = "BALANCED"
    CARBON_FOCUSED = "CARBON_FOCUSED"
    QUICK_IMPLEMENTATION = "QUICK_IMPLEMENTATION"


class ActionCategory(str, Enum):
    """Categories of quick win energy efficiency actions."""

    LIGHTING = "LIGHTING"
    HVAC = "HVAC"
    PLUG_LOADS = "PLUG_LOADS"
    CONTROLS_SCHEDULING = "CONTROLS_SCHEDULING"
    MOTORS_DRIVES = "MOTORS_DRIVES"
    COMPRESSED_AIR = "COMPRESSED_AIR"
    REFRIGERATION = "REFRIGERATION"
    WATER_HEATING = "WATER_HEATING"
    BUILDING_ENVELOPE = "BUILDING_ENVELOPE"
    BEHAVIORAL = "BEHAVIORAL"
    IT_EQUIPMENT = "IT_EQUIPMENT"
    PROCESS_OPTIMIZATION = "PROCESS_OPTIMIZATION"


class ImplementationComplexity(str, Enum):
    """Implementation complexity classification for measures."""

    NO_COST = "NO_COST"
    LOW_COST = "LOW_COST"
    MEDIUM_COST = "MEDIUM_COST"
    CAPITAL_PROJECT = "CAPITAL_PROJECT"


class MeasureStatus(str, Enum):
    """Implementation status for quick win measures."""

    IDENTIFIED = "IDENTIFIED"
    EVALUATED = "EVALUATED"
    APPROVED = "APPROVED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Facility type display names, typical energy use, and common quick wins
FACILITY_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "OFFICE": {
        "name": "Office Building",
        "typical_eui_kwh_per_m2": "150-300",
        "typical_energy_split": "Electricity 70-85%, Gas 15-30%",
        "common_quick_wins": [
            "LED lighting retrofit",
            "Occupancy sensor installation",
            "HVAC scheduling optimization",
            "IT power management",
            "Setpoint adjustment (+/- 1C)",
            "Plug load timer strips",
        ],
        "typical_savings_potential_pct": "15-30",
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_eui_kwh_per_m2": "200-1000",
        "typical_energy_split": "Electricity 50-70%, Gas 25-45%",
        "common_quick_wins": [
            "Compressed air leak repair",
            "Motor belt tensioning",
            "Lighting schedule optimization",
            "VSD on variable-load motors",
            "Compressed air pressure reduction",
            "Insulation repair on hot surfaces",
        ],
        "typical_savings_potential_pct": "10-25",
    },
    "RETAIL": {
        "name": "Retail Store",
        "typical_eui_kwh_per_m2": "200-500",
        "typical_energy_split": "Electricity 80-95%, Gas 5-20%",
        "common_quick_wins": [
            "LED lighting retrofit",
            "Night curtains on display refrigeration",
            "HVAC scheduling to trading hours",
            "Exterior signage timer controls",
            "Anti-sweat heater controls on fridges",
            "Door closer maintenance",
        ],
        "typical_savings_potential_pct": "15-35",
    },
    "WAREHOUSE": {
        "name": "Warehouse & Distribution Centre",
        "typical_eui_kwh_per_m2": "50-200",
        "typical_energy_split": "Electricity 60-80%, Gas 15-35%",
        "common_quick_wins": [
            "High-bay LED retrofit with occupancy sensors",
            "Dock door seal replacement",
            "HVAC destratification fans",
            "Forklift battery charging optimization",
            "Conveyor idle-mode programming",
            "Loading bay lighting controls",
        ],
        "typical_savings_potential_pct": "15-30",
    },
    "HEALTHCARE": {
        "name": "Hospital / Clinic",
        "typical_eui_kwh_per_m2": "300-700",
        "typical_energy_split": "Electricity 50-65%, Gas 30-45%",
        "common_quick_wins": [
            "LED lighting in corridors and car parks",
            "HVAC setback in unoccupied areas",
            "BMS scheduling optimization",
            "Hot water temperature reduction",
            "Chiller sequencing optimization",
            "Ventilation rate adjustment in non-clinical areas",
        ],
        "typical_savings_potential_pct": "10-20",
    },
    "EDUCATION": {
        "name": "School / University",
        "typical_eui_kwh_per_m2": "100-350",
        "typical_energy_split": "Electricity 45-65%, Gas 30-50%",
        "common_quick_wins": [
            "Heating schedule aligned to timetable",
            "LED lighting in classrooms and corridors",
            "IT shutdown policy during holidays",
            "Behavioural awareness campaign",
            "Boiler reset temperature optimization",
            "Sports hall and gym lighting controls",
        ],
        "typical_savings_potential_pct": "15-30",
    },
    "DATA_CENTER": {
        "name": "Data Centre / Server Room",
        "typical_eui_kwh_per_m2": "1000-5000",
        "typical_energy_split": "Electricity 95-100%",
        "common_quick_wins": [
            "Blanking panel installation",
            "Hot/cold aisle containment",
            "Raised floor tile management",
            "UPS eco-mode activation",
            "Free cooling temperature setpoint raise",
            "Decommission idle servers",
        ],
        "typical_savings_potential_pct": "10-25",
    },
    "SME": {
        "name": "Small-Medium Enterprise",
        "typical_eui_kwh_per_m2": "100-400",
        "typical_energy_split": "Electricity 60-80%, Gas 20-40%",
        "common_quick_wins": [
            "LED lighting swap",
            "Heating timer and thermostat optimization",
            "Equipment power-down policy",
            "Draught proofing",
            "Hot water temperature reduction",
            "Switch-off awareness signage",
        ],
        "typical_savings_potential_pct": "10-25",
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "office_building": "Office buildings with lighting, HVAC, plug loads, and IT focus",
    "manufacturing": "Manufacturing facilities with motors, compressed air, and process focus",
    "retail_store": "Retail stores with lighting, HVAC, and refrigeration focus",
    "warehouse": "Warehouses with high-bay lighting, HVAC, and dock seals focus",
    "healthcare": "Hospitals and clinics with HVAC, lighting, and water heating focus",
    "education": "Schools and universities with HVAC, lighting, and behavioural focus",
    "data_center": "Data centres with cooling, UPS, and airflow management focus",
    "sme_simplified": "Simplified SME configuration with quick scan and top-5 focus",
}

# Default financial parameters by region
DEFAULT_FINANCIAL_PARAMS: Dict[str, Dict[str, float]] = {
    "EU_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.22,
        "gas_rate_eur_per_kwh": 0.08,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "DE": {
        "electricity_rate_eur_per_kwh": 0.30,
        "gas_rate_eur_per_kwh": 0.10,
        "discount_rate": 0.06,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "UK": {
        "electricity_rate_eur_per_kwh": 0.28,
        "gas_rate_eur_per_kwh": 0.07,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 55.0,
    },
    "US": {
        "electricity_rate_eur_per_kwh": 0.12,
        "gas_rate_eur_per_kwh": 0.04,
        "discount_rate": 0.10,
        "carbon_price_eur_per_tco2": 30.0,
    },
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class ScanConfig(BaseModel):
    """Configuration for facility energy scanning parameters.

    Governs the depth, scope, and categories included in the quick
    wins scan. Categories can be filtered to focus on specific areas
    relevant to the facility type.
    """

    depth: ScanDepth = Field(
        ScanDepth.STANDARD,
        description="Scan depth level: QUICK (30-60 min), STANDARD (2-4 hrs), "
        "COMPREHENSIVE (1-2 days)",
    )
    categories: List[ActionCategory] = Field(
        default_factory=lambda: [
            ActionCategory.LIGHTING,
            ActionCategory.HVAC,
            ActionCategory.PLUG_LOADS,
            ActionCategory.CONTROLS_SCHEDULING,
            ActionCategory.BEHAVIORAL,
        ],
        description="Action categories to include in the quick wins scan",
    )
    max_measures_per_category: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of measures to identify per category",
    )
    include_no_cost_measures: bool = Field(
        True,
        description="Include zero-cost operational and behavioural measures",
    )
    include_low_cost_measures: bool = Field(
        True,
        description="Include low-cost measures (under EUR 5,000 per measure)",
    )
    include_medium_cost_measures: bool = Field(
        True,
        description="Include medium-cost measures (EUR 5,000 - EUR 50,000)",
    )
    max_payback_years: float = Field(
        3.0,
        ge=0.0,
        le=10.0,
        description="Maximum simple payback period (years) for quick win qualification",
    )
    minimum_annual_savings_eur: float = Field(
        100.0,
        ge=0.0,
        description="Minimum annual savings (EUR) for a measure to be included",
    )
    utility_bill_months: int = Field(
        12,
        ge=3,
        le=36,
        description="Months of utility bill data to analyse for baseline",
    )
    walkthrough_checklist: bool = Field(
        True,
        description="Generate pre-populated walkthrough checklist for on-site scan",
    )

    @field_validator("categories")
    @classmethod
    def validate_categories_not_empty(cls, v: List[ActionCategory]) -> List[ActionCategory]:
        """At least one action category must be selected."""
        if not v:
            raise ValueError("At least one action category must be selected for scanning.")
        return v


class FinancialConfig(BaseModel):
    """Configuration for financial analysis of quick win measures.

    Defines energy tariffs, discount rates, and analysis parameters
    used to calculate savings, payback periods, and NPV for each
    identified measure.
    """

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
    maintenance_cost_included: bool = Field(
        True,
        description="Include maintenance cost impact in financial analysis",
    )
    tax_incentive_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Tax deduction or incentive percentage for energy measures",
    )
    currency: str = Field(
        "EUR",
        description="Currency for financial calculations (ISO 4217)",
    )


class CarbonConfig(BaseModel):
    """Configuration for carbon impact assessment of quick win measures.

    Calculates tCO2e reduction for each measure using grid emission
    factors and fuel emission factors. Supports Scope 1 and Scope 2
    carbon accounting per GHG Protocol.
    """

    enabled: bool = Field(
        True,
        description="Enable carbon impact assessment for each measure",
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
        description="Carbon price for monetised carbon benefit (EUR/tCO2e)",
    )
    include_scope_1: bool = Field(
        True,
        description="Include Scope 1 (direct combustion) carbon savings",
    )
    include_scope_2: bool = Field(
        True,
        description="Include Scope 2 (purchased electricity) carbon savings",
    )
    sbti_alignment_tracking: bool = Field(
        False,
        description="Track carbon savings against SBTi reduction pathway",
    )
    market_based_accounting: bool = Field(
        False,
        description="Use market-based emission factors (RECs, green tariffs)",
    )


class PrioritizationConfig(BaseModel):
    """Configuration for quick win measure prioritization and ranking.

    Defines the weighting profile used to rank identified measures
    by a composite score combining financial, carbon, and
    implementation factors.
    """

    profile: PriorityProfile = Field(
        PriorityProfile.BALANCED,
        description="Prioritization profile for measure ranking",
    )
    weight_financial_savings: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Weight for annual financial savings in composite score",
    )
    weight_payback_period: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight for payback period (shorter = higher score)",
    )
    weight_carbon_reduction: float = Field(
        0.20,
        ge=0.0,
        le=1.0,
        description="Weight for carbon reduction (tCO2e) in composite score",
    )
    weight_implementation_ease: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Weight for implementation ease (no-cost ranks highest)",
    )
    weight_co_benefits: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Weight for co-benefits (comfort, maintenance, safety)",
    )
    top_n_recommendations: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of top-ranked measures to include in recommendations",
    )
    group_by_category: bool = Field(
        True,
        description="Group recommendations by action category in output",
    )
    include_implementation_roadmap: bool = Field(
        True,
        description="Generate phased implementation roadmap (immediate/short/medium)",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "PrioritizationConfig":
        """Validate that weights approximately sum to 1.0."""
        total = (
            self.weight_financial_savings
            + self.weight_payback_period
            + self.weight_carbon_reduction
            + self.weight_implementation_ease
            + self.weight_co_benefits
        )
        if abs(total - 1.0) > 0.05:
            logger.warning(
                f"Prioritization weights sum to {total:.2f}, expected ~1.0. "
                "Results may not reflect intended balance."
            )
        return self


class BehavioralConfig(BaseModel):
    """Configuration for behavioural change and awareness programmes.

    Defines settings for occupant engagement, awareness campaigns,
    and behavioural energy efficiency measures.
    """

    enabled: bool = Field(
        True,
        description="Enable behavioural quick wins identification",
    )
    awareness_campaign: bool = Field(
        True,
        description="Include energy awareness campaign recommendations",
    )
    switch_off_programme: bool = Field(
        True,
        description="Include switch-off programme for equipment and lights",
    )
    thermostat_discipline: bool = Field(
        True,
        description="Include thermostat setpoint discipline recommendations",
    )
    energy_champion_network: bool = Field(
        False,
        description="Recommend energy champion network across departments",
    )
    gamification: bool = Field(
        False,
        description="Include gamification and competition elements",
    )
    estimated_savings_pct: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Estimated savings from behavioural measures (%)",
    )


class RebateConfig(BaseModel):
    """Configuration for rebate and incentive programme matching.

    Identifies applicable utility rebates, government grants, and
    tax incentives for each identified quick win measure.
    """

    enabled: bool = Field(
        True,
        description="Enable rebate and incentive matching",
    )
    country: str = Field(
        "DE",
        description="Country for rebate programme lookup (ISO 3166-1 alpha-2)",
    )
    region: str = Field(
        "",
        description="Region or state for sub-national incentive programmes",
    )
    utility_provider: str = Field(
        "",
        description="Utility provider name for utility-specific rebate programmes",
    )
    include_federal_programmes: bool = Field(
        True,
        description="Include federal/national incentive programmes",
    )
    include_state_programmes: bool = Field(
        True,
        description="Include state/regional incentive programmes",
    )
    include_utility_rebates: bool = Field(
        True,
        description="Include utility company rebate programmes",
    )
    include_eu_programmes: bool = Field(
        True,
        description="Include EU-level funding programmes (LIFE, Horizon Europe)",
    )
    min_rebate_eur: float = Field(
        50.0,
        ge=0.0,
        description="Minimum rebate value (EUR) to include in recommendations",
    )


class ReportingConfig(BaseModel):
    """Configuration for quick wins report generation and tracking."""

    frequency: ReportingFrequency = Field(
        ReportingFrequency.QUARTERLY,
        description="Reporting frequency for savings tracking updates",
    )
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML],
        description="Output formats for quick wins reports",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary with top recommendations",
    )
    detailed_measure_cards: bool = Field(
        True,
        description="Generate detailed measure cards with savings calculations",
    )
    implementation_tracker: bool = Field(
        True,
        description="Generate implementation tracking dashboard",
    )
    savings_verification_report: bool = Field(
        True,
        description="Generate post-implementation savings verification report",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    include_photos: bool = Field(
        True,
        description="Include photo placeholders for on-site documentation",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "energy_manager",
            "facility_manager",
            "sustainability_officer",
            "engineer",
            "viewer",
            "admin",
        ],
        description="Available RBAC roles for the pack",
    )
    data_classification: str = Field(
        "INTERNAL",
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


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_facilities: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of facilities per scan run",
    )
    max_measures_per_facility: int = Field(
        200,
        ge=10,
        le=1000,
        description="Maximum measures to evaluate per facility",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for reference data and emission factors (seconds)",
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
        5,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    measure_tracking: bool = Field(
        True,
        description="Track quick win measure implementation status over time",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class QuickWinsConfig(BaseModel):
    """Main configuration for PACK-033 Quick Wins Identifier Pack.

    This is the root configuration model that contains all sub-configurations
    for quick wins identification and prioritization. The facility_type field
    drives which action categories are prioritized and which benchmarks and
    measure libraries are loaded.
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
        FacilityType.OFFICE,
        description="Primary facility type for quick wins scoping",
    )
    country: str = Field(
        "DE",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for the quick wins assessment",
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
    employees: Optional[int] = Field(
        None,
        ge=0,
        description="Number of employees or regular occupants",
    )
    operating_hours_per_year: int = Field(
        2500,
        ge=500,
        le=8760,
        description="Annual operating hours",
    )

    # Sub-configurations
    scan: ScanConfig = Field(
        default_factory=ScanConfig,
        description="Facility scanning configuration",
    )
    financial: FinancialConfig = Field(
        default_factory=FinancialConfig,
        description="Financial analysis configuration",
    )
    carbon: CarbonConfig = Field(
        default_factory=CarbonConfig,
        description="Carbon impact assessment configuration",
    )
    prioritization: PrioritizationConfig = Field(
        default_factory=PrioritizationConfig,
        description="Measure prioritization configuration",
    )
    behavioral: BehavioralConfig = Field(
        default_factory=BehavioralConfig,
        description="Behavioural change programme configuration",
    )
    rebate: RebateConfig = Field(
        default_factory=RebateConfig,
        description="Rebate and incentive matching configuration",
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
    def validate_sme_uses_quick_scan(self) -> "QuickWinsConfig":
        """SME facilities default to quick scan depth if not explicitly set."""
        if self.facility_type == FacilityType.SME:
            if self.scan.depth == ScanDepth.COMPREHENSIVE:
                logger.info(
                    "SME facility type: COMPREHENSIVE scan may be excessive. "
                    "Consider QUICK or STANDARD depth."
                )
        return self

    @model_validator(mode="after")
    def validate_data_center_categories(self) -> "QuickWinsConfig":
        """Data centres should include IT equipment category."""
        if self.facility_type == FacilityType.DATA_CENTER:
            if ActionCategory.IT_EQUIPMENT not in self.scan.categories:
                logger.info(
                    "Data centre facility: adding IT_EQUIPMENT to scan categories."
                )
                self.scan.categories.append(ActionCategory.IT_EQUIPMENT)
        return self

    @model_validator(mode="after")
    def validate_manufacturing_categories(self) -> "QuickWinsConfig":
        """Manufacturing facilities should include motors and compressed air."""
        if self.facility_type == FacilityType.MANUFACTURING:
            mfg_cats = {ActionCategory.MOTORS_DRIVES, ActionCategory.COMPRESSED_AIR}
            missing = mfg_cats - set(self.scan.categories)
            if missing:
                logger.info(
                    f"Manufacturing facility: adding {[c.value for c in missing]} "
                    "to scan categories."
                )
                self.scan.categories.extend(missing)
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

    pack: QuickWinsConfig = Field(
        default_factory=QuickWinsConfig,
        description="Main Quick Wins Identifier configuration",
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
        "PACK-033-quick-wins-identifier",
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

        pack_config = QuickWinsConfig(**preset_data)
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

        pack_config = QuickWinsConfig(**config_data)
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
        pack_config = QuickWinsConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with QUICK_WINS_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: QUICK_WINS_PACK_SCAN__DEPTH=COMPREHENSIVE
                 QUICK_WINS_PACK_FINANCIAL__ELECTRICITY_RATE_EUR_PER_KWH=0.30
        """
        overrides: Dict[str, Any] = {}
        prefix = "QUICK_WINS_PACK_"
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


def validate_config(config: QuickWinsConfig) -> List[str]:
    """Validate a quick wins configuration and return any warnings.

    Args:
        config: QuickWinsConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check facility identification
    if not config.facility_name:
        warnings.append(
            "No facility_name configured. Add a facility name for report identification."
        )

    # Check energy data availability
    if config.annual_electricity_kwh is None and config.annual_gas_kwh is None:
        warnings.append(
            "No annual energy consumption configured. Savings calculations require "
            "annual_electricity_kwh and/or annual_gas_kwh."
        )

    # Check floor area for intensity metrics
    if config.floor_area_m2 is None:
        warnings.append(
            "No floor_area_m2 configured. Energy use intensity (EUI) benchmarking "
            "cannot be calculated without floor area."
        )

    # Check scan categories relevance
    if config.facility_type == FacilityType.DATA_CENTER:
        if ActionCategory.IT_EQUIPMENT not in config.scan.categories:
            warnings.append(
                "Data centre facilities should include IT_EQUIPMENT in scan categories."
            )

    if config.facility_type == FacilityType.MANUFACTURING:
        mfg_recommended = {ActionCategory.MOTORS_DRIVES, ActionCategory.COMPRESSED_AIR}
        missing = mfg_recommended - set(config.scan.categories)
        if missing:
            warnings.append(
                f"Manufacturing facilities should include {[c.value for c in missing]} "
                "in scan categories."
            )

    if config.facility_type == FacilityType.RETAIL:
        if ActionCategory.REFRIGERATION not in config.scan.categories:
            warnings.append(
                "Retail facilities should include REFRIGERATION in scan categories."
            )

    # Check financial parameters
    if config.financial.electricity_rate_eur_per_kwh <= 0:
        warnings.append(
            "Electricity rate is zero or negative. Savings calculations will be invalid."
        )

    # Check payback threshold
    if config.scan.max_payback_years > 5.0:
        warnings.append(
            f"Max payback of {config.scan.max_payback_years} years is above typical "
            "quick win threshold of 3 years. Consider reducing for true quick wins."
        )

    # Check prioritization weights
    total_weight = (
        config.prioritization.weight_financial_savings
        + config.prioritization.weight_payback_period
        + config.prioritization.weight_carbon_reduction
        + config.prioritization.weight_implementation_ease
        + config.prioritization.weight_co_benefits
    )
    if abs(total_weight - 1.0) > 0.05:
        warnings.append(
            f"Prioritization weights sum to {total_weight:.2f}, expected ~1.0."
        )

    # Check carbon config
    if config.carbon.enabled and config.carbon.grid_emission_factor_kgco2_per_kwh <= 0:
        warnings.append(
            "Carbon assessment enabled but grid emission factor is zero."
        )

    return warnings


def get_default_config(
    facility_type: FacilityType = FacilityType.OFFICE,
) -> QuickWinsConfig:
    """Get default configuration for a given facility type.

    Args:
        facility_type: Facility type to configure for.

    Returns:
        QuickWinsConfig instance with facility-appropriate defaults.
    """
    return QuickWinsConfig(facility_type=facility_type)


def get_facility_info(facility_type: Union[str, FacilityType]) -> Dict[str, Any]:
    """Get detailed information about a facility type.

    Args:
        facility_type: Facility type enum or string value.

    Returns:
        Dictionary with name, typical EUI, energy split, and common quick wins.
    """
    key = facility_type.value if isinstance(facility_type, FacilityType) else facility_type
    return FACILITY_TYPE_INFO.get(
        key,
        {
            "name": key,
            "typical_eui_kwh_per_m2": "Varies",
            "typical_energy_split": "Varies",
            "common_quick_wins": ["LED lighting", "HVAC optimization", "Controls"],
            "typical_savings_potential_pct": "10-20",
        },
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
