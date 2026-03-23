"""
PACK-038 Peak Shaving Pack - Configuration Module

This module implements the PeakShavingConfig and PackConfig classes that load,
merge, and validate all configuration for the Peak Shaving Pack. It provides
comprehensive Pydantic v2 models for peak demand management including interval
data analysis, peak target setting, demand charge optimisation, battery energy
storage system (BESS) dispatch, load shifting, coincident peak (CP) management,
demand ratchet mitigation, power factor correction, financial analysis, carbon
impact quantification, and audit trail provenance.

Facility Types:
    - COMMERCIAL_OFFICE: Office buildings (HVAC pre-cooling, EV deferral, lighting)
    - MANUFACTURING: Industrial plants (batch scheduling, compressed air, motors)
    - RETAIL_GROCERY: Retail stores (refrigeration thermal mass, anti-sweat heaters)
    - WAREHOUSE_COLD: Cold storage (thermal mass ride-through, dock scheduling)
    - HEALTHCARE: Hospitals (non-clinical HVAC, laundry, kitchen -- safety-first)
    - DATA_CENTER: Data centres (CRAH/CRAC setpoint, UPS eco-mode -- IT excluded)
    - UNIVERSITY_CAMPUS: University campuses (HVAC scheduling, EV fleet, TES)
    - MIXED_USE_PORTFOLIO: Multi-site portfolios (virtual power plant, coordinated CP)

Peak Shaving Strategies:
    - BESS_DISPATCH: Battery discharge during peak windows
    - LOAD_SHIFT: Pre-cool/pre-heat then coast through peak
    - LOAD_CURTAILMENT: Shed non-critical loads during peak
    - PRODUCTION_SCHEDULING: Shift production batches off-peak
    - COMBINED: Multi-strategy coordinated peak reduction
    - GENERATION: Behind-the-meter generation during peak
    - THERMAL_STORAGE: Ice/chilled-water discharge during peak

CP Methodologies:
    - PJM_1CP: PJM single coincident peak (annual)
    - PJM_5CP: PJM 5 coincident peak hours
    - NYISO_ICAP: NYISO installed capacity peak
    - ISO_NE_ICAP: ISO New England ICAP
    - ERCOT_4CP: ERCOT 4 coincident peak (June-September)
    - MISO_YEARLY: MISO yearly coincident peak
    - CUSTOM: Custom CP methodology definition

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (commercial_office / manufacturing / retail_grocery /
       warehouse_cold / healthcare / data_center / university_campus /
       mixed_use_portfolio)
    3. Environment overrides (PS_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - FERC Order 745 (demand response compensation at LMP)
    - FERC Order 2222 (DER aggregation in wholesale markets)
    - EU Electricity Market Design Regulation (EU) 2024/1747
    - EU Energy Efficiency Directive (EU) 2023/1791
    - ASHRAE Standard 90.1 (energy efficiency in buildings)
    - ISO 50001:2018 (energy management systems)
    - GHG Protocol Corporate Standard (Scope 2 emissions)
    - NAESB WEQ Business Practice Standards (wholesale electric)

Example:
    >>> from packs.energy_efficiency.PACK_038_peak_shaving.config import (
    ...     PackConfig,
    ...     PeakShavingConfig,
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
# Enums - Peak shaving configuration enumeration types
# =============================================================================


class FacilityType(str, Enum):
    """Facility type classification for peak shaving scoping."""

    COMMERCIAL_OFFICE = "COMMERCIAL_OFFICE"
    MANUFACTURING = "MANUFACTURING"
    RETAIL_GROCERY = "RETAIL_GROCERY"
    WAREHOUSE_COLD = "WAREHOUSE_COLD"
    HEALTHCARE = "HEALTHCARE"
    DATA_CENTER = "DATA_CENTER"
    UNIVERSITY_CAMPUS = "UNIVERSITY_CAMPUS"
    MIXED_USE_PORTFOLIO = "MIXED_USE_PORTFOLIO"


class BatteryChemistry(str, Enum):
    """Battery energy storage system chemistry type."""

    LFP = "LFP"
    NMC = "NMC"
    NCA = "NCA"
    LTO = "LTO"
    FLOW_VANADIUM = "FLOW_VANADIUM"
    LEAD_ACID = "LEAD_ACID"


class DispatchStrategy(str, Enum):
    """Peak shaving dispatch strategy classification."""

    BESS_DISPATCH = "BESS_DISPATCH"
    LOAD_SHIFT = "LOAD_SHIFT"
    LOAD_CURTAILMENT = "LOAD_CURTAILMENT"
    PRODUCTION_SCHEDULING = "PRODUCTION_SCHEDULING"
    COMBINED = "COMBINED"
    GENERATION = "GENERATION"
    THERMAL_STORAGE = "THERMAL_STORAGE"


class CPMethodology(str, Enum):
    """Coincident peak calculation methodology."""

    PJM_1CP = "PJM_1CP"
    PJM_5CP = "PJM_5CP"
    NYISO_ICAP = "NYISO_ICAP"
    ISO_NE_ICAP = "ISO_NE_ICAP"
    ERCOT_4CP = "ERCOT_4CP"
    MISO_YEARLY = "MISO_YEARLY"
    CUSTOM = "CUSTOM"


class RatchetType(str, Enum):
    """Demand ratchet type classification."""

    ANNUAL = "ANNUAL"
    SEASONAL = "SEASONAL"
    ROLLING_12 = "ROLLING_12"
    NONE = "NONE"


class CorrectionType(str, Enum):
    """Power factor correction method."""

    CAPACITOR_BANK = "CAPACITOR_BANK"
    ACTIVE_FILTER = "ACTIVE_FILTER"
    SYNCHRONOUS_CONDENSER = "SYNCHRONOUS_CONDENSER"
    VFD_INTEGRATED = "VFD_INTEGRATED"
    NONE = "NONE"


class IncentiveProgram(str, Enum):
    """Utility incentive program classification."""

    DEMAND_RESPONSE = "DEMAND_RESPONSE"
    PEAK_REDUCTION = "PEAK_REDUCTION"
    BESS_INCENTIVE = "BESS_INCENTIVE"
    TIME_OF_USE = "TIME_OF_USE"
    REAL_TIME_PRICING = "REAL_TIME_PRICING"
    CAPACITY_MARKET = "CAPACITY_MARKET"
    CRITICAL_PEAK_PRICING = "CRITICAL_PEAK_PRICING"
    NONE = "NONE"


class TariffType(str, Enum):
    """Electricity tariff structure classification."""

    FLAT_RATE = "FLAT_RATE"
    TIME_OF_USE = "TIME_OF_USE"
    REAL_TIME_PRICING = "REAL_TIME_PRICING"
    CRITICAL_PEAK_PRICING = "CRITICAL_PEAK_PRICING"
    DEMAND_CHARGE = "DEMAND_CHARGE"
    COINCIDENT_PEAK = "COINCIDENT_PEAK"
    HYBRID = "HYBRID"


class IntervalLength(str, Enum):
    """Interval metering data granularity."""

    ONE_MINUTE = "1_MIN"
    FIVE_MINUTE = "5_MIN"
    FIFTEEN_MINUTE = "15_MIN"
    THIRTY_MINUTE = "30_MIN"
    SIXTY_MINUTE = "60_MIN"


class OutputFormat(str, Enum):
    """Output format for peak shaving reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency for peak shaving performance."""

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
        "typical_peak_kw": "500-2000",
        "typical_reduction_pct": "15-25",
        "peak_drivers": ["HVAC cooling (60%)", "Lighting (20%)", "Plug loads (20%)"],
        "key_constraints": ["Occupant comfort", "Meeting schedules", "Lease terms"],
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_peak_kw": "1000-10000",
        "typical_reduction_pct": "10-20",
        "peak_drivers": [
            "Motor loads (50%)", "Process (30%)", "HVAC (15%)", "Lighting (5%)"
        ],
        "key_constraints": [
            "Production throughput", "Product quality", "Ratchet clauses"
        ],
    },
    "RETAIL_GROCERY": {
        "name": "Retail / Grocery Store",
        "typical_peak_kw": "200-800",
        "typical_reduction_pct": "10-20",
        "peak_drivers": [
            "Refrigeration (45%)", "HVAC (30%)", "Lighting (20%)", "Cooking (5%)"
        ],
        "key_constraints": [
            "Food safety temperatures", "Customer comfort", "Case temperature"
        ],
    },
    "WAREHOUSE_COLD": {
        "name": "Warehouse / Cold Storage",
        "typical_peak_kw": "500-3000",
        "typical_reduction_pct": "15-30",
        "peak_drivers": [
            "Refrigeration (65%)", "Dock equipment (15%)", "HVAC (10%)", "Lighting (10%)"
        ],
        "key_constraints": [
            "Cold chain integrity", "HACCP limits", "Product temperature"
        ],
    },
    "HEALTHCARE": {
        "name": "Hospital / Healthcare Facility",
        "typical_peak_kw": "1000-5000",
        "typical_reduction_pct": "5-15",
        "peak_drivers": [
            "HVAC (40%)", "Medical equipment (25%)", "Lighting (15%)",
            "Kitchen (10%)", "Other (10%)"
        ],
        "key_constraints": [
            "Patient safety", "ICU/OR/ER exclusion", "Clinical equipment"
        ],
    },
    "DATA_CENTER": {
        "name": "Data Centre / Server Facility",
        "typical_peak_kw": "2000-20000",
        "typical_reduction_pct": "5-10",
        "peak_drivers": [
            "IT compute (55%)", "Cooling (35%)", "UPS/power distribution (10%)"
        ],
        "key_constraints": [
            "IT uptime SLA (99.99%)", "PUE constraint (+0.10 max)", "IT loads excluded"
        ],
    },
    "UNIVERSITY_CAMPUS": {
        "name": "University / Campus",
        "typical_peak_kw": "5000-30000",
        "typical_reduction_pct": "15-25",
        "peak_drivers": [
            "HVAC (45%)", "Research (20%)", "Lighting (15%)",
            "Dining (10%)", "Other (10%)"
        ],
        "key_constraints": [
            "Academic schedule", "Research labs", "Summer enhanced opportunity"
        ],
    },
    "MIXED_USE_PORTFOLIO": {
        "name": "Mixed-Use Portfolio",
        "typical_peak_kw": "10000-100000",
        "typical_reduction_pct": "10-20",
        "peak_drivers": ["Variable by building type"],
        "key_constraints": [
            "Portfolio coordination", "Multi-site CP management",
            "Virtual power plant dispatch"
        ],
    },
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "commercial_office": "Corporate offices with HVAC pre-cooling, EV deferral, lighting",
    "manufacturing": "Manufacturing with batch scheduling, compressed air, motor VSD",
    "retail_grocery": "Retail/grocery with refrigeration thermal mass, anti-sweat heaters",
    "warehouse_cold": "Cold storage with thermal mass ride-through, dock scheduling",
    "healthcare": "Healthcare with conservative non-clinical-only peak shaving",
    "data_center": "Data centres with CRAH/CRAC setpoint raise, UPS eco-mode",
    "university_campus": "University campuses with HVAC scheduling, EV fleet, TES",
    "mixed_use_portfolio": "Multi-site portfolios with coordinated CP and virtual power plant",
}

DEFAULT_FINANCIAL_PARAMS: Dict[str, Dict[str, float]] = {
    "US_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.12,
        "demand_charge_eur_per_kw": 15.00,
        "ratchet_demand_charge_eur_per_kw": 12.00,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 30.0,
    },
    "EU_AVERAGE": {
        "electricity_rate_eur_per_kwh": 0.22,
        "demand_charge_eur_per_kw": 10.00,
        "ratchet_demand_charge_eur_per_kw": 8.00,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 85.0,
    },
    "UK": {
        "electricity_rate_eur_per_kwh": 0.28,
        "demand_charge_eur_per_kw": 12.00,
        "ratchet_demand_charge_eur_per_kw": 10.00,
        "discount_rate": 0.08,
        "carbon_price_eur_per_tco2": 55.0,
    },
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class IntervalDataConfig(BaseModel):
    """Configuration for interval meter data ingestion and analysis.

    Defines the expected interval data granularity, data quality thresholds,
    and peak window identification parameters used by the Interval Analysis
    Engine to identify peak demand patterns.
    """

    interval_length: IntervalLength = Field(
        IntervalLength.FIFTEEN_MINUTE,
        description="Interval meter data granularity",
    )
    minimum_data_coverage_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum data coverage required for analysis (%)",
    )
    peak_window_start_hour: int = Field(
        12,
        ge=0,
        le=23,
        description="Start hour of peak demand window (24h format)",
    )
    peak_window_end_hour: int = Field(
        18,
        ge=0,
        le=23,
        description="End hour of peak demand window (24h format)",
    )
    lookback_months: int = Field(
        12,
        ge=1,
        le=36,
        description="Number of months of historical data to analyse",
    )
    outlier_threshold_sigma: float = Field(
        3.0,
        ge=1.5,
        le=5.0,
        description="Standard deviation threshold for outlier detection",
    )
    weather_normalization: bool = Field(
        True,
        description="Apply weather normalisation to interval data",
    )

    @model_validator(mode="after")
    def validate_peak_window(self) -> "IntervalDataConfig":
        """Validate that peak window start is before end."""
        if self.peak_window_start_hour >= self.peak_window_end_hour:
            logger.warning(
                f"Peak window start ({self.peak_window_start_hour}) is not before "
                f"end ({self.peak_window_end_hour}). Cross-midnight windows are "
                "not supported; using default 12-18."
            )
        return self


class PeakTargetConfig(BaseModel):
    """Configuration for peak demand reduction targets.

    Defines the target peak reduction percentage, absolute kW target,
    peak demand thresholds that trigger shaving actions, and seasonal
    adjustments for peak management.
    """

    target_reduction_pct: float = Field(
        15.0,
        ge=1.0,
        le=50.0,
        description="Target peak demand reduction as percentage of current peak",
    )
    absolute_target_kw: Optional[float] = Field(
        None,
        ge=0.0,
        description="Absolute peak demand target in kW (overrides percentage if set)",
    )
    trigger_threshold_pct: float = Field(
        85.0,
        ge=50.0,
        le=99.0,
        description="Percentage of peak at which shaving actions are triggered",
    )
    summer_enhanced: bool = Field(
        True,
        description="Apply enhanced peak shaving during summer months (June-Sept)",
    )
    summer_extra_reduction_pct: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Additional peak reduction target during summer months (%)",
    )
    min_sustained_minutes: int = Field(
        15,
        ge=5,
        le=60,
        description="Minimum sustained minutes above trigger before activation",
    )


class DemandChargeConfig(BaseModel):
    """Configuration for demand charge analysis and optimisation.

    Defines the utility tariff structure, demand charge rates, time-of-use
    periods, and demand charge reduction targets used by the Demand Charge
    Optimiser Engine.
    """

    tariff_type: TariffType = Field(
        TariffType.DEMAND_CHARGE,
        description="Utility tariff structure type",
    )
    on_peak_demand_charge_eur_per_kw: float = Field(
        15.00,
        ge=0.0,
        le=100.0,
        description="On-peak demand charge rate (EUR/kW/month)",
    )
    off_peak_demand_charge_eur_per_kw: float = Field(
        5.00,
        ge=0.0,
        le=100.0,
        description="Off-peak demand charge rate (EUR/kW/month)",
    )
    on_peak_start_hour: int = Field(
        12,
        ge=0,
        le=23,
        description="On-peak period start hour (24h format)",
    )
    on_peak_end_hour: int = Field(
        20,
        ge=0,
        le=23,
        description="On-peak period end hour (24h format)",
    )
    critical_peak_pricing_enabled: bool = Field(
        False,
        description="Enable critical peak pricing event response",
    )
    critical_peak_multiplier: float = Field(
        3.0,
        ge=1.0,
        le=10.0,
        description="Demand charge multiplier during critical peak events",
    )
    incentive_program: IncentiveProgram = Field(
        IncentiveProgram.NONE,
        description="Active utility incentive program for peak reduction",
    )


class BESSConfig(BaseModel):
    """Configuration for battery energy storage system (BESS) peak shaving.

    Defines BESS capacity, chemistry, charge/discharge parameters, state-of-
    charge management, degradation tracking, and dispatch constraints used by
    the BESS Dispatch Engine for peak clipping.
    """

    enabled: bool = Field(
        False,
        description="Enable BESS for peak shaving dispatch",
    )
    capacity_kwh: float = Field(
        0.0,
        ge=0.0,
        le=100000.0,
        description="Total usable battery capacity (kWh)",
    )
    max_discharge_kw: float = Field(
        0.0,
        ge=0.0,
        le=50000.0,
        description="Maximum discharge power rate (kW)",
    )
    max_charge_kw: float = Field(
        0.0,
        ge=0.0,
        le=50000.0,
        description="Maximum charge power rate (kW)",
    )
    chemistry: BatteryChemistry = Field(
        BatteryChemistry.LFP,
        description="Battery chemistry type",
    )
    min_soc_pct: float = Field(
        10.0,
        ge=0.0,
        le=50.0,
        description="Minimum state of charge to preserve battery health (%)",
    )
    max_soc_pct: float = Field(
        95.0,
        ge=50.0,
        le=100.0,
        description="Maximum state of charge limit (%)",
    )
    round_trip_efficiency: float = Field(
        0.92,
        ge=0.5,
        le=1.0,
        description="Round-trip charge/discharge efficiency",
    )
    cycles_per_day_limit: int = Field(
        2,
        ge=1,
        le=10,
        description="Maximum charge/discharge cycles per day",
    )
    degradation_tracking: bool = Field(
        True,
        description="Track battery degradation from peak shaving cycles",
    )
    warranty_cycles: int = Field(
        6000,
        ge=1000,
        le=15000,
        description="Battery warranty cycle count for degradation budgeting",
    )
    reserve_for_backup_pct: float = Field(
        0.0,
        ge=0.0,
        le=50.0,
        description="Percentage of BESS reserved for backup/UPS function",
    )

    @model_validator(mode="after")
    def validate_soc_range(self) -> "BESSConfig":
        """Validate that min SOC is below max SOC."""
        if self.min_soc_pct >= self.max_soc_pct:
            logger.warning(
                f"BESS min_soc_pct ({self.min_soc_pct}) >= max_soc_pct "
                f"({self.max_soc_pct}). Resetting to defaults 10/95."
            )
            self.min_soc_pct = 10.0
            self.max_soc_pct = 95.0
        return self


class LoadShiftConfig(BaseModel):
    """Configuration for load shifting strategies in peak shaving.

    Defines which loads can be pre-cooled/pre-heated before peak windows,
    which loads can be deferred, and thermal comfort constraints during
    the coast-through period.
    """

    pre_cooling_enabled: bool = Field(
        True,
        description="Enable HVAC pre-cooling before peak window",
    )
    pre_cooling_duration_minutes: int = Field(
        60,
        ge=15,
        le=240,
        description="Pre-cooling duration before peak window (minutes)",
    )
    pre_cooling_setpoint_offset_c: float = Field(
        -2.0,
        ge=-5.0,
        le=0.0,
        description="Temperature setpoint offset during pre-cooling (negative = cooler)",
    )
    coast_duration_minutes: int = Field(
        120,
        ge=30,
        le=360,
        description="Thermal coast duration through peak window (minutes)",
    )
    max_temperature_rise_c: float = Field(
        2.0,
        ge=0.5,
        le=6.0,
        description="Maximum temperature rise allowed during coast period (Celsius)",
    )
    ev_charging_deferral: bool = Field(
        True,
        description="Defer EV charging to off-peak periods during peak windows",
    )
    production_batch_scheduling: bool = Field(
        False,
        description="Enable production batch rescheduling to off-peak",
    )
    compressed_air_management: bool = Field(
        False,
        description="Enable compressed air system pressure reduction during peak",
    )
    thermal_mass_ride_through: bool = Field(
        False,
        description="Use building/product thermal mass for peak ride-through",
    )


class CPConfig(BaseModel):
    """Configuration for coincident peak (CP) management.

    Defines the CP methodology, alert parameters, response strategy, and
    performance targets for transmission-level coincident peak demand
    charge avoidance.
    """

    enabled: bool = Field(
        False,
        description="Enable coincident peak management",
    )
    methodology: CPMethodology = Field(
        CPMethodology.PJM_5CP,
        description="Coincident peak calculation methodology",
    )
    alert_threshold_mw: Optional[float] = Field(
        None,
        ge=0.0,
        description="System load threshold (MW) that triggers CP alert",
    )
    alert_lead_time_minutes: int = Field(
        60,
        ge=15,
        le=240,
        description="Minimum lead time for CP alert notification (minutes)",
    )
    cp_season_start_month: int = Field(
        6,
        ge=1,
        le=12,
        description="CP season start month (1=January, 6=June)",
    )
    cp_season_end_month: int = Field(
        9,
        ge=1,
        le=12,
        description="CP season end month (9=September)",
    )
    target_reduction_during_cp_pct: float = Field(
        25.0,
        ge=5.0,
        le=80.0,
        description="Target demand reduction during CP events (%)",
    )
    max_cp_events_per_season: int = Field(
        15,
        ge=1,
        le=50,
        description="Maximum CP events to respond to per season",
    )
    notification_channels: List[str] = Field(
        default_factory=lambda: ["EMAIL", "SMS", "BMS_SIGNAL"],
        description="Notification channels for CP alerts",
    )
    multi_site_coordination: bool = Field(
        False,
        description="Enable multi-site coordinated CP response",
    )


class RatchetConfig(BaseModel):
    """Configuration for demand ratchet analysis and mitigation.

    Demand ratchets lock in a minimum billing demand based on historical
    peaks. This configuration controls how the pack analyses ratchet
    exposure, calculates the cost of ratchet-locked demand, and
    optimises peak shaving to avoid ratchet triggers.
    """

    enabled: bool = Field(
        True,
        description="Enable demand ratchet analysis and mitigation",
    )
    ratchet_type: RatchetType = Field(
        RatchetType.ANNUAL,
        description="Demand ratchet type in the utility tariff",
    )
    ratchet_percentage: float = Field(
        80.0,
        ge=50.0,
        le=100.0,
        description="Ratchet percentage (billing demand = max(current, ratchet_pct * historical_peak))",
    )
    ratchet_lookback_months: int = Field(
        12,
        ge=1,
        le=36,
        description="Number of months in the ratchet lookback period",
    )
    reset_month: Optional[int] = Field(
        None,
        ge=1,
        le=12,
        description="Month when ratchet resets (None = rolling, 1-12 = specific month)",
    )
    current_ratchet_kw: Optional[float] = Field(
        None,
        ge=0.0,
        description="Current ratchet-locked billing demand (kW)",
    )


class PowerFactorConfig(BaseModel):
    """Configuration for power factor correction in peak shaving.

    Poor power factor increases apparent power (kVA) relative to real
    power (kW), resulting in higher demand charges and potential utility
    penalties. This configuration controls power factor analysis,
    correction recommendations, and kVA demand charge optimisation.
    """

    enabled: bool = Field(
        True,
        description="Enable power factor analysis and correction",
    )
    current_power_factor: float = Field(
        0.90,
        ge=0.5,
        le=1.0,
        description="Current facility average power factor",
    )
    target_power_factor: float = Field(
        0.95,
        ge=0.85,
        le=1.0,
        description="Target power factor after correction",
    )
    utility_penalty_threshold: float = Field(
        0.90,
        ge=0.7,
        le=1.0,
        description="Power factor below which utility applies penalty",
    )
    correction_type: CorrectionType = Field(
        CorrectionType.CAPACITOR_BANK,
        description="Power factor correction method",
    )
    kva_demand_billing: bool = Field(
        False,
        description="Utility bills on kVA demand rather than kW demand",
    )

    @model_validator(mode="after")
    def validate_power_factor_targets(self) -> "PowerFactorConfig":
        """Validate that target PF is greater than or equal to current PF."""
        if self.target_power_factor < self.current_power_factor:
            logger.warning(
                f"Target power factor ({self.target_power_factor}) is below "
                f"current ({self.current_power_factor}). Target should be "
                "higher than current for improvement."
            )
        return self


class FinancialConfig(BaseModel):
    """Configuration for financial analysis of peak shaving measures.

    Defines energy tariffs, demand charges, BESS capital costs, discount
    rates, and analysis parameters used to calculate peak shaving ROI,
    NPV, and payback period.
    """

    electricity_rate_eur_per_kwh: float = Field(
        0.12,
        ge=0.01,
        le=1.0,
        description="Electricity volumetric rate (EUR/kWh)",
    )
    demand_charge_eur_per_kw: float = Field(
        15.00,
        ge=0.0,
        le=100.0,
        description="Monthly demand charge (EUR/kW of peak demand)",
    )
    ratchet_demand_charge_eur_per_kw: float = Field(
        12.00,
        ge=0.0,
        le=100.0,
        description="Monthly ratcheted demand charge (EUR/kW)",
    )
    bess_capex_eur_per_kwh: float = Field(
        350.0,
        ge=50.0,
        le=2000.0,
        description="BESS capital cost (EUR/kWh installed)",
    )
    bess_annual_opex_pct: float = Field(
        2.0,
        ge=0.0,
        le=10.0,
        description="BESS annual operating cost as percentage of CAPEX (%)",
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
    demand_charge_escalation_pct: float = Field(
        2.0,
        ge=0.0,
        le=15.0,
        description="Annual demand charge escalation rate (%)",
    )
    currency: str = Field(
        "EUR",
        description="Currency for financial calculations (ISO 4217)",
    )
    include_incentives: bool = Field(
        True,
        description="Include utility incentive programs in financial analysis",
    )


class CarbonConfig(BaseModel):
    """Configuration for carbon impact assessment of peak shaving.

    Peak shaving reduces demand during hours when marginal generators
    are typically fossil-fuel peakers. This configuration controls carbon
    quantification using marginal emission factors, supporting both
    location-based and market-based Scope 2 accounting per GHG Protocol.
    """

    enabled: bool = Field(
        True,
        description="Enable carbon impact assessment for peak shaving",
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
        description="Track peak shaving carbon reductions against SBTi pathway",
    )
    market_based_accounting: bool = Field(
        False,
        description="Use market-based emission factors (RECs, green tariffs)",
    )
    peak_hour_premium_factor: float = Field(
        1.3,
        ge=1.0,
        le=3.0,
        description="Carbon intensity premium factor for peak hours vs average",
    )


class ReportingConfig(BaseModel):
    """Configuration for peak shaving report generation and dashboards."""

    frequency: ReportingFrequency = Field(
        ReportingFrequency.MONTHLY,
        description="Reporting frequency for peak shaving performance updates",
    )
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.MARKDOWN, OutputFormat.HTML],
        description="Output formats for peak shaving reports",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary with demand charge savings",
    )
    peak_event_log: bool = Field(
        True,
        description="Generate detailed log of all peak shaving events",
    )
    bess_performance_report: bool = Field(
        False,
        description="Generate BESS performance and degradation report",
    )
    cp_performance_report: bool = Field(
        False,
        description="Generate coincident peak performance report",
    )
    ratchet_exposure_report: bool = Field(
        True,
        description="Generate demand ratchet exposure analysis report",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )


class AlertConfig(BaseModel):
    """Configuration for peak demand alert notifications.

    Defines the alert thresholds, notification channels, escalation
    rules, and suppression windows for real-time peak demand monitoring.
    """

    enabled: bool = Field(
        True,
        description="Enable real-time peak demand alerts",
    )
    channels: List[str] = Field(
        default_factory=lambda: ["EMAIL", "SMS", "BMS_SIGNAL"],
        description="Notification channels for peak demand alerts",
    )
    warning_threshold_pct: float = Field(
        80.0,
        ge=50.0,
        le=99.0,
        description="Warning alert at this percentage of peak target (%)",
    )
    critical_threshold_pct: float = Field(
        90.0,
        ge=60.0,
        le=100.0,
        description="Critical alert at this percentage of peak target (%)",
    )
    escalation_minutes: int = Field(
        10,
        ge=1,
        le=60,
        description="Minutes before escalation to next notification tier",
    )
    suppression_window_minutes: int = Field(
        30,
        ge=5,
        le=120,
        description="Minimum minutes between repeated alerts of same severity",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "AlertConfig":
        """Validate that warning threshold is below critical threshold."""
        if self.warning_threshold_pct >= self.critical_threshold_pct:
            logger.warning(
                f"Warning threshold ({self.warning_threshold_pct}%) >= "
                f"critical threshold ({self.critical_threshold_pct}%). "
                "Warning should be below critical."
            )
        return self


class SecurityConfig(BaseModel):
    """Security and access control configuration for peak shaving operations."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "peak_shaving_manager",
            "energy_manager",
            "facility_manager",
            "bess_operator",
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
        description="Enable security audit logging for all peak shaving operations",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored peak shaving data",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_facilities: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum number of facilities per peak shaving portfolio",
    )
    max_intervals_per_analysis: int = Field(
        105120,
        ge=1000,
        le=525600,
        description="Maximum interval data points per analysis (default: 1 year of 5-min)",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for tariff data and emission factors (seconds)",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=5000,
        description="Batch size for bulk facility processing",
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
        description="Enable audit trail for all peak shaving calculations",
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
        description="Track full data lineage from meter through savings report",
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


class PeakShavingConfig(BaseModel):
    """Main configuration for PACK-038 Peak Shaving Pack.

    This is the root configuration model that contains all sub-configurations
    for peak demand management. The facility_type field drives which load
    shift strategies are prioritised, which BESS dispatch parameters apply,
    and which CP methodology is used for coincident peak management.
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
        description="Primary facility type for peak shaving scoping",
    )
    country: str = Field(
        "US",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for peak shaving performance tracking",
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
        description="Historical annual peak demand (kW)",
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
    interval_data: IntervalDataConfig = Field(
        default_factory=IntervalDataConfig,
        description="Interval meter data analysis configuration",
    )
    peak_target: PeakTargetConfig = Field(
        default_factory=PeakTargetConfig,
        description="Peak demand reduction target configuration",
    )
    demand_charge: DemandChargeConfig = Field(
        default_factory=DemandChargeConfig,
        description="Demand charge analysis and optimisation configuration",
    )
    bess: BESSConfig = Field(
        default_factory=BESSConfig,
        description="Battery energy storage system configuration",
    )
    load_shift: LoadShiftConfig = Field(
        default_factory=LoadShiftConfig,
        description="Load shifting strategy configuration",
    )
    cp: CPConfig = Field(
        default_factory=CPConfig,
        description="Coincident peak management configuration",
    )
    ratchet: RatchetConfig = Field(
        default_factory=RatchetConfig,
        description="Demand ratchet analysis configuration",
    )
    power_factor: PowerFactorConfig = Field(
        default_factory=PowerFactorConfig,
        description="Power factor correction configuration",
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
    alerts: AlertConfig = Field(
        default_factory=AlertConfig,
        description="Peak demand alert configuration",
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
    def validate_healthcare_constraints(self) -> "PeakShavingConfig":
        """Healthcare facilities must use conservative peak shaving parameters."""
        if self.facility_type == FacilityType.HEALTHCARE:
            if self.peak_target.target_reduction_pct > 15:
                logger.warning(
                    "Healthcare facility: target_reduction_pct exceeds 15%. "
                    "Consider reducing for patient safety."
                )
            if self.load_shift.max_temperature_rise_c > 2.0:
                logger.warning(
                    "Healthcare facility: max temperature rise exceeds 2.0C. "
                    "Non-clinical areas should maintain patient comfort."
                )
        return self

    @model_validator(mode="after")
    def validate_data_center_constraints(self) -> "PeakShavingConfig":
        """Data centres must enforce IT uptime constraints."""
        if self.facility_type == FacilityType.DATA_CENTER:
            if self.peak_target.target_reduction_pct > 10:
                logger.warning(
                    "Data centre: target_reduction_pct exceeds 10%. "
                    "IT loads are excluded; only cooling/UPS savings available."
                )
        return self

    @model_validator(mode="after")
    def validate_bess_financial_consistency(self) -> "PeakShavingConfig":
        """BESS enabled requires financial BESS CAPEX to be meaningful."""
        if self.bess.enabled and self.bess.capacity_kwh > 0:
            if self.financial.bess_capex_eur_per_kwh <= 0:
                logger.warning(
                    "BESS is enabled but bess_capex_eur_per_kwh is zero. "
                    "Financial analysis will undercount BESS investment cost."
                )
        return self

    @model_validator(mode="after")
    def validate_cp_season(self) -> "PeakShavingConfig":
        """CP season start should be before end."""
        if self.cp.enabled:
            if self.cp.cp_season_start_month > self.cp.cp_season_end_month:
                logger.warning(
                    f"CP season start month ({self.cp.cp_season_start_month}) is "
                    f"after end month ({self.cp.cp_season_end_month}). "
                    "Cross-year CP seasons are not supported."
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

    pack: PeakShavingConfig = Field(
        default_factory=PeakShavingConfig,
        description="Main Peak Shaving configuration",
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
        "PACK-038-peak-shaving",
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

        pack_config = PeakShavingConfig(**preset_data)
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

        pack_config = PeakShavingConfig(**config_data)
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
        pack_config = PeakShavingConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with PS_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: PS_PACK_PEAK_TARGET__TARGET_REDUCTION_PCT=20
                 PS_PACK_FINANCIAL__DEMAND_CHARGE_EUR_PER_KW=18.00
        """
        overrides: Dict[str, Any] = {}
        prefix = "PS_PACK_"
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


def validate_config(config: PeakShavingConfig) -> List[str]:
    """Validate a peak shaving configuration and return any warnings.

    Args:
        config: PeakShavingConfig instance to validate.

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
            "No peak_demand_kw configured. Peak shaving calculations require "
            "historical peak demand data."
        )

    # Check energy data availability
    if config.annual_electricity_kwh is None:
        warnings.append(
            "No annual_electricity_kwh configured. Load factor and energy cost "
            "calculations require annual electricity consumption."
        )

    # Check peak target reasonableness
    if config.peak_target.target_reduction_pct > 30:
        warnings.append(
            f"Target peak reduction ({config.peak_target.target_reduction_pct}%) "
            "exceeds 30%. Verify this is achievable for your facility type."
        )

    # Check BESS capacity vs peak demand
    if config.bess.enabled and config.peak_demand_kw is not None:
        if config.bess.max_discharge_kw > config.peak_demand_kw * 0.5:
            warnings.append(
                f"BESS max discharge ({config.bess.max_discharge_kw} kW) exceeds "
                f"50% of peak demand ({config.peak_demand_kw} kW). Verify sizing."
            )

    # Check interval data coverage
    if config.interval_data.minimum_data_coverage_pct < 80:
        warnings.append(
            f"Interval data coverage threshold ({config.interval_data.minimum_data_coverage_pct}%) "
            "is below 80%. Peak identification accuracy may be reduced."
        )

    # Check healthcare constraints
    if config.facility_type == FacilityType.HEALTHCARE:
        if config.peak_target.target_reduction_pct > 15:
            warnings.append(
                "Healthcare facilities should target 5-15% peak reduction for safety."
            )
        if config.load_shift.max_temperature_rise_c > 2.0:
            warnings.append(
                "Healthcare facilities should limit temperature rise to 2.0C max."
            )

    # Check data center constraints
    if config.facility_type == FacilityType.DATA_CENTER:
        if config.peak_target.target_reduction_pct > 10:
            warnings.append(
                "Data centres should target 5-10% peak reduction (IT loads excluded)."
            )

    # Check CP configuration consistency
    if config.cp.enabled and not config.cp.alert_threshold_mw:
        warnings.append(
            "CP management enabled but no alert_threshold_mw configured. "
            "CP alerts will not trigger without a system load threshold."
        )

    # Check ratchet configuration
    if config.ratchet.enabled and config.ratchet.current_ratchet_kw is None:
        warnings.append(
            "Ratchet analysis enabled but no current_ratchet_kw configured. "
            "Enter current ratchet-locked demand for accurate analysis."
        )

    # Check power factor
    if config.power_factor.enabled:
        if config.power_factor.current_power_factor < config.power_factor.utility_penalty_threshold:
            warnings.append(
                f"Current power factor ({config.power_factor.current_power_factor}) "
                f"is below utility penalty threshold ({config.power_factor.utility_penalty_threshold}). "
                "Power factor correction should be prioritised."
            )

    # Check carbon configuration
    if config.carbon.enabled and config.carbon.marginal_emission_factor_kgco2_per_kwh <= 0:
        warnings.append(
            "Carbon assessment enabled but marginal emission factor is zero."
        )

    # Check demand charge rates
    if config.demand_charge.on_peak_demand_charge_eur_per_kw <= 0:
        warnings.append(
            "On-peak demand charge is zero. Demand charge optimisation will "
            "show no savings. Verify tariff configuration."
        )

    return warnings


def get_default_config(
    facility_type: FacilityType = FacilityType.COMMERCIAL_OFFICE,
) -> PeakShavingConfig:
    """Get default configuration for a given facility type.

    Args:
        facility_type: Facility type to configure for.

    Returns:
        PeakShavingConfig instance with facility-appropriate defaults.
    """
    return PeakShavingConfig(facility_type=facility_type)


def get_facility_info(facility_type: Union[str, FacilityType]) -> Dict[str, Any]:
    """Get detailed information about a facility type.

    Args:
        facility_type: Facility type enum or string value.

    Returns:
        Dictionary with name, typical peak, reduction potential, and constraints.
    """
    key = facility_type.value if isinstance(facility_type, FacilityType) else facility_type
    return FACILITY_TYPE_INFO.get(
        key,
        {
            "name": key,
            "typical_peak_kw": "Varies",
            "typical_reduction_pct": "10-20",
            "peak_drivers": ["HVAC", "Lighting"],
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
    "BatteryChemistry",
    "CPMethodology",
    "CorrectionType",
    "DispatchStrategy",
    "FacilityType",
    "IncentiveProgram",
    "IntervalLength",
    "OutputFormat",
    "RatchetType",
    "ReportingFrequency",
    "TariffType",
    # Sub-config models
    "AlertConfig",
    "AuditTrailConfig",
    "BESSConfig",
    "CPConfig",
    "CarbonConfig",
    "DemandChargeConfig",
    "FinancialConfig",
    "IntervalDataConfig",
    "LoadShiftConfig",
    "PeakTargetConfig",
    "PerformanceConfig",
    "PowerFactorConfig",
    "RatchetConfig",
    "ReportingConfig",
    "SecurityConfig",
    # Main config models
    "PeakShavingConfig",
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
