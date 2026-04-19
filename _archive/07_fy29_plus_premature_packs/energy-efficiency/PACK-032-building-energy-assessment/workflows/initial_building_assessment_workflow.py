# -*- coding: utf-8 -*-
"""
Initial Building Assessment Workflow
==========================================

5-phase workflow for comprehensive building energy assessment within
PACK-032 Building Energy Assessment Pack.

Phases:
    1. BuildingRegistration    -- Collect building data, type, age, floor area, location
    2. DataCollection          -- Gather utility bills, BMS data, drawings, occupancy
    3. EnvelopeAssessment      -- Envelope engine analysis, U-values, air tightness
    4. SystemsAssessment       -- HVAC, lighting, DHW, renewables assessment
    5. ReportGeneration        -- EPC rating, benchmark comparison, improvement plan

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Schedule: on-demand
Estimated duration: 360 minutes

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class BuildingType(str, Enum):
    """Building classification types per EPBD."""

    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    UNIVERSITY = "university"
    HOTEL = "hotel"
    RESIDENTIAL_MULTI = "residential_multi"
    RESIDENTIAL_SINGLE = "residential_single"
    MIXED_USE = "mixed_use"
    INDUSTRIAL = "industrial"
    DATA_CENTRE = "data_centre"
    LEISURE = "leisure"
    RESTAURANT = "restaurant"
    SUPERMARKET = "supermarket"


class ConstructionEra(str, Enum):
    """Construction era for envelope defaults."""

    PRE_1950 = "pre_1950"
    ERA_1950_1975 = "1950_1975"
    ERA_1976_1990 = "1976_1990"
    ERA_1991_2005 = "1991_2005"
    ERA_2006_2012 = "2006_2012"
    ERA_2013_2020 = "2013_2020"
    POST_2020 = "post_2020"


class ClimateZone(str, Enum):
    """European climate zones for degree-day calculations."""

    NORTHERN = "northern"
    CENTRAL = "central"
    SOUTHERN = "southern"
    OCEANIC = "oceanic"
    CONTINENTAL = "continental"
    MEDITERRANEAN = "mediterranean"


class GlazingType(str, Enum):
    """Window glazing types."""

    SINGLE = "single"
    DOUBLE = "double"
    DOUBLE_LOW_E = "double_low_e"
    TRIPLE = "triple"
    TRIPLE_LOW_E = "triple_low_e"


class WallType(str, Enum):
    """Wall construction types."""

    SOLID_BRICK = "solid_brick"
    CAVITY_UNINSULATED = "cavity_uninsulated"
    CAVITY_INSULATED = "cavity_insulated"
    TIMBER_FRAME = "timber_frame"
    CONCRETE_UNINSULATED = "concrete_uninsulated"
    CONCRETE_INSULATED = "concrete_insulated"
    CURTAIN_WALL = "curtain_wall"
    STEEL_CLADDING = "steel_cladding"
    EWIS = "ewis"


class RoofType(str, Enum):
    """Roof construction types."""

    FLAT_UNINSULATED = "flat_uninsulated"
    FLAT_INSULATED = "flat_insulated"
    PITCHED_UNINSULATED = "pitched_uninsulated"
    PITCHED_INSULATED_RAFTER = "pitched_insulated_rafter"
    PITCHED_INSULATED_CEILING = "pitched_insulated_ceiling"
    GREEN_ROOF = "green_roof"
    METAL_DECK = "metal_deck"


class HVACSystemType(str, Enum):
    """HVAC system classifications."""

    SPLIT_SYSTEM = "split_system"
    VRF = "vrf"
    CHILLER_AHU = "chiller_ahu"
    BOILER_RADIATOR = "boiler_radiator"
    HEAT_PUMP_AIR = "heat_pump_air"
    HEAT_PUMP_GROUND = "heat_pump_ground"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    FCU = "fcu"
    PACKAGED_ROOFTOP = "packaged_rooftop"
    NATURAL_VENTILATION = "natural_ventilation"
    MIXED_MODE = "mixed_mode"


class LightingType(str, Enum):
    """Lighting technology types."""

    INCANDESCENT = "incandescent"
    HALOGEN = "halogen"
    CFL = "cfl"
    T8_FLUORESCENT = "t8_fluorescent"
    T5_FLUORESCENT = "t5_fluorescent"
    LED = "led"
    HID = "hid"
    METAL_HALIDE = "metal_halide"


class DHWSystemType(str, Enum):
    """Domestic hot water system types."""

    GAS_BOILER = "gas_boiler"
    ELECTRIC_IMMERSION = "electric_immersion"
    HEAT_PUMP = "heat_pump"
    SOLAR_THERMAL = "solar_thermal"
    DISTRICT = "district"
    INSTANTANEOUS_GAS = "instantaneous_gas"
    INSTANTANEOUS_ELECTRIC = "instantaneous_electric"


class EPCBand(str, Enum):
    """Energy Performance Certificate rating bands."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class DataQuality(str, Enum):
    """Data quality classification."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    DEFAULT = "default"
    CALCULATED = "calculated"


class FindingSeverity(str, Enum):
    """Assessment finding severity level."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


# =============================================================================
# ZERO-HALLUCINATION REFERENCE CONSTANTS
# =============================================================================

# Typical U-values by wall type (W/m2K) - CIBSE Guide A / BR 443
U_VALUES_WALL: Dict[str, float] = {
    "solid_brick": 2.10,
    "cavity_uninsulated": 1.60,
    "cavity_insulated": 0.35,
    "timber_frame": 0.30,
    "concrete_uninsulated": 1.75,
    "concrete_insulated": 0.30,
    "curtain_wall": 1.80,
    "steel_cladding": 0.70,
    "ewis": 0.22,
}

# Typical U-values by roof type (W/m2K) - CIBSE Guide A
U_VALUES_ROOF: Dict[str, float] = {
    "flat_uninsulated": 1.50,
    "flat_insulated": 0.25,
    "pitched_uninsulated": 2.30,
    "pitched_insulated_rafter": 0.18,
    "pitched_insulated_ceiling": 0.16,
    "green_roof": 0.20,
    "metal_deck": 0.35,
}

# Typical U-values by glazing type (W/m2K) - BS EN 673
U_VALUES_GLAZING: Dict[str, float] = {
    "single": 5.80,
    "double": 2.80,
    "double_low_e": 1.60,
    "triple": 1.80,
    "triple_low_e": 0.80,
}

# Floor U-value defaults (W/m2K) by era
U_VALUES_FLOOR: Dict[str, float] = {
    "pre_1950": 1.20,
    "1950_1975": 0.90,
    "1976_1990": 0.70,
    "1991_2005": 0.45,
    "2006_2012": 0.25,
    "2013_2020": 0.18,
    "post_2020": 0.13,
}

# Air permeability defaults (m3/h/m2 @ 50Pa) by era - ATTMA / CIBSE TM23
AIR_PERMEABILITY_DEFAULTS: Dict[str, float] = {
    "pre_1950": 20.0,
    "1950_1975": 15.0,
    "1976_1990": 12.0,
    "1991_2005": 10.0,
    "2006_2012": 7.0,
    "2013_2020": 5.0,
    "post_2020": 3.0,
}

# EUI benchmarks by building type (kWh/m2/yr) - CIBSE TM46 / ECON 19
EUI_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "office": {"typical": 230, "good_practice": 128, "best_practice": 95},
    "retail": {"typical": 305, "good_practice": 190, "best_practice": 140},
    "warehouse": {"typical": 120, "good_practice": 85, "best_practice": 55},
    "hospital": {"typical": 420, "good_practice": 310, "best_practice": 250},
    "school": {"typical": 150, "good_practice": 110, "best_practice": 80},
    "university": {"typical": 240, "good_practice": 170, "best_practice": 130},
    "hotel": {"typical": 340, "good_practice": 250, "best_practice": 200},
    "residential_multi": {"typical": 170, "good_practice": 120, "best_practice": 80},
    "residential_single": {"typical": 200, "good_practice": 130, "best_practice": 75},
    "mixed_use": {"typical": 250, "good_practice": 170, "best_practice": 120},
    "industrial": {"typical": 280, "good_practice": 190, "best_practice": 130},
    "data_centre": {"typical": 800, "good_practice": 500, "best_practice": 350},
    "leisure": {"typical": 380, "good_practice": 270, "best_practice": 200},
    "restaurant": {"typical": 450, "good_practice": 320, "best_practice": 240},
    "supermarket": {"typical": 500, "good_practice": 340, "best_practice": 250},
}

# Heating degree day baselines by climate zone (base 15.5C)
HDD_BY_CLIMATE_ZONE: Dict[str, float] = {
    "northern": 4200.0,
    "central": 3200.0,
    "southern": 1400.0,
    "oceanic": 2600.0,
    "continental": 3600.0,
    "mediterranean": 1200.0,
}

# Cooling degree day baselines by climate zone (base 18.3C)
CDD_BY_CLIMATE_ZONE: Dict[str, float] = {
    "northern": 50.0,
    "central": 200.0,
    "southern": 800.0,
    "oceanic": 100.0,
    "continental": 350.0,
    "mediterranean": 900.0,
}

# CO2 emission factors (kgCO2e/kWh) - DEFRA 2024
EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.207,
    "natural_gas": 0.183,
    "fuel_oil": 0.267,
    "lpg": 0.214,
    "district_heating": 0.160,
    "district_cooling": 0.180,
    "biomass": 0.015,
    "solar": 0.000,
    "heat_pump_electricity": 0.207,
}

# Primary energy factors per EN 15603 / national regulations
PRIMARY_ENERGY_FACTORS: Dict[str, float] = {
    "electricity": 2.50,
    "natural_gas": 1.10,
    "fuel_oil": 1.10,
    "lpg": 1.10,
    "district_heating": 0.70,
    "district_cooling": 0.80,
    "biomass": 0.20,
    "solar": 0.00,
}

# HVAC seasonal efficiency defaults (SCOP / SEER / seasonal eff)
HVAC_EFFICIENCY_DEFAULTS: Dict[str, Dict[str, float]] = {
    "split_system": {"heating_cop": 2.8, "cooling_eer": 3.2, "age_factor": 0.85},
    "vrf": {"heating_cop": 3.5, "cooling_eer": 4.5, "age_factor": 0.90},
    "chiller_ahu": {"heating_cop": 1.0, "cooling_eer": 4.0, "age_factor": 0.85},
    "boiler_radiator": {"heating_cop": 0.88, "cooling_eer": 0.0, "age_factor": 0.80},
    "heat_pump_air": {"heating_cop": 3.2, "cooling_eer": 3.8, "age_factor": 0.92},
    "heat_pump_ground": {"heating_cop": 4.0, "cooling_eer": 5.0, "age_factor": 0.95},
    "district_heating": {"heating_cop": 1.0, "cooling_eer": 0.0, "age_factor": 1.0},
    "district_cooling": {"heating_cop": 0.0, "cooling_eer": 1.0, "age_factor": 1.0},
    "fcu": {"heating_cop": 0.88, "cooling_eer": 3.5, "age_factor": 0.85},
    "packaged_rooftop": {"heating_cop": 2.5, "cooling_eer": 3.0, "age_factor": 0.82},
    "natural_ventilation": {"heating_cop": 0.0, "cooling_eer": 0.0, "age_factor": 1.0},
    "mixed_mode": {"heating_cop": 2.5, "cooling_eer": 3.0, "age_factor": 0.88},
}

# Lighting power density defaults (W/m2) by type - CIBSE SLL
LIGHTING_POWER_DENSITY: Dict[str, float] = {
    "incandescent": 20.0,
    "halogen": 15.0,
    "cfl": 10.0,
    "t8_fluorescent": 12.0,
    "t5_fluorescent": 9.0,
    "led": 6.0,
    "hid": 14.0,
    "metal_halide": 13.0,
}

# DHW demand defaults (litres/person/day) by building type
DHW_DEMAND_LITRES: Dict[str, float] = {
    "office": 10.0,
    "retail": 5.0,
    "warehouse": 5.0,
    "hospital": 120.0,
    "school": 15.0,
    "university": 15.0,
    "hotel": 100.0,
    "residential_multi": 50.0,
    "residential_single": 50.0,
    "mixed_use": 25.0,
    "industrial": 15.0,
    "data_centre": 5.0,
    "leisure": 60.0,
    "restaurant": 30.0,
    "supermarket": 5.0,
}

# EPC band thresholds (kWh/m2/yr primary energy) per EN 15217
EPC_BAND_THRESHOLDS: Dict[str, float] = {
    "A+": 0.0,
    "A": 25.0,
    "B": 50.0,
    "C": 75.0,
    "D": 100.0,
    "E": 125.0,
    "F": 150.0,
    "G": 200.0,
}

# Energy conversion factors to kWh
ENERGY_CONVERSION_TO_KWH: Dict[str, float] = {
    "kWh": 1.0,
    "MWh": 1000.0,
    "GJ": 277.778,
    "MJ": 0.277778,
    "therm": 29.3071,
    "m3_natural_gas": 10.55,
    "litre_fuel_oil": 10.35,
    "litre_lpg": 7.08,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class EnvelopeElement(BaseModel):
    """Building envelope element with thermal properties."""

    element_id: str = Field(default_factory=lambda: f"env-{uuid.uuid4().hex[:8]}")
    element_type: str = Field(default="wall", description="wall|roof|floor|window|door")
    construction: str = Field(default="", description="Construction type key")
    area_sqm: float = Field(default=0.0, ge=0.0, description="Element area in m2")
    u_value_measured: Optional[float] = Field(None, ge=0.0, description="Measured U-value W/m2K")
    u_value_default: float = Field(default=0.0, ge=0.0, description="Default U-value W/m2K")
    orientation: str = Field(default="", description="N|NE|E|SE|S|SW|W|NW|horizontal")
    insulation_thickness_mm: float = Field(default=0.0, ge=0.0, description="Insulation thickness")
    condition: str = Field(default="average", description="good|average|poor")
    notes: str = Field(default="")


class UtilityBillRecord(BaseModel):
    """Monthly utility bill record."""

    bill_id: str = Field(default_factory=lambda: f"bill-{uuid.uuid4().hex[:8]}")
    period: str = Field(default="", description="Period YYYY-MM")
    energy_source: str = Field(default="electricity", description="Energy carrier")
    consumption_kwh: float = Field(default=0.0, ge=0.0, description="Consumption in kWh")
    consumption_native: float = Field(default=0.0, ge=0.0, description="Native unit quantity")
    native_unit: str = Field(default="kWh", description="Native unit of billing")
    cost_eur: float = Field(default=0.0, ge=0.0, description="Bill cost in EUR")
    demand_kw: float = Field(default=0.0, ge=0.0, description="Peak demand kW")
    data_quality: DataQuality = Field(default=DataQuality.MEASURED)
    days_in_period: int = Field(default=30, ge=1, le=366, description="Billing days")


class BMSDataPoint(BaseModel):
    """Building Management System data point."""

    point_id: str = Field(default="", description="BMS point identifier")
    timestamp: str = Field(default="", description="ISO 8601 timestamp")
    point_type: str = Field(default="", description="temperature|power|flow|status")
    value: float = Field(default=0.0, description="Measured value")
    unit: str = Field(default="", description="Unit of measurement")
    zone: str = Field(default="", description="Building zone served")


class OccupancyRecord(BaseModel):
    """Occupancy data for a building zone."""

    zone: str = Field(default="whole_building", description="Zone identifier")
    typical_occupancy: int = Field(default=0, ge=0, description="Typical persons")
    max_occupancy: int = Field(default=0, ge=0, description="Maximum persons")
    occupied_hours_weekday: float = Field(default=10.0, ge=0.0, le=24.0)
    occupied_hours_weekend: float = Field(default=0.0, ge=0.0, le=24.0)
    occupied_weeks_per_year: int = Field(default=50, ge=0, le=52)


class HVACSystem(BaseModel):
    """HVAC system record."""

    system_id: str = Field(default_factory=lambda: f"hvac-{uuid.uuid4().hex[:8]}")
    system_type: HVACSystemType = Field(default=HVACSystemType.SPLIT_SYSTEM)
    name: str = Field(default="", description="System name or tag")
    manufacturer: str = Field(default="")
    model: str = Field(default="")
    year_installed: int = Field(default=0, ge=0)
    heating_capacity_kw: float = Field(default=0.0, ge=0.0)
    cooling_capacity_kw: float = Field(default=0.0, ge=0.0)
    rated_cop: float = Field(default=0.0, ge=0.0, description="Rated COP/EER")
    refrigerant: str = Field(default="", description="Refrigerant type")
    zone_served: str = Field(default="", description="Building zone served")
    area_served_sqm: float = Field(default=0.0, ge=0.0)
    condition: str = Field(default="average", description="good|average|poor")


class LightingSystem(BaseModel):
    """Lighting system record for a zone."""

    zone: str = Field(default="", description="Zone identifier")
    lighting_type: LightingType = Field(default=LightingType.LED)
    installed_power_kw: float = Field(default=0.0, ge=0.0)
    area_sqm: float = Field(default=0.0, ge=0.0, description="Zone area")
    control_type: str = Field(default="manual", description="manual|timer|daylight|occupancy|full_auto")
    annual_hours: float = Field(default=2500.0, ge=0.0, description="Annual operating hours")
    lux_level: float = Field(default=300.0, ge=0.0, description="Maintained illuminance")


class DHWSystem(BaseModel):
    """Domestic hot water system record."""

    system_id: str = Field(default_factory=lambda: f"dhw-{uuid.uuid4().hex[:8]}")
    system_type: DHWSystemType = Field(default=DHWSystemType.GAS_BOILER)
    capacity_kw: float = Field(default=0.0, ge=0.0)
    storage_litres: float = Field(default=0.0, ge=0.0)
    efficiency_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    year_installed: int = Field(default=0, ge=0)
    insulation_condition: str = Field(default="average", description="good|average|poor")
    solar_preheat: bool = Field(default=False)


class RenewableSystem(BaseModel):
    """Renewable energy system record."""

    system_id: str = Field(default_factory=lambda: f"ren-{uuid.uuid4().hex[:8]}")
    technology: str = Field(default="solar_pv", description="solar_pv|solar_thermal|wind|biomass|chp")
    capacity_kw: float = Field(default=0.0, ge=0.0, description="Installed capacity kWp")
    annual_generation_kwh: float = Field(default=0.0, ge=0.0, description="Annual output kWh")
    year_installed: int = Field(default=0, ge=0)
    tilt_degrees: float = Field(default=30.0, ge=0.0, le=90.0, description="Panel tilt")
    azimuth_degrees: float = Field(default=180.0, ge=0.0, le=360.0, description="Panel azimuth")


class EnvelopeAssessmentResult(BaseModel):
    """Results from building envelope assessment."""

    total_heat_loss_coefficient_w_k: float = Field(default=0.0, ge=0.0)
    fabric_heat_loss_w_k: float = Field(default=0.0, ge=0.0)
    ventilation_heat_loss_w_k: float = Field(default=0.0, ge=0.0)
    thermal_bridging_w_k: float = Field(default=0.0, ge=0.0)
    weighted_u_value_walls: float = Field(default=0.0, ge=0.0)
    weighted_u_value_roof: float = Field(default=0.0, ge=0.0)
    weighted_u_value_floor: float = Field(default=0.0, ge=0.0)
    weighted_u_value_windows: float = Field(default=0.0, ge=0.0)
    air_permeability_m3_h_m2: float = Field(default=0.0, ge=0.0)
    envelope_rating: str = Field(default="average", description="Rating: good|average|poor")


class SystemsAssessmentResult(BaseModel):
    """Results from building systems assessment."""

    heating_demand_kwh: float = Field(default=0.0, ge=0.0)
    cooling_demand_kwh: float = Field(default=0.0, ge=0.0)
    lighting_demand_kwh: float = Field(default=0.0, ge=0.0)
    dhw_demand_kwh: float = Field(default=0.0, ge=0.0)
    total_energy_demand_kwh: float = Field(default=0.0, ge=0.0)
    renewable_generation_kwh: float = Field(default=0.0, ge=0.0)
    net_energy_demand_kwh: float = Field(default=0.0, ge=0.0)
    hvac_efficiency_rating: str = Field(default="average")
    lighting_efficiency_rating: str = Field(default="average")
    dhw_efficiency_rating: str = Field(default="average")


class ImprovementRecommendation(BaseModel):
    """Improvement recommendation from assessment."""

    recommendation_id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:8]}")
    title: str = Field(default="")
    description: str = Field(default="")
    category: str = Field(default="", description="envelope|hvac|lighting|dhw|renewables|controls")
    severity: FindingSeverity = Field(default=FindingSeverity.MEDIUM)
    annual_savings_kwh: float = Field(default=0.0, ge=0.0)
    annual_savings_eur: float = Field(default=0.0, ge=0.0)
    implementation_cost_eur: float = Field(default=0.0, ge=0.0)
    simple_payback_years: float = Field(default=0.0, ge=0.0)
    co2_reduction_kg: float = Field(default=0.0, ge=0.0)
    epc_band_improvement: str = Field(default="", description="e.g. D -> C")
    priority: str = Field(default="medium", description="critical|high|medium|low")


class BuildingData(BaseModel):
    """Complete building information for assessment."""

    building_id: str = Field(default_factory=lambda: f"bld-{uuid.uuid4().hex[:8]}")
    building_name: str = Field(default="", description="Building name")
    address: str = Field(default="", description="Full address")
    country: str = Field(default="GB", description="ISO 3166-1 alpha-2")
    postcode: str = Field(default="")
    latitude: float = Field(default=51.5, description="Latitude")
    longitude: float = Field(default=-0.1, description="Longitude")
    building_type: BuildingType = Field(default=BuildingType.OFFICE)
    construction_era: ConstructionEra = Field(default=ConstructionEra.ERA_1991_2005)
    climate_zone: ClimateZone = Field(default=ClimateZone.OCEANIC)
    year_built: int = Field(default=2000, ge=1800, le=2030)
    year_last_refurbished: int = Field(default=0, ge=0, le=2030)
    total_floor_area_sqm: float = Field(default=0.0, ge=0.0, description="Gross internal area")
    net_lettable_area_sqm: float = Field(default=0.0, ge=0.0, description="NLA")
    number_of_floors: int = Field(default=1, ge=1, le=200)
    floor_to_ceiling_height_m: float = Field(default=2.7, ge=2.0, le=10.0)
    building_volume_m3: float = Field(default=0.0, ge=0.0)
    envelope_area_sqm: float = Field(default=0.0, ge=0.0, description="Total envelope area")
    window_to_wall_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    envelope_elements: List[EnvelopeElement] = Field(default_factory=list)
    hvac_systems: List[HVACSystem] = Field(default_factory=list)
    lighting_systems: List[LightingSystem] = Field(default_factory=list)
    dhw_systems: List[DHWSystem] = Field(default_factory=list)
    renewable_systems: List[RenewableSystem] = Field(default_factory=list)
    utility_bills: List[UtilityBillRecord] = Field(default_factory=list)
    bms_data: List[BMSDataPoint] = Field(default_factory=list)
    occupancy: List[OccupancyRecord] = Field(default_factory=list)


class InitialBuildingAssessmentInput(BaseModel):
    """Input data model for InitialBuildingAssessmentWorkflow."""

    building: BuildingData = Field(..., description="Building data")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    baseline_months: int = Field(default=12, ge=6, le=36)
    electricity_ef_kgco2_kwh: float = Field(default=0.207, ge=0.0)
    gas_ef_kgco2_kwh: float = Field(default=0.183, ge=0.0)
    discount_rate_pct: float = Field(default=8.0, ge=0.0, le=30.0)
    project_lifetime_years: int = Field(default=15, ge=1, le=30)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("building")
    @classmethod
    def validate_building(cls, v: BuildingData) -> BuildingData:
        """Ensure building has basic required data."""
        if not v.building_name and not v.building_id:
            raise ValueError("Building must have a name or ID")
        return v


class InitialBuildingAssessmentResult(BaseModel):
    """Complete result from initial building assessment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="initial_building_assessment")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    building_id: str = Field(default="")
    building_type: str = Field(default="")
    total_floor_area_sqm: float = Field(default=0.0, ge=0.0)
    envelope_assessment: Optional[EnvelopeAssessmentResult] = None
    systems_assessment: Optional[SystemsAssessmentResult] = None
    total_annual_consumption_kwh: float = Field(default=0.0, ge=0.0)
    total_annual_cost_eur: float = Field(default=0.0, ge=0.0)
    eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    epc_band: str = Field(default="")
    primary_energy_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    co2_emissions_kg_per_sqm: float = Field(default=0.0, ge=0.0)
    total_co2_emissions_kg: float = Field(default=0.0, ge=0.0)
    benchmark_comparison: str = Field(default="", description="vs typical|good|best")
    recommendations: List[ImprovementRecommendation] = Field(default_factory=list)
    total_potential_savings_kwh: float = Field(default=0.0, ge=0.0)
    total_potential_savings_eur: float = Field(default=0.0, ge=0.0)
    total_potential_co2_reduction_kg: float = Field(default=0.0, ge=0.0)
    potential_epc_improvement: str = Field(default="")
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class InitialBuildingAssessmentWorkflow:
    """
    5-phase comprehensive building energy assessment workflow.

    Performs building registration and data collection, envelope thermal
    analysis, building systems assessment (HVAC, lighting, DHW, renewables),
    and generates an EPC rating with benchmarking and improvement plan.

    Zero-hallucination: all calculations use deterministic formulas,
    validated U-values, CIBSE/ASHRAE benchmarks, and EN 15603 primary
    energy factors. No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _envelope_result: Envelope assessment output.
        _systems_result: Systems assessment output.
        _recommendations: Improvement recommendations.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = InitialBuildingAssessmentWorkflow()
        >>> inp = InitialBuildingAssessmentInput(building=building_data)
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize InitialBuildingAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._envelope_result: Optional[EnvelopeAssessmentResult] = None
        self._systems_result: Optional[SystemsAssessmentResult] = None
        self._recommendations: List[ImprovementRecommendation] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[InitialBuildingAssessmentInput] = None,
        building: Optional[BuildingData] = None,
        reporting_year: int = 2025,
    ) -> InitialBuildingAssessmentResult:
        """
        Execute the 5-phase initial building assessment workflow.

        Args:
            input_data: Full input model (preferred).
            building: Building data (fallback).
            reporting_year: Reporting year (fallback).

        Returns:
            InitialBuildingAssessmentResult with EPC, benchmarks, recommendations.

        Raises:
            ValueError: If no building data is provided.
        """
        if input_data is None:
            if building is None:
                raise ValueError("Either input_data or building must be provided")
            input_data = InitialBuildingAssessmentInput(
                building=building,
                reporting_year=reporting_year,
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting initial building assessment workflow %s for building=%s type=%s",
            self.workflow_id,
            input_data.building.building_name,
            input_data.building.building_type.value,
        )

        self._phase_results = []
        self._envelope_result = None
        self._systems_result = None
        self._recommendations = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Building Registration
            phase1 = await self._phase_building_registration(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Data Collection
            phase2 = await self._phase_data_collection(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Envelope Assessment
            phase3 = await self._phase_envelope_assessment(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Systems Assessment
            phase4 = await self._phase_systems_assessment(input_data)
            self._phase_results.append(phase4)

            # Phase 5: Report Generation
            phase5 = await self._phase_report_generation(input_data)
            self._phase_results.append(phase5)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "Initial building assessment workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        bld = input_data.building

        # Compute aggregate metrics
        total_consumption = sum(b.consumption_kwh for b in bld.utility_bills)
        total_cost = sum(b.cost_eur for b in bld.utility_bills)
        floor_area = bld.total_floor_area_sqm or 1.0
        eui = total_consumption / floor_area if floor_area > 0 else 0.0

        # EPC and emissions from systems assessment if available
        primary_energy_sqm = 0.0
        co2_sqm = 0.0
        total_co2 = 0.0
        epc_band_str = ""
        benchmark_str = ""

        if self._systems_result is not None:
            net_demand = self._systems_result.net_energy_demand_kwh
            primary_energy_sqm = self._compute_primary_energy_per_sqm(
                net_demand, bld, floor_area
            )
            co2_sqm = self._compute_co2_per_sqm(net_demand, bld, floor_area)
            total_co2 = co2_sqm * floor_area
            epc_band_str = self._assign_epc_band(primary_energy_sqm)

        benchmark_str = self._compare_benchmark(eui, bld.building_type.value)

        total_savings_kwh = sum(r.annual_savings_kwh for r in self._recommendations)
        total_savings_eur = sum(r.annual_savings_eur for r in self._recommendations)
        total_co2_reduction = sum(r.co2_reduction_kg for r in self._recommendations)

        # Potential EPC improvement
        improved_primary = primary_energy_sqm - (
            total_savings_kwh * PRIMARY_ENERGY_FACTORS.get("electricity", 2.5) / floor_area
        ) if floor_area > 0 else primary_energy_sqm
        improved_epc = self._assign_epc_band(max(improved_primary, 0.0))
        potential_improvement = (
            f"{epc_band_str} -> {improved_epc}" if improved_epc != epc_band_str else "None"
        )

        result = InitialBuildingAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            building_id=bld.building_id,
            building_type=bld.building_type.value,
            total_floor_area_sqm=round(floor_area, 2),
            envelope_assessment=self._envelope_result,
            systems_assessment=self._systems_result,
            total_annual_consumption_kwh=round(total_consumption, 2),
            total_annual_cost_eur=round(total_cost, 2),
            eui_kwh_per_sqm=round(eui, 2),
            epc_band=epc_band_str,
            primary_energy_kwh_per_sqm=round(primary_energy_sqm, 2),
            co2_emissions_kg_per_sqm=round(co2_sqm, 2),
            total_co2_emissions_kg=round(total_co2, 2),
            benchmark_comparison=benchmark_str,
            recommendations=self._recommendations,
            total_potential_savings_kwh=round(total_savings_kwh, 2),
            total_potential_savings_eur=round(total_savings_eur, 2),
            total_potential_co2_reduction_kg=round(total_co2_reduction, 2),
            potential_epc_improvement=potential_improvement,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Initial building assessment workflow %s completed in %.2fs "
            "status=%s EPC=%s EUI=%.1f kWh/m2",
            self.workflow_id, elapsed, overall_status.value, epc_band_str, eui,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Building Registration
    # -------------------------------------------------------------------------

    async def _phase_building_registration(
        self, input_data: InitialBuildingAssessmentInput
    ) -> PhaseResult:
        """Collect building data, type, age, floor area, location."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bld = input_data.building

        # Validate core building data
        if bld.total_floor_area_sqm <= 0:
            warnings.append("Building total_floor_area_sqm is zero or not provided")
        if bld.year_built <= 0:
            warnings.append("Year built not specified")
        if not bld.address:
            warnings.append("Building address not provided")

        # Auto-compute volume if not supplied
        volume = bld.building_volume_m3
        if volume <= 0 and bld.total_floor_area_sqm > 0:
            volume = (
                bld.total_floor_area_sqm
                * bld.floor_to_ceiling_height_m
                * bld.number_of_floors
            )

        # Auto-estimate envelope area if not supplied
        envelope = bld.envelope_area_sqm
        if envelope <= 0 and bld.total_floor_area_sqm > 0:
            footprint = bld.total_floor_area_sqm / max(bld.number_of_floors, 1)
            perimeter = 4.0 * math.sqrt(footprint)
            wall_height = bld.floor_to_ceiling_height_m * bld.number_of_floors
            envelope = (perimeter * wall_height) + (2.0 * footprint)

        outputs["building_id"] = bld.building_id
        outputs["building_name"] = bld.building_name
        outputs["building_type"] = bld.building_type.value
        outputs["construction_era"] = bld.construction_era.value
        outputs["year_built"] = bld.year_built
        outputs["climate_zone"] = bld.climate_zone.value
        outputs["total_floor_area_sqm"] = round(bld.total_floor_area_sqm, 2)
        outputs["number_of_floors"] = bld.number_of_floors
        outputs["building_volume_m3"] = round(volume, 2)
        outputs["envelope_area_sqm"] = round(envelope, 2)
        outputs["window_to_wall_ratio"] = round(bld.window_to_wall_ratio, 2)
        outputs["hvac_system_count"] = len(bld.hvac_systems)
        outputs["lighting_zone_count"] = len(bld.lighting_systems)
        outputs["dhw_system_count"] = len(bld.dhw_systems)
        outputs["renewable_system_count"] = len(bld.renewable_systems)
        outputs["latitude"] = bld.latitude
        outputs["longitude"] = bld.longitude
        outputs["country"] = bld.country

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 BuildingRegistration: %s (%s), %.0f m2, %d floors",
            bld.building_name, bld.building_type.value,
            bld.total_floor_area_sqm, bld.number_of_floors,
        )
        return PhaseResult(
            phase_name="building_registration", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: InitialBuildingAssessmentInput
    ) -> PhaseResult:
        """Gather utility bills, BMS data, drawings, occupancy."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bld = input_data.building

        # Validate utility bill coverage
        bill_periods = set()
        total_bill_records = len(bld.utility_bills)
        for bill in bld.utility_bills:
            if bill.period:
                bill_periods.add(bill.period)

        months_covered = len(bill_periods)
        if months_covered < input_data.baseline_months:
            warnings.append(
                f"Only {months_covered} months of utility data; "
                f"{input_data.baseline_months} required"
            )

        # Data quality assessment
        measured_count = sum(
            1 for b in bld.utility_bills if b.data_quality == DataQuality.MEASURED
        )
        quality_ratio = measured_count / max(total_bill_records, 1) * 100

        # Aggregate consumption by energy source
        source_totals: Dict[str, float] = {}
        source_costs: Dict[str, float] = {}
        for bill in bld.utility_bills:
            src = bill.energy_source
            source_totals[src] = source_totals.get(src, 0.0) + bill.consumption_kwh
            source_costs[src] = source_costs.get(src, 0.0) + bill.cost_eur

        # BMS data summary
        bms_point_count = len(bld.bms_data)
        bms_types = set(p.point_type for p in bld.bms_data if p.point_type)
        if not bld.bms_data:
            warnings.append("No BMS data provided; assessment will use utility data only")

        # Occupancy summary
        total_occupants = sum(o.typical_occupancy for o in bld.occupancy)
        if not bld.occupancy:
            warnings.append("No occupancy data provided; using building type defaults")

        # Envelope element coverage
        element_types = set(e.element_type for e in bld.envelope_elements)
        missing_elements = {"wall", "roof", "floor", "window"} - element_types
        if missing_elements:
            warnings.append(
                f"Envelope elements missing: {', '.join(missing_elements)}; "
                "defaults will be used"
            )

        outputs["utility_bill_records"] = total_bill_records
        outputs["months_covered"] = months_covered
        outputs["data_quality_measured_pct"] = round(quality_ratio, 1)
        outputs["consumption_by_source_kwh"] = {
            k: round(v, 2) for k, v in source_totals.items()
        }
        outputs["cost_by_source_eur"] = {
            k: round(v, 2) for k, v in source_costs.items()
        }
        outputs["bms_data_points"] = bms_point_count
        outputs["bms_point_types"] = sorted(bms_types)
        outputs["total_occupants"] = total_occupants
        outputs["envelope_elements_provided"] = len(bld.envelope_elements)
        outputs["envelope_element_types"] = sorted(element_types)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DataCollection: %d bills, %d months, quality=%.1f%%, "
            "%d BMS points, %d occupants",
            total_bill_records, months_covered, quality_ratio,
            bms_point_count, total_occupants,
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Envelope Assessment
    # -------------------------------------------------------------------------

    async def _phase_envelope_assessment(
        self, input_data: InitialBuildingAssessmentInput
    ) -> PhaseResult:
        """Assess building envelope: U-values, air tightness, heat loss."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bld = input_data.building

        # Calculate areas if not explicitly provided
        floor_area = bld.total_floor_area_sqm or 1.0
        footprint = floor_area / max(bld.number_of_floors, 1)
        perimeter = 4.0 * math.sqrt(footprint)
        wall_height = bld.floor_to_ceiling_height_m * bld.number_of_floors
        total_wall_area = perimeter * wall_height
        window_area = total_wall_area * bld.window_to_wall_ratio
        opaque_wall_area = total_wall_area - window_area
        roof_area = footprint
        floor_area_ground = footprint

        # Volume for ventilation heat loss
        volume = bld.building_volume_m3
        if volume <= 0:
            volume = floor_area * bld.floor_to_ceiling_height_m

        # Determine U-values from envelope elements or defaults
        wall_u, wall_area_calc = self._calc_weighted_u_value(
            bld.envelope_elements, "wall", opaque_wall_area,
            U_VALUES_WALL.get(bld.construction_era.value.replace("era_", ""), 1.0),
        )
        roof_u, roof_area_calc = self._calc_weighted_u_value(
            bld.envelope_elements, "roof", roof_area,
            U_VALUES_ROOF.get("flat_uninsulated", 1.50),
        )
        floor_u = U_VALUES_FLOOR.get(bld.construction_era.value, 0.45)
        window_u = self._calc_window_u_value(bld.envelope_elements, window_area)

        # Fabric heat loss (W/K): sum of (U * A) for each element
        fabric_wall = wall_u * opaque_wall_area
        fabric_roof = roof_u * roof_area
        fabric_floor = floor_u * floor_area_ground
        fabric_window = window_u * window_area
        fabric_total = fabric_wall + fabric_roof + fabric_floor + fabric_window

        # Thermal bridging allowance (y-value * envelope area) - EN ISO 14683
        envelope_total = opaque_wall_area + window_area + roof_area + floor_area_ground
        y_value = 0.15 if bld.construction_era.value in ("pre_1950", "1950_1975") else 0.10
        thermal_bridging = y_value * envelope_total

        # Air permeability and ventilation heat loss
        air_perm = AIR_PERMEABILITY_DEFAULTS.get(bld.construction_era.value, 10.0)
        # Ventilation heat loss (W/K): 0.33 * n * V
        # n = air changes per hour at 50Pa / 20 (rule of thumb)
        infiltration_ach = (air_perm * envelope_total) / (volume * 20.0) if volume > 0 else 0.5
        ventilation_hl = 0.33 * infiltration_ach * volume

        total_hl = fabric_total + thermal_bridging + ventilation_hl

        # Envelope rating
        if total_hl / floor_area < 1.5:
            rating = "good"
        elif total_hl / floor_area < 3.0:
            rating = "average"
        else:
            rating = "poor"

        self._envelope_result = EnvelopeAssessmentResult(
            total_heat_loss_coefficient_w_k=round(total_hl, 2),
            fabric_heat_loss_w_k=round(fabric_total, 2),
            ventilation_heat_loss_w_k=round(ventilation_hl, 2),
            thermal_bridging_w_k=round(thermal_bridging, 2),
            weighted_u_value_walls=round(wall_u, 3),
            weighted_u_value_roof=round(roof_u, 3),
            weighted_u_value_floor=round(floor_u, 3),
            weighted_u_value_windows=round(window_u, 3),
            air_permeability_m3_h_m2=round(air_perm, 1),
            envelope_rating=rating,
        )

        outputs["total_heat_loss_w_k"] = round(total_hl, 2)
        outputs["fabric_heat_loss_w_k"] = round(fabric_total, 2)
        outputs["ventilation_heat_loss_w_k"] = round(ventilation_hl, 2)
        outputs["thermal_bridging_w_k"] = round(thermal_bridging, 2)
        outputs["wall_u_value"] = round(wall_u, 3)
        outputs["roof_u_value"] = round(roof_u, 3)
        outputs["floor_u_value"] = round(floor_u, 3)
        outputs["window_u_value"] = round(window_u, 3)
        outputs["air_permeability"] = round(air_perm, 1)
        outputs["envelope_rating"] = rating
        outputs["opaque_wall_area_sqm"] = round(opaque_wall_area, 2)
        outputs["window_area_sqm"] = round(window_area, 2)
        outputs["roof_area_sqm"] = round(roof_area, 2)
        outputs["floor_area_ground_sqm"] = round(floor_area_ground, 2)
        outputs["heat_loss_per_sqm_w_k"] = round(total_hl / floor_area, 3)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 EnvelopeAssessment: HLC=%.1f W/K, U-walls=%.2f, "
            "U-roof=%.2f, U-windows=%.2f, rating=%s",
            total_hl, wall_u, roof_u, window_u, rating,
        )
        return PhaseResult(
            phase_name="envelope_assessment", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calc_weighted_u_value(
        self,
        elements: List[EnvelopeElement],
        element_type: str,
        default_area: float,
        default_u: float,
    ) -> Tuple[float, float]:
        """Calculate area-weighted U-value for an element type."""
        matching = [e for e in elements if e.element_type == element_type]
        if not matching:
            return default_u, default_area

        total_ua = 0.0
        total_area = 0.0
        for elem in matching:
            area = elem.area_sqm if elem.area_sqm > 0 else default_area / max(len(matching), 1)
            u_val = elem.u_value_measured if elem.u_value_measured is not None else elem.u_value_default
            if u_val <= 0:
                construction_key = elem.construction
                if element_type == "wall":
                    u_val = U_VALUES_WALL.get(construction_key, default_u)
                elif element_type == "roof":
                    u_val = U_VALUES_ROOF.get(construction_key, default_u)
                else:
                    u_val = default_u
            total_ua += u_val * area
            total_area += area

        weighted_u = total_ua / total_area if total_area > 0 else default_u
        return weighted_u, total_area

    def _calc_window_u_value(
        self, elements: List[EnvelopeElement], default_area: float
    ) -> float:
        """Calculate weighted U-value for windows."""
        windows = [e for e in elements if e.element_type == "window"]
        if not windows:
            return U_VALUES_GLAZING.get("double", 2.80)

        total_ua = 0.0
        total_area = 0.0
        for win in windows:
            area = win.area_sqm if win.area_sqm > 0 else default_area / max(len(windows), 1)
            u_val = win.u_value_measured if win.u_value_measured is not None else win.u_value_default
            if u_val <= 0:
                u_val = U_VALUES_GLAZING.get(win.construction, 2.80)
            total_ua += u_val * area
            total_area += area

        return total_ua / total_area if total_area > 0 else 2.80

    # -------------------------------------------------------------------------
    # Phase 4: Systems Assessment
    # -------------------------------------------------------------------------

    async def _phase_systems_assessment(
        self, input_data: InitialBuildingAssessmentInput
    ) -> PhaseResult:
        """Assess HVAC, lighting, DHW, and renewables systems."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bld = input_data.building
        floor_area = bld.total_floor_area_sqm or 1.0

        # Heating demand calculation using degree-day method
        hdd = HDD_BY_CLIMATE_ZONE.get(bld.climate_zone.value, 3200.0)
        cdd = CDD_BY_CLIMATE_ZONE.get(bld.climate_zone.value, 200.0)

        hlc = self._envelope_result.total_heat_loss_coefficient_w_k if self._envelope_result else 300.0
        heating_demand = hlc * hdd * 24.0 / 1000.0  # kWh

        # Cooling demand simplified estimation
        cooling_demand = 0.0
        if cdd > 50:
            internal_gains_w_sqm = 25.0  # W/m2 typical office internal gains
            cooling_demand = (
                internal_gains_w_sqm * floor_area * cdd * 24.0 / 1000.0 / 1000.0
                + hlc * cdd * 24.0 / 1000.0 * 0.3
            )

        # HVAC efficiency adjustment
        hvac_heating_eff, hvac_cooling_eff = self._assess_hvac_efficiency(bld)
        adjusted_heating = heating_demand / max(hvac_heating_eff, 0.5)
        adjusted_cooling = cooling_demand / max(hvac_cooling_eff, 1.0)

        # Lighting demand
        lighting_demand = self._assess_lighting_demand(bld, floor_area)

        # DHW demand
        dhw_demand = self._assess_dhw_demand(bld, floor_area)

        # Renewable generation
        renewable_gen = sum(r.annual_generation_kwh for r in bld.renewable_systems)
        if not bld.renewable_systems:
            renewable_gen = 0.0

        total_demand = adjusted_heating + adjusted_cooling + lighting_demand + dhw_demand
        net_demand = max(0.0, total_demand - renewable_gen)

        # Efficiency ratings
        hvac_rating = self._rate_efficiency(hvac_heating_eff, 0.85, 2.5, 3.5)
        lpd = lighting_demand / floor_area / 2500.0 if floor_area > 0 else 10.0  # W/m2
        lighting_rating = "good" if lpd <= 7.0 else ("average" if lpd <= 12.0 else "poor")
        dhw_eff = self._get_dhw_efficiency(bld)
        dhw_rating = "good" if dhw_eff >= 90.0 else ("average" if dhw_eff >= 75.0 else "poor")

        self._systems_result = SystemsAssessmentResult(
            heating_demand_kwh=round(adjusted_heating, 2),
            cooling_demand_kwh=round(adjusted_cooling, 2),
            lighting_demand_kwh=round(lighting_demand, 2),
            dhw_demand_kwh=round(dhw_demand, 2),
            total_energy_demand_kwh=round(total_demand, 2),
            renewable_generation_kwh=round(renewable_gen, 2),
            net_energy_demand_kwh=round(net_demand, 2),
            hvac_efficiency_rating=hvac_rating,
            lighting_efficiency_rating=lighting_rating,
            dhw_efficiency_rating=dhw_rating,
        )

        outputs["heating_demand_kwh"] = round(adjusted_heating, 2)
        outputs["cooling_demand_kwh"] = round(adjusted_cooling, 2)
        outputs["lighting_demand_kwh"] = round(lighting_demand, 2)
        outputs["dhw_demand_kwh"] = round(dhw_demand, 2)
        outputs["total_demand_kwh"] = round(total_demand, 2)
        outputs["renewable_generation_kwh"] = round(renewable_gen, 2)
        outputs["net_demand_kwh"] = round(net_demand, 2)
        outputs["hvac_heating_efficiency"] = round(hvac_heating_eff, 3)
        outputs["hvac_cooling_efficiency"] = round(hvac_cooling_eff, 3)
        outputs["hvac_rating"] = hvac_rating
        outputs["lighting_power_density_w_sqm"] = round(lpd * 1000.0, 2) if lpd < 1 else round(lpd, 2)
        outputs["lighting_rating"] = lighting_rating
        outputs["dhw_efficiency_pct"] = round(dhw_eff, 1)
        outputs["dhw_rating"] = dhw_rating
        outputs["hdd_used"] = hdd
        outputs["cdd_used"] = cdd

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 SystemsAssessment: heating=%.0f cooling=%.0f lighting=%.0f "
            "dhw=%.0f total=%.0f net=%.0f kWh",
            adjusted_heating, adjusted_cooling, lighting_demand,
            dhw_demand, total_demand, net_demand,
        )
        return PhaseResult(
            phase_name="systems_assessment", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_hvac_efficiency(self, bld: BuildingData) -> Tuple[float, float]:
        """Assess weighted HVAC heating and cooling efficiency."""
        if not bld.hvac_systems:
            return 0.85, 3.0  # Default boiler / split system

        total_heating_cap = 0.0
        total_cooling_cap = 0.0
        weighted_heating = 0.0
        weighted_cooling = 0.0

        for sys in bld.hvac_systems:
            defaults = HVAC_EFFICIENCY_DEFAULTS.get(sys.system_type.value, {})
            h_cop = sys.rated_cop if sys.rated_cop > 0 else defaults.get("heating_cop", 0.85)
            c_eer = sys.rated_cop if sys.rated_cop > 0 else defaults.get("cooling_eer", 3.0)
            age_factor = defaults.get("age_factor", 0.85)

            # Age degradation
            if sys.year_installed > 0:
                age = max(0, datetime.utcnow().year - sys.year_installed)
                degradation = max(0.7, 1.0 - age * 0.01)
            else:
                degradation = age_factor

            if sys.heating_capacity_kw > 0:
                weighted_heating += h_cop * degradation * sys.heating_capacity_kw
                total_heating_cap += sys.heating_capacity_kw
            if sys.cooling_capacity_kw > 0:
                weighted_cooling += c_eer * degradation * sys.cooling_capacity_kw
                total_cooling_cap += sys.cooling_capacity_kw

        avg_heating = weighted_heating / total_heating_cap if total_heating_cap > 0 else 0.85
        avg_cooling = weighted_cooling / total_cooling_cap if total_cooling_cap > 0 else 3.0
        return avg_heating, avg_cooling

    def _assess_lighting_demand(self, bld: BuildingData, floor_area: float) -> float:
        """Calculate annual lighting energy demand."""
        if bld.lighting_systems:
            total_kwh = 0.0
            for ls in bld.lighting_systems:
                control_factor = {
                    "manual": 1.0,
                    "timer": 0.90,
                    "daylight": 0.80,
                    "occupancy": 0.75,
                    "full_auto": 0.65,
                }.get(ls.control_type, 1.0)
                total_kwh += ls.installed_power_kw * ls.annual_hours * control_factor
            return total_kwh

        # Default estimation based on building type
        default_lpd = LIGHTING_POWER_DENSITY.get("t8_fluorescent", 12.0)
        annual_hours = 2500.0
        return default_lpd * floor_area * annual_hours / 1000.0

    def _assess_dhw_demand(self, bld: BuildingData, floor_area: float) -> float:
        """Calculate annual DHW energy demand."""
        occupants = sum(o.typical_occupancy for o in bld.occupancy)
        if occupants <= 0:
            # Estimate from floor area and building type
            density = {"office": 10.0, "retail": 15.0, "hospital": 20.0}.get(
                bld.building_type.value, 12.0
            )
            occupants = max(1, int(floor_area / density))

        litres_per_day = DHW_DEMAND_LITRES.get(bld.building_type.value, 25.0)
        daily_litres = occupants * litres_per_day
        # Energy: Q = m * Cp * dT, water Cp=4.186 kJ/(kg*K), dT=35K
        daily_kwh = daily_litres * 4.186 * 35.0 / 3600.0
        annual_kwh = daily_kwh * 365.0

        # Apply system efficiency
        dhw_eff = self._get_dhw_efficiency(bld) / 100.0
        return annual_kwh / max(dhw_eff, 0.5)

    def _get_dhw_efficiency(self, bld: BuildingData) -> float:
        """Get average DHW system efficiency."""
        if not bld.dhw_systems:
            return 80.0
        total_eff = sum(s.efficiency_pct for s in bld.dhw_systems)
        return total_eff / len(bld.dhw_systems)

    @staticmethod
    def _rate_efficiency(value: float, poor_threshold: float, avg_threshold: float, good_threshold: float) -> str:
        """Rate efficiency as good/average/poor."""
        if value >= good_threshold:
            return "good"
        if value >= avg_threshold:
            return "average"
        return "poor"

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: InitialBuildingAssessmentInput
    ) -> PhaseResult:
        """Generate EPC rating, benchmark comparison, improvement plan."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bld = input_data.building
        floor_area = bld.total_floor_area_sqm or 1.0

        # Generate recommendations
        self._recommendations = self._generate_recommendations(input_data)

        # Calculate totals
        total_savings_kwh = sum(r.annual_savings_kwh for r in self._recommendations)
        total_savings_eur = sum(r.annual_savings_eur for r in self._recommendations)
        total_investment = sum(r.implementation_cost_eur for r in self._recommendations)
        total_co2_red = sum(r.co2_reduction_kg for r in self._recommendations)

        # EPC computation
        net_demand = self._systems_result.net_energy_demand_kwh if self._systems_result else 0.0
        primary_per_sqm = self._compute_primary_energy_per_sqm(net_demand, bld, floor_area)
        epc_band = self._assign_epc_band(primary_per_sqm)

        # Benchmark comparison
        eui = net_demand / floor_area if floor_area > 0 else 0.0
        benchmark = self._compare_benchmark(eui, bld.building_type.value)

        outputs["recommendations_count"] = len(self._recommendations)
        outputs["total_savings_kwh"] = round(total_savings_kwh, 2)
        outputs["total_savings_eur"] = round(total_savings_eur, 2)
        outputs["total_investment_eur"] = round(total_investment, 2)
        outputs["total_co2_reduction_kg"] = round(total_co2_red, 2)
        outputs["portfolio_payback_years"] = round(
            total_investment / total_savings_eur if total_savings_eur > 0 else 0.0, 2
        )
        outputs["epc_band"] = epc_band
        outputs["primary_energy_kwh_per_sqm"] = round(primary_per_sqm, 2)
        outputs["eui_kwh_per_sqm"] = round(eui, 2)
        outputs["benchmark_comparison"] = benchmark
        outputs["recommendations_by_category"] = self._count_by_category()

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 ReportGeneration: %d recommendations, savings=%.0f kWh/yr, "
            "EPC=%s, benchmark=%s",
            len(self._recommendations), total_savings_kwh, epc_band, benchmark,
        )
        return PhaseResult(
            phase_name="report_generation", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_recommendations(
        self, input_data: InitialBuildingAssessmentInput
    ) -> List[ImprovementRecommendation]:
        """Generate improvement recommendations from envelope and systems assessments."""
        recommendations: List[ImprovementRecommendation] = []
        bld = input_data.building
        floor_area = bld.total_floor_area_sqm or 1.0
        cost_per_kwh = self._estimate_cost_per_kwh(bld)
        elec_ef = input_data.electricity_ef_kgco2_kwh
        gas_ef = input_data.gas_ef_kgco2_kwh

        # Envelope recommendations
        if self._envelope_result:
            env = self._envelope_result
            # Wall insulation
            if env.weighted_u_value_walls > 0.35:
                target_u = 0.22
                footprint = floor_area / max(bld.number_of_floors, 1)
                wall_area = 4.0 * math.sqrt(footprint) * bld.floor_to_ceiling_height_m * bld.number_of_floors
                opaque_area = wall_area * (1.0 - bld.window_to_wall_ratio)
                hdd = HDD_BY_CLIMATE_ZONE.get(bld.climate_zone.value, 3200.0)
                savings_kwh = (env.weighted_u_value_walls - target_u) * opaque_area * hdd * 24.0 / 1000.0
                savings_eur = savings_kwh * cost_per_kwh
                cost = opaque_area * 120.0  # EUR/m2 for EWI
                payback = cost / savings_eur if savings_eur > 0 else 99.0
                co2_red = savings_kwh * gas_ef
                recommendations.append(ImprovementRecommendation(
                    title="External wall insulation upgrade",
                    description=(
                        f"Upgrade wall U-value from {env.weighted_u_value_walls:.2f} to "
                        f"{target_u:.2f} W/m2K via external wall insulation system."
                    ),
                    category="envelope",
                    severity=FindingSeverity.HIGH if savings_kwh > 10000 else FindingSeverity.MEDIUM,
                    annual_savings_kwh=round(savings_kwh, 2),
                    annual_savings_eur=round(savings_eur, 2),
                    implementation_cost_eur=round(cost, 2),
                    simple_payback_years=round(payback, 2),
                    co2_reduction_kg=round(co2_red, 2),
                    priority="high" if payback < 8 else "medium",
                ))

            # Window upgrade
            if env.weighted_u_value_windows > 1.8:
                target_u = 1.0
                footprint = floor_area / max(bld.number_of_floors, 1)
                wall_area = 4.0 * math.sqrt(footprint) * bld.floor_to_ceiling_height_m * bld.number_of_floors
                win_area = wall_area * bld.window_to_wall_ratio
                hdd = HDD_BY_CLIMATE_ZONE.get(bld.climate_zone.value, 3200.0)
                savings_kwh = (env.weighted_u_value_windows - target_u) * win_area * hdd * 24.0 / 1000.0
                savings_eur = savings_kwh * cost_per_kwh
                cost = win_area * 350.0  # EUR/m2 for triple glazing
                payback = cost / savings_eur if savings_eur > 0 else 99.0
                co2_red = savings_kwh * gas_ef
                recommendations.append(ImprovementRecommendation(
                    title="Window upgrade to triple glazing",
                    description=(
                        f"Replace windows (U={env.weighted_u_value_windows:.2f}) with "
                        f"triple low-e glazing (U={target_u:.2f} W/m2K)."
                    ),
                    category="envelope",
                    severity=FindingSeverity.MEDIUM,
                    annual_savings_kwh=round(savings_kwh, 2),
                    annual_savings_eur=round(savings_eur, 2),
                    implementation_cost_eur=round(cost, 2),
                    simple_payback_years=round(payback, 2),
                    co2_reduction_kg=round(co2_red, 2),
                    priority="medium",
                ))

            # Roof insulation
            if env.weighted_u_value_roof > 0.25:
                target_u = 0.15
                footprint = floor_area / max(bld.number_of_floors, 1)
                hdd = HDD_BY_CLIMATE_ZONE.get(bld.climate_zone.value, 3200.0)
                savings_kwh = (env.weighted_u_value_roof - target_u) * footprint * hdd * 24.0 / 1000.0
                savings_eur = savings_kwh * cost_per_kwh
                cost = footprint * 60.0  # EUR/m2
                payback = cost / savings_eur if savings_eur > 0 else 99.0
                co2_red = savings_kwh * gas_ef
                recommendations.append(ImprovementRecommendation(
                    title="Roof insulation upgrade",
                    description=(
                        f"Increase roof insulation to achieve U={target_u:.2f} W/m2K "
                        f"from current {env.weighted_u_value_roof:.2f} W/m2K."
                    ),
                    category="envelope",
                    severity=FindingSeverity.MEDIUM,
                    annual_savings_kwh=round(savings_kwh, 2),
                    annual_savings_eur=round(savings_eur, 2),
                    implementation_cost_eur=round(cost, 2),
                    simple_payback_years=round(payback, 2),
                    co2_reduction_kg=round(co2_red, 2),
                    priority="high" if payback < 5 else "medium",
                ))

            # Air tightness
            if env.air_permeability_m3_h_m2 > 7.0:
                target_perm = 5.0
                improvement_ratio = (env.air_permeability_m3_h_m2 - target_perm) / env.air_permeability_m3_h_m2
                vent_savings = env.ventilation_heat_loss_w_k * improvement_ratio
                hdd = HDD_BY_CLIMATE_ZONE.get(bld.climate_zone.value, 3200.0)
                savings_kwh = vent_savings * hdd * 24.0 / 1000.0
                savings_eur = savings_kwh * cost_per_kwh
                cost = floor_area * 15.0  # EUR/m2 for draught-proofing
                payback = cost / savings_eur if savings_eur > 0 else 99.0
                co2_red = savings_kwh * gas_ef
                recommendations.append(ImprovementRecommendation(
                    title="Air tightness improvement",
                    description=(
                        f"Reduce air permeability from {env.air_permeability_m3_h_m2:.1f} to "
                        f"{target_perm:.1f} m3/h/m2 via draught-proofing and sealing."
                    ),
                    category="envelope",
                    severity=FindingSeverity.LOW,
                    annual_savings_kwh=round(savings_kwh, 2),
                    annual_savings_eur=round(savings_eur, 2),
                    implementation_cost_eur=round(cost, 2),
                    simple_payback_years=round(payback, 2),
                    co2_reduction_kg=round(co2_red, 2),
                    priority="high" if payback < 3 else "medium",
                ))

        # HVAC recommendations
        if self._systems_result:
            sys_result = self._systems_result
            # Heat pump upgrade
            if sys_result.hvac_efficiency_rating == "poor":
                current_heating = sys_result.heating_demand_kwh
                hp_cop = 3.5
                hp_demand = current_heating / hp_cop
                savings_kwh = current_heating - hp_demand
                savings_eur = savings_kwh * cost_per_kwh
                cost = floor_area * 80.0  # EUR/m2 for heat pump system
                payback = cost / savings_eur if savings_eur > 0 else 99.0
                co2_red = savings_kwh * gas_ef - hp_demand * elec_ef
                recommendations.append(ImprovementRecommendation(
                    title="Heat pump system installation",
                    description=(
                        f"Replace existing heating system (rated {sys_result.hvac_efficiency_rating}) "
                        f"with air-source heat pump (SCOP {hp_cop})."
                    ),
                    category="hvac",
                    severity=FindingSeverity.HIGH,
                    annual_savings_kwh=round(savings_kwh, 2),
                    annual_savings_eur=round(savings_eur, 2),
                    implementation_cost_eur=round(cost, 2),
                    simple_payback_years=round(payback, 2),
                    co2_reduction_kg=round(co2_red, 2),
                    priority="high",
                ))

            # BMS / controls upgrade
            controls_savings_pct = 0.15
            controls_savings = sys_result.total_energy_demand_kwh * controls_savings_pct
            if controls_savings > 0:
                controls_eur = controls_savings * cost_per_kwh
                controls_cost = floor_area * 25.0
                controls_payback = controls_cost / controls_eur if controls_eur > 0 else 99.0
                co2_red = controls_savings * elec_ef
                recommendations.append(ImprovementRecommendation(
                    title="BMS optimisation and controls upgrade",
                    description=(
                        "Install or upgrade building management system with optimised "
                        "start/stop, weather compensation, and zone scheduling."
                    ),
                    category="controls",
                    severity=FindingSeverity.MEDIUM,
                    annual_savings_kwh=round(controls_savings, 2),
                    annual_savings_eur=round(controls_eur, 2),
                    implementation_cost_eur=round(controls_cost, 2),
                    simple_payback_years=round(controls_payback, 2),
                    co2_reduction_kg=round(co2_red, 2),
                    priority="high" if controls_payback < 5 else "medium",
                ))

        # Lighting upgrade
        for ls in bld.lighting_systems:
            if ls.lighting_type.value in ("incandescent", "halogen", "t8_fluorescent", "hid", "metal_halide"):
                current_kwh = ls.installed_power_kw * ls.annual_hours
                led_lpd = LIGHTING_POWER_DENSITY["led"] / 1000.0  # kW/m2
                led_power = led_lpd * ls.area_sqm if ls.area_sqm > 0 else ls.installed_power_kw * 0.4
                led_kwh = led_power * ls.annual_hours
                savings_kwh = max(0, current_kwh - led_kwh)
                savings_eur = savings_kwh * cost_per_kwh
                cost = ls.area_sqm * 45.0 if ls.area_sqm > 0 else savings_kwh * 0.5
                payback = cost / savings_eur if savings_eur > 0 else 99.0
                co2_red = savings_kwh * elec_ef
                recommendations.append(ImprovementRecommendation(
                    title=f"LED lighting upgrade - {ls.zone or 'zone'}",
                    description=(
                        f"Replace {ls.lighting_type.value} lighting with LED and "
                        f"occupancy/daylight controls in {ls.zone or 'zone'}."
                    ),
                    category="lighting",
                    severity=FindingSeverity.MEDIUM,
                    annual_savings_kwh=round(savings_kwh, 2),
                    annual_savings_eur=round(savings_eur, 2),
                    implementation_cost_eur=round(cost, 2),
                    simple_payback_years=round(payback, 2),
                    co2_reduction_kg=round(co2_red, 2),
                    priority="high" if payback < 3 else "medium",
                ))

        # Solar PV recommendation if none exists
        if not bld.renewable_systems:
            footprint = floor_area / max(bld.number_of_floors, 1)
            usable_roof = footprint * 0.5  # 50% of roof usable
            pv_capacity_kwp = usable_roof * 0.15  # 150 Wp/m2
            annual_gen = pv_capacity_kwp * 1000.0  # ~1000 kWh/kWp UK average
            savings_eur = annual_gen * cost_per_kwh
            cost = pv_capacity_kwp * 1200.0  # EUR/kWp
            payback = cost / savings_eur if savings_eur > 0 else 99.0
            co2_red = annual_gen * elec_ef
            recommendations.append(ImprovementRecommendation(
                title="Rooftop solar PV installation",
                description=(
                    f"Install {pv_capacity_kwp:.1f} kWp solar PV system on "
                    f"{usable_roof:.0f} m2 of available roof area."
                ),
                category="renewables",
                severity=FindingSeverity.MEDIUM,
                annual_savings_kwh=round(annual_gen, 2),
                annual_savings_eur=round(savings_eur, 2),
                implementation_cost_eur=round(cost, 2),
                simple_payback_years=round(payback, 2),
                co2_reduction_kg=round(co2_red, 2),
                priority="medium",
            ))

        # Sort by payback
        recommendations.sort(key=lambda r: r.simple_payback_years)
        return recommendations

    def _count_by_category(self) -> Dict[str, int]:
        """Count recommendations by category."""
        counts: Dict[str, int] = {}
        for rec in self._recommendations:
            counts[rec.category] = counts.get(rec.category, 0) + 1
        return counts

    def _estimate_cost_per_kwh(self, bld: BuildingData) -> float:
        """Estimate cost per kWh from utility bills."""
        total_kwh = sum(b.consumption_kwh for b in bld.utility_bills)
        total_cost = sum(b.cost_eur for b in bld.utility_bills)
        return total_cost / total_kwh if total_kwh > 0 else 0.15

    def _compute_primary_energy_per_sqm(
        self, net_demand: float, bld: BuildingData, floor_area: float
    ) -> float:
        """Compute primary energy per m2 using EN 15603 factors."""
        if floor_area <= 0:
            return 0.0
        # Simplified: assume weighted primary energy factor
        # Split energy by source based on utility bills
        elec_share = 0.6
        gas_share = 0.4
        elec_kwh = sum(
            b.consumption_kwh for b in bld.utility_bills if b.energy_source == "electricity"
        )
        gas_kwh = sum(
            b.consumption_kwh for b in bld.utility_bills if b.energy_source in ("natural_gas", "gas")
        )
        total = elec_kwh + gas_kwh
        if total > 0:
            elec_share = elec_kwh / total
            gas_share = gas_kwh / total

        pef_electricity = PRIMARY_ENERGY_FACTORS["electricity"]
        pef_gas = PRIMARY_ENERGY_FACTORS["natural_gas"]
        weighted_pef = elec_share * pef_electricity + gas_share * pef_gas

        primary_energy = net_demand * weighted_pef
        return primary_energy / floor_area

    def _compute_co2_per_sqm(
        self, net_demand: float, bld: BuildingData, floor_area: float
    ) -> float:
        """Compute CO2 emissions per m2."""
        if floor_area <= 0:
            return 0.0
        elec_kwh = sum(
            b.consumption_kwh for b in bld.utility_bills if b.energy_source == "electricity"
        )
        gas_kwh = sum(
            b.consumption_kwh for b in bld.utility_bills if b.energy_source in ("natural_gas", "gas")
        )
        total = elec_kwh + gas_kwh
        elec_share = elec_kwh / total if total > 0 else 0.6
        gas_share = gas_kwh / total if total > 0 else 0.4

        ef_elec = EMISSION_FACTORS["electricity"]
        ef_gas = EMISSION_FACTORS["natural_gas"]
        weighted_ef = elec_share * ef_elec + gas_share * ef_gas

        total_co2 = net_demand * weighted_ef
        return total_co2 / floor_area

    @staticmethod
    def _assign_epc_band(primary_energy_per_sqm: float) -> str:
        """Assign EPC band based on primary energy per m2."""
        if primary_energy_per_sqm <= 25.0:
            return "A+"
        elif primary_energy_per_sqm <= 50.0:
            return "A"
        elif primary_energy_per_sqm <= 75.0:
            return "B"
        elif primary_energy_per_sqm <= 100.0:
            return "C"
        elif primary_energy_per_sqm <= 125.0:
            return "D"
        elif primary_energy_per_sqm <= 150.0:
            return "E"
        elif primary_energy_per_sqm <= 200.0:
            return "F"
        else:
            return "G"

    @staticmethod
    def _compare_benchmark(eui: float, building_type: str) -> str:
        """Compare EUI against CIBSE TM46 benchmarks."""
        benchmarks = EUI_BENCHMARKS.get(building_type, EUI_BENCHMARKS["office"])
        if eui <= benchmarks["best_practice"]:
            return f"Best practice (EUI {eui:.0f} vs benchmark {benchmarks['best_practice']:.0f} kWh/m2)"
        elif eui <= benchmarks["good_practice"]:
            return f"Good practice (EUI {eui:.0f} vs benchmark {benchmarks['good_practice']:.0f} kWh/m2)"
        elif eui <= benchmarks["typical"]:
            return f"Typical (EUI {eui:.0f} vs benchmark {benchmarks['typical']:.0f} kWh/m2)"
        else:
            return f"Below typical (EUI {eui:.0f} vs typical {benchmarks['typical']:.0f} kWh/m2)"

    # -------------------------------------------------------------------------
    # Provenance Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: InitialBuildingAssessmentResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
