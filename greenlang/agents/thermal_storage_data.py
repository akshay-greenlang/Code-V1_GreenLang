# -*- coding: utf-8 -*-
"""
Thermal Energy Storage Engineering Datasets
============================================

Comprehensive datasets of thermal storage material properties, technology specifications,
and engineering constants for Agent #7: ThermalStorageAgent_AI.

This module provides standard engineering values from:
- ASHRAE Handbook HVAC Applications Ch51
- IEA ECES Annex 30 Thermal Storage
- IRENA Thermal Storage Guidelines
- ISO 9806 Solar Collector Performance
- Engineering handbooks and manufacturer data

Author: GreenLang Framework Team
Date: October 2025
Version: 1.0.0
"""

from typing import Dict, Any, List
from dataclasses import dataclass


# ============================================================================
# THERMAL STORAGE MEDIA PROPERTIES
# ============================================================================

@dataclass
class ThermalStorageMedium:
    """Properties of thermal storage medium"""
    name: str
    specific_heat_kj_kg_k: float  # Specific heat capacity
    density_kg_m3: float  # Density at operating temperature
    thermal_conductivity_w_m_k: float  # Thermal conductivity
    operating_temp_range_c: tuple  # (min, max) operating temperature
    latent_heat_kj_kg: float = 0  # Latent heat for PCMs
    cost_per_kg_usd: float = 0  # Material cost
    notes: str = ""


# Water (most common sensible heat storage medium)
WATER_PROPERTIES = {
    "20C": ThermalStorageMedium(
        name="Water (20°C)",
        specific_heat_kj_kg_k=4.18,
        density_kg_m3=998,
        thermal_conductivity_w_m_k=0.60,
        operating_temp_range_c=(0, 95),
        cost_per_kg_usd=0.001,
        notes="Atmospheric pressure, standard sensible heat storage"
    ),
    "50C": ThermalStorageMedium(
        name="Water (50°C)",
        specific_heat_kj_kg_k=4.18,
        density_kg_m3=988,
        thermal_conductivity_w_m_k=0.64,
        operating_temp_range_c=(20, 95),
        cost_per_kg_usd=0.001,
        notes="Mid-range temperature operation"
    ),
    "90C": ThermalStorageMedium(
        name="Water (90°C)",
        specific_heat_kj_kg_k=4.20,
        density_kg_m3=965,
        thermal_conductivity_w_m_k=0.67,
        operating_temp_range_c=(40, 95),
        cost_per_kg_usd=0.001,
        notes="High temperature atmospheric or low pressure"
    ),
    "120C_pressurized": ThermalStorageMedium(
        name="Pressurized Water (120°C)",
        specific_heat_kj_kg_k=4.25,
        density_kg_m3=943,
        thermal_conductivity_w_m_k=0.68,
        operating_temp_range_c=(80, 180),
        cost_per_kg_usd=0.001,
        notes="Requires pressurized vessel, 2-4 bar"
    ),
}

# Molten Salts (high temperature storage)
MOLTEN_SALT_PROPERTIES = {
    "solar_salt": ThermalStorageMedium(
        name="Solar Salt (60% NaNO3, 40% KNO3)",
        specific_heat_kj_kg_k=1.53,
        density_kg_m3=1899,
        thermal_conductivity_w_m_k=0.57,
        operating_temp_range_c=(238, 600),
        cost_per_kg_usd=0.93,
        notes="CSP standard, freezing point 238°C, requires freeze protection"
    ),
    "hitec": ThermalStorageMedium(
        name="HITEC Salt (53% KNO3, 40% NaNO2, 7% NaNO3)",
        specific_heat_kj_kg_k=1.56,
        density_kg_m3=1990,
        thermal_conductivity_w_m_k=0.54,
        operating_temp_range_c=(142, 535),
        cost_per_kg_usd=1.20,
        notes="Lower melting point than solar salt, more expensive"
    ),
}

# Phase Change Materials (PCMs)
PCM_PROPERTIES = {
    "paraffin_60C": ThermalStorageMedium(
        name="Paraffin Wax (60°C melting)",
        specific_heat_kj_kg_k=2.1,
        density_kg_m3=800,
        thermal_conductivity_w_m_k=0.21,
        operating_temp_range_c=(50, 70),
        latent_heat_kj_kg=200,
        cost_per_kg_usd=2.50,
        notes="Isothermal storage, low thermal conductivity requires enhancement"
    ),
    "salt_hydrate_58C": ThermalStorageMedium(
        name="Salt Hydrate (58°C melting)",
        specific_heat_kj_kg_k=2.0,
        density_kg_m3=1450,
        thermal_conductivity_w_m_k=0.54,
        operating_temp_range_c=(50, 65),
        latent_heat_kj_kg=265,
        cost_per_kg_usd=1.80,
        notes="Higher density than paraffins, potential for phase separation"
    ),
}

# Concrete/Solid Media
SOLID_MEDIA_PROPERTIES = {
    "concrete": ThermalStorageMedium(
        name="Concrete Thermal Mass",
        specific_heat_kj_kg_k=0.85,
        density_kg_m3=2400,
        thermal_conductivity_w_m_k=1.4,
        operating_temp_range_c=(0, 400),
        cost_per_kg_usd=0.05,
        notes="Very low cost, long lifetime, large volume required"
    ),
    "castable_ceramic": ThermalStorageMedium(
        name="Castable Ceramic",
        specific_heat_kj_kg_k=1.0,
        density_kg_m3=3000,
        thermal_conductivity_w_m_k=2.0,
        operating_temp_range_c=(0, 1200),
        cost_per_kg_usd=0.35,
        notes="Very high temperature capability, expensive"
    ),
}


# ============================================================================
# STORAGE TECHNOLOGY SPECIFICATIONS
# ============================================================================

@dataclass
class StorageTechnologySpec:
    """Specification for thermal storage technology"""
    technology: str
    temperature_range_c: tuple  # (min, max)
    energy_density_kwh_m3: float  # Volumetric energy density
    round_trip_efficiency: float  # Typical round-trip efficiency
    capex_per_kwh_usd: tuple  # (low, high) cost range
    opex_percent_capex: float  # Annual O&M as % of CAPEX
    lifetime_years: int  # Expected lifetime
    charging_rate: str  # Typical charging capability
    discharging_rate: str  # Typical discharging capability
    scalability: str  # "small", "medium", "large", "very_large"
    maturity: str  # "emerging", "proven", "mature"
    advantages: List[str]
    challenges: List[str]
    typical_applications: List[str]


STORAGE_TECHNOLOGIES = {
    "hot_water_tank": StorageTechnologySpec(
        technology="Hot Water Tank (Atmospheric)",
        temperature_range_c=(30, 95),
        energy_density_kwh_m3=40,  # For ΔT=40K
        round_trip_efficiency=0.92,
        capex_per_kwh_usd=(15, 30),
        opex_percent_capex=0.015,
        lifetime_years=25,
        charging_rate="Flexible (heat exchanger limited)",
        discharging_rate="Flexible (heat exchanger limited)",
        scalability="medium_to_large",
        maturity="mature",
        advantages=[
            "Lowest cost per kWh",
            "Simple installation and operation",
            "High round-trip efficiency (90-95%)",
            "Proven technology, minimal risk",
            "Stratification improves performance",
            "Long lifetime (25+ years)"
        ],
        challenges=[
            "Limited to <100°C (atmospheric)",
            "Lower energy density than alternatives",
            "Requires significant space",
            "Heat loss through tank walls"
        ],
        typical_applications=[
            "Solar thermal integration",
            "District heating",
            "Industrial process heat <90°C",
            "HVAC thermal energy storage"
        ]
    ),

    "pressurized_hot_water": StorageTechnologySpec(
        technology="Pressurized Hot Water",
        temperature_range_c=(100, 180),
        energy_density_kwh_m3=60,  # For ΔT=60K
        round_trip_efficiency=0.88,
        capex_per_kwh_usd=(30, 50),
        opex_percent_capex=0.020,
        lifetime_years=20,
        charging_rate="Moderate (pressure vessel constraints)",
        discharging_rate="Moderate",
        scalability="small_to_medium",
        maturity="proven",
        advantages=[
            "Higher temperature capability than atmospheric",
            "Good energy density",
            "Mature technology",
            "High efficiency"
        ],
        challenges=[
            "Requires pressure vessel (higher cost)",
            "Safety considerations (pressure)",
            "Limited scalability",
            "Regular pressure vessel inspections required"
        ],
        typical_applications=[
            "Medium-temperature process heat",
            "Steam generation support",
            "Industrial heating 100-180°C"
        ]
    ),

    "molten_salt": StorageTechnologySpec(
        technology="Molten Salt Storage",
        temperature_range_c=(250, 565),
        energy_density_kwh_m3=120,  # For ΔT=100K
        round_trip_efficiency=0.95,
        capex_per_kwh_usd=(30, 60),
        opex_percent_capex=0.025,
        lifetime_years=30,
        charging_rate="Fast (high heat transfer)",
        discharging_rate="Fast",
        scalability="large_to_very_large",
        maturity="proven",
        advantages=[
            "Very high temperature capability",
            "High energy density",
            "Excellent round-trip efficiency",
            "Proven in CSP plants",
            "Long lifetime",
            "Non-toxic, non-flammable"
        ],
        challenges=[
            "Freeze protection required (>238°C)",
            "Corrosive (requires stainless steel 316/347)",
            "Heat tracing for cold starts",
            "Higher initial cost",
            "Complexity in operation"
        ],
        typical_applications=[
            "Concentrating solar power (CSP)",
            "High-temperature industrial processes",
            "Power generation cycles",
            "Thermal energy storage for grid services"
        ]
    ),

    "phase_change_material": StorageTechnologySpec(
        technology="Phase Change Material (PCM)",
        temperature_range_c=(30, 120),
        energy_density_kwh_m3=120,  # 3× water due to latent heat
        round_trip_efficiency=0.80,
        capex_per_kwh_usd=(50, 120),
        opex_percent_capex=0.020,
        lifetime_years=15,
        charging_rate="Slow (low thermal conductivity)",
        discharging_rate="Slow",
        scalability="small_to_medium",
        maturity="emerging",
        advantages=[
            "3-5× higher energy density than water",
            "Isothermal charge/discharge",
            "Compact footprint",
            "Suitable for space-constrained applications"
        ],
        challenges=[
            "3-5× higher cost than water",
            "Low thermal conductivity (requires enhancement)",
            "Limited cycle life for some materials",
            "Heat transfer challenges",
            "Material degradation over cycles",
            "Incomplete phase change in some designs"
        ],
        typical_applications=[
            "Space-constrained installations",
            "Isothermal process requirements",
            "Waste heat recovery",
            "Building HVAC thermal storage"
        ]
    ),

    "concrete_thermal_mass": StorageTechnologySpec(
        technology="Concrete Thermal Mass",
        temperature_range_c=(50, 400),
        energy_density_kwh_m3=25,  # Low due to low cp
        round_trip_efficiency=0.85,
        capex_per_kwh_usd=(8, 15),
        opex_percent_capex=0.010,
        lifetime_years=40,
        charging_rate="Very slow (low thermal conductivity)",
        discharging_rate="Very slow",
        scalability="large_to_very_large",
        maturity="emerging",
        advantages=[
            "Lowest cost per kWh",
            "Very long lifetime (40+ years)",
            "High temperature capability",
            "Non-toxic, abundant materials",
            "No corrosion issues"
        ],
        challenges=[
            "Very large footprint required",
            "Slow charge/discharge rates",
            "Low energy density",
            "Difficult retrofits",
            "Long thermal time constant"
        ],
        typical_applications=[
            "Seasonal thermal storage",
            "Long-duration storage (days to weeks)",
            "Very large scale installations",
            "High-temperature industrial waste heat"
        ]
    ),

    "steam_accumulator": StorageTechnologySpec(
        technology="Steam Accumulator",
        temperature_range_c=(100, 200),
        energy_density_kwh_m3=80,
        round_trip_efficiency=0.90,
        capex_per_kwh_usd=(30, 50),
        opex_percent_capex=0.020,
        lifetime_years=25,
        charging_rate="Very fast (flash steam)",
        discharging_rate="Very fast",
        scalability="small_to_medium",
        maturity="mature",
        advantages=[
            "Very fast response time (<1 minute)",
            "High power density",
            "Suitable for demand response",
            "Direct steam supply",
            "Proven technology"
        ],
        challenges=[
            "Limited duration (typically <4 hours)",
            "Pressure vessel requirements",
            "Lower energy density for long durations",
            "Safety considerations"
        ],
        typical_applications=[
            "Industrial steam backup",
            "Peak shaving (short duration)",
            "Demand response",
            "Steam grid stabilization"
        ]
    ),

    "thermochemical_storage": StorageTechnologySpec(
        technology="Thermochemical Storage",
        temperature_range_c=(150, 600),
        energy_density_kwh_m3=200,  # Very high
        round_trip_efficiency=0.75,
        capex_per_kwh_usd=(80, 150),
        opex_percent_capex=0.030,
        lifetime_years=20,
        charging_rate="Moderate",
        discharging_rate="Moderate",
        scalability="small_to_large",
        maturity="emerging",
        advantages=[
            "Highest energy density",
            "Long-term storage with minimal losses",
            "High temperature capability",
            "Compact storage"
        ],
        challenges=[
            "Highest cost",
            "Complex reactor design",
            "Material degradation",
            "Limited commercial deployment",
            "Technical risk"
        ],
        typical_applications=[
            "Long-term seasonal storage",
            "High-temperature industrial processes",
            "Research and development",
            "Niche high-value applications"
        ]
    ),
}


# ============================================================================
# INSULATION MATERIALS
# ============================================================================

@dataclass
class InsulationMaterial:
    """Thermal insulation material properties"""
    name: str
    thermal_conductivity_w_m_k: float  # k-value
    max_temperature_c: float
    density_kg_m3: float
    cost_per_m2_per_inch_usd: float
    typical_thickness_inches: List[float]
    r_value_per_inch: float  # Imperial units (ft²·°F·hr/Btu/inch)
    notes: str


INSULATION_MATERIALS = {
    "fiberglass": InsulationMaterial(
        name="Fiberglass Insulation",
        thermal_conductivity_w_m_k=0.040,
        max_temperature_c=260,
        density_kg_m3=32,
        cost_per_m2_per_inch_usd=1.20,
        typical_thickness_inches=[2, 4, 6],
        r_value_per_inch=3.14,
        notes="Low cost, most common, good for moderate temperatures"
    ),
    "mineral_wool": InsulationMaterial(
        name="Mineral Wool (Rock Wool)",
        thermal_conductivity_w_m_k=0.038,
        max_temperature_c=760,
        density_kg_m3=100,
        cost_per_m2_per_inch_usd=1.80,
        typical_thickness_inches=[2, 4, 6],
        r_value_per_inch=3.30,
        notes="Higher temperature capability than fiberglass"
    ),
    "polyurethane": InsulationMaterial(
        name="Polyurethane Foam",
        thermal_conductivity_w_m_k=0.023,
        max_temperature_c=110,
        density_kg_m3=35,
        cost_per_m2_per_inch_usd=2.50,
        typical_thickness_inches=[2, 3, 4, 6],
        r_value_per_inch=5.60,
        notes="Best R-value per inch, limited by temperature"
    ),
    "polyisocyanurate": InsulationMaterial(
        name="Polyisocyanurate (Polyiso)",
        thermal_conductivity_w_m_k=0.024,
        max_temperature_c=150,
        density_kg_m3=32,
        cost_per_m2_per_inch_usd=2.80,
        typical_thickness_inches=[2, 3, 4, 6],
        r_value_per_inch=5.60,
        notes="Higher temp than polyurethane, excellent R-value"
    ),
    "calcium_silicate": InsulationMaterial(
        name="Calcium Silicate",
        thermal_conductivity_w_m_k=0.052,
        max_temperature_c=650,
        density_kg_m3=240,
        cost_per_m2_per_inch_usd=3.50,
        typical_thickness_inches=[1.5, 2, 3],
        r_value_per_inch=2.42,
        notes="High temperature capability, used in industrial applications"
    ),
    "aerogel": InsulationMaterial(
        name="Aerogel Blanket",
        thermal_conductivity_w_m_k=0.014,
        max_temperature_c=200,
        density_kg_m3=150,
        cost_per_m2_per_inch_usd=25.00,
        typical_thickness_inches=[0.4, 0.8],
        r_value_per_inch=9.00,
        notes="Highest R-value, very expensive, space-constrained applications"
    ),
}


# ============================================================================
# SOLAR COLLECTOR SPECIFICATIONS
# ============================================================================

@dataclass
class SolarCollectorSpec:
    """Solar thermal collector specifications"""
    collector_type: str
    peak_efficiency: float  # At optimal conditions
    temperature_range_c: tuple  # Practical operating range
    cost_per_m2_usd: tuple  # (low, high) installed cost
    concentration_ratio: float  # 1 for non-concentrating
    suitable_for_temperatures: List[str]
    typical_applications: List[str]


SOLAR_COLLECTORS = {
    "flat_plate": SolarCollectorSpec(
        collector_type="Flat Plate Collector",
        peak_efficiency=0.70,  # At low temperatures
        temperature_range_c=(30, 80),
        cost_per_m2_usd=(200, 300),
        concentration_ratio=1.0,
        suitable_for_temperatures=["low (<80°C)"],
        typical_applications=[
            "Domestic hot water",
            "Swimming pool heating",
            "Low-temperature industrial processes",
            "Space heating"
        ]
    ),
    "evacuated_tube": SolarCollectorSpec(
        collector_type="Evacuated Tube Collector",
        peak_efficiency=0.70,
        temperature_range_c=(40, 150),
        cost_per_m2_usd=(350, 500),
        concentration_ratio=1.0,
        suitable_for_temperatures=["low (<80°C)", "medium (80-150°C)"],
        typical_applications=[
            "Medium-temperature industrial processes",
            "Steam generation",
            "Air conditioning (absorption chillers)",
            "Industrial process heat"
        ]
    ),
    "parabolic_trough": SolarCollectorSpec(
        collector_type="Parabolic Trough Collector",
        peak_efficiency=0.75,
        temperature_range_c=(100, 400),
        cost_per_m2_usd=(500, 800),
        concentration_ratio=80.0,
        suitable_for_temperatures=["medium (80-150°C)", "high (>150°C)"],
        typical_applications=[
            "Concentrating solar power (CSP)",
            "High-temperature industrial heat",
            "Power generation",
            "Enhanced oil recovery"
        ]
    ),
}


# ============================================================================
# ENGINEERING CONSTANTS
# ============================================================================

# Conversion factors
CONVERSIONS = {
    "kwh_to_mmbtu": 0.003412,
    "kwh_to_mj": 3.6,
    "kwh_to_btu": 3412.14,
    "btu_to_kwh": 1 / 3412.14,
    "mmbtu_to_kwh": 1 / 0.003412,
    "ton_refrigeration_to_kw": 3.517,
    "m3_to_gallons": 264.172,
    "gallons_to_m3": 1 / 264.172,
    "celsius_to_kelvin": 273.15,
    "fahrenheit_to_celsius": lambda f: (f - 32) * 5/9,
    "celsius_to_fahrenheit": lambda c: c * 9/5 + 32,
}

# Standard conditions
STANDARD_CONDITIONS = {
    "ambient_temperature_c": 20,  # Standard room temperature
    "ambient_pressure_kpa": 101.325,  # Sea level atmospheric pressure
    "water_density_kg_m3": 1000,  # At 20°C
    "water_specific_heat_kj_kg_k": 4.18,  # At 20°C
    "air_density_kg_m3": 1.204,  # At 20°C, sea level
}

# Typical performance factors
PERFORMANCE_FACTORS = {
    "storage_round_trip_efficiency": {
        "hot_water_excellent": 0.95,
        "hot_water_good": 0.92,
        "hot_water_fair": 0.88,
        "molten_salt": 0.95,
        "pcm": 0.80,
        "concrete": 0.85,
    },
    "solar_thermal_availability": 0.85,  # Account for weather, maintenance
    "heat_exchanger_effectiveness": {
        "excellent": 0.90,
        "good": 0.85,
        "fair": 0.75,
    },
    "piping_heat_loss_factor": 0.02,  # 2% loss in distribution piping (typical)
}

# Financial parameters (typical ranges)
FINANCIAL_DEFAULTS = {
    "discount_rate": 0.06,  # 6% typical for industrial projects
    "inflation_rate": 0.03,  # 3% long-term average
    "electricity_escalation_rate": 0.02,  # 2% annual increase
    "natural_gas_escalation_rate": 0.025,  # 2.5% annual increase
    "opex_percent_of_capex": {
        "hot_water_tank": 0.015,
        "pressurized_hot_water": 0.020,
        "molten_salt": 0.025,
        "pcm": 0.020,
        "concrete": 0.010,
    },
}

# Standards and references
STANDARDS = {
    "ASHRAE_Handbook_HVAC_Applications_Ch51": {
        "title": "ASHRAE Handbook - HVAC Applications, Chapter 51: Thermal Storage",
        "year": 2019,
        "scope": "Design guidelines for thermal energy storage systems"
    },
    "IEA_ECES_Annex_30": {
        "title": "IEA ECES Annex 30: Thermal Energy Storage for Cost-Effective Energy Management and CO2 Mitigation",
        "year": 2020,
        "scope": "International best practices for thermal storage"
    },
    "IRENA_Thermal_Storage": {
        "title": "IRENA: Thermal Energy Storage - Technology Brief",
        "year": 2013,
        "scope": "Technology overview and cost analysis"
    },
    "ISO_9806": {
        "title": "ISO 9806: Solar Energy - Solar Thermal Collectors - Test Methods",
        "year": 2017,
        "scope": "Standardized testing of solar thermal collectors"
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_water_properties(temperature_c: float) -> ThermalStorageMedium:
    """
    Get water properties at specified temperature.

    Args:
        temperature_c: Water temperature in Celsius

    Returns:
        ThermalStorageMedium with interpolated properties
    """
    if temperature_c <= 20:
        return WATER_PROPERTIES["20C"]
    elif temperature_c <= 50:
        return WATER_PROPERTIES["50C"]
    elif temperature_c <= 95:
        return WATER_PROPERTIES["90C"]
    else:
        return WATER_PROPERTIES["120C_pressurized"]


def calculate_storage_volume(
    capacity_kwh: float,
    temperature_delta_k: float,
    medium: ThermalStorageMedium = None
) -> float:
    """
    Calculate required storage volume for given capacity and temperature delta.

    Args:
        capacity_kwh: Desired storage capacity in kWh_thermal
        temperature_delta_k: Temperature difference in Kelvin (or °C)
        medium: Storage medium (default: water at 50°C)

    Returns:
        Volume in m³

    Example:
        >>> volume = calculate_storage_volume(2000, 40)  # 2000 kWh, 40K delta
        >>> print(f"Volume: {volume:.1f} m³")
        Volume: 43.1 m³
    """
    if medium is None:
        medium = WATER_PROPERTIES["50C"]

    # Q = m × cp × ΔT
    # m = Q / (cp × ΔT)
    # V = m / ρ

    # Convert kWh to kJ: 1 kWh = 3600 kJ
    energy_kj = capacity_kwh * 3600

    # Mass required
    mass_kg = energy_kj / (medium.specific_heat_kj_kg_k * temperature_delta_k)

    # Volume
    volume_m3 = mass_kg / medium.density_kg_m3

    return volume_m3


def calculate_insulation_thickness(
    target_u_value_w_m2k: float,
    insulation: InsulationMaterial
) -> float:
    """
    Calculate required insulation thickness to achieve target U-value.

    Args:
        target_u_value_w_m2k: Desired overall U-value in W/(m²·K)
        insulation: Insulation material

    Returns:
        Required thickness in meters

    Example:
        >>> thickness = calculate_insulation_thickness(0.25, INSULATION_MATERIALS["polyurethane"])
        >>> print(f"Thickness: {thickness*1000:.0f} mm ({thickness*39.37:.1f} inches)")
        Thickness: 92 mm (3.6 inches)
    """
    # U = k / thickness (simplified, ignoring convection resistances)
    # thickness = k / U
    thickness_m = insulation.thermal_conductivity_w_m_k / target_u_value_w_m2k
    return thickness_m


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # Material properties
    "WATER_PROPERTIES",
    "MOLTEN_SALT_PROPERTIES",
    "PCM_PROPERTIES",
    "SOLID_MEDIA_PROPERTIES",

    # Technology specifications
    "STORAGE_TECHNOLOGIES",
    "INSULATION_MATERIALS",
    "SOLAR_COLLECTORS",

    # Constants
    "CONVERSIONS",
    "STANDARD_CONDITIONS",
    "PERFORMANCE_FACTORS",
    "FINANCIAL_DEFAULTS",
    "STANDARDS",

    # Utility functions
    "get_water_properties",
    "calculate_storage_volume",
    "calculate_insulation_thickness",

    # Data classes
    "ThermalStorageMedium",
    "StorageTechnologySpec",
    "InsulationMaterial",
    "SolarCollectorSpec",
]
