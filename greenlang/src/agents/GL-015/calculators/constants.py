"""
GL-015 INSULSCAN Constants and Lookup Tables

This module provides comprehensive lookup tables for industrial insulation
thermal analysis, heat loss calculations, and energy efficiency assessments.
All values are sourced from industry standards (ASTM, ISO, ASHRAE) and
manufacturer specifications.

Key Features:
- Temperature-dependent thermal conductivity for 20+ insulation types
- Surface emissivity tables for various cladding materials
- Convection correlation coefficients (Churchill-Chu, Morgan, McAdams)
- Economic parameters for ROI calculations
- Safety limits per ASTM C1055 and OSHA standards

Usage:
    >>> from constants import InsulationMaterials, SurfaceEmissivity
    >>> k = InsulationMaterials.get_thermal_conductivity("mineral_wool_rock", 150.0)
    >>> emissivity = SurfaceEmissivity.get_emissivity("aluminum_polished")

References:
    - ASTM C680-14: Standard Practice for Estimate of the Heat Gain or Loss
    - ASTM C1055-03: Standard Guide for Heated System Surface Conditions
    - ASHRAE Handbook - Fundamentals (2021)
    - ISO 12241:2008: Thermal insulation for building equipment

Author: GreenLang Engineering Team
Version: 1.0.0
License: Apache 2.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
from functools import lru_cache

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """
    Fundamental physical constants for heat transfer calculations.

    All values are in SI units unless otherwise specified.
    These constants are immutable and represent fundamental physics.

    Attributes:
        STEFAN_BOLTZMANN: Stefan-Boltzmann constant [W/m^2*K^4]
        GRAVITATIONAL_ACCELERATION: Standard gravity [m/s^2]
        ABSOLUTE_ZERO_CELSIUS: Absolute zero in Celsius [C]
        ATMOSPHERIC_PRESSURE: Standard atmospheric pressure [Pa]
    """

    # Stefan-Boltzmann constant for radiation heat transfer
    # Reference: NIST CODATA 2018
    STEFAN_BOLTZMANN: float = 5.670374419e-8  # W/(m^2*K^4)

    # Gravitational acceleration (standard)
    GRAVITATIONAL_ACCELERATION: float = 9.80665  # m/s^2

    # Temperature references
    ABSOLUTE_ZERO_CELSIUS: float = -273.15  # C
    ABSOLUTE_ZERO_FAHRENHEIT: float = -459.67  # F

    # Pressure references
    ATMOSPHERIC_PRESSURE: float = 101325.0  # Pa
    ATMOSPHERIC_PRESSURE_PSI: float = 14.696  # psi

    # Gas constant
    UNIVERSAL_GAS_CONSTANT: float = 8.314462618  # J/(mol*K)

    # Pi value for calculations
    PI: float = 3.141592653589793

    # Standard temperature for reference conditions
    STANDARD_TEMPERATURE_K: float = 298.15  # K (25 C)
    STANDARD_TEMPERATURE_C: float = 25.0  # C
    STANDARD_TEMPERATURE_F: float = 77.0  # F


# =============================================================================
# AIR PROPERTIES (TEMPERATURE DEPENDENT)
# =============================================================================

@dataclass
class AirPropertiesTable:
    """
    Temperature-dependent properties of air at atmospheric pressure.

    Properties are tabulated at various temperatures for interpolation.
    All values are in SI units.

    Data Source: ASHRAE Handbook - Fundamentals (2021), Chapter 33
    """

    # Temperature points for interpolation [K]
    TEMPERATURES_K: Tuple[float, ...] = (
        200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0,
        600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0,
        1000.0, 1050.0, 1100.0, 1150.0, 1200.0
    )

    # Density [kg/m^3]
    DENSITY: Tuple[float, ...] = (
        1.7684, 1.4128, 1.1774, 1.0085, 0.8826, 0.7842, 0.7056, 0.6414,
        0.5879, 0.5426, 0.5037, 0.4698, 0.4405, 0.4149, 0.3925, 0.3727,
        0.3550, 0.3391, 0.3247, 0.3117, 0.2999
    )

    # Specific heat at constant pressure [J/(kg*K)]
    SPECIFIC_HEAT: Tuple[float, ...] = (
        1007.0, 1006.0, 1007.0, 1009.0, 1014.0, 1021.0, 1030.0, 1040.0,
        1051.0, 1063.0, 1075.0, 1087.0, 1099.0, 1110.0, 1121.0, 1131.0,
        1141.0, 1150.0, 1159.0, 1167.0, 1175.0
    )

    # Thermal conductivity [W/(m*K)]
    THERMAL_CONDUCTIVITY: Tuple[float, ...] = (
        0.01809, 0.02227, 0.02624, 0.03003, 0.03365, 0.03710, 0.04041,
        0.04357, 0.04661, 0.04954, 0.05236, 0.05509, 0.05774, 0.06030,
        0.06279, 0.06522, 0.06759, 0.06990, 0.07216, 0.07437, 0.07653
    )

    # Dynamic viscosity [Pa*s] (multiply by 1e-5)
    DYNAMIC_VISCOSITY_E5: Tuple[float, ...] = (
        1.329, 1.599, 1.846, 2.075, 2.286, 2.484, 2.671, 2.848,
        3.018, 3.181, 3.338, 3.491, 3.640, 3.785, 3.926, 4.064,
        4.198, 4.330, 4.459, 4.585, 4.709
    )

    # Kinematic viscosity [m^2/s] (multiply by 1e-5)
    KINEMATIC_VISCOSITY_E5: Tuple[float, ...] = (
        0.7516, 1.132, 1.568, 2.057, 2.591, 3.168, 3.786, 4.441,
        5.133, 5.861, 6.624, 7.422, 8.254, 9.121, 10.02, 10.95,
        11.91, 12.91, 13.93, 14.99, 16.08
    )

    # Prandtl number [dimensionless]
    PRANDTL: Tuple[float, ...] = (
        0.739, 0.722, 0.708, 0.697, 0.689, 0.683, 0.680, 0.680,
        0.680, 0.682, 0.684, 0.687, 0.690, 0.693, 0.696, 0.699,
        0.702, 0.704, 0.707, 0.709, 0.711
    )

    @classmethod
    def get_property(cls, property_name: str, temperature_k: float) -> float:
        """
        Get air property at a given temperature using linear interpolation.

        Args:
            property_name: Name of property (density, specific_heat, etc.)
            temperature_k: Temperature in Kelvin

        Returns:
            Interpolated property value

        Raises:
            ValueError: If temperature is out of range or property unknown
        """
        property_map = {
            'density': cls.DENSITY,
            'specific_heat': cls.SPECIFIC_HEAT,
            'thermal_conductivity': cls.THERMAL_CONDUCTIVITY,
            'dynamic_viscosity': tuple(v * 1e-5 for v in cls.DYNAMIC_VISCOSITY_E5),
            'kinematic_viscosity': tuple(v * 1e-5 for v in cls.KINEMATIC_VISCOSITY_E5),
            'prandtl': cls.PRANDTL
        }

        if property_name not in property_map:
            raise ValueError(f"Unknown property: {property_name}")

        values = property_map[property_name]
        temps = cls.TEMPERATURES_K

        # Clamp temperature to valid range
        if temperature_k < temps[0]:
            return values[0]
        if temperature_k > temps[-1]:
            return values[-1]

        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= temperature_k <= temps[i + 1]:
                t_ratio = (temperature_k - temps[i]) / (temps[i + 1] - temps[i])
                return values[i] + t_ratio * (values[i + 1] - values[i])

        return values[-1]


# =============================================================================
# INSULATION MATERIAL TYPES
# =============================================================================

class InsulationType(Enum):
    """
    Enumeration of industrial insulation material types.

    Each type has specific temperature limits, applications, and
    thermal properties. The naming convention follows industry standards.
    """

    # Mineral Fiber Insulations
    MINERAL_WOOL_ROCK = "mineral_wool_rock"
    MINERAL_WOOL_GLASS = "mineral_wool_glass"
    MINERAL_WOOL_SLAG = "mineral_wool_slag"

    # Calcium Silicate Insulations
    CALCIUM_SILICATE = "calcium_silicate"
    CALCIUM_SILICATE_HIGH_TEMP = "calcium_silicate_high_temp"

    # Cellular Glass Insulations
    CELLULAR_GLASS = "cellular_glass"
    CELLULAR_GLASS_HIGH_DENSITY = "cellular_glass_high_density"

    # Foam Plastic Insulations
    POLYURETHANE_FOAM = "polyurethane_foam"
    POLYURETHANE_SPRAY = "polyurethane_spray"
    POLYISOCYANURATE = "polyisocyanurate"
    POLYSTYRENE_EPS = "polystyrene_eps"
    POLYSTYRENE_XPS = "polystyrene_xps"
    PHENOLIC_FOAM = "phenolic_foam"

    # High-Performance Insulations
    AEROGEL_BLANKET = "aerogel_blanket"
    AEROGEL_COMPOSITE = "aerogel_composite"
    MICROPOROUS = "microporous"

    # Granular Insulations
    PERLITE_EXPANDED = "perlite_expanded"
    PERLITE_POWDER = "perlite_powder"
    VERMICULITE = "vermiculite"
    DIATOMACEOUS_EARTH = "diatomaceous_earth"

    # Refractory Insulations
    CERAMIC_FIBER = "ceramic_fiber"
    CERAMIC_FIBER_BOARD = "ceramic_fiber_board"
    REFRACTORY_BRICK = "refractory_brick"
    FIREBRICK_INSULATING = "firebrick_insulating"

    # Elastomeric Insulations
    ELASTOMERIC_FOAM = "elastomeric_foam"
    ELASTOMERIC_SHEET = "elastomeric_sheet"

    # Specialty Insulations
    MELAMINE_FOAM = "melamine_foam"
    CRYOGENIC_PERLITE = "cryogenic_perlite"
    CRYOGENIC_MLI = "cryogenic_mli"


# =============================================================================
# INSULATION MATERIAL PROPERTIES
# =============================================================================

@dataclass
class InsulationMaterialSpec:
    """
    Complete specification for an insulation material.

    Contains thermal, physical, and operational properties needed
    for heat loss calculations and material selection.

    Attributes:
        name: Common name of the material
        material_type: Enumeration type
        k_coefficients: Polynomial coefficients for k(T) calculation
        min_temp_c: Minimum operating temperature (Celsius)
        max_temp_c: Maximum operating temperature (Celsius)
        density_kg_m3: Nominal density range (min, max)
        specific_heat_j_kgk: Specific heat capacity
        fire_rating: ASTM E84 flame spread rating
        moisture_absorption: Moisture absorption percentage
        compressive_strength_kpa: Compressive strength
        astm_standard: Applicable ASTM standard
        applications: List of typical applications
    """

    name: str
    material_type: InsulationType
    k_coefficients: Tuple[float, ...]  # a0 + a1*T + a2*T^2 + a3*T^3
    min_temp_c: float
    max_temp_c: float
    density_kg_m3: Tuple[float, float]  # (min, max)
    specific_heat_j_kgk: float
    fire_rating: str
    moisture_absorption: float  # percentage by weight
    compressive_strength_kpa: Optional[float]
    astm_standard: str
    applications: List[str] = field(default_factory=list)

    def thermal_conductivity(self, temperature_c: float) -> float:
        """
        Calculate thermal conductivity at given temperature.

        Uses polynomial fit: k(T) = a0 + a1*T + a2*T^2 + a3*T^3
        where T is in Celsius and k is in W/(m*K).

        Args:
            temperature_c: Mean temperature in Celsius

        Returns:
            Thermal conductivity in W/(m*K)

        Raises:
            ValueError: If temperature is outside operating range
        """
        if temperature_c < self.min_temp_c or temperature_c > self.max_temp_c:
            raise ValueError(
                f"Temperature {temperature_c}C outside range "
                f"[{self.min_temp_c}, {self.max_temp_c}]C for {self.name}"
            )

        k = 0.0
        for i, coeff in enumerate(self.k_coefficients):
            k += coeff * (temperature_c ** i)

        return max(k, 0.001)  # Ensure positive value


# =============================================================================
# INSULATION MATERIALS DATABASE
# =============================================================================

class InsulationMaterials:
    """
    Comprehensive database of industrial insulation materials.

    Contains 25+ insulation types with temperature-dependent thermal
    conductivity values and physical properties. All values are from
    manufacturer specifications and ASTM standards.

    Usage:
        >>> k = InsulationMaterials.get_thermal_conductivity("mineral_wool_rock", 150.0)
        >>> spec = InsulationMaterials.get_material_spec("calcium_silicate")
    """

    # =========================================================================
    # MINERAL WOOL INSULATIONS
    # =========================================================================

    MINERAL_WOOL_ROCK = InsulationMaterialSpec(
        name="Rock Wool (Mineral Wool)",
        material_type=InsulationType.MINERAL_WOOL_ROCK,
        # k(T) = 0.0340 + 0.000080*T + 0.00000020*T^2 [W/mK, T in C]
        # Valid 0-650C, Manufacturer: Rockwool, Owens Corning
        k_coefficients=(0.0340, 8.0e-5, 2.0e-7, 0.0),
        min_temp_c=-40.0,
        max_temp_c=650.0,
        density_kg_m3=(40.0, 200.0),
        specific_heat_j_kgk=840.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.2,
        compressive_strength_kpa=40.0,
        astm_standard="ASTM C547, C612",
        applications=[
            "Pipe insulation up to 650C",
            "Equipment insulation",
            "Boiler insulation",
            "Tank insulation",
            "HVAC ductwork"
        ]
    )

    MINERAL_WOOL_GLASS = InsulationMaterialSpec(
        name="Glass Wool (Fiberglass)",
        material_type=InsulationType.MINERAL_WOOL_GLASS,
        # k(T) = 0.0330 + 0.000085*T + 0.00000015*T^2
        k_coefficients=(0.0330, 8.5e-5, 1.5e-7, 0.0),
        min_temp_c=-40.0,
        max_temp_c=450.0,
        density_kg_m3=(10.0, 100.0),
        specific_heat_j_kgk=700.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.5,
        compressive_strength_kpa=20.0,
        astm_standard="ASTM C547, C612, C592",
        applications=[
            "Pipe insulation up to 450C",
            "HVAC systems",
            "Building insulation",
            "Acoustic applications",
            "Medium temperature equipment"
        ]
    )

    MINERAL_WOOL_SLAG = InsulationMaterialSpec(
        name="Slag Wool",
        material_type=InsulationType.MINERAL_WOOL_SLAG,
        # k(T) = 0.0360 + 0.000090*T + 0.00000018*T^2
        k_coefficients=(0.0360, 9.0e-5, 1.8e-7, 0.0),
        min_temp_c=-40.0,
        max_temp_c=760.0,
        density_kg_m3=(48.0, 192.0),
        specific_heat_j_kgk=800.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.3,
        compressive_strength_kpa=35.0,
        astm_standard="ASTM C547, C795",
        applications=[
            "High temperature pipe insulation",
            "Power plant applications",
            "Industrial furnaces",
            "Petrochemical processing"
        ]
    )

    # =========================================================================
    # CALCIUM SILICATE INSULATIONS
    # =========================================================================

    CALCIUM_SILICATE = InsulationMaterialSpec(
        name="Calcium Silicate",
        material_type=InsulationType.CALCIUM_SILICATE,
        # k(T) = 0.0520 + 0.000100*T + 0.00000015*T^2
        # Standard grade, valid to 650C
        k_coefficients=(0.0520, 1.0e-4, 1.5e-7, 0.0),
        min_temp_c=-18.0,
        max_temp_c=650.0,
        density_kg_m3=(220.0, 350.0),
        specific_heat_j_kgk=840.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=90.0,  # High absorption by design
        compressive_strength_kpa=690.0,
        astm_standard="ASTM C533",
        applications=[
            "High temperature pipe insulation",
            "Steam distribution systems",
            "Valves and fittings",
            "Equipment operating to 650C",
            "Fire protection"
        ]
    )

    CALCIUM_SILICATE_HIGH_TEMP = InsulationMaterialSpec(
        name="Calcium Silicate High Temperature",
        material_type=InsulationType.CALCIUM_SILICATE_HIGH_TEMP,
        # k(T) = 0.0580 + 0.000120*T + 0.00000020*T^2
        # High temp grade, valid to 1050C
        k_coefficients=(0.0580, 1.2e-4, 2.0e-7, 0.0),
        min_temp_c=0.0,
        max_temp_c=1050.0,
        density_kg_m3=(240.0, 400.0),
        specific_heat_j_kgk=920.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=85.0,
        compressive_strength_kpa=800.0,
        astm_standard="ASTM C533 Type I",
        applications=[
            "Very high temperature applications",
            "Reformer furnaces",
            "Heater tubes",
            "Turbine exhaust"
        ]
    )

    # =========================================================================
    # CELLULAR GLASS INSULATIONS
    # =========================================================================

    CELLULAR_GLASS = InsulationMaterialSpec(
        name="Cellular Glass (Foamglas)",
        material_type=InsulationType.CELLULAR_GLASS,
        # k(T) = 0.0400 + 0.000080*T + 0.00000012*T^2
        # Standard density, valid -268C to 430C
        k_coefficients=(0.0400, 8.0e-5, 1.2e-7, 0.0),
        min_temp_c=-268.0,  # Cryogenic applications
        max_temp_c=430.0,
        density_kg_m3=(100.0, 165.0),
        specific_heat_j_kgk=840.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.0,  # Impermeable to water
        compressive_strength_kpa=700.0,
        astm_standard="ASTM C552",
        applications=[
            "Cryogenic insulation (LNG, LN2)",
            "Underground piping",
            "Below-grade applications",
            "Cold storage",
            "Where moisture resistance critical"
        ]
    )

    CELLULAR_GLASS_HIGH_DENSITY = InsulationMaterialSpec(
        name="Cellular Glass High Density",
        material_type=InsulationType.CELLULAR_GLASS_HIGH_DENSITY,
        # k(T) = 0.0480 + 0.000090*T + 0.00000015*T^2
        k_coefficients=(0.0480, 9.0e-5, 1.5e-7, 0.0),
        min_temp_c=-268.0,
        max_temp_c=430.0,
        density_kg_m3=(165.0, 240.0),
        specific_heat_j_kgk=840.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.0,
        compressive_strength_kpa=1400.0,
        astm_standard="ASTM C552 Type IV",
        applications=[
            "Heavy load bearing",
            "Pipe supports",
            "Tank bottoms",
            "Structural insulation"
        ]
    )

    # =========================================================================
    # FOAM PLASTIC INSULATIONS
    # =========================================================================

    POLYURETHANE_FOAM = InsulationMaterialSpec(
        name="Polyurethane Foam (Rigid)",
        material_type=InsulationType.POLYURETHANE_FOAM,
        # k(T) = 0.0220 + 0.000050*T + 0.00000010*T^2
        # Very low k value, limited temperature
        k_coefficients=(0.0220, 5.0e-5, 1.0e-7, 0.0),
        min_temp_c=-200.0,
        max_temp_c=120.0,
        density_kg_m3=(30.0, 80.0),
        specific_heat_j_kgk=1500.0,
        fire_rating="Class I (25 flame spread)",
        moisture_absorption=2.0,
        compressive_strength_kpa=200.0,
        astm_standard="ASTM C591",
        applications=[
            "Low temperature piping",
            "Refrigeration systems",
            "Chilled water lines",
            "Cold storage",
            "District cooling"
        ]
    )

    POLYURETHANE_SPRAY = InsulationMaterialSpec(
        name="Polyurethane Spray Foam",
        material_type=InsulationType.POLYURETHANE_SPRAY,
        # k(T) = 0.0240 + 0.000055*T + 0.00000012*T^2
        k_coefficients=(0.0240, 5.5e-5, 1.2e-7, 0.0),
        min_temp_c=-180.0,
        max_temp_c=100.0,
        density_kg_m3=(28.0, 55.0),
        specific_heat_j_kgk=1400.0,
        fire_rating="Class II (75 flame spread)",
        moisture_absorption=3.0,
        compressive_strength_kpa=150.0,
        astm_standard="ASTM C1029",
        applications=[
            "Complex geometries",
            "Tank exteriors",
            "Irregular surfaces",
            "Field-applied insulation"
        ]
    )

    POLYISOCYANURATE = InsulationMaterialSpec(
        name="Polyisocyanurate (PIR)",
        material_type=InsulationType.POLYISOCYANURATE,
        # k(T) = 0.0230 + 0.000048*T + 0.00000008*T^2
        # Better fire resistance than PU
        k_coefficients=(0.0230, 4.8e-5, 8.0e-8, 0.0),
        min_temp_c=-180.0,
        max_temp_c=150.0,
        density_kg_m3=(32.0, 64.0),
        specific_heat_j_kgk=1450.0,
        fire_rating="Class I (25 flame spread)",
        moisture_absorption=1.5,
        compressive_strength_kpa=250.0,
        astm_standard="ASTM C591",
        applications=[
            "Industrial piping",
            "Commercial HVAC",
            "Pre-insulated pipe systems",
            "Moderate temperature applications"
        ]
    )

    POLYSTYRENE_EPS = InsulationMaterialSpec(
        name="Expanded Polystyrene (EPS)",
        material_type=InsulationType.POLYSTYRENE_EPS,
        # k(T) = 0.0350 + 0.000060*T + 0.00000010*T^2
        k_coefficients=(0.0350, 6.0e-5, 1.0e-7, 0.0),
        min_temp_c=-50.0,
        max_temp_c=75.0,
        density_kg_m3=(15.0, 35.0),
        specific_heat_j_kgk=1300.0,
        fire_rating="Class III (burning)",
        moisture_absorption=3.0,
        compressive_strength_kpa=100.0,
        astm_standard="ASTM C578",
        applications=[
            "Cold storage walls",
            "Building insulation",
            "Below grade",
            "Low temperature only"
        ]
    )

    POLYSTYRENE_XPS = InsulationMaterialSpec(
        name="Extruded Polystyrene (XPS)",
        material_type=InsulationType.POLYSTYRENE_XPS,
        # k(T) = 0.0290 + 0.000055*T + 0.00000010*T^2
        k_coefficients=(0.0290, 5.5e-5, 1.0e-7, 0.0),
        min_temp_c=-50.0,
        max_temp_c=75.0,
        density_kg_m3=(25.0, 45.0),
        specific_heat_j_kgk=1300.0,
        fire_rating="Class I (25 flame spread)",
        moisture_absorption=0.3,
        compressive_strength_kpa=300.0,
        astm_standard="ASTM C578",
        applications=[
            "Below grade insulation",
            "Roofing",
            "Cold storage",
            "Moisture-prone areas"
        ]
    )

    PHENOLIC_FOAM = InsulationMaterialSpec(
        name="Phenolic Foam",
        material_type=InsulationType.PHENOLIC_FOAM,
        # k(T) = 0.0200 + 0.000040*T + 0.00000008*T^2
        # Excellent k value, good fire properties
        k_coefficients=(0.0200, 4.0e-5, 8.0e-8, 0.0),
        min_temp_c=-180.0,
        max_temp_c=150.0,
        density_kg_m3=(35.0, 120.0),
        specific_heat_j_kgk=1400.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=5.0,
        compressive_strength_kpa=150.0,
        astm_standard="ASTM C1126",
        applications=[
            "HVAC ductwork",
            "Building services",
            "Refrigeration",
            "Where fire rating critical"
        ]
    )

    # =========================================================================
    # HIGH-PERFORMANCE INSULATIONS
    # =========================================================================

    AEROGEL_BLANKET = InsulationMaterialSpec(
        name="Aerogel Blanket",
        material_type=InsulationType.AEROGEL_BLANKET,
        # k(T) = 0.0130 + 0.000030*T + 0.00000005*T^2
        # Lowest k of any practical insulation
        k_coefficients=(0.0130, 3.0e-5, 5.0e-8, 0.0),
        min_temp_c=-200.0,
        max_temp_c=650.0,
        density_kg_m3=(100.0, 200.0),
        specific_heat_j_kgk=1000.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.5,
        compressive_strength_kpa=50.0,
        astm_standard="ASTM C1728",
        applications=[
            "Space-constrained applications",
            "Subsea pipelines",
            "LNG facilities",
            "High-value equipment",
            "Retrofit insulation upgrades"
        ]
    )

    AEROGEL_COMPOSITE = InsulationMaterialSpec(
        name="Aerogel Composite",
        material_type=InsulationType.AEROGEL_COMPOSITE,
        # k(T) = 0.0150 + 0.000035*T + 0.00000006*T^2
        k_coefficients=(0.0150, 3.5e-5, 6.0e-8, 0.0),
        min_temp_c=-200.0,
        max_temp_c=1000.0,
        density_kg_m3=(150.0, 300.0),
        specific_heat_j_kgk=950.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.3,
        compressive_strength_kpa=100.0,
        astm_standard="ASTM C1728",
        applications=[
            "Ultra-high temperature",
            "Turbine insulation",
            "Aerospace applications",
            "Premium installations"
        ]
    )

    MICROPOROUS = InsulationMaterialSpec(
        name="Microporous Insulation",
        material_type=InsulationType.MICROPOROUS,
        # k(T) = 0.0180 + 0.000025*T + 0.00000008*T^2
        k_coefficients=(0.0180, 2.5e-5, 8.0e-8, 0.0),
        min_temp_c=-200.0,
        max_temp_c=1000.0,
        density_kg_m3=(200.0, 400.0),
        specific_heat_j_kgk=900.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.5,
        compressive_strength_kpa=200.0,
        astm_standard="No specific ASTM",
        applications=[
            "Very high temperature",
            "Space-limited applications",
            "Turbine casings",
            "Fire protection"
        ]
    )

    # =========================================================================
    # GRANULAR INSULATIONS
    # =========================================================================

    PERLITE_EXPANDED = InsulationMaterialSpec(
        name="Expanded Perlite",
        material_type=InsulationType.PERLITE_EXPANDED,
        # k(T) = 0.0450 + 0.000095*T + 0.00000012*T^2
        k_coefficients=(0.0450, 9.5e-5, 1.2e-7, 0.0),
        min_temp_c=-200.0,
        max_temp_c=980.0,
        density_kg_m3=(50.0, 150.0),
        specific_heat_j_kgk=840.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=1.0,
        compressive_strength_kpa=None,  # Loose fill
        astm_standard="ASTM C549",
        applications=[
            "Loose fill insulation",
            "Masonry cavities",
            "Cryogenic tank annulus",
            "Refractory backup"
        ]
    )

    PERLITE_POWDER = InsulationMaterialSpec(
        name="Perlite Powder",
        material_type=InsulationType.PERLITE_POWDER,
        # k(T) = 0.0500 + 0.000100*T + 0.00000015*T^2
        k_coefficients=(0.0500, 1.0e-4, 1.5e-7, 0.0),
        min_temp_c=-268.0,
        max_temp_c=650.0,
        density_kg_m3=(30.0, 80.0),
        specific_heat_j_kgk=840.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.5,
        compressive_strength_kpa=None,
        astm_standard="ASTM C549",
        applications=[
            "Cryogenic applications",
            "LNG storage tanks",
            "Evacuated annular spaces"
        ]
    )

    VERMICULITE = InsulationMaterialSpec(
        name="Vermiculite",
        material_type=InsulationType.VERMICULITE,
        # k(T) = 0.0550 + 0.000110*T + 0.00000015*T^2
        k_coefficients=(0.0550, 1.1e-4, 1.5e-7, 0.0),
        min_temp_c=-40.0,
        max_temp_c=1100.0,
        density_kg_m3=(60.0, 200.0),
        specific_heat_j_kgk=1050.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=5.0,
        compressive_strength_kpa=None,
        astm_standard="ASTM C516",
        applications=[
            "Loose fill",
            "Fire protection",
            "Refractory backup",
            "Acoustic applications"
        ]
    )

    DIATOMACEOUS_EARTH = InsulationMaterialSpec(
        name="Diatomaceous Earth",
        material_type=InsulationType.DIATOMACEOUS_EARTH,
        # k(T) = 0.0600 + 0.000120*T + 0.00000020*T^2
        k_coefficients=(0.0600, 1.2e-4, 2.0e-7, 0.0),
        min_temp_c=0.0,
        max_temp_c=900.0,
        density_kg_m3=(220.0, 450.0),
        specific_heat_j_kgk=880.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=50.0,
        compressive_strength_kpa=500.0,
        astm_standard="ASTM C533",
        applications=[
            "High temperature equipment",
            "Legacy installations",
            "Boiler casings"
        ]
    )

    # =========================================================================
    # REFRACTORY INSULATIONS
    # =========================================================================

    CERAMIC_FIBER = InsulationMaterialSpec(
        name="Ceramic Fiber Blanket",
        material_type=InsulationType.CERAMIC_FIBER,
        # k(T) = 0.0350 + 0.000120*T + 0.00000025*T^2
        k_coefficients=(0.0350, 1.2e-4, 2.5e-7, 0.0),
        min_temp_c=0.0,
        max_temp_c=1260.0,
        density_kg_m3=(64.0, 192.0),
        specific_heat_j_kgk=1130.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.5,
        compressive_strength_kpa=None,  # Flexible blanket
        astm_standard="ASTM C892",
        applications=[
            "Furnace linings",
            "Kiln insulation",
            "High temperature gaskets",
            "Expansion joints"
        ]
    )

    CERAMIC_FIBER_BOARD = InsulationMaterialSpec(
        name="Ceramic Fiber Board",
        material_type=InsulationType.CERAMIC_FIBER_BOARD,
        # k(T) = 0.0400 + 0.000130*T + 0.00000028*T^2
        k_coefficients=(0.0400, 1.3e-4, 2.8e-7, 0.0),
        min_temp_c=0.0,
        max_temp_c=1430.0,
        density_kg_m3=(200.0, 400.0),
        specific_heat_j_kgk=1130.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.5,
        compressive_strength_kpa=200.0,
        astm_standard="ASTM C892",
        applications=[
            "Backup insulation",
            "Furnace door linings",
            "High temperature gasketing"
        ]
    )

    REFRACTORY_BRICK = InsulationMaterialSpec(
        name="Insulating Refractory Brick",
        material_type=InsulationType.REFRACTORY_BRICK,
        # k(T) = 0.1200 + 0.000150*T + 0.00000020*T^2
        k_coefficients=(0.1200, 1.5e-4, 2.0e-7, 0.0),
        min_temp_c=0.0,
        max_temp_c=1650.0,
        density_kg_m3=(400.0, 800.0),
        specific_heat_j_kgk=1050.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=10.0,
        compressive_strength_kpa=1000.0,
        astm_standard="ASTM C155",
        applications=[
            "Furnace linings",
            "Kiln construction",
            "Boiler settings",
            "Incinerators"
        ]
    )

    FIREBRICK_INSULATING = InsulationMaterialSpec(
        name="Insulating Firebrick (IFB)",
        material_type=InsulationType.FIREBRICK_INSULATING,
        # k(T) = 0.1500 + 0.000180*T + 0.00000025*T^2
        k_coefficients=(0.1500, 1.8e-4, 2.5e-7, 0.0),
        min_temp_c=0.0,
        max_temp_c=1500.0,
        density_kg_m3=(500.0, 1000.0),
        specific_heat_j_kgk=1000.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=15.0,
        compressive_strength_kpa=1500.0,
        astm_standard="ASTM C155",
        applications=[
            "Hot face lining",
            "Kiln construction",
            "Glass furnaces"
        ]
    )

    # =========================================================================
    # ELASTOMERIC INSULATIONS
    # =========================================================================

    ELASTOMERIC_FOAM = InsulationMaterialSpec(
        name="Elastomeric Foam",
        material_type=InsulationType.ELASTOMERIC_FOAM,
        # k(T) = 0.0350 + 0.000055*T + 0.00000008*T^2
        k_coefficients=(0.0350, 5.5e-5, 8.0e-8, 0.0),
        min_temp_c=-40.0,
        max_temp_c=105.0,
        density_kg_m3=(40.0, 100.0),
        specific_heat_j_kgk=2000.0,
        fire_rating="Class I (25 flame spread)",
        moisture_absorption=0.0,  # Closed cell
        compressive_strength_kpa=100.0,
        astm_standard="ASTM C534",
        applications=[
            "HVAC systems",
            "Refrigeration lines",
            "Chilled water piping",
            "Condensation prevention"
        ]
    )

    ELASTOMERIC_SHEET = InsulationMaterialSpec(
        name="Elastomeric Sheet",
        material_type=InsulationType.ELASTOMERIC_SHEET,
        # k(T) = 0.0360 + 0.000058*T + 0.00000009*T^2
        k_coefficients=(0.0360, 5.8e-5, 9.0e-8, 0.0),
        min_temp_c=-40.0,
        max_temp_c=105.0,
        density_kg_m3=(50.0, 120.0),
        specific_heat_j_kgk=2000.0,
        fire_rating="Class I (25 flame spread)",
        moisture_absorption=0.0,
        compressive_strength_kpa=120.0,
        astm_standard="ASTM C534",
        applications=[
            "Flat surfaces",
            "Tanks and vessels",
            "Equipment insulation"
        ]
    )

    # =========================================================================
    # SPECIALTY INSULATIONS
    # =========================================================================

    MELAMINE_FOAM = InsulationMaterialSpec(
        name="Melamine Foam",
        material_type=InsulationType.MELAMINE_FOAM,
        # k(T) = 0.0350 + 0.000050*T + 0.00000008*T^2
        k_coefficients=(0.0350, 5.0e-5, 8.0e-8, 0.0),
        min_temp_c=-40.0,
        max_temp_c=150.0,
        density_kg_m3=(8.0, 12.0),
        specific_heat_j_kgk=1400.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.5,
        compressive_strength_kpa=10.0,
        astm_standard="No specific ASTM",
        applications=[
            "Acoustic applications",
            "HVAC ductwork",
            "Lightweight insulation"
        ]
    )

    CRYOGENIC_PERLITE = InsulationMaterialSpec(
        name="Cryogenic Perlite",
        material_type=InsulationType.CRYOGENIC_PERLITE,
        # k(T) = 0.0200 + 0.000040*T + 0.00000005*T^2
        # Evacuated perlite for cryogenic tanks
        k_coefficients=(0.0200, 4.0e-5, 5.0e-8, 0.0),
        min_temp_c=-268.0,
        max_temp_c=50.0,
        density_kg_m3=(50.0, 100.0),
        specific_heat_j_kgk=840.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.0,  # Evacuated
        compressive_strength_kpa=None,
        astm_standard="ASTM C549",
        applications=[
            "LNG storage tanks",
            "Liquid nitrogen",
            "Cryogenic vessels"
        ]
    )

    CRYOGENIC_MLI = InsulationMaterialSpec(
        name="Multi-Layer Insulation (MLI)",
        material_type=InsulationType.CRYOGENIC_MLI,
        # Extremely low k in vacuum, expressed as apparent k
        # k(T) = 0.0001 + 0.000005*T + 0.00000001*T^2
        k_coefficients=(0.0001, 5.0e-6, 1.0e-8, 0.0),
        min_temp_c=-268.0,
        max_temp_c=50.0,
        density_kg_m3=(20.0, 80.0),
        specific_heat_j_kgk=500.0,
        fire_rating="Class A (0 flame spread)",
        moisture_absorption=0.0,
        compressive_strength_kpa=None,
        astm_standard="No specific ASTM",
        applications=[
            "Space applications",
            "LHe containers",
            "Research cryostats",
            "Ultra-low temperature"
        ]
    )

    # =========================================================================
    # MATERIAL LOOKUP DICTIONARY
    # =========================================================================

    _MATERIALS: Dict[str, InsulationMaterialSpec] = {
        "mineral_wool_rock": MINERAL_WOOL_ROCK,
        "mineral_wool_glass": MINERAL_WOOL_GLASS,
        "mineral_wool_slag": MINERAL_WOOL_SLAG,
        "calcium_silicate": CALCIUM_SILICATE,
        "calcium_silicate_high_temp": CALCIUM_SILICATE_HIGH_TEMP,
        "cellular_glass": CELLULAR_GLASS,
        "cellular_glass_high_density": CELLULAR_GLASS_HIGH_DENSITY,
        "polyurethane_foam": POLYURETHANE_FOAM,
        "polyurethane_spray": POLYURETHANE_SPRAY,
        "polyisocyanurate": POLYISOCYANURATE,
        "polystyrene_eps": POLYSTYRENE_EPS,
        "polystyrene_xps": POLYSTYRENE_XPS,
        "phenolic_foam": PHENOLIC_FOAM,
        "aerogel_blanket": AEROGEL_BLANKET,
        "aerogel_composite": AEROGEL_COMPOSITE,
        "microporous": MICROPOROUS,
        "perlite_expanded": PERLITE_EXPANDED,
        "perlite_powder": PERLITE_POWDER,
        "vermiculite": VERMICULITE,
        "diatomaceous_earth": DIATOMACEOUS_EARTH,
        "ceramic_fiber": CERAMIC_FIBER,
        "ceramic_fiber_board": CERAMIC_FIBER_BOARD,
        "refractory_brick": REFRACTORY_BRICK,
        "firebrick_insulating": FIREBRICK_INSULATING,
        "elastomeric_foam": ELASTOMERIC_FOAM,
        "elastomeric_sheet": ELASTOMERIC_SHEET,
        "melamine_foam": MELAMINE_FOAM,
        "cryogenic_perlite": CRYOGENIC_PERLITE,
        "cryogenic_mli": CRYOGENIC_MLI,
    }

    @classmethod
    def get_material_spec(cls, material_id: str) -> InsulationMaterialSpec:
        """
        Get complete material specification.

        Args:
            material_id: Material identifier string

        Returns:
            InsulationMaterialSpec for the material

        Raises:
            KeyError: If material not found
        """
        if material_id not in cls._MATERIALS:
            available = ", ".join(sorted(cls._MATERIALS.keys()))
            raise KeyError(
                f"Unknown material: {material_id}. "
                f"Available materials: {available}"
            )
        return cls._MATERIALS[material_id]

    @classmethod
    @lru_cache(maxsize=1000)
    def get_thermal_conductivity(
        cls,
        material_id: str,
        temperature_c: float
    ) -> float:
        """
        Get thermal conductivity at specified temperature.

        Uses cached polynomial evaluation for performance.

        Args:
            material_id: Material identifier string
            temperature_c: Mean temperature in Celsius

        Returns:
            Thermal conductivity in W/(m*K)

        Raises:
            KeyError: If material not found
            ValueError: If temperature outside valid range
        """
        material = cls.get_material_spec(material_id)
        return material.thermal_conductivity(temperature_c)

    @classmethod
    def list_materials(cls) -> List[str]:
        """Get list of all available material IDs."""
        return sorted(cls._MATERIALS.keys())

    @classmethod
    def get_materials_for_temperature(
        cls,
        temperature_c: float
    ) -> List[InsulationMaterialSpec]:
        """
        Get all materials suitable for given operating temperature.

        Args:
            temperature_c: Operating temperature in Celsius

        Returns:
            List of suitable material specifications
        """
        suitable = []
        for material in cls._MATERIALS.values():
            if material.min_temp_c <= temperature_c <= material.max_temp_c:
                suitable.append(material)
        return sorted(suitable, key=lambda m: m.thermal_conductivity(temperature_c))


# =============================================================================
# SURFACE EMISSIVITY TABLES
# =============================================================================

@dataclass(frozen=True)
class EmissivitySpec:
    """
    Surface emissivity specification.

    Attributes:
        name: Description of surface condition
        emissivity_low: Lower bound of emissivity range
        emissivity_high: Upper bound of emissivity range
        emissivity_typical: Typical/nominal value
        temperature_range_c: Valid temperature range (min, max)
        notes: Additional notes about conditions
    """
    name: str
    emissivity_low: float
    emissivity_high: float
    emissivity_typical: float
    temperature_range_c: Tuple[float, float]
    notes: str = ""

    @property
    def emissivity(self) -> float:
        """Return typical emissivity value."""
        return self.emissivity_typical


class SurfaceEmissivity:
    """
    Comprehensive emissivity database for industrial surfaces.

    Contains emissivity values for various cladding materials, bare
    pipe surfaces, and weathered conditions. Values are sourced from
    ASHRAE Handbook and manufacturer data.

    Usage:
        >>> e = SurfaceEmissivity.get_emissivity("aluminum_polished")
        >>> spec = SurfaceEmissivity.get_spec("stainless_steel_weathered")
    """

    # =========================================================================
    # ALUMINUM SURFACES
    # =========================================================================

    ALUMINUM_POLISHED = EmissivitySpec(
        name="Polished Aluminum",
        emissivity_low=0.02,
        emissivity_high=0.05,
        emissivity_typical=0.04,
        temperature_range_c=(-50, 200),
        notes="Highly polished, new condition"
    )

    ALUMINUM_COMMERCIAL = EmissivitySpec(
        name="Commercial Aluminum Sheet",
        emissivity_low=0.05,
        emissivity_high=0.10,
        emissivity_typical=0.07,
        temperature_range_c=(-50, 200),
        notes="Mill finish, unpolished"
    )

    ALUMINUM_OXIDIZED = EmissivitySpec(
        name="Oxidized Aluminum",
        emissivity_low=0.10,
        emissivity_high=0.20,
        emissivity_typical=0.15,
        temperature_range_c=(-50, 200),
        notes="Natural oxide layer after exposure"
    )

    ALUMINUM_WEATHERED = EmissivitySpec(
        name="Weathered Aluminum",
        emissivity_low=0.20,
        emissivity_high=0.35,
        emissivity_typical=0.28,
        temperature_range_c=(-50, 200),
        notes="Long-term outdoor exposure"
    )

    ALUMINUM_ANODIZED = EmissivitySpec(
        name="Anodized Aluminum",
        emissivity_low=0.55,
        emissivity_high=0.75,
        emissivity_typical=0.65,
        temperature_range_c=(-50, 300),
        notes="Clear anodized finish"
    )

    ALUMINUM_PAINTED_WHITE = EmissivitySpec(
        name="White Painted Aluminum",
        emissivity_low=0.85,
        emissivity_high=0.95,
        emissivity_typical=0.90,
        temperature_range_c=(-50, 150),
        notes="High-quality white paint"
    )

    # =========================================================================
    # STAINLESS STEEL SURFACES
    # =========================================================================

    STAINLESS_STEEL_POLISHED = EmissivitySpec(
        name="Polished Stainless Steel",
        emissivity_low=0.05,
        emissivity_high=0.10,
        emissivity_typical=0.07,
        temperature_range_c=(-50, 500),
        notes="Type 304/316, highly polished"
    )

    STAINLESS_STEEL_MILL = EmissivitySpec(
        name="Mill Finish Stainless Steel",
        emissivity_low=0.10,
        emissivity_high=0.20,
        emissivity_typical=0.15,
        temperature_range_c=(-50, 500),
        notes="2B or 2D finish"
    )

    STAINLESS_STEEL_OXIDIZED = EmissivitySpec(
        name="Oxidized Stainless Steel",
        emissivity_low=0.20,
        emissivity_high=0.40,
        emissivity_typical=0.30,
        temperature_range_c=(-50, 800),
        notes="Heat-tinted or oxidized"
    )

    STAINLESS_STEEL_WEATHERED = EmissivitySpec(
        name="Weathered Stainless Steel",
        emissivity_low=0.40,
        emissivity_high=0.60,
        emissivity_typical=0.50,
        temperature_range_c=(-50, 500),
        notes="Long-term outdoor exposure"
    )

    # =========================================================================
    # GALVANIZED STEEL SURFACES
    # =========================================================================

    GALVANIZED_BRIGHT = EmissivitySpec(
        name="Bright Galvanized Steel",
        emissivity_low=0.20,
        emissivity_high=0.30,
        emissivity_typical=0.25,
        temperature_range_c=(-50, 200),
        notes="New, spangled surface"
    )

    GALVANIZED_DULL = EmissivitySpec(
        name="Dull Galvanized Steel",
        emissivity_low=0.25,
        emissivity_high=0.35,
        emissivity_typical=0.30,
        temperature_range_c=(-50, 200),
        notes="Aged or matte finish"
    )

    GALVANIZED_OXIDIZED = EmissivitySpec(
        name="Oxidized Galvanized Steel",
        emissivity_low=0.30,
        emissivity_high=0.45,
        emissivity_typical=0.38,
        temperature_range_c=(-50, 200),
        notes="White rust formation"
    )

    GALVANIZED_WEATHERED = EmissivitySpec(
        name="Weathered Galvanized Steel",
        emissivity_low=0.45,
        emissivity_high=0.60,
        emissivity_typical=0.52,
        temperature_range_c=(-50, 200),
        notes="Long-term outdoor exposure"
    )

    # =========================================================================
    # CARBON STEEL SURFACES
    # =========================================================================

    CARBON_STEEL_POLISHED = EmissivitySpec(
        name="Polished Carbon Steel",
        emissivity_low=0.05,
        emissivity_high=0.10,
        emissivity_typical=0.07,
        temperature_range_c=(-50, 500),
        notes="Highly polished surface"
    )

    CARBON_STEEL_MILL_SCALE = EmissivitySpec(
        name="Carbon Steel with Mill Scale",
        emissivity_low=0.55,
        emissivity_high=0.70,
        emissivity_typical=0.62,
        temperature_range_c=(-50, 500),
        notes="Hot rolled with scale"
    )

    CARBON_STEEL_RUSTED = EmissivitySpec(
        name="Rusted Carbon Steel",
        emissivity_low=0.60,
        emissivity_high=0.85,
        emissivity_typical=0.75,
        temperature_range_c=(-50, 400),
        notes="Light to moderate rust"
    )

    CARBON_STEEL_HEAVILY_RUSTED = EmissivitySpec(
        name="Heavily Rusted Carbon Steel",
        emissivity_low=0.80,
        emissivity_high=0.95,
        emissivity_typical=0.88,
        temperature_range_c=(-50, 400),
        notes="Heavy rust, flaking"
    )

    CARBON_STEEL_OXIDIZED_HIGH_TEMP = EmissivitySpec(
        name="High-Temp Oxidized Carbon Steel",
        emissivity_low=0.75,
        emissivity_high=0.90,
        emissivity_typical=0.82,
        temperature_range_c=(200, 800),
        notes="Black oxide at high temperature"
    )

    # =========================================================================
    # JACKETING MATERIALS
    # =========================================================================

    ASJ_ALL_SERVICE_JACKET = EmissivitySpec(
        name="All Service Jacket (ASJ)",
        emissivity_low=0.85,
        emissivity_high=0.95,
        emissivity_typical=0.90,
        temperature_range_c=(-50, 80),
        notes="White kraft/foil facing"
    )

    PVC_JACKET_WHITE = EmissivitySpec(
        name="White PVC Jacket",
        emissivity_low=0.85,
        emissivity_high=0.95,
        emissivity_typical=0.90,
        temperature_range_c=(-50, 65),
        notes="Standard white PVC"
    )

    PVC_JACKET_BLACK = EmissivitySpec(
        name="Black PVC Jacket",
        emissivity_low=0.90,
        emissivity_high=0.97,
        emissivity_typical=0.93,
        temperature_range_c=(-50, 65),
        notes="Black PVC for UV resistance"
    )

    ALUMINUM_JACKET_SMOOTH = EmissivitySpec(
        name="Smooth Aluminum Jacket",
        emissivity_low=0.05,
        emissivity_high=0.12,
        emissivity_typical=0.08,
        temperature_range_c=(-50, 200),
        notes="New, smooth aluminum"
    )

    ALUMINUM_JACKET_CORRUGATED = EmissivitySpec(
        name="Corrugated Aluminum Jacket",
        emissivity_low=0.08,
        emissivity_high=0.15,
        emissivity_typical=0.12,
        temperature_range_c=(-50, 200),
        notes="Corrugated for flexibility"
    )

    ALUMINUM_JACKET_EMBOSSED = EmissivitySpec(
        name="Embossed Aluminum Jacket",
        emissivity_low=0.10,
        emissivity_high=0.20,
        emissivity_typical=0.15,
        temperature_range_c=(-50, 200),
        notes="Stucco embossed pattern"
    )

    STAINLESS_JACKET_304 = EmissivitySpec(
        name="Stainless Steel Jacket (304)",
        emissivity_low=0.12,
        emissivity_high=0.20,
        emissivity_typical=0.16,
        temperature_range_c=(-50, 500),
        notes="Type 304 stainless"
    )

    STAINLESS_JACKET_316 = EmissivitySpec(
        name="Stainless Steel Jacket (316)",
        emissivity_low=0.12,
        emissivity_high=0.20,
        emissivity_typical=0.16,
        temperature_range_c=(-50, 500),
        notes="Type 316 marine grade"
    )

    # =========================================================================
    # PAINTED SURFACES
    # =========================================================================

    PAINT_WHITE = EmissivitySpec(
        name="White Paint",
        emissivity_low=0.85,
        emissivity_high=0.95,
        emissivity_typical=0.90,
        temperature_range_c=(-50, 150),
        notes="Flat or semi-gloss white"
    )

    PAINT_BLACK = EmissivitySpec(
        name="Black Paint",
        emissivity_low=0.90,
        emissivity_high=0.98,
        emissivity_typical=0.95,
        temperature_range_c=(-50, 150),
        notes="Flat or semi-gloss black"
    )

    PAINT_ALUMINUM = EmissivitySpec(
        name="Aluminum Paint",
        emissivity_low=0.25,
        emissivity_high=0.45,
        emissivity_typical=0.35,
        temperature_range_c=(-50, 200),
        notes="Aluminum-based paint"
    )

    PAINT_HIGH_TEMP_SILICONE = EmissivitySpec(
        name="High-Temp Silicone Paint",
        emissivity_low=0.80,
        emissivity_high=0.90,
        emissivity_typical=0.85,
        temperature_range_c=(-50, 650),
        notes="Silicone-based heat resistant"
    )

    PAINT_ZINC_RICH = EmissivitySpec(
        name="Zinc-Rich Primer",
        emissivity_low=0.30,
        emissivity_high=0.50,
        emissivity_typical=0.40,
        temperature_range_c=(-50, 200),
        notes="Zinc-rich primer coating"
    )

    # =========================================================================
    # MASTIC AND COATINGS
    # =========================================================================

    MASTIC_WHITE = EmissivitySpec(
        name="White Mastic Coating",
        emissivity_low=0.85,
        emissivity_high=0.95,
        emissivity_typical=0.90,
        temperature_range_c=(-50, 100),
        notes="Weather barrier mastic"
    )

    MASTIC_GRAY = EmissivitySpec(
        name="Gray Mastic Coating",
        emissivity_low=0.80,
        emissivity_high=0.92,
        emissivity_typical=0.86,
        temperature_range_c=(-50, 100),
        notes="Industrial gray mastic"
    )

    # =========================================================================
    # BARE INSULATION SURFACES
    # =========================================================================

    FIBERGLASS_BARE = EmissivitySpec(
        name="Bare Fiberglass Insulation",
        emissivity_low=0.70,
        emissivity_high=0.85,
        emissivity_typical=0.78,
        temperature_range_c=(-50, 300),
        notes="Unfinished fiberglass surface"
    )

    MINERAL_WOOL_BARE = EmissivitySpec(
        name="Bare Mineral Wool",
        emissivity_low=0.75,
        emissivity_high=0.88,
        emissivity_typical=0.82,
        temperature_range_c=(-50, 650),
        notes="Unfinished mineral wool"
    )

    CALCIUM_SILICATE_BARE = EmissivitySpec(
        name="Bare Calcium Silicate",
        emissivity_low=0.80,
        emissivity_high=0.92,
        emissivity_typical=0.86,
        temperature_range_c=(-50, 650),
        notes="Unpainted calcium silicate"
    )

    # =========================================================================
    # LOOKUP DICTIONARY
    # =========================================================================

    _EMISSIVITIES: Dict[str, EmissivitySpec] = {
        # Aluminum
        "aluminum_polished": ALUMINUM_POLISHED,
        "aluminum_commercial": ALUMINUM_COMMERCIAL,
        "aluminum_oxidized": ALUMINUM_OXIDIZED,
        "aluminum_weathered": ALUMINUM_WEATHERED,
        "aluminum_anodized": ALUMINUM_ANODIZED,
        "aluminum_painted_white": ALUMINUM_PAINTED_WHITE,

        # Stainless Steel
        "stainless_steel_polished": STAINLESS_STEEL_POLISHED,
        "stainless_steel_mill": STAINLESS_STEEL_MILL,
        "stainless_steel_oxidized": STAINLESS_STEEL_OXIDIZED,
        "stainless_steel_weathered": STAINLESS_STEEL_WEATHERED,

        # Galvanized Steel
        "galvanized_bright": GALVANIZED_BRIGHT,
        "galvanized_dull": GALVANIZED_DULL,
        "galvanized_oxidized": GALVANIZED_OXIDIZED,
        "galvanized_weathered": GALVANIZED_WEATHERED,

        # Carbon Steel
        "carbon_steel_polished": CARBON_STEEL_POLISHED,
        "carbon_steel_mill_scale": CARBON_STEEL_MILL_SCALE,
        "carbon_steel_rusted": CARBON_STEEL_RUSTED,
        "carbon_steel_heavily_rusted": CARBON_STEEL_HEAVILY_RUSTED,
        "carbon_steel_oxidized_high_temp": CARBON_STEEL_OXIDIZED_HIGH_TEMP,

        # Jacketing
        "asj_jacket": ASJ_ALL_SERVICE_JACKET,
        "pvc_jacket_white": PVC_JACKET_WHITE,
        "pvc_jacket_black": PVC_JACKET_BLACK,
        "aluminum_jacket_smooth": ALUMINUM_JACKET_SMOOTH,
        "aluminum_jacket_corrugated": ALUMINUM_JACKET_CORRUGATED,
        "aluminum_jacket_embossed": ALUMINUM_JACKET_EMBOSSED,
        "stainless_jacket_304": STAINLESS_JACKET_304,
        "stainless_jacket_316": STAINLESS_JACKET_316,

        # Paint
        "paint_white": PAINT_WHITE,
        "paint_black": PAINT_BLACK,
        "paint_aluminum": PAINT_ALUMINUM,
        "paint_high_temp_silicone": PAINT_HIGH_TEMP_SILICONE,
        "paint_zinc_rich": PAINT_ZINC_RICH,

        # Mastic
        "mastic_white": MASTIC_WHITE,
        "mastic_gray": MASTIC_GRAY,

        # Bare Insulation
        "fiberglass_bare": FIBERGLASS_BARE,
        "mineral_wool_bare": MINERAL_WOOL_BARE,
        "calcium_silicate_bare": CALCIUM_SILICATE_BARE,
    }

    @classmethod
    def get_spec(cls, surface_id: str) -> EmissivitySpec:
        """
        Get emissivity specification for a surface.

        Args:
            surface_id: Surface identifier

        Returns:
            EmissivitySpec for the surface

        Raises:
            KeyError: If surface not found
        """
        if surface_id not in cls._EMISSIVITIES:
            available = ", ".join(sorted(cls._EMISSIVITIES.keys()))
            raise KeyError(
                f"Unknown surface: {surface_id}. "
                f"Available surfaces: {available}"
            )
        return cls._EMISSIVITIES[surface_id]

    @classmethod
    def get_emissivity(cls, surface_id: str) -> float:
        """Get typical emissivity value for a surface."""
        return cls.get_spec(surface_id).emissivity

    @classmethod
    def list_surfaces(cls) -> List[str]:
        """Get list of all available surface IDs."""
        return sorted(cls._EMISSIVITIES.keys())


# =============================================================================
# CONVECTION CORRELATION COEFFICIENTS
# =============================================================================

@dataclass(frozen=True)
class ConvectionCorrelation:
    """
    Convection heat transfer correlation parameters.

    Attributes:
        name: Correlation name
        geometry: Applicable geometry (horizontal cylinder, vertical plate, etc.)
        c_coefficient: Leading coefficient C in Nu = C * Ra^n
        n_exponent: Exponent n in Nu = C * Ra^n
        ra_min: Minimum Rayleigh number for validity
        ra_max: Maximum Rayleigh number for validity
        reference: Literature reference
    """
    name: str
    geometry: str
    c_coefficient: float
    n_exponent: float
    ra_min: float
    ra_max: float
    reference: str


class ConvectionCorrelations:
    """
    Natural and forced convection correlations for heat transfer.

    Contains industry-standard correlations for calculating convection
    coefficients on pipes, flat surfaces, and complex geometries.

    References:
        - Churchill & Chu (1975): Horizontal cylinders
        - Morgan (1975): Horizontal cylinders (simplified)
        - McAdams (1954): Various geometries
        - ASHRAE Handbook - Fundamentals (2021)
    """

    # =========================================================================
    # NATURAL CONVECTION - HORIZONTAL CYLINDER
    # =========================================================================

    # Churchill-Chu correlation (recommended for horizontal cylinders)
    # Nu = {0.60 + 0.387*Ra^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27)}^2
    # Valid for Ra < 10^12
    CHURCHILL_CHU_HORIZONTAL = ConvectionCorrelation(
        name="Churchill-Chu (Horizontal Cylinder)",
        geometry="horizontal_cylinder",
        c_coefficient=0.387,
        n_exponent=1/6,
        ra_min=1e-6,
        ra_max=1e12,
        reference="Churchill & Chu, Int. J. Heat Mass Transfer, 1975"
    )

    # Morgan correlation (simplified, laminar regime)
    MORGAN_HORIZONTAL_LAMINAR = ConvectionCorrelation(
        name="Morgan (Horizontal Cylinder, Laminar)",
        geometry="horizontal_cylinder",
        c_coefficient=0.53,
        n_exponent=0.25,
        ra_min=1e4,
        ra_max=1e9,
        reference="Morgan, Advances in Heat Transfer, 1975"
    )

    # Morgan correlation (turbulent regime)
    MORGAN_HORIZONTAL_TURBULENT = ConvectionCorrelation(
        name="Morgan (Horizontal Cylinder, Turbulent)",
        geometry="horizontal_cylinder",
        c_coefficient=0.13,
        n_exponent=0.333,
        ra_min=1e9,
        ra_max=1e12,
        reference="Morgan, Advances in Heat Transfer, 1975"
    )

    # =========================================================================
    # NATURAL CONVECTION - VERTICAL CYLINDER/PLATE
    # =========================================================================

    MCADAMS_VERTICAL_LAMINAR = ConvectionCorrelation(
        name="McAdams (Vertical Surface, Laminar)",
        geometry="vertical_surface",
        c_coefficient=0.59,
        n_exponent=0.25,
        ra_min=1e4,
        ra_max=1e9,
        reference="McAdams, Heat Transmission, 1954"
    )

    MCADAMS_VERTICAL_TURBULENT = ConvectionCorrelation(
        name="McAdams (Vertical Surface, Turbulent)",
        geometry="vertical_surface",
        c_coefficient=0.10,
        n_exponent=0.333,
        ra_min=1e9,
        ra_max=1e13,
        reference="McAdams, Heat Transmission, 1954"
    )

    # Churchill-Chu for vertical plates
    CHURCHILL_CHU_VERTICAL = ConvectionCorrelation(
        name="Churchill-Chu (Vertical Surface)",
        geometry="vertical_surface",
        c_coefficient=0.387,
        n_exponent=1/6,
        ra_min=1e-1,
        ra_max=1e12,
        reference="Churchill & Chu, Int. J. Heat Mass Transfer, 1975"
    )

    # =========================================================================
    # NATURAL CONVECTION - HORIZONTAL FLAT SURFACE
    # =========================================================================

    MCADAMS_HORIZONTAL_UP_LAMINAR = ConvectionCorrelation(
        name="McAdams (Horizontal, Hot Side Up, Laminar)",
        geometry="horizontal_plate_up",
        c_coefficient=0.54,
        n_exponent=0.25,
        ra_min=1e4,
        ra_max=1e7,
        reference="McAdams, Heat Transmission, 1954"
    )

    MCADAMS_HORIZONTAL_UP_TURBULENT = ConvectionCorrelation(
        name="McAdams (Horizontal, Hot Side Up, Turbulent)",
        geometry="horizontal_plate_up",
        c_coefficient=0.15,
        n_exponent=0.333,
        ra_min=1e7,
        ra_max=1e11,
        reference="McAdams, Heat Transmission, 1954"
    )

    MCADAMS_HORIZONTAL_DOWN = ConvectionCorrelation(
        name="McAdams (Horizontal, Hot Side Down)",
        geometry="horizontal_plate_down",
        c_coefficient=0.27,
        n_exponent=0.25,
        ra_min=1e5,
        ra_max=1e11,
        reference="McAdams, Heat Transmission, 1954"
    )

    # =========================================================================
    # FORCED CONVECTION WIND FACTORS
    # =========================================================================

    # Wind speed correction factors for outdoor installations
    # h_forced = h_natural + k_wind * v^n

    @staticmethod
    def wind_correction_factor(wind_speed_m_s: float) -> float:
        """
        Calculate wind correction factor for forced convection.

        Based on ASTM C680 methodology for outdoor installations.

        Args:
            wind_speed_m_s: Wind speed in m/s

        Returns:
            Multiplier to add to natural convection coefficient

        Note:
            For still air (wind < 0.5 m/s), returns 1.0 (no correction)
        """
        if wind_speed_m_s < 0.5:
            return 1.0
        elif wind_speed_m_s < 2.0:
            # Light wind: linear interpolation
            return 1.0 + 0.5 * (wind_speed_m_s - 0.5)
        elif wind_speed_m_s < 5.0:
            # Moderate wind
            return 1.75 + 0.35 * (wind_speed_m_s - 2.0)
        elif wind_speed_m_s < 10.0:
            # Strong wind
            return 2.8 + 0.25 * (wind_speed_m_s - 5.0)
        else:
            # Very strong wind (capped)
            return 4.05 + 0.15 * (wind_speed_m_s - 10.0)

    @staticmethod
    def forced_convection_coefficient(
        wind_speed_m_s: float,
        outer_diameter_m: float
    ) -> float:
        """
        Calculate forced convection coefficient for external flow.

        Uses simplified Hilpert correlation for cross-flow over cylinder.
        h = 0.3 + (0.62 * Re^0.5 * Pr^0.33) * (k/D)

        Args:
            wind_speed_m_s: Wind velocity perpendicular to pipe
            outer_diameter_m: Outer diameter of insulated pipe

        Returns:
            Convection coefficient in W/(m^2*K)
        """
        # Air properties at 25C (typical outdoor)
        k_air = 0.0262  # W/(m*K)
        nu_air = 1.568e-5  # m^2/s (kinematic viscosity)
        pr_air = 0.708  # Prandtl number

        # Reynolds number
        re = wind_speed_m_s * outer_diameter_m / nu_air

        if re < 1:
            # Very low flow, use natural convection
            return 5.0  # Approximate still air value

        # Hilpert correlation coefficients based on Re
        if re < 4:
            c, m = 0.989, 0.330
        elif re < 40:
            c, m = 0.911, 0.385
        elif re < 4000:
            c, m = 0.683, 0.466
        elif re < 40000:
            c, m = 0.193, 0.618
        else:
            c, m = 0.0266, 0.805

        # Nusselt number
        nu = c * (re ** m) * (pr_air ** 0.33)

        # Convection coefficient
        h = nu * k_air / outer_diameter_m

        return h


# =============================================================================
# ECONOMIC PARAMETERS
# =============================================================================

@dataclass
class FuelPrice:
    """
    Fuel price specification with regional data.

    Attributes:
        fuel_type: Type of fuel
        price_per_unit: Price per unit
        unit: Unit of measurement
        heating_value: Energy content per unit
        heating_value_unit: Unit for heating value
        region: Geographic region
        currency: Currency code
        effective_date: Date price was effective
    """
    fuel_type: str
    price_per_unit: float
    unit: str
    heating_value: float
    heating_value_unit: str
    region: str
    currency: str
    effective_date: str


class EconomicParameters:
    """
    Economic parameters for insulation ROI calculations.

    Contains fuel prices, carbon emission factors, labor rates,
    and material costs for economic analysis.

    Data Sources:
        - EIA (U.S. Energy Information Administration)
        - EPA GHG Emission Factors Hub
        - RSMeans Data (labor rates)
        - Industry surveys (material costs)
    """

    # =========================================================================
    # FUEL PRICES (2024 US NATIONAL AVERAGE)
    # =========================================================================

    FUEL_PRICES_US_2024: Dict[str, FuelPrice] = {
        "natural_gas_commercial": FuelPrice(
            fuel_type="Natural Gas (Commercial)",
            price_per_unit=10.50,
            unit="$/MMBtu",
            heating_value=1.0,
            heating_value_unit="MMBtu/MMBtu",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
        "natural_gas_industrial": FuelPrice(
            fuel_type="Natural Gas (Industrial)",
            price_per_unit=6.50,
            unit="$/MMBtu",
            heating_value=1.0,
            heating_value_unit="MMBtu/MMBtu",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
        "fuel_oil_no2": FuelPrice(
            fuel_type="Fuel Oil #2",
            price_per_unit=3.50,
            unit="$/gallon",
            heating_value=0.138,
            heating_value_unit="MMBtu/gallon",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
        "fuel_oil_no6": FuelPrice(
            fuel_type="Fuel Oil #6 (Bunker)",
            price_per_unit=2.80,
            unit="$/gallon",
            heating_value=0.150,
            heating_value_unit="MMBtu/gallon",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
        "propane": FuelPrice(
            fuel_type="Propane (LPG)",
            price_per_unit=1.80,
            unit="$/gallon",
            heating_value=0.0915,
            heating_value_unit="MMBtu/gallon",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
        "electricity_commercial": FuelPrice(
            fuel_type="Electricity (Commercial)",
            price_per_unit=0.13,
            unit="$/kWh",
            heating_value=0.003412,
            heating_value_unit="MMBtu/kWh",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
        "electricity_industrial": FuelPrice(
            fuel_type="Electricity (Industrial)",
            price_per_unit=0.085,
            unit="$/kWh",
            heating_value=0.003412,
            heating_value_unit="MMBtu/kWh",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
        "steam_purchased": FuelPrice(
            fuel_type="Purchased Steam",
            price_per_unit=15.00,
            unit="$/Mlb",
            heating_value=1.0,
            heating_value_unit="MMBtu/Mlb",
            region="US National Average",
            currency="USD",
            effective_date="2024-01-01"
        ),
    }

    # Regional fuel price multipliers
    REGIONAL_MULTIPLIERS_US: Dict[str, Dict[str, float]] = {
        "northeast": {
            "natural_gas": 1.25,
            "electricity": 1.40,
            "fuel_oil": 1.10,
        },
        "midwest": {
            "natural_gas": 0.90,
            "electricity": 0.95,
            "fuel_oil": 1.00,
        },
        "south": {
            "natural_gas": 0.95,
            "electricity": 0.90,
            "fuel_oil": 1.05,
        },
        "west": {
            "natural_gas": 1.15,
            "electricity": 1.20,
            "fuel_oil": 1.15,
        },
        "california": {
            "natural_gas": 1.35,
            "electricity": 1.60,
            "fuel_oil": 1.20,
        },
    }

    # =========================================================================
    # CARBON EMISSION FACTORS
    # =========================================================================

    # EPA GHG Emission Factors Hub (2024)
    CARBON_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
        "natural_gas": {
            "co2_kg_per_mmbtu": 53.06,
            "ch4_kg_per_mmbtu": 0.001,
            "n2o_kg_per_mmbtu": 0.0001,
            "co2e_kg_per_mmbtu": 53.11,
            "source": "EPA GHG Emission Factors Hub 2024"
        },
        "fuel_oil_no2": {
            "co2_kg_per_mmbtu": 73.96,
            "ch4_kg_per_mmbtu": 0.003,
            "n2o_kg_per_mmbtu": 0.0006,
            "co2e_kg_per_mmbtu": 74.21,
            "source": "EPA GHG Emission Factors Hub 2024"
        },
        "fuel_oil_no6": {
            "co2_kg_per_mmbtu": 75.10,
            "ch4_kg_per_mmbtu": 0.003,
            "n2o_kg_per_mmbtu": 0.0006,
            "co2e_kg_per_mmbtu": 75.35,
            "source": "EPA GHG Emission Factors Hub 2024"
        },
        "propane": {
            "co2_kg_per_mmbtu": 62.87,
            "ch4_kg_per_mmbtu": 0.003,
            "n2o_kg_per_mmbtu": 0.0006,
            "co2e_kg_per_mmbtu": 63.12,
            "source": "EPA GHG Emission Factors Hub 2024"
        },
        "electricity_us_avg": {
            "co2_kg_per_kwh": 0.386,
            "ch4_kg_per_kwh": 0.00004,
            "n2o_kg_per_kwh": 0.000005,
            "co2e_kg_per_kwh": 0.389,
            "source": "EPA eGRID 2022"
        },
        "coal_bituminous": {
            "co2_kg_per_mmbtu": 93.28,
            "ch4_kg_per_mmbtu": 0.011,
            "n2o_kg_per_mmbtu": 0.0016,
            "co2e_kg_per_mmbtu": 94.12,
            "source": "EPA GHG Emission Factors Hub 2024"
        },
    }

    # Social cost of carbon ($/tonne CO2)
    CARBON_PRICES: Dict[str, Dict[str, float]] = {
        "epa_social_cost_2024": {
            "low": 51.0,
            "central": 51.0,  # EPA 2024 estimate
            "high": 51.0,
            "source": "EPA Technical Support Document 2024"
        },
        "eu_ets_2024": {
            "low": 60.0,
            "central": 85.0,
            "high": 100.0,
            "source": "EU ETS Market Price 2024"
        },
        "internal_carbon_price": {
            "low": 50.0,
            "central": 100.0,
            "high": 200.0,
            "source": "Corporate internal pricing range"
        },
    }

    # =========================================================================
    # LABOR RATES (RSMeans 2024)
    # =========================================================================

    # Bare labor rates ($/hour) by craft
    LABOR_RATES_2024: Dict[str, Dict[str, float]] = {
        "insulator_journeyman": {
            "bare_rate": 52.00,
            "fringes": 28.50,
            "total_rate": 80.50,
            "productivity_factor": 1.0
        },
        "insulator_apprentice": {
            "bare_rate": 35.00,
            "fringes": 18.00,
            "total_rate": 53.00,
            "productivity_factor": 0.65
        },
        "insulator_foreman": {
            "bare_rate": 58.00,
            "fringes": 31.00,
            "total_rate": 89.00,
            "productivity_factor": 0.5  # Supervisory
        },
        "sheet_metal_worker": {
            "bare_rate": 55.00,
            "fringes": 30.00,
            "total_rate": 85.00,
            "productivity_factor": 1.0
        },
        "laborer": {
            "bare_rate": 28.00,
            "fringes": 15.00,
            "total_rate": 43.00,
            "productivity_factor": 0.5
        },
    }

    # Labor productivity factors by condition
    PRODUCTIVITY_FACTORS: Dict[str, float] = {
        "ideal_conditions": 1.00,
        "good_conditions": 0.95,
        "average_conditions": 0.85,
        "poor_conditions": 0.70,
        "confined_space": 0.60,
        "elevated_work": 0.75,
        "hot_work_conditions": 0.65,
        "cold_weather": 0.80,
        "overtime_first_4hrs": 0.90,
        "overtime_after_4hrs": 0.75,
        "shift_differential_night": 0.85,
    }

    # =========================================================================
    # MATERIAL COSTS (2024 US MARKET)
    # =========================================================================

    # Insulation material costs ($/board foot or $/linear foot)
    MATERIAL_COSTS_2024: Dict[str, Dict[str, float]] = {
        "mineral_wool_pipe": {
            "1_inch_thick": 3.50,
            "1.5_inch_thick": 4.25,
            "2_inch_thick": 5.00,
            "3_inch_thick": 7.50,
            "unit": "$/linear_foot",
            "nominal_size": "2_inch_pipe"
        },
        "calcium_silicate_pipe": {
            "1_inch_thick": 8.00,
            "1.5_inch_thick": 10.50,
            "2_inch_thick": 13.00,
            "3_inch_thick": 18.00,
            "unit": "$/linear_foot",
            "nominal_size": "2_inch_pipe"
        },
        "cellular_glass_pipe": {
            "1_inch_thick": 12.00,
            "1.5_inch_thick": 15.00,
            "2_inch_thick": 19.00,
            "3_inch_thick": 27.00,
            "unit": "$/linear_foot",
            "nominal_size": "2_inch_pipe"
        },
        "aerogel_blanket": {
            "5mm_thick": 35.00,
            "10mm_thick": 65.00,
            "unit": "$/sq_foot"
        },
        "aluminum_jacket": {
            "0.016_inch": 2.50,
            "0.020_inch": 2.85,
            "0.024_inch": 3.25,
            "0.032_inch": 4.00,
            "unit": "$/sq_foot"
        },
        "stainless_jacket_304": {
            "0.010_inch": 6.50,
            "0.016_inch": 8.00,
            "0.020_inch": 9.50,
            "unit": "$/sq_foot"
        },
        "pvc_jacket": {
            "20_mil": 1.25,
            "30_mil": 1.65,
            "40_mil": 2.00,
            "unit": "$/sq_foot"
        },
    }

    # =========================================================================
    # BOILER EFFICIENCY VALUES
    # =========================================================================

    BOILER_EFFICIENCY: Dict[str, Dict[str, float]] = {
        "natural_gas_standard": {
            "efficiency": 0.80,
            "description": "Standard gas-fired boiler"
        },
        "natural_gas_high_eff": {
            "efficiency": 0.88,
            "description": "High-efficiency condensing boiler"
        },
        "natural_gas_premium": {
            "efficiency": 0.95,
            "description": "Premium condensing boiler"
        },
        "fuel_oil_standard": {
            "efficiency": 0.78,
            "description": "Standard oil-fired boiler"
        },
        "fuel_oil_high_eff": {
            "efficiency": 0.85,
            "description": "High-efficiency oil boiler"
        },
        "electric_resistance": {
            "efficiency": 0.98,
            "description": "Electric resistance heater"
        },
        "steam_generator": {
            "efficiency": 0.82,
            "description": "Steam generator (gas-fired)"
        },
    }


# =============================================================================
# SAFETY LIMITS
# =============================================================================

@dataclass(frozen=True)
class SafetyLimit:
    """
    Safety limit specification.

    Attributes:
        name: Description of the limit
        value: Numerical limit value
        unit: Unit of measurement
        limit_type: Type (max, min, recommended)
        standard: Reference standard
        notes: Additional notes
    """
    name: str
    value: float
    unit: str
    limit_type: str  # "max", "min", "recommended"
    standard: str
    notes: str = ""


class SafetyLimits:
    """
    Safety limits for personnel protection and equipment operation.

    Based on ASTM C1055, OSHA regulations, and industry practices.
    All values assume brief contact (5 seconds or less).
    """

    # =========================================================================
    # PERSONNEL PROTECTION LIMITS (ASTM C1055-03)
    # =========================================================================

    # Maximum surface temperature for personnel protection
    # Based on 5-second contact time, ungloved hand

    PERSONNEL_PROTECTION_MOMENTARY = SafetyLimit(
        name="Personnel Protection - Momentary Contact",
        value=60.0,
        unit="C",
        limit_type="max",
        standard="ASTM C1055-03",
        notes="5-second contact, ungloved hand"
    )

    PERSONNEL_PROTECTION_MOMENTARY_F = SafetyLimit(
        name="Personnel Protection - Momentary Contact",
        value=140.0,
        unit="F",
        limit_type="max",
        standard="ASTM C1055-03",
        notes="5-second contact, ungloved hand"
    )

    PERSONNEL_PROTECTION_PROLONGED = SafetyLimit(
        name="Personnel Protection - Prolonged Contact",
        value=49.0,
        unit="C",
        limit_type="max",
        standard="ASTM C1055-03",
        notes="Continuous contact possible"
    )

    PERSONNEL_PROTECTION_PROLONGED_F = SafetyLimit(
        name="Personnel Protection - Prolonged Contact",
        value=120.0,
        unit="F",
        limit_type="max",
        standard="ASTM C1055-03",
        notes="Continuous contact possible"
    )

    # =========================================================================
    # OSHA HEAT EXPOSURE LIMITS
    # =========================================================================

    OSHA_IMMEDIATE_DANGER = SafetyLimit(
        name="OSHA Immediate Danger Level",
        value=71.0,
        unit="C",
        limit_type="max",
        standard="OSHA 29 CFR 1910",
        notes="Burns occur with brief contact"
    )

    OSHA_DANGER_ZONE = SafetyLimit(
        name="OSHA Danger Zone",
        value=60.0,
        unit="C",
        limit_type="max",
        standard="OSHA 29 CFR 1910",
        notes="Potential for injury"
    )

    OSHA_CAUTION_ZONE = SafetyLimit(
        name="OSHA Caution Zone",
        value=49.0,
        unit="C",
        limit_type="max",
        standard="OSHA 29 CFR 1910",
        notes="Discomfort with prolonged contact"
    )

    # =========================================================================
    # FIRE HAZARD TEMPERATURES
    # =========================================================================

    FIRE_COMBUSTIBLE_AUTOIGNITION = SafetyLimit(
        name="Combustible Ignition Risk",
        value=204.0,
        unit="C",
        limit_type="max",
        standard="NFPA 30",
        notes="Autoignition temp for many combustibles"
    )

    FIRE_PAPER_IGNITION = SafetyLimit(
        name="Paper/Cardboard Ignition",
        value=232.0,
        unit="C",
        limit_type="max",
        standard="NFPA 30",
        notes="Paper autoignition temperature"
    )

    FIRE_WOOD_IGNITION = SafetyLimit(
        name="Wood Ignition",
        value=260.0,
        unit="C",
        limit_type="max",
        standard="NFPA 30",
        notes="Wood autoignition temperature"
    )

    FIRE_OIL_FLASH_POINT = SafetyLimit(
        name="Hydraulic Oil Flash Point",
        value=150.0,
        unit="C",
        limit_type="max",
        standard="NFPA 30",
        notes="Typical mineral oil flash point"
    )

    # =========================================================================
    # CONDENSATION PREVENTION
    # =========================================================================

    CONDENSATION_MINIMUM = SafetyLimit(
        name="Condensation Prevention",
        value=3.0,
        unit="C above dew point",
        limit_type="min",
        standard="ASHRAE Handbook",
        notes="Minimum margin above dew point"
    )

    # =========================================================================
    # EQUIPMENT LIMITS
    # =========================================================================

    ASJ_JACKET_MAX_TEMP = SafetyLimit(
        name="ASJ Jacket Maximum Temperature",
        value=80.0,
        unit="C",
        limit_type="max",
        standard="ASTM C1136",
        notes="All-Service Jacket temperature limit"
    )

    PVC_JACKET_MAX_TEMP = SafetyLimit(
        name="PVC Jacket Maximum Temperature",
        value=65.0,
        unit="C",
        limit_type="max",
        standard="ASTM C1136",
        notes="PVC jacket temperature limit"
    )

    ALUMINUM_JACKET_MAX_TEMP = SafetyLimit(
        name="Aluminum Jacket Maximum Temperature",
        value=200.0,
        unit="C",
        limit_type="max",
        standard="Industry Practice",
        notes="Aluminum softening consideration"
    )

    STAINLESS_JACKET_MAX_TEMP = SafetyLimit(
        name="Stainless Steel Jacket Maximum Temperature",
        value=500.0,
        unit="C",
        limit_type="max",
        standard="Industry Practice",
        notes="Stainless steel service limit"
    )

    # =========================================================================
    # LOOKUP DICTIONARY
    # =========================================================================

    _LIMITS: Dict[str, SafetyLimit] = {
        "personnel_protection_momentary_c": PERSONNEL_PROTECTION_MOMENTARY,
        "personnel_protection_momentary_f": PERSONNEL_PROTECTION_MOMENTARY_F,
        "personnel_protection_prolonged_c": PERSONNEL_PROTECTION_PROLONGED,
        "personnel_protection_prolonged_f": PERSONNEL_PROTECTION_PROLONGED_F,
        "osha_immediate_danger": OSHA_IMMEDIATE_DANGER,
        "osha_danger_zone": OSHA_DANGER_ZONE,
        "osha_caution_zone": OSHA_CAUTION_ZONE,
        "fire_combustible": FIRE_COMBUSTIBLE_AUTOIGNITION,
        "fire_paper": FIRE_PAPER_IGNITION,
        "fire_wood": FIRE_WOOD_IGNITION,
        "fire_oil_flash": FIRE_OIL_FLASH_POINT,
        "condensation_min": CONDENSATION_MINIMUM,
        "asj_max_temp": ASJ_JACKET_MAX_TEMP,
        "pvc_max_temp": PVC_JACKET_MAX_TEMP,
        "aluminum_jacket_max": ALUMINUM_JACKET_MAX_TEMP,
        "stainless_jacket_max": STAINLESS_JACKET_MAX_TEMP,
    }

    @classmethod
    def get_limit(cls, limit_id: str) -> SafetyLimit:
        """Get safety limit by ID."""
        if limit_id not in cls._LIMITS:
            raise KeyError(f"Unknown safety limit: {limit_id}")
        return cls._LIMITS[limit_id]

    @classmethod
    def get_personnel_protection_temp(cls, contact_type: str = "momentary") -> float:
        """
        Get personnel protection temperature limit.

        Args:
            contact_type: "momentary" (5 sec) or "prolonged"

        Returns:
            Maximum surface temperature in Celsius
        """
        if contact_type == "momentary":
            return cls.PERSONNEL_PROTECTION_MOMENTARY.value
        elif contact_type == "prolonged":
            return cls.PERSONNEL_PROTECTION_PROLONGED.value
        else:
            raise ValueError(f"Unknown contact type: {contact_type}")

    @classmethod
    def check_personnel_safety(
        cls,
        surface_temp_c: float,
        contact_type: str = "momentary"
    ) -> Tuple[bool, str]:
        """
        Check if surface temperature is safe for personnel.

        Args:
            surface_temp_c: Surface temperature in Celsius
            contact_type: "momentary" or "prolonged"

        Returns:
            Tuple of (is_safe: bool, message: str)
        """
        limit = cls.get_personnel_protection_temp(contact_type)

        if surface_temp_c <= limit:
            return True, f"Surface temperature {surface_temp_c}C is safe (limit: {limit}C)"
        else:
            excess = surface_temp_c - limit
            return False, f"Surface temperature {surface_temp_c}C exceeds limit by {excess:.1f}C"


# =============================================================================
# PIPE DIMENSIONS (NPS SCHEDULE)
# =============================================================================

@dataclass(frozen=True)
class PipeDimension:
    """
    Pipe dimension specification per ASME B36.10M.

    Attributes:
        nps: Nominal Pipe Size
        schedule: Pipe schedule (40, 80, etc.)
        outer_diameter_mm: Outer diameter in mm
        wall_thickness_mm: Wall thickness in mm
        inner_diameter_mm: Inner diameter in mm
    """
    nps: str
    schedule: str
    outer_diameter_mm: float
    wall_thickness_mm: float
    inner_diameter_mm: float


class PipeDimensions:
    """
    Standard pipe dimensions per ASME B36.10M.

    Contains dimensions for common NPS sizes from 1/2" to 36"
    for Schedule 40 and Schedule 80.
    """

    # Schedule 40 Pipe Dimensions (mm)
    SCHEDULE_40: Dict[str, PipeDimension] = {
        "0.5": PipeDimension("0.5", "40", 21.3, 2.77, 15.80),
        "0.75": PipeDimension("0.75", "40", 26.7, 2.87, 20.93),
        "1": PipeDimension("1", "40", 33.4, 3.38, 26.64),
        "1.25": PipeDimension("1.25", "40", 42.2, 3.56, 35.05),
        "1.5": PipeDimension("1.5", "40", 48.3, 3.68, 40.89),
        "2": PipeDimension("2", "40", 60.3, 3.91, 52.50),
        "2.5": PipeDimension("2.5", "40", 73.0, 5.16, 62.71),
        "3": PipeDimension("3", "40", 88.9, 5.49, 77.93),
        "3.5": PipeDimension("3.5", "40", 101.6, 5.74, 90.12),
        "4": PipeDimension("4", "40", 114.3, 6.02, 102.26),
        "5": PipeDimension("5", "40", 141.3, 6.55, 128.20),
        "6": PipeDimension("6", "40", 168.3, 7.11, 154.08),
        "8": PipeDimension("8", "40", 219.1, 8.18, 202.74),
        "10": PipeDimension("10", "40", 273.0, 9.27, 254.46),
        "12": PipeDimension("12", "40", 323.8, 10.31, 303.18),
        "14": PipeDimension("14", "40", 355.6, 11.13, 333.34),
        "16": PipeDimension("16", "40", 406.4, 12.70, 381.00),
        "18": PipeDimension("18", "40", 457.2, 14.27, 428.66),
        "20": PipeDimension("20", "40", 508.0, 15.09, 477.82),
        "24": PipeDimension("24", "40", 609.6, 17.48, 574.64),
        "30": PipeDimension("30", "40", 762.0, 19.05, 723.90),
        "36": PipeDimension("36", "40", 914.4, 19.05, 876.30),
    }

    # Schedule 80 Pipe Dimensions (mm)
    SCHEDULE_80: Dict[str, PipeDimension] = {
        "0.5": PipeDimension("0.5", "80", 21.3, 3.73, 13.87),
        "0.75": PipeDimension("0.75", "80", 26.7, 3.91, 18.85),
        "1": PipeDimension("1", "80", 33.4, 4.55, 24.31),
        "1.25": PipeDimension("1.25", "80", 42.2, 4.85, 32.46),
        "1.5": PipeDimension("1.5", "80", 48.3, 5.08, 38.10),
        "2": PipeDimension("2", "80", 60.3, 5.54, 49.25),
        "2.5": PipeDimension("2.5", "80", 73.0, 7.01, 59.00),
        "3": PipeDimension("3", "80", 88.9, 7.62, 73.66),
        "3.5": PipeDimension("3.5", "80", 101.6, 8.08, 85.44),
        "4": PipeDimension("4", "80", 114.3, 8.56, 97.18),
        "5": PipeDimension("5", "80", 141.3, 9.53, 122.24),
        "6": PipeDimension("6", "80", 168.3, 10.97, 146.36),
        "8": PipeDimension("8", "80", 219.1, 12.70, 193.70),
        "10": PipeDimension("10", "80", 273.0, 15.09, 242.82),
        "12": PipeDimension("12", "80", 323.8, 17.48, 288.84),
        "14": PipeDimension("14", "80", 355.6, 19.05, 317.50),
        "16": PipeDimension("16", "80", 406.4, 21.44, 363.52),
        "18": PipeDimension("18", "80", 457.2, 23.83, 409.54),
        "20": PipeDimension("20", "80", 508.0, 26.19, 455.62),
        "24": PipeDimension("24", "80", 609.6, 30.96, 547.68),
    }

    @classmethod
    def get_dimension(cls, nps: str, schedule: str = "40") -> PipeDimension:
        """
        Get pipe dimensions for given NPS and schedule.

        Args:
            nps: Nominal pipe size (e.g., "2", "4", "6")
            schedule: Pipe schedule ("40" or "80")

        Returns:
            PipeDimension object

        Raises:
            KeyError: If NPS/schedule combination not found
        """
        schedule_data = cls.SCHEDULE_40 if schedule == "40" else cls.SCHEDULE_80
        if nps not in schedule_data:
            raise KeyError(f"NPS {nps} Schedule {schedule} not found")
        return schedule_data[nps]

    @classmethod
    def get_outer_diameter_m(cls, nps: str, schedule: str = "40") -> float:
        """Get outer diameter in meters."""
        dim = cls.get_dimension(nps, schedule)
        return dim.outer_diameter_mm / 1000.0


# =============================================================================
# STANDARD INSULATION THICKNESSES
# =============================================================================

class StandardThicknesses:
    """
    Standard insulation thickness values per ASTM standards.

    Contains standard thickness increments for pipe insulation
    and flat surface insulation.
    """

    # Standard pipe insulation thicknesses (inches)
    PIPE_THICKNESSES_INCH: Tuple[float, ...] = (
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0
    )

    # Standard pipe insulation thicknesses (mm)
    PIPE_THICKNESSES_MM: Tuple[float, ...] = (
        13, 25, 38, 51, 64, 76, 89, 102, 114, 127, 140, 152
    )

    # Standard board insulation thicknesses (inches)
    BOARD_THICKNESSES_INCH: Tuple[float, ...] = (
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0
    )

    @classmethod
    def get_nearest_standard_thickness_mm(cls, thickness_mm: float) -> float:
        """
        Find nearest standard thickness.

        Args:
            thickness_mm: Calculated thickness in mm

        Returns:
            Nearest standard thickness (rounded up)
        """
        for std_thick in cls.PIPE_THICKNESSES_MM:
            if std_thick >= thickness_mm:
                return std_thick
        return cls.PIPE_THICKNESSES_MM[-1]

    @classmethod
    def get_nearest_standard_thickness_inch(cls, thickness_inch: float) -> float:
        """
        Find nearest standard thickness in inches.

        Args:
            thickness_inch: Calculated thickness in inches

        Returns:
            Nearest standard thickness (rounded up)
        """
        for std_thick in cls.PIPE_THICKNESSES_INCH:
            if std_thick >= thickness_inch:
                return std_thick
        return cls.PIPE_THICKNESSES_INCH[-1]


# =============================================================================
# OPERATING CONDITIONS PRESETS
# =============================================================================

@dataclass
class OperatingCondition:
    """
    Preset operating condition specification.

    Attributes:
        name: Description of condition
        ambient_temp_c: Ambient temperature (Celsius)
        wind_speed_m_s: Wind speed (m/s)
        relative_humidity: Relative humidity (%)
        indoor_outdoor: "indoor" or "outdoor"
    """
    name: str
    ambient_temp_c: float
    wind_speed_m_s: float
    relative_humidity: float
    indoor_outdoor: str


class OperatingConditions:
    """
    Preset operating conditions for common scenarios.
    """

    PRESETS: Dict[str, OperatingCondition] = {
        "indoor_standard": OperatingCondition(
            name="Indoor Standard",
            ambient_temp_c=21.0,
            wind_speed_m_s=0.0,
            relative_humidity=50.0,
            indoor_outdoor="indoor"
        ),
        "indoor_mechanical_room": OperatingCondition(
            name="Indoor Mechanical Room",
            ambient_temp_c=27.0,
            wind_speed_m_s=0.5,
            relative_humidity=45.0,
            indoor_outdoor="indoor"
        ),
        "outdoor_summer": OperatingCondition(
            name="Outdoor Summer",
            ambient_temp_c=35.0,
            wind_speed_m_s=2.0,
            relative_humidity=60.0,
            indoor_outdoor="outdoor"
        ),
        "outdoor_winter": OperatingCondition(
            name="Outdoor Winter",
            ambient_temp_c=-10.0,
            wind_speed_m_s=5.0,
            relative_humidity=70.0,
            indoor_outdoor="outdoor"
        ),
        "outdoor_mild": OperatingCondition(
            name="Outdoor Mild",
            ambient_temp_c=21.0,
            wind_speed_m_s=3.0,
            relative_humidity=55.0,
            indoor_outdoor="outdoor"
        ),
        "outdoor_tropical": OperatingCondition(
            name="Outdoor Tropical",
            ambient_temp_c=32.0,
            wind_speed_m_s=1.5,
            relative_humidity=85.0,
            indoor_outdoor="outdoor"
        ),
        "outdoor_arctic": OperatingCondition(
            name="Outdoor Arctic",
            ambient_temp_c=-40.0,
            wind_speed_m_s=8.0,
            relative_humidity=60.0,
            indoor_outdoor="outdoor"
        ),
        "outdoor_desert": OperatingCondition(
            name="Outdoor Desert",
            ambient_temp_c=45.0,
            wind_speed_m_s=4.0,
            relative_humidity=15.0,
            indoor_outdoor="outdoor"
        ),
    }

    @classmethod
    def get_condition(cls, preset_name: str) -> OperatingCondition:
        """Get preset operating condition."""
        if preset_name not in cls.PRESETS:
            raise KeyError(f"Unknown preset: {preset_name}")
        return cls.PRESETS[preset_name]


# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Engineering Team"
__license__ = "Apache 2.0"

# Module-level exports
__all__ = [
    # Physical constants
    "PhysicalConstants",
    "AirPropertiesTable",

    # Insulation materials
    "InsulationType",
    "InsulationMaterialSpec",
    "InsulationMaterials",

    # Surface properties
    "EmissivitySpec",
    "SurfaceEmissivity",

    # Convection
    "ConvectionCorrelation",
    "ConvectionCorrelations",

    # Economic
    "FuelPrice",
    "EconomicParameters",

    # Safety
    "SafetyLimit",
    "SafetyLimits",

    # Pipe dimensions
    "PipeDimension",
    "PipeDimensions",

    # Thicknesses
    "StandardThicknesses",

    # Operating conditions
    "OperatingCondition",
    "OperatingConditions",
]
