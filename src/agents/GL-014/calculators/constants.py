"""
Heat Exchanger Constants Library for GL-014 EXCHANGER-PRO

This module provides comprehensive lookup tables for heat exchanger
calculations including fouling factors, heat transfer correlations,
material properties, fluid properties, and TEMA standards.

All values are industry-standard references from:
- TEMA (Tubular Exchanger Manufacturers Association) Standards
- GPSA Engineering Data Book
- Perry's Chemical Engineers' Handbook
- ASME Standards

Example:
    >>> from constants import FoulingFactors, MaterialProperties
    >>> rf = FoulingFactors.get_fouling_factor("cooling_water", "treated")
    >>> k = MaterialProperties.get_thermal_conductivity("carbon_steel", 100)
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FOULING FACTORS (TEMA STANDARDS)
# =============================================================================
# Units: m2.K/W (SI) - multiply by 5.678 for hr.ft2.F/BTU

class FoulingCategory(Enum):
    """Categories of fouling in heat exchangers."""
    COOLING_WATER = "cooling_water"
    PROCESS_FLUID = "process_fluid"
    STEAM = "steam"
    REFRIGERANT = "refrigerant"
    GAS = "gas"
    OIL = "oil"


@dataclass
class FoulingFactorData:
    """Data class for fouling factor information."""
    value_si: float  # m2.K/W
    value_imperial: float  # hr.ft2.F/BTU
    min_value: float  # Minimum expected value (SI)
    max_value: float  # Maximum expected value (SI)
    description: str
    cleaning_frequency_days: int
    notes: str = ""


class FoulingFactors:
    """
    TEMA Standard Fouling Factors.

    Provides fouling resistance values for various fluid types.
    All values based on TEMA 9th Edition standards.
    """

    # Conversion factor: SI to Imperial
    SI_TO_IMPERIAL = 5.678263  # m2.K/W to hr.ft2.F/BTU

    # Cooling Water Fouling Factors
    COOLING_WATER: Dict[str, FoulingFactorData] = {
        "treated_below_52C": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Treated cooling water below 52C (125F)",
            cleaning_frequency_days=365,
            notes="Chromate or phosphate treatment"
        ),
        "treated_above_52C": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Treated cooling water above 52C (125F)",
            cleaning_frequency_days=180,
            notes="Higher fouling due to scale formation"
        ),
        "untreated_below_52C": FoulingFactorData(
            value_si=0.000528,
            value_imperial=0.003,
            min_value=0.000352,
            max_value=0.000704,
            description="Untreated cooling water below 52C",
            cleaning_frequency_days=90,
            notes="Municipal or well water"
        ),
        "untreated_above_52C": FoulingFactorData(
            value_si=0.000880,
            value_imperial=0.005,
            min_value=0.000528,
            max_value=0.001232,
            description="Untreated cooling water above 52C",
            cleaning_frequency_days=60,
            notes="High scaling potential"
        ),
        "sea_water_below_52C": FoulingFactorData(
            value_si=0.000088,
            value_imperial=0.0005,
            min_value=0.000044,
            max_value=0.000176,
            description="Sea water below 52C (125F)",
            cleaning_frequency_days=180,
            notes="Clean ocean water"
        ),
        "sea_water_above_52C": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000352,
            description="Sea water above 52C (125F)",
            cleaning_frequency_days=90,
            notes="Biological fouling increases"
        ),
        "brackish_water": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Brackish or estuary water",
            cleaning_frequency_days=120,
            notes="Variable salinity"
        ),
        "river_water_clean": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Clean river water",
            cleaning_frequency_days=120,
            notes="Low suspended solids"
        ),
        "river_water_muddy": FoulingFactorData(
            value_si=0.000528,
            value_imperial=0.003,
            min_value=0.000352,
            max_value=0.000880,
            description="Muddy or silty river water",
            cleaning_frequency_days=60,
            notes="High suspended solids"
        ),
        "cooling_tower_treated": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000352,
            description="Treated cooling tower water",
            cleaning_frequency_days=180,
            notes="With biocide and scale inhibitor"
        ),
        "cooling_tower_untreated": FoulingFactorData(
            value_si=0.000528,
            value_imperial=0.003,
            min_value=0.000352,
            max_value=0.000704,
            description="Untreated cooling tower water",
            cleaning_frequency_days=60,
            notes="Biological growth risk"
        ),
    }

    # Process Fluid Fouling Factors
    PROCESS_FLUIDS: Dict[str, FoulingFactorData] = {
        "light_hydrocarbons_clean": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Clean light hydrocarbons (C1-C4)",
            cleaning_frequency_days=730,
            notes="Methane, ethane, propane, butane"
        ),
        "light_hydrocarbons_with_traces": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Light hydrocarbons with trace contaminants",
            cleaning_frequency_days=365,
            notes="Trace water or sulfur compounds"
        ),
        "gasoline": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Gasoline and naphtha",
            cleaning_frequency_days=365,
            notes="C5-C12 range"
        ),
        "kerosene_jet_fuel": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Kerosene and jet fuel",
            cleaning_frequency_days=365,
            notes="C12-C16 range"
        ),
        "diesel_fuel": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Diesel fuel",
            cleaning_frequency_days=365,
            notes="C16-C20 range"
        ),
        "gas_oil": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000264,
            max_value=0.000528,
            description="Gas oil",
            cleaning_frequency_days=270,
            notes="C20-C35 range"
        ),
        "heavy_fuel_oil": FoulingFactorData(
            value_si=0.000880,
            value_imperial=0.005,
            min_value=0.000528,
            max_value=0.001232,
            description="Heavy fuel oil (No. 6)",
            cleaning_frequency_days=120,
            notes="High asphaltene content"
        ),
        "crude_oil_light": FoulingFactorData(
            value_si=0.000528,
            value_imperial=0.003,
            min_value=0.000352,
            max_value=0.000704,
            description="Light crude oil (API > 31)",
            cleaning_frequency_days=180,
            notes="Low viscosity crude"
        ),
        "crude_oil_medium": FoulingFactorData(
            value_si=0.000704,
            value_imperial=0.004,
            min_value=0.000528,
            max_value=0.000880,
            description="Medium crude oil (22 < API < 31)",
            cleaning_frequency_days=120,
            notes="Medium viscosity crude"
        ),
        "crude_oil_heavy": FoulingFactorData(
            value_si=0.001056,
            value_imperial=0.006,
            min_value=0.000880,
            max_value=0.001408,
            description="Heavy crude oil (API < 22)",
            cleaning_frequency_days=60,
            notes="High viscosity, asphaltenes"
        ),
        "asphalt_bitumen": FoulingFactorData(
            value_si=0.001760,
            value_imperial=0.010,
            min_value=0.001232,
            max_value=0.002640,
            description="Asphalt and bitumen",
            cleaning_frequency_days=30,
            notes="Very heavy, requires heating"
        ),
        "vegetable_oil": FoulingFactorData(
            value_si=0.000528,
            value_imperial=0.003,
            min_value=0.000352,
            max_value=0.000704,
            description="Vegetable oils",
            cleaning_frequency_days=180,
            notes="Polymerization risk at high temp"
        ),
        "lube_oil": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Lubricating oil",
            cleaning_frequency_days=365,
            notes="Clean base oil"
        ),
        "hydraulic_fluid": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Hydraulic fluid",
            cleaning_frequency_days=365,
            notes="Synthetic or mineral based"
        ),
    }

    # Steam Fouling Factors
    STEAM: Dict[str, FoulingFactorData] = {
        "clean_steam": FoulingFactorData(
            value_si=0.000088,
            value_imperial=0.0005,
            min_value=0.000044,
            max_value=0.000132,
            description="Clean steam (boiler feed quality)",
            cleaning_frequency_days=730,
            notes="Deaerated, treated water"
        ),
        "exhaust_steam_oil_free": FoulingFactorData(
            value_si=0.000088,
            value_imperial=0.0005,
            min_value=0.000044,
            max_value=0.000176,
            description="Exhaust steam, oil free",
            cleaning_frequency_days=365,
            notes="No oil carryover"
        ),
        "exhaust_steam_with_oil": FoulingFactorData(
            value_si=0.000264,
            value_imperial=0.0015,
            min_value=0.000176,
            max_value=0.000352,
            description="Exhaust steam with oil traces",
            cleaning_frequency_days=180,
            notes="From reciprocating equipment"
        ),
        "condensate_clean": FoulingFactorData(
            value_si=0.000088,
            value_imperial=0.0005,
            min_value=0.000044,
            max_value=0.000132,
            description="Clean condensate return",
            cleaning_frequency_days=730,
            notes="No contamination"
        ),
        "condensate_contaminated": FoulingFactorData(
            value_si=0.000352,
            value_imperial=0.002,
            min_value=0.000176,
            max_value=0.000528,
            description="Contaminated condensate",
            cleaning_frequency_days=180,
            notes="With process leaks"
        ),
    }

    # Refrigerant Fouling Factors
    REFRIGERANTS: Dict[str, FoulingFactorData] = {
        "ammonia_liquid": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Liquid ammonia (R-717)",
            cleaning_frequency_days=730,
            notes="Clean system"
        ),
        "ammonia_vapor": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Ammonia vapor",
            cleaning_frequency_days=730,
            notes="Superheated or saturated"
        ),
        "halocarbon_liquid": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Halocarbon refrigerants (R-22, R-134a)",
            cleaning_frequency_days=730,
            notes="Clean with proper oil"
        ),
        "halocarbon_vapor": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Halocarbon refrigerant vapor",
            cleaning_frequency_days=730,
            notes="Superheated discharge"
        ),
        "co2_liquid": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Liquid CO2 (R-744)",
            cleaning_frequency_days=730,
            notes="High pressure system"
        ),
        "propane_refrigerant": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Propane refrigerant (R-290)",
            cleaning_frequency_days=730,
            notes="Hydrocarbon refrigerant"
        ),
    }

    # Gas Fouling Factors
    GASES: Dict[str, FoulingFactorData] = {
        "air_clean": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000352,
            description="Clean air",
            cleaning_frequency_days=365,
            notes="Filtered air"
        ),
        "air_with_dust": FoulingFactorData(
            value_si=0.000528,
            value_imperial=0.003,
            min_value=0.000352,
            max_value=0.000704,
            description="Air with dust or particulates",
            cleaning_frequency_days=90,
            notes="Unfiltered industrial air"
        ),
        "flue_gas_clean": FoulingFactorData(
            value_si=0.000880,
            value_imperial=0.005,
            min_value=0.000528,
            max_value=0.001232,
            description="Clean flue gas (natural gas combustion)",
            cleaning_frequency_days=180,
            notes="Low ash content"
        ),
        "flue_gas_dirty": FoulingFactorData(
            value_si=0.001760,
            value_imperial=0.010,
            min_value=0.001232,
            max_value=0.002640,
            description="Dirty flue gas (coal, heavy oil)",
            cleaning_frequency_days=60,
            notes="High particulate loading"
        ),
        "natural_gas": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000352,
            description="Natural gas",
            cleaning_frequency_days=365,
            notes="Pipeline quality"
        ),
        "hydrogen": FoulingFactorData(
            value_si=0.000176,
            value_imperial=0.001,
            min_value=0.000088,
            max_value=0.000264,
            description="Hydrogen gas",
            cleaning_frequency_days=365,
            notes="Clean hydrogen"
        ),
        "nitrogen": FoulingFactorData(
            value_si=0.000088,
            value_imperial=0.0005,
            min_value=0.000044,
            max_value=0.000176,
            description="Nitrogen gas",
            cleaning_frequency_days=730,
            notes="Industrial grade"
        ),
        "oxygen": FoulingFactorData(
            value_si=0.000088,
            value_imperial=0.0005,
            min_value=0.000044,
            max_value=0.000176,
            description="Oxygen gas",
            cleaning_frequency_days=730,
            notes="Industrial grade"
        ),
    }

    @classmethod
    def get_fouling_factor(
        cls,
        category: str,
        fluid_type: str,
        units: str = "si"
    ) -> float:
        """
        Get fouling factor for a specific fluid.

        Args:
            category: Fluid category (cooling_water, process_fluid, steam, etc.)
            fluid_type: Specific fluid type within category
            units: "si" for m2.K/W or "imperial" for hr.ft2.F/BTU

        Returns:
            Fouling resistance value

        Raises:
            KeyError: If category or fluid type not found
        """
        category_map = {
            "cooling_water": cls.COOLING_WATER,
            "process_fluid": cls.PROCESS_FLUIDS,
            "steam": cls.STEAM,
            "refrigerant": cls.REFRIGERANTS,
            "gas": cls.GASES,
        }

        if category not in category_map:
            raise KeyError(f"Unknown fouling category: {category}")

        fluid_data = category_map[category].get(fluid_type)
        if fluid_data is None:
            raise KeyError(f"Unknown fluid type: {fluid_type} in category {category}")

        if units.lower() == "imperial":
            return fluid_data.value_imperial
        return fluid_data.value_si

    @classmethod
    def get_all_fouling_factors(cls) -> Dict[str, Dict[str, FoulingFactorData]]:
        """Return all fouling factor tables."""
        return {
            "cooling_water": cls.COOLING_WATER,
            "process_fluids": cls.PROCESS_FLUIDS,
            "steam": cls.STEAM,
            "refrigerants": cls.REFRIGERANTS,
            "gases": cls.GASES,
        }


# =============================================================================
# HEAT TRANSFER CORRELATIONS
# =============================================================================

@dataclass
class CorrelationCoefficients:
    """Coefficients for heat transfer correlations."""
    name: str
    coefficient: float
    reynolds_exponent: float
    prandtl_exponent: float
    viscosity_correction_exponent: float
    valid_reynolds_min: float
    valid_reynolds_max: float
    valid_prandtl_min: float
    valid_prandtl_max: float
    description: str
    reference: str


class HeatTransferCorrelations:
    """
    Standard heat transfer correlations for tube-side and shell-side.

    Includes Dittus-Boelter, Sieder-Tate, Gnielinski, and Colburn correlations.
    """

    # Dittus-Boelter correlation: Nu = C * Re^a * Pr^b
    DITTUS_BOELTER_HEATING = CorrelationCoefficients(
        name="Dittus-Boelter (Heating)",
        coefficient=0.023,
        reynolds_exponent=0.8,
        prandtl_exponent=0.4,
        viscosity_correction_exponent=0.0,
        valid_reynolds_min=10000,
        valid_reynolds_max=120000,
        valid_prandtl_min=0.7,
        valid_prandtl_max=160,
        description="Turbulent flow in smooth tubes, fluid being heated",
        reference="Dittus & Boelter, 1930"
    )

    DITTUS_BOELTER_COOLING = CorrelationCoefficients(
        name="Dittus-Boelter (Cooling)",
        coefficient=0.023,
        reynolds_exponent=0.8,
        prandtl_exponent=0.3,
        viscosity_correction_exponent=0.0,
        valid_reynolds_min=10000,
        valid_reynolds_max=120000,
        valid_prandtl_min=0.7,
        valid_prandtl_max=160,
        description="Turbulent flow in smooth tubes, fluid being cooled",
        reference="Dittus & Boelter, 1930"
    )

    # Sieder-Tate correlation: Nu = C * Re^a * Pr^b * (mu_b/mu_w)^c
    SIEDER_TATE = CorrelationCoefficients(
        name="Sieder-Tate",
        coefficient=0.027,
        reynolds_exponent=0.8,
        prandtl_exponent=0.333,
        viscosity_correction_exponent=0.14,
        valid_reynolds_min=10000,
        valid_reynolds_max=1000000,
        valid_prandtl_min=0.7,
        valid_prandtl_max=16700,
        description="Turbulent flow with viscosity correction for wall temperature",
        reference="Sieder & Tate, 1936"
    )

    # Gnielinski correlation (more accurate for transitional flow)
    GNIELINSKI = CorrelationCoefficients(
        name="Gnielinski",
        coefficient=0.0,  # Uses different formula
        reynolds_exponent=0.0,
        prandtl_exponent=0.0,
        viscosity_correction_exponent=0.0,
        valid_reynolds_min=2300,
        valid_reynolds_max=5000000,
        valid_prandtl_min=0.5,
        valid_prandtl_max=2000,
        description="Accurate for transition and turbulent flow",
        reference="Gnielinski, 1976"
    )

    # Colburn j-factor correlation
    COLBURN_TURBULENT = CorrelationCoefficients(
        name="Colburn (Turbulent)",
        coefficient=0.023,
        reynolds_exponent=-0.2,
        prandtl_exponent=0.0,
        viscosity_correction_exponent=0.0,
        valid_reynolds_min=10000,
        valid_reynolds_max=1000000,
        valid_prandtl_min=0.5,
        valid_prandtl_max=100,
        description="j-factor for turbulent tube flow: j = St * Pr^(2/3)",
        reference="Colburn, 1933"
    )

    # Laminar flow correlations
    LAMINAR_CONSTANT_WALL_TEMP = CorrelationCoefficients(
        name="Laminar (Constant Wall Temperature)",
        coefficient=3.66,
        reynolds_exponent=0.0,
        prandtl_exponent=0.0,
        viscosity_correction_exponent=0.0,
        valid_reynolds_min=0,
        valid_reynolds_max=2300,
        valid_prandtl_min=0.0,
        valid_prandtl_max=10000,
        description="Fully developed laminar flow, constant wall temp (Nu = 3.66)",
        reference="Analytical solution"
    )

    LAMINAR_CONSTANT_HEAT_FLUX = CorrelationCoefficients(
        name="Laminar (Constant Heat Flux)",
        coefficient=4.36,
        reynolds_exponent=0.0,
        prandtl_exponent=0.0,
        viscosity_correction_exponent=0.0,
        valid_reynolds_min=0,
        valid_reynolds_max=2300,
        valid_prandtl_min=0.0,
        valid_prandtl_max=10000,
        description="Fully developed laminar flow, constant heat flux (Nu = 4.36)",
        reference="Analytical solution"
    )

    # Shell-side correlations (Bell-Delaware method factors)
    BELL_DELAWARE_IDEAL_TUBE_BANK = CorrelationCoefficients(
        name="Bell-Delaware Ideal Tube Bank",
        coefficient=0.36,
        reynolds_exponent=0.55,
        prandtl_exponent=0.333,
        viscosity_correction_exponent=0.14,
        valid_reynolds_min=100,
        valid_reynolds_max=1000000,
        valid_prandtl_min=0.7,
        valid_prandtl_max=1000,
        description="Ideal tube bank coefficient for Bell-Delaware method",
        reference="Bell, 1963"
    )

    @classmethod
    def get_correlation(cls, name: str) -> Optional[CorrelationCoefficients]:
        """Get correlation by name."""
        correlations = {
            "dittus_boelter_heating": cls.DITTUS_BOELTER_HEATING,
            "dittus_boelter_cooling": cls.DITTUS_BOELTER_COOLING,
            "sieder_tate": cls.SIEDER_TATE,
            "gnielinski": cls.GNIELINSKI,
            "colburn_turbulent": cls.COLBURN_TURBULENT,
            "laminar_const_temp": cls.LAMINAR_CONSTANT_WALL_TEMP,
            "laminar_const_flux": cls.LAMINAR_CONSTANT_HEAT_FLUX,
            "bell_delaware": cls.BELL_DELAWARE_IDEAL_TUBE_BANK,
        }
        return correlations.get(name.lower())

    @classmethod
    def calculate_gnielinski_nusselt(
        cls,
        reynolds: float,
        prandtl: float,
        friction_factor: float
    ) -> float:
        """
        Calculate Nusselt number using Gnielinski correlation.

        Nu = (f/8)(Re-1000)Pr / (1 + 12.7*(f/8)^0.5 * (Pr^(2/3) - 1))

        Args:
            reynolds: Reynolds number
            prandtl: Prandtl number
            friction_factor: Darcy friction factor

        Returns:
            Nusselt number
        """
        if reynolds < 2300:
            return 3.66  # Laminar flow

        f_over_8 = friction_factor / 8.0
        numerator = f_over_8 * (reynolds - 1000) * prandtl
        denominator = 1.0 + 12.7 * (f_over_8 ** 0.5) * (prandtl ** (2/3) - 1)
        return numerator / denominator

    @classmethod
    def calculate_friction_factor(cls, reynolds: float, roughness: float = 0.0) -> float:
        """
        Calculate Darcy friction factor.

        Uses Blasius for smooth tubes, Haaland for rough tubes.

        Args:
            reynolds: Reynolds number
            roughness: Relative roughness (epsilon/D)

        Returns:
            Darcy friction factor
        """
        if reynolds < 2300:
            # Laminar flow: f = 64/Re
            return 64.0 / reynolds

        if roughness < 1e-10:
            # Smooth tube - Blasius equation
            return 0.316 * reynolds ** (-0.25)
        else:
            # Rough tube - Haaland equation
            import math
            term1 = (roughness / 3.7) ** 1.11
            term2 = 6.9 / reynolds
            return (1.0 / (-1.8 * math.log10(term1 + term2))) ** 2


# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================

@dataclass
class MaterialThermalData:
    """Thermal properties of tube/shell materials."""
    name: str
    thermal_conductivity_20C: float  # W/m.K at 20C
    thermal_conductivity_coefficients: Tuple[float, float, float]  # a, b, c for k = a + bT + cT^2
    density: float  # kg/m3
    specific_heat: float  # J/kg.K
    melting_point: float  # C
    max_service_temp: float  # C
    youngs_modulus: float  # GPa
    corrosion_allowance: float  # mm
    material_code: str
    notes: str


class MaterialProperties:
    """
    Material thermal properties database.

    Provides temperature-dependent thermal conductivity and other properties.
    """

    TUBE_MATERIALS: Dict[str, MaterialThermalData] = {
        "carbon_steel": MaterialThermalData(
            name="Carbon Steel (SA-179)",
            thermal_conductivity_20C=51.9,
            thermal_conductivity_coefficients=(52.0, -0.01, -0.00003),
            density=7850,
            specific_heat=486,
            melting_point=1510,
            max_service_temp=540,
            youngs_modulus=200,
            corrosion_allowance=3.2,
            material_code="SA-179",
            notes="Most common tube material, good thermal conductivity"
        ),
        "carbon_steel_killed": MaterialThermalData(
            name="Carbon Steel Killed (SA-192)",
            thermal_conductivity_20C=51.2,
            thermal_conductivity_coefficients=(51.5, -0.012, -0.00002),
            density=7850,
            specific_heat=486,
            melting_point=1510,
            max_service_temp=455,
            youngs_modulus=200,
            corrosion_allowance=3.2,
            material_code="SA-192",
            notes="For boiler and superheater tubes"
        ),
        "stainless_304": MaterialThermalData(
            name="Stainless Steel 304 (SA-213 TP304)",
            thermal_conductivity_20C=16.2,
            thermal_conductivity_coefficients=(14.8, 0.015, 0.0),
            density=8000,
            specific_heat=500,
            melting_point=1450,
            max_service_temp=815,
            youngs_modulus=193,
            corrosion_allowance=0.0,
            material_code="SA-213 TP304",
            notes="General purpose stainless, good corrosion resistance"
        ),
        "stainless_316": MaterialThermalData(
            name="Stainless Steel 316 (SA-213 TP316)",
            thermal_conductivity_20C=16.3,
            thermal_conductivity_coefficients=(14.6, 0.016, 0.0),
            density=8000,
            specific_heat=500,
            melting_point=1400,
            max_service_temp=815,
            youngs_modulus=193,
            corrosion_allowance=0.0,
            material_code="SA-213 TP316",
            notes="Better pitting resistance than 304"
        ),
        "stainless_316L": MaterialThermalData(
            name="Stainless Steel 316L (SA-213 TP316L)",
            thermal_conductivity_20C=16.3,
            thermal_conductivity_coefficients=(14.6, 0.016, 0.0),
            density=8000,
            specific_heat=500,
            melting_point=1400,
            max_service_temp=455,
            youngs_modulus=193,
            corrosion_allowance=0.0,
            material_code="SA-213 TP316L",
            notes="Low carbon for welded construction"
        ),
        "stainless_321": MaterialThermalData(
            name="Stainless Steel 321 (SA-213 TP321)",
            thermal_conductivity_20C=16.1,
            thermal_conductivity_coefficients=(14.5, 0.015, 0.0),
            density=8000,
            specific_heat=500,
            melting_point=1400,
            max_service_temp=815,
            youngs_modulus=193,
            corrosion_allowance=0.0,
            material_code="SA-213 TP321",
            notes="Ti stabilized for high temp service"
        ),
        "copper": MaterialThermalData(
            name="Copper (SB-111 C12200)",
            thermal_conductivity_20C=388,
            thermal_conductivity_coefficients=(401, -0.065, 0.0),
            density=8940,
            specific_heat=385,
            melting_point=1083,
            max_service_temp=200,
            youngs_modulus=117,
            corrosion_allowance=0.0,
            material_code="SB-111 C12200",
            notes="Excellent thermal conductivity, limited temp range"
        ),
        "copper_nickel_90_10": MaterialThermalData(
            name="Copper-Nickel 90/10 (SB-111 C70600)",
            thermal_conductivity_20C=45,
            thermal_conductivity_coefficients=(44, 0.005, 0.0),
            density=8900,
            specific_heat=377,
            melting_point=1170,
            max_service_temp=315,
            youngs_modulus=135,
            corrosion_allowance=0.0,
            material_code="SB-111 C70600",
            notes="Excellent seawater resistance"
        ),
        "copper_nickel_70_30": MaterialThermalData(
            name="Copper-Nickel 70/30 (SB-111 C71500)",
            thermal_conductivity_20C=29,
            thermal_conductivity_coefficients=(28, 0.004, 0.0),
            density=8900,
            specific_heat=377,
            melting_point=1240,
            max_service_temp=315,
            youngs_modulus=150,
            corrosion_allowance=0.0,
            material_code="SB-111 C71500",
            notes="Best seawater resistance of CuNi alloys"
        ),
        "admiralty_brass": MaterialThermalData(
            name="Admiralty Brass (SB-111 C44300)",
            thermal_conductivity_20C=111,
            thermal_conductivity_coefficients=(116, -0.025, 0.0),
            density=8530,
            specific_heat=380,
            melting_point=940,
            max_service_temp=175,
            youngs_modulus=100,
            corrosion_allowance=0.0,
            material_code="SB-111 C44300",
            notes="Good for fresh water, not seawater"
        ),
        "aluminum_brass": MaterialThermalData(
            name="Aluminum Brass (SB-111 C68700)",
            thermal_conductivity_20C=100,
            thermal_conductivity_coefficients=(103, -0.015, 0.0),
            density=8350,
            specific_heat=380,
            melting_point=1040,
            max_service_temp=175,
            youngs_modulus=103,
            corrosion_allowance=0.0,
            material_code="SB-111 C68700",
            notes="Better corrosion resistance than admiralty"
        ),
        "titanium_grade2": MaterialThermalData(
            name="Titanium Grade 2 (SB-338 Grade 2)",
            thermal_conductivity_20C=21.9,
            thermal_conductivity_coefficients=(20.8, 0.006, 0.0),
            density=4510,
            specific_heat=523,
            melting_point=1660,
            max_service_temp=315,
            youngs_modulus=103,
            corrosion_allowance=0.0,
            material_code="SB-338 Gr2",
            notes="Excellent corrosion resistance, lightweight"
        ),
        "titanium_grade12": MaterialThermalData(
            name="Titanium Grade 12 (SB-338 Grade 12)",
            thermal_conductivity_20C=21.0,
            thermal_conductivity_coefficients=(20.0, 0.005, 0.0),
            density=4510,
            specific_heat=523,
            melting_point=1660,
            max_service_temp=315,
            youngs_modulus=103,
            corrosion_allowance=0.0,
            material_code="SB-338 Gr12",
            notes="Higher strength than Grade 2"
        ),
        "hastelloy_c276": MaterialThermalData(
            name="Hastelloy C-276 (SB-622 N10276)",
            thermal_conductivity_20C=9.8,
            thermal_conductivity_coefficients=(9.0, 0.008, 0.0),
            density=8890,
            specific_heat=427,
            melting_point=1370,
            max_service_temp=1095,
            youngs_modulus=205,
            corrosion_allowance=0.0,
            material_code="SB-622 N10276",
            notes="Excellent chemical resistance"
        ),
        "inconel_625": MaterialThermalData(
            name="Inconel 625 (SB-444 N06625)",
            thermal_conductivity_20C=9.8,
            thermal_conductivity_coefficients=(8.7, 0.012, 0.0),
            density=8440,
            specific_heat=410,
            melting_point=1350,
            max_service_temp=980,
            youngs_modulus=205,
            corrosion_allowance=0.0,
            material_code="SB-444 N06625",
            notes="High strength at elevated temperatures"
        ),
        "monel_400": MaterialThermalData(
            name="Monel 400 (SB-163 N04400)",
            thermal_conductivity_20C=21.8,
            thermal_conductivity_coefficients=(20.5, 0.008, 0.0),
            density=8800,
            specific_heat=427,
            melting_point=1350,
            max_service_temp=480,
            youngs_modulus=179,
            corrosion_allowance=0.0,
            material_code="SB-163 N04400",
            notes="Good seawater resistance"
        ),
        "duplex_2205": MaterialThermalData(
            name="Duplex 2205 (SA-789 S32205)",
            thermal_conductivity_20C=19.0,
            thermal_conductivity_coefficients=(17.5, 0.012, 0.0),
            density=7800,
            specific_heat=500,
            melting_point=1450,
            max_service_temp=315,
            youngs_modulus=200,
            corrosion_allowance=0.0,
            material_code="SA-789 S32205",
            notes="High strength, good chloride resistance"
        ),
        "super_duplex_2507": MaterialThermalData(
            name="Super Duplex 2507 (SA-789 S32750)",
            thermal_conductivity_20C=14.0,
            thermal_conductivity_coefficients=(13.0, 0.010, 0.0),
            density=7800,
            specific_heat=500,
            melting_point=1450,
            max_service_temp=315,
            youngs_modulus=200,
            corrosion_allowance=0.0,
            material_code="SA-789 S32750",
            notes="Superior pitting resistance"
        ),
    }

    @classmethod
    def get_thermal_conductivity(cls, material: str, temperature: float) -> float:
        """
        Get temperature-dependent thermal conductivity.

        Args:
            material: Material key
            temperature: Temperature in Celsius

        Returns:
            Thermal conductivity in W/m.K
        """
        mat_data = cls.TUBE_MATERIALS.get(material)
        if mat_data is None:
            raise KeyError(f"Unknown material: {material}")

        a, b, c = mat_data.thermal_conductivity_coefficients
        return a + b * temperature + c * temperature * temperature

    @classmethod
    def get_material_data(cls, material: str) -> MaterialThermalData:
        """Get full material data."""
        mat_data = cls.TUBE_MATERIALS.get(material)
        if mat_data is None:
            raise KeyError(f"Unknown material: {material}")
        return mat_data


# =============================================================================
# FLUID PROPERTY TABLES
# =============================================================================

@dataclass
class FluidPropertyPoint:
    """Single point fluid property data."""
    temperature: float  # C
    density: float  # kg/m3
    specific_heat: float  # J/kg.K
    viscosity: float  # Pa.s
    thermal_conductivity: float  # W/m.K
    prandtl: float


class FluidPropertyTables:
    """
    Fluid property lookup tables.

    Provides temperature-dependent properties for common fluids.
    """

    # Water properties at atmospheric pressure (0 to 100 C)
    WATER_PROPERTIES: List[FluidPropertyPoint] = [
        FluidPropertyPoint(0, 999.8, 4217, 0.001792, 0.5610, 13.44),
        FluidPropertyPoint(5, 1000.0, 4202, 0.001519, 0.5710, 11.19),
        FluidPropertyPoint(10, 999.7, 4192, 0.001307, 0.5800, 9.45),
        FluidPropertyPoint(15, 999.1, 4186, 0.001138, 0.5890, 8.09),
        FluidPropertyPoint(20, 998.2, 4182, 0.001002, 0.5980, 7.01),
        FluidPropertyPoint(25, 997.0, 4180, 0.000891, 0.6070, 6.14),
        FluidPropertyPoint(30, 995.7, 4178, 0.000798, 0.6150, 5.42),
        FluidPropertyPoint(35, 994.0, 4178, 0.000720, 0.6230, 4.83),
        FluidPropertyPoint(40, 992.2, 4179, 0.000653, 0.6310, 4.33),
        FluidPropertyPoint(45, 990.2, 4180, 0.000596, 0.6380, 3.91),
        FluidPropertyPoint(50, 988.0, 4181, 0.000547, 0.6450, 3.55),
        FluidPropertyPoint(55, 985.7, 4183, 0.000504, 0.6520, 3.23),
        FluidPropertyPoint(60, 983.2, 4185, 0.000467, 0.6580, 2.97),
        FluidPropertyPoint(65, 980.5, 4188, 0.000433, 0.6630, 2.73),
        FluidPropertyPoint(70, 977.8, 4190, 0.000404, 0.6680, 2.53),
        FluidPropertyPoint(75, 974.8, 4194, 0.000378, 0.6730, 2.35),
        FluidPropertyPoint(80, 971.8, 4197, 0.000355, 0.6770, 2.20),
        FluidPropertyPoint(85, 968.6, 4201, 0.000334, 0.6810, 2.06),
        FluidPropertyPoint(90, 965.3, 4206, 0.000315, 0.6840, 1.94),
        FluidPropertyPoint(95, 961.9, 4212, 0.000298, 0.6870, 1.83),
        FluidPropertyPoint(100, 958.4, 4217, 0.000282, 0.6890, 1.73),
    ]

    # Steam saturation properties
    STEAM_SATURATION: Dict[str, Dict[str, float]] = {
        # Pressure (bar): {temp (C), hf (kJ/kg), hfg (kJ/kg), hg (kJ/kg), vf (m3/kg), vg (m3/kg)}
        "0.1": {"temp": 45.8, "hf": 191.8, "hfg": 2392.8, "hg": 2584.6, "vf": 0.001010, "vg": 14.67},
        "0.5": {"temp": 81.3, "hf": 340.5, "hfg": 2305.4, "hg": 2645.9, "vf": 0.001030, "vg": 3.240},
        "1.0": {"temp": 99.6, "hf": 417.4, "hfg": 2258.0, "hg": 2675.4, "vf": 0.001043, "vg": 1.694},
        "1.5": {"temp": 111.4, "hf": 467.1, "hfg": 2226.5, "hg": 2693.6, "vf": 0.001053, "vg": 1.159},
        "2.0": {"temp": 120.2, "hf": 504.7, "hfg": 2201.9, "hg": 2706.6, "vf": 0.001061, "vg": 0.886},
        "3.0": {"temp": 133.5, "hf": 561.5, "hfg": 2163.8, "hg": 2725.3, "vf": 0.001073, "vg": 0.606},
        "4.0": {"temp": 143.6, "hf": 604.7, "hfg": 2133.8, "hg": 2738.5, "vf": 0.001084, "vg": 0.463},
        "5.0": {"temp": 151.8, "hf": 640.2, "hfg": 2108.5, "hg": 2748.7, "vf": 0.001093, "vg": 0.375},
        "6.0": {"temp": 158.8, "hf": 670.6, "hfg": 2086.3, "hg": 2756.9, "vf": 0.001101, "vg": 0.316},
        "7.0": {"temp": 165.0, "hf": 697.2, "hfg": 2066.3, "hg": 2763.5, "vf": 0.001108, "vg": 0.273},
        "8.0": {"temp": 170.4, "hf": 721.1, "hfg": 2048.0, "hg": 2769.1, "vf": 0.001115, "vg": 0.240},
        "10.0": {"temp": 179.9, "hf": 762.8, "hfg": 2015.3, "hg": 2778.1, "vf": 0.001127, "vg": 0.194},
        "15.0": {"temp": 198.3, "hf": 844.8, "hfg": 1947.3, "hg": 2792.1, "vf": 0.001154, "vg": 0.132},
        "20.0": {"temp": 212.4, "hf": 908.8, "hfg": 1890.7, "hg": 2799.5, "vf": 0.001177, "vg": 0.100},
        "25.0": {"temp": 224.0, "hf": 962.1, "hfg": 1841.0, "hg": 2803.1, "vf": 0.001197, "vg": 0.080},
        "30.0": {"temp": 233.9, "hf": 1008.4, "hfg": 1795.7, "hg": 2804.1, "vf": 0.001216, "vg": 0.067},
        "40.0": {"temp": 250.4, "hf": 1087.3, "hfg": 1714.1, "hg": 2801.4, "vf": 0.001252, "vg": 0.050},
        "50.0": {"temp": 264.0, "hf": 1154.2, "hfg": 1640.1, "hg": 2794.3, "vf": 0.001286, "vg": 0.039},
    }

    # Common organic fluid properties at 25C
    ORGANIC_FLUIDS: Dict[str, FluidPropertyPoint] = {
        "methanol": FluidPropertyPoint(25, 787, 2530, 0.000544, 0.200, 6.88),
        "ethanol": FluidPropertyPoint(25, 785, 2440, 0.001074, 0.171, 15.35),
        "acetone": FluidPropertyPoint(25, 784, 2160, 0.000306, 0.161, 4.11),
        "benzene": FluidPropertyPoint(25, 874, 1720, 0.000604, 0.144, 7.22),
        "toluene": FluidPropertyPoint(25, 862, 1680, 0.000560, 0.131, 7.18),
        "hexane": FluidPropertyPoint(25, 655, 2270, 0.000300, 0.120, 5.67),
        "heptane": FluidPropertyPoint(25, 680, 2240, 0.000386, 0.124, 6.98),
        "octane": FluidPropertyPoint(25, 698, 2220, 0.000508, 0.128, 8.82),
        "ethylene_glycol": FluidPropertyPoint(25, 1110, 2350, 0.01610, 0.258, 146.7),
        "propylene_glycol": FluidPropertyPoint(25, 1036, 2480, 0.04200, 0.200, 521.0),
    }

    @classmethod
    def interpolate_water_property(cls, temperature: float, property_name: str) -> float:
        """
        Interpolate water property at given temperature.

        Args:
            temperature: Temperature in Celsius (0-100)
            property_name: One of 'density', 'specific_heat', 'viscosity',
                          'thermal_conductivity', 'prandtl'

        Returns:
            Interpolated property value
        """
        if temperature < 0 or temperature > 100:
            logger.warning(f"Temperature {temperature}C outside water table range (0-100C)")

        # Find bracketing points
        lower = cls.WATER_PROPERTIES[0]
        upper = cls.WATER_PROPERTIES[-1]

        for i, point in enumerate(cls.WATER_PROPERTIES[:-1]):
            if point.temperature <= temperature <= cls.WATER_PROPERTIES[i+1].temperature:
                lower = point
                upper = cls.WATER_PROPERTIES[i+1]
                break

        # Linear interpolation
        if upper.temperature == lower.temperature:
            frac = 0.0
        else:
            frac = (temperature - lower.temperature) / (upper.temperature - lower.temperature)

        prop_map = {
            "density": (lower.density, upper.density),
            "specific_heat": (lower.specific_heat, upper.specific_heat),
            "viscosity": (lower.viscosity, upper.viscosity),
            "thermal_conductivity": (lower.thermal_conductivity, upper.thermal_conductivity),
            "prandtl": (lower.prandtl, upper.prandtl),
        }

        if property_name not in prop_map:
            raise KeyError(f"Unknown property: {property_name}")

        lower_val, upper_val = prop_map[property_name]
        return lower_val + frac * (upper_val - lower_val)

    @classmethod
    def get_steam_saturation(cls, pressure_bar: float) -> Dict[str, float]:
        """
        Get steam saturation properties at given pressure.

        Args:
            pressure_bar: Pressure in bar

        Returns:
            Dict with temp, hf, hfg, hg, vf, vg
        """
        # Find closest pressure in table
        pressures = sorted([float(p) for p in cls.STEAM_SATURATION.keys()])
        closest = min(pressures, key=lambda x: abs(x - pressure_bar))
        return cls.STEAM_SATURATION[str(closest)]


# =============================================================================
# TEMA SHELL AND TUBE CONFIGURATIONS
# =============================================================================

@dataclass
class ShellTypeData:
    """TEMA shell type characteristics."""
    type_code: str
    name: str
    passes: int
    description: str
    typical_applications: List[str]
    pressure_drop_factor: float  # Relative to E-shell
    heat_transfer_factor: float  # Relative to E-shell
    cost_factor: float  # Relative to E-shell


class TEMAShellTypes:
    """
    TEMA Shell Type Configurations.

    Defines standard shell geometries and their characteristics.
    """

    SHELL_TYPES: Dict[str, ShellTypeData] = {
        "E": ShellTypeData(
            type_code="E",
            name="One Pass Shell",
            passes=1,
            description="Single pass, most common shell type",
            typical_applications=[
                "General purpose",
                "Single phase fluids",
                "Low to moderate heat recovery"
            ],
            pressure_drop_factor=1.0,
            heat_transfer_factor=1.0,
            cost_factor=1.0
        ),
        "F": ShellTypeData(
            type_code="F",
            name="Two Pass Shell with Longitudinal Baffle",
            passes=2,
            description="Longitudinal baffle creates two shell passes",
            typical_applications=[
                "True countercurrent flow",
                "High temperature approaches",
                "Temperature cross situations"
            ],
            pressure_drop_factor=2.2,
            heat_transfer_factor=1.15,
            cost_factor=1.25
        ),
        "G": ShellTypeData(
            type_code="G",
            name="Split Flow",
            passes=1,
            description="Central inlet/outlet nozzles, split flow to ends",
            typical_applications=[
                "Thermosiphon reboilers",
                "High flow rates",
                "Low pressure drop requirements"
            ],
            pressure_drop_factor=0.5,
            heat_transfer_factor=0.85,
            cost_factor=1.10
        ),
        "H": ShellTypeData(
            type_code="H",
            name="Double Split Flow",
            passes=1,
            description="Two inlet/two outlet nozzles",
            typical_applications=[
                "Very high flow rates",
                "Minimum pressure drop",
                "Large condensers"
            ],
            pressure_drop_factor=0.3,
            heat_transfer_factor=0.75,
            cost_factor=1.20
        ),
        "J": ShellTypeData(
            type_code="J",
            name="Divided Flow",
            passes=1,
            description="Shell inlet at center, outlets at both ends",
            typical_applications=[
                "Low pressure drop services",
                "Condensers",
                "Vacuum operation"
            ],
            pressure_drop_factor=0.4,
            heat_transfer_factor=0.80,
            cost_factor=1.15
        ),
        "K": ShellTypeData(
            type_code="K",
            name="Kettle Type Reboiler",
            passes=1,
            description="Enlarged shell for liquid/vapor disengagement",
            typical_applications=[
                "Kettle reboilers",
                "Chillers with flooded evaporator",
                "Steam generators"
            ],
            pressure_drop_factor=0.2,
            heat_transfer_factor=0.90,
            cost_factor=1.50
        ),
        "X": ShellTypeData(
            type_code="X",
            name="Cross Flow",
            passes=1,
            description="Pure cross flow, no baffles",
            typical_applications=[
                "Low pressure drop condensers",
                "Air coolers",
                "Low pressure gas cooling"
            ],
            pressure_drop_factor=0.15,
            heat_transfer_factor=0.70,
            cost_factor=1.05
        ),
    }

    @classmethod
    def get_shell_type(cls, type_code: str) -> ShellTypeData:
        """Get shell type data by code."""
        if type_code not in cls.SHELL_TYPES:
            raise KeyError(f"Unknown shell type: {type_code}")
        return cls.SHELL_TYPES[type_code]


@dataclass
class BaffleConfiguration:
    """Baffle configuration data."""
    type_name: str
    cut_percent: float  # Baffle cut as % of shell ID
    description: str
    heat_transfer_factor: float
    pressure_drop_factor: float
    leakage_factor: float
    typical_spacing_factor: float  # As fraction of shell ID


class BaffleConfigurations:
    """
    Standard baffle configurations.
    """

    BAFFLE_TYPES: Dict[str, BaffleConfiguration] = {
        "single_segmental_25": BaffleConfiguration(
            type_name="Single Segmental 25%",
            cut_percent=25.0,
            description="Standard single segmental baffle with 25% cut",
            heat_transfer_factor=1.0,
            pressure_drop_factor=1.0,
            leakage_factor=0.15,
            typical_spacing_factor=0.4
        ),
        "single_segmental_35": BaffleConfiguration(
            type_name="Single Segmental 35%",
            cut_percent=35.0,
            description="Single segmental baffle with 35% cut",
            heat_transfer_factor=0.90,
            pressure_drop_factor=0.75,
            leakage_factor=0.18,
            typical_spacing_factor=0.5
        ),
        "single_segmental_45": BaffleConfiguration(
            type_name="Single Segmental 45%",
            cut_percent=45.0,
            description="Single segmental baffle with 45% cut (condensers)",
            heat_transfer_factor=0.80,
            pressure_drop_factor=0.50,
            leakage_factor=0.22,
            typical_spacing_factor=0.6
        ),
        "double_segmental": BaffleConfiguration(
            type_name="Double Segmental",
            cut_percent=25.0,
            description="Double segmental baffles for reduced pressure drop",
            heat_transfer_factor=0.85,
            pressure_drop_factor=0.40,
            leakage_factor=0.20,
            typical_spacing_factor=0.3
        ),
        "triple_segmental": BaffleConfiguration(
            type_name="Triple Segmental",
            cut_percent=20.0,
            description="Triple segmental for very low pressure drop",
            heat_transfer_factor=0.75,
            pressure_drop_factor=0.25,
            leakage_factor=0.25,
            typical_spacing_factor=0.25
        ),
        "disk_and_doughnut": BaffleConfiguration(
            type_name="Disk and Doughnut",
            cut_percent=0.0,
            description="Alternating disk and doughnut baffles",
            heat_transfer_factor=0.95,
            pressure_drop_factor=0.60,
            leakage_factor=0.10,
            typical_spacing_factor=0.35
        ),
        "orifice_baffle": BaffleConfiguration(
            type_name="Orifice Baffle",
            cut_percent=0.0,
            description="Perforated plate baffles",
            heat_transfer_factor=0.70,
            pressure_drop_factor=0.20,
            leakage_factor=0.05,
            typical_spacing_factor=0.5
        ),
        "no_tubes_in_window": BaffleConfiguration(
            type_name="No Tubes in Window (NTIW)",
            cut_percent=25.0,
            description="Segmental with no tubes in baffle window",
            heat_transfer_factor=0.88,
            pressure_drop_factor=0.55,
            leakage_factor=0.08,
            typical_spacing_factor=0.4
        ),
        "rod_baffle": BaffleConfiguration(
            type_name="Rod Baffle",
            cut_percent=0.0,
            description="Support rods instead of plates, minimal vibration",
            heat_transfer_factor=0.65,
            pressure_drop_factor=0.15,
            leakage_factor=0.02,
            typical_spacing_factor=0.15
        ),
        "helical_baffle": BaffleConfiguration(
            type_name="Helical Baffle",
            cut_percent=0.0,
            description="Helical/spiral flow pattern",
            heat_transfer_factor=1.10,
            pressure_drop_factor=0.45,
            leakage_factor=0.05,
            typical_spacing_factor=0.3
        ),
    }

    @classmethod
    def get_baffle_config(cls, baffle_type: str) -> BaffleConfiguration:
        """Get baffle configuration by type."""
        if baffle_type not in cls.BAFFLE_TYPES:
            raise KeyError(f"Unknown baffle type: {baffle_type}")
        return cls.BAFFLE_TYPES[baffle_type]


@dataclass
class TubePatternData:
    """Tube layout pattern data."""
    pattern_name: str
    angle_degrees: float
    pitch_ratio_min: float  # Minimum pitch/OD ratio
    pitch_ratio_typical: float
    heat_transfer_factor: float  # Relative to triangular
    pressure_drop_factor: float
    cleanability: str  # mechanical, chemical, both
    description: str


class TubePatterns:
    """
    Standard tube layout patterns.
    """

    PATTERNS: Dict[str, TubePatternData] = {
        "triangular_30": TubePatternData(
            pattern_name="Triangular (30 deg)",
            angle_degrees=30.0,
            pitch_ratio_min=1.25,
            pitch_ratio_typical=1.25,
            heat_transfer_factor=1.0,
            pressure_drop_factor=1.0,
            cleanability="chemical",
            description="Standard triangular pitch, maximum tube count"
        ),
        "rotated_triangular_60": TubePatternData(
            pattern_name="Rotated Triangular (60 deg)",
            angle_degrees=60.0,
            pitch_ratio_min=1.25,
            pitch_ratio_typical=1.25,
            heat_transfer_factor=0.98,
            pressure_drop_factor=0.95,
            cleanability="chemical",
            description="Rotated triangular, slightly lower tube count"
        ),
        "square_90": TubePatternData(
            pattern_name="Square (90 deg)",
            angle_degrees=90.0,
            pitch_ratio_min=1.25,
            pitch_ratio_typical=1.25,
            heat_transfer_factor=0.85,
            pressure_drop_factor=0.70,
            cleanability="both",
            description="Square inline pattern, mechanical cleaning possible"
        ),
        "rotated_square_45": TubePatternData(
            pattern_name="Rotated Square (45 deg)",
            angle_degrees=45.0,
            pitch_ratio_min=1.25,
            pitch_ratio_typical=1.25,
            heat_transfer_factor=0.92,
            pressure_drop_factor=0.85,
            cleanability="chemical",
            description="45 degree rotated square pattern"
        ),
    }

    @classmethod
    def get_pattern(cls, pattern_name: str) -> TubePatternData:
        """Get tube pattern data by name."""
        if pattern_name not in cls.PATTERNS:
            raise KeyError(f"Unknown tube pattern: {pattern_name}")
        return cls.PATTERNS[pattern_name]


# =============================================================================
# CLEANING PARAMETERS
# =============================================================================

@dataclass
class CleaningMethodData:
    """Cleaning method parameters."""
    method_name: str
    effectiveness: float  # 0 to 1, fraction of fouling removed
    restoration_factor: float  # Fraction of original U restored
    duration_hours: float  # Typical cleaning duration
    cost_factor: float  # Relative cost (1.0 = base)
    applicable_fouling_types: List[str]
    limitations: List[str]
    frequency_months: int  # Typical cleaning frequency
    description: str


class CleaningParameters:
    """
    Heat exchanger cleaning parameters.

    Provides data for various cleaning methods and their effectiveness.
    """

    CLEANING_METHODS: Dict[str, CleaningMethodData] = {
        "hydroblasting": CleaningMethodData(
            method_name="Hydroblasting",
            effectiveness=0.95,
            restoration_factor=0.95,
            duration_hours=8,
            cost_factor=1.0,
            applicable_fouling_types=[
                "scale", "sludge", "biological", "particulate"
            ],
            limitations=[
                "Requires bundle removal for shell-side",
                "May damage soft materials"
            ],
            frequency_months=12,
            description="High-pressure water jetting (10000-15000 psi)"
        ),
        "chemical_acid": CleaningMethodData(
            method_name="Acid Cleaning",
            effectiveness=0.90,
            restoration_factor=0.90,
            duration_hours=6,
            cost_factor=0.8,
            applicable_fouling_types=[
                "scale", "rust", "carbonate", "sulfate"
            ],
            limitations=[
                "Material compatibility required",
                "Neutralization needed",
                "Environmental disposal"
            ],
            frequency_months=12,
            description="Acidic solution circulation (HCl, citric, sulfamic)"
        ),
        "chemical_alkaline": CleaningMethodData(
            method_name="Alkaline Cleaning",
            effectiveness=0.85,
            restoration_factor=0.85,
            duration_hours=8,
            cost_factor=0.7,
            applicable_fouling_types=[
                "oil", "grease", "biological", "organic"
            ],
            limitations=[
                "Less effective on scale",
                "Rinsing required"
            ],
            frequency_months=6,
            description="Alkaline solution with surfactants"
        ),
        "mechanical_brushing": CleaningMethodData(
            method_name="Mechanical Brushing",
            effectiveness=0.80,
            restoration_factor=0.80,
            duration_hours=12,
            cost_factor=1.2,
            applicable_fouling_types=[
                "soft deposits", "biological", "loose scale"
            ],
            limitations=[
                "Tube-side only",
                "Straight tubes only",
                "Labor intensive"
            ],
            frequency_months=6,
            description="Rotating brush or pig cleaning"
        ),
        "pigging": CleaningMethodData(
            method_name="Pigging (Projectile)",
            effectiveness=0.85,
            restoration_factor=0.85,
            duration_hours=4,
            cost_factor=0.6,
            applicable_fouling_types=[
                "soft deposits", "scale", "sludge"
            ],
            limitations=[
                "Tube-side only",
                "Straight tubes only",
                "May stick in severe fouling"
            ],
            frequency_months=3,
            description="Foam or hard projectile cleaning"
        ),
        "thermal_shock": CleaningMethodData(
            method_name="Thermal Shock",
            effectiveness=0.70,
            restoration_factor=0.70,
            duration_hours=2,
            cost_factor=0.3,
            applicable_fouling_types=[
                "scale", "brittle deposits"
            ],
            limitations=[
                "May cause thermal stress",
                "Not for all materials",
                "Limited effectiveness"
            ],
            frequency_months=3,
            description="Rapid heating/cooling to crack deposits"
        ),
        "ultrasonic": CleaningMethodData(
            method_name="Ultrasonic Cleaning",
            effectiveness=0.88,
            restoration_factor=0.88,
            duration_hours=6,
            cost_factor=1.5,
            applicable_fouling_types=[
                "scale", "biological", "particulate", "oil"
            ],
            limitations=[
                "Requires immersion tank",
                "Bundle removal needed",
                "Size limitations"
            ],
            frequency_months=12,
            description="Ultrasonic cavitation in cleaning solution"
        ),
        "steam_blow": CleaningMethodData(
            method_name="Steam Blowing",
            effectiveness=0.60,
            restoration_factor=0.60,
            duration_hours=4,
            cost_factor=0.4,
            applicable_fouling_types=[
                "oil", "grease", "volatile deposits"
            ],
            limitations=[
                "Limited effectiveness",
                "Material temperature limits",
                "Condensate handling"
            ],
            frequency_months=3,
            description="High pressure steam cleaning"
        ),
        "solvent_cleaning": CleaningMethodData(
            method_name="Solvent Cleaning",
            effectiveness=0.85,
            restoration_factor=0.85,
            duration_hours=8,
            cost_factor=1.3,
            applicable_fouling_types=[
                "oil", "grease", "tar", "polymer"
            ],
            limitations=[
                "Environmental concerns",
                "Fire/safety hazards",
                "Material compatibility"
            ],
            frequency_months=12,
            description="Organic solvent circulation"
        ),
        "online_mechanical": CleaningMethodData(
            method_name="Online Mechanical (TAPROGGE)",
            effectiveness=0.95,
            restoration_factor=0.98,
            duration_hours=0,  # Continuous
            cost_factor=3.0,  # Higher capital, lower operating
            applicable_fouling_types=[
                "biological", "soft deposits", "scale"
            ],
            limitations=[
                "High capital cost",
                "Tube-side only",
                "Requires ball basket systems"
            ],
            frequency_months=0,  # Continuous
            description="Continuous sponge ball cleaning system"
        ),
        "chemical_chelant": CleaningMethodData(
            method_name="Chelant Cleaning",
            effectiveness=0.92,
            restoration_factor=0.92,
            duration_hours=10,
            cost_factor=1.4,
            applicable_fouling_types=[
                "iron oxide", "copper deposits", "scale"
            ],
            limitations=[
                "Temperature dependent",
                "Longer contact time",
                "Higher chemical cost"
            ],
            frequency_months=24,
            description="EDTA or citric acid based chelation"
        ),
    }

    # Cost factors by fouling severity
    SEVERITY_COST_MULTIPLIERS: Dict[str, float] = {
        "light": 0.7,
        "moderate": 1.0,
        "heavy": 1.5,
        "severe": 2.5,
    }

    # Downtime cost factors by equipment criticality
    CRITICALITY_FACTORS: Dict[str, float] = {
        "spare_available": 0.2,
        "redundant": 0.5,
        "single_train": 1.0,
        "bottleneck": 2.0,
        "safety_critical": 3.0,
    }

    @classmethod
    def get_cleaning_method(cls, method: str) -> CleaningMethodData:
        """Get cleaning method data."""
        if method not in cls.CLEANING_METHODS:
            raise KeyError(f"Unknown cleaning method: {method}")
        return cls.CLEANING_METHODS[method]

    @classmethod
    def estimate_cleaning_cost(
        cls,
        method: str,
        base_cost: float,
        severity: str = "moderate",
        criticality: str = "single_train"
    ) -> float:
        """
        Estimate total cleaning cost including downtime.

        Args:
            method: Cleaning method name
            base_cost: Base cost for standard cleaning
            severity: Fouling severity level
            criticality: Equipment criticality

        Returns:
            Estimated total cost
        """
        method_data = cls.get_cleaning_method(method)
        severity_mult = cls.SEVERITY_COST_MULTIPLIERS.get(severity, 1.0)
        criticality_mult = cls.CRITICALITY_FACTORS.get(criticality, 1.0)

        return base_cost * method_data.cost_factor * severity_mult * criticality_mult

    @classmethod
    def recommend_cleaning_method(
        cls,
        fouling_type: str,
        tube_side: bool = True,
        straight_tubes: bool = True
    ) -> List[str]:
        """
        Recommend cleaning methods for given conditions.

        Args:
            fouling_type: Type of fouling present
            tube_side: True if cleaning tube side
            straight_tubes: True if tubes are straight

        Returns:
            List of recommended method names, sorted by effectiveness
        """
        recommendations = []

        for method_name, method_data in cls.CLEANING_METHODS.items():
            # Check if fouling type is applicable
            if fouling_type not in method_data.applicable_fouling_types:
                continue

            # Check tube-side limitations
            if not tube_side and "Tube-side only" in method_data.limitations:
                continue

            # Check straight tube limitations
            if not straight_tubes and "Straight tubes only" in method_data.limitations:
                continue

            recommendations.append((method_name, method_data.effectiveness))

        # Sort by effectiveness descending
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return [r[0] for r in recommendations]


# =============================================================================
# STANDARD TUBE DIMENSIONS
# =============================================================================

@dataclass
class TubeDimension:
    """Standard tube dimension data."""
    od_inch: float
    od_mm: float
    bwg: int
    wall_inch: float
    wall_mm: float
    id_inch: float
    id_mm: float
    area_per_foot_ft2: float
    area_per_meter_m2: float


class StandardTubeDimensions:
    """
    Standard tube dimensions per BWG (Birmingham Wire Gauge).
    """

    TUBE_SIZES: Dict[str, List[TubeDimension]] = {
        "0.500": [  # 1/2 inch OD
            TubeDimension(0.500, 12.70, 18, 0.049, 1.24, 0.402, 10.21, 0.1309, 0.0399),
            TubeDimension(0.500, 12.70, 16, 0.065, 1.65, 0.370, 9.40, 0.1309, 0.0399),
            TubeDimension(0.500, 12.70, 14, 0.083, 2.11, 0.334, 8.48, 0.1309, 0.0399),
        ],
        "0.625": [  # 5/8 inch OD
            TubeDimension(0.625, 15.88, 18, 0.049, 1.24, 0.527, 13.39, 0.1636, 0.0499),
            TubeDimension(0.625, 15.88, 16, 0.065, 1.65, 0.495, 12.57, 0.1636, 0.0499),
            TubeDimension(0.625, 15.88, 14, 0.083, 2.11, 0.459, 11.66, 0.1636, 0.0499),
        ],
        "0.750": [  # 3/4 inch OD
            TubeDimension(0.750, 19.05, 18, 0.049, 1.24, 0.652, 16.56, 0.1963, 0.0598),
            TubeDimension(0.750, 19.05, 16, 0.065, 1.65, 0.620, 15.75, 0.1963, 0.0598),
            TubeDimension(0.750, 19.05, 14, 0.083, 2.11, 0.584, 14.83, 0.1963, 0.0598),
            TubeDimension(0.750, 19.05, 12, 0.109, 2.77, 0.532, 13.51, 0.1963, 0.0598),
        ],
        "1.000": [  # 1 inch OD
            TubeDimension(1.000, 25.40, 18, 0.049, 1.24, 0.902, 22.91, 0.2618, 0.0798),
            TubeDimension(1.000, 25.40, 16, 0.065, 1.65, 0.870, 22.10, 0.2618, 0.0798),
            TubeDimension(1.000, 25.40, 14, 0.083, 2.11, 0.834, 21.18, 0.2618, 0.0798),
            TubeDimension(1.000, 25.40, 12, 0.109, 2.77, 0.782, 19.86, 0.2618, 0.0798),
            TubeDimension(1.000, 25.40, 10, 0.134, 3.40, 0.732, 18.59, 0.2618, 0.0798),
        ],
        "1.250": [  # 1-1/4 inch OD
            TubeDimension(1.250, 31.75, 16, 0.065, 1.65, 1.120, 28.45, 0.3272, 0.0997),
            TubeDimension(1.250, 31.75, 14, 0.083, 2.11, 1.084, 27.53, 0.3272, 0.0997),
            TubeDimension(1.250, 31.75, 12, 0.109, 2.77, 1.032, 26.21, 0.3272, 0.0997),
            TubeDimension(1.250, 31.75, 10, 0.134, 3.40, 0.982, 24.94, 0.3272, 0.0997),
        ],
        "1.500": [  # 1-1/2 inch OD
            TubeDimension(1.500, 38.10, 16, 0.065, 1.65, 1.370, 34.80, 0.3927, 0.1197),
            TubeDimension(1.500, 38.10, 14, 0.083, 2.11, 1.334, 33.88, 0.3927, 0.1197),
            TubeDimension(1.500, 38.10, 12, 0.109, 2.77, 1.282, 32.56, 0.3927, 0.1197),
            TubeDimension(1.500, 38.10, 10, 0.134, 3.40, 1.232, 31.29, 0.3927, 0.1197),
        ],
        "2.000": [  # 2 inch OD
            TubeDimension(2.000, 50.80, 14, 0.083, 2.11, 1.834, 46.58, 0.5236, 0.1596),
            TubeDimension(2.000, 50.80, 12, 0.109, 2.77, 1.782, 45.26, 0.5236, 0.1596),
            TubeDimension(2.000, 50.80, 10, 0.134, 3.40, 1.732, 43.99, 0.5236, 0.1596),
        ],
    }

    @classmethod
    def get_tube_dimensions(cls, od_inch: float, bwg: int) -> Optional[TubeDimension]:
        """
        Get tube dimensions for given OD and BWG.

        Args:
            od_inch: Outer diameter in inches
            bwg: Birmingham Wire Gauge number

        Returns:
            TubeDimension object or None if not found
        """
        od_key = f"{od_inch:.3f}"
        tubes = cls.TUBE_SIZES.get(od_key, [])

        for tube in tubes:
            if tube.bwg == bwg:
                return tube
        return None

    @classmethod
    def get_available_sizes(cls) -> List[str]:
        """Get list of available tube OD sizes."""
        return list(cls.TUBE_SIZES.keys())


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class PhysicalConstants:
    """
    Physical constants used in heat exchanger calculations.
    """

    # Gravitational acceleration
    G_SI = 9.80665  # m/s2
    G_IMPERIAL = 32.174  # ft/s2

    # Universal gas constant
    R_SI = 8.314  # J/(mol.K)
    R_IMPERIAL = 1.987  # BTU/(lbmol.R)

    # Stefan-Boltzmann constant
    STEFAN_BOLTZMANN = 5.670374e-8  # W/(m2.K4)

    # Standard conditions
    STD_TEMP_C = 25.0
    STD_TEMP_K = 298.15
    STD_PRESSURE_PA = 101325
    STD_PRESSURE_BAR = 1.01325

    # Water properties at standard conditions
    WATER_DENSITY_25C = 997.0  # kg/m3
    WATER_CP_25C = 4180  # J/(kg.K)
    WATER_VISCOSITY_25C = 0.000891  # Pa.s
    WATER_K_25C = 0.607  # W/(m.K)

    # Air properties at standard conditions
    AIR_DENSITY_25C = 1.184  # kg/m3
    AIR_CP_25C = 1006  # J/(kg.K)
    AIR_VISCOSITY_25C = 1.849e-5  # Pa.s
    AIR_K_25C = 0.0261  # W/(m.K)


# Export all classes
__all__ = [
    "FoulingCategory",
    "FoulingFactorData",
    "FoulingFactors",
    "CorrelationCoefficients",
    "HeatTransferCorrelations",
    "MaterialThermalData",
    "MaterialProperties",
    "FluidPropertyPoint",
    "FluidPropertyTables",
    "ShellTypeData",
    "TEMAShellTypes",
    "BaffleConfiguration",
    "BaffleConfigurations",
    "TubePatternData",
    "TubePatterns",
    "CleaningMethodData",
    "CleaningParameters",
    "TubeDimension",
    "StandardTubeDimensions",
    "PhysicalConstants",
]
