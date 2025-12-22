"""
GreenLang Framework - Physical Constants and Emission Factors

Authoritative constants for all GreenLang calculations.
Sources: NIST, IPCC, EPA, DEFRA
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class ConstantSource(Enum):
    """Source of constant values."""
    NIST = "NIST"          # National Institute of Standards and Technology
    IPCC = "IPCC"          # Intergovernmental Panel on Climate Change
    EPA = "EPA"            # US Environmental Protection Agency
    DEFRA = "DEFRA"        # UK Department for Environment, Food & Rural Affairs
    ISO = "ISO"            # International Organization for Standardization
    ASHRAE = "ASHRAE"      # American Society of Heating, Refrigerating and Air-Conditioning Engineers
    IAPWS = "IAPWS"        # International Association for Properties of Water and Steam
    CALCULATED = "CALCULATED"


@dataclass
class ConstantValue:
    """A physical constant with metadata."""
    value: float
    unit: str
    source: ConstantSource
    uncertainty: Optional[float] = None
    year: Optional[int] = None
    notes: Optional[str] = None


class PhysicalConstants:
    """
    Physical constants for industrial calculations.

    All values from authoritative sources with full traceability.
    """

    # Thermodynamic constants
    ABSOLUTE_ZERO_C = ConstantValue(
        value=-273.15,
        unit="°C",
        source=ConstantSource.NIST,
        notes="Absolute zero in Celsius"
    )

    STANDARD_TEMPERATURE_K = ConstantValue(
        value=273.15,
        unit="K",
        source=ConstantSource.ISO,
        notes="Standard temperature (0°C)"
    )

    STANDARD_PRESSURE_PA = ConstantValue(
        value=101325.0,
        unit="Pa",
        source=ConstantSource.ISO,
        notes="Standard atmospheric pressure"
    )

    STANDARD_PRESSURE_BAR = ConstantValue(
        value=1.01325,
        unit="bar",
        source=ConstantSource.ISO,
        notes="Standard atmospheric pressure"
    )

    # Universal gas constant
    GAS_CONSTANT = ConstantValue(
        value=8.314462618,
        unit="J/(mol·K)",
        source=ConstantSource.NIST,
        uncertainty=0.0,
        year=2019,
        notes="Exact value as of 2019 SI redefinition"
    )

    # Specific gas constant for air
    GAS_CONSTANT_AIR = ConstantValue(
        value=287.058,
        unit="J/(kg·K)",
        source=ConstantSource.NIST,
        notes="For dry air"
    )

    # Stefan-Boltzmann constant
    STEFAN_BOLTZMANN = ConstantValue(
        value=5.670374419e-8,
        unit="W/(m²·K⁴)",
        source=ConstantSource.NIST,
        uncertainty=0.0,
        year=2019,
        notes="Exact value as of 2019 SI redefinition"
    )

    # Water properties
    WATER_DENSITY_STP = ConstantValue(
        value=999.97,
        unit="kg/m³",
        source=ConstantSource.IAPWS,
        notes="At 0°C, 101325 Pa"
    )

    WATER_SPECIFIC_HEAT = ConstantValue(
        value=4186.0,
        unit="J/(kg·K)",
        source=ConstantSource.IAPWS,
        notes="At 25°C, liquid water"
    )

    WATER_LATENT_HEAT_VAPORIZATION = ConstantValue(
        value=2257000.0,
        unit="J/kg",
        source=ConstantSource.IAPWS,
        notes="At 100°C, 101325 Pa"
    )

    WATER_CRITICAL_TEMPERATURE = ConstantValue(
        value=647.096,
        unit="K",
        source=ConstantSource.IAPWS,
        notes="Critical point temperature"
    )

    WATER_CRITICAL_PRESSURE = ConstantValue(
        value=22064000.0,
        unit="Pa",
        source=ConstantSource.IAPWS,
        notes="Critical point pressure"
    )

    # Steam properties
    STEAM_SPECIFIC_HEAT_100C = ConstantValue(
        value=2010.0,
        unit="J/(kg·K)",
        source=ConstantSource.IAPWS,
        notes="Saturated steam at 100°C"
    )

    # Air properties
    AIR_DENSITY_STP = ConstantValue(
        value=1.225,
        unit="kg/m³",
        source=ConstantSource.ASHRAE,
        notes="At 15°C, 101325 Pa"
    )

    AIR_SPECIFIC_HEAT = ConstantValue(
        value=1006.0,
        unit="J/(kg·K)",
        source=ConstantSource.ASHRAE,
        notes="Dry air at 25°C"
    )

    # Combustion constants
    NATURAL_GAS_HHV = ConstantValue(
        value=55500000.0,
        unit="J/kg",
        source=ConstantSource.EPA,
        notes="Higher Heating Value, typical natural gas"
    )

    NATURAL_GAS_LHV = ConstantValue(
        value=50000000.0,
        unit="J/kg",
        source=ConstantSource.EPA,
        notes="Lower Heating Value, typical natural gas"
    )

    # Gravity
    GRAVITY_STANDARD = ConstantValue(
        value=9.80665,
        unit="m/s²",
        source=ConstantSource.NIST,
        notes="Standard acceleration of gravity"
    )

    @classmethod
    def get_all(cls) -> Dict[str, ConstantValue]:
        """Get all constants as dictionary."""
        return {
            name: value for name, value in vars(cls).items()
            if isinstance(value, ConstantValue)
        }


@dataclass
class EmissionFactor:
    """An emission factor with full provenance."""
    value: float
    unit: str
    fuel_type: str
    source: ConstantSource
    year: int
    scope: int  # 1, 2, or 3
    region: str = "Global"
    uncertainty_percent: Optional[float] = None
    notes: Optional[str] = None


class EmissionFactors:
    """
    Emission factors for carbon accounting.

    Sources: IPCC 2006/2019, EPA, DEFRA 2023
    All factors in kg CO2e per unit specified.
    """

    # Natural Gas (Scope 1)
    NATURAL_GAS_KG_CO2_PER_KWH = EmissionFactor(
        value=0.18293,
        unit="kg CO2e/kWh",
        fuel_type="Natural Gas",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1,
        notes="Gross CV basis"
    )

    NATURAL_GAS_KG_CO2_PER_M3 = EmissionFactor(
        value=2.02,
        unit="kg CO2e/m³",
        fuel_type="Natural Gas",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1,
        notes="At standard conditions"
    )

    NATURAL_GAS_KG_CO2_PER_THERM = EmissionFactor(
        value=5.31,
        unit="kg CO2e/therm",
        fuel_type="Natural Gas",
        source=ConstantSource.EPA,
        year=2023,
        scope=1,
        region="USA"
    )

    # Diesel/Gas Oil (Scope 1)
    DIESEL_KG_CO2_PER_LITER = EmissionFactor(
        value=2.70,
        unit="kg CO2e/liter",
        fuel_type="Diesel",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1
    )

    DIESEL_KG_CO2_PER_KWH = EmissionFactor(
        value=0.25301,
        unit="kg CO2e/kWh",
        fuel_type="Diesel",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1
    )

    # Fuel Oil (Scope 1)
    FUEL_OIL_KG_CO2_PER_LITER = EmissionFactor(
        value=3.18,
        unit="kg CO2e/liter",
        fuel_type="Fuel Oil",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1,
        notes="Residual fuel oil"
    )

    # Coal (Scope 1)
    COAL_KG_CO2_PER_KG = EmissionFactor(
        value=2.42,
        unit="kg CO2e/kg",
        fuel_type="Coal",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1,
        notes="Industrial coal"
    )

    COAL_KG_CO2_PER_KWH = EmissionFactor(
        value=0.32307,
        unit="kg CO2e/kWh",
        fuel_type="Coal",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1
    )

    # LPG (Scope 1)
    LPG_KG_CO2_PER_KWH = EmissionFactor(
        value=0.21445,
        unit="kg CO2e/kWh",
        fuel_type="LPG",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1
    )

    LPG_KG_CO2_PER_LITER = EmissionFactor(
        value=1.56,
        unit="kg CO2e/liter",
        fuel_type="LPG",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=1
    )

    # Electricity Grid Factors (Scope 2) - 2023 values
    ELECTRICITY_UK_KG_CO2_PER_KWH = EmissionFactor(
        value=0.20705,
        unit="kg CO2e/kWh",
        fuel_type="Grid Electricity",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=2,
        region="UK"
    )

    ELECTRICITY_USA_KG_CO2_PER_KWH = EmissionFactor(
        value=0.417,
        unit="kg CO2e/kWh",
        fuel_type="Grid Electricity",
        source=ConstantSource.EPA,
        year=2022,
        scope=2,
        region="USA",
        notes="eGRID national average"
    )

    ELECTRICITY_EU_KG_CO2_PER_KWH = EmissionFactor(
        value=0.276,
        unit="kg CO2e/kWh",
        fuel_type="Grid Electricity",
        source=ConstantSource.IPCC,
        year=2022,
        scope=2,
        region="EU27"
    )

    ELECTRICITY_GERMANY_KG_CO2_PER_KWH = EmissionFactor(
        value=0.366,
        unit="kg CO2e/kWh",
        fuel_type="Grid Electricity",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=2,
        region="Germany"
    )

    ELECTRICITY_FRANCE_KG_CO2_PER_KWH = EmissionFactor(
        value=0.056,
        unit="kg CO2e/kWh",
        fuel_type="Grid Electricity",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=2,
        region="France",
        notes="High nuclear share"
    )

    ELECTRICITY_CHINA_KG_CO2_PER_KWH = EmissionFactor(
        value=0.581,
        unit="kg CO2e/kWh",
        fuel_type="Grid Electricity",
        source=ConstantSource.IPCC,
        year=2022,
        scope=2,
        region="China"
    )

    ELECTRICITY_INDIA_KG_CO2_PER_KWH = EmissionFactor(
        value=0.716,
        unit="kg CO2e/kWh",
        fuel_type="Grid Electricity",
        source=ConstantSource.IPCC,
        year=2022,
        scope=2,
        region="India"
    )

    # Steam (Scope 2)
    STEAM_KG_CO2_PER_KWH = EmissionFactor(
        value=0.17069,
        unit="kg CO2e/kWh",
        fuel_type="Steam",
        source=ConstantSource.DEFRA,
        year=2023,
        scope=2,
        notes="District heating/steam"
    )

    # Refrigerants (Scope 1)
    R134A_KG_CO2_PER_KG = EmissionFactor(
        value=1430.0,
        unit="kg CO2e/kg",
        fuel_type="R-134a",
        source=ConstantSource.IPCC,
        year=2021,
        scope=1,
        notes="GWP-100 AR6"
    )

    R410A_KG_CO2_PER_KG = EmissionFactor(
        value=2088.0,
        unit="kg CO2e/kg",
        fuel_type="R-410A",
        source=ConstantSource.IPCC,
        year=2021,
        scope=1,
        notes="GWP-100 AR6"
    )

    R32_KG_CO2_PER_KG = EmissionFactor(
        value=675.0,
        unit="kg CO2e/kg",
        fuel_type="R-32",
        source=ConstantSource.IPCC,
        year=2021,
        scope=1,
        notes="GWP-100 AR6"
    )

    @classmethod
    def get_by_fuel_type(cls, fuel_type: str) -> list:
        """Get all emission factors for a fuel type."""
        factors = []
        for name, value in vars(cls).items():
            if isinstance(value, EmissionFactor) and value.fuel_type.lower() == fuel_type.lower():
                factors.append((name, value))
        return factors

    @classmethod
    def get_by_scope(cls, scope: int) -> list:
        """Get all emission factors for a scope."""
        factors = []
        for name, value in vars(cls).items():
            if isinstance(value, EmissionFactor) and value.scope == scope:
                factors.append((name, value))
        return factors

    @classmethod
    def get_electricity_factor(cls, region: str) -> Optional[EmissionFactor]:
        """Get electricity emission factor for a region."""
        region_map = {
            "UK": cls.ELECTRICITY_UK_KG_CO2_PER_KWH,
            "USA": cls.ELECTRICITY_USA_KG_CO2_PER_KWH,
            "US": cls.ELECTRICITY_USA_KG_CO2_PER_KWH,
            "EU": cls.ELECTRICITY_EU_KG_CO2_PER_KWH,
            "EU27": cls.ELECTRICITY_EU_KG_CO2_PER_KWH,
            "Germany": cls.ELECTRICITY_GERMANY_KG_CO2_PER_KWH,
            "DE": cls.ELECTRICITY_GERMANY_KG_CO2_PER_KWH,
            "France": cls.ELECTRICITY_FRANCE_KG_CO2_PER_KWH,
            "FR": cls.ELECTRICITY_FRANCE_KG_CO2_PER_KWH,
            "China": cls.ELECTRICITY_CHINA_KG_CO2_PER_KWH,
            "CN": cls.ELECTRICITY_CHINA_KG_CO2_PER_KWH,
            "India": cls.ELECTRICITY_INDIA_KG_CO2_PER_KWH,
            "IN": cls.ELECTRICITY_INDIA_KG_CO2_PER_KWH,
        }
        return region_map.get(region)

    @classmethod
    def get_all(cls) -> Dict[str, EmissionFactor]:
        """Get all emission factors as dictionary."""
        return {
            name: value for name, value in vars(cls).items()
            if isinstance(value, EmissionFactor)
        }


# Global Warming Potentials (GWP-100, AR6)
class GWP:
    """Global Warming Potentials from IPCC AR6."""
    CO2 = 1
    CH4 = 27.9  # Fossil origin
    CH4_BIOGENIC = 27.2
    N2O = 273
    SF6 = 25200
    HFC_134A = 1430
    HFC_32 = 675
    R410A = 2088
    R404A = 4728
    R407C = 1774
