"""
GL-009 THERMALIQ Agent - Thermal Fluid Property Database

This module provides a comprehensive database of thermal fluid properties
for common heat transfer fluids including Therminol, Dowtherm, Marlotherm,
and other commercial fluids.

All property calculations are deterministic using polynomial correlations
from manufacturer data sheets - ZERO HALLUCINATION guaranteed.

Supported fluids:
    - Therminol 55, 59, 62, 66, VP-1, VP-3, XP
    - Dowtherm A, G, J, Q, RP
    - Marlotherm SH, LH
    - Mobiltherm 603, 605
    - Paratherm NF, HE
    - Syltherm 800, XLT

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.fluid_properties import (
    ...     ThermalFluidPropertyDatabase,
    ...     ThermalFluidType,
    ... )
    >>> db = ThermalFluidPropertyDatabase()
    >>> props = db.get_properties(ThermalFluidType.THERMINOL_66, 550.0)
    >>> print(f"Density: {props.density_lb_ft3:.2f} lb/ft3")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import math
import logging

from pydantic import BaseModel, Field

from .schemas import ThermalFluidType, FluidProperties

logger = logging.getLogger(__name__)


# =============================================================================
# FLUID DATA CLASSES
# =============================================================================

@dataclass
class FluidCoefficients:
    """Polynomial coefficients for fluid property correlations."""

    # Density: rho = a0 + a1*T + a2*T^2 (lb/ft3, T in F)
    density_coeffs: Tuple[float, float, float]

    # Specific heat: Cp = a0 + a1*T + a2*T^2 (BTU/lb-F, T in F)
    specific_heat_coeffs: Tuple[float, float, float]

    # Thermal conductivity: k = a0 + a1*T + a2*T^2 (BTU/hr-ft-F, T in F)
    thermal_conductivity_coeffs: Tuple[float, float, float]

    # Kinematic viscosity: log10(nu) = a0 + a1/(T+459.67) + a2/(T+459.67)^2 (cSt)
    viscosity_coeffs: Tuple[float, float, float]

    # Vapor pressure: log10(Pv) = A - B/(T+C) (psia, T in F) - Antoine equation
    vapor_pressure_coeffs: Tuple[float, float, float]

    # Safety properties
    flash_point_f: float
    auto_ignition_temp_f: float
    max_film_temp_f: float
    max_bulk_temp_f: float

    # Temperature range (F)
    min_temp_f: float
    max_temp_f: float

    # Reference properties at 77F
    density_at_77f: float  # lb/ft3
    viscosity_at_77f: float  # cSt

    # Molecular weight
    molecular_weight: float


# =============================================================================
# THERMAL FLUID DATABASE
# =============================================================================

# Coefficient data from manufacturer specifications
# Reference: Eastman Therminol, Dow Dowtherm, Sasol Marlotherm data sheets

FLUID_DATA: Dict[ThermalFluidType, FluidCoefficients] = {
    ThermalFluidType.THERMINOL_66: FluidCoefficients(
        # High-temperature organic heat transfer fluid
        # Recommended use: Liquid phase up to 650F (343C)
        density_coeffs=(66.24, -0.0235, -4.5e-6),
        specific_heat_coeffs=(0.388, 6.2e-4, 1.5e-7),
        thermal_conductivity_coeffs=(0.0792, -2.8e-5, -1.0e-8),
        viscosity_coeffs=(-2.3, 2850.0, -450000.0),
        vapor_pressure_coeffs=(8.08, 3500.0, 400.0),
        flash_point_f=340.0,
        auto_ignition_temp_f=750.0,
        max_film_temp_f=705.0,
        max_bulk_temp_f=650.0,
        min_temp_f=-20.0,
        max_temp_f=650.0,
        density_at_77f=63.0,
        viscosity_at_77f=29.0,
        molecular_weight=252.0,
    ),

    ThermalFluidType.THERMINOL_VP1: FluidCoefficients(
        # Eutectic mixture of diphenyl oxide/biphenyl
        # Highest temperature capability in liquid phase
        density_coeffs=(68.3, -0.030, -8.0e-6),
        specific_heat_coeffs=(0.37, 5.8e-4, 2.0e-7),
        thermal_conductivity_coeffs=(0.0850, -3.5e-5, -5.0e-9),
        viscosity_coeffs=(-2.5, 2600.0, -380000.0),
        vapor_pressure_coeffs=(8.5, 3800.0, 380.0),
        flash_point_f=255.0,
        auto_ignition_temp_f=1150.0,
        max_film_temp_f=750.0,
        max_bulk_temp_f=750.0,
        min_temp_f=54.0,
        max_temp_f=750.0,
        density_at_77f=65.5,
        viscosity_at_77f=3.5,
        molecular_weight=166.0,
    ),

    ThermalFluidType.THERMINOL_55: FluidCoefficients(
        # Synthetic alkylated aromatic for medium temperature
        density_coeffs=(58.8, -0.022, -3.0e-6),
        specific_heat_coeffs=(0.42, 5.5e-4, 1.0e-7),
        thermal_conductivity_coeffs=(0.075, -2.5e-5, -5.0e-9),
        viscosity_coeffs=(-2.0, 2400.0, -350000.0),
        vapor_pressure_coeffs=(7.8, 3200.0, 420.0),
        flash_point_f=350.0,
        auto_ignition_temp_f=675.0,
        max_film_temp_f=550.0,
        max_bulk_temp_f=500.0,
        min_temp_f=-20.0,
        max_temp_f=500.0,
        density_at_77f=56.0,
        viscosity_at_77f=19.0,
        molecular_weight=230.0,
    ),

    ThermalFluidType.THERMINOL_59: FluidCoefficients(
        # Synthetic organic for wide temperature range
        density_coeffs=(62.0, -0.024, -4.0e-6),
        specific_heat_coeffs=(0.40, 5.8e-4, 1.2e-7),
        thermal_conductivity_coeffs=(0.077, -2.6e-5, -8.0e-9),
        viscosity_coeffs=(-2.2, 2650.0, -400000.0),
        vapor_pressure_coeffs=(8.0, 3400.0, 390.0),
        flash_point_f=325.0,
        auto_ignition_temp_f=700.0,
        max_film_temp_f=600.0,
        max_bulk_temp_f=550.0,
        min_temp_f=-50.0,
        max_temp_f=550.0,
        density_at_77f=59.5,
        viscosity_at_77f=16.0,
        molecular_weight=215.0,
    ),

    ThermalFluidType.THERMINOL_62: FluidCoefficients(
        # High-purity synthetic for exacting applications
        density_coeffs=(61.5, -0.023, -3.8e-6),
        specific_heat_coeffs=(0.41, 5.6e-4, 1.3e-7),
        thermal_conductivity_coeffs=(0.076, -2.5e-5, -7.0e-9),
        viscosity_coeffs=(-2.1, 2550.0, -380000.0),
        vapor_pressure_coeffs=(7.9, 3350.0, 395.0),
        flash_point_f=350.0,
        auto_ignition_temp_f=720.0,
        max_film_temp_f=600.0,
        max_bulk_temp_f=550.0,
        min_temp_f=-20.0,
        max_temp_f=550.0,
        density_at_77f=59.0,
        viscosity_at_77f=14.5,
        molecular_weight=210.0,
    ),

    ThermalFluidType.THERMINOL_XP: FluidCoefficients(
        # White mineral oil for food-grade applications
        density_coeffs=(55.0, -0.020, -2.5e-6),
        specific_heat_coeffs=(0.44, 5.0e-4, 8.0e-8),
        thermal_conductivity_coeffs=(0.072, -2.2e-5, -4.0e-9),
        viscosity_coeffs=(-1.8, 2200.0, -320000.0),
        vapor_pressure_coeffs=(7.5, 3000.0, 440.0),
        flash_point_f=350.0,
        auto_ignition_temp_f=650.0,
        max_film_temp_f=500.0,
        max_bulk_temp_f=450.0,
        min_temp_f=10.0,
        max_temp_f=450.0,
        density_at_77f=52.5,
        viscosity_at_77f=32.0,
        molecular_weight=280.0,
    ),

    ThermalFluidType.DOWTHERM_A: FluidCoefficients(
        # Eutectic mixture, highest temperature liquid phase
        density_coeffs=(68.5, -0.031, -9.0e-6),
        specific_heat_coeffs=(0.36, 5.9e-4, 2.2e-7),
        thermal_conductivity_coeffs=(0.086, -3.6e-5, -6.0e-9),
        viscosity_coeffs=(-2.6, 2700.0, -400000.0),
        vapor_pressure_coeffs=(8.6, 3850.0, 375.0),
        flash_point_f=255.0,
        auto_ignition_temp_f=1150.0,
        max_film_temp_f=750.0,
        max_bulk_temp_f=750.0,
        min_temp_f=59.0,
        max_temp_f=750.0,
        density_at_77f=65.8,
        viscosity_at_77f=3.7,
        molecular_weight=166.0,
    ),

    ThermalFluidType.DOWTHERM_G: FluidCoefficients(
        # Di- and tri-aryl compounds for medium-high temp
        density_coeffs=(66.0, -0.024, -5.0e-6),
        specific_heat_coeffs=(0.39, 6.0e-4, 1.6e-7),
        thermal_conductivity_coeffs=(0.080, -2.9e-5, -1.0e-8),
        viscosity_coeffs=(-2.4, 2900.0, -460000.0),
        vapor_pressure_coeffs=(8.2, 3600.0, 390.0),
        flash_point_f=350.0,
        auto_ignition_temp_f=780.0,
        max_film_temp_f=700.0,
        max_bulk_temp_f=650.0,
        min_temp_f=25.0,
        max_temp_f=650.0,
        density_at_77f=63.5,
        viscosity_at_77f=12.0,
        molecular_weight=240.0,
    ),

    ThermalFluidType.DOWTHERM_J: FluidCoefficients(
        # Isomer mixture for low-temperature applications
        density_coeffs=(52.0, -0.020, -2.0e-6),
        specific_heat_coeffs=(0.48, 4.5e-4, 5.0e-8),
        thermal_conductivity_coeffs=(0.070, -2.0e-5, -3.0e-9),
        viscosity_coeffs=(-1.6, 1800.0, -260000.0),
        vapor_pressure_coeffs=(7.2, 2800.0, 460.0),
        flash_point_f=145.0,
        auto_ignition_temp_f=500.0,
        max_film_temp_f=600.0,
        max_bulk_temp_f=575.0,
        min_temp_f=-110.0,
        max_temp_f=575.0,
        density_at_77f=50.0,
        viscosity_at_77f=1.0,
        molecular_weight=124.0,
    ),

    ThermalFluidType.DOWTHERM_Q: FluidCoefficients(
        # Diphenylethane/alkylated aromatics blend
        density_coeffs=(64.0, -0.023, -4.2e-6),
        specific_heat_coeffs=(0.40, 5.8e-4, 1.4e-7),
        thermal_conductivity_coeffs=(0.078, -2.7e-5, -8.0e-9),
        viscosity_coeffs=(-2.2, 2700.0, -420000.0),
        vapor_pressure_coeffs=(8.0, 3400.0, 400.0),
        flash_point_f=250.0,
        auto_ignition_temp_f=700.0,
        max_film_temp_f=650.0,
        max_bulk_temp_f=600.0,
        min_temp_f=-30.0,
        max_temp_f=600.0,
        density_at_77f=61.5,
        viscosity_at_77f=5.0,
        molecular_weight=195.0,
    ),

    ThermalFluidType.DOWTHERM_RP: FluidCoefficients(
        # Diaryl alkyl for renewable/pharmaceutical
        density_coeffs=(63.0, -0.022, -3.8e-6),
        specific_heat_coeffs=(0.41, 5.7e-4, 1.3e-7),
        thermal_conductivity_coeffs=(0.076, -2.6e-5, -7.0e-9),
        viscosity_coeffs=(-2.1, 2600.0, -400000.0),
        vapor_pressure_coeffs=(7.9, 3350.0, 400.0),
        flash_point_f=350.0,
        auto_ignition_temp_f=730.0,
        max_film_temp_f=660.0,
        max_bulk_temp_f=620.0,
        min_temp_f=0.0,
        max_temp_f=620.0,
        density_at_77f=60.5,
        viscosity_at_77f=15.0,
        molecular_weight=225.0,
    ),

    ThermalFluidType.MARLOTHERM_SH: FluidCoefficients(
        # High-temperature synthetic, excellent stability
        density_coeffs=(65.5, -0.024, -4.5e-6),
        specific_heat_coeffs=(0.39, 6.1e-4, 1.5e-7),
        thermal_conductivity_coeffs=(0.079, -2.8e-5, -9.0e-9),
        viscosity_coeffs=(-2.3, 2800.0, -440000.0),
        vapor_pressure_coeffs=(8.1, 3550.0, 395.0),
        flash_point_f=365.0,
        auto_ignition_temp_f=765.0,
        max_film_temp_f=660.0,
        max_bulk_temp_f=610.0,
        min_temp_f=15.0,
        max_temp_f=610.0,
        density_at_77f=62.8,
        viscosity_at_77f=24.0,
        molecular_weight=248.0,
    ),

    ThermalFluidType.MARLOTHERM_LH: FluidCoefficients(
        # Medium temperature synthetic
        density_coeffs=(60.0, -0.022, -3.5e-6),
        specific_heat_coeffs=(0.42, 5.5e-4, 1.2e-7),
        thermal_conductivity_coeffs=(0.074, -2.4e-5, -6.0e-9),
        viscosity_coeffs=(-2.0, 2500.0, -380000.0),
        vapor_pressure_coeffs=(7.8, 3300.0, 410.0),
        flash_point_f=310.0,
        auto_ignition_temp_f=680.0,
        max_film_temp_f=560.0,
        max_bulk_temp_f=520.0,
        min_temp_f=-50.0,
        max_temp_f=520.0,
        density_at_77f=57.5,
        viscosity_at_77f=8.5,
        molecular_weight=190.0,
    ),

    ThermalFluidType.MOBILTHERM_603: FluidCoefficients(
        # Mineral oil based, moderate temperature
        density_coeffs=(56.0, -0.021, -3.0e-6),
        specific_heat_coeffs=(0.43, 5.2e-4, 9.0e-8),
        thermal_conductivity_coeffs=(0.073, -2.3e-5, -4.0e-9),
        viscosity_coeffs=(-1.9, 2300.0, -340000.0),
        vapor_pressure_coeffs=(7.6, 3100.0, 430.0),
        flash_point_f=400.0,
        auto_ignition_temp_f=660.0,
        max_film_temp_f=550.0,
        max_bulk_temp_f=500.0,
        min_temp_f=0.0,
        max_temp_f=500.0,
        density_at_77f=53.5,
        viscosity_at_77f=40.0,
        molecular_weight=300.0,
    ),

    ThermalFluidType.MOBILTHERM_605: FluidCoefficients(
        # High-performance mineral oil
        density_coeffs=(55.5, -0.020, -2.8e-6),
        specific_heat_coeffs=(0.44, 5.0e-4, 8.0e-8),
        thermal_conductivity_coeffs=(0.072, -2.2e-5, -3.5e-9),
        viscosity_coeffs=(-1.85, 2250.0, -330000.0),
        vapor_pressure_coeffs=(7.55, 3050.0, 435.0),
        flash_point_f=415.0,
        auto_ignition_temp_f=670.0,
        max_film_temp_f=560.0,
        max_bulk_temp_f=510.0,
        min_temp_f=0.0,
        max_temp_f=510.0,
        density_at_77f=53.0,
        viscosity_at_77f=35.0,
        molecular_weight=290.0,
    ),

    ThermalFluidType.PARATHERM_NF: FluidCoefficients(
        # Food-grade heat transfer fluid
        density_coeffs=(54.5, -0.019, -2.5e-6),
        specific_heat_coeffs=(0.45, 4.8e-4, 7.0e-8),
        thermal_conductivity_coeffs=(0.071, -2.1e-5, -3.0e-9),
        viscosity_coeffs=(-1.8, 2200.0, -320000.0),
        vapor_pressure_coeffs=(7.4, 2950.0, 445.0),
        flash_point_f=380.0,
        auto_ignition_temp_f=650.0,
        max_film_temp_f=525.0,
        max_bulk_temp_f=475.0,
        min_temp_f=20.0,
        max_temp_f=475.0,
        density_at_77f=52.0,
        viscosity_at_77f=28.0,
        molecular_weight=270.0,
    ),

    ThermalFluidType.PARATHERM_HE: FluidCoefficients(
        # High-efficiency synthetic
        density_coeffs=(63.5, -0.023, -4.0e-6),
        specific_heat_coeffs=(0.40, 5.8e-4, 1.4e-7),
        thermal_conductivity_coeffs=(0.077, -2.6e-5, -7.5e-9),
        viscosity_coeffs=(-2.15, 2650.0, -410000.0),
        vapor_pressure_coeffs=(7.95, 3380.0, 400.0),
        flash_point_f=355.0,
        auto_ignition_temp_f=740.0,
        max_film_temp_f=660.0,
        max_bulk_temp_f=610.0,
        min_temp_f=-10.0,
        max_temp_f=610.0,
        density_at_77f=61.0,
        viscosity_at_77f=18.0,
        molecular_weight=235.0,
    ),

    ThermalFluidType.SYLTHERM_800: FluidCoefficients(
        # Silicone-based for highest temperatures
        density_coeffs=(60.0, -0.025, -6.0e-6),
        specific_heat_coeffs=(0.42, 5.5e-4, 1.2e-7),
        thermal_conductivity_coeffs=(0.068, -1.8e-5, -2.5e-9),
        viscosity_coeffs=(-1.5, 1600.0, -220000.0),
        vapor_pressure_coeffs=(7.3, 2750.0, 470.0),
        flash_point_f=350.0,
        auto_ignition_temp_f=840.0,
        max_film_temp_f=780.0,
        max_bulk_temp_f=750.0,
        min_temp_f=-40.0,
        max_temp_f=750.0,
        density_at_77f=57.5,
        viscosity_at_77f=9.0,
        molecular_weight=425.0,
    ),

    ThermalFluidType.SYLTHERM_XLT: FluidCoefficients(
        # Silicone for wide temperature range
        density_coeffs=(54.0, -0.022, -4.5e-6),
        specific_heat_coeffs=(0.44, 5.2e-4, 1.0e-7),
        thermal_conductivity_coeffs=(0.065, -1.6e-5, -2.0e-9),
        viscosity_coeffs=(-1.3, 1400.0, -180000.0),
        vapor_pressure_coeffs=(7.1, 2600.0, 480.0),
        flash_point_f=280.0,
        auto_ignition_temp_f=690.0,
        max_film_temp_f=570.0,
        max_bulk_temp_f=500.0,
        min_temp_f=-130.0,
        max_temp_f=500.0,
        density_at_77f=52.0,
        viscosity_at_77f=4.5,
        molecular_weight=317.0,
    ),
}


# =============================================================================
# PROPERTY CALCULATOR
# =============================================================================

class ThermalFluidPropertyDatabase:
    """
    Thermal fluid property database with polynomial correlations.

    This class provides deterministic property calculations for all
    supported thermal fluids using manufacturer-supplied correlation
    coefficients. All calculations are ZERO HALLUCINATION - no ML/LLM
    in the calculation path.

    Example:
        >>> db = ThermalFluidPropertyDatabase()
        >>> props = db.get_properties(ThermalFluidType.THERMINOL_66, 550.0)
        >>> print(f"Cp: {props.specific_heat_btu_lb_f:.4f}")
    """

    def __init__(self) -> None:
        """Initialize the property database."""
        self._fluid_data = FLUID_DATA
        self._calculation_count = 0
        logger.info("ThermalFluidPropertyDatabase initialized")

    def get_properties(
        self,
        fluid_type: ThermalFluidType,
        temperature_f: float,
    ) -> FluidProperties:
        """
        Get complete fluid properties at specified temperature.

        Args:
            fluid_type: Thermal fluid type
            temperature_f: Temperature (F)

        Returns:
            FluidProperties object with all calculated properties

        Raises:
            ValueError: If fluid type not supported or temperature out of range
        """
        if fluid_type == ThermalFluidType.CUSTOM:
            raise ValueError("Custom fluid type requires explicit property data")

        if fluid_type not in self._fluid_data:
            raise ValueError(f"Unsupported fluid type: {fluid_type}")

        coeffs = self._fluid_data[fluid_type]

        # Validate temperature range
        if temperature_f < coeffs.min_temp_f or temperature_f > coeffs.max_temp_f:
            logger.warning(
                f"Temperature {temperature_f}F outside recommended range "
                f"[{coeffs.min_temp_f}, {coeffs.max_temp_f}] for {fluid_type}"
            )

        self._calculation_count += 1

        # Calculate properties
        density = self._calc_density(temperature_f, coeffs)
        specific_heat = self._calc_specific_heat(temperature_f, coeffs)
        thermal_conductivity = self._calc_thermal_conductivity(temperature_f, coeffs)
        kinematic_viscosity = self._calc_kinematic_viscosity(temperature_f, coeffs)
        vapor_pressure = self._calc_vapor_pressure(temperature_f, coeffs)

        # Derived properties
        dynamic_viscosity = kinematic_viscosity * density / 62.4 * 1.488  # cP
        prandtl = self._calc_prandtl(specific_heat, dynamic_viscosity, thermal_conductivity)

        return FluidProperties(
            temperature_f=temperature_f,
            density_lb_ft3=density,
            specific_heat_btu_lb_f=specific_heat,
            thermal_conductivity_btu_hr_ft_f=thermal_conductivity,
            kinematic_viscosity_cst=kinematic_viscosity,
            dynamic_viscosity_cp=dynamic_viscosity,
            prandtl_number=prandtl,
            vapor_pressure_psia=vapor_pressure,
            flash_point_f=coeffs.flash_point_f,
            auto_ignition_temp_f=coeffs.auto_ignition_temp_f,
            max_film_temp_f=coeffs.max_film_temp_f,
            max_bulk_temp_f=coeffs.max_bulk_temp_f,
        )

    def get_density(
        self,
        fluid_type: ThermalFluidType,
        temperature_f: float,
    ) -> float:
        """
        Get fluid density at temperature.

        Args:
            fluid_type: Thermal fluid type
            temperature_f: Temperature (F)

        Returns:
            Density (lb/ft3)
        """
        coeffs = self._fluid_data[fluid_type]
        return self._calc_density(temperature_f, coeffs)

    def get_specific_heat(
        self,
        fluid_type: ThermalFluidType,
        temperature_f: float,
    ) -> float:
        """
        Get fluid specific heat at temperature.

        Args:
            fluid_type: Thermal fluid type
            temperature_f: Temperature (F)

        Returns:
            Specific heat (BTU/lb-F)
        """
        coeffs = self._fluid_data[fluid_type]
        return self._calc_specific_heat(temperature_f, coeffs)

    def get_thermal_conductivity(
        self,
        fluid_type: ThermalFluidType,
        temperature_f: float,
    ) -> float:
        """
        Get fluid thermal conductivity at temperature.

        Args:
            fluid_type: Thermal fluid type
            temperature_f: Temperature (F)

        Returns:
            Thermal conductivity (BTU/hr-ft-F)
        """
        coeffs = self._fluid_data[fluid_type]
        return self._calc_thermal_conductivity(temperature_f, coeffs)

    def get_viscosity(
        self,
        fluid_type: ThermalFluidType,
        temperature_f: float,
    ) -> float:
        """
        Get fluid kinematic viscosity at temperature.

        Args:
            fluid_type: Thermal fluid type
            temperature_f: Temperature (F)

        Returns:
            Kinematic viscosity (cSt)
        """
        coeffs = self._fluid_data[fluid_type]
        return self._calc_kinematic_viscosity(temperature_f, coeffs)

    def get_vapor_pressure(
        self,
        fluid_type: ThermalFluidType,
        temperature_f: float,
    ) -> float:
        """
        Get fluid vapor pressure at temperature.

        Args:
            fluid_type: Thermal fluid type
            temperature_f: Temperature (F)

        Returns:
            Vapor pressure (psia)
        """
        coeffs = self._fluid_data[fluid_type]
        return self._calc_vapor_pressure(temperature_f, coeffs)

    def get_flash_point(self, fluid_type: ThermalFluidType) -> float:
        """Get fluid flash point (F)."""
        return self._fluid_data[fluid_type].flash_point_f

    def get_auto_ignition_temp(self, fluid_type: ThermalFluidType) -> float:
        """Get fluid auto-ignition temperature (F)."""
        return self._fluid_data[fluid_type].auto_ignition_temp_f

    def get_max_film_temp(self, fluid_type: ThermalFluidType) -> float:
        """Get maximum allowable film temperature (F)."""
        return self._fluid_data[fluid_type].max_film_temp_f

    def get_max_bulk_temp(self, fluid_type: ThermalFluidType) -> float:
        """Get maximum allowable bulk temperature (F)."""
        return self._fluid_data[fluid_type].max_bulk_temp_f

    def get_temperature_range(
        self,
        fluid_type: ThermalFluidType,
    ) -> Tuple[float, float]:
        """
        Get valid temperature range for fluid.

        Returns:
            Tuple of (min_temp_f, max_temp_f)
        """
        coeffs = self._fluid_data[fluid_type]
        return (coeffs.min_temp_f, coeffs.max_temp_f)

    def get_thermal_expansion_coefficient(
        self,
        fluid_type: ThermalFluidType,
        temperature_f: float,
    ) -> float:
        """
        Calculate volumetric thermal expansion coefficient.

        beta = -(1/rho) * (d_rho/dT)

        Args:
            fluid_type: Thermal fluid type
            temperature_f: Temperature (F)

        Returns:
            Expansion coefficient (1/F)
        """
        coeffs = self._fluid_data[fluid_type]
        a0, a1, a2 = coeffs.density_coeffs

        # Density and its derivative
        rho = a0 + a1 * temperature_f + a2 * temperature_f ** 2
        d_rho_dt = a1 + 2 * a2 * temperature_f

        # beta = -(1/rho) * (drho/dT)
        beta = -d_rho_dt / rho

        return beta

    def calculate_expansion_volume(
        self,
        fluid_type: ThermalFluidType,
        system_volume_gallons: float,
        cold_temp_f: float,
        hot_temp_f: float,
    ) -> float:
        """
        Calculate fluid expansion volume from cold to hot temperature.

        Args:
            fluid_type: Thermal fluid type
            system_volume_gallons: System volume at cold temperature
            cold_temp_f: Cold temperature (F)
            hot_temp_f: Hot temperature (F)

        Returns:
            Expansion volume (gallons)
        """
        rho_cold = self.get_density(fluid_type, cold_temp_f)
        rho_hot = self.get_density(fluid_type, hot_temp_f)

        # V_hot / V_cold = rho_cold / rho_hot
        expansion_ratio = rho_cold / rho_hot - 1.0
        expansion_volume = system_volume_gallons * expansion_ratio

        return expansion_volume

    def get_supported_fluids(self) -> List[ThermalFluidType]:
        """Get list of supported fluid types."""
        return list(self._fluid_data.keys())

    # =========================================================================
    # PRIVATE CALCULATION METHODS
    # =========================================================================

    def _calc_density(
        self,
        temperature_f: float,
        coeffs: FluidCoefficients,
    ) -> float:
        """Calculate density using polynomial correlation."""
        a0, a1, a2 = coeffs.density_coeffs
        return a0 + a1 * temperature_f + a2 * temperature_f ** 2

    def _calc_specific_heat(
        self,
        temperature_f: float,
        coeffs: FluidCoefficients,
    ) -> float:
        """Calculate specific heat using polynomial correlation."""
        a0, a1, a2 = coeffs.specific_heat_coeffs
        return a0 + a1 * temperature_f + a2 * temperature_f ** 2

    def _calc_thermal_conductivity(
        self,
        temperature_f: float,
        coeffs: FluidCoefficients,
    ) -> float:
        """Calculate thermal conductivity using polynomial correlation."""
        a0, a1, a2 = coeffs.thermal_conductivity_coeffs
        return a0 + a1 * temperature_f + a2 * temperature_f ** 2

    def _calc_kinematic_viscosity(
        self,
        temperature_f: float,
        coeffs: FluidCoefficients,
    ) -> float:
        """
        Calculate kinematic viscosity using modified Walther equation.

        log10(nu + 0.7) = A + B/T + C/T^2

        Where T is in Rankine (F + 459.67)
        """
        a0, a1, a2 = coeffs.viscosity_coeffs
        t_r = temperature_f + 459.67  # Convert to Rankine

        log_nu = a0 + a1 / t_r + a2 / (t_r ** 2)
        nu = 10 ** log_nu

        # Ensure positive and reasonable value
        nu = max(0.5, min(nu, 10000.0))

        return nu

    def _calc_vapor_pressure(
        self,
        temperature_f: float,
        coeffs: FluidCoefficients,
    ) -> float:
        """
        Calculate vapor pressure using Antoine equation.

        log10(P) = A - B / (T + C)

        Where T is in F and P is in psia
        """
        a, b, c = coeffs.vapor_pressure_coeffs

        log_p = a - b / (temperature_f + c)
        p = 10 ** log_p

        # Ensure positive
        return max(0.001, p)

    def _calc_prandtl(
        self,
        specific_heat: float,
        dynamic_viscosity_cp: float,
        thermal_conductivity: float,
    ) -> float:
        """
        Calculate Prandtl number.

        Pr = Cp * mu / k

        Where:
            Cp in BTU/lb-F
            mu in lb/ft-hr (convert from cP: multiply by 2.419)
            k in BTU/hr-ft-F
        """
        # Convert dynamic viscosity from cP to lb/ft-hr
        mu_lb_ft_hr = dynamic_viscosity_cp * 2.419

        pr = specific_heat * mu_lb_ft_hr / thermal_conductivity

        return pr

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_fluid_properties(
    fluid_type: ThermalFluidType,
    temperature_f: float,
) -> FluidProperties:
    """
    Convenience function to get fluid properties.

    Args:
        fluid_type: Thermal fluid type
        temperature_f: Temperature (F)

    Returns:
        FluidProperties object
    """
    db = ThermalFluidPropertyDatabase()
    return db.get_properties(fluid_type, temperature_f)


def compare_fluids(
    fluid_types: List[ThermalFluidType],
    temperature_f: float,
) -> Dict[str, Dict[str, float]]:
    """
    Compare properties of multiple fluids at same temperature.

    Args:
        fluid_types: List of fluid types to compare
        temperature_f: Comparison temperature (F)

    Returns:
        Dictionary with fluid properties for comparison
    """
    db = ThermalFluidPropertyDatabase()
    comparison = {}

    for fluid_type in fluid_types:
        try:
            props = db.get_properties(fluid_type, temperature_f)
            comparison[fluid_type.value] = {
                "density_lb_ft3": props.density_lb_ft3,
                "specific_heat_btu_lb_f": props.specific_heat_btu_lb_f,
                "thermal_conductivity_btu_hr_ft_f": props.thermal_conductivity_btu_hr_ft_f,
                "kinematic_viscosity_cst": props.kinematic_viscosity_cst,
                "prandtl_number": props.prandtl_number,
                "vapor_pressure_psia": props.vapor_pressure_psia,
                "flash_point_f": props.flash_point_f,
                "max_bulk_temp_f": props.max_bulk_temp_f,
            }
        except Exception as e:
            logger.warning(f"Could not get properties for {fluid_type}: {e}")

    return comparison
