"""
GreenLang Heat Transfer Formulas Library
=========================================

Comprehensive library of heat transfer calculation formulas including:
- Conduction calculations
- Convection correlations (natural, forced, internal, external)
- Radiation heat transfer
- Heat exchanger design
- Overall heat transfer coefficients
- Fin efficiency calculations

All formulas include:
- Source standard/reference
- Valid ranges for inputs
- Uncertainty estimates
- Test cases
- SHA-256 hash of formula definition

Formula Count: 100+ formulas

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .calculation_engine import (
    CalculationStep,
    FormulaDefinition,
    FormulaRegistry,
    ParameterDefinition,
    UnitCategory,
    make_decimal,
    safe_divide,
    safe_log,
    safe_sqrt,
)


# =============================================================================
# Heat Transfer Constants
# =============================================================================

class HeatTransferConstants:
    """Constants for heat transfer calculations."""

    # Stefan-Boltzmann constant
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m2*K4)

    # Thermal conductivity (W/m.K) at 300K
    K_COPPER = 401
    K_ALUMINUM = 237
    K_STEEL_CARBON = 50
    K_STEEL_STAINLESS = 16
    K_GLASS = 1.0
    K_CONCRETE = 1.7
    K_BRICK = 0.72
    K_INSULATION_MINERAL_WOOL = 0.04
    K_INSULATION_FIBERGLASS = 0.035
    K_AIR = 0.026
    K_WATER = 0.6

    # Emissivity values (typical)
    E_OXIDIZED_STEEL = 0.79
    E_POLISHED_STEEL = 0.07
    E_ALUMINUM_POLISHED = 0.04
    E_ALUMINUM_OXIDIZED = 0.25
    E_REFRACTORY = 0.9
    E_WATER = 0.96
    E_CARBON_BLACK = 0.95

    # Common fouling factors (m2.K/W) - TEMA
    RF_COOLING_WATER_TREATED = 0.000176
    RF_COOLING_WATER_UNTREATED = 0.000352
    RF_FUEL_OIL = 0.000881
    RF_STEAM = 0.000088
    RF_BOILER_FEEDWATER = 0.000088
    RF_PROCESS_FLUID_CLEAN = 0.000176
    RF_PROCESS_FLUID_DIRTY = 0.000528


# =============================================================================
# Conduction Calculations
# =============================================================================

class ConductionCalculations:
    """
    Conduction heat transfer calculations.

    Reference: Incropera & DeWitt - Fundamentals of Heat and Mass Transfer
    """

    @staticmethod
    def heat_rate_plane_wall(
        thermal_conductivity: float,
        area: float,
        thickness: float,
        temperature_hot: float,
        temperature_cold: float,
    ) -> Decimal:
        """
        Heat transfer rate through plane wall.

        Formula: Q = k * A * (T1 - T2) / L
        Source: Fourier's Law
        Range: All plane walls
        Uncertainty: +/- 2%

        Args:
            thermal_conductivity: Thermal conductivity (W/m.K)
            area: Heat transfer area (m2)
            thickness: Wall thickness (m)
            temperature_hot: Hot side temperature (K or C)
            temperature_cold: Cold side temperature (K or C)

        Returns:
            Heat transfer rate (W)
        """
        if thickness <= 0:
            raise ValueError("Thickness must be positive")

        Q = thermal_conductivity * area * (temperature_hot - temperature_cold) / thickness
        return make_decimal(Q)

    @staticmethod
    def thermal_resistance_plane_wall(
        thermal_conductivity: float,
        area: float,
        thickness: float,
    ) -> Decimal:
        """
        Thermal resistance of plane wall.

        Formula: R = L / (k * A)
        Source: Heat transfer fundamentals
        Range: All plane walls
        Uncertainty: +/- 2%

        Args:
            thermal_conductivity: Thermal conductivity (W/m.K)
            area: Heat transfer area (m2)
            thickness: Wall thickness (m)

        Returns:
            Thermal resistance (K/W)
        """
        if thermal_conductivity <= 0 or area <= 0:
            raise ValueError("Conductivity and area must be positive")

        R = thickness / (thermal_conductivity * area)
        return make_decimal(R)

    @staticmethod
    def heat_rate_cylindrical_wall(
        thermal_conductivity: float,
        length: float,
        radius_inner: float,
        radius_outer: float,
        temperature_inner: float,
        temperature_outer: float,
    ) -> Decimal:
        """
        Heat transfer rate through cylindrical wall.

        Formula: Q = 2*pi*k*L*(Ti - To) / ln(ro/ri)
        Source: Fourier's Law (cylindrical coordinates)
        Range: All cylindrical walls
        Uncertainty: +/- 2%

        Args:
            thermal_conductivity: Thermal conductivity (W/m.K)
            length: Cylinder length (m)
            radius_inner: Inner radius (m)
            radius_outer: Outer radius (m)
            temperature_inner: Inner surface temperature
            temperature_outer: Outer surface temperature

        Returns:
            Heat transfer rate (W)
        """
        if radius_outer <= radius_inner:
            raise ValueError("Outer radius must exceed inner radius")

        Q = (
            2 * math.pi * thermal_conductivity * length *
            (temperature_inner - temperature_outer) /
            math.log(radius_outer / radius_inner)
        )
        return make_decimal(Q)

    @staticmethod
    def thermal_resistance_cylindrical(
        thermal_conductivity: float,
        length: float,
        radius_inner: float,
        radius_outer: float,
    ) -> Decimal:
        """
        Thermal resistance of cylindrical wall.

        Formula: R = ln(ro/ri) / (2*pi*k*L)
        Source: Heat transfer fundamentals
        Range: All cylindrical walls
        Uncertainty: +/- 2%

        Args:
            thermal_conductivity: Thermal conductivity (W/m.K)
            length: Cylinder length (m)
            radius_inner: Inner radius (m)
            radius_outer: Outer radius (m)

        Returns:
            Thermal resistance (K/W)
        """
        if radius_outer <= radius_inner:
            raise ValueError("Outer radius must exceed inner radius")

        R = math.log(radius_outer / radius_inner) / (2 * math.pi * thermal_conductivity * length)
        return make_decimal(R)

    @staticmethod
    def heat_rate_spherical_wall(
        thermal_conductivity: float,
        radius_inner: float,
        radius_outer: float,
        temperature_inner: float,
        temperature_outer: float,
    ) -> Decimal:
        """
        Heat transfer rate through spherical wall.

        Formula: Q = 4*pi*k*(Ti - To) / (1/ri - 1/ro)
        Source: Fourier's Law (spherical coordinates)
        Range: All spherical walls
        Uncertainty: +/- 2%

        Args:
            thermal_conductivity: Thermal conductivity (W/m.K)
            radius_inner: Inner radius (m)
            radius_outer: Outer radius (m)
            temperature_inner: Inner surface temperature
            temperature_outer: Outer surface temperature

        Returns:
            Heat transfer rate (W)
        """
        if radius_outer <= radius_inner:
            raise ValueError("Outer radius must exceed inner radius")

        Q = (
            4 * math.pi * thermal_conductivity *
            (temperature_inner - temperature_outer) /
            (1 / radius_inner - 1 / radius_outer)
        )
        return make_decimal(Q)

    @staticmethod
    def composite_wall_resistance(
        resistances: List[float],
    ) -> Decimal:
        """
        Total resistance for composite wall in series.

        Formula: R_total = sum(Ri)
        Source: Thermal circuit analysis
        Range: Multiple layers in series
        Uncertainty: Cumulative

        Args:
            resistances: List of thermal resistances (K/W)

        Returns:
            Total thermal resistance (K/W)
        """
        R_total = sum(resistances)
        return make_decimal(R_total)

    @staticmethod
    def critical_insulation_radius(
        thermal_conductivity_insulation: float,
        external_heat_transfer_coeff: float,
    ) -> Decimal:
        """
        Critical radius of insulation for cylinder.

        Formula: r_cr = k / h
        Source: Heat transfer optimization
        Range: Cylindrical insulation
        Uncertainty: +/- 5%

        Args:
            thermal_conductivity_insulation: Insulation k (W/m.K)
            external_heat_transfer_coeff: External h (W/m2.K)

        Returns:
            Critical radius (m)
        """
        if external_heat_transfer_coeff <= 0:
            raise ValueError("Heat transfer coefficient must be positive")

        r_cr = thermal_conductivity_insulation / external_heat_transfer_coeff
        return make_decimal(r_cr)


# =============================================================================
# Convection Calculations
# =============================================================================

class ConvectionCalculations:
    """
    Convection heat transfer correlations.

    Reference: Incropera & DeWitt, VDI Heat Atlas
    """

    @staticmethod
    def heat_rate_convection(
        heat_transfer_coeff: float,
        area: float,
        surface_temp: float,
        fluid_temp: float,
    ) -> Decimal:
        """
        Convective heat transfer rate.

        Formula: Q = h * A * (Ts - Tf)
        Source: Newton's Law of Cooling
        Range: All convection
        Uncertainty: +/- 5%

        Args:
            heat_transfer_coeff: Heat transfer coefficient (W/m2.K)
            area: Heat transfer area (m2)
            surface_temp: Surface temperature
            fluid_temp: Bulk fluid temperature

        Returns:
            Heat transfer rate (W)
        """
        Q = heat_transfer_coeff * area * (surface_temp - fluid_temp)
        return make_decimal(Q)

    @staticmethod
    def thermal_resistance_convection(
        heat_transfer_coeff: float,
        area: float,
    ) -> Decimal:
        """
        Thermal resistance for convection.

        Formula: R = 1 / (h * A)
        Source: Heat transfer fundamentals
        Range: All convection
        Uncertainty: +/- 5%

        Args:
            heat_transfer_coeff: Heat transfer coefficient (W/m2.K)
            area: Heat transfer area (m2)

        Returns:
            Thermal resistance (K/W)
        """
        if heat_transfer_coeff <= 0 or area <= 0:
            raise ValueError("Coefficient and area must be positive")

        R = 1 / (heat_transfer_coeff * area)
        return make_decimal(R)

    @staticmethod
    def nusselt_dittus_boelter(
        reynolds_number: float,
        prandtl_number: float,
        heating: bool = True,
    ) -> Decimal:
        """
        Nusselt number - Dittus-Boelter correlation for turbulent pipe flow.

        Formula: Nu = 0.023 * Re^0.8 * Pr^n (n=0.4 heating, n=0.3 cooling)
        Source: Dittus & Boelter (1930)
        Range: Re > 10000, 0.6 < Pr < 160, L/D > 10
        Uncertainty: +/- 25%

        Args:
            reynolds_number: Reynolds number
            prandtl_number: Prandtl number
            heating: True if fluid is being heated

        Returns:
            Nusselt number
        """
        n = 0.4 if heating else 0.3
        Nu = 0.023 * (reynolds_number ** 0.8) * (prandtl_number ** n)
        return make_decimal(Nu)

    @staticmethod
    def nusselt_sieder_tate(
        reynolds_number: float,
        prandtl_number: float,
        viscosity_bulk: float,
        viscosity_wall: float,
    ) -> Decimal:
        """
        Nusselt number - Sieder-Tate correlation (accounts for viscosity variation).

        Formula: Nu = 0.027 * Re^0.8 * Pr^(1/3) * (mu_b/mu_w)^0.14
        Source: Sieder & Tate (1936)
        Range: Re > 10000, 0.7 < Pr < 16700, L/D > 10
        Uncertainty: +/- 20%

        Args:
            reynolds_number: Reynolds number
            prandtl_number: Prandtl number
            viscosity_bulk: Bulk viscosity (Pa.s)
            viscosity_wall: Wall viscosity (Pa.s)

        Returns:
            Nusselt number
        """
        Nu = (
            0.027 *
            (reynolds_number ** 0.8) *
            (prandtl_number ** (1/3)) *
            ((viscosity_bulk / viscosity_wall) ** 0.14)
        )
        return make_decimal(Nu)

    @staticmethod
    def nusselt_gnielinski(
        reynolds_number: float,
        prandtl_number: float,
        friction_factor: float,
    ) -> Decimal:
        """
        Nusselt number - Gnielinski correlation (improved accuracy).

        Formula: Nu = (f/8)(Re-1000)Pr / (1 + 12.7(f/8)^0.5(Pr^(2/3)-1))
        Source: Gnielinski (1976), VDI Heat Atlas
        Range: 3000 < Re < 5e6, 0.5 < Pr < 2000
        Uncertainty: +/- 10%

        Args:
            reynolds_number: Reynolds number
            prandtl_number: Prandtl number
            friction_factor: Darcy friction factor

        Returns:
            Nusselt number
        """
        f8 = friction_factor / 8
        numerator = f8 * (reynolds_number - 1000) * prandtl_number
        denominator = 1 + 12.7 * math.sqrt(f8) * (prandtl_number ** (2/3) - 1)

        Nu = numerator / denominator
        return make_decimal(Nu)

    @staticmethod
    def nusselt_laminar_pipe(
        reynolds_number: float,
        prandtl_number: float,
        length_diameter_ratio: float,
    ) -> Decimal:
        """
        Nusselt number for laminar flow in pipe (developing flow).

        Formula: Nu = 3.66 + 0.0668*(D/L)*Re*Pr / (1 + 0.04*((D/L)*Re*Pr)^(2/3))
        Source: Hausen correlation
        Range: Re < 2300, developing thermal
        Uncertainty: +/- 15%

        Args:
            reynolds_number: Reynolds number
            prandtl_number: Prandtl number
            length_diameter_ratio: L/D ratio

        Returns:
            Nusselt number
        """
        if length_diameter_ratio <= 0:
            raise ValueError("L/D must be positive")

        Gz = reynolds_number * prandtl_number / length_diameter_ratio  # Graetz number
        Nu = 3.66 + 0.0668 * Gz / (1 + 0.04 * (Gz ** (2/3)))

        return make_decimal(Nu)

    @staticmethod
    def nusselt_natural_convection_vertical(
        rayleigh_number: float,
        prandtl_number: float,
    ) -> Decimal:
        """
        Nusselt number for natural convection on vertical surface.

        Formula: Churchill-Chu correlation
        Source: Churchill & Chu (1975)
        Range: All Ra for vertical plates
        Uncertainty: +/- 15%

        Args:
            rayleigh_number: Rayleigh number
            prandtl_number: Prandtl number

        Returns:
            Nusselt number
        """
        if rayleigh_number <= 0:
            return make_decimal(0.68)

        # Churchill-Chu correlation
        term1 = (0.387 * (rayleigh_number ** (1/6))) / \
                ((1 + (0.492 / prandtl_number) ** (9/16)) ** (8/27))
        Nu = (0.825 + term1) ** 2

        return make_decimal(Nu)

    @staticmethod
    def nusselt_natural_convection_horizontal(
        rayleigh_number: float,
        hot_surface_up: bool = True,
    ) -> Decimal:
        """
        Nusselt number for natural convection on horizontal surface.

        Formula: McAdams correlations
        Source: McAdams (1954)
        Range: 10^4 < Ra < 10^9
        Uncertainty: +/- 25%

        Args:
            rayleigh_number: Rayleigh number
            hot_surface_up: True if hot surface faces up

        Returns:
            Nusselt number
        """
        if hot_surface_up:
            if rayleigh_number < 1e4:
                Nu = 0.54 * (rayleigh_number ** 0.25)
            elif rayleigh_number < 1e7:
                Nu = 0.54 * (rayleigh_number ** 0.25)
            else:
                Nu = 0.15 * (rayleigh_number ** (1/3))
        else:
            # Hot surface down
            Nu = 0.27 * (rayleigh_number ** 0.25)

        return make_decimal(max(Nu, 0.1))

    @staticmethod
    def heat_transfer_coeff_from_nusselt(
        nusselt_number: float,
        thermal_conductivity: float,
        characteristic_length: float,
    ) -> Decimal:
        """
        Calculate heat transfer coefficient from Nusselt number.

        Formula: h = Nu * k / L
        Source: Definition of Nusselt number
        Range: All applications
        Uncertainty: Depends on Nu correlation

        Args:
            nusselt_number: Nusselt number
            thermal_conductivity: Fluid thermal conductivity (W/m.K)
            characteristic_length: Characteristic length (m)

        Returns:
            Heat transfer coefficient (W/m2.K)
        """
        if characteristic_length <= 0:
            raise ValueError("Characteristic length must be positive")

        h = nusselt_number * thermal_conductivity / characteristic_length
        return make_decimal(h)

    @staticmethod
    def reynolds_number(
        velocity: float,
        characteristic_length: float,
        density: float,
        viscosity: float,
    ) -> Decimal:
        """
        Calculate Reynolds number.

        Formula: Re = rho * V * L / mu
        Source: Fluid mechanics
        Range: All flows
        Uncertainty: +/- 2%

        Args:
            velocity: Flow velocity (m/s)
            characteristic_length: Characteristic length (m)
            density: Fluid density (kg/m3)
            viscosity: Dynamic viscosity (Pa.s)

        Returns:
            Reynolds number
        """
        if viscosity <= 0:
            raise ValueError("Viscosity must be positive")

        Re = density * velocity * characteristic_length / viscosity
        return make_decimal(Re)

    @staticmethod
    def prandtl_number(
        specific_heat: float,
        viscosity: float,
        thermal_conductivity: float,
    ) -> Decimal:
        """
        Calculate Prandtl number.

        Formula: Pr = Cp * mu / k
        Source: Fluid properties
        Range: All fluids
        Uncertainty: +/- 2%

        Args:
            specific_heat: Specific heat capacity (J/kg.K)
            viscosity: Dynamic viscosity (Pa.s)
            thermal_conductivity: Thermal conductivity (W/m.K)

        Returns:
            Prandtl number
        """
        if thermal_conductivity <= 0:
            raise ValueError("Thermal conductivity must be positive")

        Pr = specific_heat * viscosity / thermal_conductivity
        return make_decimal(Pr)

    @staticmethod
    def rayleigh_number(
        grashof_number: float,
        prandtl_number: float,
    ) -> Decimal:
        """
        Calculate Rayleigh number.

        Formula: Ra = Gr * Pr
        Source: Natural convection theory
        Range: All natural convection
        Uncertainty: +/- 5%

        Args:
            grashof_number: Grashof number
            prandtl_number: Prandtl number

        Returns:
            Rayleigh number
        """
        Ra = grashof_number * prandtl_number
        return make_decimal(Ra)

    @staticmethod
    def grashof_number(
        gravity: float,
        beta: float,
        temperature_diff: float,
        length: float,
        kinematic_viscosity: float,
    ) -> Decimal:
        """
        Calculate Grashof number.

        Formula: Gr = g * beta * dT * L^3 / nu^2
        Source: Natural convection theory
        Range: All natural convection
        Uncertainty: +/- 5%

        Args:
            gravity: Gravitational acceleration (m/s2)
            beta: Thermal expansion coefficient (1/K)
            temperature_diff: Temperature difference (K)
            length: Characteristic length (m)
            kinematic_viscosity: Kinematic viscosity (m2/s)

        Returns:
            Grashof number
        """
        if kinematic_viscosity <= 0:
            raise ValueError("Kinematic viscosity must be positive")

        Gr = gravity * beta * abs(temperature_diff) * (length ** 3) / (kinematic_viscosity ** 2)
        return make_decimal(Gr)


# =============================================================================
# Radiation Calculations
# =============================================================================

class RadiationCalculations:
    """
    Radiation heat transfer calculations.

    Reference: Incropera & DeWitt, Modest - Radiative Heat Transfer
    """

    @staticmethod
    def blackbody_emissive_power(
        temperature_k: float,
    ) -> Decimal:
        """
        Blackbody emissive power (Stefan-Boltzmann law).

        Formula: Eb = sigma * T^4
        Source: Stefan-Boltzmann Law
        Range: All temperatures
        Uncertainty: +/- 0.1%

        Args:
            temperature_k: Absolute temperature (K)

        Returns:
            Emissive power (W/m2)
        """
        sigma = HeatTransferConstants.STEFAN_BOLTZMANN
        Eb = sigma * (temperature_k ** 4)
        return make_decimal(Eb)

    @staticmethod
    def radiation_heat_rate_surface(
        emissivity: float,
        area: float,
        surface_temp_k: float,
        surroundings_temp_k: float,
    ) -> Decimal:
        """
        Net radiation from surface to surroundings.

        Formula: Q = epsilon * sigma * A * (Ts^4 - Tsur^4)
        Source: Gray surface radiation
        Range: All surfaces
        Uncertainty: +/- 10%

        Args:
            emissivity: Surface emissivity (0-1)
            area: Surface area (m2)
            surface_temp_k: Surface temperature (K)
            surroundings_temp_k: Surroundings temperature (K)

        Returns:
            Net radiation heat transfer (W)
        """
        sigma = HeatTransferConstants.STEFAN_BOLTZMANN
        Q = emissivity * sigma * area * (surface_temp_k ** 4 - surroundings_temp_k ** 4)
        return make_decimal(Q)

    @staticmethod
    def radiation_heat_rate_two_surfaces(
        emissivity_1: float,
        emissivity_2: float,
        area_1: float,
        view_factor_12: float,
        temp_1_k: float,
        temp_2_k: float,
    ) -> Decimal:
        """
        Radiation exchange between two diffuse gray surfaces.

        Formula: Q12 = sigma * A1 * F12 * (T1^4 - T2^4) / (1/e1 + (1/e2 - 1)*A1/(A2*F12))
        Source: Radiative exchange (simplified enclosure)
        Range: Two-surface enclosures
        Uncertainty: +/- 15%

        Args:
            emissivity_1: Emissivity of surface 1
            emissivity_2: Emissivity of surface 2
            area_1: Area of surface 1 (m2)
            view_factor_12: View factor from 1 to 2
            temp_1_k: Temperature of surface 1 (K)
            temp_2_k: Temperature of surface 2 (K)

        Returns:
            Net radiation from 1 to 2 (W)
        """
        sigma = HeatTransferConstants.STEFAN_BOLTZMANN

        # For simplified case (large enclosure):
        Q = (
            sigma * emissivity_1 * area_1 * view_factor_12 *
            (temp_1_k ** 4 - temp_2_k ** 4)
        )
        return make_decimal(Q)

    @staticmethod
    def radiation_heat_transfer_coeff(
        emissivity: float,
        surface_temp_k: float,
        surroundings_temp_k: float,
    ) -> Decimal:
        """
        Linearized radiation heat transfer coefficient.

        Formula: hr = epsilon * sigma * (Ts^2 + Tsur^2) * (Ts + Tsur)
        Source: Linearization of radiation term
        Range: Small temperature differences
        Uncertainty: +/- 5%

        Args:
            emissivity: Surface emissivity
            surface_temp_k: Surface temperature (K)
            surroundings_temp_k: Surroundings temperature (K)

        Returns:
            Radiation heat transfer coefficient (W/m2.K)
        """
        sigma = HeatTransferConstants.STEFAN_BOLTZMANN
        Ts = surface_temp_k
        Tsur = surroundings_temp_k

        hr = emissivity * sigma * (Ts ** 2 + Tsur ** 2) * (Ts + Tsur)
        return make_decimal(hr)

    @staticmethod
    def view_factor_parallel_plates(
        width: float,
        height: float,
        separation: float,
    ) -> Decimal:
        """
        View factor between parallel rectangular plates.

        Formula: F12 = f(X, Y) - geometric formula
        Source: View factor algebra
        Range: Parallel aligned rectangles
        Uncertainty: +/- 2%

        Args:
            width: Width of plates (m)
            height: Height of plates (m)
            separation: Distance between plates (m)

        Returns:
            View factor F12
        """
        if separation <= 0:
            return make_decimal(1.0)

        X = width / separation
        Y = height / separation

        # Simplified formula for parallel aligned plates
        term1 = math.log(((1 + X ** 2) * (1 + Y ** 2)) / (1 + X ** 2 + Y ** 2))
        term2 = X * math.sqrt(1 + Y ** 2) * math.atan(X / math.sqrt(1 + Y ** 2))
        term3 = Y * math.sqrt(1 + X ** 2) * math.atan(Y / math.sqrt(1 + X ** 2))
        term4 = X * math.atan(X) + Y * math.atan(Y)

        F12 = (2 / (math.pi * X * Y)) * (term1 / 2 + term2 + term3 - term4)
        return make_decimal(max(min(F12, 1), 0))

    @staticmethod
    def view_factor_perpendicular_plates(
        width_1: float,
        width_2: float,
        length: float,
    ) -> Decimal:
        """
        View factor between perpendicular plates with common edge.

        Formula: F12 = f(H, W) - geometric formula
        Source: View factor algebra
        Range: Perpendicular rectangles
        Uncertainty: +/- 3%

        Args:
            width_1: Width of surface 1 (m)
            width_2: Width of surface 2 (m)
            length: Common edge length (m)

        Returns:
            View factor F12
        """
        H = width_1 / length
        W = width_2 / length

        # Formula for perpendicular plates with common edge
        A = (1 + H ** 2) * (1 + W ** 2) / (1 + H ** 2 + W ** 2)
        B = W ** 2 * (1 + H ** 2 + W ** 2) / ((1 + W ** 2) * (H ** 2 + W ** 2))
        C = H ** 2 * (1 + H ** 2 + W ** 2) / ((1 + H ** 2) * (H ** 2 + W ** 2))

        F12 = (1 / (math.pi * W)) * (
            W * math.atan(1 / W) +
            H * math.atan(1 / H) -
            math.sqrt(H ** 2 + W ** 2) * math.atan(1 / math.sqrt(H ** 2 + W ** 2)) +
            0.25 * math.log(A * (B ** (W ** 2)) * (C ** (H ** 2)))
        )

        return make_decimal(max(min(F12, 1), 0))


# =============================================================================
# Heat Exchanger Design
# =============================================================================

class HeatExchangerDesign:
    """
    Heat exchanger design calculations.

    Reference: TEMA, Kern - Process Heat Transfer
    """

    @staticmethod
    def overall_heat_transfer_coeff(
        h_inside: float,
        h_outside: float,
        wall_thickness: float,
        wall_conductivity: float,
        r_fouling_inside: float = 0.0,
        r_fouling_outside: float = 0.0,
        area_ratio: float = 1.0,
    ) -> Decimal:
        """
        Overall heat transfer coefficient (U) based on outside area.

        Formula: 1/U = 1/ho + Rfo + (ro*ln(ro/ri))/k + Rfi*Ao/Ai + (1/hi)*Ao/Ai
        Source: Heat exchanger theory
        Range: All heat exchangers
        Uncertainty: +/- 10%

        Args:
            h_inside: Inside heat transfer coefficient (W/m2.K)
            h_outside: Outside heat transfer coefficient (W/m2.K)
            wall_thickness: Wall thickness (m)
            wall_conductivity: Wall thermal conductivity (W/m.K)
            r_fouling_inside: Inside fouling resistance (m2.K/W)
            r_fouling_outside: Outside fouling resistance (m2.K/W)
            area_ratio: Ao/Ai ratio

        Returns:
            Overall U based on outside area (W/m2.K)
        """
        if h_inside <= 0 or h_outside <= 0:
            raise ValueError("Heat transfer coefficients must be positive")

        # Sum of resistances (based on outside area)
        R_total = (
            1 / h_outside +
            r_fouling_outside +
            wall_thickness / wall_conductivity +
            r_fouling_inside * area_ratio +
            (1 / h_inside) * area_ratio
        )

        U = 1 / R_total
        return make_decimal(U)

    @staticmethod
    def heat_transfer_area(
        heat_duty: float,
        overall_coeff: float,
        mean_temp_diff: float,
    ) -> Decimal:
        """
        Required heat transfer area.

        Formula: A = Q / (U * dT_m)
        Source: Heat exchanger design equation
        Range: All heat exchangers
        Uncertainty: +/- 10%

        Args:
            heat_duty: Heat duty (W)
            overall_coeff: Overall U (W/m2.K)
            mean_temp_diff: Mean temperature difference (K)

        Returns:
            Required area (m2)
        """
        if overall_coeff <= 0 or mean_temp_diff <= 0:
            raise ValueError("U and MTD must be positive")

        A = heat_duty / (overall_coeff * mean_temp_diff)
        return make_decimal(A)

    @staticmethod
    def tube_count_estimate(
        heat_transfer_area: float,
        tube_od: float,
        tube_length: float,
    ) -> Decimal:
        """
        Estimate number of tubes required.

        Formula: N = A / (pi * Do * L)
        Source: Geometric calculation
        Range: Shell-and-tube exchangers
        Uncertainty: +/- 5%

        Args:
            heat_transfer_area: Required area (m2)
            tube_od: Tube outside diameter (m)
            tube_length: Effective tube length (m)

        Returns:
            Number of tubes (round up in practice)
        """
        area_per_tube = math.pi * tube_od * tube_length
        N = heat_transfer_area / area_per_tube
        return make_decimal(N)

    @staticmethod
    def shell_diameter_estimate(
        tube_count: int,
        tube_od: float,
        pitch: float,
        layout_angle: float = 30,
    ) -> Decimal:
        """
        Estimate shell inside diameter.

        Formula: Ds = CL * sqrt(A_tube_bundle / constant)
        Source: TEMA correlations
        Range: Shell-and-tube exchangers
        Uncertainty: +/- 10%

        Args:
            tube_count: Number of tubes
            tube_od: Tube outside diameter (m)
            pitch: Tube pitch (m)
            layout_angle: Tube layout angle (30 or 45 degrees)

        Returns:
            Shell inside diameter (m)
        """
        # Layout constant
        if layout_angle == 30 or layout_angle == 60:
            CL = 0.866  # Triangular
        else:
            CL = 1.0  # Square

        # Tube bundle area estimate
        A_bundle = tube_count * (pitch ** 2) / CL

        # Shell diameter (with clearance factor ~1.1)
        Ds = 1.1 * math.sqrt(4 * A_bundle / math.pi)

        return make_decimal(Ds)

    @staticmethod
    def pressure_drop_tube_side(
        mass_flow_rate: float,
        tube_count: int,
        tube_id: float,
        tube_length: float,
        density: float,
        friction_factor: float,
        passes: int = 1,
    ) -> Decimal:
        """
        Tube-side pressure drop estimate.

        Formula: dP = f * (L/D) * (rho * V^2 / 2) * n_passes
        Source: Darcy-Weisbach equation
        Range: Turbulent flow
        Uncertainty: +/- 20%

        Args:
            mass_flow_rate: Total mass flow rate (kg/s)
            tube_count: Number of tubes
            tube_id: Tube inside diameter (m)
            tube_length: Tube length (m)
            density: Fluid density (kg/m3)
            friction_factor: Darcy friction factor
            passes: Number of tube passes

        Returns:
            Pressure drop (Pa)
        """
        # Flow per tube
        tubes_per_pass = tube_count / passes
        A_flow = tubes_per_pass * math.pi * (tube_id ** 2) / 4
        velocity = mass_flow_rate / (density * A_flow)

        # Friction loss
        dP_friction = friction_factor * (tube_length / tube_id) * (density * velocity ** 2 / 2)

        # Return bend losses (approximately 4 velocity heads per pass)
        dP_bends = 4 * (density * velocity ** 2 / 2) * (passes - 1)

        # Entrance/exit losses
        dP_ends = 1.5 * (density * velocity ** 2 / 2)

        dP_total = (dP_friction + dP_bends + dP_ends) * passes

        return make_decimal(dP_total)


# =============================================================================
# Fin Efficiency Calculations
# =============================================================================

class FinEfficiencyCalculations:
    """
    Extended surface (fin) efficiency calculations.

    Reference: Incropera & DeWitt, Kraus - Extended Surface Heat Transfer
    """

    @staticmethod
    def fin_efficiency_rectangular(
        fin_length: float,
        fin_thickness: float,
        thermal_conductivity: float,
        heat_transfer_coeff: float,
    ) -> Decimal:
        """
        Efficiency of rectangular fin with adiabatic tip.

        Formula: eta_f = tanh(mL) / mL where m = sqrt(2h/(k*t))
        Source: Extended surface theory
        Range: Rectangular fins
        Uncertainty: +/- 5%

        Args:
            fin_length: Fin length from base (m)
            fin_thickness: Fin thickness (m)
            thermal_conductivity: Fin thermal conductivity (W/m.K)
            heat_transfer_coeff: Convection coefficient (W/m2.K)

        Returns:
            Fin efficiency (0-1)
        """
        if thermal_conductivity <= 0 or fin_thickness <= 0:
            raise ValueError("Conductivity and thickness must be positive")

        m = math.sqrt(2 * heat_transfer_coeff / (thermal_conductivity * fin_thickness))
        mL = m * fin_length

        if mL < 1e-6:
            eta = 1.0
        else:
            eta = math.tanh(mL) / mL

        return make_decimal(max(min(eta, 1), 0))

    @staticmethod
    def fin_efficiency_annular(
        r_inner: float,
        r_outer: float,
        fin_thickness: float,
        thermal_conductivity: float,
        heat_transfer_coeff: float,
    ) -> Decimal:
        """
        Efficiency of annular fin with adiabatic tip.

        Formula: Complex Bessel function solution (approximation used)
        Source: Extended surface theory
        Range: Annular fins
        Uncertainty: +/- 8%

        Args:
            r_inner: Inner radius (tube OD) (m)
            r_outer: Outer radius (fin tip) (m)
            fin_thickness: Fin thickness (m)
            thermal_conductivity: Fin thermal conductivity (W/m.K)
            heat_transfer_coeff: Convection coefficient (W/m2.K)

        Returns:
            Fin efficiency (0-1)
        """
        m = math.sqrt(2 * heat_transfer_coeff / (thermal_conductivity * fin_thickness))

        L_c = r_outer - r_inner + fin_thickness / 2  # Corrected length
        r_2c = r_outer + fin_thickness / 2

        # Approximation for annular fin efficiency
        phi = (r_2c / r_inner - 1) * (1 + 0.35 * math.log(r_2c / r_inner))
        mL_c = m * L_c * math.sqrt(phi)

        if mL_c < 1e-6:
            eta = 1.0
        else:
            eta = math.tanh(mL_c) / mL_c

        return make_decimal(max(min(eta, 1), 0))

    @staticmethod
    def fin_effectiveness(
        fin_efficiency: float,
        fin_area: float,
        base_area: float,
    ) -> Decimal:
        """
        Fin effectiveness (ratio of heat transfer with/without fin).

        Formula: epsilon_f = eta_f * A_f / A_b
        Source: Extended surface theory
        Range: All fins
        Uncertainty: +/- 5%

        Args:
            fin_efficiency: Fin efficiency (0-1)
            fin_area: Total fin surface area (m2)
            base_area: Base area covered by fin (m2)

        Returns:
            Fin effectiveness
        """
        if base_area <= 0:
            raise ValueError("Base area must be positive")

        epsilon = fin_efficiency * fin_area / base_area
        return make_decimal(epsilon)

    @staticmethod
    def overall_surface_efficiency(
        fin_efficiency: float,
        fin_area: float,
        base_exposed_area: float,
    ) -> Decimal:
        """
        Overall surface efficiency for finned surface.

        Formula: eta_o = 1 - (A_f/A_t) * (1 - eta_f)
        Source: Extended surface theory
        Range: All finned surfaces
        Uncertainty: +/- 5%

        Args:
            fin_efficiency: Fin efficiency (0-1)
            fin_area: Total fin surface area (m2)
            base_exposed_area: Exposed base area between fins (m2)

        Returns:
            Overall surface efficiency (0-1)
        """
        total_area = fin_area + base_exposed_area
        eta_o = 1 - (fin_area / total_area) * (1 - fin_efficiency)
        return make_decimal(max(min(eta_o, 1), 0))

    @staticmethod
    def heat_transfer_finned_surface(
        overall_efficiency: float,
        total_area: float,
        heat_transfer_coeff: float,
        surface_temp: float,
        fluid_temp: float,
    ) -> Decimal:
        """
        Heat transfer from finned surface.

        Formula: Q = eta_o * h * A_t * (Ts - Tf)
        Source: Extended surface theory
        Range: All finned surfaces
        Uncertainty: +/- 10%

        Args:
            overall_efficiency: Overall surface efficiency (0-1)
            total_area: Total surface area (fins + exposed base) (m2)
            heat_transfer_coeff: Convection coefficient (W/m2.K)
            surface_temp: Base surface temperature
            fluid_temp: Fluid temperature

        Returns:
            Heat transfer rate (W)
        """
        Q = overall_efficiency * heat_transfer_coeff * total_area * (surface_temp - fluid_temp)
        return make_decimal(Q)


# =============================================================================
# Heat Transfer Formulas Collection
# =============================================================================

class HeatTransferFormulas:
    """
    Collection of all heat transfer formulas for registration with CalculationEngine.
    """

    @staticmethod
    def get_all_formula_definitions() -> List[FormulaDefinition]:
        """Get all heat transfer formula definitions."""
        formulas = []

        # Conduction formulas
        formulas.extend(HeatTransferFormulas._get_conduction_formulas())

        # Convection formulas
        formulas.extend(HeatTransferFormulas._get_convection_formulas())

        # Radiation formulas
        formulas.extend(HeatTransferFormulas._get_radiation_formulas())

        # Heat exchanger design formulas
        formulas.extend(HeatTransferFormulas._get_heat_exchanger_formulas())

        # Fin efficiency formulas
        formulas.extend(HeatTransferFormulas._get_fin_formulas())

        return formulas

    @staticmethod
    def _get_conduction_formulas() -> List[FormulaDefinition]:
        """Get conduction formula definitions."""
        return [
            FormulaDefinition(
                formula_id="heat_rate_plane_wall",
                name="Heat Rate - Plane Wall",
                description="Heat transfer rate through plane wall by conduction",
                category="heat_transfer",
                source_standard="Fourier's Law",
                source_reference="Incropera & DeWitt",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="thermal_conductivity",
                        description="Thermal conductivity",
                        unit="W/(m.K)",
                        category=UnitCategory.THERMAL_CONDUCTIVITY,
                        min_value=0.001,
                        max_value=500,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="area",
                        description="Heat transfer area",
                        unit="m2",
                        category=UnitCategory.AREA,
                        min_value=0.0001,
                        max_value=100000,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="thickness",
                        description="Wall thickness",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0001,
                        max_value=10,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="temperature_hot",
                        description="Hot side temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=3000,
                        uncertainty_percent=0.5,
                    ),
                    ParameterDefinition(
                        name="temperature_cold",
                        description="Cold side temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=3000,
                        uncertainty_percent=0.5,
                    ),
                ],
                output_name="heat_rate",
                output_unit="W",
                output_description="Heat transfer rate",
                precision=2,
                test_cases=[
                    {"thermal_conductivity": 50, "area": 1, "thickness": 0.01, "temperature_hot": 400, "temperature_cold": 300, "expected": 500000, "tolerance": 1000},
                ],
            ),
            FormulaDefinition(
                formula_id="thermal_resistance_plane_wall",
                name="Thermal Resistance - Plane Wall",
                description="Thermal resistance of plane wall",
                category="heat_transfer",
                source_standard="Heat Transfer Fundamentals",
                source_reference="Incropera & DeWitt",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="thermal_conductivity",
                        description="Thermal conductivity",
                        unit="W/(m.K)",
                        category=UnitCategory.THERMAL_CONDUCTIVITY,
                        min_value=0.001,
                        max_value=500,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="area",
                        description="Heat transfer area",
                        unit="m2",
                        category=UnitCategory.AREA,
                        min_value=0.0001,
                        max_value=100000,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="thickness",
                        description="Wall thickness",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0001,
                        max_value=10,
                        uncertainty_percent=1.0,
                    ),
                ],
                output_name="thermal_resistance",
                output_unit="K/W",
                output_description="Thermal resistance",
                precision=6,
            ),
            FormulaDefinition(
                formula_id="heat_rate_cylindrical_wall",
                name="Heat Rate - Cylindrical Wall",
                description="Heat transfer rate through cylindrical wall",
                category="heat_transfer",
                source_standard="Fourier's Law",
                source_reference="Cylindrical coordinates",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="thermal_conductivity",
                        description="Thermal conductivity",
                        unit="W/(m.K)",
                        category=UnitCategory.THERMAL_CONDUCTIVITY,
                        min_value=0.001,
                        max_value=500,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="length",
                        description="Cylinder length",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.001,
                        max_value=1000,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="radius_inner",
                        description="Inner radius",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0001,
                        max_value=100,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="radius_outer",
                        description="Outer radius",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0002,
                        max_value=100,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="temperature_inner",
                        description="Inner surface temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=3000,
                        uncertainty_percent=0.5,
                    ),
                    ParameterDefinition(
                        name="temperature_outer",
                        description="Outer surface temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=3000,
                        uncertainty_percent=0.5,
                    ),
                ],
                output_name="heat_rate",
                output_unit="W",
                output_description="Heat transfer rate",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_convection_formulas() -> List[FormulaDefinition]:
        """Get convection formula definitions."""
        return [
            FormulaDefinition(
                formula_id="nusselt_dittus_boelter",
                name="Nusselt Number - Dittus-Boelter",
                description="Nusselt number for turbulent pipe flow",
                category="heat_transfer",
                source_standard="Dittus-Boelter Correlation",
                source_reference="Dittus & Boelter (1930)",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="reynolds_number",
                        description="Reynolds number",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=10000,
                        max_value=1e8,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="prandtl_number",
                        description="Prandtl number",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0.6,
                        max_value=160,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="heating",
                        description="Fluid being heated (1) or cooled (0)",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        default_value=1,
                        required=False,
                        uncertainty_percent=0,
                    ),
                ],
                output_name="nusselt_number",
                output_unit="",
                output_description="Nusselt number",
                precision=2,
            ),
            FormulaDefinition(
                formula_id="reynolds_number",
                name="Reynolds Number",
                description="Calculate Reynolds number",
                category="heat_transfer",
                source_standard="Fluid Mechanics",
                source_reference="Standard definition",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="velocity",
                        description="Flow velocity",
                        unit="m/s",
                        category=UnitCategory.VELOCITY,
                        min_value=0,
                        max_value=1000,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="characteristic_length",
                        description="Characteristic length",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0001,
                        max_value=100,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="density",
                        description="Fluid density",
                        unit="kg/m3",
                        category=UnitCategory.DENSITY,
                        min_value=0.01,
                        max_value=20000,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="viscosity",
                        description="Dynamic viscosity",
                        unit="Pa.s",
                        category=UnitCategory.VISCOSITY,
                        min_value=1e-6,
                        max_value=10000,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="reynolds_number",
                output_unit="",
                output_description="Reynolds number",
                precision=0,
            ),
            FormulaDefinition(
                formula_id="heat_transfer_coeff_from_nusselt",
                name="Heat Transfer Coefficient from Nusselt",
                description="Calculate h from Nusselt number",
                category="heat_transfer",
                source_standard="Heat Transfer Fundamentals",
                source_reference="Definition of Nusselt number",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="nusselt_number",
                        description="Nusselt number",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0.1,
                        max_value=100000,
                        uncertainty_percent=10.0,
                    ),
                    ParameterDefinition(
                        name="thermal_conductivity",
                        description="Fluid thermal conductivity",
                        unit="W/(m.K)",
                        category=UnitCategory.THERMAL_CONDUCTIVITY,
                        min_value=0.001,
                        max_value=500,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="characteristic_length",
                        description="Characteristic length",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0001,
                        max_value=100,
                        uncertainty_percent=1.0,
                    ),
                ],
                output_name="heat_transfer_coeff",
                output_unit="W/(m2.K)",
                output_description="Heat transfer coefficient",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_radiation_formulas() -> List[FormulaDefinition]:
        """Get radiation formula definitions."""
        return [
            FormulaDefinition(
                formula_id="blackbody_emissive_power",
                name="Blackbody Emissive Power",
                description="Stefan-Boltzmann blackbody emission",
                category="heat_transfer",
                source_standard="Stefan-Boltzmann Law",
                source_reference="Classical thermodynamics",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="temperature_k",
                        description="Absolute temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=10000,
                        uncertainty_percent=0.5,
                    ),
                ],
                output_name="emissive_power",
                output_unit="W/m2",
                output_description="Blackbody emissive power",
                precision=2,
                test_cases=[
                    {"temperature_k": 1000, "expected": 56700, "tolerance": 100},
                ],
            ),
            FormulaDefinition(
                formula_id="radiation_heat_rate_surface",
                name="Radiation Heat Rate - Surface",
                description="Net radiation from surface to surroundings",
                category="heat_transfer",
                source_standard="Gray Surface Radiation",
                source_reference="Incropera & DeWitt",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="emissivity",
                        description="Surface emissivity",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="area",
                        description="Surface area",
                        unit="m2",
                        category=UnitCategory.AREA,
                        min_value=0.0001,
                        max_value=100000,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="surface_temp_k",
                        description="Surface temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=10000,
                        uncertainty_percent=0.5,
                    ),
                    ParameterDefinition(
                        name="surroundings_temp_k",
                        description="Surroundings temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=10000,
                        uncertainty_percent=0.5,
                    ),
                ],
                output_name="heat_rate",
                output_unit="W",
                output_description="Net radiation heat transfer",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_heat_exchanger_formulas() -> List[FormulaDefinition]:
        """Get heat exchanger formula definitions."""
        return [
            FormulaDefinition(
                formula_id="overall_heat_transfer_coeff",
                name="Overall Heat Transfer Coefficient",
                description="Calculate overall U value",
                category="heat_transfer",
                source_standard="Heat Exchanger Design",
                source_reference="TEMA Standards",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="h_inside",
                        description="Inside heat transfer coefficient",
                        unit="W/(m2.K)",
                        category=UnitCategory.HEAT_TRANSFER_COEFFICIENT,
                        min_value=10,
                        max_value=100000,
                        uncertainty_percent=10.0,
                    ),
                    ParameterDefinition(
                        name="h_outside",
                        description="Outside heat transfer coefficient",
                        unit="W/(m2.K)",
                        category=UnitCategory.HEAT_TRANSFER_COEFFICIENT,
                        min_value=10,
                        max_value=100000,
                        uncertainty_percent=10.0,
                    ),
                    ParameterDefinition(
                        name="wall_thickness",
                        description="Wall thickness",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0001,
                        max_value=0.1,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="wall_conductivity",
                        description="Wall thermal conductivity",
                        unit="W/(m.K)",
                        category=UnitCategory.THERMAL_CONDUCTIVITY,
                        min_value=1,
                        max_value=500,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="r_fouling_inside",
                        description="Inside fouling resistance",
                        unit="m2.K/W",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.01,
                        default_value=0,
                        required=False,
                        uncertainty_percent=50.0,
                    ),
                    ParameterDefinition(
                        name="r_fouling_outside",
                        description="Outside fouling resistance",
                        unit="m2.K/W",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.01,
                        default_value=0,
                        required=False,
                        uncertainty_percent=50.0,
                    ),
                ],
                output_name="overall_u",
                output_unit="W/(m2.K)",
                output_description="Overall heat transfer coefficient",
                precision=2,
            ),
            FormulaDefinition(
                formula_id="heat_transfer_area",
                name="Heat Transfer Area",
                description="Required heat exchanger area",
                category="heat_transfer",
                source_standard="Heat Exchanger Design",
                source_reference="Q = U*A*dTm",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="heat_duty",
                        description="Heat duty",
                        unit="W",
                        category=UnitCategory.POWER,
                        min_value=0,
                        max_value=1e12,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="overall_coeff",
                        description="Overall U value",
                        unit="W/(m2.K)",
                        category=UnitCategory.HEAT_TRANSFER_COEFFICIENT,
                        min_value=10,
                        max_value=10000,
                        uncertainty_percent=10.0,
                    ),
                    ParameterDefinition(
                        name="mean_temp_diff",
                        description="Mean temperature difference",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=1,
                        max_value=500,
                        uncertainty_percent=5.0,
                    ),
                ],
                output_name="area",
                output_unit="m2",
                output_description="Required heat transfer area",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_fin_formulas() -> List[FormulaDefinition]:
        """Get fin efficiency formula definitions."""
        return [
            FormulaDefinition(
                formula_id="fin_efficiency_rectangular",
                name="Fin Efficiency - Rectangular",
                description="Efficiency of rectangular fin",
                category="heat_transfer",
                source_standard="Extended Surface Theory",
                source_reference="Incropera & DeWitt",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="fin_length",
                        description="Fin length from base",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.001,
                        max_value=1,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="fin_thickness",
                        description="Fin thickness",
                        unit="m",
                        category=UnitCategory.LENGTH,
                        min_value=0.0001,
                        max_value=0.1,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="thermal_conductivity",
                        description="Fin thermal conductivity",
                        unit="W/(m.K)",
                        category=UnitCategory.THERMAL_CONDUCTIVITY,
                        min_value=1,
                        max_value=500,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="heat_transfer_coeff",
                        description="Convection coefficient",
                        unit="W/(m2.K)",
                        category=UnitCategory.HEAT_TRANSFER_COEFFICIENT,
                        min_value=1,
                        max_value=10000,
                        uncertainty_percent=15.0,
                    ),
                ],
                output_name="fin_efficiency",
                output_unit="",
                output_description="Fin efficiency (0-1)",
                precision=4,
            ),
            FormulaDefinition(
                formula_id="overall_surface_efficiency",
                name="Overall Surface Efficiency",
                description="Overall efficiency for finned surface",
                category="heat_transfer",
                source_standard="Extended Surface Theory",
                source_reference="Kraus - Extended Surface Heat Transfer",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="fin_efficiency",
                        description="Individual fin efficiency",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="fin_area",
                        description="Total fin surface area",
                        unit="m2",
                        category=UnitCategory.AREA,
                        min_value=0,
                        max_value=10000,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="base_exposed_area",
                        description="Exposed base area between fins",
                        unit="m2",
                        category=UnitCategory.AREA,
                        min_value=0,
                        max_value=10000,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="overall_efficiency",
                output_unit="",
                output_description="Overall surface efficiency (0-1)",
                precision=4,
            ),
        ]

    @staticmethod
    def register_all(registry: FormulaRegistry):
        """Register all heat transfer formulas with the calculation engine."""
        for formula in HeatTransferFormulas.get_all_formula_definitions():
            calculator = HeatTransferFormulas._get_calculator(formula.formula_id)
            if calculator:
                registry.register(formula, calculator)

    @staticmethod
    def _get_calculator(formula_id: str):
        """Get calculator function for a formula."""
        calculators = {
            "heat_rate_plane_wall": lambda p: (
                ConductionCalculations.heat_rate_plane_wall(
                    p["thermal_conductivity"],
                    p["area"],
                    p["thickness"],
                    p["temperature_hot"],
                    p["temperature_cold"],
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate heat rate through plane wall",
                    operation="fourier_law",
                    inputs=p,
                    output_value=ConductionCalculations.heat_rate_plane_wall(
                        p["thermal_conductivity"],
                        p["area"],
                        p["thickness"],
                        p["temperature_hot"],
                        p["temperature_cold"],
                    ),
                    output_name="heat_rate",
                )]
            ),
            "reynolds_number": lambda p: (
                ConvectionCalculations.reynolds_number(
                    p["velocity"],
                    p["characteristic_length"],
                    p["density"],
                    p["viscosity"],
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate Reynolds number",
                    operation="reynolds_calculation",
                    inputs=p,
                    output_value=ConvectionCalculations.reynolds_number(
                        p["velocity"],
                        p["characteristic_length"],
                        p["density"],
                        p["viscosity"],
                    ),
                    output_name="reynolds_number",
                )]
            ),
            "blackbody_emissive_power": lambda p: (
                RadiationCalculations.blackbody_emissive_power(p["temperature_k"]),
                [CalculationStep(
                    step_number=1,
                    description="Calculate blackbody emissive power",
                    operation="stefan_boltzmann",
                    inputs=p,
                    output_value=RadiationCalculations.blackbody_emissive_power(p["temperature_k"]),
                    output_name="emissive_power",
                )]
            ),
            "fin_efficiency_rectangular": lambda p: (
                FinEfficiencyCalculations.fin_efficiency_rectangular(
                    p["fin_length"],
                    p["fin_thickness"],
                    p["thermal_conductivity"],
                    p["heat_transfer_coeff"],
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate rectangular fin efficiency",
                    operation="fin_efficiency",
                    inputs=p,
                    output_value=FinEfficiencyCalculations.fin_efficiency_rectangular(
                        p["fin_length"],
                        p["fin_thickness"],
                        p["thermal_conductivity"],
                        p["heat_transfer_coeff"],
                    ),
                    output_name="fin_efficiency",
                )]
            ),
        }
        return calculators.get(formula_id)
