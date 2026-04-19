"""
GL-009 THERMALIQ Agent - Heat Transfer Coefficient Calculations

This module provides heat transfer coefficient calculations for thermal
fluid systems, including film coefficients, overall heat transfer
coefficients, and fouling analysis.

Correlations implemented:
    - Dittus-Boelter (turbulent internal flow)
    - Sieder-Tate (turbulent with viscosity correction)
    - Gnielinski (transitional and turbulent)
    - Petukhov (high accuracy turbulent)
    - Laminar flow correlations

All calculations are deterministic - ZERO HALLUCINATION guaranteed.

Reference:
    - Incropera & DeWitt, "Heat and Mass Transfer"
    - TEMA Standards
    - VDI Heat Atlas

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.heat_transfer import (
    ...     HeatTransferCalculator,
    ... )
    >>> calc = HeatTransferCalculator(fluid_type=ThermalFluidType.THERMINOL_66)
    >>> result = calc.calculate_film_coefficient(
    ...     temperature_f=550.0,
    ...     velocity_ft_s=8.0,
    ...     tube_id_in=2.0,
    ... )
    >>> print(f"h = {result.film_coefficient_btu_hr_ft2_f:.1f} BTU/hr-ft2-F")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from pydantic import BaseModel, Field

from .schemas import (
    ThermalFluidType,
    FlowRegime,
    HeatTransferAnalysis,
)
from .fluid_properties import ThermalFluidPropertyDatabase

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Reynolds number thresholds
RE_LAMINAR_LIMIT = 2300
RE_TURBULENT_LIMIT = 10000

# Typical fouling factors (hr-ft2-F/BTU)
FOULING_FACTORS = {
    "clean": 0.0001,
    "light": 0.001,
    "moderate": 0.002,
    "heavy": 0.005,
    "severe": 0.010,
}


# =============================================================================
# CORRELATION ENUM
# =============================================================================

class HeatTransferCorrelation(str, Enum):
    """Heat transfer correlation types."""
    DITTUS_BOELTER = "dittus_boelter"
    SIEDER_TATE = "sieder_tate"
    GNIELINSKI = "gnielinski"
    PETUKHOV = "petukhov"
    LAMINAR_CONSTANT_WALL = "laminar_constant_wall"
    LAMINAR_CONSTANT_FLUX = "laminar_constant_flux"


# =============================================================================
# HEAT TRANSFER CALCULATOR
# =============================================================================

class HeatTransferCalculator:
    """
    Heat transfer coefficient calculator for thermal fluid systems.

    This class calculates film heat transfer coefficients using established
    correlations for different flow regimes. All calculations are
    deterministic with no ML/LLM in the calculation path.

    Supported correlations:
        - Dittus-Boelter: Nu = 0.023 Re^0.8 Pr^0.4 (heating)
        - Sieder-Tate: With viscosity ratio correction
        - Gnielinski: Valid for transitional and turbulent
        - Petukhov: High accuracy for turbulent

    Example:
        >>> calc = HeatTransferCalculator(
        ...     fluid_type=ThermalFluidType.THERMINOL_66
        ... )
        >>> h = calc.calculate_film_coefficient(
        ...     temperature_f=550.0,
        ...     velocity_ft_s=8.0,
        ...     tube_id_in=2.0,
        ... )
    """

    def __init__(
        self,
        fluid_type: ThermalFluidType,
        wall_temperature_f: Optional[float] = None,
    ) -> None:
        """
        Initialize the heat transfer calculator.

        Args:
            fluid_type: Type of thermal fluid
            wall_temperature_f: Wall temperature for viscosity correction
        """
        self.fluid_type = fluid_type
        self.wall_temperature_f = wall_temperature_f

        self._property_db = ThermalFluidPropertyDatabase()
        self._calculation_count = 0

        logger.info(f"HeatTransferCalculator initialized for {fluid_type}")

    def calculate_film_coefficient(
        self,
        temperature_f: float,
        velocity_ft_s: float,
        tube_id_in: float,
        correlation: HeatTransferCorrelation = HeatTransferCorrelation.GNIELINSKI,
        wall_temperature_f: Optional[float] = None,
    ) -> HeatTransferAnalysis:
        """
        Calculate film heat transfer coefficient.

        Args:
            temperature_f: Bulk fluid temperature (F)
            velocity_ft_s: Fluid velocity (ft/s)
            tube_id_in: Tube inner diameter (inches)
            correlation: Heat transfer correlation to use
            wall_temperature_f: Wall temperature for Sieder-Tate (F)

        Returns:
            HeatTransferAnalysis with results
        """
        self._calculation_count += 1
        warnings = []

        # Get fluid properties at bulk temperature
        props = self._property_db.get_properties(self.fluid_type, temperature_f)

        # Convert units
        tube_id_ft = tube_id_in / 12.0
        kinematic_viscosity_ft2_s = props.kinematic_viscosity_cst * 1.076e-5

        # Calculate Reynolds number
        reynolds = velocity_ft_s * tube_id_ft / kinematic_viscosity_ft2_s

        # Determine flow regime
        if reynolds < RE_LAMINAR_LIMIT:
            flow_regime = FlowRegime.LAMINAR
        elif reynolds < RE_TURBULENT_LIMIT:
            flow_regime = FlowRegime.TRANSITIONAL
            warnings.append(
                f"Transitional flow (Re={reynolds:.0f}) - consider increasing velocity"
            )
        else:
            flow_regime = FlowRegime.TURBULENT

        # Get Prandtl number
        prandtl = props.prandtl_number

        # Calculate Nusselt number based on correlation
        if flow_regime == FlowRegime.LAMINAR:
            nusselt, correlation_used = self._laminar_nusselt(
                reynolds, prandtl, tube_id_ft, correlation
            )
        else:
            nusselt, correlation_used = self._turbulent_nusselt(
                reynolds, prandtl, temperature_f, wall_temperature_f, correlation
            )

        # Calculate film coefficient
        # h = Nu * k / D
        h = nusselt * props.thermal_conductivity_btu_hr_ft_f / tube_id_ft

        # Validate result
        if h < 10:
            warnings.append(f"Low heat transfer coefficient ({h:.1f}), verify flow conditions")
        elif h > 1000:
            warnings.append(f"High heat transfer coefficient ({h:.1f}), verify inputs")

        return HeatTransferAnalysis(
            reynolds_number=round(reynolds, 0),
            flow_regime=flow_regime,
            film_coefficient_btu_hr_ft2_f=round(h, 1),
            nusselt_number=round(nusselt, 2),
            correlation_used=correlation_used,
            warnings=warnings,
        )

    def _laminar_nusselt(
        self,
        reynolds: float,
        prandtl: float,
        diameter_ft: float,
        correlation: HeatTransferCorrelation,
    ) -> Tuple[float, str]:
        """
        Calculate Nusselt number for laminar flow.

        For fully developed laminar flow:
        - Constant wall temperature: Nu = 3.66
        - Constant heat flux: Nu = 4.36
        """
        if correlation == HeatTransferCorrelation.LAMINAR_CONSTANT_FLUX:
            nusselt = 4.36
            corr_name = "Laminar constant heat flux (Nu=4.36)"
        else:
            nusselt = 3.66
            corr_name = "Laminar constant wall temp (Nu=3.66)"

        return nusselt, corr_name

    def _turbulent_nusselt(
        self,
        reynolds: float,
        prandtl: float,
        bulk_temp_f: float,
        wall_temp_f: Optional[float],
        correlation: HeatTransferCorrelation,
    ) -> Tuple[float, str]:
        """Calculate Nusselt number for turbulent/transitional flow."""
        if correlation == HeatTransferCorrelation.DITTUS_BOELTER:
            return self._dittus_boelter(reynolds, prandtl)

        elif correlation == HeatTransferCorrelation.SIEDER_TATE:
            return self._sieder_tate(reynolds, prandtl, bulk_temp_f, wall_temp_f)

        elif correlation == HeatTransferCorrelation.GNIELINSKI:
            return self._gnielinski(reynolds, prandtl)

        elif correlation == HeatTransferCorrelation.PETUKHOV:
            return self._petukhov(reynolds, prandtl)

        else:
            # Default to Gnielinski
            return self._gnielinski(reynolds, prandtl)

    def _dittus_boelter(
        self,
        reynolds: float,
        prandtl: float,
    ) -> Tuple[float, str]:
        """
        Dittus-Boelter correlation for turbulent flow.

        Nu = 0.023 * Re^0.8 * Pr^0.4 (heating)
        Nu = 0.023 * Re^0.8 * Pr^0.3 (cooling)

        Valid for:
            - Re > 10,000
            - 0.6 < Pr < 160
            - L/D > 10
        """
        # Assume heating
        nusselt = 0.023 * (reynolds ** 0.8) * (prandtl ** 0.4)

        return nusselt, "Dittus-Boelter (Nu=0.023*Re^0.8*Pr^0.4)"

    def _sieder_tate(
        self,
        reynolds: float,
        prandtl: float,
        bulk_temp_f: float,
        wall_temp_f: Optional[float],
    ) -> Tuple[float, str]:
        """
        Sieder-Tate correlation with viscosity correction.

        Nu = 0.027 * Re^0.8 * Pr^(1/3) * (mu_bulk/mu_wall)^0.14

        Accounts for viscosity variation near wall.
        """
        # Start with base correlation
        nusselt = 0.027 * (reynolds ** 0.8) * (prandtl ** (1/3))

        # Apply viscosity correction if wall temp available
        if wall_temp_f is not None:
            props_bulk = self._property_db.get_properties(self.fluid_type, bulk_temp_f)
            props_wall = self._property_db.get_properties(self.fluid_type, wall_temp_f)

            viscosity_ratio = (
                props_bulk.kinematic_viscosity_cst /
                props_wall.kinematic_viscosity_cst
            )
            nusselt *= viscosity_ratio ** 0.14

        return nusselt, "Sieder-Tate (with viscosity correction)"

    def _gnielinski(
        self,
        reynolds: float,
        prandtl: float,
    ) -> Tuple[float, str]:
        """
        Gnielinski correlation for transitional and turbulent flow.

        Nu = (f/8)(Re-1000)Pr / [1 + 12.7(f/8)^0.5 * (Pr^(2/3) - 1)]

        Where f is Darcy friction factor from Petukhov:
        f = (0.79*ln(Re) - 1.64)^(-2)

        Valid for:
            - 3000 < Re < 5e6
            - 0.5 < Pr < 2000
        """
        # Calculate friction factor
        f = (0.79 * math.log(reynolds) - 1.64) ** (-2)

        # Gnielinski correlation
        numerator = (f / 8) * (reynolds - 1000) * prandtl
        denominator = 1 + 12.7 * math.sqrt(f / 8) * (prandtl ** (2/3) - 1)

        nusselt = numerator / denominator

        return nusselt, "Gnielinski"

    def _petukhov(
        self,
        reynolds: float,
        prandtl: float,
    ) -> Tuple[float, str]:
        """
        Petukhov correlation for fully turbulent flow.

        Nu = (f/8)*Re*Pr / [1.07 + 12.7(f/8)^0.5 * (Pr^(2/3) - 1)]

        Valid for:
            - 10^4 < Re < 5e6
            - 0.5 < Pr < 2000
        """
        # Calculate friction factor
        f = (0.79 * math.log(reynolds) - 1.64) ** (-2)

        # Petukhov correlation
        numerator = (f / 8) * reynolds * prandtl
        denominator = 1.07 + 12.7 * math.sqrt(f / 8) * (prandtl ** (2/3) - 1)

        nusselt = numerator / denominator

        return nusselt, "Petukhov"

    def calculate_overall_coefficient(
        self,
        h_inside: float,
        h_outside: float,
        tube_od_in: float,
        tube_id_in: float,
        tube_conductivity_btu_hr_ft_f: float = 26.0,
        fouling_inside: float = 0.001,
        fouling_outside: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Calculate overall heat transfer coefficient.

        1/U = 1/h_o + R_fo + (r_o*ln(r_o/r_i))/k + (r_o/r_i)*R_fi + (r_o/r_i)/h_i

        Args:
            h_inside: Inside film coefficient (BTU/hr-ft2-F)
            h_outside: Outside film coefficient (BTU/hr-ft2-F)
            tube_od_in: Tube outer diameter (inches)
            tube_id_in: Tube inner diameter (inches)
            tube_conductivity_btu_hr_ft_f: Tube wall thermal conductivity
            fouling_inside: Inside fouling factor (hr-ft2-F/BTU)
            fouling_outside: Outside fouling factor (hr-ft2-F/BTU)

        Returns:
            Dictionary with overall coefficient and resistance breakdown
        """
        self._calculation_count += 1

        # Convert to feet
        r_o = tube_od_in / 24.0  # radius in ft
        r_i = tube_id_in / 24.0

        # Individual resistances (referenced to outside area)
        r_inside = (r_o / r_i) / h_inside
        r_fouling_inside = (r_o / r_i) * fouling_inside
        r_wall = r_o * math.log(r_o / r_i) / tube_conductivity_btu_hr_ft_f
        r_fouling_outside = fouling_outside
        r_outside = 1 / h_outside

        # Total resistance
        r_total = r_inside + r_fouling_inside + r_wall + r_fouling_outside + r_outside

        # Overall coefficient
        u_overall = 1 / r_total

        # Clean coefficient (no fouling)
        r_clean = r_inside + r_wall + r_outside
        u_clean = 1 / r_clean

        # Fouling effect
        fouling_effect_pct = (1 - u_overall / u_clean) * 100

        return {
            "overall_coefficient_btu_hr_ft2_f": round(u_overall, 1),
            "clean_coefficient_btu_hr_ft2_f": round(u_clean, 1),
            "fouling_effect_pct": round(fouling_effect_pct, 1),
            "resistance_breakdown": {
                "inside_film_pct": round(r_inside / r_total * 100, 1),
                "inside_fouling_pct": round(r_fouling_inside / r_total * 100, 1),
                "tube_wall_pct": round(r_wall / r_total * 100, 1),
                "outside_fouling_pct": round(r_fouling_outside / r_total * 100, 1),
                "outside_film_pct": round(r_outside / r_total * 100, 1),
            },
            "individual_resistances_hr_ft2_f_btu": {
                "inside_film": round(r_inside, 6),
                "inside_fouling": round(r_fouling_inside, 6),
                "tube_wall": round(r_wall, 6),
                "outside_fouling": round(r_fouling_outside, 6),
                "outside_film": round(r_outside, 6),
                "total": round(r_total, 6),
            },
        }

    def estimate_fouling(
        self,
        design_ua: float,
        actual_ua: float,
        tube_od_in: float,
        tube_id_in: float,
    ) -> Dict[str, Any]:
        """
        Estimate fouling factor from performance degradation.

        Args:
            design_ua: Design UA value (BTU/hr-F)
            actual_ua: Actual measured UA value (BTU/hr-F)
            tube_od_in: Tube OD (inches)
            tube_id_in: Tube ID (inches)

        Returns:
            Dictionary with fouling estimate
        """
        self._calculation_count += 1

        if actual_ua >= design_ua:
            return {
                "fouling_factor_hr_ft2_f_btu": 0.0,
                "fouling_level": "clean",
                "message": "No fouling detected - actual UA meets or exceeds design",
            }

        # Estimate fouling from UA degradation
        # 1/U_actual - 1/U_design = R_fouling
        # Assume same area, so (1/UA_actual - 1/UA_design) * A = R_fouling * A
        # Can estimate total fouling resistance

        # This is simplified - actual calculation requires area
        # Assuming area cancels if we're looking at ratio
        performance_ratio = actual_ua / design_ua
        effectiveness_loss_pct = (1 - performance_ratio) * 100

        # Estimate fouling factor (simplified approximation)
        # Assuming typical U ~ 50-100 BTU/hr-ft2-F
        typical_u = 75.0
        delta_r = (1 / (typical_u * performance_ratio)) - (1 / typical_u)
        estimated_fouling = max(0, delta_r)

        # Determine fouling level
        if estimated_fouling < 0.0005:
            level = "clean"
        elif estimated_fouling < 0.001:
            level = "light"
        elif estimated_fouling < 0.002:
            level = "moderate"
        elif estimated_fouling < 0.005:
            level = "heavy"
        else:
            level = "severe"

        recommendations = []
        if level in ["moderate", "heavy", "severe"]:
            recommendations.append("Consider chemical cleaning")
            recommendations.append("Evaluate fluid degradation")
        if level in ["heavy", "severe"]:
            recommendations.append("Schedule mechanical cleaning")
            recommendations.append("Check for coking on heater tubes")

        return {
            "fouling_factor_hr_ft2_f_btu": round(estimated_fouling, 6),
            "fouling_level": level,
            "performance_ratio": round(performance_ratio, 3),
            "effectiveness_loss_pct": round(effectiveness_loss_pct, 1),
            "recommendations": recommendations,
        }

    def calculate_minimum_velocity(
        self,
        temperature_f: float,
        tube_id_in: float,
        target_reynolds: float = 10000,
    ) -> float:
        """
        Calculate minimum velocity for turbulent flow.

        Args:
            temperature_f: Fluid temperature (F)
            tube_id_in: Tube ID (inches)
            target_reynolds: Target Reynolds number

        Returns:
            Minimum velocity (ft/s)
        """
        props = self._property_db.get_properties(self.fluid_type, temperature_f)

        tube_id_ft = tube_id_in / 12.0
        kinematic_viscosity_ft2_s = props.kinematic_viscosity_cst * 1.076e-5

        # Re = V * D / nu
        # V = Re * nu / D
        min_velocity = target_reynolds * kinematic_viscosity_ft2_s / tube_id_ft

        return round(min_velocity, 2)

    def calculate_heater_tube_analysis(
        self,
        bulk_temp_f: float,
        outlet_temp_f: float,
        flow_rate_gpm: float,
        tube_id_in: float,
        tube_count: int,
        heat_flux_btu_hr_ft2: float,
    ) -> Dict[str, Any]:
        """
        Analyze heater tube heat transfer and film temperature.

        Args:
            bulk_temp_f: Average bulk fluid temperature (F)
            outlet_temp_f: Outlet temperature (F)
            flow_rate_gpm: Total flow rate (GPM)
            tube_id_in: Tube inner diameter (inches)
            tube_count: Number of tubes
            heat_flux_btu_hr_ft2: Surface heat flux (BTU/hr-ft2)

        Returns:
            Dictionary with tube analysis including film temperature
        """
        self._calculation_count += 1

        # Calculate velocity per tube
        tube_area_ft2 = math.pi * (tube_id_in / 24) ** 2
        total_area_ft2 = tube_area_ft2 * tube_count

        flow_rate_ft3_s = flow_rate_gpm / 7.48 / 60
        velocity_ft_s = flow_rate_ft3_s / total_area_ft2

        # Calculate film coefficient
        ht_result = self.calculate_film_coefficient(
            temperature_f=bulk_temp_f,
            velocity_ft_s=velocity_ft_s,
            tube_id_in=tube_id_in,
        )

        h = ht_result.film_coefficient_btu_hr_ft2_f

        # Calculate film temperature rise
        # q = h * (T_wall - T_bulk)
        # T_wall = T_bulk + q/h
        film_temp_rise = heat_flux_btu_hr_ft2 / h
        film_temp_f = outlet_temp_f + film_temp_rise  # Use outlet for max film temp

        # Get max allowable film temperature
        max_film_temp = self._property_db.get_max_film_temp(self.fluid_type)

        # Safety margin
        film_temp_margin = max_film_temp - film_temp_f

        warnings = []
        if film_temp_margin < 50:
            warnings.append(
                f"Low film temperature margin ({film_temp_margin:.0f}F)"
            )
        if film_temp_margin < 25:
            warnings.append("CRITICAL: Film temperature approaching limit")

        if velocity_ft_s < 3.0:
            warnings.append(f"Low velocity ({velocity_ft_s:.1f} ft/s) - increase flow")

        return {
            "velocity_ft_s": round(velocity_ft_s, 2),
            "reynolds_number": ht_result.reynolds_number,
            "flow_regime": ht_result.flow_regime.value,
            "film_coefficient_btu_hr_ft2_f": h,
            "film_temp_rise_f": round(film_temp_rise, 1),
            "estimated_film_temp_f": round(film_temp_f, 1),
            "max_film_temp_f": max_film_temp,
            "film_temp_margin_f": round(film_temp_margin, 1),
            "warnings": warnings,
        }

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_reynolds(
    fluid_type: ThermalFluidType,
    temperature_f: float,
    velocity_ft_s: float,
    diameter_in: float,
) -> float:
    """
    Quick Reynolds number calculation.

    Args:
        fluid_type: Thermal fluid type
        temperature_f: Temperature (F)
        velocity_ft_s: Velocity (ft/s)
        diameter_in: Diameter (inches)

    Returns:
        Reynolds number
    """
    db = ThermalFluidPropertyDatabase()
    props = db.get_properties(fluid_type, temperature_f)

    diameter_ft = diameter_in / 12.0
    kinematic_viscosity_ft2_s = props.kinematic_viscosity_cst * 1.076e-5

    return velocity_ft_s * diameter_ft / kinematic_viscosity_ft2_s


def get_minimum_velocity(
    fluid_type: ThermalFluidType,
    temperature_f: float,
    tube_id_in: float,
) -> float:
    """
    Get minimum velocity for turbulent flow.

    Args:
        fluid_type: Thermal fluid type
        temperature_f: Temperature (F)
        tube_id_in: Tube ID (inches)

    Returns:
        Minimum velocity (ft/s) for Re=10000
    """
    calc = HeatTransferCalculator(fluid_type=fluid_type)
    return calc.calculate_minimum_velocity(temperature_f, tube_id_in)
