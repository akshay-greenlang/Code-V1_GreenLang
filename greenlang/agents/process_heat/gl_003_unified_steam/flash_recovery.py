"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - Flash Steam Recovery Module

This module provides thermodynamic flash steam recovery calculations.
When high-pressure condensate is released to a lower pressure vessel,
a portion "flashes" to steam that can be recovered.

Features:
    - Thermodynamic flash fraction calculation
    - Flash tank sizing
    - Energy recovery analysis
    - Economic evaluation
    - Multi-stage flash optimization

Thermodynamic Basis:
    Flash fraction = (h_condensate - h_f_flash) / h_fg_flash

    Where:
    - h_condensate: Enthalpy of incoming condensate
    - h_f_flash: Saturated liquid enthalpy at flash pressure
    - h_fg_flash: Latent heat at flash pressure

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.flash_recovery import (
    ...     FlashSteamCalculator,
    ... )
    >>>
    >>> calc = FlashSteamCalculator()
    >>> result = calc.calculate_flash(
    ...     condensate_flow_lb_hr=5000,
    ...     condensate_pressure_psig=150,
    ...     flash_pressure_psig=15,
    ... )
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from .config import FlashRecoveryConfig
from .schemas import FlashSteamInput, FlashSteamOutput

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class FlashConstants:
    """Constants for flash steam calculations."""

    # Minimum flash pressure differential (psi)
    MIN_PRESSURE_DIFFERENTIAL = 5.0

    # Typical flash tank efficiency
    FLASH_TANK_EFFICIENCY = 0.95

    # Steam table reference data
    # Format: psig -> (T_sat, h_f, h_fg, h_g)
    SATURATION_DATA = {
        0: (212.0, 180.2, 970.3, 1150.5),
        5: (227.1, 195.5, 960.1, 1155.6),
        10: (239.4, 208.5, 951.9, 1160.4),
        15: (250.3, 218.9, 945.4, 1164.3),
        20: (259.3, 227.8, 939.3, 1167.1),
        30: (274.0, 243.4, 928.6, 1172.0),
        50: (298.0, 267.6, 911.0, 1178.6),
        75: (320.3, 290.7, 893.7, 1184.4),
        100: (337.9, 309.0, 879.5, 1188.5),
        125: (352.4, 324.4, 867.0, 1191.4),
        150: (365.9, 339.2, 856.8, 1196.0),
        175: (377.5, 351.8, 846.8, 1198.6),
        200: (387.9, 362.2, 837.4, 1199.6),
        250: (406.1, 381.2, 820.1, 1201.3),
        300: (421.7, 397.0, 804.3, 1201.3),
        350: (435.4, 411.0, 789.4, 1200.4),
        400: (448.0, 424.2, 774.4, 1198.6),
        450: (459.3, 436.2, 760.0, 1196.2),
        500: (470.0, 447.7, 747.1, 1194.8),
        600: (489.0, 468.4, 721.4, 1189.8),
    }


# =============================================================================
# FLASH STEAM CALCULATOR
# =============================================================================

class FlashSteamCalculator:
    """
    Calculator for flash steam recovery thermodynamics.

    Calculates the fraction of condensate that flashes to steam
    when pressure is reduced, using rigorous thermodynamic relationships.
    """

    def __init__(self) -> None:
        """Initialize flash steam calculator."""
        logger.debug("FlashSteamCalculator initialized")

    def get_saturation_properties(
        self,
        pressure_psig: float,
    ) -> Dict[str, float]:
        """
        Get saturation properties by interpolation.

        Args:
            pressure_psig: Gauge pressure (psig)

        Returns:
            Dictionary with T_sat, h_f, h_fg, h_g
        """
        pressures = sorted(FlashConstants.SATURATION_DATA.keys())

        # Clamp to valid range
        pressure_psig = max(0, min(600, pressure_psig))

        # Find bracketing pressures
        p_low = pressures[0]
        p_high = pressures[-1]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p_low = pressures[i]
                p_high = pressures[i + 1]
                break

        # Get data
        data_low = FlashConstants.SATURATION_DATA[p_low]
        data_high = FlashConstants.SATURATION_DATA[p_high]

        # Interpolation factor
        if p_high > p_low:
            factor = (pressure_psig - p_low) / (p_high - p_low)
        else:
            factor = 0

        # Interpolate
        def interp(idx: int) -> float:
            return data_low[idx] + factor * (data_high[idx] - data_low[idx])

        return {
            "T_sat_f": interp(0),
            "h_f_btu_lb": interp(1),
            "h_fg_btu_lb": interp(2),
            "h_g_btu_lb": interp(3),
        }

    def calculate_flash_fraction(
        self,
        condensate_pressure_psig: float,
        flash_pressure_psig: float,
        condensate_temperature_f: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the thermodynamic flash fraction.

        The flash fraction is the mass percentage of condensate
        that converts to steam when pressure is reduced.

        Args:
            condensate_pressure_psig: Incoming condensate pressure (psig)
            flash_pressure_psig: Flash vessel pressure (psig)
            condensate_temperature_f: Optional - defaults to saturation

        Returns:
            Tuple of (flash_fraction, calculation_details)

        Raises:
            ValueError: If flash pressure >= condensate pressure
        """
        if flash_pressure_psig >= condensate_pressure_psig:
            raise ValueError(
                "Flash pressure must be less than condensate pressure"
            )

        # Get saturation properties at both pressures
        cond_props = self.get_saturation_properties(condensate_pressure_psig)
        flash_props = self.get_saturation_properties(flash_pressure_psig)

        # Condensate enthalpy (at saturation by default)
        if condensate_temperature_f is None:
            h_condensate = cond_props["h_f_btu_lb"]
        else:
            # If subcooled, adjust enthalpy
            if condensate_temperature_f < cond_props["T_sat_f"]:
                # h = h_f - Cp * (T_sat - T)
                subcooling = cond_props["T_sat_f"] - condensate_temperature_f
                h_condensate = cond_props["h_f_btu_lb"] - 1.0 * subcooling
            else:
                h_condensate = cond_props["h_f_btu_lb"]

        # Flash calculation
        # Energy balance: h_condensate = x * h_g_flash + (1-x) * h_f_flash
        # Solving for x (flash fraction):
        # x = (h_condensate - h_f_flash) / h_fg_flash

        h_f_flash = flash_props["h_f_btu_lb"]
        h_fg_flash = flash_props["h_fg_btu_lb"]

        flash_fraction = (h_condensate - h_f_flash) / h_fg_flash

        # Validate (should be 0-1)
        flash_fraction = max(0, min(1, flash_fraction))

        details = {
            "condensate_pressure_psig": condensate_pressure_psig,
            "flash_pressure_psig": flash_pressure_psig,
            "condensate_enthalpy_btu_lb": h_condensate,
            "h_f_flash_btu_lb": h_f_flash,
            "h_fg_flash_btu_lb": h_fg_flash,
            "h_g_flash_btu_lb": flash_props["h_g_btu_lb"],
            "flash_temperature_f": flash_props["T_sat_f"],
        }

        return flash_fraction, details

    def calculate_flash(
        self,
        condensate_flow_lb_hr: float,
        condensate_pressure_psig: float,
        flash_pressure_psig: float,
        condensate_temperature_f: Optional[float] = None,
    ) -> FlashSteamOutput:
        """
        Calculate complete flash steam recovery.

        Args:
            condensate_flow_lb_hr: Condensate flow rate (lb/hr)
            condensate_pressure_psig: Incoming pressure (psig)
            flash_pressure_psig: Flash vessel pressure (psig)
            condensate_temperature_f: Optional condensate temperature

        Returns:
            FlashSteamOutput with complete analysis
        """
        # Calculate flash fraction
        flash_fraction, details = self.calculate_flash_fraction(
            condensate_pressure_psig,
            flash_pressure_psig,
            condensate_temperature_f,
        )

        flash_fraction_pct = flash_fraction * 100

        # Calculate mass flows
        flash_steam_lb_hr = condensate_flow_lb_hr * flash_fraction
        residual_condensate_lb_hr = condensate_flow_lb_hr * (1 - flash_fraction)

        # Apply typical efficiency
        flash_steam_lb_hr *= FlashConstants.FLASH_TANK_EFFICIENCY

        # Energy in flash steam
        h_g_flash = details["h_g_flash_btu_lb"]
        energy_recovered_btu_hr = flash_steam_lb_hr * h_g_flash

        # Recovery efficiency (vs theoretical)
        theoretical_energy = condensate_flow_lb_hr * flash_fraction * h_g_flash
        recovery_efficiency = (
            (energy_recovered_btu_hr / theoretical_energy * 100)
            if theoretical_energy > 0 else 0
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            condensate_flow_lb_hr,
            condensate_pressure_psig,
            flash_pressure_psig,
        )

        return FlashSteamOutput(
            timestamp=datetime.now(timezone.utc),
            flash_fraction_pct=flash_fraction_pct,
            flash_steam_lb_hr=flash_steam_lb_hr,
            residual_condensate_lb_hr=residual_condensate_lb_hr,
            condensate_enthalpy_in_btu_lb=details["condensate_enthalpy_btu_lb"],
            flash_steam_enthalpy_btu_lb=h_g_flash,
            residual_enthalpy_btu_lb=details["h_f_flash_btu_lb"],
            energy_recovered_btu_hr=energy_recovered_btu_hr,
            recovery_efficiency_pct=recovery_efficiency,
            provenance_hash=provenance_hash,
            formula_reference="Thermodynamic flash calculation (IAPWS-IF97)",
        )

    def _calculate_provenance_hash(
        self,
        flow: float,
        p_cond: float,
        p_flash: float,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "condensate_flow_lb_hr": flow,
            "condensate_pressure_psig": p_cond,
            "flash_pressure_psig": p_flash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# FLASH TANK SIZER
# =============================================================================

class FlashTankSizer:
    """
    Flash tank sizing calculator.

    Calculates required flash tank dimensions based on
    vapor-liquid separation requirements.
    """

    def __init__(self) -> None:
        """Initialize flash tank sizer."""
        self.flash_calc = FlashSteamCalculator()

        logger.debug("FlashTankSizer initialized")

    def size_flash_tank(
        self,
        condensate_flow_lb_hr: float,
        condensate_pressure_psig: float,
        flash_pressure_psig: float,
        separation_velocity_ft_s: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Size a flash tank for given conditions.

        Args:
            condensate_flow_lb_hr: Condensate flow (lb/hr)
            condensate_pressure_psig: Inlet pressure (psig)
            flash_pressure_psig: Operating pressure (psig)
            separation_velocity_ft_s: Maximum vapor velocity (ft/s)

        Returns:
            Dictionary with sizing results
        """
        # Calculate flash steam generation
        result = self.flash_calc.calculate_flash(
            condensate_flow_lb_hr,
            condensate_pressure_psig,
            flash_pressure_psig,
        )

        flash_steam_lb_hr = result.flash_steam_lb_hr

        # Get steam properties at flash pressure
        flash_props = self.flash_calc.get_saturation_properties(flash_pressure_psig)

        # Estimate specific volume at flash conditions
        # Using ideal gas approximation: v = RT/P
        # For steam, rough estimate: v (ft3/lb) ~ 26.8 * (T+460) / (P+14.7)
        t_sat_r = flash_props["T_sat_f"] + 459.67
        p_abs = flash_pressure_psig + 14.696
        specific_volume_ft3_lb = 26.8 * t_sat_r / (p_abs * 144)  # Approximate

        # Volumetric flow rate
        volumetric_flow_ft3_hr = flash_steam_lb_hr * specific_volume_ft3_lb
        volumetric_flow_ft3_s = volumetric_flow_ft3_hr / 3600

        # Required cross-sectional area
        area_ft2 = volumetric_flow_ft3_s / separation_velocity_ft_s

        # Diameter for circular tank
        diameter_ft = math.sqrt(4 * area_ft2 / math.pi)
        diameter_in = diameter_ft * 12

        # Height (typical L/D ratio of 3:1)
        height_ft = diameter_ft * 3
        height_in = height_ft * 12

        # Volume
        volume_ft3 = math.pi * (diameter_ft / 2) ** 2 * height_ft
        volume_gal = volume_ft3 * 7.48

        # Residence time (seconds)
        residence_time_s = volume_ft3 / volumetric_flow_ft3_s if volumetric_flow_ft3_s > 0 else 0

        # Round up to standard sizes
        standard_diameters = [12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96]
        recommended_diameter_in = min(
            d for d in standard_diameters if d >= diameter_in
        ) if diameter_in <= 96 else math.ceil(diameter_in)

        return {
            "flash_steam_lb_hr": flash_steam_lb_hr,
            "flash_fraction_pct": result.flash_fraction_pct,
            "specific_volume_ft3_lb": specific_volume_ft3_lb,
            "volumetric_flow_ft3_hr": volumetric_flow_ft3_hr,
            "required_area_ft2": area_ft2,
            "calculated_diameter_in": diameter_in,
            "recommended_diameter_in": recommended_diameter_in,
            "calculated_height_in": height_in,
            "volume_ft3": volume_ft3,
            "volume_gal": volume_gal,
            "residence_time_s": residence_time_s,
            "design_velocity_ft_s": separation_velocity_ft_s,
        }


# =============================================================================
# MULTI-STAGE FLASH OPTIMIZER
# =============================================================================

class MultiStageFlashOptimizer:
    """
    Optimizer for multi-stage flash steam recovery.

    Evaluates benefits of multiple flash stages for high-pressure
    condensate recovery.
    """

    def __init__(
        self,
        fuel_cost_per_mmbtu: float = 5.0,
        operating_hours_per_year: int = 8000,
    ) -> None:
        """
        Initialize multi-stage optimizer.

        Args:
            fuel_cost_per_mmbtu: Fuel cost ($/MMBTU)
            operating_hours_per_year: Annual operating hours
        """
        self.flash_calc = FlashSteamCalculator()
        self.fuel_cost = fuel_cost_per_mmbtu
        self.operating_hours = operating_hours_per_year

        logger.debug("MultiStageFlashOptimizer initialized")

    def optimize_stages(
        self,
        condensate_flow_lb_hr: float,
        condensate_pressure_psig: float,
        final_pressure_psig: float,
        max_stages: int = 3,
    ) -> Dict[str, Any]:
        """
        Optimize multi-stage flash configuration.

        Args:
            condensate_flow_lb_hr: Condensate flow (lb/hr)
            condensate_pressure_psig: Initial pressure (psig)
            final_pressure_psig: Final pressure (psig)
            max_stages: Maximum number of stages to consider

        Returns:
            Dictionary with optimization results
        """
        results = {}

        # Evaluate 1 to max_stages
        for n_stages in range(1, max_stages + 1):
            stage_results = self._evaluate_stages(
                condensate_flow_lb_hr,
                condensate_pressure_psig,
                final_pressure_psig,
                n_stages,
            )
            results[f"{n_stages}_stage"] = stage_results

        # Find optimal configuration
        optimal_stages = 1
        optimal_value = 0

        for n_stages in range(1, max_stages + 1):
            key = f"{n_stages}_stage"
            # Consider net present value or simple payback
            # For simplicity, maximize annual savings minus incremental cost
            annual_savings = results[key]["annual_savings_usd"]
            incremental_cost = (n_stages - 1) * 25000  # Rough estimate per stage

            net_value = annual_savings - incremental_cost / 5  # 5-year payback

            if net_value > optimal_value:
                optimal_value = net_value
                optimal_stages = n_stages

        return {
            "stage_analysis": results,
            "optimal_stages": optimal_stages,
            "optimal_configuration": results[f"{optimal_stages}_stage"],
            "recommendation": self._generate_recommendation(
                optimal_stages,
                results[f"{optimal_stages}_stage"],
            ),
        }

    def _evaluate_stages(
        self,
        condensate_flow_lb_hr: float,
        high_pressure_psig: float,
        low_pressure_psig: float,
        n_stages: int,
    ) -> Dict[str, Any]:
        """Evaluate a specific number of flash stages."""
        if n_stages < 1:
            raise ValueError("Number of stages must be >= 1")

        # Calculate intermediate pressures (equal ratios)
        pressure_ratio = (high_pressure_psig + 14.696) / (low_pressure_psig + 14.696)
        stage_ratio = pressure_ratio ** (1 / n_stages)

        pressures = [high_pressure_psig]
        for i in range(n_stages):
            next_p_abs = (pressures[-1] + 14.696) / stage_ratio
            next_p_gauge = next_p_abs - 14.696
            pressures.append(max(0, next_p_gauge))

        # Ensure final pressure matches target
        pressures[-1] = low_pressure_psig

        # Calculate flash at each stage
        stages = []
        remaining_flow = condensate_flow_lb_hr
        total_flash_steam = 0.0
        total_energy_recovered = 0.0

        for i in range(n_stages):
            p_in = pressures[i]
            p_out = pressures[i + 1]

            result = self.flash_calc.calculate_flash(
                remaining_flow,
                p_in,
                p_out,
            )

            stages.append({
                "stage": i + 1,
                "inlet_pressure_psig": p_in,
                "outlet_pressure_psig": p_out,
                "flash_fraction_pct": result.flash_fraction_pct,
                "flash_steam_lb_hr": result.flash_steam_lb_hr,
                "energy_recovered_btu_hr": result.energy_recovered_btu_hr,
            })

            total_flash_steam += result.flash_steam_lb_hr
            total_energy_recovered += result.energy_recovered_btu_hr
            remaining_flow = result.residual_condensate_lb_hr

        # Calculate overall recovery
        overall_flash_pct = (total_flash_steam / condensate_flow_lb_hr * 100)

        # Economic analysis
        # Assume 80% boiler efficiency for fuel savings
        boiler_efficiency = 0.82
        fuel_saved_mmbtu_hr = total_energy_recovered / (1_000_000 * boiler_efficiency)
        hourly_savings = fuel_saved_mmbtu_hr * self.fuel_cost
        annual_savings = hourly_savings * self.operating_hours

        return {
            "stages": stages,
            "intermediate_pressures_psig": pressures,
            "total_flash_steam_lb_hr": total_flash_steam,
            "overall_flash_pct": overall_flash_pct,
            "total_energy_recovered_btu_hr": total_energy_recovered,
            "fuel_saved_mmbtu_hr": fuel_saved_mmbtu_hr,
            "hourly_savings_usd": hourly_savings,
            "annual_savings_usd": annual_savings,
            "residual_condensate_lb_hr": remaining_flow,
        }

    def _generate_recommendation(
        self,
        optimal_stages: int,
        results: Dict[str, Any],
    ) -> str:
        """Generate recommendation text."""
        if optimal_stages == 1:
            return (
                f"Single-stage flash recovery recommended. "
                f"Expected recovery: {results['total_flash_steam_lb_hr']:.0f} lb/hr, "
                f"Annual savings: ${results['annual_savings_usd']:,.0f}"
            )
        else:
            return (
                f"{optimal_stages}-stage flash recovery recommended. "
                f"Total recovery: {results['total_flash_steam_lb_hr']:.0f} lb/hr "
                f"({results['overall_flash_pct']:.1f}%), "
                f"Annual savings: ${results['annual_savings_usd']:,.0f}"
            )


# =============================================================================
# FLASH RECOVERY OPTIMIZER
# =============================================================================

class FlashRecoveryOptimizer:
    """
    Complete flash steam recovery optimization.

    Combines flash calculations, tank sizing, and economic analysis
    for comprehensive flash recovery optimization.

    Example:
        >>> config = FlashRecoveryConfig(
        ...     condensate_pressure_psig=150,
        ...     flash_pressure_psig=15,
        ... )
        >>> optimizer = FlashRecoveryOptimizer(config)
        >>> result = optimizer.analyze(condensate_flow_lb_hr=5000)
    """

    def __init__(
        self,
        config: FlashRecoveryConfig,
    ) -> None:
        """
        Initialize flash recovery optimizer.

        Args:
            config: Flash recovery configuration
        """
        self.config = config
        self.flash_calc = FlashSteamCalculator()
        self.tank_sizer = FlashTankSizer()
        self.multi_stage = MultiStageFlashOptimizer(
            fuel_cost_per_mmbtu=config.fuel_cost_per_mmbtu,
            operating_hours_per_year=config.operating_hours_per_year,
        )

        logger.info(
            f"FlashRecoveryOptimizer initialized: "
            f"P_cond={config.condensate_pressure_psig} psig, "
            f"P_flash={config.flash_pressure_psig} psig"
        )

    def analyze(
        self,
        condensate_flow_lb_hr: float,
        condensate_temperature_f: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze flash steam recovery opportunity.

        Args:
            condensate_flow_lb_hr: Condensate flow (lb/hr)
            condensate_temperature_f: Optional temperature

        Returns:
            Dictionary with complete analysis
        """
        # Basic flash calculation
        flash_result = self.flash_calc.calculate_flash(
            condensate_flow_lb_hr=condensate_flow_lb_hr,
            condensate_pressure_psig=self.config.condensate_pressure_psig,
            flash_pressure_psig=self.config.flash_pressure_psig,
            condensate_temperature_f=condensate_temperature_f,
        )

        # Tank sizing
        tank_sizing = self.tank_sizer.size_flash_tank(
            condensate_flow_lb_hr=condensate_flow_lb_hr,
            condensate_pressure_psig=self.config.condensate_pressure_psig,
            flash_pressure_psig=self.config.flash_pressure_psig,
        )

        # Economic analysis
        # Fuel savings (assuming 82% boiler efficiency)
        boiler_efficiency = 0.82
        fuel_saved_mmbtu_hr = (
            flash_result.energy_recovered_btu_hr /
            (1_000_000 * boiler_efficiency)
        )
        hourly_savings = fuel_saved_mmbtu_hr * self.config.fuel_cost_per_mmbtu
        annual_savings = hourly_savings * self.config.operating_hours_per_year

        # Add annual savings to flash result
        flash_result.annual_savings_usd = annual_savings

        # Check if multi-stage would help
        pressure_ratio = (
            (self.config.condensate_pressure_psig + 14.696) /
            (self.config.flash_pressure_psig + 14.696)
        )

        multi_stage_recommendation = None
        if pressure_ratio > 5:
            # Significant pressure drop - evaluate multi-stage
            multi_stage_result = self.multi_stage.optimize_stages(
                condensate_flow_lb_hr=condensate_flow_lb_hr,
                condensate_pressure_psig=self.config.condensate_pressure_psig,
                final_pressure_psig=self.config.flash_pressure_psig,
                max_stages=3,
            )
            if multi_stage_result["optimal_stages"] > 1:
                multi_stage_recommendation = multi_stage_result

        # Generate recommendations
        recommendations = self._generate_recommendations(
            flash_result,
            tank_sizing,
            annual_savings,
            multi_stage_recommendation,
        )

        return {
            "flash_analysis": flash_result,
            "tank_sizing": tank_sizing,
            "economic_analysis": {
                "fuel_saved_mmbtu_hr": fuel_saved_mmbtu_hr,
                "hourly_savings_usd": hourly_savings,
                "annual_savings_usd": annual_savings,
                "fuel_cost_per_mmbtu": self.config.fuel_cost_per_mmbtu,
                "operating_hours_per_year": self.config.operating_hours_per_year,
            },
            "multi_stage_analysis": multi_stage_recommendation,
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self,
        flash_result: FlashSteamOutput,
        tank_sizing: Dict[str, Any],
        annual_savings: float,
        multi_stage: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate analysis recommendations."""
        recommendations = []

        # Basic flash recovery
        if flash_result.flash_fraction_pct > 5:
            recommendations.append(
                f"Flash recovery viable: {flash_result.flash_fraction_pct:.1f}% "
                f"flash potential with ${annual_savings:,.0f}/year savings"
            )

        # Tank sizing
        recommendations.append(
            f"Recommended flash tank: {tank_sizing['recommended_diameter_in']:.0f}\" "
            f"diameter, {tank_sizing['volume_gal']:.0f} gallon capacity"
        )

        # Multi-stage
        if multi_stage:
            multi_savings = (
                multi_stage["optimal_configuration"]["annual_savings_usd"]
            )
            additional_savings = multi_savings - annual_savings

            if additional_savings > 5000:
                recommendations.append(
                    f"Consider {multi_stage['optimal_stages']}-stage flash for "
                    f"additional ${additional_savings:,.0f}/year"
                )

        # Destination header
        if self.config.flash_steam_destination:
            recommendations.append(
                f"Route flash steam to {self.config.flash_steam_destination} header"
            )

        return recommendations
