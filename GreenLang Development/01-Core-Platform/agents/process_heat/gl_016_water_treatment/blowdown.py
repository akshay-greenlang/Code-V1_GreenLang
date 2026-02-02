"""
GL-016 WATERGUARD Agent - Blowdown Optimization Module

Implements blowdown optimization including:
- Cycles of concentration calculation
- Optimal blowdown rate determination
- Energy and water savings analysis
- Heat recovery potential
- Continuous vs intermittent blowdown selection

All calculations are deterministic with zero hallucination.

References:
    - ASME Boiler Efficiency Guidelines
    - DOE Steam Best Practices
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    BlowdownInput,
    BlowdownOutput,
    BlowdownType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class BlowdownConstants:
    """Constants for blowdown calculations."""

    # Water properties
    WATER_DENSITY_LB_GAL = 8.34
    BTU_PER_LB_F = 1.0

    # Steam table reference points for flash steam calculations
    # Format: pressure_psig: (saturation_temp_F, hf_BTU_lb, hfg_BTU_lb)
    STEAM_PROPERTIES = {
        0: (212.0, 180.2, 970.3),
        5: (227.1, 195.2, 960.1),
        10: (240.1, 208.5, 952.1),
        15: (250.3, 218.8, 945.3),
        50: (298.0, 267.5, 912.0),
        100: (338.0, 309.0, 881.0),
        150: (366.0, 339.0, 857.0),
        200: (388.0, 362.0, 837.0),
        250: (406.0, 382.0, 820.0),
        300: (422.0, 399.0, 804.0),
        400: (448.0, 428.0, 775.0),
        500: (470.0, 453.0, 750.0),
        600: (489.0, 475.0, 728.0),
    }

    # Economic factors
    HOURS_PER_YEAR = 8760
    BOILER_EFFICIENCY = 0.82

    # Blowdown limits
    MIN_BLOWDOWN_PCT = 1.0
    MAX_BLOWDOWN_PCT = 10.0
    MIN_CYCLES = 3.0
    MAX_CYCLES_RECOMMENDED = 10.0


# =============================================================================
# BLOWDOWN OPTIMIZER CLASS
# =============================================================================

class BlowdownOptimizer:
    """
    Optimizes boiler blowdown for water and energy savings.

    Calculates optimal cycles of concentration and blowdown rate
    based on water quality limits and operating conditions.

    Attributes:
        boiler_efficiency: Assumed boiler efficiency

    Example:
        >>> optimizer = BlowdownOptimizer()
        >>> result = optimizer.optimize(blowdown_input)
        >>> print(f"Savings: ${result.total_savings_usd_yr}/year")
    """

    def __init__(
        self,
        boiler_efficiency: float = 0.82,
    ) -> None:
        """
        Initialize BlowdownOptimizer.

        Args:
            boiler_efficiency: Boiler thermal efficiency (0-1)
        """
        self.boiler_efficiency = boiler_efficiency
        logger.info("BlowdownOptimizer initialized")

    def optimize(self, input_data: BlowdownInput) -> BlowdownOutput:
        """
        Optimize blowdown rate and calculate savings.

        Args:
            input_data: Blowdown operating data

        Returns:
            BlowdownOutput with optimization results
        """
        start_time = datetime.now(timezone.utc)
        logger.debug("Optimizing blowdown")

        # Calculate current cycles of concentration
        current_cycles = self.calculate_cycles_of_concentration(
            input_data.boiler_tds_ppm,
            input_data.feedwater_tds_ppm,
        )

        # Calculate current blowdown rate and flow
        current_blowdown_rate = self.calculate_blowdown_rate(current_cycles)
        current_blowdown_flow = self.calculate_blowdown_flow(
            input_data.steam_flow_rate_lb_hr,
            current_blowdown_rate,
        )

        # Calculate optimal cycles based on TDS limit
        optimal_cycles = self.calculate_optimal_cycles(
            input_data.feedwater_tds_ppm,
            input_data.tds_max_ppm,
        )

        # Limit to practical maximum
        optimal_cycles = min(optimal_cycles, BlowdownConstants.MAX_CYCLES_RECOMMENDED)

        # Calculate optimal blowdown rate and flow
        optimal_blowdown_rate = self.calculate_blowdown_rate(optimal_cycles)
        optimal_blowdown_flow = self.calculate_blowdown_flow(
            input_data.steam_flow_rate_lb_hr,
            optimal_blowdown_rate,
        )

        # Calculate potential savings
        blowdown_reduction_pct = (
            (current_blowdown_rate - optimal_blowdown_rate) / current_blowdown_rate * 100
            if current_blowdown_rate > 0 else 0
        )

        # Energy savings
        energy_savings = self.calculate_energy_savings(
            current_blowdown_flow,
            optimal_blowdown_flow,
            input_data.operating_pressure_psig,
            input_data.blowdown_heat_recovery_enabled,
        )

        # Water savings
        water_savings = self.calculate_water_savings(
            current_blowdown_flow,
            optimal_blowdown_flow,
        )

        # Total economic savings
        total_savings = self.calculate_total_savings(
            energy_savings,
            water_savings,
            input_data.fuel_cost_per_mmbtu,
            input_data.water_cost_per_kgal,
            input_data.chemical_cost_per_kgal,
        )

        # Heat recovery potential
        heat_recovery, flash_steam = self.calculate_heat_recovery_potential(
            optimal_blowdown_flow,
            input_data.operating_pressure_psig,
            input_data.flash_tank_pressure_psig,
        )

        # Check if optimized values are within limits
        within_limits = (
            optimal_blowdown_rate >= BlowdownConstants.MIN_BLOWDOWN_PCT and
            optimal_cycles >= BlowdownConstants.MIN_CYCLES
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            input_data, current_cycles, optimal_cycles,
            blowdown_reduction_pct, energy_savings,
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(input_data)
        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return BlowdownOutput(
            timestamp=datetime.now(timezone.utc),
            current_cycles_of_concentration=round(current_cycles, 2),
            current_blowdown_rate_pct=round(current_blowdown_rate, 2),
            current_blowdown_flow_lb_hr=round(current_blowdown_flow, 1),
            optimal_cycles_of_concentration=round(optimal_cycles, 2),
            optimal_blowdown_rate_pct=round(optimal_blowdown_rate, 2),
            optimal_blowdown_flow_lb_hr=round(optimal_blowdown_flow, 1),
            blowdown_reduction_pct=round(blowdown_reduction_pct, 1),
            energy_savings_mmbtu_yr=round(energy_savings, 1),
            water_savings_kgal_yr=round(water_savings, 1),
            total_savings_usd_yr=round(total_savings, 0),
            heat_recovery_potential_btu_hr=round(heat_recovery, 0),
            flash_steam_lb_hr=round(flash_steam, 1) if flash_steam else None,
            optimization_status="complete",
            within_limits=within_limits,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

    def calculate_cycles_of_concentration(
        self,
        boiler_tds_ppm: float,
        feedwater_tds_ppm: float,
    ) -> float:
        """
        Calculate cycles of concentration.

        Cycles = Boiler TDS / Feedwater TDS

        Args:
            boiler_tds_ppm: Boiler water TDS (ppm)
            feedwater_tds_ppm: Feedwater TDS (ppm)

        Returns:
            Cycles of concentration
        """
        if feedwater_tds_ppm <= 0:
            logger.warning("Feedwater TDS is zero or negative")
            return 1.0

        cycles = boiler_tds_ppm / feedwater_tds_ppm
        return max(cycles, 1.0)

    def calculate_blowdown_rate(self, cycles: float) -> float:
        """
        Calculate blowdown rate from cycles of concentration.

        Blowdown % = 100 / Cycles

        Args:
            cycles: Cycles of concentration

        Returns:
            Blowdown rate as percentage
        """
        if cycles <= 1:
            return 100.0
        return 100.0 / cycles

    def calculate_blowdown_flow(
        self,
        steam_flow_lb_hr: float,
        blowdown_rate_pct: float,
    ) -> float:
        """
        Calculate blowdown flow rate.

        Args:
            steam_flow_lb_hr: Steam flow rate (lb/hr)
            blowdown_rate_pct: Blowdown rate (%)

        Returns:
            Blowdown flow (lb/hr)
        """
        # Blowdown flow = Feedwater flow * (Blowdown% / 100)
        # Feedwater flow ~ Steam flow + Blowdown flow
        # Solving: Blowdown = Steam * (BD% / (100 - BD%))
        if blowdown_rate_pct >= 100:
            return steam_flow_lb_hr

        return steam_flow_lb_hr * (blowdown_rate_pct / (100 - blowdown_rate_pct))

    def calculate_optimal_cycles(
        self,
        feedwater_tds_ppm: float,
        tds_limit_ppm: float,
    ) -> float:
        """
        Calculate optimal cycles of concentration based on TDS limit.

        Args:
            feedwater_tds_ppm: Feedwater TDS (ppm)
            tds_limit_ppm: Maximum boiler TDS (ppm)

        Returns:
            Optimal cycles of concentration
        """
        if feedwater_tds_ppm <= 0:
            return BlowdownConstants.MIN_CYCLES

        optimal_cycles = tds_limit_ppm / feedwater_tds_ppm

        # Apply practical limits
        optimal_cycles = max(optimal_cycles, BlowdownConstants.MIN_CYCLES)
        optimal_cycles = min(optimal_cycles, BlowdownConstants.MAX_CYCLES_RECOMMENDED)

        return optimal_cycles

    def calculate_energy_savings(
        self,
        current_blowdown_lb_hr: float,
        optimal_blowdown_lb_hr: float,
        pressure_psig: float,
        heat_recovery_enabled: bool = False,
    ) -> float:
        """
        Calculate annual energy savings from blowdown reduction.

        Args:
            current_blowdown_lb_hr: Current blowdown flow (lb/hr)
            optimal_blowdown_lb_hr: Optimal blowdown flow (lb/hr)
            pressure_psig: Operating pressure (psig)
            heat_recovery_enabled: Whether heat recovery is in place

        Returns:
            Annual energy savings (MMBTU/yr)
        """
        blowdown_reduction = current_blowdown_lb_hr - optimal_blowdown_lb_hr
        if blowdown_reduction <= 0:
            return 0.0

        # Get enthalpy at operating pressure
        hf = self._get_liquid_enthalpy(pressure_psig)

        # Assume feedwater at 200F (enthalpy ~168 BTU/lb)
        feedwater_enthalpy = 168.0

        # Energy lost per lb blowdown (above feedwater)
        energy_per_lb = hf - feedwater_enthalpy

        # Apply heat recovery factor
        recovery_factor = 0.5 if heat_recovery_enabled else 0.0
        net_energy_per_lb = energy_per_lb * (1 - recovery_factor)

        # Annual savings (BTU/yr)
        annual_savings_btu = (
            blowdown_reduction *
            net_energy_per_lb *
            BlowdownConstants.HOURS_PER_YEAR
        )

        # Account for boiler efficiency (fuel savings)
        fuel_savings_btu = annual_savings_btu / self.boiler_efficiency

        # Convert to MMBTU
        return fuel_savings_btu / 1_000_000

    def calculate_water_savings(
        self,
        current_blowdown_lb_hr: float,
        optimal_blowdown_lb_hr: float,
    ) -> float:
        """
        Calculate annual water savings from blowdown reduction.

        Args:
            current_blowdown_lb_hr: Current blowdown flow (lb/hr)
            optimal_blowdown_lb_hr: Optimal blowdown flow (lb/hr)

        Returns:
            Annual water savings (kgal/yr)
        """
        blowdown_reduction = current_blowdown_lb_hr - optimal_blowdown_lb_hr
        if blowdown_reduction <= 0:
            return 0.0

        # Convert lb/hr to gal/hr
        gallons_per_hr = blowdown_reduction / BlowdownConstants.WATER_DENSITY_LB_GAL

        # Annual savings
        annual_gallons = gallons_per_hr * BlowdownConstants.HOURS_PER_YEAR

        # Convert to kgal
        return annual_gallons / 1000

    def calculate_total_savings(
        self,
        energy_savings_mmbtu: float,
        water_savings_kgal: float,
        fuel_cost: float,
        water_cost: float,
        chemical_cost: float,
    ) -> float:
        """
        Calculate total annual savings.

        Args:
            energy_savings_mmbtu: Energy savings (MMBTU/yr)
            water_savings_kgal: Water savings (kgal/yr)
            fuel_cost: Fuel cost ($/MMBTU)
            water_cost: Water cost ($/kgal)
            chemical_cost: Chemical cost ($/kgal makeup)

        Returns:
            Total annual savings ($/yr)
        """
        fuel_savings = energy_savings_mmbtu * fuel_cost
        water_savings_cost = water_savings_kgal * water_cost
        chemical_savings = water_savings_kgal * chemical_cost

        return fuel_savings + water_savings_cost + chemical_savings

    def calculate_heat_recovery_potential(
        self,
        blowdown_flow_lb_hr: float,
        boiler_pressure_psig: float,
        flash_tank_pressure_psig: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Calculate blowdown heat recovery potential.

        Args:
            blowdown_flow_lb_hr: Blowdown flow rate (lb/hr)
            boiler_pressure_psig: Boiler operating pressure (psig)
            flash_tank_pressure_psig: Flash tank pressure (psig)

        Returns:
            Tuple of (heat_recovery_btu_hr, flash_steam_lb_hr)
        """
        if flash_tank_pressure_psig is None:
            flash_tank_pressure_psig = 5.0  # Typical flash tank pressure

        # Get enthalpies
        hf_boiler = self._get_liquid_enthalpy(boiler_pressure_psig)
        hf_flash = self._get_liquid_enthalpy(flash_tank_pressure_psig)
        hfg_flash = self._get_latent_heat(flash_tank_pressure_psig)

        # Calculate flash steam fraction
        # Flash steam % = (hf_boiler - hf_flash) / hfg_flash * 100
        if hfg_flash > 0:
            flash_fraction = (hf_boiler - hf_flash) / hfg_flash
        else:
            flash_fraction = 0.0

        flash_fraction = max(0, min(flash_fraction, 0.5))  # Limit to 50%

        # Flash steam flow
        flash_steam_lb_hr = blowdown_flow_lb_hr * flash_fraction

        # Heat recovery from flash steam
        heat_recovery = flash_steam_lb_hr * hfg_flash

        # Add remaining liquid heat
        remaining_liquid = blowdown_flow_lb_hr - flash_steam_lb_hr
        feedwater_enthalpy = 168.0  # Assume 200F feedwater
        liquid_recovery = remaining_liquid * (hf_flash - feedwater_enthalpy) * 0.7  # 70% HX efficiency

        total_recovery = heat_recovery + liquid_recovery

        return total_recovery, flash_steam_lb_hr

    def _get_liquid_enthalpy(self, pressure_psig: float) -> float:
        """Get saturated liquid enthalpy at given pressure."""
        return self._interpolate_steam_property(pressure_psig, 1)

    def _get_latent_heat(self, pressure_psig: float) -> float:
        """Get latent heat of vaporization at given pressure."""
        return self._interpolate_steam_property(pressure_psig, 2)

    def _interpolate_steam_property(
        self,
        pressure_psig: float,
        property_index: int,
    ) -> float:
        """Interpolate steam property from table."""
        pressures = sorted(BlowdownConstants.STEAM_PROPERTIES.keys())

        if pressure_psig <= pressures[0]:
            return BlowdownConstants.STEAM_PROPERTIES[pressures[0]][property_index]
        if pressure_psig >= pressures[-1]:
            return BlowdownConstants.STEAM_PROPERTIES[pressures[-1]][property_index]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                v1 = BlowdownConstants.STEAM_PROPERTIES[p1][property_index]
                v2 = BlowdownConstants.STEAM_PROPERTIES[p2][property_index]
                return v1 + (v2 - v1) * (pressure_psig - p1) / (p2 - p1)

        return BlowdownConstants.STEAM_PROPERTIES[pressures[0]][property_index]

    def _generate_recommendations(
        self,
        input_data: BlowdownInput,
        current_cycles: float,
        optimal_cycles: float,
        reduction_pct: float,
        energy_savings: float,
    ) -> List[str]:
        """Generate blowdown optimization recommendations."""
        recommendations = []

        # Cycles improvement opportunity
        if current_cycles < optimal_cycles * 0.8:
            recommendations.append(
                f"Increase cycles of concentration from {current_cycles:.1f} to "
                f"{optimal_cycles:.1f} for optimal efficiency"
            )

        # Significant savings potential
        if reduction_pct > 10:
            recommendations.append(
                f"Potential {reduction_pct:.0f}% blowdown reduction - "
                f"implement continuous blowdown control"
            )

        # Heat recovery
        if not input_data.blowdown_heat_recovery_enabled and energy_savings > 100:
            recommendations.append(
                "Install blowdown heat recovery to capture flash steam and "
                "preheat makeup water"
            )

        # Conductivity control
        if input_data.boiler_conductivity_umho and input_data.feedwater_conductivity_umho:
            recommendations.append(
                "Use conductivity-based automatic blowdown control for "
                "optimal cycles maintenance"
            )

        # Low cycles warning
        if current_cycles < BlowdownConstants.MIN_CYCLES:
            recommendations.append(
                f"WARNING: Cycles ({current_cycles:.1f}) below minimum - "
                "excessive blowdown, check TDS control"
            )

        # High cycles warning
        if current_cycles > BlowdownConstants.MAX_CYCLES_RECOMMENDED:
            recommendations.append(
                f"WARNING: Cycles ({current_cycles:.1f}) may be too high - "
                "risk of carryover and scaling"
            )

        return recommendations

    def _calculate_provenance_hash(self, input_data: BlowdownInput) -> str:
        """Calculate SHA-256 hash for audit trail."""
        import json
        data_str = json.dumps(input_data.dict(), sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


def calculate_makeup_requirement(
    steam_flow_lb_hr: float,
    condensate_return_pct: float,
    blowdown_rate_pct: float,
) -> float:
    """
    Calculate makeup water requirement.

    Args:
        steam_flow_lb_hr: Steam production rate (lb/hr)
        condensate_return_pct: Condensate return percentage
        blowdown_rate_pct: Blowdown rate percentage

    Returns:
        Makeup water requirement (lb/hr)
    """
    # Losses = Steam not returned + Blowdown + Other (assume 2%)
    steam_lost = steam_flow_lb_hr * (1 - condensate_return_pct / 100)
    blowdown = steam_flow_lb_hr * (blowdown_rate_pct / 100)
    other_losses = steam_flow_lb_hr * 0.02

    makeup = steam_lost + blowdown + other_losses
    return makeup
