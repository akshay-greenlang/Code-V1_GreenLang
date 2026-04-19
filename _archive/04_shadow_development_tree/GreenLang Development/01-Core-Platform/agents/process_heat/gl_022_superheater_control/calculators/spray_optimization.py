"""
GL-022 SUPERHEATER CONTROL - Spray Optimization Calculator Module

This module provides spray desuperheater optimization calculations including:
- Spray water flow rate calculations
- Mixing thermodynamics (steam + water)
- Droplet evaporation calculations
- Spray efficiency calculations
- Water quality impact analysis

All calculations are ZERO-HALLUCINATION deterministic with complete provenance tracking.

Standards Reference:
    - ASME PTC 19.11: Steam and Water Sampling
    - ISA-77.43: Fossil Fuel Power Plant Desuperheater Controls
    - EPRI Guidelines for Spray Desuperheater Operation

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.calculators.spray_optimization import (
    ...     SprayFlowCalculator,
    ...     SprayEfficiencyCalculator,
    ... )
    >>>
    >>> calc = SprayFlowCalculator()
    >>> result = calc.calculate_spray_requirements(
    ...     steam_flow_lb_hr=100000,
    ...     inlet_temp_f=950.0,
    ...     target_temp_f=850.0,
    ...     spray_water_temp_f=250.0,
    ...     pressure_psig=600.0,
    ... )
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from .thermodynamics import (
    SteamThermodynamicsCalculator,
    EnthalpyBalanceCalculator,
    IAPWSIF97Constants,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - SPRAY DESUPERHEATER PARAMETERS
# =============================================================================

class SprayDesuperheaterConstants:
    """
    Constants for spray desuperheater calculations.

    Based on ISA-77.43 and EPRI guidelines for spray desuperheater design
    and operation in power plant applications.
    """

    # Spray nozzle characteristics
    NOZZLE_DISCHARGE_COEFF = 0.65  # Typical for spray nozzles
    NOZZLE_PRESSURE_DROP_MIN_PSI = 20.0  # Minimum pressure drop for atomization
    NOZZLE_PRESSURE_DROP_MAX_PSI = 150.0  # Maximum for reasonable wear

    # Droplet characteristics
    DROPLET_DIAMETER_MICRON_MIN = 50  # Minimum for proper atomization
    DROPLET_DIAMETER_MICRON_MAX = 500  # Maximum before poor evaporation
    DROPLET_DIAMETER_OPTIMAL = 150  # Optimal droplet size (microns)

    # Evaporation zone requirements
    MIN_EVAPORATION_LENGTH_FT = 5.0  # Minimum straight run after spray
    EVAPORATION_LENGTH_PER_INCH_DIA = 6.0  # L/D ratio

    # Efficiency factors
    MAX_SPRAY_RATIO = 0.15  # Maximum spray/steam ratio (15%)
    IDEAL_SPRAY_RATIO = 0.08  # Ideal spray/steam ratio (8%)
    MIN_SUPERHEAT_AFTER_SPRAY_F = 25.0  # Minimum superheat to avoid wet steam

    # Mixing efficiency
    PERFECT_MIXING_EFFICIENCY = 1.0
    TYPICAL_MIXING_EFFICIENCY = 0.95
    POOR_MIXING_EFFICIENCY = 0.85

    # Water quality limits (EPRI guidelines)
    MAX_TDS_PPM = 3.0  # Total dissolved solids in spray water
    MAX_SILICA_PPM = 0.02  # Silica limit
    MAX_IRON_PPB = 5.0  # Iron limit
    MAX_COPPER_PPB = 2.0  # Copper limit
    MAX_CONDUCTIVITY_US_CM = 1.0  # Cation conductivity limit


class DropletEvaporationConstants:
    """Constants for droplet evaporation calculations."""

    # Water properties at typical spray conditions
    WATER_DENSITY_LB_FT3 = 62.4
    WATER_SURFACE_TENSION_LB_FT = 0.005  # At ~250F

    # Heat transfer coefficients
    DROPLET_NUSSELT_CORRELATION_A = 2.0  # Nu = A + B*Re^0.5*Pr^0.33
    DROPLET_NUSSELT_CORRELATION_B = 0.6

    # Evaporation time factor (dimensionless)
    EVAPORATION_TIME_FACTOR = 0.85


# =============================================================================
# DATA CLASSES FOR SPRAY CALCULATIONS
# =============================================================================

@dataclass
class SprayRequirementsResult:
    """Result of spray flow requirements calculation."""
    spray_flow_lb_hr: float
    spray_ratio_pct: float
    inlet_enthalpy_btu_lb: float
    outlet_enthalpy_btu_lb: float
    water_enthalpy_btu_lb: float
    heat_removed_btu_hr: float
    outlet_temperature_f: float
    superheat_remaining_f: float
    is_within_limits: bool
    warnings: List[str]
    calculation_method: str
    provenance_hash: str


@dataclass
class MixingThermodynamicsResult:
    """Result of steam-water mixing calculation."""
    mixed_temperature_f: float
    mixed_enthalpy_btu_lb: float
    mixed_flow_lb_hr: float
    mixing_efficiency: float
    temperature_drop_f: float
    heat_absorbed_by_spray_btu_hr: float
    outlet_superheat_f: float
    calculation_method: str
    provenance_hash: str


@dataclass
class DropletEvaporationResult:
    """Result of droplet evaporation calculation."""
    evaporation_time_ms: float
    evaporation_distance_ft: float
    initial_droplet_diameter_micron: float
    final_droplet_diameter_micron: float
    complete_evaporation: bool
    evaporation_efficiency_pct: float
    steam_velocity_fps: float
    calculation_method: str
    provenance_hash: str


@dataclass
class SprayEfficiencyResult:
    """Result of spray efficiency analysis."""
    overall_efficiency_pct: float
    atomization_efficiency_pct: float
    mixing_efficiency_pct: float
    evaporation_efficiency_pct: float
    thermal_efficiency_pct: float
    spray_pattern_quality: str  # "excellent", "good", "fair", "poor"
    recommendations: List[str]
    calculation_method: str
    provenance_hash: str


@dataclass
class WaterQualityImpactResult:
    """Result of water quality impact analysis."""
    tds_in_steam_ppm: float
    silica_in_steam_ppm: float
    estimated_deposition_rate_mg_ft2_hr: float
    fouling_risk_level: str  # "low", "medium", "high", "critical"
    scale_formation_potential: float
    recommended_actions: List[str]
    is_acceptable: bool
    calculation_method: str
    provenance_hash: str


# =============================================================================
# SPRAY FLOW CALCULATOR
# =============================================================================

class SprayFlowCalculator:
    """
    Calculator for spray water flow rate in desuperheaters.

    Implements the fundamental enthalpy balance equation for determining
    spray water requirements to achieve target steam temperature.

    Formula:
        m_spray = m_steam * (h_in - h_out) / (h_out - h_water)

    where:
        m_spray = Spray water mass flow rate (lb/hr)
        m_steam = Inlet steam mass flow rate (lb/hr)
        h_in = Inlet steam enthalpy (BTU/lb)
        h_out = Target outlet steam enthalpy (BTU/lb)
        h_water = Spray water enthalpy (BTU/lb)

    Example:
        >>> calc = SprayFlowCalculator()
        >>> result = calc.calculate_spray_requirements(
        ...     steam_flow_lb_hr=100000,
        ...     inlet_temp_f=950.0,
        ...     target_temp_f=850.0,
        ...     spray_water_temp_f=250.0,
        ...     pressure_psig=600.0,
        ... )
        >>> print(f"Spray flow required: {result.spray_flow_lb_hr:.0f} lb/hr")
    """

    def __init__(self) -> None:
        """Initialize spray flow calculator."""
        self.thermo_calc = SteamThermodynamicsCalculator()
        self.enthalpy_calc = EnthalpyBalanceCalculator()
        logger.debug("SprayFlowCalculator initialized")

    def calculate_spray_requirements(
        self,
        steam_flow_lb_hr: float,
        inlet_temp_f: float,
        target_temp_f: float,
        spray_water_temp_f: float,
        pressure_psig: float,
    ) -> SprayRequirementsResult:
        """
        Calculate spray water flow requirements - DETERMINISTIC.

        This is the primary calculation for spray desuperheater control.

        Args:
            steam_flow_lb_hr: Inlet steam mass flow rate (lb/hr)
            inlet_temp_f: Steam inlet temperature (F)
            target_temp_f: Target outlet temperature (F)
            spray_water_temp_f: Spray water temperature (F)
            pressure_psig: System pressure (psig)

        Returns:
            SprayRequirementsResult with complete analysis
        """
        warnings = []

        # Get saturation temperature
        sat_props = self.thermo_calc.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]

        # Calculate enthalpies
        h_in = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, inlet_temp_f)
        h_out = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, target_temp_f)
        h_water = self.thermo_calc.calculate_water_enthalpy(spray_water_temp_f)

        # Validate conditions
        if target_temp_f >= inlet_temp_f:
            warnings.append("Target temperature >= inlet temperature - no spray needed")
            spray_flow = 0.0
        elif h_out <= h_water:
            warnings.append("Invalid thermodynamic conditions")
            spray_flow = 0.0
        else:
            # Calculate spray flow using enthalpy balance
            spray_flow = steam_flow_lb_hr * (h_in - h_out) / (h_out - h_water)

        # Calculate spray ratio
        spray_ratio = (spray_flow / steam_flow_lb_hr * 100) if steam_flow_lb_hr > 0 else 0.0

        # Check spray ratio limits
        if spray_ratio > SprayDesuperheaterConstants.MAX_SPRAY_RATIO * 100:
            warnings.append(
                f"Spray ratio {spray_ratio:.1f}% exceeds maximum recommended "
                f"{SprayDesuperheaterConstants.MAX_SPRAY_RATIO * 100:.0f}%"
            )

        # Calculate heat removed
        heat_removed = spray_flow * (h_out - h_water) if spray_flow > 0 else 0.0

        # Check superheat remaining
        superheat_remaining = target_temp_f - t_sat
        if superheat_remaining < SprayDesuperheaterConstants.MIN_SUPERHEAT_AFTER_SPRAY_F:
            warnings.append(
                f"Superheat after spray ({superheat_remaining:.1f}F) is below "
                f"minimum recommended ({SprayDesuperheaterConstants.MIN_SUPERHEAT_AFTER_SPRAY_F}F)"
            )

        # Determine if within limits
        is_within_limits = (
            spray_ratio <= SprayDesuperheaterConstants.MAX_SPRAY_RATIO * 100 and
            superheat_remaining >= SprayDesuperheaterConstants.MIN_SUPERHEAT_AFTER_SPRAY_F and
            len(warnings) == 0
        )

        # Provenance hash
        provenance_data = {
            "steam_flow_lb_hr": steam_flow_lb_hr,
            "inlet_temp_f": inlet_temp_f,
            "target_temp_f": target_temp_f,
            "spray_water_temp_f": spray_water_temp_f,
            "pressure_psig": pressure_psig,
            "h_in": h_in,
            "h_out": h_out,
            "h_water": h_water,
            "spray_flow_lb_hr": spray_flow,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return SprayRequirementsResult(
            spray_flow_lb_hr=round(spray_flow, 0),
            spray_ratio_pct=round(spray_ratio, 2),
            inlet_enthalpy_btu_lb=round(h_in, 1),
            outlet_enthalpy_btu_lb=round(h_out, 1),
            water_enthalpy_btu_lb=round(h_water, 1),
            heat_removed_btu_hr=round(heat_removed, 0),
            outlet_temperature_f=target_temp_f,
            superheat_remaining_f=round(superheat_remaining, 1),
            is_within_limits=is_within_limits,
            warnings=warnings,
            calculation_method="enthalpy_balance",
            provenance_hash=provenance_hash,
        )

    def calculate_maximum_spray_flow(
        self,
        steam_flow_lb_hr: float,
        inlet_temp_f: float,
        spray_water_temp_f: float,
        pressure_psig: float,
        min_superheat_f: float = 25.0,
    ) -> float:
        """
        Calculate maximum allowable spray flow - DETERMINISTIC.

        The maximum spray is limited by the requirement to maintain
        minimum superheat after desuperheating.

        Args:
            steam_flow_lb_hr: Inlet steam mass flow rate (lb/hr)
            inlet_temp_f: Steam inlet temperature (F)
            spray_water_temp_f: Spray water temperature (F)
            pressure_psig: System pressure (psig)
            min_superheat_f: Minimum required superheat (F)

        Returns:
            Maximum spray flow (lb/hr)
        """
        # Get saturation temperature
        sat_props = self.thermo_calc.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]

        # Minimum outlet temperature
        min_outlet_temp = t_sat + min_superheat_f

        # Calculate spray flow for minimum temperature
        if inlet_temp_f <= min_outlet_temp:
            return 0.0

        result = self.calculate_spray_requirements(
            steam_flow_lb_hr=steam_flow_lb_hr,
            inlet_temp_f=inlet_temp_f,
            target_temp_f=min_outlet_temp,
            spray_water_temp_f=spray_water_temp_f,
            pressure_psig=pressure_psig,
        )

        return result.spray_flow_lb_hr

    def calculate_valve_position(
        self,
        spray_flow_lb_hr: float,
        valve_cv: float,
        pressure_drop_psi: float,
    ) -> float:
        """
        Calculate spray valve position for required flow - DETERMINISTIC.

        Uses standard valve flow equation:
            Q = Cv * sqrt(delta_P / SG)

        For water, SG = 1.0

        Args:
            spray_flow_lb_hr: Required spray flow (lb/hr)
            valve_cv: Valve flow coefficient (Cv)
            pressure_drop_psi: Available pressure drop across valve (psi)

        Returns:
            Valve position (0-100%)
        """
        if valve_cv <= 0 or pressure_drop_psi <= 0:
            return 0.0

        # Convert flow to gpm (water at ~250F has SG ~ 0.94)
        water_sg = 0.94
        spray_flow_gpm = spray_flow_lb_hr / (60 * 8.33 * water_sg)

        # Calculate required Cv
        cv_required = spray_flow_gpm / math.sqrt(pressure_drop_psi / water_sg)

        # Calculate valve position (assuming linear characteristic)
        valve_position = (cv_required / valve_cv) * 100

        return round(max(0.0, min(100.0, valve_position)), 1)


# =============================================================================
# MIXING THERMODYNAMICS CALCULATOR
# =============================================================================

class MixingThermodynamicsCalculator:
    """
    Calculator for steam-water mixing thermodynamics.

    Models the mixing process when spray water is injected into
    superheated steam, accounting for:
    - Energy balance
    - Mass balance
    - Mixing efficiency

    Example:
        >>> calc = MixingThermodynamicsCalculator()
        >>> result = calc.calculate_mixing(
        ...     steam_flow_lb_hr=100000,
        ...     steam_temp_f=950.0,
        ...     spray_flow_lb_hr=5000,
        ...     spray_temp_f=250.0,
        ...     pressure_psig=600.0,
        ...     mixing_efficiency=0.95,
        ... )
    """

    def __init__(self) -> None:
        """Initialize mixing thermodynamics calculator."""
        self.thermo_calc = SteamThermodynamicsCalculator()
        logger.debug("MixingThermodynamicsCalculator initialized")

    def calculate_mixing(
        self,
        steam_flow_lb_hr: float,
        steam_temp_f: float,
        spray_flow_lb_hr: float,
        spray_temp_f: float,
        pressure_psig: float,
        mixing_efficiency: float = 0.95,
    ) -> MixingThermodynamicsResult:
        """
        Calculate mixed steam conditions - DETERMINISTIC.

        Energy balance:
            m_steam * h_steam + m_spray * h_spray = m_total * h_mixed

        With mixing efficiency:
            h_mixed_actual = h_steam - eta * (h_steam - h_mixed_ideal)

        Args:
            steam_flow_lb_hr: Inlet steam mass flow rate (lb/hr)
            steam_temp_f: Inlet steam temperature (F)
            spray_flow_lb_hr: Spray water flow rate (lb/hr)
            spray_temp_f: Spray water temperature (F)
            pressure_psig: System pressure (psig)
            mixing_efficiency: Mixing efficiency factor (0-1)

        Returns:
            MixingThermodynamicsResult with mixed conditions
        """
        # Calculate enthalpies
        h_steam = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, steam_temp_f)
        h_spray = self.thermo_calc.calculate_water_enthalpy(spray_temp_f)

        # Mass balance
        total_flow = steam_flow_lb_hr + spray_flow_lb_hr

        if total_flow <= 0:
            # No flow - return inlet conditions
            return self._create_no_flow_result(steam_temp_f, h_steam, pressure_psig)

        # Energy balance for ideal mixing
        h_mixed_ideal = (steam_flow_lb_hr * h_steam + spray_flow_lb_hr * h_spray) / total_flow

        # Apply mixing efficiency
        # Actual enthalpy is between inlet steam and ideal mixed
        h_mixed_actual = h_steam - mixing_efficiency * (h_steam - h_mixed_ideal)

        # Convert to temperature
        mixed_temp = self.thermo_calc.calculate_temperature_from_enthalpy(
            pressure_psig, h_mixed_actual
        )

        # Calculate outlet superheat
        sat_props = self.thermo_calc.get_saturation_properties(pressure_psig)
        outlet_superheat = mixed_temp - sat_props["saturation_temp_f"]

        # Temperature drop
        temp_drop = steam_temp_f - mixed_temp

        # Heat absorbed by spray
        heat_absorbed = spray_flow_lb_hr * (h_mixed_actual - h_spray)

        # Provenance hash
        provenance_data = {
            "steam_flow_lb_hr": steam_flow_lb_hr,
            "steam_temp_f": steam_temp_f,
            "spray_flow_lb_hr": spray_flow_lb_hr,
            "spray_temp_f": spray_temp_f,
            "pressure_psig": pressure_psig,
            "mixing_efficiency": mixing_efficiency,
            "h_steam": h_steam,
            "h_spray": h_spray,
            "h_mixed_actual": h_mixed_actual,
            "mixed_temp": mixed_temp,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return MixingThermodynamicsResult(
            mixed_temperature_f=round(mixed_temp, 1),
            mixed_enthalpy_btu_lb=round(h_mixed_actual, 1),
            mixed_flow_lb_hr=round(total_flow, 0),
            mixing_efficiency=mixing_efficiency,
            temperature_drop_f=round(temp_drop, 1),
            heat_absorbed_by_spray_btu_hr=round(heat_absorbed, 0),
            outlet_superheat_f=round(outlet_superheat, 1),
            calculation_method="energy_balance",
            provenance_hash=provenance_hash,
        )

    def _create_no_flow_result(
        self,
        temp_f: float,
        enthalpy: float,
        pressure_psig: float,
    ) -> MixingThermodynamicsResult:
        """Create result for no-flow condition."""
        return MixingThermodynamicsResult(
            mixed_temperature_f=temp_f,
            mixed_enthalpy_btu_lb=enthalpy,
            mixed_flow_lb_hr=0.0,
            mixing_efficiency=1.0,
            temperature_drop_f=0.0,
            heat_absorbed_by_spray_btu_hr=0.0,
            outlet_superheat_f=0.0,
            calculation_method="no_flow",
            provenance_hash=hashlib.sha256(b"no_flow").hexdigest(),
        )


# =============================================================================
# DROPLET EVAPORATION CALCULATOR
# =============================================================================

class DropletEvaporationCalculator:
    """
    Calculator for spray droplet evaporation in superheated steam.

    Models the evaporation process of spray water droplets including:
    - Droplet size distribution
    - Heat transfer to droplets
    - Evaporation time and distance
    - Complete evaporation verification

    Example:
        >>> calc = DropletEvaporationCalculator()
        >>> result = calc.calculate_evaporation(
        ...     droplet_diameter_micron=150,
        ...     steam_temp_f=950.0,
        ...     spray_temp_f=250.0,
        ...     steam_velocity_fps=100.0,
        ...     pressure_psig=600.0,
        ... )
    """

    def __init__(self) -> None:
        """Initialize droplet evaporation calculator."""
        self.thermo_calc = SteamThermodynamicsCalculator()
        logger.debug("DropletEvaporationCalculator initialized")

    def calculate_evaporation(
        self,
        droplet_diameter_micron: float,
        steam_temp_f: float,
        spray_temp_f: float,
        steam_velocity_fps: float,
        pressure_psig: float,
        pipe_diameter_in: float = 12.0,
    ) -> DropletEvaporationResult:
        """
        Calculate droplet evaporation characteristics - DETERMINISTIC.

        Uses simplified model based on heat transfer correlations.

        The evaporation time is estimated using:
            t_evap = (rho_w * d^2 * h_fg) / (12 * k_s * delta_T)

        where:
            rho_w = water density
            d = droplet diameter
            h_fg = latent heat of vaporization
            k_s = steam thermal conductivity
            delta_T = temperature difference

        Args:
            droplet_diameter_micron: Droplet diameter (microns)
            steam_temp_f: Steam temperature (F)
            spray_temp_f: Spray water temperature (F)
            steam_velocity_fps: Steam velocity (ft/s)
            pressure_psig: System pressure (psig)
            pipe_diameter_in: Pipe inner diameter (inches)

        Returns:
            DropletEvaporationResult with evaporation analysis
        """
        # Get saturation properties
        sat_props = self.thermo_calc.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]
        h_fg = sat_props["h_fg_btu_lb"]

        # Convert droplet diameter to feet
        d_ft = droplet_diameter_micron * 3.281e-6

        # Temperature difference (driving force for evaporation)
        delta_t = steam_temp_f - t_sat

        if delta_t <= 0:
            # No superheat - cannot evaporate
            return self._create_no_evaporation_result(
                droplet_diameter_micron, steam_velocity_fps
            )

        # Steam thermal conductivity (approximate, BTU/hr-ft-F)
        k_steam = 0.02  # At typical superheater conditions

        # Water density (lb/ft3)
        rho_water = DropletEvaporationConstants.WATER_DENSITY_LB_FT3

        # Calculate evaporation time using D-squared law
        # t = (rho * d^2 * h_fg) / (12 * k * delta_T)
        evap_time_hr = (rho_water * d_ft**2 * h_fg) / (12 * k_steam * delta_t)

        # Convert to milliseconds
        evap_time_ms = evap_time_hr * 3600 * 1000

        # Apply efficiency factor
        evap_time_ms *= DropletEvaporationConstants.EVAPORATION_TIME_FACTOR

        # Calculate evaporation distance
        evap_distance_ft = steam_velocity_fps * (evap_time_ms / 1000)

        # Determine if complete evaporation occurs within pipe
        available_length_ft = pipe_diameter_in * SprayDesuperheaterConstants.EVAPORATION_LENGTH_PER_INCH_DIA / 12
        complete_evaporation = evap_distance_ft <= available_length_ft

        # Calculate evaporation efficiency
        if complete_evaporation:
            evap_efficiency = 100.0
            final_diameter = 0.0
        else:
            # Partial evaporation
            evap_fraction = min(1.0, available_length_ft / evap_distance_ft)
            evap_efficiency = evap_fraction * 100
            # Diameter decreases with sqrt of remaining mass
            final_diameter = droplet_diameter_micron * math.sqrt(1 - evap_fraction)

        # Provenance hash
        provenance_data = {
            "droplet_diameter_micron": droplet_diameter_micron,
            "steam_temp_f": steam_temp_f,
            "spray_temp_f": spray_temp_f,
            "steam_velocity_fps": steam_velocity_fps,
            "pressure_psig": pressure_psig,
            "evap_time_ms": evap_time_ms,
            "evap_distance_ft": evap_distance_ft,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return DropletEvaporationResult(
            evaporation_time_ms=round(evap_time_ms, 1),
            evaporation_distance_ft=round(evap_distance_ft, 2),
            initial_droplet_diameter_micron=droplet_diameter_micron,
            final_droplet_diameter_micron=round(final_diameter, 1),
            complete_evaporation=complete_evaporation,
            evaporation_efficiency_pct=round(evap_efficiency, 1),
            steam_velocity_fps=steam_velocity_fps,
            calculation_method="d_squared_law",
            provenance_hash=provenance_hash,
        )

    def _create_no_evaporation_result(
        self,
        droplet_diameter: float,
        velocity: float,
    ) -> DropletEvaporationResult:
        """Create result for no evaporation condition."""
        return DropletEvaporationResult(
            evaporation_time_ms=float('inf'),
            evaporation_distance_ft=float('inf'),
            initial_droplet_diameter_micron=droplet_diameter,
            final_droplet_diameter_micron=droplet_diameter,
            complete_evaporation=False,
            evaporation_efficiency_pct=0.0,
            steam_velocity_fps=velocity,
            calculation_method="no_superheat",
            provenance_hash=hashlib.sha256(b"no_evap").hexdigest(),
        )

    def calculate_optimal_droplet_size(
        self,
        steam_temp_f: float,
        steam_velocity_fps: float,
        pressure_psig: float,
        available_length_ft: float,
    ) -> float:
        """
        Calculate optimal droplet size for complete evaporation - DETERMINISTIC.

        Args:
            steam_temp_f: Steam temperature (F)
            steam_velocity_fps: Steam velocity (ft/s)
            pressure_psig: System pressure (psig)
            available_length_ft: Available evaporation length (ft)

        Returns:
            Optimal droplet diameter (microns)
        """
        # Binary search for optimal diameter
        low = SprayDesuperheaterConstants.DROPLET_DIAMETER_MICRON_MIN
        high = SprayDesuperheaterConstants.DROPLET_DIAMETER_MICRON_MAX

        for _ in range(20):  # Max iterations
            mid = (low + high) / 2

            result = self.calculate_evaporation(
                droplet_diameter_micron=mid,
                steam_temp_f=steam_temp_f,
                spray_temp_f=250.0,  # Typical spray temperature
                steam_velocity_fps=steam_velocity_fps,
                pressure_psig=pressure_psig,
                pipe_diameter_in=available_length_ft * 2,  # Rough estimate
            )

            if result.evaporation_distance_ft <= available_length_ft:
                low = mid  # Can use larger droplets
            else:
                high = mid  # Need smaller droplets

        return round(low, 0)


# =============================================================================
# SPRAY EFFICIENCY CALCULATOR
# =============================================================================

class SprayEfficiencyCalculator:
    """
    Calculator for overall spray desuperheater efficiency.

    Evaluates multiple efficiency components:
    - Atomization efficiency (nozzle performance)
    - Mixing efficiency (steam-spray contact)
    - Evaporation efficiency (complete vaporization)
    - Thermal efficiency (temperature control accuracy)

    Example:
        >>> calc = SprayEfficiencyCalculator()
        >>> result = calc.calculate_efficiency(
        ...     steam_flow_lb_hr=100000,
        ...     spray_flow_lb_hr=5000,
        ...     inlet_temp_f=950.0,
        ...     target_temp_f=850.0,
        ...     actual_outlet_temp_f=855.0,
        ...     pressure_psig=600.0,
        ...     nozzle_pressure_drop_psi=50.0,
        ... )
    """

    def __init__(self) -> None:
        """Initialize spray efficiency calculator."""
        self.thermo_calc = SteamThermodynamicsCalculator()
        self.droplet_calc = DropletEvaporationCalculator()
        logger.debug("SprayEfficiencyCalculator initialized")

    def calculate_efficiency(
        self,
        steam_flow_lb_hr: float,
        spray_flow_lb_hr: float,
        inlet_temp_f: float,
        target_temp_f: float,
        actual_outlet_temp_f: float,
        pressure_psig: float,
        nozzle_pressure_drop_psi: float,
        pipe_diameter_in: float = 12.0,
    ) -> SprayEfficiencyResult:
        """
        Calculate comprehensive spray efficiency - DETERMINISTIC.

        Args:
            steam_flow_lb_hr: Steam mass flow rate (lb/hr)
            spray_flow_lb_hr: Spray water flow rate (lb/hr)
            inlet_temp_f: Steam inlet temperature (F)
            target_temp_f: Target outlet temperature (F)
            actual_outlet_temp_f: Actual measured outlet temperature (F)
            pressure_psig: System pressure (psig)
            nozzle_pressure_drop_psi: Pressure drop across spray nozzle (psi)
            pipe_diameter_in: Pipe inner diameter (inches)

        Returns:
            SprayEfficiencyResult with complete efficiency analysis
        """
        recommendations = []

        # 1. Atomization efficiency (based on nozzle pressure drop)
        atomization_efficiency = self._calculate_atomization_efficiency(
            nozzle_pressure_drop_psi
        )
        if atomization_efficiency < 90:
            recommendations.append("Increase spray water pressure for better atomization")

        # 2. Evaporation efficiency
        steam_velocity = self._estimate_steam_velocity(
            steam_flow_lb_hr, pressure_psig, inlet_temp_f, pipe_diameter_in
        )

        droplet_size = self._estimate_droplet_size(nozzle_pressure_drop_psi)

        evap_result = self.droplet_calc.calculate_evaporation(
            droplet_diameter_micron=droplet_size,
            steam_temp_f=inlet_temp_f,
            spray_temp_f=250.0,
            steam_velocity_fps=steam_velocity,
            pressure_psig=pressure_psig,
            pipe_diameter_in=pipe_diameter_in,
        )
        evaporation_efficiency = evap_result.evaporation_efficiency_pct

        if evaporation_efficiency < 95:
            recommendations.append(
                "Incomplete droplet evaporation - increase straight run or reduce droplet size"
            )

        # 3. Thermal efficiency (temperature control accuracy)
        thermal_efficiency = self._calculate_thermal_efficiency(
            target_temp_f, actual_outlet_temp_f, inlet_temp_f
        )
        if thermal_efficiency < 95:
            recommendations.append("Temperature control deviation detected - check control tuning")

        # 4. Mixing efficiency (derived from thermal efficiency)
        mixing_efficiency = min(100.0, thermal_efficiency * 1.02)  # Slightly higher than thermal

        # Overall efficiency
        overall_efficiency = (
            atomization_efficiency * 0.25 +
            mixing_efficiency * 0.25 +
            evaporation_efficiency * 0.25 +
            thermal_efficiency * 0.25
        )

        # Determine spray pattern quality
        spray_pattern_quality = self._determine_spray_quality(overall_efficiency)

        # Provenance hash
        provenance_data = {
            "steam_flow_lb_hr": steam_flow_lb_hr,
            "spray_flow_lb_hr": spray_flow_lb_hr,
            "inlet_temp_f": inlet_temp_f,
            "target_temp_f": target_temp_f,
            "actual_outlet_temp_f": actual_outlet_temp_f,
            "atomization_efficiency": atomization_efficiency,
            "mixing_efficiency": mixing_efficiency,
            "evaporation_efficiency": evaporation_efficiency,
            "thermal_efficiency": thermal_efficiency,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return SprayEfficiencyResult(
            overall_efficiency_pct=round(overall_efficiency, 1),
            atomization_efficiency_pct=round(atomization_efficiency, 1),
            mixing_efficiency_pct=round(mixing_efficiency, 1),
            evaporation_efficiency_pct=round(evaporation_efficiency, 1),
            thermal_efficiency_pct=round(thermal_efficiency, 1),
            spray_pattern_quality=spray_pattern_quality,
            recommendations=recommendations,
            calculation_method="multi_factor",
            provenance_hash=provenance_hash,
        )

    def _calculate_atomization_efficiency(
        self,
        pressure_drop_psi: float,
    ) -> float:
        """Calculate atomization efficiency from nozzle pressure drop."""
        min_dp = SprayDesuperheaterConstants.NOZZLE_PRESSURE_DROP_MIN_PSI
        optimal_dp = 75.0  # Optimal pressure drop

        if pressure_drop_psi < min_dp:
            # Poor atomization
            return 50.0 + 40.0 * (pressure_drop_psi / min_dp)
        elif pressure_drop_psi <= optimal_dp:
            # Improving atomization
            return 90.0 + 10.0 * ((pressure_drop_psi - min_dp) / (optimal_dp - min_dp))
        else:
            # Good atomization (slight decrease at very high DP due to wear concerns)
            return 100.0 - 0.1 * (pressure_drop_psi - optimal_dp)

    def _calculate_thermal_efficiency(
        self,
        target_temp_f: float,
        actual_temp_f: float,
        inlet_temp_f: float,
    ) -> float:
        """Calculate thermal efficiency from temperature control accuracy."""
        expected_drop = inlet_temp_f - target_temp_f
        actual_drop = inlet_temp_f - actual_temp_f

        if expected_drop <= 0:
            return 100.0

        # Calculate accuracy
        temp_error = abs(actual_temp_f - target_temp_f)
        accuracy = 1.0 - (temp_error / expected_drop)

        return max(0.0, min(100.0, accuracy * 100))

    def _estimate_steam_velocity(
        self,
        flow_lb_hr: float,
        pressure_psig: float,
        temp_f: float,
        pipe_diameter_in: float,
    ) -> float:
        """Estimate steam velocity in pipe."""
        # Get specific volume
        v_specific = self.thermo_calc.calculate_specific_volume(pressure_psig, temp_f)

        # Volumetric flow rate (ft3/hr)
        vol_flow = flow_lb_hr * v_specific

        # Pipe area (ft2)
        pipe_area = math.pi * (pipe_diameter_in / 12) ** 2 / 4

        # Velocity (ft/hr -> ft/s)
        velocity = vol_flow / pipe_area / 3600

        return velocity

    def _estimate_droplet_size(self, pressure_drop_psi: float) -> float:
        """Estimate droplet size from nozzle pressure drop."""
        # Higher pressure drop = smaller droplets
        # Empirical correlation
        if pressure_drop_psi < 20:
            return 400
        elif pressure_drop_psi < 50:
            return 300 - 4 * (pressure_drop_psi - 20)
        elif pressure_drop_psi < 100:
            return 180 - 1.6 * (pressure_drop_psi - 50)
        else:
            return max(50, 100 - 0.5 * (pressure_drop_psi - 100))

    def _determine_spray_quality(self, efficiency: float) -> str:
        """Determine spray pattern quality from efficiency."""
        if efficiency >= 95:
            return "excellent"
        elif efficiency >= 85:
            return "good"
        elif efficiency >= 70:
            return "fair"
        else:
            return "poor"


# =============================================================================
# WATER QUALITY IMPACT ANALYZER
# =============================================================================

class WaterQualityImpactAnalyzer:
    """
    Analyzer for spray water quality impact on steam purity.

    Evaluates the impact of spray water contaminants on:
    - Steam purity degradation
    - Turbine blade deposits
    - Superheater tube fouling

    Based on EPRI guidelines for cycle chemistry.

    Example:
        >>> analyzer = WaterQualityImpactAnalyzer()
        >>> result = analyzer.analyze_impact(
        ...     spray_flow_lb_hr=5000,
        ...     steam_flow_lb_hr=100000,
        ...     water_tds_ppm=2.0,
        ...     water_silica_ppm=0.01,
        ...     water_iron_ppb=3.0,
        ... )
    """

    def __init__(self) -> None:
        """Initialize water quality impact analyzer."""
        logger.debug("WaterQualityImpactAnalyzer initialized")

    def analyze_impact(
        self,
        spray_flow_lb_hr: float,
        steam_flow_lb_hr: float,
        water_tds_ppm: float,
        water_silica_ppm: float = 0.0,
        water_iron_ppb: float = 0.0,
        water_copper_ppb: float = 0.0,
        water_conductivity_us_cm: float = 0.0,
    ) -> WaterQualityImpactResult:
        """
        Analyze water quality impact on steam purity - DETERMINISTIC.

        Args:
            spray_flow_lb_hr: Spray water flow rate (lb/hr)
            steam_flow_lb_hr: Steam flow rate (lb/hr)
            water_tds_ppm: Total dissolved solids in spray water (ppm)
            water_silica_ppm: Silica in spray water (ppm)
            water_iron_ppb: Iron in spray water (ppb)
            water_copper_ppb: Copper in spray water (ppb)
            water_conductivity_us_cm: Cation conductivity (uS/cm)

        Returns:
            WaterQualityImpactResult with impact analysis
        """
        recommendations = []

        # Calculate dilution ratio
        total_flow = steam_flow_lb_hr + spray_flow_lb_hr
        spray_fraction = spray_flow_lb_hr / total_flow if total_flow > 0 else 0

        # TDS contribution to steam
        tds_in_steam = water_tds_ppm * spray_fraction

        # Silica contribution to steam
        silica_in_steam = water_silica_ppm * spray_fraction

        # Check against limits
        tds_limit = SprayDesuperheaterConstants.MAX_TDS_PPM
        silica_limit = SprayDesuperheaterConstants.MAX_SILICA_PPM

        # Evaluate risk
        risk_factors = []

        if water_tds_ppm > tds_limit:
            risk_factors.append(f"TDS ({water_tds_ppm:.1f} ppm) exceeds limit ({tds_limit} ppm)")
            recommendations.append("Improve spray water treatment to reduce TDS")

        if water_silica_ppm > silica_limit:
            risk_factors.append(f"Silica ({water_silica_ppm:.3f} ppm) exceeds limit ({silica_limit} ppm)")
            recommendations.append("Install silica removal for spray water")

        if water_iron_ppb > SprayDesuperheaterConstants.MAX_IRON_PPB:
            risk_factors.append(f"Iron exceeds limit")
            recommendations.append("Check condensate polisher performance")

        if water_copper_ppb > SprayDesuperheaterConstants.MAX_COPPER_PPB:
            risk_factors.append(f"Copper exceeds limit")
            recommendations.append("Investigate copper source in feedwater system")

        # Determine fouling risk level
        if len(risk_factors) == 0:
            fouling_risk = "low"
        elif len(risk_factors) == 1:
            fouling_risk = "medium"
        elif len(risk_factors) == 2:
            fouling_risk = "high"
        else:
            fouling_risk = "critical"

        # Estimate deposition rate (simplified model)
        # Based on TDS and flow rates
        deposition_rate = (tds_in_steam * total_flow * 0.001) / 1000  # mg/ft2/hr

        # Scale formation potential (0-1 scale)
        scale_potential = min(1.0, (water_tds_ppm / tds_limit) * 0.5 +
                              (water_silica_ppm / silica_limit) * 0.5) if silica_limit > 0 else 0

        is_acceptable = fouling_risk in ["low", "medium"]

        # Provenance hash
        provenance_data = {
            "spray_flow_lb_hr": spray_flow_lb_hr,
            "steam_flow_lb_hr": steam_flow_lb_hr,
            "water_tds_ppm": water_tds_ppm,
            "water_silica_ppm": water_silica_ppm,
            "tds_in_steam": tds_in_steam,
            "silica_in_steam": silica_in_steam,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return WaterQualityImpactResult(
            tds_in_steam_ppm=round(tds_in_steam, 4),
            silica_in_steam_ppm=round(silica_in_steam, 6),
            estimated_deposition_rate_mg_ft2_hr=round(deposition_rate, 4),
            fouling_risk_level=fouling_risk,
            scale_formation_potential=round(scale_potential, 3),
            recommended_actions=recommendations,
            is_acceptable=is_acceptable,
            calculation_method="epri_guidelines",
            provenance_hash=provenance_hash,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_spray_flow_calculator() -> SprayFlowCalculator:
    """Factory function to create SprayFlowCalculator."""
    return SprayFlowCalculator()


def create_mixing_thermodynamics_calculator() -> MixingThermodynamicsCalculator:
    """Factory function to create MixingThermodynamicsCalculator."""
    return MixingThermodynamicsCalculator()


def create_droplet_evaporation_calculator() -> DropletEvaporationCalculator:
    """Factory function to create DropletEvaporationCalculator."""
    return DropletEvaporationCalculator()


def create_spray_efficiency_calculator() -> SprayEfficiencyCalculator:
    """Factory function to create SprayEfficiencyCalculator."""
    return SprayEfficiencyCalculator()


def create_water_quality_impact_analyzer() -> WaterQualityImpactAnalyzer:
    """Factory function to create WaterQualityImpactAnalyzer."""
    return WaterQualityImpactAnalyzer()
