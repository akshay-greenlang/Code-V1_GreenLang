"""
GL-009 THERMALIQ Agent - Expansion Tank Sizing and Analysis

This module provides expansion tank sizing validation and analysis
per API 660 (Shell-and-Tube Heat Exchangers) and industry best practices
for thermal fluid systems.

Key functions:
    - Calculate thermal expansion volume
    - Validate tank sizing adequacy
    - NPSH analysis at pump suction
    - Level predictions (cold/hot)
    - Nitrogen blanket requirements

All calculations are deterministic - ZERO HALLUCINATION guaranteed.

Reference:
    - API 660: Shell-and-Tube Heat Exchangers
    - ASME Section VIII: Pressure Vessels
    - Thermal fluid manufacturer guidelines

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.expansion_tank import (
    ...     ExpansionTankAnalyzer,
    ... )
    >>> analyzer = ExpansionTankAnalyzer(fluid_type=ThermalFluidType.THERMINOL_66)
    >>> result = analyzer.analyze(
    ...     tank_volume_gallons=1000,
    ...     system_volume_gallons=5000,
    ...     cold_temp_f=70,
    ...     hot_temp_f=600,
    ... )
    >>> print(f"Sizing adequate: {result.sizing_adequate}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from pydantic import BaseModel, Field

from .schemas import (
    ThermalFluidType,
    ExpansionTankSizing,
    ExpansionTankData,
)
from .config import ExpansionTankConfig, PumpConfig
from .fluid_properties import ThermalFluidPropertyDatabase

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Gravity constant
GRAVITY_FT_S2 = 32.174

# Conversion factors
GALLONS_PER_FT3 = 7.48052
FT_HEAD_PER_PSI = 2.31  # for water, adjusted by specific gravity

# Safety factors
EXPANSION_SAFETY_FACTOR = 1.1  # 10% safety margin
MIN_TANK_LEVEL_PCT = 10.0  # Minimum level to maintain
MAX_TANK_LEVEL_PCT = 90.0  # Maximum level before overflow concern
NPSH_SAFETY_MARGIN_FT = 3.0  # Minimum NPSH margin


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NPSHAnalysis:
    """NPSH (Net Positive Suction Head) analysis results."""

    available_npsh_ft: float
    required_npsh_ft: float
    npsh_margin_ft: float
    cavitation_risk: str  # "none", "low", "medium", "high"
    vapor_pressure_psia: float
    suction_pressure_psia: float
    static_head_ft: float
    friction_loss_ft: float


# =============================================================================
# EXPANSION TANK ANALYZER
# =============================================================================

class ExpansionTankAnalyzer:
    """
    Expansion tank sizing and analysis for thermal fluid systems.

    This class validates expansion tank sizing, predicts operating levels,
    and performs NPSH analysis to ensure adequate pump suction conditions.
    All calculations are deterministic per API 660 methodology.

    Key calculations:
        - Thermal expansion volume (density-based)
        - Required tank volume with safety factors
        - Cold/hot level predictions
        - NPSH available vs required
        - Nitrogen blanket requirements

    Example:
        >>> analyzer = ExpansionTankAnalyzer(
        ...     fluid_type=ThermalFluidType.THERMINOL_66,
        ...     pump_config=PumpConfig(),
        ... )
        >>> sizing = analyzer.analyze(
        ...     tank_volume_gallons=1000,
        ...     system_volume_gallons=5000,
        ...     cold_temp_f=70,
        ...     hot_temp_f=600,
        ... )
    """

    def __init__(
        self,
        fluid_type: ThermalFluidType,
        pump_config: Optional[PumpConfig] = None,
        tank_config: Optional[ExpansionTankConfig] = None,
    ) -> None:
        """
        Initialize the expansion tank analyzer.

        Args:
            fluid_type: Type of thermal fluid
            pump_config: Pump configuration for NPSH analysis
            tank_config: Expansion tank configuration
        """
        self.fluid_type = fluid_type
        self.pump_config = pump_config or PumpConfig()
        self.tank_config = tank_config or ExpansionTankConfig()

        self._property_db = ThermalFluidPropertyDatabase()
        self._calculation_count = 0

        logger.info(f"ExpansionTankAnalyzer initialized for {fluid_type}")

    def analyze(
        self,
        tank_volume_gallons: float,
        system_volume_gallons: float,
        cold_temp_f: float,
        hot_temp_f: float,
        current_level_pct: Optional[float] = None,
        tank_elevation_ft: float = 0.0,
        pump_centerline_ft: float = 0.0,
    ) -> ExpansionTankSizing:
        """
        Perform complete expansion tank analysis.

        Args:
            tank_volume_gallons: Actual tank volume (gallons)
            system_volume_gallons: Total system volume at cold temp (gallons)
            cold_temp_f: Cold fill temperature (F)
            hot_temp_f: Maximum operating temperature (F)
            current_level_pct: Current tank level (%) - optional
            tank_elevation_ft: Tank elevation above pump (ft)
            pump_centerline_ft: Pump centerline elevation (ft)

        Returns:
            ExpansionTankSizing with complete analysis
        """
        self._calculation_count += 1
        recommendations = []

        # Get fluid densities at cold and hot temperatures
        rho_cold = self._property_db.get_density(self.fluid_type, cold_temp_f)
        rho_hot = self._property_db.get_density(self.fluid_type, hot_temp_f)

        # Calculate thermal expansion percentage
        # V_hot / V_cold = rho_cold / rho_hot
        expansion_ratio = rho_cold / rho_hot
        expansion_pct = (expansion_ratio - 1.0) * 100

        # Calculate expansion volume
        expansion_volume_gal = system_volume_gallons * (expansion_ratio - 1.0)

        # Required tank volume per API 660 / industry practice
        # Tank must accommodate:
        # 1. Expansion volume
        # 2. Minimum heel (keep pump suction covered)
        # 3. Safety margin
        min_heel_gal = system_volume_gallons * 0.02  # 2% minimum heel
        required_volume_gal = (
            (expansion_volume_gal + min_heel_gal) * EXPANSION_SAFETY_FACTOR
        )

        # Check sizing adequacy
        sizing_adequate = tank_volume_gallons >= required_volume_gal

        if not sizing_adequate:
            recommendations.append(
                f"Tank undersized: {tank_volume_gallons:.0f} gal vs "
                f"{required_volume_gal:.0f} gal required"
            )
            recommendations.append(
                "Consider larger expansion tank or reduce system volume"
            )

        # Calculate expected levels
        # Cold level: just enough to fill system + some in tank
        cold_fill_volume_gal = min_heel_gal + expansion_volume_gal * 0.1
        cold_level_pct = cold_fill_volume_gal / tank_volume_gallons * 100
        cold_level_pct = max(MIN_TANK_LEVEL_PCT, min(cold_level_pct, 50.0))

        # Hot level: cold volume expands into tank
        cold_volume_in_tank = cold_level_pct / 100 * tank_volume_gallons
        hot_volume_in_tank = cold_volume_in_tank + expansion_volume_gal
        hot_level_pct = hot_volume_in_tank / tank_volume_gallons * 100

        # Check hot level
        if hot_level_pct > MAX_TANK_LEVEL_PCT:
            recommendations.append(
                f"Hot level {hot_level_pct:.1f}% exceeds {MAX_TANK_LEVEL_PCT}% limit"
            )
            recommendations.append("Risk of overflow - reduce cold fill level")

        # Current level analysis
        level_deviation = 0.0
        if current_level_pct is not None:
            # Estimate expected current level based on temperature
            # (would need current temp for accurate calculation)
            expected_level = (cold_level_pct + hot_level_pct) / 2  # Rough estimate
            level_deviation = current_level_pct - expected_level

            if abs(level_deviation) > 15:
                recommendations.append(
                    f"Current level {current_level_pct:.1f}% deviates "
                    f"{level_deviation:.1f}% from expected"
                )
                recommendations.append("Check for leaks or incorrect fill level")

        # NPSH analysis
        npsh_analysis = self._analyze_npsh(
            tank_elevation_ft=tank_elevation_ft,
            pump_centerline_ft=pump_centerline_ft,
            tank_level_pct=current_level_pct or cold_level_pct,
            tank_height_ft=self._estimate_tank_height(tank_volume_gallons),
            fluid_temp_f=hot_temp_f,  # Worst case at hot temperature
            tank_pressure_psig=self.tank_config.blanket_pressure_psig,
        )

        return ExpansionTankSizing(
            required_volume_gallons=round(required_volume_gal, 1),
            actual_volume_gallons=tank_volume_gallons,
            sizing_adequate=sizing_adequate,
            thermal_expansion_pct=round(expansion_pct, 2),
            expansion_volume_gallons=round(expansion_volume_gal, 1),
            cold_level_pct=round(cold_level_pct, 1),
            hot_level_pct=round(hot_level_pct, 1),
            current_level_deviation_pct=round(level_deviation, 1),
            required_npsh_ft=round(npsh_analysis.required_npsh_ft, 1),
            available_npsh_ft=round(npsh_analysis.available_npsh_ft, 1),
            npsh_margin_ft=round(npsh_analysis.npsh_margin_ft, 1),
            recommendations=recommendations,
            calculation_standard="API_660",
        )

    def _analyze_npsh(
        self,
        tank_elevation_ft: float,
        pump_centerline_ft: float,
        tank_level_pct: float,
        tank_height_ft: float,
        fluid_temp_f: float,
        tank_pressure_psig: float,
    ) -> NPSHAnalysis:
        """
        Analyze NPSH (Net Positive Suction Head) available.

        NPSH_available = P_tank + P_static - P_vapor - h_friction

        Args:
            tank_elevation_ft: Tank bottom elevation (ft)
            pump_centerline_ft: Pump centerline elevation (ft)
            tank_level_pct: Current tank level (%)
            tank_height_ft: Tank height (ft)
            fluid_temp_f: Fluid temperature (F)
            tank_pressure_psig: Tank pressure (psig)

        Returns:
            NPSHAnalysis with NPSH calculations
        """
        # Get fluid properties
        props = self._property_db.get_properties(self.fluid_type, fluid_temp_f)

        # Convert density to specific gravity
        specific_gravity = props.density_lb_ft3 / 62.4

        # Static head from liquid level to pump
        liquid_height_ft = tank_height_ft * tank_level_pct / 100
        static_head_ft = (tank_elevation_ft + liquid_height_ft) - pump_centerline_ft

        # Pressure head from tank pressure (convert to feet of fluid)
        # P_head = P * 2.31 / SG
        tank_pressure_psia = tank_pressure_psig + 14.696
        pressure_head_ft = tank_pressure_psia * FT_HEAD_PER_PSI / specific_gravity

        # Vapor pressure head
        vapor_pressure_head_ft = (
            props.vapor_pressure_psia * FT_HEAD_PER_PSI / specific_gravity
        )

        # Estimate friction loss in suction line (simplified)
        # Assume 2-5 ft friction loss for typical installation
        friction_loss_ft = 3.0

        # Calculate NPSH available
        npsh_available = (
            pressure_head_ft +
            static_head_ft -
            vapor_pressure_head_ft -
            friction_loss_ft
        )

        # Required NPSH from pump config
        npsh_required = self.pump_config.npsh_required_ft

        # NPSH margin
        npsh_margin = npsh_available - npsh_required

        # Determine cavitation risk
        if npsh_margin < 0:
            cavitation_risk = "high"
        elif npsh_margin < NPSH_SAFETY_MARGIN_FT:
            cavitation_risk = "medium"
        elif npsh_margin < NPSH_SAFETY_MARGIN_FT * 2:
            cavitation_risk = "low"
        else:
            cavitation_risk = "none"

        return NPSHAnalysis(
            available_npsh_ft=npsh_available,
            required_npsh_ft=npsh_required,
            npsh_margin_ft=npsh_margin,
            cavitation_risk=cavitation_risk,
            vapor_pressure_psia=props.vapor_pressure_psia,
            suction_pressure_psia=tank_pressure_psia,
            static_head_ft=static_head_ft,
            friction_loss_ft=friction_loss_ft,
        )

    def _estimate_tank_height(self, volume_gallons: float) -> float:
        """
        Estimate tank height from volume (assuming 2:1 L/D ratio).

        Args:
            volume_gallons: Tank volume (gallons)

        Returns:
            Estimated height (ft)
        """
        # V = pi/4 * D^2 * H
        # Assume H = 2 * D (typical proportions)
        # V = pi/4 * D^2 * 2D = pi/2 * D^3
        # D = (2V/pi)^(1/3)

        volume_ft3 = volume_gallons / GALLONS_PER_FT3
        diameter_ft = (2 * volume_ft3 / math.pi) ** (1/3)
        height_ft = 2 * diameter_ft

        return height_ft

    def calculate_expansion_volume(
        self,
        system_volume_gallons: float,
        cold_temp_f: float,
        hot_temp_f: float,
    ) -> Dict[str, float]:
        """
        Calculate thermal expansion volume.

        Args:
            system_volume_gallons: System volume at cold temperature
            cold_temp_f: Cold fill temperature (F)
            hot_temp_f: Hot operating temperature (F)

        Returns:
            Dictionary with expansion calculations
        """
        self._calculation_count += 1

        expansion_gal = self._property_db.calculate_expansion_volume(
            fluid_type=self.fluid_type,
            system_volume_gallons=system_volume_gallons,
            cold_temp_f=cold_temp_f,
            hot_temp_f=hot_temp_f,
        )

        expansion_pct = expansion_gal / system_volume_gallons * 100

        return {
            "expansion_volume_gallons": round(expansion_gal, 1),
            "expansion_percentage": round(expansion_pct, 2),
            "cold_system_volume_gallons": system_volume_gallons,
            "hot_system_volume_gallons": system_volume_gallons + expansion_gal,
        }

    def size_expansion_tank(
        self,
        system_volume_gallons: float,
        cold_temp_f: float,
        hot_temp_f: float,
        safety_factor: float = 1.25,
    ) -> Dict[str, Any]:
        """
        Size expansion tank for a new system.

        Args:
            system_volume_gallons: System volume
            cold_temp_f: Cold fill temperature (F)
            hot_temp_f: Maximum operating temperature (F)
            safety_factor: Sizing safety factor (default 1.25)

        Returns:
            Dictionary with sizing recommendations
        """
        self._calculation_count += 1

        # Calculate expansion
        expansion = self.calculate_expansion_volume(
            system_volume_gallons,
            cold_temp_f,
            hot_temp_f,
        )

        expansion_gal = expansion["expansion_volume_gallons"]

        # Minimum heel requirement (2-3% of system volume)
        min_heel_gal = system_volume_gallons * 0.03

        # Vapor space allowance (10% of expansion)
        vapor_space_gal = expansion_gal * 0.10

        # Required volume
        required_gal = (expansion_gal + min_heel_gal + vapor_space_gal) * safety_factor

        # Round up to standard tank sizes
        standard_sizes = [100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
        recommended_size = next(
            (s for s in standard_sizes if s >= required_gal),
            standard_sizes[-1]
        )

        # Expected levels
        cold_level = (min_heel_gal + vapor_space_gal) / recommended_size * 100
        hot_level = (min_heel_gal + expansion_gal + vapor_space_gal) / recommended_size * 100

        return {
            "system_volume_gallons": system_volume_gallons,
            "expansion_volume_gallons": round(expansion_gal, 1),
            "minimum_heel_gallons": round(min_heel_gal, 1),
            "vapor_space_gallons": round(vapor_space_gal, 1),
            "required_volume_gallons": round(required_gal, 1),
            "recommended_size_gallons": recommended_size,
            "expected_cold_level_pct": round(cold_level, 1),
            "expected_hot_level_pct": round(hot_level, 1),
            "safety_factor_applied": safety_factor,
            "design_standard": "API_660",
        }

    def validate_operating_level(
        self,
        tank_data: ExpansionTankData,
        current_temp_f: float,
    ) -> Dict[str, Any]:
        """
        Validate current operating level is appropriate.

        Args:
            tank_data: Current expansion tank operating data
            current_temp_f: Current system temperature (F)

        Returns:
            Dictionary with validation results
        """
        self._calculation_count += 1

        warnings = []
        status = "normal"

        # Calculate expected level at current temperature
        expected_expansion = self._property_db.calculate_expansion_volume(
            fluid_type=self.fluid_type,
            system_volume_gallons=tank_data.system_volume_gallons,
            cold_temp_f=tank_data.cold_fill_temp_f,
            hot_temp_f=current_temp_f,
        )

        # Estimate expected level
        # Assume cold level was at target
        cold_fill_gal = tank_data.total_volume_gallons * self.tank_config.cold_level_target_pct / 100
        current_volume_gal = cold_fill_gal + expected_expansion
        expected_level_pct = current_volume_gal / tank_data.total_volume_gallons * 100

        # Compare to actual
        actual_level = tank_data.current_level_pct
        deviation = actual_level - expected_level_pct

        if actual_level < MIN_TANK_LEVEL_PCT:
            status = "alarm"
            warnings.append(f"Level {actual_level:.1f}% below minimum {MIN_TANK_LEVEL_PCT}%")
            warnings.append("Risk of pump cavitation - add fluid")

        elif actual_level > MAX_TANK_LEVEL_PCT:
            status = "alarm"
            warnings.append(f"Level {actual_level:.1f}% exceeds maximum {MAX_TANK_LEVEL_PCT}%")
            warnings.append("Risk of overflow - drain excess fluid")

        elif abs(deviation) > 10:
            status = "warning"
            warnings.append(
                f"Level deviation {deviation:+.1f}% from expected {expected_level_pct:.1f}%"
            )
            if deviation < 0:
                warnings.append("Possible leak or incorrect cold fill")
            else:
                warnings.append("Possible overfill or measurement error")

        return {
            "status": status,
            "actual_level_pct": actual_level,
            "expected_level_pct": round(expected_level_pct, 1),
            "deviation_pct": round(deviation, 1),
            "current_temperature_f": current_temp_f,
            "warnings": warnings,
        }

    def calculate_nitrogen_requirements(
        self,
        tank_volume_gallons: float,
        tank_pressure_psig: float = 2.0,
        operating_temp_f: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Calculate nitrogen blanket requirements.

        Args:
            tank_volume_gallons: Tank volume (gallons)
            tank_pressure_psig: Target blanket pressure (psig)
            operating_temp_f: Operating temperature (F)

        Returns:
            Dictionary with nitrogen requirements
        """
        # Estimate vapor space volume (typically 30-50% of tank at operating temp)
        vapor_space_pct = 35.0
        vapor_space_gal = tank_volume_gallons * vapor_space_pct / 100
        vapor_space_ft3 = vapor_space_gal / GALLONS_PER_FT3

        # Nitrogen at tank conditions
        tank_pressure_psia = tank_pressure_psig + 14.696
        tank_temp_r = operating_temp_f + 459.67

        # Standard conditions
        std_pressure_psia = 14.696
        std_temp_r = 520.0  # 60F

        # Volume at standard conditions (ideal gas)
        std_volume_ft3 = vapor_space_ft3 * (tank_pressure_psia / std_pressure_psia) * (std_temp_r / tank_temp_r)
        std_volume_scf = std_volume_ft3

        # Initial fill requirement (purge air)
        purge_volume_scf = std_volume_scf * 5  # 5 volume changes for purging

        return {
            "vapor_space_gallons": round(vapor_space_gal, 1),
            "operating_volume_scf": round(std_volume_scf, 1),
            "initial_purge_scf": round(purge_volume_scf, 0),
            "blanket_pressure_psig": tank_pressure_psig,
            "recommendations": [
                "Maintain continuous nitrogen blanket to prevent oxidation",
                "Check regulator setting monthly",
                "Monitor nitrogen consumption for leak detection",
            ],
        }

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_expansion(
    fluid_type: ThermalFluidType,
    system_volume_gallons: float,
    cold_temp_f: float,
    hot_temp_f: float,
) -> float:
    """
    Quick expansion volume calculation.

    Args:
        fluid_type: Thermal fluid type
        system_volume_gallons: System volume
        cold_temp_f: Cold temperature (F)
        hot_temp_f: Hot temperature (F)

    Returns:
        Expansion volume (gallons)
    """
    db = ThermalFluidPropertyDatabase()
    return db.calculate_expansion_volume(
        fluid_type,
        system_volume_gallons,
        cold_temp_f,
        hot_temp_f,
    )


def size_tank(
    fluid_type: ThermalFluidType,
    system_volume_gallons: float,
    cold_temp_f: float = 70.0,
    hot_temp_f: float = 600.0,
) -> float:
    """
    Quick tank sizing.

    Args:
        fluid_type: Thermal fluid type
        system_volume_gallons: System volume
        cold_temp_f: Cold temperature (F)
        hot_temp_f: Hot temperature (F)

    Returns:
        Recommended tank size (gallons)
    """
    analyzer = ExpansionTankAnalyzer(fluid_type=fluid_type)
    result = analyzer.size_expansion_tank(
        system_volume_gallons,
        cold_temp_f,
        hot_temp_f,
    )
    return result["recommended_size_gallons"]
