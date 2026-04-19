# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Condensate Load Calculation Module

This module provides deterministic condensate load calculations for steam
trap sizing, including startup loads, operating loads, and DOE-recommended
safety factors.

Features:
    - Startup condensate load calculation (pipe warming)
    - Operating condensate load calculation (heat transfer losses)
    - DOE Best Practice safety factors
    - Multiple application types (drip leg, process, tracer)
    - Complete provenance tracking

Standards:
    - DOE Steam System Best Practices
    - Spirax Sarco Sizing Guidelines
    - ASHRAE Fundamentals

Calculations are ZERO-HALLUCINATION: All formulas are deterministic
engineering calculations with documented references.

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor.condensate_load import (
    ...     CondensateLoadCalculator,
    ... )
    >>> calculator = CondensateLoadCalculator(steam_pressure_psig=150)
    >>> result = calculator.calculate_drip_leg_load(
    ...     pipe_diameter_in=6.0,
    ...     pipe_length_ft=100.0,
    ... )
    >>> print(f"Design load: {result.design_load_lb_hr} lb/hr")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    CondensateLoadInput,
    CondensateLoadOutput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Engineering Reference Values
# =============================================================================

class PipeConstants:
    """Pipe properties for common materials and sizes."""

    # Pipe weight per foot (lb/ft) for Schedule 40 steel pipe
    PIPE_WEIGHT_LB_PER_FT: Dict[float, float] = {
        0.5: 0.85,
        0.75: 1.13,
        1.0: 1.68,
        1.25: 2.27,
        1.5: 2.72,
        2.0: 3.65,
        2.5: 5.79,
        3.0: 7.58,
        4.0: 10.79,
        6.0: 18.97,
        8.0: 28.55,
        10.0: 40.48,
        12.0: 49.56,
    }

    # Pipe outside diameter (inches) for Schedule 40
    PIPE_OD_IN: Dict[float, float] = {
        0.5: 0.840,
        0.75: 1.050,
        1.0: 1.315,
        1.25: 1.660,
        1.5: 1.900,
        2.0: 2.375,
        2.5: 2.875,
        3.0: 3.500,
        4.0: 4.500,
        6.0: 6.625,
        8.0: 8.625,
        10.0: 10.750,
        12.0: 12.750,
    }

    # Specific heat of pipe materials (BTU/lb-F)
    SPECIFIC_HEAT_STEEL = 0.12
    SPECIFIC_HEAT_STAINLESS = 0.12
    SPECIFIC_HEAT_COPPER = 0.092


class InsulationConstants:
    """Insulation thermal conductivity values."""

    # Thermal conductivity (BTU-in/hr-ft2-F) at mean temperature
    K_CALCIUM_SILICATE = 0.36  # at 200F
    K_MINERAL_WOOL = 0.27      # at 200F
    K_FIBERGLASS = 0.24        # at 200F
    K_CELLULAR_GLASS = 0.35    # at 200F
    K_PERLITE = 0.38           # at 200F


class SteamTableConstants:
    """Steam table reference values."""

    # Saturation temperature (F) at pressure (psig)
    SATURATION_TEMP: Dict[int, float] = {
        0: 212.0,
        15: 250.0,
        30: 274.0,
        50: 298.0,
        75: 320.0,
        100: 338.0,
        125: 353.0,
        150: 366.0,
        175: 377.0,
        200: 388.0,
        250: 406.0,
        300: 422.0,
        400: 448.0,
        500: 470.0,
        600: 489.0,
    }

    # Latent heat of vaporization (BTU/lb) at pressure (psig)
    LATENT_HEAT: Dict[int, float] = {
        0: 970.3,
        15: 945.5,
        30: 928.6,
        50: 911.6,
        75: 894.7,
        100: 880.6,
        125: 868.2,
        150: 857.0,
        175: 847.0,
        200: 837.7,
        250: 820.4,
        300: 805.0,
        400: 777.5,
        500: 752.5,
        600: 729.6,
    }

    # Specific volume of steam (ft3/lb) at pressure (psig)
    SPECIFIC_VOLUME: Dict[int, float] = {
        0: 26.80,
        15: 13.75,
        30: 9.40,
        50: 6.66,
        75: 4.89,
        100: 3.88,
        125: 3.22,
        150: 2.75,
        175: 2.40,
        200: 2.13,
        250: 1.73,
        300: 1.45,
        400: 1.09,
        500: 0.857,
        600: 0.698,
    }


# =============================================================================
# SAFETY FACTORS (DOE Best Practices)
# =============================================================================

class SafetyFactors:
    """DOE-recommended safety factors for trap sizing."""

    # Startup safety factors
    STARTUP_DRIP_LEG = 3.0
    STARTUP_HEAT_EXCHANGER = 2.0
    STARTUP_TRACER = 2.0
    STARTUP_UNIT_HEATER = 2.0
    STARTUP_PROCESS = 2.0

    # Operating safety factors
    OPERATING_DRIP_LEG = 2.0
    OPERATING_HEAT_EXCHANGER = 2.0
    OPERATING_TRACER = 2.0
    OPERATING_UNIT_HEATER = 2.0
    OPERATING_PROCESS = 2.0

    @classmethod
    def get_startup_factor(cls, application: str) -> float:
        """Get startup safety factor for application."""
        factors = {
            "drip_leg": cls.STARTUP_DRIP_LEG,
            "heat_exchanger": cls.STARTUP_HEAT_EXCHANGER,
            "tracer": cls.STARTUP_TRACER,
            "unit_heater": cls.STARTUP_UNIT_HEATER,
            "process": cls.STARTUP_PROCESS,
        }
        return factors.get(application.lower(), 2.0)

    @classmethod
    def get_operating_factor(cls, application: str) -> float:
        """Get operating safety factor for application."""
        factors = {
            "drip_leg": cls.OPERATING_DRIP_LEG,
            "heat_exchanger": cls.OPERATING_HEAT_EXCHANGER,
            "tracer": cls.OPERATING_TRACER,
            "unit_heater": cls.OPERATING_UNIT_HEATER,
            "process": cls.OPERATING_PROCESS,
        }
        return factors.get(application.lower(), 2.0)


# =============================================================================
# CALCULATION RESULT MODELS
# =============================================================================

class LoadCalculationResult(BaseModel):
    """Result of a condensate load calculation."""

    load_lb_hr: float = Field(
        ...,
        ge=0,
        description="Calculated condensate load (lb/hr)"
    )
    calculation_type: str = Field(
        ...,
        description="Calculation type (startup, operating, peak)"
    )
    formula_id: str = Field(
        ...,
        description="Formula identifier for provenance"
    )
    formula_reference: str = Field(
        ...,
        description="Engineering reference for formula"
    )
    inputs_hash: str = Field(
        ...,
        description="SHA-256 hash of calculation inputs"
    )
    intermediate_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Intermediate calculation values"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Calculation warnings"
    )


# =============================================================================
# STARTUP LOAD CALCULATOR
# =============================================================================

class StartupLoadCalculator:
    """
    Calculator for startup condensate loads.

    Startup loads occur when steam is first admitted to a cold system.
    The condensate load from warming the pipe/equipment is typically
    much higher than operating loads.

    Formula (DOE Steam Tip Sheet #1):
        Q_startup = (W_pipe * Cp * dT) / (t_startup * h_fg)

    Where:
        W_pipe = Weight of pipe (lb)
        Cp = Specific heat of pipe material (BTU/lb-F)
        dT = Temperature rise (F)
        t_startup = Startup time (hours)
        h_fg = Latent heat of vaporization (BTU/lb)
    """

    def __init__(self) -> None:
        """Initialize the startup load calculator."""
        self._calculation_count = 0

    def calculate_pipe_warming_load(
        self,
        pipe_diameter_in: float,
        pipe_length_ft: float,
        steam_pressure_psig: float,
        ambient_temp_f: float = 70.0,
        startup_time_minutes: float = 15.0,
        pipe_material: str = "carbon_steel",
    ) -> LoadCalculationResult:
        """
        Calculate startup condensate load for pipe warming.

        This is the condensate generated when steam first enters a cold
        pipe and heats it from ambient to steam temperature.

        Args:
            pipe_diameter_in: Nominal pipe diameter (inches)
            pipe_length_ft: Pipe length (feet)
            steam_pressure_psig: Steam pressure (psig)
            ambient_temp_f: Starting temperature (F)
            startup_time_minutes: Warmup period (minutes)
            pipe_material: Pipe material

        Returns:
            LoadCalculationResult with startup load

        Raises:
            ValueError: If pipe diameter not in lookup table
        """
        self._calculation_count += 1

        # Calculate inputs hash
        inputs = {
            "diameter": pipe_diameter_in,
            "length": pipe_length_ft,
            "pressure": steam_pressure_psig,
            "ambient": ambient_temp_f,
            "startup_time": startup_time_minutes,
            "material": pipe_material,
        }
        inputs_hash = self._hash_inputs(inputs)

        warnings = []

        # Get pipe weight per foot
        if pipe_diameter_in not in PipeConstants.PIPE_WEIGHT_LB_PER_FT:
            # Interpolate or use nearest
            available = sorted(PipeConstants.PIPE_WEIGHT_LB_PER_FT.keys())
            if pipe_diameter_in < available[0]:
                pipe_weight_per_ft = PipeConstants.PIPE_WEIGHT_LB_PER_FT[available[0]]
                warnings.append(
                    f"Pipe diameter {pipe_diameter_in}\" below minimum, "
                    f"using {available[0]}\" weight"
                )
            elif pipe_diameter_in > available[-1]:
                pipe_weight_per_ft = PipeConstants.PIPE_WEIGHT_LB_PER_FT[available[-1]]
                warnings.append(
                    f"Pipe diameter {pipe_diameter_in}\" above maximum, "
                    f"using {available[-1]}\" weight"
                )
            else:
                # Linear interpolation
                for i in range(len(available) - 1):
                    if available[i] <= pipe_diameter_in <= available[i + 1]:
                        d1, d2 = available[i], available[i + 1]
                        w1 = PipeConstants.PIPE_WEIGHT_LB_PER_FT[d1]
                        w2 = PipeConstants.PIPE_WEIGHT_LB_PER_FT[d2]
                        pipe_weight_per_ft = w1 + (w2 - w1) * (
                            (pipe_diameter_in - d1) / (d2 - d1)
                        )
                        break
        else:
            pipe_weight_per_ft = PipeConstants.PIPE_WEIGHT_LB_PER_FT[pipe_diameter_in]

        # Total pipe weight
        total_pipe_weight = pipe_weight_per_ft * pipe_length_ft

        # Get steam properties
        sat_temp = self._get_saturation_temp(steam_pressure_psig)
        latent_heat = self._get_latent_heat(steam_pressure_psig)

        # Temperature rise
        delta_t = sat_temp - ambient_temp_f

        if delta_t <= 0:
            warnings.append("Ambient temperature at or above steam temperature")
            return LoadCalculationResult(
                load_lb_hr=0.0,
                calculation_type="startup",
                formula_id="STARTUP_PIPE_WARMING",
                formula_reference="DOE Steam Tip Sheet #1",
                inputs_hash=inputs_hash,
                intermediate_values={
                    "pipe_weight_lb": total_pipe_weight,
                    "delta_t_f": delta_t,
                },
                warnings=warnings,
            )

        # Specific heat of pipe material
        if "stainless" in pipe_material.lower():
            specific_heat = PipeConstants.SPECIFIC_HEAT_STAINLESS
        elif "copper" in pipe_material.lower():
            specific_heat = PipeConstants.SPECIFIC_HEAT_COPPER
        else:
            specific_heat = PipeConstants.SPECIFIC_HEAT_STEEL

        # Heat required to warm pipe (BTU)
        heat_required = total_pipe_weight * specific_heat * delta_t

        # Convert startup time to hours
        startup_time_hours = startup_time_minutes / 60.0

        # Condensate load (lb/hr)
        # Q = (W * Cp * dT) / (t * h_fg)
        if startup_time_hours > 0 and latent_heat > 0:
            startup_load = heat_required / (startup_time_hours * latent_heat)
        else:
            startup_load = 0.0

        logger.debug(
            f"Startup load calculation: {startup_load:.1f} lb/hr "
            f"for {pipe_diameter_in}\" x {pipe_length_ft}' pipe"
        )

        return LoadCalculationResult(
            load_lb_hr=round(startup_load, 2),
            calculation_type="startup",
            formula_id="STARTUP_PIPE_WARMING",
            formula_reference="DOE Steam Tip Sheet #1",
            inputs_hash=inputs_hash,
            intermediate_values={
                "pipe_weight_lb": round(total_pipe_weight, 2),
                "pipe_weight_per_ft": round(pipe_weight_per_ft, 2),
                "specific_heat": specific_heat,
                "delta_t_f": round(delta_t, 1),
                "saturation_temp_f": round(sat_temp, 1),
                "latent_heat_btu_lb": round(latent_heat, 1),
                "heat_required_btu": round(heat_required, 0),
                "startup_time_hours": startup_time_hours,
            },
            warnings=warnings,
        )

    def _get_saturation_temp(self, pressure_psig: float) -> float:
        """Get saturation temperature from steam tables."""
        pressures = sorted(SteamTableConstants.SATURATION_TEMP.keys())

        if pressure_psig <= pressures[0]:
            return SteamTableConstants.SATURATION_TEMP[pressures[0]]
        if pressure_psig >= pressures[-1]:
            return SteamTableConstants.SATURATION_TEMP[pressures[-1]]

        # Linear interpolation
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1 = SteamTableConstants.SATURATION_TEMP[p1]
                t2 = SteamTableConstants.SATURATION_TEMP[p2]
                return t1 + (t2 - t1) * ((pressure_psig - p1) / (p2 - p1))

        return 212.0

    def _get_latent_heat(self, pressure_psig: float) -> float:
        """Get latent heat of vaporization from steam tables."""
        pressures = sorted(SteamTableConstants.LATENT_HEAT.keys())

        if pressure_psig <= pressures[0]:
            return SteamTableConstants.LATENT_HEAT[pressures[0]]
        if pressure_psig >= pressures[-1]:
            return SteamTableConstants.LATENT_HEAT[pressures[-1]]

        # Linear interpolation
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                h1 = SteamTableConstants.LATENT_HEAT[p1]
                h2 = SteamTableConstants.LATENT_HEAT[p2]
                return h1 + (h2 - h1) * ((pressure_psig - p1) / (p2 - p1))

        return 970.3

    def _hash_inputs(self, inputs: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of inputs."""
        inputs_str = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(inputs_str.encode()).hexdigest()

    @property
    def calculation_count(self) -> int:
        """Get calculation count."""
        return self._calculation_count


# =============================================================================
# OPERATING LOAD CALCULATOR
# =============================================================================

class OperatingLoadCalculator:
    """
    Calculator for operating (running) condensate loads.

    Operating loads are the steady-state condensate formation rate
    due to heat losses from insulated or bare pipe, or heat transfer
    in process equipment.

    Formula (Spirax Sarco):
        For insulated pipe: Q = (A * U * dT) / h_fg
        For heat transfer: Q = Q_heat / h_fg

    Where:
        A = Surface area (ft2)
        U = Overall heat transfer coefficient (BTU/hr-ft2-F)
        dT = Temperature difference (F)
        h_fg = Latent heat of vaporization (BTU/lb)
        Q_heat = Heat transfer rate (BTU/hr)
    """

    def __init__(self) -> None:
        """Initialize the operating load calculator."""
        self._calculation_count = 0

    def calculate_pipe_heat_loss_load(
        self,
        pipe_diameter_in: float,
        pipe_length_ft: float,
        steam_pressure_psig: float,
        ambient_temp_f: float = 70.0,
        insulation_thickness_in: float = 2.0,
        insulation_type: str = "calcium_silicate",
        wind_speed_mph: float = 0.0,
    ) -> LoadCalculationResult:
        """
        Calculate operating condensate load from pipe heat loss.

        Args:
            pipe_diameter_in: Nominal pipe diameter (inches)
            pipe_length_ft: Pipe length (feet)
            steam_pressure_psig: Steam pressure (psig)
            ambient_temp_f: Ambient temperature (F)
            insulation_thickness_in: Insulation thickness (inches)
            insulation_type: Insulation type
            wind_speed_mph: Wind speed for outdoor (mph)

        Returns:
            LoadCalculationResult with operating load
        """
        self._calculation_count += 1

        inputs = {
            "diameter": pipe_diameter_in,
            "length": pipe_length_ft,
            "pressure": steam_pressure_psig,
            "ambient": ambient_temp_f,
            "insulation_thickness": insulation_thickness_in,
            "insulation_type": insulation_type,
            "wind_speed": wind_speed_mph,
        }
        inputs_hash = self._hash_inputs(inputs)

        warnings = []

        # Get pipe outer diameter
        if pipe_diameter_in in PipeConstants.PIPE_OD_IN:
            pipe_od = PipeConstants.PIPE_OD_IN[pipe_diameter_in]
        else:
            # Approximate OD for nominal diameter
            pipe_od = pipe_diameter_in * 1.1
            warnings.append(
                f"Pipe OD for {pipe_diameter_in}\" approximated"
            )

        # Get steam properties
        sat_temp = self._get_saturation_temp(steam_pressure_psig)
        latent_heat = self._get_latent_heat(steam_pressure_psig)

        # Temperature difference
        delta_t = sat_temp - ambient_temp_f

        if delta_t <= 0:
            return LoadCalculationResult(
                load_lb_hr=0.0,
                calculation_type="operating",
                formula_id="PIPE_HEAT_LOSS",
                formula_reference="Spirax Sarco Engineering Guide",
                inputs_hash=inputs_hash,
                intermediate_values={"delta_t_f": delta_t},
                warnings=["No temperature difference - no heat loss"],
            )

        # Get insulation thermal conductivity
        k_insulation = self._get_insulation_k(insulation_type)

        # Calculate heat loss per linear foot
        if insulation_thickness_in > 0:
            # Insulated pipe heat loss (BTU/hr-ft)
            # Using simplified log-mean radius formula
            r1 = pipe_od / 2  # Pipe outer radius (inches)
            r2 = r1 + insulation_thickness_in  # Insulation outer radius

            # Thermal resistance of insulation (hr-ft-F/BTU per linear foot)
            # R = ln(r2/r1) / (2 * pi * k)
            r_insulation = math.log(r2 / r1) / (2 * math.pi * k_insulation / 12)

            # Surface heat transfer coefficient (BTU/hr-ft2-F)
            # Simplified: h = 1.0 + 0.7 * sqrt(wind_speed) for outdoor
            h_surface = 1.0 + 0.7 * math.sqrt(wind_speed_mph)

            # Outer surface area per linear foot (ft2/ft)
            a_outer = math.pi * (r2 / 12) * 2 / 12  # Convert to ft

            # Surface resistance
            r_surface = 1 / (h_surface * a_outer) if a_outer > 0 else 0

            # Total resistance
            r_total = r_insulation + r_surface

            # Heat loss per linear foot
            if r_total > 0:
                heat_loss_per_ft = delta_t / r_total
            else:
                heat_loss_per_ft = 0

        else:
            # Bare pipe heat loss (BTU/hr-ft)
            # Higher heat transfer without insulation
            pipe_surface_area_per_ft = math.pi * pipe_od / 12  # ft2/ft
            h_bare = 2.0 + 1.0 * math.sqrt(wind_speed_mph)  # Simplified
            heat_loss_per_ft = h_bare * pipe_surface_area_per_ft * delta_t
            warnings.append("Bare pipe - consider adding insulation")

        # Total heat loss (BTU/hr)
        total_heat_loss = heat_loss_per_ft * pipe_length_ft

        # Condensate load (lb/hr)
        if latent_heat > 0:
            operating_load = total_heat_loss / latent_heat
        else:
            operating_load = 0.0

        logger.debug(
            f"Operating load calculation: {operating_load:.2f} lb/hr "
            f"for {pipe_diameter_in}\" x {pipe_length_ft}' pipe"
        )

        return LoadCalculationResult(
            load_lb_hr=round(operating_load, 2),
            calculation_type="operating",
            formula_id="PIPE_HEAT_LOSS",
            formula_reference="Spirax Sarco Engineering Guide",
            inputs_hash=inputs_hash,
            intermediate_values={
                "saturation_temp_f": round(sat_temp, 1),
                "delta_t_f": round(delta_t, 1),
                "latent_heat_btu_lb": round(latent_heat, 1),
                "heat_loss_per_ft_btu_hr": round(heat_loss_per_ft, 2),
                "total_heat_loss_btu_hr": round(total_heat_loss, 0),
                "k_insulation": k_insulation,
            },
            warnings=warnings,
        )

    def calculate_heat_transfer_load(
        self,
        heat_transfer_rate_btu_hr: float,
        steam_pressure_psig: float,
    ) -> LoadCalculationResult:
        """
        Calculate condensate load from heat transfer rate.

        For heat exchangers, reboilers, and process equipment where
        the heat transfer rate is known.

        Args:
            heat_transfer_rate_btu_hr: Heat transfer rate (BTU/hr)
            steam_pressure_psig: Steam pressure (psig)

        Returns:
            LoadCalculationResult
        """
        self._calculation_count += 1

        inputs = {
            "heat_rate": heat_transfer_rate_btu_hr,
            "pressure": steam_pressure_psig,
        }
        inputs_hash = self._hash_inputs(inputs)

        # Get latent heat
        latent_heat = self._get_latent_heat(steam_pressure_psig)

        # Condensate load = Q / h_fg
        if latent_heat > 0:
            condensate_load = heat_transfer_rate_btu_hr / latent_heat
        else:
            condensate_load = 0.0

        return LoadCalculationResult(
            load_lb_hr=round(condensate_load, 2),
            calculation_type="operating",
            formula_id="HEAT_TRANSFER_LOAD",
            formula_reference="Basic Heat Transfer",
            inputs_hash=inputs_hash,
            intermediate_values={
                "heat_rate_btu_hr": heat_transfer_rate_btu_hr,
                "latent_heat_btu_lb": round(latent_heat, 1),
            },
            warnings=[],
        )

    def _get_saturation_temp(self, pressure_psig: float) -> float:
        """Get saturation temperature from steam tables."""
        pressures = sorted(SteamTableConstants.SATURATION_TEMP.keys())

        if pressure_psig <= pressures[0]:
            return SteamTableConstants.SATURATION_TEMP[pressures[0]]
        if pressure_psig >= pressures[-1]:
            return SteamTableConstants.SATURATION_TEMP[pressures[-1]]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1 = SteamTableConstants.SATURATION_TEMP[p1]
                t2 = SteamTableConstants.SATURATION_TEMP[p2]
                return t1 + (t2 - t1) * ((pressure_psig - p1) / (p2 - p1))

        return 212.0

    def _get_latent_heat(self, pressure_psig: float) -> float:
        """Get latent heat of vaporization from steam tables."""
        pressures = sorted(SteamTableConstants.LATENT_HEAT.keys())

        if pressure_psig <= pressures[0]:
            return SteamTableConstants.LATENT_HEAT[pressures[0]]
        if pressure_psig >= pressures[-1]:
            return SteamTableConstants.LATENT_HEAT[pressures[-1]]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                h1 = SteamTableConstants.LATENT_HEAT[p1]
                h2 = SteamTableConstants.LATENT_HEAT[p2]
                return h1 + (h2 - h1) * ((pressure_psig - p1) / (p2 - p1))

        return 970.3

    def _get_insulation_k(self, insulation_type: str) -> float:
        """Get insulation thermal conductivity."""
        k_values = {
            "calcium_silicate": InsulationConstants.K_CALCIUM_SILICATE,
            "mineral_wool": InsulationConstants.K_MINERAL_WOOL,
            "fiberglass": InsulationConstants.K_FIBERGLASS,
            "cellular_glass": InsulationConstants.K_CELLULAR_GLASS,
            "perlite": InsulationConstants.K_PERLITE,
        }
        return k_values.get(insulation_type.lower(), InsulationConstants.K_CALCIUM_SILICATE)

    def _hash_inputs(self, inputs: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of inputs."""
        inputs_str = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(inputs_str.encode()).hexdigest()

    @property
    def calculation_count(self) -> int:
        """Get calculation count."""
        return self._calculation_count


# =============================================================================
# SAFETY FACTOR CALCULATOR
# =============================================================================

class SafetyFactorCalculator:
    """
    Calculator for applying DOE-recommended safety factors.

    Safety factors account for:
    - Measurement uncertainties
    - Load variations
    - Startup conditions
    - Future capacity needs

    DOE Best Practice recommends:
    - 2:1 for normal operating conditions
    - 3:1 for startup/warmup conditions
    """

    def apply_safety_factor(
        self,
        base_load_lb_hr: float,
        application: str,
        load_type: str = "operating",
        custom_factor: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Apply safety factor to condensate load.

        Args:
            base_load_lb_hr: Base calculated load (lb/hr)
            application: Application type
            load_type: "startup" or "operating"
            custom_factor: Custom safety factor (overrides default)

        Returns:
            Tuple of (design_load, safety_factor_used)
        """
        if custom_factor is not None:
            factor = custom_factor
        elif load_type.lower() == "startup":
            factor = SafetyFactors.get_startup_factor(application)
        else:
            factor = SafetyFactors.get_operating_factor(application)

        design_load = base_load_lb_hr * factor

        logger.debug(
            f"Applied safety factor {factor} to {base_load_lb_hr:.1f} lb/hr "
            f"-> {design_load:.1f} lb/hr"
        )

        return round(design_load, 2), factor


# =============================================================================
# MAIN CONDENSATE LOAD CALCULATOR
# =============================================================================

class CondensateLoadCalculator:
    """
    Main condensate load calculator combining startup and operating loads.

    This class provides a unified interface for calculating condensate
    loads for trap sizing, applying appropriate safety factors per
    DOE Best Practices.

    All calculations are ZERO-HALLUCINATION: deterministic engineering
    formulas with documented references.

    Example:
        >>> calculator = CondensateLoadCalculator(steam_pressure_psig=150)
        >>> result = calculator.calculate_drip_leg_load(
        ...     pipe_diameter_in=6.0,
        ...     pipe_length_ft=100.0,
        ... )
        >>> print(f"Design load: {result.design_load_lb_hr} lb/hr")
    """

    def __init__(self, steam_pressure_psig: float = 150.0) -> None:
        """
        Initialize the condensate load calculator.

        Args:
            steam_pressure_psig: Default steam pressure for calculations
        """
        self.steam_pressure_psig = steam_pressure_psig
        self._startup_calc = StartupLoadCalculator()
        self._operating_calc = OperatingLoadCalculator()
        self._safety_calc = SafetyFactorCalculator()

        logger.info(
            f"CondensateLoadCalculator initialized at {steam_pressure_psig} psig"
        )

    def calculate_drip_leg_load(
        self,
        pipe_diameter_in: float,
        pipe_length_ft: float,
        steam_pressure_psig: Optional[float] = None,
        ambient_temp_f: float = 70.0,
        insulation_thickness_in: float = 2.0,
        insulation_type: str = "calcium_silicate",
        startup_time_minutes: float = 15.0,
        include_startup: bool = True,
        apply_safety_factor: bool = True,
    ) -> CondensateLoadOutput:
        """
        Calculate condensate load for a steam main drip leg.

        Drip leg sizing must consider both startup (pipe warming) loads
        and operating (heat loss) loads.

        Args:
            pipe_diameter_in: Pipe diameter (inches)
            pipe_length_ft: Pipe length (feet)
            steam_pressure_psig: Steam pressure (optional, uses default)
            ambient_temp_f: Ambient temperature (F)
            insulation_thickness_in: Insulation thickness (inches)
            insulation_type: Insulation type
            startup_time_minutes: Warmup time (minutes)
            include_startup: Include startup load in sizing
            apply_safety_factor: Apply DOE safety factors

        Returns:
            CondensateLoadOutput with complete sizing information
        """
        pressure = steam_pressure_psig or self.steam_pressure_psig
        warnings = []

        # Calculate startup load
        startup_result = self._startup_calc.calculate_pipe_warming_load(
            pipe_diameter_in=pipe_diameter_in,
            pipe_length_ft=pipe_length_ft,
            steam_pressure_psig=pressure,
            ambient_temp_f=ambient_temp_f,
            startup_time_minutes=startup_time_minutes,
        )
        startup_load = startup_result.load_lb_hr
        warnings.extend(startup_result.warnings)

        # Calculate operating load
        operating_result = self._operating_calc.calculate_pipe_heat_loss_load(
            pipe_diameter_in=pipe_diameter_in,
            pipe_length_ft=pipe_length_ft,
            steam_pressure_psig=pressure,
            ambient_temp_f=ambient_temp_f,
            insulation_thickness_in=insulation_thickness_in,
            insulation_type=insulation_type,
        )
        operating_load = operating_result.load_lb_hr
        warnings.extend(operating_result.warnings)

        # Determine peak load (typically startup)
        if include_startup:
            peak_load = max(startup_load, operating_load)
        else:
            peak_load = operating_load

        # Apply safety factor
        if apply_safety_factor:
            design_load, safety_factor = self._safety_calc.apply_safety_factor(
                peak_load,
                application="drip_leg",
                load_type="startup" if include_startup else "operating",
            )
        else:
            design_load = peak_load
            safety_factor = 1.0

        # Calculate provenance hash
        provenance_data = {
            "pipe_diameter": pipe_diameter_in,
            "pipe_length": pipe_length_ft,
            "pressure": pressure,
            "startup_load": startup_load,
            "operating_load": operating_load,
            "design_load": design_load,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return CondensateLoadOutput(
            request_id=startup_result.inputs_hash[:16],
            startup_load_lb_hr=startup_load,
            operating_load_lb_hr=operating_load,
            peak_load_lb_hr=peak_load,
            safety_factor=safety_factor,
            design_load_lb_hr=design_load,
            recommended_trap_capacity_lb_hr=design_load * 1.1,  # 10% margin
            recommended_trap_types=["inverted_bucket", "thermodynamic"],
            calculation_method="DOE Steam Tip Sheet #1",
            formula_reference="DOE/GO-102012-3393",
            warnings=warnings,
            provenance_hash=provenance_hash,
        )

    def calculate_heat_exchanger_load(
        self,
        heat_transfer_rate_btu_hr: float,
        steam_pressure_psig: Optional[float] = None,
        apply_safety_factor: bool = True,
    ) -> CondensateLoadOutput:
        """
        Calculate condensate load for heat exchanger.

        Args:
            heat_transfer_rate_btu_hr: Heat transfer rate (BTU/hr)
            steam_pressure_psig: Steam pressure (optional)
            apply_safety_factor: Apply DOE safety factors

        Returns:
            CondensateLoadOutput
        """
        pressure = steam_pressure_psig or self.steam_pressure_psig

        # Calculate operating load
        operating_result = self._operating_calc.calculate_heat_transfer_load(
            heat_transfer_rate_btu_hr=heat_transfer_rate_btu_hr,
            steam_pressure_psig=pressure,
        )
        operating_load = operating_result.load_lb_hr

        # For heat exchangers, startup typically uses same load
        startup_load = operating_load

        # Apply safety factor
        if apply_safety_factor:
            design_load, safety_factor = self._safety_calc.apply_safety_factor(
                operating_load,
                application="heat_exchanger",
                load_type="operating",
            )
        else:
            design_load = operating_load
            safety_factor = 1.0

        # Provenance
        provenance_data = {
            "heat_rate": heat_transfer_rate_btu_hr,
            "pressure": pressure,
            "design_load": design_load,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return CondensateLoadOutput(
            request_id=operating_result.inputs_hash[:16],
            startup_load_lb_hr=startup_load,
            operating_load_lb_hr=operating_load,
            peak_load_lb_hr=operating_load,
            safety_factor=safety_factor,
            design_load_lb_hr=design_load,
            recommended_trap_capacity_lb_hr=design_load * 1.1,
            recommended_trap_types=["float_thermostatic", "inverted_bucket"],
            calculation_method="Direct Heat Transfer",
            formula_reference="Basic Thermodynamics",
            warnings=[],
            provenance_hash=provenance_hash,
        )

    def process(self, input_data: CondensateLoadInput) -> CondensateLoadOutput:
        """
        Process condensate load calculation from input model.

        Args:
            input_data: Condensate load calculation input

        Returns:
            CondensateLoadOutput
        """
        application = input_data.application.lower()

        if application in ["drip_leg", "tracer"]:
            if input_data.pipe_diameter_in and input_data.pipe_length_ft:
                return self.calculate_drip_leg_load(
                    pipe_diameter_in=input_data.pipe_diameter_in,
                    pipe_length_ft=input_data.pipe_length_ft,
                    steam_pressure_psig=input_data.steam_pressure_psig,
                    ambient_temp_f=input_data.ambient_temperature_f,
                    insulation_thickness_in=input_data.insulation_thickness_in,
                    insulation_type=input_data.insulation_type,
                    startup_time_minutes=input_data.startup_time_minutes,
                    include_startup=input_data.calculate_startup,
                    apply_safety_factor=True,
                )
            else:
                raise ValueError(
                    "Pipe diameter and length required for drip leg calculation"
                )

        elif application in ["heat_exchanger", "process", "reboiler"]:
            if input_data.heat_transfer_rate_btu_hr:
                return self.calculate_heat_exchanger_load(
                    heat_transfer_rate_btu_hr=input_data.heat_transfer_rate_btu_hr,
                    steam_pressure_psig=input_data.steam_pressure_psig,
                    apply_safety_factor=True,
                )
            else:
                raise ValueError(
                    "Heat transfer rate required for heat exchanger calculation"
                )

        else:
            raise ValueError(f"Unknown application type: {application}")

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return (
            self._startup_calc.calculation_count +
            self._operating_calc.calculation_count
        )
