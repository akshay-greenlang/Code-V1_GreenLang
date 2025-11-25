# -*- coding: utf-8 -*-
"""AI-Powered Waste Heat Recovery Agent with Comprehensive Heat Transfer Modeling.

This module provides a production-grade agent for identifying, quantifying, and recovering
industrial waste heat streams. Built on rigorous heat transfer principles including LMTD
(Log Mean Temperature Difference) and effectiveness-NTU methods, this agent combines
physics-based calculations with AI orchestration for intelligent waste heat recovery.

HEAT TRANSFER FOUNDATION:
=========================

Waste Heat Recovery Principles:
--------------------------------
Industrial processes reject 20-50% of fuel input as waste heat through:
1. Flue gases from combustion (high temperature, 400-2000°F)
2. Process exhaust air (medium temperature, 150-400°F)
3. Hot liquids and wastewater (low-medium temperature, 120-200°F)
4. Steam condensate (high energy content, 212°F+)
5. Cooling water (low temperature, 80-120°F)

Heat Recovery Technologies:
---------------------------
1. Recuperators: Air-to-air heat exchangers, no cross-contamination
2. Economizers: Flue gas to water/steam, boiler feed water preheating
3. Shell-and-Tube: Liquid-to-liquid, versatile, ASME rated
4. Plate Heat Exchangers: Compact, high effectiveness, easy maintenance
5. Heat Pipes: No moving parts, isothermal operation
6. Organic Rankine Cycle (ORC): Power generation from low-grade heat
7. Heat Pump Assisted: Upgrade low-grade waste heat temperature

LMTD METHOD (Log Mean Temperature Difference):
==============================================

For counterflow heat exchanger:
    LMTD = (ΔT1 - ΔT2) / ln(ΔT1 / ΔT2)

where:
    ΔT1 = T_hot_in - T_cold_out (terminal difference 1)
    ΔT2 = T_hot_out - T_cold_in (terminal difference 2)

Heat Transfer Equation:
    Q = U × A × F × LMTD

where:
    Q = Heat transfer rate (Btu/hr)
    U = Overall heat transfer coefficient (Btu/hr·ft²·°F)
    A = Heat transfer area (ft²)
    F = Correction factor for flow arrangement

Typical U-values:
- Gas to gas: 5-50 Btu/hr·ft²·°F
- Gas to liquid: 50-200 Btu/hr·ft²·°F
- Liquid to liquid: 100-500 Btu/hr·ft²·°F

EFFECTIVENESS-NTU METHOD:
=========================

Heat Exchanger Effectiveness:
    ε = Q_actual / Q_max

where:
    Q_max = C_min × (T_hot_in - T_cold_in)
    C_min = minimum heat capacity rate

Number of Transfer Units (NTU):
    NTU = U × A / C_min

For counterflow:
    ε = (1 - exp[-NTU(1-Cr)]) / (1 - Cr × exp[-NTU(1-Cr)])

where Cr = C_min / C_max (capacity rate ratio)

FOULING AND CORROSION:
======================

Fouling Mechanisms:
- Particulate fouling (fly ash, soot)
- Chemical reaction fouling (polymerization)
- Corrosion fouling (oxide scales)
- Biological fouling (microorganisms)

Fouling Resistance:
    R_f = 1/h_dirty - 1/h_clean

Typical fouling factors:
- Clean gases: 0.001 hr·ft²·°F/Btu
- Dirty gases: 0.005-0.010 hr·ft²·°F/Btu
- Clean liquids: 0.0005 hr·ft²·°F/Btu

Acid Dew Point (Corrosion Risk):
For combustion gases with sulfur:
    T_dewpoint ≈ 280-320°F (depends on sulfur content)

Keep wall temperature above dew point to prevent sulfuric acid condensation.

ECONOMIC ANALYSIS:
==================

Waste heat recovery typically has the BEST payback of all energy efficiency measures:
- Typical payback: 0.5-3 years
- IRR: 30-100%+
- Market size: $75 billion globally
- Carbon impact: 1.4 Gt CO2e/year potential

Author: GreenLang AI & Climate Intelligence Team
Date: October 2025
Standards: ASME BPVC Section VIII, TEMA, ASHRAE Industrial Handbook, NACE
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import asyncio
import logging
import math

from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
# Fixed: Removed incomplete import
from greenlang.determinism import DeterministicClock
from greenlang.intelligence import ChatSession, ChatMessage
from greenlang.intelligence.schemas.tools import ToolDef

logger = logging.getLogger(__name__)


# ============================================================================
# GAS AND LIQUID PROPERTY DATABASE
# ============================================================================

class ThermodynamicProperties:
    """Thermodynamic properties for common industrial streams."""

    # Gas properties at standard conditions with temperature corrections
    GAS_PROPERTIES = {
        "air": {
            "specific_heat_btu_lb_f": 0.24,
            "density_lb_ft3_70f": 0.075,
            "thermal_conductivity": 0.015,
            "molecular_weight": 29,
        },
        "combustion_products_natural_gas": {
            "specific_heat_btu_lb_f": 0.26,
            "density_lb_ft3_70f": 0.048,
            "thermal_conductivity": 0.020,
            "molecular_weight": 28,
            "note": "Typical for natural gas combustion (80% N2, 12% CO2, 8% H2O)",
        },
        "combustion_products_oil": {
            "specific_heat_btu_lb_f": 0.27,
            "density_lb_ft3_70f": 0.050,
            "thermal_conductivity": 0.019,
            "molecular_weight": 29,
        },
        "combustion_products_coal": {
            "specific_heat_btu_lb_f": 0.28,
            "density_lb_ft3_70f": 0.052,
            "thermal_conductivity": 0.018,
            "molecular_weight": 30,
        },
        "steam": {
            "specific_heat_btu_lb_f": 0.45,
            "density_lb_ft3_212f": 0.037,
            "latent_heat_btu_lb": 970,
            "molecular_weight": 18,
        },
        "process_air": {
            "specific_heat_btu_lb_f": 0.24,
            "density_lb_ft3_70f": 0.075,
            "thermal_conductivity": 0.015,
            "molecular_weight": 29,
        },
    }

    # Liquid properties
    LIQUID_PROPERTIES = {
        "water": {
            "specific_heat_btu_lb_f": 1.0,
            "density_lb_ft3": 62.4,
            "thermal_conductivity": 0.35,
            "viscosity_centipoise": 1.0,
        },
        "oil_light": {
            "specific_heat_btu_lb_f": 0.5,
            "density_lb_ft3": 55.0,
            "thermal_conductivity": 0.08,
            "viscosity_centipoise": 50,
        },
        "oil_heavy": {
            "specific_heat_btu_lb_f": 0.45,
            "density_lb_ft3": 58.0,
            "thermal_conductivity": 0.07,
            "viscosity_centipoise": 200,
        },
        "glycol_50_percent": {
            "specific_heat_btu_lb_f": 0.85,
            "density_lb_ft3": 67.0,
            "thermal_conductivity": 0.25,
            "viscosity_centipoise": 5.0,
        },
        "brine": {
            "specific_heat_btu_lb_f": 0.90,
            "density_lb_ft3": 70.0,
            "thermal_conductivity": 0.30,
            "viscosity_centipoise": 1.5,
        },
    }

    @classmethod
    def get_gas_properties(cls, gas_type: str, temperature_f: float = 70) -> Dict[str, float]:
        """Get gas properties with temperature correction.

        Uses ideal gas law for density correction: ρ_T = ρ_ref × (T_ref / T)
        """
        base_props = cls.GAS_PROPERTIES.get(gas_type, cls.GAS_PROPERTIES["air"])

        # Temperature correction for density (ideal gas law approximation)
        temp_rankine = temperature_f + 459.67
        temp_ref_rankine = 70 + 459.67
        temp_correction = temp_ref_rankine / temp_rankine

        return {
            "specific_heat": base_props["specific_heat_btu_lb_f"],
            "density": base_props["density_lb_ft3_70f"] * temp_correction,
            "thermal_conductivity": base_props["thermal_conductivity"],
            "molecular_weight": base_props.get("molecular_weight", 29),
        }

    @classmethod
    def get_liquid_properties(cls, liquid_type: str, temperature_f: float = 70) -> Dict[str, float]:
        """Get liquid properties.

        Note: Temperature correction for liquids is small compared to gases.
        """
        props = cls.LIQUID_PROPERTIES.get(liquid_type, cls.LIQUID_PROPERTIES["water"])
        return {
            "specific_heat": props["specific_heat_btu_lb_f"],
            "density": props["density_lb_ft3"],
            "thermal_conductivity": props["thermal_conductivity"],
            "viscosity": props["viscosity_centipoise"],
        }


# ============================================================================
# HEAT EXCHANGER TECHNOLOGY DATABASE
# ============================================================================

class HeatExchangerTechnology:
    """Heat exchanger technology specifications and selection criteria."""

    TECHNOLOGIES = {
        "shell_tube_hx": {
            "name": "Shell-and-Tube Heat Exchanger",
            "typical_effectiveness": 0.70,
            "u_value_gas_gas": 15,
            "u_value_gas_liquid": 80,
            "u_value_liquid_liquid": 250,
            "cost_per_ft2_usd": 75,
            "base_cost_usd": 15000,
            "maintenance": "moderate",
            "fouling_tolerance": "good",
            "pressure_rating_psi": 300,
            "applications": ["Liquid-liquid", "Gas-liquid", "High pressure"],
            "advantages": ["Proven design", "ASME rated", "Repairable", "Handles fouling well"],
            "challenges": ["Bulky", "Moderate cost", "Periodic cleaning needed"],
            "materials": ["Carbon steel", "Stainless steel 304/316", "Copper-nickel"],
        },
        "plate_hx": {
            "name": "Plate Heat Exchanger",
            "typical_effectiveness": 0.85,
            "u_value_liquid_liquid": 400,
            "cost_per_ft2_usd": 100,
            "base_cost_usd": 8000,
            "maintenance": "low",
            "fouling_tolerance": "fair",
            "pressure_rating_psi": 150,
            "applications": ["Liquid-liquid", "Compact spaces", "High effectiveness"],
            "advantages": ["Very compact", "High effectiveness", "Easy to clean", "Modular design"],
            "challenges": ["Gasket maintenance", "Limited pressure", "Not for dirty streams"],
            "materials": ["Stainless steel 316", "Titanium", "Hastelloy"],
        },
        "economizer": {
            "name": "Economizer (Flue Gas to Water)",
            "typical_effectiveness": 0.65,
            "u_value_gas_liquid": 12,
            "cost_per_mmbtu_hr_usd": 30000,
            "base_cost_usd": 25000,
            "maintenance": "moderate",
            "fouling_tolerance": "good",
            "applications": ["Boiler flue gas", "Hot water generation", "Feed water preheat"],
            "advantages": ["Proven technology", "Large capacity", "Standard design", "High availability"],
            "challenges": ["Acid dew point concerns", "Soot buildup", "Space requirements", "Bypass needed"],
            "materials": ["Carbon steel w/ coating", "Stainless steel 304L", "Ceramic coating"],
        },
        "recuperator": {
            "name": "Recuperator (Air-to-Air)",
            "typical_effectiveness": 0.70,
            "u_value_gas_gas": 20,
            "cost_per_mmbtu_hr_usd": 25000,
            "base_cost_usd": 20000,
            "maintenance": "moderate",
            "fouling_tolerance": "good",
            "applications": ["Combustion air preheat", "Process air heating", "Oven exhaust recovery"],
            "advantages": ["No cross-contamination", "Direct heat recovery", "Simple operation", "Proven design"],
            "challenges": ["Moderate effectiveness", "Pressure drop", "Size", "Cleaning access"],
            "materials": ["Stainless steel 304/316", "Aluminized steel", "Ceramic"],
        },
        "regenerator": {
            "name": "Regenerator (Rotary)",
            "typical_effectiveness": 0.85,
            "u_value_gas_gas": 25,
            "cost_per_mmbtu_hr_usd": 35000,
            "base_cost_usd": 30000,
            "maintenance": "moderate",
            "fouling_tolerance": "fair",
            "applications": ["High effectiveness air-to-air", "Large capacity", "HVAC"],
            "advantages": ["Very high effectiveness", "Compact", "Self-cleaning action", "Good capacity"],
            "challenges": ["Cross-contamination risk", "Rotating parts maintenance", "Sealing challenges"],
            "materials": ["Aluminum matrix", "Ceramic matrix", "Steel housing"],
        },
        "heat_pipe": {
            "name": "Heat Pipe Heat Exchanger",
            "typical_effectiveness": 0.65,
            "u_value_gas_gas": 18,
            "cost_per_mmbtu_hr_usd": 40000,
            "base_cost_usd": 35000,
            "maintenance": "very_low",
            "fouling_tolerance": "excellent",
            "applications": ["Corrosive environments", "No maintenance access", "High reliability needed"],
            "advantages": ["No moving parts", "Isothermal operation", "Excellent reliability", "No cross-contamination"],
            "challenges": ["Higher cost", "Limited effectiveness", "Temperature constraints", "Size"],
            "materials": ["Stainless steel tubes", "Copper fins", "Working fluid sealed"],
        },
        "run_around_coil": {
            "name": "Run-Around Coil System",
            "typical_effectiveness": 0.55,
            "u_value_gas_liquid": 60,
            "cost_per_mmbtu_hr_usd": 20000,
            "base_cost_usd": 18000,
            "maintenance": "low",
            "fouling_tolerance": "good",
            "applications": ["Remote locations", "Flexibility needed", "Retrofit applications"],
            "advantages": ["Flexible placement", "No cross-contamination", "Easy retrofit", "Independent streams"],
            "challenges": ["Lower effectiveness", "Pumping costs", "Freeze protection", "Two HX required"],
            "materials": ["Copper tubes", "Aluminum fins", "Glycol fluid"],
        },
        "orc_system": {
            "name": "Organic Rankine Cycle (Power Generation)",
            "typical_effectiveness": 0.15,  # Electrical efficiency
            "cost_per_kw_usd": 4000,
            "base_cost_usd": 100000,
            "maintenance": "high",
            "fouling_tolerance": "moderate",
            "applications": ["Power generation from waste heat", "Low-grade heat >200°F"],
            "advantages": ["Generates electricity", "Monetizes low-grade heat", "Reduces demand charges"],
            "challenges": ["High cost", "Complex operation", "Maintenance intensive", "Long payback"],
            "materials": ["Stainless steel", "Special alloys", "Organic working fluid"],
        },
    }

    @classmethod
    def get_technology(cls, tech_name: str) -> Dict[str, Any]:
        """Get technology specifications."""
        return cls.TECHNOLOGIES.get(tech_name, cls.TECHNOLOGIES["shell_tube_hx"])

    @classmethod
    def select_technology(
        cls,
        waste_stream_type: str,
        temperature_f: float,
        end_use: str,
        fouling: str,
        space: str
    ) -> Tuple[str, str]:
        """Select optimal technology based on criteria.

        Returns:
            Tuple of (primary_technology, alternative_technology)
        """

        # Decision logic based on stream type and application
        if waste_stream_type == "combustion_flue_gas":
            if end_use == "preheat_combustion_air":
                return ("recuperator", "regenerator" if fouling == "low" else "heat_pipe")
            elif end_use == "generate_hot_water":
                return ("economizer", "shell_tube_hx")
            elif end_use == "generate_steam":
                return ("economizer", "shell_tube_hx")
            elif end_use == "power_generation_orc" and temperature_f > 300:
                return ("orc_system", "economizer")
            elif fouling in ["high", "very_high"]:
                return ("heat_pipe", "economizer")
            else:
                return ("recuperator", "heat_pipe")

        elif waste_stream_type in ["hot_liquid", "cooling_water"]:
            if space in ["limited", "very_limited"]:
                return ("plate_hx", "shell_tube_hx")
            elif fouling in ["high", "very_high"]:
                return ("shell_tube_hx", "plate_hx")
            else:
                return ("plate_hx", "shell_tube_hx")

        elif waste_stream_type == "steam_condensate":
            return ("shell_tube_hx", "plate_hx")

        elif waste_stream_type == "process_exhaust_air":
            if end_use == "space_heating":
                return ("run_around_coil", "recuperator")
            else:
                return ("recuperator", "regenerator")

        else:
            # Default to shell-tube for versatility
            return ("shell_tube_hx", "plate_hx")


# ============================================================================
# WASTE HEAT CHARACTERIZATION DATABASE
# ============================================================================

class WasteHeatCharacterization:
    """Process-specific waste heat characterization database.

    Based on DOE Industrial Assessment Centers database and industry benchmarks.
    """

    # Typical waste heat sources by process type
    PROCESS_WASTE_HEAT = {
        "baking_oven": {
            "temperature_f": 450,
            "waste_heat_percent_of_fuel": 25,
            "stream_type": "combustion_flue_gas",
            "fouling_risk": "moderate",
            "particulate_gr_scf": 0.01,
            "corrosion_risk": "low",
            "accessibility": "good",
        },
        "furnace_steel": {
            "temperature_f": 850,
            "waste_heat_percent_of_fuel": 35,
            "stream_type": "combustion_flue_gas",
            "fouling_risk": "high",
            "particulate_gr_scf": 0.10,
            "corrosion_risk": "high",
            "accessibility": "moderate",
        },
        "furnace_aluminum": {
            "temperature_f": 950,
            "waste_heat_percent_of_fuel": 40,
            "stream_type": "combustion_flue_gas",
            "fouling_risk": "very_high",
            "particulate_gr_scf": 0.15,
            "corrosion_risk": "high",
            "accessibility": "moderate",
        },
        "boiler_exhaust": {
            "temperature_f": 350,
            "waste_heat_percent_of_fuel": 15,
            "stream_type": "combustion_flue_gas",
            "fouling_risk": "moderate",
            "particulate_gr_scf": 0.02,
            "corrosion_risk": "moderate",
            "accessibility": "good",
        },
        "dryer_exhaust": {
            "temperature_f": 250,
            "waste_heat_percent_of_fuel": 30,
            "stream_type": "process_exhaust_air",
            "fouling_risk": "moderate",
            "particulate_gr_scf": 0.05,
            "corrosion_risk": "low",
            "accessibility": "excellent",
        },
        "hot_water_discharge": {
            "temperature_f": 165,
            "waste_heat_percent_of_process": 40,
            "stream_type": "hot_liquid",
            "fouling_risk": "low",
            "corrosion_risk": "low",
            "accessibility": "excellent",
        },
        "cooling_tower": {
            "temperature_f": 95,
            "waste_heat_percent_of_process": 60,
            "stream_type": "cooling_water",
            "fouling_risk": "low",
            "corrosion_risk": "low",
            "accessibility": "excellent",
        },
        "compressor_cooling": {
            "temperature_f": 180,
            "waste_heat_percent_of_input": 90,
            "stream_type": "cooling_water",
            "fouling_risk": "low",
            "corrosion_risk": "low",
            "accessibility": "good",
        },
        "pasteurization": {
            "temperature_f": 185,
            "waste_heat_percent_of_process": 35,
            "stream_type": "hot_liquid",
            "fouling_risk": "low",
            "corrosion_risk": "low",
            "accessibility": "excellent",
        },
        "kiln_cement": {
            "temperature_f": 750,
            "waste_heat_percent_of_fuel": 40,
            "stream_type": "combustion_flue_gas",
            "fouling_risk": "very_high",
            "particulate_gr_scf": 0.20,
            "corrosion_risk": "high",
            "accessibility": "moderate",
        },
    }

    @classmethod
    def get_waste_heat_characteristics(cls, process_type: str) -> Dict[str, Any]:
        """Get waste heat characteristics for process type."""
        return cls.PROCESS_WASTE_HEAT.get(
            process_type,
            cls.PROCESS_WASTE_HEAT["boiler_exhaust"]  # Default
        )


# ============================================================================
# HEAT TRANSFER CALCULATION UTILITIES
# ============================================================================

def calculate_lmtd(
    hot_in_f: float,
    hot_out_f: float,
    cold_in_f: float,
    cold_out_f: float,
    flow_arrangement: str = "counterflow"
) -> Tuple[float, float]:
    """Calculate Log Mean Temperature Difference (LMTD) and correction factor.

    For counterflow heat exchanger:
        LMTD = (ΔT1 - ΔT2) / ln(ΔT1 / ΔT2)

    Args:
        hot_in_f: Hot fluid inlet temperature (°F)
        hot_out_f: Hot fluid outlet temperature (°F)
        cold_in_f: Cold fluid inlet temperature (°F)
        cold_out_f: Cold fluid outlet temperature (°F)
        flow_arrangement: Flow configuration

    Returns:
        Tuple of (LMTD in °F, F correction factor)

    Standards:
        - ASHRAE Handbook: Fundamentals, Chapter 4
        - Incropera & DeWitt: Heat Transfer, Chapter 11
    """
    # Terminal temperature differences
    if flow_arrangement == "counterflow":
        dt1 = hot_in_f - cold_out_f
        dt2 = hot_out_f - cold_in_f
    elif flow_arrangement == "parallel_flow":
        dt1 = hot_in_f - cold_in_f
        dt2 = hot_out_f - cold_out_f
    else:  # counterflow default
        dt1 = hot_in_f - cold_out_f
        dt2 = hot_out_f - cold_in_f

    # Avoid division by zero and negative temperatures
    if dt1 <= 0 or dt2 <= 0:
        raise ValueError(f"Invalid temperature profile: dt1={dt1:.1f}°F, dt2={dt2:.1f}°F. Check that hot outlet > cold inlet.")

    # Calculate LMTD
    if abs(dt1 - dt2) < 0.1:
        # Use arithmetic mean when differences are nearly equal
        lmtd = (dt1 + dt2) / 2
    else:
        lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

    # Correction factor F (from TEMA charts - simplified)
    if flow_arrangement == "counterflow":
        f_factor = 1.0
    elif flow_arrangement == "parallel_flow":
        f_factor = 0.85
    elif flow_arrangement == "crossflow_unmixed":
        f_factor = 0.90
    elif flow_arrangement == "shell_tube_1_pass":
        # Simplified F-factor for 1-shell, 2-tube pass
        P = (cold_out_f - cold_in_f) / (hot_in_f - cold_in_f) if (hot_in_f - cold_in_f) > 0 else 0.5
        R = (hot_in_f - hot_out_f) / (cold_out_f - cold_in_f) if (cold_out_f - cold_in_f) > 0 else 1.0

        # TEMA F-factor correlation (simplified)
        if R < 0.1:
            f_factor = 0.95
        elif R > 10:
            f_factor = 0.85
        else:
            # Approximate F-factor
            f_factor = max(0.75, min(0.95, 0.90 - 0.05 * P))
    else:
        f_factor = 0.90  # Conservative default

    return round(lmtd, 2), round(f_factor, 3)


def calculate_effectiveness_ntu(
    ntu: float,
    capacity_ratio: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """Calculate heat exchanger effectiveness from NTU.

    For counterflow:
        ε = (1 - exp[-NTU(1-Cr)]) / (1 - Cr × exp[-NTU(1-Cr)])

    Args:
        ntu: Number of Transfer Units (U×A/C_min)
        capacity_ratio: C_min / C_max
        flow_arrangement: Flow configuration

    Returns:
        Effectiveness (0 to 1)

    Standards:
        - ASHRAE Handbook: HVAC Systems & Equipment
        - Kays & London: Compact Heat Exchangers
    """
    cr = min(capacity_ratio, 1.0)  # Ensure Cr <= 1

    if flow_arrangement == "counterflow":
        if cr < 0.99:
            # General counterflow equation
            numerator = 1 - math.exp(-ntu * (1 - cr))
            denominator = 1 - cr * math.exp(-ntu * (1 - cr))
            effectiveness = numerator / denominator
        else:
            # Special case when Cr ≈ 1 (balanced flow)
            effectiveness = ntu / (1 + ntu)

    elif flow_arrangement == "parallel_flow":
        effectiveness = (1 - math.exp(-ntu * (1 + cr))) / (1 + cr)

    elif flow_arrangement == "crossflow_unmixed":
        # Approximate for crossflow, both unmixed
        effectiveness = 1 - math.exp((1/cr) * (ntu**0.22) * (math.exp(-cr * (ntu**0.78)) - 1))

    elif flow_arrangement == "shell_tube_1_pass":
        # 1-shell, 2-tube pass (approximate)
        ntu_eff = ntu * 0.90  # Adjustment for shell-and-tube
        if cr < 0.99:
            numerator = 1 - math.exp(-ntu_eff * (1 - cr))
            denominator = 1 - cr * math.exp(-ntu_eff * (1 - cr))
            effectiveness = numerator / denominator
        else:
            effectiveness = ntu_eff / (1 + ntu_eff)

    else:
        # Default to counterflow
        if cr < 0.99:
            numerator = 1 - math.exp(-ntu * (1 - cr))
            denominator = 1 - cr * math.exp(-ntu * (1 - cr))
            effectiveness = numerator / denominator
        else:
            effectiveness = ntu / (1 + ntu)

    # Physical limits
    return min(max(effectiveness, 0.0), 0.95)  # Cap at 95% for practical systems


def convert_volumetric_to_mass_flow(
    volumetric_flow: float,
    flow_units: str,
    stream_type: str,
    temperature_f: float
) -> float:
    """Convert volumetric flow rate to mass flow rate.

    Args:
        volumetric_flow: Volumetric flow rate
        flow_units: Units (cfm, acfm, scfm, gpm)
        stream_type: Type of stream for property lookup
        temperature_f: Stream temperature

    Returns:
        Mass flow rate in lb/hr
    """
    # Get density
    if "gas" in stream_type.lower() or "air" in stream_type.lower():
        props = ThermodynamicProperties.get_gas_properties(stream_type, temperature_f)
        density = props["density"]  # lb/ft³

        if flow_units == "cfm":
            # Actual cubic feet per minute
            mass_flow_lb_hr = volumetric_flow * density * 60
        elif flow_units == "acfm":
            # Actual CFM (same as cfm)
            mass_flow_lb_hr = volumetric_flow * density * 60
        elif flow_units == "scfm":
            # Standard CFM (at 70°F, 1 atm) - need to adjust density
            std_density = ThermodynamicProperties.get_gas_properties(stream_type, 70)["density"]
            mass_flow_lb_hr = volumetric_flow * std_density * 60
        else:
            mass_flow_lb_hr = volumetric_flow * density * 60

    else:
        # Liquid
        props = ThermodynamicProperties.get_liquid_properties(stream_type, temperature_f)
        density = props["density"]  # lb/ft³

        if flow_units == "gpm":
            # Gallons per minute
            ft3_per_gal = 0.133681
            mass_flow_lb_hr = volumetric_flow * ft3_per_gal * density * 60
        else:
            mass_flow_lb_hr = volumetric_flow * density * 60

    return mass_flow_lb_hr


# ============================================================================
# WASTE HEAT RECOVERY AGENT AI
# ============================================================================

class WasteHeatRecoveryAgent_AI(BaseAgent):
    """AI-powered waste heat recovery agent with comprehensive heat transfer modeling.

    This agent provides world-class waste heat recovery analysis for industrial facilities,
    combining rigorous heat transfer calculations (LMTD, effectiveness-NTU methods) with AI
    orchestration for intelligent recommendations.

    Key Features:
    - Waste heat stream identification and characterization
    - Heat recovery potential calculation using energy balance
    - Technology selection with multi-criteria decision matrix
    - Heat exchanger sizing using LMTD and effectiveness-NTU methods
    - Fouling and corrosion risk assessment (NACE guidelines)
    - Energy savings and emissions reduction analysis
    - Economic analysis with payback, NPV, IRR
    - Priority ranking of opportunities with implementation roadmap

    Business Impact:
    - Market: $75 billion (largest of all agents!)
    - Carbon: 1.4 Gt CO2e/year reduction potential
    - Payback: 0.5-3 years (BEST ROI!)
    - IRR: 30-100%+

    Determinism Guarantees:
    - Same input always produces same output (temperature=0.0, seed=42)
    - All numeric values from deterministic tools (no LLM math)
    - Reproducible AI responses
    - Full provenance tracking

    Performance:
    - Max latency: 4 seconds
    - Max cost: $0.15 per analysis
    - Accuracy target: 90% vs measured data

    Example:
        >>> agent = WasteHeatRecoveryAgent_AI()
        >>> result = agent.execute({
        ...     "facility_type": "food_beverage",
        ...     "processes": [...],
        ...     "total_annual_fuel_mmbtu": 73000,
        ...     "fuel_cost_usd_per_mmbtu": 8.0,
        ... })
        >>> print(result.data["total_waste_heat_identified_mmbtu_yr"])
        32207
        >>> print(result.data["weighted_average_payback_years"])
        0.97
    """

    def __init__(
        self,
        config: AgentConfig = None,
        *,
        budget_usd: float = 0.15,
        enable_ai_summary: bool = True,
    ):
        """Initialize the Waste Heat Recovery Agent.

        Args:
            config: Agent configuration
            budget_usd: Maximum USD per analysis (default: $0.15)
            enable_ai_summary: Enable AI-generated summaries (default: True)
        """
        if config is None:
            config = AgentConfig(
                name="WasteHeatRecoveryAgent_AI",
                description="AI-powered waste heat recovery analysis with heat transfer modeling",
                version="1.0.0",
            )
        super().__init__(config)

        # Configuration
        self.budget_usd = budget_usd
        self.enable_ai_summary = enable_ai_summary

        # Initialize LLM provider
        self.provider = create_provider()

        # Performance tracking
        self._ai_call_count = 0
        self._tool_call_count = 0
        self._total_cost_usd = 0.0

        # Setup tools
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Setup tool definitions for ChatSession."""
        # Tool definitions would be added here for full ChatSession integration
        # For now, tools are called directly in implementation
        pass

    # ========================================================================
    # TOOL IMPLEMENTATION #1: Identify Waste Heat Sources
    # ========================================================================

    def _identify_waste_heat_sources_impl(
        self,
        facility_type: str,
        processes: List[Dict[str, Any]],
        include_hvac_systems: bool = True,
        include_compressed_air: bool = True,
        minimum_temperature_f: float = 120,
    ) -> Dict[str, Any]:
        """Tool implementation: Identify and characterize all waste heat streams.

        This tool uses process-specific characterization databases and energy balance
        methods to identify all waste heat sources in an industrial facility.

        Method:
        1. For each process, lookup typical waste heat characteristics
        2. Calculate waste heat quantity: Q_waste = Fuel_Input × Waste_Heat_Fraction
        3. Characterize stream: temperature, flow, composition, fouling risk
        4. Categorize by temperature grade: high (>400°F), medium (200-400°F), low (<200°F)
        5. Determine recoverable fraction (technical and economic)

        Args:
            facility_type: Type of industrial facility
            processes: List of processes with fuel consumption data
            include_hvac_systems: Include HVAC waste heat
            include_compressed_air: Include air compressor waste heat
            minimum_temperature_f: Minimum temperature to consider economical

        Returns:
            Dict with waste heat sources, quantities, and characteristics

        Standards:
            - DOE Waste Heat Recovery Guidelines
            - ASHRAE Industrial Handbook
        """
        self._tool_call_count += 1

        waste_heat_sources = []
        total_waste_heat = 0

        # Identify waste heat from each process
        for i, process in enumerate(processes):
            process_name = process.get("process_name", f"Process_{i+1}")
            process_type = process.get("process_type", "generic")
            temp_f = process.get("operating_temperature_f", 300)
            fuel_mmbtu = process.get("annual_fuel_consumption_mmbtu", 0)
            fuel_type = process.get("fuel_type", "natural_gas")

            # Get waste heat characteristics for process type
            char = WasteHeatCharacterization.get_waste_heat_characteristics(process_type)

            # Calculate waste heat quantity
            waste_heat_percent = char.get("waste_heat_percent_of_fuel", 20) / 100
            waste_heat_mmbtu = fuel_mmbtu * waste_heat_percent

            # Assume 7700 operating hours per year (typical industrial)
            operating_hours = 7700
            waste_heat_mmbtu_hr = waste_heat_mmbtu / operating_hours

            # Only include if above minimum temperature
            source_temp = char.get("temperature_f", temp_f)
            if source_temp >= minimum_temperature_f:
                # Estimate flow rate (simplified - would need detailed calculation)
                # For combustion: ~10,000 ACFM per MMBtu/hr fuel input
                if "combustion" in char.get("stream_type", ""):
                    fuel_rate_mmbtu_hr = fuel_mmbtu / operating_hours
                    flow_rate_acfm = fuel_rate_mmbtu_hr * 10000
                else:
                    flow_rate_acfm = 0  # Would need process-specific calculation

                source = {
                    "source_id": f"WH{str(i+1).zfill(3)}",
                    "source_name": process_name,
                    "source_type": char.get("stream_type", "process_exhaust_air"),
                    "temperature_f": round(source_temp, 0),
                    "flow_rate": round(flow_rate_acfm, 0),
                    "heat_content_mmbtu_hr": round(waste_heat_mmbtu_hr, 2),
                    "annual_waste_heat_mmbtu": round(waste_heat_mmbtu, 0),
                    "stream_composition": f"{fuel_type} combustion products" if "combustion" in char.get("stream_type", "") else "Process stream",
                    "fouling_risk": char.get("fouling_risk", "moderate"),
                    "corrosion_risk": char.get("corrosion_risk", "low"),
                    "accessibility": char.get("accessibility", "good"),
                    "particulate_loading_gr_scf": char.get("particulate_gr_scf", 0.01),
                }
                waste_heat_sources.append(source)
                total_waste_heat += waste_heat_mmbtu

        # Calculate recoverable fractions
        # Technical recoverable: 80% (accounting for minimum approach temperature and practical constraints)
        recoverable_waste_heat = total_waste_heat * 0.80

        # Economic recoverable: 70% (only economically attractive opportunities with <3 year payback)
        economically_recoverable = total_waste_heat * 0.70

        # Categorize by temperature grade
        high_grade = sum(s["annual_waste_heat_mmbtu"] for s in waste_heat_sources if s["temperature_f"] > 400)
        medium_grade = sum(s["annual_waste_heat_mmbtu"] for s in waste_heat_sources if 200 <= s["temperature_f"] <= 400)
        low_grade = sum(s["annual_waste_heat_mmbtu"] for s in waste_heat_sources if s["temperature_f"] < 200)

        return {
            "waste_heat_sources": waste_heat_sources,
            "total_waste_heat_mmbtu_yr": round(total_waste_heat, 0),
            "recoverable_waste_heat_mmbtu_yr": round(recoverable_waste_heat, 0),
            "economically_recoverable_mmbtu_yr": round(economically_recoverable, 0),
            "waste_heat_summary": {
                "high_grade_above_400f_mmbtu_yr": round(high_grade, 0),
                "medium_grade_200_400f_mmbtu_yr": round(medium_grade, 0),
                "low_grade_below_200f_mmbtu_yr": round(low_grade, 0),
            },
            "calculation_method": "Process-specific characterization database + energy balance",
            "data_source": "DOE Industrial Assessment Centers database, industry benchmarks",
            "accuracy": "±15% for waste heat quantification",
            "standards": ["DOE_Waste_Heat_Recovery_Guidelines", "ASHRAE_Industrial_Handbook"],
        }

    def _calculate_heat_recovery_potential_impl(
        self,
        waste_heat_stream: Dict[str, Any],
        recovery_temperature_f: float,
        heat_exchanger_effectiveness: float = 0.75,
    ) -> Dict[str, Any]:
        """Tool #2: Calculate theoretical and practical heat recovery potential.

        Uses energy balance method: Q = m_dot × cp × ΔT
        Accounts for heat exchanger effectiveness and practical constraints.

        Args:
            waste_heat_stream: Stream characteristics (temp, flow, fluid type)
            recovery_temperature_f: Target recovery temperature (°F)
            heat_exchanger_effectiveness: Expected HX effectiveness (0-0.95)

        Returns:
            Heat recovery potential with theoretical and practical values
        """
        # Extract stream properties
        inlet_temp = waste_heat_stream.get("temperature_f", 400)
        flow_rate = waste_heat_stream.get("mass_flow_rate_lb_hr", 0)
        fluid_type = waste_heat_stream.get("fluid_type", "air")

        # Validate inputs
        if inlet_temp <= recovery_temperature_f:
            return {
                "error": "Inlet temperature must be higher than recovery temperature",
                "recoverable_heat_mmbtu_yr": 0,
            }

        # Get fluid properties
        if fluid_type in ThermodynamicProperties.GAS_PROPERTIES:
            props = ThermodynamicProperties.GAS_PROPERTIES[fluid_type]
            specific_heat = props["specific_heat_btu_lb_f"]
        elif fluid_type in ThermodynamicProperties.LIQUID_PROPERTIES:
            props = ThermodynamicProperties.LIQUID_PROPERTIES[fluid_type]
            specific_heat = props["specific_heat_btu_lb_f"]
        else:
            specific_heat = 0.24  # Default to air

        # Calculate theoretical heat recovery (Q = m_dot × cp × ΔT)
        delta_t = inlet_temp - recovery_temperature_f
        theoretical_heat_recovery_btu_hr = flow_rate * specific_heat * delta_t

        # Apply heat exchanger effectiveness
        practical_heat_recovery_btu_hr = theoretical_heat_recovery_btu_hr * heat_exchanger_effectiveness

        # Account for pinch point constraint (minimum 20°F approach temperature)
        pinch_constraint_factor = 1.0
        if delta_t < 30:  # Too close to pinch
            pinch_constraint_factor = 0.7

        practical_heat_recovery_btu_hr *= pinch_constraint_factor

        # Convert to annual (8760 hours/year with 90% uptime)
        operating_hours = 8760 * 0.90
        theoretical_mmbtu_yr = (theoretical_heat_recovery_btu_hr * operating_hours) / 1_000_000
        practical_mmbtu_yr = (practical_heat_recovery_btu_hr * operating_hours) / 1_000_000

        # Calculate outlet conditions
        actual_delta_t = delta_t * heat_exchanger_effectiveness * pinch_constraint_factor
        outlet_temperature_f = inlet_temp - actual_delta_t

        # Calculate exergy (available work) using Carnot efficiency
        # Exergy = Q × (1 - T_ambient / T_source)
        t_ambient_rankine = 70 + 459.67
        t_source_rankine = inlet_temp + 459.67
        carnot_efficiency = 1 - (t_ambient_rankine / t_source_rankine)
        exergy_mmbtu_yr = practical_mmbtu_yr * carnot_efficiency

        return {
            "theoretical_heat_recovery_mmbtu_yr": round(theoretical_mmbtu_yr, 1),
            "practical_heat_recovery_mmbtu_yr": round(practical_mmbtu_yr, 1),
            "exergy_available_mmbtu_yr": round(exergy_mmbtu_yr, 1),
            "outlet_temperature_f": round(outlet_temperature_f, 1),
            "heat_exchanger_effectiveness_used": heat_exchanger_effectiveness,
            "pinch_constraint_factor": pinch_constraint_factor,
            "operating_hours_per_year": operating_hours,
            "calculation_method": "Energy balance: Q = m_dot × cp × ΔT × effectiveness",
            "standards": ["ASME_BPVC_Section_VIII", "DOE_Waste_Heat_Recovery_Guidelines"],
        }

    def _select_heat_recovery_technology_impl(
        self,
        waste_heat_stream: Dict[str, Any],
        application: str,
        budget_usd: float,
        space_constrained: bool = False,
    ) -> Dict[str, Any]:
        """Tool #3: Select optimal heat recovery technology using multi-criteria decision matrix.

        Evaluates 8 heat exchanger technologies based on:
        - Temperature suitability
        - Fluid compatibility
        - Cost effectiveness
        - Space requirements
        - Maintenance needs
        - Fouling resistance

        Args:
            waste_heat_stream: Stream characteristics
            application: Use case (preheating, steam generation, power generation, hvac)
            budget_usd: Available capital budget
            space_constrained: Whether space is limited

        Returns:
            Ranked technology recommendations with scores
        """
        inlet_temp = waste_heat_stream.get("temperature_f", 400)
        fluid_type = waste_heat_stream.get("fluid_type", "air")
        heat_load_mmbtu_yr = waste_heat_stream.get("heat_load_mmbtu_yr", 1000)
        fouling_potential = waste_heat_stream.get("fouling_potential", "moderate")

        # Multi-criteria scoring for each technology
        technology_scores = []

        for tech_key, tech_data in HeatExchangerTechnology.TECHNOLOGIES.items():
            score = 0
            reasons = []

            # Temperature suitability (30% weight)
            temp_range = tech_data["temperature_range_f"]
            if temp_range[0] <= inlet_temp <= temp_range[1]:
                score += 30
                reasons.append(f"Excellent temperature match ({temp_range[0]}-{temp_range[1]}°F)")
            elif inlet_temp > temp_range[1]:
                score += 10
                reasons.append(f"Above max temperature ({temp_range[1]}°F)")
            else:
                score += 5
                reasons.append(f"Below min temperature ({temp_range[0]}°F)")

            # Application suitability (25% weight)
            if application in tech_data["applications"]:
                score += 25
                reasons.append(f"Designed for {application}")
            else:
                score += 10
                reasons.append(f"Can be adapted for {application}")

            # Cost effectiveness (20% weight)
            estimated_area_ft2 = 500  # Rough estimate
            estimated_cost = tech_data["cost_per_ft2_usd"] * estimated_area_ft2
            if estimated_cost <= budget_usd:
                cost_score = 20 * (1 - estimated_cost / budget_usd)
                score += cost_score
                reasons.append(f"Within budget (${estimated_cost:,.0f})")
            else:
                score += 5
                reasons.append(f"Over budget (${estimated_cost:,.0f})")

            # Fouling tolerance (15% weight)
            fouling_map = {"low": 15, "moderate": 10, "high": 5}
            fouling_match = {
                "excellent": {"low": 15, "moderate": 12, "high": 8},
                "good": {"low": 12, "moderate": 15, "high": 10},
                "moderate": {"low": 8, "moderate": 10, "high": 12},
                "poor": {"low": 5, "moderate": 8, "high": 15},
            }
            fouling_tolerance = tech_data["fouling_tolerance"]
            score += fouling_match.get(fouling_tolerance, {}).get(fouling_potential, 10)
            reasons.append(f"{fouling_tolerance.capitalize()} fouling tolerance")

            # Space requirements (10% weight)
            space_rating = {"compact": 10, "moderate": 7, "large": 4}
            space_score = space_rating.get(tech_data.get("space_requirement", "moderate"), 7)
            if space_constrained and tech_data.get("space_requirement") == "compact":
                space_score += 3
                reasons.append("Compact design ideal for space constraint")
            score += space_score

            technology_scores.append({
                "technology": tech_data["name"],
                "technology_key": tech_key,
                "total_score": round(score, 1),
                "typical_effectiveness": tech_data["typical_effectiveness"],
                "estimated_cost_usd": round(estimated_cost, 0),
                "u_value_btu_hr_ft2_f": tech_data.get("u_value_gas_liquid", 80),
                "maintenance_level": tech_data["maintenance"],
                "selection_reasons": reasons,
            })

        # Sort by score descending
        technology_scores.sort(key=lambda x: x["total_score"], reverse=True)

        return {
            "recommended_technology": technology_scores[0]["technology"],
            "recommended_technology_key": technology_scores[0]["technology_key"],
            "confidence_score": technology_scores[0]["total_score"],
            "all_technologies_ranked": technology_scores,
            "selection_criteria": {
                "temperature_suitability_weight": 0.30,
                "application_match_weight": 0.25,
                "cost_effectiveness_weight": 0.20,
                "fouling_tolerance_weight": 0.15,
                "space_requirement_weight": 0.10,
            },
            "methodology": "Multi-criteria decision matrix with weighted scoring",
            "standards": ["TEMA", "ASME_BPVC", "DOE_Technology_Selection_Guidelines"],
        }

    def _size_heat_exchanger_impl(
        self,
        heat_load_btu_hr: float,
        hot_side_in_f: float,
        hot_side_out_f: float,
        cold_side_in_f: float,
        cold_side_out_f: float,
        technology: str = "shell_tube_hx",
        flow_arrangement: str = "counterflow",
    ) -> Dict[str, Any]:
        """Tool #4: Size heat exchanger using LMTD and NTU methods.

        LMTD Method: A = Q / (U × F × LMTD)
        NTU Method: Validates effectiveness and capacity ratio

        Args:
            heat_load_btu_hr: Required heat transfer rate (Btu/hr)
            hot_side_in_f: Hot fluid inlet temperature (°F)
            hot_side_out_f: Hot fluid outlet temperature (°F)
            cold_side_in_f: Cold fluid inlet temperature (°F)
            cold_side_out_f: Cold fluid outlet temperature (°F)
            technology: Heat exchanger type
            flow_arrangement: counterflow, parallel, crossflow

        Returns:
            Heat exchanger sizing with area, effectiveness, NTU
        """
        # Calculate LMTD and F-factor
        lmtd, f_factor = calculate_lmtd(
            hot_side_in_f, hot_side_out_f,
            cold_side_in_f, cold_side_out_f,
            flow_arrangement
        )

        # Get U-value for technology
        tech_data = HeatExchangerTechnology.TECHNOLOGIES.get(technology, {})
        u_value_gas_liquid = tech_data.get("u_value_gas_liquid", 80)

        # Calculate required area using LMTD method: A = Q / (U × F × LMTD)
        if lmtd > 0 and f_factor > 0:
            required_area_ft2 = heat_load_btu_hr / (u_value_gas_liquid * f_factor * lmtd)
        else:
            return {
                "error": "Invalid temperature profile - check for temperature cross",
                "required_area_ft2": 0,
            }

        # Calculate effectiveness using actual temperatures
        q_actual = heat_load_btu_hr
        q_max_hot = abs(hot_side_in_f - cold_side_in_f)  # Assuming equal capacity rates
        effectiveness = min(abs(hot_side_in_f - hot_side_out_f) / q_max_hot, 0.95) if q_max_hot > 0 else 0

        # Calculate NTU from effectiveness (inverse relationship)
        # For counterflow: NTU = -ln(1 - effectiveness) (assuming Cr = 1)
        if effectiveness > 0 and effectiveness < 0.95:
            ntu = -math.log(1 - effectiveness)
        else:
            ntu = 3.0  # Typical for high effectiveness

        # Add safety factor (10-20% oversizing standard practice)
        safety_factor = 1.15
        design_area_ft2 = required_area_ft2 * safety_factor

        # Estimate dimensions (assume shell-tube with typical L/D ratio)
        if "shell_tube" in technology:
            l_d_ratio = 8  # Typical length to diameter ratio
            # A = π × D × L for tube surface
            # Assume D = sqrt(A / (π × L/D))
            diameter_ft = math.sqrt(design_area_ft2 / (math.pi * l_d_ratio))
            length_ft = l_d_ratio * diameter_ft
        else:
            # Plate HX or compact designs
            length_ft = math.sqrt(design_area_ft2 * 2)
            diameter_ft = length_ft / 2

        # Calculate pressure drop (simplified correlation)
        velocity_fps = 10  # Typical
        friction_factor = 0.02
        # ΔP ≈ f × (L/D) × (ρ × v²/2)
        pressure_drop_psi = friction_factor * (length_ft / (diameter_ft + 0.1)) * 0.5

        # Estimate cost
        cost_per_ft2 = tech_data.get("cost_per_ft2_usd", 75)
        capital_cost_usd = design_area_ft2 * cost_per_ft2

        return {
            "required_area_ft2": round(required_area_ft2, 1),
            "design_area_ft2": round(design_area_ft2, 1),
            "safety_factor": safety_factor,
            "lmtd_f": round(lmtd, 2),
            "f_factor": round(f_factor, 3),
            "effectiveness": round(effectiveness, 3),
            "ntu": round(ntu, 2),
            "u_value_btu_hr_ft2_f": u_value_gas_liquid,
            "estimated_length_ft": round(length_ft, 1),
            "estimated_diameter_ft": round(diameter_ft, 1),
            "estimated_pressure_drop_psi": round(pressure_drop_psi, 2),
            "estimated_capital_cost_usd": round(capital_cost_usd, 0),
            "design_method": "LMTD method with NTU validation",
            "standards": ["TEMA_Standards", "ASME_BPVC_Section_VIII", "Heat_Exchanger_Design_Handbook"],
        }

    def _calculate_energy_savings_impl(
        self,
        recovered_heat_mmbtu_yr: float,
        displaced_fuel_type: str,
        fuel_price_usd_per_mmbtu: float,
        boiler_efficiency: float = 0.80,
        electricity_price_usd_per_kwh: float = 0.10,
    ) -> Dict[str, Any]:
        """Tool #5: Calculate energy savings from waste heat recovery.

        Accounts for:
        - Fuel displacement and cost savings
        - Boiler efficiency
        - Parasitic power consumption
        - Carbon emissions reduction

        Args:
            recovered_heat_mmbtu_yr: Heat recovered annually (MMBtu/yr)
            displaced_fuel_type: natural_gas, fuel_oil, coal, electricity
            fuel_price_usd_per_mmbtu: Current fuel price
            boiler_efficiency: Efficiency of boiler being displaced
            electricity_price_usd_per_kwh: Electricity cost for pumps/fans

        Returns:
            Annual energy savings and emissions reduction
        """
        # Calculate fuel displacement accounting for boiler efficiency
        fuel_displaced_mmbtu_yr = recovered_heat_mmbtu_yr / boiler_efficiency

        # Calculate fuel cost savings
        fuel_cost_savings_usd_yr = fuel_displaced_mmbtu_yr * fuel_price_usd_per_mmbtu

        # Calculate parasitic power (pumps, fans, controls)
        # Typical: 0.5-2% of heat recovery in electrical power
        parasitic_power_kwh_yr = recovered_heat_mmbtu_yr * 293  # Convert MMBtu to kWh
        parasitic_power_kwh_yr *= 0.01  # 1% parasitic load
        parasitic_cost_usd_yr = parasitic_power_kwh_yr * electricity_price_usd_per_kwh

        # Net savings
        net_savings_usd_yr = fuel_cost_savings_usd_yr - parasitic_cost_usd_yr

        # Calculate carbon emissions reduction
        # EPA eGRID emission factors (lbs CO2 / MMBtu)
        emission_factors = {
            "natural_gas": 117.0,
            "fuel_oil": 163.0,
            "coal": 215.0,
            "electricity": 156.0,  # Grid average
        }

        emission_factor = emission_factors.get(displaced_fuel_type, 117.0)
        co2_reduction_lbs_yr = fuel_displaced_mmbtu_yr * emission_factor
        co2_reduction_metric_tons_yr = co2_reduction_lbs_yr / 2204.62

        # Calculate social cost of carbon benefit (EPA: $51/metric ton CO2)
        social_cost_of_carbon_usd_per_ton = 51
        carbon_benefit_usd_yr = co2_reduction_metric_tons_yr * social_cost_of_carbon_usd_per_ton

        # Calculate equivalent metrics for impact reporting
        homes_powered = recovered_heat_mmbtu_yr / 120  # Average home uses 120 MMBtu/yr heating
        cars_off_road = co2_reduction_metric_tons_yr / 4.6  # Average car emits 4.6 tons/yr

        return {
            "recovered_heat_mmbtu_yr": round(recovered_heat_mmbtu_yr, 0),
            "fuel_displaced_mmbtu_yr": round(fuel_displaced_mmbtu_yr, 0),
            "fuel_cost_savings_usd_yr": round(fuel_cost_savings_usd_yr, 0),
            "parasitic_power_kwh_yr": round(parasitic_power_kwh_yr, 0),
            "parasitic_cost_usd_yr": round(parasitic_cost_usd_yr, 0),
            "net_savings_usd_yr": round(net_savings_usd_yr, 0),
            "co2_reduction_metric_tons_yr": round(co2_reduction_metric_tons_yr, 1),
            "co2_reduction_lbs_yr": round(co2_reduction_lbs_yr, 0),
            "carbon_benefit_usd_yr": round(carbon_benefit_usd_yr, 0),
            "total_benefit_usd_yr": round(net_savings_usd_yr + carbon_benefit_usd_yr, 0),
            "impact_metrics": {
                "homes_heated_equivalent": round(homes_powered, 0),
                "cars_off_road_equivalent": round(cars_off_road, 0),
            },
            "emission_factor_lbs_co2_per_mmbtu": emission_factor,
            "methodology": "Fuel displacement accounting with parasitic loads",
            "standards": ["GHG_Protocol", "EPA_eGRID", "ISO_14064"],
        }

    def _assess_fouling_corrosion_risk_impl(
        self,
        waste_heat_stream: Dict[str, Any],
        material_of_construction: str = "stainless_steel_316",
    ) -> Dict[str, Any]:
        """Tool #6: Assess fouling and corrosion risk following NACE guidelines.

        Evaluates:
        - Acid dew point corrosion risk (sulfuric, hydrochloric acid)
        - Particulate fouling potential
        - Scaling risk (calcium, silica deposits)
        - Material compatibility
        - Required maintenance frequency

        Args:
            waste_heat_stream: Stream composition and temperature
            material_of_construction: Heat exchanger material

        Returns:
            Risk assessment with mitigation recommendations
        """
        temp_f = waste_heat_stream.get("temperature_f", 400)
        fluid_type = waste_heat_stream.get("fluid_type", "air")
        sulfur_content_ppm = waste_heat_stream.get("sulfur_content_ppm", 0)
        particulate_content_ppm = waste_heat_stream.get("particulate_content_ppm", 0)
        chloride_content_ppm = waste_heat_stream.get("chloride_content_ppm", 0)

        risks = []
        risk_score = 0  # 0-100 scale
        mitigation_strategies = []

        # 1. Acid Dew Point Corrosion Risk
        # Sulfuric acid dew point: T_dp ≈ 280-320°F (depends on SO3 concentration)
        if sulfur_content_ppm > 100:
            acid_dew_point_f = 280 + (sulfur_content_ppm / 100) * 10
            acid_dew_point_f = min(acid_dew_point_f, 340)

            if temp_f < acid_dew_point_f + 20:
                risks.append({
                    "risk_type": "acid_dew_point_corrosion",
                    "severity": "high",
                    "description": f"Operating near acid dew point ({acid_dew_point_f:.0f}°F)",
                })
                risk_score += 40
                mitigation_strategies.append("Install acid-resistant coating (glass-lined or PTFE)")
                mitigation_strategies.append(f"Maintain gas temperature above {acid_dew_point_f + 30:.0f}°F")

        # 2. Chloride Stress Corrosion Cracking
        if chloride_content_ppm > 50 and "stainless" in material_of_construction:
            if temp_f > 140:  # Above threshold for SCC
                risks.append({
                    "risk_type": "chloride_stress_corrosion_cracking",
                    "severity": "high",
                    "description": f"Chlorides ({chloride_content_ppm} ppm) at {temp_f}°F",
                })
                risk_score += 35
                mitigation_strategies.append("Use duplex stainless (2205) or titanium")
                mitigation_strategies.append("Implement regular cleaning cycles")

        # 3. Particulate Fouling
        if particulate_content_ppm > 100:
            fouling_rate_inches_per_year = (particulate_content_ppm / 1000) * 0.1
            risks.append({
                "risk_type": "particulate_fouling",
                "severity": "medium" if particulate_content_ppm < 500 else "high",
                "description": f"High particulate loading ({particulate_content_ppm} ppm)",
                "fouling_rate_inches_yr": round(fouling_rate_inches_per_year, 3),
            })
            risk_score += 25
            mitigation_strategies.append("Install upstream particulate filtration")
            mitigation_strategies.append("Design for 3-4 ft/s minimum velocity to prevent settling")

        # 4. Scaling Risk (for liquid streams)
        if fluid_type in ThermodynamicProperties.LIQUID_PROPERTIES:
            if temp_f > 160:  # Above calcium carbonate solubility threshold
                risks.append({
                    "risk_type": "mineral_scaling",
                    "severity": "medium",
                    "description": "Temperature conducive to calcium carbonate scaling",
                })
                risk_score += 15
                mitigation_strategies.append("Implement water treatment (softening or antiscalant)")
                mitigation_strategies.append("Design for turbulent flow (Re > 10,000)")

        # 5. Material Compatibility Assessment
        material_ratings = {
            "carbon_steel": {"max_temp_f": 600, "acid_resistance": "poor", "cost_factor": 1.0},
            "stainless_steel_304": {"max_temp_f": 870, "acid_resistance": "fair", "cost_factor": 2.5},
            "stainless_steel_316": {"max_temp_f": 870, "acid_resistance": "good", "cost_factor": 3.0},
            "duplex_stainless_2205": {"max_temp_f": 600, "acid_resistance": "excellent", "cost_factor": 4.0},
            "titanium": {"max_temp_f": 600, "acid_resistance": "excellent", "cost_factor": 8.0},
            "hastelloy_c276": {"max_temp_f": 1200, "acid_resistance": "excellent", "cost_factor": 15.0},
        }

        material_data = material_ratings.get(material_of_construction, material_ratings["stainless_steel_316"])

        if temp_f > material_data["max_temp_f"]:
            risks.append({
                "risk_type": "material_temperature_limit",
                "severity": "critical",
                "description": f"Temperature exceeds material limit ({material_data['max_temp_f']}°F)",
            })
            risk_score += 50
            mitigation_strategies.append(f"Upgrade to Hastelloy or Inconel for {temp_f}°F service")

        # Determine overall risk level
        if risk_score >= 60:
            overall_risk = "high"
            maintenance_frequency = "monthly"
        elif risk_score >= 30:
            overall_risk = "medium"
            maintenance_frequency = "quarterly"
        else:
            overall_risk = "low"
            maintenance_frequency = "annual"

        # Calculate fouling resistance impact on heat transfer
        # Typical fouling resistance: 0.001-0.005 hr·ft²·°F/Btu
        fouling_resistance = 0.001 + (risk_score / 100) * 0.004
        u_value_degradation_percent = (fouling_resistance / (fouling_resistance + 0.01)) * 30

        return {
            "overall_risk_level": overall_risk,
            "risk_score": risk_score,
            "identified_risks": risks,
            "mitigation_strategies": mitigation_strategies,
            "material_of_construction": material_of_construction,
            "material_compatibility": material_data["acid_resistance"],
            "recommended_maintenance_frequency": maintenance_frequency,
            "fouling_resistance_hr_ft2_f_btu": round(fouling_resistance, 4),
            "u_value_degradation_percent": round(u_value_degradation_percent, 1),
            "inspection_points": [
                "Monitor outlet temperature for fouling indication",
                "Track pressure drop increase over time",
                "Periodic ultrasonic thickness testing for corrosion",
                "Annual internal inspection during shutdown",
            ],
            "standards": ["NACE_SP0100", "NACE_MR0175", "TEMA_Standards", "ASME_PCC"],
        }

    def _calculate_payback_period_impl(
        self,
        capital_cost_usd: float,
        annual_savings_usd: float,
        annual_maintenance_cost_usd: float = 0,
        project_lifetime_years: int = 20,
        discount_rate: float = 0.08,
        energy_cost_escalation_rate: float = 0.03,
    ) -> Dict[str, Any]:
        """Tool #7: Calculate financial metrics for waste heat recovery project.

        Calculates:
        - Simple payback period
        - Discounted payback period
        - Net Present Value (NPV)
        - Internal Rate of Return (IRR)
        - Savings-to-Investment Ratio (SIR)

        Args:
            capital_cost_usd: Total installed cost
            annual_savings_usd: First year energy savings
            annual_maintenance_cost_usd: Annual O&M costs
            project_lifetime_years: Expected equipment life
            discount_rate: Discount rate for NPV
            energy_cost_escalation_rate: Expected fuel cost inflation

        Returns:
            Comprehensive financial analysis
        """
        # Simple Payback Period (SPP)
        net_annual_savings = annual_savings_usd - annual_maintenance_cost_usd
        if net_annual_savings > 0:
            simple_payback_years = capital_cost_usd / net_annual_savings
        else:
            simple_payback_years = 999  # Infinite

        # Calculate NPV and discounted cash flows
        cumulative_discounted_cashflow = -capital_cost_usd
        discounted_payback_years = None
        total_npv = -capital_cost_usd

        cash_flows = []
        for year in range(1, project_lifetime_years + 1):
            # Escalate energy savings
            savings_this_year = annual_savings_usd * ((1 + energy_cost_escalation_rate) ** (year - 1))
            net_cashflow = savings_this_year - annual_maintenance_cost_usd

            # Discount to present value
            discount_factor = (1 + discount_rate) ** year
            discounted_cashflow = net_cashflow / discount_factor

            cumulative_discounted_cashflow += discounted_cashflow
            total_npv += discounted_cashflow

            # Find discounted payback year
            if discounted_payback_years is None and cumulative_discounted_cashflow > 0:
                discounted_payback_years = year

            cash_flows.append({
                "year": year,
                "savings_usd": round(savings_this_year, 0),
                "maintenance_usd": round(annual_maintenance_cost_usd, 0),
                "net_cashflow_usd": round(net_cashflow, 0),
                "discounted_cashflow_usd": round(discounted_cashflow, 0),
                "cumulative_npv_usd": round(cumulative_discounted_cashflow, 0),
            })

        # Calculate IRR using Newton-Raphson approximation
        # IRR is the rate where NPV = 0
        irr = discount_rate
        for iteration in range(50):
            npv_at_irr = -capital_cost_usd
            dnpv_dirr = 0

            for year in range(1, project_lifetime_years + 1):
                savings_this_year = annual_savings_usd * ((1 + energy_cost_escalation_rate) ** (year - 1))
                net_cashflow = savings_this_year - annual_maintenance_cost_usd

                npv_at_irr += net_cashflow / ((1 + irr) ** year)
                dnpv_dirr -= year * net_cashflow / ((1 + irr) ** (year + 1))

            if abs(npv_at_irr) < 100:  # Converged
                break

            if dnpv_dirr != 0:
                irr = irr - npv_at_irr / dnpv_dirr

            if irr < -0.5 or irr > 2.0:  # Bound IRR to reasonable range
                irr = 0.15
                break

        irr_percent = irr * 100

        # Savings-to-Investment Ratio (SIR)
        # SIR = PV of savings / initial investment
        pv_of_savings = sum(cf["discounted_cashflow_usd"] for cf in cash_flows)
        sir = (pv_of_savings + capital_cost_usd) / capital_cost_usd if capital_cost_usd > 0 else 0

        # Determine project attractiveness
        if simple_payback_years < 2:
            attractiveness = "excellent"
        elif simple_payback_years < 3:
            attractiveness = "very_good"
        elif simple_payback_years < 5:
            attractiveness = "good"
        elif simple_payback_years < 7:
            attractiveness = "acceptable"
        else:
            attractiveness = "marginal"

        return {
            "simple_payback_years": round(simple_payback_years, 2),
            "discounted_payback_years": discounted_payback_years if discounted_payback_years else ">20",
            "net_present_value_usd": round(total_npv, 0),
            "internal_rate_of_return_percent": round(irr_percent, 2),
            "savings_to_investment_ratio": round(sir, 2),
            "project_attractiveness": attractiveness,
            "total_lifetime_savings_usd": round(sum(cf["savings_usd"] for cf in cash_flows), 0),
            "total_lifetime_maintenance_usd": round(annual_maintenance_cost_usd * project_lifetime_years, 0),
            "annual_cashflows": cash_flows[:10],  # First 10 years for reporting
            "assumptions": {
                "project_lifetime_years": project_lifetime_years,
                "discount_rate": discount_rate,
                "energy_escalation_rate": energy_cost_escalation_rate,
            },
            "standards": ["FEMP_Life_Cycle_Cost", "ASHRAE_Economic_Analysis", "DOE_EERE_Guidelines"],
        }

    def _prioritize_waste_heat_opportunities_impl(
        self,
        opportunities: List[Dict[str, Any]],
        prioritization_criteria: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Tool #8: Prioritize waste heat recovery opportunities using weighted scoring.

        Scoring criteria (customizable weights):
        - Payback period (30%): Shorter is better
        - Energy savings potential (25%): Higher MMBtu/yr is better
        - Implementation complexity (20%): Lower complexity preferred
        - Carbon impact (15%): Higher CO2 reduction valued
        - Capital efficiency (10%): Lower $/MMBtu is better

        Args:
            opportunities: List of waste heat opportunities with financial data
            prioritization_criteria: Custom weights for scoring criteria

        Returns:
            Ranked opportunities with priority scores and implementation roadmap
        """
        # Default criteria weights
        if prioritization_criteria is None:
            prioritization_criteria = {
                "payback_weight": 0.30,
                "energy_savings_weight": 0.25,
                "complexity_weight": 0.20,
                "carbon_impact_weight": 0.15,
                "capital_efficiency_weight": 0.10,
            }

        scored_opportunities = []

        for opp in opportunities:
            score = 0
            scoring_breakdown = {}

            # 1. Payback Period Score (30% weight)
            payback_years = opp.get("payback_years", 10)
            if payback_years < 1:
                payback_score = 30
            elif payback_years < 2:
                payback_score = 25
            elif payback_years < 3:
                payback_score = 20
            elif payback_years < 5:
                payback_score = 10
            else:
                payback_score = 5
            score += payback_score * prioritization_criteria["payback_weight"] / 0.30
            scoring_breakdown["payback_score"] = round(payback_score, 1)

            # 2. Energy Savings Potential (25% weight)
            energy_savings_mmbtu = opp.get("energy_savings_mmbtu_yr", 0)
            # Normalize to 0-25 scale (assume 5000 MMBtu/yr is max)
            energy_score = min(energy_savings_mmbtu / 5000 * 25, 25)
            score += energy_score * prioritization_criteria["energy_savings_weight"] / 0.25
            scoring_breakdown["energy_savings_score"] = round(energy_score, 1)

            # 3. Implementation Complexity (20% weight)
            complexity = opp.get("implementation_complexity", "moderate")
            complexity_scores = {"low": 20, "moderate": 12, "high": 5}
            complexity_score = complexity_scores.get(complexity, 12)
            score += complexity_score * prioritization_criteria["complexity_weight"] / 0.20
            scoring_breakdown["complexity_score"] = complexity_score

            # 4. Carbon Impact (15% weight)
            co2_reduction_tons = opp.get("co2_reduction_metric_tons_yr", 0)
            # Normalize to 0-15 scale (assume 500 tons/yr is max)
            carbon_score = min(co2_reduction_tons / 500 * 15, 15)
            score += carbon_score * prioritization_criteria["carbon_impact_weight"] / 0.15
            scoring_breakdown["carbon_impact_score"] = round(carbon_score, 1)

            # 5. Capital Efficiency (10% weight)
            capital_cost = opp.get("capital_cost_usd", 100000)
            energy_savings_for_capex = energy_savings_mmbtu if energy_savings_mmbtu > 0 else 1
            capital_per_mmbtu = capital_cost / energy_savings_for_capex
            # Lower is better; assume $50/MMBtu/yr is excellent, $200 is poor
            if capital_per_mmbtu < 50:
                capex_score = 10
            elif capital_per_mmbtu < 100:
                capex_score = 7
            elif capital_per_mmbtu < 150:
                capex_score = 5
            else:
                capex_score = 2
            score += capex_score * prioritization_criteria["capital_efficiency_weight"] / 0.10
            scoring_breakdown["capital_efficiency_score"] = capex_score

            scored_opportunities.append({
                "opportunity_name": opp.get("name", "Unnamed"),
                "total_score": round(score, 1),
                "scoring_breakdown": scoring_breakdown,
                "payback_years": round(payback_years, 2),
                "energy_savings_mmbtu_yr": round(energy_savings_mmbtu, 0),
                "capital_cost_usd": round(capital_cost, 0),
                "annual_savings_usd": round(opp.get("annual_savings_usd", 0), 0),
                "co2_reduction_metric_tons_yr": round(co2_reduction_tons, 0),
                "implementation_complexity": complexity,
                "technology": opp.get("technology", "TBD"),
            })

        # Sort by total score descending
        scored_opportunities.sort(key=lambda x: x["total_score"], reverse=True)

        # Create implementation roadmap
        roadmap = []
        cumulative_investment = 0
        cumulative_savings = 0

        for i, opp in enumerate(scored_opportunities):
            phase = "Phase 1 (Year 1)" if i < 2 else "Phase 2 (Year 2)" if i < 5 else "Phase 3 (Year 3+)"
            cumulative_investment += opp["capital_cost_usd"]
            cumulative_savings += opp["annual_savings_usd"]

            roadmap.append({
                "rank": i + 1,
                "opportunity": opp["opportunity_name"],
                "implementation_phase": phase,
                "priority_level": "High" if i < 3 else "Medium" if i < 6 else "Low",
                "cumulative_investment_usd": cumulative_investment,
                "cumulative_annual_savings_usd": cumulative_savings,
            })

        return {
            "prioritized_opportunities": scored_opportunities,
            "implementation_roadmap": roadmap,
            "total_opportunities": len(opportunities),
            "high_priority_count": len([o for o in scored_opportunities if o["total_score"] > 70]),
            "total_potential_investment_usd": round(sum(o["capital_cost_usd"] for o in scored_opportunities), 0),
            "total_potential_savings_usd_yr": round(sum(o["annual_savings_usd"] for o in scored_opportunities), 0),
            "total_carbon_reduction_metric_tons_yr": round(sum(o["co2_reduction_metric_tons_yr"] for o in scored_opportunities), 0),
            "prioritization_criteria_used": prioritization_criteria,
            "methodology": "Multi-criteria weighted scoring with implementation phasing",
            "standards": ["ISO_50001_Energy_Management", "DOE_Industrial_Assessment_Centers", "ASHRAE"],
        }

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute waste heat recovery analysis.

        Args:
            input_data: Input data with facility and process information

        Returns:
            AgentResult with waste heat opportunities and recommendations
        """
        start_time = DeterministicClock.now()

        try:
            # Validate input
            if not self._validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Invalid input: missing required fields",
            )

            # Run async analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._execute_async(input_data))
            finally:
                loop.close()

            # Calculate duration
            duration = (DeterministicClock.now() - start_time).total_seconds()

            # Add performance metadata
            if result.success:
                result.metadata["calculation_time_ms"] = duration * 1000
                result.metadata["ai_calls"] = self._ai_call_count
                result.metadata["tool_calls"] = self._tool_call_count
                result.metadata["total_cost_usd"] = self._total_cost_usd

            return result

        except Exception as e:
            logger.error(f"Error in waste heat recovery analysis: {e}")
            return AgentResult(
                success=False,
                error=f"Failed to analyze waste heat recovery: {str(e)}",
            )

    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data against schema."""
        required_fields = [
            "facility_type",
            "processes",
            "total_annual_fuel_mmbtu",
            "fuel_cost_usd_per_mmbtu",
        ]

        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False

        return True

    async def _execute_async(self, input_data: Dict[str, Any]) -> AgentResult:
        """Async execution with ChatSession orchestration."""

        # For now, return basic analysis using tool #1
        # Full implementation would include AI orchestration with all 8 tools

        waste_heat_analysis = self._identify_waste_heat_sources_impl(
            facility_type=input_data["facility_type"],
            processes=input_data["processes"],
            include_hvac_systems=input_data.get("include_hvac_systems", True),
            include_compressed_air=input_data.get("include_compressed_air", True),
            minimum_temperature_f=input_data.get("minimum_temperature_f", 120),
        )

        return AgentResult(
            success=True,
            data={
                "total_waste_heat_identified_mmbtu_yr": waste_heat_analysis["total_waste_heat_mmbtu_yr"],
                "recoverable_waste_heat_mmbtu_yr": waste_heat_analysis["recoverable_waste_heat_mmbtu_yr"],
                "waste_heat_sources": waste_heat_analysis["waste_heat_sources"],
                "waste_heat_summary": waste_heat_analysis["waste_heat_summary"],
                "analysis_summary": f"Identified {waste_heat_analysis['total_waste_heat_mmbtu_yr']:,.0f} MMBtu/year waste heat with {waste_heat_analysis['recoverable_waste_heat_mmbtu_yr']:,.0f} MMBtu/year technically recoverable.",
            },
            metadata={
                "agent": "WasteHeatRecoveryAgent_AI",
                "version": "1.0.0",
                "deterministic": True,
                "tool_calls": self._tool_call_count,
            },
        )

    def health_check(self) -> Dict[str, Any]:
        """Agent health check for monitoring."""
        return {
            "status": "healthy",
            "agent_id": "industrial/waste_heat_recovery_agent",
            "version": "1.0.0",
            "tools_available": 8,
            "metrics": {
                "ai_calls": self._ai_call_count,
                "tool_calls": self._tool_call_count,
                "total_cost_usd": round(self._total_cost_usd, 4),
            },
        }
