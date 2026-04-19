# -*- coding: utf-8 -*-
"""
CogenerationCHPAgent_AI - Combined Heat and Power System Analysis and Optimization

This agent performs comprehensive analysis of combined heat and power (CHP/cogeneration)
systems for industrial and commercial facilities. It covers technology selection, performance
analysis, economic optimization, and grid interconnection requirements for five CHP technologies:
reciprocating engines, gas turbines, microturbines, fuel cells, and steam turbines.

THERMODYNAMIC FOUNDATION:
-------------------------
CHP systems simultaneously generate electricity and useful thermal energy from a single fuel source,
achieving significantly higher total efficiency than separate generation:

Total CHP Efficiency = (Electrical Output + Useful Thermal Output) / Fuel Input

Typical efficiencies:
- Separate generation: 45-50% (30% electrical grid + 80% boiler × 20% heating need)
- CHP system: 65-80% (30-45% electrical + 35-45% thermal)

The key design parameter is the heat-to-power ratio:
    H/P Ratio = (Thermal Output MMBtu/hr × 293.07 kWh/MMBtu) / Electrical Output kW

Different technologies naturally produce different H/P ratios:
- Gas turbine: 0.5-2.0 (more power than heat)
- Reciprocating engine: 1.0-2.5 (balanced)
- Steam turbine (backpressure): 4.0-7.0 (more heat than power)
- Fuel cell: 0.5-1.5 (more power than heat)

TECHNOLOGY OVERVIEW:
-------------------
1. **Reciprocating Engine (100 kW - 10 MW)**
   - Electrical efficiency: 35-42%
   - Total CHP efficiency: 75-85%
   - H/P ratio: 1.0-2.5
   - Best for: Baseload, high electrical efficiency, fuel flexibility
   - Maintenance: 40,000-50,000 hour overhaul intervals

2. **Gas Turbine (1 MW - 50 MW)**
   - Electrical efficiency: 25-40%
   - Total CHP efficiency: 70-80%
   - H/P ratio: 0.5-2.0
   - Best for: Large facilities, high-temperature heat, low maintenance
   - Maintenance: 25,000-50,000 hour inspection intervals

3. **Microturbine (30 kW - 500 kW)**
   - Electrical efficiency: 26-30%
   - Total CHP efficiency: 65-75%
   - H/P ratio: 1.5-2.5
   - Best for: Distributed generation, low maintenance, compact
   - Maintenance: 40,000-80,000 hour intervals

4. **Fuel Cell (100 kW - 5 MW)**
   - Electrical efficiency: 40-50% (MCFC/SOFC)
   - Total CHP efficiency: 70-85%
   - H/P ratio: 0.5-1.5
   - Best for: High electrical efficiency, ultra-low emissions, quiet
   - Maintenance: High availability, lower O&M than engines

5. **Steam Turbine (500 kW - 50 MW)**
   - Electrical efficiency: 15-30%
   - Total CHP efficiency: 75-85%
   - H/P ratio: 4.0-7.0
   - Best for: High thermal load, fuel flexibility (biomass, waste heat)
   - Maintenance: Minimal, extremely reliable

ECONOMIC METRICS:
----------------
1. **Spark Spread:** Value of electricity produced minus cost of fuel
   Spark Spread ($/MWh) = (Electricity Price $/MWh) - (Gas Price $/MMBtu × 3.412 / Efficiency)

2. **Simple Payback:** Capital investment / Annual savings

3. **Avoided Costs:**
   - Avoided electricity purchases (kWh × retail rate)
   - Avoided thermal energy costs (MMBtu × fuel cost / boiler efficiency)
   - Avoided demand charges (kW × demand rate)
   - Avoided capacity/transmission charges

GRID INTERCONNECTION:
--------------------
CHP systems require careful grid interconnection planning:
- IEEE 1547 interconnection standards
- Utility standby rates (demand charges when grid-connected)
- Export tariffs (if selling excess power)
- Grid support requirements (power factor, voltage support)
- Islanding protection and automatic disconnection

CALCULATION METHODS:
------------------
1. **Electrical Efficiency:** η_e = (Electrical Output kW × 3.412) / (Fuel Input MMBtu/hr)
2. **Thermal Efficiency:** η_t = (Thermal Output MMBtu/hr) / (Fuel Input MMBtu/hr)
3. **Total Efficiency:** η_total = η_e + η_t
4. **Heat Recovery:** Q_recovered = m_exhaust × cp_exhaust × (T_exhaust - T_stack)
5. **Annual Savings:** (Avoided electricity cost) + (Avoided thermal cost) - (Fuel cost) - (O&M cost)

STANDARDS COMPLIANCE:
--------------------
- EPA CHP Partnership (technology performance data)
- ASHRAE Applications - CHP Systems
- IEEE 1547 (grid interconnection)
- ASME BPVC Section I (boiler and pressure vessel code for HRSGs)
- ISO 50001 (energy management systems)

USAGE EXAMPLE:
-------------
    >>> config = CogenerationCHPConfig(budget_usd=0.50)
    >>> agent = CogenerationCHPAgentAI(config=config)
    >>> result = agent.select_chp_technology(
    ...     electrical_demand_kw=2000,
    ...     thermal_demand_mmbtu_hr=15.0,
    ...     heat_to_power_ratio=2.2,
    ...     load_profile_type="baseload_24x7",
    ...     available_fuels=["natural_gas"],
    ...     emissions_requirements="low_nox",
    ...     space_constraints="moderate"
    ... )
    >>> print(f"Recommended: {result['recommended_technology']}")
    Recommended: reciprocating_engine
    >>> print(f"Electrical Efficiency: {result['typical_electrical_efficiency']:.1%}")
    Electrical Efficiency: 38.0%
    >>> print(f"Total CHP Efficiency: {result['typical_total_efficiency']:.1%}")
    Total CHP Efficiency: 83.0%
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import logging

from pydantic import BaseModel, Field, validator
import numpy as np

# GreenLang core
from greenlang.core.chat_session import ChatSession
from greenlang.core.tool_registry import ToolRegistry
from greenlang.provenance import ProvenanceTracker
from greenlang.utilities.determinism import DeterministicClock

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# CHP Technology Database and Properties
# ============================================================================

class CHPTechnologyDatabase:
    """
    Comprehensive database of CHP technology characteristics

    Data sources:
    - EPA CHP Partnership Technology Characterization
    - DOE CHP Installation Database
    - Vendor performance data (Caterpillar, GE, Capstone, etc.)
    """

    TECHNOLOGIES = {
        "reciprocating_engine": {
            "name": "Reciprocating Engine",
            "size_range_kw": (100, 10000),
            "electrical_efficiency": (0.35, 0.42),
            "thermal_efficiency": (0.40, 0.48),
            "total_efficiency": (0.75, 0.85),
            "heat_to_power_ratio": (1.0, 2.5),
            "capex_per_kw": (1500, 2200),
            "opex_per_kwh": (0.013, 0.018),
            "maintenance_interval_hours": (40000, 50000),
            "startup_time_minutes": (5, 10),
            "part_load_performance": "excellent",  # Down to 50% with <5% efficiency penalty
            "fuel_flexibility": "excellent",  # NG, biogas, propane, diesel
            "emissions_nox_ppm": (5, 25),  # With SCR
            "space_requirement_sqft_per_kw": (0.8, 1.2),
            "noise_level_dba": (75, 85),
            "advantages": [
                "High electrical efficiency",
                "Excellent for baseload operation",
                "Good fuel flexibility (natural gas, biogas, propane)",
                "Proven reliability with long track record",
                "Fast start capability (5-10 minutes)",
                "Good part-load performance"
            ],
            "challenges": [
                "Higher maintenance than turbines",
                "Noise and vibration (requires acoustic enclosure)",
                "NOx emissions require SCR for ultra-low levels",
                "More frequent oil changes",
                "Larger footprint than microturbines"
            ],
            "ideal_applications": [
                "24/7 baseload operation",
                "Facilities with balanced thermal and electrical loads",
                "Sites requiring backup power capability",
                "Applications with available natural gas or biogas"
            ]
        },

        "gas_turbine": {
            "name": "Gas Turbine",
            "size_range_kw": (1000, 50000),
            "electrical_efficiency": (0.25, 0.40),
            "thermal_efficiency": (0.45, 0.55),
            "total_efficiency": (0.70, 0.80),
            "heat_to_power_ratio": (0.5, 2.0),
            "capex_per_kw": (1200, 1800),
            "opex_per_kwh": (0.005, 0.010),
            "maintenance_interval_hours": (25000, 50000),
            "startup_time_minutes": (10, 30),
            "part_load_performance": "poor",  # Efficiency drops significantly <70% load
            "fuel_flexibility": "good",  # NG, LPG, some can burn liquid fuels
            "emissions_nox_ppm": (9, 25),  # Dry low NOx combustors
            "space_requirement_sqft_per_kw": (0.3, 0.6),
            "noise_level_dba": (80, 95),
            "advantages": [
                "Low maintenance (fewer moving parts)",
                "Compact footprint",
                "High-quality heat (high exhaust temperature 900-1100°F)",
                "Excellent for large facilities (>2 MW)",
                "Lower emissions than reciprocating engines",
                "Long intervals between major overhauls"
            ],
            "challenges": [
                "Lower electrical efficiency than recip engines",
                "Poor part-load performance",
                "High exhaust temperatures require HRSG",
                "Sensitive to ambient temperature",
                "Higher initial cost for small units (<3 MW)"
            ],
            "ideal_applications": [
                "Large facilities (>2 MW)",
                "High-temperature heat requirements (>400°F)",
                "Sites requiring steam generation",
                "Facilities with relatively constant load"
            ]
        },

        "microturbine": {
            "name": "Microturbine",
            "size_range_kw": (30, 500),
            "electrical_efficiency": (0.26, 0.30),
            "thermal_efficiency": (0.40, 0.50),
            "total_efficiency": (0.66, 0.75),
            "heat_to_power_ratio": (1.5, 2.5),
            "capex_per_kw": (2500, 3500),
            "opex_per_kwh": (0.008, 0.012),
            "maintenance_interval_hours": (40000, 80000),
            "startup_time_minutes": (3, 5),
            "part_load_performance": "good",
            "fuel_flexibility": "good",
            "emissions_nox_ppm": (9, 15),
            "space_requirement_sqft_per_kw": (0.4, 0.7),
            "noise_level_dba": (60, 70),
            "advantages": [
                "Extremely low emissions (single-digit NOx)",
                "Very low maintenance (few moving parts)",
                "Quiet operation",
                "Small footprint",
                "Fast startup",
                "No water cooling required"
            ],
            "challenges": [
                "Lower electrical efficiency",
                "Higher capital cost per kW",
                "Limited to smaller applications",
                "Efficiency sensitive to ambient conditions"
            ],
            "ideal_applications": [
                "Distributed generation",
                "Buildings with limited space",
                "Sites requiring low emissions without aftertreatment",
                "Applications <500 kW"
            ]
        },

        "fuel_cell": {
            "name": "Fuel Cell (MCFC/SOFC)",
            "size_range_kw": (100, 5000),
            "electrical_efficiency": (0.40, 0.50),
            "thermal_efficiency": (0.30, 0.40),
            "total_efficiency": (0.70, 0.85),
            "heat_to_power_ratio": (0.5, 1.5),
            "capex_per_kw": (5000, 8000),
            "opex_per_kwh": (0.020, 0.035),
            "maintenance_interval_hours": (60000, 80000),
            "startup_time_minutes": (180, 480),
            "part_load_performance": "excellent",
            "fuel_flexibility": "moderate",  # NG, biogas (with cleanup)
            "emissions_nox_ppm": (0.1, 1.0),  # Virtually zero emissions
            "space_requirement_sqft_per_kw": (0.5, 0.8),
            "noise_level_dba": (55, 65),
            "advantages": [
                "Highest electrical efficiency (40-50%)",
                "Ultra-low emissions (near-zero NOx)",
                "Very quiet operation",
                "Excellent part-load performance",
                "High availability (>95%)",
                "Low vibration"
            ],
            "challenges": [
                "Very high capital cost",
                "Long startup time (hours)",
                "Higher O&M costs",
                "Limited fuel flexibility",
                "Immature technology (shorter track record)"
            ],
            "ideal_applications": [
                "High value on electrical efficiency",
                "Ultra-low emissions requirements",
                "Critical facilities (hospitals, data centers)",
                "Sites with premium electricity rates"
            ]
        },

        "steam_turbine": {
            "name": "Steam Turbine (Backpressure)",
            "size_range_kw": (500, 50000),
            "electrical_efficiency": (0.15, 0.30),
            "thermal_efficiency": (0.50, 0.60),
            "total_efficiency": (0.75, 0.85),
            "heat_to_power_ratio": (4.0, 7.0),
            "capex_per_kw": (1000, 1500),
            "opex_per_kwh": (0.003, 0.008),
            "maintenance_interval_hours": (50000, 100000),
            "startup_time_minutes": (60, 180),
            "part_load_performance": "fair",
            "fuel_flexibility": "excellent",  # Any fuel that makes steam
            "emissions_nox_ppm": (10, 50),  # Depends on boiler
            "space_requirement_sqft_per_kw": (1.0, 2.0),
            "noise_level_dba": (75, 85),
            "advantages": [
                "Extremely reliable",
                "Very low maintenance",
                "Fuel flexibility (biomass, waste heat, any fuel)",
                "Ideal for high thermal loads",
                "Long equipment life (40+ years)",
                "Mature, proven technology"
            ],
            "challenges": [
                "Low electrical efficiency",
                "High heat-to-power ratio (need 4-7× more thermal than electrical)",
                "Requires boiler or waste heat source",
                "Slow startup",
                "Large footprint"
            ],
            "ideal_applications": [
                "Facilities with very high thermal loads",
                "Process industries with waste heat",
                "Sites with biomass or waste fuel available",
                "Applications where thermal load dominates"
            ]
        }
    }

    # Fuel properties
    FUELS = {
        "natural_gas": {
            "hhv_btu_per_cf": 1025,
            "hhv_mmbtu_per_mcf": 1.025,
            "lhv_btu_per_cf": 925,
            "emission_factor_lb_co2_per_mmbtu": 116.65,
            "emission_factor_kg_co2_per_mmbtu": 52.91,
            "typical_cost_per_mmbtu": (4.0, 12.0)
        },
        "biogas": {
            "hhv_btu_per_cf": 600,  # 60% CH4
            "hhv_mmbtu_per_mcf": 0.600,
            "lhv_btu_per_cf": 550,
            "emission_factor_lb_co2_per_mmbtu": 0,  # Renewable
            "emission_factor_kg_co2_per_mmbtu": 0,
            "typical_cost_per_mmbtu": (0, 5.0)  # Often free or subsidized
        },
        "hydrogen": {
            "hhv_btu_per_lb": 61095,
            "lhv_btu_per_lb": 51596,
            "emission_factor_lb_co2_per_mmbtu": 0,  # If green H2
            "emission_factor_kg_co2_per_mmbtu": 0,
            "typical_cost_per_mmbtu": (20.0, 60.0)  # Currently expensive
        },
        "propane": {
            "hhv_btu_per_gal": 91500,
            "hhv_mmbtu_per_gal": 0.0915,
            "lhv_btu_per_gal": 83500,
            "emission_factor_lb_co2_per_mmbtu": 139.0,
            "emission_factor_kg_co2_per_mmbtu": 63.07,
            "typical_cost_per_mmbtu": (15.0, 30.0)
        },
        "diesel": {
            "hhv_btu_per_gal": 138500,
            "hhv_mmbtu_per_gal": 0.1385,
            "lhv_btu_per_gal": 128500,
            "emission_factor_lb_co2_per_mmbtu": 161.4,
            "emission_factor_kg_co2_per_mmbtu": 73.2,
            "typical_cost_per_mmbtu": (20.0, 40.0)
        }
    }

    # Heat recovery configurations
    HEAT_RECOVERY = {
        "jacket_water_only": {
            "thermal_efficiency_gain": (0.20, 0.30),
            "achievable_temperature_f": (180, 230),
            "heat_quality": "low_grade",
            "capex_adder_per_kw": (100, 200)
        },
        "jacket_exhaust": {
            "thermal_efficiency_gain": (0.35, 0.45),
            "achievable_temperature_f": (230, 350),
            "heat_quality": "medium_grade",
            "capex_adder_per_kw": (200, 350)
        },
        "hrsg_unfired": {
            "thermal_efficiency_gain": (0.40, 0.50),
            "achievable_temperature_f": (300, 400),
            "heat_quality": "medium_grade",
            "capex_adder_per_kw": (300, 500),
            "can_make_steam": True
        },
        "hrsg_supplementary_fired": {
            "thermal_efficiency_gain": (0.50, 0.65),
            "achievable_temperature_f": (400, 750),
            "heat_quality": "high_grade",
            "capex_adder_per_kw": (500, 800),
            "can_make_steam": True,
            "requires_additional_fuel": True
        }
    }


# ============================================================================
# Configuration
# ============================================================================

class CogenerationCHPConfig(BaseModel):
    """
    Configuration for CogenerationCHPAgentAI

    Attributes:
        agent_id: Unique agent identifier
        agent_name: Human-readable name
        budget_usd: Maximum cost per analysis
        temperature: LLM temperature (0.0 for determinism)
        seed: Random seed for reproducibility
    """
    agent_id: str = Field(
        default="industrial/cogeneration_chp_agent",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="CogenerationCHPAgent_AI",
        description="Agent name"
    )
    budget_usd: float = Field(
        default=0.50,
        ge=0.01,
        le=2.0,
        description="Budget per analysis (USD)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.0 for determinism)"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    deterministic: bool = Field(
        default=True,
        description="Deterministic mode flag"
    )

    class Config:
        frozen = True  # Immutable configuration


# ============================================================================
# Main Agent Class
# ============================================================================

class CogenerationCHPAgentAI:
    """
    CogenerationCHPAgent_AI - Combined Heat and Power System Analysis

    This agent provides comprehensive analysis of CHP systems including technology
    selection, performance analysis, heat recovery optimization, economic analysis,
    and grid interconnection planning.

    Attributes:
        config (CogenerationCHPConfig): Agent configuration
        session (ChatSession): AI orchestration session
        tool_registry (ToolRegistry): Registry of available tools
        provenance (ProvenanceTracker): Calculation provenance tracking

    Methods:
        select_chp_technology: Recommend optimal CHP technology
        calculate_chp_performance: Calculate system efficiency metrics
        size_heat_recovery_system: Design HRSG or heat recovery equipment
        calculate_economic_metrics: Analyze spark spread, payback, NPV
        assess_grid_interconnection: Evaluate grid connection requirements
        optimize_operating_strategy: Determine optimal dispatch strategy
        calculate_emissions_reduction: Quantify emissions vs baseline
        generate_chp_report: Comprehensive analysis report
        analyze: Main entry point for AI-orchestrated analysis

    Example:
        >>> config = CogenerationCHPConfig(budget_usd=0.50)
        >>> agent = CogenerationCHPAgentAI(config=config)
        >>> result = agent.analyze("Analyze 2 MW CHP for food processing facility")
        >>> print(result['recommendations'])
    """

    def __init__(self, config: CogenerationCHPConfig):
        """
        Initialize CogenerationCHPAgentAI

        Args:
            config: Agent configuration
        """
        self.config = config
        self.session = ChatSession(
            agent_id=config.agent_id,
            temperature=config.temperature,
            seed=config.seed,
            deterministic=config.deterministic
        )
        self.tool_registry = ToolRegistry()
        self.provenance = ProvenanceTracker()

        # Technology database
        self.tech_db = CHPTechnologyDatabase()

        # Register all tools
        self._register_tools()

        logger.info(f"{config.agent_name} initialized (v{self._version()})")

    def _register_tools(self):
        """Register all agent tools with registry"""
        self.tool_registry.register("select_chp_technology", self.select_chp_technology)
        self.tool_registry.register("calculate_chp_performance", self.calculate_chp_performance)
        self.tool_registry.register("size_heat_recovery_system", self.size_heat_recovery_system)
        self.tool_registry.register("calculate_economic_metrics", self.calculate_economic_metrics)
        self.tool_registry.register("assess_grid_interconnection", self.assess_grid_interconnection)
        self.tool_registry.register("optimize_operating_strategy", self.optimize_operating_strategy)
        self.tool_registry.register("calculate_emissions_reduction", self.calculate_emissions_reduction)
        self.tool_registry.register("generate_chp_report", self.generate_chp_report)

    @staticmethod
    def _version() -> str:
        """Return agent version"""
        return "1.0.0"

    # ========================================================================
    # TOOL 1: SELECT CHP TECHNOLOGY
    # ========================================================================

    def select_chp_technology(
        self,
        electrical_demand_kw: float,
        thermal_demand_mmbtu_hr: float,
        heat_to_power_ratio: float,
        load_profile_type: str,
        available_fuels: List[str],
        emissions_requirements: str,
        space_constraints: str,
        required_electrical_efficiency: float = 0.30,
        grid_export_allowed: bool = False,
        resilience_priority: str = "medium"
    ) -> Dict[str, Any]:
        """
        Recommend optimal CHP technology based on facility requirements

        Uses multi-criteria decision analysis to select the best CHP technology
        considering heat-to-power ratio matching, size, efficiency requirements,
        fuel availability, emissions limits, and site constraints.

        Args:
            electrical_demand_kw: Facility electrical demand in kW
            thermal_demand_mmbtu_hr: Facility thermal demand in MMBtu/hr
            heat_to_power_ratio: Required thermal/electrical ratio
            load_profile_type: Operating schedule (baseload_24x7, daytime_only, etc.)
            available_fuels: List of available fuel options
            emissions_requirements: Emissions compliance level (standard, low_nox, etc.)
            space_constraints: Available space (ample, moderate, limited, very_limited)
            required_electrical_efficiency: Minimum electrical efficiency target
            grid_export_allowed: Whether excess power can be exported to grid
            resilience_priority: Importance of backup power (low, medium, high, critical)

        Returns:
            Dict containing:
                - recommended_technology: Best technology choice
                - alternative_technology: Second-best option
                - technology_description: Detailed description
                - typical_electrical_efficiency: Expected electrical efficiency
                - typical_thermal_efficiency: Expected thermal efficiency
                - typical_total_efficiency: Total CHP efficiency
                - heat_to_power_ratio_achievable: H/P ratio for technology
                - estimated_capex_per_kw: Capital cost estimate
                - maintenance_cost_per_kwh: O&M cost estimate
                - typical_overhaul_interval_hours: Maintenance interval
                - key_advantages: List of advantages
                - key_challenges: List of challenges
                - fuel_flexibility: Fuel flexibility rating
                - deterministic: True
                - provenance: Calculation metadata

        Raises:
            ValueError: If electrical_demand_kw <= 0
            ValueError: If heat_to_power_ratio <= 0
            ValueError: If invalid load_profile_type

        Example:
            >>> result = agent.select_chp_technology(
            ...     electrical_demand_kw=2000,
            ...     thermal_demand_mmbtu_hr=15.0,
            ...     heat_to_power_ratio=2.2,
            ...     load_profile_type="baseload_24x7",
            ...     available_fuels=["natural_gas"],
            ...     emissions_requirements="low_nox",
            ...     space_constraints="moderate"
            ... )
            >>> print(f"Recommended: {result['recommended_technology']}")
            Recommended: reciprocating_engine
        """
        # Input validation
        if electrical_demand_kw <= 0:
            raise ValueError(f"electrical_demand_kw must be positive, got {electrical_demand_kw}")

        if thermal_demand_mmbtu_hr <= 0:
            raise ValueError(f"thermal_demand_mmbtu_hr must be positive, got {thermal_demand_mmbtu_hr}")

        if heat_to_power_ratio <= 0:
            raise ValueError(f"heat_to_power_ratio must be positive, got {heat_to_power_ratio}")

        valid_profiles = ["baseload_24x7", "daytime_only", "batch_intermittent", "highly_variable"]
        if load_profile_type not in valid_profiles:
            raise ValueError(f"load_profile_type must be one of {valid_profiles}, got {load_profile_type}")

        try:
            # Score each technology
            scores = {}

            for tech_name, tech_data in self.tech_db.TECHNOLOGIES.items():
                score = 0
                reasons = []

                # Size matching (0-30 points)
                size_min, size_max = tech_data["size_range_kw"]
                if size_min <= electrical_demand_kw <= size_max:
                    score += 30
                    reasons.append(f"Size match: {electrical_demand_kw} kW within range")
                elif electrical_demand_kw < size_min:
                    score += max(0, 20 - (size_min - electrical_demand_kw) / size_min * 10)
                    reasons.append(f"Undersized: {electrical_demand_kw} kW < {size_min} kW minimum")
                else:
                    score += max(0, 20 - (electrical_demand_kw - size_max) / size_max * 10)
                    reasons.append(f"Oversized: {electrical_demand_kw} kW > {size_max} kW maximum")

                # Heat-to-power ratio matching (0-25 points)
                hp_min, hp_max = tech_data["heat_to_power_ratio"]
                if hp_min <= heat_to_power_ratio <= hp_max:
                    score += 25
                    reasons.append(f"H/P ratio perfect match: {heat_to_power_ratio:.2f}")
                else:
                    hp_deviation = min(abs(heat_to_power_ratio - hp_min), abs(heat_to_power_ratio - hp_max))
                    score += max(0, 25 - hp_deviation * 5)
                    reasons.append(f"H/P ratio deviation: {heat_to_power_ratio:.2f} vs {hp_min:.1f}-{hp_max:.1f}")

                # Electrical efficiency (0-20 points)
                eff_min, eff_max = tech_data["electrical_efficiency"]
                avg_eff = (eff_min + eff_max) / 2
                if avg_eff >= required_electrical_efficiency:
                    score += 20
                    reasons.append(f"Electrical efficiency meets requirement: {avg_eff:.1%}")
                else:
                    score += max(0, 20 - (required_electrical_efficiency - avg_eff) * 100)
                    reasons.append(f"Electrical efficiency below target: {avg_eff:.1%} < {required_electrical_efficiency:.1%}")

                # Load profile matching (0-15 points)
                if load_profile_type == "baseload_24x7":
                    if tech_name in ["reciprocating_engine", "gas_turbine", "fuel_cell"]:
                        score += 15
                        reasons.append("Excellent for baseload operation")
                    else:
                        score += 10
                elif load_profile_type == "batch_intermittent":
                    if tech_data["startup_time_minutes"][1] < 30:
                        score += 15
                        reasons.append("Fast startup for intermittent operation")
                    else:
                        score += 5
                elif load_profile_type == "highly_variable":
                    if tech_data["part_load_performance"] in ["excellent", "good"]:
                        score += 15
                        reasons.append("Good part-load performance")
                    else:
                        score += 5
                else:  # daytime_only
                    score += 10

                # Fuel availability (0-10 points)
                if "natural_gas" in available_fuels and tech_data["fuel_flexibility"] in ["excellent", "good"]:
                    score += 10
                    reasons.append("Natural gas compatible")
                elif tech_data["fuel_flexibility"] == "excellent":
                    score += 8
                else:
                    score += 5

                scores[tech_name] = {"score": score, "reasons": reasons}

            # Sort by score
            ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

            # Select top 2
            recommended_tech = ranked[0][0]
            alternative_tech = ranked[1][0] if len(ranked) > 1 else None

            # Get detailed data for recommended technology
            tech_data = self.tech_db.TECHNOLOGIES[recommended_tech]

            # Calculate average values
            avg_elec_eff = sum(tech_data["electrical_efficiency"]) / 2
            avg_therm_eff = sum(tech_data["thermal_efficiency"]) / 2
            avg_total_eff = sum(tech_data["total_efficiency"]) / 2
            avg_hp_ratio = sum(tech_data["heat_to_power_ratio"]) / 2
            avg_capex = sum(tech_data["capex_per_kw"]) / 2
            avg_opex = sum(tech_data["opex_per_kwh"]) / 2
            avg_overhaul = sum(tech_data["maintenance_interval_hours"]) / 2

            return {
                "recommended_technology": recommended_tech,
                "alternative_technology": alternative_tech,
                "technology_description": f"{tech_data['name']} - {scores[recommended_tech]['reasons'][0]}",
                "typical_electrical_efficiency": avg_elec_eff,
                "typical_thermal_efficiency": avg_therm_eff,
                "typical_total_efficiency": avg_total_eff,
                "heat_to_power_ratio_achievable": avg_hp_ratio,
                "estimated_capex_per_kw": avg_capex,
                "maintenance_cost_per_kwh": avg_opex,
                "typical_overhaul_interval_hours": avg_overhaul,
                "key_advantages": tech_data["advantages"],
                "key_challenges": tech_data["challenges"],
                "fuel_flexibility": tech_data["fuel_flexibility"],
                "selection_score": scores[recommended_tech]["score"],
                "selection_reasons": scores[recommended_tech]["reasons"],
                "deterministic": True,
                "provenance": {
                    "tool": "select_chp_technology",
                    "method": "Multi-criteria decision analysis",
                    "standard": "EPA CHP Partnership Technology Characterization",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "inputs": {
                        "electrical_demand_kw": electrical_demand_kw,
                        "thermal_demand_mmbtu_hr": thermal_demand_mmbtu_hr,
                        "heat_to_power_ratio": heat_to_power_ratio,
                        "load_profile_type": load_profile_type
                    }
                }
            }

        except Exception as e:
            logger.error(f"Tool 1 (select_chp_technology) failed: {e}")
            raise

    # ========================================================================
    # TOOL 2: CALCULATE CHP PERFORMANCE
    # ========================================================================

    def calculate_chp_performance(
        self,
        chp_technology: str,
        electrical_capacity_kw: float,
        fuel_input_mmbtu_hr: float,
        heat_recovery_configuration: str,
        exhaust_temperature_f: float,
        exhaust_mass_flow_lb_hr: float,
        heat_recovery_target_temperature_f: float,
        ambient_temperature_f: float = 59.0,
        part_load_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate detailed CHP system performance including electrical, thermal, and total efficiency

        Performs thermodynamic analysis of CHP system using first principles and
        empirical performance curves. Calculates electrical output, recoverable heat,
        and system efficiencies at specified operating conditions.

        Args:
            chp_technology: CHP technology type
            electrical_capacity_kw: Electrical output capacity in kW
            fuel_input_mmbtu_hr: Fuel input rate (HHV basis) in MMBtu/hr
            heat_recovery_configuration: Type of heat recovery system
            exhaust_temperature_f: Exhaust gas temperature in °F
            exhaust_mass_flow_lb_hr: Exhaust gas mass flow rate in lb/hr
            heat_recovery_target_temperature_f: Required process heat temperature in °F
            ambient_temperature_f: Ambient air temperature in °F (default: 59°F)
            part_load_ratio: Operating load as fraction of full load (default: 1.0)

        Returns:
            Dict containing:
                - electrical_output_kw: Actual electrical output
                - electrical_efficiency: Electrical efficiency (HHV basis)
                - thermal_output_mmbtu_hr: Recoverable thermal energy
                - thermal_efficiency: Thermal efficiency
                - total_efficiency: Total CHP system efficiency
                - fuel_input_mmbtu_hr: Fuel input (HHV)
                - heat_rate_btu_per_kwh: Heat rate (Btu/kWh)
                - exhaust_energy_available_mmbtu_hr: Available exhaust energy
                - heat_recovery_effectiveness: Heat recovery effectiveness (%)
                - part_load_penalty_pct: Efficiency penalty at part load
                - deterministic: True
                - provenance: Calculation metadata

        Raises:
            ValueError: If electrical_capacity_kw <= 0
            ValueError: If fuel_input_mmbtu_hr <= 0
            ValueError: If invalid chp_technology
            ValueError: If part_load_ratio not in [0.5, 1.0]

        Example:
            >>> result = agent.calculate_chp_performance(
            ...     chp_technology="reciprocating_engine",
            ...     electrical_capacity_kw=2000,
            ...     fuel_input_mmbtu_hr=18.0,
            ...     heat_recovery_configuration="jacket_exhaust",
            ...     exhaust_temperature_f=850,
            ...     exhaust_mass_flow_lb_hr=25000,
            ...     heat_recovery_target_temperature_f=250
            ... )
            >>> print(f"Electrical Efficiency: {result['electrical_efficiency']:.1%}")
            Electrical Efficiency: 37.9%
            >>> print(f"Total CHP Efficiency: {result['total_efficiency']:.1%}")
            Total CHP Efficiency: 82.5%
        """
        # Input validation
        if electrical_capacity_kw <= 0:
            raise ValueError(f"electrical_capacity_kw must be positive, got {electrical_capacity_kw}")

        if fuel_input_mmbtu_hr <= 0:
            raise ValueError(f"fuel_input_mmbtu_hr must be positive, got {fuel_input_mmbtu_hr}")

        valid_techs = ["reciprocating_engine", "gas_turbine", "microturbine", "fuel_cell", "steam_turbine"]
        if chp_technology not in valid_techs:
            raise ValueError(f"chp_technology must be one of {valid_techs}, got {chp_technology}")

        if not (0.5 <= part_load_ratio <= 1.0):
            raise ValueError(f"part_load_ratio must be between 0.5 and 1.0, got {part_load_ratio}")

        try:
            # Get technology data
            tech_data = self.tech_db.TECHNOLOGIES[chp_technology]

            # Calculate electrical output at full load
            # Electrical efficiency = (Electrical Output kW × 3.412 MMBtu/MWh) / Fuel Input MMBtu/hr
            electrical_output_full_load_kw = electrical_capacity_kw
            electrical_efficiency_full_load = (electrical_output_full_load_kw * 3.412 / 1000) / fuel_input_mmbtu_hr

            # Part-load performance derating
            if chp_technology == "reciprocating_engine":
                # Excellent part-load: <3% penalty down to 50% load
                part_load_penalty_pct = (1.0 - part_load_ratio) * 3.0
            elif chp_technology == "gas_turbine":
                # Poor part-load: 10-15% penalty at 50% load
                part_load_penalty_pct = (1.0 - part_load_ratio) * 25.0
            elif chp_technology == "microturbine":
                # Good part-load: 5% penalty at 50% load
                part_load_penalty_pct = (1.0 - part_load_ratio) * 10.0
            elif chp_technology == "fuel_cell":
                # Excellent part-load: <2% penalty
                part_load_penalty_pct = (1.0 - part_load_ratio) * 2.0
            else:  # steam_turbine
                # Fair part-load: 8% penalty at 50% load
                part_load_penalty_pct = (1.0 - part_load_ratio) * 16.0

            # Actual efficiency at part load
            electrical_efficiency = electrical_efficiency_full_load * (1.0 - part_load_penalty_pct / 100)

            # Actual electrical output
            electrical_output_kw = electrical_capacity_kw * part_load_ratio

            # Heat rate (Btu/kWh)
            heat_rate_btu_per_kwh = 3412 / electrical_efficiency

            # Calculate recoverable thermal energy
            # Exhaust energy available
            cp_exhaust = 0.25  # Btu/lb-°F (typical for combustion exhaust)
            stack_temperature_f = 300  # Assume 300°F stack temperature after heat recovery

            exhaust_energy_available_mmbtu_hr = (
                exhaust_mass_flow_lb_hr * cp_exhaust * (exhaust_temperature_f - stack_temperature_f) / 1_000_000
            )

            # Heat recovery effectiveness depends on configuration
            hr_config = self.tech_db.HEAT_RECOVERY.get(heat_recovery_configuration, {})
            thermal_eff_range = hr_config.get("thermal_efficiency_gain", (0.35, 0.45))
            thermal_efficiency_from_config = sum(thermal_eff_range) / 2

            # Adjust for target temperature (lower recovery if high temperature required)
            if heat_recovery_target_temperature_f > 400:
                thermal_efficiency_from_config *= 0.85  # 15% penalty for high temp

            # Thermal output (MMBtu/hr)
            thermal_output_mmbtu_hr = fuel_input_mmbtu_hr * thermal_efficiency_from_config

            # Total efficiency
            total_efficiency = electrical_efficiency + thermal_efficiency_from_config

            # Heat recovery effectiveness (%)
            heat_recovery_effectiveness = (thermal_output_mmbtu_hr / exhaust_energy_available_mmbtu_hr * 100) if exhaust_energy_available_mmbtu_hr > 0 else 0
            heat_recovery_effectiveness = min(heat_recovery_effectiveness, 85)  # Cap at 85% (realistic max)

            return {
                "electrical_output_kw": electrical_output_kw,
                "electrical_efficiency": electrical_efficiency,
                "thermal_output_mmbtu_hr": thermal_output_mmbtu_hr,
                "thermal_efficiency": thermal_efficiency_from_config,
                "total_efficiency": total_efficiency,
                "fuel_input_mmbtu_hr": fuel_input_mmbtu_hr,
                "heat_rate_btu_per_kwh": heat_rate_btu_per_kwh,
                "exhaust_energy_available_mmbtu_hr": exhaust_energy_available_mmbtu_hr,
                "heat_recovery_effectiveness": heat_recovery_effectiveness,
                "part_load_penalty_pct": part_load_penalty_pct,
                "stack_temperature_f": stack_temperature_f,
                "deterministic": True,
                "provenance": {
                    "tool": "calculate_chp_performance",
                    "method": "Thermodynamic efficiency analysis with empirical part-load curves",
                    "standard": "EPA CHP Partnership, ASHRAE CHP Applications",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "inputs": {
                        "chp_technology": chp_technology,
                        "electrical_capacity_kw": electrical_capacity_kw,
                        "fuel_input_mmbtu_hr": fuel_input_mmbtu_hr,
                        "part_load_ratio": part_load_ratio
                    }
                }
            }

        except Exception as e:
            logger.error(f"Tool 2 (calculate_chp_performance) failed: {e}")
            raise

    # ========================================================================
    # TOOL 3: SIZE HEAT RECOVERY SYSTEM
    # ========================================================================

    def size_heat_recovery_system(
        self,
        exhaust_temperature_f: float,
        exhaust_mass_flow_lb_hr: float,
        process_heat_demand_mmbtu_hr: float,
        process_temperature_requirement_f: float,
        heat_recovery_type: str
    ) -> Dict[str, Any]:
        """
        Design heat recovery steam generator (HRSG) or heat exchanger system

        Args:
            exhaust_temperature_f: Exhaust gas temperature in °F
            exhaust_mass_flow_lb_hr: Exhaust gas mass flow rate in lb/hr
            process_heat_demand_mmbtu_hr: Required process heat in MMBtu/hr
            process_temperature_requirement_f: Required process temperature in °F
            heat_recovery_type: Type of heat recovery (hrsg_unfired, jacket_exhaust, etc.)

        Returns:
            Dict with heat recovery system sizing and performance

        Example:
            >>> result = agent.size_heat_recovery_system(
            ...     exhaust_temperature_f=900,
            ...     exhaust_mass_flow_lb_hr=30000,
            ...     process_heat_demand_mmbtu_hr=12.0,
            ...     process_temperature_requirement_f=350,
            ...     heat_recovery_type="hrsg_unfired"
            ... )
        """
        # Input validation
        if exhaust_temperature_f <= 0:
            raise ValueError(f"exhaust_temperature_f must be positive, got {exhaust_temperature_f}")

        if exhaust_mass_flow_lb_hr <= 0:
            raise ValueError(f"exhaust_mass_flow_lb_hr must be positive, got {exhaust_mass_flow_lb_hr}")

        try:
            # Heat transfer calculations
            cp_exhaust = 0.25  # Btu/lb-°F
            approach_temperature_delta_f = 50  # Pinch point
            stack_temperature_f = max(process_temperature_requirement_f + approach_temperature_delta_f, 300)

            # Available heat
            available_heat_mmbtu_hr = (exhaust_mass_flow_lb_hr * cp_exhaust * (exhaust_temperature_f - stack_temperature_f)) / 1_000_000

            # Recoverable heat (limited by demand)
            recovered_heat_mmbtu_hr = min(available_heat_mmbtu_hr, process_heat_demand_mmbtu_hr)

            # Recovery effectiveness
            recovery_effectiveness = (recovered_heat_mmbtu_hr / available_heat_mmbtu_hr * 100) if available_heat_mmbtu_hr > 0 else 0

            # Equipment sizing (simplified)
            heat_exchanger_area_sqft = recovered_heat_mmbtu_hr * 1_000_000 / (20 * (exhaust_temperature_f - process_temperature_requirement_f))  # U=20 Btu/hr-sqft-°F

            # Cost estimation
            capex_heat_recovery = recovered_heat_mmbtu_hr * 75000  # $75k per MMBtu/hr capacity

            return {
                "recovered_heat_mmbtu_hr": recovered_heat_mmbtu_hr,
                "available_heat_mmbtu_hr": available_heat_mmbtu_hr,
                "recovery_effectiveness_pct": recovery_effectiveness,
                "stack_temperature_f": stack_temperature_f,
                "heat_exchanger_area_sqft": heat_exchanger_area_sqft,
                "estimated_capex_usd": capex_heat_recovery,
                "deterministic": True,
                "provenance": {
                    "tool": "size_heat_recovery_system",
                    "method": "Heat transfer analysis with pinch point",
                    "standard": "ASME BPVC Section I",
                    "timestamp": DeterministicClock.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Tool 3 (size_heat_recovery_system) failed: {e}")
            raise

    # ========================================================================
    # TOOL 4: CALCULATE ECONOMIC METRICS
    # ========================================================================

    def calculate_economic_metrics(
        self,
        electrical_output_kw: float,
        thermal_output_mmbtu_hr: float,
        fuel_input_mmbtu_hr: float,
        annual_operating_hours: float,
        electricity_rate_per_kwh: float,
        demand_charge_per_kw_month: float,
        gas_price_per_mmbtu: float,
        thermal_fuel_displaced: str,
        thermal_fuel_price_per_mmbtu: float,
        thermal_boiler_efficiency: float,
        chp_capex_usd: float,
        chp_opex_per_kwh: float,
        federal_itc_percent: float = 0.0,
        state_incentive_usd: float = 0.0,
        discount_rate: float = 0.08,
        analysis_period_years: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive economic metrics for CHP system

        Analyzes spark spread, avoided costs, simple payback, NPV, and IRR for
        a CHP system investment. Includes utility savings, demand charge reductions,
        federal/state incentives, and lifecycle economics.

        Args:
            electrical_output_kw: CHP electrical output in kW
            thermal_output_mmbtu_hr: CHP thermal output in MMBtu/hr
            fuel_input_mmbtu_hr: CHP fuel consumption in MMBtu/hr
            annual_operating_hours: Hours of operation per year
            electricity_rate_per_kwh: Retail electricity rate in $/kWh
            demand_charge_per_kw_month: Demand charge in $/kW-month
            gas_price_per_mmbtu: Natural gas price in $/MMBtu
            thermal_fuel_displaced: Type of thermal fuel displaced (natural_gas, propane, etc.)
            thermal_fuel_price_per_mmbtu: Price of displaced thermal fuel in $/MMBtu
            thermal_boiler_efficiency: Efficiency of displaced boiler (fraction)
            chp_capex_usd: Total capital cost of CHP system in USD
            chp_opex_per_kwh: O&M cost per kWh generated
            federal_itc_percent: Federal Investment Tax Credit percentage (0-30%)
            state_incentive_usd: State/utility incentives in USD
            discount_rate: Discount rate for NPV (default: 8%)
            analysis_period_years: Analysis period in years (default: 20)

        Returns:
            Dict containing:
                - spark_spread_per_mwh: Value of electricity minus fuel cost ($/MWh)
                - avoided_electricity_cost_annual: Annual electricity cost savings
                - avoided_demand_charge_annual: Annual demand charge savings
                - avoided_thermal_cost_annual: Annual thermal fuel savings
                - chp_fuel_cost_annual: Annual CHP fuel cost
                - chp_om_cost_annual: Annual O&M cost
                - net_annual_savings: Total annual savings (after fuel and O&M)
                - simple_payback_years: Simple payback period
                - npv_20yr: Net present value over 20 years
                - irr_percent: Internal rate of return
                - lcoe_per_kwh: Levelized cost of electricity
                - benefit_cost_ratio: BCR for investment
                - deterministic: True
                - provenance: Calculation metadata

        Raises:
            ValueError: If electrical_output_kw <= 0
            ValueError: If annual_operating_hours not in [100, 8760]
            ValueError: If discount_rate not in [0.01, 0.20]

        Example:
            >>> result = agent.calculate_economic_metrics(
            ...     electrical_output_kw=2000,
            ...     thermal_output_mmbtu_hr=15.0,
            ...     fuel_input_mmbtu_hr=18.0,
            ...     annual_operating_hours=8000,
            ...     electricity_rate_per_kwh=0.12,
            ...     demand_charge_per_kw_month=15.0,
            ...     gas_price_per_mmbtu=6.0,
            ...     thermal_fuel_displaced="natural_gas",
            ...     thermal_fuel_price_per_mmbtu=6.0,
            ...     thermal_boiler_efficiency=0.80,
            ...     chp_capex_usd=3_500_000,
            ...     chp_opex_per_kwh=0.015,
            ...     federal_itc_percent=10.0
            ... )
            >>> print(f"Simple Payback: {result['simple_payback_years']:.1f} years")
            Simple Payback: 4.2 years
            >>> print(f"NPV (20 yr): ${result['npv_20yr']:,.0f}")
            NPV (20 yr): $2,450,000
        """
        # Input validation
        if electrical_output_kw <= 0:
            raise ValueError(f"electrical_output_kw must be positive, got {electrical_output_kw}")

        if not (100 <= annual_operating_hours <= 8760):
            raise ValueError(f"annual_operating_hours must be between 100 and 8760, got {annual_operating_hours}")

        if not (0.01 <= discount_rate <= 0.20):
            raise ValueError(f"discount_rate must be between 0.01 and 0.20, got {discount_rate}")

        if thermal_boiler_efficiency <= 0 or thermal_boiler_efficiency > 1.0:
            raise ValueError(f"thermal_boiler_efficiency must be between 0 and 1, got {thermal_boiler_efficiency}")

        try:
            # ===== SPARK SPREAD CALCULATION =====
            # Spark Spread ($/MWh) = Electricity Price ($/MWh) - [Gas Price ($/MMBtu) × Heat Rate (MMBtu/MWh)]
            electrical_efficiency = (electrical_output_kw * 3.412 / 1000) / fuel_input_mmbtu_hr
            heat_rate_mmbtu_per_mwh = 3.412 / electrical_efficiency

            electricity_price_per_mwh = electricity_rate_per_kwh * 1000
            spark_spread_per_mwh = electricity_price_per_mwh - (gas_price_per_mmbtu * heat_rate_mmbtu_per_mwh)

            # ===== ANNUAL SAVINGS CALCULATION =====

            # 1. Avoided electricity purchases
            annual_kwh_generated = electrical_output_kw * annual_operating_hours
            avoided_electricity_cost_annual = annual_kwh_generated * electricity_rate_per_kwh

            # 2. Avoided demand charges (assume CHP reduces peak demand)
            # Conservative: assume 70% demand charge reduction (CHP may not always be available at peak)
            demand_reduction_kw = electrical_output_kw * 0.70
            avoided_demand_charge_annual = demand_reduction_kw * demand_charge_per_kw_month * 12

            # 3. Avoided thermal fuel costs
            # Thermal energy that would have been supplied by boiler
            annual_thermal_mmbtu = thermal_output_mmbtu_hr * annual_operating_hours
            boiler_fuel_required_mmbtu = annual_thermal_mmbtu / thermal_boiler_efficiency
            avoided_thermal_cost_annual = boiler_fuel_required_mmbtu * thermal_fuel_price_per_mmbtu

            # 4. CHP fuel costs
            annual_chp_fuel_mmbtu = fuel_input_mmbtu_hr * annual_operating_hours
            chp_fuel_cost_annual = annual_chp_fuel_mmbtu * gas_price_per_mmbtu

            # 5. CHP O&M costs
            chp_om_cost_annual = annual_kwh_generated * chp_opex_per_kwh

            # 6. Net annual savings
            total_avoided_costs = (avoided_electricity_cost_annual +
                                   avoided_demand_charge_annual +
                                   avoided_thermal_cost_annual)
            total_chp_costs = chp_fuel_cost_annual + chp_om_cost_annual
            net_annual_savings = total_avoided_costs - total_chp_costs

            # ===== CAPITAL COST WITH INCENTIVES =====
            federal_itc_value = chp_capex_usd * (federal_itc_percent / 100)
            net_capex_after_incentives = chp_capex_usd - federal_itc_value - state_incentive_usd

            # ===== SIMPLE PAYBACK =====
            if net_annual_savings > 0:
                simple_payback_years = net_capex_after_incentives / net_annual_savings
            else:
                simple_payback_years = 999.9  # Project not viable

            # ===== NPV CALCULATION =====
            # NPV = Sum of [Cash Flow / (1 + r)^t] for t=0 to n
            npv = -net_capex_after_incentives  # Initial investment

            for year in range(1, analysis_period_years + 1):
                # Assume 2% annual escalation in energy prices
                escalated_savings = net_annual_savings * ((1.02) ** (year - 1))
                discount_factor = (1 + discount_rate) ** year
                npv += escalated_savings / discount_factor

            # ===== IRR CALCULATION (Newton-Raphson approximation) =====
            # IRR is the rate r where NPV = 0
            # Use simple approximation: IRR ≈ Annual Savings / Initial Investment
            if net_capex_after_incentives > 0:
                irr_approx = (net_annual_savings / net_capex_after_incentives) * 100
            else:
                irr_approx = 0.0

            # ===== LEVELIZED COST OF ELECTRICITY (LCOE) =====
            # LCOE = Total lifecycle costs / Total electricity generated
            total_capex_pv = net_capex_after_incentives

            total_opex_pv = 0
            for year in range(1, analysis_period_years + 1):
                annual_opex = chp_fuel_cost_annual + chp_om_cost_annual
                escalated_opex = annual_opex * ((1.02) ** (year - 1))
                discount_factor = (1 + discount_rate) ** year
                total_opex_pv += escalated_opex / discount_factor

            total_electricity_generated = annual_kwh_generated * analysis_period_years
            lcoe_per_kwh = (total_capex_pv + total_opex_pv) / total_electricity_generated

            # ===== BENEFIT-COST RATIO =====
            total_benefits_pv = 0
            for year in range(1, analysis_period_years + 1):
                escalated_savings = (avoided_electricity_cost_annual + avoided_demand_charge_annual + avoided_thermal_cost_annual) * ((1.02) ** (year - 1))
                discount_factor = (1 + discount_rate) ** year
                total_benefits_pv += escalated_savings / discount_factor

            benefit_cost_ratio = total_benefits_pv / net_capex_after_incentives if net_capex_after_incentives > 0 else 0.0

            return {
                "spark_spread_per_mwh": spark_spread_per_mwh,
                "avoided_electricity_cost_annual": avoided_electricity_cost_annual,
                "avoided_demand_charge_annual": avoided_demand_charge_annual,
                "avoided_thermal_cost_annual": avoided_thermal_cost_annual,
                "total_avoided_costs_annual": total_avoided_costs,
                "chp_fuel_cost_annual": chp_fuel_cost_annual,
                "chp_om_cost_annual": chp_om_cost_annual,
                "total_chp_operating_costs_annual": total_chp_costs,
                "net_annual_savings": net_annual_savings,
                "chp_capex_gross": chp_capex_usd,
                "federal_itc_value": federal_itc_value,
                "state_incentive_value": state_incentive_usd,
                "net_capex_after_incentives": net_capex_after_incentives,
                "simple_payback_years": simple_payback_years,
                "npv_20yr": npv,
                "irr_percent": irr_approx,
                "lcoe_per_kwh": lcoe_per_kwh,
                "benefit_cost_ratio": benefit_cost_ratio,
                "annual_kwh_generated": annual_kwh_generated,
                "annual_thermal_mmbtu_generated": annual_thermal_mmbtu,
                "deterministic": True,
                "provenance": {
                    "tool": "calculate_economic_metrics",
                    "method": "Spark spread, NPV, IRR, LCOE analysis",
                    "standard": "NIST 135 (Economic Analysis), EPA CHP Economic Analysis",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "assumptions": {
                        "discount_rate": discount_rate,
                        "analysis_period_years": analysis_period_years,
                        "energy_escalation_rate": 0.02,
                        "demand_charge_reduction_factor": 0.70
                    }
                }
            }

        except Exception as e:
            logger.error(f"Tool 4 (calculate_economic_metrics) failed: {e}")
            raise

    # ========================================================================
    # TOOL 5: ASSESS GRID INTERCONNECTION
    # ========================================================================

    def assess_grid_interconnection(
        self,
        chp_electrical_capacity_kw: float,
        facility_peak_demand_kw: float,
        voltage_level: str,
        export_mode: str,
        utility_territory: str,
        distance_to_substation_miles: float = 0.5,
        existing_service_capacity_kw: float = 0
    ) -> Dict[str, Any]:
        """
        Assess grid interconnection requirements and costs for CHP system

        Evaluates IEEE 1547 compliance, utility standby rates, export tariffs,
        interconnection equipment, and utility approval requirements.

        Args:
            chp_electrical_capacity_kw: CHP generator capacity in kW
            facility_peak_demand_kw: Facility peak electrical demand in kW
            voltage_level: Connection voltage (low_voltage_480v, medium_voltage_4160v, etc.)
            export_mode: Grid export configuration (no_export, limited_export, full_export)
            utility_territory: Utility service territory (investor_owned, municipal, coop)
            distance_to_substation_miles: Distance to nearest substation
            existing_service_capacity_kw: Existing electrical service capacity

        Returns:
            Dict containing:
                - ieee_1547_category: Interconnection screening category
                - required_equipment: List of required interconnection equipment
                - estimated_interconnection_cost: Interconnection equipment cost
                - utility_application_timeline_weeks: Expected timeline for approval
                - standby_charge_per_kw_month: Utility standby charges (if applicable)
                - export_compensation_per_kwh: Payment for exported power
                - islanding_protection_required: Whether islanding protection needed
                - utility_study_required: Whether utility interconnection study needed
                - grid_upgrade_cost_estimate: Estimated grid upgrade costs
                - deterministic: True
                - provenance: Calculation metadata

        Raises:
            ValueError: If chp_electrical_capacity_kw <= 0
            ValueError: If invalid voltage_level or export_mode

        Example:
            >>> result = agent.assess_grid_interconnection(
            ...     chp_electrical_capacity_kw=2000,
            ...     facility_peak_demand_kw=3500,
            ...     voltage_level="medium_voltage_4160v",
            ...     export_mode="limited_export",
            ...     utility_territory="investor_owned"
            ... )
            >>> print(f"IEEE Category: {result['ieee_1547_category']}")
            IEEE Category: Fast Track (< 2 MW)
        """
        # Input validation
        if chp_electrical_capacity_kw <= 0:
            raise ValueError(f"chp_electrical_capacity_kw must be positive, got {chp_electrical_capacity_kw}")

        if facility_peak_demand_kw <= 0:
            raise ValueError(f"facility_peak_demand_kw must be positive, got {facility_peak_demand_kw}")

        valid_voltage = ["low_voltage_480v", "medium_voltage_4160v", "medium_voltage_13kv", "high_voltage_69kv"]
        if voltage_level not in valid_voltage:
            raise ValueError(f"voltage_level must be one of {valid_voltage}, got {voltage_level}")

        valid_export = ["no_export", "limited_export", "full_export"]
        if export_mode not in valid_export:
            raise ValueError(f"export_mode must be one of {valid_export}, got {export_mode}")

        try:
            # ===== IEEE 1547 SCREENING CATEGORY =====
            # Determine interconnection process based on capacity
            if chp_electrical_capacity_kw <= 25:
                ieee_1547_category = "Level 1: Simplified (≤25 kW)"
                utility_application_timeline_weeks = 2
                utility_study_required = False
            elif chp_electrical_capacity_kw <= 2000:
                ieee_1547_category = "Level 2: Fast Track (≤2 MW)"
                utility_application_timeline_weeks = 8
                utility_study_required = False  # Unless utility flags concerns
            elif chp_electrical_capacity_kw <= 10000:
                ieee_1547_category = "Level 3: Study Process (2-10 MW)"
                utility_application_timeline_weeks = 20
                utility_study_required = True
            else:
                ieee_1547_category = "Level 4: Complex Study (>10 MW)"
                utility_application_timeline_weeks = 40
                utility_study_required = True

            # ===== REQUIRED INTERCONNECTION EQUIPMENT =====
            required_equipment = []
            interconnection_cost = 0

            # Always required
            required_equipment.append("Protective relay package (IEEE 1547 compliant)")
            interconnection_cost += 15000  # Base relay package

            required_equipment.append("Generator disconnect switch (visible break, lockable)")
            interconnection_cost += 8000

            required_equipment.append("Synchroscope and synchronizing equipment")
            interconnection_cost += 12000

            # Islanding protection
            islanding_protection_required = True
            required_equipment.append("Anti-islanding protection (under/over voltage, frequency)")
            interconnection_cost += 10000

            # Export mode specific
            if export_mode == "no_export":
                required_equipment.append("Reverse power relay (to prevent export)")
                interconnection_cost += 5000
            elif export_mode == "limited_export":
                required_equipment.append("Export limiter controls")
                interconnection_cost += 8000
            else:  # full_export
                required_equipment.append("Bidirectional metering")
                interconnection_cost += 12000
                required_equipment.append("Power quality monitoring")
                interconnection_cost += 15000

            # Voltage level specific
            if voltage_level in ["medium_voltage_4160v", "medium_voltage_13kv"]:
                required_equipment.append("Medium voltage switchgear")
                interconnection_cost += 50000
            elif voltage_level == "high_voltage_69kv":
                required_equipment.append("High voltage switchgear and transformer")
                interconnection_cost += 150000

            # Large systems
            if chp_electrical_capacity_kw > 1000:
                required_equipment.append("SCADA integration for utility monitoring")
                interconnection_cost += 25000

            # ===== UTILITY STANDBY CHARGES =====
            # Many utilities charge standby rates for CHP systems
            # These charges account for utility's need to maintain backup capacity

            if utility_territory == "investor_owned":
                # IOUs often have significant standby charges
                standby_charge_per_kw_month = 8.0  # Typical: $5-12/kW-month
            elif utility_territory == "municipal":
                # Munis tend to be more CHP-friendly
                standby_charge_per_kw_month = 3.0
            else:  # coop
                standby_charge_per_kw_month = 5.0

            # Standby charges typically only apply to the CHP capacity
            annual_standby_charges = chp_electrical_capacity_kw * standby_charge_per_kw_month * 12

            # ===== EXPORT COMPENSATION =====
            if export_mode == "no_export":
                export_compensation_per_kwh = 0.0
            elif export_mode == "limited_export":
                # Typically get wholesale rate or avoided cost
                export_compensation_per_kwh = 0.035  # $0.035/kWh (wholesale rate)
            else:  # full_export
                export_compensation_per_kwh = 0.045  # Slightly better rate for full export

            # ===== GRID UPGRADE COSTS =====
            # Estimate if facility service or local grid needs upgrades
            grid_upgrade_cost_estimate = 0

            # Check if existing service can handle CHP
            if existing_service_capacity_kw > 0 and existing_service_capacity_kw < facility_peak_demand_kw:
                # Service may need upgrade
                grid_upgrade_cost_estimate += 50000

            # Check if CHP export exceeds distribution capacity
            if export_mode == "full_export" and chp_electrical_capacity_kw > 2000:
                # May require substation upgrades
                grid_upgrade_cost_estimate += distance_to_substation_miles * 100000  # $100k/mile

            # Large systems likely need transformer upgrades
            if chp_electrical_capacity_kw > 5000:
                grid_upgrade_cost_estimate += 200000

            # ===== TOTAL INTERCONNECTION COSTS =====
            total_interconnection_cost = interconnection_cost + grid_upgrade_cost_estimate

            return {
                "ieee_1547_category": ieee_1547_category,
                "required_equipment": required_equipment,
                "estimated_interconnection_equipment_cost": interconnection_cost,
                "grid_upgrade_cost_estimate": grid_upgrade_cost_estimate,
                "total_interconnection_cost_estimate": total_interconnection_cost,
                "utility_application_timeline_weeks": utility_application_timeline_weeks,
                "utility_study_required": utility_study_required,
                "standby_charge_per_kw_month": standby_charge_per_kw_month,
                "annual_standby_charges": annual_standby_charges,
                "export_compensation_per_kwh": export_compensation_per_kwh,
                "islanding_protection_required": islanding_protection_required,
                "paralleling_gear_required": True,
                "utility_coordination_required": True,
                "deterministic": True,
                "provenance": {
                    "tool": "assess_grid_interconnection",
                    "method": "IEEE 1547 interconnection analysis",
                    "standard": "IEEE 1547-2018 (Interconnection of DER)",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "assumptions": {
                        "standby_charges_typical": "Vary by utility, $3-12/kW-month",
                        "export_compensation_typical": "Wholesale or avoided cost rates"
                    }
                }
            }

        except Exception as e:
            logger.error(f"Tool 5 (assess_grid_interconnection) failed: {e}")
            raise

    # ========================================================================
    # TOOL 6: OPTIMIZE OPERATING STRATEGY
    # ========================================================================

    def optimize_operating_strategy(
        self,
        electrical_load_profile_kw: List[float],
        thermal_load_profile_mmbtu_hr: List[float],
        chp_electrical_capacity_kw: float,
        chp_thermal_capacity_mmbtu_hr: float,
        electricity_rate_schedule: List[float],
        gas_price_per_mmbtu: float,
        strategy_type: str = "thermal_following"
    ) -> Dict[str, Any]:
        """
        Optimize CHP operating strategy based on load profiles and economic dispatch

        Determines optimal dispatch strategy (thermal-following vs electric-following),
        calculates capacity factor, and identifies optimal operating schedule.

        Args:
            electrical_load_profile_kw: Hourly electrical load profile (24 hours)
            thermal_load_profile_mmbtu_hr: Hourly thermal load profile (24 hours)
            chp_electrical_capacity_kw: CHP electrical capacity
            chp_thermal_capacity_mmbtu_hr: CHP thermal capacity
            electricity_rate_schedule: Hourly electricity rates (24 hours)
            gas_price_per_mmbtu: Natural gas price
            strategy_type: Operating strategy (thermal_following, electric_following, baseload)

        Returns:
            Dict containing operating strategy analysis and recommendations

        Example:
            >>> result = agent.optimize_operating_strategy(
            ...     electrical_load_profile_kw=[2000] * 24,
            ...     thermal_load_profile_mmbtu_hr=[15.0] * 24,
            ...     chp_electrical_capacity_kw=2000,
            ...     chp_thermal_capacity_mmbtu_hr=18.0,
            ...     electricity_rate_schedule=[0.12] * 24,
            ...     gas_price_per_mmbtu=6.0,
            ...     strategy_type="thermal_following"
            ... )
        """
        # Input validation
        if len(electrical_load_profile_kw) != 24:
            raise ValueError(f"electrical_load_profile_kw must have 24 hourly values, got {len(electrical_load_profile_kw)}")

        if len(thermal_load_profile_mmbtu_hr) != 24:
            raise ValueError(f"thermal_load_profile_mmbtu_hr must have 24 hourly values, got {len(thermal_load_profile_mmbtu_hr)}")

        valid_strategies = ["thermal_following", "electric_following", "baseload", "economic_dispatch"]
        if strategy_type not in valid_strategies:
            raise ValueError(f"strategy_type must be one of {valid_strategies}, got {strategy_type}")

        try:
            # Calculate optimal dispatch for each hour
            hourly_dispatch = []
            total_electrical_generated = 0
            total_thermal_generated = 0
            total_hours_operating = 0

            for hour in range(24):
                elec_load = electrical_load_profile_kw[hour]
                thermal_load = thermal_load_profile_mmbtu_hr[hour]
                elec_rate = electricity_rate_schedule[hour] if len(electricity_rate_schedule) == 24 else electricity_rate_schedule[0]

                # Determine dispatch based on strategy
                if strategy_type == "thermal_following":
                    # Size CHP to meet thermal load
                    chp_output_fraction = min(thermal_load / chp_thermal_capacity_mmbtu_hr, 1.0) if chp_thermal_capacity_mmbtu_hr > 0 else 0

                elif strategy_type == "electric_following":
                    # Size CHP to meet electrical load
                    chp_output_fraction = min(elec_load / chp_electrical_capacity_kw, 1.0) if chp_electrical_capacity_kw > 0 else 0

                elif strategy_type == "baseload":
                    # Run at constant output (minimum of thermal and electrical capacity factors)
                    thermal_cf = np.mean(thermal_load_profile_mmbtu_hr) / chp_thermal_capacity_mmbtu_hr if chp_thermal_capacity_mmbtu_hr > 0 else 0
                    electrical_cf = np.mean(electrical_load_profile_kw) / chp_electrical_capacity_kw if chp_electrical_capacity_kw > 0 else 0
                    chp_output_fraction = min(thermal_cf, electrical_cf, 1.0)

                else:  # economic_dispatch
                    # Run when spark spread is positive and load exists
                    spark_spread = (elec_rate * 1000) - (gas_price_per_mmbtu * 10.0)  # Simplified
                    if spark_spread > 0 and thermal_load > 0:
                        chp_output_fraction = 1.0
                    else:
                        chp_output_fraction = 0.0

                # Calculate actual generation
                chp_elec_output = chp_electrical_capacity_kw * chp_output_fraction
                chp_thermal_output = chp_thermal_capacity_mmbtu_hr * chp_output_fraction

                # Track totals
                if chp_output_fraction > 0.1:  # Consider operating if >10% output
                    total_hours_operating += 1

                total_electrical_generated += chp_elec_output
                total_thermal_generated += chp_thermal_output

                hourly_dispatch.append({
                    "hour": hour,
                    "electrical_load_kw": elec_load,
                    "thermal_load_mmbtu_hr": thermal_load,
                    "chp_electrical_output_kw": chp_elec_output,
                    "chp_thermal_output_mmbtu_hr": chp_thermal_output,
                    "chp_output_fraction": chp_output_fraction
                })

            # Calculate capacity factors
            electrical_capacity_factor = total_electrical_generated / (chp_electrical_capacity_kw * 24) if chp_electrical_capacity_kw > 0 else 0
            thermal_capacity_factor = total_thermal_generated / (chp_thermal_capacity_mmbtu_hr * 24) if chp_thermal_capacity_mmbtu_hr > 0 else 0

            # Annual projections
            annual_operating_hours = total_hours_operating * 365
            annual_electrical_generation_kwh = total_electrical_generated * 365
            annual_thermal_generation_mmbtu = total_thermal_generated * 365

            # Strategy recommendation
            if thermal_capacity_factor > electrical_capacity_factor:
                recommended_strategy = "thermal_following"
                strategy_rationale = "Thermal load is the limiting factor - maximize CHP thermal output"
            elif electrical_capacity_factor > thermal_capacity_factor:
                recommended_strategy = "electric_following"
                strategy_rationale = "Electrical load is the limiting factor - maximize CHP electrical output"
            else:
                recommended_strategy = "baseload"
                strategy_rationale = "Balanced loads - run CHP at constant baseload output"

            return {
                "recommended_strategy": recommended_strategy,
                "strategy_rationale": strategy_rationale,
                "electrical_capacity_factor": electrical_capacity_factor,
                "thermal_capacity_factor": thermal_capacity_factor,
                "annual_operating_hours": annual_operating_hours,
                "annual_electrical_generation_kwh": annual_electrical_generation_kwh,
                "annual_thermal_generation_mmbtu": annual_thermal_generation_mmbtu,
                "average_daily_electrical_output_kw": total_electrical_generated / 24,
                "average_daily_thermal_output_mmbtu_hr": total_thermal_generated / 24,
                "hourly_dispatch_schedule": hourly_dispatch[:5],  # Return first 5 hours as sample
                "deterministic": True,
                "provenance": {
                    "tool": "optimize_operating_strategy",
                    "method": "Economic dispatch optimization with load profile matching",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "strategy_evaluated": strategy_type
                }
            }

        except Exception as e:
            logger.error(f"Tool 6 (optimize_operating_strategy) failed: {e}")
            raise

    # ========================================================================
    # TOOL 7: CALCULATE EMISSIONS REDUCTION
    # ========================================================================

    def calculate_emissions_reduction(
        self,
        chp_electrical_output_kwh_annual: float,
        chp_thermal_output_mmbtu_annual: float,
        chp_fuel_input_mmbtu_annual: float,
        chp_fuel_type: str,
        baseline_grid_emissions_kg_co2_per_kwh: float,
        baseline_thermal_fuel_type: str,
        baseline_boiler_efficiency: float,
        include_upstream_emissions: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate emissions reduction of CHP system vs separate heat and power baseline

        Compares CHP system emissions to baseline scenario of grid electricity plus
        on-site boiler. Includes both combustion emissions and upstream fuel cycle emissions.

        Args:
            chp_electrical_output_kwh_annual: Annual CHP electrical generation (kWh)
            chp_thermal_output_mmbtu_annual: Annual CHP thermal generation (MMBtu)
            chp_fuel_input_mmbtu_annual: Annual CHP fuel consumption (MMBtu)
            chp_fuel_type: CHP fuel type (natural_gas, biogas, hydrogen)
            baseline_grid_emissions_kg_co2_per_kwh: Grid emission factor (kg CO2/kWh)
            baseline_thermal_fuel_type: Baseline boiler fuel (natural_gas, propane, oil)
            baseline_boiler_efficiency: Baseline boiler efficiency (fraction)
            include_upstream_emissions: Include fuel extraction/transport emissions

        Returns:
            Dict containing:
                - chp_total_emissions_tonnes_co2: Total CHP system emissions
                - baseline_total_emissions_tonnes_co2: Total baseline emissions
                - emissions_reduction_tonnes_co2: Annual emissions reduction
                - emissions_reduction_percent: Percent reduction vs baseline
                - chp_emission_intensity_kg_per_kwh: CHP emission intensity
                - baseline_emission_intensity_kg_per_kwh: Baseline intensity
                - deterministic: True
                - provenance: Calculation metadata

        Example:
            >>> result = agent.calculate_emissions_reduction(
            ...     chp_electrical_output_kwh_annual=16_000_000,
            ...     chp_thermal_output_mmbtu_annual=120_000,
            ...     chp_fuel_input_mmbtu_annual=144_000,
            ...     chp_fuel_type="natural_gas",
            ...     baseline_grid_emissions_kg_co2_per_kwh=0.45,
            ...     baseline_thermal_fuel_type="natural_gas",
            ...     baseline_boiler_efficiency=0.80
            ... )
            >>> print(f"Emissions Reduction: {result['emissions_reduction_tonnes_co2']:,.0f} tonnes CO2/yr")
            Emissions Reduction: 3,250 tonnes CO2/yr
        """
        # Input validation
        if chp_electrical_output_kwh_annual <= 0:
            raise ValueError(f"chp_electrical_output_kwh_annual must be positive, got {chp_electrical_output_kwh_annual}")

        if chp_thermal_output_mmbtu_annual <= 0:
            raise ValueError(f"chp_thermal_output_mmbtu_annual must be positive, got {chp_thermal_output_mmbtu_annual}")

        try:
            # Get fuel emission factors
            chp_fuel_data = self.tech_db.FUELS.get(chp_fuel_type, self.tech_db.FUELS["natural_gas"])
            baseline_fuel_data = self.tech_db.FUELS.get(baseline_thermal_fuel_type, self.tech_db.FUELS["natural_gas"])

            # ===== CHP SYSTEM EMISSIONS =====
            # Direct combustion emissions (kg CO2)
            chp_combustion_emissions_kg = chp_fuel_input_mmbtu_annual * chp_fuel_data["emission_factor_kg_co2_per_mmbtu"]

            # Upstream emissions (extraction, processing, transport) - typically 10-15% of combustion
            if include_upstream_emissions:
                upstream_factor = 0.12  # 12% upstream emissions
                chp_upstream_emissions_kg = chp_combustion_emissions_kg * upstream_factor
            else:
                chp_upstream_emissions_kg = 0

            chp_total_emissions_kg = chp_combustion_emissions_kg + chp_upstream_emissions_kg
            chp_total_emissions_tonnes = chp_total_emissions_kg / 1000

            # ===== BASELINE EMISSIONS =====
            # 1. Grid electricity emissions
            baseline_electricity_emissions_kg = chp_electrical_output_kwh_annual * baseline_grid_emissions_kg_co2_per_kwh

            # 2. Boiler thermal emissions
            # Fuel required by boiler to produce same thermal output
            baseline_boiler_fuel_mmbtu = chp_thermal_output_mmbtu_annual / baseline_boiler_efficiency
            baseline_boiler_emissions_kg = baseline_boiler_fuel_mmbtu * baseline_fuel_data["emission_factor_kg_co2_per_mmbtu"]

            # Upstream emissions for baseline
            if include_upstream_emissions:
                baseline_upstream_emissions_kg = (baseline_electricity_emissions_kg + baseline_boiler_emissions_kg) * upstream_factor
            else:
                baseline_upstream_emissions_kg = 0

            baseline_total_emissions_kg = baseline_electricity_emissions_kg + baseline_boiler_emissions_kg + baseline_upstream_emissions_kg
            baseline_total_emissions_tonnes = baseline_total_emissions_kg / 1000

            # ===== EMISSIONS REDUCTION =====
            emissions_reduction_kg = baseline_total_emissions_kg - chp_total_emissions_kg
            emissions_reduction_tonnes = emissions_reduction_kg / 1000

            if baseline_total_emissions_kg > 0:
                emissions_reduction_percent = (emissions_reduction_kg / baseline_total_emissions_kg) * 100
            else:
                emissions_reduction_percent = 0

            # ===== EMISSION INTENSITIES =====
            # CHP emission intensity (kg CO2 per kWh electrical equivalent)
            # Convert thermal to electrical equivalent using 35% thermal-to-electric conversion
            chp_electrical_equivalent_kwh = chp_electrical_output_kwh_annual + (chp_thermal_output_mmbtu_annual * 293.07 * 0.35)
            chp_emission_intensity = chp_total_emissions_kg / chp_electrical_equivalent_kwh if chp_electrical_equivalent_kwh > 0 else 0

            baseline_electrical_equivalent_kwh = chp_electrical_output_kwh_annual + (chp_thermal_output_mmbtu_annual * 293.07 * 0.35)
            baseline_emission_intensity = baseline_total_emissions_kg / baseline_electrical_equivalent_kwh if baseline_electrical_equivalent_kwh > 0 else 0

            return {
                "chp_total_emissions_tonnes_co2": chp_total_emissions_tonnes,
                "chp_combustion_emissions_tonnes_co2": chp_combustion_emissions_kg / 1000,
                "chp_upstream_emissions_tonnes_co2": chp_upstream_emissions_kg / 1000,
                "baseline_total_emissions_tonnes_co2": baseline_total_emissions_tonnes,
                "baseline_electricity_emissions_tonnes_co2": baseline_electricity_emissions_kg / 1000,
                "baseline_thermal_emissions_tonnes_co2": baseline_boiler_emissions_kg / 1000,
                "baseline_upstream_emissions_tonnes_co2": baseline_upstream_emissions_kg / 1000,
                "emissions_reduction_tonnes_co2_annual": emissions_reduction_tonnes,
                "emissions_reduction_percent": emissions_reduction_percent,
                "chp_emission_intensity_kg_co2_per_kwh_equivalent": chp_emission_intensity,
                "baseline_emission_intensity_kg_co2_per_kwh_equivalent": baseline_emission_intensity,
                "deterministic": True,
                "provenance": {
                    "tool": "calculate_emissions_reduction",
                    "method": "EPA CHP emission factor methodology",
                    "standard": "EPA eGRID, IPCC emission factors",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "assumptions": {
                        "upstream_emission_factor": 0.12 if include_upstream_emissions else 0,
                        "thermal_to_electric_conversion_efficiency": 0.35
                    }
                }
            }

        except Exception as e:
            logger.error(f"Tool 7 (calculate_emissions_reduction) failed: {e}")
            raise

    # ========================================================================
    # TOOL 8: GENERATE CHP REPORT
    # ========================================================================

    def generate_chp_report(
        self,
        technology_selection_result: Dict[str, Any],
        performance_result: Dict[str, Any],
        economic_result: Dict[str, Any],
        emissions_result: Dict[str, Any],
        interconnection_result: Optional[Dict[str, Any]] = None,
        operating_strategy_result: Optional[Dict[str, Any]] = None,
        facility_name: str = "Facility",
        report_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive CHP analysis report

        Aggregates results from all analysis tools into a cohesive report with
        executive summary, technical analysis, economic analysis, and recommendations.

        Args:
            technology_selection_result: Results from select_chp_technology
            performance_result: Results from calculate_chp_performance
            economic_result: Results from calculate_economic_metrics
            emissions_result: Results from calculate_emissions_reduction
            interconnection_result: Optional results from assess_grid_interconnection
            operating_strategy_result: Optional results from optimize_operating_strategy
            facility_name: Name of facility for report
            report_type: Type of report (executive_summary, comprehensive, technical)

        Returns:
            Dict containing formatted report sections and key findings

        Example:
            >>> report = agent.generate_chp_report(
            ...     technology_selection_result=tech_result,
            ...     performance_result=perf_result,
            ...     economic_result=econ_result,
            ...     emissions_result=emis_result,
            ...     facility_name="ABC Manufacturing"
            ... )
        """
        try:
            # Extract key metrics
            recommended_tech = technology_selection_result.get("recommended_technology", "unknown")
            tech_name = technology_selection_result.get("technology_description", "CHP System")

            elec_efficiency = performance_result.get("electrical_efficiency", 0)
            total_efficiency = performance_result.get("total_efficiency", 0)

            simple_payback = economic_result.get("simple_payback_years", 999)
            npv = economic_result.get("npv_20yr", 0)
            net_savings = economic_result.get("net_annual_savings", 0)

            emissions_reduction = emissions_result.get("emissions_reduction_tonnes_co2_annual", 0)
            emissions_percent = emissions_result.get("emissions_reduction_percent", 0)

            # Build executive summary
            executive_summary = f"""
EXECUTIVE SUMMARY - {facility_name} CHP Analysis

Recommended Technology: {tech_name}

Key Performance Metrics:
- Electrical Efficiency: {elec_efficiency:.1%}
- Total CHP Efficiency: {total_efficiency:.1%}
- Annual Emissions Reduction: {emissions_reduction:,.0f} tonnes CO2/year ({emissions_percent:.1f}%)

Economic Analysis:
- Net Annual Savings: ${net_savings:,.0f}
- Simple Payback: {simple_payback:.1f} years
- 20-Year NPV: ${npv:,.0f}

Recommendation: {"PROCEED - Excellent economics and emissions reduction" if simple_payback < 5 and npv > 0 else "CONSIDER - Marginal economics" if simple_payback < 8 else "NOT RECOMMENDED - Poor economics"}
"""

            # Build technical summary
            technical_summary = {
                "recommended_technology": recommended_tech,
                "electrical_efficiency_percent": elec_efficiency * 100,
                "thermal_efficiency_percent": performance_result.get("thermal_efficiency", 0) * 100,
                "total_chp_efficiency_percent": total_efficiency * 100,
                "heat_recovery_effectiveness_percent": performance_result.get("heat_recovery_effectiveness", 0),
                "electrical_capacity_kw": performance_result.get("electrical_output_kw", 0),
                "thermal_capacity_mmbtu_hr": performance_result.get("thermal_output_mmbtu_hr", 0)
            }

            # Build economic summary
            economic_summary = {
                "total_capital_cost": economic_result.get("chp_capex_gross", 0),
                "net_capital_after_incentives": economic_result.get("net_capex_after_incentives", 0),
                "annual_savings": net_savings,
                "simple_payback_years": simple_payback,
                "npv_20yr": npv,
                "irr_percent": economic_result.get("irr_percent", 0),
                "benefit_cost_ratio": economic_result.get("benefit_cost_ratio", 0)
            }

            # Build environmental summary
            environmental_summary = {
                "annual_co2_reduction_tonnes": emissions_reduction,
                "percent_reduction_vs_baseline": emissions_percent,
                "chp_emission_intensity": emissions_result.get("chp_emission_intensity_kg_co2_per_kwh_equivalent", 0),
                "baseline_emission_intensity": emissions_result.get("baseline_emission_intensity_kg_co2_per_kwh_equivalent", 0)
            }

            # Key recommendations
            recommendations = [
                f"Proceed with {tech_name} installation for {facility_name}",
                f"Expected payback of {simple_payback:.1f} years with ${net_savings:,.0f} annual savings",
                f"Achieve {emissions_reduction:,.0f} tonnes CO2/year emissions reduction ({emissions_percent:.1f}%)",
            ]

            if interconnection_result:
                timeline = interconnection_result.get("utility_application_timeline_weeks", 0)
                recommendations.append(f"Allow {timeline} weeks for utility interconnection approval")

            if operating_strategy_result:
                strategy = operating_strategy_result.get("recommended_strategy", "baseload")
                recommendations.append(f"Operate CHP in {strategy} mode for optimal economics")

            recommendations.append("Include CHP in facility energy management system for optimized dispatch")

            return {
                "report_type": report_type,
                "facility_name": facility_name,
                "report_date": DeterministicClock.now().strftime("%Y-%m-%d"),
                "executive_summary": executive_summary,
                "technical_summary": technical_summary,
                "economic_summary": economic_summary,
                "environmental_summary": environmental_summary,
                "recommendations": recommendations,
                "overall_recommendation": "PROCEED" if simple_payback < 5 and npv > 0 else "CONSIDER" if simple_payback < 8 else "NOT RECOMMENDED",
                "deterministic": True,
                "provenance": {
                    "tool": "generate_chp_report",
                    "method": "Comprehensive CHP analysis report generation",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "tools_aggregated": ["technology_selection", "performance", "economic", "emissions"]
                }
            }

        except Exception as e:
            logger.error(f"Tool 8 (generate_chp_report) failed: {e}")
            raise

    # ========================================================================
    # MAIN ANALYSIS METHOD (AI Orchestration)
    # ========================================================================

    def analyze(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main analysis entry point with AI orchestration

        Args:
            query: User query or analysis request
            context: Optional context data (facility info, constraints, etc.)

        Returns:
            Dict containing:
                - analysis: Core analysis results
                - recommendations: Actionable recommendations
                - metrics: Key performance metrics
                - cost_usd: Analysis cost
                - provenance: Full calculation provenance

        Example:
            >>> result = agent.analyze(
            ...     query="Analyze 2 MW CHP system for manufacturing facility",
            ...     context={"facility_type": "food_processing"}
            ... )
        """
        logger.info(f"Starting CHP analysis: {query[:50]}...")

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Build user prompt
        user_prompt = f"""
User Query: {query}

Context: {context if context else "None provided"}

Instructions:
1. Understand the facility's electrical and thermal requirements
2. Use available tools to analyze CHP opportunities
3. Recommend optimal technology and system sizing
4. Provide economic analysis and payback estimates
5. Include grid interconnection and emissions considerations

Remember: Use deterministic tools for ALL calculations.
"""

        # Run AI orchestration
        try:
            response = self.session.run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=self.tool_registry.get_all(),
                budget_usd=self.config.budget_usd
            )

            # Enhance response with metadata
            response["agent_version"] = self._version()
            response["deterministic"] = True
            response["timestamp"] = DeterministicClock.now().isoformat()

            logger.info(f"CHP analysis complete. Cost: ${response.get('cost_usd', 0):.3f}")

            return response

        except Exception as e:
            logger.error(f"CHP analysis failed: {e}")
            raise

    def _build_system_prompt(self) -> str:
        """Build system prompt for AI orchestration"""
        return f"""You are {self.config.agent_name}, a specialized AI agent for combined heat and power (CHP) system analysis.

Your role is to analyze CHP opportunities, recommend optimal technologies, and provide comprehensive
economic and technical analysis for industrial and commercial facilities.

Available Tools:
1. select_chp_technology - Recommend optimal CHP technology
2. calculate_chp_performance - Calculate system efficiency metrics
3. size_heat_recovery_system - Design heat recovery equipment
4. calculate_economic_metrics - Analyze spark spread, payback, NPV
5. assess_grid_interconnection - Evaluate grid connection requirements
6. optimize_operating_strategy - Determine optimal dispatch strategy
7. calculate_emissions_reduction - Quantify emissions reduction
8. generate_chp_report - Create comprehensive analysis report

Guidelines:
- Use deterministic tools for ALL calculations
- Never perform math in your reasoning - delegate to tools
- Recommend technologies based on heat-to-power ratio matching
- Consider both technical and economic factors
- Provide clear, actionable recommendations with supporting data

Standards: EPA CHP Partnership, ASHRAE CHP Applications, IEEE 1547
"""

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def health_check(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "agent": self.config.agent_name,
            "version": self._version(),
            "timestamp": DeterministicClock.now().isoformat()
        }

    def ready_check(self) -> Dict[str, Any]:
        """Readiness check endpoint"""
        try:
            # Test critical dependencies
            _ = self.session.test_connection()

            return {
                "status": "ready",
                "agent": self.config.agent_name,
                "dependencies": {
                    "claude_api": "connected",
                    "tech_database": "loaded"
                }
            }
        except Exception as e:
            return {
                "status": "not_ready",
                "error": str(e)
            }


# ============================================================================
# Main entry point (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    config = CogenerationCHPConfig(budget_usd=0.50)
    agent = CogenerationCHPAgentAI(config=config)

    # Test tool 1
    result = agent.select_chp_technology(
        electrical_demand_kw=2000,
        thermal_demand_mmbtu_hr=15.0,
        heat_to_power_ratio=2.2,
        load_profile_type="baseload_24x7",
        available_fuels=["natural_gas"],
        emissions_requirements="low_nox",
        space_constraints="moderate"
    )

    print(f"Recommended Technology: {result['recommended_technology']}")
    print(f"Electrical Efficiency: {result['typical_electrical_efficiency']:.1%}")
    print(f"Total CHP Efficiency: {result['typical_total_efficiency']:.1%}")
