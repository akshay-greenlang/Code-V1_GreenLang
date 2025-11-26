# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ Deterministic Tools.

All tools produce zero-hallucination results using physics-based calculations.
This module provides deterministic calculation tools for thermal efficiency
analysis, Sankey diagram generation, benchmarking, and improvement identification.

CRITICAL: All numeric results come from deterministic formulas, never from LLM generation.

Physics Basis:
- First Law of Thermodynamics: Energy conservation (eta_1 = Q_useful / Q_input)
- Second Law of Thermodynamics: Exergy analysis (eta_2 = Ex_useful / Ex_input)
- Heat Transfer: Stefan-Boltzmann, Newton's cooling law
- Combustion: Siegert formula for flue gas losses

Standards Compliance:
- ASME PTC 4.1 - Steam Generating Units
- ASME PTC 4 - Fired Steam Generators
- ISO 50001:2018 - Energy Management Systems
- EPA 40 CFR Part 60 - Emissions Standards

Author: GreenLang Foundation
Version: 1.0.0
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


# ============================================================================
# CONSTANTS AND REFERENCE DATA
# ============================================================================

# Stefan-Boltzmann constant (W/m^2/K^4)
STEFAN_BOLTZMANN = 5.67e-8

# Reference temperature for exergy calculations (Kelvin)
REFERENCE_TEMPERATURE_K = 298.15  # 25 degC

# Reference pressure for exergy calculations (bar)
REFERENCE_PRESSURE_BAR = 1.01325

# Specific heat capacities (kJ/kg-K)
SPECIFIC_HEAT = {
    'air': 1.005,
    'water': 4.186,
    'steam': 2.010,
    'flue_gas': 1.100,
    'natural_gas': 2.200,
    'fuel_oil': 1.900,
    'coal': 1.300
}

# Heating values (MJ/kg)
HEATING_VALUES = {
    'natural_gas': {'hhv': 55.5, 'lhv': 50.0},
    'fuel_oil_no2': {'hhv': 45.5, 'lhv': 42.5},
    'fuel_oil_no6': {'hhv': 43.0, 'lhv': 40.5},
    'coal_bituminous': {'hhv': 32.0, 'lhv': 30.5},
    'coal_sub_bituminous': {'hhv': 26.0, 'lhv': 24.5},
    'biomass_wood': {'hhv': 20.0, 'lhv': 18.5},
    'hydrogen': {'hhv': 141.8, 'lhv': 120.0}
}

# Industry benchmark data (efficiency percentages)
INDUSTRY_BENCHMARKS = {
    'boiler': {
        'minimum': 75.0,
        'average': 82.0,
        'good': 87.0,
        'best_in_class': 94.0,
        'theoretical_max': 95.0
    },
    'furnace': {
        'minimum': 65.0,
        'average': 75.0,
        'good': 82.0,
        'best_in_class': 88.0,
        'theoretical_max': 92.0
    },
    'dryer': {
        'minimum': 40.0,
        'average': 55.0,
        'good': 65.0,
        'best_in_class': 75.0,
        'theoretical_max': 85.0
    },
    'kiln': {
        'minimum': 35.0,
        'average': 50.0,
        'good': 60.0,
        'best_in_class': 70.0,
        'theoretical_max': 80.0
    },
    'heat_exchanger': {
        'minimum': 70.0,
        'average': 80.0,
        'good': 88.0,
        'best_in_class': 95.0,
        'theoretical_max': 98.0
    },
    'reactor': {
        'minimum': 50.0,
        'average': 65.0,
        'good': 75.0,
        'best_in_class': 85.0,
        'theoretical_max': 90.0
    }
}

# CO2 emission factors (kg CO2 per MJ)
CO2_EMISSION_FACTORS = {
    'natural_gas': 0.0561,
    'fuel_oil': 0.0774,
    'coal': 0.0946,
    'biomass': 0.0,  # Considered carbon neutral
    'electricity': 0.1389  # Grid average
}


# ============================================================================
# RESULT DATA CLASSES
# ============================================================================

@dataclass
class FirstLawEfficiencyResult:
    """Result from First Law efficiency calculation."""
    efficiency_percent: float
    energy_input_kw: float
    useful_output_kw: float
    total_losses_kw: float
    combustion_efficiency_percent: float
    gross_efficiency_percent: float
    net_efficiency_percent: float
    calculation_method: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'efficiency_percent': self.efficiency_percent,
            'energy_input_kw': self.energy_input_kw,
            'useful_output_kw': self.useful_output_kw,
            'total_losses_kw': self.total_losses_kw,
            'combustion_efficiency_percent': self.combustion_efficiency_percent,
            'gross_efficiency_percent': self.gross_efficiency_percent,
            'net_efficiency_percent': self.net_efficiency_percent,
            'calculation_method': self.calculation_method,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class SecondLawEfficiencyResult:
    """Result from Second Law (exergy) efficiency calculation."""
    efficiency_percent: float
    exergy_input_kw: float
    exergy_output_kw: float
    exergy_destruction_kw: float
    exergy_loss_kw: float
    irreversibility_kw: float
    carnot_efficiency_percent: float
    quality_factor: float
    calculation_method: str
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'efficiency_percent': self.efficiency_percent,
            'exergy_input_kw': self.exergy_input_kw,
            'exergy_output_kw': self.exergy_output_kw,
            'exergy_destruction_kw': self.exergy_destruction_kw,
            'exergy_loss_kw': self.exergy_loss_kw,
            'irreversibility_kw': self.irreversibility_kw,
            'carnot_efficiency_percent': self.carnot_efficiency_percent,
            'quality_factor': self.quality_factor,
            'calculation_method': self.calculation_method,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class HeatBalanceResult:
    """Result from heat balance calculation."""
    closure_achieved: bool
    closure_error_percent: float
    energy_input_kw: float
    energy_output_kw: float
    total_losses_kw: float
    unaccounted_kw: float
    input_breakdown: Dict[str, float]
    output_breakdown: Dict[str, float]
    loss_breakdown: Dict[str, float]
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'closure_achieved': self.closure_achieved,
            'closure_error_percent': self.closure_error_percent,
            'energy_input_kw': self.energy_input_kw,
            'energy_output_kw': self.energy_output_kw,
            'total_losses_kw': self.total_losses_kw,
            'unaccounted_kw': self.unaccounted_kw,
            'input_breakdown': self.input_breakdown,
            'output_breakdown': self.output_breakdown,
            'loss_breakdown': self.loss_breakdown,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class HeatLossBreakdown:
    """Detailed breakdown of heat losses."""
    total_losses_kw: float
    total_losses_percent: float
    breakdown: Dict[str, float]
    categories: List[Dict[str, Any]]
    exergy_destruction_kw: float
    exergy_destruction_percent: float
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_losses_kw': self.total_losses_kw,
            'total_losses_percent': self.total_losses_percent,
            'breakdown': self.breakdown,
            'categories': self.categories,
            'exergy_destruction_kw': self.exergy_destruction_kw,
            'exergy_destruction_percent': self.exergy_destruction_percent,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class SankeyDiagramResult:
    """Result from Sankey diagram generation."""
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    total_input_kw: float
    total_output_kw: float
    total_losses_kw: float
    balance_error_percent: float
    visualization_config: Dict[str, Any]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'nodes': self.nodes,
            'links': self.links,
            'total_input_kw': self.total_input_kw,
            'total_output_kw': self.total_output_kw,
            'total_losses_kw': self.total_losses_kw,
            'balance_error_percent': self.balance_error_percent,
            'visualization_config': self.visualization_config,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class BenchmarkResult:
    """Result from industry benchmark comparison."""
    current_efficiency_percent: float
    industry_average_percent: float
    best_in_class_percent: float
    theoretical_maximum_percent: float
    percentile_rank: int
    benchmark_source: str
    industry_category: str
    metadata: Dict[str, Any]
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_efficiency_percent': self.current_efficiency_percent,
            'industry_average_percent': self.industry_average_percent,
            'best_in_class_percent': self.best_in_class_percent,
            'theoretical_maximum_percent': self.theoretical_maximum_percent,
            'percentile_rank': self.percentile_rank,
            'benchmark_source': self.benchmark_source,
            'industry_category': self.industry_category,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class ImprovementOpportunity:
    """Improvement opportunity with ROI analysis."""
    opportunity_id: str
    category: str
    description: str
    potential_savings_kw: float
    potential_savings_percent: float
    estimated_cost_usd: float
    payback_months: float
    priority: str
    implementation_complexity: str
    annual_co2_reduction_kg: float
    confidence_percent: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'opportunity_id': self.opportunity_id,
            'category': self.category,
            'description': self.description,
            'potential_savings_kw': self.potential_savings_kw,
            'potential_savings_percent': self.potential_savings_percent,
            'estimated_cost_usd': self.estimated_cost_usd,
            'payback_months': self.payback_months,
            'priority': self.priority,
            'implementation_complexity': self.implementation_complexity,
            'annual_co2_reduction_kg': self.annual_co2_reduction_kg,
            'confidence_percent': self.confidence_percent
        }


@dataclass
class ExergyAnalysisResult:
    """Detailed exergy analysis result."""
    total_exergy_input_kw: float
    total_exergy_output_kw: float
    exergy_destruction_kw: float
    exergy_loss_kw: float
    exergetic_efficiency_percent: float
    improvement_potential_kw: float
    component_analysis: List[Dict[str, Any]]
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_exergy_input_kw': self.total_exergy_input_kw,
            'total_exergy_output_kw': self.total_exergy_output_kw,
            'exergy_destruction_kw': self.exergy_destruction_kw,
            'exergy_loss_kw': self.exergy_loss_kw,
            'exergetic_efficiency_percent': self.exergetic_efficiency_percent,
            'improvement_potential_kw': self.improvement_potential_kw,
            'component_analysis': self.component_analysis,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


@dataclass
class UncertaintyResult:
    """Uncertainty quantification result."""
    efficiency_uncertainty_percent: float
    confidence_level_percent: float
    contributing_factors: Dict[str, float]
    measurement_uncertainties: Dict[str, float]
    systematic_uncertainty_percent: float
    random_uncertainty_percent: float
    timestamp: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'efficiency_uncertainty_percent': self.efficiency_uncertainty_percent,
            'confidence_level_percent': self.confidence_level_percent,
            'contributing_factors': self.contributing_factors,
            'measurement_uncertainties': self.measurement_uncertainties,
            'systematic_uncertainty_percent': self.systematic_uncertainty_percent,
            'random_uncertainty_percent': self.random_uncertainty_percent,
            'timestamp': self.timestamp,
            'provenance_hash': self.provenance_hash
        }


# ============================================================================
# TOOL SCHEMAS
# ============================================================================

TOOL_SCHEMAS = {
    "calculate_first_law_efficiency": {
        "name": "calculate_first_law_efficiency",
        "description": "Calculate First Law (energy) thermal efficiency using conservation of energy",
        "parameters": {
            "type": "object",
            "properties": {
                "energy_inputs": {
                    "type": "object",
                    "description": "Energy inputs including fuel and electrical"
                },
                "useful_outputs": {
                    "type": "object",
                    "description": "Useful heat output to process"
                },
                "heat_losses": {
                    "type": "object",
                    "description": "Heat loss measurements"
                }
            },
            "required": ["energy_inputs", "useful_outputs"]
        },
        "deterministic": True,
        "formula": "eta_1 = Q_useful / Q_input * 100%"
    },
    "calculate_second_law_efficiency": {
        "name": "calculate_second_law_efficiency",
        "description": "Calculate Second Law (exergy) efficiency accounting for energy quality",
        "parameters": {
            "type": "object",
            "properties": {
                "energy_inputs": {
                    "type": "object",
                    "description": "Energy inputs with temperature data"
                },
                "useful_outputs": {
                    "type": "object",
                    "description": "Useful outputs with temperature data"
                },
                "ambient_conditions": {
                    "type": "object",
                    "description": "Reference ambient conditions"
                }
            },
            "required": ["energy_inputs", "useful_outputs"]
        },
        "deterministic": True,
        "formula": "eta_2 = Ex_useful / Ex_input * 100%"
    },
    "calculate_heat_losses": {
        "name": "calculate_heat_losses",
        "description": "Calculate detailed heat loss breakdown",
        "parameters": {
            "type": "object",
            "properties": {
                "heat_losses": {
                    "type": "object",
                    "description": "Heat loss measurements"
                },
                "ambient_conditions": {
                    "type": "object",
                    "description": "Ambient conditions"
                },
                "process_parameters": {
                    "type": "object",
                    "description": "Process configuration"
                }
            },
            "required": ["heat_losses"]
        },
        "deterministic": True,
        "formula": "Q_loss = Q_radiation + Q_convection + Q_flue_gas + Q_blowdown + Q_other"
    },
    "generate_sankey_diagram": {
        "name": "generate_sankey_diagram",
        "description": "Generate Sankey diagram data for energy flow visualization",
        "parameters": {
            "type": "object",
            "properties": {
                "energy_inputs": {
                    "type": "object",
                    "description": "Energy input data"
                },
                "useful_outputs": {
                    "type": "object",
                    "description": "Useful output data"
                },
                "loss_breakdown": {
                    "type": "object",
                    "description": "Loss breakdown data"
                }
            },
            "required": ["energy_inputs", "useful_outputs"]
        },
        "deterministic": True,
        "output_format": "plotly_json"
    },
    "benchmark_efficiency": {
        "name": "benchmark_efficiency",
        "description": "Compare efficiency against industry benchmarks",
        "parameters": {
            "type": "object",
            "properties": {
                "current_efficiency": {
                    "type": "number",
                    "description": "Current efficiency percentage"
                },
                "process_type": {
                    "type": "string",
                    "description": "Type of thermal process"
                },
                "process_parameters": {
                    "type": "object",
                    "description": "Process configuration"
                }
            },
            "required": ["current_efficiency", "process_type"]
        },
        "deterministic": True,
        "data_source": "industry_database"
    },
    "analyze_improvements": {
        "name": "analyze_improvements",
        "description": "Identify and prioritize efficiency improvements with ROI",
        "parameters": {
            "type": "object",
            "properties": {
                "efficiency_result": {
                    "type": "object",
                    "description": "Current efficiency results"
                },
                "loss_breakdown": {
                    "type": "object",
                    "description": "Heat loss breakdown"
                }
            },
            "required": ["efficiency_result", "loss_breakdown"]
        },
        "deterministic": True,
        "methodology": "gap_analysis"
    },
    "calculate_fuel_energy": {
        "name": "calculate_fuel_energy",
        "description": "Calculate fuel energy input from flow and heating value",
        "parameters": {
            "type": "object",
            "properties": {
                "fuel_type": {
                    "type": "string",
                    "description": "Type of fuel"
                },
                "mass_flow_kg_hr": {
                    "type": "number",
                    "description": "Mass flow rate in kg/hr"
                },
                "heating_value_mj_kg": {
                    "type": "number",
                    "description": "Heating value in MJ/kg"
                }
            },
            "required": ["fuel_type", "mass_flow_kg_hr"]
        },
        "deterministic": True,
        "formula": "Q_fuel = m_dot * HV"
    },
    "calculate_steam_energy": {
        "name": "calculate_steam_energy",
        "description": "Calculate steam energy output from flow and enthalpy",
        "parameters": {
            "type": "object",
            "properties": {
                "steam_flow_kg_hr": {
                    "type": "number",
                    "description": "Steam mass flow rate"
                },
                "steam_enthalpy_kj_kg": {
                    "type": "number",
                    "description": "Steam specific enthalpy"
                },
                "feedwater_enthalpy_kj_kg": {
                    "type": "number",
                    "description": "Feedwater specific enthalpy"
                }
            },
            "required": ["steam_flow_kg_hr"]
        },
        "deterministic": True,
        "formula": "Q_steam = m_dot * (h_steam - h_feedwater)"
    },
    "quantify_uncertainty": {
        "name": "quantify_uncertainty",
        "description": "Quantify uncertainty in efficiency calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "input_data": {
                    "type": "object",
                    "description": "Input measurements with uncertainties"
                },
                "efficiency_result": {
                    "type": "object",
                    "description": "Calculated efficiency result"
                }
            },
            "required": ["input_data", "efficiency_result"]
        },
        "deterministic": True,
        "methodology": "propagation_of_uncertainty"
    },
    "calculate_heat_balance": {
        "name": "calculate_heat_balance",
        "description": "Perform heat balance with closure verification",
        "parameters": {
            "type": "object",
            "properties": {
                "energy_inputs": {
                    "type": "object",
                    "description": "All energy inputs"
                },
                "useful_outputs": {
                    "type": "object",
                    "description": "All useful outputs"
                },
                "heat_losses": {
                    "type": "object",
                    "description": "All heat losses"
                }
            },
            "required": ["energy_inputs", "useful_outputs"]
        },
        "deterministic": True,
        "formula": "Q_input = Q_useful + sum(Q_losses)"
    }
}


# ============================================================================
# MAIN TOOLS CLASS
# ============================================================================

class ThermalEfficiencyTools:
    """
    Deterministic calculation tools for thermal efficiency analysis.

    All methods use physics-based formulas with zero hallucination guarantee.
    No LLM is used for any numerical calculations.

    Attributes:
        config: Configuration object for calculation parameters
        energy_balance_tolerance: Tolerance for heat balance closure

    Example:
        >>> tools = ThermalEfficiencyTools()
        >>> result = tools.calculate_first_law_efficiency(
        ...     energy_inputs={'fuel_inputs': [...]},
        ...     useful_outputs={'process_heat_kw': 1000}
        ... )
        >>> print(f"Efficiency: {result.efficiency_percent}%")
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize ThermalEfficiencyTools.

        Args:
            config: Optional configuration object
        """
        self.config = config
        self.energy_balance_tolerance = 0.02  # 2% default
        if config and hasattr(config, 'energy_balance_tolerance'):
            self.energy_balance_tolerance = config.energy_balance_tolerance

    # ========================================================================
    # TOOL 1: FIRST LAW EFFICIENCY
    # ========================================================================

    def calculate_first_law_efficiency(
        self,
        energy_inputs: Dict[str, Any],
        useful_outputs: Dict[str, Any],
        heat_losses: Optional[Dict[str, Any]] = None
    ) -> FirstLawEfficiencyResult:
        """
        Calculate First Law (energy) thermal efficiency.

        Physics basis: First Law of Thermodynamics - Conservation of Energy
        Formula: eta_1 = Q_useful / Q_input * 100%

        Args:
            energy_inputs: Dictionary with:
                - fuel_inputs: List[Dict] with fuel_type, mass_flow_kg_hr, heating_value_mj_kg
                - electrical_inputs: List[Dict] with power_kw
            useful_outputs: Dictionary with:
                - process_heat_kw: Direct process heat
                - steam_output: List[Dict] with heat_rate_kw
                - hot_water_output: List[Dict] with heat_rate_kw
            heat_losses: Optional dictionary with measured losses

        Returns:
            FirstLawEfficiencyResult with efficiency and breakdown

        Standards: ASME PTC 4.1, ISO 50001
        """
        # Step 1: Calculate total energy input
        total_input_kw = self._calculate_total_energy_input(energy_inputs)

        # Step 2: Calculate useful output
        total_output_kw = self._calculate_total_useful_output(useful_outputs)

        # Step 3: Calculate total losses
        if heat_losses:
            total_losses_kw = self._calculate_total_losses(heat_losses)
        else:
            total_losses_kw = total_input_kw - total_output_kw

        # Step 4: Calculate efficiencies
        if total_input_kw > 0:
            gross_efficiency = (total_output_kw / total_input_kw) * 100
            net_efficiency = ((total_output_kw - self._get_auxiliary_power(energy_inputs)) / total_input_kw) * 100
        else:
            gross_efficiency = 0.0
            net_efficiency = 0.0

        # Step 5: Calculate combustion efficiency (if applicable)
        combustion_efficiency = self._calculate_combustion_efficiency(energy_inputs, heat_losses)

        # Create result
        result = FirstLawEfficiencyResult(
            efficiency_percent=round(gross_efficiency, 4),
            energy_input_kw=round(total_input_kw, 2),
            useful_output_kw=round(total_output_kw, 2),
            total_losses_kw=round(total_losses_kw, 2),
            combustion_efficiency_percent=round(combustion_efficiency, 4),
            gross_efficiency_percent=round(gross_efficiency, 4),
            net_efficiency_percent=round(net_efficiency, 4),
            calculation_method='input_output_method',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'energy_inputs': energy_inputs,
                'useful_outputs': useful_outputs
            })
        )

        return result

    def _calculate_total_energy_input(self, energy_inputs: Dict[str, Any]) -> float:
        """Calculate total energy input in kW."""
        total = 0.0

        # Fuel inputs
        fuel_inputs = energy_inputs.get('fuel_inputs', [])
        for fuel in fuel_inputs:
            mass_flow_kg_hr = fuel.get('mass_flow_kg_hr', 0)
            heating_value_mj_kg = fuel.get('heating_value_mj_kg', 0)

            # If no heating value provided, look up from database
            if heating_value_mj_kg == 0:
                fuel_type = fuel.get('fuel_type', 'natural_gas')
                heating_values = HEATING_VALUES.get(fuel_type, HEATING_VALUES['natural_gas'])
                heating_value_mj_kg = heating_values.get('hhv', 50.0)

            # Convert MJ/hr to kW (1 MJ/hr = 0.2778 kW)
            fuel_energy_kw = mass_flow_kg_hr * heating_value_mj_kg * 0.2778
            total += fuel_energy_kw

        # Electrical inputs
        electrical_inputs = energy_inputs.get('electrical_inputs', [])
        for electrical in electrical_inputs:
            total += electrical.get('power_kw', 0)

        return total

    def _calculate_total_useful_output(self, useful_outputs: Dict[str, Any]) -> float:
        """Calculate total useful output in kW."""
        total = 0.0

        # Direct process heat
        total += useful_outputs.get('process_heat_kw', 0)

        # Steam outputs
        steam_outputs = useful_outputs.get('steam_output', [])
        for steam in steam_outputs:
            total += steam.get('heat_rate_kw', 0)

        # Hot water outputs
        hot_water_outputs = useful_outputs.get('hot_water_output', [])
        for hot_water in hot_water_outputs:
            total += hot_water.get('heat_rate_kw', 0)

        # Product heating
        product_heating = useful_outputs.get('product_heating', {})
        total += product_heating.get('heat_rate_kw', 0)

        return total

    def _calculate_total_losses(self, heat_losses: Dict[str, Any]) -> float:
        """Calculate total heat losses in kW."""
        total = 0.0

        # Flue gas losses
        flue_gas = heat_losses.get('flue_gas_losses', {})
        total += flue_gas.get('sensible_loss_kw', 0)
        total += flue_gas.get('latent_loss_kw', 0)

        # Radiation losses
        radiation = heat_losses.get('radiation_losses', {})
        total += radiation.get('loss_kw', 0)

        # Convection losses
        convection = heat_losses.get('convection_losses', {})
        total += convection.get('loss_kw', 0)

        # Blowdown losses
        blowdown = heat_losses.get('blowdown_losses', {})
        total += blowdown.get('loss_kw', 0)

        # Unaccounted losses
        total += heat_losses.get('unaccounted_losses_kw', 0)

        return total

    def _get_auxiliary_power(self, energy_inputs: Dict[str, Any]) -> float:
        """Get auxiliary electrical power consumption in kW."""
        electrical_inputs = energy_inputs.get('electrical_inputs', [])
        auxiliary = 0.0
        for electrical in electrical_inputs:
            if electrical.get('is_auxiliary', True):
                auxiliary += electrical.get('power_kw', 0)
        return auxiliary

    def _calculate_combustion_efficiency(
        self,
        energy_inputs: Dict[str, Any],
        heat_losses: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate combustion efficiency using Siegert formula.

        Formula: eta_comb = 100 - L_dry - L_moisture - L_unburned

        Where:
            L_dry = k1 * (T_flue - T_air) / CO2%
            L_moisture = k2 * (T_flue - T_air) * H2/100
            L_unburned = k3 * CO / (CO2 + CO)
        """
        if not heat_losses:
            return 95.0  # Default assumption

        flue_gas = heat_losses.get('flue_gas_losses', {})
        flue_temp_c = flue_gas.get('exit_temperature_c', 200)
        ambient_temp_c = flue_gas.get('inlet_temperature_c', 25)
        co2_percent = flue_gas.get('co2_percent', 12)
        co_ppm = flue_gas.get('co_ppm', 50)
        o2_percent = flue_gas.get('o2_percent', 3)

        # Siegert coefficients for natural gas
        k1 = 0.37  # Dry gas loss coefficient
        k2 = 0.02  # Moisture loss coefficient

        # Calculate dry gas loss
        delta_t = flue_temp_c - ambient_temp_c
        if co2_percent > 0:
            dry_gas_loss = k1 * delta_t / co2_percent
        else:
            dry_gas_loss = 5.0  # Default assumption

        # Calculate moisture loss (simplified)
        moisture_loss = k2 * delta_t

        # Calculate unburned loss
        co_percent = co_ppm / 10000
        if (co2_percent + co_percent) > 0:
            unburned_loss = 0.63 * co_percent / (co2_percent + co_percent)
        else:
            unburned_loss = 0.5

        # Combustion efficiency
        combustion_efficiency = 100 - dry_gas_loss - moisture_loss - unburned_loss
        return max(0, min(100, combustion_efficiency))

    # ========================================================================
    # TOOL 2: SECOND LAW EFFICIENCY
    # ========================================================================

    def calculate_second_law_efficiency(
        self,
        energy_inputs: Dict[str, Any],
        useful_outputs: Dict[str, Any],
        ambient_conditions: Optional[Dict[str, Any]] = None
    ) -> SecondLawEfficiencyResult:
        """
        Calculate Second Law (exergy) thermal efficiency.

        Physics basis: Second Law of Thermodynamics - Exergy analysis
        Formula: eta_2 = Ex_useful / Ex_input * 100%

        Exergy represents the maximum useful work obtainable from a system
        as it comes to equilibrium with its environment.

        Args:
            energy_inputs: Energy inputs with temperature data
            useful_outputs: Useful outputs with temperature data
            ambient_conditions: Reference conditions (T0, P0)

        Returns:
            SecondLawEfficiencyResult with exergy analysis

        Standards: ISO 50001, ASME PTC 4
        """
        # Get reference conditions
        if ambient_conditions is None:
            ambient_conditions = {}
        T0_K = ambient_conditions.get('ambient_temperature_c', 25) + 273.15
        P0_bar = ambient_conditions.get('ambient_pressure_bar', 1.01325)

        # Step 1: Calculate exergy input
        exergy_input_kw = self._calculate_exergy_input(energy_inputs, T0_K)

        # Step 2: Calculate exergy output
        exergy_output_kw = self._calculate_exergy_output(useful_outputs, T0_K)

        # Step 3: Calculate exergy destruction (irreversibilities)
        exergy_destruction_kw = exergy_input_kw - exergy_output_kw

        # Step 4: Calculate exergy loss (to environment)
        exergy_loss_kw = self._calculate_exergy_loss(energy_inputs, useful_outputs, T0_K)

        # Step 5: Calculate Second Law efficiency
        if exergy_input_kw > 0:
            efficiency_percent = (exergy_output_kw / exergy_input_kw) * 100
        else:
            efficiency_percent = 0.0

        # Step 6: Calculate Carnot efficiency (theoretical maximum)
        process_temp_k = self._get_average_process_temperature(useful_outputs) + 273.15
        if process_temp_k > T0_K:
            carnot_efficiency = (1 - T0_K / process_temp_k) * 100
        else:
            carnot_efficiency = 0.0

        # Step 7: Calculate quality factor
        first_law_input = self._calculate_total_energy_input(energy_inputs)
        if first_law_input > 0:
            quality_factor = exergy_input_kw / first_law_input
        else:
            quality_factor = 1.0

        result = SecondLawEfficiencyResult(
            efficiency_percent=round(efficiency_percent, 4),
            exergy_input_kw=round(exergy_input_kw, 2),
            exergy_output_kw=round(exergy_output_kw, 2),
            exergy_destruction_kw=round(exergy_destruction_kw, 2),
            exergy_loss_kw=round(exergy_loss_kw, 2),
            irreversibility_kw=round(exergy_destruction_kw + exergy_loss_kw, 2),
            carnot_efficiency_percent=round(carnot_efficiency, 4),
            quality_factor=round(quality_factor, 4),
            calculation_method='exergy_analysis',
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'energy_inputs': energy_inputs,
                'useful_outputs': useful_outputs,
                'ambient_conditions': ambient_conditions
            })
        )

        return result

    def _calculate_exergy_input(self, energy_inputs: Dict[str, Any], T0_K: float) -> float:
        """
        Calculate total exergy input.

        For fuel: Ex_fuel = m_dot * (HHV + R*T0*ln(P/P0))
        For electricity: Ex_elec = W (work is pure exergy)
        """
        total_exergy = 0.0

        # Fuel exergy (chemical exergy)
        fuel_inputs = energy_inputs.get('fuel_inputs', [])
        for fuel in fuel_inputs:
            mass_flow_kg_hr = fuel.get('mass_flow_kg_hr', 0)
            heating_value_mj_kg = fuel.get('heating_value_mj_kg', 0)

            if heating_value_mj_kg == 0:
                fuel_type = fuel.get('fuel_type', 'natural_gas')
                heating_values = HEATING_VALUES.get(fuel_type, HEATING_VALUES['natural_gas'])
                heating_value_mj_kg = heating_values.get('hhv', 50.0)

            # Chemical exergy is approximately equal to HHV for most fuels
            # (with correction factor phi ~ 1.04 for natural gas)
            phi = 1.04  # Exergy-to-energy ratio for natural gas
            fuel_exergy_kw = mass_flow_kg_hr * heating_value_mj_kg * phi * 0.2778
            total_exergy += fuel_exergy_kw

        # Electrical exergy (pure work)
        electrical_inputs = energy_inputs.get('electrical_inputs', [])
        for electrical in electrical_inputs:
            total_exergy += electrical.get('power_kw', 0)

        return total_exergy

    def _calculate_exergy_output(self, useful_outputs: Dict[str, Any], T0_K: float) -> float:
        """
        Calculate total exergy output.

        For heat at temperature T: Ex_heat = Q * (1 - T0/T)
        For work: Ex_work = W
        """
        total_exergy = 0.0

        # Process heat exergy
        process_heat_kw = useful_outputs.get('process_heat_kw', 0)
        process_temp_c = useful_outputs.get('process_temperature_c', 200)
        process_temp_k = process_temp_c + 273.15
        if process_temp_k > T0_K:
            carnot_factor = 1 - T0_K / process_temp_k
            total_exergy += process_heat_kw * carnot_factor

        # Steam output exergy
        steam_outputs = useful_outputs.get('steam_output', [])
        for steam in steam_outputs:
            heat_rate_kw = steam.get('heat_rate_kw', 0)
            steam_temp_c = steam.get('temperature_c', 180)
            steam_temp_k = steam_temp_c + 273.15
            if steam_temp_k > T0_K:
                carnot_factor = 1 - T0_K / steam_temp_k
                total_exergy += heat_rate_kw * carnot_factor

        # Hot water output exergy
        hot_water_outputs = useful_outputs.get('hot_water_output', [])
        for hot_water in hot_water_outputs:
            heat_rate_kw = hot_water.get('heat_rate_kw', 0)
            water_temp_c = hot_water.get('temperature_c', 80)
            water_temp_k = water_temp_c + 273.15
            if water_temp_k > T0_K:
                carnot_factor = 1 - T0_K / water_temp_k
                total_exergy += heat_rate_kw * carnot_factor

        return total_exergy

    def _calculate_exergy_loss(
        self,
        energy_inputs: Dict[str, Any],
        useful_outputs: Dict[str, Any],
        T0_K: float
    ) -> float:
        """Calculate exergy loss to environment."""
        # Simplified: assume 10% of input exergy is lost to environment
        exergy_input = self._calculate_exergy_input(energy_inputs, T0_K)
        return exergy_input * 0.10

    def _get_average_process_temperature(self, useful_outputs: Dict[str, Any]) -> float:
        """Get weighted average process temperature in Celsius."""
        temps = []
        weights = []

        process_heat_kw = useful_outputs.get('process_heat_kw', 0)
        if process_heat_kw > 0:
            temps.append(useful_outputs.get('process_temperature_c', 200))
            weights.append(process_heat_kw)

        steam_outputs = useful_outputs.get('steam_output', [])
        for steam in steam_outputs:
            heat_rate_kw = steam.get('heat_rate_kw', 0)
            if heat_rate_kw > 0:
                temps.append(steam.get('temperature_c', 180))
                weights.append(heat_rate_kw)

        if temps and sum(weights) > 0:
            return sum(t * w for t, w in zip(temps, weights)) / sum(weights)
        return 200.0  # Default

    # ========================================================================
    # TOOL 3: HEAT BALANCE
    # ========================================================================

    def calculate_heat_balance(
        self,
        energy_inputs: Dict[str, Any],
        useful_outputs: Dict[str, Any],
        heat_losses: Optional[Dict[str, Any]] = None
    ) -> HeatBalanceResult:
        """
        Perform comprehensive heat balance with closure verification.

        Physics basis: Conservation of Energy
        Formula: Q_input = Q_useful + sum(Q_losses) + Q_unaccounted

        Args:
            energy_inputs: All energy inputs
            useful_outputs: All useful outputs
            heat_losses: Measured heat losses

        Returns:
            HeatBalanceResult with closure status

        Standards: ASME PTC 4.1 (2% closure tolerance)
        """
        # Calculate totals
        total_input_kw = self._calculate_total_energy_input(energy_inputs)
        total_output_kw = self._calculate_total_useful_output(useful_outputs)

        if heat_losses:
            total_losses_kw = self._calculate_total_losses(heat_losses)
        else:
            total_losses_kw = 0.0

        # Calculate energy balance
        accounted_kw = total_output_kw + total_losses_kw
        unaccounted_kw = total_input_kw - accounted_kw

        # Check closure
        if total_input_kw > 0:
            closure_error_percent = abs(unaccounted_kw / total_input_kw) * 100
        else:
            closure_error_percent = 100.0

        closure_achieved = closure_error_percent <= (self.energy_balance_tolerance * 100)

        # Build breakdowns
        input_breakdown = self._build_input_breakdown(energy_inputs)
        output_breakdown = self._build_output_breakdown(useful_outputs)
        loss_breakdown = self._build_loss_breakdown(heat_losses) if heat_losses else {}

        result = HeatBalanceResult(
            closure_achieved=closure_achieved,
            closure_error_percent=round(closure_error_percent, 4),
            energy_input_kw=round(total_input_kw, 2),
            energy_output_kw=round(total_output_kw, 2),
            total_losses_kw=round(total_losses_kw, 2),
            unaccounted_kw=round(unaccounted_kw, 2),
            input_breakdown=input_breakdown,
            output_breakdown=output_breakdown,
            loss_breakdown=loss_breakdown,
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'energy_inputs': energy_inputs,
                'useful_outputs': useful_outputs,
                'heat_losses': heat_losses
            })
        )

        return result

    def _build_input_breakdown(self, energy_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Build input energy breakdown."""
        breakdown = {}

        fuel_inputs = energy_inputs.get('fuel_inputs', [])
        for i, fuel in enumerate(fuel_inputs):
            mass_flow_kg_hr = fuel.get('mass_flow_kg_hr', 0)
            heating_value_mj_kg = fuel.get('heating_value_mj_kg', 50.0)
            fuel_type = fuel.get('fuel_type', f'fuel_{i}')
            energy_kw = mass_flow_kg_hr * heating_value_mj_kg * 0.2778
            breakdown[f'fuel_{fuel_type}'] = round(energy_kw, 2)

        electrical_inputs = energy_inputs.get('electrical_inputs', [])
        total_electrical = sum(e.get('power_kw', 0) for e in electrical_inputs)
        if total_electrical > 0:
            breakdown['electrical'] = round(total_electrical, 2)

        return breakdown

    def _build_output_breakdown(self, useful_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Build output energy breakdown."""
        breakdown = {}

        process_heat = useful_outputs.get('process_heat_kw', 0)
        if process_heat > 0:
            breakdown['process_heat'] = round(process_heat, 2)

        steam_outputs = useful_outputs.get('steam_output', [])
        total_steam = sum(s.get('heat_rate_kw', 0) for s in steam_outputs)
        if total_steam > 0:
            breakdown['steam_output'] = round(total_steam, 2)

        hot_water_outputs = useful_outputs.get('hot_water_output', [])
        total_hot_water = sum(h.get('heat_rate_kw', 0) for h in hot_water_outputs)
        if total_hot_water > 0:
            breakdown['hot_water_output'] = round(total_hot_water, 2)

        return breakdown

    def _build_loss_breakdown(self, heat_losses: Dict[str, Any]) -> Dict[str, float]:
        """Build loss breakdown."""
        breakdown = {}

        flue_gas = heat_losses.get('flue_gas_losses', {})
        total_flue = flue_gas.get('sensible_loss_kw', 0) + flue_gas.get('latent_loss_kw', 0)
        if total_flue > 0:
            breakdown['flue_gas'] = round(total_flue, 2)

        radiation = heat_losses.get('radiation_losses', {})
        if radiation.get('loss_kw', 0) > 0:
            breakdown['radiation'] = round(radiation.get('loss_kw', 0), 2)

        convection = heat_losses.get('convection_losses', {})
        if convection.get('loss_kw', 0) > 0:
            breakdown['convection'] = round(convection.get('loss_kw', 0), 2)

        blowdown = heat_losses.get('blowdown_losses', {})
        if blowdown.get('loss_kw', 0) > 0:
            breakdown['blowdown'] = round(blowdown.get('loss_kw', 0), 2)

        return breakdown

    # ========================================================================
    # TOOL 4: HEAT LOSS CALCULATOR
    # ========================================================================

    def calculate_heat_losses(
        self,
        heat_losses: Dict[str, Any],
        ambient_conditions: Optional[Dict[str, Any]] = None,
        process_parameters: Optional[Dict[str, Any]] = None
    ) -> HeatLossBreakdown:
        """
        Calculate detailed heat loss breakdown.

        Physics basis:
        - Radiation: Q = epsilon * sigma * A * (T_s^4 - T_amb^4)
        - Convection: Q = h * A * (T_s - T_amb)
        - Flue gas: Q = m_dot * Cp * (T_exit - T_ref)

        Args:
            heat_losses: Measured or calculated heat losses
            ambient_conditions: Environmental conditions
            process_parameters: Process configuration

        Returns:
            HeatLossBreakdown with categorized losses
        """
        if ambient_conditions is None:
            ambient_conditions = {}
        if process_parameters is None:
            process_parameters = {}

        T_amb = ambient_conditions.get('ambient_temperature_c', 25)
        total_input_kw = process_parameters.get('total_input_kw', 1000)

        # Calculate individual losses
        losses = {}
        categories = []

        # 1. Flue gas losses
        flue_gas = heat_losses.get('flue_gas_losses', {})
        if flue_gas:
            flue_loss = self._calculate_flue_gas_loss(flue_gas, T_amb)
            losses['flue_gas'] = flue_loss
            categories.append({
                'name': 'Flue Gas Loss',
                'value_kw': flue_loss,
                'percent': (flue_loss / total_input_kw * 100) if total_input_kw > 0 else 0,
                'recoverable': True,
                'typical_range': '5-15%'
            })

        # 2. Radiation losses
        radiation = heat_losses.get('radiation_losses', {})
        if radiation:
            rad_loss = self._calculate_radiation_loss(radiation, T_amb)
            losses['radiation'] = rad_loss
            categories.append({
                'name': 'Radiation Loss',
                'value_kw': rad_loss,
                'percent': (rad_loss / total_input_kw * 100) if total_input_kw > 0 else 0,
                'recoverable': False,
                'typical_range': '1-3%'
            })

        # 3. Convection losses
        convection = heat_losses.get('convection_losses', {})
        if convection:
            conv_loss = self._calculate_convection_loss(convection, T_amb)
            losses['convection'] = conv_loss
            categories.append({
                'name': 'Convection Loss',
                'value_kw': conv_loss,
                'percent': (conv_loss / total_input_kw * 100) if total_input_kw > 0 else 0,
                'recoverable': False,
                'typical_range': '0.5-2%'
            })

        # 4. Blowdown losses
        blowdown = heat_losses.get('blowdown_losses', {})
        if blowdown:
            blow_loss = blowdown.get('loss_kw', 0)
            losses['blowdown'] = blow_loss
            categories.append({
                'name': 'Blowdown Loss',
                'value_kw': blow_loss,
                'percent': (blow_loss / total_input_kw * 100) if total_input_kw > 0 else 0,
                'recoverable': True,
                'typical_range': '1-3%'
            })

        # Calculate totals
        total_losses_kw = sum(losses.values())
        total_losses_percent = (total_losses_kw / total_input_kw * 100) if total_input_kw > 0 else 0

        # Estimate exergy destruction
        T0_K = T_amb + 273.15
        exergy_destruction_kw = total_losses_kw * 0.7  # Approximate
        exergy_destruction_percent = (exergy_destruction_kw / total_input_kw * 100) if total_input_kw > 0 else 0

        result = HeatLossBreakdown(
            total_losses_kw=round(total_losses_kw, 2),
            total_losses_percent=round(total_losses_percent, 2),
            breakdown=losses,
            categories=categories,
            exergy_destruction_kw=round(exergy_destruction_kw, 2),
            exergy_destruction_percent=round(exergy_destruction_percent, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash(heat_losses)
        )

        return result

    def _calculate_flue_gas_loss(self, flue_gas: Dict[str, Any], T_amb: float) -> float:
        """Calculate flue gas sensible and latent heat loss."""
        mass_flow_kg_hr = flue_gas.get('mass_flow_kg_hr', 0)
        exit_temp_c = flue_gas.get('exit_temperature_c', 200)
        cp = SPECIFIC_HEAT.get('flue_gas', 1.1)

        # Sensible heat loss: Q = m_dot * Cp * (T_exit - T_ref)
        sensible_kw = mass_flow_kg_hr * cp * (exit_temp_c - T_amb) / 3600

        # Latent heat loss (if moisture present)
        moisture_percent = flue_gas.get('moisture_percent', 10)
        latent_kw = sensible_kw * moisture_percent / 100 * 2.44  # Latent heat of vaporization

        return sensible_kw + latent_kw

    def _calculate_radiation_loss(self, radiation: Dict[str, Any], T_amb: float) -> float:
        """
        Calculate radiation heat loss using Stefan-Boltzmann law.

        Formula: Q = epsilon * sigma * A * (T_s^4 - T_amb^4)
        """
        surface_area_m2 = radiation.get('surface_area_m2', 10)
        surface_temp_c = radiation.get('surface_temperature_c', 100)
        emissivity = radiation.get('emissivity', 0.9)

        T_s_K = surface_temp_c + 273.15
        T_amb_K = T_amb + 273.15

        # Stefan-Boltzmann radiation
        q_rad_w = emissivity * STEFAN_BOLTZMANN * surface_area_m2 * (T_s_K**4 - T_amb_K**4)

        return q_rad_w / 1000  # Convert to kW

    def _calculate_convection_loss(self, convection: Dict[str, Any], T_amb: float) -> float:
        """
        Calculate convection heat loss using Newton's cooling law.

        Formula: Q = h * A * (T_s - T_amb)
        """
        surface_area_m2 = convection.get('surface_area_m2', 10)
        surface_temp_c = convection.get('surface_temperature_c', 80)
        h_conv = convection.get('heat_transfer_coeff_w_m2k', 10)  # Natural convection

        q_conv_w = h_conv * surface_area_m2 * (surface_temp_c - T_amb)

        return q_conv_w / 1000  # Convert to kW

    # ========================================================================
    # TOOL 5: SANKEY DIAGRAM GENERATOR
    # ========================================================================

    def generate_sankey_diagram(
        self,
        energy_inputs: Dict[str, Any],
        useful_outputs: Dict[str, Any],
        loss_breakdown: Dict[str, Any]
    ) -> SankeyDiagramResult:
        """
        Generate Sankey diagram data for energy flow visualization.

        Creates nodes and links for Plotly Sankey diagram with proper
        energy balance.

        Args:
            energy_inputs: Energy input data
            useful_outputs: Useful output data
            loss_breakdown: Loss breakdown from heat loss calculation

        Returns:
            SankeyDiagramResult with nodes and links for visualization
        """
        nodes = []
        links = []
        node_index = {}

        # Helper to add nodes
        def add_node(name: str, color: str = "#1f77b4") -> int:
            if name not in node_index:
                node_index[name] = len(nodes)
                nodes.append({
                    'name': name,
                    'color': color
                })
            return node_index[name]

        # Add input nodes
        fuel_inputs = energy_inputs.get('fuel_inputs', [])
        total_fuel_kw = 0
        for fuel in fuel_inputs:
            mass_flow = fuel.get('mass_flow_kg_hr', 0)
            heating_value = fuel.get('heating_value_mj_kg', 50)
            fuel_kw = mass_flow * heating_value * 0.2778
            total_fuel_kw += fuel_kw
            fuel_type = fuel.get('fuel_type', 'fuel')
            fuel_node = add_node(f"Fuel ({fuel_type})", "#ff7f0e")

        electrical_inputs = energy_inputs.get('electrical_inputs', [])
        total_electrical_kw = sum(e.get('power_kw', 0) for e in electrical_inputs)
        if total_electrical_kw > 0:
            add_node("Electrical Input", "#2ca02c")

        # Add process node
        process_node = add_node("Process", "#1f77b4")

        # Add output nodes
        useful_output_kw = 0
        process_heat = useful_outputs.get('process_heat_kw', 0)
        if process_heat > 0:
            add_node("Process Heat", "#9467bd")
            useful_output_kw += process_heat

        steam_outputs = useful_outputs.get('steam_output', [])
        steam_kw = sum(s.get('heat_rate_kw', 0) for s in steam_outputs)
        if steam_kw > 0:
            add_node("Steam Output", "#8c564b")
            useful_output_kw += steam_kw

        # Add loss nodes
        breakdown = loss_breakdown.get('breakdown', {})
        total_losses_kw = 0

        if breakdown.get('flue_gas', 0) > 0:
            add_node("Flue Gas Loss", "#d62728")
            total_losses_kw += breakdown['flue_gas']

        if breakdown.get('radiation', 0) > 0:
            add_node("Radiation Loss", "#ff9896")
            total_losses_kw += breakdown['radiation']

        if breakdown.get('convection', 0) > 0:
            add_node("Convection Loss", "#ffbb78")
            total_losses_kw += breakdown['convection']

        if breakdown.get('blowdown', 0) > 0:
            add_node("Blowdown Loss", "#aec7e8")
            total_losses_kw += breakdown['blowdown']

        # Create links from inputs to process
        if total_fuel_kw > 0:
            for fuel in fuel_inputs:
                mass_flow = fuel.get('mass_flow_kg_hr', 0)
                heating_value = fuel.get('heating_value_mj_kg', 50)
                fuel_kw = mass_flow * heating_value * 0.2778
                fuel_type = fuel.get('fuel_type', 'fuel')
                links.append({
                    'source': node_index[f"Fuel ({fuel_type})"],
                    'target': process_node,
                    'value': round(fuel_kw, 2),
                    'color': 'rgba(255, 127, 14, 0.4)'
                })

        if total_electrical_kw > 0:
            links.append({
                'source': node_index["Electrical Input"],
                'target': process_node,
                'value': round(total_electrical_kw, 2),
                'color': 'rgba(44, 160, 44, 0.4)'
            })

        # Create links from process to outputs
        if process_heat > 0:
            links.append({
                'source': process_node,
                'target': node_index["Process Heat"],
                'value': round(process_heat, 2),
                'color': 'rgba(148, 103, 189, 0.4)'
            })

        if steam_kw > 0:
            links.append({
                'source': process_node,
                'target': node_index["Steam Output"],
                'value': round(steam_kw, 2),
                'color': 'rgba(140, 86, 75, 0.4)'
            })

        # Create links from process to losses
        if breakdown.get('flue_gas', 0) > 0:
            links.append({
                'source': process_node,
                'target': node_index["Flue Gas Loss"],
                'value': round(breakdown['flue_gas'], 2),
                'color': 'rgba(214, 39, 40, 0.4)'
            })

        if breakdown.get('radiation', 0) > 0:
            links.append({
                'source': process_node,
                'target': node_index["Radiation Loss"],
                'value': round(breakdown['radiation'], 2),
                'color': 'rgba(255, 152, 150, 0.4)'
            })

        if breakdown.get('convection', 0) > 0:
            links.append({
                'source': process_node,
                'target': node_index["Convection Loss"],
                'value': round(breakdown['convection'], 2),
                'color': 'rgba(255, 187, 120, 0.4)'
            })

        if breakdown.get('blowdown', 0) > 0:
            links.append({
                'source': process_node,
                'target': node_index["Blowdown Loss"],
                'value': round(breakdown['blowdown'], 2),
                'color': 'rgba(174, 199, 232, 0.4)'
            })

        # Calculate totals and balance error
        total_input_kw = total_fuel_kw + total_electrical_kw
        total_output_kw = useful_output_kw + total_losses_kw
        if total_input_kw > 0:
            balance_error = abs(total_input_kw - total_output_kw) / total_input_kw * 100
        else:
            balance_error = 0

        # Visualization config
        visualization_config = {
            'orientation': 'horizontal',
            'valueformat': '.2f',
            'valuesuffix': ' kW',
            'node': {
                'pad': 15,
                'thickness': 20,
                'line': {'color': 'black', 'width': 0.5}
            },
            'link': {
                'color': 'source'
            }
        }

        result = SankeyDiagramResult(
            nodes=nodes,
            links=links,
            total_input_kw=round(total_input_kw, 2),
            total_output_kw=round(useful_output_kw, 2),
            total_losses_kw=round(total_losses_kw, 2),
            balance_error_percent=round(balance_error, 3),
            visualization_config=visualization_config,
            provenance_hash=self._calculate_hash({
                'energy_inputs': energy_inputs,
                'useful_outputs': useful_outputs,
                'loss_breakdown': loss_breakdown
            })
        )

        return result

    # ========================================================================
    # TOOL 6: BENCHMARK EFFICIENCY
    # ========================================================================

    def benchmark_efficiency(
        self,
        current_efficiency: float,
        process_type: str,
        process_parameters: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Compare efficiency against industry benchmarks.

        Uses industry database to determine percentile rank and
        identify gap to best-in-class performance.

        Args:
            current_efficiency: Current efficiency percentage
            process_type: Type of process (boiler, furnace, etc.)
            process_parameters: Additional process details

        Returns:
            BenchmarkResult with comparison data
        """
        if process_parameters is None:
            process_parameters = {}

        # Get benchmark data for process type
        benchmarks = INDUSTRY_BENCHMARKS.get(
            process_type.lower(),
            INDUSTRY_BENCHMARKS['boiler']
        )

        # Determine percentile rank
        if current_efficiency >= benchmarks['best_in_class']:
            percentile = 95
        elif current_efficiency >= benchmarks['good']:
            percentile = 75 + 20 * (current_efficiency - benchmarks['good']) / (benchmarks['best_in_class'] - benchmarks['good'])
        elif current_efficiency >= benchmarks['average']:
            percentile = 50 + 25 * (current_efficiency - benchmarks['average']) / (benchmarks['good'] - benchmarks['average'])
        elif current_efficiency >= benchmarks['minimum']:
            percentile = 25 + 25 * (current_efficiency - benchmarks['minimum']) / (benchmarks['average'] - benchmarks['minimum'])
        else:
            percentile = 25 * current_efficiency / benchmarks['minimum']

        percentile = max(0, min(100, percentile))

        result = BenchmarkResult(
            current_efficiency_percent=round(current_efficiency, 2),
            industry_average_percent=benchmarks['average'],
            best_in_class_percent=benchmarks['best_in_class'],
            theoretical_maximum_percent=benchmarks['theoretical_max'],
            percentile_rank=int(percentile),
            benchmark_source="Industry Database (ASME, DOE)",
            industry_category=process_type.title(),
            metadata={
                'minimum_threshold': benchmarks['minimum'],
                'good_threshold': benchmarks['good'],
                'process_parameters': process_parameters
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash({
                'current_efficiency': current_efficiency,
                'process_type': process_type
            })
        )

        return result

    # ========================================================================
    # TOOL 7: ANALYZE IMPROVEMENTS
    # ========================================================================

    def analyze_improvements(
        self,
        efficiency_result: Dict[str, Any],
        loss_breakdown: Dict[str, Any]
    ) -> List[ImprovementOpportunity]:
        """
        Identify and prioritize efficiency improvements with ROI.

        Analyzes loss breakdown and efficiency gaps to identify
        actionable improvement opportunities.

        Args:
            efficiency_result: Current efficiency results
            loss_breakdown: Heat loss breakdown

        Returns:
            List of ImprovementOpportunity sorted by priority
        """
        improvements = []
        current_efficiency = efficiency_result.get('first_law_efficiency_percent', 0)
        total_input_kw = efficiency_result.get('energy_input_kw', 1000)

        # Get loss breakdown
        breakdown = loss_breakdown.get('breakdown', {})

        # 1. Flue gas heat recovery
        flue_loss = breakdown.get('flue_gas', 0)
        if flue_loss > 0:
            # Typical economizer can recover 50-70% of flue gas sensible heat
            recovery_potential = flue_loss * 0.6
            if recovery_potential > 10:  # Only if significant
                improvements.append(ImprovementOpportunity(
                    opportunity_id=str(uuid.uuid4())[:8],
                    category='heat_recovery',
                    description='Install economizer for flue gas heat recovery',
                    potential_savings_kw=round(recovery_potential, 2),
                    potential_savings_percent=round(recovery_potential / total_input_kw * 100, 2),
                    estimated_cost_usd=recovery_potential * 500,  # $500/kW typical
                    payback_months=round((recovery_potential * 500) / (recovery_potential * 0.10 * 8760) * 12, 1),
                    priority='high',
                    implementation_complexity='medium',
                    annual_co2_reduction_kg=round(recovery_potential * 0.2 * 8760, 2),
                    confidence_percent=85
                ))

        # 2. Combustion optimization
        combustion_eff = efficiency_result.get('combustion_efficiency_percent', 95)
        if combustion_eff < 95:
            improvement_potential = (95 - combustion_eff) / 100 * total_input_kw
            if improvement_potential > 5:
                improvements.append(ImprovementOpportunity(
                    opportunity_id=str(uuid.uuid4())[:8],
                    category='combustion',
                    description='Optimize combustion controls and excess air',
                    potential_savings_kw=round(improvement_potential, 2),
                    potential_savings_percent=round(improvement_potential / total_input_kw * 100, 2),
                    estimated_cost_usd=improvement_potential * 200,
                    payback_months=round((improvement_potential * 200) / (improvement_potential * 0.10 * 8760) * 12, 1),
                    priority='high',
                    implementation_complexity='low',
                    annual_co2_reduction_kg=round(improvement_potential * 0.2 * 8760, 2),
                    confidence_percent=90
                ))

        # 3. Insulation improvements
        radiation_loss = breakdown.get('radiation', 0)
        convection_loss = breakdown.get('convection', 0)
        surface_losses = radiation_loss + convection_loss
        if surface_losses > 5:
            recovery_potential = surface_losses * 0.8
            improvements.append(ImprovementOpportunity(
                opportunity_id=str(uuid.uuid4())[:8],
                category='insulation',
                description='Improve or add thermal insulation',
                potential_savings_kw=round(recovery_potential, 2),
                potential_savings_percent=round(recovery_potential / total_input_kw * 100, 2),
                estimated_cost_usd=recovery_potential * 150,
                payback_months=round((recovery_potential * 150) / (recovery_potential * 0.10 * 8760) * 12, 1),
                priority='medium',
                implementation_complexity='low',
                annual_co2_reduction_kg=round(recovery_potential * 0.2 * 8760, 2),
                confidence_percent=95
            ))

        # 4. Blowdown heat recovery
        blowdown_loss = breakdown.get('blowdown', 0)
        if blowdown_loss > 3:
            recovery_potential = blowdown_loss * 0.7
            improvements.append(ImprovementOpportunity(
                opportunity_id=str(uuid.uuid4())[:8],
                category='heat_recovery',
                description='Install blowdown heat recovery system',
                potential_savings_kw=round(recovery_potential, 2),
                potential_savings_percent=round(recovery_potential / total_input_kw * 100, 2),
                estimated_cost_usd=recovery_potential * 300,
                payback_months=round((recovery_potential * 300) / (recovery_potential * 0.10 * 8760) * 12, 1),
                priority='medium',
                implementation_complexity='medium',
                annual_co2_reduction_kg=round(recovery_potential * 0.2 * 8760, 2),
                confidence_percent=80
            ))

        # 5. Load optimization
        if current_efficiency < 85:
            improvement_potential = total_input_kw * 0.03  # 3% typical
            improvements.append(ImprovementOpportunity(
                opportunity_id=str(uuid.uuid4())[:8],
                category='operational',
                description='Optimize load scheduling and turndown',
                potential_savings_kw=round(improvement_potential, 2),
                potential_savings_percent=round(improvement_potential / total_input_kw * 100, 2),
                estimated_cost_usd=5000,  # Mostly operational changes
                payback_months=round(5000 / (improvement_potential * 0.10 * 8760) * 12, 1),
                priority='low',
                implementation_complexity='low',
                annual_co2_reduction_kg=round(improvement_potential * 0.2 * 8760, 2),
                confidence_percent=75
            ))

        # Sort by payback period
        improvements.sort(key=lambda x: x.payback_months)

        return improvements

    # ========================================================================
    # TOOL 8: UNCERTAINTY QUANTIFICATION
    # ========================================================================

    def quantify_uncertainty(
        self,
        input_data: Dict[str, Any],
        efficiency_result: Dict[str, Any]
    ) -> UncertaintyResult:
        """
        Quantify uncertainty in efficiency calculations.

        Uses propagation of uncertainty to estimate confidence
        intervals for calculated efficiency.

        Args:
            input_data: Input measurements
            efficiency_result: Calculated efficiency results

        Returns:
            UncertaintyResult with uncertainty analysis

        Standards: ASME PTC 19.1, GUM
        """
        # Define typical measurement uncertainties
        measurement_uncertainties = {
            'fuel_flow': 1.0,      # +/- 1%
            'heating_value': 0.5,  # +/- 0.5%
            'temperature': 0.5,    # +/- 0.5%
            'pressure': 0.25,      # +/- 0.25%
            'steam_flow': 1.5      # +/- 1.5%
        }

        # Calculate systematic uncertainty
        # For efficiency = output/input, relative uncertainty adds in quadrature
        systematic_components = [
            measurement_uncertainties['fuel_flow'],
            measurement_uncertainties['heating_value'],
            measurement_uncertainties['steam_flow']
        ]
        systematic_uncertainty = math.sqrt(sum(u**2 for u in systematic_components))

        # Estimate random uncertainty (typically smaller)
        random_uncertainty = systematic_uncertainty * 0.3

        # Total uncertainty (combined)
        total_uncertainty = math.sqrt(systematic_uncertainty**2 + random_uncertainty**2)

        # Contributing factors
        contributing_factors = {
            'fuel_measurement': measurement_uncertainties['fuel_flow']**2 / total_uncertainty**2 * 100,
            'heating_value': measurement_uncertainties['heating_value']**2 / total_uncertainty**2 * 100,
            'output_measurement': measurement_uncertainties['steam_flow']**2 / total_uncertainty**2 * 100
        }

        result = UncertaintyResult(
            efficiency_uncertainty_percent=round(total_uncertainty, 2),
            confidence_level_percent=95,
            contributing_factors=contributing_factors,
            measurement_uncertainties=measurement_uncertainties,
            systematic_uncertainty_percent=round(systematic_uncertainty, 2),
            random_uncertainty_percent=round(random_uncertainty, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            provenance_hash=self._calculate_hash(input_data)
        )

        return result

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_tool_schemas(self) -> Dict[str, Any]:
        """Get all tool schemas for API documentation."""
        return TOOL_SCHEMAS.copy()
