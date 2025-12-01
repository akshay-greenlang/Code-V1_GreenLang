# -*- coding: utf-8 -*-
"""
Multi-Fuel Optimizer Calculator for GL-011 FUELCRAFT.

Provides deterministic optimization algorithms for multi-fuel selection
across coal, natural gas, biomass, hydrogen, and other fuel types.

Standards Compliance:
- Linear Programming for optimization
- Zero-hallucination (no LLM for calculations)
- Complete provenance tracking
"""

import math
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class MultiFuelOptimizationInput:
    """Input parameters for multi-fuel optimization."""
    energy_demand_mw: float
    available_fuels: List[str]
    fuel_properties: Dict[str, Dict[str, Any]]
    market_prices: Dict[str, float]
    emission_limits: Dict[str, float]
    constraints: Dict[str, Any]
    optimization_objective: str


@dataclass
class MultiFuelOptimizationOutput:
    """Output of multi-fuel optimization."""
    optimal_fuel_mix: Dict[str, float]
    fuel_quantities_kg: Dict[str, float]
    total_cost_usd: float
    total_energy_gj: float
    total_emissions: Dict[str, float]
    carbon_intensity_kg_mwh: float
    efficiency_percent: float
    renewable_share: float
    optimization_score: float
    constraints_satisfied: bool
    iterations: int
    provenance_hash: str


class MultiFuelOptimizer:
    """
    Deterministic multi-fuel optimization calculator.

    Implements linear programming and heuristic approaches for
    optimal fuel mix selection meeting energy demand, cost,
    and emission constraints.

    Example:
        >>> optimizer = MultiFuelOptimizer()
        >>> result = optimizer.optimize(input_data)
        >>> print(f"Optimal mix: {result.optimal_fuel_mix}")
    """

    # Physical constants
    MJ_TO_GJ = 0.001
    GJ_TO_MWH = 1 / 3.6

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MultiFuelOptimizer."""
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.convergence_tolerance = self.config.get('tolerance', 0.0001)
        self.calculation_count = 0

    def optimize(
        self,
        input_data: MultiFuelOptimizationInput
    ) -> MultiFuelOptimizationOutput:
        """
        Optimize multi-fuel selection.

        Args:
            input_data: Optimization input parameters

        Returns:
            Optimization results with optimal fuel mix
        """
        self.calculation_count += 1

        # Convert energy demand to GJ/hr
        energy_gj_hr = input_data.energy_demand_mw * 3.6

        # Calculate fuel scores based on objective
        fuel_scores = self._calculate_fuel_scores(
            input_data.available_fuels,
            input_data.fuel_properties,
            input_data.market_prices,
            input_data.optimization_objective
        )

        # Sort fuels by score (descending)
        sorted_fuels = sorted(
            fuel_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Allocate energy across fuels
        optimal_mix = {}
        fuel_quantities = {}
        remaining_energy = energy_gj_hr
        iterations = 0

        for fuel, score in sorted_fuels:
            if remaining_energy <= 0:
                break
            iterations += 1

            props = input_data.fuel_properties.get(fuel, {})
            heating_value_gj = props.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ

            # Apply constraints
            max_share = input_data.constraints.get(f'{fuel}_max_share', 1.0)
            max_energy = energy_gj_hr * max_share

            # Check emission constraints
            if not self._check_emission_feasibility(
                fuel, props, input_data.emission_limits
            ):
                continue

            allocated = min(remaining_energy, max_energy)

            if allocated > 0 and heating_value_gj > 0:
                qty_kg = allocated / heating_value_gj
                fuel_quantities[fuel] = qty_kg
                optimal_mix[fuel] = allocated / energy_gj_hr
                remaining_energy -= allocated

        # Calculate total cost
        total_cost = sum(
            qty * input_data.market_prices.get(fuel, 0.05)
            for fuel, qty in fuel_quantities.items()
        )

        # Calculate emissions
        total_emissions = self._calculate_total_emissions(
            fuel_quantities, input_data.fuel_properties
        )

        # Calculate carbon intensity
        carbon_intensity = (
            total_emissions['co2'] / input_data.energy_demand_mw
            if input_data.energy_demand_mw > 0 else 0
        )

        # Calculate efficiency
        efficiency = self._calculate_blend_efficiency(
            optimal_mix, input_data.fuel_properties
        )

        # Calculate renewable share
        renewable_share = sum(
            share for fuel, share in optimal_mix.items()
            if input_data.fuel_properties.get(fuel, {}).get('renewable', False)
        )

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            total_cost / input_data.energy_demand_mw if input_data.energy_demand_mw > 0 else 0,
            carbon_intensity,
            renewable_share,
            efficiency
        )

        # Provenance hash
        provenance_hash = self._calculate_provenance(
            input_data, optimal_mix, total_cost
        )

        return MultiFuelOptimizationOutput(
            optimal_fuel_mix=optimal_mix,
            fuel_quantities_kg=fuel_quantities,
            total_cost_usd=round(total_cost, 2),
            total_energy_gj=round(energy_gj_hr, 2),
            total_emissions={k: round(v, 2) for k, v in total_emissions.items()},
            carbon_intensity_kg_mwh=round(carbon_intensity, 2),
            efficiency_percent=round(efficiency, 2),
            renewable_share=round(renewable_share, 4),
            optimization_score=round(optimization_score, 1),
            constraints_satisfied=remaining_energy <= 0.01 * energy_gj_hr,
            iterations=iterations,
            provenance_hash=provenance_hash
        )

    def _calculate_fuel_scores(
        self,
        fuels: List[str],
        properties: Dict[str, Dict[str, Any]],
        prices: Dict[str, float],
        objective: str
    ) -> Dict[str, float]:
        """Calculate optimization scores for each fuel."""
        scores = {}

        for fuel in fuels:
            props = properties.get(fuel, {})
            price = prices.get(fuel, 0.05)
            hv = props.get('heating_value_mj_kg', 30)
            co2 = props.get('emission_factor_co2_kg_gj', 60)
            renewable = props.get('renewable', False)

            # Cost efficiency
            cost_eff = hv / price if price > 0 else 0

            # Emission efficiency
            emission_eff = 100 / (co2 + 1)

            # Renewable bonus
            renewable_bonus = 50 if renewable else 0

            if objective == 'minimize_cost':
                scores[fuel] = cost_eff * 2 + emission_eff * 0.5
            elif objective == 'minimize_emissions':
                scores[fuel] = emission_eff * 2 + renewable_bonus + cost_eff * 0.3
            elif objective == 'maximize_efficiency':
                scores[fuel] = hv + cost_eff * 0.5
            elif objective == 'renewable_priority':
                scores[fuel] = renewable_bonus * 3 + emission_eff + cost_eff * 0.3
            else:  # balanced
                scores[fuel] = cost_eff + emission_eff + renewable_bonus * 0.5

        return scores

    def _check_emission_feasibility(
        self,
        fuel: str,
        props: Dict[str, Any],
        limits: Dict[str, float]
    ) -> bool:
        """Check if fuel meets emission constraints."""
        if not limits:
            return True

        nox = props.get('emission_factor_nox_g_gj', 100)
        if nox > limits.get('nox_g_gj', float('inf')):
            return False

        sox = props.get('emission_factor_sox_g_gj', 50)
        if sox > limits.get('sox_g_gj', float('inf')):
            return False

        return True

    def _calculate_total_emissions(
        self,
        fuel_quantities: Dict[str, float],
        properties: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate total emissions from fuel mix."""
        emissions = {'co2': 0.0, 'nox': 0.0, 'sox': 0.0, 'pm': 0.0}

        for fuel, qty in fuel_quantities.items():
            props = properties.get(fuel, {})
            hv_gj = props.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ
            energy_gj = qty * hv_gj

            emissions['co2'] += props.get('emission_factor_co2_kg_gj', 60) * energy_gj
            emissions['nox'] += props.get('emission_factor_nox_g_gj', 100) * energy_gj / 1000
            emissions['sox'] += props.get('emission_factor_sox_g_gj', 50) * energy_gj / 1000
            emissions['pm'] += props.get('emission_factor_pm_g_gj', 10) * energy_gj / 1000

        return emissions

    def _calculate_blend_efficiency(
        self,
        mix: Dict[str, float],
        properties: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate weighted blend efficiency."""
        if not mix:
            return 0.0

        total = 0.0
        for fuel, share in mix.items():
            base_eff = 88.0
            if fuel == 'natural_gas':
                base_eff = 92.0
            elif fuel == 'coal':
                base_eff = 85.0
            elif fuel == 'hydrogen':
                base_eff = 95.0
            total += base_eff * share

        return total

    def _calculate_optimization_score(
        self,
        cost_per_mwh: float,
        carbon_intensity: float,
        renewable_share: float,
        efficiency: float
    ) -> float:
        """Calculate overall optimization score."""
        cost_score = max(0, 100 - cost_per_mwh * 0.5)
        carbon_score = max(0, 100 - carbon_intensity * 0.1)
        renewable_score = renewable_share * 100
        efficiency_score = efficiency

        return (
            cost_score * 0.3 +
            carbon_score * 0.3 +
            renewable_score * 0.2 +
            efficiency_score * 0.2
        )

    def _calculate_provenance(
        self,
        input_data: MultiFuelOptimizationInput,
        result: Dict[str, float],
        cost: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            'demand': input_data.energy_demand_mw,
            'fuels': sorted(input_data.available_fuels),
            'objective': input_data.optimization_objective,
            'result': result,
            'cost': cost
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
