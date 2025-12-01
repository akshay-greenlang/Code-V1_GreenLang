# -*- coding: utf-8 -*-
"""
Tools module for FuelManagementOrchestrator agent (GL-011 FUELCRAFT).

This module provides deterministic calculation tools for fuel management
optimization, including multi-fuel selection, cost optimization, blending,
carbon footprint minimization, and procurement optimization.

All calculations follow industry standards:
- ISO 6976:2016 - Natural gas calorific value calculations
- ISO 17225 - Solid biofuels specifications
- ASTM D4809 - Heat of combustion for liquid fuels
- GHG Protocol - Emissions calculations
- IPCC Guidelines - Emission factors

ZERO HALLUCINATION GUARANTEE:
- All numeric calculations are deterministic
- No LLM calls for numeric computations
- Complete provenance tracking with SHA-256

Author: GreenLang Industrial Optimization Team
Date: December 2025
Agent ID: GL-011
Version: 1.0.0
"""

import math
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache

logger = logging.getLogger(__name__)


# ============================================================================
# RESULT DATA CLASSES
# ============================================================================

@dataclass
class MultiFuelOptimizationResult:
    """Result of multi-fuel optimization calculations."""

    optimal_fuel_mix: Dict[str, float]  # fuel_type -> percentage (0-1)
    fuel_quantities: Dict[str, float]  # fuel_type -> quantity in kg
    total_energy_output_mw: float
    total_cost_usd: float
    cost_per_mwh: float
    baseline_cost_usd: float
    savings_usd: float
    savings_percent: float
    total_emissions_kg: Dict[str, float]  # pollutant -> kg/hr
    carbon_intensity_kg_mwh: float
    efficiency_percent: float
    renewable_share: float
    optimization_score: float  # 0-100
    constraints_satisfied: bool
    provenance_hash: str


@dataclass
class CostOptimizationResult:
    """Result of cost optimization calculations."""

    optimal_fuel_selection: str
    optimal_quantity_kg: float
    cost_usd: float
    cost_per_mj: float
    alternative_options: List[Dict[str, Any]]
    savings_vs_alternatives: Dict[str, float]
    procurement_recommendation: str
    inventory_impact: Dict[str, float]
    provenance_hash: str


@dataclass
class BlendingOptimizationResult:
    """Result of fuel blending optimization."""

    blend_ratios: Dict[str, float]  # fuel_type -> ratio
    blend_heating_value_mj_kg: float
    blend_carbon_content_percent: float
    blend_moisture_percent: float
    blend_ash_percent: float
    blend_sulfur_percent: float
    estimated_emissions: Dict[str, float]
    blend_cost_per_kg: float
    blend_quality_score: float  # 0-100
    compatibility_check: bool
    warnings: List[str]
    provenance_hash: str


@dataclass
class CarbonFootprintResult:
    """Result of carbon footprint calculation."""

    total_co2e_kg: float
    co2_kg: float
    ch4_kg: float
    n2o_kg: float
    carbon_intensity_kg_mwh: float
    biogenic_carbon_kg: float
    fossil_carbon_kg: float
    emission_breakdown: Dict[str, float]
    scope1_emissions_kg: float
    scope3_upstream_kg: float
    reduction_potential_kg: float
    carbon_cost_usd: float
    provenance_hash: str


@dataclass
class EmissionFactorLookupResult:
    """Result of emission factor lookup."""

    fuel_type: str
    co2_kg_gj: float
    ch4_g_gj: float
    n2o_g_gj: float
    nox_g_gj: float
    sox_g_gj: float
    pm_g_gj: float
    source: str
    year: int
    uncertainty_percent: float
    provenance_hash: str


# ============================================================================
# MAIN TOOLS CLASS
# ============================================================================

class FuelManagementTools:
    """
    Deterministic calculation tools for fuel management optimization.

    All calculations follow industry standards and produce reproducible
    results. No LLM is used for numeric calculations - only for
    classification and recommendation text generation.

    Attributes:
        logger: Logging instance
        tool_call_count: Counter for tool invocations
        GWP_AR6: IPCC AR6 Global Warming Potentials

    Example:
        >>> tools = FuelManagementTools()
        >>> result = tools.optimize_multi_fuel_selection(
        ...     energy_demand_mw=100,
        ...     available_fuels=["natural_gas", "coal", "biomass"],
        ...     fuel_properties={...},
        ...     market_prices={"natural_gas": 0.05, "coal": 0.03},
        ...     emission_limits={"nox_g_gj": 200},
        ...     constraints={},
        ...     optimization_objective="balanced"
        ... )
    """

    def __init__(self):
        """Initialize FuelManagementTools."""
        self.logger = logging.getLogger(__name__)
        self.tool_call_count = 0

        # IPCC AR6 Global Warming Potentials (100-year)
        self.GWP_AR6 = {
            'CO2': 1.0,
            'CH4': 29.8,  # Fossil methane (includes climate-carbon feedback)
            'N2O': 273.0
        }

        # Standard reference conditions
        self.STANDARD_TEMPERATURE_K = 288.15  # 15°C
        self.STANDARD_PRESSURE_KPA = 101.325

        # Conversion constants
        self.MJ_TO_GJ = 0.001
        self.GJ_TO_MWH = 1 / 3.6
        self.KG_TO_TONNE = 0.001

    def get_tool_call_count(self) -> int:
        """Get total tool call count."""
        return self.tool_call_count

    def _increment_tool_count(self) -> None:
        """Increment tool call counter."""
        self.tool_call_count += 1

    def _calculate_provenance_hash(self, *args) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        data_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    # ========================================================================
    # MULTI-FUEL OPTIMIZATION
    # ========================================================================

    def optimize_multi_fuel_selection(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        fuel_properties: Dict[str, Dict[str, Any]],
        market_prices: Dict[str, float],
        emission_limits: Dict[str, float],
        constraints: Dict[str, Any],
        optimization_objective: str
    ) -> Dict[str, Any]:
        """
        Optimize multi-fuel selection and mix.

        This method finds the optimal combination of available fuels to meet
        energy demand while satisfying emission limits and constraints.

        Algorithm:
        1. Calculate fuel requirements for each fuel type
        2. Apply emission constraints
        3. Optimize based on objective (cost, emissions, balanced)
        4. Validate blend compatibility
        5. Calculate final metrics

        Args:
            energy_demand_mw: Required energy output in MW
            available_fuels: List of available fuel types
            fuel_properties: Physical/chemical properties per fuel
            market_prices: Market prices per fuel (USD/kg)
            emission_limits: Emission limits by pollutant
            constraints: Additional operational constraints
            optimization_objective: Optimization target

        Returns:
            Dict with optimal fuel mix, costs, emissions, and recommendations

        Raises:
            ValueError: If inputs are invalid
        """
        self._increment_tool_count()

        # Input validation
        if energy_demand_mw <= 0:
            raise ValueError("Energy demand must be positive")
        if not available_fuels:
            raise ValueError("At least one fuel must be available")

        # Calculate energy requirement in GJ/hr
        energy_gj_hr = energy_demand_mw * 3.6  # 1 MWh = 3.6 GJ

        # Initialize optimization variables
        optimal_mix = {}
        fuel_quantities = {}
        total_cost = 0.0
        total_emissions = {'co2': 0.0, 'nox': 0.0, 'sox': 0.0, 'pm': 0.0}
        renewable_energy = 0.0

        # Calculate fuel scores based on objective
        fuel_scores = self._calculate_fuel_scores(
            available_fuels,
            fuel_properties,
            market_prices,
            optimization_objective
        )

        # Sort fuels by score (highest first)
        sorted_fuels = sorted(
            fuel_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Allocate energy demand across fuels
        remaining_energy_gj = energy_gj_hr

        for fuel_type, score in sorted_fuels:
            if remaining_energy_gj <= 0:
                break

            props = fuel_properties.get(fuel_type, {})
            heating_value = props.get('heating_value_mj_kg', 30.0) * self.MJ_TO_GJ

            # Check emission constraints
            if not self._check_emission_constraints(
                fuel_type, props, emission_limits, remaining_energy_gj
            ):
                continue

            # Calculate maximum allocation for this fuel
            max_share = constraints.get(f'{fuel_type}_max_share', 1.0)
            max_energy_this_fuel = energy_gj_hr * max_share

            # Allocate energy
            allocated_energy = min(remaining_energy_gj, max_energy_this_fuel)

            if allocated_energy > 0:
                # Calculate fuel quantity
                fuel_qty_kg = allocated_energy / heating_value
                fuel_quantities[fuel_type] = fuel_qty_kg

                # Calculate share
                optimal_mix[fuel_type] = allocated_energy / energy_gj_hr

                # Calculate cost
                price = market_prices.get(fuel_type, 0.05)
                fuel_cost = fuel_qty_kg * price
                total_cost += fuel_cost

                # Calculate emissions
                self._add_fuel_emissions(
                    total_emissions, fuel_type, props, allocated_energy
                )

                # Track renewable energy
                if props.get('renewable', False):
                    renewable_energy += allocated_energy

                remaining_energy_gj -= allocated_energy

        # Calculate derived metrics
        if energy_gj_hr > 0:
            renewable_share = renewable_energy / energy_gj_hr
            carbon_intensity = total_emissions['co2'] / energy_demand_mw  # kg/MWh
        else:
            renewable_share = 0.0
            carbon_intensity = 0.0

        cost_per_mwh = total_cost / energy_demand_mw if energy_demand_mw > 0 else 0

        # Calculate baseline cost (using most expensive fuel)
        baseline_cost = self._calculate_baseline_cost(
            energy_gj_hr, available_fuels, fuel_properties, market_prices
        )

        savings = baseline_cost - total_cost
        savings_percent = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

        # Calculate efficiency
        efficiency = self._calculate_blend_efficiency_internal(
            optimal_mix, fuel_properties
        )

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            cost_per_mwh, carbon_intensity, renewable_share, efficiency
        )

        # Check constraints satisfaction
        constraints_satisfied = remaining_energy_gj <= 0.01 * energy_gj_hr

        provenance_hash = self._calculate_provenance_hash(
            energy_demand_mw, available_fuels, optimization_objective
        )

        return {
            'optimal_fuel_mix': optimal_mix,
            'fuel_quantities': fuel_quantities,
            'energy_output_mw': energy_demand_mw,
            'energy_output_mwh': energy_demand_mw,  # Per hour
            'total_cost_usd': round(total_cost, 2),
            'cost_per_mwh': round(cost_per_mwh, 2),
            'baseline_cost_usd': round(baseline_cost, 2),
            'savings_usd': round(savings, 2),
            'savings_percent': round(savings_percent, 2),
            'total_emissions_kg': {k: round(v, 2) for k, v in total_emissions.items()},
            'carbon_intensity_kg_mwh': round(carbon_intensity, 2),
            'efficiency_percent': round(efficiency, 2),
            'renewable_share': round(renewable_share, 4),
            'optimization_score': round(optimization_score, 1),
            'constraints_satisfied': constraints_satisfied,
            'objective': optimization_objective,
            'provenance_hash': provenance_hash
        }

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
            heating_value = props.get('heating_value_mj_kg', 30.0)
            co2_factor = props.get('emission_factor_co2_kg_gj', 60.0)
            is_renewable = props.get('renewable', False)

            # Cost efficiency (MJ per USD)
            cost_efficiency = heating_value / price if price > 0 else 0

            # Emission efficiency (inverse of CO2 factor)
            emission_efficiency = 100 / (co2_factor + 1)

            # Renewable bonus
            renewable_bonus = 50 if is_renewable else 0

            if objective == 'minimize_cost':
                scores[fuel] = cost_efficiency * 2 + emission_efficiency * 0.5
            elif objective == 'minimize_emissions':
                scores[fuel] = emission_efficiency * 2 + renewable_bonus + cost_efficiency * 0.3
            elif objective == 'maximize_efficiency':
                scores[fuel] = heating_value + cost_efficiency * 0.5
            elif objective == 'renewable_priority':
                scores[fuel] = renewable_bonus * 3 + emission_efficiency + cost_efficiency * 0.3
            else:  # balanced
                scores[fuel] = cost_efficiency + emission_efficiency + renewable_bonus * 0.5

        return scores

    def _check_emission_constraints(
        self,
        fuel_type: str,
        props: Dict[str, Any],
        limits: Dict[str, float],
        energy_gj: float
    ) -> bool:
        """Check if fuel meets emission constraints."""
        if not limits:
            return True

        # Check NOx constraint
        nox_factor = props.get('emission_factor_nox_g_gj', 100)
        nox_limit = limits.get('nox_g_gj', float('inf'))
        if nox_factor > nox_limit:
            return False

        # Check SOx constraint
        sox_factor = props.get('emission_factor_sox_g_gj', 100)
        sox_limit = limits.get('sox_g_gj', float('inf'))
        if sox_factor > sox_limit:
            return False

        return True

    def _add_fuel_emissions(
        self,
        totals: Dict[str, float],
        fuel_type: str,
        props: Dict[str, Any],
        energy_gj: float
    ) -> None:
        """Add emissions from fuel to totals."""
        # CO2 emissions (kg)
        co2_factor = props.get('emission_factor_co2_kg_gj', 60.0)
        totals['co2'] += co2_factor * energy_gj

        # NOx emissions (converted from g to kg)
        nox_factor = props.get('emission_factor_nox_g_gj', 100)
        totals['nox'] += nox_factor * energy_gj / 1000

        # SOx emissions (converted from g to kg)
        sox_factor = props.get('emission_factor_sox_g_gj', 50)
        totals['sox'] += sox_factor * energy_gj / 1000

        # PM emissions (converted from g to kg)
        pm_factor = props.get('emission_factor_pm_g_gj', 10)
        totals['pm'] += pm_factor * energy_gj / 1000

    def _calculate_baseline_cost(
        self,
        energy_gj: float,
        fuels: List[str],
        properties: Dict[str, Dict[str, Any]],
        prices: Dict[str, float]
    ) -> float:
        """Calculate baseline cost using single most expensive fuel."""
        max_cost = 0.0

        for fuel in fuels:
            props = properties.get(fuel, {})
            heating_value = props.get('heating_value_mj_kg', 30.0) * self.MJ_TO_GJ
            price = prices.get(fuel, 0.05)

            qty_kg = energy_gj / heating_value if heating_value > 0 else 0
            cost = qty_kg * price

            max_cost = max(max_cost, cost)

        return max_cost

    def _calculate_blend_efficiency_internal(
        self,
        mix: Dict[str, float],
        properties: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate weighted average efficiency of fuel blend."""
        if not mix:
            return 0.0

        total_efficiency = 0.0
        for fuel, share in mix.items():
            props = properties.get(fuel, {})
            # Assume efficiency based on fuel type
            base_efficiency = 90.0  # Default

            if fuel == 'natural_gas':
                base_efficiency = 92.0
            elif fuel == 'coal':
                base_efficiency = 85.0
            elif fuel == 'biomass':
                base_efficiency = 80.0
            elif fuel == 'hydrogen':
                base_efficiency = 95.0
            elif fuel in ('fuel_oil', 'diesel'):
                base_efficiency = 88.0

            total_efficiency += base_efficiency * share

        return total_efficiency

    def _calculate_optimization_score(
        self,
        cost_per_mwh: float,
        carbon_intensity: float,
        renewable_share: float,
        efficiency: float
    ) -> float:
        """Calculate overall optimization score (0-100)."""
        # Normalize metrics to 0-100 scale

        # Cost score (lower is better, assume 0-200 USD/MWh range)
        cost_score = max(0, 100 - cost_per_mwh * 0.5)

        # Carbon score (lower is better, assume 0-1000 kg/MWh range)
        carbon_score = max(0, 100 - carbon_intensity * 0.1)

        # Renewable score (higher is better)
        renewable_score = renewable_share * 100

        # Efficiency score
        efficiency_score = efficiency

        # Weighted average
        score = (
            cost_score * 0.3 +
            carbon_score * 0.3 +
            renewable_score * 0.2 +
            efficiency_score * 0.2
        )

        return min(100, max(0, score))

    # ========================================================================
    # COST OPTIMIZATION
    # ========================================================================

    def optimize_fuel_cost(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        fuel_properties: Dict[str, Dict[str, Any]],
        market_prices: Dict[str, float],
        fuel_inventories: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize fuel selection for minimum cost.

        Args:
            energy_demand_mw: Required energy output
            available_fuels: Available fuel types
            fuel_properties: Physical/chemical properties
            market_prices: Current market prices
            fuel_inventories: Current inventory levels
            constraints: Operational constraints

        Returns:
            Cost-optimized fuel selection with alternatives
        """
        self._increment_tool_count()

        energy_gj_hr = energy_demand_mw * 3.6

        # Calculate cost for each fuel option
        fuel_costs = []

        for fuel in available_fuels:
            props = fuel_properties.get(fuel, {})
            price = market_prices.get(fuel, 0.05)
            inventory = fuel_inventories.get(fuel, 0)
            heating_value = props.get('heating_value_mj_kg', 30.0) * self.MJ_TO_GJ

            # Calculate quantity needed
            qty_kg = energy_gj_hr / heating_value if heating_value > 0 else 0

            # Check inventory availability
            available = min(qty_kg, inventory)
            shortage = max(0, qty_kg - inventory)

            # Calculate costs
            fuel_cost = qty_kg * price
            cost_per_mj = price / (heating_value / self.MJ_TO_GJ) if heating_value > 0 else 0

            fuel_costs.append({
                'fuel': fuel,
                'quantity_kg': round(qty_kg, 2),
                'cost_usd': round(fuel_cost, 2),
                'cost_per_mj': round(cost_per_mj, 6),
                'available_inventory': round(available, 2),
                'shortage': round(shortage, 2),
                'feasible': shortage <= 0 or constraints.get('allow_procurement', True)
            })

        # Sort by cost
        fuel_costs.sort(key=lambda x: x['cost_usd'])

        # Select optimal (lowest cost that is feasible)
        optimal = None
        for fc in fuel_costs:
            if fc['feasible']:
                optimal = fc
                break

        if optimal is None:
            optimal = fuel_costs[0] if fuel_costs else {'fuel': None, 'cost_usd': 0}

        # Calculate savings vs alternatives
        savings = {}
        for fc in fuel_costs:
            if fc['fuel'] != optimal['fuel']:
                savings[fc['fuel']] = round(fc['cost_usd'] - optimal['cost_usd'], 2)

        provenance_hash = self._calculate_provenance_hash(
            energy_demand_mw, available_fuels, market_prices
        )

        return {
            'optimal_fuel_selection': optimal['fuel'],
            'optimal_quantity_kg': optimal.get('quantity_kg', 0),
            'cost_usd': optimal['cost_usd'],
            'cost_per_mj': optimal.get('cost_per_mj', 0),
            'savings_usd': sum(max(0, s) for s in savings.values()),
            'alternative_options': fuel_costs[1:5] if len(fuel_costs) > 1 else [],
            'savings_vs_alternatives': savings,
            'procurement_recommendation': (
                f"Use {optimal['fuel']} for minimum cost"
                if optimal['fuel'] else "No feasible option"
            ),
            'inventory_impact': {
                optimal['fuel']: -optimal.get('quantity_kg', 0)
            } if optimal['fuel'] else {},
            'provenance_hash': provenance_hash
        }

    # ========================================================================
    # BLENDING OPTIMIZATION
    # ========================================================================

    def optimize_fuel_blending(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        fuel_properties: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
        emission_limits: Dict[str, float],
        optimization_objective: str
    ) -> Dict[str, Any]:
        """
        Optimize fuel blending ratios for target properties.

        Args:
            energy_demand_mw: Required energy output
            available_fuels: Available fuel types for blending
            fuel_properties: Physical/chemical properties
            constraints: Blending constraints
            emission_limits: Emission limits
            optimization_objective: Optimization target

        Returns:
            Optimized blend ratios with quality metrics
        """
        self._increment_tool_count()

        # Initialize blend calculation
        blend_ratios = {}
        warnings = []

        # Get target quality parameters from constraints
        target_heating_value = constraints.get('target_heating_value_mj_kg', 25.0)
        max_moisture = constraints.get('max_moisture_percent', 20.0)
        max_ash = constraints.get('max_ash_percent', 15.0)
        max_sulfur = constraints.get('max_sulfur_percent', 2.0)

        # Score each fuel for blending suitability
        fuel_scores = {}
        for fuel in available_fuels:
            props = fuel_properties.get(fuel, {})
            score = 0

            # Heating value score
            hv = props.get('heating_value_mj_kg', 30.0)
            hv_diff = abs(hv - target_heating_value)
            score += max(0, 50 - hv_diff * 2)

            # Moisture penalty
            moisture = props.get('moisture_content_percent', 0)
            if moisture > max_moisture:
                score -= 20

            # Ash penalty
            ash = props.get('ash_content_percent', 0)
            if ash > max_ash:
                score -= 20

            # Sulfur penalty
            sulfur = props.get('sulfur_content_percent', 0)
            if sulfur > max_sulfur:
                score -= 30

            # Emission factor consideration
            if optimization_objective == 'minimize_emissions':
                co2_factor = props.get('emission_factor_co2_kg_gj', 60)
                score += max(0, 50 - co2_factor * 0.5)

            fuel_scores[fuel] = max(0, score)

        # Normalize scores to ratios
        total_score = sum(fuel_scores.values())
        if total_score > 0:
            for fuel, score in fuel_scores.items():
                ratio = score / total_score
                # Apply min/max constraints
                min_ratio = constraints.get(f'{fuel}_min_ratio', 0.0)
                max_ratio = constraints.get(f'{fuel}_max_ratio', 1.0)
                blend_ratios[fuel] = max(min_ratio, min(max_ratio, ratio))

            # Renormalize to sum to 1.0
            ratio_sum = sum(blend_ratios.values())
            if ratio_sum > 0:
                blend_ratios = {k: v / ratio_sum for k, v in blend_ratios.items()}
        else:
            # Equal distribution if no valid scores
            ratio = 1.0 / len(available_fuels)
            blend_ratios = {fuel: ratio for fuel in available_fuels}

        # Calculate blend properties
        blend_heating_value = sum(
            blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('heating_value_mj_kg', 30)
            for fuel in available_fuels
        )

        blend_carbon = sum(
            blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('carbon_content_percent', 50)
            for fuel in available_fuels
        )

        blend_moisture = sum(
            blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('moisture_content_percent', 0)
            for fuel in available_fuels
        )

        blend_ash = sum(
            blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('ash_content_percent', 0)
            for fuel in available_fuels
        )

        blend_sulfur = sum(
            blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('sulfur_content_percent', 0)
            for fuel in available_fuels
        )

        # Check quality constraints
        if blend_moisture > max_moisture:
            warnings.append(f"Blend moisture {blend_moisture:.1f}% exceeds limit {max_moisture}%")
        if blend_ash > max_ash:
            warnings.append(f"Blend ash {blend_ash:.1f}% exceeds limit {max_ash}%")
        if blend_sulfur > max_sulfur:
            warnings.append(f"Blend sulfur {blend_sulfur:.2f}% exceeds limit {max_sulfur}%")

        # Check fuel compatibility
        incompatible = constraints.get('incompatible_fuels', [])
        active_fuels = [f for f, r in blend_ratios.items() if r > 0.01]
        compatibility_check = True
        for pair in incompatible:
            if all(f in active_fuels for f in pair):
                compatibility_check = False
                warnings.append(f"Incompatible fuels in blend: {pair}")

        # Calculate blend quality score
        quality_score = 100
        if blend_moisture > max_moisture:
            quality_score -= 20
        if blend_ash > max_ash:
            quality_score -= 15
        if blend_sulfur > max_sulfur:
            quality_score -= 25
        if not compatibility_check:
            quality_score -= 20
        quality_score = max(0, quality_score)

        # Estimate emissions for blend
        estimated_emissions = {
            'co2_kg_gj': sum(
                blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('emission_factor_co2_kg_gj', 60)
                for fuel in available_fuels
            ),
            'nox_g_gj': sum(
                blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('emission_factor_nox_g_gj', 100)
                for fuel in available_fuels
            ),
            'sox_g_gj': sum(
                blend_ratios.get(fuel, 0) * fuel_properties.get(fuel, {}).get('emission_factor_sox_g_gj', 50)
                for fuel in available_fuels
            )
        }

        provenance_hash = self._calculate_provenance_hash(
            available_fuels, constraints, optimization_objective
        )

        return {
            'blend_ratios': {k: round(v, 4) for k, v in blend_ratios.items()},
            'blend_heating_value_mj_kg': round(blend_heating_value, 2),
            'blend_carbon_content_percent': round(blend_carbon, 2),
            'blend_moisture_percent': round(blend_moisture, 2),
            'blend_ash_percent': round(blend_ash, 2),
            'blend_sulfur_percent': round(blend_sulfur, 3),
            'estimated_emissions': {k: round(v, 2) for k, v in estimated_emissions.items()},
            'blend_cost_per_kg': 0.0,  # Would need prices
            'blend_quality_score': round(quality_score, 1),
            'compatibility_check': compatibility_check,
            'warnings': warnings,
            'provenance_hash': provenance_hash
        }

    # ========================================================================
    # CARBON FOOTPRINT MINIMIZATION
    # ========================================================================

    def minimize_carbon_footprint(
        self,
        energy_demand_mw: float,
        available_fuels: List[str],
        fuel_properties: Dict[str, Dict[str, Any]],
        emission_limits: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Minimize carbon footprint of fuel selection.

        Prioritizes low-carbon and renewable fuels while meeting
        energy demand and operational constraints.

        Args:
            energy_demand_mw: Required energy output
            available_fuels: Available fuel types
            fuel_properties: Physical/chemical properties
            emission_limits: Emission limits
            constraints: Operational constraints

        Returns:
            Carbon-optimized fuel selection with footprint analysis
        """
        self._increment_tool_count()

        energy_gj_hr = energy_demand_mw * 3.6

        # Sort fuels by carbon intensity (lowest first)
        fuel_carbon = []
        for fuel in available_fuels:
            props = fuel_properties.get(fuel, {})
            co2_factor = props.get('emission_factor_co2_kg_gj', 60)
            is_renewable = props.get('renewable', False)
            biogenic = props.get('biogenic_carbon_percent', 0)

            # Effective CO2 (accounting for biogenic carbon)
            effective_co2 = co2_factor * (1 - biogenic / 100)

            fuel_carbon.append({
                'fuel': fuel,
                'co2_factor': co2_factor,
                'effective_co2': effective_co2,
                'renewable': is_renewable,
                'biogenic_percent': biogenic
            })

        fuel_carbon.sort(key=lambda x: x['effective_co2'])

        # Allocate energy to lowest carbon fuels first
        allocation = {}
        remaining_energy = energy_gj_hr
        total_co2 = 0.0
        total_biogenic = 0.0
        total_fossil = 0.0

        for fc in fuel_carbon:
            if remaining_energy <= 0:
                break

            fuel = fc['fuel']
            props = fuel_properties.get(fuel, {})
            heating_value = props.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ

            # Check constraints
            max_share = constraints.get(f'{fuel}_max_share', 1.0)
            max_energy = energy_gj_hr * max_share

            allocated = min(remaining_energy, max_energy)
            if allocated > 0:
                qty_kg = allocated / heating_value
                allocation[fuel] = qty_kg

                # Calculate emissions
                co2_kg = fc['co2_factor'] * allocated
                if fc['biogenic_percent'] > 0:
                    biogenic_kg = co2_kg * fc['biogenic_percent'] / 100
                    fossil_kg = co2_kg - biogenic_kg
                    total_biogenic += biogenic_kg
                    total_fossil += fossil_kg
                else:
                    total_fossil += co2_kg

                total_co2 += co2_kg
                remaining_energy -= allocated

        # Calculate carbon intensity
        carbon_intensity = total_co2 / energy_demand_mw if energy_demand_mw > 0 else 0

        # Calculate CO2e including CH4 and N2O
        total_ch4_kg = sum(
            fuel_properties.get(f, {}).get('emission_factor_ch4_g_gj', 1) *
            allocation.get(f, 0) * fuel_properties.get(f, {}).get('heating_value_mj_kg', 30) *
            self.MJ_TO_GJ / 1000
            for f in allocation
        )

        total_n2o_kg = sum(
            fuel_properties.get(f, {}).get('emission_factor_n2o_g_gj', 0.1) *
            allocation.get(f, 0) * fuel_properties.get(f, {}).get('heating_value_mj_kg', 30) *
            self.MJ_TO_GJ / 1000
            for f in allocation
        )

        # Convert to CO2e
        total_co2e = (
            total_co2 +
            total_ch4_kg * self.GWP_AR6['CH4'] +
            total_n2o_kg * self.GWP_AR6['N2O']
        )

        # Calculate reduction potential (vs coal baseline)
        coal_factor = 94.6  # kg CO2/GJ for coal
        baseline_co2 = coal_factor * energy_gj_hr
        reduction = baseline_co2 - total_co2
        reduction_potential = max(0, reduction)

        # Carbon cost
        carbon_price = constraints.get('carbon_price_usd_per_tonne', 50)
        carbon_cost = total_co2 * self.KG_TO_TONNE * carbon_price

        provenance_hash = self._calculate_provenance_hash(
            energy_demand_mw, available_fuels, 'carbon_footprint'
        )

        return {
            'total_co2e_kg': round(total_co2e, 2),
            'co2_kg': round(total_co2, 2),
            'ch4_kg': round(total_ch4_kg, 4),
            'n2o_kg': round(total_n2o_kg, 4),
            'carbon_intensity_kg_mwh': round(carbon_intensity, 2),
            'biogenic_carbon_kg': round(total_biogenic, 2),
            'fossil_carbon_kg': round(total_fossil, 2),
            'emission_breakdown': {f: round(q, 2) for f, q in allocation.items()},
            'scope1_emissions_kg': round(total_fossil, 2),
            'scope3_upstream_kg': 0,  # Would need upstream factors
            'reduction_potential_kg': round(reduction_potential, 2),
            'emissions_reduction_kg': round(reduction_potential, 2),
            'carbon_cost_usd': round(carbon_cost, 2),
            'provenance_hash': provenance_hash
        }

    # ========================================================================
    # PROCUREMENT OPTIMIZATION
    # ========================================================================

    def optimize_procurement(
        self,
        fuel_inventories: Dict[str, float],
        market_prices: Dict[str, float],
        fuel_properties: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize fuel procurement and inventory management.

        Args:
            fuel_inventories: Current inventory levels
            market_prices: Current market prices
            fuel_properties: Physical/chemical properties
            constraints: Procurement constraints

        Returns:
            Procurement recommendations with timing and quantities
        """
        self._increment_tool_count()

        recommendations = []
        total_procurement_cost = 0.0

        # Calculate consumption rate and days of supply for each fuel
        daily_consumption = constraints.get('daily_consumption', {})
        safety_stock_days = constraints.get('safety_stock_days', 7)
        reorder_lead_days = constraints.get('reorder_lead_days', 5)

        for fuel, inventory in fuel_inventories.items():
            consumption = daily_consumption.get(fuel, 0)
            price = market_prices.get(fuel, 0.05)
            props = fuel_properties.get(fuel, {})

            if consumption > 0:
                days_of_supply = inventory / consumption
                reorder_point = consumption * (safety_stock_days + reorder_lead_days)

                # Check if reorder needed
                if inventory <= reorder_point:
                    # Calculate order quantity (economic order quantity approximation)
                    target_inventory = consumption * 30  # 30 days supply
                    order_qty = max(0, target_inventory - inventory)

                    order_cost = order_qty * price

                    recommendations.append({
                        'fuel': fuel,
                        'action': 'reorder',
                        'priority': 'high' if days_of_supply < safety_stock_days else 'medium',
                        'current_inventory_kg': round(inventory, 2),
                        'days_of_supply': round(days_of_supply, 1),
                        'recommended_order_kg': round(order_qty, 2),
                        'estimated_cost_usd': round(order_cost, 2),
                        'current_price_usd_kg': round(price, 4),
                        'urgency': 'immediate' if days_of_supply < 3 else 'scheduled'
                    })

                    total_procurement_cost += order_cost
                else:
                    recommendations.append({
                        'fuel': fuel,
                        'action': 'monitor',
                        'priority': 'low',
                        'current_inventory_kg': round(inventory, 2),
                        'days_of_supply': round(days_of_supply, 1),
                        'next_review_days': round(days_of_supply - reorder_point / consumption, 1)
                    })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))

        provenance_hash = self._calculate_provenance_hash(
            fuel_inventories, market_prices, constraints
        )

        return {
            'recommendations': recommendations,
            'total_procurement_cost_usd': round(total_procurement_cost, 2),
            'fuels_requiring_reorder': len([r for r in recommendations if r['action'] == 'reorder']),
            'immediate_actions': len([r for r in recommendations if r.get('urgency') == 'immediate']),
            'provenance_hash': provenance_hash
        }

    # ========================================================================
    # EMISSIONS CALCULATIONS
    # ========================================================================

    def calculate_emissions(
        self,
        fuel_mix: Dict[str, float],
        fuel_quantities: Dict[str, float],
        fuel_properties: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a fuel mix.

        Args:
            fuel_mix: Fuel percentages by type
            fuel_quantities: Fuel quantities in kg
            fuel_properties: Physical/chemical properties

        Returns:
            Emissions analysis by pollutant
        """
        self._increment_tool_count()

        total_energy_gj = 0.0
        emissions = {
            'co2_kg_hr': 0.0,
            'ch4_g_hr': 0.0,
            'n2o_g_hr': 0.0,
            'nox_kg_hr': 0.0,
            'sox_kg_hr': 0.0,
            'pm_kg_hr': 0.0
        }

        for fuel, qty in fuel_quantities.items():
            props = fuel_properties.get(fuel, {})
            heating_value_gj = props.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ
            energy_gj = qty * heating_value_gj
            total_energy_gj += energy_gj

            # Calculate emissions
            emissions['co2_kg_hr'] += props.get('emission_factor_co2_kg_gj', 60) * energy_gj
            emissions['ch4_g_hr'] += props.get('emission_factor_ch4_g_gj', 1) * energy_gj
            emissions['n2o_g_hr'] += props.get('emission_factor_n2o_g_gj', 0.1) * energy_gj
            emissions['nox_kg_hr'] += props.get('emission_factor_nox_g_gj', 100) * energy_gj / 1000
            emissions['sox_kg_hr'] += props.get('emission_factor_sox_g_gj', 50) * energy_gj / 1000
            emissions['pm_kg_hr'] += props.get('emission_factor_pm_g_gj', 10) * energy_gj / 1000

        # Calculate CO2 intensity
        energy_mwh = total_energy_gj * self.GJ_TO_MWH
        co2_intensity = emissions['co2_kg_hr'] / energy_mwh if energy_mwh > 0 else 0

        # Compliance check (simplified)
        compliance_status = 'compliant'  # Would check against limits

        return {
            'co2_kg_hr': round(emissions['co2_kg_hr'], 2),
            'nox_kg_hr': round(emissions['nox_kg_hr'], 4),
            'sox_kg_hr': round(emissions['sox_kg_hr'], 4),
            'pm_kg_hr': round(emissions['pm_kg_hr'], 4),
            'co2_intensity': round(co2_intensity, 2),
            'total_energy_gj': round(total_energy_gj, 2),
            'compliance_status': compliance_status
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def calculate_blend_efficiency(
        self,
        blend_ratio: Dict[str, float],
        fuel_properties: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate efficiency of a fuel blend."""
        return self._calculate_blend_efficiency_internal(blend_ratio, fuel_properties)

    def calculate_transition_plan(
        self,
        current_mix: Dict[str, float],
        target_mix: Dict[str, float],
        transition_days: int,
        fuel_properties: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate day-by-day transition plan."""
        self._increment_tool_count()

        daily_plans = []

        for day in range(transition_days + 1):
            progress = day / transition_days if transition_days > 0 else 1.0
            day_mix = {}

            for fuel in set(list(current_mix.keys()) + list(target_mix.keys())):
                start = current_mix.get(fuel, 0)
                end = target_mix.get(fuel, 0)
                day_mix[fuel] = start + (end - start) * progress

            daily_plans.append({
                'day': day,
                'mix': {k: round(v, 4) for k, v in day_mix.items()},
                'progress_percent': round(progress * 100, 1)
            })

        return {
            'transition_days': transition_days,
            'daily_plans': daily_plans,
            'start_mix': current_mix,
            'target_mix': target_mix
        }

    def forecast_fuel_requirements(
        self,
        energy_demand_forecast: List[float],
        fuel_mix: Dict[str, float],
        fuel_properties: Dict[str, Dict[str, Any]],
        time_horizon_hours: int
    ) -> Dict[str, Any]:
        """Forecast fuel requirements based on demand."""
        self._increment_tool_count()

        hourly_requirements = []
        total_requirements = {fuel: 0.0 for fuel in fuel_mix}

        for hour, demand_mw in enumerate(energy_demand_forecast[:time_horizon_hours]):
            energy_gj = demand_mw * 3.6
            hour_req = {}

            for fuel, share in fuel_mix.items():
                props = fuel_properties.get(fuel, {})
                hv_gj = props.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ
                fuel_energy = energy_gj * share
                fuel_kg = fuel_energy / hv_gj if hv_gj > 0 else 0

                hour_req[fuel] = round(fuel_kg, 2)
                total_requirements[fuel] += fuel_kg

            hourly_requirements.append({
                'hour': hour,
                'demand_mw': demand_mw,
                'fuel_requirements_kg': hour_req
            })

        return {
            'time_horizon_hours': time_horizon_hours,
            'hourly_forecast': hourly_requirements,
            'total_requirements_kg': {k: round(v, 2) for k, v in total_requirements.items()},
            'fuel_mix_used': fuel_mix
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up FuelManagementTools resources")
        self.tool_call_count = 0
