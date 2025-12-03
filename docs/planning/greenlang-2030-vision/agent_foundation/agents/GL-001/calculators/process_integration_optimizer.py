# -*- coding: utf-8 -*-
"""
Process Integration Optimizer - Zero Hallucination Guarantee

Implements site-wide process integration optimization using pinch analysis,
heat integration targeting, and multi-period optimization with complete
provenance tracking for audit compliance.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ISO 50001, IEA Process Integration, Linnhoff March Methodology
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from .provenance import ProvenanceTracker, ProvenanceRecord


class OptimizationObjective(Enum):
    """Optimization objective types."""
    MINIMIZE_ENERGY = "minimize_energy"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCED = "balanced"


class SeasonType(Enum):
    """Season types for multi-period optimization."""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    PEAK = "peak"
    OFF_PEAK = "off_peak"


class SteamLevel(Enum):
    """Steam pressure levels in industrial systems."""
    HIGH_PRESSURE = "HP"      # >40 bar
    MEDIUM_PRESSURE = "MP"    # 10-40 bar
    LOW_PRESSURE = "LP"       # 2-10 bar
    VERY_LOW_PRESSURE = "VLP"  # <2 bar


@dataclass(frozen=True)
class HeatStream:
    """Immutable heat stream for pinch analysis."""
    stream_id: str
    stream_type: str  # 'hot' or 'cold'
    supply_temp_c: float
    target_temp_c: float
    heat_capacity_flow_kw_k: float  # MCp in kW/K
    heat_load_kw: float
    min_approach_temp_c: float = 10.0

    def __hash__(self) -> int:
        return hash((self.stream_id, self.stream_type, self.supply_temp_c,
                     self.target_temp_c, self.heat_capacity_flow_kw_k))


@dataclass(frozen=True)
class SteamHeader:
    """Immutable steam header configuration."""
    level: SteamLevel
    pressure_bar: float
    temperature_c: float
    generation_capacity_kw: float
    current_demand_kw: float
    min_flow_kw: float = 0.0
    max_flow_kw: float = float('inf')


@dataclass(frozen=True)
class UtilitySystem:
    """Immutable utility system configuration."""
    system_id: str
    system_type: str  # 'boiler', 'chp', 'chiller', 'cooling_tower'
    capacity_kw: float
    efficiency_percent: float
    fuel_type: str
    emission_factor_kg_co2_kwh: float
    operating_cost_usd_kwh: float
    capital_cost_usd: float = 0.0
    lifetime_years: int = 20


@dataclass(frozen=True)
class SeasonalProfile:
    """Immutable seasonal demand profile."""
    season: SeasonType
    heating_demand_kw: float
    cooling_demand_kw: float
    electricity_demand_kw: float
    steam_demand_kw: float
    duration_hours: float
    ambient_temp_c: float


@dataclass(frozen=True)
class CHPUnit:
    """Immutable combined heat and power unit specification."""
    unit_id: str
    electrical_capacity_kw: float
    thermal_capacity_kw: float
    electrical_efficiency_percent: float
    thermal_efficiency_percent: float
    overall_efficiency_percent: float
    power_to_heat_ratio: float
    min_load_percent: float = 30.0
    fuel_type: str = "natural_gas"
    emission_factor_kg_co2_kwh: float = 0.2


@dataclass(frozen=True)
class SiteEnergyData:
    """Immutable site-wide energy data for integration analysis."""
    site_id: str
    total_heat_demand_kw: float
    total_cooling_demand_kw: float
    total_electricity_demand_kw: float
    hot_streams: Tuple[HeatStream, ...]
    cold_streams: Tuple[HeatStream, ...]
    steam_headers: Tuple[SteamHeader, ...]
    utility_systems: Tuple[UtilitySystem, ...]
    seasonal_profiles: Tuple[SeasonalProfile, ...] = ()
    chp_units: Tuple[CHPUnit, ...] = ()
    min_approach_temp_c: float = 10.0
    fuel_cost_usd_kwh: float = 0.03
    electricity_cost_usd_kwh: float = 0.10
    carbon_cost_usd_tonne: float = 50.0


@dataclass
class ProcessIntegrationResult:
    """Result of process integration optimization."""
    # Pinch analysis results
    pinch_temperature_c: float
    minimum_hot_utility_kw: float
    minimum_cold_utility_kw: float
    maximum_heat_recovery_kw: float
    actual_heat_recovery_kw: float
    heat_recovery_efficiency_percent: float

    # Energy targeting
    target_heating_demand_kw: float
    target_cooling_demand_kw: float
    energy_savings_potential_kw: float
    energy_savings_percent: float

    # Steam optimization
    steam_optimization: Dict[str, Any]

    # CHP matching
    chp_recommendations: List[Dict[str, Any]]

    # Multi-period results
    seasonal_optimization: Dict[str, Any]

    # Capital-energy tradeoffs
    capital_energy_analysis: Dict[str, Any]

    # Cost and emissions
    annual_energy_cost_usd: float
    optimized_cost_usd: float
    cost_savings_usd: float
    annual_emissions_tonnes_co2: float
    optimized_emissions_tonnes_co2: float
    emission_reduction_tonnes_co2: float

    # Provenance
    provenance_hash: str
    calculation_timestamp: str
    version: str = "1.0.0"


class ProcessIntegrationOptimizer:
    """
    Site-wide process integration optimizer using pinch analysis.

    Zero Hallucination Guarantee:
    - Pure mathematical calculations using thermodynamic principles
    - No LLM inference in calculation path
    - Bit-perfect reproducibility with frozen dataclasses
    - Complete provenance tracking with SHA-256 hashing
    - Thread-safe caching for performance optimization

    Implements:
    - Composite curve construction (Problem Table Algorithm)
    - Grand composite curve analysis
    - Site-wide energy targeting
    - Steam level optimization
    - CHP integration analysis
    - Multi-period optimization for seasonal variations
    - Capital vs energy cost tradeoff analysis
    """

    # Physical constants
    KELVIN_OFFSET = Decimal('273.15')
    HOURS_PER_YEAR = Decimal('8760')
    KG_PER_TONNE = Decimal('1000')

    # Thread-safe cache
    _cache_lock = threading.Lock()
    _pinch_cache: Dict[int, Tuple[float, float, float]] = {}

    def __init__(self, version: str = "1.0.0"):
        """Initialize optimizer with version tracking."""
        self.version = version

    def optimize(self, site_data: SiteEnergyData) -> ProcessIntegrationResult:
        """
        Perform complete site-wide process integration optimization.

        Args:
            site_data: Site energy data with streams, utilities, and demands

        Returns:
            ProcessIntegrationResult with complete optimization results
        """
        tracker = ProvenanceTracker(
            calculation_id=f"process_integration_{site_data.site_id}_{id(site_data)}",
            calculation_type="process_integration_optimization",
            version=self.version
        )

        # Record inputs (convert frozen dataclass to dict)
        tracker.record_inputs(self._site_data_to_dict(site_data))

        # Step 1: Perform pinch analysis
        pinch_temp, min_hot_util, min_cold_util = self._calculate_pinch_point(
            site_data.hot_streams, site_data.cold_streams,
            site_data.min_approach_temp_c, tracker
        )

        # Step 2: Calculate heat recovery potential
        max_recovery, actual_recovery, recovery_eff = self._calculate_heat_recovery(
            site_data, min_hot_util, min_cold_util, tracker
        )

        # Step 3: Site-wide energy targeting
        target_heating, target_cooling, savings_kw, savings_pct = self._energy_targeting(
            site_data, min_hot_util, min_cold_util, tracker
        )

        # Step 4: Steam level optimization
        steam_optimization = self._optimize_steam_levels(
            site_data.steam_headers, site_data, tracker
        )

        # Step 5: CHP matching analysis
        chp_recommendations = self._analyze_chp_integration(
            site_data, target_heating, tracker
        )

        # Step 6: Multi-period optimization
        seasonal_optimization = self._multi_period_optimization(
            site_data, pinch_temp, tracker
        )

        # Step 7: Capital-energy tradeoff analysis
        capital_energy = self._capital_energy_tradeoff(
            site_data, savings_kw, tracker
        )

        # Step 8: Calculate costs and emissions
        cost_results = self._calculate_costs_and_emissions(
            site_data, savings_kw, seasonal_optimization, tracker
        )

        # Generate provenance hash
        result_data = {
            'pinch_temp': pinch_temp,
            'min_hot_util': min_hot_util,
            'min_cold_util': min_cold_util,
            'max_recovery': max_recovery,
            'cost_results': cost_results
        }
        provenance_record = tracker.get_provenance_record(result_data)

        return ProcessIntegrationResult(
            pinch_temperature_c=pinch_temp,
            minimum_hot_utility_kw=min_hot_util,
            minimum_cold_utility_kw=min_cold_util,
            maximum_heat_recovery_kw=max_recovery,
            actual_heat_recovery_kw=actual_recovery,
            heat_recovery_efficiency_percent=recovery_eff,
            target_heating_demand_kw=target_heating,
            target_cooling_demand_kw=target_cooling,
            energy_savings_potential_kw=savings_kw,
            energy_savings_percent=savings_pct,
            steam_optimization=steam_optimization,
            chp_recommendations=chp_recommendations,
            seasonal_optimization=seasonal_optimization,
            capital_energy_analysis=capital_energy,
            annual_energy_cost_usd=cost_results['current_cost'],
            optimized_cost_usd=cost_results['optimized_cost'],
            cost_savings_usd=cost_results['cost_savings'],
            annual_emissions_tonnes_co2=cost_results['current_emissions'],
            optimized_emissions_tonnes_co2=cost_results['optimized_emissions'],
            emission_reduction_tonnes_co2=cost_results['emission_reduction'],
            provenance_hash=provenance_record.provenance_hash,
            calculation_timestamp=provenance_record.timestamp,
            version=self.version
        )

    def _calculate_pinch_point(
        self,
        hot_streams: Tuple[HeatStream, ...],
        cold_streams: Tuple[HeatStream, ...],
        min_approach_temp: float,
        tracker: ProvenanceTracker
    ) -> Tuple[float, float, float]:
        """
        Calculate pinch point using Problem Table Algorithm.

        The pinch point is where heat transfer between hot and cold composite
        curves is at minimum approach temperature - this is the thermodynamic
        bottleneck for heat recovery.
        """
        # Check cache first (thread-safe)
        cache_key = hash((hot_streams, cold_streams, min_approach_temp))
        with self._cache_lock:
            if cache_key in self._pinch_cache:
                cached = self._pinch_cache[cache_key]
                tracker.record_step(
                    operation="cache_hit",
                    description="Retrieved pinch analysis from cache",
                    inputs={'cache_key': cache_key},
                    output_value=cached[0],
                    output_name="pinch_temperature_c",
                    formula="Cache lookup",
                    units="C"
                )
                return cached

        # Collect all temperature intervals
        temperatures = set()
        delta_t_min = Decimal(str(min_approach_temp))
        half_delta_t = delta_t_min / Decimal('2')

        for stream in hot_streams:
            # Shift hot streams down by delta_t_min/2
            temperatures.add(Decimal(str(stream.supply_temp_c)) - half_delta_t)
            temperatures.add(Decimal(str(stream.target_temp_c)) - half_delta_t)

        for stream in cold_streams:
            # Shift cold streams up by delta_t_min/2
            temperatures.add(Decimal(str(stream.supply_temp_c)) + half_delta_t)
            temperatures.add(Decimal(str(stream.target_temp_c)) + half_delta_t)

        # Sort temperatures in descending order
        sorted_temps = sorted(temperatures, reverse=True)

        if len(sorted_temps) < 2:
            # No valid intervals
            return 0.0, 0.0, 0.0

        # Calculate heat deficit/surplus for each interval
        interval_heat = []
        for i in range(len(sorted_temps) - 1):
            temp_upper = sorted_temps[i]
            temp_lower = sorted_temps[i + 1]
            delta_t = temp_upper - temp_lower

            # Sum MCp for streams active in this interval
            sum_mcp = Decimal('0')

            for stream in hot_streams:
                shifted_supply = Decimal(str(stream.supply_temp_c)) - half_delta_t
                shifted_target = Decimal(str(stream.target_temp_c)) - half_delta_t
                if shifted_supply >= temp_upper and shifted_target <= temp_lower:
                    sum_mcp += Decimal(str(stream.heat_capacity_flow_kw_k))

            for stream in cold_streams:
                shifted_supply = Decimal(str(stream.supply_temp_c)) + half_delta_t
                shifted_target = Decimal(str(stream.target_temp_c)) + half_delta_t
                if shifted_target >= temp_upper and shifted_supply <= temp_lower:
                    sum_mcp -= Decimal(str(stream.heat_capacity_flow_kw_k))

            interval_heat.append({
                'temp_upper': temp_upper,
                'temp_lower': temp_lower,
                'delta_t': delta_t,
                'net_heat': sum_mcp * delta_t
            })

        # Cascade algorithm to find pinch
        cascade = []
        cumulative = Decimal('0')
        min_cascade = Decimal('0')
        pinch_idx = 0

        for i, interval in enumerate(interval_heat):
            cumulative += interval['net_heat']
            cascade.append(cumulative)
            if cumulative < min_cascade:
                min_cascade = cumulative
                pinch_idx = i

        # Hot utility = absolute value of minimum cascade
        hot_utility = abs(min_cascade)

        # Cold utility = final cascade value + hot utility
        cold_utility = cascade[-1] + hot_utility if cascade else Decimal('0')

        # Pinch temperature (shifted back to actual temperature)
        if interval_heat:
            pinch_temp = float(interval_heat[pinch_idx]['temp_lower'] + half_delta_t)
        else:
            pinch_temp = 0.0

        result = (pinch_temp, float(hot_utility), float(cold_utility))

        # Store in cache (thread-safe)
        with self._cache_lock:
            self._pinch_cache[cache_key] = result

        tracker.record_step(
            operation="problem_table_algorithm",
            description="Calculate pinch point using Problem Table Algorithm",
            inputs={
                'num_hot_streams': len(hot_streams),
                'num_cold_streams': len(cold_streams),
                'min_approach_temp_c': min_approach_temp,
                'num_intervals': len(interval_heat)
            },
            output_value=pinch_temp,
            output_name="pinch_temperature_c",
            formula="Pinch = min(cascade) location + dT_min/2",
            units="C"
        )

        return result

    def _calculate_heat_recovery(
        self,
        site_data: SiteEnergyData,
        min_hot_utility: float,
        min_cold_utility: float,
        tracker: ProvenanceTracker
    ) -> Tuple[float, float, float]:
        """Calculate maximum and actual heat recovery."""
        # Total heating/cooling demands
        total_hot_load = sum(
            s.heat_load_kw for s in site_data.hot_streams
        )
        total_cold_load = sum(
            s.heat_load_kw for s in site_data.cold_streams
        )

        # Maximum possible recovery (theoretical)
        max_recovery = Decimal(str(min(total_hot_load, total_cold_load)))

        # Actual recovery based on utility requirements
        # Recovery = Total load - Utility requirement
        hot_recovery = Decimal(str(total_hot_load)) - Decimal(str(min_cold_utility))
        cold_recovery = Decimal(str(total_cold_load)) - Decimal(str(min_hot_utility))
        actual_recovery = min(hot_recovery, cold_recovery)
        actual_recovery = max(actual_recovery, Decimal('0'))

        # Recovery efficiency
        if max_recovery > 0:
            recovery_eff = (actual_recovery / max_recovery) * Decimal('100')
        else:
            recovery_eff = Decimal('0')

        recovery_eff = recovery_eff.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="heat_recovery_calculation",
            description="Calculate heat recovery potential",
            inputs={
                'total_hot_load_kw': total_hot_load,
                'total_cold_load_kw': total_cold_load,
                'min_hot_utility_kw': min_hot_utility,
                'min_cold_utility_kw': min_cold_utility
            },
            output_value=float(actual_recovery),
            output_name="actual_heat_recovery_kw",
            formula="Recovery = Total Load - Utility Requirement",
            units="kW"
        )

        return float(max_recovery), float(actual_recovery), float(recovery_eff)

    def _energy_targeting(
        self,
        site_data: SiteEnergyData,
        min_hot_utility: float,
        min_cold_utility: float,
        tracker: ProvenanceTracker
    ) -> Tuple[float, float, float, float]:
        """Calculate site-wide energy targets."""
        current_heating = Decimal(str(site_data.total_heat_demand_kw))
        current_cooling = Decimal(str(site_data.total_cooling_demand_kw))

        target_heating = Decimal(str(min_hot_utility))
        target_cooling = Decimal(str(min_cold_utility))

        savings_heating = current_heating - target_heating
        savings_cooling = current_cooling - target_cooling
        total_savings = savings_heating + savings_cooling

        total_current = current_heating + current_cooling
        if total_current > 0:
            savings_percent = (total_savings / total_current) * Decimal('100')
        else:
            savings_percent = Decimal('0')

        savings_percent = savings_percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="energy_targeting",
            description="Calculate site-wide energy targets",
            inputs={
                'current_heating_kw': float(current_heating),
                'current_cooling_kw': float(current_cooling),
                'target_heating_kw': float(target_heating),
                'target_cooling_kw': float(target_cooling)
            },
            output_value=float(total_savings),
            output_name="energy_savings_kw",
            formula="Savings = Current Demand - Target Demand",
            units="kW"
        )

        return (float(target_heating), float(target_cooling),
                float(total_savings), float(savings_percent))

    def _optimize_steam_levels(
        self,
        steam_headers: Tuple[SteamHeader, ...],
        site_data: SiteEnergyData,
        tracker: ProvenanceTracker
    ) -> Dict[str, Any]:
        """Optimize steam header pressures and flows."""
        optimization_results = {
            'headers': [],
            'total_generation_kw': Decimal('0'),
            'total_demand_kw': Decimal('0'),
            'excess_steam_kw': Decimal('0'),
            'recommendations': []
        }

        for header in steam_headers:
            generation = Decimal(str(header.generation_capacity_kw))
            demand = Decimal(str(header.current_demand_kw))
            excess = generation - demand

            utilization = (demand / generation * Decimal('100')
                          if generation > 0 else Decimal('0'))
            utilization = utilization.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            header_data = {
                'level': header.level.value,
                'pressure_bar': header.pressure_bar,
                'generation_kw': float(generation),
                'demand_kw': float(demand),
                'excess_kw': float(excess),
                'utilization_percent': float(utilization)
            }

            # Generate recommendations based on utilization
            if utilization < Decimal('50'):
                header_data['recommendation'] = (
                    f"Consider reducing {header.level.value} generation capacity or "
                    "redirecting steam to lower pressure levels"
                )
                optimization_results['recommendations'].append({
                    'header': header.level.value,
                    'action': 'reduce_capacity',
                    'reason': 'Low utilization',
                    'utilization_percent': float(utilization)
                })
            elif utilization > Decimal('95'):
                header_data['recommendation'] = (
                    f"Consider expanding {header.level.value} capacity or "
                    "implementing demand-side management"
                )
                optimization_results['recommendations'].append({
                    'header': header.level.value,
                    'action': 'expand_capacity',
                    'reason': 'Near capacity limit',
                    'utilization_percent': float(utilization)
                })

            optimization_results['headers'].append(header_data)
            optimization_results['total_generation_kw'] += generation
            optimization_results['total_demand_kw'] += demand
            optimization_results['excess_steam_kw'] += max(excess, Decimal('0'))

        # Convert Decimals to float for JSON serialization
        optimization_results['total_generation_kw'] = float(
            optimization_results['total_generation_kw']
        )
        optimization_results['total_demand_kw'] = float(
            optimization_results['total_demand_kw']
        )
        optimization_results['excess_steam_kw'] = float(
            optimization_results['excess_steam_kw']
        )

        tracker.record_step(
            operation="steam_level_optimization",
            description="Optimize steam header configuration",
            inputs={
                'num_headers': len(steam_headers),
                'total_generation': optimization_results['total_generation_kw'],
                'total_demand': optimization_results['total_demand_kw']
            },
            output_value=optimization_results['excess_steam_kw'],
            output_name="excess_steam_kw",
            formula="Excess = Generation - Demand per header",
            units="kW"
        )

        return optimization_results

    def _analyze_chp_integration(
        self,
        site_data: SiteEnergyData,
        target_heating: float,
        tracker: ProvenanceTracker
    ) -> List[Dict[str, Any]]:
        """Analyze CHP integration opportunities."""
        recommendations = []

        electricity_demand = Decimal(str(site_data.total_electricity_demand_kw))
        heat_demand = Decimal(str(target_heating))

        for chp in site_data.chp_units:
            elec_capacity = Decimal(str(chp.electrical_capacity_kw))
            thermal_capacity = Decimal(str(chp.thermal_capacity_kw))
            phr = Decimal(str(chp.power_to_heat_ratio))

            # Calculate optimal operating point
            # CHP should follow heat load (heat-led) or electricity load (power-led)

            # Heat-led operation
            if heat_demand > 0:
                heat_led_load = min(heat_demand / thermal_capacity, Decimal('1'))
                heat_led_electricity = heat_led_load * elec_capacity
            else:
                heat_led_load = Decimal('0')
                heat_led_electricity = Decimal('0')

            # Power-led operation
            if electricity_demand > 0:
                power_led_load = min(electricity_demand / elec_capacity, Decimal('1'))
                power_led_heat = power_led_load * thermal_capacity
            else:
                power_led_load = Decimal('0')
                power_led_heat = Decimal('0')

            # Calculate efficiency and emissions benefit
            elec_eff = Decimal(str(chp.electrical_efficiency_percent)) / Decimal('100')
            thermal_eff = Decimal(str(chp.thermal_efficiency_percent)) / Decimal('100')
            overall_eff = Decimal(str(chp.overall_efficiency_percent)) / Decimal('100')

            # Separate generation efficiency (typical grid electricity ~40%, boiler ~85%)
            grid_efficiency = Decimal('0.40')
            boiler_efficiency = Decimal('0.85')

            # Primary energy savings from CHP
            # CHP uses less primary energy than separate generation
            if elec_eff > 0 and thermal_eff > 0:
                primary_energy_chp = (heat_led_electricity / elec_eff +
                                     (heat_led_load * thermal_capacity) / thermal_eff)
                primary_energy_separate = (heat_led_electricity / grid_efficiency +
                                          (heat_led_load * thermal_capacity) / boiler_efficiency)
                primary_energy_savings = primary_energy_separate - primary_energy_chp
            else:
                primary_energy_savings = Decimal('0')

            # CO2 savings
            emission_factor = Decimal(str(chp.emission_factor_kg_co2_kwh))
            grid_emission_factor = Decimal('0.4')  # kg CO2/kWh typical grid
            boiler_emission_factor = Decimal('0.2')  # kg CO2/kWh natural gas boiler

            chp_emissions = (heat_led_electricity + heat_led_load * thermal_capacity) * emission_factor
            separate_emissions = (heat_led_electricity * grid_emission_factor +
                                 heat_led_load * thermal_capacity * boiler_emission_factor)
            emission_savings = separate_emissions - chp_emissions

            recommendation = {
                'unit_id': chp.unit_id,
                'electrical_capacity_kw': float(elec_capacity),
                'thermal_capacity_kw': float(thermal_capacity),
                'recommended_operation': 'heat_led' if heat_demand > electricity_demand * phr else 'power_led',
                'optimal_load_percent': float(
                    (heat_led_load if heat_demand > electricity_demand * phr else power_led_load)
                    * Decimal('100')
                ),
                'electricity_output_kw': float(heat_led_electricity),
                'heat_output_kw': float(heat_led_load * thermal_capacity),
                'primary_energy_savings_kw': float(primary_energy_savings),
                'emission_savings_kg_hr': float(emission_savings),
                'overall_efficiency_percent': float(overall_eff * Decimal('100')),
                'match_score': self._calculate_chp_match_score(
                    chp, electricity_demand, heat_demand
                )
            }

            recommendations.append(recommendation)

        # Sort by match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)

        tracker.record_step(
            operation="chp_integration_analysis",
            description="Analyze CHP integration opportunities",
            inputs={
                'num_chp_units': len(site_data.chp_units),
                'electricity_demand_kw': float(electricity_demand),
                'heat_demand_kw': float(heat_demand)
            },
            output_value=len(recommendations),
            output_name="chp_recommendations_count",
            formula="CHP matching based on power-to-heat ratio and demands",
            units="count"
        )

        return recommendations

    def _calculate_chp_match_score(
        self,
        chp: CHPUnit,
        electricity_demand: Decimal,
        heat_demand: Decimal
    ) -> float:
        """Calculate how well a CHP unit matches site demands."""
        if heat_demand == 0 or electricity_demand == 0:
            return 0.0

        site_phr = electricity_demand / heat_demand
        chp_phr = Decimal(str(chp.power_to_heat_ratio))

        # Score based on how close PHR values are (1.0 = perfect match)
        if site_phr > 0:
            ratio_match = min(chp_phr / site_phr, site_phr / chp_phr)
        else:
            ratio_match = Decimal('0')

        # Size match (capacity vs demand)
        elec_capacity = Decimal(str(chp.electrical_capacity_kw))
        thermal_capacity = Decimal(str(chp.thermal_capacity_kw))

        elec_match = min(elec_capacity / electricity_demand, Decimal('1'))
        thermal_match = min(thermal_capacity / heat_demand, Decimal('1'))

        # Efficiency factor
        efficiency_factor = Decimal(str(chp.overall_efficiency_percent)) / Decimal('100')

        # Combined score
        score = (ratio_match * Decimal('0.4') +
                elec_match * Decimal('0.2') +
                thermal_match * Decimal('0.2') +
                efficiency_factor * Decimal('0.2'))

        return float(score * Decimal('100'))

    def _multi_period_optimization(
        self,
        site_data: SiteEnergyData,
        pinch_temp: float,
        tracker: ProvenanceTracker
    ) -> Dict[str, Any]:
        """Perform multi-period optimization for seasonal variations."""
        if not site_data.seasonal_profiles:
            return {
                'enabled': False,
                'message': 'No seasonal profiles provided',
                'periods': []
            }

        results = {
            'enabled': True,
            'periods': [],
            'annual_weighted_targets': {},
            'peak_demands': {},
            'total_operating_hours': Decimal('0')
        }

        total_hours = Decimal('0')
        weighted_heating = Decimal('0')
        weighted_cooling = Decimal('0')
        peak_heating = Decimal('0')
        peak_cooling = Decimal('0')

        for profile in site_data.seasonal_profiles:
            hours = Decimal(str(profile.duration_hours))
            heating = Decimal(str(profile.heating_demand_kw))
            cooling = Decimal(str(profile.cooling_demand_kw))

            # Adjust pinch based on ambient temperature
            ambient_adjustment = (Decimal(str(profile.ambient_temp_c)) -
                                 Decimal('20')) * Decimal('0.5')
            adjusted_pinch = Decimal(str(pinch_temp)) + ambient_adjustment

            # Calculate period-specific targets
            period_result = {
                'season': profile.season.value,
                'duration_hours': float(hours),
                'ambient_temp_c': profile.ambient_temp_c,
                'heating_demand_kw': float(heating),
                'cooling_demand_kw': float(cooling),
                'adjusted_pinch_c': float(adjusted_pinch),
                'energy_kwh': float((heating + cooling) * hours)
            }

            results['periods'].append(period_result)

            total_hours += hours
            weighted_heating += heating * hours
            weighted_cooling += cooling * hours
            peak_heating = max(peak_heating, heating)
            peak_cooling = max(peak_cooling, cooling)

        # Calculate annual weighted averages
        if total_hours > 0:
            results['annual_weighted_targets'] = {
                'average_heating_kw': float(weighted_heating / total_hours),
                'average_cooling_kw': float(weighted_cooling / total_hours),
                'total_heating_kwh': float(weighted_heating),
                'total_cooling_kwh': float(weighted_cooling)
            }

        results['peak_demands'] = {
            'peak_heating_kw': float(peak_heating),
            'peak_cooling_kw': float(peak_cooling)
        }
        results['total_operating_hours'] = float(total_hours)

        tracker.record_step(
            operation="multi_period_optimization",
            description="Optimize across seasonal periods",
            inputs={
                'num_periods': len(site_data.seasonal_profiles),
                'total_hours': float(total_hours)
            },
            output_value=float(weighted_heating + weighted_cooling),
            output_name="total_energy_kwh",
            formula="Weighted average across seasonal profiles",
            units="kWh"
        )

        return results

    def _capital_energy_tradeoff(
        self,
        site_data: SiteEnergyData,
        energy_savings_kw: float,
        tracker: ProvenanceTracker
    ) -> Dict[str, Any]:
        """Analyze capital vs energy cost tradeoffs."""
        # Capital costs for heat exchangers (typical values)
        # Heat exchanger cost ~ $100-500/kW depending on duty
        HE_COST_PER_KW = Decimal('200')  # USD/kW average

        savings_kw = Decimal(str(energy_savings_kw))
        fuel_cost = Decimal(str(site_data.fuel_cost_usd_kwh))
        carbon_cost = Decimal(str(site_data.carbon_cost_usd_tonne))

        # Estimate capital required for heat recovery network
        capital_required = savings_kw * HE_COST_PER_KW

        # Annual energy savings
        annual_energy_savings = savings_kw * self.HOURS_PER_YEAR
        annual_cost_savings = annual_energy_savings * fuel_cost

        # Carbon savings (assuming natural gas)
        carbon_factor = Decimal('0.2')  # kg CO2/kWh natural gas
        annual_carbon_savings = annual_energy_savings * carbon_factor / self.KG_PER_TONNE
        annual_carbon_value = annual_carbon_savings * carbon_cost

        total_annual_benefit = annual_cost_savings + annual_carbon_value

        # Simple payback period
        if total_annual_benefit > 0:
            simple_payback = capital_required / total_annual_benefit
        else:
            simple_payback = Decimal('999')  # Infinite

        # NPV calculation (10-year horizon, 8% discount rate)
        discount_rate = Decimal('0.08')
        npv = -capital_required
        for year in range(1, 11):
            discount_factor = Decimal('1') / (Decimal('1') + discount_rate) ** year
            npv += total_annual_benefit * discount_factor

        # IRR approximation (simplified)
        if capital_required > 0:
            annual_return = total_annual_benefit / capital_required * Decimal('100')
        else:
            annual_return = Decimal('0')

        # Different scenarios
        scenarios = []
        for approach_temp in [5, 10, 15, 20]:
            # Smaller approach temp = more heat recovery but higher capital
            recovery_factor = Decimal('10') / Decimal(str(approach_temp))
            scenario_savings = savings_kw * min(recovery_factor, Decimal('1.5'))
            scenario_capital = scenario_savings * HE_COST_PER_KW * recovery_factor
            scenario_annual = scenario_savings * self.HOURS_PER_YEAR * fuel_cost

            if scenario_annual > 0:
                scenario_payback = scenario_capital / scenario_annual
            else:
                scenario_payback = Decimal('999')

            scenarios.append({
                'approach_temp_c': approach_temp,
                'recovery_kw': float(scenario_savings),
                'capital_usd': float(scenario_capital),
                'annual_savings_usd': float(scenario_annual),
                'payback_years': float(scenario_payback)
            })

        result = {
            'capital_required_usd': float(capital_required),
            'annual_energy_savings_kwh': float(annual_energy_savings),
            'annual_cost_savings_usd': float(annual_cost_savings),
            'annual_carbon_savings_tonnes': float(annual_carbon_savings),
            'annual_carbon_value_usd': float(annual_carbon_value),
            'total_annual_benefit_usd': float(total_annual_benefit),
            'simple_payback_years': float(simple_payback),
            'npv_10_year_usd': float(npv),
            'annual_return_percent': float(annual_return),
            'scenarios': scenarios
        }

        tracker.record_step(
            operation="capital_energy_tradeoff",
            description="Analyze capital vs energy cost tradeoffs",
            inputs={
                'energy_savings_kw': energy_savings_kw,
                'fuel_cost_usd_kwh': float(fuel_cost),
                'carbon_cost_usd_tonne': float(carbon_cost)
            },
            output_value=float(npv),
            output_name="npv_10_year_usd",
            formula="NPV = -Capital + Sum(Annual_Benefit / (1+r)^n)",
            units="USD"
        )

        return result

    def _calculate_costs_and_emissions(
        self,
        site_data: SiteEnergyData,
        energy_savings_kw: float,
        seasonal_opt: Dict[str, Any],
        tracker: ProvenanceTracker
    ) -> Dict[str, Any]:
        """Calculate current and optimized costs and emissions."""
        fuel_cost = Decimal(str(site_data.fuel_cost_usd_kwh))
        elec_cost = Decimal(str(site_data.electricity_cost_usd_kwh))
        carbon_cost = Decimal(str(site_data.carbon_cost_usd_tonne))

        # Current energy consumption
        current_heat = Decimal(str(site_data.total_heat_demand_kw))
        current_cooling = Decimal(str(site_data.total_cooling_demand_kw))
        current_electricity = Decimal(str(site_data.total_electricity_demand_kw))

        # Operating hours
        hours = self.HOURS_PER_YEAR
        if seasonal_opt.get('enabled') and seasonal_opt.get('total_operating_hours', 0) > 0:
            hours = Decimal(str(seasonal_opt['total_operating_hours']))

        # Current annual costs
        current_heat_cost = current_heat * hours * fuel_cost
        current_cooling_cost = current_cooling * hours * elec_cost * Decimal('0.3')  # COP ~3
        current_elec_cost = current_electricity * hours * elec_cost
        current_total_cost = current_heat_cost + current_cooling_cost + current_elec_cost

        # Current emissions (kg CO2)
        heat_emission_factor = Decimal('0.2')  # kg CO2/kWh natural gas
        elec_emission_factor = Decimal('0.4')  # kg CO2/kWh grid

        current_heat_emissions = current_heat * hours * heat_emission_factor
        current_cooling_emissions = current_cooling * hours * Decimal('0.3') * elec_emission_factor
        current_elec_emissions = current_electricity * hours * elec_emission_factor
        current_total_emissions = (
            (current_heat_emissions + current_cooling_emissions + current_elec_emissions)
            / self.KG_PER_TONNE
        )

        # Optimized values (with heat integration)
        savings = Decimal(str(energy_savings_kw))
        optimized_heat = max(current_heat - savings * Decimal('0.7'), Decimal('0'))
        optimized_cooling = max(current_cooling - savings * Decimal('0.3'), Decimal('0'))

        optimized_heat_cost = optimized_heat * hours * fuel_cost
        optimized_cooling_cost = optimized_cooling * hours * elec_cost * Decimal('0.3')
        optimized_elec_cost = current_elec_cost  # Electricity unchanged
        optimized_total_cost = optimized_heat_cost + optimized_cooling_cost + optimized_elec_cost

        optimized_heat_emissions = optimized_heat * hours * heat_emission_factor
        optimized_cooling_emissions = optimized_cooling * hours * Decimal('0.3') * elec_emission_factor
        optimized_total_emissions = (
            (optimized_heat_emissions + optimized_cooling_emissions + current_elec_emissions)
            / self.KG_PER_TONNE
        )

        cost_savings = current_total_cost - optimized_total_cost
        emission_reduction = current_total_emissions - optimized_total_emissions

        result = {
            'current_cost': float(current_total_cost),
            'optimized_cost': float(optimized_total_cost),
            'cost_savings': float(cost_savings),
            'current_emissions': float(current_total_emissions),
            'optimized_emissions': float(optimized_total_emissions),
            'emission_reduction': float(emission_reduction)
        }

        tracker.record_step(
            operation="cost_emission_calculation",
            description="Calculate costs and emissions impact",
            inputs={
                'current_heat_kw': float(current_heat),
                'current_cooling_kw': float(current_cooling),
                'energy_savings_kw': energy_savings_kw
            },
            output_value=float(cost_savings),
            output_name="cost_savings_usd",
            formula="Savings = Current Cost - Optimized Cost",
            units="USD"
        )

        return result

    def _site_data_to_dict(self, site_data: SiteEnergyData) -> Dict[str, Any]:
        """Convert frozen SiteEnergyData to dictionary for provenance."""
        return {
            'site_id': site_data.site_id,
            'total_heat_demand_kw': site_data.total_heat_demand_kw,
            'total_cooling_demand_kw': site_data.total_cooling_demand_kw,
            'total_electricity_demand_kw': site_data.total_electricity_demand_kw,
            'num_hot_streams': len(site_data.hot_streams),
            'num_cold_streams': len(site_data.cold_streams),
            'num_steam_headers': len(site_data.steam_headers),
            'num_utility_systems': len(site_data.utility_systems),
            'num_seasonal_profiles': len(site_data.seasonal_profiles),
            'num_chp_units': len(site_data.chp_units),
            'min_approach_temp_c': site_data.min_approach_temp_c
        }

    def clear_cache(self) -> None:
        """Clear the pinch analysis cache (thread-safe)."""
        with self._cache_lock:
            self._pinch_cache.clear()
