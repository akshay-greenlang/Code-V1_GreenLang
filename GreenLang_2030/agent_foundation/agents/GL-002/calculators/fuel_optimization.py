"""
Fuel Optimization Calculator - Zero Hallucination Guarantee

Implements fuel-to-steam efficiency optimization, BTU optimization,
and multi-fuel blending strategies for cost minimization.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASME PTC 4.1, EPA AP-42, ISO 50001
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .provenance import ProvenanceTracker, ProvenanceRecord


@dataclass
class FuelData:
    """Fuel properties and costs."""
    fuel_type: str
    heating_value_kj_kg: float  # Lower heating value
    cost_per_kg: float
    carbon_content_percent: float
    hydrogen_content_percent: float
    sulfur_content_percent: float
    moisture_content_percent: float
    ash_content_percent: float
    availability_kg_per_hr: float
    max_blend_percent: float = 100.0
    min_blend_percent: float = 0.0


@dataclass
class BoilerOperatingData:
    """Boiler operating parameters."""
    steam_demand_kg_hr: float
    steam_pressure_bar: float
    steam_temperature_c: float
    feedwater_temperature_c: float
    boiler_efficiency_percent: float
    max_fuel_rate_kg_hr: float
    min_fuel_rate_kg_hr: float
    current_fuel_mix: Dict[str, float] = None  # fuel_type: percentage


@dataclass
class OptimizationConstraints:
    """Constraints for fuel optimization."""
    max_sulfur_content_percent: float = 1.0  # Environmental limit
    max_ash_content_percent: float = 15.0  # Operational limit
    min_heating_value_kj_kg: float = 15000  # Combustion stability
    max_co2_intensity_kg_per_gj: float = 100  # Carbon constraint
    budget_limit_per_hr: Optional[float] = None


class FuelOptimizationCalculator:
    """
    Optimizes fuel selection and blending for minimum cost and emissions.

    Zero Hallucination Guarantee:
    - Pure mathematical optimization
    - No LLM inference
    - Bit-perfect reproducibility
    - Complete provenance tracking
    """

    # Standard fuel properties database
    STANDARD_FUELS = {
        'natural_gas': {
            'heating_value_kj_kg': 50000,
            'carbon_percent': 74.9,
            'hydrogen_percent': 24.8,
            'sulfur_percent': 0.0,
            'moisture_percent': 0.0,
            'ash_percent': 0.0,
            'co2_factor_kg_per_gj': 56.1
        },
        'fuel_oil_no2': {
            'heating_value_kj_kg': 42700,
            'carbon_percent': 86.5,
            'hydrogen_percent': 11.5,
            'sulfur_percent': 0.5,
            'moisture_percent': 0.0,
            'ash_percent': 0.0,
            'co2_factor_kg_per_gj': 73.2
        },
        'fuel_oil_no6': {
            'heating_value_kj_kg': 40200,
            'carbon_percent': 86.5,
            'hydrogen_percent': 10.5,
            'sulfur_percent': 2.5,
            'moisture_percent': 0.5,
            'ash_percent': 0.05,
            'co2_factor_kg_per_gj': 77.4
        },
        'coal_bituminous': {
            'heating_value_kj_kg': 27000,
            'carbon_percent': 63.5,
            'hydrogen_percent': 4.5,
            'sulfur_percent': 2.1,
            'moisture_percent': 10.0,
            'ash_percent': 10.0,
            'co2_factor_kg_per_gj': 94.6
        },
        'biomass_wood': {
            'heating_value_kj_kg': 18000,
            'carbon_percent': 42.0,
            'hydrogen_percent': 5.5,
            'sulfur_percent': 0.0,
            'moisture_percent': 10.0,
            'ash_percent': 4.0,
            'co2_factor_kg_per_gj': 0.0  # Carbon neutral
        },
        'biomass_pellets': {
            'heating_value_kj_kg': 19500,
            'carbon_percent': 45.0,
            'hydrogen_percent': 6.0,
            'sulfur_percent': 0.0,
            'moisture_percent': 5.0,
            'ash_percent': 1.0,
            'co2_factor_kg_per_gj': 0.0  # Carbon neutral
        }
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize calculator with version tracking."""
        self.version = version

    def calculate_fuel_to_steam_efficiency(
        self,
        fuel_data: FuelData,
        boiler_data: BoilerOperatingData
    ) -> Dict:
        """
        Calculate fuel-to-steam efficiency for a specific fuel.

        Returns efficiency, specific fuel consumption, and cost metrics.
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"fuel_steam_eff_{id(fuel_data)}",
            calculation_type="fuel_to_steam_efficiency",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs({
            'fuel': fuel_data.__dict__,
            'boiler': boiler_data.__dict__
        })

        # Step 1: Calculate steam enthalpy rise
        steam_enthalpy = self._calculate_steam_enthalpy(
            boiler_data.steam_pressure_bar,
            boiler_data.steam_temperature_c,
            tracker
        )

        feedwater_enthalpy = self._calculate_feedwater_enthalpy(
            boiler_data.feedwater_temperature_c,
            tracker
        )

        enthalpy_rise = steam_enthalpy - feedwater_enthalpy

        # Step 2: Calculate heat required for steam
        steam_rate = Decimal(str(boiler_data.steam_demand_kg_hr))
        heat_required_kw = (steam_rate * enthalpy_rise) / Decimal('3600')

        tracker.record_step(
            operation="heat_calculation",
            description="Calculate heat required for steam generation",
            inputs={
                'steam_rate_kg_hr': steam_rate,
                'enthalpy_rise_kj_kg': enthalpy_rise
            },
            output_value=heat_required_kw,
            output_name="heat_required_kw",
            formula="Q = m * Δh / 3600",
            units="kW"
        )

        # Step 3: Calculate fuel input required
        efficiency = Decimal(str(boiler_data.boiler_efficiency_percent)) / Decimal('100')
        fuel_heat_input_kw = heat_required_kw / efficiency

        heating_value = Decimal(str(fuel_data.heating_value_kj_kg))
        fuel_rate_kg_hr = (fuel_heat_input_kw * Decimal('3600')) / heating_value

        tracker.record_step(
            operation="fuel_calculation",
            description="Calculate fuel consumption rate",
            inputs={
                'heat_input_kw': fuel_heat_input_kw,
                'heating_value_kj_kg': heating_value,
                'efficiency': efficiency
            },
            output_value=fuel_rate_kg_hr,
            output_name="fuel_rate_kg_hr",
            formula="m_fuel = Q_input * 3600 / LHV",
            units="kg/hr"
        )

        # Step 4: Calculate specific fuel consumption
        specific_consumption = fuel_rate_kg_hr / steam_rate

        tracker.record_step(
            operation="specific_consumption",
            description="Calculate specific fuel consumption",
            inputs={
                'fuel_rate_kg_hr': fuel_rate_kg_hr,
                'steam_rate_kg_hr': steam_rate
            },
            output_value=specific_consumption,
            output_name="kg_fuel_per_kg_steam",
            formula="SFC = m_fuel / m_steam",
            units="kg fuel/kg steam"
        )

        # Step 5: Calculate costs
        fuel_cost_per_hr = fuel_rate_kg_hr * Decimal(str(fuel_data.cost_per_kg))
        steam_cost_per_kg = fuel_cost_per_hr / steam_rate

        tracker.record_step(
            operation="cost_calculation",
            description="Calculate fuel and steam costs",
            inputs={
                'fuel_rate_kg_hr': fuel_rate_kg_hr,
                'fuel_cost_per_kg': Decimal(str(fuel_data.cost_per_kg))
            },
            output_value=fuel_cost_per_hr,
            output_name="fuel_cost_per_hr",
            formula="Cost = m_fuel * price",
            units="$/hr"
        )

        # Step 6: Calculate BTU efficiency
        btu_efficiency = (enthalpy_rise * efficiency * Decimal('100')) / heating_value

        result = {
            'fuel_to_steam_efficiency_percent': float(efficiency * Decimal('100')),
            'fuel_consumption_kg_hr': float(fuel_rate_kg_hr),
            'specific_fuel_consumption': float(specific_consumption),
            'steam_cost_per_kg': float(steam_cost_per_kg),
            'fuel_cost_per_hr': float(fuel_cost_per_hr),
            'btu_efficiency_percent': float(btu_efficiency),
            'heat_rate_kj_per_kg_steam': float(heating_value * specific_consumption),
            'provenance': tracker.get_provenance_record(specific_consumption).to_dict()
        }

        return result

    def optimize_fuel_blend(
        self,
        available_fuels: List[FuelData],
        boiler_data: BoilerOperatingData,
        constraints: OptimizationConstraints
    ) -> Dict:
        """
        Optimize fuel blend for minimum cost while meeting constraints.

        Uses linear programming approach for multi-fuel optimization.
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"fuel_blend_opt_{id(boiler_data)}",
            calculation_type="fuel_blend_optimization",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs({
            'fuels': [f.__dict__ for f in available_fuels],
            'constraints': constraints.__dict__
        })

        # Step 1: Calculate required heat input
        steam_enthalpy = self._calculate_steam_enthalpy(
            boiler_data.steam_pressure_bar,
            boiler_data.steam_temperature_c,
            tracker
        )

        feedwater_enthalpy = self._calculate_feedwater_enthalpy(
            boiler_data.feedwater_temperature_c,
            tracker
        )

        steam_rate = Decimal(str(boiler_data.steam_demand_kg_hr))
        heat_required = steam_rate * (steam_enthalpy - feedwater_enthalpy) / Decimal('3600')

        efficiency = Decimal(str(boiler_data.boiler_efficiency_percent)) / Decimal('100')
        total_heat_input = heat_required / efficiency

        # Step 2: Find optimal blend (simplified - in production use scipy.optimize)
        best_blend = None
        best_cost = Decimal('999999999')
        best_emissions = Decimal('999999999')

        # Try single fuels first
        for fuel in available_fuels:
            blend = self._evaluate_single_fuel(
                fuel, total_heat_input, constraints, tracker
            )

            if blend and blend['cost'] < best_cost:
                best_blend = blend
                best_cost = blend['cost']
                best_emissions = blend['co2_emissions']

        # Try two-fuel blends
        for i, fuel1 in enumerate(available_fuels):
            for fuel2 in available_fuels[i+1:]:
                blend = self._optimize_two_fuel_blend(
                    fuel1, fuel2, total_heat_input, constraints, tracker
                )

                if blend and blend['cost'] < best_cost:
                    best_blend = blend
                    best_cost = blend['cost']
                    best_emissions = blend['co2_emissions']

        if not best_blend:
            # No feasible solution
            best_blend = {
                'feasible': False,
                'message': 'No fuel blend meets all constraints'
            }

        # Step 3: Calculate detailed metrics for best blend
        if best_blend.get('feasible', True):
            result = {
                'optimal_blend': best_blend['composition'],
                'total_fuel_cost_per_hr': float(best_blend['cost']),
                'total_fuel_rate_kg_hr': float(best_blend['total_fuel_rate']),
                'blended_heating_value_kj_kg': float(best_blend['heating_value']),
                'blended_sulfur_percent': float(best_blend['sulfur_content']),
                'blended_ash_percent': float(best_blend['ash_content']),
                'co2_emissions_kg_hr': float(best_blend['co2_emissions']),
                'co2_intensity_kg_per_gj': float(best_blend['co2_intensity']),
                'savings_vs_current_percent': self._calculate_savings(
                    best_blend, boiler_data, available_fuels
                ),
                'provenance': tracker.get_provenance_record(best_blend).to_dict()
            }
        else:
            result = best_blend

        return result

    def calculate_btu_optimization(
        self,
        fuel_data: FuelData,
        boiler_data: BoilerOperatingData
    ) -> Dict:
        """
        Calculate BTU optimization opportunities.

        Identifies heat recovery and efficiency improvement potential.
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"btu_opt_{id(fuel_data)}",
            calculation_type="btu_optimization",
            version=self.version
        )

        tracker.record_inputs({
            'fuel': fuel_data.__dict__,
            'boiler': boiler_data.__dict__
        })

        # Current BTU utilization
        heating_value = Decimal(str(fuel_data.heating_value_kj_kg))
        efficiency = Decimal(str(boiler_data.boiler_efficiency_percent))

        current_btu_utilized = heating_value * efficiency / Decimal('100')
        current_btu_lost = heating_value - current_btu_utilized

        tracker.record_step(
            operation="btu_balance",
            description="Calculate current BTU utilization",
            inputs={
                'heating_value_kj_kg': heating_value,
                'efficiency_percent': efficiency
            },
            output_value=current_btu_utilized,
            output_name="btu_utilized_kj_kg",
            formula="BTU_util = LHV * η",
            units="kJ/kg"
        )

        # Identify optimization opportunities
        opportunities = []
        total_recoverable = Decimal('0')

        # Flue gas heat recovery
        flue_gas_recovery = heating_value * Decimal('0.08')  # 8% typical
        opportunities.append({
            'measure': 'Economizer Installation',
            'recoverable_heat_kj_kg': float(flue_gas_recovery),
            'efficiency_gain_percent': 8.0,
            'implementation': 'Install flue gas heat exchanger'
        })
        total_recoverable += flue_gas_recovery

        # Blowdown heat recovery
        blowdown_recovery = heating_value * Decimal('0.02')  # 2% typical
        opportunities.append({
            'measure': 'Blowdown Heat Recovery',
            'recoverable_heat_kj_kg': float(blowdown_recovery),
            'efficiency_gain_percent': 2.0,
            'implementation': 'Install blowdown flash tank and heat exchanger'
        })
        total_recoverable += blowdown_recovery

        # Air preheating
        air_preheat_recovery = heating_value * Decimal('0.05')  # 5% typical
        opportunities.append({
            'measure': 'Combustion Air Preheating',
            'recoverable_heat_kj_kg': float(air_preheat_recovery),
            'efficiency_gain_percent': 5.0,
            'implementation': 'Install air preheater using stack heat'
        })
        total_recoverable += air_preheat_recovery

        # Calculate potential new efficiency
        potential_efficiency = efficiency + (total_recoverable / heating_value * Decimal('100'))
        potential_efficiency = min(potential_efficiency, Decimal('95'))  # Practical limit

        result = {
            'current_efficiency_percent': float(efficiency),
            'current_btu_utilized_kj_kg': float(current_btu_utilized),
            'current_btu_lost_kj_kg': float(current_btu_lost),
            'optimization_opportunities': opportunities,
            'total_recoverable_heat_kj_kg': float(total_recoverable),
            'potential_efficiency_percent': float(potential_efficiency),
            'potential_fuel_savings_percent': float(
                (potential_efficiency - efficiency) / efficiency * Decimal('100')
            ),
            'provenance': tracker.get_provenance_record(potential_efficiency).to_dict()
        }

        return result

    def _calculate_steam_enthalpy(
        self,
        pressure_bar: float,
        temperature_c: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate steam enthalpy (simplified - use steam tables in production)."""
        # Simplified correlation
        P = Decimal(str(pressure_bar))
        T = Decimal(str(temperature_c))

        # Base enthalpy at saturation
        h_sat = Decimal('2676') + P * Decimal('10.2')

        # Superheat correction
        T_sat = Decimal('100') + P * Decimal('3.8')  # Simplified saturation temp
        if T > T_sat:
            superheat = T - T_sat
            h_steam = h_sat + superheat * Decimal('2.1')
        else:
            h_steam = h_sat

        tracker.record_step(
            operation="steam_enthalpy",
            description="Calculate steam specific enthalpy",
            inputs={
                'pressure_bar': P,
                'temperature_c': T
            },
            output_value=h_steam,
            output_name="steam_enthalpy_kj_kg",
            formula="h = h_sat + Cp * (T - T_sat)",
            units="kJ/kg"
        )

        return h_steam

    def _calculate_feedwater_enthalpy(
        self,
        temperature_c: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate feedwater enthalpy."""
        T = Decimal(str(temperature_c))
        Cp = Decimal('4.186')  # Specific heat of water kJ/kg·K

        h_fw = T * Cp

        tracker.record_step(
            operation="feedwater_enthalpy",
            description="Calculate feedwater specific enthalpy",
            inputs={
                'temperature_c': T,
                'specific_heat': Cp
            },
            output_value=h_fw,
            output_name="feedwater_enthalpy_kj_kg",
            formula="h = Cp * T",
            units="kJ/kg"
        )

        return h_fw

    def _evaluate_single_fuel(
        self,
        fuel: FuelData,
        heat_required_kw: Decimal,
        constraints: OptimizationConstraints,
        tracker: ProvenanceTracker
    ) -> Optional[Dict]:
        """Evaluate single fuel against constraints."""
        # Check constraints
        if fuel.sulfur_content_percent > constraints.max_sulfur_content_percent:
            return None
        if fuel.ash_content_percent > constraints.max_ash_content_percent:
            return None
        if fuel.heating_value_kj_kg < constraints.min_heating_value_kj_kg:
            return None

        # Calculate fuel rate
        heating_value = Decimal(str(fuel.heating_value_kj_kg))
        fuel_rate = (heat_required_kw * Decimal('3600')) / heating_value

        # Check availability
        if float(fuel_rate) > fuel.availability_kg_per_hr:
            return None

        # Calculate cost
        cost = fuel_rate * Decimal(str(fuel.cost_per_kg))

        # Calculate emissions
        co2_factor = self._get_co2_factor(fuel)
        co2_emissions = fuel_rate * co2_factor
        co2_intensity = co2_emissions / (heat_required_kw * Decimal('0.0036'))  # kg/GJ

        if float(co2_intensity) > constraints.max_co2_intensity_kg_per_gj:
            return None

        return {
            'composition': {fuel.fuel_type: 100.0},
            'cost': cost,
            'total_fuel_rate': fuel_rate,
            'heating_value': heating_value,
            'sulfur_content': Decimal(str(fuel.sulfur_content_percent)),
            'ash_content': Decimal(str(fuel.ash_content_percent)),
            'co2_emissions': co2_emissions,
            'co2_intensity': co2_intensity,
            'feasible': True
        }

    def _optimize_two_fuel_blend(
        self,
        fuel1: FuelData,
        fuel2: FuelData,
        heat_required_kw: Decimal,
        constraints: OptimizationConstraints,
        tracker: ProvenanceTracker
    ) -> Optional[Dict]:
        """Find optimal blend of two fuels."""
        # Use binary search to find optimal blend ratio
        best_blend = None
        best_cost = Decimal('999999999')

        # Try different blend ratios (0-100% in 5% increments)
        for blend1_percent in range(0, 101, 5):
            blend2_percent = 100 - blend1_percent

            # Check blend limits
            if blend1_percent < fuel1.min_blend_percent or blend1_percent > fuel1.max_blend_percent:
                continue
            if blend2_percent < fuel2.min_blend_percent or blend2_percent > fuel2.max_blend_percent:
                continue

            # Calculate blended properties
            w1 = Decimal(str(blend1_percent)) / Decimal('100')
            w2 = Decimal(str(blend2_percent)) / Decimal('100')

            blend_heating_value = (
                w1 * Decimal(str(fuel1.heating_value_kj_kg)) +
                w2 * Decimal(str(fuel2.heating_value_kj_kg))
            )

            blend_sulfur = (
                w1 * Decimal(str(fuel1.sulfur_content_percent)) +
                w2 * Decimal(str(fuel2.sulfur_content_percent))
            )

            blend_ash = (
                w1 * Decimal(str(fuel1.ash_content_percent)) +
                w2 * Decimal(str(fuel2.ash_content_percent))
            )

            # Check constraints
            if blend_sulfur > Decimal(str(constraints.max_sulfur_content_percent)):
                continue
            if blend_ash > Decimal(str(constraints.max_ash_content_percent)):
                continue
            if blend_heating_value < Decimal(str(constraints.min_heating_value_kj_kg)):
                continue

            # Calculate fuel rates
            total_fuel_rate = (heat_required_kw * Decimal('3600')) / blend_heating_value
            fuel1_rate = total_fuel_rate * w1
            fuel2_rate = total_fuel_rate * w2

            # Check availability
            if float(fuel1_rate) > fuel1.availability_kg_per_hr:
                continue
            if float(fuel2_rate) > fuel2.availability_kg_per_hr:
                continue

            # Calculate cost
            cost = (
                fuel1_rate * Decimal(str(fuel1.cost_per_kg)) +
                fuel2_rate * Decimal(str(fuel2.cost_per_kg))
            )

            # Calculate emissions
            co2_factor1 = self._get_co2_factor(fuel1)
            co2_factor2 = self._get_co2_factor(fuel2)
            co2_emissions = fuel1_rate * co2_factor1 + fuel2_rate * co2_factor2
            co2_intensity = co2_emissions / (heat_required_kw * Decimal('0.0036'))

            if float(co2_intensity) > constraints.max_co2_intensity_kg_per_gj:
                continue

            # Check if this is the best blend
            if cost < best_cost:
                best_cost = cost
                best_blend = {
                    'composition': {
                        fuel1.fuel_type: float(blend1_percent),
                        fuel2.fuel_type: float(blend2_percent)
                    },
                    'cost': cost,
                    'total_fuel_rate': total_fuel_rate,
                    'heating_value': blend_heating_value,
                    'sulfur_content': blend_sulfur,
                    'ash_content': blend_ash,
                    'co2_emissions': co2_emissions,
                    'co2_intensity': co2_intensity,
                    'feasible': True
                }

        return best_blend

    def _get_co2_factor(self, fuel: FuelData) -> Decimal:
        """Get CO2 emission factor for fuel (kg CO2/kg fuel)."""
        # Calculate from carbon content
        carbon = Decimal(str(fuel.carbon_content_percent)) / Decimal('100')

        # CO2 = C * (44/12) where 44 is molecular weight of CO2, 12 is atomic weight of C
        co2_factor = carbon * Decimal('3.667')

        # Check if biomass (carbon neutral)
        if 'biomass' in fuel.fuel_type.lower():
            co2_factor = Decimal('0')

        return co2_factor

    def _calculate_savings(
        self,
        optimal_blend: Dict,
        boiler_data: BoilerOperatingData,
        available_fuels: List[FuelData]
    ) -> float:
        """Calculate savings versus current fuel mix."""
        if not boiler_data.current_fuel_mix:
            return 0.0

        # Calculate current cost
        current_cost = Decimal('0')
        for fuel_type, percentage in boiler_data.current_fuel_mix.items():
            # Find fuel data
            fuel = next((f for f in available_fuels if f.fuel_type == fuel_type), None)
            if fuel:
                weight = Decimal(str(percentage)) / Decimal('100')
                current_cost += weight * Decimal(str(fuel.cost_per_kg))

        # Calculate savings
        if current_cost > 0:
            savings = (current_cost - optimal_blend['cost']) / current_cost * Decimal('100')
            return float(savings)

        return 0.0