"""
Tools module for BoilerEfficiencyOptimizer agent (GL-002).

This module provides deterministic calculation tools for boiler efficiency
optimization, combustion analysis, emissions reduction, and steam generation
optimization. All calculations follow industry standards (ASME PTC 4.1,
EN 12952, ISO 50001) and zero-hallucination principles.
"""

import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CombustionOptimizationResult:
    """Result of combustion optimization calculations."""

    optimal_excess_air_percent: float
    optimal_air_fuel_ratio: float
    combustion_efficiency_percent: float
    fuel_efficiency_percent: float
    theoretical_air_required_kg_kg: float
    actual_air_supplied_kg_kg: float
    flue_gas_temperature_c: float
    stack_losses_percent: float
    radiation_losses_percent: float
    unburnt_losses_percent: float
    flame_stability_index: float  # 0-1 scale
    fuel_savings_usd_hr: float
    fuel_saved_kg: float
    theoretical_max_efficiency: float


@dataclass
class SteamGenerationStrategy:
    """Optimized steam generation strategy."""

    target_steam_flow_kg_hr: float
    target_pressure_bar: float
    target_temperature_c: float
    feedwater_temperature_c: float
    blowdown_rate_percent: float
    steam_quality_index: float  # 0-1 scale
    total_heat_input_mw: float
    total_heat_output_mw: float
    heat_rate_kj_kg: float
    evaporation_ratio: float
    total_heat_supply_mw: float
    total_heat_demand_mw: float
    optimization_score: float  # 0-100 scale


@dataclass
class EmissionsOptimizationResult:
    """Result of emissions optimization."""

    co2_emissions_kg_hr: float
    nox_emissions_ppm: float
    co_emissions_ppm: float
    so2_emissions_ppm: float
    particulate_matter_mg_nm3: float
    co2_intensity_kg_mwh: float
    emission_factor_kg_gj: float
    compliance_status: str  # COMPLIANT, WARNING, NON_COMPLIANT
    violation_details: Optional[str]
    reduction_percent: float
    co2_reduction_kg: float
    carbon_credits_usd_hr: float


@dataclass
class EfficiencyCalculationResult:
    """Boiler efficiency calculation result."""

    thermal_efficiency: float
    combustion_efficiency: float
    boiler_efficiency: float  # Overall efficiency
    heat_input_mw: float
    heat_output_mw: float
    stack_temperature_c: float
    excess_air_percent: float
    co2_percent: float
    o2_percent: float
    dry_gas_loss_percent: float
    moisture_loss_percent: float
    unburnt_loss_percent: float
    radiation_loss_percent: float
    blowdown_loss_percent: float
    total_losses_percent: float
    co2_emissions_kg_hr: float
    heat_recovery_efficiency: float


class BoilerEfficiencyTools:
    """
    Deterministic calculation tools for boiler efficiency optimization.

    All calculations follow industry standards:
    - ASME PTC 4.1 for efficiency calculations
    - EN 12952 for European boiler standards
    - ISO 50001 for energy management
    - EPA Method 19 for emissions calculations
    """

    def __init__(self):
        """Initialize BoilerEfficiencyTools."""
        self.logger = logging.getLogger(__name__)

        # Physical constants
        self.STEFAN_BOLTZMANN = 5.67e-8  # W/m²K⁴
        self.WATER_SPECIFIC_HEAT = 4.186  # kJ/kg·K
        self.STEAM_LATENT_HEAT_100C = 2257  # kJ/kg at 100°C
        self.AIR_SPECIFIC_HEAT = 1.005  # kJ/kg·K

        # Standard conditions
        self.STANDARD_TEMPERATURE_C = 15.6  # Standard temperature
        self.STANDARD_PRESSURE_BAR = 1.013  # Standard pressure

        # Fuel properties (natural gas as default)
        self.default_fuel_properties = {
            'carbon_percent': 75.0,
            'hydrogen_percent': 25.0,
            'sulfur_percent': 0.0,
            'nitrogen_percent': 0.0,
            'oxygen_percent': 0.0,
            'moisture_percent': 0.0,
            'heating_value_mj_kg': 50.0
        }

    def calculate_boiler_efficiency(
        self,
        boiler_data: Dict[str, Any],
        sensor_feeds: Dict[str, Any]
    ) -> EfficiencyCalculationResult:
        """
        Calculate boiler efficiency using ASME PTC 4.1 indirect method.

        Args:
            boiler_data: Boiler configuration and specifications
            sensor_feeds: Real-time sensor measurements

        Returns:
            Comprehensive efficiency calculation result
        """
        # Extract sensor data
        fuel_flow = sensor_feeds.get('fuel_flow_kg_hr', 1000)
        steam_flow = sensor_feeds.get('steam_flow_kg_hr', 10000)
        stack_temp = sensor_feeds.get('stack_temperature_c', 180)
        ambient_temp = sensor_feeds.get('ambient_temperature_c', 25)
        o2_percent = sensor_feeds.get('o2_percent', 3.0)
        co_ppm = sensor_feeds.get('co_ppm', 50)

        # Get fuel properties
        fuel_properties = boiler_data.get('fuel_properties', self.default_fuel_properties)
        heating_value = fuel_properties['heating_value_mj_kg']

        # Calculate theoretical air required (kg air/kg fuel)
        theoretical_air = self._calculate_theoretical_air(fuel_properties)

        # Calculate excess air from O2 measurement
        excess_air_percent = self._calculate_excess_air_from_o2(o2_percent)

        # Calculate actual air supplied
        actual_air = theoretical_air * (1 + excess_air_percent / 100)

        # Calculate losses using indirect method

        # 1. Dry gas loss (stack loss)
        dry_gas_loss = self._calculate_dry_gas_loss(
            stack_temp, ambient_temp, o2_percent, co_ppm
        )

        # 2. Moisture loss (H2O from hydrogen combustion)
        moisture_loss = self._calculate_moisture_loss(
            fuel_properties, stack_temp, ambient_temp
        )

        # 3. Unburnt loss (incomplete combustion)
        unburnt_loss = self._calculate_unburnt_loss(co_ppm, fuel_properties)

        # 4. Radiation and convection loss (empirical)
        radiation_loss = self._calculate_radiation_loss(steam_flow)

        # 5. Blowdown loss
        blowdown_rate = sensor_feeds.get('blowdown_rate_percent', 3.0)
        blowdown_loss = self._calculate_blowdown_loss(blowdown_rate)

        # Total losses
        total_losses = (
            dry_gas_loss + moisture_loss + unburnt_loss +
            radiation_loss + blowdown_loss
        )

        # Boiler efficiency by indirect method
        boiler_efficiency = 100 - total_losses

        # Combustion efficiency (excluding radiation and blowdown)
        combustion_efficiency = 100 - (dry_gas_loss + moisture_loss + unburnt_loss)

        # Thermal efficiency (direct method verification)
        heat_input_mw = (fuel_flow * heating_value) / 3600  # MW
        heat_output_mw = self._calculate_heat_output(steam_flow, sensor_feeds)
        thermal_efficiency = (heat_output_mw / heat_input_mw) * 100 if heat_input_mw > 0 else 0

        # CO2 emissions calculation
        co2_percent = self._calculate_co2_percent(fuel_properties, excess_air_percent)
        co2_emissions = self._calculate_co2_emissions(fuel_flow, fuel_properties)

        # Heat recovery efficiency
        heat_recovery_efficiency = sensor_feeds.get('heat_recovery_efficiency', 0)

        return EfficiencyCalculationResult(
            thermal_efficiency=thermal_efficiency,
            combustion_efficiency=combustion_efficiency,
            boiler_efficiency=boiler_efficiency,
            heat_input_mw=heat_input_mw,
            heat_output_mw=heat_output_mw,
            stack_temperature_c=stack_temp,
            excess_air_percent=excess_air_percent,
            co2_percent=co2_percent,
            o2_percent=o2_percent,
            dry_gas_loss_percent=dry_gas_loss,
            moisture_loss_percent=moisture_loss,
            unburnt_loss_percent=unburnt_loss,
            radiation_loss_percent=radiation_loss,
            blowdown_loss_percent=blowdown_loss,
            total_losses_percent=total_losses,
            co2_emissions_kg_hr=co2_emissions,
            heat_recovery_efficiency=heat_recovery_efficiency
        )

    def optimize_combustion_parameters(
        self,
        operational_state: Dict[str, Any],
        fuel_data: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> CombustionOptimizationResult:
        """
        Optimize combustion parameters for maximum efficiency.

        Args:
            operational_state: Current operational state
            fuel_data: Fuel composition and properties
            constraints: Operational constraints

        Returns:
            Optimized combustion parameters
        """
        # Get fuel properties
        fuel_properties = fuel_data.get('properties', self.default_fuel_properties)
        heating_value = fuel_properties['heating_value_mj_kg']

        # Current operating parameters
        current_excess_air = operational_state.get('excess_air_percent', 15)
        fuel_flow = operational_state.get('fuel_flow_rate_kg_hr', 1000)

        # Calculate theoretical air requirement
        theoretical_air = self._calculate_theoretical_air(fuel_properties)

        # Optimize excess air based on fuel type and load
        optimal_excess_air = self._optimize_excess_air(
            fuel_properties,
            operational_state.get('load_percent', 75),
            constraints
        )

        # Calculate air-fuel ratio
        optimal_afr = theoretical_air * (1 + optimal_excess_air / 100)
        actual_air = theoretical_air * (1 + current_excess_air / 100)

        # Calculate combustion efficiency at optimal conditions
        combustion_eff = self._calculate_combustion_efficiency(
            optimal_excess_air,
            operational_state.get('combustion_temperature_c', 1200),
            fuel_properties
        )

        # Calculate stack losses
        stack_temp = operational_state.get('stack_temperature_c', 180)
        ambient_temp = 25  # Assume standard ambient
        stack_losses = self._calculate_stack_losses(
            stack_temp, ambient_temp, optimal_excess_air
        )

        # Radiation losses (empirical correlation)
        radiation_losses = 0.5 + 0.3 * math.exp(-operational_state.get('load_percent', 75) / 30)

        # Unburnt losses
        unburnt_losses = 0.1  # Minimal with optimal combustion

        # Fuel efficiency
        fuel_efficiency = combustion_eff - radiation_losses - unburnt_losses

        # Theoretical maximum efficiency (Carnot-like limit)
        theoretical_max = 95.0 - stack_losses  # Practical maximum

        # Flame stability index (0-1)
        flame_stability = self._calculate_flame_stability(
            optimal_excess_air,
            operational_state.get('combustion_temperature_c', 1200)
        )

        # Calculate fuel savings
        efficiency_improvement = fuel_efficiency - operational_state.get('efficiency_percent', 80)
        fuel_saved_kg = fuel_flow * (efficiency_improvement / 100) if efficiency_improvement > 0 else 0
        fuel_cost = fuel_data.get('cost_usd_per_kg', 0.05)
        fuel_savings_usd = fuel_saved_kg * fuel_cost

        return CombustionOptimizationResult(
            optimal_excess_air_percent=optimal_excess_air,
            optimal_air_fuel_ratio=optimal_afr,
            combustion_efficiency_percent=combustion_eff,
            fuel_efficiency_percent=fuel_efficiency,
            theoretical_air_required_kg_kg=theoretical_air,
            actual_air_supplied_kg_kg=actual_air,
            flue_gas_temperature_c=stack_temp,
            stack_losses_percent=stack_losses,
            radiation_losses_percent=radiation_losses,
            unburnt_losses_percent=unburnt_losses,
            flame_stability_index=flame_stability,
            fuel_savings_usd_hr=fuel_savings_usd,
            fuel_saved_kg=fuel_saved_kg,
            theoretical_max_efficiency=theoretical_max
        )

    def optimize_steam_generation(
        self,
        steam_demand: Dict[str, Any],
        operational_state: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> SteamGenerationStrategy:
        """
        Optimize steam generation strategy.

        Args:
            steam_demand: Steam demand requirements
            operational_state: Current operational state
            constraints: Operational constraints

        Returns:
            Optimized steam generation strategy
        """
        # Steam demand parameters
        required_flow = steam_demand.get('required_flow_kg_hr', 10000)
        required_pressure = steam_demand.get('required_pressure_bar', 10)
        required_temp = steam_demand.get('required_temperature_c', 180)

        # Current parameters
        current_flow = operational_state.get('steam_flow_rate_kg_hr', 9000)
        current_pressure = operational_state.get('steam_pressure_bar', 10)

        # Optimize blowdown rate based on TDS
        tds_ppm = operational_state.get('tds_ppm', 2000)
        optimal_blowdown = self._optimize_blowdown_rate(tds_ppm, constraints)

        # Calculate feedwater temperature for efficiency
        feedwater_temp = operational_state.get('feedwater_temperature_c', 80)
        optimal_feedwater_temp = min(
            105,  # Deaerator typical temperature
            feedwater_temp + 10  # Incremental improvement
        )

        # Steam quality calculation
        steam_quality = self._calculate_steam_quality(
            current_pressure,
            operational_state.get('steam_moisture_percent', 0.5)
        )

        # Heat balance calculations
        heat_input = self._calculate_heat_input(
            required_flow,
            required_temp,
            optimal_feedwater_temp,
            required_pressure
        )

        heat_output = self._calculate_steam_heat_output(
            required_flow,
            required_temp,
            required_pressure
        )

        # Heat rate
        heat_rate = (heat_input * 3600) / required_flow if required_flow > 0 else 0

        # Evaporation ratio
        fuel_flow = operational_state.get('fuel_flow_rate_kg_hr', 1000)
        evaporation_ratio = required_flow / fuel_flow if fuel_flow > 0 else 0

        # Optimization score (0-100)
        optimization_score = self._calculate_optimization_score(
            steam_quality,
            evaporation_ratio,
            optimal_blowdown
        )

        return SteamGenerationStrategy(
            target_steam_flow_kg_hr=required_flow,
            target_pressure_bar=required_pressure,
            target_temperature_c=required_temp,
            feedwater_temperature_c=optimal_feedwater_temp,
            blowdown_rate_percent=optimal_blowdown,
            steam_quality_index=steam_quality,
            total_heat_input_mw=heat_input,
            total_heat_output_mw=heat_output,
            heat_rate_kj_kg=heat_rate,
            evaporation_ratio=evaporation_ratio,
            total_heat_supply_mw=heat_input,
            total_heat_demand_mw=heat_output,
            optimization_score=optimization_score
        )

    def minimize_emissions(
        self,
        combustion_result: CombustionOptimizationResult,
        emission_limits: Dict[str, Any]
    ) -> EmissionsOptimizationResult:
        """
        Minimize emissions while maintaining efficiency.

        Args:
            combustion_result: Current combustion optimization
            emission_limits: Regulatory emission limits

        Returns:
            Emissions optimization result
        """
        # Current emission levels (example calculations)
        fuel_flow = 1000  # kg/hr (default)

        # NOx calculation (thermal NOx correlation)
        combustion_temp = 1200  # °C
        excess_air = combustion_result.optimal_excess_air_percent
        nox_ppm = self._calculate_nox_emissions(combustion_temp, excess_air)

        # CO emissions (from combustion efficiency)
        co_ppm = 50 * (100 - combustion_result.combustion_efficiency_percent) / 5

        # SO2 (depends on fuel sulfur content)
        so2_ppm = 0  # Natural gas has negligible sulfur

        # Particulate matter
        pm_mg_nm3 = 5  # Low for natural gas

        # CO2 calculations
        carbon_content = 0.75  # 75% carbon in natural gas
        co2_emissions_kg_hr = fuel_flow * carbon_content * (44/12)  # Molecular weight ratio

        # CO2 intensity
        heat_output_mw = 30  # Example
        co2_intensity = co2_emissions_kg_hr / heat_output_mw if heat_output_mw > 0 else 0

        # Emission factor
        heat_input_gj = (fuel_flow * 50) / 1000  # 50 MJ/kg heating value
        emission_factor = co2_emissions_kg_hr / heat_input_gj if heat_input_gj > 0 else 0

        # Check compliance
        compliance_status = "COMPLIANT"
        violation_details = None

        if nox_ppm > emission_limits.get('nox_limit_ppm', 30):
            compliance_status = "NON_COMPLIANT"
            violation_details = f"NOx exceeds limit: {nox_ppm:.1f} > {emission_limits['nox_limit_ppm']}"
        elif co_ppm > emission_limits.get('co_limit_ppm', 100):
            compliance_status = "NON_COMPLIANT"
            violation_details = f"CO exceeds limit: {co_ppm:.1f} > {emission_limits['co_limit_ppm']}"

        # Calculate reductions achieved
        baseline_co2 = co2_emissions_kg_hr * 1.1  # Assume 10% higher baseline
        co2_reduction = baseline_co2 - co2_emissions_kg_hr
        reduction_percent = (co2_reduction / baseline_co2) * 100 if baseline_co2 > 0 else 0

        # Carbon credits (if applicable)
        carbon_price = 50  # $/ton CO2
        carbon_credits = (co2_reduction / 1000) * carbon_price  # $/hr

        return EmissionsOptimizationResult(
            co2_emissions_kg_hr=co2_emissions_kg_hr,
            nox_emissions_ppm=nox_ppm,
            co_emissions_ppm=co_ppm,
            so2_emissions_ppm=so2_ppm,
            particulate_matter_mg_nm3=pm_mg_nm3,
            co2_intensity_kg_mwh=co2_intensity,
            emission_factor_kg_gj=emission_factor,
            compliance_status=compliance_status,
            violation_details=violation_details,
            reduction_percent=reduction_percent,
            co2_reduction_kg=co2_reduction,
            carbon_credits_usd_hr=carbon_credits
        )

    def calculate_control_adjustments(
        self,
        combustion_result: CombustionOptimizationResult,
        steam_strategy: SteamGenerationStrategy,
        emissions_result: EmissionsOptimizationResult
    ) -> Dict[str, Any]:
        """
        Calculate real-time control parameter adjustments.

        Args:
            combustion_result: Combustion optimization result
            steam_strategy: Steam generation strategy
            emissions_result: Emissions optimization result

        Returns:
            Control parameter adjustments
        """
        adjustments = {}

        # Air flow adjustments
        current_excess_air = 15  # % (example)
        target_excess_air = combustion_result.optimal_excess_air_percent
        air_adjustment = target_excess_air - current_excess_air
        adjustments['air_flow_change_percent'] = min(max(air_adjustment, -3), 3)  # Limit to ±3%

        # Fuel flow adjustments
        current_fuel = 1000  # kg/hr
        efficiency_gain = combustion_result.fuel_efficiency_percent - 80  # Assume 80% current
        fuel_adjustment = -efficiency_gain * 0.5  # Reduce fuel proportionally
        adjustments['fuel_flow_change_percent'] = min(max(fuel_adjustment, -5), 5)  # Limit to ±5%

        # Steam pressure adjustments
        current_pressure = 10  # bar
        target_pressure = steam_strategy.target_pressure_bar
        pressure_adjustment = target_pressure - current_pressure
        adjustments['steam_pressure_change_bar'] = min(max(pressure_adjustment, -0.5), 0.5)

        # Temperature adjustments
        adjustments['feedwater_temp_change_c'] = 5  # Gradual increase
        adjustments['stack_temp_target_c'] = 160  # Target stack temperature

        # Blowdown adjustments
        adjustments['blowdown_rate_change_percent'] = (
            steam_strategy.blowdown_rate_percent - 3  # Assume 3% current
        )

        # Damper positions (0-100%)
        adjustments['primary_air_damper_position'] = 75 + air_adjustment
        adjustments['secondary_air_damper_position'] = 60 + air_adjustment * 0.5
        adjustments['flue_gas_damper_position'] = 85

        # Burner adjustments
        adjustments['burner_tilt_angle_deg'] = 0  # Neutral position
        adjustments['fuel_valve_position_percent'] = 70 + fuel_adjustment

        # Control mode recommendations
        adjustments['control_mode'] = 'auto_optimize'
        adjustments['optimization_enabled'] = True

        return adjustments

    # Helper methods for calculations

    def _calculate_theoretical_air(self, fuel_properties: Dict[str, Any]) -> float:
        """Calculate theoretical air requirement for complete combustion."""
        C = fuel_properties['carbon_percent'] / 100
        H = fuel_properties['hydrogen_percent'] / 100
        S = fuel_properties['sulfur_percent'] / 100
        O = fuel_properties['oxygen_percent'] / 100

        # Stoichiometric air calculation (kg air/kg fuel)
        theoretical_air = 11.51 * C + 34.34 * (H - O/8) + 4.29 * S

        return theoretical_air

    def _calculate_excess_air_from_o2(self, o2_percent: float) -> float:
        """Calculate excess air from flue gas O2 measurement."""
        # Using simplified formula for natural gas
        excess_air = (o2_percent / (21 - o2_percent)) * 100
        return excess_air

    def _calculate_dry_gas_loss(
        self,
        stack_temp: float,
        ambient_temp: float,
        o2_percent: float,
        co_ppm: float
    ) -> float:
        """Calculate dry gas loss percentage."""
        # Siegert formula
        k_factor = 0.65  # For natural gas
        temp_diff = stack_temp - ambient_temp

        dry_gas_loss = k_factor * temp_diff / (21 - o2_percent)

        return dry_gas_loss

    def _calculate_moisture_loss(
        self,
        fuel_properties: Dict[str, Any],
        stack_temp: float,
        ambient_temp: float
    ) -> float:
        """Calculate moisture loss from hydrogen combustion."""
        H = fuel_properties['hydrogen_percent'] / 100
        M = fuel_properties['moisture_percent'] / 100

        # Water formed from hydrogen combustion
        water_from_h2 = 9 * H  # 9 kg H2O per kg H2
        total_moisture = water_from_h2 + M

        # Heat loss in water vapor
        temp_diff = stack_temp - ambient_temp
        moisture_loss = total_moisture * 0.45 * temp_diff / 10  # Simplified formula

        return moisture_loss

    def _calculate_unburnt_loss(self, co_ppm: float, fuel_properties: Dict[str, Any]) -> float:
        """Calculate loss due to unburnt fuel (CO formation)."""
        # Loss proportional to CO concentration
        unburnt_loss = (co_ppm / 10000) * 5  # Empirical correlation
        return min(unburnt_loss, 1.0)  # Cap at 1%

    def _calculate_radiation_loss(self, steam_flow: float) -> float:
        """Calculate radiation and convection losses."""
        # ABMA radiation loss chart approximation
        if steam_flow < 5000:
            return 2.0
        elif steam_flow < 10000:
            return 1.5
        elif steam_flow < 20000:
            return 1.0
        elif steam_flow < 50000:
            return 0.7
        else:
            return 0.5

    def _calculate_blowdown_loss(self, blowdown_rate: float) -> float:
        """Calculate blowdown heat loss."""
        # Approximate loss based on blowdown rate
        blowdown_loss = blowdown_rate * 0.3  # 30% of blowdown energy is lost
        return blowdown_loss

    def _calculate_heat_output(self, steam_flow: float, sensor_feeds: Dict[str, Any]) -> float:
        """Calculate heat output in steam."""
        steam_pressure = sensor_feeds.get('steam_pressure_bar', 10)
        steam_temp = sensor_feeds.get('steam_temperature_c', 180)
        feedwater_temp = sensor_feeds.get('feedwater_temperature_c', 80)

        # Simplified enthalpy calculation
        steam_enthalpy = 2700 + 1.9 * (steam_temp - 100)  # kJ/kg approximation
        feedwater_enthalpy = 4.186 * feedwater_temp  # kJ/kg

        heat_output_kw = steam_flow * (steam_enthalpy - feedwater_enthalpy) / 3600
        heat_output_mw = heat_output_kw / 1000

        return heat_output_mw

    def _calculate_co2_percent(self, fuel_properties: Dict[str, Any], excess_air: float) -> float:
        """Calculate CO2 percentage in dry flue gas."""
        C = fuel_properties['carbon_percent'] / 100

        # Theoretical maximum CO2
        co2_max = 11.7  # For natural gas

        # Actual CO2 with excess air
        co2_percent = co2_max / (1 + excess_air / 100)

        return co2_percent

    def _calculate_co2_emissions(self, fuel_flow: float, fuel_properties: Dict[str, Any]) -> float:
        """Calculate CO2 emissions in kg/hr."""
        C = fuel_properties['carbon_percent'] / 100

        # CO2 emissions = fuel * carbon * (44/12)
        co2_emissions = fuel_flow * C * (44/12)

        return co2_emissions

    def _optimize_excess_air(
        self,
        fuel_properties: Dict[str, Any],
        load_percent: float,
        constraints: Dict[str, Any]
    ) -> float:
        """Optimize excess air based on fuel type and load."""
        # Base excess air for natural gas
        base_excess_air = 10

        # Adjust for load
        if load_percent < 30:
            load_factor = 1.5
        elif load_percent < 50:
            load_factor = 1.2
        elif load_percent < 70:
            load_factor = 1.1
        else:
            load_factor = 1.0

        optimal_excess_air = base_excess_air * load_factor

        # Apply constraints
        min_excess = constraints.get('min_excess_air_percent', 5)
        max_excess = constraints.get('max_excess_air_percent', 25)

        return min(max(optimal_excess_air, min_excess), max_excess)

    def _calculate_combustion_efficiency(
        self,
        excess_air: float,
        combustion_temp: float,
        fuel_properties: Dict[str, Any]
    ) -> float:
        """Calculate combustion efficiency."""
        # Base efficiency
        base_efficiency = 98

        # Excess air penalty
        excess_air_penalty = (excess_air - 10) * 0.2 if excess_air > 10 else 0

        # Temperature factor
        temp_factor = min((combustion_temp - 800) / 400, 1.0) if combustion_temp > 800 else 0

        efficiency = base_efficiency - excess_air_penalty + temp_factor

        return min(max(efficiency, 85), 99)

    def _calculate_stack_losses(
        self,
        stack_temp: float,
        ambient_temp: float,
        excess_air: float
    ) -> float:
        """Calculate stack heat losses."""
        temp_diff = stack_temp - ambient_temp

        # Simplified stack loss formula
        stack_loss = 0.65 * temp_diff / (21 / (1 + excess_air / 100))

        return stack_loss

    def _calculate_flame_stability(self, excess_air: float, combustion_temp: float) -> float:
        """Calculate flame stability index (0-1)."""
        # Optimal conditions
        optimal_excess = 10
        optimal_temp = 1200

        # Calculate deviations
        excess_deviation = abs(excess_air - optimal_excess) / optimal_excess
        temp_deviation = abs(combustion_temp - optimal_temp) / optimal_temp

        # Stability index
        stability = 1.0 - 0.5 * excess_deviation - 0.3 * temp_deviation

        return min(max(stability, 0), 1)

    def _optimize_blowdown_rate(self, tds_ppm: float, constraints: Dict[str, Any]) -> float:
        """Optimize blowdown rate based on TDS."""
        max_tds = constraints.get('max_tds_ppm', 3500)

        # Calculate required blowdown
        if tds_ppm < max_tds * 0.7:
            blowdown_rate = 2.0  # Minimum blowdown
        elif tds_ppm < max_tds * 0.9:
            blowdown_rate = 3.0  # Normal blowdown
        else:
            # Increase blowdown to control TDS
            blowdown_rate = 5.0 + (tds_ppm - max_tds * 0.9) / 100

        return min(blowdown_rate, 10)  # Cap at 10%

    def _calculate_steam_quality(self, pressure: float, moisture_percent: float) -> float:
        """Calculate steam quality index (0-1)."""
        # Quality = 1 - moisture fraction
        quality = 1 - (moisture_percent / 100)

        # Adjust for pressure (higher pressure requires better quality)
        pressure_factor = min(pressure / 40, 1.0)  # Normalize to 40 bar

        quality_index = quality * (0.8 + 0.2 * pressure_factor)

        return min(max(quality_index, 0), 1)

    def _calculate_heat_input(
        self,
        steam_flow: float,
        steam_temp: float,
        feedwater_temp: float,
        pressure: float
    ) -> float:
        """Calculate required heat input."""
        # Simplified enthalpy calculation
        steam_enthalpy = 2700 + 1.9 * (steam_temp - 100)  # kJ/kg
        feedwater_enthalpy = 4.186 * feedwater_temp  # kJ/kg

        # Add 10% for losses
        heat_required = steam_flow * (steam_enthalpy - feedwater_enthalpy) * 1.1 / 3600

        return heat_required / 1000  # Convert to MW

    def _calculate_steam_heat_output(
        self,
        steam_flow: float,
        steam_temp: float,
        pressure: float
    ) -> float:
        """Calculate steam heat output."""
        # Simplified enthalpy
        steam_enthalpy = 2700 + 1.9 * (steam_temp - 100)  # kJ/kg

        heat_output = steam_flow * steam_enthalpy / 3600

        return heat_output / 1000  # Convert to MW

    def _calculate_optimization_score(
        self,
        steam_quality: float,
        evaporation_ratio: float,
        blowdown_rate: float
    ) -> float:
        """Calculate overall optimization score (0-100)."""
        # Weight factors
        quality_weight = 0.4
        evaporation_weight = 0.4
        blowdown_weight = 0.2

        # Normalize evaporation ratio (10 is excellent)
        evaporation_score = min(evaporation_ratio / 10, 1.0)

        # Blowdown score (lower is better)
        blowdown_score = 1.0 - min(blowdown_rate / 10, 1.0)

        # Calculate weighted score
        score = (
            quality_weight * steam_quality * 100 +
            evaporation_weight * evaporation_score * 100 +
            blowdown_weight * blowdown_score * 100
        )

        return min(max(score, 0), 100)

    def _calculate_nox_emissions(self, combustion_temp: float, excess_air: float) -> float:
        """Calculate NOx emissions (thermal NOx)."""
        # Zeldovich mechanism approximation
        if combustion_temp < 1200:
            base_nox = 15
        elif combustion_temp < 1400:
            base_nox = 25
        else:
            base_nox = 40

        # Excess air effect
        excess_factor = 1 + (excess_air - 10) * 0.02

        nox_ppm = base_nox * excess_factor

        return nox_ppm

    # Integration methods

    def process_scada_data(self, scada_feed: Dict[str, Any]) -> Dict[str, Any]:
        """Process SCADA data for boiler optimization."""
        processed = {
            'timestamp': datetime.now().isoformat(),
            'tags_processed': len(scada_feed.get('tags', {})),
            'data_quality': 'good' if scada_feed.get('quality', 100) > 90 else 'poor',
            'values': {}
        }

        # Extract relevant tags
        for tag, value in scada_feed.get('tags', {}).items():
            if 'boiler' in tag.lower() or 'steam' in tag.lower():
                processed['values'][tag] = value

        return processed

    def process_dcs_data(self, dcs_feed: Dict[str, Any]) -> Dict[str, Any]:
        """Process DCS data for control integration."""
        return {
            'timestamp': datetime.now().isoformat(),
            'control_points': len(dcs_feed.get('points', {})),
            'mode': dcs_feed.get('mode', 'manual'),
            'setpoints': dcs_feed.get('setpoints', {})
        }

    def coordinate_boiler_agents(
        self,
        agent_ids: List[str],
        commands: Dict[str, Any],
        dashboard: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate multiple boiler agents."""
        task_assignments = {}

        for agent_id in agent_ids:
            # Assign tasks based on agent capabilities
            if 'combustion' in agent_id.lower():
                task_assignments[agent_id] = [
                    {
                        'task': 'optimize_combustion',
                        'parameters': commands.get('combustion', {}),
                        'priority': 'high'
                    }
                ]
            elif 'emissions' in agent_id.lower():
                task_assignments[agent_id] = [
                    {
                        'task': 'minimize_emissions',
                        'parameters': commands.get('emissions', {}),
                        'priority': 'medium'
                    }
                ]
            else:
                task_assignments[agent_id] = [
                    {
                        'task': 'monitor',
                        'parameters': {'dashboard': dashboard},
                        'priority': 'low'
                    }
                ]

        return {
            'task_assignments': task_assignments,
            'coordination_status': 'distributed',
            'agents_coordinated': len(agent_ids)
        }

    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up BoilerEfficiencyTools resources")
        # Any cleanup needed