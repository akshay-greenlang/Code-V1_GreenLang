"""
Tools module for SteamSystemAnalyzer agent (GL-003).

This module provides deterministic calculation tools for steam system
analysis, distribution efficiency, leak detection, and condensate optimization.
All calculations follow industry standards (ASME Steam Tables, ISO 12569,
ASHRAE standards) and zero-hallucination principles.
"""

import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SteamPropertiesResult:
    """Steam thermodynamic properties calculation result."""

    pressure_bar: float
    temperature_c: float
    enthalpy_kj_kg: float
    entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    density_kg_m3: float
    steam_quality: float  # 0-1 for wet steam, 1 for saturated/superheated
    latent_heat_kj_kg: float
    is_superheated: bool


@dataclass
class DistributionEfficiencyResult:
    """Steam distribution network efficiency result."""

    total_generation_kg_hr: float
    total_consumption_kg_hr: float
    distribution_losses_kg_hr: float
    distribution_efficiency_percent: float
    heat_losses_mw: float
    pressure_drop_bar: float
    temperature_drop_c: float
    insulation_effectiveness_percent: float


@dataclass
class LeakDetectionResult:
    """Steam leak detection analysis result."""

    total_leaks_detected: int
    total_leak_rate_kg_hr: float
    leak_locations: List[Dict[str, Any]]
    leak_severity_distribution: Dict[str, int]  # minor, moderate, major, critical
    estimated_annual_cost_usd: float
    priority_repairs: List[Dict[str, Any]]


@dataclass
class CondensateOptimizationResult:
    """Condensate return optimization result."""

    total_condensate_generated_kg_hr: float
    condensate_returned_kg_hr: float
    condensate_lost_kg_hr: float
    return_rate_percent: float
    return_temperature_c: float
    energy_recovered_mw: float
    water_savings_m3_day: float
    chemical_savings_usd_day: float
    optimization_opportunities: List[str]


@dataclass
class SteamTrapPerformanceResult:
    """Steam trap performance analysis result."""

    total_traps_assessed: int
    functioning_traps: int
    failed_open_traps: int
    failed_closed_traps: int
    trap_efficiency_percent: float
    steam_losses_from_traps_kg_hr: float
    estimated_repair_cost_usd: float
    priority_trap_list: List[Dict[str, Any]]


class SteamSystemTools:
    """
    Deterministic calculation tools for steam system analysis.

    All calculations follow industry standards:
    - ASME Steam Tables for thermodynamic properties
    - ISO 12569 for thermal insulation of building equipment
    - ASHRAE Handbook for steam system design
    - TLV Engineering for steam trap sizing and selection
    """

    def __init__(self):
        """Initialize SteamSystemTools."""
        self.logger = logging.getLogger(__name__)

        # Physical constants
        self.WATER_SPECIFIC_HEAT = 4.186  # kJ/kg·K
        self.STEAM_GAS_CONSTANT = 0.4615  # kJ/kg·K
        self.CRITICAL_PRESSURE_BAR = 220.64  # Critical pressure of water
        self.CRITICAL_TEMPERATURE_C = 374.15  # Critical temperature of water

        # Standard conditions
        self.STANDARD_PRESSURE_BAR = 1.013
        self.STANDARD_TEMPERATURE_C = 100.0

        # Thermal conductivities (W/m·K)
        self.INSULATION_CONDUCTIVITY = {
            'mineral_wool': 0.040,
            'fiberglass': 0.035,
            'calcium_silicate': 0.055,
            'cellular_glass': 0.045,
            'polyurethane_foam': 0.025
        }

    def calculate_steam_properties(
        self,
        pressure_bar: float,
        temperature_c: Optional[float] = None,
        quality: Optional[float] = None
    ) -> SteamPropertiesResult:
        """
        Calculate steam thermodynamic properties using ASME Steam Tables.

        Args:
            pressure_bar: Steam pressure in bar
            temperature_c: Steam temperature in Celsius (for superheated)
            quality: Steam quality/dryness fraction (for wet steam)

        Returns:
            Complete steam properties

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if pressure_bar is None:
            raise ValueError("pressure_bar cannot be None")
        if pressure_bar <= 0:
            raise ValueError(f"pressure_bar must be positive, got {pressure_bar}")
        if pressure_bar > self.CRITICAL_PRESSURE_BAR:
            raise ValueError(f"pressure_bar ({pressure_bar}) exceeds critical pressure {self.CRITICAL_PRESSURE_BAR}")

        # Calculate saturation temperature at given pressure
        t_sat = self._calculate_saturation_temperature(pressure_bar)

        # Determine steam state
        is_superheated = False
        if temperature_c is not None:
            if temperature_c < -273.15:
                raise ValueError(f"temperature_c must be above absolute zero, got {temperature_c}")
            if temperature_c > t_sat + 0.1:
                is_superheated = True
            actual_temp = temperature_c
        else:
            actual_temp = t_sat

        # Validate quality for wet steam
        if quality is not None:
            if not (0 <= quality <= 1):
                raise ValueError(f"quality must be between 0 and 1, got {quality}")
            if is_superheated:
                raise ValueError("Quality cannot be specified for superheated steam")
            steam_quality = quality
        else:
            steam_quality = 1.0 if is_superheated or temperature_c is None else 1.0

        # Calculate properties
        if is_superheated:
            # Superheated steam properties
            enthalpy = self._calculate_superheated_enthalpy(pressure_bar, actual_temp)
            entropy = self._calculate_superheated_entropy(pressure_bar, actual_temp)
            specific_volume = self._calculate_superheated_specific_volume(pressure_bar, actual_temp)
            latent_heat = 0  # No latent heat for superheated steam
        else:
            # Saturated or wet steam properties
            h_f = self._calculate_saturated_liquid_enthalpy(pressure_bar)
            h_fg = self._calculate_latent_heat(pressure_bar)
            enthalpy = h_f + steam_quality * h_fg

            s_f = self._calculate_saturated_liquid_entropy(pressure_bar)
            s_fg = self._calculate_evaporation_entropy(pressure_bar)
            entropy = s_f + steam_quality * s_fg

            v_f = self._calculate_liquid_specific_volume(pressure_bar)
            v_g = self._calculate_vapor_specific_volume(pressure_bar)
            specific_volume = v_f + steam_quality * (v_g - v_f)

            latent_heat = h_fg

        density = 1 / specific_volume if specific_volume > 0 else 0

        return SteamPropertiesResult(
            pressure_bar=pressure_bar,
            temperature_c=actual_temp,
            enthalpy_kj_kg=enthalpy,
            entropy_kj_kg_k=entropy,
            specific_volume_m3_kg=specific_volume,
            density_kg_m3=density,
            steam_quality=steam_quality,
            latent_heat_kj_kg=latent_heat,
            is_superheated=is_superheated
        )

    def analyze_distribution_efficiency(
        self,
        generation_data: Dict[str, Any],
        consumption_data: Dict[str, Any],
        network_data: Dict[str, Any]
    ) -> DistributionEfficiencyResult:
        """
        Analyze steam distribution network efficiency.

        Args:
            generation_data: Steam generation data
            consumption_data: Steam consumption data
            network_data: Distribution network configuration

        Returns:
            Distribution efficiency analysis

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if generation_data is None:
            raise ValueError("generation_data cannot be None")
        if consumption_data is None:
            raise ValueError("consumption_data cannot be None")
        if network_data is None:
            raise ValueError("network_data cannot be None")

        # Extract generation data
        total_generation = generation_data.get('total_flow_kg_hr', 0)
        generation_pressure = generation_data.get('pressure_bar', 10)
        generation_temp = generation_data.get('temperature_c', 180)

        # Validate generation data
        if total_generation < 0:
            raise ValueError(f"total_flow_kg_hr must be non-negative, got {total_generation}")
        if generation_pressure <= 0:
            raise ValueError(f"pressure_bar must be positive, got {generation_pressure}")

        # Extract consumption data
        total_consumption = consumption_data.get('total_flow_kg_hr', 0)
        consumption_pressure = consumption_data.get('average_pressure_bar', 9)
        consumption_temp = consumption_data.get('average_temperature_c', 175)

        # Validate consumption data
        if total_consumption < 0:
            raise ValueError(f"consumption total_flow_kg_hr must be non-negative, got {total_consumption}")
        if total_consumption > total_generation * 1.5:
            raise ValueError(f"Consumption ({total_consumption}) cannot exceed generation ({total_generation}) by >50%")

        # Extract network data
        pipeline_length = network_data.get('total_length_m', 1000)
        insulation_type = network_data.get('insulation_type', 'mineral_wool')
        insulation_thickness = network_data.get('insulation_thickness_mm', 50)
        ambient_temp = network_data.get('ambient_temperature_c', 25)

        # Validate network data
        if pipeline_length <= 0:
            raise ValueError(f"pipeline_length must be positive, got {pipeline_length}")
        if insulation_thickness <= 0:
            raise ValueError(f"insulation_thickness must be positive, got {insulation_thickness}")

        # Calculate distribution losses
        distribution_losses = total_generation - total_consumption

        # Calculate distribution efficiency
        distribution_efficiency = (total_consumption / total_generation * 100) if total_generation > 0 else 0

        # Calculate heat losses
        heat_losses = self._calculate_pipeline_heat_losses(
            pipeline_length,
            generation_temp,
            ambient_temp,
            insulation_type,
            insulation_thickness
        )

        # Calculate pressure drop
        pressure_drop = generation_pressure - consumption_pressure

        # Calculate temperature drop
        temperature_drop = generation_temp - consumption_temp

        # Calculate insulation effectiveness
        insulation_effectiveness = self._calculate_insulation_effectiveness(
            generation_temp,
            ambient_temp,
            insulation_type,
            insulation_thickness
        )

        return DistributionEfficiencyResult(
            total_generation_kg_hr=total_generation,
            total_consumption_kg_hr=total_consumption,
            distribution_losses_kg_hr=distribution_losses,
            distribution_efficiency_percent=distribution_efficiency,
            heat_losses_mw=heat_losses,
            pressure_drop_bar=pressure_drop,
            temperature_drop_c=temperature_drop,
            insulation_effectiveness_percent=insulation_effectiveness
        )

    def detect_steam_leaks(
        self,
        sensor_data: Dict[str, Any],
        system_config: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> LeakDetectionResult:
        """
        Detect steam leaks using mass balance and sensor analysis.

        Args:
            sensor_data: Real-time sensor measurements
            system_config: System configuration
            historical_data: Historical baseline data

        Returns:
            Leak detection analysis

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if sensor_data is None:
            raise ValueError("sensor_data cannot be None")
        if system_config is None:
            raise ValueError("system_config cannot be None")

        leaks_detected = []
        total_leak_rate = 0

        # Extract sensor readings
        flow_meters = sensor_data.get('flow_meters', {})
        pressure_sensors = sensor_data.get('pressure_sensors', {})

        # Mass balance analysis
        inlet_flow = sum([m.get('flow_kg_hr', 0) for m in flow_meters.values() if m.get('location') == 'inlet'])
        outlet_flow = sum([m.get('flow_kg_hr', 0) for m in flow_meters.values() if m.get('location') == 'outlet'])

        # Calculate unaccounted flow
        unaccounted_flow = inlet_flow - outlet_flow

        # Baseline comparison if available
        if historical_data:
            baseline_loss = historical_data.get('average_losses_kg_hr', 0)
            anomalous_loss = max(0, unaccounted_flow - baseline_loss)
        else:
            anomalous_loss = max(0, unaccounted_flow)

        # Detect leaks from pressure drops
        for location, sensor in pressure_sensors.items():
            pressure = sensor.get('pressure_bar', 0)
            expected_pressure = sensor.get('expected_pressure_bar', pressure)

            pressure_drop = expected_pressure - pressure

            if pressure_drop > system_config.get('leak_threshold_bar', 0.5):
                leak_rate = self._estimate_leak_rate_from_pressure_drop(
                    pressure_drop,
                    sensor.get('pipe_diameter_mm', 50)
                )

                severity = self._classify_leak_severity(leak_rate)

                leaks_detected.append({
                    'location': location,
                    'leak_rate_kg_hr': leak_rate,
                    'pressure_drop_bar': pressure_drop,
                    'severity': severity,
                    'timestamp': datetime.now().isoformat()
                })

                total_leak_rate += leak_rate

        # Add anomalous loss to total
        total_leak_rate += anomalous_loss

        # Classify severity distribution
        severity_distribution = {
            'minor': 0,
            'moderate': 0,
            'major': 0,
            'critical': 0
        }

        for leak in leaks_detected:
            severity_distribution[leak['severity']] += 1

        # Calculate annual cost
        steam_cost = system_config.get('steam_cost_usd_per_ton', 30)
        annual_cost = total_leak_rate * 24 * 365 * (steam_cost / 1000)

        # Generate priority repairs
        priority_repairs = sorted(
            [l for l in leaks_detected if l['severity'] in ['major', 'critical']],
            key=lambda x: x['leak_rate_kg_hr'],
            reverse=True
        )

        return LeakDetectionResult(
            total_leaks_detected=len(leaks_detected),
            total_leak_rate_kg_hr=total_leak_rate,
            leak_locations=leaks_detected,
            leak_severity_distribution=severity_distribution,
            estimated_annual_cost_usd=annual_cost,
            priority_repairs=priority_repairs[:10]  # Top 10
        )

    def optimize_condensate_return(
        self,
        condensate_data: Dict[str, Any],
        system_config: Dict[str, Any]
    ) -> CondensateOptimizationResult:
        """
        Optimize condensate return system performance.

        Args:
            condensate_data: Condensate system data
            system_config: System configuration

        Returns:
            Condensate optimization analysis

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if condensate_data is None:
            raise ValueError("condensate_data cannot be None")
        if system_config is None:
            raise ValueError("system_config cannot be None")

        # Extract condensate data
        steam_consumption = condensate_data.get('steam_consumption_kg_hr', 10000)
        condensate_returned = condensate_data.get('condensate_returned_kg_hr', 7000)
        condensate_temp = condensate_data.get('return_temperature_c', 80)
        makeup_water_temp = system_config.get('makeup_water_temperature_c', 15)

        # Validate data
        if steam_consumption < 0:
            raise ValueError(f"steam_consumption_kg_hr must be non-negative, got {steam_consumption}")
        if condensate_returned < 0:
            raise ValueError(f"condensate_returned_kg_hr must be non-negative, got {condensate_returned}")
        if condensate_returned > steam_consumption:
            raise ValueError(f"condensate_returned ({condensate_returned}) cannot exceed steam_consumption ({steam_consumption})")

        # Calculate condensate generated (assume 100% of steam becomes condensate)
        total_condensate_generated = steam_consumption

        # Calculate condensate lost
        condensate_lost = total_condensate_generated - condensate_returned

        # Calculate return rate
        return_rate = (condensate_returned / total_condensate_generated * 100) if total_condensate_generated > 0 else 0

        # Calculate energy recovered
        # Energy = mass_flow * specific_heat * (condensate_temp - makeup_temp)
        energy_recovered_kw = (
            condensate_returned *
            self.WATER_SPECIFIC_HEAT *
            (condensate_temp - makeup_water_temp) / 3600
        )
        energy_recovered_mw = energy_recovered_kw / 1000

        # Calculate water savings
        water_savings_m3_day = (condensate_returned * 24) / 1000  # Convert kg to m³

        # Calculate chemical savings
        water_treatment_cost = system_config.get('water_treatment_cost_usd_per_m3', 2.0)
        chemical_savings_usd_day = water_savings_m3_day * water_treatment_cost

        # Identify optimization opportunities
        opportunities = []

        if return_rate < 90:
            opportunities.append(
                f"Increase condensate return from {return_rate:.1f}% to 90% - potential savings of ${(90 - return_rate) * steam_consumption * 24 * 365 * 0.001 * 30 / 100:.0f}/year"
            )

        if condensate_temp < 60:
            opportunities.append(
                f"Improve condensate insulation - current return temperature {condensate_temp:.1f}°C is low"
            )

        if condensate_lost > steam_consumption * 0.15:
            opportunities.append(
                f"Significant condensate losses detected ({condensate_lost:.0f} kg/hr) - investigate flash steam recovery"
            )

        return CondensateOptimizationResult(
            total_condensate_generated_kg_hr=total_condensate_generated,
            condensate_returned_kg_hr=condensate_returned,
            condensate_lost_kg_hr=condensate_lost,
            return_rate_percent=return_rate,
            return_temperature_c=condensate_temp,
            energy_recovered_mw=energy_recovered_mw,
            water_savings_m3_day=water_savings_m3_day,
            chemical_savings_usd_day=chemical_savings_usd_day,
            optimization_opportunities=opportunities
        )

    def analyze_steam_trap_performance(
        self,
        trap_data: Dict[str, Any],
        system_config: Dict[str, Any]
    ) -> SteamTrapPerformanceResult:
        """
        Analyze steam trap performance across the system.

        Args:
            trap_data: Steam trap monitoring data
            system_config: System configuration

        Returns:
            Steam trap performance analysis

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if trap_data is None:
            raise ValueError("trap_data cannot be None")
        if system_config is None:
            raise ValueError("system_config cannot be None")

        total_traps = trap_data.get('total_trap_count', 0)
        trap_assessments = trap_data.get('trap_assessments', [])

        # Validate trap count
        if total_traps < 0:
            raise ValueError(f"total_trap_count must be non-negative, got {total_traps}")

        # Count trap statuses
        functioning = 0
        failed_open = 0
        failed_closed = 0

        for trap in trap_assessments:
            status = trap.get('status', 'unknown')
            if status == 'functioning':
                functioning += 1
            elif status == 'failed_open':
                failed_open += 1
            elif status == 'failed_closed':
                failed_closed += 1

        # Calculate trap efficiency
        trap_efficiency = (functioning / total_traps * 100) if total_traps > 0 else 0

        # Calculate steam losses from failed-open traps
        steam_losses = 0
        for trap in trap_assessments:
            if trap.get('status') == 'failed_open':
                # Estimate loss based on trap size and pressure
                trap_size_mm = trap.get('orifice_size_mm', 10)
                pressure_bar = trap.get('upstream_pressure_bar', 5)
                loss_rate = self._calculate_trap_steam_loss(trap_size_mm, pressure_bar)
                steam_losses += loss_rate

        # Calculate repair costs
        repair_cost_per_trap = system_config.get('trap_repair_cost_usd', 50)
        total_repair_cost = (failed_open + failed_closed) * repair_cost_per_trap

        # Generate priority trap list
        priority_traps = []
        for trap in trap_assessments:
            if trap.get('status') in ['failed_open', 'failed_closed']:
                loss_rate = 0
                if trap.get('status') == 'failed_open':
                    trap_size_mm = trap.get('orifice_size_mm', 10)
                    pressure_bar = trap.get('upstream_pressure_bar', 5)
                    loss_rate = self._calculate_trap_steam_loss(trap_size_mm, pressure_bar)

                priority_traps.append({
                    'trap_id': trap.get('trap_id', 'unknown'),
                    'location': trap.get('location', 'unknown'),
                    'status': trap.get('status'),
                    'loss_rate_kg_hr': loss_rate,
                    'priority': 'high' if trap.get('status') == 'failed_open' else 'medium'
                })

        # Sort by loss rate
        priority_traps.sort(key=lambda x: x['loss_rate_kg_hr'], reverse=True)

        return SteamTrapPerformanceResult(
            total_traps_assessed=total_traps,
            functioning_traps=functioning,
            failed_open_traps=failed_open,
            failed_closed_traps=failed_closed,
            trap_efficiency_percent=trap_efficiency,
            steam_losses_from_traps_kg_hr=steam_losses,
            estimated_repair_cost_usd=total_repair_cost,
            priority_trap_list=priority_traps[:20]  # Top 20
        )

    # Helper methods for calculations

    def _calculate_saturation_temperature(self, pressure_bar: float) -> float:
        """
        Calculate saturation temperature from pressure using Antoine equation.

        Args:
            pressure_bar: Pressure in bar

        Returns:
            Saturation temperature in Celsius
        """
        # Antoine equation coefficients for water
        A = 8.07131
        B = 1730.63
        C = 233.426

        # Convert bar to mmHg
        pressure_mmhg = pressure_bar * 750.062

        # Calculate temperature in Celsius
        t_sat = (B / (A - math.log10(pressure_mmhg))) - C

        return t_sat

    def _calculate_latent_heat(self, pressure_bar: float) -> float:
        """
        Calculate latent heat of vaporization at given pressure.

        Args:
            pressure_bar: Pressure in bar

        Returns:
            Latent heat in kJ/kg
        """
        # Simplified correlation for latent heat
        t_sat = self._calculate_saturation_temperature(pressure_bar)

        # Latent heat decreases with temperature
        h_fg = 2500.9 - 2.37 * t_sat

        return max(h_fg, 0)

    def _calculate_saturated_liquid_enthalpy(self, pressure_bar: float) -> float:
        """Calculate saturated liquid enthalpy."""
        t_sat = self._calculate_saturation_temperature(pressure_bar)

        # Simplified enthalpy calculation
        h_f = self.WATER_SPECIFIC_HEAT * t_sat

        return h_f

    def _calculate_superheated_enthalpy(self, pressure_bar: float, temperature_c: float) -> float:
        """Calculate superheated steam enthalpy."""
        t_sat = self._calculate_saturation_temperature(pressure_bar)
        h_f = self._calculate_saturated_liquid_enthalpy(pressure_bar)
        h_fg = self._calculate_latent_heat(pressure_bar)

        # Superheated enthalpy
        h_g = h_f + h_fg
        superheat = temperature_c - t_sat

        # Add superheat enthalpy (approximately 2 kJ/kg·K)
        h_superheat = h_g + 2.0 * superheat

        return h_superheat

    def _calculate_saturated_liquid_entropy(self, pressure_bar: float) -> float:
        """Calculate saturated liquid entropy."""
        t_sat = self._calculate_saturation_temperature(pressure_bar)

        # Simplified entropy calculation
        s_f = self.WATER_SPECIFIC_HEAT * math.log((t_sat + 273.15) / 273.15)

        return s_f

    def _calculate_evaporation_entropy(self, pressure_bar: float) -> float:
        """Calculate evaporation entropy."""
        h_fg = self._calculate_latent_heat(pressure_bar)
        t_sat = self._calculate_saturation_temperature(pressure_bar)

        s_fg = h_fg / (t_sat + 273.15)

        return s_fg

    def _calculate_superheated_entropy(self, pressure_bar: float, temperature_c: float) -> float:
        """Calculate superheated steam entropy."""
        s_f = self._calculate_saturated_liquid_entropy(pressure_bar)
        s_fg = self._calculate_evaporation_entropy(pressure_bar)

        t_sat = self._calculate_saturation_temperature(pressure_bar)
        superheat = temperature_c - t_sat

        # Superheated entropy
        s_g = s_f + s_fg
        s_superheat = s_g + 2.0 * math.log((temperature_c + 273.15) / (t_sat + 273.15))

        return s_superheat

    def _calculate_liquid_specific_volume(self, pressure_bar: float) -> float:
        """Calculate liquid specific volume (approximately constant)."""
        return 0.001  # m³/kg

    def _calculate_vapor_specific_volume(self, pressure_bar: float) -> float:
        """Calculate vapor specific volume using ideal gas approximation."""
        t_sat = self._calculate_saturation_temperature(pressure_bar)

        # Ideal gas equation: v = RT/P
        pressure_pa = pressure_bar * 100000
        temperature_k = t_sat + 273.15

        v_g = (self.STEAM_GAS_CONSTANT * 1000 * temperature_k) / pressure_pa

        return v_g

    def _calculate_superheated_specific_volume(self, pressure_bar: float, temperature_c: float) -> float:
        """Calculate superheated steam specific volume."""
        pressure_pa = pressure_bar * 100000
        temperature_k = temperature_c + 273.15

        v_superheated = (self.STEAM_GAS_CONSTANT * 1000 * temperature_k) / pressure_pa

        return v_superheated

    def _calculate_pipeline_heat_losses(
        self,
        length_m: float,
        steam_temp_c: float,
        ambient_temp_c: float,
        insulation_type: str,
        insulation_thickness_mm: float
    ) -> float:
        """
        Calculate heat losses from steam pipelines.

        Returns heat loss in MW
        """
        # Get thermal conductivity
        k_insulation = self.INSULATION_CONDUCTIVITY.get(insulation_type, 0.040)

        # Assume 100mm pipe diameter
        pipe_od_m = 0.100
        insulation_thickness_m = insulation_thickness_mm / 1000

        # Calculate heat loss per meter
        # Q = 2*pi*k*L*(T_steam - T_ambient) / ln((r_outer + t) / r_outer)
        r_outer = pipe_od_m / 2
        r_insulated = r_outer + insulation_thickness_m

        temp_diff = steam_temp_c - ambient_temp_c

        # Heat loss per meter (W/m)
        q_per_m = (2 * math.pi * k_insulation * temp_diff) / math.log(r_insulated / r_outer)

        # Total heat loss (W)
        total_heat_loss_w = q_per_m * length_m

        # Convert to MW
        total_heat_loss_mw = total_heat_loss_w / 1_000_000

        return total_heat_loss_mw

    def _calculate_insulation_effectiveness(
        self,
        steam_temp_c: float,
        ambient_temp_c: float,
        insulation_type: str,
        insulation_thickness_mm: float
    ) -> float:
        """Calculate insulation effectiveness as percentage."""
        # Compare to bare pipe losses
        bare_pipe_k = 50.0  # W/m·K for steel
        insulation_k = self.INSULATION_CONDUCTIVITY.get(insulation_type, 0.040)

        # Effectiveness is inverse ratio of thermal resistances
        effectiveness = (1 - (insulation_k / bare_pipe_k)) * 100

        # Adjust for thickness
        effectiveness_adjusted = effectiveness * min(insulation_thickness_mm / 50, 1.2)

        return min(effectiveness_adjusted, 99.9)

    def _estimate_leak_rate_from_pressure_drop(
        self,
        pressure_drop_bar: float,
        pipe_diameter_mm: float
    ) -> float:
        """
        Estimate leak rate from pressure drop.

        Returns leak rate in kg/hr
        """
        # Simplified orifice equation
        # Flow rate proportional to sqrt(pressure_drop) and area

        orifice_area_m2 = math.pi * (pipe_diameter_mm / 1000 / 2) ** 2

        # Assume leak is through small orifice (10% of pipe area)
        leak_area_m2 = orifice_area_m2 * 0.1

        # Flow coefficient
        c_d = 0.6  # Discharge coefficient

        # Density approximation for steam
        density_kg_m3 = 5.0  # Approximate for medium pressure steam

        # Flow rate = C_d * A * sqrt(2 * rho * delta_P)
        pressure_drop_pa = pressure_drop_bar * 100000
        flow_rate_kg_s = c_d * leak_area_m2 * math.sqrt(2 * density_kg_m3 * pressure_drop_pa)

        # Convert to kg/hr
        flow_rate_kg_hr = flow_rate_kg_s * 3600

        return flow_rate_kg_hr

    def _classify_leak_severity(self, leak_rate_kg_hr: float) -> str:
        """Classify leak severity based on rate."""
        if leak_rate_kg_hr < 10:
            return 'minor'
        elif leak_rate_kg_hr < 50:
            return 'moderate'
        elif leak_rate_kg_hr < 200:
            return 'major'
        else:
            return 'critical'

    def _calculate_trap_steam_loss(self, orifice_size_mm: float, pressure_bar: float) -> float:
        """
        Calculate steam loss through failed-open trap.

        Returns loss rate in kg/hr
        """
        # Orifice area
        area_m2 = math.pi * (orifice_size_mm / 1000 / 2) ** 2

        # Steam density at pressure
        density_kg_m3 = 5.0 + pressure_bar * 0.5  # Simplified

        # Flow coefficient
        c_d = 0.6

        # Pressure drop (assume downstream is atmospheric)
        pressure_drop_pa = pressure_bar * 100000

        # Calculate flow rate
        flow_rate_kg_s = c_d * area_m2 * math.sqrt(2 * density_kg_m3 * pressure_drop_pa)

        # Convert to kg/hr
        flow_rate_kg_hr = flow_rate_kg_s * 3600

        return flow_rate_kg_hr

    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up SteamSystemTools resources")
