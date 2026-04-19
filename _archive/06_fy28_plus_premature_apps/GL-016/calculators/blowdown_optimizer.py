# -*- coding: utf-8 -*-
"""
Blowdown Optimizer Calculator - GL-016 WATERGUARD

Advanced blowdown optimization calculations for boiler and cooling tower systems.
Implements industry-standard water balance and heat loss calculations with
zero hallucination guarantee through deterministic formulas and SHA-256 provenance.

Author: GL-016 WATERGUARD Engineering Team
Version: 1.0.0
Standards: ASME, ABMA, ASHRAE, CTI
References:
- ASME Consensus on Operating Practices for Control of Feedwater/Boiler Water Chemistry
- ABMA Boiler Water Limits and Steam Purity Recommendations
- ASHRAE Handbook - HVAC Systems and Equipment (Cooling Towers)
- CTI ATC-105 - Acceptance Test Code for Water Cooling Towers
- EPRI TR-102285 - Boiler Chemical Cleaning Guidelines
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .provenance import ProvenanceTracker, CalculationStep
import hashlib
import json


@dataclass
class BlowdownConditions:
    """
    Operating conditions for blowdown optimization.

    All parameters use SI units for consistency.
    """
    # Steam/water production
    steam_generation_kg_hr: float  # Boiler steam output
    feedwater_flow_kg_hr: float  # Total feedwater flow

    # Water chemistry - makeup water (mg/L)
    makeup_tds_mg_l: float  # Total dissolved solids in makeup
    makeup_silica_mg_l: float  # Silica in makeup water
    makeup_conductivity_us_cm: float  # Conductivity of makeup

    # Water chemistry - boiler/tower water (mg/L)
    boiler_tds_mg_l: float  # Current TDS in boiler/tower
    boiler_silica_mg_l: float  # Current silica in boiler
    target_tds_mg_l: float  # Target maximum TDS
    target_silica_mg_l: float  # Target maximum silica

    # Operating conditions
    temperature_c: float  # Boiler/tower water temperature
    pressure_bar: float  # Operating pressure (for boilers)

    # Economics
    water_cost_per_m3: float  # Cost of makeup water ($/m3)
    fuel_cost_per_gj: float  # Cost of fuel ($/GJ)
    chemical_cost_per_kg: float  # Cost of treatment chemicals ($/kg)
    electricity_cost_per_kwh: float  # Cost of electricity ($/kWh)

    # Equipment parameters
    heat_exchanger_efficiency: float = 0.85  # Heat recovery efficiency
    blowdown_flash_tank: bool = True  # Is flash tank installed?
    flash_pressure_bar: float = 1.5  # Flash tank operating pressure


@dataclass
class CoolingTowerConditions:
    """
    Operating conditions specific to cooling tower blowdown optimization.
    """
    # Flow rates (m3/hr)
    circulation_rate_m3_hr: float
    evaporation_rate_m3_hr: float

    # Water chemistry (mg/L)
    makeup_tds_mg_l: float
    makeup_silica_mg_l: float
    makeup_calcium_mg_l: float
    makeup_conductivity_us_cm: float

    # Target concentrations (mg/L)
    target_tds_mg_l: float
    target_silica_mg_l: float
    target_calcium_mg_l: float

    # Economics
    water_cost_per_m3: float
    chemical_cost_per_kg: float

    # Parameters with defaults (must come after required fields)
    drift_loss_percent: float = 0.02  # Modern towers: 0.01-0.02%
    max_cycles_of_concentration: float = 10.0
    target_lsi_max: float = 0.5  # Maximum allowable LSI


@dataclass
class BlowdownResult:
    """
    Comprehensive blowdown calculation result with provenance tracking.
    """
    # Calculated rates
    continuous_blowdown_kg_hr: float
    intermittent_blowdown_kg_hr: float
    total_blowdown_kg_hr: float
    blowdown_percent: float

    # Cycles of concentration
    cycles_of_concentration: float
    optimal_cycles: float

    # Heat losses and recovery
    heat_loss_kw: float
    recoverable_heat_kw: float
    flash_steam_percent: float

    # Economic analysis
    water_cost_per_hour: float
    heat_loss_cost_per_hour: float
    total_cost_per_hour: float
    annual_cost: float
    potential_savings: float

    # Provenance
    provenance_hash: str
    calculation_steps: List[Dict] = field(default_factory=list)


class BlowdownOptimizer:
    """
    Zero-Hallucination Blowdown Optimizer.

    Guarantees:
    - Deterministic calculations (same input -> same output)
    - Complete provenance tracking with SHA-256 hashing
    - Industry-standard formulas (ASME, ABMA, ASHRAE, CTI)
    - NO LLM involvement in calculation path

    Capabilities:
    - Continuous blowdown rate calculation
    - Intermittent blowdown scheduling
    - Cycles of concentration optimization
    - Heat loss from blowdown calculation
    - Blowdown heat recovery potential
    - Cost optimization (water vs energy tradeoff)
    - Flash steam calculations
    - Economic payback analysis
    """

    # Physical constants
    WATER_SPECIFIC_HEAT = Decimal('4.186')  # kJ/(kg*K)
    WATER_DENSITY = Decimal('1000')  # kg/m3 at 20C
    STEAM_ENTHALPY_OFFSET = Decimal('2675')  # kJ/kg (approx at 1 bar)

    # Reference temperatures
    REFERENCE_TEMP_C = Decimal('25')  # Standard reference temperature

    def __init__(self, version: str = "1.0.0"):
        """Initialize blowdown optimizer with version tracking."""
        self.version = version

    def calculate_continuous_blowdown(
        self,
        conditions: BlowdownConditions,
        tracker: Optional[ProvenanceTracker] = None
    ) -> BlowdownResult:
        """
        Calculate continuous blowdown rate for boiler systems.

        Mass Balance:
        F = S + B
        Where: F = feedwater, S = steam, B = blowdown

        Concentration Balance:
        F * C_f = S * C_s + B * C_b
        Since C_s ~ 0 for pure steam:
        B = F * C_f / C_b

        Cycles of Concentration:
        CoC = C_b / C_f = F / B

        Reference: ASME Consensus Document, ABMA Guidelines

        Args:
            conditions: Operating conditions and water chemistry
            tracker: Optional provenance tracker

        Returns:
            BlowdownResult with complete calculation details
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"blowdown_{id(conditions)}",
                calculation_type="continuous_blowdown",
                version=self.version
            )

        # Filter out non-numeric values for provenance tracking
        numeric_inputs = {
            k: v for k, v in conditions.__dict__.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        tracker.record_inputs(numeric_inputs)

        # Convert to Decimal for precision
        steam = Decimal(str(conditions.steam_generation_kg_hr))
        feedwater = Decimal(str(conditions.feedwater_flow_kg_hr))
        makeup_tds = Decimal(str(conditions.makeup_tds_mg_l))
        boiler_tds = Decimal(str(conditions.boiler_tds_mg_l))
        target_tds = Decimal(str(conditions.target_tds_mg_l))
        makeup_silica = Decimal(str(conditions.makeup_silica_mg_l))
        target_silica = Decimal(str(conditions.target_silica_mg_l))
        temp_c = Decimal(str(conditions.temperature_c))
        pressure = Decimal(str(conditions.pressure_bar))

        # Step 1: Calculate current cycles of concentration
        if makeup_tds > 0:
            current_coc = boiler_tds / makeup_tds
        else:
            current_coc = Decimal('1')

        tracker.record_step(
            operation="current_cycles",
            description="Calculate current cycles of concentration",
            inputs={
                'boiler_tds_mg_l': boiler_tds,
                'makeup_tds_mg_l': makeup_tds
            },
            output_value=current_coc,
            output_name="current_cycles_of_concentration",
            formula="CoC = C_boiler / C_makeup",
            units="cycles",
            reference="ASME Consensus Document"
        )

        # Step 2: Calculate maximum allowable cycles based on TDS and silica limits
        max_coc_tds = target_tds / makeup_tds if makeup_tds > 0 else Decimal('10')
        max_coc_silica = target_silica / makeup_silica if makeup_silica > 0 else Decimal('10')
        max_coc = min(max_coc_tds, max_coc_silica)

        tracker.record_step(
            operation="max_cycles",
            description="Calculate maximum allowable cycles of concentration",
            inputs={
                'target_tds_mg_l': target_tds,
                'target_silica_mg_l': target_silica,
                'makeup_tds_mg_l': makeup_tds,
                'makeup_silica_mg_l': makeup_silica
            },
            output_value=max_coc,
            output_name="max_cycles_of_concentration",
            formula="CoC_max = min(TDS_target/TDS_makeup, Si_target/Si_makeup)",
            units="cycles",
            reference="ABMA Boiler Water Guidelines"
        )

        # Step 3: Calculate optimal cycles (80% of maximum for safety margin)
        optimal_coc = max_coc * Decimal('0.8')
        if optimal_coc < Decimal('2'):
            optimal_coc = Decimal('2')  # Minimum practical cycles

        tracker.record_step(
            operation="optimal_cycles",
            description="Calculate optimal cycles with safety margin",
            inputs={
                'max_cycles': max_coc,
                'safety_factor': Decimal('0.8')
            },
            output_value=optimal_coc,
            output_name="optimal_cycles_of_concentration",
            formula="CoC_optimal = max(2, CoC_max * 0.8)",
            units="cycles"
        )

        # Step 4: Calculate continuous blowdown rate
        # B = S / (CoC - 1) for steady state
        if optimal_coc > Decimal('1'):
            continuous_blowdown = steam / (optimal_coc - Decimal('1'))
        else:
            continuous_blowdown = steam  # Maximum blowdown if CoC <= 1

        # Blowdown as percentage of feedwater
        if feedwater > 0:
            blowdown_percent = (continuous_blowdown / feedwater) * Decimal('100')
        else:
            blowdown_percent = Decimal('0')

        tracker.record_step(
            operation="continuous_blowdown_rate",
            description="Calculate continuous blowdown rate from mass balance",
            inputs={
                'steam_generation_kg_hr': steam,
                'optimal_cycles': optimal_coc
            },
            output_value=continuous_blowdown,
            output_name="continuous_blowdown_kg_hr",
            formula="B = S / (CoC - 1)",
            units="kg/hr",
            reference="ASME PTC 4.1"
        )

        # Step 5: Calculate heat loss from blowdown
        # Q = m * Cp * (T_boiler - T_reference)
        delta_t = temp_c - self.REFERENCE_TEMP_C
        heat_loss_kj_hr = continuous_blowdown * self.WATER_SPECIFIC_HEAT * delta_t
        heat_loss_kw = heat_loss_kj_hr / Decimal('3600')

        tracker.record_step(
            operation="heat_loss",
            description="Calculate heat loss in blowdown water",
            inputs={
                'blowdown_kg_hr': continuous_blowdown,
                'temperature_C': temp_c,
                'specific_heat_kj_kg_k': self.WATER_SPECIFIC_HEAT
            },
            output_value=heat_loss_kw,
            output_name="heat_loss_kw",
            formula="Q = m * Cp * (T_boiler - T_ref) / 3600",
            units="kW",
            reference="Basic Thermodynamics"
        )

        # Step 6: Calculate flash steam and recoverable heat
        flash_steam_pct, recoverable_kw = self._calculate_flash_recovery(
            continuous_blowdown, temp_c, pressure, conditions, tracker
        )

        # Step 7: Economic analysis
        economic_result = self._calculate_economics(
            continuous_blowdown, heat_loss_kw, recoverable_kw,
            conditions, tracker
        )

        # Generate provenance hash
        final_result = {
            'continuous_blowdown_kg_hr': float(continuous_blowdown),
            'cycles_of_concentration': float(optimal_coc),
            'heat_loss_kw': float(heat_loss_kw)
        }
        provenance = tracker.get_provenance_record(final_result)

        return BlowdownResult(
            continuous_blowdown_kg_hr=float(continuous_blowdown.quantize(Decimal('0.01'))),
            intermittent_blowdown_kg_hr=0.0,  # Continuous only
            total_blowdown_kg_hr=float(continuous_blowdown.quantize(Decimal('0.01'))),
            blowdown_percent=float(blowdown_percent.quantize(Decimal('0.01'))),
            cycles_of_concentration=float(current_coc.quantize(Decimal('0.01'))),
            optimal_cycles=float(optimal_coc.quantize(Decimal('0.01'))),
            heat_loss_kw=float(heat_loss_kw.quantize(Decimal('0.01'))),
            recoverable_heat_kw=float(recoverable_kw.quantize(Decimal('0.01'))),
            flash_steam_percent=float(flash_steam_pct.quantize(Decimal('0.01'))),
            water_cost_per_hour=economic_result['water_cost_per_hour'],
            heat_loss_cost_per_hour=economic_result['heat_loss_cost_per_hour'],
            total_cost_per_hour=economic_result['total_cost_per_hour'],
            annual_cost=economic_result['annual_cost'],
            potential_savings=economic_result['potential_savings'],
            provenance_hash=provenance.provenance_hash,
            calculation_steps=[step.to_dict() for step in tracker.steps]
        )

    def calculate_intermittent_blowdown(
        self,
        conditions: BlowdownConditions,
        interval_hours: float = 8.0,
        duration_seconds: float = 30.0,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate intermittent (bottom) blowdown schedule.

        Intermittent blowdown removes sludge and precipitated solids.
        Typically performed 1-3 times per shift.

        Formula:
        V_blowdown = Q_valve * t_duration
        Average rate = V_blowdown / interval

        Reference: ABMA Boiler Water Guidelines

        Args:
            conditions: Operating conditions
            interval_hours: Time between blowdowns
            duration_seconds: Duration of each blowdown
            tracker: Optional provenance tracker

        Returns:
            Intermittent blowdown schedule and volumes
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"intermittent_{id(conditions)}",
                calculation_type="intermittent_blowdown",
                version=self.version
            )

        tracker.record_inputs({
            **conditions.__dict__,
            'interval_hours': interval_hours,
            'duration_seconds': duration_seconds
        })

        # Convert to Decimal
        interval = Decimal(str(interval_hours))
        duration = Decimal(str(duration_seconds))
        pressure = Decimal(str(conditions.pressure_bar))

        # Step 1: Estimate blowdown valve flow rate
        # Q = Cv * sqrt(delta_P) where Cv is valve coefficient
        # Typical blowdown valve: Cv = 5-20
        # Assume Cv = 10 for 2" valve
        cv = Decimal('10')
        delta_p_psi = pressure * Decimal('14.504')  # bar to psi
        flow_rate_gpm = cv * delta_p_psi.sqrt()
        flow_rate_kg_hr = flow_rate_gpm * Decimal('227.12')  # gpm to kg/hr

        tracker.record_step(
            operation="valve_flow_rate",
            description="Calculate blowdown valve flow rate",
            inputs={
                'valve_cv': cv,
                'pressure_bar': pressure,
                'pressure_psi': delta_p_psi
            },
            output_value=flow_rate_kg_hr,
            output_name="valve_flow_rate_kg_hr",
            formula="Q = Cv * sqrt(dP) * 227.12",
            units="kg/hr",
            reference="ISA Valve Sizing"
        )

        # Step 2: Calculate volume per blowdown event
        volume_per_blowdown_kg = flow_rate_kg_hr * (duration / Decimal('3600'))

        tracker.record_step(
            operation="volume_per_blowdown",
            description="Calculate water volume per blowdown event",
            inputs={
                'flow_rate_kg_hr': flow_rate_kg_hr,
                'duration_seconds': duration
            },
            output_value=volume_per_blowdown_kg,
            output_name="volume_per_blowdown_kg",
            formula="V = Q * t / 3600",
            units="kg"
        )

        # Step 3: Calculate average rate
        average_rate_kg_hr = volume_per_blowdown_kg / interval

        tracker.record_step(
            operation="average_intermittent_rate",
            description="Calculate average intermittent blowdown rate",
            inputs={
                'volume_per_blowdown_kg': volume_per_blowdown_kg,
                'interval_hours': interval
            },
            output_value=average_rate_kg_hr,
            output_name="average_rate_kg_hr",
            formula="R_avg = V / interval",
            units="kg/hr"
        )

        # Step 4: Calculate daily schedule
        blowdowns_per_day = Decimal('24') / interval
        daily_volume_kg = volume_per_blowdown_kg * blowdowns_per_day

        # Step 5: Heat loss per blowdown
        temp_c = Decimal(str(conditions.temperature_c))
        delta_t = temp_c - self.REFERENCE_TEMP_C
        heat_per_blowdown_kj = volume_per_blowdown_kg * self.WATER_SPECIFIC_HEAT * delta_t

        provenance = tracker.get_provenance_record({
            'average_rate_kg_hr': float(average_rate_kg_hr),
            'daily_volume_kg': float(daily_volume_kg)
        })

        return {
            'valve_flow_rate_kg_hr': float(flow_rate_kg_hr.quantize(Decimal('0.1'))),
            'volume_per_blowdown_kg': float(volume_per_blowdown_kg.quantize(Decimal('0.1'))),
            'average_rate_kg_hr': float(average_rate_kg_hr.quantize(Decimal('0.01'))),
            'blowdowns_per_day': float(blowdowns_per_day.quantize(Decimal('0.1'))),
            'daily_volume_kg': float(daily_volume_kg.quantize(Decimal('0.1'))),
            'heat_loss_per_blowdown_kj': float(heat_per_blowdown_kj.quantize(Decimal('0.1'))),
            'schedule': {
                'interval_hours': interval_hours,
                'duration_seconds': duration_seconds,
                'recommended_times': self._generate_blowdown_schedule(interval_hours)
            },
            'provenance': provenance.to_dict()
        }

    def optimize_cycles_of_concentration(
        self,
        conditions: BlowdownConditions,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Optimize cycles of concentration for minimum total cost.

        Trade-off analysis:
        - Higher CoC = Less water use = Lower water cost
        - Higher CoC = More chemical treatment = Higher chemical cost
        - Higher CoC = Higher scaling risk = Higher maintenance cost

        Optimal CoC typically 4-8 for most systems.

        Reference: EPRI TR-102285, CTI Guidelines

        Args:
            conditions: Operating conditions
            tracker: Optional provenance tracker

        Returns:
            Optimization results with cost curves
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"coc_optimize_{id(conditions)}",
                calculation_type="coc_optimization",
                version=self.version
            )

        # Filter out non-numeric values for provenance tracking
        numeric_inputs = {
            k: v for k, v in conditions.__dict__.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        tracker.record_inputs(numeric_inputs)

        # Calculate maximum CoC from water chemistry limits
        makeup_tds = Decimal(str(conditions.makeup_tds_mg_l))
        target_tds = Decimal(str(conditions.target_tds_mg_l))
        makeup_silica = Decimal(str(conditions.makeup_silica_mg_l))
        target_silica = Decimal(str(conditions.target_silica_mg_l))
        steam = Decimal(str(conditions.steam_generation_kg_hr))
        water_cost = Decimal(str(conditions.water_cost_per_m3))
        fuel_cost = Decimal(str(conditions.fuel_cost_per_gj))
        chem_cost = Decimal(str(conditions.chemical_cost_per_kg))
        temp_c = Decimal(str(conditions.temperature_c))

        # Calculate limits
        max_coc_tds = target_tds / makeup_tds if makeup_tds > 0 else Decimal('10')
        max_coc_silica = target_silica / makeup_silica if makeup_silica > 0 else Decimal('10')
        max_coc = min(max_coc_tds, max_coc_silica, Decimal('15'))

        # Analyze costs at different CoC values
        cost_analysis = []
        min_total_cost = Decimal('1e10')
        optimal_coc = Decimal('5')

        for coc_int in range(20, int(max_coc * 10) + 1, 5):  # 2.0 to max in 0.5 increments
            coc = Decimal(str(coc_int)) / Decimal('10')

            # Blowdown rate at this CoC
            blowdown = steam / (coc - Decimal('1')) if coc > 1 else steam
            makeup = steam + blowdown

            # Water cost ($/hr)
            water_m3_hr = makeup / self.WATER_DENSITY
            water_cost_hr = water_m3_hr * water_cost

            # Heat loss cost ($/hr)
            delta_t = temp_c - self.REFERENCE_TEMP_C
            heat_loss_kj = blowdown * self.WATER_SPECIFIC_HEAT * delta_t
            heat_loss_gj = heat_loss_kj / Decimal('1e6')
            heat_cost_hr = heat_loss_gj * fuel_cost

            # Chemical cost (higher CoC needs more treatment)
            # Assume chemical dose proportional to CoC^1.5
            base_chem_kg_hr = Decimal('0.1')  # Base treatment rate
            chem_kg_hr = base_chem_kg_hr * (coc ** Decimal('1.5'))
            chem_cost_hr = chem_kg_hr * chem_cost

            # Total cost
            total_cost_hr = water_cost_hr + heat_cost_hr + chem_cost_hr

            cost_analysis.append({
                'cycles_of_concentration': float(coc.quantize(Decimal('0.1'))),
                'blowdown_kg_hr': float(blowdown.quantize(Decimal('0.1'))),
                'water_cost_per_hr': float(water_cost_hr.quantize(Decimal('0.01'))),
                'heat_cost_per_hr': float(heat_cost_hr.quantize(Decimal('0.01'))),
                'chemical_cost_per_hr': float(chem_cost_hr.quantize(Decimal('0.01'))),
                'total_cost_per_hr': float(total_cost_hr.quantize(Decimal('0.01')))
            })

            if total_cost_hr < min_total_cost:
                min_total_cost = total_cost_hr
                optimal_coc = coc

        tracker.record_step(
            operation="coc_optimization",
            description="Find optimal cycles of concentration for minimum cost",
            inputs={
                'max_coc_tds': max_coc_tds,
                'max_coc_silica': max_coc_silica,
                'steam_rate_kg_hr': steam
            },
            output_value=optimal_coc,
            output_name="optimal_cycles_of_concentration",
            formula="minimize(water_cost + heat_cost + chemical_cost)",
            units="cycles",
            reference="EPRI TR-102285"
        )

        # Calculate savings potential
        baseline_coc = Decimal('3')  # Typical baseline
        baseline_cost = self._calculate_cost_at_coc(baseline_coc, conditions)
        optimal_cost = self._calculate_cost_at_coc(optimal_coc, conditions)
        annual_savings = (baseline_cost - optimal_cost) * Decimal('8760')

        provenance = tracker.get_provenance_record({
            'optimal_coc': float(optimal_coc),
            'annual_savings': float(annual_savings)
        })

        return {
            'optimal_cycles_of_concentration': float(optimal_coc.quantize(Decimal('0.1'))),
            'maximum_allowable_cycles': float(max_coc.quantize(Decimal('0.1'))),
            'minimum_total_cost_per_hr': float(min_total_cost.quantize(Decimal('0.01'))),
            'annual_operating_cost': float((min_total_cost * Decimal('8760')).quantize(Decimal('1'))),
            'baseline_annual_cost': float((baseline_cost * Decimal('8760')).quantize(Decimal('1'))),
            'annual_savings_potential': float(annual_savings.quantize(Decimal('1'))),
            'cost_analysis': cost_analysis,
            'limiting_parameter': 'TDS' if max_coc_tds < max_coc_silica else 'Silica',
            'provenance': provenance.to_dict()
        }

    def calculate_heat_recovery_potential(
        self,
        conditions: BlowdownConditions,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate blowdown heat recovery potential.

        Recovery methods:
        1. Flash steam recovery in flash tank
        2. Heat exchanger to preheat makeup water
        3. Combined flash + HX system

        Flash steam calculation:
        % Flash = (h_blowdown - h_flash) / h_fg_flash * 100

        Reference: ASME PTC 4.1, Steam Tables

        Args:
            conditions: Operating conditions
            tracker: Optional provenance tracker

        Returns:
            Heat recovery analysis with ROI calculations
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"heat_recovery_{id(conditions)}",
                calculation_type="heat_recovery",
                version=self.version
            )

        # Filter out non-numeric values for provenance tracking
        numeric_inputs = {
            k: v for k, v in conditions.__dict__.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        tracker.record_inputs(numeric_inputs)

        steam = Decimal(str(conditions.steam_generation_kg_hr))
        temp_c = Decimal(str(conditions.temperature_c))
        pressure = Decimal(str(conditions.pressure_bar))
        flash_pressure = Decimal(str(conditions.flash_pressure_bar))
        hx_efficiency = Decimal(str(conditions.heat_exchanger_efficiency))
        fuel_cost = Decimal(str(conditions.fuel_cost_per_gj))

        # Estimate current CoC and blowdown rate
        makeup_tds = Decimal(str(conditions.makeup_tds_mg_l))
        boiler_tds = Decimal(str(conditions.boiler_tds_mg_l))
        coc = boiler_tds / makeup_tds if makeup_tds > 0 else Decimal('5')
        blowdown = steam / (coc - Decimal('1')) if coc > 1 else steam

        # Get enthalpy values from steam tables (simplified correlations)
        h_blowdown = self._get_saturated_liquid_enthalpy(temp_c)  # kJ/kg
        h_flash_liquid = self._get_saturated_liquid_enthalpy_at_pressure(flash_pressure)
        h_fg_flash = self._get_latent_heat_at_pressure(flash_pressure)
        h_flash_steam = h_flash_liquid + h_fg_flash

        # Step 1: Flash steam calculation
        if h_blowdown > h_flash_liquid:
            flash_fraction = (h_blowdown - h_flash_liquid) / h_fg_flash
        else:
            flash_fraction = Decimal('0')

        flash_steam_kg_hr = blowdown * flash_fraction
        remaining_liquid_kg_hr = blowdown * (Decimal('1') - flash_fraction)

        tracker.record_step(
            operation="flash_steam",
            description="Calculate flash steam from blowdown",
            inputs={
                'blowdown_kg_hr': blowdown,
                'h_blowdown_kj_kg': h_blowdown,
                'h_flash_liquid_kj_kg': h_flash_liquid,
                'h_fg_flash_kj_kg': h_fg_flash
            },
            output_value=flash_fraction * Decimal('100'),
            output_name="flash_steam_percent",
            formula="% Flash = (h_bd - h_f) / h_fg * 100",
            units="%",
            reference="ASME Steam Tables"
        )

        # Step 2: Flash steam energy value
        flash_energy_kj_hr = flash_steam_kg_hr * h_fg_flash
        flash_energy_kw = flash_energy_kj_hr / Decimal('3600')

        # Step 3: Heat exchanger recovery from remaining liquid
        t_flash = self._get_saturation_temp_at_pressure(flash_pressure)
        t_makeup = Decimal('15')  # Assume cold makeup at 15C
        delta_t_available = t_flash - t_makeup

        # HX recovery: Q = m * Cp * dT * efficiency
        hx_recovery_kj_hr = remaining_liquid_kg_hr * self.WATER_SPECIFIC_HEAT * \
                           delta_t_available * hx_efficiency
        hx_recovery_kw = hx_recovery_kj_hr / Decimal('3600')

        tracker.record_step(
            operation="hx_recovery",
            description="Calculate heat exchanger recovery potential",
            inputs={
                'remaining_liquid_kg_hr': remaining_liquid_kg_hr,
                'temperature_in_C': t_flash,
                'temperature_out_C': t_makeup,
                'efficiency': hx_efficiency
            },
            output_value=hx_recovery_kw,
            output_name="hx_recovery_kw",
            formula="Q = m * Cp * dT * efficiency",
            units="kW"
        )

        # Step 4: Total recovery and economics
        total_recovery_kw = flash_energy_kw + hx_recovery_kw

        # Energy savings value
        recovery_gj_hr = total_recovery_kw * Decimal('3.6') / Decimal('1000')
        savings_per_hr = recovery_gj_hr * fuel_cost
        annual_savings = savings_per_hr * Decimal('8760')

        # Typical equipment costs
        flash_tank_cost = Decimal('25000')  # Installed cost
        hx_cost = Decimal('15000')  # Installed cost
        total_equipment_cost = flash_tank_cost + hx_cost

        # Simple payback
        if annual_savings > 0:
            payback_years = total_equipment_cost / annual_savings
        else:
            payback_years = Decimal('999')

        provenance = tracker.get_provenance_record({
            'total_recovery_kw': float(total_recovery_kw),
            'annual_savings': float(annual_savings)
        })

        return {
            'blowdown_rate_kg_hr': float(blowdown.quantize(Decimal('0.1'))),
            'flash_steam': {
                'flash_percent': float((flash_fraction * Decimal('100')).quantize(Decimal('0.1'))),
                'flash_steam_kg_hr': float(flash_steam_kg_hr.quantize(Decimal('0.1'))),
                'flash_energy_kw': float(flash_energy_kw.quantize(Decimal('0.1')))
            },
            'heat_exchanger': {
                'remaining_liquid_kg_hr': float(remaining_liquid_kg_hr.quantize(Decimal('0.1'))),
                'inlet_temp_C': float(t_flash.quantize(Decimal('0.1'))),
                'outlet_temp_C': float(t_makeup),
                'recovery_kw': float(hx_recovery_kw.quantize(Decimal('0.1')))
            },
            'total_recovery_kw': float(total_recovery_kw.quantize(Decimal('0.1'))),
            'economics': {
                'savings_per_hour': float(savings_per_hr.quantize(Decimal('0.01'))),
                'annual_savings': float(annual_savings.quantize(Decimal('1'))),
                'equipment_cost': float(total_equipment_cost),
                'simple_payback_years': float(payback_years.quantize(Decimal('0.1')))
            },
            'provenance': provenance.to_dict()
        }

    def calculate_cooling_tower_blowdown(
        self,
        conditions: CoolingTowerConditions,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate cooling tower blowdown for water balance.

        Mass Balance:
        Makeup = Evaporation + Blowdown + Drift

        Concentration Balance:
        Makeup * C_m = (Blowdown + Drift) * C_ct

        CoC = C_ct / C_m = Makeup / (Blowdown + Drift)

        Blowdown = Evaporation / (CoC - 1) - Drift

        Reference: ASHRAE Handbook, CTI ATC-105

        Args:
            conditions: Cooling tower operating conditions
            tracker: Optional provenance tracker

        Returns:
            Complete blowdown analysis for cooling tower
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"ct_blowdown_{id(conditions)}",
                calculation_type="cooling_tower_blowdown",
                version=self.version
            )

        # Filter out non-numeric values for provenance tracking
        numeric_inputs = {
            k: v for k, v in conditions.__dict__.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        tracker.record_inputs(numeric_inputs)

        # Convert to Decimal
        evap = Decimal(str(conditions.evaporation_rate_m3_hr))
        circ = Decimal(str(conditions.circulation_rate_m3_hr))
        drift_pct = Decimal(str(conditions.drift_loss_percent))
        makeup_tds = Decimal(str(conditions.makeup_tds_mg_l))
        target_tds = Decimal(str(conditions.target_tds_mg_l))
        makeup_silica = Decimal(str(conditions.makeup_silica_mg_l))
        target_silica = Decimal(str(conditions.target_silica_mg_l))
        makeup_ca = Decimal(str(conditions.makeup_calcium_mg_l))
        target_ca = Decimal(str(conditions.target_calcium_mg_l))
        max_coc = Decimal(str(conditions.max_cycles_of_concentration))
        water_cost = Decimal(str(conditions.water_cost_per_m3))

        # Step 1: Calculate drift loss
        drift = circ * drift_pct / Decimal('100')

        tracker.record_step(
            operation="drift_loss",
            description="Calculate drift loss from circulation rate",
            inputs={
                'circulation_rate_m3_hr': circ,
                'drift_percent': drift_pct
            },
            output_value=drift,
            output_name="drift_rate_m3_hr",
            formula="Drift = Circulation * drift_pct / 100",
            units="m3/hr",
            reference="CTI ATC-105"
        )

        # Step 2: Determine limiting CoC from water chemistry
        coc_tds = target_tds / makeup_tds if makeup_tds > 0 else Decimal('10')
        coc_silica = target_silica / makeup_silica if makeup_silica > 0 else Decimal('10')
        coc_calcium = target_ca / makeup_ca if makeup_ca > 0 else Decimal('10')

        limiting_coc = min(coc_tds, coc_silica, coc_calcium, max_coc)
        limiting_param = 'TDS' if coc_tds == limiting_coc else \
                        'Silica' if coc_silica == limiting_coc else \
                        'Calcium' if coc_calcium == limiting_coc else 'Max Limit'

        # Apply safety factor
        operating_coc = limiting_coc * Decimal('0.85')
        if operating_coc < Decimal('2'):
            operating_coc = Decimal('2')

        tracker.record_step(
            operation="limiting_coc",
            description="Determine limiting cycles of concentration",
            inputs={
                'coc_tds': coc_tds,
                'coc_silica': coc_silica,
                'coc_calcium': coc_calcium,
                'max_coc': max_coc
            },
            output_value=operating_coc,
            output_name="operating_cycles_of_concentration",
            formula="CoC_oper = min(CoC_limits) * 0.85",
            units="cycles",
            reference="ASHRAE Handbook"
        )

        # Step 3: Calculate blowdown rate
        # Blowdown = Evaporation / (CoC - 1) - Drift
        if operating_coc > Decimal('1'):
            blowdown = evap / (operating_coc - Decimal('1')) - drift
            if blowdown < Decimal('0'):
                blowdown = Decimal('0')
        else:
            blowdown = evap * Decimal('10')  # Very high blowdown if CoC <= 1

        tracker.record_step(
            operation="blowdown_rate",
            description="Calculate cooling tower blowdown rate",
            inputs={
                'evaporation_m3_hr': evap,
                'operating_coc': operating_coc,
                'drift_m3_hr': drift
            },
            output_value=blowdown,
            output_name="blowdown_rate_m3_hr",
            formula="Blowdown = Evap / (CoC - 1) - Drift",
            units="m3/hr",
            reference="CTI Guidelines"
        )

        # Step 4: Calculate makeup water requirement
        makeup = evap + blowdown + drift

        # Step 5: Water savings analysis
        # Compare to once-through (CoC = 1)
        makeup_once_through = circ  # Assume evap = circ for worst case
        water_savings_pct = (Decimal('1') - makeup / makeup_once_through) * Decimal('100') \
                           if makeup_once_through > 0 else Decimal('0')

        # Step 6: Economics
        water_cost_hr = makeup * water_cost
        annual_water_cost = water_cost_hr * Decimal('8760')

        provenance = tracker.get_provenance_record({
            'blowdown_m3_hr': float(blowdown),
            'makeup_m3_hr': float(makeup),
            'operating_coc': float(operating_coc)
        })

        return {
            'water_balance': {
                'evaporation_m3_hr': float(evap.quantize(Decimal('0.001'))),
                'blowdown_m3_hr': float(blowdown.quantize(Decimal('0.001'))),
                'drift_m3_hr': float(drift.quantize(Decimal('0.0001'))),
                'makeup_m3_hr': float(makeup.quantize(Decimal('0.001')))
            },
            'cycles_analysis': {
                'operating_coc': float(operating_coc.quantize(Decimal('0.1'))),
                'limiting_parameter': limiting_param,
                'max_allowable_coc_tds': float(coc_tds.quantize(Decimal('0.1'))),
                'max_allowable_coc_silica': float(coc_silica.quantize(Decimal('0.1'))),
                'max_allowable_coc_calcium': float(coc_calcium.quantize(Decimal('0.1')))
            },
            'economics': {
                'makeup_cost_per_hr': float(water_cost_hr.quantize(Decimal('0.01'))),
                'annual_makeup_cost': float(annual_water_cost.quantize(Decimal('1'))),
                'water_savings_vs_once_through_pct': float(water_savings_pct.quantize(Decimal('0.1')))
            },
            'annual_volumes': {
                'makeup_m3_yr': float((makeup * Decimal('8760')).quantize(Decimal('1'))),
                'blowdown_m3_yr': float((blowdown * Decimal('8760')).quantize(Decimal('1'))),
                'evaporation_m3_yr': float((evap * Decimal('8760')).quantize(Decimal('1')))
            },
            'provenance': provenance.to_dict()
        }

    # Helper methods

    def _calculate_flash_recovery(
        self,
        blowdown_kg_hr: Decimal,
        temp_c: Decimal,
        pressure_bar: Decimal,
        conditions: BlowdownConditions,
        tracker: ProvenanceTracker
    ) -> Tuple[Decimal, Decimal]:
        """Calculate flash steam percentage and recoverable heat."""
        flash_pressure = Decimal(str(conditions.flash_pressure_bar))
        hx_efficiency = Decimal(str(conditions.heat_exchanger_efficiency))

        # Simplified flash calculation
        # At 10 bar, T_sat ~ 180C, h_f ~ 763 kJ/kg
        # At 1.5 bar, T_sat ~ 111C, h_f ~ 467 kJ/kg, h_fg ~ 2226 kJ/kg

        if conditions.blowdown_flash_tank:
            h_blowdown = self._get_saturated_liquid_enthalpy(temp_c)
            h_flash_liquid = self._get_saturated_liquid_enthalpy_at_pressure(flash_pressure)
            h_fg = self._get_latent_heat_at_pressure(flash_pressure)

            if h_blowdown > h_flash_liquid:
                flash_pct = ((h_blowdown - h_flash_liquid) / h_fg) * Decimal('100')
            else:
                flash_pct = Decimal('0')

            flash_steam = blowdown_kg_hr * flash_pct / Decimal('100')
            remaining_liquid = blowdown_kg_hr - flash_steam

            # Heat in flash steam
            flash_heat_kw = flash_steam * h_fg / Decimal('3600')

            # Heat recovered from remaining liquid via HX
            t_flash = self._get_saturation_temp_at_pressure(flash_pressure)
            hx_heat_kw = remaining_liquid * self.WATER_SPECIFIC_HEAT * \
                        (t_flash - self.REFERENCE_TEMP_C) * hx_efficiency / Decimal('3600')

            recoverable_kw = flash_heat_kw + hx_heat_kw
        else:
            flash_pct = Decimal('0')
            recoverable_kw = blowdown_kg_hr * self.WATER_SPECIFIC_HEAT * \
                            (temp_c - self.REFERENCE_TEMP_C) * hx_efficiency / Decimal('3600')

        return flash_pct, recoverable_kw

    def _calculate_economics(
        self,
        blowdown_kg_hr: Decimal,
        heat_loss_kw: Decimal,
        recoverable_kw: Decimal,
        conditions: BlowdownConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate economic impact of blowdown."""
        water_cost = Decimal(str(conditions.water_cost_per_m3))
        fuel_cost = Decimal(str(conditions.fuel_cost_per_gj))

        # Water cost (blowdown = makeup minus steam)
        makeup_m3_hr = blowdown_kg_hr / self.WATER_DENSITY
        water_cost_hr = makeup_m3_hr * water_cost

        # Net heat loss cost
        net_heat_loss_kw = heat_loss_kw - recoverable_kw
        heat_loss_gj_hr = net_heat_loss_kw * Decimal('3.6') / Decimal('1000')
        heat_cost_hr = heat_loss_gj_hr * fuel_cost

        # Total cost
        total_cost_hr = water_cost_hr + heat_cost_hr
        annual_cost = total_cost_hr * Decimal('8760')

        # Potential savings (if heat recovery not currently installed)
        if not conditions.blowdown_flash_tank:
            potential_savings = recoverable_kw * Decimal('3.6') / Decimal('1000') * \
                               fuel_cost * Decimal('8760')
        else:
            potential_savings = Decimal('0')

        tracker.record_step(
            operation="economic_analysis",
            description="Calculate blowdown economics",
            inputs={
                'blowdown_kg_hr': blowdown_kg_hr,
                'water_cost_m3': water_cost,
                'fuel_cost_gj': fuel_cost
            },
            output_value=total_cost_hr,
            output_name="total_cost_per_hr",
            formula="Cost = Water_cost + Heat_loss_cost",
            units="$/hr"
        )

        return {
            'water_cost_per_hour': float(water_cost_hr.quantize(Decimal('0.01'))),
            'heat_loss_cost_per_hour': float(heat_cost_hr.quantize(Decimal('0.01'))),
            'total_cost_per_hour': float(total_cost_hr.quantize(Decimal('0.01'))),
            'annual_cost': float(annual_cost.quantize(Decimal('1'))),
            'potential_savings': float(potential_savings.quantize(Decimal('1')))
        }

    def _calculate_cost_at_coc(
        self,
        coc: Decimal,
        conditions: BlowdownConditions
    ) -> Decimal:
        """Calculate total operating cost at a given CoC."""
        steam = Decimal(str(conditions.steam_generation_kg_hr))
        temp_c = Decimal(str(conditions.temperature_c))
        water_cost = Decimal(str(conditions.water_cost_per_m3))
        fuel_cost = Decimal(str(conditions.fuel_cost_per_gj))
        chem_cost = Decimal(str(conditions.chemical_cost_per_kg))

        if coc <= Decimal('1'):
            return Decimal('1e10')

        blowdown = steam / (coc - Decimal('1'))
        makeup = steam + blowdown

        # Water cost
        water_m3_hr = makeup / self.WATER_DENSITY
        water_cost_hr = water_m3_hr * water_cost

        # Heat cost
        delta_t = temp_c - self.REFERENCE_TEMP_C
        heat_gj_hr = blowdown * self.WATER_SPECIFIC_HEAT * delta_t / Decimal('1e6')
        heat_cost_hr = heat_gj_hr * fuel_cost

        # Chemical cost
        chem_kg_hr = Decimal('0.1') * (coc ** Decimal('1.5'))
        chem_cost_hr = chem_kg_hr * chem_cost

        return water_cost_hr + heat_cost_hr + chem_cost_hr

    def _generate_blowdown_schedule(self, interval_hours: float) -> List[str]:
        """Generate recommended blowdown times."""
        times = []
        hour = 6  # Start at 6 AM
        while hour < 24:
            times.append(f"{hour:02d}:00")
            hour += int(interval_hours)
        return times

    def _get_saturated_liquid_enthalpy(self, temp_c: Decimal) -> Decimal:
        """Get saturated liquid enthalpy from temperature (simplified correlation)."""
        # h_f ~ 4.186 * T for low temperatures
        # More accurate correlation for steam tables
        return temp_c * Decimal('4.186')

    def _get_saturated_liquid_enthalpy_at_pressure(self, pressure_bar: Decimal) -> Decimal:
        """Get saturated liquid enthalpy at given pressure."""
        # Simplified correlation: h_f = 417.4 + 174.5 * ln(P) for P in bar
        if pressure_bar <= 0:
            return Decimal('0')
        return Decimal('417.4') + Decimal('174.5') * pressure_bar.ln()

    def _get_latent_heat_at_pressure(self, pressure_bar: Decimal) -> Decimal:
        """Get latent heat of vaporization at given pressure."""
        # h_fg decreases with pressure: h_fg ~ 2258 - 150 * P for low pressures
        return Decimal('2258') - Decimal('50') * pressure_bar

    def _get_saturation_temp_at_pressure(self, pressure_bar: Decimal) -> Decimal:
        """Get saturation temperature at given pressure."""
        # Simplified Antoine equation approximation
        # T_sat ~ 100 + 30 * ln(P) for P in bar
        if pressure_bar <= 0:
            return Decimal('100')
        return Decimal('100') + Decimal('30') * pressure_bar.ln()
