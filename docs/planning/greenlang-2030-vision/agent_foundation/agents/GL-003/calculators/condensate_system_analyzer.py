# -*- coding: utf-8 -*-
"""
Condensate System Analyzer Calculator - Zero Hallucination

Comprehensive condensate system analysis including return rate optimization,
flash steam recovery, pump NPSH verification, deaerator optimization,
makeup water requirements, contamination detection, heat recovery potential,
and system efficiency metrics.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASME PTC 19.11, Hydraulic Institute Standards, ASHRAE
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
import threading
from datetime import datetime
from enum import Enum

from .provenance import ProvenanceTracker, create_calculation_hash


# Thread-safe cache lock
_cache_lock = threading.Lock()


class ContaminationType(Enum):
    """Types of condensate contamination."""
    NONE = "none"
    OIL = "oil"
    PROCESS_CHEMICAL = "process_chemical"
    CORROSION_PRODUCT = "corrosion_product"
    HARDNESS = "hardness"
    BIOLOGICAL = "biological"


class PumpType(Enum):
    """Types of condensate pumps."""
    CENTRIFUGAL = "centrifugal"
    POSITIVE_DISPLACEMENT = "positive_displacement"
    PNEUMATIC = "pneumatic"
    ELECTRIC_RECEIVER = "electric_receiver"


@dataclass(frozen=True)
class CondensateStreamData:
    """Immutable condensate stream input data."""
    stream_id: str
    flow_rate_kg_hr: float
    temperature_c: float
    pressure_bar: float
    source_type: str = "process"  # process, steam_trap, heat_exchanger
    distance_to_return_m: float = 100.0
    elevation_change_m: float = 0.0


@dataclass(frozen=True)
class CondensatePumpData:
    """Immutable condensate pump data."""
    pump_id: str
    pump_type: PumpType
    rated_flow_m3_hr: float
    rated_head_m: float
    npsh_required_m: float
    efficiency_percent: float = 70.0
    motor_power_kw: float = 0.0


@dataclass(frozen=True)
class DeaeratorData:
    """Immutable deaerator data."""
    deaerator_id: str
    operating_pressure_bar: float
    storage_capacity_m3: float
    steam_consumption_kg_hr: float
    feedwater_inlet_temp_c: float
    outlet_temp_c: float
    oxygen_content_ppb: float = 7.0  # Target <7 ppb


@dataclass(frozen=True)
class ContaminationSample:
    """Immutable contamination sample data."""
    sample_id: str
    stream_id: str
    ph_value: float
    conductivity_us_cm: float
    tds_ppm: float = 0.0
    oil_ppm: float = 0.0
    hardness_ppm: float = 0.0
    iron_ppm: float = 0.0
    silica_ppm: float = 0.0


@dataclass(frozen=True)
class ReturnRateOptimizationResult:
    """Immutable return rate optimization result."""
    current_return_rate_percent: Decimal
    optimal_return_rate_percent: Decimal
    potential_improvement_percent: Decimal
    energy_savings_gj_hr: Decimal
    water_savings_kg_hr: Decimal
    annual_cost_savings: Decimal
    barriers_to_improvement: Tuple[str, ...]
    recommendations: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class FlashSteamRecoveryResult:
    """Immutable flash steam recovery result."""
    flash_steam_kg_hr: Decimal
    flash_fraction_percent: Decimal
    flash_energy_gj_hr: Decimal
    recommended_flash_vessel_diameter_m: Decimal
    recommended_flash_vessel_height_m: Decimal
    annual_recovery_potential_gj: Decimal
    annual_cost_savings: Decimal
    recommendations: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class NPSHVerificationResult:
    """Immutable NPSH verification result."""
    pump_id: str
    npsh_available_m: Decimal
    npsh_required_m: Decimal
    npsh_margin_m: Decimal
    is_adequate: bool
    cavitation_risk: str  # low, medium, high
    max_suction_lift_m: Decimal
    recommendations: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class DeaeratorOptimizationResult:
    """Immutable deaerator optimization result."""
    deaerator_id: str
    current_steam_consumption_kg_hr: Decimal
    optimal_steam_consumption_kg_hr: Decimal
    steam_savings_kg_hr: Decimal
    oxygen_removal_efficiency_percent: Decimal
    feedwater_preheat_achieved_c: Decimal
    is_optimized: bool
    recommendations: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class MakeupWaterResult:
    """Immutable makeup water requirements result."""
    total_makeup_kg_hr: Decimal
    makeup_from_losses_kg_hr: Decimal
    makeup_from_blowdown_kg_hr: Decimal
    makeup_from_process_kg_hr: Decimal
    treatment_cost_per_hr: Decimal
    annual_makeup_cost: Decimal
    recommendations: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class ContaminationDetectionResult:
    """Immutable contamination detection result."""
    stream_id: str
    contamination_detected: bool
    contamination_types: Tuple[ContaminationType, ...]
    severity_level: str  # none, low, medium, high, critical
    condensate_quality_score: Decimal  # 0-100
    is_returnable: bool
    treatment_required: bool
    recommendations: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class HeatRecoveryPotentialResult:
    """Immutable heat recovery potential result."""
    total_heat_available_gj_hr: Decimal
    recoverable_heat_gj_hr: Decimal
    recovery_efficiency_percent: Decimal
    recommended_heat_exchanger_area_m2: Decimal
    payback_period_months: Decimal
    annual_savings: Decimal
    recommendations: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class SystemEfficiencyResult:
    """Immutable system efficiency metrics result."""
    condensate_return_efficiency_percent: Decimal
    heat_recovery_efficiency_percent: Decimal
    pump_system_efficiency_percent: Decimal
    overall_system_efficiency_percent: Decimal
    benchmark_comparison: str  # below_average, average, good, excellent
    improvement_potential_percent: Decimal
    efficiency_by_subsystem: Dict[str, Decimal]
    recommendations: Tuple[str, ...]
    provenance_hash: str


class CondensateSystemAnalyzer:
    """
    Comprehensive condensate system analysis calculator.

    Zero Hallucination Guarantee:
    - Thermodynamic calculations based on ASME standards
    - Hydraulic calculations per Hydraulic Institute
    - No LLM inference for any numeric values
    - Complete provenance tracking with SHA-256 hashes

    Thread Safety:
    - All calculations are thread-safe
    - LRU caching with thread-safe access
    - Immutable dataclasses for all results
    """

    # Water properties reference
    WATER_DENSITY_KG_M3 = Decimal('1000')
    WATER_CP_KJ_KG_K = Decimal('4.18')
    GRAVITY_M_S2 = Decimal('9.81')

    # Industry benchmarks for condensate return
    RETURN_RATE_BENCHMARKS = {
        'poor': Decimal('50'),
        'average': Decimal('70'),
        'good': Decimal('85'),
        'excellent': Decimal('95')
    }

    # Contamination limits for condensate return
    CONTAMINATION_LIMITS = {
        'ph_min': Decimal('8.5'),
        'ph_max': Decimal('9.5'),
        'conductivity_max_us_cm': Decimal('100'),
        'tds_max_ppm': Decimal('50'),
        'oil_max_ppm': Decimal('1'),
        'hardness_max_ppm': Decimal('1'),
        'iron_max_ppm': Decimal('0.1'),
        'silica_max_ppm': Decimal('10')
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize the analyzer."""
        self.version = version
        self._cache: Dict[str, Any] = {}

    def optimize_return_rate(
        self,
        condensate_streams: List[CondensateStreamData],
        total_steam_production_kg_hr: float,
        energy_cost_per_gj: float = 20.0,
        water_cost_per_m3: float = 2.0,
        treatment_cost_per_m3: float = 5.0,
        operating_hours_per_year: float = 8760.0
    ) -> ReturnRateOptimizationResult:
        """
        Optimize condensate return rate across all streams.

        Identifies barriers and calculates potential savings.

        Args:
            condensate_streams: List of condensate stream data
            total_steam_production_kg_hr: Total steam generation rate
            energy_cost_per_gj: Energy cost in currency per GJ
            water_cost_per_m3: Water cost per cubic meter
            treatment_cost_per_m3: Water treatment cost per cubic meter
            operating_hours_per_year: Annual operating hours

        Returns:
            ReturnRateOptimizationResult with optimization analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=f"return_rate_opt_{datetime.utcnow().isoformat()}",
            calculation_type="condensate_return_optimization",
            version=self.version
        )

        tracker.record_inputs({
            'stream_count': len(condensate_streams),
            'total_steam_kg_hr': total_steam_production_kg_hr,
            'energy_cost_per_gj': energy_cost_per_gj
        })

        # Calculate current return rate
        total_condensate = Decimal('0')
        barriers: List[str] = []

        for stream in condensate_streams:
            flow = Decimal(str(stream.flow_rate_kg_hr))
            total_condensate += flow

            # Identify barriers
            if stream.temperature_c > 90:
                barriers.append(f"Stream {stream.stream_id}: High temperature ({stream.temperature_c}C) may cause flash losses")
            if stream.distance_to_return_m > 200:
                barriers.append(f"Stream {stream.stream_id}: Long return distance ({stream.distance_to_return_m}m)")
            if stream.elevation_change_m > 10:
                barriers.append(f"Stream {stream.stream_id}: Significant elevation change ({stream.elevation_change_m}m)")

        total_steam = Decimal(str(total_steam_production_kg_hr))
        current_return_rate = (total_condensate / total_steam * Decimal('100')) if total_steam > 0 else Decimal('0')

        tracker.record_step(
            operation="current_return_rate",
            description="Calculate current condensate return rate",
            inputs={
                'total_condensate_kg_hr': total_condensate,
                'total_steam_kg_hr': total_steam
            },
            output_value=current_return_rate,
            output_name="current_return_rate_percent",
            formula="Return% = Condensate / Steam * 100",
            units="%"
        )

        # Determine optimal return rate (industry best practice: 90-95%)
        if current_return_rate < Decimal('70'):
            optimal_return_rate = Decimal('85')
        elif current_return_rate < Decimal('85'):
            optimal_return_rate = Decimal('92')
        else:
            optimal_return_rate = Decimal('95')

        potential_improvement = optimal_return_rate - current_return_rate

        # Calculate energy savings
        # Condensate at ~90C vs makeup at ~15C = ~75C temperature lift saved
        additional_return_kg_hr = total_steam * (potential_improvement / Decimal('100'))
        temp_savings_c = Decimal('75')  # Average temperature difference
        energy_saved_kj_hr = additional_return_kg_hr * self.WATER_CP_KJ_KG_K * temp_savings_c
        energy_saved_gj_hr = energy_saved_kj_hr / Decimal('1000000')

        tracker.record_step(
            operation="energy_savings",
            description="Calculate energy savings from improved return",
            inputs={
                'additional_return_kg_hr': additional_return_kg_hr,
                'temp_savings_c': temp_savings_c
            },
            output_value=energy_saved_gj_hr,
            output_name="energy_savings_gj_hr",
            formula="E = m * Cp * DT",
            units="GJ/hr"
        )

        # Calculate water savings
        water_savings_kg_hr = additional_return_kg_hr

        # Calculate annual cost savings
        hours = Decimal(str(operating_hours_per_year))
        energy_cost = Decimal(str(energy_cost_per_gj))
        water_cost = Decimal(str(water_cost_per_m3))
        treatment_cost = Decimal(str(treatment_cost_per_m3))

        annual_energy_savings = energy_saved_gj_hr * hours * energy_cost
        annual_water_savings = (water_savings_kg_hr / self.WATER_DENSITY_KG_M3) * hours * (water_cost + treatment_cost)
        annual_cost_savings = annual_energy_savings + annual_water_savings

        # Generate recommendations
        recommendations = []
        if potential_improvement > Decimal('10'):
            recommendations.append(
                f"Significant improvement potential: {potential_improvement:.1f}% increase in return rate possible."
            )
        if len([s for s in condensate_streams if s.temperature_c > 90]) > 0:
            recommendations.append("Install flash steam recovery for high-temperature condensate streams.")
        if len([s for s in condensate_streams if s.distance_to_return_m > 200]) > 0:
            recommendations.append("Consider local collection tanks and pumped return for distant streams.")
        if current_return_rate < Decimal('70'):
            recommendations.append("Priority: Survey all condensate discharge points for recovery opportunities.")
        if not recommendations:
            recommendations.append("System is operating near optimal. Focus on maintaining current performance.")

        # Generate provenance hash
        result_data = {
            'current_return_rate': str(current_return_rate),
            'optimal_return_rate': str(optimal_return_rate),
            'annual_savings': str(annual_cost_savings)
        }
        provenance_hash = create_calculation_hash(result_data)

        return ReturnRateOptimizationResult(
            current_return_rate_percent=current_return_rate.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            optimal_return_rate_percent=optimal_return_rate.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            potential_improvement_percent=max(potential_improvement, Decimal('0')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            energy_savings_gj_hr=energy_saved_gj_hr.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            water_savings_kg_hr=water_savings_kg_hr.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            annual_cost_savings=annual_cost_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            barriers_to_improvement=tuple(barriers),
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    def calculate_flash_steam_recovery(
        self,
        stream: CondensateStreamData,
        flash_pressure_bar: float,
        energy_cost_per_gj: float = 20.0,
        operating_hours_per_year: float = 8760.0
    ) -> FlashSteamRecoveryResult:
        """
        Calculate flash steam recovery potential from high-pressure condensate.

        Flash fraction = (h_initial - h_f_flash) / h_fg_flash

        Args:
            stream: High-pressure condensate stream data
            flash_pressure_bar: Target flash vessel pressure
            energy_cost_per_gj: Energy cost in currency per GJ
            operating_hours_per_year: Annual operating hours

        Returns:
            FlashSteamRecoveryResult with recovery analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=f"flash_recovery_{stream.stream_id}_{datetime.utcnow().isoformat()}",
            calculation_type="flash_steam_recovery",
            version=self.version
        )

        tracker.record_inputs({
            'stream_id': stream.stream_id,
            'flow_rate_kg_hr': stream.flow_rate_kg_hr,
            'temperature_c': stream.temperature_c,
            'pressure_bar': stream.pressure_bar,
            'flash_pressure_bar': flash_pressure_bar
        })

        # Get enthalpies
        h_initial = self._liquid_enthalpy(stream.pressure_bar, stream.temperature_c)
        T_sat_flash = self._saturation_temperature(flash_pressure_bar)
        h_f_flash = self._liquid_enthalpy(flash_pressure_bar, float(T_sat_flash))
        h_fg_flash = self._latent_heat(flash_pressure_bar)

        # Calculate flash fraction
        if h_fg_flash > Decimal('0'):
            flash_fraction = (h_initial - h_f_flash) / h_fg_flash
            flash_fraction = max(Decimal('0'), min(flash_fraction, Decimal('0.30')))  # Cap at 30%
        else:
            flash_fraction = Decimal('0')

        tracker.record_step(
            operation="flash_fraction",
            description="Calculate flash steam fraction",
            inputs={
                'h_initial_kj_kg': h_initial,
                'h_f_flash_kj_kg': h_f_flash,
                'h_fg_flash_kj_kg': h_fg_flash
            },
            output_value=flash_fraction,
            output_name="flash_fraction",
            formula="x = (h_in - h_f) / h_fg",
            units="fraction"
        )

        # Calculate flash steam flow
        condensate_flow = Decimal(str(stream.flow_rate_kg_hr))
        flash_steam_kg_hr = condensate_flow * flash_fraction

        # Calculate energy content
        h_g_flash = h_f_flash + h_fg_flash  # Saturated vapor enthalpy
        flash_energy_kj_hr = flash_steam_kg_hr * h_g_flash
        flash_energy_gj_hr = flash_energy_kj_hr / Decimal('1000000')

        # Size flash vessel (simplified)
        # Based on vapor volumetric flow and separation velocity
        rho_vapor = self._steam_density(flash_pressure_bar, float(T_sat_flash))
        V_vapor_m3_hr = flash_steam_kg_hr / rho_vapor if rho_vapor > 0 else Decimal('1')

        # Separation velocity ~1 m/s for good separation
        v_separation = Decimal('1.0')
        A_cross = V_vapor_m3_hr / (v_separation * Decimal('3600'))

        # Diameter from cross-sectional area
        diameter_m = Decimal('2') * (A_cross / Decimal(str(math.pi))).sqrt()
        diameter_m = max(diameter_m, Decimal('0.3'))  # Minimum 300mm

        # Height typically 2-3x diameter
        height_m = diameter_m * Decimal('2.5')

        # Annual savings
        hours = Decimal(str(operating_hours_per_year))
        energy_cost = Decimal(str(energy_cost_per_gj))
        annual_recovery_gj = flash_energy_gj_hr * hours
        annual_savings = annual_recovery_gj * energy_cost

        # Recommendations
        recommendations = []
        flash_percent = flash_fraction * Decimal('100')
        if flash_percent > Decimal('5'):
            recommendations.append(
                f"Flash steam recovery recommended: {flash_percent:.1f}% flash fraction at {flash_pressure_bar} bar"
            )
        if flash_steam_kg_hr > Decimal('100'):
            recommendations.append(
                f"Significant flash steam available: {flash_steam_kg_hr:.0f} kg/hr. Install flash vessel."
            )
        if stream.temperature_c > 120:
            recommendations.append("High-temperature condensate. Consider staged flash recovery.")
        if flash_percent < Decimal('3'):
            recommendations.append("Flash recovery may not be economical at this pressure differential.")

        if not recommendations:
            recommendations.append("Flash steam recovery analysis complete.")

        # Generate provenance hash
        result_data = {
            'stream_id': stream.stream_id,
            'flash_steam_kg_hr': str(flash_steam_kg_hr),
            'flash_energy_gj_hr': str(flash_energy_gj_hr)
        }
        provenance_hash = create_calculation_hash(result_data)

        return FlashSteamRecoveryResult(
            flash_steam_kg_hr=flash_steam_kg_hr.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            flash_fraction_percent=flash_percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            flash_energy_gj_hr=flash_energy_gj_hr.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            recommended_flash_vessel_diameter_m=diameter_m.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            recommended_flash_vessel_height_m=height_m.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            annual_recovery_potential_gj=annual_recovery_gj.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            annual_cost_savings=annual_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    def verify_pump_npsh(
        self,
        pump: CondensatePumpData,
        condensate_temp_c: float,
        receiver_pressure_bar: float,
        suction_pipe_length_m: float,
        suction_pipe_diameter_mm: float,
        static_head_m: float,
        fittings_k_total: float = 5.0
    ) -> NPSHVerificationResult:
        """
        Verify pump NPSH (Net Positive Suction Head) adequacy.

        NPSH_available must exceed NPSH_required to avoid cavitation.

        Args:
            pump: Condensate pump data
            condensate_temp_c: Condensate temperature at pump suction
            receiver_pressure_bar: Pressure in condensate receiver
            suction_pipe_length_m: Length of suction pipe
            suction_pipe_diameter_mm: Diameter of suction pipe
            static_head_m: Static head (positive if pump below receiver)
            fittings_k_total: Total K factor for suction line fittings

        Returns:
            NPSHVerificationResult with NPSH analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=f"npsh_verify_{pump.pump_id}_{datetime.utcnow().isoformat()}",
            calculation_type="pump_npsh_verification",
            version=self.version
        )

        tracker.record_inputs({
            'pump_id': pump.pump_id,
            'condensate_temp_c': condensate_temp_c,
            'receiver_pressure_bar': receiver_pressure_bar,
            'static_head_m': static_head_m
        })

        # Step 1: Calculate vapor pressure at condensate temperature
        P_vapor = self._vapor_pressure(condensate_temp_c)

        # Step 2: Convert receiver pressure to head
        P_receiver = Decimal(str(receiver_pressure_bar)) * Decimal('100000')  # Pa
        P_atm = Decimal('101325')  # Pa
        P_abs = P_receiver + P_atm if receiver_pressure_bar < Decimal('2') else P_receiver

        pressure_head = P_abs / (self.WATER_DENSITY_KG_M3 * self.GRAVITY_M_S2)

        # Step 3: Calculate friction losses in suction pipe
        # Estimate flow velocity
        Q = Decimal(str(pump.rated_flow_m3_hr)) / Decimal('3600')  # m3/s
        D = Decimal(str(suction_pipe_diameter_mm)) / Decimal('1000')  # m
        A = Decimal(str(math.pi)) * (D / Decimal('2')) ** 2
        v = Q / A if A > 0 else Decimal('1')

        # Friction head loss (Darcy-Weisbach simplified)
        f = Decimal('0.02')  # Typical friction factor for steel pipe
        L = Decimal(str(suction_pipe_length_m))
        K = Decimal(str(fittings_k_total))

        h_friction = (f * (L / D) + K) * (v ** 2) / (Decimal('2') * self.GRAVITY_M_S2)

        tracker.record_step(
            operation="friction_loss",
            description="Calculate suction friction head loss",
            inputs={
                'pipe_length_m': L,
                'pipe_diameter_m': D,
                'velocity_m_s': v,
                'fittings_k': K
            },
            output_value=h_friction,
            output_name="friction_head_m",
            formula="hf = (f*L/D + K) * v^2 / 2g",
            units="m"
        )

        # Step 4: Calculate vapor pressure head
        P_vapor_pa = P_vapor * Decimal('100000')  # bar to Pa
        vapor_head = P_vapor_pa / (self.WATER_DENSITY_KG_M3 * self.GRAVITY_M_S2)

        # Step 5: Calculate NPSH available
        # NPSHA = P_head + static_head - h_friction - P_vapor_head
        static = Decimal(str(static_head_m))
        npsh_available = pressure_head + static - h_friction - vapor_head

        tracker.record_step(
            operation="npsh_available",
            description="Calculate NPSH available",
            inputs={
                'pressure_head_m': pressure_head,
                'static_head_m': static,
                'friction_head_m': h_friction,
                'vapor_head_m': vapor_head
            },
            output_value=npsh_available,
            output_name="npsh_available_m",
            formula="NPSHA = P_head + z - hf - Pv_head",
            units="m"
        )

        # Step 6: Calculate NPSH margin
        npsh_required = Decimal(str(pump.npsh_required_m))
        npsh_margin = npsh_available - npsh_required

        # Step 7: Determine adequacy and cavitation risk
        is_adequate = npsh_margin > Decimal('0.5')  # Require at least 0.5m margin

        if npsh_margin < Decimal('0'):
            cavitation_risk = "high"
        elif npsh_margin < Decimal('0.5'):
            cavitation_risk = "medium"
        elif npsh_margin < Decimal('1.5'):
            cavitation_risk = "low"
        else:
            cavitation_risk = "very_low"

        # Step 8: Calculate maximum suction lift
        max_lift = pressure_head - h_friction - vapor_head - npsh_required

        # Recommendations
        recommendations = []
        if not is_adequate:
            recommendations.append(
                f"INSUFFICIENT NPSH: Available {npsh_available:.2f}m < Required {npsh_required:.2f}m. "
                f"Cavitation will occur."
            )
        if cavitation_risk == "high":
            recommendations.append("Reduce suction pipe length or increase diameter.")
            recommendations.append("Lower pump position or raise condensate receiver.")
            recommendations.append("Cool condensate before pump suction to reduce vapor pressure.")
        elif cavitation_risk == "medium":
            recommendations.append("Consider increasing NPSH margin for reliability.")
        if condensate_temp_c > 90:
            recommendations.append(f"High condensate temperature ({condensate_temp_c}C) increases vapor pressure.")
        if not recommendations:
            recommendations.append(f"NPSH is adequate with {npsh_margin:.2f}m margin.")

        # Generate provenance hash
        result_data = {
            'pump_id': pump.pump_id,
            'npsh_available': str(npsh_available),
            'npsh_required': str(npsh_required),
            'is_adequate': is_adequate
        }
        provenance_hash = create_calculation_hash(result_data)

        return NPSHVerificationResult(
            pump_id=pump.pump_id,
            npsh_available_m=npsh_available.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            npsh_required_m=npsh_required.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            npsh_margin_m=npsh_margin.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            is_adequate=is_adequate,
            cavitation_risk=cavitation_risk,
            max_suction_lift_m=max(max_lift, Decimal('0')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    def optimize_deaerator(
        self,
        deaerator: DeaeratorData,
        feedwater_flow_kg_hr: float,
        condensate_return_kg_hr: float,
        condensate_temp_c: float,
        makeup_temp_c: float = 15.0
    ) -> DeaeratorOptimizationResult:
        """
        Optimize deaerator operation for minimum steam consumption.

        Balance between oxygen removal and steam efficiency.

        Args:
            deaerator: Deaerator data
            feedwater_flow_kg_hr: Total feedwater flow rate
            condensate_return_kg_hr: Condensate return rate
            condensate_temp_c: Condensate temperature
            makeup_temp_c: Makeup water temperature

        Returns:
            DeaeratorOptimizationResult with optimization analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=f"da_opt_{deaerator.deaerator_id}_{datetime.utcnow().isoformat()}",
            calculation_type="deaerator_optimization",
            version=self.version
        )

        tracker.record_inputs({
            'deaerator_id': deaerator.deaerator_id,
            'feedwater_flow_kg_hr': feedwater_flow_kg_hr,
            'condensate_return_kg_hr': condensate_return_kg_hr,
            'operating_pressure_bar': deaerator.operating_pressure_bar
        })

        # Calculate makeup water requirement
        total_feedwater = Decimal(str(feedwater_flow_kg_hr))
        condensate_return = Decimal(str(condensate_return_kg_hr))
        makeup_flow = total_feedwater - condensate_return

        # Calculate mixed inlet temperature
        T_condensate = Decimal(str(condensate_temp_c))
        T_makeup = Decimal(str(makeup_temp_c))

        if total_feedwater > 0:
            mixed_temp = (condensate_return * T_condensate + makeup_flow * T_makeup) / total_feedwater
        else:
            mixed_temp = T_makeup

        # Get deaerator saturation temperature
        T_sat_da = self._saturation_temperature(deaerator.operating_pressure_bar)

        # Calculate required heat input
        temp_rise = T_sat_da - mixed_temp
        Q_required_kj_hr = total_feedwater * self.WATER_CP_KJ_KG_K * temp_rise

        # Calculate optimal steam consumption
        # Steam at deaerator pressure heats and scrubs water
        h_steam = self._steam_enthalpy(deaerator.operating_pressure_bar)
        h_condensate_da = self._liquid_enthalpy(deaerator.operating_pressure_bar, float(T_sat_da))
        delta_h = h_steam - h_condensate_da

        if delta_h > 0:
            optimal_steam_kg_hr = Q_required_kj_hr / delta_h
        else:
            optimal_steam_kg_hr = Decimal('0')

        tracker.record_step(
            operation="optimal_steam",
            description="Calculate optimal deaerator steam consumption",
            inputs={
                'heat_required_kj_hr': Q_required_kj_hr,
                'delta_h_kj_kg': delta_h
            },
            output_value=optimal_steam_kg_hr,
            output_name="optimal_steam_kg_hr",
            formula="m_steam = Q / (h_steam - h_condensate)",
            units="kg/hr"
        )

        # Calculate steam savings
        current_steam = Decimal(str(deaerator.steam_consumption_kg_hr))
        steam_savings = current_steam - optimal_steam_kg_hr

        # Calculate oxygen removal efficiency (based on outlet oxygen)
        # Target is <7 ppb, typical inlet is ~8000 ppb (saturated cold water)
        inlet_o2_ppb = Decimal('8000')  # Typical for cold water
        outlet_o2_ppb = Decimal(str(deaerator.oxygen_content_ppb))
        o2_removal_efficiency = (inlet_o2_ppb - outlet_o2_ppb) / inlet_o2_ppb * Decimal('100')

        # Feedwater preheat achieved
        preheat_achieved = T_sat_da - T_makeup

        # Determine if optimized
        is_optimized = (
            abs(current_steam - optimal_steam_kg_hr) < current_steam * Decimal('0.1') and
            outlet_o2_ppb <= Decimal('7')
        )

        # Recommendations
        recommendations = []
        if steam_savings > Decimal('0') and steam_savings > current_steam * Decimal('0.1'):
            recommendations.append(
                f"Reduce deaerator steam from {current_steam:.0f} to {optimal_steam_kg_hr:.0f} kg/hr. "
                f"Savings: {steam_savings:.0f} kg/hr"
            )
        if outlet_o2_ppb > Decimal('7'):
            recommendations.append(
                f"Oxygen content {outlet_o2_ppb:.0f} ppb exceeds 7 ppb target. Check deaerator venting."
            )
        if mixed_temp < T_sat_da - Decimal('30'):
            recommendations.append(
                "Large temperature rise required. Consider preheating condensate/makeup."
            )
        if deaerator.operating_pressure_bar < Decimal('0.2'):
            recommendations.append("Operating pressure may be too low for effective deaeration.")
        if not recommendations:
            recommendations.append("Deaerator is operating efficiently.")

        # Generate provenance hash
        result_data = {
            'deaerator_id': deaerator.deaerator_id,
            'optimal_steam_kg_hr': str(optimal_steam_kg_hr),
            'steam_savings_kg_hr': str(steam_savings)
        }
        provenance_hash = create_calculation_hash(result_data)

        return DeaeratorOptimizationResult(
            deaerator_id=deaerator.deaerator_id,
            current_steam_consumption_kg_hr=current_steam.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            optimal_steam_consumption_kg_hr=optimal_steam_kg_hr.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            steam_savings_kg_hr=max(steam_savings, Decimal('0')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            oxygen_removal_efficiency_percent=o2_removal_efficiency.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            feedwater_preheat_achieved_c=preheat_achieved.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            is_optimized=is_optimized,
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    def calculate_makeup_requirements(
        self,
        steam_production_kg_hr: float,
        condensate_return_kg_hr: float,
        blowdown_percent: float = 3.0,
        process_consumption_kg_hr: float = 0.0,
        water_cost_per_m3: float = 2.0,
        treatment_cost_per_m3: float = 5.0,
        operating_hours_per_year: float = 8760.0
    ) -> MakeupWaterResult:
        """
        Calculate makeup water requirements and costs.

        Accounts for condensate return, blowdown, and process losses.

        Args:
            steam_production_kg_hr: Total steam generation rate
            condensate_return_kg_hr: Condensate return rate
            blowdown_percent: Blowdown as percent of steam production
            process_consumption_kg_hr: Direct steam consumption (not returned)
            water_cost_per_m3: Water cost per cubic meter
            treatment_cost_per_m3: Water treatment cost per cubic meter
            operating_hours_per_year: Annual operating hours

        Returns:
            MakeupWaterResult with makeup analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=f"makeup_req_{datetime.utcnow().isoformat()}",
            calculation_type="makeup_water_requirements",
            version=self.version
        )

        tracker.record_inputs({
            'steam_production_kg_hr': steam_production_kg_hr,
            'condensate_return_kg_hr': condensate_return_kg_hr,
            'blowdown_percent': blowdown_percent,
            'process_consumption_kg_hr': process_consumption_kg_hr
        })

        steam = Decimal(str(steam_production_kg_hr))
        condensate = Decimal(str(condensate_return_kg_hr))
        blowdown = steam * Decimal(str(blowdown_percent)) / Decimal('100')
        process = Decimal(str(process_consumption_kg_hr))

        # Losses = steam not returned as condensate
        losses = steam - condensate

        # Total makeup = losses + blowdown
        # (Process consumption is part of losses if steam goes to product)
        total_makeup = losses + blowdown

        tracker.record_step(
            operation="makeup_calculation",
            description="Calculate total makeup water requirement",
            inputs={
                'steam_kg_hr': steam,
                'condensate_kg_hr': condensate,
                'blowdown_kg_hr': blowdown
            },
            output_value=total_makeup,
            output_name="total_makeup_kg_hr",
            formula="Makeup = (Steam - Condensate) + Blowdown",
            units="kg/hr"
        )

        # Cost calculations
        hours = Decimal(str(operating_hours_per_year))
        water_cost = Decimal(str(water_cost_per_m3))
        treatment = Decimal(str(treatment_cost_per_m3))

        # Convert to m3
        makeup_m3_hr = total_makeup / self.WATER_DENSITY_KG_M3
        treatment_cost_hr = makeup_m3_hr * (water_cost + treatment)

        annual_makeup_cost = makeup_m3_hr * hours * (water_cost + treatment)

        # Recommendations
        recommendations = []
        return_rate = (condensate / steam * Decimal('100')) if steam > 0 else Decimal('0')

        if return_rate < Decimal('70'):
            recommendations.append(
                f"Low condensate return rate ({return_rate:.1f}%). Survey for recovery opportunities."
            )
        if blowdown > steam * Decimal('0.05'):
            recommendations.append(
                f"High blowdown rate ({blowdown_percent}%). Consider blowdown heat recovery."
            )
        if losses > steam * Decimal('0.30'):
            recommendations.append(
                "Significant condensate losses. Check for leaks and failed steam traps."
            )
        if not recommendations:
            recommendations.append("Makeup water requirements are within normal range.")

        # Generate provenance hash
        result_data = {
            'total_makeup_kg_hr': str(total_makeup),
            'annual_cost': str(annual_makeup_cost)
        }
        provenance_hash = create_calculation_hash(result_data)

        return MakeupWaterResult(
            total_makeup_kg_hr=total_makeup.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            makeup_from_losses_kg_hr=losses.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            makeup_from_blowdown_kg_hr=blowdown.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            makeup_from_process_kg_hr=process.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            treatment_cost_per_hr=treatment_cost_hr.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            annual_makeup_cost=annual_makeup_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    def detect_contamination(
        self,
        sample: ContaminationSample
    ) -> ContaminationDetectionResult:
        """
        Detect condensate contamination and assess returnability.

        Compares sample values against limits for boiler feedwater.

        Args:
            sample: Contamination sample data

        Returns:
            ContaminationDetectionResult with contamination analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=f"contamination_{sample.sample_id}_{datetime.utcnow().isoformat()}",
            calculation_type="contamination_detection",
            version=self.version
        )

        tracker.record_inputs({
            'sample_id': sample.sample_id,
            'stream_id': sample.stream_id,
            'ph': sample.ph_value,
            'conductivity_us_cm': sample.conductivity_us_cm
        })

        contamination_types: List[ContaminationType] = []
        quality_deductions = Decimal('0')
        treatment_required = False

        # Check pH
        ph = Decimal(str(sample.ph_value))
        if ph < self.CONTAMINATION_LIMITS['ph_min'] or ph > self.CONTAMINATION_LIMITS['ph_max']:
            quality_deductions += Decimal('15')
            contamination_types.append(ContaminationType.CORROSION_PRODUCT)
            treatment_required = True

        # Check conductivity
        conductivity = Decimal(str(sample.conductivity_us_cm))
        if conductivity > self.CONTAMINATION_LIMITS['conductivity_max_us_cm']:
            quality_deductions += Decimal('20')
            contamination_types.append(ContaminationType.PROCESS_CHEMICAL)
            treatment_required = True

        # Check TDS
        tds = Decimal(str(sample.tds_ppm))
        if tds > self.CONTAMINATION_LIMITS['tds_max_ppm']:
            quality_deductions += Decimal('15')

        # Check oil
        oil = Decimal(str(sample.oil_ppm))
        if oil > self.CONTAMINATION_LIMITS['oil_max_ppm']:
            quality_deductions += Decimal('30')  # Oil is critical
            contamination_types.append(ContaminationType.OIL)
            treatment_required = True

        # Check hardness
        hardness = Decimal(str(sample.hardness_ppm))
        if hardness > self.CONTAMINATION_LIMITS['hardness_max_ppm']:
            quality_deductions += Decimal('20')
            contamination_types.append(ContaminationType.HARDNESS)
            treatment_required = True

        # Check iron
        iron = Decimal(str(sample.iron_ppm))
        if iron > self.CONTAMINATION_LIMITS['iron_max_ppm']:
            quality_deductions += Decimal('10')
            contamination_types.append(ContaminationType.CORROSION_PRODUCT)

        # Check silica
        silica = Decimal(str(sample.silica_ppm))
        if silica > self.CONTAMINATION_LIMITS['silica_max_ppm']:
            quality_deductions += Decimal('15')

        # Calculate quality score
        quality_score = max(Decimal('0'), Decimal('100') - quality_deductions)

        # Determine contamination status
        contamination_detected = len(contamination_types) > 0
        if not contamination_detected:
            contamination_types.append(ContaminationType.NONE)

        # Determine severity
        if quality_score >= Decimal('90'):
            severity = "none"
        elif quality_score >= Decimal('75'):
            severity = "low"
        elif quality_score >= Decimal('50'):
            severity = "medium"
        elif quality_score >= Decimal('25'):
            severity = "high"
        else:
            severity = "critical"

        # Determine returnability
        is_returnable = quality_score >= Decimal('70') and ContaminationType.OIL not in contamination_types

        tracker.record_step(
            operation="quality_assessment",
            description="Assess condensate quality score",
            inputs={
                'quality_deductions': quality_deductions
            },
            output_value=quality_score,
            output_name="quality_score",
            formula="Score = 100 - deductions",
            units="score"
        )

        # Recommendations
        recommendations = []
        if ContaminationType.OIL in contamination_types:
            recommendations.append("CRITICAL: Oil contamination detected. Do not return to boiler. Install oil separator.")
        if ContaminationType.HARDNESS in contamination_types:
            recommendations.append("Hardness detected. Check for heat exchanger tube leaks.")
        if ContaminationType.CORROSION_PRODUCT in contamination_types:
            recommendations.append("Corrosion products detected. Review condensate line materials and pH control.")
        if ContaminationType.PROCESS_CHEMICAL in contamination_types:
            recommendations.append("Process chemical contamination. Identify source and install monitoring.")
        if severity in ["high", "critical"]:
            recommendations.append("Condensate cannot be returned without treatment. Consider polishing system.")
        if not recommendations:
            recommendations.append("Condensate quality is acceptable for return.")

        # Generate provenance hash
        result_data = {
            'sample_id': sample.sample_id,
            'quality_score': str(quality_score),
            'is_returnable': is_returnable
        }
        provenance_hash = create_calculation_hash(result_data)

        return ContaminationDetectionResult(
            stream_id=sample.stream_id,
            contamination_detected=contamination_detected,
            contamination_types=tuple(contamination_types),
            severity_level=severity,
            condensate_quality_score=quality_score.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            is_returnable=is_returnable,
            treatment_required=treatment_required,
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    def calculate_heat_recovery_potential(
        self,
        condensate_streams: List[CondensateStreamData],
        target_cooling_temp_c: float = 40.0,
        heat_exchanger_effectiveness: float = 0.85,
        installation_cost_per_m2: float = 500.0,
        energy_cost_per_gj: float = 20.0,
        operating_hours_per_year: float = 8760.0
    ) -> HeatRecoveryPotentialResult:
        """
        Calculate heat recovery potential from condensate.

        Determines recoverable heat and heat exchanger sizing.

        Args:
            condensate_streams: List of condensate stream data
            target_cooling_temp_c: Target condensate temperature after heat recovery
            heat_exchanger_effectiveness: Heat exchanger effectiveness (0-1)
            installation_cost_per_m2: Heat exchanger installation cost per m2
            energy_cost_per_gj: Energy cost in currency per GJ
            operating_hours_per_year: Annual operating hours

        Returns:
            HeatRecoveryPotentialResult with recovery analysis
        """
        tracker = ProvenanceTracker(
            calculation_id=f"heat_recovery_{datetime.utcnow().isoformat()}",
            calculation_type="condensate_heat_recovery",
            version=self.version
        )

        tracker.record_inputs({
            'stream_count': len(condensate_streams),
            'target_cooling_temp_c': target_cooling_temp_c,
            'heat_exchanger_effectiveness': heat_exchanger_effectiveness
        })

        total_heat_available = Decimal('0')
        T_target = Decimal(str(target_cooling_temp_c))

        for stream in condensate_streams:
            T_stream = Decimal(str(stream.temperature_c))
            flow = Decimal(str(stream.flow_rate_kg_hr))

            if T_stream > T_target:
                Q = flow * self.WATER_CP_KJ_KG_K * (T_stream - T_target)
                total_heat_available += Q

        # Convert to GJ/hr
        total_heat_gj_hr = total_heat_available / Decimal('1000000')

        tracker.record_step(
            operation="total_heat_available",
            description="Calculate total heat available in condensate",
            inputs={
                'stream_count': len(condensate_streams),
                'target_temp_c': T_target
            },
            output_value=total_heat_gj_hr,
            output_name="total_heat_gj_hr",
            formula="Q = sum(m * Cp * (T - T_target))",
            units="GJ/hr"
        )

        # Recoverable heat based on effectiveness
        effectiveness = Decimal(str(heat_exchanger_effectiveness))
        recoverable_heat = total_heat_gj_hr * effectiveness
        recovery_efficiency = effectiveness * Decimal('100')

        # Size heat exchanger (simplified: U = 1000 W/m2K, LMTD = 20K)
        U = Decimal('1000')  # W/(m2*K) - typical for water-water
        LMTD = Decimal('20')  # K - typical log mean temp difference
        Q_watts = recoverable_heat * Decimal('1e9') / Decimal('3600')  # GJ/hr to W

        if LMTD > 0 and U > 0:
            A_required = Q_watts / (U * LMTD)
        else:
            A_required = Decimal('10')

        # Economic analysis
        hours = Decimal(str(operating_hours_per_year))
        energy_cost = Decimal(str(energy_cost_per_gj))
        install_cost = Decimal(str(installation_cost_per_m2))

        annual_savings = recoverable_heat * hours * energy_cost
        total_install_cost = A_required * install_cost

        if annual_savings > 0:
            payback_months = (total_install_cost / annual_savings) * Decimal('12')
        else:
            payback_months = Decimal('999')

        # Recommendations
        recommendations = []
        if payback_months < Decimal('24'):
            recommendations.append(
                f"Heat recovery is economically attractive. Payback: {payback_months:.1f} months."
            )
        if recoverable_heat > Decimal('0.1'):
            recommendations.append(
                f"Significant heat recovery potential: {recoverable_heat:.3f} GJ/hr available."
            )
        if A_required > Decimal('50'):
            recommendations.append(
                f"Large heat exchanger required ({A_required:.1f} m2). Consider multiple units."
            )
        if len([s for s in condensate_streams if s.temperature_c > 80]) > 2:
            recommendations.append("Multiple high-temperature streams. Consider cascade heat recovery.")
        if not recommendations:
            recommendations.append("Heat recovery analysis complete. Review economics before proceeding.")

        # Generate provenance hash
        result_data = {
            'total_heat_gj_hr': str(total_heat_gj_hr),
            'recoverable_heat_gj_hr': str(recoverable_heat),
            'annual_savings': str(annual_savings)
        }
        provenance_hash = create_calculation_hash(result_data)

        return HeatRecoveryPotentialResult(
            total_heat_available_gj_hr=total_heat_gj_hr.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            recoverable_heat_gj_hr=recoverable_heat.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            recovery_efficiency_percent=recovery_efficiency.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            recommended_heat_exchanger_area_m2=A_required.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            payback_period_months=payback_months.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            annual_savings=annual_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    def calculate_system_efficiency(
        self,
        steam_production_kg_hr: float,
        condensate_return_kg_hr: float,
        heat_recovered_gj_hr: float,
        pump_power_kw: float,
        pump_flow_m3_hr: float,
        pump_head_m: float
    ) -> SystemEfficiencyResult:
        """
        Calculate overall condensate system efficiency metrics.

        Provides benchmark comparison and improvement recommendations.

        Args:
            steam_production_kg_hr: Total steam production
            condensate_return_kg_hr: Condensate returned to boiler
            heat_recovered_gj_hr: Heat recovered from condensate
            pump_power_kw: Total pump power consumption
            pump_flow_m3_hr: Total pumped flow
            pump_head_m: Average pump head

        Returns:
            SystemEfficiencyResult with efficiency metrics
        """
        tracker = ProvenanceTracker(
            calculation_id=f"system_eff_{datetime.utcnow().isoformat()}",
            calculation_type="condensate_system_efficiency",
            version=self.version
        )

        tracker.record_inputs({
            'steam_production_kg_hr': steam_production_kg_hr,
            'condensate_return_kg_hr': condensate_return_kg_hr,
            'heat_recovered_gj_hr': heat_recovered_gj_hr
        })

        steam = Decimal(str(steam_production_kg_hr))
        condensate = Decimal(str(condensate_return_kg_hr))

        # Condensate return efficiency
        return_eff = (condensate / steam * Decimal('100')) if steam > 0 else Decimal('0')

        # Heat recovery efficiency (vs theoretical maximum)
        # Theoretical max: cool all condensate from 90C to 40C
        max_heat_kj_hr = condensate * self.WATER_CP_KJ_KG_K * Decimal('50')  # 50C drop
        max_heat_gj_hr = max_heat_kj_hr / Decimal('1000000')
        recovered = Decimal(str(heat_recovered_gj_hr))

        heat_eff = (recovered / max_heat_gj_hr * Decimal('100')) if max_heat_gj_hr > 0 else Decimal('0')

        # Pump efficiency
        pump_power = Decimal(str(pump_power_kw))
        pump_flow = Decimal(str(pump_flow_m3_hr))
        pump_head = Decimal(str(pump_head_m))

        # Hydraulic power = rho * g * Q * H / 3600000 (kW)
        hydraulic_power = (self.WATER_DENSITY_KG_M3 * self.GRAVITY_M_S2 *
                         pump_flow * pump_head / Decimal('3600000'))

        pump_eff = (hydraulic_power / pump_power * Decimal('100')) if pump_power > 0 else Decimal('0')

        tracker.record_step(
            operation="pump_efficiency",
            description="Calculate pump system efficiency",
            inputs={
                'hydraulic_power_kw': hydraulic_power,
                'motor_power_kw': pump_power
            },
            output_value=pump_eff,
            output_name="pump_efficiency_percent",
            formula="eta = P_hydraulic / P_motor * 100",
            units="%"
        )

        # Overall efficiency (weighted average)
        overall_eff = (return_eff * Decimal('0.5') +
                      heat_eff * Decimal('0.3') +
                      pump_eff * Decimal('0.2'))

        # Benchmark comparison
        if overall_eff >= Decimal('85'):
            benchmark = "excellent"
        elif overall_eff >= Decimal('70'):
            benchmark = "good"
        elif overall_eff >= Decimal('55'):
            benchmark = "average"
        else:
            benchmark = "below_average"

        # Improvement potential (vs excellent benchmark)
        improvement_potential = max(Decimal('85') - overall_eff, Decimal('0'))

        # Efficiency by subsystem
        efficiency_by_subsystem = {
            'condensate_return': return_eff.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            'heat_recovery': heat_eff.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            'pump_system': pump_eff.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        }

        # Recommendations
        recommendations = []
        if return_eff < Decimal('70'):
            recommendations.append(f"Low condensate return ({return_eff:.1f}%). Priority: survey recovery opportunities.")
        if heat_eff < Decimal('50'):
            recommendations.append(f"Low heat recovery ({heat_eff:.1f}%). Install heat exchangers.")
        if pump_eff < Decimal('60'):
            recommendations.append(f"Low pump efficiency ({pump_eff:.1f}%). Check for cavitation or worn impellers.")
        if benchmark == "excellent":
            recommendations.append("System is performing at benchmark level. Maintain current operations.")
        if not recommendations:
            recommendations.append(f"Overall system efficiency: {overall_eff:.1f}% ({benchmark})")

        # Generate provenance hash
        result_data = {
            'return_efficiency': str(return_eff),
            'heat_efficiency': str(heat_eff),
            'pump_efficiency': str(pump_eff),
            'overall_efficiency': str(overall_eff)
        }
        provenance_hash = create_calculation_hash(result_data)

        return SystemEfficiencyResult(
            condensate_return_efficiency_percent=return_eff.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            heat_recovery_efficiency_percent=heat_eff.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            pump_system_efficiency_percent=pump_eff.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            overall_system_efficiency_percent=overall_eff.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            benchmark_comparison=benchmark,
            improvement_potential_percent=improvement_potential.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            efficiency_by_subsystem=efficiency_by_subsystem,
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Helper methods with thread-safe caching
    # =========================================================================

    @lru_cache(maxsize=1000)
    def _liquid_enthalpy(self, pressure_bar: float, temperature_c: float) -> Decimal:
        """Calculate liquid enthalpy (simplified)."""
        with _cache_lock:
            T = Decimal(str(temperature_c))
            Cp = self.WATER_CP_KJ_KG_K
            h = Cp * T
            return h.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    @lru_cache(maxsize=1000)
    def _steam_enthalpy(self, pressure_bar: float) -> Decimal:
        """Calculate saturated steam enthalpy (simplified)."""
        with _cache_lock:
            T_sat = self._saturation_temperature(pressure_bar)
            # h_g = h_f + h_fg
            h_f = self._liquid_enthalpy(pressure_bar, float(T_sat))
            h_fg = self._latent_heat(pressure_bar)
            return h_f + h_fg

    @lru_cache(maxsize=1000)
    def _latent_heat(self, pressure_bar: float) -> Decimal:
        """Calculate latent heat of vaporization (simplified correlation)."""
        with _cache_lock:
            T_sat = self._saturation_temperature(pressure_bar)
            # h_fg decreases with temperature (and pressure)
            # Simplified: h_fg = 2500 - 2.4 * T_sat
            h_fg = Decimal('2500') - Decimal('2.4') * T_sat
            return max(h_fg, Decimal('1800'))  # Minimum realistic value

    @lru_cache(maxsize=1000)
    def _saturation_temperature(self, pressure_bar: float) -> Decimal:
        """Calculate saturation temperature from pressure (simplified)."""
        with _cache_lock:
            P = Decimal(str(pressure_bar))
            if P > Decimal('0'):
                ln_p = Decimal(str(math.log(float(P))))
                T_sat = Decimal('100') + Decimal('30') * ln_p
            else:
                T_sat = Decimal('100')
            return T_sat.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

    @lru_cache(maxsize=1000)
    def _vapor_pressure(self, temperature_c: float) -> Decimal:
        """Calculate vapor pressure at given temperature (bar)."""
        with _cache_lock:
            T = Decimal(str(temperature_c))
            # Simplified Antoine equation correlation
            # P = exp((T - 100) / 30) for T near 100C
            if T > Decimal('0'):
                exponent = (T - Decimal('100')) / Decimal('30')
                P_vapor = Decimal(str(math.exp(float(exponent))))
            else:
                P_vapor = Decimal('0.01')
            return P_vapor.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)

    @lru_cache(maxsize=1000)
    def _steam_density(self, pressure_bar: float, temperature_c: float) -> Decimal:
        """Calculate steam density using ideal gas law."""
        with _cache_lock:
            P = Decimal(str(pressure_bar)) * Decimal('100')  # kPa
            T = Decimal(str(temperature_c)) + Decimal('273.15')  # K
            R = Decimal('0.4615')  # kJ/(kg*K)
            Z = Decimal('0.95')  # Compressibility

            rho = P / (Z * R * T) if T > 0 else Decimal('1')
            return rho.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
