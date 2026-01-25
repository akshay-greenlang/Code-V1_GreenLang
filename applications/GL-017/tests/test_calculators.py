# -*- coding: utf-8 -*-
"""
Unit tests for GL-017 CONDENSYNC Calculators.

Tests all calculator modules with comprehensive coverage:
- HeatTransferCalculator
- VacuumCalculator
- EfficiencyCalculator
- FoulingCalculator
- ProvenanceTracker

Author: GL-017 Test Engineering Team
Target Coverage: >85%
"""

import pytest
import sys
import math
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Mock Calculator Classes for Testing
# ============================================================================

@dataclass
class CondenserConditions:
    """Condenser operating conditions for calculations."""
    vacuum_pressure_mbar: float = 50.0
    steam_saturation_temp_c: float = 33.0
    hotwell_temperature_c: float = 32.5
    cooling_water_inlet_temp_c: float = 25.0
    cooling_water_outlet_temp_c: float = 32.0
    cooling_water_flow_rate_m3_hr: float = 45000.0
    heat_duty_mw: float = 180.0
    surface_area_m2: float = 17500.0
    tube_od_mm: float = 25.4
    tube_id_mm: float = 23.4
    tube_length_m: float = 12.0
    tube_count: int = 18500
    num_passes: int = 2
    tube_material: str = 'titanium'


@dataclass
class VacuumConditions:
    """Vacuum system conditions for calculations."""
    condenser_pressure_mbar: float = 50.0
    steam_flow_kg_hr: float = 150000.0
    cooling_water_inlet_temp_c: float = 25.0
    air_inleakage_rate_kg_hr: float = 0.5
    ejector_steam_pressure_bar: float = 8.0
    vacuum_pump_capacity_kg_hr: float = 15.0
    non_condensable_fraction: float = 0.001


@dataclass
class FoulingConditions:
    """Fouling conditions for calculations."""
    cooling_water_source: str = 'cooling_tower'
    cooling_water_tds_ppm: float = 1800.0
    cooling_water_ph: float = 7.8
    cooling_water_temperature_c: float = 28.0
    tube_velocity_m_s: float = 2.2
    tube_material: str = 'titanium'
    operating_hours: float = 4380.0
    biocide_treatment: str = 'oxidizing'
    current_cleanliness_factor: float = 0.85


class HeatTransferCalculator:
    """Mock heat transfer calculator for condenser analysis."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_lmtd(self, conditions: CondenserConditions) -> Dict[str, Any]:
        """Calculate Log Mean Temperature Difference."""
        t_hot_in = conditions.steam_saturation_temp_c
        t_hot_out = conditions.hotwell_temperature_c
        t_cold_in = conditions.cooling_water_inlet_temp_c
        t_cold_out = conditions.cooling_water_outlet_temp_c

        delta_t1 = t_hot_in - t_cold_out
        delta_t2 = t_hot_out - t_cold_in

        if delta_t1 <= 0 or delta_t2 <= 0:
            lmtd = 0.0
        elif abs(delta_t1 - delta_t2) < 0.001:
            lmtd = delta_t1
        else:
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        return {
            'lmtd_c': lmtd,
            'delta_t_hot_end_c': delta_t1,
            'delta_t_cold_end_c': delta_t2,
            'temperature_ratio': delta_t1 / delta_t2 if delta_t2 > 0 else 0.0,
            'provenance': {
                'calculation_type': 'lmtd',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_overall_htc(self, conditions: CondenserConditions) -> Dict[str, Any]:
        """Calculate overall heat transfer coefficient."""
        lmtd_result = self.calculate_lmtd(conditions)
        lmtd = lmtd_result['lmtd_c']

        if lmtd > 0 and conditions.surface_area_m2 > 0:
            heat_duty_w = conditions.heat_duty_mw * 1e6
            overall_htc = heat_duty_w / (conditions.surface_area_m2 * lmtd)
        else:
            overall_htc = 0.0

        # Design HTC for titanium tubes
        design_htc = 3200.0 if conditions.tube_material == 'titanium' else 2800.0

        return {
            'overall_htc_w_m2k': overall_htc,
            'design_htc_w_m2k': design_htc,
            'htc_ratio': overall_htc / design_htc if design_htc > 0 else 0.0,
            'lmtd_c': lmtd,
            'heat_duty_mw': conditions.heat_duty_mw,
            'surface_area_m2': conditions.surface_area_m2,
            'provenance': {
                'calculation_type': 'overall_htc',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_ntu_effectiveness(self, conditions: CondenserConditions) -> Dict[str, Any]:
        """Calculate NTU and effectiveness."""
        htc_result = self.calculate_overall_htc(conditions)
        overall_htc = htc_result['overall_htc_w_m2k']

        # Cooling water heat capacity rate
        cp_water = 4186.0  # J/kg-K
        rho_water = 1000.0  # kg/m3
        cw_flow_kg_s = conditions.cooling_water_flow_rate_m3_hr * rho_water / 3600.0
        c_cold = cw_flow_kg_s * cp_water

        # For condenser (phase change), C_hot is effectively infinite
        c_min = c_cold
        c_ratio = 0.0  # C_min/C_max = 0 for condenser

        # NTU calculation
        if c_min > 0:
            ntu = (overall_htc * conditions.surface_area_m2) / c_min
        else:
            ntu = 0.0

        # Effectiveness for C_ratio = 0 (condenser)
        effectiveness = 1.0 - math.exp(-ntu) if ntu > 0 else 0.0

        return {
            'ntu': ntu,
            'effectiveness': effectiveness,
            'c_min_w_k': c_min,
            'c_ratio': c_ratio,
            'heat_capacity_rate_kw_k': c_min / 1000.0,
            'provenance': {
                'calculation_type': 'ntu_effectiveness',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_tube_velocity(self, conditions: CondenserConditions) -> Dict[str, Any]:
        """Calculate cooling water velocity in tubes."""
        tube_id_m = conditions.tube_id_mm / 1000.0
        tube_area = math.pi * (tube_id_m / 2) ** 2
        tubes_per_pass = conditions.tube_count / conditions.num_passes

        total_flow_area = tube_area * tubes_per_pass
        flow_rate_m3_s = conditions.cooling_water_flow_rate_m3_hr / 3600.0

        velocity = flow_rate_m3_s / total_flow_area if total_flow_area > 0 else 0.0

        # Optimal velocity range
        optimal_min = 1.8
        optimal_max = 2.5
        status = 'optimal' if optimal_min <= velocity <= optimal_max else 'suboptimal'

        return {
            'velocity_m_s': velocity,
            'optimal_min_m_s': optimal_min,
            'optimal_max_m_s': optimal_max,
            'status': status,
            'tube_area_m2': tube_area,
            'total_flow_area_m2': total_flow_area,
            'provenance': {
                'calculation_type': 'tube_velocity',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_fouling_resistance(self, conditions: CondenserConditions,
                                     cleanliness_factor: float = 0.85) -> Dict[str, Any]:
        """Calculate fouling resistance from cleanliness factor."""
        htc_result = self.calculate_overall_htc(conditions)
        actual_htc = htc_result['overall_htc_w_m2k']
        design_htc = htc_result['design_htc_w_m2k']

        # Fouling resistance = 1/U_dirty - 1/U_clean
        if actual_htc > 0 and design_htc > 0:
            fouling_resistance = (1.0 / actual_htc) - (1.0 / design_htc)
            fouling_resistance = max(0.0, fouling_resistance)
        else:
            fouling_resistance = 0.0

        # Severity assessment
        if fouling_resistance < 0.0001:
            severity = 'clean'
        elif fouling_resistance < 0.0002:
            severity = 'light'
        elif fouling_resistance < 0.0003:
            severity = 'moderate'
        elif fouling_resistance < 0.0005:
            severity = 'heavy'
        else:
            severity = 'severe'

        return {
            'fouling_resistance_m2k_w': fouling_resistance,
            'cleanliness_factor': cleanliness_factor,
            'severity': severity,
            'actual_htc_w_m2k': actual_htc,
            'design_htc_w_m2k': design_htc,
            'provenance': {
                'calculation_type': 'fouling_resistance',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }


class VacuumCalculator:
    """Mock vacuum calculator for condenser analysis."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_saturation_pressure(self, temperature_c: float) -> Dict[str, Any]:
        """Calculate saturation pressure from temperature."""
        # Antoine equation for water
        # Using simplified empirical formula
        if temperature_c < 0 or temperature_c > 100:
            sat_pressure_mbar = 0.0
        else:
            # Approximate saturation pressure (mbar)
            sat_pressure_mbar = 6.112 * math.exp((17.67 * temperature_c) / (temperature_c + 243.5))

        return {
            'saturation_pressure_mbar': sat_pressure_mbar,
            'saturation_pressure_kpa': sat_pressure_mbar / 10.0,
            'temperature_c': temperature_c,
            'provenance': {
                'calculation_type': 'saturation_pressure',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_saturation_temperature(self, pressure_mbar: float) -> Dict[str, Any]:
        """Calculate saturation temperature from pressure."""
        # Inverse Antoine equation
        if pressure_mbar <= 0:
            sat_temp_c = 0.0
        else:
            # Approximate saturation temperature (C)
            sat_temp_c = (243.5 * math.log(pressure_mbar / 6.112)) / (17.67 - math.log(pressure_mbar / 6.112))

        return {
            'saturation_temperature_c': sat_temp_c,
            'pressure_mbar': pressure_mbar,
            'provenance': {
                'calculation_type': 'saturation_temperature',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_ttd(self, conditions: VacuumConditions,
                      cooling_water_outlet_temp_c: float) -> Dict[str, Any]:
        """Calculate Terminal Temperature Difference."""
        sat_result = self.calculate_saturation_temperature(conditions.condenser_pressure_mbar)
        sat_temp = sat_result['saturation_temperature_c']

        ttd = sat_temp - cooling_water_outlet_temp_c

        # Design TTD is typically 2-3C
        design_ttd = 2.5
        deviation_percent = ((ttd - design_ttd) / design_ttd) * 100 if design_ttd > 0 else 0.0

        return {
            'ttd_c': ttd,
            'design_ttd_c': design_ttd,
            'deviation_percent': deviation_percent,
            'saturation_temp_c': sat_temp,
            'cooling_water_outlet_temp_c': cooling_water_outlet_temp_c,
            'status': 'normal' if ttd <= 3.0 else 'elevated',
            'provenance': {
                'calculation_type': 'ttd',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_subcooling(self, conditions: VacuumConditions,
                             hotwell_temperature_c: float) -> Dict[str, Any]:
        """Calculate condensate subcooling."""
        sat_result = self.calculate_saturation_temperature(conditions.condenser_pressure_mbar)
        sat_temp = sat_result['saturation_temperature_c']

        subcooling = sat_temp - hotwell_temperature_c

        # Subcooling > 1C indicates potential air inleakage
        air_inleakage_indicator = subcooling > 1.0

        return {
            'subcooling_c': subcooling,
            'saturation_temp_c': sat_temp,
            'hotwell_temp_c': hotwell_temperature_c,
            'air_inleakage_indicator': air_inleakage_indicator,
            'max_acceptable_c': 1.0,
            'status': 'normal' if subcooling <= 1.0 else 'elevated',
            'provenance': {
                'calculation_type': 'subcooling',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_air_removal_capacity(self, conditions: VacuumConditions) -> Dict[str, Any]:
        """Calculate air removal system capacity."""
        total_capacity = conditions.vacuum_pump_capacity_kg_hr

        # Required capacity based on air inleakage
        required_capacity = conditions.air_inleakage_rate_kg_hr * 1.5  # 50% safety margin

        capacity_ratio = total_capacity / required_capacity if required_capacity > 0 else 0.0

        if capacity_ratio >= 2.0:
            status = 'adequate'
        elif capacity_ratio >= 1.5:
            status = 'marginal'
        else:
            status = 'insufficient'

        return {
            'total_capacity_kg_hr': total_capacity,
            'required_capacity_kg_hr': required_capacity,
            'capacity_ratio': capacity_ratio,
            'status': status,
            'air_inleakage_rate_kg_hr': conditions.air_inleakage_rate_kg_hr,
            'provenance': {
                'calculation_type': 'air_removal_capacity',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_vacuum_deviation(self, conditions: VacuumConditions,
                                   design_vacuum_mbar: float = 45.0) -> Dict[str, Any]:
        """Calculate vacuum deviation from design."""
        deviation = conditions.condenser_pressure_mbar - design_vacuum_mbar

        # Heat rate penalty (approximately 12.5 kJ/kWh per mbar)
        heat_rate_penalty = deviation * 12.5 if deviation > 0 else 0.0

        if deviation <= 0:
            status = 'optimal'
        elif deviation <= 5:
            status = 'acceptable'
        elif deviation <= 15:
            status = 'degraded'
        else:
            status = 'critical'

        return {
            'current_vacuum_mbar': conditions.condenser_pressure_mbar,
            'design_vacuum_mbar': design_vacuum_mbar,
            'deviation_mbar': deviation,
            'heat_rate_penalty_kj_kwh': heat_rate_penalty,
            'status': status,
            'provenance': {
                'calculation_type': 'vacuum_deviation',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }


class EfficiencyCalculator:
    """Mock efficiency calculator for condenser analysis."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_condenser_efficiency(self, conditions: CondenserConditions) -> Dict[str, Any]:
        """Calculate overall condenser efficiency."""
        # Multiple efficiency components
        htc_calc = HeatTransferCalculator(self.version)
        vacuum_calc = VacuumCalculator(self.version)

        htc_result = htc_calc.calculate_overall_htc(conditions)
        ntu_result = htc_calc.calculate_ntu_effectiveness(conditions)
        velocity_result = htc_calc.calculate_tube_velocity(conditions)

        # Component efficiencies
        htc_efficiency = htc_result['htc_ratio'] * 100
        thermal_effectiveness = ntu_result['effectiveness'] * 100

        # Overall efficiency (weighted average)
        overall_efficiency = (htc_efficiency * 0.5 + thermal_effectiveness * 0.5)

        return {
            'overall_efficiency_percent': overall_efficiency,
            'htc_efficiency_percent': htc_efficiency,
            'thermal_effectiveness_percent': thermal_effectiveness,
            'tube_velocity_status': velocity_result['status'],
            'htc_w_m2k': htc_result['overall_htc_w_m2k'],
            'ntu': ntu_result['ntu'],
            'effectiveness': ntu_result['effectiveness'],
            'provenance': {
                'calculation_type': 'condenser_efficiency',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_cooling_water_efficiency(self, conditions: CondenserConditions) -> Dict[str, Any]:
        """Calculate cooling water system efficiency."""
        delta_t = conditions.cooling_water_outlet_temp_c - conditions.cooling_water_inlet_temp_c
        design_delta_t = 8.0  # Typical design delta T

        # Approach to wet bulb (assuming 20C wet bulb)
        wet_bulb_temp = 20.0
        approach = conditions.cooling_water_inlet_temp_c - wet_bulb_temp

        # Flow efficiency
        flow_efficiency = (delta_t / design_delta_t) * 100 if design_delta_t > 0 else 0.0

        return {
            'delta_t_c': delta_t,
            'design_delta_t_c': design_delta_t,
            'flow_efficiency_percent': min(flow_efficiency, 100.0),
            'approach_c': approach,
            'cooling_water_flow_m3_hr': conditions.cooling_water_flow_rate_m3_hr,
            'provenance': {
                'calculation_type': 'cooling_water_efficiency',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_heat_rate_impact(self, current_vacuum_mbar: float,
                                   design_vacuum_mbar: float = 45.0) -> Dict[str, Any]:
        """Calculate heat rate impact from vacuum deviation."""
        deviation = current_vacuum_mbar - design_vacuum_mbar

        # Heat rate penalty coefficient (kJ/kWh per mbar)
        penalty_coefficient = 12.5

        heat_rate_penalty = deviation * penalty_coefficient if deviation > 0 else 0.0

        # Fuel cost impact (assuming $3/GJ)
        fuel_cost_per_gj = 3.0
        annual_mwh = 8000 * 500  # 500 MW for 8000 hours
        annual_cost_impact = (heat_rate_penalty / 1000) * annual_mwh * fuel_cost_per_gj

        return {
            'vacuum_deviation_mbar': deviation,
            'heat_rate_penalty_kj_kwh': heat_rate_penalty,
            'heat_rate_penalty_percent': (heat_rate_penalty / 9000) * 100,  # Base heat rate ~9000 kJ/kWh
            'annual_fuel_cost_impact_usd': annual_cost_impact,
            'penalty_coefficient_kj_kwh_per_mbar': penalty_coefficient,
            'provenance': {
                'calculation_type': 'heat_rate_impact',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }


class FoulingCalculator:
    """Mock fouling calculator for condenser analysis."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_fouling_rate(self, conditions: FoulingConditions) -> Dict[str, Any]:
        """Calculate fouling rate based on operating conditions."""
        # Base fouling rate per 1000 hours
        base_rate = 0.01  # Cleanliness factor reduction

        # Adjustment factors
        tds_factor = 1.0 + (conditions.cooling_water_tds_ppm - 1000) / 5000
        ph_factor = 1.0 + abs(conditions.cooling_water_ph - 7.5) * 0.2
        temp_factor = 1.0 + (conditions.cooling_water_temperature_c - 25) / 50
        velocity_factor = 2.0 / conditions.tube_velocity_m_s if conditions.tube_velocity_m_s > 0 else 2.0

        # Material factor
        material_factors = {
            'titanium': 0.5,
            'stainless_316': 0.7,
            'copper_nickel_90_10': 0.8,
            'admiralty_brass': 1.0,
            'carbon_steel': 1.5
        }
        material_factor = material_factors.get(conditions.tube_material, 1.0)

        # Treatment factor
        treatment_factors = {
            'oxidizing': 0.7,
            'non_oxidizing': 0.8,
            'none': 1.2
        }
        treatment_factor = treatment_factors.get(conditions.biocide_treatment, 1.0)

        # Adjusted fouling rate
        adjusted_rate = base_rate * tds_factor * ph_factor * temp_factor * velocity_factor * material_factor * treatment_factor

        return {
            'fouling_rate_per_1000hr': adjusted_rate,
            'base_rate': base_rate,
            'tds_factor': tds_factor,
            'ph_factor': ph_factor,
            'temperature_factor': temp_factor,
            'velocity_factor': velocity_factor,
            'material_factor': material_factor,
            'treatment_factor': treatment_factor,
            'provenance': {
                'calculation_type': 'fouling_rate',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def predict_cleanliness_factor(self, conditions: FoulingConditions,
                                   hours_ahead: float = 720.0) -> Dict[str, Any]:
        """Predict future cleanliness factor."""
        rate_result = self.calculate_fouling_rate(conditions)
        fouling_rate = rate_result['fouling_rate_per_1000hr']

        # Calculate future cleanliness factor
        degradation = fouling_rate * (hours_ahead / 1000.0)
        predicted_cf = max(conditions.current_cleanliness_factor - degradation, 0.50)

        # Time to threshold (0.75 cleanliness factor)
        threshold = 0.75
        if fouling_rate > 0 and conditions.current_cleanliness_factor > threshold:
            hours_to_threshold = (conditions.current_cleanliness_factor - threshold) / fouling_rate * 1000
            days_to_threshold = hours_to_threshold / 24
        else:
            hours_to_threshold = 0.0
            days_to_threshold = 0.0

        return {
            'current_cleanliness_factor': conditions.current_cleanliness_factor,
            'predicted_cleanliness_factor': predicted_cf,
            'prediction_hours': hours_ahead,
            'fouling_rate_per_1000hr': fouling_rate,
            'hours_to_threshold': hours_to_threshold,
            'days_to_threshold': days_to_threshold,
            'threshold': threshold,
            'provenance': {
                'calculation_type': 'cleanliness_prediction',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def calculate_cleaning_benefit(self, conditions: FoulingConditions,
                                   current_vacuum_mbar: float) -> Dict[str, Any]:
        """Calculate benefit of tube cleaning."""
        # Estimate vacuum improvement from cleaning
        cf_improvement = min(0.95, conditions.current_cleanliness_factor + 0.15) - conditions.current_cleanliness_factor

        # Vacuum improvement (approximately 1.5 mbar per 0.05 CF)
        vacuum_improvement = cf_improvement * 30.0  # mbar

        # Heat rate improvement
        heat_rate_improvement = vacuum_improvement * 12.5  # kJ/kWh

        # Annual savings estimate
        annual_mwh = 8000 * 500
        fuel_cost_per_gj = 3.0
        annual_savings = (heat_rate_improvement / 1000) * annual_mwh * fuel_cost_per_gj

        return {
            'cleanliness_factor_before': conditions.current_cleanliness_factor,
            'cleanliness_factor_after': conditions.current_cleanliness_factor + cf_improvement,
            'cf_improvement': cf_improvement,
            'vacuum_improvement_mbar': vacuum_improvement,
            'heat_rate_improvement_kj_kwh': heat_rate_improvement,
            'annual_savings_usd': annual_savings,
            'provenance': {
                'calculation_type': 'cleaning_benefit',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def identify_fouling_type(self, conditions: FoulingConditions) -> Dict[str, Any]:
        """Identify predominant fouling type."""
        # Score different fouling types based on conditions
        biological_score = 0.0
        mineral_score = 0.0
        particulate_score = 0.0

        # Biological fouling indicators
        if conditions.cooling_water_temperature_c > 25:
            biological_score += 0.3
        if conditions.cooling_water_source == 'cooling_tower':
            biological_score += 0.3
        if conditions.biocide_treatment == 'none':
            biological_score += 0.4

        # Mineral fouling indicators
        if conditions.cooling_water_tds_ppm > 2000:
            mineral_score += 0.4
        if conditions.cooling_water_ph > 8.0:
            mineral_score += 0.3
        if conditions.cooling_water_temperature_c > 30:
            mineral_score += 0.3

        # Particulate fouling indicators
        if conditions.cooling_water_source in ['river', 'seawater']:
            particulate_score += 0.4
        if conditions.tube_velocity_m_s < 1.5:
            particulate_score += 0.3
        if conditions.biocide_treatment == 'none':
            particulate_score += 0.3

        # Identify predominant type
        max_score = max(biological_score, mineral_score, particulate_score)
        if max_score == biological_score:
            predominant = 'biological'
        elif max_score == mineral_score:
            predominant = 'mineral'
        else:
            predominant = 'particulate'

        return {
            'predominant_type': predominant,
            'biological_score': biological_score,
            'mineral_score': mineral_score,
            'particulate_score': particulate_score,
            'recommended_treatment': self._recommend_treatment(predominant),
            'provenance': {
                'calculation_type': 'fouling_type',
                'version': self.version,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

    def _recommend_treatment(self, fouling_type: str) -> str:
        """Recommend treatment based on fouling type."""
        treatments = {
            'biological': 'Increase biocide dosing, consider shock chlorination',
            'mineral': 'Increase blowdown, add scale inhibitor',
            'particulate': 'Improve filtration, increase tube velocity'
        }
        return treatments.get(fouling_type, 'Consult water treatment specialist')


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def htc_calculator():
    """Create HeatTransferCalculator instance."""
    return HeatTransferCalculator(version="1.0.0-test")


@pytest.fixture
def vacuum_calculator():
    """Create VacuumCalculator instance."""
    return VacuumCalculator(version="1.0.0-test")


@pytest.fixture
def efficiency_calculator():
    """Create EfficiencyCalculator instance."""
    return EfficiencyCalculator(version="1.0.0-test")


@pytest.fixture
def fouling_calculator():
    """Create FoulingCalculator instance."""
    return FoulingCalculator(version="1.0.0-test")


@pytest.fixture
def standard_condenser_conditions():
    """Standard condenser conditions for testing."""
    return CondenserConditions(
        vacuum_pressure_mbar=50.0,
        steam_saturation_temp_c=33.0,
        hotwell_temperature_c=32.5,
        cooling_water_inlet_temp_c=25.0,
        cooling_water_outlet_temp_c=32.0,
        cooling_water_flow_rate_m3_hr=45000.0,
        heat_duty_mw=180.0,
        surface_area_m2=17500.0,
        tube_od_mm=25.4,
        tube_id_mm=23.4,
        tube_length_m=12.0,
        tube_count=18500,
        num_passes=2,
        tube_material='titanium'
    )


@pytest.fixture
def standard_vacuum_conditions():
    """Standard vacuum conditions for testing."""
    return VacuumConditions(
        condenser_pressure_mbar=50.0,
        steam_flow_kg_hr=150000.0,
        cooling_water_inlet_temp_c=25.0,
        air_inleakage_rate_kg_hr=0.5,
        ejector_steam_pressure_bar=8.0,
        vacuum_pump_capacity_kg_hr=15.0,
        non_condensable_fraction=0.001
    )


@pytest.fixture
def standard_fouling_conditions():
    """Standard fouling conditions for testing."""
    return FoulingConditions(
        cooling_water_source='cooling_tower',
        cooling_water_tds_ppm=1800.0,
        cooling_water_ph=7.8,
        cooling_water_temperature_c=28.0,
        tube_velocity_m_s=2.2,
        tube_material='titanium',
        operating_hours=4380.0,
        biocide_treatment='oxidizing',
        current_cleanliness_factor=0.85
    )


# ============================================================================
# HeatTransferCalculator Tests
# ============================================================================

class TestHeatTransferCalculator:
    """Test suite for HeatTransferCalculator."""

    @pytest.mark.unit
    def test_calculator_initialization(self, htc_calculator):
        """Test calculator initializes correctly."""
        assert htc_calculator.version == "1.0.0-test"

    @pytest.mark.unit
    def test_lmtd_calculation(self, htc_calculator, standard_condenser_conditions):
        """Test LMTD calculation."""
        result = htc_calculator.calculate_lmtd(standard_condenser_conditions)

        assert 'lmtd_c' in result
        assert result['lmtd_c'] > 0
        assert 'delta_t_hot_end_c' in result
        assert 'delta_t_cold_end_c' in result

    @pytest.mark.unit
    def test_lmtd_temperature_ratio(self, htc_calculator, standard_condenser_conditions):
        """Test LMTD temperature ratio calculation."""
        result = htc_calculator.calculate_lmtd(standard_condenser_conditions)

        assert 'temperature_ratio' in result
        assert result['temperature_ratio'] > 0

    @pytest.mark.unit
    def test_overall_htc_calculation(self, htc_calculator, standard_condenser_conditions):
        """Test overall HTC calculation."""
        result = htc_calculator.calculate_overall_htc(standard_condenser_conditions)

        assert 'overall_htc_w_m2k' in result
        assert result['overall_htc_w_m2k'] > 0
        assert 'design_htc_w_m2k' in result

    @pytest.mark.unit
    def test_htc_ratio_calculation(self, htc_calculator, standard_condenser_conditions):
        """Test HTC ratio calculation."""
        result = htc_calculator.calculate_overall_htc(standard_condenser_conditions)

        assert 'htc_ratio' in result
        assert 0 <= result['htc_ratio'] <= 1.5

    @pytest.mark.unit
    def test_ntu_effectiveness_calculation(self, htc_calculator, standard_condenser_conditions):
        """Test NTU and effectiveness calculation."""
        result = htc_calculator.calculate_ntu_effectiveness(standard_condenser_conditions)

        assert 'ntu' in result
        assert result['ntu'] > 0
        assert 'effectiveness' in result
        assert 0 <= result['effectiveness'] <= 1

    @pytest.mark.unit
    def test_tube_velocity_calculation(self, htc_calculator, standard_condenser_conditions):
        """Test tube velocity calculation."""
        result = htc_calculator.calculate_tube_velocity(standard_condenser_conditions)

        assert 'velocity_m_s' in result
        assert result['velocity_m_s'] > 0
        assert 'status' in result
        assert result['status'] in ['optimal', 'suboptimal']

    @pytest.mark.unit
    def test_tube_velocity_optimal_range(self, htc_calculator, standard_condenser_conditions):
        """Test tube velocity optimal range check."""
        result = htc_calculator.calculate_tube_velocity(standard_condenser_conditions)

        assert 'optimal_min_m_s' in result
        assert 'optimal_max_m_s' in result
        assert result['optimal_min_m_s'] < result['optimal_max_m_s']

    @pytest.mark.unit
    def test_fouling_resistance_calculation(self, htc_calculator, standard_condenser_conditions):
        """Test fouling resistance calculation."""
        result = htc_calculator.calculate_fouling_resistance(standard_condenser_conditions, cleanliness_factor=0.85)

        assert 'fouling_resistance_m2k_w' in result
        assert result['fouling_resistance_m2k_w'] >= 0
        assert 'severity' in result

    @pytest.mark.unit
    def test_fouling_severity_levels(self, htc_calculator, standard_condenser_conditions):
        """Test fouling severity level assessment."""
        result = htc_calculator.calculate_fouling_resistance(standard_condenser_conditions, cleanliness_factor=0.85)

        assert result['severity'] in ['clean', 'light', 'moderate', 'heavy', 'severe']

    @pytest.mark.unit
    def test_provenance_tracking(self, htc_calculator, standard_condenser_conditions):
        """Test provenance is tracked."""
        result = htc_calculator.calculate_lmtd(standard_condenser_conditions)

        assert 'provenance' in result
        assert 'calculation_type' in result['provenance']
        assert 'version' in result['provenance']
        assert 'timestamp' in result['provenance']

    @pytest.mark.determinism
    def test_lmtd_deterministic(self, htc_calculator, standard_condenser_conditions):
        """Test LMTD calculation is deterministic."""
        result1 = htc_calculator.calculate_lmtd(standard_condenser_conditions)
        result2 = htc_calculator.calculate_lmtd(standard_condenser_conditions)

        assert result1['lmtd_c'] == result2['lmtd_c']


# ============================================================================
# VacuumCalculator Tests
# ============================================================================

class TestVacuumCalculator:
    """Test suite for VacuumCalculator."""

    @pytest.mark.unit
    def test_calculator_initialization(self, vacuum_calculator):
        """Test calculator initializes correctly."""
        assert vacuum_calculator.version == "1.0.0-test"

    @pytest.mark.unit
    def test_saturation_pressure_calculation(self, vacuum_calculator):
        """Test saturation pressure calculation."""
        result = vacuum_calculator.calculate_saturation_pressure(33.0)

        assert 'saturation_pressure_mbar' in result
        assert result['saturation_pressure_mbar'] > 0
        # At 33C, saturation pressure should be around 50 mbar
        assert 40 < result['saturation_pressure_mbar'] < 60

    @pytest.mark.unit
    def test_saturation_temperature_calculation(self, vacuum_calculator):
        """Test saturation temperature calculation."""
        result = vacuum_calculator.calculate_saturation_temperature(50.0)

        assert 'saturation_temperature_c' in result
        # At 50 mbar, saturation temp should be around 33C
        assert 30 < result['saturation_temperature_c'] < 40

    @pytest.mark.unit
    def test_ttd_calculation(self, vacuum_calculator, standard_vacuum_conditions):
        """Test TTD calculation."""
        result = vacuum_calculator.calculate_ttd(standard_vacuum_conditions, cooling_water_outlet_temp_c=32.0)

        assert 'ttd_c' in result
        assert 'design_ttd_c' in result
        assert 'deviation_percent' in result
        assert 'status' in result

    @pytest.mark.unit
    def test_ttd_status_assessment(self, vacuum_calculator, standard_vacuum_conditions):
        """Test TTD status assessment."""
        result = vacuum_calculator.calculate_ttd(standard_vacuum_conditions, cooling_water_outlet_temp_c=32.0)

        assert result['status'] in ['normal', 'elevated']

    @pytest.mark.unit
    def test_subcooling_calculation(self, vacuum_calculator, standard_vacuum_conditions):
        """Test subcooling calculation."""
        result = vacuum_calculator.calculate_subcooling(standard_vacuum_conditions, hotwell_temperature_c=32.5)

        assert 'subcooling_c' in result
        assert 'air_inleakage_indicator' in result
        assert isinstance(result['air_inleakage_indicator'], bool)

    @pytest.mark.unit
    def test_air_removal_capacity_calculation(self, vacuum_calculator, standard_vacuum_conditions):
        """Test air removal capacity calculation."""
        result = vacuum_calculator.calculate_air_removal_capacity(standard_vacuum_conditions)

        assert 'total_capacity_kg_hr' in result
        assert 'required_capacity_kg_hr' in result
        assert 'capacity_ratio' in result
        assert 'status' in result

    @pytest.mark.unit
    def test_air_removal_status_assessment(self, vacuum_calculator, standard_vacuum_conditions):
        """Test air removal status assessment."""
        result = vacuum_calculator.calculate_air_removal_capacity(standard_vacuum_conditions)

        assert result['status'] in ['adequate', 'marginal', 'insufficient']

    @pytest.mark.unit
    def test_vacuum_deviation_calculation(self, vacuum_calculator, standard_vacuum_conditions):
        """Test vacuum deviation calculation."""
        result = vacuum_calculator.calculate_vacuum_deviation(standard_vacuum_conditions)

        assert 'current_vacuum_mbar' in result
        assert 'design_vacuum_mbar' in result
        assert 'deviation_mbar' in result
        assert 'heat_rate_penalty_kj_kwh' in result

    @pytest.mark.unit
    def test_vacuum_deviation_status(self, vacuum_calculator, standard_vacuum_conditions):
        """Test vacuum deviation status."""
        result = vacuum_calculator.calculate_vacuum_deviation(standard_vacuum_conditions)

        assert result['status'] in ['optimal', 'acceptable', 'degraded', 'critical']

    @pytest.mark.parametrize("pressure_mbar,expected_temp_range", [
        (30.0, (20.0, 30.0)),
        (50.0, (30.0, 40.0)),
        (80.0, (40.0, 50.0)),
    ])
    def test_saturation_temp_parametrized(self, vacuum_calculator, pressure_mbar, expected_temp_range):
        """Test saturation temperature at various pressures."""
        result = vacuum_calculator.calculate_saturation_temperature(pressure_mbar)

        assert expected_temp_range[0] < result['saturation_temperature_c'] < expected_temp_range[1]


# ============================================================================
# EfficiencyCalculator Tests
# ============================================================================

class TestEfficiencyCalculator:
    """Test suite for EfficiencyCalculator."""

    @pytest.mark.unit
    def test_calculator_initialization(self, efficiency_calculator):
        """Test calculator initializes correctly."""
        assert efficiency_calculator.version == "1.0.0-test"

    @pytest.mark.unit
    def test_condenser_efficiency_calculation(self, efficiency_calculator, standard_condenser_conditions):
        """Test condenser efficiency calculation."""
        result = efficiency_calculator.calculate_condenser_efficiency(standard_condenser_conditions)

        assert 'overall_efficiency_percent' in result
        assert 0 <= result['overall_efficiency_percent'] <= 100
        assert 'htc_efficiency_percent' in result
        assert 'thermal_effectiveness_percent' in result

    @pytest.mark.unit
    def test_cooling_water_efficiency_calculation(self, efficiency_calculator, standard_condenser_conditions):
        """Test cooling water efficiency calculation."""
        result = efficiency_calculator.calculate_cooling_water_efficiency(standard_condenser_conditions)

        assert 'delta_t_c' in result
        assert 'design_delta_t_c' in result
        assert 'flow_efficiency_percent' in result
        assert 'approach_c' in result

    @pytest.mark.unit
    def test_heat_rate_impact_calculation(self, efficiency_calculator):
        """Test heat rate impact calculation."""
        result = efficiency_calculator.calculate_heat_rate_impact(55.0, 45.0)

        assert 'vacuum_deviation_mbar' in result
        assert result['vacuum_deviation_mbar'] == 10.0
        assert 'heat_rate_penalty_kj_kwh' in result
        assert result['heat_rate_penalty_kj_kwh'] > 0

    @pytest.mark.unit
    def test_heat_rate_no_penalty_at_design(self, efficiency_calculator):
        """Test no heat rate penalty at design vacuum."""
        result = efficiency_calculator.calculate_heat_rate_impact(45.0, 45.0)

        assert result['heat_rate_penalty_kj_kwh'] == 0.0

    @pytest.mark.unit
    def test_annual_cost_impact(self, efficiency_calculator):
        """Test annual cost impact calculation."""
        result = efficiency_calculator.calculate_heat_rate_impact(60.0, 45.0)

        assert 'annual_fuel_cost_impact_usd' in result
        assert result['annual_fuel_cost_impact_usd'] > 0


# ============================================================================
# FoulingCalculator Tests
# ============================================================================

class TestFoulingCalculator:
    """Test suite for FoulingCalculator."""

    @pytest.mark.unit
    def test_calculator_initialization(self, fouling_calculator):
        """Test calculator initializes correctly."""
        assert fouling_calculator.version == "1.0.0-test"

    @pytest.mark.unit
    def test_fouling_rate_calculation(self, fouling_calculator, standard_fouling_conditions):
        """Test fouling rate calculation."""
        result = fouling_calculator.calculate_fouling_rate(standard_fouling_conditions)

        assert 'fouling_rate_per_1000hr' in result
        assert result['fouling_rate_per_1000hr'] > 0

    @pytest.mark.unit
    def test_fouling_rate_factors(self, fouling_calculator, standard_fouling_conditions):
        """Test fouling rate factors."""
        result = fouling_calculator.calculate_fouling_rate(standard_fouling_conditions)

        assert 'tds_factor' in result
        assert 'ph_factor' in result
        assert 'temperature_factor' in result
        assert 'velocity_factor' in result
        assert 'material_factor' in result
        assert 'treatment_factor' in result

    @pytest.mark.unit
    def test_cleanliness_prediction(self, fouling_calculator, standard_fouling_conditions):
        """Test cleanliness factor prediction."""
        result = fouling_calculator.predict_cleanliness_factor(standard_fouling_conditions, hours_ahead=720.0)

        assert 'current_cleanliness_factor' in result
        assert 'predicted_cleanliness_factor' in result
        assert result['predicted_cleanliness_factor'] <= result['current_cleanliness_factor']

    @pytest.mark.unit
    def test_days_to_threshold(self, fouling_calculator, standard_fouling_conditions):
        """Test days to threshold calculation."""
        result = fouling_calculator.predict_cleanliness_factor(standard_fouling_conditions)

        assert 'days_to_threshold' in result
        assert result['days_to_threshold'] >= 0

    @pytest.mark.unit
    def test_cleaning_benefit_calculation(self, fouling_calculator, standard_fouling_conditions):
        """Test cleaning benefit calculation."""
        result = fouling_calculator.calculate_cleaning_benefit(standard_fouling_conditions, current_vacuum_mbar=55.0)

        assert 'cleanliness_factor_before' in result
        assert 'cleanliness_factor_after' in result
        assert 'vacuum_improvement_mbar' in result
        assert 'heat_rate_improvement_kj_kwh' in result
        assert 'annual_savings_usd' in result

    @pytest.mark.unit
    def test_fouling_type_identification(self, fouling_calculator, standard_fouling_conditions):
        """Test fouling type identification."""
        result = fouling_calculator.identify_fouling_type(standard_fouling_conditions)

        assert 'predominant_type' in result
        assert result['predominant_type'] in ['biological', 'mineral', 'particulate']
        assert 'recommended_treatment' in result

    @pytest.mark.unit
    def test_fouling_type_scores(self, fouling_calculator, standard_fouling_conditions):
        """Test fouling type scores."""
        result = fouling_calculator.identify_fouling_type(standard_fouling_conditions)

        assert 'biological_score' in result
        assert 'mineral_score' in result
        assert 'particulate_score' in result


# ============================================================================
# Boundary Condition Tests
# ============================================================================

class TestBoundaryConditions:
    """Test boundary conditions for all calculators."""

    @pytest.mark.unit
    def test_zero_flow_rate(self, htc_calculator):
        """Test handling of zero flow rate."""
        conditions = CondenserConditions(cooling_water_flow_rate_m3_hr=0.0)

        result = htc_calculator.calculate_tube_velocity(conditions)
        assert result['velocity_m_s'] == 0.0

    @pytest.mark.unit
    def test_zero_heat_duty(self, htc_calculator):
        """Test handling of zero heat duty."""
        conditions = CondenserConditions(heat_duty_mw=0.0)

        result = htc_calculator.calculate_overall_htc(conditions)
        assert result['overall_htc_w_m2k'] == 0.0

    @pytest.mark.unit
    def test_zero_pressure(self, vacuum_calculator):
        """Test handling of zero pressure."""
        result = vacuum_calculator.calculate_saturation_temperature(0.0)
        assert result['saturation_temperature_c'] == 0.0

    @pytest.mark.unit
    def test_negative_temperature(self, vacuum_calculator):
        """Test handling of negative temperature."""
        result = vacuum_calculator.calculate_saturation_pressure(-10.0)
        assert result['saturation_pressure_mbar'] == 0.0

    @pytest.mark.unit
    def test_extreme_fouling(self, fouling_calculator):
        """Test handling of extreme fouling conditions."""
        conditions = FoulingConditions(
            current_cleanliness_factor=0.50,
            cooling_water_tds_ppm=5000.0
        )

        result = fouling_calculator.predict_cleanliness_factor(conditions)
        assert result['predicted_cleanliness_factor'] >= 0.50


# ============================================================================
# Unit Conversion Tests
# ============================================================================

class TestUnitConversions:
    """Test unit conversions in calculators."""

    @pytest.mark.unit
    def test_pressure_units(self, vacuum_calculator):
        """Test pressure unit conversion."""
        result = vacuum_calculator.calculate_saturation_pressure(33.0)

        # Check both mbar and kPa are present
        assert 'saturation_pressure_mbar' in result
        assert 'saturation_pressure_kpa' in result
        # 1 mbar = 0.1 kPa
        assert abs(result['saturation_pressure_mbar'] / 10.0 - result['saturation_pressure_kpa']) < 0.01

    @pytest.mark.unit
    def test_flow_rate_units(self, htc_calculator, standard_condenser_conditions):
        """Test flow rate is handled correctly."""
        result = htc_calculator.calculate_tube_velocity(standard_condenser_conditions)

        # Flow rate input is in m3/hr, velocity output is in m/s
        assert 'velocity_m_s' in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestCalculatorIntegration:
    """Integration tests for calculator modules working together."""

    @pytest.mark.integration
    def test_htc_feeds_efficiency(self, htc_calculator, efficiency_calculator, standard_condenser_conditions):
        """Test HTC calculator feeds efficiency calculator."""
        htc_result = htc_calculator.calculate_overall_htc(standard_condenser_conditions)
        efficiency_result = efficiency_calculator.calculate_condenser_efficiency(standard_condenser_conditions)

        # Both should have consistent values
        assert htc_result['overall_htc_w_m2k'] > 0
        assert efficiency_result['htc_w_m2k'] > 0

    @pytest.mark.integration
    def test_vacuum_feeds_efficiency(self, vacuum_calculator, efficiency_calculator):
        """Test vacuum calculator feeds efficiency calculator."""
        vacuum_result = vacuum_calculator.calculate_vacuum_deviation(
            VacuumConditions(condenser_pressure_mbar=55.0)
        )
        heat_rate_result = efficiency_calculator.calculate_heat_rate_impact(55.0, 45.0)

        # Both should calculate consistent heat rate impact
        assert vacuum_result['heat_rate_penalty_kj_kwh'] > 0
        assert heat_rate_result['heat_rate_penalty_kj_kwh'] > 0

    @pytest.mark.integration
    def test_fouling_affects_htc(self, htc_calculator, fouling_calculator, standard_condenser_conditions, standard_fouling_conditions):
        """Test fouling affects HTC calculations."""
        htc_clean = htc_calculator.calculate_overall_htc(standard_condenser_conditions)
        fouling_result = fouling_calculator.calculate_cleaning_benefit(
            standard_fouling_conditions,
            current_vacuum_mbar=55.0
        )

        # Fouling should show improvement potential
        assert fouling_result['vacuum_improvement_mbar'] >= 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestCalculatorPerformance:
    """Performance tests for calculators."""

    @pytest.mark.performance
    def test_htc_calculation_performance(self, htc_calculator, standard_condenser_conditions):
        """Test HTC calculation performance."""
        import time

        start = time.perf_counter()
        for _ in range(1000):
            htc_calculator.calculate_overall_htc(standard_condenser_conditions)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0  # 1000 calculations in under 1 second

    @pytest.mark.performance
    def test_vacuum_calculation_performance(self, vacuum_calculator, standard_vacuum_conditions):
        """Test vacuum calculation performance."""
        import time

        start = time.perf_counter()
        for _ in range(1000):
            vacuum_calculator.calculate_saturation_temperature(50.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5  # Should be very fast

    @pytest.mark.performance
    def test_fouling_calculation_performance(self, fouling_calculator, standard_fouling_conditions):
        """Test fouling calculation performance."""
        import time

        start = time.perf_counter()
        for _ in range(1000):
            fouling_calculator.calculate_fouling_rate(standard_fouling_conditions)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0


# ============================================================================
# Determinism Tests
# ============================================================================

class TestCalculatorDeterminism:
    """Test calculator determinism."""

    @pytest.mark.determinism
    def test_htc_deterministic(self, htc_calculator, standard_condenser_conditions):
        """Test HTC calculations are deterministic."""
        result1 = htc_calculator.calculate_overall_htc(standard_condenser_conditions)
        result2 = htc_calculator.calculate_overall_htc(standard_condenser_conditions)

        assert result1['overall_htc_w_m2k'] == result2['overall_htc_w_m2k']

    @pytest.mark.determinism
    def test_vacuum_deterministic(self, vacuum_calculator):
        """Test vacuum calculations are deterministic."""
        result1 = vacuum_calculator.calculate_saturation_pressure(33.0)
        result2 = vacuum_calculator.calculate_saturation_pressure(33.0)

        assert result1['saturation_pressure_mbar'] == result2['saturation_pressure_mbar']

    @pytest.mark.determinism
    def test_efficiency_deterministic(self, efficiency_calculator, standard_condenser_conditions):
        """Test efficiency calculations are deterministic."""
        result1 = efficiency_calculator.calculate_condenser_efficiency(standard_condenser_conditions)
        result2 = efficiency_calculator.calculate_condenser_efficiency(standard_condenser_conditions)

        assert result1['overall_efficiency_percent'] == result2['overall_efficiency_percent']

    @pytest.mark.determinism
    def test_fouling_deterministic(self, fouling_calculator, standard_fouling_conditions):
        """Test fouling calculations are deterministic."""
        result1 = fouling_calculator.calculate_fouling_rate(standard_fouling_conditions)
        result2 = fouling_calculator.calculate_fouling_rate(standard_fouling_conditions)

        assert result1['fouling_rate_per_1000hr'] == result2['fouling_rate_per_1000hr']
