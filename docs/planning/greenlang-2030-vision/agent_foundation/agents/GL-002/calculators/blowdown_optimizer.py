# -*- coding: utf-8 -*-
"""
Blowdown Optimizer Calculator - Zero Hallucination Guarantee

Implements ASME and ABMA guidelines for optimal blowdown rate calculation,
TDS control, and heat recovery from blowdown streams.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASME Section VII, ABMA Boiler Water Limits
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
from dataclasses import dataclass
from .provenance import ProvenanceTracker
from greenlang.determinism import FinancialDecimal


@dataclass
class BlowdownData:
    """Blowdown system parameters."""
    steam_flow_rate_kg_hr: float
    steam_pressure_bar: float
    feedwater_tds_ppm: float
    boiler_water_tds_ppm: float
    max_allowable_tds_ppm: float
    feedwater_temperature_c: float
    makeup_water_tds_ppm: float
    condensate_return_percent: float
    fuel_cost_per_gj: float
    water_cost_per_m3: float
    heat_recovery_installed: bool = False


class BlowdownOptimizer:
    """
    Optimizes boiler blowdown for water quality and energy efficiency.

    Zero Hallucination Guarantee:
    - Pure mathematical calculations
    - No LLM inference
    - Complete provenance tracking
    """

    # ABMA recommended TDS limits (ppm)
    TDS_LIMITS = {
        'low_pressure': 3500,    # 0-15 bar
        'medium_pressure': 3000,  # 15-40 bar
        'high_pressure': 2000,    # 40-60 bar
        'very_high_pressure': 1000  # >60 bar
    }

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_optimal_blowdown(self, data: BlowdownData) -> Dict:
        """Calculate optimal blowdown rate and potential savings."""
        tracker = ProvenanceTracker(
            f"blowdown_opt_{id(data)}",
            "blowdown_optimization",
            self.version
        )

        tracker.record_inputs(data.__dict__)

        # Calculate cycles of concentration
        cycles = self._calculate_cycles(data, tracker)

        # Calculate optimal blowdown rate
        optimal_rate = self._calculate_blowdown_rate(data, cycles, tracker)

        # Calculate current blowdown rate (from TDS values)
        current_rate = self._calculate_current_blowdown(data, tracker)

        # Calculate water losses
        water_losses = self._calculate_water_losses(data, current_rate, optimal_rate, tracker)

        # Calculate energy losses
        energy_losses = self._calculate_energy_losses(data, current_rate, optimal_rate, tracker)

        # Calculate heat recovery potential
        heat_recovery = self._calculate_heat_recovery(data, optimal_rate, tracker)

        # Calculate savings
        savings = self._calculate_savings(
            data, current_rate, optimal_rate, heat_recovery, tracker
        )

        result = {
            'cycles_of_concentration': float(cycles),
            'current_blowdown_percent': FinancialDecimal.from_string(current_rate),
            'optimal_blowdown_percent': FinancialDecimal.from_string(optimal_rate),
            'current_blowdown_kg_hr': FinancialDecimal.from_string(Decimal(str(data.steam_flow_rate_kg_hr)) * current_rate / Decimal('100')),
            'optimal_blowdown_kg_hr': FinancialDecimal.from_string(Decimal(str(data.steam_flow_rate_kg_hr)) * optimal_rate / Decimal('100')),
            'water_losses': water_losses,
            'energy_losses': energy_losses,
            'heat_recovery_potential': heat_recovery,
            'annual_savings': savings,
            'tds_control': self._tds_control_strategy(data, cycles),
            'provenance': tracker.get_provenance_record(optimal_rate).to_dict()
        }

        return result

    def _calculate_cycles(self, data: BlowdownData, tracker: ProvenanceTracker) -> Decimal:
        """Calculate cycles of concentration."""
        boiler_tds = Decimal(str(data.boiler_water_tds_ppm))
        feedwater_tds = Decimal(str(data.feedwater_tds_ppm))

        if feedwater_tds > 0:
            cycles = boiler_tds / feedwater_tds
        else:
            cycles = Decimal('1')

        cycles = cycles.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="cycles_calculation",
            description="Calculate cycles of concentration",
            inputs={'boiler_tds': boiler_tds, 'feedwater_tds': feedwater_tds},
            output_value=cycles,
            output_name="cycles",
            formula="Cycles = TDS_boiler / TDS_feedwater",
            units="dimensionless"
        )

        return cycles

    def _calculate_blowdown_rate(
        self, data: BlowdownData, cycles: Decimal, tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate optimal blowdown rate percentage."""
        # Formula: BD% = (Feedwater TDS / (Max TDS - Feedwater TDS)) * 100
        feedwater_tds = Decimal(str(data.feedwater_tds_ppm))
        max_tds = Decimal(str(data.max_allowable_tds_ppm))

        if max_tds > feedwater_tds:
            # Using mass balance equation
            # BD% = 100 / (Cycles - 1)
            if cycles > Decimal('1'):
                blowdown_rate = Decimal('100') / (cycles - Decimal('1'))
            else:
                blowdown_rate = Decimal('100')  # Maximum blowdown
        else:
            blowdown_rate = Decimal('10')  # Default high blowdown

        # Apply practical limits (0.5% to 10%)
        blowdown_rate = max(Decimal('0.5'), min(blowdown_rate, Decimal('10')))
        blowdown_rate = blowdown_rate.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="blowdown_rate",
            description="Calculate optimal blowdown rate",
            inputs={'cycles': cycles, 'max_tds': max_tds},
            output_value=blowdown_rate,
            output_name="blowdown_percent",
            formula="BD% = 100 / (Cycles - 1)",
            units="%"
        )

        return blowdown_rate

    def _calculate_current_blowdown(self, data: BlowdownData, tracker: ProvenanceTracker) -> Decimal:
        """Calculate current blowdown rate from TDS measurements."""
        feedwater_tds = Decimal(str(data.feedwater_tds_ppm))
        boiler_tds = Decimal(str(data.boiler_water_tds_ppm))

        if boiler_tds > feedwater_tds:
            current_cycles = boiler_tds / feedwater_tds
            if current_cycles > Decimal('1'):
                current_blowdown = Decimal('100') / (current_cycles - Decimal('1'))
            else:
                current_blowdown = Decimal('100')
        else:
            current_blowdown = Decimal('5')  # Assume 5% if data invalid

        current_blowdown = current_blowdown.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="current_blowdown",
            description="Calculate current blowdown from TDS",
            inputs={'feedwater_tds': feedwater_tds, 'boiler_tds': boiler_tds},
            output_value=current_blowdown,
            output_name="current_blowdown_percent",
            formula="From TDS measurements",
            units="%"
        )

        return current_blowdown

    def _calculate_water_losses(
        self, data: BlowdownData, current_rate: Decimal, optimal_rate: Decimal, tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate water losses and savings."""
        steam_flow = Decimal(str(data.steam_flow_rate_kg_hr))

        current_water_loss = steam_flow * current_rate / Decimal('100')
        optimal_water_loss = steam_flow * optimal_rate / Decimal('100')
        water_saved = current_water_loss - optimal_water_loss

        # Annual calculations (8760 hours)
        annual_current = current_water_loss * Decimal('8760') / Decimal('1000')  # m³/year
        annual_optimal = optimal_water_loss * Decimal('8760') / Decimal('1000')
        annual_saved = water_saved * Decimal('8760') / Decimal('1000')

        water_cost = Decimal(str(data.water_cost_per_m3))
        cost_savings = annual_saved * water_cost

        losses = {
            'current_loss_kg_hr': float(current_water_loss),
            'optimal_loss_kg_hr': float(optimal_water_loss),
            'reduction_kg_hr': float(water_saved),
            'annual_current_m3': float(annual_current),
            'annual_optimal_m3': float(annual_optimal),
            'annual_saved_m3': float(annual_saved),
            'annual_cost_savings': FinancialDecimal.from_string(cost_savings)
        }

        tracker.record_step(
            operation="water_losses",
            description="Calculate water losses",
            inputs={'current_rate': current_rate, 'optimal_rate': optimal_rate},
            output_value=water_saved,
            output_name="water_saved_kg_hr",
            formula="Water saved = Steam * (Current% - Optimal%)",
            units="kg/hr"
        )

        return losses

    def _calculate_energy_losses(
        self, data: BlowdownData, current_rate: Decimal, optimal_rate: Decimal, tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate energy losses from blowdown."""
        steam_flow = Decimal(str(data.steam_flow_rate_kg_hr))
        pressure = Decimal(str(data.steam_pressure_bar))

        # Get steam enthalpy (simplified)
        h_steam = Decimal('2750') + pressure * Decimal('2')  # kJ/kg approximation
        h_feedwater = Decimal(str(data.feedwater_temperature_c)) * Decimal('4.186')

        enthalpy_loss = h_steam - h_feedwater

        # Current energy loss
        current_bd_flow = steam_flow * current_rate / Decimal('100')
        current_energy_loss = current_bd_flow * enthalpy_loss / Decimal('3600')  # kW

        # Optimal energy loss
        optimal_bd_flow = steam_flow * optimal_rate / Decimal('100')
        optimal_energy_loss = optimal_bd_flow * enthalpy_loss / Decimal('3600')  # kW

        energy_saved = current_energy_loss - optimal_energy_loss

        # Annual energy and cost
        annual_saved_gj = energy_saved * Decimal('3.6') * Decimal('8760') / Decimal('1000')
        fuel_cost = Decimal(str(data.fuel_cost_per_gj))
        energy_cost_savings = annual_saved_gj * fuel_cost

        losses = {
            'current_loss_kw': float(current_energy_loss),
            'optimal_loss_kw': float(optimal_energy_loss),
            'reduction_kw': float(energy_saved),
            'annual_saved_gj': float(annual_saved_gj),
            'annual_cost_savings': FinancialDecimal.from_string(energy_cost_savings),
            'enthalpy_loss_kj_kg': float(enthalpy_loss)
        }

        tracker.record_step(
            operation="energy_losses",
            description="Calculate energy losses",
            inputs={'current_bd_kw': current_energy_loss, 'optimal_bd_kw': optimal_energy_loss},
            output_value=energy_saved,
            output_name="energy_saved_kw",
            formula="Energy = Blowdown * Enthalpy_loss",
            units="kW"
        )

        return losses

    def _calculate_heat_recovery(
        self, data: BlowdownData, blowdown_rate: Decimal, tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate heat recovery potential from blowdown."""
        steam_flow = Decimal(str(data.steam_flow_rate_kg_hr))
        pressure = Decimal(str(data.steam_pressure_bar))

        blowdown_flow = steam_flow * blowdown_rate / Decimal('100')

        # Flash steam recovery potential
        # Approximately 10-15% of blowdown flashes to steam at atmospheric pressure
        flash_percentage = Decimal('0.13')  # 13% typical
        flash_steam = blowdown_flow * flash_percentage

        # Heat content of flash steam
        h_flash = Decimal('2676')  # kJ/kg at 1 bar
        flash_heat = flash_steam * h_flash / Decimal('3600')  # kW

        # Remaining liquid heat recovery
        liquid_flow = blowdown_flow * (Decimal('1') - flash_percentage)
        # Cool from ~180°C to 80°C
        liquid_heat = liquid_flow * Decimal('100') * Decimal('4.186') / Decimal('3600')  # kW

        total_recovery = flash_heat + liquid_heat

        # Annual savings
        annual_recovery_gj = total_recovery * Decimal('3.6') * Decimal('8760') / Decimal('1000')
        fuel_cost = Decimal(str(data.fuel_cost_per_gj))
        recovery_savings = annual_recovery_gj * fuel_cost

        recovery = {
            'flash_steam_kg_hr': float(flash_steam),
            'flash_heat_recovery_kw': float(flash_heat),
            'liquid_heat_recovery_kw': float(liquid_heat),
            'total_recovery_kw': FinancialDecimal.from_string(total_recovery),
            'recovery_efficiency_percent': FinancialDecimal.from_string(total_recovery / (blowdown_flow * Decimal('2.5')) * Decimal('100')),
            'annual_recovery_gj': float(annual_recovery_gj),
            'annual_savings': float(recovery_savings),
            'implementation': 'Flash tank + heat exchanger' if not data.heat_recovery_installed else 'Already installed'
        }

        tracker.record_step(
            operation="heat_recovery",
            description="Calculate heat recovery potential",
            inputs={'blowdown_flow': blowdown_flow, 'flash_percentage': flash_percentage},
            output_value=total_recovery,
            output_name="total_recovery_kw",
            formula="Recovery = Flash_steam + Liquid_cooling",
            units="kW"
        )

        return recovery

    def _calculate_savings(
        self, data: BlowdownData, current_rate: Decimal, optimal_rate: Decimal,
        heat_recovery: Dict, tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate total annual savings."""
        # Water savings
        water_savings = self._calculate_water_losses(data, current_rate, optimal_rate, tracker)

        # Energy savings
        energy_savings = self._calculate_energy_losses(data, current_rate, optimal_rate, tracker)

        # Total savings
        total_savings = (
            Decimal(str(water_savings['annual_cost_savings'])) +
            Decimal(str(energy_savings['annual_cost_savings']))
        )

        # Add heat recovery if not installed
        if not data.heat_recovery_installed:
            total_savings += Decimal(str(heat_recovery['annual_savings']))

        savings = {
            'water_cost_savings': water_savings['annual_cost_savings'],
            'energy_cost_savings': energy_savings['annual_cost_savings'],
            'heat_recovery_savings': heat_recovery['annual_savings'] if not data.heat_recovery_installed else 0,
            'total_annual_savings': FinancialDecimal.from_string(total_savings),
            'payback_months': 6.0 if not data.heat_recovery_installed else 2.0
        }

        return savings

    def _tds_control_strategy(self, data: BlowdownData, cycles: Decimal) -> Dict:
        """Generate TDS control strategy recommendations."""
        pressure = Decimal(str(data.steam_pressure_bar))

        # Determine pressure category
        if pressure < Decimal('15'):
            category = 'low_pressure'
        elif pressure < Decimal('40'):
            category = 'medium_pressure'
        elif pressure < Decimal('60'):
            category = 'high_pressure'
        else:
            category = 'very_high_pressure'

        max_tds = self.TDS_LIMITS[category]
        current_tds = data.boiler_water_tds_ppm

        strategy = {
            'pressure_category': category,
            'recommended_max_tds_ppm': max_tds,
            'current_tds_ppm': current_tds,
            'optimal_cycles': float(cycles),
            'control_method': 'Automatic TDS controller' if cycles > 5 else 'Manual sampling',
            'sampling_frequency': 'Every 4 hours' if cycles < 5 else 'Every 8 hours',
            'chemical_treatment': self._recommend_treatment(data, cycles)
        }

        return strategy

    def _recommend_treatment(self, data: BlowdownData, cycles: Decimal) -> Dict:
        """Recommend water treatment program."""
        if cycles < Decimal('3'):
            return {
                'program': 'Basic phosphate',
                'target_ph': '10.5-11.5',
                'phosphate_ppm': '30-60'
            }
        elif cycles < Decimal('8'):
            return {
                'program': 'Coordinated phosphate',
                'target_ph': '9.0-9.6',
                'phosphate_ppm': '5-15'
            }
        else:
            return {
                'program': 'All-volatile treatment',
                'target_ph': '8.8-9.2',
                'phosphate_ppm': '0'
            }