# -*- coding: utf-8 -*-
"""
Unit tests for Blowdown Optimizer - GL-016 WATERGUARD

Comprehensive test suite covering:
- Blowdown rate calculations
- Cycles of concentration optimization
- Heat loss calculations
- Cost optimization scenarios
- Edge cases (minimum/maximum blowdown)

Target: 95%+ code coverage
"""

import pytest
import math
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any
from hypothesis import given, strategies as st, settings, assume

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.provenance import ProvenanceTracker


# ============================================================================
# Blowdown Calculator Class (for testing)
# ============================================================================

class BlowdownOptimizer:
    """
    Blowdown optimization calculator for boiler systems.

    Calculates optimal blowdown rates to balance:
    - Water chemistry control (TDS, conductivity)
    - Heat energy conservation
    - Chemical costs
    - Water costs
    """

    # Constants
    WATER_SPECIFIC_HEAT = Decimal('4.186')  # kJ/kg/K
    STEAM_ENTHALPY_100C = Decimal('2676')    # kJ/kg at 100C

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_blowdown_rate(
        self,
        steam_generation_rate_kg_hr: float,
        cycles_of_concentration: float
    ) -> Dict[str, Any]:
        """
        Calculate required blowdown rate.

        Blowdown Rate = Steam Rate / (Cycles - 1)
        """
        tracker = ProvenanceTracker(
            calculation_id=f"blowdown_{id(self)}",
            calculation_type="blowdown_rate",
            version=self.version
        )

        steam_rate = Decimal(str(steam_generation_rate_kg_hr))
        cycles = Decimal(str(cycles_of_concentration))

        tracker.record_inputs({
            'steam_generation_rate_kg_hr': steam_rate,
            'cycles_of_concentration': cycles
        })

        # Blowdown rate calculation
        if cycles <= 1:
            raise ValueError("Cycles of concentration must be > 1")

        blowdown_rate = steam_rate / (cycles - Decimal('1'))
        blowdown_percent = (blowdown_rate / steam_rate) * Decimal('100')

        tracker.record_step(
            operation="blowdown_rate",
            description="Calculate blowdown rate from steam rate and cycles",
            inputs={'steam_rate': steam_rate, 'cycles': cycles},
            output_value=blowdown_rate,
            output_name="blowdown_rate_kg_hr",
            formula="B = S / (C - 1)",
            units="kg/hr"
        )

        return {
            'blowdown_rate_kg_hr': float(blowdown_rate.quantize(Decimal('0.01'))),
            'blowdown_rate_percent': float(blowdown_percent.quantize(Decimal('0.01'))),
            'makeup_rate_kg_hr': float((steam_rate + blowdown_rate).quantize(Decimal('0.01'))),
            'provenance': tracker.get_provenance_record(blowdown_rate).to_dict()
        }

    def calculate_optimal_cycles(
        self,
        makeup_tds_ppm: float,
        max_boiler_tds_ppm: float,
        silica_limit_ppm: float = None,
        makeup_silica_ppm: float = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal cycles of concentration.

        Limited by TDS or silica, whichever gives lower cycles.
        """
        tracker = ProvenanceTracker(
            calculation_id=f"optimal_cycles_{id(self)}",
            calculation_type="cycles_optimization",
            version=self.version
        )

        makeup_tds = Decimal(str(makeup_tds_ppm))
        max_tds = Decimal(str(max_boiler_tds_ppm))

        tracker.record_inputs({
            'makeup_tds_ppm': makeup_tds,
            'max_boiler_tds_ppm': max_tds,
            'silica_limit_ppm': silica_limit_ppm,
            'makeup_silica_ppm': makeup_silica_ppm
        })

        # TDS-limited cycles
        if makeup_tds <= 0:
            raise ValueError("Makeup TDS must be > 0")

        cycles_tds = max_tds / makeup_tds
        limiting_factor = "TDS"

        # Silica-limited cycles (if provided)
        if silica_limit_ppm and makeup_silica_ppm and makeup_silica_ppm > 0:
            silica_limit = Decimal(str(silica_limit_ppm))
            makeup_silica = Decimal(str(makeup_silica_ppm))
            cycles_silica = silica_limit / makeup_silica

            if cycles_silica < cycles_tds:
                optimal_cycles = cycles_silica
                limiting_factor = "Silica"
            else:
                optimal_cycles = cycles_tds
        else:
            optimal_cycles = cycles_tds

        tracker.record_step(
            operation="optimal_cycles",
            description="Calculate optimal cycles limited by TDS or silica",
            inputs={'cycles_tds': cycles_tds, 'limiting_factor': limiting_factor},
            output_value=optimal_cycles,
            output_name="optimal_cycles",
            formula="C = Max_TDS / Makeup_TDS",
            units="dimensionless"
        )

        return {
            'optimal_cycles': float(optimal_cycles.quantize(Decimal('0.1'))),
            'limiting_factor': limiting_factor,
            'cycles_by_tds': float(cycles_tds.quantize(Decimal('0.1'))),
            'provenance': tracker.get_provenance_record(optimal_cycles).to_dict()
        }

    def calculate_heat_loss(
        self,
        blowdown_rate_kg_hr: float,
        boiler_pressure_bar: float,
        feedwater_temp_c: float
    ) -> Dict[str, Any]:
        """
        Calculate heat loss from blowdown.

        Heat Loss = Blowdown Rate x (h_blowdown - h_feedwater)
        """
        tracker = ProvenanceTracker(
            calculation_id=f"heat_loss_{id(self)}",
            calculation_type="heat_loss",
            version=self.version
        )

        blowdown_rate = Decimal(str(blowdown_rate_kg_hr))
        pressure = Decimal(str(boiler_pressure_bar))
        feedwater_temp = Decimal(str(feedwater_temp_c))

        tracker.record_inputs({
            'blowdown_rate_kg_hr': blowdown_rate,
            'boiler_pressure_bar': pressure,
            'feedwater_temp_c': feedwater_temp
        })

        # Approximate saturation temperature from pressure
        # T_sat (C) approx = 100 + 28.5 * ln(P)
        if pressure <= 0:
            raise ValueError("Pressure must be > 0")

        t_sat = Decimal('100') + Decimal('28.5') * pressure.ln()

        # Enthalpy of saturated water (approximate)
        # h = 4.186 * T (kJ/kg) for liquid water
        h_blowdown = self.WATER_SPECIFIC_HEAT * t_sat
        h_feedwater = self.WATER_SPECIFIC_HEAT * feedwater_temp

        # Heat loss (kJ/hr)
        heat_loss_kj_hr = blowdown_rate * (h_blowdown - h_feedwater)
        heat_loss_kw = heat_loss_kj_hr / Decimal('3600')

        tracker.record_step(
            operation="heat_loss",
            description="Calculate heat energy lost in blowdown",
            inputs={
                'blowdown_rate': blowdown_rate,
                'h_blowdown': h_blowdown,
                'h_feedwater': h_feedwater
            },
            output_value=heat_loss_kw,
            output_name="heat_loss_kw",
            formula="Q = m * (h_bd - h_fw)",
            units="kW"
        )

        return {
            'heat_loss_kw': float(heat_loss_kw.quantize(Decimal('0.01'))),
            'heat_loss_kj_hr': float(heat_loss_kj_hr.quantize(Decimal('0.1'))),
            'saturation_temp_c': float(t_sat.quantize(Decimal('0.1'))),
            'provenance': tracker.get_provenance_record(heat_loss_kw).to_dict()
        }

    def calculate_cost_optimization(
        self,
        steam_rate_kg_hr: float,
        makeup_tds_ppm: float,
        max_tds_ppm: float,
        water_cost_per_m3: float,
        energy_cost_per_kwh: float,
        boiler_pressure_bar: float,
        feedwater_temp_c: float,
        chemical_cost_per_m3: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate optimal blowdown for minimum total cost.

        Total Cost = Water Cost + Energy Cost + Chemical Cost
        """
        tracker = ProvenanceTracker(
            calculation_id=f"cost_opt_{id(self)}",
            calculation_type="cost_optimization",
            version=self.version
        )

        steam_rate = Decimal(str(steam_rate_kg_hr))
        makeup_tds = Decimal(str(makeup_tds_ppm))
        max_tds = Decimal(str(max_tds_ppm))
        water_cost = Decimal(str(water_cost_per_m3))
        energy_cost = Decimal(str(energy_cost_per_kwh))
        pressure = Decimal(str(boiler_pressure_bar))
        feedwater_temp = Decimal(str(feedwater_temp_c))
        chem_cost = Decimal(str(chemical_cost_per_m3))

        tracker.record_inputs({
            'steam_rate_kg_hr': steam_rate,
            'makeup_tds_ppm': makeup_tds,
            'max_tds_ppm': max_tds,
            'water_cost_per_m3': water_cost,
            'energy_cost_per_kwh': energy_cost
        })

        # Calculate for various cycles
        best_cycles = Decimal('2')
        min_total_cost = Decimal('999999999')

        max_cycles = max_tds / makeup_tds

        # Analyze at different cycles from 2 to max
        analysis_results = []
        cycles = Decimal('2')

        while cycles <= max_cycles and cycles <= Decimal('20'):
            # Blowdown rate
            blowdown_rate = steam_rate / (cycles - Decimal('1'))
            makeup_rate = steam_rate + blowdown_rate

            # Convert to m3/hr (assuming 1000 kg/m3)
            makeup_m3_hr = makeup_rate / Decimal('1000')

            # Water cost ($/hr)
            water_cost_hr = makeup_m3_hr * water_cost

            # Heat loss
            if pressure > 0:
                t_sat = Decimal('100') + Decimal('28.5') * pressure.ln()
                h_blowdown = self.WATER_SPECIFIC_HEAT * t_sat
                h_feedwater = self.WATER_SPECIFIC_HEAT * feedwater_temp
                heat_loss_kw = blowdown_rate * (h_blowdown - h_feedwater) / Decimal('3600')
            else:
                heat_loss_kw = Decimal('0')

            # Energy cost ($/hr)
            energy_cost_hr = heat_loss_kw * energy_cost

            # Chemical cost ($/hr)
            chem_cost_hr = makeup_m3_hr * chem_cost

            # Total cost ($/hr)
            total_cost_hr = water_cost_hr + energy_cost_hr + chem_cost_hr

            analysis_results.append({
                'cycles': float(cycles),
                'blowdown_kg_hr': float(blowdown_rate.quantize(Decimal('0.1'))),
                'total_cost_hr': float(total_cost_hr.quantize(Decimal('0.01')))
            })

            if total_cost_hr < min_total_cost:
                min_total_cost = total_cost_hr
                best_cycles = cycles

            cycles += Decimal('0.5')

        # Calculate costs at optimal cycles
        optimal_blowdown = steam_rate / (best_cycles - Decimal('1'))
        optimal_makeup = steam_rate + optimal_blowdown

        tracker.record_step(
            operation="cost_optimization",
            description="Find optimal cycles for minimum cost",
            inputs={'max_cycles': max_cycles, 'analysis_points': len(analysis_results)},
            output_value=best_cycles,
            output_name="optimal_cycles",
            formula="Minimize(Water + Energy + Chemical costs)",
            units="dimensionless"
        )

        return {
            'optimal_cycles': float(best_cycles.quantize(Decimal('0.1'))),
            'optimal_blowdown_kg_hr': float(optimal_blowdown.quantize(Decimal('0.1'))),
            'minimum_cost_per_hr': float(min_total_cost.quantize(Decimal('0.01'))),
            'makeup_rate_kg_hr': float(optimal_makeup.quantize(Decimal('0.1'))),
            'analysis': analysis_results[:10],  # First 10 points
            'provenance': tracker.get_provenance_record(best_cycles).to_dict()
        }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def optimizer():
    """Create BlowdownOptimizer instance."""
    return BlowdownOptimizer(version="1.0.0-test")


@pytest.fixture
def standard_boiler_params():
    """Standard boiler parameters for testing."""
    return {
        'steam_rate_kg_hr': 10000.0,
        'boiler_pressure_bar': 10.0,
        'feedwater_temp_c': 105.0,
        'makeup_tds_ppm': 200.0,
        'max_tds_ppm': 3000.0
    }


# ============================================================================
# Blowdown Rate Calculation Tests
# ============================================================================

@pytest.mark.unit
class TestBlowdownRateCalculations:
    """Test blowdown rate calculations."""

    def test_basic_blowdown_rate(self, optimizer):
        """Test basic blowdown rate calculation."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=10.0
        )

        # Blowdown = 10000 / (10 - 1) = 1111.11 kg/hr
        assert result['blowdown_rate_kg_hr'] == pytest.approx(1111.11, rel=0.01)
        assert 'provenance' in result

    def test_blowdown_rate_low_cycles(self, optimizer):
        """Test blowdown rate with low cycles (high blowdown)."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=2.0
        )

        # Blowdown = 10000 / (2 - 1) = 10000 kg/hr
        assert result['blowdown_rate_kg_hr'] == pytest.approx(10000.0, rel=0.01)

    def test_blowdown_rate_high_cycles(self, optimizer):
        """Test blowdown rate with high cycles (low blowdown)."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=20.0
        )

        # Blowdown = 10000 / (20 - 1) = 526.32 kg/hr
        assert result['blowdown_rate_kg_hr'] == pytest.approx(526.32, rel=0.01)

    def test_blowdown_rate_invalid_cycles(self, optimizer):
        """Test blowdown rate with invalid cycles <= 1."""
        with pytest.raises(ValueError):
            optimizer.calculate_blowdown_rate(
                steam_generation_rate_kg_hr=10000.0,
                cycles_of_concentration=1.0
            )

        with pytest.raises(ValueError):
            optimizer.calculate_blowdown_rate(
                steam_generation_rate_kg_hr=10000.0,
                cycles_of_concentration=0.5
            )

    def test_blowdown_percent_calculation(self, optimizer):
        """Test blowdown percentage calculation."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=5.0
        )

        # Blowdown = 10000 / 4 = 2500 kg/hr
        # Percent = 2500 / 10000 * 100 = 25%
        assert result['blowdown_rate_percent'] == pytest.approx(25.0, rel=0.01)

    def test_makeup_rate_calculation(self, optimizer):
        """Test makeup water rate calculation."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=5.0
        )

        # Makeup = Steam + Blowdown = 10000 + 2500 = 12500 kg/hr
        assert result['makeup_rate_kg_hr'] == pytest.approx(12500.0, rel=0.01)

    @pytest.mark.parametrize("steam_rate,cycles,expected_blowdown", [
        (1000, 5, 250.0),
        (5000, 10, 555.56),
        (20000, 8, 2857.14),
        (50000, 15, 3571.43),
    ])
    def test_blowdown_rate_parametrized(self, optimizer, steam_rate, cycles, expected_blowdown):
        """Parametrized blowdown rate tests."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=steam_rate,
            cycles_of_concentration=cycles
        )

        assert result['blowdown_rate_kg_hr'] == pytest.approx(expected_blowdown, rel=0.01)


# ============================================================================
# Cycles of Concentration Optimization Tests
# ============================================================================

@pytest.mark.unit
class TestCyclesOptimization:
    """Test cycles of concentration optimization."""

    def test_optimal_cycles_tds_limited(self, optimizer):
        """Test optimal cycles limited by TDS."""
        result = optimizer.calculate_optimal_cycles(
            makeup_tds_ppm=200.0,
            max_boiler_tds_ppm=3000.0
        )

        # Cycles = 3000 / 200 = 15
        assert result['optimal_cycles'] == pytest.approx(15.0, rel=0.01)
        assert result['limiting_factor'] == "TDS"

    def test_optimal_cycles_silica_limited(self, optimizer):
        """Test optimal cycles limited by silica."""
        result = optimizer.calculate_optimal_cycles(
            makeup_tds_ppm=200.0,
            max_boiler_tds_ppm=3000.0,
            silica_limit_ppm=150.0,
            makeup_silica_ppm=20.0
        )

        # TDS cycles = 15
        # Silica cycles = 150 / 20 = 7.5
        # Silica is more limiting
        assert result['optimal_cycles'] == pytest.approx(7.5, rel=0.01)
        assert result['limiting_factor'] == "Silica"

    def test_optimal_cycles_tds_more_limiting_than_silica(self, optimizer):
        """Test when TDS is more limiting than silica."""
        result = optimizer.calculate_optimal_cycles(
            makeup_tds_ppm=500.0,
            max_boiler_tds_ppm=2000.0,
            silica_limit_ppm=150.0,
            makeup_silica_ppm=10.0
        )

        # TDS cycles = 2000 / 500 = 4
        # Silica cycles = 150 / 10 = 15
        # TDS is more limiting
        assert result['optimal_cycles'] == pytest.approx(4.0, rel=0.01)
        assert result['limiting_factor'] == "TDS"

    def test_optimal_cycles_zero_makeup_tds(self, optimizer):
        """Test optimal cycles with zero makeup TDS."""
        with pytest.raises(ValueError):
            optimizer.calculate_optimal_cycles(
                makeup_tds_ppm=0.0,
                max_boiler_tds_ppm=3000.0
            )

    def test_optimal_cycles_no_silica_limit(self, optimizer):
        """Test optimal cycles without silica limit specified."""
        result = optimizer.calculate_optimal_cycles(
            makeup_tds_ppm=300.0,
            max_boiler_tds_ppm=3000.0
        )

        assert result['optimal_cycles'] == pytest.approx(10.0, rel=0.01)
        assert result['limiting_factor'] == "TDS"

    @pytest.mark.parametrize("makeup_tds,max_tds,expected_cycles", [
        (100, 1000, 10.0),
        (200, 3000, 15.0),
        (500, 2500, 5.0),
        (50, 1500, 30.0),
    ])
    def test_optimal_cycles_parametrized(self, optimizer, makeup_tds, max_tds, expected_cycles):
        """Parametrized optimal cycles tests."""
        result = optimizer.calculate_optimal_cycles(
            makeup_tds_ppm=makeup_tds,
            max_boiler_tds_ppm=max_tds
        )

        assert result['optimal_cycles'] == pytest.approx(expected_cycles, rel=0.01)


# ============================================================================
# Heat Loss Calculation Tests
# ============================================================================

@pytest.mark.unit
class TestHeatLossCalculations:
    """Test heat loss calculations."""

    def test_basic_heat_loss(self, optimizer):
        """Test basic heat loss calculation."""
        result = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=1000.0,
            boiler_pressure_bar=10.0,
            feedwater_temp_c=105.0
        )

        assert result['heat_loss_kw'] > 0
        assert 'saturation_temp_c' in result
        assert 'provenance' in result

    def test_heat_loss_high_blowdown(self, optimizer):
        """Test heat loss with high blowdown rate."""
        result = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=5000.0,
            boiler_pressure_bar=10.0,
            feedwater_temp_c=105.0
        )

        # Higher blowdown should result in higher heat loss
        assert result['heat_loss_kw'] > 100  # Significant loss expected

    def test_heat_loss_low_blowdown(self, optimizer):
        """Test heat loss with low blowdown rate."""
        result = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=100.0,
            boiler_pressure_bar=10.0,
            feedwater_temp_c=105.0
        )

        # Lower blowdown should result in lower heat loss
        assert result['heat_loss_kw'] < 50

    def test_heat_loss_high_pressure(self, optimizer):
        """Test heat loss at high pressure."""
        result_low = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=1000.0,
            boiler_pressure_bar=5.0,
            feedwater_temp_c=105.0
        )
        result_high = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=1000.0,
            boiler_pressure_bar=50.0,
            feedwater_temp_c=105.0
        )

        # Higher pressure = higher saturation temp = more heat loss
        assert result_high['heat_loss_kw'] > result_low['heat_loss_kw']

    def test_heat_loss_invalid_pressure(self, optimizer):
        """Test heat loss with invalid pressure."""
        with pytest.raises(ValueError):
            optimizer.calculate_heat_loss(
                blowdown_rate_kg_hr=1000.0,
                boiler_pressure_bar=0.0,
                feedwater_temp_c=105.0
            )

    def test_heat_loss_hot_feedwater(self, optimizer):
        """Test heat loss with hot feedwater (reduced delta T)."""
        result_cold = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=1000.0,
            boiler_pressure_bar=10.0,
            feedwater_temp_c=50.0
        )
        result_hot = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=1000.0,
            boiler_pressure_bar=10.0,
            feedwater_temp_c=150.0
        )

        # Hotter feedwater = less heat loss (smaller delta T)
        assert result_hot['heat_loss_kw'] < result_cold['heat_loss_kw']

    @pytest.mark.parametrize("blowdown,pressure,feedwater_temp", [
        (500, 5.0, 80.0),
        (1000, 10.0, 105.0),
        (2000, 20.0, 120.0),
        (5000, 40.0, 150.0),
    ])
    def test_heat_loss_parametrized(self, optimizer, blowdown, pressure, feedwater_temp):
        """Parametrized heat loss tests."""
        result = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=blowdown,
            boiler_pressure_bar=pressure,
            feedwater_temp_c=feedwater_temp
        )

        assert result['heat_loss_kw'] > 0
        assert result['heat_loss_kj_hr'] > 0
        assert result['saturation_temp_c'] > feedwater_temp


# ============================================================================
# Cost Optimization Tests
# ============================================================================

@pytest.mark.unit
class TestCostOptimization:
    """Test cost optimization calculations."""

    def test_basic_cost_optimization(self, optimizer, standard_boiler_params):
        """Test basic cost optimization."""
        result = optimizer.calculate_cost_optimization(
            steam_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            makeup_tds_ppm=standard_boiler_params['makeup_tds_ppm'],
            max_tds_ppm=standard_boiler_params['max_tds_ppm'],
            water_cost_per_m3=2.50,
            energy_cost_per_kwh=0.12,
            boiler_pressure_bar=standard_boiler_params['boiler_pressure_bar'],
            feedwater_temp_c=standard_boiler_params['feedwater_temp_c']
        )

        assert 'optimal_cycles' in result
        assert 'minimum_cost_per_hr' in result
        assert 'analysis' in result
        assert result['optimal_cycles'] >= 2.0

    def test_cost_optimization_high_water_cost(self, optimizer, standard_boiler_params):
        """Test optimization with high water cost favors higher cycles."""
        result_low_water = optimizer.calculate_cost_optimization(
            steam_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            makeup_tds_ppm=standard_boiler_params['makeup_tds_ppm'],
            max_tds_ppm=standard_boiler_params['max_tds_ppm'],
            water_cost_per_m3=1.00,
            energy_cost_per_kwh=0.12,
            boiler_pressure_bar=standard_boiler_params['boiler_pressure_bar'],
            feedwater_temp_c=standard_boiler_params['feedwater_temp_c']
        )

        result_high_water = optimizer.calculate_cost_optimization(
            steam_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            makeup_tds_ppm=standard_boiler_params['makeup_tds_ppm'],
            max_tds_ppm=standard_boiler_params['max_tds_ppm'],
            water_cost_per_m3=10.00,
            energy_cost_per_kwh=0.12,
            boiler_pressure_bar=standard_boiler_params['boiler_pressure_bar'],
            feedwater_temp_c=standard_boiler_params['feedwater_temp_c']
        )

        # Higher water cost should favor higher cycles (less makeup)
        assert result_high_water['optimal_cycles'] >= result_low_water['optimal_cycles']

    def test_cost_optimization_analysis_points(self, optimizer, standard_boiler_params):
        """Test that optimization provides analysis points."""
        result = optimizer.calculate_cost_optimization(
            steam_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            makeup_tds_ppm=standard_boiler_params['makeup_tds_ppm'],
            max_tds_ppm=standard_boiler_params['max_tds_ppm'],
            water_cost_per_m3=2.50,
            energy_cost_per_kwh=0.12,
            boiler_pressure_bar=standard_boiler_params['boiler_pressure_bar'],
            feedwater_temp_c=standard_boiler_params['feedwater_temp_c']
        )

        assert len(result['analysis']) > 0
        for point in result['analysis']:
            assert 'cycles' in point
            assert 'blowdown_kg_hr' in point
            assert 'total_cost_hr' in point

    def test_cost_optimization_provenance(self, optimizer, standard_boiler_params):
        """Test that cost optimization includes provenance."""
        result = optimizer.calculate_cost_optimization(
            steam_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            makeup_tds_ppm=standard_boiler_params['makeup_tds_ppm'],
            max_tds_ppm=standard_boiler_params['max_tds_ppm'],
            water_cost_per_m3=2.50,
            energy_cost_per_kwh=0.12,
            boiler_pressure_bar=standard_boiler_params['boiler_pressure_bar'],
            feedwater_temp_c=standard_boiler_params['feedwater_temp_c']
        )

        assert 'provenance' in result
        assert 'provenance_hash' in result['provenance']


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for blowdown calculations."""

    def test_minimum_blowdown_high_cycles(self, optimizer):
        """Test minimum blowdown at maximum practical cycles."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=50.0
        )

        # Very high cycles should give very low blowdown
        assert result['blowdown_rate_kg_hr'] == pytest.approx(204.08, rel=0.01)
        assert result['blowdown_rate_percent'] < 3.0

    def test_maximum_blowdown_low_cycles(self, optimizer):
        """Test maximum blowdown at minimum practical cycles."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=1.5
        )

        # Low cycles = high blowdown
        assert result['blowdown_rate_kg_hr'] == pytest.approx(20000.0, rel=0.01)
        assert result['blowdown_rate_percent'] > 100.0  # More than steam rate!

    def test_very_small_steam_rate(self, optimizer):
        """Test with very small steam generation rate."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10.0,
            cycles_of_concentration=5.0
        )

        assert result['blowdown_rate_kg_hr'] == pytest.approx(2.5, rel=0.01)

    def test_very_large_steam_rate(self, optimizer):
        """Test with very large steam generation rate."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=1000000.0,
            cycles_of_concentration=10.0
        )

        assert result['blowdown_rate_kg_hr'] == pytest.approx(111111.11, rel=0.01)

    def test_cycles_just_above_one(self, optimizer):
        """Test cycles marginally above 1."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=10000.0,
            cycles_of_concentration=1.01
        )

        # Very high blowdown expected
        assert result['blowdown_rate_kg_hr'] == pytest.approx(1000000.0, rel=0.1)


# ============================================================================
# Property-Based Tests
# ============================================================================

@pytest.mark.unit
class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        steam_rate=st.floats(min_value=100.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        cycles=st.floats(min_value=1.5, max_value=30.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_blowdown_rate_always_positive(self, optimizer, steam_rate, cycles):
        """Test blowdown rate is always positive for valid inputs."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=steam_rate,
            cycles_of_concentration=cycles
        )

        assert result['blowdown_rate_kg_hr'] > 0
        assert result['blowdown_rate_percent'] > 0

    @given(
        steam_rate=st.floats(min_value=100.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        cycles=st.floats(min_value=1.5, max_value=30.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_makeup_equals_steam_plus_blowdown(self, optimizer, steam_rate, cycles):
        """Test makeup = steam + blowdown."""
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=steam_rate,
            cycles_of_concentration=cycles
        )

        expected_makeup = steam_rate + result['blowdown_rate_kg_hr']
        assert result['makeup_rate_kg_hr'] == pytest.approx(expected_makeup, rel=0.01)

    @given(
        makeup_tds=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        max_tds=st.floats(min_value=1000.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30)
    def test_optimal_cycles_within_bounds(self, optimizer, makeup_tds, max_tds):
        """Test optimal cycles are within valid bounds."""
        assume(max_tds > makeup_tds)

        result = optimizer.calculate_optimal_cycles(
            makeup_tds_ppm=makeup_tds,
            max_boiler_tds_ppm=max_tds
        )

        assert result['optimal_cycles'] >= 1.0
        assert result['optimal_cycles'] <= max_tds / makeup_tds + 0.1


# ============================================================================
# Determinism Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.determinism
class TestDeterminism:
    """Test calculation determinism."""

    def test_blowdown_rate_deterministic(self, optimizer):
        """Test blowdown rate is deterministic."""
        results = [
            optimizer.calculate_blowdown_rate(
                steam_generation_rate_kg_hr=10000.0,
                cycles_of_concentration=10.0
            )
            for _ in range(10)
        ]

        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 1  # All hashes should be identical

    def test_optimal_cycles_deterministic(self, optimizer):
        """Test optimal cycles is deterministic."""
        results = [
            optimizer.calculate_optimal_cycles(
                makeup_tds_ppm=200.0,
                max_boiler_tds_ppm=3000.0
            )
            for _ in range(10)
        ]

        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 1

    def test_heat_loss_deterministic(self, optimizer):
        """Test heat loss is deterministic."""
        results = [
            optimizer.calculate_heat_loss(
                blowdown_rate_kg_hr=1000.0,
                boiler_pressure_bar=10.0,
                feedwater_temp_c=105.0
            )
            for _ in range(10)
        ]

        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 1


# ============================================================================
# Golden Data Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.golden
class TestGoldenData:
    """Tests using golden reference data."""

    def test_golden_blowdown_values(self, optimizer, golden_blowdown_test_data):
        """Test blowdown calculations against golden reference values."""
        for test_case in golden_blowdown_test_data:
            result = optimizer.calculate_blowdown_rate(
                steam_generation_rate_kg_hr=test_case['steam_rate'],
                cycles_of_concentration=test_case['cycles']
            )

            assert result['blowdown_rate_kg_hr'] == pytest.approx(
                test_case['expected_blowdown'],
                abs=test_case['tolerance']
            )

    def test_known_industrial_case(self, optimizer):
        """Test against known industrial case."""
        # Industrial boiler: 50 t/hr steam, 8 cycles
        # Expected blowdown: 50000 / (8-1) = 7142.86 kg/hr
        result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=50000.0,
            cycles_of_concentration=8.0
        )

        assert result['blowdown_rate_kg_hr'] == pytest.approx(7142.86, rel=0.01)
        assert result['blowdown_rate_percent'] == pytest.approx(14.29, rel=0.01)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.unit
class TestIntegration:
    """Test integration between blowdown calculations."""

    def test_full_blowdown_analysis(self, optimizer, standard_boiler_params):
        """Test complete blowdown analysis workflow."""
        # Step 1: Calculate optimal cycles
        cycles_result = optimizer.calculate_optimal_cycles(
            makeup_tds_ppm=standard_boiler_params['makeup_tds_ppm'],
            max_boiler_tds_ppm=standard_boiler_params['max_tds_ppm']
        )

        # Step 2: Calculate blowdown rate
        blowdown_result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            cycles_of_concentration=cycles_result['optimal_cycles']
        )

        # Step 3: Calculate heat loss
        heat_loss_result = optimizer.calculate_heat_loss(
            blowdown_rate_kg_hr=blowdown_result['blowdown_rate_kg_hr'],
            boiler_pressure_bar=standard_boiler_params['boiler_pressure_bar'],
            feedwater_temp_c=standard_boiler_params['feedwater_temp_c']
        )

        # Verify results are consistent
        assert cycles_result['optimal_cycles'] > 1
        assert blowdown_result['blowdown_rate_kg_hr'] > 0
        assert heat_loss_result['heat_loss_kw'] > 0

    def test_cost_vs_individual_calculations(self, optimizer, standard_boiler_params):
        """Test cost optimization aligns with individual calculations."""
        cost_result = optimizer.calculate_cost_optimization(
            steam_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            makeup_tds_ppm=standard_boiler_params['makeup_tds_ppm'],
            max_tds_ppm=standard_boiler_params['max_tds_ppm'],
            water_cost_per_m3=2.50,
            energy_cost_per_kwh=0.12,
            boiler_pressure_bar=standard_boiler_params['boiler_pressure_bar'],
            feedwater_temp_c=standard_boiler_params['feedwater_temp_c']
        )

        # Verify blowdown at optimal cycles matches
        blowdown_result = optimizer.calculate_blowdown_rate(
            steam_generation_rate_kg_hr=standard_boiler_params['steam_rate_kg_hr'],
            cycles_of_concentration=cost_result['optimal_cycles']
        )

        assert blowdown_result['blowdown_rate_kg_hr'] == pytest.approx(
            cost_result['optimal_blowdown_kg_hr'],
            rel=0.05
        )
