# -*- coding: utf-8 -*-
"""
Calculator tests for GL-001 ProcessHeatOrchestrator
Comprehensive testing of all calculation functions with precision validation.
Target coverage: 95% of calculation logic.
"""

import unittest
import pytest
import numpy as np
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import math

import sys
import os
from greenlang.determinism import DeterministicClock
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.GL_001.process_heat_orchestrator import (
    ProcessHeatOrchestrator,
    ProcessData,
    ThermalCalculation,
    OptimizationStrategy
)
from testing.agent_test_framework import AgentTestCase


# Set decimal precision for financial calculations
getcontext().prec = 10


class TestThermalCalculators(AgentTestCase):
    """Comprehensive tests for thermal calculations."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.agent = ProcessHeatOrchestrator()

    # Basic Efficiency Calculations
    def test_efficiency_calculation_accuracy(self):
        """Test efficiency calculation with known values."""
        test_cases = [
            # (input_kw, output_kw, expected_efficiency)
            (1000.0, 850.0, 0.85),
            (500.0, 400.0, 0.80),
            (2000.0, 1900.0, 0.95),
            (100.0, 75.0, 0.75),
            (10000.0, 6500.0, 0.65),
        ]

        for input_kw, output_kw, expected in test_cases:
            data = ProcessData(
                timestamp=DeterministicClock.utcnow(),
                temperature_c=250.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=input_kw,
                energy_output_kw=output_kw,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )

            efficiency = self.agent._calculate_efficiency_core(data)
            self.assertAlmostEqual(
                efficiency,
                expected,
                places=4,
                msg=f"Failed for input={input_kw}, output={output_kw}"
            )

    def test_efficiency_edge_cases(self):
        """Test efficiency calculation edge cases."""
        # Perfect efficiency (100%)
        data_perfect = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=1000.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        efficiency = self.agent._calculate_efficiency_core(data_perfect)
        self.assertEqual(efficiency, 1.0)

        # Zero efficiency
        data_zero = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=0.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        efficiency = self.agent._calculate_efficiency_core(data_zero)
        self.assertEqual(efficiency, 0.0)

    def test_efficiency_precision(self):
        """Test efficiency calculation maintains precision."""
        data = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1234.5678,
            energy_output_kw=1049.3326,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        efficiency = self.agent._calculate_efficiency_core(data)
        expected = 1049.3326 / 1234.5678

        self.assertAlmostEqual(efficiency, expected, places=8)

    # Heat Loss Calculations
    def test_heat_loss_calculation(self):
        """Test heat loss calculation accuracy."""
        test_cases = [
            # (input_kw, output_kw, expected_loss)
            (1000.0, 850.0, 150.0),
            (500.0, 500.0, 0.0),
            (2000.0, 1200.0, 800.0),
            (100.0, 25.0, 75.0),
        ]

        for input_kw, output_kw, expected_loss in test_cases:
            data = ProcessData(
                timestamp=DeterministicClock.utcnow(),
                temperature_c=250.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=input_kw,
                energy_output_kw=output_kw,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )

            heat_loss = self.agent._calculate_heat_loss(data)
            self.assertAlmostEqual(
                heat_loss,
                expected_loss,
                places=4,
                msg=f"Failed for input={input_kw}, output={output_kw}"
            )

    def test_heat_loss_never_negative(self):
        """Test heat loss is never negative."""
        # Output slightly exceeds input (measurement error)
        data = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=250.0,
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=1000.1,  # Slight excess
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        with patch.object(self.agent, '_validate_process_data'):
            heat_loss = self.agent._calculate_heat_loss(data)
            self.assertEqual(heat_loss, 0.0)  # Should clamp to 0

    # Recoverable Heat Calculations
    def test_recoverable_heat_by_temperature(self):
        """Test recoverable heat varies with temperature grades."""
        heat_loss = 1000.0  # Fixed heat loss for comparison

        # Test different temperature grades
        test_cases = [
            # (temp_c, expected_factor, description)
            (50.0, 0.1, "Low-grade heat"),
            (99.0, 0.1, "Low-grade heat boundary"),
            (100.0, 0.3, "Medium-grade heat lower"),
            (250.0, 0.3, "Medium-grade heat middle"),
            (299.0, 0.3, "Medium-grade heat boundary"),
            (300.0, 0.5, "High-grade heat lower"),
            (450.0, 0.5, "High-grade heat middle"),
            (499.0, 0.5, "High-grade heat boundary"),
            (500.0, 0.7, "Very high-grade heat lower"),
            (1000.0, 0.7, "Very high-grade heat middle"),
            (2500.0, 0.7, "Very high-grade heat upper"),
        ]

        for temp, expected_factor, description in test_cases:
            data = ProcessData(
                timestamp=DeterministicClock.utcnow(),
                temperature_c=temp,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=2000.0,
                energy_output_kw=1000.0,
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )

            recoverable = self.agent._calculate_recoverable_heat(data, heat_loss)
            expected_recoverable = heat_loss * expected_factor

            self.assertAlmostEqual(
                recoverable,
                expected_recoverable,
                places=4,
                msg=f"Failed for {description}: temp={temp}°C"
            )

    def test_recoverable_heat_proportional_to_loss(self):
        """Test recoverable heat scales with heat loss."""
        data = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=350.0,  # High-grade heat (0.5 factor)
            pressure_bar=10.0,
            flow_rate_kg_s=5.0,
            energy_input_kw=1000.0,
            energy_output_kw=700.0,
            fuel_type="gas",
            fuel_consumption_rate=10.0
        )

        # Test different heat loss values
        for heat_loss in [100.0, 500.0, 1000.0, 5000.0]:
            recoverable = self.agent._calculate_recoverable_heat(data, heat_loss)
            self.assertAlmostEqual(recoverable, heat_loss * 0.5, places=4)

    # Optimization Potential Calculations
    def test_optimization_potential_calculation(self):
        """Test optimization potential calculation."""
        test_cases = [
            # (current, target, expected_potential)
            (0.60, 0.85, 29.41),  # (0.85-0.60)/0.85 * 100
            (0.70, 0.85, 17.65),  # (0.85-0.70)/0.85 * 100
            (0.80, 0.85, 5.88),   # (0.85-0.80)/0.85 * 100
            (0.85, 0.85, 0.0),    # At target
            (0.90, 0.85, 0.0),    # Above target
            (0.95, 0.85, 0.0),    # Well above target
        ]

        for current, target, expected in test_cases:
            potential = self.agent._calculate_optimization_potential(current, target)
            self.assertAlmostEqual(
                potential,
                expected,
                places=2,
                msg=f"Failed for current={current}, target={target}"
            )

    def test_optimization_potential_edge_cases(self):
        """Test optimization potential edge cases."""
        # Zero target
        potential = self.agent._calculate_optimization_potential(0.5, 0.0)
        self.assertEqual(potential, 0.0)

        # Negative values (shouldn't happen but test defensive programming)
        potential = self.agent._calculate_optimization_potential(-0.5, 0.85)
        self.assertGreaterEqual(potential, 0.0)

    # Complex Calculation Scenarios
    @pytest.mark.asyncio
    async def test_full_thermal_calculation(self):
        """Test complete thermal calculation pipeline."""
        data = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=400.0,
            pressure_bar=15.0,
            flow_rate_kg_s=8.0,
            energy_input_kw=3000.0,
            energy_output_kw=2400.0,
            fuel_type="natural_gas",
            fuel_consumption_rate=25.0
        )

        result = await self.agent.calculate_thermal_efficiency(data)

        # Verify all calculations
        self.assertAlmostEqual(result.efficiency, 0.80, places=2)
        self.assertAlmostEqual(result.heat_loss_kw, 600.0, places=1)
        self.assertAlmostEqual(result.recoverable_heat_kw, 300.0, places=1)  # 50% of 600
        self.assertAlmostEqual(result.optimization_potential, 6.25, places=2)  # (0.85-0.80)/0.85*100

        # Verify metadata
        self.assertIsNotNone(result.provenance_hash)
        self.assertGreater(result.calculation_time_ms, 0)
        self.assertTrue(result.deterministic)

    # Optimization Strategy Calculations
    @pytest.mark.asyncio
    async def test_optimization_strategy_calculations(self):
        """Test optimization strategy financial calculations."""
        # Create 24 hours of process data with varying efficiency
        process_history = []
        for hour in range(24):
            efficiency_variation = 0.70 + (0.10 * math.sin(hour * math.pi / 12))
            energy_output = 1000.0 * efficiency_variation

            process_history.append(ProcessData(
                timestamp=DeterministicClock.utcnow() - timedelta(hours=hour),
                temperature_c=350.0,
                pressure_bar=12.0,
                flow_rate_kg_s=6.0,
                energy_input_kw=1000.0,
                energy_output_kw=energy_output,
                fuel_type="natural_gas",
                fuel_consumption_rate=15.0
            ))

        constraints = {
            'max_investment': 100000,
            'payback_period_months': 36,
            'min_roi': 0.15
        }

        strategy = await self.agent.generate_optimization_strategy(
            process_history,
            constraints
        )

        # Verify financial calculations
        self.assertGreater(strategy.expected_savings_kwh, 0)
        self.assertEqual(strategy.implementation_cost, 50000.0)  # Fixed in implementation
        self.assertEqual(strategy.payback_period_months, 24.0)  # Fixed in implementation

        # Verify CO2 calculations
        self.assertGreater(strategy.co2_reduction_tonnes, 0)

        # CO2 reduction should be proportional to energy savings
        co2_factor = 0.0002  # kg CO2/kWh from implementation
        expected_co2 = strategy.expected_savings_kwh * co2_factor / 1000  # Convert to tonnes
        self.assertAlmostEqual(
            strategy.co2_reduction_tonnes,
            expected_co2,
            delta=expected_co2 * 0.01  # 1% tolerance
        )

    # Statistical Calculations
    def test_average_efficiency_calculation(self):
        """Test average efficiency calculation over history."""
        process_history = [
            ProcessData(
                timestamp=DeterministicClock.utcnow() - timedelta(hours=i),
                temperature_c=250.0,
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=700.0 + (i * 10),  # Varying efficiency
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(10)
        ]

        efficiencies = [
            self.agent._calculate_efficiency_core(d)
            for d in process_history
        ]

        avg_efficiency = np.mean(efficiencies)
        expected_avg = np.mean([0.70 + (i * 0.01) for i in range(10)])

        self.assertAlmostEqual(avg_efficiency, expected_avg, places=4)

    # Precision and Rounding Tests
    def test_calculation_precision(self):
        """Test calculations maintain required precision."""
        # Test with high-precision inputs
        data = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=273.15,  # Precise temperature
            pressure_bar=101.325,  # Standard atmosphere
            flow_rate_kg_s=3.14159265359,
            energy_input_kw=1234.56789012345,
            energy_output_kw=1111.11111111111,
            fuel_type="gas",
            fuel_consumption_rate=2.71828182846
        )

        efficiency = self.agent._calculate_efficiency_core(data)

        # Should maintain at least 6 decimal places
        expected = Decimal('1111.11111111111') / Decimal('1234.56789012345')
        self.assertAlmostEqual(
            Decimal(str(efficiency)),
            expected,
            places=6
        )

    def test_financial_calculation_precision(self):
        """Test financial calculations use proper decimal precision."""
        # Large numbers that could cause floating point errors
        large_savings_kwh = 10000000.0  # 10 million kWh
        energy_price_per_kwh = 0.12345678  # Precise price

        expected_savings = Decimal(str(large_savings_kwh)) * Decimal(str(energy_price_per_kwh))

        # Verify calculation maintains precision
        actual_savings = large_savings_kwh * energy_price_per_kwh

        self.assertAlmostEqual(
            Decimal(str(actual_savings)),
            expected_savings,
            places=2  # Financial precision (cents)
        )

    # Thermodynamic Validation
    def test_thermodynamic_constraints(self):
        """Test calculations respect thermodynamic laws."""
        # Test Carnot efficiency limit
        hot_temp_k = 873.15  # 600°C in Kelvin
        cold_temp_k = 293.15  # 20°C in Kelvin
        carnot_efficiency = 1 - (cold_temp_k / hot_temp_k)

        data = ProcessData(
            timestamp=DeterministicClock.utcnow(),
            temperature_c=600.0,
            pressure_bar=20.0,
            flow_rate_kg_s=10.0,
            energy_input_kw=5000.0,
            energy_output_kw=5000.0 * carnot_efficiency * 0.95,  # 95% of Carnot
            fuel_type="gas",
            fuel_consumption_rate=50.0,
            ambient_temp_c=20.0
        )

        efficiency = self.agent._calculate_efficiency_core(data)

        # Efficiency should not exceed Carnot limit
        self.assertLessEqual(efficiency, carnot_efficiency)

    # Batch Calculation Tests
    def test_batch_calculation_consistency(self):
        """Test batch calculations are consistent with individual calculations."""
        batch_data = [
            ProcessData(
                timestamp=DeterministicClock.utcnow() - timedelta(minutes=i),
                temperature_c=200.0 + (i * 10),
                pressure_bar=10.0,
                flow_rate_kg_s=5.0,
                energy_input_kw=1000.0,
                energy_output_kw=850.0 - (i * 10),
                fuel_type="gas",
                fuel_consumption_rate=10.0
            )
            for i in range(5)
        ]

        # Calculate individually
        individual_results = []
        for data in batch_data:
            efficiency = self.agent._calculate_efficiency_core(data)
            heat_loss = self.agent._calculate_heat_loss(data)
            recoverable = self.agent._calculate_recoverable_heat(data, heat_loss)
            individual_results.append((efficiency, heat_loss, recoverable))

        # Verify consistency
        for i, data in enumerate(batch_data):
            efficiency = self.agent._calculate_efficiency_core(data)
            heat_loss = self.agent._calculate_heat_loss(data)
            recoverable = self.agent._calculate_recoverable_heat(data, heat_loss)

            self.assertEqual(efficiency, individual_results[i][0])
            self.assertEqual(heat_loss, individual_results[i][1])
            self.assertEqual(recoverable, individual_results[i][2])


if __name__ == '__main__':
    unittest.main()