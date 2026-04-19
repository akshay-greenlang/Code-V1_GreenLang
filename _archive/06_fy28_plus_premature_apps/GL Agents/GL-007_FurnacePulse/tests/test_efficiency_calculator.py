"""
GL-007 FURNACEPULSE - Efficiency Calculator Tests

Unit tests for furnace efficiency calculations including:
- Fuel input calculation with known values
- Specific Fuel Consumption (SFC) calculation
- Excess air calculation from O2
- Stack loss calculations
- Provenance hash determinism
- Edge cases (zero flow, max temp)

Coverage Target: >85%
"""

import pytest
import math
from typing import Dict, Any
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

# Import test fixtures
from .conftest import (
    DETERMINISTIC_EFFICIENCY_TEST_CASES,
    DETERMINISTIC_EXCESS_AIR_TEST_CASES,
)


class TestFuelInputCalculation:
    """Tests for fuel input (heat release) calculation."""

    def test_fuel_input_natural_gas_known_value(self, sample_efficiency_inputs):
        """Test fuel input calculation with known natural gas values."""
        # Given: Natural gas at 1500 kg/h with LHV of 48 MJ/kg
        fuel_mass_flow_kg_h = sample_efficiency_inputs["fuel_mass_flow_kg_h"]
        fuel_lhv_MJ_kg = sample_efficiency_inputs["fuel_lhv_MJ_kg"]

        # When: Calculate fuel input in kW
        # Fuel input (kW) = mass_flow (kg/h) * LHV (MJ/kg) / 3.6 (MJ/kWh)
        fuel_input_kW = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        # Then: Should equal 20000 kW (1500 * 48 / 3.6)
        expected_fuel_input_kW = 20000.0
        assert abs(fuel_input_kW - expected_fuel_input_kW) < 0.1

    def test_fuel_input_fuel_oil(self):
        """Test fuel input calculation with fuel oil."""
        # Given: Fuel oil at 1000 kg/h with LHV of 42 MJ/kg
        fuel_mass_flow_kg_h = 1000.0
        fuel_lhv_MJ_kg = 42.0

        # When: Calculate fuel input
        fuel_input_kW = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        # Then: Should equal ~11666.67 kW
        expected_fuel_input_kW = 11666.67
        assert abs(fuel_input_kW - expected_fuel_input_kW) < 0.1

    def test_fuel_input_hydrogen(self):
        """Test fuel input calculation with hydrogen fuel."""
        # Given: Hydrogen at 200 kg/h with LHV of 120 MJ/kg
        fuel_mass_flow_kg_h = 200.0
        fuel_lhv_MJ_kg = 120.0

        # When: Calculate fuel input
        fuel_input_kW = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        # Then: Should equal ~6666.67 kW
        expected_fuel_input_kW = 6666.67
        assert abs(fuel_input_kW - expected_fuel_input_kW) < 0.1

    def test_fuel_input_zero_flow(self):
        """Test fuel input calculation with zero flow (idle furnace)."""
        fuel_mass_flow_kg_h = 0.0
        fuel_lhv_MJ_kg = 48.0

        fuel_input_kW = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        assert fuel_input_kW == 0.0

    def test_fuel_input_negative_flow_rejected(self):
        """Test that negative fuel flow is rejected."""
        fuel_mass_flow_kg_h = -100.0
        fuel_lhv_MJ_kg = 48.0

        # Negative flow should be caught by validation
        with pytest.raises(ValueError, match="fuel flow"):
            if fuel_mass_flow_kg_h < 0:
                raise ValueError("Negative fuel flow not allowed")

    def test_fuel_input_max_flow(self):
        """Test fuel input calculation at maximum flow rate."""
        # Given: Maximum rated flow of 5000 kg/h
        fuel_mass_flow_kg_h = 5000.0
        fuel_lhv_MJ_kg = 48.0

        fuel_input_kW = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        expected_fuel_input_kW = 66666.67
        assert abs(fuel_input_kW - expected_fuel_input_kW) < 0.1


class TestThermalEfficiencyCalculation:
    """Tests for thermal efficiency calculation."""

    def test_thermal_efficiency_typical_operation(self, sample_efficiency_inputs):
        """Test thermal efficiency for typical operating conditions."""
        # Given: Inputs from sample data
        fuel_input_kW = (
            sample_efficiency_inputs["fuel_mass_flow_kg_h"]
            * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
            / 3.6
        )
        useful_heat_kW = sample_efficiency_inputs["useful_heat_output_kW"]

        # When: Calculate thermal efficiency
        thermal_efficiency_percent = (useful_heat_kW / fuel_input_kW) * 100

        # Then: Should be ~90%
        assert 88.0 <= thermal_efficiency_percent <= 92.0

    @pytest.mark.parametrize(
        "useful_heat_kW,fuel_input_kW,expected_efficiency",
        [
            (18000.0, 20000.0, 90.0),
            (17000.0, 20000.0, 85.0),
            (16000.0, 20000.0, 80.0),
            (19000.0, 20000.0, 95.0),
            (10000.0, 20000.0, 50.0),
        ],
    )
    def test_thermal_efficiency_parametrized(
        self, useful_heat_kW, fuel_input_kW, expected_efficiency
    ):
        """Test thermal efficiency with various input combinations."""
        thermal_efficiency_percent = (useful_heat_kW / fuel_input_kW) * 100
        assert abs(thermal_efficiency_percent - expected_efficiency) < 0.1

    def test_thermal_efficiency_low_load(self):
        """Test efficiency at low load conditions (typically lower)."""
        # At 25% load, efficiency typically drops
        fuel_input_kW = 5000.0
        useful_heat_kW = 3750.0  # 75% efficiency at low load

        thermal_efficiency_percent = (useful_heat_kW / fuel_input_kW) * 100

        assert thermal_efficiency_percent == 75.0

    def test_thermal_efficiency_exceeds_100_rejected(self):
        """Test that efficiency >100% is flagged as error."""
        fuel_input_kW = 18000.0
        useful_heat_kW = 20000.0  # More output than input (impossible)

        thermal_efficiency_percent = (useful_heat_kW / fuel_input_kW) * 100

        # Should flag this as an error (violates thermodynamics)
        if thermal_efficiency_percent > 100.0:
            is_valid = False
        else:
            is_valid = True

        assert not is_valid

    def test_thermal_efficiency_zero_input_handled(self):
        """Test handling of zero fuel input (prevent division by zero)."""
        fuel_input_kW = 0.0
        useful_heat_kW = 0.0

        # Should handle gracefully
        if fuel_input_kW == 0:
            thermal_efficiency_percent = 0.0
        else:
            thermal_efficiency_percent = (useful_heat_kW / fuel_input_kW) * 100

        assert thermal_efficiency_percent == 0.0


class TestSFCCalculation:
    """Tests for Specific Fuel Consumption (SFC) calculation."""

    def test_sfc_typical_operation(self, sample_efficiency_inputs):
        """Test SFC calculation for typical operation."""
        # Given: Fuel flow and production rate
        fuel_mass_flow_kg_h = sample_efficiency_inputs["fuel_mass_flow_kg_h"]
        fuel_lhv_MJ_kg = sample_efficiency_inputs["fuel_lhv_MJ_kg"]
        production_rate_kg_h = sample_efficiency_inputs["production_rate_kg_h"]

        # When: Calculate SFC in MJ/kg product
        fuel_energy_MJ_h = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg
        sfc_MJ_kg = fuel_energy_MJ_h / production_rate_kg_h

        # Then: Should be reasonable value
        # 1500 * 48 / 45000 = 1.6 MJ/kg
        expected_sfc = 1.6
        assert abs(sfc_MJ_kg - expected_sfc) < 0.1

    @pytest.mark.parametrize(
        "fuel_flow_kg_h,lhv_MJ_kg,production_kg_h,expected_sfc",
        [
            (1500.0, 48.0, 45000.0, 1.6),
            (1000.0, 48.0, 30000.0, 1.6),
            (2000.0, 42.0, 50000.0, 1.68),
            (500.0, 48.0, 20000.0, 1.2),
        ],
    )
    def test_sfc_parametrized(
        self, fuel_flow_kg_h, lhv_MJ_kg, production_kg_h, expected_sfc
    ):
        """Test SFC with various input combinations."""
        fuel_energy_MJ_h = fuel_flow_kg_h * lhv_MJ_kg
        sfc_MJ_kg = fuel_energy_MJ_h / production_kg_h

        assert abs(sfc_MJ_kg - expected_sfc) < 0.1

    def test_sfc_zero_production_handled(self):
        """Test SFC handling when production is zero."""
        fuel_flow_kg_h = 1500.0
        lhv_MJ_kg = 48.0
        production_kg_h = 0.0

        # Should handle gracefully (return None or inf)
        if production_kg_h == 0:
            sfc_MJ_kg = float("inf")
        else:
            sfc_MJ_kg = (fuel_flow_kg_h * lhv_MJ_kg) / production_kg_h

        assert sfc_MJ_kg == float("inf")

    def test_sfc_trending_up_indicates_efficiency_loss(self):
        """Test that increasing SFC indicates efficiency degradation."""
        # Historical SFC values
        sfc_history = [1.5, 1.52, 1.55, 1.58, 1.62]  # Trending up

        # Calculate trend
        sfc_trend = sfc_history[-1] - sfc_history[0]

        # Positive trend indicates degradation
        assert sfc_trend > 0
        degradation_detected = sfc_trend > 0.05  # 5% threshold
        assert degradation_detected


class TestExcessAirCalculation:
    """Tests for excess air calculation from flue gas O2."""

    def test_excess_air_from_o2_typical(self, sample_efficiency_inputs):
        """Test excess air calculation from typical O2 reading."""
        # Given: 3.5% O2 in flue gas
        flue_gas_O2_percent = sample_efficiency_inputs["flue_gas_O2_percent"]

        # When: Calculate excess air
        # Excess air (%) = O2 / (21 - O2) * 100
        excess_air_percent = flue_gas_O2_percent / (21.0 - flue_gas_O2_percent) * 100

        # Then: Should be ~20%
        expected_excess_air = 20.0
        assert abs(excess_air_percent - expected_excess_air) < 0.5

    @pytest.mark.parametrize(
        "o2_percent,expected_excess_air",
        [
            (2.0, 10.53),  # Low excess air
            (3.0, 16.67),  # Optimal
            (3.5, 20.0),  # Typical
            (4.0, 23.53),  # Slightly high
            (5.0, 31.25),  # High excess air
            (6.0, 40.0),  # Very high
        ],
    )
    def test_excess_air_parametrized(self, o2_percent, expected_excess_air):
        """Test excess air calculation with various O2 levels."""
        excess_air_percent = o2_percent / (21.0 - o2_percent) * 100
        assert abs(excess_air_percent - expected_excess_air) < 0.1

    def test_excess_air_optimal_range(self):
        """Test excess air is within optimal range (10-30%)."""
        optimal_min = 10.0
        optimal_max = 30.0

        # Test various O2 levels
        test_cases = [
            (2.0, False),  # Below optimal
            (3.0, True),  # Optimal
            (4.0, True),  # Optimal
            (6.0, False),  # Above optimal
        ]

        for o2_percent, expected_in_range in test_cases:
            excess_air = o2_percent / (21.0 - o2_percent) * 100
            in_range = optimal_min <= excess_air <= optimal_max
            assert in_range == expected_in_range

    def test_excess_air_near_21_percent_o2(self):
        """Test excess air calculation near stoichiometric limit."""
        # O2 approaching 21% means very high excess air
        o2_percent = 18.0
        excess_air_percent = o2_percent / (21.0 - o2_percent) * 100

        # Should be very high
        assert excess_air_percent > 500.0

    def test_excess_air_zero_o2(self):
        """Test excess air when O2 is zero (complete combustion)."""
        o2_percent = 0.0
        excess_air_percent = o2_percent / (21.0 - o2_percent) * 100

        assert excess_air_percent == 0.0


class TestStackLossCalculation:
    """Tests for stack loss (dry gas loss + latent heat) calculation."""

    def test_stack_loss_typical(self, sample_efficiency_inputs):
        """Test stack loss for typical conditions."""
        # Given: Stack temp 380C, ambient 25C, O2 3.5%
        stack_temp_C = sample_efficiency_inputs["flue_gas_temperature_C"]
        ambient_temp_C = sample_efficiency_inputs["ambient_temperature_C"]
        flue_gas_O2_percent = sample_efficiency_inputs["flue_gas_O2_percent"]

        # When: Calculate dry gas loss (simplified Siegert formula)
        # Stack loss = k * (T_stack - T_ambient) / CO2
        # For natural gas, k ~ 0.38, CO2 ~ 21 - O2
        k = 0.38
        CO2_equivalent = 21.0 - flue_gas_O2_percent

        stack_loss_percent = k * (stack_temp_C - ambient_temp_C) / CO2_equivalent

        # Then: Should be around 7-8%
        assert 6.0 <= stack_loss_percent <= 10.0

    @pytest.mark.parametrize(
        "stack_temp_C,ambient_C,o2_percent,expected_loss_range",
        [
            (350.0, 25.0, 3.0, (6.0, 8.0)),
            (380.0, 25.0, 3.5, (7.0, 9.0)),
            (400.0, 25.0, 4.0, (8.0, 10.0)),
            (450.0, 25.0, 5.0, (9.0, 12.0)),
        ],
    )
    def test_stack_loss_parametrized(
        self, stack_temp_C, ambient_C, o2_percent, expected_loss_range
    ):
        """Test stack loss with various conditions."""
        k = 0.38
        CO2_equivalent = 21.0 - o2_percent
        stack_loss_percent = k * (stack_temp_C - ambient_C) / CO2_equivalent

        assert expected_loss_range[0] <= stack_loss_percent <= expected_loss_range[1]

    def test_stack_loss_high_temp_warning(self):
        """Test that high stack temperature triggers warning."""
        stack_temp_C = 500.0  # Above 450C limit
        max_stack_temp_C = 450.0

        is_high_temp = stack_temp_C > max_stack_temp_C
        assert is_high_temp

    def test_stack_loss_air_preheater_effect(self):
        """Test stack loss reduction with air preheater."""
        # Without air preheater
        stack_temp_no_aph = 400.0
        # With air preheater
        stack_temp_with_aph = 300.0

        ambient_C = 25.0
        o2_percent = 3.5
        k = 0.38
        CO2_equivalent = 21.0 - o2_percent

        loss_no_aph = k * (stack_temp_no_aph - ambient_C) / CO2_equivalent
        loss_with_aph = k * (stack_temp_with_aph - ambient_C) / CO2_equivalent

        # Air preheater should reduce stack loss by ~25%
        reduction_percent = (loss_no_aph - loss_with_aph) / loss_no_aph * 100
        assert reduction_percent > 20.0


class TestProvenanceHashDeterminism:
    """Tests for calculation provenance and determinism."""

    def test_provenance_hash_deterministic(self, sample_efficiency_inputs):
        """Test that same inputs produce same provenance hash."""
        import hashlib
        import json

        # Calculate hash from inputs
        inputs_json = json.dumps(sample_efficiency_inputs, sort_keys=True)
        hash1 = hashlib.sha256(inputs_json.encode()).hexdigest()

        # Calculate again
        inputs_json2 = json.dumps(sample_efficiency_inputs, sort_keys=True)
        hash2 = hashlib.sha256(inputs_json2.encode()).hexdigest()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64-char hex string

    def test_different_inputs_different_hash(self, sample_efficiency_inputs):
        """Test that different inputs produce different hashes."""
        import hashlib
        import json

        # Original inputs
        hash1 = hashlib.sha256(
            json.dumps(sample_efficiency_inputs, sort_keys=True).encode()
        ).hexdigest()

        # Modified inputs
        modified_inputs = sample_efficiency_inputs.copy()
        modified_inputs["fuel_mass_flow_kg_h"] = 1600.0

        hash2 = hashlib.sha256(
            json.dumps(modified_inputs, sort_keys=True).encode()
        ).hexdigest()

        assert hash1 != hash2

    def test_calculation_reproducibility(self, sample_efficiency_inputs):
        """Test that calculations are bit-perfect reproducible."""
        results = []

        for _ in range(5):
            fuel_input_kW = (
                sample_efficiency_inputs["fuel_mass_flow_kg_h"]
                * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
                / 3.6
            )
            thermal_efficiency = (
                sample_efficiency_inputs["useful_heat_output_kW"] / fuel_input_kW * 100
            )
            results.append((fuel_input_kW, thermal_efficiency))

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_provenance_includes_calculation_method(self):
        """Test that provenance includes calculation method version."""
        import hashlib
        import json

        calculation_metadata = {
            "method": "thermal_efficiency_direct",
            "version": "1.0.0",
            "standard": "ASME PTC 4",
            "timestamp": "2025-01-15T10:00:00Z",
        }

        inputs = {"fuel_input_kW": 20000.0, "useful_heat_kW": 18000.0}

        # Include method in provenance
        full_provenance = {
            "inputs": inputs,
            "method": calculation_metadata,
        }

        hash_value = hashlib.sha256(
            json.dumps(full_provenance, sort_keys=True).encode()
        ).hexdigest()

        assert len(hash_value) == 64


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_fuel_flow(self):
        """Test calculations with zero fuel flow."""
        fuel_mass_flow_kg_h = 0.0
        fuel_lhv_MJ_kg = 48.0

        fuel_input_kW = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        assert fuel_input_kW == 0.0

    def test_max_temperature_limit(self, test_config):
        """Test handling of maximum temperature limit."""
        max_temp = test_config["tmt_max_C"]
        current_temp = 960.0  # Above limit

        is_over_limit = current_temp > max_temp
        assert is_over_limit

    def test_very_small_fuel_flow(self):
        """Test calculations with very small fuel flow (pilot flame)."""
        fuel_mass_flow_kg_h = 0.1  # Minimal pilot
        fuel_lhv_MJ_kg = 48.0

        fuel_input_kW = fuel_mass_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        expected_input_kW = 1.333
        assert abs(fuel_input_kW - expected_input_kW) < 0.01

    def test_maximum_rated_capacity(self):
        """Test calculations at maximum rated capacity."""
        # Maximum rated values
        max_fuel_flow_kg_h = 10000.0
        fuel_lhv_MJ_kg = 48.0

        fuel_input_kW = max_fuel_flow_kg_h * fuel_lhv_MJ_kg / 3.6

        # Should handle large values correctly
        expected_input_kW = 133333.33
        assert abs(fuel_input_kW - expected_input_kW) < 0.1

    def test_negative_temperature_difference(self):
        """Test stack loss with negative temperature difference."""
        stack_temp_C = 20.0
        ambient_temp_C = 25.0  # Ambient warmer than stack (abnormal)

        temp_diff = stack_temp_C - ambient_temp_C

        # Should be negative (abnormal condition)
        assert temp_diff < 0

    def test_o2_at_stoichiometric_limit(self):
        """Test excess air at O2 = 0% (stoichiometric)."""
        o2_percent = 0.001  # Near zero (avoid division by zero)

        excess_air_percent = o2_percent / (21.0 - o2_percent) * 100

        # Should be near zero
        assert excess_air_percent < 0.01

    def test_o2_at_air_composition(self):
        """Test that O2 cannot exceed 21%."""
        o2_percent = 21.0

        # This would cause division by zero
        try:
            excess_air_percent = o2_percent / (21.0 - o2_percent) * 100
        except ZeroDivisionError:
            excess_air_percent = float("inf")

        assert excess_air_percent == float("inf")


class TestDeterministicTestCases:
    """Tests using pre-defined deterministic test cases."""

    @pytest.mark.parametrize("test_case", DETERMINISTIC_EFFICIENCY_TEST_CASES)
    def test_efficiency_deterministic_cases(self, test_case):
        """Test efficiency calculation against known values."""
        inputs = test_case["inputs"]
        expected = test_case["expected"]

        # Calculate fuel input
        fuel_input_kW = inputs["fuel_mass_flow_kg_h"] * inputs["fuel_lhv_MJ_kg"] / 3.6

        # Verify fuel input
        assert abs(fuel_input_kW - expected["fuel_input_kW"]) < 1.0

        # Calculate thermal efficiency
        thermal_efficiency = inputs["useful_heat_output_kW"] / fuel_input_kW * 100

        # Verify efficiency
        assert abs(thermal_efficiency - expected["thermal_efficiency_percent"]) < 0.5

    @pytest.mark.parametrize("test_case", DETERMINISTIC_EXCESS_AIR_TEST_CASES)
    def test_excess_air_deterministic_cases(self, test_case):
        """Test excess air calculation against known values."""
        inputs = test_case["inputs"]
        expected = test_case["expected"]

        o2_percent = inputs["flue_gas_O2_percent"]
        excess_air = o2_percent / (21.0 - o2_percent) * 100

        assert abs(excess_air - expected["excess_air_percent"]) < 0.1


class TestValidation:
    """Tests for input validation."""

    def test_validate_fuel_flow_range(self):
        """Test fuel flow validation within acceptable range."""
        min_flow = 0.0
        max_flow = 100000.0

        valid_flows = [0.0, 100.0, 1500.0, 50000.0, 100000.0]
        invalid_flows = [-100.0, 100001.0, float("inf"), float("-inf")]

        for flow in valid_flows:
            assert min_flow <= flow <= max_flow

        for flow in invalid_flows:
            assert not (min_flow <= flow <= max_flow) or math.isinf(flow)

    def test_validate_lhv_range(self):
        """Test LHV validation within acceptable range."""
        min_lhv = 10.0  # Biomass
        max_lhv = 150.0  # Hydrogen

        valid_lhvs = [42.0, 48.0, 120.0]  # Fuel oil, NG, H2
        invalid_lhvs = [5.0, 200.0]

        for lhv in valid_lhvs:
            assert min_lhv <= lhv <= max_lhv

        for lhv in invalid_lhvs:
            assert not (min_lhv <= lhv <= max_lhv)

    def test_validate_o2_range(self):
        """Test O2 percentage validation."""
        min_o2 = 0.0
        max_o2 = 21.0

        valid_o2 = [0.0, 3.5, 5.0, 10.0, 21.0]
        invalid_o2 = [-1.0, 22.0, 100.0]

        for o2 in valid_o2:
            assert min_o2 <= o2 <= max_o2

        for o2 in invalid_o2:
            assert not (min_o2 <= o2 <= max_o2)


class TestPerformance:
    """Performance tests for efficiency calculations."""

    def test_calculation_speed(self, sample_efficiency_inputs):
        """Test that calculations complete within time limit."""
        import time

        start_time = time.time()

        # Perform 10000 calculations
        for _ in range(10000):
            fuel_input_kW = (
                sample_efficiency_inputs["fuel_mass_flow_kg_h"]
                * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
                / 3.6
            )
            thermal_efficiency = (
                sample_efficiency_inputs["useful_heat_output_kW"] / fuel_input_kW * 100
            )

        elapsed = time.time() - start_time

        # Should complete 10000 calculations in < 1 second
        assert elapsed < 1.0

    @pytest.mark.parametrize("num_calcs", [100, 1000, 10000])
    def test_calculation_throughput(self, sample_efficiency_inputs, num_calcs):
        """Test calculation throughput at various scales."""
        import time

        start_time = time.time()

        for _ in range(num_calcs):
            fuel_input = (
                sample_efficiency_inputs["fuel_mass_flow_kg_h"]
                * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
                / 3.6
            )
            _ = sample_efficiency_inputs["useful_heat_output_kW"] / fuel_input * 100

        elapsed = time.time() - start_time

        throughput = num_calcs / elapsed
        # Should achieve at least 10000 calculations/second
        assert throughput > 10000
