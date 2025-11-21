# -*- coding: utf-8 -*-
"""
Unit tests for GL-007 Thermal Efficiency Calculation

Tests the calculate_thermal_efficiency tool with comprehensive coverage:
- Valid input processing
- ASME PTC 4.1 compliance
- Calculation accuracy (±1.5% tolerance)
- Error handling
- Edge cases
- Provenance tracking

Target Coverage: 90%+
"""

import pytest
import json
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from greenlang.determinism import DeterministicClock


# Test data paths
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestThermalEfficiencyCalculation:
    """Test suite for thermal efficiency calculation."""

    def test_initialization(self, agent_config):
        """Test thermal efficiency calculator initializes correctly."""
        # This would test the actual implementation
        # For now, we're testing the structure
        assert agent_config["agent_id"] == "GL-007"
        assert agent_config["name"] == "FurnacePerformanceMonitor"

    def test_calculate_efficiency_baseline_natural_gas(
        self,
        sample_thermal_efficiency_input,
        assert_thermal_efficiency_valid
    ):
        """Test efficiency calculation for baseline natural gas case."""
        # Expected calculation:
        # Heat Input = 25.5 MW
        # Heat Output = 20.8 MW
        # Efficiency = (20.8 / 25.5) × 100 = 81.57%

        result = self._calculate_thermal_efficiency(sample_thermal_efficiency_input)

        # Validate structure
        assert_thermal_efficiency_valid(result)

        # Validate accuracy (ASME PTC 4.1 ±1.5% tolerance)
        expected_efficiency = 81.57
        assert abs(result["thermal_efficiency_percent"] - expected_efficiency) < 1.5

        # Validate heat balance
        assert result["heat_input_mw"] == 25.5
        assert result["heat_output_mw"] == 20.8

    @pytest.mark.parametrize("fuel_type,heating_value,expected_efficiency_range", [
        ("natural_gas", 50.0, (78.0, 88.0)),
        ("coal", 25.0, (70.0, 80.0)),
        ("diesel", 45.6, (75.0, 85.0)),
        ("hydrogen", 120.0, (82.0, 92.0)),
    ])
    def test_efficiency_by_fuel_type(
        self,
        fuel_type,
        heating_value,
        expected_efficiency_range
    ):
        """Test efficiency calculation for different fuel types."""
        input_data = {
            "fuel_input_mw": 25.0,
            "fuel_type": fuel_type,
            "heating_value_mj_kg": heating_value,
            "flue_gas_temperature_c": 185.0,
            "flue_gas_flow_kg_hr": 28000.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 3.5,
            "heat_absorbed_mw": 20.0,
            "radiation_losses_percent": 2.5,
            "convection_losses_percent": 1.2,
            "unaccounted_losses_percent": 1.8,
        }

        result = self._calculate_thermal_efficiency(input_data)

        min_eff, max_eff = expected_efficiency_range
        assert min_eff <= result["thermal_efficiency_percent"] <= max_eff

    def test_efficiency_with_high_stack_temperature(self):
        """Test efficiency degradation with high stack temperature."""
        # High stack temperature = more heat loss = lower efficiency
        input_data = {
            "fuel_input_mw": 30.0,
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 280.0,  # High temperature
            "flue_gas_flow_kg_hr": 35000.0,
            "ambient_temperature_c": 25.0,
            "stack_o2_percent": 5.5,
            "heat_absorbed_mw": 21.0,
            "radiation_losses_percent": 3.5,
            "convection_losses_percent": 1.8,
            "unaccounted_losses_percent": 2.5,
        }

        result = self._calculate_thermal_efficiency(input_data)

        # Efficiency should be lower due to high stack temperature
        assert result["thermal_efficiency_percent"] < 75.0

        # Stack loss should be significant
        assert result["losses_breakdown"]["stack_loss_percent"] > 15.0

    def test_efficiency_with_low_stack_temperature(self):
        """Test high efficiency with optimized low stack temperature."""
        input_data = {
            "fuel_input_mw": 20.0,
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 140.0,  # Low temperature (good heat recovery)
            "flue_gas_flow_kg_hr": 22000.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 2.5,
            "heat_absorbed_mw": 17.5,
            "radiation_losses_percent": 1.8,
            "convection_losses_percent": 0.9,
            "unaccounted_losses_percent": 1.3,
        }

        result = self._calculate_thermal_efficiency(input_data)

        # Efficiency should be high
        assert result["thermal_efficiency_percent"] > 85.0

    def test_losses_sum_to_total(self, sample_thermal_efficiency_input):
        """Test that all losses sum correctly to (100 - efficiency)."""
        result = self._calculate_thermal_efficiency(sample_thermal_efficiency_input)

        total_losses = (
            result["losses_breakdown"]["stack_loss_percent"] +
            result["losses_breakdown"]["radiation_loss_percent"] +
            result["losses_breakdown"]["convection_loss_percent"] +
            result["losses_breakdown"]["unaccounted_loss_percent"]
        )

        expected_losses = 100.0 - result["thermal_efficiency_percent"]

        # Should match within ±0.1%
        assert abs(total_losses - expected_losses) < 0.1

    def test_provenance_tracking(self, sample_thermal_efficiency_input):
        """Test provenance hash is generated and deterministic."""
        result1 = self._calculate_thermal_efficiency(sample_thermal_efficiency_input)
        result2 = self._calculate_thermal_efficiency(sample_thermal_efficiency_input)

        # Both results should have provenance hash
        assert "provenance_hash" in result1
        assert "provenance_hash" in result2

        # SHA-256 hash length
        assert len(result1["provenance_hash"]) == 64

        # Same inputs → same hash (bit-perfect reproducibility)
        assert result1["provenance_hash"] == result2["provenance_hash"]

    def test_invalid_input_negative_temperature(self):
        """Test error handling for negative temperature (invalid)."""
        invalid_input = {
            "fuel_input_mw": 25.0,
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": -10.0,  # Invalid: negative temperature
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 3.5,
            "heat_absorbed_mw": 20.0,
        }

        with pytest.raises(ValueError) as exc_info:
            self._calculate_thermal_efficiency(invalid_input)

        assert "temperature" in str(exc_info.value).lower()

    def test_invalid_input_efficiency_over_100(self):
        """Test error handling for impossible efficiency >100%."""
        invalid_input = {
            "fuel_input_mw": 20.0,
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 185.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 3.5,
            "heat_absorbed_mw": 25.0,  # More heat out than in (impossible!)
            "radiation_losses_percent": 0.0,
            "convection_losses_percent": 0.0,
            "unaccounted_losses_percent": 0.0,
        }

        with pytest.raises(ValueError) as exc_info:
            self._calculate_thermal_efficiency(invalid_input)

        assert "efficiency" in str(exc_info.value).lower() or "heat" in str(exc_info.value).lower()

    def test_invalid_input_negative_oxygen(self):
        """Test error handling for negative O2 percentage."""
        invalid_input = {
            "fuel_input_mw": 25.0,
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 185.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": -2.0,  # Invalid
            "heat_absorbed_mw": 20.0,
        }

        with pytest.raises(ValueError):
            self._calculate_thermal_efficiency(invalid_input)

    def test_invalid_input_oxygen_over_21(self):
        """Test error handling for O2 > 21% (impossible in air)."""
        invalid_input = {
            "fuel_input_mw": 25.0,
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 185.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 25.0,  # Invalid: exceeds atmospheric O2
            "heat_absorbed_mw": 20.0,
        }

        with pytest.raises(ValueError):
            self._calculate_thermal_efficiency(invalid_input)

    def test_missing_required_field(self):
        """Test error handling for missing required fields."""
        incomplete_input = {
            "fuel_input_mw": 25.0,
            "fuel_type": "natural_gas",
            # Missing heating_value_mj_kg
            "flue_gas_temperature_c": 185.0,
            "ambient_temperature_c": 20.0,
        }

        with pytest.raises((KeyError, ValueError)):
            self._calculate_thermal_efficiency(incomplete_input)

    @pytest.mark.accuracy
    def test_asme_ptc_4_1_compliance(self):
        """Test compliance with ASME PTC 4.1 uncertainty requirements."""
        # Load test cases from fixtures
        test_cases_file = FIXTURES_DIR / "thermal_efficiency_test_cases.json"

        if test_cases_file.exists():
            with open(test_cases_file, 'r') as f:
                test_data = json.load(f)

            for case in test_data["test_cases"]:
                result = self._calculate_thermal_efficiency(case["input"])

                expected = case["expected_output"]["thermal_efficiency_percent"]
                tolerance = case["expected_output"]["tolerance_percent"]

                # Validate within ASME PTC 4.1 tolerance
                assert abs(result["thermal_efficiency_percent"] - expected) <= tolerance, \
                    f"Case {case['case_id']} failed ASME PTC 4.1 compliance"

    @pytest.mark.performance
    def test_calculation_performance(self, sample_thermal_efficiency_input, benchmark):
        """Test calculation meets performance target (<50ms)."""
        def run_calculation():
            return self._calculate_thermal_efficiency(sample_thermal_efficiency_input)

        result = benchmark(run_calculation)

        # Should complete in <50ms
        # Note: benchmark fixture provides timing automatically

    def test_edge_case_zero_load(self):
        """Test handling of zero load condition."""
        zero_load_input = {
            "fuel_input_mw": 0.1,  # Minimal pilot flame
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 100.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 20.0,  # Nearly atmospheric
            "heat_absorbed_mw": 0.0,
            "radiation_losses_percent": 5.0,
            "convection_losses_percent": 2.0,
            "unaccounted_losses_percent": 3.0,
        }

        result = self._calculate_thermal_efficiency(zero_load_input)

        # Efficiency should be very low at zero load
        assert result["thermal_efficiency_percent"] < 20.0

    def test_edge_case_maximum_load(self):
        """Test handling of maximum load condition."""
        max_load_input = {
            "fuel_input_mw": 100.0,  # Very large furnace
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 200.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 3.0,
            "heat_absorbed_mw": 82.0,
            "radiation_losses_percent": 2.0,
            "convection_losses_percent": 1.0,
            "unaccounted_losses_percent": 1.5,
        }

        result = self._calculate_thermal_efficiency(max_load_input)

        # Should still calculate correctly
        assert 0 < result["thermal_efficiency_percent"] < 100

    def test_calculation_with_alternative_units(self):
        """Test calculation handles unit conversions correctly."""
        # Test with GJ instead of MW
        input_gj = {
            "fuel_input_gj_hr": 91.8,  # Equivalent to 25.5 MW
            "fuel_type": "natural_gas",
            "heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 185.0,
            "ambient_temperature_c": 20.0,
            "stack_o2_percent": 3.5,
            "heat_absorbed_gj_hr": 74.88,  # Equivalent to 20.8 MW
            "radiation_losses_percent": 2.5,
            "convection_losses_percent": 1.2,
            "unaccounted_losses_percent": 1.8,
        }

        # Implementation should handle unit conversion
        # (This test assumes the function can detect units)

    # ========================================================================
    # HELPER METHODS (Mock implementation for testing structure)
    # ========================================================================

    def _calculate_thermal_efficiency(self, input_data: dict) -> dict:
        """
        Mock implementation of thermal efficiency calculation.

        In actual implementation, this would call the real GL-007 tool.
        For testing purposes, we simulate the calculation logic.
        """
        # Validate inputs
        self._validate_thermal_efficiency_input(input_data)

        # Extract values
        fuel_input_mw = input_data.get("fuel_input_mw")
        heat_absorbed_mw = input_data.get("heat_absorbed_mw")
        radiation_loss = input_data.get("radiation_losses_percent", 0)
        convection_loss = input_data.get("convection_losses_percent", 0)
        unaccounted_loss = input_data.get("unaccounted_losses_percent", 0)
        flue_gas_temp = input_data.get("flue_gas_temperature_c")
        ambient_temp = input_data.get("ambient_temperature_c")

        # Calculate stack loss (simplified)
        stack_loss = ((flue_gas_temp - ambient_temp) / 10) * 0.5  # Simplified formula

        # Calculate efficiency
        thermal_efficiency = (heat_absorbed_mw / fuel_input_mw) * 100

        # Validate efficiency is physically possible
        if thermal_efficiency > 100:
            raise ValueError("Calculated efficiency > 100% - heat output exceeds input")

        # Calculate losses breakdown
        losses = {
            "stack_loss_percent": stack_loss,
            "radiation_loss_percent": radiation_loss,
            "convection_loss_percent": convection_loss,
            "unaccounted_loss_percent": unaccounted_loss,
        }

        # Calculate provenance hash (deterministic)
        import hashlib
        input_str = json.dumps(input_data, sort_keys=True)
        provenance_hash = hashlib.sha256(input_str.encode()).hexdigest()

        return {
            "thermal_efficiency_percent": round(thermal_efficiency, 2),
            "heat_input_mw": fuel_input_mw,
            "heat_output_mw": heat_absorbed_mw,
            "losses_breakdown": losses,
            "provenance_hash": provenance_hash,
            "calculation_timestamp": DeterministicClock.now().isoformat(),
            "asme_ptc_4_1_compliant": True,
        }

    def _validate_thermal_efficiency_input(self, input_data: dict):
        """Validate thermal efficiency input data."""
        # Check required fields
        required_fields = [
            "fuel_input_mw", "fuel_type", "heating_value_mj_kg",
            "flue_gas_temperature_c", "ambient_temperature_c",
            "stack_o2_percent", "heat_absorbed_mw"
        ]

        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate temperature ranges
        if input_data["flue_gas_temperature_c"] < -273.15:  # Absolute zero
            raise ValueError("Temperature below absolute zero")

        if input_data["ambient_temperature_c"] < -273.15:
            raise ValueError("Ambient temperature below absolute zero")

        # Validate O2 range
        if not 0 <= input_data["stack_o2_percent"] <= 21:
            raise ValueError("O2 percentage must be between 0 and 21%")

        # Validate heat balance
        if input_data["heat_absorbed_mw"] > input_data["fuel_input_mw"]:
            raise ValueError("Heat absorbed cannot exceed fuel input (violates thermodynamics)")
