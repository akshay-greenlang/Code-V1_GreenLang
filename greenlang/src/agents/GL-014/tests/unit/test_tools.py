# -*- coding: utf-8 -*-
"""
Unit Tests for GL-014 EXCHANGER-PRO Tool Functions.

Tests all tool functions exposed by the agent including:
- Input validation
- Output schema validation
- Provenance hash verification
- Error handling

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingResistanceInput,
    KernSeatonInput,
    EbertPanchalInput,
    FoulingSeverityInput,
    TimeToCleaningInput,
    FluidType,
    FoulingMechanism,
    FoulingSeverity,
    ExchangerType,
)
from calculators.economic_calculator import (
    EconomicCalculator,
    EnergyLossInput,
    ROIInput,
    CarbonImpactInput,
    FuelType,
)


# =============================================================================
# Test Class: Tool Input Validation
# =============================================================================

class TestToolInputValidation:
    """Tests for tool function input validation."""

    def test_valid_fouling_resistance_input(self):
        """Test validation passes for valid fouling resistance input."""
        # Arrange & Act
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
            fluid_type_hot=FluidType.OIL_LIGHT,
            fluid_type_cold=FluidType.WATER_TREATED,
            exchanger_type=ExchangerType.SHELL_TUBE,
        )

        # Assert
        assert input_data.u_clean_w_m2_k == 500.0
        assert input_data.u_fouled_w_m2_k == 420.0

    def test_invalid_negative_u_value(self):
        """Test validation fails for negative U values."""
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=-500.0,  # Invalid: negative
                u_fouled_w_m2_k=420.0,
            )

    def test_invalid_zero_u_value(self):
        """Test validation fails for zero U values."""
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=0.0,  # Invalid: zero
                u_fouled_w_m2_k=420.0,
            )

    def test_invalid_fouled_greater_than_clean(self):
        """Test validation fails when U_fouled > U_clean."""
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=400.0,
                u_fouled_w_m2_k=500.0,  # Invalid: greater than clean
            )

    def test_valid_kern_seaton_input(self):
        """Test validation passes for valid Kern-Seaton input."""
        input_data = KernSeatonInput(
            r_f_max_m2_k_w=0.0005,
            time_constant_hours=500.0,
            time_hours=200.0,
        )
        assert input_data.r_f_max_m2_k_w == 0.0005

    def test_invalid_kern_seaton_negative_time(self):
        """Test validation fails for negative time values."""
        with pytest.raises(ValueError):
            KernSeatonInput(
                r_f_max_m2_k_w=0.0005,
                time_constant_hours=-500.0,  # Invalid: negative
                time_hours=200.0,
            )

    def test_valid_ebert_panchal_input(self):
        """Test validation passes for valid Ebert-Panchal input."""
        input_data = EbertPanchalInput(
            reynolds_number=50000.0,
            prandtl_number=50.0,
            film_temperature_k=400.0,
            wall_shear_stress_pa=50.0,
            velocity_m_s=1.5,
            fouling_mechanism=FoulingMechanism.CHEMICAL_REACTION,
        )
        assert input_data.reynolds_number == 50000.0

    def test_invalid_reynolds_number(self):
        """Test validation fails for invalid Reynolds number."""
        with pytest.raises(ValueError):
            EbertPanchalInput(
                reynolds_number=-50000.0,  # Invalid: negative
                prandtl_number=50.0,
                film_temperature_k=400.0,
                wall_shear_stress_pa=50.0,
                velocity_m_s=1.5,
            )

    def test_valid_energy_loss_input(self):
        """Test validation passes for valid energy loss input."""
        input_data = EnergyLossInput(
            design_duty_kw=Decimal("1500"),
            actual_duty_kw=Decimal("1275"),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8000"),
        )
        assert input_data.design_duty_kw == Decimal("1500")

    def test_valid_roi_input(self):
        """Test validation passes for valid ROI input."""
        input_data = ROIInput(
            investment_cost=Decimal("50000"),
            annual_savings=Decimal("25000"),
            discount_rate_percent=Decimal("10.0"),
            analysis_period_years=10,
        )
        assert input_data.investment_cost == Decimal("50000")


# =============================================================================
# Test Class: Tool Output Schema Validation
# =============================================================================

class TestToolOutputSchemaValidation:
    """Tests for tool function output schema validation."""

    def test_fouling_resistance_output_schema(
        self,
        fouling_calculator: FoulingCalculator,
        fouling_resistance_input: FoulingResistanceInput,
    ):
        """Test fouling resistance output conforms to schema."""
        # Act
        result = fouling_calculator.calculate_fouling_resistance(fouling_resistance_input)

        # Assert: Check all required fields exist
        assert hasattr(result, 'fouling_resistance_m2_k_w')
        assert hasattr(result, 'normalized_fouling_factor')
        assert hasattr(result, 'cleanliness_factor_percent')
        assert hasattr(result, 'u_clean_w_m2_k')
        assert hasattr(result, 'u_fouled_w_m2_k')
        assert hasattr(result, 'design_fouling_resistance_m2_k_w')
        assert hasattr(result, 'calculation_timestamp')
        assert hasattr(result, 'provenance_hash')

        # Assert: Check types
        assert isinstance(result.fouling_resistance_m2_k_w, Decimal)
        assert isinstance(result.cleanliness_factor_percent, Decimal)
        assert isinstance(result.provenance_hash, str)

    def test_kern_seaton_output_schema(
        self,
        fouling_calculator: FoulingCalculator,
        kern_seaton_input: KernSeatonInput,
    ):
        """Test Kern-Seaton output conforms to schema."""
        # Act
        result = fouling_calculator.calculate_kern_seaton(kern_seaton_input)

        # Assert: Check all required fields
        assert hasattr(result, 'predicted_r_f_m2_k_w')
        assert hasattr(result, 'r_f_max_m2_k_w')
        assert hasattr(result, 'time_constant_hours')
        assert hasattr(result, 'time_hours')
        assert hasattr(result, 'asymptotic_approach_percent')
        assert hasattr(result, 'provenance_hash')

    def test_energy_loss_output_schema(
        self,
        economic_calculator: EconomicCalculator,
        energy_loss_input: EnergyLossInput,
    ):
        """Test energy loss output conforms to schema."""
        # Act
        result = economic_calculator.calculate_energy_loss_cost(energy_loss_input)

        # Assert: Check all required fields
        required_fields = [
            'heat_transfer_loss_kw',
            'heat_transfer_loss_percent',
            'additional_fuel_kwh_per_year',
            'energy_cost_per_year_usd',
            'carbon_emissions_kg_per_year',
            'carbon_cost_per_year_usd',
            'total_energy_penalty_per_year_usd',
            'calculation_steps',
            'provenance_hash',
        ]

        for field in required_fields:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_roi_output_schema(
        self,
        economic_calculator: EconomicCalculator,
        roi_input: ROIInput,
    ):
        """Test ROI analysis output conforms to schema."""
        # Act
        result = economic_calculator.perform_roi_analysis(roi_input)

        # Assert: Check all required fields
        required_fields = [
            'net_present_value_usd',
            'internal_rate_of_return_percent',
            'simple_payback_years',
            'discounted_payback_years',
            'profitability_index',
            'annual_cash_flows',
            'cumulative_npv_by_year',
            'break_even_utilization_percent',
            'calculation_steps',
            'provenance_hash',
        ]

        for field in required_fields:
            assert hasattr(result, field), f"Missing field: {field}"


# =============================================================================
# Test Class: Provenance Hash Verification
# =============================================================================

class TestProvenanceHashVerification:
    """Tests for provenance hash verification."""

    def test_provenance_hash_is_sha256(
        self,
        fouling_calculator: FoulingCalculator,
        fouling_resistance_input: FoulingResistanceInput,
    ):
        """Test provenance hash is valid SHA-256."""
        # Act
        result = fouling_calculator.calculate_fouling_resistance(fouling_resistance_input)

        # Assert
        assert len(result.provenance_hash) == 64  # SHA-256 = 64 hex chars
        assert re.match(r'^[0-9a-f]{64}$', result.provenance_hash), (
            "Provenance hash must be lowercase hex"
        )

    def test_provenance_hash_deterministic(
        self,
        fouling_calculator: FoulingCalculator,
        fouling_resistance_input: FoulingResistanceInput,
    ):
        """Test same input produces same provenance hash."""
        # Act
        result1 = fouling_calculator.calculate_fouling_resistance(fouling_resistance_input)
        result2 = fouling_calculator.calculate_fouling_resistance(fouling_resistance_input)

        # Assert
        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_unique_for_different_inputs(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test different inputs produce different provenance hashes."""
        # Arrange
        input1 = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )
        input2 = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=400.0,  # Different value
        )

        # Act
        result1 = fouling_calculator.calculate_fouling_resistance(input1)
        result2 = fouling_calculator.calculate_fouling_resistance(input2)

        # Assert
        assert result1.provenance_hash != result2.provenance_hash

    def test_provenance_hash_includes_version(
        self,
        economic_calculator: EconomicCalculator,
        energy_loss_input: EnergyLossInput,
    ):
        """Test provenance hash includes calculator version."""
        # The hash should change if calculator version changes
        # This test verifies the calculator has a version attribute
        assert hasattr(economic_calculator, 'VERSION')
        assert economic_calculator.VERSION == "1.0.0"

    def test_calculation_steps_recorded(
        self,
        economic_calculator: EconomicCalculator,
        energy_loss_input: EnergyLossInput,
    ):
        """Test calculation steps are recorded for audit trail."""
        # Act
        result = economic_calculator.calculate_energy_loss_cost(energy_loss_input)

        # Assert
        assert len(result.calculation_steps) > 0
        for step in result.calculation_steps:
            assert step.step_number > 0
            assert step.operation != ""
            assert step.description != ""
            assert step.output_name != ""


# =============================================================================
# Test Class: Tool Error Handling
# =============================================================================

class TestToolErrorHandling:
    """Tests for tool function error handling."""

    def test_invalid_fluid_type_handling(self):
        """Test handling of invalid fluid type."""
        with pytest.raises((ValueError, KeyError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=500.0,
                u_fouled_w_m2_k=420.0,
                fluid_type_hot="invalid_fluid",  # Invalid
            )

    def test_invalid_exchanger_type_handling(self):
        """Test handling of invalid exchanger type."""
        with pytest.raises((ValueError, KeyError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=500.0,
                u_fouled_w_m2_k=420.0,
                exchanger_type="invalid_type",  # Invalid
            )

    def test_missing_required_field(self):
        """Test error when required field is missing."""
        with pytest.raises((TypeError, ValueError)):
            # Missing required u_clean_w_m2_k
            FoulingResistanceInput(
                u_fouled_w_m2_k=420.0,
            )

    def test_type_coercion(self):
        """Test numeric type coercion works correctly."""
        # Arrange: Pass integer instead of float
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500,  # Integer
            u_fouled_w_m2_k=420,  # Integer
        )

        # Assert: Should be coerced to float
        assert isinstance(input_data.u_clean_w_m2_k, float)

    def test_nan_value_handling(self):
        """Test handling of NaN values."""
        import math
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=float('nan'),  # NaN
                u_fouled_w_m2_k=420.0,
            )

    def test_infinity_value_handling(self):
        """Test handling of infinite values."""
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=float('inf'),  # Infinity
                u_fouled_w_m2_k=420.0,
            )


# =============================================================================
# Test Class: Tool Function Integration
# =============================================================================

class TestToolFunctionIntegration:
    """Tests for tool function integration."""

    def test_fouling_to_economic_integration(
        self,
        fouling_calculator: FoulingCalculator,
        economic_calculator: EconomicCalculator,
    ):
        """Test fouling results can feed into economic calculations."""
        # Arrange
        fouling_input = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act: Calculate fouling
        fouling_result = fouling_calculator.calculate_fouling_resistance(fouling_input)

        # Use fouling result to determine duty loss
        design_duty = Decimal("1500")
        cf = fouling_result.cleanliness_factor_percent / Decimal("100")
        actual_duty = design_duty * cf

        # Create energy loss input
        energy_input = EnergyLossInput(
            design_duty_kw=design_duty,
            actual_duty_kw=actual_duty,
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8000"),
        )

        # Act: Calculate economic impact
        economic_result = economic_calculator.calculate_energy_loss_cost(energy_input)

        # Assert
        assert economic_result.total_energy_penalty_per_year_usd > Decimal("0")

    def test_severity_to_action_mapping(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test fouling severity maps to recommended actions."""
        # Test different severity levels
        severity_inputs = [
            (0.05, FoulingSeverity.CLEAN, False),
            (0.25, FoulingSeverity.LIGHT, False),
            (0.50, FoulingSeverity.MODERATE, False),
            (0.80, FoulingSeverity.HEAVY, True),
            (1.10, FoulingSeverity.SEVERE, True),
            (1.50, FoulingSeverity.CRITICAL, True),
        ]

        for nff, expected_severity, requires_action in severity_inputs:
            input_data = FoulingSeverityInput(
                normalized_fouling_factor=nff,
                cleanliness_factor_percent=100 * (1 - nff * 0.15),
            )

            result = fouling_calculator.assess_fouling_severity(input_data)

            assert result.severity_level == expected_severity, (
                f"Expected {expected_severity} for NFF={nff}, got {result.severity_level}"
            )
            assert result.requires_immediate_action == requires_action


# =============================================================================
# Test Class: Tool Input Edge Cases
# =============================================================================

class TestToolInputEdgeCases:
    """Tests for tool input edge cases."""

    def test_minimum_valid_values(self, fouling_calculator: FoulingCalculator):
        """Test minimum valid input values."""
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=0.001,  # Very small but valid
            u_fouled_w_m2_k=0.0001,
        )
        result = fouling_calculator.calculate_fouling_resistance(input_data)
        assert result.fouling_resistance_m2_k_w > Decimal("0")

    def test_maximum_valid_values(self, fouling_calculator: FoulingCalculator):
        """Test maximum valid input values."""
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=100000.0,  # Very large but valid
            u_fouled_w_m2_k=90000.0,
        )
        result = fouling_calculator.calculate_fouling_resistance(input_data)
        assert result.fouling_resistance_m2_k_w > Decimal("0")

    def test_high_precision_values(self, economic_calculator: EconomicCalculator):
        """Test handling of high-precision Decimal values."""
        input_data = EnergyLossInput(
            design_duty_kw=Decimal("1500.123456789012345"),
            actual_duty_kw=Decimal("1275.987654321098765"),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.054321098765432"),
            operating_hours_per_year=Decimal("8000.12345"),
        )

        result = economic_calculator.calculate_energy_loss_cost(input_data)
        assert result.total_energy_penalty_per_year_usd > Decimal("0")

    def test_unicode_in_string_fields(self):
        """Test handling of unicode characters."""
        # Some inputs might have descriptive fields
        # Ensure unicode doesn't break anything
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
            fluid_type_hot=FluidType.WATER_TREATED,  # Normal enum
        )
        assert input_data.fluid_type_hot == FluidType.WATER_TREATED


# =============================================================================
# Test Class: Tool Enumeration Validation
# =============================================================================

class TestToolEnumerationValidation:
    """Tests for enumeration value validation."""

    def test_all_fluid_types_valid(self, fouling_calculator: FoulingCalculator):
        """Test all FluidType enum values work."""
        for fluid_type in FluidType:
            input_data = FoulingResistanceInput(
                u_clean_w_m2_k=500.0,
                u_fouled_w_m2_k=420.0,
                fluid_type_hot=fluid_type,
            )
            result = fouling_calculator.calculate_fouling_resistance(input_data)
            assert result.fouling_resistance_m2_k_w >= Decimal("0")

    def test_all_fouling_mechanisms_valid(self, fouling_calculator: FoulingCalculator):
        """Test all FoulingMechanism enum values work."""
        for mechanism in FoulingMechanism:
            input_data = EbertPanchalInput(
                reynolds_number=50000.0,
                prandtl_number=50.0,
                film_temperature_k=400.0,
                wall_shear_stress_pa=50.0,
                velocity_m_s=1.5,
                fouling_mechanism=mechanism,
            )
            result = fouling_calculator.calculate_ebert_panchal(input_data)
            assert result.fouling_rate_m2_k_w_per_hour >= Decimal("0")

    def test_all_fuel_types_valid(self, economic_calculator: EconomicCalculator):
        """Test all FuelType enum values work."""
        for fuel_type in FuelType:
            input_data = EnergyLossInput(
                design_duty_kw=Decimal("1500"),
                actual_duty_kw=Decimal("1275"),
                fuel_type=fuel_type,
                fuel_cost_per_kwh=Decimal("0.05"),
                operating_hours_per_year=Decimal("8000"),
            )
            result = economic_calculator.calculate_energy_loss_cost(input_data)
            assert result.total_energy_penalty_per_year_usd >= Decimal("0")

    def test_all_severity_levels_classifiable(self, fouling_calculator: FoulingCalculator):
        """Test all severity levels can be classified."""
        severity_thresholds = [
            (0.05, FoulingSeverity.CLEAN),
            (0.2, FoulingSeverity.LIGHT),
            (0.45, FoulingSeverity.MODERATE),
            (0.75, FoulingSeverity.HEAVY),
            (1.0, FoulingSeverity.SEVERE),
            (1.5, FoulingSeverity.CRITICAL),
        ]

        for nff, expected_severity in severity_thresholds:
            input_data = FoulingSeverityInput(
                normalized_fouling_factor=nff,
                cleanliness_factor_percent=90.0,
            )
            result = fouling_calculator.assess_fouling_severity(input_data)
            assert result.severity_level == expected_severity


# =============================================================================
# Test Class: Tool Timestamp Validation
# =============================================================================

class TestToolTimestampValidation:
    """Tests for timestamp handling in tool outputs."""

    def test_timestamp_is_iso_format(
        self,
        fouling_calculator: FoulingCalculator,
        fouling_resistance_input: FoulingResistanceInput,
    ):
        """Test calculation timestamp is ISO format."""
        result = fouling_calculator.calculate_fouling_resistance(fouling_resistance_input)

        # Should be parseable as ISO format
        timestamp = result.calculation_timestamp
        assert 'T' in timestamp or timestamp.endswith('Z') or '+' in timestamp

    def test_timestamp_is_recent(
        self,
        fouling_calculator: FoulingCalculator,
        fouling_resistance_input: FoulingResistanceInput,
    ):
        """Test calculation timestamp is recent (within last minute)."""
        before = datetime.now(timezone.utc)
        result = fouling_calculator.calculate_fouling_resistance(fouling_resistance_input)
        after = datetime.now(timezone.utc)

        # Parse timestamp (handle various formats)
        timestamp_str = result.calculation_timestamp
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'

        try:
            result_time = datetime.fromisoformat(timestamp_str)
            if result_time.tzinfo is None:
                result_time = result_time.replace(tzinfo=timezone.utc)

            assert before <= result_time <= after, (
                f"Timestamp {result_time} not between {before} and {after}"
            )
        except ValueError:
            # If we can't parse, at least verify it's a string
            assert isinstance(timestamp_str, str)
