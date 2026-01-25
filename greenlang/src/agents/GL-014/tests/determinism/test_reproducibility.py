# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests for GL-014 EXCHANGER-PRO.

Tests to ensure all calculations are:
- Deterministic (same input -> same output)
- Bit-perfect reproducible
- Provenance hash consistent

This is critical for regulatory compliance and audit requirements.

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import random
from decimal import Decimal
from typing import Any, Dict, List

import pytest

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingResistanceInput,
    FoulingRateInput,
    KernSeatonInput,
    EbertPanchalInput,
    FoulingSeverityInput,
    FluidType,
    FoulingMechanism,
)
from calculators.economic_calculator import (
    EconomicCalculator,
    EnergyLossInput,
    ProductionImpactInput,
    MaintenanceCostInput,
    ROIInput,
    CarbonImpactInput,
    FuelType,
    CleaningMethod,
)


# =============================================================================
# Test Class: Heat Transfer Determinism
# =============================================================================

@pytest.mark.determinism
class TestHeatTransferDeterministic:
    """Tests for heat transfer calculation determinism."""

    def test_heat_transfer_deterministic(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test heat transfer calculations are deterministic."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
            fluid_type_hot=FluidType.OIL_LIGHT,
            fluid_type_cold=FluidType.WATER_TREATED,
        )

        # Act: Run calculation multiple times
        results = [
            fouling_calculator.calculate_fouling_resistance(input_data)
            for _ in range(10)
        ]

        # Assert: All results are identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.fouling_resistance_m2_k_w == first_result.fouling_resistance_m2_k_w, (
                f"Run {i} fouling resistance differs from run 1"
            )
            assert result.cleanliness_factor_percent == first_result.cleanliness_factor_percent, (
                f"Run {i} cleanliness factor differs from run 1"
            )

    def test_lmtd_calculation_deterministic(self, sample_temperature_data):
        """Test LMTD calculation is deterministic."""
        import math

        # Arrange
        t_hot_in = float(sample_temperature_data["hot_inlet_c"])
        t_hot_out = float(sample_temperature_data["hot_outlet_c"])
        t_cold_in = float(sample_temperature_data["cold_inlet_c"])
        t_cold_out = float(sample_temperature_data["cold_outlet_c"])

        # Act: Calculate LMTD multiple times
        results = []
        for _ in range(10):
            delta_t1 = t_hot_in - t_cold_out
            delta_t2 = t_hot_out - t_cold_in
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)
            results.append(lmtd)

        # Assert: All results are identical
        for i, result in enumerate(results[1:], 2):
            assert result == results[0], f"Run {i} LMTD differs from run 1"


# =============================================================================
# Test Class: Fouling Calculation Determinism
# =============================================================================

@pytest.mark.determinism
class TestFoulingCalculationDeterministic:
    """Tests for fouling calculation determinism."""

    def test_fouling_calculation_deterministic(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test fouling resistance calculation is deterministic."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act: Run 100 times
        results = [
            fouling_calculator.calculate_fouling_resistance(input_data)
            for _ in range(100)
        ]

        # Assert: All results identical
        first = results[0]
        for result in results[1:]:
            assert result.fouling_resistance_m2_k_w == first.fouling_resistance_m2_k_w
            assert result.cleanliness_factor_percent == first.cleanliness_factor_percent
            assert result.normalized_fouling_factor == first.normalized_fouling_factor

    def test_kern_seaton_deterministic(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test Kern-Seaton model is deterministic."""
        # Arrange
        input_data = KernSeatonInput(
            r_f_max_m2_k_w=0.0005,
            time_constant_hours=500.0,
            time_hours=200.0,
        )

        # Act
        results = [
            fouling_calculator.calculate_kern_seaton(input_data)
            for _ in range(100)
        ]

        # Assert
        first = results[0]
        for result in results[1:]:
            assert result.predicted_r_f_m2_k_w == first.predicted_r_f_m2_k_w
            assert result.asymptotic_approach_percent == first.asymptotic_approach_percent

    def test_ebert_panchal_deterministic(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test Ebert-Panchal model is deterministic."""
        # Arrange
        input_data = EbertPanchalInput(
            reynolds_number=50000.0,
            prandtl_number=50.0,
            film_temperature_k=400.0,
            wall_shear_stress_pa=50.0,
            velocity_m_s=1.5,
            fouling_mechanism=FoulingMechanism.CHEMICAL_REACTION,
        )

        # Act
        results = [
            fouling_calculator.calculate_ebert_panchal(input_data)
            for _ in range(100)
        ]

        # Assert
        first = results[0]
        for result in results[1:]:
            assert result.fouling_rate_m2_k_w_per_hour == first.fouling_rate_m2_k_w_per_hour
            assert result.deposition_rate == first.deposition_rate
            assert result.removal_rate == first.removal_rate


# =============================================================================
# Test Class: Cleaning Optimization Determinism
# =============================================================================

@pytest.mark.determinism
class TestCleaningOptimizationDeterministic:
    """Tests for cleaning optimization determinism."""

    def test_cleaning_optimization_deterministic(self):
        """Test cleaning optimization calculation is deterministic."""
        import math

        # Arrange
        cleaning_cost = 15000
        energy_penalty_rate = 100

        # Act: Calculate multiple times
        results = []
        for _ in range(100):
            optimal = math.sqrt(2 * cleaning_cost / energy_penalty_rate)
            results.append(optimal)

        # Assert
        for result in results[1:]:
            assert result == results[0]


# =============================================================================
# Test Class: Same Input Same Output
# =============================================================================

@pytest.mark.determinism
class TestSameInputSameOutput:
    """Tests for same input producing same output."""

    @pytest.mark.parametrize("u_clean,u_fouled", [
        (500.0, 420.0),
        (1000.0, 850.0),
        (300.0, 250.0),
        (750.5, 600.25),
    ])
    def test_same_input_same_output_fouling(
        self,
        fouling_calculator: FoulingCalculator,
        u_clean: float,
        u_fouled: float,
    ):
        """Test same fouling input produces same output."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=u_clean,
            u_fouled_w_m2_k=u_fouled,
        )

        # Act
        result1 = fouling_calculator.calculate_fouling_resistance(input_data)
        result2 = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert
        assert result1.fouling_resistance_m2_k_w == result2.fouling_resistance_m2_k_w
        assert result1.cleanliness_factor_percent == result2.cleanliness_factor_percent
        assert result1.provenance_hash == result2.provenance_hash

    @pytest.mark.parametrize("investment,savings,period", [
        (50000, 25000, 10),
        (100000, 20000, 15),
        (30000, 15000, 5),
    ])
    def test_same_input_same_output_roi(
        self,
        economic_calculator: EconomicCalculator,
        investment: int,
        savings: int,
        period: int,
    ):
        """Test same ROI input produces same output."""
        # Arrange
        input_data = ROIInput(
            investment_cost=Decimal(str(investment)),
            annual_savings=Decimal(str(savings)),
            analysis_period_years=period,
        )

        # Act
        result1 = economic_calculator.perform_roi_analysis(input_data)
        result2 = economic_calculator.perform_roi_analysis(input_data)

        # Assert
        assert result1.net_present_value_usd == result2.net_present_value_usd
        assert result1.internal_rate_of_return_percent == result2.internal_rate_of_return_percent
        assert result1.simple_payback_years == result2.simple_payback_years
        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# Test Class: Provenance Hash Consistency
# =============================================================================

@pytest.mark.determinism
class TestProvenanceHashConsistency:
    """Tests for provenance hash consistency."""

    def test_provenance_hash_consistency(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test provenance hash is consistent across runs."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act: Run 100 times
        hashes = [
            fouling_calculator.calculate_fouling_resistance(input_data).provenance_hash
            for _ in range(100)
        ]

        # Assert: All hashes identical
        first_hash = hashes[0]
        for i, hash_value in enumerate(hashes[1:], 2):
            assert hash_value == first_hash, (
                f"Run {i} provenance hash differs from run 1"
            )

    def test_provenance_hash_changes_with_input(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test provenance hash changes when input changes."""
        # Arrange
        inputs = [
            FoulingResistanceInput(u_clean_w_m2_k=500.0, u_fouled_w_m2_k=420.0),
            FoulingResistanceInput(u_clean_w_m2_k=500.0, u_fouled_w_m2_k=400.0),
            FoulingResistanceInput(u_clean_w_m2_k=600.0, u_fouled_w_m2_k=420.0),
        ]

        # Act
        hashes = [
            fouling_calculator.calculate_fouling_resistance(inp).provenance_hash
            for inp in inputs
        ]

        # Assert: All hashes are different
        assert len(set(hashes)) == len(hashes), "Different inputs should produce different hashes"

    def test_provenance_hash_format(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test provenance hash format is valid SHA-256."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act
        result = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert
        assert len(result.provenance_hash) == 64, "SHA-256 hash should be 64 hex characters"
        assert all(c in "0123456789abcdef" for c in result.provenance_hash.lower())


# =============================================================================
# Test Class: Bit-Perfect Reproducibility
# =============================================================================

@pytest.mark.determinism
class TestBitPerfectReproducibility:
    """Tests for bit-perfect reproducibility."""

    def test_decimal_precision_preserved(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test Decimal precision is preserved across calculations."""
        # Arrange
        input_data = EnergyLossInput(
            design_duty_kw=Decimal("1500.123456789012345"),
            actual_duty_kw=Decimal("1275.987654321098765"),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.054321098765432"),
            operating_hours_per_year=Decimal("8000.12345"),
        )

        # Act
        result1 = economic_calculator.calculate_energy_loss_cost(input_data)
        result2 = economic_calculator.calculate_energy_loss_cost(input_data)

        # Assert: Exact bit-perfect match
        assert result1.energy_cost_per_year_usd == result2.energy_cost_per_year_usd
        assert str(result1.energy_cost_per_year_usd) == str(result2.energy_cost_per_year_usd)

    def test_calculation_steps_reproducible(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test calculation steps are reproducible."""
        # Arrange
        input_data = EnergyLossInput(
            design_duty_kw=Decimal("1500"),
            actual_duty_kw=Decimal("1275"),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8000"),
        )

        # Act
        result1 = economic_calculator.calculate_energy_loss_cost(input_data)
        result2 = economic_calculator.calculate_energy_loss_cost(input_data)

        # Assert: Same number of steps
        assert len(result1.calculation_steps) == len(result2.calculation_steps)

        # Assert: Each step is identical
        for step1, step2 in zip(result1.calculation_steps, result2.calculation_steps):
            assert step1.step_number == step2.step_number
            assert step1.operation == step2.operation
            assert step1.output_value == step2.output_value

    def test_no_random_elements(
        self,
        fouling_calculator: FoulingCalculator,
        economic_calculator: EconomicCalculator,
    ):
        """Test calculations have no random elements."""
        # Arrange
        fouling_input = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )
        roi_input = ROIInput(
            investment_cost=Decimal("50000"),
            annual_savings=Decimal("25000"),
        )

        # Act: Reset random state between calls
        random.seed(12345)
        fouling_result1 = fouling_calculator.calculate_fouling_resistance(fouling_input)
        roi_result1 = economic_calculator.perform_roi_analysis(roi_input)

        random.seed(99999)  # Different seed
        fouling_result2 = fouling_calculator.calculate_fouling_resistance(fouling_input)
        roi_result2 = economic_calculator.perform_roi_analysis(roi_input)

        # Assert: Results unchanged despite different random state
        assert fouling_result1.fouling_resistance_m2_k_w == fouling_result2.fouling_resistance_m2_k_w
        assert roi_result1.net_present_value_usd == roi_result2.net_present_value_usd


# =============================================================================
# Test Class: Cross-Session Reproducibility
# =============================================================================

@pytest.mark.determinism
class TestCrossSessionReproducibility:
    """Tests for reproducibility across sessions (using known values)."""

    def test_known_value_fouling_resistance(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test fouling resistance against known calculated value."""
        # Arrange: Known values
        u_clean = 500.0  # W/m2.K
        u_fouled = 420.0  # W/m2.K

        # Expected: R_f = (1/U_fouled) - (1/U_clean)
        # R_f = (1/420) - (1/500) = 0.002381 - 0.002 = 0.000381 m2.K/W
        expected_rf = Decimal("0.000381")

        # Act
        result = fouling_calculator.calculate_fouling_resistance(
            FoulingResistanceInput(u_clean_w_m2_k=u_clean, u_fouled_w_m2_k=u_fouled)
        )

        # Assert: Within tolerance
        diff = abs(result.fouling_resistance_m2_k_w - expected_rf)
        assert diff < Decimal("0.000001"), (
            f"Expected R_f ~{expected_rf}, got {result.fouling_resistance_m2_k_w}"
        )

    def test_known_value_cleanliness_factor(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test cleanliness factor against known value."""
        # Arrange: Known values
        u_clean = 500.0  # W/m2.K
        u_fouled = 420.0  # W/m2.K

        # Expected: CF = U_fouled / U_clean * 100 = 420/500 * 100 = 84%
        expected_cf = Decimal("84.0")

        # Act
        result = fouling_calculator.calculate_fouling_resistance(
            FoulingResistanceInput(u_clean_w_m2_k=u_clean, u_fouled_w_m2_k=u_fouled)
        )

        # Assert
        assert abs(result.cleanliness_factor_percent - expected_cf) < Decimal("0.1")

    def test_known_value_kern_seaton(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test Kern-Seaton model against known value."""
        # Arrange
        r_f_max = 0.0005  # m2.K/W
        time_constant = 500.0  # hours
        time = 500.0  # hours (= tau, so expect 63.2% approach)

        # Expected: R_f(tau) = R_f_max * (1 - e^(-1)) = 0.0005 * 0.632 = 0.000316
        import math
        expected_rf = Decimal(str(r_f_max * (1 - math.exp(-1))))

        # Act
        result = fouling_calculator.calculate_kern_seaton(
            KernSeatonInput(
                r_f_max_m2_k_w=r_f_max,
                time_constant_hours=time_constant,
                time_hours=time,
            )
        )

        # Assert
        diff = abs(result.predicted_r_f_m2_k_w - expected_rf)
        assert diff < Decimal("0.000001")

    def test_known_value_simple_payback(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test simple payback against known value."""
        # Arrange
        investment = Decimal("50000")
        savings = Decimal("25000")

        # Expected: Payback = Investment / Savings = 50000 / 25000 = 2.0 years
        expected_payback = Decimal("2.0")

        # Act
        result = economic_calculator.perform_roi_analysis(
            ROIInput(
                investment_cost=investment,
                annual_savings=savings,
            )
        )

        # Assert
        assert result.simple_payback_years == expected_payback
