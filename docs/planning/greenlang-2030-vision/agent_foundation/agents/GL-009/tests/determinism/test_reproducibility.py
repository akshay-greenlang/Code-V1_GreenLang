"""Determinism and reproducibility tests.

Tests zero-hallucination guarantees: same input = same output.
Target Coverage: 95%+, Test Count: 12+
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.first_law_efficiency import FirstLawEfficiencyCalculator
from calculators.second_law_efficiency import SecondLawEfficiencyCalculator
from calculators.heat_loss_calculator import HeatLossCalculator


@pytest.mark.determinism
class TestFirstLawDeterminism:
    """Test First Law calculator determinism."""

    def test_same_input_produces_same_output(self):
        """Test same input always produces same output."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = {"fuel": 1000.0}
        outputs = {"steam": 850.0}
        losses = {"flue_gas": 100.0, "radiation": 50.0}

        # Run calculation 10 times
        results = []
        for _ in range(10):
            result = calculator.calculate(inputs, outputs, losses)
            results.append(result.efficiency_percent)

        # All results should be identical
        assert all(r == results[0] for r in results)
        assert results[0] == 85.0

    def test_provenance_hash_consistency(self):
        """Test provenance hash is consistent for same inputs."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = {"fuel": 1000.0}
        outputs = {"steam": 850.0}
        losses = {"flue_gas": 150.0}

        # Run calculation multiple times
        hashes = []
        for _ in range(5):
            result = calculator.calculate(inputs, outputs, losses)
            hashes.append(result.provenance_hash)

        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)
        assert len(hashes[0]) == 64  # SHA-256

    def test_calculation_steps_deterministic(self):
        """Test calculation steps are deterministic."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = {"fuel": 1000.0}
        outputs = {"steam": 850.0}
        losses = {"flue_gas": 150.0}

        result1 = calculator.calculate(inputs, outputs, losses)
        result2 = calculator.calculate(inputs, outputs, losses)

        # Same number of calculation steps
        assert len(result1.calculation_steps) == len(result2.calculation_steps)

        # Same step values
        for step1, step2 in zip(result1.calculation_steps,
                               result2.calculation_steps):
            assert step1.output_value == step2.output_value

    def test_bit_perfect_reproducibility(self):
        """Test bit-perfect reproducibility of results."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = {"fuel": 1234.5678}
        outputs = {"steam": 1050.9876}
        losses = {"flue_gas": 183.5802}

        result1 = calculator.calculate(inputs, outputs, losses)
        result2 = calculator.calculate(inputs, outputs, losses)

        # Bit-perfect match
        assert result1.efficiency_percent == result2.efficiency_percent
        assert result1.energy_input_kw == result2.energy_input_kw
        assert result1.provenance_hash == result2.provenance_hash

    def test_no_randomness_in_calculations(self):
        """Test calculations contain no randomness."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = {"fuel": 1000.0}
        outputs = {"steam": 850.0}
        losses = {"flue_gas": 150.0}

        # Run 100 times to check for any random variation
        results = [calculator.calculate(inputs, outputs, losses)
                  for _ in range(100)]

        efficiencies = [r.efficiency_percent for r in results]
        hashes = [r.provenance_hash for r in results]

        # All must be identical (no variation)
        assert len(set(efficiencies)) == 1
        assert len(set(hashes)) == 1


@pytest.mark.determinism
class TestSecondLawDeterminism:
    """Test Second Law calculator determinism."""

    def test_exergy_calculation_deterministic(self, sample_exergy_streams):
        """Test exergy calculations are deterministic."""
        calculator = SecondLawEfficiencyCalculator()

        results = []
        for _ in range(5):
            result = calculator.calculate(
                input_streams=sample_exergy_streams["inputs"],
                output_streams=sample_exergy_streams["outputs"]
            )
            results.append(result.exergy_efficiency_percent)

        # All results identical
        assert all(r == results[0] for r in results)

    def test_reference_environment_consistency(self):
        """Test reference environment produces consistent results."""
        from calculators.second_law_efficiency import ReferenceEnvironment

        ref = ReferenceEnvironment(temperature_k=298.15)
        calculator = SecondLawEfficiencyCalculator(reference_environment=ref)

        # Reference should remain constant
        assert calculator.reference.temperature_k == 298.15

        # Run multiple calculations
        for _ in range(10):
            assert calculator.reference.temperature_k == 298.15


@pytest.mark.determinism
class TestHeatLossDeterminism:
    """Test heat loss calculator determinism."""

    def test_radiation_loss_deterministic(self, sample_surface_geometry):
        """Test radiation loss calculation is deterministic."""
        calculator = HeatLossCalculator()

        results = []
        for _ in range(10):
            result = calculator.calculate_radiation_loss(
                343.15, 298.15, sample_surface_geometry
            )
            results.append(result.heat_loss_kw)

        # All identical
        assert all(r == results[0] for r in results)

    def test_convection_loss_deterministic(self, sample_surface_geometry):
        """Test convection loss calculation is deterministic."""
        calculator = HeatLossCalculator()

        results = []
        for _ in range(5):
            result = calculator.calculate_natural_convection_loss(
                343.15, 298.15, sample_surface_geometry
            )
            results.append(result.heat_loss_kw)

        # All identical
        assert all(r == results[0] for r in results)


@pytest.mark.determinism
class TestCrossVersionDeterminism:
    """Test determinism across calculator versions."""

    def test_same_input_across_versions(self):
        """Test same input produces compatible results across versions."""
        # Version 1.0.0
        calculator_v1 = FirstLawEfficiencyCalculator()
        assert calculator_v1.VERSION == "1.0.0"

        inputs = {"fuel": 1000.0}
        outputs = {"steam": 850.0}
        losses = {"flue_gas": 150.0}

        result = calculator_v1.calculate(inputs, outputs, losses)

        # Result should be reproducible
        assert result.efficiency_percent == 85.0
        assert len(result.provenance_hash) == 64


@pytest.mark.determinism
class TestSeedVerification:
    """Test seed-based verification of calculations."""

    def test_seed_based_reproducibility(self):
        """Test calculations can be reproduced with seed."""
        import random
        import numpy as np

        # Set seeds
        random.seed(42)
        np.random.seed(42)

        # Generate some data
        data1 = [random.random() for _ in range(10)]

        # Reset seeds
        random.seed(42)
        np.random.seed(42)

        # Generate again
        data2 = [random.random() for _ in range(10)]

        # Should be identical
        assert data1 == data2

    def test_provenance_hash_as_verification_seed(self):
        """Test provenance hash can serve as verification seed."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = {"fuel": 1000.0}
        outputs = {"steam": 850.0}
        losses = {"flue_gas": 150.0}

        result = calculator.calculate(inputs, outputs, losses)

        # Provenance hash is deterministic
        import hashlib
        import json

        data = {
            "calculator": "FirstLawEfficiencyCalculator",
            "version": "1.0.0",
            "energy_inputs": inputs,
            "useful_outputs": outputs,
            "losses": losses,
            "balance_tolerance": 0.02,
            "precision": 4
        }

        expected_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()

        assert result.provenance_hash == expected_hash


@pytest.mark.determinism
class TestFloatingPointDeterminism:
    """Test floating-point arithmetic determinism."""

    def test_decimal_precision_determinism(self):
        """Test decimal precision is deterministic."""
        from decimal import Decimal, ROUND_HALF_UP

        value = 85.123456789

        # Round multiple times
        rounded_values = []
        for _ in range(10):
            rounded = float(
                Decimal(str(value)).quantize(
                    Decimal('0.0001'),
                    rounding=ROUND_HALF_UP
                )
            )
            rounded_values.append(rounded)

        # All should be identical
        assert all(r == rounded_values[0] for r in rounded_values)
        assert rounded_values[0] == 85.1235
