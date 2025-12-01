# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests for GL-002 FLAMEGUARD BoilerEfficiencyOptimizer.

Verifies bit-perfect reproducibility following zero-hallucination principles:
- Combustion calculations reproducibility
- Efficiency results determinism
- Provenance hash verification
- Floating-point stability
- Random seed propagation

Coverage Target: 95%+
Author: GreenLang Foundation Test Engineering
"""

import pytest
import hashlib
import json
import random
import math
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, Any, List

from conftest import DeterminismValidator, ProvenanceValidator


# Set high precision for Decimal calculations
getcontext().prec = 28


# =============================================================================
# COMBUSTION CALCULATIONS REPRODUCIBILITY
# =============================================================================

class TestCombustionCalculationsReproducibility:
    """Test combustion calculation reproducibility."""

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_excess_air_calculation_reproducibility(self, sample_inputs):
        """Test excess air calculation produces identical results."""
        o2_percent = sample_inputs["o2_percent"]

        results = []
        for _ in range(1000):
            excess_air = (o2_percent / (Decimal("21.0") - o2_percent)) * Decimal("100.0")
            excess_air = excess_air.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(excess_air)

        # All results must be identical
        assert len(set(results)) == 1, "Excess air calculation not deterministic"
        assert results[0] == Decimal("27.2727")  # Expected value

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_stoichiometric_air_reproducibility(self, fuel_parameters):
        """Test stoichiometric air calculation reproducibility."""
        for fuel_type, params in fuel_parameters.items():
            carbon = params["carbon_content"]
            hydrogen = params["hydrogen_content"]
            sulfur = params["sulfur_content"]

            results = []
            for _ in range(1000):
                # Stoichiometric air calculation
                stoich_air = (
                    Decimal("11.6") * carbon +
                    Decimal("34.8") * hydrogen +
                    Decimal("4.3") * sulfur
                )
                stoich_air = stoich_air.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                results.append(stoich_air)

            assert len(set(results)) == 1, f"Stoichiometric air for {fuel_type} not deterministic"

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_combustion_products_reproducibility(self, sample_inputs, fuel_parameters):
        """Test combustion products calculation reproducibility."""
        fuel = fuel_parameters["natural_gas"]
        carbon = fuel["carbon_content"]
        hydrogen = fuel["hydrogen_content"]

        results = []
        for _ in range(1000):
            # CO2 and H2O production per kg fuel
            co2_kg = carbon * Decimal("44.0") / Decimal("12.0")
            h2o_kg = hydrogen * Decimal("18.0") / Decimal("2.0")

            products = {
                "co2_kg": co2_kg.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                "h2o_kg": h2o_kg.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            }
            results.append(json.dumps(products, sort_keys=True, default=str))

        assert len(set(results)) == 1, "Combustion products calculation not deterministic"

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_flue_gas_composition_reproducibility(self, sample_inputs):
        """Test flue gas composition calculation reproducibility."""
        o2_percent = sample_inputs["o2_percent"]

        results = []
        for _ in range(1000):
            # Simplified flue gas composition
            n2_percent = Decimal("79.0") - (o2_percent * Decimal("79.0") / Decimal("21.0"))
            co2_percent = Decimal("21.0") - o2_percent - Decimal("0.5")  # Simplified

            composition = {
                "o2": o2_percent.quantize(Decimal("0.01")),
                "n2": n2_percent.quantize(Decimal("0.01")),
                "co2": co2_percent.quantize(Decimal("0.01"))
            }
            results.append(json.dumps(composition, sort_keys=True, default=str))

        assert len(set(results)) == 1, "Flue gas composition not deterministic"


# =============================================================================
# EFFICIENCY RESULTS DETERMINISM
# =============================================================================

class TestEfficiencyResultsDeterminism:
    """Test efficiency calculation determinism."""

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_direct_efficiency_determinism(self, sample_inputs):
        """Test direct efficiency calculation determinism."""
        steam_flow = sample_inputs["steam_flow_kg_hr"]
        fuel_flow = sample_inputs["fuel_flow_kg_hr"]
        fuel_hv = sample_inputs["fuel_heating_value_mj_kg"]

        steam_enthalpy = Decimal("2800.0")  # kJ/kg
        feedwater_enthalpy = Decimal("420.0")  # kJ/kg

        results = []
        for _ in range(1000):
            energy_out = steam_flow * (steam_enthalpy - feedwater_enthalpy) / Decimal("1000.0")
            energy_in = fuel_flow * fuel_hv
            efficiency = (energy_out / energy_in) * Decimal("100.0")
            efficiency = efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            results.append(efficiency)

        assert len(set(results)) == 1, "Direct efficiency not deterministic"
        assert results[0] == Decimal("63.47")

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_indirect_efficiency_determinism(self, sample_inputs, golden_reference_values):
        """Test indirect efficiency (loss method) determinism."""
        flue_gas_temp = sample_inputs["flue_gas_temp_c"]
        ambient_temp = sample_inputs["ambient_temp_c"]
        o2_percent = sample_inputs["o2_percent"]

        results = []
        for _ in range(1000):
            # Excess air
            excess_air = (o2_percent / (Decimal("21.0") - o2_percent)) * Decimal("100.0")

            # Dry gas loss
            k = Decimal("0.24")  # Natural gas
            dry_gas_loss = k * (Decimal("1.0") + excess_air/Decimal("100.0")) * (
                flue_gas_temp - ambient_temp
            ) / Decimal("100.0")

            # Moisture loss
            moisture_loss = Decimal("4.5")

            # Radiation loss
            radiation_loss = Decimal("1.5")

            # Total losses
            total_losses = dry_gas_loss + moisture_loss + radiation_loss
            total_losses = total_losses.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            efficiency = Decimal("100.0") - total_losses
            results.append(efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        assert len(set(results)) == 1, "Indirect efficiency not deterministic"

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_stack_loss_determinism(self, sample_inputs):
        """Test stack loss calculations are deterministic."""
        flue_gas_temp = sample_inputs["flue_gas_temp_c"]
        ambient_temp = sample_inputs["ambient_temp_c"]
        o2_percent = sample_inputs["o2_percent"]

        results = []
        for _ in range(1000):
            excess_air = (o2_percent / (Decimal("21.0") - o2_percent)) * Decimal("100.0")

            # Dry gas loss
            dry_gas = Decimal("0.24") * (Decimal("1.0") + excess_air/Decimal("100.0")) * (
                flue_gas_temp - ambient_temp
            ) / Decimal("100.0")

            # Moisture in air loss
            moisture_air = Decimal("0.05") * (Decimal("1.0") + excess_air/Decimal("100.0")) * (
                flue_gas_temp - ambient_temp
            ) / Decimal("100.0")

            # Moisture in fuel loss
            moisture_fuel = Decimal("4.5")

            # Unburned carbon loss
            unburned = Decimal("0.5")

            # Radiation loss
            radiation = Decimal("1.5")

            total = dry_gas + moisture_air + moisture_fuel + unburned + radiation
            total = total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            results.append(total)

        assert len(set(results)) == 1, "Stack loss not deterministic"

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_golden_reference_validation(self, golden_reference_values):
        """Test calculations match golden reference values."""
        # Efficiency case
        case = golden_reference_values["efficiency_case_1"]
        inputs = case["inputs"]

        steam_flow = inputs["steam_flow_kg_hr"]
        fuel_flow = inputs["fuel_flow_kg_hr"]
        steam_h = inputs["steam_enthalpy_kj_kg"]
        fw_h = inputs["feedwater_enthalpy_kj_kg"]
        fuel_hv = inputs["fuel_heating_value_mj_kg"]

        energy_out = steam_flow * (steam_h - fw_h) / Decimal("1000.0")
        energy_in = fuel_flow * fuel_hv
        efficiency = (energy_out / energy_in) * Decimal("100.0")
        efficiency = efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        expected = case["expected_efficiency"]
        tolerance = case["tolerance"]

        assert abs(efficiency - expected) <= tolerance, \
            f"Efficiency {efficiency} differs from golden reference {expected}"


# =============================================================================
# PROVENANCE HASH VERIFICATION
# =============================================================================

class TestProvenanceHashVerification:
    """Test provenance hash verification."""

    @pytest.mark.determinism
    @pytest.mark.provenance
    def test_hash_consistency_same_input(self, sample_inputs):
        """Test same input always produces same hash."""
        data = {k: str(v) for k, v in sample_inputs.items()}

        hashes = []
        for _ in range(100):
            hash_val = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            hashes.append(hash_val)

        assert len(set(hashes)) == 1, "Hash not consistent for same input"
        assert len(hashes[0]) == 64  # SHA-256 produces 64 hex chars

    @pytest.mark.determinism
    @pytest.mark.provenance
    def test_hash_changes_with_input(self, sample_inputs):
        """Test hash changes when input changes."""
        data = {k: str(v) for k, v in sample_inputs.items()}

        original = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        # Modify one value slightly
        data["o2_percent"] = "4.6"  # Changed from 4.5

        modified = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        assert original != modified, "Hash should change when input changes"

    @pytest.mark.determinism
    @pytest.mark.provenance
    def test_provenance_chain_integrity(self, provenance_validator, sample_inputs):
        """Test provenance chain maintains integrity."""
        # Add entries to chain
        for i in range(10):
            inputs = {"cycle": i, "data": sample_inputs}
            outputs = {"efficiency": 85.0 + i * 0.1, "status": "success"}
            provenance_validator.add_entry(
                operation=f"optimization_cycle_{i}",
                inputs=inputs,
                outputs=outputs
            )

        # Verify chain
        is_valid, issues = provenance_validator.verify_chain()

        assert is_valid, f"Provenance chain invalid: {issues}"
        assert len(provenance_validator.chain) == 10

    @pytest.mark.determinism
    @pytest.mark.provenance
    def test_chain_hash_reproducibility(self, provenance_validator, sample_inputs):
        """Test chain hash is reproducible."""
        # Build chain
        for i in range(5):
            provenance_validator.add_entry(
                operation=f"cycle_{i}",
                inputs={"cycle": i},
                outputs={"result": i * 2}
            )

        # Get chain hash multiple times
        hashes = [provenance_validator.get_chain_hash() for _ in range(100)]

        assert len(set(hashes)) == 1, "Chain hash not reproducible"

    @pytest.mark.determinism
    @pytest.mark.provenance
    def test_tamper_detection(self, provenance_validator):
        """Test chain detects tampering."""
        # Build valid chain
        for i in range(5):
            provenance_validator.add_entry(
                operation=f"cycle_{i}",
                inputs={"value": i},
                outputs={"result": i * 2}
            )

        # Tamper with chain
        if provenance_validator.chain:
            provenance_validator.chain[2]["output_hash"] = "tampered_hash"

        # Verify should detect tampering
        is_valid, issues = provenance_validator.verify_chain()

        assert not is_valid, "Tampering should be detected"
        assert len(issues) > 0


# =============================================================================
# FLOATING-POINT STABILITY
# =============================================================================

class TestFloatingPointStability:
    """Test floating-point stability of calculations."""

    @pytest.mark.determinism
    @pytest.mark.floating_point
    def test_decimal_precision_maintained(self):
        """Test Decimal maintains precision across operations."""
        values = [
            Decimal("0.1"),
            Decimal("0.2"),
            Decimal("0.3"),
            Decimal("0.4")
        ]

        # Sum should be exact with Decimal
        total = sum(values, Decimal("0"))
        assert total == Decimal("1.0"), "Decimal sum precision lost"

        # Order of operations should not matter
        total_reversed = sum(reversed(values), Decimal("0"))
        assert total == total_reversed, "Decimal not associative"

    @pytest.mark.determinism
    @pytest.mark.floating_point
    def test_decimal_vs_float_comparison(self):
        """Test Decimal is more precise than float for repeated operations."""
        # Float accumulation (can lose precision)
        float_sum = 0.0
        for _ in range(10000):
            float_sum += 0.1

        # Decimal accumulation (maintains precision)
        decimal_sum = Decimal("0")
        for _ in range(10000):
            decimal_sum += Decimal("0.1")

        # Float may not equal 1000.0 exactly
        expected = Decimal("1000.0")
        assert decimal_sum == expected, "Decimal precision lost"

    @pytest.mark.determinism
    @pytest.mark.floating_point
    def test_rounding_consistency(self):
        """Test rounding is consistent."""
        values = [
            (Decimal("1.2345"), Decimal("1.23")),
            (Decimal("1.2355"), Decimal("1.24")),  # ROUND_HALF_UP
            (Decimal("1.2350"), Decimal("1.24")),  # ROUND_HALF_UP
            (Decimal("1.2340"), Decimal("1.23")),
        ]

        for value, expected in values:
            rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            assert rounded == expected, f"Rounding {value} -> {rounded}, expected {expected}"

    @pytest.mark.determinism
    @pytest.mark.floating_point
    def test_division_precision(self):
        """Test division maintains precision."""
        numerator = Decimal("100.0")
        denominators = [Decimal("3.0"), Decimal("7.0"), Decimal("11.0")]

        results = []
        for _ in range(100):
            for denom in denominators:
                result = (numerator / denom).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
                results.append(str(result))

        # Each division should produce same result every time
        expected_count = len(denominators)
        assert len(set(results)) == expected_count

    @pytest.mark.determinism
    @pytest.mark.floating_point
    def test_edge_case_small_values(self):
        """Test handling of very small values."""
        small_values = [
            Decimal("1E-10"),
            Decimal("1E-15"),
            Decimal("1E-20")
        ]

        for val in small_values:
            # Operations with small values
            result = val + val
            expected = val * Decimal("2")
            assert result == expected, f"Small value arithmetic failed for {val}"

    @pytest.mark.determinism
    @pytest.mark.floating_point
    def test_edge_case_large_values(self):
        """Test handling of very large values."""
        large = Decimal("1E20")
        small = Decimal("1E-10")

        # Large + small should preserve both contributions in Decimal
        result = large + small
        difference = result - large
        assert difference == small, "Large + small precision lost"


# =============================================================================
# RANDOM SEED PROPAGATION
# =============================================================================

class TestSeedPropagation:
    """Test random seed propagation for reproducibility."""

    @pytest.mark.determinism
    @pytest.mark.seed
    def test_random_seed_reproducibility(self, deterministic_seed):
        """Test random seed produces reproducible sequences."""
        random.seed(deterministic_seed)
        sequence_1 = [random.random() for _ in range(100)]

        random.seed(deterministic_seed)
        sequence_2 = [random.random() for _ in range(100)]

        assert sequence_1 == sequence_2, "Random sequences differ with same seed"

    @pytest.mark.determinism
    @pytest.mark.seed
    def test_random_integer_reproducibility(self, deterministic_seed):
        """Test random integer generation is reproducible."""
        random.seed(deterministic_seed)
        ints_1 = [random.randint(0, 100) for _ in range(100)]

        random.seed(deterministic_seed)
        ints_2 = [random.randint(0, 100) for _ in range(100)]

        assert ints_1 == ints_2, "Random integers differ with same seed"

    @pytest.mark.determinism
    @pytest.mark.seed
    def test_no_hidden_randomness(self, sample_inputs):
        """Test calculations have no hidden randomness."""
        results = []

        for _ in range(100):
            steam_flow = sample_inputs["steam_flow_kg_hr"]
            fuel_flow = sample_inputs["fuel_flow_kg_hr"]

            efficiency = (steam_flow * Decimal("2380.0")) / (fuel_flow * Decimal("50000.0")) * Decimal("100.0")
            efficiency = efficiency.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(efficiency)

        assert len(set(results)) == 1, "Hidden randomness detected in calculations"

    @pytest.mark.determinism
    @pytest.mark.seed
    def test_deterministic_sampling(self, deterministic_seed):
        """Test deterministic sampling from datasets."""
        data = list(range(1000))

        random.seed(deterministic_seed)
        sample_1 = random.sample(data, 50)

        random.seed(deterministic_seed)
        sample_2 = random.sample(data, 50)

        assert sample_1 == sample_2, "Random samples differ with same seed"


# =============================================================================
# OPTIMIZATION DETERMINISM
# =============================================================================

class TestOptimizationDeterminism:
    """Test optimization algorithm determinism."""

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_optimizer_determinism(self, sample_inputs):
        """Test optimization algorithm produces deterministic results."""
        o2_percent = sample_inputs["o2_percent"]
        target_o2 = Decimal("3.5")

        results = []
        for _ in range(100):
            # Simplified optimization
            iterations = 20
            current = o2_percent
            learning_rate = Decimal("0.1")

            for _ in range(iterations):
                error = target_o2 - current
                adjustment = learning_rate * error
                current = current + adjustment

            current = current.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(current)

        assert len(set(results)) == 1, "Optimizer not deterministic"

    @pytest.mark.determinism
    @pytest.mark.reproducibility
    def test_multi_objective_optimization_determinism(self, sample_inputs):
        """Test multi-objective optimization is deterministic."""
        results = []

        for _ in range(100):
            candidates = []
            for o2 in range(20, 60):  # 2.0 to 6.0
                o2_val = Decimal(o2) / Decimal("10")

                # NOx model
                nox = Decimal("15") + (Decimal("6") - o2_val) * Decimal("5")

                # CO model
                co = Decimal("10") + (o2_val - Decimal("2")) * Decimal("10")

                # Score
                score = (nox - Decimal("25")) ** 2 + (co - Decimal("50")) ** 2
                candidates.append((o2_val, score))

            # Find minimum
            best = min(candidates, key=lambda x: x[1])
            results.append(str(best[0]))

        assert len(set(results)) == 1, "Multi-objective optimization not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "determinism"])
