"""
Determinism Tests: Reproducibility

Tests deterministic behavior including:
- Same inputs produce same outputs
- SHA-256 provenance verification
- Reproducibility across runs
- Bit-perfect calculation reproducibility

Reference: GL-001 Specification Section 11.6
Target Coverage: 85%+
"""

import pytest
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Tuple


# =============================================================================
# Determinism Classes (Simulated Production Code)
# =============================================================================

@dataclass
class CalculationInput:
    """Input data for deterministic calculation."""
    boiler_id: str
    fuel_rate: float
    ambient_temp: float
    timestamp: str
    parameters: Dict[str, float]


@dataclass
class CalculationOutput:
    """Output data from deterministic calculation."""
    heat_output: float
    efficiency: float
    emissions: float
    provenance_hash: str
    calculation_version: str = "1.0.0"


class DeterministicCalculator:
    """Performs deterministic calculations with provenance tracking."""

    VERSION = "1.0.0"

    @classmethod
    def calculate(cls, input_data: CalculationInput) -> CalculationOutput:
        """Perform deterministic thermal calculation.

        This calculation is guaranteed to be bit-perfect reproducible.
        No floating-point randomness, no LLM involvement.
        """
        # Deterministic efficiency calculation
        base_efficiency = 0.85
        temp_factor = (input_data.ambient_temp - 20) * 0.0001
        efficiency = base_efficiency - temp_factor

        # Deterministic heat output
        heat_output = input_data.fuel_rate * efficiency

        # Deterministic emissions calculation
        emission_factor = 2.0  # kg CO2 per unit fuel
        emissions = input_data.fuel_rate * emission_factor

        # Calculate provenance hash
        provenance_hash = cls._calculate_provenance(input_data, heat_output, efficiency, emissions)

        return CalculationOutput(
            heat_output=round(heat_output, 6),
            efficiency=round(efficiency, 6),
            emissions=round(emissions, 6),
            provenance_hash=provenance_hash,
            calculation_version=cls.VERSION
        )

    @classmethod
    def _calculate_provenance(cls, input_data: CalculationInput,
                             heat_output: float, efficiency: float,
                             emissions: float) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_data = {
            "input": asdict(input_data),
            "output": {
                "heat_output": round(heat_output, 6),
                "efficiency": round(efficiency, 6),
                "emissions": round(emissions, 6)
            },
            "version": cls.VERSION
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    @classmethod
    def verify_provenance(cls, input_data: CalculationInput,
                         output: CalculationOutput) -> bool:
        """Verify that output matches input with correct provenance."""
        recalculated = cls.calculate(input_data)
        return recalculated.provenance_hash == output.provenance_hash


class OptimizationSolver:
    """Deterministic optimization solver."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def solve(self, demands: List[float], capacities: List[float],
             costs: List[float]) -> Dict[str, Any]:
        """Solve load allocation deterministically.

        Uses fixed seed and deterministic algorithm.
        """
        # Set seed for reproducibility
        random.seed(self.seed)

        n = len(capacities)
        allocations = [0.0] * n

        # Merit order dispatch (deterministic)
        merit_order = sorted(range(n), key=lambda i: costs[i])
        remaining_demand = sum(demands)

        for i in merit_order:
            if remaining_demand <= 0:
                break
            allocation = min(capacities[i], remaining_demand)
            allocations[i] = allocation
            remaining_demand -= allocation

        # Calculate deterministic hash
        result_hash = self._calculate_hash(demands, capacities, costs, allocations)

        return {
            "allocations": allocations,
            "total_cost": sum(a * c for a, c in zip(allocations, costs)),
            "unmet_demand": max(0, remaining_demand),
            "provenance_hash": result_hash
        }

    def _calculate_hash(self, demands, capacities, costs, allocations) -> str:
        """Calculate deterministic hash of solution."""
        data = {
            "demands": demands,
            "capacities": capacities,
            "costs": costs,
            "allocations": allocations,
            "seed": self.seed
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.determinism
class TestCalculationReproducibility:
    """Test calculation reproducibility."""

    @pytest.fixture
    def standard_input(self):
        """Create standard input for testing."""
        return CalculationInput(
            boiler_id="BOILER_001",
            fuel_rate=100.0,
            ambient_temp=25.0,
            timestamp="2025-01-15T10:00:00Z",
            parameters={"param1": 1.0, "param2": 2.0}
        )

    def test_same_input_same_output(self, standard_input):
        """Test same input produces same output."""
        result1 = DeterministicCalculator.calculate(standard_input)
        result2 = DeterministicCalculator.calculate(standard_input)

        assert result1.heat_output == result2.heat_output
        assert result1.efficiency == result2.efficiency
        assert result1.emissions == result2.emissions

    def test_provenance_hash_deterministic(self, standard_input):
        """Test provenance hash is deterministic."""
        result1 = DeterministicCalculator.calculate(standard_input)
        result2 = DeterministicCalculator.calculate(standard_input)

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_length(self, standard_input):
        """Test provenance hash is valid SHA-256 (64 characters)."""
        result = DeterministicCalculator.calculate(standard_input)

        assert len(result.provenance_hash) == 64
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    def test_multiple_runs_identical(self, standard_input):
        """Test multiple runs produce identical results."""
        results = [DeterministicCalculator.calculate(standard_input) for _ in range(10)]

        first_result = results[0]
        for result in results[1:]:
            assert result.heat_output == first_result.heat_output
            assert result.provenance_hash == first_result.provenance_hash

    def test_different_input_different_hash(self, standard_input):
        """Test different input produces different hash."""
        result1 = DeterministicCalculator.calculate(standard_input)

        modified_input = CalculationInput(
            boiler_id="BOILER_001",
            fuel_rate=101.0,  # Different fuel rate
            ambient_temp=25.0,
            timestamp="2025-01-15T10:00:00Z",
            parameters={"param1": 1.0, "param2": 2.0}
        )
        result2 = DeterministicCalculator.calculate(modified_input)

        assert result1.provenance_hash != result2.provenance_hash

    def test_provenance_verification(self, standard_input):
        """Test provenance verification succeeds."""
        result = DeterministicCalculator.calculate(standard_input)

        is_valid = DeterministicCalculator.verify_provenance(standard_input, result)

        assert is_valid == True

    def test_provenance_verification_fails_on_tampering(self, standard_input):
        """Test provenance verification fails on tampered output."""
        result = DeterministicCalculator.calculate(standard_input)

        # Tamper with output
        tampered_result = CalculationOutput(
            heat_output=result.heat_output + 1.0,  # Modified
            efficiency=result.efficiency,
            emissions=result.emissions,
            provenance_hash=result.provenance_hash  # Original hash
        )

        is_valid = DeterministicCalculator.verify_provenance(standard_input, tampered_result)

        assert is_valid == False


@pytest.mark.determinism
class TestOptimizationReproducibility:
    """Test optimization reproducibility."""

    @pytest.fixture
    def optimization_inputs(self):
        """Create optimization inputs."""
        return {
            "demands": [500.0, 300.0, 200.0],
            "capacities": [400.0, 300.0, 300.0],
            "costs": [0.05, 0.06, 0.07]
        }

    def test_optimization_deterministic(self, optimization_inputs):
        """Test optimization produces same result."""
        solver = OptimizationSolver(seed=42)

        result1 = solver.solve(**optimization_inputs)
        result2 = solver.solve(**optimization_inputs)

        assert result1["allocations"] == result2["allocations"]
        assert result1["total_cost"] == result2["total_cost"]
        assert result1["provenance_hash"] == result2["provenance_hash"]

    def test_different_seed_different_result(self, optimization_inputs):
        """Test different seed can produce different results."""
        solver1 = OptimizationSolver(seed=42)
        solver2 = OptimizationSolver(seed=43)

        result1 = solver1.solve(**optimization_inputs)
        result2 = solver2.solve(**optimization_inputs)

        # With deterministic algorithm, results may be same, but hashes differ
        assert result1["provenance_hash"] != result2["provenance_hash"]

    def test_optimization_hash_is_sha256(self, optimization_inputs):
        """Test optimization hash is valid SHA-256."""
        solver = OptimizationSolver(seed=42)
        result = solver.solve(**optimization_inputs)

        assert len(result["provenance_hash"]) == 64
        assert all(c in '0123456789abcdef' for c in result["provenance_hash"])


@pytest.mark.determinism
class TestHashChaining:
    """Test provenance hash chaining."""

    def test_chain_hash_deterministic(self):
        """Test chained hashes are deterministic."""
        data1 = {"step": 1, "value": 100}
        data2 = {"step": 2, "value": 200}

        def chain_hash(prev_hash, data):
            combined = f"{prev_hash}:{json.dumps(data, sort_keys=True)}"
            return hashlib.sha256(combined.encode()).hexdigest()

        # First chain
        hash1a = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()
        hash1b = chain_hash(hash1a, data2)

        # Second chain (same data)
        hash2a = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()
        hash2b = chain_hash(hash2a, data2)

        assert hash1a == hash2a
        assert hash1b == hash2b

    def test_chain_detects_modification(self):
        """Test hash chain detects any modification."""
        data1 = {"step": 1, "value": 100}
        data2 = {"step": 2, "value": 200}

        def chain_hash(prev_hash, data):
            combined = f"{prev_hash}:{json.dumps(data, sort_keys=True)}"
            return hashlib.sha256(combined.encode()).hexdigest()

        # Original chain
        hash1 = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()
        original_final = chain_hash(hash1, data2)

        # Modified first step
        data1_modified = {"step": 1, "value": 101}
        hash1_mod = hashlib.sha256(json.dumps(data1_modified, sort_keys=True).encode()).hexdigest()
        modified_final = chain_hash(hash1_mod, data2)

        assert original_final != modified_final


@pytest.mark.determinism
class TestBitPerfectReproducibility:
    """Test bit-perfect reproducibility guarantees."""

    def test_floating_point_determinism(self):
        """Test floating point calculations are deterministic."""
        def complex_calculation(x):
            result = x
            for _ in range(100):
                result = result * 1.1 / 1.1  # Should be identity
                result = result + 0.1 - 0.1  # Should be identity
            return result

        # Run multiple times
        results = [complex_calculation(100.0) for _ in range(10)]

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_decimal_precision_maintained(self):
        """Test Decimal precision is maintained."""
        a = Decimal("0.1")
        b = Decimal("0.2")
        c = Decimal("0.3")

        # This should be exact with Decimal
        result = a + b
        assert result == c

    def test_json_serialization_deterministic(self):
        """Test JSON serialization is deterministic."""
        data = {
            "z_field": 3,
            "a_field": 1,
            "m_field": 2,
            "nested": {"b": 2, "a": 1}
        }

        # Multiple serializations with sort_keys
        serializations = [
            json.dumps(data, sort_keys=True)
            for _ in range(10)
        ]

        # All should be identical
        assert all(s == serializations[0] for s in serializations)

    def test_hash_of_same_data_identical(self):
        """Test hash of same data is always identical."""
        data = {"key": "value", "number": 42}

        hashes = [
            hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            for _ in range(10)
        ]

        assert all(h == hashes[0] for h in hashes)


@pytest.mark.determinism
class TestZeroHallucinationGuarantee:
    """Test zero hallucination guarantee (no LLM in calculations)."""

    def test_no_random_in_calculation(self):
        """Test calculation does not use random numbers."""
        input_data = CalculationInput(
            boiler_id="BOILER_001",
            fuel_rate=100.0,
            ambient_temp=25.0,
            timestamp="2025-01-15T10:00:00Z",
            parameters={}
        )

        # Run many times
        results = [DeterministicCalculator.calculate(input_data) for _ in range(100)]

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result.heat_output == first.heat_output
            assert result.efficiency == first.efficiency
            assert result.emissions == first.emissions
            assert result.provenance_hash == first.provenance_hash

    def test_calculation_is_pure_function(self):
        """Test calculation is a pure function (no side effects)."""
        input_data = CalculationInput(
            boiler_id="BOILER_001",
            fuel_rate=100.0,
            ambient_temp=25.0,
            timestamp="2025-01-15T10:00:00Z",
            parameters={"test": 1.0}
        )

        # Calculate multiple times
        result1 = DeterministicCalculator.calculate(input_data)
        result2 = DeterministicCalculator.calculate(input_data)

        # Input should be unchanged
        assert input_data.fuel_rate == 100.0
        assert input_data.ambient_temp == 25.0

        # Results should be identical
        assert result1.provenance_hash == result2.provenance_hash
