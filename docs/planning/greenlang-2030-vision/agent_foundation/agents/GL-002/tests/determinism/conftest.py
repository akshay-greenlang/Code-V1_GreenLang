# -*- coding: utf-8 -*-
"""
Determinism Test Fixtures and Configuration for GL-002 FLAMEGUARD.

Provides fixtures for:
- Deterministic seed management
- Reproducibility validation utilities
- Provenance tracking helpers
- Floating-point precision fixtures

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import random
import hashlib
import json
from typing import Dict, List, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with determinism-specific markers."""
    markers = [
        "determinism: Determinism verification tests",
        "reproducibility: Bit-perfect reproducibility tests",
        "provenance: Provenance hash tests",
        "floating_point: Floating-point stability tests",
        "seed: Random seed propagation tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DeterminismResult:
    """Result of determinism verification."""
    input_hash: str
    output_hashes: List[str]
    is_deterministic: bool
    unique_outputs: int
    iterations: int


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def deterministic_seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def sample_inputs():
    """Sample input data with Decimal precision."""
    return {
        "steam_flow_kg_hr": Decimal("20000.0"),
        "fuel_flow_kg_hr": Decimal("1500.0"),
        "steam_pressure_bar": Decimal("35.0"),
        "steam_temperature_c": Decimal("400.0"),
        "o2_percent": Decimal("4.5"),
        "co_ppm": Decimal("15.0"),
        "nox_ppm": Decimal("22.0"),
        "flue_gas_temp_c": Decimal("180.0"),
        "feedwater_temp_c": Decimal("100.0"),
        "ambient_temp_c": Decimal("25.0"),
        "fuel_heating_value_mj_kg": Decimal("50.0")
    }


@pytest.fixture
def fuel_parameters():
    """Fuel-specific parameters for combustion calculations."""
    return {
        "natural_gas": {
            "carbon_content": Decimal("0.75"),
            "hydrogen_content": Decimal("0.25"),
            "sulfur_content": Decimal("0.0"),
            "ash_content": Decimal("0.0"),
            "moisture_content": Decimal("0.0"),
            "heating_value_mj_kg": Decimal("50.0"),
            "stoichiometric_air_kg": Decimal("17.2")
        },
        "fuel_oil": {
            "carbon_content": Decimal("0.86"),
            "hydrogen_content": Decimal("0.12"),
            "sulfur_content": Decimal("0.02"),
            "ash_content": Decimal("0.001"),
            "moisture_content": Decimal("0.0"),
            "heating_value_mj_kg": Decimal("42.5"),
            "stoichiometric_air_kg": Decimal("14.0")
        },
        "coal": {
            "carbon_content": Decimal("0.70"),
            "hydrogen_content": Decimal("0.05"),
            "sulfur_content": Decimal("0.02"),
            "ash_content": Decimal("0.10"),
            "moisture_content": Decimal("0.08"),
            "heating_value_mj_kg": Decimal("28.0"),
            "stoichiometric_air_kg": Decimal("11.5")
        }
    }


@pytest.fixture
def golden_reference_values():
    """Golden reference values for validation."""
    return {
        "efficiency_case_1": {
            "inputs": {
                "steam_flow_kg_hr": Decimal("20000.0"),
                "fuel_flow_kg_hr": Decimal("1500.0"),
                "steam_enthalpy_kj_kg": Decimal("2800.0"),
                "feedwater_enthalpy_kj_kg": Decimal("420.0"),
                "fuel_heating_value_mj_kg": Decimal("50.0")
            },
            "expected_efficiency": Decimal("63.47"),  # (20000 * (2800-420) / 1000) / (1500 * 50) * 100
            "tolerance": Decimal("0.01")
        },
        "excess_air_case_1": {
            "inputs": {
                "o2_percent": Decimal("4.5")
            },
            "expected_excess_air": Decimal("27.27"),  # (4.5 / (21 - 4.5)) * 100
            "tolerance": Decimal("0.01")
        },
        "combustion_efficiency_case_1": {
            "inputs": {
                "flue_gas_temp_c": Decimal("180.0"),
                "ambient_temp_c": Decimal("25.0"),
                "excess_air_percent": Decimal("27.27")
            },
            "expected_dry_gas_loss": Decimal("4.58"),
            "tolerance": Decimal("0.01")
        }
    }


# =============================================================================
# HELPER CLASSES
# =============================================================================

class DeterminismValidator:
    """Validates deterministic behavior of calculations."""

    @staticmethod
    def validate_reproducibility(
        func,
        inputs: Dict[str, Any],
        iterations: int = 100
    ) -> DeterminismResult:
        """
        Validate that a function produces identical outputs for identical inputs.

        Args:
            func: Function to test
            inputs: Input parameters
            iterations: Number of times to run the function

        Returns:
            DeterminismResult with verification details
        """
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        output_hashes = []
        for _ in range(iterations):
            result = func(**inputs) if isinstance(inputs, dict) else func(inputs)
            output_hash = hashlib.sha256(
                json.dumps(result, sort_keys=True, default=str).encode()
            ).hexdigest()
            output_hashes.append(output_hash)

        unique_outputs = len(set(output_hashes))
        is_deterministic = unique_outputs == 1

        return DeterminismResult(
            input_hash=input_hash,
            output_hashes=output_hashes,
            is_deterministic=is_deterministic,
            unique_outputs=unique_outputs,
            iterations=iterations
        )


class ProvenanceValidator:
    """Validates provenance hash chain integrity."""

    def __init__(self):
        self.chain: List[Dict[str, Any]] = []

    def add_entry(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> str:
        """Add entry to provenance chain."""
        previous_hash = self.chain[-1]["hash"] if self.chain else None

        entry = {
            "sequence": len(self.chain),
            "operation": operation,
            "input_hash": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "output_hash": hashlib.sha256(
                json.dumps(outputs, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "previous_hash": previous_hash
        }

        entry_str = json.dumps(entry, sort_keys=True)
        entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()

        self.chain.append(entry)
        return entry["hash"]

    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify integrity of provenance chain."""
        issues = []

        for i, entry in enumerate(self.chain):
            # Verify sequence
            if entry["sequence"] != i:
                issues.append(f"Sequence mismatch at entry {i}")

            # Verify previous hash linkage
            if i > 0 and entry["previous_hash"] != self.chain[i-1]["hash"]:
                issues.append(f"Chain broken at entry {i}")

            # Verify hash integrity
            entry_copy = entry.copy()
            stored_hash = entry_copy.pop("hash")
            computed_hash = hashlib.sha256(
                json.dumps(entry_copy, sort_keys=True).encode()
            ).hexdigest()
            if computed_hash != stored_hash:
                issues.append(f"Hash mismatch at entry {i}")

        return len(issues) == 0, issues

    def get_chain_hash(self) -> str:
        """Get hash of entire chain."""
        chain_str = json.dumps(self.chain, sort_keys=True, default=str)
        return hashlib.sha256(chain_str.encode()).hexdigest()


@pytest.fixture
def determinism_validator():
    """Provide determinism validator."""
    return DeterminismValidator()


@pytest.fixture
def provenance_validator():
    """Provide provenance validator."""
    return ProvenanceValidator()
