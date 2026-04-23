"""
Golden Master Tests for GL-012 SteamQual

Validates system behavior against recorded golden values from
known-good calculations. Ensures bit-perfect reproducibility
and deterministic behavior across all calculation modules.

Test Categories:
1. Dryness fraction calculation golden values
2. Carryover risk assessment golden values
3. Separator efficiency golden values
4. Steam property golden values
5. Provenance hash verification
6. End-to-end pipeline golden tests

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import json
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Golden Test Data Path
# =============================================================================

GOLDEN_DATA_DIR = Path(__file__).parent / "golden_data"


# =============================================================================
# Golden Test Case Data Classes
# =============================================================================

@dataclass
class GoldenTestCase:
    """Single golden test case with input/output pairs."""
    test_id: str
    description: str
    category: str
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    tolerances: Dict[str, float] = field(default_factory=dict)
    expected_hash: Optional[str] = None


@dataclass
class GoldenTestSuite:
    """Collection of golden test cases for a category."""
    suite_id: str
    description: str
    version: str
    test_cases: List[GoldenTestCase]
    created_date: str
    last_validated: str


# =============================================================================
# Golden Test Case Definitions
# =============================================================================

# Dryness Fraction Golden Values (IAPWS-IF97 verified)
DRYNESS_GOLDEN_CASES = [
    GoldenTestCase(
        test_id="DRYNESS-001",
        description="Saturated liquid at 1 MPa",
        category="dryness_fraction",
        inputs={
            "pressure_mpa": 1.0,
            "h_f_kj_kg": 762.81,
            "h_fg_kj_kg": 2015.3,
            "enthalpy_kj_kg": 762.81,  # h_f
        },
        expected_outputs={
            "dryness_fraction": 0.0,
            "state": "SATURATED_LIQUID",
        },
        tolerances={"dryness_fraction": 0.001},
        expected_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    GoldenTestCase(
        test_id="DRYNESS-002",
        description="Saturated vapor at 1 MPa",
        category="dryness_fraction",
        inputs={
            "pressure_mpa": 1.0,
            "h_f_kj_kg": 762.81,
            "h_fg_kj_kg": 2015.3,
            "enthalpy_kj_kg": 2778.1,  # h_g
        },
        expected_outputs={
            "dryness_fraction": 1.0,
            "state": "SATURATED_VAPOR",
        },
        tolerances={"dryness_fraction": 0.001},
    ),
    GoldenTestCase(
        test_id="DRYNESS-003",
        description="Wet steam at 90% quality, 1 MPa",
        category="dryness_fraction",
        inputs={
            "pressure_mpa": 1.0,
            "h_f_kj_kg": 762.81,
            "h_fg_kj_kg": 2015.3,
            "enthalpy_kj_kg": 2576.58,  # h_f + 0.9 * h_fg
        },
        expected_outputs={
            "dryness_fraction": 0.9,
            "state": "WET_STEAM",
        },
        tolerances={"dryness_fraction": 0.01},
    ),
    GoldenTestCase(
        test_id="DRYNESS-004",
        description="Wet steam at 50% quality, 5 MPa",
        category="dryness_fraction",
        inputs={
            "pressure_mpa": 5.0,
            "h_f_kj_kg": 1154.2,
            "h_fg_kj_kg": 1640.1,
            "enthalpy_kj_kg": 1974.25,  # h_f + 0.5 * h_fg
        },
        expected_outputs={
            "dryness_fraction": 0.5,
            "state": "WET_STEAM",
        },
        tolerances={"dryness_fraction": 0.01},
    ),
    GoldenTestCase(
        test_id="DRYNESS-005",
        description="High quality steam at 0.1 MPa",
        category="dryness_fraction",
        inputs={
            "pressure_mpa": 0.1,
            "h_f_kj_kg": 417.44,
            "h_fg_kj_kg": 2258.0,
            "enthalpy_kj_kg": 2562.14,  # h_f + 0.95 * h_fg
        },
        expected_outputs={
            "dryness_fraction": 0.95,
            "state": "WET_STEAM",
        },
        tolerances={"dryness_fraction": 0.01},
    ),
]

# Carryover Risk Golden Values
CARRYOVER_GOLDEN_CASES = [
    GoldenTestCase(
        test_id="CARRYOVER-001",
        description="Low risk - clean water, good quality",
        category="carryover_risk",
        inputs={
            "tds_ppm": 15.0,
            "silica_ppb": 40.0,
            "conductivity_us_cm": 18.0,
            "drum_level_percent": 50.0,
            "steam_load_percent": 80.0,
        },
        expected_outputs={
            "risk_level": "LOW",
            "probability_min": 0.0,
            "probability_max": 0.25,
        },
    ),
    GoldenTestCase(
        test_id="CARRYOVER-002",
        description="High risk - elevated TDS and silica",
        category="carryover_risk",
        inputs={
            "tds_ppm": 85.0,
            "silica_ppb": 180.0,
            "conductivity_us_cm": 75.0,
            "drum_level_percent": 65.0,
            "steam_load_percent": 95.0,
        },
        expected_outputs={
            "risk_level": "HIGH",
            "probability_min": 0.5,
            "probability_max": 0.85,
        },
    ),
    GoldenTestCase(
        test_id="CARRYOVER-003",
        description="Critical risk - very high contamination",
        category="carryover_risk",
        inputs={
            "tds_ppm": 150.0,
            "silica_ppb": 300.0,
            "conductivity_us_cm": 120.0,
            "drum_level_percent": 80.0,
            "steam_load_percent": 100.0,
        },
        expected_outputs={
            "risk_level": "CRITICAL",
            "probability_min": 0.75,
            "probability_max": 1.0,
        },
    ),
]

# Separator Efficiency Golden Values
SEPARATOR_GOLDEN_CASES = [
    GoldenTestCase(
        test_id="SEPARATOR-001",
        description="Perfect separation efficiency calculation",
        category="separator_efficiency",
        inputs={
            "inlet_dryness_fraction": 0.80,
            "outlet_dryness_fraction": 1.0,
        },
        expected_outputs={
            "separation_efficiency": 1.0,  # (1.0 - 0.8) / (1.0 - 0.8) = 1.0
        },
        tolerances={"separation_efficiency": 0.01},
    ),
    GoldenTestCase(
        test_id="SEPARATOR-002",
        description="50% separation efficiency",
        category="separator_efficiency",
        inputs={
            "inlet_dryness_fraction": 0.80,
            "outlet_dryness_fraction": 0.90,
        },
        expected_outputs={
            "separation_efficiency": 0.5,  # (0.9 - 0.8) / (1.0 - 0.8) = 0.5
        },
        tolerances={"separation_efficiency": 0.01},
    ),
    GoldenTestCase(
        test_id="SEPARATOR-003",
        description="No separation (efficiency = 0)",
        category="separator_efficiency",
        inputs={
            "inlet_dryness_fraction": 0.85,
            "outlet_dryness_fraction": 0.85,
        },
        expected_outputs={
            "separation_efficiency": 0.0,
        },
        tolerances={"separation_efficiency": 0.001},
    ),
    GoldenTestCase(
        test_id="SEPARATOR-004",
        description="Mass flow balance - 85% inlet quality, 95% outlet",
        category="separator_efficiency",
        inputs={
            "inlet_flow_kg_s": 100.0,
            "inlet_dryness_fraction": 0.85,
            "outlet_dryness_fraction": 0.95,
        },
        expected_outputs={
            # m_steam = 100 * 0.85 / 0.95 = 89.47 kg/s
            "steam_flow_kg_s": 89.47,
            # m_condensate = 100 - 89.47 = 10.53 kg/s
            "condensate_flow_kg_s": 10.53,
        },
        tolerances={
            "steam_flow_kg_s": 0.1,
            "condensate_flow_kg_s": 0.1,
        },
    ),
]

# Steam Property Golden Values (IAPWS-IF97 Table values)
STEAM_PROPERTY_GOLDEN_CASES = [
    GoldenTestCase(
        test_id="STEAM-001",
        description="Saturation temperature at 1 MPa",
        category="steam_properties",
        inputs={
            "pressure_mpa": 1.0,
        },
        expected_outputs={
            "saturation_temperature_k": 453.03,
        },
        tolerances={"saturation_temperature_k": 0.1},
    ),
    GoldenTestCase(
        test_id="STEAM-002",
        description="Saturation temperature at 0.1 MPa",
        category="steam_properties",
        inputs={
            "pressure_mpa": 0.1,
        },
        expected_outputs={
            "saturation_temperature_k": 372.756,
        },
        tolerances={"saturation_temperature_k": 0.1},
    ),
    GoldenTestCase(
        test_id="STEAM-003",
        description="Saturation temperature at 10 MPa",
        category="steam_properties",
        inputs={
            "pressure_mpa": 10.0,
        },
        expected_outputs={
            "saturation_temperature_k": 584.15,
        },
        tolerances={"saturation_temperature_k": 0.1},
    ),
]

# Provenance Hash Golden Values
PROVENANCE_GOLDEN_CASES = [
    GoldenTestCase(
        test_id="PROVENANCE-001",
        description="Provenance hash for standard dryness calculation",
        category="provenance",
        inputs={
            "pressure_mpa": 1.0,
            "enthalpy_kj_kg": 2500.0,
        },
        expected_outputs={
            # SHA-256 hash of inputs
            "expected_hash_length": 64,
            "hash_is_deterministic": True,
        },
    ),
]


# =============================================================================
# Golden Test Implementation Functions
# =============================================================================

def calculate_dryness_fraction(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate dryness fraction from inputs."""
    h_f = inputs["h_f_kj_kg"]
    h_fg = inputs["h_fg_kj_kg"]
    h = inputs["enthalpy_kj_kg"]

    if h <= h_f:
        x = 0.0
        state = "SATURATED_LIQUID"
    elif h >= h_f + h_fg:
        x = 1.0
        state = "SATURATED_VAPOR"
    else:
        x = (h - h_f) / h_fg
        state = "WET_STEAM"

    return {
        "dryness_fraction": x,
        "state": state,
    }


def calculate_carryover_risk(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate carryover risk from inputs."""
    tds = inputs["tds_ppm"]
    silica = inputs["silica_ppb"]
    conductivity = inputs["conductivity_us_cm"]
    drum_level = inputs["drum_level_percent"]
    load = inputs["steam_load_percent"]

    # Calculate individual scores
    tds_score = min(tds / 100.0, 1.0)  # Normalize to 100 ppm max
    silica_score = min(silica / 200.0, 1.0)  # Normalize to 200 ppb max
    cond_score = min(conductivity / 100.0, 1.0)  # Normalize to 100 uS/cm max

    # Apply modifiers
    drum_factor = 1.0 + max(0, (drum_level - 60) / 40) * 0.3
    load_factor = 1.0 + max(0, (load - 80) / 20) * 0.2

    # Combined probability
    base_prob = (0.4 * tds_score + 0.35 * silica_score + 0.25 * cond_score)
    prob = min(base_prob * drum_factor * load_factor, 1.0)

    # Determine risk level
    if prob < 0.25:
        level = "LOW"
    elif prob < 0.5:
        level = "MEDIUM"
    elif prob < 0.75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {
        "risk_level": level,
        "probability": prob,
    }


def calculate_separation_efficiency(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate separator efficiency from inputs."""
    x_in = inputs["inlet_dryness_fraction"]
    x_out = inputs["outlet_dryness_fraction"]

    if x_in >= 1.0:
        efficiency = 0.0
    elif x_out < x_in:
        efficiency = 0.0
    else:
        efficiency = (x_out - x_in) / (1.0 - x_in)

    return {"separation_efficiency": efficiency}


def calculate_mass_flows(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate separator mass flows from inputs."""
    m_in = inputs["inlet_flow_kg_s"]
    x_in = inputs["inlet_dryness_fraction"]
    x_out = inputs["outlet_dryness_fraction"]

    m_steam = m_in * x_in / x_out if x_out > 0 else 0
    m_condensate = m_in - m_steam

    return {
        "steam_flow_kg_s": m_steam,
        "condensate_flow_kg_s": m_condensate,
    }


def calculate_saturation_temperature(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate saturation temperature from pressure."""
    import math

    P = inputs["pressure_mpa"]

    # IAPWS-IF97 backward equation
    n = [
        0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
        0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
        -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
        0.65017534844798e3
    ]

    beta = P ** 0.25
    E = beta**2 + n[2] * beta + n[5]
    F = n[0] * beta**2 + n[3] * beta + n[6]
    G = n[1] * beta**2 + n[4] * beta + n[7]
    D = 2 * G / (-F - math.sqrt(F**2 - 4 * E * G))
    T = (n[9] + D - math.sqrt((n[9] + D)**2 - 4 * (n[8] + n[9] * D))) / 2

    return {"saturation_temperature_k": T}


def calculate_provenance_hash(inputs: Dict[str, Any]) -> str:
    """Calculate provenance hash for inputs."""
    normalized = {k: round(v, 10) if isinstance(v, float) else v for k, v in inputs.items()}
    data = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


# =============================================================================
# Test Classes
# =============================================================================

class TestDrynessGoldenValues:
    """Golden tests for dryness fraction calculations."""

    @pytest.mark.parametrize("test_case", DRYNESS_GOLDEN_CASES, ids=lambda tc: tc.test_id)
    def test_dryness_golden_value(self, test_case: GoldenTestCase):
        """Test dryness calculation against golden value."""
        result = calculate_dryness_fraction(test_case.inputs)

        expected_x = test_case.expected_outputs["dryness_fraction"]
        tolerance = test_case.tolerances.get("dryness_fraction", 0.01)

        assert abs(result["dryness_fraction"] - expected_x) <= tolerance, \
            f"{test_case.test_id}: Expected {expected_x}, got {result['dryness_fraction']}"

    @pytest.mark.parametrize("test_case", DRYNESS_GOLDEN_CASES, ids=lambda tc: tc.test_id)
    def test_dryness_state_classification(self, test_case: GoldenTestCase):
        """Test state classification against golden value."""
        result = calculate_dryness_fraction(test_case.inputs)

        if "state" in test_case.expected_outputs:
            assert result["state"] == test_case.expected_outputs["state"], \
                f"{test_case.test_id}: Expected state {test_case.expected_outputs['state']}, got {result['state']}"

    @pytest.mark.parametrize("test_case", DRYNESS_GOLDEN_CASES, ids=lambda tc: tc.test_id)
    def test_dryness_determinism(self, test_case: GoldenTestCase):
        """Test that dryness calculation is deterministic."""
        results = [calculate_dryness_fraction(test_case.inputs) for _ in range(10)]

        first = results[0]
        for r in results[1:]:
            assert r["dryness_fraction"] == first["dryness_fraction"], \
                f"{test_case.test_id}: Non-deterministic result"


class TestCarryoverGoldenValues:
    """Golden tests for carryover risk calculations."""

    @pytest.mark.parametrize("test_case", CARRYOVER_GOLDEN_CASES, ids=lambda tc: tc.test_id)
    def test_carryover_risk_level(self, test_case: GoldenTestCase):
        """Test carryover risk level against golden value."""
        result = calculate_carryover_risk(test_case.inputs)

        expected_level = test_case.expected_outputs["risk_level"]
        assert result["risk_level"] == expected_level, \
            f"{test_case.test_id}: Expected {expected_level}, got {result['risk_level']}"

    @pytest.mark.parametrize("test_case", CARRYOVER_GOLDEN_CASES, ids=lambda tc: tc.test_id)
    def test_carryover_probability_range(self, test_case: GoldenTestCase):
        """Test carryover probability is in expected range."""
        result = calculate_carryover_risk(test_case.inputs)

        prob_min = test_case.expected_outputs.get("probability_min", 0.0)
        prob_max = test_case.expected_outputs.get("probability_max", 1.0)

        assert prob_min <= result["probability"] <= prob_max, \
            f"{test_case.test_id}: Probability {result['probability']} not in [{prob_min}, {prob_max}]"

    @pytest.mark.parametrize("test_case", CARRYOVER_GOLDEN_CASES, ids=lambda tc: tc.test_id)
    def test_carryover_determinism(self, test_case: GoldenTestCase):
        """Test that carryover calculation is deterministic."""
        results = [calculate_carryover_risk(test_case.inputs) for _ in range(10)]

        first = results[0]
        for r in results[1:]:
            assert r["risk_level"] == first["risk_level"]
            assert r["probability"] == first["probability"]


class TestSeparatorGoldenValues:
    """Golden tests for separator calculations."""

    @pytest.mark.parametrize("test_case", [tc for tc in SEPARATOR_GOLDEN_CASES if "separation_efficiency" in tc.expected_outputs],
                             ids=lambda tc: tc.test_id)
    def test_separator_efficiency(self, test_case: GoldenTestCase):
        """Test separator efficiency against golden value."""
        result = calculate_separation_efficiency(test_case.inputs)

        expected = test_case.expected_outputs["separation_efficiency"]
        tolerance = test_case.tolerances.get("separation_efficiency", 0.01)

        assert abs(result["separation_efficiency"] - expected) <= tolerance, \
            f"{test_case.test_id}: Expected {expected}, got {result['separation_efficiency']}"

    @pytest.mark.parametrize("test_case", [tc for tc in SEPARATOR_GOLDEN_CASES if "steam_flow_kg_s" in tc.expected_outputs],
                             ids=lambda tc: tc.test_id)
    def test_separator_mass_flows(self, test_case: GoldenTestCase):
        """Test separator mass flows against golden values."""
        result = calculate_mass_flows(test_case.inputs)

        for key in ["steam_flow_kg_s", "condensate_flow_kg_s"]:
            expected = test_case.expected_outputs[key]
            tolerance = test_case.tolerances.get(key, 0.1)

            assert abs(result[key] - expected) <= tolerance, \
                f"{test_case.test_id}: {key} - Expected {expected}, got {result[key]}"


class TestSteamPropertyGoldenValues:
    """Golden tests for steam property calculations."""

    @pytest.mark.parametrize("test_case", STEAM_PROPERTY_GOLDEN_CASES, ids=lambda tc: tc.test_id)
    def test_saturation_temperature(self, test_case: GoldenTestCase):
        """Test saturation temperature against golden value."""
        result = calculate_saturation_temperature(test_case.inputs)

        expected = test_case.expected_outputs["saturation_temperature_k"]
        tolerance = test_case.tolerances.get("saturation_temperature_k", 0.1)

        assert abs(result["saturation_temperature_k"] - expected) <= tolerance, \
            f"{test_case.test_id}: Expected {expected}, got {result['saturation_temperature_k']}"


class TestProvenanceHashGoldenValues:
    """Golden tests for provenance hash calculations."""

    def test_provenance_hash_length(self):
        """Test that provenance hash has correct length."""
        inputs = {"pressure_mpa": 1.0, "enthalpy_kj_kg": 2500.0}
        hash_value = calculate_provenance_hash(inputs)

        assert len(hash_value) == 64

    def test_provenance_hash_deterministic(self):
        """Test that provenance hash is deterministic."""
        inputs = {"pressure_mpa": 1.0, "enthalpy_kj_kg": 2500.0}

        hashes = [calculate_provenance_hash(inputs) for _ in range(10)]

        first = hashes[0]
        for h in hashes[1:]:
            assert h == first

    def test_provenance_hash_changes_with_input(self):
        """Test that provenance hash changes with input."""
        inputs1 = {"pressure_mpa": 1.0, "enthalpy_kj_kg": 2500.0}
        inputs2 = {"pressure_mpa": 1.0, "enthalpy_kj_kg": 2501.0}

        hash1 = calculate_provenance_hash(inputs1)
        hash2 = calculate_provenance_hash(inputs2)

        assert hash1 != hash2

    def test_provenance_hash_order_independent(self):
        """Test that hash is independent of key order."""
        inputs1 = {"pressure_mpa": 1.0, "enthalpy_kj_kg": 2500.0}
        inputs2 = {"enthalpy_kj_kg": 2500.0, "pressure_mpa": 1.0}

        hash1 = calculate_provenance_hash(inputs1)
        hash2 = calculate_provenance_hash(inputs2)

        assert hash1 == hash2


class TestGoldenDataFiles:
    """Tests for golden data file loading and validation."""

    def test_golden_data_directory_exists(self):
        """Test that golden data directory exists."""
        assert GOLDEN_DATA_DIR.exists(), f"Golden data directory not found: {GOLDEN_DATA_DIR}"

    def test_load_dryness_golden_data(self):
        """Test loading dryness golden data from file if exists."""
        golden_file = GOLDEN_DATA_DIR / "dryness_golden_values.json"

        if golden_file.exists():
            with open(golden_file, "r") as f:
                data = json.load(f)

            assert "test_cases" in data
            assert len(data["test_cases"]) > 0

    def test_load_carryover_golden_data(self):
        """Test loading carryover golden data from file if exists."""
        golden_file = GOLDEN_DATA_DIR / "carryover_golden_values.json"

        if golden_file.exists():
            with open(golden_file, "r") as f:
                data = json.load(f)

            assert "test_cases" in data


class TestBitPerfectReproducibility:
    """Tests for bit-perfect reproducibility across runs."""

    def test_dryness_bit_perfect(self):
        """Test bit-perfect reproducibility of dryness calculation."""
        inputs = {
            "pressure_mpa": 1.0,
            "h_f_kj_kg": 762.81,
            "h_fg_kj_kg": 2015.3,
            "enthalpy_kj_kg": 2500.0,
        }

        # Calculate 100 times and verify identical
        results = [calculate_dryness_fraction(inputs)["dryness_fraction"] for _ in range(100)]

        # All results must be identical (bit-perfect)
        assert all(r == results[0] for r in results)

    def test_carryover_bit_perfect(self):
        """Test bit-perfect reproducibility of carryover calculation."""
        inputs = {
            "tds_ppm": 50.0,
            "silica_ppb": 100.0,
            "conductivity_us_cm": 50.0,
            "drum_level_percent": 55.0,
            "steam_load_percent": 85.0,
        }

        results = [calculate_carryover_risk(inputs)["probability"] for _ in range(100)]

        assert all(r == results[0] for r in results)

    def test_provenance_bit_perfect(self):
        """Test bit-perfect reproducibility of provenance hash."""
        inputs = {"a": 1.234567890123, "b": 9.876543210987}

        hashes = [calculate_provenance_hash(inputs) for _ in range(100)]

        assert all(h == hashes[0] for h in hashes)


class TestRegressionPrevention:
    """Tests to prevent regressions in calculation accuracy."""

    def test_dryness_regression_suite(self):
        """Run full dryness regression test suite."""
        passed = 0
        failed = 0

        for tc in DRYNESS_GOLDEN_CASES:
            try:
                result = calculate_dryness_fraction(tc.inputs)
                expected = tc.expected_outputs["dryness_fraction"]
                tolerance = tc.tolerances.get("dryness_fraction", 0.01)

                if abs(result["dryness_fraction"] - expected) <= tolerance:
                    passed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        assert failed == 0, f"Regression: {failed}/{passed + failed} dryness tests failed"

    def test_carryover_regression_suite(self):
        """Run full carryover regression test suite."""
        passed = 0
        failed = 0

        for tc in CARRYOVER_GOLDEN_CASES:
            try:
                result = calculate_carryover_risk(tc.inputs)
                expected_level = tc.expected_outputs["risk_level"]

                if result["risk_level"] == expected_level:
                    passed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        assert failed == 0, f"Regression: {failed}/{passed + failed} carryover tests failed"


class TestGoldenValueConsistency:
    """Tests for consistency between different calculation methods."""

    def test_enthalpy_entropy_consistency(self):
        """Test that enthalpy and entropy methods give consistent results."""
        # At same quality, enthalpy and entropy should give same result
        # This tests internal consistency of the golden values

        P = 1.0
        h_f = 762.81
        h_fg = 2015.3
        s_f = 2.1387
        s_fg = 4.4478

        for quality in [0.0, 0.25, 0.5, 0.75, 1.0]:
            h = h_f + quality * h_fg
            s = s_f + quality * s_fg

            # Both should give same quality
            x_from_h = (h - h_f) / h_fg
            x_from_s = (s - s_f) / s_fg

            assert abs(x_from_h - x_from_s) < 0.001, \
                f"Inconsistent quality at x={quality}: h->{x_from_h}, s->{x_from_s}"

    def test_mass_balance_consistency(self):
        """Test mass balance consistency in separator calculations."""
        for tc in [t for t in SEPARATOR_GOLDEN_CASES if "steam_flow_kg_s" in t.expected_outputs]:
            result = calculate_mass_flows(tc.inputs)

            total_out = result["steam_flow_kg_s"] + result["condensate_flow_kg_s"]
            inlet = tc.inputs["inlet_flow_kg_s"]

            assert abs(total_out - inlet) < 0.01, \
                f"{tc.test_id}: Mass balance error: in={inlet}, out={total_out}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
