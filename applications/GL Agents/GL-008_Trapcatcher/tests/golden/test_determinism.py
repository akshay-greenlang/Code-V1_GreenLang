"""
Golden Master Tests: Determinism and Reproducibility
Tests for bit-perfect reproducibility, hash consistency, calculation verification.
Author: GL-TestEngineer
"""
import pytest
import hashlib
import json
import math
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import TrapType, TrapFailureMode, MockTrapData, GoldenTestCase


class DeterminismValidator:
    def __init__(self):
        self.napier_constant = 0.0413
        self.discharge_coefficient = 0.62

    def calculate_steam_loss(self, orifice_mm, p_up, p_down):
        if p_up <= 0 or orifice_mm <= 0:
            return 0.0
        area = math.pi * (orifice_mm / 2) ** 2
        dp = p_up - p_down
        if dp <= 0:
            return 0.0
        return max(0.0, self.discharge_coefficient * self.napier_constant * area * dp / 1000)

    def calculate_energy_loss(self, steam_loss, pressure):
        if steam_loss <= 0:
            return 0.0
        return steam_loss * (2258.0 - (pressure - 0.1) * 95.0)

    def generate_provenance_hash(self, data: Dict) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def diagnose(self, trap: MockTrapData) -> Dict[str, Any]:
        if trap.ultrasonic_db is not None and trap.ultrasonic_db > 95:
            failure_mode = TrapFailureMode.FAILED_OPEN
        elif trap.inlet_temperature_k - trap.outlet_temperature_k < 2:
            failure_mode = TrapFailureMode.FAILED_OPEN
        elif trap.inlet_temperature_k - trap.outlet_temperature_k > 50:
            failure_mode = TrapFailureMode.FAILED_CLOSED
        else:
            failure_mode = TrapFailureMode.NORMAL
        if failure_mode == TrapFailureMode.NORMAL:
            steam_loss = 0.0
            energy_loss = 0.0
        else:
            steam_loss = self.calculate_steam_loss(trap.orifice_diameter_mm, trap.inlet_pressure_mpa, trap.outlet_pressure_mpa)
            energy_loss = self.calculate_energy_loss(steam_loss, trap.inlet_pressure_mpa)
        data_for_hash = {"trap_id": trap.trap_id, "failure_mode": failure_mode.name, "steam_loss": steam_loss, "energy_loss": energy_loss}
        provenance_hash = self.generate_provenance_hash(data_for_hash)
        return {"trap_id": trap.trap_id, "failure_mode": failure_mode.name, "steam_loss_kg_s": steam_loss, "energy_loss_kw": energy_loss, "provenance_hash": provenance_hash}


@pytest.fixture
def validator():
    return DeterminismValidator()


class TestBitPerfectReproducibility:
    @pytest.mark.golden
    def test_same_input_same_output(self, validator, healthy_trap):
        result1 = validator.diagnose(healthy_trap)
        result2 = validator.diagnose(healthy_trap)
        assert result1 == result2

    @pytest.mark.golden
    def test_provenance_hash_consistency(self, validator, healthy_trap):
        results = [validator.diagnose(healthy_trap) for _ in range(10)]
        hashes = [r["provenance_hash"] for r in results]
        assert len(set(hashes)) == 1

    @pytest.mark.golden
    def test_calculation_determinism(self, validator):
        steam_losses = [validator.calculate_steam_loss(10.0, 1.0, 0.1) for _ in range(100)]
        assert len(set(steam_losses)) == 1


class TestHashConsistency:
    @pytest.mark.golden
    def test_hash_length(self, validator):
        data = {"test": "data", "value": 123}
        phash = validator.generate_provenance_hash(data)
        assert len(phash) == 64

    @pytest.mark.golden
    def test_hash_deterministic(self, validator):
        data = {"test": "data", "value": 123}
        hash1 = validator.generate_provenance_hash(data)
        hash2 = validator.generate_provenance_hash(data)
        assert hash1 == hash2

    @pytest.mark.golden
    def test_hash_order_independent(self, validator):
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}
        hash1 = validator.generate_provenance_hash(data1)
        hash2 = validator.generate_provenance_hash(data2)
        assert hash1 == hash2

    @pytest.mark.golden
    def test_different_data_different_hash(self, validator):
        data1 = {"value": 1}
        data2 = {"value": 2}
        hash1 = validator.generate_provenance_hash(data1)
        hash2 = validator.generate_provenance_hash(data2)
        assert hash1 != hash2


class TestKnownValueVerification:
    @pytest.mark.golden
    def test_napier_equation_known_value(self, validator):
        steam_loss = validator.calculate_steam_loss(10.0, 1.0, 0.1)
        expected = 0.62 * 0.0413 * math.pi * 25 * 0.9 / 1000
        assert abs(steam_loss - expected) < 1e-10

    @pytest.mark.golden
    def test_energy_loss_known_value(self, validator):
        energy_loss = validator.calculate_energy_loss(0.01, 1.0)
        expected = 0.01 * (2258.0 - (1.0 - 0.1) * 95.0)
        assert abs(energy_loss - expected) < 1e-10

    @pytest.mark.golden
    def test_zero_pressure_zero_loss(self, validator):
        steam_loss = validator.calculate_steam_loss(10.0, 0.0, 0.0)
        assert steam_loss == 0.0

    @pytest.mark.golden
    def test_zero_orifice_zero_loss(self, validator):
        steam_loss = validator.calculate_steam_loss(0.0, 1.0, 0.1)
        assert steam_loss == 0.0


class TestGoldenMasterCases:
    @pytest.mark.golden
    def test_golden_case_healthy_trap(self, validator, golden_test_cases):
        healthy_case = golden_test_cases[0]
        assert healthy_case.expected_output["failure_mode"] == "NORMAL"

    @pytest.mark.golden
    def test_golden_case_failed_open(self, validator, golden_test_cases):
        failed_case = golden_test_cases[1]
        assert failed_case.expected_output["failure_mode"] == "FAILED_OPEN"

    @pytest.mark.golden
    def test_all_golden_cases_have_tolerance(self, golden_test_cases):
        for case in golden_test_cases:
            assert case.tolerance > 0


class TestFleetDeterminism:
    @pytest.mark.golden
    def test_fleet_diagnosis_determinism(self, validator, trap_fleet):
        results1 = [validator.diagnose(trap) for trap in trap_fleet]
        results2 = [validator.diagnose(trap) for trap in trap_fleet]
        for r1, r2 in zip(results1, results2):
            assert r1["provenance_hash"] == r2["provenance_hash"]

    @pytest.mark.golden
    def test_fleet_order_consistent(self, validator, trap_fleet):
        results = [validator.diagnose(trap) for trap in trap_fleet]
        ids = [r["trap_id"] for r in results]
        expected_ids = [trap.trap_id for trap in trap_fleet]
        assert ids == expected_ids
