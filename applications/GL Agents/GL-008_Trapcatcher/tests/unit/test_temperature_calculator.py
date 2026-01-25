"""
Unit Tests: Temperature Calculator
Tests for differential analysis, subcooling detection, and temperature diagnostics.
Author: GL-TestEngineer
"""
import pytest
import math
import hashlib
import json
from typing import Dict, Any
from conftest import TrapType, TrapFailureMode, MockTrapData


class TemperatureCalculator:
    SATURATION_TEMP_AT_1MPA = 453.0  # Kelvin
    SUBCOOLING_NORMAL_MAX_K = 15.0
    SUPERHEAT_THRESHOLD_K = 5.0

    def __init__(self):
        self.measurement_uncertainty = 0.02

    def calculate_subcooling(self, inlet_temp_k, outlet_temp_k):
        if inlet_temp_k <= 0 or outlet_temp_k <= 0:
            raise ValueError("Temperature must be positive")
        return inlet_temp_k - outlet_temp_k

    def calculate_superheat(self, outlet_temp_k, saturation_temp_k):
        if outlet_temp_k <= 0 or saturation_temp_k <= 0:
            raise ValueError("Temperature must be positive")
        return outlet_temp_k - saturation_temp_k

    def analyze_temperature_differential(self, inlet_temp_k, outlet_temp_k):
        if inlet_temp_k <= 0 or outlet_temp_k <= 0:
            raise ValueError("Temperature must be positive")
        diff = inlet_temp_k - outlet_temp_k
        if diff < 2:
            status = "FAILED_OPEN"
            failure_mode = TrapFailureMode.FAILED_OPEN
        elif diff > 50:
            status = "FAILED_CLOSED"
            failure_mode = TrapFailureMode.FAILED_CLOSED
        elif diff > 20:
            status = "WARNING"
            failure_mode = TrapFailureMode.PARTIAL_BLOCKAGE
        else:
            status = "NORMAL"
            failure_mode = TrapFailureMode.NORMAL
        return {"inlet_temp_k": inlet_temp_k, "outlet_temp_k": outlet_temp_k, "differential_k": diff, "status": status, "failure_mode": failure_mode.name}

    def detect_subcooling_status(self, subcooling_k):
        if subcooling_k < 0:
            return {"status": "SUPERHEAT", "indication": "possible_leak"}
        elif subcooling_k <= self.SUBCOOLING_NORMAL_MAX_K:
            return {"status": "NORMAL", "indication": "healthy"}
        else:
            return {"status": "EXCESSIVE", "indication": "possible_blockage"}

    def calculate_heat_loss_rate(self, temp_diff_k, flow_rate_kg_s):
        if temp_diff_k <= 0 or flow_rate_kg_s <= 0:
            return 0.0
        specific_heat = 4.186  # kJ/kg*K for water
        return temp_diff_k * flow_rate_kg_s * specific_heat

    def analyze_trap_temperature(self, trap):
        diff_analysis = self.analyze_temperature_differential(trap.inlet_temperature_k, trap.outlet_temperature_k)
        subcooling = self.calculate_subcooling(trap.inlet_temperature_k, trap.outlet_temperature_k)
        subcooling_status = self.detect_subcooling_status(subcooling)
        data_for_hash = {"trap_id": trap.trap_id, "inlet_temp": trap.inlet_temperature_k, "outlet_temp": trap.outlet_temperature_k}
        provenance_hash = hashlib.sha256(json.dumps(data_for_hash, sort_keys=True).encode()).hexdigest()
        return {"trap_id": trap.trap_id, "temperature_analysis": diff_analysis, "subcooling_k": subcooling, "subcooling_status": subcooling_status, "provenance_hash": provenance_hash}


@pytest.fixture
def calculator():
    return TemperatureCalculator()


class TestSubcoolingCalculation:
    def test_positive_subcooling(self, calculator):
        result = calculator.calculate_subcooling(453.0, 440.0)
        assert result == 13.0

    def test_zero_subcooling(self, calculator):
        result = calculator.calculate_subcooling(453.0, 453.0)
        assert result == 0.0

    def test_negative_subcooling_superheat(self, calculator):
        result = calculator.calculate_subcooling(453.0, 460.0)
        assert result == -7.0

    def test_invalid_temperature_raises_error(self, calculator):
        with pytest.raises(ValueError):
            calculator.calculate_subcooling(-10.0, 440.0)


class TestTemperatureDifferentialAnalysis:
    def test_normal_differential(self, calculator):
        result = calculator.analyze_temperature_differential(453.0, 440.0)
        assert result["status"] == "NORMAL"

    def test_failed_open_low_differential(self, calculator):
        result = calculator.analyze_temperature_differential(453.0, 452.0)
        assert result["status"] == "FAILED_OPEN"

    def test_failed_closed_high_differential(self, calculator):
        result = calculator.analyze_temperature_differential(453.0, 400.0)
        assert result["status"] == "FAILED_CLOSED"

    def test_warning_elevated_differential(self, calculator):
        result = calculator.analyze_temperature_differential(453.0, 430.0)
        assert result["status"] == "WARNING"

    def test_invalid_temperature_raises_error(self, calculator):
        with pytest.raises(ValueError):
            calculator.analyze_temperature_differential(0.0, 440.0)


class TestSubcoolingStatus:
    def test_normal_subcooling(self, calculator):
        result = calculator.detect_subcooling_status(10.0)
        assert result["status"] == "NORMAL"

    def test_excessive_subcooling(self, calculator):
        result = calculator.detect_subcooling_status(25.0)
        assert result["status"] == "EXCESSIVE"

    def test_superheat_condition(self, calculator):
        result = calculator.detect_subcooling_status(-5.0)
        assert result["status"] == "SUPERHEAT"


class TestHeatLossRate:
    def test_positive_heat_loss(self, calculator):
        result = calculator.calculate_heat_loss_rate(10.0, 0.5)
        assert result > 0

    def test_zero_temp_diff_returns_zero(self, calculator):
        result = calculator.calculate_heat_loss_rate(0.0, 0.5)
        assert result == 0.0

    def test_zero_flow_returns_zero(self, calculator):
        result = calculator.calculate_heat_loss_rate(10.0, 0.0)
        assert result == 0.0


class TestTrapTemperatureAnalysis:
    def test_healthy_trap_analysis(self, calculator, healthy_trap):
        result = calculator.analyze_trap_temperature(healthy_trap)
        assert result["temperature_analysis"]["status"] == "NORMAL"
        assert len(result["provenance_hash"]) == 64

    def test_failed_open_trap_analysis(self, calculator, failed_open_trap):
        result = calculator.analyze_trap_temperature(failed_open_trap)
        assert result["temperature_analysis"]["status"] == "FAILED_OPEN"

    def test_failed_closed_trap_analysis(self, calculator, failed_closed_trap):
        result = calculator.analyze_trap_temperature(failed_closed_trap)
        assert result["temperature_analysis"]["status"] == "FAILED_CLOSED"

    def test_provenance_hash_deterministic(self, calculator, healthy_trap):
        result1 = calculator.analyze_trap_temperature(healthy_trap)
        result2 = calculator.analyze_trap_temperature(healthy_trap)
        assert result1["provenance_hash"] == result2["provenance_hash"]
