"""
Unit Tests: Energy Loss Calculator
Author: GL-TestEngineer
"""
import pytest
import math
import hashlib
import json
from conftest import TrapFailureMode, MockTrapData


class EnergyLossCalculator:
    def __init__(self):
        self.napier_constant = 0.0413
        self.discharge_coefficient = 0.62

    def calculate_steam_loss_napier(self, orifice_mm, p_up, p_down):
        if p_up <= 0 or orifice_mm <= 0: return 0.0
        area = math.pi * (orifice_mm / 2) ** 2
        dp = p_up - p_down
        if dp <= 0: return 0.0
        return max(0.0, self.discharge_coefficient * self.napier_constant * area * dp / 1000)

    def calculate_energy_loss(self, steam_loss, pressure):
        if steam_loss <= 0: return 0.0
        return steam_loss * (2258.0 - (pressure - 0.1) * 95.0)

    def calculate_roi(self, savings, cost):
        if savings <= 0: return float("inf")
        return (cost / savings) * 12

    def analyze_trap(self, trap, mode):
        loss = 0.0 if mode == TrapFailureMode.NORMAL else self.calculate_steam_loss_napier(trap.orifice_diameter_mm, trap.inlet_pressure_mpa, trap.outlet_pressure_mpa)
        energy = self.calculate_energy_loss(loss, trap.inlet_pressure_mpa)
        phash = hashlib.sha256(json.dumps({"id": trap.trap_id, "loss": loss}, sort_keys=True).encode()).hexdigest()
        return {"trap_id": trap.trap_id, "failure_mode": mode.name, "steam_loss_kg_s": loss, "energy_loss_kw": energy, "provenance_hash": phash}


@pytest.fixture
def calculator(): return EnergyLossCalculator()


class TestSteamLoss:
    def test_positive(self, calculator): assert calculator.calculate_steam_loss_napier(10.0, 1.0, 0.1) > 0
    def test_zero_pressure(self, calculator): assert calculator.calculate_steam_loss_napier(10.0, 0.0, 0.0) == 0.0
    def test_zero_orifice(self, calculator): assert calculator.calculate_steam_loss_napier(0.0, 1.0, 0.1) == 0.0


class TestEnergyLoss:
    def test_positive(self, calculator): assert calculator.calculate_energy_loss(0.01, 1.0) > 0
    def test_zero(self, calculator): assert calculator.calculate_energy_loss(0.0, 1.0) == 0.0


class TestROI:
    def test_positive(self, calculator): assert calculator.calculate_roi(1000.0, 500.0) > 0


class TestAnalysis:
    def test_healthy(self, calculator, healthy_trap):
        r = calculator.analyze_trap(healthy_trap, TrapFailureMode.NORMAL)
        assert r["failure_mode"] == "NORMAL"

    def test_failed(self, calculator, failed_open_trap):
        r = calculator.analyze_trap(failed_open_trap, TrapFailureMode.FAILED_OPEN)
        assert r["steam_loss_kg_s"] > 0


class TestProvenance:
    def test_hash(self, calculator, healthy_trap):
        r = calculator.analyze_trap(healthy_trap, TrapFailureMode.NORMAL)
        assert len(r["provenance_hash"]) == 64
