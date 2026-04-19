"""Integration Tests: Trapcatcher Orchestrator"""
import pytest
import hashlib
import json
from typing import Dict, List, Any
from datetime import datetime, timezone
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import TrapType, TrapFailureMode, MaintenancePriority, MockTrapData


class TrapcatcherOrchestrator:
    def __init__(self, config):
        self.config = config
        self.agent_id = config.get("agent_id", "GL-008")

    async def diagnose_trap(self, trap):
        acoustic = self._analyze_acoustic(trap)
        temp = self._analyze_temperature(trap)
        mode = self._combine_diagnostics(acoustic, temp)
        energy_loss = self._calculate_energy_loss(trap, mode)
        priority = self._determine_priority(mode, energy_loss)
        phash = self._generate_provenance_hash(trap, mode, energy_loss)
        return {"trap_id": trap.trap_id, "failure_mode": mode.name, "energy_loss_kw": energy_loss, "priority": priority.name, "provenance_hash": phash}

    async def diagnose_fleet(self, traps):
        diagnoses = [await self.diagnose_trap(t) for t in traps]
        return {"diagnosis_count": len(diagnoses), "diagnoses": diagnoses}

    def _analyze_acoustic(self, trap):
        if trap.ultrasonic_db is None:
            return {"status": "NO_DATA"}
        if trap.ultrasonic_db <= 70:
            return {"status": "NORMAL"}
        elif trap.ultrasonic_db <= 85:
            return {"status": "WARNING"}
        return {"status": "FAILED"}

    def _analyze_temperature(self, trap):
        diff = trap.inlet_temperature_k - trap.outlet_temperature_k
        if diff < 2:
            return {"status": "FAILED_OPEN"}
        elif diff > 50:
            return {"status": "FAILED_CLOSED"}
        return {"status": "NORMAL"}

    def _combine_diagnostics(self, acoustic, temp):
        if acoustic.get("status") == "FAILED" or temp.get("status") == "FAILED_OPEN":
            return TrapFailureMode.FAILED_OPEN
        elif temp.get("status") == "FAILED_CLOSED":
            return TrapFailureMode.FAILED_CLOSED
        elif acoustic.get("status") == "WARNING":
            return TrapFailureMode.BLOW_THROUGH
        return TrapFailureMode.NORMAL

    def _calculate_energy_loss(self, trap, mode):
        if mode == TrapFailureMode.NORMAL or trap.orifice_diameter_mm <= 0 or trap.inlet_pressure_mpa <= 0:
            return 0.0
        area = math.pi * (trap.orifice_diameter_mm / 2) ** 2
        dp = trap.inlet_pressure_mpa - trap.outlet_pressure_mpa
        steam_loss = 0.62 * 0.0413 * area * dp / 1000
        return max(0.0, steam_loss * (2258.0 - (trap.inlet_pressure_mpa - 0.1) * 95.0))

    def _determine_priority(self, mode, energy_loss):
        if mode == TrapFailureMode.NORMAL:
            return MaintenancePriority.NONE
        if energy_loss > 100:
            return MaintenancePriority.CRITICAL
        elif energy_loss > 50:
            return MaintenancePriority.HIGH
        elif energy_loss > 20:
            return MaintenancePriority.MEDIUM
        return MaintenancePriority.LOW

    def _generate_provenance_hash(self, trap, mode, energy_loss):
        data = {"trap_id": trap.trap_id, "failure_mode": mode.name, "energy_loss": energy_loss}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def orchestrator(agent_config):
    return TrapcatcherOrchestrator(agent_config)


class TestSingleTrapDiagnosis:
    @pytest.mark.asyncio
    async def test_healthy_trap(self, orchestrator, healthy_trap):
        result = await orchestrator.diagnose_trap(healthy_trap)
        assert result["failure_mode"] == "NORMAL"

    @pytest.mark.asyncio
    async def test_failed_open_trap(self, orchestrator, failed_open_trap):
        result = await orchestrator.diagnose_trap(failed_open_trap)
        assert result["failure_mode"] == "FAILED_OPEN"

    @pytest.mark.asyncio
    async def test_provenance_hash(self, orchestrator, healthy_trap):
        result = await orchestrator.diagnose_trap(healthy_trap)
        assert len(result["provenance_hash"]) == 64


class TestFleetDiagnosis:
    @pytest.mark.asyncio
    async def test_fleet_diagnosis(self, orchestrator, trap_fleet):
        result = await orchestrator.diagnose_fleet(trap_fleet)
        assert result["diagnosis_count"] == len(trap_fleet)

    @pytest.mark.asyncio
    async def test_empty_fleet(self, orchestrator):
        result = await orchestrator.diagnose_fleet([])
        assert result["diagnosis_count"] == 0


class TestReproducibility:
    @pytest.mark.asyncio
    async def test_reproducibility(self, orchestrator, healthy_trap):
        r1 = await orchestrator.diagnose_trap(healthy_trap)
        r2 = await orchestrator.diagnose_trap(healthy_trap)
        assert r1["provenance_hash"] == r2["provenance_hash"]
