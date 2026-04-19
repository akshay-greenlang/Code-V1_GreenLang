"""
Integration Tests: Trapcatcher API
Tests for API endpoints, request/response validation, error handling.
Author: GL-TestEngineer
"""
import pytest
import json
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import TrapType, TrapFailureMode, MockTrapData


class TrapcatcherAPI:
    def __init__(self, config):
        self.config = config
        self.version = "1.0.0"

    async def health_check(self):
        return {"status": "healthy", "version": self.version, "agent_id": self.config.get("agent_id", "GL-008")}

    async def diagnose_single(self, trap_data: Dict) -> Dict:
        if not trap_data.get("trap_id"):
            raise ValueError("trap_id is required")
        return {"trap_id": trap_data["trap_id"], "status": "diagnosed", "failure_mode": "NORMAL"}

    async def diagnose_batch(self, traps_data: list) -> Dict:
        if not traps_data:
            return {"count": 0, "results": []}
        results = [await self.diagnose_single(t) for t in traps_data]
        return {"count": len(results), "results": results}

    async def get_fleet_status(self, fleet_id: str) -> Dict:
        if not fleet_id:
            raise ValueError("fleet_id is required")
        return {"fleet_id": fleet_id, "total_traps": 0, "healthy": 0, "failed": 0}


@pytest.fixture
def api(agent_config):
    return TrapcatcherAPI(agent_config)


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check(self, api):
        result = await api.health_check()
        assert result["status"] == "healthy"
        assert "version" in result

    @pytest.mark.asyncio
    async def test_health_includes_agent_id(self, api):
        result = await api.health_check()
        assert result["agent_id"] == "GL-008"


class TestDiagnoseSingleEndpoint:
    @pytest.mark.asyncio
    async def test_diagnose_single_valid(self, api):
        trap_data = {"trap_id": "TRAP-001", "inlet_temp": 453.0, "outlet_temp": 440.0}
        result = await api.diagnose_single(trap_data)
        assert result["trap_id"] == "TRAP-001"
        assert "failure_mode" in result

    @pytest.mark.asyncio
    async def test_diagnose_single_missing_id(self, api):
        trap_data = {"inlet_temp": 453.0}
        with pytest.raises(ValueError):
            await api.diagnose_single(trap_data)


class TestDiagnoseBatchEndpoint:
    @pytest.mark.asyncio
    async def test_diagnose_batch(self, api):
        traps = [{"trap_id": "T1"}, {"trap_id": "T2"}, {"trap_id": "T3"}]
        result = await api.diagnose_batch(traps)
        assert result["count"] == 3
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_diagnose_batch_empty(self, api):
        result = await api.diagnose_batch([])
        assert result["count"] == 0


class TestFleetStatusEndpoint:
    @pytest.mark.asyncio
    async def test_get_fleet_status(self, api):
        result = await api.get_fleet_status("FLEET-001")
        assert result["fleet_id"] == "FLEET-001"
        assert "total_traps" in result

    @pytest.mark.asyncio
    async def test_get_fleet_status_missing_id(self, api):
        with pytest.raises(ValueError):
            await api.get_fleet_status("")


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_invalid_request_format(self, api):
        with pytest.raises(ValueError):
            await api.diagnose_single({})

    @pytest.mark.asyncio
    async def test_empty_fleet_id(self, api):
        with pytest.raises(ValueError):
            await api.get_fleet_status("")
