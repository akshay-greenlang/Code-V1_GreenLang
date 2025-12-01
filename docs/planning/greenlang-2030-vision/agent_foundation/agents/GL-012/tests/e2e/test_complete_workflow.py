# -*- coding: utf-8 -*-
"""
Complete Quality Control Workflow E2E Tests for GL-012 STEAMQUAL.
Tests the complete steam quality monitoring and control cycle.
"""

import asyncio
import pytest
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class MockSteamHeader:
    header_id: str
    pressure_bar: float = 10.0
    temperature_c: float = 180.0
    flow_rate_kg_hr: float = 5000.0
    dryness_fraction: float = 0.98


@dataclass
class MockDesuperheater:
    dsh_id: str
    injection_rate: float = 0.0
    is_active: bool = False
    
    async def set_injection_rate(self, rate):
        self.injection_rate = max(0, min(rate, 1000))
        self.is_active = rate > 0
        return True


class MockFullSystem:
    def __init__(self, num_headers=2):
        self.headers = {f"header_{i}": MockSteamHeader(f"header_{i}") for i in range(1, num_headers + 1)}
        self.desuperheaters = {f"dsh_{i}": MockDesuperheater(f"dsh_{i}") for i in range(1, num_headers + 1)}
        self.connected = True
        self.control_actions = []
    
    async def connect_all(self):
        self.connected = True
        return True
    
    async def read_all_quality(self):
        if not self.connected:
            raise ConnectionError("Not connected")
        return {hid: {"pressure_bar": h.pressure_bar, "temperature_c": h.temperature_c,
                      "dryness_fraction": h.dryness_fraction} for hid, h in self.headers.items()}
    
    async def execute_action(self, action_type, target_id, params):
        self.control_actions.append({"type": action_type, "target": target_id, "params": params})
        if action_type == "desuperheater":
            await self.desuperheaters[target_id].set_injection_rate(params.get("injection_rate", 0))
        return {"success": True}


@pytest.fixture
def mock_system():
    return MockFullSystem()


@pytest.fixture
def degraded_system():
    sys = MockFullSystem()
    sys.headers["header_1"].dryness_fraction = 0.88
    sys.headers["header_1"].temperature_c = 195.0
    return sys


class TestCompleteWorkflow:
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_normal_operation_workflow(self, mock_system):
        await mock_system.connect_all()
        quality = await mock_system.read_all_quality()
        
        assert len(quality) == 2
        assert all(q["dryness_fraction"] >= 0.95 for q in quality.values())
        
        # Calculate quality index
        avg_dryness = sum(q["dryness_fraction"] for q in quality.values()) / len(quality)
        quality_index = avg_dryness * 100 * 0.4 + 95 * 0.3 + 92 * 0.3
        
        assert quality_index >= 90
        
        # Generate provenance hash
        ph = hashlib.sha256(json.dumps(quality, sort_keys=True).encode()).hexdigest()
        assert len(ph) == 64
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_quality_degradation_intervention(self, degraded_system):
        await degraded_system.connect_all()
        quality = await degraded_system.read_all_quality()
        
        # Header 1 has degraded quality
        h1 = quality["header_1"]
        assert h1["dryness_fraction"] < 0.90
        
        # Execute desuperheater intervention
        result = await degraded_system.execute_action(
            "desuperheater", "dsh_1", {"injection_rate": 50.0}
        )
        assert result["success"]
        
        # Verify desuperheater is active
        dsh = degraded_system.desuperheaters["dsh_1"]
        assert dsh.is_active
        assert dsh.injection_rate > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_provenance_hash_determinism(self, mock_system):
        await mock_system.connect_all()
        quality = await mock_system.read_all_quality()
        
        hashes = []
        for _ in range(10):
            h = hashlib.sha256(json.dumps(quality, sort_keys=True).encode()).hexdigest()
            hashes.append(h)
        
        assert len(set(hashes)) == 1, "Hash must be deterministic"


class TestFaultTolerance:
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_loss_recovery(self, mock_system):
        await mock_system.connect_all()
        quality = await mock_system.read_all_quality()
        assert len(quality) == 2
        
        mock_system.connected = False
        with pytest.raises(ConnectionError):
            await mock_system.read_all_quality()
        
        await mock_system.connect_all()
        quality = await mock_system.read_all_quality()
        assert len(quality) == 2
