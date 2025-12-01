# -*- coding: utf-8 -*-
"""
E2E Test Fixtures and Configuration for GL-002 FLAMEGUARD BoilerEfficiencyOptimizer.

Comprehensive fixtures for end-to-end testing including:
- Full system mock setup (orchestrator, SCADA, DCS, fuel management)
- Test data scenarios (normal, high-load, low-efficiency, high-emissions)
- Performance timing utilities
- Multi-boiler coordination fixtures
- Determinism validation utilities
- Provenance tracking helpers

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import asyncio
import os
import random
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with E2E-specific markers."""
    markers = [
        "e2e: End-to-end workflow tests",
        "workflow: Complete workflow tests",
        "optimization_loop: Real-time optimization loop tests",
        "fault_tolerance: Fault tolerance and recovery tests",
        "multi_boiler: Multi-boiler coordination tests",
        "performance: Performance benchmark tests",
        "determinism: Determinism verification tests",
        "provenance: Provenance tracking tests",
        "emergency: Emergency scenario tests",
        "combustion: Combustion optimization tests",
        "slow: Slow-running tests (>30s)",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_addoption(parser):
    """Add custom command-line options for E2E tests."""
    parser.addoption(
        "--extended-cycles",
        action="store_true",
        default=False,
        help="Run extended optimization loop tests (1000+ cycles)"
    )
    parser.addoption(
        "--stress-test",
        action="store_true",
        default=False,
        help="Run stress tests with high load"
    )


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# =============================================================================
# ENUMS FOR TEST SCENARIOS
# =============================================================================

class BoilerTestScenario(str, Enum):
    """Boiler test scenario types."""
    NORMAL_OPERATION = "normal_operation"
    HIGH_LOAD = "high_load"
    LOW_EFFICIENCY = "low_efficiency"
    HIGH_EMISSIONS = "high_emissions"
    FUEL_SWITCH = "fuel_switch"
    COMBUSTION_OPTIMIZATION = "combustion_optimization"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    SENSOR_FAILURE = "sensor_failure"
    LOAD_RAMP = "load_ramp"
    MULTI_BOILER_COORDINATION = "multi_boiler_coordination"


class ControlAction(str, Enum):
    """Control action types."""
    NO_ACTION = "no_action"
    FUEL_VALVE_ADJUST = "fuel_valve_adjust"
    AIR_DAMPER_ADJUST = "air_damper_adjust"
    FEEDWATER_ADJUST = "feedwater_adjust"
    BURNER_MODULATION = "burner_modulation"
    LOAD_SHIFT = "load_shift"
    EMERGENCY_STOP = "emergency_stop"


# =============================================================================
# DATA CLASSES FOR E2E TESTING
# =============================================================================

@dataclass
class BoilerState:
    """State of a boiler for testing."""
    boiler_id: str
    load_percent: float = 75.0
    efficiency_percent: float = 85.0
    steam_flow_kg_hr: float = 20000.0
    fuel_flow_kg_hr: float = 1500.0
    steam_pressure_bar: float = 35.0
    steam_temperature_c: float = 400.0
    o2_percent: float = 4.5
    co_ppm: float = 15.0
    nox_ppm: float = 22.0
    flue_gas_temp_c: float = 180.0
    excess_air_percent: float = 15.0
    feedwater_temp_c: float = 100.0
    is_running: bool = True
    fuel_type: str = "natural_gas"


@dataclass
class OptimizationCycle:
    """Single optimization cycle data."""
    cycle_number: int
    timestamp: datetime
    sensor_readings: Dict[str, float]
    optimization_actions: List[Dict[str, Any]]
    efficiency_before: float
    efficiency_after: float
    emissions_before: Dict[str, float]
    emissions_after: Dict[str, float]
    execution_time_ms: float
    provenance_hash: str
    determinism_verified: bool = True


@dataclass
class E2ETestMetrics:
    """Metrics collected during E2E test execution."""
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    total_execution_time_ms: float = 0.0
    min_cycle_time_ms: float = float('inf')
    max_cycle_time_ms: float = 0.0
    avg_cycle_time_ms: float = 0.0
    efficiency_improvements: int = 0
    emission_reductions: int = 0
    control_actions_taken: int = 0
    provenance_hashes: List[str] = field(default_factory=list)


# =============================================================================
# MOCK BOILER EFFICIENCY ORCHESTRATOR FOR E2E
# =============================================================================

class MockE2EBoilerEfficiencyOrchestrator:
    """
    Full mock orchestrator for E2E testing of GL-002 FLAMEGUARD.

    Simulates complete boiler efficiency optimization workflow including:
    - Sensor data acquisition from SCADA/DCS
    - Combustion efficiency analysis
    - Emission monitoring and optimization
    - Fuel management optimization
    - Multi-boiler load coordination
    - Control action execution
    - Provenance tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize E2E orchestrator."""
        self._config = config or self._default_config()
        self._state = "initialized"
        self._cycle_count = 0
        self._boilers: Dict[str, BoilerState] = {}
        self._scada_connected = False
        self._dcs_connected = False
        self._fuel_system_connected = False
        self._emissions_system_connected = False
        self._control_mode = "auto"
        self._provenance_chain: List[str] = []
        self._metrics = E2ETestMetrics()
        self._fault_injection: Optional[str] = None

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "agent_id": "GL-002",
            "agent_name": "BoilerEfficiencyOptimizer",
            "target_efficiency_percent": 92.0,
            "max_excess_air_percent": 25.0,
            "min_excess_air_percent": 5.0,
            "max_o2_percent": 6.0,
            "min_o2_percent": 2.0,
            "max_co_ppm": 100.0,
            "max_nox_ppm": 30.0,
            "optimization_interval_seconds": 60,
            "safety_margin_percent": 5.0,
        }

    async def initialize(self) -> Dict[str, Any]:
        """Initialize orchestrator."""
        self._state = "ready"
        return {
            "status": "initialized",
            "config": self._config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def shutdown(self) -> Dict[str, Any]:
        """Shutdown orchestrator."""
        self._state = "shutdown"
        return {
            "status": "shutdown",
            "cycles_completed": self._cycle_count,
            "metrics": self._get_metrics_summary()
        }

    async def connect_scada(self, scada_config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to SCADA system."""
        if self._fault_injection == "scada_connection_failure":
            return {"status": "error", "error": "SCADA connection failed"}

        self._scada_connected = True
        return {
            "status": "connected",
            "protocol": scada_config.get("protocol", "opc_ua"),
            "server_state": "Running"
        }

    async def connect_dcs(self, dcs_config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to DCS system."""
        if self._fault_injection == "dcs_connection_failure":
            return {"status": "error", "error": "DCS connection failed"}

        self._dcs_connected = True
        return {
            "status": "connected",
            "protocol": dcs_config.get("protocol", "modbus"),
            "plc_state": "Running"
        }

    async def connect_fuel_system(self, fuel_config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to fuel management system."""
        if self._fault_injection == "fuel_system_failure":
            return {"status": "error", "error": "Fuel system connection failed"}

        self._fuel_system_connected = True
        return {
            "status": "connected",
            "fuel_types": fuel_config.get("fuel_types", ["natural_gas"]),
            "multi_fuel_enabled": fuel_config.get("multi_fuel_enabled", False)
        }

    async def connect_emissions_system(self, cems_config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to CEMS emissions monitoring system."""
        if self._fault_injection == "cems_failure":
            return {"status": "error", "error": "CEMS connection failed"}

        self._emissions_system_connected = True
        return {
            "status": "connected",
            "analyzers": cems_config.get("analyzers", ["o2", "co", "nox"]),
            "compliance_standard": cems_config.get("compliance_standard", "EPA")
        }

    async def read_boiler_parameters(
        self,
        boiler_id: str
    ) -> Dict[str, Any]:
        """Read boiler operating parameters from sensors."""
        if self._fault_injection == "sensor_failure":
            return {"status": "error", "quality": "BAD", "error": "Sensor failure"}

        # Get or create boiler state
        if boiler_id not in self._boilers:
            self._boilers[boiler_id] = BoilerState(boiler_id=boiler_id)

        boiler = self._boilers[boiler_id]

        # Add some noise for realism
        noise = lambda base, pct: base * (1 + random.uniform(-pct, pct) / 100)

        return {
            "boiler_id": boiler_id,
            "load_percent": noise(boiler.load_percent, 1.0),
            "efficiency_percent": noise(boiler.efficiency_percent, 0.5),
            "steam_flow_kg_hr": noise(boiler.steam_flow_kg_hr, 2.0),
            "fuel_flow_kg_hr": noise(boiler.fuel_flow_kg_hr, 1.5),
            "steam_pressure_bar": noise(boiler.steam_pressure_bar, 1.0),
            "steam_temperature_c": noise(boiler.steam_temperature_c, 0.5),
            "o2_percent": noise(boiler.o2_percent, 2.0),
            "co_ppm": max(0, noise(boiler.co_ppm, 5.0)),
            "nox_ppm": max(0, noise(boiler.nox_ppm, 3.0)),
            "flue_gas_temp_c": noise(boiler.flue_gas_temp_c, 1.0),
            "excess_air_percent": noise(boiler.excess_air_percent, 2.0),
            "feedwater_temp_c": noise(boiler.feedwater_temp_c, 1.0),
            "fuel_type": boiler.fuel_type,
            "is_running": boiler.is_running,
            "quality": "GOOD",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def analyze_combustion_efficiency(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze combustion efficiency and determine optimization actions."""
        o2_percent = parameters.get("o2_percent", 4.5)
        co_ppm = parameters.get("co_ppm", 15.0)
        flue_gas_temp = parameters.get("flue_gas_temp_c", 180.0)
        efficiency = parameters.get("efficiency_percent", 85.0)

        # Calculate excess air from O2
        excess_air = (o2_percent / (21 - o2_percent)) * 100

        # Determine optimization actions
        actions_required = []

        # Check O2 optimization
        if o2_percent > self._config["max_o2_percent"]:
            actions_required.append({
                "action": "reduce_excess_air",
                "reason": f"High O2: {o2_percent:.1f}%",
                "priority": "high"
            })
        elif o2_percent < self._config["min_o2_percent"]:
            actions_required.append({
                "action": "increase_excess_air",
                "reason": f"Low O2: {o2_percent:.1f}%",
                "priority": "high"
            })

        # Check CO levels
        if co_ppm > self._config["max_co_ppm"]:
            actions_required.append({
                "action": "adjust_combustion",
                "reason": f"High CO: {co_ppm:.0f} ppm",
                "priority": "critical"
            })

        # Check efficiency
        if efficiency < self._config["target_efficiency_percent"] - 5:
            actions_required.append({
                "action": "optimize_air_fuel_ratio",
                "reason": f"Low efficiency: {efficiency:.1f}%",
                "priority": "medium"
            })

        return {
            "combustion_efficiency": efficiency,
            "excess_air_percent": excess_air,
            "stack_losses": {
                "dry_gas_loss_percent": (flue_gas_temp - 25) * 0.03,
                "moisture_loss_percent": 4.5,
                "radiation_loss_percent": 1.5
            },
            "actions_required": actions_required,
            "optimization_potential_percent": max(0, self._config["target_efficiency_percent"] - efficiency)
        }

    async def analyze_emissions(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze emissions and determine reduction strategies."""
        nox_ppm = parameters.get("nox_ppm", 22.0)
        co_ppm = parameters.get("co_ppm", 15.0)
        o2_percent = parameters.get("o2_percent", 4.5)

        # Determine emission status
        nox_compliant = nox_ppm <= self._config["max_nox_ppm"]
        co_compliant = co_ppm <= self._config["max_co_ppm"]

        actions_required = []

        if not nox_compliant:
            actions_required.append({
                "action": "reduce_nox",
                "reason": f"NOx exceeds limit: {nox_ppm:.0f} ppm",
                "strategy": "reduce_flame_temperature"
            })

        if not co_compliant:
            actions_required.append({
                "action": "reduce_co",
                "reason": f"CO exceeds limit: {co_ppm:.0f} ppm",
                "strategy": "improve_combustion"
            })

        return {
            "nox_ppm": nox_ppm,
            "co_ppm": co_ppm,
            "o2_percent": o2_percent,
            "nox_compliant": nox_compliant,
            "co_compliant": co_compliant,
            "overall_compliant": nox_compliant and co_compliant,
            "actions_required": actions_required
        }

    async def execute_fuel_valve_control(
        self,
        boiler_id: str,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fuel valve control action."""
        if self._fault_injection == "fuel_valve_failure":
            return {"status": "error", "error": "Fuel valve control failed"}

        valve_change = action.get("valve_position_change_percent", 2.0)

        # Update boiler state
        if boiler_id in self._boilers:
            boiler = self._boilers[boiler_id]
            boiler.fuel_flow_kg_hr *= (1 + valve_change / 100)

        return {
            "status": "success",
            "boiler_id": boiler_id,
            "action_executed": "fuel_valve_adjust",
            "valve_position_change_percent": valve_change,
            "response_time_ms": random.uniform(50, 150)
        }

    async def execute_air_damper_control(
        self,
        boiler_id: str,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute air damper control action."""
        if self._fault_injection == "damper_failure":
            return {"status": "error", "error": "Air damper control failed"}

        damper_change = action.get("damper_position_change_percent", 3.0)

        # Update boiler state
        if boiler_id in self._boilers:
            boiler = self._boilers[boiler_id]
            boiler.excess_air_percent += damper_change
            boiler.o2_percent = boiler.excess_air_percent * 0.21 / (1 + boiler.excess_air_percent / 100)

        return {
            "status": "success",
            "boiler_id": boiler_id,
            "action_executed": "air_damper_adjust",
            "damper_position_change_percent": damper_change,
            "response_time_ms": random.uniform(100, 250)
        }

    async def execute_optimization_cycle(
        self,
        boiler_id: str,
        scenario: Optional[BoilerTestScenario] = None
    ) -> Dict[str, Any]:
        """Execute single optimization cycle."""
        start_time = time.time()
        self._cycle_count += 1

        # Apply scenario if specified
        if scenario:
            self._apply_scenario(boiler_id, scenario)

        # 1. Read parameters
        params = await self.read_boiler_parameters(boiler_id)
        if params.get("status") == "error":
            return {"status": "error", "phase": "read", "error": params.get("error")}

        efficiency_before = params.get("efficiency_percent", 85.0)
        emissions_before = {
            "nox_ppm": params.get("nox_ppm", 22.0),
            "co_ppm": params.get("co_ppm", 15.0)
        }

        # 2. Analyze combustion
        combustion = await self.analyze_combustion_efficiency(params)

        # 3. Analyze emissions
        emissions = await self.analyze_emissions(params)

        # 4. Execute control actions
        actions_executed = []
        all_actions = combustion.get("actions_required", []) + emissions.get("actions_required", [])

        for action in all_actions:
            if "air" in action.get("action", ""):
                result = await self.execute_air_damper_control(boiler_id, action)
                actions_executed.append(result)
            elif "fuel" in action.get("action", "") or "valve" in action.get("action", ""):
                result = await self.execute_fuel_valve_control(boiler_id, action)
                actions_executed.append(result)

        # 5. Re-read parameters after actions
        params_after = await self.read_boiler_parameters(boiler_id)
        efficiency_after = params_after.get("efficiency_percent", 85.0)
        emissions_after = {
            "nox_ppm": params_after.get("nox_ppm", 22.0),
            "co_ppm": params_after.get("co_ppm", 15.0)
        }

        # Simulate efficiency improvement from optimization
        if actions_executed:
            if boiler_id in self._boilers:
                self._boilers[boiler_id].efficiency_percent = min(
                    95.0,
                    self._boilers[boiler_id].efficiency_percent + random.uniform(0.1, 0.5)
                )

        execution_time_ms = (time.time() - start_time) * 1000

        # Calculate provenance hash
        provenance_data = {
            "cycle": self._cycle_count,
            "params": params,
            "combustion": combustion,
            "emissions": emissions,
            "actions": actions_executed,
            "efficiency_before": efficiency_before,
            "efficiency_after": efficiency_after
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        self._provenance_chain.append(provenance_hash)

        # Update metrics
        self._update_metrics(
            execution_time_ms,
            len(actions_executed),
            efficiency_after > efficiency_before,
            emissions_after["nox_ppm"] < emissions_before["nox_ppm"]
        )

        return {
            "status": "success",
            "cycle_number": self._cycle_count,
            "boiler_id": boiler_id,
            "parameters": params,
            "combustion_analysis": combustion,
            "emissions_analysis": emissions,
            "actions_executed": actions_executed,
            "efficiency_before": efficiency_before,
            "efficiency_after": efficiency_after,
            "emissions_before": emissions_before,
            "emissions_after": emissions_after,
            "improvement": efficiency_after - efficiency_before,
            "execution_time_ms": execution_time_ms,
            "provenance_hash": provenance_hash,
            "determinism_verified": True
        }

    async def coordinate_multi_boiler(
        self,
        boiler_ids: List[str],
        total_load_target: float
    ) -> Dict[str, Any]:
        """Coordinate multiple boilers for optimal load distribution."""
        load_distribution = {}
        total_capacity = 0.0

        # Calculate total capacity
        for boiler_id in boiler_ids:
            if boiler_id not in self._boilers:
                self._boilers[boiler_id] = BoilerState(boiler_id=boiler_id)
            total_capacity += 100.0  # Max capacity per boiler

        # Distribute load based on efficiency
        for boiler_id in boiler_ids:
            boiler = self._boilers[boiler_id]
            # Higher efficiency boilers get more load
            efficiency_weight = boiler.efficiency_percent / 100
            load_share = (total_load_target / len(boiler_ids)) * efficiency_weight
            load_distribution[boiler_id] = min(100.0, load_share)

        return {
            "status": "success",
            "total_load_target": total_load_target,
            "load_distribution": load_distribution,
            "coordination_strategy": "efficiency_weighted",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def execute_emergency_shutdown(
        self,
        boiler_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """Execute emergency shutdown sequence."""
        if boiler_id in self._boilers:
            self._boilers[boiler_id].is_running = False

        return {
            "status": "shutdown_complete",
            "boiler_id": boiler_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions_taken": [
                {"action": "fuel_valve_close", "status": "success"},
                {"action": "air_damper_close", "status": "success"},
                {"action": "feedwater_stop", "status": "success"},
                {"action": "activate_alarm", "status": "success"},
                {"action": "notify_operators", "status": "success"}
            ],
            "response_time_ms": random.uniform(200, 500)
        }

    def _apply_scenario(self, boiler_id: str, scenario: BoilerTestScenario):
        """Apply test scenario to boiler."""
        if boiler_id not in self._boilers:
            self._boilers[boiler_id] = BoilerState(boiler_id=boiler_id)

        boiler = self._boilers[boiler_id]

        if scenario == BoilerTestScenario.HIGH_LOAD:
            boiler.load_percent = 95.0
            boiler.steam_flow_kg_hr = 45000.0
        elif scenario == BoilerTestScenario.LOW_EFFICIENCY:
            boiler.efficiency_percent = 78.0
            boiler.flue_gas_temp_c = 220.0
        elif scenario == BoilerTestScenario.HIGH_EMISSIONS:
            boiler.nox_ppm = 45.0
            boiler.co_ppm = 120.0
        elif scenario == BoilerTestScenario.FUEL_SWITCH:
            boiler.fuel_type = "fuel_oil"
        elif scenario == BoilerTestScenario.EMERGENCY_SHUTDOWN:
            boiler.steam_pressure_bar = 50.0  # Over pressure

    def _update_metrics(
        self,
        execution_time_ms: float,
        actions_count: int,
        efficiency_improved: bool,
        emissions_reduced: bool
    ):
        """Update E2E test metrics."""
        self._metrics.total_cycles += 1
        self._metrics.successful_cycles += 1
        self._metrics.total_execution_time_ms += execution_time_ms
        self._metrics.min_cycle_time_ms = min(self._metrics.min_cycle_time_ms, execution_time_ms)
        self._metrics.max_cycle_time_ms = max(self._metrics.max_cycle_time_ms, execution_time_ms)
        self._metrics.avg_cycle_time_ms = self._metrics.total_execution_time_ms / self._metrics.total_cycles
        self._metrics.control_actions_taken += actions_count
        if efficiency_improved:
            self._metrics.efficiency_improvements += 1
        if emissions_reduced:
            self._metrics.emission_reductions += 1

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_cycles": self._metrics.total_cycles,
            "successful_cycles": self._metrics.successful_cycles,
            "failed_cycles": self._metrics.failed_cycles,
            "avg_cycle_time_ms": self._metrics.avg_cycle_time_ms,
            "min_cycle_time_ms": self._metrics.min_cycle_time_ms,
            "max_cycle_time_ms": self._metrics.max_cycle_time_ms,
            "efficiency_improvements": self._metrics.efficiency_improvements,
            "emission_reductions": self._metrics.emission_reductions,
            "control_actions_taken": self._metrics.control_actions_taken
        }

    def inject_fault(self, fault_type: str):
        """Inject fault for testing."""
        self._fault_injection = fault_type

    def clear_fault(self):
        """Clear injected fault."""
        self._fault_injection = None

    def get_provenance_chain(self) -> List[str]:
        """Get provenance hash chain."""
        return self._provenance_chain.copy()

    def get_metrics(self) -> E2ETestMetrics:
        """Get current metrics."""
        return self._metrics


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
async def e2e_orchestrator():
    """Create E2E orchestrator for testing."""
    orchestrator = MockE2EBoilerEfficiencyOrchestrator()
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()


@pytest.fixture
def default_config():
    """Default orchestrator configuration."""
    return {
        "agent_id": "GL-002",
        "agent_name": "BoilerEfficiencyOptimizer",
        "target_efficiency_percent": 92.0,
        "max_excess_air_percent": 25.0,
        "min_excess_air_percent": 5.0,
        "max_o2_percent": 6.0,
        "min_o2_percent": 2.0,
        "max_co_ppm": 100.0,
        "max_nox_ppm": 30.0,
        "optimization_interval_seconds": 60,
        "safety_margin_percent": 5.0,
    }


@pytest.fixture
def scada_config():
    """SCADA system configuration."""
    return {
        "protocol": "opc_ua",
        "endpoint": "opc.tcp://localhost:4840",
        "security_mode": "SignAndEncrypt",
        "polling_interval_ms": 1000
    }


@pytest.fixture
def dcs_config():
    """DCS system configuration."""
    return {
        "protocol": "modbus",
        "host": "localhost",
        "port": 502,
        "slave_id": 1
    }


@pytest.fixture
def fuel_config():
    """Fuel management configuration."""
    return {
        "fuel_types": ["natural_gas", "fuel_oil"],
        "multi_fuel_enabled": True,
        "auto_switching_enabled": True
    }


@pytest.fixture
def cems_config():
    """CEMS emissions monitoring configuration."""
    return {
        "analyzers": ["o2", "co", "co2", "nox", "so2"],
        "compliance_standard": "EPA",
        "sampling_interval_seconds": 5
    }


@pytest.fixture
def multi_boiler_config():
    """Multi-boiler system configuration."""
    return {
        "boilers": [
            {"boiler_id": "BOILER-001", "capacity_kg_hr": 50000, "fuel_type": "natural_gas"},
            {"boiler_id": "BOILER-002", "capacity_kg_hr": 30000, "fuel_type": "natural_gas"},
            {"boiler_id": "BOILER-003", "capacity_kg_hr": 20000, "fuel_type": "fuel_oil"}
        ],
        "load_balancing": True,
        "efficiency_optimization": True
    }


@pytest.fixture
def test_scenarios():
    """Test scenario configurations."""
    return {
        BoilerTestScenario.NORMAL_OPERATION: {
            "load_percent": 75.0,
            "efficiency_percent": 85.0,
            "expected_actions": 0
        },
        BoilerTestScenario.HIGH_LOAD: {
            "load_percent": 95.0,
            "efficiency_percent": 83.0,
            "expected_actions": 1
        },
        BoilerTestScenario.LOW_EFFICIENCY: {
            "load_percent": 70.0,
            "efficiency_percent": 78.0,
            "expected_actions": 2
        },
        BoilerTestScenario.HIGH_EMISSIONS: {
            "load_percent": 80.0,
            "nox_ppm": 45.0,
            "expected_actions": 2
        }
    }


# =============================================================================
# E2E ASSERTION HELPERS
# =============================================================================

class E2EAssertions:
    """Custom assertions for E2E tests."""

    @staticmethod
    def assert_workflow_completed(result: Dict[str, Any]):
        """Assert workflow completed successfully."""
        assert result.get("status") == "success", f"Workflow failed: {result.get('error')}"

    @staticmethod
    def assert_efficiency_in_range(efficiency: float, min_val: float = 70.0, max_val: float = 100.0):
        """Assert efficiency is within range."""
        assert min_val <= efficiency <= max_val, f"Efficiency {efficiency}% not in range [{min_val}, {max_val}]"

    @staticmethod
    def assert_emissions_compliant(nox_ppm: float, co_ppm: float, limits: Dict[str, float] = None):
        """Assert emissions are within compliance limits."""
        limits = limits or {"nox": 30.0, "co": 100.0}
        assert nox_ppm <= limits["nox"], f"NOx {nox_ppm} ppm exceeds limit {limits['nox']} ppm"
        assert co_ppm <= limits["co"], f"CO {co_ppm} ppm exceeds limit {limits['co']} ppm"

    @staticmethod
    def assert_provenance_valid(provenance_hash: str):
        """Assert provenance hash is valid."""
        assert provenance_hash is not None, "Provenance hash is None"
        assert len(provenance_hash) == 64, f"Invalid provenance hash length: {len(provenance_hash)}"

    @staticmethod
    def assert_control_actions_valid(actions: List[Dict[str, Any]]):
        """Assert control actions are valid."""
        for action in actions:
            assert action.get("status") == "success", f"Action failed: {action}"

    @staticmethod
    def assert_cycle_time_acceptable(cycle_time_ms: float, target_ms: float = 3000.0):
        """Assert cycle time meets target."""
        assert cycle_time_ms <= target_ms, f"Cycle time {cycle_time_ms}ms exceeds target {target_ms}ms"


@pytest.fixture
def e2e_assertions():
    """E2E assertions fixture."""
    return E2EAssertions()
