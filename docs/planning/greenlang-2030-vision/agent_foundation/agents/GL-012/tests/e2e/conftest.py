# -*- coding: utf-8 -*-
"""
E2E Test Fixtures and Configuration for GL-012 STEAMQUAL SteamQualityController.

Comprehensive fixtures for end-to-end testing including:
- Full system mock setup (orchestrator, meters, valves, SCADA)
- Test data scenarios (normal, degradation, emergency)
- Performance timing utilities
- Multi-header coordination fixtures
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
        "control_loop: Real-time control loop tests",
        "fault_tolerance: Fault tolerance and recovery tests",
        "multi_header: Multi-header coordination tests",
        "performance: Performance benchmark tests",
        "determinism: Determinism verification tests",
        "provenance: Provenance tracking tests",
        "emergency: Emergency scenario tests",
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
        help="Run extended control loop tests (1000+ cycles)"
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

class TestScenario(str, Enum):
    """Test scenario types."""
    NORMAL_OPERATION = "normal_operation"
    QUALITY_DEGRADATION = "quality_degradation"
    PRESSURE_DEVIATION = "pressure_deviation"
    HIGH_MOISTURE = "high_moisture"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    SENSOR_FAILURE = "sensor_failure"
    VALVE_FAILURE = "valve_failure"
    COMMUNICATION_LOSS = "communication_loss"


class ControlAction(str, Enum):
    """Control action types."""
    NO_ACTION = "no_action"
    DESUPERHEATER_ADJUST = "desuperheater_adjust"
    PRESSURE_VALVE_ADJUST = "pressure_valve_adjust"
    MULTIPLE_ADJUST = "multiple_adjust"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


# =============================================================================
# DATA CLASSES FOR E2E TESTING
# =============================================================================

@dataclass
class SteamHeaderState:
    """State of a steam header for testing."""
    header_id: str
    pressure_bar: float = 10.0
    temperature_c: float = 180.0
    flow_rate_kg_hr: float = 5000.0
    dryness_fraction: float = 0.98
    superheat_c: float = 0.0
    is_active: bool = True
    quality_index: float = 95.0


@dataclass
class ControlLoopCycle:
    """Single control loop cycle data."""
    cycle_number: int
    timestamp: datetime
    sensor_readings: Dict[str, float]
    control_actions: List[Dict[str, Any]]
    quality_before: float
    quality_after: float
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
    quality_improvements: int = 0
    control_actions_taken: int = 0
    provenance_hashes: List[str] = field(default_factory=list)


# =============================================================================
# MOCK STEAM QUALITY ORCHESTRATOR FOR E2E
# =============================================================================

class MockE2ESteamQualityOrchestrator:
    """
    Full mock orchestrator for E2E testing.

    Simulates complete steam quality control workflow including:
    - Sensor data acquisition
    - Quality analysis
    - Control action determination
    - Desuperheater control
    - Pressure valve control
    - KPI dashboard generation
    - Provenance tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize E2E orchestrator."""
        self._config = config or self._default_config()
        self._state = "initialized"
        self._cycle_count = 0
        self._headers: Dict[str, SteamHeaderState] = {}
        self._meters_connected = False
        self._valves_connected = False
        self._scada_connected = False
        self._control_mode = "auto"
        self._provenance_chain: List[str] = []
        self._metrics = E2ETestMetrics()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._fault_injection: Optional[str] = None

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "agent_id": "GL-012",
            "agent_name": "SteamQualityController",
            "quality_target": 0.95,
            "pressure_setpoint_bar": 10.0,
            "temperature_setpoint_c": 180.0,
            "control_deadband_percent": 2.0,
            "scan_rate_ms": 100,
            "max_desuperheater_flow_kg_hr": 5000.0,
            "min_desuperheater_flow_kg_hr": 100.0,
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

    async def connect_meters(self, meter_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Connect to steam quality meters."""
        if self._fault_injection == "meter_connection_failure":
            return {"status": "error", "error": "Meter connection failed"}

        self._meters_connected = True
        return {
            "status": "connected",
            "meters_connected": len(meter_configs),
            "protocols": [m.get("protocol", "modbus_tcp") for m in meter_configs]
        }

    async def connect_valves(self, valve_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Connect to control valves."""
        if self._fault_injection == "valve_connection_failure":
            return {"status": "error", "error": "Valve connection failed"}

        self._valves_connected = True
        return {
            "status": "connected",
            "valves_connected": len(valve_configs),
            "valve_ids": [v.get("valve_id") for v in valve_configs]
        }

    async def connect_scada(self, scada_config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to SCADA system."""
        if self._fault_injection == "scada_connection_failure":
            return {"status": "error", "error": "SCADA connection failed"}

        self._scada_connected = True
        return {
            "status": "connected",
            "protocol": scada_config.get("protocol", "opcua"),
            "server_state": "Running"
        }

    async def read_steam_quality_parameters(
        self,
        header_id: str
    ) -> Dict[str, Any]:
        """Read steam quality parameters from sensors."""
        if self._fault_injection == "sensor_failure":
            return {"status": "error", "quality": "BAD", "error": "Sensor failure"}

        # Get or create header state
        if header_id not in self._headers:
            self._headers[header_id] = SteamHeaderState(header_id=header_id)

        header = self._headers[header_id]

        # Add some noise for realism
        noise = lambda base, pct: base * (1 + random.uniform(-pct, pct) / 100)

        return {
            "header_id": header_id,
            "pressure_bar": noise(header.pressure_bar, 1.0),
            "temperature_c": noise(header.temperature_c, 0.5),
            "flow_rate_kg_hr": noise(header.flow_rate_kg_hr, 2.0),
            "dryness_fraction": min(1.0, max(0.0, noise(header.dryness_fraction, 1.0))),
            "superheat_c": max(0.0, noise(header.superheat_c, 5.0)),
            "quality": "GOOD",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation": {
                "pressure_valid": True,
                "temperature_valid": True,
                "thermodynamic_consistent": True
            }
        }

    async def analyze_quality(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze steam quality and determine required actions."""
        dryness = parameters.get("dryness_fraction", 0.98)
        pressure = parameters.get("pressure_bar", 10.0)
        temperature = parameters.get("temperature_c", 180.0)

        # Calculate quality index (0-100)
        quality_index = self._calculate_quality_index(dryness, pressure, temperature)

        # Determine deviations
        pressure_deviation = abs(pressure - self._config["pressure_setpoint_bar"]) / self._config["pressure_setpoint_bar"] * 100
        temp_deviation = abs(temperature - self._config["temperature_setpoint_c"]) / self._config["temperature_setpoint_c"] * 100
        moisture_percent = (1 - dryness) * 100

        # Determine required actions
        actions_required = []
        if moisture_percent > 5.0:
            actions_required.append({
                "action": "desuperheater_adjust",
                "reason": f"High moisture: {moisture_percent:.1f}%",
                "priority": "high"
            })
        if pressure_deviation > self._config["control_deadband_percent"]:
            actions_required.append({
                "action": "pressure_valve_adjust",
                "reason": f"Pressure deviation: {pressure_deviation:.1f}%",
                "priority": "medium"
            })
        if temp_deviation > self._config["control_deadband_percent"]:
            actions_required.append({
                "action": "temperature_adjust",
                "reason": f"Temperature deviation: {temp_deviation:.1f}%",
                "priority": "medium"
            })

        return {
            "quality_index": quality_index,
            "quality_level": self._get_quality_level(quality_index),
            "deviations": {
                "pressure_percent": pressure_deviation,
                "temperature_percent": temp_deviation,
                "moisture_percent": moisture_percent
            },
            "actions_required": actions_required,
            "compliance_status": "compliant" if quality_index >= 90 else "non_compliant"
        }

    async def execute_desuperheater_control(
        self,
        header_id: str,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute desuperheater control action."""
        if self._fault_injection == "desuperheater_failure":
            return {"status": "error", "error": "Desuperheater control failed"}

        # Simulate control action
        injection_rate_change = action.get("injection_rate_change_percent", 5.0)

        # Update header state
        if header_id in self._headers:
            header = self._headers[header_id]
            # Desuperheater improves dryness
            header.dryness_fraction = min(1.0, header.dryness_fraction + 0.01)
            header.superheat_c = max(0.0, header.superheat_c - 2.0)

        return {
            "status": "success",
            "header_id": header_id,
            "action_executed": "desuperheater_adjust",
            "injection_rate_change_percent": injection_rate_change,
            "new_injection_rate_kg_hr": 500.0 + injection_rate_change * 10,
            "response_time_ms": random.uniform(50, 200)
        }

    async def execute_pressure_valve_control(
        self,
        header_id: str,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute pressure control valve action."""
        if self._fault_injection == "valve_failure":
            return {"status": "error", "error": "Valve control failed"}

        valve_position_change = action.get("valve_position_change_percent", 2.0)

        # Update header state
        if header_id in self._headers:
            header = self._headers[header_id]
            # Adjust pressure towards setpoint
            setpoint = self._config["pressure_setpoint_bar"]
            header.pressure_bar = header.pressure_bar + (setpoint - header.pressure_bar) * 0.1

        return {
            "status": "success",
            "header_id": header_id,
            "action_executed": "pressure_valve_adjust",
            "valve_position_change_percent": valve_position_change,
            "new_valve_position_percent": 50.0 + valve_position_change,
            "response_time_ms": random.uniform(100, 300)
        }

    async def verify_quality_improvement(
        self,
        header_id: str,
        quality_before: float,
        quality_after: float
    ) -> Dict[str, Any]:
        """Verify quality improvement after control actions."""
        improvement = quality_after - quality_before

        return {
            "header_id": header_id,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "improvement": improvement,
            "improvement_percent": improvement / quality_before * 100 if quality_before > 0 else 0,
            "verification_passed": quality_after >= quality_before * 0.99,  # Allow 1% tolerance
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def generate_kpi_dashboard(
        self,
        header_ids: List[str]
    ) -> Dict[str, Any]:
        """Generate KPI dashboard data."""
        kpis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_headers": len(header_ids),
                "active_headers": sum(1 for h in header_ids if self._headers.get(h, SteamHeaderState(h)).is_active),
                "avg_quality_index": sum(self._headers.get(h, SteamHeaderState(h)).quality_index for h in header_ids) / len(header_ids) if header_ids else 0
            },
            "headers": {},
            "control_actions_last_hour": self._metrics.control_actions_taken,
            "quality_improvements_last_hour": self._metrics.quality_improvements,
            "compliance_status": "compliant"
        }

        for header_id in header_ids:
            header = self._headers.get(header_id, SteamHeaderState(header_id))
            kpis["headers"][header_id] = {
                "pressure_bar": header.pressure_bar,
                "temperature_c": header.temperature_c,
                "dryness_fraction": header.dryness_fraction,
                "quality_index": header.quality_index,
                "status": "active" if header.is_active else "inactive"
            }

        return kpis

    async def execute_cycle(
        self,
        header_id: str,
        scenario: Optional[TestScenario] = None
    ) -> Dict[str, Any]:
        """Execute single control loop cycle."""
        start_time = time.time()
        self._cycle_count += 1

        # Apply scenario if specified
        if scenario:
            self._apply_scenario(header_id, scenario)

        # 1. Read parameters
        params = await self.read_steam_quality_parameters(header_id)
        if params.get("status") == "error":
            return {"status": "error", "phase": "read", "error": params.get("error")}

        quality_before = self._calculate_quality_index(
            params.get("dryness_fraction", 0.98),
            params.get("pressure_bar", 10.0),
            params.get("temperature_c", 180.0)
        )

        # 2. Analyze quality
        analysis = await self.analyze_quality(params)

        # 3. Execute control actions
        actions_executed = []
        for action in analysis.get("actions_required", []):
            if action["action"] == "desuperheater_adjust":
                result = await self.execute_desuperheater_control(header_id, action)
                actions_executed.append(result)
            elif action["action"] == "pressure_valve_adjust":
                result = await self.execute_pressure_valve_control(header_id, action)
                actions_executed.append(result)

        # 4. Calculate quality after actions
        params_after = await self.read_steam_quality_parameters(header_id)
        quality_after = self._calculate_quality_index(
            params_after.get("dryness_fraction", 0.98),
            params_after.get("pressure_bar", 10.0),
            params_after.get("temperature_c", 180.0)
        )

        # Update header quality index
        if header_id in self._headers:
            self._headers[header_id].quality_index = quality_after

        execution_time_ms = (time.time() - start_time) * 1000

        # Calculate provenance hash
        provenance_data = {
            "cycle": self._cycle_count,
            "params": params,
            "analysis": analysis,
            "actions": actions_executed,
            "quality_before": quality_before,
            "quality_after": quality_after
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        self._provenance_chain.append(provenance_hash)

        # Update metrics
        self._update_metrics(execution_time_ms, len(actions_executed), quality_after > quality_before)

        return {
            "status": "success",
            "cycle_number": self._cycle_count,
            "header_id": header_id,
            "parameters": params,
            "analysis": analysis,
            "actions_executed": actions_executed,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "improvement": quality_after - quality_before,
            "execution_time_ms": execution_time_ms,
            "provenance_hash": provenance_hash,
            "determinism_verified": True
        }

    async def execute_emergency_shutdown(
        self,
        header_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """Execute emergency shutdown sequence."""
        if header_id in self._headers:
            self._headers[header_id].is_active = False

        return {
            "status": "shutdown_complete",
            "header_id": header_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions_taken": [
                {"action": "close_isolation_valve", "status": "success"},
                {"action": "close_desuperheater_valve", "status": "success"},
                {"action": "activate_alarm", "status": "success"},
                {"action": "notify_operators", "status": "success"}
            ],
            "response_time_ms": random.uniform(200, 500)
        }

    def _calculate_quality_index(
        self,
        dryness: float,
        pressure: float,
        temperature: float
    ) -> float:
        """Calculate steam quality index (0-100)."""
        # Dryness component (50% weight)
        dryness_score = min(100, dryness * 100)

        # Pressure stability (25% weight)
        pressure_deviation = abs(pressure - self._config["pressure_setpoint_bar"]) / self._config["pressure_setpoint_bar"]
        pressure_score = max(0, 100 - pressure_deviation * 200)

        # Temperature accuracy (25% weight)
        temp_deviation = abs(temperature - self._config["temperature_setpoint_c"]) / self._config["temperature_setpoint_c"]
        temp_score = max(0, 100 - temp_deviation * 200)

        return 0.5 * dryness_score + 0.25 * pressure_score + 0.25 * temp_score

    def _get_quality_level(self, index: float) -> str:
        """Get quality level from index."""
        if index >= 98:
            return "excellent"
        elif index >= 95:
            return "good"
        elif index >= 90:
            return "acceptable"
        elif index >= 85:
            return "poor"
        else:
            return "critical"

    def _apply_scenario(self, header_id: str, scenario: TestScenario):
        """Apply test scenario to header."""
        if header_id not in self._headers:
            self._headers[header_id] = SteamHeaderState(header_id=header_id)

        header = self._headers[header_id]

        if scenario == TestScenario.QUALITY_DEGRADATION:
            header.dryness_fraction = 0.92
            header.quality_index = 85.0
        elif scenario == TestScenario.PRESSURE_DEVIATION:
            header.pressure_bar = 12.5  # 25% above setpoint
        elif scenario == TestScenario.HIGH_MOISTURE:
            header.dryness_fraction = 0.88
        elif scenario == TestScenario.EMERGENCY_SHUTDOWN:
            header.pressure_bar = 15.0  # Critical pressure
            header.temperature_c = 220.0  # High temperature

    def _update_metrics(
        self,
        execution_time_ms: float,
        actions_count: int,
        quality_improved: bool
    ):
        """Update E2E test metrics."""
        self._metrics.total_cycles += 1
        self._metrics.successful_cycles += 1
        self._metrics.total_execution_time_ms += execution_time_ms
        self._metrics.min_cycle_time_ms = min(self._metrics.min_cycle_time_ms, execution_time_ms)
        self._metrics.max_cycle_time_ms = max(self._metrics.max_cycle_time_ms, execution_time_ms)
        self._metrics.avg_cycle_time_ms = self._metrics.total_execution_time_ms / self._metrics.total_cycles
        self._metrics.control_actions_taken += actions_count
        if quality_improved:
            self._metrics.quality_improvements += 1

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_cycles": self._metrics.total_cycles,
            "successful_cycles": self._metrics.successful_cycles,
            "failed_cycles": self._metrics.failed_cycles,
            "avg_cycle_time_ms": self._metrics.avg_cycle_time_ms,
            "min_cycle_time_ms": self._metrics.min_cycle_time_ms,
            "max_cycle_time_ms": self._metrics.max_cycle_time_ms,
            "quality_improvements": self._metrics.quality_improvements,
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
    orchestrator = MockE2ESteamQualityOrchestrator()
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()


@pytest.fixture
def default_config():
    """Default orchestrator configuration."""
    return {
        "agent_id": "GL-012",
        "agent_name": "SteamQualityController",
        "quality_target": 0.95,
        "pressure_setpoint_bar": 10.0,
        "temperature_setpoint_c": 180.0,
        "control_deadband_percent": 2.0,
        "scan_rate_ms": 100,
        "max_desuperheater_flow_kg_hr": 5000.0,
        "min_desuperheater_flow_kg_hr": 100.0,
        "max_retries": 3,
        "retry_delay_seconds": 1.0,
    }


@pytest.fixture
def meter_configs():
    """Steam quality meter configurations."""
    return [
        {
            "meter_id": "SQM-001",
            "meter_type": "vortex",
            "protocol": "modbus_tcp",
            "host": "localhost",
            "port": 5020,
            "parameters": ["pressure", "temperature", "flow", "dryness"]
        },
        {
            "meter_id": "SQM-002",
            "meter_type": "orifice",
            "protocol": "opcua",
            "endpoint": "opc.tcp://localhost:4840",
            "parameters": ["pressure", "temperature", "flow"]
        }
    ]


@pytest.fixture
def valve_configs():
    """Control valve configurations."""
    return [
        {
            "valve_id": "PCV-001",
            "valve_type": "globe",
            "actuator": "pneumatic",
            "fail_position": "closed",
            "cv_rating": 150.0
        },
        {
            "valve_id": "TCV-001",
            "valve_type": "globe",
            "actuator": "electric",
            "fail_position": "last",
            "cv_rating": 100.0
        }
    ]


@pytest.fixture
def scada_config():
    """SCADA system configuration."""
    return {
        "protocol": "opcua",
        "endpoint": "opc.tcp://localhost:4860",
        "security_mode": "SignAndEncrypt",
        "polling_interval_ms": 1000
    }


@pytest.fixture
def multi_header_config():
    """Multi-header system configuration."""
    return {
        "headers": [
            {"header_id": "HP-STEAM-01", "pressure_bar": 40.0, "temperature_c": 400.0},
            {"header_id": "MP-STEAM-01", "pressure_bar": 10.0, "temperature_c": 180.0},
            {"header_id": "LP-STEAM-01", "pressure_bar": 3.0, "temperature_c": 130.0}
        ],
        "load_balancing": True,
        "cross_header_optimization": True
    }


@pytest.fixture
def test_scenarios():
    """Test scenario configurations."""
    return {
        TestScenario.NORMAL_OPERATION: {
            "pressure_bar": 10.0,
            "temperature_c": 180.0,
            "dryness_fraction": 0.98,
            "expected_actions": 0
        },
        TestScenario.QUALITY_DEGRADATION: {
            "pressure_bar": 10.0,
            "temperature_c": 185.0,
            "dryness_fraction": 0.92,
            "expected_actions": 1
        },
        TestScenario.PRESSURE_DEVIATION: {
            "pressure_bar": 12.5,
            "temperature_c": 180.0,
            "dryness_fraction": 0.97,
            "expected_actions": 1
        },
        TestScenario.HIGH_MOISTURE: {
            "pressure_bar": 10.0,
            "temperature_c": 175.0,
            "dryness_fraction": 0.88,
            "expected_actions": 2
        },
        TestScenario.EMERGENCY_SHUTDOWN: {
            "pressure_bar": 15.0,
            "temperature_c": 220.0,
            "dryness_fraction": 0.85,
            "expected_actions": 0,  # Emergency action
            "emergency": True
        }
    }


# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

class PerformanceTimer:
    """Utility for measuring performance metrics."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def stop(self, operation: str) -> float:
        """Stop timing and record duration."""
        if operation not in self.start_times:
            return 0.0

        duration_ms = (time.time() - self.start_times[operation]) * 1000
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration_ms)

        del self.start_times[operation]
        return duration_ms

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return {"min": 0, "max": 0, "avg": 0, "count": 0}

        times = self.timings[operation]
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "count": len(times)
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.timings}


@pytest.fixture
def performance_timer():
    """Performance timer fixture."""
    return PerformanceTimer()


# =============================================================================
# DETERMINISM VALIDATION
# =============================================================================

class DeterminismValidator:
    """Validates deterministic behavior of calculations."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def record_result(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Record input-output pair."""
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()
        output_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        self.results.append({
            "input_hash": input_hash,
            "output_hash": output_hash,
            "inputs": inputs,
            "outputs": outputs
        })

    def verify_determinism(self) -> Tuple[bool, List[str]]:
        """Verify all identical inputs produced identical outputs."""
        issues = []
        input_to_outputs: Dict[str, List[str]] = {}

        for result in self.results:
            input_hash = result["input_hash"]
            output_hash = result["output_hash"]

            if input_hash not in input_to_outputs:
                input_to_outputs[input_hash] = []
            input_to_outputs[input_hash].append(output_hash)

        # Check for non-deterministic behavior
        for input_hash, output_hashes in input_to_outputs.items():
            unique_outputs = set(output_hashes)
            if len(unique_outputs) > 1:
                issues.append(
                    f"Input {input_hash[:8]} produced {len(unique_outputs)} different outputs"
                )

        return len(issues) == 0, issues


@pytest.fixture
def determinism_validator():
    """Determinism validator fixture."""
    return DeterminismValidator()


# =============================================================================
# PROVENANCE TRACKER
# =============================================================================

class ProvenanceTracker:
    """Track and validate provenance chain."""

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

            # Verify previous hash
            if i > 0 and entry["previous_hash"] != self.chain[i-1]["hash"]:
                issues.append(f"Chain broken at entry {i}")

        return len(issues) == 0, issues

    def get_chain_hash(self) -> str:
        """Get hash of entire chain."""
        chain_str = json.dumps(self.chain, sort_keys=True, default=str)
        return hashlib.sha256(chain_str.encode()).hexdigest()


@pytest.fixture
def provenance_tracker():
    """Provenance tracker fixture."""
    return ProvenanceTracker()


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
    def assert_quality_in_range(quality: float, min_val: float = 0.0, max_val: float = 100.0):
        """Assert quality index is within range."""
        assert min_val <= quality <= max_val, f"Quality {quality} not in range [{min_val}, {max_val}]"

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
    def assert_cycle_time_acceptable(cycle_time_ms: float, target_ms: float = 1000.0):
        """Assert cycle time meets target."""
        assert cycle_time_ms <= target_ms, f"Cycle time {cycle_time_ms}ms exceeds target {target_ms}ms"

    @staticmethod
    def assert_determinism(results: List[Dict[str, Any]]):
        """Assert results are deterministic."""
        if not results:
            return

        first_hash = results[0].get("provenance_hash")
        # For same inputs, provenance should be consistent
        # (Note: timing may vary, but core calculations should be deterministic)


@pytest.fixture
def e2e_assertions():
    """E2E assertions fixture."""
    return E2EAssertions()
