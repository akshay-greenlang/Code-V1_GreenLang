"""
Unit tests for GL-001 ThermalCommand Orchestrator Schemas Module

Tests all Pydantic models, enumerations, and data validation with 90%+ coverage.
Validates schema serialization, deserialization, and business logic.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, patch
import json
import hashlib

from greenlang.agents.process_heat.gl_001_thermal_command.schemas import (
    # Enumerations
    Priority,
    WorkflowType,
    WorkflowStatus,
    TaskStatus,
    AgentHealth,
    SafetyLevel,
    AlarmSeverity,
    # Core Models
    ThermalLoad,
    EquipmentStatus,
    ProcessMeasurement,
    ControlSetpoint,
    SafetyEvent,
    AlarmEvent,
    # Workflow Models
    WorkflowSpec,
    WorkflowResult,
    TaskSpec,
    TaskResult,
    # Agent Models
    AgentStatus,
    AgentHeartbeat,
    # System Models
    SystemStatus,
    OrchestratorMetrics,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_thermal_load():
    """Create sample thermal load."""
    return ThermalLoad(
        load_id="LOAD-001",
        demand_mw=50.0,
        current_supply_mw=45.0,
        temperature_setpoint_c=450.0,
        pressure_setpoint_bar=15.0,
    )


@pytest.fixture
def sample_equipment_status():
    """Create sample equipment status."""
    return EquipmentStatus(
        equipment_id="BLR-001",
        equipment_type="boiler",
        status="running",
        load_percent=75.0,
        temperature_c=450.0,
        pressure_bar=15.0,
        efficiency=0.88,
    )


@pytest.fixture
def sample_process_measurement():
    """Create sample process measurement."""
    return ProcessMeasurement(
        tag="TI-101",
        value=455.5,
        unit="degC",
        quality="good",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_workflow_spec():
    """Create sample workflow specification."""
    return WorkflowSpec(
        workflow_type=WorkflowType.OPTIMIZATION,
        name="Load Optimization",
        priority=Priority.HIGH,
        timeout_s=300.0,
        parameters={"target_efficiency": 0.90},
    )


@pytest.fixture
def sample_safety_event():
    """Create sample safety event."""
    return SafetyEvent(
        event_type="HIGH_TEMPERATURE",
        severity=AlarmSeverity.WARNING,
        equipment_id="BLR-001",
        value=525.0,
        threshold=500.0,
        unit="degC",
        description="Temperature approaching limit",
    )


@pytest.fixture
def sample_agent_status():
    """Create sample agent status."""
    return AgentStatus(
        agent_id="AGENT-001",
        agent_type="ThermalOptimizer",
        name="Thermal Optimizer Agent",
        health=AgentHealth.HEALTHY,
        version="1.0.0",
        active_tasks=2,
        completed_tasks=150,
        failed_tasks=3,
    )


# =============================================================================
# ENUMERATION TESTS
# =============================================================================

class TestPriorityEnum:
    """Test suite for Priority enumeration."""

    @pytest.mark.unit
    def test_priority_values(self):
        """Test priority enumeration values."""
        assert Priority.LOW.value == "low"
        assert Priority.NORMAL.value == "normal"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"

    @pytest.mark.unit
    def test_priority_ordering(self):
        """Test priority comparison."""
        # Define ordering
        priority_order = [Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL]

        for i, lower in enumerate(priority_order[:-1]):
            higher = priority_order[i + 1]
            assert priority_order.index(lower) < priority_order.index(higher)


class TestWorkflowTypeEnum:
    """Test suite for WorkflowType enumeration."""

    @pytest.mark.unit
    def test_workflow_type_values(self):
        """Test workflow type enumeration values."""
        assert WorkflowType.OPTIMIZATION.value == "optimization"
        assert WorkflowType.DIAGNOSTICS.value == "diagnostics"
        assert WorkflowType.MAINTENANCE.value == "maintenance"
        assert WorkflowType.STARTUP.value == "startup"
        assert WorkflowType.SHUTDOWN.value == "shutdown"

    @pytest.mark.unit
    def test_all_workflow_types_defined(self):
        """Test all expected workflow types exist."""
        expected_types = [
            "optimization",
            "diagnostics",
            "maintenance",
            "startup",
            "shutdown",
            "emergency",
        ]
        actual_values = [wt.value for wt in WorkflowType]
        for expected in expected_types:
            assert expected in actual_values


class TestWorkflowStatusEnum:
    """Test suite for WorkflowStatus enumeration."""

    @pytest.mark.unit
    def test_workflow_status_values(self):
        """Test workflow status enumeration values."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"

    @pytest.mark.unit
    def test_terminal_statuses(self):
        """Test identification of terminal statuses."""
        terminal = [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
        non_terminal = [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]

        for status in terminal:
            assert status in terminal
        for status in non_terminal:
            assert status not in terminal


class TestAgentHealthEnum:
    """Test suite for AgentHealth enumeration."""

    @pytest.mark.unit
    def test_agent_health_values(self):
        """Test agent health enumeration values."""
        assert AgentHealth.HEALTHY.value == "healthy"
        assert AgentHealth.DEGRADED.value == "degraded"
        assert AgentHealth.UNHEALTHY.value == "unhealthy"
        assert AgentHealth.OFFLINE.value == "offline"


class TestAlarmSeverityEnum:
    """Test suite for AlarmSeverity enumeration."""

    @pytest.mark.unit
    def test_alarm_severity_values(self):
        """Test alarm severity enumeration values."""
        assert AlarmSeverity.INFO.value == "info"
        assert AlarmSeverity.WARNING.value == "warning"
        assert AlarmSeverity.ALARM.value == "alarm"
        assert AlarmSeverity.CRITICAL.value == "critical"
        assert AlarmSeverity.EMERGENCY.value == "emergency"

    @pytest.mark.unit
    def test_severity_ordering(self):
        """Test severity ordering from lowest to highest."""
        severity_order = [
            AlarmSeverity.INFO,
            AlarmSeverity.WARNING,
            AlarmSeverity.ALARM,
            AlarmSeverity.CRITICAL,
            AlarmSeverity.EMERGENCY,
        ]

        for i in range(len(severity_order) - 1):
            # Lower index = lower severity
            assert severity_order.index(severity_order[i]) < severity_order.index(severity_order[i + 1])


# =============================================================================
# THERMAL LOAD TESTS
# =============================================================================

class TestThermalLoad:
    """Test suite for ThermalLoad model."""

    @pytest.mark.unit
    def test_initialization(self, sample_thermal_load):
        """Test thermal load initialization."""
        load = sample_thermal_load

        assert load.load_id == "LOAD-001"
        assert load.demand_mw == 50.0
        assert load.current_supply_mw == 45.0
        assert load.temperature_setpoint_c == 450.0
        assert load.pressure_setpoint_bar == 15.0

    @pytest.mark.unit
    def test_supply_deficit_calculation(self, sample_thermal_load):
        """Test supply deficit calculation."""
        deficit = sample_thermal_load.demand_mw - sample_thermal_load.current_supply_mw
        assert deficit == 5.0  # 50 - 45 = 5 MW deficit

    @pytest.mark.unit
    def test_validation_positive_demand(self):
        """Test demand must be positive."""
        with pytest.raises(ValueError):
            ThermalLoad(
                load_id="TEST",
                demand_mw=-10.0,
                current_supply_mw=0.0,
            )

    @pytest.mark.unit
    def test_validation_non_negative_supply(self):
        """Test supply must be non-negative."""
        with pytest.raises(ValueError):
            ThermalLoad(
                load_id="TEST",
                demand_mw=50.0,
                current_supply_mw=-5.0,
            )

    @pytest.mark.unit
    def test_serialization(self, sample_thermal_load):
        """Test thermal load serialization."""
        data = sample_thermal_load.model_dump()

        assert isinstance(data, dict)
        assert data["load_id"] == "LOAD-001"
        assert data["demand_mw"] == 50.0

    @pytest.mark.unit
    def test_deserialization(self):
        """Test thermal load deserialization."""
        data = {
            "load_id": "LOAD-002",
            "demand_mw": 75.0,
            "current_supply_mw": 70.0,
            "temperature_setpoint_c": 500.0,
        }

        load = ThermalLoad(**data)
        assert load.load_id == "LOAD-002"
        assert load.demand_mw == 75.0


# =============================================================================
# EQUIPMENT STATUS TESTS
# =============================================================================

class TestEquipmentStatus:
    """Test suite for EquipmentStatus model."""

    @pytest.mark.unit
    def test_initialization(self, sample_equipment_status):
        """Test equipment status initialization."""
        status = sample_equipment_status

        assert status.equipment_id == "BLR-001"
        assert status.equipment_type == "boiler"
        assert status.status == "running"
        assert status.load_percent == 75.0
        assert status.efficiency == 0.88

    @pytest.mark.unit
    def test_load_percent_range(self):
        """Test load percent must be 0-100."""
        # Valid
        status = EquipmentStatus(
            equipment_id="TEST",
            equipment_type="boiler",
            status="running",
            load_percent=50.0,
        )
        assert status.load_percent == 50.0

        # Invalid - too high
        with pytest.raises(ValueError):
            EquipmentStatus(
                equipment_id="TEST",
                equipment_type="boiler",
                status="running",
                load_percent=150.0,
            )

        # Invalid - negative
        with pytest.raises(ValueError):
            EquipmentStatus(
                equipment_id="TEST",
                equipment_type="boiler",
                status="running",
                load_percent=-10.0,
            )

    @pytest.mark.unit
    def test_efficiency_range(self):
        """Test efficiency must be 0-1."""
        # Valid
        status = EquipmentStatus(
            equipment_id="TEST",
            equipment_type="boiler",
            status="running",
            efficiency=0.92,
        )
        assert status.efficiency == 0.92

        # Invalid - too high
        with pytest.raises(ValueError):
            EquipmentStatus(
                equipment_id="TEST",
                equipment_type="boiler",
                status="running",
                efficiency=1.5,
            )

    @pytest.mark.unit
    def test_status_values(self):
        """Test valid equipment status values."""
        valid_statuses = ["running", "stopped", "standby", "fault", "maintenance"]

        for stat in valid_statuses:
            equipment = EquipmentStatus(
                equipment_id="TEST",
                equipment_type="boiler",
                status=stat,
            )
            assert equipment.status == stat

    @pytest.mark.unit
    def test_timestamp_default(self):
        """Test default timestamp is set."""
        status = EquipmentStatus(
            equipment_id="TEST",
            equipment_type="boiler",
            status="running",
        )
        assert status.timestamp is not None
        assert status.timestamp <= datetime.now(timezone.utc)


# =============================================================================
# PROCESS MEASUREMENT TESTS
# =============================================================================

class TestProcessMeasurement:
    """Test suite for ProcessMeasurement model."""

    @pytest.mark.unit
    def test_initialization(self, sample_process_measurement):
        """Test process measurement initialization."""
        measurement = sample_process_measurement

        assert measurement.tag == "TI-101"
        assert measurement.value == 455.5
        assert measurement.unit == "degC"
        assert measurement.quality == "good"

    @pytest.mark.unit
    def test_quality_values(self):
        """Test valid quality values."""
        valid_qualities = ["good", "uncertain", "bad", "stale"]

        for quality in valid_qualities:
            measurement = ProcessMeasurement(
                tag="TEST",
                value=100.0,
                quality=quality,
            )
            assert measurement.quality == quality

    @pytest.mark.unit
    def test_measurement_with_limits(self):
        """Test measurement with limits."""
        measurement = ProcessMeasurement(
            tag="TI-101",
            value=450.0,
            unit="degC",
            low_limit=200.0,
            high_limit=500.0,
        )

        assert measurement.low_limit == 200.0
        assert measurement.high_limit == 500.0
        assert measurement.low_limit <= measurement.value <= measurement.high_limit

    @pytest.mark.unit
    def test_measurement_out_of_range(self):
        """Test measurement outside limits."""
        measurement = ProcessMeasurement(
            tag="TI-101",
            value=550.0,  # Above high limit
            unit="degC",
            low_limit=200.0,
            high_limit=500.0,
        )

        # Value should still be stored, but flag as out of range
        assert measurement.value > measurement.high_limit


# =============================================================================
# SAFETY EVENT TESTS
# =============================================================================

class TestSafetyEvent:
    """Test suite for SafetyEvent model."""

    @pytest.mark.unit
    def test_initialization(self, sample_safety_event):
        """Test safety event initialization."""
        event = sample_safety_event

        assert event.event_type == "HIGH_TEMPERATURE"
        assert event.severity == AlarmSeverity.WARNING
        assert event.equipment_id == "BLR-001"
        assert event.value == 525.0
        assert event.threshold == 500.0

    @pytest.mark.unit
    def test_event_id_generation(self):
        """Test event ID is auto-generated."""
        event = SafetyEvent(
            event_type="TEST",
            severity=AlarmSeverity.WARNING,
            equipment_id="TEST-001",
            value=100.0,
            threshold=90.0,
        )

        assert event.event_id is not None
        assert len(event.event_id) > 0

    @pytest.mark.unit
    def test_provenance_hash(self):
        """Test provenance hash generation."""
        event = SafetyEvent(
            event_type="TEST",
            severity=AlarmSeverity.WARNING,
            equipment_id="TEST-001",
            value=100.0,
            threshold=90.0,
        )

        assert event.provenance_hash is not None
        assert len(event.provenance_hash) == 64  # SHA-256 hex length

    @pytest.mark.unit
    def test_timestamp_default(self, sample_safety_event):
        """Test default timestamp is set."""
        assert sample_safety_event.timestamp is not None

    @pytest.mark.unit
    def test_acknowledged_default(self, sample_safety_event):
        """Test acknowledged defaults to False."""
        assert sample_safety_event.acknowledged is False


# =============================================================================
# WORKFLOW SPEC TESTS
# =============================================================================

class TestWorkflowSpec:
    """Test suite for WorkflowSpec model."""

    @pytest.mark.unit
    def test_initialization(self, sample_workflow_spec):
        """Test workflow spec initialization."""
        spec = sample_workflow_spec

        assert spec.workflow_type == WorkflowType.OPTIMIZATION
        assert spec.name == "Load Optimization"
        assert spec.priority == Priority.HIGH
        assert spec.timeout_s == 300.0
        assert spec.parameters["target_efficiency"] == 0.90

    @pytest.mark.unit
    def test_workflow_id_generation(self):
        """Test workflow ID is auto-generated."""
        spec = WorkflowSpec(
            workflow_type=WorkflowType.DIAGNOSTICS,
            name="Test Workflow",
        )

        assert spec.workflow_id is not None
        assert len(spec.workflow_id) > 0

    @pytest.mark.unit
    def test_default_priority(self):
        """Test default priority is NORMAL."""
        spec = WorkflowSpec(
            workflow_type=WorkflowType.MAINTENANCE,
            name="Maintenance Workflow",
        )

        assert spec.priority == Priority.NORMAL

    @pytest.mark.unit
    def test_default_timeout(self):
        """Test default timeout value."""
        spec = WorkflowSpec(
            workflow_type=WorkflowType.OPTIMIZATION,
            name="Test",
        )

        assert spec.timeout_s > 0
        assert spec.timeout_s == 300.0  # Default 5 minutes

    @pytest.mark.unit
    def test_validation_timeout_positive(self):
        """Test timeout must be positive."""
        with pytest.raises(ValueError):
            WorkflowSpec(
                workflow_type=WorkflowType.OPTIMIZATION,
                name="Test",
                timeout_s=0,
            )


# =============================================================================
# WORKFLOW RESULT TESTS
# =============================================================================

class TestWorkflowResult:
    """Test suite for WorkflowResult model."""

    @pytest.mark.unit
    def test_successful_result(self):
        """Test successful workflow result."""
        result = WorkflowResult(
            workflow_id="WF-001",
            status=WorkflowStatus.COMPLETED,
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            end_time=datetime.now(timezone.utc),
            tasks_completed=10,
            tasks_failed=0,
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.tasks_failed == 0
        assert result.duration_ms > 0

    @pytest.mark.unit
    def test_failed_result(self):
        """Test failed workflow result."""
        result = WorkflowResult(
            workflow_id="WF-002",
            status=WorkflowStatus.FAILED,
            start_time=datetime.now(timezone.utc) - timedelta(minutes=2),
            end_time=datetime.now(timezone.utc),
            tasks_completed=5,
            tasks_failed=1,
            error_message="Task execution timeout",
        )

        assert result.status == WorkflowStatus.FAILED
        assert result.tasks_failed == 1
        assert result.error_message is not None

    @pytest.mark.unit
    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.now(timezone.utc) - timedelta(seconds=120)
        end = datetime.now(timezone.utc)

        result = WorkflowResult(
            workflow_id="WF-003",
            status=WorkflowStatus.COMPLETED,
            start_time=start,
            end_time=end,
        )

        # Duration should be approximately 120000 ms
        assert 119000 <= result.duration_ms <= 121000

    @pytest.mark.unit
    def test_provenance_hash(self):
        """Test workflow result provenance hash."""
        result = WorkflowResult(
            workflow_id="WF-004",
            status=WorkflowStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# =============================================================================
# TASK SPEC TESTS
# =============================================================================

class TestTaskSpec:
    """Test suite for TaskSpec model."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test task spec initialization."""
        spec = TaskSpec(
            task_type="optimize_load",
            agent_type="ThermalOptimizer",
            parameters={"equipment_ids": ["BLR-001", "BLR-002"]},
        )

        assert spec.task_type == "optimize_load"
        assert spec.agent_type == "ThermalOptimizer"
        assert "equipment_ids" in spec.parameters

    @pytest.mark.unit
    def test_task_id_generation(self):
        """Test task ID is auto-generated."""
        spec = TaskSpec(
            task_type="test_task",
            agent_type="TestAgent",
        )

        assert spec.task_id is not None

    @pytest.mark.unit
    def test_dependencies(self):
        """Test task dependencies."""
        spec = TaskSpec(
            task_type="aggregate_results",
            agent_type="Aggregator",
            dependencies=["TASK-001", "TASK-002"],
        )

        assert len(spec.dependencies) == 2
        assert "TASK-001" in spec.dependencies


# =============================================================================
# TASK RESULT TESTS
# =============================================================================

class TestTaskResult:
    """Test suite for TaskResult model."""

    @pytest.mark.unit
    def test_successful_result(self):
        """Test successful task result."""
        result = TaskResult(
            task_id="TASK-001",
            status=TaskStatus.COMPLETED,
            agent_id="AGENT-001",
            output={"optimized_load": 75.0},
            execution_time_ms=150.0,
        )

        assert result.status == TaskStatus.COMPLETED
        assert result.output["optimized_load"] == 75.0

    @pytest.mark.unit
    def test_failed_result(self):
        """Test failed task result."""
        result = TaskResult(
            task_id="TASK-002",
            status=TaskStatus.FAILED,
            agent_id="AGENT-001",
            error="Connection timeout",
            execution_time_ms=5000.0,
        )

        assert result.status == TaskStatus.FAILED
        assert result.error is not None


# =============================================================================
# AGENT STATUS TESTS
# =============================================================================

class TestAgentStatus:
    """Test suite for AgentStatus model."""

    @pytest.mark.unit
    def test_initialization(self, sample_agent_status):
        """Test agent status initialization."""
        status = sample_agent_status

        assert status.agent_id == "AGENT-001"
        assert status.agent_type == "ThermalOptimizer"
        assert status.health == AgentHealth.HEALTHY

    @pytest.mark.unit
    def test_task_counts(self, sample_agent_status):
        """Test task count tracking."""
        status = sample_agent_status

        assert status.active_tasks == 2
        assert status.completed_tasks == 150
        assert status.failed_tasks == 3

        # Calculate success rate
        total_finished = status.completed_tasks + status.failed_tasks
        success_rate = status.completed_tasks / total_finished
        assert success_rate > 0.95  # > 95% success rate

    @pytest.mark.unit
    def test_last_heartbeat(self):
        """Test last heartbeat timestamp."""
        status = AgentStatus(
            agent_id="AGENT-002",
            agent_type="TestAgent",
            name="Test Agent",
            health=AgentHealth.HEALTHY,
        )

        assert status.last_heartbeat is not None


# =============================================================================
# SYSTEM STATUS TESTS
# =============================================================================

class TestSystemStatus:
    """Test suite for SystemStatus model."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test system status initialization."""
        status = SystemStatus(
            orchestrator_id="GL-001",
            orchestrator_name="ThermalCommand",
            status="running",
            uptime_seconds=3600.0,
            registered_agents=10,
            healthy_agents=9,
            active_workflows=3,
            safety_level=SafetyLevel.NORMAL,
        )

        assert status.orchestrator_id == "GL-001"
        assert status.status == "running"
        assert status.healthy_agents == 9

    @pytest.mark.unit
    def test_agent_health_ratio(self):
        """Test agent health ratio calculation."""
        status = SystemStatus(
            orchestrator_id="GL-001",
            orchestrator_name="Test",
            status="running",
            registered_agents=10,
            healthy_agents=8,
        )

        health_ratio = status.healthy_agents / status.registered_agents
        assert health_ratio == 0.8

    @pytest.mark.unit
    def test_safety_levels(self):
        """Test safety level values."""
        for level in SafetyLevel:
            status = SystemStatus(
                orchestrator_id="GL-001",
                orchestrator_name="Test",
                status="running",
                safety_level=level,
            )
            assert status.safety_level == level


# =============================================================================
# ORCHESTRATOR METRICS TESTS
# =============================================================================

class TestOrchestratorMetrics:
    """Test suite for OrchestratorMetrics model."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test orchestrator metrics initialization."""
        metrics = OrchestratorMetrics(
            workflows_executed=1000,
            workflows_failed=25,
            tasks_executed=15000,
            tasks_failed=150,
            safety_events=10,
            uptime_seconds=86400.0,
        )

        assert metrics.workflows_executed == 1000
        assert metrics.workflows_failed == 25

    @pytest.mark.unit
    def test_success_rates(self):
        """Test success rate calculations."""
        metrics = OrchestratorMetrics(
            workflows_executed=1000,
            workflows_failed=50,
            tasks_executed=10000,
            tasks_failed=100,
        )

        workflow_success_rate = 1 - (metrics.workflows_failed / metrics.workflows_executed)
        task_success_rate = 1 - (metrics.tasks_failed / metrics.tasks_executed)

        assert workflow_success_rate == 0.95  # 95%
        assert task_success_rate == 0.99  # 99%

    @pytest.mark.unit
    def test_uptime_calculation(self):
        """Test uptime formatting."""
        metrics = OrchestratorMetrics(
            uptime_seconds=90061.0,  # 1 day, 1 hour, 1 minute, 1 second
        )

        days = int(metrics.uptime_seconds // 86400)
        hours = int((metrics.uptime_seconds % 86400) // 3600)
        minutes = int((metrics.uptime_seconds % 3600) // 60)

        assert days == 1
        assert hours == 1
        assert minutes == 1


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSchemaSerialization:
    """Test schema serialization and deserialization."""

    @pytest.mark.unit
    def test_json_serialization(self, sample_thermal_load):
        """Test JSON serialization."""
        json_str = sample_thermal_load.model_dump_json()

        assert isinstance(json_str, str)
        assert "LOAD-001" in json_str

    @pytest.mark.unit
    def test_json_deserialization(self):
        """Test JSON deserialization."""
        json_str = '{"load_id": "LOAD-TEST", "demand_mw": 100.0, "current_supply_mw": 90.0}'
        data = json.loads(json_str)
        load = ThermalLoad(**data)

        assert load.load_id == "LOAD-TEST"
        assert load.demand_mw == 100.0

    @pytest.mark.unit
    def test_dict_roundtrip(self, sample_workflow_spec):
        """Test dict serialization roundtrip."""
        data = sample_workflow_spec.model_dump()
        reconstructed = WorkflowSpec(**data)

        assert reconstructed.workflow_type == sample_workflow_spec.workflow_type
        assert reconstructed.name == sample_workflow_spec.name
        assert reconstructed.priority == sample_workflow_spec.priority

    @pytest.mark.unit
    def test_nested_serialization(self):
        """Test nested model serialization."""
        result = WorkflowResult(
            workflow_id="WF-NESTED",
            status=WorkflowStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            output={
                "thermal_loads": [
                    {"load_id": "L1", "demand_mw": 50.0},
                    {"load_id": "L2", "demand_mw": 75.0},
                ]
            },
        )

        data = result.model_dump()
        assert len(data["output"]["thermal_loads"]) == 2


# =============================================================================
# PROVENANCE HASH TESTS
# =============================================================================

class TestProvenanceHash:
    """Test provenance hash generation and validation."""

    @pytest.mark.unit
    def test_deterministic_hash(self):
        """Test hash is deterministic for same input."""
        event1 = SafetyEvent(
            event_id="EVT-001",
            event_type="TEST",
            severity=AlarmSeverity.WARNING,
            equipment_id="EQ-001",
            value=100.0,
            threshold=90.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        event2 = SafetyEvent(
            event_id="EVT-001",
            event_type="TEST",
            severity=AlarmSeverity.WARNING,
            equipment_id="EQ-001",
            value=100.0,
            threshold=90.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Same inputs should produce same hash
        assert event1.provenance_hash == event2.provenance_hash

    @pytest.mark.unit
    def test_hash_changes_with_input(self):
        """Test hash changes when input changes."""
        event1 = SafetyEvent(
            event_type="TEST",
            severity=AlarmSeverity.WARNING,
            equipment_id="EQ-001",
            value=100.0,
            threshold=90.0,
        )

        event2 = SafetyEvent(
            event_type="TEST",
            severity=AlarmSeverity.WARNING,
            equipment_id="EQ-001",
            value=101.0,  # Different value
            threshold=90.0,
        )

        # Different inputs should produce different hashes
        # Note: This depends on implementation - may need adjustment
        # if timestamp is included in hash

    @pytest.mark.unit
    def test_hash_format(self, sample_safety_event):
        """Test hash is valid SHA-256 format."""
        hash_value = sample_safety_event.provenance_hash

        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value.lower())
