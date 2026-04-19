"""
Test suite for MultiBurnerOrchestrator.

Tests cover:
- Sequencing logic (coordinated start/stop)
- Load balancing strategies
- Lead/lag rotation
- Failover scenarios
- Safety coordination
- Communication layer

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

from orchestration.multi_burner import (
    MultiBurnerOrchestrator,
    OrchestrationConfig,
    BurnerState,
    BurnerRole,
    BurnerStatus,
    SequencePhase,
    LoadBalancingStrategy,
    CoordinationStrategy,
    CommandType,
    BurnerCommand,
    LoadDistribution,
    SafetyCoordinationStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Create default orchestration configuration."""
    return OrchestrationConfig(
        num_burners=4,
        redundancy_mode=True,
        coordination_strategy=CoordinationStrategy.SEQUENTIAL,
        load_balancing_strategy=LoadBalancingStrategy.EQUAL,
        start_delay_between_burners_s=0.1,  # Fast for testing
        stop_delay_between_burners_s=0.1,
        flame_prove_timeout_s=1.0,
        prepurge_duration_s=0.1,
        postpurge_duration_s=0.1,
        rotation_interval_hours=168.0,
        heartbeat_interval_ms=100,
        communication_timeout_ms=1000,
    )


@pytest.fixture
def orchestrator(default_config):
    """Create orchestrator with default configuration."""
    orch = MultiBurnerOrchestrator(default_config)
    orch.initialize_burners(["BRN-001", "BRN-002", "BRN-003", "BRN-004"])
    return orch


@pytest.fixture
def two_burner_config():
    """Create configuration for 2-burner system."""
    return OrchestrationConfig(
        num_burners=2,
        redundancy_mode=False,
        load_balancing_strategy=LoadBalancingStrategy.EQUAL,
        start_delay_between_burners_s=0.1,
        prepurge_duration_s=0.1,
        postpurge_duration_s=0.1,
        flame_prove_timeout_s=1.0,
    )


@pytest.fixture
def two_burner_orchestrator(two_burner_config):
    """Create 2-burner orchestrator."""
    orch = MultiBurnerOrchestrator(two_burner_config)
    orch.initialize_burners(["BRN-001", "BRN-002"])
    return orch


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Tests for orchestrator initialization."""

    def test_initialize_with_valid_burners(self, default_config):
        """Test initialization with correct number of burners."""
        orch = MultiBurnerOrchestrator(default_config)
        orch.initialize_burners(["BRN-001", "BRN-002", "BRN-003", "BRN-004"])

        assert len(orch.state.burner_statuses) == 4
        assert "BRN-001" in orch.state.burner_statuses
        assert "BRN-004" in orch.state.burner_statuses

    def test_initialize_wrong_count_raises_error(self, default_config):
        """Test that wrong burner count raises ValueError."""
        orch = MultiBurnerOrchestrator(default_config)

        with pytest.raises(ValueError) as exc_info:
            orch.initialize_burners(["BRN-001", "BRN-002"])

        assert "Expected 4 burners" in str(exc_info.value)

    def test_initial_role_assignment(self, orchestrator):
        """Test that roles are assigned correctly on initialization."""
        roles = orchestrator.state.role_assignments

        assert roles["BRN-001"] == BurnerRole.LEAD
        assert roles["BRN-002"] == BurnerRole.LAG_1
        assert roles["BRN-003"] == BurnerRole.LAG_2
        # Last burner is standby in redundancy mode
        assert roles["BRN-004"] == BurnerRole.STANDBY_RESERVE

    def test_initial_state_is_standby(self, orchestrator):
        """Test that all burners start in STANDBY state."""
        for status in orchestrator.state.burner_statuses.values():
            assert status.state == BurnerState.STANDBY

    def test_safety_status_initialized(self, orchestrator):
        """Test that safety status is properly initialized."""
        safety = orchestrator.state.safety_status

        assert safety is not None
        assert len(safety.burner_interlock_status) == 4
        assert len(safety.burner_flame_status) == 4


# =============================================================================
# SEQUENCING TESTS
# =============================================================================

class TestCoordinatedStart:
    """Tests for coordinated start sequence."""

    @pytest.mark.asyncio
    async def test_coordinated_start_success(self, orchestrator):
        """Test successful coordinated start sequence."""
        # Set interlocks satisfied and flame proven for simulation
        for status in orchestrator.state.burner_statuses.values():
            status.interlocks_satisfied = True
            status.flame_proven = True

        success, events = await orchestrator.execute_coordinated_start(target_load_pct=50.0)

        assert success is True
        assert len(events) > 0
        assert orchestrator.state.total_demand_pct == 50.0

    @pytest.mark.asyncio
    async def test_coordinated_start_blocked_by_interlock(self, orchestrator):
        """Test that start is blocked when interlocks not satisfied."""
        # Leave interlocks unsatisfied (default)
        success, events = await orchestrator.execute_coordinated_start(target_load_pct=50.0)

        assert success is False
        # Find the blocking event
        blocking_events = [e for e in events if "blocked" in e.event_type]
        assert len(blocking_events) > 0

    @pytest.mark.asyncio
    async def test_start_sequence_creates_events(self, orchestrator):
        """Test that start sequence creates proper event log."""
        for status in orchestrator.state.burner_statuses.values():
            status.interlocks_satisfied = True
            status.flame_proven = True

        success, events = await orchestrator.execute_coordinated_start(target_load_pct=50.0)

        # Check for sequence started event
        started_events = [e for e in events if e.event_type == "sequence_started"]
        assert len(started_events) >= 1


class TestCoordinatedStop:
    """Tests for coordinated stop sequence."""

    @pytest.mark.asyncio
    async def test_coordinated_stop_success(self, orchestrator):
        """Test successful coordinated stop sequence."""
        # First start some burners
        for bid in ["BRN-001", "BRN-002"]:
            status = orchestrator.state.burner_statuses[bid]
            status.state = BurnerState.MODULATING
            status.flame_proven = True
            status.interlocks_satisfied = True

        success, events = await orchestrator.execute_coordinated_stop()

        assert success is True
        # All burners should be in standby
        for bid in ["BRN-001", "BRN-002"]:
            assert orchestrator.state.burner_statuses[bid].state == BurnerState.STANDBY

    @pytest.mark.asyncio
    async def test_stop_order_is_correct(self, orchestrator):
        """Test that burners stop in correct order (lead last)."""
        # Set up 3 active burners
        for bid, state in [("BRN-001", BurnerState.MODULATING),
                          ("BRN-002", BurnerState.MODULATING),
                          ("BRN-003", BurnerState.MODULATING)]:
            status = orchestrator.state.burner_statuses[bid]
            status.state = state
            status.flame_proven = True
            status.interlocks_satisfied = True

        success, events = await orchestrator.execute_coordinated_stop()

        # Get stop events in order
        stop_events = [e for e in events if e.event_type == "burner_stopped"]

        # Lead burner (BRN-001) should be last
        if len(stop_events) >= 3:
            assert stop_events[-1].burner_id == "BRN-001"


class TestEmergencyShutdown:
    """Tests for emergency shutdown."""

    @pytest.mark.asyncio
    async def test_emergency_shutdown_stops_all_burners(self, orchestrator):
        """Test that emergency shutdown stops all burners immediately."""
        # Set all burners active
        for status in orchestrator.state.burner_statuses.values():
            status.state = BurnerState.MODULATING
            status.flame_proven = True

        success, events = await orchestrator.execute_coordinated_stop(emergency=True)

        assert success is True
        assert orchestrator.state.emergency_stop_triggered is True

        # All burners should be in lockout
        for status in orchestrator.state.burner_statuses.values():
            assert status.state == BurnerState.LOCKOUT
            assert status.in_lockout is True

    @pytest.mark.asyncio
    async def test_emergency_shutdown_updates_safety_status(self, orchestrator):
        """Test that emergency shutdown updates safety status."""
        for status in orchestrator.state.burner_statuses.values():
            status.state = BurnerState.MODULATING

        await orchestrator.execute_coordinated_stop(emergency=True)

        assert orchestrator.state.safety_status.emergency_stop_active is True


# =============================================================================
# LOAD BALANCING TESTS
# =============================================================================

class TestLoadBalancing:
    """Tests for load balancing strategies."""

    def test_equal_distribution(self, two_burner_orchestrator):
        """Test equal load distribution."""
        orch = two_burner_orchestrator
        orch.config.load_balancing_strategy = LoadBalancingStrategy.EQUAL

        # Set both burners active
        for status in orch.state.burner_statuses.values():
            status.state = BurnerState.MODULATING

        orch.state.total_demand_pct = 80.0
        distribution = orch.calculate_load_distribution()

        assert distribution.strategy_used == LoadBalancingStrategy.EQUAL
        assert len(distribution.burner_loads) == 2

        # Equal distribution should give 40% each
        for load in distribution.burner_loads.values():
            assert abs(load - 40.0) < 0.01

    def test_efficiency_based_distribution(self, two_burner_orchestrator):
        """Test efficiency-based load distribution."""
        orch = two_burner_orchestrator
        orch.config.load_balancing_strategy = LoadBalancingStrategy.EFFICIENCY_BASED

        # Set different efficiencies
        orch.state.burner_statuses["BRN-001"].efficiency_pct = 90.0
        orch.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING

        orch.state.burner_statuses["BRN-002"].efficiency_pct = 80.0
        orch.state.burner_statuses["BRN-002"].state = BurnerState.MODULATING

        orch.state.total_demand_pct = 100.0
        distribution = orch.calculate_load_distribution()

        # Higher efficiency burner should get more load
        assert distribution.burner_loads["BRN-001"] > distribution.burner_loads["BRN-002"]

    def test_wear_leveling_distribution(self, two_burner_orchestrator):
        """Test wear-leveling load distribution."""
        orch = two_burner_orchestrator
        orch.config.load_balancing_strategy = LoadBalancingStrategy.WEAR_LEVELING

        # Set different runtimes
        orch.state.burner_statuses["BRN-001"].total_runtime_hours = 1000.0
        orch.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING

        orch.state.burner_statuses["BRN-002"].total_runtime_hours = 500.0
        orch.state.burner_statuses["BRN-002"].state = BurnerState.MODULATING

        orch.state.total_demand_pct = 100.0
        distribution = orch.calculate_load_distribution()

        # Less runtime burner should get more load
        assert distribution.burner_loads["BRN-002"] > distribution.burner_loads["BRN-001"]

    def test_hybrid_distribution(self, two_burner_orchestrator):
        """Test hybrid load distribution."""
        orch = two_burner_orchestrator
        orch.config.load_balancing_strategy = LoadBalancingStrategy.HYBRID

        for status in orch.state.burner_statuses.values():
            status.state = BurnerState.MODULATING
            status.efficiency_pct = 85.0

        orch.state.total_demand_pct = 100.0
        distribution = orch.calculate_load_distribution()

        assert distribution.strategy_used == LoadBalancingStrategy.HYBRID
        # Should have load distributed
        assert sum(distribution.burner_loads.values()) == pytest.approx(100.0, rel=0.01)

    def test_load_distribution_with_no_active_burners(self, orchestrator):
        """Test load distribution when no burners active."""
        # All burners in standby (default)
        orchestrator.state.total_demand_pct = 50.0
        distribution = orchestrator.calculate_load_distribution()

        assert len(distribution.active_burners) == 0
        assert len(distribution.standby_burners) == 4

    def test_set_load_demand_updates_distribution(self, two_burner_orchestrator):
        """Test that set_load_demand updates distribution."""
        orch = two_burner_orchestrator

        for status in orch.state.burner_statuses.values():
            status.state = BurnerState.MODULATING

        distribution = orch.set_load_demand(80.0)

        assert orch.state.total_demand_pct == 80.0
        assert orch.state.current_distribution is not None
        assert sum(distribution.burner_loads.values()) == pytest.approx(80.0, rel=0.01)

    def test_set_load_demand_invalid_value(self, orchestrator):
        """Test that invalid demand value raises error."""
        with pytest.raises(ValueError):
            orchestrator.set_load_demand(150.0)

        with pytest.raises(ValueError):
            orchestrator.set_load_demand(-10.0)


# =============================================================================
# LEAD/LAG ROTATION TESTS
# =============================================================================

class TestLeadLagRotation:
    """Tests for lead/lag rotation."""

    @pytest.mark.asyncio
    async def test_rotation_changes_roles(self, orchestrator):
        """Test that rotation properly changes roles."""
        # Get initial lead
        initial_lead = None
        for bid, role in orchestrator.state.role_assignments.items():
            if role == BurnerRole.LEAD:
                initial_lead = bid
                break

        success, new_assignments = await orchestrator.execute_lead_lag_rotation()

        assert success is True

        # Original lead should no longer be lead
        assert new_assignments[initial_lead] != BurnerRole.LEAD

        # There should still be exactly one lead
        lead_count = sum(1 for role in new_assignments.values() if role == BurnerRole.LEAD)
        assert lead_count == 1

    @pytest.mark.asyncio
    async def test_rotation_updates_history(self, orchestrator):
        """Test that rotation is recorded in history."""
        initial_history_len = len(orchestrator.state.rotation_history)

        await orchestrator.execute_lead_lag_rotation()

        assert len(orchestrator.state.rotation_history) == initial_history_len + 1
        assert orchestrator.state.last_rotation_time is not None

    @pytest.mark.asyncio
    async def test_rotation_resets_runtime_tracking(self, orchestrator):
        """Test that rotation resets runtime tracking."""
        # Set some runtime
        for status in orchestrator.state.burner_statuses.values():
            status.runtime_since_last_rotation = 50.0

        await orchestrator.execute_lead_lag_rotation()

        # Runtime should be reset for rotated burners
        for bid, role in orchestrator.state.role_assignments.items():
            if role != BurnerRole.STANDBY_RESERVE:
                assert orchestrator.state.burner_statuses[bid].runtime_since_last_rotation == 0.0

    def test_should_rotate_time_based(self, orchestrator):
        """Test time-based rotation trigger."""
        # No rotation yet
        assert orchestrator.should_rotate() is False

        # Set old rotation time
        orchestrator.state.last_rotation_time = datetime.now(timezone.utc) - timedelta(hours=200)

        assert orchestrator.should_rotate() is True

    def test_should_rotate_imbalance_based(self, orchestrator):
        """Test runtime imbalance rotation trigger."""
        orchestrator.state.last_rotation_time = datetime.now(timezone.utc)

        # Set imbalanced runtimes
        orchestrator.state.burner_statuses["BRN-001"].runtime_since_last_rotation = 150.0
        orchestrator.state.burner_statuses["BRN-002"].runtime_since_last_rotation = 10.0

        assert orchestrator.should_rotate() is True


# =============================================================================
# FAILOVER TESTS
# =============================================================================

class TestFailover:
    """Tests for N+1 redundancy failover."""

    @pytest.mark.asyncio
    async def test_failover_success(self, orchestrator):
        """Test successful failover from failed burner to standby."""
        # Set up BRN-001 as active lead
        orchestrator.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING
        orchestrator.state.burner_statuses["BRN-001"].flame_proven = True
        orchestrator.state.burner_statuses["BRN-001"].target_load_pct = 50.0
        orchestrator.state.burner_statuses["BRN-001"].interlocks_satisfied = True

        # BRN-004 is standby
        orchestrator.state.burner_statuses["BRN-004"].interlocks_satisfied = True
        orchestrator.state.burner_statuses["BRN-004"].flame_proven = True

        success, replacement = await orchestrator.execute_failover("BRN-001")

        assert success is True
        assert replacement == "BRN-004"

        # Failed burner should be in maintenance
        assert orchestrator.state.burner_statuses["BRN-001"].role == BurnerRole.MAINTENANCE
        assert orchestrator.state.burner_statuses["BRN-001"].state == BurnerState.FAULT

        # Replacement should have taken over role
        assert orchestrator.state.burner_statuses["BRN-004"].role == BurnerRole.LEAD
        assert orchestrator.state.burner_statuses["BRN-004"].state == BurnerState.MODULATING

    @pytest.mark.asyncio
    async def test_failover_no_standby_available(self, two_burner_orchestrator):
        """Test failover when no standby burner available."""
        orch = two_burner_orchestrator
        # 2-burner config has no redundancy

        success, replacement = await orch.execute_failover("BRN-001")

        assert success is False
        assert replacement is None

    @pytest.mark.asyncio
    async def test_failover_preserves_load(self, orchestrator):
        """Test that failover preserves load assignment."""
        original_load = 60.0
        orchestrator.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING
        orchestrator.state.burner_statuses["BRN-001"].target_load_pct = original_load
        orchestrator.state.burner_statuses["BRN-001"].flame_proven = True
        orchestrator.state.burner_statuses["BRN-001"].interlocks_satisfied = True

        orchestrator.state.burner_statuses["BRN-004"].interlocks_satisfied = True
        orchestrator.state.burner_statuses["BRN-004"].flame_proven = True

        success, replacement = await orchestrator.execute_failover("BRN-001")

        assert success is True
        assert orchestrator.state.burner_statuses[replacement].target_load_pct == original_load


# =============================================================================
# SAFETY COORDINATION TESTS
# =============================================================================

class TestSafetyCoordination:
    """Tests for safety coordination."""

    def test_check_cross_burner_interlocks(self, orchestrator):
        """Test cross-burner interlock checking."""
        # Set all interlocks satisfied
        for status in orchestrator.state.burner_statuses.values():
            status.interlocks_satisfied = True

        safety_status = orchestrator.check_cross_burner_interlocks()

        assert safety_status.all_interlocks_satisfied is True
        assert safety_status.cross_burner_check_passed is True
        assert len(safety_status.active_interlocks) == 0

    def test_check_interlocks_with_failure(self, orchestrator):
        """Test interlock check detects failure."""
        # Set one interlock failed
        orchestrator.state.burner_statuses["BRN-002"].interlocks_satisfied = False

        safety_status = orchestrator.check_cross_burner_interlocks()

        assert safety_status.all_interlocks_satisfied is False
        assert len(safety_status.active_interlocks) > 0

    def test_supervise_flames(self, orchestrator):
        """Test flame supervision across burners."""
        # Set some burners active
        orchestrator.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING
        orchestrator.state.burner_statuses["BRN-001"].flame_proven = True

        orchestrator.state.burner_statuses["BRN-002"].state = BurnerState.MODULATING
        orchestrator.state.burner_statuses["BRN-002"].flame_proven = False  # Flame failure

        flame_status = orchestrator.supervise_flames()

        assert flame_status["BRN-001"] is True
        assert flame_status["BRN-002"] is False

    def test_supervise_flames_logs_failure(self, orchestrator):
        """Test that flame failure is logged."""
        orchestrator.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING
        orchestrator.state.burner_statuses["BRN-001"].flame_proven = False

        initial_events = len(orchestrator.state.event_log)

        orchestrator.supervise_flames()

        # Should have logged flame failure event
        assert len(orchestrator.state.event_log) > initial_events
        failure_events = [e for e in orchestrator.state.event_log if "flame_failure" in e.event_type]
        assert len(failure_events) > 0

    @pytest.mark.asyncio
    async def test_coordinated_purge_sequence(self, orchestrator):
        """Test coordinated purge across all burners."""
        success = await orchestrator.coordinate_purge_sequence()

        assert success is True
        assert orchestrator.state.safety_status.purge_in_progress is False
        assert len(orchestrator.state.safety_status.purge_complete_burners) == 4

    def test_cross_burner_conditions_communication_timeout(self, orchestrator):
        """Test that communication timeout fails cross-burner check."""
        # Set one burner with old communication timestamp
        old_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        orchestrator.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING
        orchestrator.state.burner_statuses["BRN-001"].last_communication = old_time

        result = orchestrator._check_cross_burner_conditions()

        assert result is False

    def test_safety_status_provenance_hash(self, orchestrator):
        """Test that safety status includes provenance hash."""
        for status in orchestrator.state.burner_statuses.values():
            status.interlocks_satisfied = True

        safety_status = orchestrator.check_cross_burner_interlocks()

        assert safety_status.provenance_hash != ""
        assert len(safety_status.provenance_hash) == 16


# =============================================================================
# COMMUNICATION TESTS
# =============================================================================

class TestCommunication:
    """Tests for communication and state sharing."""

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring_start_stop(self, orchestrator):
        """Test heartbeat monitoring lifecycle."""
        await orchestrator.start_heartbeat_monitoring()
        assert orchestrator._running is True
        assert orchestrator._heartbeat_task is not None

        await orchestrator.stop_heartbeat_monitoring()
        assert orchestrator._running is False

    def test_update_burner_status(self, orchestrator):
        """Test updating burner status."""
        orchestrator.update_burner_status("BRN-001", {
            "firing_rate_pct": 75.0,
            "flame_proven": True,
            "efficiency_pct": 88.0,
        })

        status = orchestrator.state.burner_statuses["BRN-001"]
        assert status.firing_rate_pct == 75.0
        assert status.flame_proven is True
        assert status.efficiency_pct == 88.0

    def test_update_burner_status_updates_communication_time(self, orchestrator):
        """Test that status update refreshes communication timestamp."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        orchestrator.state.burner_statuses["BRN-001"].last_communication = old_time

        orchestrator.update_burner_status("BRN-001", {"firing_rate_pct": 50.0})

        new_time = orchestrator.state.burner_statuses["BRN-001"].last_communication
        assert new_time > old_time

    def test_update_unknown_burner(self, orchestrator):
        """Test updating unknown burner doesn't crash."""
        # Should log warning but not raise
        orchestrator.update_burner_status("UNKNOWN-001", {"firing_rate_pct": 50.0})

    def test_broadcast_event(self, orchestrator):
        """Test event broadcasting."""
        initial_events = len(orchestrator.state.event_log)

        orchestrator.broadcast_event("test_event", {"key": "value"})

        assert len(orchestrator.state.event_log) == initial_events + 1
        last_event = orchestrator.state.event_log[-1]
        assert last_event.event_type == "test_event"
        assert last_event.details["key"] == "value"

    def test_get_orchestration_status(self, orchestrator):
        """Test getting full orchestration status."""
        status = orchestrator.get_orchestration_status()

        assert "current_phase" in status
        assert "total_demand_pct" in status
        assert "burner_count" in status
        assert "burner_statuses" in status
        assert "role_assignments" in status
        assert "safety_status" in status

        assert status["burner_count"] == 4


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Tests for configuration validation."""

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = OrchestrationConfig()

        assert config.num_burners == 2
        assert config.redundancy_mode is True
        assert config.coordination_strategy == CoordinationStrategy.SEQUENTIAL
        assert config.load_balancing_strategy == LoadBalancingStrategy.EQUAL

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = OrchestrationConfig(
            num_burners=4,
            start_delay_between_burners_s=30.0,
            prepurge_duration_s=60.0,
        )
        assert config.num_burners == 4

    def test_config_invalid_num_burners(self):
        """Test that invalid num_burners raises error."""
        with pytest.raises(ValueError):
            OrchestrationConfig(num_burners=0)

        with pytest.raises(ValueError):
            OrchestrationConfig(num_burners=25)

    def test_config_with_callbacks(self, default_config):
        """Test configuration with callbacks."""
        interlock_cb = Mock(return_value=True)
        command_cb = Mock(return_value=True)

        orch = MultiBurnerOrchestrator(
            default_config,
            interlock_callback=interlock_cb,
            command_callback=command_cb,
        )

        assert orch._interlock_callback is interlock_cb
        assert orch._command_callback is command_cb


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_start_with_burner_in_lockout(self, orchestrator):
        """Test that burners in lockout are skipped during start."""
        # Put one burner in lockout
        orchestrator.state.burner_statuses["BRN-002"].in_lockout = True

        for bid in ["BRN-001", "BRN-003"]:
            orchestrator.state.burner_statuses[bid].interlocks_satisfied = True
            orchestrator.state.burner_statuses[bid].flame_proven = True

        # Verify lockout burner not included
        burners_for_load = orchestrator._get_burners_for_load(50.0)
        assert "BRN-002" not in burners_for_load

    def test_load_distribution_empty_active(self, orchestrator):
        """Test load distribution with no active burners."""
        distribution = orchestrator.calculate_load_distribution()

        assert len(distribution.active_burners) == 0
        assert len(distribution.burner_loads) == 0

    @pytest.mark.asyncio
    async def test_rotation_with_insufficient_burners(self):
        """Test rotation with only one active burner."""
        config = OrchestrationConfig(num_burners=1, redundancy_mode=False)
        orch = MultiBurnerOrchestrator(config)
        orch.initialize_burners(["BRN-001"])

        success, assignments = await orch.execute_lead_lag_rotation()

        # Should fail gracefully with only one burner
        assert success is False

    def test_event_log_trimming(self, orchestrator):
        """Test that event log is trimmed when too long."""
        # Add many events
        for i in range(1500):
            orchestrator._log_event(
                SequencePhase.IDLE,
                f"test_event_{i}",
            )

        # Should be trimmed to 500
        assert len(orchestrator.state.event_log) <= 1000

    def test_provenance_hash_computation(self, orchestrator):
        """Test that provenance hashes are computed correctly."""
        status = orchestrator.state.burner_statuses["BRN-001"]

        assert status.provenance_hash != ""
        assert len(status.provenance_hash) == 16


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================

class TestIntegrationScenarios:
    """Integration-style tests for complete scenarios."""

    @pytest.mark.asyncio
    async def test_full_operation_cycle(self, orchestrator):
        """Test complete operation cycle: start -> modulate -> stop."""
        # Setup
        for status in orchestrator.state.burner_statuses.values():
            status.interlocks_satisfied = True
            status.flame_proven = True

        # Start
        success, _ = await orchestrator.execute_coordinated_start(target_load_pct=50.0)
        assert success is True

        # Modulate load
        distribution = orchestrator.set_load_demand(75.0)
        assert orchestrator.state.total_demand_pct == 75.0

        # Stop
        success, _ = await orchestrator.execute_coordinated_stop()
        assert success is True

        # Verify final state
        assert orchestrator.state.total_demand_pct == 0.0
        for status in orchestrator.state.burner_statuses.values():
            if status.role != BurnerRole.STANDBY_RESERVE:
                assert status.state == BurnerState.STANDBY

    @pytest.mark.asyncio
    async def test_failover_then_rotation(self, orchestrator):
        """Test failover followed by rotation."""
        # Setup active burner
        orchestrator.state.burner_statuses["BRN-001"].state = BurnerState.MODULATING
        orchestrator.state.burner_statuses["BRN-001"].flame_proven = True
        orchestrator.state.burner_statuses["BRN-001"].interlocks_satisfied = True

        orchestrator.state.burner_statuses["BRN-004"].interlocks_satisfied = True
        orchestrator.state.burner_statuses["BRN-004"].flame_proven = True

        # Execute failover
        failover_success, replacement = await orchestrator.execute_failover("BRN-001")
        assert failover_success is True

        # Execute rotation
        rotation_success, new_roles = await orchestrator.execute_lead_lag_rotation()
        assert rotation_success is True

        # Verify roles changed
        assert orchestrator.state.role_assignments["BRN-001"] == BurnerRole.MAINTENANCE


# =============================================================================
# DETERMINISTIC BEHAVIOR TESTS
# =============================================================================

class TestDeterministicBehavior:
    """Tests ensuring deterministic behavior for safety."""

    def test_load_calculation_deterministic(self, two_burner_orchestrator):
        """Test that load calculation is deterministic."""
        orch = two_burner_orchestrator

        for status in orch.state.burner_statuses.values():
            status.state = BurnerState.MODULATING
            status.efficiency_pct = 85.0

        orch.state.total_demand_pct = 80.0

        # Calculate twice
        dist1 = orch.calculate_load_distribution()
        dist2 = orch.calculate_load_distribution()

        # Should be identical
        assert dist1.burner_loads == dist2.burner_loads

    def test_stop_order_deterministic(self, orchestrator):
        """Test that stop order is deterministic."""
        for status in orchestrator.state.burner_statuses.values():
            status.state = BurnerState.MODULATING

        active = orchestrator._get_active_burners()

        # Get stop order twice
        order1 = orchestrator._get_stop_order(active)
        order2 = orchestrator._get_stop_order(active)

        assert order1 == order2

    def test_rotation_calculation_deterministic(self, orchestrator):
        """Test that rotation calculation is deterministic."""
        current_roles = {
            "BRN-001": BurnerRole.LEAD,
            "BRN-002": BurnerRole.LAG_1,
            "BRN-003": BurnerRole.LAG_2,
        }

        # Calculate twice
        new1 = orchestrator._calculate_rotation(current_roles, "BRN-001")
        new2 = orchestrator._calculate_rotation(current_roles, "BRN-001")

        assert new1 == new2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
