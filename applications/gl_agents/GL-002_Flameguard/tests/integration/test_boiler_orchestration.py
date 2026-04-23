"""
GL-002 FLAMEGUARD - Boiler Orchestration Integration Tests

End-to-end tests for the full boiler control loop including:
- Data acquisition
- Safety monitoring
- Optimization cycles
- Efficiency calculations
- Event handling
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_flameguard_config():
    """Create a mock FlameguardConfig."""
    config = MagicMock()
    config.agent_id = "GL-002-TEST"
    config.version = "1.0.0"

    # Boiler specs
    config.boiler = MagicMock()
    config.boiler.boiler_id = "BOILER-001"
    config.boiler.boiler_name = "Test Boiler"
    config.boiler.blowdown_rate_percent = 3.0

    # Fuel config
    config.fuel = MagicMock()
    config.fuel.primary_fuel = MagicMock()
    config.fuel.primary_fuel.fuel_type = MagicMock()
    config.fuel.primary_fuel.fuel_type.value = "natural_gas"
    config.fuel.primary_fuel.higher_heating_value_btu_lb = 23875.0
    config.fuel.primary_fuel.lower_heating_value_btu_lb = 21500.0
    config.fuel.primary_fuel.stoichiometric_air_fuel_ratio = 17.2
    config.fuel.primary_fuel.hydrogen_content_percent = 25.0
    config.fuel.primary_fuel.moisture_content_percent = 0.0
    config.fuel.primary_fuel.ash_content_percent = 0.0

    # Combustion config
    config.combustion = MagicMock()
    config.combustion.o2_trim = MagicMock()
    config.combustion.o2_trim.target_o2_percent = 3.0
    config.combustion.o2_trim.o2_setpoint_curve = {0.25: 5.0, 0.50: 3.5, 0.75: 3.0, 1.0: 2.5}
    config.combustion.o2_trim.deadband_percent = 0.2
    config.combustion.excess_air = MagicMock()
    config.combustion.excess_air.design_excess_air_percent = 15.0
    config.combustion.excess_air.excess_air_curve = {0.25: 25.0, 0.50: 18.0, 0.75: 15.0, 1.0: 12.0}
    config.combustion.co_monitoring = MagicMock()
    config.combustion.co_monitoring.co_limit_ppm = 400.0
    config.combustion.co_monitoring.co_alarm_ppm = 300.0
    config.combustion.co_monitoring.breakthrough_threshold_ppm = 200.0

    # Safety config
    config.safety = MagicMock()
    config.safety.interlocks = MagicMock()
    config.safety.interlocks.high_steam_pressure_psig = 150.0
    config.safety.interlocks.low_water_level_inches = -4.0
    config.safety.interlocks.high_flue_gas_temp_f = 700.0

    # Efficiency config
    config.efficiency = MagicMock()
    config.efficiency.design_efficiency_percent = 82.0

    # Optimization config
    config.optimization = MagicMock()
    config.optimization.max_optimization_cycles_per_hour = 12
    config.optimization.setpoints = MagicMock()
    config.optimization.setpoints.require_operator_approval = False

    # Metrics config
    config.metrics = MagicMock()
    config.metrics.collection_interval_s = 15

    return config


@pytest.fixture
def mock_process_data():
    """Create mock boiler process data."""
    data = MagicMock()
    data.boiler_id = "BOILER-001"
    data.timestamp = datetime.now(timezone.utc)
    data.operating_state = MagicMock()
    data.operating_state.value = "modulating"
    data.load_percent = 75.0
    data.data_quality = 1.0
    data.steam_flow_klb_hr = 150.0
    data.steam_pressure_psig = 125.0
    data.steam_temperature_f = 450.0
    data.feedwater_temperature_f = 220.0
    data.drum_level_inches = 0.5
    data.flue_gas_temperature_f = 350.0
    data.flue_gas_o2_percent = 3.5
    data.flue_gas_co_ppm = 25.0
    data.flue_gas_co2_percent = 10.5
    data.fuel_flow_rate = 25000.0
    data.air_damper_position_percent = 65.0
    data.ambient_temperature_f = 70.0
    data.combustion_air_humidity_percent = 50.0
    data.flame_status = True
    data.flame_signal_percent = 85.0

    # For dict() call
    data.dict = Mock(return_value={
        "boiler_id": "BOILER-001",
        "load_percent": 75.0,
        "flue_gas_o2_percent": 3.5,
    })

    return data


# =============================================================================
# CONTROL LOOP TESTS
# =============================================================================


class TestControlLoopBasic:
    """Test basic control loop operations."""

    @pytest.mark.asyncio
    async def test_register_and_unregister_boiler(self, mock_flameguard_config):
        """Test boiler registration and unregistration."""
        # Import with mocked core module
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # Register boiler
            orchestrator.register_boiler("BOILER-001")
            assert "BOILER-001" in orchestrator._boiler_configs

            # Unregister boiler
            orchestrator.unregister_boiler("BOILER-001")
            assert "BOILER-001" not in orchestrator._boiler_configs

    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_flameguard_config):
        """Test orchestrator start and stop."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # Start
            await orchestrator.start()
            assert orchestrator._running is True
            assert len(orchestrator._tasks) > 0

            # Stop
            await orchestrator.stop()
            assert orchestrator._running is False
            assert len(orchestrator._tasks) == 0

    @pytest.mark.asyncio
    async def test_double_start(self, mock_flameguard_config):
        """Test starting already running orchestrator."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            await orchestrator.start()
            await orchestrator.start()  # Should not create duplicate tasks

            # Still should have same number of tasks
            assert orchestrator._running is True

            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stop_not_running(self, mock_flameguard_config):
        """Test stopping non-running orchestrator."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # Should not raise exception
            await orchestrator.stop()


class TestDataAcquisition:
    """Test data acquisition from SCADA/DCS."""

    @pytest.mark.asyncio
    async def test_process_data_update(self, mock_flameguard_config, mock_process_data):
        """Test process data update."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            # Mock the core module with required classes
            core_mock = MagicMock()
            core_mock.OperatingState = MagicMock()
            core_mock.OperatingState.OFFLINE = "offline"
            core_mock.OperatingState.COLD_STANDBY = "cold_standby"
            core_mock.OperatingState.PURGING = "purging"
            core_mock.OperatingState.MODULATING = "modulating"

            with patch.dict('sys.modules', {'core': core_mock}):
                from boiler_efficiency_orchestrator import FlameGuardOrchestrator

                orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
                orchestrator.register_boiler("BOILER-001")

                # Mock event handlers
                orchestrator._emit_event = AsyncMock()
                orchestrator._check_safety_conditions = AsyncMock()
                orchestrator._update_combustion_analysis = AsyncMock()

                await orchestrator.update_process_data(mock_process_data)

                # Data should be stored
                assert "BOILER-001" in orchestrator._boilers


class TestOptimizationCycle:
    """Test optimization cycle operations."""

    @pytest.mark.asyncio
    async def test_optimization_request(self, mock_flameguard_config, mock_process_data):
        """Test optimization request processing."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            core_mock = MagicMock()
            core_mock.OptimizationStatus = MagicMock()
            core_mock.OptimizationStatus.COMPLETED = "completed"
            core_mock.OptimizationStatus.FAILED = "failed"
            core_mock.OptimizationRequest = MagicMock
            core_mock.OptimizationResult = MagicMock
            core_mock.EfficiencyCalculation = MagicMock
            core_mock.CombustionAnalysis = MagicMock
            core_mock.SeverityLevel = MagicMock()
            core_mock.SeverityLevel.WARNING = "warning"

            with patch.dict('sys.modules', {'core': core_mock}):
                from boiler_efficiency_orchestrator import FlameGuardOrchestrator

                orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
                orchestrator.register_boiler("BOILER-001")
                orchestrator._boilers["BOILER-001"] = mock_process_data

                # Mock methods
                orchestrator._emit_event = AsyncMock()
                orchestrator._run_optimization = AsyncMock(return_value=MagicMock(
                    status=core_mock.OptimizationStatus.COMPLETED,
                    efficiency_improvement_percent=2.5,
                    cost_savings_hr=50.0,
                    optimization_id="OPT-001",
                ))

                # Request optimization
                result = await orchestrator.optimize("BOILER-001")

                assert result is not None
                orchestrator._run_optimization.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimization_without_data(self, mock_flameguard_config):
        """Test optimization fails without process data."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            core_mock = MagicMock()
            core_mock.OptimizationStatus = MagicMock()
            core_mock.OptimizationStatus.FAILED = "failed"
            core_mock.OptimizationRequest = MagicMock

            with patch.dict('sys.modules', {'core': core_mock}):
                from boiler_efficiency_orchestrator import FlameGuardOrchestrator

                orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
                orchestrator.register_boiler("BOILER-001")
                orchestrator._emit_event = AsyncMock()

                # No process data - should fail
                result = await orchestrator.optimize("BOILER-001")

                # Should return failed result
                assert result.status == core_mock.OptimizationStatus.FAILED


class TestEfficiencyCalculation:
    """Test efficiency calculation integration."""

    def test_o2_to_excess_air_calculation(self, mock_flameguard_config):
        """Test O2 to excess air conversion."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # Test conversion
            excess_air = orchestrator._o2_to_excess_air(3.0)

            # 3% O2 ~ 16.7% excess air
            assert 15.0 <= excess_air <= 20.0

    def test_o2_to_excess_air_edge_cases(self, mock_flameguard_config):
        """Test O2 to excess air edge cases."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # At 21% O2 (atmospheric), return max
            assert orchestrator._o2_to_excess_air(21.0) == 500.0

            # At 0% O2, return 0
            assert orchestrator._o2_to_excess_air(0.0) == 0.0


class TestStatusReporting:
    """Test status reporting functionality."""

    def test_get_status(self, mock_flameguard_config):
        """Test agent status retrieval."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
            status = orchestrator.get_status()

            assert status.agent_id == mock_flameguard_config.agent_id
            assert status.agent_type == "GL-002"

    def test_get_statistics(self, mock_flameguard_config):
        """Test statistics retrieval."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
            stats = orchestrator.get_statistics()

            assert "optimizations_performed" in stats
            assert "total_efficiency_improvement" in stats
            assert "managed_boilers" in stats

    def test_get_boiler_status(self, mock_flameguard_config):
        """Test boiler status retrieval."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
            orchestrator.register_boiler("BOILER-001")

            status = orchestrator.get_boiler_status("BOILER-001")

            assert status is not None
            assert status.boiler_id == "BOILER-001"

    def test_get_nonexistent_boiler_status(self, mock_flameguard_config):
        """Test getting status for nonexistent boiler."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            status = orchestrator.get_boiler_status("NONEXISTENT")

            assert status is None


# =============================================================================
# SAFETY INTEGRATION TESTS
# =============================================================================


class TestSafetyIntegration:
    """Test safety system integration."""

    @pytest.mark.asyncio
    async def test_high_pressure_event(self, mock_flameguard_config, mock_process_data):
        """Test high pressure generates safety event."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            core_mock = MagicMock()
            core_mock.OperatingState = MagicMock()
            core_mock.OperatingState.OFFLINE = "offline"
            core_mock.OperatingState.COLD_STANDBY = "cold_standby"
            core_mock.OperatingState.PURGING = "purging"
            core_mock.OperatingState.MODULATING = "modulating"
            core_mock.SeverityLevel = MagicMock()
            core_mock.SeverityLevel.CRITICAL = "critical"
            core_mock.SeverityLevel.EMERGENCY = "emergency"
            core_mock.SeverityLevel.WARNING = "warning"
            core_mock.FlameguardEvent = MagicMock

            with patch.dict('sys.modules', {'core': core_mock}):
                from boiler_efficiency_orchestrator import FlameGuardOrchestrator

                orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
                orchestrator.register_boiler("BOILER-001")

                # Set up high pressure
                mock_process_data.steam_pressure_psig = 160.0
                mock_process_data.operating_state = core_mock.OperatingState.MODULATING

                orchestrator._emit_event = AsyncMock()

                await orchestrator._check_safety_conditions(mock_process_data)

                # Should emit high pressure event
                orchestrator._emit_event.assert_called()

    @pytest.mark.asyncio
    async def test_low_water_level_event(self, mock_flameguard_config, mock_process_data):
        """Test low water level generates emergency event."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            core_mock = MagicMock()
            core_mock.OperatingState = MagicMock()
            core_mock.OperatingState.OFFLINE = "offline"
            core_mock.OperatingState.COLD_STANDBY = "cold_standby"
            core_mock.OperatingState.PURGING = "purging"
            core_mock.SeverityLevel = MagicMock()
            core_mock.SeverityLevel.EMERGENCY = "emergency"
            core_mock.FlameguardEvent = MagicMock

            with patch.dict('sys.modules', {'core': core_mock}):
                from boiler_efficiency_orchestrator import FlameGuardOrchestrator

                orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
                orchestrator.register_boiler("BOILER-001")

                # Set up low water
                mock_process_data.drum_level_inches = -5.0
                mock_process_data.operating_state = core_mock.OperatingState.OFFLINE

                orchestrator._emit_event = AsyncMock()

                await orchestrator._check_safety_conditions(mock_process_data)

                # Should emit low water event
                orchestrator._emit_event.assert_called()


class TestTripHandling:
    """Test trip handling functionality."""

    def test_trip_callback(self, mock_flameguard_config):
        """Test trip callback updates status."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            core_mock = MagicMock()
            core_mock.OperatingState = MagicMock()
            core_mock.OperatingState.EMERGENCY_SHUTDOWN = "emergency_shutdown"

            with patch.dict('sys.modules', {'core': core_mock}):
                from boiler_efficiency_orchestrator import FlameGuardOrchestrator

                orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
                orchestrator.register_boiler("BOILER-001")

                orchestrator._handle_trip("BOILER-001", "High pressure")

                # Should increment safety events
                assert orchestrator._stats["safety_events"] == 1

                # Should update boiler state
                boiler_status = orchestrator._boiler_statuses.get("BOILER-001")
                assert boiler_status.operating_state == core_mock.OperatingState.EMERGENCY_SHUTDOWN


# =============================================================================
# EVENT HANDLING TESTS
# =============================================================================


class TestEventHandling:
    """Test event emission and handling."""

    @pytest.mark.asyncio
    async def test_emit_event_to_handlers(self, mock_flameguard_config):
        """Test events are emitted to all handlers."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # Mock handlers
            for handler in [
                orchestrator._safety_handler,
                orchestrator._combustion_handler,
                orchestrator._efficiency_handler,
            ]:
                handler.handle = AsyncMock()

            event = MagicMock()
            await orchestrator._emit_event(event)

            # All handlers should be called
            orchestrator._safety_handler.handle.assert_called_once()
            orchestrator._combustion_handler.handle.assert_called_once()
            orchestrator._efficiency_handler.handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_error_isolated(self, mock_flameguard_config):
        """Test handler errors don't affect other handlers."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # First handler raises exception
            orchestrator._safety_handler.handle = AsyncMock(side_effect=Exception("Test error"))
            orchestrator._combustion_handler.handle = AsyncMock()
            orchestrator._efficiency_handler.handle = AsyncMock()

            event = MagicMock()
            await orchestrator._emit_event(event)

            # Other handlers should still be called
            orchestrator._combustion_handler.handle.assert_called_once()
            orchestrator._efficiency_handler.handle.assert_called_once()


# =============================================================================
# HASH COMPUTATION TESTS
# =============================================================================


class TestHashComputation:
    """Test provenance hash computation."""

    def test_compute_hash(self, mock_flameguard_config):
        """Test hash computation."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            data = {"key": "value", "number": 42}
            hash_val = orchestrator._compute_hash(data)

            assert len(hash_val) == 64  # SHA-256 hex

    def test_compute_hash_deterministic(self, mock_flameguard_config):
        """Test hash is deterministic."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            data = {"key": "value", "number": 42}
            hash1 = orchestrator._compute_hash(data)
            hash2 = orchestrator._compute_hash(data)

            assert hash1 == hash2

    def test_compute_hash_different_data(self, mock_flameguard_config):
        """Test different data produces different hash."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            hash1 = orchestrator._compute_hash({"value": 1})
            hash2 = orchestrator._compute_hash({"value": 2})

            assert hash1 != hash2


# =============================================================================
# BACKGROUND LOOP TESTS
# =============================================================================


class TestBackgroundLoops:
    """Test background monitoring loops."""

    @pytest.mark.asyncio
    async def test_optimization_loop_runs(self, mock_flameguard_config):
        """Test optimization loop runs."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)

            # Start and immediately stop
            await orchestrator.start()
            await asyncio.sleep(0.1)
            await orchestrator.stop()

            # Should have started tasks
            assert orchestrator._start_time is not None

    @pytest.mark.asyncio
    async def test_metrics_loop_runs(self, mock_flameguard_config, mock_process_data):
        """Test metrics collection loop."""
        with patch.dict('sys.modules', {'core': MagicMock()}):
            from boiler_efficiency_orchestrator import FlameGuardOrchestrator

            orchestrator = FlameGuardOrchestrator(mock_flameguard_config)
            orchestrator.register_boiler("BOILER-001")
            orchestrator._boilers["BOILER-001"] = mock_process_data
            orchestrator._emit_event = AsyncMock()

            # Run one iteration of metrics collection
            await orchestrator._emit_event(MagicMock())

            orchestrator._emit_event.assert_called()
