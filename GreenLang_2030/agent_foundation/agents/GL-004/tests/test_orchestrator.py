"""
GL-004 BurnerOptimizationAgent - Orchestrator Tests

Unit tests for the main burner optimization orchestrator.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from burner_optimization_orchestrator import (
    BurnerOptimizationOrchestrator,
    BurnerState,
    OptimizationResult,
    SafetyInterlocks
)


class TestBurnerOptimizationOrchestrator:
    """Test suite for BurnerOptimizationOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return BurnerOptimizationOrchestrator()

    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        assert orchestrator.agent_id == "GL-004"
        assert orchestrator.agent_name == "BurnerOptimizationAgent"
        assert orchestrator.version == "1.0.0"
        assert orchestrator.is_running is False
        assert len(orchestrator.optimization_history) == 0

    @pytest.mark.asyncio
    async def test_collect_burner_state(self, orchestrator):
        """Test burner state collection"""
        # Mock integrations
        orchestrator.burner_controller = AsyncMock()
        orchestrator.o2_analyzer = AsyncMock()
        orchestrator.emissions_monitor = AsyncMock()
        orchestrator.temperature_sensors = AsyncMock()

        # Set up mock return values
        orchestrator.burner_controller.get_fuel_flow_rate = AsyncMock(return_value=500.0)
        orchestrator.burner_controller.get_air_flow_rate = AsyncMock(return_value=8500.0)
        orchestrator.burner_controller.get_burner_load = AsyncMock(return_value=75.0)
        orchestrator.o2_analyzer.get_o2_concentration = AsyncMock(return_value=3.5)
        orchestrator.emissions_monitor.get_emissions_data = AsyncMock(
            return_value={'CO': 25.0, 'NOx': 35.0}
        )
        orchestrator.temperature_sensors.get_flame_temperature = AsyncMock(return_value=1650.0)
        orchestrator.temperature_sensors.get_furnace_temperature = AsyncMock(return_value=1200.0)
        orchestrator.temperature_sensors.get_flue_gas_temperature = AsyncMock(return_value=320.0)

        # Collect state
        state = await orchestrator.collect_burner_state()

        # Assertions
        assert isinstance(state, BurnerState)
        assert state.fuel_flow_rate == 500.0
        assert state.air_flow_rate == 8500.0
        assert state.air_fuel_ratio == 17.0  # 8500/500
        assert state.o2_level == 3.5
        assert state.co_level == 25.0
        assert state.nox_level == 35.0
        assert state.burner_load == 75.0

    @pytest.mark.asyncio
    async def test_check_safety_interlocks(self, orchestrator):
        """Test safety interlock checking"""
        # Mock burner controller and flame scanner
        orchestrator.burner_controller = AsyncMock()
        orchestrator.flame_scanner = AsyncMock()

        # All interlocks OK
        orchestrator.flame_scanner.is_flame_present = AsyncMock(return_value=True)
        orchestrator.burner_controller.check_fuel_pressure = AsyncMock(return_value=True)
        orchestrator.burner_controller.check_air_pressure = AsyncMock(return_value=True)
        orchestrator.burner_controller.is_purge_complete = AsyncMock(return_value=True)
        orchestrator.burner_controller.check_temperature_limits = AsyncMock(return_value=True)
        orchestrator.burner_controller.is_emergency_stop_clear = AsyncMock(return_value=True)

        interlocks = await orchestrator.check_safety_interlocks()

        assert isinstance(interlocks, SafetyInterlocks)
        assert interlocks.all_safe() is True

    @pytest.mark.asyncio
    async def test_check_safety_interlocks_flame_loss(self, orchestrator):
        """Test safety interlocks with flame loss"""
        orchestrator.burner_controller = AsyncMock()
        orchestrator.flame_scanner = AsyncMock()

        # Flame lost
        orchestrator.flame_scanner.is_flame_present = AsyncMock(return_value=False)
        orchestrator.burner_controller.check_fuel_pressure = AsyncMock(return_value=True)
        orchestrator.burner_controller.check_air_pressure = AsyncMock(return_value=True)
        orchestrator.burner_controller.is_purge_complete = AsyncMock(return_value=True)
        orchestrator.burner_controller.check_temperature_limits = AsyncMock(return_value=True)
        orchestrator.burner_controller.is_emergency_stop_clear = AsyncMock(return_value=True)

        interlocks = await orchestrator.check_safety_interlocks()

        assert interlocks.all_safe() is False
        assert interlocks.flame_present is False

    def test_burner_state_validation(self):
        """Test BurnerState validation"""
        # Valid state
        state = BurnerState(
            fuel_flow_rate=500.0,
            air_flow_rate=8500.0,
            air_fuel_ratio=17.0,
            o2_level=3.5,
            furnace_temperature=1200.0,
            flue_gas_temperature=320.0,
            burner_load=75.0
        )
        assert state.o2_level == 3.5

        # Invalid O2 (too high)
        with pytest.raises(ValueError):
            BurnerState(
                fuel_flow_rate=500.0,
                air_flow_rate=8500.0,
                air_fuel_ratio=17.0,
                o2_level=25.0,  # > 21%
                furnace_temperature=1200.0,
                flue_gas_temperature=320.0,
                burner_load=75.0
            )

    def test_optimization_result_hash(self):
        """Test OptimizationResult hash calculation for determinism"""
        result1 = OptimizationResult(
            current_air_fuel_ratio=18.0,
            current_efficiency=87.5,
            current_nox=45.0,
            current_co=30.0,
            optimal_air_fuel_ratio=17.2,
            optimal_fuel_flow=490.0,
            optimal_air_flow=8428.0,
            optimal_excess_air=15.0,
            predicted_efficiency=89.5,
            predicted_nox=38.0,
            predicted_co=20.0,
            efficiency_improvement=2.0,
            nox_reduction=15.6,
            co_reduction=33.3,
            fuel_savings=10.0,
            iterations=50,
            convergence_status="converged",
            confidence_score=0.85,
            hash=""
        )

        result1.hash = result1.calculate_hash()

        # Same data should produce same hash
        result2 = OptimizationResult(
            current_air_fuel_ratio=18.0,
            current_efficiency=87.5,
            current_nox=45.0,
            current_co=30.0,
            optimal_air_fuel_ratio=17.2,
            optimal_fuel_flow=490.0,
            optimal_air_flow=8428.0,
            optimal_excess_air=15.0,
            predicted_efficiency=89.5,
            predicted_nox=38.0,
            predicted_co=20.0,
            efficiency_improvement=2.0,
            nox_reduction=15.6,
            co_reduction=33.3,
            fuel_savings=10.0,
            iterations=50,
            convergence_status="converged",
            confidence_score=0.85,
            hash=""
        )

        result2.hash = result2.calculate_hash()

        assert result1.hash == result2.hash

    def test_get_status(self, orchestrator):
        """Test get_status method"""
        status = orchestrator.get_status()

        assert status['agent_id'] == "GL-004"
        assert status['agent_name'] == "BurnerOptimizationAgent"
        assert status['version'] == "1.0.0"
        assert status['is_running'] is False
        assert status['optimization_count'] == 0


class TestSafetyInterlocks:
    """Test suite for SafetyInterlocks"""

    def test_all_safe_true(self):
        """Test all_safe when all interlocks satisfied"""
        interlocks = SafetyInterlocks(
            flame_present=True,
            fuel_pressure_ok=True,
            air_pressure_ok=True,
            purge_complete=True,
            temperature_ok=True,
            emergency_stop_clear=True
        )
        assert interlocks.all_safe() is True

    def test_all_safe_false(self):
        """Test all_safe when any interlock fails"""
        interlocks = SafetyInterlocks(
            flame_present=True,
            fuel_pressure_ok=False,  # Failed
            air_pressure_ok=True,
            purge_complete=True,
            temperature_ok=True,
            emergency_stop_clear=True
        )
        assert interlocks.all_safe() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
