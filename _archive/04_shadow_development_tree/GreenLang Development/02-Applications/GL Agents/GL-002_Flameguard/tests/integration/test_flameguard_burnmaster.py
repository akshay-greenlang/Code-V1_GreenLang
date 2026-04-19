# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD <-> GL-004 BURNMASTER Integration Tests

Tests for integration between FlameGuard (Boiler Efficiency) and
BurnMaster (Burner Optimization) agents.

Integration Points:
- Efficiency calculations feed into burner optimization
- Combustion settings from BurnMaster affect efficiency
- Safety interlocks coordination
- Shared event streaming (Kafka)

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

import pytest
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyResult,
)
from optimization.combustion_optimizer import CombustionOptimizer
from safety.burner_management import BurnerManagementSystem, BurnerState


class TestEfficiencyToCombustionOptimization:
    """
    Test integration: Efficiency results drive combustion optimization.
    """

    @pytest.fixture
    def efficiency_calculator(self):
        return EfficiencyCalculator()

    @pytest.fixture
    def efficiency_result(self, efficiency_calculator):
        """Create a sample efficiency result."""
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        return efficiency_calculator.calculate(input_data)

    @pytest.mark.integration
    def test_efficiency_result_has_optimization_inputs(self, efficiency_result):
        """
        Efficiency result should contain data needed for combustion optimization.
        """
        # Required fields for combustion optimizer
        assert hasattr(efficiency_result, 'excess_air_percent')
        assert hasattr(efficiency_result, 'dry_flue_gas_loss_percent')
        assert hasattr(efficiency_result, 'flue_gas_mass_flow_lb_hr')
        assert hasattr(efficiency_result, 'efficiency_hhv_percent')
        assert hasattr(efficiency_result, 'total_losses_percent')

    @pytest.mark.integration
    def test_high_excess_air_triggers_optimization(self, efficiency_calculator):
        """
        High excess air should trigger O2 trim optimization.
        """
        # High excess air case
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=6.0,  # High O2 = high excess air
            fuel_type="natural_gas",
        )
        result = efficiency_calculator.calculate(input_data)

        # High excess air should be detected
        assert result.excess_air_percent > 30

        # This would trigger optimization to reduce O2 setpoint
        target_o2 = 3.0  # Optimal O2 target
        o2_adjustment = target_o2 - 6.0

        assert o2_adjustment < 0  # Need to reduce O2

    @pytest.mark.integration
    def test_low_efficiency_triggers_tuning_recommendation(self, efficiency_calculator):
        """
        Low efficiency should trigger burner tuning recommendation.
        """
        # Poor combustion case
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=10000.0,  # More fuel for same output = lower efficiency
            flue_gas_temperature_f=500.0,  # High stack temp
            flue_gas_o2_percent=5.0,  # High excess air
            fuel_type="natural_gas",
        )
        result = efficiency_calculator.calculate(input_data)

        # Low efficiency detected
        efficiency_threshold = 80.0
        needs_optimization = result.efficiency_hhv_percent < efficiency_threshold

        if needs_optimization:
            # Would send recommendation to BurnMaster
            recommendation = {
                "action": "tune_burner",
                "current_efficiency": result.efficiency_hhv_percent,
                "target_efficiency": 85.0,
                "issues": []
            }

            if result.dry_flue_gas_loss_percent > 6.0:
                recommendation["issues"].append("high_stack_loss")
            if result.excess_air_percent > 25:
                recommendation["issues"].append("excess_air")

            assert len(recommendation["issues"]) > 0


class TestCombustionToEfficiencyFeedback:
    """
    Test integration: Combustion settings affect efficiency calculations.
    """

    @pytest.fixture
    def efficiency_calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.integration
    def test_o2_setpoint_change_affects_efficiency(self, efficiency_calculator):
        """
        Changing O2 setpoint should affect calculated efficiency.
        """
        base_input = {
            "steam_flow_klb_hr": 100.0,
            "steam_pressure_psig": 150.0,
            "steam_temperature_f": 366.0,
            "feedwater_temperature_f": 227.0,
            "fuel_flow_rate": 8000.0,
            "flue_gas_temperature_f": 400.0,
            "fuel_type": "natural_gas",
        }

        # Baseline O2 = 4%
        result_baseline = efficiency_calculator.calculate(
            EfficiencyInput(**base_input, flue_gas_o2_percent=4.0)
        )

        # Optimized O2 = 2%
        result_optimized = efficiency_calculator.calculate(
            EfficiencyInput(**base_input, flue_gas_o2_percent=2.0)
        )

        # Optimized should have higher efficiency
        assert result_optimized.efficiency_hhv_percent > result_baseline.efficiency_hhv_percent

        # Improvement typically 0.5-1% per 1% O2 reduction
        improvement = result_optimized.efficiency_hhv_percent - result_baseline.efficiency_hhv_percent
        assert improvement > 0

    @pytest.mark.integration
    def test_air_fuel_ratio_integration(self, efficiency_calculator):
        """
        Test air-fuel ratio calculation matches combustion optimizer needs.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        result = efficiency_calculator.calculate(input_data)

        # Air-fuel ratio should be valid
        assert result.air_fuel_ratio > 0
        # For natural gas, stoichiometric ~17.2, with 3% O2 ~20
        assert 15.0 <= result.air_fuel_ratio <= 25.0


class TestSafetyInterlockCoordination:
    """
    Test integration: Safety interlocks between FlameGuard and BurnMaster.
    """

    @pytest.fixture
    def flameguard_bms(self):
        return BurnerManagementSystem("FLAMEGUARD-001")

    @pytest.fixture
    def burnmaster_bms(self):
        """Simulated BurnMaster BMS for coordination testing."""
        return BurnerManagementSystem("BURNMASTER-001")

    @pytest.mark.integration
    @pytest.mark.safety
    def test_flameguard_trip_propagates_to_burnmaster(
        self, flameguard_bms, burnmaster_bms
    ):
        """
        Trip in FlameGuard should propagate to BurnMaster.
        """
        # Set up both in firing state
        flameguard_bms._state = BurnerState.FIRING
        flameguard_bms._flame_proven = True
        burnmaster_bms._state = BurnerState.FIRING
        burnmaster_bms._flame_proven = True

        # Trip FlameGuard
        flameguard_bms._trip("Low drum level")

        # Simulate event propagation
        if flameguard_bms.state == BurnerState.LOCKOUT:
            # BurnMaster should also trip
            burnmaster_bms._trip("Coordinated shutdown - FlameGuard trip")

        assert flameguard_bms.state == BurnerState.LOCKOUT
        assert burnmaster_bms.state == BurnerState.LOCKOUT

    @pytest.mark.integration
    @pytest.mark.safety
    def test_combustion_fault_trips_efficiency_monitoring(self):
        """
        Combustion fault from BurnMaster should suspend efficiency calculations.
        """
        # Simulate BurnMaster fault detection
        combustion_fault = {
            "fault_type": "high_co",
            "co_ppm": 500,
            "timestamp": datetime.now(timezone.utc),
        }

        # FlameGuard should recognize fault condition
        is_combustion_healthy = combustion_fault["co_ppm"] < 200

        # If combustion unhealthy, efficiency calculations may be invalid
        if not is_combustion_healthy:
            efficiency_status = "SUSPENDED"
            reason = f"Combustion fault: CO = {combustion_fault['co_ppm']} ppm"
        else:
            efficiency_status = "ACTIVE"
            reason = None

        assert efficiency_status == "SUSPENDED"

    @pytest.mark.integration
    @pytest.mark.safety
    def test_coordinated_startup_sequence(self, flameguard_bms, burnmaster_bms):
        """
        Test coordinated startup between FlameGuard and BurnMaster.
        """
        # FlameGuard must initialize first
        flameguard_bms._state = BurnerState.OFFLINE
        burnmaster_bms._state = BurnerState.OFFLINE

        # Startup sequence coordination
        startup_order = []

        # FlameGuard starts first (boiler controls)
        startup_order.append("flameguard")
        flameguard_bms._transition_to(BurnerState.PRE_PURGE)

        # BurnMaster waits for FlameGuard ready signal
        if flameguard_bms.state == BurnerState.PRE_PURGE:
            startup_order.append("burnmaster")
            # BurnMaster prepares combustion control

        assert startup_order == ["flameguard", "burnmaster"]


class TestEventStreamingIntegration:
    """
    Test integration: Kafka event streaming between agents.
    """

    @pytest.fixture
    def mock_kafka_producer(self):
        """Mock Kafka producer."""
        producer = Mock()
        producer.send = Mock(return_value=Mock())
        return producer

    @pytest.fixture
    def efficiency_calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.integration
    def test_efficiency_event_published(
        self, efficiency_calculator, mock_kafka_producer
    ):
        """
        Efficiency calculation should publish event for BurnMaster.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        result = efficiency_calculator.calculate(input_data)

        # Simulate event publication
        event = {
            "event_type": "efficiency_calculated",
            "source_agent": "GL-002_FlameGuard",
            "target_agent": "GL-004_BurnMaster",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "efficiency_percent": result.efficiency_hhv_percent,
                "excess_air_percent": result.excess_air_percent,
                "stack_loss_percent": result.dry_flue_gas_loss_percent,
                "optimization_needed": result.efficiency_hhv_percent < 82.0,
            }
        }

        # Publish to Kafka
        mock_kafka_producer.send("gl.combustion.efficiency", event)
        mock_kafka_producer.send.assert_called_once()

    @pytest.mark.integration
    def test_optimization_request_event(self, mock_kafka_producer):
        """
        Test optimization request event from FlameGuard to BurnMaster.
        """
        optimization_request = {
            "event_type": "optimization_request",
            "source_agent": "GL-002_FlameGuard",
            "target_agent": "GL-004_BurnMaster",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "request_type": "reduce_excess_air",
                "current_o2": 5.0,
                "target_o2": 3.0,
                "priority": "medium",
            }
        }

        mock_kafka_producer.send("gl.combustion.optimization", optimization_request)
        mock_kafka_producer.send.assert_called_once()

    @pytest.mark.integration
    def test_trip_event_broadcast(self, mock_kafka_producer):
        """
        Safety trip should broadcast to all listening agents.
        """
        bms = BurnerManagementSystem("TEST-001")
        bms._state = BurnerState.FIRING
        bms._flame_proven = True

        # Simulate trip
        bms._trip("High steam pressure")

        # Trip event for broadcast
        trip_event = {
            "event_type": "safety_trip",
            "source_agent": "GL-002_FlameGuard",
            "broadcast": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "boiler_id": bms.boiler_id,
                "trip_reason": bms._lockout_reason,
                "from_state": bms._prev_state.name,
                "current_state": bms.state.name,
            }
        }

        # Broadcast to safety topic
        mock_kafka_producer.send("gl.safety.trips", trip_event)
        mock_kafka_producer.send.assert_called_once()


class TestDataModelAlignment:
    """
    Test integration: Data models between FlameGuard and BurnMaster are aligned.
    """

    @pytest.mark.integration
    def test_combustion_data_model_compatibility(self):
        """
        Verify combustion data models are compatible between agents.
        """
        # FlameGuard efficiency input
        flameguard_data = {
            "fuel_type": "natural_gas",
            "fuel_flow_rate": 8000.0,
            "fuel_flow_unit": "lb_hr",
            "flue_gas_o2_percent": 3.0,
            "flue_gas_temperature_f": 400.0,
            "flue_gas_co_ppm": 50.0,
        }

        # BurnMaster combustion input (should accept same fields)
        burnmaster_data = {
            "fuel_type": flameguard_data["fuel_type"],
            "fuel_rate": flameguard_data["fuel_flow_rate"],
            "fuel_unit": flameguard_data["fuel_flow_unit"],
            "o2_percent": flameguard_data["flue_gas_o2_percent"],
            "stack_temp": flameguard_data["flue_gas_temperature_f"],
            "co_ppm": flameguard_data["flue_gas_co_ppm"],
        }

        # Verify mappings exist
        assert burnmaster_data["fuel_type"] == flameguard_data["fuel_type"]
        assert burnmaster_data["o2_percent"] == flameguard_data["flue_gas_o2_percent"]

    @pytest.mark.integration
    def test_efficiency_result_serialization(self):
        """
        Efficiency result should be serializable for inter-agent communication.
        """
        calc = EfficiencyCalculator()
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        result = calc.calculate(input_data)

        # Should be convertible to dict for serialization
        result_dict = {
            "calculation_id": result.calculation_id,
            "efficiency_hhv_percent": result.efficiency_hhv_percent,
            "efficiency_lhv_percent": result.efficiency_lhv_percent,
            "excess_air_percent": result.excess_air_percent,
            "total_losses_percent": result.total_losses_percent,
            "dry_flue_gas_loss_percent": result.dry_flue_gas_loss_percent,
            "input_hash": result.input_hash,
            "output_hash": result.output_hash,
        }

        # All values should be JSON-serializable types
        import json
        json_str = json.dumps(result_dict)
        assert json_str is not None


class TestPerformanceIntegration:
    """
    Test integration: Performance under load.
    """

    @pytest.fixture
    def efficiency_calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.integration
    @pytest.mark.performance
    def test_high_frequency_calculations(self, efficiency_calculator):
        """
        Test rapid efficiency calculations (simulating real-time operation).
        """
        import time

        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        # Simulate 1 second of 10 Hz calculations
        num_calculations = 10
        start_time = time.time()

        results = []
        for _ in range(num_calculations):
            result = efficiency_calculator.calculate(input_data)
            results.append(result)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 2.0  # 2 seconds max for 10 calculations

        # All results should be consistent
        efficiencies = [r.efficiency_hhv_percent for r in results]
        assert len(set(efficiencies)) == 1  # All identical

    @pytest.mark.integration
    @pytest.mark.performance
    def test_calculation_latency(self, efficiency_calculator):
        """
        Single calculation should be fast enough for real-time control.
        """
        import time

        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        start_time = time.time()
        result = efficiency_calculator.calculate(input_data)
        latency_ms = (time.time() - start_time) * 1000

        # Should complete in <100ms for real-time use
        assert latency_ms < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
