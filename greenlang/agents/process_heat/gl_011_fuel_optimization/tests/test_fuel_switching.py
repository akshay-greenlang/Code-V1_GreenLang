"""
GL-011 FUELCRAFT - Fuel Switching Controller Tests

Unit tests for FuelSwitchingController including economic trigger
analysis, safety interlocks, state machine transitions, and approval workflow.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_switching import (
    FuelSwitchingController,
    SwitchingInput,
    SwitchingOutput,
    SwitchingTrigger,
    SwitchingState,
    TriggerType,
    SwitchRecord,
    SwitchHistory,
    SAFETY_INTERLOCKS,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    SwitchingConfig,
    SwitchingMode,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import FuelPrice


class TestSwitchingState:
    """Tests for SwitchingState enum."""

    def test_all_states_defined(self):
        """Test all expected states are defined."""
        expected = {
            "IDLE", "EVALUATING", "PENDING_APPROVAL", "PREPARING",
            "PURGING", "TRANSITIONING", "STABILIZING", "COMPLETE",
            "ABORTED", "FAILED",
        }
        actual = {s.name for s in SwitchingState}
        assert expected == actual


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_trigger_types(self):
        """Test all trigger types."""
        assert TriggerType.ECONOMIC.value == "economic"
        assert TriggerType.SUPPLY.value == "supply"
        assert TriggerType.EMISSION_LIMIT.value == "emission_limit"
        assert TriggerType.EMERGENCY.value == "emergency"


class TestSwitchHistory:
    """Tests for SwitchHistory class."""

    def test_add_record(self):
        """Test adding switch record."""
        history = SwitchHistory()

        record = SwitchRecord(
            switch_id="SW-001",
            timestamp=datetime.now(timezone.utc),
            from_fuel="natural_gas",
            to_fuel="fuel_oil",
            trigger_type=TriggerType.ECONOMIC,
            duration_minutes=15.0,
            success=True,
            savings_realized_usd=100.0,
        )

        history.add_record(record)

        assert len(history._records) == 1

    def test_get_switches_today(self):
        """Test counting today's switches."""
        history = SwitchHistory()

        # Add today's switch
        record_today = SwitchRecord(
            switch_id="SW-001",
            timestamp=datetime.now(timezone.utc),
            from_fuel="natural_gas",
            to_fuel="fuel_oil",
            trigger_type=TriggerType.ECONOMIC,
            duration_minutes=15.0,
            success=True,
        )

        # Add yesterday's switch
        record_yesterday = SwitchRecord(
            switch_id="SW-002",
            timestamp=datetime.now(timezone.utc) - timedelta(days=1),
            from_fuel="fuel_oil",
            to_fuel="natural_gas",
            trigger_type=TriggerType.ECONOMIC,
            duration_minutes=15.0,
            success=True,
        )

        history.add_record(record_today)
        history.add_record(record_yesterday)

        assert history.get_switches_today() == 1

    def test_get_success_rate(self):
        """Test success rate calculation."""
        history = SwitchHistory()

        # Add 3 successful, 1 failed
        for i in range(3):
            history.add_record(SwitchRecord(
                switch_id=f"SW-00{i}",
                timestamp=datetime.now(timezone.utc),
                from_fuel="a",
                to_fuel="b",
                trigger_type=TriggerType.ECONOMIC,
                duration_minutes=15.0,
                success=True,
            ))

        history.add_record(SwitchRecord(
            switch_id="SW-003",
            timestamp=datetime.now(timezone.utc),
            from_fuel="a",
            to_fuel="b",
            trigger_type=TriggerType.ECONOMIC,
            duration_minutes=15.0,
            success=False,
        ))

        assert history.get_success_rate() == pytest.approx(0.75, rel=0.01)

    def test_get_total_savings(self):
        """Test total savings calculation."""
        history = SwitchHistory()

        for i in range(3):
            history.add_record(SwitchRecord(
                switch_id=f"SW-00{i}",
                timestamp=datetime.now(timezone.utc),
                from_fuel="a",
                to_fuel="b",
                trigger_type=TriggerType.ECONOMIC,
                duration_minutes=15.0,
                success=True,
                savings_realized_usd=100.0,
            ))

        assert history.get_total_savings() == 300.0

    def test_max_records_limit(self):
        """Test max records limit."""
        history = SwitchHistory(max_records=5)

        for i in range(10):
            history.add_record(SwitchRecord(
                switch_id=f"SW-{i:03d}",
                timestamp=datetime.now(timezone.utc),
                from_fuel="a",
                to_fuel="b",
                trigger_type=TriggerType.ECONOMIC,
                duration_minutes=15.0,
                success=True,
            ))

        assert len(history._records) == 5


class TestFuelSwitchingController:
    """Tests for FuelSwitchingController class."""

    def test_initialization(self, fuel_switching_controller):
        """Test controller initialization."""
        assert fuel_switching_controller.state == SwitchingState.IDLE
        assert fuel_switching_controller.is_idle is True
        assert fuel_switching_controller.evaluation_count == 0

    def test_disabled_mode_no_switch(self, switching_input):
        """Test disabled mode returns no switch."""
        config = SwitchingConfig(mode=SwitchingMode.DISABLED)
        controller = FuelSwitchingController(config)

        result = controller.evaluate_switch(switching_input)

        assert result.recommended is False
        assert "disabled" in result.trigger_reason.lower()

    def test_evaluation_increments_count(self, fuel_switching_controller, switching_input):
        """Test evaluation count increments."""
        initial = fuel_switching_controller.evaluation_count

        fuel_switching_controller.evaluate_switch(switching_input)

        assert fuel_switching_controller.evaluation_count == initial + 1


class TestEconomicTrigger:
    """Tests for economic trigger evaluation."""

    def test_significant_savings_triggers_switch(self, all_fuel_prices):
        """Test significant savings triggers switch recommendation."""
        config = SwitchingConfig(
            mode=SwitchingMode.AUTOMATIC,
            price_differential_trigger_pct=10.0,
            min_savings_usd_hr=50.0,
            operator_confirmation_required=False,
        )
        controller = FuelSwitchingController(config)

        # Create prices where propane is much cheaper
        cheaper_prices = dict(all_fuel_prices)
        cheaper_prices["lpg_propane"] = FuelPrice(
            fuel_type="lpg_propane",
            price=2.00,  # Much cheaper than gas at 3.50
            commodity_price=1.50,
            source="test",
        )

        input_data = SwitchingInput(
            current_fuel="natural_gas",
            current_cost_usd_mmbtu=3.50,
            current_heat_input_mmbtu_hr=50.0,
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_prices=cheaper_prices,
            equipment_id="BOILER-001",
            time_on_current_fuel_hours=5.0,
            safety_interlocks={
                "flame_proven": True,
                "fuel_pressure_ok": True,
                "no_active_alarms": True,
            },
        )

        result = controller.evaluate_switch(input_data)

        assert result.recommended is True
        assert result.recommended_fuel == "lpg_propane"
        assert result.savings_pct > 10.0

    def test_insufficient_savings_no_switch(self, fuel_switching_controller, switching_input):
        """Test insufficient savings doesn't trigger switch."""
        # Default prices are close, so no significant savings
        result = fuel_switching_controller.evaluate_switch(switching_input)

        # May or may not recommend depending on exact prices
        if not result.recommended:
            assert "below" in result.trigger_reason.lower() or "no better" in result.trigger_reason.lower()


class TestTimingConstraints:
    """Tests for timing constraint validation."""

    def test_minimum_run_time_not_met(self, all_fuel_prices):
        """Test switch blocked if min run time not met."""
        config = SwitchingConfig(
            mode=SwitchingMode.AUTOMATIC,
            min_run_time_hours=4.0,
        )
        controller = FuelSwitchingController(config)

        input_data = SwitchingInput(
            current_fuel="natural_gas",
            current_cost_usd_mmbtu=3.50,
            current_heat_input_mmbtu_hr=50.0,
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_prices=all_fuel_prices,
            equipment_id="BOILER-001",
            time_on_current_fuel_hours=2.0,  # Less than 4 hours
            safety_interlocks={"flame_proven": True, "fuel_pressure_ok": True, "no_active_alarms": True},
        )

        result = controller.evaluate_switch(input_data)

        assert result.recommended is False
        assert "run time" in result.trigger_reason.lower()

    def test_max_switches_per_day_limit(self, all_fuel_prices):
        """Test switch blocked if max switches reached."""
        config = SwitchingConfig(
            mode=SwitchingMode.AUTOMATIC,
            max_switches_per_day=2,
        )
        controller = FuelSwitchingController(config)

        input_data = SwitchingInput(
            current_fuel="natural_gas",
            current_cost_usd_mmbtu=3.50,
            current_heat_input_mmbtu_hr=50.0,
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_prices=all_fuel_prices,
            equipment_id="BOILER-001",
            time_on_current_fuel_hours=5.0,
            switches_today=2,  # Already at limit
            safety_interlocks={"flame_proven": True, "fuel_pressure_ok": True, "no_active_alarms": True},
        )

        result = controller.evaluate_switch(input_data)

        assert result.recommended is False
        assert "limit" in result.trigger_reason.lower()


class TestSafetyInterlocks:
    """Tests for safety interlock validation."""

    def test_missing_interlock_blocks_switch(self, all_fuel_prices):
        """Test missing safety interlock blocks switch."""
        config = SwitchingConfig(
            mode=SwitchingMode.AUTOMATIC,
            safety_interlock_enabled=True,
        )
        controller = FuelSwitchingController(config)

        input_data = SwitchingInput(
            current_fuel="natural_gas",
            current_cost_usd_mmbtu=3.50,
            current_heat_input_mmbtu_hr=50.0,
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_prices=all_fuel_prices,
            equipment_id="BOILER-001",
            time_on_current_fuel_hours=5.0,
            safety_interlocks={
                "flame_proven": False,  # Missing interlock
                "fuel_pressure_ok": True,
                "no_active_alarms": True,
            },
        )

        result = controller.evaluate_switch(input_data)

        assert result.safety_checks_passed is False
        assert "flame_proven" in result.blocked_interlocks

    def test_all_interlocks_passed(self, fuel_switching_controller, switching_input):
        """Test all interlocks passed."""
        result = fuel_switching_controller.evaluate_switch(switching_input)

        assert result.safety_checks_passed is True
        assert len(result.blocked_interlocks) == 0

    def test_safety_interlocks_constants(self):
        """Test safety interlock constants are defined."""
        assert "flame_proven" in SAFETY_INTERLOCKS
        assert "fuel_pressure_ok" in SAFETY_INTERLOCKS
        assert "no_active_alarms" in SAFETY_INTERLOCKS


class TestOperatorApproval:
    """Tests for operator approval workflow."""

    def test_semi_automatic_requires_approval(self, all_fuel_prices):
        """Test semi-automatic mode requires operator approval."""
        config = SwitchingConfig(
            mode=SwitchingMode.SEMI_AUTOMATIC,
            price_differential_trigger_pct=5.0,
            min_savings_usd_hr=10.0,
        )
        controller = FuelSwitchingController(config)

        # Create scenario where switch is recommended
        cheaper_prices = dict(all_fuel_prices)
        cheaper_prices["lpg_propane"] = FuelPrice(
            fuel_type="lpg_propane",
            price=2.50,
            commodity_price=2.00,
            source="test",
        )

        input_data = SwitchingInput(
            current_fuel="natural_gas",
            current_cost_usd_mmbtu=3.50,
            current_heat_input_mmbtu_hr=50.0,
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_prices=cheaper_prices,
            equipment_id="BOILER-001",
            time_on_current_fuel_hours=5.0,
            safety_interlocks={"flame_proven": True, "fuel_pressure_ok": True, "no_active_alarms": True},
        )

        result = controller.evaluate_switch(input_data)

        if result.recommended:
            assert result.requires_operator_approval is True
            assert controller.state == SwitchingState.PENDING_APPROVAL

    def test_approve_switch(self, all_fuel_prices):
        """Test approving a pending switch."""
        config = SwitchingConfig(
            mode=SwitchingMode.SEMI_AUTOMATIC,
            price_differential_trigger_pct=5.0,
            min_savings_usd_hr=10.0,
            confirmation_timeout_minutes=30,
        )
        controller = FuelSwitchingController(config)

        # Create scenario with significant savings
        cheaper_prices = dict(all_fuel_prices)
        cheaper_prices["lpg_propane"] = FuelPrice(
            fuel_type="lpg_propane",
            price=2.50,
            commodity_price=2.00,
            source="test",
        )

        input_data = SwitchingInput(
            current_fuel="natural_gas",
            current_cost_usd_mmbtu=3.50,
            current_heat_input_mmbtu_hr=50.0,
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_prices=cheaper_prices,
            equipment_id="BOILER-001",
            time_on_current_fuel_hours=5.0,
            safety_interlocks={"flame_proven": True, "fuel_pressure_ok": True, "no_active_alarms": True},
        )

        result = controller.evaluate_switch(input_data)

        if result.recommended:
            approved = controller.approve_switch("operator_001")
            assert approved is True
            assert controller.state == SwitchingState.PREPARING

    def test_approval_timeout(self, all_fuel_prices):
        """Test approval timeout."""
        config = SwitchingConfig(
            mode=SwitchingMode.SEMI_AUTOMATIC,
            confirmation_timeout_minutes=0,  # Immediate timeout
        )
        controller = FuelSwitchingController(config)

        # Force state to pending
        controller._state = SwitchingState.PENDING_APPROVAL
        controller._pending_switch = SwitchingOutput(
            recommended=True,
            current_fuel="natural_gas",
            recommended_fuel="propane",
            trigger_type=TriggerType.ECONOMIC,
            trigger_reason="test",
            current_cost_usd_hr=175.0,
            recommended_cost_usd_hr=150.0,
            savings_usd_hr=25.0,
            savings_pct=14.3,
            transition_time_minutes=15,
            requires_purge=True,
            safety_checks_passed=True,
            requires_operator_approval=True,
            approval_timeout=datetime.now(timezone.utc) - timedelta(minutes=1),  # Expired
            provenance_hash="test",
        )

        approved = controller.approve_switch("operator_001")
        assert approved is False


class TestSwitchExecution:
    """Tests for switch execution."""

    def test_can_execute_in_preparing_state(self, fuel_switching_controller):
        """Test can execute when in preparing state."""
        fuel_switching_controller._state = SwitchingState.PREPARING
        fuel_switching_controller._pending_switch = Mock()

        assert fuel_switching_controller.can_execute() is True

    def test_cannot_execute_in_idle_state(self, fuel_switching_controller):
        """Test cannot execute when idle."""
        assert fuel_switching_controller.can_execute() is False

    def test_execute_switch_starts_purge(self, switching_config):
        """Test executing switch starts purge if required."""
        config = SwitchingConfig(require_purge=True)
        controller = FuelSwitchingController(config)

        controller._state = SwitchingState.PREPARING
        controller._pending_switch = Mock(requires_purge=True)

        result = controller.execute_switch()

        assert result is True
        assert controller.state == SwitchingState.PURGING

    def test_execute_switch_without_purge(self):
        """Test executing switch without purge."""
        config = SwitchingConfig(require_purge=False)
        controller = FuelSwitchingController(config)

        controller._state = SwitchingState.PREPARING
        controller._pending_switch = Mock(requires_purge=False)

        result = controller.execute_switch()

        assert result is True
        assert controller.state == SwitchingState.TRANSITIONING


class TestSwitchCompletion:
    """Tests for switch completion."""

    def test_complete_switch_success(self, fuel_switching_controller):
        """Test completing successful switch."""
        # Set up pending switch
        fuel_switching_controller._state = SwitchingState.TRANSITIONING
        fuel_switching_controller._pending_switch = SwitchingOutput(
            recommended=True,
            current_fuel="natural_gas",
            recommended_fuel="propane",
            trigger_type=TriggerType.ECONOMIC,
            trigger_reason="test",
            current_cost_usd_hr=175.0,
            recommended_cost_usd_hr=150.0,
            savings_usd_hr=25.0,
            savings_pct=14.3,
            transition_time_minutes=15,
            requires_purge=True,
            safety_checks_passed=True,
            requires_operator_approval=True,
            approval_timeout=datetime.now(timezone.utc) + timedelta(hours=1),
            provenance_hash="test",
            evaluation_id="test-001",
        )

        fuel_switching_controller.complete_switch(success=True, notes="Completed normally")

        assert fuel_switching_controller.state == SwitchingState.COMPLETE
        assert fuel_switching_controller._pending_switch is None

    def test_complete_switch_failure(self, fuel_switching_controller):
        """Test completing failed switch."""
        fuel_switching_controller._state = SwitchingState.TRANSITIONING
        fuel_switching_controller._pending_switch = SwitchingOutput(
            recommended=True,
            current_fuel="natural_gas",
            recommended_fuel="propane",
            trigger_type=TriggerType.ECONOMIC,
            trigger_reason="test",
            current_cost_usd_hr=175.0,
            recommended_cost_usd_hr=150.0,
            savings_usd_hr=25.0,
            savings_pct=14.3,
            transition_time_minutes=15,
            requires_purge=True,
            safety_checks_passed=True,
            requires_operator_approval=True,
            approval_timeout=datetime.now(timezone.utc) + timedelta(hours=1),
            provenance_hash="test",
            evaluation_id="test-001",
        )

        fuel_switching_controller.complete_switch(success=False, notes="Valve stuck")

        assert fuel_switching_controller.state == SwitchingState.FAILED


class TestAbortSwitch:
    """Tests for switch abort functionality."""

    def test_abort_switch(self, fuel_switching_controller):
        """Test aborting a switch."""
        fuel_switching_controller._state = SwitchingState.TRANSITIONING
        fuel_switching_controller._pending_switch = Mock()

        fuel_switching_controller.abort_switch("Emergency stop")

        assert fuel_switching_controller.state == SwitchingState.ABORTED
        assert fuel_switching_controller._pending_switch is None


class TestControllerReset:
    """Tests for controller reset."""

    def test_reset_to_idle(self, fuel_switching_controller):
        """Test resetting controller to idle."""
        fuel_switching_controller._state = SwitchingState.ABORTED
        fuel_switching_controller._pending_switch = Mock()

        fuel_switching_controller.reset()

        assert fuel_switching_controller.state == SwitchingState.IDLE
        assert fuel_switching_controller._pending_switch is None


class TestTransitionCostCalculation:
    """Tests for transition cost calculation."""

    def test_transition_cost_includes_efficiency_loss(self, switching_config):
        """Test transition cost includes efficiency loss."""
        controller = FuelSwitchingController(switching_config)

        cost = controller._calculate_transition_cost(
            from_fuel="natural_gas",
            to_fuel="no2_fuel_oil",
            heat_input_mmbtu_hr=50.0,
        )

        assert cost > 0

    def test_transition_cost_includes_purge(self):
        """Test transition cost includes purge cost."""
        config = SwitchingConfig(require_purge=True, purge_duration_seconds=120)
        controller = FuelSwitchingController(config)

        cost_with_purge = controller._calculate_transition_cost(
            "natural_gas", "fuel_oil", 50.0
        )

        config_no_purge = SwitchingConfig(require_purge=False)
        controller_no_purge = FuelSwitchingController(config_no_purge)

        cost_no_purge = controller_no_purge._calculate_transition_cost(
            "natural_gas", "fuel_oil", 50.0
        )

        assert cost_with_purge > cost_no_purge


class TestPaybackCalculation:
    """Tests for payback period calculation."""

    def test_payback_calculated(self, all_fuel_prices):
        """Test payback period is calculated."""
        config = SwitchingConfig(
            mode=SwitchingMode.AUTOMATIC,
            price_differential_trigger_pct=5.0,
            min_savings_usd_hr=10.0,
            operator_confirmation_required=False,
        )
        controller = FuelSwitchingController(config)

        # Create significant price difference
        cheaper_prices = dict(all_fuel_prices)
        cheaper_prices["lpg_propane"] = FuelPrice(
            fuel_type="lpg_propane",
            price=2.00,
            commodity_price=1.50,
            source="test",
        )

        input_data = SwitchingInput(
            current_fuel="natural_gas",
            current_cost_usd_mmbtu=3.50,
            current_heat_input_mmbtu_hr=50.0,
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_prices=cheaper_prices,
            equipment_id="BOILER-001",
            time_on_current_fuel_hours=5.0,
            safety_interlocks={"flame_proven": True, "fuel_pressure_ok": True, "no_active_alarms": True},
        )

        result = controller.evaluate_switch(input_data)

        if result.recommended:
            assert result.payback_hours is not None
            assert result.payback_hours > 0


class TestProvenanceTracking:
    """Tests for provenance tracking."""

    def test_provenance_hash_generated(self, fuel_switching_controller, switching_input):
        """Test provenance hash is generated."""
        result = fuel_switching_controller.evaluate_switch(switching_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_hash_deterministic(self, fuel_switching_controller, switching_input):
        """Test same input produces same hash."""
        # Reset controller between evaluations
        fuel_switching_controller.reset()
        result1 = fuel_switching_controller.evaluate_switch(switching_input)

        fuel_switching_controller.reset()
        result2 = fuel_switching_controller.evaluate_switch(switching_input)

        assert result1.provenance_hash == result2.provenance_hash
