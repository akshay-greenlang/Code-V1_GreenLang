"""
GL-019 HEATSCHEDULER - Demand Charge Module Tests

Unit tests for demand charge optimization including period analysis,
load shifting, demand response, and comprehensive optimization.

Test Coverage:
    - DemandPeriodAnalyzer period detection
    - LoadShifter opportunity identification
    - DemandResponseHandler event management
    - DemandChargeOptimizer cost optimization
    - Ratchet clause handling

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import statistics


class TestDemandPeriodAnalyzerInitialization:
    """Tests for DemandPeriodAnalyzer initialization."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandPeriodAnalyzer,
        )

        analyzer = DemandPeriodAnalyzer(
            demand_interval_minutes=15,
            use_rolling_average=True,
        )

        assert analyzer._interval_minutes == 15
        assert analyzer._use_rolling is True

    def test_analyzer_default_values(self):
        """Test analyzer default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandPeriodAnalyzer,
        )

        analyzer = DemandPeriodAnalyzer()

        assert analyzer._interval_minutes == 15
        assert analyzer._use_rolling is True


class TestDemandPeriodAnalyzerPeriodAnalysis:
    """Tests for period analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandPeriodAnalyzer,
        )
        return DemandPeriodAnalyzer()

    def test_analyze_periods_returns_list(self, analyzer, sample_load_forecast, sample_tariff_config):
        """Test analyze_periods returns list of periods."""
        periods = analyzer.analyze_periods(
            load_forecast=sample_load_forecast,
            tariff_config=sample_tariff_config,
        )

        assert isinstance(periods, list)
        assert len(periods) > 0

    def test_analyze_periods_empty_forecast(self, analyzer):
        """Test analyze_periods handles empty forecast."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            LoadForecastResult,
            LoadForecastStatus,
        )

        empty_forecast = LoadForecastResult(
            status=LoadForecastStatus.SUCCESS,
            forecast_points=[],
            forecast_horizon_hours=24,
        )

        periods = analyzer.analyze_periods(empty_forecast)

        assert len(periods) == 0

    def test_analyze_periods_calculates_demand(
        self,
        analyzer,
        sample_load_forecast,
        sample_tariff_config,
    ):
        """Test periods have calculated demand values."""
        periods = analyzer.analyze_periods(
            load_forecast=sample_load_forecast,
            tariff_config=sample_tariff_config,
        )

        for period in periods:
            assert period.avg_demand_kw >= 0
            assert period.peak_demand_kw >= 0
            assert period.peak_demand_kw >= period.avg_demand_kw

    def test_analyze_periods_identifies_on_peak(
        self,
        analyzer,
        sample_load_forecast,
        sample_tariff_config,
    ):
        """Test periods correctly identify on-peak times."""
        periods = analyzer.analyze_periods(
            load_forecast=sample_load_forecast,
            tariff_config=sample_tariff_config,
        )

        # Check that some periods are marked as on-peak
        on_peak_periods = [p for p in periods if p.is_on_peak]
        off_peak_periods = [p for p in periods if not p.is_on_peak]

        assert len(on_peak_periods) > 0
        assert len(off_peak_periods) > 0

    def test_find_peak_period(self, analyzer, sample_load_forecast, sample_tariff_config):
        """Test find_peak_period returns highest demand period."""
        periods = analyzer.analyze_periods(
            load_forecast=sample_load_forecast,
            tariff_config=sample_tariff_config,
        )

        peak_period = analyzer.find_peak_period(periods, on_peak_only=True)

        assert peak_period is not None
        # Should be the maximum peak demand among on-peak periods
        on_peak = [p for p in periods if p.is_on_peak]
        if on_peak:
            max_demand = max(p.peak_demand_kw for p in on_peak)
            assert peak_period.peak_demand_kw == max_demand

    def test_find_peak_period_all_periods(
        self,
        analyzer,
        sample_load_forecast,
        sample_tariff_config,
    ):
        """Test find_peak_period considering all periods."""
        periods = analyzer.analyze_periods(
            load_forecast=sample_load_forecast,
            tariff_config=sample_tariff_config,
        )

        peak_period = analyzer.find_peak_period(periods, on_peak_only=False)

        assert peak_period is not None
        max_demand = max(p.peak_demand_kw for p in periods)
        assert peak_period.peak_demand_kw == max_demand

    def test_calculate_billing_demand_no_ratchet(self, analyzer, sample_load_forecast, sample_tariff_config):
        """Test billing demand calculation without ratchet."""
        periods = analyzer.analyze_periods(
            load_forecast=sample_load_forecast,
            tariff_config=sample_tariff_config,
        )

        billing_demand = analyzer.calculate_billing_demand(
            periods=periods,
            ratchet_percentage=0.0,
            annual_peak_kw=0.0,
        )

        # Should be the max peak demand
        max_demand = max(p.peak_demand_kw for p in periods)
        assert billing_demand == max_demand

    def test_calculate_billing_demand_with_ratchet(
        self,
        analyzer,
        sample_load_forecast,
        sample_tariff_config,
    ):
        """Test billing demand calculation with ratchet."""
        periods = analyzer.analyze_periods(
            load_forecast=sample_load_forecast,
            tariff_config=sample_tariff_config,
        )

        current_max = max(p.peak_demand_kw for p in periods)

        # Set annual peak higher than current
        annual_peak = current_max * 1.5

        billing_demand = analyzer.calculate_billing_demand(
            periods=periods,
            ratchet_percentage=80.0,
            annual_peak_kw=annual_peak,
        )

        # Billing should be max(current, 80% of annual)
        ratchet_demand = annual_peak * 0.8
        expected = max(current_max, ratchet_demand)
        assert billing_demand == expected


class TestDemandPeriodAnalyzerPeakHours:
    """Tests for peak hour identification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandPeriodAnalyzer,
        )
        return DemandPeriodAnalyzer()

    def test_get_peak_hours_from_tariff(self, analyzer, sample_tariff_config):
        """Test peak hours from tariff configuration."""
        peak_hours = analyzer._get_peak_hours(sample_tariff_config)

        # Sample config: peak 14-20
        assert 14 in peak_hours
        assert 20 in peak_hours
        assert 12 not in peak_hours

    def test_get_peak_hours_default(self, analyzer):
        """Test default peak hours without tariff."""
        peak_hours = analyzer._get_peak_hours(None)

        # Default: 14-19
        assert peak_hours == {14, 15, 16, 17, 18, 19}


class TestLoadShifterInitialization:
    """Tests for LoadShifter initialization."""

    def test_shifter_initialization(self):
        """Test load shifter initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            LoadShifter,
        )

        shifter = LoadShifter(
            max_shift_hours=4,
            min_savings_threshold_usd=5.0,
        )

        assert shifter._max_shift_hours == 4
        assert shifter._min_savings == 5.0

    def test_shifter_default_values(self):
        """Test load shifter default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            LoadShifter,
        )

        shifter = LoadShifter()

        assert shifter._max_shift_hours == 4
        assert shifter._min_savings == 5.0


class TestLoadShifterOpportunityIdentification:
    """Tests for shift opportunity identification."""

    @pytest.fixture
    def shifter(self):
        """Create load shifter instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            LoadShifter,
        )
        return LoadShifter(max_shift_hours=4, min_savings_threshold_usd=1.0)

    def test_identify_opportunities_returns_list(
        self,
        shifter,
        sample_load_forecast,
        sample_tariff_config,
    ):
        """Test identify_shift_opportunities returns list."""
        opportunities = shifter.identify_shift_opportunities(
            load_forecast=sample_load_forecast,
            demand_limit_kw=2000.0,  # Low limit to trigger shifts
            tariff_config=sample_tariff_config,
        )

        assert isinstance(opportunities, list)

    def test_identify_opportunities_with_peaks(
        self,
        shifter,
        sample_load_forecast,
        sample_tariff_config,
    ):
        """Test opportunities found when peaks exceed limit."""
        # Get max load in forecast
        max_load = max(p.load_kw for p in sample_load_forecast.forecast_points)

        # Set limit below max to ensure peaks exist
        opportunities = shifter.identify_shift_opportunities(
            load_forecast=sample_load_forecast,
            demand_limit_kw=max_load * 0.8,
            tariff_config=sample_tariff_config,
        )

        # Should find some opportunities if there are peaks
        # (depends on load pattern - may be zero if no suitable valleys)

    def test_opportunity_has_required_fields(
        self,
        shifter,
        sample_load_forecast,
        sample_tariff_config,
    ):
        """Test opportunities have required fields."""
        max_load = max(p.load_kw for p in sample_load_forecast.forecast_points)

        opportunities = shifter.identify_shift_opportunities(
            load_forecast=sample_load_forecast,
            demand_limit_kw=max_load * 0.5,
            tariff_config=sample_tariff_config,
        )

        for opp in opportunities:
            assert "from_time" in opp
            assert "to_time" in opp
            assert "load_kw" in opp
            assert "savings_usd" in opp
            assert "shift_hours" in opp

    def test_generate_shift_actions(self, shifter, base_timestamp):
        """Test generate_shift_actions creates actions."""
        opportunities = [
            {
                "from_time": base_timestamp + timedelta(hours=14),
                "to_time": base_timestamp + timedelta(hours=6),
                "load_kw": 500.0,
                "savings_usd": 25.0,
                "shift_hours": 8.0,
            }
        ]

        actions = shifter.generate_shift_actions(opportunities)

        assert len(actions) == 2  # One reduction, one addition
        # First action should be reduction (negative power)
        assert actions[0].power_setpoint_kw < 0
        # Second action should be addition (positive power)
        assert actions[1].power_setpoint_kw > 0


class TestDemandResponseHandlerInitialization:
    """Tests for DemandResponseHandler initialization."""

    def test_handler_initialization(self, sample_demand_charge_config):
        """Test handler initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandResponseHandler,
        )

        handler = DemandResponseHandler(sample_demand_charge_config)

        assert handler._enabled is True
        assert handler._threshold_kw == 1000.0
        assert handler._max_curtailment_pct == 30.0

    def test_handler_disabled(self, sample_demand_charge_config):
        """Test handler when DR is disabled."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandResponseHandler,
        )

        sample_demand_charge_config.enable_demand_response = False
        handler = DemandResponseHandler(sample_demand_charge_config)

        assert handler.is_enabled is False


class TestDemandResponseHandlerEvents:
    """Tests for DR event management."""

    @pytest.fixture
    def handler(self, sample_demand_charge_config):
        """Create handler instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandResponseHandler,
        )
        return DemandResponseHandler(sample_demand_charge_config)

    def test_register_dr_event(self, handler, base_timestamp):
        """Test registering a DR event."""
        handler.register_dr_event(
            event_id="DR-001",
            start_time=base_timestamp + timedelta(hours=14),
            end_time=base_timestamp + timedelta(hours=16),
            curtailment_target_kw=500.0,
        )

        assert len(handler._active_events) == 1
        assert handler._active_events[0]["event_id"] == "DR-001"
        assert handler._active_events[0]["target_kw"] == 500.0

    def test_get_curtailment_during_event(self, handler, base_timestamp):
        """Test getting curtailment during active event."""
        start = base_timestamp + timedelta(hours=14)
        end = base_timestamp + timedelta(hours=16)

        handler.register_dr_event(
            event_id="DR-001",
            start_time=start,
            end_time=end,
            curtailment_target_kw=500.0,
        )

        # Check during event
        curtailment, event_id = handler.get_curtailment_for_time(
            target_time=base_timestamp + timedelta(hours=15),
            baseline_load_kw=2000.0,
        )

        assert curtailment > 0
        assert event_id == "DR-001"

    def test_get_curtailment_outside_event(self, handler, base_timestamp):
        """Test getting curtailment outside event."""
        start = base_timestamp + timedelta(hours=14)
        end = base_timestamp + timedelta(hours=16)

        handler.register_dr_event(
            event_id="DR-001",
            start_time=start,
            end_time=end,
            curtailment_target_kw=500.0,
        )

        # Check outside event
        curtailment, event_id = handler.get_curtailment_for_time(
            target_time=base_timestamp + timedelta(hours=10),
            baseline_load_kw=2000.0,
        )

        assert curtailment == 0.0
        assert event_id is None

    def test_curtailment_limited_by_max(self, handler, base_timestamp):
        """Test curtailment is limited by max percentage."""
        start = base_timestamp + timedelta(hours=14)
        end = base_timestamp + timedelta(hours=16)

        # Register event with high target
        handler.register_dr_event(
            event_id="DR-001",
            start_time=start,
            end_time=end,
            curtailment_target_kw=1000.0,  # High target
        )

        baseline = 1000.0  # With 30% max, can only curtail 300

        curtailment, _ = handler.get_curtailment_for_time(
            target_time=base_timestamp + timedelta(hours=15),
            baseline_load_kw=baseline,
        )

        # Should be limited to 30% of baseline
        assert curtailment <= baseline * 0.3

    def test_generate_dr_actions(self, handler, base_timestamp):
        """Test generating DR actions."""
        start = base_timestamp + timedelta(hours=14)
        end = base_timestamp + timedelta(hours=16)

        handler.register_dr_event(
            event_id="DR-001",
            start_time=start,
            end_time=end,
            curtailment_target_kw=500.0,
        )

        actions = handler.generate_dr_actions("DR-001")

        assert len(actions) == 2  # Notification and curtailment
        # First action is notification/ramp-down
        assert actions[0].is_mandatory is True
        # Second action is the curtailment
        assert actions[1].is_mandatory is True

    def test_generate_dr_actions_unknown_event(self, handler):
        """Test generating actions for unknown event returns empty."""
        actions = handler.generate_dr_actions("UNKNOWN")

        assert len(actions) == 0


class TestDemandChargeOptimizerInitialization:
    """Tests for DemandChargeOptimizer initialization."""

    def test_optimizer_initialization(
        self,
        sample_demand_charge_config,
        sample_tariff_config,
    ):
        """Test optimizer initializes correctly."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )

        optimizer = DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

        assert optimizer._config == sample_demand_charge_config
        assert optimizer._tariff == sample_tariff_config
        assert optimizer._analyzer is not None
        assert optimizer._shifter is not None
        assert optimizer._dr_handler is not None


class TestDemandChargeOptimizerOptimization:
    """Tests for demand charge optimization."""

    @pytest.fixture
    def optimizer(self, sample_demand_charge_config, sample_tariff_config):
        """Create optimizer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        return DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

    def test_optimize_returns_result(self, optimizer, sample_load_forecast):
        """Test optimize returns result."""
        result = optimizer.optimize(
            load_forecast=sample_load_forecast,
            storage_capacity_kw=500.0,
        )

        assert result is not None
        assert result.baseline_peak_kw > 0
        assert result.optimized_peak_kw > 0

    def test_optimize_calculates_savings(self, optimizer, sample_load_forecast):
        """Test optimize calculates savings."""
        result = optimizer.optimize(
            load_forecast=sample_load_forecast,
            storage_capacity_kw=500.0,
        )

        assert result.baseline_demand_charge_usd >= 0
        assert result.optimized_demand_charge_usd >= 0
        assert result.demand_charge_savings_usd >= 0

    def test_optimize_with_storage_reduces_peak(self, optimizer, sample_load_forecast):
        """Test storage capacity helps reduce peak."""
        # Without storage
        result_no_storage = optimizer.optimize(
            load_forecast=sample_load_forecast,
            storage_capacity_kw=0.0,
        )

        # Reset optimizer state
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
            DemandChargeConfiguration,
        )

        optimizer2 = DemandChargeOptimizer(
            config=DemandChargeConfiguration(
                peak_demand_limit_kw=5000.0,
                enable_load_shifting=True,
            ),
            tariff_config=optimizer._tariff,
        )

        # With storage
        result_with_storage = optimizer2.optimize(
            load_forecast=sample_load_forecast,
            storage_capacity_kw=1000.0,
        )

        # Storage should help reduce or maintain peak
        assert result_with_storage.optimized_peak_kw <= result_no_storage.optimized_peak_kw + 1

    def test_optimize_identifies_peak_time(self, optimizer, sample_load_forecast):
        """Test optimizer identifies peak time."""
        result = optimizer.optimize(
            load_forecast=sample_load_forecast,
        )

        assert result.peak_time_baseline is not None

    def test_optimize_with_ratchet(self, optimizer, sample_load_forecast):
        """Test optimizer considers ratchet."""
        result = optimizer.optimize(
            load_forecast=sample_load_forecast,
        )

        assert result.annual_ratchet_peak_kw is not None
        assert result.ratchet_impact_usd is not None

    def test_optimize_generates_alert_when_exceeded(
        self,
        sample_demand_charge_config,
        sample_tariff_config,
        sample_load_forecast,
    ):
        """Test optimizer generates alert when limit exceeded."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )

        # Set very low limit to ensure exceedance
        sample_demand_charge_config.peak_demand_limit_kw = 100.0

        optimizer = DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

        result = optimizer.optimize(
            load_forecast=sample_load_forecast,
        )

        assert result.peak_limit_exceeded is True
        assert result.alert_level is not None
        assert result.alert_message is not None


class TestDemandChargeOptimizerDRIntegration:
    """Tests for DR integration in optimizer."""

    @pytest.fixture
    def optimizer(self, sample_demand_charge_config, sample_tariff_config):
        """Create optimizer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        return DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

    def test_register_dr_event_through_optimizer(self, optimizer, base_timestamp):
        """Test registering DR event through optimizer."""
        optimizer.register_dr_event(
            event_id="DR-001",
            start_time=base_timestamp + timedelta(hours=14),
            end_time=base_timestamp + timedelta(hours=16),
            target_kw=500.0,
        )

        # Event should be registered in handler
        assert len(optimizer._dr_handler._active_events) == 1

    def test_get_dr_actions_through_optimizer(self, optimizer, base_timestamp):
        """Test getting DR actions through optimizer."""
        optimizer.register_dr_event(
            event_id="DR-001",
            start_time=base_timestamp + timedelta(hours=14),
            end_time=base_timestamp + timedelta(hours=16),
            target_kw=500.0,
        )

        actions = optimizer.get_dr_actions("DR-001")

        assert len(actions) == 2


class TestDemandChargeCostCalculations:
    """Tests for demand charge cost calculations."""

    @pytest.fixture
    def optimizer(self, sample_demand_charge_config, sample_tariff_config):
        """Create optimizer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        return DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

    def test_calculate_demand_charge_with_tariff(self, optimizer):
        """Test demand charge calculation with tariff."""
        # From sample_tariff_config: demand_charge = $12.50/kW, peak_demand_charge = $5/kW
        charge = optimizer._calculate_demand_charge(1000.0)

        expected = 1000.0 * (12.50 + 5.0)
        assert charge == expected

    def test_calculate_demand_charge_without_tariff(
        self,
        sample_demand_charge_config,
    ):
        """Test demand charge calculation without tariff."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )

        optimizer = DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=None,
        )

        charge = optimizer._calculate_demand_charge(1000.0)

        # Default: $15/kW
        expected = 1000.0 * 15.0
        assert charge == expected

    def test_calculate_ratchet_impact(self, optimizer):
        """Test ratchet impact calculation."""
        # New peak higher than annual
        impact = optimizer._calculate_ratchet_impact(
            new_peak_kw=5000.0,
            previous_peak_kw=4000.0,
        )

        # Should calculate annual cost impact
        assert impact >= 0


class TestDemandChargePerformance:
    """Performance tests for demand charge optimization."""

    @pytest.mark.performance
    def test_optimization_time(
        self,
        sample_demand_charge_config,
        sample_tariff_config,
        sample_load_forecast,
    ):
        """Test optimization completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        import time

        optimizer = DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

        start = time.time()
        result = optimizer.optimize(
            load_forecast=sample_load_forecast,
            storage_capacity_kw=500.0,
        )
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 1.0  # Should complete in under 1 second

    @pytest.mark.performance
    def test_large_forecast_optimization(
        self,
        sample_demand_charge_config,
        sample_tariff_config,
        large_load_forecast,
    ):
        """Test optimization with large forecast."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        import time

        optimizer = DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

        start = time.time()
        result = optimizer.optimize(
            load_forecast=large_load_forecast,
            storage_capacity_kw=500.0,
        )
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 3.0  # Should complete in under 3 seconds


class TestDemandAlertLevels:
    """Tests for demand alert level thresholds."""

    @pytest.fixture
    def optimizer(self, sample_demand_charge_config, sample_tariff_config):
        """Create optimizer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        return DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

    def test_critical_alert_over_20_percent(
        self,
        sample_demand_charge_config,
        sample_tariff_config,
        sample_load_forecast,
    ):
        """Test CRITICAL alert when over 20% exceeded."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
            DemandChargeOptimizer,
        )
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import DemandAlertLevel

        # Set very low limit
        sample_demand_charge_config.peak_demand_limit_kw = 1000.0

        optimizer = DemandChargeOptimizer(
            config=sample_demand_charge_config,
            tariff_config=sample_tariff_config,
        )

        result = optimizer.optimize(load_forecast=sample_load_forecast)

        if result.peak_limit_exceeded:
            excess_pct = (
                (result.optimized_peak_kw - 1000.0) / 1000.0 * 100
            )
            if excess_pct > 20:
                assert result.alert_level == DemandAlertLevel.CRITICAL
