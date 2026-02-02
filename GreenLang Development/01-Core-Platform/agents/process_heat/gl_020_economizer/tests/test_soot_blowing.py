"""
Unit tests for GL-020 ECONOPULSE Soot Blower Optimizer

Tests soot blower scheduling, trigger logic, and steam optimization.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units

Zero-Hallucination: All calculations use deterministic formulas.
"""

import pytest
from datetime import datetime, timezone, timedelta

from ..soot_blowing import (
    SootBlowerOptimizer,
    SootBlowerConfig,
    SootBlowerInput,
    SootBlowerResult,
    BlowEffectivenessRecord,
    create_soot_blower_optimizer,
    SOOT_BLOWING_STEAM_ENTHALPY,
    STEAM_COST_PER_KLBS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default soot blower configuration."""
    return SootBlowerConfig()


@pytest.fixture
def optimizer(default_config):
    """Create default soot blower optimizer."""
    return SootBlowerOptimizer(default_config)


@pytest.fixture
def custom_config():
    """Custom soot blower configuration."""
    return SootBlowerConfig(
        num_blowers=6,
        steam_per_blower_lb=600.0,
        fixed_interval_hours=6.0,
        min_interval_hours=1.5,
        max_interval_hours=10.0,
        dp_trigger_ratio=1.25,
        effectiveness_trigger_ratio=0.92,
    )


@pytest.fixture
def normal_input():
    """Normal operating condition input."""
    return SootBlowerInput(
        timestamp=datetime.now(timezone.utc),
        gas_side_dp_ratio=1.1,
        effectiveness_ratio=0.95,
        boiler_load_pct=80.0,
        hours_since_last_blow=4.0,
    )


@pytest.fixture
def fouled_input():
    """Fouled condition input triggering blowing."""
    return SootBlowerInput(
        timestamp=datetime.now(timezone.utc),
        gas_side_dp_ratio=1.3,
        effectiveness_ratio=0.88,
        boiler_load_pct=80.0,
        hours_since_last_blow=6.0,
    )


@pytest.fixture
def time_trigger_input():
    """Input with time trigger active."""
    return SootBlowerInput(
        timestamp=datetime.now(timezone.utc),
        gas_side_dp_ratio=1.05,
        effectiveness_ratio=0.97,
        boiler_load_pct=80.0,
        hours_since_last_blow=13.0,  # Exceeds max_interval
    )


@pytest.fixture
def blow_effectiveness_history():
    """Historical blow effectiveness records."""
    base_time = datetime.now(timezone.utc)
    return [
        BlowEffectivenessRecord(
            timestamp=base_time - timedelta(hours=48),
            pre_blow_dp_ratio=1.25,
            post_blow_dp_ratio=1.05,
            pre_blow_effectiveness=0.88,
            post_blow_effectiveness=0.95,
            dp_improvement_pct=16.0,
            effectiveness_gain=0.07,
            steam_used_lb=2000.0,
        ),
        BlowEffectivenessRecord(
            timestamp=base_time - timedelta(hours=24),
            pre_blow_dp_ratio=1.22,
            post_blow_dp_ratio=1.03,
            pre_blow_effectiveness=0.89,
            post_blow_effectiveness=0.96,
            dp_improvement_pct=15.6,
            effectiveness_gain=0.07,
            steam_used_lb=2000.0,
        ),
        BlowEffectivenessRecord(
            timestamp=base_time,
            pre_blow_dp_ratio=1.20,
            post_blow_dp_ratio=1.02,
            pre_blow_effectiveness=0.90,
            post_blow_effectiveness=0.96,
            dp_improvement_pct=15.0,
            effectiveness_gain=0.06,
            steam_used_lb=2000.0,
        ),
    ]


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestSootBlowerConfig:
    """Test soot blower configuration."""

    def test_default_values(self, default_config):
        """Test default configuration values."""
        assert default_config.num_blowers == 4
        assert default_config.steam_per_blower_lb == 500.0
        assert default_config.blowing_duration_s == 90
        assert default_config.fixed_interval_hours == 8.0
        assert default_config.min_interval_hours == 2.0
        assert default_config.max_interval_hours == 12.0
        assert default_config.dp_trigger_ratio == 1.2
        assert default_config.effectiveness_trigger_ratio == 0.95

    def test_custom_config(self, custom_config):
        """Test custom configuration."""
        assert custom_config.num_blowers == 6
        assert custom_config.steam_per_blower_lb == 600.0
        assert custom_config.min_interval_hours == 1.5
        assert custom_config.max_interval_hours == 10.0


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestSootBlowerOptimizerInit:
    """Test optimizer initialization."""

    def test_default_initialization(self, optimizer, default_config):
        """Test default initialization."""
        assert optimizer.config == default_config
        assert len(optimizer.blow_history) == 0

    def test_factory_function(self):
        """Test factory function."""
        optimizer = create_soot_blower_optimizer()
        assert isinstance(optimizer, SootBlowerOptimizer)

    def test_factory_with_config(self, custom_config):
        """Test factory with custom config."""
        optimizer = create_soot_blower_optimizer(custom_config)
        assert optimizer.config == custom_config


# =============================================================================
# TRIGGER LOGIC TESTS
# =============================================================================

class TestTriggerLogic:
    """Test blowing trigger logic."""

    def test_no_triggers_active(self, optimizer):
        """Test no triggers when conditions are good."""
        dp_trigger, eff_trigger, time_trigger, reason = optimizer.check_triggers(
            dp_ratio=1.0,
            effectiveness_ratio=0.98,
            outlet_temp_deviation_f=5.0,
            hours_since_blow=4.0,
        )

        assert dp_trigger is False
        assert eff_trigger is False
        assert time_trigger is False
        assert "No triggers" in reason

    def test_dp_trigger_active(self, optimizer):
        """Test DP ratio trigger."""
        dp_trigger, eff_trigger, time_trigger, reason = optimizer.check_triggers(
            dp_ratio=1.25,
            effectiveness_ratio=0.98,
            outlet_temp_deviation_f=5.0,
            hours_since_blow=4.0,
        )

        assert dp_trigger is True
        assert "DP ratio" in reason

    def test_effectiveness_trigger_active(self, optimizer):
        """Test effectiveness trigger."""
        dp_trigger, eff_trigger, time_trigger, reason = optimizer.check_triggers(
            dp_ratio=1.1,
            effectiveness_ratio=0.92,
            outlet_temp_deviation_f=5.0,
            hours_since_blow=4.0,
        )

        assert eff_trigger is True
        assert "Effectiveness" in reason

    def test_temperature_trigger_active(self, optimizer):
        """Test outlet temperature trigger."""
        dp_trigger, eff_trigger, time_trigger, reason = optimizer.check_triggers(
            dp_ratio=1.1,
            effectiveness_ratio=0.96,
            outlet_temp_deviation_f=25.0,  # High deviation
            hours_since_blow=4.0,
        )

        assert eff_trigger is True
        assert "temp" in reason.lower()

    def test_time_trigger_active(self, optimizer):
        """Test time trigger when max interval exceeded."""
        dp_trigger, eff_trigger, time_trigger, reason = optimizer.check_triggers(
            dp_ratio=1.1,
            effectiveness_ratio=0.98,
            outlet_temp_deviation_f=5.0,
            hours_since_blow=13.0,  # > 12 hour max
        )

        assert time_trigger is True
        assert "Time" in reason

    def test_multiple_triggers(self, optimizer):
        """Test multiple triggers active."""
        dp_trigger, eff_trigger, time_trigger, reason = optimizer.check_triggers(
            dp_ratio=1.25,
            effectiveness_ratio=0.90,
            outlet_temp_deviation_f=25.0,
            hours_since_blow=13.0,
        )

        assert dp_trigger is True
        assert eff_trigger is True
        assert time_trigger is True
        assert ";" in reason  # Multiple reasons separated


# =============================================================================
# OPTIMAL INTERVAL CALCULATION TESTS
# =============================================================================

class TestOptimalIntervalCalculation:
    """Test optimal blowing interval calculation."""

    def test_zero_fouling_rate(self, optimizer):
        """Test optimal interval with zero fouling rate."""
        interval = optimizer.calculate_optimal_interval(
            fouling_rate_pct_per_hour=0.0,
            blow_effectiveness=0.05,
        )

        assert interval == optimizer.config.max_interval_hours

    def test_high_fouling_rate(self, optimizer):
        """Test optimal interval with high fouling rate."""
        interval = optimizer.calculate_optimal_interval(
            fouling_rate_pct_per_hour=2.0,
            blow_effectiveness=0.05,
        )

        # High fouling should result in shorter interval
        assert interval < optimizer.config.fixed_interval_hours

    def test_interval_clamped_to_minimum(self, optimizer):
        """Test interval clamped to minimum."""
        interval = optimizer.calculate_optimal_interval(
            fouling_rate_pct_per_hour=10.0,  # Very high
            blow_effectiveness=0.05,
        )

        assert interval >= optimizer.config.min_interval_hours

    def test_interval_clamped_to_maximum(self, optimizer):
        """Test interval clamped to maximum."""
        interval = optimizer.calculate_optimal_interval(
            fouling_rate_pct_per_hour=0.01,  # Very low
            blow_effectiveness=0.05,
        )

        assert interval <= optimizer.config.max_interval_hours


# =============================================================================
# FOULING RATE ESTIMATION TESTS
# =============================================================================

class TestFoulingRateEstimation:
    """Test fouling rate estimation from history."""

    def test_default_rate_no_history(self, optimizer):
        """Test default rate with no history."""
        rate = optimizer.estimate_fouling_rate([])

        assert rate == 0.5  # Default value

    def test_rate_with_history(self, optimizer, blow_effectiveness_history):
        """Test rate estimation with history."""
        rate = optimizer.estimate_fouling_rate(blow_effectiveness_history)

        # Rate should be positive and reasonable
        assert rate > 0
        assert rate < 5.0  # Less than 5% per hour


# =============================================================================
# BLOW EFFECTIVENESS TESTS
# =============================================================================

class TestBlowEffectiveness:
    """Test blow effectiveness calculation and recording."""

    def test_calculate_effectiveness(self, optimizer):
        """Test blow effectiveness calculation."""
        dp_improvement, eff_gain = optimizer.calculate_blow_effectiveness(
            pre_blow_dp_ratio=1.25,
            post_blow_dp_ratio=1.05,
            pre_blow_effectiveness=0.88,
            post_blow_effectiveness=0.95,
        )

        # DP improvement = (1.25 - 1.05) / 1.25 * 100 = 16%
        assert dp_improvement == pytest.approx(16.0, rel=0.01)

        # Effectiveness gain = 0.95 - 0.88 = 0.07
        assert eff_gain == pytest.approx(0.07, rel=0.01)

    def test_record_blow_cycle(self, optimizer):
        """Test recording a blow cycle."""
        record = optimizer.record_blow_cycle(
            timestamp=datetime.now(timezone.utc),
            pre_blow_dp_ratio=1.3,
            post_blow_dp_ratio=1.05,
            pre_blow_effectiveness=0.85,
            post_blow_effectiveness=0.94,
        )

        assert isinstance(record, BlowEffectivenessRecord)
        assert record.dp_improvement_pct > 0
        assert record.effectiveness_gain > 0
        assert len(optimizer.blow_history) == 1

    def test_history_limit(self, optimizer):
        """Test blow history is limited to 100 records."""
        for i in range(110):
            optimizer.record_blow_cycle(
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
                pre_blow_dp_ratio=1.2,
                post_blow_dp_ratio=1.05,
                pre_blow_effectiveness=0.9,
                post_blow_effectiveness=0.95,
            )

        assert len(optimizer.blow_history) == 100


# =============================================================================
# STEAM SAVINGS CALCULATION TESTS
# =============================================================================

class TestSteamSavings:
    """Test steam savings calculation."""

    def test_no_savings_same_interval(self, optimizer):
        """Test no savings when intervals are equal."""
        savings_pct, daily_optimal, daily_fixed = optimizer.calculate_steam_savings(
            optimal_interval_hours=8.0,
            fixed_interval_hours=8.0,
        )

        assert savings_pct == pytest.approx(0.0, abs=0.1)
        assert daily_optimal == daily_fixed

    def test_savings_with_longer_interval(self, optimizer):
        """Test savings when optimal interval is longer."""
        savings_pct, daily_optimal, daily_fixed = optimizer.calculate_steam_savings(
            optimal_interval_hours=10.0,
            fixed_interval_hours=8.0,
        )

        assert savings_pct > 0
        assert daily_optimal < daily_fixed

    def test_negative_savings_shorter_interval(self, optimizer):
        """Test negative savings when optimal is shorter."""
        savings_pct, daily_optimal, daily_fixed = optimizer.calculate_steam_savings(
            optimal_interval_hours=6.0,
            fixed_interval_hours=8.0,
        )

        assert savings_pct < 0
        assert daily_optimal > daily_fixed

    def test_steam_quantity_calculation(self, optimizer):
        """Test daily steam quantity calculation."""
        _, daily_optimal, daily_fixed = optimizer.calculate_steam_savings(
            optimal_interval_hours=10.0,
            fixed_interval_hours=8.0,
        )

        steam_per_cycle = optimizer.config.num_blowers * optimizer.config.steam_per_blower_lb

        expected_fixed = (24.0 / 8.0) * steam_per_cycle
        assert daily_fixed == pytest.approx(expected_fixed, rel=0.01)


# =============================================================================
# BLOWING EFFICIENCY SCORE TESTS
# =============================================================================

class TestBlowingEfficiencyScore:
    """Test blowing efficiency score calculation."""

    def test_default_score_no_history(self, optimizer):
        """Test default score with no history."""
        score = optimizer.calculate_blowing_efficiency_score([])

        assert score == 0.5  # Default neutral score

    def test_good_effectiveness_score(self, optimizer, blow_effectiveness_history):
        """Test good score with effective blowing."""
        score = optimizer.calculate_blowing_efficiency_score(blow_effectiveness_history)

        assert score > 0.5  # Better than neutral

    def test_score_range(self, optimizer, blow_effectiveness_history):
        """Test score is in valid range."""
        score = optimizer.calculate_blowing_efficiency_score(blow_effectiveness_history)

        assert 0.0 <= score <= 1.0


# =============================================================================
# ECONOMICS CALCULATION TESTS
# =============================================================================

class TestEconomicsCalculation:
    """Test economic impact calculations."""

    def test_steam_cost_calculation(self, optimizer):
        """Test daily steam cost calculation."""
        steam_cost, _, _ = optimizer.calculate_economics(
            daily_steam_lb=6000.0,
            effectiveness_ratio=0.95,
        )

        expected_cost = 6000.0 * optimizer.config.steam_cost_per_klbs / 1000
        assert steam_cost == pytest.approx(expected_cost, rel=0.01)

    def test_efficiency_loss_cost(self, optimizer):
        """Test efficiency loss cost calculation."""
        _, eff_loss_cost, _ = optimizer.calculate_economics(
            daily_steam_lb=6000.0,
            effectiveness_ratio=0.90,  # 10% below design
        )

        assert eff_loss_cost > 0  # Should have cost from efficiency loss

    def test_net_impact(self, optimizer):
        """Test net economic impact calculation."""
        steam_cost, eff_loss_cost, net_impact = optimizer.calculate_economics(
            daily_steam_lb=6000.0,
            effectiveness_ratio=0.95,
        )

        # Net = efficiency loss cost - steam cost
        expected_net = eff_loss_cost - steam_cost
        assert net_impact == pytest.approx(expected_net, rel=0.01)


# =============================================================================
# COMPLETE OPTIMIZATION TESTS
# =============================================================================

class TestCompleteOptimization:
    """Test complete soot blower optimization."""

    def test_normal_operation(self, optimizer, normal_input):
        """Test optimization during normal operation."""
        result = optimizer.optimize(normal_input)

        assert isinstance(result, SootBlowerResult)
        assert result.blowing_recommended is False
        assert result.blowing_status == "idle"

    def test_fouled_condition_triggers_blowing(self, optimizer, fouled_input):
        """Test blowing recommended when fouled."""
        result = optimizer.optimize(fouled_input)

        assert result.blowing_recommended is True
        assert result.blowing_status == "scheduled"
        assert result.dp_trigger_active is True

    def test_time_trigger_blowing(self, optimizer, time_trigger_input):
        """Test blowing from time trigger."""
        result = optimizer.optimize(time_trigger_input)

        assert result.blowing_recommended is True
        assert result.time_trigger_active is True

    def test_minimum_interval_respected(self, optimizer):
        """Test minimum interval is respected."""
        input_data = SootBlowerInput(
            timestamp=datetime.now(timezone.utc),
            gas_side_dp_ratio=1.3,
            effectiveness_ratio=0.88,
            boiler_load_pct=80.0,
            hours_since_last_blow=1.0,  # Less than min_interval
        )

        result = optimizer.optimize(input_data)

        # Should not recommend blowing despite high DP
        assert result.blowing_recommended is False
        assert "min interval" in result.trigger_reason

    def test_steam_unavailable(self, optimizer, fouled_input):
        """Test behavior when steam unavailable."""
        fouled_input.steam_available = False

        result = optimizer.optimize(fouled_input)

        assert result.blowing_recommended is False
        assert result.blowing_status == "bypassed"

    def test_provenance_hash_included(self, optimizer, normal_input):
        """Test provenance hash in result."""
        result = optimizer.optimize(normal_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16

    def test_pre_post_blow_data_recorded(self, optimizer):
        """Test pre/post blow data is recorded."""
        input_data = SootBlowerInput(
            timestamp=datetime.now(timezone.utc),
            gas_side_dp_ratio=1.1,
            effectiveness_ratio=0.95,
            boiler_load_pct=80.0,
            hours_since_last_blow=4.0,
            pre_blow_effectiveness=0.88,
            post_blow_effectiveness=0.95,
            pre_blow_dp_ratio=1.25,
            post_blow_dp_ratio=1.05,
        )

        result = optimizer.optimize(input_data)

        assert result.pre_blow_effectiveness == 0.88
        assert result.post_blow_effectiveness == 0.95
        assert result.blow_effectiveness_gain is not None

    def test_recommended_next_blow_calculation(self, optimizer, normal_input):
        """Test recommended next blow time calculation."""
        result = optimizer.optimize(normal_input)

        # Should recommend next blow in the future
        assert result.recommended_next_blow_hours >= 0

    def test_optimal_interval_calculated(self, optimizer, normal_input):
        """Test optimal interval is calculated."""
        result = optimizer.optimize(normal_input)

        assert result.optimal_blow_interval_hours > 0
        assert optimizer.config.min_interval_hours <= result.optimal_blow_interval_hours
        assert result.optimal_blow_interval_hours <= optimizer.config.max_interval_hours


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestParameterized:
    """Parameterized tests for various conditions."""

    @pytest.mark.parametrize("dp_ratio,expected_trigger", [
        (1.0, False),
        (1.1, False),
        (1.2, True),  # At threshold
        (1.3, True),
        (1.5, True),
    ])
    def test_dp_trigger_threshold(self, optimizer, dp_ratio, expected_trigger):
        """Test DP trigger at various ratios."""
        dp_trigger, _, _, _ = optimizer.check_triggers(
            dp_ratio=dp_ratio,
            effectiveness_ratio=0.98,
            outlet_temp_deviation_f=5.0,
            hours_since_blow=4.0,
        )

        assert dp_trigger == expected_trigger

    @pytest.mark.parametrize("eff_ratio,expected_trigger", [
        (0.98, False),
        (0.96, False),
        (0.95, True),  # At threshold
        (0.92, True),
        (0.85, True),
    ])
    def test_effectiveness_trigger_threshold(self, optimizer, eff_ratio, expected_trigger):
        """Test effectiveness trigger at various ratios."""
        _, eff_trigger, _, _ = optimizer.check_triggers(
            dp_ratio=1.0,
            effectiveness_ratio=eff_ratio,
            outlet_temp_deviation_f=5.0,
            hours_since_blow=4.0,
        )

        assert eff_trigger == expected_trigger

    @pytest.mark.parametrize("hours,expected_trigger", [
        (4.0, False),
        (8.0, False),
        (12.0, True),  # At threshold
        (15.0, True),
        (24.0, True),
    ])
    def test_time_trigger_threshold(self, optimizer, hours, expected_trigger):
        """Test time trigger at various hours."""
        _, _, time_trigger, _ = optimizer.check_triggers(
            dp_ratio=1.0,
            effectiveness_ratio=0.98,
            outlet_temp_deviation_f=5.0,
            hours_since_blow=hours,
        )

        assert time_trigger == expected_trigger
