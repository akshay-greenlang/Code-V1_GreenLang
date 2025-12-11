"""
GL-078: Tariff Optimizer Agent - Test Suite

Comprehensive test coverage for TariffOptimizerAgent including:
- Input validation tests
- TOU cost calculation tests
- Demand charge calculation tests
- Load shifting analysis tests
- Tariff comparison tests
- Provenance tracking tests

Test Coverage Target: 85%+
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, List

from .agent import (
    TariffOptimizerAgent,
    TariffOptimizerInput,
    TariffOptimizerOutput,
    UsageProfile,
    TariffOption,
    RateSchedule,
    TariffRecommendation,
    DemandChargeAnalysis,
    LoadShiftOpportunity,
    SavingsAnalysis,
    RateType,
    SeasonType,
    LoadType,
)
from .formulas import (
    calculate_tou_cost,
    calculate_flat_rate_cost,
    calculate_tiered_cost,
    calculate_demand_charge,
    calculate_optimal_shift,
    calculate_annual_savings,
    calculate_peak_shaving_benefit,
    calculate_load_factor,
    calculate_battery_shift_value,
    TOUCostResult,
    DemandChargeResult,
    LoadShiftResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create TariffOptimizerAgent instance."""
    return TariffOptimizerAgent()


@pytest.fixture
def basic_usage_profile():
    """Create basic usage profile."""
    # 24-hour profile with typical commercial pattern
    hourly_kwh = [
        50, 45, 42, 40, 42, 50,  # 12am-6am (off-peak)
        80, 120, 150, 160, 165, 170,  # 6am-12pm (ramp up)
        175, 180, 185, 190,  # 12pm-4pm (mid-peak)
        200, 195, 190, 180, 160,  # 4pm-9pm (on-peak)
        130, 90, 70  # 9pm-12am (ramp down)
    ]
    return UsageProfile(
        utility="TEST_UTILITY",
        hourly_kwh=hourly_kwh,
        peak_demand_kw=200,
        load_factor=0.65,
        shiftable_load_kw=30,
        shiftable_load_types=[LoadType.HVAC],
    )


@pytest.fixture
def flat_rate_schedule():
    """Create flat rate schedule."""
    return RateSchedule(
        rate_id="FLAT-01",
        name="Commercial Flat Rate",
        rate_type=RateType.FLAT,
        energy_charge_flat=0.12,
        demand_charge_facility=15.00,
        customer_charge_monthly=50.00,
    )


@pytest.fixture
def tou_rate_schedule():
    """Create TOU rate schedule."""
    return RateSchedule(
        rate_id="TOU-01",
        name="Commercial TOU Rate",
        rate_type=RateType.TOU,
        energy_charge_on_peak=0.25,
        energy_charge_mid_peak=0.15,
        energy_charge_off_peak=0.08,
        demand_charge_facility=12.00,
        demand_charge_on_peak=8.00,
        on_peak_hours=list(range(16, 21)),
        mid_peak_hours=list(range(12, 16)) + list(range(21, 24)),
        customer_charge_monthly=75.00,
    )


@pytest.fixture
def basic_input(basic_usage_profile, flat_rate_schedule, tou_rate_schedule):
    """Create basic input for testing."""
    return TariffOptimizerInput(
        usage_profile=basic_usage_profile,
        available_tariffs=[
            TariffOption(
                rate_schedule=flat_rate_schedule,
                season=SeasonType.SUMMER,
                is_current=True,
            ),
            TariffOption(
                rate_schedule=tou_rate_schedule,
                season=SeasonType.SUMMER,
                is_current=False,
            ),
        ],
    )


# =============================================================================
# AGENT INITIALIZATION TESTS
# =============================================================================

class TestAgentInitialization:
    """Test agent initialization."""

    def test_agent_creates_successfully(self, agent):
        """Test agent creates with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "GL-078"
        assert agent.AGENT_NAME == "TARIFFOPTIMIZER"

    def test_agent_with_custom_config(self):
        """Test agent with custom configuration."""
        config = {"custom_setting": "value"}
        agent = TariffOptimizerAgent(config=config)
        assert agent.config == config

    def test_agent_constants(self, agent):
        """Test agent constants are set."""
        assert agent.VERSION == "1.0.0"
        assert agent.DESCRIPTION == "Utility Tariff Optimization Agent"


# =============================================================================
# INPUT MODEL TESTS
# =============================================================================

class TestInputModels:
    """Test input model validation."""

    def test_valid_usage_profile(self):
        """Test valid usage profile creation."""
        profile = UsageProfile(
            hourly_kwh=[100] * 24,
            peak_demand_kw=150,
        )
        assert profile.peak_demand_kw == 150

    def test_usage_requires_24_hours_minimum(self):
        """Test hourly data requires at least 24 values."""
        with pytest.raises(ValueError):
            UsageProfile(
                hourly_kwh=[100] * 12,  # Only 12 hours
                peak_demand_kw=100,
            )

    def test_peak_demand_must_be_positive(self):
        """Test peak demand validation."""
        with pytest.raises(ValueError):
            UsageProfile(
                hourly_kwh=[100] * 24,
                peak_demand_kw=-50,
            )

    def test_load_factor_must_be_0_to_1(self):
        """Test load factor validation."""
        with pytest.raises(ValueError):
            UsageProfile(
                hourly_kwh=[100] * 24,
                peak_demand_kw=100,
                load_factor=1.5,
            )

    def test_valid_rate_schedule(self):
        """Test valid rate schedule creation."""
        rate = RateSchedule(
            rate_id="TEST-01",
            name="Test Rate",
            rate_type=RateType.TOU,
            energy_charge_on_peak=0.20,
        )
        assert rate.rate_type == RateType.TOU

    def test_rate_schedule_energy_charges_positive(self):
        """Test energy charges must be positive."""
        with pytest.raises(ValueError):
            RateSchedule(
                rate_id="TEST",
                name="Test",
                rate_type=RateType.FLAT,
                energy_charge_flat=-0.10,
            )


# =============================================================================
# TOU COST CALCULATION TESTS
# =============================================================================

class TestTOUCostCalculation:
    """Test TOU cost calculations."""

    def test_tou_cost_calculation(self):
        """Test basic TOU cost calculation."""
        # Simple 24-hour profile
        hourly_kwh = [10] * 24  # 10 kWh every hour

        result = calculate_tou_cost(
            hourly_kwh=hourly_kwh,
            on_peak_rate=0.25,
            mid_peak_rate=0.15,
            off_peak_rate=0.08,
        )

        assert isinstance(result, TOUCostResult)
        assert result.total_energy_cost > 0
        assert result.on_peak_kwh + result.mid_peak_kwh + result.off_peak_kwh == 240

    def test_tou_cost_with_custom_periods(self):
        """Test TOU with custom peak periods."""
        hourly_kwh = [100] * 24

        result = calculate_tou_cost(
            hourly_kwh=hourly_kwh,
            on_peak_rate=0.30,
            mid_peak_rate=0.20,
            off_peak_rate=0.10,
            on_peak_hours=[14, 15, 16, 17, 18, 19],  # 2pm-8pm
            mid_peak_hours=[10, 11, 12, 13, 20, 21],  # 10am-2pm, 8pm-10pm
        )

        # 6 on-peak hours, 6 mid-peak hours, 12 off-peak hours
        assert result.on_peak_kwh == 600
        assert result.mid_peak_kwh == 600
        assert result.off_peak_kwh == 1200

    def test_tou_cost_annual_profile(self):
        """Test TOU with annual (8760 hour) profile."""
        hourly_kwh = [50] * 8760  # 50 kWh every hour for year

        result = calculate_tou_cost(
            hourly_kwh=hourly_kwh,
            on_peak_rate=0.25,
            mid_peak_rate=0.15,
            off_peak_rate=0.08,
        )

        total_kwh = 50 * 8760
        assert result.on_peak_kwh + result.mid_peak_kwh + result.off_peak_kwh == total_kwh


# =============================================================================
# FLAT AND TIERED RATE TESTS
# =============================================================================

class TestFlatAndTieredRates:
    """Test flat and tiered rate calculations."""

    def test_flat_rate_calculation(self):
        """Test flat rate calculation."""
        cost = calculate_flat_rate_cost(10000, 0.12)
        assert cost == 1200.00

    def test_tiered_rate_calculation(self):
        """Test tiered rate calculation."""
        # Tiers: 0-500 kWh @ $0.10, 500-1000 @ $0.15, 1000+ @ $0.20
        cost = calculate_tiered_cost(
            total_kwh=1500,
            tier_limits=[500, 1000],
            tier_rates=[0.10, 0.15, 0.20],
        )

        # 500 * 0.10 + 500 * 0.15 + 500 * 0.20 = 50 + 75 + 100 = 225
        assert cost == 225.00

    def test_tiered_under_first_tier(self):
        """Test consumption under first tier limit."""
        cost = calculate_tiered_cost(
            total_kwh=300,
            tier_limits=[500, 1000],
            tier_rates=[0.10, 0.15, 0.20],
        )
        assert cost == 30.00  # 300 * 0.10


# =============================================================================
# DEMAND CHARGE TESTS
# =============================================================================

class TestDemandCharges:
    """Test demand charge calculations."""

    def test_facility_demand_charge(self):
        """Test facility demand charge calculation."""
        result = calculate_demand_charge(
            peak_demand_kw=200,
            facility_demand_rate=15.00,
        )

        assert result.facility_demand_charge == 3000.00
        assert result.peak_demand_kw == 200

    def test_combined_demand_charges(self):
        """Test combined facility and TOU demand charges."""
        result = calculate_demand_charge(
            peak_demand_kw=200,
            facility_demand_rate=12.00,
            on_peak_demand_kw=180,
            on_peak_demand_rate=8.00,
        )

        assert result.facility_demand_charge == 2400.00
        assert result.tou_demand_charge == 1440.00
        assert result.total_demand_charge == 3840.00

    def test_demand_with_default_on_peak(self):
        """Test demand calculation with default on-peak estimation."""
        result = calculate_demand_charge(
            peak_demand_kw=100,
            facility_demand_rate=10.00,
            on_peak_demand_rate=5.00,
        )

        # Default on-peak = 90% of peak
        assert result.on_peak_demand_kw == 90.0


# =============================================================================
# LOAD SHIFTING TESTS
# =============================================================================

class TestLoadShifting:
    """Test load shifting calculations."""

    def test_optimal_shift_calculation(self):
        """Test optimal load shift calculation."""
        result = calculate_optimal_shift(
            on_peak_kwh=1000,
            mid_peak_kwh=800,
            off_peak_kwh=1200,
            on_peak_rate=0.25,
            mid_peak_rate=0.15,
            off_peak_rate=0.08,
            max_shiftable_percent=0.30,
        )

        assert isinstance(result, LoadShiftResult)
        assert result.shifted_kwh == 300  # 30% of 1000
        assert result.savings > 0  # Should save money

    def test_no_shift_when_rates_equal(self):
        """Test no savings when all rates equal."""
        result = calculate_optimal_shift(
            on_peak_kwh=1000,
            mid_peak_kwh=800,
            off_peak_kwh=1200,
            on_peak_rate=0.10,
            mid_peak_rate=0.10,
            off_peak_rate=0.10,
        )

        assert result.savings == 0
        assert result.effective_rate_reduction == 0

    def test_battery_shift_value(self):
        """Test battery load shifting value calculation."""
        annual_savings, spread = calculate_battery_shift_value(
            battery_capacity_kwh=100,
            on_peak_rate=0.25,
            off_peak_rate=0.08,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        )

        assert annual_savings > 0
        assert spread > 0


# =============================================================================
# SAVINGS CALCULATION TESTS
# =============================================================================

class TestSavingsCalculations:
    """Test savings calculations."""

    def test_annual_savings_calculation(self):
        """Test annual savings calculation."""
        savings, percent = calculate_annual_savings(
            current_annual_cost=100000,
            proposed_annual_cost=85000,
        )

        assert savings == 15000
        assert percent == 15.0

    def test_zero_current_cost(self):
        """Test with zero current cost."""
        savings, percent = calculate_annual_savings(
            current_annual_cost=0,
            proposed_annual_cost=1000,
        )

        assert savings == -1000
        assert percent == 0  # Can't calculate percentage

    def test_peak_shaving_benefit(self):
        """Test peak shaving benefit calculation."""
        annual_savings, reduction = calculate_peak_shaving_benefit(
            current_peak_kw=200,
            reduced_peak_kw=180,
            demand_charge_rate=15.00,
        )

        # 20 kW reduction * $15/kW * 12 months = $3,600
        assert annual_savings == 3600.00
        assert reduction == 20.0


# =============================================================================
# LOAD FACTOR TESTS
# =============================================================================

class TestLoadFactor:
    """Test load factor calculations."""

    def test_load_factor_calculation(self):
        """Test load factor calculation."""
        lf = calculate_load_factor(
            total_kwh=175200,  # 20 kW avg * 8760 hours
            peak_demand_kw=100,
            hours_in_period=8760,
        )

        assert lf == 0.2  # 20 / 100 = 0.2

    def test_load_factor_max_1(self):
        """Test load factor capped at 1.0."""
        lf = calculate_load_factor(
            total_kwh=10000,
            peak_demand_kw=1,  # Unrealistically low peak
            hours_in_period=100,
        )

        assert lf == 1.0

    def test_load_factor_zero_peak(self):
        """Test load factor with zero peak."""
        lf = calculate_load_factor(
            total_kwh=1000,
            peak_demand_kw=0,
            hours_in_period=100,
        )

        assert lf == 0.0


# =============================================================================
# RUN METHOD TESTS
# =============================================================================

class TestRunMethod:
    """Test the main run method."""

    def test_run_returns_output(self, agent, basic_input):
        """Test run returns TariffOptimizerOutput."""
        result = agent.run(basic_input)
        assert isinstance(result, TariffOptimizerOutput)

    def test_run_has_analysis_id(self, agent, basic_input):
        """Test output has analysis ID."""
        result = agent.run(basic_input)
        assert result.analysis_id.startswith("TARIFF-")

    def test_run_has_recommendations(self, agent, basic_input):
        """Test run generates recommendations."""
        result = agent.run(basic_input)
        assert len(result.tariff_recommendations) > 0

    def test_run_has_optimal_tariff(self, agent, basic_input):
        """Test run identifies optimal tariff."""
        result = agent.run(basic_input)
        assert result.optimal_tariff is not None
        assert result.optimal_tariff.rank == 1

    def test_run_has_demand_analysis(self, agent, basic_input):
        """Test run includes demand analysis."""
        result = agent.run(basic_input)
        assert result.demand_analysis is not None
        assert result.demand_analysis.current_peak_kw > 0

    def test_run_has_savings_analysis(self, agent, basic_input):
        """Test run includes savings analysis."""
        result = agent.run(basic_input)
        assert result.savings_analysis is not None

    def test_run_tracks_processing_time(self, agent, basic_input):
        """Test run tracks processing time."""
        result = agent.run(basic_input)
        assert result.processing_time_ms > 0


# =============================================================================
# PROVENANCE TRACKING TESTS
# =============================================================================

class TestProvenanceTracking:
    """Test provenance tracking."""

    def test_provenance_chain_exists(self, agent, basic_input):
        """Test provenance chain is created."""
        result = agent.run(basic_input)
        assert len(result.provenance_chain) > 0

    def test_provenance_has_operations(self, agent, basic_input):
        """Test provenance records operations."""
        result = agent.run(basic_input)
        operations = [p.operation for p in result.provenance_chain]
        assert "usage_analysis" in operations
        assert "tariff_costing" in operations

    def test_provenance_hash_is_sha256(self, agent, basic_input):
        """Test provenance hash is SHA-256."""
        result = agent.run(basic_input)
        assert len(result.provenance_hash) == 64

    def test_provenance_has_timestamps(self, agent, basic_input):
        """Test provenance records have timestamps."""
        result = agent.run(basic_input)
        for record in result.provenance_chain:
            assert isinstance(record.timestamp, datetime)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_tariff(self, agent, basic_usage_profile, flat_rate_schedule):
        """Test with single tariff option."""
        input_data = TariffOptimizerInput(
            usage_profile=basic_usage_profile,
            available_tariffs=[
                TariffOption(
                    rate_schedule=flat_rate_schedule,
                    is_current=True,
                ),
            ],
        )
        result = agent.run(input_data)
        assert len(result.tariff_recommendations) == 1

    def test_no_shiftable_load(self, agent, flat_rate_schedule, tou_rate_schedule):
        """Test with no shiftable load."""
        profile = UsageProfile(
            hourly_kwh=[100] * 24,
            peak_demand_kw=150,
            shiftable_load_kw=0,
        )
        input_data = TariffOptimizerInput(
            usage_profile=profile,
            available_tariffs=[
                TariffOption(rate_schedule=flat_rate_schedule, is_current=True),
                TariffOption(rate_schedule=tou_rate_schedule),
            ],
        )
        result = agent.run(input_data)
        assert result.savings_analysis.load_shift_savings_usd == 0

    def test_annual_usage_profile(self, agent, flat_rate_schedule):
        """Test with full annual (8760 hour) profile."""
        profile = UsageProfile(
            hourly_kwh=[100] * 8760,
            peak_demand_kw=150,
        )
        input_data = TariffOptimizerInput(
            usage_profile=profile,
            available_tariffs=[
                TariffOption(rate_schedule=flat_rate_schedule, is_current=True),
            ],
        )
        result = agent.run(input_data)
        assert isinstance(result, TariffOptimizerOutput)


# =============================================================================
# OUTPUT VALIDATION TESTS
# =============================================================================

class TestOutputValidation:
    """Test output model validation."""

    def test_output_has_required_fields(self, agent, basic_input):
        """Test output has all required fields."""
        result = agent.run(basic_input)
        assert hasattr(result, 'analysis_id')
        assert hasattr(result, 'optimal_tariff')
        assert hasattr(result, 'tariff_recommendations')
        assert hasattr(result, 'provenance_hash')

    def test_recommendations_are_ranked(self, agent, basic_input):
        """Test recommendations are properly ranked."""
        result = agent.run(basic_input)
        for i, rec in enumerate(result.tariff_recommendations):
            assert rec.rank == i + 1

    def test_confidence_scores_valid(self, agent, basic_input):
        """Test confidence scores are 0-1."""
        result = agent.run(basic_input)
        for rec in result.tariff_recommendations:
            assert 0 <= rec.confidence_score <= 1


# =============================================================================
# PACK_SPEC TESTS
# =============================================================================

class TestPackSpec:
    """Test PACK_SPEC configuration."""

    def test_pack_spec_exists(self):
        """Test PACK_SPEC is defined."""
        from .agent import PACK_SPEC
        assert PACK_SPEC is not None

    def test_pack_spec_has_required_fields(self):
        """Test PACK_SPEC has required fields."""
        from .agent import PACK_SPEC
        assert PACK_SPEC["id"] == "GL-078"
        assert PACK_SPEC["name"]
        assert PACK_SPEC["version"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
