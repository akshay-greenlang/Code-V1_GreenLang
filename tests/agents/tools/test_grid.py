"""
Tests for Grid Integration Tool
================================

Comprehensive test suite for GridIntegrationTool including:
- Capacity utilization analysis
- Demand charge calculations
- TOU rate analysis
- Demand response program benefits
- Peak shaving opportunities
- Energy storage optimization

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.agents.tools.grid import GridIntegrationTool


class TestGridIntegrationTool:
    """Test suite for GridIntegrationTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return GridIntegrationTool()

    @pytest.fixture
    def sample_load_profile_24h(self):
        """Sample 24-hour load profile (kW)."""
        return [
            # Night (0-6): Lower loads
            200, 180, 170, 165, 170, 190,
            # Morning (6-12): Rising loads
            250, 300, 350, 400, 420, 450,
            # Afternoon (12-18): Peak loads
            480, 500, 490, 485, 470, 460,
            # Evening (18-24): Declining loads
            440, 400, 350, 300, 250, 220
        ]

    def test_basic_capacity_utilization(self, tool, sample_load_profile_24h):
        """Test basic capacity utilization calculation."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert "capacity_utilization_percent" in result.data
        assert abs(result.data["capacity_utilization_percent"] - 83.33) < 0.1  # 500/600 * 100
        assert result.data["capacity_headroom_kw"] == 100  # 600 - 500

    def test_at_capacity_risk(self, tool, sample_load_profile_24h):
        """Test detection of at-capacity risk."""
        result = tool.execute(
            peak_demand_kw=550,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["at_capacity_risk"]  # > 90%

    def test_demand_charge_calculation(self, tool, sample_load_profile_24h):
        """Test demand charge calculation."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["monthly_demand_charge"] == 7500.0  # 500 * 15

    def test_energy_cost_calculation(self, tool, sample_load_profile_24h):
        """Test energy cost calculation."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert "monthly_energy_cost" in result.data
        assert result.data["monthly_energy_cost"] > 0

        # Daily energy = sum of load profile
        daily_energy = sum(sample_load_profile_24h)
        monthly_energy = daily_energy * 30
        expected_energy_cost = monthly_energy * 0.12

        assert abs(result.data["monthly_energy_cost"] - expected_energy_cost) < 1.0

    def test_total_monthly_cost(self, tool, sample_load_profile_24h):
        """Test total monthly utility cost."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        expected_total = result.data["monthly_demand_charge"] + result.data["monthly_energy_cost"]
        assert abs(result.data["total_monthly_cost"] - expected_total) < 0.01

    def test_tou_rates(self, tool, sample_load_profile_24h):
        """Test time-of-use rate calculations."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            tou_rates={
                "peak": 0.18,
                "off_peak": 0.08
            },
            tou_schedule={
                "peak": [12, 13, 14, 15, 16, 17, 18],
                "off_peak": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23]
            }
        )

        assert result.success
        assert "tou_cost_breakdown" in result.data
        assert "peak" in result.data["tou_cost_breakdown"]
        assert "off_peak" in result.data["tou_cost_breakdown"]

        # TOU cost should be different from flat rate
        result_flat = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        # Cost could be higher or lower depending on load profile
        assert result.data["monthly_energy_cost"] != result_flat.data["monthly_energy_cost"]

    def test_peak_shaving_opportunity(self, tool, sample_load_profile_24h):
        """Test peak shaving opportunity analysis."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert "peak_shaving_opportunity_kw" in result.data
        assert result.data["peak_shaving_opportunity_kw"] >= 0
        assert "peak_shaving_potential_savings" in result.data
        assert result.data["peak_shaving_potential_savings"] >= 0

    def test_demand_response_benefits(self, tool, sample_load_profile_24h):
        """Test demand response program benefit analysis."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            dr_program_available=True,
            dr_incentive_per_kwh=0.50,
            dr_hours=[14, 15, 16, 17, 18],
        )

        assert result.success
        assert result.data["dr_available"]
        assert result.data["dr_potential_savings"] > 0
        assert result.data["dr_average_load_reduction_kw"] > 0

    def test_no_demand_response(self, tool, sample_load_profile_24h):
        """Test when demand response is not available."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            dr_program_available=False,
        )

        assert result.success
        assert not result.data["dr_available"]
        assert result.data["dr_potential_savings"] == 0

    def test_energy_storage_optimization(self, tool, sample_load_profile_24h):
        """Test energy storage optimization analysis."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            storage_capacity_kwh=200,
            storage_power_kw=100,
            tou_rates={"peak": 0.18, "off_peak": 0.08},
            tou_schedule={
                "peak": [12, 13, 14, 15, 16, 17, 18],
                "off_peak": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23]
            },
        )

        assert result.success
        assert result.data["storage_enabled"]
        assert result.data["storage_peak_reduction_kw"] > 0
        assert result.data["storage_annual_savings"] > 0
        assert result.data["storage_arbitrage_value"] > 0

    def test_no_energy_storage(self, tool, sample_load_profile_24h):
        """Test when energy storage is not available."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            storage_capacity_kwh=0,
            storage_power_kw=0,
        )

        assert result.success
        assert not result.data["storage_enabled"]
        assert result.data["storage_peak_reduction_kw"] == 0
        assert result.data["storage_annual_savings"] == 0

    def test_annual_load_profile(self, tool):
        """Test with 8760-hour annual load profile."""
        # Create simplified annual profile
        annual_profile = []
        for month in range(12):
            for day in range(30):
                for hour in range(24):
                    # Simple pattern: higher in summer, higher during day
                    base_load = 300
                    summer_factor = 1.3 if 5 <= month <= 8 else 1.0
                    hour_factor = 1.2 if 10 <= hour <= 18 else 0.8
                    load = base_load * summer_factor * hour_factor
                    annual_profile.append(load)

        # Trim to exactly 8760 hours
        annual_profile = annual_profile[:8760]

        result = tool.execute(
            peak_demand_kw=max(annual_profile),
            load_profile=annual_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["monthly_energy_kwh"] > 0
        assert result.data["total_monthly_cost"] > 0

    def test_invalid_peak_demand(self, tool, sample_load_profile_24h):
        """Test input validation for negative peak demand."""
        result = tool.execute(
            peak_demand_kw=-100,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert not result.success
        assert "peak_demand_kw must be non-negative" in result.error

    def test_invalid_grid_capacity(self, tool, sample_load_profile_24h):
        """Test input validation for invalid grid capacity."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=0,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert not result.success
        assert "grid_capacity_kw must be positive" in result.error

    def test_empty_load_profile(self, tool):
        """Test input validation for empty load profile."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=[],
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert not result.success
        assert "load_profile cannot be empty" in result.error

    def test_invalid_load_profile_length(self, tool):
        """Test input validation for invalid load profile length."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=[100, 200, 300],  # Only 3 hours
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert not result.success
        assert "load_profile must be 24 hours or 8760 hours" in result.error

    def test_different_billing_periods(self, tool, sample_load_profile_24h):
        """Test different billing periods."""
        result_30 = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            billing_period_days=30,
        )

        result_31 = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            billing_period_days=31,
        )

        assert result_30.success
        assert result_31.success
        # 31-day billing should have higher energy cost
        assert result_31.data["monthly_energy_cost"] > result_30.data["monthly_energy_cost"]

    def test_citations_included(self, tool, sample_load_profile_24h):
        """Test that calculation citations are included."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert len(result.citations) > 0
        assert any(c.step_name == "calculate_capacity_utilization" for c in result.citations)
        assert any(c.step_name == "calculate_demand_charge" for c in result.citations)

    def test_metadata_included(self, tool, sample_load_profile_24h):
        """Test that metadata is included in result."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            grid_region="US-WECC",
        )

        assert result.success
        assert "grid_region" in result.metadata
        assert result.metadata["grid_region"] == "US-WECC"
        assert "summary" in result.metadata

    def test_tool_definition(self, tool):
        """Test that tool definition is valid."""
        tool_def = tool.get_tool_def()

        assert tool_def.name == "analyze_grid_integration"
        assert tool_def.safety.value == "idempotent"
        assert "peak_demand_kw" in tool_def.parameters["properties"]
        assert "load_profile" in tool_def.parameters["properties"]
        assert "grid_capacity_kw" in tool_def.parameters["properties"]

    def test_realistic_commercial_building(self, tool):
        """Test realistic commercial building load profile."""
        # Office building profile (weekday)
        load_profile = [
            # Night (0-6): Minimal HVAC
            150, 140, 135, 135, 140, 160,
            # Morning ramp-up (6-9)
            220, 320, 420,
            # Business hours (9-18)
            450, 460, 470, 480, 490, 485, 480, 470, 460, 450,
            # Evening ramp-down (18-24)
            380, 300, 230, 180, 160, 150
        ]

        result = tool.execute(
            peak_demand_kw=490,
            load_profile=load_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=18.50,
            energy_rate_per_kwh=0.14,
            tou_rates={"peak": 0.22, "off_peak": 0.10},
            tou_schedule={
                "peak": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                "off_peak": [0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23]
            },
            dr_program_available=True,
            dr_incentive_per_kwh=0.75,
        )

        assert result.success
        assert result.data["capacity_utilization_percent"] > 80
        assert result.data["total_monthly_cost"] > 10000  # Typical commercial cost
        assert result.data["dr_potential_savings"] > 0

    def test_realistic_manufacturing_facility(self, tool):
        """Test realistic manufacturing facility with high constant load."""
        # Manufacturing with 24/7 operations
        load_profile = [800] * 24  # Constant high load

        result = tool.execute(
            peak_demand_kw=800,
            load_profile=load_profile,
            grid_capacity_kw=1000,
            demand_charge_per_kw=20.0,
            energy_rate_per_kwh=0.11,
            storage_capacity_kwh=500,
            storage_power_kw=250,
        )

        assert result.success
        assert result.data["capacity_utilization_percent"] == 80
        assert result.data["monthly_demand_charge"] == 16000  # 800 * 20
        assert result.data["peak_shaving_opportunity_kw"] < 50  # Limited opportunity with flat load

    def test_realistic_data_center(self, tool):
        """Test realistic data center with very high load factor."""
        # Data center with high baseline and some variation
        load_profile = []
        for hour in range(24):
            # Base load 900 kW, +/- 5%
            import math
            variation = math.sin(hour / 24 * 2 * math.pi) * 0.05
            load = 900 * (1 + variation)
            load_profile.append(load)

        result = tool.execute(
            peak_demand_kw=max(load_profile),
            load_profile=load_profile,
            grid_capacity_kw=1200,
            demand_charge_per_kw=25.0,
            energy_rate_per_kwh=0.10,
            dr_program_available=False,  # Can't reduce load
            storage_capacity_kwh=1000,
            storage_power_kw=500,
        )

        assert result.success
        assert result.data["capacity_utilization_percent"] > 75
        assert result.data["average_load_kw"] > 890

    def test_tool_execution_metrics(self, tool, sample_load_profile_24h):
        """Test that execution metrics are tracked."""
        result = tool.execute(
            peak_demand_kw=500,
            load_profile=sample_load_profile_24h,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.execution_time_ms > 0

        # Check tool stats
        stats = tool.get_stats()
        assert stats["executions"] >= 1
        assert stats["total_time_ms"] > 0


class TestGridIntegrationEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return GridIntegrationTool()

    def test_zero_demand_charge(self, tool):
        """Test with zero demand charge."""
        load_profile = [100] * 24

        result = tool.execute(
            peak_demand_kw=100,
            load_profile=load_profile,
            grid_capacity_kw=200,
            demand_charge_per_kw=0.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["monthly_demand_charge"] == 0

    def test_at_full_capacity(self, tool):
        """Test when operating at 100% capacity."""
        load_profile = [500] * 24

        result = tool.execute(
            peak_demand_kw=500,
            load_profile=load_profile,
            grid_capacity_kw=500,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["capacity_utilization_percent"] == 100
        assert result.data["capacity_headroom_kw"] == 0
        assert result.data["at_capacity_risk"]

    def test_minimal_load_variation(self, tool):
        """Test with minimal load variation (flat profile)."""
        load_profile = [300] * 24

        result = tool.execute(
            peak_demand_kw=300,
            load_profile=load_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["peak_shaving_opportunity_kw"] < 10  # Very small opportunity

    def test_extreme_peak_variation(self, tool):
        """Test with extreme peak variation."""
        load_profile = [100] * 23 + [1000]  # One extreme peak

        result = tool.execute(
            peak_demand_kw=1000,
            load_profile=load_profile,
            grid_capacity_kw=1200,
            demand_charge_per_kw=20.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["peak_shaving_opportunity_kw"] > 100  # Large opportunity

    def test_very_low_utilization(self, tool):
        """Test with very low capacity utilization."""
        load_profile = [50] * 24

        result = tool.execute(
            peak_demand_kw=50,
            load_profile=load_profile,
            grid_capacity_kw=1000,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
        )

        assert result.success
        assert result.data["capacity_utilization_percent"] == 5  # Very low
        assert not result.data["at_capacity_risk"]
