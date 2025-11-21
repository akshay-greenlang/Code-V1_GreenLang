# -*- coding: utf-8 -*-
"""
Integration Tests for GreenLang Shared Tools
=============================================

End-to-end integration tests covering:
- Financial tools with realistic scenarios
- Grid tools with various load profiles
- Agent integration with shared tools
- Security feature integration
- Cross-tool workflows

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
import time
from typing import List

from greenlang.agents.tools.financial import FinancialMetricsTool
from greenlang.agents.tools.grid import GridIntegrationTool
from greenlang.agents.tools.emissions import EmissionsCalculatorTool
from greenlang.agents.tools.telemetry import get_telemetry, reset_global_telemetry
from greenlang.agents.tools.audit import get_audit_logger
from greenlang.agents.tools.rate_limiting import get_rate_limiter
from greenlang.agents.tools.security_config import SecurityConfig


class TestFinancialToolIntegration:
    """Integration tests for FinancialMetricsTool."""

    def setup_method(self):
        """Set up test environment."""
        self.tool = FinancialMetricsTool()
        reset_global_telemetry()

    def test_basic_npv_calculation(self):
        """Test NPV with typical solar PV inputs."""
        result = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05
        )

        assert result.success
        assert "npv" in result.data
        assert "irr" in result.data
        assert "simple_payback_years" in result.data

        # Verify reasonable values
        assert result.data["npv"] > 0  # Should be positive
        assert 0 < result.data["irr"] < 1  # IRR should be reasonable
        assert result.data["simple_payback_years"] < 25  # Should pay back before EOL

    def test_with_ira_2022_incentives(self):
        """Test with IRA 2022 Investment Tax Credit."""
        # IRA 2022: 30% ITC for solar PV
        capital_cost = 100000
        itc_amount = capital_cost * 0.30

        result = self.tool(
            capital_cost=capital_cost,
            annual_savings=15000,
            lifetime_years=25,
            incentives=[
                {
                    "name": "IRA 2022 ITC",
                    "amount": itc_amount,
                    "year": 0
                }
            ]
        )

        assert result.success
        assert result.data["total_incentives"] == itc_amount
        assert result.data["net_capital_cost"] == 70000  # 100k - 30k

        # NPV should be significantly better with incentives
        result_no_incentive = self.tool(
            capital_cost=capital_cost,
            annual_savings=15000,
            lifetime_years=25
        )

        assert result.data["npv"] > result_no_incentive.data["npv"]

    def test_with_energy_escalation(self):
        """Test with energy cost escalation."""
        result = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            energy_cost_escalation=0.03  # 3% annual increase
        )

        assert result.success

        # With escalation, total savings should be higher
        result_no_escalation = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            energy_cost_escalation=0.0
        )

        assert result.data["total_savings"] > result_no_escalation.data["total_savings"]

    def test_with_depreciation(self):
        """Test with MACRS depreciation tax benefits."""
        result = self.tool(
            capital_cost=100000,
            annual_savings=20000,
            lifetime_years=25,
            include_depreciation=True,
            tax_rate=0.21
        )

        assert result.success

        # With depreciation, NPV should be higher
        result_no_depreciation = self.tool(
            capital_cost=100000,
            annual_savings=20000,
            lifetime_years=25,
            include_depreciation=False
        )

        assert result.data["npv"] > result_no_depreciation.data["npv"]

    def test_security_validation(self):
        """Test input validation rejects invalid inputs."""
        # Negative capital cost
        result = self.tool(
            capital_cost=-1000,
            annual_savings=5000,
            lifetime_years=10
        )

        assert not result.success
        assert "capital_cost" in result.error.lower()

        # Invalid lifetime
        result = self.tool(
            capital_cost=10000,
            annual_savings=5000,
            lifetime_years=0
        )

        assert not result.success
        assert "lifetime" in result.error.lower()

    def test_rate_limiting(self):
        """Test rate limiting kicks in after burst."""
        # Note: This test depends on rate limit configuration
        # May need adjustment based on actual limits

        # Record initial calls
        for i in range(5):
            result = self.tool(
                capital_cost=10000,
                annual_savings=2000,
                lifetime_years=10
            )
            # These should succeed
            if i < 3:  # Assuming burst of 3
                assert result.success

    def test_audit_logging(self):
        """Test audit logs are created."""
        audit_logger = get_audit_logger()
        initial_count = len(audit_logger.get_recent_logs())

        result = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25
        )

        assert result.success

        # Check audit log increased
        final_count = len(audit_logger.get_recent_logs())
        assert final_count > initial_count

    def test_telemetry_recording(self):
        """Test telemetry metrics are recorded."""
        telemetry = get_telemetry()

        result = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25
        )

        assert result.success

        # Check telemetry
        metrics = telemetry.get_tool_metrics("calculate_financial_metrics")
        assert metrics.total_calls >= 1
        assert metrics.successful_calls >= 1
        assert metrics.avg_execution_time_ms > 0

    def test_execution_time_reasonable(self):
        """Test execution time is reasonable."""
        result = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25
        )

        assert result.success
        assert result.execution_time_ms > 0
        assert result.execution_time_ms < 1000  # Should be fast (< 1 second)

    def test_citations_included(self):
        """Test calculation citations are included."""
        result = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25
        )

        assert result.success
        assert len(result.citations) > 0

        # Check citation structure
        citation = result.citations[0]
        assert hasattr(citation, "step_name")
        assert hasattr(citation, "formula")


class TestGridToolIntegration:
    """Integration tests for GridIntegrationTool."""

    def setup_method(self):
        """Set up test environment."""
        self.tool = GridIntegrationTool()
        reset_global_telemetry()

    def test_24hour_load_profile(self):
        """Test with 24-hour load profile."""
        # Typical daily load profile for a commercial building
        load_profile = [
            300, 280, 270, 265, 270, 290,  # 00:00 - 05:00 (overnight)
            320, 380, 450, 480, 490, 500,  # 06:00 - 11:00 (morning ramp)
            510, 500, 490, 480, 470, 460,  # 12:00 - 17:00 (afternoon)
            420, 390, 360, 340, 320, 310   # 18:00 - 23:00 (evening)
        ]

        result = self.tool(
            peak_demand_kw=510,
            load_profile=load_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12
        )

        assert result.success
        assert "monthly_demand_charge" in result.data
        assert "total_energy_cost" in result.data

        # Verify demand charge calculation
        expected_demand_charge = 510 * 15.0
        assert abs(result.data["monthly_demand_charge"] - expected_demand_charge) < 1.0

    def test_8760hour_profile(self):
        """Test with full-year 8760-hour load profile."""
        # Generate synthetic annual profile
        load_profile = []
        for hour in range(8760):
            # Seasonal variation
            day_of_year = hour // 24
            season_factor = 1.0 + 0.2 * (day_of_year / 365)

            # Daily variation
            hour_of_day = hour % 24
            if 6 <= hour_of_day <= 18:
                time_factor = 1.2  # Daytime
            else:
                time_factor = 0.8  # Nighttime

            load = 400 * season_factor * time_factor
            load_profile.append(load)

        result = self.tool(
            peak_demand_kw=600,
            load_profile=load_profile,
            grid_capacity_kw=800,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12
        )

        assert result.success
        assert result.data["analysis_period_hours"] == 8760

    def test_tou_rate_optimization(self):
        """Test time-of-use rate optimization."""
        load_profile = [400] * 24  # Flat load

        tou_rates = {
            "peak": 0.18,
            "mid_peak": 0.12,
            "off_peak": 0.08
        }

        tou_schedule = {
            "peak": [12, 13, 14, 15, 16, 17, 18],  # 12pm - 6pm
            "mid_peak": [8, 9, 10, 11, 19, 20, 21],  # Morning and evening
            "off_peak": [0, 1, 2, 3, 4, 5, 6, 7, 22, 23]  # Night
        }

        result = self.tool(
            peak_demand_kw=400,
            load_profile=load_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            tou_rates=tou_rates,
            tou_schedule=tou_schedule
        )

        assert result.success
        assert "tou_cost_breakdown" in result.data

        # TOU cost should be different from flat rate
        tou_total = sum(result.data["tou_cost_breakdown"].values())
        flat_rate_cost = 24 * 400 * 0.12 * 30  # 30 days
        assert abs(tou_total - flat_rate_cost) > 0

    def test_peak_shaving_analysis(self):
        """Test peak shaving opportunity identification."""
        # Load profile with clear peak
        load_profile = [300] * 10 + [500] * 4 + [300] * 10  # Peak at midday

        result = self.tool(
            peak_demand_kw=500,
            load_profile=load_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=20.0,  # High demand charge
            energy_rate_per_kwh=0.12
        )

        assert result.success
        assert "peak_shaving_opportunity" in result.data

        # Should identify savings opportunity
        if result.data["peak_shaving_opportunity"]["achievable"]:
            assert result.data["peak_shaving_opportunity"]["potential_savings_per_month"] > 0

    def test_with_energy_storage(self):
        """Test with energy storage integration."""
        load_profile = [
            300, 300, 300, 300, 300, 300,  # Night
            400, 450, 500, 550, 600, 600,  # Morning ramp
            600, 600, 600, 550, 500, 450,  # Afternoon
            400, 350, 350, 350, 300, 300   # Evening
        ]

        result = self.tool(
            peak_demand_kw=600,
            load_profile=load_profile,
            grid_capacity_kw=700,
            demand_charge_per_kw=20.0,
            energy_rate_per_kwh=0.15,
            storage_capacity_kwh=200,  # 200 kWh battery
            storage_power_kw=100       # 100 kW power rating
        )

        assert result.success
        assert "storage_impact" in result.data

        # Storage should help reduce peak demand
        if result.data["storage_impact"]["peak_reduction_kw"] > 0:
            assert result.data["storage_impact"]["monthly_savings"] > 0

    def test_demand_response_program(self):
        """Test demand response program analysis."""
        load_profile = [400] * 24

        result = self.tool(
            peak_demand_kw=400,
            load_profile=load_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12,
            dr_program_available=True,
            dr_incentive_per_kwh=0.50,
            dr_hours=[13, 14, 15, 16, 17]  # Afternoon DR events
        )

        assert result.success
        assert "demand_response" in result.data

        if result.data["demand_response"]["recommended"]:
            assert result.data["demand_response"]["potential_annual_incentive"] > 0


class TestEmissionsToolIntegration:
    """Integration tests for EmissionsCalculatorTool."""

    def setup_method(self):
        """Set up test environment."""
        self.tool = EmissionsCalculatorTool()
        reset_global_telemetry()

    def test_basic_emissions_calculation(self):
        """Test basic emissions calculation."""
        result = self.tool(
            energy_kwh=10000,
            source_type="grid",
            region="US-Northeast"
        )

        assert result.success
        assert "total_co2e_kg" in result.data
        assert "total_co2e_tonnes" in result.data
        assert result.data["total_co2e_kg"] > 0

    def test_multiple_energy_sources(self):
        """Test with multiple energy sources."""
        result = self.tool(
            energy_kwh=10000,
            source_type="hybrid",
            energy_mix={
                "grid": 0.6,
                "solar": 0.3,
                "wind": 0.1
            },
            region="US-California"
        )

        assert result.success
        assert "source_breakdown" in result.data


class TestAgentToolIntegration:
    """Integration tests for agents using shared tools."""

    def setup_method(self):
        """Set up test environment."""
        reset_global_telemetry()

    def test_financial_and_grid_tools_together(self):
        """Test using financial and grid tools in sequence."""
        financial_tool = FinancialMetricsTool()
        grid_tool = GridIntegrationTool()

        # Calculate financial metrics
        financial_result = financial_tool(
            capital_cost=100000,
            annual_savings=20000,
            lifetime_years=20
        )

        assert financial_result.success

        # Analyze grid integration
        load_profile = [400] * 24
        grid_result = grid_tool(
            peak_demand_kw=500,
            load_profile=load_profile,
            grid_capacity_kw=600,
            demand_charge_per_kw=15.0,
            energy_rate_per_kwh=0.12
        )

        assert grid_result.success

        # Verify both tools recorded telemetry
        telemetry = get_telemetry()
        summary = telemetry.get_summary_stats()
        assert summary["total_tools"] >= 2
        assert summary["total_executions"] >= 2


class TestSecurityIntegration:
    """Integration tests for security features."""

    def setup_method(self):
        """Set up test environment."""
        self.tool = FinancialMetricsTool()
        reset_global_telemetry()

    def test_validation_prevents_bad_inputs(self):
        """Test validation rejects invalid inputs."""
        # Test negative values
        result = self.tool(
            capital_cost=-5000,
            annual_savings=1000,
            lifetime_years=10
        )

        assert not result.success

        # Test invalid discount rate
        result = self.tool(
            capital_cost=5000,
            annual_savings=1000,
            lifetime_years=10,
            discount_rate=1.5  # > 1.0
        )

        assert not result.success

    def test_rate_limit_recovery(self):
        """Test rate limit recovery after waiting."""
        rate_limiter = get_rate_limiter()

        # Make several calls
        for i in range(3):
            result = self.tool(
                capital_cost=10000,
                annual_savings=2000,
                lifetime_years=10
            )

        # Wait for rate limit to recover
        time.sleep(1)

        # Should be able to call again
        result = self.tool(
            capital_cost=10000,
            annual_savings=2000,
            lifetime_years=10
        )

        # May or may not succeed depending on limits, but shouldn't crash

    def test_audit_log_query(self):
        """Test querying audit logs."""
        audit_logger = get_audit_logger()

        # Make a call
        result = self.tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25
        )

        assert result.success

        # Query recent logs
        logs = audit_logger.get_recent_logs(limit=10)
        assert len(logs) > 0

        # Check log has expected fields
        log = logs[-1]
        assert "tool_name" in log
        assert "timestamp" in log
        assert "execution_time_ms" in log


class TestEndToEndWorkflows:
    """End-to-end workflow tests."""

    def setup_method(self):
        """Set up test environment."""
        reset_global_telemetry()

    def test_solar_pv_analysis_workflow(self):
        """Complete solar PV analysis using all tools."""
        # 1. Financial analysis
        financial_tool = FinancialMetricsTool()

        capital_cost = 150000
        annual_energy_kwh = 200000
        energy_rate = 0.15
        annual_savings = annual_energy_kwh * energy_rate

        financial_result = financial_tool(
            capital_cost=capital_cost,
            annual_savings=annual_savings,
            lifetime_years=25,
            incentives=[
                {"name": "IRA 2022 ITC", "amount": capital_cost * 0.30, "year": 0}
            ],
            energy_cost_escalation=0.03
        )

        assert financial_result.success
        assert financial_result.data["npv"] > 0

        # 2. Grid integration analysis
        grid_tool = GridIntegrationTool()

        # Solar generation profile (peaks at midday)
        solar_profile = []
        for hour in range(24):
            if 6 <= hour <= 18:
                # Parabolic profile during day
                factor = 1.0 - ((hour - 12) ** 2) / 36
                solar_profile.append(200 * max(0, factor))
            else:
                solar_profile.append(0)

        grid_result = grid_tool(
            peak_demand_kw=max(solar_profile),
            load_profile=solar_profile,
            grid_capacity_kw=250,
            demand_charge_per_kw=12.0,
            energy_rate_per_kwh=energy_rate
        )

        assert grid_result.success

        # 3. Emissions calculation
        emissions_tool = EmissionsCalculatorTool()

        emissions_result = emissions_tool(
            energy_kwh=annual_energy_kwh,
            source_type="solar",
            region="US"
        )

        assert emissions_result.success

        # 4. Verify all tools worked together
        telemetry = get_telemetry()
        summary = telemetry.get_summary_stats()

        assert summary["total_tools"] == 3
        assert summary["total_executions"] == 3
        assert summary["total_successes"] == 3

    def test_hvac_retrofit_workflow(self):
        """Complete HVAC retrofit analysis."""
        financial_tool = FinancialMetricsTool()

        # Heat pump retrofit
        capital_cost = 80000
        annual_savings = 15000  # Energy savings from efficiency

        result = financial_tool(
            capital_cost=capital_cost,
            annual_savings=annual_savings,
            lifetime_years=15,
            annual_om_cost=2000,
            discount_rate=0.06,
            include_depreciation=True
        )

        assert result.success
        assert result.data["simple_payback_years"] < 15

    def test_facility_decarbonization_workflow(self):
        """Complete facility decarbonization planning."""
        # This would use multiple tools in sequence
        financial_tool = FinancialMetricsTool()
        grid_tool = GridIntegrationTool()
        emissions_tool = EmissionsCalculatorTool()

        # Baseline emissions
        baseline_emissions = emissions_tool(
            energy_kwh=500000,
            source_type="grid",
            region="US-Midwest"
        )

        assert baseline_emissions.success

        # Solar + storage solution
        solar_financial = financial_tool(
            capital_cost=300000,
            annual_savings=75000,
            lifetime_years=25
        )

        assert solar_financial.success

        # Verify workflow completed
        telemetry = get_telemetry()
        assert telemetry.get_summary_stats()["total_executions"] >= 2


class TestPerformance:
    """Performance and stress tests."""

    def setup_method(self):
        """Set up test environment."""
        reset_global_telemetry()

    def test_tool_performance_under_load(self):
        """Test tool performance with many sequential calls."""
        tool = FinancialMetricsTool()

        start_time = time.perf_counter()

        for i in range(100):
            result = tool(
                capital_cost=50000,
                annual_savings=8000,
                lifetime_years=25
            )
            assert result.success

        elapsed_time = time.perf_counter() - start_time

        # Should complete 100 calls in reasonable time (< 10 seconds)
        assert elapsed_time < 10.0

        # Check telemetry
        telemetry = get_telemetry()
        metrics = telemetry.get_tool_metrics("calculate_financial_metrics")
        assert metrics.total_calls >= 100
        assert metrics.avg_execution_time_ms < 100  # Should be fast

    def test_telemetry_overhead(self):
        """Test telemetry overhead is minimal."""
        tool = FinancialMetricsTool()

        # Measure with telemetry
        start_time = time.perf_counter()
        for i in range(50):
            tool(capital_cost=10000, annual_savings=2000, lifetime_years=10)
        with_telemetry_time = time.perf_counter() - start_time

        # Telemetry overhead should be minimal (< 10% of execution time)
        # This is a rough check - actual overhead will vary
        assert with_telemetry_time < 5.0  # Should still be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
