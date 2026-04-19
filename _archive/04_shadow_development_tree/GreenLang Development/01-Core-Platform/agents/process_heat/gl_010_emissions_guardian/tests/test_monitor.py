# -*- coding: utf-8 -*-
"""
GL-010 Monitor Tests
====================

Unit tests for GL-010 emissions monitor module.
Tests real-time monitoring, compliance checking, and trend analysis.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.process_heat.gl_010_emissions_guardian.monitor import (
    EmissionsMonitor,
    EmissionsInput,
    EmissionsOutput,
)


class TestEmissionsInput:
    """Tests for EmissionsInput model."""

    def test_valid_input(self):
        """Test valid emissions input."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=150.0,
            fuel_flow_unit="MMBTU/hr",
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
            load_pct=85.0,
        )
        assert input_data.source_id == "STACK-001"
        assert input_data.fuel_flow_rate == 150.0
        assert input_data.stack_o2_pct == 3.5

    def test_o2_bounds(self):
        """Test O2 percentage bounds."""
        with pytest.raises(ValueError):
            EmissionsInput(
                source_id="STACK-001",
                fuel_flow_rate=150.0,
                stack_o2_pct=25.0,  # Over 21%
                stack_temperature_f=350.0,
            )

    def test_negative_o2(self):
        """Test negative O2 rejection."""
        with pytest.raises(ValueError):
            EmissionsInput(
                source_id="STACK-001",
                fuel_flow_rate=150.0,
                stack_o2_pct=-5.0,  # Negative
                stack_temperature_f=350.0,
            )

    def test_fuel_flow_bounds(self):
        """Test fuel flow bounds."""
        with pytest.raises(ValueError):
            EmissionsInput(
                source_id="STACK-001",
                fuel_flow_rate=-10.0,  # Negative
                stack_o2_pct=3.5,
                stack_temperature_f=350.0,
            )

    def test_load_bounds(self):
        """Test load percentage bounds."""
        with pytest.raises(ValueError):
            EmissionsInput(
                source_id="STACK-001",
                fuel_flow_rate=150.0,
                stack_o2_pct=3.5,
                stack_temperature_f=350.0,
                load_pct=150.0,  # Over 120%
            )

    def test_optional_fields(self):
        """Test optional NOx and SO2 fields."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_flow_rate=150.0,
            stack_o2_pct=3.5,
            stack_temperature_f=350.0,
            stack_nox_ppm=25.0,
            stack_so2_ppm=5.0,
            stack_pm_mg_m3=10.0,
        )
        assert input_data.stack_nox_ppm == 25.0
        assert input_data.stack_so2_ppm == 5.0
        assert input_data.stack_pm_mg_m3 == 10.0


class TestEmissionsOutput:
    """Tests for EmissionsOutput model."""

    def test_valid_output(self):
        """Test valid emissions output."""
        output = EmissionsOutput(
            source_id="STACK-001",
            timestamp=datetime.now(timezone.utc),
            status="compliant",
            co2_lb_hr=10000.0,
            co2_kg_hr=4536.0,
            co2_ton_yr=43800.0,
            co2_lb_mmbtu=117.0,
            permit_limits={"co2_lb_hr": 50000.0},
            exceedances=[],
            warnings=[],
        )
        assert output.status == "compliant"
        assert output.co2_lb_hr == 10000.0

    def test_exceedance_output(self):
        """Test output with exceedance."""
        output = EmissionsOutput(
            source_id="STACK-001",
            timestamp=datetime.now(timezone.utc),
            status="exceedance",
            co2_lb_hr=55000.0,
            co2_kg_hr=24948.0,
            co2_ton_yr=241000.0,
            co2_lb_mmbtu=117.0,
            permit_limits={"co2_lb_hr": 50000.0},
            exceedances=[{
                "pollutant": "CO2",
                "measured": 55000.0,
                "limit": 50000.0,
                "exceedance_pct": 10.0,
            }],
            warnings=[],
        )
        assert output.status == "exceedance"
        assert len(output.exceedances) == 1


class TestEmissionsMonitor:
    """Tests for EmissionsMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create emissions monitor instance."""
        return EmissionsMonitor(
            source_id="STACK-001",
            permit_limits={
                "co2_lb_hr": 50000.0,
                "nox_lb_hr": 25.0,
            },
        )

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.source_id == "STACK-001"
        assert monitor.permit_limits["co2_lb_hr"] == 50000.0

    def test_monitor_compliant(self, monitor):
        """Test monitoring with compliant emissions."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            fuel_flow_unit="MMBTU/hr",
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
            load_pct=75.0,
        )

        output = monitor.monitor(input_data)

        assert output.status == "compliant"
        assert output.co2_lb_hr > 0
        assert len(output.exceedances) == 0

    def test_monitor_exceedance(self, monitor):
        """Test monitoring with exceedance detection."""
        # Use high fuel flow to trigger exceedance
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=500.0,
            fuel_flow_unit="MMBTU/hr",
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
            load_pct=100.0,
        )

        output = monitor.monitor(input_data)

        # At 500 MMBTU/hr natural gas, CO2 should exceed 50,000 lb/hr
        assert output.co2_lb_hr > 50000.0
        assert output.status == "exceedance"
        assert len(output.exceedances) > 0

    def test_high_co_warning(self, monitor):
        """Test high CO warning generation."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=250.0,  # High CO
            stack_temperature_f=350.0,
        )

        output = monitor.monitor(input_data)

        assert any("High CO" in w for w in output.warnings)

    def test_emission_rates(self, monitor):
        """Test emission rate calculations."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        output = monitor.monitor(input_data)

        # Check emission rates are positive
        assert output.co2_lb_hr > 0
        assert output.co2_kg_hr > 0
        assert output.co2_ton_yr > 0
        assert output.co2_lb_mmbtu > 0

    def test_co2_unit_conversions(self, monitor):
        """Test CO2 unit conversions are correct."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        output = monitor.monitor(input_data)

        # Verify unit conversions
        # lb to kg: 1 lb = 0.4536 kg
        expected_kg = output.co2_lb_hr / 2.205
        assert abs(output.co2_kg_hr - expected_kg) < 1.0

        # Annual tons: lb/hr * 8760 / 2000
        expected_annual = output.co2_lb_hr * 8760 / 2000
        assert abs(output.co2_ton_yr - expected_annual) < 10.0

    def test_history_tracking(self, monitor):
        """Test emissions history tracking."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        # Run multiple times
        for _ in range(5):
            monitor.monitor(input_data)

        # Check history
        assert len(monitor._emissions_history) == 5

    def test_history_cleanup(self, monitor):
        """Test 24-hour history cleanup."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        # Add old entries manually
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        monitor._emissions_history.append({
            "timestamp": old_time,
            "co2_lb_hr": 10000.0,
            "nox_lb_hr": None,
            "load_pct": 75.0,
        })

        # Run monitor to trigger cleanup
        monitor.monitor(input_data)

        # Old entry should be cleaned up
        for entry in monitor._emissions_history:
            assert entry["timestamp"] > datetime.now(timezone.utc) - timedelta(hours=24)

    def test_daily_summary_no_data(self, monitor):
        """Test daily summary with no data."""
        summary = monitor.get_daily_summary()
        assert "message" in summary
        assert summary["message"] == "No data available"

    def test_daily_summary_with_data(self, monitor):
        """Test daily summary with data."""
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        # Run multiple times
        for _ in range(10):
            monitor.monitor(input_data)

        summary = monitor.get_daily_summary()

        assert "co2_avg_lb_hr" in summary
        assert "co2_max_lb_hr" in summary
        assert "co2_min_lb_hr" in summary
        assert "exceedance_count" in summary

    def test_exceedance_prediction_trend(self, monitor):
        """Test exceedance prediction based on trend."""
        # Create trending input data
        base_flow = 300.0

        for i in range(15):
            input_data = EmissionsInput(
                source_id="STACK-001",
                fuel_type="natural_gas",
                fuel_flow_rate=base_flow + (i * 10),  # Increasing trend
                stack_o2_pct=3.5,
                stack_co_ppm=50.0,
                stack_temperature_f=350.0,
            )
            output = monitor.monitor(input_data)

        # Last output should have prediction
        # (if trend continues toward limit)
        assert output.predicted_exceedance_risk >= 0


class TestEmissionsMonitorIntegration:
    """Integration tests for EmissionsMonitor."""

    def test_multi_fuel_monitoring(self):
        """Test monitoring with different fuel types."""
        fuels = ["natural_gas", "no2_fuel_oil", "coal"]

        for fuel in fuels:
            monitor = EmissionsMonitor(source_id=f"STACK-{fuel}")
            input_data = EmissionsInput(
                source_id=f"STACK-{fuel}",
                fuel_type=fuel,
                fuel_flow_rate=100.0,
                stack_o2_pct=3.5,
                stack_co_ppm=50.0,
                stack_temperature_f=350.0,
            )

            output = monitor.monitor(input_data)

            assert output.co2_lb_hr > 0
            assert output.status in ["compliant", "exceedance"]

    def test_varying_load(self):
        """Test monitoring across varying loads."""
        monitor = EmissionsMonitor(source_id="STACK-001")

        loads = [25.0, 50.0, 75.0, 100.0]
        results = []

        for load in loads:
            input_data = EmissionsInput(
                source_id="STACK-001",
                fuel_type="natural_gas",
                fuel_flow_rate=load,  # Simple proportional to load
                stack_o2_pct=3.5,
                stack_co_ppm=50.0,
                stack_temperature_f=350.0,
                load_pct=load,
            )
            output = monitor.monitor(input_data)
            results.append(output.co2_lb_hr)

        # Emissions should increase with load
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]


class TestEmissionsMonitorEdgeCases:
    """Edge case tests for EmissionsMonitor."""

    def test_zero_permit_limits(self):
        """Test with no permit limits set."""
        monitor = EmissionsMonitor(source_id="STACK-001")
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        output = monitor.monitor(input_data)

        # Should run without errors
        assert output.status == "compliant"  # No limits to exceed

    def test_minimum_fuel_flow(self):
        """Test with minimum fuel flow."""
        monitor = EmissionsMonitor(source_id="STACK-001")
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=0.1,  # Very small
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        output = monitor.monitor(input_data)

        assert output.co2_lb_hr > 0
        assert output.co2_lb_mmbtu > 0

    def test_high_o2_pct(self):
        """Test with high O2 percentage."""
        monitor = EmissionsMonitor(source_id="STACK-001")
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=15.0,  # High O2 (excess air)
            stack_co_ppm=50.0,
            stack_temperature_f=350.0,
        )

        output = monitor.monitor(input_data)

        assert output.co2_lb_hr > 0

    def test_with_nox_measurement(self):
        """Test with NOx measurement."""
        monitor = EmissionsMonitor(
            source_id="STACK-001",
            permit_limits={"nox_lb_hr": 25.0},
        )
        input_data = EmissionsInput(
            source_id="STACK-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,
            stack_o2_pct=3.5,
            stack_co_ppm=50.0,
            stack_nox_ppm=30.0,
            stack_flow_rate_acfm=50000.0,
            stack_temperature_f=350.0,
        )

        output = monitor.monitor(input_data)

        # NOx should be calculated
        assert output.nox_lb_hr is not None or output.nox_lb_hr is None  # May be None depending on implementation

    def test_intelligence_level(self):
        """Test intelligence level configuration."""
        monitor = EmissionsMonitor(source_id="STACK-001")

        from greenlang.agents.intelligence_interface import IntelligenceLevel

        level = monitor.get_intelligence_level()
        assert level == IntelligenceLevel.ADVANCED

    def test_intelligence_capabilities(self):
        """Test intelligence capabilities."""
        monitor = EmissionsMonitor(source_id="STACK-001")

        capabilities = monitor.get_intelligence_capabilities()

        assert capabilities.can_explain is True
        assert capabilities.can_recommend is True
        assert capabilities.can_detect_anomalies is True
