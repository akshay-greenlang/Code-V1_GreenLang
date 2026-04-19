# -*- coding: utf-8 -*-
"""
GL-007 Schema Tests
===================

Unit tests for GL-007 data schemas module.
Tests Pydantic models for furnace and cooling tower data.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_007_furnace_optimizer.schemas import (
    FurnaceReading,
    CoolingTowerReading,
    FlueGasAnalysis,
    ZoneTemperature,
    TubeMetalTemperature,
    FanReading,
    WaterQuality,
    CombustionAnalysis,
    HeatTransferAnalysis,
    FurnaceOptimizationResult,
    CoolingTowerOptimizationResult,
    OptimizationResult,
    OptimizationRecommendation,
    SafetyStatus,
    ValidationStatus,
    OperatingMode,
    OptimizationStatus,
    CombustionStatus,
)


class TestFlueGasAnalysis:
    """Tests for flue gas analysis schema."""

    def test_valid_analysis(self, sample_flue_gas_analysis):
        """Test valid flue gas analysis."""
        analysis = sample_flue_gas_analysis
        assert analysis.o2_pct == 3.5
        assert analysis.excess_air_pct == 20.0

    def test_excess_air_calculation(self):
        """Test excess air auto-calculation from O2."""
        analysis = FlueGasAnalysis(o2_pct=3.0)
        # Excess air = O2 / (21 - O2) * 100 = 3 / 18 * 100 = 16.67%
        assert abs(analysis.excess_air_pct - 16.7) < 0.5

    def test_o2_bounds(self):
        """Test O2 percentage bounds."""
        with pytest.raises(ValidationError):
            FlueGasAnalysis(o2_pct=25.0)  # Must be <= 21


class TestZoneTemperature:
    """Tests for zone temperature schema."""

    def test_valid_zone(self, sample_zone_temperature):
        """Test valid zone temperature."""
        zone = sample_zone_temperature
        assert zone.zone_id == "ZONE-1"
        assert zone.temperature_f == 1750.0

    def test_deviation_calculation(self):
        """Test deviation auto-calculation."""
        zone = ZoneTemperature(
            zone_id="ZONE-1",
            temperature_f=1750.0,
            setpoint_f=1800.0,
        )
        assert zone.deviation_f == -50.0


class TestTubeMetalTemperature:
    """Tests for tube metal temperature schema."""

    def test_valid_tmt(self, sample_tmt_reading):
        """Test valid TMT reading."""
        tmt = sample_tmt_reading
        assert tmt.sensor_id == "TMT-001"
        assert tmt.temperature_f == 1400.0

    def test_margin_calculation(self):
        """Test margin auto-calculation."""
        tmt = TubeMetalTemperature(
            sensor_id="TMT-001",
            temperature_f=1400.0,
            design_limit_f=1500.0,
        )
        assert tmt.margin_f == 100.0

    def test_safety_status_safe(self):
        """Test safe status when margin is large."""
        tmt = TubeMetalTemperature(
            sensor_id="TMT-001",
            temperature_f=1300.0,
            design_limit_f=1500.0,
        )
        assert tmt.status == SafetyStatus.SAFE

    def test_safety_status_warning(self):
        """Test warning status when margin is small."""
        tmt = TubeMetalTemperature(
            sensor_id="TMT-001",
            temperature_f=1425.0,
            design_limit_f=1500.0,
        )
        # Margin = 75F, which is in warning range
        assert tmt.status == SafetyStatus.WARNING

    def test_safety_status_violation(self):
        """Test violation status when over limit."""
        tmt = TubeMetalTemperature(
            sensor_id="TMT-001",
            temperature_f=1550.0,
            design_limit_f=1500.0,
        )
        assert tmt.status == SafetyStatus.VIOLATION


class TestFurnaceReading:
    """Tests for furnace reading schema."""

    def test_valid_reading(self, sample_furnace_reading):
        """Test valid furnace reading."""
        reading = sample_furnace_reading
        assert reading.furnace_id == "FUR-001"
        assert reading.furnace_temp_f == 1750.0

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            FurnaceReading()

    def test_excess_air_calculation(self):
        """Test excess air auto-calculation from O2."""
        reading = FurnaceReading(
            furnace_id="FUR-001",
            furnace_temp_f=1800.0,
            fuel_flow_rate_scfh=5000.0,
            flue_gas_temp_f=450.0,
            flue_gas_o2_pct=3.0,
        )
        # Excess air = 3 / (21-3) * 100 = 16.67%
        assert abs(reading.excess_air_pct - 16.7) < 0.5


class TestFanReading:
    """Tests for fan reading schema."""

    def test_valid_fan(self, sample_fan_reading):
        """Test valid fan reading."""
        fan = sample_fan_reading
        assert fan.fan_id == "FAN-1"
        assert fan.speed_pct == 75.0

    def test_speed_bounds(self):
        """Test speed percentage bounds."""
        with pytest.raises(ValidationError):
            FanReading(fan_id="FAN-1", speed_pct=150.0)  # Must be <= 110


class TestWaterQuality:
    """Tests for water quality schema."""

    def test_valid_quality(self, sample_water_quality):
        """Test valid water quality."""
        quality = sample_water_quality
        assert quality.ph == 7.5
        assert quality.cycles_of_concentration == 5.0

    def test_ph_bounds(self):
        """Test pH bounds."""
        with pytest.raises(ValidationError):
            WaterQuality(ph=3.0)  # Must be >= 5


class TestCoolingTowerReading:
    """Tests for cooling tower reading schema."""

    def test_valid_reading(self, sample_cooling_tower_reading):
        """Test valid cooling tower reading."""
        reading = sample_cooling_tower_reading
        assert reading.tower_id == "CT-001"
        assert reading.hot_water_temp_f == 100.0

    def test_range_calculation(self):
        """Test range auto-calculation."""
        reading = CoolingTowerReading(
            tower_id="CT-001",
            ambient_dry_bulb_f=90.0,
            ambient_wet_bulb_f=75.0,
            hot_water_temp_f=100.0,
            cold_water_temp_f=85.0,
            water_flow_gpm=5000.0,
        )
        assert reading.range_f == 15.0

    def test_approach_calculation(self):
        """Test approach auto-calculation."""
        reading = CoolingTowerReading(
            tower_id="CT-001",
            ambient_dry_bulb_f=90.0,
            ambient_wet_bulb_f=75.0,
            hot_water_temp_f=100.0,
            cold_water_temp_f=82.0,
            water_flow_gpm=5000.0,
        )
        # Approach = cold - wet bulb = 82 - 75 = 7
        assert reading.approach_f == 7.0

    def test_heat_rejection_tons(self):
        """Test heat rejection tons calculation."""
        reading = CoolingTowerReading(
            tower_id="CT-001",
            ambient_dry_bulb_f=90.0,
            ambient_wet_bulb_f=75.0,
            hot_water_temp_f=100.0,
            cold_water_temp_f=85.0,
            water_flow_gpm=5000.0,
            heat_rejection_mmbtu_hr=24.0,
        )
        # 24 MMBtu/hr / 0.012 = 2000 tons
        assert abs(reading.heat_rejection_tons - 2000.0) < 1.0


class TestCombustionAnalysis:
    """Tests for combustion analysis schema."""

    def test_valid_analysis(self, sample_combustion_analysis):
        """Test valid combustion analysis."""
        analysis = sample_combustion_analysis
        assert analysis.combustion_status == CombustionStatus.OPTIMAL
        assert analysis.thermal_efficiency_pct == 85.0

    def test_efficiency_bounds(self):
        """Test efficiency bounds."""
        with pytest.raises(ValidationError):
            CombustionAnalysis(
                stoichiometric_air_scf_per_scf_fuel=9.5,
                actual_air_scf_per_scf_fuel=11.0,
                excess_air_pct=15.0,
                air_fuel_ratio=17.2,
                combustion_status=CombustionStatus.OPTIMAL,
                co2_pct_dry=9.5,
                o2_pct_dry=3.0,
                n2_pct_dry=77.5,
                h2o_pct_wet=10.0,
                heat_input_mmbtu_hr=51.0,
                heat_available_mmbtu_hr=43.5,
                dry_flue_gas_loss_pct=8.5,
                moisture_loss_pct=4.5,
                radiation_loss_pct=2.0,
                total_losses_pct=15.0,
                combustion_efficiency_pct=110.0,  # Invalid: > 100
                thermal_efficiency_pct=85.0,
                co2_lb_mmbtu=117.0,
            )


class TestHeatTransferAnalysis:
    """Tests for heat transfer analysis schema."""

    def test_valid_analysis(self, sample_heat_transfer_analysis):
        """Test valid heat transfer analysis."""
        analysis = sample_heat_transfer_analysis
        assert analysis.design_duty_mmbtu_hr == 50.0
        assert analysis.duty_ratio_pct == 87.0


class TestFurnaceOptimizationResult:
    """Tests for furnace optimization result schema."""

    def test_valid_result(self, sample_furnace_optimization_result):
        """Test valid optimization result."""
        result = sample_furnace_optimization_result
        assert result.furnace_id == "FUR-001"
        assert result.status == OptimizationStatus.SUCCESS
        assert result.efficiency_improvement_pct == 3.0

    def test_savings_calculations(self, sample_furnace_optimization_result):
        """Test savings are present."""
        result = sample_furnace_optimization_result
        assert result.estimated_savings_usd_hr == 50.0
        assert result.co2_reduction_tons_year == 2190.0


class TestCoolingTowerOptimizationResult:
    """Tests for cooling tower optimization result schema."""

    def test_valid_result(self, sample_cooling_tower_optimization_result):
        """Test valid optimization result."""
        result = sample_cooling_tower_optimization_result
        assert result.tower_id == "CT-001"
        assert result.status == OptimizationStatus.SUCCESS
        assert result.energy_savings_pct == 23.0


class TestOptimizationRecommendation:
    """Tests for optimization recommendation schema."""

    def test_valid_recommendation(self):
        """Test valid recommendation."""
        rec = OptimizationRecommendation(
            parameter="excess_air_pct",
            current_value=20.0,
            recommended_value=15.0,
            unit="%",
            expected_improvement_pct=1.5,
            priority="high",
            rationale="Reduce excess air to improve efficiency",
        )
        assert rec.parameter == "excess_air_pct"
        assert rec.priority == "high"


class TestEnums:
    """Tests for schema enums."""

    def test_safety_status_values(self):
        """Test safety status enum values."""
        assert SafetyStatus.SAFE.value == "safe"
        assert SafetyStatus.WARNING.value == "warning"
        assert SafetyStatus.VIOLATION.value == "violation"

    def test_operating_mode_values(self):
        """Test operating mode enum values."""
        assert OperatingMode.NORMAL.value == "normal"
        assert OperatingMode.STARTUP.value == "startup"
        assert OperatingMode.SHUTDOWN.value == "shutdown"

    def test_optimization_status_values(self):
        """Test optimization status enum values."""
        assert OptimizationStatus.SUCCESS.value == "success"
        assert OptimizationStatus.FAILED.value == "failed"

    def test_combustion_status_values(self):
        """Test combustion status enum values."""
        assert CombustionStatus.OPTIMAL.value == "optimal"
        assert CombustionStatus.LEAN.value == "lean"
        assert CombustionStatus.RICH.value == "rich"
