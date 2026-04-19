# -*- coding: utf-8 -*-
"""
GL-007 FurnaceOptimizer Test Fixtures
=====================================

Pytest fixtures for GL-007 test suite.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_007_furnace_optimizer.config import (
    GL007Config,
    FurnaceOptimizerConfig,
    CoolingTowerConfig,
    CombustionConfig,
    HeatTransferConfig,
    BurnerConfig,
    NFPA86Config,
    ASHRAEConfig,
    ExplainabilityConfig,
    ProvenanceConfig,
    FurnaceType,
    FuelType,
    CoolingTowerType,
    FillType,
    ControlMode,
    SafetyIntegrityLevel,
)

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
    SafetyStatus,
    ValidationStatus,
    OperatingMode,
    OptimizationStatus,
    CombustionStatus,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def default_burner_config():
    """Default burner configuration."""
    return BurnerConfig(
        burner_id="BNR-001",
        burner_type="premix",
        capacity_mmbtu_hr=10.0,
        min_firing_rate_pct=10.0,
        max_firing_rate_pct=100.0,
        turndown_ratio=10.0,
        nox_emissions_lb_mmbtu=0.05,
        co_emissions_ppm=50.0,
    )


@pytest.fixture
def default_combustion_config(default_burner_config):
    """Default combustion configuration."""
    return CombustionConfig(
        fuel_type=FuelType.NATURAL_GAS,
        fuel_hhv_btu_scf=1020.0,
        fuel_lhv_btu_scf=920.0,
        target_excess_air_pct=15.0,
        target_o2_pct=3.0,
        max_co_ppm=100.0,
        max_nox_lb_mmbtu=0.1,
        burners=[default_burner_config],
    )


@pytest.fixture
def default_heat_transfer_config():
    """Default heat transfer configuration."""
    return HeatTransferConfig(
        radiant_surface_area_ft2=500.0,
        convective_surface_area_ft2=1000.0,
        radiant_htc_btu_hr_ft2_f=10.0,
        convective_htc_btu_hr_ft2_f=5.0,
        wall_loss_pct=2.0,
        opening_loss_pct=1.0,
    )


@pytest.fixture
def default_furnace_config(default_combustion_config, default_heat_transfer_config):
    """Default furnace optimizer configuration."""
    return FurnaceOptimizerConfig(
        furnace_id="FUR-001",
        furnace_type=FurnaceType.DIRECT_FIRED,
        design_temp_f=1800.0,
        min_operating_temp_f=400.0,
        max_operating_temp_f=2000.0,
        design_duty_mmbtu_hr=50.0,
        design_efficiency_pct=85.0,
        target_efficiency_pct=88.0,
        tmt_design_limit_f=1500.0,
        tmt_alarm_f=1450.0,
        tmt_trip_f=1500.0,
        combustion=default_combustion_config,
        heat_transfer=default_heat_transfer_config,
        control_mode=ControlMode.AUTOMATIC,
    )


@pytest.fixture
def default_cooling_tower_config():
    """Default cooling tower configuration."""
    return CoolingTowerConfig(
        tower_id="CT-001",
        tower_type=CoolingTowerType.MECHANICAL_INDUCED,
        fill_type=FillType.FILM,
        design_wet_bulb_f=78.0,
        design_dry_bulb_f=95.0,
        design_hot_water_temp_f=105.0,
        design_cold_water_temp_f=85.0,
        design_range_f=20.0,
        design_approach_f=7.0,
        design_flow_gpm=5000.0,
        design_air_flow_cfm=200000.0,
        design_heat_rejection_mmbtu_hr=25.0,
        design_lg_ratio=1.2,
        num_fans=2,
        fan_motor_hp=100.0,
        fan_vfd_enabled=True,
    )


@pytest.fixture
def default_nfpa86_config():
    """Default NFPA 86 safety configuration."""
    return NFPA86Config(
        nfpa_86_compliance=True,
        furnace_class="A",
        purge_required=True,
        min_purge_time_sec=60,
        min_purge_volume_changes=4,
        flame_safety_system=True,
        max_flame_failure_response_sec=4.0,
        high_temp_alarm_f=1850.0,
        high_temp_trip_f=1900.0,
        sil_rating=SafetyIntegrityLevel.SIL_2,
    )


@pytest.fixture
def default_ashrae_config():
    """Default ASHRAE configuration."""
    return ASHRAEConfig(
        ashrae_90_1_compliance=True,
        climate_zone="4A",
        min_tower_efficiency_gpm_hp=42.1,
        design_db_01_pct_f=95.0,
        design_wb_01_pct_f=78.0,
        guideline_12_compliance=True,
    )


@pytest.fixture
def gl007_config(
    default_furnace_config,
    default_cooling_tower_config,
    default_nfpa86_config,
    default_ashrae_config,
):
    """Complete GL-007 configuration."""
    return GL007Config(
        furnace=default_furnace_config,
        cooling_tower=default_cooling_tower_config,
        nfpa86=default_nfpa86_config,
        ashrae=default_ashrae_config,
        agent_id="GL-007-TEST",
        optimization_interval_sec=60,
    )


# =============================================================================
# DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_flue_gas_analysis():
    """Sample flue gas analysis."""
    return FlueGasAnalysis(
        o2_pct=3.5,
        co2_pct=9.5,
        co_ppm=50.0,
        nox_ppm=25.0,
        excess_air_pct=20.0,
    )


@pytest.fixture
def sample_zone_temperature():
    """Sample zone temperature."""
    return ZoneTemperature(
        zone_id="ZONE-1",
        zone_name="Radiant Section",
        temperature_f=1750.0,
        setpoint_f=1800.0,
    )


@pytest.fixture
def sample_tmt_reading():
    """Sample tube metal temperature reading."""
    return TubeMetalTemperature(
        sensor_id="TMT-001",
        location="Pass 1 Outlet",
        temperature_f=1400.0,
        design_limit_f=1500.0,
    )


@pytest.fixture
def sample_furnace_reading(sample_flue_gas_analysis, sample_zone_temperature, sample_tmt_reading):
    """Sample furnace reading."""
    return FurnaceReading(
        furnace_id="FUR-001",
        operating_mode=OperatingMode.NORMAL,
        is_firing=True,
        furnace_temp_f=1750.0,
        furnace_temp_setpoint_f=1800.0,
        zone_temperatures=[sample_zone_temperature],
        tmt_readings=[sample_tmt_reading],
        fuel_flow_rate_scfh=5000.0,
        fuel_pressure_psig=10.0,
        flue_gas_temp_f=450.0,
        flue_gas_analysis=sample_flue_gas_analysis,
        flue_gas_o2_pct=3.5,
        flue_gas_co_ppm=50.0,
        heat_input_mmbtu_hr=51.0,
    )


@pytest.fixture
def sample_fan_reading():
    """Sample fan reading."""
    return FanReading(
        fan_id="FAN-1",
        status="running",
        speed_pct=75.0,
        motor_amps=85.0,
        motor_power_kw=65.0,
    )


@pytest.fixture
def sample_water_quality():
    """Sample water quality."""
    return WaterQuality(
        conductivity_umho_cm=1500.0,
        ph=7.5,
        tds_ppm=750.0,
        cycles_of_concentration=5.0,
    )


@pytest.fixture
def sample_cooling_tower_reading(sample_fan_reading, sample_water_quality):
    """Sample cooling tower reading."""
    return CoolingTowerReading(
        tower_id="CT-001",
        operating_mode=OperatingMode.NORMAL,
        ambient_dry_bulb_f=90.0,
        ambient_wet_bulb_f=75.0,
        relative_humidity_pct=65.0,
        hot_water_temp_f=100.0,
        cold_water_temp_f=82.0,
        water_flow_gpm=4500.0,
        air_flow_cfm=180000.0,
        fan_readings=[sample_fan_reading],
        total_fan_power_kw=65.0,
        water_quality=sample_water_quality,
        basin_level_pct=85.0,
    )


@pytest.fixture
def sample_combustion_analysis():
    """Sample combustion analysis result."""
    return CombustionAnalysis(
        analysis_id="CA-001",
        stoichiometric_air_scf_per_scf_fuel=9.5,
        actual_air_scf_per_scf_fuel=11.0,
        excess_air_pct=15.8,
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
        combustion_efficiency_pct=98.5,
        thermal_efficiency_pct=85.0,
        co2_lb_mmbtu=117.0,
    )


@pytest.fixture
def sample_heat_transfer_analysis():
    """Sample heat transfer analysis result."""
    return HeatTransferAnalysis(
        analysis_id="HT-001",
        design_duty_mmbtu_hr=50.0,
        actual_duty_mmbtu_hr=43.5,
        duty_ratio_pct=87.0,
        radiant_heat_transfer_mmbtu_hr=30.0,
        convective_heat_transfer_mmbtu_hr=13.5,
        overall_htc_btu_hr_ft2_f=7.5,
        design_htc_btu_hr_ft2_f=8.0,
        htc_ratio_pct=93.75,
        lmtd_f=150.0,
        approach_temp_f=50.0,
        fouling_factor_hr_ft2_f_btu=0.001,
        fouling_severity="low",
        wall_loss_mmbtu_hr=1.0,
        wall_loss_pct=2.0,
        heat_transfer_effectiveness_pct=87.0,
    )


@pytest.fixture
def sample_furnace_optimization_result(sample_combustion_analysis, sample_heat_transfer_analysis):
    """Sample furnace optimization result."""
    return FurnaceOptimizationResult(
        furnace_id="FUR-001",
        execution_id="EX-001",
        status=OptimizationStatus.SUCCESS,
        safety_status=SafetyStatus.SAFE,
        current_efficiency_pct=85.0,
        current_excess_air_pct=20.0,
        current_fuel_rate_scfh=5000.0,
        current_heat_input_mmbtu_hr=51.0,
        optimal_efficiency_pct=88.0,
        optimal_excess_air_pct=15.0,
        optimal_o2_pct=3.0,
        optimal_fuel_rate_scfh=4800.0,
        efficiency_improvement_pct=3.0,
        fuel_savings_pct=4.0,
        estimated_savings_usd_hr=50.0,
        estimated_savings_usd_year=350000.0,
        co2_reduction_lb_hr=500.0,
        co2_reduction_tons_year=2190.0,
        combustion_analysis=sample_combustion_analysis,
        heat_transfer_analysis=sample_heat_transfer_analysis,
        provenance_hash="abc123",
        processing_time_ms=150.0,
    )


@pytest.fixture
def sample_cooling_tower_optimization_result():
    """Sample cooling tower optimization result."""
    return CoolingTowerOptimizationResult(
        tower_id="CT-001",
        execution_id="EX-002",
        status=OptimizationStatus.SUCCESS,
        safety_status=SafetyStatus.SAFE,
        current_approach_f=7.0,
        current_range_f=18.0,
        current_efficiency_pct=85.0,
        current_fan_power_kw=130.0,
        optimal_approach_f=6.0,
        optimal_fan_speed_pct=65.0,
        optimal_lg_ratio=1.1,
        optimal_fan_power_kw=100.0,
        approach_improvement_f=1.0,
        energy_savings_pct=23.0,
        estimated_savings_kwh_hr=30.0,
        estimated_savings_usd_year=20000.0,
        merkel_number=1.5,
        ntu=2.0,
        provenance_hash="def456",
        processing_time_ms=100.0,
    )
