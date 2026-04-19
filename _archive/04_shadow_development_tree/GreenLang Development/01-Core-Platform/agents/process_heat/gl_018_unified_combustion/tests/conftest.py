# -*- coding: utf-8 -*-
"""
GL-018 UnifiedCombustionOptimizer Test Fixtures
===============================================

Pytest fixtures for GL-018 test suite.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from greenlang.agents.process_heat.gl_018_unified_combustion.config import (
    UnifiedCombustionConfig,
    BurnerConfig,
    AirFuelConfig,
    FlueGasConfig,
    FlameStabilityConfig,
    EmissionsConfig,
    BMSConfig,
    SootBlowingConfig,
    BlowdownConfig,
    EfficiencyConfig,
    FuelType,
    EquipmentType,
    BurnerType,
    ControlMode,
    EmissionControlTechnology,
    BMSSequence,
)

from greenlang.agents.process_heat.gl_018_unified_combustion.schemas import (
    CombustionInput,
    CombustionOutput,
    FlueGasReading,
    BurnerStatus,
    EfficiencyResult,
    FlueGasAnalysis,
    FlameStabilityAnalysis,
    EmissionsAnalysis,
    BMSStatus,
    BurnerTuningResult,
    OptimizationRecommendation,
    Alert,
    AlertSeverity,
    RecommendationPriority,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def default_burner_config():
    """Default burner configuration."""
    return BurnerConfig(
        burner_id="BNR-001",
        burner_type=BurnerType.LOW_NOX,
        burner_count=2,
        capacity_mmbtu_hr=50.0,
        turndown_ratio=4.0,
        min_firing_rate_pct=25.0,
        design_nox_ppm=30.0,
        design_co_ppm=50.0,
    )


@pytest.fixture
def default_air_fuel_config():
    """Default air-fuel configuration."""
    return AirFuelConfig(
        control_mode=ControlMode.CROSS_LIMITING,
        target_o2_pct=3.0,
        min_o2_pct=1.5,
        max_o2_pct=6.0,
        target_excess_air_pct=15.0,
        min_excess_air_pct=10.0,
        max_excess_air_pct=25.0,
        o2_trim_enabled=True,
        o2_trim_bias_max_pct=5.0,
        o2_trim_response_time_s=30.0,
        cross_limiting_enabled=True,
    )


@pytest.fixture
def default_flue_gas_config():
    """Default flue gas configuration."""
    return FlueGasConfig(
        analyzer_type="in_situ_zirconia",
        analyzer_response_time_s=10.0,
        max_flue_temp_f=500.0,
        min_flue_temp_f=250.0,
        acid_dew_point_margin_f=25.0,
        co_alarm_ppm=100.0,
        co_trip_ppm=400.0,
        nox_limit_ppm=30.0,
    )


@pytest.fixture
def default_flame_stability_config():
    """Default flame stability configuration."""
    return FlameStabilityConfig(
        detector_type="uv_ir",
        detector_count_per_burner=2,
        flame_signal_min_pct=30.0,
        fsi_optimal_min=0.85,
        fsi_warning_threshold=0.70,
        fsi_alarm_threshold=0.50,
        flame_failure_response_s=4.0,
    )


@pytest.fixture
def default_emissions_config():
    """Default emissions configuration."""
    return EmissionsConfig(
        nox_control=EmissionControlTechnology.LOW_NOX_BURNER,
        co_control=EmissionControlTechnology.NONE,
        fgr_enabled=False,
        scr_enabled=False,
        nox_permit_limit_lb_mmbtu=0.05,
        co_permit_limit_lb_mmbtu=0.04,
    )


@pytest.fixture
def default_bms_config():
    """Default BMS configuration."""
    return BMSConfig(
        sil_level=2,
        pre_purge_time_s=60.0,
        post_purge_time_s=30.0,
        pilot_trial_time_s=10.0,
        main_flame_trial_time_s=10.0,
        purge_air_flow_pct=25.0,
        purge_volume_changes=4,
        low_fire_interlock=True,
        flame_detector_redundancy="1oo2",
    )


@pytest.fixture
def default_efficiency_config():
    """Default efficiency configuration."""
    return EfficiencyConfig(
        calculation_method="losses",
        design_efficiency_pct=82.0,
        guarantee_efficiency_pct=80.0,
        full_load_target_pct=84.0,
        min_load_target_pct=80.0,
        measurement_uncertainty_pct=1.5,
    )


@pytest.fixture
def default_combustion_config(
    default_burner_config,
    default_air_fuel_config,
    default_flue_gas_config,
    default_flame_stability_config,
    default_emissions_config,
    default_bms_config,
    default_efficiency_config,
):
    """Complete combustion configuration."""
    return UnifiedCombustionConfig(
        equipment_id="BOILER-001",
        name="Main Process Boiler",
        equipment_type=EquipmentType.BOILER_WATERTUBE,
        fuel_type=FuelType.NATURAL_GAS,
        design_capacity_mmbtu_hr=100.0,
        min_load_pct=25.0,
        max_load_pct=110.0,
        burner=default_burner_config,
        air_fuel=default_air_fuel_config,
        flue_gas=default_flue_gas_config,
        flame_stability=default_flame_stability_config,
        emissions=default_emissions_config,
        bms=default_bms_config,
        efficiency=default_efficiency_config,
        control_mode=ControlMode.OPTIMIZING,
        optimization_enabled=True,
    )


# =============================================================================
# INPUT DATA FIXTURES
# =============================================================================


@pytest.fixture
def default_flue_gas_reading():
    """Default flue gas reading."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        o2_pct=3.5,
        co2_pct=8.5,
        co_ppm=50.0,
        nox_ppm=25.0,
        so2_ppm=0.0,
        temperature_f=350.0,
    )


@pytest.fixture
def default_burner_status():
    """Default burner status."""
    return BurnerStatus(
        burner_id="BNR-001",
        status="firing",
        flame_signal_pct=85.0,
        firing_rate_pct=75.0,
        air_damper_position_pct=60.0,
        fuel_valve_position_pct=65.0,
    )


@pytest.fixture
def default_combustion_input(default_flue_gas_reading, default_burner_status):
    """Default combustion input data."""
    return CombustionInput(
        equipment_id="BOILER-001",
        timestamp=datetime.now(timezone.utc),
        fuel_type="natural_gas",
        fuel_flow_rate=75.0,  # MMBTU/hr
        fuel_hhv=23875,  # BTU/lb
        flue_gas=default_flue_gas_reading,
        burners=[default_burner_status],
        load_pct=75.0,
        air_damper_position_pct=60.0,
        ambient_temperature_f=70.0,
        combustion_air_temperature_f=80.0,
        steam_flow_rate_lb_hr=50000.0,
        steam_pressure_psig=150.0,
        steam_temperature_f=366.0,
        feedwater_temperature_f=227.0,
        blowdown_rate_pct=2.0,
    )


# =============================================================================
# OUTPUT DATA FIXTURES
# =============================================================================


@pytest.fixture
def default_efficiency_result():
    """Default efficiency result."""
    return EfficiencyResult(
        net_efficiency_pct=82.5,
        combustion_efficiency_pct=98.5,
        total_losses_pct=17.5,
        dry_flue_gas_loss_pct=8.5,
        moisture_in_fuel_loss_pct=0.5,
        moisture_from_combustion_loss_pct=4.0,
        moisture_in_air_loss_pct=0.2,
        radiation_convection_loss_pct=1.5,
        unburned_carbon_loss_pct=0.3,
        blowdown_loss_pct=1.0,
        other_losses_pct=1.5,
        calculation_method="losses",
        formula_reference="ASME PTC 4.1",
    )


@pytest.fixture
def default_flue_gas_analysis():
    """Default flue gas analysis."""
    return FlueGasAnalysis(
        timestamp=datetime.now(timezone.utc),
        o2_pct=3.5,
        excess_air_pct=18.0,
        optimal_o2_pct=3.0,
        o2_deviation_pct=0.5,
        adjust_air_fuel=True,
        estimated_improvement_pct=0.5,
        acid_dew_point_f=275.0,
        acid_dew_point_margin_f=75.0,
        air_fuel_ratio_actual=17.0,
        air_fuel_ratio_stoichiometric=14.5,
    )


@pytest.fixture
def default_flame_stability_analysis():
    """Default flame stability analysis."""
    return FlameStabilityAnalysis(
        flame_stability_index=0.88,
        fsi_status="normal",
        flame_intensity_avg=85.0,
        flame_intensity_variance=5.0,
        burner_flame_status={"BNR-001": "stable"},
        burner_fsi={"BNR-001": 0.88},
        tuning_required=False,
        tuning_recommendations=[],
    )


@pytest.fixture
def default_emissions_analysis():
    """Default emissions analysis."""
    return EmissionsAnalysis(
        in_compliance=True,
        compliance_issues=[],
        nox_ppm=25.0,
        nox_lb_mmbtu=0.035,
        nox_compliance_pct=70.0,
        nox_permit_limit_lb_mmbtu=0.05,
        co_ppm=50.0,
        co_lb_mmbtu=0.02,
        co_compliance_pct=50.0,
        co2_lb_mmbtu=117.0,
        co2_tons_hr=4.4,
        recommendations=[],
    )


@pytest.fixture
def default_bms_status():
    """Default BMS status."""
    return BMSStatus(
        sequence=BMSSequence.RUNNING,
        all_interlocks_ok=True,
        flame_detected=True,
        pre_purge_complete=True,
        ready_to_fire=True,
        active_alarms=[],
        interlock_status={
            "fuel_supply_pressure": True,
            "combustion_air_pressure": True,
            "low_water_cutoff": True,
            "flame_failure_relay": True,
        },
    )


@pytest.fixture
def default_optimization_recommendation():
    """Default optimization recommendation."""
    return OptimizationRecommendation(
        category="combustion",
        priority=RecommendationPriority.MEDIUM,
        title="Optimize Air-Fuel Ratio",
        description="O2 is 0.5% above optimal. Recommend adjusting to 3.0%.",
        parameter="flue_gas_o2_pct",
        current_value=3.5,
        recommended_value=3.0,
        unit="%",
        estimated_efficiency_gain_pct=0.5,
        implementation_difficulty="low",
        auto_implementable=True,
    )


@pytest.fixture
def default_alert():
    """Default alert."""
    return Alert(
        severity=AlertSeverity.WARNING,
        category="combustion",
        message="O2 slightly above optimal",
        parameter="flue_gas_o2_pct",
        value=3.5,
        threshold=3.0,
    )
