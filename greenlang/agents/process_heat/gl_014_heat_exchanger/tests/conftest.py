# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Test Configuration and Fixtures

Shared fixtures for all GL-014 tests. Provides reusable test data,
mock objects, and configuration for the heat exchanger test suite.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import math

# Import GL-014 modules
from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    HeatExchangerConfig,
    TubeGeometryConfig,
    ShellGeometryConfig,
    PlateGeometryConfig,
    AirCooledGeometryConfig,
    FoulingConfig,
    CleaningConfig,
    TubeIntegrityConfig,
    OperatingLimitsConfig,
    EconomicsConfig,
    MLConfig,
    TEMAFoulingFactors,
    ExchangerType,
    TEMAClass,
    FlowArrangement,
    FoulingCategory,
    CleaningMethod,
    TubeLayout,
    TubeMaterial,
    AlertSeverity,
)

from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    HeatExchangerInput,
    HeatExchangerOperatingData,
    StreamConditions,
    TubeInspectionData,
    CleaningRecord,
    HealthStatus,
    TrendDirection,
    OperatingMode,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def tube_geometry_config():
    """Standard tube geometry for shell-tube exchanger."""
    return TubeGeometryConfig(
        outer_diameter_mm=25.4,
        wall_thickness_mm=2.11,
        tube_length_m=6.096,
        tube_count=100,
        tube_passes=2,
        tube_pitch_mm=31.75,
        tube_layout=TubeLayout.TRIANGULAR_30,
        tube_material=TubeMaterial.CARBON_STEEL,
    )


@pytest.fixture
def shell_geometry_config():
    """Standard shell geometry configuration."""
    return ShellGeometryConfig(
        inner_diameter_mm=610.0,
        shell_passes=1,
        baffle_cut_percent=25.0,
        baffle_spacing_mm=300.0,
        baffle_count=10,
        sealing_strips=0,
    )


@pytest.fixture
def plate_geometry_config():
    """Plate heat exchanger geometry configuration."""
    return PlateGeometryConfig(
        plate_count=50,
        plate_length_mm=1000.0,
        plate_width_mm=400.0,
        plate_spacing_mm=3.0,
        chevron_angle_deg=60.0,
        port_diameter_mm=150.0,
        plate_material=TubeMaterial.STAINLESS_316,
    )


@pytest.fixture
def air_cooled_geometry_config():
    """Air-cooled exchanger geometry configuration."""
    return AirCooledGeometryConfig(
        bundle_count=2,
        fan_count=4,
        fan_diameter_m=3.0,
        tubes_per_row=30,
        tube_rows=6,
        fin_pitch_mm=2.5,
        fin_height_mm=15.0,
        fin_thickness_mm=0.4,
        induced_draft=True,
    )


@pytest.fixture
def fouling_config():
    """Standard fouling configuration per TEMA."""
    return FoulingConfig(
        shell_side_fouling_m2kw=0.00017,
        tube_side_fouling_m2kw=0.00017,
        fouling_category=FoulingCategory.PARTICULATE,
        design_fouling_factor=1.0,
        fouling_rate_m2kw_per_day=0.000001,
        asymptotic_fouling_m2kw=None,
        removal_rate_coefficient=0.1,
        ml_prediction_enabled=True,
        prediction_horizon_days=30,
    )


@pytest.fixture
def cleaning_config():
    """Standard cleaning configuration."""
    return CleaningConfig(
        preferred_methods=[CleaningMethod.HIGH_PRESSURE_WATER],
        minimum_interval_days=30,
        maximum_interval_days=365,
        effectiveness_threshold=0.70,
        fouling_threshold_m2kw=0.00035,
        cleaning_duration_hours=8.0,
        cleaning_cost_usd=5000.0,
        production_loss_usd_per_hour=1000.0,
        optimize_schedule=True,
        target_availability=0.95,
    )


@pytest.fixture
def tube_integrity_config():
    """Standard tube integrity configuration."""
    return TubeIntegrityConfig(
        design_life_years=20.0,
        installed_date=datetime.now(timezone.utc) - timedelta(days=5*365),
        last_inspection_date=datetime.now(timezone.utc) - timedelta(days=180),
        minimum_wall_thickness_mm=1.25,
        corrosion_allowance_mm=1.5,
        expected_corrosion_rate_mm_year=0.1,
        inspection_interval_months=24,
        eddy_current_enabled=True,
        plugging_threshold_percent=10.0,
        predict_tube_failures=True,
        weibull_beta=2.5,
        weibull_eta=20.0,
    )


@pytest.fixture
def operating_limits_config():
    """Standard operating limits configuration."""
    return OperatingLimitsConfig(
        max_shell_inlet_temp_c=300.0,
        max_tube_inlet_temp_c=300.0,
        max_tube_wall_temp_c=350.0,
        min_approach_temp_c=5.0,
        max_shell_pressure_barg=15.0,
        max_tube_pressure_barg=15.0,
        max_differential_pressure_bar=5.0,
        max_shell_velocity_m_s=3.0,
        max_tube_velocity_m_s=3.0,
        min_tube_velocity_m_s=0.5,
        max_shell_dp_bar=1.0,
        max_tube_dp_bar=1.0,
        min_effectiveness=0.5,
        alarm_effectiveness=0.65,
    )


@pytest.fixture
def economics_config():
    """Standard economics configuration."""
    return EconomicsConfig(
        energy_cost_usd_per_kwh=0.10,
        steam_cost_usd_per_ton=30.0,
        cooling_water_cost_usd_per_m3=0.50,
        replacement_cost_usd=500000.0,
        retube_cost_usd=150000.0,
        production_value_usd_per_hour=10000.0,
        discount_rate=0.10,
        analysis_horizon_years=10,
    )


@pytest.fixture
def ml_config():
    """ML configuration for predictions."""
    return MLConfig(
        enabled=True,
        fouling_prediction_enabled=True,
        tube_failure_prediction_enabled=True,
        cleaning_optimization_enabled=True,
        model_update_interval_days=30,
        uncertainty_quantification=True,
        confidence_threshold=0.80,
        explainability_enabled=True,
    )


@pytest.fixture
def shell_tube_config(
    tube_geometry_config,
    shell_geometry_config,
    fouling_config,
    cleaning_config,
    tube_integrity_config,
    operating_limits_config,
    economics_config,
    ml_config,
):
    """Complete shell-tube heat exchanger configuration."""
    return HeatExchangerConfig(
        exchanger_id="E-1001",
        exchanger_type=ExchangerType.SHELL_TUBE,
        tema_type="AES",
        tema_class=TEMAClass.R,
        service_description="Crude oil preheat",
        tag_number="E-1001",
        location="Unit 100",
        tube_geometry=tube_geometry_config,
        shell_geometry=shell_geometry_config,
        flow_arrangement=FlowArrangement.COUNTER_FLOW,
        design_duty_kw=1000.0,
        design_u_w_m2k=500.0,
        design_lmtd_c=30.0,
        design_effectiveness=0.75,
        shell_side_fluid="crude_oil",
        tube_side_fluid="cooling_water",
        shell_flow_kg_s=10.0,
        tube_flow_kg_s=10.0,
        fouling=fouling_config,
        cleaning=cleaning_config,
        tube_integrity=tube_integrity_config,
        operating_limits=operating_limits_config,
        economics=economics_config,
        ml=ml_config,
        audit_enabled=True,
        provenance_tracking=True,
    )


@pytest.fixture
def plate_exchanger_config(
    plate_geometry_config,
    fouling_config,
    cleaning_config,
    economics_config,
):
    """Plate heat exchanger configuration."""
    return HeatExchangerConfig(
        exchanger_id="PHE-001",
        exchanger_type=ExchangerType.PLATE,
        service_description="Process water cooling",
        plate_geometry=plate_geometry_config,
        flow_arrangement=FlowArrangement.COUNTER_FLOW,
        design_duty_kw=500.0,
        design_u_w_m2k=2500.0,
        design_lmtd_c=10.0,
        design_effectiveness=0.85,
        shell_side_fluid="hot_water",
        tube_side_fluid="cold_water",
        shell_flow_kg_s=5.0,
        tube_flow_kg_s=5.0,
        fouling=fouling_config,
        cleaning=cleaning_config,
        economics=economics_config,
    )


# =============================================================================
# INPUT DATA FIXTURES
# =============================================================================

@pytest.fixture
def stream_conditions_hot():
    """Hot side stream conditions."""
    return StreamConditions(
        temperature_c=150.0,
        pressure_barg=5.0,
        mass_flow_kg_s=10.0,
        density_kg_m3=850.0,
        viscosity_cp=2.5,
        specific_heat_kj_kgk=2.1,
        thermal_conductivity_w_mk=0.13,
    )


@pytest.fixture
def stream_conditions_cold():
    """Cold side stream conditions."""
    return StreamConditions(
        temperature_c=30.0,
        pressure_barg=3.0,
        mass_flow_kg_s=15.0,
        density_kg_m3=998.0,
        viscosity_cp=0.8,
        specific_heat_kj_kgk=4.18,
        thermal_conductivity_w_mk=0.62,
    )


@pytest.fixture
def operating_data(stream_conditions_hot, stream_conditions_cold):
    """Heat exchanger operating data."""
    return HeatExchangerOperatingData(
        timestamp=datetime.now(timezone.utc),
        shell_inlet=stream_conditions_hot,
        shell_outlet=StreamConditions(
            temperature_c=100.0,
            pressure_barg=4.8,
            mass_flow_kg_s=10.0,
            density_kg_m3=870.0,
            viscosity_cp=3.0,
            specific_heat_kj_kgk=2.1,
        ),
        shell_pressure_drop_bar=0.2,
        tube_inlet=stream_conditions_cold,
        tube_outlet=StreamConditions(
            temperature_c=60.0,
            pressure_barg=2.8,
            mass_flow_kg_s=15.0,
            density_kg_m3=990.0,
            viscosity_cp=0.6,
            specific_heat_kj_kgk=4.18,
        ),
        tube_pressure_drop_bar=0.3,
        operating_mode=OperatingMode.NORMAL,
        load_percent=100.0,
    )


@pytest.fixture
def tube_inspection_data():
    """Sample tube inspection data."""
    return TubeInspectionData(
        inspection_date=datetime.now(timezone.utc),
        inspection_method="eddy_current",
        total_tubes=100,
        tubes_inspected=100,
        tubes_with_defects=5,
        tubes_plugged=2,
        wall_loss_summary={
            "<20%": 90,
            "20-40%": 7,
            "40-60%": 2,
            "60-80%": 1,
            ">80%": 0,
        },
        defect_locations=[
            {"tube": 45, "location": "inlet", "severity": "moderate"},
            {"tube": 67, "location": "middle", "severity": "minor"},
        ],
        tubes_recommended_for_plugging=[45, 67, 89],
        retube_recommended=False,
    )


@pytest.fixture
def cleaning_record():
    """Sample cleaning record."""
    return CleaningRecord(
        cleaning_date=datetime.now(timezone.utc) - timedelta(days=90),
        cleaning_method=CleaningMethod.HIGH_PRESSURE_WATER,
        duration_hours=8.0,
        cost_usd=5500.0,
        u_before_cleaning=380.0,
        u_after_cleaning=480.0,
        effectiveness_before=0.65,
        effectiveness_after=0.82,
        notes="Standard maintenance cleaning",
    )


@pytest.fixture
def heat_exchanger_input(operating_data, tube_inspection_data, cleaning_record):
    """Complete heat exchanger input data."""
    return HeatExchangerInput(
        exchanger_id="E-1001",
        operating_data=operating_data,
        operating_history=[],
        inspection_data=tube_inspection_data,
        cleaning_history=[cleaning_record],
        time_since_last_cleaning_days=90.0,
        running_hours=40000.0,
    )


# =============================================================================
# FLUID PROPERTIES FIXTURES
# =============================================================================

@pytest.fixture
def water_properties():
    """Properties of water at ~25C."""
    return {
        "density_kg_m3": 998.0,
        "viscosity_pa_s": 0.001,
        "specific_heat_j_kgk": 4180.0,
        "thermal_conductivity_w_mk": 0.62,
    }


@pytest.fixture
def crude_oil_properties():
    """Properties of crude oil at ~100C."""
    return {
        "density_kg_m3": 850.0,
        "viscosity_pa_s": 0.003,
        "specific_heat_j_kgk": 2100.0,
        "thermal_conductivity_w_mk": 0.13,
    }


# =============================================================================
# CALCULATION FIXTURES
# =============================================================================

@pytest.fixture
def known_effectiveness_values():
    """Known effectiveness values for validation (from Kays & London)."""
    return [
        # (NTU, Cr, arrangement, expected_effectiveness)
        (0.5, 0.5, "counter_flow", 0.3934),
        (1.0, 0.5, "counter_flow", 0.5934),
        (2.0, 0.5, "counter_flow", 0.7754),
        (3.0, 0.5, "counter_flow", 0.8647),
        (1.0, 1.0, "counter_flow", 0.5),
        (2.0, 1.0, "counter_flow", 0.6667),
        (0.5, 0.5, "parallel_flow", 0.3542),
        (1.0, 0.5, "parallel_flow", 0.5109),
        (2.0, 0.5, "parallel_flow", 0.6166),
        (1.0, 0.0, "any", 0.6321),  # Cr=0 (phase change)
    ]


@pytest.fixture
def tema_fouling_factors():
    """TEMA standard fouling factors for validation."""
    return {
        "cooling_tower_water": 0.00035,
        "sea_water": 0.00017,
        "boiler_feedwater": 0.00009,
        "river_water": 0.00035,
        "fuel_oil": 0.00088,
        "crude_oil_dry": 0.00035,
        "crude_oil_wet": 0.00053,
        "gas_oil": 0.00035,
        "gasoline": 0.00018,
        "naphtha": 0.00018,
        "steam": 0.00009,
        "process_gas": 0.00018,
        "organic_solvents": 0.00018,
    }


@pytest.fixture
def pressure_drop_test_cases():
    """Test cases for pressure drop calculations."""
    return [
        {
            "name": "turbulent_water",
            "mass_flow_kg_s": 10.0,
            "density_kg_m3": 998.0,
            "viscosity_pa_s": 0.001,
            "tube_id_mm": 21.18,
            "tube_length_m": 6.096,
            "tubes_per_pass": 50,
            "passes": 2,
        },
        {
            "name": "laminar_oil",
            "mass_flow_kg_s": 5.0,
            "density_kg_m3": 850.0,
            "viscosity_pa_s": 0.1,
            "tube_id_mm": 21.18,
            "tube_length_m": 6.096,
            "tubes_per_pass": 50,
            "passes": 2,
        },
    ]


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_provenance_tracker():
    """Mock provenance tracker."""
    tracker = Mock()
    tracker.record_calculation.return_value = Mock(
        provenance_hash="a" * 64,
        timestamp=datetime.now(timezone.utc),
    )
    return tracker


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger."""
    logger = Mock()
    logger.log_calculation.return_value = None
    return logger


# =============================================================================
# PERFORMANCE TEST FIXTURES
# =============================================================================

@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    def _generate(n_records: int) -> List[Dict[str, Any]]:
        import random
        records = []
        for i in range(n_records):
            records.append({
                "exchanger_id": f"E-{i:04d}",
                "shell_inlet_temp": random.uniform(100, 200),
                "shell_outlet_temp": random.uniform(50, 100),
                "tube_inlet_temp": random.uniform(20, 40),
                "tube_outlet_temp": random.uniform(50, 80),
                "shell_flow": random.uniform(5, 20),
                "tube_flow": random.uniform(5, 20),
            })
        return records
    return _generate


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_close(actual, expected, rel_tol=0.01, abs_tol=1e-9):
    """Assert that two values are close within tolerance."""
    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), \
        f"Expected {expected}, got {actual} (rel_tol={rel_tol})"


def create_operating_data_at_conditions(
    shell_in_temp: float,
    shell_out_temp: float,
    tube_in_temp: float,
    tube_out_temp: float,
    shell_flow: float = 10.0,
    tube_flow: float = 10.0,
) -> HeatExchangerOperatingData:
    """Create operating data with specified conditions."""
    return HeatExchangerOperatingData(
        timestamp=datetime.now(timezone.utc),
        shell_inlet=StreamConditions(
            temperature_c=shell_in_temp,
            pressure_barg=5.0,
            mass_flow_kg_s=shell_flow,
        ),
        shell_outlet=StreamConditions(
            temperature_c=shell_out_temp,
            pressure_barg=4.8,
            mass_flow_kg_s=shell_flow,
        ),
        tube_inlet=StreamConditions(
            temperature_c=tube_in_temp,
            pressure_barg=3.0,
            mass_flow_kg_s=tube_flow,
        ),
        tube_outlet=StreamConditions(
            temperature_c=tube_out_temp,
            pressure_barg=2.8,
            mass_flow_kg_s=tube_flow,
        ),
        operating_mode=OperatingMode.NORMAL,
        load_percent=100.0,
    )
