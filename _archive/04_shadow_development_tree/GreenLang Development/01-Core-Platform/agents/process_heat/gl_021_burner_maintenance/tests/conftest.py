# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY Test Fixtures

Shared fixtures for all GL-021 test modules.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any


# =============================================================================
# FLAME ANALYSIS FIXTURES
# =============================================================================

@pytest.fixture
def sample_flame_scanner_signal():
    """Generate sample flame scanner signal data."""
    return {
        "signal_type": "UV",
        "intensity": 85.0,
        "frequency": 60.0,
        "noise_level": 2.5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_flame_geometry():
    """Generate sample flame geometry data."""
    return {
        "length_inches": 24.0,
        "width_inches": 8.0,
        "luminosity": 0.85,
        "symmetry_index": 0.92,
        "attachment_point": "center",
    }


@pytest.fixture
def sample_temperature_profile():
    """Generate sample temperature profile data."""
    return {
        "flame_core_temp_f": 2800.0,
        "flame_envelope_temp_f": 1800.0,
        "radiant_section_temp_f": 1650.0,
        "convection_section_temp_f": 1200.0,
        "stack_temp_f": 350.0,
    }


@pytest.fixture
def sample_flame_input(sample_flame_scanner_signal, sample_flame_geometry, sample_temperature_profile):
    """Complete flame analysis input."""
    return {
        "burner_id": "BNR-001",
        "scanner_signal": sample_flame_scanner_signal,
        "geometry": sample_flame_geometry,
        "temperature_profile": sample_temperature_profile,
        "operating_load_pct": 75.0,
    }


# =============================================================================
# BURNER HEALTH FIXTURES
# =============================================================================

@pytest.fixture
def sample_nozzle_data():
    """Sample nozzle component data."""
    return {
        "nozzle_id": "NZL-001",
        "operating_hours": 15000,
        "last_inspection_date": "2024-06-15",
        "wear_percentage": 15.0,
        "pressure_drop_psi": 2.5,
        "spray_angle_deg": 60.0,
        "flow_rate_gph": 10.5,
    }


@pytest.fixture
def sample_refractory_data():
    """Sample refractory tile data."""
    return {
        "tile_id": "REF-001",
        "material_type": "alumina",
        "thickness_inches": 2.0,
        "installed_date": "2023-01-15",
        "operating_hours": 25000,
        "hot_spots_detected": 1,
        "spalling_severity": "minor",
    }


@pytest.fixture
def sample_igniter_data():
    """Sample igniter data."""
    return {
        "igniter_id": "IGN-001",
        "igniter_type": "spark",
        "operating_cycles": 5000,
        "last_test_date": "2024-10-01",
        "success_rate_pct": 98.5,
        "spark_gap_mm": 3.2,
        "response_time_ms": 150,
    }


@pytest.fixture
def sample_flame_scanner_data():
    """Sample flame scanner component data."""
    return {
        "scanner_id": "FS-001",
        "scanner_type": "UV",
        "operating_hours": 20000,
        "signal_strength_pct": 85.0,
        "self_check_status": "pass",
        "lens_condition": "clean",
        "calibration_date": "2024-09-01",
    }


@pytest.fixture
def sample_air_register_data():
    """Sample air register data."""
    return {
        "register_id": "AR-001",
        "position_pct": 65.0,
        "actuator_status": "operational",
        "response_time_sec": 2.5,
        "backlash_deg": 0.5,
        "calibration_date": "2024-08-15",
    }


@pytest.fixture
def sample_fuel_valve_data():
    """Sample fuel valve data."""
    return {
        "valve_id": "FV-001",
        "valve_type": "butterfly",
        "operating_cycles": 100000,
        "leakage_rate_scfh": 0.02,
        "response_time_ms": 200,
        "position_accuracy_pct": 99.5,
        "actuator_torque_pct": 85.0,
    }


@pytest.fixture
def sample_burner_health_input(
    sample_nozzle_data,
    sample_refractory_data,
    sample_igniter_data,
    sample_flame_scanner_data,
    sample_air_register_data,
    sample_fuel_valve_data
):
    """Complete burner health input."""
    return {
        "burner_id": "BNR-001",
        "operating_hours": 25000,
        "nozzle": sample_nozzle_data,
        "refractory": sample_refractory_data,
        "igniter": sample_igniter_data,
        "flame_scanner": sample_flame_scanner_data,
        "air_register": sample_air_register_data,
        "fuel_valve": sample_fuel_valve_data,
    }


# =============================================================================
# MAINTENANCE PREDICTION FIXTURES
# =============================================================================

@pytest.fixture
def sample_failure_history():
    """Sample failure history data."""
    return [
        {"date": "2023-01-15", "component": "igniter", "failure_mode": "electrode_wear", "downtime_hours": 4},
        {"date": "2023-06-20", "component": "nozzle", "failure_mode": "clogging", "downtime_hours": 2},
        {"date": "2024-02-10", "component": "flame_scanner", "failure_mode": "lens_contamination", "downtime_hours": 1},
    ]


@pytest.fixture
def sample_operating_conditions():
    """Sample operating conditions for prediction."""
    return {
        "load_pct": 75.0,
        "ambient_temp_f": 85.0,
        "humidity_pct": 65.0,
        "fuel_type": "natural_gas",
        "cycling_frequency": "medium",
        "startup_count_last_month": 15,
    }


@pytest.fixture
def sample_weibull_params():
    """Sample Weibull parameters."""
    return {
        "shape": 2.5,
        "scale": 25000.0,
        "location": 0.0,
    }


@pytest.fixture
def sample_prediction_input(sample_failure_history, sample_operating_conditions):
    """Complete prediction input."""
    return {
        "burner_id": "BNR-001",
        "current_age_hours": 20000,
        "failure_history": sample_failure_history,
        "operating_conditions": sample_operating_conditions,
        "maintenance_history": [
            {"date": "2024-01-15", "type": "preventive", "components_serviced": ["nozzle", "igniter"]},
            {"date": "2024-06-01", "type": "inspection", "components_serviced": ["flame_scanner"]},
        ],
    }


# =============================================================================
# FUEL IMPACT FIXTURES
# =============================================================================

@pytest.fixture
def sample_natural_gas_properties():
    """Sample natural gas fuel properties."""
    return {
        "fuel_type": "natural_gas",
        "heating_value_btu_scf": 1020,
        "specific_gravity": 0.60,
        "methane_pct": 95.0,
        "ethane_pct": 3.0,
        "propane_pct": 1.0,
        "co2_pct": 0.5,
        "n2_pct": 0.5,
        "h2s_ppm": 0.0,
        "moisture_pct": 0.0,
    }


@pytest.fixture
def sample_fuel_oil_properties():
    """Sample fuel oil properties."""
    return {
        "fuel_type": "no2_oil",
        "heating_value_btu_gal": 140000,
        "api_gravity": 35.0,
        "viscosity_cst_100f": 3.5,
        "sulfur_pct": 0.25,
        "vanadium_ppm": 0.0,
        "sodium_ppm": 0.0,
        "ash_pct": 0.01,
        "water_pct": 0.05,
    }


@pytest.fixture
def sample_heavy_fuel_oil_properties():
    """Sample heavy fuel oil (HFO) properties."""
    return {
        "fuel_type": "hfo",
        "heating_value_btu_gal": 150000,
        "api_gravity": 12.0,
        "viscosity_cst_100f": 500.0,
        "sulfur_pct": 3.5,
        "vanadium_ppm": 150.0,
        "sodium_ppm": 50.0,
        "ash_pct": 0.10,
        "water_pct": 0.5,
    }


# =============================================================================
# EXPLAINABILITY FIXTURES
# =============================================================================

@pytest.fixture
def sample_component_scores():
    """Sample component health scores."""
    return {
        "flame_scanner": 85.0,
        "ignitor": 72.0,
        "fuel_valve": 90.0,
        "air_damper": 95.0,
        "pilot_assembly": 68.0,
        "main_burner": 88.0,
        "combustion_air_fan": 92.0,
        "gas_train": 96.0,
        "flame_stability": 78.0,
        "emission_quality": 82.0,
    }


@pytest.fixture
def sample_historical_scores():
    """Sample historical health scores for trend analysis."""
    return [
        {"flame_scanner": 95.0, "ignitor": 90.0, "fuel_valve": 98.0, "timestamp": "2024-01-01"},
        {"flame_scanner": 92.0, "ignitor": 85.0, "fuel_valve": 95.0, "timestamp": "2024-03-01"},
        {"flame_scanner": 88.0, "ignitor": 78.0, "fuel_valve": 92.0, "timestamp": "2024-06-01"},
        {"flame_scanner": 85.0, "ignitor": 72.0, "fuel_valve": 90.0, "timestamp": "2024-09-01"},
    ]


@pytest.fixture
def sample_feature_data():
    """Sample feature data for ML explainability."""
    return {
        "flame_intensity": 85.0,
        "flame_stability_index": 0.92,
        "flame_color_score": 0.88,
        "fuel_flow_rate": 1000.0,
        "air_fuel_ratio": 10.5,
        "combustion_efficiency": 87.5,
        "nox_emissions": 25.0,
        "co_emissions": 45.0,
        "flue_gas_temp": 380.0,
        "oxygen_level": 3.2,
        "burner_age_hours": 20000,
        "days_since_maintenance": 120,
        "ignition_success_rate": 98.5,
        "flame_scanner_voltage": 4.2,
        "pilot_flame_strength": 92.0,
        "main_valve_response_ms": 180,
        "fuel_pressure": 8.5,
        "air_damper_position": 65.0,
        "combustion_air_flow": 5000.0,
        "heat_release_rate": 25.0,
    }


@pytest.fixture
def sample_feature_array(sample_feature_data):
    """Convert feature data to numpy array."""
    return np.array(list(sample_feature_data.values())).reshape(1, -1)


@pytest.fixture
def sample_feature_names(sample_feature_data):
    """Feature names list."""
    return list(sample_feature_data.keys())


# =============================================================================
# MOCK MODEL FIXTURE
# =============================================================================

class MockBurnerModel:
    """Mock ML model for testing explainability."""

    def __init__(self, prediction_value: float = 0.75):
        self.prediction_value = prediction_value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mock predictions."""
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        return np.full(n_samples, self.prediction_value)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return mock probability predictions."""
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        proba = np.zeros((n_samples, 2))
        proba[:, 0] = 1 - self.prediction_value
        proba[:, 1] = self.prediction_value
        return proba


@pytest.fixture
def mock_model():
    """Mock ML model fixture."""
    return MockBurnerModel(prediction_value=0.75)


@pytest.fixture
def mock_model_low_risk():
    """Mock model predicting low risk."""
    return MockBurnerModel(prediction_value=0.15)


@pytest.fixture
def mock_model_high_risk():
    """Mock model predicting high risk."""
    return MockBurnerModel(prediction_value=0.92)


# =============================================================================
# CMMS INTEGRATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_work_order_input():
    """Sample work order generation input."""
    return {
        "burner_id": "BNR-001",
        "equipment_tag": "BOILER-001-BNR-001",
        "failure_prediction": {
            "component": "igniter",
            "failure_probability": 0.35,
            "rul_hours": 2500,
            "recommended_action": "preventive_replacement",
        },
        "priority": "medium",
        "requested_by": "predictive_maintenance_system",
    }


@pytest.fixture
def sample_cmms_config():
    """Sample CMMS configuration."""
    return {
        "cmms_type": "sap_pm",
        "endpoint": "https://cmms.example.com/api/v1",
        "plant_code": "PLANT-001",
        "work_center": "MAINT-BURNER",
        "default_priority": "2",
        "auto_approval_threshold": "low",
    }


# =============================================================================
# SAFETY COMPLIANCE FIXTURES
# =============================================================================

@pytest.fixture
def sample_safety_config():
    """Sample safety configuration per NFPA 85/86."""
    return {
        "flame_failure_response_time_sec": 4.0,
        "purge_time_sec": 60.0,
        "trial_for_ignition_sec": 10.0,
        "pilot_flame_establishing_time_sec": 10.0,
        "main_flame_establishing_time_sec": 10.0,
        "low_fire_hold_time_sec": 30.0,
        "high_fire_rate_limit_pct_sec": 5.0,
    }


@pytest.fixture
def sample_interlock_status():
    """Sample interlock status data."""
    return {
        "fuel_pressure_low": False,
        "fuel_pressure_high": False,
        "combustion_air_low": False,
        "flame_failure": False,
        "high_temperature": False,
        "purge_not_proven": False,
        "pilot_not_proven": False,
        "atomizing_media_low": False,
    }
