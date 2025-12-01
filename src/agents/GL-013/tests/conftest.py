# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Test Fixtures and Configuration
Comprehensive pytest fixtures for predictive maintenance testing.

This module provides:
- Session and function scoped fixtures
- Equipment data factories
- Vibration data generators
- Maintenance history mocks
- Calculator instances
- Mock connectors for CMMS and CMS
- Provenance validation utilities

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import hashlib
import json
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator, Tuple
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
import random
import math

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance/benchmark tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "determinism: Determinism verification tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on directory structure."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "determinism" in str(item.fspath):
            item.add_marker(pytest.mark.determinism)


# =============================================================================
# EVENT LOOP FIXTURE
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create event loop for async tests.

    Session-scoped to reuse loop across all async tests.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# ENUMS AND CONSTANTS (Local definitions for test isolation)
# =============================================================================


class EquipmentType:
    """Equipment type constants for testing."""
    PUMP = "pump"
    MOTOR = "motor"
    COMPRESSOR = "compressor"
    FAN = "fan"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    CONVEYOR = "conveyor"
    TURBINE = "turbine"


class MachineClass:
    """ISO 10816 machine class constants."""
    CLASS_I = "class_i"
    CLASS_II = "class_ii"
    CLASS_III = "class_iii"
    CLASS_IV = "class_iv"


class VibrationZone:
    """ISO 10816 vibration zones."""
    ZONE_A = "zone_a"
    ZONE_B = "zone_b"
    ZONE_C = "zone_c"
    ZONE_D = "zone_d"


class HealthState:
    """Equipment health states."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class AlertSeverity:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# =============================================================================
# ISO 10816 VIBRATION LIMITS
# =============================================================================


ISO_10816_LIMITS: Dict[str, Dict[str, Decimal]] = {
    MachineClass.CLASS_I: {
        "zone_a_upper": Decimal("0.71"),
        "zone_b_upper": Decimal("1.8"),
        "zone_c_upper": Decimal("4.5"),
    },
    MachineClass.CLASS_II: {
        "zone_a_upper": Decimal("1.12"),
        "zone_b_upper": Decimal("2.8"),
        "zone_c_upper": Decimal("7.1"),
    },
    MachineClass.CLASS_III: {
        "zone_a_upper": Decimal("1.8"),
        "zone_b_upper": Decimal("4.5"),
        "zone_c_upper": Decimal("11.2"),
    },
    MachineClass.CLASS_IV: {
        "zone_a_upper": Decimal("2.8"),
        "zone_b_upper": Decimal("7.1"),
        "zone_c_upper": Decimal("18.0"),
    },
}


# =============================================================================
# WEIBULL PARAMETERS DATABASE
# =============================================================================


WEIBULL_PARAMETERS: Dict[str, Dict[str, Decimal]] = {
    "pump_centrifugal": {
        "beta": Decimal("2.5"),
        "eta": Decimal("45000"),
        "gamma": Decimal("0"),
        "mtbf_hours": Decimal("40000"),
    },
    "motor_ac_induction_large": {
        "beta": Decimal("2.8"),
        "eta": Decimal("60000"),
        "gamma": Decimal("0"),
        "mtbf_hours": Decimal("53000"),
    },
    "gearbox_helical": {
        "beta": Decimal("3.5"),
        "eta": Decimal("70000"),
        "gamma": Decimal("0"),
        "mtbf_hours": Decimal("63000"),
    },
    "bearing_6205": {
        "beta": Decimal("2.0"),
        "eta": Decimal("35000"),
        "gamma": Decimal("0"),
        "mtbf_hours": Decimal("31000"),
    },
    "compressor_reciprocating": {
        "beta": Decimal("2.4"),
        "eta": Decimal("40000"),
        "gamma": Decimal("0"),
        "mtbf_hours": Decimal("35000"),
    },
}


# =============================================================================
# EQUIPMENT DATA FIXTURES
# =============================================================================


@pytest.fixture
def pump_equipment_data() -> Dict[str, Any]:
    """
    Create test data for centrifugal pump equipment.

    Returns complete equipment data dict for pump testing.
    """
    return {
        "equipment_id": "PUMP-001",
        "equipment_type": EquipmentType.PUMP,
        "equipment_subtype": "centrifugal",
        "manufacturer": "Flowserve",
        "model": "HDX-450",
        "serial_number": "FSV-2024-001234",
        "installation_date": "2020-01-15",
        "current_age_hours": Decimal("25000"),
        "rated_power_kw": Decimal("75"),
        "rated_speed_rpm": Decimal("1480"),
        "vibration_velocity_mm_s": Decimal("2.5"),
        "temperature_c": Decimal("65.0"),
        "pressure_bar": Decimal("10.5"),
        "flow_rate_m3_h": Decimal("150"),
        "operating_hours": Decimal("25000"),
        "last_maintenance_date": "2024-06-15",
        "criticality": "high",
        "location": "Plant A - Building 1 - Bay 3",
    }


@pytest.fixture
def motor_equipment_data() -> Dict[str, Any]:
    """Create test data for AC induction motor."""
    return {
        "equipment_id": "MTR-001",
        "equipment_type": EquipmentType.MOTOR,
        "equipment_subtype": "ac_induction_large",
        "manufacturer": "ABB",
        "model": "M3BP-315-SMC",
        "serial_number": "ABB-2019-567890",
        "installation_date": "2019-06-01",
        "current_age_hours": Decimal("40000"),
        "rated_power_kw": Decimal("200"),
        "rated_speed_rpm": Decimal("1485"),
        "rated_voltage_v": Decimal("400"),
        "rated_current_a": Decimal("345"),
        "vibration_velocity_mm_s": Decimal("1.8"),
        "temperature_c": Decimal("72.0"),
        "winding_temperature_c": Decimal("95.0"),
        "bearing_temperature_de_c": Decimal("55.0"),
        "bearing_temperature_nde_c": Decimal("52.0"),
        "operating_hours": Decimal("40000"),
        "insulation_class": "F",
        "efficiency_class": "IE4",
        "criticality": "critical",
    }


@pytest.fixture
def bearing_equipment_data() -> Dict[str, Any]:
    """Create test data for rolling element bearing."""
    return {
        "equipment_id": "BRG-001",
        "equipment_type": EquipmentType.BEARING,
        "bearing_designation": "6205-2RS",
        "manufacturer": "SKF",
        "bore_diameter_mm": Decimal("25"),
        "outer_diameter_mm": Decimal("52"),
        "width_mm": Decimal("15"),
        "number_of_balls": 9,
        "ball_diameter_mm": Decimal("7.938"),
        "pitch_diameter_mm": Decimal("38.5"),
        "contact_angle_deg": Decimal("0"),
        "operating_hours": Decimal("18000"),
        "shaft_speed_rpm": Decimal("1480"),
        "radial_load_kn": Decimal("2.5"),
        "axial_load_kn": Decimal("0.5"),
        "lubrication_type": "grease",
        "temperature_c": Decimal("58.0"),
        "vibration_velocity_mm_s": Decimal("2.2"),
    }


@pytest.fixture
def gearbox_equipment_data() -> Dict[str, Any]:
    """Create test data for helical gearbox."""
    return {
        "equipment_id": "GBX-001",
        "equipment_type": EquipmentType.GEARBOX,
        "equipment_subtype": "helical",
        "manufacturer": "SEW-Eurodrive",
        "model": "X150/R97",
        "serial_number": "SEW-2021-112233",
        "installation_date": "2021-03-20",
        "current_age_hours": Decimal("22000"),
        "input_speed_rpm": Decimal("1480"),
        "output_speed_rpm": Decimal("185"),
        "gear_ratio": Decimal("8.0"),
        "rated_torque_nm": Decimal("5000"),
        "oil_temperature_c": Decimal("68.0"),
        "oil_level_percent": Decimal("95"),
        "vibration_velocity_mm_s": Decimal("3.2"),
        "operating_hours": Decimal("22000"),
        "criticality": "high",
    }


@pytest.fixture
def compressor_equipment_data() -> Dict[str, Any]:
    """Create test data for reciprocating compressor."""
    return {
        "equipment_id": "CMP-001",
        "equipment_type": EquipmentType.COMPRESSOR,
        "equipment_subtype": "reciprocating",
        "manufacturer": "Atlas Copco",
        "model": "GA-250",
        "serial_number": "AC-2022-445566",
        "installation_date": "2022-01-10",
        "current_age_hours": Decimal("15000"),
        "rated_power_kw": Decimal("250"),
        "discharge_pressure_bar": Decimal("10.0"),
        "suction_pressure_bar": Decimal("1.0"),
        "discharge_temperature_c": Decimal("85.0"),
        "oil_temperature_c": Decimal("55.0"),
        "vibration_velocity_mm_s": Decimal("4.5"),
        "operating_hours": Decimal("15000"),
        "criticality": "critical",
    }


# =============================================================================
# VIBRATION DATA FIXTURES
# =============================================================================


@pytest.fixture
def healthy_vibration_data() -> Dict[str, Decimal]:
    """
    Create vibration data for healthy equipment.

    Values well within ISO 10816 Zone A limits.
    """
    return {
        "velocity_rms_mm_s": Decimal("1.5"),
        "velocity_peak_mm_s": Decimal("2.12"),
        "acceleration_rms_g": Decimal("0.5"),
        "acceleration_peak_g": Decimal("0.71"),
        "displacement_pp_um": Decimal("25"),
        "frequency_1x_mm_s": Decimal("1.2"),
        "frequency_2x_mm_s": Decimal("0.3"),
        "frequency_3x_mm_s": Decimal("0.1"),
        "bpfo_amplitude_g": Decimal("0.01"),
        "bpfi_amplitude_g": Decimal("0.01"),
        "bsf_amplitude_g": Decimal("0.005"),
        "ftf_amplitude_g": Decimal("0.005"),
    }


@pytest.fixture
def degraded_vibration_data() -> Dict[str, Decimal]:
    """
    Create vibration data for degraded equipment.

    Values in ISO 10816 Zone B-C, indicating wear.
    """
    return {
        "velocity_rms_mm_s": Decimal("5.5"),
        "velocity_peak_mm_s": Decimal("7.78"),
        "acceleration_rms_g": Decimal("2.0"),
        "acceleration_peak_g": Decimal("2.83"),
        "displacement_pp_um": Decimal("75"),
        "frequency_1x_mm_s": Decimal("4.0"),
        "frequency_2x_mm_s": Decimal("1.5"),
        "frequency_3x_mm_s": Decimal("0.5"),
        "bpfo_amplitude_g": Decimal("0.15"),
        "bpfi_amplitude_g": Decimal("0.1"),
        "bsf_amplitude_g": Decimal("0.05"),
        "ftf_amplitude_g": Decimal("0.03"),
    }


@pytest.fixture
def critical_vibration_data() -> Dict[str, Decimal]:
    """
    Create vibration data for critical equipment condition.

    Values in ISO 10816 Zone D, immediate action required.
    """
    return {
        "velocity_rms_mm_s": Decimal("12.5"),
        "velocity_peak_mm_s": Decimal("17.68"),
        "acceleration_rms_g": Decimal("8.0"),
        "acceleration_peak_g": Decimal("11.31"),
        "displacement_pp_um": Decimal("200"),
        "frequency_1x_mm_s": Decimal("8.0"),
        "frequency_2x_mm_s": Decimal("4.5"),
        "frequency_3x_mm_s": Decimal("2.0"),
        "bpfo_amplitude_g": Decimal("0.8"),
        "bpfi_amplitude_g": Decimal("0.6"),
        "bsf_amplitude_g": Decimal("0.3"),
        "ftf_amplitude_g": Decimal("0.15"),
    }


@pytest.fixture
def vibration_time_series() -> List[Dict[str, Any]]:
    """Generate time series vibration data for trend analysis."""
    base_time = datetime(2024, 1, 1)
    data = []
    base_velocity = 2.0

    for i in range(90):  # 90 days of data
        # Simulate gradual degradation with noise
        degradation = 0.02 * i  # 0.02 mm/s per day increase
        noise = random.gauss(0, 0.1)
        velocity = base_velocity + degradation + noise

        data.append({
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "velocity_rms_mm_s": Decimal(str(round(velocity, 3))),
            "temperature_c": Decimal(str(round(65 + random.gauss(0, 2), 1))),
            "operating_load_percent": Decimal(str(round(75 + random.gauss(0, 5), 1))),
        })

    return data


# =============================================================================
# THERMAL DATA FIXTURES
# =============================================================================


@pytest.fixture
def normal_thermal_data() -> Dict[str, Decimal]:
    """Thermal data within normal operating range."""
    return {
        "ambient_temperature_c": Decimal("25.0"),
        "winding_temperature_c": Decimal("85.0"),
        "hot_spot_temperature_c": Decimal("105.0"),
        "bearing_de_temperature_c": Decimal("55.0"),
        "bearing_nde_temperature_c": Decimal("52.0"),
        "oil_temperature_c": Decimal("65.0"),
        "load_factor": Decimal("0.85"),
    }


@pytest.fixture
def elevated_thermal_data() -> Dict[str, Decimal]:
    """Thermal data showing elevated temperatures."""
    return {
        "ambient_temperature_c": Decimal("35.0"),
        "winding_temperature_c": Decimal("125.0"),
        "hot_spot_temperature_c": Decimal("145.0"),
        "bearing_de_temperature_c": Decimal("75.0"),
        "bearing_nde_temperature_c": Decimal("72.0"),
        "oil_temperature_c": Decimal("85.0"),
        "load_factor": Decimal("1.10"),
    }


@pytest.fixture
def critical_thermal_data() -> Dict[str, Decimal]:
    """Thermal data at critical levels."""
    return {
        "ambient_temperature_c": Decimal("40.0"),
        "winding_temperature_c": Decimal("155.0"),
        "hot_spot_temperature_c": Decimal("175.0"),
        "bearing_de_temperature_c": Decimal("95.0"),
        "bearing_nde_temperature_c": Decimal("90.0"),
        "oil_temperature_c": Decimal("100.0"),
        "load_factor": Decimal("1.25"),
    }


# =============================================================================
# MAINTENANCE HISTORY FIXTURES
# =============================================================================


@pytest.fixture
def maintenance_history() -> List[Dict[str, Any]]:
    """Create sample maintenance history records."""
    return [
        {
            "work_order_id": "WO-2024-001",
            "equipment_id": "PUMP-001",
            "maintenance_type": "preventive",
            "description": "Annual pump inspection and seal replacement",
            "scheduled_date": "2024-01-15",
            "completion_date": "2024-01-15",
            "duration_hours": Decimal("4"),
            "technician": "John Smith",
            "parts_used": ["mechanical seal", "gasket set"],
            "labor_cost": Decimal("400"),
            "parts_cost": Decimal("850"),
            "total_cost": Decimal("1250"),
            "findings": "Minor seal wear detected, replaced as preventive measure",
            "follow_up_required": False,
        },
        {
            "work_order_id": "WO-2024-002",
            "equipment_id": "PUMP-001",
            "maintenance_type": "corrective",
            "description": "Bearing replacement due to elevated vibration",
            "scheduled_date": "2024-06-10",
            "completion_date": "2024-06-11",
            "duration_hours": Decimal("8"),
            "technician": "Jane Doe",
            "parts_used": ["SKF 6205-2RS", "lubricant"],
            "labor_cost": Decimal("800"),
            "parts_cost": Decimal("250"),
            "total_cost": Decimal("1050"),
            "findings": "Bearing showed signs of spalling on outer race",
            "follow_up_required": True,
            "follow_up_date": "2024-07-11",
        },
        {
            "work_order_id": "WO-2023-015",
            "equipment_id": "PUMP-001",
            "maintenance_type": "preventive",
            "description": "Quarterly vibration analysis and lubrication",
            "scheduled_date": "2023-10-01",
            "completion_date": "2023-10-01",
            "duration_hours": Decimal("2"),
            "technician": "John Smith",
            "parts_used": ["lubricant"],
            "labor_cost": Decimal("200"),
            "parts_cost": Decimal("50"),
            "total_cost": Decimal("250"),
            "findings": "All parameters within acceptable limits",
            "follow_up_required": False,
        },
    ]


@pytest.fixture
def failure_history() -> List[Dict[str, Any]]:
    """Create sample failure history records."""
    return [
        {
            "failure_id": "F-2023-001",
            "equipment_id": "PUMP-001",
            "failure_date": "2023-03-15",
            "failure_mode": "bearing_wear",
            "root_cause": "Insufficient lubrication",
            "symptoms": ["elevated_vibration", "noise", "temperature_rise"],
            "downtime_hours": Decimal("12"),
            "production_loss": Decimal("15000"),
            "repair_cost": Decimal("2500"),
            "preventable": True,
            "operating_hours_at_failure": Decimal("18500"),
        },
        {
            "failure_id": "F-2021-003",
            "equipment_id": "PUMP-001",
            "failure_date": "2021-08-20",
            "failure_mode": "seal_failure",
            "root_cause": "Material degradation",
            "symptoms": ["leakage", "pressure_drop"],
            "downtime_hours": Decimal("6"),
            "production_loss": Decimal("8000"),
            "repair_cost": Decimal("1800"),
            "preventable": True,
            "operating_hours_at_failure": Decimal("8500"),
        },
    ]


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> Dict[str, Any]:
    """
    Create default predictive maintenance configuration.

    Returns comprehensive configuration for testing.
    """
    return {
        "agent_id": "GL-013",
        "agent_name": "PREDICTMAINT",
        "version": "1.0.0",
        "deterministic": True,
        "seed": 42,
        "decimal_precision": 6,
        "store_provenance": True,
        "vibration_analysis": {
            "default_machine_class": MachineClass.CLASS_II,
            "alarm_on_zone_c": True,
            "trip_on_zone_d": True,
            "trend_window_days": 30,
        },
        "rul_calculation": {
            "default_model": "weibull",
            "confidence_level": "90%",
            "health_adjustment_enabled": True,
        },
        "maintenance_scheduling": {
            "optimization_window_days": 365,
            "min_interval_hours": 168,  # 1 week minimum
            "max_interval_hours": 8760,  # 1 year maximum
        },
        "anomaly_detection": {
            "threshold_sigma": Decimal("3.0"),
            "cusum_k": Decimal("0.5"),
            "cusum_h": Decimal("5.0"),
        },
        "alerts": {
            "email_enabled": False,
            "sms_enabled": False,
            "webhook_enabled": False,
        },
    }


@pytest.fixture
def test_config_strict() -> Dict[str, Any]:
    """Configuration with strict validation for edge case testing."""
    return {
        "agent_id": "GL-013",
        "deterministic": True,
        "seed": 42,
        "decimal_precision": 12,
        "store_provenance": True,
        "strict_validation": True,
        "fail_on_warning": True,
    }


# =============================================================================
# CALCULATOR FIXTURES
# =============================================================================


@pytest.fixture
def rul_calculator():
    """
    Create RUL Calculator instance.

    Uses mock import for test isolation.
    """
    # Mock implementation for testing
    class MockRULCalculator:
        def __init__(self, precision=6, store_provenance_records=True):
            self.precision = precision
            self.store_provenance = store_provenance_records

        def calculate_weibull_rul(
            self,
            equipment_type: str,
            operating_hours,
            target_reliability="0.5",
            confidence_level="90%",
            health_state=None,
            custom_beta=None,
            custom_eta=None,
            custom_gamma=None
        ):
            """Calculate RUL using Weibull model."""
            t = Decimal(str(operating_hours))
            R_target = Decimal(str(target_reliability))

            # Get parameters
            if custom_beta and custom_eta:
                beta = Decimal(str(custom_beta))
                eta = Decimal(str(custom_eta))
                gamma = Decimal(str(custom_gamma)) if custom_gamma else Decimal("0")
            else:
                params = WEIBULL_PARAMETERS.get(equipment_type, {
                    "beta": Decimal("2.0"),
                    "eta": Decimal("50000"),
                    "gamma": Decimal("0"),
                })
                beta = params["beta"]
                eta = params["eta"]
                gamma = params.get("gamma", Decimal("0"))

            # Calculate time to target reliability
            # t_target = gamma + eta * (-ln(R_target))^(1/beta)
            import math
            neg_ln_R = -Decimal(str(math.log(float(R_target))))
            t_target = gamma + eta * Decimal(str(math.pow(float(neg_ln_R), 1.0/float(beta))))

            rul_hours = max(t_target - t, Decimal("0"))
            rul_days = rul_hours / Decimal("24")
            rul_years = rul_hours / Decimal("8760")

            # Calculate current reliability
            t_effective = max(t - gamma, Decimal("0"))
            current_reliability = Decimal(str(
                math.exp(-math.pow(float(t_effective / eta), float(beta)))
            ))

            # Generate provenance hash
            hash_input = f"{equipment_type}|{t}|{R_target}|{beta}|{eta}|{gamma}"
            provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            return {
                "rul_hours": rul_hours.quantize(Decimal("0.001")),
                "rul_days": rul_days.quantize(Decimal("0.01")),
                "rul_years": rul_years.quantize(Decimal("0.001")),
                "current_reliability": current_reliability.quantize(Decimal("0.000001")),
                "confidence_lower": (rul_hours * Decimal("0.8")).quantize(Decimal("0.001")),
                "confidence_upper": (rul_hours * Decimal("1.2")).quantize(Decimal("0.001")),
                "confidence_level": confidence_level,
                "model_used": "weibull",
                "equipment_type": equipment_type,
                "operating_hours": t,
                "health_state": health_state,
                "health_adjustment": Decimal("1.0"),
                "provenance_hash": provenance_hash,
            }

        def calculate_exponential_rul(
            self,
            failure_rate: Decimal,
            operating_hours: Decimal,
            target_reliability: Decimal = Decimal("0.5"),
        ):
            """Calculate RUL using exponential model."""
            import math
            t = Decimal(str(operating_hours))
            lam = Decimal(str(failure_rate))
            R_target = Decimal(str(target_reliability))

            # t_target = -ln(R_target) / lambda
            t_target = -Decimal(str(math.log(float(R_target)))) / lam
            rul_hours = max(t_target - t, Decimal("0"))

            # Current reliability: R(t) = exp(-lambda * t)
            current_reliability = Decimal(str(math.exp(-float(lam * t))))

            hash_input = f"exp|{lam}|{t}|{R_target}"
            provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            return {
                "rul_hours": rul_hours.quantize(Decimal("0.001")),
                "current_reliability": current_reliability.quantize(Decimal("0.000001")),
                "model_used": "exponential",
                "provenance_hash": provenance_hash,
            }

    return MockRULCalculator()


@pytest.fixture
def vibration_analyzer():
    """Create Vibration Analyzer instance."""
    class MockVibrationAnalyzer:
        def __init__(self, precision=6, store_provenance_records=True):
            self.precision = precision
            self.store_provenance = store_provenance_records

        def assess_severity(
            self,
            velocity_rms,
            machine_class: str,
            measurement_unit: str = "mm/s",
            equipment_id: str = None
        ):
            """Assess vibration severity per ISO 10816."""
            value = Decimal(str(velocity_rms))
            limits = ISO_10816_LIMITS.get(machine_class, ISO_10816_LIMITS[MachineClass.CLASS_II])

            if value <= limits["zone_a_upper"]:
                zone = VibrationZone.ZONE_A
                alarm_level = "normal"
                assessment = "Good"
                margin = limits["zone_a_upper"] - value
            elif value <= limits["zone_b_upper"]:
                zone = VibrationZone.ZONE_B
                alarm_level = "normal"
                assessment = "Acceptable"
                margin = limits["zone_b_upper"] - value
            elif value <= limits["zone_c_upper"]:
                zone = VibrationZone.ZONE_C
                alarm_level = "warning"
                assessment = "Alert"
                margin = limits["zone_c_upper"] - value
            else:
                zone = VibrationZone.ZONE_D
                alarm_level = "critical"
                assessment = "Danger"
                margin = Decimal("0")

            hash_input = f"{value}|{machine_class}"
            provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            return {
                "velocity_rms": value,
                "zone": zone,
                "machine_class": machine_class,
                "zone_limits": limits,
                "alarm_level": alarm_level,
                "assessment": assessment,
                "recommendation": f"Zone {zone[-1].upper()} - {assessment}",
                "margin_to_next_zone": margin,
                "provenance_hash": provenance_hash,
            }

        def calculate_bearing_frequencies(
            self,
            shaft_speed_rpm: Decimal,
            num_balls: int,
            ball_diameter: Decimal,
            pitch_diameter: Decimal,
            contact_angle_deg: Decimal = Decimal("0")
        ):
            """Calculate bearing fault frequencies."""
            import math
            f_s = Decimal(str(shaft_speed_rpm)) / Decimal("60")  # Hz
            n = num_balls
            Bd = Decimal(str(ball_diameter))
            Pd = Decimal(str(pitch_diameter))
            theta = Decimal(str(math.radians(float(contact_angle_deg))))
            cos_theta = Decimal(str(math.cos(float(theta))))

            # BPFO = (n/2) * f_s * (1 - (Bd/Pd) * cos(theta))
            bpfo = (Decimal(str(n)) / Decimal("2")) * f_s * (Decimal("1") - (Bd / Pd) * cos_theta)

            # BPFI = (n/2) * f_s * (1 + (Bd/Pd) * cos(theta))
            bpfi = (Decimal(str(n)) / Decimal("2")) * f_s * (Decimal("1") + (Bd / Pd) * cos_theta)

            # BSF = (Pd/(2*Bd)) * f_s * (1 - ((Bd/Pd)*cos(theta))^2)
            bsf = (Pd / (Decimal("2") * Bd)) * f_s * (Decimal("1") - ((Bd / Pd) * cos_theta) ** 2)

            # FTF = (f_s/2) * (1 - (Bd/Pd) * cos(theta))
            ftf = (f_s / Decimal("2")) * (Decimal("1") - (Bd / Pd) * cos_theta)

            return {
                "shaft_speed_hz": f_s.quantize(Decimal("0.001")),
                "bpfo": bpfo.quantize(Decimal("0.001")),
                "bpfi": bpfi.quantize(Decimal("0.001")),
                "bsf": bsf.quantize(Decimal("0.001")),
                "ftf": ftf.quantize(Decimal("0.001")),
            }

    return MockVibrationAnalyzer()


@pytest.fixture
def failure_probability_calculator():
    """Create Failure Probability Calculator instance."""
    class MockFailureProbabilityCalculator:
        def __init__(self, precision=6, store_provenance_records=True):
            self.precision = precision
            self.store_provenance = store_provenance_records

        def calculate_weibull_failure_probability(
            self,
            beta,
            eta,
            time_hours,
            gamma="0",
            equipment_type=None
        ):
            """Calculate Weibull failure probability."""
            import math
            b = Decimal(str(beta))
            e = Decimal(str(eta))
            t = Decimal(str(time_hours))
            g = Decimal(str(gamma))

            t_eff = t - g
            if t_eff < Decimal("0"):
                return {
                    "failure_probability": Decimal("0"),
                    "reliability": Decimal("1"),
                    "hazard_rate": Decimal("0"),
                }

            # F(t) = 1 - exp(-((t-gamma)/eta)^beta)
            exponent = -((t_eff / e) ** b)
            reliability = Decimal(str(math.exp(float(exponent))))
            failure_prob = Decimal("1") - reliability

            # h(t) = (beta/eta) * ((t-gamma)/eta)^(beta-1)
            if t_eff > Decimal("0"):
                hazard_rate = (b / e) * ((t_eff / e) ** (b - Decimal("1")))
            else:
                hazard_rate = Decimal("0")

            # Mean life (MTBF) for Weibull
            # MTBF = eta * Gamma(1 + 1/beta) where Gamma is gamma function
            gamma_val = Decimal(str(math.gamma(1 + 1/float(b))))
            mean_life = e * gamma_val

            hash_input = f"weibull|{b}|{e}|{t}|{g}"
            provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            return {
                "failure_probability": failure_prob.quantize(Decimal("0.000001")),
                "reliability": reliability.quantize(Decimal("0.000001")),
                "hazard_rate": hazard_rate.quantize(Decimal("0.00000001")),
                "pdf_value": Decimal("0"),
                "cumulative_hazard": (-exponent).quantize(Decimal("0.000001")),
                "mean_life": mean_life.quantize(Decimal("0.001")),
                "time_hours": t,
                "distribution": "weibull",
                "parameters": {"beta": b, "eta": e, "gamma": g},
                "provenance_hash": provenance_hash,
            }

    return MockFailureProbabilityCalculator()


@pytest.fixture
def anomaly_detector():
    """Create Anomaly Detector instance."""
    class MockAnomalyDetector:
        def __init__(self, precision=6, store_provenance_records=True):
            self.precision = precision
            self.store_provenance = store_provenance_records

        def detect_univariate_anomaly(
            self,
            value,
            historical_data: List,
            threshold_sigma="3.0",
            use_mad=False
        ):
            """Detect anomalies using z-score method."""
            x = Decimal(str(value))
            data = [Decimal(str(v)) for v in historical_data]
            threshold = Decimal(str(threshold_sigma))

            # Calculate mean and std
            n = len(data)
            mean = sum(data) / Decimal(str(n))
            variance = sum((d - mean) ** 2 for d in data) / Decimal(str(n))
            import math
            std = Decimal(str(math.sqrt(float(variance))))

            if std == Decimal("0"):
                z_score = Decimal("0")
            else:
                z_score = abs(x - mean) / std

            is_anomaly = z_score > threshold

            # Anomaly score normalized to 0-1
            anomaly_score = min(z_score / (Decimal("2") * threshold), Decimal("1"))

            hash_input = f"anomaly|{x}|{mean}|{std}|{threshold}"
            provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score.quantize(Decimal("0.0001")),
                "anomaly_type": "point_anomaly" if is_anomaly else None,
                "severity": "high" if z_score > 4 else ("medium" if z_score > 3 else "low"),
                "z_score": z_score.quantize(Decimal("0.01")),
                "explanation": f"Value {x} is {z_score:.2f} std devs from mean {mean:.2f}",
                "contributing_factors": ("high_deviation",) if is_anomaly else (),
                "confidence": (Decimal("1") - Decimal("1") / (z_score + Decimal("1"))).quantize(Decimal("0.01")),
                "provenance_hash": provenance_hash,
            }

        def detect_cusum_shift(
            self,
            values: List,
            k: Decimal = Decimal("0.5"),
            h: Decimal = Decimal("5.0"),
        ):
            """Detect mean shift using CUSUM."""
            data = [Decimal(str(v)) for v in values]
            n = len(data)
            mean = sum(data) / Decimal(str(n))

            # Normalize by standard deviation
            variance = sum((d - mean) ** 2 for d in data) / Decimal(str(n))
            import math
            std = Decimal(str(math.sqrt(float(variance))))

            if std == Decimal("0"):
                return {
                    "cusum_upper": Decimal("0"),
                    "cusum_lower": Decimal("0"),
                    "is_out_of_control": False,
                    "shift_detected": False,
                }

            # Calculate CUSUM
            cusum_upper = Decimal("0")
            cusum_lower = Decimal("0")
            max_cusum = Decimal("0")

            for x in data:
                z = (x - mean) / std
                cusum_upper = max(Decimal("0"), cusum_upper + z - k)
                cusum_lower = min(Decimal("0"), cusum_lower + z + k)
                max_cusum = max(max_cusum, cusum_upper, abs(cusum_lower))

            is_out_of_control = max_cusum > h

            return {
                "cusum_upper": cusum_upper.quantize(Decimal("0.01")),
                "cusum_lower": cusum_lower.quantize(Decimal("0.01")),
                "is_out_of_control": is_out_of_control,
                "shift_detected": is_out_of_control,
                "shift_direction": "up" if cusum_upper > abs(cusum_lower) else "down",
                "decision_value": max_cusum.quantize(Decimal("0.01")),
                "threshold": h,
            }

    return MockAnomalyDetector()


@pytest.fixture
def thermal_degradation_calculator():
    """Create Thermal Degradation Calculator instance."""
    class MockThermalDegradationCalculator:
        def __init__(self, precision=6, store_provenance_records=True):
            self.precision = precision
            self.store_provenance = store_provenance_records

        def calculate_arrhenius_aging_factor(
            self,
            operating_temperature_c: Decimal,
            reference_temperature_c: Decimal = Decimal("110"),
            activation_energy_ev: Decimal = Decimal("1.1")
        ):
            """Calculate aging acceleration factor using Arrhenius equation."""
            import math
            T_op = Decimal(str(operating_temperature_c)) + Decimal("273.15")  # To Kelvin
            T_ref = Decimal(str(reference_temperature_c)) + Decimal("273.15")
            Ea = Decimal(str(activation_energy_ev))
            k_B = Decimal("8.617333262e-5")  # Boltzmann constant in eV/K

            # AAF = exp(Ea/k_B * (1/T_ref - 1/T_op))
            exponent = float(Ea / k_B * (Decimal("1") / T_ref - Decimal("1") / T_op))
            aaf = Decimal(str(math.exp(exponent)))

            return {
                "acceleration_factor": aaf.quantize(Decimal("0.001")),
                "operating_temperature_c": operating_temperature_c,
                "reference_temperature_c": reference_temperature_c,
                "activation_energy_ev": Ea,
            }

        def calculate_thermal_life(
            self,
            hot_spot_temperature_c: Decimal,
            operating_hours: Decimal,
            reference_life_hours: Decimal = Decimal("180000"),
            reference_temperature_c: Decimal = Decimal("110"),
        ):
            """Calculate remaining thermal life."""
            aaf_result = self.calculate_arrhenius_aging_factor(
                hot_spot_temperature_c,
                reference_temperature_c,
            )
            aaf = aaf_result["acceleration_factor"]

            equivalent_hours = Decimal(str(operating_hours)) * aaf
            life_consumed_percent = (equivalent_hours / reference_life_hours) * Decimal("100")
            remaining_life_hours = max(reference_life_hours - equivalent_hours, Decimal("0"))
            remaining_life_years = remaining_life_hours / Decimal("8760")

            return {
                "remaining_life_hours": remaining_life_hours.quantize(Decimal("0.001")),
                "remaining_life_years": remaining_life_years.quantize(Decimal("0.001")),
                "aging_acceleration_factor": aaf,
                "equivalent_aging_hours": equivalent_hours.quantize(Decimal("0.001")),
                "life_consumed_percent": life_consumed_percent.quantize(Decimal("0.01")),
                "hot_spot_temperature_c": hot_spot_temperature_c,
                "reference_temperature_c": reference_temperature_c,
            }

    return MockThermalDegradationCalculator()


@pytest.fixture
def maintenance_scheduler():
    """Create Maintenance Scheduler instance."""
    class MockMaintenanceScheduler:
        def __init__(self, precision=6, store_provenance_records=True):
            self.precision = precision
            self.store_provenance = store_provenance_records

        def calculate_optimal_interval(
            self,
            beta: Decimal,
            eta: Decimal,
            preventive_cost: Decimal,
            corrective_cost: Decimal,
        ):
            """Calculate optimal maintenance interval."""
            import math
            b = Decimal(str(beta))
            e = Decimal(str(eta))
            Cp = Decimal(str(preventive_cost))
            Cf = Decimal(str(corrective_cost))

            # Optimal interval using cost ratio method
            # t_opt = eta * ((Cp/Cf) * (beta-1))^(1/beta) for beta > 1
            if b <= Decimal("1"):
                # Run-to-failure is optimal for beta <= 1
                return {
                    "optimal_interval_hours": e,
                    "cost_ratio": Cp / Cf,
                    "strategy": "run_to_failure",
                }

            cost_ratio = Cp / Cf
            inner = cost_ratio * (b - Decimal("1"))
            t_opt = e * Decimal(str(math.pow(float(inner), 1/float(b))))

            return {
                "optimal_interval_hours": t_opt.quantize(Decimal("0.001")),
                "cost_ratio": cost_ratio,
                "strategy": "time_based_preventive",
                "expected_cost_per_hour": (Cp / t_opt).quantize(Decimal("0.0001")),
            }

    return MockMaintenanceScheduler()


@pytest.fixture
def spare_parts_calculator():
    """Create Spare Parts Calculator instance."""
    class MockSparePartsCalculator:
        def __init__(self, precision=6, store_provenance_records=True):
            self.precision = precision
            self.store_provenance = store_provenance_records

        def calculate_eoq(
            self,
            annual_demand: Decimal,
            ordering_cost: Decimal,
            holding_cost_rate: Decimal,
            unit_cost: Decimal,
        ):
            """Calculate Economic Order Quantity."""
            import math
            D = Decimal(str(annual_demand))
            S = Decimal(str(ordering_cost))
            H = Decimal(str(holding_cost_rate)) * Decimal(str(unit_cost))

            # EOQ = sqrt(2*D*S / H)
            eoq = Decimal(str(math.sqrt(float(2 * D * S / H))))

            # Number of orders per year
            orders_per_year = D / eoq

            # Total cost
            total_ordering_cost = orders_per_year * S
            total_holding_cost = (eoq / Decimal("2")) * H
            total_cost = total_ordering_cost + total_holding_cost

            return {
                "eoq": eoq.quantize(Decimal("1")),
                "orders_per_year": orders_per_year.quantize(Decimal("0.1")),
                "total_ordering_cost": total_ordering_cost.quantize(Decimal("0.01")),
                "total_holding_cost": total_holding_cost.quantize(Decimal("0.01")),
                "total_cost": total_cost.quantize(Decimal("0.01")),
            }

        def calculate_safety_stock(
            self,
            demand_std_dev: Decimal,
            lead_time_days: Decimal,
            service_level: Decimal = Decimal("0.95"),
        ):
            """Calculate safety stock for target service level."""
            import math
            from scipy.stats import norm

            sigma = Decimal(str(demand_std_dev))
            L = Decimal(str(lead_time_days))

            # Z-score for service level
            z = Decimal(str(norm.ppf(float(service_level))))

            # Safety stock = z * sigma * sqrt(L)
            safety_stock = z * sigma * Decimal(str(math.sqrt(float(L))))

            return {
                "safety_stock": safety_stock.quantize(Decimal("1")),
                "z_score": z.quantize(Decimal("0.001")),
                "service_level": service_level,
            }

    return MockSparePartsCalculator()


# =============================================================================
# MOCK CONNECTOR FIXTURES
# =============================================================================


@pytest.fixture
def mock_cmms_connector():
    """Create mock CMMS connector for integration testing."""
    connector = Mock()

    # Configure mock methods
    connector.get_equipment.return_value = {
        "equipment_id": "PUMP-001",
        "equipment_type": "pump",
        "status": "active",
    }

    connector.get_work_orders.return_value = [
        {"wo_id": "WO-001", "status": "completed"},
        {"wo_id": "WO-002", "status": "open"},
    ]

    connector.create_work_order = Mock(return_value="WO-003")
    connector.update_equipment = Mock(return_value=True)

    # Async methods
    connector.get_equipment_async = AsyncMock(return_value={
        "equipment_id": "PUMP-001",
        "equipment_type": "pump",
    })

    return connector


@pytest.fixture
def mock_cms_connector():
    """Create mock Condition Monitoring System connector."""
    connector = Mock()

    # Configure sensor data retrieval
    connector.get_latest_readings.return_value = {
        "vibration_velocity_mm_s": Decimal("2.5"),
        "temperature_c": Decimal("65.0"),
        "timestamp": datetime.now().isoformat(),
    }

    connector.get_historical_data = Mock(return_value=[
        {"timestamp": "2024-01-01T00:00:00", "value": 2.0},
        {"timestamp": "2024-01-02T00:00:00", "value": 2.1},
        {"timestamp": "2024-01-03T00:00:00", "value": 2.2},
    ])

    connector.get_alarm_history = Mock(return_value=[])

    return connector


@pytest.fixture
def mock_database():
    """Create mock database connection."""
    db = Mock()

    db.execute = Mock(return_value=None)
    db.fetchone = Mock(return_value={"id": 1, "value": "test"})
    db.fetchall = Mock(return_value=[{"id": 1}, {"id": 2}])
    db.commit = Mock(return_value=None)
    db.rollback = Mock(return_value=None)

    return db


# =============================================================================
# PROVENANCE UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def provenance_validator():
    """Create provenance validation utility."""
    class ProvenanceValidator:
        def validate_hash(self, data: Dict[str, Any], expected_hash: str) -> bool:
            """Validate SHA-256 hash of data."""
            json_data = json.dumps(data, sort_keys=True, default=str)
            computed_hash = hashlib.sha256(json_data.encode()).hexdigest()
            return computed_hash == expected_hash

        def compute_hash(self, data: Dict[str, Any]) -> str:
            """Compute SHA-256 hash of data."""
            json_data = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_data.encode()).hexdigest()

        def verify_merkle_root(self, leaves: List[str], root: str) -> bool:
            """Verify Merkle tree root."""
            if not leaves:
                return root == hashlib.sha256(b"").hexdigest()

            current_level = leaves[:]
            while len(current_level) > 1:
                next_level = []
                for i in range(0, len(current_level), 2):
                    left = current_level[i]
                    right = current_level[i + 1] if i + 1 < len(current_level) else left
                    combined = hashlib.sha256(
                        (left + right).encode()
                    ).hexdigest()
                    next_level.append(combined)
                current_level = next_level

            return current_level[0] == root

    return ProvenanceValidator()


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


@pytest.fixture
def equipment_data_generator():
    """Factory for generating test equipment data."""
    class EquipmentDataGenerator:
        def __init__(self, seed: int = 42):
            self.rng = random.Random(seed)

        def generate_pump_data(self, num_records: int = 10) -> List[Dict[str, Any]]:
            """Generate pump equipment records."""
            records = []
            for i in range(num_records):
                records.append({
                    "equipment_id": f"PUMP-{i+1:03d}",
                    "equipment_type": "pump",
                    "operating_hours": Decimal(str(self.rng.randint(1000, 50000))),
                    "vibration_velocity_mm_s": Decimal(str(round(self.rng.uniform(0.5, 8.0), 2))),
                    "temperature_c": Decimal(str(round(self.rng.uniform(50, 90), 1))),
                })
            return records

        def generate_vibration_data(
            self,
            num_points: int = 100,
            base_value: float = 2.0,
            trend: float = 0.0,
        ) -> List[Decimal]:
            """Generate vibration time series."""
            data = []
            for i in range(num_points):
                value = base_value + trend * i + self.rng.gauss(0, 0.1)
                data.append(Decimal(str(round(max(0, value), 3))))
            return data

    return EquipmentDataGenerator()


@pytest.fixture
def random_seed():
    """Provide consistent random seed for reproducibility."""
    return 42


# =============================================================================
# CLEANUP AND TEARDOWN
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state between tests."""
    yield
    # Cleanup code here if needed


@pytest.fixture
def temp_provenance_store(tmp_path):
    """Create temporary provenance store for testing."""
    store_path = tmp_path / "provenance"
    store_path.mkdir()
    return store_path


# =============================================================================
# PARAMETRIZE HELPERS
# =============================================================================


def equipment_types():
    """Return list of equipment types for parametrization."""
    return [
        EquipmentType.PUMP,
        EquipmentType.MOTOR,
        EquipmentType.COMPRESSOR,
        EquipmentType.FAN,
        EquipmentType.GEARBOX,
    ]


def machine_classes():
    """Return list of machine classes for parametrization."""
    return [
        MachineClass.CLASS_I,
        MachineClass.CLASS_II,
        MachineClass.CLASS_III,
        MachineClass.CLASS_IV,
    ]


def health_states():
    """Return list of health states for parametrization."""
    return [
        HealthState.EXCELLENT,
        HealthState.GOOD,
        HealthState.FAIR,
        HealthState.POOR,
        HealthState.CRITICAL,
    ]
