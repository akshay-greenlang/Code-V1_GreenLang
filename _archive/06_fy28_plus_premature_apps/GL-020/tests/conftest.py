"""
GL-020 ECONOPULSE - Test Configuration and Fixtures

Provides shared fixtures, test data generators, and test utilities
for comprehensive testing of EconomizerPerformanceAgent calculators.

Author: GL-TestEngineer
Version: 1.0.0
"""

import sys
import os
import json
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, time, timedelta, timezone
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
import math

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EconomizerType(str, Enum):
    """Types of economizers."""
    BARE_TUBE = "bare_tube"
    FINNED_TUBE = "finned_tube"
    EXTENDED_SURFACE = "extended_surface"
    CONDENSING = "condensing"
    SPIRAL = "spiral"


class FlowArrangement(str, Enum):
    """Heat exchanger flow arrangements."""
    COUNTER_FLOW = "counter_flow"
    PARALLEL_FLOW = "parallel_flow"
    CROSS_FLOW = "cross_flow"
    MIXED_FLOW = "mixed_flow"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(str, Enum):
    """Types of alerts."""
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    TREND = "trend"
    ANOMALY = "anomaly"


class FoulingLevel(str, Enum):
    """Fouling severity levels."""
    CLEAN = "clean"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"


# =============================================================================
# DATA CLASSES FOR TEST INPUTS
# =============================================================================

@dataclass
class TemperatureReading:
    """Temperature sensor reading."""
    sensor_id: str
    location: str  # gas_inlet, gas_outlet, water_inlet, water_outlet
    timestamp: datetime
    value_c: float
    quality: float = 1.0  # 0-1 data quality score
    unit: str = "C"


@dataclass
class FlowReading:
    """Flow rate sensor reading."""
    sensor_id: str
    fluid_type: str  # flue_gas, water, steam
    timestamp: datetime
    value: float
    unit: str = "kg/s"
    quality: float = 1.0


@dataclass
class PressureReading:
    """Pressure sensor reading."""
    sensor_id: str
    location: str
    timestamp: datetime
    value_kpa: float
    quality: float = 1.0


@dataclass
class EconomizerConfig:
    """Economizer configuration."""
    economizer_id: str
    name: str
    economizer_type: EconomizerType
    flow_arrangement: FlowArrangement
    design_heat_duty_kw: float
    design_u_value_w_m2k: float
    heat_transfer_area_m2: float
    tube_diameter_mm: float = 50.0
    tube_count: int = 100
    tube_passes: int = 2
    fin_density_per_m: float = 0.0  # For finned tubes
    design_water_flow_kg_s: float = 10.0
    design_gas_flow_kg_s: float = 15.0
    design_water_inlet_c: float = 105.0
    design_water_outlet_c: float = 140.0
    design_gas_inlet_c: float = 350.0
    design_gas_outlet_c: float = 180.0
    min_approach_temp_c: float = 20.0
    commissioning_date: date = field(default_factory=date.today)


@dataclass
class AlertConfig:
    """Alert configuration."""
    alert_id: str
    name: str
    alert_type: AlertType
    parameter: str  # fouling_factor, effectiveness, u_value, etc.
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    rate_of_change_limit: Optional[float] = None  # per hour
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    cooldown_minutes: int = 60
    deduplication_window_minutes: int = 30


@dataclass
class PerformanceBaseline:
    """Clean baseline performance data."""
    economizer_id: str
    baseline_date: date
    clean_u_value_w_m2k: float
    clean_effectiveness: float
    clean_heat_duty_kw: float
    clean_approach_temp_c: float
    clean_gas_pressure_drop_kpa: float
    clean_water_pressure_drop_kpa: float
    reference_conditions: Dict[str, float] = field(default_factory=dict)


@dataclass
class FoulingData:
    """Fouling measurement data."""
    economizer_id: str
    timestamp: datetime
    fouling_factor_m2k_w: float
    fouling_level: FoulingLevel
    efficiency_loss_pct: float
    estimated_fuel_penalty_pct: float
    days_since_cleaning: int
    cleaning_recommended: bool
    estimated_days_to_cleaning: int


@dataclass
class HeatTransferResult:
    """Heat transfer calculation result."""
    heat_duty_kw: float
    lmtd_c: float
    u_value_w_m2k: float
    effectiveness: float
    approach_temp_c: float
    ntu: float
    calculation_timestamp: datetime
    provenance_hash: str


@dataclass
class Alert:
    """Generated alert."""
    alert_id: str
    economizer_id: str
    alert_type: AlertType
    severity: AlertSeverity
    parameter: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# =============================================================================
# ECONOMIZER CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def bare_tube_economizer():
    """Bare tube economizer configuration."""
    return EconomizerConfig(
        economizer_id="ECON-001",
        name="Primary Boiler Economizer",
        economizer_type=EconomizerType.BARE_TUBE,
        flow_arrangement=FlowArrangement.COUNTER_FLOW,
        design_heat_duty_kw=2500.0,
        design_u_value_w_m2k=45.0,
        heat_transfer_area_m2=350.0,
        tube_diameter_mm=50.8,
        tube_count=120,
        tube_passes=2,
        design_water_flow_kg_s=12.5,
        design_gas_flow_kg_s=18.0,
        design_water_inlet_c=105.0,
        design_water_outlet_c=145.0,
        design_gas_inlet_c=380.0,
        design_gas_outlet_c=175.0,
        min_approach_temp_c=25.0
    )


@pytest.fixture
def finned_tube_economizer():
    """Finned tube economizer configuration."""
    return EconomizerConfig(
        economizer_id="ECON-002",
        name="Secondary Economizer with Fins",
        economizer_type=EconomizerType.FINNED_TUBE,
        flow_arrangement=FlowArrangement.COUNTER_FLOW,
        design_heat_duty_kw=3500.0,
        design_u_value_w_m2k=55.0,
        heat_transfer_area_m2=420.0,
        tube_diameter_mm=38.1,
        tube_count=180,
        tube_passes=4,
        fin_density_per_m=275.0,
        design_water_flow_kg_s=15.0,
        design_gas_flow_kg_s=22.0,
        design_water_inlet_c=100.0,
        design_water_outlet_c=150.0,
        design_gas_inlet_c=420.0,
        design_gas_outlet_c=165.0,
        min_approach_temp_c=20.0
    )


@pytest.fixture
def extended_surface_economizer():
    """Extended surface economizer configuration."""
    return EconomizerConfig(
        economizer_id="ECON-003",
        name="High Efficiency Extended Surface",
        economizer_type=EconomizerType.EXTENDED_SURFACE,
        flow_arrangement=FlowArrangement.CROSS_FLOW,
        design_heat_duty_kw=5000.0,
        design_u_value_w_m2k=65.0,
        heat_transfer_area_m2=600.0,
        tube_diameter_mm=31.75,
        tube_count=250,
        tube_passes=6,
        fin_density_per_m=350.0,
        design_water_flow_kg_s=25.0,
        design_gas_flow_kg_s=35.0,
        design_water_inlet_c=95.0,
        design_water_outlet_c=155.0,
        design_gas_inlet_c=450.0,
        design_gas_outlet_c=155.0,
        min_approach_temp_c=15.0
    )


@pytest.fixture
def condensing_economizer():
    """Condensing economizer configuration (for low temperature heat recovery)."""
    return EconomizerConfig(
        economizer_id="ECON-004",
        name="Condensing Flue Gas Economizer",
        economizer_type=EconomizerType.CONDENSING,
        flow_arrangement=FlowArrangement.COUNTER_FLOW,
        design_heat_duty_kw=1500.0,
        design_u_value_w_m2k=80.0,
        heat_transfer_area_m2=200.0,
        tube_diameter_mm=25.4,
        tube_count=300,
        tube_passes=4,
        design_water_flow_kg_s=8.0,
        design_gas_flow_kg_s=12.0,
        design_water_inlet_c=40.0,
        design_water_outlet_c=80.0,
        design_gas_inlet_c=180.0,
        design_gas_outlet_c=55.0,
        min_approach_temp_c=10.0
    )


@pytest.fixture
def multiple_economizers(bare_tube_economizer, finned_tube_economizer, extended_surface_economizer):
    """Multiple economizer configurations for multi-unit tests."""
    return [bare_tube_economizer, finned_tube_economizer, extended_surface_economizer]


# =============================================================================
# TEMPERATURE READING FIXTURES
# =============================================================================

@pytest.fixture
def clean_operation_temperatures():
    """Temperature readings for clean economizer operation."""
    now = datetime.now(timezone.utc)
    return {
        "gas_inlet": TemperatureReading(
            sensor_id="TT-GAS-IN-001",
            location="gas_inlet",
            timestamp=now,
            value_c=380.0,
            quality=0.99
        ),
        "gas_outlet": TemperatureReading(
            sensor_id="TT-GAS-OUT-001",
            location="gas_outlet",
            timestamp=now,
            value_c=175.0,
            quality=0.99
        ),
        "water_inlet": TemperatureReading(
            sensor_id="TT-WATER-IN-001",
            location="water_inlet",
            timestamp=now,
            value_c=105.0,
            quality=0.99
        ),
        "water_outlet": TemperatureReading(
            sensor_id="TT-WATER-OUT-001",
            location="water_outlet",
            timestamp=now,
            value_c=145.0,
            quality=0.99
        )
    }


@pytest.fixture
def fouled_operation_temperatures():
    """Temperature readings for fouled economizer operation."""
    now = datetime.now(timezone.utc)
    return {
        "gas_inlet": TemperatureReading(
            sensor_id="TT-GAS-IN-001",
            location="gas_inlet",
            timestamp=now,
            value_c=380.0,
            quality=0.99
        ),
        "gas_outlet": TemperatureReading(
            sensor_id="TT-GAS-OUT-001",
            location="gas_outlet",
            timestamp=now,
            value_c=210.0,  # Higher than clean - less heat transfer
            quality=0.99
        ),
        "water_inlet": TemperatureReading(
            sensor_id="TT-WATER-IN-001",
            location="water_inlet",
            timestamp=now,
            value_c=105.0,
            quality=0.99
        ),
        "water_outlet": TemperatureReading(
            sensor_id="TT-WATER-OUT-001",
            location="water_outlet",
            timestamp=now,
            value_c=132.0,  # Lower than clean - less heat transfer
            quality=0.99
        )
    }


@pytest.fixture
def varying_load_temperatures():
    """Temperature readings at different load levels."""
    now = datetime.now(timezone.utc)

    def create_readings(load_pct: float, gas_in: float, gas_out: float, water_in: float, water_out: float):
        return {
            "load_pct": load_pct,
            "gas_inlet": TemperatureReading(
                sensor_id="TT-GAS-IN-001",
                location="gas_inlet",
                timestamp=now,
                value_c=gas_in,
                quality=0.99
            ),
            "gas_outlet": TemperatureReading(
                sensor_id="TT-GAS-OUT-001",
                location="gas_outlet",
                timestamp=now,
                value_c=gas_out,
                quality=0.99
            ),
            "water_inlet": TemperatureReading(
                sensor_id="TT-WATER-IN-001",
                location="water_inlet",
                timestamp=now,
                value_c=water_in,
                quality=0.99
            ),
            "water_outlet": TemperatureReading(
                sensor_id="TT-WATER-OUT-001",
                location="water_outlet",
                timestamp=now,
                value_c=water_out,
                quality=0.99
            )
        }

    return [
        create_readings(100.0, 380.0, 175.0, 105.0, 145.0),
        create_readings(75.0, 350.0, 170.0, 105.0, 138.0),
        create_readings(50.0, 320.0, 165.0, 105.0, 130.0),
        create_readings(25.0, 280.0, 160.0, 105.0, 120.0),
    ]


@pytest.fixture
def edge_case_temperatures():
    """Edge case temperature readings."""
    now = datetime.now(timezone.utc)
    return {
        "equal_inlet_temps": {
            "gas_inlet": TemperatureReading(
                sensor_id="TT-GAS-IN-001", location="gas_inlet",
                timestamp=now, value_c=150.0, quality=0.99
            ),
            "gas_outlet": TemperatureReading(
                sensor_id="TT-GAS-OUT-001", location="gas_outlet",
                timestamp=now, value_c=120.0, quality=0.99
            ),
            "water_inlet": TemperatureReading(
                sensor_id="TT-WATER-IN-001", location="water_inlet",
                timestamp=now, value_c=150.0, quality=0.99  # Same as gas inlet
            ),
            "water_outlet": TemperatureReading(
                sensor_id="TT-WATER-OUT-001", location="water_outlet",
                timestamp=now, value_c=145.0, quality=0.99
            )
        },
        "very_small_delta_t": {
            "gas_inlet": TemperatureReading(
                sensor_id="TT-GAS-IN-001", location="gas_inlet",
                timestamp=now, value_c=200.0, quality=0.99
            ),
            "gas_outlet": TemperatureReading(
                sensor_id="TT-GAS-OUT-001", location="gas_outlet",
                timestamp=now, value_c=198.0, quality=0.99  # Only 2C drop
            ),
            "water_inlet": TemperatureReading(
                sensor_id="TT-WATER-IN-001", location="water_inlet",
                timestamp=now, value_c=100.0, quality=0.99
            ),
            "water_outlet": TemperatureReading(
                sensor_id="TT-WATER-OUT-001", location="water_outlet",
                timestamp=now, value_c=101.0, quality=0.99  # Only 1C rise
            )
        },
        "zero_delta_t": {
            "gas_inlet": TemperatureReading(
                sensor_id="TT-GAS-IN-001", location="gas_inlet",
                timestamp=now, value_c=200.0, quality=0.99
            ),
            "gas_outlet": TemperatureReading(
                sensor_id="TT-GAS-OUT-001", location="gas_outlet",
                timestamp=now, value_c=200.0, quality=0.99  # Zero drop
            ),
            "water_inlet": TemperatureReading(
                sensor_id="TT-WATER-IN-001", location="water_inlet",
                timestamp=now, value_c=100.0, quality=0.99
            ),
            "water_outlet": TemperatureReading(
                sensor_id="TT-WATER-OUT-001", location="water_outlet",
                timestamp=now, value_c=100.0, quality=0.99  # Zero rise
            )
        }
    }


# =============================================================================
# FLOW READING FIXTURES
# =============================================================================

@pytest.fixture
def design_flow_readings():
    """Flow readings at design conditions."""
    now = datetime.now(timezone.utc)
    return {
        "water": FlowReading(
            sensor_id="FT-WATER-001",
            fluid_type="water",
            timestamp=now,
            value=12.5,
            unit="kg/s",
            quality=0.99
        ),
        "flue_gas": FlowReading(
            sensor_id="FT-GAS-001",
            fluid_type="flue_gas",
            timestamp=now,
            value=18.0,
            unit="kg/s",
            quality=0.99
        )
    }


@pytest.fixture
def reduced_flow_readings():
    """Flow readings at reduced load."""
    now = datetime.now(timezone.utc)
    return {
        "water": FlowReading(
            sensor_id="FT-WATER-001",
            fluid_type="water",
            timestamp=now,
            value=8.0,  # 64% of design
            unit="kg/s",
            quality=0.99
        ),
        "flue_gas": FlowReading(
            sensor_id="FT-GAS-001",
            fluid_type="flue_gas",
            timestamp=now,
            value=12.0,  # 67% of design
            unit="kg/s",
            quality=0.99
        )
    }


@pytest.fixture
def zero_flow_readings():
    """Zero flow readings for edge case testing."""
    now = datetime.now(timezone.utc)
    return {
        "water": FlowReading(
            sensor_id="FT-WATER-001",
            fluid_type="water",
            timestamp=now,
            value=0.0,
            unit="kg/s",
            quality=0.99
        ),
        "flue_gas": FlowReading(
            sensor_id="FT-GAS-001",
            fluid_type="flue_gas",
            timestamp=now,
            value=0.0,
            unit="kg/s",
            quality=0.99
        )
    }


# =============================================================================
# BASELINE PERFORMANCE FIXTURES
# =============================================================================

@pytest.fixture
def clean_baseline_performance(bare_tube_economizer):
    """Clean baseline performance data."""
    return PerformanceBaseline(
        economizer_id=bare_tube_economizer.economizer_id,
        baseline_date=date(2024, 1, 15),
        clean_u_value_w_m2k=45.0,
        clean_effectiveness=0.745,
        clean_heat_duty_kw=2500.0,
        clean_approach_temp_c=30.0,
        clean_gas_pressure_drop_kpa=0.8,
        clean_water_pressure_drop_kpa=25.0,
        reference_conditions={
            "gas_flow_kg_s": 18.0,
            "water_flow_kg_s": 12.5,
            "gas_inlet_c": 380.0,
            "water_inlet_c": 105.0
        }
    )


@pytest.fixture
def fouled_baseline_performance(bare_tube_economizer):
    """Fouled economizer performance data for comparison."""
    return PerformanceBaseline(
        economizer_id=bare_tube_economizer.economizer_id,
        baseline_date=date(2024, 6, 15),
        clean_u_value_w_m2k=32.0,  # Degraded from 45
        clean_effectiveness=0.58,  # Degraded from 0.745
        clean_heat_duty_kw=1950.0,  # Degraded from 2500
        clean_approach_temp_c=45.0,  # Increased from 30
        clean_gas_pressure_drop_kpa=1.4,  # Increased from 0.8
        clean_water_pressure_drop_kpa=28.0,
        reference_conditions={
            "gas_flow_kg_s": 18.0,
            "water_flow_kg_s": 12.5,
            "gas_inlet_c": 380.0,
            "water_inlet_c": 105.0
        }
    )


# =============================================================================
# FOULING DATA FIXTURES
# =============================================================================

@pytest.fixture
def clean_fouling_data():
    """Fouling data for clean economizer."""
    return FoulingData(
        economizer_id="ECON-001",
        timestamp=datetime.now(timezone.utc),
        fouling_factor_m2k_w=0.0001,  # Very low fouling
        fouling_level=FoulingLevel.CLEAN,
        efficiency_loss_pct=0.5,
        estimated_fuel_penalty_pct=0.1,
        days_since_cleaning=5,
        cleaning_recommended=False,
        estimated_days_to_cleaning=180
    )


@pytest.fixture
def moderate_fouling_data():
    """Fouling data for moderately fouled economizer."""
    return FoulingData(
        economizer_id="ECON-001",
        timestamp=datetime.now(timezone.utc),
        fouling_factor_m2k_w=0.0005,  # Moderate fouling
        fouling_level=FoulingLevel.MODERATE,
        efficiency_loss_pct=8.5,
        estimated_fuel_penalty_pct=1.8,
        days_since_cleaning=120,
        cleaning_recommended=False,
        estimated_days_to_cleaning=45
    )


@pytest.fixture
def severe_fouling_data():
    """Fouling data for severely fouled economizer."""
    return FoulingData(
        economizer_id="ECON-001",
        timestamp=datetime.now(timezone.utc),
        fouling_factor_m2k_w=0.0012,  # Severe fouling
        fouling_level=FoulingLevel.SEVERE,
        efficiency_loss_pct=22.0,
        estimated_fuel_penalty_pct=4.5,
        days_since_cleaning=365,
        cleaning_recommended=True,
        estimated_days_to_cleaning=0
    )


@pytest.fixture
def fouling_trend_data():
    """Historical fouling data for trend analysis."""
    base_time = datetime.now(timezone.utc)
    data = []

    for days_ago in range(0, 180, 7):  # Weekly data for 6 months
        timestamp = base_time - timedelta(days=days_ago)
        # Simulate gradual fouling increase
        fouling_factor = 0.0001 + (days_ago / 180) * 0.0008
        efficiency_loss = 0.5 + (days_ago / 180) * 15.0

        if fouling_factor < 0.0002:
            level = FoulingLevel.CLEAN
        elif fouling_factor < 0.0004:
            level = FoulingLevel.LIGHT
        elif fouling_factor < 0.0007:
            level = FoulingLevel.MODERATE
        elif fouling_factor < 0.001:
            level = FoulingLevel.HEAVY
        else:
            level = FoulingLevel.SEVERE

        data.append(FoulingData(
            economizer_id="ECON-001",
            timestamp=timestamp,
            fouling_factor_m2k_w=fouling_factor,
            fouling_level=level,
            efficiency_loss_pct=efficiency_loss,
            estimated_fuel_penalty_pct=efficiency_loss * 0.2,
            days_since_cleaning=days_ago,
            cleaning_recommended=fouling_factor > 0.0008,
            estimated_days_to_cleaning=max(0, 180 - days_ago)
        ))

    return sorted(data, key=lambda x: x.timestamp)


# =============================================================================
# ALERT CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def fouling_alert_config():
    """Alert configuration for fouling monitoring."""
    return AlertConfig(
        alert_id="ALERT-FOULING-001",
        name="High Fouling Factor Alert",
        alert_type=AlertType.THRESHOLD,
        parameter="fouling_factor_m2k_w",
        threshold_high=0.0008,
        severity=AlertSeverity.WARNING,
        enabled=True,
        cooldown_minutes=120,
        deduplication_window_minutes=60
    )


@pytest.fixture
def effectiveness_alert_config():
    """Alert configuration for effectiveness monitoring."""
    return AlertConfig(
        alert_id="ALERT-EFF-001",
        name="Low Effectiveness Alert",
        alert_type=AlertType.THRESHOLD,
        parameter="effectiveness",
        threshold_low=0.60,
        severity=AlertSeverity.WARNING,
        enabled=True,
        cooldown_minutes=60,
        deduplication_window_minutes=30
    )


@pytest.fixture
def rate_of_change_alert_config():
    """Alert configuration for rate of change monitoring."""
    return AlertConfig(
        alert_id="ALERT-ROC-001",
        name="Rapid Fouling Increase Alert",
        alert_type=AlertType.RATE_OF_CHANGE,
        parameter="fouling_factor_m2k_w",
        rate_of_change_limit=0.0001,  # Max 0.0001 m2K/W per hour
        severity=AlertSeverity.CRITICAL,
        enabled=True,
        cooldown_minutes=30,
        deduplication_window_minutes=15
    )


@pytest.fixture
def multiple_alert_configs(fouling_alert_config, effectiveness_alert_config, rate_of_change_alert_config):
    """Multiple alert configurations."""
    return [
        fouling_alert_config,
        effectiveness_alert_config,
        rate_of_change_alert_config,
        AlertConfig(
            alert_id="ALERT-APPROACH-001",
            name="High Approach Temperature Alert",
            alert_type=AlertType.THRESHOLD,
            parameter="approach_temp_c",
            threshold_high=50.0,
            severity=AlertSeverity.WARNING,
            enabled=True,
            cooldown_minutes=60,
            deduplication_window_minutes=30
        ),
        AlertConfig(
            alert_id="ALERT-UVALUE-001",
            name="Critical U-Value Drop Alert",
            alert_type=AlertType.THRESHOLD,
            parameter="u_value_w_m2k",
            threshold_low=25.0,
            severity=AlertSeverity.CRITICAL,
            enabled=True,
            cooldown_minutes=30,
            deduplication_window_minutes=15
        ),
    ]


# =============================================================================
# ASME PTC 4.3 TEST CASES
# =============================================================================

@pytest.fixture
def asme_ptc_43_test_cases():
    """
    Test cases based on ASME PTC 4.3 examples for economizer testing.
    These provide known inputs and expected outputs for validation.
    """
    return [
        {
            "name": "ASME PTC 4.3 Example 1 - Counter Flow",
            "flow_arrangement": FlowArrangement.COUNTER_FLOW,
            "gas_inlet_c": 400.0,
            "gas_outlet_c": 200.0,
            "water_inlet_c": 120.0,
            "water_outlet_c": 180.0,
            "gas_flow_kg_s": 20.0,
            "water_flow_kg_s": 15.0,
            "gas_cp_kj_kg_k": 1.1,
            "water_cp_kj_kg_k": 4.2,
            "expected_lmtd_c": 138.26,  # Calculated LMTD
            "expected_heat_duty_kw": 3780.0,  # Q = m * Cp * dT (water side)
            "tolerance": 0.02  # 2% tolerance
        },
        {
            "name": "ASME PTC 4.3 Example 2 - Parallel Flow",
            "flow_arrangement": FlowArrangement.PARALLEL_FLOW,
            "gas_inlet_c": 350.0,
            "gas_outlet_c": 180.0,
            "water_inlet_c": 100.0,
            "water_outlet_c": 150.0,
            "gas_flow_kg_s": 18.0,
            "water_flow_kg_s": 12.0,
            "gas_cp_kj_kg_k": 1.08,
            "water_cp_kj_kg_k": 4.19,
            "expected_lmtd_c": 102.85,
            "expected_heat_duty_kw": 2514.0,
            "tolerance": 0.02
        },
        {
            "name": "ASME PTC 4.3 Example 3 - High Temperature",
            "flow_arrangement": FlowArrangement.COUNTER_FLOW,
            "gas_inlet_c": 500.0,
            "gas_outlet_c": 220.0,
            "water_inlet_c": 150.0,
            "water_outlet_c": 200.0,
            "gas_flow_kg_s": 25.0,
            "water_flow_kg_s": 20.0,
            "gas_cp_kj_kg_k": 1.12,
            "water_cp_kj_kg_k": 4.22,
            "expected_lmtd_c": 174.25,
            "expected_heat_duty_kw": 4220.0,
            "tolerance": 0.02
        }
    ]


@pytest.fixture
def lmtd_test_cases():
    """Parametrized test cases for LMTD calculation."""
    return [
        # Counter flow cases
        {
            "flow": FlowArrangement.COUNTER_FLOW,
            "T_hot_in": 400.0, "T_hot_out": 200.0,
            "T_cold_in": 100.0, "T_cold_out": 150.0,
            "expected_lmtd": 163.93,
            "description": "Counter flow - standard case"
        },
        {
            "flow": FlowArrangement.COUNTER_FLOW,
            "T_hot_in": 300.0, "T_hot_out": 200.0,
            "T_cold_in": 150.0, "T_cold_out": 200.0,
            "expected_lmtd": 68.79,
            "description": "Counter flow - equal outlet temps"
        },
        # Parallel flow cases
        {
            "flow": FlowArrangement.PARALLEL_FLOW,
            "T_hot_in": 350.0, "T_hot_out": 180.0,
            "T_cold_in": 100.0, "T_cold_out": 140.0,
            "expected_lmtd": 129.63,
            "description": "Parallel flow - standard case"
        },
        {
            "flow": FlowArrangement.PARALLEL_FLOW,
            "T_hot_in": 300.0, "T_hot_out": 150.0,
            "T_cold_in": 50.0, "T_cold_out": 120.0,
            "expected_lmtd": 129.51,
            "description": "Parallel flow - large temp difference"
        },
    ]


# =============================================================================
# THERMAL PROPERTY TEST CASES
# =============================================================================

@pytest.fixture
def water_cp_test_cases():
    """Test cases for water specific heat capacity vs IAPWS-IF97."""
    return [
        # (temperature_c, pressure_kpa, expected_cp_kj_kg_k, tolerance)
        (20.0, 101.325, 4.182, 0.001),
        (50.0, 101.325, 4.181, 0.001),
        (80.0, 101.325, 4.195, 0.001),
        (100.0, 200.0, 4.216, 0.002),
        (120.0, 300.0, 4.250, 0.002),
        (150.0, 500.0, 4.310, 0.002),
        (180.0, 1000.0, 4.410, 0.003),
        (200.0, 1500.0, 4.497, 0.003),
    ]


@pytest.fixture
def flue_gas_cp_test_cases():
    """Test cases for flue gas specific heat capacity."""
    return [
        # (temperature_c, composition, expected_cp_kj_kg_k, tolerance)
        (200.0, {"CO2": 0.12, "H2O": 0.08, "N2": 0.74, "O2": 0.06}, 1.05, 0.02),
        (300.0, {"CO2": 0.12, "H2O": 0.08, "N2": 0.74, "O2": 0.06}, 1.08, 0.02),
        (400.0, {"CO2": 0.12, "H2O": 0.08, "N2": 0.74, "O2": 0.06}, 1.11, 0.02),
        (500.0, {"CO2": 0.12, "H2O": 0.08, "N2": 0.74, "O2": 0.06}, 1.14, 0.02),
        (200.0, {"CO2": 0.15, "H2O": 0.10, "N2": 0.70, "O2": 0.05}, 1.06, 0.02),
        (400.0, {"CO2": 0.15, "H2O": 0.10, "N2": 0.70, "O2": 0.05}, 1.12, 0.02),
    ]


# =============================================================================
# PARAMETERIZED TEST DATA
# =============================================================================

@pytest.fixture
def fouling_factor_test_cases():
    """Test cases for fouling factor calculation."""
    return [
        # (clean_u_value, fouled_u_value, expected_fouling_factor, tolerance)
        (45.0, 45.0, 0.0, 0.0001),  # No fouling
        (45.0, 40.0, 0.00278, 0.0001),  # Light fouling
        (45.0, 35.0, 0.00635, 0.0001),  # Moderate fouling
        (45.0, 30.0, 0.01111, 0.0001),  # Heavy fouling
        (45.0, 25.0, 0.01778, 0.0001),  # Severe fouling
        (50.0, 45.0, 0.00222, 0.0001),
        (55.0, 40.0, 0.00682, 0.0001),
    ]


@pytest.fixture
def effectiveness_test_cases():
    """Test cases for effectiveness calculation."""
    return [
        # Counter flow: effectiveness = (T_c_out - T_c_in) / (T_h_in - T_c_in)
        {
            "flow": FlowArrangement.COUNTER_FLOW,
            "T_hot_in": 400.0, "T_cold_in": 100.0,
            "T_cold_out": 175.0,
            "expected_effectiveness": 0.25,
            "description": "Counter flow - 25% effectiveness"
        },
        {
            "flow": FlowArrangement.COUNTER_FLOW,
            "T_hot_in": 380.0, "T_cold_in": 105.0,
            "T_cold_out": 145.0,
            "expected_effectiveness": 0.145,
            "description": "Counter flow - design conditions"
        },
        {
            "flow": FlowArrangement.COUNTER_FLOW,
            "T_hot_in": 350.0, "T_cold_in": 100.0,
            "T_cold_out": 200.0,
            "expected_effectiveness": 0.40,
            "description": "Counter flow - 40% effectiveness"
        },
    ]


@pytest.fixture
def heat_duty_test_cases():
    """Test cases for heat duty calculation."""
    return [
        # Q = m * Cp * dT (kW = kg/s * kJ/kg.K * K)
        {
            "mass_flow_kg_s": 10.0,
            "cp_kj_kg_k": 4.2,
            "delta_t_c": 40.0,
            "expected_q_kw": 1680.0,
            "description": "Standard water heating"
        },
        {
            "mass_flow_kg_s": 15.0,
            "cp_kj_kg_k": 1.1,
            "delta_t_c": 200.0,
            "expected_q_kw": 3300.0,
            "description": "Flue gas cooling"
        },
        {
            "mass_flow_kg_s": 12.5,
            "cp_kj_kg_k": 4.2,
            "delta_t_c": 45.0,
            "expected_q_kw": 2362.5,
            "description": "Design conditions"
        },
    ]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

@pytest.fixture
def provenance_validator():
    """Helper function to validate provenance records."""
    def validate(provenance):
        """Validate provenance record structure and hashes."""
        assert provenance is not None
        assert hasattr(provenance, 'calculator_name') or 'calculator_name' in provenance
        assert hasattr(provenance, 'provenance_hash') or 'provenance_hash' in provenance

        hash_val = getattr(provenance, 'provenance_hash', None) or provenance.get('provenance_hash')
        assert len(hash_val) == 64  # SHA-256

        return True

    return validate


@pytest.fixture
def tolerance_checker():
    """Helper function for floating point comparisons."""
    def check(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
        """Check if actual is within tolerance of expected."""
        return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)

    return check


@pytest.fixture
def performance_result_validator():
    """Helper function to validate heat transfer results."""
    def validate(result: HeatTransferResult, economizer: EconomizerConfig):
        """Validate heat transfer calculation result."""
        errors = []

        # Validate heat duty is positive and reasonable
        if result.heat_duty_kw <= 0:
            errors.append("Heat duty must be positive")
        if result.heat_duty_kw > economizer.design_heat_duty_kw * 1.5:
            errors.append("Heat duty exceeds 150% of design")

        # Validate LMTD is positive
        if result.lmtd_c <= 0:
            errors.append("LMTD must be positive")

        # Validate U-value is positive and reasonable
        if result.u_value_w_m2k <= 0:
            errors.append("U-value must be positive")
        if result.u_value_w_m2k > economizer.design_u_value_w_m2k * 2.0:
            errors.append("U-value exceeds 200% of design")

        # Validate effectiveness is between 0 and 1
        if not 0 <= result.effectiveness <= 1:
            errors.append("Effectiveness must be between 0 and 1")

        # Validate approach temperature is positive
        if result.approach_temp_c < 0:
            errors.append("Approach temperature must be non-negative")

        # Validate provenance hash
        if len(result.provenance_hash) != 64:
            errors.append("Invalid provenance hash length")

        return len(errors) == 0, errors

    return validate


@pytest.fixture
def alert_validator():
    """Helper function to validate alerts."""
    def validate(alert: Alert, config: AlertConfig):
        """Validate alert against its configuration."""
        errors = []

        if alert.alert_type != config.alert_type:
            errors.append("Alert type mismatch")

        if alert.parameter != config.parameter:
            errors.append("Parameter mismatch")

        if config.threshold_high and alert.current_value < config.threshold_high:
            errors.append("Alert generated below high threshold")

        if config.threshold_low and alert.current_value > config.threshold_low:
            errors.append("Alert generated above low threshold")

        return len(errors) == 0, errors

    return validate


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

@pytest.fixture
def sensor_reading_generator():
    """Generate mock sensor readings for testing."""
    def generate(
        num_readings: int = 100,
        base_time: datetime = None,
        interval_minutes: int = 5,
        economizer_id: str = "ECON-001",
        add_noise: bool = True,
        seed: int = 42
    ):
        """Generate time series of sensor readings."""
        random.seed(seed)
        base_time = base_time or datetime.now(timezone.utc)

        readings = []
        for i in range(num_readings):
            timestamp = base_time + timedelta(minutes=i * interval_minutes)

            # Base temperatures with optional noise
            gas_in_base = 380.0
            gas_out_base = 175.0
            water_in_base = 105.0
            water_out_base = 145.0

            if add_noise:
                gas_in = gas_in_base + random.gauss(0, 2)
                gas_out = gas_out_base + random.gauss(0, 1.5)
                water_in = water_in_base + random.gauss(0, 0.5)
                water_out = water_out_base + random.gauss(0, 1)
            else:
                gas_in = gas_in_base
                gas_out = gas_out_base
                water_in = water_in_base
                water_out = water_out_base

            readings.append({
                "timestamp": timestamp,
                "economizer_id": economizer_id,
                "temperatures": {
                    "gas_inlet": TemperatureReading(
                        sensor_id="TT-GAS-IN-001",
                        location="gas_inlet",
                        timestamp=timestamp,
                        value_c=gas_in,
                        quality=random.uniform(0.95, 1.0)
                    ),
                    "gas_outlet": TemperatureReading(
                        sensor_id="TT-GAS-OUT-001",
                        location="gas_outlet",
                        timestamp=timestamp,
                        value_c=gas_out,
                        quality=random.uniform(0.95, 1.0)
                    ),
                    "water_inlet": TemperatureReading(
                        sensor_id="TT-WATER-IN-001",
                        location="water_inlet",
                        timestamp=timestamp,
                        value_c=water_in,
                        quality=random.uniform(0.95, 1.0)
                    ),
                    "water_outlet": TemperatureReading(
                        sensor_id="TT-WATER-OUT-001",
                        location="water_outlet",
                        timestamp=timestamp,
                        value_c=water_out,
                        quality=random.uniform(0.95, 1.0)
                    )
                },
                "flows": {
                    "water": FlowReading(
                        sensor_id="FT-WATER-001",
                        fluid_type="water",
                        timestamp=timestamp,
                        value=12.5 + (random.gauss(0, 0.3) if add_noise else 0),
                        unit="kg/s",
                        quality=random.uniform(0.95, 1.0)
                    ),
                    "flue_gas": FlowReading(
                        sensor_id="FT-GAS-001",
                        fluid_type="flue_gas",
                        timestamp=timestamp,
                        value=18.0 + (random.gauss(0, 0.5) if add_noise else 0),
                        unit="kg/s",
                        quality=random.uniform(0.95, 1.0)
                    )
                }
            })

        return readings

    return generate


@pytest.fixture
def fouling_progression_generator():
    """Generate progressive fouling data over time."""
    def generate(
        num_days: int = 180,
        initial_fouling: float = 0.0001,
        fouling_rate_per_day: float = 0.000005,
        seed: int = 42
    ):
        """Generate fouling progression data."""
        random.seed(seed)
        base_time = datetime.now(timezone.utc) - timedelta(days=num_days)

        data = []
        for day in range(num_days):
            timestamp = base_time + timedelta(days=day)

            # Progressive fouling with some random variation
            fouling_factor = initial_fouling + day * fouling_rate_per_day
            fouling_factor *= (1 + random.gauss(0, 0.05))  # 5% daily variation

            # Determine fouling level
            if fouling_factor < 0.0002:
                level = FoulingLevel.CLEAN
            elif fouling_factor < 0.0004:
                level = FoulingLevel.LIGHT
            elif fouling_factor < 0.0007:
                level = FoulingLevel.MODERATE
            elif fouling_factor < 0.001:
                level = FoulingLevel.HEAVY
            else:
                level = FoulingLevel.SEVERE

            efficiency_loss = (fouling_factor / 0.001) * 20.0  # Linear approximation

            data.append(FoulingData(
                economizer_id="ECON-001",
                timestamp=timestamp,
                fouling_factor_m2k_w=fouling_factor,
                fouling_level=level,
                efficiency_loss_pct=efficiency_loss,
                estimated_fuel_penalty_pct=efficiency_loss * 0.2,
                days_since_cleaning=day,
                cleaning_recommended=fouling_factor > 0.0008,
                estimated_days_to_cleaning=max(0, int((0.001 - fouling_factor) / fouling_rate_per_day))
            ))

        return data

    return generate


# =============================================================================
# TEST DATA FILES
# =============================================================================

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_readings_data(test_data_dir):
    """Load sample readings test data."""
    readings_file = test_data_dir / "sample_readings.json"

    if readings_file.exists():
        return json.loads(readings_file.read_text())

    # Return default data if file doesn't exist
    return {
        "readings": [],
        "metadata": {}
    }


@pytest.fixture
def sample_economizer_configs(test_data_dir):
    """Load sample economizer configurations."""
    config_file = test_data_dir / "sample_economizer_config.json"

    if config_file.exists():
        return json.loads(config_file.read_text())

    # Return default data if file doesn't exist
    return {
        "economizers": [],
        "baselines": []
    }


# =============================================================================
# BENCHMARK FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_sensor_data():
    """Large dataset for performance benchmarking."""
    base_time = datetime.now(timezone.utc)
    return [
        {
            "timestamp": base_time + timedelta(minutes=i),
            "gas_inlet_c": 380.0 + (i % 20) - 10,
            "gas_outlet_c": 175.0 + (i % 15) - 7,
            "water_inlet_c": 105.0 + (i % 5) - 2,
            "water_outlet_c": 145.0 + (i % 10) - 5,
            "water_flow_kg_s": 12.5 + (i % 10) * 0.1,
            "gas_flow_kg_s": 18.0 + (i % 10) * 0.15
        }
        for i in range(10000)
    ]


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "critical: mark test as critical (must pass)"
    )
    config.addinivalue_line(
        "markers", "calculator: mark test as calculator test (95%+ coverage)"
    )
    config.addinivalue_line(
        "markers", "heat_transfer: mark test as heat transfer calculation test"
    )
    config.addinivalue_line(
        "markers", "fouling: mark test as fouling analysis test"
    )
    config.addinivalue_line(
        "markers", "thermal: mark test as thermal property test"
    )
    config.addinivalue_line(
        "markers", "alerts: mark test as alert management test"
    )
    config.addinivalue_line(
        "markers", "asme: mark test as ASME PTC 4.3 validation test"
    )
    config.addinivalue_line(
        "markers", "iapws: mark test as IAPWS-IF97 validation test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add calculator marker to calculator tests
        if "calculator" in item.nodeid:
            item.add_marker(pytest.mark.calculator)

        # Add heat_transfer marker to heat transfer tests
        if "heat_transfer" in item.nodeid or "lmtd" in item.nodeid:
            item.add_marker(pytest.mark.heat_transfer)

        # Add fouling marker to fouling tests
        if "fouling" in item.nodeid:
            item.add_marker(pytest.mark.fouling)

        # Add thermal marker to thermal property tests
        if "thermal" in item.nodeid or "cp" in item.nodeid:
            item.add_marker(pytest.mark.thermal)

        # Add alerts marker to alert tests
        if "alert" in item.nodeid:
            item.add_marker(pytest.mark.alerts)

        # Add critical marker to validation tests
        if "validation" in item.nodeid or "provenance" in item.nodeid or "asme" in item.nodeid:
            item.add_marker(pytest.mark.critical)
