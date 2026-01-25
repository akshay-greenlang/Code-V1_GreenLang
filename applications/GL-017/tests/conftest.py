# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for GL-017 CONDENSYNC test suite.

Provides common fixtures, markers, and test configuration for all test modules.
Includes fixtures for condenser data, mock SCADA connections, cooling water
configurations, vacuum system test data, and performance baselines.
"""

import pytest
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass, field

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress debug logs from dependencies
logging.getLogger('asyncio').setLevel(logging.WARNING)


# ============================================================================
# Session Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Condenser Data Fixtures
# ============================================================================

@pytest.fixture
def sample_condenser_data():
    """Standard condenser operating data for testing."""
    return {
        'condenser_id': 'COND-001',
        'condenser_name': 'Main Turbine Condenser',
        'timestamp': datetime.utcnow().isoformat(),
        'vacuum_pressure_mbar': 50.0,
        'hotwell_temperature_c': 33.2,
        'steam_inlet_temperature_c': 35.5,
        'cooling_water_inlet_temp_c': 25.0,
        'cooling_water_outlet_temp_c': 32.0,
        'cooling_water_flow_rate_m3_hr': 45000.0,
        'condensate_flow_rate_kg_hr': 150000.0,
        'tube_inlet_pressure_bar': 2.5,
        'tube_outlet_pressure_bar': 1.8,
        'air_inleakage_rate_kg_hr': 0.5,
        'ttd_c': 2.3,  # Terminal Temperature Difference
        'subcooling_c': 0.5,
        'heat_duty_mw': 180.0,
        'cleanliness_factor': 0.85,
        'overall_heat_transfer_coeff_w_m2k': 2800.0,
        'tube_fouling_factor': 0.00015,
        'num_passes': 2,
        'tube_material': 'titanium',
        'tube_count': 18500,
        'tube_od_mm': 25.4,
        'tube_length_m': 12.0,
        'surface_area_m2': 17500.0
    }


@pytest.fixture
def high_backpressure_condenser():
    """Condenser data with high backpressure condition."""
    return {
        'condenser_id': 'COND-002',
        'condenser_name': 'High Backpressure Test',
        'timestamp': datetime.utcnow().isoformat(),
        'vacuum_pressure_mbar': 85.0,  # High backpressure
        'hotwell_temperature_c': 42.5,
        'steam_inlet_temperature_c': 45.0,
        'cooling_water_inlet_temp_c': 28.0,
        'cooling_water_outlet_temp_c': 38.0,
        'cooling_water_flow_rate_m3_hr': 40000.0,  # Reduced flow
        'condensate_flow_rate_kg_hr': 145000.0,
        'tube_inlet_pressure_bar': 2.2,
        'tube_outlet_pressure_bar': 1.5,
        'air_inleakage_rate_kg_hr': 2.5,  # High air inleakage
        'ttd_c': 4.5,  # High TTD indicates fouling
        'subcooling_c': 1.5,
        'heat_duty_mw': 175.0,
        'cleanliness_factor': 0.65,  # Low cleanliness
        'overall_heat_transfer_coeff_w_m2k': 2200.0,  # Degraded
        'tube_fouling_factor': 0.00035,  # High fouling
        'num_passes': 2,
        'tube_material': 'admiralty_brass',
        'tube_count': 18500,
        'tube_od_mm': 25.4,
        'tube_length_m': 12.0,
        'surface_area_m2': 17500.0
    }


@pytest.fixture
def air_inleakage_condenser():
    """Condenser data with significant air inleakage."""
    return {
        'condenser_id': 'COND-003',
        'condenser_name': 'Air Inleakage Test',
        'timestamp': datetime.utcnow().isoformat(),
        'vacuum_pressure_mbar': 70.0,
        'hotwell_temperature_c': 39.0,
        'steam_inlet_temperature_c': 41.5,
        'cooling_water_inlet_temp_c': 26.0,
        'cooling_water_outlet_temp_c': 33.5,
        'cooling_water_flow_rate_m3_hr': 43000.0,
        'condensate_flow_rate_kg_hr': 148000.0,
        'tube_inlet_pressure_bar': 2.4,
        'tube_outlet_pressure_bar': 1.7,
        'air_inleakage_rate_kg_hr': 8.0,  # Very high air inleakage
        'ttd_c': 3.5,
        'subcooling_c': 2.5,  # High subcooling indicates air
        'heat_duty_mw': 178.0,
        'cleanliness_factor': 0.80,
        'overall_heat_transfer_coeff_w_m2k': 2600.0,
        'tube_fouling_factor': 0.00018,
        'num_passes': 2,
        'tube_material': 'stainless_316',
        'tube_count': 18500,
        'tube_od_mm': 25.4,
        'tube_length_m': 12.0,
        'surface_area_m2': 17500.0
    }


@pytest.fixture
def fouled_condenser():
    """Condenser data with severe tube fouling."""
    return {
        'condenser_id': 'COND-004',
        'condenser_name': 'Fouled Condenser Test',
        'timestamp': datetime.utcnow().isoformat(),
        'vacuum_pressure_mbar': 75.0,
        'hotwell_temperature_c': 40.5,
        'steam_inlet_temperature_c': 43.0,
        'cooling_water_inlet_temp_c': 25.5,
        'cooling_water_outlet_temp_c': 35.5,
        'cooling_water_flow_rate_m3_hr': 44000.0,
        'condensate_flow_rate_kg_hr': 147000.0,
        'tube_inlet_pressure_bar': 2.8,  # High pressure drop
        'tube_outlet_pressure_bar': 1.4,
        'air_inleakage_rate_kg_hr': 1.0,
        'ttd_c': 5.0,  # Very high TTD indicates severe fouling
        'subcooling_c': 0.8,
        'heat_duty_mw': 176.0,
        'cleanliness_factor': 0.55,  # Very low cleanliness
        'overall_heat_transfer_coeff_w_m2k': 1800.0,  # Severely degraded
        'tube_fouling_factor': 0.00055,  # Very high fouling
        'num_passes': 2,
        'tube_material': 'copper_nickel_90_10',
        'tube_count': 18500,
        'tube_od_mm': 25.4,
        'tube_length_m': 12.0,
        'surface_area_m2': 17500.0
    }


# ============================================================================
# Cooling Water Configuration Fixtures
# ============================================================================

@pytest.fixture
def standard_cooling_water_config():
    """Standard cooling water system configuration."""
    return {
        'cooling_source': 'cooling_tower',
        'design_flow_rate_m3_hr': 48000.0,
        'min_flow_rate_m3_hr': 30000.0,
        'max_flow_rate_m3_hr': 55000.0,
        'design_inlet_temp_c': 24.0,
        'max_inlet_temp_c': 32.0,
        'design_outlet_temp_c': 32.0,
        'max_outlet_temp_c': 40.0,
        'pump_count': 3,
        'pump_efficiency': 0.82,
        'pipe_diameter_mm': 2000.0,
        'pipe_length_m': 150.0,
        'pipe_material': 'concrete_lined_steel',
        'chlorination_enabled': True,
        'biocide_treatment': 'oxidizing',
        'cycles_of_concentration': 4.0,
        'makeup_water_source': 'river',
        'makeup_water_tds_ppm': 450.0
    }


@pytest.fixture
def once_through_cooling_config():
    """Once-through cooling water system configuration."""
    return {
        'cooling_source': 'once_through_river',
        'design_flow_rate_m3_hr': 85000.0,
        'min_flow_rate_m3_hr': 60000.0,
        'max_flow_rate_m3_hr': 100000.0,
        'design_inlet_temp_c': 18.0,
        'max_inlet_temp_c': 28.0,
        'design_outlet_temp_c': 26.0,
        'max_outlet_temp_c': 35.0,
        'pump_count': 4,
        'pump_efficiency': 0.85,
        'pipe_diameter_mm': 2500.0,
        'pipe_length_m': 500.0,
        'pipe_material': 'concrete',
        'chlorination_enabled': True,
        'biocide_treatment': 'chlorine_dioxide',
        'cycles_of_concentration': 1.0,
        'makeup_water_source': 'river',
        'makeup_water_tds_ppm': 280.0,
        'discharge_temperature_limit_c': 30.0,
        'environmental_permit_id': 'ENV-2024-001'
    }


@pytest.fixture
def seawater_cooling_config():
    """Seawater cooling system configuration."""
    return {
        'cooling_source': 'seawater',
        'design_flow_rate_m3_hr': 95000.0,
        'min_flow_rate_m3_hr': 70000.0,
        'max_flow_rate_m3_hr': 110000.0,
        'design_inlet_temp_c': 20.0,
        'max_inlet_temp_c': 30.0,
        'design_outlet_temp_c': 28.0,
        'max_outlet_temp_c': 38.0,
        'pump_count': 4,
        'pump_efficiency': 0.80,
        'pipe_diameter_mm': 2800.0,
        'pipe_length_m': 800.0,
        'pipe_material': 'fiberglass_reinforced_plastic',
        'chlorination_enabled': True,
        'biocide_treatment': 'electro_chlorination',
        'cycles_of_concentration': 1.0,
        'makeup_water_source': 'seawater',
        'makeup_water_tds_ppm': 35000.0,
        'tube_material_compatibility': 'titanium_required',
        'cathodic_protection_enabled': True
    }


# ============================================================================
# Vacuum System Test Data Fixtures
# ============================================================================

@pytest.fixture
def vacuum_system_data():
    """Standard vacuum system operating data."""
    return {
        'vacuum_pump_type': 'liquid_ring',
        'vacuum_pump_count': 2,
        'vacuum_pump_capacity_kg_hr': 15.0,
        'seal_water_temperature_c': 30.0,
        'seal_water_flow_rate_m3_hr': 25.0,
        'ejector_steam_pressure_bar': 8.0,
        'ejector_steam_flow_kg_hr': 450.0,
        'first_stage_ejector_active': True,
        'second_stage_ejector_active': True,
        'hogging_ejector_available': True,
        'air_extraction_rate_kg_hr': 1.2,
        'non_condensable_gas_composition': {
            'air': 0.85,
            'ammonia': 0.05,
            'co2': 0.08,
            'other': 0.02
        },
        'inter_condenser_pressure_mbar': 120.0,
        'after_condenser_pressure_mbar': 200.0,
        'air_cooler_outlet_temp_c': 35.0
    }


@pytest.fixture
def degraded_vacuum_system():
    """Vacuum system with degraded performance."""
    return {
        'vacuum_pump_type': 'liquid_ring',
        'vacuum_pump_count': 2,
        'vacuum_pump_capacity_kg_hr': 12.0,  # Reduced capacity
        'seal_water_temperature_c': 38.0,  # Elevated seal water temp
        'seal_water_flow_rate_m3_hr': 20.0,
        'ejector_steam_pressure_bar': 7.2,  # Low steam pressure
        'ejector_steam_flow_kg_hr': 520.0,  # Higher steam consumption
        'first_stage_ejector_active': True,
        'second_stage_ejector_active': True,
        'hogging_ejector_available': False,  # Unavailable
        'air_extraction_rate_kg_hr': 0.8,  # Reduced extraction
        'non_condensable_gas_composition': {
            'air': 0.90,
            'ammonia': 0.02,
            'co2': 0.06,
            'other': 0.02
        },
        'inter_condenser_pressure_mbar': 150.0,  # Elevated
        'after_condenser_pressure_mbar': 250.0,  # Elevated
        'air_cooler_outlet_temp_c': 42.0  # High temperature
    }


# ============================================================================
# Performance Baseline Fixtures
# ============================================================================

@pytest.fixture
def performance_baseline():
    """Performance baseline for condenser optimization."""
    return {
        'design_vacuum_mbar': 45.0,
        'design_ttd_c': 2.0,
        'design_subcooling_c': 0.3,
        'design_heat_transfer_coeff_w_m2k': 3200.0,
        'design_cleanliness_factor': 0.95,
        'design_air_inleakage_kg_hr': 0.3,
        'design_cooling_water_delta_t_c': 8.0,
        'guaranteed_backpressure_mbar': 50.0,
        'alarm_backpressure_mbar': 75.0,
        'trip_backpressure_mbar': 100.0,
        'heat_rate_penalty_per_mbar_kj_kwh': 12.5,
        'design_lmtd_c': 10.5,
        'design_ntu': 1.8,
        'tube_velocity_design_m_s': 2.2,
        'tube_velocity_min_m_s': 1.5,
        'tube_velocity_max_m_s': 2.8
    }


@pytest.fixture
def historical_performance_data():
    """Historical performance data for trend analysis."""
    base_time = datetime.utcnow()
    data = []

    for i in range(168):  # 7 days of hourly data
        hour_offset = timedelta(hours=i)
        timestamp = base_time - hour_offset

        # Simulate gradual fouling over time
        fouling_factor = 0.00015 + (i * 0.000001)
        cleanliness = 0.90 - (i * 0.0002)
        vacuum = 48.0 + (i * 0.01)

        data.append({
            'timestamp': timestamp.isoformat(),
            'vacuum_pressure_mbar': vacuum,
            'cleanliness_factor': max(cleanliness, 0.60),
            'tube_fouling_factor': min(fouling_factor, 0.0005),
            'cooling_water_inlet_temp_c': 25.0 + (5 * (i % 24) / 24),
            'cooling_water_flow_rate_m3_hr': 45000.0,
            'heat_duty_mw': 175.0 + (10 * (i % 24) / 24),
            'ttd_c': 2.3 + (i * 0.005),
            'air_inleakage_rate_kg_hr': 0.5 + (i * 0.002)
        })

    return data


# ============================================================================
# Mock SCADA Server Fixtures
# ============================================================================

@pytest.fixture
def mock_scada_server():
    """Mock SCADA server for testing."""
    server = AsyncMock()
    server.connect = AsyncMock(return_value=True)
    server.disconnect = AsyncMock(return_value=True)
    server.read_tag = AsyncMock()
    server.write_tag = AsyncMock(return_value=True)
    server.read_multiple_tags = AsyncMock()
    server.subscribe_to_tag = AsyncMock(return_value=True)
    server.is_connected = Mock(return_value=True)
    return server


@pytest.fixture
def mock_scada_condenser_tags():
    """Mock SCADA tag data for condenser."""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'tags': {
            'COND_VACUUM': {'value': 50.0, 'quality': 'GOOD', 'unit': 'mbar'},
            'COND_HOTWELL_TEMP': {'value': 33.2, 'quality': 'GOOD', 'unit': 'C'},
            'CW_INLET_TEMP': {'value': 25.0, 'quality': 'GOOD', 'unit': 'C'},
            'CW_OUTLET_TEMP': {'value': 32.0, 'quality': 'GOOD', 'unit': 'C'},
            'CW_FLOW_RATE': {'value': 45000.0, 'quality': 'GOOD', 'unit': 'm3/hr'},
            'CONDENSATE_FLOW': {'value': 150000.0, 'quality': 'GOOD', 'unit': 'kg/hr'},
            'AIR_EXTRACTION': {'value': 0.5, 'quality': 'GOOD', 'unit': 'kg/hr'},
            'TUBE_DP': {'value': 0.7, 'quality': 'GOOD', 'unit': 'bar'},
            'STEAM_INLET_TEMP': {'value': 35.5, 'quality': 'GOOD', 'unit': 'C'},
            'SUBCOOLING': {'value': 0.5, 'quality': 'GOOD', 'unit': 'C'}
        }
    }


# ============================================================================
# Mock Cooling Tower Fixtures
# ============================================================================

@pytest.fixture
def mock_cooling_tower():
    """Mock cooling tower system."""
    tower = AsyncMock()
    tower.connect = AsyncMock(return_value=True)
    tower.disconnect = AsyncMock(return_value=True)
    tower.get_status = AsyncMock()
    tower.set_fan_speed = AsyncMock(return_value=True)
    tower.get_approach_temperature = AsyncMock(return_value=5.5)
    tower.get_range_temperature = AsyncMock(return_value=8.0)
    tower.is_connected = Mock(return_value=True)
    return tower


@pytest.fixture
def mock_cooling_tower_data():
    """Mock cooling tower operating data."""
    return {
        'tower_id': 'CT-001',
        'timestamp': datetime.utcnow().isoformat(),
        'wet_bulb_temp_c': 20.0,
        'dry_bulb_temp_c': 28.0,
        'relative_humidity_percent': 65.0,
        'cold_water_temp_c': 25.0,
        'hot_water_temp_c': 33.0,
        'approach_temp_c': 5.0,
        'range_temp_c': 8.0,
        'fan_speed_percent': [75.0, 75.0, 80.0, 80.0],
        'total_fan_power_kw': 450.0,
        'makeup_water_flow_m3_hr': 1200.0,
        'blowdown_flow_m3_hr': 300.0,
        'cycles_of_concentration': 4.0,
        'basin_level_percent': 85.0
    }


# ============================================================================
# Mock DCS Integration Fixtures
# ============================================================================

@pytest.fixture
def mock_dcs_system():
    """Mock DCS system for integration tests."""
    dcs = AsyncMock()
    dcs.connect = AsyncMock(return_value=True)
    dcs.disconnect = AsyncMock(return_value=True)
    dcs.read_point = AsyncMock()
    dcs.write_point = AsyncMock(return_value=True)
    dcs.read_block = AsyncMock()
    dcs.is_connected = Mock(return_value=True)
    return dcs


@pytest.fixture
def mock_historian():
    """Mock historian for trend data retrieval."""
    historian = AsyncMock()
    historian.connect = AsyncMock(return_value=True)
    historian.disconnect = AsyncMock(return_value=True)
    historian.query_tag_history = AsyncMock()
    historian.write_tag_value = AsyncMock(return_value=True)
    historian.is_connected = Mock(return_value=True)
    return historian


# ============================================================================
# Agent Configuration Fixtures
# ============================================================================

@pytest.fixture
def agent_config():
    """Standard agent configuration."""
    return {
        'agent_id': 'GL-017-TEST',
        'agent_name': 'TestCondenSyncAgent',
        'version': '1.0.0-test',
        'llm_temperature': 0.0,
        'llm_seed': 42,
        'llm_model': 'claude-sonnet-4',
        'max_retries': 3,
        'timeout_seconds': 30,
        'enable_caching': True,
        'cache_ttl_seconds': 300,
        'enable_provenance': True,
        'enable_audit_logging': True,
        'log_level': 'DEBUG'
    }


@pytest.fixture
def condenser_config():
    """Condenser system configuration."""
    return {
        'condenser_id': 'COND-TEST-001',
        'condenser_name': 'Test Steam Condenser',
        'condenser_type': 'shell_and_tube',
        'design_pressure_mbar': 45.0,
        'alarm_pressure_mbar': 75.0,
        'trip_pressure_mbar': 100.0,
        'design_heat_duty_mw': 200.0,
        'design_cooling_water_flow_m3_hr': 48000.0,
        'design_heat_transfer_coeff_w_m2k': 3200.0,
        'tube_material': 'titanium',
        'tube_count': 18500,
        'tube_od_mm': 25.4,
        'tube_id_mm': 23.4,
        'tube_length_m': 12.0,
        'num_passes': 2,
        'surface_area_m2': 17500.0,
        'scada_host': 'localhost',
        'scada_port': 4840
    }


# ============================================================================
# Tool Testing Fixtures
# ============================================================================

@pytest.fixture
def vacuum_optimization_input():
    """Input data for vacuum optimization testing."""
    return {
        'current_vacuum_mbar': 55.0,
        'target_vacuum_mbar': 48.0,
        'cooling_water_inlet_temp_c': 26.0,
        'cooling_water_flow_rate_m3_hr': 44000.0,
        'air_inleakage_rate_kg_hr': 1.2,
        'cleanliness_factor': 0.82,
        'steam_load_percent': 85.0
    }


@pytest.fixture
def heat_transfer_input():
    """Input data for heat transfer calculation testing."""
    return {
        'heat_duty_mw': 180.0,
        'lmtd_c': 10.5,
        'surface_area_m2': 17500.0,
        'tube_material': 'titanium',
        'tube_od_mm': 25.4,
        'tube_id_mm': 23.4,
        'cooling_water_velocity_m_s': 2.2,
        'fouling_factor_m2k_w': 0.00015
    }


@pytest.fixture
def fouling_prediction_input():
    """Input data for fouling prediction testing."""
    return {
        'cooling_water_source': 'cooling_tower',
        'cooling_water_tds_ppm': 1800.0,
        'cooling_water_ph': 7.8,
        'cooling_water_temperature_c': 28.0,
        'tube_velocity_m_s': 2.2,
        'tube_material': 'titanium',
        'operating_hours': 4380.0,
        'biocide_treatment': 'oxidizing',
        'current_cleanliness_factor': 0.85
    }


# ============================================================================
# Pytest Markers Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "determinism: mark test as determinism test"
    )
    config.addinivalue_line(
        "markers", "compliance: mark test as compliance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "scada: mark test as SCADA integration test"
    )
    config.addinivalue_line(
        "markers", "cooling_tower: mark test as cooling tower integration"
    )


# ============================================================================
# Efficiency Calculator Fixtures
# ============================================================================

@pytest.fixture
def efficiency_input_standard():
    """Standard efficiency calculation input."""
    from calculators.efficiency_calculator import EfficiencyInput
    return EfficiencyInput(
        steam_temp_c=40.0,
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.0,
        cw_flow_rate_m3_hr=50000.0,
        heat_duty_mw=200.0,
        turbine_output_mw=300.0,
        design_backpressure_mmhg=50.8,
        actual_backpressure_mmhg=55.0,
        design_u_value_w_m2k=3500.0,
        actual_u_value_w_m2k=3000.0,
        heat_transfer_area_m2=17500.0,
        electricity_cost_usd_mwh=50.0,
        operating_hours_per_year=8000,
    )


@pytest.fixture
def efficiency_input_degraded():
    """Degraded condenser efficiency input."""
    from calculators.efficiency_calculator import EfficiencyInput
    return EfficiencyInput(
        steam_temp_c=48.0,
        cw_inlet_temp_c=28.0,
        cw_outlet_temp_c=40.0,
        cw_flow_rate_m3_hr=45000.0,
        heat_duty_mw=180.0,
        turbine_output_mw=270.0,
        design_backpressure_mmhg=50.8,
        actual_backpressure_mmhg=80.0,
        design_u_value_w_m2k=3500.0,
        actual_u_value_w_m2k=2000.0,
        heat_transfer_area_m2=17500.0,
        electricity_cost_usd_mwh=60.0,
        operating_hours_per_year=8000,
    )


# ============================================================================
# Heat Transfer Calculator Fixtures
# ============================================================================

@pytest.fixture
def heat_transfer_input_standard():
    """Standard heat transfer calculation input."""
    from calculators.heat_transfer_calculator import HeatTransferInput
    return HeatTransferInput(
        heat_duty_mw=200.0,
        steam_temp_c=40.0,
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.0,
        cw_flow_rate_m3_hr=50000.0,
        tube_od_mm=25.4,
        tube_id_mm=23.4,
        tube_length_m=12.0,
        tube_count=18500,
        tube_material="titanium",
        design_u_value_w_m2k=3500.0,
        fouling_factor_m2k_w=0.00015,
    )


# ============================================================================
# Fouling Calculator Fixtures
# ============================================================================

@pytest.fixture
def fouling_input_standard():
    """Standard fouling calculation input."""
    from calculators.fouling_calculator import FoulingInput
    return FoulingInput(
        tube_material="titanium",
        cooling_water_source="cooling_tower",
        cooling_water_tds_ppm=2000.0,
        cooling_water_ph=7.8,
        cooling_water_temp_c=28.0,
        tube_velocity_m_s=2.2,
        operating_hours=4000.0,
        biocide_treatment="oxidizing",
        current_cleanliness_factor=0.85,
        design_fouling_factor_m2k_w=0.00015,
        cycles_of_concentration=4.0,
    )


@pytest.fixture
def fouling_input_severe():
    """Severe fouling calculation input."""
    from calculators.fouling_calculator import FoulingInput
    return FoulingInput(
        tube_material="admiralty_brass",
        cooling_water_source="river",
        cooling_water_tds_ppm=4000.0,
        cooling_water_ph=8.5,
        cooling_water_temp_c=35.0,
        tube_velocity_m_s=1.5,
        operating_hours=6000.0,
        biocide_treatment="none",
        current_cleanliness_factor=0.60,
        design_fouling_factor_m2k_w=0.00015,
        cycles_of_concentration=6.0,
    )


# ============================================================================
# Vacuum Calculator Fixtures
# ============================================================================

@pytest.fixture
def vacuum_input_standard():
    """Standard vacuum calculation input."""
    from calculators.vacuum_calculator import VacuumInput
    return VacuumInput(
        steam_temp_c=40.0,
        heat_load_mw=200.0,
        cw_inlet_temp_c=25.0,
        cw_flow_rate_m3_hr=50000.0,
        air_inleakage_rate_kg_hr=1.0,
        design_vacuum_mbar=50.0,
    )


@pytest.fixture
def vacuum_input_high_air():
    """Vacuum input with high air inleakage."""
    from calculators.vacuum_calculator import VacuumInput
    return VacuumInput(
        steam_temp_c=42.0,
        heat_load_mw=190.0,
        cw_inlet_temp_c=27.0,
        cw_flow_rate_m3_hr=48000.0,
        air_inleakage_rate_kg_hr=5.0,
        design_vacuum_mbar=50.0,
    )


# ============================================================================
# Golden Test Data Fixtures
# ============================================================================

@pytest.fixture
def golden_efficiency_test_data():
    """Golden test data for efficiency calculations."""
    return [
        {
            "name": "Standard Operation",
            "input": {
                "steam_temp_c": 40.0,
                "cw_inlet_temp_c": 25.0,
                "cw_outlet_temp_c": 35.0,
                "design_u_w_m2k": 3500.0,
                "actual_u_w_m2k": 3000.0,
            },
            "expected": {
                "ttd_c": 5.0,
                "itd_c": 15.0,
                "cleanliness_factor": 0.8571,
                "cw_temp_rise_c": 10.0,
            }
        },
        {
            "name": "Degraded Condition",
            "input": {
                "steam_temp_c": 45.0,
                "cw_inlet_temp_c": 28.0,
                "cw_outlet_temp_c": 38.0,
                "design_u_w_m2k": 3500.0,
                "actual_u_w_m2k": 2200.0,
            },
            "expected": {
                "ttd_c": 7.0,
                "itd_c": 17.0,
                "cleanliness_factor": 0.6286,
                "cw_temp_rise_c": 10.0,
            }
        },
    ]


@pytest.fixture
def golden_heat_transfer_test_data():
    """Golden test data for heat transfer calculations."""
    return [
        {
            "name": "Standard Titanium Tubes",
            "input": {
                "heat_duty_mw": 200.0,
                "steam_temp_c": 40.0,
                "cw_inlet_temp_c": 25.0,
                "cw_outlet_temp_c": 35.0,
                "tube_material": "titanium",
            },
            "expected_lmtd_range": (8.0, 12.0),
        },
    ]


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def generate_condenser_data():
    """Factory fixture for generating condenser data."""
    def _generator(
        condenser_id: str = "COND-GEN-001",
        steam_temp_c: float = 40.0,
        cw_inlet_temp_c: float = 25.0,
        cleanliness_factor: float = 0.85,
    ) -> Dict[str, Any]:
        return {
            "condenser_id": condenser_id,
            "steam_temp_c": steam_temp_c,
            "cw_inlet_temp_c": cw_inlet_temp_c,
            "cw_outlet_temp_c": cw_inlet_temp_c + 10.0,
            "heat_duty_mw": 200.0,
            "turbine_output_mw": 300.0,
            "design_bp_mmhg": 50.8,
            "actual_bp_mmhg": 55.0 + (1 - cleanliness_factor) * 50,
            "design_u_w_m2k": 3500.0,
            "actual_u_w_m2k": 3500.0 * cleanliness_factor,
            "area_m2": 17500.0,
            "tube_material": "titanium",
            "cleanliness_factor": cleanliness_factor,
        }
    return _generator


@pytest.fixture
def generate_batch_condenser_data():
    """Generate batch of condenser data for load testing."""
    def _generator(count: int = 10) -> List[Dict[str, Any]]:
        import random
        random.seed(42)  # Deterministic
        data = []
        for i in range(count):
            cf = 0.70 + random.random() * 0.25
            data.append({
                "condenser_id": f"COND-BATCH-{i:03d}",
                "steam_temp_c": 38.0 + random.random() * 10,
                "cw_inlet_temp_c": 22.0 + random.random() * 8,
                "heat_duty_mw": 150.0 + random.random() * 100,
                "cleanliness_factor": cf,
                "actual_u_w_m2k": 3500.0 * cf,
            })
        return data
    return _generator


# ============================================================================
# Calculator Instance Fixtures
# ============================================================================

@pytest.fixture
def efficiency_calculator():
    """Create EfficiencyCalculator instance."""
    from calculators.efficiency_calculator import EfficiencyCalculator
    return EfficiencyCalculator()


@pytest.fixture
def heat_transfer_calculator():
    """Create HeatTransferCalculator instance."""
    from calculators.heat_transfer_calculator import HeatTransferCalculator
    return HeatTransferCalculator()


@pytest.fixture
def fouling_calculator():
    """Create FoulingCalculator instance."""
    from calculators.fouling_calculator import FoulingCalculator
    return FoulingCalculator()


@pytest.fixture
def vacuum_calculator():
    """Create VacuumCalculator instance."""
    from calculators.vacuum_calculator import VacuumCalculator
    return VacuumCalculator()


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None

    return Timer()


@pytest.fixture
def performance_thresholds():
    """Performance threshold configurations."""
    return {
        "single_calculation_ms": 10.0,
        "batch_throughput_per_sec": 100,
        "memory_increase_mb": 50,
        "concurrent_execution_factor": 0.8,
    }


# ============================================================================
# Collection Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Add markers based on test path
        test_path = str(item.fspath)
        if "unit" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in test_path:
            item.add_marker(pytest.mark.e2e)
        elif "determinism" in test_path:
            item.add_marker(pytest.mark.determinism)
