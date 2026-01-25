# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for GL-016 WATERGUARD test suite.

Provides comprehensive fixtures for water chemistry, boiler configurations,
mock connectors, and test data generators for 95%+ coverage testing.
"""

import pytest
import asyncio
import logging
import sys
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass

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
# Session-scoped fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def deterministic_seed():
    """Fixed seed for deterministic tests."""
    return 42


@pytest.fixture(scope="session")
def deterministic_timestamp():
    """Fixed timestamp for reproducibility tests."""
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# Temporary directory fixtures
# ============================================================================

@pytest.fixture
def test_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def golden_data_dir():
    """Directory for golden test data."""
    golden_dir = Path(__file__).parent / "golden_data"
    golden_dir.mkdir(parents=True, exist_ok=True)
    return golden_dir


# ============================================================================
# WaterSample fixtures for calculator testing
# ============================================================================

@pytest.fixture
def water_sample_fixture():
    """Create a WaterSample object for calculator testing."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=85.0,
        ph=8.5,
        conductivity_us_cm=1200.0,
        calcium_mg_l=50.0,
        magnesium_mg_l=30.0,
        sodium_mg_l=100.0,
        potassium_mg_l=10.0,
        chloride_mg_l=150.0,
        sulfate_mg_l=100.0,
        bicarbonate_mg_l=200.0,
        carbonate_mg_l=10.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=25.0,
        iron_mg_l=0.05,
        copper_mg_l=0.01,
        phosphate_mg_l=15.0,
        dissolved_oxygen_mg_l=0.02,
        total_alkalinity_mg_l_caco3=250.0,
        total_hardness_mg_l_caco3=180.0
    )


@pytest.fixture
def soft_water_sample():
    """Soft water sample (low hardness)."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=25.0,
        ph=7.0,
        conductivity_us_cm=150.0,
        calcium_mg_l=10.0,
        magnesium_mg_l=5.0,
        sodium_mg_l=20.0,
        potassium_mg_l=2.0,
        chloride_mg_l=15.0,
        sulfate_mg_l=10.0,
        bicarbonate_mg_l=30.0,
        carbonate_mg_l=0.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=5.0,
        iron_mg_l=0.01,
        copper_mg_l=0.005,
        phosphate_mg_l=0.0,
        dissolved_oxygen_mg_l=8.0,
        total_alkalinity_mg_l_caco3=30.0,
        total_hardness_mg_l_caco3=45.0
    )


@pytest.fixture
def hard_water_sample():
    """Hard water sample (high hardness)."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=25.0,
        ph=8.2,
        conductivity_us_cm=800.0,
        calcium_mg_l=120.0,
        magnesium_mg_l=60.0,
        sodium_mg_l=50.0,
        potassium_mg_l=8.0,
        chloride_mg_l=100.0,
        sulfate_mg_l=80.0,
        bicarbonate_mg_l=250.0,
        carbonate_mg_l=20.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=20.0,
        iron_mg_l=0.1,
        copper_mg_l=0.02,
        phosphate_mg_l=0.0,
        dissolved_oxygen_mg_l=6.0,
        total_alkalinity_mg_l_caco3=270.0,
        total_hardness_mg_l_caco3=480.0
    )


@pytest.fixture
def very_hard_water_sample():
    """Very hard water sample."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=30.0,
        ph=8.5,
        conductivity_us_cm=1500.0,
        calcium_mg_l=200.0,
        magnesium_mg_l=80.0,
        sodium_mg_l=80.0,
        potassium_mg_l=15.0,
        chloride_mg_l=150.0,
        sulfate_mg_l=200.0,
        bicarbonate_mg_l=350.0,
        carbonate_mg_l=30.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=35.0,
        iron_mg_l=0.2,
        copper_mg_l=0.03,
        phosphate_mg_l=0.0,
        dissolved_oxygen_mg_l=5.0,
        total_alkalinity_mg_l_caco3=400.0,
        total_hardness_mg_l_caco3=700.0
    )


@pytest.fixture
def high_ph_water_sample():
    """High pH water sample (alkaline)."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=85.0,
        ph=11.5,
        conductivity_us_cm=2500.0,
        calcium_mg_l=5.0,
        magnesium_mg_l=2.0,
        sodium_mg_l=400.0,
        potassium_mg_l=20.0,
        chloride_mg_l=100.0,
        sulfate_mg_l=50.0,
        bicarbonate_mg_l=50.0,
        carbonate_mg_l=100.0,
        hydroxide_mg_l=200.0,
        silica_mg_l=150.0,
        iron_mg_l=0.01,
        copper_mg_l=0.005,
        phosphate_mg_l=50.0,
        dissolved_oxygen_mg_l=0.01,
        total_alkalinity_mg_l_caco3=600.0,
        total_hardness_mg_l_caco3=20.0
    )


@pytest.fixture
def low_ph_water_sample():
    """Low pH water sample (acidic, corrosive)."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=25.0,
        ph=5.5,
        conductivity_us_cm=500.0,
        calcium_mg_l=30.0,
        magnesium_mg_l=15.0,
        sodium_mg_l=50.0,
        potassium_mg_l=5.0,
        chloride_mg_l=200.0,
        sulfate_mg_l=150.0,
        bicarbonate_mg_l=10.0,
        carbonate_mg_l=0.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=10.0,
        iron_mg_l=0.5,
        copper_mg_l=0.1,
        phosphate_mg_l=0.0,
        dissolved_oxygen_mg_l=9.0,
        total_alkalinity_mg_l_caco3=15.0,
        total_hardness_mg_l_caco3=130.0
    )


@pytest.fixture
def high_silica_water_sample():
    """High silica water sample for scaling tests."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=90.0,
        ph=9.0,
        conductivity_us_cm=2000.0,
        calcium_mg_l=40.0,
        magnesium_mg_l=50.0,
        sodium_mg_l=150.0,
        potassium_mg_l=12.0,
        chloride_mg_l=80.0,
        sulfate_mg_l=60.0,
        bicarbonate_mg_l=180.0,
        carbonate_mg_l=25.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=180.0,  # High silica
        iron_mg_l=0.08,
        copper_mg_l=0.015,
        phosphate_mg_l=10.0,
        dissolved_oxygen_mg_l=0.05,
        total_alkalinity_mg_l_caco3=220.0,
        total_hardness_mg_l_caco3=320.0
    )


@pytest.fixture
def boiler_water_sample():
    """Typical boiler water sample."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=180.0,
        ph=10.5,
        conductivity_us_cm=3000.0,
        calcium_mg_l=2.0,
        magnesium_mg_l=0.5,
        sodium_mg_l=800.0,
        potassium_mg_l=30.0,
        chloride_mg_l=150.0,
        sulfate_mg_l=100.0,
        bicarbonate_mg_l=100.0,
        carbonate_mg_l=200.0,
        hydroxide_mg_l=150.0,
        silica_mg_l=100.0,
        iron_mg_l=0.02,
        copper_mg_l=0.01,
        phosphate_mg_l=40.0,
        dissolved_oxygen_mg_l=0.005,
        total_alkalinity_mg_l_caco3=500.0,
        total_hardness_mg_l_caco3=8.0
    )


@pytest.fixture
def makeup_water_sample():
    """Typical makeup water sample for blowdown calculations."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=25.0,
        ph=7.5,
        conductivity_us_cm=300.0,
        calcium_mg_l=40.0,
        magnesium_mg_l=20.0,
        sodium_mg_l=30.0,
        potassium_mg_l=5.0,
        chloride_mg_l=35.0,
        sulfate_mg_l=25.0,
        bicarbonate_mg_l=120.0,
        carbonate_mg_l=0.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=15.0,
        iron_mg_l=0.05,
        copper_mg_l=0.01,
        phosphate_mg_l=0.0,
        dissolved_oxygen_mg_l=8.0,
        total_alkalinity_mg_l_caco3=100.0,
        total_hardness_mg_l_caco3=180.0
    )


@pytest.fixture
def circulating_water_sample():
    """Circulating water sample for cycles of concentration."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=40.0,
        ph=8.0,
        conductivity_us_cm=1500.0,
        calcium_mg_l=200.0,
        magnesium_mg_l=100.0,
        sodium_mg_l=150.0,
        potassium_mg_l=25.0,
        chloride_mg_l=175.0,
        sulfate_mg_l=125.0,
        bicarbonate_mg_l=400.0,
        carbonate_mg_l=20.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=75.0,
        iron_mg_l=0.1,
        copper_mg_l=0.02,
        phosphate_mg_l=5.0,
        dissolved_oxygen_mg_l=5.0,
        total_alkalinity_mg_l_caco3=450.0,
        total_hardness_mg_l_caco3=900.0
    )


# ============================================================================
# Scale and Corrosion condition fixtures
# ============================================================================

@pytest.fixture
def scale_conditions_fixture():
    """Create a ScaleConditions object for scale calculator testing."""
    from calculators.scale_formation_calculator import ScaleConditions
    return ScaleConditions(
        temperature_c=85.0,
        pressure_bar=10.0,
        flow_velocity_m_s=2.0,
        surface_roughness_um=10.0,
        operating_time_hours=1000.0,
        cycles_of_concentration=5.0,
        calcium_mg_l=50.0,
        magnesium_mg_l=30.0,
        sulfate_mg_l=100.0,
        silica_mg_l=25.0,
        iron_mg_l=0.05,
        copper_mg_l=0.01,
        ph=8.5,
        alkalinity_mg_l_caco3=250.0
    )


@pytest.fixture
def high_scale_conditions():
    """High scale formation conditions."""
    from calculators.scale_formation_calculator import ScaleConditions
    return ScaleConditions(
        temperature_c=95.0,
        pressure_bar=15.0,
        flow_velocity_m_s=0.5,  # Low velocity promotes scaling
        surface_roughness_um=50.0,
        operating_time_hours=2000.0,
        cycles_of_concentration=10.0,  # High concentration
        calcium_mg_l=150.0,  # High calcium
        magnesium_mg_l=80.0,
        sulfate_mg_l=200.0,
        silica_mg_l=100.0,  # High silica
        iron_mg_l=0.2,
        copper_mg_l=0.05,
        ph=9.0,  # Higher pH promotes scaling
        alkalinity_mg_l_caco3=400.0
    )


@pytest.fixture
def corrosion_conditions_fixture():
    """Create a CorrosionConditions object for corrosion calculator testing."""
    from calculators.corrosion_rate_calculator import CorrosionConditions
    return CorrosionConditions(
        temperature_c=85.0,
        pressure_bar=10.0,
        flow_velocity_m_s=2.0,
        ph=8.5,
        dissolved_oxygen_mg_l=0.02,
        carbon_dioxide_mg_l=5.0,
        chloride_mg_l=150.0,
        sulfate_mg_l=100.0,
        ammonia_mg_l=0.5,
        conductivity_us_cm=1200.0,
        material_type='carbon_steel',
        surface_finish='machined',
        operating_time_hours=1000.0,
        stress_level_mpa=100.0
    )


@pytest.fixture
def high_corrosion_conditions():
    """High corrosion conditions."""
    from calculators.corrosion_rate_calculator import CorrosionConditions
    return CorrosionConditions(
        temperature_c=60.0,
        pressure_bar=5.0,
        flow_velocity_m_s=5.0,  # High velocity
        ph=5.5,  # Low pH (acidic)
        dissolved_oxygen_mg_l=5.0,  # High oxygen
        carbon_dioxide_mg_l=50.0,  # High CO2
        chloride_mg_l=500.0,  # High chloride
        sulfate_mg_l=300.0,
        ammonia_mg_l=5.0,
        conductivity_us_cm=2500.0,
        material_type='carbon_steel',
        surface_finish='as_welded',
        operating_time_hours=5000.0,
        stress_level_mpa=200.0
    )


# ============================================================================
# Water chemistry dictionary fixtures for tools testing
# ============================================================================

@pytest.fixture
def sample_water_chemistry():
    """Standard water chemistry data dictionary for testing."""
    return {
        'ph': 8.5,
        'alkalinity_ppm': 250.0,
        'hardness_ppm': 180.0,
        'calcium_ppm': 50.0,
        'magnesium_ppm': 30.0,
        'chloride_ppm': 150.0,
        'sulfate_ppm': 100.0,
        'silica_ppm': 25.0,
        'tds_ppm': 800.0,
        'conductivity_us_cm': 1200.0,
        'temperature_c': 85.0,
        'dissolved_oxygen_ppm': 0.02,
        'iron_ppm': 0.05,
        'copper_ppm': 0.01,
        'phosphate_ppm': 15.0,
        'sulfite_ppm': 20.0,
        'hydrazine_ppm': 0.05,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def high_hardness_water():
    """High hardness water chemistry data."""
    return {
        'ph': 7.8,
        'alkalinity_ppm': 300.0,
        'hardness_ppm': 450.0,
        'calcium_ppm': 150.0,
        'magnesium_ppm': 80.0,
        'chloride_ppm': 200.0,
        'sulfate_ppm': 150.0,
        'silica_ppm': 40.0,
        'tds_ppm': 1500.0,
        'conductivity_us_cm': 2200.0,
        'temperature_c': 90.0,
        'dissolved_oxygen_ppm': 0.05,
        'iron_ppm': 0.15,
        'copper_ppm': 0.02,
        'phosphate_ppm': 10.0,
        'sulfite_ppm': 15.0,
        'hydrazine_ppm': 0.03,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def low_ph_water():
    """Low pH (corrosive) water chemistry data."""
    return {
        'ph': 6.5,
        'alkalinity_ppm': 100.0,
        'hardness_ppm': 80.0,
        'calcium_ppm': 25.0,
        'magnesium_ppm': 10.0,
        'chloride_ppm': 250.0,
        'sulfate_ppm': 200.0,
        'silica_ppm': 15.0,
        'tds_ppm': 900.0,
        'conductivity_us_cm': 1400.0,
        'temperature_c': 80.0,
        'dissolved_oxygen_ppm': 0.15,
        'iron_ppm': 0.20,
        'copper_ppm': 0.05,
        'phosphate_ppm': 5.0,
        'sulfite_ppm': 10.0,
        'hydrazine_ppm': 0.01,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def tools_water_chemistry_data():
    """Water chemistry data for tools testing (dictionary format)."""
    return {
        'pH': 8.5,
        'temperature': 85.0,
        'calcium_hardness': 150.0,
        'alkalinity': 250.0,
        'tds': 800.0,
        'chloride': 150.0,
        'sulfate': 100.0,
        'pressure': 10.0
    }


@pytest.fixture
def blowdown_water_data():
    """Water data for blowdown optimization testing."""
    return {
        'makeup_conductivity': 200.0,
        'blowdown_conductivity': 2000.0,
        'tds': 2000.0,
        'alkalinity': 400.0,
        'temperature': 180.0,
        'pressure': 10.0,
        'water_cost': 0.5,
        'energy_cost': 0.08
    }


@pytest.fixture
def chemical_usage_data():
    """Chemical usage data for optimization testing."""
    return {
        'phosphate': 5.0,
        'oxygen_scavenger': 2.0,
        'amine': 1.0,
        'polymer': 0.5
    }


@pytest.fixture
def optimization_targets():
    """Optimization targets for chemical testing."""
    return {
        'phosphate_residual': 50.0,
        'condensate_pH': 8.8,
        'sludge_conditioner_need': 50.0,
        'chemical_prices': {
            'phosphate': 2.50,
            'oxygen_scavenger': 3.00,
            'amine': 5.00,
            'polymer': 4.00
        }
    }


# ============================================================================
# Boiler configuration fixtures
# ============================================================================

@pytest.fixture
def standard_boiler_config():
    """Standard boiler configuration."""
    return {
        'boiler_id': 'BOILER-001',
        'boiler_name': 'Test Industrial Boiler',
        'capacity_mw': 25.0,
        'operating_pressure_bar': 40.0,
        'max_pressure_bar': 50.0,
        'feedwater_temp_c': 105.0,
        'steam_temp_c': 250.0,
        'blowdown_rate_percent': 5.0,
        'cycles_of_concentration': 10.0,
        'fuel_type': 'natural_gas',
        'boiler_type': 'water_tube',
        'location': 'Test Plant A',
        'scada_host': 'localhost',
        'scada_port': 4840,
        'erp_host': 'localhost',
        'erp_port': 8000,
        'target_ph': 8.5,
        'target_conductivity': 1200.0,
        'max_chloride_ppm': 200.0,
        'max_silica_ppm': 30.0
    }


@pytest.fixture
def high_pressure_boiler_config():
    """High-pressure boiler configuration."""
    return {
        'boiler_id': 'BOILER-HP-001',
        'boiler_name': 'High Pressure Test Boiler',
        'capacity_mw': 50.0,
        'operating_pressure_bar': 100.0,
        'max_pressure_bar': 120.0,
        'feedwater_temp_c': 110.0,
        'steam_temp_c': 350.0,
        'blowdown_rate_percent': 3.0,
        'cycles_of_concentration': 15.0,
        'fuel_type': 'natural_gas',
        'boiler_type': 'water_tube',
        'location': 'Test Plant B',
        'scada_host': 'localhost',
        'scada_port': 4840,
        'erp_host': 'localhost',
        'erp_port': 8000,
        'target_ph': 9.0,
        'target_conductivity': 800.0,
        'max_chloride_ppm': 100.0,
        'max_silica_ppm': 20.0
    }


@pytest.fixture
def low_pressure_boiler_config():
    """Low-pressure boiler configuration."""
    return {
        'boiler_id': 'BOILER-LP-001',
        'boiler_name': 'Low Pressure Test Boiler',
        'capacity_mw': 10.0,
        'operating_pressure_bar': 10.0,
        'max_pressure_bar': 15.0,
        'feedwater_temp_c': 80.0,
        'steam_temp_c': 180.0,
        'blowdown_rate_percent': 8.0,
        'cycles_of_concentration': 5.0,
        'fuel_type': 'natural_gas',
        'boiler_type': 'fire_tube',
        'location': 'Test Plant C',
        'target_ph': 10.5,
        'target_conductivity': 3000.0,
        'max_chloride_ppm': 300.0,
        'max_silica_ppm': 150.0
    }


# ============================================================================
# Mock SCADA and connector fixtures
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
def mock_scada_data():
    """Mock SCADA sensor data."""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'tags': {
            'BOILER_PRESSURE': {'value': 40.5, 'quality': 'GOOD', 'unit': 'bar'},
            'FEEDWATER_TEMP': {'value': 105.2, 'quality': 'GOOD', 'unit': 'C'},
            'STEAM_TEMP': {'value': 250.8, 'quality': 'GOOD', 'unit': 'C'},
            'FEEDWATER_FLOW': {'value': 45.5, 'quality': 'GOOD', 'unit': 'm3/hr'},
            'BLOWDOWN_FLOW': {'value': 2.3, 'quality': 'GOOD', 'unit': 'm3/hr'},
            'BOILER_LEVEL': {'value': 75.0, 'quality': 'GOOD', 'unit': '%'},
            'CONDUCTIVITY': {'value': 1200.0, 'quality': 'GOOD', 'unit': 'uS/cm'},
            'PH_SENSOR': {'value': 8.5, 'quality': 'GOOD', 'unit': 'pH'},
            'DO_SENSOR': {'value': 0.02, 'quality': 'GOOD', 'unit': 'ppm'},
            'SILICA_SENSOR': {'value': 25.0, 'quality': 'GOOD', 'unit': 'ppm'},
            'CHLORIDE_SENSOR': {'value': 150.0, 'quality': 'GOOD', 'unit': 'ppm'},
        }
    }


@pytest.fixture
def mock_water_analyzer():
    """Mock water analyzer device."""
    analyzer = AsyncMock()
    analyzer.connect = AsyncMock(return_value=True)
    analyzer.disconnect = AsyncMock(return_value=True)
    analyzer.get_analysis = AsyncMock()
    analyzer.calibrate = AsyncMock(return_value=True)
    analyzer.get_status = AsyncMock(return_value={
        'status': 'OK',
        'last_calibration': datetime.utcnow().isoformat()
    })
    analyzer.is_connected = Mock(return_value=True)
    return analyzer


@pytest.fixture
def mock_chemical_dosing_system():
    """Mock chemical dosing system."""
    dosing_system = AsyncMock()
    dosing_system.connect = AsyncMock(return_value=True)
    dosing_system.disconnect = AsyncMock(return_value=True)
    dosing_system.dose_chemical = AsyncMock(return_value=True)
    dosing_system.get_chemical_inventory = AsyncMock()
    dosing_system.get_pump_status = AsyncMock()
    dosing_system.is_connected = Mock(return_value=True)
    return dosing_system


@pytest.fixture
def mock_erp_system():
    """Mock ERP system."""
    erp = AsyncMock()
    erp.connect = AsyncMock(return_value=True)
    erp.disconnect = AsyncMock(return_value=True)
    erp.get_chemical_cost = AsyncMock()
    erp.get_water_cost = AsyncMock()
    erp.get_energy_cost = AsyncMock()
    erp.create_work_order = AsyncMock()
    erp.get_maintenance_schedule = AsyncMock()
    erp.is_connected = Mock(return_value=True)
    return erp


# ============================================================================
# Chemical and cost fixtures
# ============================================================================

@pytest.fixture
def chemical_inventory():
    """Sample chemical inventory data."""
    return {
        'phosphate': {
            'chemical_id': 'CHEM-PO4',
            'name': 'Trisodium Phosphate',
            'concentration_percent': 30.0,
            'volume_liters': 500.0,
            'unit_cost_per_liter': 5.50,
            'supplier': 'ChemSupply Inc.',
            'last_refill': (datetime.utcnow() - timedelta(days=15)).isoformat(),
            'reorder_level_liters': 100.0
        },
        'sulfite': {
            'chemical_id': 'CHEM-SO3',
            'name': 'Sodium Sulfite',
            'concentration_percent': 25.0,
            'volume_liters': 300.0,
            'unit_cost_per_liter': 4.25,
            'supplier': 'ChemSupply Inc.',
            'last_refill': (datetime.utcnow() - timedelta(days=10)).isoformat(),
            'reorder_level_liters': 50.0
        },
        'caustic': {
            'chemical_id': 'CHEM-NaOH',
            'name': 'Sodium Hydroxide',
            'concentration_percent': 50.0,
            'volume_liters': 400.0,
            'unit_cost_per_liter': 3.75,
            'supplier': 'ChemSupply Inc.',
            'last_refill': (datetime.utcnow() - timedelta(days=20)).isoformat(),
            'reorder_level_liters': 75.0
        }
    }


@pytest.fixture
def chemical_prices():
    """Chemical prices for cost calculations."""
    return {
        'phosphate': 2.50,
        'oxygen_scavenger': 3.00,
        'amine': 5.00,
        'polymer': 4.00,
        'sulfite': 1.50,
        'caustic': 0.80,
        'acid': 0.60
    }


@pytest.fixture
def mock_erp_data():
    """Mock ERP system data."""
    return {
        'chemical_costs': {
            'phosphate': 5.50,
            'sulfite': 4.25,
            'caustic': 3.75,
            'hydrazine': 12.00
        },
        'water_cost_per_m3': 2.50,
        'energy_cost_per_kwh': 0.12,
        'steam_value_per_ton': 25.00,
        'maintenance_costs': {
            'preventive': 500.00,
            'corrective': 2000.00,
            'emergency': 5000.00
        }
    }


# ============================================================================
# Golden test data fixtures
# ============================================================================

@pytest.fixture
def golden_lsi_test_data():
    """Golden test data for LSI calculations."""
    return [
        {
            'inputs': {'pH': 7.5, 'temperature': 25, 'calcium_hardness': 100, 'alkalinity': 100, 'tds': 500},
            'expected_lsi': -0.21,
            'tolerance': 0.05
        },
        {
            'inputs': {'pH': 8.5, 'temperature': 50, 'calcium_hardness': 200, 'alkalinity': 200, 'tds': 1000},
            'expected_lsi': 0.85,
            'tolerance': 0.05
        },
        {
            'inputs': {'pH': 6.5, 'temperature': 25, 'calcium_hardness': 50, 'alkalinity': 50, 'tds': 300},
            'expected_lsi': -1.73,
            'tolerance': 0.05
        },
    ]


@pytest.fixture
def golden_rsi_test_data():
    """Golden test data for RSI calculations."""
    return [
        {'pH': 7.5, 'pHs': 7.71, 'expected_rsi': 7.92, 'tolerance': 0.05},
        {'pH': 8.0, 'pHs': 7.50, 'expected_rsi': 7.00, 'tolerance': 0.05},
        {'pH': 8.5, 'pHs': 7.65, 'expected_rsi': 6.80, 'tolerance': 0.05},
    ]


@pytest.fixture
def golden_blowdown_test_data():
    """Golden test data for blowdown calculations."""
    return [
        {'steam_rate': 1000, 'cycles': 5, 'expected_blowdown': 250.0, 'tolerance': 1.0},
        {'steam_rate': 2000, 'cycles': 10, 'expected_blowdown': 222.22, 'tolerance': 1.0},
        {'steam_rate': 5000, 'cycles': 8, 'expected_blowdown': 714.29, 'tolerance': 1.0},
    ]


# ============================================================================
# Historical data fixtures
# ============================================================================

@pytest.fixture
def historical_water_data():
    """Historical water chemistry data batch."""
    base_time = datetime.utcnow()
    return [
        {
            'timestamp': (base_time - timedelta(hours=i)).isoformat(),
            'ph': 8.5 + (i * 0.05),
            'alkalinity_ppm': 250.0 + (i * 5),
            'hardness_ppm': 180.0 + (i * 3),
            'conductivity_us_cm': 1200.0 + (i * 20),
            'tds_ppm': 800.0 + (i * 10),
            'chloride_ppm': 150.0 + (i * 2),
            'silica_ppm': 25.0 + (i * 0.5)
        }
        for i in range(24)
    ]


@pytest.fixture
def historical_blowdown_data():
    """Historical blowdown data."""
    base_time = datetime.utcnow()
    return [
        {
            'timestamp': (base_time - timedelta(hours=i)).isoformat(),
            'blowdown_rate_m3_hr': 2.3 + (i * 0.1),
            'cycles_of_concentration': 10.0 + (i * 0.2),
            'tds_feedwater': 200.0 + (i * 5),
            'tds_boiler': 2000.0 + (i * 50),
            'energy_loss_kw': 150.0 + (i * 5)
        }
        for i in range(24)
    ]


# ============================================================================
# Agent configuration fixtures
# ============================================================================

@pytest.fixture
def agent_config():
    """Standard agent configuration."""
    return {
        'agent_id': 'GL-016-TEST',
        'agent_name': 'TestWaterGuardAgent',
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


# ============================================================================
# Helper functions for tests
# ============================================================================

def calculate_expected_hash(data: Dict[str, Any]) -> str:
    """Calculate expected SHA-256 hash for verification."""
    canonical_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def create_water_sample_from_dict(data: Dict[str, Any]):
    """Create WaterSample from dictionary."""
    from calculators.water_chemistry_calculator import WaterSample
    return WaterSample(
        temperature_c=data.get('temperature_c', 25.0),
        ph=data.get('ph', 7.0),
        conductivity_us_cm=data.get('conductivity_us_cm', 500.0),
        calcium_mg_l=data.get('calcium_mg_l', 50.0),
        magnesium_mg_l=data.get('magnesium_mg_l', 25.0),
        sodium_mg_l=data.get('sodium_mg_l', 50.0),
        potassium_mg_l=data.get('potassium_mg_l', 5.0),
        chloride_mg_l=data.get('chloride_mg_l', 50.0),
        sulfate_mg_l=data.get('sulfate_mg_l', 50.0),
        bicarbonate_mg_l=data.get('bicarbonate_mg_l', 100.0),
        carbonate_mg_l=data.get('carbonate_mg_l', 0.0),
        hydroxide_mg_l=data.get('hydroxide_mg_l', 0.0),
        silica_mg_l=data.get('silica_mg_l', 10.0),
        iron_mg_l=data.get('iron_mg_l', 0.05),
        copper_mg_l=data.get('copper_mg_l', 0.01),
        phosphate_mg_l=data.get('phosphate_mg_l', 0.0),
        dissolved_oxygen_mg_l=data.get('dissolved_oxygen_mg_l', 8.0),
        total_alkalinity_mg_l_caco3=data.get('total_alkalinity_mg_l_caco3', 100.0),
        total_hardness_mg_l_caco3=data.get('total_hardness_mg_l_caco3', 200.0)
    )


# ============================================================================
# Pytest configuration and markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "determinism: mark test as determinism test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "compliance: mark test as compliance test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "scada: mark test as SCADA integration test")
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "golden: mark test as golden data test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
