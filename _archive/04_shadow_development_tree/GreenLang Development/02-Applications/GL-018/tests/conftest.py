"""
GL-018 FLUEFLOW - Test Configuration and Fixtures

Provides shared fixtures, test data generators, and test utilities
for comprehensive testing of FLUEFLOW calculators and agent.

Author: GL-TestEngineer
Version: 1.0.0
"""

import sys
import os
import json
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from decimal import Decimal
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calculators.combustion_analyzer import (
    CombustionAnalyzer,
    CombustionInput,
    FuelType,
    GasBasis,
    FUEL_PROPERTIES
)
from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput
)
from calculators.air_fuel_ratio_calculator import (
    AirFuelRatioCalculator,
    AirFuelRatioInput
)
from calculators.emissions_calculator import (
    EmissionsCalculator,
    EmissionsInput
)
from calculators.provenance import ProvenanceTracker


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
# CALCULATOR FIXTURES
# =============================================================================

@pytest.fixture
def combustion_analyzer():
    """Create CombustionAnalyzer instance."""
    return CombustionAnalyzer()


@pytest.fixture
def efficiency_calculator():
    """Create EfficiencyCalculator instance."""
    return EfficiencyCalculator()


@pytest.fixture
def air_fuel_ratio_calculator():
    """Create AirFuelRatioCalculator instance."""
    return AirFuelRatioCalculator()


@pytest.fixture
def emissions_calculator():
    """Create EmissionsCalculator instance."""
    return EmissionsCalculator()


# =============================================================================
# INPUT DATA FIXTURES - NATURAL GAS
# =============================================================================

@pytest.fixture
def natural_gas_combustion_input():
    """Valid natural gas combustion input data."""
    return CombustionInput(
        O2_pct=3.5,
        CO2_pct=12.0,
        CO_ppm=50.0,
        NOx_ppm=150.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        fuel_type=FuelType.NATURAL_GAS.value,
        gas_basis=GasBasis.DRY.value,
        SO2_ppm=0.0,
        h2o_pct_wet=None
    )


@pytest.fixture
def natural_gas_efficiency_input():
    """Valid natural gas efficiency input data."""
    return EfficiencyInput(
        fuel_type="Natural Gas",
        fuel_flow_rate_kg_hr=1000.0,
        O2_pct_dry=3.5,
        CO2_pct_dry=12.0,
        CO_ppm=50.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        excess_air_pct=20.0,
        heat_input_mw=10.0,
        heat_output_mw=8.5,
        fuel_temp_c=25.0,
        moisture_in_fuel_pct=0.0
    )


@pytest.fixture
def natural_gas_air_fuel_input():
    """Valid natural gas air-fuel ratio input data."""
    return AirFuelRatioInput(
        fuel_type="Natural Gas",
        O2_measured_pct=3.5,
        moisture_pct=0.0
    )


@pytest.fixture
def natural_gas_emissions_input():
    """Valid natural gas emissions input data."""
    return EmissionsInput(
        NOx_ppm=150.0,
        CO_ppm=50.0,
        SO2_ppm=0.0,
        CO2_pct=12.0,
        O2_pct=3.5,
        flue_gas_temp_c=180.0,
        flue_gas_flow_nm3_hr=50000.0,
        fuel_type="Natural Gas",
        reference_O2_pct=3.0,
        moisture_pct=10.0
    )


# =============================================================================
# INPUT DATA FIXTURES - FUEL OIL
# =============================================================================

@pytest.fixture
def fuel_oil_combustion_input():
    """Valid fuel oil combustion input data."""
    return CombustionInput(
        O2_pct=3.0,
        CO2_pct=14.0,
        CO_ppm=100.0,
        NOx_ppm=200.0,
        flue_gas_temp_c=220.0,
        ambient_temp_c=25.0,
        fuel_type=FuelType.FUEL_OIL.value,
        gas_basis=GasBasis.DRY.value,
        SO2_ppm=150.0
    )


@pytest.fixture
def fuel_oil_efficiency_input():
    """Valid fuel oil efficiency input data."""
    return EfficiencyInput(
        fuel_type="Fuel Oil",
        fuel_flow_rate_kg_hr=800.0,
        O2_pct_dry=3.0,
        CO2_pct_dry=14.0,
        CO_ppm=100.0,
        flue_gas_temp_c=220.0,
        ambient_temp_c=25.0,
        excess_air_pct=17.6,
        heat_input_mw=9.0,
        heat_output_mw=7.5,
        moisture_in_fuel_pct=0.0
    )


# =============================================================================
# INPUT DATA FIXTURES - COAL
# =============================================================================

@pytest.fixture
def coal_combustion_input():
    """Valid coal combustion input data."""
    return CombustionInput(
        O2_pct=4.5,
        CO2_pct=16.0,
        CO_ppm=200.0,
        NOx_ppm=350.0,
        flue_gas_temp_c=250.0,
        ambient_temp_c=25.0,
        fuel_type=FuelType.COAL.value,
        gas_basis=GasBasis.DRY.value,
        SO2_ppm=500.0
    )


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================

@pytest.fixture
def low_O2_combustion_input():
    """Low O2 combustion input (rich combustion)."""
    return CombustionInput(
        O2_pct=1.0,
        CO2_pct=15.0,
        CO_ppm=800.0,
        NOx_ppm=100.0,
        flue_gas_temp_c=200.0,
        ambient_temp_c=25.0,
        fuel_type=FuelType.NATURAL_GAS.value,
        gas_basis=GasBasis.DRY.value
    )


@pytest.fixture
def high_O2_combustion_input():
    """High O2 combustion input (excessive air)."""
    return CombustionInput(
        O2_pct=10.0,
        CO2_pct=5.0,
        CO_ppm=20.0,
        NOx_ppm=80.0,
        flue_gas_temp_c=150.0,
        ambient_temp_c=25.0,
        fuel_type=FuelType.NATURAL_GAS.value,
        gas_basis=GasBasis.DRY.value
    )


@pytest.fixture
def wet_basis_combustion_input():
    """Wet basis combustion input (requires conversion)."""
    return CombustionInput(
        O2_pct=3.0,
        CO2_pct=10.8,
        CO_ppm=45.0,
        NOx_ppm=135.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        fuel_type=FuelType.NATURAL_GAS.value,
        gas_basis=GasBasis.WET.value,
        h2o_pct_wet=10.0
    )


# =============================================================================
# PARAMETERIZED TEST DATA
# =============================================================================

@pytest.fixture
def excess_air_test_cases():
    """Test cases for excess air calculation with known values."""
    return [
        # (O2_pct, expected_excess_air_pct)
        (0.0, 0.0),
        (1.0, 5.0),
        (2.0, 10.5),
        (3.0, 16.7),
        (3.5, 20.0),
        (4.0, 23.5),
        (5.0, 31.2),
        (6.0, 40.0),
        (8.0, 61.5),
        (10.0, 90.9),
    ]


@pytest.fixture
def fuel_properties_test_cases():
    """Test cases for all fuel types."""
    return [
        {
            "fuel_type": FuelType.NATURAL_GAS.value,
            "expected_CO2_max": 11.8,
            "expected_stoich_air": 17.2,
        },
        {
            "fuel_type": FuelType.FUEL_OIL.value,
            "expected_CO2_max": 15.5,
            "expected_stoich_air": 14.5,
        },
        {
            "fuel_type": FuelType.COAL.value,
            "expected_CO2_max": 18.5,
            "expected_stoich_air": 9.5,
        },
        {
            "fuel_type": FuelType.DIESEL.value,
            "expected_CO2_max": 15.3,
            "expected_stoich_air": 14.3,
        },
        {
            "fuel_type": FuelType.PROPANE.value,
            "expected_CO2_max": 13.7,
            "expected_stoich_air": 15.7,
        },
    ]


@pytest.fixture
def emissions_conversion_test_cases():
    """Test cases for emissions unit conversions."""
    return [
        # (ppm, MW, expected_mg_nm3)
        (100.0, 46.0, 205.25),  # NOx
        (100.0, 28.0, 124.93),  # CO
        (100.0, 64.0, 285.58),  # SO2
        (50.0, 46.0, 102.62),   # NOx
        (200.0, 28.0, 249.87),  # CO
    ]


# =============================================================================
# BENCHMARK FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_dataset():
    """Large dataset for performance benchmarking."""
    return [
        CombustionInput(
            O2_pct=3.5 + (i % 10) * 0.1,
            CO2_pct=12.0 - (i % 10) * 0.1,
            CO_ppm=50.0 + (i % 100) * 5.0,
            NOx_ppm=150.0 + (i % 100) * 2.0,
            flue_gas_temp_c=180.0 + (i % 50) * 2.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )
        for i in range(1000)
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
        assert provenance.calculator_name != ""
        assert provenance.calculator_version != ""
        assert provenance.provenance_hash is not None
        assert len(provenance.provenance_hash) == 64  # SHA-256
        assert len(provenance.calculation_steps) > 0

        # Validate each step
        for step in provenance.calculation_steps:
            assert step.step_number > 0
            assert step.description != ""
            assert step.operation != ""

        return True

    return validate


@pytest.fixture
def tolerance_checker():
    """Helper function for floating point comparisons."""
    def check(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
        """Check if actual is within tolerance of expected."""
        import math
        return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)

    return check


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

@pytest.fixture
def scada_data_generator():
    """Generate mock SCADA data for testing."""
    def generate(num_points=100, burner_id="BURNER_001"):
        """Generate time-series SCADA data."""
        import random
        from datetime import timedelta

        base_time = datetime.now(timezone.utc)
        data_points = []

        for i in range(num_points):
            timestamp = base_time + timedelta(seconds=i * 60)  # 1 minute intervals

            data_point = {
                "timestamp": timestamp.isoformat(),
                "burner_id": burner_id,
                "O2_pct": 3.5 + random.gauss(0, 0.2),
                "CO2_pct": 12.0 + random.gauss(0, 0.5),
                "CO_ppm": 50.0 + random.gauss(0, 10.0),
                "NOx_ppm": 150.0 + random.gauss(0, 20.0),
                "SO2_ppm": 0.0,
                "stack_temp_f": 356.0 + random.gauss(0, 10.0),  # ~180°C
                "ambient_temp_f": 77.0,  # ~25°C
                "fuel_flow_rate": 1000.0 + random.gauss(0, 50.0),
                "air_flow_rate": 17200.0 + random.gauss(0, 500.0),
            }
            data_points.append(data_point)

        return data_points

    return generate


# =============================================================================
# TEST DATA FILES
# =============================================================================

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def asme_ptc_reference_data(test_data_dir):
    """Load ASME PTC 4.1 reference test data."""
    reference_file = test_data_dir / "asme_ptc_reference.json"

    # Create default reference data if file doesn't exist
    if not reference_file.exists():
        test_data_dir.mkdir(exist_ok=True)
        reference_data = {
            "natural_gas_test_1": {
                "O2_pct": 3.5,
                "CO2_pct": 12.0,
                "excess_air_pct": 20.0,
                "stack_temp_c": 180.0,
                "efficiency_pct": 85.0
            },
            "fuel_oil_test_1": {
                "O2_pct": 3.0,
                "CO2_pct": 14.0,
                "excess_air_pct": 17.6,
                "stack_temp_c": 220.0,
                "efficiency_pct": 83.0
            }
        }
        reference_file.write_text(json.dumps(reference_data, indent=2))
        return reference_data

    return json.loads(reference_file.read_text())


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


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add calculator marker to calculator tests
        if "calculator" in item.nodeid:
            item.add_marker(pytest.mark.calculator)

        # Add critical marker to validation tests
        if "validation" in item.nodeid or "provenance" in item.nodeid:
            item.add_marker(pytest.mark.critical)
