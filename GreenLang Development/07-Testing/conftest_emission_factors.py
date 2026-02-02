# -*- coding: utf-8 -*-
"""
Pytest Configuration and Shared Fixtures for Emission Factor Tests

This module provides:
- Shared test fixtures
- Database setup/teardown
- Test data generators
- Mock configurations
- Performance measurement utilities
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
import random
import string
from datetime import date, datetime
from typing import List, Dict, Any
import sys
from greenlang.determinism import deterministic_random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.db.emission_factors_schema import create_database
from greenlang.sdk.emission_factor_client import EmissionFactorClient


# ==================== SESSION-SCOPE FIXTURES ====================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data (session scope)."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def sample_emission_factors_data():
    """Sample emission factors data for testing."""
    return {
        'diesel': {
            'factor_id': 'diesel_us_2024',
            'name': 'Diesel Combustion US 2024',
            'category': 'fuels',
            'subcategory': 'diesel',
            'emission_factor_kg_co2e': 10.21,
            'unit': 'gallon',
            'scope': 'Scope 1',
            'source_org': 'EPA',
            'source_uri': 'https://epa.gov/ghg',
            'standard': 'GHG Protocol',
            'year_applicable': 2024,
            'last_updated': '2024-01-01',
            'geographic_scope': 'United States',
            'geography_level': 'Country',
            'country_code': 'US',
            'data_quality_tier': 'Tier 1',
            'uncertainty_percent': 5.0,
            'gas_vectors': [
                {'gas_type': 'CO2', 'kg_per_unit': 10.15, 'gwp': 1},
                {'gas_type': 'CH4', 'kg_per_unit': 0.0004, 'gwp': 28},
                {'gas_type': 'N2O', 'kg_per_unit': 0.0002, 'gwp': 265}
            ]
        },
        'natural_gas': {
            'factor_id': 'natural_gas_us_2024',
            'name': 'Natural Gas Combustion US 2024',
            'category': 'fuels',
            'subcategory': 'natural_gas',
            'emission_factor_kg_co2e': 5.30,
            'unit': 'therms',
            'scope': 'Scope 1',
            'source_org': 'EPA',
            'source_uri': 'https://epa.gov/ghg',
            'standard': 'GHG Protocol',
            'year_applicable': 2024,
            'last_updated': '2024-01-01',
            'geographic_scope': 'United States',
            'geography_level': 'Country',
            'country_code': 'US',
            'data_quality_tier': 'Tier 1',
            'uncertainty_percent': 4.5,
            'gas_vectors': [
                {'gas_type': 'CO2', 'kg_per_unit': 5.28, 'gwp': 1},
                {'gas_type': 'CH4', 'kg_per_unit': 0.001, 'gwp': 28},
                {'gas_type': 'N2O', 'kg_per_unit': 0.0001, 'gwp': 265}
            ]
        },
        'electricity': {
            'factor_id': 'electricity_us_avg_2024',
            'name': 'US Grid Average 2024',
            'category': 'grids',
            'subcategory': 'us_average',
            'emission_factor_kg_co2e': 0.385,
            'unit': 'kwh',
            'scope': 'Scope 2 - Location-Based',
            'source_org': 'EPA eGRID',
            'source_uri': 'https://epa.gov/egrid',
            'standard': 'GHG Protocol',
            'year_applicable': 2024,
            'last_updated': '2024-01-01',
            'geographic_scope': 'United States',
            'geography_level': 'Country',
            'country_code': 'US',
            'data_quality_tier': 'Tier 2',
            'uncertainty_percent': 8.0,
            'renewable_share': 0.20
        }
    }


# ==================== FUNCTION-SCOPE FIXTURES ====================

@pytest.fixture
def temp_db():
    """Create temporary database for testing (function scope)."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_emission_factors.db"

    create_database(str(db_path))

    yield str(db_path)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def populated_db(temp_db, sample_emission_factors_data):
    """Create database populated with sample data."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert sample factors
    for fuel_type, factor_data in sample_emission_factors_data.items():
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level, country_code,
                data_quality_tier, uncertainty_percent,
                renewable_share
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            factor_data['factor_id'],
            factor_data['name'],
            factor_data['category'],
            factor_data['subcategory'],
            factor_data['emission_factor_kg_co2e'],
            factor_data['unit'],
            factor_data['scope'],
            factor_data['source_org'],
            factor_data['source_uri'],
            factor_data['standard'],
            factor_data['last_updated'],
            factor_data['year_applicable'],
            factor_data['geographic_scope'],
            factor_data['geography_level'],
            factor_data['country_code'],
            factor_data['data_quality_tier'],
            factor_data['uncertainty_percent'],
            factor_data.get('renewable_share')
        ))

        # Insert gas vectors if present
        if 'gas_vectors' in factor_data:
            for gas_vector in factor_data['gas_vectors']:
                cursor.execute("""
                    INSERT INTO factor_gas_vectors (factor_id, gas_type, kg_per_unit, gwp)
                    VALUES (?, ?, ?, ?)
                """, (
                    factor_data['factor_id'],
                    gas_vector['gas_type'],
                    gas_vector['kg_per_unit'],
                    gas_vector['gwp']
                ))

    conn.commit()
    conn.close()

    return temp_db


@pytest.fixture
def emission_factor_client(populated_db):
    """Create EmissionFactorClient instance."""
    client = EmissionFactorClient(db_path=populated_db)

    yield client

    client.close()


# ==================== TEST DATA GENERATORS ====================

class TestDataGenerator:
    """Generate test data for emission factors."""

    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate random string."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def random_factor_id() -> str:
        """Generate random factor ID."""
        fuel_types = ['diesel', 'gasoline', 'natural_gas', 'propane', 'coal']
        regions = ['us', 'eu', 'uk', 'cn', 'jp']
        year = deterministic_random().randint(2020, 2024)

        return f"{deterministic_random().choice(fuel_types)}_{deterministic_random().choice(regions)}_{year}"

    @staticmethod
    def generate_emission_factor(
        factor_id: str = None,
        category: str = 'fuels',
        ef_value: float = None
    ) -> Dict[str, Any]:
        """Generate random emission factor data."""
        if factor_id is None:
            factor_id = TestDataGenerator.random_factor_id()

        if ef_value is None:
            ef_value = random.uniform(1.0, 15.0)

        return {
            'factor_id': factor_id,
            'name': f"Test Factor {TestDataGenerator.random_string(8)}",
            'category': category,
            'subcategory': deterministic_random().choice(['diesel', 'gasoline', 'natural_gas', 'coal']),
            'emission_factor_value': ef_value,
            'unit': deterministic_random().choice(['gallon', 'liter', 'kg', 'm3', 'therms']),
            'scope': deterministic_random().choice(['Scope 1', 'Scope 2 - Location-Based', 'Scope 3']),
            'source_org': deterministic_random().choice(['EPA', 'DEFRA', 'IEA', 'IPCC']),
            'source_uri': f"https://test.com/{TestDataGenerator.random_string(8)}",
            'standard': 'GHG Protocol',
            'last_updated': '2024-01-01',
            'year_applicable': deterministic_random().randint(2020, 2024),
            'geographic_scope': deterministic_random().choice(['United States', 'United Kingdom', 'European Union']),
            'geography_level': 'Country',
            'country_code': deterministic_random().choice(['US', 'UK', 'EU', 'CN']),
            'data_quality_tier': deterministic_random().choice(['Tier 1', 'Tier 2', 'Tier 3']),
            'uncertainty_percent': random.uniform(3.0, 15.0)
        }

    @staticmethod
    def generate_batch_factors(count: int) -> List[Dict[str, Any]]:
        """Generate multiple emission factors."""
        return [TestDataGenerator.generate_emission_factor() for _ in range(count)]


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# ==================== PERFORMANCE MEASUREMENT UTILITIES ====================

@pytest.fixture
def performance_timer():
    """Provide performance timing utility."""
    import time

    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        def elapsed_ms(self):
            if self.start_time is None or self.end_time is None:
                return None
            return (self.end_time - self.start_time) * 1000

        def elapsed_seconds(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

    return PerformanceTimer()


# ==================== MOCK CONFIGURATIONS ====================

@pytest.fixture
def mock_api_response():
    """Mock API response structure."""
    return {
        'status': 'success',
        'data': {},
        'timestamp': DeterministicClock.now().isoformat(),
        'version': '1.0.0'
    }


# ==================== PYTEST HOOKS ====================

def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (requires database)"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests (benchmarking)"
    )
    config.addinivalue_line(
        "markers", "compliance: Compliance tests (regulatory)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Auto-mark tests based on file/function names
    for item in items:
        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)

        # Mark integration tests
        if "integration" in item.nodeid or "e2e" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark database tests
        if "database" in item.nodeid or "db" in item.nodeid:
            item.add_marker(pytest.mark.database)

        # Mark slow tests
        if "slow" in item.nodeid or item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)


def pytest_report_header(config):
    """Add custom header to pytest report."""
    return [
        "GreenLang Emission Factors Test Suite",
        "Target: 90%+ test coverage",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {sys.platform}"
    ]


# ==================== EXAMPLE USAGE ====================
"""
Example test using these fixtures:

def test_example(populated_db, emission_factor_client, performance_timer):
    # Use populated database
    # Use client to query
    # Use timer to measure performance

    performance_timer.start()

    factor = emission_factor_client.get_factor('diesel_us_2024')

    performance_timer.stop()

    assert factor.factor_id == 'diesel_us_2024'
    assert performance_timer.elapsed_ms() < 10  # <10ms target"""
