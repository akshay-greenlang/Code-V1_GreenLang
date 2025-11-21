"""
Enhanced pytest configuration for comprehensive GreenLang testing.
Provides fixtures, mocks, and test utilities for achieving 85%+ coverage.
"""

import os
import sys
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Any, Dict, List, Optional, Generator
from pathlib import Path
import tempfile
import json
import random
import string
from faker import Faker

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize faker
fake = Faker()
Faker.seed(42)
random.seed(42)


# ================ Core Test Fixtures ================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return {
        "environment": "test",
        "debug": True,
        "database": {
            "url": "sqlite:///:memory:",
            "pool_size": 5,
            "echo": False
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "cors_origins": ["http://localhost:3000"]
        },
        "security": {
            "secret_key": "test-secret-key",
            "algorithm": "HS256",
            "token_expire_minutes": 30
        },
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 60
        }
    }


# ================ Database Fixtures ================

@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.query = Mock(return_value=Mock(
        filter=Mock(return_value=Mock(
            first=Mock(return_value=None),
            all=Mock(return_value=[])
        ))
    ))
    session.close = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)
    return session


@pytest.fixture
def mock_transaction_manager():
    """Create mock transaction manager."""
    manager = Mock()
    manager.begin = Mock(return_value=Mock())
    manager.commit = Mock()
    manager.rollback = Mock()
    manager.savepoint = Mock(return_value="savepoint_1")
    manager.release_savepoint = Mock()
    return manager


# ================ Authentication Fixtures ================

@pytest.fixture
def mock_user():
    """Create mock user object."""
    return {
        "id": "user-123",
        "username": "test_user",
        "email": "test@example.com",
        "roles": ["user", "analyst"],
        "permissions": ["read", "write", "calculate"],
        "is_active": True,
        "created_at": datetime.utcnow()
    }


@pytest.fixture
def mock_jwt_token():
    """Create mock JWT token."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"


@pytest.fixture
def mock_auth_service():
    """Create mock authentication service."""
    service = Mock()
    service.authenticate = Mock(return_value={"user_id": "123", "token": "mock-token"})
    service.verify_token = Mock(return_value=True)
    service.refresh_token = Mock(return_value="new-mock-token")
    service.has_permission = Mock(return_value=True)
    return service


# ================ Pipeline Fixtures ================

@pytest.fixture
def mock_pipeline_stage():
    """Create mock pipeline stage."""
    stage = Mock()
    stage.name = "test_stage"
    stage.execute = Mock(return_value={"status": "success", "data": {}})
    stage.validate = Mock(return_value=True)
    stage.rollback = Mock()
    return stage


@pytest.fixture
def mock_pipeline_context():
    """Create mock pipeline context."""
    return {
        "request_id": "req-" + "".join(random.choices(string.hexdigits, k=8)),
        "user_id": "user-123",
        "timestamp": datetime.utcnow(),
        "metadata": {},
        "checkpoints": [],
        "errors": []
    }


# ================ Agent Fixtures ================

@pytest.fixture
def base_agent_config():
    """Create base agent configuration."""
    return {
        "name": "test_agent",
        "version": "1.0.0",
        "type": "calculator",
        "enabled": True,
        "timeout": 30,
        "retry_count": 3,
        "retry_delay": 1
    }


@pytest.fixture
def mock_emission_factors():
    """Create mock emission factor database."""
    return {
        ("diesel", "US", "stationary_combustion"): Decimal("2.68"),
        ("natural_gas", "US", "stationary_combustion"): Decimal("1.93"),
        ("electricity", "US", "grid"): Decimal("0.42"),
        ("coal", "US", "stationary_combustion"): Decimal("3.45"),
        ("gasoline", "US", "mobile_combustion"): Decimal("2.35"),
        ("jet_fuel", "GLOBAL", "aviation"): Decimal("3.16"),
        # Scope 3 factors
        ("steel", "GLOBAL", "purchased_goods"): Decimal("2.0"),
        ("aluminum", "GLOBAL", "purchased_goods"): Decimal("11.7"),
        ("cement", "GLOBAL", "purchased_goods"): Decimal("0.93"),
        ("plastic", "GLOBAL", "purchased_goods"): Decimal("3.5"),
        # Transport factors
        ("truck", "US", "freight"): Decimal("0.162"),  # kg CO2e per tonne-km
        ("rail", "US", "freight"): Decimal("0.021"),
        ("ship", "GLOBAL", "freight"): Decimal("0.012"),
        ("air", "GLOBAL", "freight"): Decimal("0.602")
    }


@pytest.fixture
def sample_shipment_data():
    """Create sample shipment data for testing."""
    return {
        "shipment_id": fake.uuid4(),
        "product_category": random.choice(["steel", "aluminum", "cement"]),
        "weight_tonnes": round(random.uniform(0.1, 100.0), 2),
        "origin_country": fake.country_code(),
        "destination_country": "US",
        "transport_mode": random.choice(["truck", "rail", "ship"]),
        "distance_km": round(random.uniform(100, 5000), 0),
        "import_date": fake.date_between(start_date="-1y", end_date="today"),
        "supplier_name": fake.company(),
        "hs_code": f"{random.randint(2500, 2900)}.{random.randint(10, 99)}"
    }


# ================ Provenance Fixtures ================

@pytest.fixture
def mock_provenance_tracker():
    """Create mock provenance tracker."""
    tracker = Mock()
    tracker.track_input = Mock(return_value="input-hash-123")
    tracker.track_calculation = Mock(return_value="calc-hash-456")
    tracker.track_output = Mock(return_value="output-hash-789")
    tracker.generate_hash = Mock(return_value="a" * 64)  # SHA-256 hash
    tracker.get_audit_trail = Mock(return_value=[])
    return tracker


# ================ Security Fixtures ================

@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter."""
    limiter = Mock()
    limiter.check = Mock(return_value=True)
    limiter.increment = Mock()
    limiter.reset = Mock()
    limiter.get_remaining = Mock(return_value=59)
    return limiter


@pytest.fixture
def mock_csrf_protection():
    """Create mock CSRF protection."""
    protection = Mock()
    protection.generate_token = Mock(return_value="csrf-token-123")
    protection.validate_token = Mock(return_value=True)
    return protection


# ================ Test Data Generators ================

class TestDataGenerator:
    """Generate realistic test data for various scenarios."""

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)

    def generate_fuel_consumption_data(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """Generate test fuel consumption data."""
        records = []
        for _ in range(num_records):
            record = {
                'record_id': self.faker.uuid4(),
                'fuel_type': random.choice(['diesel', 'natural_gas', 'coal', 'gasoline']),
                'quantity': round(random.uniform(10, 10000), 2),
                'unit': random.choice(['liters', 'gallons', 'kg', 'cubic_meters']),
                'date': self.faker.date_between(start_date='-1y', end_date='today'),
                'facility_id': f"FAC-{random.randint(100, 999)}",
                'region': random.choice(['US', 'EU', 'ASIA'])
            }
            records.append(record)
        return records

    def generate_scope3_data(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """Generate Scope 3 emissions test data."""
        categories = [
            'purchased_goods', 'capital_goods', 'fuel_energy',
            'upstream_transport', 'waste', 'business_travel',
            'employee_commuting', 'downstream_transport'
        ]

        records = []
        for _ in range(num_records):
            record = {
                'record_id': self.faker.uuid4(),
                'category': random.choice(categories),
                'activity_data': round(random.uniform(100, 50000), 2),
                'unit': random.choice(['kg', 'tonnes', 'units', 'km']),
                'supplier': self.faker.company(),
                'date': self.faker.date_between(start_date='-1y', end_date='today'),
                'region': random.choice(['US', 'EU', 'ASIA', 'GLOBAL'])
            }
            records.append(record)
        return records

    def generate_erp_data(self, num_records: int = 100) -> List[Dict[str, Any]]:
        """Generate mock ERP system data."""
        records = []
        for _ in range(num_records):
            record = {
                'transaction_id': self.faker.uuid4(),
                'type': random.choice(['purchase_order', 'invoice', 'receipt']),
                'vendor': self.faker.company(),
                'amount': round(random.uniform(1000, 100000), 2),
                'currency': random.choice(['USD', 'EUR', 'GBP']),
                'date': self.faker.date_between(start_date='-1y', end_date='today'),
                'items': [
                    {
                        'sku': f"SKU-{random.randint(1000, 9999)}",
                        'description': self.faker.sentence(nb_words=4),
                        'quantity': random.randint(1, 100),
                        'unit_price': round(random.uniform(10, 1000), 2)
                    }
                    for _ in range(random.randint(1, 5))
                ]
            }
            records.append(record)
        return records


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()


# ================ Performance Testing Fixtures ================

@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = datetime.now()

        def stop(self):
            self.end_time = datetime.now()

        def elapsed_ms(self):
            if self.start_time and self.end_time:
                delta = self.end_time - self.start_time
                return delta.total_seconds() * 1000
            return 0

    return Timer()


# ================ Mock External Services ================

@pytest.fixture
def mock_external_apis():
    """Mock external API services."""
    apis = {}

    # Mock emission factor API
    emission_api = Mock()
    emission_api.get_factor = Mock(return_value=Decimal("2.5"))
    emission_api.list_factors = Mock(return_value=[])
    apis['emission_factors'] = emission_api

    # Mock weather API
    weather_api = Mock()
    weather_api.get_temperature = Mock(return_value=20.5)
    weather_api.get_forecast = Mock(return_value=[])
    apis['weather'] = weather_api

    # Mock regulatory API
    regulatory_api = Mock()
    regulatory_api.check_compliance = Mock(return_value=True)
    regulatory_api.get_requirements = Mock(return_value=[])
    apis['regulatory'] = regulatory_api

    return apis


# ================ Test Helpers ================

def assert_provenance_hash(hash_value: str) -> None:
    """Assert that a value is a valid provenance hash."""
    assert hash_value is not None, "Provenance hash should not be None"
    assert isinstance(hash_value, str), "Provenance hash should be a string"
    assert len(hash_value) == 64, f"Provenance hash should be 64 chars (SHA-256), got {len(hash_value)}"
    assert all(c in '0123456789abcdef' for c in hash_value.lower()), "Invalid hash characters"


def assert_decimal_equal(value1: Decimal, value2: Decimal, places: int = 6) -> None:
    """Assert two decimal values are equal to specified decimal places."""
    assert abs(value1 - value2) < Decimal(10) ** -places, \
        f"Values differ: {value1} != {value2} (tolerance: {places} decimal places)"


def create_mock_pipeline(num_stages: int = 3) -> Mock:
    """Create a mock pipeline with specified number of stages."""
    pipeline = Mock()
    pipeline.stages = []
    for i in range(num_stages):
        stage = Mock()
        stage.name = f"stage_{i}"
        stage.execute = Mock(return_value={"status": "success"})
        pipeline.stages.append(stage)
    pipeline.execute = Mock(return_value={"all_stages": "success"})
    return pipeline


# ================ Cleanup and Teardown ================

@pytest.fixture(autouse=True)
def cleanup_test_files(request):
    """Cleanup test files after each test."""
    test_files = []

    def register_file(filepath):
        test_files.append(filepath)

    request.addfinalizer(lambda: [
        os.unlink(f) for f in test_files if os.path.exists(f)
    ])

    return register_file


# ================ Test Markers ================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "compliance: Compliance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "critical: Critical functionality tests")