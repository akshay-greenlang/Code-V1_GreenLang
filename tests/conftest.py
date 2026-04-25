# -*- coding: utf-8 -*-
"""
Pytest configuration and shared fixtures for GreenLang test suites.

This module consolidates all shared test fixtures, mocks, utilities,
and configuration from the previous conftest.py, conftest_enhanced.py,
and conftest_v2.py into a single auto-loaded conftest.

Sections:
    - Core Test Fixtures
    - Database Fixtures
    - Authentication Fixtures
    - Pipeline Fixtures
    - Agent Fixtures
    - Provenance Fixtures
    - Security Fixtures
    - AI Agent Testing Fixtures
    - Test Data Library
    - Performance Testing
    - Mock External Services
    - Test Helpers and Utilities
    - Coverage and Configuration
"""

import asyncio
import hashlib
import json
import math
import os
import random
import string
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Set test environment for ephemeral signing
os.environ["GL_SIGNING_MODE"] = "ephemeral"

# Configure for fast testing
os.environ.setdefault("HYPOTHESIS_PROFILE", "fast")


# ============================================================================
# Coverage and Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom pytest markers (merged from all conftest files)."""
    # From original conftest_enhanced.py
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "compliance: Compliance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "critical: Critical functionality tests")
    # From original conftest_v2.py
    config.addinivalue_line("markers", "v2: V2 refactored agent tests")
    config.addinivalue_line("markers", "infrastructure: Infrastructure component tests")
    config.addinivalue_line("markers", "services: Shared service tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    # Additional useful markers
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "network: Tests requiring network access")
    config.addinivalue_line("markers", "requires_postgres: Tests that need a real Postgres instance (skipped if no DSN)")


# ============================================================================
# Core Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def data_dir():
    """Get the data directory path."""
    return Path(__file__).parent.parent / "greenlang" / "data"


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def anyio_backend():
    """Backend for async tests."""
    return "asyncio"


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch, request):
    """Disable network calls in all tests."""
    # Allow network for integration and e2e tests
    if any(mark in request.keywords for mark in ("integration", "e2e", "network")):
        return

    # Note: Not blocking socket to avoid import issues
    pass


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return {
        "environment": "test",
        "debug": True,
        "database": {
            "url": "sqlite:///:memory:",
            "pool_size": 5,
            "echo": False,
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "cors_origins": ["http://localhost:3000"],
        },
        "security": {
            "secret_key": "test-secret-key",
            "algorithm": "HS256",
            "token_expire_minutes": 30,
        },
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 60,
        },
    }


# ============================================================================
# Signing Fixtures - NO HARDCODED KEYS
# ============================================================================


@pytest.fixture(autouse=True)
def _ephemeral_signing_keys(monkeypatch):
    """Auto-inject ephemeral signing keys for all tests."""
    priv_key = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7W8jYPqDHw6Ev
qNfXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
-----END PRIVATE KEY-----"""

    pub_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAu1vI2D6gx8OhL6jX1111
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
-----END PUBLIC KEY-----"""

    monkeypatch.setenv("GL_SIGNING_PRIVATE_KEY_PEM", priv_key)
    monkeypatch.setenv("GL_SIGNING_PUBLIC_KEY_PEM", pub_key)


@pytest.fixture
def temp_pack_dir(tmp_path):
    """Create a temporary pack directory for testing."""
    pack_dir = tmp_path / "test-pack"
    pack_dir.mkdir()

    # Create minimal pack.yaml
    manifest = pack_dir / "pack.yaml"
    manifest.write_text("""
name: test-pack
version: 1.0.0
kind: pack
license: MIT
contents:
  pipelines:
    - pipeline.yaml
""")

    # Create dummy pipeline
    pipeline = pack_dir / "pipeline.yaml"
    pipeline.write_text("""
version: "1.0"
name: test-pipeline
steps: []
""")

    return pack_dir


# ============================================================================
# Database Fixtures (from enhanced)
# ============================================================================


@pytest.fixture
def mock_db_session():
    """Create mock database session with CRUD operations."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.query = Mock(
        return_value=Mock(
            filter=Mock(
                return_value=Mock(
                    first=Mock(return_value=None),
                    all=Mock(return_value=[]),
                )
            )
        )
    )
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


# ============================================================================
# Authentication Fixtures (from enhanced)
# ============================================================================


@pytest.fixture
def mock_user():
    """Create mock user object with auth fields."""
    return {
        "id": "user-123",
        "username": "test_user",
        "email": "test@example.com",
        "roles": ["user", "analyst"],
        "permissions": ["read", "write", "calculate"],
        "is_active": True,
        "created_at": datetime(2025, 1, 15, 10, 30, 0),
    }


@pytest.fixture
def mock_jwt_token():
    """Create mock JWT token string."""
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


# ============================================================================
# Pipeline Fixtures (from enhanced)
# ============================================================================


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
        "request_id": "req-a1b2c3d4",
        "user_id": "user-123",
        "timestamp": datetime(2025, 6, 15, 12, 0, 0),
        "metadata": {},
        "checkpoints": [],
        "errors": [],
    }


# ============================================================================
# Agent Fixtures (from enhanced)
# ============================================================================


@pytest.fixture
def base_agent_config():
    """Create base agent configuration dict."""
    return {
        "name": "test_agent",
        "version": "1.0.0",
        "type": "calculator",
        "enabled": True,
        "timeout": 30,
        "retry_count": 3,
        "retry_delay": 1,
    }


@pytest.fixture
def mock_emission_factors():
    """Create mock emission factor database (Decimal-based, keyed by tuple)."""
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
        # Transport factors (kg CO2e per tonne-km)
        ("truck", "US", "freight"): Decimal("0.162"),
        ("rail", "US", "freight"): Decimal("0.021"),
        ("ship", "GLOBAL", "freight"): Decimal("0.012"),
        ("air", "GLOBAL", "freight"): Decimal("0.602"),
    }


@pytest.fixture
def sample_shipment_data():
    """Create sample shipment data for testing (deterministic, no faker)."""
    return {
        "shipment_id": "b7e4c8a1-3f2d-4e6b-9a1c-5d8f7e2b3a4c",
        "product_category": "steel",
        "weight_tonnes": 42.75,
        "origin_country": "DE",
        "destination_country": "US",
        "transport_mode": "ship",
        "distance_km": 7200.0,
        "import_date": "2025-03-15",
        "supplier_name": "Acme Steel GmbH",
        "hs_code": "7208.51",
    }


# ============================================================================
# Provenance Fixtures (from enhanced)
# ============================================================================


@pytest.fixture
def mock_provenance_tracker():
    """Create mock provenance tracker with hash tracking."""
    tracker = Mock()
    tracker.track_input = Mock(return_value="input-hash-123")
    tracker.track_calculation = Mock(return_value="calc-hash-456")
    tracker.track_output = Mock(return_value="output-hash-789")
    tracker.generate_hash = Mock(return_value="a" * 64)  # SHA-256 hash
    tracker.get_audit_trail = Mock(return_value=[])
    return tracker


# ============================================================================
# Security Fixtures (from enhanced)
# ============================================================================


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


# ============================================================================
# AI Agent Testing Fixtures - ChatSession Mocking
# ============================================================================


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM responses for deterministic testing."""

    def mock_response(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Mocked LLM response for testing",
                    }
                }
            ]
        }

    def mock_langchain_response(*args, **kwargs):
        class MockMessage:
            content = "Mocked LangChain response for testing"

        return MockMessage()

    # Mock OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    return mock_response


@pytest.fixture
def mock_chat_response():
    """Create a mock ChatResponse for testing AI agents."""

    def _create_response(
        text="Mock AI response for testing",
        tool_calls=None,
        cost_usd=0.01,
        prompt_tokens=100,
        completion_tokens=50,
    ):
        mock_response = Mock()
        mock_response.text = text
        mock_response.tool_calls = tool_calls or []
        return mock_response

    return _create_response


@pytest.fixture
def mock_chat_session(mock_chat_response):
    """Create a mock ChatSession with async support for testing AI agents."""

    def _create_session(response=None, responses=None):
        """Create a mock ChatSession."""
        mock_session = Mock()

        if responses:
            # Multiple responses for multiple calls
            async def multi_chat(*args, **kwargs):
                if not hasattr(multi_chat, "call_count"):
                    multi_chat.call_count = 0
                idx = multi_chat.call_count
                multi_chat.call_count += 1
                if idx < len(responses):
                    return responses[idx]
                return responses[-1]  # Return last response if exceeded

            mock_session.chat = multi_chat
        else:
            # Single response (use provided or default)
            if response is None:
                response = mock_chat_response()
            mock_session.chat = AsyncMock(return_value=response)

        # Track calls for validation
        mock_session.call_count = 0

        return mock_session

    return _create_session


@pytest.fixture
def snapshot_normalizer():
    """Normalize snapshots for consistent comparison."""
    import re

    def normalize(content: str) -> str:
        """Remove timestamps, paths, and other non-deterministic content."""
        # Remove timestamps
        content = re.sub(
            r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}", "TIMESTAMP", content
        )
        content = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", content)

        # Remove file paths
        content = re.sub(r"[A-Z]:\\[^\s]+", "PATH", content)
        content = re.sub(r"/[^\s]+", "PATH", content)

        # Normalize line endings
        content = content.replace("\r\n", "\n")

        # Remove version-specific info
        content = re.sub(r"version \d+\.\d+\.\d+", "version X.X.X", content)

        return content

    return normalize


# ============================================================================
# Test Data Library (existing + enhanced)
# ============================================================================


@pytest.fixture(scope="session")
def emission_factors(data_dir):
    """Load actual emission factors from the data file."""
    factors_file = data_dir / "global_emission_factors.json"
    if not factors_file.exists():
        # Fall back to test fixtures if main data doesn't exist
        factors_file = Path(__file__).parent / "fixtures" / "factors_minimal.json"

    if factors_file.exists():
        with open(factors_file) as f:
            return json.load(f)
    else:
        # Return minimal factors for testing
        return {
            "US": {
                "electricity": {"kWh": 0.42},
                "natural_gas": {"therms": 5.3},
            },
            "metadata": {"version": "1.0.0", "last_updated": "2024-01-01"},
        }


@pytest.fixture(scope="session")
def benchmarks_data(data_dir):
    """Load actual benchmarks from the data file."""
    benchmarks_file = data_dir / "global_benchmarks.json"
    if benchmarks_file.exists():
        with open(benchmarks_file) as f:
            return json.load(f)
    else:
        # Create minimal benchmarks for testing
        return {
            "version": "0.0.1",
            "last_updated": "2024-01-01",
            "benchmarks": {
                "office": {
                    "IN": {
                        "A": {"min": 0, "max": 10, "label": "Excellent"},
                        "B": {"min": 10, "max": 15, "label": "Good"},
                        "C": {"min": 15, "max": 20, "label": "Average"},
                        "D": {"min": 20, "max": 25, "label": "Below Average"},
                        "E": {"min": 25, "max": 30, "label": "Poor"},
                        "F": {"min": 30, "max": None, "label": "Very Poor"},
                    },
                    "US": {
                        "A": {"min": 0, "max": 8, "label": "Excellent"},
                        "B": {"min": 8, "max": 12, "label": "Good"},
                        "C": {"min": 12, "max": 18, "label": "Average"},
                        "D": {"min": 18, "max": 24, "label": "Below Average"},
                        "E": {"min": 24, "max": 30, "label": "Poor"},
                        "F": {"min": 30, "max": None, "label": "Very Poor"},
                    },
                }
            },
        }


@pytest.fixture
def sample_building_india(test_data_dir):
    """Load sample India building data."""
    file_path = test_data_dir / "building_india_office.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    else:
        return {
            "building_type": "office",
            "country": "IN",
            "area_sqft": 50000,
            "occupancy": 200,
        }


@pytest.fixture
def sample_building_us(test_data_dir):
    """Load sample US building data."""
    file_path = test_data_dir / "building_us_office.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    else:
        return {
            "building_type": "office",
            "country": "US",
            "area_sqft": 100000,
            "occupancy": 400,
        }


@pytest.fixture
def electricity_factors(emission_factors):
    """Extract electricity factors for easy access."""
    factors = {}
    for country, data in emission_factors.items():
        if country != "metadata" and "electricity" in data:
            factors[country] = {"factor": data["electricity"]["kWh"]}
    return factors


@pytest.fixture
def fuel_factors(emission_factors):
    """Extract all fuel factors for easy access."""
    return emission_factors


@pytest.fixture
def benchmark_boundaries(benchmarks_data):
    """Extract benchmark boundaries for testing."""
    boundaries = {}
    benchmarks = benchmarks_data.get("benchmarks", {})

    for building_type, countries in benchmarks.items():
        boundaries[building_type] = {}
        for country, ratings in countries.items():
            boundaries[building_type][country] = []
            for rating, thresholds in ratings.items():
                if thresholds["min"] is not None:
                    boundaries[building_type][country].append(
                        {
                            "value": thresholds["min"],
                            "rating": rating,
                            "boundary": "min",
                        }
                    )
                if thresholds["max"] is not None:
                    boundaries[building_type][country].append(
                        {
                            "value": thresholds["max"],
                            "rating": rating,
                            "boundary": "max",
                        }
                    )

    return boundaries


@pytest.fixture
def sample_fuel_payload():
    """Reusable fuel agent test data."""
    return {
        "fuel_type": "natural_gas",
        "amount": 1000.0,
        "unit": "therms",
        "country": "US",
    }


@pytest.fixture
def sample_carbon_payload():
    """Reusable carbon agent test data."""
    return {
        "emissions_by_source": {
            "electricity": 15000.0,
            "natural_gas": 8500.0,
            "diesel": 3200.0,
        },
        "building_area_sqft": 50000.0,
        "occupancy": 200,
    }


@pytest.fixture
def sample_grid_payload():
    """Reusable grid factor agent test data."""
    return {
        "region": "US-CA",
        "country": "US",
        "year": 2024,
        "hour": 12,
    }


@pytest.fixture
def sample_data():
    """Sample data for V2 tests (from conftest_v2)."""
    return [{"id": i, "value": i * 1.5} for i in range(100)]


# ============================================================================
# Test Data Generator (from enhanced, faker removed - deterministic data)
# ============================================================================


class TestDataGenerator:
    """Generate deterministic test data for various scenarios.

    All data is generated using simple deterministic patterns
    instead of faker to avoid external dependencies and ensure
    reproducibility across environments.
    """

    _FUEL_TYPES = ["diesel", "natural_gas", "coal", "gasoline"]
    _UNITS_FUEL = ["liters", "gallons", "kg", "cubic_meters"]
    _REGIONS = ["US", "EU", "ASIA"]
    _SCOPE3_CATEGORIES = [
        "purchased_goods",
        "capital_goods",
        "fuel_energy",
        "upstream_transport",
        "waste",
        "business_travel",
        "employee_commuting",
        "downstream_transport",
    ]
    _SCOPE3_UNITS = ["kg", "tonnes", "units", "km"]
    _CURRENCIES = ["USD", "EUR", "GBP"]
    _TXN_TYPES = ["purchase_order", "invoice", "receipt"]
    _SUPPLIERS = [
        "Acme Corp",
        "Global Materials Ltd",
        "EcoSteel GmbH",
        "Pacific Resources",
        "Northern Energy Co",
        "Atlas Manufacturing",
        "Green Supply Inc",
        "Continental Metals",
        "Pinnacle Industries",
        "Summit Materials",
    ]

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        self._rng = random.Random(seed)

    def _uuid(self, index: int) -> str:
        """Generate a deterministic UUID-like string from an index."""
        h = hashlib.md5(f"record-{index}".encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    def _date_str(self, index: int) -> str:
        """Generate a deterministic date string within the last year."""
        base = datetime(2025, 1, 1)
        offset_days = index % 365
        dt = base + timedelta(days=offset_days)
        return dt.strftime("%Y-%m-%d")

    def generate_fuel_consumption_data(
        self, num_records: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate test fuel consumption data."""
        records = []
        for i in range(num_records):
            record = {
                "record_id": self._uuid(i),
                "fuel_type": self._FUEL_TYPES[i % len(self._FUEL_TYPES)],
                "quantity": round(10.0 + (i * 99.9), 2),
                "unit": self._UNITS_FUEL[i % len(self._UNITS_FUEL)],
                "date": self._date_str(i),
                "facility_id": f"FAC-{100 + (i % 900)}",
                "region": self._REGIONS[i % len(self._REGIONS)],
            }
            records.append(record)
        return records

    def generate_scope3_data(
        self, num_records: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate Scope 3 emissions test data."""
        records = []
        for i in range(num_records):
            record = {
                "record_id": self._uuid(1000 + i),
                "category": self._SCOPE3_CATEGORIES[
                    i % len(self._SCOPE3_CATEGORIES)
                ],
                "activity_data": round(100.0 + (i * 499.0), 2),
                "unit": self._SCOPE3_UNITS[i % len(self._SCOPE3_UNITS)],
                "supplier": self._SUPPLIERS[i % len(self._SUPPLIERS)],
                "date": self._date_str(i),
                "region": self._REGIONS[i % len(self._REGIONS)]
                if i % 4 != 3
                else "GLOBAL",
            }
            records.append(record)
        return records

    def generate_erp_data(
        self, num_records: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate mock ERP system data."""
        _descriptions = [
            "Raw steel plate",
            "Aluminum ingot",
            "Cement bags",
            "Plastic pellets",
            "Copper wire",
        ]
        records = []
        for i in range(num_records):
            num_items = 1 + (i % 5)
            items = []
            for j in range(num_items):
                items.append(
                    {
                        "sku": f"SKU-{1000 + ((i * 5 + j) % 9000)}",
                        "description": _descriptions[j % len(_descriptions)],
                        "quantity": 1 + (j * 10),
                        "unit_price": round(10.0 + (j * 95.5), 2),
                    }
                )
            record = {
                "transaction_id": self._uuid(2000 + i),
                "type": self._TXN_TYPES[i % len(self._TXN_TYPES)],
                "vendor": self._SUPPLIERS[i % len(self._SUPPLIERS)],
                "amount": round(1000.0 + (i * 990.0), 2),
                "currency": self._CURRENCIES[i % len(self._CURRENCIES)],
                "date": self._date_str(i),
                "items": items,
            }
            records.append(record)
        return records


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()


# ============================================================================
# Performance Testing (from enhanced)
# ============================================================================


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""

    class Timer:
        """Simple elapsed-time timer for benchmarking test operations."""

        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            """Record start time."""
            self.start_time = datetime.now()

        def stop(self):
            """Record end time."""
            self.end_time = datetime.now()

        def elapsed_ms(self) -> float:
            """Return elapsed time in milliseconds."""
            if self.start_time and self.end_time:
                delta = self.end_time - self.start_time
                return delta.total_seconds() * 1000
            return 0.0

    return Timer()


# ============================================================================
# Mock External Services (from enhanced)
# ============================================================================


@pytest.fixture
def mock_external_apis():
    """Mock external API services."""
    apis = {}

    # Mock emission factor API
    emission_api = Mock()
    emission_api.get_factor = Mock(return_value=Decimal("2.5"))
    emission_api.list_factors = Mock(return_value=[])
    apis["emission_factors"] = emission_api

    # Mock weather API
    weather_api = Mock()
    weather_api.get_temperature = Mock(return_value=20.5)
    weather_api.get_forecast = Mock(return_value=[])
    apis["weather"] = weather_api

    # Mock regulatory API
    regulatory_api = Mock()
    regulatory_api.check_compliance = Mock(return_value=True)
    regulatory_api.get_requirements = Mock(return_value=[])
    apis["regulatory"] = regulatory_api

    return apis


# ============================================================================
# Test Helpers and Utilities (existing + enhanced)
# ============================================================================


class AgentContractValidator:
    """Validator for agent contract compliance."""

    @staticmethod
    def validate_response(response: Dict[str, Any], agent_name: str = ""):
        """Validate that agent response follows the contract."""
        assert isinstance(response, dict), f"{agent_name}: Response must be a dict"
        assert "success" in response, f"{agent_name}: Response must have 'success' field"
        assert isinstance(
            response["success"], bool
        ), f"{agent_name}: 'success' must be bool"

        if response["success"]:
            assert (
                "data" in response
            ), f"{agent_name}: Successful response must have 'data'"
            assert isinstance(
                response["data"], dict
            ), f"{agent_name}: 'data' must be dict"
        else:
            assert (
                "error" in response
            ), f"{agent_name}: Failed response must have 'error'"
            assert isinstance(
                response["error"], dict
            ), f"{agent_name}: 'error' must be dict"
            assert (
                "type" in response["error"]
            ), f"{agent_name}: Error must have 'type'"
            assert (
                "message" in response["error"]
            ), f"{agent_name}: Error must have 'message'"


@pytest.fixture
def agent_contract_validator():
    """Provide agent contract validator."""
    return AgentContractValidator()


@pytest.fixture
def agent_test_helpers():
    """Helper functions for agent testing."""

    class AgentTestHelpers:
        @staticmethod
        def assert_successful_response(result):
            """Assert that agent returned successful response."""
            assert result is not None
            assert result.success is True
            assert result.data is not None
            assert result.error is None

        @staticmethod
        def assert_failed_response(result, error_type=None):
            """Assert that agent returned failed response."""
            assert result is not None
            assert result.success is False
            assert result.error is not None
            if error_type:
                assert error_type in result.error.lower()

        @staticmethod
        def assert_deterministic(func, *args, runs=5, **kwargs):
            """Assert that function produces identical results across runs."""
            results = []
            for _ in range(runs):
                results.append(func(*args, **kwargs))

            for i in range(1, len(results)):
                assert (
                    results[i] == results[0]
                ), f"Run {i + 1} produced different result than run 1"

    return AgentTestHelpers()


def assert_close(
    actual: Union[float, int],
    expected: Union[float, int],
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
    message: Optional[str] = None,
) -> None:
    """
    Assert that two numbers are close within tolerance.

    Args:
        actual: The actual value
        expected: The expected value
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
        message: Optional error message
    """
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        msg = message or (
            f"Values not close: {actual} != {expected} "
            f"(rel_tol={rel_tol}, abs_tol={abs_tol})"
        )
        raise AssertionError(msg)


def assert_percentage_sum(
    percentages: list,
    expected_sum: float = 100.0,
    tolerance: float = 0.01,
    message: Optional[str] = None,
) -> None:
    """
    Assert that percentages sum to expected value within tolerance.

    Args:
        percentages: List of percentage values
        expected_sum: Expected sum (default 100.0)
        tolerance: Tolerance for sum
        message: Optional error message
    """
    actual_sum = sum(percentages)
    if abs(actual_sum - expected_sum) > tolerance:
        msg = message or (
            f"Percentages sum to {actual_sum}, "
            f"expected {expected_sum} +/- {tolerance}"
        )
        raise AssertionError(msg)


def assert_provenance_hash(hash_value: str) -> None:
    """Assert that a value is a valid SHA-256 provenance hash."""
    assert hash_value is not None, "Provenance hash should not be None"
    assert isinstance(hash_value, str), "Provenance hash should be a string"
    assert len(hash_value) == 64, (
        f"Provenance hash should be 64 chars (SHA-256), got {len(hash_value)}"
    )
    assert all(
        c in "0123456789abcdef" for c in hash_value.lower()
    ), "Invalid hash characters"


def assert_decimal_equal(
    value1: Decimal, value2: Decimal, places: int = 6
) -> None:
    """Assert two decimal values are equal to specified decimal places."""
    assert abs(value1 - value2) < Decimal(10) ** -places, (
        f"Values differ: {value1} != {value2} (tolerance: {places} decimal places)"
    )


def normalize_factor(
    value: float, from_unit: str, to_unit: str
) -> float:
    """
    Normalize emission factors between units.

    Args:
        value: The value to normalize
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Normalized value
    """
    conversions = {
        ("kWh", "MWh"): 0.001,
        ("MWh", "kWh"): 1000,
        ("therms", "MMBtu"): 0.1,
        ("MMBtu", "therms"): 10,
        ("m3", "ft3"): 35.3147,
        ("ft3", "m3"): 0.0283168,
        ("sqft", "sqm"): 0.092903,
        ("sqm", "sqft"): 10.7639,
    }

    key = (from_unit, to_unit)
    if key in conversions:
        return value * conversions[key]
    elif from_unit == to_unit:
        return value
    else:
        raise ValueError(f"Unknown conversion: {from_unit} to {to_unit}")


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


# ============================================================================
# Cleanup and Teardown (from enhanced)
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_files(request):
    """Cleanup test files after each test.

    Usage in tests:
        def test_something(cleanup_test_files):
            path = "/tmp/test_output.json"
            cleanup_test_files(path)  # Will be deleted after test
            # ... test logic ...
    """
    test_files = []

    def register_file(filepath):
        test_files.append(filepath)

    request.addfinalizer(
        lambda: [os.unlink(f) for f in test_files if os.path.exists(f)]
    )

    return register_file


# ============================================================================
# Coverage Configuration
# ============================================================================


@pytest.fixture
def coverage_config():
    """Provide coverage configuration for tests."""
    return {
        "branch": True,
        "source": ["greenlang"],
        "omit": [
            "*/tests/*",
            "*/__main__.py",
            "*/conftest.py",
        ],
        "fail_under": 85,
    }
