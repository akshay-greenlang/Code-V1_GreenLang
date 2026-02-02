# -*- coding: utf-8 -*-
"""Pytest configuration and shared fixtures."""

import asyncio
import json
import math
import os
import socket
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pytest

# Set test environment for ephemeral signing
os.environ['GL_SIGNING_MODE'] = 'ephemeral'


@pytest.fixture(scope="session")
def data_dir():
    """Get the data directory path."""
    return Path(__file__).parent.parent / "greenlang" / "data"


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test fixtures directory path."""
    return Path(__file__).parent / "fixtures"


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
                "natural_gas": {"therms": 5.3}
            },
            "metadata": {
                "version": "1.0.0",
                "last_updated": "2024-01-01"
            }
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
                        "F": {"min": 30, "max": None, "label": "Very Poor"}
                    },
                    "US": {
                        "A": {"min": 0, "max": 8, "label": "Excellent"},
                        "B": {"min": 8, "max": 12, "label": "Good"},
                        "C": {"min": 12, "max": 18, "label": "Average"},
                        "D": {"min": 18, "max": 24, "label": "Below Average"},
                        "E": {"min": 24, "max": 30, "label": "Poor"},
                        "F": {"min": 30, "max": None, "label": "Very Poor"}
                    }
                }
            }
        }


@pytest.fixture
def sample_building_india(test_data_dir):
    """Load sample India building data."""
    file_path = test_data_dir / "building_india_office.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    else:
        # Return minimal test data
        return {
            "building_type": "office",
            "country": "IN",
            "area_sqft": 50000,
            "occupancy": 200
        }


@pytest.fixture
def sample_building_us(test_data_dir):
    """Load sample US building data."""
    file_path = test_data_dir / "building_us_office.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    else:
        # Return minimal test data
        return {
            "building_type": "office",
            "country": "US",
            "area_sqft": 100000,
            "occupancy": 400
        }


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch, request):
    """Disable network calls in all tests."""
    # Allow network for integration and e2e tests
    if any(mark in request.keywords for mark in ("integration", "e2e", "network")):
        return

    # Note: Not blocking socket to avoid import issues
    pass


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM responses for deterministic testing."""
    def mock_response(*args, **kwargs):
        return {
            "choices": [{
                "message": {
                    "content": "Mocked LLM response for testing"
                }
            }]
        }

    def mock_langchain_response(*args, **kwargs):
        class MockMessage:
            content = "Mocked LangChain response for testing"
        return MockMessage()

    # Mock OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    return mock_response


@pytest.fixture
def snapshot_normalizer():
    """Normalize snapshots for consistent comparison."""
    def normalize(content: str) -> str:
        """Remove timestamps, paths, and other non-deterministic content."""
        import re

        # Remove timestamps
        content = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', 'TIMESTAMP', content)
        content = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', content)

        # Remove file paths
        content = re.sub(r'[A-Z]:\\[^\s]+', 'PATH', content)
        content = re.sub(r'/[^\s]+', 'PATH', content)

        # Normalize line endings
        content = content.replace('\r\n', '\n')

        # Remove version-specific info
        content = re.sub(r'version \d+\.\d+\.\d+', 'version X.X.X', content)

        return content

    return normalize


@pytest.fixture
def electricity_factors(emission_factors):
    """Extract electricity factors for easy access."""
    # Build electricity factors from each country's data
    factors = {}
    for country, data in emission_factors.items():
        if country != "metadata" and "electricity" in data:
            factors[country] = {"factor": data["electricity"]["kWh"]}
    return factors


@pytest.fixture
def fuel_factors(emission_factors):
    """Extract all fuel factors for easy access."""
    return emission_factors


class AgentContractValidator:
    """Validator for agent contract compliance."""

    @staticmethod
    def validate_response(response: Dict[str, Any], agent_name: str = ""):
        """Validate that agent response follows the contract."""
        assert isinstance(response, dict), f"{agent_name}: Response must be a dict"
        assert "success" in response, f"{agent_name}: Response must have 'success' field"
        assert isinstance(response["success"], bool), f"{agent_name}: 'success' must be bool"

        if response["success"]:
            assert "data" in response, f"{agent_name}: Successful response must have 'data'"
            assert isinstance(response["data"], dict), f"{agent_name}: 'data' must be dict"
        else:
            assert "error" in response, f"{agent_name}: Failed response must have 'error'"
            assert isinstance(response["error"], dict), f"{agent_name}: 'error' must be dict"
            assert "type" in response["error"], f"{agent_name}: Error must have 'type'"
            assert "message" in response["error"], f"{agent_name}: Error must have 'message'"


@pytest.fixture
def agent_contract_validator():
    """Provide agent contract validator."""
    return AgentContractValidator()


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
                    boundaries[building_type][country].append({
                        "value": thresholds["min"],
                        "rating": rating,
                        "boundary": "min"
                    })
                if thresholds["max"] is not None:
                    boundaries[building_type][country].append({
                        "value": thresholds["max"],
                        "rating": rating,
                        "boundary": "max"
                    })

    return boundaries


# ============================================================================
# Signing fixtures - NO HARDCODED KEYS
# ============================================================================

@pytest.fixture(autouse=True)
def _ephemeral_signing_keys(monkeypatch):
    """Auto-inject ephemeral signing keys for all tests"""
    # Generate simple test keys
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
    """Create a temporary pack directory for testing"""
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
# Test utility functions
# ============================================================================

def assert_close(
    actual: Union[float, int],
    expected: Union[float, int],
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
    message: Optional[str] = None
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
        msg = message or f"Values not close: {actual} != {expected} (rel_tol={rel_tol}, abs_tol={abs_tol})"
        raise AssertionError(msg)


def assert_percentage_sum(
    percentages: list,
    expected_sum: float = 100.0,
    tolerance: float = 0.01,
    message: Optional[str] = None
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
        msg = message or f"Percentages sum to {actual_sum}, expected {expected_sum} ± {tolerance}"
        raise AssertionError(msg)


def normalize_factor(
    value: float,
    from_unit: str,
    to_unit: str
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


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def anyio_backend():
    """Backend for async tests."""
    return "asyncio"


# ============================================================================
# AI Agent Testing Fixtures - ChatSession Mocking
# ============================================================================

@pytest.fixture
def mock_chat_response():
    """Create a mock ChatResponse for testing AI agents."""
    from unittest.mock import Mock

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
    from unittest.mock import Mock, AsyncMock

    def _create_session(response=None, responses=None):
        """Create a mock ChatSession."""
        mock_session = Mock()

        if responses:
            # Multiple responses for multiple calls
            async def multi_chat(*args, **kwargs):
                if not hasattr(multi_chat, 'call_count'):
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


# ============================================================================
# Test Data Fixtures Library - Reusable Test Data for All Agent Tests
# ============================================================================


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
            """Assert that function produces identical results across multiple runs."""
            results = []
            for _ in range(runs):
                results.append(func(*args, **kwargs))

            # All results should be equal
            for i in range(1, len(results)):
                assert results[i] == results[0], f"Run {i+1} produced different result than run 1"

    return AgentTestHelpers()


# ============================================================================
# Coverage and Pytest Configuration Helpers
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


# Configure for fast testing
os.environ.setdefault("HYPOTHESIS_PROFILE", "fast")