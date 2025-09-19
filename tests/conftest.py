"""Pytest configuration and shared fixtures."""

import asyncio
import json
import math
import os
import socket
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pytest
from hypothesis import settings, Verbosity

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
    
    with open(factors_file) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def benchmarks_data(data_dir):
    """Load actual benchmarks from the data file."""
    benchmarks_file = data_dir / "global_benchmarks.json"
    if not benchmarks_file.exists():
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
    
    with open(benchmarks_file) as f:
        return json.load(f)


@pytest.fixture
def sample_building_india(test_data_dir):
    """Load sample India building data."""
    with open(test_data_dir / "building_india_office.json") as f:
        return json.load(f)


@pytest.fixture
def sample_building_us(test_data_dir):
    """Load sample US building data."""
    with open(test_data_dir / "building_us_office.json") as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    """Disable network calls in all tests."""
    def mock_network_call(*args, **kwargs):
        raise RuntimeError("Network calls are disabled in tests")
    
    # Disable socket connections entirely
    def guard(*args, **kwargs):
        raise RuntimeError("Socket connections are disabled in tests")
    monkeypatch.setattr(socket, "socket", guard)
    
    # Disable common network libraries
    monkeypatch.setattr("urllib.request.urlopen", mock_network_call)
    monkeypatch.setattr("urllib.request.Request", mock_network_call)
    
    # Disable httpx if it's imported
    try:
        import httpx
        monkeypatch.setattr("httpx.Client.request", mock_network_call)
        monkeypatch.setattr("httpx.AsyncClient.request", mock_network_call)
        monkeypatch.setattr("httpx.Client.get", mock_network_call)
        monkeypatch.setattr("httpx.Client.post", mock_network_call)
    except ImportError:
        pass
    
    # Disable requests if it's imported
    try:
        import requests
        monkeypatch.setattr("requests.get", mock_network_call)
        monkeypatch.setattr("requests.post", mock_network_call)
        monkeypatch.setattr("requests.put", mock_network_call)
        monkeypatch.setattr("requests.delete", mock_network_call)
        monkeypatch.setattr("requests.Session.request", mock_network_call)
    except ImportError:
        pass
    
    # Disable OpenAI calls (both old and new API)
    try:
        import openai
        monkeypatch.setattr("openai.ChatCompletion.create", mock_network_call)
        monkeypatch.setattr("openai.Completion.create", mock_network_call)
        # New OpenAI client
        monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: None)
        monkeypatch.setattr("openai.Client", lambda *args, **kwargs: None)
    except (ImportError, AttributeError):
        pass
    
    # Disable LangChain
    try:
        monkeypatch.setattr("langchain.llms.openai.OpenAI", lambda *args, **kwargs: None)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda *args, **kwargs: None)
        monkeypatch.setattr("langchain_openai.OpenAI", lambda *args, **kwargs: None)
    except (ImportError, AttributeError):
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
    try:
        import openai
        monkeypatch.setattr("openai.ChatCompletion.create", mock_response)
        # New OpenAI client
        class MockOpenAI:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        return mock_response()
        monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: MockOpenAI())
    except (ImportError, AttributeError):
        pass
    
    # Mock LangChain
    try:
        from langchain_openai import ChatOpenAI
        monkeypatch.setattr("langchain_openai.ChatOpenAI.invoke", mock_langchain_response)
        monkeypatch.setattr("langchain_openai.OpenAI.invoke", mock_langchain_response)
    except ImportError:
        pass


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


# Configure Hypothesis for fast, deterministic tests
settings.register_profile(
    "ci",
    max_examples=10,  # Reduced for CI speed
    deadline=5000,  # 5 seconds deadline per test
    suppress_health_check=[],
    verbosity=Verbosity.normal,
    derandomize=True,  # Deterministic test order
    print_blob=True,
)

settings.register_profile(
    "dev",
    max_examples=100,  # More thorough for development
    deadline=10000,  # 10 seconds deadline
    verbosity=Verbosity.verbose,
)

settings.register_profile(
    "fast",
    max_examples=5,  # Minimal for quick checks
    deadline=1000,  # 1 second deadline
    verbosity=Verbosity.quiet,
    derandomize=True,
)

# Load profile from environment or default to 'fast' for <90s guarantee
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "fast"))


# ============================================================================
# Signing fixtures - NO HARDCODED KEYS
# ============================================================================

@pytest.fixture
def ephemeral_signer():
    """Provide ephemeral signer for tests - generates new keys each time"""
    from greenlang.security.signing import EphemeralKeypairSigner
    return EphemeralKeypairSigner()


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
    percentages: list[float],
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
def signed_pack(temp_pack_dir, ephemeral_signer):
    """Create a signed pack for testing with ephemeral keys"""
    from greenlang.security.signing import sign_artifact

    # Sign the pack manifest
    signature = sign_artifact(temp_pack_dir / "pack.yaml", signer=ephemeral_signer)

    # Save signature
    sig_path = temp_pack_dir / "pack.sig"
    with open(sig_path, 'w') as f:
        json.dump(signature, f)

    return temp_pack_dir, signature, ephemeral_signer


@pytest.fixture
def mock_sigstore_env(monkeypatch):
    """Mock environment for Sigstore testing"""
    monkeypatch.setenv('CI', 'true')
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'test/repo')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'test-workflow')
    monkeypatch.setenv('GL_SIGSTORE_STAGING', '1')  # Use staging for tests
    yield


@pytest.fixture
def disable_signing(monkeypatch):
    """Disable signing for tests that don't need it"""
    monkeypatch.setenv('GL_SIGNING_MODE', 'disabled')
    yield


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def anyio_backend():
    """Backend for async tests."""
    return "asyncio"


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, request):
    """Block network access by default in unit tests."""
    # Allow network for integration and e2e tests
    if any(mark in request.keywords for mark in ("integration", "e2e", "network")):
        return

    # Block socket connections for unit tests
    def guard(*args, **kwargs):
        raise RuntimeError("Network access disabled in unit tests. Use @pytest.mark.integration to allow network.")

    monkeypatch.setattr(socket, "create_connection", guard)