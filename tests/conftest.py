"""Pytest configuration and shared fixtures."""

import json
import os
import socket
from pathlib import Path
from typing import Dict, Any
import pytest
from hypothesis import settings, Verbosity


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