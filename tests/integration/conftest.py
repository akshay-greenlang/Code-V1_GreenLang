"""
Integration test configuration and fixtures for GreenLang.
"""
import os
import sys
import json
import socket
import random
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Generator
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

import pytest
import numpy as np
from click.testing import CliRunner

# Removed sys.path manipulation - using installed package


# ==================== Network Guard ====================
class NetworkBlocker:
    """Block all network calls to ensure tests run offline."""
    
    def __init__(self):
        self._original_socket = socket.socket
        self._original_create_connection = socket.create_connection
        
    def enable(self):
        """Enable network blocking."""
        def blocked(*args, **kwargs):
            raise RuntimeError(
                "Network access blocked in tests. "
                "All data must come from fixtures or mocks."
            )
        socket.socket = blocked
        socket.create_connection = blocked
        
    def disable(self):
        """Restore network access."""
        socket.socket = self._original_socket
        socket.create_connection = self._original_create_connection


# ==================== Seed Management ====================
def set_all_seeds(seed: int = 1337):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Add any framework-specific seeds here
    os.environ['GREENLANG_SEED'] = str(seed)


# ==================== Pytest Configuration ====================
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "timeout: set timeout for test execution"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


# ==================== Session Fixtures ====================
@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Block network access for all integration tests."""
    blocker = NetworkBlocker()
    blocker.enable()
    yield
    blocker.disable()


@pytest.fixture(scope="session", autouse=True)
def seed_randomness():
    """Seed all random number generators."""
    set_all_seeds(1337)
    yield


# ==================== Test Fixtures ====================
@pytest.fixture
def dataset():
    """Load emission factors and benchmarks dataset."""
    dataset_path = Path(__file__).parent.parent.parent / "data" / "global_emission_factors.json"
    if dataset_path.exists():
        with open(dataset_path) as f:
            return json.load(f)
    
    # Return mock dataset if real one doesn't exist
    return {
        "version": "0.0.1",
        "last_updated": "2024-01-01",
        "source": "Test Dataset",
        "emission_factors": {
            "IN": {
                "electricity": {"value": 0.82, "unit": "kgCO2e/kWh"},
                "natural_gas": {"value": 2.02, "unit": "kgCO2e/m3"},
                "diesel": {"value": 2.68, "unit": "kgCO2e/liter"},
                "lpg": {"value": 2.98, "unit": "kgCO2e/kg"}
            },
            "US": {
                "electricity": {"value": 0.42, "unit": "kgCO2e/kWh"},
                "natural_gas": {"value": 2.02, "unit": "kgCO2e/m3"},
                "diesel": {"value": 2.68, "unit": "kgCO2e/liter"}
            },
            "EU": {
                "electricity": {"value": 0.38, "unit": "kgCO2e/kWh"},
                "natural_gas": {"value": 2.02, "unit": "kgCO2e/m3"},
                "diesel": {"value": 2.68, "unit": "kgCO2e/liter"}
            },
            "DE": {
                "electricity": {"value": 0.35, "unit": "kgCO2e/kWh"},
                "natural_gas": {"value": 2.02, "unit": "kgCO2e/m3"},
                "diesel": {"value": 2.68, "unit": "kgCO2e/liter"}
            }
        },
        "benchmarks": {
            "commercial_office": {
                "IN": {"excellent": 30, "good": 50, "average": 80, "poor": 120},
                "US": {"excellent": 25, "good": 40, "average": 60, "poor": 90},
                "EU": {"excellent": 20, "good": 35, "average": 55, "poor": 85}
            },
            "healthcare_hospital": {
                "IN": {"excellent": 60, "good": 90, "average": 120, "poor": 180}
            }
        }
    }


@pytest.fixture
def runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def tmp_outdir(tmp_path):
    """Create a temporary output directory."""
    outdir = tmp_path / "test_output"
    outdir.mkdir(exist_ok=True)
    yield outdir
    # Cleanup is automatic with tmp_path


@pytest.fixture
def mock_llm():
    """Mock LLM/OpenAI/LangChain calls with deterministic responses."""
    mock_responses = {
        "Calculate emissions for 1.5M kWh in India": {
            "location": {"country": "IN"},
            "consumption": {"electricity": {"value": 1500000, "unit": "kWh"}}
        },
        "default": {
            "response": "Mocked LLM response",
            "emissions": 1000.0
        }
    }
    
    def mock_call(prompt, *args, **kwargs):
        return mock_responses.get(prompt, mock_responses["default"])
    
    with patch("greenlang.llm.assistant.LLMAssistant.process") as mock:
        mock.side_effect = mock_call
        yield mock


@pytest.fixture
def workflow_runner():
    """Create a workflow runner for in-process testing."""
    from greenlang.workflows.orchestrator import WorkflowOrchestrator
    
    class TestWorkflowRunner:
        def __init__(self):
            self.orchestrator = WorkflowOrchestrator()
            
        def run(self, workflow_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Run a workflow and return results."""
            return self.orchestrator.execute(workflow_path, input_data)
    
    return TestWorkflowRunner()


# ==================== Assertion Helpers ====================
@pytest.fixture
def assert_close():
    """Provide numerical comparison helper."""
    def _assert_close(actual, expected, rel_tol=1e-9, abs_tol=1e-9, msg=""):
        """Assert two numbers are close within tolerance."""
        import math
        if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            raise AssertionError(
                f"{msg}\nActual: {actual}\nExpected: {expected}\n"
                f"Difference: {abs(actual - expected)}"
            )
    return _assert_close


@pytest.fixture
def normalize_text():
    """Provide text normalization helper."""
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        import re
        # Remove timestamps
        text = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', text)
        # Remove absolute paths
        text = re.sub(r'[A-Z]:\\[^\\]+\\', '<PATH>/', text)
        text = re.sub(r'/[^/]+/', '<PATH>/', text)
        # Normalize temp directories
        text = re.sub(r'(tmp|temp)[^/\\]*[/\\][\w-]+', '<TMP>', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return _normalize


# ==================== Context Managers ====================
@pytest.fixture
def block_network_context():
    """Context manager to temporarily block network access."""
    @contextmanager
    def _block():
        blocker = NetworkBlocker()
        blocker.enable()
        try:
            yield
        finally:
            blocker.disable()
    return _block


@pytest.fixture
def capture_logs():
    """Capture and return logs during test execution."""
    import logging
    from io import StringIO
    
    @contextmanager
    def _capture(level=logging.INFO):
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(level)
        
        # Get root logger
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(level)
        logger.addHandler(handler)
        
        try:
            yield log_capture
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
    
    return _capture


# ==================== Data Loading Fixtures ====================
@pytest.fixture
def load_fixture_workflow():
    """Load a workflow fixture file."""
    def _load(filename: str) -> Dict[str, Any]:
        import yaml
        fixture_path = Path(__file__).parent.parent / "fixtures" / "workflows" / filename
        with open(fixture_path) as f:
            return yaml.safe_load(f)
    return _load


@pytest.fixture
def load_fixture_data():
    """Load a data fixture file."""
    def _load(filename: str) -> Dict[str, Any]:
        fixture_path = Path(__file__).parent.parent / "fixtures" / "data" / filename
        with open(fixture_path) as f:
            return json.load(f)
    return _load


# ==================== Mock Agents ====================
@pytest.fixture(autouse=True)
def mock_agents():
    """Mock all agents with deterministic behavior."""
    from unittest.mock import Mock
    
    # Create mock agent registry
    mock_registry = {
        "DataParserAgent": Mock(execute=lambda x: {"parsed_data": x["data"]}),
        "EmissionCalculatorAgent": Mock(execute=lambda x: {
            "emissions": {
                "total_co2e_kg": 1230000.0,
                "total_co2e_tons": 1230.0,
                "by_fuel": {
                    "electricity": 1230000.0,
                    "natural_gas": 101000.0,
                    "diesel": 26800.0
                }
            },
            "provenance": {
                "dataset_version": "0.0.1",
                "source": "Test Dataset",
                "last_updated": "2024-01-01"
            }
        }),
        "BenchmarkAgent": Mock(execute=lambda x: {
            "rating": "Good",
            "comparison": {"percentile": 75}
        }),
        "RecommendationAgent": Mock(execute=lambda x: {
            "recommendations": [
                {"action": "Install LED lighting", "savings": "10%"},
                {"action": "Optimize HVAC", "savings": "15%"},
                {"action": "Solar installation", "savings": "20%"}
            ]
        }),
        "ReportAgent": Mock(execute=lambda x: {
            "report": {
                "emissions": x.get("emissions"),
                "benchmark": x.get("benchmark"),
                "recommendations": x.get("recommendations"),
                "provenance": x.get("provenance")
            }
        })
    }
    
    with patch("greenlang.agents.registry.get_agent") as mock_get:
        mock_get.side_effect = lambda agent_id: mock_registry.get(agent_id, Mock())
        yield mock_registry