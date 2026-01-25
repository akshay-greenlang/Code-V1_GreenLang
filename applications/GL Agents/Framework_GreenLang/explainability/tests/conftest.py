"""
Pytest Configuration for Explainability Tests

Shared fixtures and configuration for all explainability test modules.

Author: GreenLang AI Team
"""

import pytest
import numpy as np
from datetime import datetime

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility across all tests."""
    np.random.seed(42)
    yield


@pytest.fixture
def sample_feature_names_4():
    """4 feature names for testing."""
    return ["temperature", "pressure", "flow_rate", "efficiency"]


@pytest.fixture
def sample_feature_names_6():
    """6 feature names for testing."""
    return ["temp_in", "temp_out", "pressure", "flow", "efficiency", "heat_duty"]


@pytest.fixture
def sample_training_data_100x4():
    """100 samples x 4 features training data."""
    np.random.seed(42)
    return np.random.randn(100, 4)


@pytest.fixture
def sample_training_data_500x4():
    """500 samples x 4 features training data."""
    np.random.seed(42)
    return np.random.randn(500, 4)


@pytest.fixture
def sample_instance_4():
    """4-feature sample instance."""
    return np.array([350.0, 101325.0, 10.5, 0.85])


@pytest.fixture
def sample_instances_10x4():
    """10 samples x 4 features test instances."""
    np.random.seed(42)
    return np.random.randn(10, 4)


@pytest.fixture
def mock_regression_predict():
    """Mock regression prediction function."""
    def predict(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.sum(x, axis=1) * 0.1
    return predict


@pytest.fixture
def mock_classification_predict():
    """Mock classification prediction function."""
    def predict_proba(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        prob = 1 / (1 + np.exp(-np.sum(x, axis=1)))
        return np.column_stack([1 - prob, prob])
    return predict_proba


@pytest.fixture
def combustion_inputs():
    """Sample combustion calculation inputs."""
    return {
        "fuel_type": "natural_gas",
        "fuel_flow_rate": 100.0,
        "air_flow_rate": 1200.0,
        "excess_air_ratio": 1.15,
        "inlet_air_temp": 300.0,
        "fuel_temperature": 288.0
    }


@pytest.fixture
def combustion_outputs():
    """Sample combustion calculation outputs."""
    return {
        "combustion_efficiency": 0.92,
        "flue_gas_temperature": 450.0,
        "co2_emissions": 1200.0,
        "nox_emissions": 50.0,
        "heat_output": 4500.0
    }


@pytest.fixture
def steam_inputs():
    """Sample steam calculation inputs."""
    return {
        "pressure": 1.0,  # MPa
        "temperature": 453.15,  # K (180C)
        "flow_rate": 10.0  # kg/s
    }


@pytest.fixture
def steam_outputs():
    """Sample steam calculation outputs."""
    return {
        "enthalpy": 2778.0,  # kJ/kg
        "entropy": 6.585,  # kJ/kg-K
        "quality": 1.0,  # Superheated
        "density": 2.85  # kg/m3
    }


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_shap: marks tests that require SHAP library"
    )
    config.addinivalue_line(
        "markers", "requires_lime: marks tests that require LIME library"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available libraries."""
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False

    try:
        import lime
        lime_available = True
    except ImportError:
        lime_available = False

    skip_shap = pytest.mark.skip(reason="SHAP library not installed")
    skip_lime = pytest.mark.skip(reason="LIME library not installed")

    for item in items:
        if "requires_shap" in item.keywords and not shap_available:
            item.add_marker(skip_shap)
        if "requires_lime" in item.keywords and not lime_available:
            item.add_marker(skip_lime)
