# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Explainability module tests.

Provides common fixtures for testing SHAP/LIME explainability
with determinism and reproducibility guarantees.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import hashlib


# Set global random seed for reproducibility
RANDOM_SEED = 42


@pytest.fixture(scope="session")
def random_seed():
    """Provide consistent random seed for all tests."""
    return RANDOM_SEED


@pytest.fixture
def set_random_seed():
    """Reset random seed before each test."""
    np.random.seed(RANDOM_SEED)
    return RANDOM_SEED


@pytest.fixture
def feature_names() -> List[str]:
    """Standard feature names for thermal command system."""
    return [
        "temperature_inlet_c",
        "temperature_outlet_c",
        "pressure_bar",
        "flow_rate_kg_s",
        "ambient_temp_c",
        "humidity_percent",
        "load_factor",
        "time_of_day",
        "day_of_week",
        "seasonal_factor"
    ]


@pytest.fixture
def training_data(set_random_seed, feature_names) -> np.ndarray:
    """Generate synthetic training data for testing."""
    n_samples = 500
    n_features = len(feature_names)

    # Generate realistic thermal data
    X = np.zeros((n_samples, n_features))

    # Temperature inlet: 200-600 C
    X[:, 0] = np.random.uniform(200, 600, n_samples)
    # Temperature outlet: 100-400 C (less than inlet)
    X[:, 1] = np.random.uniform(100, 400, n_samples)
    # Pressure: 1-50 bar
    X[:, 2] = np.random.uniform(1, 50, n_samples)
    # Flow rate: 1-20 kg/s
    X[:, 3] = np.random.uniform(1, 20, n_samples)
    # Ambient temp: -10 to 40 C
    X[:, 4] = np.random.uniform(-10, 40, n_samples)
    # Humidity: 20-90%
    X[:, 5] = np.random.uniform(20, 90, n_samples)
    # Load factor: 0.3-1.0
    X[:, 6] = np.random.uniform(0.3, 1.0, n_samples)
    # Time of day: 0-23
    X[:, 7] = np.random.randint(0, 24, n_samples).astype(float)
    # Day of week: 0-6
    X[:, 8] = np.random.randint(0, 7, n_samples).astype(float)
    # Seasonal factor: 0-1
    X[:, 9] = np.random.uniform(0, 1, n_samples)

    return X


@pytest.fixture
def target_values(training_data) -> np.ndarray:
    """Generate target values (heat demand) based on features."""
    X = training_data

    # Simple model: demand = f(temperature_diff, flow_rate, load_factor)
    temp_diff = X[:, 0] - X[:, 1]  # inlet - outlet
    flow = X[:, 3]
    load = X[:, 6]
    ambient = X[:, 4]

    # Heat demand in MW
    y = (
        0.1 * temp_diff * flow * load  # Main heat transfer
        + 0.5 * (20 - ambient)  # Ambient temperature effect
        + 5 * load  # Base load
        + np.random.normal(0, 2, len(X))  # Noise
    )

    return y


@pytest.fixture
def trained_random_forest(training_data, target_values, set_random_seed):
    """Train a RandomForest model for testing."""
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(training_data, target_values)
    return model


@pytest.fixture
def trained_gradient_boosting(training_data, target_values, set_random_seed):
    """Train a GradientBoosting model for testing."""
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=RANDOM_SEED
    )
    model.fit(training_data, target_values)
    return model


@pytest.fixture
def trained_linear_regression(training_data, target_values):
    """Train a Linear Regression model for testing."""
    model = LinearRegression()
    model.fit(training_data, target_values)
    return model


@pytest.fixture
def sample_instance(training_data, set_random_seed) -> np.ndarray:
    """Generate a sample instance for explanation."""
    # Use mean values with some variation
    instance = np.mean(training_data, axis=0)
    instance += np.random.normal(0, 0.1, len(instance)) * instance
    return instance


@pytest.fixture
def sample_batch(training_data, set_random_seed) -> np.ndarray:
    """Generate a batch of instances for explanation."""
    indices = np.random.choice(len(training_data), 10, replace=False)
    return training_data[indices]


@pytest.fixture
def optimization_context():
    """Create sample optimization context for testing."""
    from explainability.explainability_service import OptimizationContext

    return OptimizationContext(
        objective_value=12500.0,
        variable_values={
            "boiler_1_output": 50.0,
            "boiler_2_output": 35.0,
            "heat_pump_output": 15.0,
            "storage_charge": 5.0,
            "storage_discharge": 0.0
        },
        constraint_values={
            "total_demand": 100.0,
            "max_boiler_1": 50.0,
            "max_boiler_2": 40.0,
            "min_efficiency": 0.85,
            "max_emissions": 180.0
        },
        constraint_bounds={
            "total_demand": {"lower": 95.0, "upper": 105.0},
            "max_boiler_1": {"upper": 50.0},
            "max_boiler_2": {"upper": 40.0},
            "min_efficiency": {"lower": 0.85},
            "max_emissions": {"upper": 200.0}
        },
        shadow_prices={
            "total_demand": 25.0,
            "max_boiler_1": 5.0,
            "max_boiler_2": 0.0,
            "min_efficiency": -10.0,
            "max_emissions": -3.0
        },
        reduced_costs={
            "boiler_1_output": 0.0,
            "boiler_2_output": 2.0,
            "heat_pump_output": -1.0,
            "storage_charge": 5.0,
            "storage_discharge": 8.0
        },
        solver_status="optimal",
        optimality_gap=0.001
    )


@pytest.fixture
def mock_prediction_function(trained_random_forest):
    """Create a prediction function wrapper."""
    def predict_fn(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return trained_random_forest.predict(X)
    return predict_fn


# Markers for test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "determinism: tests for deterministic behavior"
    )
    config.addinivalue_line(
        "markers", "shap: tests for SHAP explainer"
    )
    config.addinivalue_line(
        "markers", "lime: tests for LIME explainer"
    )
    config.addinivalue_line(
        "markers", "service: tests for explainability service"
    )
    config.addinivalue_line(
        "markers", "reports: tests for report generation"
    )
    config.addinivalue_line(
        "markers", "slow: slow tests that may be skipped"
    )
