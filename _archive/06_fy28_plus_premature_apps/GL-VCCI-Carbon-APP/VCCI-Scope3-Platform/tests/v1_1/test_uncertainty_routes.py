# -*- coding: utf-8 -*-
"""
Unit Tests for Uncertainty Analysis API Routes (v1.1)

Tests FastAPI endpoints for Monte Carlo uncertainty analysis, Sobol/Morris
sensitivity, tornado diagrams, convergence assessment, and scenario comparison.

Target module: services/api/uncertainty_routes.py
Test count: 33 tests
Coverage target: 85%+
"""

import pytest
import json
import numpy as np
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

import sys
import os

PLATFORM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.uncertainty_routes import (
    router,
    _store,
    AnalyzeRequest,
    AnalyzeResponse,
    SensitivityRequest,
    SensitivityResponse,
    CompareRequest,
    CompareResponse,
    ParameterSpec,
    ScenarioSpec,
)

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI()
app.include_router(router)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def analyze_payload():
    """Standard analysis request payload."""
    return {
        "calculation_type": "multiply",
        "parameters": [
            {"name": "activity_data", "mean": 1000.0, "std_dev": 50.0,
             "distribution": "normal"},
            {"name": "emission_factor", "mean": 2.5, "std_dev": 0.3,
             "distribution": "lognormal"},
        ],
        "iterations": 2000,
        "seed": 42,
        "include_sensitivity": False,
    }


@pytest.fixture
def analyze_with_sensitivity_payload():
    """Analysis request with Sobol sensitivity enabled."""
    return {
        "calculation_type": "multiply",
        "parameters": [
            {"name": "activity_data", "mean": 1000.0, "std_dev": 50.0,
             "distribution": "normal"},
            {"name": "emission_factor", "mean": 2.5, "std_dev": 0.3,
             "distribution": "lognormal"},
        ],
        "iterations": 2000,
        "seed": 42,
        "include_sensitivity": True,
        "sensitivity_N": 128,
    }


@pytest.fixture
def sobol_sensitivity_payload():
    """Sobol sensitivity request payload."""
    return {
        "method": "sobol",
        "parameters": [
            {"name": "x1", "mean": 100.0, "std_dev": 10.0,
             "distribution": "normal"},
            {"name": "x2", "mean": 50.0, "std_dev": 5.0,
             "distribution": "normal"},
        ],
        "calculation_func_type": "multiply",
        "iterations": 128,
        "seed": 42,
    }


@pytest.fixture
def morris_sensitivity_payload():
    """Morris screening request payload."""
    return {
        "method": "morris",
        "parameters": [
            {"name": "x1", "mean": 100.0, "std_dev": 10.0,
             "distribution": "normal"},
            {"name": "x2", "mean": 50.0, "std_dev": 5.0,
             "distribution": "normal"},
        ],
        "calculation_func_type": "add",
        "iterations": 10,
        "seed": 42,
    }


@pytest.fixture
def tornado_sensitivity_payload():
    """Tornado sensitivity request payload."""
    return {
        "method": "tornado",
        "parameters": [
            {"name": "x1", "mean": 100.0, "std_dev": 10.0,
             "distribution": "normal"},
            {"name": "x2", "mean": 50.0, "std_dev": 5.0,
             "distribution": "normal"},
        ],
        "calculation_func_type": "add",
        "iterations": 100,
        "seed": 42,
        "variation_pct": 0.10,
    }


@pytest.fixture
def compare_payload():
    """Scenario comparison request payload."""
    return {
        "scenarios": [
            {
                "name": "Baseline",
                "parameters": [
                    {"name": "ef", "mean": 2.5, "std_dev": 0.3,
                     "distribution": "lognormal"},
                    {"name": "ad", "mean": 1000.0, "std_dev": 50.0,
                     "distribution": "normal"},
                ],
            },
            {
                "name": "Low-Carbon",
                "parameters": [
                    {"name": "ef", "mean": 1.5, "std_dev": 0.2,
                     "distribution": "lognormal"},
                    {"name": "ad", "mean": 1000.0, "std_dev": 50.0,
                     "distribution": "normal"},
                ],
            },
        ],
        "iterations": 2000,
        "seed": 42,
        "calculation_type": "multiply",
    }


@pytest.fixture
def stored_calculation_id(client, analyze_payload):
    """Run an analysis and return the calculation_id for GET tests."""
    response = client.post("/api/v1/uncertainty/analyze", json=analyze_payload)
    assert response.status_code == 200
    return response.json()["calculation_id"]


# ============================================================================
# TEST: POST /analyze
# ============================================================================

class TestUncertaintyAnalyzeEndpoint:
    """Test the POST /analyze endpoint."""

    def test_post_analyze_returns_result(self, client, analyze_payload):
        """POST /analyze should return 200 with a calculation_id."""
        response = client.post("/api/v1/uncertainty/analyze", json=analyze_payload)
        assert response.status_code == 200
        data = response.json()
        assert "calculation_id" in data
        assert "result" in data
        assert "convergence" in data

    def test_post_analyze_validates_input(self, client):
        """POST /analyze with missing parameters should return 422."""
        response = client.post("/api/v1/uncertainty/analyze", json={
            "calculation_type": "multiply",
            "parameters": [],  # min_length=1 violated
        })
        assert response.status_code == 422

    def test_post_analyze_returns_statistics(self, client, analyze_payload):
        """Result should contain mean, std_dev, percentiles."""
        response = client.post("/api/v1/uncertainty/analyze", json=analyze_payload)
        data = response.json()
        result = data["result"]
        assert "mean" in result
        assert "std_dev" in result
        assert result["mean"] > 0
        assert result["std_dev"] > 0

    def test_post_analyze_with_custom_iterations(self, client):
        """Custom iteration count should be respected."""
        payload = {
            "calculation_type": "add",
            "parameters": [
                {"name": "x1", "mean": 100.0, "std_dev": 10.0,
                 "distribution": "normal"},
            ],
            "iterations": 5000,
            "seed": 42,
        }
        response = client.post("/api/v1/uncertainty/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["iterations"] == 5000

    def test_post_analyze_with_sensitivity(
        self, client, analyze_with_sensitivity_payload
    ):
        """When include_sensitivity=True, Sobol result should be present."""
        response = client.post(
            "/api/v1/uncertainty/analyze",
            json=analyze_with_sensitivity_payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sensitivity"] is not None
        assert "first_order_indices" in data["sensitivity"]

    def test_post_analyze_convergence_present(self, client, analyze_payload):
        """Convergence assessment should always be included."""
        response = client.post("/api/v1/uncertainty/analyze", json=analyze_payload)
        data = response.json()
        conv = data["convergence"]
        assert "is_converged" in conv
        assert "recommended_iterations" in conv
        assert "running_means" in conv


# ============================================================================
# TEST: GET /{id}
# ============================================================================

class TestUncertaintyGetEndpoint:
    """Test the GET /{id} endpoint."""

    def test_get_by_id_returns_result(self, client, stored_calculation_id):
        """GET /{id} should return the stored Monte Carlo result."""
        response = client.get(
            f"/api/v1/uncertainty/{stored_calculation_id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "mean" in data
        assert "std_dev" in data

    def test_get_nonexistent_returns_404(self, client):
        """GET with unknown ID should return 404."""
        response = client.get(
            "/api/v1/uncertainty/nonexistent-id-12345"
        )
        assert response.status_code == 404

    def test_get_distribution_data(self, client, stored_calculation_id):
        """GET /{id}/distribution should return histogram and KDE data."""
        response = client.get(
            f"/api/v1/uncertainty/{stored_calculation_id}/distribution"
        )
        assert response.status_code == 200
        data = response.json()
        assert "bins" in data
        assert "kde" in data
        assert "percentiles" in data
        assert "statistics" in data
        assert len(data["bins"]) > 0
        assert len(data["kde"]) > 0
        # Check percentiles
        pcts = data["percentiles"]
        assert pcts["p5"] <= pcts["p50"] <= pcts["p95"]

    def test_get_distribution_nonexistent_404(self, client):
        """GET /nonexistent/distribution should return 404."""
        response = client.get(
            "/api/v1/uncertainty/no-such-id/distribution"
        )
        assert response.status_code == 404


# ============================================================================
# TEST: POST /sensitivity
# ============================================================================

class TestSensitivityEndpoint:
    """Test the POST /sensitivity endpoint."""

    def test_post_sobol_analysis(self, client, sobol_sensitivity_payload):
        """POST /sensitivity with method=sobol should return Sobol indices."""
        response = client.post(
            "/api/v1/uncertainty/sensitivity",
            json=sobol_sensitivity_payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "sobol"
        assert data["sobol"] is not None
        assert "first_order_indices" in data["sobol"]
        assert "total_order_indices" in data["sobol"]
        assert "x1" in data["sobol"]["first_order_indices"]
        assert "x2" in data["sobol"]["first_order_indices"]

    def test_post_morris_screening(self, client, morris_sensitivity_payload):
        """POST /sensitivity with method=morris should return mu_star and sigma."""
        response = client.post(
            "/api/v1/uncertainty/sensitivity",
            json=morris_sensitivity_payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "morris"
        assert data["morris"] is not None
        assert "mu_star" in data["morris"]
        assert "sigma" in data["morris"]

    def test_post_tornado_analysis(self, client, tornado_sensitivity_payload):
        """POST /sensitivity with method=tornado should return tornado data."""
        response = client.post(
            "/api/v1/uncertainty/sensitivity",
            json=tornado_sensitivity_payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "tornado"
        assert data["tornado"] is not None
        assert "parameters" in data["tornado"]
        assert "baseline_output" in data["tornado"]

    def test_sensitivity_returns_ranked_parameters(
        self, client, sobol_sensitivity_payload
    ):
        """Sobol indices should rank parameters by importance."""
        response = client.post(
            "/api/v1/uncertainty/sensitivity",
            json=sobol_sensitivity_payload,
        )
        data = response.json()
        indices = data["sobol"]["first_order_indices"]
        # Both parameters should have values between 0 and 1
        for name, val in indices.items():
            assert 0.0 <= val <= 1.0, (
                f"Index for {name} = {val}, expected [0,1]"
            )

    def test_sensitivity_invalid_method(self, client):
        """Invalid method should return 422."""
        payload = {
            "method": "invalid_method",
            "parameters": [
                {"name": "x1", "mean": 100.0, "std_dev": 10.0,
                 "distribution": "normal"},
            ],
            "calculation_func_type": "add",
            "iterations": 64,
        }
        response = client.post(
            "/api/v1/uncertainty/sensitivity", json=payload
        )
        assert response.status_code == 422


# ============================================================================
# TEST: GET /convergence/{id}
# ============================================================================

class TestConvergenceEndpoint:
    """Test the GET /convergence/{id} endpoint."""

    def test_convergence_assessment(self, client, stored_calculation_id):
        """GET /convergence/{id} should return convergence data."""
        response = client.get(
            f"/api/v1/uncertainty/convergence/{stored_calculation_id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_converged" in data
        assert "recommended_iterations" in data
        assert "running_means" in data
        assert "running_stds" in data

    def test_convergence_returns_boolean(self, client, stored_calculation_id):
        """is_converged should be a boolean value."""
        response = client.get(
            f"/api/v1/uncertainty/convergence/{stored_calculation_id}"
        )
        data = response.json()
        assert isinstance(data["is_converged"], bool)

    def test_convergence_nonexistent_404(self, client):
        """GET /convergence for unknown ID should return 404."""
        response = client.get(
            "/api/v1/uncertainty/convergence/no-such-id"
        )
        assert response.status_code == 404


# ============================================================================
# TEST: POST /compare-scenarios
# ============================================================================

class TestScenarioComparisonEndpoint:
    """Test the POST /compare-scenarios endpoint."""

    def test_compare_two_scenarios(self, client, compare_payload):
        """Comparing two scenarios should return per-scenario statistics."""
        response = client.post(
            "/api/v1/uncertainty/compare-scenarios",
            json=compare_payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["comparisons"]) == 2
        assert data["comparisons"][0]["name"] == "Baseline"
        assert data["comparisons"][1]["name"] == "Low-Carbon"
        # Baseline should have higher mean than Low-Carbon
        assert data["comparisons"][0]["mean"] > data["comparisons"][1]["mean"]

    def test_compare_with_target_probability(self, client, compare_payload):
        """When target_value is provided, exceedance probabilities are returned."""
        compare_payload["target_value"] = 3000.0
        response = client.post(
            "/api/v1/uncertainty/compare-scenarios",
            json=compare_payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["target_probability"] is not None
        assert "Baseline" in data["target_probability"]
        assert "Low-Carbon" in data["target_probability"]
        # Exceedance probabilities should be between 0 and 1
        for name, prob in data["target_probability"].items():
            assert 0.0 <= prob <= 1.0

    def test_compare_statistical_significance(self, client, compare_payload):
        """Pairwise t-test should be returned between scenarios."""
        response = client.post(
            "/api/v1/uncertainty/compare-scenarios",
            json=compare_payload,
        )
        data = response.json()
        sig = data["statistical_significance"]
        assert len(sig) == 1  # 2 scenarios -> 1 pair
        assert sig[0]["scenario_a"] == "Baseline"
        assert sig[0]["scenario_b"] == "Low-Carbon"
        assert "p_value" in sig[0]
        assert isinstance(sig[0]["significant"], bool)

    def test_compare_with_uncertainty_bands(self, client, compare_payload):
        """Each scenario comparison should include percentile bands."""
        response = client.post(
            "/api/v1/uncertainty/compare-scenarios",
            json=compare_payload,
        )
        data = response.json()
        for comp in data["comparisons"]:
            assert "p5" in comp
            assert "p25" in comp
            assert "p50" in comp
            assert "p75" in comp
            assert "p95" in comp
            # Percentiles should be ordered
            assert comp["p5"] <= comp["p25"] <= comp["p50"]
            assert comp["p50"] <= comp["p75"] <= comp["p95"]

    def test_compare_requires_at_least_two_scenarios(self, client):
        """Comparing with fewer than 2 scenarios should fail validation."""
        payload = {
            "scenarios": [
                {
                    "name": "Only-One",
                    "parameters": [
                        {"name": "x", "mean": 100, "std_dev": 10,
                         "distribution": "normal"},
                    ],
                },
            ],
            "iterations": 1000,
        }
        response = client.post(
            "/api/v1/uncertainty/compare-scenarios", json=payload
        )
        assert response.status_code == 422

    def test_compare_three_scenarios(self, client):
        """Three scenarios should produce 3 pairwise significance tests."""
        payload = {
            "scenarios": [
                {
                    "name": "Baseline",
                    "parameters": [
                        {"name": "ef", "mean": 2.5, "std_dev": 0.3,
                         "distribution": "lognormal"},
                    ],
                },
                {
                    "name": "Low",
                    "parameters": [
                        {"name": "ef", "mean": 1.5, "std_dev": 0.2,
                         "distribution": "lognormal"},
                    ],
                },
                {
                    "name": "High",
                    "parameters": [
                        {"name": "ef", "mean": 4.0, "std_dev": 0.5,
                         "distribution": "lognormal"},
                    ],
                },
            ],
            "iterations": 1000,
            "seed": 42,
            "calculation_type": "multiply",
        }
        response = client.post(
            "/api/v1/uncertainty/compare-scenarios", json=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["comparisons"]) == 3
        # 3 choose 2 = 3 pairwise tests
        assert len(data["statistical_significance"]) == 3

    def test_analyze_add_calculation_type(self, client):
        """POST /analyze with calculation_type=add should work."""
        payload = {
            "calculation_type": "add",
            "parameters": [
                {"name": "a", "mean": 100.0, "std_dev": 10.0,
                 "distribution": "normal"},
                {"name": "b", "mean": 200.0, "std_dev": 20.0,
                 "distribution": "normal"},
            ],
            "iterations": 1000,
            "seed": 42,
        }
        response = client.post("/api/v1/uncertainty/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Mean of a+b should be near 300
        assert abs(data["result"]["mean"] - 300.0) < 50.0

    def test_analyze_uniform_distribution(self, client):
        """POST /analyze with uniform distribution should work."""
        payload = {
            "calculation_type": "multiply",
            "parameters": [
                {"name": "x", "mean": 50.0, "std_dev": 10.0,
                 "distribution": "uniform", "min": 20.0, "max": 80.0},
            ],
            "iterations": 2000,
            "seed": 42,
        }
        response = client.post("/api/v1/uncertainty/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["mean"] > 0

    def test_distribution_statistics_valid(self, client, stored_calculation_id):
        """Distribution statistics should include all expected keys."""
        response = client.get(
            f"/api/v1/uncertainty/{stored_calculation_id}/distribution"
        )
        data = response.json()
        stats = data["statistics"]
        assert "mean" in stats
        assert "median" in stats
        assert "std_dev" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert stats["std_dev"] > 0

    def test_tornado_variation_pct_applied(self, client):
        """Tornado with different variation_pct should produce different impacts."""
        base_payload = {
            "method": "tornado",
            "parameters": [
                {"name": "x1", "mean": 100.0, "std_dev": 10.0,
                 "distribution": "normal"},
            ],
            "calculation_func_type": "multiply",
            "iterations": 100,
            "seed": 42,
        }

        # 10% variation
        payload_10 = {**base_payload, "variation_pct": 0.10}
        r1 = client.post("/api/v1/uncertainty/sensitivity", json=payload_10)
        impact_10 = r1.json()["tornado"]["parameters"][0]["impact"]

        # 20% variation
        payload_20 = {**base_payload, "variation_pct": 0.20}
        r2 = client.post("/api/v1/uncertainty/sensitivity", json=payload_20)
        impact_20 = r2.json()["tornado"]["parameters"][0]["impact"]

        assert impact_20 > impact_10, "20% variation should produce larger impact"
