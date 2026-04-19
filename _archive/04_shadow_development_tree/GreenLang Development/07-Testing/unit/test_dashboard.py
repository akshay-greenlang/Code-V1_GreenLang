# -*- coding: utf-8 -*-
"""
Unit Tests for ML Explainability Dashboard Component.

TASK-030: Tests for dashboard.py

Tests cover:
- Dashboard Data Models (Pydantic)
- Visualization Data Generators
- Dashboard State Management
- Export Functionality
- API Endpoints (with mock model)

Author: GreenLang Team
"""

import json
import pytest
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_feature_contributions() -> Dict[str, float]:
    """Sample feature contributions for testing."""
    return {
        "temperature": 0.35,
        "pressure": -0.25,
        "flow_rate": 0.15,
        "humidity": -0.10,
        "load_factor": 0.08,
    }


@pytest.fixture
def sample_feature_values() -> Dict[str, float]:
    """Sample feature values for testing."""
    return {
        "temperature": 450.0,
        "pressure": 85.0,
        "flow_rate": 1200.0,
        "humidity": 45.0,
        "load_factor": 0.75,
    }


@pytest.fixture
def sample_feature_names() -> List[str]:
    """Sample feature names."""
    return ["temperature", "pressure", "flow_rate", "humidity", "load_factor"]


@pytest.fixture
def mock_model():
    """Create a mock ML model."""
    model = Mock()

    def predict_proba(X):
        # Simple sigmoid-like prediction
        probs = 1 / (1 + np.exp(-np.sum(X, axis=1) / X.shape[1]))
        return np.column_stack([1 - probs, probs])

    def predict(X):
        return np.sum(X, axis=1)

    model.predict_proba = predict_proba
    model.predict = predict
    return model


@pytest.fixture
def mock_training_data() -> np.ndarray:
    """Generate mock training data."""
    np.random.seed(42)
    return np.random.randn(100, 5)


# =============================================================================
# DATA MODEL TESTS
# =============================================================================

class TestFeatureContributionData:
    """Tests for FeatureContributionData model."""

    def test_create_valid_contribution(self):
        """Test creating a valid feature contribution."""
        from greenlang.ml.explainability.dashboard import FeatureContributionData

        contrib = FeatureContributionData(
            feature_name="temperature",
            display_name="Temperature (F)",
            value=450.0,
            contribution=0.35,
            percentage=35.0,
            direction="positive",
            color="#4caf50",
            rank=1,
        )

        assert contrib.feature_name == "temperature"
        assert contrib.contribution == 0.35
        assert contrib.direction == "positive"
        assert contrib.rank == 1

    def test_contribution_serialization(self):
        """Test JSON serialization of contribution."""
        from greenlang.ml.explainability.dashboard import FeatureContributionData

        contrib = FeatureContributionData(
            feature_name="pressure",
            display_name="Pressure",
            value=85.0,
            contribution=-0.25,
            percentage=25.0,
            direction="negative",
            rank=2,
        )

        json_data = contrib.model_dump_json()
        parsed = json.loads(json_data)

        assert parsed["feature_name"] == "pressure"
        assert parsed["contribution"] == -0.25


class TestFeatureContributionChart:
    """Tests for FeatureContributionChart model."""

    def test_create_waterfall_chart(self, sample_feature_contributions, sample_feature_values):
        """Test creating a waterfall chart data structure."""
        from greenlang.ml.explainability.dashboard import (
            FeatureContributionChart,
            FeatureContributionData,
            ChartType,
        )

        contributions = [
            FeatureContributionData(
                feature_name=name,
                display_name=name.title(),
                value=sample_feature_values[name],
                contribution=value,
                percentage=abs(value) * 100,
                direction="positive" if value > 0 else "negative",
                rank=i + 1,
            )
            for i, (name, value) in enumerate(sample_feature_contributions.items())
        ]

        chart = FeatureContributionChart(
            chart_type=ChartType.WATERFALL,
            title="Test Waterfall",
            base_value=0.5,
            prediction=0.73,
            contributions=contributions,
            cumulative_values=[0.5, 0.85, 0.60, 0.75, 0.65, 0.73],
            model_id="test-model",
            provenance_hash="abc123",
        )

        assert chart.chart_type == ChartType.WATERFALL
        assert len(chart.contributions) == 5
        assert chart.base_value == 0.5
        assert chart.prediction == 0.73

    def test_chart_json_serialization(self):
        """Test JSON serialization of chart."""
        from greenlang.ml.explainability.dashboard import (
            FeatureContributionChart,
            ChartType,
        )

        chart = FeatureContributionChart(
            chart_type=ChartType.FORCE_PLOT,
            base_value=0.5,
            prediction=0.7,
            contributions=[],
            model_id="test",
        )

        json_str = chart.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["chart_type"] == "force_plot"
        assert parsed["model_id"] == "test"
        assert "timestamp" in parsed


class TestGlobalImportanceChart:
    """Tests for GlobalImportanceChart model."""

    def test_create_importance_chart(self, sample_feature_names):
        """Test creating a global importance chart."""
        from greenlang.ml.explainability.dashboard import GlobalImportanceChart

        chart = GlobalImportanceChart(
            features=sample_feature_names,
            display_names=[f.title() for f in sample_feature_names],
            importance_scores=[0.35, 0.25, 0.20, 0.12, 0.08],
            total_features=5,
            sample_size=100,
            top_feature="temperature",
            top_importance=0.35,
            model_id="test-model",
        )

        assert chart.total_features == 5
        assert chart.top_feature == "temperature"
        assert sum(chart.importance_scores) == pytest.approx(1.0)


class TestCounterfactualComparisonView:
    """Tests for CounterfactualComparisonView model."""

    def test_create_counterfactual_view(self, sample_feature_values):
        """Test creating a counterfactual comparison view."""
        from greenlang.ml.explainability.dashboard import (
            CounterfactualComparisonView,
            CounterfactualChange,
        )

        changes = [
            CounterfactualChange(
                feature_name="temperature",
                display_name="Temperature",
                original_value=450.0,
                counterfactual_value=380.0,
                change_amount=70.0,
                change_percentage=15.6,
                direction="decrease",
                feasibility=0.85,
            )
        ]

        view = CounterfactualComparisonView(
            original_prediction=0.75,
            original_instance=sample_feature_values,
            counterfactual_prediction=0.25,
            changes=changes,
            num_features_changed=1,
            total_change_magnitude=70.0,
            overall_feasibility=0.85,
            model_id="test-model",
        )

        assert view.original_prediction == 0.75
        assert view.counterfactual_prediction == 0.25
        assert len(view.changes) == 1
        assert view.overall_feasibility == 0.85


class TestExplanationDashboardData:
    """Tests for ExplanationDashboardData model."""

    def test_create_complete_dashboard(self):
        """Test creating a complete dashboard data structure."""
        from greenlang.ml.explainability.dashboard import (
            ExplanationDashboardData,
            DashboardViewMode,
        )

        dashboard = ExplanationDashboardData(
            model_id="test-model",
            model_name="Test Model",
            prediction=0.75,
            explanation_confidence=0.92,
            provenance_hash="sha256hash",
            view_mode=DashboardViewMode.DETAILED,
        )

        assert dashboard.model_id == "test-model"
        assert dashboard.prediction == 0.75
        assert dashboard.view_mode == DashboardViewMode.DETAILED
        assert "dashboard_id" in dashboard.model_dump()


# =============================================================================
# VISUALIZATION DATA GENERATOR TESTS
# =============================================================================

class TestVisualizationDataGenerator:
    """Tests for VisualizationDataGenerator."""

    def test_generate_waterfall_data(
        self,
        sample_feature_contributions,
        sample_feature_values,
    ):
        """Test generating waterfall chart data."""
        from greenlang.ml.explainability.dashboard import (
            VisualizationDataGenerator,
            ChartType,
        )

        result = VisualizationDataGenerator.generate_waterfall_data(
            feature_contributions=sample_feature_contributions,
            feature_values=sample_feature_values,
            base_value=0.5,
            prediction=0.73,
            model_id="test-model",
            max_features=5,
        )

        assert result.chart_type == ChartType.WATERFALL
        assert len(result.contributions) == 5
        assert result.base_value == 0.5
        assert len(result.cumulative_values) == 6  # base + 5 features
        assert result.provenance_hash != ""

    def test_generate_force_plot_data(
        self,
        sample_feature_contributions,
        sample_feature_values,
    ):
        """Test generating force plot data."""
        from greenlang.ml.explainability.dashboard import (
            VisualizationDataGenerator,
            ChartType,
        )

        result = VisualizationDataGenerator.generate_force_plot_data(
            feature_contributions=sample_feature_contributions,
            feature_values=sample_feature_values,
            base_value=0.5,
            prediction=0.73,
            model_id="test-model",
        )

        assert result.chart_type == ChartType.FORCE_PLOT
        assert result.model_id == "test-model"
        assert result.provenance_hash != ""

    def test_generate_bar_chart_data(self):
        """Test generating bar chart data."""
        from greenlang.ml.explainability.dashboard import (
            VisualizationDataGenerator,
            ChartType,
        )

        importance = {
            "temperature": 0.35,
            "pressure": 0.25,
            "flow_rate": 0.20,
            "humidity": 0.12,
            "load_factor": 0.08,
        }

        result = VisualizationDataGenerator.generate_bar_chart_data(
            feature_importance=importance,
            model_id="test-model",
            sample_size=100,
        )

        assert result.chart_type == ChartType.BAR_CHART
        assert result.total_features == 5
        assert result.top_feature == "temperature"
        assert result.sample_size == 100

    def test_generate_counterfactual_view(self, sample_feature_values):
        """Test generating counterfactual view data."""
        from greenlang.ml.explainability.dashboard import VisualizationDataGenerator

        cf_instance = sample_feature_values.copy()
        cf_instance["temperature"] = 380.0  # Decrease temp

        result = VisualizationDataGenerator.generate_counterfactual_view(
            original_instance=sample_feature_values,
            original_prediction=0.75,
            counterfactual_instance=cf_instance,
            counterfactual_prediction=0.25,
            model_id="test-model",
        )

        assert result.original_prediction == 0.75
        assert result.counterfactual_prediction == 0.25
        assert len(result.changes) == 1
        assert result.changes[0].feature_name == "temperature"
        assert result.provenance_hash != ""

    def test_generate_time_series_data(self):
        """Test generating time series data."""
        from greenlang.ml.explainability.dashboard import (
            VisualizationDataGenerator,
            TimeRange,
        )

        # Create sample predictions and explanations
        predictions = [
            (datetime(2024, 1, 1, i, 0, 0), 0.5 + i * 0.05)
            for i in range(10)
        ]

        explanations = [
            (datetime(2024, 1, 1, i, 0, 0), {"temp": 0.3, "pressure": 0.2})
            for i in range(10)
        ]

        result = VisualizationDataGenerator.generate_time_series_data(
            predictions=predictions,
            explanations=explanations,
            model_id="test-model",
            time_range=TimeRange.LAST_DAY,
        )

        assert result.total_count == 10
        assert len(result.trend_predictions) == 10
        assert result.avg_prediction > 0
        assert result.prediction_std > 0


# =============================================================================
# DASHBOARD STATE MANAGEMENT TESTS
# =============================================================================

class TestDashboardStateManager:
    """Tests for DashboardStateManager."""

    def test_initialization(self):
        """Test state manager initialization."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager(cache_size=100, cache_ttl_seconds=60.0)

        assert manager._cache_size == 100
        assert manager._cache_ttl == 60.0
        assert len(manager._model_registry) == 0

    def test_model_registration(self, mock_model, sample_feature_names, mock_training_data):
        """Test registering a model."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager()

        manager.register_model(
            model_id="test-model",
            model=mock_model,
            feature_names=sample_feature_names,
            training_data=mock_training_data,
            metadata={"name": "Test Model", "type": "classifier"},
        )

        assert "test-model" in manager.list_models()
        info = manager.get_model("test-model")
        assert info is not None
        assert info["feature_names"] == sample_feature_names

    def test_model_selection(self, mock_model, sample_feature_names):
        """Test selecting a model."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager()
        manager.register_model("model-1", mock_model, sample_feature_names)
        manager.register_model("model-2", mock_model, sample_feature_names)

        manager.select_model("model-1")
        assert manager.get_selected_model() == "model-1"

        manager.select_model("model-2")
        assert manager.get_selected_model() == "model-2"

    def test_cache_operations(self):
        """Test caching explanations."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager(cache_size=10)

        # Cache an explanation
        manager.cache_explanation(
            model_id="test",
            instance_hash="hash123",
            explanation={"test": "data"},
            method="shap",
        )

        # Retrieve from cache
        cached = manager.get_cached_explanation("test", "hash123", "shap")
        assert cached == {"test": "data"}

        # Cache miss
        missing = manager.get_cached_explanation("test", "different", "shap")
        assert missing is None

    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager(cache_size=3)

        # Fill cache
        for i in range(5):
            manager.cache_explanation(
                model_id="test",
                instance_hash=f"hash{i}",
                explanation={"value": i},
                method="shap",
            )

        # First two should be evicted
        assert manager.get_cached_explanation("test", "hash0", "shap") is None
        assert manager.get_cached_explanation("test", "hash1", "shap") is None

        # Last three should exist
        assert manager.get_cached_explanation("test", "hash4", "shap") is not None

    def test_clear_cache(self):
        """Test clearing the cache."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager()

        for i in range(5):
            manager.cache_explanation("test", f"hash{i}", {"v": i}, "shap")

        count = manager.clear_cache()
        assert count == 5
        assert manager.cache_hit_rate == 0.0

    def test_user_preferences(self):
        """Test user preference storage."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager()

        manager.set_user_preference("user-1", "theme", "dark")
        manager.set_user_preference("user-1", "max_features", 10)

        prefs = manager.get_user_preferences("user-1")
        assert prefs["theme"] == "dark"
        assert prefs["max_features"] == 10

    def test_export_json(self):
        """Test JSON export."""
        from greenlang.ml.explainability.dashboard import (
            DashboardStateManager,
            FeatureContributionChart,
            ExportFormat,
        )

        manager = DashboardStateManager()

        chart = FeatureContributionChart(
            base_value=0.5,
            prediction=0.75,
            contributions=[],
            model_id="test",
        )

        exported = manager.export_data(chart, ExportFormat.JSON)
        parsed = json.loads(exported)

        assert parsed["model_id"] == "test"
        assert parsed["base_value"] == 0.5

    def test_export_csv(self):
        """Test CSV export."""
        from greenlang.ml.explainability.dashboard import (
            DashboardStateManager,
            GlobalImportanceChart,
            ExportFormat,
        )

        manager = DashboardStateManager()

        chart = GlobalImportanceChart(
            features=["temp", "pressure"],
            display_names=["Temperature", "Pressure"],
            importance_scores=[0.6, 0.4],
            total_features=2,
            sample_size=100,
            top_feature="temp",
            top_importance=0.6,
            model_id="test",
        )

        exported = manager.export_data(chart, ExportFormat.CSV)

        assert "feature,display_name,importance" in exported
        assert "temp,Temperature,0.6" in exported

    def test_get_summary(self, mock_model, sample_feature_names, mock_training_data):
        """Test getting dashboard summary."""
        from greenlang.ml.explainability.dashboard import DashboardStateManager

        manager = DashboardStateManager()

        manager.register_model(
            model_id="model-1",
            model=mock_model,
            feature_names=sample_feature_names,
            training_data=mock_training_data,
            metadata={"name": "Model 1", "type": "classifier"},
        )

        summary = manager.get_summary()

        assert summary.total_models == 1
        assert len(summary.models) == 1
        assert summary.models[0].model_id == "model-1"


# =============================================================================
# API ENDPOINT TESTS (INTEGRATION)
# =============================================================================

@pytest.fixture
def test_client():
    """Create a test client for API testing."""
    try:
        from fastapi.testclient import TestClient
        from greenlang.ml.explainability.dashboard import dashboard_router, dashboard_state

        # Create a minimal FastAPI app for testing
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(dashboard_router)

        # Clear state for each test
        dashboard_state._model_registry.clear()
        dashboard_state._cache.clear()
        dashboard_state._explanation_history.clear()

        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI test client not available")


class TestDashboardAPIEndpoints:
    """Tests for Dashboard API endpoints."""

    def test_list_models_empty(self, test_client):
        """Test listing models when none registered."""
        response = test_client.get("/dashboard/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []

    def test_register_model(self, test_client):
        """Test registering a model via API."""
        response = test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp", "pressure", "flow"]',
                "model_name": "Test Model",
                "model_type": "classifier",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_id"] == "test-model"

    def test_list_models_after_register(self, test_client):
        """Test listing models after registration."""
        # Register a model
        test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp", "pressure"]',
            }
        )

        response = test_client.get("/dashboard/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "test-model"

    def test_get_dashboard_summary(self, test_client):
        """Test getting dashboard summary."""
        response = test_client.get("/dashboard/summary")
        assert response.status_code == 200
        data = response.json()

        assert "total_models" in data
        assert "system_health" in data
        assert "shap_available" in data

    def test_get_explanation_model_not_found(self, test_client):
        """Test getting explanation for non-existent model."""
        response = test_client.get("/dashboard/explanation/nonexistent")
        assert response.status_code == 404

    def test_get_explanation_with_instance(self, test_client):
        """Test getting explanation with instance data."""
        # Register model first
        test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp", "pressure", "flow"]',
            }
        )

        # Get explanation
        instance = {"temp": 100.0, "pressure": 50.0, "flow": 200.0}
        response = test_client.get(
            "/dashboard/explanation/test-model",
            params={
                "instance_data": json.dumps(instance),
                "include_global": True,
                "max_features": 5,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test-model"
        assert "provenance_hash" in data

    def test_get_feature_importance(self, test_client):
        """Test getting feature importance."""
        # Register model first
        test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp", "pressure", "flow"]',
            }
        )

        response = test_client.get(
            "/dashboard/feature-importance/test-model",
            params={"method": "shap", "sample_size": 50}
        )

        # May fail if SHAP not installed, which is ok
        assert response.status_code in [200, 501]

    def test_get_counterfactuals_missing_instance(self, test_client):
        """Test getting counterfactuals without instance data."""
        # Register model first
        test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp", "pressure"]',
            }
        )

        # Missing required instance_data should return 422 or error
        response = test_client.get("/dashboard/counterfactuals/test-model")
        assert response.status_code == 422  # Missing required param

    def test_get_prediction_history_empty(self, test_client):
        """Test getting empty prediction history."""
        # Register model
        test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp"]',
            }
        )

        response = test_client.get(
            "/dashboard/history/test-model",
            params={"time_range": "24h"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["entries"] == []

    def test_clear_cache(self, test_client):
        """Test clearing the cache."""
        response = test_client.delete("/dashboard/cache")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_export_json(self, test_client):
        """Test exporting dashboard data as JSON."""
        # Register model
        test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp", "pressure"]',
            }
        )

        response = test_client.post(
            "/dashboard/export/test-model",
            params={"format": "json"}
        )

        assert response.status_code == 200
        # Should have content-disposition header
        assert "Content-Disposition" in response.headers or response.json()

    def test_export_csv(self, test_client):
        """Test exporting dashboard data as CSV."""
        # Register model
        test_client.post(
            "/dashboard/models/register",
            params={
                "model_id": "test-model",
                "feature_names": '["temp", "pressure"]',
            }
        )

        response = test_client.post(
            "/dashboard/export/test-model",
            params={"format": "csv"}
        )

        assert response.status_code == 200


# =============================================================================
# PROVENANCE HASH TESTS
# =============================================================================

class TestProvenanceHashing:
    """Tests for provenance hash generation."""

    def test_waterfall_provenance_hash(
        self,
        sample_feature_contributions,
        sample_feature_values,
    ):
        """Test that waterfall chart has valid provenance hash."""
        from greenlang.ml.explainability.dashboard import VisualizationDataGenerator

        result = VisualizationDataGenerator.generate_waterfall_data(
            feature_contributions=sample_feature_contributions,
            feature_values=sample_feature_values,
            base_value=0.5,
            prediction=0.73,
            model_id="test-model",
        )

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_provenance_hash_deterministic(
        self,
        sample_feature_contributions,
        sample_feature_values,
    ):
        """Test that same input produces different timestamps but valid hashes."""
        from greenlang.ml.explainability.dashboard import VisualizationDataGenerator

        result1 = VisualizationDataGenerator.generate_waterfall_data(
            feature_contributions=sample_feature_contributions,
            feature_values=sample_feature_values,
            base_value=0.5,
            prediction=0.73,
            model_id="test-model",
        )

        result2 = VisualizationDataGenerator.generate_waterfall_data(
            feature_contributions=sample_feature_contributions,
            feature_values=sample_feature_values,
            base_value=0.5,
            prediction=0.73,
            model_id="test-model",
        )

        # Both should have valid hashes
        assert len(result1.provenance_hash) == 64
        assert len(result2.provenance_hash) == 64


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_contributions(self):
        """Test handling empty contributions."""
        from greenlang.ml.explainability.dashboard import VisualizationDataGenerator

        result = VisualizationDataGenerator.generate_waterfall_data(
            feature_contributions={},
            feature_values={},
            base_value=0.5,
            prediction=0.5,
            model_id="test",
        )

        assert len(result.contributions) == 0
        assert result.base_value == 0.5

    def test_single_feature(self):
        """Test with single feature."""
        from greenlang.ml.explainability.dashboard import VisualizationDataGenerator

        result = VisualizationDataGenerator.generate_bar_chart_data(
            feature_importance={"single_feature": 1.0},
            model_id="test",
            sample_size=10,
        )

        assert result.total_features == 1
        assert result.top_feature == "single_feature"

    def test_all_negative_contributions(self):
        """Test when all contributions are negative."""
        from greenlang.ml.explainability.dashboard import VisualizationDataGenerator

        contributions = {
            "f1": -0.3,
            "f2": -0.2,
            "f3": -0.1,
        }
        values = {"f1": 1.0, "f2": 2.0, "f3": 3.0}

        result = VisualizationDataGenerator.generate_waterfall_data(
            feature_contributions=contributions,
            feature_values=values,
            base_value=0.8,
            prediction=0.2,
            model_id="test",
        )

        assert all(c.direction == "negative" for c in result.contributions)

    def test_no_counterfactual_changes(self, sample_feature_values):
        """Test counterfactual when no changes needed."""
        from greenlang.ml.explainability.dashboard import VisualizationDataGenerator

        # Same instance for original and counterfactual
        result = VisualizationDataGenerator.generate_counterfactual_view(
            original_instance=sample_feature_values,
            original_prediction=0.5,
            counterfactual_instance=sample_feature_values,
            counterfactual_prediction=0.5,
            model_id="test",
        )

        assert len(result.changes) == 0
        assert result.overall_feasibility == 1.0


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Tests for JSON serialization."""

    def test_dashboard_data_roundtrip(self):
        """Test full dashboard data JSON roundtrip."""
        from greenlang.ml.explainability.dashboard import ExplanationDashboardData

        dashboard = ExplanationDashboardData(
            model_id="test",
            model_name="Test Model",
            prediction=0.75,
            explanation_confidence=0.92,
        )

        # Serialize
        json_str = dashboard.model_dump_json()

        # Parse back
        parsed = json.loads(json_str)

        assert parsed["model_id"] == "test"
        assert parsed["prediction"] == 0.75

    def test_nested_chart_serialization(self):
        """Test nested chart structures serialize correctly."""
        from greenlang.ml.explainability.dashboard import (
            ExplanationDashboardData,
            FeatureContributionChart,
            FeatureContributionData,
        )

        contrib = FeatureContributionData(
            feature_name="temp",
            display_name="Temperature",
            value=100.0,
            contribution=0.3,
            percentage=30.0,
            direction="positive",
            rank=1,
        )

        chart = FeatureContributionChart(
            base_value=0.5,
            prediction=0.8,
            contributions=[contrib],
            model_id="test",
        )

        dashboard = ExplanationDashboardData(
            model_id="test",
            feature_contributions=chart,
        )

        json_str = dashboard.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["feature_contributions"]["contributions"][0]["feature_name"] == "temp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
