# -*- coding: utf-8 -*-
"""
Unit tests for Weights & Biases Integration module.

Tests cover all components of the W&B integration including:
- WandBConfig configuration
- WandBExperimentTracker
- ProcessHeatRunConfig
- WandBSweepManager
- WandBAlerting
- WandBReportGenerator
- WandBCacheManager
- WandBMLflowBridge

All tests mock the W&B API to avoid requiring actual W&B credentials.
"""

import json
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_wandb():
    """Mock wandb module."""
    with patch.dict('sys.modules', {'wandb': MagicMock()}):
        import wandb

        # Configure mock
        wandb.init = MagicMock(return_value=MagicMock(
            id="test_run_id",
            name="test_run_name",
            config=MagicMock(update=MagicMock()),
            log=MagicMock(),
            log_artifact=MagicMock(),
        ))
        wandb.finish = MagicMock()
        wandb.login = MagicMock()
        wandb.log = MagicMock()
        wandb.Artifact = MagicMock()
        wandb.Table = MagicMock()
        wandb.sweep = MagicMock(return_value="sweep_123")
        wandb.agent = MagicMock()
        wandb.Api = MagicMock()

        yield wandb


@pytest.fixture
def wandb_config(temp_dir):
    """Create a WandBConfig instance for testing."""
    from greenlang.ml.mlops.wandb_integration import WandBConfig

    return WandBConfig(
        project="test-project",
        entity="test-entity",
        dir=os.path.join(temp_dir, "wandb"),
        cache_dir=os.path.join(temp_dir, "cache"),
        offline_mode=True,
        enable_caching=True,
        enable_provenance=True
    )


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing."""
    return {
        "loss": 0.1,
        "accuracy": 0.95,
        "rmse": 0.05,
        "r2": 0.92
    }


@pytest.fixture
def sample_hyperparameters():
    """Sample hyperparameters for testing."""
    return {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100,
        "n_estimators": 200
    }


# =============================================================================
# WandBConfig Tests
# =============================================================================

class TestWandBConfig:
    """Tests for WandBConfig."""

    def test_default_config(self, temp_dir):
        """Test default configuration values."""
        from greenlang.ml.mlops.wandb_integration import WandBConfig

        config = WandBConfig(
            dir=os.path.join(temp_dir, "wandb"),
            cache_dir=os.path.join(temp_dir, "cache")
        )

        assert config.project == "greenlang-process-heat"
        assert config.entity is None
        assert config.run_name_prefix == "greenlang"
        assert config.log_frequency == 100
        assert config.enable_caching is True
        assert config.enable_provenance is True

    def test_custom_config(self, temp_dir):
        """Test custom configuration values."""
        from greenlang.ml.mlops.wandb_integration import WandBConfig

        config = WandBConfig(
            project="custom-project",
            entity="custom-entity",
            run_name_prefix="custom",
            log_frequency=50,
            offline_mode=True,
            dir=os.path.join(temp_dir, "wandb"),
            cache_dir=os.path.join(temp_dir, "cache")
        )

        assert config.project == "custom-project"
        assert config.entity == "custom-entity"
        assert config.run_name_prefix == "custom"
        assert config.log_frequency == 50
        assert config.offline_mode is True

    def test_get_api_key_from_config(self, temp_dir):
        """Test getting API key from config."""
        from greenlang.ml.mlops.wandb_integration import WandBConfig

        config = WandBConfig(
            api_key="test_api_key",
            dir=os.path.join(temp_dir, "wandb"),
            cache_dir=os.path.join(temp_dir, "cache")
        )

        assert config.get_api_key() == "test_api_key"

    def test_get_api_key_from_env(self, temp_dir):
        """Test getting API key from environment."""
        from greenlang.ml.mlops.wandb_integration import WandBConfig

        with patch.dict(os.environ, {"WANDB_API_KEY": "env_api_key"}):
            config = WandBConfig(
                dir=os.path.join(temp_dir, "wandb"),
                cache_dir=os.path.join(temp_dir, "cache")
            )

            assert config.get_api_key() == "env_api_key"

    def test_default_tags(self, temp_dir):
        """Test default tags are set."""
        from greenlang.ml.mlops.wandb_integration import WandBConfig

        config = WandBConfig(
            dir=os.path.join(temp_dir, "wandb"),
            cache_dir=os.path.join(temp_dir, "cache")
        )

        assert "process-heat" in config.default_tags
        assert "greenlang" in config.default_tags
        assert "zero-hallucination" in config.default_tags

    def test_directory_validation(self, temp_dir):
        """Test that directories are created."""
        from greenlang.ml.mlops.wandb_integration import WandBConfig

        wandb_dir = os.path.join(temp_dir, "new_wandb")
        cache_dir = os.path.join(temp_dir, "new_cache")

        config = WandBConfig(
            dir=wandb_dir,
            cache_dir=cache_dir
        )

        assert os.path.exists(config.dir)
        assert os.path.exists(config.cache_dir)


# =============================================================================
# ProcessHeatRunConfig Tests
# =============================================================================

class TestProcessHeatRunConfig:
    """Tests for ProcessHeatRunConfig."""

    def test_create_run_config(self):
        """Test creating a run configuration."""
        from greenlang.ml.mlops.wandb_integration import (
            ProcessHeatRunConfig,
            AgentType
        )

        config = ProcessHeatRunConfig(
            agent_type=AgentType.GL_008_FUEL,
            model_name="fuel_emission_model"
        )

        assert config.agent_type == AgentType.GL_008_FUEL
        assert config.model_name == "fuel_emission_model"
        assert config.agent_version == "1.0.0"
        assert config.model_type == "sklearn"

    def test_get_default_hyperparameters_carbon(self):
        """Test getting default hyperparameters for Carbon agent."""
        from greenlang.ml.mlops.wandb_integration import (
            ProcessHeatRunConfig,
            AgentType
        )

        params = ProcessHeatRunConfig.get_default_hyperparameters(
            AgentType.GL_001_CARBON
        )

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "emission_factor_precision" in params

    def test_get_default_hyperparameters_fuel(self):
        """Test getting default hyperparameters for Fuel agent."""
        from greenlang.ml.mlops.wandb_integration import (
            ProcessHeatRunConfig,
            AgentType
        )

        params = ProcessHeatRunConfig.get_default_hyperparameters(
            AgentType.GL_008_FUEL
        )

        assert params["n_estimators"] == 200
        assert params["max_depth"] == 15
        assert params["fuel_type_embeddings"] is True
        assert params["thermal_efficiency_min"] == 0.80

    def test_get_default_hyperparameters_combustion(self):
        """Test getting default hyperparameters for Combustion agent."""
        from greenlang.ml.mlops.wandb_integration import (
            ProcessHeatRunConfig,
            AgentType
        )

        params = ProcessHeatRunConfig.get_default_hyperparameters(
            AgentType.GL_010_COMBUSTION
        )

        assert "hidden_layers" in params
        assert "dropout_rate" in params
        assert params["epochs"] == 100

    def test_get_sweep_config(self):
        """Test getting sweep configuration."""
        from greenlang.ml.mlops.wandb_integration import (
            ProcessHeatRunConfig,
            AgentType
        )

        sweep_config = ProcessHeatRunConfig.get_sweep_config(
            AgentType.GL_001_CARBON
        )

        assert "parameters" in sweep_config
        assert "n_estimators" in sweep_config["parameters"]
        assert "max_depth" in sweep_config["parameters"]
        assert "learning_rate" in sweep_config["parameters"]

    def test_sweep_settings(self):
        """Test sweep settings."""
        from greenlang.ml.mlops.wandb_integration import (
            ProcessHeatRunConfig,
            AgentType,
            SweepMethod
        )

        config = ProcessHeatRunConfig(
            agent_type=AgentType.GL_017_PREDICTION,
            model_name="prediction_model",
            sweep_enabled=True,
            sweep_method=SweepMethod.BAYES,
            sweep_count=100
        )

        assert config.sweep_enabled is True
        assert config.sweep_method == SweepMethod.BAYES
        assert config.sweep_count == 100


# =============================================================================
# WandBCacheManager Tests
# =============================================================================

class TestWandBCacheManager:
    """Tests for WandBCacheManager."""

    def test_cache_set_and_get(self, temp_dir):
        """Test setting and getting cache values."""
        from greenlang.ml.mlops.wandb_integration import WandBCacheManager

        cache = WandBCacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            ttl_hours=24
        )

        cache.set("test_key", {"value": 42})
        result = cache.get("test_key")

        assert result == {"value": 42}

    def test_cache_miss(self, temp_dir):
        """Test cache miss returns None."""
        from greenlang.ml.mlops.wandb_integration import WandBCacheManager

        cache = WandBCacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            ttl_hours=24
        )

        result = cache.get("nonexistent_key")

        assert result is None

    def test_cache_expiration(self, temp_dir):
        """Test cache expiration."""
        from greenlang.ml.mlops.wandb_integration import WandBCacheManager

        cache = WandBCacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            ttl_hours=0  # Expire immediately
        )

        cache.set("test_key", {"value": 42})

        # Force expiration by modifying timestamp
        cache._memory_cache["test_key"] = (
            {"value": 42},
            datetime.now(timezone.utc) - timedelta(hours=2)
        )

        result = cache.get("test_key")

        assert result is None

    def test_get_or_compute(self, temp_dir):
        """Test get_or_compute with caching."""
        from greenlang.ml.mlops.wandb_integration import WandBCacheManager

        cache = WandBCacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            ttl_hours=24
        )

        compute_count = [0]

        def expensive_computation():
            compute_count[0] += 1
            return {"result": compute_count[0]}

        # First call should compute
        result1 = cache.get_or_compute(
            {"input": "test"},
            expensive_computation
        )

        # Second call should use cache
        result2 = cache.get_or_compute(
            {"input": "test"},
            expensive_computation
        )

        assert result1 == {"result": 1}
        assert result2 == {"result": 1}
        assert compute_count[0] == 1  # Only computed once

    def test_clear_expired(self, temp_dir):
        """Test clearing expired cache entries."""
        from greenlang.ml.mlops.wandb_integration import WandBCacheManager

        cache = WandBCacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            ttl_hours=1
        )

        # Add entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Force one to expire
        cache._memory_cache["key1"] = (
            "value1",
            datetime.now(timezone.utc) - timedelta(hours=2)
        )

        cleared = cache.clear_expired()

        assert cleared >= 1

    def test_get_stats(self, temp_dir):
        """Test getting cache statistics."""
        from greenlang.ml.mlops.wandb_integration import WandBCacheManager

        cache = WandBCacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            ttl_hours=24
        )

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()

        assert "memory_entries" in stats
        assert "disk_entries" in stats
        assert "disk_size_mb" in stats
        assert stats["memory_entries"] >= 2


# =============================================================================
# WandBExperimentTracker Tests
# =============================================================================

class TestWandBExperimentTracker:
    """Tests for WandBExperimentTracker."""

    def test_init_tracker(self, wandb_config, mock_wandb):
        """Test initializing experiment tracker."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        tracker = WandBExperimentTracker(config=wandb_config)

        assert tracker.config.project == "test-project"
        assert tracker.config.entity == "test-entity"

    def test_compute_sha256_string(self, wandb_config, mock_wandb):
        """Test SHA-256 computation for strings."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        tracker = WandBExperimentTracker(config=wandb_config)

        hash1 = tracker._compute_sha256("test")
        hash2 = tracker._compute_sha256("test")
        hash3 = tracker._compute_sha256("different")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64

    def test_compute_sha256_dict(self, wandb_config, mock_wandb):
        """Test SHA-256 computation for dictionaries."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        tracker = WandBExperimentTracker(config=wandb_config)

        data = {"key": "value", "number": 42}
        hash1 = tracker._compute_sha256(data)
        hash2 = tracker._compute_sha256(data)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_generate_run_name(self, wandb_config, mock_wandb):
        """Test run name generation."""
        from greenlang.ml.mlops.wandb_integration import (
            WandBExperimentTracker,
            AgentType
        )

        tracker = WandBExperimentTracker(config=wandb_config)

        name = tracker._generate_run_name("test_run", AgentType.GL_008_FUEL)

        assert "greenlang" in name
        assert "GL-008-Fuel" in name
        assert "test_run" in name

    def test_detect_framework_sklearn(self, wandb_config, mock_wandb):
        """Test framework detection for sklearn."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        tracker = WandBExperimentTracker(config=wandb_config)

        # Create a mock sklearn model
        class MockSklearnModel:
            __module__ = "sklearn.ensemble.forest"

        framework = tracker._detect_framework(MockSklearnModel())

        assert framework == "sklearn"

    def test_detect_framework_pytorch(self, wandb_config, mock_wandb):
        """Test framework detection for PyTorch."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        tracker = WandBExperimentTracker(config=wandb_config)

        # Create a mock PyTorch model
        class MockPyTorchModel:
            __module__ = "torch.nn.modules.linear"

        framework = tracker._detect_framework(MockPyTorchModel())

        assert framework == "pytorch"

    def test_list_runs(self, wandb_config, mock_wandb):
        """Test listing runs."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        tracker = WandBExperimentTracker(config=wandb_config)

        runs = tracker.list_runs()

        assert isinstance(runs, list)


# =============================================================================
# WandBSweepManager Tests
# =============================================================================

class TestWandBSweepManager:
    """Tests for WandBSweepManager."""

    def test_init_sweep_manager(self, wandb_config, mock_wandb):
        """Test initializing sweep manager."""
        from greenlang.ml.mlops.wandb_integration import WandBSweepManager

        manager = WandBSweepManager(config=wandb_config)

        assert manager.config.project == "test-project"

    def test_create_sweep(self, wandb_config, mock_wandb):
        """Test creating a sweep."""
        from greenlang.ml.mlops.wandb_integration import (
            WandBSweepManager,
            AgentType,
            SweepMethod
        )

        with patch('greenlang.ml.mlops.wandb_integration.WandBSweepManager._initialize_wandb') as mock_init:
            mock_init.return_value = True
            manager = WandBSweepManager(config=wandb_config)
            manager._wandb = mock_wandb

            sweep_id = manager.create_sweep(
                agent_type=AgentType.GL_008_FUEL,
                method=SweepMethod.BAYES,
                metric="rmse",
                goal="minimize"
            )

            # Sweep should be created (returns mocked sweep_id)
            assert sweep_id == "sweep_123"


# =============================================================================
# WandBAlerting Tests
# =============================================================================

class TestWandBAlerting:
    """Tests for WandBAlerting."""

    def test_add_alert(self, wandb_config):
        """Test adding an alert."""
        from greenlang.ml.mlops.wandb_integration import (
            WandBAlerting,
            AlertLevel
        )

        alerting = WandBAlerting(config=wandb_config)

        alerting.add_alert(
            name="high_loss",
            metric="loss",
            condition="above",
            threshold=0.5,
            level=AlertLevel.WARNING
        )

        assert "high_loss" in alerting.list_alerts()

    def test_check_alerts_triggered(self, wandb_config):
        """Test checking alerts - triggered."""
        from greenlang.ml.mlops.wandb_integration import WandBAlerting

        alerting = WandBAlerting(config=wandb_config)

        alerting.add_alert(
            name="high_loss",
            metric="loss",
            condition="above",
            threshold=0.5
        )

        triggered = alerting.check_alerts("run_123", {"loss": 0.6})

        assert "high_loss" in triggered

    def test_check_alerts_not_triggered(self, wandb_config):
        """Test checking alerts - not triggered."""
        from greenlang.ml.mlops.wandb_integration import WandBAlerting

        alerting = WandBAlerting(config=wandb_config)

        alerting.add_alert(
            name="high_loss",
            metric="loss",
            condition="above",
            threshold=0.5
        )

        triggered = alerting.check_alerts("run_123", {"loss": 0.3})

        assert "high_loss" not in triggered

    def test_alert_cooldown(self, wandb_config):
        """Test alert cooldown."""
        from greenlang.ml.mlops.wandb_integration import WandBAlerting

        alerting = WandBAlerting(config=wandb_config)

        alerting.add_alert(
            name="high_loss",
            metric="loss",
            condition="above",
            threshold=0.5,
            cooldown_minutes=30
        )

        # First alert should trigger
        triggered1 = alerting.check_alerts("run_123", {"loss": 0.6})

        # Second alert within cooldown should not trigger
        triggered2 = alerting.check_alerts("run_123", {"loss": 0.7})

        assert "high_loss" in triggered1
        assert "high_loss" not in triggered2

    def test_remove_alert(self, wandb_config):
        """Test removing an alert."""
        from greenlang.ml.mlops.wandb_integration import WandBAlerting

        alerting = WandBAlerting(config=wandb_config)

        alerting.add_alert(
            name="test_alert",
            metric="loss",
            condition="above",
            threshold=0.5
        )

        assert alerting.remove_alert("test_alert") is True
        assert "test_alert" not in alerting.list_alerts()

    def test_alert_conditions(self, wandb_config):
        """Test different alert conditions."""
        from greenlang.ml.mlops.wandb_integration import WandBAlerting

        alerting = WandBAlerting(config=wandb_config)

        # Test 'below' condition
        alerting.add_alert(
            name="low_accuracy",
            metric="accuracy",
            condition="below",
            threshold=0.9
        )

        triggered = alerting.check_alerts("run_123", {"accuracy": 0.85})

        assert "low_accuracy" in triggered


# =============================================================================
# WandBReportGenerator Tests
# =============================================================================

class TestWandBReportGenerator:
    """Tests for WandBReportGenerator."""

    def test_init_report_generator(self, wandb_config, mock_wandb):
        """Test initializing report generator."""
        from greenlang.ml.mlops.wandb_integration import WandBReportGenerator

        generator = WandBReportGenerator(config=wandb_config)

        assert generator.config.project == "test-project"

    def test_generate_summary_report(self, wandb_config, temp_dir, mock_wandb):
        """Test generating summary report."""
        from greenlang.ml.mlops.wandb_integration import WandBReportGenerator

        generator = WandBReportGenerator(config=wandb_config)

        # Mock compare_runs to return test data
        generator.compare_runs = MagicMock(return_value={
            "run_1": {
                "name": "Run 1",
                "state": "finished",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {"loss": 0.1, "accuracy": 0.95}
            },
            "run_2": {
                "name": "Run 2",
                "state": "finished",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {"loss": 0.08, "accuracy": 0.97}
            }
        })

        output_path = os.path.join(temp_dir, "report.md")
        report = generator.generate_summary_report(
            ["run_1", "run_2"],
            output_path=output_path
        )

        assert "W&B Experiment Summary Report" in report
        assert "Run 1" in report
        assert "Run 2" in report
        assert os.path.exists(output_path)

    def test_select_best_model_minimize(self, wandb_config, mock_wandb):
        """Test selecting best model (minimize)."""
        from greenlang.ml.mlops.wandb_integration import WandBReportGenerator

        generator = WandBReportGenerator(config=wandb_config)

        # Mock compare_runs
        generator.compare_runs = MagicMock(return_value={
            "run_1": {
                "name": "Run 1",
                "state": "finished",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {"rmse": 0.1}
            },
            "run_2": {
                "name": "Run 2",
                "state": "finished",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {"rmse": 0.05}
            }
        })

        best = generator.select_best_model(
            ["run_1", "run_2"],
            metric="rmse",
            goal="minimize"
        )

        assert best is not None
        assert best["run_id"] == "run_2"
        assert best["value"] == 0.05

    def test_select_best_model_maximize(self, wandb_config, mock_wandb):
        """Test selecting best model (maximize)."""
        from greenlang.ml.mlops.wandb_integration import WandBReportGenerator

        generator = WandBReportGenerator(config=wandb_config)

        # Mock compare_runs
        generator.compare_runs = MagicMock(return_value={
            "run_1": {
                "name": "Run 1",
                "state": "finished",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {"accuracy": 0.95}
            },
            "run_2": {
                "name": "Run 2",
                "state": "finished",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {"accuracy": 0.98}
            }
        })

        best = generator.select_best_model(
            ["run_1", "run_2"],
            metric="accuracy",
            goal="maximize"
        )

        assert best is not None
        assert best["run_id"] == "run_2"
        assert best["value"] == 0.98


# =============================================================================
# RunInfo Tests
# =============================================================================

class TestRunInfo:
    """Tests for RunInfo dataclass."""

    def test_create_run_info(self):
        """Test creating RunInfo."""
        from greenlang.ml.mlops.wandb_integration import RunInfo, RunStatus

        run_info = RunInfo(
            run_id="test_run_id",
            run_name="test_run",
            project="test_project",
            entity="test_entity",
            config={"lr": 0.01},
            tags=["test"],
            start_time=datetime.now(timezone.utc)
        )

        assert run_info.run_id == "test_run_id"
        assert run_info.status == RunStatus.RUNNING
        assert run_info.config["lr"] == 0.01

    def test_run_info_to_dict(self):
        """Test converting RunInfo to dict."""
        from greenlang.ml.mlops.wandb_integration import RunInfo

        run_info = RunInfo(
            run_id="test_run_id",
            run_name="test_run",
            project="test_project",
            entity="test_entity",
            config={"lr": 0.01},
            tags=["test"],
            start_time=datetime.now(timezone.utc)
        )

        data = run_info.to_dict()

        assert data["run_id"] == "test_run_id"
        assert data["project"] == "test_project"
        assert "start_time" in data


# =============================================================================
# SweepInfo Tests
# =============================================================================

class TestSweepInfo:
    """Tests for SweepInfo dataclass."""

    def test_create_sweep_info(self):
        """Test creating SweepInfo."""
        from greenlang.ml.mlops.wandb_integration import SweepInfo, SweepMethod

        sweep_info = SweepInfo(
            sweep_id="sweep_123",
            sweep_name="test_sweep",
            project="test_project",
            entity="test_entity",
            method=SweepMethod.BAYES,
            metric="rmse",
            goal="minimize",
            config={"parameters": {"lr": {"min": 0.001, "max": 0.1}}}
        )

        assert sweep_info.sweep_id == "sweep_123"
        assert sweep_info.method == SweepMethod.BAYES
        assert sweep_info.status == "created"


# =============================================================================
# AgentType Enum Tests
# =============================================================================

class TestAgentType:
    """Tests for AgentType enum."""

    def test_all_agent_types(self):
        """Test all agent types are defined."""
        from greenlang.ml.mlops.wandb_integration import AgentType

        assert AgentType.GL_001_CARBON.value == "GL-001-Carbon"
        assert AgentType.GL_008_FUEL.value == "GL-008-Fuel"
        assert AgentType.GL_010_COMBUSTION.value == "GL-010-Combustion"
        assert AgentType.GL_017_PREDICTION.value == "GL-017-Prediction"
        assert AgentType.GL_020_SAFETY.value == "GL-020-Safety"

    def test_agent_type_count(self):
        """Test all 20 agents are defined."""
        from greenlang.ml.mlops.wandb_integration import AgentType

        # Should have GL-001 through GL-020
        assert len(AgentType) == 20


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_experiment_tracker(self, mock_wandb, temp_dir):
        """Test create_experiment_tracker function."""
        from greenlang.ml.mlops.wandb_integration import create_experiment_tracker

        with patch('greenlang.ml.mlops.wandb_integration.WandBConfig') as mock_config:
            mock_config.return_value = MagicMock(
                project="test-project",
                entity="test-entity",
                get_api_key=MagicMock(return_value=None),
                offline_mode=True,
                enable_caching=False,
                enable_provenance=True,
                cache_dir=os.path.join(temp_dir, "cache"),
                cache_ttl_hours=24
            )

            tracker = create_experiment_tracker(
                project="test-project",
                entity="test-entity",
                offline=True
            )

            assert tracker is not None


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_cache_thread_safety(self, temp_dir):
        """Test cache is thread safe."""
        from greenlang.ml.mlops.wandb_integration import WandBCacheManager

        cache = WandBCacheManager(
            cache_dir=os.path.join(temp_dir, "cache"),
            ttl_hours=24
        )

        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_alerting_thread_safety(self, wandb_config):
        """Test alerting is thread safe."""
        from greenlang.ml.mlops.wandb_integration import WandBAlerting

        alerting = WandBAlerting(config=wandb_config)

        errors = []

        def add_alerts():
            try:
                for i in range(50):
                    alerting.add_alert(
                        name=f"alert_{threading.current_thread().name}_{i}",
                        metric="loss",
                        condition="above",
                        threshold=0.5
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_alerts, name=f"thread_{i}")
            for i in range(4)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Integration Tests (with mocked W&B)
# =============================================================================

class TestWandBIntegration:
    """Integration tests with mocked W&B."""

    def test_full_experiment_workflow(self, wandb_config, mock_wandb, sample_metrics, sample_hyperparameters):
        """Test full experiment workflow."""
        from greenlang.ml.mlops.wandb_integration import (
            WandBExperimentTracker,
            AgentType
        )

        with patch('greenlang.ml.mlops.wandb_integration.WandBExperimentTracker._initialize_wandb') as mock_init:
            mock_init.return_value = True
            tracker = WandBExperimentTracker(config=wandb_config)
            tracker._wandb = mock_wandb
            tracker._initialized = True

            # The workflow should complete without errors
            # Note: Context manager won't work fully with mocks, so we test components

            # Test logging metrics
            tracker._current_run = mock_wandb.init()
            tracker._current_run_info = MagicMock()
            tracker._current_run_info.metrics = {}
            tracker._current_run_info.config = {}

            tracker.log_metrics(sample_metrics)
            tracker.log_hyperparameters(sample_hyperparameters)

            # Verify logging was called
            assert tracker._current_run.log.called


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_metrics(self, wandb_config, mock_wandb):
        """Test logging empty metrics."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        with patch('greenlang.ml.mlops.wandb_integration.WandBExperimentTracker._initialize_wandb') as mock_init:
            mock_init.return_value = True
            tracker = WandBExperimentTracker(config=wandb_config)
            tracker._wandb = mock_wandb
            tracker._initialized = True
            tracker._current_run = mock_wandb.init()
            tracker._current_run_info = MagicMock(metrics={}, config={})

            # Should not raise an error
            tracker.log_metrics({})

    def test_none_api_key(self, temp_dir):
        """Test handling None API key."""
        from greenlang.ml.mlops.wandb_integration import WandBConfig

        config = WandBConfig(
            api_key=None,
            dir=os.path.join(temp_dir, "wandb"),
            cache_dir=os.path.join(temp_dir, "cache")
        )

        # Should return None if not in env
        with patch.dict(os.environ, {}, clear=True):
            # Remove WANDB_API_KEY if present
            os.environ.pop("WANDB_API_KEY", None)
            assert config.get_api_key() is None

    def test_special_characters_in_run_name(self, wandb_config, mock_wandb):
        """Test run name with special characters."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        tracker = WandBExperimentTracker(config=wandb_config)

        name = tracker._generate_run_name("test/run:with@special#chars")

        # Name should be generated without errors
        assert "test/run:with@special#chars" in name

    def test_large_metrics_dict(self, wandb_config, mock_wandb):
        """Test logging large metrics dictionary."""
        from greenlang.ml.mlops.wandb_integration import WandBExperimentTracker

        with patch('greenlang.ml.mlops.wandb_integration.WandBExperimentTracker._initialize_wandb') as mock_init:
            mock_init.return_value = True
            tracker = WandBExperimentTracker(config=wandb_config)
            tracker._wandb = mock_wandb
            tracker._initialized = True
            tracker._current_run = mock_wandb.init()
            tracker._current_run_info = MagicMock(metrics={}, config={})

            # Create large metrics dict
            large_metrics = {f"metric_{i}": float(i) * 0.01 for i in range(1000)}

            # Should not raise an error
            tracker.log_metrics(large_metrics)


# =============================================================================
# WandBMLflowBridge Tests
# =============================================================================

class TestWandBMLflowBridge:
    """Tests for WandBMLflowBridge."""

    def test_init_bridge(self, wandb_config, mock_wandb):
        """Test initializing bridge."""
        from greenlang.ml.mlops.wandb_integration import WandBMLflowBridge

        with patch('greenlang.ml.mlops.wandb_integration.WandBExperimentTracker'):
            bridge = WandBMLflowBridge(wandb_config=wandb_config)

            assert bridge.wandb_tracker is not None

    def test_log_metrics_dual(self, wandb_config, mock_wandb):
        """Test logging metrics to both platforms."""
        from greenlang.ml.mlops.wandb_integration import WandBMLflowBridge

        with patch('greenlang.ml.mlops.wandb_integration.WandBExperimentTracker') as mock_tracker:
            mock_tracker_instance = MagicMock()
            mock_tracker.return_value = mock_tracker_instance

            bridge = WandBMLflowBridge(wandb_config=wandb_config)
            bridge.log_metrics({"loss": 0.1})

            mock_tracker_instance.log_metrics.assert_called_once_with(
                {"loss": 0.1},
                step=None
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
