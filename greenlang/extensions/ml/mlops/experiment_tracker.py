# -*- coding: utf-8 -*-
"""
Experiment Tracker Module

This module provides comprehensive experiment tracking for GreenLang ML
development, enabling reproducibility, comparison, and audit trails
for all ML experiments.

Experiment tracking is essential for regulatory compliance, enabling
complete traceability of model development decisions and results.

Example:
    >>> from greenlang.ml.mlops import ExperimentTracker
    >>> tracker = ExperimentTracker()
    >>> with tracker.start_run("emission_model_v2"):
    ...     tracker.log_params({"n_estimators": 100})
    ...     tracker.log_metrics({"rmse": 0.05})
    ...     tracker.log_artifact("feature_importance.json")
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Experiment run status."""
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class ExperimentTrackerConfig(BaseModel):
    """Configuration for experiment tracker."""

    tracking_uri: str = Field(
        default="./experiments",
        description="Tracking storage location"
    )
    experiment_name: str = Field(
        default="greenlang_experiments",
        description="Default experiment name"
    )
    auto_log_system_metrics: bool = Field(
        default=True,
        description="Auto-log system metrics"
    )
    log_git_info: bool = Field(
        default=True,
        description="Log git commit info"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


class RunInfo(BaseModel):
    """Information about an experiment run."""

    run_id: str = Field(
        ...,
        description="Unique run identifier"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Human-readable run name"
    )
    experiment_name: str = Field(
        ...,
        description="Parent experiment name"
    )
    status: RunStatus = Field(
        default=RunStatus.RUNNING,
        description="Run status"
    )
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="Start timestamp"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="End timestamp"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Run parameters"
    )
    metrics: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Run metrics with history"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Run tags"
    )
    artifacts: List[str] = Field(
        default_factory=list,
        description="Artifact paths"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="Provenance hash"
    )


class ExperimentInfo(BaseModel):
    """Information about an experiment."""

    experiment_id: str = Field(
        ...,
        description="Unique experiment identifier"
    )
    experiment_name: str = Field(
        ...,
        description="Experiment name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Experiment description"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    runs: List[str] = Field(
        default_factory=list,
        description="List of run IDs"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Experiment tags"
    )


class ExperimentTracker:
    """
    Experiment Tracker for GreenLang ML development.

    This class provides comprehensive experiment tracking capabilities,
    enabling reproducibility, comparison, and audit trails for all
    ML experiments in the GreenLang ecosystem.

    Key capabilities:
    - Run management (start, end, fail)
    - Parameter and metric logging
    - Artifact storage
    - Experiment comparison
    - Provenance tracking
    - Git integration

    Attributes:
        config: Tracker configuration
        _experiments: Experiment store
        _runs: Run store
        _current_run: Currently active run

    Example:
        >>> tracker = ExperimentTracker(config=ExperimentTrackerConfig(
        ...     experiment_name="emission_models"
        ... ))
        >>> with tracker.start_run("xgboost_v1"):
        ...     tracker.log_params({"max_depth": 6, "n_estimators": 100})
        ...     model = train_model(X, y)
        ...     tracker.log_metrics({"rmse": evaluate(model)})
        ...     tracker.log_model(model)
    """

    def __init__(self, config: Optional[ExperimentTrackerConfig] = None):
        """
        Initialize experiment tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or ExperimentTrackerConfig()
        self._experiments: Dict[str, ExperimentInfo] = {}
        self._runs: Dict[str, RunInfo] = {}
        self._current_run: Optional[RunInfo] = None
        self._mlflow_available = False

        # Create tracking directory
        Path(self.config.tracking_uri).mkdir(parents=True, exist_ok=True)

        # Try to initialize MLflow
        self._initialize_mlflow()

        # Initialize default experiment
        self._get_or_create_experiment(self.config.experiment_name)

        logger.info(
            f"ExperimentTracker initialized: {self.config.experiment_name}"
        )

    def _initialize_mlflow(self) -> None:
        """Initialize MLflow if available."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._mlflow_available = True
            logger.info("MLflow integration enabled")
        except ImportError:
            logger.info("MLflow not available, using local tracking")

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def _generate_experiment_id(self, name: str) -> str:
        """Generate experiment ID from name."""
        return hashlib.sha256(name.encode()).hexdigest()[:12]

    def _get_or_create_experiment(self, name: str) -> ExperimentInfo:
        """Get or create an experiment."""
        if name not in self._experiments:
            exp_id = self._generate_experiment_id(name)
            self._experiments[name] = ExperimentInfo(
                experiment_id=exp_id,
                experiment_name=name
            )
        return self._experiments[name]

    def _calculate_provenance(self, run: RunInfo) -> str:
        """Calculate provenance hash for a run."""
        combined = (
            f"{run.run_id}|{run.experiment_name}|"
            f"{json.dumps(run.params, sort_keys=True)}|"
            f"{json.dumps({k: v[-1] if v else None for k, v in run.metrics.items()}, sort_keys=True)}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit info."""
        try:
            import subprocess

            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            return {"git_commit": commit[:8], "git_branch": branch}
        except Exception:
            return {}

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start a new experiment run.

        Args:
            run_name: Optional run name
            experiment_name: Experiment to log to (default from config)
            tags: Run tags

        Yields:
            RunInfo for the active run

        Example:
            >>> with tracker.start_run("my_experiment"):
            ...     tracker.log_params({"lr": 0.01})
            ...     train_model()
            ...     tracker.log_metrics({"loss": 0.1})
        """
        exp_name = experiment_name or self.config.experiment_name
        experiment = self._get_or_create_experiment(exp_name)

        run_id = self._generate_run_id()
        tags = tags or {}

        # Add git info if enabled
        if self.config.log_git_info:
            tags.update(self._get_git_info())

        run = RunInfo(
            run_id=run_id,
            run_name=run_name,
            experiment_name=exp_name,
            tags=tags
        )

        self._runs[run_id] = run
        self._current_run = run
        experiment.runs.append(run_id)

        logger.info(f"Started run: {run_id} ({run_name or 'unnamed'})")

        # Start MLflow run if available
        mlflow_run = None
        if self._mlflow_available:
            try:
                import mlflow
                mlflow_run = mlflow.start_run(run_name=run_name)
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")

        try:
            yield run
            run.status = RunStatus.FINISHED
        except Exception as e:
            run.status = RunStatus.FAILED
            self.log_params({"error": str(e)})
            raise
        finally:
            run.end_time = datetime.utcnow()
            run.provenance_hash = self._calculate_provenance(run)
            self._current_run = None

            # End MLflow run
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception:
                    pass

            logger.info(
                f"Ended run: {run_id}, status={run.status.value}, "
                f"duration={(run.end_time - run.start_time).total_seconds():.2f}s"
            )

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run.

        Args:
            params: Dictionary of parameters

        Example:
            >>> tracker.log_params({"n_estimators": 100, "max_depth": 6})
        """
        if self._current_run is None:
            logger.warning("No active run, parameters not logged")
            return

        self._current_run.params.update(params)

        if self._mlflow_available:
            try:
                import mlflow
                for key, value in params.items():
                    mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {e}")

        logger.debug(f"Logged params: {list(params.keys())}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number for time series

        Example:
            >>> tracker.log_metrics({"rmse": 0.05, "r2": 0.95})
            >>> # Or with steps
            >>> for epoch in range(10):
            ...     tracker.log_metrics({"loss": loss}, step=epoch)
        """
        if self._current_run is None:
            logger.warning("No active run, metrics not logged")
            return

        timestamp = datetime.utcnow()

        for key, value in metrics.items():
            if key not in self._current_run.metrics:
                self._current_run.metrics[key] = []

            self._current_run.metrics[key].append({
                "value": value,
                "step": step,
                "timestamp": timestamp.isoformat()
            })

        if self._mlflow_available:
            try:
                import mlflow
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

        logger.debug(f"Logged metrics: {list(metrics.keys())}")

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Log an artifact for the current run.

        Args:
            local_path: Path to local file
            artifact_path: Path in artifact storage

        Example:
            >>> tracker.log_artifact("./feature_importance.json")
        """
        if self._current_run is None:
            logger.warning("No active run, artifact not logged")
            return

        self._current_run.artifacts.append(local_path)

        if self._mlflow_available:
            try:
                import mlflow
                mlflow.log_artifact(local_path, artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log artifact to MLflow: {e}")

        logger.debug(f"Logged artifact: {local_path}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model"
    ) -> None:
        """
        Log a model for the current run.

        Args:
            model: Model to log
            artifact_path: Path in artifact storage
        """
        if self._current_run is None:
            logger.warning("No active run, model not logged")
            return

        # Save model locally
        run_dir = Path(self.config.tracking_uri) / self._current_run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        import pickle
        model_path = run_dir / f"{artifact_path}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        self._current_run.artifacts.append(str(model_path))

        if self._mlflow_available:
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(model, artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")

        logger.debug(f"Logged model to {model_path}")

    def log_tags(self, tags: Dict[str, str]) -> None:
        """Log tags for the current run."""
        if self._current_run is None:
            return

        self._current_run.tags.update(tags)

        if self._mlflow_available:
            try:
                import mlflow
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            except Exception:
                pass

    def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run information by ID."""
        return self._runs.get(run_id)

    def get_current_run(self) -> Optional[RunInfo]:
        """Get the currently active run."""
        return self._current_run

    def list_experiments(self) -> List[ExperimentInfo]:
        """List all experiments."""
        return list(self._experiments.values())

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        status: Optional[RunStatus] = None
    ) -> List[RunInfo]:
        """
        List runs with optional filtering.

        Args:
            experiment_name: Filter by experiment
            status: Filter by status

        Returns:
            List of matching runs
        """
        runs = list(self._runs.values())

        if experiment_name:
            runs = [r for r in runs if r.experiment_name == experiment_name]

        if status:
            runs = [r for r in runs if r.status == status]

        return sorted(runs, key=lambda r: r.start_time, reverse=True)

    def compare_runs(
        self,
        run_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison dictionary
        """
        comparison = {}

        for run_id in run_ids:
            run = self._runs.get(run_id)
            if run:
                # Get latest metrics
                latest_metrics = {
                    k: v[-1]["value"] if v else None
                    for k, v in run.metrics.items()
                }

                comparison[run_id] = {
                    "name": run.run_name,
                    "status": run.status.value,
                    "params": run.params,
                    "metrics": latest_metrics,
                    "duration": (
                        (run.end_time - run.start_time).total_seconds()
                        if run.end_time else None
                    )
                }

        return comparison

    def search_runs(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        filter_metrics: Optional[Dict[str, tuple]] = None,
        order_by: Optional[str] = None
    ) -> List[RunInfo]:
        """
        Search runs with filters.

        Args:
            filter_params: Filter by parameter values
            filter_metrics: Filter by metric ranges (key: (min, max))
            order_by: Metric to order by

        Returns:
            Filtered and sorted runs
        """
        runs = list(self._runs.values())

        # Filter by params
        if filter_params:
            runs = [
                r for r in runs
                if all(r.params.get(k) == v for k, v in filter_params.items())
            ]

        # Filter by metrics
        if filter_metrics:
            filtered = []
            for run in runs:
                matches = True
                for key, (min_val, max_val) in filter_metrics.items():
                    if key in run.metrics and run.metrics[key]:
                        value = run.metrics[key][-1]["value"]
                        if not (min_val <= value <= max_val):
                            matches = False
                            break
                    else:
                        matches = False
                        break
                if matches:
                    filtered.append(run)
            runs = filtered

        # Order by metric
        if order_by and runs:
            runs = sorted(
                runs,
                key=lambda r: (
                    r.metrics.get(order_by, [{}])[-1].get("value", float("inf"))
                )
            )

        return runs


# Unit test stubs
class TestExperimentTracker:
    """Unit tests for ExperimentTracker."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        tracker = ExperimentTracker()
        assert tracker.config.experiment_name == "greenlang_experiments"

    def test_start_end_run(self):
        """Test run lifecycle."""
        tracker = ExperimentTracker()

        with tracker.start_run("test_run") as run:
            assert run.status == RunStatus.RUNNING
            tracker.log_params({"a": 1})
            tracker.log_metrics({"loss": 0.5})

        assert run.status == RunStatus.FINISHED
        assert run.end_time is not None
        assert run.provenance_hash is not None

    def test_log_params(self):
        """Test parameter logging."""
        tracker = ExperimentTracker()

        with tracker.start_run("test"):
            tracker.log_params({"lr": 0.01, "epochs": 10})
            run = tracker.get_current_run()
            assert run.params["lr"] == 0.01

    def test_log_metrics_with_steps(self):
        """Test metric logging with steps."""
        tracker = ExperimentTracker()

        with tracker.start_run("test"):
            for i in range(5):
                tracker.log_metrics({"loss": 1.0 / (i + 1)}, step=i)

            run = tracker.get_current_run()
            assert len(run.metrics["loss"]) == 5

    def test_compare_runs(self):
        """Test run comparison."""
        tracker = ExperimentTracker()
        run_ids = []

        for i in range(3):
            with tracker.start_run(f"run_{i}"):
                tracker.log_params({"n": i})
                tracker.log_metrics({"score": i * 0.1})
                run_ids.append(tracker.get_current_run().run_id)

        comparison = tracker.compare_runs(run_ids)
        assert len(comparison) == 3
