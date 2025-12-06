# -*- coding: utf-8 -*-
"""
MLflow Integration Module for GreenLang Process Heat Agents

This module provides comprehensive MLflow integration for the GreenLang
Process Heat Agent ecosystem, enabling:
- Automatic experiment tracking with SHA-256 provenance
- Model versioning with staging (Development/Staging/Production)
- Artifact logging with cryptographic hashing
- Zero-hallucination principles enforcement

The MLflowExperimentManager class is the primary interface for all
MLflow operations within GreenLang agents.

Example:
    >>> from greenlang.ml.mlflow_integration import MLflowExperimentManager
    >>> manager = MLflowExperimentManager(
    ...     experiment_name="greenlang-fuel-analyzer"
    ... )
    >>> with manager.start_run("fuel_model_v2") as run:
    ...     manager.log_agent_run(
    ...         agent_name="FuelAnalyzer",
    ...         input_data=fuel_data,
    ...         output_data=predictions,
    ...         metrics={"accuracy": 0.95}
    ...     )
    ...     manager.register_model(model, "fuel_emission_model")
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
import hashlib
import logging
import json
import pickle
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
import tempfile

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model deployment stages following MLflow convention."""
    NONE = "None"
    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelFramework(str, Enum):
    """Supported ML frameworks."""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    CUSTOM = "custom"


class ProvenanceType(str, Enum):
    """Types of provenance tracking."""
    INPUT_DATA = "input_data"
    OUTPUT_DATA = "output_data"
    MODEL = "model"
    PARAMETERS = "parameters"
    METRICS = "metrics"
    ARTIFACT = "artifact"
    ENVIRONMENT = "environment"


@dataclass
class ProvenanceRecord:
    """Record for SHA-256 provenance tracking."""
    record_type: ProvenanceType
    timestamp: datetime
    sha256_hash: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "type": self.record_type.value,
            "timestamp": self.timestamp.isoformat(),
            "sha256_hash": self.sha256_hash,
            "description": self.description,
            "metadata": self.metadata
        }


class MLflowConfig(BaseModel):
    """Configuration for MLflow integration."""

    tracking_uri: str = Field(
        default="http://mlflow-tracking.greenlang-mlops.svc.cluster.local:80",
        description="MLflow tracking server URI"
    )
    experiment_name: str = Field(
        default="greenlang-process-heat",
        description="Default experiment name"
    )
    artifact_location: str = Field(
        default="s3://greenlang-mlflow-artifacts/",
        description="S3 artifact storage location"
    )
    s3_endpoint_url: Optional[str] = Field(
        default=None,
        description="S3/MinIO endpoint URL (for local development)"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    auto_log_system_info: bool = Field(
        default=True,
        description="Automatically log system information"
    )
    auto_log_git_info: bool = Field(
        default=True,
        description="Automatically log git commit info"
    )
    registry_uri: Optional[str] = Field(
        default=None,
        description="Model registry URI (defaults to tracking_uri)"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class AgentRunMetadata(BaseModel):
    """Metadata for an agent run."""

    agent_name: str = Field(
        ...,
        description="Name of the agent"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    run_type: str = Field(
        default="inference",
        description="Type of run: training, inference, evaluation"
    )
    input_hash: str = Field(
        ...,
        description="SHA-256 hash of input data"
    )
    output_hash: str = Field(
        ...,
        description="SHA-256 hash of output data"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    end_time: Optional[datetime] = Field(default=None)
    status: str = Field(default="running")
    provenance_records: List[Dict[str, Any]] = Field(default_factory=list)


class RegisteredModelInfo(BaseModel):
    """Information about a registered model."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: ModelStage = Field(default=ModelStage.NONE)
    framework: ModelFramework = Field(default=ModelFramework.SKLEARN)
    run_id: str = Field(..., description="MLflow run ID")
    artifact_uri: str = Field(..., description="Model artifact URI")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MLflowExperimentManager:
    """
    MLflow Experiment Manager for GreenLang Process Heat Agents.

    This class provides a comprehensive interface for MLflow operations,
    implementing zero-hallucination principles with SHA-256 provenance
    tracking for all data, models, and artifacts.

    Key Features:
    - Automatic experiment tracking
    - SHA-256 provenance for all artifacts
    - Model versioning with staging
    - Process Heat agent integration
    - Production-ready configuration

    Attributes:
        config: MLflow configuration
        _mlflow: MLflow module reference
        _client: MLflow tracking client
        _current_run: Currently active run
        _provenance_chain: Chain of provenance records

    Example:
        >>> manager = MLflowExperimentManager(
        ...     experiment_name="greenlang-fuel-analyzer"
        ... )
        >>>
        >>> # Log an agent run
        >>> with manager.start_run("fuel_analysis_v1") as run:
        ...     # Log parameters
        ...     manager.log_params({
        ...         "fuel_type": "natural_gas",
        ...         "analysis_method": "GHG_Protocol"
        ...     })
        ...
        ...     # Run agent and log metrics
        ...     results = fuel_agent.analyze(data)
        ...     manager.log_metrics({
        ...         "emission_factor": results.emission_factor,
        ...         "uncertainty": results.uncertainty
        ...     })
        ...
        ...     # Log agent run with provenance
        ...     manager.log_agent_run(
        ...         agent_name="FuelAnalyzer",
        ...         input_data=data,
        ...         output_data=results,
        ...         metrics={"accuracy": 0.99}
        ...     )
        ...
        ...     # Register model
        ...     model_info = manager.register_model(
        ...         model=fuel_model,
        ...         name="fuel_emission_model",
        ...         metrics={"rmse": 0.05}
        ...     )
        >>>
        >>> # Promote model to production
        >>> manager.promote_model(
        ...     "fuel_emission_model",
        ...     model_info.version,
        ...     ModelStage.PRODUCTION
        ... )
        >>>
        >>> # Load production model
        >>> prod_model = manager.get_production_model("fuel_emission_model")
    """

    def __init__(
        self,
        config: Optional[MLflowConfig] = None,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow Experiment Manager.

        Args:
            config: MLflow configuration
            experiment_name: Override experiment name from config
            tracking_uri: Override tracking URI from config
        """
        self.config = config or MLflowConfig()

        if experiment_name:
            self.config.experiment_name = experiment_name
        if tracking_uri:
            self.config.tracking_uri = tracking_uri

        self._mlflow = None
        self._client = None
        self._current_run = None
        self._current_run_id: Optional[str] = None
        self._experiment_id: Optional[str] = None
        self._provenance_chain: List[ProvenanceRecord] = []
        self._initialized = False

        # Initialize MLflow
        self._initialize_mlflow()

        logger.info(
            f"MLflowExperimentManager initialized: "
            f"experiment={self.config.experiment_name}, "
            f"tracking_uri={self.config.tracking_uri}"
        )

    def _initialize_mlflow(self) -> bool:
        """Initialize MLflow connection and experiment."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            self._mlflow = mlflow

            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)

            # Set S3 endpoint if configured (for MinIO)
            if self.config.s3_endpoint_url:
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config.s3_endpoint_url

            # Initialize client
            self._client = MlflowClient()

            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(
                self.config.experiment_name
            )
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location
                )
                logger.info(
                    f"Created experiment: {self.config.experiment_name}"
                )
            else:
                self._experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.config.experiment_name)
            self._initialized = True

            logger.info("MLflow initialized successfully")
            return True

        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            return False

    def _compute_sha256(self, data: Any) -> str:
        """
        Compute SHA-256 hash for any data.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hash string
        """
        if data is None:
            return hashlib.sha256(b"null").hexdigest()

        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()

        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()

        if isinstance(data, (dict, list)):
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()

        # For other objects, pickle and hash
        try:
            pickled = pickle.dumps(data)
            return hashlib.sha256(pickled).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.sha256(str(data).encode()).hexdigest()

    def _add_provenance_record(
        self,
        record_type: ProvenanceType,
        data: Any,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProvenanceRecord:
        """
        Add a provenance record to the chain.

        Args:
            record_type: Type of provenance record
            data: Data to compute hash for
            description: Human-readable description
            metadata: Additional metadata

        Returns:
            Created provenance record
        """
        record = ProvenanceRecord(
            record_type=record_type,
            timestamp=datetime.now(timezone.utc),
            sha256_hash=self._compute_sha256(data),
            description=description,
            metadata=metadata or {}
        )
        self._provenance_chain.append(record)
        return record

    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit information."""
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

            return {
                "git_commit": commit,
                "git_commit_short": commit[:8],
                "git_branch": branch
            }
        except Exception:
            return {}

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for logging."""
        import platform

        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Additional tags to set
            nested: Whether this is a nested run

        Yields:
            MLflow Run object

        Example:
            >>> with manager.start_run("training_v1") as run:
            ...     manager.log_params({"epochs": 100})
            ...     train_model()
            ...     manager.log_metrics({"loss": 0.1})
        """
        if not self._initialized:
            logger.warning("MLflow not initialized, run will be local only")
            yield None
            return

        # Clear provenance chain for new run
        self._provenance_chain = []

        # Prepare tags
        all_tags = tags or {}
        all_tags["greenlang.provenance_enabled"] = str(
            self.config.enable_provenance
        )

        if self.config.auto_log_git_info:
            all_tags.update(self._get_git_info())

        try:
            run = self._mlflow.start_run(
                run_name=run_name,
                experiment_id=self._experiment_id,
                tags=all_tags,
                nested=nested
            )
            self._current_run = run
            self._current_run_id = run.info.run_id

            # Log system info
            if self.config.auto_log_system_info:
                system_info = self._get_system_info()
                for key, value in system_info.items():
                    self._mlflow.set_tag(f"system.{key}", value)

            logger.info(f"Started run: {self._current_run_id} ({run_name})")

            yield run

        except Exception as e:
            logger.error(f"Run failed: {e}")
            raise
        finally:
            # Log provenance chain
            if self._provenance_chain:
                provenance_path = self._save_provenance_chain()
                if provenance_path:
                    self._mlflow.log_artifact(provenance_path)

            self._mlflow.end_run()
            self._current_run = None
            self._current_run_id = None
            logger.info("Run ended")

    def _save_provenance_chain(self) -> Optional[str]:
        """Save provenance chain to a JSON file."""
        if not self._provenance_chain:
            return None

        chain_data = {
            "run_id": self._current_run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "records": [r.to_dict() for r in self._provenance_chain],
            "chain_hash": self._compute_sha256(
                [r.sha256_hash for r in self._provenance_chain]
            )
        }

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_provenance.json',
            delete=False
        ) as f:
            json.dump(chain_data, f, indent=2)
            return f.name

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run.

        Args:
            params: Dictionary of parameters

        Example:
            >>> manager.log_params({
            ...     "learning_rate": 0.01,
            ...     "batch_size": 32,
            ...     "fuel_type": "natural_gas"
            ... })
        """
        if not self._initialized or not self._current_run:
            logger.warning("No active run, parameters not logged to MLflow")
            return

        # Add provenance record
        if self.config.enable_provenance:
            self._add_provenance_record(
                ProvenanceType.PARAMETERS,
                params,
                f"Logged {len(params)} parameters"
            )

        for key, value in params.items():
            self._mlflow.log_param(key, value)

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
            step: Optional step number

        Example:
            >>> manager.log_metrics({
            ...     "rmse": 0.05,
            ...     "r2_score": 0.95,
            ...     "emission_factor_accuracy": 0.99
            ... })
        """
        if not self._initialized or not self._current_run:
            logger.warning("No active run, metrics not logged to MLflow")
            return

        # Add provenance record
        if self.config.enable_provenance:
            self._add_provenance_record(
                ProvenanceType.METRICS,
                metrics,
                f"Logged {len(metrics)} metrics at step {step}"
            )

        for key, value in metrics.items():
            self._mlflow.log_metric(key, value, step=step)

        logger.debug(f"Logged metrics: {list(metrics.keys())}")

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> str:
        """
        Log an artifact with SHA-256 provenance.

        Args:
            local_path: Path to local file
            artifact_path: Optional path in artifact storage

        Returns:
            SHA-256 hash of the artifact

        Example:
            >>> hash = manager.log_artifact("./results.csv")
        """
        if not self._initialized or not self._current_run:
            logger.warning("No active run, artifact not logged to MLflow")
            return ""

        # Compute hash
        with open(local_path, 'rb') as f:
            artifact_hash = self._compute_sha256(f.read())

        # Add provenance record
        if self.config.enable_provenance:
            self._add_provenance_record(
                ProvenanceType.ARTIFACT,
                artifact_hash,
                f"Logged artifact: {local_path}",
                metadata={"path": local_path, "artifact_path": artifact_path}
            )

        self._mlflow.log_artifact(local_path, artifact_path)

        # Log hash as a tag
        artifact_name = Path(local_path).name
        self._mlflow.set_tag(f"artifact.{artifact_name}.sha256", artifact_hash)

        logger.debug(f"Logged artifact: {local_path} (hash: {artifact_hash[:16]}...)")
        return artifact_hash

    def log_agent_run(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        agent_version: str = "1.0.0",
        run_type: str = "inference"
    ) -> AgentRunMetadata:
        """
        Log a complete agent run with full provenance tracking.

        This method provides comprehensive logging for GreenLang agent
        executions, including input/output hashing for zero-hallucination
        verification.

        Args:
            agent_name: Name of the agent (e.g., "FuelAnalyzer")
            input_data: Input data to the agent
            output_data: Output data from the agent
            metrics: Performance metrics
            parameters: Agent parameters
            agent_version: Agent version string
            run_type: Type of run (training, inference, evaluation)

        Returns:
            AgentRunMetadata with provenance information

        Example:
            >>> metadata = manager.log_agent_run(
            ...     agent_name="FuelAnalyzer",
            ...     input_data={"fuel_type": "natural_gas", "volume": 1000},
            ...     output_data={"emission_factor": 2.75, "co2_kg": 2750},
            ...     metrics={"accuracy": 0.99, "confidence": 0.95}
            ... )
            >>> print(f"Input hash: {metadata.input_hash}")
            >>> print(f"Output hash: {metadata.output_hash}")
        """
        start_time = datetime.now(timezone.utc)

        # Compute hashes
        input_hash = self._compute_sha256(input_data)
        output_hash = self._compute_sha256(output_data)

        # Add provenance records
        if self.config.enable_provenance:
            self._add_provenance_record(
                ProvenanceType.INPUT_DATA,
                input_data,
                f"Agent {agent_name} input data",
                metadata={"agent": agent_name, "run_type": run_type}
            )
            self._add_provenance_record(
                ProvenanceType.OUTPUT_DATA,
                output_data,
                f"Agent {agent_name} output data",
                metadata={"agent": agent_name, "run_type": run_type}
            )

        # Log to MLflow
        if self._initialized and self._current_run:
            # Set agent tags
            self._mlflow.set_tag("agent.name", agent_name)
            self._mlflow.set_tag("agent.version", agent_version)
            self._mlflow.set_tag("agent.run_type", run_type)
            self._mlflow.set_tag("agent.input_hash", input_hash)
            self._mlflow.set_tag("agent.output_hash", output_hash)

            # Log parameters
            if parameters:
                self.log_params(parameters)

            # Log metrics
            if metrics:
                self.log_metrics(metrics)

            # Log input/output as artifacts
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='_input.json',
                delete=False
            ) as f:
                json.dump(
                    {"data": input_data, "hash": input_hash},
                    f,
                    default=str,
                    indent=2
                )
                self._mlflow.log_artifact(f.name, "agent_data")

            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='_output.json',
                delete=False
            ) as f:
                json.dump(
                    {"data": output_data, "hash": output_hash},
                    f,
                    default=str,
                    indent=2
                )
                self._mlflow.log_artifact(f.name, "agent_data")

        end_time = datetime.now(timezone.utc)

        # Create metadata
        metadata = AgentRunMetadata(
            agent_name=agent_name,
            agent_version=agent_version,
            run_type=run_type,
            input_hash=input_hash,
            output_hash=output_hash,
            start_time=start_time,
            end_time=end_time,
            status="completed",
            provenance_records=[r.to_dict() for r in self._provenance_chain]
        )

        logger.info(
            f"Logged agent run: {agent_name} v{agent_version}, "
            f"input_hash={input_hash[:16]}..., output_hash={output_hash[:16]}..."
        )

        return metadata

    def register_model(
        self,
        model: Any,
        name: str,
        framework: ModelFramework = ModelFramework.SKLEARN,
        metrics: Optional[Dict[str, float]] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> RegisteredModelInfo:
        """
        Register a model in the MLflow Model Registry.

        Args:
            model: ML model to register
            name: Model name in registry
            framework: ML framework used
            metrics: Model performance metrics
            description: Model description
            tags: Additional tags

        Returns:
            RegisteredModelInfo with version and provenance

        Example:
            >>> model_info = manager.register_model(
            ...     model=trained_model,
            ...     name="fuel_emission_model",
            ...     framework=ModelFramework.SKLEARN,
            ...     metrics={"rmse": 0.05, "r2": 0.95}
            ... )
            >>> print(f"Registered version: {model_info.version}")
        """
        if not self._initialized or not self._current_run:
            raise RuntimeError("No active MLflow run for model registration")

        metrics = metrics or {}
        tags = tags or {}

        # Compute model provenance hash
        provenance_hash = self._compute_sha256({
            "model_type": str(type(model).__name__),
            "model_params": getattr(model, 'get_params', lambda: {})(),
            "metrics": metrics,
            "framework": framework.value
        })

        # Add provenance record
        if self.config.enable_provenance:
            self._add_provenance_record(
                ProvenanceType.MODEL,
                model,
                f"Registered model: {name}",
                metadata={
                    "framework": framework.value,
                    "metrics": metrics
                }
            )

        # Log model based on framework
        model_info = None
        artifact_uri = ""

        try:
            if framework == ModelFramework.SKLEARN:
                import mlflow.sklearn
                model_info = mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=name
                )
            elif framework == ModelFramework.PYTORCH:
                import mlflow.pytorch
                model_info = mlflow.pytorch.log_model(
                    model,
                    "model",
                    registered_model_name=name
                )
            elif framework == ModelFramework.TENSORFLOW:
                import mlflow.tensorflow
                model_info = mlflow.tensorflow.log_model(
                    model,
                    "model",
                    registered_model_name=name
                )
            elif framework == ModelFramework.XGBOOST:
                import mlflow.xgboost
                model_info = mlflow.xgboost.log_model(
                    model,
                    "model",
                    registered_model_name=name
                )
            elif framework == ModelFramework.LIGHTGBM:
                import mlflow.lightgbm
                model_info = mlflow.lightgbm.log_model(
                    model,
                    "model",
                    registered_model_name=name
                )
            else:
                # Generic Python model logging
                import mlflow.pyfunc
                model_info = mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    registered_model_name=name
                )

            artifact_uri = model_info.model_uri if model_info else ""

        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise

        # Get registered version
        version = "1"
        try:
            versions = self._client.search_model_versions(f"name='{name}'")
            if versions:
                version = str(max(int(v.version) for v in versions))
        except Exception as e:
            logger.warning(f"Could not get model version: {e}")

        # Set model tags
        try:
            self._client.set_model_version_tag(
                name,
                version,
                "provenance_hash",
                provenance_hash
            )
            self._client.set_model_version_tag(
                name,
                version,
                "framework",
                framework.value
            )
            for key, value in tags.items():
                self._client.set_model_version_tag(name, version, key, value)

            # Update model description
            if description:
                self._client.update_model_version(
                    name,
                    version,
                    description=description
                )
        except Exception as e:
            logger.warning(f"Could not set model tags: {e}")

        # Log metrics
        if metrics:
            self.log_metrics(metrics)

        result = RegisteredModelInfo(
            name=name,
            version=version,
            stage=ModelStage.NONE,
            framework=framework,
            run_id=self._current_run_id,
            artifact_uri=artifact_uri,
            provenance_hash=provenance_hash,
            metrics=metrics,
            tags=tags
        )

        logger.info(
            f"Registered model: {name} v{version}, "
            f"provenance: {provenance_hash[:16]}..."
        )

        return result

    def promote_model(
        self,
        name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True
    ) -> bool:
        """
        Promote a model to a new stage.

        Args:
            name: Model name
            version: Model version
            stage: Target stage
            archive_existing: Archive current production model

        Returns:
            Success status

        Example:
            >>> manager.promote_model(
            ...     "fuel_emission_model",
            ...     "5",
            ...     ModelStage.PRODUCTION
            ... )
        """
        if not self._initialized:
            logger.error("MLflow not initialized")
            return False

        try:
            # Archive existing production model if requested
            if archive_existing and stage == ModelStage.PRODUCTION:
                versions = self._client.search_model_versions(f"name='{name}'")
                for v in versions:
                    if v.current_stage == "Production":
                        self._client.transition_model_version_stage(
                            name,
                            v.version,
                            "Archived"
                        )
                        logger.info(
                            f"Archived {name} v{v.version}"
                        )

            # Transition to new stage
            self._client.transition_model_version_stage(
                name,
                version,
                stage.value
            )

            logger.info(
                f"Promoted {name} v{version} to {stage.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False

    def get_production_model(
        self,
        name: str
    ) -> Optional[Any]:
        """
        Load the production version of a model.

        Args:
            name: Model name

        Returns:
            Loaded model or None if not found

        Example:
            >>> model = manager.get_production_model("fuel_emission_model")
            >>> predictions = model.predict(data)
        """
        if not self._initialized:
            logger.error("MLflow not initialized")
            return None

        try:
            # Get production version
            versions = self._client.search_model_versions(
                f"name='{name}'"
            )
            production_version = None
            for v in versions:
                if v.current_stage == "Production":
                    production_version = v
                    break

            if not production_version:
                logger.warning(f"No production model found for {name}")
                return None

            # Load model
            model_uri = f"models:/{name}/Production"
            model = self._mlflow.pyfunc.load_model(model_uri)

            logger.info(
                f"Loaded production model: {name} v{production_version.version}"
            )
            return model

        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None

    def get_model_by_version(
        self,
        name: str,
        version: str
    ) -> Optional[Any]:
        """
        Load a specific version of a model.

        Args:
            name: Model name
            version: Model version

        Returns:
            Loaded model or None if not found
        """
        if not self._initialized:
            logger.error("MLflow not initialized")
            return None

        try:
            model_uri = f"models:/{name}/{version}"
            model = self._mlflow.pyfunc.load_model(model_uri)

            logger.info(f"Loaded model: {name} v{version}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        if not self._initialized:
            return []

        try:
            experiments = self._client.search_experiments()
            return [
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "artifact_location": e.artifact_location,
                    "lifecycle_stage": e.lifecycle_stage
                }
                for e in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        if not self._initialized:
            return []

        try:
            models = self._client.search_registered_models()
            return [
                {
                    "name": m.name,
                    "description": m.description,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id
                        }
                        for v in m.latest_versions
                    ]
                }
                for m in models
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for runs in the current experiment.

        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of results

        Returns:
            List of run dictionaries
        """
        if not self._initialized:
            return []

        try:
            runs = self._client.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            return [
                {
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "end_time": r.info.end_time,
                    "params": r.data.params,
                    "metrics": r.data.metrics,
                    "tags": r.data.tags
                }
                for r in runs
            ]
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []

    def compare_models(
        self,
        name: str,
        versions: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple versions of a model.

        Args:
            name: Model name
            versions: List of versions to compare

        Returns:
            Comparison dictionary with metrics and parameters
        """
        if not self._initialized:
            return {}

        comparison = {}

        try:
            for version in versions:
                mv = self._client.get_model_version(name, version)
                run = self._client.get_run(mv.run_id)

                comparison[version] = {
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": {
                        k: v for k, v in run.data.tags.items()
                        if not k.startswith("mlflow.")
                    }
                }

        except Exception as e:
            logger.error(f"Failed to compare models: {e}")

        return comparison


# Convenience function for creating managers
def create_experiment_manager(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    enable_provenance: bool = True
) -> MLflowExperimentManager:
    """
    Create an MLflow Experiment Manager with sensible defaults.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional MLflow tracking server URI
        enable_provenance: Enable SHA-256 provenance tracking

    Returns:
        Configured MLflowExperimentManager

    Example:
        >>> manager = create_experiment_manager("greenlang-fuel-analyzer")
        >>> with manager.start_run("training_v1"):
        ...     manager.log_params({"epochs": 100})
    """
    config = MLflowConfig(
        experiment_name=experiment_name,
        enable_provenance=enable_provenance
    )

    if tracking_uri:
        config.tracking_uri = tracking_uri

    return MLflowExperimentManager(config=config)


# Unit test stubs
class TestMLflowExperimentManager:
    """Unit tests for MLflowExperimentManager."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        # This test requires MLflow server or will use fallback
        config = MLflowConfig()
        assert config.experiment_name == "greenlang-process-heat"
        assert config.enable_provenance is True

    def test_compute_sha256_string(self):
        """Test SHA-256 computation for strings."""
        manager = MLflowExperimentManager.__new__(MLflowExperimentManager)
        manager.config = MLflowConfig()

        hash1 = manager._compute_sha256("test")
        hash2 = manager._compute_sha256("test")
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_compute_sha256_dict(self):
        """Test SHA-256 computation for dictionaries."""
        manager = MLflowExperimentManager.__new__(MLflowExperimentManager)
        manager.config = MLflowConfig()

        data = {"key": "value", "number": 42}
        hash1 = manager._compute_sha256(data)
        hash2 = manager._compute_sha256(data)
        assert hash1 == hash2

    def test_provenance_record_creation(self):
        """Test provenance record creation."""
        record = ProvenanceRecord(
            record_type=ProvenanceType.INPUT_DATA,
            timestamp=datetime.now(timezone.utc),
            sha256_hash="abc123",
            description="Test record"
        )

        record_dict = record.to_dict()
        assert record_dict["type"] == "input_data"
        assert record_dict["sha256_hash"] == "abc123"

    def test_agent_run_metadata(self):
        """Test AgentRunMetadata creation."""
        metadata = AgentRunMetadata(
            agent_name="TestAgent",
            input_hash="input123",
            output_hash="output456"
        )

        assert metadata.agent_name == "TestAgent"
        assert metadata.status == "running"

    def test_model_stage_enum(self):
        """Test ModelStage enum values."""
        assert ModelStage.PRODUCTION.value == "Production"
        assert ModelStage.STAGING.value == "Staging"
        assert ModelStage.DEVELOPMENT.value == "Development"

    def test_registered_model_info(self):
        """Test RegisteredModelInfo creation."""
        info = RegisteredModelInfo(
            name="test_model",
            version="1",
            run_id="run123",
            artifact_uri="s3://bucket/model",
            provenance_hash="hash123"
        )

        assert info.name == "test_model"
        assert info.stage == ModelStage.NONE
