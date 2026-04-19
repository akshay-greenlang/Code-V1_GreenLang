"""
Model Cards for GL-003 UNIFIEDSTEAM

Provides operator-readable model documentation following Google's
Model Card framework with extensions for industrial ML systems.

Reference: Mitchell et al., "Model Cards for Model Reporting" (2019)

Author: GL-003 MLOps Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models in GL-003."""
    TRAP_FAILURE_CLASSIFIER = "trap_failure_classifier"
    STEAM_QUALITY_INFERENTIAL = "steam_quality_inferential"
    ANOMALY_DETECTOR = "anomaly_detector"
    DEMAND_FORECASTER = "demand_forecaster"
    OPTIMIZATION_RECOMMENDER = "optimization_recommender"
    ROOT_CAUSE_CLASSIFIER = "root_cause_classifier"


class ModelPurpose(Enum):
    """Model purpose categories."""
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"
    OPTIMIZATION = "optimization"
    INFERENCE = "inference"


class DataSplit(Enum):
    """Data split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a model.

    Stores metrics with confidence intervals and data split information.
    """
    data_split: DataSplit
    evaluation_date: datetime
    sample_size: int

    # Classification metrics
    accuracy: Optional[Decimal] = None
    precision: Optional[Decimal] = None
    recall: Optional[Decimal] = None
    f1_score: Optional[Decimal] = None
    auc_roc: Optional[Decimal] = None
    auc_pr: Optional[Decimal] = None

    # Regression metrics
    rmse: Optional[Decimal] = None
    mae: Optional[Decimal] = None
    r_squared: Optional[Decimal] = None
    mape: Optional[Decimal] = None

    # Confidence intervals (95%)
    accuracy_ci: Optional[tuple] = None
    precision_ci: Optional[tuple] = None
    recall_ci: Optional[tuple] = None

    # Calibration
    brier_score: Optional[Decimal] = None
    calibration_error: Optional[Decimal] = None

    # Custom metrics
    custom_metrics: Dict[str, Decimal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "data_split": self.data_split.value,
            "evaluation_date": self.evaluation_date.isoformat(),
            "sample_size": self.sample_size,
        }

        # Add non-None metrics
        for metric in ["accuracy", "precision", "recall", "f1_score",
                      "auc_roc", "auc_pr", "rmse", "mae", "r_squared",
                      "mape", "brier_score", "calibration_error"]:
            value = getattr(self, metric)
            if value is not None:
                result[metric] = str(value)

        if self.custom_metrics:
            result["custom_metrics"] = {k: str(v) for k, v in self.custom_metrics.items()}

        return result


@dataclass
class FairnessMetrics:
    """
    Fairness metrics for model evaluation.

    Tracks performance across different segments/groups.
    """
    segment_name: str
    segment_values: List[str]
    evaluation_date: datetime

    # Per-segment metrics
    segment_accuracy: Dict[str, Decimal] = field(default_factory=dict)
    segment_precision: Dict[str, Decimal] = field(default_factory=dict)
    segment_recall: Dict[str, Decimal] = field(default_factory=dict)
    segment_sample_size: Dict[str, int] = field(default_factory=dict)

    # Fairness measures
    demographic_parity: Optional[Decimal] = None
    equalized_odds: Optional[Decimal] = None
    calibration_by_group: Dict[str, Decimal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_name": self.segment_name,
            "segment_values": self.segment_values,
            "evaluation_date": self.evaluation_date.isoformat(),
            "segment_accuracy": {k: str(v) for k, v in self.segment_accuracy.items()},
            "segment_precision": {k: str(v) for k, v in self.segment_precision.items()},
            "segment_recall": {k: str(v) for k, v in self.segment_recall.items()},
            "segment_sample_size": self.segment_sample_size,
            "demographic_parity": str(self.demographic_parity) if self.demographic_parity else None,
            "equalized_odds": str(self.equalized_odds) if self.equalized_odds else None,
        }


@dataclass
class ModelLimitations:
    """
    Known limitations and failure modes of a model.

    Documents when the model should not be used or may fail.
    """
    out_of_scope_uses: List[str]
    known_failure_modes: List[str]
    input_constraints: Dict[str, Any]
    operating_conditions: Dict[str, Any]
    degradation_conditions: List[str]
    mitigation_strategies: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "out_of_scope_uses": self.out_of_scope_uses,
            "known_failure_modes": self.known_failure_modes,
            "input_constraints": self.input_constraints,
            "operating_conditions": self.operating_conditions,
            "degradation_conditions": self.degradation_conditions,
            "mitigation_strategies": self.mitigation_strategies,
        }


@dataclass
class TrainingData:
    """
    Training data documentation.

    Records data sources, preprocessing, and lineage.
    """
    data_sources: List[str]
    collection_period_start: datetime
    collection_period_end: datetime
    total_samples: int
    positive_samples: Optional[int] = None
    negative_samples: Optional[int] = None

    # Data quality
    missing_data_handling: str = ""
    outlier_handling: str = ""
    preprocessing_steps: List[str] = field(default_factory=list)

    # Splits
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0

    # Lineage
    data_version: str = ""
    feature_engineering_version: str = ""
    tag_mapping_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_sources": self.data_sources,
            "collection_period": {
                "start": self.collection_period_start.isoformat(),
                "end": self.collection_period_end.isoformat(),
            },
            "samples": {
                "total": self.total_samples,
                "positive": self.positive_samples,
                "negative": self.negative_samples,
                "train": self.train_samples,
                "validation": self.validation_samples,
                "test": self.test_samples,
            },
            "data_quality": {
                "missing_data_handling": self.missing_data_handling,
                "outlier_handling": self.outlier_handling,
                "preprocessing_steps": self.preprocessing_steps,
            },
            "lineage": {
                "data_version": self.data_version,
                "feature_engineering_version": self.feature_engineering_version,
                "tag_mapping_version": self.tag_mapping_version,
            },
        }


@dataclass
class ModelCard:
    """
    Comprehensive model card for ML model documentation.

    Provides operator-readable documentation of model purpose, performance,
    limitations, and governance information.
    """
    # Basic information
    model_id: str
    model_name: str
    model_type: ModelType
    model_purpose: ModelPurpose
    version: str
    created_date: datetime
    last_updated: datetime

    # Description
    short_description: str
    detailed_description: str
    intended_use: str
    primary_users: List[str]

    # Technical details
    model_architecture: str
    framework: str
    framework_version: str
    input_features: List[str]
    output_description: str

    # Performance
    performance_metrics: List[PerformanceMetrics]
    fairness_metrics: Optional[FairnessMetrics] = None

    # Training data
    training_data: Optional[TrainingData] = None

    # Limitations
    limitations: Optional[ModelLimitations] = None

    # Governance
    owner: str = ""
    maintainers: List[str] = field(default_factory=list)
    review_status: str = "pending"
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None

    # Deployment
    deployment_status: str = "not_deployed"
    deployment_date: Optional[datetime] = None
    deployment_environment: str = ""

    # Ethics
    ethical_considerations: List[str] = field(default_factory=list)
    risks_and_harms: List[str] = field(default_factory=list)

    # Versioning
    parent_model_id: Optional[str] = None
    change_log: List[Dict[str, Any]] = field(default_factory=list)

    def get_summary(self) -> str:
        """
        Get operator-readable summary of model.

        Returns:
            String summary suitable for operators
        """
        summary = f"""
MODEL CARD: {self.model_name}
{'=' * 60}

Model ID: {self.model_id}
Version: {self.version}
Type: {self.model_type.value}
Purpose: {self.model_purpose.value}

DESCRIPTION:
{self.short_description}

INTENDED USE:
{self.intended_use}

PRIMARY USERS:
{', '.join(self.primary_users)}

INPUT FEATURES:
{', '.join(self.input_features[:10])}{'...' if len(self.input_features) > 10 else ''}

OUTPUT:
{self.output_description}

ARCHITECTURE:
{self.model_architecture}

PERFORMANCE (Latest):
"""
        if self.performance_metrics:
            latest = self.performance_metrics[-1]
            if latest.accuracy is not None:
                summary += f"  Accuracy: {latest.accuracy}%\n"
            if latest.precision is not None:
                summary += f"  Precision: {latest.precision}%\n"
            if latest.recall is not None:
                summary += f"  Recall: {latest.recall}%\n"
            if latest.f1_score is not None:
                summary += f"  F1 Score: {latest.f1_score}%\n"

        if self.limitations:
            summary += f"""
KNOWN LIMITATIONS:
- Out of scope: {', '.join(self.limitations.out_of_scope_uses[:3])}
- Failure modes: {', '.join(self.limitations.known_failure_modes[:3])}
"""

        summary += f"""
GOVERNANCE:
  Owner: {self.owner}
  Review Status: {self.review_status}
  Deployment Status: {self.deployment_status}

Last Updated: {self.last_updated.strftime('%Y-%m-%d')}
"""
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "model_purpose": self.model_purpose.value,
            "version": self.version,
            "created_date": self.created_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "description": {
                "short": self.short_description,
                "detailed": self.detailed_description,
                "intended_use": self.intended_use,
                "primary_users": self.primary_users,
            },
            "technical": {
                "architecture": self.model_architecture,
                "framework": self.framework,
                "framework_version": self.framework_version,
                "input_features": self.input_features,
                "output_description": self.output_description,
            },
            "performance": [m.to_dict() for m in self.performance_metrics],
            "fairness": self.fairness_metrics.to_dict() if self.fairness_metrics else None,
            "training_data": self.training_data.to_dict() if self.training_data else None,
            "limitations": self.limitations.to_dict() if self.limitations else None,
            "governance": {
                "owner": self.owner,
                "maintainers": self.maintainers,
                "review_status": self.review_status,
                "approved_by": self.approved_by,
                "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            },
            "deployment": {
                "status": self.deployment_status,
                "date": self.deployment_date.isoformat() if self.deployment_date else None,
                "environment": self.deployment_environment,
            },
            "ethics": {
                "considerations": self.ethical_considerations,
                "risks_and_harms": self.risks_and_harms,
            },
            "versioning": {
                "parent_model_id": self.parent_model_id,
                "change_log": self.change_log,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class ModelCardBuilder:
    """
    Builder for creating model cards with validation.

    Provides fluent interface for constructing model cards with
    required field validation.
    """

    def __init__(self, model_id: str, model_name: str):
        """
        Initialize builder.

        Args:
            model_id: Unique model identifier
            model_name: Human-readable model name
        """
        self._model_id = model_id
        self._model_name = model_name
        self._model_type: Optional[ModelType] = None
        self._model_purpose: Optional[ModelPurpose] = None
        self._version = "1.0.0"
        self._created_date = datetime.now(timezone.utc)
        self._short_description = ""
        self._detailed_description = ""
        self._intended_use = ""
        self._primary_users: List[str] = []
        self._model_architecture = ""
        self._framework = ""
        self._framework_version = ""
        self._input_features: List[str] = []
        self._output_description = ""
        self._performance_metrics: List[PerformanceMetrics] = []
        self._fairness_metrics: Optional[FairnessMetrics] = None
        self._training_data: Optional[TrainingData] = None
        self._limitations: Optional[ModelLimitations] = None
        self._owner = ""
        self._maintainers: List[str] = []
        self._ethical_considerations: List[str] = []
        self._risks_and_harms: List[str] = []

    def with_type(self, model_type: ModelType) -> "ModelCardBuilder":
        """Set model type."""
        self._model_type = model_type
        return self

    def with_purpose(self, purpose: ModelPurpose) -> "ModelCardBuilder":
        """Set model purpose."""
        self._model_purpose = purpose
        return self

    def with_version(self, version: str) -> "ModelCardBuilder":
        """Set model version."""
        self._version = version
        return self

    def with_description(
        self,
        short: str,
        detailed: str,
        intended_use: str,
    ) -> "ModelCardBuilder":
        """Set model descriptions."""
        self._short_description = short
        self._detailed_description = detailed
        self._intended_use = intended_use
        return self

    def with_users(self, users: List[str]) -> "ModelCardBuilder":
        """Set primary users."""
        self._primary_users = users
        return self

    def with_architecture(
        self,
        architecture: str,
        framework: str,
        framework_version: str,
    ) -> "ModelCardBuilder":
        """Set architecture details."""
        self._model_architecture = architecture
        self._framework = framework
        self._framework_version = framework_version
        return self

    def with_inputs_outputs(
        self,
        input_features: List[str],
        output_description: str,
    ) -> "ModelCardBuilder":
        """Set input features and output description."""
        self._input_features = input_features
        self._output_description = output_description
        return self

    def with_performance(
        self,
        metrics: PerformanceMetrics,
    ) -> "ModelCardBuilder":
        """Add performance metrics."""
        self._performance_metrics.append(metrics)
        return self

    def with_fairness(
        self,
        fairness: FairnessMetrics,
    ) -> "ModelCardBuilder":
        """Set fairness metrics."""
        self._fairness_metrics = fairness
        return self

    def with_training_data(
        self,
        training_data: TrainingData,
    ) -> "ModelCardBuilder":
        """Set training data documentation."""
        self._training_data = training_data
        return self

    def with_limitations(
        self,
        limitations: ModelLimitations,
    ) -> "ModelCardBuilder":
        """Set model limitations."""
        self._limitations = limitations
        return self

    def with_governance(
        self,
        owner: str,
        maintainers: Optional[List[str]] = None,
    ) -> "ModelCardBuilder":
        """Set governance information."""
        self._owner = owner
        self._maintainers = maintainers or []
        return self

    def with_ethics(
        self,
        considerations: List[str],
        risks: List[str],
    ) -> "ModelCardBuilder":
        """Set ethical considerations."""
        self._ethical_considerations = considerations
        self._risks_and_harms = risks
        return self

    def build(self) -> ModelCard:
        """
        Build the model card.

        Returns:
            Validated ModelCard

        Raises:
            ValueError: If required fields are missing
        """
        # Validate required fields
        if not self._model_type:
            raise ValueError("Model type is required")
        if not self._model_purpose:
            raise ValueError("Model purpose is required")
        if not self._short_description:
            raise ValueError("Short description is required")
        if not self._intended_use:
            raise ValueError("Intended use is required")
        if not self._input_features:
            raise ValueError("Input features are required")
        if not self._output_description:
            raise ValueError("Output description is required")
        if not self._owner:
            raise ValueError("Owner is required")

        return ModelCard(
            model_id=self._model_id,
            model_name=self._model_name,
            model_type=self._model_type,
            model_purpose=self._model_purpose,
            version=self._version,
            created_date=self._created_date,
            last_updated=datetime.now(timezone.utc),
            short_description=self._short_description,
            detailed_description=self._detailed_description,
            intended_use=self._intended_use,
            primary_users=self._primary_users,
            model_architecture=self._model_architecture,
            framework=self._framework,
            framework_version=self._framework_version,
            input_features=self._input_features,
            output_description=self._output_description,
            performance_metrics=self._performance_metrics,
            fairness_metrics=self._fairness_metrics,
            training_data=self._training_data,
            limitations=self._limitations,
            owner=self._owner,
            maintainers=self._maintainers,
            ethical_considerations=self._ethical_considerations,
            risks_and_harms=self._risks_and_harms,
        )


# Pre-built model card templates for GL-003 models
def create_trap_failure_model_card() -> ModelCard:
    """Create model card template for trap failure classifier."""
    return (
        ModelCardBuilder("GL003-TRAP-001", "Steam Trap Failure Classifier")
        .with_type(ModelType.TRAP_FAILURE_CLASSIFIER)
        .with_purpose(ModelPurpose.CLASSIFICATION)
        .with_description(
            short="Predicts steam trap failure probability using acoustic features and process context.",
            detailed=(
                "This model classifies steam traps into healthy, degraded, or failed states "
                "using ultrasonic acoustic signatures combined with operating conditions "
                "(pressure differential, temperature, flow estimates). It enables predictive "
                "maintenance by identifying traps likely to fail before catastrophic loss occurs."
            ),
            intended_use=(
                "Prioritize steam trap inspection and replacement based on predicted failure "
                "probability. Integrate with CMMS for work order generation."
            ),
        )
        .with_users([
            "Maintenance engineers",
            "Reliability engineers",
            "Steam system operators",
            "Energy managers",
        ])
        .with_architecture(
            architecture="Gradient Boosted Trees (XGBoost)",
            framework="XGBoost",
            framework_version="1.7.6",
        )
        .with_inputs_outputs(
            input_features=[
                "acoustic_rms_db",
                "acoustic_peak_freq_hz",
                "acoustic_spectral_entropy",
                "inlet_pressure_kpa",
                "outlet_pressure_kpa",
                "pressure_differential_kpa",
                "inlet_temperature_c",
                "trap_type_encoded",
                "trap_age_days",
                "time_since_last_service_days",
                "operating_hours",
                "process_load_pct",
            ],
            output_description=(
                "Failure probability (0-1) with classification into: "
                "HEALTHY (<0.3), DEGRADED (0.3-0.7), FAILED (>0.7)"
            ),
        )
        .with_limitations(ModelLimitations(
            out_of_scope_uses=[
                "Real-time safety interlocks",
                "Traps without acoustic monitoring",
                "Non-steam thermal traps",
            ],
            known_failure_modes=[
                "Low confidence with noisy acoustic data",
                "May miss slow degradation if inspection interval too long",
                "Reduced accuracy on trap types not in training data",
            ],
            input_constraints={
                "acoustic_rms_db": {"min": 20, "max": 120},
                "inlet_pressure_kpa": {"min": 100, "max": 4000},
                "inlet_temperature_c": {"min": 100, "max": 350},
            },
            operating_conditions={
                "steady_state": "Model performs best during steady-state operation",
                "startup_shutdown": "Predictions less reliable during transients",
            },
            degradation_conditions=[
                "Performance degrades if sensor calibration drifts",
                "New trap types require model retraining",
            ],
            mitigation_strategies={
                "low_confidence": "Require manual inspection for predictions with confidence < 70%",
                "missing_acoustics": "Fall back to time-based inspection schedule",
                "new_trap_type": "Flag for manual review and data collection",
            },
        ))
        .with_governance(
            owner="Reliability Engineering",
            maintainers=["ML Team", "Steam Systems Team"],
        )
        .with_ethics(
            considerations=[
                "False negatives may lead to steam losses and safety risks",
                "False positives may lead to unnecessary maintenance costs",
            ],
            risks=[
                "Over-reliance on model predictions without field verification",
                "Maintenance resource allocation based solely on model output",
            ],
        )
        .build()
    )
