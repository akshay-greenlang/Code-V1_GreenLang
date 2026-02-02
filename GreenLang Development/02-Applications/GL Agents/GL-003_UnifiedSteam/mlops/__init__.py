"""
GL-003 UNIFIEDSTEAM - MLOps Model Governance Module

This module provides comprehensive ML model lifecycle management including
model cards, versioning, registry, performance monitoring, drift detection,
and feature engineering governance.

Key Features:
    - Model cards with operator-readable documentation
    - Model versioning and artifact management
    - Model registry with deployment tracking
    - Performance monitoring (precision, recall, calibration)
    - Drift detection for feature and prediction distributions
    - Feature definitions and lineage tracking
    - Safe deployment with canary releases
    - Human feedback loop integration

Reference Standards:
    - Google Model Cards for Model Reporting
    - MLOps best practices
    - Responsible AI guidelines

Author: GL-003 MLOps Team
Version: 1.0.0
"""

from .model_cards import (
    ModelCard,
    ModelCardBuilder,
    ModelPurpose,
    ModelLimitations,
    PerformanceMetrics,
    FairnessMetrics,
)

from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelArtifact,
    DeploymentStatus,
    ModelStage,
)

from .model_monitoring import (
    ModelMonitor,
    PerformanceTracker,
    DriftDetector,
    AlertThreshold,
    MonitoringReport,
)

from .feature_store import (
    FeatureDefinition,
    FeatureGroup,
    FeatureLineage,
    FeatureStore,
)

from .deployment_governance import (
    DeploymentGovernor,
    CanaryConfig,
    RollbackPolicy,
    GatingCriteria,
)

__all__ = [
    # Model Cards
    "ModelCard",
    "ModelCardBuilder",
    "ModelPurpose",
    "ModelLimitations",
    "PerformanceMetrics",
    "FairnessMetrics",
    # Model Registry
    "ModelRegistry",
    "ModelVersion",
    "ModelArtifact",
    "DeploymentStatus",
    "ModelStage",
    # Model Monitoring
    "ModelMonitor",
    "PerformanceTracker",
    "DriftDetector",
    "AlertThreshold",
    "MonitoringReport",
    # Feature Store
    "FeatureDefinition",
    "FeatureGroup",
    "FeatureLineage",
    "FeatureStore",
    # Deployment Governance
    "DeploymentGovernor",
    "CanaryConfig",
    "RollbackPolicy",
    "GatingCriteria",
]
