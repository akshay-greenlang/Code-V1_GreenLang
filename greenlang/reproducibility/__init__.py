# -*- coding: utf-8 -*-
"""
GL-FOUND-X-008: GreenLang Reproducibility Service SDK
======================================================

This package provides the reproducibility verification, deterministic
hashing, drift detection, replay execution, and provenance tracking
SDK for the GreenLang framework. It supports:

- Deterministic artifact hashing with float normalization
- Input/output hash verification with configurable tolerance
- Drift detection with soft/hard threshold classification
- Named drift baseline management
- Replay execution with environment, seed, and version verification
- Environment fingerprinting and comparison
- Random seed management (Python, NumPy, PyTorch, custom)
- Version pinning and manifest management
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_REPRODUCIBILITY_ env prefix

Key Components:
    - config: ReproducibilityConfig with GL_REPRODUCIBILITY_ env prefix
    - models: Pydantic v2 models for all data structures
    - artifact_hasher: Deterministic hashing engine
    - determinism_verifier: Input/output verification engine
    - drift_detector: Drift detection and baseline management
    - replay_engine: Replay execution engine
    - environment_capture: Environment fingerprinting
    - seed_manager: Random seed management
    - version_pinner: Version pinning and manifest management
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: ReproducibilityService facade

Example:
    >>> from greenlang.reproducibility import ReproducibilityService
    >>> service = ReproducibilityService()
    >>> run = service.verify("exec_001", {"emissions": 100.5})
    >>> print(run.is_reproducible)
    True

    >>> h = service.compute_hash({"value": 42.0})
    >>> print(len(h))
    64

Agent ID: GL-FOUND-X-008
Agent Name: Run Reproducibility Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-008"
__agent_name__ = "Run Reproducibility Agent"

# SDK availability flag
REPRODUCIBILITY_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.reproducibility.config import (
    ReproducibilityConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, Layer 1, SDK)
# ---------------------------------------------------------------------------
from greenlang.reproducibility.models import (
    # Enumerations
    VerificationStatus,
    DriftSeverity,
    NonDeterminismSource,
    # Layer 1 models
    EnvironmentFingerprint,
    SeedConfiguration,
    VersionPin,
    VersionManifest,
    VerificationCheck,
    DriftDetection,
    ReplayConfiguration,
    ReproducibilityInput,
    ReproducibilityOutput,
    ReproducibilityReport,
    # Constants
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_DRIFT_SOFT_THRESHOLD,
    DEFAULT_DRIFT_HARD_THRESHOLD,
    # SDK models
    ArtifactHash,
    VerificationRun,
    DriftBaseline,
    ReplaySession,
    VerificationStatistics,
    # Request / Response
    HashRequest,
    HashResponse,
    DriftRequest,
    DriftResponse,
    ReplayRequest,
    ReplayResponse,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.reproducibility.artifact_hasher import ArtifactHasher
from greenlang.reproducibility.determinism_verifier import DeterminismVerifier
from greenlang.reproducibility.drift_detector import DriftDetector
from greenlang.reproducibility.replay_engine import ReplayEngine
from greenlang.reproducibility.environment_capture import EnvironmentCapture
from greenlang.reproducibility.seed_manager import SeedManager
from greenlang.reproducibility.version_pinner import VersionPinner
from greenlang.reproducibility.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.reproducibility.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    reproducibility_verifications_total,
    reproducibility_verification_duration_seconds,
    reproducibility_hash_computations_total,
    reproducibility_hash_mismatches_total,
    reproducibility_drift_detections_total,
    reproducibility_drift_percentage,
    reproducibility_replays_total,
    reproducibility_replay_duration_seconds,
    reproducibility_non_determinism_sources_total,
    reproducibility_environment_mismatches_total,
    reproducibility_cache_hits_total,
    reproducibility_cache_misses_total,
    # Helper functions
    record_verification,
    record_hash_computation,
    record_hash_mismatch,
    record_drift,
    record_replay,
    record_non_determinism,
    record_environment_mismatch,
    record_cache_hit,
    record_cache_miss,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.reproducibility.setup import (
    ReproducibilityService,
    configure_reproducibility,
    get_reproducibility,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "REPRODUCIBILITY_SDK_AVAILABLE",
    # Configuration
    "ReproducibilityConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "VerificationStatus",
    "DriftSeverity",
    "NonDeterminismSource",
    # Layer 1 models
    "EnvironmentFingerprint",
    "SeedConfiguration",
    "VersionPin",
    "VersionManifest",
    "VerificationCheck",
    "DriftDetection",
    "ReplayConfiguration",
    "ReproducibilityInput",
    "ReproducibilityOutput",
    "ReproducibilityReport",
    # Constants
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "DEFAULT_DRIFT_SOFT_THRESHOLD",
    "DEFAULT_DRIFT_HARD_THRESHOLD",
    # SDK models
    "ArtifactHash",
    "VerificationRun",
    "DriftBaseline",
    "ReplaySession",
    "VerificationStatistics",
    # Request / Response
    "HashRequest",
    "HashResponse",
    "DriftRequest",
    "DriftResponse",
    "ReplayRequest",
    "ReplayResponse",
    # Core engines
    "ArtifactHasher",
    "DeterminismVerifier",
    "DriftDetector",
    "ReplayEngine",
    "EnvironmentCapture",
    "SeedManager",
    "VersionPinner",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "reproducibility_verifications_total",
    "reproducibility_verification_duration_seconds",
    "reproducibility_hash_computations_total",
    "reproducibility_hash_mismatches_total",
    "reproducibility_drift_detections_total",
    "reproducibility_drift_percentage",
    "reproducibility_replays_total",
    "reproducibility_replay_duration_seconds",
    "reproducibility_non_determinism_sources_total",
    "reproducibility_environment_mismatches_total",
    "reproducibility_cache_hits_total",
    "reproducibility_cache_misses_total",
    # Metric helper functions
    "record_verification",
    "record_hash_computation",
    "record_hash_mismatch",
    "record_drift",
    "record_replay",
    "record_non_determinism",
    "record_environment_mismatch",
    "record_cache_hit",
    "record_cache_miss",
    # Service setup facade
    "ReproducibilityService",
    "configure_reproducibility",
    "get_reproducibility",
    "get_router",
]
