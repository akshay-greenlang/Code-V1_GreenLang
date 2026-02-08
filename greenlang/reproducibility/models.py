# -*- coding: utf-8 -*-
"""
Reproducibility Service Data Models - AGENT-FOUND-008: Reproducibility Agent

Pydantic v2 data models for the Reproducibility SDK. Re-exports the Layer 1
enumerations and models from the foundation agent, and defines additional SDK
models for verification runs, drift baselines, replay sessions, statistics,
and request/response wrappers.

Models:
    - Re-exported enums: VerificationStatus, DriftSeverity, NonDeterminismSource
    - Re-exported Layer 1: EnvironmentFingerprint, SeedConfiguration, VersionPin,
        VersionManifest, VerificationCheck, DriftDetection, ReplayConfiguration,
        ReproducibilityInput, ReproducibilityOutput, ReproducibilityReport
    - SDK models: ArtifactHash, VerificationRun, DriftBaseline, ReplaySession,
        VerificationStatistics
    - Request/Response: HashRequest, HashResponse, DriftRequest, DriftResponse,
        ReplayRequest, ReplayResponse

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 enumerations
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.reproducibility_agent import (
    VerificationStatus,
    DriftSeverity,
    NonDeterminismSource,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 models
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.reproducibility_agent import (
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
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.reproducibility_agent import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    DEFAULT_DRIFT_SOFT_THRESHOLD,
    DEFAULT_DRIFT_HARD_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# SDK Data Models
# =============================================================================


class ArtifactHash(BaseModel):
    """Computed hash record for a data artifact.

    Stores the deterministic hash of an artifact along with metadata
    about how and when it was computed.

    Attributes:
        artifact_id: Unique identifier of the artifact.
        artifact_type: Type classification (e.g. input, output, config).
        data_hash: SHA-256 hash of the artifact data.
        computed_at: Timestamp when the hash was computed.
        algorithm: Hash algorithm used.
        normalization_applied: Whether float normalization was applied.
        provenance_hash: SHA-256 chain hash for audit trail.
    """

    artifact_id: str = Field(..., description="Unique identifier of the artifact")
    artifact_type: str = Field(
        default="generic", description="Type classification of the artifact"
    )
    data_hash: str = Field(..., description="SHA-256 hash of the artifact data")
    computed_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the hash was computed"
    )
    algorithm: str = Field(
        default="sha256", description="Hash algorithm used"
    )
    normalization_applied: bool = Field(
        default=True, description="Whether float normalization was applied"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 chain hash for audit trail"
    )

    model_config = {"extra": "forbid"}

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, v: str) -> str:
        """Validate artifact ID is non-empty."""
        if not v or not v.strip():
            raise ValueError("artifact_id must be non-empty")
        return v

    @field_validator("data_hash")
    @classmethod
    def validate_data_hash(cls, v: str) -> str:
        """Validate data hash is a valid hex string."""
        if not v or not v.strip():
            raise ValueError("data_hash must be non-empty")
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("data_hash must be a valid hex string")
        return v


class VerificationRun(BaseModel):
    """Record of a complete verification execution.

    Captures the full state and results of a reproducibility
    verification run for audit and analysis purposes.

    Attributes:
        verification_id: Unique ID for this verification run.
        execution_id: ID of the execution being verified.
        status: Overall verification status.
        input_hash: SHA-256 hash of the input data.
        output_hash: SHA-256 hash of the output data.
        environment_fingerprint_id: ID of the captured environment.
        seed_config_id: ID of the seed configuration used.
        version_manifest_id: ID of the version manifest used.
        drift_baseline_id: ID of the drift baseline compared against.
        checks: Individual verification check results.
        is_reproducible: Whether the execution is reproducible.
        processing_time_ms: Duration of verification in milliseconds.
        created_at: Timestamp of the verification run.
    """

    verification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this verification run",
    )
    execution_id: str = Field(..., description="ID of the execution being verified")
    status: VerificationStatus = Field(
        ..., description="Overall verification status"
    )
    input_hash: str = Field(default="", description="SHA-256 hash of the input data")
    output_hash: str = Field(default="", description="SHA-256 hash of the output data")
    environment_fingerprint_id: str = Field(
        default="", description="ID of the captured environment fingerprint"
    )
    seed_config_id: str = Field(
        default="", description="ID of the seed configuration used"
    )
    version_manifest_id: str = Field(
        default="", description="ID of the version manifest used"
    )
    drift_baseline_id: str = Field(
        default="", description="ID of the drift baseline compared against"
    )
    checks: List[VerificationCheck] = Field(
        default_factory=list, description="Individual verification check results"
    )
    is_reproducible: bool = Field(
        default=True, description="Whether the execution is reproducible"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Duration of verification in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp of the verification run"
    )

    model_config = {"extra": "forbid"}


class DriftBaseline(BaseModel):
    """Stored baseline for drift detection comparison.

    A named snapshot of expected output data used as a reference
    point for detecting drift in subsequent executions.

    Attributes:
        baseline_id: Unique ID for this baseline.
        name: Human-readable name.
        description: Description of the baseline purpose.
        baseline_data: Snapshot of the baseline output data.
        baseline_hash: SHA-256 hash of the baseline data.
        created_at: Timestamp when the baseline was created.
        updated_at: Timestamp of the last update.
        is_active: Whether this baseline is currently active.
    """

    baseline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this baseline",
    )
    name: str = Field(..., description="Human-readable name for the baseline")
    description: str = Field(
        default="", description="Description of the baseline purpose"
    )
    baseline_data: Dict[str, Any] = Field(
        default_factory=dict, description="Snapshot of the baseline output data"
    )
    baseline_hash: str = Field(
        default="", description="SHA-256 hash of the baseline data"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the baseline was created"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp of the last update"
    )
    is_active: bool = Field(
        default=True, description="Whether this baseline is currently active"
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate baseline name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class ReplaySession(BaseModel):
    """Record of a replay execution session.

    Captures the results of re-executing a previous run with the
    same inputs, environment, seeds, and versions to verify
    reproducibility.

    Attributes:
        replay_id: Unique ID for this replay session.
        original_execution_id: ID of the original execution.
        replay_execution_id: ID assigned to the replay execution.
        environment_match: Whether the environment matched.
        seed_match: Whether the seed configuration matched.
        version_match: Whether the version manifest matched.
        output_match: Whether the outputs matched within tolerance.
        replay_status: Overall replay verification status.
        started_at: Timestamp when the replay started.
        completed_at: Timestamp when the replay completed.
    """

    replay_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this replay session",
    )
    original_execution_id: str = Field(
        ..., description="ID of the original execution"
    )
    replay_execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="ID assigned to the replay execution",
    )
    environment_match: VerificationCheck = Field(
        ..., description="Whether the environment matched"
    )
    seed_match: VerificationCheck = Field(
        ..., description="Whether the seed configuration matched"
    )
    version_match: Dict[str, VerificationCheck] = Field(
        default_factory=dict, description="Whether the version manifest matched"
    )
    output_match: Optional[VerificationCheck] = Field(
        None, description="Whether the outputs matched within tolerance"
    )
    replay_status: VerificationStatus = Field(
        ..., description="Overall replay verification status"
    )
    started_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the replay started"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when the replay completed"
    )

    model_config = {"extra": "forbid"}


class VerificationStatistics(BaseModel):
    """Aggregated statistics for reproducibility verification.

    Attributes:
        total_verifications: Total number of verification runs executed.
        pass_count: Number of verifications that passed.
        fail_count: Number of verifications that failed.
        warning_count: Number of verifications with warnings.
        avg_processing_time_ms: Average verification processing time.
        drift_detections: Total number of drift detections performed.
        replay_count: Total number of replay sessions executed.
        non_determinism_count: Total non-determinism source detections.
    """

    total_verifications: int = Field(
        default=0, description="Total number of verification runs executed"
    )
    pass_count: int = Field(
        default=0, description="Number of verifications that passed"
    )
    fail_count: int = Field(
        default=0, description="Number of verifications that failed"
    )
    warning_count: int = Field(
        default=0, description="Number of verifications with warnings"
    )
    avg_processing_time_ms: float = Field(
        default=0.0, description="Average verification processing time"
    )
    drift_detections: int = Field(
        default=0, description="Total number of drift detections performed"
    )
    replay_count: int = Field(
        default=0, description="Total number of replay sessions executed"
    )
    non_determinism_count: int = Field(
        default=0, description="Total non-determinism source detections"
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request / Response Models
# =============================================================================


class HashRequest(BaseModel):
    """Request body for computing a deterministic hash.

    Attributes:
        data: Arbitrary data to hash.
        algorithm: Hash algorithm to use (default sha256).
        normalize_floats: Whether to normalize floats before hashing.
    """

    data: Any = Field(..., description="Arbitrary data to hash")
    algorithm: str = Field(
        default="sha256", description="Hash algorithm to use"
    )
    normalize_floats: bool = Field(
        default=True, description="Whether to normalize floats before hashing"
    )

    model_config = {"extra": "forbid"}


class HashResponse(BaseModel):
    """Response body from a hash computation.

    Attributes:
        data_hash: Computed SHA-256 hash.
        algorithm: Hash algorithm used.
        normalization_applied: Whether normalization was applied.
    """

    data_hash: str = Field(..., description="Computed SHA-256 hash")
    algorithm: str = Field(
        default="sha256", description="Hash algorithm used"
    )
    normalization_applied: bool = Field(
        default=True, description="Whether normalization was applied"
    )

    model_config = {"extra": "forbid"}


class DriftRequest(BaseModel):
    """Request body for drift detection.

    Attributes:
        baseline_id: ID of the baseline to compare against.
        current_data: Current output data to check for drift.
        soft_threshold: Soft threshold for drift warning.
        hard_threshold: Hard threshold for drift failure.
        tolerance: Absolute tolerance for numeric comparisons.
    """

    baseline_id: Optional[str] = Field(
        None, description="ID of the baseline to compare against"
    )
    baseline_data: Optional[Dict[str, Any]] = Field(
        None, description="Inline baseline data (if baseline_id not provided)"
    )
    current_data: Dict[str, Any] = Field(
        ..., description="Current output data to check for drift"
    )
    soft_threshold: float = Field(
        default=DEFAULT_DRIFT_SOFT_THRESHOLD,
        description="Soft threshold for drift warning",
    )
    hard_threshold: float = Field(
        default=DEFAULT_DRIFT_HARD_THRESHOLD,
        description="Hard threshold for drift failure",
    )
    tolerance: float = Field(
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        description="Absolute tolerance for numeric comparisons",
    )

    model_config = {"extra": "forbid"}

    @field_validator("soft_threshold", "hard_threshold", "tolerance")
    @classmethod
    def validate_positive_threshold(cls, v: float) -> float:
        """Validate thresholds are non-negative."""
        if v < 0:
            raise ValueError("Threshold must be non-negative")
        return v


class DriftResponse(BaseModel):
    """Response body from drift detection.

    Attributes:
        drift_detection: Full drift detection result.
        baseline_id: ID of the baseline used.
    """

    drift_detection: DriftDetection = Field(
        ..., description="Full drift detection result"
    )
    baseline_id: str = Field(
        default="", description="ID of the baseline used"
    )

    model_config = {"extra": "forbid"}


class ReplayRequest(BaseModel):
    """Request body for executing a replay.

    Attributes:
        original_execution_id: ID of the original execution to replay.
        captured_inputs: Captured input data from the original execution.
        captured_environment: Environment fingerprint from the original.
        captured_seeds: Seed configuration from the original.
        captured_versions: Version manifest from the original.
        strict_mode: Whether to fail on any environment mismatch.
        absolute_tolerance: Absolute tolerance for output comparison.
        relative_tolerance: Relative tolerance for output comparison.
    """

    original_execution_id: str = Field(
        ..., description="ID of the original execution to replay"
    )
    captured_inputs: Dict[str, Any] = Field(
        ..., description="Captured input data from the original execution"
    )
    captured_environment: EnvironmentFingerprint = Field(
        ..., description="Environment fingerprint from the original"
    )
    captured_seeds: SeedConfiguration = Field(
        ..., description="Seed configuration from the original"
    )
    captured_versions: VersionManifest = Field(
        ..., description="Version manifest from the original"
    )
    strict_mode: bool = Field(
        default=False, description="Whether to fail on any environment mismatch"
    )
    absolute_tolerance: float = Field(
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        description="Absolute tolerance for output comparison",
    )
    relative_tolerance: float = Field(
        default=DEFAULT_RELATIVE_TOLERANCE,
        description="Relative tolerance for output comparison",
    )

    model_config = {"extra": "forbid"}


class ReplayResponse(BaseModel):
    """Response body from a replay execution.

    Attributes:
        replay_session: Full replay session record.
    """

    replay_session: ReplaySession = Field(
        ..., description="Full replay session record"
    )

    model_config = {"extra": "forbid"}


__all__ = [
    # Re-exported enums
    "VerificationStatus",
    "DriftSeverity",
    "NonDeterminismSource",
    # Re-exported Layer 1 models
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
    # Re-exported constants
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
]
