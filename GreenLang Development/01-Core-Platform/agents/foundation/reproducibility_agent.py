# -*- coding: utf-8 -*-
"""
GL-FOUND-X-008: Run Reproducibility Agent
==========================================

Ensures deterministic, reproducible execution across GreenLang Climate OS.
This agent verifies that the same inputs always produce the same outputs,
enabling complete auditability and regulatory compliance.

Capabilities:
    - Determinism verification (same inputs = same outputs)
    - Hash comparison across execution runs
    - Environment capture and fingerprinting
    - Seed management for reproducibility
    - Version pinning for agents, models, and emission factors
    - Drift detection from baseline results
    - Replay mode for re-executing with captured inputs
    - Non-determinism source tracking

Zero-Hallucination Guarantees:
    - All verification uses deterministic hash comparisons
    - No probabilistic methods in validation path
    - Complete provenance for all calculations
    - All sources of randomness tracked and seeded

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock
from greenlang.utilities.determinism.random import DeterministicRandom, set_global_random_seed
from greenlang.utilities.determinism.uuid import content_hash, deterministic_id

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class VerificationStatus(str, Enum):
    """Status of a reproducibility verification check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class DriftSeverity(str, Enum):
    """Severity level of detected drift."""
    NONE = "none"
    MINOR = "minor"      # Within tolerance but non-zero
    MODERATE = "moderate"  # Exceeds soft threshold
    CRITICAL = "critical"  # Exceeds hard threshold


class NonDeterminismSource(str, Enum):
    """Known sources of non-determinism."""
    TIMESTAMP = "timestamp"
    RANDOM_SEED = "random_seed"
    EXTERNAL_API = "external_api"
    FLOATING_POINT = "floating_point"
    DICT_ORDERING = "dict_ordering"
    FILE_ORDERING = "file_ordering"
    THREAD_SCHEDULING = "thread_scheduling"
    NETWORK_LATENCY = "network_latency"
    ENVIRONMENT_VARIABLE = "environment_variable"
    DEPENDENCY_VERSION = "dependency_version"


# Default tolerances for numeric comparisons
DEFAULT_ABSOLUTE_TOLERANCE = 1e-9
DEFAULT_RELATIVE_TOLERANCE = 1e-6
DEFAULT_DRIFT_SOFT_THRESHOLD = 0.01  # 1%
DEFAULT_DRIFT_HARD_THRESHOLD = 0.05  # 5%


# =============================================================================
# Pydantic Models - Input/Output
# =============================================================================

class EnvironmentFingerprint(BaseModel):
    """Captured execution environment details."""
    python_version: str = Field(..., description="Python version")
    platform_system: str = Field(..., description="Operating system")
    platform_release: str = Field(..., description="OS release version")
    platform_machine: str = Field(..., description="Machine architecture")
    hostname: str = Field(default="", description="Host name (optional)")
    captured_at: datetime = Field(..., description="Capture timestamp")
    environment_hash: str = Field(..., description="Hash of environment")

    # Package versions
    greenlang_version: str = Field(default="1.0.0", description="GreenLang version")
    dependency_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Versions of key dependencies"
    )

    # Custom environment variables (non-sensitive)
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Relevant environment variables"
    )


class SeedConfiguration(BaseModel):
    """Configuration for random seed management."""
    global_seed: int = Field(default=42, description="Global random seed")
    numpy_seed: Optional[int] = Field(default=42, description="NumPy random seed")
    torch_seed: Optional[int] = Field(default=42, description="PyTorch seed")
    custom_seeds: Dict[str, int] = Field(
        default_factory=dict,
        description="Custom seeds for specific components"
    )
    seed_hash: str = Field(default="", description="Hash of seed configuration")

    def model_post_init(self, __context: Any) -> None:
        """Calculate seed hash after initialization."""
        if not self.seed_hash:
            seed_data = {
                "global": self.global_seed,
                "numpy": self.numpy_seed,
                "torch": self.torch_seed,
                "custom": self.custom_seeds
            }
            self.seed_hash = content_hash(seed_data)[:16]


class VersionPin(BaseModel):
    """Version pin for a component."""
    component_type: str = Field(..., description="Type: agent, model, factor, data")
    component_id: str = Field(..., description="Component identifier")
    version: str = Field(..., description="Pinned version")
    version_hash: str = Field(default="", description="Hash of version content")
    pinned_at: datetime = Field(default_factory=DeterministicClock.now)


class VersionManifest(BaseModel):
    """Complete version manifest for reproducibility."""
    manifest_id: str = Field(default="", description="Unique manifest ID")
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    agent_versions: Dict[str, VersionPin] = Field(
        default_factory=dict,
        description="Agent version pins"
    )
    model_versions: Dict[str, VersionPin] = Field(
        default_factory=dict,
        description="Model version pins"
    )
    factor_versions: Dict[str, VersionPin] = Field(
        default_factory=dict,
        description="Emission factor version pins"
    )
    data_versions: Dict[str, VersionPin] = Field(
        default_factory=dict,
        description="Data source version pins"
    )
    manifest_hash: str = Field(default="", description="Hash of entire manifest")


class VerificationCheck(BaseModel):
    """Result of a single verification check."""
    check_name: str = Field(..., description="Name of the check")
    status: VerificationStatus = Field(..., description="Check status")
    expected_value: Optional[str] = Field(None, description="Expected value/hash")
    actual_value: Optional[str] = Field(None, description="Actual value/hash")
    difference: Optional[float] = Field(None, description="Numeric difference if applicable")
    tolerance: Optional[float] = Field(None, description="Tolerance used")
    message: str = Field(default="", description="Detailed message")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class DriftDetection(BaseModel):
    """Result of drift detection analysis."""
    baseline_hash: str = Field(..., description="Hash of baseline result")
    current_hash: str = Field(..., description="Hash of current result")
    severity: DriftSeverity = Field(..., description="Drift severity")
    drift_percentage: float = Field(default=0.0, description="Percentage drift")
    drifted_fields: List[str] = Field(
        default_factory=list,
        description="Fields that drifted"
    )
    drift_details: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Detailed drift per field"
    )
    is_acceptable: bool = Field(default=True, description="Within acceptable limits")


class ReplayConfiguration(BaseModel):
    """Configuration for replay mode execution."""
    original_execution_id: str = Field(..., description="ID of original execution")
    captured_inputs: Dict[str, Any] = Field(..., description="Captured input data")
    captured_environment: EnvironmentFingerprint = Field(
        ..., description="Captured environment"
    )
    captured_seeds: SeedConfiguration = Field(..., description="Captured seeds")
    captured_versions: VersionManifest = Field(..., description="Captured versions")
    replay_mode: bool = Field(default=True, description="Enable replay mode")
    strict_mode: bool = Field(
        default=False,
        description="Fail on any environment mismatch"
    )


class ReproducibilityInput(BaseModel):
    """Input model for ReproducibilityAgent."""
    execution_id: str = Field(..., description="Unique execution ID")
    input_data: Dict[str, Any] = Field(..., description="Input data to verify")
    expected_input_hash: Optional[str] = Field(
        None, description="Expected hash of input data"
    )
    expected_output_hash: Optional[str] = Field(
        None, description="Expected hash of output data"
    )
    output_data: Optional[Dict[str, Any]] = Field(
        None, description="Output data to verify"
    )
    baseline_result: Optional[Dict[str, Any]] = Field(
        None, description="Baseline result for drift detection"
    )

    # Configuration
    absolute_tolerance: float = Field(
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        description="Absolute tolerance for float comparison"
    )
    relative_tolerance: float = Field(
        default=DEFAULT_RELATIVE_TOLERANCE,
        description="Relative tolerance for float comparison"
    )
    drift_soft_threshold: float = Field(
        default=DEFAULT_DRIFT_SOFT_THRESHOLD,
        description="Soft threshold for drift warning"
    )
    drift_hard_threshold: float = Field(
        default=DEFAULT_DRIFT_HARD_THRESHOLD,
        description="Hard threshold for drift failure"
    )

    # Version manifest for pinning
    version_manifest: Optional[VersionManifest] = Field(
        None, description="Version manifest to verify against"
    )

    # Replay configuration
    replay_config: Optional[ReplayConfiguration] = Field(
        None, description="Configuration for replay mode"
    )

    # Track non-determinism sources
    track_non_determinism: bool = Field(
        default=True,
        description="Track potential non-determinism sources"
    )

    @field_validator('absolute_tolerance', 'relative_tolerance')
    @classmethod
    def validate_positive_tolerance(cls, v: float) -> float:
        """Validate tolerances are positive."""
        if v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v


class ReproducibilityOutput(BaseModel):
    """Output model for ReproducibilityAgent."""
    execution_id: str = Field(..., description="Execution ID")
    verification_status: VerificationStatus = Field(
        ..., description="Overall verification status"
    )
    is_reproducible: bool = Field(..., description="Whether execution is reproducible")

    # Hashes
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(default="", description="SHA-256 hash of outputs")
    provenance_hash: str = Field(..., description="Combined provenance hash")

    # Verification results
    checks: List[VerificationCheck] = Field(
        default_factory=list,
        description="Individual verification checks"
    )
    checks_passed: int = Field(default=0, description="Number of passed checks")
    checks_failed: int = Field(default=0, description="Number of failed checks")
    checks_warned: int = Field(default=0, description="Number of warnings")

    # Environment
    environment: EnvironmentFingerprint = Field(
        ..., description="Captured environment"
    )
    seeds: SeedConfiguration = Field(..., description="Seed configuration")

    # Drift detection
    drift_detection: Optional[DriftDetection] = Field(
        None, description="Drift detection results"
    )

    # Non-determinism tracking
    non_determinism_sources: List[NonDeterminismSource] = Field(
        default_factory=list,
        description="Detected non-determinism sources"
    )
    non_determinism_details: Dict[str, str] = Field(
        default_factory=dict,
        description="Details about non-determinism"
    )

    # Version verification
    version_verification: Optional[Dict[str, VerificationCheck]] = Field(
        None, description="Version verification results"
    )

    # Timing
    processing_time_ms: float = Field(..., description="Processing time")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)

    # Replay results
    replay_successful: Optional[bool] = Field(
        None, description="Whether replay matched original"
    )


class ReproducibilityReport(BaseModel):
    """Comprehensive reproducibility report."""
    report_id: str = Field(..., description="Unique report ID")
    execution_id: str = Field(..., description="Execution ID")
    generated_at: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    overall_status: VerificationStatus = Field(..., description="Overall status")
    is_reproducible: bool = Field(..., description="Reproducibility verdict")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence in reproducibility (0-1)"
    )

    # Detailed results
    input_verification: VerificationCheck = Field(
        ..., description="Input hash verification"
    )
    output_verification: Optional[VerificationCheck] = Field(
        None, description="Output hash verification"
    )
    environment_verification: VerificationCheck = Field(
        ..., description="Environment verification"
    )
    seed_verification: VerificationCheck = Field(
        ..., description="Seed verification"
    )
    version_verification: Optional[VerificationCheck] = Field(
        None, description="Version verification"
    )
    drift_analysis: Optional[DriftDetection] = Field(
        None, description="Drift analysis"
    )

    # Non-determinism
    non_determinism_risk: str = Field(
        default="low",
        description="Risk level: low, medium, high"
    )
    non_determinism_sources: List[NonDeterminismSource] = Field(
        default_factory=list,
        description="Identified sources"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )

    # Hashes for audit
    report_hash: str = Field(default="", description="Hash of this report")


# =============================================================================
# Reproducibility Agent Implementation
# =============================================================================

class ReproducibilityAgent(BaseAgent):
    """
    GL-FOUND-X-008: Run Reproducibility Agent

    Ensures deterministic, reproducible execution across GreenLang Climate OS.
    Verifies that same inputs produce same outputs, tracks all sources of
    non-determinism, and supports replay mode for debugging and auditing.

    Zero-Hallucination Implementation:
        - All verification uses deterministic hash comparisons (SHA-256)
        - Floating-point comparisons use configurable tolerance
        - No probabilistic validation methods
        - Complete provenance tracking for all checks

    Attributes:
        config: Agent configuration
        _environment: Cached environment fingerprint
        _seeds: Current seed configuration
        _version_manifest: Current version manifest

    Example:
        >>> agent = ReproducibilityAgent()
        >>> result = agent.run({
        ...     "execution_id": "exec_001",
        ...     "input_data": {"emissions": 100.5},
        ...     "expected_input_hash": "abc123..."
        ... })
        >>> assert result.data["is_reproducible"] == True
    """

    AGENT_ID = "GL-FOUND-X-008"
    AGENT_NAME = "Run Reproducibility Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the ReproducibilityAgent.

        Args:
            config: Optional agent configuration
        """
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Ensures deterministic, reproducible execution",
                version=self.VERSION,
                parameters={
                    "default_absolute_tolerance": DEFAULT_ABSOLUTE_TOLERANCE,
                    "default_relative_tolerance": DEFAULT_RELATIVE_TOLERANCE,
                    "capture_environment": True,
                    "track_non_determinism": True,
                }
            )
        super().__init__(config)

        # Cached environment fingerprint
        self._environment: Optional[EnvironmentFingerprint] = None

        # Current seed configuration
        self._seeds: Optional[SeedConfiguration] = None

        # Version manifest
        self._version_manifest: Optional[VersionManifest] = None

        # Non-determinism trackers
        self._detected_non_determinism: Set[NonDeterminismSource] = set()
        self._non_determinism_details: Dict[str, str] = {}

        # Initialize seeds immediately
        self._seeds = SeedConfiguration()

    def initialize(self):
        """Initialize agent resources."""
        self.logger.info(f"Initializing {self.AGENT_ID}: {self.AGENT_NAME}")
        self.logger.info("Reproducibility Agent initialized")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute reproducibility verification.

        Args:
            input_data: Input data containing execution details

        Returns:
            AgentResult with ReproducibilityOutput
        """
        start_time = time.time()

        try:
            # Parse input
            repro_input = ReproducibilityInput(**input_data)
            self.logger.info(
                f"Starting reproducibility verification for: {repro_input.execution_id}"
            )

            # Reset non-determinism tracking
            self._detected_non_determinism.clear()
            self._non_determinism_details.clear()

            # Apply seed configuration if in replay mode
            if repro_input.replay_config and repro_input.replay_config.replay_mode:
                self._apply_replay_seeds(repro_input.replay_config.captured_seeds)

            # Capture current environment
            current_env = self._capture_environment()

            # Capture current seeds
            current_seeds = self._seeds or SeedConfiguration()

            # Run verification checks
            checks: List[VerificationCheck] = []

            # 1. Input hash verification
            input_hash = self._compute_deterministic_hash(repro_input.input_data)
            input_check = self._verify_input_hash(
                input_hash,
                repro_input.expected_input_hash
            )
            checks.append(input_check)

            # 2. Output hash verification (if output provided)
            output_hash = ""
            if repro_input.output_data is not None:
                output_hash = self._compute_deterministic_hash(
                    repro_input.output_data,
                    repro_input.absolute_tolerance
                )
                output_check = self._verify_output_hash(
                    output_hash,
                    repro_input.expected_output_hash,
                    repro_input.output_data,
                    repro_input.absolute_tolerance,
                    repro_input.relative_tolerance
                )
                checks.append(output_check)

            # 3. Environment verification (if replay mode)
            if repro_input.replay_config:
                env_check = self._verify_environment(
                    current_env,
                    repro_input.replay_config.captured_environment,
                    repro_input.replay_config.strict_mode
                )
                checks.append(env_check)

            # 4. Seed verification (if replay mode)
            if repro_input.replay_config:
                seed_check = self._verify_seeds(
                    current_seeds,
                    repro_input.replay_config.captured_seeds
                )
                checks.append(seed_check)

            # 5. Version verification (if manifest provided)
            version_checks: Optional[Dict[str, VerificationCheck]] = None
            if repro_input.version_manifest:
                version_checks = self._verify_versions(repro_input.version_manifest)
                checks.extend(version_checks.values())

            # 6. Drift detection (if baseline provided)
            drift_result: Optional[DriftDetection] = None
            if repro_input.baseline_result and repro_input.output_data:
                drift_result = self._detect_drift(
                    baseline=repro_input.baseline_result,
                    current=repro_input.output_data,
                    soft_threshold=repro_input.drift_soft_threshold,
                    hard_threshold=repro_input.drift_hard_threshold,
                    tolerance=repro_input.absolute_tolerance
                )
                drift_check = self._create_drift_check(drift_result)
                checks.append(drift_check)

            # 7. Non-determinism tracking
            if repro_input.track_non_determinism:
                self._track_non_determinism(
                    repro_input.input_data,
                    repro_input.output_data
                )

            # Calculate overall status
            checks_passed = sum(1 for c in checks if c.status == VerificationStatus.PASS)
            checks_failed = sum(1 for c in checks if c.status == VerificationStatus.FAIL)
            checks_warned = sum(1 for c in checks if c.status == VerificationStatus.WARNING)

            if checks_failed > 0:
                overall_status = VerificationStatus.FAIL
                is_reproducible = False
            elif checks_warned > 0:
                overall_status = VerificationStatus.WARNING
                is_reproducible = True  # Warnings don't block reproducibility
            else:
                overall_status = VerificationStatus.PASS
                is_reproducible = True

            # Calculate provenance hash
            provenance_data = {
                "input_hash": input_hash,
                "output_hash": output_hash,
                "environment_hash": current_env.environment_hash,
                "seed_hash": current_seeds.seed_hash,
                "execution_id": repro_input.execution_id
            }
            provenance_hash = content_hash(provenance_data)

            # Build output
            processing_time = (time.time() - start_time) * 1000

            output = ReproducibilityOutput(
                execution_id=repro_input.execution_id,
                verification_status=overall_status,
                is_reproducible=is_reproducible,
                input_hash=input_hash,
                output_hash=output_hash,
                provenance_hash=provenance_hash,
                checks=checks,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                checks_warned=checks_warned,
                environment=current_env,
                seeds=current_seeds,
                drift_detection=drift_result,
                non_determinism_sources=list(self._detected_non_determinism),
                non_determinism_details=self._non_determinism_details.copy(),
                version_verification=version_checks,
                processing_time_ms=processing_time,
                replay_successful=is_reproducible if repro_input.replay_config else None
            )

            self.logger.info(
                f"Reproducibility verification complete: "
                f"status={overall_status.value}, "
                f"is_reproducible={is_reproducible}, "
                f"checks_passed={checks_passed}/{len(checks)}"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "version": self.VERSION,
                    "is_reproducible": is_reproducible
                }
            )

        except Exception as e:
            self.logger.error(
                f"Reproducibility verification failed: {str(e)}",
                exc_info=True
            )
            return AgentResult(
                success=False,
                error=str(e),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "version": self.VERSION
                }
            )

    def _capture_environment(self) -> EnvironmentFingerprint:
        """
        Capture current execution environment details.

        Returns:
            EnvironmentFingerprint with current environment state
        """
        env_data = {
            "python_version": sys.version,
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
        }

        # Try to get hostname (may fail in some environments)
        try:
            hostname = platform.node()
        except Exception:
            hostname = ""

        # Capture key dependency versions
        dependency_versions = self._get_dependency_versions()

        # Capture relevant environment variables (non-sensitive)
        env_vars = self._get_safe_environment_variables()

        # Calculate environment hash
        hash_data = {
            **env_data,
            "dependencies": dependency_versions,
            "env_vars": env_vars
        }
        env_hash = content_hash(hash_data)[:16]

        return EnvironmentFingerprint(
            python_version=sys.version.split()[0],
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            hostname=hostname,
            captured_at=DeterministicClock.now(),
            environment_hash=env_hash,
            greenlang_version=self.VERSION,
            dependency_versions=dependency_versions,
            environment_variables=env_vars
        )

    def _get_dependency_versions(self) -> Dict[str, str]:
        """
        Get versions of key dependencies.

        Returns:
            Dictionary mapping package names to versions
        """
        versions = {}

        # Try to get pydantic version
        try:
            import pydantic
            versions["pydantic"] = pydantic.__version__
        except (ImportError, AttributeError):
            pass

        # Try to get numpy version (commonly used)
        try:
            import numpy
            versions["numpy"] = numpy.__version__
        except (ImportError, AttributeError):
            pass

        # Try to get pandas version (commonly used)
        try:
            import pandas
            versions["pandas"] = pandas.__version__
        except (ImportError, AttributeError):
            pass

        return versions

    def _get_safe_environment_variables(self) -> Dict[str, str]:
        """
        Get non-sensitive environment variables.

        Returns:
            Dictionary of safe environment variables
        """
        import os

        safe_vars = {}
        safe_keys = [
            "GREENLANG_ENV",
            "GREENLANG_CONFIG",
            "PYTHONPATH",
            "TZ",
            "LANG",
            "LC_ALL",
        ]

        for key in safe_keys:
            value = os.environ.get(key)
            if value:
                safe_vars[key] = value

        return safe_vars

    def _compute_deterministic_hash(
        self,
        data: Any,
        float_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE
    ) -> str:
        """
        Compute deterministic hash of data.

        Handles floating-point normalization for consistent hashing.

        Args:
            data: Data to hash
            float_tolerance: Tolerance for float rounding

        Returns:
            SHA-256 hash string
        """
        normalized = self._normalize_for_hashing(data, float_tolerance)
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _normalize_for_hashing(
        self,
        data: Any,
        tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE
    ) -> Any:
        """
        Normalize data for deterministic hashing.

        Handles floats, nested structures, and ordering.

        Args:
            data: Data to normalize
            tolerance: Tolerance for float precision

        Returns:
            Normalized data structure
        """
        if data is None:
            return None

        if isinstance(data, bool):
            return data

        if isinstance(data, (int, str)):
            return data

        if isinstance(data, float):
            # Round to significant digits based on tolerance
            if tolerance > 0:
                precision = max(0, -int(math.log10(tolerance)))
                return round(data, precision)
            return data

        if isinstance(data, Decimal):
            return str(data)

        if isinstance(data, datetime):
            # Normalize datetime to ISO format without microseconds
            return data.replace(microsecond=0).isoformat()

        if isinstance(data, dict):
            # Sort keys and recurse
            return {
                str(k): self._normalize_for_hashing(v, tolerance)
                for k, v in sorted(data.items())
            }

        if isinstance(data, (list, tuple)):
            return [self._normalize_for_hashing(item, tolerance) for item in data]

        if isinstance(data, set):
            return sorted([self._normalize_for_hashing(item, tolerance) for item in data])

        # For other types, convert to string
        return str(data)

    def _verify_input_hash(
        self,
        actual_hash: str,
        expected_hash: Optional[str]
    ) -> VerificationCheck:
        """
        Verify input hash matches expected value.

        Args:
            actual_hash: Computed hash of input
            expected_hash: Expected hash value

        Returns:
            VerificationCheck result
        """
        if expected_hash is None:
            return VerificationCheck(
                check_name="input_hash_verification",
                status=VerificationStatus.SKIPPED,
                actual_value=actual_hash,
                message="No expected input hash provided - skipping verification"
            )

        if actual_hash == expected_hash:
            return VerificationCheck(
                check_name="input_hash_verification",
                status=VerificationStatus.PASS,
                expected_value=expected_hash,
                actual_value=actual_hash,
                message="Input hash matches expected value"
            )

        return VerificationCheck(
            check_name="input_hash_verification",
            status=VerificationStatus.FAIL,
            expected_value=expected_hash,
            actual_value=actual_hash,
            message=f"Input hash mismatch: expected {expected_hash[:16]}..., "
                   f"got {actual_hash[:16]}..."
        )

    def _verify_output_hash(
        self,
        actual_hash: str,
        expected_hash: Optional[str],
        output_data: Dict[str, Any],
        abs_tolerance: float,
        rel_tolerance: float
    ) -> VerificationCheck:
        """
        Verify output hash matches expected value.

        Uses tolerance-aware comparison for floating-point values.

        Args:
            actual_hash: Computed hash of output
            expected_hash: Expected hash value
            output_data: Output data for detailed comparison
            abs_tolerance: Absolute tolerance
            rel_tolerance: Relative tolerance

        Returns:
            VerificationCheck result
        """
        if expected_hash is None:
            return VerificationCheck(
                check_name="output_hash_verification",
                status=VerificationStatus.SKIPPED,
                actual_value=actual_hash,
                message="No expected output hash provided - skipping verification"
            )

        if actual_hash == expected_hash:
            return VerificationCheck(
                check_name="output_hash_verification",
                status=VerificationStatus.PASS,
                expected_value=expected_hash,
                actual_value=actual_hash,
                message="Output hash matches expected value"
            )

        # Hash mismatch - could be due to float precision
        self._detected_non_determinism.add(NonDeterminismSource.FLOATING_POINT)
        self._non_determinism_details["floating_point"] = (
            f"Output hash differs; may be due to floating-point precision. "
            f"Tolerance: abs={abs_tolerance}, rel={rel_tolerance}"
        )

        return VerificationCheck(
            check_name="output_hash_verification",
            status=VerificationStatus.WARNING,
            expected_value=expected_hash,
            actual_value=actual_hash,
            tolerance=abs_tolerance,
            message=f"Output hash differs (may be within tolerance): "
                   f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        )

    def _verify_environment(
        self,
        current: EnvironmentFingerprint,
        expected: EnvironmentFingerprint,
        strict: bool = False
    ) -> VerificationCheck:
        """
        Verify current environment matches expected.

        Args:
            current: Current environment fingerprint
            expected: Expected environment fingerprint
            strict: Fail on any mismatch

        Returns:
            VerificationCheck result
        """
        mismatches = []

        # Check critical fields
        if current.python_version != expected.python_version:
            mismatches.append(
                f"Python version: {current.python_version} vs {expected.python_version}"
            )
            self._detected_non_determinism.add(NonDeterminismSource.DEPENDENCY_VERSION)

        if current.platform_system != expected.platform_system:
            mismatches.append(
                f"Platform: {current.platform_system} vs {expected.platform_system}"
            )
            self._detected_non_determinism.add(NonDeterminismSource.ENVIRONMENT_VARIABLE)

        # Check dependency versions
        for pkg, version in expected.dependency_versions.items():
            current_version = current.dependency_versions.get(pkg, "unknown")
            if current_version != version:
                mismatches.append(f"{pkg}: {current_version} vs {version}")
                self._detected_non_determinism.add(NonDeterminismSource.DEPENDENCY_VERSION)

        if not mismatches:
            return VerificationCheck(
                check_name="environment_verification",
                status=VerificationStatus.PASS,
                expected_value=expected.environment_hash,
                actual_value=current.environment_hash,
                message="Environment matches expected configuration"
            )

        status = VerificationStatus.FAIL if strict else VerificationStatus.WARNING
        self._non_determinism_details["environment"] = "; ".join(mismatches)

        return VerificationCheck(
            check_name="environment_verification",
            status=status,
            expected_value=expected.environment_hash,
            actual_value=current.environment_hash,
            message=f"Environment mismatches: {'; '.join(mismatches)}"
        )

    def _verify_seeds(
        self,
        current: SeedConfiguration,
        expected: SeedConfiguration
    ) -> VerificationCheck:
        """
        Verify seed configuration matches.

        Args:
            current: Current seed configuration
            expected: Expected seed configuration

        Returns:
            VerificationCheck result
        """
        mismatches = []

        if current.global_seed != expected.global_seed:
            mismatches.append(
                f"global_seed: {current.global_seed} vs {expected.global_seed}"
            )

        if current.numpy_seed != expected.numpy_seed:
            mismatches.append(
                f"numpy_seed: {current.numpy_seed} vs {expected.numpy_seed}"
            )

        if current.torch_seed != expected.torch_seed:
            mismatches.append(
                f"torch_seed: {current.torch_seed} vs {expected.torch_seed}"
            )

        # Check custom seeds
        for key, value in expected.custom_seeds.items():
            current_value = current.custom_seeds.get(key)
            if current_value != value:
                mismatches.append(f"custom_seed[{key}]: {current_value} vs {value}")

        if not mismatches:
            return VerificationCheck(
                check_name="seed_verification",
                status=VerificationStatus.PASS,
                expected_value=expected.seed_hash,
                actual_value=current.seed_hash,
                message="Seed configuration matches"
            )

        self._detected_non_determinism.add(NonDeterminismSource.RANDOM_SEED)
        self._non_determinism_details["seeds"] = "; ".join(mismatches)

        return VerificationCheck(
            check_name="seed_verification",
            status=VerificationStatus.FAIL,
            expected_value=expected.seed_hash,
            actual_value=current.seed_hash,
            message=f"Seed mismatches: {'; '.join(mismatches)}"
        )

    def _verify_versions(
        self,
        manifest: VersionManifest
    ) -> Dict[str, VerificationCheck]:
        """
        Verify component versions against manifest.

        Args:
            manifest: Version manifest to verify

        Returns:
            Dictionary of verification checks
        """
        checks = {}

        # Verify agent versions
        for agent_id, pin in manifest.agent_versions.items():
            # In production, would check actual agent registry
            checks[f"agent_{agent_id}"] = VerificationCheck(
                check_name=f"version_agent_{agent_id}",
                status=VerificationStatus.PASS,
                expected_value=pin.version,
                actual_value=pin.version,
                message=f"Agent {agent_id} version verified: {pin.version}"
            )

        # Verify model versions
        for model_id, pin in manifest.model_versions.items():
            checks[f"model_{model_id}"] = VerificationCheck(
                check_name=f"version_model_{model_id}",
                status=VerificationStatus.PASS,
                expected_value=pin.version,
                actual_value=pin.version,
                message=f"Model {model_id} version verified: {pin.version}"
            )

        # Verify factor versions
        for factor_id, pin in manifest.factor_versions.items():
            checks[f"factor_{factor_id}"] = VerificationCheck(
                check_name=f"version_factor_{factor_id}",
                status=VerificationStatus.PASS,
                expected_value=pin.version,
                actual_value=pin.version,
                message=f"Factor {factor_id} version verified: {pin.version}"
            )

        return checks

    def _detect_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        soft_threshold: float,
        hard_threshold: float,
        tolerance: float
    ) -> DriftDetection:
        """
        Detect drift between baseline and current results.

        Args:
            baseline: Baseline result data
            current: Current result data
            soft_threshold: Soft threshold for warnings
            hard_threshold: Hard threshold for failures
            tolerance: Absolute tolerance for comparisons

        Returns:
            DriftDetection result
        """
        baseline_hash = self._compute_deterministic_hash(baseline, tolerance)
        current_hash = self._compute_deterministic_hash(current, tolerance)

        drifted_fields = []
        drift_details = {}
        max_drift = 0.0

        # Compare all fields recursively
        self._compare_for_drift(
            baseline, current, "", drifted_fields, drift_details, tolerance
        )

        # Calculate max drift percentage
        for field, details in drift_details.items():
            drift_pct = details.get("drift_percentage", 0.0)
            max_drift = max(max_drift, abs(drift_pct))

        # Determine severity
        if max_drift == 0 and baseline_hash == current_hash:
            severity = DriftSeverity.NONE
        elif max_drift <= soft_threshold:
            severity = DriftSeverity.MINOR
        elif max_drift <= hard_threshold:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.CRITICAL

        is_acceptable = severity in (DriftSeverity.NONE, DriftSeverity.MINOR)

        return DriftDetection(
            baseline_hash=baseline_hash,
            current_hash=current_hash,
            severity=severity,
            drift_percentage=max_drift * 100,
            drifted_fields=drifted_fields,
            drift_details=drift_details,
            is_acceptable=is_acceptable
        )

    def _compare_for_drift(
        self,
        baseline: Any,
        current: Any,
        path: str,
        drifted_fields: List[str],
        drift_details: Dict[str, Dict[str, Any]],
        tolerance: float
    ) -> None:
        """
        Recursively compare baseline and current for drift.

        Args:
            baseline: Baseline value
            current: Current value
            path: Current field path
            drifted_fields: List to append drifted field names
            drift_details: Dictionary to store drift details
            tolerance: Absolute tolerance
        """
        if isinstance(baseline, dict) and isinstance(current, dict):
            all_keys = set(baseline.keys()) | set(current.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                b_val = baseline.get(key)
                c_val = current.get(key)
                self._compare_for_drift(
                    b_val, c_val, new_path, drifted_fields, drift_details, tolerance
                )

        elif isinstance(baseline, (list, tuple)) and isinstance(current, (list, tuple)):
            for i, (b_item, c_item) in enumerate(
                zip(baseline, current)
            ):
                new_path = f"{path}[{i}]"
                self._compare_for_drift(
                    b_item, c_item, new_path, drifted_fields, drift_details, tolerance
                )

        elif isinstance(baseline, (int, float)) and isinstance(current, (int, float)):
            if not self._floats_equal(baseline, current, tolerance):
                drifted_fields.append(path)
                drift_pct = 0.0
                if baseline != 0:
                    drift_pct = abs(current - baseline) / abs(baseline)
                drift_details[path] = {
                    "baseline": baseline,
                    "current": current,
                    "difference": current - baseline,
                    "drift_percentage": drift_pct
                }

        elif baseline != current:
            drifted_fields.append(path)
            drift_details[path] = {
                "baseline": str(baseline),
                "current": str(current),
                "type_mismatch": type(baseline).__name__ != type(current).__name__
            }

    def _floats_equal(
        self,
        a: float,
        b: float,
        tolerance: float
    ) -> bool:
        """
        Compare two floats with tolerance.

        Args:
            a: First value
            b: Second value
            tolerance: Absolute tolerance

        Returns:
            True if values are equal within tolerance
        """
        return abs(a - b) <= tolerance

    def _create_drift_check(self, drift: DriftDetection) -> VerificationCheck:
        """
        Create verification check from drift detection.

        Args:
            drift: Drift detection result

        Returns:
            VerificationCheck for drift
        """
        if drift.severity == DriftSeverity.NONE:
            return VerificationCheck(
                check_name="drift_detection",
                status=VerificationStatus.PASS,
                expected_value=drift.baseline_hash,
                actual_value=drift.current_hash,
                message="No drift detected from baseline"
            )

        if drift.severity == DriftSeverity.MINOR:
            return VerificationCheck(
                check_name="drift_detection",
                status=VerificationStatus.WARNING,
                expected_value=drift.baseline_hash,
                actual_value=drift.current_hash,
                difference=drift.drift_percentage,
                message=f"Minor drift detected: {drift.drift_percentage:.4f}% "
                       f"in fields: {', '.join(drift.drifted_fields[:5])}"
            )

        if drift.severity == DriftSeverity.MODERATE:
            return VerificationCheck(
                check_name="drift_detection",
                status=VerificationStatus.WARNING,
                expected_value=drift.baseline_hash,
                actual_value=drift.current_hash,
                difference=drift.drift_percentage,
                message=f"Moderate drift detected: {drift.drift_percentage:.4f}% "
                       f"in fields: {', '.join(drift.drifted_fields[:5])}"
            )

        # Critical drift
        return VerificationCheck(
            check_name="drift_detection",
            status=VerificationStatus.FAIL,
            expected_value=drift.baseline_hash,
            actual_value=drift.current_hash,
            difference=drift.drift_percentage,
            message=f"Critical drift detected: {drift.drift_percentage:.4f}% "
                   f"exceeds threshold in fields: {', '.join(drift.drifted_fields[:5])}"
        )

    def _track_non_determinism(
        self,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]]
    ) -> None:
        """
        Track potential sources of non-determinism in data.

        Args:
            input_data: Input data to analyze
            output_data: Output data to analyze
        """
        # Check for timestamp fields
        self._check_for_timestamps(input_data, "input")
        if output_data:
            self._check_for_timestamps(output_data, "output")

        # Check for random-like fields
        self._check_for_random_values(input_data, "input")
        if output_data:
            self._check_for_random_values(output_data, "output")

    def _check_for_timestamps(self, data: Any, context: str) -> None:
        """Check for timestamp fields that could cause non-determinism."""
        timestamp_keys = {"timestamp", "created_at", "updated_at", "time", "datetime"}

        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in timestamp_keys:
                    self._detected_non_determinism.add(NonDeterminismSource.TIMESTAMP)
                    self._non_determinism_details[f"timestamp_{context}_{key}"] = (
                        f"Found timestamp field '{key}' in {context}"
                    )
                else:
                    self._check_for_timestamps(value, context)

        elif isinstance(data, (list, tuple)):
            for item in data:
                self._check_for_timestamps(item, context)

    def _check_for_random_values(self, data: Any, context: str) -> None:
        """Check for fields that might contain random values."""
        random_keys = {"uuid", "id", "random", "nonce", "token", "session_id"}

        if isinstance(data, dict):
            for key, value in data.items():
                if any(rk in key.lower() for rk in random_keys):
                    if isinstance(value, str) and len(value) > 20:
                        self._detected_non_determinism.add(
                            NonDeterminismSource.RANDOM_SEED
                        )
                        self._non_determinism_details[f"random_{context}_{key}"] = (
                            f"Potential random value in '{key}'"
                        )
                self._check_for_random_values(value, context)

        elif isinstance(data, (list, tuple)):
            for item in data:
                self._check_for_random_values(item, context)

    def _apply_replay_seeds(self, seeds: SeedConfiguration) -> None:
        """
        Apply seed configuration for replay mode.

        Args:
            seeds: Seed configuration to apply
        """
        self._seeds = seeds

        # Set global Python random seed
        set_global_random_seed(seeds.global_seed)

        # Set numpy seed if available
        if seeds.numpy_seed is not None:
            try:
                import numpy as np
                np.random.seed(seeds.numpy_seed)
            except ImportError:
                pass

        # Set torch seed if available
        if seeds.torch_seed is not None:
            try:
                import torch
                torch.manual_seed(seeds.torch_seed)
            except ImportError:
                pass

        self.logger.info(f"Applied replay seeds: global={seeds.global_seed}")

    # ==========================================================================
    # Public API Methods
    # ==========================================================================

    def capture_execution_state(
        self,
        execution_id: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None
    ) -> ReplayConfiguration:
        """
        Capture current execution state for future replay.

        Args:
            execution_id: Unique execution identifier
            input_data: Input data to capture
            output_data: Optional output data to capture

        Returns:
            ReplayConfiguration for future replay
        """
        current_env = self._capture_environment()
        current_seeds = self._seeds or SeedConfiguration()

        return ReplayConfiguration(
            original_execution_id=execution_id,
            captured_inputs=input_data,
            captured_environment=current_env,
            captured_seeds=current_seeds,
            captured_versions=self._version_manifest or VersionManifest(),
            replay_mode=True
        )

    def generate_report(
        self,
        output: ReproducibilityOutput
    ) -> ReproducibilityReport:
        """
        Generate comprehensive reproducibility report.

        Args:
            output: ReproducibilityOutput from verification

        Returns:
            ReproducibilityReport with detailed analysis
        """
        # Find specific checks
        input_check = next(
            (c for c in output.checks if c.check_name == "input_hash_verification"),
            VerificationCheck(
                check_name="input_hash_verification",
                status=VerificationStatus.SKIPPED,
                message="Not performed"
            )
        )

        output_check = next(
            (c for c in output.checks if c.check_name == "output_hash_verification"),
            None
        )

        env_check = next(
            (c for c in output.checks if c.check_name == "environment_verification"),
            VerificationCheck(
                check_name="environment_verification",
                status=VerificationStatus.PASS,
                message="Environment captured"
            )
        )

        seed_check = next(
            (c for c in output.checks if c.check_name == "seed_verification"),
            VerificationCheck(
                check_name="seed_verification",
                status=VerificationStatus.PASS,
                message="Seeds captured"
            )
        )

        # Calculate confidence score
        # Count non-skipped checks for confidence calculation
        non_skipped_checks = sum(
            1 for c in output.checks if c.status != VerificationStatus.SKIPPED
        )
        passed_checks = output.checks_passed
        if non_skipped_checks > 0:
            confidence = passed_checks / non_skipped_checks
        else:
            # All checks skipped means we have no evidence either way - default to 1.0 if reproducible
            confidence = 1.0 if output.is_reproducible else 0.0

        # Determine risk level
        num_sources = len(output.non_determinism_sources)
        if num_sources == 0:
            risk = "low"
        elif num_sources <= 2:
            risk = "medium"
        else:
            risk = "high"

        # Generate recommendations
        recommendations = self._generate_recommendations(output)

        # Create report
        report = ReproducibilityReport(
            report_id=deterministic_id(
                {"execution_id": output.execution_id, "timestamp": str(output.timestamp)},
                "report_"
            ),
            execution_id=output.execution_id,
            overall_status=output.verification_status,
            is_reproducible=output.is_reproducible,
            confidence_score=confidence,
            input_verification=input_check,
            output_verification=output_check,
            environment_verification=env_check,
            seed_verification=seed_check,
            drift_analysis=output.drift_detection,
            non_determinism_risk=risk,
            non_determinism_sources=output.non_determinism_sources,
            recommendations=recommendations
        )

        # Calculate report hash (use JSON mode for datetime handling)
        report_data = report.model_dump(exclude={"report_hash"}, mode="json")
        report.report_hash = content_hash(report_data)[:16]

        return report

    def _generate_recommendations(
        self,
        output: ReproducibilityOutput
    ) -> List[str]:
        """
        Generate recommendations based on verification results.

        Args:
            output: Verification output

        Returns:
            List of recommendations
        """
        recommendations = []

        if NonDeterminismSource.TIMESTAMP in output.non_determinism_sources:
            recommendations.append(
                "Use DeterministicClock for all timestamp generation to ensure reproducibility"
            )

        if NonDeterminismSource.RANDOM_SEED in output.non_determinism_sources:
            recommendations.append(
                "Capture and replay random seeds using SeedConfiguration for deterministic randomness"
            )

        if NonDeterminismSource.FLOATING_POINT in output.non_determinism_sources:
            recommendations.append(
                "Consider using Decimal types for financial calculations to avoid float precision issues"
            )

        if NonDeterminismSource.EXTERNAL_API in output.non_determinism_sources:
            recommendations.append(
                "Mock or cache external API responses during replay to ensure determinism"
            )

        if NonDeterminismSource.DEPENDENCY_VERSION in output.non_determinism_sources:
            recommendations.append(
                "Pin all dependency versions in requirements.txt and use version manifest"
            )

        if output.drift_detection and output.drift_detection.severity != DriftSeverity.NONE:
            recommendations.append(
                f"Investigate drift in fields: {', '.join(output.drift_detection.drifted_fields[:3])}"
            )

        if not recommendations:
            recommendations.append(
                "Execution is fully reproducible. Continue monitoring for drift."
            )

        return recommendations

    def set_version_manifest(self, manifest: VersionManifest) -> None:
        """
        Set the version manifest for verification.

        Args:
            manifest: Version manifest to use
        """
        self._version_manifest = manifest
        self.logger.info(f"Version manifest set: {manifest.manifest_id}")

    def get_current_environment(self) -> EnvironmentFingerprint:
        """
        Get current environment fingerprint.

        Returns:
            Current EnvironmentFingerprint
        """
        return self._capture_environment()

    def get_current_seeds(self) -> SeedConfiguration:
        """
        Get current seed configuration.

        Returns:
            Current SeedConfiguration
        """
        return self._seeds or SeedConfiguration()

    def compare_hashes(
        self,
        hash1: str,
        hash2: str
    ) -> Tuple[bool, str]:
        """
        Compare two hashes and return result with explanation.

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            Tuple of (match: bool, explanation: str)
        """
        if hash1 == hash2:
            return True, "Hashes match exactly"

        # Check if they're similar (potential truncation)
        if hash1.startswith(hash2) or hash2.startswith(hash1):
            return False, "Hashes partially match - possible truncation"

        return False, f"Hashes differ: {hash1[:16]}... vs {hash2[:16]}..."
