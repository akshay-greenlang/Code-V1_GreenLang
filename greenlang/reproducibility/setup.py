# -*- coding: utf-8 -*-
"""
Reproducibility Service Setup - AGENT-FOUND-008: Reproducibility Agent

Provides ``configure_reproducibility(app)`` which wires up the
Reproducibility SDK (hasher, verifier, drift detector, replay engine,
environment capture, seed manager, version pinner, provenance) and
mounts the REST API.

Also exposes ``get_reproducibility(app)`` for programmatic access
and the ``ReproducibilityService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.reproducibility.setup import configure_reproducibility
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_reproducibility(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.reproducibility.config import ReproducibilityConfig, get_config
from greenlang.reproducibility.artifact_hasher import ArtifactHasher
from greenlang.reproducibility.determinism_verifier import DeterminismVerifier
from greenlang.reproducibility.drift_detector import DriftDetector
from greenlang.reproducibility.environment_capture import EnvironmentCapture
from greenlang.reproducibility.seed_manager import SeedManager
from greenlang.reproducibility.version_pinner import VersionPinner
from greenlang.reproducibility.provenance import ProvenanceTracker
from greenlang.reproducibility.replay_engine import ReplayEngine
from greenlang.reproducibility.models import (
    VerificationStatus,
    VerificationCheck,
    VerificationRun,
    VerificationStatistics,
    DriftDetection,
    DriftBaseline,
    EnvironmentFingerprint,
    SeedConfiguration,
    VersionManifest,
    ReplaySession,
    ReplayConfiguration,
    ReproducibilityInput,
    ReproducibilityOutput,
    ReproducibilityReport,
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
)
from greenlang.reproducibility.metrics import (
    PROMETHEUS_AVAILABLE,
    record_verification,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# ReproducibilityService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["ReproducibilityService"] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class ReproducibilityService:
    """Unified facade over the Reproducibility SDK.

    Aggregates all reproducibility engines (hasher, verifier, drift
    detector, replay engine, environment capture, seed manager, version
    pinner, provenance) through a single entry point with convenience
    methods for common operations.

    Attributes:
        config: ReproducibilityConfig instance.
        hasher: ArtifactHasher instance.
        verifier: DeterminismVerifier instance.
        drift_detector: DriftDetector instance.
        replay_engine: ReplayEngine instance.
        env_capture: EnvironmentCapture instance.
        seed_manager: SeedManager instance.
        version_pinner: VersionPinner instance.
        provenance: ProvenanceTracker instance.

    Example:
        >>> service = ReproducibilityService()
        >>> run = service.verify("exec_001", {"value": 42.0})
        >>> print(run.is_reproducible)
    """

    def __init__(
        self,
        config: Optional[ReproducibilityConfig] = None,
    ) -> None:
        """Initialize the Reproducibility Service facade.

        Args:
            config: Optional reproducibility config. Uses global config if None.
        """
        self.config = config or get_config()

        # Initialize all engines
        self.provenance = ProvenanceTracker()
        self.hasher = ArtifactHasher(self.config)
        self.verifier = DeterminismVerifier(self.config, self.hasher)
        self.drift_detector = DriftDetector(self.config, self.hasher)
        self.env_capture = EnvironmentCapture(self.config)
        self.seed_manager = SeedManager(self.config)
        self.version_pinner = VersionPinner(self.config)
        self.replay_engine = ReplayEngine(
            self.config,
            self.verifier,
            self.env_capture,
            self.seed_manager,
            self.version_pinner,
        )

        # Statistics tracking
        self._stats = VerificationStatistics()
        self._started = False

        logger.info("ReproducibilityService facade created")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def verify(
        self,
        execution_id: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        expected_input_hash: Optional[str] = None,
        expected_output_hash: Optional[str] = None,
        abs_tol: float = DEFAULT_ABSOLUTE_TOLERANCE,
        rel_tol: float = DEFAULT_RELATIVE_TOLERANCE,
    ) -> VerificationRun:
        """Run a full reproducibility verification.

        Args:
            execution_id: Unique execution identifier.
            input_data: Input data to verify.
            output_data: Optional output data to verify.
            expected_input_hash: Expected hash of input data.
            expected_output_hash: Expected hash of output data.
            abs_tol: Absolute tolerance for float comparison.
            rel_tol: Relative tolerance for float comparison.

        Returns:
            VerificationRun with all check results.
        """
        start_time = time.time()

        repro_input = ReproducibilityInput(
            execution_id=execution_id,
            input_data=input_data,
            output_data=output_data,
            expected_input_hash=expected_input_hash,
            expected_output_hash=expected_output_hash,
            absolute_tolerance=abs_tol,
            relative_tolerance=rel_tol,
        )

        run = self.verifier.run_verification(execution_id, repro_input)

        # Record provenance
        self.provenance.record(
            entity_type="verification",
            entity_id=run.verification_id,
            action="verify",
            data_hash=run.input_hash,
            user_id="system",
        )

        # Update statistics
        self._update_stats(run)

        return run

    def compute_hash(self, data: Any) -> str:
        """Compute a deterministic hash of arbitrary data.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest.
        """
        return self.hasher.compute_hash(data)

    def detect_drift(
        self,
        baseline_id: str,
        current_data: Dict[str, Any],
        soft_threshold: Optional[float] = None,
        hard_threshold: Optional[float] = None,
    ) -> DriftDetection:
        """Detect drift by comparing current data against a stored baseline.

        Args:
            baseline_id: ID of the stored baseline.
            current_data: Current data to check.
            soft_threshold: Override soft threshold.
            hard_threshold: Override hard threshold.

        Returns:
            DriftDetection result.
        """
        result = self.drift_detector.compare_to_baseline(
            baseline_id=baseline_id,
            current_data=current_data,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
        )

        # Record provenance
        self.provenance.record(
            entity_type="drift",
            entity_id=baseline_id,
            action="detect",
            data_hash=result.current_hash,
        )

        # Update stats
        self._stats.drift_detections += 1

        return result

    def detect_drift_inline(
        self,
        baseline_data: Dict[str, Any],
        current_data: Dict[str, Any],
        soft_threshold: Optional[float] = None,
        hard_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> DriftDetection:
        """Detect drift using inline baseline data (no stored baseline).

        Args:
            baseline_data: Baseline data for comparison.
            current_data: Current data to check.
            soft_threshold: Override soft threshold.
            hard_threshold: Override hard threshold.
            tolerance: Override tolerance.

        Returns:
            DriftDetection result.
        """
        result = self.drift_detector.detect_drift(
            baseline=baseline_data,
            current=current_data,
            soft_threshold=soft_threshold or self.config.drift_soft_threshold,
            hard_threshold=hard_threshold or self.config.drift_hard_threshold,
            tolerance=tolerance or self.config.default_absolute_tolerance,
        )

        self._stats.drift_detections += 1
        return result

    def replay(
        self,
        original_execution_id: str,
        captured_inputs: Dict[str, Any],
        captured_env: EnvironmentFingerprint,
        captured_seeds: SeedConfiguration,
        captured_versions: VersionManifest,
        original_output: Optional[Dict[str, Any]] = None,
    ) -> ReplaySession:
        """Execute a replay of a previous execution.

        Args:
            original_execution_id: ID of the original execution.
            captured_inputs: Captured input data.
            captured_env: Captured environment fingerprint.
            captured_seeds: Captured seed configuration.
            captured_versions: Captured version manifest.
            original_output: Optional original output for comparison.

        Returns:
            ReplaySession with verification results.
        """
        replay_config = self.replay_engine.prepare_replay(
            original_execution_id=original_execution_id,
            captured_inputs=captured_inputs,
            captured_env=captured_env,
            captured_seeds=captured_seeds,
            captured_versions=captured_versions,
        )

        session = self.replay_engine.execute_replay(
            replay_config=replay_config,
            original_output=original_output,
        )

        # Record provenance
        self.provenance.record(
            entity_type="replay",
            entity_id=session.replay_id,
            action="execute",
            data_hash=self.hasher.compute_hash(captured_inputs),
        )

        # Update stats
        self._stats.replay_count += 1

        return session

    def capture_environment(self) -> EnvironmentFingerprint:
        """Capture the current execution environment.

        Returns:
            EnvironmentFingerprint with current state.
        """
        fp = self.env_capture.capture()
        self.env_capture.store_fingerprint(fp)
        return fp

    def create_seed_config(
        self,
        global_seed: int = 42,
        numpy_seed: Optional[int] = 42,
        torch_seed: Optional[int] = 42,
        custom_seeds: Optional[Dict[str, int]] = None,
    ) -> SeedConfiguration:
        """Create a seed configuration.

        Args:
            global_seed: Global Python random seed.
            numpy_seed: NumPy random seed.
            torch_seed: PyTorch manual seed.
            custom_seeds: Custom component seeds.

        Returns:
            SeedConfiguration instance.
        """
        return self.seed_manager.create_seed_config(
            global_seed=global_seed,
            numpy_seed=numpy_seed,
            torch_seed=torch_seed,
            custom_seeds=custom_seeds,
        )

    def pin_versions(
        self,
        agent_versions: Optional[Dict[str, Any]] = None,
        model_versions: Optional[Dict[str, Any]] = None,
        factor_versions: Optional[Dict[str, Any]] = None,
        data_versions: Optional[Dict[str, Any]] = None,
    ) -> VersionManifest:
        """Create a version manifest, or pin current versions if no args.

        If no version dictionaries are provided, captures the current
        system state automatically.

        Args:
            agent_versions: Agent version pins.
            model_versions: Model version pins.
            factor_versions: Factor version pins.
            data_versions: Data version pins.

        Returns:
            VersionManifest instance.
        """
        if all(v is None for v in [agent_versions, model_versions, factor_versions, data_versions]):
            return self.version_pinner.pin_current_versions()

        return self.version_pinner.create_manifest(
            agent_versions=agent_versions,
            model_versions=model_versions,
            factor_versions=factor_versions,
            data_versions=data_versions,
        )

    def generate_report(
        self,
        execution_id: str,
        verification_run: VerificationRun,
    ) -> ReproducibilityReport:
        """Generate a comprehensive reproducibility report.

        Args:
            execution_id: Execution ID.
            verification_run: VerificationRun to report on.

        Returns:
            ReproducibilityReport with detailed analysis.
        """
        # Find specific checks
        input_check = next(
            (c for c in verification_run.checks if c.check_name == "input_hash_verification"),
            VerificationCheck(
                check_name="input_hash_verification",
                status=VerificationStatus.SKIPPED,
                message="Not performed",
            ),
        )

        output_check = next(
            (c for c in verification_run.checks if c.check_name == "output_hash_verification"),
            None,
        )

        env_check = next(
            (c for c in verification_run.checks if c.check_name == "environment_verification"),
            VerificationCheck(
                check_name="environment_verification",
                status=VerificationStatus.PASS,
                message="Environment captured",
            ),
        )

        seed_check = next(
            (c for c in verification_run.checks if c.check_name == "seed_verification"),
            VerificationCheck(
                check_name="seed_verification",
                status=VerificationStatus.PASS,
                message="Seeds captured",
            ),
        )

        # Calculate confidence score
        non_skipped = [
            c for c in verification_run.checks
            if c.status != VerificationStatus.SKIPPED
        ]
        passed = [c for c in non_skipped if c.status == VerificationStatus.PASS]

        if non_skipped:
            confidence = len(passed) / len(non_skipped)
        else:
            confidence = 1.0 if verification_run.is_reproducible else 0.0

        # Generate recommendations
        recommendations: List[str] = []
        for check in verification_run.checks:
            if check.status == VerificationStatus.FAIL:
                if "input" in check.check_name:
                    recommendations.append(
                        "Input data has changed. Verify data pipeline integrity."
                    )
                elif "output" in check.check_name:
                    recommendations.append(
                        "Output differs. Check for non-deterministic computations."
                    )
                elif "environment" in check.check_name:
                    recommendations.append(
                        "Environment mismatch. Pin dependency versions and use containers."
                    )
                elif "seed" in check.check_name:
                    recommendations.append(
                        "Seed mismatch. Ensure SeedConfiguration is restored before replay."
                    )
                elif "version" in check.check_name:
                    recommendations.append(
                        "Version mismatch. Update version manifest or pin dependencies."
                    )

        if not recommendations:
            recommendations.append(
                "Execution is fully reproducible. Continue monitoring for drift."
            )

        # Determine non-determinism risk
        fail_count = sum(1 for c in verification_run.checks if c.status == VerificationStatus.FAIL)
        if fail_count == 0:
            risk = "low"
        elif fail_count <= 2:
            risk = "medium"
        else:
            risk = "high"

        # Build report hash
        report_data_for_hash = {
            "execution_id": execution_id,
            "status": verification_run.status.value,
            "is_reproducible": verification_run.is_reproducible,
            "confidence": confidence,
        }
        report_hash = hashlib.sha256(
            json.dumps(report_data_for_hash, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        report = ReproducibilityReport(
            report_id=f"report_{report_hash}",
            execution_id=execution_id,
            overall_status=verification_run.status,
            is_reproducible=verification_run.is_reproducible,
            confidence_score=round(confidence, 4),
            input_verification=input_check,
            output_verification=output_check,
            environment_verification=env_check,
            seed_verification=seed_check,
            non_determinism_risk=risk,
            non_determinism_sources=[],
            recommendations=recommendations,
            report_hash=report_hash,
        )

        # Record provenance
        self.provenance.record(
            entity_type="report",
            entity_id=report.report_id,
            action="generate",
            data_hash=report_hash,
        )

        return report

    def get_statistics(self) -> VerificationStatistics:
        """Get aggregated verification statistics.

        Returns:
            VerificationStatistics summary.
        """
        return self._stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_stats(self, run: VerificationRun) -> None:
        """Update verification statistics from a run.

        Args:
            run: Completed VerificationRun.
        """
        self._stats.total_verifications += 1

        if run.status == VerificationStatus.PASS:
            self._stats.pass_count += 1
        elif run.status == VerificationStatus.FAIL:
            self._stats.fail_count += 1
        elif run.status == VerificationStatus.WARNING:
            self._stats.warning_count += 1

        # Update average processing time
        total = self._stats.total_verifications
        prev_avg = self._stats.avg_processing_time_ms
        self._stats.avg_processing_time_ms = (
            (prev_avg * (total - 1) + run.processing_time_ms) / total
        )

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_hasher(self) -> ArtifactHasher:
        """Get the ArtifactHasher instance.

        Returns:
            ArtifactHasher used by this service.
        """
        return self.hasher

    def get_verifier(self) -> DeterminismVerifier:
        """Get the DeterminismVerifier instance.

        Returns:
            DeterminismVerifier used by this service.
        """
        return self.verifier

    def get_drift_detector(self) -> DriftDetector:
        """Get the DriftDetector instance.

        Returns:
            DriftDetector used by this service.
        """
        return self.drift_detector

    def get_replay_engine(self) -> ReplayEngine:
        """Get the ReplayEngine instance.

        Returns:
            ReplayEngine used by this service.
        """
        return self.replay_engine

    def get_provenance(self) -> ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            ProvenanceTracker used by this service.
        """
        return self.provenance

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get reproducibility service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_verifications": self._stats.total_verifications,
            "pass_count": self._stats.pass_count,
            "fail_count": self._stats.fail_count,
            "warning_count": self._stats.warning_count,
            "drift_detections": self._stats.drift_detections,
            "replay_count": self._stats.replay_count,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the reproducibility service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("ReproducibilityService already started; skipping")
            return

        logger.info("ReproducibilityService starting up...")
        self._started = True
        logger.info("ReproducibilityService startup complete")

    def shutdown(self) -> None:
        """Shutdown the reproducibility service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("ReproducibilityService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> ReproducibilityService:
    """Get or create the singleton ReproducibilityService instance.

    Returns:
        The singleton ReproducibilityService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ReproducibilityService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_reproducibility(
    app: Any,
    config: Optional[ReproducibilityConfig] = None,
) -> ReproducibilityService:
    """Configure the Reproducibility Service on a FastAPI application.

    Creates the ReproducibilityService, stores it in app.state, mounts
    the reproducibility API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional reproducibility config.

    Returns:
        ReproducibilityService instance.
    """
    global _singleton_instance

    service = ReproducibilityService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.reproducibility_service = service

    # Mount reproducibility API router
    try:
        from greenlang.reproducibility.api.router import router as repro_router
        if repro_router is not None:
            app.include_router(repro_router)
            logger.info("Reproducibility service API router mounted")
    except ImportError:
        logger.warning("Reproducibility router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("Reproducibility service configured on app")
    return service


def get_reproducibility(app: Any) -> ReproducibilityService:
    """Get the ReproducibilityService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        ReproducibilityService instance.

    Raises:
        RuntimeError: If reproducibility service not configured.
    """
    service = getattr(app.state, "reproducibility_service", None)
    if service is None:
        raise RuntimeError(
            "Reproducibility service not configured. "
            "Call configure_reproducibility(app) first."
        )
    return service


def get_router() -> Any:
    """Get the reproducibility API router.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.reproducibility.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "ReproducibilityService",
    "configure_reproducibility",
    "get_reproducibility",
    "get_router",
]
