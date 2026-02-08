# -*- coding: utf-8 -*-
"""
Reproducibility Service REST API Router - AGENT-FOUND-008: Reproducibility Agent

FastAPI router providing 20 endpoints for reproducibility verification,
artifact hashing, drift detection, replay execution, environment capture,
version pinning, report generation, and statistics.

All endpoints are mounted under ``/api/v1/reproducibility``.

Endpoints:
    1.  POST   /v1/verify               - Run full reproducibility verification
    2.  POST   /v1/verify/input          - Verify input hash only
    3.  POST   /v1/verify/output         - Verify output hash only
    4.  GET    /v1/verifications          - List verification runs
    5.  GET    /v1/verifications/{id}     - Get verification details
    6.  POST   /v1/hash                  - Compute deterministic hash
    7.  GET    /v1/hashes/{artifact_id}   - Get artifact hash history
    8.  POST   /v1/drift/detect          - Run drift detection
    9.  GET    /v1/drift/baselines        - List baselines
    10. POST   /v1/drift/baselines        - Create baseline
    11. GET    /v1/drift/baselines/{id}   - Get baseline
    12. POST   /v1/replay                - Execute replay
    13. GET    /v1/replays/{replay_id}    - Get replay session
    14. GET    /v1/environment            - Capture environment
    15. GET    /v1/environment/{id}       - Get stored fingerprint
    16. POST   /v1/versions/pin          - Pin versions
    17. GET    /v1/versions/manifest/{id} - Get manifest
    18. POST   /v1/report                - Generate report
    19. GET    /v1/statistics             - Get stats
    20. GET    /health                   - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; reproducibility router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class VerifyFullRequest(BaseModel):
        """Request body for full reproducibility verification."""
        execution_id: str = Field(..., description="Unique execution identifier")
        input_data: Dict[str, Any] = Field(..., description="Input data to verify")
        output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
        expected_input_hash: Optional[str] = Field(None, description="Expected input hash")
        expected_output_hash: Optional[str] = Field(None, description="Expected output hash")
        absolute_tolerance: float = Field(default=1e-9, description="Absolute tolerance")
        relative_tolerance: float = Field(default=1e-6, description="Relative tolerance")

    class VerifyInputRequest(BaseModel):
        """Request body for input-only verification."""
        input_data: Dict[str, Any] = Field(..., description="Input data to verify")
        expected_hash: Optional[str] = Field(None, description="Expected input hash")

    class VerifyOutputRequest(BaseModel):
        """Request body for output-only verification."""
        output_data: Dict[str, Any] = Field(..., description="Output data to verify")
        expected_hash: Optional[str] = Field(None, description="Expected output hash")
        absolute_tolerance: float = Field(default=1e-9, description="Absolute tolerance")
        relative_tolerance: float = Field(default=1e-6, description="Relative tolerance")

    class HashComputeRequest(BaseModel):
        """Request body for hash computation."""
        data: Any = Field(..., description="Data to hash")
        algorithm: str = Field(default="sha256", description="Hash algorithm")

    class DriftDetectRequest(BaseModel):
        """Request body for drift detection."""
        baseline_id: Optional[str] = Field(None, description="Baseline ID")
        baseline_data: Optional[Dict[str, Any]] = Field(None, description="Inline baseline")
        current_data: Dict[str, Any] = Field(..., description="Current data")
        soft_threshold: float = Field(default=0.01, description="Soft threshold")
        hard_threshold: float = Field(default=0.05, description="Hard threshold")
        tolerance: float = Field(default=1e-9, description="Absolute tolerance")

    class CreateBaselineRequest(BaseModel):
        """Request body for creating a drift baseline."""
        name: str = Field(..., description="Baseline name")
        description: str = Field(default="", description="Description")
        baseline_data: Dict[str, Any] = Field(..., description="Baseline data")

    class ReplayExecuteRequest(BaseModel):
        """Request body for replay execution."""
        original_execution_id: str = Field(..., description="Original execution ID")
        captured_inputs: Dict[str, Any] = Field(..., description="Captured inputs")
        captured_environment: Dict[str, Any] = Field(..., description="Captured environment")
        captured_seeds: Dict[str, Any] = Field(..., description="Captured seeds")
        captured_versions: Dict[str, Any] = Field(..., description="Captured versions")
        strict_mode: bool = Field(default=False, description="Strict mode")

    class PinVersionsRequest(BaseModel):
        """Request body for version pinning."""
        auto_detect: bool = Field(default=True, description="Auto-detect current versions")

    class GenerateReportRequest(BaseModel):
        """Request body for report generation."""
        execution_id: str = Field(..., description="Execution ID")
        verification_id: str = Field(..., description="Verification run ID")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/reproducibility",
        tags=["reproducibility"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract ReproducibilityService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        ReproducibilityService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "reproducibility_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Reproducibility service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # 1. Full verification
    @router.post("/v1/verify")
    async def verify_full(
        body: VerifyFullRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Run full reproducibility verification."""
        service = _get_service(request)
        try:
            run = service.verify(
                execution_id=body.execution_id,
                input_data=body.input_data,
                output_data=body.output_data,
                expected_input_hash=body.expected_input_hash,
                expected_output_hash=body.expected_output_hash,
                abs_tol=body.absolute_tolerance,
                rel_tol=body.relative_tolerance,
            )
            return run.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 2. Input-only verification
    @router.post("/v1/verify/input")
    async def verify_input(
        body: VerifyInputRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Verify input hash only."""
        service = _get_service(request)
        check = service.verifier.verify_input(
            body.input_data, body.expected_hash,
        )
        return check.model_dump(mode="json")

    # 3. Output-only verification
    @router.post("/v1/verify/output")
    async def verify_output(
        body: VerifyOutputRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Verify output hash only."""
        service = _get_service(request)
        check = service.verifier.verify_output(
            body.output_data, body.expected_hash,
            body.absolute_tolerance, body.relative_tolerance,
        )
        return check.model_dump(mode="json")

    # 4. List verifications
    @router.get("/v1/verifications")
    async def list_verifications(
        execution_id: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List verification runs."""
        service = _get_service(request)
        runs = service.verifier.list_verifications(
            execution_id=execution_id, limit=limit,
        )
        return {
            "verifications": [r.model_dump(mode="json") for r in runs],
            "count": len(runs),
        }

    # 5. Get verification details
    @router.get("/v1/verifications/{verification_id}")
    async def get_verification(
        verification_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get verification details."""
        service = _get_service(request)
        run = service.verifier.get_verification(verification_id)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=f"Verification {verification_id} not found",
            )
        return run.model_dump(mode="json")

    # 6. Compute hash
    @router.post("/v1/hash")
    async def compute_hash(
        body: HashComputeRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Compute deterministic hash."""
        service = _get_service(request)
        data_hash = service.hasher.compute_hash(body.data, algorithm=body.algorithm)
        return {
            "data_hash": data_hash,
            "algorithm": body.algorithm,
            "normalization_applied": True,
        }

    # 7. Get artifact hash history
    @router.get("/v1/hashes/{artifact_id}")
    async def get_hash_history(
        artifact_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get artifact hash history."""
        service = _get_service(request)
        history = service.hasher.get_hash_history(artifact_id)
        return {
            "artifact_id": artifact_id,
            "hashes": [h.model_dump(mode="json") for h in history],
            "count": len(history),
        }

    # 8. Drift detection
    @router.post("/v1/drift/detect")
    async def detect_drift(
        body: DriftDetectRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Run drift detection."""
        service = _get_service(request)
        try:
            if body.baseline_id:
                result = service.detect_drift(
                    baseline_id=body.baseline_id,
                    current_data=body.current_data,
                    soft_threshold=body.soft_threshold,
                    hard_threshold=body.hard_threshold,
                )
            elif body.baseline_data:
                result = service.detect_drift_inline(
                    baseline_data=body.baseline_data,
                    current_data=body.current_data,
                    soft_threshold=body.soft_threshold,
                    hard_threshold=body.hard_threshold,
                    tolerance=body.tolerance,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either baseline_id or baseline_data must be provided",
                )
            return {
                "drift_detection": result.model_dump(mode="json"),
                "baseline_id": body.baseline_id or "",
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # 9. List baselines
    @router.get("/v1/drift/baselines")
    async def list_baselines(
        active_only: bool = Query(True),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List drift baselines."""
        service = _get_service(request)
        baselines = service.drift_detector.list_baselines(active_only=active_only)
        return {
            "baselines": [b.model_dump(mode="json") for b in baselines],
            "count": len(baselines),
        }

    # 10. Create baseline
    @router.post("/v1/drift/baselines")
    async def create_baseline(
        body: CreateBaselineRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a drift baseline."""
        service = _get_service(request)
        try:
            baseline = service.drift_detector.create_baseline(
                name=body.name,
                description=body.description,
                baseline_data=body.baseline_data,
            )
            return baseline.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 11. Get baseline
    @router.get("/v1/drift/baselines/{baseline_id}")
    async def get_baseline(
        baseline_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a drift baseline."""
        service = _get_service(request)
        baseline = service.drift_detector.get_baseline(baseline_id)
        if baseline is None:
            raise HTTPException(
                status_code=404,
                detail=f"Baseline {baseline_id} not found",
            )
        return baseline.model_dump(mode="json")

    # 12. Execute replay
    @router.post("/v1/replay")
    async def execute_replay(
        body: ReplayExecuteRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Execute a replay session."""
        service = _get_service(request)
        try:
            from greenlang.reproducibility.models import (
                EnvironmentFingerprint,
                SeedConfiguration,
                VersionManifest,
            )

            env = EnvironmentFingerprint(**body.captured_environment)
            seeds = SeedConfiguration(**body.captured_seeds)
            versions = VersionManifest(**body.captured_versions)

            session = service.replay(
                original_execution_id=body.original_execution_id,
                captured_inputs=body.captured_inputs,
                captured_env=env,
                captured_seeds=seeds,
                captured_versions=versions,
            )
            return {"replay_session": session.model_dump(mode="json")}
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 13. Get replay session
    @router.get("/v1/replays/{replay_id}")
    async def get_replay(
        replay_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a replay session."""
        service = _get_service(request)
        session = service.replay_engine.get_replay_session(replay_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Replay session {replay_id} not found",
            )
        return session.model_dump(mode="json")

    # 14. Capture environment
    @router.get("/v1/environment")
    async def capture_environment(
        request: Request,
    ) -> Dict[str, Any]:
        """Capture current execution environment."""
        service = _get_service(request)
        fp = service.capture_environment()
        return fp.model_dump(mode="json")

    # 15. Get stored fingerprint
    @router.get("/v1/environment/{fingerprint_id}")
    async def get_fingerprint(
        fingerprint_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a stored environment fingerprint."""
        service = _get_service(request)
        fp = service.env_capture.get_fingerprint(fingerprint_id)
        if fp is None:
            raise HTTPException(
                status_code=404,
                detail=f"Fingerprint {fingerprint_id} not found",
            )
        return fp.model_dump(mode="json")

    # 16. Pin versions
    @router.post("/v1/versions/pin")
    async def pin_versions(
        body: PinVersionsRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Pin current versions as a manifest."""
        service = _get_service(request)
        manifest = service.pin_versions()
        return manifest.model_dump(mode="json")

    # 17. Get manifest
    @router.get("/v1/versions/manifest/{manifest_id}")
    async def get_manifest(
        manifest_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a version manifest."""
        service = _get_service(request)
        manifest = service.version_pinner.get_manifest(manifest_id)
        if manifest is None:
            raise HTTPException(
                status_code=404,
                detail=f"Manifest {manifest_id} not found",
            )
        return manifest.model_dump(mode="json")

    # 18. Generate report
    @router.post("/v1/report")
    async def generate_report(
        body: GenerateReportRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Generate a reproducibility report."""
        service = _get_service(request)
        run = service.verifier.get_verification(body.verification_id)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail=f"Verification {body.verification_id} not found",
            )
        report = service.generate_report(body.execution_id, run)
        return report.model_dump(mode="json")

    # 19. Get statistics
    @router.get("/v1/statistics")
    async def get_statistics(
        request: Request,
    ) -> Dict[str, Any]:
        """Get reproducibility service statistics."""
        service = _get_service(request)
        stats = service.get_statistics()
        return stats.model_dump(mode="json")

    # 20. Health check
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """Reproducibility service health check endpoint."""
        return {"status": "healthy", "service": "reproducibility"}


__all__ = [
    "router",
]
