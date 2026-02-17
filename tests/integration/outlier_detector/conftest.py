# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-013 Outlier Detection Agent integration tests.

Provides reusable test fixtures for:
- Package stub for greenlang.outlier_detector (same pattern as MVI tests)
- Environment cleanup (autouse)
- Database mock fixtures (PostgreSQL + Redis)
- Sample data fixtures (numeric data with outliers, records, time-series)
- OutlierDetectorService factory fixture
- FastAPI test client fixture (httpx/TestClient)

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures that do not apply to OD tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents fixture.

    The parent tests/integration/conftest.py defines an autouse fixture
    that patches greenlang.agents.registry.get_agent, which does not
    exist. We override it here to make it a no-op for OD integration tests.
    """
    yield {}


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent conftest block_network fixture.

    The parent tests/integration/conftest.py blocks all socket access,
    which prevents asyncio event loop creation needed by some mock DB
    tests. We disable it for OD integration tests since our tests are
    fully self-contained with mocks.
    """
    yield


# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_OD_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_od_env(monkeypatch):
    """Remove all GL_OD_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_OD_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_OD_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Also reset the singleton config so each test starts fresh
    from greenlang.outlier_detector.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Helper: build service with patched engines
# ---------------------------------------------------------------------------


def _build_service(config=None):
    """Create an OutlierDetectorService with engines patched out.

    This is the canonical way to create a service instance in tests
    without requiring the full SDK engine imports.

    Args:
        config: Optional OutlierDetectorConfig. Uses default if None.

    Returns:
        OutlierDetectorService instance with engines patched.
    """
    from greenlang.outlier_detector.config import (
        OutlierDetectorConfig,
        set_config,
    )
    from greenlang.outlier_detector.setup import OutlierDetectorService

    cfg = config or OutlierDetectorConfig()
    set_config(cfg)

    with patch.object(OutlierDetectorService, "_init_engines"):
        svc = OutlierDetectorService(config=cfg)
    return svc


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create an OutlierDetectorConfig with test defaults."""
    from greenlang.outlier_detector.config import OutlierDetectorConfig

    return OutlierDetectorConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        s3_bucket_url="s3://test-od-bucket",
        max_records=100_000,
        batch_size=1_000,
        iqr_multiplier=1.5,
        zscore_threshold=3.0,
        mad_threshold=3.5,
        grubbs_alpha=0.05,
        lof_neighbors=20,
        isolation_trees=100,
        ensemble_method="weighted_average",
        min_consensus=2,
        enable_contextual=True,
        enable_temporal=True,
        enable_multivariate=True,
        default_treatment="flag",
        winsorize_pct=0.05,
        worker_count=4,
        pool_min_size=2,
        pool_max_size=10,
        cache_ttl=3600,
        rate_limit_rpm=120,
        rate_limit_burst=20,
        enable_provenance=True,
        provenance_hash_algorithm="sha256",
        log_level="INFO",
        enable_metrics=True,
    )


# ---------------------------------------------------------------------------
# Service factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def service(config):
    """Create an OutlierDetectorService instance for integration testing.

    Engines are patched out so the service can run without the full SDK.
    """
    return _build_service(config)


@pytest.fixture
def started_service(service):
    """Create a service that has been marked as started (healthy)."""
    service._started = True
    return service


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_numeric_data() -> List[float]:
    """Return 100 numeric values with 5 outliers.

    Normal range: 10.0 - 50.0
    Outliers at indices 10, 25, 50, 75, 95 with values far outside range.
    """
    import random
    random.seed(42)
    values = [random.uniform(10.0, 50.0) for _ in range(100)]
    # Inject 5 outliers
    values[10] = 500.0
    values[25] = -200.0
    values[50] = 1000.0
    values[75] = -500.0
    values[95] = 800.0
    return values


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Return 10 records with numeric columns and some outlier values.

    Record layout:
    - rec-001 to rec-010 with columns: id, company, revenue, emissions, region
    - Outliers: rec-003 (revenue=999999), rec-007 (emissions=50000)
    """
    return [
        {
            "id": "rec-001",
            "company": "EcoTech Corp",
            "revenue": 15200.50,
            "emissions": 1200.0,
            "region": "US",
        },
        {
            "id": "rec-002",
            "company": "GreenWorks Inc",
            "revenue": 18500.00,
            "emissions": 1400.0,
            "region": "EU",
        },
        {
            "id": "rec-003",
            "company": "SolarPeak Ltd",
            "revenue": 999999.99,
            "emissions": 1100.0,
            "region": "US",
        },
        {
            "id": "rec-004",
            "company": "BioFuel SA",
            "revenue": 8750.00,
            "emissions": 890.0,
            "region": "EU",
        },
        {
            "id": "rec-005",
            "company": "WindPower AG",
            "revenue": 22300.75,
            "emissions": 1600.0,
            "region": "DE",
        },
        {
            "id": "rec-006",
            "company": "CleanAir Co",
            "revenue": 31400.25,
            "emissions": 560.0,
            "region": "US",
        },
        {
            "id": "rec-007",
            "company": "TerraGreen Pty",
            "revenue": 19800.60,
            "emissions": 50000.0,
            "region": "AU",
        },
        {
            "id": "rec-008",
            "company": "HydroGen Inc",
            "revenue": 42000.00,
            "emissions": 1800.0,
            "region": "US",
        },
        {
            "id": "rec-009",
            "company": "CircularTech",
            "revenue": 12500.00,
            "emissions": 750.0,
            "region": "EU",
        },
        {
            "id": "rec-010",
            "company": "OceanBlue Corp",
            "revenue": 28900.80,
            "emissions": 1350.0,
            "region": "US",
        },
    ]


@pytest.fixture
def sample_time_series() -> List[Dict[str, Any]]:
    """Return 30 timestamped records with 3 temporal anomalies.

    Base values follow a linear trend with small noise.
    Anomalies injected at indices 8, 17, 26 with extreme spikes.
    """
    import random
    random.seed(99)
    records = []
    for i in range(30):
        base_value = 100.0 + i * 2.0 + random.uniform(-3.0, 3.0)
        # Inject anomalies
        if i == 8:
            base_value = 500.0
        elif i == 17:
            base_value = -100.0
        elif i == 26:
            base_value = 800.0

        records.append({
            "id": f"ts-{i:03d}",
            "timestamp": f"2025-01-{(i % 28) + 1:02d}T{i % 24:02d}:00:00Z",
            "temperature": round(base_value, 2),
            "pressure": round(1013.0 + i * 0.5 + random.uniform(-1.0, 1.0), 2),
            "humidity": round(60.0 + i * 0.3 + random.uniform(-2.0, 2.0), 2),
        })
    return records


# ---------------------------------------------------------------------------
# V043 migration table definitions (for database integration mock tests)
# ---------------------------------------------------------------------------


V043_TABLES = [
    "od_jobs",
    "od_detections",
    "od_batch_detections",
    "od_classifications",
    "od_treatments",
    "od_thresholds",
    "od_feedback",
    "od_pipeline_results",
    "od_provenance_entries",
    "od_audit_log",
]

V043_HYPERTABLES = [
    "od_audit_log",
    "od_detections",
    "od_provenance_entries",
]

V043_CONTINUOUS_AGGREGATES = [
    "od_hourly_job_stats",
    "od_daily_detection_stats",
]


# ---------------------------------------------------------------------------
# Pydantic request models for test API routes (module-level for annotation
# resolution -- FastAPI uses inspect.get_annotations(eval_str=True) which
# needs these in the module globals, NOT inside a nested function).
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel as _BaseModel

    class CreateJobRequest(_BaseModel):
        records: List[Dict[str, Any]]
        dataset_id: Optional[str] = None
        pipeline_config: Optional[Dict[str, Any]] = None

    class DetectRequest(_BaseModel):
        records: List[Dict[str, Any]]
        column: str
        methods: Optional[List[str]] = None
        options: Optional[Dict[str, Any]] = None

    class BatchDetectRequest(_BaseModel):
        records: List[Dict[str, Any]]
        columns: Optional[List[str]] = None

    class ClassifyRequest(_BaseModel):
        detections: List[Dict[str, Any]]
        records: List[Dict[str, Any]]

    class TreatRequest(_BaseModel):
        records: List[Dict[str, Any]]
        detections: List[Dict[str, Any]]
        strategy: Optional[str] = None
        options: Optional[Dict[str, Any]] = None

    class CreateThresholdRequest(_BaseModel):
        column: str
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        source: Optional[str] = None
        context: Optional[str] = None

    class SubmitFeedbackRequest(_BaseModel):
        detection_id: str
        feedback_type: Optional[str] = None
        notes: Optional[str] = None

    class ImpactRequest(_BaseModel):
        original: List[Dict[str, Any]]
        treated: List[Dict[str, Any]]

    class PipelineRequest(_BaseModel):
        records: List[Dict[str, Any]]
        config: Optional[Dict[str, Any]] = None

except ImportError:
    # pydantic not available; models will be None and tests will be skipped
    pass


# ---------------------------------------------------------------------------
# Helper: mount test routes on FastAPI app
# ---------------------------------------------------------------------------


def _mount_test_routes(app, svc):
    """Mount lightweight endpoint stubs that delegate to the service facade.

    These are simplified versions of the real API routes, built to
    validate integration behavior without requiring the full router import.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    # -- Jobs --

    @app.post("/api/v1/od/jobs")
    def create_job(req: CreateJobRequest):
        return svc.create_job(request=req.model_dump())

    @app.get("/api/v1/od/jobs")
    def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
        jobs = svc.list_jobs(status=status, limit=limit, offset=offset)
        return {"jobs": jobs, "count": len(jobs), "total": len(svc._jobs)}

    @app.get("/api/v1/od/jobs/{job_id}")
    def get_job(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.delete("/api/v1/od/jobs/{job_id}")
    def delete_job(job_id: str):
        result = svc.delete_job(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")
        return svc.get_job(job_id)

    @app.post("/api/v1/od/jobs/{job_id}/start")
    def start_job(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        job["status"] = "detecting"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        return job

    @app.get("/api/v1/od/jobs/{job_id}/status")
    def get_job_status(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "status": job["status"]}

    @app.get("/api/v1/od/jobs/{job_id}/results")
    def get_job_results(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "results": [], "status": job["status"]}

    @app.get("/api/v1/od/jobs/{job_id}/provenance")
    def get_job_provenance(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "provenance_hash": job.get("provenance_hash", "")}

    # -- Detect --

    @app.post("/api/v1/od/detect")
    def detect(req: DetectRequest):
        result = svc.detect_outliers(
            records=req.records,
            column=req.column,
            methods=req.methods,
            options=req.options,
        )
        return result.model_dump(mode="json")

    @app.post("/api/v1/od/detect/batch")
    def detect_batch(req: BatchDetectRequest):
        result = svc.detect_batch(
            records=req.records,
            columns=req.columns,
        )
        return result.model_dump(mode="json")

    @app.get("/api/v1/od/detect/{detection_id}")
    def get_detection(detection_id: str):
        result = svc.get_detection(detection_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Detection not found")
        return result.model_dump(mode="json")

    # -- Classify --

    @app.post("/api/v1/od/classify")
    def classify(req: ClassifyRequest):
        result = svc.classify_outliers(
            detections=req.detections,
            records=req.records,
        )
        return result.model_dump(mode="json")

    @app.get("/api/v1/od/classify/{classification_id}")
    def get_classification(classification_id: str):
        result = svc.get_classification(classification_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Classification not found")
        return result.model_dump(mode="json")

    # -- Treat --

    @app.post("/api/v1/od/treat")
    def treat(req: TreatRequest):
        result = svc.apply_treatment(
            records=req.records,
            detections=req.detections,
            strategy=req.strategy or "flag",
            options=req.options,
        )
        return result.model_dump(mode="json")

    @app.post("/api/v1/od/treat/{treatment_id}/undo")
    def undo_treatment(treatment_id: str):
        result = svc.undo_treatment(treatment_id)
        if not result:
            raise HTTPException(status_code=404, detail="Treatment not found or not reversible")
        return {"treatment_id": treatment_id, "undone": True}

    @app.get("/api/v1/od/treat/{treatment_id}")
    def get_treatment(treatment_id: str):
        result = svc.get_treatment(treatment_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Treatment not found")
        return result.model_dump(mode="json")

    # -- Thresholds --

    @app.post("/api/v1/od/thresholds")
    def create_threshold(req: CreateThresholdRequest):
        result = svc.create_threshold(
            column=req.column,
            min_val=req.min_val,
            max_val=req.max_val,
            source=req.source or "domain",
            context=req.context or "",
        )
        return result.model_dump(mode="json")

    @app.get("/api/v1/od/thresholds")
    def list_thresholds():
        thresholds = svc.list_thresholds()
        return {
            "thresholds": [t.model_dump(mode="json") for t in thresholds],
            "count": len(thresholds),
        }

    # -- Feedback --

    @app.post("/api/v1/od/feedback")
    def submit_feedback(req: SubmitFeedbackRequest):
        result = svc.submit_feedback(
            detection_id=req.detection_id,
            feedback_type=req.feedback_type or "confirmed_outlier",
            notes=req.notes or "",
        )
        return result.model_dump(mode="json")

    # -- Impact --

    @app.post("/api/v1/od/impact")
    def analyze_impact(req: ImpactRequest):
        result = svc.analyze_impact(
            original=req.original,
            treated=req.treated,
        )
        return result

    # -- Pipeline --

    @app.post("/api/v1/od/pipeline")
    def pipeline(req: PipelineRequest):
        result = svc.run_pipeline(records=req.records, config=req.config)
        return result.model_dump(mode="json")

    # -- Health / Stats --

    @app.get("/api/v1/od/health")
    def health():
        return svc.health_check()

    @app.get("/api/v1/od/stats")
    def stats():
        return svc.get_statistics().model_dump(mode="json")

    # -- Auth-required sentinel (simulated) --

    @app.get("/api/v1/od/protected")
    def protected_endpoint(request: Request):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# FastAPI test client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_app():
    """Create a minimal FastAPI app with the service attached for API tests.

    Returns a FastAPI app mock with state.outlier_detector_service set.
    """
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("FastAPI not installed; skipping API integration tests")

    app = FastAPI(title="OD Integration Test")
    svc = _build_service()
    svc._started = True
    app.state.outlier_detector_service = svc

    # Create simple endpoint stubs that delegate to the service
    _mount_test_routes(app, svc)

    return app


@pytest.fixture
def test_client(mock_app):
    """Create a synchronous test client for FastAPI integration tests."""
    from fastapi.testclient import TestClient

    return TestClient(mock_app)
