# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-013 Outlier Detection Agent integration tests.

Provides reusable test fixtures for:
- Package stub for greenlang.outlier_detector (same pattern as unit tests)
- Environment cleanup (autouse)
- Database mock fixtures (PostgreSQL + Redis)
- Sample data fixtures (records with known outliers)
- OutlierDetectorService factory fixture
- FastAPI test client fixture

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection Agent (GL-DATA-X-016)
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
# Helper: build service with real engines
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create an OutlierDetectorConfig with optional overrides."""
    from greenlang.outlier_detector.config import OutlierDetectorConfig

    defaults = dict(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        s3_bucket_url="s3://test-od-bucket",
        log_level="INFO",
        batch_size=1000,
        max_records=100_000,
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
        enable_metrics=True,
    )
    defaults.update(overrides)
    return OutlierDetectorConfig(**defaults)


def _build_service(config=None):
    """Create an OutlierDetectorService for integration testing.

    All sub-engines are real instances wired to the same config.

    Args:
        config: Optional OutlierDetectorConfig. Uses default if None.

    Returns:
        OutlierDetectorService instance.
    """
    from greenlang.outlier_detector.setup import OutlierDetectorService

    cfg = config or _make_config()
    return OutlierDetectorService(config=cfg)


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create an OutlierDetectorConfig with test defaults."""
    return _make_config()


# ---------------------------------------------------------------------------
# Service factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service(config):
    """Create an OutlierDetectorService instance for integration testing."""
    return _build_service(config)


@pytest.fixture
def started_service(service):
    """Create a service that has been marked as started (healthy)."""
    service._started = True
    return service


# ---------------------------------------------------------------------------
# Mock database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pg_pool():
    """Create a mock PostgreSQL connection pool (asyncpg-style).

    Provides mock connections that support execute, fetch, fetchrow,
    and transaction context managers for database integration tests.
    """
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)

    mock_tx = AsyncMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=mock_tx)

    mock_pool = AsyncMock()
    mock_pool.acquire = AsyncMock()
    mock_pool.acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.__aexit__ = AsyncMock(return_value=False)
    mock_pool.close = AsyncMock()

    return {
        "pool": mock_pool,
        "conn": mock_conn,
        "transaction": mock_tx,
    }


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for caching integration tests.

    Provides a mock that supports get, set, delete, exists, expire,
    pipeline, and other common Redis operations.
    """
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.set = AsyncMock(return_value=True)
    mock_client.delete = AsyncMock(return_value=1)
    mock_client.exists = AsyncMock(return_value=0)
    mock_client.expire = AsyncMock(return_value=True)
    mock_client.ttl = AsyncMock(return_value=-2)
    mock_client.keys = AsyncMock(return_value=[])
    mock_client.mget = AsyncMock(return_value=[])
    mock_client.mset = AsyncMock(return_value=True)
    mock_client.incr = AsyncMock(return_value=1)
    mock_client.close = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)

    mock_pipe = AsyncMock()
    mock_pipe.execute = AsyncMock(return_value=[])
    mock_client.pipeline = MagicMock(return_value=mock_pipe)

    return {
        "client": mock_client,
        "pipeline": mock_pipe,
    }


# ---------------------------------------------------------------------------
# Sample data fixtures with known outliers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records_20() -> List[Dict[str, Any]]:
    """Return 20 records with 3 clear outliers in 'emissions' column.

    Normal range: 9-15
    Outliers: record 3 (500.0), record 5 (-50.0), record 17 (1000.0)
    """
    records = []
    for i in range(20):
        emissions = 10.0 + (i % 6) * 0.8
        sector = ["energy", "transport", "industry"][i % 3]
        records.append({
            "id": f"r-{i:03d}",
            "emissions": emissions,
            "sector": sector,
            "temperature": 20.0 + (i % 5) * 0.5,
        })
    # Inject outliers
    records[3]["emissions"] = 500.0
    records[5]["emissions"] = -50.0
    records[17]["emissions"] = 1000.0
    return records


@pytest.fixture
def large_records_500() -> List[Dict[str, Any]]:
    """Return 500 records with ~10 outliers for performance tests."""
    import math

    records = []
    for i in range(500):
        val = 50.0 + math.sin(i * 0.1) * 10.0
        records.append({
            "id": f"large-{i:04d}",
            "val": val,
            "category": f"cat-{i % 10}",
        })
    # Inject 10 outliers at known positions
    outlier_indices = [42, 99, 150, 200, 251, 300, 350, 399, 450, 488]
    for idx in outlier_indices:
        records[idx]["val"] = 5000.0 + idx
    return records


@pytest.fixture
def time_series_50() -> List[float]:
    """Return 50-point time series with 3 known anomalies."""
    import math

    values = []
    for i in range(50):
        if i == 15:
            values.append(500.0)
        elif i == 30:
            values.append(-300.0)
        elif i == 45:
            values.append(800.0)
        else:
            values.append(10.0 + math.sin(i * 0.5) * 3.0)
    return values


@pytest.fixture
def all_normal_records() -> List[Dict[str, Any]]:
    """Return 10 records with no outliers."""
    return [
        {"id": f"normal-{i}", "val": 10.0 + i * 0.1}
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# V043 migration table definitions (for database integration mock tests)
# ---------------------------------------------------------------------------


V043_TABLES = [
    "outlier_jobs",
    "outlier_detections",
    "outlier_scores",
    "outlier_classifications",
    "outlier_treatments",
    "outlier_thresholds",
    "outlier_feedback",
    "outlier_impact_analyses",
    "outlier_reports",
    "outlier_audit_log",
]

V043_HYPERTABLES = [
    "outlier_events",
    "detection_events",
    "treatment_events",
]

V043_CONTINUOUS_AGGREGATES = [
    "outlier_hourly_stats",
    "detection_hourly_stats",
]


# ---------------------------------------------------------------------------
# FastAPI test client fixture
# ---------------------------------------------------------------------------


try:
    from pydantic import BaseModel as _BaseModel

    class CreateJobBody(_BaseModel):
        dataset_ids: List[str]
        columns: Optional[List[str]] = None
        methods: Optional[List[str]] = None

    class DetectBody(_BaseModel):
        records: List[Dict[str, Any]]
        column: str
        method: Optional[str] = None
        methods: Optional[List[str]] = None

    class BatchDetectBody(_BaseModel):
        records: List[Dict[str, Any]]
        columns: Optional[List[str]] = None
        methods: Optional[List[str]] = None

    class ClassifyBody(_BaseModel):
        records: List[Dict[str, Any]]
        detections: List[Dict[str, Any]]

    class TreatBody(_BaseModel):
        records: List[Dict[str, Any]]
        detections: List[Dict[str, Any]]
        strategy: Optional[str] = None

    class CreateThresholdBody(_BaseModel):
        column_name: str
        lower_bound: Optional[float] = None
        upper_bound: Optional[float] = None
        source: Optional[str] = "domain"
        description: Optional[str] = None

    class SubmitFeedbackBody(_BaseModel):
        detection_id: str
        feedback_type: str
        comment: Optional[str] = None

    class ImpactBody(_BaseModel):
        records: List[Dict[str, Any]]
        treatments: List[Dict[str, Any]]
        column: str

    class RunPipelineBody(_BaseModel):
        records: List[Dict[str, Any]]
        columns: Optional[List[str]] = None
        methods: Optional[List[str]] = None

except ImportError:
    pass


def _mount_test_routes(app, svc):
    """Mount lightweight endpoint stubs that delegate to the service facade.

    These are simplified versions of the real API routes, built to
    validate integration behavior without requiring the full router import.
    """
    from fastapi import HTTPException, Request

    @app.post("/api/v1/outlier/jobs")
    def create_job(req: CreateJobBody):
        # create_job takes a request dict
        job = svc.create_job(request={
            "records": [],
            "dataset_id": ",".join(req.dataset_ids),
        })
        # Add dataset_ids to response for test convenience
        job["dataset_ids"] = req.dataset_ids
        return job

    @app.get("/api/v1/outlier/jobs")
    def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
        jobs = svc.list_jobs(status=status, limit=limit, offset=offset)
        total = len(svc._jobs)
        return {"jobs": jobs, "count": len(jobs), "total": total}

    @app.get("/api/v1/outlier/jobs/{job_id}")
    def get_job(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.delete("/api/v1/outlier/jobs/{job_id}")
    def delete_job(job_id: str):
        result = svc.delete_job(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")
        job = svc.get_job(job_id)
        return job if job else {"job_id": job_id, "status": "deleted"}

    @app.post("/api/v1/outlier/detect")
    def detect(req: DetectBody):
        try:
            # Merge single 'method' into 'methods' list for the service
            methods = req.methods
            if not methods and req.method:
                methods = [req.method]
            return svc.detect_outliers(req.records, req.column, methods=methods)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/v1/outlier/detect/batch")
    def detect_batch(req: BatchDetectBody):
        try:
            return svc.detect_batch(req.records, columns=req.columns)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/v1/outlier/classify")
    def classify(req: ClassifyBody):
        return svc.classify_outliers(req.detections, req.records)

    @app.post("/api/v1/outlier/treat")
    def treat(req: TreatBody):
        return svc.apply_treatment(
            req.records, req.detections, strategy=req.strategy,
        )

    @app.post("/api/v1/outlier/thresholds")
    def create_threshold(req: CreateThresholdBody):
        return svc.create_threshold(
            column=req.column_name,
            min_val=req.lower_bound,
            max_val=req.upper_bound,
            source=req.source or "domain",
            context=req.description or "",
        )

    @app.get("/api/v1/outlier/thresholds")
    def list_thresholds():
        items = svc.list_thresholds()
        return {"thresholds": items, "count": len(items)}

    @app.post("/api/v1/outlier/feedback")
    def submit_feedback(req: SubmitFeedbackBody):
        return svc.submit_feedback(
            detection_id=req.detection_id,
            feedback_type=req.feedback_type,
            notes=req.comment or "",
        )

    @app.post("/api/v1/outlier/impact")
    def analyze_impact(req: ImpactBody):
        return svc.analyze_impact(req.records, req.treatments)

    @app.post("/api/v1/outlier/pipeline")
    def run_pipeline(req: RunPipelineBody):
        try:
            config = None
            if req.columns or req.methods:
                config = {}
                if req.columns:
                    config["columns"] = req.columns
                if req.methods:
                    config["methods"] = req.methods
            return svc.run_pipeline(req.records, config=config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/v1/outlier/health")
    def health():
        return svc.health_check()

    @app.get("/api/v1/outlier/stats")
    def stats():
        return svc.get_statistics()

    @app.get("/api/v1/outlier/protected")
    def protected_endpoint(request: Request):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")
        return {"status": "ok"}


@pytest.fixture
def mock_app():
    """Create a minimal FastAPI app with the service attached for API tests.

    Returns a FastAPI app with state.outlier_detector_service set.
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

    _mount_test_routes(app, svc)

    return app


@pytest.fixture
def test_client(mock_app):
    """Create a synchronous test client for FastAPI integration tests."""
    from fastapi.testclient import TestClient

    return TestClient(mock_app)
