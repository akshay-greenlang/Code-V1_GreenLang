# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-012 Missing Value Imputer integration tests.

Provides reusable test fixtures for:
- Package stub for greenlang.missing_value_imputer (same pattern as unit tests)
- Environment cleanup (autouse)
- Database mock fixtures (PostgreSQL + Redis)
- Sample data fixtures (records with missing values, time-series, rules)
- MissingValueImputerService factory fixture
- FastAPI test client fixture (httpx/TestClient)

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures that do not apply to MVI tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents fixture.

    The parent tests/integration/conftest.py defines an autouse fixture
    that patches greenlang.agents.registry.get_agent, which does not
    exist. We override it here to make it a no-op for MVI integration tests.
    """
    yield {}


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent conftest block_network fixture.

    The parent tests/integration/conftest.py blocks all socket access,
    which prevents asyncio event loop creation needed by some mock DB
    tests. We disable it for MVI integration tests since our tests are
    fully self-contained with mocks.
    """
    yield


# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_MVI_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_mvi_env(monkeypatch):
    """Remove all GL_MVI_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_MVI_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_MVI_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Also reset the singleton config so each test starts fresh
    from greenlang.missing_value_imputer.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Helper: build service with patched engines
# ---------------------------------------------------------------------------


def _build_service(config=None):
    """Create a MissingValueImputerService with engines patched out.

    This is the canonical way to create a service instance in tests
    without requiring the full SDK engine imports.

    Args:
        config: Optional MissingValueImputerConfig. Uses default if None.

    Returns:
        MissingValueImputerService instance with engines patched.
    """
    from greenlang.missing_value_imputer.config import (
        MissingValueImputerConfig,
        set_config,
    )
    from greenlang.missing_value_imputer.setup import MissingValueImputerService

    cfg = config or MissingValueImputerConfig()
    set_config(cfg)

    with patch.object(MissingValueImputerService, "_init_engines"):
        svc = MissingValueImputerService(config=cfg)
    return svc


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create a MissingValueImputerConfig with test defaults."""
    from greenlang.missing_value_imputer.config import MissingValueImputerConfig

    return MissingValueImputerConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        s3_bucket_url="s3://test-mvi-bucket",
        max_records=100_000,
        batch_size=1_000,
        default_strategy="auto",
        confidence_threshold=0.7,
        max_missing_pct=0.8,
        knn_neighbors=5,
        mice_iterations=10,
        multiple_imputations=5,
        enable_ml_imputation=True,
        enable_timeseries=True,
        enable_rule_based=True,
        enable_statistical=True,
        interpolation_method="linear",
        seasonal_period=12,
        trend_window=6,
        validation_split=0.2,
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
        default_confidence_method="ensemble",
    )


# ---------------------------------------------------------------------------
# Service factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def service(config):
    """Create a MissingValueImputerService instance for integration testing.

    Engines are patched out so the service can run without the full SDK.
    """
    return _build_service(config)


@pytest.fixture
def started_service(service):
    """Create a service that has been marked as started (healthy)."""
    service._started = True
    return service


# ---------------------------------------------------------------------------
# Sample data fixtures with missing values
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records_with_missing() -> List[Dict[str, Any]]:
    """Return 10 records with various missing value patterns.

    Missing pattern:
    - rec-001: complete (no missing)
    - rec-002: revenue missing
    - rec-003: emissions missing
    - rec-004: sector missing
    - rec-005: revenue and emissions missing
    - rec-006: complete (no missing)
    - rec-007: sector and region missing
    - rec-008: emissions missing
    - rec-009: revenue missing
    - rec-010: complete (no missing)
    """
    return [
        {
            "id": "rec-001",
            "company": "EcoTech Corp",
            "sector": "technology",
            "region": "US",
            "revenue": 15200.50,
            "emissions": 1200.0,
            "year": 2025,
        },
        {
            "id": "rec-002",
            "company": "GreenWorks Inc",
            "sector": "manufacturing",
            "region": "EU",
            "revenue": None,
            "emissions": 3400.0,
            "year": 2025,
        },
        {
            "id": "rec-003",
            "company": "SolarPeak Ltd",
            "sector": "energy",
            "region": "US",
            "revenue": 22300.75,
            "emissions": None,
            "year": 2025,
        },
        {
            "id": "rec-004",
            "company": "BioFuel SA",
            "sector": None,
            "region": "EU",
            "revenue": 8750.00,
            "emissions": 890.0,
            "year": 2025,
        },
        {
            "id": "rec-005",
            "company": "WindPower AG",
            "sector": "energy",
            "region": "DE",
            "revenue": None,
            "emissions": None,
            "year": 2025,
        },
        {
            "id": "rec-006",
            "company": "CleanAir Co",
            "sector": "services",
            "region": "US",
            "revenue": 31400.25,
            "emissions": 560.0,
            "year": 2025,
        },
        {
            "id": "rec-007",
            "company": "TerraGreen Pty",
            "sector": None,
            "region": None,
            "revenue": 19800.60,
            "emissions": 2100.0,
            "year": 2025,
        },
        {
            "id": "rec-008",
            "company": "HydroGen Inc",
            "sector": "energy",
            "region": "US",
            "revenue": 42000.00,
            "emissions": None,
            "year": 2025,
        },
        {
            "id": "rec-009",
            "company": "CircularTech",
            "sector": "technology",
            "region": "EU",
            "revenue": None,
            "emissions": 750.0,
            "year": 2025,
        },
        {
            "id": "rec-010",
            "company": "OceanBlue Corp",
            "sector": "services",
            "region": "US",
            "revenue": 28900.80,
            "emissions": 1800.0,
            "year": 2025,
        },
    ]


@pytest.fixture
def exact_missing_records() -> List[Dict[str, Any]]:
    """Return 5 records where all have the same columns missing."""
    return [
        {
            "id": f"exact-{i:03d}",
            "company": f"Company {i}",
            "sector": "energy",
            "revenue": None,
            "emissions": None,
            "year": 2025,
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_time_series() -> List[Dict[str, Any]]:
    """Return 20 timestamped records with gaps in numeric values.

    Gaps at indices: 3, 7, 11, 15 (4 missing values).
    """
    base_values = [
        100.0, 105.0, 110.0, None, 120.0, 125.0, 130.0, None,
        140.0, 145.0, 150.0, None, 160.0, 165.0, 170.0, None,
        180.0, 185.0, 190.0, 195.0,
    ]
    records = []
    for i, val in enumerate(base_values):
        records.append({
            "id": f"ts-{i:03d}",
            "timestamp": f"2025-01-{i + 1:02d}T00:00:00Z",
            "temperature": val,
            "pressure": 1013.0 + i * 0.5 if i % 5 != 0 else None,
            "humidity": 60.0 + i * 0.3,
        })
    return records


@pytest.fixture
def sample_rules() -> List[Dict[str, Any]]:
    """Return 5 domain-specific imputation rules."""
    return [
        {
            "name": "default-sector-rule",
            "target_column": "sector",
            "conditions": [
                {"column": "company", "operator": "contains", "value": "Tech"}
            ],
            "impute_value": "technology",
            "priority": "high",
            "justification": "Companies with Tech in name are technology sector",
        },
        {
            "name": "eu-emissions-default",
            "target_column": "emissions",
            "conditions": [
                {"column": "region", "operator": "equals", "value": "EU"}
            ],
            "impute_value": 1500.0,
            "priority": "medium",
            "justification": "EU average emissions for missing data",
        },
        {
            "name": "revenue-floor-rule",
            "target_column": "revenue",
            "conditions": [
                {"column": "sector", "operator": "equals", "value": "energy"}
            ],
            "impute_value": 10000.0,
            "priority": "medium",
            "justification": "Energy sector minimum revenue assumption",
        },
        {
            "name": "us-region-default",
            "target_column": "region",
            "conditions": [],
            "impute_value": "US",
            "priority": "low",
            "justification": "Default to US region when unknown",
        },
        {
            "name": "critical-emissions-cap",
            "target_column": "emissions",
            "conditions": [
                {"column": "sector", "operator": "equals", "value": "manufacturing"}
            ],
            "impute_value": 5000.0,
            "priority": "critical",
            "justification": "Regulatory cap for manufacturing emissions",
        },
    ]


# ---------------------------------------------------------------------------
# V042 migration table definitions (for database integration mock tests)
# ---------------------------------------------------------------------------


V042_TABLES = [
    "mvi_jobs",
    "mvi_analyses",
    "mvi_imputation_results",
    "mvi_batch_results",
    "mvi_validation_results",
    "mvi_rules",
    "mvi_templates",
    "mvi_column_strategies",
    "mvi_provenance_entries",
    "mvi_audit_log",
]

V042_HYPERTABLES = [
    "mvi_audit_log",
    "mvi_imputation_results",
    "mvi_provenance_entries",
]

V042_CONTINUOUS_AGGREGATES = [
    "mvi_hourly_job_stats",
    "mvi_daily_imputation_stats",
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
        template_id: Optional[str] = None

    class AnalyzeRequest(_BaseModel):
        records: List[Dict[str, Any]]
        columns: Optional[List[str]] = None

    class ImputeRequest(_BaseModel):
        records: List[Dict[str, Any]]
        column: str
        strategy: Optional[str] = None
        options: Optional[Dict[str, Any]] = None

    class BatchImputeRequest(_BaseModel):
        records: List[Dict[str, Any]]
        strategies: Optional[Dict[str, str]] = None

    class ValidateRequest(_BaseModel):
        original: List[Dict[str, Any]]
        imputed: List[Dict[str, Any]]
        method: Optional[str] = None

    class CreateRuleRequest(_BaseModel):
        name: str
        target_column: str
        conditions: Optional[List[Dict[str, Any]]] = None
        impute_value: Optional[Any] = None
        priority: Optional[str] = None
        justification: Optional[str] = None

    class UpdateRuleRequest(_BaseModel):
        name: Optional[str] = None
        target_column: Optional[str] = None
        conditions: Optional[List[Dict[str, Any]]] = None
        impute_value: Optional[Any] = None
        priority: Optional[str] = None
        is_active: Optional[bool] = None
        justification: Optional[str] = None

    class CreateTemplateRequest(_BaseModel):
        name: str
        description: Optional[str] = None
        strategies: Optional[Dict[str, str]] = None

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

    @app.post("/api/v1/mvi/jobs")
    def create_job(req: CreateJobRequest):
        return svc.create_job(request=req.model_dump())

    @app.get("/api/v1/mvi/jobs")
    def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
        jobs = svc.list_jobs(status=status, limit=limit, offset=offset)
        return {"jobs": jobs, "count": len(jobs), "total": len(svc._jobs)}

    @app.get("/api/v1/mvi/jobs/{job_id}")
    def get_job(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.delete("/api/v1/mvi/jobs/{job_id}")
    def delete_job(job_id: str):
        result = svc.delete_job(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")
        return svc.get_job(job_id)

    @app.post("/api/v1/mvi/jobs/{job_id}/start")
    def start_job(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        return job

    @app.get("/api/v1/mvi/jobs/{job_id}/status")
    def get_job_status(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "status": job["status"]}

    @app.get("/api/v1/mvi/jobs/{job_id}/results")
    def get_job_results(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "results": [], "status": job["status"]}

    @app.get("/api/v1/mvi/jobs/{job_id}/provenance")
    def get_job_provenance(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "provenance_hash": job.get("provenance_hash", "")}

    # -- Analyze --

    @app.post("/api/v1/mvi/analyze")
    def analyze(req: AnalyzeRequest):
        result = svc.analyze_missingness(records=req.records, columns=req.columns)
        return result.model_dump(mode="json")

    @app.get("/api/v1/mvi/analyze/{analysis_id}")
    def get_analysis(analysis_id: str):
        result = svc.get_analysis(analysis_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return result.model_dump(mode="json")

    # -- Impute --

    @app.post("/api/v1/mvi/impute")
    def impute(req: ImputeRequest):
        result = svc.impute_values(
            records=req.records,
            column=req.column,
            strategy=req.strategy,
            options=req.options,
        )
        return result.model_dump(mode="json")

    @app.post("/api/v1/mvi/impute/batch")
    def impute_batch(req: BatchImputeRequest):
        result = svc.impute_batch(records=req.records, strategies=req.strategies)
        return result.model_dump(mode="json")

    @app.get("/api/v1/mvi/impute/{result_id}")
    def get_result(result_id: str):
        result = svc.get_results(result_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Result not found")
        return result.model_dump(mode="json")

    # -- Validate --

    @app.post("/api/v1/mvi/validate")
    def validate(req: ValidateRequest):
        result = svc.validate_imputation(
            original=req.original,
            imputed=req.imputed,
            method=req.method or "plausibility_range",
        )
        return result.model_dump(mode="json")

    @app.get("/api/v1/mvi/validate/{validation_id}")
    def get_validation(validation_id: str):
        result = svc.get_validation(validation_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Validation not found")
        return result.model_dump(mode="json")

    # -- Rules --

    @app.post("/api/v1/mvi/rules")
    def create_rule(req: CreateRuleRequest):
        result = svc.create_rule(
            name=req.name,
            target_column=req.target_column,
            conditions=req.conditions,
            impute_value=req.impute_value,
            priority=req.priority or "medium",
            justification=req.justification or "",
        )
        return result.model_dump(mode="json")

    @app.get("/api/v1/mvi/rules")
    def list_rules(is_active: Optional[bool] = None):
        rules = svc.list_rules(is_active=is_active)
        return {"rules": rules, "count": len(rules)}

    @app.put("/api/v1/mvi/rules/{rule_id}")
    def update_rule(rule_id: str, req: UpdateRuleRequest):
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        result = svc.update_rule(rule_id, **updates)
        if result is None:
            raise HTTPException(status_code=404, detail="Rule not found")
        return result.model_dump(mode="json")

    @app.delete("/api/v1/mvi/rules/{rule_id}")
    def delete_rule(rule_id: str):
        result = svc.delete_rule(rule_id)
        if not result:
            raise HTTPException(status_code=404, detail="Rule not found")
        return {"rule_id": rule_id, "deleted": True}

    # -- Templates --

    @app.post("/api/v1/mvi/templates")
    def create_template(req: CreateTemplateRequest):
        result = svc.create_template(
            name=req.name,
            description=req.description or "",
            strategies=req.strategies,
        )
        return result.model_dump(mode="json")

    @app.get("/api/v1/mvi/templates")
    def list_templates():
        templates = svc.list_templates()
        return {"templates": templates, "count": len(templates)}

    # -- Pipeline --

    @app.post("/api/v1/mvi/pipeline")
    def pipeline(req: PipelineRequest):
        result = svc.run_pipeline(records=req.records, config=req.config)
        return result.model_dump(mode="json")

    # -- Health / Stats --

    @app.get("/api/v1/mvi/health")
    def health():
        return svc.health_check()

    @app.get("/api/v1/mvi/stats")
    def stats():
        return svc.get_statistics().model_dump(mode="json")

    # -- Auth-required sentinel (simulated) --

    @app.get("/api/v1/mvi/protected")
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

    Returns a FastAPI app mock with state.missing_value_imputer_service set.
    """
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("FastAPI not installed; skipping API integration tests")

    app = FastAPI(title="MVI Integration Test")
    svc = _build_service()
    svc._started = True
    app.state.missing_value_imputer_service = svc

    # Create simple endpoint stubs that delegate to the service
    _mount_test_routes(app, svc)

    return app


@pytest.fixture
def test_client(mock_app):
    """Create a synchronous test client for FastAPI integration tests."""
    from fastapi.testclient import TestClient

    return TestClient(mock_app)
