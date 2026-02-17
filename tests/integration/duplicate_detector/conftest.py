# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-011 Duplicate Detection Agent integration tests.

Provides reusable test fixtures for:
- Package stub for greenlang.duplicate_detector (same pattern as unit tests)
- Environment cleanup (autouse)
- Database mock fixtures (PostgreSQL + Redis)
- Sample data fixtures (10 records with known duplicates)
- DuplicateDetectorService factory fixture
- FastAPI test client fixture (httpx.AsyncClient)

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures that do not apply to DD tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents fixture.

    The parent tests/integration/conftest.py defines an autouse fixture
    that patches greenlang.agents.registry.get_agent, which does not
    exist. We override it here to make it a no-op for DD integration tests.
    """
    yield {}


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent conftest block_network fixture.

    The parent tests/integration/conftest.py blocks all socket access,
    which prevents asyncio event loop creation needed by some mock DB
    tests. We disable it for DD integration tests since our tests are
    fully self-contained with mocks.
    """
    yield


# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_DD_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_dd_env(monkeypatch):
    """Remove all GL_DD_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_DD_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_DD_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Also reset the singleton config so each test starts fresh
    from greenlang.duplicate_detector.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Helper: build service with patched engines
# ---------------------------------------------------------------------------


def _build_service(config=None):
    """Create a DuplicateDetectorService with engines patched out.

    This is the canonical way to create a service instance in tests
    without requiring the full SDK engine imports.

    Args:
        config: Optional DuplicateDetectorConfig. Uses default if None.

    Returns:
        DuplicateDetectorService instance with engines patched.
    """
    from greenlang.duplicate_detector.config import (
        DuplicateDetectorConfig,
        set_config,
    )
    from greenlang.duplicate_detector.setup import DuplicateDetectorService

    cfg = config or DuplicateDetectorConfig()
    set_config(cfg)

    with patch.object(DuplicateDetectorService, "_init_engines"):
        svc = DuplicateDetectorService(config=cfg)
    return svc


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create a DuplicateDetectorConfig with test defaults."""
    from greenlang.duplicate_detector.config import DuplicateDetectorConfig

    return DuplicateDetectorConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        s3_bucket="test-dedup-bucket",
        max_records_per_job=100_000,
        default_batch_size=5_000,
        fingerprint_algorithm="sha256",
        fingerprint_normalize=True,
        blocking_strategy="sorted_neighborhood",
        blocking_window_size=10,
        blocking_key_size=3,
        canopy_tight_threshold=0.8,
        canopy_loose_threshold=0.4,
        default_similarity_algorithm="jaro_winkler",
        ngram_size=3,
        match_threshold=0.85,
        possible_threshold=0.65,
        non_match_threshold=0.40,
        use_fellegi_sunter=False,
        cluster_algorithm="union_find",
        cluster_min_quality=0.5,
        default_merge_strategy="keep_most_complete",
        merge_conflict_resolution="most_complete",
        pipeline_checkpoint_interval=1000,
        pipeline_timeout_seconds=3600,
        max_comparisons_per_block=50_000,
        cache_ttl_seconds=3600,
        cache_enabled=True,
        pool_min_size=2,
        pool_max_size=10,
        log_level="INFO",
        enable_metrics=True,
        max_field_weights=50,
        max_rules_per_job=100,
        comparison_sample_rate=1.0,
    )


# ---------------------------------------------------------------------------
# Service factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def service(config):
    """Create a DuplicateDetectorService instance for integration testing.

    Engines are patched out so the service can run without the full SDK.
    """
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
# Sample data fixtures with known duplicates
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records_10() -> List[Dict[str, Any]]:
    """Return 10 records -- 4 unique entities, 6 are near-duplicates.

    Known duplicate groups:
    - Group A: rec-001, rec-002 (name case + address abbreviation)
    - Group B: rec-003, rec-004 (name typo + phone format)
    - Group C: rec-005, rec-006 (middle initial + address abbreviation)
    - Group D: rec-007, rec-008 (exact same name + phone variation)
    - Unique:  rec-009, rec-010
    """
    return [
        {
            "id": "rec-001",
            "name": "Alice Johnson",
            "email": "alice.johnson@greenco.com",
            "phone": "+1-555-0101",
            "address": "123 Oak Street",
            "city": "Portland",
            "state": "OR",
            "zip": "97201",
            "amount": 15200.50,
            "date": "2025-06-15",
        },
        {
            "id": "rec-002",
            "name": "alice johnson",
            "email": "alice.johnson@greenco.com",
            "phone": "15550101",
            "address": "123 Oak St",
            "city": "Portland",
            "state": "OR",
            "zip": "97201",
            "amount": 15200.50,
            "date": "2025-06-15",
        },
        {
            "id": "rec-003",
            "name": "Bob Martinez",
            "email": "bob.martinez@ecoworks.io",
            "phone": "+1-555-0102",
            "address": "456 Elm Avenue",
            "city": "Seattle",
            "state": "WA",
            "zip": "98101",
            "amount": 22300.75,
            "date": "2025-07-20",
        },
        {
            "id": "rec-004",
            "name": "Bob Martinex",
            "email": "bob.martinez@ecoworks.io",
            "phone": "+1 555 0102",
            "address": "456 Elm Avenue",
            "city": "Seattle",
            "state": "WA",
            "zip": "98101",
            "amount": 22300.75,
            "date": "2025-07-20",
        },
        {
            "id": "rec-005",
            "name": "Clara Lee",
            "email": "clara.lee@sustain.org",
            "phone": "+1-555-0103",
            "address": "789 Pine Road",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "amount": 8750.00,
            "date": "2025-08-01",
        },
        {
            "id": "rec-006",
            "name": "Clara M. Lee",
            "email": "clara.lee@sustain.org",
            "phone": "+1-555-0103",
            "address": "789 Pine Rd",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "amount": 8750.00,
            "date": "2025-08-01",
        },
        {
            "id": "rec-007",
            "name": "Henry Patel",
            "email": "henry.patel@cleantech.io",
            "phone": "+1-555-0108",
            "address": "258 Spruce Boulevard",
            "city": "Minneapolis",
            "state": "MN",
            "zip": "55401",
            "amount": 28900.80,
            "date": "2025-10-15",
        },
        {
            "id": "rec-008",
            "name": "Henry Patel",
            "email": "henry.patel@cleantech.io",
            "phone": "(555) 010-8",
            "address": "258 Spruce Blvd",
            "city": "Minneapolis",
            "state": "MN",
            "zip": "55401",
            "amount": 28900.80,
            "date": "2025-10-15",
        },
        {
            "id": "rec-009",
            "name": "David Chen",
            "email": "david.chen@carbonzero.net",
            "phone": "+1-555-0104",
            "address": "321 Birch Lane",
            "city": "Denver",
            "state": "CO",
            "zip": "80201",
            "amount": 31400.25,
            "date": "2025-08-15",
        },
        {
            "id": "rec-010",
            "name": "Emma Williams",
            "email": "emma.w@greenfuture.com",
            "phone": "+1-555-0105",
            "address": "654 Maple Drive",
            "city": "Austin",
            "state": "TX",
            "zip": "73301",
            "amount": 19800.60,
            "date": "2025-09-10",
        },
    ]


@pytest.fixture
def exact_duplicate_records() -> List[Dict[str, Any]]:
    """Return 5 records where all are exact duplicates of each other."""
    base = {
        "id": "dup-all",
        "name": "Duplicate Person",
        "email": "dup@example.com",
        "phone": "555-9999",
        "address": "100 Same Street",
        "city": "Sameville",
        "state": "CA",
        "zip": "90001",
        "amount": 1000.00,
        "date": "2025-01-01",
    }
    return [dict(base, id=f"dup-all-{i}") for i in range(5)]


@pytest.fixture
def unique_records() -> List[Dict[str, Any]]:
    """Return 5 records where none are duplicates of each other."""
    return [
        {
            "id": "uniq-001",
            "name": "Zara Nakamura",
            "email": "zara@company1.com",
            "phone": "+1-111-0001",
            "address": "1 First Avenue",
            "city": "New York",
            "state": "NY",
            "zip": "10001",
            "amount": 5000.00,
            "date": "2025-01-15",
        },
        {
            "id": "uniq-002",
            "name": "Yuki Tanaka",
            "email": "yuki@company2.com",
            "phone": "+1-222-0002",
            "address": "2 Second Boulevard",
            "city": "Los Angeles",
            "state": "CA",
            "zip": "90001",
            "amount": 7500.00,
            "date": "2025-02-20",
        },
        {
            "id": "uniq-003",
            "name": "Xavier Dupont",
            "email": "xavier@company3.com",
            "phone": "+1-333-0003",
            "address": "3 Third Lane",
            "city": "Chicago",
            "state": "IL",
            "zip": "60601",
            "amount": 9000.00,
            "date": "2025-03-25",
        },
        {
            "id": "uniq-004",
            "name": "Wanda Kowalski",
            "email": "wanda@company4.com",
            "phone": "+1-444-0004",
            "address": "4 Fourth Circle",
            "city": "Houston",
            "state": "TX",
            "zip": "77001",
            "amount": 12000.00,
            "date": "2025-04-10",
        },
        {
            "id": "uniq-005",
            "name": "Victor Alvarez",
            "email": "victor@company5.com",
            "phone": "+1-555-0005",
            "address": "5 Fifth Terrace",
            "city": "Phoenix",
            "state": "AZ",
            "zip": "85001",
            "amount": 15000.00,
            "date": "2025-05-05",
        },
    ]


@pytest.fixture
def cross_dataset_records() -> Dict[str, List[Dict[str, Any]]]:
    """Return two datasets that share overlapping records for cross-dataset dedup."""
    dataset_a = [
        {
            "id": "ds-a-001",
            "name": "Alice Johnson",
            "email": "alice@greenco.com",
            "city": "Portland",
            "state": "OR",
        },
        {
            "id": "ds-a-002",
            "name": "Bob Martinez",
            "email": "bob@ecoworks.io",
            "city": "Seattle",
            "state": "WA",
        },
        {
            "id": "ds-a-003",
            "name": "Unique Person A",
            "email": "unique-a@test.com",
            "city": "Denver",
            "state": "CO",
        },
    ]
    dataset_b = [
        {
            "id": "ds-b-001",
            "name": "Alice Johnson",
            "email": "alice@greenco.com",
            "city": "Portland",
            "state": "OR",
        },
        {
            "id": "ds-b-002",
            "name": "Unique Person B",
            "email": "unique-b@test.com",
            "city": "Miami",
            "state": "FL",
        },
        {
            "id": "ds-b-003",
            "name": "Bob Martinez",
            "email": "bob@ecoworks.io",
            "city": "Seattle",
            "state": "WA",
        },
    ]
    return {"dataset_a": dataset_a, "dataset_b": dataset_b}


# ---------------------------------------------------------------------------
# FastAPI test client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_app():
    """Create a minimal FastAPI app with the service attached for API tests.

    Returns a FastAPI app mock with state.duplicate_detector_service set.
    """
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("FastAPI not installed; skipping API integration tests")

    app = FastAPI(title="DD Integration Test")
    svc = _build_service()
    svc._started = True
    app.state.duplicate_detector_service = svc

    # Create simple endpoint stubs that delegate to the service
    _mount_test_routes(app, svc)

    return app


@pytest.fixture
def test_client(mock_app):
    """Create a synchronous test client for FastAPI integration tests."""
    from fastapi.testclient import TestClient

    return TestClient(mock_app)


# ---------------------------------------------------------------------------
# V041 migration table definitions (for database integration mock tests)
# ---------------------------------------------------------------------------


V041_TABLES = [
    "dd_jobs",
    "dd_fingerprints",
    "dd_blocks",
    "dd_comparisons",
    "dd_classifications",
    "dd_clusters",
    "dd_cluster_members",
    "dd_merge_decisions",
    "dd_rules",
    "dd_audit_log",
]

V041_HYPERTABLES = [
    "dd_audit_log",
    "dd_comparisons",
    "dd_fingerprints",
]

V041_CONTINUOUS_AGGREGATES = [
    "dd_hourly_job_stats",
    "dd_daily_comparison_stats",
]


# ---------------------------------------------------------------------------
# Pydantic request models for test API routes (module-level for annotation
# resolution -- FastAPI uses inspect.get_annotations(eval_str=True) which
# needs these in the module globals, NOT inside a nested function).
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel as _BaseModel

    class CreateJobRequest(_BaseModel):
        dataset_ids: List[str]
        rule_id: Optional[str] = None

    class FingerprintRequest(_BaseModel):
        records: List[Dict[str, Any]]
        field_set: Optional[List[str]] = None
        algorithm: Optional[str] = None

    class BlockRequest(_BaseModel):
        records: List[Dict[str, Any]]
        strategy: Optional[str] = None
        key_fields: Optional[List[str]] = None

    class CompareRequest(_BaseModel):
        pairs: List[Dict[str, Any]]
        field_configs: Optional[List[Dict[str, Any]]] = None

    class ClassifyRequest(_BaseModel):
        comparisons: List[Dict[str, Any]]
        thresholds: Optional[Dict[str, float]] = None

    class ClusterRequest(_BaseModel):
        matches: List[Dict[str, Any]]
        algorithm: Optional[str] = None

    class MergeRequest(_BaseModel):
        clusters: List[Dict[str, Any]]
        records: List[Dict[str, Any]]
        strategy: Optional[str] = None

    class PipelineRequest(_BaseModel):
        records: List[Dict[str, Any]]
        options: Optional[Dict[str, Any]] = None

    class CreateRuleRequest(_BaseModel):
        rule_config: Dict[str, Any]

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

    @app.post("/api/v1/dd/jobs")
    def create_job(req: CreateJobRequest):
        return svc.create_dedup_job(
            dataset_ids=req.dataset_ids,
            rule_id=req.rule_id,
        )

    @app.get("/api/v1/dd/jobs")
    def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
        return svc.list_jobs(status=status, limit=limit, offset=offset)

    @app.get("/api/v1/dd/jobs/{job_id}")
    def get_job(job_id: str):
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.delete("/api/v1/dd/jobs/{job_id}")
    def delete_job(job_id: str):
        result = svc.cancel_job(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return result

    # -- Fingerprint --

    @app.post("/api/v1/dd/fingerprint")
    def fingerprint(req: FingerprintRequest):
        return svc.fingerprint_records(
            records=req.records,
            field_set=req.field_set,
            algorithm=req.algorithm,
        ).model_dump(mode="json")

    # -- Block --

    @app.post("/api/v1/dd/block")
    def block(req: BlockRequest):
        return svc.create_blocks(
            records=req.records,
            strategy=req.strategy,
            key_fields=req.key_fields,
        ).model_dump(mode="json")

    # -- Compare --

    @app.post("/api/v1/dd/compare")
    def compare(req: CompareRequest):
        return svc.compare_pairs(
            block_results={"pairs": req.pairs},
            field_configs=req.field_configs,
        ).model_dump(mode="json")

    # -- Classify --

    @app.post("/api/v1/dd/classify")
    def classify(req: ClassifyRequest):
        return svc.classify_matches(
            comparisons=req.comparisons,
            thresholds=req.thresholds,
        ).model_dump(mode="json")

    # -- Matches --

    @app.get("/api/v1/dd/matches")
    def list_matches():
        return {"matches": list(svc._matches.values()), "count": len(svc._matches)}

    @app.get("/api/v1/dd/matches/{match_id}")
    def get_match(match_id: str):
        match = svc.get_match_details(match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="Match not found")
        return match

    # -- Clusters --

    @app.post("/api/v1/dd/clusters")
    def create_clusters(req: ClusterRequest):
        return svc.form_clusters(
            matches=req.matches,
            algorithm=req.algorithm,
        ).model_dump(mode="json")

    @app.get("/api/v1/dd/clusters")
    def list_clusters():
        return {"clusters": list(svc._clusters.values()), "count": len(svc._clusters)}

    # -- Merge --

    @app.post("/api/v1/dd/merge")
    def merge(req: MergeRequest):
        return svc.merge_duplicates(
            clusters=req.clusters,
            records=req.records,
            strategy=req.strategy,
        ).model_dump(mode="json")

    # -- Pipeline --

    @app.post("/api/v1/dd/pipeline")
    def pipeline(req: PipelineRequest):
        return svc.run_pipeline(
            records=req.records,
            options=req.options,
        ).model_dump(mode="json")

    # -- Health / Stats --

    @app.get("/api/v1/dd/health")
    def health():
        return svc.health_check()

    @app.get("/api/v1/dd/stats")
    def stats():
        return svc.get_statistics().model_dump(mode="json")

    # -- Rules --

    @app.post("/api/v1/dd/rules")
    def create_rule(req: CreateRuleRequest):
        return svc.create_rule(rule_config=req.rule_config)

    # -- Auth-required sentinel (simulated) --

    @app.get("/api/v1/dd/protected")
    def protected_endpoint(request: Request):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")
        return {"status": "ok"}
