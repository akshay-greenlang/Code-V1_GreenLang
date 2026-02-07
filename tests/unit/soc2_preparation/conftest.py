# -*- coding: utf-8 -*-
"""
SOC 2 Preparation Test Fixtures - SEC-009 Phase 11

Shared pytest fixtures for SOC 2 unit tests including:
- Mock services and clients
- Sample data models (Assessment, Evidence, Finding, etc.)
- FastAPI test clients
- Configuration fixtures

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def soc2_config() -> Dict[str, Any]:
    """Sample SOC 2 configuration."""
    return {
        "environment": "test",
        "postgres_url": "postgresql+asyncpg://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/15",
        "evidence_bucket": "s3://greenlang-soc2-evidence-test",
        "sla_critical_hours": 4,
        "sla_high_hours": 24,
        "sla_normal_hours": 48,
        "sla_low_hours": 72,
        "assessment_cache_ttl": 300,
        "evidence_retention_days": 90,
        "max_concurrent_assessments": 5,
        "enable_automated_evidence": True,
    }


# ---------------------------------------------------------------------------
# Mock Service Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.exists = AsyncMock(return_value=False)
    redis.expire = AsyncMock(return_value=True)
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=True)
    redis.publish = AsyncMock(return_value=1)
    return redis


@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """Mock database connection pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    pool.execute = AsyncMock()
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchone = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=None)
    return pool


@pytest.fixture
def mock_s3_client() -> MagicMock:
    """Mock S3 client."""
    client = MagicMock()
    client.put_object = MagicMock(return_value={"ETag": "test-etag"})
    client.get_object = MagicMock(return_value={"Body": MagicMock()})
    client.generate_presigned_url = MagicMock(
        return_value="https://s3.example.com/presigned-url"
    )
    client.list_objects_v2 = MagicMock(return_value={"Contents": []})
    return client


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_criterion_id() -> str:
    """Sample SOC 2 criterion ID."""
    return "CC6.1"


@pytest.fixture
def sample_criterion_ids() -> List[str]:
    """List of sample criterion IDs."""
    return ["CC6.1", "CC6.2", "CC6.3", "CC7.1", "CC7.2"]


@pytest.fixture
def sample_assessment_id() -> uuid.UUID:
    """Sample assessment UUID."""
    return uuid.uuid4()


@pytest.fixture
def sample_evidence_id() -> uuid.UUID:
    """Sample evidence UUID."""
    return uuid.uuid4()


@pytest.fixture
def sample_finding_id() -> uuid.UUID:
    """Sample finding UUID."""
    return uuid.uuid4()


@pytest.fixture
def sample_date_range() -> Dict[str, datetime]:
    """Sample date range for audit period."""
    return {
        "start": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "end": datetime(2026, 12, 31, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_assessment_data() -> Dict[str, Any]:
    """Sample assessment data."""
    return {
        "assessment_id": str(uuid.uuid4()),
        "status": "in_progress",
        "overall_score": 72.5,
        "total_criteria": 48,
        "criteria_complete": 28,
        "criteria_in_progress": 15,
        "criteria_not_started": 5,
        "category_scores": {
            "security": 78.0,
            "availability": 65.0,
            "confidentiality": 70.0,
            "processing_integrity": 68.0,
            "privacy": 72.0,
        },
        "last_run": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_evidence_data() -> Dict[str, Any]:
    """Sample evidence data."""
    return {
        "evidence_id": str(uuid.uuid4()),
        "criterion_id": "CC6.1",
        "title": "MFA Configuration Export",
        "evidence_type": "configuration",
        "source": "auth_service",
        "status": "validated",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "period_start": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "period_end": datetime(2026, 2, 1, tzinfo=timezone.utc).isoformat(),
        "provenance_hash": "a1b2c3d4e5f6789012345678901234567890abcdef",
    }


@pytest.fixture
def sample_finding_data() -> Dict[str, Any]:
    """Sample finding data."""
    return {
        "finding_id": str(uuid.uuid4()),
        "title": "Incomplete MFA coverage for service accounts",
        "description": "5 legacy service accounts do not have MFA enabled",
        "criterion_id": "CC6.7",
        "category": "control_deficiency",
        "severity": "high",
        "priority": 1,
        "status": "in_progress",
        "source": "control_test",
        "identified_by": "automation",
        "owner": "security-team",
        "due_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    }


@pytest.fixture
def sample_test_case_data() -> Dict[str, Any]:
    """Sample control test case data."""
    return {
        "test_id": "CC6.1.1",
        "criterion_id": "CC6.1",
        "test_type": "automated",
        "description": "Verify MFA is enforced for all user accounts",
        "procedure": "Query auth_service API for MFA status",
        "expected_result": "100% MFA enrollment",
        "frequency": "daily",
        "owner": "security-team",
        "enabled": True,
    }


@pytest.fixture
def sample_test_result_data() -> Dict[str, Any]:
    """Sample test result data."""
    return {
        "result_id": str(uuid.uuid4()),
        "test_id": "CC6.1.1",
        "test_run_id": str(uuid.uuid4()),
        "status": "passed",
        "actual_result": "All users have MFA enabled",
        "evidence_count": 3,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": 1250,
        "executed_by": "automation",
    }


@pytest.fixture
def sample_attestation_data() -> Dict[str, Any]:
    """Sample attestation data."""
    return {
        "attestation_id": str(uuid.uuid4()),
        "attestation_type": "soc2_readiness",
        "title": "Q1 2026 SOC 2 Readiness Attestation",
        "description": "Management attestation of SOC 2 audit readiness",
        "audit_period_start": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "audit_period_end": datetime(2026, 3, 31, tzinfo=timezone.utc).isoformat(),
        "status": "pending_signatures",
        "signers": [
            {
                "signer_id": "cto@example.com",
                "name": "Jane CTO",
                "email": "cto@example.com",
                "title": "Chief Technology Officer",
                "role": "primary_signer",
                "signature_status": "sent",
            }
        ],
    }


@pytest.fixture
def sample_project_data() -> Dict[str, Any]:
    """Sample audit project data."""
    return {
        "project_id": str(uuid.uuid4()),
        "name": "GreenLang SOC 2 Type II Audit 2026",
        "description": "Annual SOC 2 Type II audit",
        "audit_type": "soc2_type2",
        "audit_firm": "Big Four Audit LLC",
        "audit_period_start": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "audit_period_end": datetime(2026, 12, 31, tzinfo=timezone.utc).isoformat(),
        "status": "preparation",
        "overall_progress": 45,
        "trust_services_categories": ["security", "availability"],
    }


# ---------------------------------------------------------------------------
# FastAPI Test Client Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI test application."""
    from greenlang.infrastructure.soc2_preparation.api import soc2_router

    app = FastAPI(title="SOC 2 Test App")
    app.include_router(soc2_router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def authenticated_client(app: FastAPI) -> TestClient:
    """Create an authenticated test client with mock auth."""
    client = TestClient(app)
    # Add mock authentication headers
    client.headers.update(
        {
            "Authorization": "Bearer mock-jwt-token",
            "X-Tenant-ID": "test-tenant",
        }
    )
    return client


# ---------------------------------------------------------------------------
# Auth Context Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_auth_context() -> MagicMock:
    """Mock authentication context."""
    ctx = MagicMock()
    ctx.user_id = "test-user-123"
    ctx.tenant_id = "test-tenant"
    ctx.email = "test@example.com"
    ctx.roles = ["security_analyst", "compliance_officer"]
    ctx.permissions = [
        "soc2:assessment:read",
        "soc2:assessment:write",
        "soc2:evidence:read",
        "soc2:evidence:write",
        "soc2:tests:read",
        "soc2:tests:execute",
        "soc2:findings:read",
        "soc2:findings:manage",
        "soc2:dashboard:view",
    ]
    ctx.has_permission = lambda p: p in ctx.permissions
    return ctx


@pytest.fixture
def mock_auditor_context() -> MagicMock:
    """Mock auditor authentication context."""
    ctx = MagicMock()
    ctx.user_id = "auditor@example.com"
    ctx.tenant_id = "audit-firm-tenant"
    ctx.email = "auditor@example.com"
    ctx.roles = ["external_auditor"]
    ctx.permissions = ["soc2:portal:access"]
    ctx.has_permission = lambda p: p in ctx.permissions
    return ctx


# ---------------------------------------------------------------------------
# Event/Audit Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_audit_logger() -> MagicMock:
    """Mock audit logger."""
    logger = MagicMock()
    logger.log_event = MagicMock()
    logger.log_access = MagicMock()
    logger.log_modification = MagicMock()
    return logger


@pytest.fixture
def mock_metrics_client() -> MagicMock:
    """Mock Prometheus metrics client."""
    client = MagicMock()
    client.increment = MagicMock()
    client.gauge = MagicMock()
    client.histogram = MagicMock()
    client.observe = MagicMock()
    return client
