"""
Pytest fixtures for Review Console Backend tests.

This module provides shared fixtures for testing the Review Console API,
including database setup, authentication, and test data factories.
"""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from review_console.main import app
from review_console.config import Settings, get_settings
from review_console.db.models import Base, ReviewQueueItem, ReviewStatus
from review_console.db.session import get_db
from review_console.api.auth import create_access_token, Role


# ============================================================================
# Test Settings
# ============================================================================


def get_test_settings() -> Settings:
    """Get test-specific settings."""
    return Settings(
        env="development",
        debug=True,
        secret_key="test-secret-key-must-be-at-least-32-characters-long",
        database_url="sqlite+aiosqlite:///:memory:",
        rate_limit_enabled=False,
    )


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for tests."""
    async_session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session
        await session.rollback()


# ============================================================================
# Application Fixtures
# ============================================================================


@pytest.fixture
def override_settings():
    """Override settings for testing."""
    test_settings = get_test_settings()
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield test_settings
    app.dependency_overrides.pop(get_settings, None)


@pytest_asyncio.fixture
async def async_client(
    async_engine,
    db_session: AsyncSession,
    override_settings,
) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def sync_client(
    override_settings,
) -> Generator[TestClient, None, None]:
    """Create synchronous HTTP client for testing."""
    with TestClient(app) as client:
        yield client


# ============================================================================
# Authentication Fixtures
# ============================================================================


@pytest.fixture
def test_user_token() -> str:
    """Create a test JWT token for a regular reviewer."""
    return create_access_token(
        user_id="test-user-001",
        email="reviewer@test.com",
        name="Test Reviewer",
        roles=["reviewer"],
        org_id="test-org-001",
    )


@pytest.fixture
def test_admin_token() -> str:
    """Create a test JWT token for an admin user."""
    return create_access_token(
        user_id="test-admin-001",
        email="admin@test.com",
        name="Test Admin",
        roles=["admin"],
        org_id="test-org-001",
    )


@pytest.fixture
def test_viewer_token() -> str:
    """Create a test JWT token for a viewer user."""
    return create_access_token(
        user_id="test-viewer-001",
        email="viewer@test.com",
        name="Test Viewer",
        roles=["viewer"],
        org_id="test-org-001",
    )


@pytest.fixture
def auth_headers(test_user_token: str) -> dict:
    """Create authorization headers with test token."""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def admin_headers(test_admin_token: str) -> dict:
    """Create authorization headers with admin token."""
    return {"Authorization": f"Bearer {test_admin_token}"}


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def sample_queue_item(db_session: AsyncSession) -> ReviewQueueItem:
    """Create a sample review queue item."""
    item = ReviewQueueItem(
        id=str(uuid4()),
        input_text="Nat Gas",
        entity_type="fuel",
        org_id="test-org-001",
        source_record_id="src-001",
        pipeline_id="pipeline-001",
        candidates=[
            {
                "id": "GL-FUEL-NATGAS",
                "name": "Natural gas",
                "score": 0.72,
                "source": "fuels_vocab",
                "match_method": "fuzzy",
            },
            {
                "id": "GL-FUEL-LNG",
                "name": "Liquefied natural gas",
                "score": 0.58,
                "source": "fuels_vocab",
                "match_method": "fuzzy",
            },
        ],
        top_candidate_id="GL-FUEL-NATGAS",
        top_candidate_name="Natural gas",
        confidence=0.72,
        match_method="fuzzy",
        context={
            "industry_sector": "energy",
            "region": "US",
        },
        vocabulary_version="2026.01.0",
        status=ReviewStatus.PENDING,
        priority=0,
    )

    db_session.add(item)
    await db_session.commit()
    await db_session.refresh(item)

    return item


@pytest_asyncio.fixture
async def sample_queue_items(db_session: AsyncSession) -> list[ReviewQueueItem]:
    """Create multiple sample review queue items."""
    items = []

    for i in range(5):
        item = ReviewQueueItem(
            id=str(uuid4()),
            input_text=f"Test Entity {i}",
            entity_type="fuel" if i % 2 == 0 else "material",
            org_id="test-org-001" if i < 3 else "test-org-002",
            source_record_id=f"src-{i:03d}",
            candidates=[
                {
                    "id": f"GL-TEST-{i:03d}",
                    "name": f"Test Entity {i}",
                    "score": 0.70 + (i * 0.05),
                    "source": "test_vocab",
                    "match_method": "fuzzy",
                }
            ],
            top_candidate_id=f"GL-TEST-{i:03d}",
            top_candidate_name=f"Test Entity {i}",
            confidence=0.70 + (i * 0.05),
            match_method="fuzzy",
            context={"test": True},
            vocabulary_version="2026.01.0",
            status=ReviewStatus.PENDING,
            priority=i,
        )
        items.append(item)
        db_session.add(item)

    await db_session.commit()

    for item in items:
        await db_session.refresh(item)

    return items
