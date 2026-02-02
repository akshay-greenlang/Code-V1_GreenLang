"""
Pytest configuration and fixtures for Agent Registry tests.

This module provides shared fixtures for testing the Agent Registry API,
including database setup, test client, and sample data factories.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator
from datetime import datetime

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from greenlang_registry.db.models import Base, Agent, AgentVersion, LifecycleState
from greenlang_registry.db.client import DatabaseClient
from greenlang_registry.api.app import create_app


# =============================================================================
# Test Configuration
# =============================================================================

# Use SQLite for tests (in-memory) or PostgreSQL if available
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "sqlite+aiosqlite:///:memory:"
)


# =============================================================================
# Event Loop Fixture
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def db_client(test_engine) -> AsyncGenerator[DatabaseClient, None]:
    """Create a test database client."""
    client = DatabaseClient(
        database_url=TEST_DATABASE_URL,
        use_null_pool=True,
    )
    client._engine = test_engine
    client._session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    yield client


# =============================================================================
# API Client Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def app(db_client, monkeypatch):
    """Create a test FastAPI application."""
    # Monkey-patch the database client getter
    def mock_get_database_client():
        return db_client

    monkeypatch.setattr(
        "greenlang_registry.api.routes.get_database_client",
        mock_get_database_client
    )
    monkeypatch.setattr(
        "greenlang_registry.api.app.get_database_client",
        mock_get_database_client
    )

    application = create_app()

    # Skip lifespan for tests (we manage DB separately)
    yield application


@pytest_asyncio.fixture(scope="function")
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def sample_agent(db_session: AsyncSession) -> Agent:
    """Create a sample agent in the database."""
    agent = Agent(
        agent_id="gl-test-agent",
        name="Test Agent",
        description="A test agent for unit testing",
        domain="sustainability.test",
        type="calculator",
        category="testing",
        tags={"tags": ["test", "unit-test"]},
        created_by="test-user",
        team="test-team",
        tenant_id="test-tenant",
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest_asyncio.fixture
async def sample_agent_with_versions(
    db_session: AsyncSession,
    sample_agent: Agent
) -> tuple[Agent, list[AgentVersion]]:
    """Create a sample agent with multiple versions."""
    versions = []

    for i, (ver, state) in enumerate([
        ("1.0.0", LifecycleState.DEPRECATED),
        ("1.1.0", LifecycleState.CERTIFIED),
        ("2.0.0", LifecycleState.EXPERIMENTAL),
        ("2.1.0-alpha", LifecycleState.DRAFT),
    ]):
        version = AgentVersion(
            version_id=f"{sample_agent.agent_id}:{ver}",
            agent_id=sample_agent.agent_id,
            version=ver,
            semantic_version={
                "major": int(ver.split(".")[0]),
                "minor": int(ver.split(".")[1].split("-")[0]),
                "patch": 0,
            },
            lifecycle_state=state,
            container_image=f"gcr.io/greenlang/test-agent:{ver}",
            image_digest=f"sha256:test{i}",
            metadata={"test": True},
            runtime_requirements={
                "cpu_request": "500m",
                "memory_request": "512Mi",
            },
        )
        db_session.add(version)
        versions.append(version)

    await db_session.commit()
    for v in versions:
        await db_session.refresh(v)

    return sample_agent, versions


@pytest.fixture
def publish_request_data() -> dict:
    """Sample data for publish request."""
    return {
        "agent_id": "gl-new-agent",
        "name": "New Test Agent",
        "description": "A newly published test agent",
        "version": "1.0.0",
        "domain": "sustainability.carbon",
        "type": "calculator",
        "category": "emissions",
        "tags": ["carbon", "emissions", "test"],
        "team": "test-team",
        "tenant_id": "test-tenant",
        "container_image": "gcr.io/greenlang/new-agent:1.0.0",
        "runtime_requirements": {
            "cpu_request": "500m",
            "cpu_limit": "2000m",
            "memory_request": "512Mi",
            "memory_limit": "2Gi",
        },
    }


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def mock_request_id() -> str:
    """Generate a mock request ID."""
    return "test-request-id-12345"
