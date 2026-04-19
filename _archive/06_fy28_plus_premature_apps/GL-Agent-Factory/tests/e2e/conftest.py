"""
End-to-End Test Configuration and Fixtures

This module provides shared fixtures and configuration for the E2E test suite,
including:

- API client fixtures (httpx async client)
- Test database setup/teardown
- Docker Compose service management
- Authentication fixtures
- Test data generators

Run with: pytest tests/e2e/ -v -m e2e

Environment Variables:
    E2E_BASE_URL: Base URL for API server (default: http://localhost:8000)
    E2E_REGISTRY_URL: Registry service URL (default: http://localhost:8002)
    E2E_RUNTIME_URL: Agent runtime URL (default: http://localhost:8001)
    E2E_TIMEOUT: Request timeout in seconds (default: 30)
    E2E_DOCKER_COMPOSE: Path to docker-compose.yml
"""

import asyncio
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from uuid import uuid4

import pytest

# Conditional imports for optional dependencies
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    async_playwright = None
    Browser = None
    Page = None
    PLAYWRIGHT_AVAILABLE = False


# =============================================================================
# Environment Configuration
# =============================================================================

E2E_BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
E2E_REGISTRY_URL = os.environ.get("E2E_REGISTRY_URL", "http://localhost:8002")
E2E_RUNTIME_URL = os.environ.get("E2E_RUNTIME_URL", "http://localhost:8001")
E2E_TIMEOUT = int(os.environ.get("E2E_TIMEOUT", "30"))
E2E_DOCKER_COMPOSE = os.environ.get(
    "E2E_DOCKER_COMPOSE",
    os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yml")
)

# Test tenant and user configuration
TEST_TENANT_ID = "e2e-test-tenant"
TEST_USER_ID = "e2e-test-user"
TEST_API_KEY = "e2e-test-api-key-12345"


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest for E2E tests."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "requires_docker: test requires Docker services")
    config.addinivalue_line("markers", "requires_playwright: test requires Playwright")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests (session-scoped)."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Docker Compose Fixtures
# =============================================================================


class DockerComposeManager:
    """Manage Docker Compose services for E2E testing."""

    def __init__(self, compose_file: str):
        """
        Initialize Docker Compose manager.

        Args:
            compose_file: Path to docker-compose.yml file
        """
        self.compose_file = compose_file
        self.services_started = False

    def _run_compose_command(self, *args) -> subprocess.CompletedProcess:
        """
        Run a docker-compose command.

        Args:
            *args: Command arguments

        Returns:
            CompletedProcess result
        """
        cmd = ["docker-compose", "-f", self.compose_file] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def start_services(self, services: Optional[List[str]] = None) -> bool:
        """
        Start Docker Compose services.

        Args:
            services: List of services to start (default: all)

        Returns:
            True if services started successfully
        """
        args = ["up", "-d", "--wait"]
        if services:
            args.extend(services)

        result = self._run_compose_command(*args)
        self.services_started = result.returncode == 0
        return self.services_started

    def stop_services(self) -> bool:
        """
        Stop Docker Compose services.

        Returns:
            True if services stopped successfully
        """
        result = self._run_compose_command("down", "-v", "--remove-orphans")
        self.services_started = False
        return result.returncode == 0

    def is_healthy(self, service: str) -> bool:
        """
        Check if a service is healthy.

        Args:
            service: Service name

        Returns:
            True if service is healthy
        """
        result = self._run_compose_command("ps", "--format", "json", service)
        if result.returncode != 0:
            return False
        return "healthy" in result.stdout.lower() or "running" in result.stdout.lower()

    def wait_for_services(self, timeout: int = 120) -> bool:
        """
        Wait for all services to be healthy.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if all services healthy within timeout
        """
        services = ["api-server", "registry", "postgres", "redis"]
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = all(self.is_healthy(s) for s in services)
            if all_healthy:
                return True
            time.sleep(2)

        return False

    def get_logs(self, service: str, tail: int = 100) -> str:
        """
        Get logs from a service.

        Args:
            service: Service name
            tail: Number of lines to retrieve

        Returns:
            Log output string
        """
        result = self._run_compose_command("logs", "--tail", str(tail), service)
        return result.stdout


@pytest.fixture(scope="session")
def docker_compose_manager():
    """Create Docker Compose manager fixture."""
    return DockerComposeManager(E2E_DOCKER_COMPOSE)


@pytest.fixture(scope="session")
def docker_services(docker_compose_manager):
    """
    Start Docker Compose services for E2E tests.

    This fixture starts all required services at the beginning of the
    test session and stops them at the end.
    """
    # Skip if E2E_TESTS not enabled
    if os.environ.get("E2E_TESTS") != "1":
        pytest.skip("E2E tests disabled (set E2E_TESTS=1 to enable)")

    # Start services
    if not docker_compose_manager.start_services():
        pytest.fail("Failed to start Docker Compose services")

    # Wait for services to be healthy
    if not docker_compose_manager.wait_for_services():
        logs = docker_compose_manager.get_logs("api-server")
        pytest.fail(f"Services did not become healthy. API server logs:\n{logs}")

    yield docker_compose_manager

    # Cleanup
    docker_compose_manager.stop_services()


# =============================================================================
# HTTP Client Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Get API base URL."""
    return E2E_BASE_URL


@pytest.fixture(scope="session")
def registry_base_url() -> str:
    """Get registry base URL."""
    return E2E_REGISTRY_URL


@pytest.fixture(scope="session")
def runtime_base_url() -> str:
    """Get agent runtime base URL."""
    return E2E_RUNTIME_URL


@pytest.fixture
async def api_client(api_base_url) -> AsyncGenerator:
    """
    Create async HTTP client for API server.

    Yields:
        httpx.AsyncClient configured for API testing
    """
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")

    async with httpx.AsyncClient(
        base_url=api_base_url,
        timeout=httpx.Timeout(E2E_TIMEOUT),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    ) as client:
        yield client


@pytest.fixture
async def registry_client(registry_base_url) -> AsyncGenerator:
    """
    Create async HTTP client for registry service.

    Yields:
        httpx.AsyncClient configured for registry testing
    """
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")

    async with httpx.AsyncClient(
        base_url=registry_base_url,
        timeout=httpx.Timeout(E2E_TIMEOUT),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    ) as client:
        yield client


@pytest.fixture
async def authenticated_api_client(api_base_url) -> AsyncGenerator:
    """
    Create authenticated async HTTP client.

    Yields:
        httpx.AsyncClient with authentication headers
    """
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")

    async with httpx.AsyncClient(
        base_url=api_base_url,
        timeout=httpx.Timeout(E2E_TIMEOUT),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {TEST_API_KEY}",
            "X-Tenant-ID": TEST_TENANT_ID,
            "X-Request-ID": str(uuid4()),
        },
    ) as client:
        yield client


@pytest.fixture
async def authenticated_registry_client(registry_base_url) -> AsyncGenerator:
    """
    Create authenticated async HTTP client for registry.

    Yields:
        httpx.AsyncClient with authentication headers for registry
    """
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")

    async with httpx.AsyncClient(
        base_url=registry_base_url,
        timeout=httpx.Timeout(E2E_TIMEOUT),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {TEST_API_KEY}",
            "X-Tenant-ID": TEST_TENANT_ID,
            "X-Request-ID": str(uuid4()),
        },
    ) as client:
        yield client


# =============================================================================
# Playwright Browser Fixtures
# =============================================================================


@pytest.fixture(scope="session")
async def browser() -> AsyncGenerator:
    """
    Create Playwright browser instance (session-scoped).

    Yields:
        Browser instance for UI testing
    """
    if not PLAYWRIGHT_AVAILABLE:
        pytest.skip("Playwright not installed")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser) -> AsyncGenerator:
    """
    Create new browser page for each test.

    Args:
        browser: Browser instance

    Yields:
        Page instance for UI testing
    """
    if not PLAYWRIGHT_AVAILABLE:
        pytest.skip("Playwright not installed")

    context = await browser.new_context()
    page = await context.new_page()
    yield page
    await page.close()
    await context.close()


# =============================================================================
# Database Fixtures
# =============================================================================


class TestDatabaseManager:
    """Manage test database setup and teardown."""

    def __init__(self, connection_string: str):
        """
        Initialize database manager.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.tables_created = False

    async def setup(self):
        """Set up test database schema."""
        # In a real implementation, this would:
        # 1. Connect to the database
        # 2. Create test schema
        # 3. Run migrations
        # 4. Seed test data
        self.tables_created = True

    async def teardown(self):
        """Tear down test database."""
        # In a real implementation, this would:
        # 1. Drop test schema
        # 2. Clean up connections
        self.tables_created = False

    async def truncate_tables(self):
        """Truncate all test tables for clean state."""
        # Truncate tables between tests
        pass

    async def seed_test_data(self, data: Dict[str, Any]):
        """
        Seed test data into database.

        Args:
            data: Dictionary of table names to records
        """
        pass


@pytest.fixture(scope="session")
def db_connection_string() -> str:
    """Get database connection string for tests."""
    return os.environ.get(
        "E2E_DATABASE_URL",
        "postgresql://greenlang:greenlang_dev@localhost:5432/greenlang_test"
    )


@pytest.fixture(scope="session")
async def test_database(db_connection_string) -> AsyncGenerator[TestDatabaseManager, None]:
    """
    Set up test database for E2E tests.

    Yields:
        TestDatabaseManager instance
    """
    manager = TestDatabaseManager(db_connection_string)
    await manager.setup()
    yield manager
    await manager.teardown()


@pytest.fixture
async def clean_database(test_database) -> TestDatabaseManager:
    """
    Provide clean database state for each test.

    Truncates all tables before yielding.
    """
    await test_database.truncate_tables()
    return test_database


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def unique_agent_id() -> str:
    """Generate unique agent ID for test isolation."""
    return f"e2e-test/agent-{uuid4().hex[:8]}"


@pytest.fixture
def sample_agent_spec(unique_agent_id) -> Dict[str, Any]:
    """
    Create sample agent specification for testing.

    Args:
        unique_agent_id: Unique agent ID

    Returns:
        Agent specification dictionary
    """
    return {
        "agent_id": unique_agent_id,
        "name": f"E2E Test Agent {unique_agent_id}",
        "version": "1.0.0",
        "description": "Agent created for E2E testing",
        "category": "test",
        "tags": ["e2e", "test", "automated"],
        "entrypoint": "python://tests.e2e.fixtures:TestAgent",
        "deterministic": True,
        "regulatory_frameworks": ["TEST"],
        "inputs": {
            "input_value": {"type": "number", "description": "Test input value"},
            "multiplier": {"type": "number", "description": "Multiplication factor"},
        },
        "outputs": {
            "result": {"type": "number", "description": "Calculated result"},
            "provenance_hash": {"type": "string", "description": "SHA-256 hash"},
        },
        "metadata": {
            "author": TEST_USER_ID,
            "created_at": datetime.utcnow().isoformat(),
            "tenant_id": TEST_TENANT_ID,
        },
    }


@pytest.fixture
def sample_execution_request(sample_agent_spec) -> Dict[str, Any]:
    """
    Create sample execution request.

    Args:
        sample_agent_spec: Agent specification

    Returns:
        Execution request dictionary
    """
    return {
        "agent_id": sample_agent_spec["agent_id"],
        "version": sample_agent_spec["version"],
        "input_data": {
            "input_value": 100.0,
            "multiplier": 2.5,
        },
        "async_mode": False,
        "timeout_seconds": 60,
        "correlation_id": str(uuid4()),
    }


@pytest.fixture
def sample_carbon_agent_spec() -> Dict[str, Any]:
    """Create sample carbon emissions agent specification."""
    return {
        "agent_id": f"emissions/carbon-calc-{uuid4().hex[:8]}",
        "name": "Carbon Emissions Calculator",
        "version": "1.0.0",
        "description": "Calculate carbon emissions from fuel consumption",
        "category": "emissions",
        "tags": ["carbon", "ghg", "emissions", "scope-1"],
        "entrypoint": "python://agents.carbon_emissions:CarbonEmissionsAgent",
        "deterministic": True,
        "regulatory_frameworks": ["GHG Protocol", "ISO 14064"],
        "inputs": {
            "fuel_type": {"type": "string", "enum": ["diesel", "natural_gas", "coal"]},
            "quantity": {"type": "number", "minimum": 0},
            "unit": {"type": "string", "enum": ["liters", "m3", "kg"]},
            "region": {"type": "string"},
        },
        "outputs": {
            "emissions_kgco2e": {"type": "number"},
            "emission_factor_used": {"type": "number"},
            "provenance_hash": {"type": "string"},
        },
    }


@pytest.fixture
def sample_certification_request(unique_agent_id) -> Dict[str, Any]:
    """
    Create sample certification request.

    Args:
        unique_agent_id: Agent ID to certify

    Returns:
        Certification request dictionary
    """
    return {
        "agent_id": unique_agent_id,
        "version": "1.0.0",
        "certification_level": "standard",
        "regulatory_frameworks": ["TEST"],
        "dimensions": {
            "specification_completeness": True,
            "code_implementation": True,
            "test_coverage": True,
            "deterministic_guarantees": True,
            "documentation": True,
            "compliance_security": True,
        },
        "attestations": [
            {
                "dimension": "test_coverage",
                "coverage_percentage": 85.0,
                "test_count": 50,
            },
            {
                "dimension": "deterministic_guarantees",
                "provenance_verified": True,
                "reproducibility_tested": True,
            },
        ],
    }


# =============================================================================
# Authentication Fixtures
# =============================================================================


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Create authentication headers for API requests."""
    return {
        "Authorization": f"Bearer {TEST_API_KEY}",
        "X-Tenant-ID": TEST_TENANT_ID,
        "X-Request-ID": str(uuid4()),
    }


@pytest.fixture
def admin_auth_headers() -> Dict[str, str]:
    """Create admin authentication headers."""
    return {
        "Authorization": "Bearer admin-test-api-key-12345",
        "X-Tenant-ID": TEST_TENANT_ID,
        "X-Request-ID": str(uuid4()),
        "X-Admin-Access": "true",
    }


# =============================================================================
# Utility Functions
# =============================================================================


def assert_valid_provenance_hash(hash_value: str) -> None:
    """
    Assert that a provenance hash is valid SHA-256.

    Args:
        hash_value: Hash string to validate
    """
    assert hash_value is not None, "Provenance hash should not be None"
    assert len(hash_value) == 64, f"SHA-256 hash should be 64 chars, got {len(hash_value)}"
    assert all(c in "0123456789abcdef" for c in hash_value.lower()), "Invalid hex characters"


def assert_recent_timestamp(timestamp: str, max_age_seconds: int = 60) -> None:
    """
    Assert that a timestamp is recent.

    Args:
        timestamp: ISO format timestamp string
        max_age_seconds: Maximum age in seconds
    """
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    now = datetime.utcnow()
    delta = abs((now - dt.replace(tzinfo=None)).total_seconds())
    assert delta < max_age_seconds, f"Timestamp too old: {delta}s > {max_age_seconds}s"


def assert_api_success(response, expected_status: int = 200) -> Dict[str, Any]:
    """
    Assert API response is successful and return JSON.

    Args:
        response: httpx Response object
        expected_status: Expected HTTP status code

    Returns:
        Response JSON as dictionary
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}: {response.text}"
    return response.json()


def assert_api_error(response, expected_status: int, error_code: str = None) -> Dict[str, Any]:
    """
    Assert API response is an error.

    Args:
        response: httpx Response object
        expected_status: Expected HTTP error status code
        error_code: Expected error code in response

    Returns:
        Response JSON as dictionary
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}"

    data = response.json()
    if error_code:
        assert "error" in data or "detail" in data
        if "error" in data and "code" in data["error"]:
            assert data["error"]["code"] == error_code

    return data


async def wait_for_condition(
    condition_fn,
    timeout: int = 30,
    interval: float = 0.5,
    message: str = "Condition not met"
) -> bool:
    """
    Wait for a condition to become true.

    Args:
        condition_fn: Async function returning boolean
        timeout: Maximum wait time in seconds
        interval: Check interval in seconds
        message: Error message if timeout

    Returns:
        True if condition met

    Raises:
        TimeoutError: If condition not met within timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if await condition_fn():
            return True
        await asyncio.sleep(interval)

    raise TimeoutError(f"{message} (timeout: {timeout}s)")


# =============================================================================
# Test Markers
# =============================================================================


# Convenience markers for common test categories
requires_docker = pytest.mark.requires_docker
requires_playwright = pytest.mark.requires_playwright
e2e = pytest.mark.e2e
slow = pytest.mark.slow
