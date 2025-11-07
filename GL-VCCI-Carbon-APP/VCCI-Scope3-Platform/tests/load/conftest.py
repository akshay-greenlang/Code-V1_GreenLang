"""
GL-VCCI Load Testing - Shared Fixtures and Configuration

Provides pytest fixtures and configuration for load testing setup and teardown.

This module contains:
    - Test environment setup/teardown
    - Test data fixtures
    - Mock service fixtures
    - Database fixtures
    - Authentication fixtures

Author: GL-VCCI Team
Version: 1.0.0
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import json
import requests
from datetime import datetime


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest for load testing."""
    config.addinivalue_line(
        "markers", "load: mark test as a load test (deselect with '-m \"not load\"')"
    )
    config.addinivalue_line(
        "markers", "rampup: mark test as ramp-up scenario"
    )
    config.addinivalue_line(
        "markers", "sustained: mark test as sustained load scenario"
    )
    config.addinivalue_line(
        "markers", "spike: mark test as spike test scenario"
    )
    config.addinivalue_line(
        "markers", "endurance: mark test as endurance test scenario"
    )


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--host",
        action="store",
        default="http://localhost:8000",
        help="Target host URL for load tests"
    )
    parser.addoption(
        "--users",
        action="store",
        type=int,
        default=100,
        help="Number of concurrent users"
    )
    parser.addoption(
        "--duration",
        action="store",
        type=int,
        default=60,
        help="Test duration in seconds"
    )
    parser.addoption(
        "--skip-setup",
        action="store_true",
        default=False,
        help="Skip test environment setup"
    )


# ============================================================================
# SESSION-SCOPED FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def load_test_config(request) -> Dict[str, Any]:
    """
    Load test configuration.

    Returns:
        Dictionary with configuration parameters
    """
    return {
        "host": request.config.getoption("--host"),
        "users": request.config.getoption("--users"),
        "duration": request.config.getoption("--duration"),
        "skip_setup": request.config.getoption("--skip-setup"),
    }


@pytest.fixture(scope="session")
def test_output_dir() -> Path:
    """
    Create temporary directory for test outputs.

    Yields:
        Path to output directory

    Cleanup:
        Removes directory after session
    """
    output_dir = Path(tempfile.mkdtemp(prefix="gl_vcci_load_"))
    print(f"\nTest output directory: {output_dir}")

    yield output_dir

    # Cleanup (optional - keep for debugging)
    # shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def api_base_url(load_test_config) -> str:
    """
    Get API base URL from config.

    Returns:
        Base URL string
    """
    return load_test_config["host"]


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_users() -> List[Dict[str, str]]:
    """
    Generate test user credentials.

    Returns:
        List of user dictionaries with email/password
    """
    users = []
    for i in range(1, 101):  # 100 test users
        users.append({
            "email": f"loadtest_{i}@example.com",
            "password": "LoadTest123!",
            "user_id": f"loadtest_{i}"
        })
    return users


@pytest.fixture(scope="function")
def test_user(test_users) -> Dict[str, str]:
    """
    Get single test user for function-scoped tests.

    Returns:
        Single user dictionary
    """
    import random
    return random.choice(test_users)


@pytest.fixture(scope="session")
def sample_procurement_data() -> List[Dict[str, Any]]:
    """
    Generate sample procurement data for testing.

    Returns:
        List of procurement record dictionaries
    """
    from load_test_utils import generate_realistic_procurement_data
    return generate_realistic_procurement_data(1000, seed=42)


@pytest.fixture(scope="function")
def sample_csv_file(tmp_path) -> Path:
    """
    Create sample CSV file for upload testing.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to CSV file
    """
    from load_test_utils import generate_csv_data

    csv_content = generate_csv_data(100, seed=42)
    csv_file = tmp_path / "test_procurement.csv"
    csv_file.write_text(csv_content, encoding='utf-8')

    return csv_file


# ============================================================================
# AUTHENTICATION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def auth_tokens(api_base_url, test_users) -> Dict[str, str]:
    """
    Pre-authenticate test users and cache tokens.

    Args:
        api_base_url: Base API URL
        test_users: List of test users

    Returns:
        Dictionary mapping user_id to auth token
    """
    tokens = {}

    print(f"\nAuthenticating {len(test_users)} test users...")

    for user in test_users[:10]:  # Pre-auth first 10 users
        try:
            response = requests.post(
                f"{api_base_url}/api/auth/login",
                json={
                    "email": user["email"],
                    "password": user["password"]
                },
                timeout=10
            )

            if response.status_code == 200:
                token = response.json().get("access_token")
                if token:
                    tokens[user["user_id"]] = token
            else:
                print(f"Warning: Failed to authenticate {user['email']}: {response.status_code}")

        except Exception as e:
            print(f"Warning: Error authenticating {user['email']}: {e}")

    print(f"Successfully authenticated {len(tokens)} users")

    return tokens


@pytest.fixture(scope="function")
def authenticated_client(api_base_url, test_user, auth_tokens):
    """
    Create authenticated HTTP client for testing.

    Args:
        api_base_url: Base API URL
        test_user: Test user credentials
        auth_tokens: Pre-authenticated tokens

    Returns:
        Tuple of (base_url, headers)
    """
    user_id = test_user["user_id"]

    # Get cached token or authenticate
    if user_id in auth_tokens:
        token = auth_tokens[user_id]
    else:
        # Authenticate on-demand
        response = requests.post(
            f"{api_base_url}/api/auth/login",
            json={
                "email": test_user["email"],
                "password": test_user["password"]
            },
            timeout=10
        )
        token = response.json().get("access_token")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-User-ID": user_id
    }

    return (api_base_url, headers)


# ============================================================================
# ENVIRONMENT SETUP/TEARDOWN
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(load_test_config):
    """
    Set up test environment before all tests.

    This fixture runs automatically at session start.
    """
    if load_test_config["skip_setup"]:
        print("\nSkipping test environment setup (--skip-setup)")
        yield
        return

    print("\n" + "="*80)
    print("GL-VCCI LOAD TEST ENVIRONMENT SETUP")
    print("="*80)

    # Check if target host is reachable
    host = load_test_config["host"]
    print(f"Target host: {host}")

    try:
        response = requests.get(f"{host}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Target host is reachable")
        else:
            print(f"⚠️ Warning: Target host returned {response.status_code}")
    except Exception as e:
        print(f"❌ Error: Cannot reach target host: {e}")
        pytest.exit("Target host unreachable. Please start the application first.")

    print("="*80 + "\n")

    yield

    # Teardown
    print("\n" + "="*80)
    print("GL-VCCI LOAD TEST ENVIRONMENT TEARDOWN")
    print("="*80)
    print("Load tests completed")
    print("="*80 + "\n")


@pytest.fixture(scope="function")
def performance_monitor():
    """
    Monitor performance metrics during test execution.

    Yields:
        Performance monitor object with start/stop methods
    """
    from load_test_utils import SystemMonitor

    monitor = SystemMonitor()
    metrics = {"start": monitor.get_current_stats()}

    yield monitor

    metrics["end"] = monitor.get_current_stats()

    # Print performance summary
    print("\n--- Performance Summary ---")
    print(f"Duration: {metrics['end']['elapsed_seconds']:.2f}s")
    print(f"CPU: {metrics['end']['cpu']['percent_overall']:.1f}%")
    print(f"Memory: {metrics['end']['memory']['percent']:.1f}%")
    print("-" * 27)


# ============================================================================
# RESULT COLLECTION FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def test_results_collector(test_output_dir):
    """
    Collect and save test results.

    Yields:
        Results collector object
    """
    results = {
        "test_name": None,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "metrics": [],
        "errors": [],
    }

    class ResultsCollector:
        def add_metric(self, name: str, value: float):
            results["metrics"].append({"name": name, "value": value, "timestamp": datetime.now().isoformat()})

        def add_error(self, error: str):
            results["errors"].append({"error": error, "timestamp": datetime.now().isoformat()})

        def set_test_name(self, name: str):
            results["test_name"] = name

        def save(self):
            results["end_time"] = datetime.now().isoformat()
            output_file = test_output_dir / f"{results['test_name']}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")

    collector = ResultsCollector()

    yield collector

    # Auto-save on teardown
    if results["test_name"]:
        collector.save()


# ============================================================================
# MOCK SERVICE FIXTURES (for isolated testing)
# ============================================================================

@pytest.fixture(scope="function")
def mock_api_server():
    """
    Create mock API server for isolated testing.

    This is useful for testing load test scripts without a real backend.
    """
    # This would typically start a mock server using Flask or FastAPI
    # For now, we'll skip implementation as it requires a running server
    pytest.skip("Mock API server not implemented - test against real server")


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def wait_time():
    """
    Configurable wait time between operations.

    Returns:
        Float representing seconds to wait
    """
    return 1.0


@pytest.fixture(scope="session")
def test_metadata():
    """
    Test metadata and environment info.

    Returns:
        Dictionary with test metadata
    """
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
        "test_suite": "GL-VCCI Load Testing Suite",
        "version": "1.0.0"
    }
