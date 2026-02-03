# -*- coding: utf-8 -*-
"""
Smoke Test Fixtures and Configuration
=====================================

Shared fixtures and configuration for release smoke tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# ==============================================================================
# Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest for smoke tests."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test"
    )
    config.addinivalue_line(
        "markers", "cli: mark test as a CLI test"
    )
    config.addinivalue_line(
        "markers", "imports: mark test as an import test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Add smoke marker to all tests in this directory."""
    for item in items:
        # Add smoke marker to all tests
        item.add_marker(pytest.mark.smoke)

        # Add specific markers based on test class names
        if "CLI" in item.nodeid:
            item.add_marker(pytest.mark.cli)
        if "Import" in item.nodeid:
            item.add_marker(pytest.mark.imports)
        if "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def expected_version() -> str:
    """Get the expected version for testing."""
    # Check environment variable first
    version = os.environ.get("GL_EXPECTED_VERSION")
    if version:
        return version.lstrip("v")

    # Try to read from pyproject.toml
    project_root = Path(__file__).parent.parent.parent
    pyproject = project_root / "pyproject.toml"

    if pyproject.exists():
        content = pyproject.read_text()
        import re
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)

    # Fallback
    return "0.3.0"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def greenlang_module():
    """Import and return the greenlang module."""
    import greenlang
    return greenlang


@pytest.fixture(scope="function")
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory(prefix="gl-smoke-") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def temp_pack_directory(temp_directory: Path) -> Path:
    """Create a temporary pack directory structure."""
    pack_dir = temp_directory / "test-pack"
    pack_dir.mkdir()

    # Create basic pack structure
    (pack_dir / "agents").mkdir()
    (pack_dir / "pipelines").mkdir()
    (pack_dir / "datasets").mkdir()

    # Create pack manifest
    pack_yaml = pack_dir / "pack.yaml"
    pack_yaml.write_text("""
name: smoke-test-pack
version: 1.0.0
kind: pack
license: MIT
description: "Smoke test pack for release validation"
contents:
  agents: []
  pipelines: []
  datasets: []
""")

    return pack_dir


@pytest.fixture(scope="session")
def is_ci() -> bool:
    """Check if running in CI environment."""
    return any([
        os.environ.get("CI"),
        os.environ.get("GITHUB_ACTIONS"),
        os.environ.get("GITLAB_CI"),
        os.environ.get("JENKINS_URL"),
    ])


@pytest.fixture(scope="session")
def strict_mode() -> bool:
    """Check if strict mode is enabled."""
    return os.environ.get("GL_SMOKE_STRICT", "0") == "1"


# ==============================================================================
# Helper Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def python_info() -> dict:
    """Get Python environment information."""
    return {
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "implementation": sys.implementation.name,
        "platform": sys.platform,
        "executable": sys.executable,
    }


@pytest.fixture(scope="function")
def isolated_env(monkeypatch) -> Generator[None, None, None]:
    """Create an isolated environment for testing."""
    # Save original environment
    original_env = os.environ.copy()

    # Set test environment variables
    monkeypatch.setenv("GL_TEST_MODE", "1")
    monkeypatch.setenv("GL_SIGNING_MODE", "ephemeral")

    yield

    # Environment is automatically restored by monkeypatch


# ==============================================================================
# Skip Conditions
# ==============================================================================

def pytest_runtest_setup(item):
    """Skip tests based on conditions."""
    # Skip Windows-specific tests on non-Windows
    if "windows" in item.keywords and sys.platform != "win32":
        pytest.skip("Windows-only test")

    # Skip Unix-specific tests on Windows
    if "unix" in item.keywords and sys.platform == "win32":
        pytest.skip("Unix-only test")


# ==============================================================================
# Reporting
# ==============================================================================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Enhanced test reporting for smoke tests."""
    outcome = yield
    report = outcome.get_result()

    # Add extra info for failures
    if report.when == "call" and report.failed:
        # Add environment info to failures
        report.sections.append((
            "Environment Info",
            f"Python: {sys.version}\n"
            f"Platform: {sys.platform}\n"
            f"Expected Version: {os.environ.get('GL_EXPECTED_VERSION', 'not set')}\n"
        ))
