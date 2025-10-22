"""
Comprehensive tests for GreenLang Provenance Environment Module.

Tests cover:
- get_environment_info() - complete environment snapshot
- get_dependency_versions() - package version tracking
- get_system_info() - OS and hardware details
- compare_environments() - environment diff analysis
- Audit trail functionality for reproducibility
"""

import pytest
import sys
import platform
import os
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from greenlang.provenance.environment import (
    get_environment_info,
    get_python_info,
    get_system_info,
    get_process_info,
    get_greenlang_info,
    get_dependency_versions,
    get_all_installed_packages,
    capture_environment_snapshot,
    compare_environments,
    _get_package_version
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_environment():
    """Mock environment for testing."""
    return {
        "timestamp": "2024-01-01T12:00:00Z",
        "python": {
            "version": "3.9.0",
            "version_info": {
                "major": 3,
                "minor": 9,
                "micro": 0,
                "releaselevel": "final",
                "serial": 0
            }
        },
        "system": {
            "os": "Linux",
            "release": "5.10.0",
            "machine": "x86_64"
        },
        "process": {
            "pid": 12345,
            "cwd": "/home/user"
        },
        "greenlang": {
            "framework_version": "1.0.0"
        }
    }


@pytest.fixture
def sample_dependencies():
    """Sample dependency dictionary."""
    return {
        "pandas": "2.0.3",
        "pydantic": "2.1.1",
        "pytest": "7.4.0"
    }


# ============================================================================
# GET_ENVIRONMENT_INFO TESTS
# ============================================================================

class TestGetEnvironmentInfo:
    """Test suite for get_environment_info() function."""

    def test_get_environment_info_structure(self):
        """Test that environment info has correct structure."""
        env = get_environment_info()

        # Check top-level keys
        assert "timestamp" in env
        assert "python" in env
        assert "system" in env
        assert "process" in env
        assert "greenlang" in env

        # Verify types
        assert isinstance(env, dict)
        assert isinstance(env["python"], dict)
        assert isinstance(env["system"], dict)
        assert isinstance(env["process"], dict)
        assert isinstance(env["greenlang"], dict)

    def test_get_environment_info_timestamp(self):
        """Test timestamp format in environment info."""
        env = get_environment_info()

        # Verify ISO 8601 format
        assert "T" in env["timestamp"]
        assert env["timestamp"].endswith("Z") or "+" in env["timestamp"]

    def test_get_environment_info_complete(self):
        """Test that environment info contains all required fields."""
        env = get_environment_info()

        # Python info
        assert "version" in env["python"]
        assert "version_info" in env["python"]
        assert "implementation" in env["python"]

        # System info
        assert "os" in env["system"]
        assert "machine" in env["system"]

        # Process info
        assert "pid" in env["process"]
        assert "cwd" in env["process"]

    def test_get_environment_info_deterministic_fields(self):
        """Test that certain fields are deterministic."""
        env1 = get_environment_info()
        env2 = get_environment_info()

        # Python version should match
        assert env1["python"]["version"] == env2["python"]["version"]

        # System info should match
        assert env1["system"]["os"] == env2["system"]["os"]
        assert env1["system"]["machine"] == env2["system"]["machine"]


# ============================================================================
# GET_PYTHON_INFO TESTS
# ============================================================================

class TestGetPythonInfo:
    """Test suite for get_python_info() function."""

    def test_get_python_info_structure(self):
        """Test Python info structure."""
        info = get_python_info()

        assert isinstance(info, dict)
        assert "version" in info
        assert "version_info" in info
        assert "implementation" in info
        assert "compiler" in info
        assert "executable" in info
        assert "platform" in info

    def test_get_python_info_version(self):
        """Test Python version information."""
        info = get_python_info()

        # Verify version matches sys.version
        assert info["version"] == sys.version

        # Verify version_info structure
        version_info = info["version_info"]
        assert version_info["major"] == sys.version_info.major
        assert version_info["minor"] == sys.version_info.minor
        assert version_info["micro"] == sys.version_info.micro

    def test_get_python_info_executable(self):
        """Test Python executable path."""
        info = get_python_info()

        assert info["executable"] == sys.executable
        assert info["prefix"] == sys.prefix

    def test_get_python_info_platform(self):
        """Test Python platform information."""
        info = get_python_info()

        assert info["platform"] == sys.platform
        assert info["implementation"] == platform.python_implementation()


# ============================================================================
# GET_SYSTEM_INFO TESTS
# ============================================================================

class TestGetSystemInfo:
    """Test suite for get_system_info() function."""

    def test_get_system_info_structure(self):
        """Test system info structure."""
        info = get_system_info()

        assert isinstance(info, dict)
        assert "os" in info
        assert "release" in info
        assert "version" in info
        assert "machine" in info
        assert "processor" in info
        assert "architecture" in info
        assert "hostname" in info
        assert "platform" in info

    def test_get_system_info_os(self):
        """Test OS information."""
        info = get_system_info()

        # Should match platform module
        assert info["os"] == platform.system()
        assert info["release"] == platform.release()
        assert info["machine"] == platform.machine()

    def test_get_system_info_cpu_count(self):
        """Test CPU count information."""
        info = get_system_info()

        assert "cpu_count" in info
        # Should be a number or "unknown"
        assert isinstance(info["cpu_count"], (int, str))

    def test_get_system_info_architecture(self):
        """Test architecture information."""
        info = get_system_info()

        assert info["architecture"] == platform.architecture()[0]

    def test_get_system_info_hostname(self):
        """Test hostname information."""
        info = get_system_info()

        assert info["hostname"] == platform.node()


# ============================================================================
# GET_PROCESS_INFO TESTS
# ============================================================================

class TestGetProcessInfo:
    """Test suite for get_process_info() function."""

    def test_get_process_info_structure(self):
        """Test process info structure."""
        info = get_process_info()

        assert isinstance(info, dict)
        assert "pid" in info
        assert "cwd" in info
        assert "user" in info
        assert "locale" in info

    def test_get_process_info_pid(self):
        """Test process ID."""
        info = get_process_info()

        assert info["pid"] == os.getpid()
        assert isinstance(info["pid"], int)

    def test_get_process_info_cwd(self):
        """Test current working directory."""
        info = get_process_info()

        assert info["cwd"] == os.getcwd()

    def test_get_process_info_user(self):
        """Test user information."""
        info = get_process_info()

        # Should have a user value (or "unknown")
        assert "user" in info
        assert info["user"] != ""


# ============================================================================
# GET_GREENLANG_INFO TESTS
# ============================================================================

class TestGetGreenlangInfo:
    """Test suite for get_greenlang_info() function."""

    def test_get_greenlang_info_structure(self):
        """Test GreenLang info structure."""
        info = get_greenlang_info()

        assert isinstance(info, dict)
        assert "framework_version" in info
        assert "provenance_version" in info
        assert "environment_vars" in info

    def test_get_greenlang_info_versions(self):
        """Test version information."""
        info = get_greenlang_info()

        # Should have version strings
        assert isinstance(info["framework_version"], str)
        assert isinstance(info["provenance_version"], str)

    def test_get_greenlang_info_env_vars(self):
        """Test environment variable capture."""
        info = get_greenlang_info()

        env_vars = info["environment_vars"]
        assert isinstance(env_vars, dict)
        assert "GREENLANG_HOME" in env_vars
        assert "GREENLANG_CONFIG" in env_vars
        assert "GREENLANG_CACHE_DIR" in env_vars


# ============================================================================
# GET_DEPENDENCY_VERSIONS TESTS
# ============================================================================

class TestGetDependencyVersions:
    """Test suite for get_dependency_versions() function."""

    def test_get_dependency_versions_default(self):
        """Test getting default dependency versions."""
        deps = get_dependency_versions()

        assert isinstance(deps, dict)
        # Should check common packages
        assert len(deps) > 0

    def test_get_dependency_versions_specific_packages(self):
        """Test getting specific package versions."""
        packages = ["pytest"]
        deps = get_dependency_versions(packages)

        assert "pytest" in deps
        # pytest should be installed (we're running tests with it)
        assert deps["pytest"] != "unknown"

    def test_get_dependency_versions_nonexistent_package(self):
        """Test getting version of non-existent package."""
        deps = get_dependency_versions(["nonexistent_package_xyz"])

        assert "nonexistent_package_xyz" in deps
        assert deps["nonexistent_package_xyz"] == "unknown"

    def test_get_dependency_versions_common_packages(self):
        """Test that common GreenLang packages are checked."""
        deps = get_dependency_versions()

        # Should include common data science packages
        expected_packages = ["pandas", "pydantic", "pytest"]
        for pkg in expected_packages:
            assert pkg in deps


# ============================================================================
# GET_ALL_INSTALLED_PACKAGES TESTS
# ============================================================================

class TestGetAllInstalledPackages:
    """Test suite for get_all_installed_packages() function."""

    def test_get_all_installed_packages_structure(self):
        """Test structure of installed packages."""
        packages = get_all_installed_packages()

        assert isinstance(packages, dict)
        # Should have many packages
        assert len(packages) > 0

    def test_get_all_installed_packages_contains_pytest(self):
        """Test that pytest is in installed packages."""
        packages = get_all_installed_packages()

        # pytest should be installed (we're using it)
        assert any("pytest" in pkg.lower() for pkg in packages.keys())


# ============================================================================
# CAPTURE_ENVIRONMENT_SNAPSHOT TESTS
# ============================================================================

class TestCaptureEnvironmentSnapshot:
    """Test suite for capture_environment_snapshot() function."""

    def test_capture_environment_snapshot_structure(self):
        """Test snapshot structure."""
        snapshot = capture_environment_snapshot()

        assert isinstance(snapshot, dict)
        assert "environment" in snapshot
        assert "dependencies" in snapshot
        assert "all_packages" in snapshot
        assert "env_variables" in snapshot
        assert "snapshot_timestamp" in snapshot

    def test_capture_environment_snapshot_complete(self):
        """Test that snapshot is comprehensive."""
        snapshot = capture_environment_snapshot()

        # Environment info
        assert isinstance(snapshot["environment"], dict)
        assert "python" in snapshot["environment"]
        assert "system" in snapshot["environment"]

        # Dependencies
        assert isinstance(snapshot["dependencies"], dict)

        # All packages
        assert isinstance(snapshot["all_packages"], dict)

        # Environment variables
        assert isinstance(snapshot["env_variables"], dict)

    def test_capture_environment_snapshot_timestamp(self):
        """Test snapshot timestamp."""
        snapshot = capture_environment_snapshot()

        assert "T" in snapshot["snapshot_timestamp"]


# ============================================================================
# COMPARE_ENVIRONMENTS TESTS
# ============================================================================

class TestCompareEnvironments:
    """Test suite for compare_environments() function."""

    def test_compare_identical_environments(self):
        """Test comparing identical environments."""
        env = capture_environment_snapshot()
        diff = compare_environments(env, env)

        assert diff["python_version_match"] is True
        assert diff["os_match"] is True
        assert len(diff["dependency_differences"]) == 0
        assert "match" in diff["summary"].lower()

    def test_compare_different_python_versions(self, mock_environment):
        """Test comparing environments with different Python versions."""
        env1 = mock_environment.copy()
        env2 = mock_environment.copy()

        env2["environment"]["python"]["version_info"]["minor"] = 10

        diff = compare_environments(
            {"environment": env1, "dependencies": {}},
            {"environment": env2, "dependencies": {}}
        )

        assert diff["python_version_match"] is False
        assert "Python versions differ" in diff["summary"]

    def test_compare_different_os(self, mock_environment):
        """Test comparing environments with different OS."""
        env1 = mock_environment.copy()
        env2 = mock_environment.copy()

        env2["system"]["os"] = "Windows"

        diff = compare_environments(
            {"environment": env1, "dependencies": {}},
            {"environment": env2, "dependencies": {}}
        )

        assert diff["os_match"] is False

    def test_compare_different_dependencies(self, sample_dependencies):
        """Test comparing environments with different dependencies."""
        deps1 = sample_dependencies.copy()
        deps2 = sample_dependencies.copy()
        deps2["pandas"] = "2.1.0"  # Different version

        env1 = {"environment": {"python": {}, "system": {}}, "dependencies": deps1}
        env2 = {"environment": {"python": {}, "system": {}}, "dependencies": deps2}

        diff = compare_environments(env1, env2)

        assert "pandas" in diff["dependency_differences"]
        assert diff["dependency_differences"]["pandas"]["env1"] == "2.0.3"
        assert diff["dependency_differences"]["pandas"]["env2"] == "2.1.0"

    def test_compare_missing_dependency(self, sample_dependencies):
        """Test comparing when dependency is missing in one environment."""
        deps1 = sample_dependencies.copy()
        deps2 = sample_dependencies.copy()
        del deps2["pandas"]

        env1 = {"environment": {"python": {}, "system": {}}, "dependencies": deps1}
        env2 = {"environment": {"python": {}, "system": {}}, "dependencies": deps2}

        diff = compare_environments(env1, env2)

        assert "pandas" in diff["dependency_differences"]
        assert diff["dependency_differences"]["pandas"]["env2"] == "not installed"

    def test_compare_environments_summary(self):
        """Test summary generation in comparison."""
        env1 = {
            "environment": {
                "python": {"version_info": {"major": 3, "minor": 9, "micro": 0}},
                "system": {"os": "Linux", "release": "5.10"}
            },
            "dependencies": {"pandas": "2.0.0"}
        }

        env2 = {
            "environment": {
                "python": {"version_info": {"major": 3, "minor": 10, "micro": 0}},
                "system": {"os": "Windows", "release": "10"}
            },
            "dependencies": {"pandas": "2.1.0"}
        }

        diff = compare_environments(env1, env2)

        # Summary should mention all differences
        assert "Python versions differ" in diff["summary"]
        assert "Operating systems differ" in diff["summary"]
        assert "package version differences" in diff["summary"]


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

class TestGetPackageVersion:
    """Test suite for _get_package_version() helper function."""

    def test_get_package_version_installed(self):
        """Test getting version of installed package."""
        version = _get_package_version("pytest")

        # pytest should be installed
        assert version != "unknown"
        assert isinstance(version, str)
        assert "." in version  # Should be in format X.Y.Z

    def test_get_package_version_not_installed(self):
        """Test getting version of non-installed package."""
        version = _get_package_version("nonexistent_package_xyz_12345")

        assert version == "unknown"


# ============================================================================
# INTEGRATION TESTS FOR AUDIT TRAIL
# ============================================================================

class TestEnvironmentAuditTrail:
    """Integration tests for environment audit trail functionality."""

    def test_complete_environment_audit_trail(self):
        """Test capturing complete environment for audit trail."""
        # Capture initial snapshot
        snapshot = capture_environment_snapshot()

        # Verify all required fields for audit
        assert "environment" in snapshot
        assert "dependencies" in snapshot
        assert "snapshot_timestamp" in snapshot

        # Verify reproducibility info
        env = snapshot["environment"]
        assert "python" in env
        assert "system" in env

        # Should be able to serialize for storage
        import json
        json_str = json.dumps(snapshot, default=str)
        assert len(json_str) > 0

    def test_environment_comparison_audit_trail(self):
        """Test environment comparison for audit trail."""
        # Capture two snapshots (simulating before/after)
        snapshot1 = capture_environment_snapshot()
        snapshot2 = capture_environment_snapshot()

        # Compare for audit trail
        diff = compare_environments(snapshot1, snapshot2)

        # Should match (same environment)
        assert diff["python_version_match"] is True
        assert diff["os_match"] is True

    def test_reproducibility_audit_record(self):
        """Test creating reproducibility audit record."""
        env_info = get_environment_info()
        deps = get_dependency_versions()

        # Create audit record
        audit_record = {
            "record_type": "environment_snapshot",
            "environment": env_info,
            "dependencies": deps,
            "purpose": "regulatory_compliance",
            "captured_for": "CBAM_reporting"
        }

        # Verify audit record structure
        assert audit_record["record_type"] == "environment_snapshot"
        assert "python" in audit_record["environment"]
        assert len(audit_record["dependencies"]) > 0
