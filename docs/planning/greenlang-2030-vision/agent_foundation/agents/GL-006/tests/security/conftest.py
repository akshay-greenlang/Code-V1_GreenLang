# -*- coding: utf-8 -*-
"""
Security Test Configuration for GL-006 HEATRECLAIM.

This module provides shared fixtures and configuration for security testing.
"""

import pytest
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_configure(config):
    """Configure pytest markers for security tests."""
    config.addinivalue_line(
        "markers", "security: mark test as a security validation test"
    )


@pytest.fixture(scope="session")
def security_config():
    """Define security configuration for GL-006 tests."""
    return {
        "max_login_attempts": 5,
        "lockout_duration_seconds": 300,
        "session_timeout_seconds": 3600,
        "password_min_length": 12,
        "rate_limit_requests_per_minute": 60,
        "max_string_length": 256,
        "max_temperature_c": 300.0,
        "emergency_shutdown_temp_c": 350.0,
    }


@pytest.fixture(scope="session")
def rbac_permissions():
    """Define RBAC permission matrix."""
    return {
        "operator": ["read_data", "view_dashboard"],
        "engineer": ["read_data", "view_dashboard", "modify_setpoints", "run_optimization"],
        "admin": ["read_data", "view_dashboard", "modify_setpoints", "run_optimization",
                  "modify_config", "manage_users", "view_audit"],
        "auditor": ["read_data", "view_dashboard", "view_audit"],
    }


@pytest.fixture(scope="session")
def owasp_test_payloads():
    """OWASP Top 10 test payloads."""
    return {
        "A03_injection": [
            "'; DROP TABLE--",
            "1 OR 1=1",
            "; rm -rf /",
            "../../../etc/passwd",
        ],
        "A07_xss": [
            "<script>alert(1)</script>",
            "javascript:alert(1)",
        ],
    }
