# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Security Scanning integration tests.

Provides fixtures for:
    - Temporary test directories
    - Scanner availability checks
    - Mock targets
    - Authentication helpers
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest


# ============================================================================
# Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_project_dir() -> Generator[str, None, None]:
    """Create a temporary project directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="security_test_")

    # Create a sample Python file with potential issues
    python_file = Path(temp_dir) / "app.py"
    python_file.write_text('''
import os
import subprocess

# Potential security issues for testing
password = "hardcoded_password_123"  # B105
api_key = "sk_live_abcdefghijklmnop"

def unsafe_exec(cmd):
    subprocess.call(cmd, shell=True)  # B602

def sql_query(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input  # SQL injection
    return query
''')

    # Create requirements.txt with vulnerable packages
    requirements = Path(temp_dir) / "requirements.txt"
    requirements.write_text("""
requests==2.25.0
django==2.2.0
pyyaml==5.3.1
""")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_terraform_dir() -> Generator[str, None, None]:
    """Create a temporary Terraform directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="tf_test_")

    tf_file = Path(temp_dir) / "main.tf"
    tf_file.write_text('''
resource "aws_s3_bucket" "public_bucket" {
  bucket = "my-public-bucket"
  acl    = "public-read"  # Security issue
}

resource "aws_security_group" "open_sg" {
  name = "open-security-group"

  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Security issue
  }
}
''')

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_container_dir() -> Generator[str, None, None]:
    """Create a temporary directory with Dockerfile for testing."""
    temp_dir = tempfile.mkdtemp(prefix="container_test_")

    dockerfile = Path(temp_dir) / "Dockerfile"
    dockerfile.write_text('''
FROM python:3.9

# Security issues for testing
USER root
RUN pip install requests==2.25.0

COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
''')

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Scanner Availability Fixtures
# ============================================================================


@pytest.fixture
def bandit_available() -> bool:
    """Check if Bandit is available."""
    return shutil.which("bandit") is not None


@pytest.fixture
def trivy_available() -> bool:
    """Check if Trivy is available."""
    return shutil.which("trivy") is not None


@pytest.fixture
def gitleaks_available() -> bool:
    """Check if Gitleaks is available."""
    return shutil.which("gitleaks") is not None


@pytest.fixture
def tfsec_available() -> bool:
    """Check if tfsec is available."""
    return shutil.which("tfsec") is not None


@pytest.fixture
def cosign_available() -> bool:
    """Check if Cosign is available."""
    return shutil.which("cosign") is not None


@pytest.fixture
def zap_available() -> bool:
    """Check if ZAP is available."""
    return shutil.which("zap-cli") is not None or shutil.which("zap.sh") is not None


# ============================================================================
# Target Fixtures
# ============================================================================


@pytest.fixture
def mock_target_url() -> str:
    """Return a mock target URL for DAST testing."""
    return os.environ.get("ZAP_TEST_TARGET", "http://localhost:8080")


@pytest.fixture
def test_image() -> str:
    """Return a small test image for container scanning."""
    return "alpine:3.18"


# ============================================================================
# Authentication Fixtures
# ============================================================================


@pytest.fixture
def test_api_token() -> str:
    """Return a test API token."""
    return os.environ.get("TEST_API_TOKEN", "test_token_for_integration_tests")


@pytest.fixture
def auth_headers(test_api_token: str) -> Dict[str, str]:
    """Return authentication headers for API calls."""
    return {
        "Authorization": f"Bearer {test_api_token}",
        "Content-Type": "application/json",
    }


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test",
    )
    config.addinivalue_line(
        "markers",
        "requires_scanner(name): mark test as requiring a specific scanner",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (may be skipped in quick runs)",
    )
