# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Security Scanning unit tests.

Provides comprehensive mocks for:
- Security scanners (Bandit, Trivy, Gitleaks, etc.)
- Subprocess execution
- Sample SARIF outputs
- Sample vulnerabilities and findings
- Database connections
- External APIs

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def scanner_config():
    """Create ScannerConfig for testing."""
    try:
        from greenlang.infrastructure.security_scanning.config import (
            ScannerConfig,
            ScannerType,
            Severity,
        )

        return ScannerConfig(
            name="test-scanner",
            scanner_type=ScannerType.SAST,
            enabled=True,
            executable="test-scanner",
            timeout_seconds=60,
            severity_threshold=Severity.LOW,
            exclude_paths=[".git", "venv"],
            exclude_rules=["test-rule-1"],
            output_format="json",
        )
    except ImportError:
        # Return a simple object with the same attributes
        class ConfigStub:
            pass

        config = ConfigStub()
        config.name = "test-scanner"
        config.scanner_type = "sast"
        config.enabled = True
        config.executable = "test-scanner"
        config.timeout_seconds = 60
        config.severity_threshold = "LOW"
        config.exclude_paths = [".git", "venv"]
        config.exclude_rules = ["test-rule-1"]
        config.output_format = "json"
        return config


@pytest.fixture
def orchestrator_config():
    """Create ScanOrchestratorConfig for testing."""
    try:
        from greenlang.infrastructure.security_scanning.config import (
            ScanOrchestratorConfig,
            ScannerType,
            Severity,
        )

        return ScanOrchestratorConfig(
            scan_path="/test/path",
            parallel_scans=2,
            global_timeout_seconds=300,
            fail_on_severity=Severity.HIGH,
            deduplication_enabled=True,
            enable_metrics=False,
            enable_audit=False,
            enabled_scanner_types={ScannerType.SAST, ScannerType.SCA},
        )
    except ImportError:
        class ConfigStub:
            pass

        config = ConfigStub()
        config.scan_path = "/test/path"
        config.parallel_scans = 2
        config.global_timeout_seconds = 300
        config.fail_on_severity = "HIGH"
        config.deduplication_enabled = True
        config.enable_metrics = False
        config.enable_audit = False
        return config


# ============================================================================
# Mock Scanner Fixtures
# ============================================================================


@pytest.fixture
def mock_bandit_scanner():
    """Create mocked BanditScanner."""
    scanner = MagicMock()
    scanner.name = "bandit"
    scanner.scanner_type = "sast"
    scanner.enabled = True

    async def mock_scan(path: str):
        return [
            {
                "issue_severity": "HIGH",
                "issue_confidence": "HIGH",
                "issue_text": "Possible hardcoded password",
                "filename": "test.py",
                "line_number": 42,
                "test_id": "B105",
            }
        ]

    scanner.scan = AsyncMock(side_effect=mock_scan)
    return scanner


@pytest.fixture
def mock_trivy_scanner():
    """Create mocked TrivyScanner."""
    scanner = MagicMock()
    scanner.name = "trivy"
    scanner.scanner_type = "sca"
    scanner.enabled = True

    async def mock_scan(path: str):
        return [
            {
                "VulnerabilityID": "CVE-2024-1234",
                "PkgName": "requests",
                "InstalledVersion": "2.31.0",
                "FixedVersion": "2.32.0",
                "Severity": "HIGH",
                "Title": "Test vulnerability",
                "Description": "A test vulnerability in requests",
            }
        ]

    scanner.scan = AsyncMock(side_effect=mock_scan)
    return scanner


@pytest.fixture
def mock_gitleaks_scanner():
    """Create mocked GitleaksScanner."""
    scanner = MagicMock()
    scanner.name = "gitleaks"
    scanner.scanner_type = "secrets"
    scanner.enabled = True

    async def mock_scan(path: str):
        return [
            {
                "Description": "AWS Access Key",
                "File": "config.py",
                "StartLine": 10,
                "EndLine": 10,
                "Secret": "AKIA***REDACTED***",
                "RuleID": "aws-access-key-id",
            }
        ]

    scanner.scan = AsyncMock(side_effect=mock_scan)
    return scanner


@pytest.fixture
def mock_tfsec_scanner():
    """Create mocked TfsecScanner."""
    scanner = MagicMock()
    scanner.name = "tfsec"
    scanner.scanner_type = "iac"
    scanner.enabled = True

    async def mock_scan(path: str):
        return [
            {
                "rule_id": "AWS002",
                "severity": "MEDIUM",
                "description": "S3 bucket does not have encryption enabled",
                "location": {
                    "filename": "main.tf",
                    "start_line": 15,
                },
            }
        ]

    scanner.scan = AsyncMock(side_effect=mock_scan)
    return scanner


@pytest.fixture
def mock_zap_scanner():
    """Create mocked ZAPScanner."""
    scanner = MagicMock()
    scanner.name = "zap"
    scanner.scanner_type = "dast"
    scanner.enabled = True

    async def mock_scan(target: str):
        return [
            {
                "alert": "Cross Site Scripting",
                "risk": "High",
                "confidence": "Medium",
                "url": f"{target}/search",
                "method": "GET",
                "param": "q",
                "evidence": "<script>alert(1)</script>",
                "cweid": "79",
                "wascid": "8",
            }
        ]

    scanner.scan = AsyncMock(side_effect=mock_scan)
    return scanner


# ============================================================================
# Mock Subprocess Fixtures
# ============================================================================


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for scanner CLI execution."""
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"results": []})
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def mock_async_subprocess():
    """Mock asyncio subprocess for async scanner execution."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps({"results": []}).encode(), b"")
        )
        mock_exec.return_value = mock_process
        yield mock_exec


# ============================================================================
# Sample SARIF Output Fixtures
# ============================================================================


@pytest.fixture
def sample_sarif_output():
    """Sample SARIF 2.1.0 output for testing."""
    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "TestScanner",
                        "version": "1.0.0",
                        "informationUri": "https://example.com",
                        "rules": [
                            {
                                "id": "TEST001",
                                "name": "TestRule",
                                "shortDescription": {"text": "Test rule"},
                                "fullDescription": {"text": "A test security rule"},
                                "defaultConfiguration": {"level": "error"},
                            }
                        ],
                    }
                },
                "results": [
                    {
                        "ruleId": "TEST001",
                        "level": "error",
                        "message": {"text": "Test finding message"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": "src/test.py",
                                        "uriBaseId": "%SRCROOT%",
                                    },
                                    "region": {
                                        "startLine": 10,
                                        "startColumn": 5,
                                        "endLine": 10,
                                        "endColumn": 20,
                                    },
                                }
                            }
                        ],
                    }
                ],
            }
        ],
    }


@pytest.fixture
def sample_bandit_sarif():
    """Sample Bandit SARIF output."""
    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Bandit",
                        "version": "1.7.7",
                        "informationUri": "https://bandit.readthedocs.io/",
                    }
                },
                "results": [
                    {
                        "ruleId": "B105",
                        "level": "warning",
                        "message": {
                            "text": "Possible hardcoded password: 'password123'"
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "app/config.py"},
                                    "region": {"startLine": 42, "startColumn": 1},
                                }
                            }
                        ],
                    }
                ],
            }
        ],
    }


# ============================================================================
# Sample Vulnerability Fixtures
# ============================================================================


@pytest.fixture
def sample_vulnerability():
    """Sample vulnerability data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "cve_id": "CVE-2024-1234",
        "title": "Test Vulnerability",
        "description": "A test vulnerability for unit testing",
        "severity": "HIGH",
        "cvss_score": 8.5,
        "epss_score": 0.75,
        "package_name": "requests",
        "current_version": "2.31.0",
        "fixed_version": "2.32.0",
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "scanner": "trivy",
        "file_path": "requirements.txt",
        "advisory_url": "https://nvd.nist.gov/vuln/detail/CVE-2024-1234",
    }


@pytest.fixture
def sample_critical_vulnerability():
    """Sample critical vulnerability."""
    return {
        "id": str(uuid.uuid4()),
        "cve_id": "CVE-2024-0001",
        "title": "Critical RCE Vulnerability",
        "description": "Remote code execution vulnerability",
        "severity": "CRITICAL",
        "cvss_score": 9.8,
        "epss_score": 0.95,
        "is_kev": True,
        "is_exploited": True,
        "package_name": "log4j",
        "current_version": "2.14.0",
        "fixed_version": "2.17.0",
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "scanner": "trivy",
    }


@pytest.fixture
def sample_vulnerabilities():
    """List of sample vulnerabilities."""
    return [
        {
            "id": str(uuid.uuid4()),
            "cve_id": "CVE-2024-1001",
            "severity": "CRITICAL",
            "cvss_score": 9.8,
            "package_name": "critical-pkg",
            "scanner": "trivy",
        },
        {
            "id": str(uuid.uuid4()),
            "cve_id": "CVE-2024-1002",
            "severity": "HIGH",
            "cvss_score": 8.1,
            "package_name": "high-pkg",
            "scanner": "snyk",
        },
        {
            "id": str(uuid.uuid4()),
            "cve_id": "CVE-2024-1003",
            "severity": "MEDIUM",
            "cvss_score": 5.5,
            "package_name": "medium-pkg",
            "scanner": "trivy",
        },
        {
            "id": str(uuid.uuid4()),
            "cve_id": "CVE-2024-1004",
            "severity": "LOW",
            "cvss_score": 2.1,
            "package_name": "low-pkg",
            "scanner": "pip-audit",
        },
        {
            "id": str(uuid.uuid4()),
            "cve_id": None,
            "severity": "INFO",
            "cvss_score": 0.0,
            "package_name": "info-pkg",
            "scanner": "trivy",
        },
    ]


# ============================================================================
# Sample Finding Fixtures
# ============================================================================


@pytest.fixture
def sample_sast_finding():
    """Sample SAST finding."""
    return {
        "id": str(uuid.uuid4()),
        "type": "code_vulnerability",
        "scanner": "bandit",
        "rule_id": "B105",
        "severity": "HIGH",
        "confidence": "HIGH",
        "message": "Possible hardcoded password",
        "file_path": "app/config.py",
        "line_number": 42,
        "code_snippet": 'password = "secret123"',
        "discovered_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_secret_finding():
    """Sample secret detection finding."""
    return {
        "id": str(uuid.uuid4()),
        "type": "secret",
        "scanner": "gitleaks",
        "rule_id": "aws-access-key-id",
        "severity": "CRITICAL",
        "message": "AWS Access Key ID detected",
        "file_path": "config/settings.py",
        "line_number": 15,
        "secret_type": "aws_access_key",
        "discovered_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_iac_finding():
    """Sample IaC misconfiguration finding."""
    return {
        "id": str(uuid.uuid4()),
        "type": "iac_misconfiguration",
        "scanner": "tfsec",
        "rule_id": "AWS002",
        "severity": "MEDIUM",
        "message": "S3 bucket does not have encryption enabled",
        "file_path": "terraform/main.tf",
        "line_number": 25,
        "remediation": "Add server_side_encryption_configuration block",
        "discovered_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_pii_finding():
    """Sample PII detection finding."""
    return {
        "id": str(uuid.uuid4()),
        "type": "pii_exposure",
        "scanner": "pii",
        "rule_id": "SSN",
        "severity": "CRITICAL",
        "message": "Social Security Number detected",
        "file_path": "tests/fixtures/data.json",
        "line_number": 100,
        "pii_type": "ssn",
        "confidence": 0.95,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_findings():
    """List of sample findings from multiple scanners."""
    return [
        {
            "id": str(uuid.uuid4()),
            "type": "code_vulnerability",
            "scanner": "bandit",
            "severity": "HIGH",
            "file_path": "app.py",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "dependency_vulnerability",
            "scanner": "trivy",
            "severity": "CRITICAL",
            "cve_id": "CVE-2024-1234",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "secret",
            "scanner": "gitleaks",
            "severity": "CRITICAL",
            "file_path": "config.py",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "iac_misconfiguration",
            "scanner": "tfsec",
            "severity": "MEDIUM",
            "file_path": "main.tf",
        },
    ]


# ============================================================================
# Mock Service Fixtures
# ============================================================================


@pytest.fixture
def mock_vulnerability_service():
    """Mock VulnerabilityService for testing."""
    service = AsyncMock()

    service.get_vulnerabilities = AsyncMock(return_value=[])
    service.get_vulnerability = AsyncMock(return_value=None)
    service.ingest_findings = AsyncMock(return_value=10)
    service.accept_risk = AsyncMock(return_value=True)
    service.mark_remediated = AsyncMock(return_value=True)
    service.get_statistics = AsyncMock(
        return_value={
            "total": 100,
            "open": 50,
            "resolved": 40,
            "accepted": 10,
            "by_severity": {"CRITICAL": 5, "HIGH": 20, "MEDIUM": 30, "LOW": 45},
            "kev_count": 3,
            "sla_breached": 5,
            "avg_risk_score": 6.5,
            "avg_mttr_days": 14.5,
        }
    )

    return service


@pytest.fixture
def mock_orchestrator():
    """Mock ScanOrchestrator for testing."""
    orchestrator = AsyncMock()

    orchestrator.run_scan = AsyncMock(
        return_value={
            "scan_id": str(uuid.uuid4()),
            "status": "completed",
            "findings_count": 10,
            "duration_seconds": 120,
        }
    )
    orchestrator.get_enabled_scanners = MagicMock(return_value=[])
    orchestrator.generate_sarif = MagicMock(return_value={})

    return orchestrator


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def mock_db_connection():
    """Mock database connection for testing."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=MagicMock(rowcount=1))
    conn.fetchone = AsyncMock(return_value=None)
    conn.fetchall = AsyncMock(return_value=[])
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    return conn


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool."""
    pool = MagicMock()

    async def acquire():
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.fetchone = AsyncMock(return_value=None)
        conn.fetchall = AsyncMock(return_value=[])
        return conn

    pool.acquire = acquire
    return pool


# ============================================================================
# HTTP Client Fixtures
# ============================================================================


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient for testing."""
    with patch("httpx.AsyncClient") as mock_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.text = '{"status": "ok"}'

        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.put = AsyncMock(return_value=mock_response)
        mock_client.delete = AsyncMock(return_value=mock_response)

        mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)

        yield mock_client


# ============================================================================
# Authentication Fixtures
# ============================================================================


@pytest.fixture
def auth_headers():
    """Generate authentication headers for API tests."""

    def _generate(
        user_id: str = "user-1",
        tenant_id: str = "t-acme",
        permissions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        permissions = permissions or ["security:read", "security:write"]
        return {
            "Authorization": "Bearer test-jwt-token",
            "X-Tenant-ID": tenant_id,
            "X-User-ID": user_id,
            "X-Permissions": ",".join(permissions),
        }

    return _generate


@pytest.fixture
def admin_auth_headers(auth_headers):
    """Generate admin authentication headers."""
    return auth_headers(
        user_id="admin-1",
        tenant_id="t-platform",
        permissions=[
            "security:admin",
            "security:read",
            "security:write",
            "security:scan",
        ],
    )


# ============================================================================
# FastAPI Test Client Fixtures
# ============================================================================


@pytest.fixture
def test_app():
    """Create a test FastAPI application."""
    try:
        from fastapi import FastAPI

        app = FastAPI()

        # Include security router if available
        try:
            from greenlang.infrastructure.security_scanning.api import security_router

            if security_router:
                app.include_router(security_router, prefix="/api/v1/security")
        except ImportError:
            pass

        return app
    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.fixture
def test_client(test_app):
    """Create test client for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient

        return TestClient(test_app)
    except ImportError:
        pytest.skip("FastAPI not installed")


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "load: mark test as a load test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_scanner: mark test as requiring a real scanner"
    )


@pytest.fixture(autouse=True)
def reset_singleton_instances():
    """Reset any singleton instances between tests."""
    yield
    # Cleanup after each test - reset module-level globals if needed


@pytest.fixture
def temp_scan_path(tmp_path):
    """Create a temporary directory with sample files to scan."""
    # Create Python file with potential issues
    py_file = tmp_path / "app.py"
    py_file.write_text(
        '''
import os
password = "secret123"  # B105 hardcoded password
os.system(input())  # B605 command injection
'''
    )

    # Create requirements file
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("requests==2.31.0\nflask==2.3.0\n")

    # Create Terraform file
    tf_file = tmp_path / "main.tf"
    tf_file.write_text(
        '''
resource "aws_s3_bucket" "test" {
  bucket = "test-bucket"
  # Missing encryption configuration
}
'''
    )

    return tmp_path
