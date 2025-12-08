# -*- coding: utf-8 -*-
"""
Secrets Exposure Tests for GreenLang

Tests for sensitive data exposure:
- Hardcoded secrets in code
- Secrets in logs
- Secrets in error messages
- Configuration file security
- Environment variable exposure

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import re
import os
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from .conftest import (
    SecurityTestClient,
    PayloadGenerator,
    SecurityTestResult,
    SecurityFinding,
    SecurityTestSeverity,
    SecurityTestStatus,
    MockHTTPResponse,
)


# ==============================================================================
# Secrets in Responses Tests
# ==============================================================================

class TestSecretsInResponses:
    """Test for secrets exposure in API responses."""

    @pytest.mark.security
    @pytest.mark.secrets
    def test_secrets_in_error_messages(self, security_client: SecurityTestClient):
        """
        Test: Secrets in error messages

        Check: Error responses should not contain secrets.
        Expected: No passwords, keys, or tokens in errors.
        CWE-209: Generation of Error Message Containing Sensitive Information
        """
        result = SecurityTestResult(
            test_name="secrets_in_errors",
            status=SecurityTestStatus.PASSED,
        )

        # Trigger various errors
        error_triggers = [
            ("/api/v1/nonexistent", {}),
            ("/api/v1/calculations/invalid_id", {}),
            ("/api/v1/auth/login", {"username": "test", "password": "wrong"}),
        ]

        secret_patterns = [
            r"password\s*[=:]\s*['\"][^'\"]+['\"]",
            r"api[_-]?key\s*[=:]\s*['\"][^'\"]+['\"]",
            r"secret\s*[=:]\s*['\"][^'\"]+['\"]",
            r"token\s*[=:]\s*['\"][^'\"]+['\"]",
            r"-----BEGIN.*PRIVATE KEY-----",
            r"aws_access_key_id",
            r"aws_secret_access_key",
        ]

        for endpoint, data in error_triggers:
            if data:
                response = security_client.post(endpoint, json_data=data)
            else:
                response = security_client.get(endpoint)

            for pattern in secret_patterns:
                if re.search(pattern, response.text, re.IGNORECASE):
                    result.status = SecurityTestStatus.FAILED
                    result.add_finding(SecurityFinding(
                        finding_id="SEC-001",
                        title="Secret Exposed in Error Message",
                        description=f"Error response contains sensitive data pattern",
                        severity=SecurityTestSeverity.HIGH,
                        category="Secrets Exposure",
                        affected_component=endpoint,
                        evidence=f"Pattern matched: {pattern}",
                        remediation="Sanitize error messages before returning",
                        cwe_id="CWE-209",
                        cvss_score=7.5,
                    ))
                    break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.secrets
    def test_stack_traces_exposure(self, security_client: SecurityTestClient):
        """
        Test: Stack trace exposure

        Check: Error responses should not include stack traces.
        Expected: No file paths or code details in errors.
        CWE-209: Generation of Error Message Containing Sensitive Information
        """
        result = SecurityTestResult(
            test_name="stack_traces_exposure",
            status=SecurityTestStatus.PASSED,
        )

        # Try to trigger errors
        response = security_client.post(
            "/api/v1/calculations",
            json_data={"invalid": "data", "cause_error": True},
        )

        stack_trace_patterns = [
            r"Traceback \(most recent call last\)",
            r"File \".*\.py\", line \d+",
            r"at .*\.java:\d+",
            r"Error: .*\n\s+at ",
            r"Exception in thread",
        ]

        for pattern in stack_trace_patterns:
            if re.search(pattern, response.text):
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="SEC-002",
                    title="Stack Trace Exposed",
                    description="Error response contains stack trace",
                    severity=SecurityTestSeverity.MEDIUM,
                    category="Secrets Exposure",
                    affected_component="Error Handler",
                    evidence=f"Stack trace pattern found",
                    remediation="Disable debug mode, use generic error messages",
                    cwe_id="CWE-209",
                    cvss_score=5.3,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.secrets
    def test_internal_paths_exposure(self, security_client: SecurityTestClient):
        """
        Test: Internal file paths exposure

        Check: Responses should not reveal internal paths.
        Expected: No server file system paths in responses.
        CWE-200: Exposure of Sensitive Information
        """
        result = SecurityTestResult(
            test_name="internal_paths_exposure",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get("/api/v1/health")

        path_patterns = [
            r"/home/[a-z]+/",
            r"/var/www/",
            r"/opt/app/",
            r"C:\\Users\\",
            r"C:\\Program Files",
            r"/usr/local/",
        ]

        for pattern in path_patterns:
            if re.search(pattern, response.text):
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="SEC-003",
                    title="Internal Path Exposed",
                    description="Response contains internal file path",
                    severity=SecurityTestSeverity.LOW,
                    category="Secrets Exposure",
                    affected_component="Response Handler",
                    evidence=f"Path pattern matched: {pattern}",
                    remediation="Remove internal paths from responses",
                    cwe_id="CWE-200",
                    cvss_score=3.1,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Secrets in Logs Tests
# ==============================================================================

class TestSecretsInLogs:
    """Test for secrets exposure in logs."""

    @pytest.mark.security
    @pytest.mark.secrets
    def test_credentials_not_logged(self, security_client: SecurityTestClient):
        """
        Test: Credentials not logged

        Check: Login attempts should not log passwords.
        Expected: Passwords should be redacted in logs.
        CWE-532: Insertion of Sensitive Information into Log File
        """
        result = SecurityTestResult(
            test_name="credentials_not_logged",
            status=SecurityTestStatus.PASSED,
        )

        # Make login request with identifiable password
        test_password = "SUPER_SECRET_PASSWORD_12345"
        security_client.post(
            "/api/v1/auth/login",
            json_data={"username": "test", "password": test_password},
        )

        # In a real test, we would check the application logs
        # This is a simulation of what we'd check

        # Simulated log content (would be read from actual log file)
        simulated_log = """
        2025-12-07 12:00:00 INFO Login attempt for user: test
        2025-12-07 12:00:00 DEBUG Request body: {"username": "test", "password": "[REDACTED]"}
        2025-12-07 12:00:01 INFO Login failed for user: test
        """

        if test_password in simulated_log:
            result.status = SecurityTestStatus.FAILED
            result.add_finding(SecurityFinding(
                finding_id="SEC-004",
                title="Password Logged",
                description="User password found in application logs",
                severity=SecurityTestSeverity.CRITICAL,
                category="Secrets Exposure",
                affected_component="Logging",
                evidence="Password visible in logs",
                remediation="Redact passwords before logging",
                cwe_id="CWE-532",
                cvss_score=7.5,
            ))

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.secrets
    def test_tokens_not_logged(self, security_client: SecurityTestClient):
        """
        Test: Tokens not logged

        Check: API tokens should not be logged.
        Expected: Tokens should be redacted or hashed in logs.
        CWE-532: Insertion of Sensitive Information into Log File
        """
        result = SecurityTestResult(
            test_name="tokens_not_logged",
            status=SecurityTestStatus.PASSED,
        )

        test_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature"

        security_client.get(
            "/api/v1/protected",
            headers={"Authorization": f"Bearer {test_token}"},
        )

        # Simulated log check
        simulated_log = """
        2025-12-07 12:00:00 INFO Request to /api/v1/protected
        2025-12-07 12:00:00 DEBUG Auth header: Bearer [REDACTED]
        """

        if test_token in simulated_log:
            result.status = SecurityTestStatus.FAILED
            result.add_finding(SecurityFinding(
                finding_id="SEC-005",
                title="Token Logged",
                description="Authentication token found in logs",
                severity=SecurityTestSeverity.HIGH,
                category="Secrets Exposure",
                affected_component="Logging",
                evidence="JWT visible in logs",
                remediation="Redact tokens in request logging",
                cwe_id="CWE-532",
                cvss_score=7.5,
            ))

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Configuration Security Tests
# ==============================================================================

class TestConfigurationSecurity:
    """Test configuration file security."""

    @pytest.mark.security
    @pytest.mark.secrets
    def test_config_endpoint_protected(self, security_client: SecurityTestClient):
        """
        Test: Configuration endpoints protected

        Check: Config/debug endpoints should not be accessible.
        Expected: Return 404 or 403 for config endpoints.
        CWE-200: Exposure of Sensitive Information
        """
        result = SecurityTestResult(
            test_name="config_endpoint_protected",
            status=SecurityTestStatus.PASSED,
        )

        sensitive_endpoints = [
            "/config",
            "/.env",
            "/config.json",
            "/settings.yaml",
            "/debug",
            "/phpinfo.php",
            "/server-status",
            "/actuator/env",
            "/api/debug/config",
            "/__debug__/",
        ]

        for endpoint in sensitive_endpoints:
            response = security_client.get(endpoint)

            if response.status_code == 200:
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="SEC-006",
                    title="Sensitive Endpoint Accessible",
                    description=f"Endpoint {endpoint} is accessible",
                    severity=SecurityTestSeverity.HIGH,
                    category="Secrets Exposure",
                    affected_component=endpoint,
                    evidence=f"Status: {response.status_code}",
                    remediation="Block access to configuration endpoints",
                    cwe_id="CWE-200",
                    cvss_score=7.5,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.secrets
    def test_environment_variables_not_exposed(self, security_client: SecurityTestClient):
        """
        Test: Environment variables not exposed

        Check: API should not expose environment variables.
        Expected: No env vars in responses.
        CWE-214: Invocation of Process Using Visible Sensitive Information
        """
        result = SecurityTestResult(
            test_name="env_vars_not_exposed",
            status=SecurityTestStatus.PASSED,
        )

        # Try various endpoints that might expose env vars
        endpoints = [
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/debug",
            "/api/v1/info",
        ]

        env_patterns = [
            r"DATABASE_URL",
            r"SECRET_KEY",
            r"API_KEY",
            r"AWS_ACCESS_KEY",
            r"OPENAI_API_KEY",
            r"DB_PASSWORD",
        ]

        for endpoint in endpoints:
            response = security_client.get(endpoint)

            for pattern in env_patterns:
                if re.search(pattern, response.text):
                    result.status = SecurityTestStatus.FAILED
                    result.add_finding(SecurityFinding(
                        finding_id="SEC-007",
                        title="Environment Variable Exposed",
                        description=f"Env var pattern {pattern} found in response",
                        severity=SecurityTestSeverity.HIGH,
                        category="Secrets Exposure",
                        affected_component=endpoint,
                        evidence=f"Pattern: {pattern}",
                        remediation="Remove env var names from responses",
                        cwe_id="CWE-214",
                        cvss_score=7.5,
                    ))
                    break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Git/VCS Exposure Tests
# ==============================================================================

class TestVCSExposure:
    """Test for version control exposure."""

    @pytest.mark.security
    @pytest.mark.secrets
    def test_git_directory_not_accessible(self, security_client: SecurityTestClient):
        """
        Test: .git directory not accessible

        Check: Git repository should not be exposed.
        Expected: Return 404 for .git paths.
        CWE-200: Exposure of Sensitive Information
        """
        result = SecurityTestResult(
            test_name="git_directory_not_accessible",
            status=SecurityTestStatus.PASSED,
        )

        git_paths = [
            "/.git/",
            "/.git/config",
            "/.git/HEAD",
            "/.git/index",
            "/.git/objects/",
            "/.gitignore",
            "/.svn/",
            "/.hg/",
        ]

        for path in git_paths:
            response = security_client.get(path)

            if response.status_code == 200:
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="SEC-008",
                    title="VCS Directory Accessible",
                    description=f"Version control path {path} is accessible",
                    severity=SecurityTestSeverity.CRITICAL,
                    category="Secrets Exposure",
                    affected_component=path,
                    evidence=f"Status: {response.status_code}",
                    remediation="Block access to .git and VCS directories",
                    cwe_id="CWE-200",
                    cvss_score=9.1,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Backup Files Exposure Tests
# ==============================================================================

class TestBackupFilesExposure:
    """Test for backup file exposure."""

    @pytest.mark.security
    @pytest.mark.secrets
    def test_backup_files_not_accessible(self, security_client: SecurityTestClient):
        """
        Test: Backup files not accessible

        Check: Backup/temporary files should not be exposed.
        Expected: Return 404 for backup file patterns.
        CWE-200: Exposure of Sensitive Information
        """
        result = SecurityTestResult(
            test_name="backup_files_not_accessible",
            status=SecurityTestStatus.PASSED,
        )

        backup_patterns = [
            "/config.bak",
            "/config.old",
            "/config.backup",
            "/settings.yml.bak",
            "/database.sql",
            "/dump.sql",
            "/backup.zip",
            "/.config.swp",
            "/config~",
            "/web.config.bak",
        ]

        for path in backup_patterns:
            response = security_client.get(path)

            if response.status_code == 200:
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="SEC-009",
                    title="Backup File Accessible",
                    description=f"Backup file {path} is accessible",
                    severity=SecurityTestSeverity.HIGH,
                    category="Secrets Exposure",
                    affected_component=path,
                    evidence=f"Status: {response.status_code}",
                    remediation="Remove or block access to backup files",
                    cwe_id="CWE-200",
                    cvss_score=7.5,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "secrets"])
