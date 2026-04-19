# -*- coding: utf-8 -*-
"""
Injection Attack Tests for GreenLang

Tests for injection vulnerabilities:
- SQL injection
- Command injection
- LDAP injection
- NoSQL injection
- Template injection

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import json
import re
from typing import Dict, Any, List
from datetime import datetime

from .conftest import (
    SecurityTestClient,
    PayloadGenerator,
    SecurityValidator,
    SecurityTestResult,
    SecurityFinding,
    SecurityTestSeverity,
    SecurityTestStatus,
    MockHTTPResponse,
)


# ==============================================================================
# SQL Injection Tests
# ==============================================================================

class TestSQLInjection:
    """Test SQL injection vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.injection
    def test_sql_injection_in_query_params(
        self,
        security_client: SecurityTestClient,
        payload_generator: PayloadGenerator,
        security_validator: SecurityValidator,
    ):
        """
        Test: SQL injection in query parameters

        Attack: SQL payloads in URL query parameters.
        Expected: Application should sanitize/escape all inputs.
        CWE-89: SQL Injection
        """
        result = SecurityTestResult(
            test_name="sql_injection_query_params",
            status=SecurityTestStatus.PASSED,
        )

        vulnerable_endpoints = [
            "/api/v1/calculations",
            "/api/v1/users",
            "/api/v1/reports",
        ]

        for endpoint in vulnerable_endpoints:
            for payload in payload_generator.sql_injection_payloads()[:5]:
                response = security_client.get(
                    endpoint,
                    params={"id": payload, "search": payload},
                )

                if security_validator.check_sql_injection_vulnerability(response):
                    result.status = SecurityTestStatus.FAILED
                    result.add_finding(SecurityFinding(
                        finding_id="INJ-001",
                        title="SQL Injection in Query Parameters",
                        description=f"Endpoint {endpoint} vulnerable to SQL injection",
                        severity=SecurityTestSeverity.CRITICAL,
                        category="Injection",
                        affected_component=endpoint,
                        evidence=f"Payload: {payload[:50]}...",
                        remediation="Use parameterized queries, ORMs, or input validation",
                        cwe_id="CWE-89",
                        cvss_score=9.8,
                    ))
                    break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.injection
    def test_sql_injection_in_json_body(
        self,
        security_client: SecurityTestClient,
        payload_generator: PayloadGenerator,
    ):
        """
        Test: SQL injection in JSON request body

        Attack: SQL payloads in JSON field values.
        Expected: All JSON inputs should be sanitized.
        CWE-89: SQL Injection
        """
        result = SecurityTestResult(
            test_name="sql_injection_json_body",
            status=SecurityTestStatus.PASSED,
        )

        for payload in payload_generator.sql_injection_payloads()[:5]:
            # Test various JSON field names
            test_payloads = [
                {"fuel_type": payload},
                {"calculation_id": payload},
                {"filter": {"name": payload}},
                {"query": payload},
            ]

            for json_payload in test_payloads:
                response = security_client.post(
                    "/api/v1/calculations",
                    json_data=json_payload,
                )

                # Check for SQL error in response
                error_indicators = ["sql", "syntax", "query", "mysql", "postgres"]
                response_lower = response.text.lower()

                if any(ind in response_lower for ind in error_indicators):
                    result.status = SecurityTestStatus.FAILED
                    result.add_finding(SecurityFinding(
                        finding_id="INJ-002",
                        title="SQL Injection in JSON Body",
                        description="JSON payload allows SQL injection",
                        severity=SecurityTestSeverity.CRITICAL,
                        category="Injection",
                        affected_component="/api/v1/calculations",
                        evidence=f"Payload: {json.dumps(json_payload)[:50]}...",
                        remediation="Validate and sanitize all JSON input fields",
                        cwe_id="CWE-89",
                        cvss_score=9.8,
                    ))
                    break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.injection
    def test_blind_sql_injection_timing(self, security_client: SecurityTestClient):
        """
        Test: Blind SQL injection (time-based)

        Attack: SQL SLEEP/WAITFOR commands to detect blind injection.
        Expected: Response time should not vary based on payload.
        CWE-89: SQL Injection
        """
        result = SecurityTestResult(
            test_name="blind_sql_injection_timing",
            status=SecurityTestStatus.PASSED,
        )

        timing_payloads = [
            "1' AND SLEEP(5)--",
            "1'; WAITFOR DELAY '0:0:5'--",
            "1 OR (SELECT SLEEP(5))--",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
        ]

        import time

        baseline_time = 0.5  # Expected baseline response time

        for payload in timing_payloads:
            start = time.time()
            response = security_client.get(
                "/api/v1/calculations",
                params={"id": payload},
            )
            elapsed = time.time() - start

            # If response takes significantly longer, might indicate blind SQL injection
            if elapsed > baseline_time + 4:  # 5 second delay
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="INJ-003",
                    title="Blind SQL Injection (Time-Based)",
                    description="Application may be vulnerable to time-based blind SQL injection",
                    severity=SecurityTestSeverity.CRITICAL,
                    category="Injection",
                    affected_component="/api/v1/calculations",
                    evidence=f"Response delayed by {elapsed:.2f}s with payload",
                    remediation="Use parameterized queries and input validation",
                    cwe_id="CWE-89",
                    cvss_score=9.8,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Command Injection Tests
# ==============================================================================

class TestCommandInjection:
    """Test command injection vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.injection
    def test_command_injection_in_filename(
        self,
        security_client: SecurityTestClient,
        payload_generator: PayloadGenerator,
        security_validator: SecurityValidator,
    ):
        """
        Test: Command injection in filename parameters

        Attack: Shell commands in filename/path parameters.
        Expected: Filenames should be sanitized.
        CWE-78: OS Command Injection
        """
        result = SecurityTestResult(
            test_name="command_injection_filename",
            status=SecurityTestStatus.PASSED,
        )

        for payload in payload_generator.command_injection_payloads()[:5]:
            response = security_client.post(
                "/api/v1/reports/export",
                json_data={"filename": payload, "format": "csv"},
            )

            if security_validator.check_command_injection_vulnerability(response):
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="INJ-004",
                    title="Command Injection in Filename",
                    description="Filename parameter allows command injection",
                    severity=SecurityTestSeverity.CRITICAL,
                    category="Injection",
                    affected_component="/api/v1/reports/export",
                    evidence=f"Payload: {payload}",
                    remediation="Validate filenames against whitelist, escape shell chars",
                    cwe_id="CWE-78",
                    cvss_score=9.8,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.injection
    def test_command_injection_in_system_calls(
        self,
        security_client: SecurityTestClient,
        payload_generator: PayloadGenerator,
    ):
        """
        Test: Command injection in system utility parameters

        Attack: Shell commands in parameters passed to system utilities.
        Expected: All system call parameters should be escaped.
        CWE-78: OS Command Injection
        """
        result = SecurityTestResult(
            test_name="command_injection_system_calls",
            status=SecurityTestStatus.PASSED,
        )

        # Endpoints that might use system calls
        system_endpoints = [
            ("/api/v1/tools/convert", {"source": "input.pdf", "target": "; id"}),
            ("/api/v1/tools/ping", {"host": "google.com; cat /etc/passwd"}),
            ("/api/v1/tools/lookup", {"domain": "`whoami`"}),
        ]

        for endpoint, params in system_endpoints:
            response = security_client.post(endpoint, json_data=params)

            # Check for command output in response
            cmd_indicators = ["uid=", "root", "bin", "daemon", "www-data"]
            if any(ind in response.text for ind in cmd_indicators):
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="INJ-005",
                    title="Command Injection in System Call",
                    description=f"Endpoint {endpoint} vulnerable to command injection",
                    severity=SecurityTestSeverity.CRITICAL,
                    category="Injection",
                    affected_component=endpoint,
                    evidence=f"Command output detected in response",
                    remediation="Avoid system calls, use libraries instead",
                    cwe_id="CWE-78",
                    cvss_score=9.8,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# NoSQL Injection Tests
# ==============================================================================

class TestNoSQLInjection:
    """Test NoSQL injection vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.injection
    def test_mongodb_injection(self, security_client: SecurityTestClient):
        """
        Test: MongoDB injection

        Attack: NoSQL injection payloads in JSON queries.
        Expected: MongoDB operators should be sanitized.
        CWE-943: NoSQL Injection
        """
        result = SecurityTestResult(
            test_name="mongodb_injection",
            status=SecurityTestStatus.PASSED,
        )

        nosql_payloads = [
            {"username": {"$ne": ""}, "password": {"$ne": ""}},
            {"username": {"$gt": ""}, "password": {"$gt": ""}},
            {"username": "admin", "password": {"$regex": ".*"}},
            {"$where": "this.password.length > 0"},
            {"username": {"$in": ["admin", "root"]}},
        ]

        for payload in nosql_payloads:
            response = security_client.post(
                "/api/v1/auth/login",
                json_data=payload,
            )

            # If login succeeds with NoSQL payload, it's vulnerable
            if response.status_code == 200 and "token" in response.text:
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="INJ-006",
                    title="NoSQL Injection (MongoDB)",
                    description="Login endpoint vulnerable to MongoDB injection",
                    severity=SecurityTestSeverity.CRITICAL,
                    category="Injection",
                    affected_component="/api/v1/auth/login",
                    evidence=f"Payload: {json.dumps(payload)[:50]}...",
                    remediation="Validate JSON structure, reject MongoDB operators",
                    cwe_id="CWE-943",
                    cvss_score=9.8,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Template Injection Tests
# ==============================================================================

class TestTemplateInjection:
    """Test server-side template injection vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.injection
    def test_ssti_jinja2(self, security_client: SecurityTestClient):
        """
        Test: Server-Side Template Injection (Jinja2)

        Attack: Jinja2 template expressions in user input.
        Expected: Template expressions should not be evaluated.
        CWE-94: Improper Control of Generation of Code
        """
        result = SecurityTestResult(
            test_name="ssti_jinja2",
            status=SecurityTestStatus.PASSED,
        )

        ssti_payloads = [
            "{{7*7}}",
            "{{config}}",
            "{{request.environ}}",
            "{{''.__class__.__mro__[2].__subclasses__()}}",
            "${7*7}",
            "#{7*7}",
            "*{7*7}",
        ]

        for payload in ssti_payloads:
            response = security_client.post(
                "/api/v1/reports/generate",
                json_data={"title": payload, "content": payload},
            )

            # Check if template was evaluated
            if "49" in response.text:  # 7*7 = 49
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="INJ-007",
                    title="Server-Side Template Injection",
                    description="Application vulnerable to SSTI",
                    severity=SecurityTestSeverity.CRITICAL,
                    category="Injection",
                    affected_component="/api/v1/reports/generate",
                    evidence=f"Template evaluated: {payload} -> 49",
                    remediation="Use sandboxed templates, escape user input",
                    cwe_id="CWE-94",
                    cvss_score=9.8,
                ))
                break

            # Check for config/environment leakage
            if "config" in response.text.lower() or "environ" in response.text.lower():
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="INJ-008",
                    title="Template Injection - Config Exposure",
                    description="Template injection exposes configuration",
                    severity=SecurityTestSeverity.CRITICAL,
                    category="Injection",
                    affected_component="/api/v1/reports/generate",
                    evidence="Configuration data leaked via template",
                    remediation="Sanitize all template inputs",
                    cwe_id="CWE-94",
                    cvss_score=9.1,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# LDAP Injection Tests
# ==============================================================================

class TestLDAPInjection:
    """Test LDAP injection vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.injection
    def test_ldap_injection_in_search(self, security_client: SecurityTestClient):
        """
        Test: LDAP injection in search parameters

        Attack: LDAP filter injection characters.
        Expected: LDAP special characters should be escaped.
        CWE-90: LDAP Injection
        """
        result = SecurityTestResult(
            test_name="ldap_injection_search",
            status=SecurityTestStatus.PASSED,
        )

        ldap_payloads = [
            "*",
            "*)(&",
            "*)(uid=*))(|(uid=*",
            "admin)(&)",
            "admin)(|(password=*))",
            "x)(|(objectClass=*)",
        ]

        for payload in ldap_payloads:
            response = security_client.get(
                "/api/v1/users/search",
                params={"username": payload},
            )

            # Check if query returned more data than expected
            try:
                data = response.json()
                if isinstance(data, list) and len(data) > 10:
                    result.status = SecurityTestStatus.FAILED
                    result.add_finding(SecurityFinding(
                        finding_id="INJ-009",
                        title="LDAP Injection",
                        description="User search vulnerable to LDAP injection",
                        severity=SecurityTestSeverity.HIGH,
                        category="Injection",
                        affected_component="/api/v1/users/search",
                        evidence=f"Payload: {payload}, returned {len(data)} users",
                        remediation="Escape LDAP special characters in queries",
                        cwe_id="CWE-90",
                        cvss_score=8.6,
                    ))
                    break
            except (json.JSONDecodeError, KeyError):
                pass

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Header Injection Tests
# ==============================================================================

class TestHeaderInjection:
    """Test HTTP header injection vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.injection
    def test_http_header_injection(self, security_client: SecurityTestClient):
        """
        Test: HTTP Header Injection

        Attack: CRLF injection in header values.
        Expected: Header values should be sanitized.
        CWE-113: Improper Neutralization of CRLF Sequences
        """
        result = SecurityTestResult(
            test_name="http_header_injection",
            status=SecurityTestStatus.PASSED,
        )

        crlf_payloads = [
            "value\r\nX-Injected: header",
            "value%0d%0aX-Injected: header",
            "value\nSet-Cookie: malicious=true",
            "value%0aSet-Cookie: malicious=true",
        ]

        for payload in crlf_payloads:
            response = security_client.get(
                "/api/v1/redirect",
                params={"url": payload},
            )

            # Check if injected header appears in response
            if "X-Injected" in str(response.headers) or "malicious" in str(response.headers):
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="INJ-010",
                    title="HTTP Header Injection (CRLF)",
                    description="Application vulnerable to CRLF injection",
                    severity=SecurityTestSeverity.HIGH,
                    category="Injection",
                    affected_component="/api/v1/redirect",
                    evidence="Injected header appeared in response",
                    remediation="Sanitize CRLF sequences from header values",
                    cwe_id="CWE-113",
                    cvss_score=7.5,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "injection"])
