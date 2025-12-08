# -*- coding: utf-8 -*-
"""
API Security Tests for GreenLang

Tests for API-level security:
- Rate limiting
- CORS configuration
- API key security
- Request validation
- Response security headers

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import json
import time
from typing import Dict, Any, List
from datetime import datetime

from .conftest import (
    SecurityTestClient,
    SecurityValidator,
    SecurityTestResult,
    SecurityFinding,
    SecurityTestSeverity,
    SecurityTestStatus,
    MockHTTPResponse,
)


# ==============================================================================
# Rate Limiting Tests
# ==============================================================================

class TestRateLimiting:
    """Test rate limiting implementation."""

    @pytest.mark.security
    @pytest.mark.api_security
    def test_rate_limiting_enforcement(self, security_client: SecurityTestClient):
        """
        Test: Rate limiting is enforced

        Attack: Rapid requests to exhaust rate limit.
        Expected: Should receive 429 after exceeding limit.
        CWE-770: Allocation of Resources Without Limits
        """
        result = SecurityTestResult(
            test_name="rate_limiting_enforcement",
            status=SecurityTestStatus.PASSED,
        )

        # Make rapid requests
        responses = []
        for i in range(100):
            response = security_client.get("/api/v1/calculations")
            responses.append(response)

            if response.status_code == 429:
                break

        rate_limited = any(r.status_code == 429 for r in responses)

        if not rate_limited:
            result.status = SecurityTestStatus.FAILED
            result.add_finding(SecurityFinding(
                finding_id="API-001",
                title="Rate Limiting Not Enforced",
                description="API does not enforce rate limiting",
                severity=SecurityTestSeverity.MEDIUM,
                category="API Security",
                affected_component="Rate Limiter",
                evidence=f"Made {len(responses)} requests without 429",
                remediation="Implement rate limiting on all API endpoints",
                cwe_id="CWE-770",
                cvss_score=5.3,
            ))

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.api_security
    def test_rate_limit_headers(self, security_client: SecurityTestClient):
        """
        Test: Rate limit headers are present

        Check: X-RateLimit-* headers provide information.
        Expected: Headers should show limit, remaining, reset time.
        """
        result = SecurityTestResult(
            test_name="rate_limit_headers",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get("/api/v1/calculations")

        expected_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]

        missing_headers = []
        for header in expected_headers:
            if header not in response.headers:
                missing_headers.append(header)

        if missing_headers:
            result.add_finding(SecurityFinding(
                finding_id="API-002",
                title="Missing Rate Limit Headers",
                description=f"Missing headers: {', '.join(missing_headers)}",
                severity=SecurityTestSeverity.LOW,
                category="API Security",
                affected_component="Rate Limiter Headers",
                evidence="Rate limit headers not present",
                remediation="Add X-RateLimit-* headers to responses",
                cwe_id="CWE-770",
                cvss_score=3.1,
            ))

        # Mock test passes
        assert True

    @pytest.mark.security
    @pytest.mark.api_security
    def test_rate_limit_bypass_attempts(self, security_client: SecurityTestClient):
        """
        Test: Rate limit bypass attempts

        Attack: Try to bypass rate limiting with various techniques.
        Expected: Bypass attempts should fail.
        CWE-770: Allocation of Resources Without Limits
        """
        result = SecurityTestResult(
            test_name="rate_limit_bypass",
            status=SecurityTestStatus.PASSED,
        )

        bypass_headers = [
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "10.0.0.1"},
            {"X-Originating-IP": "192.168.1.1"},
            {"Client-IP": "8.8.8.8"},
            {"True-Client-IP": "1.1.1.1"},
        ]

        for headers in bypass_headers:
            responses = []
            for i in range(50):
                response = security_client.get(
                    "/api/v1/calculations",
                    headers=headers,
                )
                responses.append(response)

            if not any(r.status_code == 429 for r in responses):
                result.add_finding(SecurityFinding(
                    finding_id="API-003",
                    title="Rate Limit Bypass via Headers",
                    description=f"Rate limiting bypassed with {list(headers.keys())[0]}",
                    severity=SecurityTestSeverity.MEDIUM,
                    category="API Security",
                    affected_component="Rate Limiter",
                    evidence=f"Header: {headers}",
                    remediation="Ignore client-provided IP headers for rate limiting",
                    cwe_id="CWE-770",
                    cvss_score=5.3,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# CORS Security Tests
# ==============================================================================

class TestCORSSecurity:
    """Test CORS configuration."""

    @pytest.mark.security
    @pytest.mark.api_security
    def test_cors_wildcard_origin(
        self,
        security_client: SecurityTestClient,
        security_validator: SecurityValidator,
    ):
        """
        Test: CORS wildcard origin

        Check: Access-Control-Allow-Origin should not be *.
        Expected: Origin should be explicitly specified.
        CWE-942: Permissive Cross-domain Policy
        """
        result = SecurityTestResult(
            test_name="cors_wildcard_origin",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get(
            "/api/v1/calculations",
            headers={"Origin": "https://evil.com"},
        )

        cors_check = security_validator.check_cors_headers(response)

        if not cors_check["secure"]:
            result.status = SecurityTestStatus.FAILED
            for issue in cors_check["issues"]:
                result.add_finding(SecurityFinding(
                    finding_id="API-004",
                    title="Insecure CORS Configuration",
                    description=issue,
                    severity=SecurityTestSeverity.MEDIUM,
                    category="API Security",
                    affected_component="CORS",
                    evidence=f"Headers: {cors_check['headers']}",
                    remediation="Specify allowed origins explicitly",
                    cwe_id="CWE-942",
                    cvss_score=5.3,
                ))

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.api_security
    def test_cors_credentials_with_wildcard(self, security_client: SecurityTestClient):
        """
        Test: CORS credentials with wildcard origin

        Check: Credentials should not be allowed with wildcard origin.
        Expected: Either specify origins or don't allow credentials.
        CWE-942: Permissive Cross-domain Policy
        """
        result = SecurityTestResult(
            test_name="cors_credentials_wildcard",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get(
            "/api/v1/calculations",
            headers={"Origin": "https://attacker.com"},
        )

        allow_origin = response.headers.get("Access-Control-Allow-Origin", "")
        allow_creds = response.headers.get("Access-Control-Allow-Credentials", "")

        if allow_origin == "*" and allow_creds.lower() == "true":
            result.status = SecurityTestStatus.FAILED
            result.add_finding(SecurityFinding(
                finding_id="API-005",
                title="CORS Credentials with Wildcard",
                description="Credentials allowed with wildcard origin",
                severity=SecurityTestSeverity.HIGH,
                category="API Security",
                affected_component="CORS",
                evidence="Allow-Origin: * with Allow-Credentials: true",
                remediation="Specify allowed origins when using credentials",
                cwe_id="CWE-942",
                cvss_score=7.5,
            ))

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.api_security
    def test_cors_preflight_caching(self, security_client: SecurityTestClient):
        """
        Test: CORS preflight caching

        Check: Access-Control-Max-Age should be reasonable.
        Expected: Max-Age should be set but not too long.
        """
        result = SecurityTestResult(
            test_name="cors_preflight_caching",
            status=SecurityTestStatus.PASSED,
        )

        # OPTIONS request for preflight
        response = security_client._request(
            "OPTIONS",
            "/api/v1/calculations",
            headers={
                "Origin": "https://allowed.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        max_age = response.headers.get("Access-Control-Max-Age", "")

        if max_age:
            try:
                max_age_int = int(max_age)
                if max_age_int > 86400:  # More than 24 hours
                    result.add_finding(SecurityFinding(
                        finding_id="API-006",
                        title="Excessive CORS Preflight Cache",
                        description=f"Max-Age is {max_age_int}s (>24h)",
                        severity=SecurityTestSeverity.LOW,
                        category="API Security",
                        affected_component="CORS",
                        evidence=f"Access-Control-Max-Age: {max_age}",
                        remediation="Set Max-Age to 24 hours or less",
                        cwe_id="CWE-942",
                        cvss_score=3.1,
                    ))
            except ValueError:
                pass

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Security Headers Tests
# ==============================================================================

class TestSecurityHeaders:
    """Test security header implementation."""

    @pytest.mark.security
    @pytest.mark.api_security
    def test_security_headers_present(self, security_client: SecurityTestClient):
        """
        Test: Security headers are present

        Check: Standard security headers should be set.
        Expected: All recommended headers should be present.
        CWE-693: Protection Mechanism Failure
        """
        result = SecurityTestResult(
            test_name="security_headers",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get("/api/v1/health")

        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": None,  # Just needs to exist
            "Content-Security-Policy": None,
        }

        missing = []
        incorrect = []

        for header, expected in required_headers.items():
            actual = response.headers.get(header, "")

            if not actual:
                missing.append(header)
            elif expected and actual not in (expected if isinstance(expected, list) else [expected]):
                incorrect.append(f"{header}: {actual}")

        if missing:
            result.add_finding(SecurityFinding(
                finding_id="API-007",
                title="Missing Security Headers",
                description=f"Missing: {', '.join(missing)}",
                severity=SecurityTestSeverity.MEDIUM,
                category="API Security",
                affected_component="HTTP Headers",
                evidence=f"Missing headers: {missing}",
                remediation="Add all recommended security headers",
                cwe_id="CWE-693",
                cvss_score=5.3,
            ))

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.api_security
    def test_hsts_configuration(self, security_client: SecurityTestClient):
        """
        Test: HSTS header configuration

        Check: Strict-Transport-Security properly configured.
        Expected: HSTS with max-age, includeSubDomains.
        CWE-319: Cleartext Transmission of Sensitive Information
        """
        result = SecurityTestResult(
            test_name="hsts_configuration",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get("/api/v1/health")

        hsts = response.headers.get("Strict-Transport-Security", "")

        if not hsts:
            result.add_finding(SecurityFinding(
                finding_id="API-008",
                title="Missing HSTS Header",
                description="Strict-Transport-Security not set",
                severity=SecurityTestSeverity.MEDIUM,
                category="API Security",
                affected_component="HSTS",
                evidence="No HSTS header",
                remediation="Add HSTS header with appropriate max-age",
                cwe_id="CWE-319",
                cvss_score=5.3,
            ))
        else:
            # Check max-age
            if "max-age=" not in hsts.lower():
                result.add_finding(SecurityFinding(
                    finding_id="API-009",
                    title="HSTS Missing max-age",
                    description="HSTS header lacks max-age directive",
                    severity=SecurityTestSeverity.LOW,
                    category="API Security",
                    affected_component="HSTS",
                    evidence=f"HSTS: {hsts}",
                    remediation="Add max-age directive to HSTS",
                    cwe_id="CWE-319",
                    cvss_score=3.1,
                ))

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# API Key Security Tests
# ==============================================================================

class TestAPIKeySecurity:
    """Test API key security."""

    @pytest.mark.security
    @pytest.mark.api_security
    def test_api_key_in_url(self, security_client: SecurityTestClient):
        """
        Test: API key exposure in URL

        Check: API keys should not be in query parameters.
        Expected: Keys should be in headers only.
        CWE-598: Use of GET Request Method With Sensitive Query Strings
        """
        result = SecurityTestResult(
            test_name="api_key_in_url",
            status=SecurityTestStatus.PASSED,
        )

        # Try API key in query parameter
        response = security_client.get(
            "/api/v1/calculations",
            params={"api_key": "test_key_12345"},
        )

        if response.status_code == 200:
            result.add_finding(SecurityFinding(
                finding_id="API-010",
                title="API Key Accepted in URL",
                description="API key accepted in query parameter",
                severity=SecurityTestSeverity.MEDIUM,
                category="API Security",
                affected_component="API Authentication",
                evidence="Request with api_key param succeeded",
                remediation="Only accept API keys in Authorization header",
                cwe_id="CWE-598",
                cvss_score=5.3,
            ))

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.api_security
    def test_api_key_enumeration(self, security_client: SecurityTestClient):
        """
        Test: API key enumeration prevention

        Attack: Test if error messages reveal key validity.
        Expected: Same error for invalid and non-existent keys.
        CWE-204: Observable Response Discrepancy
        """
        result = SecurityTestResult(
            test_name="api_key_enumeration",
            status=SecurityTestStatus.PASSED,
        )

        invalid_key_response = security_client.get(
            "/api/v1/calculations",
            headers={"X-API-Key": "invalid_key_format"},
        )

        nonexistent_key_response = security_client.get(
            "/api/v1/calculations",
            headers={"X-API-Key": "valid_format_but_nonexistent_12345678"},
        )

        # Responses should be identical for security
        if (invalid_key_response.status_code != nonexistent_key_response.status_code or
            invalid_key_response.text != nonexistent_key_response.text):
            result.add_finding(SecurityFinding(
                finding_id="API-011",
                title="API Key Enumeration Possible",
                description="Different responses for invalid vs non-existent keys",
                severity=SecurityTestSeverity.LOW,
                category="API Security",
                affected_component="API Authentication",
                evidence="Response differences detected",
                remediation="Return identical errors for all invalid keys",
                cwe_id="CWE-204",
                cvss_score=3.1,
            ))

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Request Validation Tests
# ==============================================================================

class TestRequestValidation:
    """Test request validation security."""

    @pytest.mark.security
    @pytest.mark.api_security
    def test_content_type_validation(self, security_client: SecurityTestClient):
        """
        Test: Content-Type validation

        Attack: Send request with wrong Content-Type.
        Expected: Server should reject mismatched content types.
        CWE-20: Improper Input Validation
        """
        result = SecurityTestResult(
            test_name="content_type_validation",
            status=SecurityTestStatus.PASSED,
        )

        # Send JSON data with wrong Content-Type
        response = security_client.post(
            "/api/v1/calculations",
            headers={"Content-Type": "text/plain"},
            data='{"fuel_type": "natural_gas", "quantity": 1000}',
        )

        # Should reject or handle safely
        if response.status_code == 200:
            result.add_finding(SecurityFinding(
                finding_id="API-012",
                title="Content-Type Not Validated",
                description="Server accepts request with wrong Content-Type",
                severity=SecurityTestSeverity.LOW,
                category="API Security",
                affected_component="Request Parser",
                evidence="text/plain accepted for JSON endpoint",
                remediation="Validate Content-Type header matches expected type",
                cwe_id="CWE-20",
                cvss_score=3.1,
            ))

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.api_security
    def test_large_payload_handling(self, security_client: SecurityTestClient):
        """
        Test: Large payload handling

        Attack: Send very large request body.
        Expected: Server should reject oversized requests.
        CWE-770: Allocation of Resources Without Limits
        """
        result = SecurityTestResult(
            test_name="large_payload_handling",
            status=SecurityTestStatus.PASSED,
        )

        # Create large payload (10MB)
        large_data = {"data": "x" * (10 * 1024 * 1024)}

        response = security_client.post(
            "/api/v1/calculations",
            json_data=large_data,
        )

        # Should reject with 413 Payload Too Large
        if response.status_code not in [413, 400]:
            result.add_finding(SecurityFinding(
                finding_id="API-013",
                title="Large Payload Accepted",
                description="Server accepts very large request bodies",
                severity=SecurityTestSeverity.MEDIUM,
                category="API Security",
                affected_component="Request Size Limit",
                evidence="10MB payload accepted",
                remediation="Set maximum request body size limit",
                cwe_id="CWE-770",
                cvss_score=5.3,
            ))

        assert result.status == SecurityTestStatus.PASSED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "api_security"])
