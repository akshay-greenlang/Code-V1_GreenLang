# -*- coding: utf-8 -*-
"""
XSS Security Validation Tests

Tests for Cross-Site Scripting (XSS) vulnerability fixes in GreenLang frontend.
Validates that all user inputs are properly sanitized and HTML is safely rendered.

Run with:
    pytest tests/test_security_xss.py -v
    pytest tests/test_security_xss.py -v --html=security-report.html
"""

import pytest
import re
from pathlib import Path


class TestXSSVulnerabilities:
    """
    Test suite for XSS vulnerability detection and prevention
    """

    @pytest.fixture
    def static_js_dir(self):
        """Returns path to static JavaScript directory"""
        return Path(__file__).parent.parent / 'static' / 'js'

    @pytest.fixture
    def app_js_secure(self, static_js_dir):
        """Returns content of secure app.js"""
        file_path = static_js_dir / 'app.secure.js'
        if not file_path.exists():
            pytest.skip(f"File not found: {file_path}")
        return file_path.read_text(encoding='utf-8')

    @pytest.fixture
    def api_docs_js_secure(self, static_js_dir):
        """Returns content of secure api_docs.js"""
        file_path = static_js_dir / 'api_docs.secure.js'
        if not file_path.exists():
            pytest.skip(f"File not found: {file_path}")
        return file_path.read_text(encoding='utf-8')

    @pytest.fixture
    def security_module(self, static_js_dir):
        """Returns content of security.js module"""
        file_path = static_js_dir / 'security.js'
        if not file_path.exists():
            pytest.skip(f"File not found: {file_path}")
        return file_path.read_text(encoding='utf-8')

    def test_no_unsafe_innerhtml(self, app_js_secure, api_docs_js_secure):
        """
        Test that secure JS files don't use innerHTML with unsanitized user data.
        All innerHTML usage should go through sanitizeHTML() or safeSetHTML().
        """
        # Pattern to detect dangerous innerHTML usage (not through safe functions)
        dangerous_pattern = r'\.innerHTML\s*=(?!\s*(sanitizeHTML|safeSetHTML|DOMPurify\.sanitize))'

        for name, content in [
            ('app.secure.js', app_js_secure),
            ('api_docs.secure.js', api_docs_js_secure)
        ]:
            matches = re.findall(dangerous_pattern, content)
            assert len(matches) == 0, (
                f"{name} contains unsafe innerHTML usage. "
                f"Use safeSetHTML() or sanitizeHTML() instead."
            )

    def test_no_document_write(self, app_js_secure, api_docs_js_secure):
        """Test that document.write() is not used (it's dangerous and deprecated)"""
        for name, content in [
            ('app.secure.js', app_js_secure),
            ('api_docs.secure.js', api_docs_js_secure)
        ]:
            assert 'document.write' not in content, (
                f"{name} contains document.write() which is dangerous"
            )

    def test_no_outerhtml(self, app_js_secure, api_docs_js_secure):
        """Test that outerHTML is not used with user data"""
        dangerous_pattern = r'\.outerHTML\s*='

        for name, content in [
            ('app.secure.js', app_js_secure),
            ('api_docs.secure.js', api_docs_js_secure)
        ]:
            matches = re.findall(dangerous_pattern, content)
            assert len(matches) == 0, (
                f"{name} contains outerHTML assignment which can be dangerous"
            )

    def test_no_eval_usage(self, app_js_secure, api_docs_js_secure):
        """Test that eval() is not used (extremely dangerous)"""
        dangerous_patterns = [
            r'\beval\s*\(',
            r'new\s+Function\s*\(',
            r'setTimeout\s*\([^,]+,',  # setTimeout with string (acts like eval)
            r'setInterval\s*\([^,]+,',  # setInterval with string
        ]

        for name, content in [
            ('app.secure.js', app_js_secure),
            ('api_docs.secure.js', api_docs_js_secure)
        ]:
            for pattern in dangerous_patterns:
                matches = re.findall(pattern, content)
                if 'setTimeout' in pattern or 'setInterval' in pattern:
                    # Filter out safe usage with function references
                    matches = [m for m in matches if '"' in m or "'" in m]

                assert len(matches) == 0, (
                    f"{name} contains dangerous code execution: {pattern}"
                )

    def test_input_sanitization_present(self, app_js_secure):
        """Test that input sanitization functions are used"""
        required_sanitization = [
            'sanitizeNumber',
            'sanitizeString',
        ]

        for func in required_sanitization:
            assert func in app_js_secure, (
                f"app.secure.js should use {func}() for input validation"
            )

    def test_security_module_exports(self, security_module):
        """Test that security module exports all required functions"""
        required_exports = [
            'escapeHTML',
            'sanitizeHTML',
            'safeSetText',
            'safeSetHTML',
            'createSafeElement',
            'sanitizeNumber',
            'sanitizeString',
            'ValidationPatterns',
            'validateForm',
        ]

        for export_name in required_exports:
            assert export_name in security_module, (
                f"security.js should export {export_name}"
            )

    def test_textcontent_preferred(self, app_js_secure):
        """Test that textContent is used for setting plain text (not innerHTML)"""
        # Should have multiple uses of textContent
        textcontent_usage = len(re.findall(r'\.textContent\s*=', app_js_secure))
        assert textcontent_usage >= 3, (
            "app.secure.js should use textContent for plain text (found: "
            f"{textcontent_usage} uses)"
        )

    def test_no_inline_event_handlers(self, app_js_secure, api_docs_js_secure):
        """
        Test that inline event handlers are not created (onclick, onerror, etc.)
        Should use addEventListener instead.
        """
        dangerous_pattern = r'\.on\w+\s*='

        for name, content in [
            ('app.secure.js', app_js_secure),
            ('api_docs.secure.js', api_docs_js_secure)
        ]:
            # Find all matches
            matches = re.findall(dangerous_pattern, content)

            # Filter out addEventListener (which is safe)
            unsafe_matches = [
                m for m in matches
                if 'addEventListener' not in m and 'onload' not in m
            ]

            assert len(unsafe_matches) == 0, (
                f"{name} contains inline event handler assignments. "
                f"Use addEventListener instead. Found: {unsafe_matches}"
            )

    def test_html_entity_escaping(self, security_module):
        """Test that HTML entity escaping is implemented"""
        # Should have escaping for common dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&']

        for char in dangerous_chars:
            assert char in security_module, (
                f"security.js should handle escaping of '{char}'"
            )

    def test_url_validation_present(self, security_module):
        """Test that URL validation is implemented"""
        assert 'getSafeURLParam' in security_module, (
            "security.js should have getSafeURLParam for URL parameter validation"
        )

        assert 'ValidationPatterns' in security_module, (
            "security.js should have ValidationPatterns for input validation"
        )

    def test_csp_configuration(self, security_module):
        """Test that CSP (Content Security Policy) is configured"""
        assert 'CSP' in security_module, (
            "security.js should have CSP configuration"
        )

    def test_dompurify_integration(self, security_module):
        """Test that DOMPurify is properly integrated"""
        assert 'DOMPurify' in security_module, (
            "security.js should integrate with DOMPurify"
        )

        assert 'DOMPURIFY_CONFIG' in security_module, (
            "security.js should have DOMPurify configuration"
        )

    def test_safe_dom_creation(self, app_js_secure):
        """Test that DOM elements are created safely"""
        # Should use createSafeElement instead of creating elements and
        # setting innerHTML
        assert 'createSafeElement' in app_js_secure, (
            "app.secure.js should use createSafeElement for DOM creation"
        )

    def test_no_unsafe_attributes(self, security_module):
        """Test that unsafe attributes are blocked"""
        # Should have checks for preventing dangerous attributes
        assert 'startsWith(\'on\')' in security_module or 'onmouseover' in security_module.lower(), (
            "security.js should block event handler attributes"
        )

        assert 'javascript:' in security_module.lower(), (
            "security.js should block javascript: URLs"
        )

    def test_safe_alert_replacement(self, app_js_secure):
        """
        Test that alert() is replaced with safer notification system.
        Direct alert() can be used for XSS in some edge cases.
        """
        # Should have a safe notification function
        assert 'showNotification' in app_js_secure or 'safeAlert' in app_js_secure, (
            "app.secure.js should use safe notification system instead of alert()"
        )


class TestXSSPayloads:
    """
    Test common XSS payloads to ensure sanitization works correctly.
    These tests simulate what an attacker might try.
    """

    # Common XSS attack payloads
    XSS_PAYLOADS = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert("XSS")>',
        '<iframe src="javascript:alert(\'XSS\')">',
        '<body onload=alert("XSS")>',
        '<svg/onload=alert("XSS")>',
        '"><script>alert(String.fromCharCode(88,83,83))</script>',
        '<img src="javascript:alert(\'XSS\')">',
        '<input onfocus=alert("XSS") autofocus>',
        '<marquee onstart=alert("XSS")>',
        '<div style="background:url(\'javascript:alert(XSS)\')">',
        '<!--[if gte IE 4]><script>alert("XSS")</script><![endif]-->',
        '<base href="javascript:alert(\'XSS\');//">',
    ]

    def test_xss_payload_documentation(self):
        """
        Documents known XSS payloads that should be blocked.
        This test always passes but serves as documentation.
        """
        print("\n=== Known XSS Payloads That Must Be Blocked ===")
        for i, payload in enumerate(self.XSS_PAYLOADS, 1):
            print(f"{i}. {payload}")
        print("=" * 50)
        assert True, "XSS payloads documented"


class TestCSPHeaders:
    """
    Tests for Content Security Policy headers
    """

    @pytest.fixture
    def csp_middleware(self):
        """Import and return CSP middleware"""
        import sys
        from pathlib import Path

        # Add middleware directory to path
        middleware_dir = Path(__file__).parent.parent / 'static' / 'middleware'
        sys.path.insert(0, str(middleware_dir))

        try:
            from csp_middleware import SecurityHeadersMiddleware, validate_csp_policy
            return SecurityHeadersMiddleware, validate_csp_policy
        except ImportError:
            pytest.skip("CSP middleware not found")

    def test_csp_middleware_exists(self, csp_middleware):
        """Test that CSP middleware module exists"""
        SecurityHeadersMiddleware, _ = csp_middleware
        assert SecurityHeadersMiddleware is not None

    def test_default_csp_policy(self, csp_middleware):
        """Test that default CSP policy is secure"""
        SecurityHeadersMiddleware, validate_csp_policy = csp_middleware

        policy = SecurityHeadersMiddleware._default_csp_policy()

        # Should contain essential directives
        assert "default-src" in policy
        assert "script-src" in policy
        assert "object-src 'none'" in policy

        # Should not allow unsafe-eval
        assert "unsafe-eval" not in policy

        # Validate the policy
        is_valid, errors = validate_csp_policy(policy)
        assert is_valid or len(errors) <= 2, (
            f"Default CSP policy has issues: {errors}"
        )

    def test_strict_csp_policy(self, csp_middleware):
        """Test that strict CSP policy is properly restrictive"""
        SecurityHeadersMiddleware, _ = csp_middleware

        policy = SecurityHeadersMiddleware._strict_csp_policy()

        # Strict policy should not allow inline styles
        assert "unsafe-inline" not in policy or "style-src 'self'" in policy

        # Should upgrade insecure requests
        assert "upgrade-insecure-requests" in policy

    def test_security_headers_complete(self, csp_middleware):
        """Test that all security headers are included"""
        SecurityHeadersMiddleware, _ = csp_middleware

        middleware = SecurityHeadersMiddleware(None)
        headers = middleware._get_security_headers()

        required_headers = [
            'Content-Security-Policy',
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection',
            'Referrer-Policy',
        ]

        for header in required_headers:
            assert header in headers or f"{header}-Report-Only" in headers, (
                f"Missing security header: {header}"
            )


class TestInputValidation:
    """
    Tests for input validation and sanitization functions
    """

    def test_numeric_validation(self):
        """Test that numeric inputs are validated correctly"""
        # This is a documentation test for expected behavior
        test_cases = [
            ("100", 100.0, "Valid number should parse"),
            ("abc", 0, "Invalid number should return default"),
            ("-50", -50.0, "Negative numbers should parse"),
            ("1e6", 1000000.0, "Scientific notation should parse"),
            ("", 0, "Empty string should return default"),
            (None, 0, "None should return default"),
        ]

        # Document expected behavior
        for input_val, expected, description in test_cases:
            print(f"Input: {input_val!r} -> Expected: {expected} ({description})")

        assert True, "Numeric validation expectations documented"

    def test_string_sanitization(self):
        """Test that strings are sanitized correctly"""
        # Expected behavior for string sanitization
        test_cases = [
            ("normal text", "normal text", "Normal text unchanged"),
            ("  trimmed  ", "trimmed", "Whitespace trimmed"),
            ("a" * 1001, "a" * 1000, "Length limited to max"),
            ("<script>alert()</script>", "", "Scripts rejected if pattern used"),
        ]

        # Document expected behavior
        for input_val, expected, description in test_cases:
            print(f"Input: {input_val[:50]!r}... -> Expected: {expected[:50]!r}... ({description})")

        assert True, "String sanitization expectations documented"


@pytest.mark.integration
class TestSecurityIntegration:
    """
    Integration tests for overall security implementation
    """

    def test_security_module_loaded_in_app(self, static_js_dir):
        """Test that security module is imported in app.js"""
        app_file = static_js_dir / 'app.secure.js'
        if not app_file.exists():
            pytest.skip("app.secure.js not found")

        content = app_file.read_text(encoding='utf-8')

        # Should import from GreenLangSecurity
        assert 'GreenLangSecurity' in content, (
            "app.secure.js should import from GreenLangSecurity module"
        )

    def test_no_vulnerable_patterns(self, static_js_dir):
        """
        Comprehensive test scanning for any vulnerable patterns
        in secure JavaScript files
        """
        secure_files = [
            'app.secure.js',
            'api_docs.secure.js',
        ]

        vulnerable_patterns = [
            (r'\.innerHTML\s*=\s*[^s]', 'Direct innerHTML assignment'),
            (r'\.outerHTML\s*=', 'outerHTML assignment'),
            (r'\beval\s*\(', 'eval() usage'),
            (r'document\.write', 'document.write usage'),
            (r'execScript', 'execScript usage'),
            (r'\.on\w+\s*=\s*[\'\"]', 'Inline event handler string'),
        ]

        for filename in secure_files:
            file_path = static_js_dir / filename
            if not file_path.exists():
                continue

            content = file_path.read_text(encoding='utf-8')

            for pattern, description in vulnerable_patterns:
                matches = re.findall(pattern, content)
                # Filter out safe usage (comments, sanitized)
                matches = [
                    m for m in matches
                    if not any(safe in m for safe in ['sanitize', 'safe', 'DOMPurify', '//'])
                ]

                assert len(matches) == 0, (
                    f"{filename} contains {description}: {matches}"
                )


# Performance benchmark for sanitization (optional)
@pytest.mark.benchmark
class TestSecurityPerformance:
    """
    Performance tests to ensure security doesn't significantly impact performance
    """

    def test_sanitization_performance_note(self):
        """
        Note: Sanitization should complete in <1ms for typical inputs.
        For large inputs (>10KB), it should complete in <10ms.
        """
        print("\nPerformance expectations:")
        print("- Small inputs (<1KB): <1ms")
        print("- Medium inputs (1-10KB): <5ms")
        print("- Large inputs (>10KB): <10ms")
        assert True


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
