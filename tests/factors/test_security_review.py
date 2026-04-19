# -*- coding: utf-8 -*-
"""
Security test suite for GreenLang Factors API.

Tests input validation, API key management, security audit helpers,
and attack vector mitigations (SQL injection, XSS, auth bypass,
rate limiting, CORS, and tier enforcement).
"""

from __future__ import annotations

import hashlib
import os
from unittest import mock

import pytest

from greenlang.factors.security.input_validation import (
    MAX_QUERY_LENGTH,
    sanitize_geography,
    sanitize_search_query,
    validate_edition_id,
    validate_factor_id,
    validate_pagination,
)
from greenlang.factors.security.api_key_manager import (
    VALID_TIERS,
    extract_tier_from_key,
    generate_api_key,
    generate_partner_api_key,
    hash_api_key,
    rotate_api_key,
    validate_api_key_format,
)
from greenlang.factors.security.audit import (
    SecurityFinding,
    Severity,
    check_auth_config,
    check_cors_config,
    check_headers,
)


# ====================================================================
# INPUT VALIDATION: SQL Injection
# ====================================================================


class TestSQLInjectionPrevention:
    """Test that SQL injection payloads are sanitized from search queries."""

    def test_basic_sql_injection_stripped(self):
        result = sanitize_search_query("diesel; DROP TABLE factors;--")
        assert "DROP" not in result
        assert ";" not in result
        assert "--" not in result
        assert "diesel" in result

    def test_union_select_injection(self):
        result = sanitize_search_query("diesel UNION SELECT * FROM users")
        assert "UNION" not in result
        assert "SELECT" not in result

    def test_comment_injection(self):
        result = sanitize_search_query("diesel /* malicious comment */ fuel")
        # Comments stripped
        assert "/*" not in result
        assert "*/" not in result

    def test_single_quote_or_injection(self):
        result = sanitize_search_query("diesel' OR '1'='1")
        assert "'" not in result

    def test_sleep_injection(self):
        result = sanitize_search_query("diesel; SLEEP(5)")
        assert "SLEEP" not in result

    def test_benchmark_injection(self):
        result = sanitize_search_query("diesel; BENCHMARK(1000000, SHA1('test'))")
        assert "BENCHMARK" not in result

    def test_nested_sql_injection(self):
        result = sanitize_search_query(
            "a; INSERT INTO factors VALUES('bad','bad'); DELETE FROM factors;--"
        )
        assert "INSERT" not in result
        assert "DELETE" not in result

    def test_clean_query_unchanged(self):
        result = sanitize_search_query("natural gas stationary combustion")
        assert result == "natural gas stationary combustion"


# ====================================================================
# INPUT VALIDATION: XSS
# ====================================================================


class TestXSSPrevention:
    """Test that XSS payloads are sanitized from search queries and factor IDs."""

    def test_script_tag_stripped(self):
        result = sanitize_search_query('diesel<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "alert" not in result
        assert "diesel" in result

    def test_img_onerror_stripped(self):
        result = sanitize_search_query('diesel<img src=x onerror=alert(1)>')
        assert "<img" not in result
        assert "onerror" not in result

    def test_javascript_protocol_stripped(self):
        result = sanitize_search_query("javascript:alert(document.cookie)")
        assert "javascript:" not in result

    def test_svg_onload_stripped(self):
        result = sanitize_search_query('<svg onload=alert(1)>diesel</svg>')
        assert "<svg" not in result
        assert "onload" not in result

    def test_iframe_stripped(self):
        result = sanitize_search_query('<iframe src="http://evil.com"></iframe>diesel')
        assert "<iframe" not in result

    def test_data_protocol_stripped(self):
        result = sanitize_search_query("data:text/html,<script>alert(1)</script>")
        assert "data:" not in result

    def test_event_handler_stripped(self):
        result = sanitize_search_query('diesel" onmouseover="alert(1)')
        assert "onmouseover" not in result

    def test_xss_in_factor_id_rejected(self):
        assert not validate_factor_id('<script>alert(1)</script>')
        assert not validate_factor_id('EF:EPA:<img src=x>:US:2024:v1')


# ====================================================================
# INPUT VALIDATION: Factor ID format
# ====================================================================


class TestFactorIDValidation:
    """Test factor_id format validation."""

    def test_valid_factor_id(self):
        assert validate_factor_id("EF:EPA:diesel:US:2024:v1")

    def test_valid_factor_id_with_hyphens(self):
        assert validate_factor_id("EF:DESNZ:natural-gas:GB:2024:v2")

    def test_valid_factor_id_with_underscores(self):
        assert validate_factor_id("EF:EPA:fuel_oil_2:US-CA:2024:v1")

    def test_valid_factor_id_multi_version(self):
        assert validate_factor_id("EF:IPCC:coal:GLOBAL:2023:v1.2")

    def test_invalid_missing_prefix(self):
        assert not validate_factor_id("EPA:diesel:US:2024:v1")

    def test_invalid_wrong_prefix(self):
        assert not validate_factor_id("XX:EPA:diesel:US:2024:v1")

    def test_invalid_empty(self):
        assert not validate_factor_id("")

    def test_invalid_too_long(self):
        assert not validate_factor_id("EF:" + "a" * 200)

    def test_invalid_no_version(self):
        assert not validate_factor_id("EF:EPA:diesel:US:2024")

    def test_invalid_bad_year(self):
        assert not validate_factor_id("EF:EPA:diesel:US:24:v1")

    def test_invalid_special_chars(self):
        assert not validate_factor_id("EF:EPA:diesel;DROP:US:2024:v1")


# ====================================================================
# INPUT VALIDATION: Edition ID format
# ====================================================================


class TestEditionIDValidation:
    """Test edition_id format validation."""

    def test_valid_dated_edition(self):
        assert validate_edition_id("2026.04.1")

    def test_valid_dated_edition_double_digit_month(self):
        assert validate_edition_id("2026.12.3")

    def test_valid_builtin_edition(self):
        assert validate_edition_id("builtin-v1.0.0")

    def test_valid_builtin_edition_simple(self):
        assert validate_edition_id("builtin-v1")

    def test_valid_slug_edition(self):
        assert validate_edition_id("test-edition")

    def test_invalid_empty(self):
        assert not validate_edition_id("")

    def test_invalid_too_long(self):
        assert not validate_edition_id("a" * 101)

    def test_invalid_starts_with_number_not_date(self):
        # Must match YYYY.MM.N pattern if starts with digits
        assert not validate_edition_id("123")

    def test_invalid_uppercase_slug(self):
        assert not validate_edition_id("Test-Edition")


# ====================================================================
# INPUT VALIDATION: Geography codes
# ====================================================================


class TestGeographySanitization:
    """Test geography code validation and normalization."""

    def test_valid_country_code(self):
        assert sanitize_geography("US") == "US"
        assert sanitize_geography("GB") == "GB"
        assert sanitize_geography("DE") == "DE"

    def test_lowercase_normalized_to_upper(self):
        assert sanitize_geography("us") == "US"
        assert sanitize_geography("gb") == "GB"

    def test_valid_subdivision_code(self):
        assert sanitize_geography("US-CA") == "US-CA"
        assert sanitize_geography("GB-ENG") == "GB-ENG"

    def test_empty_returns_empty(self):
        assert sanitize_geography("") == ""

    def test_invalid_too_long(self):
        assert sanitize_geography("ABCDEFGHIJK") == ""

    def test_invalid_numbers_only(self):
        assert sanitize_geography("123") == ""

    def test_invalid_special_chars(self):
        assert sanitize_geography("US;DROP") == ""


# ====================================================================
# INPUT VALIDATION: Pagination
# ====================================================================


class TestPaginationValidation:
    """Test pagination parameter clamping."""

    def test_default_values(self):
        offset, limit = validate_pagination()
        assert offset == 0
        assert limit == 25

    def test_negative_offset_clamped(self):
        offset, _ = validate_pagination(offset=-10)
        assert offset == 0

    def test_zero_limit_clamped(self):
        _, limit = validate_pagination(limit=0)
        assert limit == 1

    def test_excessive_limit_clamped(self):
        _, limit = validate_pagination(limit=10000)
        assert limit == 500

    def test_custom_max_limit(self):
        _, limit = validate_pagination(limit=200, max_limit=100)
        assert limit == 100


# ====================================================================
# INPUT VALIDATION: Query length
# ====================================================================


class TestQueryLengthEnforcement:
    """Test max query length enforcement."""

    def test_normal_query_not_truncated(self):
        query = "diesel fuel combustion"
        result = sanitize_search_query(query)
        assert result == query

    def test_long_query_truncated(self):
        query = "a " * 300  # 600 chars
        result = sanitize_search_query(query)
        assert len(result) <= MAX_QUERY_LENGTH

    def test_exact_max_length_not_truncated(self):
        query = "a" * MAX_QUERY_LENGTH
        result = sanitize_search_query(query)
        assert len(result) == MAX_QUERY_LENGTH

    def test_empty_query(self):
        assert sanitize_search_query("") == ""


# ====================================================================
# API KEY MANAGEMENT
# ====================================================================


class TestAPIKeyGeneration:
    """Test API key generation and format."""

    def test_generate_community_key(self):
        key = generate_api_key("community")
        assert key.startswith("gl_community_")
        assert len(key) > 20

    def test_generate_pro_key(self):
        key = generate_api_key("pro")
        assert key.startswith("gl_pro_")

    def test_generate_enterprise_key(self):
        key = generate_api_key("enterprise")
        assert key.startswith("gl_enterprise_")

    def test_generate_test_key(self):
        key = generate_api_key("test")
        assert key.startswith("gl_test_")

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="Invalid tier"):
            generate_api_key("platinum")

    def test_keys_are_unique(self):
        key1 = generate_api_key("pro")
        key2 = generate_api_key("pro")
        assert key1 != key2

    def test_all_valid_tiers(self):
        for tier in VALID_TIERS:
            key = generate_api_key(tier)
            assert key.startswith("gl_%s_" % tier)


class TestAPIKeyHashing:
    """Test API key hashing for secure storage."""

    def test_hash_returns_64_hex_chars(self):
        key = generate_api_key("pro")
        hashed = hash_api_key(key)
        assert len(hashed) == 64
        # Verify it is valid hex
        int(hashed, 16)

    def test_hash_is_deterministic(self):
        key = "gl_pro_test_key_1234567890"
        assert hash_api_key(key) == hash_api_key(key)

    def test_different_keys_different_hashes(self):
        key1 = generate_api_key("pro")
        key2 = generate_api_key("pro")
        assert hash_api_key(key1) != hash_api_key(key2)

    def test_hash_matches_sha256(self):
        key = "gl_test_known_value"
        expected = hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert hash_api_key(key) == expected

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            hash_api_key("")


class TestAPIKeyFormatValidation:
    """Test API key format validation."""

    def test_valid_generated_key(self):
        key = generate_api_key("pro")
        assert validate_api_key_format(key)

    def test_valid_community_key(self):
        assert validate_api_key_format("gl_community_abcdefghijklmnopqrstuvwx")

    def test_valid_test_key(self):
        assert validate_api_key_format("gl_test_load_key_xxxxxxxxxxxxxxxxxxxxxxxx")

    def test_invalid_empty(self):
        assert not validate_api_key_format("")

    def test_invalid_no_prefix(self):
        assert not validate_api_key_format("api_key_12345678")

    def test_invalid_wrong_prefix(self):
        assert not validate_api_key_format("gx_pro_abcdefghijklmnopqrstuvwx")

    def test_invalid_bad_tier(self):
        assert not validate_api_key_format("gl_platinum_abcdefghijklmnopqrstuvwx")

    def test_invalid_too_short_random(self):
        assert not validate_api_key_format("gl_pro_abc")

    def test_invalid_too_long(self):
        assert not validate_api_key_format("gl_pro_" + "a" * 200)


class TestAPIKeyTierExtraction:
    """Test extracting tier from API key."""

    def test_extract_pro(self):
        key = generate_api_key("pro")
        assert extract_tier_from_key(key) == "pro"

    def test_extract_enterprise(self):
        key = generate_api_key("enterprise")
        assert extract_tier_from_key(key) == "enterprise"

    def test_extract_invalid_key(self):
        assert extract_tier_from_key("invalid_key") is None


class TestAPIKeyRotation:
    """Test API key rotation."""

    def test_rotate_generates_new_key(self):
        old_key = generate_api_key("pro")
        old_hash = hash_api_key(old_key)
        new_key = rotate_api_key(old_hash)
        assert new_key != old_key
        assert validate_api_key_format(new_key)

    def test_rotate_invalid_hash_raises(self):
        with pytest.raises(ValueError, match="64-character"):
            rotate_api_key("tooshort")

    def test_rotate_non_hex_hash_raises(self):
        with pytest.raises(ValueError, match="hexadecimal"):
            rotate_api_key("g" * 64)


class TestPartnerAPIKeyGeneration:
    """Test partner-specific API key generation."""

    def test_partner_key_format(self):
        key = generate_partner_api_key("acme-corp")
        assert key.startswith("gl_partner_acme-corp_")

    def test_partner_key_empty_id_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            generate_partner_api_key("")


# ====================================================================
# SECURITY AUDIT: Headers
# ====================================================================


class TestSecurityHeadersCheck:
    """Test security header audit checks."""

    def test_all_headers_present(self):
        headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'",
            "X-Request-ID": "req-12345",
            "Cache-Control": "no-store",
        }
        findings = check_headers(headers)
        assert len(findings) == 0

    def test_missing_hsts_is_critical(self):
        findings = check_headers({})
        hsts = [f for f in findings if "Strict-Transport-Security" in f.message]
        assert len(hsts) == 1
        assert hsts[0].severity == Severity.CRITICAL

    def test_missing_content_type_options_is_warning(self):
        findings = check_headers({})
        cto = [f for f in findings if "X-Content-Type-Options" in f.message]
        assert len(cto) == 1
        assert cto[0].severity == Severity.WARNING

    def test_wrong_content_type_value(self):
        headers = {"X-Content-Type-Options": "sniff"}
        findings = check_headers(headers)
        wrong = [f for f in findings if "unexpected value" in f.message]
        assert len(wrong) >= 1

    def test_unsupported_request_type(self):
        findings = check_headers(42)  # not a dict or request
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING

    def test_finding_has_remediation(self):
        findings = check_headers({})
        for f in findings:
            assert f.remediation, "Finding %s missing remediation" % f.message


# ====================================================================
# SECURITY AUDIT: CORS
# ====================================================================


class TestCORSConfigCheck:
    """Test CORS configuration audit checks."""

    def test_no_cors_is_info(self):
        findings = check_cors_config(None)
        assert len(findings) == 1
        assert findings[0].severity == Severity.INFO

    def test_wildcard_origin_production_is_critical(self):
        with mock.patch.dict(os.environ, {"GL_ENV": "production"}):
            cors = {"allow_origins": ["*"]}
            findings = check_cors_config(cors)
            critical = [f for f in findings if f.severity == Severity.CRITICAL]
            assert len(critical) >= 1

    def test_wildcard_origin_dev_is_warning(self):
        with mock.patch.dict(os.environ, {"GL_ENV": "development"}):
            cors = {"allow_origins": ["*"]}
            findings = check_cors_config(cors)
            warnings = [f for f in findings if f.severity == Severity.WARNING]
            assert len(warnings) >= 1

    def test_credentials_with_wildcard_is_critical(self):
        cors = {
            "allow_origins": ["*"],
            "allow_credentials": True,
        }
        with mock.patch.dict(os.environ, {"GL_ENV": "development"}):
            findings = check_cors_config(cors)
            cred = [f for f in findings if "credentials" in f.message.lower()]
            assert len(cred) >= 1
            assert cred[0].severity == Severity.CRITICAL

    def test_specific_origins_no_findings(self):
        cors = {
            "allow_origins": ["https://app.greenlang.io"],
            "allow_methods": ["GET", "POST"],
            "max_age": 3600,
        }
        findings = check_cors_config(cors)
        assert len(findings) == 0

    def test_wildcard_methods_is_warning(self):
        cors = {
            "allow_origins": ["https://app.greenlang.io"],
            "allow_methods": ["*"],
        }
        findings = check_cors_config(cors)
        methods = [f for f in findings if "methods" in f.message.lower()]
        assert len(methods) >= 1

    def test_excessive_max_age_is_warning(self):
        cors = {
            "allow_origins": ["https://app.greenlang.io"],
            "max_age": 999999,
        }
        findings = check_cors_config(cors)
        age = [f for f in findings if "max_age" in f.message]
        assert len(age) >= 1


# ====================================================================
# SECURITY AUDIT: Auth Configuration
# ====================================================================


class TestAuthConfigCheck:
    """Test authentication configuration audit."""

    def test_missing_jwt_secret_is_critical(self):
        with mock.patch.dict(os.environ, {"GL_JWT_SECRET": ""}, clear=False):
            findings = check_auth_config()
            jwt = [f for f in findings if "GL_JWT_SECRET" in f.message and "not set" in f.message]
            assert len(jwt) >= 1
            assert jwt[0].severity == Severity.CRITICAL

    def test_short_jwt_secret_is_critical(self):
        with mock.patch.dict(os.environ, {"GL_JWT_SECRET": "short"}, clear=False):
            findings = check_auth_config()
            short = [f for f in findings if "too short" in f.message]
            assert len(short) >= 1

    def test_insecure_jwt_secret_is_critical(self):
        with mock.patch.dict(os.environ, {"GL_JWT_SECRET": "changeme"}, clear=False):
            findings = check_auth_config()
            insecure = [f for f in findings if "insecure" in f.message]
            assert len(insecure) >= 1

    def test_none_algorithm_is_critical(self):
        with mock.patch.dict(
            os.environ,
            {"GL_JWT_SECRET": "a" * 48, "GL_JWT_ALGORITHM": "none"},
            clear=False,
        ):
            findings = check_auth_config()
            none_alg = [f for f in findings if "'none'" in f.message]
            assert len(none_alg) >= 1

    def test_no_https_in_prod_is_critical(self):
        with mock.patch.dict(
            os.environ,
            {
                "GL_JWT_SECRET": "a" * 48,
                "GL_ENV": "production",
                "GL_FORCE_HTTPS": "",
            },
            clear=False,
        ):
            findings = check_auth_config()
            https = [f for f in findings if "HTTPS" in f.message]
            assert len(https) >= 1

    def test_valid_config_minimal_findings(self):
        with mock.patch.dict(
            os.environ,
            {
                "GL_JWT_SECRET": "a" * 48,
                "GL_JWT_ALGORITHM": "RS256",
                "GL_JWT_EXPIRATION_MINUTES": "60",
                "GL_ENV": "production",
                "GL_FORCE_HTTPS": "true",
                "GL_REQUIRE_API_KEY": "true",
            },
            clear=False,
        ):
            findings = check_auth_config()
            critical = [f for f in findings if f.severity == Severity.CRITICAL]
            assert len(critical) == 0


# ====================================================================
# SECURITY: Auth Bypass Attempts
# ====================================================================


class TestAuthBypassPrevention:
    """Test that auth bypass patterns are caught by validation."""

    def test_empty_api_key_invalid(self):
        assert not validate_api_key_format("")

    def test_null_string_api_key_invalid(self):
        assert not validate_api_key_format("null")

    def test_undefined_api_key_invalid(self):
        assert not validate_api_key_format("undefined")

    def test_bearer_only_invalid(self):
        assert not validate_api_key_format("Bearer")

    def test_jwt_like_string_invalid(self):
        assert not validate_api_key_format(
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc"
        )

    def test_admin_override_attempt_invalid(self):
        assert not validate_api_key_format("gl_admin_override")


# ====================================================================
# SECURITY: Tier Enforcement (Connector-Only Visibility)
# ====================================================================


class TestConnectorOnlyTierEnforcement:
    """Test that connector_only factors are hidden from community tier."""

    def test_community_cannot_see_connector_only(self):
        from greenlang.factors.tier_enforcement import (
            TierVisibility,
            factor_visible_for_tier,
        )

        tv = TierVisibility.from_tier("community")
        assert not factor_visible_for_tier("connector_only", tv)

    def test_pro_cannot_see_connector_only(self):
        from greenlang.factors.tier_enforcement import (
            TierVisibility,
            factor_visible_for_tier,
        )

        tv = TierVisibility.from_tier("pro")
        assert not factor_visible_for_tier("connector_only", tv)

    def test_enterprise_can_see_connector_only(self):
        from greenlang.factors.tier_enforcement import (
            TierVisibility,
            factor_visible_for_tier,
        )

        tv = TierVisibility.from_tier("enterprise")
        assert factor_visible_for_tier("connector_only", tv)

    def test_filter_removes_connector_for_community(self):
        from greenlang.factors.tier_enforcement import (
            TierVisibility,
            filter_factors_by_tier,
        )

        factors = [
            {"factor_id": "1", "factor_status": "certified"},
            {"factor_id": "2", "factor_status": "connector_only"},
            {"factor_id": "3", "factor_status": "certified"},
        ]
        tv = TierVisibility.from_tier("community")
        result = filter_factors_by_tier(factors, tv)
        assert len(result) == 2
        factor_ids = [f["factor_id"] for f in result]
        assert "2" not in factor_ids

    def test_filter_keeps_connector_for_enterprise(self):
        from greenlang.factors.tier_enforcement import (
            TierVisibility,
            filter_factors_by_tier,
        )

        factors = [
            {"factor_id": "1", "factor_status": "certified"},
            {"factor_id": "2", "factor_status": "connector_only"},
        ]
        tv = TierVisibility.from_tier("enterprise")
        tv.include_connector = True
        result = filter_factors_by_tier(factors, tv)
        assert len(result) == 2


# ====================================================================
# SECURITY: SecurityFinding dataclass
# ====================================================================


class TestSecurityFinding:
    """Test SecurityFinding data structure."""

    def test_to_dict(self):
        finding = SecurityFinding(
            severity=Severity.CRITICAL,
            category="auth",
            message="Test message",
            remediation="Fix it",
        )
        d = finding.to_dict()
        assert d["severity"] == "critical"
        assert d["category"] == "auth"
        assert d["message"] == "Test message"
        assert d["remediation"] == "Fix it"

    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"


# ====================================================================
# SECURITY: Combined attack vectors
# ====================================================================


class TestCombinedAttackVectors:
    """Test combined/chained attack patterns."""

    def test_sql_plus_xss(self):
        payload = "'; <script>alert(document.cookie)</script> -- DROP TABLE"
        result = sanitize_search_query(payload)
        assert "<script>" not in result
        assert "DROP" not in result
        assert "alert" not in result

    def test_encoded_injection_characters_stripped(self):
        # Characters outside allowed set are removed
        result = sanitize_search_query("diesel\x00\x01\x02fuel")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_unicode_normalization(self):
        # Unicode should be stripped (not in allowed charset)
        result = sanitize_search_query("diesel\u2018OR\u2019")
        # Smart quotes are stripped
        assert result == "dieselOR"

    def test_factor_id_with_path_traversal(self):
        assert not validate_factor_id("EF:EPA:../../etc/passwd:US:2024:v1")

    def test_edition_id_with_path_traversal(self):
        assert not validate_edition_id("../../etc/passwd")

    def test_geography_with_injection(self):
        assert sanitize_geography("US'; DROP TABLE") == ""
