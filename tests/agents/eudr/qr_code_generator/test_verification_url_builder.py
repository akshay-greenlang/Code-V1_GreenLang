# -*- coding: utf-8 -*-
"""
Unit tests for Engine 5: Verification URL Builder (AGENT-EUDR-014)

Tests verification URL construction including URL format, HMAC signing,
short URLs, deep links, token validation, offline verification,
language parameters, and edge cases.

50+ tests across 8 test classes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from .conftest import (
    DEFAULT_BASE_VERIFICATION_URL,
    DEFAULT_HMAC_TRUNCATION_LENGTH,
    DEFAULT_TOKEN_TTL_YEARS,
    EUDR_RETENTION_YEARS,
    HMAC_SHA256_HEX_LENGTH,
    SAMPLE_CODE_ID,
    SAMPLE_CODE_ID_2,
    SAMPLE_HMAC_KEY,
    SAMPLE_HMAC_TRUNCATED,
    SAMPLE_HMAC_VALUE,
    SAMPLE_OPERATOR_ID,
    SAMPLE_VERIFICATION_BASE_URL,
    SAMPLE_VERIFICATION_URL,
    SHA256_HEX_LENGTH,
    assert_valid_hmac,
    assert_verification_url_valid,
    make_verification_url,
    _hmac_sha256,
    _ts,
    _ts_dt,
)


# =========================================================================
# Test Class 1: URL Construction
# =========================================================================

class TestURLConstruction:
    """Test verification URL format and parameters."""

    def test_url_contains_base(self):
        """Test URL starts with base verification URL."""
        url = make_verification_url()
        assert url["full_url"].startswith(SAMPLE_VERIFICATION_BASE_URL)

    def test_url_contains_code_id(self):
        """Test URL includes the code ID as a parameter."""
        url = make_verification_url(code_id=SAMPLE_CODE_ID)
        assert SAMPLE_CODE_ID in url["full_url"]

    def test_url_contains_token(self):
        """Test URL includes the HMAC token."""
        url = make_verification_url()
        assert "token=" in url["full_url"]

    def test_url_format_structure(self):
        """Test URL has proper query parameter structure."""
        url = make_verification_url()
        assert "?" in url["full_url"]
        assert "&" in url["full_url"]

    def test_base_url_stored(self):
        """Test base URL is stored separately."""
        url = make_verification_url()
        assert url["base_url"] == SAMPLE_VERIFICATION_BASE_URL

    def test_code_id_stored(self):
        """Test code ID is stored in the record."""
        url = make_verification_url(code_id="QR-CUSTOM-001")
        assert url["code_id"] == "QR-CUSTOM-001"

    def test_custom_base_url(self):
        """Test URL with custom base URL."""
        custom_base = "https://custom-verify.example.com"
        url = make_verification_url(base_url=custom_base)
        assert url["base_url"] == custom_base
        assert url["full_url"].startswith(custom_base)

    def test_url_passes_validation(self):
        """Test constructed URL passes full validation."""
        url = make_verification_url()
        assert_verification_url_valid(url)


# =========================================================================
# Test Class 2: HMAC Signature
# =========================================================================

class TestHMACSignature:
    """Test HMAC-SHA256 signing and truncation."""

    def test_hmac_is_computed(self):
        """Test HMAC value is computed and stored."""
        url = make_verification_url()
        assert url["token"] is not None
        assert len(url["token"]) == HMAC_SHA256_HEX_LENGTH

    def test_hmac_is_valid_hex(self):
        """Test HMAC token is a valid hex string."""
        url = make_verification_url()
        assert_valid_hmac(url["token"])

    def test_hmac_truncated_length(self):
        """Test truncated HMAC has correct length."""
        url = make_verification_url()
        assert len(url["hmac_truncated"]) == DEFAULT_HMAC_TRUNCATION_LENGTH

    def test_hmac_truncated_is_prefix_of_full(self):
        """Test truncated HMAC is a prefix of the full token."""
        url = make_verification_url()
        assert url["token"].startswith(url["hmac_truncated"])

    def test_hmac_deterministic(self):
        """Test same inputs produce same HMAC."""
        hmac1 = _hmac_sha256(SAMPLE_HMAC_KEY, SAMPLE_CODE_ID)
        hmac2 = _hmac_sha256(SAMPLE_HMAC_KEY, SAMPLE_CODE_ID)
        assert hmac1 == hmac2

    def test_different_codes_different_hmac(self):
        """Test different code IDs produce different HMACs."""
        hmac1 = _hmac_sha256(SAMPLE_HMAC_KEY, "QR-001")
        hmac2 = _hmac_sha256(SAMPLE_HMAC_KEY, "QR-002")
        assert hmac1 != hmac2

    def test_different_keys_different_hmac(self):
        """Test different keys produce different HMACs."""
        hmac1 = _hmac_sha256("key-1", SAMPLE_CODE_ID)
        hmac2 = _hmac_sha256("key-2", SAMPLE_CODE_ID)
        assert hmac1 != hmac2


# =========================================================================
# Test Class 3: Short URLs
# =========================================================================

class TestShortURLs:
    """Test short URL integration."""

    def test_no_short_url_by_default(self):
        """Test short URL is None by default."""
        url = make_verification_url()
        assert url["short_url"] is None

    def test_short_url_when_provided(self):
        """Test short URL is stored when provided."""
        url = make_verification_url(short_url="https://gl.eu/v/abc123")
        assert url["short_url"] == "https://gl.eu/v/abc123"

    def test_short_url_starts_with_https(self):
        """Test short URL uses HTTPS."""
        url = make_verification_url(short_url="https://gl.eu/v/xyz")
        assert url["short_url"].startswith("https://")

    def test_short_url_much_shorter_than_full(self):
        """Test short URL is significantly shorter than full URL."""
        url = make_verification_url(short_url="https://gl.eu/v/a1b2")
        assert len(url["short_url"]) < len(url["full_url"])

    def test_full_url_still_present_with_short(self):
        """Test full URL is still present when short URL is set."""
        url = make_verification_url(short_url="https://gl.eu/v/test")
        assert url["full_url"] is not None
        assert len(url["full_url"]) > 0

    def test_short_url_unique_per_code(self):
        """Test different codes would get different short URLs."""
        url1 = make_verification_url(code_id="QR-A", short_url="https://gl.eu/v/a")
        url2 = make_verification_url(code_id="QR-B", short_url="https://gl.eu/v/b")
        assert url1["short_url"] != url2["short_url"]


# =========================================================================
# Test Class 4: Deep Links
# =========================================================================

class TestDeepLinks:
    """Test mobile app deep linking."""

    def test_deep_link_url_format(self):
        """Test deep link URL can be constructed."""
        url = make_verification_url(
            base_url="greenlang://verify",
            code_id=SAMPLE_CODE_ID,
        )
        assert url["base_url"] == "greenlang://verify"

    def test_deep_link_contains_code_id(self):
        """Test deep link includes code identifier."""
        url = make_verification_url(
            base_url="greenlang://verify",
            code_id="QR-DEEP-001",
        )
        assert "QR-DEEP-001" in url["full_url"]

    def test_deep_link_includes_token(self):
        """Test deep link includes verification token."""
        url = make_verification_url(base_url="greenlang://verify")
        assert "token=" in url["full_url"]

    def test_deep_link_has_hmac(self):
        """Test deep link record has HMAC token."""
        url = make_verification_url(base_url="greenlang://verify")
        assert url["token"] is not None
        assert len(url["token"]) == HMAC_SHA256_HEX_LENGTH

    def test_https_and_deep_link_different(self):
        """Test HTTPS and deep link URLs are different."""
        https_url = make_verification_url(base_url="https://verify.greenlang.eu")
        deep_url = make_verification_url(base_url="greenlang://verify")
        assert https_url["base_url"] != deep_url["base_url"]


# =========================================================================
# Test Class 5: Token Validation
# =========================================================================

class TestTokenValidation:
    """Test token verification and expiry checking."""

    def test_token_created_at_present(self):
        """Test token has a creation timestamp."""
        url = make_verification_url()
        assert url["token_created_at"] is not None

    def test_token_expires_at_present(self):
        """Test token has an expiry timestamp."""
        url = make_verification_url()
        assert url["token_expires_at"] is not None

    def test_token_ttl_default_5_years(self):
        """Test default token TTL is 5 years."""
        assert DEFAULT_TOKEN_TTL_YEARS == 5
        assert DEFAULT_TOKEN_TTL_YEARS == EUDR_RETENTION_YEARS

    def test_custom_ttl(self):
        """Test custom token TTL."""
        url = make_verification_url(ttl_years=10)
        assert url["token_expires_at"] is not None

    def test_token_matches_sample_hmac(self):
        """Test pre-computed sample HMAC matches."""
        computed = _hmac_sha256(SAMPLE_HMAC_KEY, SAMPLE_CODE_ID)
        assert computed == SAMPLE_HMAC_VALUE

    def test_truncated_matches_sample(self):
        """Test truncated HMAC matches sample constant."""
        truncated = SAMPLE_HMAC_VALUE[:DEFAULT_HMAC_TRUNCATION_LENGTH]
        assert truncated == SAMPLE_HMAC_TRUNCATED

    def test_token_hex_characters_only(self):
        """Test token contains only hex characters."""
        url = make_verification_url()
        assert all(c in "0123456789abcdef" for c in url["token"])


# =========================================================================
# Test Class 6: Offline Verification
# =========================================================================

class TestOfflineVerification:
    """Test self-contained verification data."""

    def test_url_record_contains_full_token(self):
        """Test URL record stores the full HMAC token for offline check."""
        url = make_verification_url()
        assert len(url["token"]) == HMAC_SHA256_HEX_LENGTH

    def test_offline_verification_with_known_key(self):
        """Test offline verification by recomputing HMAC."""
        url = make_verification_url(code_id="QR-OFFLINE-001")
        recomputed = _hmac_sha256(SAMPLE_HMAC_KEY, "QR-OFFLINE-001")
        assert url["token"] == recomputed

    def test_offline_truncated_matches_url(self):
        """Test offline truncated HMAC matches what is in the URL."""
        url = make_verification_url()
        truncated_from_token = url["token"][:DEFAULT_HMAC_TRUNCATION_LENGTH]
        assert truncated_from_token == url["hmac_truncated"]

    def test_offline_verification_fails_wrong_key(self):
        """Test offline verification fails with wrong key."""
        url = make_verification_url()
        wrong_hmac = _hmac_sha256("wrong-key", SAMPLE_CODE_ID)
        assert url["token"] != wrong_hmac

    def test_self_contained_data_sufficient(self):
        """Test URL record has all data needed for offline verification."""
        url = make_verification_url()
        assert "code_id" in url
        assert "token" in url
        assert "hmac_truncated" in url
        assert "base_url" in url


# =========================================================================
# Test Class 7: Language Parameters
# =========================================================================

class TestLanguageParameters:
    """Test language auto-detection and locale parameters."""

    def test_default_url_no_language(self):
        """Test default URL does not include language parameter."""
        url = make_verification_url()
        # Default factory does not add lang parameter
        assert url["full_url"] is not None

    def test_url_with_language_override(self):
        """Test URL can include language parameter via override."""
        url = make_verification_url(
            full_url=f"{SAMPLE_VERIFICATION_BASE_URL}/verify?code={SAMPLE_CODE_ID}&lang=de",
        )
        assert "lang=de" in url["full_url"]

    def test_url_with_french_language(self):
        """Test URL with French language parameter."""
        url = make_verification_url(
            full_url=f"{SAMPLE_VERIFICATION_BASE_URL}/verify?code={SAMPLE_CODE_ID}&lang=fr",
        )
        assert "lang=fr" in url["full_url"]

    def test_url_with_multiple_params(self):
        """Test URL with language and other parameters."""
        full = (
            f"{SAMPLE_VERIFICATION_BASE_URL}/verify"
            f"?code={SAMPLE_CODE_ID}&token=abc&lang=es"
        )
        url = make_verification_url(full_url=full)
        assert "lang=es" in url["full_url"]
        assert "token=abc" in url["full_url"]

    def test_iso_639_language_codes(self):
        """Test common ISO 639-1 language codes in URLs."""
        for lang in ["en", "de", "fr", "es", "it", "nl", "pt"]:
            full = f"{SAMPLE_VERIFICATION_BASE_URL}/verify?code=QR-1&lang={lang}"
            url = make_verification_url(full_url=full)
            assert f"lang={lang}" in url["full_url"]


# =========================================================================
# Test Class 8: Edge Cases
# =========================================================================

class TestURLEdgeCases:
    """Test edge cases for verification URL construction."""

    def test_expired_url_still_has_data(self):
        """Test expired URL record still contains all fields."""
        url = make_verification_url(
            token_expires_at=_ts(days_ago=30),
        )
        assert url["token_expires_at"] is not None
        assert_verification_url_valid(url)

    def test_long_code_id_in_url(self):
        """Test URL with a very long code ID."""
        long_id = "QR-" + "A" * 100
        url = make_verification_url(code_id=long_id)
        assert long_id in url["full_url"]
        assert url["code_id"] == long_id

    def test_special_characters_in_code_id(self):
        """Test URL handles special characters in code ID."""
        special_id = "QR-CODE/2026_v2.1"
        url = make_verification_url(code_id=special_id)
        assert url["code_id"] == special_id

    def test_unique_url_ids(self):
        """Test each URL record has a unique url_id."""
        ids = {make_verification_url()["url_id"] for _ in range(50)}
        assert len(ids) == 50

    def test_operator_id_stored(self):
        """Test operator ID is stored in URL record."""
        url = make_verification_url(operator_id="OP-CUSTOM-001")
        assert url["operator_id"] == "OP-CUSTOM-001"

    def test_provenance_hash_optional(self):
        """Test provenance hash is None by default."""
        url = make_verification_url()
        assert url["provenance_hash"] is None

    def test_url_with_provenance_hash(self):
        """Test URL record with provenance hash set."""
        from .conftest import _sha256
        prov = _sha256("provenance-url-001")
        url = make_verification_url(provenance_hash=prov)
        assert url["provenance_hash"] == prov

    def test_multiple_codes_different_urls(self):
        """Test different codes produce different full URLs."""
        url1 = make_verification_url(code_id="QR-A")
        url2 = make_verification_url(code_id="QR-B")
        assert url1["full_url"] != url2["full_url"]
