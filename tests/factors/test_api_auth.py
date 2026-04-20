# -*- coding: utf-8 -*-
"""Tests for Phase 4.1 Factors API auth."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenlang.factors.api_auth import (
    APIKeyRecord,
    APIKeyValidator,
    _hash_key,
    authenticate_headers,
    min_tier_for_endpoint,
    tier_allows_endpoint,
    tier_rank,
)


# --------------------------------------------------------------------------
# Tier ordering + endpoint gating
# --------------------------------------------------------------------------


class TestTierOrdering:
    def test_tier_rank_known(self):
        assert tier_rank("community") == 0
        assert tier_rank("pro") == 1
        assert tier_rank("enterprise") == 2
        assert tier_rank("internal") == 3

    def test_tier_rank_unknown_is_zero(self):
        assert tier_rank("") == 0
        assert tier_rank(None) == 0
        assert tier_rank("bogus") == 0

    def test_community_allowed_on_search(self):
        assert tier_allows_endpoint("community", "/api/v1/factors/search")

    def test_community_blocked_on_match(self):
        assert not tier_allows_endpoint("community", "/api/v1/factors/match")

    def test_pro_allowed_on_match(self):
        assert tier_allows_endpoint("pro", "/api/v1/factors/match")

    def test_pro_blocked_on_audit(self):
        assert not tier_allows_endpoint(
            "pro", "/api/v1/factors/abc/audit-bundle"
        )

    def test_enterprise_allowed_everywhere(self):
        for ep in [
            "/api/v1/factors/search",
            "/api/v1/factors/match",
            "/api/v1/factors/export",
            "/api/v1/factors/abc/audit-bundle",
        ]:
            assert tier_allows_endpoint("enterprise", ep), ep

    def test_min_tier_defaults_to_community(self):
        assert min_tier_for_endpoint("/api/v1/factors/unknown") == "community"

    def test_min_tier_audit_bundle(self):
        assert min_tier_for_endpoint(
            "/api/v1/factors/abc/audit-bundle"
        ) == "enterprise"


# --------------------------------------------------------------------------
# API key validation
# --------------------------------------------------------------------------


class TestAPIKeyValidator:
    def test_validate_known_key(self):
        rec = APIKeyRecord(
            key_id="dev-001",
            key_hash=_hash_key("gl_test_key"),
            tier="community",
            tenant_id="dev-tenant",
        )
        v = APIKeyValidator(keys=[rec])
        result = v.validate("gl_test_key")
        assert result is not None
        assert result.key_id == "dev-001"

    def test_validate_unknown_key_returns_none(self):
        v = APIKeyValidator(keys=[])
        assert v.validate("nope") is None

    def test_validate_empty_key_returns_none(self):
        v = APIKeyValidator(keys=[])
        assert v.validate("") is None

    def test_inactive_key_rejected(self):
        rec = APIKeyRecord(
            key_id="dev-001",
            key_hash=_hash_key("gl_test_key"),
            active=False,
        )
        v = APIKeyValidator(keys=[rec])
        assert v.validate("gl_test_key") is None

    def test_record_to_user_shape(self):
        rec = APIKeyRecord(
            key_id="enterprise-007",
            key_hash=_hash_key("x"),
            tier="enterprise",
            tenant_id="acme",
            user_id="ceo@acme.com",
        )
        user = rec.to_user()
        assert user["tenant_id"] == "acme"
        assert user["tier"] == "enterprise"
        assert user["auth_method"] == "api_key"
        assert user["api_key_id"] == "enterprise-007"

    def test_env_var_loading(self, monkeypatch):
        keyring = json.dumps(
            [
                {
                    "key_id": "k1",
                    "key": "secret1",
                    "tier": "pro",
                    "tenant_id": "t1",
                }
            ]
        )
        monkeypatch.setenv("GL_FACTORS_API_KEYS", keyring)
        v = APIKeyValidator()
        result = v.validate("secret1")
        assert result is not None
        assert result.tier == "pro"

    def test_file_loading(self, tmp_path: Path, monkeypatch):
        file = tmp_path / "keys.json"
        file.write_text(
            json.dumps(
                [
                    {
                        "key_id": "k2",
                        "key": "topsecret",
                        "tier": "enterprise",
                        "tenant_id": "acme",
                    }
                ]
            )
        )
        monkeypatch.setenv("GL_FACTORS_API_KEY_FILE", str(file))
        monkeypatch.setenv("GL_FACTORS_API_KEYS", "")
        v = APIKeyValidator()
        assert v.validate("topsecret") is not None


# --------------------------------------------------------------------------
# authenticate_headers (pure)
# --------------------------------------------------------------------------


class TestAuthenticateHeaders:
    def _validator(self) -> APIKeyValidator:
        return APIKeyValidator(
            keys=[
                APIKeyRecord(
                    key_id="dev",
                    key_hash=_hash_key("api_key_1"),
                    tier="community",
                    tenant_id="t1",
                    user_id="u1",
                )
            ]
        )

    def test_api_key_path(self):
        user = authenticate_headers(
            authorization=None,
            api_key_header="api_key_1",
            validator=self._validator(),
        )
        assert user is not None
        assert user["auth_method"] == "api_key"
        assert user["tier"] == "community"
        assert user["tenant_id"] == "t1"

    def test_jwt_path(self):
        def fake_decode(token):
            return {
                "sub": "alice",
                "tier": "pro",
                "tenant_id": "t2",
                "roles": ["admin"],
            }

        user = authenticate_headers(
            authorization="Bearer fake-jwt",
            api_key_header=None,
            jwt_decode=fake_decode,
            validator=self._validator(),
        )
        assert user is not None
        assert user["auth_method"] == "jwt"
        assert user["tier"] == "pro"
        assert user["user_id"] == "alice"

    def test_jwt_preferred_over_api_key(self):
        def fake_decode(token):
            return {"sub": "bob", "tier": "enterprise", "tenant_id": "ent"}

        user = authenticate_headers(
            authorization="Bearer jwt",
            api_key_header="api_key_1",
            jwt_decode=fake_decode,
            validator=self._validator(),
        )
        assert user["auth_method"] == "jwt"

    def test_invalid_jwt_falls_back_to_api_key(self):
        def raising_decode(token):
            raise ValueError("bad sig")

        user = authenticate_headers(
            authorization="Bearer bogus",
            api_key_header="api_key_1",
            jwt_decode=raising_decode,
            validator=self._validator(),
        )
        assert user is not None
        assert user["auth_method"] == "api_key"

    def test_no_credentials_returns_none(self):
        assert (
            authenticate_headers(
                authorization=None, api_key_header=None,
                validator=self._validator(),
            )
            is None
        )

    def test_missing_sub_in_jwt_falls_back(self):
        def fake_decode(_token):
            return {"tier": "pro"}  # no sub

        user = authenticate_headers(
            authorization="Bearer token",
            api_key_header="api_key_1",
            jwt_decode=fake_decode,
            validator=self._validator(),
        )
        # Since sub is missing, fallback to api key path.
        assert user is not None
        assert user["auth_method"] == "api_key"
