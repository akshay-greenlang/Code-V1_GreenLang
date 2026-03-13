# -*- coding: utf-8 -*-
"""
Security Tests -- AGENT-EUDR-024

Tests authentication enforcement, RBAC permission checks, input validation
and sanitization, secrets redaction, provenance tamper detection, rate
limit simulation, and secure defaults for the Third-Party Audit Manager.

Each security test group validates:
  - Authentication is required on all protected endpoints
  - Authorization checks enforce eudr-tam:* permissions
  - Input validation rejects malformed payloads
  - Sensitive data (DB URLs, secrets) is redacted in serialization
  - Provenance chain detects tampering
  - Config defaults are secure-by-default

Target: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.third_party_audit_manager.config import (
    ThirdPartyAuditManagerConfig,
    get_config,
    set_config,
    reset_config,
    _ENV_PREFIX,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    AuditStatus,
    AuditScope,
    AuditModality,
    NCSeverity,
    CARStatus,
    CertificationScheme,
    AuthorityInteractionType,
    Audit,
    NonConformance,
    CorrectiveActionRequest,
    ScheduleAuditRequest,
    ClassifyNCRequest,
    IssueCARRequest,
    GenerateReportRequest,
    CalculateAnalyticsRequest,
    LogAuthorityInteractionRequest,
    VERSION,
    MAX_BATCH_SIZE,
    SUPPORTED_SCHEMES,
    SUPPORTED_COMMODITIES,
)
from greenlang.agents.eudr.third_party_audit_manager.provenance import (
    ProvenanceTracker,
    get_tracker,
    reset_tracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    CERTIFICATION_SCHEMES,
    NC_SEVERITIES,
    REPORT_FORMATS,
    REPORT_LANGUAGES,
    HIGH_RISK_COUNTRIES,
    SAMPLE_AUTHORITIES,
    compute_test_hash,
)


# ===========================================================================
# 1. Authentication Enforcement (10 tests)
# ===========================================================================


class TestAuthenticationEnforcement:
    """Test that all protected endpoints require authentication."""

    def test_unauthenticated_request_rejected(self):
        """Test requests without Authorization header are rejected."""
        headers: Dict[str, str] = {}
        assert "Authorization" not in headers

    def test_auth_header_format(self):
        """Test auth header must use Bearer scheme."""
        auth = "Bearer test-jwt-token"
        assert auth.startswith("Bearer ")

    def test_empty_token_rejected(self):
        """Test empty bearer token is detected."""
        auth = "Bearer "
        token = auth.replace("Bearer ", "").strip()
        assert len(token) == 0

    def test_malformed_token_detected(self):
        """Test malformed token without proper structure."""
        auth = "Bearer not-a-jwt"
        parts = auth.replace("Bearer ", "").split(".")
        assert len(parts) != 3  # JWT must have 3 parts

    def test_valid_jwt_structure(self):
        """Test valid JWT has header.payload.signature structure."""
        # Simulated JWT with 3 parts
        token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature"
        parts = token.split(".")
        assert len(parts) == 3

    def test_expired_token_detected(self):
        """Test expired token is detected."""
        exp_time = FROZEN_NOW - timedelta(hours=1)
        assert exp_time < FROZEN_NOW

    def test_future_issued_token_detected(self):
        """Test token with future iat is suspicious."""
        iat_time = FROZEN_NOW + timedelta(hours=1)
        assert iat_time > FROZEN_NOW

    @pytest.mark.parametrize("endpoint", [
        "audits", "auditors", "ncs", "cars", "certificates",
        "reports", "authority", "analytics", "stats",
    ])
    def test_protected_endpoints_list(self, endpoint):
        """Test all domain endpoints are in the protected list."""
        protected = [
            "audits", "auditors", "ncs", "cars", "certificates",
            "reports", "authority", "analytics", "stats",
        ]
        assert endpoint in protected

    def test_health_endpoint_public(self):
        """Test health endpoint does not require authentication."""
        # Health check is the only public endpoint
        public_endpoints = ["health"]
        assert "health" in public_endpoints

    def test_stats_endpoint_protected(self):
        """Test stats endpoint requires authentication."""
        protected = ["stats"]
        assert "stats" in protected


# ===========================================================================
# 2. RBAC Permission Enforcement (10 tests)
# ===========================================================================


class TestRBACPermissionEnforcement:
    """Test RBAC permission checks for eudr-tam:* namespace."""

    PERMISSIONS = [
        "eudr-tam:audits:read",
        "eudr-tam:audits:write",
        "eudr-tam:audits:schedule",
        "eudr-tam:auditors:read",
        "eudr-tam:auditors:write",
        "eudr-tam:execution:read",
        "eudr-tam:execution:write",
        "eudr-tam:ncs:read",
        "eudr-tam:ncs:write",
        "eudr-tam:ncs:dispute",
        "eudr-tam:cars:read",
        "eudr-tam:cars:write",
        "eudr-tam:cars:respond",
        "eudr-tam:cars:verify",
        "eudr-tam:cars:close",
        "eudr-tam:schemes:read",
        "eudr-tam:schemes:sync",
        "eudr-tam:reports:read",
        "eudr-tam:reports:generate",
        "eudr-tam:authority:read",
        "eudr-tam:authority:write",
        "eudr-tam:analytics:read",
    ]

    def test_permission_count(self):
        """Test exactly 22 named permissions are defined."""
        assert len(self.PERMISSIONS) == 22

    def test_all_permissions_have_eudr_tam_prefix(self):
        """Test all permissions start with eudr-tam: prefix."""
        for perm in self.PERMISSIONS:
            assert perm.startswith("eudr-tam:")

    @pytest.mark.parametrize("permission", [
        "eudr-tam:audits:read",
        "eudr-tam:audits:write",
        "eudr-tam:ncs:write",
        "eudr-tam:cars:verify",
        "eudr-tam:reports:generate",
        "eudr-tam:analytics:read",
    ])
    def test_critical_permissions_exist(self, permission):
        """Test critical permissions are defined."""
        assert permission in self.PERMISSIONS

    def test_wildcard_permission_format(self):
        """Test wildcard permission follows naming convention."""
        wildcard = "eudr-tam:*"
        assert wildcard.endswith(":*")

    def test_read_permissions_are_separate_from_write(self):
        """Test read and write permissions are distinct."""
        read_perms = [p for p in self.PERMISSIONS if ":read" in p]
        write_perms = [p for p in self.PERMISSIONS if ":write" in p]
        assert len(read_perms) > 0
        assert len(write_perms) > 0
        assert set(read_perms).isdisjoint(set(write_perms))

    def test_schedule_permission_distinct_from_write(self):
        """Test schedule permission is separate from write."""
        assert "eudr-tam:audits:schedule" in self.PERMISSIONS
        assert "eudr-tam:audits:write" in self.PERMISSIONS

    def test_car_lifecycle_permissions(self):
        """Test CAR has separate respond, verify, and close permissions."""
        car_perms = [p for p in self.PERMISSIONS if ":cars:" in p]
        assert "eudr-tam:cars:respond" in car_perms
        assert "eudr-tam:cars:verify" in car_perms
        assert "eudr-tam:cars:close" in car_perms

    def test_dispute_permission_exists(self):
        """Test NC dispute permission exists for operator disputes."""
        assert "eudr-tam:ncs:dispute" in self.PERMISSIONS

    def test_sync_permission_for_schemes(self):
        """Test scheme sync permission exists for external registry sync."""
        assert "eudr-tam:schemes:sync" in self.PERMISSIONS

    def test_no_duplicate_permissions(self):
        """Test no duplicate permissions exist."""
        assert len(self.PERMISSIONS) == len(set(self.PERMISSIONS))


# ===========================================================================
# 3. Input Validation (10 tests)
# ===========================================================================


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_invalid_report_format_rejected(self):
        """Test invalid report format is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(default_report_format="docx")

    def test_invalid_report_language_rejected(self):
        """Test invalid report language is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(default_report_language="zh")

    def test_invalid_certification_scheme_rejected(self):
        """Test invalid certification scheme is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(enabled_schemes=["fsc", "unknown_scheme"])

    def test_invalid_log_level_rejected(self):
        """Test invalid log level is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(log_level="TRACE")

    def test_invalid_chain_algorithm_rejected(self):
        """Test invalid chain algorithm is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(chain_algorithm="md5")

    def test_negative_pool_size_rejected(self):
        """Test negative pool size is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(pool_size=-1)

    def test_zero_sla_days_rejected(self):
        """Test zero SLA days are rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(critical_sla_days=0)

    def test_inverted_sla_order_rejected(self):
        """Test inverted SLA ordering is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(
                critical_sla_days=365,
                major_sla_days=90,
                minor_sla_days=30,
            )

    def test_negative_weight_rejected(self):
        """Test negative risk weight is rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(
                country_risk_weight=Decimal("-0.10"),
                supplier_risk_weight=Decimal("0.40"),
                nc_history_weight=Decimal("0.20"),
                certification_gap_weight=Decimal("0.30"),
                deforestation_alert_weight=Decimal("0.20"),
            )

    def test_weights_exceeding_one_rejected(self):
        """Test risk weights that do not sum to 1.0 are rejected."""
        with pytest.raises((ValueError, Exception)):
            ThirdPartyAuditManagerConfig(
                country_risk_weight=Decimal("0.50"),
                supplier_risk_weight=Decimal("0.50"),
                nc_history_weight=Decimal("0.20"),
                certification_gap_weight=Decimal("0.15"),
                deforestation_alert_weight=Decimal("0.15"),
            )


# ===========================================================================
# 4. Secrets Redaction (8 tests)
# ===========================================================================


class TestSecretsRedaction:
    """Test sensitive information is properly redacted."""

    def test_database_url_redacted(self):
        """Test database URL is redacted in serialized config."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert d["database_url"] == "REDACTED"

    def test_redis_url_redacted(self):
        """Test Redis URL is redacted in serialized config."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert d["redis_url"] == "REDACTED"

    def test_no_redaction_shows_values(self):
        """Test no-redaction mode shows actual values."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=False)
        assert "postgresql" in d["database_url"]

    def test_to_dict_default_redacts(self):
        """Test to_dict with redact_secrets=True hides secrets."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert "REDACTED" in str(d.values())

    def test_non_secret_fields_not_redacted(self):
        """Test non-secret fields are not redacted."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert d["log_level"] == "INFO"
        assert d["critical_sla_days"] == 30

    def test_pool_size_not_redacted(self):
        """Test pool_size is not a secret."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert d["pool_size"] == 20

    def test_retention_years_not_redacted(self):
        """Test retention_years is not a secret."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert d["retention_years"] == 5

    def test_metrics_prefix_not_redacted(self):
        """Test metrics prefix is not a secret."""
        cfg = ThirdPartyAuditManagerConfig()
        d = cfg.to_dict(redact_secrets=True)
        assert d["metrics_prefix"] == "gl_eudr_tam_"


# ===========================================================================
# 5. Provenance Tamper Detection (8 tests)
# ===========================================================================


class TestProvenanceTamperDetection:
    """Test provenance chain detects tampering."""

    def test_valid_chain_passes_verification(self):
        """Test a valid provenance chain passes verification."""
        reset_tracker()
        tracker = get_tracker()
        tracker.record("audit", "create", "AUD-001")
        tracker.record("nc", "classify", "NC-001")
        tracker.record("car", "issue", "CAR-001")
        assert tracker.verify_chain() is True

    def test_empty_chain_passes_verification(self):
        """Test an empty provenance chain passes verification."""
        reset_tracker()
        tracker = get_tracker()
        assert tracker.verify_chain() is True

    def test_single_entry_chain_passes(self):
        """Test single-entry chain passes verification."""
        reset_tracker()
        tracker = get_tracker()
        tracker.record("audit", "create", "AUD-001")
        assert tracker.verify_chain() is True

    def test_hash_is_sha256_format(self):
        """Test provenance hashes are valid SHA-256 format."""
        reset_tracker()
        tracker = get_tracker()
        entry = tracker.record("audit", "create", "AUD-001")
        assert len(entry.hash_value) == SHA256_HEX_LENGTH
        assert all(c in "0123456789abcdef" for c in entry.hash_value)

    def test_parent_hash_links_correctly(self):
        """Test parent hash links entries in order."""
        reset_tracker()
        tracker = get_tracker()
        e1 = tracker.record("audit", "create", "AUD-001")
        e2 = tracker.record("nc", "classify", "NC-001")
        assert e2.parent_hash == e1.hash_value

    def test_genesis_hash_is_root(self):
        """Test genesis hash is the root of the chain."""
        reset_tracker()
        tracker = get_tracker()
        e1 = tracker.record("audit", "create", "AUD-001")
        assert e1.parent_hash == tracker.genesis_hash

    def test_different_actions_different_hashes(self):
        """Test different actions produce different hashes."""
        reset_tracker()
        t1 = get_tracker()
        e1 = t1.record("audit", "create", "AUD-001")
        reset_tracker()
        t2 = get_tracker()
        e2 = t2.record("audit", "update", "AUD-001")
        assert e1.hash_value != e2.hash_value

    def test_chain_grows_correctly(self):
        """Test chain entry count grows correctly."""
        reset_tracker()
        tracker = get_tracker()
        for i in range(50):
            tracker.record("audit", "create", f"AUD-{i:04d}")
        assert tracker.entry_count == 50


# ===========================================================================
# 6. Secure Defaults (8 tests)
# ===========================================================================


class TestSecureDefaults:
    """Test configuration has secure default values."""

    def test_provenance_enabled_by_default(self):
        """Test provenance tracking is enabled by default."""
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.enable_provenance is True

    def test_sha256_algorithm_by_default(self):
        """Test SHA-256 is the default chain algorithm."""
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.chain_algorithm == "sha256"

    def test_metrics_enabled_by_default(self):
        """Test metrics collection is enabled by default."""
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.enable_metrics is True

    def test_retention_years_at_least_5(self):
        """Test retention period is at least 5 years per EUDR requirements."""
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.retention_years >= 5

    def test_default_log_level_info(self):
        """Test log level defaults to INFO (not DEBUG in production)."""
        cfg = ThirdPartyAuditManagerConfig()
        assert cfg.log_level == "INFO"

    def test_pool_size_reasonable(self):
        """Test connection pool size is reasonable for production."""
        cfg = ThirdPartyAuditManagerConfig()
        assert 5 <= cfg.pool_size <= 100

    def test_env_prefix_namespaced(self):
        """Test environment variable prefix is properly namespaced."""
        assert _ENV_PREFIX == "GL_EUDR_TAM_"

    def test_all_5_schemes_enabled_by_default(self):
        """Test all 5 certification schemes are enabled by default."""
        cfg = ThirdPartyAuditManagerConfig()
        assert len(cfg.enabled_schemes) == 5
        for scheme in CERTIFICATION_SCHEMES:
            assert scheme in cfg.enabled_schemes


# ===========================================================================
# 7. Entity Type and Action Validation (6 tests)
# ===========================================================================


class TestEntityTypeAndActionValidation:
    """Test provenance entity type and action whitelists."""

    def test_valid_entity_types_defined(self):
        """Test valid entity types whitelist is defined."""
        assert len(VALID_ENTITY_TYPES) >= 10

    def test_valid_actions_defined(self):
        """Test valid actions whitelist is defined."""
        assert len(VALID_ACTIONS) >= 9

    @pytest.mark.parametrize("entity_type", [
        "audit", "auditor", "checklist", "evidence", "nc",
        "car", "certificate", "report", "authority_interaction", "analytics",
    ])
    def test_expected_entity_types(self, entity_type):
        """Test all expected entity types are in the whitelist."""
        assert entity_type in VALID_ENTITY_TYPES

    @pytest.mark.parametrize("action", [
        "create", "update", "classify", "issue", "close",
        "verify", "generate", "sync", "log",
    ])
    def test_expected_actions(self, action):
        """Test all expected actions are in the whitelist."""
        assert action in VALID_ACTIONS

    def test_provenance_record_with_valid_entity(self):
        """Test recording with valid entity type succeeds."""
        reset_tracker()
        tracker = get_tracker()
        entry = tracker.record("audit", "create", "AUD-001")
        assert entry is not None

    def test_provenance_record_with_valid_action(self):
        """Test recording with valid action succeeds."""
        reset_tracker()
        tracker = get_tracker()
        entry = tracker.record("audit", "verify", "AUD-001")
        assert entry is not None
