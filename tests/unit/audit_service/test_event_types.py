# -*- coding: utf-8 -*-
"""
Unit tests for Audit Event Types - SEC-005: Centralized Audit Logging Service

Tests all event type enumerations, categories, severities, and mapping functions.
Validates enum values, string representations, and category lookups.

Coverage targets: 85%+ of event_types.py
"""

from __future__ import annotations

import pytest

from greenlang.infrastructure.audit_service.event_types import (
    AuditEventCategory,
    AuditSeverity,
    AuditAction,
    AuditResult,
    UnifiedAuditEventType,
    AUTH_EVENT_TYPE_MAP,
    RBAC_EVENT_TYPE_MAP,
    ENCRYPTION_EVENT_TYPE_MAP,
)


# ============================================================================
# TestAuditEventCategory
# ============================================================================


class TestAuditEventCategory:
    """Tests for AuditEventCategory enumeration."""

    def test_all_categories_exist(self) -> None:
        """All expected categories are defined."""
        expected = {"auth", "rbac", "encryption", "data", "agent", "system", "api", "compliance"}
        actual = {c.value for c in AuditEventCategory}
        assert expected == actual

    def test_category_is_string_enum(self) -> None:
        """Categories are string-based enums."""
        for category in AuditEventCategory:
            assert isinstance(category.value, str)
            assert category.value == str(category.value)

    def test_auth_category(self) -> None:
        """AUTH category has correct value."""
        assert AuditEventCategory.AUTH.value == "auth"
        assert AuditEventCategory.AUTH == "auth"

    def test_rbac_category(self) -> None:
        """RBAC category has correct value."""
        assert AuditEventCategory.RBAC.value == "rbac"

    def test_encryption_category(self) -> None:
        """ENCRYPTION category has correct value."""
        assert AuditEventCategory.ENCRYPTION.value == "encryption"

    def test_data_category(self) -> None:
        """DATA category has correct value."""
        assert AuditEventCategory.DATA.value == "data"

    def test_agent_category(self) -> None:
        """AGENT category has correct value."""
        assert AuditEventCategory.AGENT.value == "agent"

    def test_system_category(self) -> None:
        """SYSTEM category has correct value."""
        assert AuditEventCategory.SYSTEM.value == "system"

    def test_api_category(self) -> None:
        """API category has correct value."""
        assert AuditEventCategory.API.value == "api"

    def test_compliance_category(self) -> None:
        """COMPLIANCE category has correct value."""
        assert AuditEventCategory.COMPLIANCE.value == "compliance"

    def test_category_count(self) -> None:
        """Correct number of categories exist."""
        assert len(AuditEventCategory) == 8


# ============================================================================
# TestAuditSeverity
# ============================================================================


class TestAuditSeverity:
    """Tests for AuditSeverity enumeration."""

    def test_all_severities_exist(self) -> None:
        """All expected severities are defined."""
        expected = {"debug", "info", "warning", "error", "critical"}
        actual = {s.value for s in AuditSeverity}
        assert expected == actual

    def test_severity_is_string_enum(self) -> None:
        """Severities are string-based enums."""
        for severity in AuditSeverity:
            assert isinstance(severity.value, str)

    def test_debug_severity(self) -> None:
        """DEBUG severity has correct value."""
        assert AuditSeverity.DEBUG.value == "debug"

    def test_info_severity(self) -> None:
        """INFO severity has correct value."""
        assert AuditSeverity.INFO.value == "info"

    def test_warning_severity(self) -> None:
        """WARNING severity has correct value."""
        assert AuditSeverity.WARNING.value == "warning"

    def test_error_severity(self) -> None:
        """ERROR severity has correct value."""
        assert AuditSeverity.ERROR.value == "error"

    def test_critical_severity(self) -> None:
        """CRITICAL severity has correct value."""
        assert AuditSeverity.CRITICAL.value == "critical"

    def test_severity_count(self) -> None:
        """Correct number of severities exist."""
        assert len(AuditSeverity) == 5


# ============================================================================
# TestAuditAction
# ============================================================================


class TestAuditAction:
    """Tests for AuditAction enumeration."""

    def test_all_actions_exist(self) -> None:
        """All expected actions are defined."""
        expected = {
            "create", "read", "update", "delete", "execute",
            "export", "import", "grant", "revoke", "validate",
            "encrypt", "decrypt",
        }
        actual = {a.value for a in AuditAction}
        assert expected == actual

    def test_action_is_string_enum(self) -> None:
        """Actions are string-based enums."""
        for action in AuditAction:
            assert isinstance(action.value, str)

    def test_crud_actions(self) -> None:
        """Standard CRUD actions are defined."""
        assert AuditAction.CREATE.value == "create"
        assert AuditAction.READ.value == "read"
        assert AuditAction.UPDATE.value == "update"
        assert AuditAction.DELETE.value == "delete"

    def test_execute_action(self) -> None:
        """EXECUTE action is defined."""
        assert AuditAction.EXECUTE.value == "execute"

    def test_data_actions(self) -> None:
        """EXPORT and IMPORT actions are defined."""
        assert AuditAction.EXPORT.value == "export"
        assert AuditAction.IMPORT.value == "import"

    def test_permission_actions(self) -> None:
        """GRANT and REVOKE actions are defined."""
        assert AuditAction.GRANT.value == "grant"
        assert AuditAction.REVOKE.value == "revoke"

    def test_encryption_actions(self) -> None:
        """ENCRYPT and DECRYPT actions are defined."""
        assert AuditAction.ENCRYPT.value == "encrypt"
        assert AuditAction.DECRYPT.value == "decrypt"


# ============================================================================
# TestAuditResult
# ============================================================================


class TestAuditResult:
    """Tests for AuditResult enumeration."""

    def test_all_results_exist(self) -> None:
        """All expected results are defined."""
        expected = {"success", "failure", "denied", "error", "timeout", "skipped"}
        actual = {r.value for r in AuditResult}
        assert expected == actual

    def test_result_is_string_enum(self) -> None:
        """Results are string-based enums."""
        for result in AuditResult:
            assert isinstance(result.value, str)

    def test_success_result(self) -> None:
        """SUCCESS result has correct value."""
        assert AuditResult.SUCCESS.value == "success"

    def test_failure_result(self) -> None:
        """FAILURE result has correct value."""
        assert AuditResult.FAILURE.value == "failure"

    def test_denied_result(self) -> None:
        """DENIED result has correct value."""
        assert AuditResult.DENIED.value == "denied"

    def test_error_result(self) -> None:
        """ERROR result has correct value."""
        assert AuditResult.ERROR.value == "error"

    def test_timeout_result(self) -> None:
        """TIMEOUT result has correct value."""
        assert AuditResult.TIMEOUT.value == "timeout"

    def test_skipped_result(self) -> None:
        """SKIPPED result has correct value."""
        assert AuditResult.SKIPPED.value == "skipped"


# ============================================================================
# TestUnifiedAuditEventType
# ============================================================================


class TestUnifiedAuditEventType:
    """Tests for UnifiedAuditEventType enumeration."""

    # ------------------------------------------------------------------
    # Event existence tests
    # ------------------------------------------------------------------

    def test_auth_events_count(self) -> None:
        """17 AUTH events are defined."""
        auth_events = [e for e in UnifiedAuditEventType if e.value.startswith("auth.")]
        assert len(auth_events) == 17

    def test_rbac_events_count(self) -> None:
        """13 RBAC events are defined."""
        rbac_events = [e for e in UnifiedAuditEventType if e.value.startswith("rbac.")]
        assert len(rbac_events) == 13

    def test_encryption_events_count(self) -> None:
        """20 ENCRYPTION events are defined."""
        enc_events = [e for e in UnifiedAuditEventType if e.value.startswith("encryption.")]
        assert len(enc_events) == 20

    def test_data_events_count(self) -> None:
        """8 DATA events are defined."""
        data_events = [e for e in UnifiedAuditEventType if e.value.startswith("data.")]
        assert len(data_events) == 8

    def test_agent_events_count(self) -> None:
        """8 AGENT events are defined."""
        agent_events = [e for e in UnifiedAuditEventType if e.value.startswith("agent.")]
        assert len(agent_events) == 8

    def test_system_events_count(self) -> None:
        """6 SYSTEM events are defined."""
        system_events = [e for e in UnifiedAuditEventType if e.value.startswith("system.")]
        assert len(system_events) == 6

    def test_api_events_count(self) -> None:
        """6 API events are defined."""
        api_events = [e for e in UnifiedAuditEventType if e.value.startswith("api.")]
        assert len(api_events) == 6

    def test_compliance_events_count(self) -> None:
        """6 COMPLIANCE events are defined."""
        compliance_events = [e for e in UnifiedAuditEventType if e.value.startswith("compliance.")]
        assert len(compliance_events) == 6

    def test_total_event_count(self) -> None:
        """Total event count is 84 (17+13+20+8+8+6+6+6)."""
        assert len(UnifiedAuditEventType) == 84

    # ------------------------------------------------------------------
    # Event value format tests
    # ------------------------------------------------------------------

    def test_all_events_lowercase_snake_case(self) -> None:
        """All event values are lowercase snake_case."""
        for event in UnifiedAuditEventType:
            assert event.value == event.value.lower()
            assert "_" in event.value or "." in event.value

    def test_all_events_have_category_prefix(self) -> None:
        """All event values have a category prefix with dot separator."""
        for event in UnifiedAuditEventType:
            assert "." in event.value
            prefix = event.value.split(".")[0]
            assert prefix in {c.value for c in AuditEventCategory}

    def test_event_is_string_enum(self) -> None:
        """Events are string-based enums."""
        for event in UnifiedAuditEventType:
            assert isinstance(event.value, str)
            assert event == event.value

    # ------------------------------------------------------------------
    # Specific event tests
    # ------------------------------------------------------------------

    def test_auth_login_success(self) -> None:
        """AUTH_LOGIN_SUCCESS event has correct value."""
        assert UnifiedAuditEventType.AUTH_LOGIN_SUCCESS.value == "auth.login_success"

    def test_auth_login_failure(self) -> None:
        """AUTH_LOGIN_FAILURE event has correct value."""
        assert UnifiedAuditEventType.AUTH_LOGIN_FAILURE.value == "auth.login_failure"

    def test_rbac_role_created(self) -> None:
        """RBAC_ROLE_CREATED event has correct value."""
        assert UnifiedAuditEventType.RBAC_ROLE_CREATED.value == "rbac.role_created"

    def test_encryption_performed(self) -> None:
        """ENCRYPTION_PERFORMED event has correct value."""
        assert UnifiedAuditEventType.ENCRYPTION_PERFORMED.value == "encryption.performed"

    def test_agent_started(self) -> None:
        """AGENT_STARTED event has correct value."""
        assert UnifiedAuditEventType.AGENT_STARTED.value == "agent.started"

    def test_compliance_violation_detected(self) -> None:
        """COMPLIANCE_VIOLATION_DETECTED event has correct value."""
        assert UnifiedAuditEventType.COMPLIANCE_VIOLATION_DETECTED.value == "compliance.violation_detected"

    # ------------------------------------------------------------------
    # Category lookup tests
    # ------------------------------------------------------------------

    def test_get_category_auth(self) -> None:
        """get_category returns AUTH for auth events."""
        category = UnifiedAuditEventType.get_category(UnifiedAuditEventType.AUTH_LOGIN_SUCCESS)
        assert category == AuditEventCategory.AUTH

    def test_get_category_rbac(self) -> None:
        """get_category returns RBAC for rbac events."""
        category = UnifiedAuditEventType.get_category(UnifiedAuditEventType.RBAC_ROLE_CREATED)
        assert category == AuditEventCategory.RBAC

    def test_get_category_encryption(self) -> None:
        """get_category returns ENCRYPTION for encryption events."""
        category = UnifiedAuditEventType.get_category(UnifiedAuditEventType.ENCRYPTION_PERFORMED)
        assert category == AuditEventCategory.ENCRYPTION

    def test_get_category_compliance(self) -> None:
        """get_category returns COMPLIANCE for compliance events."""
        category = UnifiedAuditEventType.get_category(UnifiedAuditEventType.COMPLIANCE_VIOLATION_DETECTED)
        assert category == AuditEventCategory.COMPLIANCE

    # ------------------------------------------------------------------
    # Default severity tests
    # ------------------------------------------------------------------

    def test_get_default_severity_critical(self) -> None:
        """Critical events return CRITICAL severity."""
        severity = UnifiedAuditEventType.get_default_severity(
            UnifiedAuditEventType.COMPLIANCE_VIOLATION_DETECTED
        )
        assert severity == AuditSeverity.CRITICAL

    def test_get_default_severity_error(self) -> None:
        """Error events return ERROR severity."""
        severity = UnifiedAuditEventType.get_default_severity(
            UnifiedAuditEventType.AUTH_LOGIN_FAILURE
        )
        assert severity == AuditSeverity.ERROR

    def test_get_default_severity_warning(self) -> None:
        """Warning events return WARNING severity."""
        severity = UnifiedAuditEventType.get_default_severity(
            UnifiedAuditEventType.RBAC_AUTHORIZATION_DENIED
        )
        assert severity == AuditSeverity.WARNING

    def test_get_default_severity_info(self) -> None:
        """Default events return INFO severity."""
        severity = UnifiedAuditEventType.get_default_severity(
            UnifiedAuditEventType.AUTH_LOGIN_SUCCESS
        )
        assert severity == AuditSeverity.INFO

    def test_account_locked_is_critical(self) -> None:
        """AUTH_ACCOUNT_LOCKED returns CRITICAL severity."""
        severity = UnifiedAuditEventType.get_default_severity(
            UnifiedAuditEventType.AUTH_ACCOUNT_LOCKED
        )
        assert severity == AuditSeverity.CRITICAL

    def test_system_error_is_critical(self) -> None:
        """SYSTEM_ERROR returns CRITICAL severity."""
        severity = UnifiedAuditEventType.get_default_severity(
            UnifiedAuditEventType.SYSTEM_ERROR
        )
        assert severity == AuditSeverity.CRITICAL


# ============================================================================
# TestEventTypeMaps
# ============================================================================


class TestEventTypeMaps:
    """Tests for legacy event type mapping dictionaries."""

    # ------------------------------------------------------------------
    # AUTH_EVENT_TYPE_MAP tests
    # ------------------------------------------------------------------

    def test_auth_map_login_success(self) -> None:
        """AUTH map contains login_success mapping."""
        assert AUTH_EVENT_TYPE_MAP["login_success"] == UnifiedAuditEventType.AUTH_LOGIN_SUCCESS

    def test_auth_map_login_failure(self) -> None:
        """AUTH map contains login_failure mapping."""
        assert AUTH_EVENT_TYPE_MAP["login_failure"] == UnifiedAuditEventType.AUTH_LOGIN_FAILURE

    def test_auth_map_token_issued(self) -> None:
        """AUTH map contains token_issued mapping."""
        assert AUTH_EVENT_TYPE_MAP["token_issued"] == UnifiedAuditEventType.AUTH_TOKEN_ISSUED

    def test_auth_map_account_locked(self) -> None:
        """AUTH map contains account_locked mapping."""
        assert AUTH_EVENT_TYPE_MAP["account_locked"] == UnifiedAuditEventType.AUTH_ACCOUNT_LOCKED

    def test_auth_map_count(self) -> None:
        """AUTH map has expected number of entries."""
        assert len(AUTH_EVENT_TYPE_MAP) >= 15

    # ------------------------------------------------------------------
    # RBAC_EVENT_TYPE_MAP tests
    # ------------------------------------------------------------------

    def test_rbac_map_role_created(self) -> None:
        """RBAC map contains role_created mapping."""
        assert RBAC_EVENT_TYPE_MAP["role_created"] == UnifiedAuditEventType.RBAC_ROLE_CREATED

    def test_rbac_map_permission_granted(self) -> None:
        """RBAC map contains permission_granted mapping."""
        assert RBAC_EVENT_TYPE_MAP["permission_granted"] == UnifiedAuditEventType.RBAC_PERMISSION_GRANTED

    def test_rbac_map_authorization_denied(self) -> None:
        """RBAC map contains authorization_denied mapping."""
        assert RBAC_EVENT_TYPE_MAP["authorization_denied"] == UnifiedAuditEventType.RBAC_AUTHORIZATION_DENIED

    def test_rbac_map_count(self) -> None:
        """RBAC map has expected number of entries."""
        assert len(RBAC_EVENT_TYPE_MAP) >= 12

    # ------------------------------------------------------------------
    # ENCRYPTION_EVENT_TYPE_MAP tests
    # ------------------------------------------------------------------

    def test_encryption_map_performed(self) -> None:
        """ENCRYPTION map contains encryption_performed mapping."""
        assert ENCRYPTION_EVENT_TYPE_MAP["encryption_performed"] == UnifiedAuditEventType.ENCRYPTION_PERFORMED

    def test_encryption_map_key_rotated(self) -> None:
        """ENCRYPTION map contains key_rotated mapping."""
        assert ENCRYPTION_EVENT_TYPE_MAP["key_rotated"] == UnifiedAuditEventType.ENCRYPTION_KEY_ROTATED

    def test_encryption_map_kms_error(self) -> None:
        """ENCRYPTION map contains kms_error mapping."""
        assert ENCRYPTION_EVENT_TYPE_MAP["kms_error"] == UnifiedAuditEventType.ENCRYPTION_KMS_ERROR

    def test_encryption_map_count(self) -> None:
        """ENCRYPTION map has expected number of entries."""
        assert len(ENCRYPTION_EVENT_TYPE_MAP) >= 18

    # ------------------------------------------------------------------
    # Map completeness tests
    # ------------------------------------------------------------------

    def test_all_maps_have_string_keys(self) -> None:
        """All map keys are strings."""
        for key in AUTH_EVENT_TYPE_MAP:
            assert isinstance(key, str)
        for key in RBAC_EVENT_TYPE_MAP:
            assert isinstance(key, str)
        for key in ENCRYPTION_EVENT_TYPE_MAP:
            assert isinstance(key, str)

    def test_all_maps_have_unified_values(self) -> None:
        """All map values are UnifiedAuditEventType instances."""
        for value in AUTH_EVENT_TYPE_MAP.values():
            assert isinstance(value, UnifiedAuditEventType)
        for value in RBAC_EVENT_TYPE_MAP.values():
            assert isinstance(value, UnifiedAuditEventType)
        for value in ENCRYPTION_EVENT_TYPE_MAP.values():
            assert isinstance(value, UnifiedAuditEventType)
