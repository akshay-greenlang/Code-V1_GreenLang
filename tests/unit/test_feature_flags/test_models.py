# -*- coding: utf-8 -*-
"""
Unit Tests for Feature Flag Models - INFRA-008

Tests all Pydantic v2 models in the feature flag system: FeatureFlag,
FlagType, FlagStatus, EvaluationContext, FlagEvaluationResult,
AuditLogEntry, FlagRule, FlagVariant, and FlagOverride.

Validates creation defaults, field constraints, validators, serialization,
and rejection of invalid inputs.
"""

import json
import re
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    EvaluationContext,
    FeatureFlag,
    FlagEvaluationResult,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagType,
    FlagVariant,
)


# ---------------------------------------------------------------------------
# FlagType Enum
# ---------------------------------------------------------------------------


class TestFlagType:
    """Tests for the FlagType enumeration."""

    def test_all_flag_types_present(self):
        """All seven flag types must be defined."""
        expected = {
            "BOOLEAN",
            "PERCENTAGE",
            "USER_LIST",
            "ENVIRONMENT",
            "SEGMENT",
            "SCHEDULED",
            "MULTIVARIATE",
        }
        actual = {member.name for member in FlagType}
        assert actual == expected

    def test_flag_type_values_are_lowercase(self):
        """Enum values must be lowercase strings."""
        for member in FlagType:
            assert member.value == member.name.lower()

    def test_flag_type_is_string_enum(self):
        """FlagType members are also strings (str, Enum)."""
        assert isinstance(FlagType.BOOLEAN, str)
        assert FlagType.BOOLEAN == "boolean"


# ---------------------------------------------------------------------------
# FlagStatus Enum
# ---------------------------------------------------------------------------


class TestFlagStatus:
    """Tests for the FlagStatus lifecycle enumeration."""

    def test_all_statuses_present(self):
        """All six lifecycle statuses must be defined."""
        expected = {
            "DRAFT",
            "ACTIVE",
            "ROLLED_OUT",
            "PERMANENT",
            "ARCHIVED",
            "KILLED",
        }
        actual = {member.name for member in FlagStatus}
        assert actual == expected

    def test_status_values_are_lowercase(self):
        """Enum values must be lowercase strings."""
        for member in FlagStatus:
            assert member.value == member.name.lower()


# ---------------------------------------------------------------------------
# FeatureFlag Model
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    """Tests for the core FeatureFlag Pydantic model."""

    def test_create_with_minimal_fields(self):
        """Creating a flag with only required fields uses correct defaults."""
        flag = FeatureFlag(key="ab-test.scope3", name="AB Test Scope 3")

        assert flag.key == "ab-test.scope3"
        assert flag.name == "AB Test Scope 3"
        assert flag.description == ""
        assert flag.flag_type == FlagType.BOOLEAN
        assert flag.status == FlagStatus.DRAFT
        assert flag.default_value is False
        assert flag.rollout_percentage == 0.0
        assert flag.environments == []
        assert flag.tags == []
        assert flag.owner == ""
        assert flag.metadata == {}
        assert flag.start_time is None
        assert flag.end_time is None
        assert flag.version == 1
        assert flag.created_at is not None
        assert flag.updated_at is not None

    def test_create_with_all_fields(self):
        """Creating a flag with all fields populated works correctly."""
        now = datetime.now(timezone.utc)
        later = now + timedelta(hours=1)
        flag = FeatureFlag(
            key="enable-scope3-calc",
            name="Enable Scope 3 Calculation",
            description="Rolling out new Scope 3 engine",
            flag_type=FlagType.PERCENTAGE,
            status=FlagStatus.ACTIVE,
            default_value=False,
            rollout_percentage=25.0,
            environments=["staging", "prod"],
            tags=["scope3", "platform"],
            owner="platform-team",
            metadata={"jira": "INFRA-008"},
            start_time=now,
            end_time=later,
            version=3,
        )
        assert flag.flag_type == FlagType.PERCENTAGE
        assert flag.rollout_percentage == 25.0
        assert flag.environments == ["staging", "prod"]
        assert flag.tags == ["scope3", "platform"]
        assert flag.version == 3

    def test_key_validation_rejects_uppercase(self):
        """Flag keys must be lowercase."""
        with pytest.raises(ValidationError):
            FeatureFlag(key="EnableScope3", name="Test")

    def test_key_validation_rejects_short_key(self):
        """Flag keys must be at least 2 characters."""
        with pytest.raises(ValidationError):
            FeatureFlag(key="a", name="Test")

    def test_key_validation_rejects_spaces(self):
        """Flag keys must not contain spaces."""
        with pytest.raises(ValidationError):
            FeatureFlag(key="enable scope3", name="Test")

    def test_key_validation_allows_dots_hyphens_underscores(self):
        """Flag keys may contain dots, hyphens, and underscores."""
        flag = FeatureFlag(key="enable-scope_3.calc", name="Test")
        assert flag.key == "enable-scope_3.calc"

    def test_rollout_percentage_lower_bound(self):
        """rollout_percentage must be >= 0.0."""
        with pytest.raises(ValidationError):
            FeatureFlag(
                key="test-flag",
                name="Test",
                rollout_percentage=-1.0,
            )

    def test_rollout_percentage_upper_bound(self):
        """rollout_percentage must be <= 100.0."""
        with pytest.raises(ValidationError):
            FeatureFlag(
                key="test-flag",
                name="Test",
                rollout_percentage=100.1,
            )

    def test_rollout_percentage_boundary_values(self):
        """rollout_percentage accepts 0.0 and 100.0 exactly."""
        flag_zero = FeatureFlag(key="test-zero", name="T", rollout_percentage=0.0)
        flag_full = FeatureFlag(key="test-full", name="T", rollout_percentage=100.0)
        assert flag_zero.rollout_percentage == 0.0
        assert flag_full.rollout_percentage == 100.0

    def test_environments_validation_rejects_invalid(self):
        """Invalid environment names are rejected."""
        with pytest.raises(ValidationError):
            FeatureFlag(key="test-flag", name="T", environments=["invalid-env"])

    def test_environments_normalized_to_lowercase(self):
        """Environment names are normalized to lowercase."""
        flag = FeatureFlag(key="test-flag", name="T", environments=["Staging", "PROD"])
        assert flag.environments == ["staging", "prod"]

    def test_tags_deduplicated_and_lowercased(self):
        """Duplicate tags are removed and all tags are lowercased."""
        flag = FeatureFlag(
            key="test-flag",
            name="T",
            tags=["Scope3", "scope3", "CBAM", "cbam"],
        )
        assert flag.tags == ["scope3", "cbam"]

    def test_scheduled_flag_requires_both_times(self):
        """SCHEDULED flags require both start_time and end_time."""
        with pytest.raises(ValidationError):
            FeatureFlag(
                key="test-scheduled",
                name="T",
                flag_type=FlagType.SCHEDULED,
                start_time=datetime.now(timezone.utc),
                # end_time missing
            )

    def test_scheduled_flag_start_must_be_before_end(self):
        """For SCHEDULED flags start_time must be before end_time."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            FeatureFlag(
                key="test-scheduled",
                name="T",
                flag_type=FlagType.SCHEDULED,
                start_time=now + timedelta(hours=1),
                end_time=now,
            )

    def test_scheduled_flag_valid_time_window(self):
        """A valid SCHEDULED flag with correct time window passes."""
        now = datetime.now(timezone.utc)
        flag = FeatureFlag(
            key="test-scheduled",
            name="T",
            flag_type=FlagType.SCHEDULED,
            start_time=now,
            end_time=now + timedelta(hours=2),
        )
        assert flag.start_time < flag.end_time

    def test_model_dump_returns_dict(self):
        """model_dump() produces a serializable dict."""
        flag = FeatureFlag(key="test-flag", name="T")
        data = flag.model_dump()
        assert isinstance(data, dict)
        assert data["key"] == "test-flag"
        assert data["flag_type"] == "boolean"
        assert data["status"] == "draft"

    def test_model_dump_json_returns_valid_json(self):
        """model_dump_json() produces valid JSON string."""
        flag = FeatureFlag(key="test-flag", name="T")
        json_str = flag.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["key"] == "test-flag"

    def test_extra_fields_forbidden(self):
        """Extra fields not in the model are rejected."""
        with pytest.raises(ValidationError):
            FeatureFlag(
                key="test-flag",
                name="T",
                unknown_field="value",
            )

    def test_version_must_be_positive(self):
        """version must be >= 1."""
        with pytest.raises(ValidationError):
            FeatureFlag(key="test-flag", name="T", version=0)


# ---------------------------------------------------------------------------
# FlagRule Model
# ---------------------------------------------------------------------------


class TestFlagRule:
    """Tests for the FlagRule targeting rule model."""

    def test_create_rule_with_defaults(self):
        """Creating a rule with required fields uses correct defaults."""
        rule = FlagRule(flag_key="test-flag", rule_type="user_list")
        assert rule.flag_key == "test-flag"
        assert rule.rule_type == "user_list"
        assert rule.priority == 100
        assert rule.conditions == {}
        assert rule.enabled is True
        assert rule.rule_id is not None

    def test_create_rule_with_conditions(self):
        """A rule with full conditions is stored correctly."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="segment",
            priority=10,
            conditions={
                "conditions": [
                    {"attribute": "user_attributes.plan", "operator": "eq", "value": "enterprise"},
                ]
            },
            enabled=True,
        )
        assert rule.priority == 10
        assert len(rule.conditions["conditions"]) == 1

    def test_rule_flag_key_validated(self):
        """Rule flag_key must match the standard pattern."""
        with pytest.raises(ValidationError):
            FlagRule(flag_key="INVALID KEY", rule_type="user_list")


# ---------------------------------------------------------------------------
# FlagVariant Model
# ---------------------------------------------------------------------------


class TestFlagVariant:
    """Tests for the FlagVariant multivariate model."""

    def test_create_variant(self):
        """Creating a variant with all fields works correctly."""
        variant = FlagVariant(
            variant_key="control",
            flag_key="ab-test.ui",
            variant_value={"color": "blue"},
            weight=50.0,
            description="Control group",
        )
        assert variant.variant_key == "control"
        assert variant.weight == 50.0
        assert variant.variant_value == {"color": "blue"}

    def test_variant_key_lowercased(self):
        """Variant keys are normalized to lowercase."""
        variant = FlagVariant(
            variant_key="Treatment",
            flag_key="ab-test.ui",
        )
        assert variant.variant_key == "treatment"

    def test_variant_weight_bounds(self):
        """Weight must be between 0.0 and 100.0."""
        with pytest.raises(ValidationError):
            FlagVariant(variant_key="bad", flag_key="ab-test.ui", weight=101.0)

    def test_variant_flag_key_validated(self):
        """Variant flag_key must match the standard pattern."""
        with pytest.raises(ValidationError):
            FlagVariant(variant_key="control", flag_key="INVALID KEY")


# ---------------------------------------------------------------------------
# FlagOverride Model
# ---------------------------------------------------------------------------


class TestFlagOverride:
    """Tests for the FlagOverride scoped override model."""

    def test_create_user_override(self):
        """Creating a user-scoped override works correctly."""
        override = FlagOverride(
            flag_key="test-flag",
            scope_type="user",
            scope_value="user-42",
            enabled=True,
        )
        assert override.scope_type == "user"
        assert override.scope_value == "user-42"
        assert override.enabled is True
        assert override.variant_key is None
        assert override.expires_at is None

    def test_create_tenant_override_with_expiry(self):
        """A tenant override with expiration is stored correctly."""
        future = datetime.now(timezone.utc) + timedelta(days=7)
        override = FlagOverride(
            flag_key="test-flag",
            scope_type="tenant",
            scope_value="tenant-acme",
            enabled=False,
            expires_at=future,
            created_by="admin@greenlang.io",
        )
        assert override.scope_type == "tenant"
        assert override.expires_at is not None

    def test_scope_type_validation(self):
        """Only user, tenant, segment, and environment scope types are allowed."""
        with pytest.raises(ValidationError):
            FlagOverride(
                flag_key="test-flag",
                scope_type="invalid",
                scope_value="v",
            )

    def test_scope_type_normalized_to_lowercase(self):
        """scope_type is normalized to lowercase."""
        override = FlagOverride(
            flag_key="test-flag",
            scope_type="ENVIRONMENT",
            scope_value="staging",
        )
        assert override.scope_type == "environment"


# ---------------------------------------------------------------------------
# EvaluationContext Model
# ---------------------------------------------------------------------------


class TestEvaluationContext:
    """Tests for the EvaluationContext model."""

    def test_create_with_defaults(self):
        """Creating a context with defaults sets environment to 'dev'."""
        ctx = EvaluationContext()
        assert ctx.user_id is None
        assert ctx.tenant_id is None
        assert ctx.environment == "dev"
        assert ctx.user_segments == []
        assert ctx.user_attributes == {}
        assert ctx.request_id is not None

    def test_create_with_all_fields(self):
        """Creating a fully populated context works correctly."""
        ctx = EvaluationContext(
            user_id="user-42",
            tenant_id="tenant-acme",
            environment="Prod",
            user_segments=["Enterprise", "early_adopter"],
            user_attributes={"plan_type": "enterprise", "seats": 100},
        )
        assert ctx.user_id == "user-42"
        assert ctx.tenant_id == "tenant-acme"
        assert ctx.environment == "prod"  # normalized
        assert ctx.user_segments == ["enterprise", "early_adopter"]  # normalized

    def test_identity_key_prefers_user_id(self):
        """identity_key returns user_id when available."""
        ctx = EvaluationContext(
            user_id="user-42",
            tenant_id="tenant-acme",
        )
        assert ctx.identity_key == "user-42"

    def test_identity_key_falls_back_to_tenant_id(self):
        """identity_key returns tenant_id when user_id is None."""
        ctx = EvaluationContext(tenant_id="tenant-acme")
        assert ctx.identity_key == "tenant-acme"

    def test_identity_key_falls_back_to_request_id(self):
        """identity_key returns request_id when both user and tenant are None."""
        ctx = EvaluationContext()
        assert ctx.identity_key == ctx.request_id


# ---------------------------------------------------------------------------
# FlagEvaluationResult Model
# ---------------------------------------------------------------------------


class TestFlagEvaluationResult:
    """Tests for the FlagEvaluationResult model."""

    def test_create_basic_result(self):
        """Creating a result with required fields uses correct defaults."""
        result = FlagEvaluationResult(flag_key="test-flag", enabled=True)
        assert result.flag_key == "test-flag"
        assert result.enabled is True
        assert result.reason == "default"
        assert result.rule_id is None
        assert result.variant_key is None
        assert result.metadata == {}
        assert result.cache_layer == "default"
        assert result.duration_us == 0

    def test_create_result_with_all_fields(self):
        """Creating a fully populated result works correctly."""
        result = FlagEvaluationResult(
            flag_key="test-flag",
            enabled=True,
            reason="rule:r-1",
            rule_id="r-1",
            variant_key="treatment-a",
            metadata={"experiment": "exp-001"},
            cache_layer="l1",
            duration_us=42,
        )
        assert result.reason == "rule:r-1"
        assert result.rule_id == "r-1"
        assert result.variant_key == "treatment-a"
        assert result.duration_us == 42

    def test_duration_cannot_be_negative(self):
        """duration_us must be >= 0."""
        with pytest.raises(ValidationError):
            FlagEvaluationResult(
                flag_key="test-flag",
                enabled=False,
                duration_us=-1,
            )


# ---------------------------------------------------------------------------
# AuditLogEntry Model
# ---------------------------------------------------------------------------


class TestAuditLogEntry:
    """Tests for the immutable AuditLogEntry model."""

    def test_create_audit_entry(self):
        """Creating an audit entry with required fields uses correct defaults."""
        entry = AuditLogEntry(
            flag_key="test-flag",
            action="created",
        )
        assert entry.flag_key == "test-flag"
        assert entry.action == "created"
        assert entry.old_value == {}
        assert entry.new_value == {}
        assert entry.changed_by == ""
        assert entry.change_reason == ""
        assert entry.ip_address is None
        assert entry.created_at is not None

    def test_audit_entry_action_normalized(self):
        """Action names are normalized to lowercase."""
        entry = AuditLogEntry(flag_key="test-flag", action="KILLED")
        assert entry.action == "killed"

    def test_audit_entry_invalid_action_rejected(self):
        """Invalid action names are rejected."""
        with pytest.raises(ValidationError):
            AuditLogEntry(flag_key="test-flag", action="invalid_action")

    def test_audit_entry_all_valid_actions(self):
        """All documented valid actions are accepted."""
        valid_actions = [
            "created", "updated", "deleted", "enabled", "disabled",
            "killed", "archived", "rolled_out", "promoted",
            "rule_added", "rule_removed", "rule_updated",
            "variant_added", "variant_removed", "variant_updated",
            "override_added", "override_removed", "override_updated",
        ]
        for action in valid_actions:
            entry = AuditLogEntry(flag_key="test-flag", action=action)
            assert entry.action == action

    def test_audit_entry_is_frozen(self):
        """AuditLogEntry is immutable (frozen=True)."""
        entry = AuditLogEntry(flag_key="test-flag", action="created")
        with pytest.raises(ValidationError):
            entry.action = "updated"

    def test_audit_entry_ip_address_validation(self):
        """IP address must be between 3 and 45 characters."""
        entry = AuditLogEntry(
            flag_key="test-flag",
            action="created",
            ip_address="192.168.1.1",
        )
        assert entry.ip_address == "192.168.1.1"

    def test_audit_entry_invalid_ip_rejected(self):
        """IP addresses that are too short are rejected."""
        with pytest.raises(ValidationError):
            AuditLogEntry(
                flag_key="test-flag",
                action="created",
                ip_address="ab",
            )

    def test_audit_entry_serialization(self):
        """Audit entry serializes to JSON correctly."""
        entry = AuditLogEntry(
            flag_key="test-flag",
            action="created",
            changed_by="admin",
            change_reason="Initial rollout",
        )
        data = json.loads(entry.model_dump_json())
        assert data["flag_key"] == "test-flag"
        assert data["action"] == "created"
        assert data["changed_by"] == "admin"
