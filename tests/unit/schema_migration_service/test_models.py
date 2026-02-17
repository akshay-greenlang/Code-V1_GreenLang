# -*- coding: utf-8 -*-
"""
Unit Tests for Schema Migration Agent Models - AGENT-DATA-017

Tests all enumerations (14), SDK models (16), request models (8), utility
helpers, constants, and re-exported Layer 1 symbols from
``greenlang.schema_migration.models``.

Target: 150+ tests, 85%+ coverage of greenlang.schema_migration.models

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

import pydantic
import pytest
from pydantic import ValidationError

from greenlang.schema_migration.models import (
    # Layer 1 re-exports
    QualityDimension,
    RuleType,
    # Availability flags
    _DQ_AVAILABLE,
    _SC_AVAILABLE,
    # Constants
    VERSION,
    MAX_SCHEMAS_PER_NAMESPACE,
    MAX_FIELDS_PER_SCHEMA,
    MAX_MIGRATION_STEPS,
    SUPPORTED_SCHEMA_TYPES,
    DEFAULT_COMPATIBILITY_RULES,
    CHANGE_SEVERITY_ORDER,
    DEFAULT_MIGRATION_BATCH_SIZE,
    AUTO_ACCEPT_CONFIDENCE_THRESHOLD,
    MAX_DRIFT_EVENTS_PER_VERSION,
    # Helper
    _utcnow,
    # Enumerations (14)
    SchemaType,
    SchemaStatus,
    ChangeType,
    ChangeSeverity,
    CompatibilityLevel,
    MigrationPlanStatus,
    ExecutionStatus,
    RollbackType,
    RollbackStatus,
    DriftType,
    DriftSeverity,
    MappingType,
    EffortLevel,
    TransformationType,
    # SDK models (16)
    SchemaDefinition,
    SchemaVersion,
    SchemaChange,
    CompatibilityResult,
    MigrationStep,
    MigrationPlan,
    MigrationExecution,
    RollbackRecord,
    FieldMapping,
    DriftEvent,
    AuditEntry,
    SchemaGroup,
    MigrationReport,
    SchemaStatistics,
    VersionComparison,
    PipelineResult,
    # Request models (8)
    RegisterSchemaRequest,
    UpdateSchemaRequest,
    CreateVersionRequest,
    DetectChangesRequest,
    CheckCompatibilityRequest,
    CreatePlanRequest,
    ExecuteMigrationRequest,
    RunPipelineRequest,
)


# ---------------------------------------------------------------------------
# Restore strict mode for model tests
# ---------------------------------------------------------------------------


def _set_model_extra(mode: str) -> None:
    """Set model_config extra to the given mode for all SM BaseModel subclasses."""
    from greenlang.schema_migration import models as sm_models

    for name in dir(sm_models):
        obj = getattr(sm_models, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, pydantic.BaseModel)
            and obj is not pydantic.BaseModel
        ):
            cfg = getattr(obj, "model_config", {})
            if isinstance(cfg, dict):
                obj.model_config = {**cfg, "extra": mode}
                obj.model_rebuild(force=True)


@pytest.fixture(autouse=True, scope="module")
def _strict_models():
    """Restore strict mode (extra='forbid') for all model tests in this module."""
    _set_model_extra("forbid")
    yield
    _set_model_extra("ignore")


# ============================================================================
# Enum Tests (14 enums)
# ============================================================================


class TestSchemaTypeEnum:
    """SchemaType enum tests."""

    def test_member_count(self):
        assert len(SchemaType) == 3

    def test_json_schema_value(self):
        assert SchemaType.JSON_SCHEMA.value == "json_schema"

    def test_avro_value(self):
        assert SchemaType.AVRO.value == "avro"

    def test_protobuf_value(self):
        assert SchemaType.PROTOBUF.value == "protobuf"

    def test_is_str_enum(self):
        assert isinstance(SchemaType.JSON_SCHEMA, str)

    def test_is_enum(self):
        assert issubclass(SchemaType, Enum)

    def test_lookup_by_value(self):
        assert SchemaType("json_schema") == SchemaType.JSON_SCHEMA


class TestSchemaStatusEnum:
    """SchemaStatus enum tests."""

    def test_member_count(self):
        assert len(SchemaStatus) == 4

    @pytest.mark.parametrize("member,value", [
        ("DRAFT", "draft"),
        ("ACTIVE", "active"),
        ("DEPRECATED", "deprecated"),
        ("ARCHIVED", "archived"),
    ])
    def test_member_value(self, member, value):
        assert SchemaStatus[member].value == value

    def test_is_str_enum(self):
        assert isinstance(SchemaStatus.DRAFT, str)


class TestChangeTypeEnum:
    """ChangeType enum tests."""

    def test_member_count(self):
        assert len(ChangeType) == 8

    @pytest.mark.parametrize("member,value", [
        ("ADDED", "added"),
        ("REMOVED", "removed"),
        ("RENAMED", "renamed"),
        ("RETYPED", "retyped"),
        ("REORDERED", "reordered"),
        ("CONSTRAINT_CHANGED", "constraint_changed"),
        ("ENUM_CHANGED", "enum_changed"),
        ("DEFAULT_CHANGED", "default_changed"),
    ])
    def test_member_value(self, member, value):
        assert ChangeType[member].value == value

    def test_is_str_enum(self):
        assert isinstance(ChangeType.ADDED, str)


class TestChangeSeverityEnum:
    """ChangeSeverity enum tests."""

    def test_member_count(self):
        assert len(ChangeSeverity) == 3

    @pytest.mark.parametrize("member,value", [
        ("BREAKING", "breaking"),
        ("NON_BREAKING", "non_breaking"),
        ("COSMETIC", "cosmetic"),
    ])
    def test_member_value(self, member, value):
        assert ChangeSeverity[member].value == value

    def test_is_str_enum(self):
        assert isinstance(ChangeSeverity.BREAKING, str)


class TestCompatibilityLevelEnum:
    """CompatibilityLevel enum tests."""

    def test_member_count(self):
        assert len(CompatibilityLevel) == 5

    @pytest.mark.parametrize("member,value", [
        ("BACKWARD", "backward"),
        ("FORWARD", "forward"),
        ("FULL", "full"),
        ("BREAKING", "breaking"),
        ("NONE", "none"),
    ])
    def test_member_value(self, member, value):
        assert CompatibilityLevel[member].value == value

    def test_is_str_enum(self):
        assert isinstance(CompatibilityLevel.BACKWARD, str)


class TestMigrationPlanStatusEnum:
    """MigrationPlanStatus enum tests."""

    def test_member_count(self):
        assert len(MigrationPlanStatus) == 7

    @pytest.mark.parametrize("member,value", [
        ("DRAFT", "draft"),
        ("VALIDATED", "validated"),
        ("APPROVED", "approved"),
        ("EXECUTING", "executing"),
        ("COMPLETED", "completed"),
        ("FAILED", "failed"),
        ("ROLLED_BACK", "rolled_back"),
    ])
    def test_member_value(self, member, value):
        assert MigrationPlanStatus[member].value == value

    def test_is_str_enum(self):
        assert isinstance(MigrationPlanStatus.DRAFT, str)


class TestExecutionStatusEnum:
    """ExecutionStatus enum tests."""

    def test_member_count(self):
        assert len(ExecutionStatus) == 6

    @pytest.mark.parametrize("member,value", [
        ("PENDING", "pending"),
        ("RUNNING", "running"),
        ("COMPLETED", "completed"),
        ("FAILED", "failed"),
        ("ROLLED_BACK", "rolled_back"),
        ("TIMED_OUT", "timed_out"),
    ])
    def test_member_value(self, member, value):
        assert ExecutionStatus[member].value == value

    def test_is_str_enum(self):
        assert isinstance(ExecutionStatus.PENDING, str)


class TestRollbackTypeEnum:
    """RollbackType enum tests."""

    def test_member_count(self):
        assert len(RollbackType) == 3

    @pytest.mark.parametrize("member,value", [
        ("FULL", "full"),
        ("PARTIAL", "partial"),
        ("CHECKPOINT", "checkpoint"),
    ])
    def test_member_value(self, member, value):
        assert RollbackType[member].value == value

    def test_is_str_enum(self):
        assert isinstance(RollbackType.FULL, str)


class TestRollbackStatusEnum:
    """RollbackStatus enum tests."""

    def test_member_count(self):
        assert len(RollbackStatus) == 4

    @pytest.mark.parametrize("member,value", [
        ("PENDING", "pending"),
        ("RUNNING", "running"),
        ("COMPLETED", "completed"),
        ("FAILED", "failed"),
    ])
    def test_member_value(self, member, value):
        assert RollbackStatus[member].value == value

    def test_is_str_enum(self):
        assert isinstance(RollbackStatus.PENDING, str)


class TestDriftTypeEnum:
    """DriftType enum tests."""

    def test_member_count(self):
        assert len(DriftType) == 5

    @pytest.mark.parametrize("member,value", [
        ("MISSING_FIELD", "missing_field"),
        ("EXTRA_FIELD", "extra_field"),
        ("TYPE_MISMATCH", "type_mismatch"),
        ("CONSTRAINT_VIOLATION", "constraint_violation"),
        ("ENUM_VIOLATION", "enum_violation"),
    ])
    def test_member_value(self, member, value):
        assert DriftType[member].value == value

    def test_is_str_enum(self):
        assert isinstance(DriftType.MISSING_FIELD, str)


class TestDriftSeverityEnum:
    """DriftSeverity enum tests."""

    def test_member_count(self):
        assert len(DriftSeverity) == 4

    @pytest.mark.parametrize("member,value", [
        ("LOW", "low"),
        ("MEDIUM", "medium"),
        ("HIGH", "high"),
        ("CRITICAL", "critical"),
    ])
    def test_member_value(self, member, value):
        assert DriftSeverity[member].value == value

    def test_is_str_enum(self):
        assert isinstance(DriftSeverity.LOW, str)


class TestMappingTypeEnum:
    """MappingType enum tests."""

    def test_member_count(self):
        assert len(MappingType) == 4

    @pytest.mark.parametrize("member,value", [
        ("EXACT", "exact"),
        ("ALIAS", "alias"),
        ("COMPUTED", "computed"),
        ("MANUAL", "manual"),
    ])
    def test_member_value(self, member, value):
        assert MappingType[member].value == value

    def test_is_str_enum(self):
        assert isinstance(MappingType.EXACT, str)


class TestEffortLevelEnum:
    """EffortLevel enum tests."""

    def test_member_count(self):
        assert len(EffortLevel) == 4

    @pytest.mark.parametrize("member,value", [
        ("LOW", "low"),
        ("MEDIUM", "medium"),
        ("HIGH", "high"),
        ("CRITICAL", "critical"),
    ])
    def test_member_value(self, member, value):
        assert EffortLevel[member].value == value

    def test_is_str_enum(self):
        assert isinstance(EffortLevel.LOW, str)


class TestTransformationTypeEnum:
    """TransformationType enum tests."""

    def test_member_count(self):
        assert len(TransformationType) == 8

    @pytest.mark.parametrize("member,value", [
        ("RENAME_FIELD", "rename_field"),
        ("CAST_TYPE", "cast_type"),
        ("SET_DEFAULT", "set_default"),
        ("COMPUTE_FIELD", "compute_field"),
        ("SPLIT_FIELD", "split_field"),
        ("MERGE_FIELDS", "merge_fields"),
        ("REMOVE_FIELD", "remove_field"),
        ("ADD_FIELD", "add_field"),
    ])
    def test_member_value(self, member, value):
        assert TransformationType[member].value == value

    def test_is_str_enum(self):
        assert isinstance(TransformationType.RENAME_FIELD, str)


# ============================================================================
# Helper Tests
# ============================================================================


class TestUtcnowHelper:
    """_utcnow helper function tests."""

    def test_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_is_utc(self):
        result = _utcnow()
        assert result.tzinfo == timezone.utc

    def test_microseconds_zeroed(self):
        result = _utcnow()
        assert result.microsecond == 0


# ============================================================================
# SDK Model Tests (16 models)
# ============================================================================


class TestSchemaDefinitionModel:
    """SchemaDefinition model tests."""

    def test_minimal_creation(self):
        sd = SchemaDefinition(namespace="greenlang.test", name="test_schema")
        assert sd.namespace == "greenlang.test"
        assert sd.name == "test_schema"

    def test_default_id_is_uuid(self):
        sd = SchemaDefinition(namespace="ns", name="schema")
        uuid.UUID(sd.id)  # should not raise

    def test_default_schema_type(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert sd.schema_type == SchemaType.JSON_SCHEMA

    def test_default_status(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert sd.status == SchemaStatus.DRAFT

    def test_default_owner_empty(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert sd.owner == ""

    def test_default_tags_empty(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert sd.tags == {}

    def test_default_description_empty(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert sd.description == ""

    def test_default_metadata_empty(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert sd.metadata == {}

    def test_default_definition_json_empty(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert sd.definition_json == {}

    def test_default_created_at_is_datetime(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        assert isinstance(sd.created_at, datetime)

    def test_empty_namespace_raises(self):
        with pytest.raises(ValidationError):
            SchemaDefinition(namespace="", name="s")

    def test_whitespace_namespace_raises(self):
        with pytest.raises(ValidationError):
            SchemaDefinition(namespace="   ", name="s")

    def test_invalid_namespace_chars_raises(self):
        with pytest.raises(ValidationError):
            SchemaDefinition(namespace="ns@invalid!", name="s")

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            SchemaDefinition(namespace="ns", name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError):
            SchemaDefinition(namespace="ns", name="   ")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaDefinition(namespace="ns", name="s", unknown_field=42)

    def test_full_creation(self):
        sd = SchemaDefinition(
            namespace="greenlang.emissions",
            name="emission_factors",
            schema_type=SchemaType.AVRO,
            owner="platform-team",
            tags={"domain": "emissions"},
            status=SchemaStatus.ACTIVE,
            description="Emission factor schema",
            metadata={"version_policy": "strict"},
            definition_json={"type": "record", "name": "EF"},
        )
        assert sd.schema_type == SchemaType.AVRO
        assert sd.status == SchemaStatus.ACTIVE
        assert sd.tags["domain"] == "emissions"

    def test_model_dump_keys(self):
        sd = SchemaDefinition(namespace="ns", name="s")
        d = sd.model_dump()
        assert "namespace" in d
        assert "name" in d
        assert "schema_type" in d

    def test_namespace_with_dots_ok(self):
        sd = SchemaDefinition(namespace="com.greenlang.v2", name="s")
        assert sd.namespace == "com.greenlang.v2"

    def test_namespace_with_hyphens_ok(self):
        sd = SchemaDefinition(namespace="my-namespace", name="s")
        assert sd.namespace == "my-namespace"


class TestSchemaVersionModel:
    """SchemaVersion model tests."""

    def test_minimal_creation(self):
        sv = SchemaVersion(schema_id="abc-123", version="1.0.0")
        assert sv.schema_id == "abc-123"
        assert sv.version == "1.0.0"

    def test_default_id_is_uuid(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0")
        uuid.UUID(sv.id)

    def test_default_definition_json_empty(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0")
        assert sv.definition_json == {}

    def test_default_changelog_empty(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0")
        assert sv.changelog == ""

    def test_default_is_deprecated_false(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0")
        assert sv.is_deprecated is False

    def test_default_deprecated_at_none(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0")
        assert sv.deprecated_at is None

    def test_default_created_at_is_datetime(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0")
        assert isinstance(sv.created_at, datetime)

    def test_empty_schema_id_raises(self):
        with pytest.raises(ValidationError):
            SchemaVersion(schema_id="", version="1.0.0")

    def test_empty_version_raises(self):
        with pytest.raises(ValidationError):
            SchemaVersion(schema_id="abc", version="")

    def test_invalid_semver_raises(self):
        with pytest.raises(ValidationError):
            SchemaVersion(schema_id="abc", version="not-semver")

    def test_valid_semver_prerelease(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0-alpha.1")
        assert sv.version == "1.0.0-alpha.1"

    def test_valid_semver_build_metadata(self):
        sv = SchemaVersion(schema_id="abc", version="1.0.0+build.123")
        assert sv.version == "1.0.0+build.123"

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaVersion(schema_id="abc", version="1.0.0", foo="bar")


class TestSchemaChangeModel:
    """SchemaChange model tests."""

    def test_minimal_creation(self):
        sc = SchemaChange(
            source_version_id="v1",
            target_version_id="v2",
            change_type=ChangeType.ADDED,
        )
        assert sc.change_type == ChangeType.ADDED

    def test_default_severity(self):
        sc = SchemaChange(
            source_version_id="v1",
            target_version_id="v2",
            change_type=ChangeType.REMOVED,
        )
        assert sc.severity == ChangeSeverity.NON_BREAKING

    def test_default_field_path_empty(self):
        sc = SchemaChange(
            source_version_id="v1",
            target_version_id="v2",
            change_type=ChangeType.RENAMED,
        )
        assert sc.field_path == ""

    def test_empty_source_version_id_raises(self):
        with pytest.raises(ValidationError):
            SchemaChange(
                source_version_id="",
                target_version_id="v2",
                change_type=ChangeType.ADDED,
            )

    def test_empty_target_version_id_raises(self):
        with pytest.raises(ValidationError):
            SchemaChange(
                source_version_id="v1",
                target_version_id="",
                change_type=ChangeType.ADDED,
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaChange(
                source_version_id="v1",
                target_version_id="v2",
                change_type=ChangeType.ADDED,
                extra="x",
            )

    def test_full_creation(self):
        sc = SchemaChange(
            source_version_id="v1",
            target_version_id="v2",
            change_type=ChangeType.RETYPED,
            field_path="user.age",
            old_value="integer",
            new_value="string",
            severity=ChangeSeverity.BREAKING,
            description="Changed age from integer to string",
        )
        assert sc.severity == ChangeSeverity.BREAKING
        assert sc.field_path == "user.age"


class TestCompatibilityResultModel:
    """CompatibilityResult model tests."""

    def test_minimal_creation(self):
        cr = CompatibilityResult(
            source_version_id="v1",
            target_version_id="v2",
        )
        assert cr.compatibility_level == CompatibilityLevel.NONE

    def test_default_issues_empty(self):
        cr = CompatibilityResult(
            source_version_id="v1",
            target_version_id="v2",
        )
        assert cr.issues == []

    def test_default_recommendations_empty(self):
        cr = CompatibilityResult(
            source_version_id="v1",
            target_version_id="v2",
        )
        assert cr.recommendations == []

    def test_compatible_result(self):
        cr = CompatibilityResult(
            source_version_id="v1",
            target_version_id="v2",
            compatibility_level=CompatibilityLevel.FULL,
        )
        assert cr.compatibility_level == CompatibilityLevel.FULL

    def test_incompatible_result(self):
        cr = CompatibilityResult(
            source_version_id="v1",
            target_version_id="v2",
            compatibility_level=CompatibilityLevel.BREAKING,
            issues=["Removed required field 'name'"],
        )
        assert len(cr.issues) == 1

    def test_empty_source_version_id_raises(self):
        with pytest.raises(ValidationError):
            CompatibilityResult(source_version_id="", target_version_id="v2")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            CompatibilityResult(
                source_version_id="v1",
                target_version_id="v2",
                extra=1,
            )


class TestMigrationStepModel:
    """MigrationStep model tests."""

    def test_minimal_creation(self):
        ms = MigrationStep(
            step_number=1,
            operation=TransformationType.ADD_FIELD,
        )
        assert ms.step_number == 1
        assert ms.operation == TransformationType.ADD_FIELD

    def test_default_reversible(self):
        ms = MigrationStep(
            step_number=1,
            operation=TransformationType.RENAME_FIELD,
        )
        assert ms.reversible is True

    def test_default_parameters_empty(self):
        ms = MigrationStep(
            step_number=1,
            operation=TransformationType.CAST_TYPE,
        )
        assert ms.parameters == {}

    def test_zero_step_number_raises(self):
        with pytest.raises(ValidationError):
            MigrationStep(
                step_number=0,
                operation=TransformationType.ADD_FIELD,
            )

    def test_negative_step_number_raises(self):
        with pytest.raises(ValidationError):
            MigrationStep(
                step_number=-1,
                operation=TransformationType.ADD_FIELD,
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            MigrationStep(
                step_number=1,
                operation=TransformationType.ADD_FIELD,
                extra="x",
            )

    def test_all_operation_types(self):
        for op in TransformationType:
            ms = MigrationStep(step_number=1, operation=op)
            assert ms.operation == op


class TestMigrationPlanModel:
    """MigrationPlan model tests."""

    def test_minimal_creation(self):
        mp = MigrationPlan(
            source_schema_id="s1",
            target_schema_id="s2",
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert mp.source_schema_id == "s1"

    def test_default_status(self):
        mp = MigrationPlan(
            source_schema_id="s1",
            target_schema_id="s2",
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert mp.status == MigrationPlanStatus.DRAFT

    def test_default_steps_empty(self):
        mp = MigrationPlan(
            source_schema_id="s1",
            target_schema_id="s2",
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert mp.steps == []

    def test_default_estimated_effort(self):
        mp = MigrationPlan(
            source_schema_id="s1",
            target_schema_id="s2",
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert mp.estimated_effort == EffortLevel.MEDIUM

    def test_empty_source_schema_id_raises(self):
        with pytest.raises(ValidationError):
            MigrationPlan(
                source_schema_id="",
                target_schema_id="s2",
                source_version="1.0.0",
                target_version="2.0.0",
            )

    def test_invalid_source_version_raises(self):
        with pytest.raises(ValidationError):
            MigrationPlan(
                source_schema_id="s1",
                target_schema_id="s2",
                source_version="not-semver",
                target_version="2.0.0",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            MigrationPlan(
                source_schema_id="s1",
                target_schema_id="s2",
                source_version="1.0.0",
                target_version="2.0.0",
                extra=1,
            )


class TestMigrationExecutionModel:
    """MigrationExecution model tests."""

    def test_minimal_creation(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.plan_id == "plan-123"

    def test_default_status(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.status == ExecutionStatus.PENDING

    def test_default_records_processed(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.records_processed == 0

    def test_default_records_failed(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.records_failed == 0

    def test_default_records_skipped(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.records_skipped == 0

    def test_default_current_step(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.current_step == 0

    def test_default_completed_at_none(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.completed_at is None

    def test_default_error_details_none(self):
        me = MigrationExecution(plan_id="plan-123")
        assert me.error_details is None

    def test_empty_plan_id_raises(self):
        with pytest.raises(ValidationError):
            MigrationExecution(plan_id="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            MigrationExecution(plan_id="p", extra=1)

    def test_progress_tracking(self):
        me = MigrationExecution(
            plan_id="plan-1",
            status=ExecutionStatus.RUNNING,
            current_step=3,
            total_steps=5,
            records_processed=500,
            records_failed=2,
            records_skipped=10,
        )
        assert me.current_step == 3
        assert me.total_steps == 5


class TestRollbackRecordModel:
    """RollbackRecord model tests."""

    def test_minimal_creation(self):
        rr = RollbackRecord(execution_id="exec-123")
        assert rr.execution_id == "exec-123"

    def test_default_rollback_type(self):
        rr = RollbackRecord(execution_id="exec-123")
        assert rr.rollback_type == RollbackType.FULL

    def test_default_status(self):
        rr = RollbackRecord(execution_id="exec-123")
        assert rr.status == RollbackStatus.PENDING

    def test_default_records_reverted(self):
        rr = RollbackRecord(execution_id="exec-123")
        assert rr.records_reverted == 0

    def test_empty_execution_id_raises(self):
        with pytest.raises(ValidationError):
            RollbackRecord(execution_id="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            RollbackRecord(execution_id="e", extra=1)

    def test_partial_rollback(self):
        rr = RollbackRecord(
            execution_id="exec-1",
            rollback_type=RollbackType.PARTIAL,
            rolled_back_to_step=3,
            records_reverted=100,
        )
        assert rr.rollback_type == RollbackType.PARTIAL
        assert rr.rolled_back_to_step == 3


class TestFieldMappingModel:
    """FieldMapping model tests."""

    def test_minimal_creation(self):
        fm = FieldMapping(
            source_schema_id="s1",
            target_schema_id="s2",
            source_field="name",
            target_field="full_name",
        )
        assert fm.source_field == "name"

    def test_default_confidence(self):
        fm = FieldMapping(
            source_schema_id="s1",
            target_schema_id="s2",
            source_field="f1",
            target_field="f2",
        )
        assert fm.confidence == pytest.approx(1.0)

    def test_default_mapping_type(self):
        fm = FieldMapping(
            source_schema_id="s1",
            target_schema_id="s2",
            source_field="f1",
            target_field="f2",
        )
        assert fm.mapping_type == MappingType.EXACT

    def test_confidence_boundary_zero(self):
        fm = FieldMapping(
            source_schema_id="s1",
            target_schema_id="s2",
            source_field="f1",
            target_field="f2",
            confidence=0.0,
        )
        assert fm.confidence == pytest.approx(0.0)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            FieldMapping(
                source_schema_id="s1",
                target_schema_id="s2",
                source_field="f1",
                target_field="f2",
                confidence=1.1,
            )

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            FieldMapping(
                source_schema_id="s1",
                target_schema_id="s2",
                source_field="f1",
                target_field="f2",
                confidence=-0.1,
            )

    def test_empty_source_field_raises(self):
        with pytest.raises(ValidationError):
            FieldMapping(
                source_schema_id="s1",
                target_schema_id="s2",
                source_field="",
                target_field="f2",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            FieldMapping(
                source_schema_id="s1",
                target_schema_id="s2",
                source_field="f1",
                target_field="f2",
                extra=1,
            )


class TestDriftEventModel:
    """DriftEvent model tests."""

    def test_minimal_creation(self):
        de = DriftEvent(
            schema_id="s1",
            version_id="v1",
            dataset_id="ds1",
            drift_type=DriftType.MISSING_FIELD,
        )
        assert de.drift_type == DriftType.MISSING_FIELD

    def test_default_severity(self):
        de = DriftEvent(
            schema_id="s1",
            version_id="v1",
            dataset_id="ds1",
            drift_type=DriftType.EXTRA_FIELD,
        )
        assert de.severity == DriftSeverity.MEDIUM

    def test_default_sample_count(self):
        de = DriftEvent(
            schema_id="s1",
            version_id="v1",
            dataset_id="ds1",
            drift_type=DriftType.TYPE_MISMATCH,
        )
        assert de.sample_count == 1

    def test_empty_schema_id_raises(self):
        with pytest.raises(ValidationError):
            DriftEvent(
                schema_id="",
                version_id="v1",
                dataset_id="ds1",
                drift_type=DriftType.MISSING_FIELD,
            )

    def test_empty_version_id_raises(self):
        with pytest.raises(ValidationError):
            DriftEvent(
                schema_id="s1",
                version_id="",
                dataset_id="ds1",
                drift_type=DriftType.MISSING_FIELD,
            )

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValidationError):
            DriftEvent(
                schema_id="s1",
                version_id="v1",
                dataset_id="",
                drift_type=DriftType.MISSING_FIELD,
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            DriftEvent(
                schema_id="s1",
                version_id="v1",
                dataset_id="ds1",
                drift_type=DriftType.MISSING_FIELD,
                extra=1,
            )

    def test_all_drift_types(self):
        for dt in DriftType:
            de = DriftEvent(
                schema_id="s1",
                version_id="v1",
                dataset_id="ds1",
                drift_type=dt,
            )
            assert de.drift_type == dt


class TestAuditEntryModel:
    """AuditEntry model tests."""

    def test_minimal_creation(self):
        ae = AuditEntry(
            action="register_schema",
            entity_type="SchemaDefinition",
            entity_id="sd-123",
        )
        assert ae.action == "register_schema"

    def test_default_actor(self):
        ae = AuditEntry(
            action="register",
            entity_type="Schema",
            entity_id="id-1",
        )
        assert ae.actor == "system"

    def test_default_provenance_hash_empty(self):
        ae = AuditEntry(
            action="register",
            entity_type="Schema",
            entity_id="id-1",
        )
        assert ae.provenance_hash == ""

    def test_default_parent_hash_empty(self):
        ae = AuditEntry(
            action="register",
            entity_type="Schema",
            entity_id="id-1",
        )
        assert ae.parent_hash == ""

    def test_empty_action_raises(self):
        with pytest.raises(ValidationError):
            AuditEntry(
                action="",
                entity_type="Schema",
                entity_id="id-1",
            )

    def test_empty_entity_type_raises(self):
        with pytest.raises(ValidationError):
            AuditEntry(
                action="register",
                entity_type="",
                entity_id="id-1",
            )

    def test_empty_entity_id_raises(self):
        with pytest.raises(ValidationError):
            AuditEntry(
                action="register",
                entity_type="Schema",
                entity_id="",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            AuditEntry(
                action="register",
                entity_type="Schema",
                entity_id="id-1",
                extra=1,
            )


class TestSchemaGroupModel:
    """SchemaGroup model tests."""

    def test_minimal_creation(self):
        sg = SchemaGroup(name="emissions-schemas")
        assert sg.name == "emissions-schemas"

    def test_default_description_empty(self):
        sg = SchemaGroup(name="group")
        assert sg.description == ""

    def test_default_schema_ids_empty(self):
        sg = SchemaGroup(name="group")
        assert sg.schema_ids == []

    def test_default_created_at_is_datetime(self):
        sg = SchemaGroup(name="group")
        assert isinstance(sg.created_at, datetime)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            SchemaGroup(name="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaGroup(name="g", extra=1)

    def test_with_schema_ids(self):
        sg = SchemaGroup(
            name="group",
            schema_ids=["s1", "s2", "s3"],
        )
        assert len(sg.schema_ids) == 3


class TestMigrationReportModel:
    """MigrationReport model tests."""

    def test_default_creation(self):
        mr = MigrationReport()
        assert mr.schemas_processed == 0
        assert mr.versions_created == 0
        assert mr.changes_detected == 0
        assert mr.migrations_executed == 0
        assert mr.rollbacks == 0
        assert mr.drift_events == 0
        assert mr.total_processing_time_ms == pytest.approx(0.0)

    def test_pipeline_id_is_uuid(self):
        mr = MigrationReport()
        uuid.UUID(mr.pipeline_id)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            MigrationReport(extra=1)


class TestSchemaStatisticsModel:
    """SchemaStatistics model tests."""

    def test_default_creation(self):
        ss = SchemaStatistics()
        assert ss.total_schemas == 0
        assert ss.total_versions == 0
        assert ss.total_changes == 0
        assert ss.total_migrations == 0
        assert ss.total_rollbacks == 0
        assert ss.total_drift_events == 0

    def test_default_schemas_by_type_empty(self):
        ss = SchemaStatistics()
        assert ss.schemas_by_type == {}

    def test_default_schemas_by_status_empty(self):
        ss = SchemaStatistics()
        assert ss.schemas_by_status == {}

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaStatistics(extra=1)


class TestVersionComparisonModel:
    """VersionComparison model tests."""

    def test_minimal_creation(self):
        vc = VersionComparison(
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert vc.source_version == "1.0.0"

    def test_default_changes_empty(self):
        vc = VersionComparison(
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert vc.changes == []

    def test_default_compatibility_level(self):
        vc = VersionComparison(
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert vc.compatibility_level == CompatibilityLevel.NONE

    def test_default_migration_needed_false(self):
        vc = VersionComparison(
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert vc.migration_needed is False

    def test_invalid_source_version_raises(self):
        with pytest.raises(ValidationError):
            VersionComparison(
                source_version="bad",
                target_version="2.0.0",
            )

    def test_invalid_target_version_raises(self):
        with pytest.raises(ValidationError):
            VersionComparison(
                source_version="1.0.0",
                target_version="bad",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            VersionComparison(
                source_version="1.0.0",
                target_version="2.0.0",
                extra=1,
            )


class TestPipelineResultModel:
    """PipelineResult model tests."""

    def test_default_creation(self):
        pr = PipelineResult()
        assert pr.stages_completed == []
        assert pr.stages_failed == []
        assert pr.changes is None
        assert pr.compatibility is None
        assert pr.plan is None
        assert pr.execution is None
        assert pr.verification == {}
        assert pr.total_time_ms == pytest.approx(0.0)

    def test_pipeline_id_is_uuid(self):
        pr = PipelineResult()
        uuid.UUID(pr.pipeline_id)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineResult(extra=1)


# ============================================================================
# Request Model Tests (8 models)
# ============================================================================


class TestRegisterSchemaRequest:
    """RegisterSchemaRequest model tests."""

    def test_creation(self):
        req = RegisterSchemaRequest(
            namespace="greenlang.test",
            name="test_schema",
        )
        assert req.namespace == "greenlang.test"
        assert req.name == "test_schema"

    def test_default_schema_type(self):
        req = RegisterSchemaRequest(namespace="ns", name="s")
        assert req.schema_type == SchemaType.JSON_SCHEMA

    def test_default_owner_empty(self):
        req = RegisterSchemaRequest(namespace="ns", name="s")
        assert req.owner == ""

    def test_default_tags_empty(self):
        req = RegisterSchemaRequest(namespace="ns", name="s")
        assert req.tags == {}

    def test_empty_namespace_raises(self):
        with pytest.raises(ValidationError):
            RegisterSchemaRequest(namespace="", name="s")

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            RegisterSchemaRequest(namespace="ns", name="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            RegisterSchemaRequest(namespace="ns", name="s", extra=1)


class TestUpdateSchemaRequest:
    """UpdateSchemaRequest model tests."""

    def test_empty_creation(self):
        req = UpdateSchemaRequest()
        assert req.owner is None
        assert req.tags is None
        assert req.status is None
        assert req.description is None

    def test_with_owner(self):
        req = UpdateSchemaRequest(owner="new-team")
        assert req.owner == "new-team"

    def test_with_status(self):
        req = UpdateSchemaRequest(status=SchemaStatus.ACTIVE)
        assert req.status == SchemaStatus.ACTIVE

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            UpdateSchemaRequest(extra=1)


class TestCreateVersionRequest:
    """CreateVersionRequest model tests."""

    def test_creation(self):
        req = CreateVersionRequest(
            schema_id="s-123",
            definition_json={"type": "object"},
        )
        assert req.schema_id == "s-123"

    def test_default_changelog_note(self):
        req = CreateVersionRequest(
            schema_id="s-123",
            definition_json={"type": "object"},
        )
        assert req.changelog_note == ""

    def test_empty_schema_id_raises(self):
        with pytest.raises(ValidationError):
            CreateVersionRequest(
                schema_id="",
                definition_json={"type": "object"},
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            CreateVersionRequest(
                schema_id="s-123",
                definition_json={"type": "object"},
                extra=1,
            )


class TestDetectChangesRequest:
    """DetectChangesRequest model tests."""

    def test_creation(self):
        req = DetectChangesRequest(
            source_version_id="v1",
            target_version_id="v2",
        )
        assert req.source_version_id == "v1"

    def test_empty_source_version_id_raises(self):
        with pytest.raises(ValidationError):
            DetectChangesRequest(
                source_version_id="",
                target_version_id="v2",
            )

    def test_empty_target_version_id_raises(self):
        with pytest.raises(ValidationError):
            DetectChangesRequest(
                source_version_id="v1",
                target_version_id="",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            DetectChangesRequest(
                source_version_id="v1",
                target_version_id="v2",
                extra=1,
            )


class TestCheckCompatibilityRequest:
    """CheckCompatibilityRequest model tests."""

    def test_creation(self):
        req = CheckCompatibilityRequest(
            source_version_id="v1",
            target_version_id="v2",
        )
        assert req.source_version_id == "v1"

    def test_empty_source_version_id_raises(self):
        with pytest.raises(ValidationError):
            CheckCompatibilityRequest(
                source_version_id="",
                target_version_id="v2",
            )

    def test_empty_target_version_id_raises(self):
        with pytest.raises(ValidationError):
            CheckCompatibilityRequest(
                source_version_id="v1",
                target_version_id="",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            CheckCompatibilityRequest(
                source_version_id="v1",
                target_version_id="v2",
                extra=1,
            )


class TestCreatePlanRequest:
    """CreatePlanRequest model tests."""

    def test_creation(self):
        req = CreatePlanRequest(
            source_schema_id="s1",
            target_schema_id="s2",
            source_version="1.0.0",
            target_version="2.0.0",
        )
        assert req.source_schema_id == "s1"

    def test_empty_source_schema_id_raises(self):
        with pytest.raises(ValidationError):
            CreatePlanRequest(
                source_schema_id="",
                target_schema_id="s2",
                source_version="1.0.0",
                target_version="2.0.0",
            )

    def test_invalid_source_version_raises(self):
        with pytest.raises(ValidationError):
            CreatePlanRequest(
                source_schema_id="s1",
                target_schema_id="s2",
                source_version="bad",
                target_version="2.0.0",
            )

    def test_invalid_target_version_raises(self):
        with pytest.raises(ValidationError):
            CreatePlanRequest(
                source_schema_id="s1",
                target_schema_id="s2",
                source_version="1.0.0",
                target_version="bad",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            CreatePlanRequest(
                source_schema_id="s1",
                target_schema_id="s2",
                source_version="1.0.0",
                target_version="2.0.0",
                extra=1,
            )


class TestExecuteMigrationRequest:
    """ExecuteMigrationRequest model tests."""

    def test_creation(self):
        req = ExecuteMigrationRequest(plan_id="plan-123")
        assert req.plan_id == "plan-123"

    def test_default_dry_run(self):
        req = ExecuteMigrationRequest(plan_id="plan-123")
        assert req.dry_run is False

    def test_default_batch_size(self):
        req = ExecuteMigrationRequest(plan_id="plan-123")
        assert req.batch_size == DEFAULT_MIGRATION_BATCH_SIZE

    def test_empty_plan_id_raises(self):
        with pytest.raises(ValidationError):
            ExecuteMigrationRequest(plan_id="")

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValidationError):
            ExecuteMigrationRequest(plan_id="p", batch_size=0)

    def test_batch_size_over_limit_raises(self):
        with pytest.raises(ValidationError):
            ExecuteMigrationRequest(plan_id="p", batch_size=100_001)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            ExecuteMigrationRequest(plan_id="p", extra=1)


class TestRunPipelineRequest:
    """RunPipelineRequest model tests."""

    def test_creation(self):
        req = RunPipelineRequest(
            schema_id="s-123",
            target_definition_json={"type": "object"},
        )
        assert req.schema_id == "s-123"

    def test_default_skip_compatibility(self):
        req = RunPipelineRequest(
            schema_id="s-123",
            target_definition_json={"type": "object"},
        )
        assert req.skip_compatibility is False

    def test_default_skip_dry_run(self):
        req = RunPipelineRequest(
            schema_id="s-123",
            target_definition_json={"type": "object"},
        )
        assert req.skip_dry_run is False

    def test_empty_schema_id_raises(self):
        with pytest.raises(ValidationError):
            RunPipelineRequest(
                schema_id="",
                target_definition_json={"type": "object"},
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            RunPipelineRequest(
                schema_id="s-123",
                target_definition_json={"type": "object"},
                extra=1,
            )


# ============================================================================
# Layer 1 Re-Export Tests
# ============================================================================


class TestLayerOneReexports:
    """Verify Layer 1 re-exports are importable and correct."""

    def test_quality_dimension_importable(self):
        assert QualityDimension is not None

    def test_quality_dimension_is_enum(self):
        assert issubclass(QualityDimension, Enum)

    def test_quality_dimension_completeness(self):
        assert QualityDimension.COMPLETENESS.value == "completeness"

    def test_quality_dimension_member_count(self):
        assert len(QualityDimension) == 6

    def test_rule_type_importable(self):
        assert RuleType is not None

    def test_rule_type_is_enum(self):
        assert issubclass(RuleType, Enum)

    def test_rule_type_completeness(self):
        assert RuleType.COMPLETENESS.value == "completeness"

    def test_rule_type_member_count(self):
        assert len(RuleType) == 6

    def test_dq_available_flag_is_bool(self):
        assert isinstance(_DQ_AVAILABLE, bool)

    def test_sc_available_flag_is_bool(self):
        assert isinstance(_SC_AVAILABLE, bool)


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_version_is_string(self):
        assert isinstance(VERSION, str)
        assert VERSION == "1.0.0"

    def test_max_schemas_per_namespace(self):
        assert MAX_SCHEMAS_PER_NAMESPACE == 10_000

    def test_max_fields_per_schema(self):
        assert MAX_FIELDS_PER_SCHEMA == 1_000

    def test_max_migration_steps(self):
        assert MAX_MIGRATION_STEPS == 500

    def test_supported_schema_types_count(self):
        assert len(SUPPORTED_SCHEMA_TYPES) == 3
        assert "json_schema" in SUPPORTED_SCHEMA_TYPES
        assert "avro" in SUPPORTED_SCHEMA_TYPES
        assert "protobuf" in SUPPORTED_SCHEMA_TYPES

    def test_default_compatibility_rules_keys(self):
        assert set(DEFAULT_COMPATIBILITY_RULES.keys()) == {
            "json_schema", "avro", "protobuf",
        }

    def test_default_compatibility_rules_json_schema(self):
        assert DEFAULT_COMPATIBILITY_RULES["json_schema"] == "backward"

    def test_default_compatibility_rules_avro(self):
        assert DEFAULT_COMPATIBILITY_RULES["avro"] == "full"

    def test_change_severity_order(self):
        assert CHANGE_SEVERITY_ORDER == ("cosmetic", "non_breaking", "breaking")

    def test_default_migration_batch_size(self):
        assert DEFAULT_MIGRATION_BATCH_SIZE == 1_000

    def test_auto_accept_confidence_threshold(self):
        assert AUTO_ACCEPT_CONFIDENCE_THRESHOLD == pytest.approx(0.90)

    def test_max_drift_events_per_version(self):
        assert MAX_DRIFT_EVENTS_PER_VERSION == 10_000


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Verify __all__ completeness."""

    def test_all_list_exists(self):
        from greenlang.schema_migration import models as mod
        assert hasattr(mod, "__all__")

    def test_all_contains_schema_type(self):
        from greenlang.schema_migration import models as mod
        assert "SchemaType" in mod.__all__

    def test_all_contains_schema_definition(self):
        from greenlang.schema_migration import models as mod
        assert "SchemaDefinition" in mod.__all__

    def test_all_contains_register_schema_request(self):
        from greenlang.schema_migration import models as mod
        assert "RegisterSchemaRequest" in mod.__all__

    def test_all_contains_quality_dimension(self):
        from greenlang.schema_migration import models as mod
        assert "QualityDimension" in mod.__all__

    def test_all_contains_rule_type(self):
        from greenlang.schema_migration import models as mod
        assert "RuleType" in mod.__all__

    def test_all_minimum_count(self):
        from greenlang.schema_migration import models as mod
        # 2 L1 re-exports + 2 flags + 10 constants + 14 enums + 16 models + 8 requests = 52+
        assert len(mod.__all__) >= 45
