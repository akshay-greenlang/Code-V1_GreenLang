# -*- coding: utf-8 -*-
"""
Unit Tests for Validation Rule Engine Models - AGENT-DATA-019

Tests all enumerations (12), SDK models (14), request models (8), Layer 1
re-export availability flags, constants, helpers, and Pydantic validation
(extra='forbid', required fields, range constraints) from
``greenlang.validation_rule_engine.models``.

Target: 80-100 tests, 85%+ coverage of greenlang.validation_rule_engine.models

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

from greenlang.validation_rule_engine.models import (
    # Layer 1 re-exports
    QualityDimension,
    RuleType,
    # Availability flags
    _QRE_AVAILABLE,
    _VC_AVAILABLE,
    _QD_AVAILABLE,
    _RT_AVAILABLE,
    # Constants
    VERSION,
    MAX_RULES_PER_NAMESPACE,
    MAX_RULES_PER_SET,
    MAX_RULE_SETS_PER_NAMESPACE,
    MAX_COMPOUND_NESTING_DEPTH,
    MAX_VERSIONS_PER_RULE,
    MAX_BATCH_RECORDS,
    DEFAULT_EVALUATION_BATCH_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_CONFLICTS_PER_REPORT,
    MAX_EVALUATION_DETAILS_PER_RUN,
    SEVERITY_ORDER,
    DEFAULT_CRITICAL_SLA_MS,
    DEFAULT_ALL_RULES_SLA_MS,
    SUPPORTED_REPORT_FORMATS,
    BUILT_IN_RULE_PACKS,
    # Helper
    _utcnow,
    # Enumerations (12)
    ValidationRuleType,
    RuleOperator,
    RuleSeverity,
    RuleStatus,
    CompoundOperator,
    EvaluationResult,
    ConflictType,
    RulePackType,
    ReportType,
    ReportFormat,
    VersionBumpType,
    SLALevel,
    # SDK models (14)
    ValidationRule,
    RuleSet,
    RuleSetMember,
    CompoundRule,
    RulePack,
    RuleVersion,
    EvaluationRun,
    EvaluationDetail,
    ConflictReport,
    ValidationReport,
    RuleTemplate,
    RuleDependency,
    SLAThreshold,
    AuditEntry,
    # Request models (8)
    CreateRuleRequest,
    UpdateRuleRequest,
    CreateRuleSetRequest,
    UpdateRuleSetRequest,
    EvaluateRequest,
    BatchEvaluateRequest,
    DetectConflictsRequest,
    GenerateReportRequest,
)


# ---------------------------------------------------------------------------
# Restore strict mode for model tests
# ---------------------------------------------------------------------------


def _set_model_extra(mode: str) -> None:
    """Set model_config extra to the given mode for all VRE BaseModel subclasses."""
    from greenlang.validation_rule_engine import models as vre_models

    for name in dir(vre_models):
        obj = getattr(vre_models, name)
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
def _strict_model_mode():
    """Enable extra='forbid' on all VRE models for this module."""
    _set_model_extra("forbid")
    yield
    _set_model_extra("ignore")


# ============================================================================
# TestConstants - module-level constants
# ============================================================================


class TestConstants:
    """Module-level constants must have expected values and types."""

    def test_version_string(self):
        assert VERSION == "1.0.0"

    def test_max_rules_per_namespace(self):
        assert MAX_RULES_PER_NAMESPACE == 100_000

    def test_max_rules_per_set(self):
        assert MAX_RULES_PER_SET == 5_000

    def test_max_rule_sets_per_namespace(self):
        assert MAX_RULE_SETS_PER_NAMESPACE == 10_000

    def test_max_compound_nesting_depth(self):
        assert MAX_COMPOUND_NESTING_DEPTH == 10

    def test_max_versions_per_rule(self):
        assert MAX_VERSIONS_PER_RULE == 500

    def test_max_batch_records(self):
        assert MAX_BATCH_RECORDS == 100_000

    def test_default_evaluation_batch_size(self):
        assert DEFAULT_EVALUATION_BATCH_SIZE == 1_000

    def test_default_confidence_threshold(self):
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.8

    def test_max_conflicts_per_report(self):
        assert MAX_CONFLICTS_PER_REPORT == 10_000

    def test_max_evaluation_details_per_run(self):
        assert MAX_EVALUATION_DETAILS_PER_RUN == 50_000

    def test_severity_order(self):
        assert SEVERITY_ORDER == ("low", "medium", "high", "critical")

    def test_default_critical_sla_ms(self):
        assert DEFAULT_CRITICAL_SLA_MS == 500.0

    def test_default_all_rules_sla_ms(self):
        assert DEFAULT_ALL_RULES_SLA_MS == 5_000.0

    def test_supported_report_formats(self):
        assert SUPPORTED_REPORT_FORMATS == ("text", "json", "html", "markdown", "csv")

    def test_built_in_rule_packs(self):
        assert BUILT_IN_RULE_PACKS == ("ghg_protocol", "csrd_esrs", "eudr", "soc2")


# ============================================================================
# TestHelper - _utcnow
# ============================================================================


class TestHelper:
    """_utcnow() helper returns UTC datetime with zeroed microseconds."""

    def test_utcnow_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_utcnow_has_timezone(self):
        result = _utcnow()
        assert result.tzinfo is not None

    def test_utcnow_microseconds_zeroed(self):
        result = _utcnow()
        assert result.microsecond == 0


# ============================================================================
# TestEnumValidationRuleType
# ============================================================================


class TestEnumValidationRuleType:
    """ValidationRuleType enum must have 10 members."""

    def test_member_count(self):
        assert len(ValidationRuleType) == 10

    def test_completeness(self):
        assert ValidationRuleType.COMPLETENESS.value == "completeness"

    def test_range(self):
        assert ValidationRuleType.RANGE.value == "range"

    def test_format(self):
        assert ValidationRuleType.FORMAT.value == "format"

    def test_uniqueness(self):
        assert ValidationRuleType.UNIQUENESS.value == "uniqueness"

    def test_custom(self):
        assert ValidationRuleType.CUSTOM.value == "custom"

    def test_freshness(self):
        assert ValidationRuleType.FRESHNESS.value == "freshness"

    def test_cross_field(self):
        assert ValidationRuleType.CROSS_FIELD.value == "cross_field"

    def test_conditional(self):
        assert ValidationRuleType.CONDITIONAL.value == "conditional"

    def test_statistical(self):
        assert ValidationRuleType.STATISTICAL.value == "statistical"

    def test_referential(self):
        assert ValidationRuleType.REFERENTIAL.value == "referential"

    def test_is_str_enum(self):
        assert issubclass(ValidationRuleType, str)
        assert issubclass(ValidationRuleType, Enum)


# ============================================================================
# TestEnumRuleOperator
# ============================================================================


class TestEnumRuleOperator:
    """RuleOperator enum must have 12 members."""

    def test_member_count(self):
        assert len(RuleOperator) == 12

    def test_equals(self):
        assert RuleOperator.EQUALS.value == "equals"

    def test_not_equals(self):
        assert RuleOperator.NOT_EQUALS.value == "not_equals"

    def test_greater_than(self):
        assert RuleOperator.GREATER_THAN.value == "greater_than"

    def test_less_than(self):
        assert RuleOperator.LESS_THAN.value == "less_than"

    def test_greater_equal(self):
        assert RuleOperator.GREATER_EQUAL.value == "greater_equal"

    def test_less_equal(self):
        assert RuleOperator.LESS_EQUAL.value == "less_equal"

    def test_between(self):
        assert RuleOperator.BETWEEN.value == "between"

    def test_matches(self):
        assert RuleOperator.MATCHES.value == "matches"

    def test_contains(self):
        assert RuleOperator.CONTAINS.value == "contains"

    def test_in_set(self):
        assert RuleOperator.IN_SET.value == "in_set"

    def test_not_in_set(self):
        assert RuleOperator.NOT_IN_SET.value == "not_in_set"

    def test_is_null(self):
        assert RuleOperator.IS_NULL.value == "is_null"


# ============================================================================
# TestEnumRuleSeverity
# ============================================================================


class TestEnumRuleSeverity:
    """RuleSeverity enum must have 4 members."""

    def test_member_count(self):
        assert len(RuleSeverity) == 4

    def test_critical(self):
        assert RuleSeverity.CRITICAL.value == "critical"

    def test_high(self):
        assert RuleSeverity.HIGH.value == "high"

    def test_medium(self):
        assert RuleSeverity.MEDIUM.value == "medium"

    def test_low(self):
        assert RuleSeverity.LOW.value == "low"


# ============================================================================
# TestEnumRuleStatus
# ============================================================================


class TestEnumRuleStatus:
    """RuleStatus enum must have 4 members."""

    def test_member_count(self):
        assert len(RuleStatus) == 4

    def test_draft(self):
        assert RuleStatus.DRAFT.value == "draft"

    def test_active(self):
        assert RuleStatus.ACTIVE.value == "active"

    def test_deprecated(self):
        assert RuleStatus.DEPRECATED.value == "deprecated"

    def test_archived(self):
        assert RuleStatus.ARCHIVED.value == "archived"


# ============================================================================
# TestEnumCompoundOperator
# ============================================================================


class TestEnumCompoundOperator:
    """CompoundOperator enum must have 3 members."""

    def test_member_count(self):
        assert len(CompoundOperator) == 3

    def test_and(self):
        assert CompoundOperator.AND.value == "and"

    def test_or(self):
        assert CompoundOperator.OR.value == "or"

    def test_not(self):
        assert CompoundOperator.NOT.value == "not"


# ============================================================================
# TestEnumEvaluationResult
# ============================================================================


class TestEnumEvaluationResult:
    """EvaluationResult enum must have 3 members."""

    def test_member_count(self):
        assert len(EvaluationResult) == 3

    def test_pass_result(self):
        assert EvaluationResult.PASS_RESULT.value == "pass"

    def test_warn(self):
        assert EvaluationResult.WARN.value == "warn"

    def test_fail(self):
        assert EvaluationResult.FAIL.value == "fail"


# ============================================================================
# TestEnumConflictType
# ============================================================================


class TestEnumConflictType:
    """ConflictType enum must have 5 members."""

    def test_member_count(self):
        assert len(ConflictType) == 5

    def test_range_overlap(self):
        assert ConflictType.RANGE_OVERLAP.value == "range_overlap"

    def test_range_contradiction(self):
        assert ConflictType.RANGE_CONTRADICTION.value == "range_contradiction"

    def test_format_conflict(self):
        assert ConflictType.FORMAT_CONFLICT.value == "format_conflict"

    def test_severity_inconsistency(self):
        assert ConflictType.SEVERITY_INCONSISTENCY.value == "severity_inconsistency"

    def test_redundancy(self):
        assert ConflictType.REDUNDANCY.value == "redundancy"


# ============================================================================
# TestEnumRulePackType
# ============================================================================


class TestEnumRulePackType:
    """RulePackType enum must have 5 members."""

    def test_member_count(self):
        assert len(RulePackType) == 5

    def test_ghg_protocol(self):
        assert RulePackType.GHG_PROTOCOL.value == "ghg_protocol"

    def test_csrd_esrs(self):
        assert RulePackType.CSRD_ESRS.value == "csrd_esrs"

    def test_eudr(self):
        assert RulePackType.EUDR.value == "eudr"

    def test_soc2(self):
        assert RulePackType.SOC2.value == "soc2"

    def test_custom(self):
        assert RulePackType.CUSTOM.value == "custom"


# ============================================================================
# TestEnumReportType
# ============================================================================


class TestEnumReportType:
    """ReportType enum must have 5 members."""

    def test_member_count(self):
        assert len(ReportType) == 5

    def test_summary(self):
        assert ReportType.SUMMARY.value == "summary"

    def test_detailed(self):
        assert ReportType.DETAILED.value == "detailed"

    def test_compliance(self):
        assert ReportType.COMPLIANCE.value == "compliance"

    def test_trend(self):
        assert ReportType.TREND.value == "trend"

    def test_executive(self):
        assert ReportType.EXECUTIVE.value == "executive"


# ============================================================================
# TestEnumReportFormat
# ============================================================================


class TestEnumReportFormat:
    """ReportFormat enum must have 5 members."""

    def test_member_count(self):
        assert len(ReportFormat) == 5

    def test_text(self):
        assert ReportFormat.TEXT.value == "text"

    def test_json(self):
        assert ReportFormat.JSON.value == "json"

    def test_html(self):
        assert ReportFormat.HTML.value == "html"

    def test_markdown(self):
        assert ReportFormat.MARKDOWN.value == "markdown"

    def test_csv(self):
        assert ReportFormat.CSV.value == "csv"


# ============================================================================
# TestEnumVersionBumpType
# ============================================================================


class TestEnumVersionBumpType:
    """VersionBumpType enum must have 3 members."""

    def test_member_count(self):
        assert len(VersionBumpType) == 3

    def test_major(self):
        assert VersionBumpType.MAJOR.value == "major"

    def test_minor(self):
        assert VersionBumpType.MINOR.value == "minor"

    def test_patch(self):
        assert VersionBumpType.PATCH.value == "patch"


# ============================================================================
# TestEnumSLALevel
# ============================================================================


class TestEnumSLALevel:
    """SLALevel enum must have 3 members."""

    def test_member_count(self):
        assert len(SLALevel) == 3

    def test_critical_rules(self):
        assert SLALevel.CRITICAL_RULES.value == "critical_rules"

    def test_all_rules(self):
        assert SLALevel.ALL_RULES.value == "all_rules"

    def test_custom(self):
        assert SLALevel.CUSTOM.value == "custom"


# ============================================================================
# TestValidationRule - SDK model
# ============================================================================


class TestValidationRule:
    """ValidationRule model construction and validation."""

    def test_minimal_valid(self):
        rule = ValidationRule(name="test_rule", rule_type=ValidationRuleType.RANGE)
        assert rule.name == "test_rule"
        assert rule.rule_type == ValidationRuleType.RANGE

    def test_auto_generated_id(self):
        rule = ValidationRule(name="test_rule", rule_type=ValidationRuleType.RANGE)
        uuid.UUID(rule.id)  # valid UUID

    def test_default_severity(self):
        rule = ValidationRule(name="test_rule", rule_type=ValidationRuleType.RANGE)
        assert rule.severity == RuleSeverity.MEDIUM

    def test_default_status(self):
        rule = ValidationRule(name="test_rule", rule_type=ValidationRuleType.RANGE)
        assert rule.status == RuleStatus.DRAFT

    def test_default_version(self):
        rule = ValidationRule(name="test_rule", rule_type=ValidationRuleType.RANGE)
        assert rule.version == 1

    def test_default_namespace(self):
        rule = ValidationRule(name="test_rule", rule_type=ValidationRuleType.RANGE)
        assert rule.namespace == "default"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            ValidationRule(name="", rule_type=ValidationRuleType.RANGE)

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError):
            ValidationRule(name="   ", rule_type=ValidationRuleType.RANGE)

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            ValidationRule(rule_type=ValidationRuleType.RANGE)

    def test_missing_rule_type_raises(self):
        with pytest.raises(ValidationError):
            ValidationRule(name="test_rule")

    def test_version_ge_1(self):
        with pytest.raises(ValidationError):
            ValidationRule(name="test_rule", rule_type=ValidationRuleType.RANGE, version=0)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ValidationRule(
                name="test_rule",
                rule_type=ValidationRuleType.RANGE,
                unknown_field="not_allowed",
            )

    def test_full_construction(self):
        rule = ValidationRule(
            name="co2e_range",
            description="CO2e range check",
            rule_type=ValidationRuleType.RANGE,
            operator=RuleOperator.BETWEEN,
            target_field="co2e",
            threshold_min=0.0,
            threshold_max=1_000_000.0,
            severity=RuleSeverity.CRITICAL,
            status=RuleStatus.ACTIVE,
            namespace="emissions",
            tags={"domain": "ghg"},
            framework="ghg_protocol",
            parameters={"unit": "tCO2e"},
            version=3,
        )
        assert rule.operator == RuleOperator.BETWEEN
        assert rule.threshold_min == 0.0
        assert rule.threshold_max == 1_000_000.0
        assert rule.severity == RuleSeverity.CRITICAL
        assert rule.status == RuleStatus.ACTIVE
        assert rule.version == 3


# ============================================================================
# TestRuleSet - SDK model
# ============================================================================


class TestRuleSet:
    """RuleSet model construction and validation."""

    def test_minimal_valid(self):
        rs = RuleSet(name="test_set")
        assert rs.name == "test_set"

    def test_default_gate_pass_threshold(self):
        rs = RuleSet(name="test_set")
        assert rs.gate_pass_threshold == 1.0

    def test_default_fail_on_critical(self):
        rs = RuleSet(name="test_set")
        assert rs.fail_on_critical is True

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            RuleSet(name="")

    def test_gate_pass_below_zero_raises(self):
        with pytest.raises(ValidationError):
            RuleSet(name="test", gate_pass_threshold=-0.1)

    def test_gate_pass_above_one_raises(self):
        with pytest.raises(ValidationError):
            RuleSet(name="test", gate_pass_threshold=1.1)

    def test_version_ge_1(self):
        with pytest.raises(ValidationError):
            RuleSet(name="test", version=0)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            RuleSet(name="test", unknown_field="bad")


# ============================================================================
# TestRuleSetMember - SDK model
# ============================================================================


class TestRuleSetMember:
    """RuleSetMember model construction and validation."""

    def test_minimal_valid(self):
        m = RuleSetMember(rule_set_id="set_1", rule_id="rule_1")
        assert m.rule_set_id == "set_1"
        assert m.rule_id == "rule_1"

    def test_default_enabled(self):
        m = RuleSetMember(rule_set_id="set_1", rule_id="rule_1")
        assert m.enabled is True

    def test_empty_rule_set_id_raises(self):
        with pytest.raises(ValidationError):
            RuleSetMember(rule_set_id="", rule_id="rule_1")

    def test_empty_rule_id_raises(self):
        with pytest.raises(ValidationError):
            RuleSetMember(rule_set_id="set_1", rule_id="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            RuleSetMember(rule_set_id="s1", rule_id="r1", unknown_field="bad")


# ============================================================================
# TestCompoundRule - SDK model
# ============================================================================


class TestCompoundRule:
    """CompoundRule model construction and validation."""

    def test_minimal_valid(self):
        cr = CompoundRule(name="compound_1", operator=CompoundOperator.AND)
        assert cr.name == "compound_1"
        assert cr.operator == CompoundOperator.AND

    def test_default_nesting_depth(self):
        cr = CompoundRule(name="compound_1", operator=CompoundOperator.AND)
        assert cr.nesting_depth == 1

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            CompoundRule(name="", operator=CompoundOperator.AND)

    def test_nesting_depth_exceeds_max_raises(self):
        with pytest.raises(ValidationError):
            CompoundRule(
                name="deep", operator=CompoundOperator.AND, nesting_depth=11
            )

    def test_nesting_depth_max_valid(self):
        cr = CompoundRule(name="max_depth", operator=CompoundOperator.OR, nesting_depth=10)
        assert cr.nesting_depth == 10

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            CompoundRule(name="c1", operator=CompoundOperator.AND, unknown_field="bad")


# ============================================================================
# TestRulePack - SDK model
# ============================================================================


class TestRulePack:
    """RulePack model construction and validation."""

    def test_minimal_valid(self):
        rp = RulePack(name="ghg_pack", pack_type=RulePackType.GHG_PROTOCOL)
        assert rp.name == "ghg_pack"
        assert rp.pack_type == RulePackType.GHG_PROTOCOL

    def test_default_version(self):
        rp = RulePack(name="pack", pack_type=RulePackType.CUSTOM)
        assert rp.version == "1.0.0"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            RulePack(name="", pack_type=RulePackType.CUSTOM)

    def test_empty_version_raises(self):
        with pytest.raises(ValidationError):
            RulePack(name="pack", pack_type=RulePackType.CUSTOM, version="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            RulePack(name="pack", pack_type=RulePackType.CUSTOM, unknown_field="bad")


# ============================================================================
# TestRuleVersion - SDK model
# ============================================================================


class TestRuleVersion:
    """RuleVersion model construction and validation."""

    def test_minimal_valid(self):
        rv = RuleVersion(rule_id="rule_001")
        assert rv.rule_id == "rule_001"
        assert rv.version_number == 1

    def test_empty_rule_id_raises(self):
        with pytest.raises(ValidationError):
            RuleVersion(rule_id="")

    def test_version_number_ge_1(self):
        with pytest.raises(ValidationError):
            RuleVersion(rule_id="rule_001", version_number=0)

    def test_default_bump_type(self):
        rv = RuleVersion(rule_id="rule_001")
        assert rv.bump_type == VersionBumpType.PATCH


# ============================================================================
# TestEvaluationRun - SDK model
# ============================================================================


class TestEvaluationRun:
    """EvaluationRun model construction and validation."""

    def test_minimal_valid(self):
        er = EvaluationRun(rule_set_id="set_001")
        assert er.rule_set_id == "set_001"

    def test_empty_rule_set_id_raises(self):
        with pytest.raises(ValidationError):
            EvaluationRun(rule_set_id="")

    def test_pass_rate_range_low(self):
        with pytest.raises(ValidationError):
            EvaluationRun(rule_set_id="set_001", pass_rate=-0.1)

    def test_pass_rate_range_high(self):
        with pytest.raises(ValidationError):
            EvaluationRun(rule_set_id="set_001", pass_rate=1.1)

    def test_default_gate_result(self):
        er = EvaluationRun(rule_set_id="set_001")
        assert er.gate_result == EvaluationResult.FAIL

    def test_default_counts_zero(self):
        er = EvaluationRun(rule_set_id="set_001")
        assert er.total_records == 0
        assert er.passed_count == 0
        assert er.failed_count == 0

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            EvaluationRun(rule_set_id="set_001", unknown_field="bad")


# ============================================================================
# TestEvaluationDetail - SDK model
# ============================================================================


class TestEvaluationDetail:
    """EvaluationDetail model construction and validation."""

    def test_minimal_valid(self):
        ed = EvaluationDetail(
            evaluation_run_id="run_001",
            rule_id="rule_001",
            result=EvaluationResult.PASS_RESULT,
        )
        assert ed.evaluation_run_id == "run_001"

    def test_empty_evaluation_run_id_raises(self):
        with pytest.raises(ValidationError):
            EvaluationDetail(
                evaluation_run_id="",
                rule_id="rule_001",
                result=EvaluationResult.PASS_RESULT,
            )

    def test_empty_rule_id_raises(self):
        with pytest.raises(ValidationError):
            EvaluationDetail(
                evaluation_run_id="run_001",
                rule_id="",
                result=EvaluationResult.PASS_RESULT,
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            EvaluationDetail(
                evaluation_run_id="run_001",
                rule_id="rule_001",
                result=EvaluationResult.FAIL,
                unknown_field="bad",
            )


# ============================================================================
# TestConflictReport - SDK model
# ============================================================================


class TestConflictReport:
    """ConflictReport model construction and validation."""

    def test_minimal_valid(self):
        cr = ConflictReport()
        assert cr.scope == "all"
        assert cr.total_conflicts == 0

    def test_defaults(self):
        cr = ConflictReport()
        assert cr.resolution_required is False
        assert cr.conflicts == []
        assert cr.recommendations == []

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ConflictReport(unknown_field="bad")


# ============================================================================
# TestValidationReport - SDK model
# ============================================================================


class TestValidationReport:
    """ValidationReport model construction and validation."""

    def test_minimal_valid(self):
        vr = ValidationReport()
        assert vr.report_type == ReportType.SUMMARY
        assert vr.report_format == ReportFormat.JSON

    def test_pass_rate_range_low(self):
        with pytest.raises(ValidationError):
            ValidationReport(pass_rate=-0.1)

    def test_pass_rate_range_high(self):
        with pytest.raises(ValidationError):
            ValidationReport(pass_rate=1.1)

    def test_framework_compliance_range(self):
        with pytest.raises(ValidationError):
            ValidationReport(framework_compliance=1.5)

    def test_framework_compliance_valid(self):
        vr = ValidationReport(framework_compliance=0.85)
        assert vr.framework_compliance == 0.85

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ValidationReport(unknown_field="bad")


# ============================================================================
# TestRuleTemplate - SDK model
# ============================================================================


class TestRuleTemplate:
    """RuleTemplate model construction and validation."""

    def test_minimal_valid(self):
        rt = RuleTemplate(name="template_1")
        assert rt.name == "template_1"

    def test_default_rule_type(self):
        rt = RuleTemplate(name="template_1")
        assert rt.rule_type == ValidationRuleType.CUSTOM

    def test_default_usage_count(self):
        rt = RuleTemplate(name="template_1")
        assert rt.usage_count == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            RuleTemplate(name="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            RuleTemplate(name="template_1", unknown_field="bad")


# ============================================================================
# TestRuleDependency - SDK model
# ============================================================================


class TestRuleDependency:
    """RuleDependency model construction and validation."""

    def test_minimal_valid(self):
        rd = RuleDependency(rule_id="rule_a", depends_on_rule_id="rule_b")
        assert rd.rule_id == "rule_a"
        assert rd.depends_on_rule_id == "rule_b"

    def test_default_dependency_type(self):
        rd = RuleDependency(rule_id="rule_a", depends_on_rule_id="rule_b")
        assert rd.dependency_type == "result"

    def test_default_is_mandatory(self):
        rd = RuleDependency(rule_id="rule_a", depends_on_rule_id="rule_b")
        assert rd.is_mandatory is True

    def test_empty_rule_id_raises(self):
        with pytest.raises(ValidationError):
            RuleDependency(rule_id="", depends_on_rule_id="rule_b")

    def test_empty_depends_on_raises(self):
        with pytest.raises(ValidationError):
            RuleDependency(rule_id="rule_a", depends_on_rule_id="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            RuleDependency(
                rule_id="r_a", depends_on_rule_id="r_b", unknown_field="bad"
            )


# ============================================================================
# TestSLAThreshold - SDK model
# ============================================================================


class TestSLAThreshold:
    """SLAThreshold model construction and validation."""

    def test_minimal_valid(self):
        sla = SLAThreshold(name="default_sla")
        assert sla.name == "default_sla"

    def test_default_level(self):
        sla = SLAThreshold(name="default_sla")
        assert sla.level == SLALevel.ALL_RULES

    def test_default_enabled(self):
        sla = SLAThreshold(name="default_sla")
        assert sla.enabled is True

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            SLAThreshold(name="")

    def test_negative_critical_threshold_raises(self):
        with pytest.raises(ValidationError):
            SLAThreshold(name="sla", critical_threshold_ms=-1.0)

    def test_negative_warning_threshold_raises(self):
        with pytest.raises(ValidationError):
            SLAThreshold(name="sla", warning_threshold_ms=-1.0)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SLAThreshold(name="sla", unknown_field="bad")


# ============================================================================
# TestAuditEntry - SDK model
# ============================================================================


class TestAuditEntry:
    """AuditEntry model construction and validation."""

    def test_minimal_valid(self):
        ae = AuditEntry(
            action="create_rule",
            entity_type="ValidationRule",
            entity_id="rule_001",
        )
        assert ae.action == "create_rule"
        assert ae.entity_type == "ValidationRule"
        assert ae.entity_id == "rule_001"

    def test_default_actor(self):
        ae = AuditEntry(
            action="create_rule",
            entity_type="ValidationRule",
            entity_id="rule_001",
        )
        assert ae.actor == "system"

    def test_empty_action_raises(self):
        with pytest.raises(ValidationError):
            AuditEntry(action="", entity_type="ValidationRule", entity_id="rule_001")

    def test_empty_entity_type_raises(self):
        with pytest.raises(ValidationError):
            AuditEntry(action="create_rule", entity_type="", entity_id="rule_001")

    def test_empty_entity_id_raises(self):
        with pytest.raises(ValidationError):
            AuditEntry(action="create_rule", entity_type="ValidationRule", entity_id="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            AuditEntry(
                action="create_rule",
                entity_type="ValidationRule",
                entity_id="rule_001",
                unknown_field="bad",
            )


# ============================================================================
# TestCreateRuleRequest - request model
# ============================================================================


class TestCreateRuleRequest:
    """CreateRuleRequest model construction and validation."""

    def test_minimal_valid(self):
        req = CreateRuleRequest(name="rule_1", rule_type=ValidationRuleType.RANGE)
        assert req.name == "rule_1"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            CreateRuleRequest(name="", rule_type=ValidationRuleType.RANGE)

    def test_missing_rule_type_raises(self):
        with pytest.raises(ValidationError):
            CreateRuleRequest(name="rule_1")

    def test_default_severity(self):
        req = CreateRuleRequest(name="rule_1", rule_type=ValidationRuleType.RANGE)
        assert req.severity == RuleSeverity.MEDIUM

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            CreateRuleRequest(
                name="rule_1",
                rule_type=ValidationRuleType.RANGE,
                unknown_field="bad",
            )


# ============================================================================
# TestUpdateRuleRequest - request model
# ============================================================================


class TestUpdateRuleRequest:
    """UpdateRuleRequest model construction and validation."""

    def test_empty_request_valid(self):
        """All fields are optional; empty request is valid."""
        req = UpdateRuleRequest()
        assert req.description is None
        assert req.severity is None

    def test_partial_update(self):
        req = UpdateRuleRequest(
            description="Updated description",
            severity=RuleSeverity.CRITICAL,
        )
        assert req.description == "Updated description"
        assert req.severity == RuleSeverity.CRITICAL

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            UpdateRuleRequest(unknown_field="bad")


# ============================================================================
# TestCreateRuleSetRequest - request model
# ============================================================================


class TestCreateRuleSetRequest:
    """CreateRuleSetRequest model construction and validation."""

    def test_minimal_valid(self):
        req = CreateRuleSetRequest(name="set_1")
        assert req.name == "set_1"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            CreateRuleSetRequest(name="")

    def test_gate_pass_threshold_below_zero_raises(self):
        with pytest.raises(ValidationError):
            CreateRuleSetRequest(name="set_1", gate_pass_threshold=-0.1)

    def test_gate_pass_threshold_above_one_raises(self):
        with pytest.raises(ValidationError):
            CreateRuleSetRequest(name="set_1", gate_pass_threshold=1.1)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            CreateRuleSetRequest(name="set_1", unknown_field="bad")


# ============================================================================
# TestUpdateRuleSetRequest - request model
# ============================================================================


class TestUpdateRuleSetRequest:
    """UpdateRuleSetRequest model construction and validation."""

    def test_empty_request_valid(self):
        req = UpdateRuleSetRequest()
        assert req.description is None

    def test_gate_pass_threshold_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            UpdateRuleSetRequest(gate_pass_threshold=1.5)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            UpdateRuleSetRequest(unknown_field="bad")


# ============================================================================
# TestEvaluateRequest - request model
# ============================================================================


class TestEvaluateRequest:
    """EvaluateRequest model construction and validation."""

    def test_minimal_valid(self):
        req = EvaluateRequest(rule_set_id="set_001")
        assert req.rule_set_id == "set_001"

    def test_empty_rule_set_id_raises(self):
        with pytest.raises(ValidationError):
            EvaluateRequest(rule_set_id="")

    def test_default_include_details(self):
        req = EvaluateRequest(rule_set_id="set_001")
        assert req.include_details is True

    def test_default_fail_fast(self):
        req = EvaluateRequest(rule_set_id="set_001")
        assert req.fail_fast is False

    def test_default_dry_run(self):
        req = EvaluateRequest(rule_set_id="set_001")
        assert req.dry_run is False

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            EvaluateRequest(rule_set_id="set_001", unknown_field="bad")


# ============================================================================
# TestBatchEvaluateRequest - request model
# ============================================================================


class TestBatchEvaluateRequest:
    """BatchEvaluateRequest model construction and validation."""

    def test_minimal_valid(self):
        eval_req = EvaluateRequest(rule_set_id="set_001")
        req = BatchEvaluateRequest(evaluations=[eval_req])
        assert len(req.evaluations) == 1

    def test_empty_evaluations_raises(self):
        with pytest.raises(ValidationError):
            BatchEvaluateRequest(evaluations=[])

    def test_default_batch_size(self):
        eval_req = EvaluateRequest(rule_set_id="set_001")
        req = BatchEvaluateRequest(evaluations=[eval_req])
        assert req.batch_size == DEFAULT_EVALUATION_BATCH_SIZE

    def test_default_parallel(self):
        eval_req = EvaluateRequest(rule_set_id="set_001")
        req = BatchEvaluateRequest(evaluations=[eval_req])
        assert req.parallel is False

    def test_default_max_workers(self):
        eval_req = EvaluateRequest(rule_set_id="set_001")
        req = BatchEvaluateRequest(evaluations=[eval_req])
        assert req.max_workers == 4

    def test_extra_field_forbidden(self):
        eval_req = EvaluateRequest(rule_set_id="set_001")
        with pytest.raises(ValidationError):
            BatchEvaluateRequest(evaluations=[eval_req], unknown_field="bad")


# ============================================================================
# TestDetectConflictsRequest - request model
# ============================================================================


class TestDetectConflictsRequest:
    """DetectConflictsRequest model construction and validation."""

    def test_minimal_valid(self):
        req = DetectConflictsRequest()
        assert req.scope == "all"

    def test_default_include_recommendations(self):
        req = DetectConflictsRequest()
        assert req.include_recommendations is True

    def test_default_namespace(self):
        req = DetectConflictsRequest()
        assert req.namespace == "default"

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            DetectConflictsRequest(unknown_field="bad")


# ============================================================================
# TestGenerateReportRequest - request model
# ============================================================================


class TestGenerateReportRequest:
    """GenerateReportRequest model construction and validation."""

    def test_minimal_valid(self):
        req = GenerateReportRequest()
        assert req.report_type == ReportType.SUMMARY
        assert req.report_format == ReportFormat.JSON

    def test_default_scope(self):
        req = GenerateReportRequest()
        assert req.scope == "full"

    def test_default_include_recommendations(self):
        req = GenerateReportRequest()
        assert req.include_recommendations is True

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            GenerateReportRequest(unknown_field="bad")


# ============================================================================
# TestLayer1ReExports - availability flags
# ============================================================================


class TestLayer1ReExports:
    """Layer 1 re-export availability flags and class existence."""

    def test_qre_available_is_bool(self):
        assert isinstance(_QRE_AVAILABLE, bool)

    def test_vc_available_is_bool(self):
        assert isinstance(_VC_AVAILABLE, bool)

    def test_qd_available_is_bool(self):
        assert isinstance(_QD_AVAILABLE, bool)

    def test_rt_available_is_bool(self):
        assert isinstance(_RT_AVAILABLE, bool)

    def test_quality_dimension_exists(self):
        assert QualityDimension is not None

    def test_rule_type_exists(self):
        assert RuleType is not None

    def test_quality_dimension_has_completeness(self):
        assert hasattr(QualityDimension, "COMPLETENESS")

    def test_rule_type_has_completeness(self):
        assert hasattr(RuleType, "COMPLETENESS")

    def test_quality_dimension_has_validity(self):
        assert hasattr(QualityDimension, "VALIDITY")

    def test_quality_dimension_has_consistency(self):
        assert hasattr(QualityDimension, "CONSISTENCY")
