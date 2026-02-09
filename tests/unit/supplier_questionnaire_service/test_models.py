# -*- coding: utf-8 -*-
"""
Unit tests for Supplier Questionnaire Processor models (AGENT-DATA-008)

Tests all Pydantic v2 data models: enums, SDK models, request models,
field validators, serialization, and edge cases.

Target: 100+ tests covering every model class and enum defined in models.py.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.supplier_questionnaire.models import (
    # Enums
    Framework,
    QuestionType,
    QuestionnaireStatus,
    DistributionStatus,
    DistributionChannel,
    ResponseStatus,
    ValidationSeverity,
    ReminderType,
    EscalationLevel,
    CDPGrade,
    PerformanceTier,
    ReportFormat,
    # Core models
    TemplateQuestion,
    TemplateSection,
    QuestionnaireTemplate,
    Answer,
    Distribution,
    QuestionnaireResponse,
    ValidationCheck,
    ValidationSummary,
    QuestionnaireScore,
    FollowUpAction,
    CampaignAnalytics,
    # Request models
    CreateTemplateRequest,
    DistributeRequest,
    SubmitResponseRequest,
    # Helper
    _utcnow,
)


# ============================================================================
# Enum value tests
# ============================================================================


class TestFrameworkEnum:
    """Validate all Framework members."""

    @pytest.mark.parametrize("member,value", [
        ("CDP_CLIMATE", "cdp_climate"),
        ("CDP_WATER", "cdp_water"),
        ("CDP_FORESTS", "cdp_forests"),
        ("ECOVADIS", "ecovadis"),
        ("DJSI", "djsi"),
        ("GRI", "gri"),
        ("SASB", "sasb"),
        ("TCFD", "tcfd"),
        ("TNFD", "tnfd"),
        ("SBT", "sbt"),
        ("CUSTOM", "custom"),
    ])
    def test_framework_member_value(self, member, value):
        assert Framework[member].value == value

    def test_framework_member_count(self):
        assert len(Framework) == 11

    def test_framework_is_str_enum(self):
        assert isinstance(Framework.CUSTOM, str)
        assert Framework.CUSTOM == "custom"


class TestQuestionTypeEnum:
    @pytest.mark.parametrize("member,value", [
        ("TEXT", "text"),
        ("NUMERIC", "numeric"),
        ("SINGLE_CHOICE", "single_choice"),
        ("MULTI_CHOICE", "multi_choice"),
        ("YES_NO", "yes_no"),
        ("DATE", "date"),
        ("FILE_UPLOAD", "file_upload"),
        ("TABLE", "table"),
        ("PERCENTAGE", "percentage"),
        ("CURRENCY", "currency"),
    ])
    def test_question_type_value(self, member, value):
        assert QuestionType[member].value == value

    def test_question_type_count(self):
        assert len(QuestionType) == 10


class TestQuestionnaireStatusEnum:
    @pytest.mark.parametrize("value", ["draft", "active", "archived", "deprecated"])
    def test_status_exists(self, value):
        assert QuestionnaireStatus(value).value == value

    def test_status_count(self):
        assert len(QuestionnaireStatus) == 4


class TestDistributionStatusEnum:
    @pytest.mark.parametrize("member,value", [
        ("PENDING", "pending"),
        ("SENT", "sent"),
        ("DELIVERED", "delivered"),
        ("OPENED", "opened"),
        ("IN_PROGRESS", "in_progress"),
        ("SUBMITTED", "submitted"),
        ("BOUNCED", "bounced"),
        ("EXPIRED", "expired"),
        ("CANCELLED", "cancelled"),
    ])
    def test_distribution_status_value(self, member, value):
        assert DistributionStatus[member].value == value

    def test_distribution_status_count(self):
        assert len(DistributionStatus) == 9


class TestDistributionChannelEnum:
    @pytest.mark.parametrize("member,value", [
        ("EMAIL", "email"),
        ("PORTAL", "portal"),
        ("API", "api"),
        ("BULK_UPLOAD", "bulk_upload"),
    ])
    def test_channel_value(self, member, value):
        assert DistributionChannel[member].value == value

    def test_channel_count(self):
        assert len(DistributionChannel) == 4


class TestResponseStatusEnum:
    @pytest.mark.parametrize("value", [
        "draft", "in_progress", "submitted", "validated",
        "scored", "reopened", "rejected",
    ])
    def test_response_status_value_exists(self, value):
        assert ResponseStatus(value).value == value

    def test_response_status_count(self):
        assert len(ResponseStatus) == 7


class TestValidationSeverityEnum:
    def test_severity_error(self):
        assert ValidationSeverity.ERROR.value == "error"

    def test_severity_warning(self):
        assert ValidationSeverity.WARNING.value == "warning"

    def test_severity_info(self):
        assert ValidationSeverity.INFO.value == "info"

    def test_severity_count(self):
        assert len(ValidationSeverity) == 3


class TestReminderTypeEnum:
    @pytest.mark.parametrize("value", ["gentle", "firm", "urgent", "final"])
    def test_reminder_type_exists(self, value):
        assert ReminderType(value).value == value

    def test_reminder_type_count(self):
        assert len(ReminderType) == 4


class TestEscalationLevelEnum:
    @pytest.mark.parametrize("value", ["level_1", "level_2", "level_3", "level_4", "level_5"])
    def test_escalation_level_exists(self, value):
        assert EscalationLevel(value).value == value

    def test_escalation_level_count(self):
        assert len(EscalationLevel) == 5


class TestCDPGradeEnum:
    @pytest.mark.parametrize("value", ["A", "A-", "B", "B-", "C", "C-", "D", "D-", "F"])
    def test_cdp_grade_exists(self, value):
        assert CDPGrade(value).value == value

    def test_cdp_grade_count(self):
        assert len(CDPGrade) == 9


class TestPerformanceTierEnum:
    @pytest.mark.parametrize("member,value", [
        ("LEADER", "leader"),
        ("ADVANCED", "advanced"),
        ("INTERMEDIATE", "intermediate"),
        ("BEGINNER", "beginner"),
        ("LAGGARD", "laggard"),
    ])
    def test_tier_value(self, member, value):
        assert PerformanceTier[member].value == value

    def test_tier_count(self):
        assert len(PerformanceTier) == 5


class TestReportFormatEnum:
    @pytest.mark.parametrize("value", ["text", "json", "markdown", "html"])
    def test_report_format_exists(self, value):
        assert ReportFormat(value).value == value

    def test_report_format_count(self):
        assert len(ReportFormat) == 4


# ============================================================================
# Helper function tests
# ============================================================================


class TestUtcNowHelper:
    def test_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_has_utc_timezone(self):
        result = _utcnow()
        assert result.tzinfo == timezone.utc

    def test_microseconds_are_zeroed(self):
        result = _utcnow()
        assert result.microsecond == 0


# ============================================================================
# TemplateQuestion model tests
# ============================================================================


class TestTemplateQuestion:
    def test_create_with_text_only(self):
        q = TemplateQuestion(text="What is your annual CO2 output?")
        assert q.text == "What is your annual CO2 output?"
        assert q.question_type == QuestionType.TEXT
        assert q.required is True

    def test_default_weight_is_one(self):
        q = TemplateQuestion(text="Q?")
        assert q.weight == 1.0

    def test_default_order_is_zero(self):
        q = TemplateQuestion(text="Q?")
        assert q.order == 0

    def test_question_id_is_valid_uuid(self):
        q = TemplateQuestion(text="Q?")
        uuid.UUID(q.question_id)

    def test_default_code_is_empty(self):
        q = TemplateQuestion(text="Q?")
        assert q.code == ""

    def test_default_choices_is_empty_list(self):
        q = TemplateQuestion(text="Q?")
        assert q.choices == []

    def test_default_help_text_is_empty(self):
        q = TemplateQuestion(text="Q?")
        assert q.help_text == ""

    def test_default_translations_empty_dict(self):
        q = TemplateQuestion(text="Q?")
        assert q.translations == {}

    def test_default_validation_rules_empty_dict(self):
        q = TemplateQuestion(text="Q?")
        assert q.validation_rules == {}

    def test_weight_negative_rejected(self):
        with pytest.raises(ValidationError):
            TemplateQuestion(text="Q?", weight=-0.1)

    def test_weight_above_10_rejected(self):
        with pytest.raises(ValidationError):
            TemplateQuestion(text="Q?", weight=10.1)

    def test_negative_order_rejected(self):
        with pytest.raises(ValidationError):
            TemplateQuestion(text="Q?", order=-1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            TemplateQuestion(text="Q?", unknown_field="x")

    def test_model_dump_keys(self):
        q = TemplateQuestion(text="Q?")
        data = q.model_dump()
        expected_keys = {
            "question_id", "code", "text", "question_type", "required",
            "choices", "help_text", "weight", "framework_ref",
            "translations", "validation_rules", "order",
        }
        assert expected_keys.issubset(set(data.keys()))

    def test_choices_round_trip(self):
        q = TemplateQuestion(text="Choose", choices=["A", "B", "C"])
        assert q.choices == ["A", "B", "C"]


# ============================================================================
# TemplateSection model tests
# ============================================================================


class TestTemplateSection:
    def test_create_with_name_only(self):
        sec = TemplateSection(name="Energy")
        assert sec.name == "Energy"

    def test_default_questions_empty(self):
        sec = TemplateSection(name="S")
        assert sec.questions == []

    def test_default_weight_is_one(self):
        sec = TemplateSection(name="S")
        assert sec.weight == 1.0

    def test_section_id_is_valid_uuid(self):
        sec = TemplateSection(name="S")
        uuid.UUID(sec.section_id)

    def test_weight_above_10_rejected(self):
        with pytest.raises(ValidationError):
            TemplateSection(name="S", weight=10.1)

    def test_negative_order_rejected(self):
        with pytest.raises(ValidationError):
            TemplateSection(name="S", order=-1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            TemplateSection(name="S", extra="bad")

    def test_translations_dict_of_dicts(self):
        sec = TemplateSection(
            name="Energy",
            translations={"de": {"name": "Energie", "description": "Energieverbrauch"}},
        )
        assert sec.translations["de"]["name"] == "Energie"


# ============================================================================
# QuestionnaireTemplate model tests
# ============================================================================


class TestQuestionnaireTemplate:
    def test_create_with_required_fields(self):
        tpl = QuestionnaireTemplate(name="Annual CDP", framework=Framework.CDP_CLIMATE)
        assert tpl.name == "Annual CDP"
        assert tpl.framework == Framework.CDP_CLIMATE
        assert tpl.version == 1
        assert tpl.status == QuestionnaireStatus.DRAFT

    def test_empty_name_raises_error(self):
        with pytest.raises(ValidationError):
            QuestionnaireTemplate(name="", framework=Framework.CUSTOM)

    def test_whitespace_name_raises_error(self):
        with pytest.raises(ValidationError):
            QuestionnaireTemplate(name="   ", framework=Framework.CUSTOM)

    def test_default_language_is_en(self):
        tpl = QuestionnaireTemplate(name="T", framework=Framework.CUSTOM)
        assert tpl.language == "en"

    def test_default_supported_languages_contains_en(self):
        tpl = QuestionnaireTemplate(name="T", framework=Framework.CUSTOM)
        assert "en" in tpl.supported_languages

    def test_created_at_is_utc(self):
        tpl = QuestionnaireTemplate(name="T", framework=Framework.CUSTOM)
        assert tpl.created_at.tzinfo == timezone.utc

    def test_default_created_by_is_system(self):
        tpl = QuestionnaireTemplate(name="T", framework=Framework.CUSTOM)
        assert tpl.created_by == "system"

    def test_default_provenance_hash_empty(self):
        tpl = QuestionnaireTemplate(name="T", framework=Framework.CUSTOM)
        assert tpl.provenance_hash == ""

    def test_version_below_1_rejected(self):
        with pytest.raises(ValidationError):
            QuestionnaireTemplate(name="T", framework=Framework.CUSTOM, version=0)

    def test_template_id_is_valid_uuid(self):
        tpl = QuestionnaireTemplate(name="T", framework=Framework.CUSTOM)
        uuid.UUID(tpl.template_id)

    def test_model_dump_contains_all_fields(self):
        tpl = QuestionnaireTemplate(name="T", framework=Framework.CUSTOM)
        data = tpl.model_dump()
        expected_keys = {
            "template_id", "name", "framework", "version", "status",
            "sections", "language", "supported_languages", "description",
            "created_at", "updated_at", "created_by", "tags", "provenance_hash",
        }
        assert expected_keys.issubset(set(data.keys()))

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            QuestionnaireTemplate(name="T", framework=Framework.CUSTOM, extra="bad")


# ============================================================================
# Answer model tests
# ============================================================================


class TestAnswer:
    def test_create_with_required_fields(self):
        a = Answer(question_id="q1", value="42")
        assert a.question_id == "q1"
        assert a.value == "42"

    def test_empty_question_id_raises_error(self):
        with pytest.raises(ValidationError):
            Answer(question_id="", value="x")

    def test_whitespace_question_id_raises_error(self):
        with pytest.raises(ValidationError):
            Answer(question_id="   ", value="x")

    def test_default_confidence_is_one(self):
        a = Answer(question_id="q1", value="v")
        assert a.confidence == 1.0

    def test_confidence_above_1_raises_error(self):
        with pytest.raises(ValidationError):
            Answer(question_id="q1", value="v", confidence=1.1)

    def test_confidence_below_0_raises_error(self):
        with pytest.raises(ValidationError):
            Answer(question_id="q1", value="v", confidence=-0.1)

    def test_default_unit_is_empty(self):
        a = Answer(question_id="q1", value="v")
        assert a.unit == ""

    def test_default_evidence_refs_empty_list(self):
        a = Answer(question_id="q1", value="v")
        assert a.evidence_refs == []

    def test_default_notes_is_empty(self):
        a = Answer(question_id="q1", value="v")
        assert a.notes == ""

    def test_value_accepts_numeric(self):
        a = Answer(question_id="q1", value=42.5)
        assert a.value == 42.5

    def test_value_accepts_list(self):
        a = Answer(question_id="q1", value=["option_a", "option_b"])
        assert a.value == ["option_a", "option_b"]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Answer(question_id="q1", value="v", extra="bad")


# ============================================================================
# Distribution model tests
# ============================================================================


class TestDistribution:
    def test_create_with_required_fields(self):
        d = Distribution(template_id="tpl1", supplier_id="s1")
        assert d.status == DistributionStatus.PENDING
        assert d.channel == DistributionChannel.EMAIL
        assert d.reminder_count == 0

    def test_empty_template_id_raises_error(self):
        with pytest.raises(ValidationError):
            Distribution(template_id="", supplier_id="s1")

    def test_whitespace_template_id_raises_error(self):
        with pytest.raises(ValidationError):
            Distribution(template_id="   ", supplier_id="s1")

    def test_empty_supplier_id_raises_error(self):
        with pytest.raises(ValidationError):
            Distribution(template_id="tpl1", supplier_id="")

    def test_distribution_id_is_valid_uuid(self):
        d = Distribution(template_id="tpl1", supplier_id="s1")
        uuid.UUID(d.distribution_id)

    def test_optional_timestamps_default_none(self):
        d = Distribution(template_id="tpl1", supplier_id="s1")
        assert d.deadline is None
        assert d.sent_at is None
        assert d.delivered_at is None
        assert d.opened_at is None
        assert d.submitted_at is None

    def test_default_supplier_name_empty(self):
        d = Distribution(template_id="tpl1", supplier_id="s1")
        assert d.supplier_name == ""

    def test_default_supplier_email_empty(self):
        d = Distribution(template_id="tpl1", supplier_id="s1")
        assert d.supplier_email == ""

    def test_default_provenance_hash_empty(self):
        d = Distribution(template_id="tpl1", supplier_id="s1")
        assert d.provenance_hash == ""

    def test_created_at_is_utc(self):
        d = Distribution(template_id="tpl1", supplier_id="s1")
        assert d.created_at.tzinfo == timezone.utc

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Distribution(template_id="tpl1", supplier_id="s1", extra="bad")


# ============================================================================
# QuestionnaireResponse model tests
# ============================================================================


class TestQuestionnaireResponse:
    def test_create_with_required_fields(self):
        r = QuestionnaireResponse(
            distribution_id="d1", template_id="t1", supplier_id="s1",
        )
        assert r.status == ResponseStatus.DRAFT
        assert r.completion_pct == 0.0

    def test_empty_distribution_id_raises_error(self):
        with pytest.raises(ValidationError):
            QuestionnaireResponse(
                distribution_id="", template_id="t1", supplier_id="s1",
            )

    def test_empty_supplier_id_raises_error(self):
        with pytest.raises(ValidationError):
            QuestionnaireResponse(
                distribution_id="d1", template_id="t1", supplier_id="",
            )

    def test_completion_pct_above_100_raises_error(self):
        with pytest.raises(ValidationError):
            QuestionnaireResponse(
                distribution_id="d1", template_id="t1", supplier_id="s1",
                completion_pct=101.0,
            )

    def test_completion_pct_below_0_raises_error(self):
        with pytest.raises(ValidationError):
            QuestionnaireResponse(
                distribution_id="d1", template_id="t1", supplier_id="s1",
                completion_pct=-1.0,
            )

    def test_default_language_is_en(self):
        r = QuestionnaireResponse(
            distribution_id="d1", template_id="t1", supplier_id="s1",
        )
        assert r.language == "en"

    def test_default_provenance_hash_empty(self):
        r = QuestionnaireResponse(
            distribution_id="d1", template_id="t1", supplier_id="s1",
        )
        assert r.provenance_hash == ""

    def test_response_id_is_valid_uuid(self):
        r = QuestionnaireResponse(
            distribution_id="d1", template_id="t1", supplier_id="s1",
        )
        uuid.UUID(r.response_id)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            QuestionnaireResponse(
                distribution_id="d1", template_id="t1", supplier_id="s1",
                extra="bad",
            )


# ============================================================================
# ValidationCheck model tests
# ============================================================================


class TestValidationCheck:
    def test_create_with_required_fields(self):
        vc = ValidationCheck(check_type="required_fields")
        assert vc.passed is True
        assert vc.severity == ValidationSeverity.ERROR

    def test_default_question_id_empty(self):
        vc = ValidationCheck(check_type="c")
        assert vc.question_id == ""

    def test_default_message_empty(self):
        vc = ValidationCheck(check_type="c")
        assert vc.message == ""

    def test_check_id_is_valid_uuid(self):
        vc = ValidationCheck(check_type="c")
        uuid.UUID(vc.check_id)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ValidationCheck(check_type="c", extra="bad")


# ============================================================================
# ValidationSummary model tests
# ============================================================================


class TestValidationSummary:
    def test_create_with_defaults(self):
        vs = ValidationSummary()
        assert vs.total_checks == 0
        assert vs.is_valid is True
        assert vs.checks == []

    def test_data_quality_score_above_100_raises_error(self):
        with pytest.raises(ValidationError):
            ValidationSummary(data_quality_score=101.0)

    def test_data_quality_score_below_0_raises_error(self):
        with pytest.raises(ValidationError):
            ValidationSummary(data_quality_score=-1.0)

    def test_negative_total_checks_rejected(self):
        with pytest.raises(ValidationError):
            ValidationSummary(total_checks=-1)

    def test_validated_at_is_utc(self):
        vs = ValidationSummary()
        assert vs.validated_at.tzinfo == timezone.utc

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ValidationSummary(extra="bad")


# ============================================================================
# QuestionnaireScore model tests
# ============================================================================


class TestQuestionnaireScore:
    def test_create_with_defaults(self):
        qs = QuestionnaireScore()
        assert qs.raw_score == 0.0
        assert qs.normalized_score == 0.0
        assert qs.performance_tier == PerformanceTier.BEGINNER
        assert qs.cdp_grade is None
        assert qs.framework == Framework.CUSTOM

    def test_normalized_score_above_100_rejected(self):
        with pytest.raises(ValidationError):
            QuestionnaireScore(normalized_score=101.0)

    def test_normalized_score_below_0_rejected(self):
        with pytest.raises(ValidationError):
            QuestionnaireScore(normalized_score=-1.0)

    def test_score_id_is_valid_uuid(self):
        qs = QuestionnaireScore()
        uuid.UUID(qs.score_id)

    def test_scored_at_is_utc(self):
        qs = QuestionnaireScore()
        assert qs.scored_at.tzinfo == timezone.utc

    def test_section_scores_round_trip(self):
        qs = QuestionnaireScore(section_scores={"energy": 85.0, "water": 72.0})
        assert qs.section_scores["energy"] == 85.0
        assert qs.section_scores["water"] == 72.0

    def test_default_methodology_empty(self):
        qs = QuestionnaireScore()
        assert qs.methodology == ""

    def test_default_provenance_hash_empty(self):
        qs = QuestionnaireScore()
        assert qs.provenance_hash == ""

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            QuestionnaireScore(extra="bad")


# ============================================================================
# FollowUpAction model tests
# ============================================================================


class TestFollowUpAction:
    def test_create_with_required_fields(self):
        fa = FollowUpAction(distribution_id="d1")
        assert fa.reminder_type == ReminderType.GENTLE
        assert fa.escalation_level is None
        assert fa.status == "scheduled"

    def test_empty_distribution_id_raises_error(self):
        with pytest.raises(ValidationError):
            FollowUpAction(distribution_id="")

    def test_whitespace_distribution_id_raises_error(self):
        with pytest.raises(ValidationError):
            FollowUpAction(distribution_id="   ")

    def test_action_id_is_valid_uuid(self):
        fa = FollowUpAction(distribution_id="d1")
        uuid.UUID(fa.action_id)

    def test_default_supplier_id_empty(self):
        fa = FollowUpAction(distribution_id="d1")
        assert fa.supplier_id == ""

    def test_default_message_empty(self):
        fa = FollowUpAction(distribution_id="d1")
        assert fa.message == ""

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            FollowUpAction(distribution_id="d1", extra="bad")


# ============================================================================
# CampaignAnalytics model tests
# ============================================================================


class TestCampaignAnalytics:
    def test_create_with_defaults(self):
        ca = CampaignAnalytics()
        assert ca.total_distributions == 0
        assert ca.response_rate == 0.0
        assert ca.score_distribution == {}

    def test_response_rate_above_100_rejected(self):
        with pytest.raises(ValidationError):
            CampaignAnalytics(response_rate=101.0)

    def test_avg_score_above_100_rejected(self):
        with pytest.raises(ValidationError):
            CampaignAnalytics(avg_score=101.0)

    def test_score_distribution_round_trip(self):
        ca = CampaignAnalytics(
            score_distribution={"0-20": 5, "20-40": 10, "40-60": 15},
        )
        assert ca.score_distribution["20-40"] == 10

    def test_status_breakdown_round_trip(self):
        ca = CampaignAnalytics(
            status_breakdown={"pending": 10, "submitted": 5},
        )
        assert ca.status_breakdown["submitted"] == 5

    def test_generated_at_is_utc(self):
        ca = CampaignAnalytics()
        assert ca.generated_at.tzinfo == timezone.utc

    def test_default_provenance_hash_empty(self):
        ca = CampaignAnalytics()
        assert ca.provenance_hash == ""

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            CampaignAnalytics(extra="bad")


# ============================================================================
# Request model tests
# ============================================================================


class TestCreateTemplateRequest:
    def test_create_with_required_fields(self):
        req = CreateTemplateRequest(name="Carbon Disclosure", framework=Framework.CDP_CLIMATE)
        assert req.language == "en"
        assert req.tags == []

    def test_empty_name_raises_error(self):
        with pytest.raises(ValidationError):
            CreateTemplateRequest(name="", framework=Framework.CUSTOM)

    def test_whitespace_name_raises_error(self):
        with pytest.raises(ValidationError):
            CreateTemplateRequest(name="   ", framework=Framework.CUSTOM)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            CreateTemplateRequest(name="T", framework=Framework.CUSTOM, extra_field="bad")

    def test_default_description_empty(self):
        req = CreateTemplateRequest(name="T", framework=Framework.CUSTOM)
        assert req.description == ""


class TestDistributeRequest:
    def test_create_with_required_fields(self):
        req = DistributeRequest(
            template_id="t1",
            supplier_list=[{"id": "s1", "name": "Acme", "email": "a@b.com"}],
        )
        assert req.channel == DistributionChannel.EMAIL
        assert req.deadline_days == 30

    def test_empty_template_id_raises_error(self):
        with pytest.raises(ValidationError):
            DistributeRequest(
                template_id="",
                supplier_list=[{"id": "s1"}],
            )

    def test_whitespace_template_id_raises_error(self):
        with pytest.raises(ValidationError):
            DistributeRequest(
                template_id="   ",
                supplier_list=[{"id": "s1"}],
            )

    def test_deadline_days_below_1_rejected(self):
        with pytest.raises(ValidationError):
            DistributeRequest(
                template_id="t1",
                supplier_list=[{"id": "s1"}],
                deadline_days=0,
            )

    def test_deadline_days_above_365_rejected(self):
        with pytest.raises(ValidationError):
            DistributeRequest(
                template_id="t1",
                supplier_list=[{"id": "s1"}],
                deadline_days=366,
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            DistributeRequest(
                template_id="t1",
                supplier_list=[{"id": "s1"}],
                extra="bad",
            )


class TestSubmitResponseRequest:
    def test_create_with_required_fields(self):
        answer = Answer(question_id="q1", value="42")
        req = SubmitResponseRequest(distribution_id="d1", answers=[answer])
        assert req.language == "en"

    def test_empty_distribution_id_raises_error(self):
        answer = Answer(question_id="q1", value="v")
        with pytest.raises(ValidationError):
            SubmitResponseRequest(distribution_id="", answers=[answer])

    def test_extra_fields_forbidden(self):
        answer = Answer(question_id="q1", value="v")
        with pytest.raises(ValidationError):
            SubmitResponseRequest(distribution_id="d1", answers=[answer], extra="bad")


# ============================================================================
# Serialization round-trip tests
# ============================================================================


class TestModelSerialization:
    def test_template_question_round_trip(self):
        q = TemplateQuestion(text="How much CO2?", weight=2.5, code="C1.1")
        data = q.model_dump()
        q2 = TemplateQuestion(**data)
        assert q2.text == q.text
        assert q2.weight == q.weight
        assert q2.code == q.code

    def test_distribution_round_trip(self):
        d = Distribution(
            template_id="tpl1", supplier_id="s1",
            supplier_name="Acme", supplier_email="a@b.com",
        )
        data = d.model_dump()
        d2 = Distribution(**data)
        assert d2.template_id == d.template_id

    def test_questionnaire_response_round_trip(self):
        r = QuestionnaireResponse(
            distribution_id="d1", template_id="t1", supplier_id="s1",
        )
        data = r.model_dump()
        r2 = QuestionnaireResponse(**data)
        assert r2.response_id == r.response_id
        assert r2.status == r.status

    def test_campaign_analytics_round_trip(self):
        ca = CampaignAnalytics(
            campaign_id="c1",
            total_distributions=100,
            total_responses=75,
            response_rate=75.0,
        )
        data = ca.model_dump()
        ca2 = CampaignAnalytics(**data)
        assert ca2.total_responses == 75
        assert ca2.response_rate == 75.0

    def test_answer_with_nested_value_round_trip(self):
        a = Answer(
            question_id="q1",
            value={"nested": "data", "number": 42},
            confidence=0.85,
            notes="See attachment",
        )
        data = a.model_dump()
        a2 = Answer(**data)
        assert a2.value["nested"] == "data"
        assert a2.confidence == 0.85

    def test_questionnaire_score_round_trip(self):
        qs = QuestionnaireScore(
            response_id="r1",
            raw_score=78.5,
            normalized_score=82.3,
            performance_tier=PerformanceTier.ADVANCED,
            cdp_grade=CDPGrade.B,
        )
        data = qs.model_dump()
        qs2 = QuestionnaireScore(**data)
        assert qs2.raw_score == 78.5
        assert qs2.performance_tier == PerformanceTier.ADVANCED
        assert qs2.cdp_grade == CDPGrade.B

    def test_follow_up_action_round_trip(self):
        fa = FollowUpAction(
            distribution_id="d1",
            reminder_type=ReminderType.URGENT,
            escalation_level=EscalationLevel.LEVEL_3,
        )
        data = fa.model_dump()
        fa2 = FollowUpAction(**data)
        assert fa2.reminder_type == ReminderType.URGENT
        assert fa2.escalation_level == EscalationLevel.LEVEL_3


# ============================================================================
# __all__ exports tests
# ============================================================================


class TestModelsExports:
    def test_all_exports_defined(self):
        import greenlang.supplier_questionnaire.models as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} in __all__ but not defined"

    def test_all_count(self):
        import greenlang.supplier_questionnaire.models as mod
        # 12 enums + 11 core models + 3 request models = 26
        assert len(mod.__all__) == 26
