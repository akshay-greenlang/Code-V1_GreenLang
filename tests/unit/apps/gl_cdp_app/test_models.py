# -*- coding: utf-8 -*-
"""
Unit tests for CDP Platform Domain Models.

Tests all Pydantic v2 domain models including helpers, enum validation,
model creation, serialization/deserialization, field constraints,
and nested model relationships with 35+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from services.config import (
    CDPModule,
    QuestionType,
    ResponseStatus,
    ScoringCategory,
    ScoringLevel,
    ScoreBand,
    GapSeverity,
    EffortLevel,
    SubmissionFormat,
    VerificationScope,
    AssuranceLevel,
    SupplierStatus,
    TransitionPathwayType,
    SBTiStatus,
    MilestoneStatus,
)
from services.models import (
    CDPOrganization,
    CDPQuestionnaire,
    CDPModuleInstance,
    CDPQuestion,
    CDPResponse,
    CDPResponseVersion,
    CDPEvidenceAttachment,
    CDPReviewAction,
    CDPScoringResult,
    CDPCategoryScore,
    CDPGapAnalysis,
    CDPGapItem,
    CDPBenchmark,
    CDPPeerComparison,
    CDPSupplyChainRequest,
    CDPSupplierResponse,
    CDPTransitionPlan,
    CDPTransitionMilestone,
    CDPVerificationRecord,
    CDPSubmission,
    _new_id,
    _now,
    _sha256,
)


# ===========================================================================
# Helper tests
# ===========================================================================

class TestHelpers:
    """Test utility functions."""

    def test_new_id_returns_string(self):
        result = _new_id()
        assert isinstance(result, str)
        assert len(result) == 36

    def test_new_id_unique(self):
        ids = {_new_id() for _ in range(200)}
        assert len(ids) == 200

    def test_now_returns_datetime(self):
        result = _now()
        assert isinstance(result, datetime)

    def test_now_no_microseconds(self):
        result = _now()
        assert result.microsecond == 0

    def test_sha256_deterministic(self):
        h1 = _sha256("cdp_test_payload")
        h2 = _sha256("cdp_test_payload")
        assert h1 == h2

    def test_sha256_length_64(self):
        result = _sha256("cdp data")
        assert len(result) == 64

    def test_sha256_different_inputs(self):
        h1 = _sha256("input_alpha")
        h2 = _sha256("input_beta")
        assert h1 != h2


# ===========================================================================
# Enum tests
# ===========================================================================

class TestEnums:
    """Test all CDP-specific enums."""

    def test_cdp_module_values(self):
        modules = [e.value for e in CDPModule]
        assert "M0" in modules
        assert "M7" in modules
        assert "M12" in modules
        assert "M13" in modules
        assert len(modules) == 14  # M0 through M13

    def test_scoring_level_values(self):
        levels = [e.value for e in ScoringLevel]
        assert "disclosure" in levels
        assert "awareness" in levels
        assert "management" in levels
        assert "leadership" in levels

    def test_scoring_category_count(self):
        categories = list(ScoringCategory)
        assert len(categories) == 17

    def test_question_type_values(self):
        types = [e.value for e in QuestionType]
        for expected in ["text", "numeric", "percentage", "table",
                         "multi_select", "single_select", "yes_no"]:
            assert expected in types

    def test_response_status_values(self):
        statuses = [e.value for e in ResponseStatus]
        for expected in ["draft", "in_review", "approved", "submitted"]:
            assert expected in statuses

    def test_score_band_ordering(self):
        bands = [e.value for e in ScoreBand]
        assert "D-" in bands
        assert "A" in bands
        assert len(bands) == 8

    def test_gap_severity_values(self):
        severities = [e.value for e in GapSeverity]
        for expected in ["critical", "high", "medium", "low"]:
            assert expected in severities

    def test_effort_level_values(self):
        levels = [e.value for e in EffortLevel]
        for expected in ["low", "medium", "high"]:
            assert expected in levels

    def test_assurance_level_values(self):
        levels = [e.value for e in AssuranceLevel]
        assert "limited" in levels
        assert "reasonable" in levels

    def test_supplier_status_values(self):
        statuses = [e.value for e in SupplierStatus]
        for expected in ["invited", "responded", "declined", "not_invited"]:
            assert expected in statuses

    def test_sbti_status_values(self):
        statuses = [e.value for e in SBTiStatus]
        for expected in ["committed", "targets_set", "validated", "none"]:
            assert expected in statuses


# ===========================================================================
# Organization model tests
# ===========================================================================

class TestCDPOrganization:
    """Test CDPOrganization model."""

    def test_create_organization(self, sample_organization):
        assert sample_organization.name == "Acme Manufacturing Corp"
        assert sample_organization.sector_gics == "20101010"
        assert sample_organization.country == "US"
        assert len(sample_organization.id) == 36

    def test_organization_timestamps(self, sample_organization):
        assert isinstance(sample_organization.created_at, datetime)
        assert isinstance(sample_organization.updated_at, datetime)

    def test_organization_revenue_decimal(self, sample_organization):
        assert isinstance(sample_organization.revenue_usd, Decimal)
        assert sample_organization.revenue_usd == Decimal("2500000000")

    def test_organization_serialization(self, sample_organization):
        data = sample_organization.model_dump()
        assert data["name"] == "Acme Manufacturing Corp"
        assert "id" in data
        assert "created_at" in data

    def test_organization_deserialization(self):
        data = {
            "name": "Test Corp",
            "sector_gics": "10101010",
            "region": "Europe",
            "country": "DE",
            "employee_count": 1000,
            "revenue_usd": "500000000",
            "fiscal_year_end": "12-31",
        }
        org = CDPOrganization(**data)
        assert org.name == "Test Corp"
        assert org.country == "DE"


# ===========================================================================
# Questionnaire model tests
# ===========================================================================

class TestCDPQuestionnaire:
    """Test CDPQuestionnaire model."""

    def test_create_questionnaire(self, sample_questionnaire, sample_organization):
        assert sample_questionnaire.org_id == sample_organization.id
        assert sample_questionnaire.reporting_year == 2025
        assert sample_questionnaire.questionnaire_version == "2025"

    def test_questionnaire_sector_modules(self, sample_questionnaire):
        assert isinstance(sample_questionnaire.sector_specific_modules, dict)
        assert "M12" in sample_questionnaire.sector_specific_modules

    def test_questionnaire_timestamps(self, sample_questionnaire):
        assert isinstance(sample_questionnaire.created_at, datetime)


# ===========================================================================
# Module model tests
# ===========================================================================

class TestCDPModuleInstance:
    """Test CDPModuleInstance model."""

    def test_create_module(self, sample_module):
        assert sample_module.module_code == "M7"
        assert sample_module.question_count == 35
        assert sample_module.is_applicable is True

    def test_module_sort_order(self, sample_module):
        assert sample_module.sort_order == 7

    def test_module_not_sector_specific(self, sample_module):
        assert sample_module.is_sector_specific is False


# ===========================================================================
# Question model tests
# ===========================================================================

class TestCDPQuestion:
    """Test CDPQuestion model."""

    def test_create_question(self, sample_question):
        assert sample_question.question_number == "C7.1"
        assert sample_question.question_type == QuestionType.YES_NO
        assert sample_question.is_required is True

    def test_question_scoring_points(self, sample_question):
        assert sample_question.disclosure_points == 2
        assert sample_question.awareness_points == 0
        assert sample_question.management_points == 0
        assert sample_question.leadership_points == 0

    def test_question_version_year(self, sample_question):
        assert sample_question.version_year == 2025


# ===========================================================================
# Response model tests
# ===========================================================================

class TestCDPResponse:
    """Test CDPResponse model."""

    def test_create_response(self, sample_response):
        assert sample_response.response_status == ResponseStatus.DRAFT
        assert sample_response.auto_populated is False

    def test_auto_populated_response(self, auto_populated_response):
        assert auto_populated_response.auto_populated is True
        assert auto_populated_response.auto_populated_source == "MRV-001 Stationary Combustion"

    def test_response_confidence(self, sample_response):
        assert isinstance(sample_response.confidence_score, Decimal)
        assert sample_response.confidence_score == Decimal("0.95")


# ===========================================================================
# Scoring model tests
# ===========================================================================

class TestCDPScoringResult:
    """Test CDPScoringResult model."""

    def test_create_scoring_result(self, sample_scoring_result):
        assert sample_scoring_result.overall_score == Decimal("72.5")
        assert sample_scoring_result.score_band == "A-"
        assert sample_scoring_result.a_level_eligible is False

    def test_confidence_interval(self, sample_scoring_result):
        assert sample_scoring_result.confidence_lower == Decimal("68.0")
        assert sample_scoring_result.confidence_upper == Decimal("77.0")
        width = sample_scoring_result.confidence_upper - sample_scoring_result.confidence_lower
        assert width == Decimal("9.0")


class TestCDPCategoryScore:
    """Test CDPCategoryScore model."""

    def test_create_category_score(self, sample_category_scores):
        assert len(sample_category_scores) == 17

    def test_category_weights_sum(self, sample_category_scores):
        total_weight = sum(cs.weight for cs in sample_category_scores)
        # Weights should sum close to 1.0 (100%)
        assert Decimal("0.95") <= total_weight <= Decimal("1.05")

    def test_governance_category(self, sample_category_scores):
        gov = sample_category_scores[0]
        assert gov.category_code == "CAT01"
        assert gov.category_name == "Governance"


# ===========================================================================
# Gap analysis model tests
# ===========================================================================

class TestCDPGapAnalysis:
    """Test CDPGapAnalysis model."""

    def test_create_gap_analysis(self, sample_gap_analysis):
        assert sample_gap_analysis.total_gaps == 25
        assert sample_gap_analysis.critical_gaps == 3

    def test_gap_totals_consistent(self, sample_gap_analysis):
        gap_sum = (
            sample_gap_analysis.critical_gaps
            + sample_gap_analysis.high_gaps
            + sample_gap_analysis.medium_gaps
            + sample_gap_analysis.low_gaps
        )
        assert gap_sum == sample_gap_analysis.total_gaps


class TestCDPGapItem:
    """Test CDPGapItem model."""

    def test_create_gap_item(self, sample_gap_items):
        critical = sample_gap_items[0]
        assert critical.severity == GapSeverity.CRITICAL
        assert critical.score_uplift == Decimal("3.5")

    def test_gap_item_levels(self, sample_gap_items):
        critical = sample_gap_items[0]
        assert critical.current_level == ScoringLevel.DISCLOSURE
        assert critical.target_level == ScoringLevel.LEADERSHIP


# ===========================================================================
# Benchmark model tests
# ===========================================================================

class TestCDPBenchmark:
    """Test CDPBenchmark model."""

    def test_create_benchmark(self, sample_benchmark):
        assert sample_benchmark.sector_gics == "20101010"
        assert sample_benchmark.respondent_count == 450

    def test_benchmark_distribution(self, sample_benchmark):
        assert isinstance(sample_benchmark.score_distribution, dict)
        total = sum(sample_benchmark.score_distribution.values())
        assert total == sample_benchmark.respondent_count


# ===========================================================================
# Supply chain model tests
# ===========================================================================

class TestCDPSupplyChainRequest:
    """Test CDPSupplyChainRequest model."""

    def test_create_supplier_request(self, sample_supplier):
        assert sample_supplier.supplier_name == "Parts Supplier Co"
        assert sample_supplier.status == SupplierStatus.INVITED


# ===========================================================================
# Transition plan model tests
# ===========================================================================

class TestCDPTransitionPlan:
    """Test CDPTransitionPlan model."""

    def test_create_transition_plan(self, sample_transition_plan):
        assert sample_transition_plan.target_year == 2050
        assert sample_transition_plan.is_sbti_aligned is True

    def test_transition_investment(self, sample_transition_plan):
        assert sample_transition_plan.total_investment_usd == Decimal("500000000")


class TestCDPTransitionMilestone:
    """Test CDPTransitionMilestone model."""

    def test_create_milestone(self, sample_milestones):
        assert len(sample_milestones) == 2
        first = sample_milestones[0]
        assert first.target_year == 2030
        assert first.target_reduction_pct == Decimal("50.0")


# ===========================================================================
# Verification model tests
# ===========================================================================

class TestCDPVerificationRecord:
    """Test CDPVerificationRecord model."""

    def test_create_verification_record(self, sample_verification_record):
        assert sample_verification_record.scope == "scope_1"
        assert sample_verification_record.coverage_pct == Decimal("100.0")
        assert sample_verification_record.assurance_level == AssuranceLevel.REASONABLE

    def test_verification_dates(self, sample_verification_record):
        assert sample_verification_record.statement_date == date(2025, 6, 15)
        assert sample_verification_record.valid_until == date(2026, 6, 14)


# ===========================================================================
# Submission model tests
# ===========================================================================

class TestCDPSubmission:
    """Test CDPSubmission model."""

    def test_create_submission(self, sample_questionnaire, sample_organization):
        submission = CDPSubmission(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            submission_format=SubmissionFormat.XML,
            file_path="/exports/cdp_2025_submission.xml",
            submission_reference="CDP-2025-SUB-001",
            status="submitted",
        )
        assert submission.submission_format == SubmissionFormat.XML
        assert submission.status == "submitted"
