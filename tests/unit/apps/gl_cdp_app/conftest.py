# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GL-CDP-APP v1.0 test suite.

Provides reusable fixtures for configuration, organizations, questionnaires,
modules, questions, responses, scoring results, gap analyses, benchmarks,
suppliers, transition plans, verification records, and mock database sessions
used across all 13 test modules.

Author: GL-TestEngineer
Date: March 2026
"""

import sys
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure the CDP services package is importable
# ---------------------------------------------------------------------------
_SERVICES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "applications", "GL-CDP-APP", "CDP-Disclosure-Platform",
)
_SERVICES_DIR = os.path.normpath(_SERVICES_DIR)
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from services.config import (
    CDPAppConfig,
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
from services.questionnaire_engine import QuestionnaireEngine
from services.response_manager import ResponseManager
from services.scoring_simulator import ScoringSimulator
from services.data_connector import DataConnector
from services.gap_analysis_engine import GapAnalysisEngine
from services.benchmarking_engine import BenchmarkingEngine
from services.supply_chain_module import SupplyChainModule
from services.transition_plan_engine import TransitionPlanEngine
from services.verification_tracker import VerificationTracker
from services.historical_tracker import HistoricalTracker
from services.report_generator import ReportGenerator


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def default_config() -> CDPAppConfig:
    """Default CDP application configuration."""
    return CDPAppConfig()


@pytest.fixture
def custom_config() -> CDPAppConfig:
    """Custom configuration with adjusted parameters for testing."""
    return CDPAppConfig(
        reporting_year=2025,
        questionnaire_version="2025",
        scoring_confidence_level=90,
        a_level_scope3_verification_threshold=Decimal("70.0"),
    )


# ============================================================================
# ORGANIZATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_organization() -> CDPOrganization:
    """Sample CDP organization for testing."""
    return CDPOrganization(
        name="Acme Manufacturing Corp",
        sector_gics="20101010",
        region="North America",
        country="US",
        employee_count=5000,
        revenue_usd=Decimal("2500000000"),
        fiscal_year_end="12-31",
        cdp_account_number="CDP-2025-ACME-001",
    )


@pytest.fixture
def financial_services_org() -> CDPOrganization:
    """Financial Services organization (triggers Module 12)."""
    return CDPOrganization(
        name="Global Finance Holdings",
        sector_gics="40101010",
        region="Europe",
        country="GB",
        employee_count=15000,
        revenue_usd=Decimal("8000000000"),
        fiscal_year_end="03-31",
        cdp_account_number="CDP-2025-GFH-001",
    )


# ============================================================================
# QUESTIONNAIRE FIXTURES
# ============================================================================

@pytest.fixture
def sample_questionnaire(sample_organization) -> CDPQuestionnaire:
    """Sample questionnaire instance for 2025."""
    return CDPQuestionnaire(
        org_id=sample_organization.id,
        reporting_year=2025,
        questionnaire_version="2025",
        status="in_progress",
        sector_specific_modules={"M12": False, "M8": False, "M9": False},
    )


@pytest.fixture
def fs_questionnaire(financial_services_org) -> CDPQuestionnaire:
    """Financial Services questionnaire with M12 enabled."""
    return CDPQuestionnaire(
        org_id=financial_services_org.id,
        reporting_year=2025,
        questionnaire_version="2025",
        status="in_progress",
        sector_specific_modules={"M12": True, "M8": False, "M9": False},
    )


# ============================================================================
# MODULE FIXTURES
# ============================================================================

@pytest.fixture
def sample_module(sample_questionnaire) -> CDPModuleInstance:
    """Sample module M7 (Environmental Performance - Climate Change)."""
    return CDPModuleInstance(
        questionnaire_id=sample_questionnaire.id,
        module_code="M7",
        module_name="Environmental Performance - Climate Change",
        description="Scope 1/2/3 emissions, methodology, verification",
        question_count=35,
        is_applicable=True,
        is_sector_specific=False,
        sort_order=7,
    )


# ============================================================================
# QUESTION FIXTURES
# ============================================================================

@pytest.fixture
def sample_question(sample_module) -> CDPQuestion:
    """Sample question from M7 module."""
    return CDPQuestion(
        module_id=sample_module.id,
        question_number="C7.1",
        question_text="Does your organization have an emissions inventory?",
        question_type=QuestionType.YES_NO,
        guidance_text="Indicate whether you maintain a GHG inventory.",
        is_required=True,
        is_conditional=False,
        condition_logic=None,
        scoring_category=ScoringCategory.SCOPE_1_2_EMISSIONS,
        disclosure_points=2,
        awareness_points=0,
        management_points=0,
        leadership_points=0,
        version_year=2025,
    )


@pytest.fixture
def sample_questions_list(sample_module) -> List[CDPQuestion]:
    """Generate a list of sample questions for testing."""
    questions = []
    for i in range(1, 11):
        q = CDPQuestion(
            module_id=sample_module.id,
            question_number=f"C7.{i}",
            question_text=f"Sample question {i} for Module 7",
            question_type=QuestionType.TEXT if i % 3 == 0 else QuestionType.NUMERIC,
            guidance_text=f"Guidance for question {i}",
            is_required=(i <= 7),
            is_conditional=(i > 7),
            condition_logic={"depends_on": f"C7.{i-1}", "value": "yes"} if i > 7 else None,
            scoring_category=ScoringCategory.SCOPE_1_2_EMISSIONS,
            disclosure_points=2,
            awareness_points=1 if i > 3 else 0,
            management_points=1 if i > 5 else 0,
            leadership_points=1 if i > 8 else 0,
            version_year=2025,
        )
        questions.append(q)
    return questions


# ============================================================================
# RESPONSE FIXTURES
# ============================================================================

@pytest.fixture
def sample_response(sample_question, sample_questionnaire, sample_organization) -> CDPResponse:
    """Sample response in draft status."""
    return CDPResponse(
        question_id=sample_question.id,
        questionnaire_id=sample_questionnaire.id,
        org_id=sample_organization.id,
        response_content={"answer": "yes"},
        response_text="Yes, we maintain a comprehensive GHG inventory.",
        response_status=ResponseStatus.DRAFT,
        auto_populated=False,
        confidence_score=Decimal("0.95"),
    )


@pytest.fixture
def auto_populated_response(sample_question, sample_questionnaire, sample_organization) -> CDPResponse:
    """Sample auto-populated response from MRV agent."""
    return CDPResponse(
        question_id=sample_question.id,
        questionnaire_id=sample_questionnaire.id,
        org_id=sample_organization.id,
        response_content={"scope1_total": 12500.5, "unit": "tCO2e"},
        response_text="12,500.5 tCO2e",
        response_status=ResponseStatus.DRAFT,
        auto_populated=True,
        auto_populated_source="MRV-001 Stationary Combustion",
        confidence_score=Decimal("0.98"),
    )


# ============================================================================
# SCORING FIXTURES
# ============================================================================

@pytest.fixture
def sample_scoring_result(sample_questionnaire, sample_organization) -> CDPScoringResult:
    """Sample scoring result."""
    return CDPScoringResult(
        questionnaire_id=sample_questionnaire.id,
        org_id=sample_organization.id,
        overall_score=Decimal("72.5"),
        score_band="A-",
        predicted_band="A-",
        confidence_lower=Decimal("68.0"),
        confidence_upper=Decimal("77.0"),
        a_level_eligible=False,
    )


@pytest.fixture
def sample_category_scores(sample_scoring_result) -> List[CDPCategoryScore]:
    """Sample category scores for all 17 categories."""
    categories = [
        ("CAT01", "Governance", Decimal("78.0"), Decimal("0.07")),
        ("CAT02", "Risk management processes", Decimal("65.0"), Decimal("0.06")),
        ("CAT03", "Risk disclosure", Decimal("70.0"), Decimal("0.05")),
        ("CAT04", "Opportunity disclosure", Decimal("60.0"), Decimal("0.05")),
        ("CAT05", "Business strategy", Decimal("75.0"), Decimal("0.06")),
        ("CAT06", "Scenario analysis", Decimal("55.0"), Decimal("0.05")),
        ("CAT07", "Targets", Decimal("80.0"), Decimal("0.08")),
        ("CAT08", "Emissions reduction initiatives", Decimal("72.0"), Decimal("0.07")),
        ("CAT09", "Scope 1 & 2 emissions", Decimal("85.0"), Decimal("0.10")),
        ("CAT10", "Scope 3 emissions", Decimal("68.0"), Decimal("0.08")),
        ("CAT11", "Energy", Decimal("70.0"), Decimal("0.06")),
        ("CAT12", "Carbon pricing", Decimal("50.0"), Decimal("0.04")),
        ("CAT13", "Value chain engagement", Decimal("65.0"), Decimal("0.06")),
        ("CAT14", "Public policy engagement", Decimal("40.0"), Decimal("0.03")),
        ("CAT15", "Transition plan", Decimal("60.0"), Decimal("0.06")),
        ("CAT16", "Portfolio climate performance", Decimal("0.0"), Decimal("0.05")),
        ("CAT17", "Financial impact assessment", Decimal("55.0"), Decimal("0.03")),
    ]
    scores = []
    for code, name, raw, weight in categories:
        scores.append(CDPCategoryScore(
            scoring_result_id=sample_scoring_result.id,
            category_code=code,
            category_name=name,
            raw_score=raw,
            weighted_score=raw * weight,
            weight=weight,
            disclosure_score=raw * Decimal("0.4"),
            awareness_score=raw * Decimal("0.2"),
            management_score=raw * Decimal("0.25"),
            leadership_score=raw * Decimal("0.15"),
        ))
    return scores


# ============================================================================
# GAP ANALYSIS FIXTURES
# ============================================================================

@pytest.fixture
def sample_gap_analysis(sample_questionnaire, sample_organization) -> CDPGapAnalysis:
    """Sample gap analysis result."""
    return CDPGapAnalysis(
        questionnaire_id=sample_questionnaire.id,
        org_id=sample_organization.id,
        total_gaps=25,
        critical_gaps=3,
        high_gaps=7,
        medium_gaps=10,
        low_gaps=5,
        potential_uplift=Decimal("12.5"),
    )


@pytest.fixture
def sample_gap_items(sample_gap_analysis, sample_question) -> List[CDPGapItem]:
    """Sample gap items for testing."""
    return [
        CDPGapItem(
            gap_analysis_id=sample_gap_analysis.id,
            question_id=sample_question.id,
            module_code="M7",
            severity=GapSeverity.CRITICAL,
            current_level=ScoringLevel.DISCLOSURE,
            target_level=ScoringLevel.LEADERSHIP,
            recommendation="Provide third-party verification for Scope 1 and 2.",
            effort=EffortLevel.HIGH,
            score_uplift=Decimal("3.5"),
        ),
        CDPGapItem(
            gap_analysis_id=sample_gap_analysis.id,
            question_id=sample_question.id,
            module_code="M5",
            severity=GapSeverity.HIGH,
            current_level=ScoringLevel.AWARENESS,
            target_level=ScoringLevel.MANAGEMENT,
            recommendation="Develop a 1.5C-aligned transition plan.",
            effort=EffortLevel.HIGH,
            score_uplift=Decimal("2.8"),
        ),
        CDPGapItem(
            gap_analysis_id=sample_gap_analysis.id,
            question_id=sample_question.id,
            module_code="M1",
            severity=GapSeverity.MEDIUM,
            current_level=ScoringLevel.MANAGEMENT,
            target_level=ScoringLevel.LEADERSHIP,
            recommendation="Document board-level climate competency.",
            effort=EffortLevel.MEDIUM,
            score_uplift=Decimal("1.2"),
        ),
    ]


# ============================================================================
# BENCHMARK FIXTURES
# ============================================================================

@pytest.fixture
def sample_benchmark() -> CDPBenchmark:
    """Sample sector benchmark data."""
    return CDPBenchmark(
        sector_gics="20101010",
        region="Global",
        year=2025,
        mean_score=Decimal("55.3"),
        median_score=Decimal("52.0"),
        p25_score=Decimal("38.0"),
        p75_score=Decimal("68.0"),
        a_list_rate=Decimal("4.2"),
        respondent_count=450,
        score_distribution={
            "D-": 45, "D": 67, "C-": 78, "C": 89,
            "B-": 72, "B": 55, "A-": 30, "A": 14,
        },
    )


# ============================================================================
# SUPPLY CHAIN FIXTURES
# ============================================================================

@pytest.fixture
def sample_supplier(sample_organization) -> CDPSupplyChainRequest:
    """Sample supply chain engagement request."""
    return CDPSupplyChainRequest(
        org_id=sample_organization.id,
        supplier_name="Parts Supplier Co",
        supplier_email="sustainability@parts-supplier.com",
        supplier_sector="25501010",
        status=SupplierStatus.INVITED,
    )


# ============================================================================
# TRANSITION PLAN FIXTURES
# ============================================================================

@pytest.fixture
def sample_transition_plan(sample_organization) -> CDPTransitionPlan:
    """Sample 1.5C transition plan."""
    return CDPTransitionPlan(
        org_id=sample_organization.id,
        plan_name="Acme 2050 Net Zero Transition Plan",
        base_year=2020,
        target_year=2050,
        pathway_type=TransitionPathwayType.NET_ZERO,
        is_sbti_aligned=True,
        sbti_status=SBTiStatus.TARGETS_SET,
        total_investment_usd=Decimal("500000000"),
        revenue_alignment_pct=Decimal("35.0"),
        is_publicly_available=True,
        status="active",
    )


@pytest.fixture
def sample_milestones(sample_transition_plan) -> List[CDPTransitionMilestone]:
    """Sample transition milestones."""
    return [
        CDPTransitionMilestone(
            plan_id=sample_transition_plan.id,
            milestone_name="50% Scope 1+2 Reduction",
            target_year=2030,
            target_reduction_pct=Decimal("50.0"),
            scope="scope_1_2",
            technology_lever="renewable_energy",
            investment_usd=Decimal("150000000"),
            status=MilestoneStatus.ON_TRACK,
            progress_pct=Decimal("42.0"),
        ),
        CDPTransitionMilestone(
            plan_id=sample_transition_plan.id,
            milestone_name="Net Zero Scope 1+2+3",
            target_year=2050,
            target_reduction_pct=Decimal("90.0"),
            scope="all_scopes",
            technology_lever="multiple",
            investment_usd=Decimal("500000000"),
            status=MilestoneStatus.ON_TRACK,
            progress_pct=Decimal("15.0"),
        ),
    ]


# ============================================================================
# VERIFICATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_verification_record(sample_organization, sample_questionnaire) -> CDPVerificationRecord:
    """Sample verification record."""
    return CDPVerificationRecord(
        org_id=sample_organization.id,
        questionnaire_id=sample_questionnaire.id,
        scope="scope_1",
        coverage_pct=Decimal("100.0"),
        verifier_name="Big Four Audit LLP",
        verifier_accreditation="ISO 14065",
        assurance_level=AssuranceLevel.REASONABLE,
        verification_standard="ISO 14064-3:2019",
        statement_date=date(2025, 6, 15),
        valid_until=date(2026, 6, 14),
        statement_path="/documents/verification/scope1_2025.pdf",
    )


# ============================================================================
# ENGINE / SERVICE FIXTURES
# ============================================================================

@pytest.fixture
def questionnaire_engine(default_config) -> QuestionnaireEngine:
    """Fresh QuestionnaireEngine instance."""
    return QuestionnaireEngine(default_config)


@pytest.fixture
def response_manager(default_config) -> ResponseManager:
    """Fresh ResponseManager instance."""
    return ResponseManager(default_config)


@pytest.fixture
def scoring_simulator(default_config) -> ScoringSimulator:
    """Fresh ScoringSimulator instance."""
    return ScoringSimulator(default_config)


@pytest.fixture
def data_connector(default_config) -> DataConnector:
    """Fresh DataConnector instance."""
    return DataConnector(default_config)


@pytest.fixture
def gap_analysis_engine(default_config) -> GapAnalysisEngine:
    """Fresh GapAnalysisEngine instance."""
    return GapAnalysisEngine(default_config)


@pytest.fixture
def benchmarking_engine(default_config) -> BenchmarkingEngine:
    """Fresh BenchmarkingEngine instance."""
    return BenchmarkingEngine(default_config)


@pytest.fixture
def supply_chain_module(default_config) -> SupplyChainModule:
    """Fresh SupplyChainModule instance."""
    return SupplyChainModule(default_config)


@pytest.fixture
def transition_plan_engine(default_config) -> TransitionPlanEngine:
    """Fresh TransitionPlanEngine instance."""
    return TransitionPlanEngine(default_config)


@pytest.fixture
def verification_tracker(default_config) -> VerificationTracker:
    """Fresh VerificationTracker instance."""
    return VerificationTracker(default_config)


@pytest.fixture
def historical_tracker(default_config) -> HistoricalTracker:
    """Fresh HistoricalTracker instance."""
    return HistoricalTracker(default_config)


@pytest.fixture
def report_generator(default_config) -> ReportGenerator:
    """Fresh ReportGenerator instance."""
    return ReportGenerator(default_config)


# ============================================================================
# MOCK DATABASE SESSION FIXTURES
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Mock async database session for integration-style tests."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock())
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_db_results():
    """Mock database result set."""
    result = MagicMock()
    result.scalars = MagicMock(return_value=MagicMock())
    result.scalars.return_value.all = MagicMock(return_value=[])
    result.scalars.return_value.first = MagicMock(return_value=None)
    return result
