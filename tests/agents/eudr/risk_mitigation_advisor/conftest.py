# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-025 Risk Mitigation Advisor test suite.

Provides 80+ reusable fixtures for configuration objects, engine instances,
risk inputs, strategy objects, plan objects, capacity enrollments, mitigation
measures, effectiveness records, trigger events, optimization requests,
stakeholder collaboration, provenance helpers, and test data factories.

Fixture count: 80+ fixtures
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.config import (
    RiskMitigationAdvisorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    # Enumerations
    RiskCategory,
    ISO31000TreatmentType,
    ImplementationComplexity,
    PlanStatus,
    PlanPhaseType,
    MilestoneStatus,
    CapacityTier,
    EnrollmentStatus,
    TriggerEventType,
    AdjustmentType,
    StakeholderRole,
    ReportType,
    EUDRCommodity,
    EvidenceQuality,
    # Core Models
    RiskInput,
    MitigationStrategy,
    CostEstimate,
    CostRange,
    RemediationPlan,
    PlanPhase,
    Milestone,
    EvidenceDocument,
    ResponsibleParty,
    EscalationTrigger,
    KPI,
    MitigationMeasure,
    MeasureApplicability,
    EffectivenessRecord,
    CapacityBuildingEnrollment,
    TriggerEvent,
    # Request Models
    RecommendStrategiesRequest,
    CreatePlanRequest,
    EnrollSupplierRequest,
    SearchMeasuresRequest,
    MeasureEffectivenessRequest,
    OptimizeBudgetRequest,
    CollaborateRequest,
    GenerateReportRequest,
    AdaptiveScanRequest,
    # Response Models
    RecommendStrategiesResponse,
    CreatePlanResponse,
    EnrollSupplierResponse,
    SearchMeasuresResponse,
    MeasureEffectivenessResponse,
    OptimizeBudgetResponse,
    CollaborateResponse,
    GenerateReportResponse,
    AdaptiveScanResponse,
    # Constants
    VERSION,
    EUDR_CUTOFF_DATE,
    UPSTREAM_AGENT_COUNT,
    RISK_CATEGORY_COUNT,
    SUPPORTED_COMMODITIES,
    ISO_31000_TYPES,
    CAPACITY_TIER_COUNT,
    MODULES_PER_COMMODITY,
    DEFAULT_TOP_K,
)
from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    get_tracker,
    reset_tracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
)


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------


class DeterministicUUID:
    """Generate sequential identifiers for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:06d}"

    def reset(self) -> None:
        self._counter = 0


# ---------------------------------------------------------------------------
# Helper: Fixed timestamps
# ---------------------------------------------------------------------------

FIXED_DATETIME = datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc)
FIXED_DATE = date(2026, 3, 11)
FIXED_DATE_STR = "2026-03-11"
EUDR_CUTOFF = date(2020, 12, 31)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all module-level singletons between tests."""
    yield
    reset_config()
    reset_tracker()


@pytest.fixture
def test_config() -> RiskMitigationAdvisorConfig:
    """Create test configuration with deterministic settings."""
    cfg = RiskMitigationAdvisorConfig(
        database_url="postgresql://test:test@localhost:5432/test_rma",
        redis_url="redis://localhost:6379/15",
        log_level="DEBUG",
        ml_model_type="rule_based",
        ml_confidence_threshold=Decimal("0.7"),
        deterministic_mode=True,
        top_k_strategies=5,
        shap_enabled=False,
        enable_provenance=True,
        enable_metrics=False,
    )
    set_config(cfg)
    return cfg


@pytest.fixture
def ml_config() -> RiskMitigationAdvisorConfig:
    """Create config with ML mode enabled for ML-specific tests."""
    cfg = RiskMitigationAdvisorConfig(
        database_url="postgresql://test:test@localhost:5432/test_rma",
        redis_url="redis://localhost:6379/15",
        log_level="DEBUG",
        ml_model_type="xgboost",
        ml_confidence_threshold=Decimal("0.7"),
        deterministic_mode=False,
        top_k_strategies=5,
        shap_enabled=True,
        enable_provenance=True,
        enable_metrics=False,
    )
    set_config(cfg)
    return cfg


@pytest.fixture
def minimal_config() -> RiskMitigationAdvisorConfig:
    """Create minimal config with all optional features disabled."""
    cfg = RiskMitigationAdvisorConfig(
        database_url="postgresql://test:test@localhost:5432/test_rma",
        redis_url="redis://localhost:6379/15",
        log_level="WARNING",
        deterministic_mode=True,
        shap_enabled=False,
        enable_provenance=False,
        enable_metrics=False,
        supplier_portal_enabled=False,
        ngo_workspace_enabled=False,
        communication_threads_enabled=False,
        report_scheduling_enabled=False,
    )
    set_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strategy_engine(test_config):
    """Create a StrategySelectionEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.strategy_selection_engine import (
        StrategySelectionEngine,
    )
    return StrategySelectionEngine(config=test_config)


@pytest.fixture
def remediation_engine(test_config):
    """Create a RemediationPlanDesignEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.remediation_plan_design_engine import (
        RemediationPlanDesignEngine,
    )
    return RemediationPlanDesignEngine(config=test_config)


@pytest.fixture
def capacity_engine(test_config):
    """Create a CapacityBuildingManagerEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.capacity_building_manager_engine import (
        CapacityBuildingManagerEngine,
    )
    return CapacityBuildingManagerEngine(config=test_config)


@pytest.fixture
def measure_library_engine(test_config):
    """Create a MeasureLibraryEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.measure_library_engine import (
        MeasureLibraryEngine,
    )
    return MeasureLibraryEngine(config=test_config)


@pytest.fixture
def effectiveness_engine(test_config):
    """Create an EffectivenessTrackingEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.effectiveness_tracking_engine import (
        EffectivenessTrackingEngine,
    )
    return EffectivenessTrackingEngine(config=test_config)


@pytest.fixture
def monitoring_engine(test_config):
    """Create a ContinuousMonitoringEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.continuous_monitoring_engine import (
        ContinuousMonitoringEngine,
    )
    return ContinuousMonitoringEngine(config=test_config)


@pytest.fixture
def optimizer_engine(test_config):
    """Create a CostBenefitOptimizerEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.cost_benefit_optimizer_engine import (
        CostBenefitOptimizerEngine,
    )
    return CostBenefitOptimizerEngine(config=test_config)


@pytest.fixture
def collaboration_engine(test_config):
    """Create a StakeholderCollaborationEngine instance for testing."""
    from greenlang.agents.eudr.risk_mitigation_advisor.stakeholder_collaboration_engine import (
        StakeholderCollaborationEngine,
    )
    return StakeholderCollaborationEngine(config=test_config)


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh provenance tracker for testing."""
    return ProvenanceTracker(
        genesis_hash="TEST-GENESIS-HASH",
        algorithm="sha256",
    )


# ---------------------------------------------------------------------------
# Deterministic UUID fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uuid_gen() -> DeterministicUUID:
    """Create a deterministic UUID generator."""
    return DeterministicUUID(prefix="test")


# ---------------------------------------------------------------------------
# Risk input fixtures (by commodity and scenario)
# ---------------------------------------------------------------------------


@pytest.fixture
def high_risk_input() -> RiskInput:
    """Create a high-risk input profile across all dimensions."""
    return RiskInput(
        operator_id="op-001",
        supplier_id="sup-001",
        country_code="CD",
        commodity="palm_oil",
        country_risk_score=Decimal("85"),
        supplier_risk_score=Decimal("78"),
        commodity_risk_score=Decimal("72"),
        corruption_risk_score=Decimal("80"),
        deforestation_risk_score=Decimal("90"),
        indigenous_rights_score=Decimal("65"),
        protected_areas_score=Decimal("70"),
        legal_compliance_score=Decimal("75"),
        audit_risk_score=Decimal("68"),
        due_diligence_level="enhanced",
        risk_factors={"post_cutoff_deforestation": True, "critical_ncs": 2},
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def medium_risk_input() -> RiskInput:
    """Create a medium-risk input profile."""
    return RiskInput(
        operator_id="op-002",
        supplier_id="sup-002",
        country_code="BR",
        commodity="soya",
        country_risk_score=Decimal("55"),
        supplier_risk_score=Decimal("45"),
        commodity_risk_score=Decimal("50"),
        corruption_risk_score=Decimal("40"),
        deforestation_risk_score=Decimal("60"),
        indigenous_rights_score=Decimal("35"),
        protected_areas_score=Decimal("30"),
        legal_compliance_score=Decimal("42"),
        audit_risk_score=Decimal("38"),
        due_diligence_level="standard",
        risk_factors={},
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def low_risk_input() -> RiskInput:
    """Create a low-risk input profile."""
    return RiskInput(
        operator_id="op-003",
        supplier_id="sup-003",
        country_code="SE",
        commodity="wood",
        country_risk_score=Decimal("10"),
        supplier_risk_score=Decimal("12"),
        commodity_risk_score=Decimal("8"),
        corruption_risk_score=Decimal("5"),
        deforestation_risk_score=Decimal("3"),
        indigenous_rights_score=Decimal("2"),
        protected_areas_score=Decimal("5"),
        legal_compliance_score=Decimal("7"),
        audit_risk_score=Decimal("10"),
        due_diligence_level="simplified",
        risk_factors={},
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def cattle_risk_input() -> RiskInput:
    """High-risk cattle supply chain from Brazil."""
    return RiskInput(
        operator_id="op-cattle",
        supplier_id="sup-cattle-001",
        country_code="BR",
        commodity="cattle",
        country_risk_score=Decimal("70"),
        supplier_risk_score=Decimal("75"),
        commodity_risk_score=Decimal("80"),
        corruption_risk_score=Decimal("45"),
        deforestation_risk_score=Decimal("85"),
        indigenous_rights_score=Decimal("60"),
        protected_areas_score=Decimal("55"),
        legal_compliance_score=Decimal("50"),
        audit_risk_score=Decimal("65"),
        due_diligence_level="enhanced",
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def cocoa_risk_input() -> RiskInput:
    """High-risk cocoa supply chain from Ghana."""
    return RiskInput(
        operator_id="op-cocoa",
        supplier_id="sup-cocoa-001",
        country_code="GH",
        commodity="cocoa",
        country_risk_score=Decimal("65"),
        supplier_risk_score=Decimal("70"),
        commodity_risk_score=Decimal("75"),
        corruption_risk_score=Decimal("55"),
        deforestation_risk_score=Decimal("80"),
        indigenous_rights_score=Decimal("40"),
        protected_areas_score=Decimal("50"),
        legal_compliance_score=Decimal("45"),
        audit_risk_score=Decimal("60"),
        due_diligence_level="enhanced",
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def coffee_risk_input() -> RiskInput:
    """Medium-risk coffee supply chain from Colombia."""
    return RiskInput(
        operator_id="op-coffee",
        supplier_id="sup-coffee-001",
        country_code="CO",
        commodity="coffee",
        country_risk_score=Decimal("50"),
        supplier_risk_score=Decimal("45"),
        commodity_risk_score=Decimal("55"),
        corruption_risk_score=Decimal("40"),
        deforestation_risk_score=Decimal("50"),
        indigenous_rights_score=Decimal("35"),
        protected_areas_score=Decimal("30"),
        legal_compliance_score=Decimal("40"),
        audit_risk_score=Decimal("42"),
        due_diligence_level="standard",
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def rubber_risk_input() -> RiskInput:
    """Medium-risk rubber supply chain from Indonesia."""
    return RiskInput(
        operator_id="op-rubber",
        supplier_id="sup-rubber-001",
        country_code="ID",
        commodity="rubber",
        country_risk_score=Decimal("60"),
        supplier_risk_score=Decimal("55"),
        commodity_risk_score=Decimal("50"),
        corruption_risk_score=Decimal("50"),
        deforestation_risk_score=Decimal("65"),
        indigenous_rights_score=Decimal("45"),
        protected_areas_score=Decimal("40"),
        legal_compliance_score=Decimal("48"),
        audit_risk_score=Decimal("50"),
        due_diligence_level="enhanced",
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def wood_risk_input() -> RiskInput:
    """Low-risk wood supply chain from Finland."""
    return RiskInput(
        operator_id="op-wood",
        supplier_id="sup-wood-001",
        country_code="FI",
        commodity="wood",
        country_risk_score=Decimal("8"),
        supplier_risk_score=Decimal("10"),
        commodity_risk_score=Decimal("12"),
        corruption_risk_score=Decimal("5"),
        deforestation_risk_score=Decimal("5"),
        indigenous_rights_score=Decimal("3"),
        protected_areas_score=Decimal("7"),
        legal_compliance_score=Decimal("5"),
        audit_risk_score=Decimal("8"),
        due_diligence_level="simplified",
        assessment_date=FIXED_DATE,
    )


@pytest.fixture
def all_commodity_risk_inputs(
    cattle_risk_input,
    cocoa_risk_input,
    coffee_risk_input,
    high_risk_input,  # palm_oil
    rubber_risk_input,
    medium_risk_input,  # soya
    wood_risk_input,
) -> Dict[str, RiskInput]:
    """All 7 commodity risk inputs for parametrized tests."""
    return {
        "cattle": cattle_risk_input,
        "cocoa": cocoa_risk_input,
        "coffee": coffee_risk_input,
        "palm_oil": high_risk_input,
        "rubber": rubber_risk_input,
        "soya": medium_risk_input,
        "wood": wood_risk_input,
    }


# ---------------------------------------------------------------------------
# Strategy fixtures
# ---------------------------------------------------------------------------


def _make_cost_range(min_val: str = "1000", max_val: str = "5000") -> CostRange:
    return CostRange(min_value=Decimal(min_val), max_value=Decimal(max_val))


def _make_cost_estimate(
    level: str = "medium",
    min_val: str = "1000",
    max_val: str = "5000",
) -> CostEstimate:
    return CostEstimate(
        level=level,
        range_eur=_make_cost_range(min_val, max_val),
        annual_recurring=False,
    )


@pytest.fixture
def sample_strategy() -> MitigationStrategy:
    """Create a sample mitigation strategy for testing."""
    return MitigationStrategy(
        strategy_id="strat-001",
        name="Enhanced Country Monitoring",
        description="Deploy enhanced monitoring for high-risk country sourcing.",
        risk_categories=[RiskCategory.COUNTRY, RiskCategory.DEFORESTATION],
        iso_31000_type=ISO31000TreatmentType.REDUCE,
        target_risk_factors=["country_risk_score", "deforestation_risk_score"],
        predicted_effectiveness=Decimal("75"),
        confidence_score=Decimal("0.85"),
        cost_estimate=_make_cost_estimate("medium", "5000", "15000"),
        implementation_complexity=ImplementationComplexity.MEDIUM,
        time_to_effect_weeks=8,
        eudr_articles=["Article 10", "Article 11"],
        model_version="1.0.0",
        provenance_hash="abc123def456",
    )


@pytest.fixture
def sample_strategies() -> List[MitigationStrategy]:
    """Create a list of sample strategies."""
    return [
        MitigationStrategy(
            strategy_id=f"strat-{i:03d}",
            name=f"Strategy {i}",
            description=f"Test strategy number {i}.",
            risk_categories=[list(RiskCategory)[i % len(RiskCategory)]],
            iso_31000_type=list(ISO31000TreatmentType)[i % 4],
            predicted_effectiveness=Decimal(str(80 - i * 5)),
            confidence_score=Decimal("0.85"),
            cost_estimate=_make_cost_estimate("medium"),
            implementation_complexity=ImplementationComplexity.MEDIUM,
            time_to_effect_weeks=4 + i * 2,
            model_version="1.0.0",
        )
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# Plan fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_plan() -> RemediationPlan:
    """Create a sample remediation plan for testing."""
    return RemediationPlan(
        plan_id="plan-001",
        operator_id="op-001",
        supplier_id="sup-001",
        plan_name="Palm Oil Supplier Capacity Building - Supplier 001",
        risk_finding_ids=["finding-001", "finding-002"],
        strategy_ids=["strat-001"],
        status=PlanStatus.ACTIVE,
        budget_allocated=Decimal("50000"),
        budget_spent=Decimal("12000"),
        start_date=FIXED_DATE,
        target_end_date=FIXED_DATE + timedelta(weeks=24),
        plan_template="supplier_capacity_building",
        version=1,
    )


@pytest.fixture
def draft_plan() -> RemediationPlan:
    """Create a draft plan for status transition tests."""
    return RemediationPlan(
        plan_id="plan-draft-001",
        operator_id="op-001",
        supplier_id="sup-001",
        plan_name="Draft Plan for Testing",
        status=PlanStatus.DRAFT,
        budget_allocated=Decimal("25000"),
        start_date=FIXED_DATE,
        target_end_date=FIXED_DATE + timedelta(weeks=12),
    )


@pytest.fixture
def create_plan_request() -> CreatePlanRequest:
    """Create a plan creation request."""
    return CreatePlanRequest(
        operator_id="op-001",
        supplier_id="sup-001",
        strategy_ids=["strat-001"],
        risk_finding_ids=["finding-001"],
        template_name="supplier_capacity_building",
        budget_eur=Decimal("50000"),
        target_duration_weeks=24,
    )


# ---------------------------------------------------------------------------
# Milestone fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_milestone() -> Milestone:
    """Create a sample milestone for testing."""
    return Milestone(
        milestone_id="ms-001",
        plan_id="plan-001",
        name="Complete baseline assessment",
        description="Assess current supplier compliance baseline.",
        phase=PlanPhaseType.PREPARATION,
        due_date=FIXED_DATE + timedelta(weeks=2),
        status=MilestoneStatus.PENDING,
        kpi_target="baseline_score >= 60",
        evidence_required=["report", "checklist"],
        eudr_article="Article 10",
    )


@pytest.fixture
def completed_milestone() -> Milestone:
    """Create a completed milestone for testing."""
    return Milestone(
        milestone_id="ms-002",
        plan_id="plan-001",
        name="Deploy monitoring technology",
        phase=PlanPhaseType.IMPLEMENTATION,
        due_date=FIXED_DATE + timedelta(weeks=6),
        completed_date=FIXED_DATE + timedelta(weeks=5),
        status=MilestoneStatus.COMPLETED,
        evidence_required=["report"],
        evidence_uploaded=[
            EvidenceDocument(
                document_id="doc-001",
                name="deployment_report.pdf",
                document_type="report",
                content_hash="sha256_test_hash",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Capacity building fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_enrollment() -> CapacityBuildingEnrollment:
    """Create a sample capacity building enrollment."""
    return CapacityBuildingEnrollment(
        enrollment_id="enr-001",
        supplier_id="sup-001",
        program_id="prog-palm-oil",
        commodity="palm_oil",
        current_tier=1,
        modules_completed=3,
        modules_total=22,
        competency_scores={"module_1": Decimal("75"), "module_2": Decimal("80")},
        target_completion_date=FIXED_DATE + timedelta(weeks=24),
        status=EnrollmentStatus.ACTIVE,
        risk_score_at_enrollment=Decimal("78"),
        current_risk_score=Decimal("65"),
    )


@pytest.fixture
def enroll_supplier_request() -> EnrollSupplierRequest:
    """Create a supplier enrollment request."""
    return EnrollSupplierRequest(
        supplier_id="sup-001",
        commodity="palm_oil",
        initial_tier=1,
        target_completion_weeks=24,
    )


# ---------------------------------------------------------------------------
# Mitigation measure fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_measure() -> MitigationMeasure:
    """Create a sample mitigation measure from the library."""
    return MitigationMeasure(
        measure_id="meas-001",
        name="Satellite Monitoring Enhancement",
        description="Deploy high-resolution satellite monitoring for supplier plots.",
        risk_category=RiskCategory.DEFORESTATION,
        sub_category="monitoring",
        target_risk_factors=["deforestation_risk_score"],
        effectiveness_rating=Decimal("80"),
        cost_estimate_eur=CostRange(
            min_value=Decimal("5000"), max_value=Decimal("15000")
        ),
        implementation_complexity=ImplementationComplexity.MEDIUM,
        time_to_effect_weeks=4,
        expected_risk_reduction_pct=CostRange(
            min_value=Decimal("20"), max_value=Decimal("40")
        ),
        iso_31000_type=ISO31000TreatmentType.REDUCE,
        eudr_articles=["Article 10", "Article 11"],
        certification_schemes=["FSC", "RSPO"],
        tags=["satellite", "monitoring", "deforestation"],
    )


@pytest.fixture
def sample_measures() -> List[MitigationMeasure]:
    """Create a list of sample measures across risk categories."""
    measures = []
    categories = list(RiskCategory)
    for i, cat in enumerate(categories):
        measures.append(
            MitigationMeasure(
                measure_id=f"meas-{i+1:03d}",
                name=f"Measure for {cat.value}",
                description=f"Test measure addressing {cat.value} risk.",
                risk_category=cat,
                effectiveness_rating=Decimal(str(70 + i * 3)),
                cost_estimate_eur=CostRange(
                    min_value=Decimal(str(1000 * (i + 1))),
                    max_value=Decimal(str(3000 * (i + 1))),
                ),
                expected_risk_reduction_pct=CostRange(
                    min_value=Decimal("15"), max_value=Decimal("35")
                ),
                tags=[cat.value, "test"],
            )
        )
    return measures


@pytest.fixture
def search_measures_request() -> SearchMeasuresRequest:
    """Create a measure search request."""
    return SearchMeasuresRequest(
        query="satellite monitoring",
        risk_category=RiskCategory.DEFORESTATION,
        limit=20,
        offset=0,
    )


# ---------------------------------------------------------------------------
# Effectiveness tracking fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_effectiveness_record() -> EffectivenessRecord:
    """Create a sample effectiveness measurement record."""
    return EffectivenessRecord(
        record_id="eff-001",
        plan_id="plan-001",
        supplier_id="sup-001",
        baseline_risk_scores={
            "country": Decimal("85"),
            "supplier": Decimal("78"),
            "deforestation": Decimal("90"),
        },
        current_risk_scores={
            "country": Decimal("70"),
            "supplier": Decimal("55"),
            "deforestation": Decimal("60"),
        },
        risk_reduction_pct={
            "country": Decimal("17.65"),
            "supplier": Decimal("29.49"),
            "deforestation": Decimal("33.33"),
        },
        composite_reduction_pct=Decimal("26.82"),
        predicted_reduction_pct=Decimal("30.00"),
        deviation_pct=Decimal("-3.18"),
        roi=Decimal("245.50"),
        cost_to_date=Decimal("12000"),
        statistical_significance=True,
        p_value=Decimal("0.023"),
    )


@pytest.fixture
def effectiveness_request() -> MeasureEffectivenessRequest:
    """Create an effectiveness measurement request."""
    return MeasureEffectivenessRequest(
        plan_id="plan-001",
        supplier_id="sup-001",
        include_roi=True,
        include_statistics=True,
    )


# ---------------------------------------------------------------------------
# Trigger event and monitoring fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deforestation_trigger() -> TriggerEvent:
    """Create a deforestation alert trigger event."""
    return TriggerEvent(
        event_id="evt-001",
        event_type=TriggerEventType.DEFORESTATION_ALERT,
        severity="critical",
        source_agent="EUDR-020",
        plan_ids=["plan-001"],
        supplier_id="sup-001",
        description="New deforestation alert detected on monitored plot.",
        risk_score_before=Decimal("60"),
        risk_score_after=Decimal("90"),
        recommended_adjustment=AdjustmentType.EMERGENCY_RESPONSE,
        response_sla_hours=4,
    )


@pytest.fixture
def country_reclassification_trigger() -> TriggerEvent:
    """Create a country reclassification trigger event."""
    return TriggerEvent(
        event_id="evt-002",
        event_type=TriggerEventType.COUNTRY_RECLASSIFICATION,
        severity="high",
        source_agent="EUDR-016",
        plan_ids=["plan-001", "plan-002"],
        supplier_id=None,
        description="Country reclassified from standard to high risk.",
        risk_score_before=Decimal("45"),
        risk_score_after=Decimal("75"),
        recommended_adjustment=AdjustmentType.SCOPE_EXPANSION,
        response_sla_hours=48,
    )


@pytest.fixture
def supplier_spike_trigger() -> TriggerEvent:
    """Create a supplier risk spike trigger event."""
    return TriggerEvent(
        event_id="evt-003",
        event_type=TriggerEventType.SUPPLIER_RISK_SPIKE,
        severity="high",
        source_agent="EUDR-017",
        plan_ids=["plan-003"],
        supplier_id="sup-003",
        description="Supplier risk score increased by 35%.",
        risk_score_before=Decimal("40"),
        risk_score_after=Decimal("54"),
        recommended_adjustment=AdjustmentType.PLAN_ACCELERATION,
        response_sla_hours=24,
    )


@pytest.fixture
def adaptive_scan_request() -> AdaptiveScanRequest:
    """Create an adaptive management scan request."""
    return AdaptiveScanRequest(
        operator_id="op-001",
        plan_ids=["plan-001"],
        include_recommendations=True,
    )


# ---------------------------------------------------------------------------
# Optimization fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def optimize_budget_request() -> OptimizeBudgetRequest:
    """Create a budget optimization request."""
    return OptimizeBudgetRequest(
        operator_id="op-001",
        total_budget_eur=Decimal("500000"),
        per_supplier_cap_eur=Decimal("50000"),
        supplier_ids=["sup-001", "sup-002", "sup-003"],
        candidate_measure_ids=["meas-001", "meas-002", "meas-003"],
    )


@pytest.fixture
def small_optimization_request() -> OptimizeBudgetRequest:
    """Small optimization for performance testing."""
    return OptimizeBudgetRequest(
        operator_id="op-perf",
        total_budget_eur=Decimal("100000"),
        supplier_ids=[f"sup-{i}" for i in range(10)],
        candidate_measure_ids=[f"meas-{i}" for i in range(5)],
    )


# ---------------------------------------------------------------------------
# Collaboration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def collaborate_message_request() -> CollaborateRequest:
    """Create a collaboration message request."""
    return CollaborateRequest(
        action="message",
        plan_id="plan-001",
        stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
        message="Please review the latest milestone evidence.",
    )


@pytest.fixture
def collaborate_task_request() -> CollaborateRequest:
    """Create a collaboration task assignment request."""
    return CollaborateRequest(
        action="task",
        plan_id="plan-001",
        stakeholder_role=StakeholderRole.PROCUREMENT,
        task_assignments=[
            {
                "assignee": "supplier-team",
                "task": "Upload GPS coordinates",
                "due_date": "2026-04-15",
            }
        ],
    )


# ---------------------------------------------------------------------------
# Report fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generate_dds_report_request() -> GenerateReportRequest:
    """Create a DDS mitigation report request."""
    return GenerateReportRequest(
        report_type=ReportType.DDS_MITIGATION,
        operator_id="op-001",
        supplier_id="sup-001",
        format="pdf",
        language="en",
    )


@pytest.fixture
def generate_portfolio_report_request() -> GenerateReportRequest:
    """Create a portfolio summary report request."""
    return GenerateReportRequest(
        report_type=ReportType.PORTFOLIO_SUMMARY,
        operator_id="op-001",
        format="json",
        language="en",
    )


# ---------------------------------------------------------------------------
# Request model fixtures (recommend strategies)
# ---------------------------------------------------------------------------


@pytest.fixture
def recommend_request(high_risk_input) -> RecommendStrategiesRequest:
    """Create a strategy recommendation request."""
    return RecommendStrategiesRequest(
        risk_input=high_risk_input,
        top_k=5,
        deterministic_mode=True,
        include_shap=False,
    )


@pytest.fixture
def recommend_ml_request(high_risk_input) -> RecommendStrategiesRequest:
    """Create an ML-mode strategy recommendation request."""
    return RecommendStrategiesRequest(
        risk_input=high_risk_input,
        top_k=5,
        deterministic_mode=False,
        include_shap=True,
    )


# ---------------------------------------------------------------------------
# Batch / list fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def supplier_batch() -> List[str]:
    """Create a batch of supplier IDs for batch testing."""
    return [f"sup-{i:03d}" for i in range(1, 51)]


@pytest.fixture
def risk_input_batch(high_risk_input) -> List[RiskInput]:
    """Create a batch of risk inputs for batch testing."""
    inputs = []
    for i in range(20):
        ri = RiskInput(
            operator_id=f"op-{i:03d}",
            supplier_id=f"sup-{i:03d}",
            country_code="BR",
            commodity=SUPPORTED_COMMODITIES[i % len(SUPPORTED_COMMODITIES)],
            country_risk_score=Decimal(str(40 + i * 2)),
            supplier_risk_score=Decimal(str(35 + i * 2)),
            commodity_risk_score=Decimal(str(30 + i * 2)),
            corruption_risk_score=Decimal(str(25 + i)),
            deforestation_risk_score=Decimal(str(45 + i * 2)),
            indigenous_rights_score=Decimal(str(20 + i)),
            protected_areas_score=Decimal(str(15 + i)),
            legal_compliance_score=Decimal(str(30 + i)),
            audit_risk_score=Decimal(str(25 + i)),
            assessment_date=FIXED_DATE,
        )
        inputs.append(ri)
    return inputs


# ---------------------------------------------------------------------------
# Composite weight fixtures
# ---------------------------------------------------------------------------


COMPOSITE_WEIGHTS = {
    "country": Decimal("0.15"),
    "supplier": Decimal("0.15"),
    "commodity": Decimal("0.10"),
    "corruption": Decimal("0.10"),
    "deforestation": Decimal("0.20"),
    "indigenous_rights": Decimal("0.10"),
    "protected_areas": Decimal("0.10"),
    "legal_compliance": Decimal("0.10"),
}


@pytest.fixture
def composite_weights() -> Dict[str, Decimal]:
    """Default composite risk weights."""
    return dict(COMPOSITE_WEIGHTS)


# ---------------------------------------------------------------------------
# Mock DB / Redis fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
    db.fetchone = AsyncMock(return_value=None)
    db.fetchall = AsyncMock(return_value=[])
    return db


@pytest.fixture
def mock_redis():
    """Create a mock Redis connection."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.exists = AsyncMock(return_value=False)
    return redis


# ---------------------------------------------------------------------------
# Plan template names constant
# ---------------------------------------------------------------------------

PLAN_TEMPLATES = [
    "supplier_capacity_building",
    "emergency_deforestation_response",
    "certification_enrollment",
    "enhanced_monitoring_deployment",
    "fpic_remediation",
    "legal_gap_closure",
    "anti_corruption_measures",
    "buffer_zone_restoration",
]


@pytest.fixture
def plan_template_names() -> List[str]:
    """All 8 plan template names."""
    return list(PLAN_TEMPLATES)


# ---------------------------------------------------------------------------
# Stakeholder role fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def all_stakeholder_roles() -> List[StakeholderRole]:
    """All 6 stakeholder roles."""
    return list(StakeholderRole)


@pytest.fixture
def internal_party() -> ResponsibleParty:
    """Internal compliance team responsible party."""
    return ResponsibleParty(
        party_id="party-001",
        name="Maria Compliance",
        role=StakeholderRole.INTERNAL_COMPLIANCE,
        email="maria@operator.eu",
        organization="EU Operator GmbH",
    )


@pytest.fixture
def supplier_party() -> ResponsibleParty:
    """Supplier responsible party."""
    return ResponsibleParty(
        party_id="party-002",
        name="Supplier Contact",
        role=StakeholderRole.SUPPLIER,
        email="contact@supplier.com",
        organization="Supplier Co",
    )
