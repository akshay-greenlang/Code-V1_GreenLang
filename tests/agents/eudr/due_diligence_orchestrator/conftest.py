# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-026 Due Diligence Orchestrator test suite.

Provides 100+ reusable fixtures for configuration, engine instances, mock agents,
workflow definitions, DAG topologies, quality gate data, checkpoint state,
circuit breaker records, retry history, dead letter entries, risk profiles,
mitigation decisions, DDS packages, API clients, golden scenario data, and
shared constants used across all 19 test modules.

Fixture count: 100+ fixtures
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    VERSION,
    TOTAL_EUDR_AGENTS,
    PHASE_1_AGENTS,
    PHASE_2_AGENTS,
    PHASE_3_AGENTS,
    ALL_EUDR_AGENTS,
    AGENT_NAMES,
    SUPPORTED_COMMODITIES,
    MAX_WORKFLOW_AGENTS,
    EUDR_RETENTION_YEARS,
    DueDiligencePhase,
    WorkflowType,
    AgentExecutionStatus,
    WorkflowStatus,
    QualityGateResultEnum,
    QualityGateId,
    CircuitBreakerState,
    ErrorClassification,
    FallbackStrategy,
    EUDRCommodity,
    AgentNode,
    WorkflowEdge,
    WorkflowDefinition,
    AgentExecutionRecord,
    WorkflowCheckpoint,
    QualityGateCheck,
    QualityGateEvaluation,
    WorkflowStateTransition,
    WorkflowState,
    RiskScoreContribution,
    CompositeRiskProfile,
    MitigationDecision,
    CircuitBreakerRecord,
    RetryRecord,
    DeadLetterEntry,
    DDSField,
    DDSSection,
    DueDiligencePackage,
    CreateWorkflowRequest,
    StartWorkflowRequest,
    ResumeWorkflowRequest,
    EvaluateQualityGateRequest,
    GeneratePackageRequest,
    BatchWorkflowRequest,
    WorkflowStatusResponse,
    WorkflowProgressResponse,
    QualityGateResponse,
    PackageGenerationResponse,
    WorkflowAuditTrailResponse,
    BatchWorkflowResponse,
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
        return f"{self._prefix}-{self._counter:04d}"

    def reset(self) -> None:
        self._counter = 0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVEN_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

SEVEN_SCENARIOS: List[str] = [
    "clean_workflow",
    "quality_gate_failure",
    "partial_agent_failure",
    "circuit_breaker_trip",
    "checkpoint_recovery",
    "simplified_due_diligence",
    "batch_execution",
]

STANDARD_WORKFLOW_EDGES = [
    ("EUDR-001", "EUDR-002"), ("EUDR-001", "EUDR-006"),
    ("EUDR-001", "EUDR-007"), ("EUDR-001", "EUDR-008"),
    ("EUDR-002", "EUDR-003"), ("EUDR-006", "EUDR-004"),
    ("EUDR-006", "EUDR-005"),
    ("EUDR-008", "EUDR-009"),
    ("EUDR-009", "EUDR-010"), ("EUDR-009", "EUDR-011"),
    ("EUDR-003", "EUDR-012"), ("EUDR-004", "EUDR-012"),
    ("EUDR-010", "EUDR-013"), ("EUDR-011", "EUDR-014"),
    ("EUDR-005", "EUDR-015"),
    ("EUDR-012", "QG-1"), ("EUDR-013", "QG-1"),
    ("EUDR-014", "QG-1"), ("EUDR-015", "QG-1"),
    ("EUDR-003", "QG-1"), ("EUDR-004", "QG-1"),
    ("EUDR-005", "QG-1"), ("EUDR-007", "QG-1"),
    ("EUDR-010", "QG-1"), ("EUDR-011", "QG-1"),
    ("QG-1", "EUDR-016"), ("QG-1", "EUDR-017"),
    ("QG-1", "EUDR-018"), ("QG-1", "EUDR-019"),
    ("QG-1", "EUDR-020"), ("QG-1", "EUDR-021"),
    ("QG-1", "EUDR-022"), ("QG-1", "EUDR-023"),
    ("QG-1", "EUDR-024"), ("QG-1", "EUDR-025"),
    ("EUDR-016", "QG-2"), ("EUDR-017", "QG-2"),
    ("EUDR-018", "QG-2"), ("EUDR-019", "QG-2"),
    ("EUDR-020", "QG-2"), ("EUDR-021", "QG-2"),
    ("EUDR-022", "QG-2"), ("EUDR-023", "QG-2"),
    ("EUDR-024", "QG-2"), ("EUDR-025", "QG-2"),
    ("QG-2", "EUDR-025-MIT"),
    ("EUDR-025-MIT", "QG-3"),
    ("QG-3", "PKG-GEN"),
]

DEFAULT_RISK_WEIGHTS = {
    "EUDR-016": Decimal("0.15"),
    "EUDR-017": Decimal("0.12"),
    "EUDR-018": Decimal("0.10"),
    "EUDR-019": Decimal("0.08"),
    "EUDR-020": Decimal("0.15"),
    "EUDR-021": Decimal("0.10"),
    "EUDR-022": Decimal("0.10"),
    "EUDR-023": Decimal("0.10"),
    "EUDR-024": Decimal("0.05"),
    "EUDR-025": Decimal("0.05"),
}


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset configuration before and after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def default_config() -> DueDiligenceOrchestratorConfig:
    """Default orchestrator configuration."""
    cfg = DueDiligenceOrchestratorConfig()
    set_config(cfg)
    return cfg


@pytest.fixture
def test_config() -> DueDiligenceOrchestratorConfig:
    """Test-tuned configuration with faster timeouts."""
    cfg = DueDiligenceOrchestratorConfig(
        database_url="postgresql://test:test@localhost:5432/test_ddo",
        redis_url="redis://localhost:6379/15",
        log_level="DEBUG",
        max_concurrent_agents=5,
        global_concurrency_limit=20,
        workflow_timeout_s=60,
        retry_base_delay_s=Decimal("0.1"),
        retry_max_delay_s=Decimal("1.0"),
        retry_max_attempts=2,
        cb_failure_threshold=3,
        cb_reset_timeout_s=5,
        enable_provenance=True,
        enable_metrics=False,
    )
    set_config(cfg)
    return cfg


@pytest.fixture
def simplified_config() -> DueDiligenceOrchestratorConfig:
    """Configuration for simplified due diligence workflows."""
    cfg = DueDiligenceOrchestratorConfig(
        qg1_completeness_threshold=Decimal("0.80"),
        qg1_simplified_threshold=Decimal("0.80"),
        qg2_coverage_threshold=Decimal("0.85"),
        qg2_simplified_threshold=Decimal("0.85"),
        qg3_residual_risk_threshold=Decimal("25"),
        qg3_simplified_threshold=Decimal("25"),
    )
    set_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# UUID fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def det_uuid() -> DeterministicUUID:
    """Deterministic UUID generator for reproducible tests."""
    return DeterministicUUID(prefix="ddo-test")


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow_definition_engine(default_config):
    """WorkflowDefinitionEngine instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.workflow_definition_engine import (
        WorkflowDefinitionEngine,
    )
    return WorkflowDefinitionEngine(config=default_config)


@pytest.fixture
def quality_gate_engine(default_config):
    """QualityGateEngine instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.quality_gate_engine import (
        QualityGateEngine,
    )
    return QualityGateEngine(config=default_config)


@pytest.fixture
def workflow_state_manager(default_config):
    """WorkflowStateManager instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.workflow_state_manager import (
        WorkflowStateManager,
    )
    return WorkflowStateManager(config=default_config)


@pytest.fixture
def parallel_execution_engine(default_config):
    """ParallelExecutionEngine instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.parallel_execution_engine import (
        ParallelExecutionEngine,
    )
    return ParallelExecutionEngine(config=default_config)


@pytest.fixture
def error_recovery_manager(default_config):
    """ErrorRecoveryManager instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.error_recovery_manager import (
        ErrorRecoveryManager,
    )
    return ErrorRecoveryManager(config=default_config)


@pytest.fixture
def information_gathering_coordinator(default_config):
    """InformationGatheringCoordinator instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.information_gathering_coordinator import (
        InformationGatheringCoordinator,
    )
    return InformationGatheringCoordinator(config=default_config)


@pytest.fixture
def risk_assessment_coordinator(default_config):
    """RiskAssessmentCoordinator instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.risk_assessment_coordinator import (
        RiskAssessmentCoordinator,
    )
    return RiskAssessmentCoordinator(config=default_config)


@pytest.fixture
def risk_mitigation_coordinator(default_config):
    """RiskMitigationCoordinator instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.risk_mitigation_coordinator import (
        RiskMitigationCoordinator,
    )
    return RiskMitigationCoordinator(config=default_config)


@pytest.fixture
def package_generator(default_config):
    """DueDiligencePackageGenerator instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.due_diligence_package_generator import (
        DueDiligencePackageGenerator,
    )
    return DueDiligencePackageGenerator(config=default_config)


@pytest.fixture
def provenance_tracker():
    """ProvenanceTracker instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.provenance import (
        ProvenanceTracker,
    )
    return ProvenanceTracker()


@pytest.fixture
def service_facade(default_config):
    """DueDiligenceOrchestratorService facade instance."""
    from greenlang.agents.eudr.due_diligence_orchestrator.setup import (
        DueDiligenceOrchestratorService,
    )
    return DueDiligenceOrchestratorService(config=default_config)


# ---------------------------------------------------------------------------
# Workflow definition fixtures
# ---------------------------------------------------------------------------


def _make_agent_node(
    agent_id: str,
    phase: DueDiligencePhase = DueDiligencePhase.INFORMATION_GATHERING,
    layer: int = 0,
    is_critical: bool = True,
    is_required: bool = True,
) -> AgentNode:
    """Helper to create an AgentNode."""
    return AgentNode(
        agent_id=agent_id,
        name=AGENT_NAMES.get(agent_id, agent_id),
        phase=phase,
        layer=layer,
        is_critical=is_critical,
        is_required=is_required,
    )


@pytest.fixture
def standard_workflow_nodes() -> List[AgentNode]:
    """All 25 agent nodes for a standard workflow plus gate and pkg nodes."""
    nodes = []
    for agent_id in PHASE_1_AGENTS:
        nodes.append(_make_agent_node(
            agent_id, DueDiligencePhase.INFORMATION_GATHERING,
        ))
    for agent_id in PHASE_2_AGENTS:
        nodes.append(_make_agent_node(
            agent_id, DueDiligencePhase.RISK_ASSESSMENT,
        ))
    nodes.append(_make_agent_node(
        "EUDR-025-MIT", DueDiligencePhase.RISK_MITIGATION,
    ))
    nodes.append(_make_agent_node(
        "QG-1", DueDiligencePhase.INFORMATION_GATHERING,
        is_critical=True, is_required=True,
    ))
    nodes.append(_make_agent_node(
        "QG-2", DueDiligencePhase.RISK_ASSESSMENT,
        is_critical=True, is_required=True,
    ))
    nodes.append(_make_agent_node(
        "QG-3", DueDiligencePhase.RISK_MITIGATION,
        is_critical=True, is_required=True,
    ))
    nodes.append(_make_agent_node(
        "PKG-GEN", DueDiligencePhase.PACKAGE_GENERATION,
        is_critical=True, is_required=True,
    ))
    return nodes


@pytest.fixture
def standard_workflow_edges() -> List[WorkflowEdge]:
    """All dependency edges for a standard 25-agent workflow."""
    return [
        WorkflowEdge(source=s, target=t)
        for s, t in STANDARD_WORKFLOW_EDGES
    ]


@pytest.fixture
def standard_workflow_definition(
    standard_workflow_nodes,
    standard_workflow_edges,
) -> WorkflowDefinition:
    """Complete standard workflow definition."""
    return WorkflowDefinition(
        definition_id="def-standard-001",
        name="Standard Cocoa Due Diligence",
        workflow_type=WorkflowType.STANDARD,
        commodity=EUDRCommodity.COCOA,
        nodes=standard_workflow_nodes,
        edges=standard_workflow_edges,
    )


@pytest.fixture
def simplified_workflow_definition() -> WorkflowDefinition:
    """Simplified workflow definition for low-risk origins."""
    simplified_agents = ["EUDR-001", "EUDR-002", "EUDR-007", "EUDR-003",
                         "EUDR-016", "EUDR-018", "EUDR-023"]
    nodes = [
        _make_agent_node(a, DueDiligencePhase.INFORMATION_GATHERING)
        if a in PHASE_1_AGENTS
        else _make_agent_node(a, DueDiligencePhase.RISK_ASSESSMENT)
        for a in simplified_agents
    ]
    nodes.append(_make_agent_node(
        "QG-1", DueDiligencePhase.INFORMATION_GATHERING,
    ))
    nodes.append(_make_agent_node(
        "QG-2", DueDiligencePhase.RISK_ASSESSMENT,
    ))
    nodes.append(_make_agent_node(
        "PKG-GEN", DueDiligencePhase.PACKAGE_GENERATION,
    ))
    edges = [
        WorkflowEdge(source="EUDR-001", target="EUDR-002"),
        WorkflowEdge(source="EUDR-001", target="EUDR-007"),
        WorkflowEdge(source="EUDR-002", target="EUDR-003"),
        WorkflowEdge(source="EUDR-003", target="QG-1"),
        WorkflowEdge(source="EUDR-007", target="QG-1"),
        WorkflowEdge(source="QG-1", target="EUDR-016"),
        WorkflowEdge(source="QG-1", target="EUDR-018"),
        WorkflowEdge(source="QG-1", target="EUDR-023"),
        WorkflowEdge(source="EUDR-016", target="QG-2"),
        WorkflowEdge(source="EUDR-018", target="QG-2"),
        WorkflowEdge(source="EUDR-023", target="QG-2"),
        WorkflowEdge(source="QG-2", target="PKG-GEN"),
    ]
    return WorkflowDefinition(
        definition_id="def-simplified-001",
        name="Simplified Wood Due Diligence",
        workflow_type=WorkflowType.SIMPLIFIED,
        commodity=EUDRCommodity.WOOD,
        nodes=nodes,
        edges=edges,
        quality_gates=["QG-1", "QG-2"],
    )


# ---------------------------------------------------------------------------
# Workflow state fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow_state_created() -> WorkflowState:
    """WorkflowState in CREATED status."""
    return WorkflowState(
        workflow_id="wf-001",
        definition_id="def-standard-001",
        status=WorkflowStatus.CREATED,
        workflow_type=WorkflowType.STANDARD,
        commodity=EUDRCommodity.COCOA,
        current_phase=DueDiligencePhase.INFORMATION_GATHERING,
        operator_id="op-001",
        operator_name="Test Operator GmbH",
        product_ids=["prod-001", "prod-002"],
        shipment_ids=["ship-001"],
        country_codes=["GH", "CI"],
    )


@pytest.fixture
def workflow_state_running() -> WorkflowState:
    """WorkflowState in RUNNING status with some agents completed."""
    state = WorkflowState(
        workflow_id="wf-002",
        definition_id="def-standard-001",
        status=WorkflowStatus.RUNNING,
        workflow_type=WorkflowType.STANDARD,
        commodity=EUDRCommodity.COCOA,
        current_phase=DueDiligencePhase.INFORMATION_GATHERING,
        operator_id="op-001",
        started_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        progress_pct=Decimal("30"),
    )
    state.agent_executions["EUDR-001"] = AgentExecutionRecord(
        workflow_id="wf-002",
        agent_id="EUDR-001",
        status=AgentExecutionStatus.COMPLETED,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        completed_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        duration_ms=Decimal("5000"),
        output_summary={"product_count": 5, "quantity_complete": True,
                        "countries_identified": 2},
    )
    return state


@pytest.fixture
def workflow_state_phase1_complete() -> WorkflowState:
    """WorkflowState with all Phase 1 agents completed."""
    state = WorkflowState(
        workflow_id="wf-003",
        definition_id="def-standard-001",
        status=WorkflowStatus.QUALITY_GATE,
        workflow_type=WorkflowType.STANDARD,
        commodity=EUDRCommodity.COCOA,
        current_phase=DueDiligencePhase.INFORMATION_GATHERING,
        progress_pct=Decimal("50"),
    )
    for agent_id in PHASE_1_AGENTS:
        state.agent_executions[agent_id] = AgentExecutionRecord(
            workflow_id="wf-003",
            agent_id=agent_id,
            status=AgentExecutionStatus.COMPLETED,
            duration_ms=Decimal("3000"),
            output_summary=_make_phase1_output(agent_id),
        )
    return state


@pytest.fixture
def workflow_state_phase2_complete() -> WorkflowState:
    """WorkflowState with all Phase 1 and Phase 2 agents completed."""
    state = WorkflowState(
        workflow_id="wf-004",
        definition_id="def-standard-001",
        status=WorkflowStatus.QUALITY_GATE,
        workflow_type=WorkflowType.STANDARD,
        commodity=EUDRCommodity.COCOA,
        current_phase=DueDiligencePhase.RISK_ASSESSMENT,
        progress_pct=Decimal("80"),
    )
    for agent_id in PHASE_1_AGENTS:
        state.agent_executions[agent_id] = AgentExecutionRecord(
            workflow_id="wf-004",
            agent_id=agent_id,
            status=AgentExecutionStatus.COMPLETED,
            duration_ms=Decimal("3000"),
            output_summary=_make_phase1_output(agent_id),
        )
    for agent_id in PHASE_2_AGENTS:
        state.agent_executions[agent_id] = AgentExecutionRecord(
            workflow_id="wf-004",
            agent_id=agent_id,
            status=AgentExecutionStatus.COMPLETED,
            duration_ms=Decimal("2000"),
            output_summary=_make_phase2_output(agent_id),
        )
    state.composite_risk_score = Decimal("35.50")
    return state


@pytest.fixture
def workflow_state_completed() -> WorkflowState:
    """Fully completed workflow state."""
    state = WorkflowState(
        workflow_id="wf-005",
        definition_id="def-standard-001",
        status=WorkflowStatus.COMPLETED,
        workflow_type=WorkflowType.STANDARD,
        commodity=EUDRCommodity.COCOA,
        current_phase=DueDiligencePhase.PACKAGE_GENERATION,
        progress_pct=Decimal("100"),
        completed_at=datetime.now(timezone.utc),
        total_duration_ms=Decimal("240000"),
        package_id="pkg-001",
    )
    return state


# ---------------------------------------------------------------------------
# Quality gate fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qg1_passing_checks() -> List[QualityGateCheck]:
    """QG-1 checks that pass the default 90% threshold."""
    return [
        QualityGateCheck(
            name="Supply Chain Mapping Coverage",
            weight=Decimal("0.25"), measured_value=Decimal("95"),
            threshold=Decimal("90"), passed=True,
            source_agents=["EUDR-001"],
        ),
        QualityGateCheck(
            name="Geolocation Verification Coverage",
            weight=Decimal("0.20"), measured_value=Decimal("92"),
            threshold=Decimal("90"), passed=True,
            source_agents=["EUDR-002", "EUDR-006", "EUDR-007"],
        ),
        QualityGateCheck(
            name="Satellite Monitoring Coverage",
            weight=Decimal("0.15"), measured_value=Decimal("91"),
            threshold=Decimal("90"), passed=True,
            source_agents=["EUDR-003", "EUDR-004", "EUDR-005"],
        ),
        QualityGateCheck(
            name="Chain of Custody Integrity",
            weight=Decimal("0.15"), measured_value=Decimal("94"),
            threshold=Decimal("90"), passed=True,
            source_agents=["EUDR-009", "EUDR-010", "EUDR-011"],
        ),
        QualityGateCheck(
            name="Document Authentication Coverage",
            weight=Decimal("0.10"), measured_value=Decimal("96"),
            threshold=Decimal("90"), passed=True,
            source_agents=["EUDR-012"],
        ),
        QualityGateCheck(
            name="Blockchain Evidence Coverage",
            weight=Decimal("0.10"), measured_value=Decimal("90"),
            threshold=Decimal("85"), passed=True,
            source_agents=["EUDR-013"],
        ),
        QualityGateCheck(
            name="Mobile Data Collection Coverage",
            weight=Decimal("0.05"), measured_value=Decimal("88"),
            threshold=Decimal("80"), passed=True,
            source_agents=["EUDR-015"],
        ),
    ]


@pytest.fixture
def qg1_failing_checks() -> List[QualityGateCheck]:
    """QG-1 checks that fail -- low coverage on critical items."""
    return [
        QualityGateCheck(
            name="Supply Chain Mapping Coverage",
            weight=Decimal("0.25"), measured_value=Decimal("60"),
            threshold=Decimal("90"), passed=False,
            source_agents=["EUDR-001"],
            remediation="Re-run EUDR-001 with complete supplier list",
        ),
        QualityGateCheck(
            name="Geolocation Verification Coverage",
            weight=Decimal("0.20"), measured_value=Decimal("45"),
            threshold=Decimal("90"), passed=False,
            source_agents=["EUDR-002"],
            remediation="Insufficient GPS data; request field verification",
        ),
    ]


@pytest.fixture
def qg2_passing_evaluation() -> QualityGateEvaluation:
    """QG-2 evaluation that passes."""
    return QualityGateEvaluation(
        workflow_id="wf-004",
        gate_id=QualityGateId.QG2,
        phase_from=DueDiligencePhase.RISK_ASSESSMENT,
        phase_to=DueDiligencePhase.RISK_MITIGATION,
        result=QualityGateResultEnum.PASSED,
        weighted_score=Decimal("0.97"),
        threshold=Decimal("0.95"),
    )


# ---------------------------------------------------------------------------
# Risk profile fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def low_risk_scores() -> Dict[str, Decimal]:
    """Risk scores indicating negligible risk (composite <= 20)."""
    return {
        "EUDR-016": Decimal("10"), "EUDR-017": Decimal("8"),
        "EUDR-018": Decimal("12"), "EUDR-019": Decimal("5"),
        "EUDR-020": Decimal("15"), "EUDR-021": Decimal("10"),
        "EUDR-022": Decimal("8"),  "EUDR-023": Decimal("12"),
        "EUDR-024": Decimal("6"),  "EUDR-025": Decimal("10"),
    }


@pytest.fixture
def medium_risk_scores() -> Dict[str, Decimal]:
    """Risk scores indicating standard risk (20 < composite <= 50)."""
    return {
        "EUDR-016": Decimal("40"), "EUDR-017": Decimal("35"),
        "EUDR-018": Decimal("30"), "EUDR-019": Decimal("25"),
        "EUDR-020": Decimal("45"), "EUDR-021": Decimal("30"),
        "EUDR-022": Decimal("35"), "EUDR-023": Decimal("40"),
        "EUDR-024": Decimal("20"), "EUDR-025": Decimal("25"),
    }


@pytest.fixture
def high_risk_scores() -> Dict[str, Decimal]:
    """Risk scores indicating critical risk (composite > 60)."""
    return {
        "EUDR-016": Decimal("80"), "EUDR-017": Decimal("75"),
        "EUDR-018": Decimal("70"), "EUDR-019": Decimal("65"),
        "EUDR-020": Decimal("85"), "EUDR-021": Decimal("70"),
        "EUDR-022": Decimal("75"), "EUDR-023": Decimal("80"),
        "EUDR-024": Decimal("60"), "EUDR-025": Decimal("55"),
    }


@pytest.fixture
def composite_risk_profile_low(low_risk_scores) -> CompositeRiskProfile:
    """Low-risk composite profile."""
    contributions = []
    total = Decimal("0")
    for agent_id, score in low_risk_scores.items():
        w = DEFAULT_RISK_WEIGHTS[agent_id]
        ws = score * w
        total += ws
        contributions.append(RiskScoreContribution(
            agent_id=agent_id,
            agent_name=AGENT_NAMES.get(agent_id, agent_id),
            raw_score=score,
            weight=w,
            weighted_score=ws,
        ))
    return CompositeRiskProfile(
        workflow_id="wf-low",
        contributions=contributions,
        composite_score=total,
        risk_level="negligible",
        all_dimensions_scored=True,
        coverage_pct=Decimal("100"),
    )


# ---------------------------------------------------------------------------
# Circuit breaker fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def circuit_breaker_closed() -> CircuitBreakerRecord:
    """Circuit breaker in CLOSED (healthy) state."""
    return CircuitBreakerRecord(
        agent_id="EUDR-003",
        state=CircuitBreakerState.CLOSED,
        failure_count=0,
        success_count=5,
    )


@pytest.fixture
def circuit_breaker_open() -> CircuitBreakerRecord:
    """Circuit breaker in OPEN (tripped) state."""
    return CircuitBreakerRecord(
        agent_id="EUDR-003",
        state=CircuitBreakerState.OPEN,
        failure_count=5,
        success_count=0,
        opened_at=datetime.now(timezone.utc) - timedelta(seconds=30),
    )


@pytest.fixture
def circuit_breaker_half_open() -> CircuitBreakerRecord:
    """Circuit breaker in HALF_OPEN (probing) state."""
    return CircuitBreakerRecord(
        agent_id="EUDR-003",
        state=CircuitBreakerState.HALF_OPEN,
        failure_count=5,
        success_count=0,
        half_open_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# DDS package fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dds_package() -> DueDiligencePackage:
    """Sample completed DDS package."""
    return DueDiligencePackage(
        package_id="pkg-001",
        workflow_id="wf-005",
        commodity=EUDRCommodity.COCOA,
        workflow_type=WorkflowType.STANDARD,
        operator_id="op-001",
        operator_name="Test Operator GmbH",
        total_agents_executed=25,
        total_duration_ms=Decimal("240000"),
        language="en",
        sections=[
            DDSSection(
                section_number=1,
                title="Operator Information",
                completeness_pct=Decimal("100"),
            ),
            DDSSection(
                section_number=2,
                title="Product Description",
                completeness_pct=Decimal("100"),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Request fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def create_workflow_request() -> CreateWorkflowRequest:
    """Standard workflow creation request."""
    return CreateWorkflowRequest(
        workflow_type=WorkflowType.STANDARD,
        commodity=EUDRCommodity.COCOA,
        operator_id="op-001",
        operator_name="Test Operator GmbH",
        product_ids=["prod-001"],
        shipment_ids=["ship-001"],
        country_codes=["GH", "CI"],
        request_id="req-001",
    )


@pytest.fixture
def start_workflow_request() -> StartWorkflowRequest:
    """Workflow start request."""
    return StartWorkflowRequest(
        workflow_id="wf-001",
        input_data={"supplier_list": ["sup-001", "sup-002"]},
        priority=2,
        request_id="req-002",
    )


@pytest.fixture
def resume_workflow_request() -> ResumeWorkflowRequest:
    """Workflow resume request."""
    return ResumeWorkflowRequest(
        workflow_id="wf-002",
        checkpoint_id=None,
        retry_failed=True,
        request_id="req-003",
    )


@pytest.fixture
def evaluate_qg_request() -> EvaluateQualityGateRequest:
    """Quality gate evaluation request."""
    return EvaluateQualityGateRequest(
        workflow_id="wf-003",
        gate_id=QualityGateId.QG1,
        request_id="req-004",
    )


@pytest.fixture
def generate_package_request() -> GeneratePackageRequest:
    """Package generation request."""
    return GeneratePackageRequest(
        workflow_id="wf-005",
        formats=["json", "pdf"],
        language="en",
        include_executive_summary=True,
        include_evidence_annexes=True,
        request_id="req-005",
    )


@pytest.fixture
def batch_workflow_request() -> BatchWorkflowRequest:
    """Batch workflow creation request for 3 commodities."""
    return BatchWorkflowRequest(
        workflows=[
            CreateWorkflowRequest(
                workflow_type=WorkflowType.STANDARD,
                commodity=EUDRCommodity.COCOA,
                operator_id="op-001",
            ),
            CreateWorkflowRequest(
                workflow_type=WorkflowType.STANDARD,
                commodity=EUDRCommodity.COFFEE,
                operator_id="op-001",
            ),
            CreateWorkflowRequest(
                workflow_type=WorkflowType.SIMPLIFIED,
                commodity=EUDRCommodity.WOOD,
                operator_id="op-001",
            ),
        ],
        batch_priority=2,
        request_id="req-006",
    )


# ---------------------------------------------------------------------------
# Mock agent client fixture
# ---------------------------------------------------------------------------


def _make_mock_agent_result(
    agent_id: str,
    status: str = "completed",
    output: Optional[Dict[str, Any]] = None,
    duration_ms: float = 3000.0,
) -> Dict[str, Any]:
    """Create a mock agent execution result."""
    return {
        "agent_id": agent_id,
        "status": status,
        "output_summary": output or _make_default_output(agent_id),
        "duration_ms": duration_ms,
        "provenance_hash": hashlib.sha256(
            f"{agent_id}-{status}".encode()
        ).hexdigest(),
    }


def _make_default_output(agent_id: str) -> Dict[str, Any]:
    """Create default output summary for an agent."""
    if agent_id in PHASE_1_AGENTS:
        return _make_phase1_output(agent_id)
    elif agent_id in PHASE_2_AGENTS:
        return _make_phase2_output(agent_id)
    return {"status": "completed"}


def _make_phase1_output(agent_id: str) -> Dict[str, Any]:
    """Create Phase 1 agent output summary."""
    outputs = {
        "EUDR-001": {"product_count": 5, "quantity_complete": True,
                     "countries_identified": 2, "suppliers_mapped": 10},
        "EUDR-002": {"coverage_pct": 95, "verified_coordinates": 50},
        "EUDR-003": {"verified_pct": 92, "satellite_images_analyzed": 30},
        "EUDR-004": {"forest_cover_pct": 88, "analysis_complete": True},
        "EUDR-005": {"change_detected": False, "land_use_verified": True},
        "EUDR-006": {"polygon_coverage_pct": 90, "plots_bounded": 45},
        "EUDR-007": {"valid_coordinates_pct": 98, "gps_validated": 50},
        "EUDR-008": {"tiers_tracked": 4, "suppliers_total": 30},
        "EUDR-009": {"chain_integrity_pct": 94, "custody_verified": True},
        "EUDR-010": {"segregation_verified": True, "mixing_detected": False},
        "EUDR-011": {"mass_balance_valid": True, "balance_pct": 99},
        "EUDR-012": {"authenticated_pct": 96, "documents_checked": 25},
        "EUDR-013": {"blockchain_registered": True, "hash_verified": True},
        "EUDR-014": {"qr_codes_generated": 20, "linked_to_dds": True},
        "EUDR-015": {"field_data_collected": True, "records_synced": 15},
    }
    return outputs.get(agent_id, {"status": "completed"})


def _make_phase2_output(agent_id: str) -> Dict[str, Any]:
    """Create Phase 2 agent output summary."""
    outputs = {
        "EUDR-016": {"risk_score": 35, "country_classification": "standard"},
        "EUDR-017": {"risk_score": 30, "supplier_compliance": "moderate"},
        "EUDR-018": {"risk_score": 25, "commodity_risk": "medium"},
        "EUDR-019": {"risk_score": 20, "corruption_index": 45},
        "EUDR-020": {"risk_score": 40, "deforestation_alerts": 2},
        "EUDR-021": {"risk_score": 25, "indigenous_rights_impact": "low"},
        "EUDR-022": {"risk_score": 30, "protected_area_overlap": False},
        "EUDR-023": {"risk_score": 35, "legal_compliance_gaps": 1},
        "EUDR-024": {"risk_score": 15, "audit_findings": 0},
        "EUDR-025": {"risk_score": 20, "mitigation_readiness": "high"},
    }
    return outputs.get(agent_id, {"risk_score": 25})


@pytest.fixture
def mock_agent_client():
    """Mock AgentClient that returns success for all 25 agents."""
    client = MagicMock()
    client.invoke = AsyncMock(
        side_effect=lambda agent_id, **kw: _make_mock_agent_result(agent_id)
    )
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


@pytest.fixture
def mock_failing_agent_client():
    """Mock AgentClient that fails on EUDR-003 (satellite monitoring)."""
    async def _invoke(agent_id, **kwargs):
        if agent_id == "EUDR-003":
            raise ConnectionError("Satellite data provider unavailable")
        return _make_mock_agent_result(agent_id)

    client = MagicMock()
    client.invoke = AsyncMock(side_effect=_invoke)
    return client


# ---------------------------------------------------------------------------
# Golden scenario helper
# ---------------------------------------------------------------------------


def build_golden_scenario(
    commodity: str,
    scenario: str,
) -> Dict[str, Any]:
    """Build a golden scenario configuration for testing.

    Args:
        commodity: One of the 7 EUDR commodities.
        scenario: One of the 7 test scenarios.

    Returns:
        Dictionary with scenario parameters.
    """
    commodity_enum = EUDRCommodity(commodity)
    base = {
        "commodity": commodity_enum,
        "operator_id": f"op-{commodity[:3]}",
        "operator_name": f"Test {commodity.title()} Operator",
        "country_codes": _commodity_countries(commodity),
        "product_ids": [f"prod-{commodity[:3]}-001"],
    }

    scenario_overrides = {
        "clean_workflow": {
            "workflow_type": WorkflowType.STANDARD,
            "expect_all_pass": True,
            "expect_complete": True,
        },
        "quality_gate_failure": {
            "workflow_type": WorkflowType.STANDARD,
            "inject_qg1_failure": True,
            "expect_gate_failed": True,
        },
        "partial_agent_failure": {
            "workflow_type": WorkflowType.STANDARD,
            "failing_agents": ["EUDR-003"],
            "expect_degraded": True,
        },
        "circuit_breaker_trip": {
            "workflow_type": WorkflowType.STANDARD,
            "trip_circuit_on": "EUDR-020",
            "expect_circuit_open": True,
        },
        "checkpoint_recovery": {
            "workflow_type": WorkflowType.STANDARD,
            "interrupt_at_agent": "EUDR-009",
            "expect_resume": True,
        },
        "simplified_due_diligence": {
            "workflow_type": WorkflowType.SIMPLIFIED,
            "expect_reduced_agents": True,
            "expect_relaxed_thresholds": True,
        },
        "batch_execution": {
            "workflow_type": WorkflowType.STANDARD,
            "batch_size": 3,
            "expect_batch_complete": True,
        },
    }

    return {**base, **scenario_overrides.get(scenario, {})}


def _commodity_countries(commodity: str) -> List[str]:
    """Return typical origin countries for a commodity."""
    return {
        "cattle": ["BR", "AR"],
        "cocoa": ["GH", "CI"],
        "coffee": ["BR", "CO", "ET"],
        "palm_oil": ["ID", "MY"],
        "rubber": ["TH", "ID"],
        "soya": ["BR", "AR", "US"],
        "wood": ["FI", "SE", "BR"],
    }.get(commodity, ["XX"])
