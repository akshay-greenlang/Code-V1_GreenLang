# -*- coding: utf-8 -*-
"""
AGENT-EUDR-026: Due Diligence Orchestrator Agent

DAG-based workflow orchestrator for EUDR (EU Deforestation Regulation)
due diligence compliance. Orchestrates all 25 upstream EUDR agents
(EUDR-001 through EUDR-025) in a three-phase due diligence process:

    Phase 1 (Article 9):  Information Gathering  -- 15 agents
    Phase 2 (Article 10): Risk Assessment         -- 10 agents
    Phase 3 (Article 11): Risk Mitigation         --  1 agent

Each phase transition is gated by a quality gate:

    QG-1: Information Gathering Completeness  (>= 90% standard, >= 80% simplified)
    QG-2: Risk Assessment Coverage            (>= 95% standard, >= 85% simplified)
    QG-3: Mitigation Adequacy                 (residual risk <= 15 standard, <= 25 simplified)

After all three phases complete, the orchestrator compiles a Due Diligence
Statement (DDS) package per Article 12(2) with full provenance tracking.

Package Structure:
    Core Modules:
        - models.py          -- 10 enums, 18 core models, 6 request, 6 response
        - config.py          -- ~60 environment variables with GL_EUDR_DDO_ prefix
        - provenance.py      -- SHA-256 chain hashing with 14 entity types
        - metrics.py         -- 20 Prometheus metrics with gl_eudr_ddo_ prefix

    Processing Engines:
        - workflow_definition_engine.py          -- DAG + Kahn's algorithm
        - information_gathering_coordinator.py   -- Phase 1 coordinator
        - risk_assessment_coordinator.py         -- Phase 2 coordinator
        - risk_mitigation_coordinator.py         -- Phase 3 coordinator
        - quality_gate_engine.py                 -- QG-1/QG-2/QG-3 evaluator
        - workflow_state_manager.py              -- 11-state machine
        - parallel_execution_engine.py           -- DAG-aware parallelism
        - error_recovery_manager.py              -- Circuit breaker + backoff
        - due_diligence_package_generator.py     -- DDS compiler

    Reference Data:
        - reference_data/workflow_templates.py       -- 7 commodity templates
        - reference_data/quality_gate_rules.py       -- QG check definitions
        - reference_data/article_8_fields.py         -- 26 DDS fields
        - reference_data/error_classifications.py    -- Error taxonomy

    Integration:
        - integration/agent_client.py            -- Generic HTTP client
        - integration/supply_chain_clients.py    -- 15 Phase 1 clients
        - integration/risk_assessment_clients.py -- 10 Phase 2 clients
        - integration/event_bus.py               -- Workflow event bus

    Service Facade:
        - setup.py           -- DueDiligenceOrchestratorService facade

Commodities Supported:
    cattle, cocoa, coffee, palm_oil, rubber, soya, wood

Regulatory References:
    - EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11, 12, 13
    - Due Diligence Statement (DDS) per Article 12(2)
    - Simplified due diligence per Article 13
    - Data retention per Article 31 (5 years)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.agents.eudr.due_diligence_orchestrator.models import VERSION

__version__ = VERSION

__all__: list[str] = [
    "__version__",
    # Service facade
    "DueDiligenceOrchestratorService",
    # Configuration
    "DueDiligenceOrchestratorConfig",
    "get_config",
    # Core engines
    "WorkflowDefinitionEngine",
    "InformationGatheringCoordinator",
    "RiskAssessmentCoordinator",
    "RiskMitigationCoordinator",
    "QualityGateEngine",
    "WorkflowStateManager",
    "ParallelExecutionEngine",
    "ErrorRecoveryManager",
    "DueDiligencePackageGenerator",
    # Provenance and metrics
    "ProvenanceTracker",
    "get_tracker",
    # Integration
    "AgentClient",
    "EventBus",
    "get_event_bus",
    # Models (enums)
    "DueDiligencePhase",
    "WorkflowType",
    "AgentExecutionStatus",
    "WorkflowStatus",
    "QualityGateResultEnum",
    "QualityGateId",
    "CircuitBreakerState",
    "ErrorClassification",
    "FallbackStrategy",
    "EUDRCommodity",
    # Models (core)
    "AgentNode",
    "WorkflowEdge",
    "WorkflowDefinition",
    "AgentExecutionRecord",
    "QualityGateCheck",
    "QualityGateEvaluation",
    "WorkflowState",
    "CompositeRiskProfile",
    "MitigationDecision",
    "DueDiligencePackage",
    # Models (requests)
    "CreateWorkflowRequest",
    "StartWorkflowRequest",
    "ResumeWorkflowRequest",
    "EvaluateQualityGateRequest",
    "GeneratePackageRequest",
    "BatchWorkflowRequest",
    # Models (responses)
    "WorkflowStatusResponse",
    "WorkflowProgressResponse",
    "QualityGateResponse",
    "PackageGenerationResponse",
    "WorkflowAuditTrailResponse",
    "BatchWorkflowResponse",
]


def _lazy_import(name: str) -> object:
    """Lazy import to avoid circular imports at module load time.

    Args:
        name: Name of the attribute to import.

    Returns:
        The imported object.

    Raises:
        AttributeError: If the name is not in __all__.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Service facade
    if name == "DueDiligenceOrchestratorService":
        from greenlang.agents.eudr.due_diligence_orchestrator.setup import (
            DueDiligenceOrchestratorService,
        )
        return DueDiligenceOrchestratorService

    # Configuration
    if name == "DueDiligenceOrchestratorConfig":
        from greenlang.agents.eudr.due_diligence_orchestrator.config import (
            DueDiligenceOrchestratorConfig,
        )
        return DueDiligenceOrchestratorConfig
    if name == "get_config":
        from greenlang.agents.eudr.due_diligence_orchestrator.config import (
            get_config,
        )
        return get_config

    # Engines
    engine_map = {
        "WorkflowDefinitionEngine": (
            "workflow_definition_engine", "WorkflowDefinitionEngine"
        ),
        "InformationGatheringCoordinator": (
            "information_gathering_coordinator",
            "InformationGatheringCoordinator",
        ),
        "RiskAssessmentCoordinator": (
            "risk_assessment_coordinator", "RiskAssessmentCoordinator"
        ),
        "RiskMitigationCoordinator": (
            "risk_mitigation_coordinator", "RiskMitigationCoordinator"
        ),
        "QualityGateEngine": (
            "quality_gate_engine", "QualityGateEngine"
        ),
        "WorkflowStateManager": (
            "workflow_state_manager", "WorkflowStateManager"
        ),
        "ParallelExecutionEngine": (
            "parallel_execution_engine", "ParallelExecutionEngine"
        ),
        "ErrorRecoveryManager": (
            "error_recovery_manager", "ErrorRecoveryManager"
        ),
        "DueDiligencePackageGenerator": (
            "due_diligence_package_generator",
            "DueDiligencePackageGenerator",
        ),
    }
    if name in engine_map:
        module_name, class_name = engine_map[name]
        import importlib
        mod = importlib.import_module(
            f"greenlang.agents.eudr.due_diligence_orchestrator.{module_name}"
        )
        return getattr(mod, class_name)

    # Provenance
    if name == "ProvenanceTracker":
        from greenlang.agents.eudr.due_diligence_orchestrator.provenance import (
            ProvenanceTracker,
        )
        return ProvenanceTracker
    if name == "get_tracker":
        from greenlang.agents.eudr.due_diligence_orchestrator.provenance import (
            get_tracker,
        )
        return get_tracker

    # Integration
    if name == "AgentClient":
        from greenlang.agents.eudr.due_diligence_orchestrator.integration.agent_client import (
            AgentClient,
        )
        return AgentClient
    if name == "EventBus":
        from greenlang.agents.eudr.due_diligence_orchestrator.integration.event_bus import (
            EventBus,
        )
        return EventBus
    if name == "get_event_bus":
        from greenlang.agents.eudr.due_diligence_orchestrator.integration.event_bus import (
            get_event_bus,
        )
        return get_event_bus

    # All models are in models.py
    from greenlang.agents.eudr.due_diligence_orchestrator import models
    if hasattr(models, name):
        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.due_diligence_orchestrator import X``
    without eagerly loading all submodules at package import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The lazily imported object.
    """
    return _lazy_import(name)
