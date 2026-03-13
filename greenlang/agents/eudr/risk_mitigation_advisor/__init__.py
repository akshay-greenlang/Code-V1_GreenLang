# -*- coding: utf-8 -*-
"""
Risk Mitigation Advisor Agent - AGENT-EUDR-025

Production-grade ML-powered risk mitigation strategy recommendation,
remediation plan management, supplier capacity building, effectiveness
tracking, adaptive management, cost-benefit optimization, stakeholder
collaboration, and audit-ready documentation for EUDR compliance. This
is the 25th agent in the EUDR agent family and establishes the Risk
Mitigation and Remediation Intelligence sub-category.

The agent is the operational bridge between the risk assessment layer
(EUDR-016 through EUDR-024) and the compliance reporting layer
(GL-EUDR-APP DDS generation). It consumes risk signals from all nine
upstream risk assessment agents, applies ML-powered recommendation
algorithms and deterministic rule engines to select appropriate
mitigation strategies, designs structured remediation plans, tracks
implementation and effectiveness, and generates the mitigation evidence
required for Due Diligence Statements.

Core capabilities:
    1. Strategy Selection Engine -- ML-powered (XGBoost/LightGBM with
       SHAP explainability) recommendation engine consuming risk inputs
       from EUDR-016 through EUDR-024, with deterministic fallback mode
       for zero-hallucination operation
    2. Remediation Plan Design Engine -- Structured multi-phase plans
       with SMART milestones, Gantt generation, ISO 31000 alignment
    3. Capacity Building Manager Engine -- 4-tier supplier development
       programs with 7 commodity-specific curricula (154 total modules)
    4. Measure Library Engine -- 500+ proven mitigation measures with
       PostgreSQL full-text search across 8 risk categories
    5. Effectiveness Tracking Engine -- Before/after risk scoring,
       ROI analysis, paired t-test statistical significance testing
    6. Continuous Monitoring Engine -- Event-driven adaptive management
       consuming real-time signals from 9 upstream agents
    7. Cost-Benefit Optimizer Engine -- Linear programming
       (scipy.optimize.linprog) for budget allocation optimization
    8. Stakeholder Collaboration Engine -- Multi-party coordination
       with supplier portal and role-based access

Foundational modules:
    - config: RiskMitigationAdvisorConfig with GL_EUDR_RMA_ env var
      support (50+ settings covering ML, strategy, plan, capacity,
      library, effectiveness, monitoring, optimization, collaboration,
      reporting, and rate limiting)
    - models: Pydantic v2 data models with 14 enumerations,
      16 core models, 9 request models, and 9 response models
    - provenance: SHA-256 chain-hashed audit trail tracking with
      14 entity types and 14 actions for full traceability
    - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_rma_)
      covering strategy recommendation, plan management, capacity
      building, measure library, effectiveness, monitoring,
      optimization, and collaboration

PRD: PRD-AGENT-EUDR-025
Agent ID: GL-EUDR-RMA-025
Regulation: EU 2023/1115 (EUDR) Articles 8, 10, 11, 29, 31;
           ISO 31000:2018 Risk Management; ISO 14001:2015
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.risk_mitigation_advisor import (
    ...     RiskCategory,
    ...     ISO31000TreatmentType,
    ...     PlanStatus,
    ...     RiskMitigationAdvisorConfig,
    ...     get_config,
    ... )
    >>> from decimal import Decimal
    >>> cfg = get_config()
    >>> print(cfg.ml_confidence_threshold, cfg.top_k_strategies)
    0.7 5

    >>> from greenlang.agents.eudr.risk_mitigation_advisor import (
    ...     RiskInput,
    ...     MitigationStrategy,
    ...     RemediationPlan,
    ... )
    >>> risk = RiskInput(
    ...     operator_id="op-001",
    ...     supplier_id="sup-001",
    ...     country_code="ID",
    ...     commodity="palm_oil",
    ...     country_risk_score=Decimal("75"),
    ...     supplier_risk_score=Decimal("60"),
    ... )
    >>> print(risk.commodity, risk.country_risk_score)
    palm_oil 75

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-RMA-025"

# ---------------------------------------------------------------------------
# Foundational imports: config
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.risk_mitigation_advisor.config import (
        RiskMitigationAdvisorConfig,
        get_config,
        reset_config,
        set_config,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import config module: {e}")
    RiskMitigationAdvisorConfig = None  # type: ignore[misc,assignment]
    get_config = None  # type: ignore[misc,assignment]
    reset_config = None  # type: ignore[misc,assignment]
    set_config = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: models
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.risk_mitigation_advisor.models import (
        # Enumerations (14)
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
        # Core Models (16)
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
        # Request Models (9)
        RecommendStrategiesRequest,
        CreatePlanRequest,
        EnrollSupplierRequest,
        SearchMeasuresRequest,
        MeasureEffectivenessRequest,
        OptimizeBudgetRequest,
        CollaborateRequest,
        GenerateReportRequest,
        AdaptiveScanRequest,
        # Response Models (9)
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
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        UPSTREAM_AGENT_COUNT,
        RISK_CATEGORY_COUNT,
        MIN_LIBRARY_MEASURES,
        SUPPORTED_COMMODITIES,
        ISO_31000_TYPES,
        CAPACITY_TIER_COUNT,
        MODULES_PER_COMMODITY,
        DEFAULT_TOP_K,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import models module: {e}")
    # Enumerations (14)
    RiskCategory = None  # type: ignore[misc,assignment]
    ISO31000TreatmentType = None  # type: ignore[misc,assignment]
    ImplementationComplexity = None  # type: ignore[misc,assignment]
    PlanStatus = None  # type: ignore[misc,assignment]
    PlanPhaseType = None  # type: ignore[misc,assignment]
    MilestoneStatus = None  # type: ignore[misc,assignment]
    CapacityTier = None  # type: ignore[misc,assignment]
    EnrollmentStatus = None  # type: ignore[misc,assignment]
    TriggerEventType = None  # type: ignore[misc,assignment]
    AdjustmentType = None  # type: ignore[misc,assignment]
    StakeholderRole = None  # type: ignore[misc,assignment]
    ReportType = None  # type: ignore[misc,assignment]
    EUDRCommodity = None  # type: ignore[misc,assignment]
    EvidenceQuality = None  # type: ignore[misc,assignment]
    # Core Models (16)
    RiskInput = None  # type: ignore[misc,assignment]
    MitigationStrategy = None  # type: ignore[misc,assignment]
    CostEstimate = None  # type: ignore[misc,assignment]
    CostRange = None  # type: ignore[misc,assignment]
    RemediationPlan = None  # type: ignore[misc,assignment]
    PlanPhase = None  # type: ignore[misc,assignment]
    Milestone = None  # type: ignore[misc,assignment]
    EvidenceDocument = None  # type: ignore[misc,assignment]
    ResponsibleParty = None  # type: ignore[misc,assignment]
    EscalationTrigger = None  # type: ignore[misc,assignment]
    KPI = None  # type: ignore[misc,assignment]
    MitigationMeasure = None  # type: ignore[misc,assignment]
    MeasureApplicability = None  # type: ignore[misc,assignment]
    EffectivenessRecord = None  # type: ignore[misc,assignment]
    CapacityBuildingEnrollment = None  # type: ignore[misc,assignment]
    TriggerEvent = None  # type: ignore[misc,assignment]
    # Request Models (9)
    RecommendStrategiesRequest = None  # type: ignore[misc,assignment]
    CreatePlanRequest = None  # type: ignore[misc,assignment]
    EnrollSupplierRequest = None  # type: ignore[misc,assignment]
    SearchMeasuresRequest = None  # type: ignore[misc,assignment]
    MeasureEffectivenessRequest = None  # type: ignore[misc,assignment]
    OptimizeBudgetRequest = None  # type: ignore[misc,assignment]
    CollaborateRequest = None  # type: ignore[misc,assignment]
    GenerateReportRequest = None  # type: ignore[misc,assignment]
    AdaptiveScanRequest = None  # type: ignore[misc,assignment]
    # Response Models (9)
    RecommendStrategiesResponse = None  # type: ignore[misc,assignment]
    CreatePlanResponse = None  # type: ignore[misc,assignment]
    EnrollSupplierResponse = None  # type: ignore[misc,assignment]
    SearchMeasuresResponse = None  # type: ignore[misc,assignment]
    MeasureEffectivenessResponse = None  # type: ignore[misc,assignment]
    OptimizeBudgetResponse = None  # type: ignore[misc,assignment]
    CollaborateResponse = None  # type: ignore[misc,assignment]
    GenerateReportResponse = None  # type: ignore[misc,assignment]
    AdaptiveScanResponse = None  # type: ignore[misc,assignment]
    # Constants
    VERSION = None  # type: ignore[misc,assignment]
    EUDR_CUTOFF_DATE = None  # type: ignore[misc,assignment]
    MAX_BATCH_SIZE = None  # type: ignore[misc,assignment]
    EUDR_RETENTION_YEARS = None  # type: ignore[misc,assignment]
    UPSTREAM_AGENT_COUNT = None  # type: ignore[misc,assignment]
    RISK_CATEGORY_COUNT = None  # type: ignore[misc,assignment]
    MIN_LIBRARY_MEASURES = None  # type: ignore[misc,assignment]
    SUPPORTED_COMMODITIES = None  # type: ignore[misc,assignment]
    ISO_31000_TYPES = None  # type: ignore[misc,assignment]
    CAPACITY_TIER_COUNT = None  # type: ignore[misc,assignment]
    MODULES_PER_COMMODITY = None  # type: ignore[misc,assignment]
    DEFAULT_TOP_K = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_tracker,
        reset_tracker,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import provenance module: {e}")
    ProvenanceRecord = None  # type: ignore[misc,assignment]
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_tracker = None  # type: ignore[misc,assignment]
    reset_tracker = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.risk_mitigation_advisor.metrics import (
        PROMETHEUS_AVAILABLE,
        # Counter helpers (8)
        record_strategy_recommended,
        record_plan_created,
        record_milestone_completed,
        record_capacity_enrollment,
        record_measure_searched,
        record_effectiveness_measured,
        record_trigger_event,
        record_api_error,
        # Histogram helpers (4)
        observe_strategy_latency,
        observe_plan_generation_duration,
        observe_optimization_duration,
        observe_effectiveness_calc_duration,
        # Gauge helpers (6)
        set_active_plans,
        set_active_enrollments,
        set_library_measures,
        set_pending_adjustments,
        set_total_risk_reduction,
        set_optimization_backlog,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import metrics module: {e}")
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    record_strategy_recommended = None  # type: ignore[misc,assignment]
    record_plan_created = None  # type: ignore[misc,assignment]
    record_milestone_completed = None  # type: ignore[misc,assignment]
    record_capacity_enrollment = None  # type: ignore[misc,assignment]
    record_measure_searched = None  # type: ignore[misc,assignment]
    record_effectiveness_measured = None  # type: ignore[misc,assignment]
    record_trigger_event = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]
    observe_strategy_latency = None  # type: ignore[misc,assignment]
    observe_plan_generation_duration = None  # type: ignore[misc,assignment]
    observe_optimization_duration = None  # type: ignore[misc,assignment]
    observe_effectiveness_calc_duration = None  # type: ignore[misc,assignment]
    set_active_plans = None  # type: ignore[misc,assignment]
    set_active_enrollments = None  # type: ignore[misc,assignment]
    set_library_measures = None  # type: ignore[misc,assignment]
    set_pending_adjustments = None  # type: ignore[misc,assignment]
    set_total_risk_reduction = None  # type: ignore[misc,assignment]
    set_optimization_backlog = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional)
# ---------------------------------------------------------------------------

# ---- Engine 1: Strategy Selection ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.strategy_selection_engine import (
        StrategySelectionEngine,
    )
except ImportError:
    StrategySelectionEngine = None  # type: ignore[misc,assignment]

# ---- Engine 2: Remediation Plan Design ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.remediation_plan_design_engine import (
        RemediationPlanDesignEngine,
    )
except ImportError:
    RemediationPlanDesignEngine = None  # type: ignore[misc,assignment]

# ---- Engine 3: Capacity Building Manager ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.capacity_building_manager_engine import (
        CapacityBuildingManagerEngine,
    )
except ImportError:
    CapacityBuildingManagerEngine = None  # type: ignore[misc,assignment]

# ---- Engine 4: Measure Library ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.measure_library_engine import (
        MeasureLibraryEngine,
    )
except ImportError:
    MeasureLibraryEngine = None  # type: ignore[misc,assignment]

# ---- Engine 5: Effectiveness Tracking ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.effectiveness_tracking_engine import (
        EffectivenessTrackingEngine,
    )
except ImportError:
    EffectivenessTrackingEngine = None  # type: ignore[misc,assignment]

# ---- Engine 6: Continuous Monitoring ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.continuous_monitoring_engine import (
        ContinuousMonitoringEngine,
    )
except ImportError:
    ContinuousMonitoringEngine = None  # type: ignore[misc,assignment]

# ---- Engine 7: Cost-Benefit Optimizer ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.cost_benefit_optimizer_engine import (
        CostBenefitOptimizerEngine,
    )
except ImportError:
    CostBenefitOptimizerEngine = None  # type: ignore[misc,assignment]

# ---- Engine 8: Stakeholder Collaboration ----
try:
    from greenlang.agents.eudr.risk_mitigation_advisor.stakeholder_collaboration_engine import (
        StakeholderCollaborationEngine,
    )
except ImportError:
    StakeholderCollaborationEngine = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Service facade import
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.risk_mitigation_advisor.setup import (
        RiskMitigationAdvisorSetup,
        get_service,
        reset_service,
        lifespan,
    )
except ImportError:
    RiskMitigationAdvisorSetup = None  # type: ignore[misc,assignment]
    get_service = None  # type: ignore[misc,assignment]
    reset_service = None  # type: ignore[misc,assignment]
    lifespan = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Config --
    "RiskMitigationAdvisorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Enumerations (14) --
    "RiskCategory",
    "ISO31000TreatmentType",
    "ImplementationComplexity",
    "PlanStatus",
    "PlanPhaseType",
    "MilestoneStatus",
    "CapacityTier",
    "EnrollmentStatus",
    "TriggerEventType",
    "AdjustmentType",
    "StakeholderRole",
    "ReportType",
    "EUDRCommodity",
    "EvidenceQuality",
    # -- Core Models (16) --
    "RiskInput",
    "MitigationStrategy",
    "CostEstimate",
    "CostRange",
    "RemediationPlan",
    "PlanPhase",
    "Milestone",
    "EvidenceDocument",
    "ResponsibleParty",
    "EscalationTrigger",
    "KPI",
    "MitigationMeasure",
    "MeasureApplicability",
    "EffectivenessRecord",
    "CapacityBuildingEnrollment",
    "TriggerEvent",
    # -- Request Models (9) --
    "RecommendStrategiesRequest",
    "CreatePlanRequest",
    "EnrollSupplierRequest",
    "SearchMeasuresRequest",
    "MeasureEffectivenessRequest",
    "OptimizeBudgetRequest",
    "CollaborateRequest",
    "GenerateReportRequest",
    "AdaptiveScanRequest",
    # -- Response Models (9) --
    "RecommendStrategiesResponse",
    "CreatePlanResponse",
    "EnrollSupplierResponse",
    "SearchMeasuresResponse",
    "MeasureEffectivenessResponse",
    "OptimizeBudgetResponse",
    "CollaborateResponse",
    "GenerateReportResponse",
    "AdaptiveScanResponse",
    # -- Constants --
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "UPSTREAM_AGENT_COUNT",
    "RISK_CATEGORY_COUNT",
    "MIN_LIBRARY_MEASURES",
    "SUPPORTED_COMMODITIES",
    "ISO_31000_TYPES",
    "CAPACITY_TIER_COUNT",
    "MODULES_PER_COMMODITY",
    "DEFAULT_TOP_K",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_tracker",
    "reset_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_strategy_recommended",
    "record_plan_created",
    "record_milestone_completed",
    "record_capacity_enrollment",
    "record_measure_searched",
    "record_effectiveness_measured",
    "record_trigger_event",
    "record_api_error",
    "observe_strategy_latency",
    "observe_plan_generation_duration",
    "observe_optimization_duration",
    "observe_effectiveness_calc_duration",
    "set_active_plans",
    "set_active_enrollments",
    "set_library_measures",
    "set_pending_adjustments",
    "set_total_risk_reduction",
    "set_optimization_backlog",
    # -- Engines (8) --
    "StrategySelectionEngine",
    "RemediationPlanDesignEngine",
    "CapacityBuildingManagerEngine",
    "MeasureLibraryEngine",
    "EffectivenessTrackingEngine",
    "ContinuousMonitoringEngine",
    "CostBenefitOptimizerEngine",
    "StakeholderCollaborationEngine",
    # -- Setup Facade --
    "RiskMitigationAdvisorSetup",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string.

    Returns:
        Version string in semver format (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata.

    Returns:
        Dictionary with agent_id, version, regulation references,
        engine listing, risk categories, and model counts for
        the Risk Mitigation Advisor agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-RMA-025'
        >>> info["engine_count"]
        8
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Risk Mitigation Advisor",
        "prd": "PRD-AGENT-EUDR-025",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["8", "10", "11", "29", "31"],
        "iso_standards": ["ISO 31000:2018", "ISO 14001:2015"],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "risk_categories": [
            "country", "supplier", "commodity", "corruption",
            "deforestation", "indigenous_rights", "protected_areas",
            "legal_compliance",
        ],
        "upstream_agents": [
            "EUDR-016 Country Risk Evaluator",
            "EUDR-017 Supplier Risk Scorer",
            "EUDR-018 Commodity Risk Analyzer",
            "EUDR-019 Corruption Index Monitor",
            "EUDR-020 Deforestation Alert System",
            "EUDR-021 Indigenous Rights Checker",
            "EUDR-022 Protected Area Validator",
            "EUDR-023 Legal Compliance Verifier",
            "EUDR-024 Third-Party Audit Manager",
        ],
        "eudr_commodities": [
            "cattle", "cocoa", "coffee", "palm_oil",
            "rubber", "soya", "wood",
        ],
        "iso_31000_types": ["avoid", "reduce", "share", "retain"],
        "capacity_tiers": 4,
        "modules_per_commodity": 22,
        "min_library_measures": 500,
        "engines": [
            "StrategySelectionEngine",
            "RemediationPlanDesignEngine",
            "CapacityBuildingManagerEngine",
            "MeasureLibraryEngine",
            "EffectivenessTrackingEngine",
            "ContinuousMonitoringEngine",
            "CostBenefitOptimizerEngine",
            "StakeholderCollaborationEngine",
        ],
        "engine_count": 8,
        "enum_count": 14,
        "core_model_count": 16,
        "request_model_count": 9,
        "response_model_count": 9,
        "metrics_count": 18,
        "db_prefix": "gl_eudr_rma_",
        "metrics_prefix": "gl_eudr_rma_",
        "env_prefix": "GL_EUDR_RMA_",
    }
