# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness - Engines Module
=============================================

Eight deterministic, zero-hallucination engines for EU Corporate
Sustainability Due Diligence Directive (CSDDD / Directive 2024/1760)
readiness assessment.

Engines:
    1. DueDiligencePolicyEngine      - Articles 5-11 policy assessment
    2. AdverseImpactEngine           - Adverse impact identification and scoring
    3. PreventionMitigationEngine    - Prevention and mitigation measure tracking
    4. RemediationTrackingEngine     - Remediation action monitoring
    5. GrievanceMechanismEngine      - Article 11 grievance mechanism assessment
    6. StakeholderEngagementEngine   - Stakeholder engagement quality scoring
    7. ClimateTransitionEngine       - Article 22 climate transition plan assessment
    8. CivilLiabilityEngine          - Civil liability exposure analysis

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-019"
__pack_name__: str = "CSDDD Readiness Pack"
__engines_count__: int = 8

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: Due Diligence Policy
# ---------------------------------------------------------------------------
_ENGINE_1_SYMBOLS: list[str] = [
    "DueDiligencePolicyEngine",
    "ArticleReference",
    "CompanyScope",
    "ComplianceStatus",
    "CompanyProfile",
    "PolicyArea",
    "PolicyAssessment",
    "ArticleAssessment",
    "ScopeAssessment",
    "DueDiligencePolicyResult",
]
try:
    from .due_diligence_policy_engine import (
        DueDiligencePolicyEngine,
        ArticleReference,
        CompanyScope,
        ComplianceStatus,
        CompanyProfile,
        PolicyArea,
        PolicyAssessment,
        ArticleAssessment,
        ScopeAssessment,
        DueDiligencePolicyResult,
    )
    _loaded_engines.append("DueDiligencePolicyEngine")
except ImportError as e:
    logger.debug("Engine 1 (DueDiligencePolicyEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 2: Adverse Impact
# ---------------------------------------------------------------------------
_ENGINE_2_SYMBOLS: list[str] = [
    "AdverseImpactEngine",
    "AdverseImpactType",
    "ImpactSeverity",
    "ImpactLikelihood",
    "ImpactStatus",
    "ValueChainPosition",
    "HumanRightsCategory",
    "EnvironmentalCategory",
    "RiskLevel",
    "AdverseImpact",
    "RiskMatrix",
    "SummaryStatistics",
    "ImpactAssessmentResult",
]
try:
    from .adverse_impact_engine import (
        AdverseImpactEngine,
        AdverseImpactType,
        ImpactSeverity,
        ImpactLikelihood,
        ImpactStatus,
        ValueChainPosition,
        HumanRightsCategory,
        EnvironmentalCategory,
        RiskLevel,
        AdverseImpact,
        RiskMatrix,
        SummaryStatistics,
        ImpactAssessmentResult,
    )
    _loaded_engines.append("AdverseImpactEngine")
except ImportError as e:
    logger.debug("Engine 2 (AdverseImpactEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 3: Prevention & Mitigation
# ---------------------------------------------------------------------------
_ENGINE_3_SYMBOLS: list[str] = [
    "PreventionMitigationEngine",
    "MeasureType",
    "MeasureStatus",
    "EffectivenessRating",
    "MeasureCategory",
    "PreventionMeasure",
    "MeasureEffectiveness",
    "BudgetSummary",
    "CoverageAnalysis",
    "GapAnalysis",
    "EffectivenessSummary",
    "PreventionResult",
]
try:
    from .prevention_mitigation_engine import (
        PreventionMitigationEngine,
        MeasureType,
        MeasureStatus,
        EffectivenessRating,
        MeasureCategory,
        PreventionMeasure,
        MeasureEffectiveness,
        BudgetSummary,
        CoverageAnalysis,
        GapAnalysis,
        EffectivenessSummary,
        PreventionResult,
    )
    _loaded_engines.append("PreventionMitigationEngine")
except ImportError as e:
    logger.debug("Engine 3 (PreventionMitigationEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 4: Remediation Tracking
# ---------------------------------------------------------------------------
_ENGINE_4_SYMBOLS: list[str] = [
    "RemediationTrackingEngine",
    "RemediationStatus",
    "RemediationType",
    "CompanyContribution",
    "VictimEngagementLevel",
    "RemediationAction",
    "TimelineAnalysis",
    "FinancialAnalysis",
    "VictimEngagementAnalysis",
    "CompletenessAssessment",
    "RemediationResult",
]
try:
    from .remediation_tracking_engine import (
        RemediationTrackingEngine,
        RemediationStatus,
        RemediationType,
        CompanyContribution,
        VictimEngagementLevel,
        RemediationAction,
        TimelineAnalysis,
        FinancialAnalysis,
        VictimEngagementAnalysis,
        CompletenessAssessment,
        RemediationResult,
    )
    _loaded_engines.append("RemediationTrackingEngine")
except ImportError as e:
    logger.debug("Engine 4 (RemediationTrackingEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 5: Grievance Mechanism
# ---------------------------------------------------------------------------
_ENGINE_5_SYMBOLS: list[str] = [
    "GrievanceMechanismEngine",
    "GrievanceCase",
    "GrievanceResult",
    "GrievanceStatus",
    "GrievanceChannel",
    "MechanismCriteria",
    "MechanismConfig",
    "MechanismAssessment",
    "GrievanceStakeholderGroup",
]
try:
    from .grievance_mechanism_engine import (
        GrievanceMechanismEngine,
        GrievanceCase,
        GrievanceResult,
        GrievanceStatus,
        GrievanceChannel,
        MechanismCriteria,
        MechanismConfig,
        MechanismAssessment,
        StakeholderGroup as GrievanceStakeholderGroup,
    )
    _loaded_engines.append("GrievanceMechanismEngine")
except ImportError as e:
    logger.debug("Engine 5 (GrievanceMechanismEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 6: Stakeholder Engagement
# ---------------------------------------------------------------------------
_ENGINE_6_SYMBOLS: list[str] = [
    "StakeholderEngagementEngine",
    "StakeholderEngagement",
    "EngagementResult",
    "EngagementMethod",
    "EngagementQuality",
    "DueDiligenceStage",
    "EngagementStakeholderGroup",
]
try:
    from .stakeholder_engagement_engine import (
        StakeholderEngagementEngine,
        StakeholderEngagement,
        EngagementResult,
        EngagementMethod,
        EngagementQuality,
        DueDiligenceStage,
        StakeholderGroup as EngagementStakeholderGroup,
    )
    _loaded_engines.append("StakeholderEngagementEngine")
except ImportError as e:
    logger.debug("Engine 6 (StakeholderEngagementEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 7: Climate Transition
# ---------------------------------------------------------------------------
_ENGINE_7_SYMBOLS: list[str] = [
    "ClimateTransitionEngine",
    "ClimateTarget",
    "ClimateTransitionResult",
    "TransitionPlanStatus",
    "EmissionScope",
    "AlignmentLevel",
    "TransitionElement",
    "TransitionPlanDetails",
    "InterimMilestone",
]
try:
    from .climate_transition_engine import (
        ClimateTransitionEngine,
        ClimateTarget,
        ClimateTransitionResult,
        TransitionPlanStatus,
        EmissionScope,
        AlignmentLevel,
        TransitionElement,
        TransitionPlanDetails,
        InterimMilestone,
    )
    _loaded_engines.append("ClimateTransitionEngine")
except ImportError as e:
    logger.debug("Engine 7 (ClimateTransitionEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 8: Civil Liability
# ---------------------------------------------------------------------------
_ENGINE_8_SYMBOLS: list[str] = [
    "CivilLiabilityEngine",
    "LiabilityScenario",
    "CivilLiabilityResult",
    "LiabilityTrigger",
    "DefencePosition",
    "ExposureLevel",
    "LiabilityImpactSeverity",
    "ImpactDomain",
]
try:
    from .civil_liability_engine import (
        CivilLiabilityEngine,
        LiabilityScenario,
        CivilLiabilityResult,
        LiabilityTrigger,
        DefencePosition,
        ExposureLevel,
        ImpactSeverity as LiabilityImpactSeverity,
        ImpactDomain,
    )
    _loaded_engines.append("CivilLiabilityEngine")
except ImportError as e:
    logger.debug("Engine 8 (CivilLiabilityEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    *_ENGINE_1_SYMBOLS,
    *_ENGINE_2_SYMBOLS,
    *_ENGINE_3_SYMBOLS,
    *_ENGINE_4_SYMBOLS,
    *_ENGINE_5_SYMBOLS,
    *_ENGINE_6_SYMBOLS,
    *_ENGINE_7_SYMBOLS,
    *_ENGINE_8_SYMBOLS,
    "get_loaded_engines",
    "get_engine_count",
]


def get_loaded_engines() -> list[str]:
    """Return list of successfully loaded engine class names."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-019 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
