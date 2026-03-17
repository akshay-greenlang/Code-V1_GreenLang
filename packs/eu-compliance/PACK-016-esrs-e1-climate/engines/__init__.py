# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Engines Module
=================================================

Deterministic, zero-hallucination calculation engines for ESRS E1
Climate Change disclosure requirements.  Each engine covers one or
more ESRS E1 disclosure requirements (E1-1 through E1-9) and produces
bit-perfect reproducible results with SHA-256 provenance hashing.
No LLM is used in any scoring, classification, or calculation path.

Engines:
    1. GHGInventoryEngine           - Gross GHG emissions Scopes 1, 2, 3 (ESRS E1-6)
    2. EnergyMixEngine              - Energy consumption and mix (ESRS E1-5)
    3. TransitionPlanEngine         - Transition plan for climate mitigation (ESRS E1-1)
    4. ClimateTargetEngine          - Targets related to climate change (ESRS E1-4)
    5. ClimateActionEngine          - Policies (E1-2) and actions/resources (E1-3)
    6. CarbonCreditEngine           - GHG removals and carbon credits (ESRS E1-7)
    7. CarbonPricingEngine          - Internal carbon pricing (ESRS E1-8)
    8. ClimateRiskEngine            - Financial effects from climate risks (ESRS E1-9)

Regulatory Basis:
    EU Directive 2022/2464 (CSRD)
    EU Delegated Regulation 2023/2772 (ESRS Set 1)
    ESRS E1 Climate Change (all disclosure requirements)
    GHG Protocol Corporate Standard (2004, revised 2015)
    GHG Protocol Scope 3 Standard (2011)
    IPCC AR6 WG1 (2021) - GWP-100 values
    EU Taxonomy Regulation 2020/852
    Paris Agreement (2015) temperature targets
    SBTi Corporate Net-Zero Standard (2024)
    TCFD Final Recommendations (2017)
    NGFS Climate Scenarios (2024)

Pack Tier: Standalone Topical Standard (PACK-016)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-016"
__pack_name__: str = "ESRS E1 Climate Pack"
__engines_count__: int = 8

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: GHG Inventory (ESRS E1-6)
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "GHGInventoryEngine",
    "EmissionEntry",
    "GHGInventoryResult",
    "EmissionsByGas",
    "Scope3Breakdown",
    "IntensityMetric",
    "BatchInventoryResult",
    "GHGScope",
    "EmissionGas",
    "Scope3Category",
    "ConsolidationApproach",
    "DataQualityLevel",
    "GWP_AR6",
    "E1_6_DATAPOINTS",
]

try:
    from .ghg_inventory_engine import (
        E1_6_DATAPOINTS,
        GWP_AR6,
        BatchInventoryResult,
        ConsolidationApproach,
        DataQualityLevel,
        EmissionEntry,
        EmissionGas,
        EmissionsByGas,
        GHGInventoryEngine,
        GHGInventoryResult,
        GHGScope,
        IntensityMetric,
        Scope3Breakdown,
        Scope3Category,
    )
    _loaded_engines.append("GHGInventoryEngine")
except ImportError as e:
    logger.debug("Engine 1 (GHGInventoryEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Energy Mix (ESRS E1-5)
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "EnergyMixEngine",
    "EnergyConsumptionEntry",
    "EnergyMixResult",
    "EnergyIntensity",
    "RenewableBreakdown",
    "EnergySource",
    "EnergyCategory",
    "EnergyUnit",
    "ENERGY_CONVERSION_FACTORS",
    "SOURCE_CLASSIFICATION",
    "E1_5_DATAPOINTS",
]

try:
    from .energy_mix_engine import (
        E1_5_DATAPOINTS,
        ENERGY_CONVERSION_FACTORS,
        SOURCE_CLASSIFICATION,
        EnergyCategory,
        EnergyConsumptionEntry,
        EnergyIntensity,
        EnergyMixEngine,
        EnergyMixResult,
        EnergySource,
        EnergyUnit,
        RenewableBreakdown,
    )
    _loaded_engines.append("EnergyMixEngine")
except ImportError as e:
    logger.debug("Engine 2 (EnergyMixEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Transition Plan (ESRS E1-1)
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "TransitionPlanEngine",
    "TransitionPlanAction",
    "LockedInEmission",
    "TransitionPlanResult",
    "PlanGapAnalysis",
    "DecarbonizationLever",
    "PlanStatus",
    "ScenarioAlignment",
    "LockedInEmissionType",
    "E1_1_DATAPOINTS",
    "LEVER_TYPICAL_ABATEMENT",
]

try:
    from .transition_plan_engine import (
        E1_1_DATAPOINTS,
        LEVER_TYPICAL_ABATEMENT,
        DecarbonizationLever,
        LockedInEmission,
        LockedInEmissionType,
        PlanGapAnalysis,
        PlanStatus,
        ScenarioAlignment,
        TransitionPlanAction,
        TransitionPlanEngine,
        TransitionPlanResult,
    )
    _loaded_engines.append("TransitionPlanEngine")
except ImportError as e:
    logger.debug("Engine 3 (TransitionPlanEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Climate Target (ESRS E1-4)
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "ClimateTargetEngine",
    "ClimateTarget",
    "TargetProgressResult",
    "BaseYearRecalculation",
    "BatchTargetResult",
    "TargetType",
    "TargetScope",
    "TargetPathway",
    "TargetStatus",
    "BaseYearApproach",
    "SBTI_MINIMUM_RATES",
    "E1_4_DATAPOINTS",
]

try:
    from .climate_target_engine import (
        E1_4_DATAPOINTS,
        SBTI_MINIMUM_RATES,
        BaseYearApproach,
        BaseYearRecalculation,
        BatchTargetResult,
        ClimateTarget,
        ClimateTargetEngine,
        TargetPathway,
        TargetProgressResult,
        TargetScope,
        TargetStatus,
        TargetType,
    )
    _loaded_engines.append("ClimateTargetEngine")
except ImportError as e:
    logger.debug("Engine 4 (ClimateTargetEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Climate Action (ESRS E1-2 + E1-3)
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "ClimateActionEngine",
    "ClimatePolicy",
    "ClimateAction",
    "ResourceAllocation",
    "ClimateActionResult",
    "PolicyType",
    "PolicyScope",
    "ActionCategory",
    "ActionStatus",
    "ResourceType",
    "E1_2_DATAPOINTS",
    "E1_3_DATAPOINTS",
    "ACTION_TAXONOMY_ALIGNMENT",
    "POLICY_TYPE_DESCRIPTIONS",
    "ACTION_CATEGORY_DESCRIPTIONS",
]

try:
    from .climate_action_engine import (
        ACTION_CATEGORY_DESCRIPTIONS,
        ACTION_TAXONOMY_ALIGNMENT,
        ActionCategory,
        ActionStatus,
        ClimateAction,
        ClimateActionEngine,
        ClimateActionResult,
        ClimatePolicy,
        E1_2_DATAPOINTS,
        E1_3_DATAPOINTS,
        POLICY_TYPE_DESCRIPTIONS,
        PolicyScope,
        PolicyType,
        ResourceAllocation,
        ResourceType,
    )
    _loaded_engines.append("ClimateActionEngine")
except ImportError as e:
    logger.debug("Engine 5 (ClimateActionEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Carbon Credit (ESRS E1-7)
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "CarbonCreditEngine",
    "CarbonCredit",
    "GHGRemoval",
    "CarbonCreditResult",
    "QualityAssessment",
    "CreditStandard",
    "CreditType",
    "ProjectType",
    "CreditStatus",
    "RemovalType",
    "VerificationStatus",
    "E1_7_DATAPOINTS",
    "QUALITY_CRITERIA",
    "SBTI_BEYONDVALUECHAINMITIGATION",
    "CREDIT_STANDARD_DESCRIPTIONS",
    "PROJECT_TYPE_DESCRIPTIONS",
]

try:
    from .carbon_credit_engine import (
        CREDIT_STANDARD_DESCRIPTIONS,
        CarbonCredit,
        CarbonCreditEngine,
        CarbonCreditResult,
        CreditStandard,
        CreditStatus,
        CreditType,
        E1_7_DATAPOINTS,
        GHGRemoval,
        PROJECT_TYPE_DESCRIPTIONS,
        ProjectType,
        QUALITY_CRITERIA,
        QualityAssessment,
        RemovalType,
        SBTI_BEYONDVALUECHAINMITIGATION,
        VerificationStatus,
    )
    _loaded_engines.append("CarbonCreditEngine")
except ImportError as e:
    logger.debug("Engine 6 (CarbonCreditEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Carbon Pricing (ESRS E1-8)
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "CarbonPricingEngine",
    "CarbonPrice",
    "ShadowPriceScenario",
    "CarbonPricingResult",
    "PricingMechanism",
    "PricingScope",
    "CurrencyCode",
    "E1_8_DATAPOINTS",
    "REFERENCE_CARBON_PRICES",
    "SHADOW_PRICE_BENCHMARKS",
    "MECHANISM_DESCRIPTIONS",
    "SCOPE_DESCRIPTIONS",
]

try:
    from .carbon_pricing_engine import (
        CarbonPrice,
        CarbonPricingEngine,
        CarbonPricingResult,
        CurrencyCode,
        E1_8_DATAPOINTS,
        MECHANISM_DESCRIPTIONS,
        PricingMechanism,
        PricingScope,
        REFERENCE_CARBON_PRICES,
        SCOPE_DESCRIPTIONS,
        SHADOW_PRICE_BENCHMARKS,
        ShadowPriceScenario,
    )
    _loaded_engines.append("CarbonPricingEngine")
except ImportError as e:
    logger.debug("Engine 7 (CarbonPricingEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Climate Risk (ESRS E1-9)
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "ClimateRiskEngine",
    "PhysicalRisk",
    "TransitionRisk",
    "ClimateOpportunity",
    "ClimateRiskResult",
    "PhysicalRiskType",
    "TransitionRiskType",
    "ClimateOpportunityType",
    "RiskTimeHorizon",
    "ClimateScenario",
    "LikelihoodLevel",
    "E1_9_DATAPOINTS",
    "PHYSICAL_RISK_DESCRIPTIONS",
    "TRANSITION_RISK_DESCRIPTIONS",
    "LIKELIHOOD_PROBABILITIES",
    "SCENARIO_DESCRIPTIONS",
    "DAMAGE_FUNCTION_PARAMS",
    "OPPORTUNITY_DESCRIPTIONS",
]

try:
    from .climate_risk_engine import (
        ClimateOpportunity,
        ClimateOpportunityType,
        ClimateRiskEngine,
        ClimateRiskResult,
        ClimateScenario,
        DAMAGE_FUNCTION_PARAMS,
        E1_9_DATAPOINTS,
        LIKELIHOOD_PROBABILITIES,
        LikelihoodLevel,
        OPPORTUNITY_DESCRIPTIONS,
        PHYSICAL_RISK_DESCRIPTIONS,
        PhysicalRisk,
        PhysicalRiskType,
        RiskTimeHorizon,
        SCENARIO_DESCRIPTIONS,
        TRANSITION_RISK_DESCRIPTIONS,
        TransitionRisk,
        TransitionRiskType,
    )
    _loaded_engines.append("ClimateRiskEngine")
except ImportError as e:
    logger.debug("Engine 8 (ClimateRiskEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Public API - dynamically collected from successfully loaded engines
# ===================================================================

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

_UTILITY_SYMBOLS: list[str] = [
    "get_loaded_engines",
    "get_engine_count",
    "get_all_datapoints",
    "get_engine_map",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    *_UTILITY_SYMBOLS,
    # Engine 1: GHG Inventory
    *_ENGINE_1_SYMBOLS,
    # Engine 2: Energy Mix
    *_ENGINE_2_SYMBOLS,
    # Engine 3: Transition Plan
    *_ENGINE_3_SYMBOLS,
    # Engine 4: Climate Target
    *_ENGINE_4_SYMBOLS,
    # Engine 5: Climate Action
    *_ENGINE_5_SYMBOLS,
    # Engine 6: Carbon Credit
    *_ENGINE_6_SYMBOLS,
    # Engine 7: Carbon Pricing
    *_ENGINE_7_SYMBOLS,
    # Engine 8: Climate Risk
    *_ENGINE_8_SYMBOLS,
]


# ===================================================================
# Utility Functions
# ===================================================================

def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


def get_all_datapoints() -> dict[str, list[str]]:
    """Return all ESRS E1 datapoints organised by disclosure requirement.

    Returns a dictionary keyed by disclosure requirement ID (e.g. ``"E1-1"``)
    whose values are the ordered list of XBRL-tagged datapoint identifiers
    required by that disclosure.  Only datapoints from successfully loaded
    engines are included.

    Returns:
        Mapping of disclosure requirement IDs to datapoint lists.
    """
    datapoints: dict[str, list[str]] = {}

    # E1-1: Transition Plan
    if "TransitionPlanEngine" in _loaded_engines:
        datapoints["E1-1"] = list(E1_1_DATAPOINTS)  # type: ignore[name-defined]

    # E1-2: Policies related to climate change mitigation and adaptation
    if "ClimateActionEngine" in _loaded_engines:
        datapoints["E1-2"] = list(E1_2_DATAPOINTS)  # type: ignore[name-defined]

    # E1-3: Actions and resources related to climate change
    if "ClimateActionEngine" in _loaded_engines:
        datapoints["E1-3"] = list(E1_3_DATAPOINTS)  # type: ignore[name-defined]

    # E1-4: Targets related to climate change
    if "ClimateTargetEngine" in _loaded_engines:
        datapoints["E1-4"] = list(E1_4_DATAPOINTS)  # type: ignore[name-defined]

    # E1-5: Energy consumption and mix
    if "EnergyMixEngine" in _loaded_engines:
        datapoints["E1-5"] = list(E1_5_DATAPOINTS)  # type: ignore[name-defined]

    # E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions
    if "GHGInventoryEngine" in _loaded_engines:
        datapoints["E1-6"] = list(E1_6_DATAPOINTS)  # type: ignore[name-defined]

    # E1-7: GHG removals and GHG mitigation projects financed through carbon credits
    if "CarbonCreditEngine" in _loaded_engines:
        datapoints["E1-7"] = list(E1_7_DATAPOINTS)  # type: ignore[name-defined]

    # E1-8: Internal carbon pricing
    if "CarbonPricingEngine" in _loaded_engines:
        datapoints["E1-8"] = list(E1_8_DATAPOINTS)  # type: ignore[name-defined]

    # E1-9: Anticipated financial effects from material physical and transition risks
    if "ClimateRiskEngine" in _loaded_engines:
        datapoints["E1-9"] = list(E1_9_DATAPOINTS)  # type: ignore[name-defined]

    return datapoints


def get_engine_map() -> dict[str, type]:
    """Return mapping of disclosure requirement IDs to engine classes.

    Returns a dictionary keyed by ESRS E1 disclosure requirement ID
    (e.g. ``"E1-6"``) whose values are the engine class responsible
    for computing that disclosure.  Only successfully loaded engines
    are included.

    Returns:
        Mapping of disclosure requirement IDs to engine classes.
    """
    engine_map: dict[str, type] = {}

    if "TransitionPlanEngine" in _loaded_engines:
        engine_map["E1-1"] = TransitionPlanEngine  # type: ignore[name-defined]

    if "ClimateActionEngine" in _loaded_engines:
        engine_map["E1-2"] = ClimateActionEngine  # type: ignore[name-defined]
        engine_map["E1-3"] = ClimateActionEngine  # type: ignore[name-defined]

    if "ClimateTargetEngine" in _loaded_engines:
        engine_map["E1-4"] = ClimateTargetEngine  # type: ignore[name-defined]

    if "EnergyMixEngine" in _loaded_engines:
        engine_map["E1-5"] = EnergyMixEngine  # type: ignore[name-defined]

    if "GHGInventoryEngine" in _loaded_engines:
        engine_map["E1-6"] = GHGInventoryEngine  # type: ignore[name-defined]

    if "CarbonCreditEngine" in _loaded_engines:
        engine_map["E1-7"] = CarbonCreditEngine  # type: ignore[name-defined]

    if "CarbonPricingEngine" in _loaded_engines:
        engine_map["E1-8"] = CarbonPricingEngine  # type: ignore[name-defined]

    if "ClimateRiskEngine" in _loaded_engines:
        engine_map["E1-9"] = ClimateRiskEngine  # type: ignore[name-defined]

    return engine_map


logger.info(
    "PACK-016 ESRS E1 Climate engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
