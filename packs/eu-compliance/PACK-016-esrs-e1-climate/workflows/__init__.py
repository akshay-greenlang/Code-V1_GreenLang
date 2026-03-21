# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Workflow Orchestration
=============================================================

Climate disclosure workflow orchestrators for ESRS E1 compliance operations.
Each workflow coordinates GreenLang agents, data pipelines, calculation engines,
and validation systems into end-to-end climate disclosure processes covering
GHG inventory, energy assessment, transition planning, target setting, climate
actions, carbon credits, carbon pricing, climate risk, and full E1 disclosure.

Workflows:
    - GHGInventoryWorkflow: 5-phase end-to-end GHG inventory with data
      collection, emission calculation, scope aggregation, quality check,
      and report generation per GHG Protocol / ESRS E1-6.

    - EnergyAssessmentWorkflow: 5-phase energy consumption and mix assessment
      with data collection, unit normalization, source classification, mix
      calculation, and reporting per ESRS E1-5.

    - TransitionPlanWorkflow: 6-phase transition plan development with
      baseline assessment, lever identification, action planning, gap
      analysis, scenario validation, and report generation per ESRS E1-1.

    - TargetSettingWorkflow: 5-phase climate target setting and validation
      with baseline determination, target definition, SBTi validation,
      progress assessment, and reporting per ESRS E1-4.

    - ClimateActionsWorkflow: 5-phase actions and resources tracking with
      policy review, action registration, resource allocation, taxonomy
      check, and reporting per ESRS E1-3.

    - CarbonCreditsWorkflow: 5-phase carbon credit/offset management with
      credit registration, quality assessment, portfolio analysis, SBTi
      check, and reporting per ESRS E1-7.

    - CarbonPricingWorkflow: 4-phase carbon pricing disclosure with
      mechanism setup, coverage calculation, scenario analysis, and
      reporting per ESRS E1-8.

    - ClimateRiskWorkflow: 6-phase physical and transition risk assessment
      with risk identification, quantification, opportunity assessment,
      scenario analysis, financial aggregation, and reporting per ESRS E1-9.

    - FullE1Workflow: 10-phase end-to-end ESRS E1 disclosure generation
      orchestrating all sub-workflows into a complete climate disclosure
      with aggregated results and completeness tracking.

Author: GreenLang Team
Version: 16.0.0
"""

import logging
from typing import Dict, List, Type

logger = logging.getLogger(__name__)

_loaded_workflows: Dict[str, bool] = {}

# ---------------------------------------------------------------------------
# GHG Inventory Workflow
# ---------------------------------------------------------------------------
try:
    from .ghg_inventory_workflow import (
        GHGInventoryWorkflow,
        GHGInventoryInput,
        GHGInventoryResult,
        EmissionRecord,
        ScopeAggregation,
    )
    _loaded_workflows["ghg_inventory"] = True
except ImportError as e:
    logger.debug("GHG Inventory Workflow not loaded: %s", e)
    _loaded_workflows["ghg_inventory"] = False

# ---------------------------------------------------------------------------
# Energy Assessment Workflow
# ---------------------------------------------------------------------------
try:
    from .energy_assessment_workflow import (
        EnergyAssessmentWorkflow,
        EnergyAssessmentInput,
        EnergyAssessmentResult,
        EnergySource,
        EnergyMixResult,
    )
    _loaded_workflows["energy_assessment"] = True
except ImportError as e:
    logger.debug("Energy Assessment Workflow not loaded: %s", e)
    _loaded_workflows["energy_assessment"] = False

# ---------------------------------------------------------------------------
# Transition Plan Workflow
# ---------------------------------------------------------------------------
try:
    from .transition_plan_workflow import (
        TransitionPlanWorkflow,
        TransitionPlanInput,
        TransitionPlanResult,
        DecarbonizationLever,
        TransitionAction,
        GapAnalysisItem,
    )
    _loaded_workflows["transition_plan"] = True
except ImportError as e:
    logger.debug("Transition Plan Workflow not loaded: %s", e)
    _loaded_workflows["transition_plan"] = False

# ---------------------------------------------------------------------------
# Target Setting Workflow
# ---------------------------------------------------------------------------
try:
    from .target_setting_workflow import (
        TargetSettingWorkflow,
        TargetSettingInput,
        TargetSettingResult,
        ClimateTarget,
        SBTiValidationResult,
    )
    _loaded_workflows["target_setting"] = True
except ImportError as e:
    logger.debug("Target Setting Workflow not loaded: %s", e)
    _loaded_workflows["target_setting"] = False

# ---------------------------------------------------------------------------
# Climate Actions Workflow
# ---------------------------------------------------------------------------
try:
    from .climate_actions_workflow import (
        ClimateActionsWorkflow,
        ClimateActionsInput,
        ClimateActionsResult,
        ClimateAction,
        ResourceAllocation,
    )
    _loaded_workflows["climate_actions"] = True
except ImportError as e:
    logger.debug("Climate Actions Workflow not loaded: %s", e)
    _loaded_workflows["climate_actions"] = False

# ---------------------------------------------------------------------------
# Carbon Credits Workflow
# ---------------------------------------------------------------------------
try:
    from .carbon_credits_workflow import (
        CarbonCreditsWorkflow,
        CarbonCreditsInput,
        CarbonCreditsResult,
        CarbonCredit,
        CreditQualityScore,
    )
    _loaded_workflows["carbon_credits"] = True
except ImportError as e:
    logger.debug("Carbon Credits Workflow not loaded: %s", e)
    _loaded_workflows["carbon_credits"] = False

# ---------------------------------------------------------------------------
# Carbon Pricing Workflow
# ---------------------------------------------------------------------------
try:
    from .carbon_pricing_workflow import (
        CarbonPricingWorkflow,
        CarbonPricingInput,
        CarbonPricingResult,
        PricingMechanism,
        CoverageResult,
    )
    _loaded_workflows["carbon_pricing"] = True
except ImportError as e:
    logger.debug("Carbon Pricing Workflow not loaded: %s", e)
    _loaded_workflows["carbon_pricing"] = False

# ---------------------------------------------------------------------------
# Climate Risk Workflow
# ---------------------------------------------------------------------------
try:
    from .climate_risk_workflow import (
        ClimateRiskWorkflow,
        ClimateRiskInput,
        ClimateRiskResult,
        ClimateRisk,
        ClimateOpportunity,
        FinancialEffect,
    )
    _loaded_workflows["climate_risk"] = True
except ImportError as e:
    logger.debug("Climate Risk Workflow not loaded: %s", e)
    _loaded_workflows["climate_risk"] = False

# ---------------------------------------------------------------------------
# Full E1 Workflow
# ---------------------------------------------------------------------------
try:
    from .full_e1_workflow import (
        FullE1Workflow,
        FullE1Input,
        FullE1Result,
        DisclosureItem,
        E1DisclosureStatus,
    )
    _loaded_workflows["full_e1"] = True
except ImportError as e:
    logger.debug("Full E1 Workflow not loaded: %s", e)
    _loaded_workflows["full_e1"] = False


__all__ = [
    # --- GHG Inventory Workflow ---
    "GHGInventoryWorkflow",
    "GHGInventoryInput",
    "GHGInventoryResult",
    "EmissionRecord",
    "ScopeAggregation",
    # --- Energy Assessment Workflow ---
    "EnergyAssessmentWorkflow",
    "EnergyAssessmentInput",
    "EnergyAssessmentResult",
    "EnergySource",
    "EnergyMixResult",
    # --- Transition Plan Workflow ---
    "TransitionPlanWorkflow",
    "TransitionPlanInput",
    "TransitionPlanResult",
    "DecarbonizationLever",
    "TransitionAction",
    "GapAnalysisItem",
    # --- Target Setting Workflow ---
    "TargetSettingWorkflow",
    "TargetSettingInput",
    "TargetSettingResult",
    "ClimateTarget",
    "SBTiValidationResult",
    # --- Climate Actions Workflow ---
    "ClimateActionsWorkflow",
    "ClimateActionsInput",
    "ClimateActionsResult",
    "ClimateAction",
    "ResourceAllocation",
    # --- Carbon Credits Workflow ---
    "CarbonCreditsWorkflow",
    "CarbonCreditsInput",
    "CarbonCreditsResult",
    "CarbonCredit",
    "CreditQualityScore",
    # --- Carbon Pricing Workflow ---
    "CarbonPricingWorkflow",
    "CarbonPricingInput",
    "CarbonPricingResult",
    "PricingMechanism",
    "CoverageResult",
    # --- Climate Risk Workflow ---
    "ClimateRiskWorkflow",
    "ClimateRiskInput",
    "ClimateRiskResult",
    "ClimateRisk",
    "ClimateOpportunity",
    "FinancialEffect",
    # --- Full E1 Workflow ---
    "FullE1Workflow",
    "FullE1Input",
    "FullE1Result",
    "DisclosureItem",
    "E1DisclosureStatus",
    # --- Registry ---
    "_loaded_workflows",
]


def get_loaded_workflows() -> Dict[str, bool]:
    """Return the loading status of all workflows."""
    return dict(_loaded_workflows)


def get_available_workflow_names() -> List[str]:
    """Return names of successfully loaded workflows."""
    return [name for name, loaded in _loaded_workflows.items() if loaded]
