# -*- coding: utf-8 -*-
"""
GreenLang Procurement Layer Agents
==================================

The Procurement Layer provides agents for sustainable procurement,
supplier sustainability scoring, and supply chain carbon footprint calculation.

Agents:
    GL-PROC-X-001: Supplier Sustainability Scorer - Scores supplier sustainability
    GL-PROC-X-002: Sustainable Sourcing Agent - Identifies sustainable sourcing options
    GL-PROC-X-003: Supplier Engagement Agent - Manages supplier engagement
    GL-PROC-X-004: Procurement Carbon Footprint - Calculates procurement emissions
"""

from greenlang.agents.procurement.supplier_sustainability_scorer import (
    SupplierSustainabilityScorerAgent,
    SupplierScorerInput,
    SupplierScorerOutput,
    SupplierProfile,
    SustainabilityScore,
    SupplierAssessment,
    ScoreCategory,
    DataQuality,
)

from greenlang.agents.procurement.sustainable_sourcing_agent import (
    SustainableSourcingAgent,
    SourcingInput,
    SourcingOutput,
    SourcingCriteria,
    MaterialSpecification,
    SourcingOption,
    SourcingRecommendation,
    MaterialCategory,
    CertificationStandard,
)

from greenlang.agents.procurement.supplier_engagement_agent import (
    SupplierEngagementAgent,
    EngagementInput,
    EngagementOutput,
    EngagementProgram,
    SupplierAction,
    EngagementStatus,
    EngagementPriority,
    ActionType,
)

from greenlang.agents.procurement.procurement_carbon_footprint import (
    ProcurementCarbonFootprintAgent,
    ProcurementFootprintInput,
    ProcurementFootprintOutput,
    ProcurementItem,
    EmissionCalculation,
    ProcurementSummary,
    CalculationMethod,
    SpendCategory,
)

__all__ = [
    # Supplier Sustainability Scorer (GL-PROC-X-001)
    "SupplierSustainabilityScorerAgent",
    "SupplierScorerInput",
    "SupplierScorerOutput",
    "SupplierProfile",
    "SustainabilityScore",
    "SupplierAssessment",
    "ScoreCategory",
    "DataQuality",
    # Sustainable Sourcing Agent (GL-PROC-X-002)
    "SustainableSourcingAgent",
    "SourcingInput",
    "SourcingOutput",
    "SourcingCriteria",
    "MaterialSpecification",
    "SourcingOption",
    "SourcingRecommendation",
    "MaterialCategory",
    "CertificationStandard",
    # Supplier Engagement Agent (GL-PROC-X-003)
    "SupplierEngagementAgent",
    "EngagementInput",
    "EngagementOutput",
    "EngagementProgram",
    "SupplierAction",
    "EngagementStatus",
    "EngagementPriority",
    "ActionType",
    # Procurement Carbon Footprint (GL-PROC-X-004)
    "ProcurementCarbonFootprintAgent",
    "ProcurementFootprintInput",
    "ProcurementFootprintOutput",
    "ProcurementItem",
    "EmissionCalculation",
    "ProcurementSummary",
    "CalculationMethod",
    "SpendCategory",
]
