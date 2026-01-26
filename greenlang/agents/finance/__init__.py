# -*- coding: utf-8 -*-
"""
GreenLang Finance Layer Agents
==============================

The Finance Layer provides agents for carbon pricing, TCO calculations,
investment screening, carbon credit valuation, and climate finance tracking.

Agents:
    GL-FIN-X-001: Carbon Pricing Agent - Internal carbon pricing models
    GL-FIN-X-002: TCO Calculator Agent - Total cost of ownership with carbon
    GL-FIN-X-003: Green Investment Screener - Screens investments for sustainability
    GL-FIN-X-004: Carbon Credit Valuation - Values carbon credits/offsets
    GL-FIN-X-005: Climate Finance Tracker - Tracks climate-related investments
    GL-FIN-X-006: EU Taxonomy Alignment Agent - Aligns activities with EU Taxonomy
    GL-FIN-X-007: Green Bond Analyzer - Analyzes green bond frameworks
    GL-FIN-X-008: Climate Budget Agent - Manages carbon budgets
    GL-FIN-X-009: Stranded Asset Analyzer - Identifies stranded asset risks
"""

from greenlang.agents.finance.carbon_pricing_agent import (
    CarbonPricingAgent,
    CarbonPricingInput,
    CarbonPricingOutput,
    CarbonPriceScenario,
    PricingMechanism,
    InternalCarbonPrice,
    CarbonPriceImpact,
)

from greenlang.agents.finance.tco_calculator_agent import (
    TCOCalculatorAgent,
    TCOCalculatorInput,
    TCOCalculatorOutput,
    CostComponent,
    CarbonCostMethod,
    TCOResult,
    AssetComparison,
)

from greenlang.agents.finance.green_investment_screener import (
    GreenInvestmentScreenerAgent,
    InvestmentScreenerInput,
    InvestmentScreenerOutput,
    InvestmentCriteria,
    ESGRating,
    ScreeningResult,
    SustainabilityScore,
)

from greenlang.agents.finance.carbon_credit_valuation import (
    CarbonCreditValuationAgent,
    CreditValuationInput,
    CreditValuationOutput,
    CreditType,
    CreditStandard,
    CreditValuation,
    CreditRiskAssessment,
)

from greenlang.agents.finance.climate_finance_tracker import (
    ClimateFinanceTrackerAgent,
    ClimateFinanceInput,
    ClimateFinanceOutput,
    FinanceCategory,
    ClimateFinanceFlow,
    FinanceAlignment,
    ClimateFinanceSummary,
)

from greenlang.agents.finance.eu_taxonomy_alignment_agent import (
    EUTaxonomyAlignmentAgent,
    TaxonomyAlignmentInput,
    TaxonomyAlignmentOutput,
    EconomicActivity,
    EnvironmentalObjective,
    TaxonomyEligibility,
    TaxonomyAlignment,
    DNSHAssessment,
)

from greenlang.agents.finance.green_bond_analyzer import (
    GreenBondAnalyzerAgent,
    GreenBondInput,
    GreenBondOutput,
    BondFramework,
    UseOfProceeds,
    GreenBondAssessment,
    AlignmentScore,
)

from greenlang.agents.finance.climate_budget_agent import (
    ClimateBudgetAgent,
    ClimateBudgetInput,
    ClimateBudgetOutput,
    BudgetPeriod,
    BudgetAllocation,
    BudgetVariance,
    CarbonBudgetStatus,
)

from greenlang.agents.finance.stranded_asset_analyzer import (
    StrandedAssetAnalyzerAgent,
    StrandedAssetInput,
    StrandedAssetOutput,
    AssetCategory,
    RiskFactor,
    StrandingRisk,
    AssetValuationImpact,
)

__all__ = [
    # Carbon Pricing Agent (GL-FIN-X-001)
    "CarbonPricingAgent",
    "CarbonPricingInput",
    "CarbonPricingOutput",
    "CarbonPriceScenario",
    "PricingMechanism",
    "InternalCarbonPrice",
    "CarbonPriceImpact",
    # TCO Calculator Agent (GL-FIN-X-002)
    "TCOCalculatorAgent",
    "TCOCalculatorInput",
    "TCOCalculatorOutput",
    "CostComponent",
    "CarbonCostMethod",
    "TCOResult",
    "AssetComparison",
    # Green Investment Screener (GL-FIN-X-003)
    "GreenInvestmentScreenerAgent",
    "InvestmentScreenerInput",
    "InvestmentScreenerOutput",
    "InvestmentCriteria",
    "ESGRating",
    "ScreeningResult",
    "SustainabilityScore",
    # Carbon Credit Valuation (GL-FIN-X-004)
    "CarbonCreditValuationAgent",
    "CreditValuationInput",
    "CreditValuationOutput",
    "CreditType",
    "CreditStandard",
    "CreditValuation",
    "CreditRiskAssessment",
    # Climate Finance Tracker (GL-FIN-X-005)
    "ClimateFinanceTrackerAgent",
    "ClimateFinanceInput",
    "ClimateFinanceOutput",
    "FinanceCategory",
    "ClimateFinanceFlow",
    "FinanceAlignment",
    "ClimateFinanceSummary",
    # EU Taxonomy Alignment Agent (GL-FIN-X-006)
    "EUTaxonomyAlignmentAgent",
    "TaxonomyAlignmentInput",
    "TaxonomyAlignmentOutput",
    "EconomicActivity",
    "EnvironmentalObjective",
    "TaxonomyEligibility",
    "TaxonomyAlignment",
    "DNSHAssessment",
    # Green Bond Analyzer (GL-FIN-X-007)
    "GreenBondAnalyzerAgent",
    "GreenBondInput",
    "GreenBondOutput",
    "BondFramework",
    "UseOfProceeds",
    "GreenBondAssessment",
    "AlignmentScore",
    # Climate Budget Agent (GL-FIN-X-008)
    "ClimateBudgetAgent",
    "ClimateBudgetInput",
    "ClimateBudgetOutput",
    "BudgetPeriod",
    "BudgetAllocation",
    "BudgetVariance",
    "CarbonBudgetStatus",
    # Stranded Asset Analyzer (GL-FIN-X-009)
    "StrandedAssetAnalyzerAgent",
    "StrandedAssetInput",
    "StrandedAssetOutput",
    "AssetCategory",
    "RiskFactor",
    "StrandingRisk",
    "AssetValuationImpact",
]
