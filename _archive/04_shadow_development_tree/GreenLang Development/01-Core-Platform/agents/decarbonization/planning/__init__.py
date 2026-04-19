# -*- coding: utf-8 -*-
"""
GreenLang Decarbonization Planning Layer Agents
================================================

The Decarbonization Planning Layer provides 21 core agents for
building and executing comprehensive decarbonization strategies.

Agents:
    GL-DECARB-X-001: Abatement Options Library - Catalog of decarbonization levers
    GL-DECARB-X-002: MACC Generator - Marginal Abatement Cost Curves
    GL-DECARB-X-003: Target Setting Agent - SBTi-aligned targets
    GL-DECARB-X-004: Pathway Scenario Builder - Decarbonization scenarios
    GL-DECARB-X-005: Investment Prioritization Agent - NPV/IRR ranking
    GL-DECARB-X-006: Technology Readiness Assessor - TRL evaluation
    GL-DECARB-X-007: Implementation Roadmap Agent - Phased plans
    GL-DECARB-X-008: Avoided Emissions Calculator - Intervention impacts
    GL-DECARB-X-009: Carbon Intensity Tracker - Intensity metrics
    GL-DECARB-X-010: Renewable Energy Planner - RE adoption strategy
    GL-DECARB-X-011: Electrification Planner - Process electrification
    GL-DECARB-X-012: Fuel Switching Optimizer - Fuel transitions
    GL-DECARB-X-013: Energy Efficiency Identifier - Efficiency opportunities
    GL-DECARB-X-014: Carbon Capture Assessor - CCUS evaluation
    GL-DECARB-X-015: Offset Strategy Agent - Carbon credits
    GL-DECARB-X-016: Supplier Engagement Planner - Scope 3 supplier programs
    GL-DECARB-X-017: Scope 3 Reduction Planner - Scope 3 strategies
    GL-DECARB-X-018: Progress Monitoring Agent - Progress tracking
    GL-DECARB-X-019: Scenario Comparison Agent - Scenario analysis
    GL-DECARB-X-020: Cost-Benefit Analyzer - CBA for investments
    GL-DECARB-X-021: Transition Risk Assessor - Climate transition risks

Zero-Hallucination Principle:
    All agents in this layer follow the zero-hallucination principle.
    Calculations use deterministic formulas with documented provenance.
    No AI-generated numeric estimates without source documentation.

Author: GreenLang Team
Version: 1.0.0
"""

# GL-DECARB-X-001: Abatement Options Library
from greenlang.agents.decarbonization.planning.abatement_options_library import (
    AbatementOptionsLibraryAgent,
    AbatementOption,
    AbatementCategory,
    TechnologyReadinessLevel,
    SectorApplicability,
    CostRange,
    EmissionReductionPotential,
    ImplementationTimeline,
    CoBenefit,
    SourceCitation,
    AbatementOptionsLibraryInput,
    AbatementOptionsLibraryOutput,
)

# GL-DECARB-X-002: MACC Generator
from greenlang.agents.decarbonization.planning.macc_generator import (
    MACCGeneratorAgent,
    MACCDataPoint,
    MACCCurve,
    MACCGeneratorInput,
    MACCGeneratorOutput,
    MACCOutputFormat,
    CostMetric,
)

# GL-DECARB-X-003: Target Setting Agent
from greenlang.agents.decarbonization.planning.target_setting_agent import (
    TargetSettingAgent,
    EmissionTarget,
    TargetMilestone,
    BaseYearEmissions,
    TargetType,
    TargetTimeframe,
    TemperatureAlignment,
    SBTiSector,
    ScopeCategory,
    TargetSettingInput,
    TargetSettingOutput,
)

# GL-DECARB-X-004: Pathway Scenario Builder
from greenlang.agents.decarbonization.planning.pathway_scenario_builder import (
    PathwayScenarioBuilderAgent,
    DecarbonizationScenario,
    ScheduledOption,
    ScenarioMilestone,
    ScenarioConstraint,
    ScenarioType,
    ConstraintType,
    PathwayScenarioBuilderInput,
    PathwayScenarioBuilderOutput,
)

# GL-DECARB-X-005: Investment Prioritization Agent
from greenlang.agents.decarbonization.planning.investment_prioritization_agent import (
    InvestmentPrioritizationAgent,
    InvestmentProject,
    InvestmentMetrics,
    CashFlow,
    RankingCriteria,
    InvestmentCategory,
    RiskLevel,
    InvestmentPrioritizationInput,
    InvestmentPrioritizationOutput,
)

# GL-DECARB-X-006: Technology Readiness Assessor
from greenlang.agents.decarbonization.planning.technology_readiness_assessor import (
    TechnologyReadinessAssessor,
    TechnologyAssessment,
    TRLLevel,
    CRLLevel,
    TRLEvidence,
    TechnologyReadinessInput,
    TechnologyReadinessOutput,
)

# GL-DECARB-X-007: Implementation Roadmap Agent
from greenlang.agents.decarbonization.planning.implementation_roadmap_agent import (
    ImplementationRoadmapAgent,
    ImplementationRoadmap,
    RoadmapPhase,
    RoadmapMilestone,
    PhaseType,
    MilestoneStatus,
    ImplementationRoadmapInput,
    ImplementationRoadmapOutput,
)

# GL-DECARB-X-008: Avoided Emissions Calculator
from greenlang.agents.decarbonization.planning.avoided_emissions_calculator import (
    AvoidedEmissionsCalculator,
    AvoidedEmissionsResult,
    BaselineType,
    InterventionType,
    AvoidedEmissionsInput,
    AvoidedEmissionsOutput,
)

# GL-DECARB-X-009: Carbon Intensity Tracker
from greenlang.agents.decarbonization.planning.carbon_intensity_tracker import (
    CarbonIntensityTracker,
    IntensityDataPoint,
    IntensityTrend,
    IntensityMetricType,
    CarbonIntensityInput,
    CarbonIntensityOutput,
)

# GL-DECARB-X-010: Renewable Energy Planner
from greenlang.agents.decarbonization.planning.renewable_energy_planner import (
    RenewableEnergyPlanner,
    RenewableEnergyPlan,
    RenewableProject,
    RenewableType,
    ProcurementOption,
    RenewableEnergyInput,
    RenewableEnergyOutput,
)

# GL-DECARB-X-011: Electrification Planner
from greenlang.agents.decarbonization.planning.electrification_planner import (
    ElectrificationPlanner,
    ElectrificationPlan,
    ElectrificationProject,
    ProcessType,
    ElectrificationTechnology,
    ElectrificationInput,
    ElectrificationOutput,
)

# GL-DECARB-X-012: Fuel Switching Optimizer
from greenlang.agents.decarbonization.planning.fuel_switching_optimizer import (
    FuelSwitchingOptimizer,
    FuelSwitchOption,
    FuelType,
    FuelSwitchingInput,
    FuelSwitchingOutput,
)

# GL-DECARB-X-013: Energy Efficiency Identifier
from greenlang.agents.decarbonization.planning.energy_efficiency_identifier import (
    EnergyEfficiencyIdentifier,
    EfficiencyAssessment,
    EfficiencyOpportunity,
    EfficiencyCategory,
    EnergyEfficiencyInput,
    EnergyEfficiencyOutput,
)

# GL-DECARB-X-014: Carbon Capture Assessor
from greenlang.agents.decarbonization.planning.carbon_capture_assessor import (
    CarbonCaptureAssessor,
    CCUSAssessment,
    CCUSOpportunity,
    CaptureType,
    StorageType,
    CarbonCaptureInput,
    CarbonCaptureOutput,
)

# GL-DECARB-X-015: Offset Strategy Agent
from greenlang.agents.decarbonization.planning.offset_strategy_agent import (
    OffsetStrategyAgent,
    OffsetPortfolio,
    OffsetProject,
    OffsetType,
    OffsetStandard,
    QualityRating,
    OffsetStrategyInput,
    OffsetStrategyOutput,
)

# GL-DECARB-X-016: Supplier Engagement Planner
from greenlang.agents.decarbonization.planning.supplier_engagement_planner import (
    SupplierEngagementPlanner,
    SupplierEngagementPlan,
    SupplierSegment,
    EngagementTier,
    SupplierEngagementInput,
    SupplierEngagementOutput,
)

# GL-DECARB-X-017: Scope 3 Reduction Planner
from greenlang.agents.decarbonization.planning.scope3_reduction_planner import (
    Scope3ReductionPlanner,
    Scope3ReductionPlan,
    Scope3Intervention,
    Scope3Category,
    Scope3ReductionInput,
    Scope3ReductionOutput,
)

# GL-DECARB-X-018: Progress Monitoring Agent
from greenlang.agents.decarbonization.planning.progress_monitoring_agent import (
    ProgressMonitoringAgent,
    ProgressReport,
    ProgressDataPoint,
    ProgressStatus,
    ProgressMonitoringInput,
    ProgressMonitoringOutput,
)

# GL-DECARB-X-019: Scenario Comparison Agent
from greenlang.agents.decarbonization.planning.scenario_comparison_agent import (
    ScenarioComparisonAgent,
    ScenarioComparison,
    ScenarioMetrics,
    ScenarioComparisonInput,
    ScenarioComparisonOutput,
)

# GL-DECARB-X-020: Cost-Benefit Analyzer
from greenlang.agents.decarbonization.planning.cost_benefit_analyzer import (
    CostBenefitAnalyzer,
    CostBenefitAnalysis,
    CostItem,
    BenefitItem,
    BenefitType,
    CostCategory,
    CostBenefitInput,
    CostBenefitOutput,
)

# GL-DECARB-X-021: Transition Risk Assessor
from greenlang.agents.decarbonization.planning.transition_risk_assessor import (
    TransitionRiskAssessor,
    TransitionRiskAssessment,
    TransitionRisk,
    TransitionRiskType,
    RiskLikelihood,
    RiskImpact,
    TransitionRiskInput,
    TransitionRiskOutput,
)


__all__ = [
    # GL-DECARB-X-001: Abatement Options Library
    "AbatementOptionsLibraryAgent",
    "AbatementOption",
    "AbatementCategory",
    "TechnologyReadinessLevel",
    "SectorApplicability",
    "CostRange",
    "EmissionReductionPotential",
    "ImplementationTimeline",
    "CoBenefit",
    "SourceCitation",
    "AbatementOptionsLibraryInput",
    "AbatementOptionsLibraryOutput",
    # GL-DECARB-X-002: MACC Generator
    "MACCGeneratorAgent",
    "MACCDataPoint",
    "MACCCurve",
    "MACCGeneratorInput",
    "MACCGeneratorOutput",
    "MACCOutputFormat",
    "CostMetric",
    # GL-DECARB-X-003: Target Setting Agent
    "TargetSettingAgent",
    "EmissionTarget",
    "TargetMilestone",
    "BaseYearEmissions",
    "TargetType",
    "TargetTimeframe",
    "TemperatureAlignment",
    "SBTiSector",
    "ScopeCategory",
    "TargetSettingInput",
    "TargetSettingOutput",
    # GL-DECARB-X-004: Pathway Scenario Builder
    "PathwayScenarioBuilderAgent",
    "DecarbonizationScenario",
    "ScheduledOption",
    "ScenarioMilestone",
    "ScenarioConstraint",
    "ScenarioType",
    "ConstraintType",
    "PathwayScenarioBuilderInput",
    "PathwayScenarioBuilderOutput",
    # GL-DECARB-X-005: Investment Prioritization Agent
    "InvestmentPrioritizationAgent",
    "InvestmentProject",
    "InvestmentMetrics",
    "CashFlow",
    "RankingCriteria",
    "InvestmentCategory",
    "RiskLevel",
    "InvestmentPrioritizationInput",
    "InvestmentPrioritizationOutput",
    # GL-DECARB-X-006: Technology Readiness Assessor
    "TechnologyReadinessAssessor",
    "TechnologyAssessment",
    "TRLLevel",
    "CRLLevel",
    "TRLEvidence",
    "TechnologyReadinessInput",
    "TechnologyReadinessOutput",
    # GL-DECARB-X-007: Implementation Roadmap Agent
    "ImplementationRoadmapAgent",
    "ImplementationRoadmap",
    "RoadmapPhase",
    "RoadmapMilestone",
    "PhaseType",
    "MilestoneStatus",
    "ImplementationRoadmapInput",
    "ImplementationRoadmapOutput",
    # GL-DECARB-X-008: Avoided Emissions Calculator
    "AvoidedEmissionsCalculator",
    "AvoidedEmissionsResult",
    "BaselineType",
    "InterventionType",
    "AvoidedEmissionsInput",
    "AvoidedEmissionsOutput",
    # GL-DECARB-X-009: Carbon Intensity Tracker
    "CarbonIntensityTracker",
    "IntensityDataPoint",
    "IntensityTrend",
    "IntensityMetricType",
    "CarbonIntensityInput",
    "CarbonIntensityOutput",
    # GL-DECARB-X-010: Renewable Energy Planner
    "RenewableEnergyPlanner",
    "RenewableEnergyPlan",
    "RenewableProject",
    "RenewableType",
    "ProcurementOption",
    "RenewableEnergyInput",
    "RenewableEnergyOutput",
    # GL-DECARB-X-011: Electrification Planner
    "ElectrificationPlanner",
    "ElectrificationPlan",
    "ElectrificationProject",
    "ProcessType",
    "ElectrificationTechnology",
    "ElectrificationInput",
    "ElectrificationOutput",
    # GL-DECARB-X-012: Fuel Switching Optimizer
    "FuelSwitchingOptimizer",
    "FuelSwitchOption",
    "FuelType",
    "FuelSwitchingInput",
    "FuelSwitchingOutput",
    # GL-DECARB-X-013: Energy Efficiency Identifier
    "EnergyEfficiencyIdentifier",
    "EfficiencyAssessment",
    "EfficiencyOpportunity",
    "EfficiencyCategory",
    "EnergyEfficiencyInput",
    "EnergyEfficiencyOutput",
    # GL-DECARB-X-014: Carbon Capture Assessor
    "CarbonCaptureAssessor",
    "CCUSAssessment",
    "CCUSOpportunity",
    "CaptureType",
    "StorageType",
    "CarbonCaptureInput",
    "CarbonCaptureOutput",
    # GL-DECARB-X-015: Offset Strategy Agent
    "OffsetStrategyAgent",
    "OffsetPortfolio",
    "OffsetProject",
    "OffsetType",
    "OffsetStandard",
    "QualityRating",
    "OffsetStrategyInput",
    "OffsetStrategyOutput",
    # GL-DECARB-X-016: Supplier Engagement Planner
    "SupplierEngagementPlanner",
    "SupplierEngagementPlan",
    "SupplierSegment",
    "EngagementTier",
    "SupplierEngagementInput",
    "SupplierEngagementOutput",
    # GL-DECARB-X-017: Scope 3 Reduction Planner
    "Scope3ReductionPlanner",
    "Scope3ReductionPlan",
    "Scope3Intervention",
    "Scope3Category",
    "Scope3ReductionInput",
    "Scope3ReductionOutput",
    # GL-DECARB-X-018: Progress Monitoring Agent
    "ProgressMonitoringAgent",
    "ProgressReport",
    "ProgressDataPoint",
    "ProgressStatus",
    "ProgressMonitoringInput",
    "ProgressMonitoringOutput",
    # GL-DECARB-X-019: Scenario Comparison Agent
    "ScenarioComparisonAgent",
    "ScenarioComparison",
    "ScenarioMetrics",
    "ScenarioComparisonInput",
    "ScenarioComparisonOutput",
    # GL-DECARB-X-020: Cost-Benefit Analyzer
    "CostBenefitAnalyzer",
    "CostBenefitAnalysis",
    "CostItem",
    "BenefitItem",
    "BenefitType",
    "CostCategory",
    "CostBenefitInput",
    "CostBenefitOutput",
    # GL-DECARB-X-021: Transition Risk Assessor
    "TransitionRiskAssessor",
    "TransitionRiskAssessment",
    "TransitionRisk",
    "TransitionRiskType",
    "RiskLikelihood",
    "RiskImpact",
    "TransitionRiskInput",
    "TransitionRiskOutput",
]
