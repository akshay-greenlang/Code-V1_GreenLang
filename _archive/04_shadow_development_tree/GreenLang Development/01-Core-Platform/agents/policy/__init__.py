# -*- coding: utf-8 -*-
"""
GreenLang Policy Layer Agents
=============================

This module contains agents for regulatory compliance and policy intelligence.
These agents follow the Intelligence Paradox architecture with CRITICAL PATH
(zero-hallucination) and INSIGHT PATH (deterministic + AI analysis) patterns.

Agent Registry:
    GL-POL-X-001: Regulatory Mapping Agent - Maps applicable regulations
    GL-POL-X-002: Compliance Gap Analyzer - Identifies compliance gaps
    GL-POL-X-003: Policy Intelligence Agent - Monitors regulatory changes
    GL-POL-X-004: Standard Alignment Agent - Aligns with standards (GRI, SASB)
    GL-POL-X-005: Carbon Tax Calculator - Calculates carbon tax exposure
    GL-POL-X-006: CBAM Compliance Agent - EU CBAM compliance
    GL-POL-X-007: CSRD Compliance Agent - EU CSRD compliance
    GL-POL-X-008: Biodiversity Compliance Agent - Biodiversity regulations

Zero-Hallucination Guarantees:
    All regulatory determinations and compliance calculations are derived
    from curated databases and deterministic formulas. No LLM inference
    is used for compliance status determination or numeric calculations.

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.agents.policy.regulatory_mapping_agent import (
    RegulatoryMappingAgent,
    Jurisdiction,
    RegulationType,
    ApplicabilityStatus,
    IndustryClassification,
    RegulationDefinition,
    OrganizationProfile,
    ApplicabilityResult,
    RegulatoryMappingResult,
    RegulatoryMappingInput,
    RegulatoryMappingOutput,
    REGULATORY_DATABASE,
)

from greenlang.agents.policy.compliance_gap_analyzer import (
    ComplianceGapAnalyzer,
    ComplianceDomain,
    MaturityLevel,
    GapSeverity,
    GapStatus,
    RemediationEffort,
    ComplianceRequirement,
    CurrentStateAssessment,
    ComplianceGap,
    GapAnalysisResult,
    GapAnalysisInput,
    GapAnalysisOutput,
    COMPLIANCE_REQUIREMENTS,
)

from greenlang.agents.policy.policy_intelligence_agent import (
    PolicyIntelligenceAgent,
    PolicyChangeType,
    ChangeStatus,
    ImpactLevel,
    AlertPriority,
    PolicyChange,
    PolicyAlert,
    ImpactAssessment,
    PolicyIntelligenceInput,
    PolicyIntelligenceOutput,
)

from greenlang.agents.policy.standard_alignment_agent import (
    StandardAlignmentAgent,
    StandardFramework,
    DisclosureCategory,
    DataPointType,
    AlignmentStatus,
    StandardIndicator,
    CrossWalkMapping,
    DataPointMapping,
    AlignmentResult,
    StandardAlignmentInput,
    StandardAlignmentOutput,
    STANDARD_INDICATORS,
    CROSSWALK_MAPPINGS,
)

from greenlang.agents.policy.carbon_tax_calculator import (
    CarbonTaxCalculator,
    TaxJurisdiction,
    CoverageScope,
    SectorType,
    CarbonTaxRate,
    EmissionsProfile,
    TaxLiabilityItem,
    CarbonTaxResult,
    CarbonTaxInput,
    CarbonTaxOutput,
    CARBON_TAX_RATES,
)

from greenlang.agents.policy.cbam_compliance_agent import (
    CBAMComplianceAgent,
    CBAMSector,
    CBAMProductCategory,
    ReportingPeriod,
    EmissionsDataSource,
    ImportedGood,
    CBAMCalculationResult,
    CBAMQuarterlyReport,
    CBAMComplianceInput,
    CBAMComplianceOutput,
    DEFAULT_EMISSION_FACTORS,
)

from greenlang.agents.policy.csrd_compliance_agent import (
    CSRDComplianceAgent,
    ESRSStandard,
    DisclosureRequirement,
    ComplianceStatus,
    PhaseInCategory,
    ESRSDataPoint,
    MaterialityAssessment,
    DataPointResponse,
    CSRDComplianceResult,
    CSRDComplianceInput,
    CSRDComplianceOutput,
    ESRS_DATA_POINTS,
)

from greenlang.agents.policy.biodiversity_compliance_agent import (
    BiodiversityComplianceAgent,
    TNFDPillar,
    BiodiversityPressure,
    EcosystemType,
    DependencyLevel,
    RiskCategory,
    LEAPPhase,
    OperationalSite,
    BiodiversityImpact,
    NatureDependency,
    NatureRisk,
    LEAPAnalysis,
    BiodiversityComplianceInput,
    BiodiversityComplianceOutput,
)


__all__ = [
    # GL-POL-X-001: Regulatory Mapping Agent
    "RegulatoryMappingAgent",
    "Jurisdiction",
    "RegulationType",
    "ApplicabilityStatus",
    "IndustryClassification",
    "RegulationDefinition",
    "OrganizationProfile",
    "ApplicabilityResult",
    "RegulatoryMappingResult",
    "RegulatoryMappingInput",
    "RegulatoryMappingOutput",
    "REGULATORY_DATABASE",

    # GL-POL-X-002: Compliance Gap Analyzer
    "ComplianceGapAnalyzer",
    "ComplianceDomain",
    "MaturityLevel",
    "GapSeverity",
    "GapStatus",
    "RemediationEffort",
    "ComplianceRequirement",
    "CurrentStateAssessment",
    "ComplianceGap",
    "GapAnalysisResult",
    "GapAnalysisInput",
    "GapAnalysisOutput",
    "COMPLIANCE_REQUIREMENTS",

    # GL-POL-X-003: Policy Intelligence Agent
    "PolicyIntelligenceAgent",
    "PolicyChangeType",
    "ChangeStatus",
    "ImpactLevel",
    "AlertPriority",
    "PolicyChange",
    "PolicyAlert",
    "ImpactAssessment",
    "PolicyIntelligenceInput",
    "PolicyIntelligenceOutput",

    # GL-POL-X-004: Standard Alignment Agent
    "StandardAlignmentAgent",
    "StandardFramework",
    "DisclosureCategory",
    "DataPointType",
    "AlignmentStatus",
    "StandardIndicator",
    "CrossWalkMapping",
    "DataPointMapping",
    "AlignmentResult",
    "StandardAlignmentInput",
    "StandardAlignmentOutput",
    "STANDARD_INDICATORS",
    "CROSSWALK_MAPPINGS",

    # GL-POL-X-005: Carbon Tax Calculator
    "CarbonTaxCalculator",
    "TaxJurisdiction",
    "CoverageScope",
    "SectorType",
    "CarbonTaxRate",
    "EmissionsProfile",
    "TaxLiabilityItem",
    "CarbonTaxResult",
    "CarbonTaxInput",
    "CarbonTaxOutput",
    "CARBON_TAX_RATES",

    # GL-POL-X-006: CBAM Compliance Agent
    "CBAMComplianceAgent",
    "CBAMSector",
    "CBAMProductCategory",
    "ReportingPeriod",
    "EmissionsDataSource",
    "ImportedGood",
    "CBAMCalculationResult",
    "CBAMQuarterlyReport",
    "CBAMComplianceInput",
    "CBAMComplianceOutput",
    "DEFAULT_EMISSION_FACTORS",

    # GL-POL-X-007: CSRD Compliance Agent
    "CSRDComplianceAgent",
    "ESRSStandard",
    "DisclosureRequirement",
    "ComplianceStatus",
    "PhaseInCategory",
    "ESRSDataPoint",
    "MaterialityAssessment",
    "DataPointResponse",
    "CSRDComplianceResult",
    "CSRDComplianceInput",
    "CSRDComplianceOutput",
    "ESRS_DATA_POINTS",

    # GL-POL-X-008: Biodiversity Compliance Agent
    "BiodiversityComplianceAgent",
    "TNFDPillar",
    "BiodiversityPressure",
    "EcosystemType",
    "DependencyLevel",
    "RiskCategory",
    "LEAPPhase",
    "OperationalSite",
    "BiodiversityImpact",
    "NatureDependency",
    "NatureRisk",
    "LEAPAnalysis",
    "BiodiversityComplianceInput",
    "BiodiversityComplianceOutput",
]


# Agent registry for discovery
POLICY_AGENTS = {
    "GL-POL-X-001": RegulatoryMappingAgent,
    "GL-POL-X-002": ComplianceGapAnalyzer,
    "GL-POL-X-003": PolicyIntelligenceAgent,
    "GL-POL-X-004": StandardAlignmentAgent,
    "GL-POL-X-005": CarbonTaxCalculator,
    "GL-POL-X-006": CBAMComplianceAgent,
    "GL-POL-X-007": CSRDComplianceAgent,
    "GL-POL-X-008": BiodiversityComplianceAgent,
}
