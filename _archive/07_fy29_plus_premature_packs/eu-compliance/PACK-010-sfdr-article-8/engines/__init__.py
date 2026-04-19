# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Engines Module

Eight calculation engines for SFDR Article 8 product compliance and disclosure:

    1. PAIIndicatorCalculatorEngine          - All 18 mandatory PAI indicators
    2. TaxonomyAlignmentRatioEngine          - EU Taxonomy alignment ratios
    3. SFDRDNSHEngine                        - SFDR-specific DNSH assessment
    4. GoodGovernanceEngine                  - Article 2(17) good governance checks
    5. ESGCharacteristicsEngine              - ESG characteristic definition and
                                               attainment monitoring
    6. SustainableInvestmentCalculatorEngine  - Article 2(17) sustainable investment
                                               classification
    7. PortfolioCarbonFootprintEngine        - WACI, financed emissions, and
                                               temperature alignment
    8. EETDataEngine                         - European ESG Template data management

Regulatory Basis:
    EU Regulation 2019/2088 (SFDR)
    EU Delegated Regulation 2022/1288 (SFDR RTS)
    EU Regulation 2020/852 (Taxonomy)

Pack Tier: Specialist (PACK-010)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-010"
__pack_name__: str = "SFDR Article 8"
__engines_count__: int = 8

_loaded_engines: list[str] = []

# ===================================================================
# Engine 1: PAI Indicator Calculator
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "PAIIndicatorCalculatorEngine",
    "PAIIndicatorConfig",
    "PAIIndicatorId",
    "PAIPAICategory",
    "PAIResult",
    "PAISingleResult",
    "PAICoverage",
    "PAIPeriodComparison",
    "InvesteeData",
    "InvesteeGHGData",
    "InvesteeEnvironmentalData",
    "InvesteeSocialData",
    "InvesteeEnergyData",
    "SovereignData",
    "RealEstateData",
    "DataQualityFlag",
    "NACESector",
]
try:
    from .pai_indicator_calculator import (
        PAIIndicatorCalculatorEngine,
        PAIIndicatorConfig,
        PAIIndicatorId,
        PAICategory as PAIPAICategory,
        PAIResult,
        PAISingleResult,
        PAICoverage,
        PAIPeriodComparison,
        InvesteeData,
        InvesteeGHGData,
        InvesteeEnvironmentalData,
        InvesteeSocialData,
        InvesteeEnergyData,
        SovereignData,
        RealEstateData,
        DataQualityFlag,
        NACESector,
    )
    _loaded_engines.append("PAIIndicatorCalculatorEngine")
except ImportError as e:
    logger.debug("Engine 1 (PAIIndicatorCalculatorEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ===================================================================
# Engine 2: Taxonomy Alignment Ratio
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "TaxonomyAlignmentRatioEngine",
    "TaxonomyAlignmentConfig",
    "HoldingAlignmentData",
    "AlignmentResult",
    "AlignmentCategory",
    "EnvironmentalObjective",
    "ObjectiveBreakdown",
    "PieChartSlice",
    "CommitmentAdherence",
    "GASExposureType",
]
try:
    from .taxonomy_alignment_ratio import (
        TaxonomyAlignmentRatioEngine,
        TaxonomyAlignmentConfig,
        HoldingAlignmentData,
        AlignmentResult,
        AlignmentCategory,
        EnvironmentalObjective,
        ObjectiveBreakdown,
        PieChartSlice,
        CommitmentAdherence,
        GASExposureType,
    )
    _loaded_engines.append("TaxonomyAlignmentRatioEngine")
except ImportError as e:
    logger.debug("Engine 2 (TaxonomyAlignmentRatioEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ===================================================================
# Engine 3: SFDR DNSH
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "SFDRDNSHEngine",
    "DNSHConfig",
    "DNSHAssessment",
    "DNSHStatus",
    "InvestmentPAIData",
    "PAIThreshold",
    "PAIDNSHCheck",
    "PortfolioDNSHResult",
    "DNSHReportSection",
    "ThresholdDirection",
    "SeverityLevel",
    "DNSHPAICategory",
]
try:
    from .sfdr_dnsh_engine import (
        SFDRDNSHEngine,
        DNSHConfig,
        DNSHAssessment,
        DNSHStatus,
        InvestmentPAIData,
        PAIThreshold,
        PAIDNSHCheck,
        PortfolioDNSHResult,
        DNSHReportSection,
        ThresholdDirection,
        SeverityLevel,
        PAICategory as DNSHPAICategory,
    )
    _loaded_engines.append("SFDRDNSHEngine")
except ImportError as e:
    logger.debug("Engine 3 (SFDRDNSHEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ===================================================================
# Engine 4: Good Governance
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "GoodGovernanceEngine",
    "GovernanceConfig",
    "CompanyGovernanceData",
    "GovernanceResult",
    "GovernanceArea",
    "GovernanceStatus",
    "GovernanceCriterion",
    "AreaResult",
    "GovernanceViolation",
    "PortfolioGovernanceResult",
    "ManagementStructureData",
    "EmployeeRelationsData",
    "RemunerationData",
    "TaxComplianceData",
    "ViolationType",
]
try:
    from .good_governance_engine import (
        GoodGovernanceEngine,
        GovernanceConfig,
        CompanyGovernanceData,
        GovernanceResult,
        GovernanceArea,
        GovernanceStatus,
        GovernanceCriterion,
        AreaResult,
        GovernanceViolation,
        PortfolioGovernanceResult,
        ManagementStructureData,
        EmployeeRelationsData,
        RemunerationData,
        TaxComplianceData,
        ViolationType,
    )
    _loaded_engines.append("GoodGovernanceEngine")
except ImportError as e:
    logger.debug("Engine 4 (GoodGovernanceEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ===================================================================
# Engine 5: ESG Characteristics
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "ESGCharacteristicsEngine",
    "ESGCharacteristicsConfig",
    "CharacteristicDefinition",
    "BindingElement",
    "SustainabilityIndicator",
    "AttainmentResult",
    "BenchmarkComparison",
    "StrategyValidationResult",
    "CharacteristicsSummary",
    "CharacteristicType",
    "CharacteristicStatus",
    "AttainmentStatus",
    "BindingElementStatus",
    "BenchmarkType",
]
try:
    from .esg_characteristics_engine import (
        ESGCharacteristicsEngine,
        ESGCharacteristicsConfig,
        CharacteristicDefinition,
        BindingElement,
        SustainabilityIndicator,
        AttainmentResult,
        BenchmarkComparison,
        StrategyValidationResult,
        CharacteristicsSummary,
        CharacteristicType,
        CharacteristicStatus,
        AttainmentStatus,
        BindingElementStatus,
        BenchmarkType,
    )
    _loaded_engines.append("ESGCharacteristicsEngine")
except ImportError as e:
    logger.debug("Engine 5 (ESGCharacteristicsEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ===================================================================
# Engine 6: Sustainable Investment Calculator
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "SustainableInvestmentCalculatorEngine",
    "SustainableInvestmentConfig",
    "InvestmentData",
    "InvestmentClassification",
    "InvestmentClassificationType",
    "ProportionResult",
    "SICommitmentAdherence",
    "ClassificationSummary",
    "SIDNSHAssessment",
    "GovernanceAssessment",
    "SIDNSHStatus",
    "SIGovernanceStatus",
    "ObjectiveContribution",
    "AdherenceStatus",
]
try:
    from .sustainable_investment_calculator import (
        SustainableInvestmentCalculatorEngine,
        SustainableInvestmentConfig,
        InvestmentData,
        InvestmentClassification,
        InvestmentClassificationType,
        ProportionResult,
        CommitmentAdherence as SICommitmentAdherence,
        ClassificationSummary,
        DNSHAssessment as SIDNSHAssessment,
        GovernanceAssessment,
        DNSHStatus as SIDNSHStatus,
        GovernanceStatus as SIGovernanceStatus,
        ObjectiveContribution,
        AdherenceStatus,
    )
    _loaded_engines.append("SustainableInvestmentCalculatorEngine")
except ImportError as e:
    logger.debug("Engine 6 (SustainableInvestmentCalculatorEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ===================================================================
# Engine 7: Portfolio Carbon Footprint
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "PortfolioCarbonFootprintEngine",
    "CarbonFootprintConfig",
    "HoldingEmissions",
    "WACIResult",
    "CarbonFootprintResult",
    "FinancedEmissionsResult",
    "TemperatureAlignment",
    "SectorBreakdown",
    "CarbonSummary",
    "CarbonMethodology",
    "ScopeCoverage",
    "AttributionMethod",
    "CarbonDataQuality",
]
try:
    from .portfolio_carbon_footprint import (
        PortfolioCarbonFootprintEngine,
        CarbonFootprintConfig,
        HoldingEmissions,
        WACIResult,
        CarbonFootprintResult,
        FinancedEmissionsResult,
        TemperatureAlignment,
        SectorBreakdown,
        CarbonSummary,
        CarbonMethodology,
        ScopeCoverage,
        AttributionMethod,
        DataQuality as CarbonDataQuality,
    )
    _loaded_engines.append("PortfolioCarbonFootprintEngine")
except ImportError as e:
    logger.debug("Engine 7 (PortfolioCarbonFootprintEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

# ===================================================================
# Engine 8: EET Data
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "EETDataEngine",
    "EETConfig",
    "EETField",
    "EETDataSet",
    "EETValidationResult",
    "EETExportResult",
    "EETVersion",
    "EETSection",
    "EETDataType",
    "SFDRClassification",
    "ExportFormat",
]
try:
    from .eet_data_engine import (
        EETDataEngine,
        EETConfig,
        EETField,
        EETDataSet,
        EETValidationResult,
        EETExportResult,
        EETVersion,
        EETSection,
        EETDataType,
        SFDRClassification,
        ExportFormat,
    )
    _loaded_engines.append("EETDataEngine")
except ImportError as e:
    logger.debug("Engine 8 (EETDataEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []

# ===================================================================
# Public API
# ===================================================================

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
]


def get_loaded_engines() -> list[str]:
    """Return list of successfully loaded engine class names."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of successfully loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-010 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
