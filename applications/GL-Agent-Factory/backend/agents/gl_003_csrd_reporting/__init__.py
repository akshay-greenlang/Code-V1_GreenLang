"""
GL-003: CSRD Reporting Agent

Corporate Sustainability Reporting Directive compliance analyzer implementing
ESRS (European Sustainability Reporting Standards) per EU Directive 2022/2464.

Features:
- Double materiality assessment (impact + financial)
- Complete ESRS coverage (ESRS 1-2, E1-E5, S1-S4, G1)
- Phase-in disclosure tracking
- iXBRL/ESEF report generation
- Sector-specific standards support
"""

from .agent import (
    # Main Agent and I/O
    CSRDReportingAgent,
    CSRDInput,
    CSRDOutput,
    # Enumerations
    ESRSStandard,
    MaterialityLevel,
    CompanySize,
    DisclosureType,
    AssuranceLevel,
    SectorCategory,
    IROMaterialityType,
    # Double Materiality Models
    MaterialityAssessment,
    IROAssessment,
    # Cross-Cutting Standards (ESRS 2)
    ESRS2Governance,
    ESRS2Strategy,
    ESRS2IRO,
    # Environmental Standards
    E1ClimateData,
    E2PollutionData,
    E3WaterData,
    E4BiodiversityData,
    E5CircularEconomyData,
    # Social Standards
    S1WorkforceData,
    S2ValueChainWorkersData,
    S3CommunitiesData,
    S4ConsumersData,
    # Governance Standards
    G1GovernanceData,
    # Disclosure Models
    ESRSDatapoint,
    GapAnalysisItem,
    ComplianceMetrics,
    # ESEF/iXBRL Models
    XBRLTag,
    ESEFReportOutput,
    # Reference Data
    ESRS_DISCLOSURE_REQUIREMENTS,
    SECTOR_SPECIFIC_REQUIREMENTS,
    PACK_SPEC,
)

__all__ = [
    # Main Agent and I/O
    "CSRDReportingAgent",
    "CSRDInput",
    "CSRDOutput",
    # Enumerations
    "ESRSStandard",
    "MaterialityLevel",
    "CompanySize",
    "DisclosureType",
    "AssuranceLevel",
    "SectorCategory",
    "IROMaterialityType",
    # Double Materiality Models
    "MaterialityAssessment",
    "IROAssessment",
    # Cross-Cutting Standards (ESRS 2)
    "ESRS2Governance",
    "ESRS2Strategy",
    "ESRS2IRO",
    # Environmental Standards
    "E1ClimateData",
    "E2PollutionData",
    "E3WaterData",
    "E4BiodiversityData",
    "E5CircularEconomyData",
    # Social Standards
    "S1WorkforceData",
    "S2ValueChainWorkersData",
    "S3CommunitiesData",
    "S4ConsumersData",
    # Governance Standards
    "G1GovernanceData",
    # Disclosure Models
    "ESRSDatapoint",
    "GapAnalysisItem",
    "ComplianceMetrics",
    # ESEF/iXBRL Models
    "XBRLTag",
    "ESEFReportOutput",
    # Reference Data
    "ESRS_DISCLOSURE_REQUIREMENTS",
    "SECTOR_SPECIFIC_REQUIREMENTS",
    "PACK_SPEC",
]
