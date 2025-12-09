"""
GL-004: EUDR Compliance Agent

EU Deforestation Regulation compliance validator per EU 2023/1115.

This module provides comprehensive EUDR compliance validation including:
- GeoJSON polygon validation with CRS transformation
- Self-intersection detection for plot boundaries
- Forest cover change detection via satellite data integration
- Commodity risk assessment for all 7 regulated commodities
- Supply chain traceability tracking
- Due Diligence Statement (DDS) generation

Regulatory Reference:
    EU Regulation 2023/1115 (OJ L 150, 9.6.2023)
    Enforcement Date: December 30, 2025
    SME Enforcement Date: June 30, 2026
    Cutoff Date: December 31, 2020

Example:
    >>> from backend.agents.gl_004_eudr_compliance import (
    ...     EUDRComplianceAgent,
    ...     EUDRInput,
    ...     CommodityType,
    ...     GeoLocation,
    ...     GeometryType,
    ... )
    >>> from datetime import date
    >>> agent = EUDRComplianceAgent()
    >>> result = agent.run(EUDRInput(
    ...     commodity_type=CommodityType.COFFEE,
    ...     cn_code="0901.11.00",
    ...     quantity_kg=50000,
    ...     country_of_origin="BR",
    ...     geolocation=GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5]),
    ...     production_date=date(2024, 6, 1)
    ... ))
    >>> print(f"Status: {result.compliance_status}, Risk: {result.risk_level}")
"""

from .agent import (
    # Main Agent
    EUDRComplianceAgent,
    # Input/Output Models
    EUDRInput,
    EUDROutput,
    # Core Enums
    CommodityType,
    RiskLevel,
    ComplianceStatus,
    GeometryType,
    DueDiligenceType,
    DeforestationStatus,
    ValidationSeverity,
    # Geolocation Models
    GeoLocation,
    GeolocationValidationResult,
    # Supply Chain Models
    SupplierInfo,
    SupplyChainNode,
    # Analysis Models
    ForestCoverAnalysis,
    RiskAssessment,
    DDSDocument,
    ValidationError,
    # Country Risk
    CountryRisk,
    HIGH_RISK_COUNTRIES,
    STANDARD_RISK_COUNTRIES,
    LOW_RISK_COUNTRIES,
    DEFAULT_COUNTRY_RISK,
    # Mappings
    CN_TO_COMMODITY,
    RECOGNIZED_CERTIFICATIONS,
    # Pack Spec
    PACK_SPEC,
)

__all__ = [
    # Main Agent
    "EUDRComplianceAgent",
    # Input/Output Models
    "EUDRInput",
    "EUDROutput",
    # Core Enums
    "CommodityType",
    "RiskLevel",
    "ComplianceStatus",
    "GeometryType",
    "DueDiligenceType",
    "DeforestationStatus",
    "ValidationSeverity",
    # Geolocation Models
    "GeoLocation",
    "GeolocationValidationResult",
    # Supply Chain Models
    "SupplierInfo",
    "SupplyChainNode",
    # Analysis Models
    "ForestCoverAnalysis",
    "RiskAssessment",
    "DDSDocument",
    "ValidationError",
    # Country Risk
    "CountryRisk",
    "HIGH_RISK_COUNTRIES",
    "STANDARD_RISK_COUNTRIES",
    "LOW_RISK_COUNTRIES",
    "DEFAULT_COUNTRY_RISK",
    # Mappings
    "CN_TO_COMMODITY",
    "RECOGNIZED_CERTIFICATIONS",
    # Pack Spec
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "regulatory/eudr_compliance_v1"
