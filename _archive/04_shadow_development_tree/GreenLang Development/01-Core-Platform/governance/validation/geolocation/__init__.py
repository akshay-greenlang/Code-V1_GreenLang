# -*- coding: utf-8 -*-
"""
GreenLang EUDR Geolocation Validation Module

Zero-hallucination geolocation validation for EUDR compliance.
All calculations are deterministic with complete provenance tracking.

This module provides:
- GeoJSON parsing and validation
- WGS84 coordinate validation
- Protected area intersection detection (WDPA)
- Deforestation baseline checking (Dec 31, 2020 cutoff)
- Country and commodity risk assessment
- Complete validation orchestration

Example:
    from greenlang.validation.geolocation import (
        EUDRGeolocationValidator,
        EUDRCommodity
    )

    # Create validator
    validator = EUDRGeolocationValidator()

    # Validate a polygon
    geojson = {
        "type": "Polygon",
        "coordinates": [[
            [-61.234567, -3.456789],
            [-61.234567, -3.556789],
            [-61.134567, -3.556789],
            [-61.134567, -3.456789],
            [-61.234567, -3.456789]
        ]]
    }

    report = validator.validate(
        geojson=geojson,
        country_iso3="BRA",
        commodity=EUDRCommodity.SOYA
    )

    print(f"Compliance: {report.compliance_status.value}")
    print(f"Score: {report.compliance_score}")
    print(f"EUDR Compliant: {report.is_eudr_compliant}")
    print(f"Provenance Hash: {report.provenance_hash}")

Author: GreenLang Calculator Engine
License: Proprietary
"""

# GeoJSON Parser
from .geojson_parser import (
    GeoJSONParser,
    GeoJSONParseResult,
    GeoJSONType,
    Coordinate,
    BoundingBox,
    PolygonRing,
    ParsedPoint,
    ParsedPolygon,
    ParsedMultiPolygon,
)

# Coordinate Validator
from .coordinate_validator import (
    CoordinateValidator,
    CoordinateValidationResult,
    DistanceResult,
    BufferResult,
    CoordinateSystem,
    DistanceUnit,
    AreaUnit,
    PrecisionLevel,
)

# Protected Area Checker
from .protected_area_checker import (
    ProtectedAreaChecker,
    ProtectedAreaCheckResult,
    ProtectedArea,
    ProtectedAreaIntersection,
    ProtectionStatus,
    ProtectionLevel,
    IUCNCategory,
    WDPAQueryResult,
)

# Deforestation Baseline
from .deforestation_baseline import (
    DeforestationBaselineChecker,
    BaselineCheckResult,
    ForestStatus,
    DeforestationRisk,
    ForestDefinition,
    ForestCoverData,
    DeforestationEvent,
    EUDR_CUTOFF_DATE,
)

# Country Risk Database
from .country_risk_db import (
    CountryRiskDatabase,
    CountryRiskProfile,
    CountryRiskQueryResult,
    CountryInfo,
    ForestStatistics,
    CommodityRisk,
    GovernanceScores,
    EUDRRiskCategory,
    EUDRCommodity,
    DeforestationDriver,
    GovernanceIndicator,
)

# Validation Engine
from .validation_engine import (
    EUDRGeolocationValidator,
    GeoLocationValidationReport,
    ValidationEngineConfig,
    ValidationCheckType,
    ValidationSeverity,
    ValidationFinding,
    CheckResult,
    ComplianceStatus,
)


__all__ = [
    # GeoJSON Parser
    "GeoJSONParser",
    "GeoJSONParseResult",
    "GeoJSONType",
    "Coordinate",
    "BoundingBox",
    "PolygonRing",
    "ParsedPoint",
    "ParsedPolygon",
    "ParsedMultiPolygon",

    # Coordinate Validator
    "CoordinateValidator",
    "CoordinateValidationResult",
    "DistanceResult",
    "BufferResult",
    "CoordinateSystem",
    "DistanceUnit",
    "AreaUnit",
    "PrecisionLevel",

    # Protected Area Checker
    "ProtectedAreaChecker",
    "ProtectedAreaCheckResult",
    "ProtectedArea",
    "ProtectedAreaIntersection",
    "ProtectionStatus",
    "ProtectionLevel",
    "IUCNCategory",
    "WDPAQueryResult",

    # Deforestation Baseline
    "DeforestationBaselineChecker",
    "BaselineCheckResult",
    "ForestStatus",
    "DeforestationRisk",
    "ForestDefinition",
    "ForestCoverData",
    "DeforestationEvent",
    "EUDR_CUTOFF_DATE",

    # Country Risk Database
    "CountryRiskDatabase",
    "CountryRiskProfile",
    "CountryRiskQueryResult",
    "CountryInfo",
    "ForestStatistics",
    "CommodityRisk",
    "GovernanceScores",
    "EUDRRiskCategory",
    "EUDRCommodity",
    "DeforestationDriver",
    "GovernanceIndicator",

    # Validation Engine
    "EUDRGeolocationValidator",
    "GeoLocationValidationReport",
    "ValidationEngineConfig",
    "ValidationCheckType",
    "ValidationSeverity",
    "ValidationFinding",
    "CheckResult",
    "ComplianceStatus",
]


# Module version
__version__ = "1.0.0"

# Module metadata
__author__ = "GreenLang Calculator Engine"
__license__ = "Proprietary"
