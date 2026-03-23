# -*- coding: utf-8 -*-
"""
Normalizer Data Models - AGENT-FOUND-003: Unit & Reference Normalizer

Pydantic v2 models for the Normalizer SDK covering:
- Enumerations: UnitDimension, GHGGas, GWPVersion, ConfidenceLevel
- Conversion results: ConversionResult, BatchConversionResult
- Entity resolution: EntityMatch, EntityResolutionResult
- Dimensional info: DimensionInfo, GWPInfo
- Provenance: ConversionProvenance

All models use Pydantic v2 patterns with Field descriptions and
validators for data integrity.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# ENUMERATIONS
# =============================================================================


class UnitDimension(str, Enum):
    """Supported physical dimensions for unit classification."""
    MASS = "mass"
    ENERGY = "energy"
    VOLUME = "volume"
    AREA = "area"
    DISTANCE = "distance"
    EMISSIONS = "emissions"
    CURRENCY = "currency"
    TIME = "time"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"


class GHGGas(str, Enum):
    """Greenhouse gas types recognised by the normalizer."""
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFCs = "HFCs"
    PFCs = "PFCs"
    SF6 = "SF6"
    NF3 = "NF3"
    CO2E = "CO2e"


class GWPVersion(str, Enum):
    """IPCC Assessment Report versions for GWP tables."""
    AR5 = "AR5"
    AR6 = "AR6"


class ConfidenceLevel(str, Enum):
    """Confidence levels for entity resolution matches."""
    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    LOW = "low"
    UNRESOLVED = "unresolved"


# =============================================================================
# CONVERSION MODELS
# =============================================================================


class ConversionResult(BaseModel):
    """Result of a single unit conversion operation.

    Attributes:
        value: Legacy alias for converted_value.
        from_unit: Source unit name as provided.
        to_unit: Target unit name as provided.
        from_value: Original numeric value before conversion.
        converted_value: Numeric value after conversion.
        dimension: Physical dimension of the conversion.
        conversion_factor: Factor applied (from_base / to_base).
        provenance_hash: SHA-256 hash for audit trail.
    """
    value: Decimal = Field(
        ..., description="Converted value (alias for converted_value)",
    )
    from_unit: str = Field(..., description="Source unit")
    to_unit: str = Field(..., description="Target unit")
    from_value: Decimal = Field(..., description="Original input value")
    converted_value: Decimal = Field(..., description="Converted output value")
    dimension: UnitDimension = Field(..., description="Physical dimension")
    conversion_factor: Decimal = Field(..., description="Conversion factor applied")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    model_config = {"arbitrary_types_allowed": True}


class BatchConversionResult(BaseModel):
    """Result of a batch conversion operation.

    Attributes:
        results: List of individual conversion results.
        total: Total items attempted.
        succeeded: Number of successful conversions.
        failed: Number of failed conversions.
        duration_ms: Total processing duration in milliseconds.
    """
    results: List[ConversionResult] = Field(
        default_factory=list, description="Individual conversion results",
    )
    total: int = Field(..., description="Total items attempted")
    succeeded: int = Field(..., description="Successful conversions")
    failed: int = Field(..., description="Failed conversions")
    duration_ms: float = Field(..., description="Processing duration in ms")


# =============================================================================
# ENTITY RESOLUTION MODELS
# =============================================================================


class EntityMatch(BaseModel):
    """Result of resolving a single entity (fuel, material, process).

    Attributes:
        raw_input: Original input string.
        resolved_id: Canonical identifier code.
        canonical_name: Standardized entity name.
        entity_type: Type of entity (fuel, material, process).
        confidence: Numeric confidence score 0.0-1.0.
        confidence_level: Categorical confidence level.
        match_method: Method used to match (exact, alias, fuzzy).
    """
    raw_input: str = Field(..., description="Original input string")
    resolved_id: str = Field(..., description="Canonical identifier code")
    canonical_name: str = Field(..., description="Standardized entity name")
    entity_type: str = Field(..., description="Entity type (fuel/material/process)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Match confidence 0.0-1.0",
    )
    confidence_level: ConfidenceLevel = Field(
        ..., description="Categorical confidence level",
    )
    match_method: str = Field(..., description="Match method used")


class EntityResolutionResult(BaseModel):
    """Result of batch entity resolution.

    Attributes:
        matches: Successfully resolved entities.
        unresolved: Entity names that could not be resolved.
    """
    matches: List[EntityMatch] = Field(
        default_factory=list, description="Resolved entity matches",
    )
    unresolved: List[str] = Field(
        default_factory=list, description="Unresolved entity names",
    )


# =============================================================================
# DIMENSIONAL INFO MODELS
# =============================================================================


class DimensionInfo(BaseModel):
    """Information about a physical dimension.

    Attributes:
        dimension: The dimension enum value.
        base_unit: Base unit for this dimension.
        supported_units: List of all supported unit names.
    """
    dimension: UnitDimension = Field(..., description="Physical dimension")
    base_unit: str = Field(..., description="Base unit symbol")
    supported_units: List[str] = Field(
        default_factory=list, description="Supported unit names",
    )


class UnitInfo(BaseModel):
    """Information about a specific unit.

    Attributes:
        symbol: Unit symbol.
        dimension: Physical dimension.
        to_base_factor: Factor to convert to dimension base unit.
        base_unit: Base unit for this dimension.
    """
    symbol: str = Field(..., description="Unit symbol")
    dimension: UnitDimension = Field(..., description="Physical dimension")
    to_base_factor: str = Field(..., description="Factor to convert to base unit")
    base_unit: str = Field(..., description="Base unit for dimension")


class GWPInfo(BaseModel):
    """GWP information for a greenhouse gas.

    Attributes:
        gas: Greenhouse gas type.
        gwp_100: 100-year GWP value.
        gwp_20: 20-year GWP value.
        source: Source assessment report.
        version: IPCC assessment report version.
    """
    gas: GHGGas = Field(..., description="Greenhouse gas type")
    gwp_100: Decimal = Field(..., description="100-year GWP value")
    gwp_20: Decimal = Field(..., description="20-year GWP value")
    source: str = Field(..., description="Source reference")
    version: GWPVersion = Field(..., description="IPCC AR version")

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# PROVENANCE MODELS
# =============================================================================


class ConversionProvenance(BaseModel):
    """Provenance record for a conversion or resolution operation.

    Attributes:
        operation_id: Unique identifier for this operation.
        timestamp: When the operation was performed.
        input_hash: SHA-256 hash of the input data.
        output_hash: SHA-256 hash of the output data.
        chain_hash: Cumulative chain hash linking operations.
        factors_used: Conversion factors or match details.
        version: Normalizer SDK version.
    """
    operation_id: str = Field(..., description="Unique operation identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Operation timestamp",
    )
    input_hash: str = Field(..., description="SHA-256 hash of input")
    output_hash: str = Field(..., description="SHA-256 hash of output")
    chain_hash: str = Field(..., description="Cumulative chain hash")
    factors_used: Dict[str, Any] = Field(
        default_factory=dict, description="Factors or match details",
    )
    version: str = Field(default="1.0.0", description="Normalizer SDK version")


__all__ = [
    # Enumerations
    "UnitDimension",
    "GHGGas",
    "GWPVersion",
    "ConfidenceLevel",
    # Conversion models
    "ConversionResult",
    "BatchConversionResult",
    # Entity resolution models
    "EntityMatch",
    "EntityResolutionResult",
    # Dimensional info models
    "DimensionInfo",
    "UnitInfo",
    "GWPInfo",
    # Provenance models
    "ConversionProvenance",
]
