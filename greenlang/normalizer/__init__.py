# -*- coding: utf-8 -*-
"""
GL-FOUND-X-003: GreenLang Normalizer SDK
=========================================

This package provides the unit conversion, entity resolution, and
dimensional analysis SDK for the GreenLang framework. It supports:

- Deterministic unit conversion across 7+ physical dimensions
- GHG gas conversion with IPCC AR5/AR6 GWP factors
- Fuel, material, and process name standardisation
- Dimensional compatibility checking
- SHA-256 provenance tracking for audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 15 endpoints

Key Components:
    - converter: UnitConverter with Decimal-precision arithmetic
    - entity_resolver: EntityResolver with Levenshtein fuzzy matching
    - dimensional: DimensionalAnalyzer for compatibility checks
    - provenance: ConversionProvenanceTracker for audit trails
    - config: NormalizerConfig with GL_NORMALIZER_ env prefix
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: NormalizerService facade

Example:
    >>> from greenlang.normalizer import UnitConverter
    >>> c = UnitConverter()
    >>> r = c.convert(100, "kWh", "MWh")
    >>> print(r.converted_value)  # Decimal('0.1')

    >>> from greenlang.normalizer import EntityResolver
    >>> r = EntityResolver()
    >>> m = r.resolve_fuel("natural gas")
    >>> print(m.canonical_name)  # "Natural Gas"

Agent ID: GL-FOUND-X-003
Agent Name: Unit & Reference Normalizer
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-003"
__agent_name__ = "Unit & Reference Normalizer"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.normalizer.config import (
    NormalizerConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, results, provenance)
# ---------------------------------------------------------------------------
from greenlang.normalizer.models import (
    # Enumerations
    UnitDimension,
    GHGGas,
    GWPVersion,
    ConfidenceLevel,
    # Conversion models
    ConversionResult,
    BatchConversionResult,
    # Entity resolution models
    EntityMatch,
    EntityResolutionResult,
    # Dimensional info models
    DimensionInfo,
    UnitInfo,
    GWPInfo,
    # Provenance models
    ConversionProvenance,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.normalizer.converter import (
    UnitConverter,
    MASS_UNITS,
    ENERGY_UNITS,
    VOLUME_UNITS,
    AREA_UNITS,
    DISTANCE_UNITS,
    EMISSIONS_UNITS,
    TIME_UNITS,
    DIMENSION_UNITS,
    BASE_UNITS,
    GWP_AR6_100,
    GWP_AR6_20,
    GWP_AR5_100,
    GWP_AR5_20,
    GWP_TABLES,
)

from greenlang.normalizer.entity_resolver import (
    EntityResolver,
    FUEL_STANDARDIZATION,
    MATERIAL_STANDARDIZATION,
    PROCESS_STANDARDIZATION,
    ENTITY_VOCABULARIES,
)

from greenlang.normalizer.dimensional import DimensionalAnalyzer

from greenlang.normalizer.provenance import ConversionProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.normalizer.metrics import (
    PROMETHEUS_AVAILABLE,
    record_conversion,
    record_entity_resolution,
    record_dimension_error,
    record_gwp_conversion,
    record_batch,
    update_vocabulary_entries,
    record_cache_hit,
    record_cache_miss,
    update_active_conversions,
    update_custom_factors,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.normalizer.setup import (
    NormalizerService,
    configure_normalizer_service,
    get_normalizer_service,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Configuration
    "NormalizerConfig",
    "get_config",
    "set_config",
    "reset_config",
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
    # Core engines
    "UnitConverter",
    "EntityResolver",
    "DimensionalAnalyzer",
    "ConversionProvenanceTracker",
    # Unit tables
    "MASS_UNITS",
    "ENERGY_UNITS",
    "VOLUME_UNITS",
    "AREA_UNITS",
    "DISTANCE_UNITS",
    "EMISSIONS_UNITS",
    "TIME_UNITS",
    "DIMENSION_UNITS",
    "BASE_UNITS",
    # GWP tables
    "GWP_AR6_100",
    "GWP_AR6_20",
    "GWP_AR5_100",
    "GWP_AR5_20",
    "GWP_TABLES",
    # Entity vocabularies
    "FUEL_STANDARDIZATION",
    "MATERIAL_STANDARDIZATION",
    "PROCESS_STANDARDIZATION",
    "ENTITY_VOCABULARIES",
    # Metrics
    "PROMETHEUS_AVAILABLE",
    "record_conversion",
    "record_entity_resolution",
    "record_dimension_error",
    "record_gwp_conversion",
    "record_batch",
    "update_vocabulary_entries",
    "record_cache_hit",
    "record_cache_miss",
    "update_active_conversions",
    "update_custom_factors",
    # Service setup facade
    "NormalizerService",
    "configure_normalizer_service",
    "get_normalizer_service",
]
