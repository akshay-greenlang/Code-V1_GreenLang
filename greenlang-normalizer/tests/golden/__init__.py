"""
Golden Test Suite for GL-FOUND-X-003: GreenLang Unit & Reference Normalizer.

This package provides comprehensive golden file testing for the GreenLang
normalizer, ensuring deterministic, audit-ready normalization of units,
conversions, and reference data.

Test Categories:
    - Unit Conversions: All GL Canonical Unit conversions (energy, mass, volume, etc.)
    - Entity Resolution: Fuel, material, and process matching
    - Full Pipeline: End-to-end scenarios for GHG Protocol and EU CSRD

Test Features:
    - Automatic discovery of golden files from YAML
    - Tolerance-based comparison for floating point
    - Clear failure messages with diff
    - Markers for slow tests
    - Cross-validation with Pint results
    - Provenance hash verification

Version: 1.0.0
Last Updated: 2026-01-30
"""

__version__ = "1.0.0"
__author__ = "GreenLang Foundation Team"

# Test categories
TEST_CATEGORIES = [
    "unit_conversions",
    "entity_resolution",
    "full_pipeline",
]

# Dimension categories for unit conversions
DIMENSION_CATEGORIES = [
    "energy",
    "mass",
    "volume",
    "emissions",
    "pressure",
    "temperature",
]
