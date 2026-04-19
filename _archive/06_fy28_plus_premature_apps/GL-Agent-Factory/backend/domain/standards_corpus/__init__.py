# -*- coding: utf-8 -*-
"""
GreenLang Standards Corpus
==========================

Comprehensive indexed standards documentation for regulatory compliance
covering ASME, API, NFPA, IEC, ISO, and EPA regulations.

This module provides:
- Central registry of all standards with metadata
- Section-level indexing and cross-references
- Full-text search across standards corpus
- Equipment-to-standard applicability mappings
- Formula and calculation method lookups
"""

from .standards_registry import (
    StandardsRegistry,
    Standard,
    StandardSection,
    StandardFormula,
    CrossReference,
    EquipmentMapping,
    ComplianceRequirement,
    get_standards_registry,
)

__all__ = [
    "StandardsRegistry",
    "Standard",
    "StandardSection",
    "StandardFormula",
    "CrossReference",
    "EquipmentMapping",
    "ComplianceRequirement",
    "get_standards_registry",
]

__version__ = "1.0.0"
