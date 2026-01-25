# -*- coding: utf-8 -*-
"""
Industry Mappings Service for GL-VCCI Scope 3 Carbon Platform

This package provides comprehensive industry classification and mapping services
including NAICS, ISIC, and custom product taxonomies for accurate emission
factor matching.

Author: GL-VCCI Platform Team
Phase: Phase 2, Week 5 - Industry Mappings
"""

from typing import Dict, List, Optional, Tuple

from .models import (
    NAICSCode,
    ISICCode,
    ProductMapping,
    TaxonomyEntry,
    MappingResult,
    IndustryCategory,
    CodeHierarchy,
    ValidationResult
)

from .naics import (
    NAICSDatabase,
    search_naics,
    get_naics_hierarchy,
    validate_naics_code
)

from .isic import (
    ISICDatabase,
    search_isic,
    get_isic_hierarchy,
    validate_isic_code,
    naics_to_isic
)

from .custom_taxonomy import (
    CustomTaxonomy,
    get_product_category,
    search_products,
    get_emission_factor_link
)

from .mapper import (
    IndustryMapper,
    MappingStrategy,
    MatchingEngine
)

from .validation import (
    MappingValidator,
    validate_mapping,
    check_coverage,
    analyze_mapping_quality
)

from .config import (
    IndustryMappingConfig,
    get_default_config,
    MatchThresholds
)

__version__ = "1.0.0"
__author__ = "GL-VCCI Platform Team"
__phase__ = "Phase 2, Week 5"

__all__ = [
    # Models
    "NAICSCode",
    "ISICCode",
    "ProductMapping",
    "TaxonomyEntry",
    "MappingResult",
    "IndustryCategory",
    "CodeHierarchy",
    "ValidationResult",

    # NAICS
    "NAICSDatabase",
    "search_naics",
    "get_naics_hierarchy",
    "validate_naics_code",

    # ISIC
    "ISICDatabase",
    "search_isic",
    "get_isic_hierarchy",
    "validate_isic_code",
    "naics_to_isic",

    # Custom Taxonomy
    "CustomTaxonomy",
    "get_product_category",
    "search_products",
    "get_emission_factor_link",

    # Mapper
    "IndustryMapper",
    "MappingStrategy",
    "MatchingEngine",

    # Validation
    "MappingValidator",
    "validate_mapping",
    "check_coverage",
    "analyze_mapping_quality",

    # Config
    "IndustryMappingConfig",
    "get_default_config",
    "MatchThresholds",
]
