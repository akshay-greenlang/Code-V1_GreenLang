# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-005: Land Use Change Detector Agent

Provides built-in reference datasets for land use change detection:
    - land_use_parameters: IPCC land use class definitions, EUDR transition sets
    - spectral_signatures: Reference spectral data per land use type and biome
    - transition_rules: Regulatory classification rules and verdict determination

These datasets enable deterministic, zero-hallucination land use classification
and EUDR compliance assessment without external API dependencies. All data is
version-tracked and provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent
"""

from greenlang.agents.eudr.land_use_change.reference_data.land_use_parameters import (
    COMMODITY_LAND_USE_MAP,
    EUDR_COMMODITIES,
    EUDR_CUTOFF_DATE,
    EUDR_DEFORESTATION_TRANSITIONS,
    EUDR_DEGRADATION_TRANSITIONS,
    EUDR_EXCLUDED_TRANSITIONS,
    FAO_FOREST_DEFINITION,
    LAND_USE_CLASSES,
    LandUseCategory,
    get_all_agricultural_classes,
    get_all_eudr_relevant_classes,
    get_all_forest_classes,
    get_commodity_land_use,
    get_land_use_params,
    get_spectral_thresholds,
    is_agricultural_class,
    is_deforestation_transition,
    is_degradation_transition,
    is_eudr_relevant,
    is_forest_class,
)
from greenlang.agents.eudr.land_use_change.reference_data.spectral_signatures import (
    BIOME_ADJUSTED_SIGNATURES,
    COMMODITY_SPECTRAL_PROFILES,
    SPECTRAL_SIGNATURES,
    classify_by_spectral_distance,
    compute_spectral_distance,
    get_all_biome_names,
    get_all_commodity_names,
    get_biome_adjustment,
    get_commodity_profile,
    get_spectral_signature,
)
from greenlang.agents.eudr.land_use_change.reference_data.transition_rules import (
    COMMODITY_CONVERSION_RULES,
    TRANSITION_CLASSIFICATION_RULES,
    TRANSITION_SEVERITY,
    VERDICT_DETERMINATION_RULES,
    ComplianceVerdict,
    TransitionType,
    classify_transition,
    determine_verdict,
    get_all_deforestation_transitions,
    get_all_non_compliant_verdicts,
    get_commodity_deforestation_transitions,
    get_eudr_article,
    get_transition_severity,
    is_commodity_deforestation,
    is_deforestation,
    is_degradation,
)

__all__ = [
    # land_use_parameters
    "LandUseCategory",
    "LAND_USE_CLASSES",
    "EUDR_DEFORESTATION_TRANSITIONS",
    "EUDR_DEGRADATION_TRANSITIONS",
    "EUDR_EXCLUDED_TRANSITIONS",
    "COMMODITY_LAND_USE_MAP",
    "FAO_FOREST_DEFINITION",
    "EUDR_CUTOFF_DATE",
    "EUDR_COMMODITIES",
    "get_land_use_params",
    "is_forest_class",
    "is_agricultural_class",
    "is_eudr_relevant",
    "is_deforestation_transition",
    "is_degradation_transition",
    "get_commodity_land_use",
    "get_spectral_thresholds",
    "get_all_forest_classes",
    "get_all_agricultural_classes",
    "get_all_eudr_relevant_classes",
    # spectral_signatures
    "SPECTRAL_SIGNATURES",
    "BIOME_ADJUSTED_SIGNATURES",
    "COMMODITY_SPECTRAL_PROFILES",
    "get_spectral_signature",
    "compute_spectral_distance",
    "classify_by_spectral_distance",
    "get_commodity_profile",
    "get_biome_adjustment",
    "get_all_biome_names",
    "get_all_commodity_names",
    # transition_rules
    "TransitionType",
    "ComplianceVerdict",
    "TRANSITION_CLASSIFICATION_RULES",
    "TRANSITION_SEVERITY",
    "VERDICT_DETERMINATION_RULES",
    "COMMODITY_CONVERSION_RULES",
    "classify_transition",
    "is_deforestation",
    "is_degradation",
    "determine_verdict",
    "get_eudr_article",
    "get_transition_severity",
    "get_commodity_deforestation_transitions",
    "is_commodity_deforestation",
    "get_all_deforestation_transitions",
    "get_all_non_compliant_verdicts",
]
