# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-010: Segregation Verifier Agent

Provides built-in reference datasets for segregation verification:
    - segregation_standards: Per-certification-scheme segregation rules,
      commodity requirements, risk levels, barrier hierarchies
    - cleaning_protocols: Transport cleaning and processing changeover
      protocols, verification methods, cleaning agent compatibility
    - labeling_requirements: Label content rules, color codes, placement
      rules, durability specs, and size specifications

These datasets enable deterministic, zero-hallucination verification
of physical segregation compliance without external API dependencies.
All data is version-tracked and provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Segregation Verifier)
Agent ID: GL-EUDR-SGV-010
"""

from greenlang.agents.eudr.segregation_verifier.reference_data.segregation_standards import (
    COMMODITY_SEGREGATION_REQUIREMENTS,
    MINIMUM_BARRIER_TYPES,
    RISK_LEVEL_MAPPING,
    SEGREGATION_STANDARDS,
    TOTAL_BARRIER_TYPES,
    TOTAL_COMMODITIES,
    TOTAL_STANDARDS,
    get_all_standards,
    get_audit_frequency,
    get_barrier_effectiveness,
    get_barrier_rank,
    get_commodity_requirements,
    get_risk_level,
    get_standard,
    is_dedicated_required,
    meets_minimum_barrier,
)
from greenlang.agents.eudr.segregation_verifier.reference_data.cleaning_protocols import (
    CLEANING_AGENT_COMPATIBILITY,
    CLEANING_PROTOCOLS,
    PROCESSING_CHANGEOVER_PROTOCOLS,
    TOTAL_CLEANING_AGENTS,
    TOTAL_LINE_TYPES,
    TOTAL_TRANSPORT_TYPES,
    TOTAL_VERIFICATION_METHODS,
    VERIFICATION_METHODS,
    get_all_line_types,
    get_all_transport_types,
    get_changeover_protocol,
    get_cleaning_protocol,
    get_first_run_discard_kg,
    get_residue_tolerance,
    get_verification_method,
    is_changeover_sufficient,
    is_cleaning_sufficient,
)
from greenlang.agents.eudr.segregation_verifier.reference_data.labeling_requirements import (
    COLOR_CODE_STANDARD,
    LABEL_CONTENT_REQUIREMENTS,
    LABEL_DURABILITY_REQUIREMENTS,
    LABEL_PLACEMENT_RULES,
    LABEL_SIZE_SPECIFICATIONS,
    TOTAL_COLORS,
    TOTAL_DURABILITY_ENVIRONMENTS,
    TOTAL_LABEL_TYPES,
    TOTAL_PLACEMENT_DOMAINS,
    get_all_label_types,
    get_color_hex,
    get_color_meaning,
    get_label_requirements,
    get_label_size,
    get_min_label_count,
    get_optional_fields,
    get_placement_rules,
    get_required_fields,
    is_field_required,
    validate_label_completeness,
)

__all__ = [
    # -- segregation_standards --
    "SEGREGATION_STANDARDS",
    "COMMODITY_SEGREGATION_REQUIREMENTS",
    "RISK_LEVEL_MAPPING",
    "MINIMUM_BARRIER_TYPES",
    "TOTAL_STANDARDS",
    "TOTAL_COMMODITIES",
    "TOTAL_BARRIER_TYPES",
    "get_standard",
    "get_commodity_requirements",
    "get_risk_level",
    "get_barrier_effectiveness",
    "get_all_standards",
    "get_audit_frequency",
    "is_dedicated_required",
    "get_barrier_rank",
    "meets_minimum_barrier",
    # -- cleaning_protocols --
    "CLEANING_PROTOCOLS",
    "PROCESSING_CHANGEOVER_PROTOCOLS",
    "VERIFICATION_METHODS",
    "CLEANING_AGENT_COMPATIBILITY",
    "TOTAL_TRANSPORT_TYPES",
    "TOTAL_LINE_TYPES",
    "TOTAL_VERIFICATION_METHODS",
    "TOTAL_CLEANING_AGENTS",
    "get_cleaning_protocol",
    "get_changeover_protocol",
    "get_verification_method",
    "is_cleaning_sufficient",
    "is_changeover_sufficient",
    "get_all_transport_types",
    "get_all_line_types",
    "get_first_run_discard_kg",
    "get_residue_tolerance",
    # -- labeling_requirements --
    "LABEL_CONTENT_REQUIREMENTS",
    "COLOR_CODE_STANDARD",
    "LABEL_PLACEMENT_RULES",
    "LABEL_DURABILITY_REQUIREMENTS",
    "LABEL_SIZE_SPECIFICATIONS",
    "TOTAL_LABEL_TYPES",
    "TOTAL_COLORS",
    "TOTAL_PLACEMENT_DOMAINS",
    "TOTAL_DURABILITY_ENVIRONMENTS",
    "get_label_requirements",
    "get_placement_rules",
    "get_color_meaning",
    "get_required_fields",
    "get_optional_fields",
    "is_field_required",
    "validate_label_completeness",
    "get_min_label_count",
    "get_all_label_types",
    "get_color_hex",
    "get_label_size",
]
