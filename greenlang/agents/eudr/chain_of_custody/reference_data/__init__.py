# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-009: Chain of Custody Agent

Provides built-in reference datasets for chain of custody operations:
    - conversion_factors: Commodity yield ratios, loss tolerances, by-products
    - document_requirements: Required/optional documents per event type
    - coc_model_rules: ISO 22095 model rules, hierarchy, handoff validation

These datasets enable deterministic, zero-hallucination mass balance
verification, document completeness checks, and CoC model compliance
validation without external API dependencies. All data is version-tracked
and provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009 (Chain of Custody)
Agent ID: GL-EUDR-COC-009
"""

from greenlang.agents.eudr.chain_of_custody.reference_data.conversion_factors import (
    CONVERSION_FACTORS,
    TOTAL_COMMODITIES,
    TOTAL_CONVERSION_FACTORS,
    get_all_processes_for_commodity,
    get_by_products,
    get_commodities,
    get_conversion_factor,
    get_expected_yield,
    get_loss_tolerance,
    get_mass_balance_check,
    validate_yield,
)
from greenlang.agents.eudr.chain_of_custody.reference_data.document_requirements import (
    DOCUMENT_REQUIREMENTS,
    DOCUMENT_TYPES,
    TOTAL_DOCUMENT_TYPES,
    TOTAL_EVENT_TYPES,
    get_all_document_types,
    get_document_type_info,
    get_event_types,
    get_optional_documents,
    get_required_documents,
    is_document_required,
    validate_document_completeness,
)
from greenlang.agents.eudr.chain_of_custody.reference_data.coc_model_rules import (
    COC_MODEL_RULES,
    CREDIT_PERIOD_BY_CERTIFICATION,
    CROSS_MODEL_HANDOFF_RULES,
    MODEL_HIERARCHY,
    TOTAL_MODELS,
    can_upgrade_model,
    get_all_models,
    get_applicable_certifications,
    get_credit_period,
    get_downgrade_options,
    get_model_hierarchy_level,
    get_model_rules,
    get_model_validation_rules,
    validate_handoff,
    validate_model_compliance,
)

__all__ = [
    # conversion_factors
    "CONVERSION_FACTORS",
    "TOTAL_COMMODITIES",
    "TOTAL_CONVERSION_FACTORS",
    "get_conversion_factor",
    "get_expected_yield",
    "get_loss_tolerance",
    "validate_yield",
    "get_by_products",
    "get_all_processes_for_commodity",
    "get_commodities",
    "get_mass_balance_check",
    # document_requirements
    "DOCUMENT_REQUIREMENTS",
    "DOCUMENT_TYPES",
    "TOTAL_DOCUMENT_TYPES",
    "TOTAL_EVENT_TYPES",
    "get_required_documents",
    "get_optional_documents",
    "is_document_required",
    "get_document_type_info",
    "get_all_document_types",
    "validate_document_completeness",
    "get_event_types",
    # coc_model_rules
    "COC_MODEL_RULES",
    "CREDIT_PERIOD_BY_CERTIFICATION",
    "CROSS_MODEL_HANDOFF_RULES",
    "MODEL_HIERARCHY",
    "TOTAL_MODELS",
    "get_model_rules",
    "get_credit_period",
    "validate_handoff",
    "get_applicable_certifications",
    "get_model_hierarchy_level",
    "can_upgrade_model",
    "get_model_validation_rules",
    "get_all_models",
    "get_downgrade_options",
    "validate_model_compliance",
]
