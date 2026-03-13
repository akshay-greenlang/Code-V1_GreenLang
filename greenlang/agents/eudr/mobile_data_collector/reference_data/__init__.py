# -*- coding: utf-8 -*-
"""
Reference Data Package for Mobile Data Collector - AGENT-EUDR-015

Provides pre-loaded reference data for EUDR mobile data collection:

    - eudr_form_templates: Six built-in EUDR form template definitions
      (producer registration, plot survey, harvest log, custody transfer,
      quality inspection, smallholder declaration)
    - commodity_specifications: EUDR commodity specs for the 7 regulated
      commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood)
      with HS codes, quality parameters, and seasonal patterns
    - language_packs: Multi-language translations for form UI elements
      covering 24 EU official languages plus 6 local languages

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

# ---- EUDR form templates ----
from greenlang.agents.eudr.mobile_data_collector.reference_data.eudr_form_templates import (
    PRODUCER_REGISTRATION_TEMPLATE,
    PLOT_SURVEY_TEMPLATE,
    HARVEST_LOG_TEMPLATE,
    CUSTODY_TRANSFER_TEMPLATE,
    QUALITY_INSPECTION_TEMPLATE,
    SMALLHOLDER_DECLARATION_TEMPLATE,
    ALL_TEMPLATES,
    TEMPLATE_REGISTRY,
    get_template,
    list_template_names,
    get_template_fields,
    get_required_fields,
    validate_template_data,
)

# ---- Commodity specifications ----
from greenlang.agents.eudr.mobile_data_collector.reference_data.commodity_specifications import (
    CATTLE_SPECIFICATION,
    COCOA_SPECIFICATION,
    COFFEE_SPECIFICATION,
    OIL_PALM_SPECIFICATION,
    RUBBER_SPECIFICATION,
    SOYA_SPECIFICATION,
    WOOD_SPECIFICATION,
    ALL_COMMODITIES,
    VALID_COMMODITY_CODES,
    get_commodity,
    is_valid_commodity,
    get_hs_codes,
    get_derived_products,
    get_quality_parameters,
    get_commodity_unit,
    lookup_commodity_by_hs,
)

# ---- Language packs ----
from greenlang.agents.eudr.mobile_data_collector.reference_data.language_packs import (
    COMMON_LABELS,
    EUDR_TERMS,
    BUTTON_LABELS,
    STATUS_MESSAGES,
    VALIDATION_MESSAGES,
    get_label,
    get_all_labels_for_language,
    list_supported_languages,
)


__all__ = [
    # Form templates
    "PRODUCER_REGISTRATION_TEMPLATE",
    "PLOT_SURVEY_TEMPLATE",
    "HARVEST_LOG_TEMPLATE",
    "CUSTODY_TRANSFER_TEMPLATE",
    "QUALITY_INSPECTION_TEMPLATE",
    "SMALLHOLDER_DECLARATION_TEMPLATE",
    "ALL_TEMPLATES",
    "TEMPLATE_REGISTRY",
    "get_template",
    "list_template_names",
    "get_template_fields",
    "get_required_fields",
    "validate_template_data",
    # Commodity specifications
    "CATTLE_SPECIFICATION",
    "COCOA_SPECIFICATION",
    "COFFEE_SPECIFICATION",
    "OIL_PALM_SPECIFICATION",
    "RUBBER_SPECIFICATION",
    "SOYA_SPECIFICATION",
    "WOOD_SPECIFICATION",
    "ALL_COMMODITIES",
    "VALID_COMMODITY_CODES",
    "get_commodity",
    "is_valid_commodity",
    "get_hs_codes",
    "get_derived_products",
    "get_quality_parameters",
    "get_commodity_unit",
    "lookup_commodity_by_hs",
    # Language packs
    "COMMON_LABELS",
    "EUDR_TERMS",
    "BUTTON_LABELS",
    "STATUS_MESSAGES",
    "VALIDATION_MESSAGES",
    "get_label",
    "get_all_labels_for_language",
    "list_supported_languages",
]
