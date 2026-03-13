# -*- coding: utf-8 -*-
"""
Reference Data Package for QR Code Generator - AGENT-EUDR-014

Provides pre-loaded reference data for QR code generation, label
rendering, GS1 Digital Link construction, and EUDR commodity
identification:

    - label_templates: Five pre-defined label template definitions
      (product, shipping, pallet, container, consumer)
    - gs1_specifications: GS1 Digital Link formatting, GTIN validation,
      and Application Identifier definitions
    - commodity_codes: EUDR commodity codes, HS/CN/TARIC mappings,
      derived product registry, and country risk classification

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
Status: Production Ready
"""

from __future__ import annotations

# ---- Label templates ----
from greenlang.agents.eudr.qr_code_generator.reference_data.label_templates import (
    PRODUCT_LABEL_TEMPLATE,
    SHIPPING_LABEL_TEMPLATE,
    PALLET_LABEL_TEMPLATE,
    CONTAINER_LABEL_TEMPLATE,
    CONSUMER_LABEL_TEMPLATE,
    ALL_TEMPLATES,
    TEMPLATE_REGISTRY,
    validate_template,
    get_template,
    list_template_names,
    get_template_dimensions,
)

# ---- GS1 specifications ----
from greenlang.agents.eudr.qr_code_generator.reference_data.gs1_specifications import (
    GS1_DIGITAL_LINK_BASE_URL,
    GS1_APPLICATION_IDENTIFIERS,
    AI_DESCRIPTIONS,
    EUDR_RELEVANT_AIS,
    GS1_URI_SYNTAX_REGEX,
    calculate_gtin_check_digit,
    validate_gtin,
    normalize_to_gtin14,
    build_gs1_digital_link_uri,
    parse_gs1_digital_link_uri,
    validate_gs1_digital_link_uri,
    get_ai_description,
    get_eudr_relevant_ais,
)

# ---- Commodity codes ----
from greenlang.agents.eudr.qr_code_generator.reference_data.commodity_codes import (
    EUDR_COMMODITIES,
    COMMODITY_CODE_PREFIX,
    HS_CODE_RANGES,
    DERIVED_PRODUCTS,
    COUNTRY_RISK_CLASSIFICATION,
    VALID_COMMODITY_NAMES,
    is_eudr_commodity,
    get_commodity_from_hs,
    get_commodity_prefix,
    get_country_risk,
    get_hs_codes_for_commodity,
    validate_commodity,
)


__all__ = [
    # Label templates
    "PRODUCT_LABEL_TEMPLATE",
    "SHIPPING_LABEL_TEMPLATE",
    "PALLET_LABEL_TEMPLATE",
    "CONTAINER_LABEL_TEMPLATE",
    "CONSUMER_LABEL_TEMPLATE",
    "ALL_TEMPLATES",
    "TEMPLATE_REGISTRY",
    "validate_template",
    "get_template",
    "list_template_names",
    "get_template_dimensions",
    # GS1 specifications
    "GS1_DIGITAL_LINK_BASE_URL",
    "GS1_APPLICATION_IDENTIFIERS",
    "AI_DESCRIPTIONS",
    "EUDR_RELEVANT_AIS",
    "GS1_URI_SYNTAX_REGEX",
    "calculate_gtin_check_digit",
    "validate_gtin",
    "normalize_to_gtin14",
    "build_gs1_digital_link_uri",
    "parse_gs1_digital_link_uri",
    "validate_gs1_digital_link_uri",
    "get_ai_description",
    "get_eudr_relevant_ais",
    # Commodity codes
    "EUDR_COMMODITIES",
    "COMMODITY_CODE_PREFIX",
    "HS_CODE_RANGES",
    "DERIVED_PRODUCTS",
    "COUNTRY_RISK_CLASSIFICATION",
    "VALID_COMMODITY_NAMES",
    "is_eudr_commodity",
    "get_commodity_from_hs",
    "get_commodity_prefix",
    "get_country_risk",
    "get_hs_codes_for_commodity",
    "validate_commodity",
]
