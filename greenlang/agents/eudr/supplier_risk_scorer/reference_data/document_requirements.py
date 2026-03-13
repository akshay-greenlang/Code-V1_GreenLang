# -*- coding: utf-8 -*-
"""
Document Requirements Database - AGENT-EUDR-017 Supplier Risk Scorer

Comprehensive EUDR document requirements reference data for supplier
assessment. Provides per-commodity required document types, document
templates, validation rules, expiry policies, and language requirements
for deterministic documentation analysis per EU 2023/1115.

Data includes:
    - EUDR_REQUIRED_DOCUMENTS: per-commodity list of required document types
      (geolocation, DDS reference, product description, quantity declaration,
      harvest date, compliance declaration, certificate, trade license,
      phytosanitary) with descriptions
    - DOCUMENT_TEMPLATES: template metadata for each document type
    - VALIDATION_RULES: per-document validation rules (required fields, formats, ranges)
    - EXPIRY_POLICIES: expiry periods for each document type
    - LANGUAGE_REQUIREMENTS: accepted languages per EU member state

Data Sources:
    - EU Regulation 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11
    - European Commission EUDR Implementing Regulation (EU) 2023/2464
    - EC Information System for Due Diligence Statements Guidance 2024
    - National Competent Authority Guidelines (EU-27)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DocumentRequirement = Dict[str, Any]
DocumentTemplate = Dict[str, Any]
ValidationRule = Dict[str, Any]
ExpiryPolicy = Dict[str, Any]
LanguageRequirement = Dict[str, List[str]]

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "EU Regulation 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11",
    "European Commission EUDR Implementing Regulation (EU) 2023/2464",
    "EC Information System for Due Diligence Statements Guidance 2024",
    "National Competent Authority Guidelines (EU-27 Member States)",
]

# ---------------------------------------------------------------------------
# EUDR commodities
# ---------------------------------------------------------------------------

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ===========================================================================
# EUDR Required Documents (per commodity)
# ===========================================================================
#
# Article 9 Due Diligence: Operators must collect information on:
#   (a) geolocation coordinates
#   (b) country of production
#   (c) product description and quantity
#   (d) reference to DDS in EU system
#   (e) information on supplier/trader

EUDR_REQUIRED_DOCUMENTS: Dict[str, List[DocumentRequirement]] = {
    "cattle": [
        {
            "type": "geolocation",
            "description": "Geolocation coordinates of production plots (latitude/longitude)",
            "mandatory": True,
            "article_reference": "Article 9(1)(a)",
        },
        {
            "type": "dds_reference",
            "description": "Reference number of Due Diligence Statement in EU system",
            "mandatory": True,
            "article_reference": "Article 9(1)(d)",
        },
        {
            "type": "product_description",
            "description": "Product description with CN code, commodity type, quantity",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "quantity_declaration",
            "description": "Quantity in net mass (kg) or volume (m³)",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "harvest_date",
            "description": "Date or period of production (after 2020-12-31)",
            "mandatory": True,
            "article_reference": "Article 2(1) cutoff date",
        },
        {
            "type": "compliance_declaration",
            "description": "Declaration that product is deforestation-free and legally produced",
            "mandatory": True,
            "article_reference": "Article 9(1)(f)",
        },
        {
            "type": "supplier_information",
            "description": "Supplier name, address, contact, tax ID",
            "mandatory": True,
            "article_reference": "Article 9(1)(e)",
        },
        {
            "type": "trade_license",
            "description": "Trade license or business registration",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
    ],
    "cocoa": [
        {
            "type": "geolocation",
            "description": "Geolocation coordinates of cocoa farms (latitude/longitude)",
            "mandatory": True,
            "article_reference": "Article 9(1)(a)",
        },
        {
            "type": "dds_reference",
            "description": "Reference number of Due Diligence Statement in EU system",
            "mandatory": True,
            "article_reference": "Article 9(1)(d)",
        },
        {
            "type": "product_description",
            "description": "Product description with CN code, cocoa type, processing level",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "quantity_declaration",
            "description": "Quantity in net mass (kg)",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "harvest_date",
            "description": "Harvest season/date (after 2020-12-31)",
            "mandatory": True,
            "article_reference": "Article 2(1) cutoff date",
        },
        {
            "type": "compliance_declaration",
            "description": "Declaration that cocoa is deforestation-free and legally produced",
            "mandatory": True,
            "article_reference": "Article 9(1)(f)",
        },
        {
            "type": "supplier_information",
            "description": "Supplier/cooperative name, address, contact",
            "mandatory": True,
            "article_reference": "Article 9(1)(e)",
        },
        {
            "type": "certificate",
            "description": "Certification (Rainforest Alliance, UTZ, Organic, Fair Trade)",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
    ],
    "coffee": [
        {
            "type": "geolocation",
            "description": "Geolocation coordinates of coffee farms (latitude/longitude)",
            "mandatory": True,
            "article_reference": "Article 9(1)(a)",
        },
        {
            "type": "dds_reference",
            "description": "Reference number of Due Diligence Statement in EU system",
            "mandatory": True,
            "article_reference": "Article 9(1)(d)",
        },
        {
            "type": "product_description",
            "description": "Product description with CN code, coffee variety, processing",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "quantity_declaration",
            "description": "Quantity in net mass (kg)",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "harvest_date",
            "description": "Harvest season/date (after 2020-12-31)",
            "mandatory": True,
            "article_reference": "Article 2(1) cutoff date",
        },
        {
            "type": "compliance_declaration",
            "description": "Declaration that coffee is deforestation-free and legally produced",
            "mandatory": True,
            "article_reference": "Article 9(1)(f)",
        },
        {
            "type": "supplier_information",
            "description": "Supplier/cooperative name, address, contact",
            "mandatory": True,
            "article_reference": "Article 9(1)(e)",
        },
        {
            "type": "certificate",
            "description": "Certification (Rainforest Alliance, UTZ, Organic, Fair Trade)",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
    ],
    "oil_palm": [
        {
            "type": "geolocation",
            "description": "Geolocation coordinates of palm oil plantations (latitude/longitude)",
            "mandatory": True,
            "article_reference": "Article 9(1)(a)",
        },
        {
            "type": "dds_reference",
            "description": "Reference number of Due Diligence Statement in EU system",
            "mandatory": True,
            "article_reference": "Article 9(1)(d)",
        },
        {
            "type": "product_description",
            "description": "Product description with CN code, crude/refined, palm kernel oil",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "quantity_declaration",
            "description": "Quantity in net mass (kg) or volume (liters)",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "harvest_date",
            "description": "Date or period of production (after 2020-12-31)",
            "mandatory": True,
            "article_reference": "Article 2(1) cutoff date",
        },
        {
            "type": "compliance_declaration",
            "description": "Declaration that palm oil is deforestation-free and legally produced",
            "mandatory": True,
            "article_reference": "Article 9(1)(f)",
        },
        {
            "type": "supplier_information",
            "description": "Supplier/mill name, address, contact, RSPO ID",
            "mandatory": True,
            "article_reference": "Article 9(1)(e)",
        },
        {
            "type": "certificate",
            "description": "Certification (RSPO, ISCC, MSPO)",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
        {
            "type": "hcv_assessment",
            "description": "High Conservation Value assessment (if applicable)",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
    ],
    "rubber": [
        {
            "type": "geolocation",
            "description": "Geolocation coordinates of rubber plantations (latitude/longitude)",
            "mandatory": True,
            "article_reference": "Article 9(1)(a)",
        },
        {
            "type": "dds_reference",
            "description": "Reference number of Due Diligence Statement in EU system",
            "mandatory": True,
            "article_reference": "Article 9(1)(d)",
        },
        {
            "type": "product_description",
            "description": "Product description with CN code, natural rubber type",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "quantity_declaration",
            "description": "Quantity in net mass (kg)",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "harvest_date",
            "description": "Date or period of production (after 2020-12-31)",
            "mandatory": True,
            "article_reference": "Article 2(1) cutoff date",
        },
        {
            "type": "compliance_declaration",
            "description": "Declaration that rubber is deforestation-free and legally produced",
            "mandatory": True,
            "article_reference": "Article 9(1)(f)",
        },
        {
            "type": "supplier_information",
            "description": "Supplier/processor name, address, contact",
            "mandatory": True,
            "article_reference": "Article 9(1)(e)",
        },
    ],
    "soya": [
        {
            "type": "geolocation",
            "description": "Geolocation coordinates of soy farms (latitude/longitude)",
            "mandatory": True,
            "article_reference": "Article 9(1)(a)",
        },
        {
            "type": "dds_reference",
            "description": "Reference number of Due Diligence Statement in EU system",
            "mandatory": True,
            "article_reference": "Article 9(1)(d)",
        },
        {
            "type": "product_description",
            "description": "Product description with CN code, soybean/meal/oil",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "quantity_declaration",
            "description": "Quantity in net mass (kg)",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "harvest_date",
            "description": "Harvest season/date (after 2020-12-31)",
            "mandatory": True,
            "article_reference": "Article 2(1) cutoff date",
        },
        {
            "type": "compliance_declaration",
            "description": "Declaration that soy is deforestation-free and legally produced",
            "mandatory": True,
            "article_reference": "Article 9(1)(f)",
        },
        {
            "type": "supplier_information",
            "description": "Supplier/cooperative name, address, contact",
            "mandatory": True,
            "article_reference": "Article 9(1)(e)",
        },
        {
            "type": "certificate",
            "description": "Certification (RTRS, ISCC, Organic)",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
    ],
    "wood": [
        {
            "type": "geolocation",
            "description": "Geolocation coordinates of forest plots (latitude/longitude)",
            "mandatory": True,
            "article_reference": "Article 9(1)(a)",
        },
        {
            "type": "dds_reference",
            "description": "Reference number of Due Diligence Statement in EU system",
            "mandatory": True,
            "article_reference": "Article 9(1)(d)",
        },
        {
            "type": "product_description",
            "description": "Product description with CN code, wood species, product type",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "quantity_declaration",
            "description": "Quantity in volume (m³) or net mass (kg)",
            "mandatory": True,
            "article_reference": "Article 9(1)(c)",
        },
        {
            "type": "harvest_date",
            "description": "Harvest date or period (after 2020-12-31)",
            "mandatory": True,
            "article_reference": "Article 2(1) cutoff date",
        },
        {
            "type": "compliance_declaration",
            "description": "Declaration that wood is deforestation-free and legally produced",
            "mandatory": True,
            "article_reference": "Article 9(1)(f)",
        },
        {
            "type": "supplier_information",
            "description": "Supplier/forest manager name, address, contact",
            "mandatory": True,
            "article_reference": "Article 9(1)(e)",
        },
        {
            "type": "certificate",
            "description": "Certification (FSC, PEFC)",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
        {
            "type": "harvest_permit",
            "description": "Legal harvest permit or concession",
            "mandatory": True,
            "article_reference": "Article 9(1)(b)",
        },
        {
            "type": "phytosanitary",
            "description": "Phytosanitary certificate (for international trade)",
            "mandatory": False,
            "article_reference": "Supporting document",
        },
    ],
}

# ===========================================================================
# Document Templates
# ===========================================================================
#
# Template metadata for each document type.

DOCUMENT_TEMPLATES: Dict[str, DocumentTemplate] = {
    "geolocation": {
        "type": "geolocation",
        "name": "Geolocation Coordinates",
        "format": "JSON/GeoJSON/CSV",
        "required_fields": ["latitude", "longitude", "plot_id"],
        "optional_fields": ["area_hectares", "accuracy_meters", "collection_date"],
    },
    "dds_reference": {
        "type": "dds_reference",
        "name": "Due Diligence Statement Reference",
        "format": "TEXT",
        "required_fields": ["dds_number", "submission_date"],
        "optional_fields": ["status"],
    },
    "product_description": {
        "type": "product_description",
        "name": "Product Description",
        "format": "TEXT/JSON",
        "required_fields": ["cn_code", "commodity", "product_name"],
        "optional_fields": ["variety", "processing_level", "grade"],
    },
    "quantity_declaration": {
        "type": "quantity_declaration",
        "name": "Quantity Declaration",
        "format": "TEXT/JSON",
        "required_fields": ["quantity", "unit"],
        "optional_fields": ["gross_weight", "net_weight"],
    },
    "harvest_date": {
        "type": "harvest_date",
        "name": "Harvest Date",
        "format": "DATE",
        "required_fields": ["harvest_date"],
        "optional_fields": ["harvest_period_start", "harvest_period_end"],
    },
    "compliance_declaration": {
        "type": "compliance_declaration",
        "name": "Compliance Declaration",
        "format": "PDF/TEXT",
        "required_fields": ["declaration_text", "signatory_name", "signature_date"],
        "optional_fields": ["witness_name"],
    },
    "supplier_information": {
        "type": "supplier_information",
        "name": "Supplier Information",
        "format": "JSON/TEXT",
        "required_fields": ["supplier_name", "address", "country"],
        "optional_fields": ["tax_id", "phone", "email"],
    },
    "certificate": {
        "type": "certificate",
        "name": "Certification",
        "format": "PDF",
        "required_fields": ["certificate_number", "scheme", "issue_date", "expiry_date"],
        "optional_fields": ["certification_body", "scope"],
    },
    "trade_license": {
        "type": "trade_license",
        "name": "Trade License",
        "format": "PDF",
        "required_fields": ["license_number", "issue_date", "issuing_authority"],
        "optional_fields": ["expiry_date"],
    },
    "harvest_permit": {
        "type": "harvest_permit",
        "name": "Harvest Permit",
        "format": "PDF",
        "required_fields": ["permit_number", "issue_date", "issuing_authority", "forest_plot_id"],
        "optional_fields": ["expiry_date", "volume_authorized"],
    },
    "phytosanitary": {
        "type": "phytosanitary",
        "name": "Phytosanitary Certificate",
        "format": "PDF",
        "required_fields": ["certificate_number", "issue_date", "issuing_authority"],
        "optional_fields": ["expiry_date"],
    },
    "hcv_assessment": {
        "type": "hcv_assessment",
        "name": "High Conservation Value Assessment",
        "format": "PDF",
        "required_fields": ["assessment_date", "assessor_name", "hcv_identified"],
        "optional_fields": ["management_plan"],
    },
}

# ===========================================================================
# Validation Rules (per document type)
# ===========================================================================
#
# Validation rules for each document type.

VALIDATION_RULES: Dict[str, ValidationRule] = {
    "geolocation": {
        "type": "geolocation",
        "latitude_range": (-90.0, 90.0),
        "longitude_range": (-180.0, 180.0),
        "accuracy_max_meters": 10000,
        "required_precision": 6,  # decimal places
    },
    "harvest_date": {
        "type": "harvest_date",
        "cutoff_date": "2020-12-31",
        "max_future_days": 0,  # Cannot be in future
    },
    "quantity_declaration": {
        "type": "quantity_declaration",
        "min_quantity": 0.0,
        "valid_units": ["kg", "tonnes", "m3", "liters"],
    },
    "certificate": {
        "type": "certificate",
        "expiry_buffer_days": 90,
        "valid_schemes": ["FSC", "PEFC", "RSPO", "RAINFOREST_ALLIANCE", "UTZ", "ORGANIC", "FAIR_TRADE", "ISCC"],
    },
}

# ===========================================================================
# Expiry Policies (per document type)
# ===========================================================================
#
# Document expiry periods and renewal requirements.

EXPIRY_POLICIES: Dict[str, ExpiryPolicy] = {
    "certificate": {
        "type": "certificate",
        "expiry_period_months": 36,  # 3 years (varies by scheme)
        "renewal_warning_days": 90,
        "grace_period_days": 0,
    },
    "trade_license": {
        "type": "trade_license",
        "expiry_period_months": 12,  # 1 year
        "renewal_warning_days": 60,
        "grace_period_days": 30,
    },
    "harvest_permit": {
        "type": "harvest_permit",
        "expiry_period_months": 12,  # 1 year
        "renewal_warning_days": 60,
        "grace_period_days": 0,
    },
    "phytosanitary": {
        "type": "phytosanitary",
        "expiry_period_months": 3,  # 3 months
        "renewal_warning_days": 30,
        "grace_period_days": 0,
    },
    "compliance_declaration": {
        "type": "compliance_declaration",
        "expiry_period_months": 12,  # 1 year
        "renewal_warning_days": 90,
        "grace_period_days": 0,
    },
}

# ===========================================================================
# Language Requirements (per EU member state)
# ===========================================================================
#
# Accepted languages for due diligence documentation per EU member state.

LANGUAGE_REQUIREMENTS: Dict[str, LanguageRequirement] = {
    "DEU": {"country": "DEU", "name": "Germany", "accepted_languages": ["de", "en"]},
    "FRA": {"country": "FRA", "name": "France", "accepted_languages": ["fr", "en"]},
    "ITA": {"country": "ITA", "name": "Italy", "accepted_languages": ["it", "en"]},
    "ESP": {"country": "ESP", "name": "Spain", "accepted_languages": ["es", "en"]},
    "POL": {"country": "POL", "name": "Poland", "accepted_languages": ["pl", "en"]},
    "NLD": {"country": "NLD", "name": "Netherlands", "accepted_languages": ["nl", "en"]},
    "BEL": {"country": "BEL", "name": "Belgium", "accepted_languages": ["nl", "fr", "en"]},
    "AUT": {"country": "AUT", "name": "Austria", "accepted_languages": ["de", "en"]},
    "SWE": {"country": "SWE", "name": "Sweden", "accepted_languages": ["sv", "en"]},
    "DNK": {"country": "DNK", "name": "Denmark", "accepted_languages": ["da", "en"]},
    "FIN": {"country": "FIN", "name": "Finland", "accepted_languages": ["fi", "sv", "en"]},
    "PRT": {"country": "PRT", "name": "Portugal", "accepted_languages": ["pt", "en"]},
    "CZE": {"country": "CZE", "name": "Czech Republic", "accepted_languages": ["cs", "en"]},
    "HUN": {"country": "HUN", "name": "Hungary", "accepted_languages": ["hu", "en"]},
    "GRC": {"country": "GRC", "name": "Greece", "accepted_languages": ["el", "en"]},
    "ROU": {"country": "ROU", "name": "Romania", "accepted_languages": ["ro", "en"]},
    "BGR": {"country": "BGR", "name": "Bulgaria", "accepted_languages": ["bg", "en"]},
    "SVK": {"country": "SVK", "name": "Slovakia", "accepted_languages": ["sk", "en"]},
    "HRV": {"country": "HRV", "name": "Croatia", "accepted_languages": ["hr", "en"]},
    "IRL": {"country": "IRL", "name": "Ireland", "accepted_languages": ["en", "ga"]},
    "LTU": {"country": "LTU", "name": "Lithuania", "accepted_languages": ["lt", "en"]},
    "LVA": {"country": "LVA", "name": "Latvia", "accepted_languages": ["lv", "en"]},
    "EST": {"country": "EST", "name": "Estonia", "accepted_languages": ["et", "en"]},
    "SVN": {"country": "SVN", "name": "Slovenia", "accepted_languages": ["sl", "en"]},
    "LUX": {"country": "LUX", "name": "Luxembourg", "accepted_languages": ["de", "fr", "en"]},
    "MLT": {"country": "MLT", "name": "Malta", "accepted_languages": ["mt", "en"]},
    "CYP": {"country": "CYP", "name": "Cyprus", "accepted_languages": ["el", "en"]},
}

# ===========================================================================
# Helper functions
# ===========================================================================


def get_required_docs(commodity: str) -> Optional[List[DocumentRequirement]]:
    """
    Retrieve required documents for EUDR commodity.

    Args:
        commodity: EUDR commodity (e.g., "cattle", "cocoa")

    Returns:
        List of DocumentRequirement dicts or None if not found
    """
    return EUDR_REQUIRED_DOCUMENTS.get(commodity)


def get_template(document_type: str) -> Optional[DocumentTemplate]:
    """
    Retrieve document template by document type.

    Args:
        document_type: Document type (e.g., "geolocation", "certificate")

    Returns:
        DocumentTemplate dict or None if not found
    """
    return DOCUMENT_TEMPLATES.get(document_type)


def get_validation_rules(document_type: str) -> Optional[ValidationRule]:
    """
    Retrieve validation rules by document type.

    Args:
        document_type: Document type (e.g., "geolocation")

    Returns:
        ValidationRule dict or None if not found
    """
    return VALIDATION_RULES.get(document_type)


def get_expiry_policy(document_type: str) -> Optional[ExpiryPolicy]:
    """
    Retrieve expiry policy by document type.

    Args:
        document_type: Document type (e.g., "certificate")

    Returns:
        ExpiryPolicy dict or None if not found
    """
    return EXPIRY_POLICIES.get(document_type)


def get_language_requirements(country_code: str) -> Optional[List[str]]:
    """
    Retrieve accepted languages for EU member state.

    Args:
        country_code: ISO 3166-1 alpha-3 country code (e.g., "DEU")

    Returns:
        List of accepted language codes or None if not found
    """
    req = LANGUAGE_REQUIREMENTS.get(country_code)
    return req["accepted_languages"] if req else None


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "EUDR_COMMODITIES",
    "EUDR_REQUIRED_DOCUMENTS",
    "DOCUMENT_TEMPLATES",
    "VALIDATION_RULES",
    "EXPIRY_POLICIES",
    "LANGUAGE_REQUIREMENTS",
    "get_required_docs",
    "get_template",
    "get_validation_rules",
    "get_expiry_policy",
    "get_language_requirements",
]
