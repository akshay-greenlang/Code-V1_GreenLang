# -*- coding: utf-8 -*-
"""
Article 8 DDS Field Definitions - AGENT-EUDR-026

Reference data defining the mandatory Due Diligence Statement (DDS)
fields per EUDR Article 12(2), mapped to the EUDR agent IDs responsible
for providing each data element and the Article 8/9/10/11 requirements
they satisfy.

This module is used by the DueDiligencePackageGenerator to validate
completeness and by the Quality Gate Engine to assess information
gathering coverage.

Article Structure:
    Article 8: Due diligence obligation (general)
    Article 9: Information gathering requirements
    Article 10: Risk assessment requirements
    Article 11: Risk mitigation requirements
    Article 12: Due diligence statements (DDS content)
    Article 13: Simplified due diligence (low-risk origins)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Article 12(2) DDS field definitions
# ---------------------------------------------------------------------------

#: Complete field definitions for the Due Diligence Statement per Article 12(2).
#: Each field maps to:
#:   - article_ref: EUDR article and sub-paragraph reference
#:   - field_name: Canonical field name used in the DDS package
#:   - description: Human-readable field description
#:   - source_agents: EUDR agent IDs providing this data
#:   - required: Whether the field is mandatory for standard DDS
#:   - simplified_required: Whether required for simplified (Art. 13) DDS
#:   - data_type: Expected data type (str, list, dict, decimal, etc.)
ARTICLE_12_FIELDS: List[Dict[str, Any]] = [
    # Section (a): Operator identification
    {
        "article_ref": "12(2)(a)(i)",
        "field_name": "operator_name",
        "description": "Full legal name of the operator or trader",
        "source_agents": ["EUDR-001"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(a)(ii)",
        "field_name": "postal_address",
        "description": "Registered postal address of the operator",
        "source_agents": ["EUDR-001"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(a)(iii)",
        "field_name": "email_address",
        "description": "Contact email address of the operator",
        "source_agents": ["EUDR-001"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(a)(iv)",
        "field_name": "eori_number",
        "description": "EORI number (Economic Operator Registration and Identification)",
        "source_agents": ["EUDR-001"],
        "required": False,
        "simplified_required": False,
        "data_type": "str",
    },
    # Section (b): Product description
    {
        "article_ref": "12(2)(b)(i)",
        "field_name": "product_description",
        "description": "Description of the relevant product",
        "source_agents": ["EUDR-001", "EUDR-009"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(b)(ii)",
        "field_name": "trade_name",
        "description": "Commercial or trade name of the product",
        "source_agents": ["EUDR-001"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(b)(iii)",
        "field_name": "hs_heading",
        "description": "HS (Harmonised System) tariff heading",
        "source_agents": ["EUDR-001"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(b)(iv)",
        "field_name": "commodity_type",
        "description": "EUDR commodity classification",
        "source_agents": ["EUDR-001"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    # Section (c): Country of production
    {
        "article_ref": "12(2)(c)(i)",
        "field_name": "country_of_production",
        "description": "Country or countries of production",
        "source_agents": ["EUDR-001", "EUDR-002", "EUDR-016"],
        "required": True,
        "simplified_required": True,
        "data_type": "list",
    },
    {
        "article_ref": "12(2)(c)(ii)",
        "field_name": "country_iso_code",
        "description": "ISO 3166-1 alpha-2 country codes",
        "source_agents": ["EUDR-001", "EUDR-016"],
        "required": True,
        "simplified_required": True,
        "data_type": "list",
    },
    {
        "article_ref": "12(2)(c)(iii)",
        "field_name": "region_province",
        "description": "Sub-national region or province if applicable",
        "source_agents": ["EUDR-001", "EUDR-002"],
        "required": False,
        "simplified_required": False,
        "data_type": "str",
    },
    # Section (d): Geolocation
    {
        "article_ref": "12(2)(d)(i)",
        "field_name": "plot_coordinates",
        "description": "GPS coordinates of all production plots",
        "source_agents": ["EUDR-002", "EUDR-006", "EUDR-007"],
        "required": True,
        "simplified_required": True,
        "data_type": "list",
    },
    {
        "article_ref": "12(2)(d)(ii)",
        "field_name": "coordinate_accuracy",
        "description": "Accuracy of GPS coordinates in meters",
        "source_agents": ["EUDR-007"],
        "required": True,
        "simplified_required": True,
        "data_type": "decimal",
    },
    {
        "article_ref": "12(2)(d)(iii)",
        "field_name": "polygon_boundaries",
        "description": "GeoJSON polygon boundaries for plots > 4 hectares",
        "source_agents": ["EUDR-006"],
        "required": False,
        "simplified_required": False,
        "data_type": "dict",
    },
    # Section (e): Quantity
    {
        "article_ref": "12(2)(e)(i)",
        "field_name": "quantity_net_mass_kg",
        "description": "Net mass of the product in kilograms",
        "source_agents": ["EUDR-001", "EUDR-011"],
        "required": True,
        "simplified_required": True,
        "data_type": "decimal",
    },
    {
        "article_ref": "12(2)(e)(ii)",
        "field_name": "supplementary_units",
        "description": "Supplementary measurement units if applicable",
        "source_agents": ["EUDR-001"],
        "required": False,
        "simplified_required": False,
        "data_type": "str",
    },
    # Section (f): Date/time
    {
        "article_ref": "12(2)(f)(i)",
        "field_name": "production_start_date",
        "description": "Start date of the production period",
        "source_agents": ["EUDR-001", "EUDR-003"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(f)(ii)",
        "field_name": "production_end_date",
        "description": "End date of the production period",
        "source_agents": ["EUDR-001", "EUDR-003"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    # Section (g): Deforestation-free
    {
        "article_ref": "12(2)(g)(i)",
        "field_name": "deforestation_free_determination",
        "description": "Determination that products are deforestation-free",
        "source_agents": ["EUDR-003", "EUDR-004", "EUDR-005"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(g)(ii)",
        "field_name": "forest_cover_analysis",
        "description": "Forest cover analysis supporting determination",
        "source_agents": ["EUDR-004"],
        "required": True,
        "simplified_required": False,
        "data_type": "dict",
    },
    {
        "article_ref": "12(2)(g)(iii)",
        "field_name": "land_use_change_analysis",
        "description": "Land use change analysis since Dec 31 2020 cutoff",
        "source_agents": ["EUDR-005"],
        "required": True,
        "simplified_required": False,
        "data_type": "dict",
    },
    # Section (h): Legal compliance
    {
        "article_ref": "12(2)(h)(i)",
        "field_name": "legal_compliance_status",
        "description": "Compliance with relevant legislation of country of production",
        "source_agents": ["EUDR-023"],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(h)(ii)",
        "field_name": "applicable_legislation",
        "description": "List of applicable legislation verified",
        "source_agents": ["EUDR-023"],
        "required": True,
        "simplified_required": True,
        "data_type": "list",
    },
    # Section (i): Risk assessment
    {
        "article_ref": "12(2)(i)(i)",
        "field_name": "composite_risk_score",
        "description": "Weighted composite risk score (0-100)",
        "source_agents": [f"EUDR-{i:03d}" for i in range(16, 26)],
        "required": True,
        "simplified_required": True,
        "data_type": "decimal",
    },
    {
        "article_ref": "12(2)(i)(ii)",
        "field_name": "risk_level_classification",
        "description": "Risk level classification (negligible/low/medium/high/critical)",
        "source_agents": [f"EUDR-{i:03d}" for i in range(16, 26)],
        "required": True,
        "simplified_required": True,
        "data_type": "str",
    },
    {
        "article_ref": "12(2)(i)(iii)",
        "field_name": "risk_dimension_breakdown",
        "description": "Individual risk dimension scores and weights",
        "source_agents": [f"EUDR-{i:03d}" for i in range(16, 26)],
        "required": True,
        "simplified_required": False,
        "data_type": "dict",
    },
]


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_all_fields() -> List[Dict[str, Any]]:
    """Get all DDS field definitions.

    Returns:
        List of field definition dictionaries.
    """
    return list(ARTICLE_12_FIELDS)


def get_required_fields(simplified: bool = False) -> List[Dict[str, Any]]:
    """Get required DDS fields.

    Args:
        simplified: If True, return simplified due diligence requirements.

    Returns:
        List of required field definitions.
    """
    key = "simplified_required" if simplified else "required"
    return [f for f in ARTICLE_12_FIELDS if f.get(key, False)]


def get_fields_by_section(article_ref_prefix: str) -> List[Dict[str, Any]]:
    """Get DDS fields for a specific article section.

    Args:
        article_ref_prefix: Article reference prefix (e.g., "12(2)(a)").

    Returns:
        List of field definitions for that section.
    """
    return [
        f for f in ARTICLE_12_FIELDS
        if f["article_ref"].startswith(article_ref_prefix)
    ]


def get_fields_by_agent(agent_id: str) -> List[Dict[str, Any]]:
    """Get DDS fields sourced from a specific agent.

    Args:
        agent_id: EUDR agent identifier.

    Returns:
        List of field definitions sourced by this agent.
    """
    return [
        f for f in ARTICLE_12_FIELDS
        if agent_id in f.get("source_agents", [])
    ]


def get_field_by_name(field_name: str) -> Optional[Dict[str, Any]]:
    """Get a specific DDS field definition by name.

    Args:
        field_name: Field name to look up.

    Returns:
        Field definition dictionary or None.
    """
    for f in ARTICLE_12_FIELDS:
        if f["field_name"] == field_name:
            return f
    return None


def get_required_field_count(simplified: bool = False) -> int:
    """Get the count of required fields.

    Args:
        simplified: If True, count simplified requirements.

    Returns:
        Number of required fields.
    """
    return len(get_required_fields(simplified))


def get_total_field_count() -> int:
    """Get the total number of DDS fields.

    Returns:
        Total field count.
    """
    return len(ARTICLE_12_FIELDS)
