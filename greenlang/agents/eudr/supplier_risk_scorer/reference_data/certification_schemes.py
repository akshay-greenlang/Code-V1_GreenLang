# -*- coding: utf-8 -*-
"""
Certification Schemes Database - AGENT-EUDR-017 Supplier Risk Scorer

Comprehensive certification scheme reference data for EUDR compliance supplier
assessment. Provides metadata for 8 major certification schemes (FSC, PEFC,
RSPO, Rainforest Alliance, UTZ, Organic, Fair Trade, ISCC), scheme equivalences,
accredited certification bodies, certification requirements, and commodity-scheme
mapping for deterministic certification validation.

Data includes:
    - CERTIFICATION_SCHEMES: 8 major schemes with metadata (name, type,
      commodities_covered, regions, validity_period, chain_of_custody_types,
      accreditation_body, website)
    - SCHEME_EQUIVALENCES: mapping of equivalent certifications across schemes
    - ACCREDITED_CERTIFICATION_BODIES: list of accredited CBs per scheme
    - CERTIFICATION_REQUIREMENTS: per-scheme requirements (documents, audits, frequency)
    - COMMODITY_SCHEME_MAPPING: which schemes apply to which EUDR commodities

Data Sources:
    - FSC Forest Stewardship Council Standards 2024
    - PEFC Programme for the Endorsement of Forest Certification 2024
    - RSPO Roundtable on Sustainable Palm Oil Principles & Criteria 2024
    - Rainforest Alliance Certification Program 2024
    - UTZ Certification Standards 2024
    - IFOAM Organic Standards 2024
    - Fairtrade International Standards 2024
    - ISCC International Sustainability & Carbon Certification 2024

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

CertificationScheme = Dict[str, Any]
SchemeEquivalence = Dict[str, List[str]]
CertificationBody = Dict[str, Any]
CertificationRequirement = Dict[str, Any]
CommoditySchemeMap = Dict[str, List[str]]

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "FSC Forest Stewardship Council Standards 2024",
    "PEFC Programme for the Endorsement of Forest Certification 2024",
    "RSPO Roundtable on Sustainable Palm Oil Principles & Criteria 2024",
    "Rainforest Alliance Certification Program 2024",
    "UTZ Certification Standards 2024",
    "IFOAM Organic International Standards 2024",
    "Fairtrade International Standards 2024",
    "ISCC International Sustainability & Carbon Certification 2024",
]

# ===========================================================================
# Certification Schemes (8 major schemes)
# ===========================================================================
#
# Each record keys:
#   scheme_id               - Unique identifier
#   name                    - Full scheme name
#   abbreviation            - Abbreviation/acronym
#   type                    - forest | agricultural | multi_sector
#   commodities_covered     - List of EUDR commodities
#   regions                 - Geographic regions where active
#   validity_period_years   - Certificate validity period (years)
#   chain_of_custody_types  - Supported CoC types (physical/segregation/mass_balance/identity_preserved)
#   accreditation_body      - Primary accreditation body
#   website                 - Official website URL
#   eudr_recognized         - Whether recognized for EUDR compliance

CERTIFICATION_SCHEMES: Dict[str, CertificationScheme] = {
    "FSC": {
        "scheme_id": "FSC",
        "name": "Forest Stewardship Council",
        "abbreviation": "FSC",
        "type": "forest",
        "commodities_covered": ["wood"],
        "regions": ["global"],
        "validity_period_years": 5,
        "chain_of_custody_types": ["physical_separation", "segregation", "percentage_based", "credit_system"],
        "accreditation_body": "ASI (Assurance Services International)",
        "website": "https://fsc.org",
        "eudr_recognized": True,
        "description": "Leading forest certification system ensuring responsible forest management",
    },
    "PEFC": {
        "scheme_id": "PEFC",
        "name": "Programme for the Endorsement of Forest Certification",
        "abbreviation": "PEFC",
        "type": "forest",
        "commodities_covered": ["wood"],
        "regions": ["global"],
        "validity_period_years": 5,
        "chain_of_custody_types": ["physical_separation", "segregation", "percentage_based", "credit_system"],
        "accreditation_body": "PEFC Council",
        "website": "https://pefc.org",
        "eudr_recognized": True,
        "description": "World's largest forest certification system with national endorsements",
    },
    "RSPO": {
        "scheme_id": "RSPO",
        "name": "Roundtable on Sustainable Palm Oil",
        "abbreviation": "RSPO",
        "type": "agricultural",
        "commodities_covered": ["oil_palm"],
        "regions": ["southeast_asia", "africa", "south_america"],
        "validity_period_years": 5,
        "chain_of_custody_types": ["identity_preserved", "segregation", "mass_balance", "book_and_claim"],
        "accreditation_body": "RSPO Secretariat",
        "website": "https://rspo.org",
        "eudr_recognized": True,
        "description": "Global standard for sustainable palm oil production",
    },
    "RAINFOREST_ALLIANCE": {
        "scheme_id": "RAINFOREST_ALLIANCE",
        "name": "Rainforest Alliance",
        "abbreviation": "RA",
        "type": "agricultural",
        "commodities_covered": ["coffee", "cocoa", "wood"],
        "regions": ["global"],
        "validity_period_years": 3,
        "chain_of_custody_types": ["segregation", "mass_balance"],
        "accreditation_body": "Rainforest Alliance",
        "website": "https://www.rainforest-alliance.org",
        "eudr_recognized": True,
        "description": "Certification for sustainable agriculture, forestry, and tourism",
    },
    "UTZ": {
        "scheme_id": "UTZ",
        "name": "UTZ Certified",
        "abbreviation": "UTZ",
        "type": "agricultural",
        "commodities_covered": ["coffee", "cocoa"],
        "regions": ["global"],
        "validity_period_years": 3,
        "chain_of_custody_types": ["segregation", "mass_balance"],
        "accreditation_body": "Rainforest Alliance (merged 2018)",
        "website": "https://utz.org",
        "eudr_recognized": True,
        "description": "Certification program for sustainable farming of coffee, cocoa, and tea (merged with RA)",
    },
    "ORGANIC": {
        "scheme_id": "ORGANIC",
        "name": "Organic Certification (EU/USDA/IFOAM)",
        "abbreviation": "ORGANIC",
        "type": "agricultural",
        "commodities_covered": ["coffee", "cocoa", "cattle", "soya"],
        "regions": ["global"],
        "validity_period_years": 1,
        "chain_of_custody_types": ["segregation"],
        "accreditation_body": "IFOAM Organics International / EU / USDA NOP",
        "website": "https://www.ifoam.bio",
        "eudr_recognized": False,
        "description": "Organic farming certification (various national and international schemes)",
    },
    "FAIR_TRADE": {
        "scheme_id": "FAIR_TRADE",
        "name": "Fairtrade International",
        "abbreviation": "FT",
        "type": "agricultural",
        "commodities_covered": ["coffee", "cocoa"],
        "regions": ["global"],
        "validity_period_years": 3,
        "chain_of_custody_types": ["physical_separation", "mass_balance"],
        "accreditation_body": "FLOCERT",
        "website": "https://www.fairtrade.net",
        "eudr_recognized": False,
        "description": "Certification focused on fair prices, labor rights, and community development",
    },
    "ISCC": {
        "scheme_id": "ISCC",
        "name": "International Sustainability & Carbon Certification",
        "abbreviation": "ISCC",
        "type": "multi_sector",
        "commodities_covered": ["oil_palm", "soya", "wood"],
        "regions": ["global"],
        "validity_period_years": 1,
        "chain_of_custody_types": ["segregation", "mass_balance"],
        "accreditation_body": "ISCC System GmbH",
        "website": "https://www.iscc-system.org",
        "eudr_recognized": True,
        "description": "Multi-sector certification for sustainable biomass, bioenergy, and circular materials",
    },
}

# ===========================================================================
# Scheme Equivalences
# ===========================================================================
#
# Maps schemes to their equivalents/mutually recognized schemes.

SCHEME_EQUIVALENCES: Dict[str, SchemeEquivalence] = {
    "FSC": {
        "scheme_id": "FSC",
        "equivalents": ["PEFC"],
        "description": "FSC and PEFC mutually recognized for forest management",
    },
    "PEFC": {
        "scheme_id": "PEFC",
        "equivalents": ["FSC"],
        "description": "PEFC and FSC mutually recognized for forest management",
    },
    "RAINFOREST_ALLIANCE": {
        "scheme_id": "RAINFOREST_ALLIANCE",
        "equivalents": ["UTZ"],
        "description": "Rainforest Alliance merged with UTZ in 2018",
    },
    "UTZ": {
        "scheme_id": "UTZ",
        "equivalents": ["RAINFOREST_ALLIANCE"],
        "description": "UTZ merged with Rainforest Alliance in 2018",
    },
    "RSPO": {
        "scheme_id": "RSPO",
        "equivalents": ["ISCC"],
        "description": "RSPO and ISCC both recognized for palm oil sustainability",
    },
    "ISCC": {
        "scheme_id": "ISCC",
        "equivalents": ["RSPO"],
        "description": "ISCC and RSPO both recognized for palm oil sustainability",
    },
}

# ===========================================================================
# Accredited Certification Bodies (per scheme)
# ===========================================================================
#
# List of accredited CBs authorized to issue certificates for each scheme.

ACCREDITED_CERTIFICATION_BODIES: Dict[str, List[CertificationBody]] = {
    "FSC": [
        {"cb_id": "FSC-CB-001", "name": "SGS", "country": "CHE", "status": "active"},
        {"cb_id": "FSC-CB-002", "name": "Bureau Veritas", "country": "FRA", "status": "active"},
        {"cb_id": "FSC-CB-003", "name": "NEPCon", "country": "DNK", "status": "active"},
        {"cb_id": "FSC-CB-004", "name": "Control Union", "country": "NLD", "status": "active"},
        {"cb_id": "FSC-CB-005", "name": "SCS Global Services", "country": "USA", "status": "active"},
        {"cb_id": "FSC-CB-006", "name": "Rainforest Alliance", "country": "USA", "status": "active"},
    ],
    "PEFC": [
        {"cb_id": "PEFC-CB-001", "name": "SGS", "country": "CHE", "status": "active"},
        {"cb_id": "PEFC-CB-002", "name": "TÜV SÜD", "country": "DEU", "status": "active"},
        {"cb_id": "PEFC-CB-003", "name": "DNV GL", "country": "NOR", "status": "active"},
        {"cb_id": "PEFC-CB-004", "name": "Control Union", "country": "NLD", "status": "active"},
    ],
    "RSPO": [
        {"cb_id": "RSPO-CB-001", "name": "SGS", "country": "CHE", "status": "active"},
        {"cb_id": "RSPO-CB-002", "name": "Bureau Veritas", "country": "FRA", "status": "active"},
        {"cb_id": "RSPO-CB-003", "name": "Control Union", "country": "NLD", "status": "active"},
        {"cb_id": "RSPO-CB-004", "name": "BSI", "country": "GBR", "status": "active"},
        {"cb_id": "RSPO-CB-005", "name": "TÜV NORD", "country": "DEU", "status": "active"},
    ],
    "RAINFOREST_ALLIANCE": [
        {"cb_id": "RA-CB-001", "name": "Rainforest Alliance", "country": "USA", "status": "active"},
        {"cb_id": "RA-CB-002", "name": "Control Union", "country": "NLD", "status": "active"},
        {"cb_id": "RA-CB-003", "name": "SGS", "country": "CHE", "status": "active"},
        {"cb_id": "RA-CB-004", "name": "FLOCERT", "country": "DEU", "status": "active"},
    ],
    "UTZ": [
        {"cb_id": "UTZ-CB-001", "name": "Control Union", "country": "NLD", "status": "active"},
        {"cb_id": "UTZ-CB-002", "name": "FLOCERT", "country": "DEU", "status": "active"},
        {"cb_id": "UTZ-CB-003", "name": "SGS", "country": "CHE", "status": "active"},
    ],
    "ORGANIC": [
        {"cb_id": "ORG-CB-001", "name": "Ecocert", "country": "FRA", "status": "active"},
        {"cb_id": "ORG-CB-002", "name": "Control Union", "country": "NLD", "status": "active"},
        {"cb_id": "ORG-CB-003", "name": "IMO", "country": "CHE", "status": "active"},
        {"cb_id": "ORG-CB-004", "name": "Kiwa BCS", "country": "DEU", "status": "active"},
        {"cb_id": "ORG-CB-005", "name": "CCOF", "country": "USA", "status": "active"},
    ],
    "FAIR_TRADE": [
        {"cb_id": "FT-CB-001", "name": "FLOCERT", "country": "DEU", "status": "active"},
    ],
    "ISCC": [
        {"cb_id": "ISCC-CB-001", "name": "SGS", "country": "CHE", "status": "active"},
        {"cb_id": "ISCC-CB-002", "name": "Control Union", "country": "NLD", "status": "active"},
        {"cb_id": "ISCC-CB-003", "name": "Bureau Veritas", "country": "FRA", "status": "active"},
        {"cb_id": "ISCC-CB-004", "name": "TÜV SÜD", "country": "DEU", "status": "active"},
    ],
}

# ===========================================================================
# Certification Requirements (per scheme)
# ===========================================================================
#
# Required documents, audits, and frequencies for each scheme.

CERTIFICATION_REQUIREMENTS: Dict[str, CertificationRequirement] = {
    "FSC": {
        "scheme_id": "FSC",
        "required_documents": [
            "forest_management_plan",
            "harvest_records",
            "chain_of_custody_documentation",
            "social_impact_assessment",
            "environmental_impact_assessment",
        ],
        "audit_frequency": "annual",
        "surveillance_audits_per_cycle": 4,
        "cycle_years": 5,
        "initial_audit_required": True,
        "surveillance_audit_required": True,
        "recertification_audit_required": True,
    },
    "PEFC": {
        "scheme_id": "PEFC",
        "required_documents": [
            "forest_management_plan",
            "harvest_records",
            "chain_of_custody_documentation",
            "legal_compliance_records",
        ],
        "audit_frequency": "annual",
        "surveillance_audits_per_cycle": 4,
        "cycle_years": 5,
        "initial_audit_required": True,
        "surveillance_audit_required": True,
        "recertification_audit_required": True,
    },
    "RSPO": {
        "scheme_id": "RSPO",
        "required_documents": [
            "plantation_management_plan",
            "harvest_records",
            "chain_of_custody_documentation",
            "hcv_assessment",
            "hcs_assessment",
            "fpic_documentation",
        ],
        "audit_frequency": "annual",
        "surveillance_audits_per_cycle": 4,
        "cycle_years": 5,
        "initial_audit_required": True,
        "surveillance_audit_required": True,
        "recertification_audit_required": True,
    },
    "RAINFOREST_ALLIANCE": {
        "scheme_id": "RAINFOREST_ALLIANCE",
        "required_documents": [
            "farm_management_plan",
            "harvest_records",
            "chain_of_custody_documentation",
            "risk_assessment",
            "training_records",
        ],
        "audit_frequency": "annual",
        "surveillance_audits_per_cycle": 2,
        "cycle_years": 3,
        "initial_audit_required": True,
        "surveillance_audit_required": True,
        "recertification_audit_required": True,
    },
    "UTZ": {
        "scheme_id": "UTZ",
        "required_documents": [
            "farm_management_plan",
            "harvest_records",
            "chain_of_custody_documentation",
            "training_records",
        ],
        "audit_frequency": "annual",
        "surveillance_audits_per_cycle": 2,
        "cycle_years": 3,
        "initial_audit_required": True,
        "surveillance_audit_required": True,
        "recertification_audit_required": True,
    },
    "ORGANIC": {
        "scheme_id": "ORGANIC",
        "required_documents": [
            "organic_system_plan",
            "harvest_records",
            "input_records",
            "handling_records",
        ],
        "audit_frequency": "annual",
        "surveillance_audits_per_cycle": 0,
        "cycle_years": 1,
        "initial_audit_required": True,
        "surveillance_audit_required": False,
        "recertification_audit_required": True,
    },
    "FAIR_TRADE": {
        "scheme_id": "FAIR_TRADE",
        "required_documents": [
            "producer_organization_plan",
            "harvest_records",
            "fairtrade_premium_usage_records",
            "financial_records",
        ],
        "audit_frequency": "every_3_years",
        "surveillance_audits_per_cycle": 0,
        "cycle_years": 3,
        "initial_audit_required": True,
        "surveillance_audit_required": False,
        "recertification_audit_required": True,
    },
    "ISCC": {
        "scheme_id": "ISCC",
        "required_documents": [
            "sustainability_declaration",
            "mass_balance_documentation",
            "chain_of_custody_documentation",
            "greenhouse_gas_calculation",
        ],
        "audit_frequency": "annual",
        "surveillance_audits_per_cycle": 0,
        "cycle_years": 1,
        "initial_audit_required": True,
        "surveillance_audit_required": False,
        "recertification_audit_required": True,
    },
}

# ===========================================================================
# Commodity-Scheme Mapping
# ===========================================================================
#
# Which certification schemes apply to which EUDR commodities.

COMMODITY_SCHEME_MAPPING: Dict[str, CommoditySchemeMap] = {
    "cattle": {
        "commodity": "cattle",
        "applicable_schemes": ["ORGANIC"],
        "recommended_schemes": ["ORGANIC"],
    },
    "cocoa": {
        "commodity": "cocoa",
        "applicable_schemes": ["RAINFOREST_ALLIANCE", "UTZ", "ORGANIC", "FAIR_TRADE"],
        "recommended_schemes": ["RAINFOREST_ALLIANCE", "UTZ", "FAIR_TRADE"],
    },
    "coffee": {
        "commodity": "coffee",
        "applicable_schemes": ["RAINFOREST_ALLIANCE", "UTZ", "ORGANIC", "FAIR_TRADE"],
        "recommended_schemes": ["RAINFOREST_ALLIANCE", "UTZ", "FAIR_TRADE"],
    },
    "oil_palm": {
        "commodity": "oil_palm",
        "applicable_schemes": ["RSPO", "ISCC"],
        "recommended_schemes": ["RSPO"],
    },
    "rubber": {
        "commodity": "rubber",
        "applicable_schemes": [],
        "recommended_schemes": [],
    },
    "soya": {
        "commodity": "soya",
        "applicable_schemes": ["ORGANIC", "ISCC"],
        "recommended_schemes": ["ISCC"],
    },
    "wood": {
        "commodity": "wood",
        "applicable_schemes": ["FSC", "PEFC", "RAINFOREST_ALLIANCE"],
        "recommended_schemes": ["FSC", "PEFC"],
    },
}

# ===========================================================================
# Helper functions
# ===========================================================================


def get_scheme(scheme_id: str) -> Optional[CertificationScheme]:
    """
    Retrieve certification scheme by scheme_id.

    Args:
        scheme_id: Unique scheme identifier (e.g., "FSC", "RSPO")

    Returns:
        CertificationScheme dict or None if not found
    """
    return CERTIFICATION_SCHEMES.get(scheme_id)


def get_equivalences(scheme_id: str) -> Optional[List[str]]:
    """
    Retrieve equivalent schemes for a given scheme_id.

    Args:
        scheme_id: Unique scheme identifier (e.g., "FSC")

    Returns:
        List of equivalent scheme_ids or None if not found
    """
    equiv_data = SCHEME_EQUIVALENCES.get(scheme_id)
    return equiv_data["equivalents"] if equiv_data else None


def is_accredited(scheme_id: str, cb_name: str) -> bool:
    """
    Check if certification body is accredited for scheme.

    Args:
        scheme_id: Unique scheme identifier (e.g., "FSC")
        cb_name: Certification body name (e.g., "SGS")

    Returns:
        True if CB is accredited and active, False otherwise
    """
    cbs = ACCREDITED_CERTIFICATION_BODIES.get(scheme_id, [])
    for cb in cbs:
        if cb["name"].lower() == cb_name.lower() and cb["status"] == "active":
            return True
    return False


def get_requirements(scheme_id: str) -> Optional[CertificationRequirement]:
    """
    Retrieve certification requirements for scheme.

    Args:
        scheme_id: Unique scheme identifier (e.g., "FSC")

    Returns:
        CertificationRequirement dict or None if not found
    """
    return CERTIFICATION_REQUIREMENTS.get(scheme_id)


def get_schemes_for_commodity(commodity: str) -> Optional[List[str]]:
    """
    Retrieve applicable certification schemes for EUDR commodity.

    Args:
        commodity: EUDR commodity (e.g., "coffee")

    Returns:
        List of applicable scheme_ids or None if not found
    """
    mapping = COMMODITY_SCHEME_MAPPING.get(commodity)
    return mapping["applicable_schemes"] if mapping else None


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "CERTIFICATION_SCHEMES",
    "SCHEME_EQUIVALENCES",
    "ACCREDITED_CERTIFICATION_BODIES",
    "CERTIFICATION_REQUIREMENTS",
    "COMMODITY_SCHEME_MAPPING",
    "get_scheme",
    "get_equivalences",
    "is_accredited",
    "get_requirements",
    "get_schemes_for_commodity",
]
