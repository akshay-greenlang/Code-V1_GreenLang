# -*- coding: utf-8 -*-
"""
Certification Scheme Profiles - AGENT-EUDR-024

Static reference data for five major certification schemes recognized
for EUDR compliance support. Each profile includes scheme details,
audit standards, commodities, recertification cycles, and EUDR
coverage characteristics.

Supported Schemes (5):
    - FSC (Forest Stewardship Council): timber/wood, 5-year
    - PEFC (Programme for Endorsement of Forest Cert): timber, 5-year
    - RSPO (Roundtable on Sustainable Palm Oil): palm oil, 5-year
    - Rainforest Alliance: cocoa/coffee, 3-year
    - ISCC (International Sustainability & Carbon Cert): multi, annual

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Certification Scheme Profiles
# ---------------------------------------------------------------------------

CERTIFICATION_SCHEME_PROFILES: Dict[str, Dict[str, Any]] = {
    "fsc": {
        "scheme_id": "fsc",
        "full_name": "Forest Stewardship Council",
        "abbreviation": "FSC",
        "headquarters": "Bonn, Germany",
        "website": "https://fsc.org",
        "established_year": 1993,
        "primary_commodities": ["wood"],
        "secondary_commodities": ["rubber"],
        "recertification_cycle_years": 5,
        "surveillance_audit_frequency": "annual",
        "audit_standards": [
            "FSC-STD-01-001 (Principles and Criteria for Forest Management)",
            "FSC-STD-40-004 (Chain of Custody Certification)",
            "FSC-STD-40-005 (Controlled Wood Standard)",
            "FSC-STD-20-007 (Forest Management Evaluations)",
        ],
        "nc_classification": {
            "major": "Fundamental failure to achieve an FSC criterion",
            "minor": "Non-systematic or non-persistent failure",
            "observation": "Observation for improvement opportunity",
        },
        "supply_chain_models": ["IP", "SG", "MB"],
        "eudr_relevance": {
            "coverage_pct": 75,
            "strong_areas": ["Art. 3", "Art. 9", "Art. 10", "Art. 29"],
            "gap_areas": ["Art. 11 (EUDR-specific mitigation)"],
            "notes": "Strong alignment for timber/wood supply chains. "
                     "Controlled Wood standard addresses some DDS elements.",
        },
        "accreditation_body": "ASI (Assurance Services International)",
        "total_certificates_global": 58000,
    },
    "pefc": {
        "scheme_id": "pefc",
        "full_name": "Programme for the Endorsement of Forest Certification",
        "abbreviation": "PEFC",
        "headquarters": "Geneva, Switzerland",
        "website": "https://pefc.org",
        "established_year": 1999,
        "primary_commodities": ["wood"],
        "secondary_commodities": [],
        "recertification_cycle_years": 5,
        "surveillance_audit_frequency": "annual",
        "audit_standards": [
            "PEFC ST 1003 (Sustainable Forest Management)",
            "PEFC ST 2002 (Chain of Custody)",
            "PEFC ST 2001 (PEFC Logo Usage)",
        ],
        "nc_classification": {
            "major": "Significant failure to meet PEFC requirements",
            "minor": "Non-systematic failure with limited impact",
            "observation": "Area for improvement",
        },
        "supply_chain_models": ["IP", "SG", "MB"],
        "eudr_relevance": {
            "coverage_pct": 70,
            "strong_areas": ["Art. 3", "Art. 4", "Art. 9", "Art. 29"],
            "gap_areas": ["Art. 9(1)(d) geolocation depth", "Art. 11"],
            "notes": "Good alignment through national endorsement framework. "
                     "PEFC DDS (ST 2002) partially addresses EUDR requirements.",
        },
        "accreditation_body": "National accreditation bodies (NABs)",
        "total_certificates_global": 20000,
    },
    "rspo": {
        "scheme_id": "rspo",
        "full_name": "Roundtable on Sustainable Palm Oil",
        "abbreviation": "RSPO",
        "headquarters": "Kuala Lumpur, Malaysia",
        "website": "https://rspo.org",
        "established_year": 2004,
        "primary_commodities": ["palm_oil"],
        "secondary_commodities": [],
        "recertification_cycle_years": 5,
        "surveillance_audit_frequency": "annual",
        "audit_standards": [
            "RSPO P&C 2018 (Principles and Criteria)",
            "RSPO SCCS (Supply Chain Certification Standard)",
            "RSPO RISS (Risk Assessment for Independent Smallholders)",
        ],
        "nc_classification": {
            "major": "Fundamental failure with potential for significant environmental or social harm",
            "minor": "Non-systematic or isolated failure",
            "observation": "Opportunity for improvement",
        },
        "supply_chain_models": ["IP", "SG", "MB"],
        "eudr_relevance": {
            "coverage_pct": 65,
            "strong_areas": ["Art. 3 (NDE since 2018)", "Art. 4", "Art. 10"],
            "gap_areas": ["Art. 9(1)(d) geolocation for plantations", "Art. 11"],
            "notes": "Good for palm oil supply chains. No Deforestation commitment "
                     "since November 2018 aligns with EUDR cutoff date concept.",
        },
        "accreditation_body": "ASI (Assurance Services International)",
        "total_certificates_global": 5500,
    },
    "rainforest_alliance": {
        "scheme_id": "rainforest_alliance",
        "full_name": "Rainforest Alliance",
        "abbreviation": "RA",
        "headquarters": "Amsterdam, Netherlands / New York, USA",
        "website": "https://www.rainforest-alliance.org",
        "established_year": 1987,
        "primary_commodities": ["cocoa", "coffee"],
        "secondary_commodities": ["rubber"],
        "recertification_cycle_years": 3,
        "surveillance_audit_frequency": "annual",
        "audit_standards": [
            "RA 2020 Sustainable Agriculture Standard",
            "RA 2020 Supply Chain Standard",
            "RA Assurance Manual",
        ],
        "nc_classification": {
            "major": "Systematic failure or single failure with severe impact",
            "minor": "Isolated or non-persistent failure",
            "observation": "Recommendation for improvement",
        },
        "supply_chain_models": ["IP", "SG", "MB"],
        "eudr_relevance": {
            "coverage_pct": 60,
            "strong_areas": ["Art. 3 (deforestation-free)", "Art. 4"],
            "gap_areas": ["Art. 9(1)(d) geolocation detail", "Art. 10 depth", "Art. 11"],
            "notes": "Good for cocoa/coffee supply chains. 2020 Standard includes "
                     "enhanced deforestation requirements. Geolocation at farm level.",
        },
        "accreditation_body": "RA Certification Bodies",
        "total_certificates_global": 70000,
    },
    "iscc": {
        "scheme_id": "iscc",
        "full_name": "International Sustainability and Carbon Certification",
        "abbreviation": "ISCC",
        "headquarters": "Cologne, Germany",
        "website": "https://www.iscc-system.org",
        "established_year": 2010,
        "primary_commodities": ["palm_oil", "soya"],
        "secondary_commodities": ["rubber"],
        "recertification_cycle_years": 1,
        "surveillance_audit_frequency": "annual",
        "audit_standards": [
            "ISCC PLUS",
            "ISCC EU",
            "ISCC System Basics",
            "ISCC Audit Procedures",
        ],
        "nc_classification": {
            "major": "Significant non-compliance requiring corrective action",
            "minor": "Minor deviation not undermining system integrity",
            "observation": "Area for improvement",
        },
        "supply_chain_models": ["MB", "SG"],
        "eudr_relevance": {
            "coverage_pct": 55,
            "strong_areas": ["Art. 3 (sustainability)", "Art. 4 (GHG focus)"],
            "gap_areas": ["Art. 9(1)(d) geolocation", "Art. 10 depth", "Art. 11", "Art. 31 scope"],
            "notes": "Focus on bioenergy and multi-commodity certification. "
                     "Strong GHG monitoring but limited traceability depth for EUDR. "
                     "Annual cycle provides more frequent oversight.",
        },
        "accreditation_body": "ISCC approved certification bodies",
        "total_certificates_global": 12000,
    },
}


def get_scheme_profile(scheme_id: str) -> Optional[Dict[str, Any]]:
    """Get certification scheme profile by identifier.

    Args:
        scheme_id: Scheme identifier (fsc, pefc, rspo, rainforest_alliance, iscc).

    Returns:
        Scheme profile dictionary or None if not found.
    """
    return CERTIFICATION_SCHEME_PROFILES.get(scheme_id.lower())


def get_all_scheme_ids() -> List[str]:
    """Get all supported scheme identifiers.

    Returns:
        List of scheme identifiers.
    """
    return list(CERTIFICATION_SCHEME_PROFILES.keys())


def get_schemes_for_commodity(commodity: str) -> List[Dict[str, Any]]:
    """Get certification schemes applicable to a commodity.

    Args:
        commodity: EUDR commodity name.

    Returns:
        List of applicable scheme profiles.
    """
    applicable = []
    for scheme in CERTIFICATION_SCHEME_PROFILES.values():
        all_commodities = (
            scheme["primary_commodities"] + scheme["secondary_commodities"]
        )
        if commodity.lower() in all_commodities:
            applicable.append(scheme)
    return applicable
