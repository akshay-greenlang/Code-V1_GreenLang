# -*- coding: utf-8 -*-
"""
Certification Standards Reference Data - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Provides certification type definitions, EUDR acceptance rules, commodity
applicability, validity periods, renewal requirements, audit criteria, and
certification hierarchy mappings for the Multi-Tier Supplier Tracker Agent.
Used for supplier compliance assessment and certification gap analysis
without external API dependencies.

Certification Coverage:
    - Forest Stewardship Council (FSC) - FM, CoC, CW
    - Programme for the Endorsement of Forest Certification (PEFC)
    - Roundtable on Sustainable Palm Oil (RSPO) - P&C, SCC, SCCS
    - Indonesian Sustainable Palm Oil (ISPO)
    - UTZ Certified (now Rainforest Alliance)
    - Rainforest Alliance (RA)
    - Fairtrade International
    - ISO 14001 Environmental Management
    - Organic Certification (EU / USDA)
    - Sustainable Agriculture Network (SAN)
    - Round Table on Responsible Soy (RTRS)
    - Bonsucro (for cane-derived commodities)

Data Sources:
    FSC International Standards v6.0 (2024)
    RSPO Principles & Criteria 2023
    PEFC International Standard PEFC ST 1003:2024
    EU Regulation 2023/1115 (EUDR) Annex II
    Rainforest Alliance 2020 Certification Program

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Certification Standard Definitions
# ---------------------------------------------------------------------------

CERTIFICATION_STANDARDS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # FSC - Forest Stewardship Council
    # ------------------------------------------------------------------
    "FSC_FM": {
        "name": "FSC Forest Management",
        "short_name": "FSC-FM",
        "organization": "Forest Stewardship Council",
        "description": (
            "Certification for responsible forest management ensuring "
            "ecological, social, and economic sustainability."
        ),
        "applicable_commodities": ["wood"],
        "validity_period_months": 60,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": (
            "Accepted as supporting evidence for EUDR due diligence. "
            "Does not replace operator obligations under Article 4."
        ),
        "certification_body_accreditation": "ASI",
        "hierarchy_level": "primary",
        "parent_standard": None,
        "child_standards": ["FSC_COC", "FSC_CW"],
        "scope": "forest_management",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative", "forest_owner"],
        "key_requirements": [
            "Compliance with national and international laws",
            "Workers rights and employment conditions",
            "Indigenous peoples rights",
            "Community relations and workers benefits",
            "Environmental impact assessment",
            "Management plan",
            "Monitoring and assessment",
            "High conservation value forest protection",
        ],
    },
    "FSC_COC": {
        "name": "FSC Chain of Custody",
        "short_name": "FSC-CoC",
        "organization": "Forest Stewardship Council",
        "description": (
            "Tracks FSC-certified material through the supply chain "
            "from forest to final product."
        ),
        "applicable_commodities": ["wood"],
        "validity_period_months": 60,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": (
            "Chain of custody certification provides traceability evidence "
            "for EUDR Article 9 requirements."
        ),
        "certification_body_accreditation": "ASI",
        "hierarchy_level": "chain_of_custody",
        "parent_standard": "FSC_FM",
        "child_standards": [],
        "scope": "supply_chain_traceability",
        "geographic_coverage": "global",
        "supply_chain_tiers": [
            "processor", "trader", "manufacturer", "retailer",
        ],
        "key_requirements": [
            "Material sourcing and traceability",
            "Product group management",
            "Controlled wood due diligence",
            "Labelling and sales",
            "Record keeping",
        ],
    },
    "FSC_CW": {
        "name": "FSC Controlled Wood",
        "short_name": "FSC-CW",
        "organization": "Forest Stewardship Council",
        "description": (
            "Ensures non-certified wood mixed with FSC material avoids "
            "unacceptable sources (illegal, HCV, GMO, land conversion)."
        ),
        "applicable_commodities": ["wood"],
        "validity_period_months": 60,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "partially_accepted",
        "eudr_acceptance_notes": (
            "Controlled wood alone insufficient for EUDR; supports risk "
            "assessment but does not guarantee deforestation-free."
        ),
        "certification_body_accreditation": "ASI",
        "hierarchy_level": "supplementary",
        "parent_standard": "FSC_FM",
        "child_standards": [],
        "scope": "risk_mitigation",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["processor", "trader"],
        "key_requirements": [
            "Risk assessment for controlled wood categories",
            "Supplier verification",
            "Field verification where required",
        ],
    },
    # ------------------------------------------------------------------
    # PEFC
    # ------------------------------------------------------------------
    "PEFC": {
        "name": "PEFC Sustainable Forest Management",
        "short_name": "PEFC",
        "organization": "Programme for the Endorsement of Forest Certification",
        "description": (
            "International framework for mutual recognition of national "
            "forest certification schemes."
        ),
        "applicable_commodities": ["wood"],
        "validity_period_months": 60,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": (
            "PEFC certification accepted as supporting evidence. "
            "Covers 75% of certified forest area globally."
        ),
        "certification_body_accreditation": "National accreditation bodies",
        "hierarchy_level": "primary",
        "parent_standard": None,
        "child_standards": ["PEFC_COC"],
        "scope": "forest_management",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative", "forest_owner"],
        "key_requirements": [
            "Maintenance of forest ecosystem integrity",
            "Biological diversity conservation",
            "Water resource protection",
            "Forest productivity maintenance",
            "Legal compliance",
            "Worker safety and health",
        ],
    },
    "PEFC_COC": {
        "name": "PEFC Chain of Custody",
        "short_name": "PEFC-CoC",
        "organization": "Programme for the Endorsement of Forest Certification",
        "description": (
            "Chain of custody standard for tracking PEFC-certified "
            "material through the supply chain."
        ),
        "applicable_commodities": ["wood"],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 4,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": "Supports EUDR traceability requirements.",
        "certification_body_accreditation": "National accreditation bodies",
        "hierarchy_level": "chain_of_custody",
        "parent_standard": "PEFC",
        "child_standards": [],
        "scope": "supply_chain_traceability",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["processor", "trader", "manufacturer"],
        "key_requirements": [
            "Material identification and traceability",
            "Due diligence system for non-certified material",
            "Record keeping and documentation",
        ],
    },
    # ------------------------------------------------------------------
    # RSPO - Roundtable on Sustainable Palm Oil
    # ------------------------------------------------------------------
    "RSPO_PC": {
        "name": "RSPO Principles and Criteria",
        "short_name": "RSPO-P&C",
        "organization": "Roundtable on Sustainable Palm Oil",
        "description": (
            "Certification for sustainable palm oil production, "
            "including no deforestation, no peat, no exploitation."
        ),
        "applicable_commodities": ["palm_oil"],
        "validity_period_months": 60,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": (
            "RSPO certification with NDPE commitment provides strong "
            "evidence for EUDR palm oil due diligence."
        ),
        "certification_body_accreditation": "ASI",
        "hierarchy_level": "primary",
        "parent_standard": None,
        "child_standards": ["RSPO_SCC", "RSPO_SCCS"],
        "scope": "production",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative", "mill"],
        "key_requirements": [
            "No deforestation (HCS/HCV approach)",
            "No peat development",
            "No exploitation (workers, communities)",
            "Free, Prior and Informed Consent",
            "Transparency and traceability",
            "Environmental responsibility",
            "Continuous improvement",
        ],
    },
    "RSPO_SCC": {
        "name": "RSPO Supply Chain Certification",
        "short_name": "RSPO-SCC",
        "organization": "Roundtable on Sustainable Palm Oil",
        "description": (
            "Supply chain certification for trading, refining, "
            "and manufacturing RSPO-certified palm oil."
        ),
        "applicable_commodities": ["palm_oil"],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 4,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": "Supports EUDR palm oil traceability.",
        "certification_body_accreditation": "ASI",
        "hierarchy_level": "chain_of_custody",
        "parent_standard": "RSPO_PC",
        "child_standards": [],
        "scope": "supply_chain_traceability",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["refinery", "trader", "processor", "importer"],
        "key_requirements": [
            "Identity preserved, segregated, or mass balance model",
            "Certified volume reconciliation",
            "Documentation and record keeping",
        ],
    },
    "RSPO_SCCS": {
        "name": "RSPO Supply Chain Certification Standard (Smallholders)",
        "short_name": "RSPO-SCCS",
        "organization": "Roundtable on Sustainable Palm Oil",
        "description": (
            "Simplified supply chain standard for independent "
            "smallholder groups producing palm oil."
        ),
        "applicable_commodities": ["palm_oil"],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 4,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": "Supports smallholder EUDR compliance.",
        "certification_body_accreditation": "ASI",
        "hierarchy_level": "smallholder",
        "parent_standard": "RSPO_PC",
        "child_standards": [],
        "scope": "smallholder_production",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative"],
        "key_requirements": [
            "Group management system",
            "Internal control system",
            "Simplified P&C compliance",
        ],
    },
    # ------------------------------------------------------------------
    # ISPO - Indonesian Sustainable Palm Oil
    # ------------------------------------------------------------------
    "ISPO": {
        "name": "Indonesian Sustainable Palm Oil",
        "short_name": "ISPO",
        "organization": "Ministry of Agriculture, Republic of Indonesia",
        "description": (
            "Mandatory Indonesian national standard for sustainable "
            "palm oil production."
        ),
        "applicable_commodities": ["palm_oil"],
        "validity_period_months": 60,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "partially_accepted",
        "eudr_acceptance_notes": (
            "ISPO provides baseline evidence but may not meet all EUDR "
            "deforestation-free requirements. Supplementary verification needed."
        ),
        "certification_body_accreditation": "KAN (Indonesia)",
        "hierarchy_level": "primary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "production",
        "geographic_coverage": "indonesia",
        "supply_chain_tiers": ["farmer", "cooperative", "mill"],
        "key_requirements": [
            "Legal land use rights",
            "Environmental management",
            "No primary forest conversion (post-2008)",
            "Community partnership",
            "Business management transparency",
        ],
    },
    # ------------------------------------------------------------------
    # Rainforest Alliance
    # ------------------------------------------------------------------
    "RAINFOREST_ALLIANCE": {
        "name": "Rainforest Alliance Certified",
        "short_name": "RA",
        "organization": "Rainforest Alliance",
        "description": (
            "Sustainability certification covering environmental, "
            "social, and economic practices for tropical agriculture."
        ),
        "applicable_commodities": [
            "cocoa", "coffee", "palm_oil", "rubber",
        ],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 3,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": (
            "Rainforest Alliance 2020 standard includes no-deforestation "
            "requirements aligned with EUDR Article 3 definitions."
        ),
        "certification_body_accreditation": "RA Certification Bodies",
        "hierarchy_level": "primary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "production",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative", "processor"],
        "key_requirements": [
            "No deforestation (HCV/HCS)",
            "Biodiversity conservation",
            "Climate-smart agriculture",
            "Living income and wages",
            "Human rights and working conditions",
            "Traceability and transparency",
        ],
    },
    # ------------------------------------------------------------------
    # UTZ (merged with Rainforest Alliance, legacy standard)
    # ------------------------------------------------------------------
    "UTZ": {
        "name": "UTZ Certified",
        "short_name": "UTZ",
        "organization": "UTZ (now Rainforest Alliance)",
        "description": (
            "Legacy sustainability certification for cocoa, coffee, "
            "and tea. Merged with Rainforest Alliance in 2018."
        ),
        "applicable_commodities": ["cocoa", "coffee"],
        "validity_period_months": 12,
        "renewal_required": True,
        "renewal_lead_time_months": 3,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "legacy_accepted",
        "eudr_acceptance_notes": (
            "UTZ certificates issued before 2021 may still be valid "
            "but should be transitioned to RA 2020 standard."
        ),
        "certification_body_accreditation": "RA Certification Bodies",
        "hierarchy_level": "legacy",
        "parent_standard": "RAINFOREST_ALLIANCE",
        "child_standards": [],
        "scope": "production",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative"],
        "key_requirements": [
            "Good agricultural practices",
            "Farm management",
            "Working conditions",
            "Environmental management",
        ],
    },
    # ------------------------------------------------------------------
    # Fairtrade
    # ------------------------------------------------------------------
    "FAIRTRADE": {
        "name": "Fairtrade Certified",
        "short_name": "FT",
        "organization": "Fairtrade International",
        "description": (
            "Fair trade certification ensuring minimum prices, "
            "premium payments, and social standards."
        ),
        "applicable_commodities": [
            "cocoa", "coffee", "rubber",
        ],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 3,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "partially_accepted",
        "eudr_acceptance_notes": (
            "Fairtrade includes social and economic criteria but "
            "deforestation-free requirements may need supplementation."
        ),
        "certification_body_accreditation": "FLOCERT",
        "hierarchy_level": "primary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "production_and_trade",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative", "trader"],
        "key_requirements": [
            "Fairtrade minimum price",
            "Fairtrade premium",
            "Democratic organization",
            "Labor conditions",
            "Environmental protection",
            "Traceability",
        ],
    },
    # ------------------------------------------------------------------
    # ISO 14001
    # ------------------------------------------------------------------
    "ISO_14001": {
        "name": "ISO 14001 Environmental Management System",
        "short_name": "ISO 14001",
        "organization": "International Organization for Standardization",
        "description": (
            "International standard for environmental management "
            "systems (EMS) applicable to any organization."
        ),
        "applicable_commodities": [
            "cocoa", "coffee", "palm_oil", "soya", "rubber", "cattle", "wood",
        ],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "supplementary",
        "eudr_acceptance_notes": (
            "ISO 14001 demonstrates environmental management capability "
            "but does not directly address deforestation-free requirements."
        ),
        "certification_body_accreditation": "IAF member bodies",
        "hierarchy_level": "supplementary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "environmental_management",
        "geographic_coverage": "global",
        "supply_chain_tiers": [
            "processor", "trader", "manufacturer", "refinery",
        ],
        "key_requirements": [
            "Environmental policy",
            "Aspects and impacts identification",
            "Legal and regulatory compliance",
            "Objectives and targets",
            "Operational control",
            "Monitoring and measurement",
            "Continual improvement",
        ],
    },
    # ------------------------------------------------------------------
    # Organic Certification (EU)
    # ------------------------------------------------------------------
    "ORGANIC_EU": {
        "name": "EU Organic Certification",
        "short_name": "EU Organic",
        "organization": "European Commission",
        "description": (
            "European Union organic production certification under "
            "Regulation (EU) 2018/848."
        ),
        "applicable_commodities": [
            "cocoa", "coffee", "palm_oil", "soya",
        ],
        "validity_period_months": 12,
        "renewal_required": True,
        "renewal_lead_time_months": 3,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "partially_accepted",
        "eudr_acceptance_notes": (
            "Organic certification ensures production practices but "
            "does not explicitly verify deforestation-free status."
        ),
        "certification_body_accreditation": "EU-recognized control bodies",
        "hierarchy_level": "supplementary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "production_practices",
        "geographic_coverage": "eu_plus_equivalent",
        "supply_chain_tiers": ["farmer", "cooperative", "processor"],
        "key_requirements": [
            "No synthetic pesticides or fertilizers",
            "No GMOs",
            "Crop rotation",
            "Animal welfare (where applicable)",
            "Soil and water conservation",
        ],
    },
    # ------------------------------------------------------------------
    # Organic Certification (USDA)
    # ------------------------------------------------------------------
    "ORGANIC_USDA": {
        "name": "USDA Organic Certification",
        "short_name": "USDA Organic",
        "organization": "United States Department of Agriculture",
        "description": (
            "US organic certification under the National Organic Program."
        ),
        "applicable_commodities": [
            "cocoa", "coffee", "soya",
        ],
        "validity_period_months": 12,
        "renewal_required": True,
        "renewal_lead_time_months": 3,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "partially_accepted",
        "eudr_acceptance_notes": (
            "USDA Organic has equivalency with EU Organic for certain "
            "products. Same EUDR limitations apply."
        ),
        "certification_body_accreditation": "USDA NOP accredited certifiers",
        "hierarchy_level": "supplementary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "production_practices",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative", "processor"],
        "key_requirements": [
            "3-year transition period",
            "Organic system plan",
            "Prohibited substances list compliance",
            "Record keeping",
        ],
    },
    # ------------------------------------------------------------------
    # SAN - Sustainable Agriculture Network
    # ------------------------------------------------------------------
    "SAN": {
        "name": "Sustainable Agriculture Network Standard",
        "short_name": "SAN",
        "organization": "Sustainable Agriculture Network",
        "description": (
            "Sustainability standard for tropical and subtropical "
            "agriculture, precursor to RA 2020 standard."
        ),
        "applicable_commodities": [
            "cocoa", "coffee", "palm_oil", "rubber",
        ],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 3,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "legacy_accepted",
        "eudr_acceptance_notes": (
            "SAN certificates should be transitioned to RA 2020. "
            "Legacy certificates provide limited EUDR evidence."
        ),
        "certification_body_accreditation": "SAN accredited bodies",
        "hierarchy_level": "legacy",
        "parent_standard": "RAINFOREST_ALLIANCE",
        "child_standards": [],
        "scope": "production",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative"],
        "key_requirements": [
            "Ecosystem conservation",
            "Wildlife protection",
            "Water resource management",
            "Fair treatment of workers",
            "Occupational health and safety",
            "Community relations",
        ],
    },
    # ------------------------------------------------------------------
    # RTRS - Round Table on Responsible Soy
    # ------------------------------------------------------------------
    "RTRS": {
        "name": "Round Table on Responsible Soy",
        "short_name": "RTRS",
        "organization": "RTRS Association",
        "description": (
            "Certification for responsible soy production ensuring "
            "legal compliance, environmental and social responsibility."
        ),
        "applicable_commodities": ["soya"],
        "validity_period_months": 60,
        "renewal_required": True,
        "renewal_lead_time_months": 6,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "accepted",
        "eudr_acceptance_notes": (
            "RTRS includes zero-deforestation commitments aligned "
            "with EUDR requirements for soy supply chains."
        ),
        "certification_body_accreditation": "RTRS accredited bodies",
        "hierarchy_level": "primary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "production",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "cooperative", "silo", "crusher"],
        "key_requirements": [
            "Legal compliance and good business practices",
            "Responsible labor conditions",
            "Responsible community relations",
            "Environmental responsibility",
            "Good agricultural practices",
            "Zero conversion after May 2009 cutoff",
        ],
    },
    # ------------------------------------------------------------------
    # Bonsucro
    # ------------------------------------------------------------------
    "BONSUCRO": {
        "name": "Bonsucro Production Standard",
        "short_name": "Bonsucro",
        "organization": "Bonsucro Ltd",
        "description": (
            "Global sustainability certification for sugarcane and "
            "derived products including bioethanol."
        ),
        "applicable_commodities": ["soya"],
        "validity_period_months": 36,
        "renewal_required": True,
        "renewal_lead_time_months": 4,
        "annual_audit_required": True,
        "surveillance_audit_interval_months": 12,
        "eudr_acceptance": "supplementary",
        "eudr_acceptance_notes": (
            "Bonsucro primarily covers sugarcane but relevant for "
            "integrated soy-sugarcane operations in Brazil."
        ),
        "certification_body_accreditation": "Bonsucro accredited bodies",
        "hierarchy_level": "supplementary",
        "parent_standard": None,
        "child_standards": [],
        "scope": "production",
        "geographic_coverage": "global",
        "supply_chain_tiers": ["farmer", "mill", "processor"],
        "key_requirements": [
            "Legal compliance",
            "Human rights and labor",
            "Efficient production management",
            "Biodiversity and ecosystem services",
            "Continuous improvement",
        ],
    },
}

# Totals
TOTAL_CERTIFICATIONS: int = len(CERTIFICATION_STANDARDS)


# ---------------------------------------------------------------------------
# EUDR Acceptance Levels
# ---------------------------------------------------------------------------

EUDR_ACCEPTANCE_LEVELS: Dict[str, Dict[str, Any]] = {
    "accepted": {
        "label": "Accepted",
        "description": (
            "Certification is accepted as supporting evidence for EUDR "
            "due diligence. Reduces but does not replace operator obligations."
        ),
        "risk_reduction_pct": 30,
        "dds_contribution": True,
    },
    "partially_accepted": {
        "label": "Partially Accepted",
        "description": (
            "Certification provides partial evidence. Additional verification "
            "needed for full EUDR compliance."
        ),
        "risk_reduction_pct": 15,
        "dds_contribution": True,
    },
    "supplementary": {
        "label": "Supplementary",
        "description": (
            "Certification demonstrates good practices but does not directly "
            "address EUDR deforestation-free requirements."
        ),
        "risk_reduction_pct": 5,
        "dds_contribution": False,
    },
    "legacy_accepted": {
        "label": "Legacy Accepted",
        "description": (
            "Historical certification that has been superseded. Valid "
            "certificates may still be recognized but transition recommended."
        ),
        "risk_reduction_pct": 10,
        "dds_contribution": True,
    },
    "not_accepted": {
        "label": "Not Accepted",
        "description": (
            "Certification is not recognized for EUDR due diligence purposes."
        ),
        "risk_reduction_pct": 0,
        "dds_contribution": False,
    },
}


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------


def get_certification(cert_key: str) -> Optional[Dict[str, Any]]:
    """Get the full certification standard definition.

    Args:
        cert_key: Certification key (e.g. 'FSC_FM', 'RSPO_PC').

    Returns:
        Dictionary with certification details or None if not found.

    Example:
        >>> cert = get_certification("FSC_FM")
        >>> assert cert["validity_period_months"] == 60
    """
    return CERTIFICATION_STANDARDS.get(cert_key.upper()) if cert_key else None


def is_eudr_accepted(cert_key: str) -> bool:
    """Check whether a certification is accepted for EUDR due diligence.

    Args:
        cert_key: Certification key.

    Returns:
        True if the certification has acceptance level 'accepted' or
        'partially_accepted', False otherwise.

    Example:
        >>> assert is_eudr_accepted("FSC_FM") is True
        >>> assert is_eudr_accepted("ISO_14001") is False
    """
    cert = get_certification(cert_key)
    if cert is None:
        return False
    return cert.get("eudr_acceptance", "") in ("accepted", "partially_accepted")


def get_certifications_for_commodity(
    commodity: str,
) -> List[Dict[str, Any]]:
    """Get all certification standards applicable to a commodity.

    Args:
        commodity: EUDR commodity key (e.g. 'cocoa', 'palm_oil').

    Returns:
        List of certification dictionaries applicable to the commodity,
        sorted by EUDR acceptance level (accepted first).

    Example:
        >>> certs = get_certifications_for_commodity("palm_oil")
        >>> assert any(c["key"] == "RSPO_PC" for c in certs)
    """
    commodity_lower = commodity.lower()
    acceptance_order = {
        "accepted": 0,
        "partially_accepted": 1,
        "legacy_accepted": 2,
        "supplementary": 3,
        "not_accepted": 4,
    }
    results = []
    for key, cert in CERTIFICATION_STANDARDS.items():
        if commodity_lower in cert.get("applicable_commodities", []):
            results.append({
                "key": key,
                "name": cert["name"],
                "short_name": cert["short_name"],
                "eudr_acceptance": cert["eudr_acceptance"],
                "validity_period_months": cert["validity_period_months"],
                "organization": cert["organization"],
            })
    results.sort(
        key=lambda x: acceptance_order.get(x["eudr_acceptance"], 99),
    )
    return results


def check_validity(
    cert_key: str,
    issue_date_iso: str,
    current_date_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Check whether a certification is currently valid based on dates.

    Args:
        cert_key: Certification key.
        issue_date_iso: Certificate issue date in ISO format (YYYY-MM-DD).
        current_date_iso: Current date in ISO format. If None, uses today.

    Returns:
        Dictionary with validity status, days remaining, renewal required,
        and renewal deadline.

    Example:
        >>> result = check_validity("FSC_FM", "2024-01-15", "2026-03-08")
        >>> assert "is_valid" in result
    """
    from datetime import date, timedelta

    cert = get_certification(cert_key)
    if cert is None:
        return {
            "is_valid": False,
            "error": f"Unknown certification: {cert_key}",
        }

    try:
        issue = date.fromisoformat(issue_date_iso)
    except (ValueError, TypeError):
        return {
            "is_valid": False,
            "error": f"Invalid issue date: {issue_date_iso}",
        }

    if current_date_iso:
        try:
            current = date.fromisoformat(current_date_iso)
        except (ValueError, TypeError):
            current = date.today()
    else:
        current = date.today()

    validity_months = cert["validity_period_months"]
    expiry = issue + timedelta(days=validity_months * 30)
    days_remaining = (expiry - current).days
    is_valid = days_remaining > 0

    renewal_lead_months = cert.get("renewal_lead_time_months", 3)
    renewal_deadline = expiry - timedelta(days=renewal_lead_months * 30)
    renewal_overdue = current > renewal_deadline and is_valid

    return {
        "is_valid": is_valid,
        "cert_key": cert_key,
        "issue_date": issue_date_iso,
        "expiry_date": expiry.isoformat(),
        "days_remaining": max(0, days_remaining),
        "renewal_required": cert.get("renewal_required", True),
        "renewal_deadline": renewal_deadline.isoformat(),
        "renewal_overdue": renewal_overdue,
        "eudr_acceptance": cert["eudr_acceptance"],
    }


def get_certification_hierarchy(cert_key: str) -> Dict[str, Any]:
    """Get the certification hierarchy (parent/child relationships).

    Args:
        cert_key: Certification key.

    Returns:
        Dictionary with parent, children, and hierarchy level.

    Example:
        >>> hierarchy = get_certification_hierarchy("FSC_COC")
        >>> assert hierarchy["parent"] == "FSC_FM"
    """
    cert = get_certification(cert_key)
    if cert is None:
        return {"error": f"Unknown certification: {cert_key}"}

    parent_key = cert.get("parent_standard")
    parent_data = None
    if parent_key:
        parent_cert = get_certification(parent_key)
        if parent_cert:
            parent_data = {
                "key": parent_key,
                "name": parent_cert["name"],
                "short_name": parent_cert["short_name"],
            }

    children = []
    for child_key in cert.get("child_standards", []):
        child_cert = get_certification(child_key)
        if child_cert:
            children.append({
                "key": child_key,
                "name": child_cert["name"],
                "short_name": child_cert["short_name"],
            })

    return {
        "key": cert_key,
        "name": cert["name"],
        "hierarchy_level": cert["hierarchy_level"],
        "parent": parent_key,
        "parent_data": parent_data,
        "children": [c["key"] for c in children],
        "children_data": children,
    }


def get_all_certifications() -> List[Dict[str, Any]]:
    """Get a summary list of all certification standards.

    Returns:
        List of certification summaries sorted alphabetically by name.

    Example:
        >>> certs = get_all_certifications()
        >>> assert len(certs) >= 10
    """
    results = []
    for key, cert in CERTIFICATION_STANDARDS.items():
        results.append({
            "key": key,
            "name": cert["name"],
            "short_name": cert["short_name"],
            "organization": cert["organization"],
            "applicable_commodities": cert["applicable_commodities"],
            "eudr_acceptance": cert["eudr_acceptance"],
            "validity_period_months": cert["validity_period_months"],
            "hierarchy_level": cert["hierarchy_level"],
        })
    results.sort(key=lambda x: x["name"])
    return results
