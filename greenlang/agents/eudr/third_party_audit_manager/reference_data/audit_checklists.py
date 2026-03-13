# -*- coding: utf-8 -*-
"""
Audit Checklists Reference Data - AGENT-EUDR-024

Static audit checklist criteria for EUDR compliance and certification
scheme audits. Each checklist contains structured criteria items with
article/clause references, descriptions, and assessment guidance.

Checklist Types (6):
    - EUDR: 17 criteria covering Articles 3, 4, 9-11, 29, 31
    - FSC: 12 criteria covering Principles 1-10 and CoC
    - PEFC: 8 criteria covering Criteria 1-6 and CoC
    - RSPO: 8 criteria covering Principles 1-7 and SC
    - RA: 7 criteria covering Chapters 1-6 and SC
    - ISCC: 8 criteria covering Principles 1-6 and SC

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# EUDR Compliance Checklist (17 criteria)
# ---------------------------------------------------------------------------

EUDR_CHECKLIST_CRITERIA: List[Dict[str, Any]] = [
    {
        "criterion_id": "EUDR-001",
        "article": "Art. 3",
        "title": "Prohibition of non-compliant products",
        "description": "Verify no deforestation-linked products placed on or exported from EU market after 31 December 2020 cutoff date",
        "guidance": "Review product sourcing records, deforestation assessment reports, and satellite monitoring data",
        "severity_if_failed": "critical",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-002",
        "article": "Art. 4",
        "title": "Due diligence obligation",
        "description": "Verify operator has implemented a complete due diligence system covering information collection, risk assessment, and risk mitigation",
        "guidance": "Review DDS documentation, procedures, and organizational assignments",
        "severity_if_failed": "critical",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-003",
        "article": "Art. 9(1)(a)",
        "title": "Product description",
        "description": "Verify product description including trade name and common name of product and commodity",
        "guidance": "Cross-reference product catalogs with customs documentation",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-004",
        "article": "Art. 9(1)(b)",
        "title": "Quantity",
        "description": "Verify quantity including net mass and volume where applicable",
        "guidance": "Review purchase orders, delivery notes, and inventory records",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-005",
        "article": "Art. 9(1)(c)",
        "title": "Country of production",
        "description": "Verify country of production is identified and documented",
        "guidance": "Review origin certificates, supplier declarations, and customs data",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-006",
        "article": "Art. 9(1)(d)",
        "title": "Geolocation of production plots",
        "description": "Verify geolocation coordinates of all plots of land where the commodity was produced",
        "guidance": "Review GPS coordinates, polygon data, and geolocation verification reports",
        "severity_if_failed": "critical",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-007",
        "article": "Art. 9(1)(e)",
        "title": "Production period",
        "description": "Verify date or time range of production is recorded",
        "guidance": "Review harvest records, processing dates, and batch documentation",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-008",
        "article": "Art. 9(1)(f)",
        "title": "Supplier details",
        "description": "Verify supplier name, address, email address, and phone number",
        "guidance": "Review supplier registry, contracts, and communication records",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-009",
        "article": "Art. 9(1)(g)",
        "title": "Buyer details",
        "description": "Verify buyer name, address, email address, and phone number",
        "guidance": "Review customer registry and sales documentation",
        "severity_if_failed": "minor",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-010",
        "article": "Art. 10(1)",
        "title": "Risk assessment adequacy",
        "description": "Verify risk assessment covers all relevant criteria including country risk, complexity, deforestation risk",
        "guidance": "Review risk assessment methodology, criteria coverage, and outputs",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-011",
        "article": "Art. 10(2)",
        "title": "Country benchmarking",
        "description": "Verify country benchmarking is incorporated in risk assessment using official EU classification",
        "guidance": "Review country risk scores, EU benchmarking data usage, and risk categorization",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-012",
        "article": "Art. 10(3)",
        "title": "Supply chain complexity assessment",
        "description": "Verify supply chain complexity is assessed including multi-tier considerations",
        "guidance": "Review supply chain mapping depth, tier identification, and complexity scoring",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-013",
        "article": "Art. 10(4)",
        "title": "Deforestation risk evaluation",
        "description": "Verify deforestation and forest degradation risk is specifically evaluated",
        "guidance": "Review satellite monitoring data, alert systems, and deforestation assessment reports",
        "severity_if_failed": "critical",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-014",
        "article": "Art. 11(1)",
        "title": "Risk mitigation measures",
        "description": "Verify adequate risk mitigation measures are implemented where risk is not negligible",
        "guidance": "Review mitigation plans, implementation evidence, and effectiveness monitoring",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-015",
        "article": "Art. 11(2)",
        "title": "Risk mitigation documentation",
        "description": "Verify risk mitigation is documented and traceable to specific risks identified",
        "guidance": "Review mitigation documentation, traceability records, and verification evidence",
        "severity_if_failed": "minor",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-016",
        "article": "Art. 29",
        "title": "Record keeping",
        "description": "Verify records are maintained for at least 5 years as required by Article 29",
        "guidance": "Review record retention policies, archival systems, and sample historical records",
        "severity_if_failed": "major",
        "mandatory": True,
    },
    {
        "criterion_id": "EUDR-017",
        "article": "Art. 31",
        "title": "Audit trail",
        "description": "Verify complete audit trail is maintained per Article 31 requirements",
        "guidance": "Review audit trail systems, provenance tracking, and data integrity controls",
        "severity_if_failed": "major",
        "mandatory": True,
    },
]

# ---------------------------------------------------------------------------
# FSC Checklist (12 criteria)
# ---------------------------------------------------------------------------

FSC_CHECKLIST_CRITERIA: List[Dict[str, Any]] = [
    {"criterion_id": "FSC-P1", "title": "Legal compliance", "description": "Compliance with all applicable laws and regulations", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "FSC-P2", "title": "Workers rights", "description": "Workers rights and employment conditions", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "FSC-P3", "title": "Indigenous peoples", "description": "Indigenous peoples rights respected", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "FSC-P4", "title": "Community relations", "description": "Community relations and workers rights maintained", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "FSC-P5", "title": "Benefits from forest", "description": "Efficient use and benefits from the forest", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "FSC-P6", "title": "Environmental values", "description": "Environmental values and impacts maintained", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "FSC-P7", "title": "Management plan", "description": "Management plan documented and implemented", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "FSC-P8", "title": "Monitoring", "description": "Monitoring and assessment conducted", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "FSC-P9", "title": "High conservation", "description": "High conservation value areas maintained", "severity_if_failed": "critical", "mandatory": True},
    {"criterion_id": "FSC-P10", "title": "Plantations", "description": "Plantation management meets criteria", "severity_if_failed": "minor", "mandatory": False},
    {"criterion_id": "FSC-CoC1", "title": "Chain of custody", "description": "Material identification and traceability", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "FSC-CoC2", "title": "Product groups", "description": "Product group management", "severity_if_failed": "minor", "mandatory": True},
]

# ---------------------------------------------------------------------------
# PEFC Checklist (8 criteria)
# ---------------------------------------------------------------------------

PEFC_CHECKLIST_CRITERIA: List[Dict[str, Any]] = [
    {"criterion_id": "PEFC-C1", "title": "Forest management policy", "description": "Forest management policy and objectives documented", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "PEFC-C2", "title": "Legal compliance", "description": "Legal and regulatory compliance verified", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "PEFC-C3", "title": "Productive functions", "description": "Maintenance and enhancement of productive functions", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "PEFC-C4", "title": "Biodiversity", "description": "Maintenance and conservation of biodiversity", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "PEFC-C5", "title": "Protective functions", "description": "Maintenance of protective functions", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "PEFC-C6", "title": "Socio-economic", "description": "Maintenance of socio-economic functions", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "PEFC-CoC1", "title": "Chain of custody", "description": "PEFC chain of custody management system", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "PEFC-CoC2", "title": "Due diligence", "description": "PEFC due diligence system implemented", "severity_if_failed": "major", "mandatory": True},
]

# ---------------------------------------------------------------------------
# RSPO Checklist (8 criteria)
# ---------------------------------------------------------------------------

RSPO_CHECKLIST_CRITERIA: List[Dict[str, Any]] = [
    {"criterion_id": "RSPO-P1", "title": "Transparency", "description": "Commitment to transparency demonstrated", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "RSPO-P2", "title": "Laws and regulations", "description": "Compliance with applicable laws and regulations", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "RSPO-P3", "title": "Economic viability", "description": "Commitment to long-term economic viability", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "RSPO-P4", "title": "Best practices", "description": "Best practices by growers and millers", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "RSPO-P5", "title": "Environmental", "description": "Environmental responsibility and conservation", "severity_if_failed": "critical", "mandatory": True},
    {"criterion_id": "RSPO-P6", "title": "Workers and communities", "description": "Responsible consideration of employees and communities", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "RSPO-P7", "title": "New plantings", "description": "Responsible development of new plantings", "severity_if_failed": "critical", "mandatory": True},
    {"criterion_id": "RSPO-SC1", "title": "Supply chain", "description": "Supply chain certification and traceability", "severity_if_failed": "major", "mandatory": True},
]

# ---------------------------------------------------------------------------
# Rainforest Alliance Checklist (7 criteria)
# ---------------------------------------------------------------------------

RA_CHECKLIST_CRITERIA: List[Dict[str, Any]] = [
    {"criterion_id": "RA-C1", "title": "Management", "description": "Management system and documentation", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "RA-C2", "title": "Traceability", "description": "Traceability and chain of custody", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "RA-C3", "title": "Forests", "description": "Forest and ecosystem conservation", "severity_if_failed": "critical", "mandatory": True},
    {"criterion_id": "RA-C4", "title": "Climate", "description": "Climate change mitigation and adaptation", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "RA-C5", "title": "Human rights", "description": "Human rights and working conditions", "severity_if_failed": "critical", "mandatory": True},
    {"criterion_id": "RA-C6", "title": "Livelihoods", "description": "Improved livelihoods and human well-being", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "RA-SC1", "title": "Supply chain", "description": "Supply chain traceability requirements", "severity_if_failed": "major", "mandatory": True},
]

# ---------------------------------------------------------------------------
# ISCC Checklist (8 criteria)
# ---------------------------------------------------------------------------

ISCC_CHECKLIST_CRITERIA: List[Dict[str, Any]] = [
    {"criterion_id": "ISCC-P1", "title": "Biomass protection", "description": "Protection of land with high biodiversity value", "severity_if_failed": "critical", "mandatory": True},
    {"criterion_id": "ISCC-P2", "title": "Sustainable production", "description": "Environmentally responsible production", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "ISCC-P3", "title": "Safe working conditions", "description": "Safe working conditions ensured", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "ISCC-P4", "title": "Laws and international treaties", "description": "Compliance with laws and international treaties", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "ISCC-P5", "title": "Good management", "description": "Good management practices implemented", "severity_if_failed": "minor", "mandatory": True},
    {"criterion_id": "ISCC-P6", "title": "GHG emissions", "description": "GHG emission monitoring and reduction", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "ISCC-SC1", "title": "Chain of custody", "description": "Chain of custody management system", "severity_if_failed": "major", "mandatory": True},
    {"criterion_id": "ISCC-SC2", "title": "Mass balance", "description": "Mass balance system implemented and verified", "severity_if_failed": "major", "mandatory": True},
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

_CHECKLISTS: Dict[str, List[Dict[str, Any]]] = {
    "eudr": EUDR_CHECKLIST_CRITERIA,
    "fsc": FSC_CHECKLIST_CRITERIA,
    "pefc": PEFC_CHECKLIST_CRITERIA,
    "rspo": RSPO_CHECKLIST_CRITERIA,
    "ra": RA_CHECKLIST_CRITERIA,
    "iscc": ISCC_CHECKLIST_CRITERIA,
}


def get_checklist_by_type(checklist_type: str) -> List[Dict[str, Any]]:
    """Get checklist criteria by type.

    Args:
        checklist_type: Checklist type (eudr, fsc, pefc, rspo, ra, iscc).

    Returns:
        List of criteria dictionaries.

    Raises:
        ValueError: If checklist type is not supported.
    """
    criteria = _CHECKLISTS.get(checklist_type.lower())
    if criteria is None:
        raise ValueError(
            f"Unsupported checklist type: {checklist_type}. "
            f"Must be one of {list(_CHECKLISTS.keys())}"
        )
    return criteria
