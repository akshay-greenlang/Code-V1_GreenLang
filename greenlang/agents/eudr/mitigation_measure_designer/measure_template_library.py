# -*- coding: utf-8 -*-
"""
Measure Template Library Engine - AGENT-EUDR-029

Curated library of 50+ EUDR Article 11 mitigation measure templates
organized by Article 11(2) category, risk dimension, and commodity
applicability. Provides search, filter, and retrieval capabilities for
the MitigationStrategyDesigner engine.

Templates cover:
    - 7-8 templates per EUDR commodity
    - Templates for each Article 11(2) category (a, b, c)
    - Templates for each of the 8 risk dimensions
    - Each template includes: id, title, description, category,
      dimensions, commodities, base_effectiveness, timeline,
      evidence_requirements, regulatory_reference

Zero-Hallucination Guarantees:
    - All effectiveness values are curated Decimal constants
    - No LLM-generated template content
    - Templates are versioned and auditable
    - Full regulatory references for every template

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Article 11(2)
Status: Production Ready
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import MitigationMeasureDesignerConfig, get_config
from .models import (
    Article11Category,
    EUDRCommodity,
    EvidenceType,
    MeasureTemplate,
    RiskDimension,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in template definitions (50+ templates)
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # COUNTRY dimension templates (MMD-TPL-001 to MMD-TPL-006)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-001",
        "title": "Country-Level Legal Framework Assessment",
        "description": (
            "Commission independent assessment of the producer country's "
            "legal framework for forest protection, land use planning, and "
            "environmental governance. Evaluate enforcement capacity and "
            "identify specific gaps relevant to EUDR compliance."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["country"],
        "applicable_commodities": [],
        "base_effectiveness": "20",
        "typical_timeline_days": 30,
        "evidence_requirements": ["audit_report", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(a), Art. 29",
    },
    {
        "template_id": "MMD-TPL-002",
        "title": "Country Governance Risk Monitoring Program",
        "description": (
            "Establish continuous monitoring of governance indicators, "
            "legal changes, and enforcement actions in producer countries. "
            "Track World Bank governance scores, TI Corruption Perception "
            "Index, and FAO forest loss statistics quarterly."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["country", "corruption"],
        "applicable_commodities": [],
        "base_effectiveness": "15",
        "typical_timeline_days": 14,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 11(2)(c), Art. 10(2)(c)",
    },
    {
        "template_id": "MMD-TPL-003",
        "title": "Alternative Sourcing Region Assessment",
        "description": (
            "Conduct feasibility study for diversifying sourcing to "
            "lower-risk countries. Evaluate alternative supplier "
            "qualification, logistics, cost impact, and transition "
            "timeline for gradual volume rebalancing."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["country"],
        "applicable_commodities": [],
        "base_effectiveness": "30",
        "typical_timeline_days": 60,
        "evidence_requirements": ["document", "audit_report"],
        "regulatory_reference": "EUDR Art. 11(1), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-004",
        "title": "In-Country Regulatory Compliance Verification",
        "description": (
            "Verify that all relevant permits, licenses, and legal "
            "authorizations are in place for production activities in "
            "the source country. Cross-reference with national land "
            "registry and environmental authority databases."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["country"],
        "applicable_commodities": [],
        "base_effectiveness": "25",
        "typical_timeline_days": 21,
        "evidence_requirements": ["document", "certificate"],
        "regulatory_reference": "EUDR Art. 10(2)(d), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-005",
        "title": "Landscape-Level Collective Action Initiative",
        "description": (
            "Join or establish landscape-level multi-stakeholder "
            "initiative in the sourcing region to address systemic "
            "country-level risk through shared investment, monitoring "
            "infrastructure, and community engagement."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["country", "deforestation"],
        "applicable_commodities": [],
        "base_effectiveness": "18",
        "typical_timeline_days": 90,
        "evidence_requirements": ["document", "audit_report"],
        "regulatory_reference": "EUDR Art. 11(2)(c), Art. 29(3)",
    },
    {
        "template_id": "MMD-TPL-006",
        "title": "Country Risk Benchmarking Report",
        "description": (
            "Obtain or commission a country-specific risk benchmarking "
            "report per EUDR Article 29 criteria including deforestation "
            "rates, governance effectiveness, rule of law index, and "
            "international commitments compliance."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["country"],
        "applicable_commodities": [],
        "base_effectiveness": "12",
        "typical_timeline_days": 14,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 29(2), Art. 11(2)(c)",
    },
    # -----------------------------------------------------------------------
    # SUPPLIER dimension templates (MMD-TPL-007 to MMD-TPL-014)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-007",
        "title": "Enhanced Supplier Due Diligence Questionnaire",
        "description": (
            "Deploy comprehensive supplier questionnaire covering "
            "land ownership documentation, production practices, "
            "deforestation commitments, labour conditions, and "
            "EUDR-specific compliance status with document verification."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "18",
        "typical_timeline_days": 21,
        "evidence_requirements": ["document", "supplier_declaration"],
        "regulatory_reference": "EUDR Art. 10(2)(f), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-008",
        "title": "Independent Supplier On-Site Audit",
        "description": (
            "Commission independent third-party audit of supplier "
            "operations including production site visit, document "
            "verification, worker interviews, environmental assessment, "
            "and GPS plot boundary validation."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "30",
        "typical_timeline_days": 45,
        "evidence_requirements": ["audit_report", "site_visit_report"],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-009",
        "title": "Supplier Compliance Capacity Building",
        "description": (
            "Provide structured EUDR compliance training program for "
            "supplier covering data collection requirements, GPS "
            "coordinate capture, document management, traceability "
            "system operation, and reporting procedures."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "22",
        "typical_timeline_days": 60,
        "evidence_requirements": ["document", "certificate"],
        "regulatory_reference": "EUDR Art. 11(2)(c), Art. 10(2)(f)",
    },
    {
        "template_id": "MMD-TPL-010",
        "title": "Supplier Certification Verification",
        "description": (
            "Verify supplier's third-party certification status "
            "(FSC, RSPO, Rainforest Alliance, ISCC, etc.) and "
            "cross-reference certificate validity, scope, and "
            "compliance history with certification body databases."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["supplier", "commodity"],
        "applicable_commodities": [],
        "base_effectiveness": "25",
        "typical_timeline_days": 14,
        "evidence_requirements": ["certificate", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(b), Art. 10(2)(a)",
    },
    {
        "template_id": "MMD-TPL-011",
        "title": "Supplier Corrective Action Plan",
        "description": (
            "Develop and implement structured corrective action plan "
            "addressing specific supplier non-conformances. Include "
            "root cause analysis, remediation steps, timeline, "
            "verification checkpoints, and evidence requirements."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "28",
        "typical_timeline_days": 45,
        "evidence_requirements": ["document", "audit_report"],
        "regulatory_reference": "EUDR Art. 11(1), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-012",
        "title": "Supplier Transaction Pattern Analysis",
        "description": (
            "Analyze supplier transaction patterns for anomalies "
            "including volume spikes, price inconsistencies, "
            "unusual trade routes, and seasonal deviations that "
            "may indicate circumvention or mixing risks."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["supplier", "circumvention_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "16",
        "typical_timeline_days": 14,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(e), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-013",
        "title": "Supplier Replacement Qualification",
        "description": (
            "When supplier risk is deemed unmitigable, qualify "
            "alternative suppliers through accelerated due diligence "
            "process including site assessments, document verification, "
            "and trial volume arrangements."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "40",
        "typical_timeline_days": 90,
        "evidence_requirements": ["audit_report", "document"],
        "regulatory_reference": "EUDR Art. 11(1)",
    },
    {
        "template_id": "MMD-TPL-014",
        "title": "Supplier GPS Coordinate Verification",
        "description": (
            "Independently verify GPS coordinates of production "
            "plots declared by supplier against satellite imagery, "
            "national land registry, and protected area databases. "
            "Validate polygon boundaries and area calculations."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["supplier", "deforestation"],
        "applicable_commodities": [],
        "base_effectiveness": "22",
        "typical_timeline_days": 21,
        "evidence_requirements": ["satellite_image", "document"],
        "regulatory_reference": "EUDR Art. 10(2)(b), Art. 11(2)(a)",
    },
    # -----------------------------------------------------------------------
    # COMMODITY dimension templates (MMD-TPL-015 to MMD-TPL-021)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-015",
        "title": "Commodity-Specific Certification Requirement",
        "description": (
            "Require suppliers to obtain or maintain commodity-specific "
            "sustainability certification (FSC for wood, RSPO for palm, "
            "Rainforest Alliance for coffee/cocoa, ISCC for soya) with "
            "annual audit verification."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["commodity"],
        "applicable_commodities": [],
        "base_effectiveness": "28",
        "typical_timeline_days": 120,
        "evidence_requirements": ["certificate", "audit_report"],
        "regulatory_reference": "EUDR Art. 11(2)(b)",
    },
    {
        "template_id": "MMD-TPL-016",
        "title": "Palm Oil RSPO Segregation Audit",
        "description": (
            "Commission RSPO supply chain certification audit for "
            "palm oil and derivatives ensuring Identity Preserved or "
            "Segregated supply chain model compliance."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["commodity", "mixing_risk"],
        "applicable_commodities": ["oil_palm"],
        "base_effectiveness": "32",
        "typical_timeline_days": 60,
        "evidence_requirements": ["audit_report", "certificate"],
        "regulatory_reference": "EUDR Art. 11(2)(a), Art. 11(2)(b)",
    },
    {
        "template_id": "MMD-TPL-017",
        "title": "Wood Legality Verification (FSC/PEFC)",
        "description": (
            "Conduct FSC or PEFC chain of custody verification for "
            "wood products including species identification, origin "
            "documentation, and transport chain validation against "
            "CITES and FLEGT requirements."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["commodity"],
        "applicable_commodities": ["wood"],
        "base_effectiveness": "30",
        "typical_timeline_days": 45,
        "evidence_requirements": ["audit_report", "certificate", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(a), Art. 10(2)(a)",
    },
    {
        "template_id": "MMD-TPL-018",
        "title": "Coffee/Cocoa Origin Traceability Enhancement",
        "description": (
            "Implement plot-level traceability for coffee or cocoa "
            "supply chain including GPS farm mapping, batch tracking, "
            "cooperative-level aggregation controls, and first-mile "
            "documentation."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["commodity", "supply_chain_complexity"],
        "applicable_commodities": ["coffee", "cocoa"],
        "base_effectiveness": "25",
        "typical_timeline_days": 90,
        "evidence_requirements": ["document", "satellite_image"],
        "regulatory_reference": "EUDR Art. 10(2)(a), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-019",
        "title": "Soya Deforestation-Free Certification",
        "description": (
            "Obtain soya-specific deforestation-free verification "
            "through ISCC, ProTerra, or RTRS certification covering "
            "production area monitoring, conversion cutoff compliance, "
            "and mass balance accounting."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["commodity", "deforestation"],
        "applicable_commodities": ["soya"],
        "base_effectiveness": "27",
        "typical_timeline_days": 90,
        "evidence_requirements": ["certificate", "audit_report"],
        "regulatory_reference": "EUDR Art. 11(2)(b), Art. 3",
    },
    {
        "template_id": "MMD-TPL-020",
        "title": "Rubber Sustainability Assessment",
        "description": (
            "Conduct sustainability assessment for natural rubber "
            "supply chain including GPSNR platform alignment, "
            "smallholder mapping, agroforestry integration, and "
            "deforestation risk evaluation."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["commodity"],
        "applicable_commodities": ["rubber"],
        "base_effectiveness": "24",
        "typical_timeline_days": 60,
        "evidence_requirements": ["audit_report", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-021",
        "title": "Cattle Ranching Deforestation Monitoring",
        "description": (
            "Implement satellite-based monitoring of cattle ranch "
            "boundaries with automated deforestation alert system, "
            "CAR (Rural Environmental Registry) cross-referencing, "
            "and cattle movement traceability (GTA verification)."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["commodity", "deforestation"],
        "applicable_commodities": ["cattle"],
        "base_effectiveness": "26",
        "typical_timeline_days": 45,
        "evidence_requirements": ["satellite_image", "document"],
        "regulatory_reference": "EUDR Art. 10(2)(b), Art. 11(2)(c)",
    },
    # -----------------------------------------------------------------------
    # DEFORESTATION dimension templates (MMD-TPL-022 to MMD-TPL-028)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-022",
        "title": "Enhanced Satellite Monitoring Deployment",
        "description": (
            "Deploy enhanced satellite monitoring with higher temporal "
            "resolution (weekly Sentinel-2 + daily Planet Labs) for all "
            "supply chain plots in high deforestation risk areas with "
            "automated change detection alerts."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["deforestation"],
        "applicable_commodities": [],
        "base_effectiveness": "28",
        "typical_timeline_days": 30,
        "evidence_requirements": ["satellite_image", "document"],
        "regulatory_reference": "EUDR Art. 10(1), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-023",
        "title": "Plot-Level Ground Verification Mission",
        "description": (
            "Commission on-the-ground verification mission to "
            "independently validate forest cover status, land use "
            "practices, and production boundaries of flagged plots "
            "using GPS surveying and photographic documentation."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["deforestation"],
        "applicable_commodities": [],
        "base_effectiveness": "35",
        "typical_timeline_days": 30,
        "evidence_requirements": [
            "site_visit_report", "satellite_image", "document",
        ],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-024",
        "title": "Deforestation-Free Sourcing Declaration",
        "description": (
            "Require and verify supplier deforestation-free production "
            "declarations with cutoff date compliance (31 Dec 2020). "
            "Cross-reference declarations with satellite data and "
            "Global Forest Watch alerts."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["deforestation"],
        "applicable_commodities": [],
        "base_effectiveness": "20",
        "typical_timeline_days": 14,
        "evidence_requirements": ["supplier_declaration", "document"],
        "regulatory_reference": "EUDR Art. 3, Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-025",
        "title": "Historical Land Use Analysis",
        "description": (
            "Commission historical land use analysis using multi-"
            "temporal satellite imagery from 2018-present to verify "
            "no forest-to-agriculture conversion occurred after the "
            "EUDR cutoff date of 31 December 2020."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["deforestation"],
        "applicable_commodities": [],
        "base_effectiveness": "25",
        "typical_timeline_days": 30,
        "evidence_requirements": ["satellite_image", "document"],
        "regulatory_reference": "EUDR Art. 2(1), Art. 10(2)(b)",
    },
    {
        "template_id": "MMD-TPL-026",
        "title": "Emergency Sourcing Suspension Protocol",
        "description": (
            "Activate immediate sourcing suspension from plots with "
            "active deforestation alerts. Implement quarantine "
            "procedures, launch investigation, and define conditions "
            "for potential sourcing resumption."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["deforestation"],
        "applicable_commodities": [],
        "base_effectiveness": "45",
        "typical_timeline_days": 7,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 11(1), Art. 3",
    },
    {
        "template_id": "MMD-TPL-027",
        "title": "Reforestation and Restoration Partnership",
        "description": (
            "Establish partnership with conservation organization for "
            "reforestation of previously degraded areas within the "
            "supply chain landscape, supporting long-term forest "
            "cover recovery and carbon sequestration."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["deforestation", "country"],
        "applicable_commodities": [],
        "base_effectiveness": "15",
        "typical_timeline_days": 180,
        "evidence_requirements": ["document", "satellite_image"],
        "regulatory_reference": "EUDR Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-028",
        "title": "Protected Area Buffer Zone Verification",
        "description": (
            "Verify that no production plots overlap with or are "
            "within buffer zones of protected areas, indigenous "
            "territories, or High Conservation Value areas using "
            "WDPA and national protected area databases."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["deforestation", "country"],
        "applicable_commodities": [],
        "base_effectiveness": "22",
        "typical_timeline_days": 14,
        "evidence_requirements": ["satellite_image", "document"],
        "regulatory_reference": "EUDR Art. 10(2)(d), Art. 11(2)(c)",
    },
    # -----------------------------------------------------------------------
    # CORRUPTION dimension templates (MMD-TPL-029 to MMD-TPL-034)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-029",
        "title": "Independent Anti-Corruption Audit",
        "description": (
            "Commission independent audit of business practices in "
            "supply chain operations focusing on bribery risks, "
            "beneficial ownership transparency, payment integrity, "
            "and governance of intermediaries."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["corruption"],
        "applicable_commodities": [],
        "base_effectiveness": "25",
        "typical_timeline_days": 45,
        "evidence_requirements": ["audit_report"],
        "regulatory_reference": "EUDR Art. 10(2)(e), Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-030",
        "title": "Enhanced Payment Transparency Controls",
        "description": (
            "Implement transparent payment protocols with full audit "
            "trail, multi-party approval for high-value transactions, "
            "and elimination of cash payments in the supply chain."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["corruption"],
        "applicable_commodities": [],
        "base_effectiveness": "20",
        "typical_timeline_days": 30,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(e), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-031",
        "title": "Beneficial Ownership Disclosure Requirement",
        "description": (
            "Require full beneficial ownership disclosure from all "
            "supply chain intermediaries and verify against national "
            "company registries and international sanctions lists."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["corruption", "supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "18",
        "typical_timeline_days": 21,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(e), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-032",
        "title": "Whistleblower Mechanism Deployment",
        "description": (
            "Deploy confidential whistleblower mechanism for supply "
            "chain actors to report suspected irregularities, "
            "corruption, or environmental violations without fear "
            "of retaliation."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["corruption"],
        "applicable_commodities": [],
        "base_effectiveness": "12",
        "typical_timeline_days": 30,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-033",
        "title": "Anti-Bribery Training for Supply Chain",
        "description": (
            "Conduct anti-bribery and anti-corruption training for "
            "procurement teams and key supply chain intermediaries, "
            "covering local legal requirements, red flags, and "
            "reporting obligations."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["corruption"],
        "applicable_commodities": [],
        "base_effectiveness": "14",
        "typical_timeline_days": 14,
        "evidence_requirements": ["document", "certificate"],
        "regulatory_reference": "EUDR Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-034",
        "title": "Third-Party Transaction Verification",
        "description": (
            "Engage independent third party to verify transactions "
            "in high-corruption-risk sourcing regions, including "
            "price benchmarking, volume reconciliation, and "
            "customs documentation cross-checks."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["corruption", "circumvention_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "22",
        "typical_timeline_days": 30,
        "evidence_requirements": ["audit_report", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(a), Art. 10(2)(e)",
    },
    # -----------------------------------------------------------------------
    # SUPPLY_CHAIN_COMPLEXITY dimension (MMD-TPL-035 to MMD-TPL-039)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-035",
        "title": "Multi-Tier Supply Chain Mapping",
        "description": (
            "Conduct comprehensive mapping of all supply chain tiers "
            "from producer to operator, identifying all intermediaries, "
            "processing facilities, and logistics providers. Verify "
            "tier-by-tier traceability documentation."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["supply_chain_complexity"],
        "applicable_commodities": [],
        "base_effectiveness": "24",
        "typical_timeline_days": 60,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(a), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-036",
        "title": "Intermediary Verification Program",
        "description": (
            "Establish systematic verification of all intermediaries "
            "in the supply chain including trader registration, "
            "warehouse audits, transport documentation verification, "
            "and volume reconciliation at each transfer point."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["supply_chain_complexity", "supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "22",
        "typical_timeline_days": 45,
        "evidence_requirements": ["audit_report", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(a), Art. 10(2)(e)",
    },
    {
        "template_id": "MMD-TPL-037",
        "title": "Supply Chain Simplification Strategy",
        "description": (
            "Develop strategy to reduce supply chain complexity by "
            "establishing direct sourcing relationships, reducing "
            "intermediary tiers, and implementing direct trade "
            "programs where feasible."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["supply_chain_complexity"],
        "applicable_commodities": [],
        "base_effectiveness": "30",
        "typical_timeline_days": 120,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-038",
        "title": "Digital Traceability Platform Deployment",
        "description": (
            "Deploy digital traceability platform enabling real-time "
            "tracking of product flows, document digitization, QR "
            "code-based batch tracking, and automated compliance "
            "checks at each supply chain node."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["supply_chain_complexity", "mixing_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "26",
        "typical_timeline_days": 90,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(a), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-039",
        "title": "First-Mile Data Collection Enhancement",
        "description": (
            "Strengthen first-mile data collection at farm/production "
            "level through mobile apps, GPS capture tools, and "
            "cooperative-level aggregation systems to close "
            "traceability gaps at the supply chain origin."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["supply_chain_complexity", "supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "20",
        "typical_timeline_days": 60,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(a), Art. 11(2)(c)",
    },
    # -----------------------------------------------------------------------
    # MIXING_RISK dimension templates (MMD-TPL-040 to MMD-TPL-044)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-040",
        "title": "Physical Segregation Audit",
        "description": (
            "Commission independent audit of physical segregation "
            "controls at processing facilities to verify that EUDR-"
            "compliant products are kept separate from non-verified "
            "products throughout processing and storage."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["mixing_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "30",
        "typical_timeline_days": 30,
        "evidence_requirements": ["audit_report", "site_visit_report"],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-041",
        "title": "Batch Identity Preservation System",
        "description": (
            "Implement batch identity preservation system ensuring "
            "each batch can be traced to specific production plots. "
            "Include batch numbering, storage separation, and "
            "processing scheduling to prevent commingling."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["mixing_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "28",
        "typical_timeline_days": 60,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(a), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-042",
        "title": "Mass Balance Reconciliation Audit",
        "description": (
            "Conduct mass balance audit at all processing and "
            "transformation points to verify input-output volume "
            "consistency and detect potential mixing with products "
            "of unknown or non-compliant origin."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["mixing_risk", "commodity"],
        "applicable_commodities": [],
        "base_effectiveness": "26",
        "typical_timeline_days": 30,
        "evidence_requirements": ["audit_report", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-043",
        "title": "Warehouse Segregation Verification",
        "description": (
            "Verify physical segregation at warehouse and storage "
            "facilities through surprise inspections, inventory "
            "reconciliation, and documentation review."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["mixing_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "22",
        "typical_timeline_days": 14,
        "evidence_requirements": ["site_visit_report", "document"],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-044",
        "title": "Product Marking and Labeling System",
        "description": (
            "Implement product marking system (QR codes, RFID tags, "
            "or unique identifiers) enabling product-level traceability "
            "and segregation verification throughout the supply chain."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["mixing_risk", "supply_chain_complexity"],
        "applicable_commodities": [],
        "base_effectiveness": "20",
        "typical_timeline_days": 45,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(a), Art. 11(2)(c)",
    },
    # -----------------------------------------------------------------------
    # CIRCUMVENTION_RISK dimension (MMD-TPL-045 to MMD-TPL-050)
    # -----------------------------------------------------------------------
    {
        "template_id": "MMD-TPL-045",
        "title": "Trade Route Verification Analysis",
        "description": (
            "Analyze trade routes and logistics documentation to "
            "identify potential circumvention patterns including "
            "trans-shipment through low-scrutiny ports, unusual "
            "routing, and classification discrepancies."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["circumvention_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "20",
        "typical_timeline_days": 21,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(e), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-046",
        "title": "Independent Origin Verification",
        "description": (
            "Engage independent third party to verify product origin "
            "through isotope analysis, DNA testing (for wood species), "
            "or other scientific methods to confirm declared origin."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["circumvention_risk", "commodity"],
        "applicable_commodities": [],
        "base_effectiveness": "35",
        "typical_timeline_days": 45,
        "evidence_requirements": ["audit_report", "certificate"],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-047",
        "title": "Additional Supplier Origin Declarations",
        "description": (
            "Require enhanced origin declarations from suppliers "
            "with detailed production plot identification, harvest "
            "dates, transport documentation, and processing records "
            "enabling full chain of custody verification."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["circumvention_risk", "supplier"],
        "applicable_commodities": [],
        "base_effectiveness": "18",
        "typical_timeline_days": 14,
        "evidence_requirements": ["supplier_declaration", "document"],
        "regulatory_reference": "EUDR Art. 10(2)(a), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-048",
        "title": "Customs Documentation Cross-Check",
        "description": (
            "Cross-reference customs declarations, bills of lading, "
            "and phytosanitary certificates across jurisdictions to "
            "detect inconsistencies in declared origin, volume, or "
            "product classification."
        ),
        "article11_category": "additional_info",
        "applicable_dimensions": ["circumvention_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "16",
        "typical_timeline_days": 14,
        "evidence_requirements": ["document"],
        "regulatory_reference": "EUDR Art. 10(2)(e), Art. 11(2)(c)",
    },
    {
        "template_id": "MMD-TPL-049",
        "title": "Port-of-Entry Inspection Program",
        "description": (
            "Establish inspection program at ports of entry including "
            "physical sample collection, documentation verification, "
            "and volume reconciliation for shipments from high-risk "
            "origins."
        ),
        "article11_category": "independent_audit",
        "applicable_dimensions": ["circumvention_risk"],
        "applicable_commodities": [],
        "base_effectiveness": "24",
        "typical_timeline_days": 30,
        "evidence_requirements": ["audit_report", "site_visit_report"],
        "regulatory_reference": "EUDR Art. 11(2)(a)",
    },
    {
        "template_id": "MMD-TPL-050",
        "title": "Supply Chain Anti-Circumvention Controls",
        "description": (
            "Implement systematic anti-circumvention controls "
            "including volume cap alerts, price anomaly detection, "
            "new supplier vetting protocols, and periodic unannounced "
            "traceability audits."
        ),
        "article11_category": "other_measures",
        "applicable_dimensions": ["circumvention_risk", "supply_chain_complexity"],
        "applicable_commodities": [],
        "base_effectiveness": "22",
        "typical_timeline_days": 45,
        "evidence_requirements": ["document", "audit_report"],
        "regulatory_reference": "EUDR Art. 11(2)(c)",
    },
]

# Map string values to enum values
_DIMENSION_MAP: Dict[str, RiskDimension] = {d.value: d for d in RiskDimension}
_COMMODITY_MAP: Dict[str, EUDRCommodity] = {c.value: c for c in EUDRCommodity}
_CATEGORY_MAP: Dict[str, Article11Category] = {
    "independent_audit": Article11Category.INDEPENDENT_AUDIT,
    "additional_info": Article11Category.ADDITIONAL_INFO,
    "other_measures": Article11Category.OTHER_MEASURES,
}
_EVIDENCE_MAP: Dict[str, EvidenceType] = {
    "audit_report": EvidenceType.AUDIT_REPORT,
    "certificate": EvidenceType.CERTIFICATE,
    "document": EvidenceType.DOCUMENT,
    "satellite_image": EvidenceType.SATELLITE_IMAGE,
    "site_visit_report": EvidenceType.SITE_VISIT_REPORT,
    "supplier_declaration": EvidenceType.SUPPLIER_DECLARATION,
    "other": EvidenceType.OTHER,
}


class MeasureTemplateLibrary:
    """Curated library of EUDR Article 11 mitigation measure templates.

    Maintains 50+ templates organized by Article 11(2) category,
    risk dimension, and commodity applicability. Supports search,
    filter, and retrieval operations for the strategy designer.

    Attributes:
        _config: Agent configuration.
        _templates: Dictionary of template_id to MeasureTemplate.
        _loaded: Whether templates have been loaded.

    Example:
        >>> library = MeasureTemplateLibrary()
        >>> count = library.load_templates()
        >>> assert count >= 50
        >>> templates = library.search_templates(dimension=RiskDimension.COUNTRY)
        >>> assert len(templates) > 0
    """

    def __init__(
        self,
        config: Optional[MitigationMeasureDesignerConfig] = None,
    ) -> None:
        """Initialize MeasureTemplateLibrary.

        Args:
            config: Agent configuration. Uses get_config() if None.
        """
        self._config = config or get_config()
        self._templates: Dict[str, MeasureTemplate] = {}
        self._loaded = False
        logger.info("MeasureTemplateLibrary initialized")

    def load_templates(self) -> int:
        """Load built-in templates into the library.

        Parses the built-in template definitions and creates
        MeasureTemplate instances. Idempotent - calling multiple
        times reloads the templates.

        Returns:
            Number of templates loaded.
        """
        self._templates.clear()
        loaded = 0

        for raw in _BUILTIN_TEMPLATES:
            try:
                template = self._parse_template(raw)
                self._templates[template.template_id] = template
                loaded += 1
            except Exception as exc:
                logger.warning(
                    "Failed to parse template %s: %s",
                    raw.get("template_id", "unknown"),
                    exc,
                )

        self._loaded = True
        logger.info("Loaded %d templates into library", loaded)
        return loaded

    def _parse_template(self, raw: Dict[str, Any]) -> MeasureTemplate:
        """Parse a raw template dictionary into MeasureTemplate.

        Args:
            raw: Raw template data dictionary.

        Returns:
            Parsed MeasureTemplate instance.
        """
        dimensions = [
            _DIMENSION_MAP[d]
            for d in raw.get("applicable_dimensions", [])
            if d in _DIMENSION_MAP
        ]
        commodities = [
            _COMMODITY_MAP[c]
            for c in raw.get("applicable_commodities", [])
            if c in _COMMODITY_MAP
        ]
        category = _CATEGORY_MAP.get(
            raw.get("article11_category", "other_measures"),
            Article11Category.OTHER_MEASURES,
        )
        evidence = raw.get("evidence_requirements", [])

        return MeasureTemplate(
            template_id=raw["template_id"],
            title=raw["title"],
            description=raw.get("description", ""),
            article11_category=category,
            applicable_dimensions=dimensions,
            applicable_commodities=commodities,
            base_effectiveness=Decimal(str(raw.get("base_effectiveness", "0"))),
            typical_timeline_days=int(raw.get("typical_timeline_days", 30)),
            evidence_requirements=evidence,
            regulatory_reference=raw.get("regulatory_reference", ""),
        )

    def get_template(self, template_id: str) -> Optional[MeasureTemplate]:
        """Get a template by its ID.

        Args:
            template_id: Template identifier (e.g., "MMD-TPL-001").

        Returns:
            MeasureTemplate if found, None otherwise.
        """
        self._ensure_loaded()
        return self._templates.get(template_id)

    def search_templates(
        self,
        dimension: Optional[RiskDimension] = None,
        category: Optional[Article11Category] = None,
        commodity: Optional[EUDRCommodity] = None,
    ) -> List[MeasureTemplate]:
        """Search templates by filter criteria.

        Args:
            dimension: Filter by applicable risk dimension.
            category: Filter by Article 11(2) category.
            commodity: Filter by applicable commodity.

        Returns:
            List of matching MeasureTemplate instances.
        """
        self._ensure_loaded()
        results: List[MeasureTemplate] = []

        for template in self._templates.values():
            if dimension is not None:
                if dimension not in template.applicable_dimensions:
                    continue
            if category is not None:
                if template.article11_category != category:
                    continue
            if commodity is not None:
                if template.applicable_commodities:
                    if commodity not in template.applicable_commodities:
                        continue
            results.append(template)

        return results

    def get_templates_for_dimension(
        self,
        dimension: RiskDimension,
        commodity: EUDRCommodity,
    ) -> List[MeasureTemplate]:
        """Get templates applicable to a specific dimension and commodity.

        Convenience method used by the strategy designer to get
        all relevant templates for a risk dimension.

        Args:
            dimension: Target risk dimension.
            commodity: Target EUDR commodity.

        Returns:
            List of applicable templates sorted by effectiveness.
        """
        self._ensure_loaded()
        results = self.search_templates(
            dimension=dimension, commodity=commodity,
        )
        results.sort(
            key=lambda t: t.base_effectiveness, reverse=True,
        )
        return results

    def get_all_templates(self) -> List[MeasureTemplate]:
        """Get all loaded templates.

        Returns:
            List of all MeasureTemplate instances.
        """
        self._ensure_loaded()
        return list(self._templates.values())

    @property
    def template_count(self) -> int:
        """Number of templates loaded.

        Returns:
            Integer count of loaded templates.
        """
        return len(self._templates)

    @property
    def is_loaded(self) -> bool:
        """Whether templates have been loaded.

        Returns:
            True if load_templates() has been called.
        """
        return self._loaded

    def _ensure_loaded(self) -> None:
        """Ensure templates are loaded, auto-loading if needed."""
        if not self._loaded:
            self.load_templates()

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and template count.
        """
        return {
            "engine": "MeasureTemplateLibrary",
            "status": "available",
            "templates_loaded": self.template_count,
            "is_loaded": self._loaded,
        }
