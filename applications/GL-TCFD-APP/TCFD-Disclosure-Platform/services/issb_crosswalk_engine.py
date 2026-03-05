"""
ISSB/IFRS S2 Cross-Walk Engine -- Mapping between TCFD and IFRS S2 requirements.

This module implements the ``ISSBCrosswalkEngine`` for GL-TCFD-APP v1.0.
It provides a comprehensive mapping between the 11 TCFD recommended disclosures
and the corresponding IFRS S2 Climate-related Disclosures paragraphs, identifies
gaps where IFRS S2 requires additional disclosure beyond TCFD, generates
migration pathways for organizations transitioning from TCFD to IFRS S2,
maps SASB-derived industry-specific metrics, checks connected reporting
linkages to IFRS S1, and produces dual-standard compliance scorecards.

IFRS S2 was issued in June 2023 and is designed to supersede TCFD. While
fully incorporating TCFD's four-pillar structure, IFRS S2 extends requirements
in several areas: mandatory Scope 3, transition plan disclosures, carbon credit
treatment, current-period financial effects, and industry-specific metrics
derived from SASB standards.

Reference:
    - IFRS S2 Climate-related Disclosures (June 2023)
    - IFRS S1 General Requirements for Sustainability-related Disclosures (June 2023)
    - TCFD Final Report (June 2017)
    - SASB Standards (Industry-specific metrics)

Example:
    >>> from services.config import TCFDAppConfig
    >>> engine = ISSBCrosswalkEngine(TCFDAppConfig())
    >>> mappings = engine.get_tcfd_to_issb_mapping()
    >>> gaps = engine.identify_issb_gaps("org-1", "disc-1")
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    ISSB_CROSS_INDUSTRY_METRICS,
    ISSBMetricType,
    SectorType,
    TCFDAppConfig,
    TCFD_DISCLOSURES,
    TCFD_TO_IFRS_S2_MAPPING,
)
from .models import ISSBMapping, _new_id, _now, _sha256

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complete TCFD -> IFRS S2 mapping with paragraph-level detail
# ---------------------------------------------------------------------------

TCFD_TO_ISSB_MAPPING: Dict[str, Dict[str, Any]] = {
    "gov_a": {
        "tcfd_ref": "Governance (a)",
        "tcfd_requirement": "Board oversight of climate-related risks and opportunities",
        "ifrs_s2_paragraphs": ["5", "6"],
        "ifrs_s2_topic": "Governance",
        "mapping_status": "fully_mapped",
        "ifrs_s2_requirement": (
            "Disclose information about the governance body or individual "
            "responsible for oversight of climate-related risks and opportunities."
        ),
        "additional_ifrs_s2": [],
        "notes": "IFRS S2 para 5-6 fully align with TCFD Gov(a). No gaps.",
    },
    "gov_b": {
        "tcfd_ref": "Governance (b)",
        "tcfd_requirement": "Management's role in assessing and managing climate risks",
        "ifrs_s2_paragraphs": ["5", "6"],
        "ifrs_s2_topic": "Governance",
        "mapping_status": "fully_mapped",
        "ifrs_s2_requirement": (
            "Disclose management's role in the governance processes, controls, "
            "and procedures used to monitor, manage, and oversee climate risks."
        ),
        "additional_ifrs_s2": [],
        "notes": "IFRS S2 para 5-6 align with TCFD Gov(b). Management skills/competency also required.",
    },
    "str_a": {
        "tcfd_ref": "Strategy (a)",
        "tcfd_requirement": "Climate risks and opportunities over short/medium/long term",
        "ifrs_s2_paragraphs": ["10", "11", "12"],
        "ifrs_s2_topic": "Strategy",
        "mapping_status": "enhanced",
        "ifrs_s2_requirement": (
            "Disclose climate-related risks and opportunities that could reasonably "
            "be expected to affect the entity's cash flows, access to finance, "
            "or cost of capital over the short, medium, or long term."
        ),
        "additional_ifrs_s2": [
            "Concentration of risk exposure (para 10(d))",
            "Current and anticipated financial effects quantification (para 14-21)",
        ],
        "notes": "IFRS S2 extends TCFD with quantitative financial effects and concentration of exposure.",
    },
    "str_b": {
        "tcfd_ref": "Strategy (b)",
        "tcfd_requirement": "Impact on business, strategy, and financial planning",
        "ifrs_s2_paragraphs": ["13", "14", "15", "16", "17", "18", "19", "20", "21"],
        "ifrs_s2_topic": "Strategy",
        "mapping_status": "enhanced",
        "ifrs_s2_requirement": (
            "Disclose current and anticipated effects of climate-related risks "
            "and opportunities on the entity's financial position, financial "
            "performance, and cash flows."
        ),
        "additional_ifrs_s2": [
            "Transition plan disclosures (para 14(a))",
            "Climate-related targets linked to financial plans (para 14(b))",
            "Current-period financial effects on balance sheet, P&L, cash flows (para 15-21)",
            "Carbon credits and offsets treatment (para 14(a)(iv))",
        ],
        "notes": (
            "IFRS S2 significantly extends TCFD with mandatory transition plan disclosure, "
            "quantified current-period financial effects, and carbon credit treatment."
        ),
    },
    "str_c": {
        "tcfd_ref": "Strategy (c)",
        "tcfd_requirement": "Resilience of strategy under climate scenarios incl. 2C",
        "ifrs_s2_paragraphs": ["22"],
        "ifrs_s2_topic": "Climate Resilience",
        "mapping_status": "enhanced",
        "ifrs_s2_requirement": (
            "Disclose the entity's assessment of its climate resilience using "
            "climate-related scenario analysis. Must include scenario consistent "
            "with latest international agreement on climate change."
        ),
        "additional_ifrs_s2": [
            "Scenario analysis required for all entities (not just where relevant)",
            "Must use scenario consistent with Paris Agreement (para 22(b))",
            "Must disclose key assumptions, time horizons, inputs, outputs (para 22(c))",
        ],
        "notes": (
            "IFRS S2 makes scenario analysis mandatory for all entities. "
            "TCFD's 'where relevant' qualifier is removed."
        ),
    },
    "rm_a": {
        "tcfd_ref": "Risk Management (a)",
        "tcfd_requirement": "Processes for identifying and assessing climate risks",
        "ifrs_s2_paragraphs": ["25"],
        "ifrs_s2_topic": "Risk Management",
        "mapping_status": "fully_mapped",
        "ifrs_s2_requirement": (
            "Disclose the processes and related policies used to identify, "
            "assess, prioritize, and monitor climate-related risks."
        ),
        "additional_ifrs_s2": [],
        "notes": "IFRS S2 para 25 fully aligns with TCFD RM(a).",
    },
    "rm_b": {
        "tcfd_ref": "Risk Management (b)",
        "tcfd_requirement": "Processes for managing climate-related risks",
        "ifrs_s2_paragraphs": ["25"],
        "ifrs_s2_topic": "Risk Management",
        "mapping_status": "fully_mapped",
        "ifrs_s2_requirement": (
            "Disclose the processes used to manage climate-related risks "
            "including how decisions are made to mitigate, transfer, accept, "
            "or avoid climate-related risks."
        ),
        "additional_ifrs_s2": [],
        "notes": "IFRS S2 para 25 fully aligns with TCFD RM(b).",
    },
    "rm_c": {
        "tcfd_ref": "Risk Management (c)",
        "tcfd_requirement": "Integration into overall risk management (ERM)",
        "ifrs_s2_paragraphs": ["25"],
        "ifrs_s2_topic": "Risk Management",
        "mapping_status": "fully_mapped",
        "ifrs_s2_requirement": (
            "Disclose how climate risk processes are integrated into the "
            "entity's overall risk management process."
        ),
        "additional_ifrs_s2": [],
        "notes": "IFRS S2 para 25 fully aligns with TCFD RM(c).",
    },
    "mt_a": {
        "tcfd_ref": "Metrics & Targets (a)",
        "tcfd_requirement": "Metrics used to assess climate risks and opportunities",
        "ifrs_s2_paragraphs": ["29"],
        "ifrs_s2_topic": "Metrics and Targets",
        "mapping_status": "enhanced",
        "ifrs_s2_requirement": (
            "Disclose 7 cross-industry metrics (GHG, transition risk assets, "
            "physical risk assets, opportunity revenue, capex, internal carbon "
            "price, remuneration linked) plus industry-specific metrics."
        ),
        "additional_ifrs_s2": [
            "7 mandatory cross-industry metrics (para 29(a)-(g))",
            "Industry-specific metrics from SASB-derived appendix (para 32)",
            "Internal carbon price mandatory if used (para 29(f))",
        ],
        "notes": (
            "IFRS S2 specifies 7 mandatory cross-industry metrics plus "
            "industry-specific metrics derived from SASB standards."
        ),
    },
    "mt_b": {
        "tcfd_ref": "Metrics & Targets (b)",
        "tcfd_requirement": "Scope 1, 2, and (if appropriate) Scope 3 GHG emissions",
        "ifrs_s2_paragraphs": ["29(a)"],
        "ifrs_s2_topic": "GHG Emissions",
        "mapping_status": "enhanced",
        "ifrs_s2_requirement": (
            "Disclose absolute gross Scope 1, Scope 2, and Scope 3 GHG emissions. "
            "Scope 3 is mandatory for all entities (not 'if appropriate')."
        ),
        "additional_ifrs_s2": [
            "Scope 3 mandatory for ALL entities (removes TCFD 'if appropriate' qualifier)",
            "Must use GHG Protocol Corporate Standard (para 29(a)(ii))",
            "Scope 2: both location-based AND market-based required",
            "Financed emissions for financial institutions (PCAF methodology)",
        ],
        "notes": (
            "IFRS S2 removes TCFD's 'if appropriate' qualifier for Scope 3. "
            "ALL entities must disclose Scope 3 emissions."
        ),
    },
    "mt_c": {
        "tcfd_ref": "Metrics & Targets (c)",
        "tcfd_requirement": "Targets and performance against targets",
        "ifrs_s2_paragraphs": ["33", "34", "35", "36"],
        "ifrs_s2_topic": "Targets",
        "mapping_status": "enhanced",
        "ifrs_s2_requirement": (
            "Disclose each climate-related target including metric used, "
            "objective, part of entity covered, time period, base period, "
            "milestones, and progress. Must state if target is based on "
            "scientific evidence."
        ),
        "additional_ifrs_s2": [
            "Must state if target is validated by third party (para 33(e))",
            "Must disclose approach to target setting (para 33(a))",
            "Must state if target is based on scientific evidence (para 33(f))",
            "Detailed progress tracking with quantified performance (para 36)",
            "Net-zero targets must separately disclose gross and net (para 36(b))",
            "Carbon credit/offset plans and quality criteria (para 36(e))",
        ],
        "notes": (
            "IFRS S2 significantly extends TCFD target disclosures with "
            "detailed progress tracking, SBTi alignment disclosure, and "
            "carbon credit/offset treatment requirements."
        ),
    },
}


# ---------------------------------------------------------------------------
# Requirements in IFRS S2 that go BEYOND TCFD
# ---------------------------------------------------------------------------

ISSB_ADDITIONAL_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "id": "ISSB-ADD-001",
        "category": "transition_plan",
        "ifrs_s2_paragraph": "14(a)",
        "title": "Transition Plan Disclosure",
        "description": (
            "IFRS S2 requires disclosure of any transition plan the entity has, "
            "including targets, actions, resources, and how the plan affects "
            "the entity's financial position and performance."
        ),
        "tcfd_gap": "TCFD mentions transition plans in supplementary guidance but does not require them.",
        "priority": "high",
        "effort_days": 30,
    },
    {
        "id": "ISSB-ADD-002",
        "category": "carbon_credits",
        "ifrs_s2_paragraph": "14(a)(iv)",
        "title": "Carbon Credits and Offsets",
        "description": (
            "IFRS S2 requires entities to disclose the planned use of carbon "
            "credits to achieve targets, including credit type, registry, "
            "quality criteria, and underlying projects."
        ),
        "tcfd_gap": "TCFD does not specifically address carbon credit disclosure.",
        "priority": "medium",
        "effort_days": 10,
    },
    {
        "id": "ISSB-ADD-003",
        "category": "current_period_financial_effects",
        "ifrs_s2_paragraph": "15-21",
        "title": "Current-Period Financial Effects",
        "description": (
            "IFRS S2 requires quantified disclosure of how climate risks and "
            "opportunities have affected the entity's financial position, "
            "financial performance, and cash flows in the current reporting period."
        ),
        "tcfd_gap": "TCFD focuses on forward-looking impacts; current-period financial effects are new.",
        "priority": "high",
        "effort_days": 40,
    },
    {
        "id": "ISSB-ADD-004",
        "category": "scope_3_mandatory",
        "ifrs_s2_paragraph": "29(a)",
        "title": "Mandatory Scope 3 GHG Emissions",
        "description": (
            "IFRS S2 requires Scope 3 emissions disclosure for ALL entities. "
            "TCFD qualified this as 'if appropriate'. Relief measures available "
            "in the first year (estimates acceptable, Category 15 for financials)."
        ),
        "tcfd_gap": "TCFD uses 'if appropriate' qualifier for Scope 3; IFRS S2 makes it mandatory.",
        "priority": "high",
        "effort_days": 60,
    },
    {
        "id": "ISSB-ADD-005",
        "category": "industry_metrics",
        "ifrs_s2_paragraph": "32",
        "title": "Industry-Specific Metrics (SASB-derived)",
        "description": (
            "IFRS S2 includes industry-specific metrics derived from SASB "
            "standards, organized by GICS sector. Entities must disclose "
            "applicable metrics for their industry."
        ),
        "tcfd_gap": "TCFD mentions industry-specific metrics but does not specify which ones.",
        "priority": "medium",
        "effort_days": 25,
    },
    {
        "id": "ISSB-ADD-006",
        "category": "cross_industry_metrics",
        "ifrs_s2_paragraph": "29(b)-(g)",
        "title": "Seven Cross-Industry Climate Metrics",
        "description": (
            "IFRS S2 specifies 7 mandatory cross-industry metrics: (a) GHG "
            "emissions, (b) transition risk assets, (c) physical risk assets, "
            "(d) opportunity revenue, (e) capital deployment, (f) internal "
            "carbon price, (g) remuneration linked to climate."
        ),
        "tcfd_gap": "TCFD recommends metrics but does not specify the exact 7 cross-industry metrics.",
        "priority": "medium",
        "effort_days": 20,
    },
    {
        "id": "ISSB-ADD-007",
        "category": "connected_reporting",
        "ifrs_s2_paragraph": "IFRS S1 para 21-24",
        "title": "Connected Reporting with IFRS S1",
        "description": (
            "IFRS S2 climate disclosures must be connected to IFRS S1 general "
            "sustainability requirements, including linkage to financial statements "
            "and explanation of connectivity between climate and other ESG topics."
        ),
        "tcfd_gap": "TCFD is standalone; no explicit connection to broader sustainability reporting.",
        "priority": "medium",
        "effort_days": 15,
    },
    {
        "id": "ISSB-ADD-008",
        "category": "scenario_analysis_mandatory",
        "ifrs_s2_paragraph": "22",
        "title": "Mandatory Scenario Analysis for All Entities",
        "description": (
            "IFRS S2 requires scenario analysis for ALL entities, removing "
            "TCFD's 'where relevant' qualifier. Must include at least one "
            "scenario consistent with the latest international climate agreement."
        ),
        "tcfd_gap": "TCFD recommends scenario analysis; IFRS S2 mandates it for all entities.",
        "priority": "high",
        "effort_days": 35,
    },
]


# ---------------------------------------------------------------------------
# SASB-derived industry metrics by sector
# ---------------------------------------------------------------------------

_SASB_INDUSTRY_METRICS: Dict[str, List[Dict[str, str]]] = {
    "energy": [
        {"metric_id": "EM-EP-110a.1", "name": "Gross global Scope 1 emissions", "unit": "tCO2e"},
        {"metric_id": "EM-EP-110a.2", "name": "Methane emissions percentage", "unit": "%"},
        {"metric_id": "EM-EP-110a.3", "name": "Discussion of long-term/short-term strategy to manage emissions", "unit": "narrative"},
        {"metric_id": "EM-EP-420a.1", "name": "Reserves in countries with carbon constraints", "unit": "currency"},
        {"metric_id": "EM-EP-420a.2", "name": "Estimated CO2 embedded in proved reserves", "unit": "tCO2"},
    ],
    "transport": [
        {"metric_id": "TR-RO-110a.1", "name": "Gross global Scope 1 emissions", "unit": "tCO2e"},
        {"metric_id": "TR-RO-110a.2", "name": "Discussion of fleet fuel management", "unit": "narrative"},
        {"metric_id": "TR-AL-110a.1", "name": "Gross Scope 1 emissions from owned/controlled aircraft", "unit": "tCO2e"},
        {"metric_id": "TR-AL-110a.2", "name": "Revenue ton-miles (RTM)", "unit": "RTM"},
    ],
    "materials": [
        {"metric_id": "EM-CM-110a.1", "name": "Gross global Scope 1 emissions", "unit": "tCO2e"},
        {"metric_id": "EM-CM-110a.2", "name": "Scope 1 percentage covered by emissions-limiting regulations", "unit": "%"},
        {"metric_id": "EM-IS-110a.1", "name": "Gross Scope 1 emissions from iron and steel production", "unit": "tCO2e"},
    ],
    "banking": [
        {"metric_id": "FN-CB-410a.1", "name": "Commercial lending exposure to climate-related industries", "unit": "currency"},
        {"metric_id": "FN-CB-410a.2", "name": "Financed emissions (Scope 3 Category 15)", "unit": "tCO2e"},
        {"metric_id": "FN-CB-1", "name": "Weighted average carbon intensity of portfolio", "unit": "tCO2e/revenue"},
    ],
    "insurance": [
        {"metric_id": "FN-IN-410a.1", "name": "Probable Maximum Loss from weather events", "unit": "currency"},
        {"metric_id": "FN-IN-410a.2", "name": "Total amount of assets in regions with high physical risk", "unit": "currency"},
        {"metric_id": "FN-IN-450a.1", "name": "Net premiums written related to energy/weather events", "unit": "currency"},
    ],
    "asset_management": [
        {"metric_id": "FN-AC-410a.1", "name": "AUM invested in climate solutions", "unit": "currency"},
        {"metric_id": "FN-AC-410a.2", "name": "Portfolio weighted average carbon intensity (WACI)", "unit": "tCO2e/M_revenue"},
        {"metric_id": "FN-AC-410a.3", "name": "Percentage of AUM aligned with Paris Agreement", "unit": "%"},
    ],
    "agriculture": [
        {"metric_id": "FB-AG-110a.1", "name": "Gross global Scope 1 emissions", "unit": "tCO2e"},
        {"metric_id": "FB-AG-110a.2", "name": "Discussion of strategy to manage emissions from biological processes", "unit": "narrative"},
        {"metric_id": "FB-AG-440a.1", "name": "Percentage of agricultural land under sustainable management", "unit": "%"},
    ],
    "buildings": [
        {"metric_id": "IF-RE-130a.1", "name": "Energy consumption data coverage (%)", "unit": "%"},
        {"metric_id": "IF-RE-130a.2", "name": "Total energy consumed by portfolio", "unit": "GJ"},
        {"metric_id": "IF-RE-130a.3", "name": "Like-for-like change in energy consumption", "unit": "%"},
        {"metric_id": "IF-RE-130a.4", "name": "Percentage of eligible portfolio with energy rating", "unit": "%"},
    ],
    "consumer_goods": [
        {"metric_id": "CG-AA-430a.1", "name": "Scope 1+2 emissions from manufacturing", "unit": "tCO2e"},
        {"metric_id": "CG-HP-410a.1", "name": "Revenue from products designed for environmental impact reduction", "unit": "currency"},
    ],
    "technology": [
        {"metric_id": "TC-SI-130a.1", "name": "Total energy consumed by data centers", "unit": "GJ"},
        {"metric_id": "TC-SI-130a.2", "name": "Percentage of energy from renewable sources", "unit": "%"},
        {"metric_id": "TC-SI-130a.3", "name": "Power Usage Effectiveness (PUE) for data centers", "unit": "ratio"},
    ],
    "healthcare": [
        {"metric_id": "HC-DY-410a.1", "name": "Revenue from products designed for environmental sustainability", "unit": "currency"},
        {"metric_id": "HC-DI-410a.1", "name": "Discussion of strategy to manage environmental impact of products", "unit": "narrative"},
    ],
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ISSBComplianceScore(BaseModel):
    """IFRS S2 compliance score."""
    org_id: str = Field(...)
    disclosure_id: str = Field(default="")
    tcfd_score: float = Field(default=0.0, ge=0.0, le=100.0)
    issb_score: float = Field(default=0.0, ge=0.0, le=100.0)
    gap_count: int = Field(default=0, ge=0)
    fully_mapped_count: int = Field(default=0)
    enhanced_count: int = Field(default=0)
    additional_requirements_met: int = Field(default=0)
    additional_requirements_total: int = Field(default=8)
    details: List[Dict[str, Any]] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class ISSBGap(BaseModel):
    """A gap between TCFD and IFRS S2 requirements."""
    id: str = Field(default_factory=_new_id)
    tcfd_code: str = Field(default="")
    ifrs_s2_paragraph: str = Field(default="")
    category: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    priority: str = Field(default="medium")
    effort_days: int = Field(default=0)
    action_required: str = Field(default="")
    current_status: str = Field(default="open")


class MigrationPathway(BaseModel):
    """Migration pathway from TCFD to IFRS S2."""
    org_id: str = Field(...)
    current_tcfd_score: float = Field(default=0.0)
    target_issb_score: float = Field(default=0.0)
    total_gap_count: int = Field(default=0)
    total_effort_days: int = Field(default=0)
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    timeline_months: int = Field(default=0)
    quick_wins: List[Dict[str, Any]] = Field(default_factory=list)
    critical_actions: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class DualStandardScorecard(BaseModel):
    """Side-by-side TCFD vs IFRS S2 compliance scorecard."""
    org_id: str = Field(...)
    tcfd_overall: float = Field(default=0.0)
    issb_overall: float = Field(default=0.0)
    pillar_comparison: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    disclosure_comparison: List[Dict[str, Any]] = Field(default_factory=list)
    additional_requirements_status: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: str = Field(default="")
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# ISSBCrosswalkEngine
# ---------------------------------------------------------------------------

class ISSBCrosswalkEngine:
    """
    ISSB/IFRS S2 Cross-Walk Engine.

    Provides comprehensive mapping between TCFD and IFRS S2, identifies gaps,
    generates migration pathways, maps industry-specific metrics, and produces
    dual-standard compliance scorecards.

    Attributes:
        config: Application configuration.
        _org_issb_status: Track per-org ISSB compliance status.

    Example:
        >>> engine = ISSBCrosswalkEngine(TCFDAppConfig())
        >>> mappings = engine.get_tcfd_to_issb_mapping()
        >>> len(mappings)
        11
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize the ISSBCrosswalkEngine.

        Args:
            config: Optional application configuration.
        """
        self.config = config or TCFDAppConfig()
        self._org_issb_status: Dict[str, Dict[str, Any]] = {}
        logger.info("ISSBCrosswalkEngine initialized")

    def get_tcfd_to_issb_mapping(self) -> List[ISSBMapping]:
        """
        Get the complete TCFD-to-IFRS S2 mapping table.

        Returns:
            List of 11 ISSBMapping objects covering all TCFD disclosures.
        """
        mappings: List[ISSBMapping] = []
        for code, data in TCFD_TO_ISSB_MAPPING.items():
            mapping = ISSBMapping(
                tcfd_disclosure_ref=code,
                ifrs_s2_paragraph=", ".join(data["ifrs_s2_paragraphs"]),
                mapping_status=data["mapping_status"],
                gap_description=(
                    "; ".join(data["additional_ifrs_s2"])
                    if data["additional_ifrs_s2"] else None
                ),
                action_required=(
                    f"Address {len(data['additional_ifrs_s2'])} additional IFRS S2 requirement(s)"
                    if data["additional_ifrs_s2"] else None
                ),
            )
            mappings.append(mapping)

        logger.info("Retrieved %d TCFD-to-ISSB mappings", len(mappings))
        return mappings

    def check_issb_compliance(
        self,
        org_id: str,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> ISSBComplianceScore:
        """
        Check IFRS S2 compliance for an organization's disclosure.

        Evaluates both the TCFD base requirements and the additional
        IFRS S2 requirements (transition plan, carbon credits, etc.).

        Args:
            org_id: Organization ID.
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator for compliance checks.

        Returns:
            ISSBComplianceScore with overall and per-disclosure scores.
        """
        tcfd_score = 0.0
        if disclosure_generator:
            try:
                compliance = disclosure_generator.check_compliance(disclosure_id)
                tcfd_score = compliance.overall_score
            except (ValueError, AttributeError):
                tcfd_score = 0.0

        # Compute ISSB score from TCFD base + additional requirements
        fully_mapped = sum(
            1 for d in TCFD_TO_ISSB_MAPPING.values()
            if d["mapping_status"] == "fully_mapped"
        )
        enhanced = sum(
            1 for d in TCFD_TO_ISSB_MAPPING.values()
            if d["mapping_status"] == "enhanced"
        )

        # Check additional requirements status
        org_status = self._org_issb_status.get(org_id, {})
        additional_met = sum(
            1 for req in ISSB_ADDITIONAL_REQUIREMENTS
            if org_status.get(req["id"], {}).get("status") == "complete"
        )

        # ISSB score = TCFD base * fraction + additional requirement fraction
        tcfd_fraction = tcfd_score / 100.0
        additional_fraction = additional_met / len(ISSB_ADDITIONAL_REQUIREMENTS)
        issb_score = round((tcfd_fraction * 70.0 + additional_fraction * 30.0), 1)

        details: List[Dict[str, Any]] = []
        for code, data in TCFD_TO_ISSB_MAPPING.items():
            details.append({
                "tcfd_code": code,
                "tcfd_ref": data["tcfd_ref"],
                "ifrs_s2_paragraphs": data["ifrs_s2_paragraphs"],
                "mapping_status": data["mapping_status"],
                "additional_count": len(data["additional_ifrs_s2"]),
            })

        provenance = _sha256(
            f"{org_id}:{disclosure_id}:{tcfd_score}:{issb_score}"
        )

        result = ISSBComplianceScore(
            org_id=org_id,
            disclosure_id=disclosure_id,
            tcfd_score=tcfd_score,
            issb_score=issb_score,
            gap_count=len(ISSB_ADDITIONAL_REQUIREMENTS) - additional_met,
            fully_mapped_count=fully_mapped,
            enhanced_count=enhanced,
            additional_requirements_met=additional_met,
            additional_requirements_total=len(ISSB_ADDITIONAL_REQUIREMENTS),
            details=details,
            provenance_hash=provenance,
        )

        logger.info(
            "ISSB compliance for org %s: tcfd=%.1f%%, issb=%.1f%%, gaps=%d",
            org_id, tcfd_score, issb_score, result.gap_count,
        )
        return result

    def identify_issb_gaps(
        self,
        org_id: str,
        disclosure_id: str,
        disclosure_generator: Optional[Any] = None,
    ) -> List[ISSBGap]:
        """
        Identify gaps between TCFD disclosure and IFRS S2 requirements.

        Args:
            org_id: Organization ID.
            disclosure_id: Disclosure ID.
            disclosure_generator: Optional DisclosureGenerator reference.

        Returns:
            List of ISSBGap objects identifying specific gaps.
        """
        gaps: List[ISSBGap] = []
        org_status = self._org_issb_status.get(org_id, {})

        # Gaps from enhanced mappings (IFRS S2 requires more than TCFD)
        for code, data in TCFD_TO_ISSB_MAPPING.items():
            if data["mapping_status"] == "enhanced":
                for additional in data["additional_ifrs_s2"]:
                    gaps.append(ISSBGap(
                        tcfd_code=code,
                        ifrs_s2_paragraph=", ".join(data["ifrs_s2_paragraphs"]),
                        category=data["ifrs_s2_topic"],
                        title=f"Enhanced requirement for {data['tcfd_ref']}",
                        description=additional,
                        priority="high" if code in ("str_b", "mt_b") else "medium",
                        effort_days=15,
                        action_required=f"Address IFRS S2 enhancement: {additional}",
                        current_status=(
                            org_status.get(f"{code}_enhanced", {}).get("status", "open")
                        ),
                    ))

        # Gaps from additional requirements beyond TCFD
        for req in ISSB_ADDITIONAL_REQUIREMENTS:
            req_status = org_status.get(req["id"], {}).get("status", "open")
            if req_status != "complete":
                gaps.append(ISSBGap(
                    tcfd_code="",
                    ifrs_s2_paragraph=req["ifrs_s2_paragraph"],
                    category=req["category"],
                    title=req["title"],
                    description=req["description"],
                    priority=req["priority"],
                    effort_days=req["effort_days"],
                    action_required=req["description"],
                    current_status=req_status,
                ))

        logger.info(
            "Identified %d ISSB gaps for org %s", len(gaps), org_id,
        )
        return gaps

    def get_additional_issb_requirements(self) -> List[Dict[str, Any]]:
        """
        Get requirements in IFRS S2 that go beyond TCFD.

        Returns:
            List of additional requirement dictionaries.
        """
        return ISSB_ADDITIONAL_REQUIREMENTS.copy()

    def generate_migration_pathway(self, org_id: str) -> MigrationPathway:
        """
        Generate a phased migration pathway from TCFD to IFRS S2.

        Organizes the migration into three phases:
          Phase 1: Quick wins (0-3 months) - fully mapped, low effort
          Phase 2: Core gaps (3-9 months) - enhanced requirements
          Phase 3: Advanced (9-18 months) - additional requirements

        Args:
            org_id: Organization ID.

        Returns:
            MigrationPathway with phased actions and timeline.
        """
        org_status = self._org_issb_status.get(org_id, {})
        all_gaps = self.identify_issb_gaps(org_id, "")

        open_gaps = [g for g in all_gaps if g.current_status != "complete"]
        total_effort = sum(g.effort_days for g in open_gaps)

        quick_wins: List[Dict[str, Any]] = [
            {
                "title": g.title,
                "effort_days": g.effort_days,
                "priority": g.priority,
                "category": g.category,
            }
            for g in open_gaps if g.effort_days <= 15
        ]

        critical_actions: List[Dict[str, Any]] = [
            {
                "title": g.title,
                "effort_days": g.effort_days,
                "priority": g.priority,
                "category": g.category,
                "description": g.description,
            }
            for g in open_gaps if g.priority == "high"
        ]

        phases: List[Dict[str, Any]] = [
            {
                "phase": 1,
                "name": "Foundation & Quick Wins",
                "duration_months": 3,
                "description": (
                    "Address fully-mapped TCFD disclosures, close easy ISSB gaps, "
                    "and establish data collection for cross-industry metrics."
                ),
                "actions": [
                    "Audit current TCFD disclosures against IFRS S2 mapping",
                    "Establish Scope 3 data collection process",
                    "Map 7 cross-industry metrics to existing data sources",
                    "Identify applicable industry-specific SASB metrics",
                ],
                "effort_days": min(total_effort * 0.3, 45),
                "target_score_improvement": 15.0,
            },
            {
                "phase": 2,
                "name": "Core Gaps & Enhanced Requirements",
                "duration_months": 6,
                "description": (
                    "Address enhanced IFRS S2 requirements including transition plan, "
                    "current-period financial effects, and mandatory scenario analysis."
                ),
                "actions": [
                    "Develop IFRS S2-compliant transition plan",
                    "Quantify current-period financial effects of climate risks",
                    "Conduct mandatory scenario analysis (minimum 2 scenarios)",
                    "Calculate and report Scope 3 across all material categories",
                    "Implement carbon credit disclosure framework",
                ],
                "effort_days": min(total_effort * 0.5, 120),
                "target_score_improvement": 20.0,
            },
            {
                "phase": 3,
                "name": "Advanced & Connected Reporting",
                "duration_months": 9,
                "description": (
                    "Achieve full IFRS S2 compliance with connected reporting, "
                    "industry metrics, and third-party validation."
                ),
                "actions": [
                    "Implement connected reporting with IFRS S1",
                    "Deploy industry-specific SASB metrics",
                    "Establish limited assurance for climate disclosures",
                    "Validate targets with SBTi or equivalent",
                    "Integrate climate disclosures into financial statements",
                ],
                "effort_days": min(total_effort * 0.2, 60),
                "target_score_improvement": 10.0,
            },
        ]

        timeline = sum(p["duration_months"] for p in phases)
        provenance = _sha256(f"{org_id}:{len(open_gaps)}:{total_effort}:{timeline}")

        pathway = MigrationPathway(
            org_id=org_id,
            current_tcfd_score=0.0,
            target_issb_score=85.0,
            total_gap_count=len(open_gaps),
            total_effort_days=total_effort,
            phases=phases,
            timeline_months=timeline,
            quick_wins=quick_wins,
            critical_actions=critical_actions,
            provenance_hash=provenance,
        )

        logger.info(
            "Generated migration pathway for org %s: %d gaps, %d days, %d months",
            org_id, len(open_gaps), total_effort, timeline,
        )
        return pathway

    def map_industry_metrics(
        self,
        org_id: str,
        industry: str,
    ) -> List[Dict[str, str]]:
        """
        Map SASB-derived industry-specific metrics for the organization's sector.

        Args:
            org_id: Organization ID.
            industry: Industry/sector key (e.g. "energy", "banking").

        Returns:
            List of applicable industry-specific metric dictionaries.
        """
        industry_key = industry.lower().replace(" ", "_")
        metrics = _SASB_INDUSTRY_METRICS.get(industry_key, [])

        if not metrics:
            logger.warning(
                "No SASB metrics found for industry '%s' (org %s). "
                "Available: %s",
                industry, org_id, list(_SASB_INDUSTRY_METRICS.keys()),
            )
            return []

        logger.info(
            "Mapped %d industry metrics for org %s (industry=%s)",
            len(metrics), org_id, industry,
        )
        return metrics

    def check_connected_reporting(self, org_id: str) -> Dict[str, Any]:
        """
        Check IFRS S1 general requirements linkage for connected reporting.

        IFRS S2 climate disclosures must be presented alongside IFRS S1
        general sustainability disclosures with clear connectivity.

        Args:
            org_id: Organization ID.

        Returns:
            Dictionary with connected reporting status and requirements.
        """
        org_status = self._org_issb_status.get(org_id, {})

        requirements = [
            {
                "id": "S1-GOV",
                "requirement": "IFRS S1 Governance disclosures",
                "ifrs_s1_paragraph": "26-27",
                "description": "Governance processes for sustainability-related risks",
                "status": org_status.get("s1_governance", {}).get("status", "not_started"),
            },
            {
                "id": "S1-STRATEGY",
                "requirement": "IFRS S1 Strategy disclosures",
                "ifrs_s1_paragraph": "28-42",
                "description": "Effects of sustainability risks on business model and value chain",
                "status": org_status.get("s1_strategy", {}).get("status", "not_started"),
            },
            {
                "id": "S1-RM",
                "requirement": "IFRS S1 Risk Management disclosures",
                "ifrs_s1_paragraph": "43-44",
                "description": "Processes to identify, assess, and manage sustainability risks",
                "status": org_status.get("s1_risk_mgmt", {}).get("status", "not_started"),
            },
            {
                "id": "S1-MT",
                "requirement": "IFRS S1 Metrics and Targets",
                "ifrs_s1_paragraph": "45-53",
                "description": "Metrics and targets used to measure and manage sustainability",
                "status": org_status.get("s1_metrics", {}).get("status", "not_started"),
            },
            {
                "id": "S1-CONNECT",
                "requirement": "Connected information linkage",
                "ifrs_s1_paragraph": "21-24",
                "description": "Connectivity between climate and general sustainability disclosures",
                "status": org_status.get("s1_connected", {}).get("status", "not_started"),
            },
        ]

        completed = sum(1 for r in requirements if r["status"] == "complete")
        total = len(requirements)

        result = {
            "org_id": org_id,
            "connected_reporting_status": "complete" if completed == total else "in_progress" if completed > 0 else "not_started",
            "requirements": requirements,
            "completed": completed,
            "total": total,
            "completeness_pct": round(completed / max(total, 1) * 100, 1),
            "next_actions": [
                r["requirement"] for r in requirements if r["status"] != "complete"
            ],
        }

        logger.info(
            "Connected reporting check for org %s: %d/%d complete",
            org_id, completed, total,
        )
        return result

    def get_dual_standard_scorecard(
        self,
        org_id: str,
        disclosure_generator: Optional[Any] = None,
        disclosure_id: str = "",
    ) -> DualStandardScorecard:
        """
        Produce a side-by-side TCFD vs IFRS S2 compliance scorecard.

        Args:
            org_id: Organization ID.
            disclosure_generator: Optional DisclosureGenerator for TCFD scores.
            disclosure_id: Optional disclosure ID for TCFD scoring.

        Returns:
            DualStandardScorecard with comparative scores.
        """
        # TCFD score
        tcfd_score = 0.0
        if disclosure_generator and disclosure_id:
            try:
                compliance = disclosure_generator.check_compliance(disclosure_id)
                tcfd_score = compliance.overall_score
            except (ValueError, AttributeError):
                pass

        # ISSB score
        issb_result = self.check_issb_compliance(
            org_id, disclosure_id, disclosure_generator,
        )

        # Pillar comparison
        pillar_comparison: Dict[str, Dict[str, float]] = {}
        pillar_tcfd_codes = {
            "governance": ["gov_a", "gov_b"],
            "strategy": ["str_a", "str_b", "str_c"],
            "risk_management": ["rm_a", "rm_b", "rm_c"],
            "metrics_targets": ["mt_a", "mt_b", "mt_c"],
        }

        for pillar, codes in pillar_tcfd_codes.items():
            mapped_count = sum(
                1 for c in codes
                if TCFD_TO_ISSB_MAPPING.get(c, {}).get("mapping_status") == "fully_mapped"
            )
            enhanced_count = sum(
                1 for c in codes
                if TCFD_TO_ISSB_MAPPING.get(c, {}).get("mapping_status") == "enhanced"
            )
            total = len(codes)
            tcfd_pillar = round(mapped_count / total * 100, 1) if total > 0 else 0.0
            issb_pillar = round(
                (mapped_count + enhanced_count * 0.6) / total * 100, 1
            ) if total > 0 else 0.0

            pillar_comparison[pillar] = {
                "tcfd_coverage": tcfd_pillar,
                "issb_coverage": issb_pillar,
                "gap": round(tcfd_pillar - issb_pillar, 1),
            }

        # Disclosure-level comparison
        disclosure_comparison: List[Dict[str, Any]] = []
        for code, data in TCFD_TO_ISSB_MAPPING.items():
            disclosure_comparison.append({
                "tcfd_code": code,
                "tcfd_ref": data["tcfd_ref"],
                "ifrs_s2_paragraphs": data["ifrs_s2_paragraphs"],
                "mapping_status": data["mapping_status"],
                "additional_requirements": len(data["additional_ifrs_s2"]),
                "tcfd_sufficient": data["mapping_status"] == "fully_mapped",
            })

        # Additional requirements status
        org_status = self._org_issb_status.get(org_id, {})
        additional_status: List[Dict[str, Any]] = []
        for req in ISSB_ADDITIONAL_REQUIREMENTS:
            status = org_status.get(req["id"], {}).get("status", "not_started")
            additional_status.append({
                "id": req["id"],
                "title": req["title"],
                "priority": req["priority"],
                "status": status,
                "effort_days": req["effort_days"],
            })

        # Recommendation
        score_gap = round(issb_result.issb_score - tcfd_score, 1)
        if issb_result.issb_score >= 80:
            recommendation = (
                "Organization is well-positioned for IFRS S2 compliance. "
                "Focus on closing remaining additional requirements."
            )
        elif issb_result.issb_score >= 50:
            recommendation = (
                "Organization has a solid TCFD foundation. Priority migration "
                "actions: transition plan, Scope 3 completeness, and current-period "
                "financial effects."
            )
        else:
            recommendation = (
                "Significant gaps remain for IFRS S2 compliance. Recommend "
                "completing TCFD base disclosures first, then systematically "
                "addressing the 8 additional IFRS S2 requirements."
            )

        scorecard = DualStandardScorecard(
            org_id=org_id,
            tcfd_overall=tcfd_score,
            issb_overall=issb_result.issb_score,
            pillar_comparison=pillar_comparison,
            disclosure_comparison=disclosure_comparison,
            additional_requirements_status=additional_status,
            recommendation=recommendation,
        )

        logger.info(
            "Dual scorecard for org %s: tcfd=%.1f%%, issb=%.1f%%",
            org_id, tcfd_score, issb_result.issb_score,
        )
        return scorecard
