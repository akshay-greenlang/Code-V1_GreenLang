# -*- coding: utf-8 -*-
"""
ComplianceScorecardEngine - Multi-Standard Compliance Scorecard Engine

This module implements a comprehensive multi-standard compliance assessment
engine for GL-VCCI Scope 3 Platform v1.1. It evaluates organizational
emissions data against five major reporting and disclosure standards,
identifies cross-standard gaps, generates prioritized action items,
and calculates an overall compliance score.

Standards assessed:
    - GHG Protocol (Corporate Standard, Scope 2 Guidance, Scope 3 Standard)
    - ESRS E1 (EU CSRD Climate Change disclosure)
    - CDP Climate Change Questionnaire
    - IFRS S2 (Climate-related Disclosures)
    - ISO 14083 (Transport-specific carbon accounting)

Version: 1.1.0
Date: 2026-03-01
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ComplianceRequirement(BaseModel):
    """A single requirement within a compliance standard."""
    id: str = Field(..., description="Unique requirement identifier")
    standard: str = Field(..., description="Standard name (e.g., GHG Protocol)")
    category: str = Field(default="", description="Requirement category")
    description: str = Field(..., description="Requirement description")
    criticality: str = Field(default="required", description="required, recommended, or optional")
    status: str = Field(default="not_assessed", description="met, partially_met, not_met, not_assessed, not_applicable")
    evidence: Optional[str] = Field(None, description="Evidence or data supporting the assessment")
    notes: str = Field(default="", description="Assessor notes")
    data_fields_required: List[str] = Field(default_factory=list, description="Input data fields needed")


class StandardCompliance(BaseModel):
    """Compliance assessment result for a single standard."""
    standard_name: str = Field(..., description="Standard display name")
    standard_code: str = Field(..., description="Standard code identifier")
    version: str = Field(default="", description="Standard version or year")
    total_requirements: int = Field(default=0, ge=0, description="Total requirements assessed")
    met_count: int = Field(default=0, ge=0, description="Requirements fully met")
    partially_met_count: int = Field(default=0, ge=0, description="Requirements partially met")
    not_met_count: int = Field(default=0, ge=0, description="Requirements not met")
    not_applicable_count: int = Field(default=0, ge=0, description="Requirements not applicable")
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Coverage percentage")
    requirements: List[ComplianceRequirement] = Field(default_factory=list, description="All requirements")
    predicted_score: Optional[str] = Field(None, description="Predicted score for scored standards (e.g., CDP)")
    summary: str = Field(default="", description="Assessment summary narrative")


class ComplianceGap(BaseModel):
    """Cross-standard compliance gap."""
    gap_id: str = Field(..., description="Unique gap identifier")
    standards_affected: List[str] = Field(..., description="Standards affected by this gap")
    category: str = Field(default="", description="Gap category")
    description: str = Field(..., description="Gap description")
    severity: str = Field(default="medium", description="critical, high, medium, low")
    current_state: str = Field(default="", description="Current state of the data or process")
    target_state: str = Field(default="", description="Required state for compliance")
    remediation_effort: str = Field(default="medium", description="Effort to close: low, medium, high")
    data_fields_needed: List[str] = Field(default_factory=list, description="Data fields needed to close the gap")


class ActionItem(BaseModel):
    """Prioritized action item for closing compliance gaps."""
    action_id: str = Field(..., description="Unique action identifier")
    title: str = Field(..., description="Action item title")
    description: str = Field(..., description="Detailed action description")
    priority: str = Field(default="medium", description="critical, high, medium, low")
    gap_ids: List[str] = Field(default_factory=list, description="Related gap IDs")
    standards_affected: List[str] = Field(default_factory=list, description="Standards this action supports")
    estimated_effort_hours: Optional[float] = Field(None, description="Estimated effort in hours")
    responsible_role: str = Field(default="", description="Recommended responsible role")
    deadline_recommendation: str = Field(default="", description="Recommended deadline")
    status: str = Field(default="open", description="open, in_progress, completed")


class ComplianceScorecard(BaseModel):
    """Complete multi-standard compliance scorecard."""
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Generation timestamp")
    company_name: str = Field(default="", description="Company name")
    reporting_year: int = Field(default=0, description="Reporting year")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Weighted overall compliance score")
    overall_grade: str = Field(default="", description="Letter grade (A+ through F)")
    standards: Dict[str, StandardCompliance] = Field(default_factory=dict, description="Per-standard results")
    gaps: List[ComplianceGap] = Field(default_factory=list, description="Cross-standard gaps")
    action_items: List[ActionItem] = Field(default_factory=list, description="Prioritized action items")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")


# ============================================================================
# EMBEDDED REQUIREMENTS DATA
# ============================================================================

GHG_PROTOCOL_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "GHG-001", "category": "Organizational Boundary", "description": "Define organizational boundary using operational control, financial control, or equity share approach", "criticality": "required", "data_fields": ["consolidation_approach"]},
    {"id": "GHG-002", "category": "Organizational Boundary", "description": "Document reasons for choosing the consolidation approach", "criticality": "required", "data_fields": ["consolidation_approach_rationale"]},
    {"id": "GHG-003", "category": "Operational Boundary", "description": "Categorize emissions into Scope 1 direct emissions", "criticality": "required", "data_fields": ["scope1_tco2e"]},
    {"id": "GHG-004", "category": "Operational Boundary", "description": "Categorize emissions into Scope 2 indirect emissions from purchased energy", "criticality": "required", "data_fields": ["scope2_location_tco2e", "scope2_market_tco2e"]},
    {"id": "GHG-005", "category": "Operational Boundary", "description": "Categorize emissions into Scope 3 other indirect emissions (all 15 categories screened)", "criticality": "required", "data_fields": ["scope3_categories"]},
    {"id": "GHG-006", "category": "Base Year", "description": "Establish a base year for emissions tracking and comparison", "criticality": "required", "data_fields": ["base_year", "base_year_emissions"]},
    {"id": "GHG-007", "category": "Base Year", "description": "Define base year recalculation policy with trigger thresholds", "criticality": "required", "data_fields": ["recalculation_policy"]},
    {"id": "GHG-008", "category": "Quantification", "description": "Use recognized calculation methodologies (emission factors, direct measurement, mass balance)", "criticality": "required", "data_fields": ["calculation_methodology"]},
    {"id": "GHG-009", "category": "Quantification", "description": "Report all seven Kyoto GHGs (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)", "criticality": "required", "data_fields": ["gases_reported"]},
    {"id": "GHG-010", "category": "Quantification", "description": "Use consistent GWP values (IPCC AR5 or AR6) and disclose source", "criticality": "required", "data_fields": ["gwp_source"]},
    {"id": "GHG-011", "category": "Scope 2", "description": "Report Scope 2 using both location-based and market-based methods (dual reporting)", "criticality": "required", "data_fields": ["scope2_location_tco2e", "scope2_market_tco2e"]},
    {"id": "GHG-012", "category": "Scope 2", "description": "Apply the Scope 2 quality criteria hierarchy for contractual instruments", "criticality": "recommended", "data_fields": ["scope2_instruments"]},
    {"id": "GHG-013", "category": "Scope 3", "description": "Screen all 15 Scope 3 categories for relevance and disclose exclusion rationale", "criticality": "required", "data_fields": ["scope3_categories", "scope3_exclusion_rationale"]},
    {"id": "GHG-014", "category": "Scope 3", "description": "Quantify emissions for all relevant Scope 3 categories", "criticality": "required", "data_fields": ["scope3_categories"]},
    {"id": "GHG-015", "category": "Scope 3", "description": "Apply data quality scoring (DQI) to Scope 3 calculations", "criticality": "recommended", "data_fields": ["avg_dqi_score", "data_quality_by_scope"]},
    {"id": "GHG-016", "category": "Uncertainty", "description": "Assess and disclose quantitative uncertainty for emissions estimates", "criticality": "recommended", "data_fields": ["uncertainty_results"]},
    {"id": "GHG-017", "category": "Reporting", "description": "Report total Scope 1, Scope 2, and Scope 3 emissions in tCO2e", "criticality": "required", "data_fields": ["scope1_tco2e", "scope2_location_tco2e", "scope3_tco2e"]},
    {"id": "GHG-018", "category": "Reporting", "description": "Report emissions intensity metrics (per revenue, per FTE, per product unit)", "criticality": "recommended", "data_fields": ["intensity_per_revenue", "intensity_per_fte"]},
    {"id": "GHG-019", "category": "Reporting", "description": "Report year-over-year emissions changes with explanations", "criticality": "recommended", "data_fields": ["prior_year_emissions", "yoy_change_pct"]},
    {"id": "GHG-020", "category": "Verification", "description": "Obtain third-party verification of emissions data (limited or reasonable assurance)", "criticality": "recommended", "data_fields": ["verification_status"]},
    {"id": "GHG-021", "category": "Completeness", "description": "Disclose any material exclusions and their estimated percentage impact", "criticality": "required", "data_fields": ["exclusions"]},
    {"id": "GHG-022", "category": "Completeness", "description": "Document emission factor sources and vintage", "criticality": "required", "data_fields": ["emission_factor_sources"]},
    {"id": "GHG-023", "category": "Temporal", "description": "Report data for a complete 12-month reporting period", "criticality": "required", "data_fields": ["reporting_period_start", "reporting_period_end"]},
    {"id": "GHG-024", "category": "Biogenic", "description": "Report biogenic CO2 emissions separately from Scope totals", "criticality": "recommended", "data_fields": ["biogenic_emissions"]},
    {"id": "GHG-025", "category": "Provenance", "description": "Maintain data provenance chain from source to reported figure", "criticality": "recommended", "data_fields": ["provenance_chains"]},
]

ESRS_E1_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "ESRS-001", "category": "Transition Plan", "description": "Disclose the transition plan for climate change mitigation (E1-1)", "criticality": "required", "data_fields": ["transition_plan"]},
    {"id": "ESRS-002", "category": "Policies", "description": "Disclose policies adopted to manage climate change (E1-2)", "criticality": "required", "data_fields": ["climate_policies"]},
    {"id": "ESRS-003", "category": "Actions", "description": "Disclose actions and resources in relation to climate change (E1-3)", "criticality": "required", "data_fields": ["climate_actions"]},
    {"id": "ESRS-004", "category": "Targets", "description": "Disclose emission reduction targets aligned with Paris Agreement (E1-4)", "criticality": "required", "data_fields": ["targets", "sbti_status"]},
    {"id": "ESRS-005", "category": "Energy", "description": "Disclose energy consumption and mix (E1-5)", "criticality": "required", "data_fields": ["total_energy_mwh", "renewable_pct", "energy_by_source"]},
    {"id": "ESRS-006", "category": "Emissions", "description": "Disclose Scope 1 GHG emissions (E1-6)", "criticality": "required", "data_fields": ["scope1_tco2e"]},
    {"id": "ESRS-007", "category": "Emissions", "description": "Disclose Scope 2 GHG emissions (location and market-based) (E1-6)", "criticality": "required", "data_fields": ["scope2_location_tco2e", "scope2_market_tco2e"]},
    {"id": "ESRS-008", "category": "Emissions", "description": "Disclose Scope 3 GHG emissions by significant category (E1-6)", "criticality": "required", "data_fields": ["scope3_tco2e", "scope3_categories"]},
    {"id": "ESRS-009", "category": "Emissions", "description": "Disclose total GHG emissions (Scope 1+2+3) (E1-6)", "criticality": "required", "data_fields": ["total_tco2e"]},
    {"id": "ESRS-010", "category": "Removals", "description": "Disclose GHG removals and storage in own operations and value chain (E1-7)", "criticality": "recommended", "data_fields": ["ghg_removals"]},
    {"id": "ESRS-011", "category": "Carbon Pricing", "description": "Disclose internal carbon pricing mechanisms (E1-8)", "criticality": "recommended", "data_fields": ["internal_carbon_price"]},
    {"id": "ESRS-012", "category": "Financial Effects", "description": "Disclose anticipated financial effects of physical and transition risks (E1-9)", "criticality": "required", "data_fields": ["financial_effects"]},
    {"id": "ESRS-013", "category": "Base Year", "description": "Disclose base year and base year emissions with recalculation policy", "criticality": "required", "data_fields": ["base_year", "base_year_emissions"]},
    {"id": "ESRS-014", "category": "Intensity", "description": "Disclose GHG intensity per net revenue", "criticality": "required", "data_fields": ["intensity_per_revenue"]},
    {"id": "ESRS-015", "category": "Methodology", "description": "Disclose calculation methodologies and emission factor sources", "criticality": "required", "data_fields": ["calculation_methodology"]},
    {"id": "ESRS-016", "category": "Materiality", "description": "Perform double materiality assessment for climate change", "criticality": "required", "data_fields": ["materiality_assessment"]},
    {"id": "ESRS-017", "category": "Verification", "description": "Limited assurance on sustainability reporting (ESRS requirement)", "criticality": "required", "data_fields": ["verification_status"]},
    {"id": "ESRS-018", "category": "Carbon Credits", "description": "Disclose use of carbon credits and their role relative to emission reduction targets", "criticality": "recommended", "data_fields": ["carbon_credits"]},
    {"id": "ESRS-019", "category": "Governance", "description": "Disclose board and management responsibility for climate change", "criticality": "required", "data_fields": ["governance_data"]},
    {"id": "ESRS-020", "category": "Comparative", "description": "Provide comparative data for at least the preceding period", "criticality": "required", "data_fields": ["prior_year_emissions"]},
]

CDP_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "CDP-001", "category": "Introduction", "description": "Complete C0 Introduction section with company details and reporting boundary", "criticality": "required", "data_fields": ["company_info"]},
    {"id": "CDP-002", "category": "Governance", "description": "Disclose board-level oversight and management responsibility (C1)", "criticality": "required", "data_fields": ["governance_data"]},
    {"id": "CDP-003", "category": "Risks", "description": "Identify and assess climate-related risks and opportunities (C2)", "criticality": "required", "data_fields": ["risks_data"]},
    {"id": "CDP-004", "category": "Strategy", "description": "Describe business strategy integration and transition plan (C3)", "criticality": "required", "data_fields": ["targets_data"]},
    {"id": "CDP-005", "category": "Targets", "description": "Disclose emission reduction targets and progress (C4)", "criticality": "required", "data_fields": ["targets_data"]},
    {"id": "CDP-006", "category": "Methodology", "description": "Disclose emissions accounting methodology (C5)", "criticality": "required", "data_fields": ["emissions_data"]},
    {"id": "CDP-007", "category": "Emissions", "description": "Report Scope 1, 2, 3 emissions with breakdowns (C6)", "criticality": "required", "data_fields": ["scope1_tco2e", "scope2_location_tco2e", "scope3_categories"]},
    {"id": "CDP-008", "category": "Breakdown", "description": "Provide emissions breakdown by gas, country, division (C7)", "criticality": "required", "data_fields": ["scope1_by_gas", "scope1_by_country"]},
    {"id": "CDP-009", "category": "Energy", "description": "Report energy consumption, renewable %, and targets (C8)", "criticality": "required", "data_fields": ["total_energy_mwh", "renewable_pct"]},
    {"id": "CDP-010", "category": "Metrics", "description": "Report additional climate-related intensity metrics (C9)", "criticality": "recommended", "data_fields": ["intensity_per_revenue"]},
    {"id": "CDP-011", "category": "Verification", "description": "Disclose third-party verification status and attach statements (C10)", "criticality": "required", "data_fields": ["verification_status"]},
    {"id": "CDP-012", "category": "Carbon Pricing", "description": "Disclose participation in carbon pricing and internal carbon price (C11)", "criticality": "recommended", "data_fields": ["ets_exposure", "internal_carbon_price"]},
    {"id": "CDP-013", "category": "Engagement", "description": "Disclose value chain engagement on climate issues (C12)", "criticality": "required", "data_fields": ["engagement_data"]},
    {"id": "CDP-014", "category": "Targets", "description": "SBTi target validation or commitment", "criticality": "recommended", "data_fields": ["sbti_status"]},
    {"id": "CDP-015", "category": "Completeness", "description": "Achieve 90%+ questionnaire completion rate", "criticality": "recommended", "data_fields": ["cdp_completion_pct"]},
]

IFRS_S2_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "IFRS-001", "category": "Governance", "description": "Disclose governance body(ies) or individual(s) responsible for oversight of climate-related risks and opportunities", "criticality": "required", "data_fields": ["governance_data"]},
    {"id": "IFRS-002", "category": "Governance", "description": "Disclose management's role in assessing and managing climate-related risks and opportunities", "criticality": "required", "data_fields": ["governance_data"]},
    {"id": "IFRS-003", "category": "Strategy", "description": "Disclose climate-related risks and opportunities that could affect the entity's prospects", "criticality": "required", "data_fields": ["risks_data"]},
    {"id": "IFRS-004", "category": "Strategy", "description": "Describe the effects of climate-related risks and opportunities on business model and value chain", "criticality": "required", "data_fields": ["risks_data"]},
    {"id": "IFRS-005", "category": "Strategy", "description": "Describe the effects of climate-related risks on strategy and decision-making", "criticality": "required", "data_fields": ["risks_data"]},
    {"id": "IFRS-006", "category": "Strategy", "description": "Describe the current and anticipated financial effects of climate-related risks and opportunities", "criticality": "required", "data_fields": ["financial_effects"]},
    {"id": "IFRS-007", "category": "Strategy", "description": "Disclose climate resilience assessment including scenario analysis", "criticality": "required", "data_fields": ["scenario_analysis"]},
    {"id": "IFRS-008", "category": "Risk Management", "description": "Describe the processes used to identify, assess, prioritize, and monitor climate-related risks", "criticality": "required", "data_fields": ["risk_process"]},
    {"id": "IFRS-009", "category": "Risk Management", "description": "Describe how climate-related risk management is integrated into overall risk management", "criticality": "required", "data_fields": ["risk_integration"]},
    {"id": "IFRS-010", "category": "Metrics", "description": "Disclose Scope 1 GHG emissions", "criticality": "required", "data_fields": ["scope1_tco2e"]},
    {"id": "IFRS-011", "category": "Metrics", "description": "Disclose Scope 2 GHG emissions", "criticality": "required", "data_fields": ["scope2_location_tco2e", "scope2_market_tco2e"]},
    {"id": "IFRS-012", "category": "Metrics", "description": "Disclose Scope 3 GHG emissions", "criticality": "required", "data_fields": ["scope3_tco2e"]},
    {"id": "IFRS-013", "category": "Metrics", "description": "Disclose amount and percentage of assets or business activities vulnerable to climate-related risks", "criticality": "required", "data_fields": ["assets_at_risk"]},
    {"id": "IFRS-014", "category": "Targets", "description": "Disclose climate-related targets set by the entity, including emission reduction targets", "criticality": "required", "data_fields": ["targets"]},
    {"id": "IFRS-015", "category": "Targets", "description": "Disclose progress against each climate-related target and any revisions", "criticality": "required", "data_fields": ["target_progress"]},
]

ISO_14083_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "ISO-001", "category": "Scope", "description": "Define the transport chain or transport service under assessment", "criticality": "required", "data_fields": ["transport_chain"]},
    {"id": "ISO-002", "category": "Modes", "description": "Report emissions by transport mode (road, rail, maritime, air, pipeline, intermodal)", "criticality": "required", "data_fields": ["transport_by_mode"]},
    {"id": "ISO-003", "category": "WTW", "description": "Apply well-to-wheel (WTW) system boundary for all transport modes", "criticality": "required", "data_fields": ["wtw_boundary"]},
    {"id": "ISO-004", "category": "Allocation", "description": "Apply appropriate allocation method for shared transport (mass, volume, TEU, etc.)", "criticality": "required", "data_fields": ["allocation_method"]},
    {"id": "ISO-005", "category": "Emission Factors", "description": "Use emission factors from recognized sources (DEFRA, EPA, GLEC, ICAO, IMO)", "criticality": "required", "data_fields": ["emission_factor_sources"]},
    {"id": "ISO-006", "category": "Data Quality", "description": "Assess and disclose data quality using the ISO 14083 data quality framework", "criticality": "required", "data_fields": ["data_quality_score"]},
    {"id": "ISO-007", "category": "Methodology", "description": "Document calculation methodology (distance-based, fuel-based, spend-based)", "criticality": "required", "data_fields": ["calculation_methodology"]},
    {"id": "ISO-008", "category": "Reporting", "description": "Report total transport emissions in tCO2e and per tonne-km", "criticality": "required", "data_fields": ["total_emissions_tco2e", "total_tonne_km"]},
    {"id": "ISO-009", "category": "Chain Approach", "description": "Apply the transport chain approach for multi-leg shipments", "criticality": "recommended", "data_fields": ["multi_leg_chains"]},
    {"id": "ISO-010", "category": "Verification", "description": "Obtain independent verification of transport emissions calculations", "criticality": "recommended", "data_fields": ["transport_verification"]},
]


# ============================================================================
# COMPLIANCE SCORECARD ENGINE
# ============================================================================

class ComplianceScorecardEngine:
    """
    Multi-standard compliance scorecard engine.

    Evaluates emissions data, company information, and CDP questionnaire
    responses against five major reporting standards. Produces a unified
    scorecard with per-standard coverage, cross-standard gap analysis,
    and prioritized action items.

    All assessments use deterministic rules (no LLM in the assessment path).

    Example:
        >>> engine = ComplianceScorecardEngine()
        >>> scorecard = engine.generate_scorecard(
        ...     emissions_data=emissions,
        ...     company_info=company,
        ...     cdp_questionnaire=cdp_response,
        ...     targets_data=targets,
        ... )
        >>> print(f"Overall score: {scorecard.overall_score:.1f}%")
    """

    # Standard weights for overall score calculation
    STANDARD_WEIGHTS: Dict[str, float] = {
        "ghg_protocol": 0.30,
        "esrs_e1": 0.25,
        "cdp": 0.20,
        "ifrs_s2": 0.15,
        "iso_14083": 0.10,
    }

    def __init__(self) -> None:
        """Initialize the ComplianceScorecardEngine."""
        logger.info("ComplianceScorecardEngine initialized (v1.1.0)")

    def generate_scorecard(
        self,
        emissions_data: Dict[str, Any],
        company_info: Dict[str, Any],
        cdp_questionnaire: Optional[Dict[str, Any]] = None,
        targets_data: Optional[Dict[str, Any]] = None,
        risks_data: Optional[Dict[str, Any]] = None,
        governance_data: Optional[Dict[str, Any]] = None,
        transport_data: Optional[Dict[str, Any]] = None,
    ) -> ComplianceScorecard:
        """
        Generate a complete multi-standard compliance scorecard.

        Args:
            emissions_data: Scope 1/2/3 emissions data.
            company_info: Company information.
            cdp_questionnaire: CDP questionnaire response (optional).
            targets_data: Emission reduction targets (optional).
            risks_data: Climate risk data (optional).
            governance_data: Governance data (optional).
            transport_data: Transport data for ISO 14083 (optional).

        Returns:
            ComplianceScorecard with per-standard and overall results.
        """
        start_time = datetime.utcnow()
        all_data = self._merge_data_sources(
            emissions_data, company_info, cdp_questionnaire,
            targets_data, risks_data, governance_data, transport_data,
        )

        logger.info("Generating compliance scorecard for %s", company_info.get("name", "Unknown"))

        # Assess each standard
        standards: Dict[str, StandardCompliance] = {}
        standards["ghg_protocol"] = self.assess_ghg_protocol(all_data)
        standards["esrs_e1"] = self.assess_esrs_e1(all_data)
        standards["cdp"] = self.assess_cdp(all_data, cdp_questionnaire)
        standards["ifrs_s2"] = self.assess_ifrs_s2(all_data)
        standards["iso_14083"] = self.assess_iso_14083(all_data)

        # Cross-standard gap analysis
        gaps = self.identify_cross_standard_gaps(standards)

        # Action items
        action_items = self.generate_action_items(gaps)

        # Overall score
        overall_score = self.calculate_overall_score(standards)
        overall_grade = self._score_to_grade(overall_score)

        # Provenance
        provenance_str = json.dumps({
            "company": company_info.get("name", ""),
            "generated_at": start_time.isoformat(),
            "overall_score": overall_score,
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "Compliance scorecard generated: overall %.1f%% (%s), %d gaps, %d actions, %.0fms",
            overall_score, overall_grade, len(gaps), len(action_items), elapsed_ms,
        )

        return ComplianceScorecard(
            generated_at=start_time.isoformat(),
            company_name=company_info.get("name", ""),
            reporting_year=company_info.get("reporting_year", 0),
            overall_score=round(overall_score, 1),
            overall_grade=overall_grade,
            standards=standards,
            gaps=gaps,
            action_items=action_items,
            provenance_hash=provenance_hash,
        )

    # ========================================================================
    # PER-STANDARD ASSESSMENT METHODS
    # ========================================================================

    def assess_ghg_protocol(self, data: Dict[str, Any]) -> StandardCompliance:
        """
        Assess compliance with the GHG Protocol Corporate Standard.

        Evaluates 25 requirements covering organizational boundary, operational
        boundary, base year, quantification, reporting, and verification.

        Args:
            data: Merged data from all sources.

        Returns:
            StandardCompliance for GHG Protocol.
        """
        requirements = self._assess_requirements(GHG_PROTOCOL_REQUIREMENTS, data)
        met, partial, not_met, na = self._count_statuses(requirements)
        total = len(requirements) - na
        coverage = (met + partial * 0.5) / total * 100.0 if total > 0 else 0.0

        return StandardCompliance(
            standard_name="GHG Protocol",
            standard_code="ghg_protocol",
            version="Corporate Standard + Scope 2 Guidance + Scope 3 Standard",
            total_requirements=len(requirements),
            met_count=met,
            partially_met_count=partial,
            not_met_count=not_met,
            not_applicable_count=na,
            coverage_pct=round(coverage, 1),
            requirements=requirements,
            summary=f"GHG Protocol compliance: {coverage:.0f}% coverage ({met} met, {partial} partial, {not_met} not met).",
        )

    def assess_esrs_e1(self, data: Dict[str, Any]) -> StandardCompliance:
        """
        Assess compliance with ESRS E1 (EU CSRD Climate Change).

        Evaluates 20 requirements covering transition plan, policies, actions,
        targets, energy, emissions (Scope 1/2/3), removals, carbon pricing,
        financial effects, and verification.

        Args:
            data: Merged data from all sources.

        Returns:
            StandardCompliance for ESRS E1.
        """
        requirements = self._assess_requirements(ESRS_E1_REQUIREMENTS, data)
        met, partial, not_met, na = self._count_statuses(requirements)
        total = len(requirements) - na
        coverage = (met + partial * 0.5) / total * 100.0 if total > 0 else 0.0

        return StandardCompliance(
            standard_name="ESRS E1 (EU CSRD)",
            standard_code="esrs_e1",
            version="ESRS Set 1 (2024)",
            total_requirements=len(requirements),
            met_count=met,
            partially_met_count=partial,
            not_met_count=not_met,
            not_applicable_count=na,
            coverage_pct=round(coverage, 1),
            requirements=requirements,
            summary=f"ESRS E1 compliance: {coverage:.0f}% coverage ({met} met, {partial} partial, {not_met} not met).",
        )

    def assess_cdp(
        self,
        data: Dict[str, Any],
        cdp_questionnaire: Optional[Dict[str, Any]] = None,
    ) -> StandardCompliance:
        """
        Assess CDP Climate Change questionnaire compliance.

        Evaluates 15 requirements covering all 13 sections plus verification
        and targets. If a CDP questionnaire response is provided, uses its
        completion percentage for a more accurate score prediction.

        Args:
            data: Merged data from all sources.
            cdp_questionnaire: Optional CDP questionnaire response.

        Returns:
            StandardCompliance for CDP with predicted score.
        """
        requirements = self._assess_requirements(CDP_REQUIREMENTS, data)
        met, partial, not_met, na = self._count_statuses(requirements)
        total = len(requirements) - na
        coverage = (met + partial * 0.5) / total * 100.0 if total > 0 else 0.0

        # Predict CDP score from completion
        completion_pct = 0.0
        if cdp_questionnaire:
            completion_pct = cdp_questionnaire.get("overall_completion_pct", 0.0)
        else:
            completion_pct = coverage

        predicted_score = self._predict_cdp_score_from_coverage(completion_pct)

        return StandardCompliance(
            standard_name="CDP Climate Change",
            standard_code="cdp",
            version="2025",
            total_requirements=len(requirements),
            met_count=met,
            partially_met_count=partial,
            not_met_count=not_met,
            not_applicable_count=na,
            coverage_pct=round(coverage, 1),
            requirements=requirements,
            predicted_score=predicted_score,
            summary=f"CDP compliance: {coverage:.0f}% coverage, predicted score: {predicted_score}.",
        )

    def assess_ifrs_s2(self, data: Dict[str, Any]) -> StandardCompliance:
        """
        Assess compliance with IFRS S2 Climate-related Disclosures.

        Evaluates 15 requirements across the four TCFD pillars (Governance,
        Strategy, Risk Management, Metrics & Targets).

        Args:
            data: Merged data from all sources.

        Returns:
            StandardCompliance for IFRS S2.
        """
        requirements = self._assess_requirements(IFRS_S2_REQUIREMENTS, data)
        met, partial, not_met, na = self._count_statuses(requirements)
        total = len(requirements) - na
        coverage = (met + partial * 0.5) / total * 100.0 if total > 0 else 0.0

        return StandardCompliance(
            standard_name="IFRS S2 Climate-related Disclosures",
            standard_code="ifrs_s2",
            version="ISSB S2 (2023)",
            total_requirements=len(requirements),
            met_count=met,
            partially_met_count=partial,
            not_met_count=not_met,
            not_applicable_count=na,
            coverage_pct=round(coverage, 1),
            requirements=requirements,
            summary=f"IFRS S2 compliance: {coverage:.0f}% coverage ({met} met, {partial} partial, {not_met} not met).",
        )

    def assess_iso_14083(self, data: Dict[str, Any]) -> StandardCompliance:
        """
        Assess compliance with ISO 14083 transport-specific requirements.

        Evaluates 10 requirements covering transport chain definition,
        modal breakdown, WTW boundary, allocation, emission factors,
        data quality, and verification.

        Args:
            data: Merged data from all sources.

        Returns:
            StandardCompliance for ISO 14083.
        """
        # Check if transport data exists at all
        has_transport = bool(data.get("transport_by_mode") or data.get("transport_chain"))

        if not has_transport:
            # All N/A if no transport data
            requirements = []
            for req_def in ISO_14083_REQUIREMENTS:
                requirements.append(ComplianceRequirement(
                    id=req_def["id"],
                    standard="ISO 14083",
                    category=req_def["category"],
                    description=req_def["description"],
                    criticality=req_def["criticality"],
                    status="not_applicable",
                    notes="No transport data provided; ISO 14083 assessment not applicable.",
                ))
            return StandardCompliance(
                standard_name="ISO 14083 Transport Emissions",
                standard_code="iso_14083",
                version="ISO 14083:2023",
                total_requirements=len(requirements),
                met_count=0,
                partially_met_count=0,
                not_met_count=0,
                not_applicable_count=len(requirements),
                coverage_pct=0.0,
                requirements=requirements,
                summary="ISO 14083 not applicable: no transport data provided.",
            )

        requirements = self._assess_requirements(ISO_14083_REQUIREMENTS, data)
        met, partial, not_met, na = self._count_statuses(requirements)
        total = len(requirements) - na
        coverage = (met + partial * 0.5) / total * 100.0 if total > 0 else 0.0

        return StandardCompliance(
            standard_name="ISO 14083 Transport Emissions",
            standard_code="iso_14083",
            version="ISO 14083:2023",
            total_requirements=len(requirements),
            met_count=met,
            partially_met_count=partial,
            not_met_count=not_met,
            not_applicable_count=na,
            coverage_pct=round(coverage, 1),
            requirements=requirements,
            summary=f"ISO 14083 compliance: {coverage:.0f}% coverage ({met} met, {partial} partial, {not_met} not met).",
        )

    # ========================================================================
    # CROSS-STANDARD ANALYSIS
    # ========================================================================

    def identify_cross_standard_gaps(self, standards: Dict[str, StandardCompliance]) -> List[ComplianceGap]:
        """
        Identify gaps that affect multiple standards.

        Groups unmet requirements by data field to find common gaps, then
        assigns severity based on the number of standards affected and
        whether the requirements are mandatory.

        Args:
            standards: Per-standard compliance results.

        Returns:
            List of ComplianceGap objects sorted by severity.
        """
        # Collect all not-met and partially-met requirements grouped by data field
        field_to_standards: Dict[str, List[Dict[str, Any]]] = {}

        for std_code, std_compliance in standards.items():
            for req in std_compliance.requirements:
                if req.status in ("not_met", "partially_met"):
                    for field in req.data_fields_required:
                        if field not in field_to_standards:
                            field_to_standards[field] = []
                        field_to_standards[field].append({
                            "standard": std_code,
                            "requirement_id": req.id,
                            "description": req.description,
                            "criticality": req.criticality,
                            "status": req.status,
                        })

        gaps: List[ComplianceGap] = []
        gap_counter = 0

        for field, affected in field_to_standards.items():
            affected_standards = list(set(a["standard"] for a in affected))
            has_required = any(a["criticality"] == "required" for a in affected)

            # Severity based on breadth and criticality
            if len(affected_standards) >= 3 and has_required:
                severity = "critical"
            elif len(affected_standards) >= 2 and has_required:
                severity = "high"
            elif has_required:
                severity = "medium"
            else:
                severity = "low"

            # Remediation effort
            if len(affected_standards) >= 3:
                effort = "high"
            elif len(affected_standards) >= 2:
                effort = "medium"
            else:
                effort = "low"

            gap_counter += 1
            gaps.append(ComplianceGap(
                gap_id=f"GAP-{gap_counter:03d}",
                standards_affected=affected_standards,
                category=field.replace("_", " ").title(),
                description=f"Data field '{field}' is missing or incomplete, affecting {len(affected)} requirement(s) across {len(affected_standards)} standard(s).",
                severity=severity,
                current_state=f"Field '{field}' not provided or insufficient.",
                target_state=f"Provide complete '{field}' data to satisfy {', '.join(affected_standards)}.",
                remediation_effort=effort,
                data_fields_needed=[field],
            ))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 4))

        return gaps

    def generate_action_items(self, gaps: List[ComplianceGap]) -> List[ActionItem]:
        """
        Generate prioritized action items from compliance gaps.

        Groups related gaps and creates actionable items with estimated
        effort, responsible roles, and deadline recommendations.

        Args:
            gaps: List of identified compliance gaps.

        Returns:
            List of ActionItem objects sorted by priority.
        """
        actions: List[ActionItem] = []
        action_counter = 0

        # Group gaps by category for consolidated actions
        category_gaps: Dict[str, List[ComplianceGap]] = {}
        for gap in gaps:
            cat = gap.category
            if cat not in category_gaps:
                category_gaps[cat] = []
            category_gaps[cat].append(gap)

        for category, cat_gaps in category_gaps.items():
            # Determine priority from highest severity in group
            severities = [g.severity for g in cat_gaps]
            if "critical" in severities:
                priority = "critical"
            elif "high" in severities:
                priority = "high"
            elif "medium" in severities:
                priority = "medium"
            else:
                priority = "low"

            all_standards = list(set(
                std for g in cat_gaps for std in g.standards_affected
            ))
            all_gap_ids = [g.gap_id for g in cat_gaps]
            all_fields = list(set(
                f for g in cat_gaps for f in g.data_fields_needed
            ))

            # Estimate effort
            effort = len(all_fields) * 4.0  # ~4 hours per data field

            # Deadline recommendation
            if priority == "critical":
                deadline = "Within 2 weeks"
                role = "Sustainability Director"
            elif priority == "high":
                deadline = "Within 1 month"
                role = "ESG Data Manager"
            elif priority == "medium":
                deadline = "Within 3 months"
                role = "ESG Analyst"
            else:
                deadline = "Next reporting cycle"
                role = "ESG Analyst"

            action_counter += 1
            actions.append(ActionItem(
                action_id=f"ACT-{action_counter:03d}",
                title=f"Provide {category} data for multi-standard compliance",
                description=(
                    f"Collect and integrate data for: {', '.join(all_fields)}. "
                    f"This will close {len(cat_gaps)} gap(s) affecting {', '.join(all_standards)}."
                ),
                priority=priority,
                gap_ids=all_gap_ids,
                standards_affected=all_standards,
                estimated_effort_hours=effort,
                responsible_role=role,
                deadline_recommendation=deadline,
            ))

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        actions.sort(key=lambda a: priority_order.get(a.priority, 4))

        return actions

    def calculate_overall_score(self, standards: Dict[str, StandardCompliance]) -> float:
        """
        Calculate the weighted overall compliance score.

        Weights:
            - GHG Protocol: 30%
            - ESRS E1: 25%
            - CDP: 20%
            - IFRS S2: 15%
            - ISO 14083: 10%

        Args:
            standards: Per-standard compliance results.

        Returns:
            Weighted overall score (0.0 to 100.0).
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for std_code, weight in self.STANDARD_WEIGHTS.items():
            std = standards.get(std_code)
            if std and std.not_applicable_count < std.total_requirements:
                weighted_sum += std.coverage_pct * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 1)

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _merge_data_sources(
        self,
        emissions_data: Dict[str, Any],
        company_info: Dict[str, Any],
        cdp_questionnaire: Optional[Dict[str, Any]],
        targets_data: Optional[Dict[str, Any]],
        risks_data: Optional[Dict[str, Any]],
        governance_data: Optional[Dict[str, Any]],
        transport_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge all data sources into a single flat dictionary for requirement evaluation."""
        merged: Dict[str, Any] = {}

        # Emissions data
        if emissions_data:
            merged.update(emissions_data)

        # Company info
        if company_info:
            merged["company_info"] = company_info
            for key in ["name", "reporting_year", "consolidation_approach", "verification_status",
                        "ets_exposure", "internal_carbon_price"]:
                if key in company_info:
                    merged[key] = company_info[key]

        # Targets
        if targets_data:
            merged["targets_data"] = targets_data
            merged["targets"] = targets_data.get("absolute_targets", [])
            if "sbti_status" in targets_data:
                merged["sbti_status"] = targets_data["sbti_status"]
            if "transition_plan" in targets_data or "has_transition_plan" in targets_data:
                merged["transition_plan"] = targets_data.get("transition_plan", targets_data.get("has_transition_plan"))
            if "target_progress" in targets_data:
                merged["target_progress"] = targets_data["target_progress"]

        # Risks
        if risks_data:
            merged["risks_data"] = risks_data
            if "scenario_analysis" in risks_data:
                merged["scenario_analysis"] = risks_data["scenario_analysis"]
            if "risk_process_details" in risks_data:
                merged["risk_process"] = risks_data["risk_process_details"]
            if "financial_effects" in risks_data:
                merged["financial_effects"] = risks_data["financial_effects"]

        # Governance
        if governance_data:
            merged["governance_data"] = governance_data

        # Transport
        if transport_data:
            for key in ["transport_by_mode", "transport_chain", "total_tonne_km",
                        "total_emissions_tco2e", "data_quality_score", "allocation_method",
                        "multi_leg_chains", "wtw_boundary", "transport_verification"]:
                if key in transport_data:
                    merged[key] = transport_data[key]

        # CDP questionnaire
        if cdp_questionnaire:
            merged["cdp_completion_pct"] = cdp_questionnaire.get("overall_completion_pct", 0.0)
            merged["engagement_data"] = cdp_questionnaire.get("sections", {}).get("C12", {})

        return merged

    def _assess_requirements(
        self,
        requirement_defs: List[Dict[str, Any]],
        data: Dict[str, Any],
    ) -> List[ComplianceRequirement]:
        """Assess a list of requirements against the provided data."""
        results: List[ComplianceRequirement] = []

        for req_def in requirement_defs:
            fields = req_def.get("data_fields", [])
            status, evidence = self._evaluate_requirement_fields(fields, data)

            results.append(ComplianceRequirement(
                id=req_def["id"],
                standard=req_def.get("standard", ""),
                category=req_def.get("category", ""),
                description=req_def["description"],
                criticality=req_def.get("criticality", "required"),
                status=status,
                evidence=evidence,
                data_fields_required=fields,
            ))

        return results

    def _evaluate_requirement_fields(
        self,
        fields: List[str],
        data: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluate requirement based on presence and quality of data fields.

        Returns:
            Tuple of (status, evidence_str).
        """
        if not fields:
            return "met", "No specific data fields required."

        present_count = 0
        evidence_parts: List[str] = []

        for field in fields:
            value = data.get(field)
            if value is not None:
                if isinstance(value, (list, dict)):
                    if len(value) > 0:
                        present_count += 1
                        evidence_parts.append(f"{field}: provided ({type(value).__name__}, {len(value)} items)")
                    else:
                        evidence_parts.append(f"{field}: empty collection")
                elif isinstance(value, (int, float)):
                    if value > 0:
                        present_count += 1
                        evidence_parts.append(f"{field}: {value}")
                    else:
                        # Zero values count as partially met
                        present_count += 0.5
                        evidence_parts.append(f"{field}: {value} (zero value)")
                elif isinstance(value, str):
                    if value.strip():
                        present_count += 1
                        evidence_parts.append(f"{field}: provided")
                    else:
                        evidence_parts.append(f"{field}: empty string")
                elif isinstance(value, bool):
                    present_count += 1
                    evidence_parts.append(f"{field}: {value}")
                else:
                    present_count += 1
                    evidence_parts.append(f"{field}: provided")
            else:
                evidence_parts.append(f"{field}: not provided")

        evidence_str = "; ".join(evidence_parts) if evidence_parts else None

        total_fields = len(fields)
        ratio = present_count / total_fields if total_fields > 0 else 0.0

        if ratio >= 1.0:
            return "met", evidence_str
        elif ratio >= 0.5:
            return "partially_met", evidence_str
        else:
            return "not_met", evidence_str

    def _count_statuses(
        self,
        requirements: List[ComplianceRequirement],
    ) -> Tuple[int, int, int, int]:
        """Count requirements by status. Returns (met, partial, not_met, not_applicable)."""
        met = sum(1 for r in requirements if r.status == "met")
        partial = sum(1 for r in requirements if r.status == "partially_met")
        not_met = sum(1 for r in requirements if r.status == "not_met")
        na = sum(1 for r in requirements if r.status == "not_applicable")
        return met, partial, not_met, na

    def _predict_cdp_score_from_coverage(self, coverage_pct: float) -> str:
        """Predict CDP letter score from coverage percentage."""
        if coverage_pct >= 90:
            return "A"
        elif coverage_pct >= 85:
            return "A-"
        elif coverage_pct >= 75:
            return "B"
        elif coverage_pct >= 65:
            return "B-"
        elif coverage_pct >= 50:
            return "C"
        elif coverage_pct >= 40:
            return "C-"
        elif coverage_pct >= 25:
            return "D"
        else:
            return "D-"

    def _score_to_grade(self, score: float) -> str:
        """Convert a numeric score (0-100) to a letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D+"
        elif score >= 45:
            return "D"
        elif score >= 40:
            return "D-"
        else:
            return "F"


__all__ = [
    "ComplianceScorecardEngine",
    "ComplianceScorecard",
    "StandardCompliance",
    "ComplianceRequirement",
    "ComplianceGap",
    "ActionItem",
]
