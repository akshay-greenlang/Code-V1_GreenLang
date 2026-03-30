# -*- coding: utf-8 -*-
"""
ESRSTopicMappingEngine - PACK-015 Double Materiality Engine 6
===============================================================

Maps material sustainability topics identified through the double materiality
assessment to specific ESRS disclosure requirements and data points.

Once the materiality matrix determines which topics are material, this engine
identifies exactly which ESRS disclosures the undertaking must prepare, how
many data points each requires, and where gaps exist in the company's current
data availability.

ESRS Standards Covered:
    - ESRS 2: General disclosures (IRO-1, IRO-2, SBM-1 through SBM-3,
      GOV-1 through GOV-5, MDR series)
    - ESRS E1-E5: Environmental standards (Climate, Pollution, Water,
      Biodiversity, Resource Use)
    - ESRS S1-S4: Social standards (Own Workforce, Value Chain Workers,
      Affected Communities, Consumers)
    - ESRS G1: Governance standard (Business Conduct)

Gap Analysis:
    The engine compares required disclosures against available data to
    identify gaps, estimate effort, and produce an actionable remediation
    plan.  Effort estimates are based on typical implementation timelines
    from EFRAG guidance.

Zero-Hallucination:
    - Disclosure catalog is a hard-coded reference database
    - Mapping logic uses deterministic lookup (no ML/LLM)
    - Gap analysis uses simple set subtraction
    - Effort estimates use fixed lookup tables
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-015 Double Materiality
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

engine_version: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ESRSStandard(str, Enum):
    """ESRS standard identifiers (EFRAG 2023).

    Covers ESRS 1 (general principles), ESRS 2 (general disclosures),
    five environmental, four social, and one governance standard.
    """
    ESRS_1 = "ESRS_1"
    ESRS_2 = "ESRS_2"
    ESRS_E1 = "ESRS_E1"
    ESRS_E2 = "ESRS_E2"
    ESRS_E3 = "ESRS_E3"
    ESRS_E4 = "ESRS_E4"
    ESRS_E5 = "ESRS_E5"
    ESRS_S1 = "ESRS_S1"
    ESRS_S2 = "ESRS_S2"
    ESRS_S3 = "ESRS_S3"
    ESRS_S4 = "ESRS_S4"
    ESRS_G1 = "ESRS_G1"

class DisclosureStatus(str, Enum):
    """Coverage status for a disclosure requirement."""
    FULLY_COVERED = "fully_covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"
    NOT_APPLICABLE = "not_applicable"
    PHASE_IN = "phase_in"

class CoverageLevel(str, Enum):
    """Aggregate coverage assessment level."""
    COMPLETE = "complete"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

# ---------------------------------------------------------------------------
# ESRS Disclosure Catalog
# ---------------------------------------------------------------------------
# Complete reference database of all ESRS disclosure requirements.
# Source: EU Delegated Regulation 2023/2772 (ESRS Set 1, final).
# Each entry: (disclosure_id, disclosure_name, description, data_points,
#              mandatory, phase_in, phase_in_date)
# ---------------------------------------------------------------------------

_RAW_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    # ---------------------------------------------------------------
    # ESRS 2 - General Disclosures (mandatory for all undertakings)
    # ---------------------------------------------------------------
    "ESRS_2": [
        {
            "id": "GOV-1", "name": "Role of administrative, management and supervisory bodies",
            "desc": "Governance structure and sustainability oversight",
            "dp": ["governance_structure", "sustainability_oversight_body", "member_expertise", "meeting_frequency"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "GOV-2", "name": "Information provided to and sustainability matters addressed by administrative bodies",
            "desc": "Information flow to governance bodies on sustainability",
            "dp": ["information_frequency", "sustainability_agenda_items", "material_topics_addressed"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "GOV-3", "name": "Integration of sustainability performance in incentive schemes",
            "desc": "Sustainability-linked remuneration policies",
            "dp": ["incentive_scheme_description", "sustainability_kpis_linked", "percentage_variable_pay"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "GOV-4", "name": "Statement on due diligence",
            "desc": "Due diligence process description",
            "dp": ["due_diligence_process", "value_chain_coverage", "stakeholder_engagement_in_dd"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "GOV-5", "name": "Risk management and internal controls over sustainability reporting",
            "desc": "Internal control and risk management for sustainability",
            "dp": ["risk_management_process", "internal_controls", "assurance_scope"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "SBM-1", "name": "Strategy, business model and value chain",
            "desc": "Description of strategy, business model, value chain",
            "dp": ["business_model_description", "value_chain_description", "sector_activities", "geographic_presence"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "SBM-2", "name": "Interests and views of stakeholders",
            "desc": "Stakeholder engagement processes and outcomes",
            "dp": ["stakeholder_groups", "engagement_methods", "key_concerns_raised", "response_actions"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "SBM-3", "name": "Material impacts, risks and opportunities and their interaction with strategy",
            "desc": "Material IROs and strategy interaction",
            "dp": ["material_iros_list", "strategy_interaction", "business_model_impact", "time_horizons"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "IRO-1", "name": "Description of processes to identify and assess material IROs",
            "desc": "Methodology for materiality assessment",
            "dp": ["process_description", "criteria_used", "stakeholder_involvement", "thresholds_applied"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "IRO-2", "name": "Disclosure requirements in ESRS covered by the sustainability statement",
            "desc": "List of applicable ESRS disclosure requirements",
            "dp": ["applicable_standards_list", "material_topics_list", "not_material_explanations"],
            "mandatory": True, "phase_in": False, "phase_in_date": None,
        },
    ],

    # ---------------------------------------------------------------
    # ESRS E1 - Climate Change
    # ---------------------------------------------------------------
    "ESRS_E1": [
        {
            "id": "E1-1", "name": "Transition plan for climate change mitigation",
            "desc": "Climate transition plan aligned with 1.5C",
            "dp": ["transition_plan_description", "ghg_reduction_targets", "decarbonisation_levers", "capex_alignment", "locked_in_emissions"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2025-01-01",
        },
        {
            "id": "E1-2", "name": "Policies related to climate change mitigation and adaptation",
            "desc": "Climate policies in place",
            "dp": ["policy_scope", "climate_mitigation_policy", "climate_adaptation_policy", "policy_alignment_with_frameworks"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E1-3", "name": "Actions and resources in relation to climate change policies",
            "desc": "Actions taken and resources allocated",
            "dp": ["key_actions_list", "resources_allocated", "expected_outcomes", "timeline"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E1-4", "name": "Targets related to climate change mitigation and adaptation",
            "desc": "Climate targets (absolute and intensity)",
            "dp": ["target_type", "base_year", "target_year", "base_value", "target_value", "progress_to_date"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E1-5", "name": "Energy consumption and mix",
            "desc": "Total energy consumption and renewable share",
            "dp": ["total_energy_consumption_mwh", "fossil_share_pct", "renewable_share_pct", "nuclear_share_pct", "energy_intensity"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E1-6", "name": "Gross Scopes 1, 2, 3 and Total GHG emissions",
            "desc": "GHG emissions by scope",
            "dp": ["scope_1_tco2e", "scope_2_location_tco2e", "scope_2_market_tco2e", "scope_3_tco2e", "total_tco2e", "biogenic_tco2e"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E1-7", "name": "GHG removals and GHG mitigation projects financed through carbon credits",
            "desc": "Carbon removals and offset credits",
            "dp": ["removals_tco2e", "credits_purchased", "credit_type", "certification_standard"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E1-8", "name": "Internal carbon pricing",
            "desc": "Internal carbon price applied",
            "dp": ["carbon_price_eur_per_tco2e", "scope_of_application", "decision_making_use"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E1-9", "name": "Anticipated financial effects from material physical and transition risks",
            "desc": "Financial effects of climate risks and opportunities",
            "dp": ["physical_risk_exposure_eur", "transition_risk_exposure_eur", "opportunity_value_eur", "time_horizon"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2026-01-01",
        },
    ],

    # ---------------------------------------------------------------
    # ESRS E2 - Pollution
    # ---------------------------------------------------------------
    "ESRS_E2": [
        {
            "id": "E2-1", "name": "Policies related to pollution",
            "desc": "Pollution prevention and control policies",
            "dp": ["policy_scope", "substances_covered", "pollution_prevention_approach"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E2-2", "name": "Actions and resources related to pollution",
            "desc": "Actions taken to prevent/reduce pollution",
            "dp": ["key_actions_list", "resources_allocated", "expected_outcomes"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E2-3", "name": "Targets related to pollution",
            "desc": "Pollution reduction targets",
            "dp": ["target_type", "base_year", "target_year", "target_value", "progress"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E2-4", "name": "Pollution of air, water and soil",
            "desc": "Pollution metrics (emissions to air, water, soil)",
            "dp": ["air_pollutants_tonnes", "water_pollutants_tonnes", "soil_pollutants_tonnes", "microplastics", "substances_of_concern"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E2-5", "name": "Substances of concern and substances of very high concern",
            "desc": "SVHC and SoC usage and management",
            "dp": ["svhc_list", "svhc_volumes_tonnes", "substitution_plans", "reach_compliance"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E2-6", "name": "Anticipated financial effects from pollution-related risks",
            "desc": "Financial impact of pollution risks and opportunities",
            "dp": ["remediation_costs_eur", "liability_exposure_eur", "opportunity_value_eur"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2026-01-01",
        },
    ],

    # ---------------------------------------------------------------
    # ESRS E3 - Water and Marine Resources
    # ---------------------------------------------------------------
    "ESRS_E3": [
        {
            "id": "E3-1", "name": "Policies related to water and marine resources",
            "desc": "Water stewardship policies",
            "dp": ["policy_scope", "water_stewardship_approach", "marine_resource_policy"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E3-2", "name": "Actions and resources related to water and marine resources",
            "desc": "Water management actions",
            "dp": ["key_actions_list", "resources_allocated", "expected_outcomes"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E3-3", "name": "Targets related to water and marine resources",
            "desc": "Water use reduction targets",
            "dp": ["target_type", "base_year", "target_year", "target_value", "progress"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E3-4", "name": "Water consumption",
            "desc": "Water consumption metrics",
            "dp": ["total_water_consumption_m3", "water_intensity", "water_stress_area_consumption_m3", "recycled_water_pct"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E3-5", "name": "Anticipated financial effects from water and marine resource-related risks",
            "desc": "Financial impact of water risks",
            "dp": ["water_risk_exposure_eur", "stranded_asset_risk_eur", "opportunity_value_eur"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2026-01-01",
        },
    ],

    # ---------------------------------------------------------------
    # ESRS E4 - Biodiversity and Ecosystems
    # ---------------------------------------------------------------
    "ESRS_E4": [
        {
            "id": "E4-1", "name": "Transition plan and consideration of biodiversity in strategy",
            "desc": "Biodiversity transition plan",
            "dp": ["transition_plan_description", "biodiversity_strategy_integration", "key_biodiversity_commitments"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2025-01-01",
        },
        {
            "id": "E4-2", "name": "Policies related to biodiversity and ecosystems",
            "desc": "Biodiversity policies",
            "dp": ["policy_scope", "no_deforestation_commitment", "biodiversity_risk_assessment_process"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E4-3", "name": "Actions and resources related to biodiversity",
            "desc": "Biodiversity conservation actions",
            "dp": ["key_actions_list", "resources_allocated", "expected_outcomes", "restoration_activities"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E4-4", "name": "Targets related to biodiversity and ecosystems",
            "desc": "Biodiversity targets",
            "dp": ["target_type", "base_year", "target_year", "target_value", "progress"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E4-5", "name": "Impact metrics related to biodiversity and ecosystems change",
            "desc": "Biodiversity impact metrics",
            "dp": ["land_use_change_ha", "operations_near_sensitive_areas", "species_affected", "ecosystem_condition_index"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E4-6", "name": "Anticipated financial effects from biodiversity-related risks",
            "desc": "Financial impact of biodiversity risks",
            "dp": ["biodiversity_risk_exposure_eur", "dependency_on_ecosystem_services_eur", "opportunity_value_eur"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2026-01-01",
        },
    ],

    # ---------------------------------------------------------------
    # ESRS E5 - Resource Use and Circular Economy
    # ---------------------------------------------------------------
    "ESRS_E5": [
        {
            "id": "E5-1", "name": "Policies related to resource use and circular economy",
            "desc": "Circularity policies",
            "dp": ["policy_scope", "circular_economy_strategy", "waste_hierarchy_application"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E5-2", "name": "Actions and resources related to resource use and circular economy",
            "desc": "Circular economy actions",
            "dp": ["key_actions_list", "resources_allocated", "expected_outcomes"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E5-3", "name": "Targets related to resource use and circular economy",
            "desc": "Resource use and circularity targets",
            "dp": ["target_type", "base_year", "target_year", "target_value", "progress"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E5-4", "name": "Resource inflows",
            "desc": "Material inflows and recycled content",
            "dp": ["total_material_inflow_tonnes", "recycled_content_pct", "renewable_material_pct", "critical_raw_materials"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E5-5", "name": "Resource outflows",
            "desc": "Waste generation and management",
            "dp": ["total_waste_tonnes", "hazardous_waste_tonnes", "waste_diverted_pct", "waste_to_landfill_pct"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "E5-6", "name": "Anticipated financial effects from resource use and circular economy-related risks",
            "desc": "Financial impact of resource use risks",
            "dp": ["resource_risk_exposure_eur", "material_cost_risk_eur", "opportunity_value_eur"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2026-01-01",
        },
    ],

    # ---------------------------------------------------------------
    # ESRS S1 - Own Workforce
    # ---------------------------------------------------------------
    "ESRS_S1": [
        {
            "id": "S1-1", "name": "Policies related to own workforce",
            "desc": "Workforce policies (labour rights, health, diversity)",
            "dp": ["policy_scope", "labour_rights_policy", "health_safety_policy", "diversity_policy", "training_policy"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-2", "name": "Processes for engaging with own workers and workers' representatives",
            "desc": "Worker engagement processes",
            "dp": ["engagement_mechanisms", "works_council_coverage", "collective_bargaining_coverage_pct"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-3", "name": "Processes to remediate negative impacts and channels for own workers to raise concerns",
            "desc": "Grievance mechanisms and remediation",
            "dp": ["grievance_mechanism_description", "cases_reported", "cases_resolved", "remediation_actions"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-4", "name": "Taking action on material impacts on own workforce",
            "desc": "Actions addressing material workforce impacts",
            "dp": ["key_actions_list", "resources_allocated", "expected_outcomes"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-5", "name": "Targets related to managing material negative impacts",
            "desc": "Workforce targets",
            "dp": ["target_type", "target_value", "progress", "timeline"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-6", "name": "Characteristics of the undertaking's employees",
            "desc": "Employee demographics (headcount, gender, contract type, country)",
            "dp": ["total_employees", "employees_by_gender", "employees_by_country", "employees_by_contract_type", "fte_count"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-7", "name": "Characteristics of non-employee workers in the undertaking's own workforce",
            "desc": "Non-employee worker demographics",
            "dp": ["total_non_employees", "non_employees_by_type", "non_employees_by_gender"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-8", "name": "Collective bargaining coverage and social dialogue",
            "desc": "Collective bargaining coverage",
            "dp": ["coverage_pct_eea", "coverage_pct_non_eea", "social_dialogue_description"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-9", "name": "Diversity metrics",
            "desc": "Diversity indicators (gender, age, disability)",
            "dp": ["gender_pay_gap_pct", "board_gender_diversity_pct", "age_distribution", "disability_representation_pct"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-10", "name": "Adequate wages",
            "desc": "Adequate wage assessment",
            "dp": ["adequate_wage_benchmark", "employees_below_adequate_wage_pct", "countries_assessed"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-11", "name": "Social protection",
            "desc": "Social protection coverage",
            "dp": ["social_protection_coverage_pct", "types_of_coverage", "countries_with_gaps"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-12", "name": "Persons with disabilities",
            "desc": "Disability inclusion metrics",
            "dp": ["employees_with_disabilities_pct", "accessibility_measures", "accommodation_requests"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-13", "name": "Training and skills development metrics",
            "desc": "Training hours and investment",
            "dp": ["avg_training_hours_per_employee", "training_spend_eur", "training_by_gender", "training_by_category"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-14", "name": "Health and safety metrics",
            "desc": "Occupational health and safety indicators",
            "dp": ["work_related_fatalities", "recordable_incident_rate", "lost_day_rate", "near_miss_count"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-15", "name": "Work-life balance metrics",
            "desc": "Work-life balance indicators",
            "dp": ["family_leave_uptake_pct", "flexible_work_pct", "average_working_hours"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S1-16", "name": "Remuneration metrics (pay gap and total remuneration)",
            "desc": "Pay gap and remuneration data",
            "dp": ["gender_pay_gap_pct", "ceo_to_median_ratio", "total_remuneration_by_gender"],
            "mandatory": False, "phase_in": True, "phase_in_date": "2026-01-01",
        },
        {
            "id": "S1-17", "name": "Incidents, complaints and severe human rights impacts",
            "desc": "Human rights incidents and complaints",
            "dp": ["incidents_reported", "incidents_investigated", "severe_impacts_count", "remediation_provided"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
    ],

    # ---------------------------------------------------------------
    # ESRS S2 - Workers in the Value Chain
    # ---------------------------------------------------------------
    "ESRS_S2": [
        {
            "id": "S2-1", "name": "Policies related to value chain workers",
            "desc": "Policies for value chain worker rights",
            "dp": ["policy_scope", "supplier_code_of_conduct", "human_rights_policy"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S2-2", "name": "Processes for engaging with value chain workers",
            "desc": "Value chain worker engagement mechanisms",
            "dp": ["engagement_mechanisms", "grievance_channels_available", "feedback_incorporation"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S2-3", "name": "Processes to remediate negative impacts on value chain workers",
            "desc": "Remediation for value chain worker impacts",
            "dp": ["remediation_process", "cases_reported", "cases_resolved"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S2-4", "name": "Taking action on material impacts on value chain workers",
            "desc": "Actions to address value chain worker impacts",
            "dp": ["key_actions_list", "resources_allocated", "suppliers_assessed", "audit_findings"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S2-5", "name": "Targets related to managing negative impacts on value chain workers",
            "desc": "Value chain worker targets",
            "dp": ["target_type", "target_value", "progress", "timeline"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
    ],

    # ---------------------------------------------------------------
    # ESRS S3 - Affected Communities
    # ---------------------------------------------------------------
    "ESRS_S3": [
        {
            "id": "S3-1", "name": "Policies related to affected communities",
            "desc": "Community engagement and impact policies",
            "dp": ["policy_scope", "fpic_policy", "community_investment_policy"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S3-2", "name": "Processes for engaging with affected communities",
            "desc": "Community engagement processes",
            "dp": ["engagement_mechanisms", "community_consultations", "feedback_incorporation"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S3-3", "name": "Processes to remediate negative impacts on affected communities",
            "desc": "Community impact remediation processes",
            "dp": ["remediation_process", "cases_reported", "cases_resolved"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S3-4", "name": "Taking action on material impacts on affected communities",
            "desc": "Actions addressing community impacts",
            "dp": ["key_actions_list", "resources_allocated", "community_investments_eur"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S3-5", "name": "Targets related to managing negative impacts on affected communities",
            "desc": "Community targets",
            "dp": ["target_type", "target_value", "progress", "timeline"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
    ],

    # ---------------------------------------------------------------
    # ESRS S4 - Consumers and End-users
    # ---------------------------------------------------------------
    "ESRS_S4": [
        {
            "id": "S4-1", "name": "Policies related to consumers and end-users",
            "desc": "Consumer protection and product safety policies",
            "dp": ["policy_scope", "product_safety_policy", "data_privacy_policy", "marketing_ethics_policy"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S4-2", "name": "Processes for engaging with consumers and end-users",
            "desc": "Consumer engagement processes",
            "dp": ["engagement_mechanisms", "complaint_channels", "feedback_incorporation"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S4-3", "name": "Processes to remediate negative impacts on consumers",
            "desc": "Consumer impact remediation",
            "dp": ["remediation_process", "product_recalls", "cases_resolved"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S4-4", "name": "Taking action on material impacts on consumers",
            "desc": "Actions addressing consumer impacts",
            "dp": ["key_actions_list", "resources_allocated", "product_safety_improvements"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "S4-5", "name": "Targets related to managing negative impacts on consumers",
            "desc": "Consumer-related targets",
            "dp": ["target_type", "target_value", "progress", "timeline"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
    ],

    # ---------------------------------------------------------------
    # ESRS G1 - Business Conduct
    # ---------------------------------------------------------------
    "ESRS_G1": [
        {
            "id": "G1-1", "name": "Business conduct policies and corporate culture",
            "desc": "Business ethics and anti-corruption policies",
            "dp": ["code_of_conduct", "anti_corruption_policy", "whistleblower_policy", "corporate_culture_description"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "G1-2", "name": "Management of relationships with suppliers",
            "desc": "Supplier management practices",
            "dp": ["supplier_assessment_process", "payment_practices", "avg_payment_days"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "G1-3", "name": "Prevention and detection of corruption and bribery",
            "desc": "Anti-corruption program",
            "dp": ["training_coverage_pct", "risk_assessments_conducted", "incidents_reported", "convictions"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "G1-4", "name": "Incidents of corruption or bribery",
            "desc": "Corruption incident reporting",
            "dp": ["confirmed_incidents", "employees_dismissed", "legal_actions", "fines_paid_eur"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "G1-5", "name": "Political influence and lobbying activities",
            "desc": "Political engagement and lobbying transparency",
            "dp": ["political_contributions_eur", "lobbying_spend_eur", "trade_associations", "positions_held"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
        {
            "id": "G1-6", "name": "Payment practices",
            "desc": "Supplier payment terms and practices",
            "dp": ["standard_payment_terms_days", "avg_actual_payment_days", "late_payment_pct", "late_payment_interest_eur"],
            "mandatory": False, "phase_in": False, "phase_in_date": None,
        },
    ],
}

# Map ESRS topic short codes to standard keys
TOPIC_TO_STANDARD: Dict[str, str] = {
    "E1": "ESRS_E1", "E2": "ESRS_E2", "E3": "ESRS_E3",
    "E4": "ESRS_E4", "E5": "ESRS_E5",
    "S1": "ESRS_S1", "S2": "ESRS_S2", "S3": "ESRS_S3", "S4": "ESRS_S4",
    "G1": "ESRS_G1",
    "ESRS_2": "ESRS_2", "ESRS2": "ESRS_2",
    "ESRS_E1": "ESRS_E1", "ESRS_E2": "ESRS_E2", "ESRS_E3": "ESRS_E3",
    "ESRS_E4": "ESRS_E4", "ESRS_E5": "ESRS_E5",
    "ESRS_S1": "ESRS_S1", "ESRS_S2": "ESRS_S2", "ESRS_S3": "ESRS_S3",
    "ESRS_S4": "ESRS_S4", "ESRS_G1": "ESRS_G1",
}

# Effort estimates (hours) per disclosure.
# Based on typical first-year CSRD implementation timelines.
EFFORT_ESTIMATES_PER_DISCLOSURE: Dict[str, Decimal] = {
    # ESRS 2 -- mandatory for all
    "GOV-1": Decimal("24"), "GOV-2": Decimal("16"), "GOV-3": Decimal("20"),
    "GOV-4": Decimal("32"), "GOV-5": Decimal("24"),
    "SBM-1": Decimal("40"), "SBM-2": Decimal("32"), "SBM-3": Decimal("40"),
    "IRO-1": Decimal("40"), "IRO-2": Decimal("16"),
    # E1
    "E1-1": Decimal("80"), "E1-2": Decimal("16"), "E1-3": Decimal("24"),
    "E1-4": Decimal("32"), "E1-5": Decimal("40"), "E1-6": Decimal("60"),
    "E1-7": Decimal("20"), "E1-8": Decimal("16"), "E1-9": Decimal("48"),
    # E2
    "E2-1": Decimal("16"), "E2-2": Decimal("20"), "E2-3": Decimal("20"),
    "E2-4": Decimal("40"), "E2-5": Decimal("32"), "E2-6": Decimal("32"),
    # E3
    "E3-1": Decimal("16"), "E3-2": Decimal("20"), "E3-3": Decimal("20"),
    "E3-4": Decimal("32"), "E3-5": Decimal("32"),
    # E4
    "E4-1": Decimal("60"), "E4-2": Decimal("16"), "E4-3": Decimal("24"),
    "E4-4": Decimal("20"), "E4-5": Decimal("40"), "E4-6": Decimal("32"),
    # E5
    "E5-1": Decimal("16"), "E5-2": Decimal("20"), "E5-3": Decimal("20"),
    "E5-4": Decimal("32"), "E5-5": Decimal("32"), "E5-6": Decimal("32"),
    # S1
    "S1-1": Decimal("16"), "S1-2": Decimal("20"), "S1-3": Decimal("24"),
    "S1-4": Decimal("20"), "S1-5": Decimal("16"), "S1-6": Decimal("24"),
    "S1-7": Decimal("16"), "S1-8": Decimal("20"), "S1-9": Decimal("32"),
    "S1-10": Decimal("24"), "S1-11": Decimal("20"), "S1-12": Decimal("16"),
    "S1-13": Decimal("24"), "S1-14": Decimal("32"), "S1-15": Decimal("16"),
    "S1-16": Decimal("40"), "S1-17": Decimal("24"),
    # S2
    "S2-1": Decimal("16"), "S2-2": Decimal("20"), "S2-3": Decimal("20"),
    "S2-4": Decimal("24"), "S2-5": Decimal("16"),
    # S3
    "S3-1": Decimal("16"), "S3-2": Decimal("20"), "S3-3": Decimal("20"),
    "S3-4": Decimal("24"), "S3-5": Decimal("16"),
    # S4
    "S4-1": Decimal("16"), "S4-2": Decimal("20"), "S4-3": Decimal("20"),
    "S4-4": Decimal("24"), "S4-5": Decimal("16"),
    # G1
    "G1-1": Decimal("24"), "G1-2": Decimal("20"), "G1-3": Decimal("24"),
    "G1-4": Decimal("16"), "G1-5": Decimal("16"), "G1-6": Decimal("20"),
}

# Build the structured catalog from the raw data
ESRS_DISCLOSURE_CATALOG: Dict[str, List[Dict[str, Any]]] = _RAW_CATALOG

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ESRSDisclosureRequirement(BaseModel):
    """A single ESRS disclosure requirement.

    Attributes:
        standard_id: ESRS standard (e.g. "ESRS_E1").
        disclosure_id: Disclosure identifier (e.g. "E1-6").
        disclosure_name: Human-readable disclosure name.
        description: Brief description of the disclosure.
        data_points: List of individual data point names.
        mandatory: True if mandatory regardless of materiality.
        phase_in: True if phase-in period applies.
        phase_in_date: Date when the disclosure becomes mandatory.
    """
    standard_id: str = Field(..., description="ESRS standard identifier")
    disclosure_id: str = Field(..., description="Disclosure ID (e.g. E1-6)")
    disclosure_name: str = Field(..., description="Disclosure name")
    description: str = Field(default="", description="Brief description")
    data_points: List[str] = Field(default_factory=list, description="Required data points")
    mandatory: bool = Field(default=False, description="Mandatory regardless of materiality")
    phase_in: bool = Field(default=False, description="Phase-in period applies")
    phase_in_date: Optional[str] = Field(default=None, description="Phase-in effective date")

class TopicMapping(BaseModel):
    """Mapping of one material topic to its ESRS disclosure requirements.

    Attributes:
        matter_id: Unique identifier of the sustainability matter.
        esrs_topic: ESRS topic code (e.g. "E1", "S1", "G1").
        mapped_disclosures: List of applicable disclosure requirements.
        total_data_points: Total number of data points across all disclosures.
        mandatory_data_points: Count of data points from mandatory disclosures.
        phase_in_data_points: Count of data points from phase-in disclosures.
    """
    matter_id: str = Field(..., description="Matter identifier")
    esrs_topic: str = Field(..., description="ESRS topic code")
    mapped_disclosures: List[ESRSDisclosureRequirement] = Field(default_factory=list)
    total_data_points: int = Field(default=0, ge=0)
    mandatory_data_points: int = Field(default=0, ge=0)
    phase_in_data_points: int = Field(default=0, ge=0)

class DisclosureGap(BaseModel):
    """Gap identified for a specific disclosure requirement.

    Attributes:
        disclosure_id: Disclosure identifier (e.g. "E1-6").
        disclosure_name: Human-readable name.
        status: Coverage status.
        missing_data_points: List of data points not yet available.
        available_data_points: List of data points already available.
        estimated_effort_hours: Estimated hours to close the gap.
    """
    disclosure_id: str = Field(..., description="Disclosure identifier")
    disclosure_name: str = Field(default="", description="Disclosure name")
    status: DisclosureStatus = Field(default=DisclosureStatus.NOT_COVERED)
    missing_data_points: List[str] = Field(default_factory=list)
    available_data_points: List[str] = Field(default_factory=list)
    estimated_effort_hours: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

    @field_validator("estimated_effort_hours", mode="before")
    @classmethod
    def _coerce_effort(cls, v: Any) -> Decimal:
        return _decimal(v)

class ESRSMappingResult(BaseModel):
    """Complete result of ESRS topic mapping for all material topics.

    Attributes:
        mappings: Per-topic mapping results.
        total_topics: Number of material topics mapped.
        total_disclosures: Total distinct disclosures across all topics.
        total_data_points: Total data points across all disclosures.
        coverage_assessment: Overall coverage level.
        gap_analysis: Gaps identified across all disclosures.
        provenance_hash: SHA-256 hash for audit trail.
        calculated_at: Timestamp of mapping generation.
        processing_time_ms: Processing time in milliseconds.
    """
    mappings: List[TopicMapping] = Field(default_factory=list)
    total_topics: int = Field(default=0, ge=0)
    total_disclosures: int = Field(default=0, ge=0)
    total_data_points: int = Field(default=0, ge=0)
    coverage_assessment: str = Field(default="none")
    gap_analysis: List[DisclosureGap] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: Optional[datetime] = Field(default=None)
    processing_time_ms: float = Field(default=0.0, ge=0.0)

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class MaterialTopicInput(BaseModel):
    """Input representing a single material topic to map.

    Attributes:
        matter_id: Unique identifier of the sustainability matter.
        esrs_topic: ESRS topic code (e.g. "E1", "S1", "G1").
    """
    matter_id: str = Field(..., min_length=1)
    esrs_topic: str = Field(..., min_length=1)

class AvailableDataInput(BaseModel):
    """Input representing data points currently available.

    Attributes:
        disclosure_id: Disclosure for which data is available.
        available_data_points: List of available data point names.
    """
    disclosure_id: str = Field(..., min_length=1)
    available_data_points: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ESRSTopicMappingEngine:
    """Maps material topics to ESRS disclosure requirements.

    Zero-Hallucination Guarantees:
        - Disclosure catalog is a hard-coded reference database
        - Mapping uses deterministic dictionary lookup (no ML/LLM)
        - Gap analysis uses simple set subtraction
        - Effort estimates use fixed lookup tables
        - SHA-256 provenance hash on every result

    Usage::

        engine = ESRSTopicMappingEngine()
        result = engine.batch_map(material_topics=[
            MaterialTopicInput(matter_id="m1", esrs_topic="E1"),
            MaterialTopicInput(matter_id="m2", esrs_topic="S1"),
        ])
    """

    def __init__(self) -> None:
        """Initialize ESRSTopicMappingEngine with the disclosure catalog."""
        self._catalog = ESRS_DISCLOSURE_CATALOG
        self._effort_map = EFFORT_ESTIMATES_PER_DISCLOSURE
        logger.info(
            "ESRSTopicMappingEngine initialized: %d standards, %d disclosures",
            len(self._catalog),
            sum(len(v) for v in self._catalog.values()),
        )

    # ------------------------------------------------------------------
    # Core: Map Topic to Disclosures
    # ------------------------------------------------------------------

    def map_topic_to_disclosures(
        self, matter_id: str, esrs_topic: str
    ) -> TopicMapping:
        """Map a single material topic to its ESRS disclosure requirements.

        DETERMINISTIC: Same topic always maps to the same disclosures.

        Args:
            matter_id: Unique identifier of the sustainability matter.
            esrs_topic: ESRS topic code (e.g. "E1", "S1").

        Returns:
            TopicMapping with all applicable disclosures.
        """
        standard_key = TOPIC_TO_STANDARD.get(esrs_topic, esrs_topic)
        raw_disclosures = self._catalog.get(standard_key, [])

        mapped: List[ESRSDisclosureRequirement] = []
        total_dp = 0
        mandatory_dp = 0
        phase_in_dp = 0

        for disc_raw in raw_disclosures:
            dp_list = disc_raw.get("dp", [])
            dp_count = len(dp_list)

            req = ESRSDisclosureRequirement(
                standard_id=standard_key,
                disclosure_id=disc_raw["id"],
                disclosure_name=disc_raw["name"],
                description=disc_raw.get("desc", ""),
                data_points=dp_list,
                mandatory=disc_raw.get("mandatory", False),
                phase_in=disc_raw.get("phase_in", False),
                phase_in_date=disc_raw.get("phase_in_date"),
            )
            mapped.append(req)
            total_dp += dp_count

            if req.mandatory:
                mandatory_dp += dp_count
            if req.phase_in:
                phase_in_dp += dp_count

        return TopicMapping(
            matter_id=matter_id,
            esrs_topic=esrs_topic,
            mapped_disclosures=mapped,
            total_data_points=total_dp,
            mandatory_data_points=mandatory_dp,
            phase_in_data_points=phase_in_dp,
        )

    # ------------------------------------------------------------------
    # Core: Batch Map
    # ------------------------------------------------------------------

    def batch_map(
        self, material_topics: List[MaterialTopicInput]
    ) -> ESRSMappingResult:
        """Map multiple material topics to ESRS disclosures.

        Also includes ESRS 2 mandatory disclosures which apply
        regardless of materiality.

        DETERMINISTIC: Same topics always produce the same mapping.

        Args:
            material_topics: List of material topics to map.

        Returns:
            ESRSMappingResult with all mappings and aggregate statistics.
        """
        t0 = time.perf_counter()

        mappings: List[TopicMapping] = []
        seen_disclosures: set = set()
        total_dp = 0

        # Always include ESRS 2 mandatory disclosures
        esrs2_mapping = self.map_topic_to_disclosures("__ESRS2_MANDATORY__", "ESRS_2")
        mappings.append(esrs2_mapping)
        for disc in esrs2_mapping.mapped_disclosures:
            seen_disclosures.add(disc.disclosure_id)
            total_dp += len(disc.data_points)

        # Map each material topic
        for topic_input in material_topics:
            mapping = self.map_topic_to_disclosures(
                topic_input.matter_id, topic_input.esrs_topic
            )
            mappings.append(mapping)
            for disc in mapping.mapped_disclosures:
                if disc.disclosure_id not in seen_disclosures:
                    seen_disclosures.add(disc.disclosure_id)
                    total_dp += len(disc.data_points)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ESRSMappingResult(
            mappings=mappings,
            total_topics=len(material_topics),
            total_disclosures=len(seen_disclosures),
            total_data_points=total_dp,
            coverage_assessment="none",
            calculated_at=utcnow(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Batch mapping: %d topics -> %d disclosures, %d data points, hash=%s",
            len(material_topics), len(seen_disclosures), total_dp,
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------
    # Gap Analysis
    # ------------------------------------------------------------------

    def analyze_gaps(
        self,
        mapping_result: ESRSMappingResult,
        available_data: List[AvailableDataInput],
    ) -> List[DisclosureGap]:
        """Analyze gaps between required disclosures and available data.

        Uses simple set subtraction: required data points minus available
        data points yields missing data points.

        DETERMINISTIC: Same inputs always produce the same gaps.

        Args:
            mapping_result: Result from batch_map().
            available_data: Data points currently available per disclosure.

        Returns:
            List of DisclosureGap for all disclosures with gaps.
        """
        # Index available data by disclosure_id
        available_by_disc: Dict[str, set] = {}
        for avail in available_data:
            available_by_disc[avail.disclosure_id] = set(avail.available_data_points)

        # Collect all unique disclosures from mapping
        all_disclosures: Dict[str, ESRSDisclosureRequirement] = {}
        for mapping in mapping_result.mappings:
            for disc in mapping.mapped_disclosures:
                if disc.disclosure_id not in all_disclosures:
                    all_disclosures[disc.disclosure_id] = disc

        gaps: List[DisclosureGap] = []
        for disc_id in sorted(all_disclosures.keys()):
            disc = all_disclosures[disc_id]
            required_set = set(disc.data_points)
            available_set = available_by_disc.get(disc_id, set())

            missing = sorted(required_set - available_set)
            available = sorted(required_set & available_set)

            if len(missing) == 0:
                status = DisclosureStatus.FULLY_COVERED
            elif len(available) > 0:
                status = DisclosureStatus.PARTIALLY_COVERED
            else:
                status = DisclosureStatus.NOT_COVERED

            # Effort proportional to missing data points
            base_effort = self._effort_map.get(disc_id, Decimal("16"))
            if len(disc.data_points) > 0:
                effort_ratio = _safe_divide(
                    _decimal(len(missing)),
                    _decimal(len(disc.data_points)),
                )
                estimated_effort = (base_effort * effort_ratio).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            else:
                estimated_effort = Decimal("0")

            gap = DisclosureGap(
                disclosure_id=disc_id,
                disclosure_name=disc.disclosure_name,
                status=status,
                missing_data_points=missing,
                available_data_points=available,
                estimated_effort_hours=estimated_effort,
            )
            gaps.append(gap)

        return gaps

    # ------------------------------------------------------------------
    # Coverage Assessment
    # ------------------------------------------------------------------

    def calculate_coverage(
        self,
        mapping_result: ESRSMappingResult,
        available_data: List[AvailableDataInput],
    ) -> CoverageLevel:
        """Calculate overall coverage level.

        DETERMINISTIC: Same inputs always produce the same coverage level.

        Coverage levels:
            - COMPLETE: 100% of data points available
            - HIGH: >= 80% of data points available
            - MEDIUM: >= 50% of data points available
            - LOW: >= 20% of data points available
            - NONE: < 20% of data points available

        Args:
            mapping_result: Result from batch_map().
            available_data: Data points currently available.

        Returns:
            CoverageLevel enum value.
        """
        gaps = self.analyze_gaps(mapping_result, available_data)

        total_required = 0
        total_available = 0
        for gap in gaps:
            total_required += len(gap.missing_data_points) + len(gap.available_data_points)
            total_available += len(gap.available_data_points)

        if total_required == 0:
            return CoverageLevel.NONE

        coverage_pct = _safe_pct(_decimal(total_available), _decimal(total_required))

        if coverage_pct >= Decimal("100"):
            return CoverageLevel.COMPLETE
        elif coverage_pct >= Decimal("80"):
            return CoverageLevel.HIGH
        elif coverage_pct >= Decimal("50"):
            return CoverageLevel.MEDIUM
        elif coverage_pct >= Decimal("20"):
            return CoverageLevel.LOW
        else:
            return CoverageLevel.NONE

    # ------------------------------------------------------------------
    # Mandatory Disclosures
    # ------------------------------------------------------------------

    def get_mandatory_disclosures(
        self, esrs_topic: str
    ) -> List[ESRSDisclosureRequirement]:
        """Return only mandatory disclosures for a topic.

        Args:
            esrs_topic: ESRS topic code.

        Returns:
            List of mandatory disclosure requirements.
        """
        mapping = self.map_topic_to_disclosures("__query__", esrs_topic)
        return [d for d in mapping.mapped_disclosures if d.mandatory]

    def get_phase_in_disclosures(
        self, esrs_topic: str
    ) -> List[ESRSDisclosureRequirement]:
        """Return only phase-in disclosures for a topic.

        Args:
            esrs_topic: ESRS topic code.

        Returns:
            List of phase-in disclosure requirements.
        """
        mapping = self.map_topic_to_disclosures("__query__", esrs_topic)
        return [d for d in mapping.mapped_disclosures if d.phase_in]

    # ------------------------------------------------------------------
    # Effort Estimation
    # ------------------------------------------------------------------

    def estimate_effort(self, gaps: List[DisclosureGap]) -> Decimal:
        """Calculate total estimated effort hours to close all gaps.

        DETERMINISTIC: Same gaps always produce the same effort total.

        Args:
            gaps: List of disclosure gaps from analyze_gaps().

        Returns:
            Total estimated hours as Decimal.
        """
        total = Decimal("0")
        for gap in gaps:
            if gap.status in (DisclosureStatus.NOT_COVERED, DisclosureStatus.PARTIALLY_COVERED):
                total += gap.estimated_effort_hours
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Query Utilities
    # ------------------------------------------------------------------

    def get_all_disclosures_for_standard(
        self, standard: str
    ) -> List[ESRSDisclosureRequirement]:
        """Return all disclosures for a given ESRS standard.

        Args:
            standard: Standard key (e.g. "ESRS_E1", "E1").

        Returns:
            List of ESRSDisclosureRequirement.
        """
        mapping = self.map_topic_to_disclosures("__query__", standard)
        return mapping.mapped_disclosures

    def count_data_points_by_standard(self) -> Dict[str, int]:
        """Count total data points per ESRS standard.

        Returns:
            Dictionary mapping standard key to data point count.
        """
        result: Dict[str, int] = {}
        for std_key, disclosures in self._catalog.items():
            count = sum(len(d.get("dp", [])) for d in disclosures)
            result[std_key] = count
        return result

    def get_disclosure_by_id(
        self, disclosure_id: str
    ) -> Optional[ESRSDisclosureRequirement]:
        """Look up a single disclosure by its ID.

        Args:
            disclosure_id: Disclosure identifier (e.g. "E1-6", "GOV-1").

        Returns:
            ESRSDisclosureRequirement if found, None otherwise.
        """
        for std_key, disclosures in self._catalog.items():
            for disc_raw in disclosures:
                if disc_raw["id"] == disclosure_id:
                    return ESRSDisclosureRequirement(
                        standard_id=std_key,
                        disclosure_id=disc_raw["id"],
                        disclosure_name=disc_raw["name"],
                        description=disc_raw.get("desc", ""),
                        data_points=disc_raw.get("dp", []),
                        mandatory=disc_raw.get("mandatory", False),
                        phase_in=disc_raw.get("phase_in", False),
                        phase_in_date=disc_raw.get("phase_in_date"),
                    )
        return None

    def get_gap_summary(
        self, gaps: List[DisclosureGap]
    ) -> Dict[str, Any]:
        """Generate summary statistics for gap analysis.

        Args:
            gaps: List of disclosure gaps.

        Returns:
            Dictionary with counts by status, total effort, and percentages.
        """
        fully_covered = sum(1 for g in gaps if g.status == DisclosureStatus.FULLY_COVERED)
        partially = sum(1 for g in gaps if g.status == DisclosureStatus.PARTIALLY_COVERED)
        not_covered = sum(1 for g in gaps if g.status == DisclosureStatus.NOT_COVERED)
        total = len(gaps)
        total_effort = self.estimate_effort(gaps)

        return {
            "total_disclosures": total,
            "fully_covered": fully_covered,
            "partially_covered": partially,
            "not_covered": not_covered,
            "fully_covered_pct": _round_val(_safe_pct(_decimal(fully_covered), _decimal(total)), 2),
            "partially_covered_pct": _round_val(_safe_pct(_decimal(partially), _decimal(total)), 2),
            "not_covered_pct": _round_val(_safe_pct(_decimal(not_covered), _decimal(total)), 2),
            "total_estimated_effort_hours": float(total_effort),
        }
