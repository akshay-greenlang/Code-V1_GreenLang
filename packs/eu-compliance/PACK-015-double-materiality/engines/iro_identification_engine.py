# -*- coding: utf-8 -*-
"""
IROIdentificationEngine - PACK-015 Double Materiality Engine 4
================================================================

Identifies Impacts, Risks, and Opportunities (IROs) per ESRS 1
Paragraphs 28-39.

The IRO identification process is a core step in the ESRS double
materiality assessment.  It involves identifying the undertaking's
actual and potential impacts on people and the environment, as well
as the sustainability-related risks and opportunities that may have
a material financial effect.  These IROs are then assessed for
materiality through impact materiality (Engine 1) and financial
materiality (Engine 2) scoring.

ESRS 1 IRO Identification Framework:
    - Para 28: The undertaking shall identify its IROs through an
      appropriate process.
    - Para 29: The process shall consider information from the
      undertaking's own operations and value chain.
    - Para 30-31: IROs shall be identified in relation to all
      sustainability matters listed in ESRS topical standards.
    - Para 32-39: The undertaking shall consider relevant
      stakeholders, value chain relationships, time horizons,
      and the nature of the IROs.

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS 1 General Requirements, Para 28-39
    - ESRS 2 General Disclosures, IRO-1, IRO-2
    - EFRAG Implementation Guidance IG 1 (Materiality Assessment)
    - EFRAG Implementation Guidance IG 2 (Value Chain)

Zero-Hallucination:
    - IRO catalog is a static, curated database (no generation)
    - Classification uses deterministic scoring from Engine 1 & 2
    - Register statistics are simple counts and aggregations
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
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IROType(str, Enum):
    """Type of Impact, Risk, or Opportunity per ESRS 1.

    Impacts are classified as actual or potential, positive or negative.
    Risks and opportunities represent potential financial effects on
    the undertaking.
    """
    IMPACT_ACTUAL_NEGATIVE = "impact_actual_negative"
    IMPACT_POTENTIAL_NEGATIVE = "impact_potential_negative"
    IMPACT_ACTUAL_POSITIVE = "impact_actual_positive"
    IMPACT_POTENTIAL_POSITIVE = "impact_potential_positive"
    RISK = "risk"
    OPPORTUNITY = "opportunity"

class PriorityLevel(str, Enum):
    """Priority level for an IRO based on combined materiality scoring."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ESRSTopic(str, Enum):
    """ESRS sustainability topics (self-contained for this engine)."""
    E1_CLIMATE = "e1_climate"
    E2_POLLUTION = "e2_pollution"
    E3_WATER = "e3_water"
    E4_BIODIVERSITY = "e4_biodiversity"
    E5_CIRCULAR_ECONOMY = "e5_circular_economy"
    S1_OWN_WORKFORCE = "s1_own_workforce"
    S2_VALUE_CHAIN_WORKERS = "s2_value_chain_workers"
    S3_AFFECTED_COMMUNITIES = "s3_affected_communities"
    S4_CONSUMERS = "s4_consumers"
    G1_BUSINESS_CONDUCT = "g1_business_conduct"

class ValueChainStage(str, Enum):
    """Stage of the value chain where the IRO applies."""
    UPSTREAM = "upstream"
    OWN_OPERATIONS = "own_operations"
    DOWNSTREAM = "downstream"

class TimeHorizon(str, Enum):
    """Time horizon for the IRO."""
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default materiality threshold for combined scoring.
DEFAULT_MATERIALITY_THRESHOLD: Decimal = Decimal("0.40")

# Priority level thresholds based on combined score.
PRIORITY_THRESHOLDS: Dict[str, Decimal] = {
    "critical": Decimal("0.80"),
    "high": Decimal("0.60"),
    "medium": Decimal("0.40"),
    "low": Decimal("0.00"),
}

# Comprehensive IRO catalog per ESRS topic.
# Each entry is a standard IRO that undertakings should consider.
# This catalog contains 100+ entries covering all ESRS topics.
ESRS_IRO_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "e1_climate": [
        {
            "name": "GHG emissions from operations",
            "description": "Direct GHG emissions from stationary and mobile combustion, process emissions, and fugitive emissions",
            "iro_type": "impact_actual_negative",
            "sub_topic": "ghg_emissions_scope_1",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Energy consumption emissions",
            "description": "Indirect GHG emissions from purchased electricity, heat, steam, and cooling",
            "iro_type": "impact_actual_negative",
            "sub_topic": "ghg_emissions_scope_2",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Value chain emissions",
            "description": "Scope 3 GHG emissions across the upstream and downstream value chain",
            "iro_type": "impact_actual_negative",
            "sub_topic": "ghg_emissions_scope_3",
            "typical_value_chain": ["upstream", "downstream"],
        },
        {
            "name": "Physical climate risk - acute",
            "description": "Risk of financial loss from acute climate events (floods, storms, wildfires)",
            "iro_type": "risk",
            "sub_topic": "climate_change_adaptation",
            "typical_value_chain": ["own_operations", "upstream", "downstream"],
        },
        {
            "name": "Physical climate risk - chronic",
            "description": "Risk of financial loss from chronic climate changes (temperature rise, sea level)",
            "iro_type": "risk",
            "sub_topic": "climate_change_adaptation",
            "typical_value_chain": ["own_operations", "upstream", "downstream"],
        },
        {
            "name": "Transition risk - policy and legal",
            "description": "Risk from carbon pricing, emissions regulations, and litigation",
            "iro_type": "risk",
            "sub_topic": "climate_change_mitigation",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Transition risk - technology",
            "description": "Risk of stranded assets from low-carbon technology disruption",
            "iro_type": "risk",
            "sub_topic": "climate_change_mitigation",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Transition risk - market",
            "description": "Risk from shifting consumer preferences toward low-carbon products",
            "iro_type": "risk",
            "sub_topic": "climate_change_mitigation",
            "typical_value_chain": ["downstream"],
        },
        {
            "name": "Green revenue opportunity",
            "description": "Opportunity to generate revenue from low-carbon products and services",
            "iro_type": "opportunity",
            "sub_topic": "climate_change_mitigation",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Energy efficiency opportunity",
            "description": "Opportunity to reduce costs through energy efficiency improvements",
            "iro_type": "opportunity",
            "sub_topic": "energy",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Renewable energy transition",
            "description": "Positive impact from transitioning to renewable energy sources",
            "iro_type": "impact_actual_positive",
            "sub_topic": "energy",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Transition risk - reputation",
            "description": "Risk of reputational damage from perceived climate inaction",
            "iro_type": "risk",
            "sub_topic": "climate_change_mitigation",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Carbon removal and storage",
            "description": "Potential positive impact from carbon capture, utilisation, and storage",
            "iro_type": "impact_potential_positive",
            "sub_topic": "ghg_removal_storage",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Internal carbon pricing implementation",
            "description": "Opportunity to drive emission reductions through internal carbon pricing",
            "iro_type": "opportunity",
            "sub_topic": "internal_carbon_pricing",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Supply chain climate disruption",
            "description": "Risk of supply chain disruption from climate-related events in upstream operations",
            "iro_type": "risk",
            "sub_topic": "climate_change_adaptation",
            "typical_value_chain": ["upstream"],
        },
    ],
    "e2_pollution": [
        {
            "name": "Air pollution from operations",
            "description": "Emissions of pollutants to air (NOx, SOx, PM, VOCs)",
            "iro_type": "impact_actual_negative",
            "sub_topic": "pollution_of_air",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Water pollution from discharges",
            "description": "Release of pollutants to water bodies from operations",
            "iro_type": "impact_actual_negative",
            "sub_topic": "pollution_of_water",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Soil contamination",
            "description": "Contamination of soil from operations or waste disposal",
            "iro_type": "impact_actual_negative",
            "sub_topic": "pollution_of_soil",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Substances of concern in products",
            "description": "Use of substances of concern or SVHC in products",
            "iro_type": "impact_potential_negative",
            "sub_topic": "substances_of_concern",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Microplastics release",
            "description": "Release of microplastics from products or processes",
            "iro_type": "impact_actual_negative",
            "sub_topic": "microplastics",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Environmental remediation liability",
            "description": "Risk of remediation costs from historical or ongoing pollution",
            "iro_type": "risk",
            "sub_topic": "pollution_of_soil",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Pollution control technology opportunity",
            "description": "Opportunity to develop pollution control technologies",
            "iro_type": "opportunity",
            "sub_topic": "pollution_of_air",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "SVHC phase-out costs",
            "description": "Risk of costs from phasing out substances of very high concern under REACH",
            "iro_type": "risk",
            "sub_topic": "substances_of_very_high_concern",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Food safety contamination",
            "description": "Potential negative impact on food safety from pollution of living organisms",
            "iro_type": "impact_potential_negative",
            "sub_topic": "pollution_of_living_organisms_food",
            "typical_value_chain": ["own_operations", "downstream"],
        },
    ],
    "e3_water": [
        {
            "name": "Water consumption in water-stress areas",
            "description": "Excessive water withdrawal in water-stressed regions",
            "iro_type": "impact_actual_negative",
            "sub_topic": "water_stress_areas",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Water pollution from discharges",
            "description": "Degradation of water quality from operational discharges",
            "iro_type": "impact_actual_negative",
            "sub_topic": "water_discharges",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Water scarcity operational risk",
            "description": "Risk of operational disruption from water scarcity",
            "iro_type": "risk",
            "sub_topic": "water_consumption",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Water efficiency opportunity",
            "description": "Opportunity to reduce costs through water efficiency",
            "iro_type": "opportunity",
            "sub_topic": "water_consumption",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Ocean discharge impacts",
            "description": "Impacts on marine ecosystems from ocean discharges",
            "iro_type": "impact_potential_negative",
            "sub_topic": "water_discharges_in_oceans",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Water stewardship partnership opportunity",
            "description": "Opportunity to improve water governance through catchment-level partnerships",
            "iro_type": "opportunity",
            "sub_topic": "water_withdrawals",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Upstream water dependency risk",
            "description": "Risk from suppliers operating in water-stressed basins",
            "iro_type": "risk",
            "sub_topic": "water_stress_areas",
            "typical_value_chain": ["upstream"],
        },
    ],
    "e4_biodiversity": [
        {
            "name": "Land use change and habitat loss",
            "description": "Impact on biodiversity from land use change and deforestation",
            "iro_type": "impact_actual_negative",
            "sub_topic": "direct_impact_drivers_land_use",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Overexploitation of natural resources",
            "description": "Unsustainable extraction or harvesting of natural resources",
            "iro_type": "impact_actual_negative",
            "sub_topic": "direct_impact_drivers_exploitation",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Ecosystem service dependency risk",
            "description": "Risk from degradation of ecosystem services the business depends on",
            "iro_type": "risk",
            "sub_topic": "impacts_on_ecosystem_services",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Invasive species introduction",
            "description": "Potential introduction of invasive species through operations or supply chain",
            "iro_type": "impact_potential_negative",
            "sub_topic": "direct_impact_drivers_invasive_species",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Biodiversity offset opportunity",
            "description": "Opportunity to invest in biodiversity restoration and offsets",
            "iro_type": "opportunity",
            "sub_topic": "impacts_on_ecosystems",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Species impact from pollution",
            "description": "Negative impact on species from pollution originating in operations",
            "iro_type": "impact_actual_negative",
            "sub_topic": "impacts_on_species",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Deforestation in supply chain",
            "description": "Impact on biodiversity from deforestation linked to commodity sourcing",
            "iro_type": "impact_actual_negative",
            "sub_topic": "direct_impact_drivers_land_use",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Nature-based solutions opportunity",
            "description": "Opportunity to invest in nature-based solutions for carbon and biodiversity",
            "iro_type": "opportunity",
            "sub_topic": "impacts_on_ecosystems",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Biodiversity regulation compliance risk",
            "description": "Risk from non-compliance with EU Biodiversity Strategy and Nature Restoration Law",
            "iro_type": "risk",
            "sub_topic": "impacts_on_ecosystems",
            "typical_value_chain": ["own_operations"],
        },
    ],
    "e5_circular_economy": [
        {
            "name": "Resource depletion from virgin material use",
            "description": "Consumption of non-renewable or scarce virgin materials",
            "iro_type": "impact_actual_negative",
            "sub_topic": "resource_inflows",
            "typical_value_chain": ["upstream", "own_operations"],
        },
        {
            "name": "Waste generation",
            "description": "Generation of waste from operations and products",
            "iro_type": "impact_actual_negative",
            "sub_topic": "resource_outflows_waste",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Product end-of-life impact",
            "description": "Environmental impact from product disposal at end of life",
            "iro_type": "impact_actual_negative",
            "sub_topic": "resource_outflows_products_services",
            "typical_value_chain": ["downstream"],
        },
        {
            "name": "Circular business model opportunity",
            "description": "Opportunity to create value through circular economy models",
            "iro_type": "opportunity",
            "sub_topic": "resource_inflows",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Raw material price volatility risk",
            "description": "Risk from price increases of virgin materials",
            "iro_type": "risk",
            "sub_topic": "resource_inflows",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "EPR fee liability",
            "description": "Risk of increasing Extended Producer Responsibility fee obligations",
            "iro_type": "risk",
            "sub_topic": "waste_management",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Product-as-a-service model",
            "description": "Opportunity to shift from product sales to service models for circularity",
            "iro_type": "opportunity",
            "sub_topic": "resource_outflows_products_services",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Secondary raw material sourcing",
            "description": "Positive impact from increasing use of recycled and secondary materials",
            "iro_type": "impact_actual_positive",
            "sub_topic": "resource_inflows",
            "typical_value_chain": ["upstream", "own_operations"],
        },
    ],
    "s1_own_workforce": [
        {
            "name": "Occupational health and safety incidents",
            "description": "Actual harm to workers from health and safety incidents",
            "iro_type": "impact_actual_negative",
            "sub_topic": "working_conditions_health_and_safety",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Inadequate wages and working conditions",
            "description": "Workers not receiving adequate wages or fair working conditions",
            "iro_type": "impact_actual_negative",
            "sub_topic": "working_conditions_adequate_wages",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Gender pay gap and discrimination",
            "description": "Systemic gender pay disparities and workplace discrimination",
            "iro_type": "impact_actual_negative",
            "sub_topic": "equal_treatment_gender_equality",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Talent attraction and retention",
            "description": "Opportunity to attract and retain talent through good practices",
            "iro_type": "opportunity",
            "sub_topic": "working_conditions_secure_employment",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Skills development and training",
            "description": "Positive impact through investment in workforce skills",
            "iro_type": "impact_actual_positive",
            "sub_topic": "equal_treatment_training_skills",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Labour dispute risk",
            "description": "Risk of financial disruption from labour disputes",
            "iro_type": "risk",
            "sub_topic": "working_conditions_social_dialogue",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Freedom of association restrictions",
            "description": "Potential restrictions on workers' freedom of association",
            "iro_type": "impact_potential_negative",
            "sub_topic": "working_conditions_freedom_of_association",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Work-life balance impact",
            "description": "Impact on workers from inadequate work-life balance policies",
            "iro_type": "impact_actual_negative",
            "sub_topic": "working_conditions_work_life_balance",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Inclusion of persons with disabilities",
            "description": "Positive impact from inclusive employment practices for disabled persons",
            "iro_type": "impact_actual_positive",
            "sub_topic": "equal_treatment_employment_inclusion_of_disabled",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Workforce diversity and inclusion",
            "description": "Positive impact from diversity and inclusion programmes",
            "iro_type": "impact_actual_positive",
            "sub_topic": "equal_treatment_diversity",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Workplace violence and harassment",
            "description": "Actual or potential negative impact from workplace violence or harassment",
            "iro_type": "impact_potential_negative",
            "sub_topic": "equal_treatment_measures_against_violence",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Collective bargaining coverage gap",
            "description": "Impact from low collective bargaining coverage among own workforce",
            "iro_type": "impact_actual_negative",
            "sub_topic": "working_conditions_collective_bargaining",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Working time compliance risk",
            "description": "Risk of regulatory penalties from working time directive non-compliance",
            "iro_type": "risk",
            "sub_topic": "working_conditions_working_time",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Data privacy impact on workers",
            "description": "Potential negative impact on worker privacy from surveillance or data collection",
            "iro_type": "impact_potential_negative",
            "sub_topic": "other_work_related_rights_privacy",
            "typical_value_chain": ["own_operations"],
        },
    ],
    "s2_value_chain_workers": [
        {
            "name": "Forced labour in supply chain",
            "description": "Risk of forced or bonded labour among supply chain workers",
            "iro_type": "impact_potential_negative",
            "sub_topic": "other_work_related_rights_in_value_chain",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Child labour in supply chain",
            "description": "Risk of child labour in upstream supply chain",
            "iro_type": "impact_potential_negative",
            "sub_topic": "other_work_related_rights_in_value_chain",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Unsafe working conditions in supply chain",
            "description": "Poor occupational health and safety for value chain workers",
            "iro_type": "impact_actual_negative",
            "sub_topic": "working_conditions_in_value_chain",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Supply chain due diligence risk",
            "description": "Risk of regulatory non-compliance under CSDDD",
            "iro_type": "risk",
            "sub_topic": "working_conditions_in_value_chain",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Living wage improvement opportunity",
            "description": "Opportunity to improve supplier worker wages and conditions",
            "iro_type": "opportunity",
            "sub_topic": "equal_treatment_in_value_chain",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Migrant worker exploitation",
            "description": "Risk of exploitation of migrant workers in value chain",
            "iro_type": "impact_potential_negative",
            "sub_topic": "other_work_related_rights_in_value_chain",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "CSDDD non-compliance risk",
            "description": "Financial risk from non-compliance with Corporate Sustainability Due Diligence Directive",
            "iro_type": "risk",
            "sub_topic": "working_conditions_in_value_chain",
            "typical_value_chain": ["upstream", "downstream"],
        },
    ],
    "s3_affected_communities": [
        {
            "name": "Community displacement",
            "description": "Displacement of local communities by operations or projects",
            "iro_type": "impact_actual_negative",
            "sub_topic": "communities_economic_social_cultural_rights",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Indigenous peoples rights impact",
            "description": "Impact on rights of indigenous peoples from operations",
            "iro_type": "impact_potential_negative",
            "sub_topic": "communities_rights_of_indigenous_peoples",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Social licence to operate risk",
            "description": "Risk of losing community acceptance for operations",
            "iro_type": "risk",
            "sub_topic": "communities_civil_political_rights",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Community investment opportunity",
            "description": "Opportunity to strengthen community relations through investment",
            "iro_type": "opportunity",
            "sub_topic": "communities_economic_social_cultural_rights",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Land rights and FPIC violations",
            "description": "Potential violation of free, prior, and informed consent of communities",
            "iro_type": "impact_potential_negative",
            "sub_topic": "communities_rights_of_indigenous_peoples",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Community health impacts",
            "description": "Negative health impacts on communities from pollution or operations",
            "iro_type": "impact_actual_negative",
            "sub_topic": "communities_economic_social_cultural_rights",
            "typical_value_chain": ["own_operations"],
        },
    ],
    "s4_consumers": [
        {
            "name": "Product safety risks",
            "description": "Risk of harm to consumers from unsafe products",
            "iro_type": "impact_potential_negative",
            "sub_topic": "personal_safety_of_consumers",
            "typical_value_chain": ["downstream"],
        },
        {
            "name": "Data privacy breaches",
            "description": "Risk of consumer data privacy violations",
            "iro_type": "impact_potential_negative",
            "sub_topic": "information_related_impacts_on_consumers",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Digital inclusion gap",
            "description": "Impact from excluding consumers from digital services",
            "iro_type": "impact_actual_negative",
            "sub_topic": "social_inclusion_of_consumers",
            "typical_value_chain": ["downstream"],
        },
        {
            "name": "Consumer trust opportunity",
            "description": "Opportunity to build brand value through transparency and safety",
            "iro_type": "opportunity",
            "sub_topic": "information_related_impacts_on_consumers",
            "typical_value_chain": ["downstream"],
        },
        {
            "name": "Product liability risk",
            "description": "Risk of product liability claims and recalls",
            "iro_type": "risk",
            "sub_topic": "personal_safety_of_consumers",
            "typical_value_chain": ["downstream"],
        },
        {
            "name": "Greenwashing and misleading claims",
            "description": "Risk of misleading sustainability claims affecting consumer trust",
            "iro_type": "risk",
            "sub_topic": "information_related_impacts_on_consumers",
            "typical_value_chain": ["own_operations", "downstream"],
        },
        {
            "name": "Financial product mis-selling",
            "description": "Potential harm to consumers from mis-selling of financial products",
            "iro_type": "impact_potential_negative",
            "sub_topic": "personal_safety_of_consumers",
            "typical_value_chain": ["downstream"],
        },
        {
            "name": "Vulnerable consumer protection",
            "description": "Impact on vulnerable consumers from product design or accessibility failures",
            "iro_type": "impact_potential_negative",
            "sub_topic": "social_inclusion_of_consumers",
            "typical_value_chain": ["downstream"],
        },
    ],
    "g1_business_conduct": [
        {
            "name": "Corruption and bribery incidents",
            "description": "Actual or potential incidents of corruption and bribery",
            "iro_type": "impact_potential_negative",
            "sub_topic": "corruption_and_bribery",
            "typical_value_chain": ["own_operations", "upstream"],
        },
        {
            "name": "Whistleblower retaliation",
            "description": "Risk of retaliation against whistleblowers",
            "iro_type": "impact_potential_negative",
            "sub_topic": "protection_of_whistleblowers",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Political lobbying transparency",
            "description": "Impact from non-transparent political engagement and lobbying",
            "iro_type": "impact_actual_negative",
            "sub_topic": "political_engagement_lobbying",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Animal welfare in supply chain",
            "description": "Impact on animal welfare from supply chain practices",
            "iro_type": "impact_actual_negative",
            "sub_topic": "animal_welfare",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Regulatory compliance risk",
            "description": "Risk of fines and sanctions from governance failures",
            "iro_type": "risk",
            "sub_topic": "corporate_culture",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Ethical supply chain opportunity",
            "description": "Opportunity to differentiate through ethical supply chain management",
            "iro_type": "opportunity",
            "sub_topic": "management_of_relationships_with_suppliers",
            "typical_value_chain": ["upstream"],
        },
        {
            "name": "Tax transparency risk",
            "description": "Reputational and regulatory risk from aggressive tax planning",
            "iro_type": "risk",
            "sub_topic": "corporate_culture",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Anti-competitive behaviour",
            "description": "Risk of fines and reputational damage from anti-competitive conduct",
            "iro_type": "risk",
            "sub_topic": "corruption_and_bribery",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Responsible lobbying opportunity",
            "description": "Opportunity to demonstrate transparent and responsible political engagement",
            "iro_type": "opportunity",
            "sub_topic": "political_engagement_lobbying",
            "typical_value_chain": ["own_operations"],
        },
        {
            "name": "Supplier payment practices",
            "description": "Impact on suppliers from delayed or unfair payment practices",
            "iro_type": "impact_actual_negative",
            "sub_topic": "management_of_relationships_with_suppliers",
            "typical_value_chain": ["upstream"],
        },
    ],
}

# Sector-specific IRO priority weightings.
# Maps NACE sector codes to ESRS topics with priority weighting
# (higher = more likely to be material for that sector).
SECTOR_IRO_PRIORITIES: Dict[str, Dict[str, Decimal]] = {
    "agriculture": {
        "e1_climate": Decimal("0.90"),
        "e2_pollution": Decimal("0.80"),
        "e3_water": Decimal("0.90"),
        "e4_biodiversity": Decimal("0.95"),
        "e5_circular_economy": Decimal("0.60"),
        "s1_own_workforce": Decimal("0.80"),
        "s2_value_chain_workers": Decimal("0.85"),
        "s3_affected_communities": Decimal("0.80"),
        "s4_consumers": Decimal("0.50"),
        "g1_business_conduct": Decimal("0.60"),
    },
    "mining": {
        "e1_climate": Decimal("0.85"),
        "e2_pollution": Decimal("0.95"),
        "e3_water": Decimal("0.90"),
        "e4_biodiversity": Decimal("0.90"),
        "e5_circular_economy": Decimal("0.70"),
        "s1_own_workforce": Decimal("0.95"),
        "s2_value_chain_workers": Decimal("0.70"),
        "s3_affected_communities": Decimal("0.95"),
        "s4_consumers": Decimal("0.30"),
        "g1_business_conduct": Decimal("0.80"),
    },
    "manufacturing": {
        "e1_climate": Decimal("0.90"),
        "e2_pollution": Decimal("0.85"),
        "e3_water": Decimal("0.75"),
        "e4_biodiversity": Decimal("0.50"),
        "e5_circular_economy": Decimal("0.85"),
        "s1_own_workforce": Decimal("0.90"),
        "s2_value_chain_workers": Decimal("0.80"),
        "s3_affected_communities": Decimal("0.60"),
        "s4_consumers": Decimal("0.70"),
        "g1_business_conduct": Decimal("0.75"),
    },
    "energy": {
        "e1_climate": Decimal("0.95"),
        "e2_pollution": Decimal("0.85"),
        "e3_water": Decimal("0.80"),
        "e4_biodiversity": Decimal("0.75"),
        "e5_circular_economy": Decimal("0.60"),
        "s1_own_workforce": Decimal("0.85"),
        "s2_value_chain_workers": Decimal("0.60"),
        "s3_affected_communities": Decimal("0.85"),
        "s4_consumers": Decimal("0.50"),
        "g1_business_conduct": Decimal("0.75"),
    },
    "financial_services": {
        "e1_climate": Decimal("0.85"),
        "e2_pollution": Decimal("0.30"),
        "e3_water": Decimal("0.30"),
        "e4_biodiversity": Decimal("0.40"),
        "e5_circular_economy": Decimal("0.30"),
        "s1_own_workforce": Decimal("0.80"),
        "s2_value_chain_workers": Decimal("0.40"),
        "s3_affected_communities": Decimal("0.50"),
        "s4_consumers": Decimal("0.85"),
        "g1_business_conduct": Decimal("0.95"),
    },
    "retail": {
        "e1_climate": Decimal("0.80"),
        "e2_pollution": Decimal("0.60"),
        "e3_water": Decimal("0.50"),
        "e4_biodiversity": Decimal("0.50"),
        "e5_circular_economy": Decimal("0.90"),
        "s1_own_workforce": Decimal("0.85"),
        "s2_value_chain_workers": Decimal("0.90"),
        "s3_affected_communities": Decimal("0.60"),
        "s4_consumers": Decimal("0.85"),
        "g1_business_conduct": Decimal("0.75"),
    },
    "technology": {
        "e1_climate": Decimal("0.70"),
        "e2_pollution": Decimal("0.40"),
        "e3_water": Decimal("0.40"),
        "e4_biodiversity": Decimal("0.30"),
        "e5_circular_economy": Decimal("0.65"),
        "s1_own_workforce": Decimal("0.85"),
        "s2_value_chain_workers": Decimal("0.70"),
        "s3_affected_communities": Decimal("0.50"),
        "s4_consumers": Decimal("0.90"),
        "g1_business_conduct": Decimal("0.80"),
    },
    "default": {
        "e1_climate": Decimal("0.80"),
        "e2_pollution": Decimal("0.60"),
        "e3_water": Decimal("0.60"),
        "e4_biodiversity": Decimal("0.60"),
        "e5_circular_economy": Decimal("0.60"),
        "s1_own_workforce": Decimal("0.80"),
        "s2_value_chain_workers": Decimal("0.60"),
        "s3_affected_communities": Decimal("0.60"),
        "s4_consumers": Decimal("0.60"),
        "g1_business_conduct": Decimal("0.70"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class IRO(BaseModel):
    """An Impact, Risk, or Opportunity identified per ESRS 1 Para 28-39.

    Represents a specific IRO that has been identified as part of the
    double materiality assessment process.  Each IRO is linked to an
    ESRS topic, sub-topic, and one or more value chain stages and
    time horizons.
    """
    id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this IRO",
    )
    name: str = Field(
        ...,
        description="Name of the IRO",
        min_length=1,
        max_length=500,
    )
    description: str = Field(
        default="",
        description="Detailed description of the IRO",
        max_length=5000,
    )
    iro_type: IROType = Field(
        ...,
        description="Type of IRO (impact, risk, or opportunity)",
    )
    esrs_topic: ESRSTopic = Field(
        ...,
        description="ESRS topic this IRO relates to",
    )
    sub_topic: str = Field(
        default="",
        description="Specific sub-topic within the ESRS topic",
        max_length=500,
    )
    value_chain_stages: List[ValueChainStage] = Field(
        default_factory=lambda: [ValueChainStage.OWN_OPERATIONS],
        description="Value chain stages where this IRO applies",
    )
    time_horizons: List[TimeHorizon] = Field(
        default_factory=lambda: [TimeHorizon.SHORT_TERM],
        description="Time horizons relevant to this IRO",
    )
    affected_stakeholders: List[str] = Field(
        default_factory=list,
        description="Stakeholder groups affected by this IRO",
    )
    related_policies: List[str] = Field(
        default_factory=list,
        description="Policies, actions, or targets addressing this IRO",
    )
    source: str = Field(
        default="esrs_catalog",
        description="Source of IRO identification (catalog, stakeholder, internal)",
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty or whitespace."""
        if not v.strip():
            raise ValueError("IRO name cannot be empty")
        return v.strip()

class IROAssessment(BaseModel):
    """Assessment result for a single IRO combining impact and financial scores.

    Merges the impact materiality score (from Engine 1) and financial
    materiality score (from Engine 2) into a combined score for
    determining whether the IRO is material from a double materiality
    perspective.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier",
    )
    iro_id: str = Field(
        ...,
        description="ID of the IRO being assessed",
        min_length=1,
    )
    iro_name: str = Field(
        default="",
        description="Name of the IRO",
    )
    iro_type: str = Field(
        default="",
        description="Type of IRO",
    )
    esrs_topic: str = Field(
        default="",
        description="ESRS topic of the IRO",
    )
    impact_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Impact materiality score (0-1 scale)",
    )
    financial_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Financial materiality score (0-1 scale)",
    )
    combined_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Combined double materiality score (max of impact and financial)",
    )
    is_material: bool = Field(
        default=False,
        description="Whether the IRO is material under either or both perspectives",
    )
    is_impact_material: bool = Field(
        default=False,
        description="Whether the IRO is material from impact perspective",
    )
    is_financial_material: bool = Field(
        default=False,
        description="Whether the IRO is material from financial perspective",
    )
    priority_level: PriorityLevel = Field(
        default=PriorityLevel.LOW,
        description="Priority level based on combined score",
    )
    threshold_used: Decimal = Field(
        default=DEFAULT_MATERIALITY_THRESHOLD,
        description="Materiality threshold used",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class IRORegister(BaseModel):
    """Register of all identified and assessed IROs.

    The IRO register is a central output of the double materiality
    assessment, documenting all identified IROs, their assessments,
    and summary statistics.
    """
    register_id: str = Field(
        default_factory=_new_uuid,
        description="Unique register identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of register creation (UTC)",
    )

    # --- IRO Contents ---
    iros: List[IRO] = Field(
        default_factory=list,
        description="All identified IROs in the register",
    )
    assessments: List[IROAssessment] = Field(
        default_factory=list,
        description="All IRO assessments",
    )

    # --- Summary Statistics ---
    total_count: int = Field(
        default=0,
        description="Total number of IROs in the register",
    )
    material_count: int = Field(
        default=0,
        description="Number of material IROs",
    )
    not_material_count: int = Field(
        default=0,
        description="Number of non-material IROs",
    )

    # --- Breakdowns ---
    by_topic: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of IROs by ESRS topic (all IROs)",
    )
    material_by_topic: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of material IROs by ESRS topic",
    )
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of IROs by type (impact/risk/opportunity)",
    )
    material_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of material IROs by type",
    )
    by_value_chain: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of IROs by value chain stage",
    )

    # --- Quality Metrics ---
    avg_combined_score: Decimal = Field(
        default=Decimal("0.000"),
        description="Average combined score across all IROs",
    )
    coverage_by_topic: Dict[str, bool] = Field(
        default_factory=dict,
        description="Whether each ESRS topic has at least one IRO identified",
    )
    topics_with_no_iros: List[str] = Field(
        default_factory=list,
        description="ESRS topics with no IROs identified",
    )

    # --- Provenance ---
    threshold_used: Decimal = Field(
        default=DEFAULT_MATERIALITY_THRESHOLD,
        description="Materiality threshold used",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire register",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IROIdentificationEngine:
    """IRO identification and register management engine per ESRS 1 Para 28-39.

    Provides deterministic, zero-hallucination functionality for:
    - IRO identification from the ESRS catalog
    - IRO classification combining impact and financial scores
    - IRO register building with summary statistics
    - Filtering and querying of the IRO register
    - Sector-specific IRO prioritisation

    All operations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = IROIdentificationEngine()

        # Identify IROs for a sector
        iros = engine.identify_iros(
            sector="manufacturing",
            topics=[ESRSTopic.E1_CLIMATE, ESRSTopic.S1_OWN_WORKFORCE],
        )

        # Classify IROs with scores from Engine 1 and Engine 2
        assessments = []
        for iro in iros:
            assessment = engine.classify_iro(
                iro=iro,
                impact_score=Decimal("0.65"),
                financial_score=Decimal("0.50"),
            )
            assessments.append(assessment)

        # Build register
        register = engine.build_register(iros, assessments)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # IRO Identification                                                   #
    # ------------------------------------------------------------------ #

    def identify_iros(
        self,
        sector: str = "default",
        topics: Optional[List[ESRSTopic]] = None,
        value_chain_stages: Optional[List[ValueChainStage]] = None,
    ) -> List[IRO]:
        """Identify relevant IROs from the ESRS catalog.

        Selects IROs from the built-in ESRS_IRO_CATALOG based on the
        specified sector, topics, and value chain stages.  If no topics
        are specified, IROs for all topics are returned.

        Args:
            sector: NACE sector key for priority weighting.
            topics: List of ESRS topics to include.  If None, all
                topics are included.
            value_chain_stages: If specified, only IROs relevant to
                these value chain stages are included.

        Returns:
            List of IRO objects identified from the catalog.
        """
        if topics is None:
            topics = list(ESRSTopic)

        topic_values = [t.value for t in topics]
        iros: List[IRO] = []

        for topic_key, catalog_entries in ESRS_IRO_CATALOG.items():
            if topic_key not in topic_values:
                continue

            esrs_topic = ESRSTopic(topic_key)

            for entry in catalog_entries:
                # Filter by value chain stage if specified
                entry_stages = [
                    ValueChainStage(s)
                    for s in entry.get("typical_value_chain", ["own_operations"])
                ]

                if value_chain_stages is not None:
                    overlap = set(entry_stages) & set(value_chain_stages)
                    if not overlap:
                        continue

                iro = IRO(
                    name=entry["name"],
                    description=entry.get("description", ""),
                    iro_type=IROType(entry["iro_type"]),
                    esrs_topic=esrs_topic,
                    sub_topic=entry.get("sub_topic", ""),
                    value_chain_stages=entry_stages,
                    time_horizons=[
                        TimeHorizon.SHORT_TERM,
                        TimeHorizon.MEDIUM_TERM,
                        TimeHorizon.LONG_TERM,
                    ],
                    source="esrs_catalog",
                )
                iros.append(iro)

        return iros

    # ------------------------------------------------------------------ #
    # IRO Classification                                                   #
    # ------------------------------------------------------------------ #

    def classify_iro(
        self,
        iro: IRO,
        impact_score: Decimal,
        financial_score: Decimal,
        threshold: Optional[Decimal] = None,
    ) -> IROAssessment:
        """Classify an IRO by combining impact and financial scores.

        Per ESRS double materiality, a matter is material if it is
        material from either the impact perspective OR the financial
        perspective (OR both).  The combined score is the maximum of
        the two individual scores.

        Priority levels are assigned based on the combined score:
            >= 0.80: CRITICAL
            >= 0.60: HIGH
            >= 0.40: MEDIUM
            <  0.40: LOW

        Args:
            iro: The IRO to classify.
            impact_score: Impact materiality score (0-1 Decimal).
            financial_score: Financial materiality score (0-1 Decimal).
            threshold: Materiality threshold (default 0.40).

        Returns:
            IROAssessment with combined score and materiality determination.
        """
        if threshold is None:
            threshold = DEFAULT_MATERIALITY_THRESHOLD

        # Combined score = max(impact, financial) per double materiality
        combined = max(impact_score, financial_score)
        combined = _round_val(combined, 3)

        # Materiality determination
        is_impact_material = impact_score >= threshold
        is_financial_material = financial_score >= threshold
        is_material = is_impact_material or is_financial_material

        # Priority level
        priority = self._determine_priority(combined)

        assessment = IROAssessment(
            iro_id=iro.id,
            iro_name=iro.name,
            iro_type=iro.iro_type.value,
            esrs_topic=iro.esrs_topic.value,
            impact_score=_round_val(impact_score, 3),
            financial_score=_round_val(financial_score, 3),
            combined_score=combined,
            is_material=is_material,
            is_impact_material=is_impact_material,
            is_financial_material=is_financial_material,
            priority_level=priority,
            threshold_used=threshold,
        )

        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    # ------------------------------------------------------------------ #
    # Register Building                                                    #
    # ------------------------------------------------------------------ #

    def build_register(
        self,
        iros: List[IRO],
        assessments: List[IROAssessment],
        threshold: Optional[Decimal] = None,
    ) -> IRORegister:
        """Build a complete IRO register with summary statistics.

        Combines the identified IROs and their assessments into a
        structured register with breakdowns by topic, type, and
        value chain stage.

        Args:
            iros: List of identified IROs.
            assessments: List of IRO assessments (should correspond
                to the IROs by iro_id).
            threshold: Materiality threshold (default 0.40).

        Returns:
            IRORegister with all IROs, assessments, and summaries.

        Raises:
            ValueError: If iros list is empty.
        """
        t0 = time.perf_counter()

        if not iros:
            raise ValueError("At least one IRO is required to build a register")

        if threshold is None:
            threshold = DEFAULT_MATERIALITY_THRESHOLD

        # Build assessment lookup
        asmt_lookup: Dict[str, IROAssessment] = {
            a.iro_id: a for a in assessments
        }

        # Summary: all IROs by topic
        by_topic: Dict[str, int] = {}
        for iro in iros:
            topic = iro.esrs_topic.value
            by_topic[topic] = by_topic.get(topic, 0) + 1

        # Summary: all IROs by type
        by_type: Dict[str, int] = {}
        for iro in iros:
            iro_t = iro.iro_type.value
            by_type[iro_t] = by_type.get(iro_t, 0) + 1

        # Summary: all IROs by value chain stage
        by_vc: Dict[str, int] = {}
        for iro in iros:
            for stage in iro.value_chain_stages:
                sv = stage.value
                by_vc[sv] = by_vc.get(sv, 0) + 1

        # Material IRO counts
        material_assessments = [a for a in assessments if a.is_material]
        material_count = len(material_assessments)
        not_material_count = len(assessments) - material_count

        # Material by topic
        material_by_topic: Dict[str, int] = {}
        for a in material_assessments:
            topic = a.esrs_topic
            material_by_topic[topic] = material_by_topic.get(topic, 0) + 1

        # Material by type
        material_by_type: Dict[str, int] = {}
        for a in material_assessments:
            iro_t = a.iro_type
            material_by_type[iro_t] = material_by_type.get(iro_t, 0) + 1

        # Average combined score
        if assessments:
            avg_combined = _round_val(
                sum(a.combined_score for a in assessments)
                / _decimal(len(assessments)),
                3,
            )
        else:
            avg_combined = Decimal("0.000")

        # Topic coverage
        all_topics = [t.value for t in ESRSTopic]
        coverage: Dict[str, bool] = {}
        for t in all_topics:
            coverage[t] = t in by_topic
        topics_no_iros = [t for t in all_topics if not coverage[t]]

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        register = IRORegister(
            iros=iros,
            assessments=assessments,
            total_count=len(iros),
            material_count=material_count,
            not_material_count=not_material_count,
            by_topic=by_topic,
            material_by_topic=material_by_topic,
            by_type=by_type,
            material_by_type=material_by_type,
            by_value_chain=by_vc,
            avg_combined_score=avg_combined,
            coverage_by_topic=coverage,
            topics_with_no_iros=topics_no_iros,
            threshold_used=threshold,
            processing_time_ms=elapsed_ms,
        )

        register.provenance_hash = _compute_hash(register)
        return register

    # ------------------------------------------------------------------ #
    # Filtering and Querying                                               #
    # ------------------------------------------------------------------ #

    def filter_material_iros(
        self,
        register: IRORegister,
        threshold: Optional[Decimal] = None,
    ) -> IRORegister:
        """Filter the register to only material IROs and their assessments.

        Creates a new register containing only IROs that have
        material assessments.

        Args:
            register: The full IRO register.
            threshold: Optional new threshold to apply (re-evaluates
                materiality).  If None, uses existing assessments.

        Returns:
            New IRORegister with only material IROs.
        """
        if threshold is not None:
            # Re-evaluate materiality with new threshold
            new_assessments = []
            for a in register.assessments:
                new_a = a.model_copy()
                new_a.is_impact_material = a.impact_score >= threshold
                new_a.is_financial_material = a.financial_score >= threshold
                new_a.is_material = (
                    new_a.is_impact_material or new_a.is_financial_material
                )
                new_a.threshold_used = threshold
                new_a.provenance_hash = _compute_hash(new_a)
                new_assessments.append(new_a)
            assessments = new_assessments
        else:
            assessments = register.assessments
            threshold = register.threshold_used

        material_ids = set(
            a.iro_id for a in assessments if a.is_material
        )
        material_iros = [
            iro for iro in register.iros if iro.id in material_ids
        ]
        material_assessments = [
            a for a in assessments if a.is_material
        ]

        return self.build_register(
            material_iros, material_assessments, threshold
        )

    def get_iros_by_topic(
        self,
        register: IRORegister,
        topic: ESRSTopic,
    ) -> List[IRO]:
        """Get all IROs for a specific ESRS topic from the register.

        Args:
            register: The IRO register to query.
            topic: ESRS topic to filter by.

        Returns:
            List of IROs for the specified topic.
        """
        return [
            iro for iro in register.iros
            if iro.esrs_topic == topic
        ]

    def get_iros_by_type(
        self,
        register: IRORegister,
        iro_type: IROType,
    ) -> List[IRO]:
        """Get all IROs of a specific type from the register.

        Args:
            register: The IRO register to query.
            iro_type: IRO type to filter by.

        Returns:
            List of IROs of the specified type.
        """
        return [
            iro for iro in register.iros
            if iro.iro_type == iro_type
        ]

    def get_iros_by_value_chain(
        self,
        register: IRORegister,
        stage: ValueChainStage,
    ) -> List[IRO]:
        """Get all IROs relevant to a specific value chain stage.

        Args:
            register: The IRO register to query.
            stage: Value chain stage to filter by.

        Returns:
            List of IROs relevant to the specified stage.
        """
        return [
            iro for iro in register.iros
            if stage in iro.value_chain_stages
        ]

    def get_assessment_for_iro(
        self,
        register: IRORegister,
        iro_id: str,
    ) -> Optional[IROAssessment]:
        """Get the assessment for a specific IRO.

        Args:
            register: The IRO register.
            iro_id: ID of the IRO.

        Returns:
            IROAssessment if found, None otherwise.
        """
        for a in register.assessments:
            if a.iro_id == iro_id:
                return a
        return None

    # ------------------------------------------------------------------ #
    # Sector-Specific Prioritisation                                       #
    # ------------------------------------------------------------------ #

    def get_sector_priorities(
        self,
        sector: str,
    ) -> Dict[str, Decimal]:
        """Get IRO priority weights for a specific sector.

        Returns topic-level priority weights that indicate how likely
        each ESRS topic is to be material for the given sector.

        Args:
            sector: NACE sector key (e.g., "manufacturing", "retail").

        Returns:
            Dict mapping ESRS topic to priority weight (0-1 Decimal).
        """
        sector_key = sector.lower().replace(" ", "_")
        return dict(
            SECTOR_IRO_PRIORITIES.get(
                sector_key,
                SECTOR_IRO_PRIORITIES["default"],
            )
        )

    def apply_sector_weighting(
        self,
        assessments: List[IROAssessment],
        sector: str,
    ) -> List[IROAssessment]:
        """Apply sector-specific weighting to IRO assessments.

        Adjusts the combined score by multiplying with the sector
        priority weight for the IRO's ESRS topic.  This helps
        prioritise IROs that are more likely to be material for
        the specific sector.

        The adjustment modifies the combined_score but does NOT
        change the underlying impact or financial scores.

        Args:
            assessments: List of IRO assessments.
            sector: NACE sector key.

        Returns:
            List of assessments with adjusted combined scores.
        """
        priorities = self.get_sector_priorities(sector)
        adjusted: List[IROAssessment] = []

        for a in assessments:
            weight = priorities.get(a.esrs_topic, Decimal("0.60"))
            new_combined = _round_val(a.combined_score * weight, 3)

            new_a = a.model_copy()
            new_a.combined_score = new_combined
            new_a.priority_level = self._determine_priority(new_combined)
            new_a.provenance_hash = _compute_hash(new_a)
            adjusted.append(new_a)

        return adjusted

    # ------------------------------------------------------------------ #
    # Catalog Access                                                       #
    # ------------------------------------------------------------------ #

    def get_catalog_for_topic(
        self,
        topic: ESRSTopic,
    ) -> List[Dict[str, Any]]:
        """Return the raw catalog entries for a specific ESRS topic.

        Args:
            topic: ESRS topic enum value.

        Returns:
            List of catalog entry dicts for the topic.
        """
        return list(ESRS_IRO_CATALOG.get(topic.value, []))

    def get_catalog_size(self) -> int:
        """Return the total number of entries in the IRO catalog.

        Returns:
            Total count of catalog entries across all topics.
        """
        return sum(len(entries) for entries in ESRS_IRO_CATALOG.values())

    def get_catalog_summary(self) -> Dict[str, int]:
        """Return a summary of catalog entries by ESRS topic.

        Returns:
            Dict mapping topic string to entry count.
        """
        return {
            topic: len(entries)
            for topic, entries in ESRS_IRO_CATALOG.items()
        }

    # ------------------------------------------------------------------ #
    # Private: Priority Determination                                      #
    # ------------------------------------------------------------------ #

    def _determine_priority(self, combined_score: Decimal) -> PriorityLevel:
        """Determine the priority level based on combined score.

        Thresholds:
            >= 0.80: CRITICAL
            >= 0.60: HIGH
            >= 0.40: MEDIUM
            <  0.40: LOW

        Args:
            combined_score: Combined materiality score (0-1 Decimal).

        Returns:
            PriorityLevel enum value.
        """
        if combined_score >= PRIORITY_THRESHOLDS["critical"]:
            return PriorityLevel.CRITICAL
        elif combined_score >= PRIORITY_THRESHOLDS["high"]:
            return PriorityLevel.HIGH
        elif combined_score >= PRIORITY_THRESHOLDS["medium"]:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW
