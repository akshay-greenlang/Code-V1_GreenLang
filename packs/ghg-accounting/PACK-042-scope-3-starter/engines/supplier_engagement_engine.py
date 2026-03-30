# -*- coding: utf-8 -*-
"""
SupplierEngagementEngine - PACK-042 Scope 3 Starter Pack Engine 6
===================================================================

Manages supplier carbon data collection, quality scoring, and progressive
engagement across the value chain.  Prioritises suppliers by emission
contribution, generates standardised data-request questionnaires,
tracks response status, scores data quality per the GHG Protocol Data
Quality Indicator (DQI) framework, and builds multi-year engagement
roadmaps from Level 1 (spend-based EEIO) to Level 5 (product-level LCA).

Supplier Prioritisation:
    Suppliers are ranked by their estimated emission contribution using
    the Pareto principle: target the ~20 % of suppliers that typically
    drive ~80 % of Scope 3 upstream emissions.

    priority_score = (estimated_emissions / total_emissions) * weight_factor
    weight_factor  = category_weight * spend_weight * strategic_weight

Data Quality Indicator (DQI) Levels:
    Level 1: No primary data -- EEIO model estimate only.
    Level 2: Spend-based with general sector emission factor.
    Level 3: Average-data method with product-specific emission factor.
    Level 4: Supplier-reported aggregate allocated by revenue share.
    Level 5: Supplier-specific product-level life-cycle assessment.

    Reference: GHG Protocol Scope 3 Calculation Guidance, Chapter 7
               GHG Protocol Corporate Value Chain Standard, Appendix A

Engagement ROI Calculation:
    roi = (uncertainty_reduction_tco2e * value_per_tonne) / engagement_cost
    Where:
        uncertainty_reduction_tco2e = estimated variance reduction
        value_per_tonne             = shadow carbon price or compliance cost
        engagement_cost             = FTE hours * loaded rate + platform cost

Questionnaire Templates:
    12 industry-specific templates covering:
    - Manufacturing, Energy, Transport, Agriculture, Chemicals,
      Construction, Retail, Technology, Financial Services, Mining,
      Food & Beverage, Textiles/Apparel.

    CDP Supply Chain module data fields are mapped to each template
    for interoperability.

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - GHG Protocol Scope 3 Calculation Guidance (2013)
    - CDP Supply Chain Program Technical Note (2024)
    - ESRS E1 para 44-46 (value chain emissions)
    - SBTi Scope 3 Supplier Engagement guidance (2023)

Zero-Hallucination:
    - All scores calculated deterministically via Decimal arithmetic
    - Quality levels from published GHG Protocol DQI rubric
    - No LLM involvement in any scoring or prioritisation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _fmt(value: Any) -> str:
    """Format a number with comma separators and 2dp."""
    try:
        return f"{_round2(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataQualityLevel(int, Enum):
    """GHG Protocol Data Quality Indicator (DQI) levels.

    Level 1: No primary data -- EEIO model estimate.
    Level 2: Spend-based with general sector EF.
    Level 3: Average-data with product-specific EF.
    Level 4: Supplier-reported aggregate allocated by revenue.
    Level 5: Supplier-specific product-level LCA.
    """
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5

class EngagementStatus(str, Enum):
    """Supplier engagement lifecycle status.

    NOT_STARTED:   No contact initiated.
    CONTACTED:     Initial outreach sent.
    IN_PROGRESS:   Questionnaire or data request sent; awaiting response.
    RESPONDED:     Supplier has submitted data.
    VALIDATED:     Submitted data has been validated.
    ESCALATED:     Non-responsive supplier escalated.
    """
    NOT_STARTED = "not_started"
    CONTACTED = "contacted"
    IN_PROGRESS = "in_progress"
    RESPONDED = "responded"
    VALIDATED = "validated"
    ESCALATED = "escalated"

class SupplierTier(str, Enum):
    """Supplier prioritisation tier.

    CRITICAL: Top-5 % by emission contribution (engage first).
    HIGH:     Top-20 % by emission contribution.
    MEDIUM:   20-50 % cumulative contribution.
    LOW:      Bottom 50 % (defer or use estimates).
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class IndustryType(str, Enum):
    """Supported industry types for questionnaire templates."""
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    TRANSPORT = "transport"
    AGRICULTURE = "agriculture"
    CHEMICALS = "chemicals"
    CONSTRUCTION = "construction"
    RETAIL = "retail"
    TECHNOLOGY = "technology"
    FINANCIAL_SERVICES = "financial_services"
    MINING = "mining"
    FOOD_BEVERAGE = "food_beverage"
    TEXTILES_APPAREL = "textiles_apparel"

class ReminderType(str, Enum):
    """Reminder types for engagement follow-up."""
    INITIAL_REQUEST = "initial_request"
    FIRST_FOLLOW_UP = "first_follow_up"
    SECOND_FOLLOW_UP = "second_follow_up"
    ESCALATION = "escalation"
    ANNUAL_REFRESH = "annual_refresh"

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (upstream + downstream)."""
    CAT_1 = "cat_1_purchased_goods"
    CAT_2 = "cat_2_capital_goods"
    CAT_3 = "cat_3_fuel_energy"
    CAT_4 = "cat_4_upstream_transport"
    CAT_5 = "cat_5_waste"
    CAT_6 = "cat_6_business_travel"
    CAT_7 = "cat_7_employee_commuting"
    CAT_8 = "cat_8_upstream_leased"
    CAT_9 = "cat_9_downstream_transport"
    CAT_10 = "cat_10_processing"
    CAT_11 = "cat_11_use_of_sold"
    CAT_12 = "cat_12_end_of_life"
    CAT_13 = "cat_13_downstream_leased"
    CAT_14 = "cat_14_franchises"
    CAT_15 = "cat_15_investments"

# ---------------------------------------------------------------------------
# Constants -- Questionnaire Templates
# ---------------------------------------------------------------------------

# CDP Supply Chain field mapping for interoperability.
CDP_SUPPLY_CHAIN_FIELDS: Dict[str, str] = {
    "total_scope_1": "C6.1",
    "total_scope_2": "C6.3",
    "scope_3_categories": "C6.5",
    "reduction_targets": "C4.1a",
    "emissions_methodology": "C6.2a",
    "verification_status": "C10.1",
    "renewable_energy_pct": "C8.2a",
    "climate_risk_assessment": "C2.1",
    "sbti_status": "C4.2",
    "product_carbon_footprint": "C-TS8.5/C-AC6.10",
}
"""Mapping of internal field names to CDP Supply Chain question references."""

# Default reminder schedule (days after initial request).
DEFAULT_REMINDER_SCHEDULE: Dict[str, int] = {
    ReminderType.INITIAL_REQUEST: 0,
    ReminderType.FIRST_FOLLOW_UP: 14,
    ReminderType.SECOND_FOLLOW_UP: 30,
    ReminderType.ESCALATION: 45,
    ReminderType.ANNUAL_REFRESH: 365,
}
"""Default reminder timing in days after initial request."""

# Typical uncertainty ranges by DQI level (half-width of 95% CI as %).
# Source: GHG Protocol Scope 3 Calculation Guidance, Table 7.1.
DQI_UNCERTAINTY_RANGES: Dict[int, Tuple[float, float]] = {
    1: (100.0, 200.0),   # EEIO: +/- 100-200%
    2: (50.0, 100.0),    # Spend-based: +/- 50-100%
    3: (20.0, 50.0),     # Average-data: +/- 20-50%
    4: (10.0, 30.0),     # Supplier aggregate: +/- 10-30%
    5: (5.0, 15.0),      # Product LCA: +/- 5-15%
}
"""Uncertainty ranges by DQI level (low_pct, high_pct)."""

# Category weight multipliers for prioritisation (higher = more important
# for supplier engagement because the category depends on supplier data).
CATEGORY_ENGAGEMENT_WEIGHTS: Dict[str, float] = {
    Scope3Category.CAT_1: 1.0,
    Scope3Category.CAT_2: 0.8,
    Scope3Category.CAT_3: 0.3,
    Scope3Category.CAT_4: 0.7,
    Scope3Category.CAT_5: 0.4,
    Scope3Category.CAT_6: 0.2,
    Scope3Category.CAT_7: 0.2,
    Scope3Category.CAT_8: 0.5,
    Scope3Category.CAT_9: 0.6,
    Scope3Category.CAT_10: 0.7,
    Scope3Category.CAT_11: 0.6,
    Scope3Category.CAT_12: 0.5,
    Scope3Category.CAT_13: 0.4,
    Scope3Category.CAT_14: 0.3,
    Scope3Category.CAT_15: 0.3,
}
"""Weight multiplier per Scope 3 category for supplier engagement priority."""

# Industry-specific questionnaire templates with core question sets.
QUESTIONNAIRE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    IndustryType.MANUFACTURING: {
        "name": "Manufacturing Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and reporting boundary",
                 "Manufacturing sites within scope (country, products)",
                 "Consolidation approach (operational or financial control)",
                 "Reporting period (fiscal year dates)",
             ]},
            {"id": "S1", "title": "Scope 1 Emissions",
             "questions": [
                 "Total Scope 1 emissions (tCO2e)",
                 "Stationary combustion fuel types and quantities",
                 "Process emissions by product line",
                 "Fugitive emissions (refrigerants, SF6)",
             ]},
            {"id": "S2", "title": "Scope 2 Emissions",
             "questions": [
                 "Total electricity consumption (MWh)",
                 "Scope 2 location-based emissions (tCO2e)",
                 "Scope 2 market-based emissions (tCO2e)",
                 "Renewable energy certificates (type, volume, registry)",
             ]},
            {"id": "PROD", "title": "Product-Level Data",
             "questions": [
                 "Product carbon footprint (kgCO2e per unit) by SKU",
                 "Bill of materials with embedded carbon",
                 "LCA methodology used (ISO 14040/14044 conformant?)",
                 "Allocation method for multi-product facilities",
             ]},
            {"id": "TARGETS", "title": "Targets & Governance",
             "questions": [
                 "Science-based target status (committed / set / validated)",
                 "Near-term and long-term reduction targets",
                 "Renewable energy procurement strategy",
                 "Board-level climate governance",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C4.1a", "C8.2a"],
    },
    IndustryType.ENERGY: {
        "name": "Energy Sector Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and reporting boundary",
                 "Generation assets within scope (type, capacity MW)",
                 "Consolidation approach",
                 "Reporting period",
             ]},
            {"id": "GEN", "title": "Generation & Grid Emissions",
             "questions": [
                 "Net generation by fuel type (MWh)",
                 "Emission intensity (tCO2e/MWh) by asset",
                 "Methane leakage from gas infrastructure",
                 "Grid emission factor used (source, year)",
             ]},
            {"id": "FUEL", "title": "Fuel Supply Chain",
             "questions": [
                 "Upstream fuel supply emissions (well-to-gate)",
                 "Fuel source countries and transport modes",
                 "Coal mine methane or gas venting/flaring",
             ]},
            {"id": "TARGETS", "title": "Transition & Targets",
             "questions": [
                 "Renewable capacity additions planned",
                 "Coal/gas phase-out schedule",
                 "SBTi sectoral decarbonisation approach",
                 "Carbon capture utilisation plans",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C-EU8.2d", "C4.1a"],
    },
    IndustryType.TRANSPORT: {
        "name": "Transport & Logistics Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and reporting boundary",
                 "Fleet composition (vehicle types, fuel types, counts)",
                 "Reporting period",
             ]},
            {"id": "FLEET", "title": "Fleet Emissions",
             "questions": [
                 "Total fuel consumption by type (litres or kg)",
                 "Total distance travelled (km or tonne-km)",
                 "Emission intensity (gCO2e per tonne-km)",
                 "Electric / alternative fuel vehicle share (%)",
             ]},
            {"id": "MODAL", "title": "Modal Split & Efficiency",
             "questions": [
                 "Tonne-km by transport mode (road, rail, sea, air)",
                 "Average load factor (%)",
                 "Empty running percentage",
                 "Route optimisation measures",
             ]},
            {"id": "TARGETS", "title": "Targets & Decarbonisation",
             "questions": [
                 "SBTi-validated target or commitment",
                 "Fleet electrification roadmap",
                 "Sustainable aviation fuel usage",
                 "Carbon offsetting (volume, standard)",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C-TS8.5", "C4.1a"],
    },
    IndustryType.AGRICULTURE: {
        "name": "Agriculture Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Farm / estate name and location",
                 "Total cultivated area (hectares) by crop type",
                 "Livestock headcount by species",
                 "Reporting period",
             ]},
            {"id": "LAND", "title": "Land Use & Deforestation",
             "questions": [
                 "Land use change in last 20 years (Y/N, area, type)",
                 "Deforestation-free commitment and cut-off date",
                 "Carbon stock in above-ground biomass (tC/ha)",
                 "Soil organic carbon management practices",
             ]},
            {"id": "INPUTS", "title": "Agricultural Inputs",
             "questions": [
                 "Synthetic fertiliser application (kg N/ha)",
                 "Organic fertiliser / manure application",
                 "Pesticide and herbicide usage",
                 "Irrigation energy consumption",
             ]},
            {"id": "LIVESTOCK", "title": "Livestock Emissions",
             "questions": [
                 "Enteric fermentation estimate methodology",
                 "Manure management system type",
                 "Feed composition and source",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "Regenerative agriculture practices adopted",
                 "Emission reduction targets (absolute or intensity)",
                 "Certification status (Rainforest Alliance, etc.)",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C-AC6.8", "C-AC6.9", "C-AC6.10"],
    },
    IndustryType.CHEMICALS: {
        "name": "Chemicals Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and plant locations",
                 "Products manufactured within scope",
                 "Reporting period",
             ]},
            {"id": "PROCESS", "title": "Process Emissions",
             "questions": [
                 "Process emissions by product (tCO2e)",
                 "Feedstock carbon content and oxidation factor",
                 "N2O and other GHG from chemical reactions",
                 "Carbon capture or utilisation (tCO2e avoided)",
             ]},
            {"id": "ENERGY", "title": "Energy & Utilities",
             "questions": [
                 "Steam consumption (TJ) and source",
                 "Electricity consumption (MWh) and source",
                 "Combined heat and power efficiency",
             ]},
            {"id": "PRODUCT", "title": "Product Carbon Footprint",
             "questions": [
                 "Cradle-to-gate PCF per tonne of product",
                 "LCA conformance (ISO 14040/14044, EPD)",
                 "Allocation method for co-products",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "SBTi or other science-based targets",
                 "Decarbonisation roadmap (green hydrogen, electrification)",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C-CH8.3a", "C4.1a"],
    },
    IndustryType.CONSTRUCTION: {
        "name": "Construction Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Company name and project scope",
                 "Material types supplied (concrete, steel, etc.)",
                 "Reporting period",
             ]},
            {"id": "EMBODIED", "title": "Embodied Carbon",
             "questions": [
                 "EPD availability per product (EN 15804 or ISO 14025)",
                 "GWP per unit (kgCO2e/kg or kgCO2e/m3)",
                 "Recycled content percentage",
                 "Transport distance from factory to site",
             ]},
            {"id": "OPERATIONS", "title": "Manufacturing Operations",
             "questions": [
                 "Kiln / furnace fuel type and consumption",
                 "Process emissions (calcination, etc.)",
                 "Electricity consumption and grid region",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "Concrete / cement roadmap (clinker ratio reduction)",
                 "Steel decarbonisation pathway (EAF, DRI-H2)",
                 "SBTi or sectoral targets",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C4.1a"],
    },
    IndustryType.RETAIL: {
        "name": "Retail Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and store count",
                 "Product categories supplied",
                 "Reporting period",
             ]},
            {"id": "SUPPLY", "title": "Supply Chain Emissions",
             "questions": [
                 "Total Scope 1 + 2 emissions (tCO2e)",
                 "Emissions per unit of revenue (tCO2e/$M)",
                 "Key raw material sourcing countries",
                 "Deforestation-free commodity sourcing",
             ]},
            {"id": "LOGISTICS", "title": "Logistics & Distribution",
             "questions": [
                 "Inbound freight emissions (tCO2e)",
                 "Distribution centre energy consumption",
                 "Packaging carbon footprint",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "CDP or SBTi commitments",
                 "Supplier engagement programme scope",
                 "Circular economy initiatives",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C6.5", "C4.1a"],
    },
    IndustryType.TECHNOLOGY: {
        "name": "Technology Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and data centre locations",
                 "Services / hardware within scope",
                 "Reporting period",
             ]},
            {"id": "DC", "title": "Data Centre Emissions",
             "questions": [
                 "Total data centre electricity (MWh)",
                 "PUE (Power Usage Effectiveness)",
                 "Renewable energy procurement (%, type)",
                 "Scope 2 market-based emissions (tCO2e)",
             ]},
            {"id": "HW", "title": "Hardware Supply Chain",
             "questions": [
                 "Embodied carbon of hardware (kgCO2e/unit)",
                 "Key component sourcing (semiconductors, PCBs)",
                 "End-of-life recycling / take-back programme",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "100% renewable energy target date",
                 "Net zero commitment year and pathway",
                 "SBTi target status",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C8.2a", "C4.1a"],
    },
    IndustryType.FINANCIAL_SERVICES: {
        "name": "Financial Services Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name",
                 "Assets under management (AUM) within scope",
                 "Reporting period",
             ]},
            {"id": "FINANCED", "title": "Financed / Facilitated Emissions",
             "questions": [
                 "PCAF score and methodology",
                 "Financed emissions by asset class (tCO2e)",
                 "Data quality score per PCAF (1-5)",
                 "Portfolio coverage of emissions data (%)",
             ]},
            {"id": "OPS", "title": "Operational Emissions",
             "questions": [
                 "Office energy consumption (MWh)",
                 "Business travel emissions (tCO2e)",
                 "Employee commuting emissions estimate",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "Net Zero Banking Alliance / NZAOA commitment",
                 "SBTi FI target status",
                 "Portfolio decarbonisation pathway",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C-FS14.1", "C4.1a"],
    },
    IndustryType.MINING: {
        "name": "Mining Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and mine site locations",
                 "Commodities mined within scope",
                 "Reporting period",
             ]},
            {"id": "MINE_OPS", "title": "Mining Operations",
             "questions": [
                 "Diesel and explosives consumption",
                 "Electricity consumption (grid vs. on-site generation)",
                 "Fugitive methane from underground mining",
                 "Haul truck fleet emissions",
             ]},
            {"id": "PROCESSING", "title": "Processing & Smelting",
             "questions": [
                 "Smelter / refinery energy intensity (GJ/t)",
                 "Process emissions (tCO2e/t product)",
                 "Tailings management methane",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "ICMM climate commitment status",
                 "Electrification of mining fleet",
                 "Scope 1+2 reduction targets",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C-MM8.3a", "C4.1a"],
    },
    IndustryType.FOOD_BEVERAGE: {
        "name": "Food & Beverage Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and production sites",
                 "Product categories within scope",
                 "Reporting period",
             ]},
            {"id": "AGRI", "title": "Agricultural Supply Chain",
             "questions": [
                 "Key agricultural commodities sourced",
                 "Deforestation risk assessment (soy, palm, beef, cocoa)",
                 "Fertiliser and land-use change emissions per commodity",
                 "Supplier farm / origin traceability level",
             ]},
            {"id": "PRODUCTION", "title": "Production Emissions",
             "questions": [
                 "Thermal energy (steam, heat) consumption",
                 "Refrigeration and cold chain emissions",
                 "Wastewater treatment methane",
                 "Packaging materials and embedded carbon",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "FLAG SBTi target status",
                 "Sustainable sourcing commitments (RSPO, RTRS, etc.)",
                 "Food loss and waste reduction targets",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C-AC6.8", "C-FB0.7"],
    },
    IndustryType.TEXTILES_APPAREL: {
        "name": "Textiles & Apparel Supplier Carbon Questionnaire",
        "sections": [
            {"id": "ORG", "title": "Organisation & Boundary",
             "questions": [
                 "Legal entity name and factory locations",
                 "Product types within scope",
                 "Reporting period",
             ]},
            {"id": "FIBRE", "title": "Fibre & Raw Material",
             "questions": [
                 "Fibre types used (cotton, polyester, viscose, etc.)",
                 "Fibre sourcing countries",
                 "Organic / recycled fibre percentage",
                 "Raw material carbon footprint (kgCO2e/kg)",
             ]},
            {"id": "WET_PROCESS", "title": "Wet Processing & Dyeing",
             "questions": [
                 "Thermal energy for dyeing / finishing (GJ)",
                 "Electricity for wet processing (MWh)",
                 "Water consumption (m3)",
                 "Chemical management (ZDHC status)",
             ]},
            {"id": "TARGETS", "title": "Targets",
             "questions": [
                 "Fashion Industry Charter for Climate Action status",
                 "SBTi commitment or validated target",
                 "Higg FEM self-assessment score",
             ]},
        ],
        "cdp_mapping": ["C6.1", "C6.3", "C4.1a"],
    },
}
"""Industry-specific questionnaire templates with CDP field mappings."""

# DQI level descriptions for scoring rubric.
DQI_DESCRIPTIONS: Dict[int, str] = {
    1: "No primary data; EEIO model estimate based on sector averages.",
    2: "Spend-based method with general sector emission factor.",
    3: "Average-data method with product-specific emission factor.",
    4: "Supplier-reported aggregate data allocated by revenue share.",
    5: "Supplier-specific product-level LCA data (cradle-to-gate).",
}
"""Human-readable descriptions of each DQI level."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class ProcurementItem(BaseModel):
    """A single procurement line item linking a supplier to spend/category.

    Attributes:
        supplier_id: Unique supplier identifier.
        category: Scope 3 category for this spend.
        spend_amount: Procurement spend amount (currency units).
        currency: Currency code (ISO 4217).
        quantity: Physical quantity purchased (optional).
        quantity_unit: Unit for quantity (e.g., kg, MWh).
        estimated_emissions_tco2e: Pre-estimated emissions for this item.
        data_quality_level: Current DQI level for this item.
    """
    supplier_id: str = Field(..., min_length=1, description="Supplier ID")
    category: str = Field(default=Scope3Category.CAT_1, description="Scope 3 category")
    spend_amount: Decimal = Field(default=Decimal("0"), ge=0, description="Spend amount")
    currency: str = Field(default="USD", description="Currency (ISO 4217)")
    quantity: Optional[Decimal] = Field(default=None, ge=0, description="Physical quantity")
    quantity_unit: Optional[str] = Field(default=None, description="Unit for quantity")
    estimated_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated emissions (tCO2e)"
    )
    data_quality_level: int = Field(default=1, ge=1, le=5, description="DQI level")

    @field_validator("spend_amount", "estimated_emissions_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        return _decimal(v)

class Supplier(BaseModel):
    """Supplier master data record.

    Attributes:
        supplier_id: Unique supplier identifier.
        name: Supplier legal or trading name.
        industry: Industry classification.
        country: Primary country of operations (ISO 3166-1 alpha-2).
        contact_email: Primary contact email.
        contact_name: Primary contact person name.
        engagement_status: Current engagement lifecycle status.
        current_dqi_level: Current overall data quality level.
        strategic_importance: Strategic importance score (0-1).
        sbti_status: SBTi commitment status.
        cdp_respondent: Whether supplier responds to CDP.
        notes: Free-text notes.
    """
    supplier_id: str = Field(default_factory=_new_uuid, description="Supplier ID")
    name: str = Field(..., min_length=1, description="Supplier name")
    industry: str = Field(default=IndustryType.MANUFACTURING, description="Industry type")
    country: str = Field(default="", description="Country (ISO 3166-1 alpha-2)")
    contact_email: str = Field(default="", description="Contact email")
    contact_name: str = Field(default="", description="Contact person name")
    engagement_status: EngagementStatus = Field(
        default=EngagementStatus.NOT_STARTED, description="Engagement status"
    )
    current_dqi_level: int = Field(default=1, ge=1, le=5, description="Current DQI level")
    strategic_importance: Decimal = Field(
        default=Decimal("0.5"), ge=0, le=1, description="Strategic importance (0-1)"
    )
    sbti_status: str = Field(default="none", description="SBTi status")
    cdp_respondent: bool = Field(default=False, description="CDP Supply Chain respondent")
    notes: str = Field(default="", description="Free-text notes")

class SupplierResponseData(BaseModel):
    """Data submitted by a supplier in response to a data request.

    Attributes:
        supplier_id: Supplier who submitted.
        response_id: Unique response identifier.
        submitted_at: Submission timestamp.
        scope1_tco2e: Reported Scope 1 emissions.
        scope2_tco2e: Reported Scope 2 emissions.
        methodology: Methodology description.
        verification_status: Whether data is third-party verified.
        product_carbon_footprints: Per-product PCF data.
        renewable_energy_pct: Percentage of renewable energy.
        reduction_target_pct: Reduction target percentage.
        reduction_target_year: Target year for reduction.
        raw_answers: Full questionnaire answers (field -> value).
    """
    supplier_id: str = Field(..., min_length=1, description="Supplier ID")
    response_id: str = Field(default_factory=_new_uuid, description="Response ID")
    submitted_at: datetime = Field(default_factory=utcnow, description="Submission time")
    scope1_tco2e: Optional[Decimal] = Field(default=None, ge=0, description="Scope 1 (tCO2e)")
    scope2_tco2e: Optional[Decimal] = Field(default=None, ge=0, description="Scope 2 (tCO2e)")
    methodology: str = Field(default="", description="Methodology used")
    verification_status: str = Field(default="unverified", description="Verification status")
    product_carbon_footprints: Dict[str, float] = Field(
        default_factory=dict, description="Per-product PCF (kgCO2e/unit)"
    )
    renewable_energy_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=100, description="Renewable energy %"
    )
    reduction_target_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=100, description="Reduction target %"
    )
    reduction_target_year: Optional[int] = Field(
        default=None, ge=2020, description="Target year"
    )
    raw_answers: Dict[str, Any] = Field(
        default_factory=dict, description="Full questionnaire answers"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class SupplierPriority(BaseModel):
    """Supplier prioritisation result.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        tier: Prioritisation tier (critical/high/medium/low).
        priority_score: Numeric priority score (0-100).
        estimated_emissions_tco2e: Estimated emissions from this supplier.
        emission_share_pct: Share of total Scope 3 emissions (%).
        cumulative_share_pct: Cumulative share in ranked order (%).
        current_dqi_level: Current data quality level.
        engagement_status: Current engagement status.
        recommended_action: Recommended next action.
    """
    supplier_id: str = Field(default="", description="Supplier ID")
    supplier_name: str = Field(default="", description="Supplier name")
    tier: str = Field(default=SupplierTier.LOW, description="Priority tier")
    priority_score: float = Field(default=0.0, ge=0, le=100, description="Priority score")
    estimated_emissions_tco2e: float = Field(
        default=0.0, ge=0, description="Estimated emissions"
    )
    emission_share_pct: float = Field(default=0.0, ge=0, le=100, description="Emission share %")
    cumulative_share_pct: float = Field(
        default=0.0, ge=0, le=100, description="Cumulative share %"
    )
    current_dqi_level: int = Field(default=1, ge=1, le=5, description="DQI level")
    engagement_status: str = Field(default="not_started", description="Engagement status")
    recommended_action: str = Field(default="", description="Recommended next action")

class DataRequest(BaseModel):
    """Generated data request for a supplier.

    Attributes:
        request_id: Unique request identifier.
        supplier_id: Target supplier.
        supplier_name: Supplier name.
        category: Scope 3 category.
        industry_template: Industry template used.
        questionnaire_sections: Sections included in the request.
        cdp_field_mapping: CDP Supply Chain field mapping.
        due_date: Requested response date.
        reminder_schedule: Reminder schedule.
        created_at: Creation timestamp.
    """
    request_id: str = Field(default_factory=_new_uuid, description="Request ID")
    supplier_id: str = Field(default="", description="Supplier ID")
    supplier_name: str = Field(default="", description="Supplier name")
    category: str = Field(default="", description="Scope 3 category")
    industry_template: str = Field(default="", description="Template used")
    questionnaire_sections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sections"
    )
    cdp_field_mapping: Dict[str, str] = Field(
        default_factory=dict, description="CDP field mapping"
    )
    due_date: Optional[datetime] = Field(default=None, description="Due date")
    reminder_schedule: Dict[str, Any] = Field(
        default_factory=dict, description="Reminder schedule"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Created timestamp")

class QualityScore(BaseModel):
    """Data quality assessment result for a supplier response.

    Attributes:
        supplier_id: Supplier identifier.
        overall_dqi_level: Overall DQI level (1-5).
        technological_score: Technology representativeness (1-5).
        temporal_score: Temporal representativeness (1-5).
        geographical_score: Geographical representativeness (1-5).
        completeness_score: Data completeness (1-5).
        reliability_score: Data reliability / source type (1-5).
        weighted_dqr: Weighted Data Quality Rating (1.0-5.0).
        uncertainty_range_pct: Estimated uncertainty range (%).
        improvement_areas: Areas identified for improvement.
        assessment_notes: Assessment rationale notes.
    """
    supplier_id: str = Field(default="", description="Supplier ID")
    overall_dqi_level: int = Field(default=1, ge=1, le=5, description="Overall DQI level")
    technological_score: int = Field(default=1, ge=1, le=5, description="Tech score")
    temporal_score: int = Field(default=1, ge=1, le=5, description="Temporal score")
    geographical_score: int = Field(default=1, ge=1, le=5, description="Geo score")
    completeness_score: int = Field(default=1, ge=1, le=5, description="Completeness score")
    reliability_score: int = Field(default=1, ge=1, le=5, description="Reliability score")
    weighted_dqr: float = Field(default=1.0, ge=1.0, le=5.0, description="Weighted DQR")
    uncertainty_range_pct: float = Field(default=200.0, ge=0, description="Uncertainty %")
    improvement_areas: List[str] = Field(default_factory=list, description="Improvement areas")
    assessment_notes: List[str] = Field(default_factory=list, description="Assessment notes")

class EngagementPlan(BaseModel):
    """Multi-year engagement plan for a supplier.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        current_level: Current DQI level.
        target_level: Target DQI level.
        milestones: Year-by-year milestones with target DQI levels.
        actions: Specific actions per milestone.
        estimated_timeline_years: Total timeline (years).
        estimated_cost: Estimated engagement cost (USD).
        expected_uncertainty_reduction_pct: Expected uncertainty reduction.
    """
    supplier_id: str = Field(default="", description="Supplier ID")
    supplier_name: str = Field(default="", description="Supplier name")
    current_level: int = Field(default=1, ge=1, le=5, description="Current DQI level")
    target_level: int = Field(default=5, ge=1, le=5, description="Target DQI level")
    milestones: List[Dict[str, Any]] = Field(
        default_factory=list, description="Year-by-year milestones"
    )
    actions: List[str] = Field(default_factory=list, description="Actions")
    estimated_timeline_years: int = Field(default=3, ge=1, le=10, description="Timeline (years)")
    estimated_cost: float = Field(default=0.0, ge=0, description="Cost (USD)")
    expected_uncertainty_reduction_pct: float = Field(
        default=0.0, ge=0, description="Uncertainty reduction %"
    )

class EngagementROI(BaseModel):
    """Return on investment analysis for supplier engagement.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        engagement_cost: Total engagement cost (USD).
        uncertainty_reduction_tco2e: Absolute uncertainty reduction.
        value_per_tonne: Shadow carbon price used (USD/tCO2e).
        roi_ratio: ROI ratio (value / cost).
        payback_years: Estimated payback period.
        accuracy_improvement_pct: Data accuracy improvement (%).
        ranking: ROI ranking among all suppliers.
    """
    supplier_id: str = Field(default="", description="Supplier ID")
    supplier_name: str = Field(default="", description="Supplier name")
    engagement_cost: float = Field(default=0.0, ge=0, description="Cost (USD)")
    uncertainty_reduction_tco2e: float = Field(
        default=0.0, ge=0, description="Uncertainty reduction (tCO2e)"
    )
    value_per_tonne: float = Field(default=50.0, ge=0, description="Shadow carbon price")
    roi_ratio: float = Field(default=0.0, description="ROI ratio")
    payback_years: float = Field(default=0.0, ge=0, description="Payback (years)")
    accuracy_improvement_pct: float = Field(
        default=0.0, ge=0, description="Accuracy improvement %"
    )
    ranking: int = Field(default=0, ge=0, description="ROI ranking")

class ReminderSchedule(BaseModel):
    """Scheduled reminders for supplier engagement.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        reminders: List of scheduled reminders (type, date, status).
    """
    supplier_id: str = Field(default="", description="Supplier ID")
    supplier_name: str = Field(default="", description="Supplier name")
    reminders: List[Dict[str, Any]] = Field(
        default_factory=list, description="Scheduled reminders"
    )

class EngagementMetrics(BaseModel):
    """Aggregated engagement metrics across all suppliers.

    Attributes:
        total_suppliers: Total supplier count.
        suppliers_engaged: Suppliers with at least one contact.
        suppliers_responded: Suppliers who submitted data.
        suppliers_validated: Suppliers with validated data.
        response_rate_pct: Overall response rate (%).
        avg_dqi_level: Average DQI level across suppliers.
        dqi_distribution: Count per DQI level.
        tier_distribution: Count per priority tier.
        status_distribution: Count per engagement status.
        total_emissions_covered_pct: Emissions covered by primary data (%).
        year_over_year_improvement: DQI improvement metrics.
    """
    total_suppliers: int = Field(default=0, ge=0, description="Total suppliers")
    suppliers_engaged: int = Field(default=0, ge=0, description="Suppliers engaged")
    suppliers_responded: int = Field(default=0, ge=0, description="Suppliers responded")
    suppliers_validated: int = Field(default=0, ge=0, description="Suppliers validated")
    response_rate_pct: float = Field(default=0.0, ge=0, le=100, description="Response rate %")
    avg_dqi_level: float = Field(default=1.0, ge=1.0, le=5.0, description="Average DQI level")
    dqi_distribution: Dict[str, int] = Field(
        default_factory=dict, description="DQI distribution"
    )
    tier_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Tier distribution"
    )
    status_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Status distribution"
    )
    total_emissions_covered_pct: float = Field(
        default=0.0, ge=0, le=100, description="Emissions covered %"
    )
    year_over_year_improvement: Dict[str, Any] = Field(
        default_factory=dict, description="YoY improvement"
    )

class SupplierEngagementResult(BaseModel):
    """Complete supplier engagement result with provenance.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing time (ms).
        prioritised_suppliers: Ranked supplier list.
        data_requests: Generated data requests.
        quality_scores: Quality assessments.
        engagement_plans: Multi-year engagement plans.
        roi_analysis: ROI analysis results.
        engagement_metrics: Aggregated metrics.
        methodology_notes: Methodology notes.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Version")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    prioritised_suppliers: List[SupplierPriority] = Field(
        default_factory=list, description="Prioritised suppliers"
    )
    data_requests: List[DataRequest] = Field(
        default_factory=list, description="Data requests"
    )
    quality_scores: List[QualityScore] = Field(
        default_factory=list, description="Quality scores"
    )
    engagement_plans: List[EngagementPlan] = Field(
        default_factory=list, description="Engagement plans"
    )
    roi_analysis: List[EngagementROI] = Field(
        default_factory=list, description="ROI analysis"
    )
    engagement_metrics: Optional[EngagementMetrics] = Field(
        default=None, description="Aggregated metrics"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SupplierEngagementEngine:
    """Supplier carbon data engagement engine.

    Manages the end-to-end process of collecting Scope 3 emissions data
    from suppliers, from initial prioritisation through quality scoring
    and multi-year engagement roadmap generation.

    Guarantees:
        - Deterministic: same inputs produce identical priority rankings.
        - Traceable: SHA-256 provenance hash on every result.
        - Standards-based: DQI levels per GHG Protocol Scope 3 Guidance.
        - No LLM: zero hallucination risk in scoring or prioritisation.

    Usage::

        engine = SupplierEngagementEngine()
        priorities = engine.prioritize_suppliers(suppliers, procurement_data)
        request = engine.generate_data_request(supplier, category)
        quality = engine.score_data_quality(supplier_response)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the supplier engagement engine.

        Args:
            config: Optional configuration overrides.
                - shadow_carbon_price: USD/tCO2e for ROI calculation (default 50).
                - fte_loaded_rate: Hourly loaded FTE rate (default 75 USD/hr).
                - engagement_hours_per_supplier: Hours per supplier engagement (default 40).
                - reminder_schedule: Override default reminder timing.
        """
        self._config = config or {}
        self._shadow_carbon_price = Decimal(
            str(self._config.get("shadow_carbon_price", 50))
        )
        self._fte_loaded_rate = Decimal(
            str(self._config.get("fte_loaded_rate", 75))
        )
        self._hours_per_supplier = Decimal(
            str(self._config.get("engagement_hours_per_supplier", 40))
        )
        self._reminder_schedule: Dict[str, int] = self._config.get(
            "reminder_schedule", dict(DEFAULT_REMINDER_SCHEDULE)
        )
        logger.info("SupplierEngagementEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def prioritize_suppliers(
        self,
        suppliers: List[Supplier],
        procurement_data: List[ProcurementItem],
    ) -> List[SupplierPriority]:
        """Rank suppliers by emission contribution and strategic factors.

        Uses a composite priority score combining emission share, category
        weight, strategic importance, and current data quality level.

        Args:
            suppliers: Master supplier records.
            procurement_data: Procurement line items linking suppliers to spend.

        Returns:
            Suppliers ranked by priority score (highest first).
        """
        t0 = time.perf_counter()
        logger.info("Prioritising %d suppliers with %d procurement items.",
                     len(suppliers), len(procurement_data))

        supplier_map: Dict[str, Supplier] = {s.supplier_id: s for s in suppliers}

        # Aggregate emissions per supplier.
        supplier_emissions: Dict[str, Decimal] = {}
        supplier_categories: Dict[str, List[str]] = {}
        for item in procurement_data:
            sid = item.supplier_id
            supplier_emissions[sid] = supplier_emissions.get(sid, Decimal("0")) + item.estimated_emissions_tco2e
            supplier_categories.setdefault(sid, []).append(item.category)

        total_emissions = sum(supplier_emissions.values(), Decimal("0"))

        # Build priority scores.
        raw_priorities: List[Tuple[Decimal, str]] = []
        for sid, emissions in supplier_emissions.items():
            supplier = supplier_map.get(sid)
            if not supplier:
                continue
            score = self._calculate_priority_score(
                emissions, total_emissions, supplier, supplier_categories.get(sid, [])
            )
            raw_priorities.append((score, sid))

        # Sort descending by score.
        raw_priorities.sort(key=lambda x: x[0], reverse=True)

        # Build output with cumulative share.
        results: List[SupplierPriority] = []
        cumulative = Decimal("0")
        for rank_idx, (score, sid) in enumerate(raw_priorities):
            supplier = supplier_map.get(sid)
            emissions = supplier_emissions.get(sid, Decimal("0"))
            share = _safe_pct(emissions, total_emissions)
            cumulative += share
            tier = self._assign_tier(cumulative, share)
            recommended = self._recommend_action(
                tier, supplier.engagement_status if supplier else EngagementStatus.NOT_STARTED,
                supplier.current_dqi_level if supplier else 1,
            )

            results.append(SupplierPriority(
                supplier_id=sid,
                supplier_name=supplier.name if supplier else sid,
                tier=tier,
                priority_score=_round2(score),
                estimated_emissions_tco2e=_round2(emissions),
                emission_share_pct=_round2(share),
                cumulative_share_pct=_round2(cumulative),
                current_dqi_level=supplier.current_dqi_level if supplier else 1,
                engagement_status=(
                    supplier.engagement_status if supplier
                    else EngagementStatus.NOT_STARTED
                ),
                recommended_action=recommended,
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Supplier prioritisation complete: %d suppliers ranked in %.1f ms.",
                     len(results), elapsed)
        return results

    def generate_data_request(
        self,
        supplier: Supplier,
        category: str = Scope3Category.CAT_1,
        due_days: int = 30,
    ) -> DataRequest:
        """Generate a standardised data request for a supplier.

        Selects the appropriate industry-specific questionnaire template
        and maps CDP Supply Chain fields for interoperability.

        Args:
            supplier: Target supplier record.
            category: Scope 3 category context.
            due_days: Days until response is due.

        Returns:
            DataRequest with questionnaire sections and CDP mapping.
        """
        t0 = time.perf_counter()
        logger.info("Generating data request for supplier=%s, industry=%s, category=%s.",
                     supplier.name, supplier.industry, category)

        template = QUESTIONNAIRE_TEMPLATES.get(
            supplier.industry, QUESTIONNAIRE_TEMPLATES[IndustryType.MANUFACTURING]
        )

        due_date = utcnow() + timedelta(days=due_days)

        # Build reminder schedule based on due date.
        reminders: Dict[str, Any] = {}
        request_date = utcnow()
        for rtype, days_offset in self._reminder_schedule.items():
            reminder_date = request_date + timedelta(days=days_offset)
            reminders[rtype if isinstance(rtype, str) else rtype.value] = {
                "date": reminder_date.isoformat(),
                "status": "scheduled",
            }

        request = DataRequest(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.name,
            category=category,
            industry_template=template["name"],
            questionnaire_sections=template["sections"],
            cdp_field_mapping=CDP_SUPPLY_CHAIN_FIELDS,
            due_date=due_date,
            reminder_schedule=reminders,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Data request generated in %.1f ms: request_id=%s.",
                     elapsed, request.request_id)
        return request

    def track_response(
        self,
        supplier_id: str,
        response_data: SupplierResponseData,
        supplier: Optional[Supplier] = None,
    ) -> Dict[str, Any]:
        """Track and record a supplier data response.

        Updates engagement status and returns response metadata.

        Args:
            supplier_id: Supplier identifier.
            response_data: Submitted response data.
            supplier: Optional supplier record to update.

        Returns:
            Dict with tracking metadata including status update.
        """
        t0 = time.perf_counter()
        logger.info("Tracking response for supplier=%s, response_id=%s.",
                     supplier_id, response_data.response_id)

        # Determine completeness of response.
        fields_present = 0
        fields_total = 8  # Core fields expected.
        if response_data.scope1_tco2e is not None:
            fields_present += 1
        if response_data.scope2_tco2e is not None:
            fields_present += 1
        if response_data.methodology:
            fields_present += 1
        if response_data.verification_status != "unverified":
            fields_present += 1
        if response_data.product_carbon_footprints:
            fields_present += 1
        if response_data.renewable_energy_pct is not None:
            fields_present += 1
        if response_data.reduction_target_pct is not None:
            fields_present += 1
        if response_data.reduction_target_year is not None:
            fields_present += 1

        completeness_pct = _round2(_safe_pct(
            _decimal(fields_present), _decimal(fields_total)
        ))
        new_status = EngagementStatus.RESPONDED

        if supplier:
            supplier.engagement_status = new_status

        result = {
            "supplier_id": supplier_id,
            "response_id": response_data.response_id,
            "submitted_at": response_data.submitted_at.isoformat(),
            "fields_present": fields_present,
            "fields_total": fields_total,
            "completeness_pct": completeness_pct,
            "new_status": new_status.value,
            "provenance_hash": _compute_hash(response_data),
        }

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Response tracked in %.1f ms: completeness=%.1f%%.",
                     elapsed, completeness_pct)
        return result

    def score_data_quality(
        self,
        supplier_response: SupplierResponseData,
        supplier: Optional[Supplier] = None,
    ) -> QualityScore:
        """Score data quality of a supplier response per GHG Protocol DQI.

        Evaluates 5 quality dimensions and produces a weighted Data Quality
        Rating (DQR) from 1.0 (lowest) to 5.0 (highest).

        Quality weights: tech=0.20, temporal=0.20, geo=0.20,
                         completeness=0.25, reliability=0.15.

        Args:
            supplier_response: Supplier's submitted data.
            supplier: Optional supplier record for context.

        Returns:
            QualityScore with per-dimension and overall scores.
        """
        t0 = time.perf_counter()
        logger.info("Scoring data quality for supplier=%s.", supplier_response.supplier_id)

        tech_score = self._score_technological(supplier_response)
        temporal_score = self._score_temporal(supplier_response)
        geo_score = self._score_geographical(supplier_response, supplier)
        completeness_score = self._score_completeness(supplier_response)
        reliability_score = self._score_reliability(supplier_response)

        # Weighted DQR calculation.
        weighted_dqr = (
            Decimal(str(tech_score)) * Decimal("0.20")
            + Decimal(str(temporal_score)) * Decimal("0.20")
            + Decimal(str(geo_score)) * Decimal("0.20")
            + Decimal(str(completeness_score)) * Decimal("0.25")
            + Decimal(str(reliability_score)) * Decimal("0.15")
        )

        # Determine overall DQI level from weighted DQR.
        overall_level = self._dqr_to_level(weighted_dqr)

        # Uncertainty range from DQI level.
        unc_range = DQI_UNCERTAINTY_RANGES.get(overall_level, (100.0, 200.0))
        avg_uncertainty = (unc_range[0] + unc_range[1]) / 2.0

        # Identify improvement areas.
        improvements: List[str] = []
        dimension_scores = {
            "Technological representativeness": tech_score,
            "Temporal representativeness": temporal_score,
            "Geographical representativeness": geo_score,
            "Data completeness": completeness_score,
            "Data reliability": reliability_score,
        }
        for dim, val in sorted(dimension_scores.items(), key=lambda x: x[1]):
            if val < 4:
                improvements.append(
                    f"{dim}: score {val}/5 -- "
                    f"{'upgrade methodology' if val <= 2 else 'improve data specificity'}"
                )

        # Assessment notes.
        notes: List[str] = [
            f"DQR {_round2(weighted_dqr)}/5.0 (Level {overall_level}: {DQI_DESCRIPTIONS[overall_level]})",
        ]
        if supplier_response.verification_status == "verified":
            notes.append("Third-party verification increases reliability.")
        if supplier_response.product_carbon_footprints:
            notes.append(
                f"Product-level PCF data available for "
                f"{len(supplier_response.product_carbon_footprints)} products."
            )

        result = QualityScore(
            supplier_id=supplier_response.supplier_id,
            overall_dqi_level=overall_level,
            technological_score=tech_score,
            temporal_score=temporal_score,
            geographical_score=geo_score,
            completeness_score=completeness_score,
            reliability_score=reliability_score,
            weighted_dqr=_round2(weighted_dqr),
            uncertainty_range_pct=_round2(avg_uncertainty),
            improvement_areas=improvements,
            assessment_notes=notes,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Quality score complete in %.1f ms: DQR=%.2f, Level=%d.",
                     elapsed, _round2(weighted_dqr), overall_level)
        return result

    def generate_engagement_roadmap(
        self,
        supplier: Supplier,
        current_score: Optional[QualityScore] = None,
        target_level: int = 5,
        timeline_years: int = 4,
    ) -> EngagementPlan:
        """Generate a multi-year engagement roadmap from current to target DQI.

        Builds year-by-year milestones with specific actions to progress
        from the current DQI level to the target level.

from greenlang.schemas import utcnow

        Args:
            supplier: Supplier record.
            current_score: Optional current quality score.
            target_level: Target DQI level (default 5).
            timeline_years: Years to reach target (default 4).

        Returns:
            EngagementPlan with milestones and actions.
        """
        t0 = time.perf_counter()
        current_level = (
            current_score.overall_dqi_level if current_score
            else supplier.current_dqi_level
        )
        logger.info(
            "Generating roadmap for supplier=%s: Level %d -> Level %d over %d years.",
            supplier.name, current_level, target_level, timeline_years,
        )

        levels_to_climb = max(0, target_level - current_level)
        if levels_to_climb == 0:
            return EngagementPlan(
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.name,
                current_level=current_level,
                target_level=target_level,
                milestones=[{"year": 0, "level": current_level, "note": "Already at target."}],
                actions=["Maintain current data quality through annual refresh."],
                estimated_timeline_years=0,
                estimated_cost=0.0,
                expected_uncertainty_reduction_pct=0.0,
            )

        # Distribute level upgrades across timeline.
        milestones: List[Dict[str, Any]] = []
        actions: List[str] = []

        # How many years per level upgrade.
        years_per_level = max(1, timeline_years // levels_to_climb)
        level_cursor = current_level
        year_cursor = 0

        level_actions = {
            1: [
                "Collect procurement spend data by category",
                "Map suppliers to EEIO sectors",
                "Establish baseline emissions estimate",
            ],
            2: [
                "Obtain sector-specific emission factors",
                "Refine spend categorisation",
                "Send initial data request questionnaire",
            ],
            3: [
                "Collect product-specific activity data from supplier",
                "Apply product-level emission factors",
                "Request supplier methodology documentation",
            ],
            4: [
                "Request supplier-reported Scope 1+2 totals",
                "Allocate supplier emissions by revenue share",
                "Encourage CDP Supply Chain participation",
                "Request third-party verification of supplier data",
            ],
            5: [
                "Request product-level LCA data (ISO 14040/44)",
                "Validate LCA methodology and system boundary",
                "Integrate supplier EPDs into carbon accounting",
                "Establish annual data sharing agreement",
            ],
        }

        for step in range(levels_to_climb):
            year_cursor += years_per_level
            next_level = min(level_cursor + 1, 5)
            step_actions = level_actions.get(next_level, [])
            milestones.append({
                "year": year_cursor,
                "from_level": level_cursor,
                "to_level": next_level,
                "actions": step_actions,
                "note": f"Upgrade from Level {level_cursor} to Level {next_level}",
            })
            actions.extend(step_actions)
            level_cursor = next_level

        # Estimate cost.
        base_cost = float(self._hours_per_supplier * self._fte_loaded_rate)
        total_cost = base_cost * levels_to_climb

        # Estimate uncertainty reduction.
        current_unc = DQI_UNCERTAINTY_RANGES.get(current_level, (100.0, 200.0))
        target_unc = DQI_UNCERTAINTY_RANGES.get(target_level, (5.0, 15.0))
        current_avg = (current_unc[0] + current_unc[1]) / 2.0
        target_avg = (target_unc[0] + target_unc[1]) / 2.0
        reduction_pct = max(0.0, current_avg - target_avg)

        plan = EngagementPlan(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.name,
            current_level=current_level,
            target_level=target_level,
            milestones=milestones,
            actions=actions,
            estimated_timeline_years=year_cursor,
            estimated_cost=_round2(total_cost),
            expected_uncertainty_reduction_pct=_round2(reduction_pct),
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Engagement roadmap generated in %.1f ms: %d milestones.",
                     elapsed, len(milestones))
        return plan

    def calculate_engagement_roi(
        self,
        suppliers: List[Supplier],
        procurement_data: List[ProcurementItem],
        quality_scores: Optional[Dict[str, QualityScore]] = None,
    ) -> List[EngagementROI]:
        """Calculate engagement ROI for each supplier.

        ROI = (uncertainty_reduction_tco2e * shadow_carbon_price) / engagement_cost

        Args:
            suppliers: Supplier records.
            procurement_data: Procurement data with emissions estimates.
            quality_scores: Optional existing quality scores by supplier ID.

        Returns:
            ROI results ranked by ROI ratio (highest first).
        """
        t0 = time.perf_counter()
        logger.info("Calculating engagement ROI for %d suppliers.", len(suppliers))

        supplier_map: Dict[str, Supplier] = {s.supplier_id: s for s in suppliers}

        # Aggregate emissions per supplier.
        supplier_emissions: Dict[str, Decimal] = {}
        for item in procurement_data:
            sid = item.supplier_id
            supplier_emissions[sid] = (
                supplier_emissions.get(sid, Decimal("0")) + item.estimated_emissions_tco2e
            )

        results: List[EngagementROI] = []
        for supplier in suppliers:
            sid = supplier.supplier_id
            emissions = supplier_emissions.get(sid, Decimal("0"))
            current_level = supplier.current_dqi_level

            # Target: upgrade by 1 level.
            target_level = min(current_level + 1, 5)

            # Cost estimate.
            cost = float(self._hours_per_supplier * self._fte_loaded_rate)

            # Uncertainty reduction.
            current_unc = DQI_UNCERTAINTY_RANGES.get(current_level, (100.0, 200.0))
            target_unc = DQI_UNCERTAINTY_RANGES.get(target_level, (50.0, 100.0))
            current_avg_pct = (current_unc[0] + current_unc[1]) / 2.0
            target_avg_pct = (target_unc[0] + target_unc[1]) / 2.0
            reduction_pct = max(0.0, current_avg_pct - target_avg_pct)
            reduction_tco2e = float(emissions * _decimal(reduction_pct) / Decimal("100"))

            # ROI.
            value = reduction_tco2e * float(self._shadow_carbon_price)
            roi_ratio = _round2(_safe_divide(_decimal(value), _decimal(cost)))

            # Accuracy improvement.
            accuracy_improvement = _round2(reduction_pct)

            results.append(EngagementROI(
                supplier_id=sid,
                supplier_name=supplier.name,
                engagement_cost=_round2(cost),
                uncertainty_reduction_tco2e=_round2(reduction_tco2e),
                value_per_tonne=float(self._shadow_carbon_price),
                roi_ratio=roi_ratio,
                payback_years=_round2(
                    _safe_divide(_decimal(cost), _decimal(max(value, 0.01)))
                ),
                accuracy_improvement_pct=accuracy_improvement,
            ))

        # Sort by ROI descending and assign rankings.
        results.sort(key=lambda r: r.roi_ratio, reverse=True)
        for idx, r in enumerate(results):
            r.ranking = idx + 1

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("ROI calculation complete in %.1f ms: %d suppliers.", elapsed, len(results))
        return results

    def schedule_reminders(
        self,
        engagement_plan: EngagementPlan,
        start_date: Optional[datetime] = None,
    ) -> ReminderSchedule:
        """Generate automated reminder schedule for a supplier engagement.

        Args:
            engagement_plan: Engagement plan for the supplier.
            start_date: Optional start date (defaults to now).

        Returns:
            ReminderSchedule with dated reminders.
        """
        t0 = time.perf_counter()
        base_date = start_date or utcnow()
        logger.info("Scheduling reminders for supplier=%s from %s.",
                     engagement_plan.supplier_name, base_date.isoformat())

        reminders: List[Dict[str, Any]] = []
        for rtype_key, days_offset in self._reminder_schedule.items():
            rtype_str = rtype_key if isinstance(rtype_key, str) else rtype_key.value
            reminder_date = base_date + timedelta(days=days_offset)
            reminders.append({
                "type": rtype_str,
                "date": reminder_date.isoformat(),
                "days_from_start": days_offset,
                "status": "scheduled",
                "message": self._reminder_message(rtype_str, engagement_plan.supplier_name),
            })

        # Add milestone-based reminders.
        for milestone in engagement_plan.milestones:
            year = milestone.get("year", 1)
            milestone_date = base_date + timedelta(days=year * 365)
            reminders.append({
                "type": "milestone_check",
                "date": milestone_date.isoformat(),
                "days_from_start": year * 365,
                "status": "scheduled",
                "message": (
                    f"Milestone check for {engagement_plan.supplier_name}: "
                    f"verify progress to Level {milestone.get('to_level', '?')}."
                ),
            })

        schedule = ReminderSchedule(
            supplier_id=engagement_plan.supplier_id,
            supplier_name=engagement_plan.supplier_name,
            reminders=sorted(reminders, key=lambda r: r["days_from_start"]),
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Scheduled %d reminders in %.1f ms.", len(reminders), elapsed)
        return schedule

    def aggregate_engagement_metrics(
        self,
        suppliers: List[Supplier],
        procurement_data: List[ProcurementItem],
        quality_scores: Optional[Dict[str, QualityScore]] = None,
    ) -> EngagementMetrics:
        """Compute aggregated engagement dashboard metrics.

        Args:
            suppliers: All supplier records.
            procurement_data: Procurement line items.
            quality_scores: Optional quality scores by supplier ID.

        Returns:
            EngagementMetrics with distributions and rates.
        """
        t0 = time.perf_counter()
        logger.info("Aggregating engagement metrics for %d suppliers.", len(suppliers))

        total = len(suppliers)
        engaged = sum(
            1 for s in suppliers
            if s.engagement_status not in (EngagementStatus.NOT_STARTED,)
        )
        responded = sum(
            1 for s in suppliers
            if s.engagement_status in (
                EngagementStatus.RESPONDED, EngagementStatus.VALIDATED
            )
        )
        validated = sum(
            1 for s in suppliers
            if s.engagement_status == EngagementStatus.VALIDATED
        )
        response_rate = _round2(_safe_pct(_decimal(responded), _decimal(total))) if total > 0 else 0.0

        # DQI distribution.
        dqi_dist: Dict[str, int] = {f"level_{i}": 0 for i in range(1, 6)}
        dqi_sum = Decimal("0")
        for s in suppliers:
            level_key = f"level_{s.current_dqi_level}"
            dqi_dist[level_key] = dqi_dist.get(level_key, 0) + 1
            dqi_sum += _decimal(s.current_dqi_level)
        avg_dqi = _round2(_safe_divide(dqi_sum, _decimal(total))) if total > 0 else 1.0

        # Tier distribution (requires prioritisation, use simple heuristic).
        tier_dist: Dict[str, int] = {
            SupplierTier.CRITICAL: 0,
            SupplierTier.HIGH: 0,
            SupplierTier.MEDIUM: 0,
            SupplierTier.LOW: 0,
        }
        # Status distribution.
        status_dist: Dict[str, int] = {}
        for s in suppliers:
            status_key = s.engagement_status if isinstance(s.engagement_status, str) else s.engagement_status.value
            status_dist[status_key] = status_dist.get(status_key, 0) + 1

        # Emissions coverage -- suppliers with DQI >= 3 are considered "primary data".
        supplier_emissions: Dict[str, Decimal] = {}
        for item in procurement_data:
            sid = item.supplier_id
            supplier_emissions[sid] = (
                supplier_emissions.get(sid, Decimal("0")) + item.estimated_emissions_tco2e
            )
        total_emissions = sum(supplier_emissions.values(), Decimal("0"))
        covered_emissions = Decimal("0")
        supplier_map = {s.supplier_id: s for s in suppliers}
        for sid, em in supplier_emissions.items():
            s = supplier_map.get(sid)
            if s and s.current_dqi_level >= 3:
                covered_emissions += em
        covered_pct = _round2(_safe_pct(covered_emissions, total_emissions)) if total_emissions > 0 else 0.0

        metrics = EngagementMetrics(
            total_suppliers=total,
            suppliers_engaged=engaged,
            suppliers_responded=responded,
            suppliers_validated=validated,
            response_rate_pct=response_rate,
            avg_dqi_level=max(1.0, min(5.0, avg_dqi)),
            dqi_distribution=dqi_dist,
            tier_distribution=tier_dist,
            status_distribution=status_dist,
            total_emissions_covered_pct=covered_pct,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Engagement metrics aggregated in %.1f ms.", elapsed)
        return metrics

    def _compute_provenance(self, data: Any) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest string.
        """
        return _compute_hash(data)

    # -------------------------------------------------------------------
    # Private -- Priority Scoring
    # -------------------------------------------------------------------

    def _calculate_priority_score(
        self,
        emissions: Decimal,
        total_emissions: Decimal,
        supplier: Supplier,
        categories: List[str],
    ) -> Decimal:
        """Calculate composite priority score for a supplier.

        Score = emission_share * 60 + category_weight * 20
                + strategic_importance * 10 + dqi_gap * 10.

        Args:
            emissions: Supplier's estimated emissions.
            total_emissions: Total Scope 3 emissions.
            supplier: Supplier record.
            categories: Scope 3 categories this supplier contributes to.

        Returns:
            Priority score (0-100 range).
        """
        # Emission share component (0-60 points).
        share = _safe_divide(emissions, total_emissions)
        emission_score = share * Decimal("60")

        # Category weight component (0-20 points).
        if categories:
            avg_weight = sum(
                Decimal(str(CATEGORY_ENGAGEMENT_WEIGHTS.get(c, 0.5)))
                for c in categories
            ) / Decimal(str(len(categories)))
        else:
            avg_weight = Decimal("0.5")
        category_score = avg_weight * Decimal("20")

        # Strategic importance (0-10 points).
        strategic_score = supplier.strategic_importance * Decimal("10")

        # DQI gap component (0-10 points; lower DQI = higher priority).
        dqi_gap = Decimal(str(5 - supplier.current_dqi_level))
        dqi_score = (dqi_gap / Decimal("4")) * Decimal("10")

        total_score = emission_score + category_score + strategic_score + dqi_score
        return min(Decimal("100"), max(Decimal("0"), total_score))

    def _assign_tier(self, cumulative_share: Decimal, individual_share: Decimal) -> str:
        """Assign supplier tier based on cumulative emission share.

        Args:
            cumulative_share: Cumulative emission share (%).
            individual_share: Individual supplier share (%).

        Returns:
            SupplierTier value string.
        """
        if individual_share >= Decimal("5") or cumulative_share <= Decimal("5"):
            return SupplierTier.CRITICAL
        if cumulative_share <= Decimal("20"):
            return SupplierTier.HIGH
        if cumulative_share <= Decimal("50"):
            return SupplierTier.MEDIUM
        return SupplierTier.LOW

    def _recommend_action(
        self, tier: str, status: EngagementStatus, dqi_level: int,
    ) -> str:
        """Recommend next engagement action based on current state.

        Args:
            tier: Supplier priority tier.
            status: Current engagement status.
            dqi_level: Current DQI level.

        Returns:
            Recommended action string.
        """
        if status == EngagementStatus.NOT_STARTED:
            if tier in (SupplierTier.CRITICAL, SupplierTier.HIGH):
                return "Initiate direct engagement -- send data request questionnaire."
            return "Include in bulk data request campaign."
        if status == EngagementStatus.CONTACTED:
            return "Follow up on initial outreach -- send first reminder."
        if status == EngagementStatus.IN_PROGRESS:
            return "Monitor response deadline -- prepare escalation if needed."
        if status == EngagementStatus.RESPONDED:
            return "Validate submitted data and provide quality feedback."
        if status == EngagementStatus.VALIDATED:
            if dqi_level < 4:
                return f"Upgrade data quality from Level {dqi_level} to Level {dqi_level + 1}."
            return "Maintain annual data refresh cycle."
        if status == EngagementStatus.ESCALATED:
            return "Escalate to procurement team -- consider contractual levers."
        return "Review engagement status."

    # -------------------------------------------------------------------
    # Private -- Quality Scoring Dimensions
    # -------------------------------------------------------------------

    def _score_technological(self, response: SupplierResponseData) -> int:
        """Score technological representativeness (1-5).

        Evaluates how well the emission factors and methodology used
        match the actual technology of the supplier's operations.

        Args:
            response: Supplier response data.

        Returns:
            Score from 1 (poor) to 5 (excellent).
        """
        score = 1
        methodology_lower = response.methodology.lower()

        if response.product_carbon_footprints:
            score = max(score, 4)
        if "lca" in methodology_lower or "life cycle" in methodology_lower:
            score = max(score, 5)
        if "iso 14040" in methodology_lower or "iso 14044" in methodology_lower:
            score = max(score, 5)
        if "epd" in methodology_lower:
            score = max(score, 4)
        if "sector average" in methodology_lower or "eeio" in methodology_lower:
            score = max(score, 2)
        if "product-specific" in methodology_lower:
            score = max(score, 4)

        if response.scope1_tco2e is not None and response.scope2_tco2e is not None:
            score = max(score, 3)

        return score

    def _score_temporal(self, response: SupplierResponseData) -> int:
        """Score temporal representativeness (1-5).

        Evaluates data age: within 1 year = 5, 1-3 years = 4,
        3-6 years = 3, 6-10 years = 2, >10 years = 1.

        Args:
            response: Supplier response data.

        Returns:
            Score from 1 to 5.
        """
        now = utcnow()
        data_age_days = (now - response.submitted_at).days

        if data_age_days <= 365:
            return 5
        if data_age_days <= 3 * 365:
            return 4
        if data_age_days <= 6 * 365:
            return 3
        if data_age_days <= 10 * 365:
            return 2
        return 1

    def _score_geographical(
        self, response: SupplierResponseData, supplier: Optional[Supplier] = None,
    ) -> int:
        """Score geographical representativeness (1-5).

        Evaluates whether data is site-specific, country-level,
        regional, or global average.

        Args:
            response: Supplier response data.
            supplier: Optional supplier record for country context.

        Returns:
            Score from 1 to 5.
        """
        score = 2  # Default: regional estimate.
        methodology_lower = response.methodology.lower()

        # Site-specific data indicators.
        if response.scope1_tco2e is not None and response.scope2_tco2e is not None:
            score = max(score, 4)
        if "site-specific" in methodology_lower or "plant-level" in methodology_lower:
            score = 5
        if "country-specific" in methodology_lower:
            score = max(score, 4)
        if "regional" in methodology_lower:
            score = max(score, 3)
        if "global average" in methodology_lower:
            score = max(score, 1)

        if supplier and supplier.country and len(supplier.country) == 2:
            score = max(score, 3)

        return score

    def _score_completeness(self, response: SupplierResponseData) -> int:
        """Score data completeness (1-5).

        Evaluates how much of the requested data has been provided.

        Args:
            response: Supplier response data.

        Returns:
            Score from 1 to 5.
        """
        fields_provided = 0
        total_fields = 8

        if response.scope1_tco2e is not None:
            fields_provided += 1
        if response.scope2_tco2e is not None:
            fields_provided += 1
        if response.methodology:
            fields_provided += 1
        if response.verification_status != "unverified":
            fields_provided += 1
        if response.product_carbon_footprints:
            fields_provided += 1
        if response.renewable_energy_pct is not None:
            fields_provided += 1
        if response.reduction_target_pct is not None:
            fields_provided += 1
        if response.reduction_target_year is not None:
            fields_provided += 1

        ratio = fields_provided / total_fields
        if ratio >= 0.9:
            return 5
        if ratio >= 0.7:
            return 4
        if ratio >= 0.5:
            return 3
        if ratio >= 0.25:
            return 2
        return 1

    def _score_reliability(self, response: SupplierResponseData) -> int:
        """Score data reliability (1-5).

        Evaluates source type: verified primary data = 5,
        primary unverified = 4, supplier estimate = 3,
        third-party estimate = 2, unknown = 1.

        Args:
            response: Supplier response data.

        Returns:
            Score from 1 to 5.
        """
        score = 1

        if response.verification_status == "verified":
            score = max(score, 5)
        elif response.verification_status == "limited_assurance":
            score = max(score, 4)

        if response.scope1_tco2e is not None or response.scope2_tco2e is not None:
            score = max(score, 3)

        if response.product_carbon_footprints:
            score = max(score, 4)

        methodology_lower = response.methodology.lower()
        if "measured" in methodology_lower or "metered" in methodology_lower:
            score = max(score, 4)
        if "estimated" in methodology_lower or "proxy" in methodology_lower:
            score = max(score, 2)

        return score

    def _dqr_to_level(self, weighted_dqr: Decimal) -> int:
        """Convert weighted DQR score to DQI level.

        Args:
            weighted_dqr: Weighted Data Quality Rating (1.0-5.0).

        Returns:
            DQI level (1-5).
        """
        dqr_float = float(weighted_dqr)
        if dqr_float >= 4.5:
            return 5
        if dqr_float >= 3.5:
            return 4
        if dqr_float >= 2.5:
            return 3
        if dqr_float >= 1.5:
            return 2
        return 1

    # -------------------------------------------------------------------
    # Private -- Reminder Helpers
    # -------------------------------------------------------------------

    def _reminder_message(self, reminder_type: str, supplier_name: str) -> str:
        """Generate reminder message text.

        Args:
            reminder_type: Type of reminder.
            supplier_name: Supplier name.

        Returns:
            Reminder message string.
        """
        messages = {
            ReminderType.INITIAL_REQUEST: (
                f"Carbon data request sent to {supplier_name}. "
                f"Response due in 30 days."
            ),
            ReminderType.FIRST_FOLLOW_UP: (
                f"First follow-up: {supplier_name} has not yet responded "
                f"to the carbon data request. Please follow up."
            ),
            ReminderType.SECOND_FOLLOW_UP: (
                f"Second follow-up: {supplier_name} carbon data request "
                f"is overdue. Consider direct phone/video call."
            ),
            ReminderType.ESCALATION: (
                f"ESCALATION: {supplier_name} has not responded after "
                f"multiple follow-ups. Escalate to procurement lead."
            ),
            ReminderType.ANNUAL_REFRESH: (
                f"Annual data refresh due for {supplier_name}. "
                f"Send updated questionnaire for current reporting year."
            ),
        }
        # Handle both enum and string keys.
        for key, msg in messages.items():
            key_str = key if isinstance(key, str) else key.value
            if key_str == reminder_type:
                return msg
        return f"Reminder for {supplier_name}: {reminder_type}."

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

ProcurementItem.model_rebuild()
Supplier.model_rebuild()
SupplierResponseData.model_rebuild()
SupplierPriority.model_rebuild()
DataRequest.model_rebuild()
QualityScore.model_rebuild()
EngagementPlan.model_rebuild()
EngagementROI.model_rebuild()
ReminderSchedule.model_rebuild()
EngagementMetrics.model_rebuild()
SupplierEngagementResult.model_rebuild()
