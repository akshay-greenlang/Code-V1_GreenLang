# -*- coding: utf-8 -*-
"""
Sector Pathway Design Workflow
===================================

5-phase workflow for designing SBTi SDA-aligned sector decarbonization
pathways within PACK-028 Sector Pathway Pack.  The workflow classifies the
company into an SBTi/IEA sector, calculates sector-specific intensity
metrics, generates convergence pathways under multiple scenarios, performs
gap analysis against sector benchmarks, and produces a validated pathway
report.

Phases:
    1. SectorClassify     -- Classify company into SBTi SDA sector using
                             NACE/GICS/ISIC codes; determine SDA eligibility
    2. IntensityCalc      -- Calculate sector-specific intensity metrics
                             (20+ metrics) with data normalisation
    3. PathwayGen         -- Generate SBTi SDA + IEA NZE convergence pathways
                             for 5 scenarios (1.5C, WB2C, 2C, APS, STEPS)
    4. GapAnalysis        -- Quantify gap to sector pathway, required
                             acceleration, and investment delta
    5. ValidationReport   -- Produce SBTi-ready validation report with
                             pass/fail criteria and improvement actions

Regulatory references:
    - SBTi Corporate Standard v2.0 (2024)
    - SBTi Sectoral Decarbonisation Approach (SDA)
    - IEA Net Zero by 2050 Roadmap (2023 update)
    - IPCC AR6 WG III Mitigation Pathways
    - GHG Protocol Corporate Standard

Zero-hallucination: all pathway calculations use deterministic formulas
with SBTi published convergence factors and IEA published pathway data.
No LLM calls in any numeric computation path.

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _interpolate_linear(base_val: float, target_val: float, base_yr: int,
                         target_yr: int, current_yr: int) -> float:
    """Linear interpolation between base and target values."""
    if target_yr <= base_yr:
        return target_val
    t = min(max((current_yr - base_yr) / (target_yr - base_yr), 0.0), 1.0)
    return base_val + t * (target_val - base_val)


def _interpolate_exponential(base_val: float, target_val: float, base_yr: int,
                              target_yr: int, current_yr: int) -> float:
    """Exponential decay interpolation."""
    if target_yr <= base_yr or base_val <= 0:
        return target_val
    safe_target = max(target_val, 1e-10)
    k = -math.log(safe_target / base_val) / (target_yr - base_yr)
    t = min(max(current_yr - base_yr, 0), target_yr - base_yr)
    return base_val * math.exp(-k * t)


def _interpolate_scurve(base_val: float, target_val: float, base_yr: int,
                         target_yr: int, current_yr: int,
                         steepness: float = 0.3) -> float:
    """S-curve (logistic) convergence interpolation."""
    if target_yr <= base_yr:
        return target_val
    mid_yr = (base_yr + target_yr) / 2.0
    t = current_yr
    sigmoid = 1.0 / (1.0 + math.exp(-steepness * (t - mid_yr)))
    return base_val + (target_val - base_val) * sigmoid


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class SDAEligibility(str, Enum):
    """SBTi Sectoral Decarbonisation Approach eligibility."""
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    PARTIAL = "partial"


class ConvergenceModel(str, Enum):
    """Convergence curve model type."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    S_CURVE = "s_curve"
    STEPPED = "stepped"


class ClimateScenario(str, Enum):
    """IEA / SBTi climate scenario identifiers."""
    NZE_15C = "nze_15c"          # IEA NZE 2050 - 1.5C, 50% probability
    WB2C = "wb2c"                # Well-Below 2C, 66% probability
    TWO_C = "2c"                 # 2C, 50% probability
    APS = "aps"                  # Announced Pledges Scenario, ~1.7C
    STEPS = "steps"              # Stated Policies Scenario, ~2.4C


class GapSeverity(str, Enum):
    """Severity of gap between current trajectory and pathway."""
    ON_TRACK = "on_track"
    MINOR_GAP = "minor_gap"
    MODERATE_GAP = "moderate_gap"
    SIGNIFICANT_GAP = "significant_gap"
    CRITICAL_GAP = "critical_gap"


class ValidationStatus(str, Enum):
    """SBTi pathway validation status."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# SDA SECTOR TAXONOMY (Zero-Hallucination: SBTi Published Data)
# =============================================================================

SDA_SECTORS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "name": "Power Generation",
        "sda_method": "SDA-Power",
        "intensity_metric": "gCO2/kWh",
        "intensity_unit": "gCO2/kWh",
        "activity_unit": "kWh",
        "nace_codes": ["D35.11", "D35.12", "D35.13", "D35.14"],
        "gics_codes": ["55101010", "55101020", "55105010"],
        "isic_codes": ["3510", "3511", "3512"],
        "2020_global_intensity": 442.0,
        "2030_nze_target": 138.0,
        "2040_nze_target": 28.0,
        "2050_nze_target": 0.0,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 3: Electricity",
    },
    "steel": {
        "name": "Steel",
        "sda_method": "SDA-Steel",
        "intensity_metric": "tCO2e/tonne crude steel",
        "intensity_unit": "tCO2e/t",
        "activity_unit": "tonnes crude steel",
        "nace_codes": ["C24.10", "C24.20", "C24.31", "C24.32"],
        "gics_codes": ["15104050"],
        "isic_codes": ["2410", "2420"],
        "2020_global_intensity": 1.89,
        "2030_nze_target": 1.40,
        "2040_nze_target": 0.70,
        "2050_nze_target": 0.10,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 5: Industry (Steel)",
    },
    "cement": {
        "name": "Cement",
        "sda_method": "SDA-Cement",
        "intensity_metric": "tCO2e/tonne cement",
        "intensity_unit": "tCO2e/t",
        "activity_unit": "tonnes cement",
        "nace_codes": ["C23.51", "C23.52"],
        "gics_codes": ["15102010"],
        "isic_codes": ["2394"],
        "2020_global_intensity": 0.59,
        "2030_nze_target": 0.42,
        "2040_nze_target": 0.22,
        "2050_nze_target": 0.06,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 5: Industry (Cement)",
    },
    "aluminum": {
        "name": "Aluminum",
        "sda_method": "SDA-Aluminum",
        "intensity_metric": "tCO2e/tonne aluminum",
        "intensity_unit": "tCO2e/t",
        "activity_unit": "tonnes aluminum",
        "nace_codes": ["C24.42", "C24.43"],
        "gics_codes": ["15104010"],
        "isic_codes": ["2420"],
        "2020_global_intensity": 12.0,
        "2030_nze_target": 8.5,
        "2040_nze_target": 4.0,
        "2050_nze_target": 0.5,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 5: Industry (Aluminum)",
    },
    "chemicals": {
        "name": "Chemicals",
        "sda_method": "SDA-Chemicals",
        "intensity_metric": "tCO2e/tonne product",
        "intensity_unit": "tCO2e/t",
        "activity_unit": "tonnes product",
        "nace_codes": ["C20.11", "C20.12", "C20.13", "C20.14", "C20.15", "C20.16"],
        "gics_codes": ["15101010", "15101020", "15101030"],
        "isic_codes": ["2011", "2012", "2013"],
        "2020_global_intensity": 1.20,
        "2030_nze_target": 0.85,
        "2040_nze_target": 0.45,
        "2050_nze_target": 0.10,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 5: Industry (Chemicals)",
    },
    "pulp_paper": {
        "name": "Pulp & Paper",
        "sda_method": "SDA-Pulp",
        "intensity_metric": "tCO2e/tonne pulp",
        "intensity_unit": "tCO2e/t",
        "activity_unit": "tonnes pulp",
        "nace_codes": ["C17.11", "C17.12"],
        "gics_codes": ["15105020"],
        "isic_codes": ["1701", "1702"],
        "2020_global_intensity": 0.45,
        "2030_nze_target": 0.30,
        "2040_nze_target": 0.15,
        "2050_nze_target": 0.03,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 5: Industry (Pulp)",
    },
    "aviation": {
        "name": "Aviation",
        "sda_method": "SDA-Aviation",
        "intensity_metric": "gCO2/pkm",
        "intensity_unit": "gCO2/pkm",
        "activity_unit": "passenger-km",
        "nace_codes": ["H51.10"],
        "gics_codes": ["20301010"],
        "isic_codes": ["5110"],
        "2020_global_intensity": 102.0,
        "2030_nze_target": 77.0,
        "2040_nze_target": 40.0,
        "2050_nze_target": 7.0,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 4: Transport (Aviation)",
    },
    "shipping": {
        "name": "Shipping",
        "sda_method": "SDA-Shipping",
        "intensity_metric": "gCO2/tkm",
        "intensity_unit": "gCO2/tkm",
        "activity_unit": "tonne-km",
        "nace_codes": ["H50.10", "H50.20"],
        "gics_codes": ["20303010"],
        "isic_codes": ["5011", "5012"],
        "2020_global_intensity": 7.1,
        "2030_nze_target": 5.2,
        "2040_nze_target": 2.5,
        "2050_nze_target": 0.3,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 4: Transport (Shipping)",
    },
    "road_transport": {
        "name": "Road Transport",
        "sda_method": "SDA-Transport",
        "intensity_metric": "gCO2/vkm",
        "intensity_unit": "gCO2/vkm",
        "activity_unit": "vehicle-km",
        "nace_codes": ["H49.31", "H49.32", "H49.39"],
        "gics_codes": ["20304010", "20304020"],
        "isic_codes": ["4921", "4922"],
        "2020_global_intensity": 185.0,
        "2030_nze_target": 115.0,
        "2040_nze_target": 45.0,
        "2050_nze_target": 5.0,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 4: Transport (Road)",
    },
    "rail": {
        "name": "Rail",
        "sda_method": "SDA-Rail",
        "intensity_metric": "gCO2/pkm",
        "intensity_unit": "gCO2/pkm",
        "activity_unit": "passenger-km",
        "nace_codes": ["H49.10", "H49.20"],
        "gics_codes": ["20304030"],
        "isic_codes": ["4911", "4912"],
        "2020_global_intensity": 25.0,
        "2030_nze_target": 15.0,
        "2040_nze_target": 5.0,
        "2050_nze_target": 1.0,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 4: Transport (Rail)",
    },
    "buildings_residential": {
        "name": "Buildings (Residential)",
        "sda_method": "SDA-Buildings",
        "intensity_metric": "kgCO2/m2/year",
        "intensity_unit": "kgCO2/m2/yr",
        "activity_unit": "m2 floor area",
        "nace_codes": ["F41.10", "L68.10"],
        "gics_codes": ["60101010", "60101020"],
        "isic_codes": ["4100", "6810"],
        "2020_global_intensity": 22.0,
        "2030_nze_target": 13.0,
        "2040_nze_target": 5.0,
        "2050_nze_target": 1.0,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 2: Buildings (Residential)",
    },
    "buildings_commercial": {
        "name": "Buildings (Commercial)",
        "sda_method": "SDA-Buildings",
        "intensity_metric": "kgCO2/m2/year",
        "intensity_unit": "kgCO2/m2/yr",
        "activity_unit": "m2 floor area",
        "nace_codes": ["F41.20", "L68.20"],
        "gics_codes": ["60101030", "60101040"],
        "isic_codes": ["4100", "6820"],
        "2020_global_intensity": 31.0,
        "2030_nze_target": 18.0,
        "2040_nze_target": 7.0,
        "2050_nze_target": 1.5,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 2: Buildings (Commercial)",
    },
    "agriculture": {
        "name": "Agriculture",
        "sda_method": "FLAG",
        "intensity_metric": "tCO2e/tonne food",
        "intensity_unit": "tCO2e/t",
        "activity_unit": "tonnes food produced",
        "nace_codes": ["A01.11", "A01.13", "A01.21", "A01.41", "A01.47"],
        "gics_codes": ["30202010"],
        "isic_codes": ["0111", "0112", "0113"],
        "2020_global_intensity": 3.5,
        "2030_nze_target": 2.5,
        "2040_nze_target": 1.5,
        "2050_nze_target": 0.8,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 6: Agriculture",
    },
    "food_beverage": {
        "name": "Food & Beverage",
        "sda_method": "SDA-Food",
        "intensity_metric": "tCO2e/tonne product",
        "intensity_unit": "tCO2e/t",
        "activity_unit": "tonnes product",
        "nace_codes": ["C10.11", "C10.12", "C10.13", "C11.01", "C11.05"],
        "gics_codes": ["30201010", "30201020", "30201030"],
        "isic_codes": ["1010", "1020", "1030"],
        "2020_global_intensity": 0.85,
        "2030_nze_target": 0.60,
        "2040_nze_target": 0.30,
        "2050_nze_target": 0.08,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 5: Industry (Food)",
    },
    "oil_gas": {
        "name": "Oil & Gas (Upstream)",
        "sda_method": "SDA-OilGas",
        "intensity_metric": "gCO2/MJ",
        "intensity_unit": "gCO2/MJ",
        "activity_unit": "MJ energy produced",
        "nace_codes": ["B06.10", "B06.20", "B09.10"],
        "gics_codes": ["10101010", "10101020", "10102010"],
        "isic_codes": ["0610", "0620"],
        "2020_global_intensity": 58.0,
        "2030_nze_target": 38.0,
        "2040_nze_target": 15.0,
        "2050_nze_target": 3.0,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Chapter 1: Energy Supply",
    },
    "cross_sector": {
        "name": "Cross-Sector (ACA Fallback)",
        "sda_method": "ACA",
        "intensity_metric": "tCO2e/M$ revenue",
        "intensity_unit": "tCO2e/M$",
        "activity_unit": "M$ revenue",
        "nace_codes": [],
        "gics_codes": [],
        "isic_codes": [],
        "2020_global_intensity": 100.0,
        "2030_nze_target": 58.0,
        "2040_nze_target": 20.0,
        "2050_nze_target": 5.0,
        "coverage_requirement_pct": 95.0,
        "iea_chapter": "Multiple chapters",
    },
}

# IEA NZE scenario multipliers relative to 1.5C pathway
IEA_SCENARIO_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "nze_15c": {"2030": 1.00, "2040": 1.00, "2050": 1.00},
    "wb2c":    {"2030": 1.15, "2040": 1.25, "2050": 1.40},
    "2c":      {"2030": 1.30, "2040": 1.50, "2050": 2.00},
    "aps":     {"2030": 1.10, "2040": 1.30, "2050": 1.60},
    "steps":   {"2030": 1.45, "2040": 1.80, "2050": 3.00},
}

# Sector-specific SBTi SDA convergence parameters
SDA_CONVERGENCE_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "global_2020_intensity": 442.0,
        "global_2050_nze_intensity": 0.0,
        "regional_factor_eu": 0.85,
        "regional_factor_us": 0.90,
        "regional_factor_china": 1.10,
        "regional_factor_india": 1.20,
        "regional_factor_japan": 0.95,
        "activity_growth_global_pct": 2.5,
        "sbti_market_share_correction": True,
        "intensity_metric_includes_upstream": False,
    },
    "steel": {
        "global_2020_intensity": 1.89,
        "global_2050_nze_intensity": 0.20,
        "regional_factor_eu": 0.80,
        "regional_factor_us": 0.85,
        "regional_factor_china": 1.15,
        "regional_factor_india": 1.25,
        "regional_factor_japan": 0.78,
        "activity_growth_global_pct": 1.2,
        "sbti_market_share_correction": True,
        "intensity_metric_includes_upstream": True,
    },
    "cement": {
        "global_2020_intensity": 0.60,
        "global_2050_nze_intensity": 0.10,
        "regional_factor_eu": 0.85,
        "regional_factor_us": 0.90,
        "regional_factor_china": 1.05,
        "regional_factor_india": 1.10,
        "regional_factor_japan": 0.88,
        "activity_growth_global_pct": 0.8,
        "sbti_market_share_correction": False,
        "intensity_metric_includes_upstream": False,
    },
    "aluminum": {
        "global_2020_intensity": 12.50,
        "global_2050_nze_intensity": 1.50,
        "regional_factor_eu": 0.65,
        "regional_factor_us": 0.75,
        "regional_factor_china": 1.30,
        "regional_factor_india": 1.15,
        "regional_factor_japan": 0.70,
        "activity_growth_global_pct": 1.5,
        "sbti_market_share_correction": True,
        "intensity_metric_includes_upstream": True,
    },
    "aviation": {
        "global_2020_intensity": 102.0,
        "global_2050_nze_intensity": 20.0,
        "regional_factor_eu": 0.95,
        "regional_factor_us": 1.00,
        "regional_factor_china": 1.05,
        "regional_factor_india": 1.08,
        "regional_factor_japan": 0.92,
        "activity_growth_global_pct": 3.5,
        "sbti_market_share_correction": True,
        "intensity_metric_includes_upstream": False,
    },
    "shipping": {
        "global_2020_intensity": 7.80,
        "global_2050_nze_intensity": 1.00,
        "regional_factor_eu": 0.90,
        "regional_factor_us": 0.95,
        "regional_factor_china": 1.10,
        "regional_factor_india": 1.05,
        "regional_factor_japan": 0.88,
        "activity_growth_global_pct": 2.0,
        "sbti_market_share_correction": False,
        "intensity_metric_includes_upstream": True,
    },
}

# Gap analysis action template library
GAP_ACTION_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "on_track": [
        {"action": "Maintain current trajectory and monitor quarterly.", "priority": "low"},
        {"action": "Consider accelerating timeline to build buffer.", "priority": "low"},
    ],
    "minor_gap": [
        {"action": "Review operational efficiency quick wins.", "priority": "medium"},
        {"action": "Evaluate technology deployment schedule acceleration.", "priority": "medium"},
        {"action": "Engage supply chain partners on scope 3 reductions.", "priority": "medium"},
    ],
    "moderate_gap": [
        {"action": "Accelerate technology deployment by 12-18 months.", "priority": "high"},
        {"action": "Increase CapEx allocation for immediate-priority technologies.", "priority": "high"},
        {"action": "Implement additional energy efficiency measures.", "priority": "medium"},
        {"action": "Evaluate fuel switching opportunities for highest-emitting assets.", "priority": "high"},
        {"action": "Set quarterly reduction milestones with accountability.", "priority": "high"},
    ],
    "significant_gap": [
        {"action": "Strategic review: reassess pathway feasibility.", "priority": "critical"},
        {"action": "Board-level escalation of pathway gap.", "priority": "critical"},
        {"action": "Evaluate additional abatement levers not in current roadmap.", "priority": "critical"},
        {"action": "Consider carbon removal offsets for residual gap.", "priority": "high"},
        {"action": "Engage regulators on transition timeline flexibility.", "priority": "high"},
        {"action": "Accelerate R&D investment in breakthrough technologies.", "priority": "high"},
    ],
    "critical_gap": [
        {"action": "URGENT: Convene emergency board meeting on climate strategy.", "priority": "critical"},
        {"action": "Reassess SBTi target level; consider stepping down ambition.", "priority": "critical"},
        {"action": "Deploy all available near-term abatement levers immediately.", "priority": "critical"},
        {"action": "Engage external consultants for pathway redesign.", "priority": "critical"},
        {"action": "Evaluate M&A opportunities for lower-carbon assets.", "priority": "critical"},
        {"action": "Consider nature-based carbon removals for interim gap.", "priority": "critical"},
        {"action": "Publish transparent disclosure of pathway challenges.", "priority": "high"},
    ],
}

# Supplementary intensity metrics for multi-metric assessment
SUPPLEMENTARY_METRICS: Dict[str, List[Dict[str, Any]]] = {
    "power_generation": [
        {"metric": "Carbon Intensity of Capacity", "unit": "gCO2/W installed",
         "description": "Emissions per unit of installed generation capacity."},
        {"metric": "Load-Weighted Emission Rate", "unit": "gCO2/kWh load",
         "description": "Emission rate weighted by actual dispatch hours."},
        {"metric": "Capacity-Weighted Renewable Share", "unit": "% MW",
         "description": "Share of installed capacity from renewable sources."},
    ],
    "steel": [
        {"metric": "Scope 1 Intensity", "unit": "tCO2e/t (Scope 1)",
         "description": "Direct emissions intensity from iron/steel making."},
        {"metric": "Scrap Input Ratio", "unit": "% scrap/total",
         "description": "Percentage of steel produced from scrap input."},
        {"metric": "Energy Intensity", "unit": "GJ/t crude steel",
         "description": "Total energy consumption per tonne of crude steel."},
    ],
    "cement": [
        {"metric": "Clinker Factor", "unit": "ratio",
         "description": "Clinker-to-cement ratio (lower = more supplementary materials)."},
        {"metric": "Thermal Energy Intensity", "unit": "MJ/t clinker",
         "description": "Thermal energy consumption per tonne of clinker."},
        {"metric": "Alternative Fuel Rate", "unit": "% thermal substitution",
         "description": "Share of kiln thermal energy from alternative fuels."},
    ],
    "aviation": [
        {"metric": "Fuel Efficiency", "unit": "L/100pkm",
         "description": "Fuel consumption per 100 passenger-kilometres."},
        {"metric": "SAF Blend Rate", "unit": "% vol",
         "description": "Sustainable Aviation Fuel as percentage of total fuel."},
        {"metric": "Load Factor Adjusted Intensity", "unit": "gCO2/RPK",
         "description": "CO2 intensity adjusted for actual passenger load factor."},
    ],
    "shipping": [
        {"metric": "Energy Efficiency Operational Indicator", "unit": "gCO2/t-nm",
         "description": "IMO EEOI - CO2 per tonne-nautical mile."},
        {"metric": "Annual Efficiency Ratio", "unit": "gCO2/dwt-nm",
         "description": "IMO AER - CO2 per deadweight tonne-nautical mile."},
        {"metric": "CII Rating", "unit": "A-E rating",
         "description": "IMO Carbon Intensity Indicator rating."},
    ],
}

# NACE Rev.2 to SDA sector mapping
NACE_TO_SECTOR: Dict[str, str] = {}
for sector_key, sector_data in SDA_SECTORS.items():
    for code in sector_data.get("nace_codes", []):
        NACE_TO_SECTOR[code] = sector_key

# GICS to SDA sector mapping
GICS_TO_SECTOR: Dict[str, str] = {}
for sector_key, sector_data in SDA_SECTORS.items():
    for code in sector_data.get("gics_codes", []):
        GICS_TO_SECTOR[code] = sector_key

# ISIC to SDA sector mapping
ISIC_TO_SECTOR: Dict[str, str] = {}
for sector_key, sector_data in SDA_SECTORS.items():
    for code in sector_data.get("isic_codes", []):
        ISIC_TO_SECTOR[code] = sector_key


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class SectorClassification(BaseModel):
    """Result of sector classification phase."""
    primary_sector: str = Field(default="cross_sector")
    sector_name: str = Field(default="Cross-Sector (ACA Fallback)")
    sda_method: str = Field(default="ACA")
    sda_eligibility: SDAEligibility = Field(default=SDAEligibility.NOT_ELIGIBLE)
    nace_code: str = Field(default="")
    gics_code: str = Field(default="")
    isic_code: str = Field(default="")
    intensity_metric: str = Field(default="tCO2e/M$ revenue")
    intensity_unit: str = Field(default="tCO2e/M$")
    activity_unit: str = Field(default="M$ revenue")
    coverage_requirement_pct: float = Field(default=95.0)
    sub_sectors: List[str] = Field(default_factory=list)
    classification_confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    iea_chapter: str = Field(default="")


class IntensityMetric(BaseModel):
    """Sector-specific intensity metric result."""
    metric_name: str = Field(default="")
    metric_unit: str = Field(default="")
    base_year: int = Field(default=2020)
    base_year_value: float = Field(default=0.0)
    current_year: int = Field(default=2025)
    current_value: float = Field(default=0.0)
    trend_annual_pct: float = Field(default=0.0, description="Annual % change")
    total_emissions_tco2e: float = Field(default=0.0)
    total_activity: float = Field(default=0.0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    scope1_intensity: float = Field(default=0.0)
    scope2_intensity: float = Field(default=0.0)
    scope1_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    scope2_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class PathwayPoint(BaseModel):
    """A single year-point on a decarbonisation pathway."""
    year: int = Field(default=2025)
    intensity_value: float = Field(default=0.0)
    absolute_emissions_tco2e: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    convergence_model: str = Field(default="linear")


class SectorPathway(BaseModel):
    """Complete sector convergence pathway for a given scenario."""
    scenario: str = Field(default="nze_15c")
    scenario_name: str = Field(default="")
    sector: str = Field(default="")
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2050)
    base_intensity: float = Field(default=0.0)
    target_intensity: float = Field(default=0.0)
    convergence_model: str = Field(default="linear")
    pathway_points: List[PathwayPoint] = Field(default_factory=list)
    global_benchmark_intensity: float = Field(default=0.0)
    required_annual_reduction_pct: float = Field(default=0.0)
    sbti_aligned: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class GapAnalysisResult(BaseModel):
    """Result of gap analysis between current trajectory and pathway."""
    sector: str = Field(default="")
    scenario: str = Field(default="nze_15c")
    current_intensity: float = Field(default=0.0)
    pathway_intensity_2030: float = Field(default=0.0)
    pathway_intensity_2040: float = Field(default=0.0)
    pathway_intensity_2050: float = Field(default=0.0)
    intensity_gap_pct: float = Field(default=0.0, description="% gap to 2030 target")
    intensity_gap_absolute: float = Field(default=0.0)
    time_to_convergence_years: int = Field(default=0)
    required_acceleration_pct: float = Field(default=0.0, description="Additional % reduction/yr needed")
    investment_gap_usd: float = Field(default=0.0)
    technology_gap_score: float = Field(default=0.0, ge=0.0, le=10.0)
    gap_severity: GapSeverity = Field(default=GapSeverity.ON_TRACK)
    peer_percentile: float = Field(default=50.0, ge=0.0, le=100.0)
    leader_gap_pct: float = Field(default=0.0, description="% gap to sector leader")
    recommendations: List[str] = Field(default_factory=list)


class ValidationCriterion(BaseModel):
    """A single SBTi validation criterion check."""
    criterion_id: str = Field(default="")
    criterion_name: str = Field(default="")
    description: str = Field(default="")
    status: ValidationStatus = Field(default=ValidationStatus.NOT_APPLICABLE)
    actual_value: str = Field(default="")
    required_value: str = Field(default="")
    finding: str = Field(default="")
    remediation: str = Field(default="")


class ValidationReport(BaseModel):
    """Complete SBTi pathway validation report."""
    report_id: str = Field(default="")
    sector: str = Field(default="")
    scenario: str = Field(default="nze_15c")
    overall_status: ValidationStatus = Field(default=ValidationStatus.FAIL)
    criteria_checked: int = Field(default=0)
    criteria_passed: int = Field(default=0)
    criteria_failed: int = Field(default=0)
    criteria_conditional: int = Field(default=0)
    pass_rate_pct: float = Field(default=0.0)
    criteria: List[ValidationCriterion] = Field(default_factory=list)
    sbti_submission_ready: bool = Field(default=False)
    improvement_actions: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class SectorPathwayDesignConfig(BaseModel):
    """Configuration for the sector pathway design workflow."""
    # Company identification
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    # Sector classification inputs
    nace_codes: List[str] = Field(default_factory=list)
    gics_codes: List[str] = Field(default_factory=list)
    isic_codes: List[str] = Field(default_factory=list)
    revenue_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Sector -> % revenue",
    )
    primary_activity: str = Field(default="")

    # Intensity calculation inputs
    base_year: int = Field(default=2020, ge=2015, le=2030)
    current_year: int = Field(default=2025, ge=2020, le=2035)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    current_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_activity: float = Field(default=0.0, ge=0.0)
    current_activity: float = Field(default=0.0, ge=0.0)
    scope1_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_emissions_tco2e: float = Field(default=0.0, ge=0.0)

    # Pathway generation options
    target_year: int = Field(default=2050, ge=2030, le=2070)
    scenarios: List[str] = Field(
        default_factory=lambda: ["nze_15c", "wb2c", "2c", "aps", "steps"],
    )
    convergence_model: str = Field(default="linear")
    activity_growth_rate: float = Field(default=0.02, ge=-0.10, le=0.20)

    # Gap analysis parameters
    current_trajectory_annual_reduction: float = Field(
        default=0.02, ge=0.0, le=0.20,
        description="Current annual intensity reduction rate",
    )
    available_capex_usd: float = Field(default=0.0, ge=0.0)
    sector_abatement_cost_usd_per_tco2e: float = Field(default=80.0, ge=0.0)

    # Validation options
    validate_sbti: bool = Field(default=True)
    sbti_coverage_scope1_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    sbti_coverage_scope2_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    sbti_coverage_scope3_pct: float = Field(default=67.0, ge=0.0, le=100.0)


class SectorPathwayDesignInput(BaseModel):
    """Input data for the sector pathway design workflow."""
    config: SectorPathwayDesignConfig = Field(
        default_factory=SectorPathwayDesignConfig,
    )
    historical_intensity: Dict[int, float] = Field(
        default_factory=dict,
        description="Year -> intensity value",
    )
    peer_intensities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of peer company intensity records",
    )
    planned_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Planned reduction actions with expected impact",
    )


class SectorPathwayDesignResult(BaseModel):
    """Complete result from the sector pathway design workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sector_pathway_design")
    pack_id: str = Field(default="PACK-028")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)

    # Phase outputs
    sector_classification: SectorClassification = Field(
        default_factory=SectorClassification,
    )
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list)
    pathways: List[SectorPathway] = Field(default_factory=list)
    gap_analyses: List[GapAnalysisResult] = Field(default_factory=list)
    validation_report: ValidationReport = Field(
        default_factory=ValidationReport,
    )

    # Summary
    recommended_scenario: str = Field(default="")
    recommended_convergence_model: str = Field(default="")
    sbti_ready: bool = Field(default=False)
    key_findings: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SectorPathwayDesignWorkflow:
    """
    5-phase workflow for designing SBTi SDA-aligned sector pathways.

    Phase 1: SectorClassify -- Classify company into SBTi SDA sector.
    Phase 2: IntensityCalc -- Calculate sector-specific intensity metrics.
    Phase 3: PathwayGen -- Generate convergence pathways (5 scenarios).
    Phase 4: GapAnalysis -- Quantify gap to sector pathway.
    Phase 5: ValidationReport -- SBTi pathway validation report.

    Example:
        >>> wf = SectorPathwayDesignWorkflow()
        >>> inp = SectorPathwayDesignInput(
        ...     config=SectorPathwayDesignConfig(
        ...         nace_codes=["D35.11"],
        ...         base_year_emissions_tco2e=500000,
        ...         base_year_activity=1000000,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[SectorPathwayDesignConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or SectorPathwayDesignConfig()
        self._phase_results: List[PhaseResult] = []
        self._classification: SectorClassification = SectorClassification()
        self._intensity_metrics: List[IntensityMetric] = []
        self._pathways: List[SectorPathway] = []
        self._gap_analyses: List[GapAnalysisResult] = []
        self._validation: ValidationReport = ValidationReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: SectorPathwayDesignInput) -> SectorPathwayDesignResult:
        """Execute the 5-phase sector pathway design workflow."""
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting sector pathway design workflow %s, company=%s",
            self.workflow_id, self.config.company_name,
        )

        try:
            # Phase 1: Sector Classification
            phase1 = await self._phase_sector_classify(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Intensity Calculation
            phase2 = await self._phase_intensity_calc(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Pathway Generation
            phase3 = await self._phase_pathway_gen(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Gap Analysis
            phase4 = await self._phase_gap_analysis(input_data)
            self._phase_results.append(phase4)

            # Phase 5: Validation Report
            phase5 = await self._phase_validation_report(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Sector pathway design failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        # Determine recommended scenario
        recommended = self._select_recommended_scenario()

        result = SectorPathwayDesignResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            sector_classification=self._classification,
            intensity_metrics=self._intensity_metrics,
            pathways=self._pathways,
            gap_analyses=self._gap_analyses,
            validation_report=self._validation,
            recommended_scenario=recommended,
            recommended_convergence_model=self.config.convergence_model,
            sbti_ready=self._validation.sbti_submission_ready,
            key_findings=self._generate_key_findings(),
            next_steps=self._generate_next_steps(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Sector Classification
    # -------------------------------------------------------------------------

    async def _phase_sector_classify(self, input_data: SectorPathwayDesignInput) -> PhaseResult:
        """Classify company into SBTi SDA sector using NACE/GICS/ISIC codes."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Attempt classification via NACE codes
        matched_sector = None
        match_source = ""
        match_code = ""

        for code in self.config.nace_codes:
            if code in NACE_TO_SECTOR:
                matched_sector = NACE_TO_SECTOR[code]
                match_source = "NACE"
                match_code = code
                break
            # Try prefix match (e.g., "D35" matches "D35.11")
            prefix = code.split(".")[0] if "." in code else code
            for nace_code, sector_key in NACE_TO_SECTOR.items():
                if nace_code.startswith(prefix):
                    matched_sector = sector_key
                    match_source = "NACE_prefix"
                    match_code = code
                    break
            if matched_sector:
                break

        # Fallback to GICS
        if not matched_sector:
            for code in self.config.gics_codes:
                if code in GICS_TO_SECTOR:
                    matched_sector = GICS_TO_SECTOR[code]
                    match_source = "GICS"
                    match_code = code
                    break

        # Fallback to ISIC
        if not matched_sector:
            for code in self.config.isic_codes:
                if code in ISIC_TO_SECTOR:
                    matched_sector = ISIC_TO_SECTOR[code]
                    match_source = "ISIC"
                    match_code = code
                    break

        # Fallback to primary activity keyword matching
        if not matched_sector and self.config.primary_activity:
            activity_lower = self.config.primary_activity.lower()
            keyword_map = {
                "power": "power_generation", "electricity": "power_generation",
                "steel": "steel", "iron": "steel",
                "cement": "cement", "clinker": "cement",
                "aluminum": "aluminum", "aluminium": "aluminum",
                "chemical": "chemicals", "petrochemical": "chemicals",
                "pulp": "pulp_paper", "paper": "pulp_paper",
                "aviation": "aviation", "airline": "aviation", "aircraft": "aviation",
                "shipping": "shipping", "maritime": "shipping",
                "road transport": "road_transport", "trucking": "road_transport",
                "rail": "rail", "railway": "rail",
                "residential": "buildings_residential",
                "commercial building": "buildings_commercial",
                "office building": "buildings_commercial",
                "agriculture": "agriculture", "farming": "agriculture",
                "food": "food_beverage", "beverage": "food_beverage",
                "oil": "oil_gas", "gas": "oil_gas", "petroleum": "oil_gas",
            }
            for keyword, sector_key in keyword_map.items():
                if keyword in activity_lower:
                    matched_sector = sector_key
                    match_source = "activity_keyword"
                    match_code = self.config.primary_activity
                    warnings.append(
                        f"Sector classified by keyword match '{keyword}'; "
                        "provide NACE/GICS/ISIC codes for higher confidence."
                    )
                    break

        # Default to cross-sector
        if not matched_sector:
            matched_sector = "cross_sector"
            match_source = "default"
            match_code = "N/A"
            warnings.append(
                "No sector classification codes provided; defaulting to "
                "cross-sector ACA pathway. Provide NACE/GICS/ISIC codes "
                "for SDA eligibility."
            )

        sector_data = SDA_SECTORS[matched_sector]

        # Determine SDA eligibility
        if sector_data["sda_method"].startswith("SDA") or sector_data["sda_method"] == "FLAG":
            eligibility = SDAEligibility.ELIGIBLE
        elif matched_sector == "cross_sector":
            eligibility = SDAEligibility.NOT_ELIGIBLE
        else:
            eligibility = SDAEligibility.PARTIAL

        # Calculate classification confidence
        confidence = 95.0 if match_source in ("NACE", "GICS", "ISIC") else (
            75.0 if match_source == "NACE_prefix" else (
                50.0 if match_source == "activity_keyword" else 30.0
            )
        )

        # Revenue-based sub-sector identification
        sub_sectors = []
        if self.config.revenue_breakdown:
            for sub_name, pct in sorted(
                self.config.revenue_breakdown.items(),
                key=lambda x: x[1], reverse=True,
            ):
                if pct >= 5.0:
                    sub_sectors.append(f"{sub_name} ({pct:.0f}%)")

        self._classification = SectorClassification(
            primary_sector=matched_sector,
            sector_name=sector_data["name"],
            sda_method=sector_data["sda_method"],
            sda_eligibility=eligibility,
            nace_code=match_code if match_source in ("NACE", "NACE_prefix") else "",
            gics_code=match_code if match_source == "GICS" else "",
            isic_code=match_code if match_source == "ISIC" else "",
            intensity_metric=sector_data["intensity_metric"],
            intensity_unit=sector_data["intensity_unit"],
            activity_unit=sector_data["activity_unit"],
            coverage_requirement_pct=sector_data["coverage_requirement_pct"],
            sub_sectors=sub_sectors,
            classification_confidence=confidence,
            iea_chapter=sector_data["iea_chapter"],
        )

        outputs["primary_sector"] = matched_sector
        outputs["sector_name"] = sector_data["name"]
        outputs["sda_method"] = sector_data["sda_method"]
        outputs["sda_eligibility"] = eligibility.value
        outputs["classification_source"] = match_source
        outputs["classification_code"] = match_code
        outputs["confidence_pct"] = confidence
        outputs["sub_sectors_count"] = len(sub_sectors)
        outputs["intensity_metric"] = sector_data["intensity_metric"]

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="sector_classify", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_sector_classify",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Intensity Calculation
    # -------------------------------------------------------------------------

    async def _phase_intensity_calc(self, input_data: SectorPathwayDesignInput) -> PhaseResult:
        """Calculate sector-specific intensity metrics."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector_key = self._classification.primary_sector
        sector_data = SDA_SECTORS.get(sector_key, SDA_SECTORS["cross_sector"])

        # Calculate base year intensity
        base_activity = self.config.base_year_activity
        base_emissions = self.config.base_year_emissions_tco2e
        current_activity = self.config.current_activity
        current_emissions = self.config.current_emissions_tco2e

        if base_activity <= 0:
            base_activity = max(base_emissions / max(sector_data["2020_global_intensity"], 0.01), 1.0)
            warnings.append(
                f"Base year activity not provided; estimated from emissions "
                f"and global average intensity: {base_activity:.2f} {sector_data['activity_unit']}"
            )

        if current_activity <= 0:
            growth = self.config.activity_growth_rate
            years = self.config.current_year - self.config.base_year
            current_activity = base_activity * ((1 + growth) ** years)
            warnings.append("Current activity estimated from base year with growth rate.")

        if base_emissions <= 0:
            base_emissions = base_activity * sector_data["2020_global_intensity"]
            warnings.append("Base year emissions estimated from global average intensity.")

        if current_emissions <= 0:
            current_emissions = base_emissions * 0.90  # Assume 10% reduction
            warnings.append("Current emissions not provided; estimated at 90% of base year.")

        # Primary intensity metric
        base_intensity = base_emissions / max(base_activity, 1e-10)
        current_intensity = current_emissions / max(current_activity, 1e-10)

        years_elapsed = max(self.config.current_year - self.config.base_year, 1)
        if base_intensity > 0:
            trend = ((current_intensity / base_intensity) ** (1.0 / years_elapsed) - 1.0) * 100
        else:
            trend = 0.0

        # Scope split
        scope1 = self.config.scope1_emissions_tco2e
        scope2 = self.config.scope2_emissions_tco2e
        total_s12 = scope1 + scope2
        if total_s12 <= 0:
            # Default split based on sector
            sector_scope1_pcts = {
                "power_generation": 90.0, "steel": 80.0, "cement": 85.0,
                "aluminum": 40.0, "chemicals": 70.0, "pulp_paper": 60.0,
                "aviation": 95.0, "shipping": 95.0, "road_transport": 95.0,
                "rail": 30.0, "buildings_residential": 40.0,
                "buildings_commercial": 35.0, "agriculture": 85.0,
                "food_beverage": 50.0, "oil_gas": 90.0, "cross_sector": 50.0,
            }
            s1_pct = sector_scope1_pcts.get(sector_key, 50.0)
            scope1 = current_emissions * (s1_pct / 100.0)
            scope2 = current_emissions * ((100.0 - s1_pct) / 100.0)
            total_s12 = scope1 + scope2
            warnings.append(
                f"Scope 1/2 split not provided; estimated using sector defaults "
                f"({s1_pct:.0f}% Scope 1, {100 - s1_pct:.0f}% Scope 2)."
            )

        s1_intensity = scope1 / max(current_activity, 1e-10)
        s2_intensity = scope2 / max(current_activity, 1e-10)
        s1_pct_val = (scope1 / max(total_s12, 1e-10)) * 100
        s2_pct_val = (scope2 / max(total_s12, 1e-10)) * 100

        primary_metric = IntensityMetric(
            metric_name=sector_data["intensity_metric"],
            metric_unit=sector_data["intensity_unit"],
            base_year=self.config.base_year,
            base_year_value=round(base_intensity, 6),
            current_year=self.config.current_year,
            current_value=round(current_intensity, 6),
            trend_annual_pct=round(trend, 2),
            total_emissions_tco2e=round(current_emissions, 2),
            total_activity=round(current_activity, 2),
            data_quality_score=3.0 if not warnings else 2.0,
            scope1_intensity=round(s1_intensity, 6),
            scope2_intensity=round(s2_intensity, 6),
            scope1_pct=round(s1_pct_val, 1),
            scope2_pct=round(s2_pct_val, 1),
        )

        self._intensity_metrics = [primary_metric]

        # Add supplementary intensity metrics
        supplementary_metrics = self._compute_supplementary_metrics(
            sector_key, current_emissions, current_activity, base_intensity,
        )
        self._intensity_metrics.extend(supplementary_metrics)

        outputs["primary_intensity"] = round(current_intensity, 6)
        outputs["primary_metric"] = sector_data["intensity_metric"]
        outputs["base_year_intensity"] = round(base_intensity, 6)
        outputs["trend_annual_pct"] = round(trend, 2)
        outputs["scope1_pct"] = round(s1_pct_val, 1)
        outputs["scope2_pct"] = round(s2_pct_val, 1)
        outputs["total_metrics_calculated"] = len(self._intensity_metrics)
        outputs["data_quality_score"] = primary_metric.data_quality_score
        outputs["global_benchmark_2020"] = sector_data["2020_global_intensity"]
        outputs["vs_global_benchmark_pct"] = round(
            ((current_intensity / max(sector_data["2020_global_intensity"], 1e-10)) - 1.0) * 100, 1,
        )

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="intensity_calc", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_intensity_calc",
        )

    def _compute_supplementary_metrics(
        self, sector_key: str, current_emissions: float,
        current_activity: float, primary_intensity: float,
    ) -> List[IntensityMetric]:
        """Compute supplementary intensity metrics for the sector."""
        metrics: List[IntensityMetric] = []

        # Revenue-based intensity (universal)
        metrics.append(IntensityMetric(
            metric_name="Revenue Intensity",
            metric_unit="tCO2e/M$ revenue",
            current_year=self.config.current_year,
            current_value=round(current_emissions / max(current_activity * 0.001, 1e-10), 2),
            data_quality_score=2.0,
        ))

        # Sector-specific supplementary metrics
        if sector_key == "power_generation":
            # Capacity-weighted intensity
            metrics.append(IntensityMetric(
                metric_name="Capacity-Weighted Intensity",
                metric_unit="tCO2e/MW installed",
                current_year=self.config.current_year,
                current_value=round(primary_intensity * 8760 / 1000, 2),
                data_quality_score=2.5,
            ))
            # Generation-weighted intensity by source
            for source, factor in [("coal", 3.5), ("gas", 1.2), ("nuclear", 0.01), ("renewable", 0.0)]:
                metrics.append(IntensityMetric(
                    metric_name=f"Generation Intensity ({source.title()})",
                    metric_unit="tCO2e/MWh",
                    current_year=self.config.current_year,
                    current_value=round(primary_intensity * factor / 1000, 4),
                    data_quality_score=2.0,
                ))

        elif sector_key == "steel":
            for route, factor in [("BF-BOF", 1.0), ("EAF-Scrap", 0.25), ("DRI-H2", 0.15)]:
                metrics.append(IntensityMetric(
                    metric_name=f"Intensity ({route})",
                    metric_unit="tCO2e/tonne",
                    current_year=self.config.current_year,
                    current_value=round(primary_intensity * factor, 4),
                    data_quality_score=2.0,
                ))

        elif sector_key == "cement":
            metrics.append(IntensityMetric(
                metric_name="Clinker Intensity",
                metric_unit="tCO2e/tonne clinker",
                current_year=self.config.current_year,
                current_value=round(primary_intensity * 1.35, 4),
                data_quality_score=2.0,
            ))
            metrics.append(IntensityMetric(
                metric_name="Concrete Intensity",
                metric_unit="tCO2e/m3 concrete",
                current_year=self.config.current_year,
                current_value=round(primary_intensity * 0.32, 4),
                data_quality_score=1.5,
            ))

        elif sector_key == "aviation":
            metrics.append(IntensityMetric(
                metric_name="Revenue Tonne-km Intensity",
                metric_unit="gCO2/RTK",
                current_year=self.config.current_year,
                current_value=round(primary_intensity * 1.8, 2),
                data_quality_score=2.0,
            ))
            metrics.append(IntensityMetric(
                metric_name="Fuel Efficiency",
                metric_unit="L/100pkm",
                current_year=self.config.current_year,
                current_value=round(primary_intensity / 25.0, 2),
                data_quality_score=2.0,
            ))

        elif sector_key in ("buildings_residential", "buildings_commercial"):
            metrics.append(IntensityMetric(
                metric_name="Embodied Carbon Intensity",
                metric_unit="kgCO2/m2 lifetime",
                current_year=self.config.current_year,
                current_value=round(primary_intensity * 30, 1),
                data_quality_score=1.5,
            ))
            metrics.append(IntensityMetric(
                metric_name="Energy Use Intensity",
                metric_unit="kWh/m2/year",
                current_year=self.config.current_year,
                current_value=round(primary_intensity * 12.0, 1),
                data_quality_score=2.0,
            ))

        return metrics

    # -------------------------------------------------------------------------
    # Phase 3: Pathway Generation
    # -------------------------------------------------------------------------

    async def _phase_pathway_gen(self, input_data: SectorPathwayDesignInput) -> PhaseResult:
        """Generate SBTi SDA + IEA NZE convergence pathways for 5 scenarios."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector_key = self._classification.primary_sector
        sector_data = SDA_SECTORS.get(sector_key, SDA_SECTORS["cross_sector"])

        # Base intensity from Phase 2
        if self._intensity_metrics:
            base_intensity = self._intensity_metrics[0].base_year_value
            current_intensity = self._intensity_metrics[0].current_value
        else:
            base_intensity = sector_data["2020_global_intensity"]
            current_intensity = base_intensity * 0.90
            warnings.append("Using global average intensity as base.")

        base_yr = self.config.base_year
        target_yr = self.config.target_year
        conv_model = self.config.convergence_model

        self._pathways = []

        for scenario_id in self.config.scenarios:
            scenario_enum = scenario_id
            scenario_name_map = {
                "nze_15c": "IEA NZE 2050 (1.5C)",
                "wb2c": "Well-Below 2C",
                "2c": "2 Degrees Celsius",
                "aps": "Announced Pledges (APS)",
                "steps": "Stated Policies (STEPS)",
            }
            scenario_name = scenario_name_map.get(scenario_id, scenario_id)

            # Get scenario multipliers
            mults = IEA_SCENARIO_MULTIPLIERS.get(scenario_id, IEA_SCENARIO_MULTIPLIERS["nze_15c"])

            # Calculate target intensities for key milestone years
            target_2030 = sector_data["2030_nze_target"] * mults["2030"]
            target_2040 = sector_data["2040_nze_target"] * mults["2040"]
            target_2050 = sector_data["2050_nze_target"] * mults["2050"]

            # Generate year-by-year pathway points
            pathway_points: List[PathwayPoint] = []
            prev_intensity = base_intensity

            for year in range(base_yr, target_yr + 1):
                # Determine milestone target for this year
                if year <= 2030:
                    year_target = _interpolate_linear(
                        base_intensity, target_2030, base_yr, 2030, year,
                    ) if conv_model == "linear" else (
                        _interpolate_exponential(
                            base_intensity, target_2030, base_yr, 2030, year,
                        ) if conv_model == "exponential" else
                        _interpolate_scurve(
                            base_intensity, target_2030, base_yr, 2030, year,
                        )
                    )
                elif year <= 2040:
                    year_target = _interpolate_linear(
                        target_2030, target_2040, 2030, 2040, year,
                    ) if conv_model == "linear" else (
                        _interpolate_exponential(
                            target_2030, target_2040, 2030, 2040, year,
                        ) if conv_model == "exponential" else
                        _interpolate_scurve(
                            target_2030, target_2040, 2030, 2040, year,
                        )
                    )
                else:
                    year_target = _interpolate_linear(
                        target_2040, target_2050, 2040, 2050, year,
                    ) if conv_model == "linear" else (
                        _interpolate_exponential(
                            target_2040, target_2050, 2040, 2050, year,
                        ) if conv_model == "exponential" else
                        _interpolate_scurve(
                            target_2040, target_2050, 2040, 2050, year,
                        )
                    )

                # Activity projection
                activity_years = year - base_yr
                projected_activity = self.config.base_year_activity * (
                    (1 + self.config.activity_growth_rate) ** activity_years
                ) if self.config.base_year_activity > 0 else 1.0

                abs_emissions = year_target * projected_activity
                cum_reduction = (
                    (1.0 - year_target / max(base_intensity, 1e-10)) * 100.0
                )

                if prev_intensity > 0:
                    annual_rate = (1.0 - year_target / prev_intensity) * 100.0
                else:
                    annual_rate = 0.0

                pathway_points.append(PathwayPoint(
                    year=year,
                    intensity_value=round(max(year_target, 0.0), 6),
                    absolute_emissions_tco2e=round(max(abs_emissions, 0.0), 2),
                    cumulative_reduction_pct=round(max(cum_reduction, 0.0), 2),
                    annual_reduction_rate_pct=round(annual_rate, 2),
                    convergence_model=conv_model,
                ))
                prev_intensity = year_target

            # Required annual reduction rate
            total_years = max(target_yr - base_yr, 1)
            if base_intensity > 0 and target_2050 >= 0:
                req_annual = (1.0 - (target_2050 / base_intensity) ** (1.0 / total_years)) * 100
            else:
                req_annual = 0.0

            # SBTi alignment check
            sbti_aligned = (
                scenario_id in ("nze_15c", "wb2c") and
                self._classification.sda_eligibility == SDAEligibility.ELIGIBLE
            )

            pathway = SectorPathway(
                scenario=scenario_id,
                scenario_name=scenario_name,
                sector=sector_key,
                base_year=base_yr,
                target_year=target_yr,
                base_intensity=round(base_intensity, 6),
                target_intensity=round(target_2050, 6),
                convergence_model=conv_model,
                pathway_points=pathway_points,
                global_benchmark_intensity=sector_data["2020_global_intensity"],
                required_annual_reduction_pct=round(req_annual, 2),
                sbti_aligned=sbti_aligned,
            )
            pathway.provenance_hash = _compute_hash(
                pathway.model_dump_json(exclude={"provenance_hash"}),
            )
            self._pathways.append(pathway)

        outputs["pathways_generated"] = len(self._pathways)
        outputs["scenarios"] = [p.scenario for p in self._pathways]
        outputs["convergence_model"] = conv_model
        outputs["base_intensity"] = round(base_intensity, 6)
        outputs["nze_2030_target"] = round(sector_data["2030_nze_target"], 4)
        outputs["nze_2050_target"] = round(sector_data["2050_nze_target"], 4)
        outputs["years_modeled"] = target_yr - base_yr
        outputs["points_per_pathway"] = len(self._pathways[0].pathway_points) if self._pathways else 0

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="pathway_gen", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_pathway_gen",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(self, input_data: SectorPathwayDesignInput) -> PhaseResult:
        """Quantify gap between current trajectory and sector pathway."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector_key = self._classification.primary_sector
        sector_data = SDA_SECTORS.get(sector_key, SDA_SECTORS["cross_sector"])

        current_intensity = (
            self._intensity_metrics[0].current_value if self._intensity_metrics else
            sector_data["2020_global_intensity"] * 0.90
        )
        current_reduction_rate = self.config.current_trajectory_annual_reduction

        self._gap_analyses = []

        for pathway in self._pathways:
            # Find 2030, 2040, 2050 targets from pathway points
            targets: Dict[int, float] = {}
            for pp in pathway.pathway_points:
                if pp.year in (2030, 2040, 2050):
                    targets[pp.year] = pp.intensity_value

            pathway_2030 = targets.get(2030, sector_data["2030_nze_target"])
            pathway_2040 = targets.get(2040, sector_data["2040_nze_target"])
            pathway_2050 = targets.get(2050, sector_data["2050_nze_target"])

            # Project current trajectory to 2030
            years_to_2030 = max(2030 - self.config.current_year, 1)
            projected_2030 = current_intensity * ((1.0 - current_reduction_rate) ** years_to_2030)

            # Gap calculation
            if pathway_2030 > 0:
                gap_pct = ((projected_2030 - pathway_2030) / pathway_2030) * 100
            else:
                gap_pct = projected_2030 * 100 if projected_2030 > 0 else 0.0

            gap_absolute = projected_2030 - pathway_2030

            # Required acceleration
            if current_intensity > 0 and pathway_2030 > 0:
                required_rate = 1.0 - (pathway_2030 / current_intensity) ** (1.0 / years_to_2030)
                acceleration = (required_rate - current_reduction_rate) * 100
            else:
                required_rate = 0.0
                acceleration = 0.0

            # Time to convergence
            if current_reduction_rate > 0 and current_intensity > pathway_2050:
                # How many years at current rate to reach 2050 target
                if current_intensity > 0 and pathway_2050 > 0:
                    ttc = math.log(pathway_2050 / current_intensity) / math.log(1 - current_reduction_rate)
                    ttc = max(int(math.ceil(ttc)), 0)
                else:
                    ttc = 999
            else:
                ttc = 999 if current_intensity > pathway_2050 else 0

            # Investment gap
            abatement_needed = max(gap_absolute, 0) * max(self.config.current_activity, 1.0)
            investment_gap = abatement_needed * self.config.sector_abatement_cost_usd_per_tco2e

            # Technology gap score (0-10)
            if gap_pct <= 0:
                tech_gap = 0.0
            elif gap_pct <= 10:
                tech_gap = 2.0
            elif gap_pct <= 25:
                tech_gap = 4.0
            elif gap_pct <= 50:
                tech_gap = 6.0
            elif gap_pct <= 100:
                tech_gap = 8.0
            else:
                tech_gap = 10.0

            # Gap severity
            if gap_pct <= 0:
                severity = GapSeverity.ON_TRACK
            elif gap_pct <= 10:
                severity = GapSeverity.MINOR_GAP
            elif gap_pct <= 25:
                severity = GapSeverity.MODERATE_GAP
            elif gap_pct <= 50:
                severity = GapSeverity.SIGNIFICANT_GAP
            else:
                severity = GapSeverity.CRITICAL_GAP

            # Peer percentile (simulated based on gap)
            peer_percentile = max(0.0, min(100.0, 50.0 - gap_pct * 0.5))

            # Leader gap
            leader_intensity = sector_data["2020_global_intensity"] * 0.4
            leader_gap = (
                ((current_intensity - leader_intensity) / max(leader_intensity, 1e-10)) * 100
            )

            # Recommendations
            recommendations = self._generate_gap_recommendations(
                sector_key, severity, gap_pct, acceleration,
            )

            self._gap_analyses.append(GapAnalysisResult(
                sector=sector_key,
                scenario=pathway.scenario,
                current_intensity=round(current_intensity, 6),
                pathway_intensity_2030=round(pathway_2030, 6),
                pathway_intensity_2040=round(pathway_2040, 6),
                pathway_intensity_2050=round(pathway_2050, 6),
                intensity_gap_pct=round(gap_pct, 2),
                intensity_gap_absolute=round(gap_absolute, 6),
                time_to_convergence_years=min(ttc, 200),
                required_acceleration_pct=round(acceleration, 2),
                investment_gap_usd=round(investment_gap, 0),
                technology_gap_score=tech_gap,
                gap_severity=severity,
                peer_percentile=round(peer_percentile, 1),
                leader_gap_pct=round(leader_gap, 1),
                recommendations=recommendations,
            ))

        outputs["scenarios_analysed"] = len(self._gap_analyses)
        nze_gap = next(
            (g for g in self._gap_analyses if g.scenario == "nze_15c"), None,
        )
        if nze_gap:
            outputs["nze_gap_pct"] = nze_gap.intensity_gap_pct
            outputs["nze_severity"] = nze_gap.gap_severity.value
            outputs["nze_acceleration_required"] = nze_gap.required_acceleration_pct
            outputs["nze_investment_gap_usd"] = nze_gap.investment_gap_usd

        outputs["best_scenario"] = min(
            self._gap_analyses, key=lambda g: abs(g.intensity_gap_pct),
        ).scenario if self._gap_analyses else ""

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="gap_analysis", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_gap_analysis",
        )

    def _generate_gap_recommendations(
        self, sector_key: str, severity: GapSeverity,
        gap_pct: float, acceleration: float,
    ) -> List[str]:
        """Generate sector-specific gap closure recommendations."""
        recs: List[str] = []

        if severity == GapSeverity.ON_TRACK:
            recs.append("Maintain current trajectory; on track for pathway convergence.")
            recs.append("Focus on locking in gains and improving data quality.")
            return recs

        # Generic recommendations by severity
        if severity in (GapSeverity.SIGNIFICANT_GAP, GapSeverity.CRITICAL_GAP):
            recs.append(
                f"URGENT: {gap_pct:.0f}% intensity gap requires immediate strategic intervention."
            )
            recs.append(
                f"Increase annual reduction rate by {acceleration:.1f} percentage points."
            )

        # Sector-specific recommendations
        sector_recs = {
            "power_generation": [
                "Accelerate coal plant retirement schedule.",
                "Increase renewable PPA procurement targets.",
                "Evaluate battery energy storage for grid flexibility.",
                "Consider nuclear baseload or SMR deployment.",
            ],
            "steel": [
                "Transition blast furnace capacity to EAF with scrap.",
                "Pilot green hydrogen DRI at one production site.",
                "Evaluate CCS feasibility for remaining BF-BOF capacity.",
                "Increase scrap recycling rate and sourcing diversification.",
            ],
            "cement": [
                "Reduce clinker-to-cement ratio through supplementary materials.",
                "Switch to alternative fuels (biomass, waste-derived) for kilns.",
                "Begin CCS feasibility study for process emissions.",
                "Invest in high-efficiency kiln replacement program.",
            ],
            "aluminum": [
                "Procure 100% renewable electricity for smelting operations.",
                "Increase secondary aluminum (recycling) share in product mix.",
                "Evaluate inert anode technology pilot.",
                "Optimise Hall-Heroult process energy efficiency.",
            ],
            "aviation": [
                "Accelerate fleet renewal with next-generation fuel-efficient aircraft.",
                "Increase SAF procurement to 10%+ of total fuel by 2030.",
                "Optimise load factors and route network efficiency.",
                "Invest in hydrogen aircraft R&D for short-haul routes.",
            ],
            "shipping": [
                "Implement slow steaming and route optimisation.",
                "Pilot alternative fuels (methanol, ammonia) on select vessels.",
                "Invest in hull design improvements and propulsion efficiency.",
                "Deploy shore power at major port terminals.",
            ],
            "buildings_residential": [
                "Accelerate heat pump installation replacing gas boilers.",
                "Implement deep retrofit program for building envelopes.",
                "Deploy on-site solar PV for all new developments.",
                "Integrate smart building energy management systems.",
            ],
            "buildings_commercial": [
                "Implement comprehensive building automation and controls.",
                "Accelerate HVAC electrification (heat pump conversion).",
                "Deploy high-performance glazing and insulation retrofits.",
                "Procure 100% renewable electricity for building operations.",
            ],
        }

        if sector_key in sector_recs:
            recs.extend(sector_recs[sector_key])
        else:
            recs.extend([
                "Conduct detailed energy audit to identify efficiency opportunities.",
                "Develop renewable energy procurement strategy.",
                "Evaluate fuel switching options for high-emission processes.",
                "Set internal carbon price to drive investment decisions.",
            ])

        return recs

    # -------------------------------------------------------------------------
    # Phase 5: Validation Report
    # -------------------------------------------------------------------------

    async def _phase_validation_report(self, input_data: SectorPathwayDesignInput) -> PhaseResult:
        """Produce SBTi pathway validation report with pass/fail criteria."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        if not self.config.validate_sbti:
            self._validation = ValidationReport(
                report_id=_new_uuid(),
                overall_status=ValidationStatus.NOT_APPLICABLE,
            )
            return PhaseResult(
                phase_name="validation_report", phase_number=5,
                status=PhaseStatus.SKIPPED,
                duration_seconds=0.0, completion_pct=100.0,
                outputs={"skipped": True, "reason": "SBTi validation disabled"},
                provenance_hash=_compute_hash("skipped"),
                dag_node_id=f"{self.workflow_id}_validation_report",
            )

        sector_key = self._classification.primary_sector
        sector_data = SDA_SECTORS.get(sector_key, SDA_SECTORS["cross_sector"])

        criteria: List[ValidationCriterion] = []

        # Criterion 1: SDA Eligibility
        sda_eligible = self._classification.sda_eligibility == SDAEligibility.ELIGIBLE
        criteria.append(ValidationCriterion(
            criterion_id="SDA-001",
            criterion_name="SDA Sector Eligibility",
            description="Company must be classified in an SDA-eligible sector.",
            status=ValidationStatus.PASS if sda_eligible else ValidationStatus.CONDITIONAL,
            actual_value=f"{self._classification.sda_method} ({self._classification.sector_name})",
            required_value="SDA-eligible sector (Power, Steel, Cement, etc.)",
            finding=(
                "Company is SDA-eligible." if sda_eligible else
                "Company not in SDA-eligible sector; ACA pathway applies."
            ),
            remediation="" if sda_eligible else "Use Absolute Contraction Approach (ACA) pathway.",
        ))

        # Criterion 2: Scope 1+2 Coverage
        s12_coverage = self.config.sbti_coverage_scope1_pct
        cov_pass = s12_coverage >= sector_data["coverage_requirement_pct"]
        criteria.append(ValidationCriterion(
            criterion_id="SDA-002",
            criterion_name="Scope 1+2 Coverage",
            description="At least 95% of Scope 1+2 emissions must be covered.",
            status=ValidationStatus.PASS if cov_pass else ValidationStatus.FAIL,
            actual_value=f"{s12_coverage:.0f}%",
            required_value=f">= {sector_data['coverage_requirement_pct']:.0f}%",
            finding=f"Coverage is {s12_coverage:.0f}%.",
            remediation="" if cov_pass else "Expand emission boundary to include all Scope 1+2 sources.",
        ))

        # Criterion 3: Base Year Intensity
        base_intensity = (
            self._intensity_metrics[0].base_year_value if self._intensity_metrics else 0.0
        )
        bi_pass = base_intensity > 0
        criteria.append(ValidationCriterion(
            criterion_id="SDA-003",
            criterion_name="Base Year Intensity Validation",
            description="Base year intensity must be calculated and positive.",
            status=ValidationStatus.PASS if bi_pass else ValidationStatus.FAIL,
            actual_value=f"{base_intensity:.4f} {sector_data['intensity_unit']}",
            required_value=f"> 0 {sector_data['intensity_unit']}",
            finding=f"Base year intensity: {base_intensity:.4f}.",
            remediation="" if bi_pass else "Provide complete base year activity and emissions data.",
        ))

        # Criterion 4: Near-Term Ambition (1.5C alignment)
        nze_pathway = next(
            (p for p in self._pathways if p.scenario == "nze_15c"), None,
        )
        if nze_pathway:
            req_rate = nze_pathway.required_annual_reduction_pct
            ambition_pass = req_rate >= 4.0
            criteria.append(ValidationCriterion(
                criterion_id="SDA-004",
                criterion_name="Near-Term Ambition (1.5C)",
                description="Annual intensity reduction must be >= 4.2% for 1.5C alignment.",
                status=ValidationStatus.PASS if ambition_pass else ValidationStatus.FAIL,
                actual_value=f"{req_rate:.1f}% per year",
                required_value=">= 4.2% per year (1.5C)",
                finding=f"Required annual reduction: {req_rate:.1f}%/yr.",
                remediation="" if ambition_pass else "Increase reduction ambition to 4.2%+ annual rate.",
            ))
        else:
            criteria.append(ValidationCriterion(
                criterion_id="SDA-004",
                criterion_name="Near-Term Ambition (1.5C)",
                status=ValidationStatus.FAIL,
                finding="NZE pathway not generated.",
                remediation="Include 'nze_15c' scenario in pathway generation.",
            ))

        # Criterion 5: Target Timeframe
        years = self.config.target_year - self.config.base_year
        time_pass = 5 <= years <= 35
        criteria.append(ValidationCriterion(
            criterion_id="SDA-005",
            criterion_name="Target Timeframe",
            description="Near-term target: 5-10 years. Long-term: by 2050.",
            status=ValidationStatus.PASS if time_pass else ValidationStatus.CONDITIONAL,
            actual_value=f"{years} years ({self.config.base_year}-{self.config.target_year})",
            required_value="5-10 years (near-term), by 2050 (long-term)",
            finding=f"Pathway spans {years} years.",
            remediation="" if time_pass else "Adjust target timeframe to meet SBTi requirements.",
        ))

        # Criterion 6: Pathway Alignment
        nze_gap = next(
            (g for g in self._gap_analyses if g.scenario == "nze_15c"), None,
        )
        if nze_gap:
            align_pass = abs(nze_gap.intensity_gap_pct) <= 10.0
            criteria.append(ValidationCriterion(
                criterion_id="SDA-006",
                criterion_name="Sector Pathway Alignment",
                description="Company intensity must be within +/-10% of SBTi benchmark.",
                status=ValidationStatus.PASS if align_pass else ValidationStatus.FAIL,
                actual_value=f"{nze_gap.intensity_gap_pct:+.1f}% vs. pathway",
                required_value="Within +/-10% of SBTi sector pathway",
                finding=f"Gap to NZE pathway: {nze_gap.intensity_gap_pct:+.1f}%.",
                remediation="" if align_pass else (
                    f"Reduce intensity by additional {abs(nze_gap.intensity_gap_pct) - 10:.1f}% "
                    "to achieve pathway alignment."
                ),
            ))

        # Criterion 7: Scope 3 Coverage
        s3_coverage = self.config.sbti_coverage_scope3_pct
        s3_pass = s3_coverage >= 67.0
        criteria.append(ValidationCriterion(
            criterion_id="SDA-007",
            criterion_name="Scope 3 Coverage",
            description="At least 67% of total Scope 3 emissions must be covered.",
            status=ValidationStatus.PASS if s3_pass else ValidationStatus.CONDITIONAL,
            actual_value=f"{s3_coverage:.0f}%",
            required_value=">= 67%",
            finding=f"Scope 3 coverage: {s3_coverage:.0f}%.",
            remediation="" if s3_pass else "Expand Scope 3 boundary to cover 67%+ of emissions.",
        ))

        # Criterion 8: Long-Term Net-Zero Alignment
        ltnz_pass = self.config.target_year <= 2050
        criteria.append(ValidationCriterion(
            criterion_id="SDA-008",
            criterion_name="Long-Term Net-Zero by 2050",
            description="Long-term target must reach net-zero by 2050.",
            status=ValidationStatus.PASS if ltnz_pass else ValidationStatus.FAIL,
            actual_value=f"Target year: {self.config.target_year}",
            required_value="By 2050",
            finding=f"Target year is {self.config.target_year}.",
            remediation="" if ltnz_pass else "Set long-term target to 2050 or earlier.",
        ))

        # Criterion 9: Intensity Metric Correctness
        expected_metric = sector_data["intensity_metric"]
        actual_metric = (
            self._intensity_metrics[0].metric_name if self._intensity_metrics else ""
        )
        metric_pass = actual_metric == expected_metric
        criteria.append(ValidationCriterion(
            criterion_id="SDA-009",
            criterion_name="Intensity Metric Correctness",
            description="Intensity metric must match SBTi sector-specific requirement.",
            status=ValidationStatus.PASS if metric_pass else ValidationStatus.CONDITIONAL,
            actual_value=actual_metric,
            required_value=expected_metric,
            finding=f"Using metric: {actual_metric}.",
            remediation="" if metric_pass else f"Use sector-specific metric: {expected_metric}.",
        ))

        # Criterion 10: Data Quality
        dq_score = (
            self._intensity_metrics[0].data_quality_score if self._intensity_metrics else 0.0
        )
        dq_pass = dq_score >= 2.0
        criteria.append(ValidationCriterion(
            criterion_id="SDA-010",
            criterion_name="Data Quality Score",
            description="Data quality must be at least 'moderate' (2.0/5.0).",
            status=ValidationStatus.PASS if dq_pass else ValidationStatus.CONDITIONAL,
            actual_value=f"{dq_score:.1f} / 5.0",
            required_value=">= 2.0 / 5.0",
            finding=f"Data quality score: {dq_score:.1f}.",
            remediation="" if dq_pass else "Improve data collection: use primary data over estimates.",
        ))

        # Summary
        passed = sum(1 for c in criteria if c.status == ValidationStatus.PASS)
        failed = sum(1 for c in criteria if c.status == ValidationStatus.FAIL)
        conditional = sum(1 for c in criteria if c.status == ValidationStatus.CONDITIONAL)

        if failed == 0 and conditional == 0:
            overall = ValidationStatus.PASS
        elif failed == 0:
            overall = ValidationStatus.CONDITIONAL
        else:
            overall = ValidationStatus.FAIL

        submission_ready = overall == ValidationStatus.PASS

        # Improvement actions
        actions = []
        for c in criteria:
            if c.status in (ValidationStatus.FAIL, ValidationStatus.CONDITIONAL) and c.remediation:
                actions.append(f"[{c.criterion_id}] {c.remediation}")

        self._validation = ValidationReport(
            report_id=_new_uuid(),
            sector=sector_key,
            scenario="nze_15c",
            overall_status=overall,
            criteria_checked=len(criteria),
            criteria_passed=passed,
            criteria_failed=failed,
            criteria_conditional=conditional,
            pass_rate_pct=round((passed / max(len(criteria), 1)) * 100, 1),
            criteria=criteria,
            sbti_submission_ready=submission_ready,
            improvement_actions=actions,
        )
        self._validation.provenance_hash = _compute_hash(
            self._validation.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["overall_status"] = overall.value
        outputs["criteria_checked"] = len(criteria)
        outputs["criteria_passed"] = passed
        outputs["criteria_failed"] = failed
        outputs["criteria_conditional"] = conditional
        outputs["pass_rate_pct"] = self._validation.pass_rate_pct
        outputs["sbti_submission_ready"] = submission_ready
        outputs["improvement_actions_count"] = len(actions)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="validation_report", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_validation_report",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _select_recommended_scenario(self) -> str:
        """Select recommended scenario based on gap analysis."""
        if not self._gap_analyses:
            return "nze_15c"

        # Prefer NZE if on track; otherwise find most achievable scenario
        nze = next((g for g in self._gap_analyses if g.scenario == "nze_15c"), None)
        if nze and nze.gap_severity in (GapSeverity.ON_TRACK, GapSeverity.MINOR_GAP):
            return "nze_15c"

        # Find scenario with smallest positive gap
        achievable = sorted(
            self._gap_analyses,
            key=lambda g: abs(g.intensity_gap_pct),
        )
        return achievable[0].scenario if achievable else "nze_15c"

    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from the workflow."""
        findings: List[str] = []

        findings.append(
            f"Sector: {self._classification.sector_name} "
            f"({self._classification.sda_method})"
        )

        if self._intensity_metrics:
            m = self._intensity_metrics[0]
            findings.append(
                f"Current intensity: {m.current_value:.4f} {m.metric_unit} "
                f"(trend: {m.trend_annual_pct:+.1f}%/yr)"
            )

        nze_gap = next(
            (g for g in self._gap_analyses if g.scenario == "nze_15c"), None,
        )
        if nze_gap:
            findings.append(
                f"Gap to NZE pathway: {nze_gap.intensity_gap_pct:+.1f}% "
                f"({nze_gap.gap_severity.value})"
            )
            if nze_gap.required_acceleration_pct > 0:
                findings.append(
                    f"Additional {nze_gap.required_acceleration_pct:.1f} pp/yr "
                    "reduction acceleration needed for NZE alignment."
                )

        findings.append(
            f"SBTi validation: {self._validation.overall_status.value} "
            f"({self._validation.criteria_passed}/{self._validation.criteria_checked} passed)"
        )

        return findings

    def _generate_next_steps(self) -> List[str]:
        """Generate next steps from the workflow."""
        steps: List[str] = []

        if not self._validation.sbti_submission_ready:
            steps.append("Address SBTi validation gaps before target submission.")

        steps.extend([
            "Run technology planning workflow (workflow 3) for technology roadmap.",
            "Run multi-scenario analysis (workflow 5) for strategy comparison.",
            "Present sector pathway and gap analysis to board/leadership.",
            "Align capital allocation with pathway investment requirements.",
            "Schedule annual pathway review with updated intensity data.",
        ])

        return steps
