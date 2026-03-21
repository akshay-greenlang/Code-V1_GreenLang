# -*- coding: utf-8 -*-
"""
Technology Planning Workflow
=================================

5-phase workflow for building sector-specific technology transition
roadmaps within PACK-028 Sector Pathway Pack.  The workflow inventories
current technologies, generates an adoption roadmap with IEA milestones,
maps CapEx requirements across investment phases, analyses technology
dependencies and risks, and produces an implementation plan.

Phases:
    1. TechInventory       -- Inventory current technology portfolio, assess
                              TRL levels, and map to sector pathway levers
    2. RoadmapGen          -- Generate technology adoption roadmap with
                              IEA NZE milestones and S-curve deployment
    3. CapExMapping        -- Map CapEx requirements by technology, phase,
                              and year with cost decline curves
    4. DependencyAnalysis  -- Analyse technology interdependencies, supply
                              chain risks, and critical path constraints
    5. ImplementationPlan  -- Produce sequenced implementation plan with
                              milestones, resource requirements, and KPIs

Regulatory references:
    - IEA Net Zero by 2050 Roadmap (2023 update) - Technology milestones
    - IEA Energy Technology Perspectives (2023)
    - SBTi Sector Guidance: Technology pathways per sector
    - IRENA Renewable Energy Technology Roadmaps

Zero-hallucination: all technology data, cost assumptions, and deployment
curves use deterministic lookups from IEA/IRENA published data tables.
No LLM calls in any computation path.

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
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _scurve(t: float, k: float = 0.3) -> float:
    """S-curve (logistic) function for technology adoption modelling."""
    return 1.0 / (1.0 + math.exp(-k * t))


def _gompertz(t: float, a: float = 1.0, b: float = 2.0, c: float = 0.3) -> float:
    """
    Gompertz curve for asymmetric technology adoption modelling.

    Suitable for technologies where early adoption is slow, mid-phase
    adoption is rapid, and late-phase adoption saturates.  The Gompertz
    curve is asymmetric (unlike the logistic S-curve) and better models
    technologies with a long R&D tail.

    Parameters:
        a: upper asymptote (maximum adoption, typically 1.0)
        b: displacement along the x-axis (affects inflection point)
        c: growth rate (higher = faster adoption)

    Returns:
        Adoption fraction between 0 and `a`.
    """
    return a * math.exp(-b * math.exp(-c * t))


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class TRL(int, Enum):
    """Technology Readiness Level (1-9 scale)."""
    TRL1 = 1   # Basic principles observed
    TRL2 = 2   # Technology concept formulated
    TRL3 = 3   # Experimental proof of concept
    TRL4 = 4   # Technology validated in lab
    TRL5 = 5   # Technology validated in relevant environment
    TRL6 = 6   # Technology demonstrated in relevant environment
    TRL7 = 7   # System prototype demonstrated
    TRL8 = 8   # System complete and qualified
    TRL9 = 9   # System proven in operational environment


class TechCategory(str, Enum):
    """Technology category classification."""
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_STORAGE = "energy_storage"
    ELECTRIFICATION = "electrification"
    HYDROGEN = "hydrogen"
    CCS_CCUS = "ccs_ccus"
    EFFICIENCY = "efficiency"
    FUEL_SWITCHING = "fuel_switching"
    PROCESS_INNOVATION = "process_innovation"
    CIRCULAR_ECONOMY = "circular_economy"
    DIGITAL = "digital"
    BIOENERGY = "bioenergy"
    NUCLEAR = "nuclear"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MilestoneStatus(str, Enum):
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"


class ImplementationPriority(str, Enum):
    IMMEDIATE = "immediate"     # 2025-2027
    SHORT_TERM = "short_term"   # 2027-2030
    MEDIUM_TERM = "medium_term" # 2030-2035
    LONG_TERM = "long_term"     # 2035-2050


# =============================================================================
# IEA NZE TECHNOLOGY MILESTONES (Zero-Hallucination: IEA Published Data)
# =============================================================================

IEA_SECTOR_TECHNOLOGIES: Dict[str, List[Dict[str, Any]]] = {
    "power_generation": [
        {"tech": "Solar PV", "category": "renewable_energy", "trl": 9, "2030_target_pct": 30.0,
         "2050_target_pct": 45.0, "cost_2020_usd_kw": 1000, "cost_2030_usd_kw": 450, "cost_2050_usd_kw": 300,
         "learning_rate": 0.28, "abatement_potential_pct": 25.0},
        {"tech": "Onshore Wind", "category": "renewable_energy", "trl": 9, "2030_target_pct": 20.0,
         "2050_target_pct": 25.0, "cost_2020_usd_kw": 1200, "cost_2030_usd_kw": 850, "cost_2050_usd_kw": 700,
         "learning_rate": 0.15, "abatement_potential_pct": 18.0},
        {"tech": "Offshore Wind", "category": "renewable_energy", "trl": 8, "2030_target_pct": 8.0,
         "2050_target_pct": 15.0, "cost_2020_usd_kw": 3500, "cost_2030_usd_kw": 2000, "cost_2050_usd_kw": 1400,
         "learning_rate": 0.18, "abatement_potential_pct": 10.0},
        {"tech": "Battery Storage (Li-ion)", "category": "energy_storage", "trl": 9, "2030_target_pct": 10.0,
         "2050_target_pct": 20.0, "cost_2020_usd_kw": 350, "cost_2030_usd_kw": 150, "cost_2050_usd_kw": 80,
         "learning_rate": 0.20, "abatement_potential_pct": 8.0},
        {"tech": "Green Hydrogen Electrolysis", "category": "hydrogen", "trl": 7, "2030_target_pct": 2.0,
         "2050_target_pct": 10.0, "cost_2020_usd_kw": 1400, "cost_2030_usd_kw": 700, "cost_2050_usd_kw": 350,
         "learning_rate": 0.12, "abatement_potential_pct": 5.0},
        {"tech": "Nuclear (SMR)", "category": "nuclear", "trl": 6, "2030_target_pct": 5.0,
         "2050_target_pct": 8.0, "cost_2020_usd_kw": 6500, "cost_2030_usd_kw": 5000, "cost_2050_usd_kw": 3500,
         "learning_rate": 0.05, "abatement_potential_pct": 8.0},
        {"tech": "CCS (Fossil Power)", "category": "ccs_ccus", "trl": 7, "2030_target_pct": 3.0,
         "2050_target_pct": 5.0, "cost_2020_usd_kw": 2000, "cost_2030_usd_kw": 1500, "cost_2050_usd_kw": 1000,
         "learning_rate": 0.08, "abatement_potential_pct": 4.0},
    ],
    "steel": [
        {"tech": "EAF with Scrap", "category": "electrification", "trl": 9, "2030_target_pct": 35.0,
         "2050_target_pct": 50.0, "cost_2020_usd_kw": 200, "cost_2030_usd_kw": 180, "cost_2050_usd_kw": 160,
         "learning_rate": 0.05, "abatement_potential_pct": 30.0},
        {"tech": "Green Hydrogen DRI", "category": "hydrogen", "trl": 6, "2030_target_pct": 5.0,
         "2050_target_pct": 30.0, "cost_2020_usd_kw": 800, "cost_2030_usd_kw": 500, "cost_2050_usd_kw": 300,
         "learning_rate": 0.15, "abatement_potential_pct": 25.0},
        {"tech": "CCS for BF-BOF", "category": "ccs_ccus", "trl": 7, "2030_target_pct": 5.0,
         "2050_target_pct": 15.0, "cost_2020_usd_kw": 600, "cost_2030_usd_kw": 450, "cost_2050_usd_kw": 350,
         "learning_rate": 0.08, "abatement_potential_pct": 15.0},
        {"tech": "Waste Heat Recovery", "category": "efficiency", "trl": 9, "2030_target_pct": 50.0,
         "2050_target_pct": 80.0, "cost_2020_usd_kw": 100, "cost_2030_usd_kw": 80, "cost_2050_usd_kw": 60,
         "learning_rate": 0.06, "abatement_potential_pct": 10.0},
        {"tech": "Renewable Electricity for EAF", "category": "renewable_energy", "trl": 9, "2030_target_pct": 60.0,
         "2050_target_pct": 100.0, "cost_2020_usd_kw": 50, "cost_2030_usd_kw": 30, "cost_2050_usd_kw": 20,
         "learning_rate": 0.20, "abatement_potential_pct": 15.0},
    ],
    "cement": [
        {"tech": "Alternative Fuels (Biomass/Waste)", "category": "fuel_switching", "trl": 9, "2030_target_pct": 30.0,
         "2050_target_pct": 60.0, "cost_2020_usd_kw": 50, "cost_2030_usd_kw": 40, "cost_2050_usd_kw": 35,
         "learning_rate": 0.04, "abatement_potential_pct": 15.0},
        {"tech": "Clinker Substitution (SCM)", "category": "process_innovation", "trl": 9, "2030_target_pct": 40.0,
         "2050_target_pct": 55.0, "cost_2020_usd_kw": 30, "cost_2030_usd_kw": 25, "cost_2050_usd_kw": 20,
         "learning_rate": 0.05, "abatement_potential_pct": 20.0},
        {"tech": "CCS for Process Emissions", "category": "ccs_ccus", "trl": 6, "2030_target_pct": 5.0,
         "2050_target_pct": 35.0, "cost_2020_usd_kw": 800, "cost_2030_usd_kw": 550, "cost_2050_usd_kw": 400,
         "learning_rate": 0.10, "abatement_potential_pct": 30.0},
        {"tech": "High-Efficiency Kilns", "category": "efficiency", "trl": 8, "2030_target_pct": 60.0,
         "2050_target_pct": 90.0, "cost_2020_usd_kw": 200, "cost_2030_usd_kw": 160, "cost_2050_usd_kw": 130,
         "learning_rate": 0.06, "abatement_potential_pct": 12.0},
        {"tech": "Low-Carbon Cement (Geopolymers)", "category": "process_innovation", "trl": 5, "2030_target_pct": 3.0,
         "2050_target_pct": 15.0, "cost_2020_usd_kw": 500, "cost_2030_usd_kw": 300, "cost_2050_usd_kw": 180,
         "learning_rate": 0.12, "abatement_potential_pct": 10.0},
    ],
    "aviation": [
        {"tech": "New-Gen Fuel-Efficient Aircraft", "category": "efficiency", "trl": 8, "2030_target_pct": 25.0,
         "2050_target_pct": 70.0, "cost_2020_usd_kw": 5000, "cost_2030_usd_kw": 4500, "cost_2050_usd_kw": 4000,
         "learning_rate": 0.03, "abatement_potential_pct": 20.0},
        {"tech": "SAF (Sustainable Aviation Fuel)", "category": "fuel_switching", "trl": 8, "2030_target_pct": 10.0,
         "2050_target_pct": 65.0, "cost_2020_usd_kw": 2500, "cost_2030_usd_kw": 1500, "cost_2050_usd_kw": 800,
         "learning_rate": 0.18, "abatement_potential_pct": 40.0},
        {"tech": "Hydrogen Aircraft (Short-Haul)", "category": "hydrogen", "trl": 4, "2030_target_pct": 0.0,
         "2050_target_pct": 15.0, "cost_2020_usd_kw": 10000, "cost_2030_usd_kw": 7000, "cost_2050_usd_kw": 4000,
         "learning_rate": 0.10, "abatement_potential_pct": 10.0},
        {"tech": "Electric Aircraft (Ultra-Short-Haul)", "category": "electrification", "trl": 5, "2030_target_pct": 2.0,
         "2050_target_pct": 8.0, "cost_2020_usd_kw": 8000, "cost_2030_usd_kw": 5000, "cost_2050_usd_kw": 3000,
         "learning_rate": 0.12, "abatement_potential_pct": 5.0},
        {"tech": "Operational Efficiency (ATM/Routes)", "category": "digital", "trl": 9, "2030_target_pct": 80.0,
         "2050_target_pct": 95.0, "cost_2020_usd_kw": 20, "cost_2030_usd_kw": 15, "cost_2050_usd_kw": 10,
         "learning_rate": 0.05, "abatement_potential_pct": 8.0},
    ],
    "shipping": [
        {"tech": "Hull Design & Propulsion Efficiency", "category": "efficiency", "trl": 8, "2030_target_pct": 40.0,
         "2050_target_pct": 70.0, "cost_2020_usd_kw": 300, "cost_2030_usd_kw": 250, "cost_2050_usd_kw": 200,
         "learning_rate": 0.06, "abatement_potential_pct": 15.0},
        {"tech": "Green Ammonia Fuel", "category": "fuel_switching", "trl": 6, "2030_target_pct": 5.0,
         "2050_target_pct": 35.0, "cost_2020_usd_kw": 1200, "cost_2030_usd_kw": 700, "cost_2050_usd_kw": 400,
         "learning_rate": 0.12, "abatement_potential_pct": 25.0},
        {"tech": "Green Methanol Fuel", "category": "fuel_switching", "trl": 7, "2030_target_pct": 8.0,
         "2050_target_pct": 25.0, "cost_2020_usd_kw": 900, "cost_2030_usd_kw": 550, "cost_2050_usd_kw": 350,
         "learning_rate": 0.10, "abatement_potential_pct": 18.0},
        {"tech": "Wind-Assisted Propulsion", "category": "renewable_energy", "trl": 7, "2030_target_pct": 10.0,
         "2050_target_pct": 25.0, "cost_2020_usd_kw": 400, "cost_2030_usd_kw": 300, "cost_2050_usd_kw": 200,
         "learning_rate": 0.08, "abatement_potential_pct": 10.0},
        {"tech": "Shore Power (Cold Ironing)", "category": "electrification", "trl": 9, "2030_target_pct": 50.0,
         "2050_target_pct": 90.0, "cost_2020_usd_kw": 150, "cost_2030_usd_kw": 120, "cost_2050_usd_kw": 90,
         "learning_rate": 0.05, "abatement_potential_pct": 5.0},
    ],
    "buildings_residential": [
        {"tech": "Heat Pumps (Air-Source)", "category": "electrification", "trl": 9, "2030_target_pct": 30.0,
         "2050_target_pct": 70.0, "cost_2020_usd_kw": 1200, "cost_2030_usd_kw": 800, "cost_2050_usd_kw": 550,
         "learning_rate": 0.12, "abatement_potential_pct": 30.0},
        {"tech": "Building Envelope Retrofit", "category": "efficiency", "trl": 9, "2030_target_pct": 20.0,
         "2050_target_pct": 60.0, "cost_2020_usd_kw": 200, "cost_2030_usd_kw": 150, "cost_2050_usd_kw": 120,
         "learning_rate": 0.05, "abatement_potential_pct": 25.0},
        {"tech": "Rooftop Solar PV", "category": "renewable_energy", "trl": 9, "2030_target_pct": 25.0,
         "2050_target_pct": 50.0, "cost_2020_usd_kw": 1500, "cost_2030_usd_kw": 750, "cost_2050_usd_kw": 450,
         "learning_rate": 0.22, "abatement_potential_pct": 15.0},
        {"tech": "Smart Building Controls", "category": "digital", "trl": 8, "2030_target_pct": 40.0,
         "2050_target_pct": 80.0, "cost_2020_usd_kw": 50, "cost_2030_usd_kw": 30, "cost_2050_usd_kw": 20,
         "learning_rate": 0.15, "abatement_potential_pct": 10.0},
        {"tech": "District Heating Integration", "category": "efficiency", "trl": 8, "2030_target_pct": 15.0,
         "2050_target_pct": 30.0, "cost_2020_usd_kw": 500, "cost_2030_usd_kw": 400, "cost_2050_usd_kw": 320,
         "learning_rate": 0.06, "abatement_potential_pct": 12.0},
    ],
    "chemicals": [
        {"tech": "Green Hydrogen for Ammonia", "category": "hydrogen", "trl": 7, "2030_target_pct": 10.0,
         "2050_target_pct": 50.0, "cost_2020_usd_kw": 1400, "cost_2030_usd_kw": 700, "cost_2050_usd_kw": 350,
         "learning_rate": 0.12, "abatement_potential_pct": 25.0},
        {"tech": "Electric Steam Crackers", "category": "electrification", "trl": 5, "2030_target_pct": 3.0,
         "2050_target_pct": 30.0, "cost_2020_usd_kw": 2000, "cost_2030_usd_kw": 1400, "cost_2050_usd_kw": 900,
         "learning_rate": 0.10, "abatement_potential_pct": 20.0},
        {"tech": "Mechanical/Chemical Recycling", "category": "circular_economy", "trl": 7, "2030_target_pct": 15.0,
         "2050_target_pct": 40.0, "cost_2020_usd_kw": 300, "cost_2030_usd_kw": 200, "cost_2050_usd_kw": 150,
         "learning_rate": 0.08, "abatement_potential_pct": 15.0},
        {"tech": "Bio-based Feedstocks", "category": "bioenergy", "trl": 6, "2030_target_pct": 5.0,
         "2050_target_pct": 20.0, "cost_2020_usd_kw": 500, "cost_2030_usd_kw": 350, "cost_2050_usd_kw": 250,
         "learning_rate": 0.07, "abatement_potential_pct": 10.0},
        {"tech": "CCS for Process Emissions", "category": "ccs_ccus", "trl": 6, "2030_target_pct": 5.0,
         "2050_target_pct": 25.0, "cost_2020_usd_kw": 900, "cost_2030_usd_kw": 600, "cost_2050_usd_kw": 400,
         "learning_rate": 0.09, "abatement_potential_pct": 20.0},
        {"tech": "Heat Pump Integration", "category": "efficiency", "trl": 8, "2030_target_pct": 25.0,
         "2050_target_pct": 60.0, "cost_2020_usd_kw": 400, "cost_2030_usd_kw": 300, "cost_2050_usd_kw": 220,
         "learning_rate": 0.08, "abatement_potential_pct": 12.0},
    ],
    "aluminum": [
        {"tech": "Inert Anode Technology", "category": "process_innovation", "trl": 5, "2030_target_pct": 5.0,
         "2050_target_pct": 50.0, "cost_2020_usd_kw": 3000, "cost_2030_usd_kw": 2000, "cost_2050_usd_kw": 1200,
         "learning_rate": 0.10, "abatement_potential_pct": 40.0},
        {"tech": "Renewable Electricity for Smelting", "category": "renewable_energy", "trl": 9, "2030_target_pct": 60.0,
         "2050_target_pct": 100.0, "cost_2020_usd_kw": 60, "cost_2030_usd_kw": 35, "cost_2050_usd_kw": 25,
         "learning_rate": 0.20, "abatement_potential_pct": 30.0},
        {"tech": "Secondary Aluminum Recycling", "category": "circular_economy", "trl": 9, "2030_target_pct": 40.0,
         "2050_target_pct": 60.0, "cost_2020_usd_kw": 80, "cost_2030_usd_kw": 70, "cost_2050_usd_kw": 60,
         "learning_rate": 0.04, "abatement_potential_pct": 25.0},
        {"tech": "Waste Heat Recovery (Smelter)", "category": "efficiency", "trl": 8, "2030_target_pct": 50.0,
         "2050_target_pct": 85.0, "cost_2020_usd_kw": 150, "cost_2030_usd_kw": 120, "cost_2050_usd_kw": 90,
         "learning_rate": 0.06, "abatement_potential_pct": 10.0},
        {"tech": "CCS for Alumina Refining", "category": "ccs_ccus", "trl": 6, "2030_target_pct": 3.0,
         "2050_target_pct": 15.0, "cost_2020_usd_kw": 700, "cost_2030_usd_kw": 500, "cost_2050_usd_kw": 350,
         "learning_rate": 0.08, "abatement_potential_pct": 8.0},
    ],
    "buildings_commercial": [
        {"tech": "Heat Pumps (Ground-Source)", "category": "electrification", "trl": 9, "2030_target_pct": 20.0,
         "2050_target_pct": 55.0, "cost_2020_usd_kw": 1800, "cost_2030_usd_kw": 1200, "cost_2050_usd_kw": 800,
         "learning_rate": 0.10, "abatement_potential_pct": 25.0},
        {"tech": "Deep Energy Retrofit", "category": "efficiency", "trl": 8, "2030_target_pct": 15.0,
         "2050_target_pct": 50.0, "cost_2020_usd_kw": 350, "cost_2030_usd_kw": 280, "cost_2050_usd_kw": 220,
         "learning_rate": 0.06, "abatement_potential_pct": 30.0},
        {"tech": "Building Management System (AI)", "category": "digital", "trl": 8, "2030_target_pct": 45.0,
         "2050_target_pct": 90.0, "cost_2020_usd_kw": 40, "cost_2030_usd_kw": 25, "cost_2050_usd_kw": 15,
         "learning_rate": 0.18, "abatement_potential_pct": 12.0},
        {"tech": "On-Site Solar + Storage", "category": "renewable_energy", "trl": 9, "2030_target_pct": 30.0,
         "2050_target_pct": 60.0, "cost_2020_usd_kw": 1200, "cost_2030_usd_kw": 600, "cost_2050_usd_kw": 400,
         "learning_rate": 0.22, "abatement_potential_pct": 15.0},
        {"tech": "Green Hydrogen Fuel Cell (Backup)", "category": "hydrogen", "trl": 6, "2030_target_pct": 3.0,
         "2050_target_pct": 15.0, "cost_2020_usd_kw": 2500, "cost_2030_usd_kw": 1500, "cost_2050_usd_kw": 800,
         "learning_rate": 0.14, "abatement_potential_pct": 5.0},
        {"tech": "District Cooling Network", "category": "efficiency", "trl": 8, "2030_target_pct": 10.0,
         "2050_target_pct": 25.0, "cost_2020_usd_kw": 600, "cost_2030_usd_kw": 500, "cost_2050_usd_kw": 400,
         "learning_rate": 0.05, "abatement_potential_pct": 8.0},
    ],
    "road_transport": [
        {"tech": "Battery Electric Vehicles (BEV)", "category": "electrification", "trl": 9, "2030_target_pct": 45.0,
         "2050_target_pct": 90.0, "cost_2020_usd_kw": 140, "cost_2030_usd_kw": 80, "cost_2050_usd_kw": 55,
         "learning_rate": 0.20, "abatement_potential_pct": 55.0},
        {"tech": "Hydrogen Fuel Cell Trucks", "category": "hydrogen", "trl": 7, "2030_target_pct": 5.0,
         "2050_target_pct": 25.0, "cost_2020_usd_kw": 500, "cost_2030_usd_kw": 300, "cost_2050_usd_kw": 180,
         "learning_rate": 0.12, "abatement_potential_pct": 15.0},
        {"tech": "Biofuels (Advanced HVO/FAME)", "category": "bioenergy", "trl": 8, "2030_target_pct": 15.0,
         "2050_target_pct": 10.0, "cost_2020_usd_kw": 200, "cost_2030_usd_kw": 150, "cost_2050_usd_kw": 120,
         "learning_rate": 0.06, "abatement_potential_pct": 8.0},
        {"tech": "Smart Fleet Management", "category": "digital", "trl": 9, "2030_target_pct": 60.0,
         "2050_target_pct": 95.0, "cost_2020_usd_kw": 15, "cost_2030_usd_kw": 10, "cost_2050_usd_kw": 7,
         "learning_rate": 0.15, "abatement_potential_pct": 10.0},
        {"tech": "EV Charging Infrastructure", "category": "electrification", "trl": 8, "2030_target_pct": 35.0,
         "2050_target_pct": 85.0, "cost_2020_usd_kw": 250, "cost_2030_usd_kw": 150, "cost_2050_usd_kw": 100,
         "learning_rate": 0.10, "abatement_potential_pct": 5.0},
    ],
    "oil_gas": [
        {"tech": "Methane Leak Detection & Repair (LDAR)", "category": "efficiency", "trl": 9, "2030_target_pct": 80.0,
         "2050_target_pct": 100.0, "cost_2020_usd_kw": 20, "cost_2030_usd_kw": 15, "cost_2050_usd_kw": 10,
         "learning_rate": 0.08, "abatement_potential_pct": 20.0},
        {"tech": "Electrification of Upstream Ops", "category": "electrification", "trl": 8, "2030_target_pct": 30.0,
         "2050_target_pct": 70.0, "cost_2020_usd_kw": 300, "cost_2030_usd_kw": 220, "cost_2050_usd_kw": 160,
         "learning_rate": 0.07, "abatement_potential_pct": 15.0},
        {"tech": "CCS for Gas Processing", "category": "ccs_ccus", "trl": 7, "2030_target_pct": 15.0,
         "2050_target_pct": 50.0, "cost_2020_usd_kw": 800, "cost_2030_usd_kw": 550, "cost_2050_usd_kw": 380,
         "learning_rate": 0.09, "abatement_potential_pct": 25.0},
        {"tech": "Flare Gas Recovery", "category": "efficiency", "trl": 9, "2030_target_pct": 90.0,
         "2050_target_pct": 100.0, "cost_2020_usd_kw": 50, "cost_2030_usd_kw": 40, "cost_2050_usd_kw": 35,
         "learning_rate": 0.04, "abatement_potential_pct": 10.0},
        {"tech": "Renewable Power for LNG", "category": "renewable_energy", "trl": 8, "2030_target_pct": 25.0,
         "2050_target_pct": 60.0, "cost_2020_usd_kw": 100, "cost_2030_usd_kw": 60, "cost_2050_usd_kw": 40,
         "learning_rate": 0.18, "abatement_potential_pct": 12.0},
    ],
}

# Technology maturity and risk assessment thresholds
TRL_RISK_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "production_ready": {
        "trl_min": 8, "risk_level": "low", "deployment_timeline": "immediate",
        "description": "Technology proven in operational environment; ready for full-scale deployment.",
        "investment_confidence": 0.95,
    },
    "demonstration": {
        "trl_min": 6, "risk_level": "medium", "deployment_timeline": "short_term",
        "description": "Technology demonstrated in relevant environment; pilot projects underway.",
        "investment_confidence": 0.75,
    },
    "pilot": {
        "trl_min": 4, "risk_level": "high", "deployment_timeline": "medium_term",
        "description": "Technology validated in lab/pilot; significant scale-up risk remains.",
        "investment_confidence": 0.50,
    },
    "research": {
        "trl_min": 1, "risk_level": "very_high", "deployment_timeline": "long_term",
        "description": "Basic principles observed; extensive R&D required before deployment.",
        "investment_confidence": 0.25,
    },
}

# Critical material supply chain data for technology risk assessment
CRITICAL_MATERIAL_DATABASE: Dict[str, Dict[str, Any]] = {
    "lithium": {
        "primary_producers": ["Australia", "Chile", "China"],
        "concentration_pct": 85, "price_volatility": "high",
        "substitutes": ["sodium-ion", "solid-state"],
        "recycling_rate_pct": 5,
        "projected_demand_growth_2030_pct": 500,
    },
    "cobalt": {
        "primary_producers": ["DR Congo", "Russia", "Australia"],
        "concentration_pct": 70, "price_volatility": "very_high",
        "substitutes": ["LFP chemistry", "cobalt-free NMC"],
        "recycling_rate_pct": 32,
        "projected_demand_growth_2030_pct": 300,
    },
    "rare_earths": {
        "primary_producers": ["China", "Myanmar", "Australia"],
        "concentration_pct": 60, "price_volatility": "high",
        "substitutes": ["ferrite magnets", "reduced-RE designs"],
        "recycling_rate_pct": 1,
        "projected_demand_growth_2030_pct": 200,
    },
    "platinum": {
        "primary_producers": ["South Africa", "Russia", "Zimbabwe"],
        "concentration_pct": 72, "price_volatility": "high",
        "substitutes": ["non-PGM catalysts", "PEM alternatives"],
        "recycling_rate_pct": 25,
        "projected_demand_growth_2030_pct": 400,
    },
    "iridium": {
        "primary_producers": ["South Africa"],
        "concentration_pct": 92, "price_volatility": "very_high",
        "substitutes": ["AEM electrolysers", "SOEC technology"],
        "recycling_rate_pct": 15,
        "projected_demand_growth_2030_pct": 600,
    },
    "silicon": {
        "primary_producers": ["China", "Norway", "Brazil"],
        "concentration_pct": 55, "price_volatility": "medium",
        "substitutes": ["perovskite PV", "thin-film CdTe"],
        "recycling_rate_pct": 10,
        "projected_demand_growth_2030_pct": 150,
    },
    "copper": {
        "primary_producers": ["Chile", "Peru", "China"],
        "concentration_pct": 40, "price_volatility": "medium",
        "substitutes": ["aluminium wiring (limited)"],
        "recycling_rate_pct": 45,
        "projected_demand_growth_2030_pct": 100,
    },
    "nickel": {
        "primary_producers": ["Indonesia", "Philippines", "Russia"],
        "concentration_pct": 50, "price_volatility": "high",
        "substitutes": ["LFP chemistry", "manganese-rich"],
        "recycling_rate_pct": 68,
        "projected_demand_growth_2030_pct": 180,
    },
}

# Default technologies for sectors without specific data
DEFAULT_TECHNOLOGIES: List[Dict[str, Any]] = [
    {"tech": "Energy Efficiency Improvements", "category": "efficiency", "trl": 9, "2030_target_pct": 40.0,
     "2050_target_pct": 70.0, "cost_2020_usd_kw": 100, "cost_2030_usd_kw": 80, "cost_2050_usd_kw": 60,
     "learning_rate": 0.05, "abatement_potential_pct": 20.0},
    {"tech": "Renewable Energy Procurement", "category": "renewable_energy", "trl": 9, "2030_target_pct": 50.0,
     "2050_target_pct": 100.0, "cost_2020_usd_kw": 60, "cost_2030_usd_kw": 35, "cost_2050_usd_kw": 25,
     "learning_rate": 0.20, "abatement_potential_pct": 30.0},
    {"tech": "Process Electrification", "category": "electrification", "trl": 7, "2030_target_pct": 20.0,
     "2050_target_pct": 60.0, "cost_2020_usd_kw": 500, "cost_2030_usd_kw": 350, "cost_2050_usd_kw": 250,
     "learning_rate": 0.10, "abatement_potential_pct": 25.0},
    {"tech": "Fuel Switching (Biomass/Biogas)", "category": "fuel_switching", "trl": 8, "2030_target_pct": 15.0,
     "2050_target_pct": 35.0, "cost_2020_usd_kw": 200, "cost_2030_usd_kw": 150, "cost_2050_usd_kw": 120,
     "learning_rate": 0.06, "abatement_potential_pct": 15.0},
    {"tech": "Digitalisation & IoT", "category": "digital", "trl": 8, "2030_target_pct": 50.0,
     "2050_target_pct": 90.0, "cost_2020_usd_kw": 30, "cost_2030_usd_kw": 18, "cost_2050_usd_kw": 12,
     "learning_rate": 0.15, "abatement_potential_pct": 8.0},
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class TechnologyItem(BaseModel):
    """A single technology in the inventory."""
    tech_id: str = Field(default="")
    name: str = Field(default="")
    category: str = Field(default="")
    trl: int = Field(default=1, ge=1, le=9)
    current_adoption_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_adoption_2030_pct: float = Field(default=0.0)
    target_adoption_2050_pct: float = Field(default=0.0)
    abatement_potential_pct: float = Field(default=0.0)
    cost_current_usd_kw: float = Field(default=0.0, ge=0.0)
    cost_2030_usd_kw: float = Field(default=0.0, ge=0.0)
    cost_2050_usd_kw: float = Field(default=0.0, ge=0.0)
    learning_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    maturity_risk: RiskLevel = Field(default=RiskLevel.MEDIUM)
    supply_chain_risk: RiskLevel = Field(default=RiskLevel.MEDIUM)
    deployment_readiness: str = Field(default="")


class RoadmapMilestone(BaseModel):
    """A single technology deployment milestone."""
    milestone_id: str = Field(default="")
    tech_name: str = Field(default="")
    year: int = Field(default=2025)
    target_adoption_pct: float = Field(default=0.0)
    cumulative_abatement_pct: float = Field(default=0.0)
    cost_usd: float = Field(default=0.0)
    iea_milestone: str = Field(default="")
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED)
    dependencies: List[str] = Field(default_factory=list)


class TechRoadmap(BaseModel):
    """Technology adoption roadmap."""
    sector: str = Field(default="")
    total_technologies: int = Field(default=0)
    milestones: List[RoadmapMilestone] = Field(default_factory=list)
    total_abatement_pct: float = Field(default=0.0)
    total_capex_usd: float = Field(default=0.0)
    deployment_phases: Dict[str, int] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class CapExAllocation(BaseModel):
    """CapEx allocation for a single technology and year."""
    tech_name: str = Field(default="")
    year: int = Field(default=2025)
    capex_usd: float = Field(default=0.0, ge=0.0)
    opex_delta_usd: float = Field(default=0.0)
    cumulative_capex_usd: float = Field(default=0.0, ge=0.0)
    cost_per_tco2e_abated: float = Field(default=0.0, ge=0.0)
    roi_pct: float = Field(default=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    carbon_price_breakeven_usd: float = Field(default=0.0, ge=0.0)


class CapExPlan(BaseModel):
    """Complete CapEx investment plan."""
    total_capex_usd: float = Field(default=0.0, ge=0.0)
    annual_allocations: Dict[int, float] = Field(default_factory=dict)
    allocations_by_tech: List[CapExAllocation] = Field(default_factory=list)
    capex_by_category: Dict[str, float] = Field(default_factory=dict)
    weighted_avg_cost_per_tco2e: float = Field(default=0.0)
    total_npv_usd: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class TechDependency(BaseModel):
    """A technology dependency relationship."""
    source_tech: str = Field(default="")
    target_tech: str = Field(default="")
    dependency_type: str = Field(default="", description="prerequisite|enabler|complementary")
    description: str = Field(default="")
    risk_if_delayed: RiskLevel = Field(default=RiskLevel.MEDIUM)
    lead_time_years: int = Field(default=0)


class SupplyChainRisk(BaseModel):
    """Supply chain risk assessment for a technology."""
    tech_name: str = Field(default="")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    risk_factors: List[str] = Field(default_factory=list)
    mitigation_actions: List[str] = Field(default_factory=list)
    critical_materials: List[str] = Field(default_factory=list)
    geographic_concentration: str = Field(default="")


class DependencyAnalysis(BaseModel):
    """Complete dependency and risk analysis."""
    dependencies: List[TechDependency] = Field(default_factory=list)
    critical_path: List[str] = Field(default_factory=list)
    supply_chain_risks: List[SupplyChainRisk] = Field(default_factory=list)
    overall_risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    bottlenecks: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ImplementationAction(BaseModel):
    """A single implementation action in the plan."""
    action_id: str = Field(default="")
    tech_name: str = Field(default="")
    action_description: str = Field(default="")
    priority: ImplementationPriority = Field(default=ImplementationPriority.MEDIUM_TERM)
    start_year: int = Field(default=2025)
    end_year: int = Field(default=2030)
    capex_usd: float = Field(default=0.0, ge=0.0)
    expected_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    kpi_target: str = Field(default="")
    responsible_team: str = Field(default="")
    dependencies: List[str] = Field(default_factory=list)


class ImplementationPlan(BaseModel):
    """Complete implementation plan."""
    sector: str = Field(default="")
    total_actions: int = Field(default=0)
    actions: List[ImplementationAction] = Field(default_factory=list)
    total_capex_usd: float = Field(default=0.0, ge=0.0)
    total_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    timeline_years: int = Field(default=0)
    phase_summary: Dict[str, int] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class TechnologyPlanningConfig(BaseModel):
    """Configuration for technology planning workflow."""
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    sector: str = Field(default="cross_sector")
    base_year: int = Field(default=2025, ge=2020, le=2035)
    target_year: int = Field(default=2050, ge=2030, le=2070)
    current_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    current_intensity: float = Field(default=0.0, ge=0.0)
    target_intensity: float = Field(default=0.0, ge=0.0)
    available_capex_usd: float = Field(default=0.0, ge=0.0)
    annual_capex_budget_usd: float = Field(default=0.0, ge=0.0)
    discount_rate: float = Field(default=0.08, ge=0.0, le=0.30)
    carbon_price_usd_per_tco2e: float = Field(default=100.0, ge=0.0)
    activity_level: float = Field(default=0.0, ge=0.0)
    current_tech_portfolio: Dict[str, float] = Field(
        default_factory=dict,
        description="Technology -> current adoption %",
    )


class TechnologyPlanningInput(BaseModel):
    config: TechnologyPlanningConfig = Field(default_factory=TechnologyPlanningConfig)
    custom_technologies: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)


class TechnologyPlanningResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="technology_planning")
    pack_id: str = Field(default="PACK-028")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    tech_inventory: List[TechnologyItem] = Field(default_factory=list)
    roadmap: TechRoadmap = Field(default_factory=TechRoadmap)
    capex_plan: CapExPlan = Field(default_factory=CapExPlan)
    dependency_analysis: DependencyAnalysis = Field(default_factory=DependencyAnalysis)
    implementation_plan: ImplementationPlan = Field(default_factory=ImplementationPlan)
    key_findings: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class TechnologyPlanningWorkflow:
    """
    5-phase technology planning workflow.

    Phase 1: TechInventory -- Inventory current technologies.
    Phase 2: RoadmapGen -- Generate adoption roadmap.
    Phase 3: CapExMapping -- Map CapEx requirements.
    Phase 4: DependencyAnalysis -- Analyse dependencies and risks.
    Phase 5: ImplementationPlan -- Produce implementation plan.

    Example:
        >>> wf = TechnologyPlanningWorkflow()
        >>> inp = TechnologyPlanningInput(
        ...     config=TechnologyPlanningConfig(sector="steel"),
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[TechnologyPlanningConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or TechnologyPlanningConfig()
        self._phase_results: List[PhaseResult] = []
        self._inventory: List[TechnologyItem] = []
        self._roadmap: TechRoadmap = TechRoadmap()
        self._capex_plan: CapExPlan = CapExPlan()
        self._dependency: DependencyAnalysis = DependencyAnalysis()
        self._impl_plan: ImplementationPlan = ImplementationPlan()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: TechnologyPlanningInput) -> TechnologyPlanningResult:
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting technology planning workflow %s, sector=%s",
            self.workflow_id, self.config.sector,
        )

        try:
            phase1 = await self._phase_tech_inventory(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_roadmap_gen(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_capex_mapping(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_dependency_analysis(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_implementation_plan(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Technology planning failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        result = TechnologyPlanningResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            tech_inventory=self._inventory,
            roadmap=self._roadmap,
            capex_plan=self._capex_plan,
            dependency_analysis=self._dependency,
            implementation_plan=self._impl_plan,
            key_findings=self._generate_findings(),
            next_steps=self._generate_next_steps(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Technology Inventory
    # -------------------------------------------------------------------------

    async def _phase_tech_inventory(self, input_data: TechnologyPlanningInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector = self.config.sector
        tech_list = IEA_SECTOR_TECHNOLOGIES.get(sector, DEFAULT_TECHNOLOGIES)

        # Add custom technologies
        for ct in input_data.custom_technologies:
            tech_list.append(ct)

        self._inventory = []
        for idx, td in enumerate(tech_list):
            current_adoption = self.config.current_tech_portfolio.get(td["tech"], 0.0)

            # Assess risks based on TRL
            trl_val = td.get("trl", 5)
            if trl_val >= 8:
                maturity_risk = RiskLevel.LOW
                readiness = "Production ready"
            elif trl_val >= 6:
                maturity_risk = RiskLevel.MEDIUM
                readiness = "Demonstration phase"
            elif trl_val >= 4:
                maturity_risk = RiskLevel.HIGH
                readiness = "Pilot phase"
            else:
                maturity_risk = RiskLevel.VERY_HIGH
                readiness = "R&D phase"

            # Supply chain risk heuristic
            cat = td.get("category", "")
            if cat in ("hydrogen", "ccs_ccus"):
                sc_risk = RiskLevel.HIGH
            elif cat in ("nuclear",):
                sc_risk = RiskLevel.VERY_HIGH
            elif cat in ("renewable_energy", "energy_storage"):
                sc_risk = RiskLevel.MEDIUM
            else:
                sc_risk = RiskLevel.LOW

            item = TechnologyItem(
                tech_id=f"TECH-{sector[:4].upper()}-{idx + 1:03d}",
                name=td["tech"],
                category=cat,
                trl=trl_val,
                current_adoption_pct=round(current_adoption, 1),
                target_adoption_2030_pct=td.get("2030_target_pct", 0.0),
                target_adoption_2050_pct=td.get("2050_target_pct", 0.0),
                abatement_potential_pct=td.get("abatement_potential_pct", 0.0),
                cost_current_usd_kw=td.get("cost_2020_usd_kw", 0.0),
                cost_2030_usd_kw=td.get("cost_2030_usd_kw", 0.0),
                cost_2050_usd_kw=td.get("cost_2050_usd_kw", 0.0),
                learning_rate=td.get("learning_rate", 0.0),
                maturity_risk=maturity_risk,
                supply_chain_risk=sc_risk,
                deployment_readiness=readiness,
            )
            self._inventory.append(item)

        outputs["technologies_inventoried"] = len(self._inventory)
        outputs["sector"] = sector
        outputs["categories"] = list(set(t.category for t in self._inventory))
        outputs["avg_trl"] = round(
            sum(t.trl for t in self._inventory) / max(len(self._inventory), 1), 1,
        )
        outputs["total_abatement_potential_pct"] = round(
            sum(t.abatement_potential_pct for t in self._inventory), 1,
        )
        outputs["high_risk_count"] = sum(
            1 for t in self._inventory if t.maturity_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)
        )

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="tech_inventory", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_tech_inventory",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Roadmap Generation
    # -------------------------------------------------------------------------

    async def _phase_roadmap_gen(self, input_data: TechnologyPlanningInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        milestones: List[RoadmapMilestone] = []
        total_abatement = 0.0
        total_cost = 0.0
        phase_counts: Dict[str, int] = {
            "immediate": 0, "short_term": 0, "medium_term": 0, "long_term": 0,
        }

        base_yr = self.config.base_year
        target_yr = self.config.target_year
        emissions = max(self.config.current_emissions_tco2e, 100000)

        for tech in self._inventory:
            # Generate milestones at key years
            milestone_years = [2027, 2030, 2035, 2040, 2045, 2050]
            milestone_years = [y for y in milestone_years if base_yr <= y <= target_yr]

            for year in milestone_years:
                # S-curve adoption calculation
                t_norm = (year - base_yr) / max(target_yr - base_yr, 1) * 10 - 5
                adoption_frac = _scurve(t_norm, k=0.5)

                if year <= 2030:
                    target_pct = tech.current_adoption_pct + (
                        tech.target_adoption_2030_pct - tech.current_adoption_pct
                    ) * adoption_frac * 2  # accelerate near-term
                else:
                    target_pct = tech.target_adoption_2030_pct + (
                        tech.target_adoption_2050_pct - tech.target_adoption_2030_pct
                    ) * adoption_frac

                target_pct = min(target_pct, tech.target_adoption_2050_pct)

                # Cumulative abatement
                cum_abatement = tech.abatement_potential_pct * (target_pct / 100.0)

                # Cost with learning rate
                years_from_base = year - 2020
                cost_factor = (1.0 - tech.learning_rate) ** (years_from_base / 5.0)
                unit_cost = tech.cost_current_usd_kw * cost_factor
                capacity_fraction = target_pct / 100.0
                cost = unit_cost * capacity_fraction * emissions * 0.01  # scale factor

                # IEA milestone mapping
                iea_milestone = ""
                if year == 2030 and tech.trl >= 8:
                    iea_milestone = f"IEA NZE 2030: {tech.name} at {tech.target_adoption_2030_pct:.0f}% deployment"
                elif year == 2050:
                    iea_milestone = f"IEA NZE 2050: {tech.name} at {tech.target_adoption_2050_pct:.0f}% deployment"

                # Status determination
                if year < self.config.base_year:
                    ms_status = MilestoneStatus.ACHIEVED
                elif year <= self.config.base_year + 2:
                    ms_status = MilestoneStatus.ON_TRACK if tech.trl >= 7 else MilestoneStatus.AT_RISK
                else:
                    ms_status = MilestoneStatus.NOT_STARTED

                # Phase classification
                if year <= 2027:
                    phase_counts["immediate"] += 1
                elif year <= 2030:
                    phase_counts["short_term"] += 1
                elif year <= 2035:
                    phase_counts["medium_term"] += 1
                else:
                    phase_counts["long_term"] += 1

                milestones.append(RoadmapMilestone(
                    milestone_id=f"MS-{tech.tech_id}-{year}",
                    tech_name=tech.name,
                    year=year,
                    target_adoption_pct=round(target_pct, 1),
                    cumulative_abatement_pct=round(cum_abatement, 2),
                    cost_usd=round(cost, 0),
                    iea_milestone=iea_milestone,
                    status=ms_status,
                ))

                total_abatement += cum_abatement
                total_cost += cost

        self._roadmap = TechRoadmap(
            sector=self.config.sector,
            total_technologies=len(self._inventory),
            milestones=milestones,
            total_abatement_pct=round(min(total_abatement, 100.0), 2),
            total_capex_usd=round(total_cost, 0),
            deployment_phases=phase_counts,
        )
        self._roadmap.provenance_hash = _compute_hash(
            self._roadmap.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["milestones_generated"] = len(milestones)
        outputs["total_abatement_pct"] = self._roadmap.total_abatement_pct
        outputs["total_capex_usd"] = self._roadmap.total_capex_usd
        outputs["deployment_phases"] = phase_counts

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="roadmap_gen", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_roadmap_gen",
        )

    # -------------------------------------------------------------------------
    # Phase 3: CapEx Mapping
    # -------------------------------------------------------------------------

    async def _phase_capex_mapping(self, input_data: TechnologyPlanningInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        allocations: List[CapExAllocation] = []
        annual_totals: Dict[int, float] = {}
        category_totals: Dict[str, float] = {}
        total_capex = 0.0
        total_npv = 0.0
        emissions = max(self.config.current_emissions_tco2e, 100000)
        discount = self.config.discount_rate
        carbon_price = self.config.carbon_price_usd_per_tco2e

        for tech in self._inventory:
            # Calculate CapEx by year
            milestone_years = sorted(set(
                m.year for m in self._roadmap.milestones if m.tech_name == tech.name
            ))

            cumulative = 0.0
            for year in milestone_years:
                milestone = next(
                    (m for m in self._roadmap.milestones
                     if m.tech_name == tech.name and m.year == year),
                    None,
                )
                if not milestone:
                    continue

                capex = milestone.cost_usd
                abatement_tco2e = (
                    tech.abatement_potential_pct / 100.0 *
                    milestone.target_adoption_pct / 100.0 *
                    emissions
                )

                cost_per_tco2e = capex / max(abatement_tco2e, 1.0)
                opex_delta = capex * -0.05  # 5% OpEx saving (typical for efficiency)

                # Carbon savings value
                carbon_savings = abatement_tco2e * carbon_price
                years_remaining = max(self.config.target_year - year, 1)
                roi = (carbon_savings * years_remaining - capex) / max(capex, 1.0) * 100

                # Payback
                annual_savings = carbon_savings + abs(opex_delta)
                payback = capex / max(annual_savings, 1.0)

                # Carbon price breakeven
                breakeven = capex / max(abatement_tco2e * years_remaining, 1.0)

                cumulative += capex
                total_capex += capex

                # NPV
                t = year - self.config.base_year
                npv_factor = 1.0 / ((1 + discount) ** t)
                total_npv += (carbon_savings - capex * 0.1) * npv_factor  # annualise capex

                annual_totals[year] = annual_totals.get(year, 0.0) + capex
                category_totals[tech.category] = category_totals.get(tech.category, 0.0) + capex

                allocations.append(CapExAllocation(
                    tech_name=tech.name,
                    year=year,
                    capex_usd=round(capex, 0),
                    opex_delta_usd=round(opex_delta, 0),
                    cumulative_capex_usd=round(cumulative, 0),
                    cost_per_tco2e_abated=round(cost_per_tco2e, 2),
                    roi_pct=round(roi, 1),
                    payback_years=round(payback, 1),
                    carbon_price_breakeven_usd=round(breakeven, 2),
                ))

        # Weighted average cost per tCO2e
        total_abatement_calc = sum(
            a.capex_usd / max(a.cost_per_tco2e_abated, 1.0) for a in allocations
            if a.cost_per_tco2e_abated > 0
        )
        wavg = total_capex / max(total_abatement_calc, 1.0) if total_abatement_calc > 0 else 0

        self._capex_plan = CapExPlan(
            total_capex_usd=round(total_capex, 0),
            annual_allocations={k: round(v, 0) for k, v in sorted(annual_totals.items())},
            allocations_by_tech=allocations,
            capex_by_category={k: round(v, 0) for k, v in sorted(category_totals.items())},
            weighted_avg_cost_per_tco2e=round(wavg, 2),
            total_npv_usd=round(total_npv, 0),
        )
        self._capex_plan.provenance_hash = _compute_hash(
            self._capex_plan.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["total_capex_usd"] = self._capex_plan.total_capex_usd
        outputs["total_npv_usd"] = self._capex_plan.total_npv_usd
        outputs["allocations_count"] = len(allocations)
        outputs["categories"] = list(category_totals.keys())
        outputs["wavg_cost_per_tco2e"] = self._capex_plan.weighted_avg_cost_per_tco2e
        outputs["budget_utilisation_pct"] = round(
            total_capex / max(self.config.available_capex_usd, 1.0) * 100, 1,
        ) if self.config.available_capex_usd > 0 else 0.0

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="capex_mapping", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_capex_mapping",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Dependency Analysis
    # -------------------------------------------------------------------------

    async def _phase_dependency_analysis(self, input_data: TechnologyPlanningInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        dependencies: List[TechDependency] = []
        supply_risks: List[SupplyChainRisk] = []
        bottlenecks: List[str] = []
        critical_path: List[str] = []

        # Build dependency graph based on sector
        sector = self.config.sector
        tech_names = [t.name for t in self._inventory]

        # Generic dependency patterns
        dep_patterns: Dict[str, List[Dict[str, str]]] = {
            "steel": [
                {"source": "Renewable Electricity for EAF", "target": "EAF with Scrap",
                 "type": "prerequisite", "desc": "EAF requires renewable electricity for zero-carbon steel"},
                {"source": "Green Hydrogen Electrolysis", "target": "Green Hydrogen DRI",
                 "type": "prerequisite", "desc": "DRI requires green hydrogen production capacity"},
            ],
            "cement": [
                {"source": "High-Efficiency Kilns", "target": "CCS for Process Emissions",
                 "type": "enabler", "desc": "Efficient kilns reduce CCS capture volume requirements"},
                {"source": "Alternative Fuels (Biomass/Waste)", "target": "Clinker Substitution (SCM)",
                 "type": "complementary", "desc": "Both reduce clinker production needs"},
            ],
            "aviation": [
                {"source": "SAF (Sustainable Aviation Fuel)", "target": "New-Gen Fuel-Efficient Aircraft",
                 "type": "complementary", "desc": "Fuel-efficient aircraft also optimised for SAF blends"},
                {"source": "Green Hydrogen Electrolysis", "target": "Hydrogen Aircraft (Short-Haul)",
                 "type": "prerequisite", "desc": "Hydrogen aircraft require green hydrogen supply"},
            ],
            "shipping": [
                {"source": "Green Hydrogen Electrolysis", "target": "Green Ammonia Fuel",
                 "type": "prerequisite", "desc": "Ammonia production requires green hydrogen"},
                {"source": "Hull Design & Propulsion Efficiency", "target": "Wind-Assisted Propulsion",
                 "type": "complementary", "desc": "Hull optimisation improves wind-assist effectiveness"},
            ],
            "power_generation": [
                {"source": "Solar PV", "target": "Battery Storage (Li-ion)",
                 "type": "enabler", "desc": "Storage enables higher solar PV penetration"},
                {"source": "Green Hydrogen Electrolysis", "target": "Nuclear (SMR)",
                 "type": "complementary", "desc": "SMR can provide baseload for hydrogen production"},
            ],
            "buildings_residential": [
                {"source": "Building Envelope Retrofit", "target": "Heat Pumps (Air-Source)",
                 "type": "enabler", "desc": "Insulation improves heat pump COP and reduces sizing"},
                {"source": "Rooftop Solar PV", "target": "Smart Building Controls",
                 "type": "complementary", "desc": "Controls optimise solar self-consumption"},
            ],
        }

        sector_deps = dep_patterns.get(sector, [])
        for dep in sector_deps:
            if dep["source"] in tech_names or dep["target"] in tech_names:
                dependencies.append(TechDependency(
                    source_tech=dep["source"],
                    target_tech=dep["target"],
                    dependency_type=dep["type"],
                    description=dep["desc"],
                    risk_if_delayed=(
                        RiskLevel.HIGH if dep["type"] == "prerequisite" else RiskLevel.MEDIUM
                    ),
                    lead_time_years=3 if dep["type"] == "prerequisite" else 1,
                ))

        # Supply chain risks
        critical_material_map: Dict[str, List[str]] = {
            "renewable_energy": ["silicon", "rare earths", "copper"],
            "energy_storage": ["lithium", "cobalt", "nickel", "graphite"],
            "hydrogen": ["platinum", "iridium", "PEM membranes"],
            "electrification": ["copper", "rare earth magnets"],
            "ccs_ccus": ["specialty steel", "amine solvents"],
            "nuclear": ["uranium", "zirconium alloys"],
        }

        for tech in self._inventory:
            materials = critical_material_map.get(tech.category, [])
            risk_factors = []
            mitigations = []

            if tech.maturity_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH):
                risk_factors.append(f"Low TRL ({tech.trl}): technology maturity risk")
                mitigations.append("Invest in R&D partnerships and pilot projects.")

            if materials:
                risk_factors.append(f"Critical materials: {', '.join(materials)}")
                mitigations.append("Diversify supply sources and evaluate recycling options.")

            if tech.category == "hydrogen":
                risk_factors.append("Green hydrogen supply chain nascent globally")
                mitigations.append("Secure long-term hydrogen supply contracts.")
                geo = "Concentrated in regions with cheap renewables"
            elif tech.category == "energy_storage":
                risk_factors.append("Battery supply chain concentration (China, DRC)")
                mitigations.append("Evaluate alternative chemistries (sodium-ion, solid-state).")
                geo = "High concentration in China and DR Congo"
            elif tech.category == "nuclear":
                risk_factors.append("Nuclear regulatory and licensing constraints")
                mitigations.append("Engage regulators early; evaluate SMR pre-licensing.")
                geo = "Limited to approved jurisdictions"
            else:
                geo = "Global supply available"

            supply_risks.append(SupplyChainRisk(
                tech_name=tech.name,
                risk_level=tech.supply_chain_risk,
                risk_factors=risk_factors,
                mitigation_actions=mitigations,
                critical_materials=materials,
                geographic_concentration=geo,
            ))

        # Identify bottlenecks
        for dep in dependencies:
            if dep.dependency_type == "prerequisite":
                source_tech = next(
                    (t for t in self._inventory if t.name == dep.source_tech), None,
                )
                if source_tech and source_tech.trl < 7:
                    bottlenecks.append(
                        f"{dep.source_tech} (TRL {source_tech.trl}) is prerequisite for "
                        f"{dep.target_tech} but not yet demonstration-ready."
                    )

        # Critical path (prerequisite chain ordering)
        prerequisite_targets = set(d.target_tech for d in dependencies if d.dependency_type == "prerequisite")
        prerequisite_sources = set(d.source_tech for d in dependencies if d.dependency_type == "prerequisite")
        foundation_techs = prerequisite_sources - prerequisite_targets

        critical_path = list(foundation_techs)
        for dep in dependencies:
            if dep.dependency_type == "prerequisite" and dep.target_tech not in critical_path:
                critical_path.append(dep.target_tech)

        # Overall risk
        high_risk_count = sum(
            1 for t in self._inventory if t.maturity_risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)
        )
        if high_risk_count >= 3:
            overall_risk = RiskLevel.HIGH
        elif high_risk_count >= 1:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        self._dependency = DependencyAnalysis(
            dependencies=dependencies,
            critical_path=critical_path,
            supply_chain_risks=supply_risks,
            overall_risk_level=overall_risk,
            bottlenecks=bottlenecks,
        )
        self._dependency.provenance_hash = _compute_hash(
            self._dependency.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["dependencies_count"] = len(dependencies)
        outputs["critical_path_length"] = len(critical_path)
        outputs["supply_risks_count"] = len(supply_risks)
        outputs["bottlenecks_count"] = len(bottlenecks)
        outputs["overall_risk"] = overall_risk.value
        outputs["critical_path"] = critical_path[:5]

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="dependency_analysis", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_dependency_analysis",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Implementation Plan
    # -------------------------------------------------------------------------

    async def _phase_implementation_plan(self, input_data: TechnologyPlanningInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        actions: List[ImplementationAction] = []
        total_capex = 0.0
        total_abatement = 0.0
        phase_summary: Dict[str, int] = {
            "immediate": 0, "short_term": 0, "medium_term": 0, "long_term": 0,
        }

        emissions = max(self.config.current_emissions_tco2e, 100000)

        for idx, tech in enumerate(self._inventory):
            # Determine priority based on TRL, abatement potential, and cost
            if tech.trl >= 8 and tech.abatement_potential_pct >= 15:
                priority = ImplementationPriority.IMMEDIATE
                start_yr = self.config.base_year
                end_yr = self.config.base_year + 3
            elif tech.trl >= 7 and tech.abatement_potential_pct >= 10:
                priority = ImplementationPriority.SHORT_TERM
                start_yr = self.config.base_year + 2
                end_yr = 2030
            elif tech.trl >= 5:
                priority = ImplementationPriority.MEDIUM_TERM
                start_yr = 2030
                end_yr = 2035
            else:
                priority = ImplementationPriority.LONG_TERM
                start_yr = 2035
                end_yr = 2050

            phase_summary[priority.value] += 1

            # Calculate expected abatement
            expected_abatement = (
                tech.abatement_potential_pct / 100.0 *
                tech.target_adoption_2050_pct / 100.0 *
                emissions
            )

            # Estimated CapEx
            capex = tech.cost_current_usd_kw * expected_abatement * 0.01

            # Dependencies from analysis
            deps = [
                d.source_tech for d in self._dependency.dependencies
                if d.target_tech == tech.name and d.dependency_type == "prerequisite"
            ]

            # KPI target
            kpi = f"Deploy {tech.name} to {tech.target_adoption_2030_pct:.0f}% by 2030"

            # Responsible team
            team_map = {
                "renewable_energy": "Energy Procurement",
                "electrification": "Operations / Engineering",
                "hydrogen": "R&D / Innovation",
                "ccs_ccus": "R&D / Engineering",
                "efficiency": "Operations / Facilities",
                "fuel_switching": "Energy / Supply Chain",
                "process_innovation": "R&D / Production",
                "digital": "IT / Operations",
                "circular_economy": "Sustainability / Supply Chain",
                "bioenergy": "Energy / Sustainability",
                "nuclear": "Strategic Planning",
                "energy_storage": "Energy / Grid Ops",
            }

            actions.append(ImplementationAction(
                action_id=f"ACT-{idx + 1:03d}",
                tech_name=tech.name,
                action_description=(
                    f"Deploy {tech.name} from {tech.current_adoption_pct:.0f}% to "
                    f"{tech.target_adoption_2030_pct:.0f}% adoption by 2030, "
                    f"targeting {tech.target_adoption_2050_pct:.0f}% by 2050."
                ),
                priority=priority,
                start_year=start_yr,
                end_year=end_yr,
                capex_usd=round(capex, 0),
                expected_abatement_tco2e=round(expected_abatement, 0),
                kpi_target=kpi,
                responsible_team=team_map.get(tech.category, "Sustainability"),
                dependencies=deps,
            ))

            total_capex += capex
            total_abatement += expected_abatement

        timeline = max(
            (a.end_year for a in actions), default=self.config.target_year,
        ) - min(
            (a.start_year for a in actions), default=self.config.base_year,
        )

        self._impl_plan = ImplementationPlan(
            sector=self.config.sector,
            total_actions=len(actions),
            actions=actions,
            total_capex_usd=round(total_capex, 0),
            total_abatement_tco2e=round(total_abatement, 0),
            timeline_years=timeline,
            phase_summary=phase_summary,
        )
        self._impl_plan.provenance_hash = _compute_hash(
            self._impl_plan.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["total_actions"] = len(actions)
        outputs["total_capex_usd"] = self._impl_plan.total_capex_usd
        outputs["total_abatement_tco2e"] = self._impl_plan.total_abatement_tco2e
        outputs["timeline_years"] = timeline
        outputs["phase_summary"] = phase_summary
        outputs["immediate_actions"] = phase_summary.get("immediate", 0)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="implementation_plan", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_implementation_plan",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _generate_findings(self) -> List[str]:
        findings: List[str] = []
        findings.append(
            f"Technology inventory: {len(self._inventory)} technologies across "
            f"{len(set(t.category for t in self._inventory))} categories."
        )
        findings.append(
            f"Total potential abatement: {self._roadmap.total_abatement_pct:.1f}% "
            f"with {self._capex_plan.total_capex_usd:,.0f} USD total CapEx."
        )
        if self._dependency.bottlenecks:
            findings.append(
                f"RISK: {len(self._dependency.bottlenecks)} technology bottleneck(s) identified."
            )
        immediate = sum(
            1 for a in self._impl_plan.actions
            if a.priority == ImplementationPriority.IMMEDIATE
        )
        if immediate > 0:
            findings.append(f"{immediate} immediate-priority actions ready for deployment.")
        findings.append(
            f"Weighted average abatement cost: {self._capex_plan.weighted_avg_cost_per_tco2e:.0f} "
            f"USD/tCO2e."
        )
        return findings

    def _generate_next_steps(self) -> List[str]:
        return [
            "Present technology roadmap and CapEx plan to board for approval.",
            "Initiate procurement for immediate-priority technologies.",
            "Establish R&D partnerships for pre-commercial technologies.",
            "Develop detailed engineering studies for top 3 CapEx items.",
            "Set up quarterly milestone tracking against IEA NZE targets.",
            "Integrate technology roadmap into PACK-028 sector pathway design.",
        ]

    def _assess_supply_chain_risk_detailed(
        self, tech: TechnologyItem,
    ) -> Tuple[RiskLevel, List[str], List[str], List[str], str]:
        """
        Perform a detailed supply chain risk assessment for a single technology
        using the CRITICAL_MATERIAL_DATABASE.

        Returns:
            Tuple of (risk_level, risk_factors, mitigations, materials, geo_concentration)
        """
        cat = tech.category
        critical_material_map: Dict[str, List[str]] = {
            "renewable_energy": ["silicon", "rare_earths", "copper"],
            "energy_storage": ["lithium", "cobalt", "nickel"],
            "hydrogen": ["platinum", "iridium"],
            "electrification": ["copper", "rare_earths"],
            "ccs_ccus": [],
            "nuclear": [],
            "efficiency": [],
            "fuel_switching": [],
            "process_innovation": [],
            "circular_economy": [],
            "digital": ["rare_earths"],
            "bioenergy": [],
        }

        material_names = critical_material_map.get(cat, [])
        risk_factors: List[str] = []
        mitigations: List[str] = []
        max_concentration = 0

        for mat_name in material_names:
            mat_data = CRITICAL_MATERIAL_DATABASE.get(mat_name, {})
            if not mat_data:
                continue

            conc = mat_data.get("concentration_pct", 0)
            max_concentration = max(max_concentration, conc)
            vol = mat_data.get("price_volatility", "medium")
            growth = mat_data.get("projected_demand_growth_2030_pct", 0)
            recycle = mat_data.get("recycling_rate_pct", 0)
            subs = mat_data.get("substitutes", [])
            producers = mat_data.get("primary_producers", [])

            if conc > 70:
                risk_factors.append(
                    f"{mat_name}: extreme supply concentration ({conc}% from "
                    f"{', '.join(producers[:2])})"
                )
            elif conc > 50:
                risk_factors.append(
                    f"{mat_name}: high supply concentration ({conc}%)"
                )

            if vol in ("high", "very_high"):
                risk_factors.append(
                    f"{mat_name}: {vol} price volatility"
                )

            if growth > 200:
                risk_factors.append(
                    f"{mat_name}: projected demand growth {growth}% by 2030 "
                    "may cause supply shortages"
                )

            if recycle < 20:
                mitigations.append(
                    f"Invest in {mat_name} recycling infrastructure "
                    f"(current rate: {recycle}%)."
                )
            if subs:
                mitigations.append(
                    f"Evaluate {mat_name} substitutes: {', '.join(subs[:2])}."
                )

        # TRL-based risk
        if tech.trl <= 4:
            risk_factors.append(
                f"Low technology readiness (TRL {tech.trl}): "
                "significant scale-up and manufacturing risk."
            )
            mitigations.append(
                "Fund pilot-scale manufacturing studies and "
                "secure development stage off-take agreements."
            )
        elif tech.trl <= 6:
            risk_factors.append(
                f"Moderate technology readiness (TRL {tech.trl}): "
                "demonstration-phase manufacturing risk."
            )
            mitigations.append(
                "Partner with technology providers for "
                "co-development and risk sharing."
            )

        # Determine overall risk level
        if max_concentration > 80 or len(risk_factors) >= 4:
            risk_level = RiskLevel.VERY_HIGH
        elif max_concentration > 60 or len(risk_factors) >= 3:
            risk_level = RiskLevel.HIGH
        elif max_concentration > 40 or len(risk_factors) >= 2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Geographic concentration string
        if material_names:
            all_producers: List[str] = []
            for mn in material_names:
                md = CRITICAL_MATERIAL_DATABASE.get(mn, {})
                all_producers.extend(md.get("primary_producers", []))
            unique_geo = list(dict.fromkeys(all_producers))[:4]
            geo = f"Supply concentrated in: {', '.join(unique_geo)}"
        else:
            geo = "Global supply available; low geographic concentration risk."

        if not mitigations:
            mitigations.append("Standard procurement risk management sufficient.")

        return risk_level, risk_factors, mitigations, material_names, geo

    def _compute_npv_detailed(
        self, cash_flows: List[Tuple[int, float]], discount_rate: float,
    ) -> float:
        """
        Compute net present value for a series of (year, cash_flow) tuples.

        Uses standard DCF formula: NPV = sum( CF_t / (1+r)^t )
        where t is years from the base year.
        """
        base_year = self.config.base_year
        npv = 0.0
        for year, cf in cash_flows:
            t = max(year - base_year, 0)
            discount_factor = (1.0 + discount_rate) ** t
            npv += cf / max(discount_factor, 1e-10)
        return round(npv, 2)

    def _calculate_levelized_cost(
        self, tech: TechnologyItem, emissions: float,
    ) -> Dict[str, float]:
        """
        Calculate the Levelized Cost of Carbon Abatement (LCCA) for a
        technology over its deployment lifetime.

        Returns a dict with lcca_usd_per_tco2e, annual_abatement_tco2e,
        lifetime_abatement_tco2e, total_investment_usd, and irr_estimate_pct.
        """
        base_yr = self.config.base_year
        target_yr = self.config.target_year
        lifetime = max(target_yr - base_yr, 1)
        discount = self.config.discount_rate

        # Annual abatement
        annual_abatement = (
            tech.abatement_potential_pct / 100.0 *
            tech.target_adoption_2050_pct / 100.0 *
            emissions
        )

        # Total investment (with learning rate decline)
        total_investment = 0.0
        investment_years = min(10, lifetime)
        for yr_offset in range(investment_years):
            year = base_yr + yr_offset
            learning_factor = (1.0 - tech.learning_rate) ** (yr_offset / 3.0)
            annual_invest = (
                tech.cost_current_usd_kw * learning_factor *
                annual_abatement * 0.01 / max(investment_years, 1)
            )
            total_investment += annual_invest

        # Discounted abatement
        discounted_abatement = 0.0
        for yr_offset in range(lifetime):
            adoption_frac = min(yr_offset / max(lifetime, 1), 1.0)
            annual_ab = annual_abatement * adoption_frac
            discount_factor = (1.0 + discount) ** yr_offset
            discounted_abatement += annual_ab / max(discount_factor, 1e-10)

        # LCCA
        lcca = total_investment / max(discounted_abatement, 1.0)

        # Simplified IRR estimate
        lifetime_abatement = annual_abatement * lifetime * 0.6  # ramp factor
        carbon_price = self.config.carbon_price_usd_per_tco2e
        total_revenue = lifetime_abatement * carbon_price
        irr = ((total_revenue / max(total_investment, 1.0)) ** (1.0 / lifetime) - 1.0) * 100

        return {
            "lcca_usd_per_tco2e": round(lcca, 2),
            "annual_abatement_tco2e": round(annual_abatement, 0),
            "lifetime_abatement_tco2e": round(lifetime_abatement, 0),
            "total_investment_usd": round(total_investment, 0),
            "irr_estimate_pct": round(max(irr, -100), 1),
        }

    def _prioritize_technologies(
        self, technologies: List[TechnologyItem],
    ) -> List[Tuple[TechnologyItem, float]]:
        """
        Rank technologies by a composite priority score combining:
          - Abatement potential (30% weight)
          - Technology readiness (25% weight)
          - Cost effectiveness (25% weight)
          - Deployment timeline (20% weight)

        Returns list of (technology, score) tuples sorted by descending score.
        """
        scored: List[Tuple[TechnologyItem, float]] = []

        for tech in technologies:
            # Abatement score (0-100)
            max_abatement = max(
                (t.abatement_potential_pct for t in technologies), default=1,
            )
            abatement_score = (tech.abatement_potential_pct / max(max_abatement, 1)) * 100

            # TRL score (0-100, higher TRL = higher score)
            trl_score = (tech.trl / 9.0) * 100

            # Cost effectiveness score (0-100, lower cost = higher score)
            max_cost = max(
                (t.cost_current_usd_kw for t in technologies), default=1,
            )
            if max_cost > 0:
                cost_score = (1.0 - tech.cost_current_usd_kw / max_cost) * 100
            else:
                cost_score = 50.0

            # Timeline score (higher near-term target = higher score)
            timeline_score = min(tech.target_adoption_2030_pct, 100.0)

            # Composite score
            composite = (
                abatement_score * 0.30 +
                trl_score * 0.25 +
                cost_score * 0.25 +
                timeline_score * 0.20
            )

            scored.append((tech, round(composite, 2)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
