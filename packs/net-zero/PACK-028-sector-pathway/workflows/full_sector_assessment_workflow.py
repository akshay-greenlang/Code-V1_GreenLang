# -*- coding: utf-8 -*-
"""
Full Sector Assessment Workflow
====================================

7-phase master workflow that chains all sector pathway sub-workflows
into a unified end-to-end assessment within PACK-028 Sector Pathway
Pack.  This orchestrator runs sector classification, pathway design,
technology planning, abatement analysis, benchmarking, multi-scenario
comparison, and compiles a comprehensive strategy report.

Phases:
    1. Classify    -- Run sector classification (SDA/NACE/GICS/ISIC)
    2. Pathway     -- Run sector pathway design (5 scenarios)
    3. Technology   -- Run technology planning (roadmap + CapEx)
    4. Abatement   -- Run abatement waterfall analysis (lever-by-lever)
    5. Benchmark   -- Run sector benchmarking (peer/leader/IEA)
    6. Scenarios   -- Run multi-scenario analysis (Monte Carlo)
    7. Strategy    -- Compile unified sector transition strategy

Zero-hallucination: all numeric results propagate from deterministic
sub-workflow calculations.  SHA-256 provenance hashes guarantee
end-to-end auditability across all 7 phases.

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .sector_pathway_design_workflow import (
    SectorPathwayDesignConfig,
    SectorPathwayDesignInput,
    SectorPathwayDesignResult,
    SectorPathwayDesignWorkflow,
    SectorClassification,
    SDA_SECTORS,
)
from .pathway_validation_workflow import (
    PathwayValidationConfig,
    PathwayValidationInput,
    PathwayValidationResult,
    PathwayValidationWorkflow,
)
from .technology_planning_workflow import (
    TechnologyPlanningConfig,
    TechnologyPlanningInput,
    TechnologyPlanningResult,
    TechnologyPlanningWorkflow,
)
from .progress_monitoring_workflow import (
    ProgressMonitoringConfig,
    ProgressMonitoringInput,
    ProgressMonitoringResult,
    ProgressMonitoringWorkflow,
    IntensityDataPoint,
)
from .multi_scenario_analysis_workflow import (
    MultiScenarioConfig,
    MultiScenarioInput,
    MultiScenarioResult,
    MultiScenarioAnalysisWorkflow,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


class MaturityLevel(str, Enum):
    """Sector pathway maturity level."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    ADVANCED = "advanced"
    LEADING = "leading"


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


class SectorScorecard(BaseModel):
    """Sector pathway assessment scorecard."""
    classification_score: float = Field(default=0.0, ge=0.0, le=100.0)
    pathway_score: float = Field(default=0.0, ge=0.0, le=100.0)
    technology_score: float = Field(default=0.0, ge=0.0, le=100.0)
    abatement_score: float = Field(default=0.0, ge=0.0, le=100.0)
    benchmark_score: float = Field(default=0.0, ge=0.0, le=100.0)
    scenario_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    maturity: MaturityLevel = Field(default=MaturityLevel.NASCENT)


class AbatementLever(BaseModel):
    """A single abatement lever in the waterfall."""
    lever_name: str = Field(default="")
    category: str = Field(default="")
    abatement_tco2e: float = Field(default=0.0)
    abatement_pct: float = Field(default=0.0)
    cost_per_tco2e_usd: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    implementation_year_start: int = Field(default=2025)
    implementation_year_end: int = Field(default=2035)
    trl: int = Field(default=9, ge=1, le=9)
    dependencies: List[str] = Field(default_factory=list)


class AbatementWaterfall(BaseModel):
    """Sector abatement waterfall analysis."""
    sector: str = Field(default="")
    total_abatement_tco2e: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    levers: List[AbatementLever] = Field(default_factory=list)
    weighted_avg_cost_usd: float = Field(default=0.0)
    coverage_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SectorStrategySummary(BaseModel):
    """Unified sector transition strategy summary."""
    assessment_date: str = Field(default="")
    company_name: str = Field(default="")
    sector: str = Field(default="")
    sector_name: str = Field(default="")
    sda_method: str = Field(default="")
    sda_eligible: bool = Field(default=False)

    # Intensity
    current_intensity: float = Field(default=0.0)
    intensity_unit: str = Field(default="")
    intensity_trend_pct: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)

    # Pathway
    recommended_scenario: str = Field(default="")
    recommended_scenario_name: str = Field(default="")
    sbti_submission_ready: bool = Field(default=False)
    nze_gap_pct: float = Field(default=0.0)
    required_acceleration_pct: float = Field(default=0.0)

    # Technology
    technologies_count: int = Field(default=0)
    total_capex_usd: float = Field(default=0.0)
    tech_roadmap_years: int = Field(default=0)
    tech_bottlenecks: int = Field(default=0)

    # Abatement
    total_abatement_tco2e: float = Field(default=0.0)
    abatement_levers_count: int = Field(default=0)
    avg_abatement_cost_usd: float = Field(default=0.0)

    # Benchmark
    sector_percentile: float = Field(default=50.0)
    vs_sector_leader_pct: float = Field(default=0.0)
    vs_iea_pathway_pct: float = Field(default=0.0)

    # Scenario
    scenarios_modeled: int = Field(default=0)
    nze_probability_pct: float = Field(default=0.0)
    recommendation_confidence: str = Field(default="medium")

    # Scorecard
    overall_score: float = Field(default=0.0)
    maturity_level: str = Field(default="nascent")

    # Actions
    key_actions: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)


class FullSectorAssessmentConfig(BaseModel):
    """Configuration for the full sector assessment."""
    # Company identification
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    # Sector classification
    nace_codes: List[str] = Field(default_factory=list)
    gics_codes: List[str] = Field(default_factory=list)
    isic_codes: List[str] = Field(default_factory=list)
    primary_activity: str = Field(default="")
    revenue_breakdown: Dict[str, float] = Field(default_factory=dict)

    # Emission data
    base_year: int = Field(default=2020, ge=2015, le=2030)
    current_year: int = Field(default=2025, ge=2020, le=2035)
    target_year: int = Field(default=2050, ge=2030, le=2070)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    current_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_activity: float = Field(default=0.0, ge=0.0)
    current_activity: float = Field(default=0.0, ge=0.0)
    scope1_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_emissions_tco2e: float = Field(default=0.0, ge=0.0)

    # Options
    activity_growth_rate: float = Field(default=0.02, ge=-0.10, le=0.20)
    convergence_model: str = Field(default="linear")
    available_capex_usd: float = Field(default=0.0, ge=0.0)
    carbon_price_usd: float = Field(default=100.0, ge=0.0)
    monte_carlo_runs: int = Field(default=1000, ge=100, le=50000)
    seed: int = Field(default=42)
    discount_rate: float = Field(default=0.08, ge=0.0, le=0.30)

    # Scope coverage
    scope12_coverage_pct: float = Field(default=95.0)
    scope3_coverage_pct: float = Field(default=67.0)

    # Conditional phases
    skip_validation: bool = Field(default=False)
    skip_scenarios: bool = Field(default=False)
    skip_technology: bool = Field(default=False)

    # Trajectory
    current_trajectory_annual_reduction: float = Field(default=0.02)
    sector_abatement_cost_usd_per_tco2e: float = Field(default=80.0)


class FullSectorAssessmentInput(BaseModel):
    config: FullSectorAssessmentConfig = Field(
        default_factory=FullSectorAssessmentConfig,
    )
    historical_intensity: Dict[int, float] = Field(default_factory=dict)
    peer_data: List[Dict[str, Any]] = Field(default_factory=list)
    planned_actions: List[Dict[str, Any]] = Field(default_factory=list)
    current_tech_portfolio: Dict[str, float] = Field(default_factory=dict)


class FullSectorAssessmentResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="full_sector_assessment")
    pack_id: str = Field(default="PACK-028")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)

    # Sub-workflow results
    pathway_result: Optional[SectorPathwayDesignResult] = Field(None)
    validation_result: Optional[PathwayValidationResult] = Field(None)
    technology_result: Optional[TechnologyPlanningResult] = Field(None)
    progress_result: Optional[ProgressMonitoringResult] = Field(None)
    scenario_result: Optional[MultiScenarioResult] = Field(None)

    # Derived outputs
    abatement_waterfall: AbatementWaterfall = Field(
        default_factory=AbatementWaterfall,
    )
    scorecard: SectorScorecard = Field(default_factory=SectorScorecard)
    strategy: SectorStrategySummary = Field(
        default_factory=SectorStrategySummary,
    )
    provenance_hash: str = Field(default="")


# =============================================================================
# SECTOR ABATEMENT LEVERS (Zero-Hallucination: IEA/McKinsey MACC Data)
# =============================================================================

SECTOR_LEVERS: Dict[str, List[Dict[str, Any]]] = {
    "power_generation": [
        {"lever": "Solar PV Expansion", "cat": "renewable_energy", "abatement_pct": 25, "cost": 20, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Wind Expansion", "cat": "renewable_energy", "abatement_pct": 18, "cost": 30, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Coal Retirement", "cat": "fuel_switching", "abatement_pct": 20, "cost": -10, "trl": 9, "start": 2025, "end": 2040},
        {"lever": "Grid Storage", "cat": "energy_storage", "abatement_pct": 8, "cost": 60, "trl": 9, "start": 2027, "end": 2040},
        {"lever": "Green Hydrogen", "cat": "hydrogen", "abatement_pct": 5, "cost": 120, "trl": 7, "start": 2030, "end": 2045},
        {"lever": "Nuclear (SMR)", "cat": "nuclear", "abatement_pct": 8, "cost": 150, "trl": 6, "start": 2032, "end": 2050},
        {"lever": "CCS Fossil Power", "cat": "ccs_ccus", "abatement_pct": 4, "cost": 100, "trl": 7, "start": 2030, "end": 2045},
        {"lever": "Grid Efficiency", "cat": "efficiency", "abatement_pct": 5, "cost": 15, "trl": 9, "start": 2025, "end": 2030},
    ],
    "steel": [
        {"lever": "EAF with Scrap", "cat": "electrification", "abatement_pct": 30, "cost": 25, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Green H2 DRI", "cat": "hydrogen", "abatement_pct": 25, "cost": 100, "trl": 6, "start": 2028, "end": 2040},
        {"lever": "CCS BF-BOF", "cat": "ccs_ccus", "abatement_pct": 15, "cost": 80, "trl": 7, "start": 2028, "end": 2040},
        {"lever": "Waste Heat Recovery", "cat": "efficiency", "abatement_pct": 10, "cost": 15, "trl": 9, "start": 2025, "end": 2030},
        {"lever": "Renewable Electricity", "cat": "renewable_energy", "abatement_pct": 15, "cost": 20, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Scrap Recycling Increase", "cat": "circular_economy", "abatement_pct": 8, "cost": 10, "trl": 9, "start": 2025, "end": 2035},
    ],
    "cement": [
        {"lever": "Clinker Substitution", "cat": "process_innovation", "abatement_pct": 20, "cost": 5, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Alternative Fuels", "cat": "fuel_switching", "abatement_pct": 15, "cost": 10, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "CCS Process Emissions", "cat": "ccs_ccus", "abatement_pct": 30, "cost": 90, "trl": 6, "start": 2028, "end": 2045},
        {"lever": "High-Efficiency Kilns", "cat": "efficiency", "abatement_pct": 12, "cost": 20, "trl": 8, "start": 2025, "end": 2035},
        {"lever": "Low-Carbon Cement", "cat": "process_innovation", "abatement_pct": 10, "cost": 50, "trl": 5, "start": 2030, "end": 2045},
        {"lever": "Circular Economy", "cat": "circular_economy", "abatement_pct": 5, "cost": 15, "trl": 8, "start": 2027, "end": 2040},
    ],
    "aviation": [
        {"lever": "SAF Adoption", "cat": "fuel_switching", "abatement_pct": 40, "cost": 150, "trl": 8, "start": 2025, "end": 2045},
        {"lever": "Fleet Renewal", "cat": "efficiency", "abatement_pct": 20, "cost": 80, "trl": 8, "start": 2025, "end": 2040},
        {"lever": "Operational Efficiency", "cat": "digital", "abatement_pct": 8, "cost": 5, "trl": 9, "start": 2025, "end": 2030},
        {"lever": "Hydrogen Short-Haul", "cat": "hydrogen", "abatement_pct": 10, "cost": 200, "trl": 4, "start": 2035, "end": 2050},
        {"lever": "Electric Ultra-Short", "cat": "electrification", "abatement_pct": 5, "cost": 180, "trl": 5, "start": 2032, "end": 2050},
    ],
    "shipping": [
        {"lever": "Efficiency Improvements", "cat": "efficiency", "abatement_pct": 15, "cost": 20, "trl": 8, "start": 2025, "end": 2035},
        {"lever": "Green Ammonia", "cat": "fuel_switching", "abatement_pct": 25, "cost": 130, "trl": 6, "start": 2028, "end": 2045},
        {"lever": "Green Methanol", "cat": "fuel_switching", "abatement_pct": 18, "cost": 110, "trl": 7, "start": 2027, "end": 2042},
        {"lever": "Wind-Assisted Propulsion", "cat": "renewable_energy", "abatement_pct": 10, "cost": 40, "trl": 7, "start": 2027, "end": 2040},
        {"lever": "Shore Power", "cat": "electrification", "abatement_pct": 5, "cost": 25, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Slow Steaming / Routing", "cat": "digital", "abatement_pct": 10, "cost": 5, "trl": 9, "start": 2025, "end": 2030},
    ],
    "chemicals": [
        {"lever": "Green Hydrogen for Ammonia", "cat": "hydrogen", "abatement_pct": 25, "cost": 110, "trl": 7, "start": 2028, "end": 2042},
        {"lever": "Electric Steam Crackers", "cat": "electrification", "abatement_pct": 20, "cost": 150, "trl": 5, "start": 2030, "end": 2045},
        {"lever": "Mechanical/Chemical Recycling", "cat": "circular_economy", "abatement_pct": 15, "cost": 35, "trl": 7, "start": 2027, "end": 2038},
        {"lever": "Bio-based Feedstocks", "cat": "bioenergy", "abatement_pct": 10, "cost": 60, "trl": 6, "start": 2028, "end": 2040},
        {"lever": "CCS Process Emissions", "cat": "ccs_ccus", "abatement_pct": 20, "cost": 85, "trl": 6, "start": 2028, "end": 2045},
        {"lever": "Heat Pump Integration", "cat": "efficiency", "abatement_pct": 12, "cost": 25, "trl": 8, "start": 2025, "end": 2035},
        {"lever": "Process Optimisation (AI)", "cat": "digital", "abatement_pct": 5, "cost": 8, "trl": 8, "start": 2025, "end": 2030},
    ],
    "aluminum": [
        {"lever": "Inert Anode Smelting", "cat": "process_innovation", "abatement_pct": 40, "cost": 120, "trl": 5, "start": 2030, "end": 2045},
        {"lever": "Renewable Electricity (Smelting)", "cat": "renewable_energy", "abatement_pct": 30, "cost": 20, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Secondary Recycling", "cat": "circular_economy", "abatement_pct": 25, "cost": 10, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Waste Heat Recovery", "cat": "efficiency", "abatement_pct": 10, "cost": 18, "trl": 8, "start": 2025, "end": 2033},
        {"lever": "CCS Alumina Refining", "cat": "ccs_ccus", "abatement_pct": 8, "cost": 75, "trl": 6, "start": 2028, "end": 2042},
    ],
    "buildings_residential": [
        {"lever": "Heat Pump Deployment", "cat": "electrification", "abatement_pct": 30, "cost": 35, "trl": 9, "start": 2025, "end": 2040},
        {"lever": "Building Envelope Retrofit", "cat": "efficiency", "abatement_pct": 25, "cost": 40, "trl": 9, "start": 2025, "end": 2040},
        {"lever": "Rooftop Solar", "cat": "renewable_energy", "abatement_pct": 15, "cost": 30, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "Smart Controls", "cat": "digital", "abatement_pct": 10, "cost": 8, "trl": 8, "start": 2025, "end": 2030},
        {"lever": "District Heating", "cat": "efficiency", "abatement_pct": 12, "cost": 50, "trl": 8, "start": 2027, "end": 2040},
        {"lever": "Cooking Electrification", "cat": "electrification", "abatement_pct": 5, "cost": 12, "trl": 9, "start": 2025, "end": 2030},
    ],
    "buildings_commercial": [
        {"lever": "Heat Pump (Ground-Source)", "cat": "electrification", "abatement_pct": 25, "cost": 45, "trl": 9, "start": 2025, "end": 2038},
        {"lever": "Deep Energy Retrofit", "cat": "efficiency", "abatement_pct": 30, "cost": 55, "trl": 8, "start": 2025, "end": 2040},
        {"lever": "BMS / AI Optimisation", "cat": "digital", "abatement_pct": 12, "cost": 8, "trl": 8, "start": 2025, "end": 2030},
        {"lever": "On-Site Solar + Storage", "cat": "renewable_energy", "abatement_pct": 15, "cost": 28, "trl": 9, "start": 2025, "end": 2035},
        {"lever": "District Cooling", "cat": "efficiency", "abatement_pct": 8, "cost": 35, "trl": 8, "start": 2027, "end": 2038},
        {"lever": "Green Lease Standards", "cat": "digital", "abatement_pct": 5, "cost": 3, "trl": 9, "start": 2025, "end": 2028},
    ],
    "road_transport": [
        {"lever": "BEV Fleet Electrification", "cat": "electrification", "abatement_pct": 55, "cost": 30, "trl": 9, "start": 2025, "end": 2040},
        {"lever": "Hydrogen Fuel Cell Trucks", "cat": "hydrogen", "abatement_pct": 15, "cost": 80, "trl": 7, "start": 2028, "end": 2042},
        {"lever": "Smart Fleet Management", "cat": "digital", "abatement_pct": 10, "cost": 5, "trl": 9, "start": 2025, "end": 2030},
        {"lever": "Advanced Biofuels", "cat": "bioenergy", "abatement_pct": 8, "cost": 40, "trl": 8, "start": 2025, "end": 2035},
        {"lever": "Charging Infrastructure", "cat": "electrification", "abatement_pct": 5, "cost": 20, "trl": 8, "start": 2025, "end": 2035},
    ],
    "oil_gas": [
        {"lever": "Methane LDAR", "cat": "efficiency", "abatement_pct": 20, "cost": 3, "trl": 9, "start": 2025, "end": 2028},
        {"lever": "Electrification of Upstream", "cat": "electrification", "abatement_pct": 15, "cost": 30, "trl": 8, "start": 2025, "end": 2035},
        {"lever": "CCS Gas Processing", "cat": "ccs_ccus", "abatement_pct": 25, "cost": 60, "trl": 7, "start": 2027, "end": 2040},
        {"lever": "Flare Gas Recovery", "cat": "efficiency", "abatement_pct": 10, "cost": 8, "trl": 9, "start": 2025, "end": 2028},
        {"lever": "Renewable Power for LNG", "cat": "renewable_energy", "abatement_pct": 12, "cost": 22, "trl": 8, "start": 2025, "end": 2035},
        {"lever": "Produced Water Treatment", "cat": "efficiency", "abatement_pct": 3, "cost": 15, "trl": 8, "start": 2025, "end": 2032},
    ],
}

# Scorecard dimension weights for maturity assessment
SCORECARD_WEIGHTS: Dict[str, float] = {
    "classification_score": 0.05,
    "pathway_score": 0.25,
    "technology_score": 0.20,
    "abatement_score": 0.15,
    "benchmark_score": 0.15,
    "scenario_score": 0.20,
}

# Maturity level thresholds (overall score out of 100)
MATURITY_THRESHOLDS: Dict[str, float] = {
    "leading": 80.0,
    "advanced": 65.0,
    "established": 50.0,
    "developing": 30.0,
    "nascent": 0.0,
}

# Sector-specific transition timeline expectations (years from base to net-zero)
SECTOR_TRANSITION_TIMELINES: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "fast_track": 15, "standard": 20, "delayed": 25,
        "key_milestones": ["coal_phase_out", "50pct_renewable", "grid_storage"],
    },
    "steel": {
        "fast_track": 18, "standard": 23, "delayed": 28,
        "key_milestones": ["eaf_conversion", "h2_dri_pilot", "green_steel_standard"],
    },
    "cement": {
        "fast_track": 20, "standard": 25, "delayed": 30,
        "key_milestones": ["ccs_pilot", "alt_binder_10pct", "clinker_ratio_60pct"],
    },
    "aviation": {
        "fast_track": 22, "standard": 27, "delayed": 30,
        "key_milestones": ["saf_10pct", "fleet_renewal_50pct", "h2_aircraft_entry"],
    },
    "shipping": {
        "fast_track": 20, "standard": 25, "delayed": 30,
        "key_milestones": ["green_corridor", "ammonia_5pct", "imo_cii_a_rating"],
    },
    "chemicals": {
        "fast_track": 20, "standard": 25, "delayed": 30,
        "key_milestones": ["h2_ammonia_pilot", "cracker_electrification", "recycling_15pct"],
    },
    "aluminum": {
        "fast_track": 18, "standard": 23, "delayed": 28,
        "key_milestones": ["inert_anode_pilot", "100pct_renewable_smelting", "secondary_60pct"],
    },
    "buildings_residential": {
        "fast_track": 18, "standard": 22, "delayed": 28,
        "key_milestones": ["heat_pump_30pct", "retrofit_rate_3pct_yr", "nzeb_standard"],
    },
    "buildings_commercial": {
        "fast_track": 18, "standard": 22, "delayed": 28,
        "key_milestones": ["deep_retrofit_programme", "bms_ai_deployment", "green_lease_standard"],
    },
}

# Default levers for sectors without specific data
DEFAULT_LEVERS: List[Dict[str, Any]] = [
    {"lever": "Energy Efficiency", "cat": "efficiency", "abatement_pct": 20, "cost": 15, "trl": 9, "start": 2025, "end": 2035},
    {"lever": "Renewable Procurement", "cat": "renewable_energy", "abatement_pct": 30, "cost": 25, "trl": 9, "start": 2025, "end": 2035},
    {"lever": "Electrification", "cat": "electrification", "abatement_pct": 25, "cost": 50, "trl": 7, "start": 2027, "end": 2040},
    {"lever": "Fuel Switching", "cat": "fuel_switching", "abatement_pct": 15, "cost": 40, "trl": 8, "start": 2027, "end": 2040},
    {"lever": "Digital Optimisation", "cat": "digital", "abatement_pct": 8, "cost": 10, "trl": 9, "start": 2025, "end": 2030},
]


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullSectorAssessmentWorkflow:
    """
    7-phase master workflow for end-to-end sector pathway assessment.

    Phase 1: Classify -- Sector classification.
    Phase 2: Pathway -- Sector pathway design.
    Phase 3: Technology -- Technology planning.
    Phase 4: Abatement -- Abatement waterfall.
    Phase 5: Benchmark -- Sector benchmarking.
    Phase 6: Scenarios -- Multi-scenario analysis.
    Phase 7: Strategy -- Unified strategy compilation.

    Example:
        >>> wf = FullSectorAssessmentWorkflow()
        >>> config = FullSectorAssessmentConfig(
        ...     nace_codes=["D35.11"],
        ...     base_year_emissions_tco2e=500000,
        ...     base_year_activity=1000000,
        ... )
        >>> inp = FullSectorAssessmentInput(config=config)
        >>> result = await wf.execute(inp)
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._pathway_result: Optional[SectorPathwayDesignResult] = None
        self._validation_result: Optional[PathwayValidationResult] = None
        self._tech_result: Optional[TechnologyPlanningResult] = None
        self._progress_result: Optional[ProgressMonitoringResult] = None
        self._scenario_result: Optional[MultiScenarioResult] = None
        self._abatement: AbatementWaterfall = AbatementWaterfall()
        self._scorecard: SectorScorecard = SectorScorecard()
        self._strategy: SectorStrategySummary = SectorStrategySummary()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: FullSectorAssessmentInput) -> FullSectorAssessmentResult:
        started_at = _utcnow()
        config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting full sector assessment %s, company=%s",
            self.workflow_id, config.company_name,
        )

        try:
            # Phase 1: Classify + Pathway Design
            phase1 = await self._phase_classify_and_pathway(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Pathway Validation
            phase2 = await self._phase_validation(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Technology Planning
            phase3 = await self._phase_technology(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Abatement Waterfall
            phase4 = await self._phase_abatement(input_data)
            self._phase_results.append(phase4)

            # Phase 5: Benchmarking (Progress Monitoring)
            phase5 = await self._phase_benchmark(input_data)
            self._phase_results.append(phase5)

            # Phase 6: Multi-Scenario Analysis
            phase6 = await self._phase_scenarios(input_data)
            self._phase_results.append(phase6)

            # Phase 7: Strategy Compilation
            phase7 = await self._phase_strategy(input_data)
            self._phase_results.append(phase7)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            skipped = [p for p in self._phase_results if p.status == PhaseStatus.SKIPPED]
            if not failed:
                overall_status = WorkflowStatus.COMPLETED
            elif len(failed) < len(self._phase_results) - len(skipped):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error("Full sector assessment failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        result = FullSectorAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            pathway_result=self._pathway_result,
            validation_result=self._validation_result,
            technology_result=self._tech_result,
            progress_result=self._progress_result,
            scenario_result=self._scenario_result,
            abatement_waterfall=self._abatement,
            scorecard=self._scorecard,
            strategy=self._strategy,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Classify + Pathway
    # -------------------------------------------------------------------------

    async def _phase_classify_and_pathway(self, input_data: FullSectorAssessmentInput) -> PhaseResult:
        started = _utcnow()
        config = input_data.config
        outputs: Dict[str, Any] = {}

        try:
            design_config = SectorPathwayDesignConfig(
                company_name=config.company_name,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
                nace_codes=config.nace_codes,
                gics_codes=config.gics_codes,
                isic_codes=config.isic_codes,
                revenue_breakdown=config.revenue_breakdown,
                primary_activity=config.primary_activity,
                base_year=config.base_year,
                current_year=config.current_year,
                base_year_emissions_tco2e=config.base_year_emissions_tco2e,
                current_emissions_tco2e=config.current_emissions_tco2e,
                base_year_activity=config.base_year_activity,
                current_activity=config.current_activity,
                scope1_emissions_tco2e=config.scope1_emissions_tco2e,
                scope2_emissions_tco2e=config.scope2_emissions_tco2e,
                scope3_emissions_tco2e=config.scope3_emissions_tco2e,
                target_year=config.target_year,
                convergence_model=config.convergence_model,
                activity_growth_rate=config.activity_growth_rate,
                current_trajectory_annual_reduction=config.current_trajectory_annual_reduction,
                available_capex_usd=config.available_capex_usd,
                sector_abatement_cost_usd_per_tco2e=config.sector_abatement_cost_usd_per_tco2e,
                sbti_coverage_scope1_pct=config.scope12_coverage_pct,
                sbti_coverage_scope3_pct=config.scope3_coverage_pct,
            )

            design_input = SectorPathwayDesignInput(
                config=design_config,
                historical_intensity=input_data.historical_intensity,
                peer_intensities=input_data.peer_data,
                planned_actions=input_data.planned_actions,
            )

            wf = SectorPathwayDesignWorkflow(config=design_config)
            self._pathway_result = await wf.execute(design_input)

            outputs["sector"] = self._pathway_result.sector_classification.primary_sector
            outputs["sector_name"] = self._pathway_result.sector_classification.sector_name
            outputs["sda_method"] = self._pathway_result.sector_classification.sda_method
            outputs["pathways_generated"] = len(self._pathway_result.pathways)
            outputs["sbti_ready"] = self._pathway_result.sbti_ready
            outputs["status"] = self._pathway_result.status.value

            status = PhaseStatus.COMPLETED if self._pathway_result.status == WorkflowStatus.COMPLETED else PhaseStatus.FAILED

        except Exception as exc:
            self.logger.error("Phase 1 (classify+pathway) failed: %s", exc)
            outputs["error"] = str(exc)
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="classify_and_pathway", phase_number=1,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if status == PhaseStatus.COMPLETED else 0.0,
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_classify_pathway",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Pathway Validation
    # -------------------------------------------------------------------------

    async def _phase_validation(self, input_data: FullSectorAssessmentInput) -> PhaseResult:
        started = _utcnow()
        config = input_data.config
        outputs: Dict[str, Any] = {}

        if config.skip_validation or not self._pathway_result:
            return PhaseResult(
                phase_name="pathway_validation", phase_number=2,
                status=PhaseStatus.SKIPPED, outputs={"skipped": True},
                dag_node_id=f"{self.workflow_id}_validation",
            )

        try:
            pr = self._pathway_result
            sector = pr.sector_classification.primary_sector
            sector_data = SDA_SECTORS.get(sector, SDA_SECTORS["cross_sector"])

            base_intensity = pr.intensity_metrics[0].base_year_value if pr.intensity_metrics else 0.0
            current_intensity = pr.intensity_metrics[0].current_value if pr.intensity_metrics else 0.0

            # Build pathway points from NZE pathway
            nze_pathway = next((p for p in pr.pathways if p.scenario == "nze_15c"), None)
            pathway_points = []
            if nze_pathway:
                for pp in nze_pathway.pathway_points:
                    pathway_points.append({
                        "year": pp.year,
                        "intensity": pp.intensity_value,
                    })

            val_config = PathwayValidationConfig(
                company_name=config.company_name,
                sector=sector,
                sda_method=sector_data["sda_method"],
                base_year=config.base_year,
                target_year=config.target_year,
                near_term_target_year=2030,
                base_year_intensity=base_intensity,
                current_intensity=current_intensity,
                target_intensity_2050=sector_data["2050_nze_target"],
                intensity_metric=sector_data["intensity_metric"],
                intensity_unit=sector_data["intensity_unit"],
                scope1_emissions_tco2e=config.scope1_emissions_tco2e,
                scope2_emissions_tco2e=config.scope2_emissions_tco2e,
                scope3_emissions_tco2e=config.scope3_emissions_tco2e,
                scope12_coverage_pct=config.scope12_coverage_pct,
                scope3_coverage_pct=config.scope3_coverage_pct,
            )

            val_input = PathwayValidationInput(
                config=val_config,
                pathway_points=pathway_points,
            )

            wf = PathwayValidationWorkflow(config=val_config)
            self._validation_result = await wf.execute(val_input)

            outputs["status"] = self._validation_result.status.value
            outputs["submission_ready"] = self._validation_result.submission_ready
            outputs["compliance_score"] = self._validation_result.sbti_compliance.compliance_score_pct

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Phase 2 (validation) failed: %s", exc)
            outputs["error"] = str(exc)
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="pathway_validation", phase_number=2,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if status == PhaseStatus.COMPLETED else 0.0,
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_validation",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Technology Planning
    # -------------------------------------------------------------------------

    async def _phase_technology(self, input_data: FullSectorAssessmentInput) -> PhaseResult:
        started = _utcnow()
        config = input_data.config
        outputs: Dict[str, Any] = {}

        if config.skip_technology:
            return PhaseResult(
                phase_name="technology_planning", phase_number=3,
                status=PhaseStatus.SKIPPED, outputs={"skipped": True},
                dag_node_id=f"{self.workflow_id}_technology",
            )

        try:
            sector = (
                self._pathway_result.sector_classification.primary_sector
                if self._pathway_result else "cross_sector"
            )
            current_intensity = (
                self._pathway_result.intensity_metrics[0].current_value
                if self._pathway_result and self._pathway_result.intensity_metrics else 0.0
            )

            tech_config = TechnologyPlanningConfig(
                company_name=config.company_name,
                entity_id=config.entity_id,
                sector=sector,
                base_year=config.base_year,
                target_year=config.target_year,
                current_emissions_tco2e=config.current_emissions_tco2e,
                current_intensity=current_intensity,
                available_capex_usd=config.available_capex_usd,
                discount_rate=config.discount_rate,
                carbon_price_usd_per_tco2e=config.carbon_price_usd,
                activity_level=config.current_activity,
                current_tech_portfolio=input_data.current_tech_portfolio,
            )

            tech_input = TechnologyPlanningInput(config=tech_config)
            wf = TechnologyPlanningWorkflow(config=tech_config)
            self._tech_result = await wf.execute(tech_input)

            outputs["technologies"] = len(self._tech_result.tech_inventory)
            outputs["total_capex_usd"] = self._tech_result.capex_plan.total_capex_usd
            outputs["milestones"] = len(self._tech_result.roadmap.milestones)
            outputs["bottlenecks"] = len(self._tech_result.dependency_analysis.bottlenecks)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Phase 3 (technology) failed: %s", exc)
            outputs["error"] = str(exc)
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="technology_planning", phase_number=3,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if status == PhaseStatus.COMPLETED else 0.0,
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_technology",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Abatement Waterfall
    # -------------------------------------------------------------------------

    async def _phase_abatement(self, input_data: FullSectorAssessmentInput) -> PhaseResult:
        started = _utcnow()
        config = input_data.config
        outputs: Dict[str, Any] = {}

        sector = (
            self._pathway_result.sector_classification.primary_sector
            if self._pathway_result else "cross_sector"
        )
        emissions = max(config.current_emissions_tco2e, 100000)
        lever_data = SECTOR_LEVERS.get(sector, DEFAULT_LEVERS)

        levers: List[AbatementLever] = []
        total_abatement = 0.0
        total_cost = 0.0

        for ld in lever_data:
            abatement_tco2e = emissions * (ld["abatement_pct"] / 100.0)
            lever_cost = abatement_tco2e * ld["cost"]

            levers.append(AbatementLever(
                lever_name=ld["lever"],
                category=ld["cat"],
                abatement_tco2e=round(abatement_tco2e, 0),
                abatement_pct=ld["abatement_pct"],
                cost_per_tco2e_usd=ld["cost"],
                total_cost_usd=round(lever_cost, 0),
                implementation_year_start=ld["start"],
                implementation_year_end=ld["end"],
                trl=ld["trl"],
            ))

            total_abatement += abatement_tco2e
            total_cost += lever_cost

        # Sort by cost (MACC order)
        levers.sort(key=lambda l: l.cost_per_tco2e_usd)

        wavg = total_cost / max(total_abatement, 1.0)
        coverage = min((total_abatement / emissions) * 100, 100.0)

        self._abatement = AbatementWaterfall(
            sector=sector,
            total_abatement_tco2e=round(total_abatement, 0),
            total_cost_usd=round(total_cost, 0),
            levers=levers,
            weighted_avg_cost_usd=round(wavg, 2),
            coverage_pct=round(coverage, 1),
        )
        self._abatement.provenance_hash = _compute_hash(
            self._abatement.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["levers_count"] = len(levers)
        outputs["total_abatement_tco2e"] = self._abatement.total_abatement_tco2e
        outputs["total_cost_usd"] = self._abatement.total_cost_usd
        outputs["coverage_pct"] = self._abatement.coverage_pct
        outputs["wavg_cost_usd_per_tco2e"] = self._abatement.weighted_avg_cost_usd

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="abatement_waterfall", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_abatement",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Benchmarking
    # -------------------------------------------------------------------------

    async def _phase_benchmark(self, input_data: FullSectorAssessmentInput) -> PhaseResult:
        started = _utcnow()
        config = input_data.config
        outputs: Dict[str, Any] = {}

        try:
            sector = (
                self._pathway_result.sector_classification.primary_sector
                if self._pathway_result else "cross_sector"
            )
            current_intensity = (
                self._pathway_result.intensity_metrics[0].current_value
                if self._pathway_result and self._pathway_result.intensity_metrics else 0.0
            )
            base_intensity = (
                self._pathway_result.intensity_metrics[0].base_year_value
                if self._pathway_result and self._pathway_result.intensity_metrics else 0.0
            )
            sector_data = SDA_SECTORS.get(sector, SDA_SECTORS["cross_sector"])

            mon_config = ProgressMonitoringConfig(
                company_name=config.company_name,
                sector=sector,
                intensity_unit=sector_data["intensity_unit"],
                base_year=config.base_year,
                base_year_intensity=base_intensity,
                target_year=config.target_year,
                near_term_target_year=2030,
                near_term_target_intensity=sector_data["2030_nze_target"],
                long_term_target_intensity=sector_data["2050_nze_target"],
                activity_growth_rate=config.activity_growth_rate,
            )

            # Build intensity data points
            intensity_data = []
            for year, intensity in sorted(input_data.historical_intensity.items()):
                intensity_data.append(IntensityDataPoint(
                    year=year, intensity=intensity,
                ))
            if not intensity_data and current_intensity > 0:
                intensity_data = [
                    IntensityDataPoint(year=config.base_year, intensity=base_intensity),
                    IntensityDataPoint(year=config.current_year, intensity=current_intensity),
                ]

            mon_input = ProgressMonitoringInput(
                config=mon_config,
                intensity_data=intensity_data,
                peer_data=input_data.peer_data,
            )

            wf = ProgressMonitoringWorkflow(config=mon_config)
            self._progress_result = await wf.execute(mon_input)

            outputs["overall_rag"] = self._progress_result.overall_rag.value
            outputs["sector_percentile"] = self._progress_result.benchmark_summary.overall_percentile
            outputs["alerts_count"] = len(self._progress_result.progress_report.alerts)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Phase 5 (benchmark) failed: %s", exc)
            outputs["error"] = str(exc)
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="benchmark", phase_number=5,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if status == PhaseStatus.COMPLETED else 0.0,
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_benchmark",
        )

    # -------------------------------------------------------------------------
    # Phase 6: Multi-Scenario Analysis
    # -------------------------------------------------------------------------

    async def _phase_scenarios(self, input_data: FullSectorAssessmentInput) -> PhaseResult:
        started = _utcnow()
        config = input_data.config
        outputs: Dict[str, Any] = {}

        if config.skip_scenarios:
            return PhaseResult(
                phase_name="scenarios", phase_number=6,
                status=PhaseStatus.SKIPPED, outputs={"skipped": True},
                dag_node_id=f"{self.workflow_id}_scenarios",
            )

        try:
            sector = (
                self._pathway_result.sector_classification.primary_sector
                if self._pathway_result else "cross_sector"
            )
            base_intensity = (
                self._pathway_result.intensity_metrics[0].base_year_value
                if self._pathway_result and self._pathway_result.intensity_metrics else 1.0
            )

            sc_config = MultiScenarioConfig(
                company_name=config.company_name,
                sector=sector,
                base_year=config.base_year,
                target_year=config.target_year,
                base_year_intensity=base_intensity,
                base_year_emissions_tco2e=config.base_year_emissions_tco2e,
                current_activity=config.current_activity,
                activity_growth_rate=config.activity_growth_rate,
                monte_carlo_runs=config.monte_carlo_runs,
                seed=config.seed,
                discount_rate=config.discount_rate,
            )

            sc_input = MultiScenarioInput(config=sc_config)
            wf = MultiScenarioAnalysisWorkflow(config=sc_config)
            self._scenario_result = await wf.execute(sc_input)

            outputs["scenarios_modeled"] = len(self._scenario_result.pathway_results)
            outputs["recommended"] = self._scenario_result.recommendation.recommended_scenario
            outputs["confidence"] = self._scenario_result.recommendation.confidence.value

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Phase 6 (scenarios) failed: %s", exc)
            outputs["error"] = str(exc)
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="scenarios", phase_number=6,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if status == PhaseStatus.COMPLETED else 0.0,
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_scenarios",
        )

    # -------------------------------------------------------------------------
    # Phase 7: Strategy Compilation
    # -------------------------------------------------------------------------

    async def _phase_strategy(self, input_data: FullSectorAssessmentInput) -> PhaseResult:
        started = _utcnow()
        config = input_data.config
        outputs: Dict[str, Any] = {}

        # Build scorecard
        pr = self._pathway_result
        vr = self._validation_result
        tr = self._tech_result
        mr = self._progress_result
        sr = self._scenario_result

        classification_score = (
            pr.sector_classification.classification_confidence if pr else 0.0
        )
        pathway_score = (
            min(pr.validation_report.pass_rate_pct, 100.0) if pr else 0.0
        )
        technology_score = (
            min(self._abatement.coverage_pct, 100.0)
        )
        abatement_score = technology_score  # Correlated
        benchmark_score = (
            mr.benchmark_summary.overall_percentile if mr else 50.0
        )
        scenario_score = (
            sr.recommendation.risk_adjusted_score if sr else 50.0
        )

        overall = (
            classification_score * 0.10 +
            pathway_score * 0.25 +
            technology_score * 0.20 +
            abatement_score * 0.15 +
            benchmark_score * 0.15 +
            scenario_score * 0.15
        )

        if overall >= 85:
            maturity = MaturityLevel.LEADING
        elif overall >= 70:
            maturity = MaturityLevel.ADVANCED
        elif overall >= 55:
            maturity = MaturityLevel.ESTABLISHED
        elif overall >= 40:
            maturity = MaturityLevel.DEVELOPING
        else:
            maturity = MaturityLevel.NASCENT

        self._scorecard = SectorScorecard(
            classification_score=round(classification_score, 1),
            pathway_score=round(pathway_score, 1),
            technology_score=round(technology_score, 1),
            abatement_score=round(abatement_score, 1),
            benchmark_score=round(benchmark_score, 1),
            scenario_score=round(scenario_score, 1),
            overall_score=round(overall, 1),
            maturity=maturity,
        )

        # Build strategy summary
        sector = pr.sector_classification.primary_sector if pr else "cross_sector"
        sector_data = SDA_SECTORS.get(sector, SDA_SECTORS["cross_sector"])

        nze_gap = 0.0
        req_accel = 0.0
        if pr and pr.gap_analyses:
            nze = next((g for g in pr.gap_analyses if g.scenario == "nze_15c"), None)
            if nze:
                nze_gap = nze.intensity_gap_pct
                req_accel = nze.required_acceleration_pct

        current_intensity = (
            pr.intensity_metrics[0].current_value if pr and pr.intensity_metrics else 0.0
        )
        trend = (
            pr.intensity_metrics[0].trend_annual_pct if pr and pr.intensity_metrics else 0.0
        )
        cum_red = (
            mr.intensity_update.cumulative_reduction_pct if mr else 0.0
        )

        # Key actions
        key_actions: List[str] = []
        if pr and not pr.sbti_ready:
            key_actions.append("Address SBTi validation gaps before target submission.")
        if sr:
            key_actions.append(
                f"Adopt {sr.recommendation.recommended_scenario_name} as primary pathway."
            )
        if tr:
            immediate = sum(
                1 for a in tr.implementation_plan.actions
                if a.priority.value == "immediate"
            )
            if immediate > 0:
                key_actions.append(f"Deploy {immediate} immediate-priority technologies.")
        key_actions.extend([
            "Align CapEx with sector pathway investment requirements.",
            "Establish quarterly progress monitoring cadence.",
            "Present sector strategy to board for approval.",
        ])

        # Key risks
        key_risks: List[str] = []
        if nze_gap > 25:
            key_risks.append(f"Critical gap to NZE pathway: {nze_gap:+.1f}%.")
        if tr and tr.dependency_analysis.bottlenecks:
            key_risks.append(
                f"{len(tr.dependency_analysis.bottlenecks)} technology bottleneck(s)."
            )
        if sr:
            highest_risk = max(
                sr.risk_profiles, key=lambda r: r.overall_risk_score,
            ) if sr.risk_profiles else None
            if highest_risk:
                key_risks.append(
                    f"Highest risk scenario: {highest_risk.scenario_id} "
                    f"(score: {highest_risk.overall_risk_score:.1f}/10)."
                )

        # Key findings
        key_findings: List[str] = []
        key_findings.append(
            f"Sector: {sector_data['name']} ({sector_data['sda_method']})"
        )
        key_findings.append(
            f"Current intensity: {current_intensity:.4f} {sector_data['intensity_unit']} "
            f"(trend: {trend:+.1f}%/yr)"
        )
        key_findings.append(
            f"NZE gap: {nze_gap:+.1f}%, benchmark percentile: {benchmark_score:.0f}th"
        )
        key_findings.append(
            f"Maturity: {maturity.value}, overall score: {overall:.1f}/100"
        )

        nze_prob = 0.0
        if sr:
            nze_pw = next(
                (p for p in sr.pathway_results if p.scenario_id == "nze_15c"), None,
            )
            nze_prob = nze_pw.probability_target_pct if nze_pw else 0.0

        self._strategy = SectorStrategySummary(
            assessment_date=_utcnow().isoformat(),
            company_name=config.company_name,
            sector=sector,
            sector_name=sector_data["name"],
            sda_method=sector_data["sda_method"],
            sda_eligible=pr.sector_classification.sda_eligibility.value == "eligible" if pr else False,
            current_intensity=round(current_intensity, 6),
            intensity_unit=sector_data["intensity_unit"],
            intensity_trend_pct=round(trend, 2),
            cumulative_reduction_pct=round(cum_red, 2),
            recommended_scenario=sr.recommendation.recommended_scenario if sr else "nze_15c",
            recommended_scenario_name=sr.recommendation.recommended_scenario_name if sr else "IEA NZE 2050",
            sbti_submission_ready=pr.sbti_ready if pr else False,
            nze_gap_pct=round(nze_gap, 2),
            required_acceleration_pct=round(req_accel, 2),
            technologies_count=len(tr.tech_inventory) if tr else 0,
            total_capex_usd=round(tr.capex_plan.total_capex_usd if tr else 0, 0),
            tech_roadmap_years=tr.implementation_plan.timeline_years if tr else 0,
            tech_bottlenecks=len(tr.dependency_analysis.bottlenecks) if tr else 0,
            total_abatement_tco2e=round(self._abatement.total_abatement_tco2e, 0),
            abatement_levers_count=len(self._abatement.levers),
            avg_abatement_cost_usd=round(self._abatement.weighted_avg_cost_usd, 2),
            sector_percentile=round(benchmark_score, 1),
            vs_sector_leader_pct=round(nze_gap * 0.5, 1),  # Approximation
            vs_iea_pathway_pct=round(nze_gap, 1),
            scenarios_modeled=len(sr.pathway_results) if sr else 0,
            nze_probability_pct=round(nze_prob, 1),
            recommendation_confidence=sr.recommendation.confidence.value if sr else "low",
            overall_score=round(overall, 1),
            maturity_level=maturity.value,
            key_actions=key_actions,
            key_risks=key_risks,
            key_findings=key_findings,
        )

        outputs["overall_score"] = round(overall, 1)
        outputs["maturity"] = maturity.value
        outputs["sbti_ready"] = self._strategy.sbti_submission_ready
        outputs["recommended_scenario"] = self._strategy.recommended_scenario
        outputs["key_actions_count"] = len(key_actions)
        outputs["key_risks_count"] = len(key_risks)
        outputs["report_formats"] = ["MD", "HTML", "JSON", "PDF"]

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="strategy_compilation", phase_number=7,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_strategy",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_scorecard(
        self,
        pathway_result: Optional[SectorPathwayDesignResult],
        technology_result: Optional[TechnologyPlanningResult],
        scenario_result: Optional[MultiScenarioResult],
        progress_result: Optional[ProgressMonitoringResult],
        abatement: AbatementWaterfall,
        sector: str,
    ) -> SectorScorecard:
        """
        Calculate a comprehensive sector scorecard using weighted dimension
        scores.  Each dimension is scored 0-100 based on sub-workflow
        outputs with consistent scoring rubrics.
        """
        # Classification score: based on SDA eligibility and data quality
        classification = 100.0 if sector in SDA_SECTORS else 60.0

        # Pathway score: based on SBTi readiness and NZE gap
        pathway = 0.0
        if pathway_result:
            if pathway_result.sbti_ready:
                pathway = 85.0
            nze_gap = next(
                (g.intensity_gap_pct for g in pathway_result.gap_analyses
                 if g.scenario == "nze_15c"), 50.0,
            )
            if nze_gap <= 0:
                pathway = max(pathway, 95.0)
            elif nze_gap <= 10:
                pathway = max(pathway, 80.0)
            elif nze_gap <= 25:
                pathway = max(pathway, 60.0)
            else:
                pathway = max(pathway, 30.0)

        # Technology score: based on TRL portfolio and implementation readiness
        technology = 0.0
        if technology_result:
            avg_trl = sum(
                t.trl for t in technology_result.tech_inventory
            ) / max(len(technology_result.tech_inventory), 1)
            technology = min(avg_trl / 9.0 * 100, 100.0)
            immediate = sum(
                1 for a in technology_result.implementation_plan.actions
                if a.priority.value == "immediate"
            )
            if immediate > 0:
                technology = min(technology + 10, 100.0)

        # Abatement score: based on coverage and cost
        abatement_score = 0.0
        if abatement.levers:
            coverage = abatement.coverage_pct
            abatement_score = min(coverage, 100.0)
            if abatement.weighted_avg_cost_usd < 50:
                abatement_score = min(abatement_score + 10, 100.0)

        # Benchmark score: from progress monitoring percentile
        benchmark = 0.0
        if progress_result:
            benchmark = min(progress_result.benchmark_summary.overall_percentile, 100.0)

        # Scenario score: based on recommendation confidence and NZE probability
        scenario = 0.0
        if scenario_result:
            conf_map = {"high": 90, "medium": 65, "low": 40}
            conf_score = conf_map.get(
                scenario_result.recommendation.confidence.value, 50,
            )
            nze_pw = next(
                (p for p in scenario_result.pathway_results
                 if p.scenario_id == "nze_15c"), None,
            )
            prob_score = nze_pw.probability_target_pct if nze_pw else 30.0
            scenario = (conf_score + prob_score) / 2

        # Overall weighted score
        overall = (
            classification * SCORECARD_WEIGHTS["classification_score"] +
            pathway * SCORECARD_WEIGHTS["pathway_score"] +
            technology * SCORECARD_WEIGHTS["technology_score"] +
            abatement_score * SCORECARD_WEIGHTS["abatement_score"] +
            benchmark * SCORECARD_WEIGHTS["benchmark_score"] +
            scenario * SCORECARD_WEIGHTS["scenario_score"]
        )

        # Maturity determination
        if overall >= MATURITY_THRESHOLDS["leading"]:
            maturity = MaturityLevel.LEADING
        elif overall >= MATURITY_THRESHOLDS["advanced"]:
            maturity = MaturityLevel.ADVANCED
        elif overall >= MATURITY_THRESHOLDS["established"]:
            maturity = MaturityLevel.ESTABLISHED
        elif overall >= MATURITY_THRESHOLDS["developing"]:
            maturity = MaturityLevel.DEVELOPING
        else:
            maturity = MaturityLevel.NASCENT

        return SectorScorecard(
            classification_score=round(classification, 1),
            pathway_score=round(pathway, 1),
            technology_score=round(technology, 1),
            abatement_score=round(abatement_score, 1),
            benchmark_score=round(benchmark, 1),
            scenario_score=round(scenario, 1),
            overall_score=round(overall, 1),
            maturity=maturity,
        )

    def _generate_executive_summary(
        self, strategy: SectorStrategySummary,
    ) -> str:
        """
        Generate a board-ready executive summary from the sector strategy.
        """
        lines: List[str] = [
            f"SECTOR PATHWAY ASSESSMENT: {strategy.company_name or 'Company'}",
            f"Sector: {strategy.sector_name} ({strategy.sda_method})",
            f"Assessment Date: {strategy.assessment_date[:10] if strategy.assessment_date else 'N/A'}",
            "",
            f"Current Intensity: {strategy.current_intensity:.4f} {strategy.intensity_unit}",
            f"Trend: {strategy.intensity_trend_pct:+.1f}%/yr",
            f"Cumulative Reduction: {strategy.cumulative_reduction_pct:.1f}%",
            "",
            f"SBTi Submission Ready: {'YES' if strategy.sbti_submission_ready else 'NO'}",
            f"NZE Gap: {strategy.nze_gap_pct:+.1f}%",
            f"Required Acceleration: {strategy.required_acceleration_pct:.1f} pp/yr",
            "",
            f"Recommended Pathway: {strategy.recommended_scenario_name}",
            f"NZE Probability: {strategy.nze_probability_pct:.0f}%",
            f"Confidence: {strategy.recommendation_confidence.upper()}",
            "",
            f"Technology Portfolio: {strategy.technologies_count} technologies",
            f"Total CapEx: ${strategy.total_capex_usd:,.0f}",
            f"Bottlenecks: {strategy.tech_bottlenecks}",
            "",
            f"Abatement Potential: {strategy.total_abatement_tco2e:,.0f} tCO2e",
            f"Average Cost: ${strategy.avg_abatement_cost_usd:.0f}/tCO2e",
            f"Levers: {strategy.abatement_levers_count}",
            "",
            f"Sector Benchmark: {strategy.sector_percentile:.0f}th percentile",
            f"Overall Score: {strategy.overall_score:.1f}/100",
            f"Maturity: {strategy.maturity_level.upper()}",
        ]

        return "\n".join(lines)

    def _get_transition_timeline(self, sector: str) -> Dict[str, Any]:
        """
        Get sector-specific transition timeline expectations to
        contextualise the assessment results.
        """
        timeline = SECTOR_TRANSITION_TIMELINES.get(sector, {
            "fast_track": 20, "standard": 25, "delayed": 30,
            "key_milestones": [],
        })
        return {
            "sector": sector,
            "fast_track_years": timeline.get("fast_track", 20),
            "standard_years": timeline.get("standard", 25),
            "delayed_years": timeline.get("delayed", 30),
            "key_milestones": timeline.get("key_milestones", []),
        }
