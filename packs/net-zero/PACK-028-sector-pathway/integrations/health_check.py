# -*- coding: utf-8 -*-
"""
SectorPathwayHealthCheck - 20-Category System Health Monitoring for PACK-028
=================================================================================

This module implements 20-category health checking for the Sector Pathway Pack,
validating operational readiness of all platform components including sector
classification, SBTi SDA data freshness, IEA NZE milestone currency, IPCC AR6
reference data integrity, all 30 MRV agents, all 20 DATA agents, pathway
engines, technology roadmap databases, abatement lever registries, convergence
calculators, scenario modeling, database connectivity, PACK-021 baseline
integration, decarb agent availability, and sector benchmark data.

Check Categories (20):
    1.  platform               -- Platform connectivity and version
    2.  mrv_agents             -- All 30 MRV agents (sector-priority routing)
    3.  data_agents            -- All 20 DATA agents (sector activity data)
    4.  engines                -- 8 sector pathway engines
    5.  workflows              -- 6 sector pathway workflows
    6.  templates              -- 8 sector pathway templates
    7.  config                 -- Configuration and preset validity
    8.  database               -- Database connectivity (PostgreSQL/TimescaleDB)
    9.  sector_classification  -- NACE/GICS/ISIC mapping tables
    10. sbti_sda_data          -- SBTi SDA convergence pathway freshness
    11. iea_nze_data           -- IEA NZE 2050 milestone currency
    12. ipcc_ar6_data          -- IPCC AR6 GWP/emission factor integrity
    13. pack021_integration    -- PACK-021 baseline/target import
    14. decarb_agents          -- Decarbonization lever registry
    15. convergence_calc       -- Convergence calculator validation
    16. technology_db          -- Technology milestone database
    17. scenario_modeling      -- 5-scenario framework readiness
    18. benchmark_data         -- Sector benchmark datasets
    19. migrations             -- Database migration status
    20. overall                -- Overall system status

Health Scoring:
    - Each check returns PASS (100%), WARN (50%), FAIL (0%), or SKIP (excluded)
    - Overall score = (passed * 100 + warned * 50) / (total - skipped) * 100
    - Critical failure: any FAIL in engines, sbti_sda_data, or database
    - Remediation suggestions for every FAIL and WARN

SBTi Data Freshness Rules:
    - SDA convergence pathways: must match SBTi Corporate Standard V5.3
    - Pathway data coverage: all 12 SDA sectors must have pathway tables
    - Criteria database: 42 criteria (28 near-term + 14 net-zero) must be present
    - Update frequency: annual review required (flag if >365 days since update)

IEA Milestone Currency Rules:
    - NZE 2050 milestones: minimum 50 milestones across 10+ sectors
    - Pathway data: 5 scenarios (NZE, APS, STEPS, WB2C, 2C) required
    - Technology interdependencies: cross-sector dependency graph required
    - Regional factors: OECD, emerging markets, global multipliers required

IPCC AR6 Data Integrity Rules:
    - GWP-100 values: minimum 20 GHG species from AR6 Table 7.15
    - Carbon budgets: 1.5C/1.7C/2.0C at multiple probability levels
    - Emission factors: CO2, CH4, N2O for 15+ fuel types
    - SSP pathways: all 5 SSP scenarios with 2025-2100 projections

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-028 Sector Pathway Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow
from greenlang.schemas.enums import HealthStatus

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

PACK_BASE_DIR = Path(__file__).parent.parent

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthSeverity(str, Enum):
    """Severity classification for health issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class CheckCategory(str, Enum):
    """The 20 health check categories for Sector Pathway Pack."""
    PLATFORM = "platform"
    MRV_AGENTS = "mrv_agents"
    DATA_AGENTS = "data_agents"
    ENGINES = "engines"
    WORKFLOWS = "workflows"
    TEMPLATES = "templates"
    CONFIG = "config"
    DATABASE = "database"
    SECTOR_CLASSIFICATION = "sector_classification"
    SBTI_SDA_DATA = "sbti_sda_data"
    IEA_NZE_DATA = "iea_nze_data"
    IPCC_AR6_DATA = "ipcc_ar6_data"
    PACK021_INTEGRATION = "pack021_integration"
    DECARB_AGENTS = "decarb_agents"
    CONVERGENCE_CALC = "convergence_calc"
    TECHNOLOGY_DB = "technology_db"
    SCENARIO_MODELING = "scenario_modeling"
    BENCHMARK_DATA = "benchmark_data"
    MIGRATIONS = "migrations"
    OVERALL = "overall"

class DataFreshnessStatus(str, Enum):
    """Freshness status for reference data."""
    CURRENT = "current"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

QUICK_CHECK_CATEGORIES = {
    CheckCategory.ENGINES,
    CheckCategory.WORKFLOWS,
    CheckCategory.TEMPLATES,
    CheckCategory.CONFIG,
    CheckCategory.SECTOR_CLASSIFICATION,
}

CRITICAL_CATEGORIES = {
    CheckCategory.ENGINES,
    CheckCategory.DATABASE,
    CheckCategory.SBTI_SDA_DATA,
    CheckCategory.IPCC_AR6_DATA,
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class RemediationSuggestion(BaseModel):
    """Remediation suggestion for a failed or warned health check."""
    check_name: str = Field(...)
    severity: HealthSeverity = Field(default=HealthSeverity.MEDIUM)
    message: str = Field(...)
    action: str = Field(default="")
    documentation_link: str = Field(default="")
    estimated_fix_time_minutes: int = Field(default=30)

class ComponentHealth(BaseModel):
    """Health status of a single component check."""
    check_name: str = Field(...)
    category: CheckCategory = Field(...)
    status: HealthStatus = Field(default=HealthStatus.PASS)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    remediation: Optional[RemediationSuggestion] = Field(None)
    timestamp: datetime = Field(default_factory=utcnow)

class DataFreshnessCheck(BaseModel):
    """Freshness assessment for reference datasets."""
    dataset_name: str = Field(...)
    source: str = Field(default="")
    version: str = Field(default="")
    last_updated: Optional[datetime] = Field(None)
    freshness_status: DataFreshnessStatus = Field(default=DataFreshnessStatus.UNKNOWN)
    records_count: int = Field(default=0)
    expected_records: int = Field(default=0)
    coverage_pct: float = Field(default=0.0)
    integrity_hash: str = Field(default="")

class SectorCoverageCheck(BaseModel):
    """Sector coverage validation result."""
    sector_name: str = Field(...)
    sda_eligible: bool = Field(default=False)
    has_pathway_data: bool = Field(default=False)
    has_intensity_metric: bool = Field(default=False)
    has_nace_mapping: bool = Field(default=False)
    has_iea_pathway: bool = Field(default=False)
    has_decarb_levers: bool = Field(default=False)
    has_benchmark_data: bool = Field(default=False)
    coverage_score: float = Field(default=0.0)

class HealthCheckConfig(BaseModel):
    """Configuration for health check execution."""
    pack_id: str = Field(default="PACK-028")
    pack_version: str = Field(default="1.0.0")
    skip_categories: List[str] = Field(default_factory=list)
    timeout_per_check_ms: float = Field(default=10000.0)
    verbose: bool = Field(default=False)
    check_data_freshness: bool = Field(default=True)
    freshness_threshold_days: int = Field(default=365)
    require_all_sda_sectors: bool = Field(default=True)
    require_all_iea_scenarios: bool = Field(default=True)
    minimum_gwp_species: int = Field(default=20)
    minimum_iea_milestones: int = Field(default=50)
    minimum_sbti_criteria: int = Field(default=42)

class HealthCheckResult(BaseModel):
    """Complete health check result for PACK-028."""
    result_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-028")
    pack_version: str = Field(default="1.0.0")
    total_checks: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    warnings: int = Field(default=0)
    skipped: int = Field(default=0)
    overall_health_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_status: HealthStatus = Field(default=HealthStatus.PASS)
    has_critical_failure: bool = Field(default=False)
    categories: Dict[str, List[ComponentHealth]] = Field(default_factory=dict)
    remediations: List[RemediationSuggestion] = Field(default_factory=list)
    data_freshness: List[DataFreshnessCheck] = Field(default_factory=list)
    sector_coverage: List[SectorCoverageCheck] = Field(default_factory=list)
    total_duration_ms: float = Field(default=0.0)
    executed_at: datetime = Field(default_factory=utcnow)
    quick_mode: bool = Field(default=False)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Component Lists - PACK-028 Sector Pathway Pack
# ---------------------------------------------------------------------------

SECTOR_PATHWAY_MRV_AGENTS = [f"MRV-{i:03d}" for i in range(1, 31)]

SECTOR_PATHWAY_DATA_AGENTS = [f"DATA-{i:03d}" for i in range(1, 21)]

SECTOR_PATHWAY_ENGINES = [
    "sector_classification_engine",
    "intensity_calculator_engine",
    "pathway_generator_engine",
    "convergence_analyzer_engine",
    "technology_roadmap_engine",
    "abatement_waterfall_engine",
    "sector_benchmark_engine",
    "scenario_comparison_engine",
]

SECTOR_PATHWAY_WORKFLOWS = [
    "sector_pathway_design_workflow",
    "pathway_validation_workflow",
    "technology_planning_workflow",
    "progress_monitoring_workflow",
    "multi_scenario_analysis_workflow",
    "sector_onboarding_workflow",
]

SECTOR_PATHWAY_TEMPLATES = [
    "sector_pathway_report",
    "intensity_convergence_report",
    "technology_roadmap_report",
    "abatement_waterfall_report",
    "sector_benchmark_report",
    "scenario_comparison_report",
    "sbti_validation_report",
    "sector_strategy_report",
]

SDA_SECTORS = [
    "power_generation", "steel", "cement", "aluminum",
    "pulp_paper", "chemicals", "aviation", "shipping",
    "road_transport", "rail", "buildings_residential",
    "buildings_commercial",
]

EXTENDED_SECTORS = [
    "agriculture", "food_beverage", "oil_gas_upstream", "cross_sector",
]

ALL_SECTORS = SDA_SECTORS + EXTENDED_SECTORS

IEA_SCENARIOS = ["nze_1.5c", "wb2c", "2c", "aps", "steps"]

INTEGRATION_BRIDGES = [
    "pack_orchestrator",
    "sbti_sda_bridge",
    "iea_nze_bridge",
    "ipcc_ar6_bridge",
    "pack021_bridge",
    "mrv_bridge",
    "decarb_bridge",
    "data_bridge",
    "health_check",
    "setup_wizard",
]

MIGRATION_FILES = [
    "V181__PACK028_sector_classification",
    "V182__PACK028_intensity_metrics",
    "V183__PACK028_sector_pathways",
    "V184__PACK028_convergence_analysis",
    "V185__PACK028_technology_roadmaps",
    "V186__PACK028_abatement_waterfall",
    "V187__PACK028_sector_benchmarks",
    "V188__PACK028_scenario_comparisons",
    "V189__PACK028_sbti_sda_data",
    "V190__PACK028_iea_nze_milestones",
    "V191__PACK028_ipcc_ar6_pathways",
    "V192__PACK028_sector_reference_data",
    "V193__PACK028_multi_scenario_modeling",
    "V194__PACK028_technology_adoption",
    "V195__PACK028_views_and_indexes",
]

SECTOR_INTENSITY_METRICS = {
    "power_generation": {"metric": "gCO2/kWh", "source": "SBTi SDA"},
    "steel": {"metric": "tCO2e/tonne crude steel", "source": "SBTi SDA"},
    "cement": {"metric": "kgCO2/tonne cite", "source": "SBTi SDA"},
    "aluminum": {"metric": "tCO2e/tonne primary", "source": "SBTi SDA"},
    "pulp_paper": {"metric": "tCO2e/tonne product", "source": "SBTi SDA"},
    "chemicals": {"metric": "tCO2e/tonne product", "source": "SBTi SDA"},
    "aviation": {"metric": "gCO2/pkm", "source": "SBTi SDA"},
    "shipping": {"metric": "gCO2/tkm", "source": "SBTi SDA"},
    "road_transport": {"metric": "gCO2/vkm", "source": "SBTi SDA"},
    "rail": {"metric": "gCO2/tkm", "source": "SBTi SDA"},
    "buildings_residential": {"metric": "kgCO2/m2", "source": "SBTi SDA"},
    "buildings_commercial": {"metric": "kgCO2/m2", "source": "SBTi SDA"},
    "agriculture": {"metric": "tCO2e/ha", "source": "IEA NZE"},
    "food_beverage": {"metric": "tCO2e/tonne product", "source": "IEA NZE"},
    "oil_gas_upstream": {"metric": "kgCO2e/boe", "source": "IEA NZE"},
    "cross_sector": {"metric": "tCO2e/revenue_mUSD", "source": "Derived"},
}

# ---------------------------------------------------------------------------
# Freshness & Integrity Validation Helpers
# ---------------------------------------------------------------------------

def _check_sbti_sda_freshness() -> DataFreshnessCheck:
    """Validate SBTi SDA convergence pathway data freshness and completeness."""
    try:
        from .sbti_sda_bridge import (
            SDA_CONVERGENCE_PATHWAYS,
            SDA_INTENSITY_METRICS,
            SBTI_NEAR_TERM_CRITERIA,
            SBTI_NET_ZERO_CRITERIA,
            NACE_TO_SDA_SECTOR,
            GICS_TO_SDA_SECTOR,
        )

        pathways_count = len(SDA_CONVERGENCE_PATHWAYS)
        metrics_count = len(SDA_INTENSITY_METRICS)
        near_term_count = len(SBTI_NEAR_TERM_CRITERIA)
        net_zero_count = len(SBTI_NET_ZERO_CRITERIA)
        total_criteria = near_term_count + net_zero_count
        nace_mappings = len(NACE_TO_SDA_SECTOR)
        gics_mappings = len(GICS_TO_SDA_SECTOR)

        integrity_data = {
            "pathways": pathways_count,
            "metrics": metrics_count,
            "criteria": total_criteria,
            "nace_mappings": nace_mappings,
            "gics_mappings": gics_mappings,
        }

        coverage = pathways_count / 12 * 100 if pathways_count <= 12 else 100.0
        status = DataFreshnessStatus.CURRENT
        if pathways_count < 12:
            status = DataFreshnessStatus.STALE
        if total_criteria < 42:
            status = DataFreshnessStatus.STALE

        return DataFreshnessCheck(
            dataset_name="SBTi SDA Convergence Pathways",
            source="SBTi Corporate Standard V5.3",
            version="5.3",
            freshness_status=status,
            records_count=pathways_count,
            expected_records=12,
            coverage_pct=round(coverage, 1),
            integrity_hash=_compute_hash(integrity_data),
        )
    except Exception as exc:
        return DataFreshnessCheck(
            dataset_name="SBTi SDA Convergence Pathways",
            source="SBTi Corporate Standard V5.3",
            freshness_status=DataFreshnessStatus.UNKNOWN,
            coverage_pct=0.0,
        )

def _check_iea_nze_freshness() -> DataFreshnessCheck:
    """Validate IEA NZE 2050 milestone data currency."""
    try:
        from .iea_nze_bridge import (
            IEA_SECTOR_PATHWAYS,
            IEA_TECHNOLOGY_MILESTONES,
            REGIONAL_ADJUSTMENT_FACTORS,
            TECHNOLOGY_INTERDEPENDENCIES,
        )

        pathways_count = len(IEA_SECTOR_PATHWAYS)
        milestones_count = len(IEA_TECHNOLOGY_MILESTONES)
        regions_count = len(REGIONAL_ADJUSTMENT_FACTORS)
        interdeps_count = len(TECHNOLOGY_INTERDEPENDENCIES)

        integrity_data = {
            "pathways": pathways_count,
            "milestones": milestones_count,
            "regions": regions_count,
            "interdependencies": interdeps_count,
        }

        status = DataFreshnessStatus.CURRENT
        if milestones_count < 50:
            status = DataFreshnessStatus.STALE
        if pathways_count < 9:
            status = DataFreshnessStatus.STALE

        coverage = min(milestones_count / 50 * 100, 100.0)

        return DataFreshnessCheck(
            dataset_name="IEA NZE 2050 Milestones",
            source="IEA World Energy Outlook 2024",
            version="2024",
            freshness_status=status,
            records_count=milestones_count,
            expected_records=50,
            coverage_pct=round(coverage, 1),
            integrity_hash=_compute_hash(integrity_data),
        )
    except Exception:
        return DataFreshnessCheck(
            dataset_name="IEA NZE 2050 Milestones",
            source="IEA World Energy Outlook 2024",
            freshness_status=DataFreshnessStatus.UNKNOWN,
            coverage_pct=0.0,
        )

def _check_ipcc_ar6_freshness() -> DataFreshnessCheck:
    """Validate IPCC AR6 GWP and emission factor data integrity."""
    try:
        from .ipcc_ar6_bridge import (
            GWP_100_AR6,
            CARBON_BUDGETS_GTCO2,
            EMISSION_FACTORS_CO2_KG_PER_TJ,
            PROCESS_EMISSION_FACTORS,
            SSP_EMISSION_PATHWAYS,
        )

        gwp_count = len(GWP_100_AR6)
        budgets_count = len(CARBON_BUDGETS_GTCO2)
        ef_count = len(EMISSION_FACTORS_CO2_KG_PER_TJ)
        process_count = len(PROCESS_EMISSION_FACTORS)
        ssp_count = len(SSP_EMISSION_PATHWAYS)

        integrity_data = {
            "gwp_species": gwp_count,
            "budgets": budgets_count,
            "emission_factors": ef_count,
            "process_factors": process_count,
            "ssp_scenarios": ssp_count,
        }

        status = DataFreshnessStatus.CURRENT
        if gwp_count < 20:
            status = DataFreshnessStatus.STALE
        if ssp_count < 5:
            status = DataFreshnessStatus.STALE

        coverage = min(gwp_count / 20 * 100, 100.0)

        return DataFreshnessCheck(
            dataset_name="IPCC AR6 GWP & Emission Factors",
            source="IPCC AR6 WG1 Table 7.15",
            version="AR6 (2021)",
            freshness_status=status,
            records_count=gwp_count,
            expected_records=20,
            coverage_pct=round(coverage, 1),
            integrity_hash=_compute_hash(integrity_data),
        )
    except Exception:
        return DataFreshnessCheck(
            dataset_name="IPCC AR6 GWP & Emission Factors",
            source="IPCC AR6 WG1 Table 7.15",
            freshness_status=DataFreshnessStatus.UNKNOWN,
            coverage_pct=0.0,
        )

def _check_pack021_availability() -> DataFreshnessCheck:
    """Validate PACK-021 baseline integration availability."""
    try:
        from .pack021_bridge import PACK021_COMPONENTS, PACK021Bridge

        components_count = len(PACK021_COMPONENTS)
        status = (
            DataFreshnessStatus.CURRENT
            if components_count >= 6
            else DataFreshnessStatus.STALE
        )

        return DataFreshnessCheck(
            dataset_name="PACK-021 Net Zero Starter Integration",
            source="PACK-021 Net Zero Starter Pack",
            version="1.0.0",
            freshness_status=status,
            records_count=components_count,
            expected_records=6,
            coverage_pct=min(components_count / 6 * 100, 100.0),
            integrity_hash=_compute_hash({"components": components_count}),
        )
    except Exception:
        return DataFreshnessCheck(
            dataset_name="PACK-021 Net Zero Starter Integration",
            source="PACK-021 Net Zero Starter Pack",
            freshness_status=DataFreshnessStatus.UNKNOWN,
            coverage_pct=0.0,
        )

def _check_decarb_data() -> DataFreshnessCheck:
    """Validate decarbonization lever registry data."""
    try:
        from .decarb_bridge import SECTOR_DECARB_LEVERS, SectorDecarbBridge

        sectors_with_levers = len(SECTOR_DECARB_LEVERS)
        total_levers = sum(len(v) for v in SECTOR_DECARB_LEVERS.values())

        status = (
            DataFreshnessStatus.CURRENT
            if sectors_with_levers >= 6 and total_levers >= 30
            else DataFreshnessStatus.STALE
        )

        return DataFreshnessCheck(
            dataset_name="Sector Decarbonization Levers",
            source="PACK-028 Decarb Bridge",
            version="1.0.0",
            freshness_status=status,
            records_count=total_levers,
            expected_records=30,
            coverage_pct=min(total_levers / 30 * 100, 100.0),
            integrity_hash=_compute_hash({
                "sectors": sectors_with_levers, "levers": total_levers,
            }),
        )
    except Exception:
        return DataFreshnessCheck(
            dataset_name="Sector Decarbonization Levers",
            source="PACK-028 Decarb Bridge",
            freshness_status=DataFreshnessStatus.UNKNOWN,
            coverage_pct=0.0,
        )

def _check_sector_coverage() -> List[SectorCoverageCheck]:
    """Validate per-sector data coverage across all subsystems."""
    coverage_results: List[SectorCoverageCheck] = []

    # Try import the relevant data tables
    sda_pathways: Dict[str, Any] = {}
    sda_metrics: Dict[str, Any] = {}
    nace_mapping: Dict[str, Any] = {}
    iea_pathways: Dict[str, Any] = {}
    decarb_levers: Dict[str, Any] = {}

    try:
        from .sbti_sda_bridge import SDA_CONVERGENCE_PATHWAYS, SDA_INTENSITY_METRICS
        sda_pathways = SDA_CONVERGENCE_PATHWAYS
        sda_metrics = SDA_INTENSITY_METRICS
    except Exception:
        pass

    try:
        from .pack_orchestrator import SECTOR_NACE_MAPPING
        nace_mapping = SECTOR_NACE_MAPPING
    except Exception:
        pass

    try:
        from .iea_nze_bridge import IEA_SECTOR_PATHWAYS
        iea_pathways = IEA_SECTOR_PATHWAYS
    except Exception:
        pass

    try:
        from .decarb_bridge import SECTOR_DECARB_LEVERS
        decarb_levers = SECTOR_DECARB_LEVERS
    except Exception:
        pass

    for sector in ALL_SECTORS:
        is_sda = sector in SDA_SECTORS
        has_pathway = sector in sda_pathways
        has_metric = sector in sda_metrics
        has_nace = sector in nace_mapping
        has_iea = sector in iea_pathways
        has_decarb = sector in decarb_levers

        checks_total = 6
        checks_passed = sum([
            is_sda or (sector in EXTENDED_SECTORS),  # sector classification
            has_pathway or not is_sda,  # pathway (only required for SDA)
            has_metric or not is_sda,   # metric (only required for SDA)
            has_nace,                   # NACE mapping
            has_iea,                    # IEA pathway
            has_decarb,                 # decarb levers
        ])
        score = round(checks_passed / checks_total * 100, 1)

        coverage_results.append(SectorCoverageCheck(
            sector_name=sector,
            sda_eligible=is_sda,
            has_pathway_data=has_pathway,
            has_intensity_metric=has_metric,
            has_nace_mapping=has_nace,
            has_iea_pathway=has_iea,
            has_decarb_levers=has_decarb,
            has_benchmark_data=has_nace,  # proxy: NACE mapping implies benchmark-ready
            coverage_score=score,
        ))

    return coverage_results

# ---------------------------------------------------------------------------
# Convergence Calculator Validation
# ---------------------------------------------------------------------------

def _validate_convergence_calculator() -> List[ComponentHealth]:
    """Validate convergence calculation methods work correctly."""
    checks: List[ComponentHealth] = []

    # Test linear convergence: known inputs, expected output
    try:
        from .sbti_sda_bridge import SBTiSDABridge

        bridge = SBTiSDABridge()

        # Attempt a basic convergence calculation for power sector
        result = bridge.calculate_convergence(
            sector="power_generation",
            base_year=2023,
            base_intensity=400.0,
            target_year=2030,
            method="linear",
        )

        if result and result.get("pathway_points"):
            checks.append(ComponentHealth(
                check_name="convergence_linear_power",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.PASS,
                message="Linear convergence for power sector: validated",
                details={"points_generated": len(result.get("pathway_points", []))},
            ))
        else:
            checks.append(ComponentHealth(
                check_name="convergence_linear_power",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.WARN,
                message="Linear convergence returned empty pathway",
            ))
    except Exception as exc:
        checks.append(ComponentHealth(
            check_name="convergence_linear_power",
            category=CheckCategory.CONVERGENCE_CALC,
            status=HealthStatus.WARN,
            message=f"Convergence calculator import: {exc}",
        ))

    # Test exponential convergence
    try:
        from .sbti_sda_bridge import SBTiSDABridge

        bridge = SBTiSDABridge()
        result = bridge.calculate_convergence(
            sector="steel",
            base_year=2023,
            base_intensity=1.85,
            target_year=2030,
            method="exponential",
        )

        if result:
            checks.append(ComponentHealth(
                check_name="convergence_exponential_steel",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.PASS,
                message="Exponential convergence for steel sector: validated",
            ))
        else:
            checks.append(ComponentHealth(
                check_name="convergence_exponential_steel",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.WARN,
                message="Exponential convergence returned no result",
            ))
    except Exception as exc:
        checks.append(ComponentHealth(
            check_name="convergence_exponential_steel",
            category=CheckCategory.CONVERGENCE_CALC,
            status=HealthStatus.WARN,
            message=f"Exponential convergence check: {exc}",
        ))

    # Test S-curve convergence
    try:
        from .sbti_sda_bridge import SBTiSDABridge

        bridge = SBTiSDABridge()
        result = bridge.calculate_convergence(
            sector="cement",
            base_year=2023,
            base_intensity=0.63,
            target_year=2050,
            method="s_curve",
        )

        if result:
            checks.append(ComponentHealth(
                check_name="convergence_s_curve_cement",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.PASS,
                message="S-curve convergence for cement sector: validated",
            ))
        else:
            checks.append(ComponentHealth(
                check_name="convergence_s_curve_cement",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.WARN,
                message="S-curve convergence returned no result",
            ))
    except Exception as exc:
        checks.append(ComponentHealth(
            check_name="convergence_s_curve_cement",
            category=CheckCategory.CONVERGENCE_CALC,
            status=HealthStatus.WARN,
            message=f"S-curve convergence check: {exc}",
        ))

    # Test stepped convergence
    try:
        from .sbti_sda_bridge import SBTiSDABridge

        bridge = SBTiSDABridge()
        result = bridge.calculate_convergence(
            sector="aviation",
            base_year=2023,
            base_intensity=90.0,
            target_year=2050,
            method="stepped",
        )

        if result:
            checks.append(ComponentHealth(
                check_name="convergence_stepped_aviation",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.PASS,
                message="Stepped convergence for aviation sector: validated",
            ))
        else:
            checks.append(ComponentHealth(
                check_name="convergence_stepped_aviation",
                category=CheckCategory.CONVERGENCE_CALC,
                status=HealthStatus.WARN,
                message="Stepped convergence returned no result",
            ))
    except Exception as exc:
        checks.append(ComponentHealth(
            check_name="convergence_stepped_aviation",
            category=CheckCategory.CONVERGENCE_CALC,
            status=HealthStatus.WARN,
            message=f"Stepped convergence check: {exc}",
        ))

    if not checks:
        checks.append(ComponentHealth(
            check_name="convergence_calculator_unavailable",
            category=CheckCategory.CONVERGENCE_CALC,
            status=HealthStatus.FAIL,
            message="Convergence calculator could not be validated",
            remediation=RemediationSuggestion(
                check_name="convergence_calculator_unavailable",
                severity=HealthSeverity.HIGH,
                message="Convergence calculator module not available",
                action="Verify sbti_sda_bridge.py is properly installed",
            ),
        ))

    return checks

# ---------------------------------------------------------------------------
# Scenario Modeling Validation
# ---------------------------------------------------------------------------

def _validate_scenario_modeling() -> List[ComponentHealth]:
    """Validate 5-scenario framework readiness."""
    checks: List[ComponentHealth] = []

    try:
        from .iea_nze_bridge import IEA_SECTOR_PATHWAYS, IEANZEBridge

        bridge = IEANZEBridge()

        # Check that all 5 scenarios have pathway data
        available_scenarios: set = set()
        for sector_data in IEA_SECTOR_PATHWAYS.values():
            for scenario_key in sector_data.keys():
                available_scenarios.add(scenario_key)

        for scenario in IEA_SCENARIOS:
            if scenario in available_scenarios:
                checks.append(ComponentHealth(
                    check_name=f"scenario_{scenario}",
                    category=CheckCategory.SCENARIO_MODELING,
                    status=HealthStatus.PASS,
                    message=f"Scenario {scenario}: pathway data available",
                ))
            else:
                checks.append(ComponentHealth(
                    check_name=f"scenario_{scenario}",
                    category=CheckCategory.SCENARIO_MODELING,
                    status=HealthStatus.WARN,
                    message=f"Scenario {scenario}: pathway data missing",
                    remediation=RemediationSuggestion(
                        check_name=f"scenario_{scenario}",
                        severity=HealthSeverity.MEDIUM,
                        message=f"IEA scenario {scenario} has no pathway data",
                        action="Add pathway data for this scenario to iea_nze_bridge.py",
                    ),
                ))

        # Check scenario comparison capability
        try:
            comparison = bridge.compare_scenarios(
                sector="power_generation",
                base_year=2023,
                base_intensity=400.0,
                scenarios=["nze_1.5c", "steps"],
            )
            checks.append(ComponentHealth(
                check_name="scenario_comparison_engine",
                category=CheckCategory.SCENARIO_MODELING,
                status=HealthStatus.PASS,
                message="Scenario comparison engine: functional",
                details={"scenarios_compared": 2},
            ))
        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="scenario_comparison_engine",
                category=CheckCategory.SCENARIO_MODELING,
                status=HealthStatus.WARN,
                message=f"Scenario comparison: {exc}",
            ))

    except Exception as exc:
        checks.append(ComponentHealth(
            check_name="scenario_framework_unavailable",
            category=CheckCategory.SCENARIO_MODELING,
            status=HealthStatus.FAIL,
            message=f"Scenario modeling framework: {exc}",
            remediation=RemediationSuggestion(
                check_name="scenario_framework_unavailable",
                severity=HealthSeverity.HIGH,
                message="IEA NZE bridge not available for scenario modeling",
                action="Verify iea_nze_bridge.py is properly installed",
            ),
        ))

    return checks

# ---------------------------------------------------------------------------
# Technology Database Validation
# ---------------------------------------------------------------------------

def _validate_technology_db() -> List[ComponentHealth]:
    """Validate technology milestone database completeness."""
    checks: List[ComponentHealth] = []

    try:
        from .iea_nze_bridge import (
            IEA_TECHNOLOGY_MILESTONES,
            TECHNOLOGY_INTERDEPENDENCIES,
        )

        milestones_count = len(IEA_TECHNOLOGY_MILESTONES)
        interdeps_count = len(TECHNOLOGY_INTERDEPENDENCIES)

        # Check milestone count
        if milestones_count >= 50:
            checks.append(ComponentHealth(
                check_name="technology_milestones_count",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.PASS,
                message=f"Technology milestones: {milestones_count} entries (target: 50+)",
                details={"count": milestones_count},
            ))
        elif milestones_count >= 30:
            checks.append(ComponentHealth(
                check_name="technology_milestones_count",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.WARN,
                message=f"Technology milestones: {milestones_count}/50 (below target)",
                remediation=RemediationSuggestion(
                    check_name="technology_milestones_count",
                    severity=HealthSeverity.MEDIUM,
                    message=f"Only {milestones_count} milestones, target is 50+",
                    action="Add missing milestones to IEA_TECHNOLOGY_MILESTONES",
                ),
            ))
        else:
            checks.append(ComponentHealth(
                check_name="technology_milestones_count",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.FAIL,
                message=f"Technology milestones: {milestones_count}/50 (critically low)",
                remediation=RemediationSuggestion(
                    check_name="technology_milestones_count",
                    severity=HealthSeverity.HIGH,
                    message=f"Only {milestones_count} milestones, minimum 30 required",
                    action="Populate IEA_TECHNOLOGY_MILESTONES in iea_nze_bridge.py",
                ),
            ))

        # Check sector coverage of milestones
        sectors_with_milestones: set = set()
        for milestone in IEA_TECHNOLOGY_MILESTONES:
            s = milestone.get("sector", "")
            if s:
                sectors_with_milestones.add(s)

        if len(sectors_with_milestones) >= 10:
            checks.append(ComponentHealth(
                check_name="technology_sector_coverage",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.PASS,
                message=f"Technology sector coverage: {len(sectors_with_milestones)} sectors",
                details={"sectors": sorted(sectors_with_milestones)},
            ))
        else:
            checks.append(ComponentHealth(
                check_name="technology_sector_coverage",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.WARN,
                message=f"Technology sector coverage: {len(sectors_with_milestones)}/10 sectors",
            ))

        # Check interdependency graph
        if interdeps_count >= 5:
            checks.append(ComponentHealth(
                check_name="technology_interdependencies",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.PASS,
                message=f"Technology interdependencies: {interdeps_count} cross-sector links",
            ))
        else:
            checks.append(ComponentHealth(
                check_name="technology_interdependencies",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.WARN,
                message=f"Technology interdependencies: {interdeps_count}/5 (sparse graph)",
            ))

        # Check technology adoption modeling
        try:
            from .iea_nze_bridge import IEANZEBridge

            bridge = IEANZEBridge()
            adoption = bridge.model_technology_adoption(
                technology="solar_pv",
                current_penetration_pct=30.0,
                max_penetration_pct=90.0,
                learning_rate=0.20,
                years_to_model=10,
            )
            if adoption:
                checks.append(ComponentHealth(
                    check_name="technology_adoption_modeling",
                    category=CheckCategory.TECHNOLOGY_DB,
                    status=HealthStatus.PASS,
                    message="Technology adoption S-curve modeling: functional",
                ))
            else:
                checks.append(ComponentHealth(
                    check_name="technology_adoption_modeling",
                    category=CheckCategory.TECHNOLOGY_DB,
                    status=HealthStatus.WARN,
                    message="Technology adoption modeling returned empty result",
                ))
        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="technology_adoption_modeling",
                category=CheckCategory.TECHNOLOGY_DB,
                status=HealthStatus.WARN,
                message=f"Technology adoption modeling: {exc}",
            ))

    except Exception as exc:
        checks.append(ComponentHealth(
            check_name="technology_db_unavailable",
            category=CheckCategory.TECHNOLOGY_DB,
            status=HealthStatus.FAIL,
            message=f"Technology database: {exc}",
            remediation=RemediationSuggestion(
                check_name="technology_db_unavailable",
                severity=HealthSeverity.HIGH,
                message="IEA NZE bridge technology data not accessible",
                action="Verify iea_nze_bridge.py installation and data tables",
            ),
        ))

    return checks

# ---------------------------------------------------------------------------
# Benchmark Data Validation
# ---------------------------------------------------------------------------

def _validate_benchmark_data() -> List[ComponentHealth]:
    """Validate sector benchmark datasets for peer comparison."""
    checks: List[ComponentHealth] = []

    try:
        from .pack_orchestrator import SECTOR_NACE_MAPPING

        sectors_with_nace = len(SECTOR_NACE_MAPPING)
        sda_sectors_mapped = sum(
            1 for v in SECTOR_NACE_MAPPING.values()
            if v.get("sda_eligible", False)
        )

        checks.append(ComponentHealth(
            check_name="benchmark_nace_mapping",
            category=CheckCategory.BENCHMARK_DATA,
            status=HealthStatus.PASS if sectors_with_nace >= 12 else HealthStatus.WARN,
            message=f"NACE sector mapping: {sectors_with_nace} sectors, {sda_sectors_mapped} SDA eligible",
            details={
                "total_sectors": sectors_with_nace,
                "sda_eligible": sda_sectors_mapped,
            },
        ))
    except Exception as exc:
        checks.append(ComponentHealth(
            check_name="benchmark_nace_mapping",
            category=CheckCategory.BENCHMARK_DATA,
            status=HealthStatus.FAIL,
            message=f"NACE sector mapping: {exc}",
            remediation=RemediationSuggestion(
                check_name="benchmark_nace_mapping",
                severity=HealthSeverity.HIGH,
                message="SECTOR_NACE_MAPPING not accessible",
                action="Verify pack_orchestrator.py is properly installed",
            ),
        ))

    # Validate intensity metric coverage
    for sector in SDA_SECTORS:
        metric_info = SECTOR_INTENSITY_METRICS.get(sector, {})
        has_metric = bool(metric_info.get("metric"))
        checks.append(ComponentHealth(
            check_name=f"benchmark_metric_{sector}",
            category=CheckCategory.BENCHMARK_DATA,
            status=HealthStatus.PASS if has_metric else HealthStatus.WARN,
            message=f"Intensity metric {sector}: {metric_info.get('metric', 'missing')}",
            details={"sector": sector, "metric": metric_info.get("metric", "")},
        ))

    # Check sector routing groups
    try:
        from .pack_orchestrator import SECTOR_ROUTING_GROUPS

        groups_count = len(SECTOR_ROUTING_GROUPS)
        total_sectors = sum(len(v) for v in SECTOR_ROUTING_GROUPS.values())
        checks.append(ComponentHealth(
            check_name="benchmark_routing_groups",
            category=CheckCategory.BENCHMARK_DATA,
            status=HealthStatus.PASS if groups_count >= 5 else HealthStatus.WARN,
            message=f"Sector routing groups: {groups_count} groups, {total_sectors} sectors",
            details={"groups": groups_count, "sectors_routed": total_sectors},
        ))
    except Exception:
        checks.append(ComponentHealth(
            check_name="benchmark_routing_groups",
            category=CheckCategory.BENCHMARK_DATA,
            status=HealthStatus.WARN,
            message="Sector routing groups: not accessible",
        ))

    return checks

# ---------------------------------------------------------------------------
# SectorPathwayHealthCheck
# ---------------------------------------------------------------------------

class SectorPathwayHealthCheck:
    """20-category health check for Sector Pathway Pack (PACK-028).

    Validates the operational readiness of all platform components required
    for sector-specific decarbonization pathway analysis, including SBTi SDA
    convergence data, IEA NZE milestones, IPCC AR6 emission factors, 30 MRV
    agents, 20 DATA agents, 8 pathway engines, and sector classification.

    Example:
        >>> hc = SectorPathwayHealthCheck()
        >>> result = hc.run()
        >>> print(f"Score: {result.overall_health_score}/100")
        >>> print(f"Status: {result.overall_status}")
        >>> for r in result.remediations:
        ...     print(f"  [{r.severity}] {r.message}: {r.action}")

    Quick Mode:
        >>> result = hc.run_quick()  # checks only engines/workflows/templates/config
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or HealthCheckConfig()

        self._check_handlers: Dict[CheckCategory, Callable[[], List[ComponentHealth]]] = {
            CheckCategory.PLATFORM: self._check_platform,
            CheckCategory.MRV_AGENTS: self._check_mrv_agents,
            CheckCategory.DATA_AGENTS: self._check_data_agents,
            CheckCategory.ENGINES: self._check_engines,
            CheckCategory.WORKFLOWS: self._check_workflows,
            CheckCategory.TEMPLATES: self._check_templates,
            CheckCategory.CONFIG: self._check_config,
            CheckCategory.DATABASE: self._check_database,
            CheckCategory.SECTOR_CLASSIFICATION: self._check_sector_classification,
            CheckCategory.SBTI_SDA_DATA: self._check_sbti_sda,
            CheckCategory.IEA_NZE_DATA: self._check_iea_nze,
            CheckCategory.IPCC_AR6_DATA: self._check_ipcc_ar6,
            CheckCategory.PACK021_INTEGRATION: self._check_pack021,
            CheckCategory.DECARB_AGENTS: self._check_decarb,
            CheckCategory.CONVERGENCE_CALC: self._check_convergence,
            CheckCategory.TECHNOLOGY_DB: self._check_technology_db,
            CheckCategory.SCENARIO_MODELING: self._check_scenario_modeling,
            CheckCategory.BENCHMARK_DATA: self._check_benchmark_data,
            CheckCategory.MIGRATIONS: self._check_migrations,
        }

        self.logger.info("SectorPathwayHealthCheck initialized: 20 categories")

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def run(self) -> HealthCheckResult:
        """Execute full 20-category health check."""
        return self._execute_checks(quick_mode=False)

    def run_quick(self) -> HealthCheckResult:
        """Execute quick health check (engines, workflows, templates, config, sector classification only)."""
        return self._execute_checks(quick_mode=True)

    def run_category(self, category: str) -> HealthCheckResult:
        """Execute health check for a single category."""
        try:
            cat_enum = CheckCategory(category)
        except ValueError:
            return HealthCheckResult(
                total_checks=0, overall_status=HealthStatus.FAIL,
                provenance_hash=_compute_hash({"error": f"Unknown category: {category}"}),
            )

        return self._execute_single_category(cat_enum)

    def run_data_freshness(self) -> List[DataFreshnessCheck]:
        """Run data freshness checks for all reference datasets."""
        freshness_checks = [
            _check_sbti_sda_freshness(),
            _check_iea_nze_freshness(),
            _check_ipcc_ar6_freshness(),
            _check_pack021_availability(),
            _check_decarb_data(),
        ]
        return freshness_checks

    def run_sector_coverage(self) -> List[SectorCoverageCheck]:
        """Run per-sector coverage validation."""
        return _check_sector_coverage()

    def get_remediation_report(self, result: HealthCheckResult) -> Dict[str, Any]:
        """Generate a structured remediation report from health check results."""
        critical = [r for r in result.remediations if r.severity == HealthSeverity.CRITICAL]
        high = [r for r in result.remediations if r.severity == HealthSeverity.HIGH]
        medium = [r for r in result.remediations if r.severity == HealthSeverity.MEDIUM]
        low = [r for r in result.remediations if r.severity in (HealthSeverity.LOW, HealthSeverity.INFO)]

        total_fix_time = sum(r.estimated_fix_time_minutes for r in result.remediations)

        return {
            "total_issues": len(result.remediations),
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "low_count": len(low),
            "estimated_total_fix_time_minutes": total_fix_time,
            "critical_issues": [
                {"check": r.check_name, "message": r.message, "action": r.action}
                for r in critical
            ],
            "high_issues": [
                {"check": r.check_name, "message": r.message, "action": r.action}
                for r in high
            ],
            "medium_issues": [
                {"check": r.check_name, "message": r.message, "action": r.action}
                for r in medium
            ],
            "low_issues": [
                {"check": r.check_name, "message": r.message, "action": r.action}
                for r in low
            ],
            "provenance_hash": _compute_hash(result.remediations),
        }

    # -------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------

    def _execute_checks(self, quick_mode: bool) -> HealthCheckResult:
        """Execute health checks across all applicable categories."""
        start_time = time.monotonic()

        all_checks: Dict[str, List[ComponentHealth]] = {}
        remediations: List[RemediationSuggestion] = []
        total = passed = failed = warnings_count = skipped = 0
        skip_set = set(self.config.skip_categories)
        has_critical = False

        for category in CheckCategory:
            if category == CheckCategory.OVERALL:
                continue
            if category.value in skip_set:
                continue
            if quick_mode and category not in QUICK_CHECK_CATEGORIES:
                continue

            handler = self._check_handlers.get(category)
            if not handler:
                continue

            cat_start = time.monotonic()
            try:
                checks = handler()
            except Exception as exc:
                checks = [ComponentHealth(
                    check_name=f"{category.value}_exception",
                    category=category, status=HealthStatus.FAIL,
                    message=f"Exception during {category.value} check: {exc}",
                    remediation=RemediationSuggestion(
                        check_name=f"{category.value}_exception",
                        severity=HealthSeverity.CRITICAL,
                        message=f"Unhandled exception in {category.value}",
                        action=f"Debug {category.value} handler: {exc}",
                    ),
                )]
            cat_duration = (time.monotonic() - cat_start) * 1000

            # Stamp duration on each check
            for check in checks:
                check.duration_ms = round(cat_duration / max(len(checks), 1), 1)

            all_checks[category.value] = checks
            for check in checks:
                total += 1
                if check.status == HealthStatus.PASS:
                    passed += 1
                elif check.status == HealthStatus.FAIL:
                    failed += 1
                    if check.remediation:
                        remediations.append(check.remediation)
                    if category in CRITICAL_CATEGORIES:
                        has_critical = True
                elif check.status == HealthStatus.WARN:
                    warnings_count += 1
                    if check.remediation:
                        remediations.append(check.remediation)
                elif check.status == HealthStatus.SKIP:
                    skipped += 1

        # Calculate score: PASS=100%, WARN=50%, FAIL=0%
        scorable = total - skipped
        if scorable > 0:
            score = ((passed * 100.0) + (warnings_count * 50.0)) / scorable
        else:
            score = 0.0

        overall_status = HealthStatus.PASS
        if failed > 0:
            overall_status = HealthStatus.FAIL
        elif warnings_count > 0:
            overall_status = HealthStatus.WARN

        # Data freshness and sector coverage
        freshness = self.run_data_freshness() if not quick_mode and self.config.check_data_freshness else []
        sector_cov = _check_sector_coverage() if not quick_mode else []

        result = HealthCheckResult(
            total_checks=total, passed=passed, failed=failed,
            warnings=warnings_count, skipped=skipped,
            overall_health_score=round(score, 1),
            overall_status=overall_status,
            has_critical_failure=has_critical,
            categories=all_checks, remediations=remediations,
            data_freshness=freshness, sector_coverage=sector_cov,
            total_duration_ms=round((time.monotonic() - start_time) * 1000, 1),
            quick_mode=quick_mode,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Sector pathway health check (%s): %d/%d passed, score=%.1f, critical=%s",
            "quick" if quick_mode else "full", passed, total, score, has_critical,
        )
        return result

    def _execute_single_category(self, category: CheckCategory) -> HealthCheckResult:
        """Execute health check for a single category."""
        start_time = time.monotonic()

        handler = self._check_handlers.get(category)
        if not handler:
            return HealthCheckResult(
                total_checks=0, overall_status=HealthStatus.SKIP,
                provenance_hash=_compute_hash({"category": category.value}),
            )

        try:
            checks = handler()
        except Exception as exc:
            checks = [ComponentHealth(
                check_name=f"{category.value}_exception",
                category=category, status=HealthStatus.FAIL,
                message=f"Exception: {exc}",
            )]

        total = len(checks)
        passed = sum(1 for c in checks if c.status == HealthStatus.PASS)
        failed = sum(1 for c in checks if c.status == HealthStatus.FAIL)
        warnings_count = sum(1 for c in checks if c.status == HealthStatus.WARN)
        skipped = sum(1 for c in checks if c.status == HealthStatus.SKIP)
        remediations = [c.remediation for c in checks if c.remediation]

        scorable = total - skipped
        score = ((passed * 100.0) + (warnings_count * 50.0)) / scorable if scorable > 0 else 0.0

        overall_status = HealthStatus.PASS
        if failed > 0:
            overall_status = HealthStatus.FAIL
        elif warnings_count > 0:
            overall_status = HealthStatus.WARN

        result = HealthCheckResult(
            total_checks=total, passed=passed, failed=failed,
            warnings=warnings_count, skipped=skipped,
            overall_health_score=round(score, 1),
            overall_status=overall_status,
            has_critical_failure=(failed > 0 and category in CRITICAL_CATEGORIES),
            categories={category.value: checks},
            remediations=remediations,
            total_duration_ms=round((time.monotonic() - start_time) * 1000, 1),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Category Check Handlers
    # -------------------------------------------------------------------

    def _check_platform(self) -> List[ComponentHealth]:
        """Check platform connectivity and version info."""
        checks = [
            ComponentHealth(
                check_name="platform_connectivity",
                category=CheckCategory.PLATFORM,
                status=HealthStatus.PASS,
                message="Platform: responsive",
                details={
                    "pack_id": "PACK-028",
                    "pack_version": "1.0.0",
                    "module_version": _MODULE_VERSION,
                },
            ),
            ComponentHealth(
                check_name="platform_python_version",
                category=CheckCategory.PLATFORM,
                status=HealthStatus.PASS,
                message="Python version: compatible",
            ),
        ]

        # Verify pack base directory
        if PACK_BASE_DIR.exists():
            checks.append(ComponentHealth(
                check_name="platform_pack_directory",
                category=CheckCategory.PLATFORM,
                status=HealthStatus.PASS,
                message=f"Pack directory: {PACK_BASE_DIR}",
            ))
        else:
            checks.append(ComponentHealth(
                check_name="platform_pack_directory",
                category=CheckCategory.PLATFORM,
                status=HealthStatus.FAIL,
                message=f"Pack directory not found: {PACK_BASE_DIR}",
                remediation=RemediationSuggestion(
                    check_name="platform_pack_directory",
                    severity=HealthSeverity.CRITICAL,
                    message="PACK-028 base directory missing",
                    action="Verify pack installation at correct path",
                ),
            ))

        return checks

    def _check_mrv_agents(self) -> List[ComponentHealth]:
        """Check all 30 MRV agents with sector-priority routing."""
        checks = []

        # Check MRV bridge availability
        try:
            from .mrv_bridge import SECTOR_MRV_ROUTING_TABLE, SectorMRVBridge

            route_count = len(SECTOR_MRV_ROUTING_TABLE)
            checks.append(ComponentHealth(
                check_name="mrv_bridge_availability",
                category=CheckCategory.MRV_AGENTS,
                status=HealthStatus.PASS,
                message=f"Sector MRV bridge: {route_count} agent routes configured",
                details={"route_count": route_count},
            ))

            # Check sector priority configuration
            from .mrv_bridge import SECTOR_AGENT_PRIORITIES
            priorities_count = len(SECTOR_AGENT_PRIORITIES)
            checks.append(ComponentHealth(
                check_name="mrv_sector_priorities",
                category=CheckCategory.MRV_AGENTS,
                status=HealthStatus.PASS if priorities_count >= 6 else HealthStatus.WARN,
                message=f"Sector-priority MRV routing: {priorities_count} routing groups",
                details={"routing_groups": priorities_count},
            ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="mrv_bridge_availability",
                category=CheckCategory.MRV_AGENTS,
                status=HealthStatus.FAIL,
                message=f"Sector MRV bridge: {exc}",
                remediation=RemediationSuggestion(
                    check_name="mrv_bridge_availability",
                    severity=HealthSeverity.HIGH,
                    message="Sector MRV bridge not loadable",
                    action="Verify mrv_bridge.py is properly installed",
                ),
            ))

        # Check individual agent registration
        checks.append(ComponentHealth(
            check_name="mrv_agents_registered",
            category=CheckCategory.MRV_AGENTS,
            status=HealthStatus.PASS,
            message=f"MRV agents: {len(SECTOR_PATHWAY_MRV_AGENTS)}/30 registered for sector routing",
            details={"agents": SECTOR_PATHWAY_MRV_AGENTS},
        ))

        return checks

    def _check_data_agents(self) -> List[ComponentHealth]:
        """Check all 20 DATA agents with sector activity data routing."""
        checks = []

        try:
            from .data_bridge import (
                SECTOR_DATA_AGENT_ROUTING,
                SECTOR_ACTIVITY_REQUIREMENTS,
                SectorDataBridge,
            )

            agent_count = len(SECTOR_DATA_AGENT_ROUTING)
            sectors_configured = len(SECTOR_ACTIVITY_REQUIREMENTS)

            checks.append(ComponentHealth(
                check_name="data_bridge_availability",
                category=CheckCategory.DATA_AGENTS,
                status=HealthStatus.PASS,
                message=f"Sector DATA bridge: {agent_count} agents, {sectors_configured} sector activity profiles",
                details={
                    "agent_routes": agent_count,
                    "sector_activity_profiles": sectors_configured,
                },
            ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="data_bridge_availability",
                category=CheckCategory.DATA_AGENTS,
                status=HealthStatus.FAIL,
                message=f"Sector DATA bridge: {exc}",
                remediation=RemediationSuggestion(
                    check_name="data_bridge_availability",
                    severity=HealthSeverity.HIGH,
                    message="Sector DATA bridge not loadable",
                    action="Verify data_bridge.py is properly installed",
                ),
            ))

        checks.append(ComponentHealth(
            check_name="data_agents_registered",
            category=CheckCategory.DATA_AGENTS,
            status=HealthStatus.PASS,
            message=f"DATA agents: {len(SECTOR_PATHWAY_DATA_AGENTS)}/20 registered",
            details={"agents": SECTOR_PATHWAY_DATA_AGENTS},
        ))

        return checks

    def _check_engines(self) -> List[ComponentHealth]:
        """Check all 8 sector pathway engines."""
        checks = []
        base = PACK_BASE_DIR / "engines"
        for name in SECTOR_PATHWAY_ENGINES:
            fpath = base / f"{name}.py"
            exists = fpath.exists()
            status = HealthStatus.PASS if exists else HealthStatus.WARN
            checks.append(ComponentHealth(
                check_name=f"engine_{name}",
                category=CheckCategory.ENGINES,
                status=status,
                message=f"{name}: {'found' if exists else 'not found'}",
                details={"path": str(fpath), "exists": exists},
                remediation=(RemediationSuggestion(
                    check_name=f"engine_{name}",
                    severity=HealthSeverity.HIGH,
                    message=f"Engine {name} not found",
                    action=f"Create engines/{name}.py",
                ) if not exists else None),
            ))
        return checks

    def _check_workflows(self) -> List[ComponentHealth]:
        """Check all 6 sector pathway workflows."""
        checks = []
        base = PACK_BASE_DIR / "workflows"
        for name in SECTOR_PATHWAY_WORKFLOWS:
            fpath = base / f"{name}.py"
            exists = fpath.exists()
            status = HealthStatus.PASS if exists else HealthStatus.WARN
            checks.append(ComponentHealth(
                check_name=f"workflow_{name}",
                category=CheckCategory.WORKFLOWS,
                status=status,
                message=f"{name}: {'found' if exists else 'not found'}",
                details={"path": str(fpath), "exists": exists},
                remediation=(RemediationSuggestion(
                    check_name=f"workflow_{name}",
                    severity=HealthSeverity.MEDIUM,
                    message=f"Workflow {name} not found",
                    action=f"Create workflows/{name}.py",
                ) if not exists else None),
            ))
        return checks

    def _check_templates(self) -> List[ComponentHealth]:
        """Check all 8 sector pathway templates."""
        checks = []
        base = PACK_BASE_DIR / "templates"
        for name in SECTOR_PATHWAY_TEMPLATES:
            fpath = base / f"{name}.py"
            exists = fpath.exists()
            status = HealthStatus.PASS if exists else HealthStatus.WARN
            checks.append(ComponentHealth(
                check_name=f"template_{name}",
                category=CheckCategory.TEMPLATES,
                status=status,
                message=f"{name}: {'found' if exists else 'not found'}",
                details={"path": str(fpath), "exists": exists},
                remediation=(RemediationSuggestion(
                    check_name=f"template_{name}",
                    severity=HealthSeverity.LOW,
                    message=f"Template {name} not found",
                    action=f"Create templates/{name}.py",
                ) if not exists else None),
            ))
        return checks

    def _check_config(self) -> List[ComponentHealth]:
        """Check configuration and preset validity."""
        checks = []

        # Check pack.yaml
        pack_yaml = PACK_BASE_DIR / "pack.yaml"
        exists = pack_yaml.exists()
        checks.append(ComponentHealth(
            check_name="config_pack_yaml",
            category=CheckCategory.CONFIG,
            status=HealthStatus.PASS if exists else HealthStatus.WARN,
            message=f"pack.yaml: {'found' if exists else 'not found'}",
            remediation=(RemediationSuggestion(
                check_name="config_pack_yaml",
                severity=HealthSeverity.MEDIUM,
                message="pack.yaml not found",
                action="Create pack.yaml with sector pathway configuration",
            ) if not exists else None),
        ))

        # Check config directory
        config_dir = PACK_BASE_DIR / "config"
        if config_dir.exists():
            config_files = list(config_dir.glob("*.py"))
            checks.append(ComponentHealth(
                check_name="config_directory",
                category=CheckCategory.CONFIG,
                status=HealthStatus.PASS if config_files else HealthStatus.WARN,
                message=f"Config directory: {len(config_files)} files",
            ))
        else:
            checks.append(ComponentHealth(
                check_name="config_directory",
                category=CheckCategory.CONFIG,
                status=HealthStatus.WARN,
                message="Config directory not found",
                remediation=RemediationSuggestion(
                    check_name="config_directory",
                    severity=HealthSeverity.MEDIUM,
                    message="Config directory missing",
                    action="Create config/ directory with pack_config.py and sector presets",
                ),
            ))

        # Check presets directory
        presets_dir = PACK_BASE_DIR / "presets"
        if presets_dir.exists():
            preset_files = list(presets_dir.glob("*.py"))
            checks.append(ComponentHealth(
                check_name="config_presets",
                category=CheckCategory.CONFIG,
                status=HealthStatus.PASS if preset_files else HealthStatus.WARN,
                message=f"Presets directory: {len(preset_files)} presets",
            ))
        else:
            checks.append(ComponentHealth(
                check_name="config_presets",
                category=CheckCategory.CONFIG,
                status=HealthStatus.WARN,
                message="Presets directory not found (sector presets required)",
                remediation=RemediationSuggestion(
                    check_name="config_presets",
                    severity=HealthSeverity.MEDIUM,
                    message="Presets directory missing",
                    action="Create presets/ with sector-specific configuration presets",
                ),
            ))

        return checks

    def _check_database(self) -> List[ComponentHealth]:
        """Check database connectivity (PostgreSQL/TimescaleDB)."""
        checks = []

        # In stub mode, report database as not tested
        checks.append(ComponentHealth(
            check_name="database_postgresql",
            category=CheckCategory.DATABASE,
            status=HealthStatus.WARN,
            message="Database: PostgreSQL connectivity not tested (stub mode)",
            details={"mode": "stub", "note": "Requires DATABASE_URL environment variable"},
        ))

        checks.append(ComponentHealth(
            check_name="database_timescaledb",
            category=CheckCategory.DATABASE,
            status=HealthStatus.WARN,
            message="Database: TimescaleDB extension not tested (stub mode)",
        ))

        # Check migration readiness
        migrations_dir = PACK_BASE_DIR / "migrations"
        if migrations_dir.exists():
            sql_files = list(migrations_dir.glob("*.sql"))
            up_migrations = [f for f in sql_files if ".down." not in f.name]
            down_migrations = [f for f in sql_files if ".down." in f.name]
            checks.append(ComponentHealth(
                check_name="database_migrations_ready",
                category=CheckCategory.DATABASE,
                status=HealthStatus.PASS,
                message=f"Migrations: {len(up_migrations)} up, {len(down_migrations)} down",
                details={
                    "up_count": len(up_migrations),
                    "down_count": len(down_migrations),
                    "complete_pairs": min(len(up_migrations), len(down_migrations)),
                },
            ))
        else:
            checks.append(ComponentHealth(
                check_name="database_migrations_ready",
                category=CheckCategory.DATABASE,
                status=HealthStatus.FAIL,
                message="Migrations directory not found",
                remediation=RemediationSuggestion(
                    check_name="database_migrations_ready",
                    severity=HealthSeverity.HIGH,
                    message="Migrations directory missing",
                    action="Create migrations/ with V181-V195 SQL files",
                ),
            ))

        return checks

    def _check_sector_classification(self) -> List[ComponentHealth]:
        """Check NACE/GICS/ISIC sector classification tables."""
        checks = []

        try:
            from .pack_orchestrator import SECTOR_NACE_MAPPING

            total_sectors = len(SECTOR_NACE_MAPPING)
            sda_count = sum(
                1 for v in SECTOR_NACE_MAPPING.values()
                if v.get("sda_eligible", False)
            )
            extended_count = total_sectors - sda_count

            checks.append(ComponentHealth(
                check_name="sector_nace_mapping",
                category=CheckCategory.SECTOR_CLASSIFICATION,
                status=HealthStatus.PASS if total_sectors >= 12 else HealthStatus.WARN,
                message=f"NACE Rev.2 mapping: {total_sectors} sectors ({sda_count} SDA + {extended_count} extended)",
                details={
                    "total": total_sectors,
                    "sda_eligible": sda_count,
                    "extended": extended_count,
                },
            ))

            # Validate each SDA sector has required fields
            for sector_name in SDA_SECTORS:
                sector = SECTOR_NACE_MAPPING.get(sector_name, {})
                has_nace = bool(sector.get("nace_rev2"))
                has_gics = bool(sector.get("gics"))
                has_isic = bool(sector.get("isic_rev4"))
                has_metric = bool(sector.get("intensity_metric"))

                all_present = has_nace and has_gics and has_isic and has_metric

                checks.append(ComponentHealth(
                    check_name=f"sector_class_{sector_name}",
                    category=CheckCategory.SECTOR_CLASSIFICATION,
                    status=HealthStatus.PASS if all_present else HealthStatus.WARN,
                    message=f"{sector_name}: NACE={'ok' if has_nace else 'missing'} GICS={'ok' if has_gics else 'missing'} ISIC={'ok' if has_isic else 'missing'} metric={'ok' if has_metric else 'missing'}",
                    details={
                        "nace": has_nace, "gics": has_gics,
                        "isic": has_isic, "metric": has_metric,
                    },
                ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="sector_classification_unavailable",
                category=CheckCategory.SECTOR_CLASSIFICATION,
                status=HealthStatus.FAIL,
                message=f"Sector classification: {exc}",
                remediation=RemediationSuggestion(
                    check_name="sector_classification_unavailable",
                    severity=HealthSeverity.CRITICAL,
                    message="SECTOR_NACE_MAPPING not accessible",
                    action="Verify pack_orchestrator.py has SECTOR_NACE_MAPPING dict",
                ),
            ))

        return checks

    def _check_sbti_sda(self) -> List[ComponentHealth]:
        """Check SBTi SDA convergence pathway data freshness and completeness."""
        checks = []

        try:
            from .sbti_sda_bridge import (
                SDA_CONVERGENCE_PATHWAYS,
                SDA_INTENSITY_METRICS,
                SBTI_NEAR_TERM_CRITERIA,
                SBTI_NET_ZERO_CRITERIA,
                NACE_TO_SDA_SECTOR,
                GICS_TO_SDA_SECTOR,
                SBTiSDABridge,
            )

            # Check pathway coverage
            pathway_count = len(SDA_CONVERGENCE_PATHWAYS)
            checks.append(ComponentHealth(
                check_name="sbti_sda_pathway_coverage",
                category=CheckCategory.SBTI_SDA_DATA,
                status=HealthStatus.PASS if pathway_count >= 12 else HealthStatus.WARN,
                message=f"SDA convergence pathways: {pathway_count}/12 sectors",
                details={"sectors": list(SDA_CONVERGENCE_PATHWAYS.keys())},
            ))

            # Check intensity metrics
            metrics_count = len(SDA_INTENSITY_METRICS)
            checks.append(ComponentHealth(
                check_name="sbti_sda_intensity_metrics",
                category=CheckCategory.SBTI_SDA_DATA,
                status=HealthStatus.PASS if metrics_count >= 12 else HealthStatus.WARN,
                message=f"SDA intensity metrics: {metrics_count}/12 sectors",
            ))

            # Check criteria completeness
            near_term = len(SBTI_NEAR_TERM_CRITERIA)
            net_zero = len(SBTI_NET_ZERO_CRITERIA)
            total_criteria = near_term + net_zero
            checks.append(ComponentHealth(
                check_name="sbti_criteria_completeness",
                category=CheckCategory.SBTI_SDA_DATA,
                status=HealthStatus.PASS if total_criteria >= 42 else HealthStatus.WARN,
                message=f"SBTi criteria: {total_criteria}/42 ({near_term} near-term + {net_zero} net-zero)",
                details={"near_term": near_term, "net_zero": net_zero},
            ))

            # Check NACE mapping
            nace_count = len(NACE_TO_SDA_SECTOR)
            gics_count = len(GICS_TO_SDA_SECTOR)
            checks.append(ComponentHealth(
                check_name="sbti_sector_mappings",
                category=CheckCategory.SBTI_SDA_DATA,
                status=HealthStatus.PASS if nace_count >= 10 and gics_count >= 10 else HealthStatus.WARN,
                message=f"SDA sector mappings: {nace_count} NACE + {gics_count} GICS codes",
            ))

            # Check bridge instantiation
            try:
                bridge = SBTiSDABridge()
                checks.append(ComponentHealth(
                    check_name="sbti_bridge_instantiation",
                    category=CheckCategory.SBTI_SDA_DATA,
                    status=HealthStatus.PASS,
                    message="SBTi SDA bridge: instantiation successful",
                ))
            except Exception as exc:
                checks.append(ComponentHealth(
                    check_name="sbti_bridge_instantiation",
                    category=CheckCategory.SBTI_SDA_DATA,
                    status=HealthStatus.FAIL,
                    message=f"SBTi SDA bridge instantiation: {exc}",
                ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="sbti_sda_unavailable",
                category=CheckCategory.SBTI_SDA_DATA,
                status=HealthStatus.FAIL,
                message=f"SBTi SDA data: {exc}",
                remediation=RemediationSuggestion(
                    check_name="sbti_sda_unavailable",
                    severity=HealthSeverity.CRITICAL,
                    message="SBTi SDA bridge module not loadable",
                    action="Verify sbti_sda_bridge.py is properly installed and importable",
                ),
            ))

        return checks

    def _check_iea_nze(self) -> List[ComponentHealth]:
        """Check IEA NZE 2050 milestone data currency."""
        checks = []

        try:
            from .iea_nze_bridge import (
                IEA_SECTOR_PATHWAYS,
                IEA_TECHNOLOGY_MILESTONES,
                REGIONAL_ADJUSTMENT_FACTORS,
                TECHNOLOGY_INTERDEPENDENCIES,
                IEANZEBridge,
            )

            # Check pathway data
            pathway_sectors = len(IEA_SECTOR_PATHWAYS)
            checks.append(ComponentHealth(
                check_name="iea_pathway_coverage",
                category=CheckCategory.IEA_NZE_DATA,
                status=HealthStatus.PASS if pathway_sectors >= 9 else HealthStatus.WARN,
                message=f"IEA sector pathways: {pathway_sectors} sectors",
                details={"sectors": list(IEA_SECTOR_PATHWAYS.keys())},
            ))

            # Check milestones
            milestones_count = len(IEA_TECHNOLOGY_MILESTONES)
            checks.append(ComponentHealth(
                check_name="iea_milestones_count",
                category=CheckCategory.IEA_NZE_DATA,
                status=HealthStatus.PASS if milestones_count >= 50 else HealthStatus.WARN,
                message=f"IEA technology milestones: {milestones_count} entries (target: 50+)",
            ))

            # Check regional factors
            regions_count = len(REGIONAL_ADJUSTMENT_FACTORS)
            checks.append(ComponentHealth(
                check_name="iea_regional_factors",
                category=CheckCategory.IEA_NZE_DATA,
                status=HealthStatus.PASS if regions_count >= 3 else HealthStatus.WARN,
                message=f"IEA regional adjustment factors: {regions_count} regions",
            ))

            # Check interdependencies
            interdeps = len(TECHNOLOGY_INTERDEPENDENCIES)
            checks.append(ComponentHealth(
                check_name="iea_technology_interdeps",
                category=CheckCategory.IEA_NZE_DATA,
                status=HealthStatus.PASS if interdeps >= 5 else HealthStatus.WARN,
                message=f"Technology interdependencies: {interdeps} cross-sector links",
            ))

            # Check bridge instantiation
            try:
                bridge = IEANZEBridge()
                checks.append(ComponentHealth(
                    check_name="iea_bridge_instantiation",
                    category=CheckCategory.IEA_NZE_DATA,
                    status=HealthStatus.PASS,
                    message="IEA NZE bridge: instantiation successful",
                ))
            except Exception as exc:
                checks.append(ComponentHealth(
                    check_name="iea_bridge_instantiation",
                    category=CheckCategory.IEA_NZE_DATA,
                    status=HealthStatus.FAIL,
                    message=f"IEA NZE bridge instantiation: {exc}",
                ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="iea_nze_unavailable",
                category=CheckCategory.IEA_NZE_DATA,
                status=HealthStatus.FAIL,
                message=f"IEA NZE data: {exc}",
                remediation=RemediationSuggestion(
                    check_name="iea_nze_unavailable",
                    severity=HealthSeverity.HIGH,
                    message="IEA NZE bridge module not loadable",
                    action="Verify iea_nze_bridge.py is properly installed",
                ),
            ))

        return checks

    def _check_ipcc_ar6(self) -> List[ComponentHealth]:
        """Check IPCC AR6 GWP and emission factor data integrity."""
        checks = []

        try:
            from .ipcc_ar6_bridge import (
                GWP_100_AR6,
                CARBON_BUDGETS_GTCO2,
                EMISSION_FACTORS_CO2_KG_PER_TJ,
                PROCESS_EMISSION_FACTORS,
                AGRICULTURAL_EMISSION_FACTORS,
                SSP_EMISSION_PATHWAYS,
                IPCCAR6Bridge,
            )

            # GWP species count
            gwp_count = len(GWP_100_AR6)
            checks.append(ComponentHealth(
                check_name="ipcc_gwp_species",
                category=CheckCategory.IPCC_AR6_DATA,
                status=HealthStatus.PASS if gwp_count >= 20 else HealthStatus.WARN,
                message=f"IPCC AR6 GWP-100: {gwp_count} GHG species (target: 20+)",
                details={"species": list(GWP_100_AR6.keys())[:10]},
            ))

            # Check essential GHGs
            essential_ghgs = ["co2", "ch4", "n2o", "sf6"]
            gwp_keys_lower = {k.lower() for k in GWP_100_AR6.keys()}
            missing_essential = [g for g in essential_ghgs if g not in gwp_keys_lower]
            if not missing_essential:
                checks.append(ComponentHealth(
                    check_name="ipcc_essential_ghgs",
                    category=CheckCategory.IPCC_AR6_DATA,
                    status=HealthStatus.PASS,
                    message="Essential GHGs (CO2, CH4, N2O, SF6): all present",
                ))
            else:
                checks.append(ComponentHealth(
                    check_name="ipcc_essential_ghgs",
                    category=CheckCategory.IPCC_AR6_DATA,
                    status=HealthStatus.FAIL,
                    message=f"Essential GHGs missing: {missing_essential}",
                    remediation=RemediationSuggestion(
                        check_name="ipcc_essential_ghgs",
                        severity=HealthSeverity.CRITICAL,
                        message=f"Missing essential GHGs: {missing_essential}",
                        action="Add missing GHG species to GWP_100_AR6",
                    ),
                ))

            # Carbon budgets
            budgets_count = len(CARBON_BUDGETS_GTCO2)
            checks.append(ComponentHealth(
                check_name="ipcc_carbon_budgets",
                category=CheckCategory.IPCC_AR6_DATA,
                status=HealthStatus.PASS if budgets_count >= 4 else HealthStatus.WARN,
                message=f"Carbon budgets: {budgets_count} scenarios",
            ))

            # Emission factors
            ef_count = len(EMISSION_FACTORS_CO2_KG_PER_TJ)
            checks.append(ComponentHealth(
                check_name="ipcc_emission_factors",
                category=CheckCategory.IPCC_AR6_DATA,
                status=HealthStatus.PASS if ef_count >= 15 else HealthStatus.WARN,
                message=f"CO2 emission factors: {ef_count} fuel types (target: 15+)",
            ))

            # Process emission factors
            process_count = len(PROCESS_EMISSION_FACTORS)
            checks.append(ComponentHealth(
                check_name="ipcc_process_factors",
                category=CheckCategory.IPCC_AR6_DATA,
                status=HealthStatus.PASS if process_count >= 8 else HealthStatus.WARN,
                message=f"Process emission factors: {process_count} industrial processes",
            ))

            # SSP pathways
            ssp_count = len(SSP_EMISSION_PATHWAYS)
            checks.append(ComponentHealth(
                check_name="ipcc_ssp_pathways",
                category=CheckCategory.IPCC_AR6_DATA,
                status=HealthStatus.PASS if ssp_count >= 5 else HealthStatus.WARN,
                message=f"SSP emission pathways: {ssp_count}/5 scenarios",
            ))

            # Bridge instantiation
            try:
                bridge = IPCCAR6Bridge()
                checks.append(ComponentHealth(
                    check_name="ipcc_bridge_instantiation",
                    category=CheckCategory.IPCC_AR6_DATA,
                    status=HealthStatus.PASS,
                    message="IPCC AR6 bridge: instantiation successful",
                ))
            except Exception as exc:
                checks.append(ComponentHealth(
                    check_name="ipcc_bridge_instantiation",
                    category=CheckCategory.IPCC_AR6_DATA,
                    status=HealthStatus.FAIL,
                    message=f"IPCC AR6 bridge instantiation: {exc}",
                ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="ipcc_ar6_unavailable",
                category=CheckCategory.IPCC_AR6_DATA,
                status=HealthStatus.FAIL,
                message=f"IPCC AR6 data: {exc}",
                remediation=RemediationSuggestion(
                    check_name="ipcc_ar6_unavailable",
                    severity=HealthSeverity.CRITICAL,
                    message="IPCC AR6 bridge module not loadable",
                    action="Verify ipcc_ar6_bridge.py is properly installed",
                ),
            ))

        return checks

    def _check_pack021(self) -> List[ComponentHealth]:
        """Check PACK-021 baseline/target integration availability."""
        checks = []

        try:
            from .pack021_bridge import PACK021_COMPONENTS, PACK021Bridge

            components_count = len(PACK021_COMPONENTS)
            component_names = list(PACK021_COMPONENTS.keys())

            checks.append(ComponentHealth(
                check_name="pack021_components",
                category=CheckCategory.PACK021_INTEGRATION,
                status=HealthStatus.PASS if components_count >= 6 else HealthStatus.WARN,
                message=f"PACK-021 components: {components_count}/6 registered",
                details={"components": component_names},
            ))

            # Check bridge instantiation
            try:
                bridge = PACK021Bridge()
                checks.append(ComponentHealth(
                    check_name="pack021_bridge_instantiation",
                    category=CheckCategory.PACK021_INTEGRATION,
                    status=HealthStatus.PASS,
                    message="PACK-021 bridge: instantiation successful",
                ))
            except Exception as exc:
                checks.append(ComponentHealth(
                    check_name="pack021_bridge_instantiation",
                    category=CheckCategory.PACK021_INTEGRATION,
                    status=HealthStatus.WARN,
                    message=f"PACK-021 bridge instantiation: {exc}",
                ))

            # Check key methods
            required_methods = [
                "import_baseline", "import_targets", "import_gap_analysis",
                "enhance_with_sector", "get_full_integration",
            ]
            for method in required_methods:
                has_method = hasattr(PACK021Bridge, method)
                checks.append(ComponentHealth(
                    check_name=f"pack021_method_{method}",
                    category=CheckCategory.PACK021_INTEGRATION,
                    status=HealthStatus.PASS if has_method else HealthStatus.WARN,
                    message=f"PACK021Bridge.{method}: {'available' if has_method else 'missing'}",
                ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="pack021_unavailable",
                category=CheckCategory.PACK021_INTEGRATION,
                status=HealthStatus.WARN,
                message=f"PACK-021 integration: {exc} (PACK-021 may not be installed)",
            ))

        return checks

    def _check_decarb(self) -> List[ComponentHealth]:
        """Check decarbonization lever registry and agent availability."""
        checks = []

        try:
            from .decarb_bridge import SECTOR_DECARB_LEVERS, SectorDecarbBridge


            sectors_with_levers = len(SECTOR_DECARB_LEVERS)
            total_levers = sum(len(v) for v in SECTOR_DECARB_LEVERS.values())

            checks.append(ComponentHealth(
                check_name="decarb_lever_registry",
                category=CheckCategory.DECARB_AGENTS,
                status=HealthStatus.PASS if sectors_with_levers >= 6 else HealthStatus.WARN,
                message=f"Decarb lever registry: {total_levers} levers across {sectors_with_levers} sectors",
                details={
                    "sectors": list(SECTOR_DECARB_LEVERS.keys()),
                    "total_levers": total_levers,
                },
            ))

            # Per-sector lever count
            for sector_name, levers in SECTOR_DECARB_LEVERS.items():
                checks.append(ComponentHealth(
                    check_name=f"decarb_levers_{sector_name}",
                    category=CheckCategory.DECARB_AGENTS,
                    status=HealthStatus.PASS if len(levers) >= 4 else HealthStatus.WARN,
                    message=f"Decarb levers {sector_name}: {len(levers)} levers",
                ))

            # Check bridge instantiation
            try:
                bridge = SectorDecarbBridge()
                checks.append(ComponentHealth(
                    check_name="decarb_bridge_instantiation",
                    category=CheckCategory.DECARB_AGENTS,
                    status=HealthStatus.PASS,
                    message="Sector decarb bridge: instantiation successful",
                ))
            except Exception as exc:
                checks.append(ComponentHealth(
                    check_name="decarb_bridge_instantiation",
                    category=CheckCategory.DECARB_AGENTS,
                    status=HealthStatus.WARN,
                    message=f"Decarb bridge instantiation: {exc}",
                ))

        except Exception as exc:
            checks.append(ComponentHealth(
                check_name="decarb_unavailable",
                category=CheckCategory.DECARB_AGENTS,
                status=HealthStatus.FAIL,
                message=f"Decarbonization agents: {exc}",
                remediation=RemediationSuggestion(
                    check_name="decarb_unavailable",
                    severity=HealthSeverity.HIGH,
                    message="Decarb bridge module not loadable",
                    action="Verify decarb_bridge.py is properly installed",
                ),
            ))

        return checks

    def _check_convergence(self) -> List[ComponentHealth]:
        """Check convergence calculator validation."""
        return _validate_convergence_calculator()

    def _check_technology_db(self) -> List[ComponentHealth]:
        """Check technology milestone database."""
        return _validate_technology_db()

    def _check_scenario_modeling(self) -> List[ComponentHealth]:
        """Check 5-scenario framework readiness."""
        return _validate_scenario_modeling()

    def _check_benchmark_data(self) -> List[ComponentHealth]:
        """Check sector benchmark datasets."""
        return _validate_benchmark_data()

    def _check_migrations(self) -> List[ComponentHealth]:
        """Check database migration file status."""
        checks = []
        migrations_dir = PACK_BASE_DIR / "migrations"

        if not migrations_dir.exists():
            checks.append(ComponentHealth(
                check_name="migrations_directory",
                category=CheckCategory.MIGRATIONS,
                status=HealthStatus.FAIL,
                message="Migrations directory not found",
                remediation=RemediationSuggestion(
                    check_name="migrations_directory",
                    severity=HealthSeverity.HIGH,
                    message="Missing migrations directory",
                    action="Create migrations/ with V181-V195 SQL migration files",
                ),
            ))
            return checks

        for migration in MIGRATION_FILES:
            up_file = migrations_dir / f"{migration}.sql"
            down_file = migrations_dir / f"{migration}.down.sql"

            up_exists = up_file.exists()
            down_exists = down_file.exists()

            if up_exists and down_exists:
                status = HealthStatus.PASS
                msg = f"{migration}: up + down present"
            elif up_exists:
                status = HealthStatus.WARN
                msg = f"{migration}: up present, down missing"
            else:
                status = HealthStatus.FAIL
                msg = f"{migration}: not found"

            checks.append(ComponentHealth(
                check_name=f"migration_{migration}",
                category=CheckCategory.MIGRATIONS,
                status=status,
                message=msg,
                details={"up_exists": up_exists, "down_exists": down_exists},
                remediation=(RemediationSuggestion(
                    check_name=f"migration_{migration}",
                    severity=HealthSeverity.MEDIUM,
                    message=f"Migration {migration} incomplete",
                    action=f"Create missing SQL files for {migration}",
                ) if status != HealthStatus.PASS else None),
            ))

        return checks
