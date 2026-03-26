# -*- coding: utf-8 -*-
"""
PACK-043 Scope 3 Complete Pack - Shared Test Fixtures (conftest.py)
====================================================================

Provides pytest fixtures for the PACK-043 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Entity hierarchy with parent, subsidiaries, and JVs
  - Data maturity assessment across 15 Scope 3 categories
  - Product BOM fixtures for LCA integration
  - LCA lifecycle results (cradle-to-grave)
  - Scenario and MACC fixtures (5 interventions)
  - SBTi pathway fixtures (near-term and long-term targets)
  - Supplier programme fixtures (20 suppliers with targets)
  - Climate risk fixtures (transition and physical risks)
  - Base year data with recalculation triggers
  - PCAF portfolio fixtures (listed equity and bonds)
  - Assurance evidence fixtures
  - Pack configuration with presets

Fixture Categories:
  1.  Paths, constants, and dynamic module loading
  2.  Entity hierarchy fixtures (multi-entity boundary)
  3.  Data maturity assessment fixtures (15 categories)
  4.  Product BOM and LCA fixtures
  5.  Scenario modelling and MACC fixtures
  6.  SBTi pathway fixtures
  7.  Supplier programme fixtures
  8.  Climate risk fixtures
  9.  Base year and recalculation fixtures
  10. Sector-specific fixtures (PCAF, retail, cloud)
  11. Assurance evidence fixtures
  12. Pack configuration and presets

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-043 Scope 3 Complete
Date:    March 2026
"""

import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# =============================================================================
# Constants
# =============================================================================

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"

# Engine file mapping: logical name -> file name on disk
ENGINE_FILES = {
    "data_maturity": "data_maturity_engine.py",
    "lca_integration": "lca_integration_engine.py",
    "multi_entity_boundary": "multi_entity_boundary_engine.py",
    "scenario_modelling": "scenario_modelling_engine.py",
    "sbti_pathway": "sbti_pathway_engine.py",
    "supplier_programme": "supplier_programme_engine.py",
    "climate_risk": "climate_risk_engine.py",
    "base_year": "base_year_engine.py",
    "sector_specific": "sector_specific_engine.py",
    "assurance": "assurance_engine.py",
}

ENGINE_CLASSES = {
    "data_maturity": "DataMaturityEngine",
    "lca_integration": "LCAIntegrationEngine",
    "multi_entity_boundary": "MultiEntityBoundaryEngine",
    "scenario_modelling": "ScenarioModellingEngine",
    "sbti_pathway": "SBTiPathwayEngine",
    "supplier_programme": "SupplierProgrammeEngine",
    "climate_risk": "ClimateRiskEngine",
    "base_year": "BaseYearEngine",
    "sector_specific": "SectorSpecificEngine",
    "assurance": "AssuranceEngine",
}

WORKFLOW_FILES = {
    "maturity_assessment": "maturity_assessment_workflow.py",
    "lca_pipeline": "lca_pipeline_workflow.py",
    "boundary_consolidation": "boundary_consolidation_workflow.py",
    "scenario_analysis": "scenario_analysis_workflow.py",
    "sbti_target_setting": "sbti_target_setting_workflow.py",
    "supplier_engagement": "supplier_engagement_workflow.py",
    "risk_assessment": "risk_assessment_workflow.py",
    "full_pipeline": "full_pipeline_workflow.py",
}

WORKFLOW_CLASSES = {
    "maturity_assessment": "MaturityAssessmentWorkflow",
    "lca_pipeline": "LCAPipelineWorkflow",
    "boundary_consolidation": "BoundaryConsolidationWorkflow",
    "scenario_analysis": "ScenarioAnalysisWorkflow",
    "sbti_target_setting": "SBTiTargetSettingWorkflow",
    "supplier_engagement": "SupplierEngagementWorkflow",
    "risk_assessment": "RiskAssessmentWorkflow",
    "full_pipeline": "FullPipelineWorkflow",
}

WORKFLOW_PHASE_COUNTS = {
    "maturity_assessment": 4,
    "lca_pipeline": 5,
    "boundary_consolidation": 4,
    "scenario_analysis": 5,
    "sbti_target_setting": 6,
    "supplier_engagement": 5,
    "risk_assessment": 4,
    "full_pipeline": 8,
}

TEMPLATE_FILES = {
    "scope3_inventory_report": "scope3_inventory_report.py",
    "category_detail_report": "category_detail_report.py",
    "lca_product_report": "lca_product_report.py",
    "sbti_submission_package": "sbti_submission_package.py",
    "supplier_scorecard": "supplier_scorecard.py",
    "climate_risk_report": "climate_risk_report.py",
    "macc_chart": "macc_chart.py",
    "assurance_package": "assurance_package.py",
    "executive_summary": "executive_summary.py",
    "data_maturity_dashboard": "data_maturity_dashboard.py",
}

TEMPLATE_CLASSES = {
    "scope3_inventory_report": "Scope3InventoryReportTemplate",
    "category_detail_report": "CategoryDetailReportTemplate",
    "lca_product_report": "LCAProductReportTemplate",
    "sbti_submission_package": "SBTiSubmissionPackageTemplate",
    "supplier_scorecard": "SupplierScorecardTemplate",
    "climate_risk_report": "ClimateRiskReportTemplate",
    "macc_chart": "MACCChartTemplate",
    "assurance_package": "AssurancePackageTemplate",
    "executive_summary": "ExecutiveSummaryTemplate",
    "data_maturity_dashboard": "DataMaturityDashboardTemplate",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "pack042_bridge": "pack042_bridge.py",
    "pack041_bridge": "pack041_bridge.py",
    "lca_database_bridge": "lca_database_bridge.py",
    "sbti_bridge": "sbti_bridge.py",
    "tcfd_bridge": "tcfd_bridge.py",
    "cdp_bridge": "cdp_bridge.py",
    "supplier_data_bridge": "supplier_data_bridge.py",
    "climate_data_bridge": "climate_data_bridge.py",
    "erp_scope3_bridge": "erp_scope3_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "Scope3CompleteOrchestrator",
    "pack042_bridge": "Pack042Bridge",
    "pack041_bridge": "Pack041Bridge",
    "lca_database_bridge": "LCADatabaseBridge",
    "sbti_bridge": "SBTiBridge",
    "tcfd_bridge": "TCFDBridge",
    "cdp_bridge": "CDPBridge",
    "supplier_data_bridge": "SupplierDataBridge",
    "climate_data_bridge": "ClimateDataBridge",
    "erp_scope3_bridge": "ERPScope3Bridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
}

PRESET_NAMES = [
    "manufacturing_enterprise",
    "financial_institution",
    "retail_chain",
    "technology_company",
    "energy_utility",
    "food_agriculture",
    "healthcare",
    "sme_simplified",
]

# Scope 3 category definitions
SCOPE3_CATEGORIES = {
    1: "Purchased Goods and Services",
    2: "Capital Goods",
    3: "Fuel- and Energy-Related Activities",
    4: "Upstream Transportation and Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation and Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

# Data maturity tiers
MATURITY_TIERS = {
    1: "Spend-Based Estimates",
    2: "Average-Data Method",
    3: "Supplier-Specific Partial",
    4: "Supplier-Specific Full",
    5: "Primary Data / LCA",
}

# SBTi pathway constants
SBTI_15C_ANNUAL_RATE = Decimal("4.2")  # 4.2% per year for 1.5C
SBTI_WB2C_ANNUAL_RATE = Decimal("2.5")  # 2.5% per year for well-below 2C
SBTI_LONG_TERM_REDUCTION = Decimal("90")  # 90% by 2050
SBTI_COVERAGE_THRESHOLD = Decimal("67")  # 67% of Scope 3 must be covered


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location."""
    subdir_map = {
        "engines": ENGINES_DIR,
        "workflows": WORKFLOWS_DIR,
        "templates": TEMPLATES_DIR,
        "integrations": INTEGRATIONS_DIR,
        "config": CONFIG_DIR,
    }
    base_dir = subdir_map.get(subdir, PACK_ROOT / subdir)
    file_path = base_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Module file not found: {file_path}. "
            f"Ensure PACK-043 source files are present."
        )

    full_module_name = f"pack043_test.{subdir}.{module_name}"
    if full_module_name in sys.modules:
        return sys.modules[full_module_name]

    spec = importlib.util.spec_from_file_location(full_module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(full_module_name, None)
        raise ImportError(
            f"Failed to load module {full_module_name} from {file_path}: {exc}"
        ) from exc

    return module


def _load_engine(engine_key: str):
    """Load an engine module by its logical key."""
    file_name = ENGINE_FILES[engine_key]
    return _load_module(engine_key, file_name, "engines")


def _load_config_module():
    """Load the pack_config module."""
    return _load_module("pack_config", "pack_config.py", "config")


# =============================================================================
# Helper: Provenance hash utility
# =============================================================================


def compute_provenance_hash(data: Any) -> str:
    """Compute a SHA-256 provenance hash for any JSON-serializable data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
# Helper: Seeded RNG
# =============================================================================

_RNG = random.Random(43)


def _seeded_float(low: float, high: float) -> float:
    return _RNG.uniform(low, high)


def _seeded_int(low: int, high: int) -> int:
    return _RNG.randint(low, high)


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-043 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def engines_dir() -> Path:
    """Return the absolute path to the engines directory."""
    return ENGINES_DIR


@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="session")
def pack_yaml_data(pack_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the pack.yaml manifest as a dictionary."""
    if not pack_yaml_path.exists():
        pytest.skip("pack.yaml not found")
    with open(pack_yaml_path, "r", encoding="utf-8") as f:
        data = __import__("yaml").safe_load(f)
    assert data is not None, "pack.yaml parsed to None"
    return data


# =============================================================================
# 2. Entity Hierarchy Fixtures (Multi-Entity Boundary)
# =============================================================================


@pytest.fixture
def sample_entity_hierarchy() -> Dict[str, Any]:
    """Entity hierarchy: parent company + 5 subsidiaries + 2 JVs."""
    return {
        "group_id": "GRP-APEX-001",
        "group_name": "Apex Global Holdings",
        "reporting_year": 2025,
        "base_year": 2019,
        "default_approach": "operational_control",
        "parent": {
            "entity_id": "ENT-PARENT-001",
            "entity_name": "Apex Holdings Ltd",
            "country": "GB",
            "sector": "holding_company",
            "equity_pct": Decimal("100"),
            "has_operational_control": True,
            "has_financial_control": True,
            "scope3_tco2e": Decimal("5000"),
        },
        "subsidiaries": [
            {
                "entity_id": f"ENT-SUB-{i:03d}",
                "entity_name": name,
                "country": country,
                "sector": sector,
                "equity_pct": Decimal("100"),
                "has_operational_control": True,
                "has_financial_control": True,
                "scope3_tco2e": Decimal(str(emissions)),
            }
            for i, (name, country, sector, emissions) in enumerate([
                ("Apex Manufacturing US", "US", "manufacturing", 45000),
                ("Apex Europe GmbH", "DE", "manufacturing", 32000),
                ("Apex Logistics UK", "GB", "logistics", 18000),
                ("Apex Retail APAC", "SG", "retail", 25000),
                ("Apex Tech Ireland", "IE", "technology", 12000),
            ], start=1)
        ],
        "joint_ventures": [
            {
                "entity_id": "ENT-JV-001",
                "entity_name": "Apex-Sakura JV",
                "country": "JP",
                "sector": "manufacturing",
                "equity_pct": Decimal("50"),
                "has_operational_control": False,
                "has_financial_control": False,
                "scope3_tco2e": Decimal("22000"),
            },
            {
                "entity_id": "ENT-JV-002",
                "entity_name": "Apex-Brasil JV",
                "country": "BR",
                "sector": "agriculture",
                "equity_pct": Decimal("40"),
                "has_operational_control": False,
                "has_financial_control": False,
                "scope3_tco2e": Decimal("35000"),
            },
        ],
    }


# =============================================================================
# 3. Data Maturity Assessment Fixtures (15 Categories)
# =============================================================================


@pytest.fixture
def sample_maturity_assessment() -> Dict[str, Any]:
    """Maturity assessment for all 15 Scope 3 categories."""
    categories = {}
    current_tiers = [2, 1, 3, 2, 3, 4, 3, 1, 2, 1, 2, 1, 1, 1, 2]
    target_tiers =  [4, 3, 4, 4, 4, 5, 4, 3, 3, 3, 4, 3, 2, 2, 4]
    emissions =     [85000, 12000, 8000, 22000, 5000, 3000, 4500, 2000,
                     15000, 8000, 35000, 6000, 1500, 500, 45000]

    for cat_num in range(1, 16):
        idx = cat_num - 1
        categories[cat_num] = {
            "category_id": cat_num,
            "category_name": SCOPE3_CATEGORIES[cat_num],
            "current_tier": current_tiers[idx],
            "target_tier": target_tiers[idx],
            "current_tier_name": MATURITY_TIERS[current_tiers[idx]],
            "target_tier_name": MATURITY_TIERS[target_tiers[idx]],
            "emissions_tco2e": Decimal(str(emissions[idx])),
            "data_quality_score": Decimal(str(current_tiers[idx] * 20)),
            "upgrade_cost_usd": Decimal(str((target_tiers[idx] - current_tiers[idx]) * 25000)),
            "uncertainty_pct": Decimal(str(max(5, 50 - current_tiers[idx] * 10))),
        }

    total_emissions = sum(c["emissions_tco2e"] for c in categories.values())
    weighted_quality = sum(
        c["data_quality_score"] * c["emissions_tco2e"] / total_emissions
        for c in categories.values()
    )

    return {
        "assessment_id": "MAT-2025-001",
        "org_id": "GRP-APEX-001",
        "reporting_year": 2025,
        "categories": categories,
        "total_scope3_tco2e": total_emissions,
        "weighted_data_quality": weighted_quality,
        "total_upgrade_cost": sum(c["upgrade_cost_usd"] for c in categories.values()),
        "average_current_tier": Decimal(str(
            sum(current_tiers) / len(current_tiers)
        )).quantize(Decimal("0.01")),
        "average_target_tier": Decimal(str(
            sum(target_tiers) / len(target_tiers)
        )).quantize(Decimal("0.01")),
    }


# =============================================================================
# 4. Product BOM and LCA Fixtures
# =============================================================================


@pytest.fixture
def sample_product_bom() -> List[Dict[str, Any]]:
    """Three products with multi-component BOMs for LCA testing."""
    return [
        {
            "product_id": "PROD-001",
            "product_name": "Industrial Pump Model X200",
            "functional_unit": "1 unit, 15-year service life",
            "mass_kg": Decimal("85"),
            "components": [
                {
                    "component_id": "COMP-001-A",
                    "material": "cast_iron",
                    "mass_kg": Decimal("45"),
                    "emission_factor_kgco2e_per_kg": Decimal("1.91"),
                    "recycled_content_pct": Decimal("30"),
                    "supplier_id": "SUP-001",
                },
                {
                    "component_id": "COMP-001-B",
                    "material": "stainless_steel",
                    "mass_kg": Decimal("20"),
                    "emission_factor_kgco2e_per_kg": Decimal("6.15"),
                    "recycled_content_pct": Decimal("20"),
                    "supplier_id": "SUP-002",
                },
                {
                    "component_id": "COMP-001-C",
                    "material": "copper_winding",
                    "mass_kg": Decimal("12"),
                    "emission_factor_kgco2e_per_kg": Decimal("3.81"),
                    "recycled_content_pct": Decimal("15"),
                    "supplier_id": "SUP-003",
                },
                {
                    "component_id": "COMP-001-D",
                    "material": "rubber_seal",
                    "mass_kg": Decimal("5"),
                    "emission_factor_kgco2e_per_kg": Decimal("3.18"),
                    "recycled_content_pct": Decimal("0"),
                    "supplier_id": "SUP-004",
                },
                {
                    "component_id": "COMP-001-E",
                    "material": "electronics",
                    "mass_kg": Decimal("3"),
                    "emission_factor_kgco2e_per_kg": Decimal("25.0"),
                    "recycled_content_pct": Decimal("5"),
                    "supplier_id": "SUP-005",
                },
            ],
        },
        {
            "product_id": "PROD-002",
            "product_name": "Electric Motor Drive Unit",
            "functional_unit": "1 unit, 10-year service life",
            "mass_kg": Decimal("35"),
            "components": [
                {
                    "component_id": "COMP-002-A",
                    "material": "electrical_steel",
                    "mass_kg": Decimal("18"),
                    "emission_factor_kgco2e_per_kg": Decimal("2.80"),
                    "recycled_content_pct": Decimal("40"),
                    "supplier_id": "SUP-002",
                },
                {
                    "component_id": "COMP-002-B",
                    "material": "copper_winding",
                    "mass_kg": Decimal("10"),
                    "emission_factor_kgco2e_per_kg": Decimal("3.81"),
                    "recycled_content_pct": Decimal("15"),
                    "supplier_id": "SUP-003",
                },
                {
                    "component_id": "COMP-002-C",
                    "material": "aluminium_housing",
                    "mass_kg": Decimal("5"),
                    "emission_factor_kgco2e_per_kg": Decimal("8.24"),
                    "recycled_content_pct": Decimal("50"),
                    "supplier_id": "SUP-006",
                },
                {
                    "component_id": "COMP-002-D",
                    "material": "electronics",
                    "mass_kg": Decimal("2"),
                    "emission_factor_kgco2e_per_kg": Decimal("25.0"),
                    "recycled_content_pct": Decimal("5"),
                    "supplier_id": "SUP-005",
                },
            ],
        },
        {
            "product_id": "PROD-003",
            "product_name": "Control Valve Assembly",
            "functional_unit": "1 unit, 20-year service life",
            "mass_kg": Decimal("12"),
            "components": [
                {
                    "component_id": "COMP-003-A",
                    "material": "stainless_steel",
                    "mass_kg": Decimal("10"),
                    "emission_factor_kgco2e_per_kg": Decimal("6.15"),
                    "recycled_content_pct": Decimal("25"),
                    "supplier_id": "SUP-002",
                },
                {
                    "component_id": "COMP-003-B",
                    "material": "ptfe_seal",
                    "mass_kg": Decimal("1"),
                    "emission_factor_kgco2e_per_kg": Decimal("7.20"),
                    "recycled_content_pct": Decimal("0"),
                    "supplier_id": "SUP-007",
                },
                {
                    "component_id": "COMP-003-C",
                    "material": "pneumatic_actuator",
                    "mass_kg": Decimal("1"),
                    "emission_factor_kgco2e_per_kg": Decimal("12.0"),
                    "recycled_content_pct": Decimal("10"),
                    "supplier_id": "SUP-008",
                },
            ],
        },
    ]


@pytest.fixture
def sample_lca_results() -> Dict[str, Any]:
    """LCA lifecycle results for product PROD-001."""
    return {
        "product_id": "PROD-001",
        "methodology": "ISO_14067",
        "system_boundary": "cradle_to_grave",
        "phases": {
            "raw_material_extraction": {
                "tco2e": Decimal("210.5"),
                "pct_of_total": Decimal("28.1"),
            },
            "manufacturing": {
                "tco2e": Decimal("85.2"),
                "pct_of_total": Decimal("11.4"),
            },
            "distribution": {
                "tco2e": Decimal("32.0"),
                "pct_of_total": Decimal("4.3"),
            },
            "use_phase": {
                "tco2e": Decimal("380.0"),
                "pct_of_total": Decimal("50.7"),
                "assumptions": {
                    "electricity_kwh_per_year": Decimal("65800"),
                    "grid_factor_kgco2e_per_kwh": Decimal("0.385"),
                    "service_life_years": 15,
                    "operating_hours_per_year": 6000,
                },
            },
            "end_of_life": {
                "tco2e": Decimal("41.3"),
                "pct_of_total": Decimal("5.5"),
                "scenarios": {
                    "landfill": Decimal("52.0"),
                    "recycling": Decimal("25.0"),
                    "incineration": Decimal("47.0"),
                    "reuse": Decimal("12.0"),
                },
                "selected_scenario": "recycling_dominant",
            },
        },
        "total_tco2e": Decimal("749.0"),
        "total_per_functional_unit": Decimal("749.0"),
        "carbon_intensity_kgco2e_per_kg": Decimal("8.81"),
    }


# =============================================================================
# 5. Scenario Modelling and MACC Fixtures
# =============================================================================


@pytest.fixture
def sample_macc_interventions() -> List[Dict[str, Any]]:
    """Five interventions for MACC curve generation."""
    return [
        {
            "intervention_id": "INT-001",
            "name": "Supplier Engagement Programme",
            "category": "supplier_engagement",
            "abatement_tco2e": Decimal("15000"),
            "annual_cost_usd": Decimal("-200000"),
            "cost_per_tco2e": Decimal("-13.33"),
            "implementation_years": 2,
            "confidence": Decimal("0.75"),
            "capex_usd": Decimal("500000"),
            "opex_annual_usd": Decimal("-200000"),
        },
        {
            "intervention_id": "INT-002",
            "name": "Renewable Energy Procurement",
            "category": "energy_transition",
            "abatement_tco2e": Decimal("8000"),
            "annual_cost_usd": Decimal("50000"),
            "cost_per_tco2e": Decimal("6.25"),
            "implementation_years": 1,
            "confidence": Decimal("0.90"),
            "capex_usd": Decimal("0"),
            "opex_annual_usd": Decimal("50000"),
        },
        {
            "intervention_id": "INT-003",
            "name": "Logistics Optimization",
            "category": "logistics",
            "abatement_tco2e": Decimal("5500"),
            "annual_cost_usd": Decimal("120000"),
            "cost_per_tco2e": Decimal("21.82"),
            "implementation_years": 1,
            "confidence": Decimal("0.85"),
            "capex_usd": Decimal("300000"),
            "opex_annual_usd": Decimal("120000"),
        },
        {
            "intervention_id": "INT-004",
            "name": "Product Redesign for Circularity",
            "category": "product_design",
            "abatement_tco2e": Decimal("12000"),
            "annual_cost_usd": Decimal("350000"),
            "cost_per_tco2e": Decimal("29.17"),
            "implementation_years": 3,
            "confidence": Decimal("0.60"),
            "capex_usd": Decimal("2000000"),
            "opex_annual_usd": Decimal("350000"),
        },
        {
            "intervention_id": "INT-005",
            "name": "Carbon Removal Credits",
            "category": "offsetting",
            "abatement_tco2e": Decimal("10000"),
            "annual_cost_usd": Decimal("1500000"),
            "cost_per_tco2e": Decimal("150.00"),
            "implementation_years": 1,
            "confidence": Decimal("0.95"),
            "capex_usd": Decimal("0"),
            "opex_annual_usd": Decimal("1500000"),
        },
    ]


@pytest.fixture
def sample_scenario_config() -> Dict[str, Any]:
    """Configuration for scenario modelling."""
    return {
        "baseline_scope3_tco2e": Decimal("252500"),
        "target_year": 2030,
        "base_year": 2019,
        "reporting_year": 2025,
        "discount_rate": Decimal("0.08"),
        "carbon_price_usd_per_tco2e": Decimal("100"),
        "budget_constraint_usd": Decimal("5000000"),
        "alignment_pathway": "1.5C",
        "paris_trajectories": {
            "1.5C": {"annual_reduction_pct": Decimal("4.2")},
            "WB2C": {"annual_reduction_pct": Decimal("2.5")},
            "2C": {"annual_reduction_pct": Decimal("1.5")},
        },
    }


# =============================================================================
# 6. SBTi Pathway Fixtures
# =============================================================================


@pytest.fixture
def sample_sbti_targets() -> Dict[str, Any]:
    """SBTi target fixtures with near-term and long-term targets."""
    return {
        "org_id": "GRP-APEX-001",
        "base_year": 2019,
        "base_year_scope1_tco2e": Decimal("25000"),
        "base_year_scope2_tco2e": Decimal("16000"),
        "base_year_scope3_tco2e": Decimal("300000"),
        "base_year_total_tco2e": Decimal("341000"),
        "scope3_pct_of_total": Decimal("87.98"),
        "scope3_is_material": True,
        "near_term": {
            "target_year": 2030,
            "pathway": "1.5C",
            "annual_reduction_pct": SBTI_15C_ANNUAL_RATE,
            "scope3_coverage_pct": Decimal("72"),
            "covered_categories": [1, 2, 4, 5, 6, 7, 9, 11, 12, 15],
            "covered_emissions_tco2e": Decimal("216000"),
            "target_reduction_pct": Decimal("42"),
            "target_absolute_tco2e": Decimal("174000"),
        },
        "long_term": {
            "target_year": 2050,
            "pathway": "net_zero",
            "total_reduction_pct": SBTI_LONG_TERM_REDUCTION,
            "target_absolute_tco2e": Decimal("30000"),
        },
        "flag_pathway": {
            "applicable": False,
            "sectors": [],
        },
        "milestones": [
            {"year": 2025, "target_tco2e": Decimal("252500"), "status": "on_track"},
            {"year": 2027, "target_tco2e": Decimal("231000"), "status": "projected"},
            {"year": 2030, "target_tco2e": Decimal("174000"), "status": "target"},
        ],
    }


# =============================================================================
# 7. Supplier Programme Fixtures
# =============================================================================


@pytest.fixture
def sample_supplier_programme() -> Dict[str, Any]:
    """Supplier programme with 20 suppliers, targets, and commitments."""
    suppliers = []
    supplier_data = [
        ("SUP-001", "Steel Corp", "materials", 35000, "SBTi", True, Decimal("3.5")),
        ("SUP-002", "Metal Works AG", "materials", 22000, "SBTi", True, Decimal("4.2")),
        ("SUP-003", "Copper Global", "materials", 15000, "RE100", True, Decimal("2.1")),
        ("SUP-004", "Rubber Solutions", "materials", 8000, "CDP", True, Decimal("1.5")),
        ("SUP-005", "Electronics Co", "components", 12000, "SBTi", True, Decimal("5.0")),
        ("SUP-006", "Alu-Cast Ltd", "materials", 9500, "None", False, Decimal("0.0")),
        ("SUP-007", "Seal Tech", "components", 3000, "CDP", True, Decimal("1.0")),
        ("SUP-008", "Pneumatic Inc", "components", 4500, "None", False, Decimal("0.0")),
        ("SUP-009", "TransGlobal", "logistics", 18000, "SBTi", True, Decimal("3.8")),
        ("SUP-010", "ShipFast", "logistics", 12000, "CDP", True, Decimal("2.0")),
        ("SUP-011", "Packaging Pro", "packaging", 6000, "RE100", True, Decimal("4.5")),
        ("SUP-012", "Chemical Corp", "chemicals", 14000, "SBTi", True, Decimal("3.0")),
        ("SUP-013", "Energy Partner", "energy", 20000, "RE100", True, Decimal("8.0")),
        ("SUP-014", "Waste Mgmt Co", "waste", 5500, "CDP", True, Decimal("2.5")),
        ("SUP-015", "IT Services", "services", 3500, "SBTi", True, Decimal("6.0")),
        ("SUP-016", "Facility Mgmt", "services", 2000, "None", False, Decimal("0.0")),
        ("SUP-017", "Raw Mat Ltd", "materials", 28000, "SBTi", True, Decimal("2.8")),
        ("SUP-018", "Assembly Co", "manufacturing", 16000, "CDP", True, Decimal("1.8")),
        ("SUP-019", "Paint & Coat", "chemicals", 7000, "None", False, Decimal("0.0")),
        ("SUP-020", "Precision Parts", "components", 5000, "SBTi", True, Decimal("3.2")),
    ]

    for sup_id, name, category, emissions, commitment, has_target, yoy_reduction in supplier_data:
        suppliers.append({
            "supplier_id": sup_id,
            "supplier_name": name,
            "category": category,
            "scope3_contribution_tco2e": Decimal(str(emissions)),
            "commitment_type": commitment,
            "has_reduction_target": has_target,
            "yoy_reduction_pct": yoy_reduction,
            "tier": "critical" if emissions >= 15000 else "significant" if emissions >= 5000 else "standard",
            "engagement_score": Decimal(str(min(100, emissions // 200 + 40))),
        })

    total_supplier_emissions = sum(s["scope3_contribution_tco2e"] for s in suppliers)
    committed_emissions = sum(
        s["scope3_contribution_tco2e"] for s in suppliers if s["has_reduction_target"]
    )

    return {
        "programme_id": "SPP-2025-001",
        "org_id": "GRP-APEX-001",
        "reporting_year": 2025,
        "suppliers": suppliers,
        "total_suppliers": len(suppliers),
        "total_supplier_emissions_tco2e": total_supplier_emissions,
        "committed_suppliers": sum(1 for s in suppliers if s["has_reduction_target"]),
        "committed_emissions_tco2e": committed_emissions,
        "coverage_pct": committed_emissions / total_supplier_emissions * Decimal("100"),
        "programme_target_reduction_pct": Decimal("30"),
        "programme_target_year": 2030,
    }


# =============================================================================
# 8. Climate Risk Fixtures
# =============================================================================


@pytest.fixture
def sample_climate_risks() -> Dict[str, Any]:
    """Climate risk fixtures with transition and physical risks."""
    return {
        "org_id": "GRP-APEX-001",
        "reporting_year": 2025,
        "transition_risks": [
            {
                "risk_id": "TR-001",
                "risk_type": "carbon_pricing",
                "description": "EU ETS expansion to Scope 3",
                "likelihood": "high",
                "impact": "high",
                "carbon_price_scenarios": {
                    "low": {"price_usd_per_tco2e": Decimal("50"), "annual_exposure_usd": Decimal("12625000")},
                    "medium": {"price_usd_per_tco2e": Decimal("100"), "annual_exposure_usd": Decimal("25250000")},
                    "high": {"price_usd_per_tco2e": Decimal("150"), "annual_exposure_usd": Decimal("37875000")},
                    "extreme": {"price_usd_per_tco2e": Decimal("200"), "annual_exposure_usd": Decimal("50500000")},
                },
                "affected_scope3_tco2e": Decimal("252500"),
            },
            {
                "risk_id": "TR-002",
                "risk_type": "cbam",
                "description": "EU CBAM for imported materials",
                "likelihood": "high",
                "impact": "medium",
                "annual_exposure_usd": Decimal("8500000"),
                "affected_categories": [1, 2],
            },
            {
                "risk_id": "TR-003",
                "risk_type": "stranded_assets",
                "description": "High-carbon supply chain assets at risk",
                "likelihood": "medium",
                "impact": "high",
                "exposure_usd": Decimal("45000000"),
                "timeframe_years": 10,
            },
        ],
        "physical_risks": [
            {
                "risk_id": "PR-001",
                "risk_type": "chronic",
                "hazard": "sea_level_rise",
                "description": "Coastal supply chain disruption",
                "likelihood": "medium",
                "impact": "high",
                "affected_suppliers": 5,
                "annual_impact_usd": Decimal("3500000"),
            },
            {
                "risk_id": "PR-002",
                "risk_type": "acute",
                "hazard": "extreme_heat",
                "description": "Manufacturing productivity loss",
                "likelihood": "high",
                "impact": "medium",
                "affected_facilities": 3,
                "annual_impact_usd": Decimal("2000000"),
            },
        ],
        "opportunities": [
            {
                "opportunity_id": "OPP-001",
                "type": "resource_efficiency",
                "description": "Circular economy transition",
                "annual_value_usd": Decimal("5000000"),
                "investment_required_usd": Decimal("15000000"),
                "payback_years": 3,
            },
            {
                "opportunity_id": "OPP-002",
                "type": "products_services",
                "description": "Low-carbon product premium",
                "annual_value_usd": Decimal("8000000"),
                "investment_required_usd": Decimal("5000000"),
                "payback_years": Decimal("0.625"),
            },
        ],
        "scenario_analysis": {
            "iea_nze": {
                "aligned": False,
                "gap_pct": Decimal("35"),
                "required_reduction_by_2030": Decimal("42"),
            },
            "ngfs_orderly": {
                "carbon_price_2030": Decimal("130"),
                "total_exposure_2030": Decimal("32825000"),
            },
            "ngfs_disorderly": {
                "carbon_price_2030": Decimal("200"),
                "total_exposure_2030": Decimal("50500000"),
            },
        },
    }


# =============================================================================
# 9. Base Year and Recalculation Fixtures
# =============================================================================


@pytest.fixture
def sample_base_year_data() -> Dict[str, Any]:
    """Base year data with recalculation triggers."""
    return {
        "base_year": 2019,
        "scope3_total_tco2e": Decimal("300000"),
        "by_category": {
            1: Decimal("100000"),
            2: Decimal("15000"),
            3: Decimal("10000"),
            4: Decimal("25000"),
            5: Decimal("6000"),
            6: Decimal("4000"),
            7: Decimal("5500"),
            8: Decimal("2500"),
            9: Decimal("18000"),
            10: Decimal("10000"),
            11: Decimal("40000"),
            12: Decimal("8000"),
            13: Decimal("2000"),
            14: Decimal("1000"),
            15: Decimal("53000"),
        },
        "significance_threshold_pct": Decimal("5.0"),
        "recalculation_triggers": [
            {
                "trigger_id": "RECALC-001",
                "trigger_type": "acquisition",
                "description": "Acquired ChemTech Industries",
                "date": "2023-07-01",
                "impact_tco2e": Decimal("22000"),
                "impact_pct": Decimal("7.33"),
                "exceeds_threshold": True,
                "pro_rata_days": 184,
                "pro_rata_factor": Decimal("0.5041"),
            },
            {
                "trigger_id": "RECALC-002",
                "trigger_type": "divestiture",
                "description": "Divested Mining Operations",
                "date": "2024-01-01",
                "impact_tco2e": Decimal("-18000"),
                "impact_pct": Decimal("6.00"),
                "exceeds_threshold": True,
                "pro_rata_days": 365,
                "pro_rata_factor": Decimal("1.0"),
            },
            {
                "trigger_id": "RECALC-003",
                "trigger_type": "methodology",
                "description": "Changed Cat 1 from spend to hybrid",
                "date": "2024-06-01",
                "impact_tco2e": Decimal("-12000"),
                "impact_pct": Decimal("4.00"),
                "exceeds_threshold": False,
            },
            {
                "trigger_id": "RECALC-004",
                "trigger_type": "scope_expansion",
                "description": "Added Category 15 investments",
                "date": "2024-01-01",
                "impact_tco2e": Decimal("53000"),
                "impact_pct": Decimal("17.67"),
                "exceeds_threshold": True,
            },
            {
                "trigger_id": "RECALC-005",
                "trigger_type": "error_correction",
                "description": "Corrected Cat 4 emission factors",
                "date": "2025-01-15",
                "impact_tco2e": Decimal("-3500"),
                "impact_pct": Decimal("1.17"),
                "exceeds_threshold": False,
            },
            {
                "trigger_id": "RECALC-006",
                "trigger_type": "structural_change",
                "description": "Outsourced manufacturing to 3 new suppliers",
                "date": "2024-09-01",
                "impact_tco2e": Decimal("28000"),
                "impact_pct": Decimal("9.33"),
                "exceeds_threshold": True,
            },
        ],
        "adjusted_base_year_tco2e": Decimal("385000"),
        "yearly_actuals": {
            2019: Decimal("300000"),
            2020: Decimal("270000"),
            2021: Decimal("285000"),
            2022: Decimal("275000"),
            2023: Decimal("268000"),
            2024: Decimal("260000"),
            2025: Decimal("252500"),
        },
    }


# =============================================================================
# 10. Sector-Specific Fixtures (PCAF, Retail, Cloud)
# =============================================================================


@pytest.fixture
def sample_pcaf_portfolio() -> Dict[str, Any]:
    """PCAF financed emissions portfolio (listed equity + bonds)."""
    return {
        "portfolio_id": "PF-2025-001",
        "reporting_year": 2025,
        "asset_classes": {
            "listed_equity": {
                "investments": [
                    {
                        "investee_id": "INV-001",
                        "investee_name": "GreenTech Corp",
                        "sector": "technology",
                        "invested_amount_usd": Decimal("50000000"),
                        "evic_usd": Decimal("500000000"),
                        "investee_scope12_tco2e": Decimal("25000"),
                        "investee_scope3_tco2e": Decimal("120000"),
                        "data_quality_score": 2,
                    },
                    {
                        "investee_id": "INV-002",
                        "investee_name": "Steel Industries",
                        "sector": "materials",
                        "invested_amount_usd": Decimal("30000000"),
                        "evic_usd": Decimal("200000000"),
                        "investee_scope12_tco2e": Decimal("180000"),
                        "investee_scope3_tco2e": Decimal("450000"),
                        "data_quality_score": 1,
                    },
                    {
                        "investee_id": "INV-003",
                        "investee_name": "Clean Energy Fund",
                        "sector": "utilities",
                        "invested_amount_usd": Decimal("20000000"),
                        "evic_usd": Decimal("800000000"),
                        "investee_scope12_tco2e": Decimal("50000"),
                        "investee_scope3_tco2e": Decimal("30000"),
                        "data_quality_score": 3,
                    },
                ],
            },
            "corporate_bonds": {
                "investments": [
                    {
                        "investee_id": "BOND-001",
                        "investee_name": "HighCarbon Mining",
                        "sector": "mining",
                        "invested_amount_usd": Decimal("25000000"),
                        "evic_usd": Decimal("150000000"),
                        "investee_scope12_tco2e": Decimal("350000"),
                        "investee_scope3_tco2e": Decimal("200000"),
                        "data_quality_score": 2,
                    },
                ],
            },
        },
        "total_portfolio_value_usd": Decimal("125000000"),
    }


@pytest.fixture
def sample_retail_data() -> Dict[str, Any]:
    """Retail-specific data for last-mile and packaging emissions."""
    return {
        "last_mile": {
            "total_deliveries": 5000000,
            "average_distance_km": Decimal("12.5"),
            "emission_factor_kgco2e_per_km": Decimal("0.21"),
            "electric_vehicle_pct": Decimal("15"),
            "ev_emission_factor_kgco2e_per_km": Decimal("0.05"),
        },
        "packaging": {
            "total_units": 8000000,
            "materials": [
                {"type": "cardboard", "mass_kg_per_unit": Decimal("0.35"), "ef_kgco2e_per_kg": Decimal("0.94")},
                {"type": "plastic_film", "mass_kg_per_unit": Decimal("0.05"), "ef_kgco2e_per_kg": Decimal("3.10")},
                {"type": "paper_fill", "mass_kg_per_unit": Decimal("0.10"), "ef_kgco2e_per_kg": Decimal("0.84")},
            ],
        },
        "returns": {
            "return_rate_pct": Decimal("12"),
            "return_emissions_multiplier": Decimal("1.5"),
        },
    }


@pytest.fixture
def sample_cloud_data() -> Dict[str, Any]:
    """Cloud computing carbon data for technology companies."""
    return {
        "providers": {
            "aws": {
                "spend_usd": Decimal("2400000"),
                "kwh_consumed": Decimal("3500000"),
                "region": "us-east-1",
                "grid_factor_kgco2e_per_kwh": Decimal("0.379"),
                "pue": Decimal("1.135"),
                "renewable_pct": Decimal("85"),
            },
            "azure": {
                "spend_usd": Decimal("1200000"),
                "kwh_consumed": Decimal("1800000"),
                "region": "west-europe",
                "grid_factor_kgco2e_per_kwh": Decimal("0.276"),
                "pue": Decimal("1.125"),
                "renewable_pct": Decimal("100"),
            },
            "gcp": {
                "spend_usd": Decimal("600000"),
                "kwh_consumed": Decimal("900000"),
                "region": "europe-west1",
                "grid_factor_kgco2e_per_kwh": Decimal("0.052"),
                "pue": Decimal("1.100"),
                "renewable_pct": Decimal("100"),
            },
        },
        "embodied_carbon": {
            "servers": {"units": 200, "kgco2e_per_unit": Decimal("1200"), "amortization_years": 4},
            "networking": {"units": 50, "kgco2e_per_unit": Decimal("350"), "amortization_years": 5},
            "storage": {"units": 100, "kgco2e_per_unit": Decimal("150"), "amortization_years": 3},
        },
        "saas_use_phase": {
            "active_users": 50000,
            "avg_session_hours_per_month": Decimal("40"),
            "device_power_w": Decimal("65"),
            "grid_factor_kgco2e_per_kwh": Decimal("0.385"),
        },
    }


# =============================================================================
# 11. Assurance Evidence Fixtures
# =============================================================================


@pytest.fixture
def sample_assurance_evidence() -> Dict[str, Any]:
    """Assurance evidence package for Scope 3 verification."""
    return {
        "assurance_id": "ASR-2025-001",
        "org_id": "GRP-APEX-001",
        "reporting_year": 2025,
        "assurance_level": "limited",
        "standard": "ISAE_3410",
        "evidence_items": [
            {
                "item_id": "EV-001",
                "type": "calculation_provenance",
                "description": "SHA-256 hash chain for all 15 category calculations",
                "hash": "a" * 64,
                "status": "verified",
            },
            {
                "item_id": "EV-002",
                "type": "methodology_decision_log",
                "description": "Documented methodology choices for each category",
                "entries": 47,
                "status": "complete",
            },
            {
                "item_id": "EV-003",
                "type": "data_source_inventory",
                "description": "Complete inventory of all data sources",
                "sources": 85,
                "primary_pct": Decimal("35"),
                "secondary_pct": Decimal("65"),
                "status": "complete",
            },
            {
                "item_id": "EV-004",
                "type": "assumption_register",
                "description": "All assumptions documented with justification",
                "assumptions": 62,
                "status": "complete",
            },
            {
                "item_id": "EV-005",
                "type": "completeness_statement",
                "description": "Statement of boundary and coverage",
                "categories_covered": 15,
                "coverage_pct": Decimal("96"),
                "exclusions": ["Category 14 immaterial (<0.2% of total)"],
                "status": "complete",
            },
        ],
        "readiness_score": Decimal("82"),
        "findings": [
            {
                "finding_id": "FND-001",
                "severity": "observation",
                "category": 1,
                "description": "Spend-based factors could be improved with supplier data",
                "status": "open",
            },
            {
                "finding_id": "FND-002",
                "severity": "minor",
                "category": 11,
                "description": "Use-phase assumptions need sensitivity analysis",
                "status": "remediated",
            },
        ],
        "verifier_queries": [
            {
                "query_id": "VQ-001",
                "question": "Basis for Category 15 attribution factors?",
                "status": "answered",
                "response_date": "2025-03-10",
            },
            {
                "query_id": "VQ-002",
                "question": "Methodology for supplier-specific data in Cat 1?",
                "status": "pending",
            },
        ],
    }


# =============================================================================
# 12. Pack Configuration and Presets
# =============================================================================


@pytest.fixture
def sample_pack_config() -> Dict[str, Any]:
    """Default PackConfig for PACK-043."""
    return {
        "pack_id": "PACK-043",
        "pack_name": "Scope 3 Complete Pack",
        "version": "1.0.0",
        "category": "ghg-accounting",
        "environment": "test",
        "dependencies": ["PACK-042", "PACK-041"],
        "decimal_precision": 4,
        "provenance_enabled": True,
        "multi_tenant_enabled": True,
        "scope3": {
            "enabled_categories": list(range(1, 16)),
            "methodology_hierarchy": [
                "primary_data",
                "supplier_specific",
                "hybrid",
                "average_data",
                "spend_based",
            ],
            "default_ef_source": "ecoinvent_3.10",
            "gwp_source": "ar6",
        },
        "boundary": {
            "default_approach": "operational_control",
            "significance_threshold_pct": Decimal("5.0"),
            "multi_entity_enabled": True,
        },
        "sbti": {
            "pathway": "1.5C",
            "near_term_year": 2030,
            "long_term_year": 2050,
            "coverage_threshold_pct": Decimal("67"),
            "annual_reduction_rate": SBTI_15C_ANNUAL_RATE,
        },
        "lca": {
            "methodology": "ISO_14067",
            "system_boundary": "cradle_to_grave",
            "allocation_method": "mass",
        },
        "uncertainty": {
            "method": "analytical",
            "monte_carlo_iterations": 10000,
            "confidence_level": Decimal("0.95"),
            "seed": 43,
        },
        "reporting": {
            "output_formats": ["markdown", "html", "json", "pdf"],
            "frameworks": [
                "ghg_protocol_scope3",
                "iso_14064",
                "cdp",
                "sbti",
                "tcfd",
                "esrs_e1",
                "sec",
            ],
        },
    }


@pytest.fixture
def sample_manufacturing_config(sample_pack_config) -> Dict[str, Any]:
    """Manufacturing enterprise preset configuration."""
    config = dict(sample_pack_config)
    config["preset"] = "manufacturing_enterprise"
    config["scope3"]["priority_categories"] = [1, 2, 4, 5, 11, 12]
    config["lca"]["enabled"] = True
    config["lca"]["circular_economy"] = True
    return config


@pytest.fixture
def sample_financial_config(sample_pack_config) -> Dict[str, Any]:
    """Financial institution preset configuration."""
    config = dict(sample_pack_config)
    config["preset"] = "financial_institution"
    config["scope3"]["priority_categories"] = [15]
    config["pcaf"] = {
        "enabled": True,
        "asset_classes": ["listed_equity", "corporate_bonds", "project_finance"],
        "methodology": "PCAF_2022",
    }
    return config


# =============================================================================
# Scope 3 Screening Results (from PACK-042)
# =============================================================================


@pytest.fixture
def sample_scope3_screening() -> Dict[str, Any]:
    """Scope 3 screening results from PACK-042 dependency."""
    return {
        "screening_id": "SCR-2025-001",
        "org_id": "GRP-APEX-001",
        "reporting_year": 2025,
        "total_scope3_tco2e": Decimal("252500"),
        "by_category": {
            1: {"tco2e": Decimal("85000"), "pct": Decimal("33.66"), "material": True, "method": "spend_based"},
            2: {"tco2e": Decimal("12000"), "pct": Decimal("4.75"), "material": True, "method": "spend_based"},
            3: {"tco2e": Decimal("8000"), "pct": Decimal("3.17"), "material": True, "method": "average_data"},
            4: {"tco2e": Decimal("22000"), "pct": Decimal("8.71"), "material": True, "method": "distance_based"},
            5: {"tco2e": Decimal("5000"), "pct": Decimal("1.98"), "material": True, "method": "waste_type"},
            6: {"tco2e": Decimal("3000"), "pct": Decimal("1.19"), "material": False, "method": "spend_based"},
            7: {"tco2e": Decimal("4500"), "pct": Decimal("1.78"), "material": False, "method": "average_data"},
            8: {"tco2e": Decimal("2000"), "pct": Decimal("0.79"), "material": False, "method": "asset_based"},
            9: {"tco2e": Decimal("15000"), "pct": Decimal("5.94"), "material": True, "method": "distance_based"},
            10: {"tco2e": Decimal("8000"), "pct": Decimal("3.17"), "material": True, "method": "average_data"},
            11: {"tco2e": Decimal("35000"), "pct": Decimal("13.86"), "material": True, "method": "use_phase"},
            12: {"tco2e": Decimal("6000"), "pct": Decimal("2.38"), "material": True, "method": "waste_type"},
            13: {"tco2e": Decimal("1500"), "pct": Decimal("0.59"), "material": False, "method": "asset_based"},
            14: {"tco2e": Decimal("500"), "pct": Decimal("0.20"), "material": False, "method": "average_data"},
            15: {"tco2e": Decimal("45000"), "pct": Decimal("17.82"), "material": True, "method": "investment_based"},
        },
        "material_categories": [1, 2, 3, 4, 5, 9, 10, 11, 12, 15],
        "immaterial_categories": [6, 7, 8, 13, 14],
    }


# =============================================================================
# Composite / Convenience Fixtures
# =============================================================================


@pytest.fixture
def full_scope3_context(
    sample_entity_hierarchy,
    sample_maturity_assessment,
    sample_product_bom,
    sample_lca_results,
    sample_macc_interventions,
    sample_scenario_config,
    sample_sbti_targets,
    sample_supplier_programme,
    sample_climate_risks,
    sample_base_year_data,
    sample_pcaf_portfolio,
    sample_assurance_evidence,
    sample_pack_config,
    sample_scope3_screening,
) -> Dict[str, Any]:
    """Aggregate fixture combining all data for integration/e2e tests."""
    return {
        "entity_hierarchy": sample_entity_hierarchy,
        "maturity_assessment": sample_maturity_assessment,
        "product_bom": sample_product_bom,
        "lca_results": sample_lca_results,
        "macc_interventions": sample_macc_interventions,
        "scenario_config": sample_scenario_config,
        "sbti_targets": sample_sbti_targets,
        "supplier_programme": sample_supplier_programme,
        "climate_risks": sample_climate_risks,
        "base_year_data": sample_base_year_data,
        "pcaf_portfolio": sample_pcaf_portfolio,
        "assurance_evidence": sample_assurance_evidence,
        "pack_config": sample_pack_config,
        "scope3_screening": sample_scope3_screening,
    }
