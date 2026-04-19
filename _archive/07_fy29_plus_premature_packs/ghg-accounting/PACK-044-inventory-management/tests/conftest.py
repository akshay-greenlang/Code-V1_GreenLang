# -*- coding: utf-8 -*-
"""
PACK-044 Inventory Management Pack - Shared Test Fixtures (conftest.py)
========================================================================

Provides pytest fixtures for the PACK-044 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Sample configuration objects (PackConfig, InventoryManagementConfig)
  - Inventory period data with milestones and transitions
  - Data collection campaign fixtures
  - Quality management QA/QC run data
  - Change request fixtures with affected sources
  - Review/approval workflow fixtures
  - Inventory version snapshots
  - Entity hierarchy for consolidation testing
  - Gap analysis assessment data
  - Benchmark comparison data
  - Mock database session

Fixture Categories:
  1.  Paths and dynamic module loading
  2.  Configuration fixtures
  3.  Inventory period fixtures
  4.  Data collection fixtures
  5.  Quality management fixtures
  6.  Change management fixtures
  7.  Review/approval fixtures
  8.  Versioning fixtures
  9.  Consolidation fixtures
  10. Gap analysis fixtures
  11. Documentation fixtures
  12. Benchmarking fixtures
  13. Mock database session

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-044 Inventory Management
Date:    March 2026
"""

import hashlib
import importlib
import importlib.util
import json
import random
import sys
from datetime import date, datetime, timedelta, timezone
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

ENGINE_FILES = {
    "inventory_period": "inventory_period_engine.py",
    "data_collection": "data_collection_engine.py",
    "quality_management": "quality_management_engine.py",
    "change_management": "change_management_engine.py",
    "review_approval": "review_approval_engine.py",
    "inventory_versioning": "inventory_versioning_engine.py",
    "consolidation_management": "consolidation_management_engine.py",
    "gap_analysis": "gap_analysis_engine.py",
    "documentation": "documentation_engine.py",
    "benchmarking": "benchmarking_engine.py",
}

ENGINE_CLASSES = {
    "inventory_period": "InventoryPeriodEngine",
    "data_collection": "DataCollectionEngine",
    "quality_management": "QualityManagementEngine",
    "change_management": "ChangeManagementEngine",
    "review_approval": "ReviewApprovalEngine",
    "inventory_versioning": "InventoryVersioningEngine",
    "consolidation_management": "ConsolidationManagementEngine",
    "gap_analysis": "GapAnalysisEngine",
    "documentation": "DocumentationEngine",
    "benchmarking": "BenchmarkingEngine",
}

WORKFLOW_FILES = {
    "annual_inventory_cycle": "annual_inventory_cycle_workflow.py",
    "data_collection_campaign": "data_collection_campaign_workflow.py",
    "quality_review": "quality_review_workflow.py",
    "change_assessment": "change_assessment_workflow.py",
    "inventory_finalization": "inventory_finalization_workflow.py",
    "consolidation": "consolidation_workflow.py",
    "improvement_planning": "improvement_planning_workflow.py",
    "full_management_pipeline": "full_management_pipeline_workflow.py",
}

WORKFLOW_CLASSES = {
    "annual_inventory_cycle": "AnnualInventoryCycleWorkflow",
    "data_collection_campaign": "DataCollectionCampaignWorkflow",
    "quality_review": "QualityReviewWorkflow",
    "change_assessment": "ChangeAssessmentWorkflow",
    "inventory_finalization": "InventoryFinalizationWorkflow",
    "consolidation": "ConsolidationWorkflow",
    "improvement_planning": "ImprovementPlanningWorkflow",
    "full_management_pipeline": "FullManagementPipelineWorkflow",
}

WORKFLOW_PHASE_COUNTS = {
    "annual_inventory_cycle": 8,
    "data_collection_campaign": 5,
    "quality_review": 4,
    "change_assessment": 4,
    "inventory_finalization": 5,
    "consolidation": 4,
    "improvement_planning": 4,
    "full_management_pipeline": 12,
}

TEMPLATE_FILES = {
    "inventory_status_dashboard": "inventory_status_dashboard.py",
    "data_collection_tracker": "data_collection_tracker.py",
    "quality_scorecard": "quality_scorecard.py",
    "change_log_report": "change_log_report.py",
    "review_summary_report": "review_summary_report.py",
    "version_comparison_report": "version_comparison_report.py",
    "consolidation_status_report": "consolidation_status_report.py",
    "gap_analysis_report": "gap_analysis_report.py",
    "documentation_index": "documentation_index.py",
    "benchmarking_report": "benchmarking_report.py",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "pack041_bridge": "pack041_bridge.py",
    "pack042_bridge": "pack042_bridge.py",
    "pack043_bridge": "pack043_bridge.py",
    "mrv_bridge": "mrv_bridge.py",
    "data_bridge": "data_bridge.py",
    "foundation_bridge": "foundation_bridge.py",
    "erp_connector": "erp_connector.py",
    "notification_bridge": "notification_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "InventoryManagementOrchestrator",
    "pack041_bridge": "Pack041Bridge",
    "pack042_bridge": "Pack042Bridge",
    "pack043_bridge": "Pack043Bridge",
    "mrv_bridge": "MRVBridge",
    "data_bridge": "DataBridge",
    "foundation_bridge": "FoundationBridge",
    "erp_connector": "ERPConnector",
    "notification_bridge": "NotificationBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

PRESET_NAMES = [
    "corporate_office",
    "manufacturing",
    "energy_utility",
    "transport_logistics",
    "food_agriculture",
    "real_estate",
    "healthcare",
    "sme_simplified",
]


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
            f"Ensure PACK-044 source files are present."
        )

    full_module_name = f"pack044_test.{subdir}.{module_name}"
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

_RNG = random.Random(44)


def _seeded_float(low: float, high: float) -> float:
    return _RNG.uniform(low, high)


def _seeded_int(low: int, high: int) -> int:
    return _RNG.randint(low, high)


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-044 root directory."""
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
# 2. Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """PackConfig-compatible dict with default values."""
    return {
        "pack_id": "PACK-044-inventory-management",
        "config_version": "1.0.0",
        "preset_name": "corporate_office",
        "pack": {
            "company_name": "Acme Global Industries",
            "sector_type": "OFFICE",
            "country": "DE",
            "reporting_year": 2026,
            "revenue_meur": 950.0,
            "employees_fte": 2500,
            "floor_area_m2": 138000.0,
            "period_management": {
                "auto_create_periods": True,
                "lock_after_approval": True,
                "max_open_periods": 3,
                "retention_years": 7,
            },
            "data_collection": {
                "auto_scheduling": True,
                "reminder_frequency_days": 7,
                "escalation_after_days": 21,
                "default_deadline_days": 30,
                "min_data_quality_score": 3.0,
                "collection_frequency": "QUARTERLY",
            },
            "quality_management": {
                "enabled": True,
                "auto_qaqc": True,
                "completeness_threshold_pct": 95.0,
                "consistency_threshold_pct": 20.0,
                "review_levels": 2,
            },
            "change_management": {
                "require_impact_assessment": True,
                "significance_threshold_pct": 5.0,
                "base_year_recalculation_threshold_pct": 5.0,
            },
            "review_approval": {
                "review_levels": ["PREPARER", "REVIEWER", "APPROVER"],
                "require_digital_signature": True,
            },
            "versioning": {
                "auto_version_on_changes": True,
                "max_draft_versions": 10,
                "allow_rollback": True,
            },
            "consolidation": {
                "approach": "OPERATIONAL_CONTROL",
                "equity_threshold_pct": 20.0,
                "eliminate_intragroup": True,
            },
            "gap_analysis": {
                "enabled": True,
                "methodology_tier_target": 2,
                "data_quality_target": 4.0,
            },
            "documentation": {
                "require_methodology_docs": True,
                "require_assumptions_register": True,
                "verification_readiness": True,
            },
            "benchmarking": {
                "enabled": True,
                "peer_group_size": 10,
                "sector_comparison": True,
            },
        },
    }


# =============================================================================
# 3. Inventory Period Fixtures
# =============================================================================

@pytest.fixture
def sample_period() -> Dict[str, Any]:
    """InventoryPeriod test data dictionary."""
    return {
        "organisation_id": "ORG-ACME-001",
        "period_name": "FY2025 GHG Inventory",
        "period_type": "calendar_year",
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 12, 31),
        "base_year": False,
        "base_year_reference": "PER-BASE-2019",
        "fiscal_year_start_month": 1,
        "created_by": "admin@acme.com",
        "metadata": {"framework": "GHG Protocol", "tier": "enterprise"},
    }


# =============================================================================
# 4. Data Collection Fixtures
# =============================================================================

@pytest.fixture
def sample_campaign() -> Dict[str, Any]:
    """CollectionCampaign test data dictionary."""
    return {
        "period_id": "per-2025-001",
        "organisation_id": "ORG-ACME-001",
        "campaign_name": "FY2025 Data Collection Campaign",
        "start_date": date(2026, 1, 15),
        "end_date": date(2026, 3, 31),
        "created_by": "sustainability@acme.com",
        "notes": "Annual GHG data collection covering Scope 1 and 2",
    }


# =============================================================================
# 5. Quality Management Fixtures
# =============================================================================

@pytest.fixture
def sample_quality_run() -> Dict[str, Any]:
    """QA/QC run data dictionary with check inputs."""
    return {
        "period_id": "per-2025-001",
        "organisation_id": "ORG-ACME-001",
        "check_inputs": {
            "COMP-001": True,   # All Scope 1 categories covered
            "COMP-002": True,   # Scope 2 dual reporting present
            "COMP-003": True,   # All facilities included
            "COMP-004": True,   # All greenhouse gases reported
            "COMP-005": True,   # Full reporting period coverage
            "CONS-001": True,   # Consistent calculation methodology
            "CONS-002": True,   # Consistent GWP values
            "CONS-003": True,   # YoY methodology consistency
            "CONS-004": True,   # EF source consistency
            "ACCU-001": True,   # No negative emission values
            "ACCU-002": True,   # YoY variance within bounds
            "ACCU-003": True,   # Activity data reasonableness
            "ACCU-004": True,   # Calculation cross-checks
            "ACCU-005": True,   # Unit conversion correctness
            "TRAN-001": True,   # Boundary description documented
            "TRAN-002": True,   # Emission factors sourced
            "TRAN-003": True,   # Assumptions documented
            "TRAN-004": True,   # Exclusions justified
            "TRAN-005": True,   # Methodology changes disclosed
        },
        "entity_id": "ENT-001",
        "facility_id": "FAC-001",
    }


# =============================================================================
# 6. Change Management Fixtures
# =============================================================================

@pytest.fixture
def sample_change_request() -> Dict[str, Any]:
    """ChangeRequest test data dictionary."""
    return {
        "title": "Updated grid emission factors for DE region",
        "description": (
            "IEA 2025 emission factors released with updated grid "
            "carbon intensity for Germany (0.365 kgCO2e/kWh, down from 0.385)."
        ),
        "category": "methodological",
        "requester_id": "user-env-mgr-001",
        "requester_name": "Jane Sustainability",
        "reporting_year": 2025,
        "total_inventory_tco2e": Decimal("55000"),
        "affected_sources": [
            {
                "source_id": "SRC-ELEC-DE-001",
                "source_name": "Grid electricity Frankfurt Plant",
                "scope": "scope2",
                "category": "purchased_electricity",
                "facility_id": "FAC-004",
                "old_value_tco2e": Decimal("4000"),
                "new_value_tco2e": Decimal("3800"),
                "delta_tco2e": Decimal("-200"),
            },
            {
                "source_id": "SRC-ELEC-DE-002",
                "source_name": "Grid electricity Berlin Office",
                "scope": "scope2",
                "category": "purchased_electricity",
                "facility_id": "FAC-005",
                "old_value_tco2e": Decimal("1500"),
                "new_value_tco2e": Decimal("1425"),
                "delta_tco2e": Decimal("-75"),
            },
        ],
        "affected_scopes": ["scope2"],
        "affected_facilities": ["FAC-004", "FAC-005"],
    }


# =============================================================================
# 7. Review/Approval Fixtures
# =============================================================================

@pytest.fixture
def sample_review_request() -> Dict[str, Any]:
    """ReviewRequest test data dictionary."""
    return {
        "period_id": "per-2025-001",
        "organisation_id": "ORG-ACME-001",
        "section_name": "Scope 1 - Stationary Combustion",
        "scope": "scope1",
        "preparer_id": "user-analyst-001",
        "preparer_name": "Data Analyst",
        "reviewer_id": "user-reviewer-001",
        "reviewer_name": "Senior Reviewer",
        "approver_id": "user-approver-001",
        "approver_name": "Sustainability Director",
        "data_summary": {
            "total_tco2e": Decimal("12500"),
            "source_count": 6,
            "facility_count": 3,
            "data_quality_score": Decimal("4.2"),
        },
        "checklist_items": [
            "All fuel consumption data sourced from invoices",
            "Emission factors from DEFRA 2025 database",
            "GWP values use IPCC AR6",
            "Uncertainty analysis completed",
        ],
    }


# =============================================================================
# 8. Versioning Fixtures
# =============================================================================

@pytest.fixture
def sample_version() -> Dict[str, Any]:
    """InventoryVersion test data dictionary."""
    return {
        "period_id": "per-2025-001",
        "organisation_id": "ORG-ACME-001",
        "version_name": "FY2025 Inventory v1.0",
        "description": "Initial draft of FY2025 GHG inventory",
        "created_by": "user-analyst-001",
        "inventory_data": {
            "scope1_total_tco2e": Decimal("22300"),
            "scope2_location_tco2e": Decimal("14700"),
            "scope2_market_tco2e": Decimal("9200"),
            "total_scope12_location_tco2e": Decimal("37000"),
            "total_scope12_market_tco2e": Decimal("31500"),
            "data_quality_score": Decimal("82.5"),
            "completeness_pct": Decimal("96.0"),
            "uncertainty_pct": Decimal("8.5"),
            "facility_count": 6,
            "entity_count": 2,
            "by_gas": {
                "CO2": Decimal("28500"),
                "CH4": Decimal("850"),
                "N2O": Decimal("450"),
                "HFCs": Decimal("1200"),
            },
        },
    }


# =============================================================================
# 9. Consolidation Fixtures
# =============================================================================

@pytest.fixture
def sample_entity_hierarchy() -> Dict[str, Any]:
    """Entity hierarchy for consolidation testing."""
    return {
        "group_id": "GRP-ACME-001",
        "group_name": "Acme Global Industries Group",
        "consolidation_approach": "operational_control",
        "reporting_year": 2025,
        "entities": [
            {
                "entity_id": "ENT-001",
                "entity_name": "Acme Manufacturing US",
                "parent_entity_id": None,
                "relationship_type": "subsidiary",
                "equity_share_pct": Decimal("100"),
                "has_operational_control": True,
                "has_financial_control": True,
                "scope1_tco2e": Decimal("12500"),
                "scope2_location_tco2e": Decimal("8200"),
                "scope2_market_tco2e": Decimal("5100"),
                "status": "submitted",
            },
            {
                "entity_id": "ENT-002",
                "entity_name": "Acme Europe GmbH",
                "parent_entity_id": None,
                "relationship_type": "subsidiary",
                "equity_share_pct": Decimal("100"),
                "has_operational_control": True,
                "has_financial_control": True,
                "scope1_tco2e": Decimal("9800"),
                "scope2_location_tco2e": Decimal("6500"),
                "scope2_market_tco2e": Decimal("4100"),
                "status": "submitted",
            },
            {
                "entity_id": "ENT-003",
                "entity_name": "Acme-Nippon JV",
                "parent_entity_id": None,
                "relationship_type": "joint_venture",
                "equity_share_pct": Decimal("50"),
                "has_operational_control": False,
                "has_financial_control": False,
                "scope1_tco2e": Decimal("7200"),
                "scope2_location_tco2e": Decimal("5100"),
                "scope2_market_tco2e": Decimal("3600"),
                "status": "submitted",
            },
        ],
        "intragroup_transfers": [
            {
                "from_entity_id": "ENT-001",
                "to_entity_id": "ENT-002",
                "transfer_tco2e": Decimal("150"),
                "description": "Intercompany product transfer US->DE",
            },
        ],
    }


# =============================================================================
# 10. Gap Analysis Fixtures
# =============================================================================

@pytest.fixture
def sample_gap_assessment() -> Dict[str, Any]:
    """Gap analysis assessment data."""
    return {
        "period_id": "per-2025-001",
        "organisation_id": "ORG-ACME-001",
        "total_emissions_tco2e": Decimal("55000"),
        "categories": [
            {
                "scope": "scope1",
                "category": "stationary_combustion",
                "emissions_tco2e": Decimal("12800"),
                "emissions_pct": Decimal("23.3"),
                "methodology_tier": 2,
                "data_quality_score": Decimal("4.0"),
                "has_data": True,
                "data_source": "Fuel invoices + DEFRA EFs",
            },
            {
                "scope": "scope1",
                "category": "mobile_combustion",
                "emissions_tco2e": Decimal("2500"),
                "emissions_pct": Decimal("4.5"),
                "methodology_tier": 2,
                "data_quality_score": Decimal("3.5"),
                "has_data": True,
                "data_source": "Fleet fuel cards",
            },
            {
                "scope": "scope1",
                "category": "process_emissions",
                "emissions_tco2e": Decimal("4200"),
                "emissions_pct": Decimal("7.6"),
                "methodology_tier": 1,
                "data_quality_score": Decimal("2.5"),
                "has_data": True,
                "data_source": "Industry average EFs",
            },
            {
                "scope": "scope1",
                "category": "fugitive_emissions",
                "emissions_tco2e": Decimal("350"),
                "emissions_pct": Decimal("0.6"),
                "methodology_tier": 1,
                "data_quality_score": Decimal("2.0"),
                "has_data": True,
                "data_source": "Estimated from equipment type",
            },
            {
                "scope": "scope2",
                "category": "purchased_electricity",
                "emissions_tco2e": Decimal("14700"),
                "emissions_pct": Decimal("26.7"),
                "methodology_tier": 2,
                "data_quality_score": Decimal("4.5"),
                "has_data": True,
                "data_source": "Utility bills + IEA grid factors",
            },
            {
                "scope": "scope3",
                "category": "purchased_goods",
                "emissions_tco2e": Decimal("8000"),
                "emissions_pct": Decimal("14.5"),
                "methodology_tier": 1,
                "data_quality_score": Decimal("2.0"),
                "has_data": True,
                "data_source": "Spend-based screening",
            },
            {
                "scope": "scope3",
                "category": "employee_commuting",
                "emissions_tco2e": Decimal("0"),
                "emissions_pct": Decimal("0"),
                "methodology_tier": 0,
                "data_quality_score": Decimal("0"),
                "has_data": False,
                "data_source": None,
            },
        ],
    }


# =============================================================================
# 11. Benchmarking Fixtures
# =============================================================================

@pytest.fixture
def sample_benchmark_data() -> Dict[str, Any]:
    """Benchmark comparison data."""
    return {
        "organisation_id": "ORG-ACME-001",
        "reporting_year": 2025,
        "sector": "manufacturing",
        "entity_data": {
            "total_scope12_tco2e": Decimal("37000"),
            "revenue_meur": Decimal("950"),
            "employees_fte": 2500,
            "floor_area_m2": Decimal("138000"),
            "production_tonnes": Decimal("50000"),
        },
        "peer_data": [
            {
                "peer_id": "PEER-001",
                "total_scope12_tco2e": Decimal("42000"),
                "revenue_meur": Decimal("1100"),
                "employees_fte": 3000,
            },
            {
                "peer_id": "PEER-002",
                "total_scope12_tco2e": Decimal("28000"),
                "revenue_meur": Decimal("700"),
                "employees_fte": 1800,
            },
            {
                "peer_id": "PEER-003",
                "total_scope12_tco2e": Decimal("55000"),
                "revenue_meur": Decimal("1500"),
                "employees_fte": 4000,
            },
            {
                "peer_id": "PEER-004",
                "total_scope12_tco2e": Decimal("31000"),
                "revenue_meur": Decimal("800"),
                "employees_fte": 2200,
            },
            {
                "peer_id": "PEER-005",
                "total_scope12_tco2e": Decimal("48000"),
                "revenue_meur": Decimal("1300"),
                "employees_fte": 3500,
            },
        ],
        "sector_averages": {
            "intensity_tco2e_per_meur": Decimal("38.5"),
            "intensity_tco2e_per_fte": Decimal("14.8"),
        },
        "historical": [
            {"year": 2023, "total_scope12_tco2e": Decimal("39500"), "revenue_meur": Decimal("900")},
            {"year": 2024, "total_scope12_tco2e": Decimal("38100"), "revenue_meur": Decimal("920")},
            {"year": 2025, "total_scope12_tco2e": Decimal("37000"), "revenue_meur": Decimal("950")},
        ],
    }


# =============================================================================
# 12. Mock Database Session
# =============================================================================

@pytest.fixture
def mock_db_session():
    """Return None -- engines operate in-memory; no DB needed for unit tests."""
    return None


# =============================================================================
# Composite / Convenience Fixtures
# =============================================================================

@pytest.fixture
def full_inventory_context(
    sample_config,
    sample_period,
    sample_campaign,
    sample_quality_run,
    sample_change_request,
    sample_review_request,
    sample_version,
    sample_entity_hierarchy,
    sample_gap_assessment,
    sample_benchmark_data,
) -> Dict[str, Any]:
    """Aggregate fixture combining all data for integration/e2e tests."""
    return {
        "config": sample_config,
        "period": sample_period,
        "campaign": sample_campaign,
        "quality_run": sample_quality_run,
        "change_request": sample_change_request,
        "review_request": sample_review_request,
        "version": sample_version,
        "entity_hierarchy": sample_entity_hierarchy,
        "gap_assessment": sample_gap_assessment,
        "benchmark_data": sample_benchmark_data,
    }
