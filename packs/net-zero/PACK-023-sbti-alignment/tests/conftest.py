# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-023 SBTi Alignment Pack.

Adds the pack root to sys.path so that ``from engines.X import Y`` works
in every test module without requiring an installed package. Provides
shared pytest fixtures for engine, workflow, template, and integration
testing.
"""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


# ---------------------------------------------------------------------------
# Common test helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Return current UTC datetime for test timestamps."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Emissions inventory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_emissions_inventory() -> Dict[str, Any]:
    """Provide a sample emissions inventory for engine tests.

    Represents a mid-size manufacturing company with Scope 1, 2, and 3
    emissions typical for SBTi target-setting.
    """
    return {
        "organization_id": "test-org-001",
        "organization_name": "Test Manufacturing Corp",
        "base_year": 2022,
        "current_year": 2024,
        "sector": "manufacturing",
        "scope1_tco2e": Decimal("15000.00"),
        "scope2_location_tco2e": Decimal("8000.00"),
        "scope2_market_tco2e": Decimal("7500.00"),
        "scope3_total_tco2e": Decimal("45000.00"),
        "scope3_categories": {
            "cat1_purchased_goods": Decimal("18000.00"),
            "cat2_capital_goods": Decimal("3000.00"),
            "cat3_fuel_energy": Decimal("2500.00"),
            "cat4_upstream_transport": Decimal("4000.00"),
            "cat5_waste": Decimal("1500.00"),
            "cat6_business_travel": Decimal("2000.00"),
            "cat7_commuting": Decimal("1800.00"),
            "cat8_upstream_leased": Decimal("500.00"),
            "cat9_downstream_transport": Decimal("3500.00"),
            "cat10_processing": Decimal("2200.00"),
            "cat11_use_of_sold": Decimal("4000.00"),
            "cat12_end_of_life": Decimal("1000.00"),
            "cat13_downstream_leased": Decimal("300.00"),
            "cat14_franchises": Decimal("0.00"),
            "cat15_investments": Decimal("700.00"),
        },
        "total_tco2e": Decimal("67500.00"),
        "revenue_musd": Decimal("500.00"),
        "employees": 2500,
    }


@pytest.fixture
def sample_target_config() -> Dict[str, Any]:
    """Provide a sample SBTi target configuration."""
    return {
        "ambition_level": "1.5C",
        "pathway_method": "ACA",
        "near_term_year": 2030,
        "long_term_year": 2050,
        "scope12_coverage_pct": Decimal("95.0"),
        "scope3_coverage_pct": Decimal("67.0"),
        "flag_relevant": False,
        "fi_relevant": False,
        "consolidation_approach": "operational_control",
    }


@pytest.fixture
def sample_flag_config() -> Dict[str, Any]:
    """Provide a sample FLAG assessment configuration."""
    return {
        "flag_threshold_pct": Decimal("20.0"),
        "linear_reduction_rate": Decimal("3.03"),
        "commodities": ["cattle", "soy", "palm_oil"],
        "no_deforestation_target_year": 2025,
    }


@pytest.fixture
def sample_fi_portfolio_config() -> Dict[str, Any]:
    """Provide a sample FI portfolio configuration."""
    return {
        "asset_classes": [
            "corporate_loans",
            "listed_equity",
            "corporate_bonds",
        ],
        "pcaf_target_quality": 3,
        "coverage_target_pct": Decimal("67.0"),
        "engagement_target_pct": Decimal("50.0"),
    }


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def target_setting_engine_config() -> Dict[str, Any]:
    """Configuration for TargetSettingEngine tests."""
    return {
        "enable_flag_check": True,
        "enable_fi_check": False,
        "default_ambition": "1.5C",
        "near_term_min_years": 5,
        "near_term_max_years": 10,
    }


@pytest.fixture
def criteria_validation_config() -> Dict[str, Any]:
    """Configuration for CriteriaValidationEngine tests."""
    return {
        "enable_near_term": True,
        "enable_net_zero": True,
        "scope12_boundary_pct": Decimal("95.0"),
        "scope3_trigger_pct": Decimal("40.0"),
    }


@pytest.fixture
def temperature_rating_config() -> Dict[str, Any]:
    """Configuration for TemperatureRatingEngine tests."""
    return {
        "default_score": Decimal("3.20"),
        "aggregation_methods": ["WATS", "TETS", "MOTS"],
        "scope_filter": "S1S2S3",
    }


# ---------------------------------------------------------------------------
# Workflow fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow_base_config() -> Dict[str, Any]:
    """Base configuration shared across workflow tests."""
    return {
        "organization_id": "test-org-001",
        "base_year": 2022,
        "current_year": 2024,
        "sector": "manufacturing",
        "enable_provenance": True,
        "timeout_seconds": 300,
    }


# ---------------------------------------------------------------------------
# Template fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def template_config() -> Dict[str, Any]:
    """Configuration for template rendering tests."""
    return {
        "output_format": "markdown",
        "include_provenance": True,
        "include_methodology": True,
        "include_recommendations": True,
        "language": "en",
    }


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_config() -> Dict[str, Any]:
    """Configuration for integration bridge tests."""
    return {
        "enable_mrv": True,
        "enable_ghg_app": True,
        "enable_pack021": False,
        "enable_pack022": False,
        "enable_decarb": True,
        "enable_data": True,
        "retry_max_attempts": 3,
        "retry_backoff_factor": 2.0,
    }
