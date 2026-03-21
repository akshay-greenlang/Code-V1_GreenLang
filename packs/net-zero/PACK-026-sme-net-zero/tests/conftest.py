# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-026 SME Net Zero Pack.
=============================================================

Provides pytest fixtures for all 8 engines, 6 workflows, sample SME data
builders, database mock setup, mock accounting API helpers, and common
test utilities tailored for micro/small/medium business scenarios.

Adds the pack root to sys.path so ``from engines.X import Y`` works
in every test module without requiring an installed package.

Fixtures cover:
    - Engine instantiation (8 engines)
    - Workflow instantiation (6 workflows)
    - Sample SME input builders (micro/small/medium, multi-sector)
    - Database session mocking (PostgreSQL + TimescaleDB)
    - Redis cache mocking
    - Mock accounting API helpers (Xero, QuickBooks, Sage)
    - SHA-256 provenance validation helpers
    - Decimal arithmetic assertion helpers
    - Performance timing context managers
    - SME tier validation helpers
    - Grant database mock fixtures

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-026 SME Net Zero Pack
Tests:   conftest.py (~500 lines)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure pack root is importable
# ---------------------------------------------------------------------------

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

_REPO_ROOT = _PACK_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

ENGINES_DIR = _PACK_ROOT / "engines"
WORKFLOWS_DIR = _PACK_ROOT / "workflows"
TEMPLATES_DIR = _PACK_ROOT / "templates"
INTEGRATIONS_DIR = _PACK_ROOT / "integrations"
CONFIG_DIR = _PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"


# ---------------------------------------------------------------------------
# Engine imports (lazy, with graceful fallback)
# ---------------------------------------------------------------------------

try:
    from engines.sme_baseline_engine import (
        SMEBaselineEngine,
        SMEBaselineInput,
        SMEBaselineResult,
        SMETier,
        BaselineMethod,
        IndustryAverageDB,
    )
    _HAS_BASELINE = True
except ImportError:
    _HAS_BASELINE = False

try:
    from engines.simplified_target_engine import (
        SimplifiedTargetEngine,
        SimplifiedTargetInput,
        SimplifiedTargetResult,
        TargetPathway,
        ScopeTarget,
    )
    _HAS_TARGET = True
except ImportError:
    _HAS_TARGET = False

try:
    from engines.quick_wins_engine import (
        QuickWinsEngine,
        QuickWinsInput,
        QuickWinsResult,
        QuickWinAction,
        QuickWinCategory,
        QUICK_WINS_DATABASE,
    )
    _HAS_QUICK_WINS = True
except ImportError:
    _HAS_QUICK_WINS = False

try:
    from engines.scope3_estimator_engine import (
        Scope3EstimatorEngine,
        Scope3EstimatorInput,
        Scope3EstimatorResult,
        SpendCategory,
        EmissionFactorSource,
    )
    _HAS_SCOPE3 = True
except ImportError:
    _HAS_SCOPE3 = False

try:
    from engines.action_prioritization_engine import (
        ActionPrioritizationEngine,
        ActionPrioritizationInput,
        ActionPrioritizationResult,
        PrioritizedAction,
        MACCDataPoint,
    )
    _HAS_ACTION_PRIO = True
except ImportError:
    _HAS_ACTION_PRIO = False

try:
    from engines.cost_benefit_engine import (
        CostBenefitEngine,
        CostBenefitInput,
        CostBenefitResult,
        FinancialMetrics,
        SensitivityResult,
    )
    _HAS_COST_BENEFIT = True
except ImportError:
    _HAS_COST_BENEFIT = False

try:
    from engines.grant_finder_engine import (
        GrantFinderEngine,
        GrantFinderInput,
        GrantFinderResult,
        GrantMatch,
        GrantDatabase,
        GrantRegion,
    )
    _HAS_GRANT_FINDER = True
except ImportError:
    _HAS_GRANT_FINDER = False

try:
    from engines.certification_readiness_engine import (
        CertificationReadinessEngine,
        CertificationReadinessInput,
        CertificationReadinessResult,
        CertificationType,
        ReadinessLevel,
        GapItem,
    )
    _HAS_CERT_READINESS = True
except ImportError:
    _HAS_CERT_READINESS = False


# ---------------------------------------------------------------------------
# Helper: Decimal assertion
# ---------------------------------------------------------------------------


def assert_decimal_close(
    actual: Decimal,
    expected: Decimal,
    tolerance: Decimal = Decimal("0.01"),
    msg: str = "",
) -> None:
    """Assert two Decimal values are within tolerance."""
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"Decimal mismatch{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}, diff={diff}, tol={tolerance}"
    )


def assert_provenance_hash(result: Any) -> None:
    """Assert that result has a non-empty SHA-256 provenance hash."""
    assert hasattr(result, "provenance_hash"), "Result missing provenance_hash"
    h = result.provenance_hash
    assert isinstance(h, str), "provenance_hash must be a string"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Hash must be hex"


def assert_processing_time(result: Any, max_ms: float = 30000.0) -> None:
    """Assert processing time is within acceptable range."""
    assert hasattr(result, "processing_time_ms"), "Result missing processing_time_ms"
    assert result.processing_time_ms >= 0, "Processing time must be non-negative"
    assert result.processing_time_ms < max_ms, (
        f"Processing time {result.processing_time_ms}ms exceeds {max_ms}ms"
    )


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@contextmanager
def timed_block(label: str = "", max_seconds: float = 10.0):
    """Context manager that asserts a block completes within max_seconds."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    assert elapsed < max_seconds, (
        f"Block '{label}' took {elapsed:.3f}s, exceeding {max_seconds}s"
    )


# ---------------------------------------------------------------------------
# SME Business Constants
# ---------------------------------------------------------------------------

SME_TIERS = ["micro", "small", "medium"]

SME_SECTORS = [
    "wholesale_retail", "accommodation_food", "professional_services", "manufacturing",
    "construction", "information_technology", "healthcare",
    "transport_logistics", "agriculture", "financial_services",
    "education", "arts_entertainment", "other_services",
]

SME_COUNTRIES = ["GB", "DE", "FR", "US", "NL", "IE", "AU", "CA"]

ACCOUNTING_PLATFORMS = ["xero", "quickbooks", "sage", "freshbooks", "wave"]


# ---------------------------------------------------------------------------
# Fixtures -- SME Business Profile Builders
# ---------------------------------------------------------------------------


@pytest.fixture
def micro_business_profile() -> Dict[str, Any]:
    """Build a micro-business profile (<10 employees, <2M EUR revenue)."""
    return {
        "entity_name": "Green Cafe Ltd",
        "company_size": "micro",
        "sector": "accommodation_food",
        "country": "GB",
        "headcount": 6,
        "revenue_usd": Decimal("350000"),
        "total_annual_spend_usd": Decimal("210000"),
        "floor_area_sqm": Decimal("120"),
        "vehicle_count": 1,
        "has_gas_heating": True,
        "has_refrigeration": True,
        "electricity_kwh_annual": Decimal("45000"),
        "gas_kwh_annual": Decimal("28000"),
        "accounting_platform": "xero",
    }


@pytest.fixture
def small_business_profile() -> Dict[str, Any]:
    """Build a small-business profile (10-49 employees, 2-10M EUR revenue)."""
    return {
        "entity_name": "TechSoft Solutions Ltd",
        "company_size": "small",
        "sector": "information_technology",
        "country": "DE",
        "headcount": 32,
        "revenue_usd": Decimal("4500000"),
        "total_annual_spend_usd": Decimal("3200000"),
        "floor_area_sqm": Decimal("450"),
        "vehicle_count": 5,
        "has_gas_heating": True,
        "has_refrigeration": False,
        "electricity_kwh_annual": Decimal("180000"),
        "gas_kwh_annual": Decimal("95000"),
        "accounting_platform": "quickbooks",
        "has_data_centers": True,
        "cloud_spend_usd": Decimal("120000"),
    }


@pytest.fixture
def medium_business_profile() -> Dict[str, Any]:
    """Build a medium-business profile (50-249 employees, 10-50M EUR revenue)."""
    return {
        "entity_name": "EuroManufact GmbH",
        "company_size": "medium",
        "sector": "manufacturing",
        "country": "DE",
        "headcount": 145,
        "revenue_usd": Decimal("28000000"),
        "total_annual_spend_usd": Decimal("21000000"),
        "floor_area_sqm": Decimal("4500"),
        "vehicle_count": 22,
        "has_gas_heating": True,
        "has_refrigeration": True,
        "has_industrial_processes": True,
        "electricity_kwh_annual": Decimal("850000"),
        "gas_kwh_annual": Decimal("420000"),
        "accounting_platform": "sage",
        "has_warehouse": True,
        "raw_material_spend_usd": Decimal("8500000"),
        "logistics_spend_usd": Decimal("1200000"),
    }


@pytest.fixture(params=["micro", "small", "medium"], ids=["micro", "small", "medium"])
def sme_tier(request) -> str:
    """Parameterized fixture yielding each SME tier."""
    return request.param


@pytest.fixture(params=SME_SECTORS, ids=SME_SECTORS)
def sme_sector(request) -> str:
    """Parameterized fixture yielding each SME sector."""
    return request.param


# ---------------------------------------------------------------------------
# Fixtures -- Mock Accounting API Responses
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_xero_response() -> Dict[str, Any]:
    """Build a mock Xero API response with GL code mappings."""
    return {
        "organisation": {"name": "Green Cafe Ltd", "country_code": "GB"},
        "accounts": [
            {"code": "200", "name": "Sales", "type": "REVENUE", "balance": 350000},
            {"code": "310", "name": "Electricity", "type": "EXPENSE", "balance": 8500},
            {"code": "311", "name": "Gas", "type": "EXPENSE", "balance": 3200},
            {"code": "312", "name": "Water", "type": "EXPENSE", "balance": 1100},
            {"code": "320", "name": "Motor Vehicle", "type": "EXPENSE", "balance": 4500},
            {"code": "400", "name": "Purchases", "type": "EXPENSE", "balance": 85000},
            {"code": "410", "name": "Office Supplies", "type": "EXPENSE", "balance": 3200},
            {"code": "500", "name": "Wages", "type": "EXPENSE", "balance": 95000},
        ],
        "journals": [
            {"date": "2025-01-31", "entries": [
                {"account_code": "310", "debit": 720, "credit": 0},
                {"account_code": "311", "debit": 280, "credit": 0},
            ]},
            {"date": "2025-02-28", "entries": [
                {"account_code": "310", "debit": 690, "credit": 0},
                {"account_code": "311", "debit": 250, "credit": 0},
            ]},
        ],
    }


@pytest.fixture
def mock_quickbooks_response() -> Dict[str, Any]:
    """Build a mock QuickBooks API response."""
    return {
        "company_name": "TechSoft Solutions Ltd",
        "fiscal_year_end": "12-31",
        "expense_accounts": [
            {"id": "60", "name": "Utilities", "amount": 24000, "currency": "EUR"},
            {"id": "61", "name": "Fuel & Travel", "amount": 18000, "currency": "EUR"},
            {"id": "62", "name": "Cloud Services", "amount": 120000, "currency": "EUR"},
            {"id": "63", "name": "Office Rent", "amount": 72000, "currency": "EUR"},
            {"id": "64", "name": "Supplies", "amount": 15000, "currency": "EUR"},
            {"id": "65", "name": "Professional Services", "amount": 45000, "currency": "EUR"},
        ],
        "monthly_totals": {f"2025-{m:02d}": Decimal(str(3200000 / 12)) for m in range(1, 13)},
    }


@pytest.fixture
def mock_sage_response() -> Dict[str, Any]:
    """Build a mock Sage API response."""
    return {
        "company": {"name": "EuroManufact GmbH", "country": "DE"},
        "nominal_codes": [
            {"code": "5000", "name": "Raw Materials", "total": 8500000},
            {"code": "5100", "name": "Energy", "total": 156000},
            {"code": "5200", "name": "Transport", "total": 1200000},
            {"code": "5300", "name": "Maintenance", "total": 340000},
            {"code": "6000", "name": "Payroll", "total": 5800000},
            {"code": "6100", "name": "Rent", "total": 480000},
        ],
    }


# ---------------------------------------------------------------------------
# Fixtures -- Mock Grant Database
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_grant_database() -> List[Dict[str, Any]]:
    """Build a mock grant database with UK/EU/US grants."""
    return [
        {
            "id": "UK-IETF-2026",
            "name": "Industrial Energy Transformation Fund",
            "region": "UK",
            "max_amount_eur": 250000,
            "min_employees": 1,
            "max_employees": 500,
            "sectors": ["manufacturing", "accommodation_food"],
            "deadline": "2026-09-30",
            "match_rate_pct": 40,
            "eligibility": ["energy_efficiency", "heat_decarbonization"],
        },
        {
            "id": "UK-GBF-2026",
            "name": "Green Business Fund",
            "region": "UK",
            "max_amount_eur": 5000,
            "min_employees": 1,
            "max_employees": 249,
            "sectors": ["all"],
            "deadline": "2026-12-31",
            "match_rate_pct": 100,
            "eligibility": ["energy_audit", "LED_lighting"],
        },
        {
            "id": "EU-LIFE-2026",
            "name": "LIFE Clean Energy Transition",
            "region": "EU",
            "max_amount_eur": 2000000,
            "min_employees": 10,
            "max_employees": 500,
            "sectors": ["all"],
            "deadline": "2026-06-30",
            "match_rate_pct": 60,
            "eligibility": ["renewable_energy", "energy_efficiency"],
        },
        {
            "id": "DE-BAFA-2026",
            "name": "BAFA Energy Consulting Subsidy",
            "region": "DE",
            "max_amount_eur": 6000,
            "min_employees": 1,
            "max_employees": 249,
            "sectors": ["all"],
            "deadline": "2026-12-31",
            "match_rate_pct": 80,
            "eligibility": ["energy_audit"],
        },
        {
            "id": "US-DOE-2026",
            "name": "DOE Small Business Energy Loans",
            "region": "US",
            "max_amount_eur": 500000,
            "min_employees": 1,
            "max_employees": 500,
            "sectors": ["manufacturing", "information_technology"],
            "deadline": "2026-12-31",
            "match_rate_pct": 30,
            "eligibility": ["energy_efficiency", "renewable_energy", "electrification"],
        },
        {
            "id": "UK-BEC-2026",
            "name": "Boiler Upgrade Scheme",
            "region": "UK",
            "max_amount_eur": 7500,
            "min_employees": 1,
            "max_employees": 500,
            "sectors": ["all"],
            "deadline": "2026-03-31",
            "match_rate_pct": 100,
            "eligibility": ["heat_pump", "heat_decarbonization"],
        },
    ]


# ---------------------------------------------------------------------------
# Fixtures -- Mock Database Session & Cache
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session():
    """Create a mock async database session (PostgreSQL + TimescaleDB)."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(
        fetchall=MagicMock(return_value=[]),
        fetchone=MagicMock(return_value=None),
        scalar=MagicMock(return_value=0),
    ))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.begin = MagicMock()
    return session


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock(return_value=True)
    redis_client.delete = AsyncMock(return_value=True)
    redis_client.exists = AsyncMock(return_value=False)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.hget = AsyncMock(return_value=None)
    redis_client.hset = AsyncMock(return_value=True)
    redis_client.pipeline = MagicMock(return_value=MagicMock(
        execute=AsyncMock(return_value=[]),
    ))
    return redis_client


# ---------------------------------------------------------------------------
# Fixtures -- Mock Accounting API Client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_xero_client(mock_xero_response):
    """Create a mock Xero API client."""
    client = MagicMock()
    client.get_organisation = AsyncMock(return_value=mock_xero_response["organisation"])
    client.get_accounts = AsyncMock(return_value=mock_xero_response["accounts"])
    client.get_journals = AsyncMock(return_value=mock_xero_response["journals"])
    client.get_trial_balance = AsyncMock(return_value=mock_xero_response["accounts"])
    client.is_connected = MagicMock(return_value=True)
    client.refresh_token = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_quickbooks_client(mock_quickbooks_response):
    """Create a mock QuickBooks API client."""
    client = MagicMock()
    client.get_company_info = AsyncMock(
        return_value={"company_name": mock_quickbooks_response["company_name"]}
    )
    client.get_expense_accounts = AsyncMock(
        return_value=mock_quickbooks_response["expense_accounts"]
    )
    client.get_monthly_totals = AsyncMock(
        return_value=mock_quickbooks_response["monthly_totals"]
    )
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_sage_client(mock_sage_response):
    """Create a mock Sage API client."""
    client = MagicMock()
    client.get_company = AsyncMock(return_value=mock_sage_response["company"])
    client.get_nominal_codes = AsyncMock(return_value=mock_sage_response["nominal_codes"])
    client.is_connected = MagicMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Fixtures -- Pack paths
# ---------------------------------------------------------------------------


@pytest.fixture
def pack_yaml_path() -> Path:
    """Return the path to pack.yaml."""
    return _PACK_ROOT / "pack.yaml"


@pytest.fixture
def pack_root() -> Path:
    """Return the pack root directory."""
    return _PACK_ROOT


@pytest.fixture
def presets_dir() -> Path:
    """Return the presets directory."""
    return PRESETS_DIR


# ---------------------------------------------------------------------------
# Fixtures -- Quick win actions sample data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_quick_wins() -> List[Dict[str, Any]]:
    """Build sample quick win actions for testing."""
    return [
        {
            "id": "QW-001",
            "name": "LED Lighting Upgrade",
            "category": "energy_efficiency",
            "sector_applicability": ["all"],
            "estimated_savings_pct": Decimal("15"),
            "capex_usd": Decimal("2500"),
            "annual_savings_usd": Decimal("1200"),
            "payback_months": 25,
            "annual_tco2e_reduction": Decimal("2.4"),
            "difficulty": "easy",
            "tier_applicability": ["micro", "small", "medium"],
        },
        {
            "id": "QW-002",
            "name": "Smart Thermostat Installation",
            "category": "heating",
            "sector_applicability": ["all"],
            "estimated_savings_pct": Decimal("10"),
            "capex_usd": Decimal("800"),
            "annual_savings_usd": Decimal("600"),
            "payback_months": 16,
            "annual_tco2e_reduction": Decimal("1.2"),
            "difficulty": "easy",
            "tier_applicability": ["micro", "small", "medium"],
        },
        {
            "id": "QW-003",
            "name": "Solar PV Installation",
            "category": "renewable_energy",
            "sector_applicability": ["all"],
            "estimated_savings_pct": Decimal("30"),
            "capex_usd": Decimal("25000"),
            "annual_savings_usd": Decimal("4500"),
            "payback_months": 67,
            "annual_tco2e_reduction": Decimal("8.5"),
            "difficulty": "medium",
            "tier_applicability": ["small", "medium"],
        },
        {
            "id": "QW-004",
            "name": "Heat Pump Replacement",
            "category": "heating",
            "sector_applicability": ["all"],
            "estimated_savings_pct": Decimal("50"),
            "capex_usd": Decimal("12000"),
            "annual_savings_usd": Decimal("2800"),
            "payback_months": 51,
            "annual_tco2e_reduction": Decimal("5.2"),
            "difficulty": "medium",
            "tier_applicability": ["small", "medium"],
        },
        {
            "id": "QW-005",
            "name": "EV Fleet Transition",
            "category": "transport",
            "sector_applicability": ["transport_logistics", "construction", "wholesale_retail"],
            "estimated_savings_pct": Decimal("60"),
            "capex_usd": Decimal("35000"),
            "annual_savings_usd": Decimal("5500"),
            "payback_months": 76,
            "annual_tco2e_reduction": Decimal("12.0"),
            "difficulty": "hard",
            "tier_applicability": ["medium"],
        },
    ]


# ---------------------------------------------------------------------------
# Fixtures -- Spend data for Scope 3 estimation
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_spend_data() -> Dict[str, Decimal]:
    """Build sample categorized spend data for Scope 3 estimation."""
    return {
        "purchased_goods_services": Decimal("850000"),
        "capital_goods": Decimal("120000"),
        "fuel_energy_activities": Decimal("35000"),
        "upstream_transportation": Decimal("45000"),
        "waste_generated": Decimal("8000"),
        "business_travel": Decimal("22000"),
        "employee_commuting": Decimal("18000"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Certification readiness data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_certification_status() -> Dict[str, Any]:
    """Build sample certification readiness data."""
    return {
        "iso_14001": {"has_certification": False, "readiness_pct": 35},
        "iso_50001": {"has_certification": False, "readiness_pct": 20},
        "sme_climate_hub": {"has_commitment": True, "readiness_pct": 60},
        "sbti_sme": {"has_target": False, "readiness_pct": 15},
        "carbon_trust_standard": {"has_certification": False, "readiness_pct": 25},
        "b_corp": {"has_certification": False, "readiness_pct": 10},
    }
