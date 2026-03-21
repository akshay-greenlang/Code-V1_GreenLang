# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-027 Enterprise Net Zero Pack.
=================================================================

Provides pytest fixtures for all 8 engines, 8 workflows, sample enterprise
data builders, database mock setup, mock ERP API helpers (SAP, Oracle,
Workday), and common test utilities tailored for large enterprise scenarios.

Adds the pack root to sys.path so ``from engines.X import Y`` works
in every test module without requiring an installed package.

Fixtures cover:
    - Engine instantiation (8 engines)
    - Workflow instantiation (8 workflows)
    - Sample enterprise input builders (manufacturing, financial, technology)
    - Database session mocking (PostgreSQL + TimescaleDB)
    - Redis cache mocking
    - Mock ERP API helpers (SAP S/4HANA, Oracle ERP Cloud, Workday HCM)
    - SHA-256 provenance validation helpers
    - Decimal arithmetic assertion helpers
    - Performance timing context managers
    - Multi-entity consolidation helpers
    - Carbon pricing fixtures
    - Supply chain mapping fixtures
    - SBTi criteria validation helpers
    - Scenario modeling fixtures (Monte Carlo)

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-027 Enterprise Net Zero Pack
Tests:   conftest.py (~700 lines)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    from engines.enterprise_baseline_engine import (
        EnterpriseBaselineEngine,
        EnterpriseBaselineInput,
        EnterpriseBaselineResult,
        DataQualityLevel,
        DataQualityMatrix,
        MaterialityAssessment,
    )
    _HAS_BASELINE = True
except ImportError:
    _HAS_BASELINE = False

try:
    from engines.sbti_target_engine import (
        SBTiTargetEngine,
        SBTiTargetInput,
        SBTiTargetResult,
        TargetPathwayType,
        CriterionValidation,
    )
    # Aliases for backward compatibility
    TargetPathway = TargetPathwayType
    CriteriaValidation = CriterionValidation
    SBTiPathwayType = TargetPathwayType
    _HAS_SBTI = True
except ImportError:
    _HAS_SBTI = False

try:
    from engines.scenario_modeling_engine import (
        ScenarioModelingEngine,
        ScenarioModelingInput,
        ScenarioModelingResult,
        ScenarioType,
        ParameterDistribution,
        SensitivityDriver,
    )
    # Aliases for backward compatibility
    ScenarioConfig = ScenarioType
    ScenarioInput = ScenarioModelingInput
    ScenarioResult = ScenarioModelingResult
    MonteCarloRun = None
    SensitivityAnalysis = SensitivityDriver
    _HAS_SCENARIO = True
except ImportError:
    _HAS_SCENARIO = False

try:
    from engines.carbon_pricing_engine import (
        CarbonPricingEngine,
        CarbonPricingInput,
        CarbonPricingResult,
        CarbonPricingApproach,
        InvestmentAppraisal,
        CBAMExposure,
    )
    CarbonPricingConfig = CarbonPricingApproach  # alias
    _HAS_CARBON_PRICING = True
except ImportError:
    _HAS_CARBON_PRICING = False

try:
    from engines.scope4_avoided_emissions_engine import (
        Scope4AvoidedEmissionsEngine,
        Scope4Input,
        Scope4Result,
        AvoidedEmissionCategory,
        BaselineType,
        ProductAvoidedEmissionEntry,
    )
    # Aliases for backward compatibility
    AvoidedEmissionsConfig = AvoidedEmissionCategory
    AvoidedEmissionsInput = Scope4Input
    AvoidedEmissionsResult = Scope4Result
    BaselineScenario = BaselineType
    AttributionShare = None
    _HAS_SCOPE4 = True
except ImportError:
    _HAS_SCOPE4 = False

try:
    from engines.supply_chain_mapping_engine import (
        SupplyChainMappingEngine,
        SupplyChainMappingInput,
        SupplyChainMappingResult,
        SupplierScorecard,
        EngagementProgramStatus,
        SupplierEntry,
    )
    # Aliases for backward compatibility
    SupplyChainConfig = None
    SupplyChainInput = SupplyChainMappingInput
    SupplyChainResult = SupplyChainMappingResult
    EngagementProgram = EngagementProgramStatus
    _HAS_SUPPLY_CHAIN = True
except ImportError:
    _HAS_SUPPLY_CHAIN = False

try:
    from engines.multi_entity_consolidation_engine import (
        MultiEntityConsolidationEngine,
        ConsolidationInput,
        ConsolidationResult,
        EntityEmissions,
        BaseYearRecalculation,
        ConsolidationApproach,
    )
    # Aliases for backward compatibility
    ConsolidationConfig = ConsolidationApproach
    EntityHierarchy = EntityEmissions
    _HAS_CONSOLIDATION = True
except ImportError:
    _HAS_CONSOLIDATION = False

try:
    from engines.financial_integration_engine import (
        FinancialIntegrationEngine,
        FinancialIntegrationInput,
        FinancialIntegrationResult,
        CarbonPnLAllocation,
        CarbonBalanceSheet,
    )
    # Aliases for backward compatibility
    FinancialIntegrationConfig = None
    CarbonPnL = CarbonPnLAllocation
    _HAS_FINANCIAL = True
except ImportError:
    _HAS_FINANCIAL = False


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


def assert_decimal_positive(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is positive."""
    assert value > Decimal("0"), (
        f"Expected positive Decimal{' (' + msg + ')' if msg else ''}, got {value}"
    )


def assert_percentage_range(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is between 0 and 100."""
    assert Decimal("0") <= value <= Decimal("100"), (
        f"Percentage out of range{' (' + msg + ')' if msg else ''}: {value}"
    )


def assert_provenance_hash(result: Any) -> None:
    """Assert that result has a non-empty SHA-256 provenance hash."""
    assert hasattr(result, "provenance_hash"), "Result missing provenance_hash"
    h = result.provenance_hash
    assert isinstance(h, str), "provenance_hash must be a string"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Hash must be hex"


def assert_processing_time(result: Any, max_ms: float = 60000.0) -> None:
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
def timed_block(label: str = "", max_seconds: float = 30.0):
    """Context manager that asserts a block completes within max_seconds."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    assert elapsed < max_seconds, (
        f"Block '{label}' took {elapsed:.3f}s, exceeding {max_seconds}s"
    )


# ---------------------------------------------------------------------------
# Enterprise Business Constants
# ---------------------------------------------------------------------------

ENTERPRISE_SECTORS = [
    "manufacturing", "energy_utilities", "financial_services",
    "technology", "consumer_goods", "transport_logistics",
    "real_estate", "healthcare_pharma",
]

CONSOLIDATION_APPROACHES = [
    "financial_control", "operational_control", "equity_share",
]

SBTI_PATHWAYS = ["ACA_15C", "ACA_WB2C", "SDA", "FLAG"]

SBTI_NEAR_TERM_CRITERIA = [f"C{i}" for i in range(1, 29)]  # C1-C28
SBTI_NET_ZERO_CRITERIA = [f"NZ-C{i}" for i in range(1, 15)]  # NZ-C1 to NZ-C14

SDA_SECTORS = [
    "power_generation", "cement", "iron_steel", "aluminium",
    "pulp_paper", "chemicals", "aviation", "maritime_shipping",
    "road_transport", "commercial_buildings", "residential_buildings",
    "food_beverage",
]

SCOPE3_CATEGORIES = list(range(1, 16))  # Cat 1-15

ERP_PLATFORMS = ["sap_s4hana", "oracle_erp_cloud", "workday_hcm"]

DATA_QUALITY_LEVELS = [1, 2, 3, 4, 5]

ENTERPRISE_COUNTRIES = [
    "US", "GB", "DE", "FR", "JP", "CN", "IN", "BR", "AU", "CA",
    "NL", "CH", "SE", "KR", "SG", "AE", "ZA", "MX", "IT", "ES",
]

REGULATORY_FRAMEWORKS = [
    "ghg_protocol", "sbti_corporate", "sbti_net_zero", "cdp_climate",
    "tcfd", "sec_climate_rule", "csrd_esrs_e1", "california_sb253",
    "iso_14064_1", "issb_s2",
]


# ---------------------------------------------------------------------------
# Fixtures -- Enterprise Business Profile Builders
# ---------------------------------------------------------------------------


@pytest.fixture
def manufacturing_enterprise_profile() -> Dict[str, Any]:
    """Build a large manufacturing enterprise profile (>250 employees)."""
    return {
        "entity_name": "GlobalManufact Corp",
        "sector": "manufacturing",
        "country": "DE",
        "headcount": 12500,
        "revenue_usd": Decimal("2800000000"),
        "total_annual_spend_usd": Decimal("1950000000"),
        "facility_count": 45,
        "subsidiary_count": 28,
        "countries_of_operation": 18,
        "scope1_sources": ["stationary_combustion", "mobile_combustion",
                           "process_emissions", "fugitive_emissions", "refrigerants"],
        "scope2_method": "dual_reporting",
        "scope3_categories": list(range(1, 16)),
        "sbti_pathway": "mixed",
        "erp_platform": "sap_s4hana",
        "has_ets_obligations": True,
        "has_cbam_exposure": True,
        "vehicle_count": 320,
        "electricity_kwh_annual": Decimal("185000000"),
        "gas_kwh_annual": Decimal("92000000"),
        "raw_material_spend_usd": Decimal("780000000"),
        "data_quality_target": Decimal("3"),
    }


@pytest.fixture
def financial_services_profile() -> Dict[str, Any]:
    """Build a financial services enterprise profile."""
    return {
        "entity_name": "GlobalBank Holdings plc",
        "sector": "financial_services",
        "country": "GB",
        "headcount": 85000,
        "revenue_usd": Decimal("45000000000"),
        "total_annual_spend_usd": Decimal("28000000000"),
        "facility_count": 1200,
        "subsidiary_count": 180,
        "countries_of_operation": 62,
        "scope1_sources": ["stationary_combustion", "refrigerants"],
        "scope2_method": "dual_reporting",
        "scope3_categories": [1, 6, 7, 15],
        "scope3_cat15_dominant": True,
        "sbti_pathway": "ACA_15C",
        "pcaf_enabled": True,
        "finz_enabled": True,
        "erp_platform": "oracle_erp_cloud",
        "aum_usd": Decimal("2500000000000"),
        "lending_portfolio_usd": Decimal("850000000000"),
        "electricity_kwh_annual": Decimal("420000000"),
        "data_quality_target": Decimal("3"),
    }


@pytest.fixture
def technology_enterprise_profile() -> Dict[str, Any]:
    """Build a technology enterprise profile."""
    return {
        "entity_name": "TechGlobal Inc",
        "sector": "technology",
        "country": "US",
        "headcount": 42000,
        "revenue_usd": Decimal("18000000000"),
        "total_annual_spend_usd": Decimal("12000000000"),
        "facility_count": 85,
        "subsidiary_count": 35,
        "countries_of_operation": 28,
        "scope1_sources": ["stationary_combustion", "refrigerants"],
        "scope2_method": "dual_reporting",
        "scope2_pue_tracking": True,
        "scope3_categories": [1, 2, 3, 11, 12],
        "sbti_pathway": "ACA_15C",
        "re100_commitment": True,
        "erp_platform": "workday_hcm",
        "data_center_count": 12,
        "data_center_pue_avg": Decimal("1.18"),
        "electricity_kwh_annual": Decimal("3200000000"),
        "scope4_enabled": True,
        "data_quality_target": Decimal("3"),
    }


@pytest.fixture
def consumer_goods_profile() -> Dict[str, Any]:
    """Build a consumer goods enterprise profile."""
    return {
        "entity_name": "NaturalGoods International",
        "sector": "consumer_goods",
        "country": "NL",
        "headcount": 95000,
        "revenue_usd": Decimal("52000000000"),
        "total_annual_spend_usd": Decimal("38000000000"),
        "facility_count": 320,
        "subsidiary_count": 120,
        "countries_of_operation": 78,
        "scope3_categories": [1, 4, 9, 11, 12],
        "flag_enabled": True,
        "flag_commodities": ["palm_oil", "soy", "cocoa", "coffee", "cotton"],
        "sbti_pathway": "mixed",
        "supplier_count": 85000,
        "tier_depth": 5,
        "data_quality_target": Decimal("3"),
    }


@pytest.fixture(params=ENTERPRISE_SECTORS, ids=ENTERPRISE_SECTORS)
def enterprise_sector(request) -> str:
    """Parameterized fixture yielding each enterprise sector."""
    return request.param


@pytest.fixture(params=CONSOLIDATION_APPROACHES, ids=CONSOLIDATION_APPROACHES)
def consolidation_approach(request) -> str:
    """Parameterized fixture yielding each consolidation approach."""
    return request.param


@pytest.fixture(params=SBTI_PATHWAYS, ids=SBTI_PATHWAYS)
def sbti_pathway(request) -> str:
    """Parameterized fixture yielding each SBTi pathway type."""
    return request.param


# ---------------------------------------------------------------------------
# Fixtures -- Multi-Entity Hierarchy
# ---------------------------------------------------------------------------


@pytest.fixture
def entity_hierarchy() -> Dict[str, Any]:
    """Build a sample multi-entity hierarchy for consolidation testing."""
    return {
        "parent": {
            "id": "CORP-001",
            "name": "GlobalManufact Corp",
            "country": "DE",
            "ownership_pct": Decimal("100"),
            "control_type": "financial_control",
        },
        "entities": [
            {
                "id": "SUB-001", "name": "GM Europe GmbH", "country": "DE",
                "ownership_pct": Decimal("100"), "control_type": "financial_control",
                "emissions_tco2e": Decimal("45000"),
            },
            {
                "id": "SUB-002", "name": "GM Americas Inc", "country": "US",
                "ownership_pct": Decimal("100"), "control_type": "financial_control",
                "emissions_tco2e": Decimal("68000"),
            },
            {
                "id": "SUB-003", "name": "GM Asia Pacific Pte", "country": "SG",
                "ownership_pct": Decimal("100"), "control_type": "operational_control",
                "emissions_tco2e": Decimal("32000"),
            },
            {
                "id": "JV-001", "name": "GM-Partner JV", "country": "CN",
                "ownership_pct": Decimal("51"), "control_type": "joint_venture",
                "emissions_tco2e": Decimal("55000"),
            },
            {
                "id": "ASSOC-001", "name": "TechPartner Associates", "country": "JP",
                "ownership_pct": Decimal("33"), "control_type": "associate",
                "emissions_tco2e": Decimal("28000"),
            },
            {
                "id": "SUB-004", "name": "GM Logistics BV", "country": "NL",
                "ownership_pct": Decimal("80"), "control_type": "financial_control",
                "emissions_tco2e": Decimal("18000"),
            },
        ],
        "intercompany_transactions": [
            {
                "from_entity": "SUB-001", "to_entity": "SUB-002",
                "type": "energy_supply", "emissions_tco2e": Decimal("2500"),
            },
            {
                "from_entity": "SUB-002", "to_entity": "SUB-003",
                "type": "shared_services", "emissions_tco2e": Decimal("1200"),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Fixtures -- SAP / Oracle / Workday Mock Responses
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sap_response() -> Dict[str, Any]:
    """Build a mock SAP S/4HANA API response with procurement and energy data."""
    return {
        "company_code": "1000",
        "company_name": "GlobalManufact Corp",
        "fiscal_year": "2025",
        "modules": {
            "MM": {
                "purchase_orders": [
                    {"po_number": "4500001234", "vendor": "V001", "material": "RAW-STEEL",
                     "quantity": Decimal("15000"), "unit": "KG", "amount": Decimal("45000"),
                     "currency": "EUR", "plant": "P001"},
                    {"po_number": "4500001235", "vendor": "V002", "material": "ELECTRICITY",
                     "quantity": Decimal("850000"), "unit": "KWH", "amount": Decimal("127500"),
                     "currency": "EUR", "plant": "P001"},
                ],
                "vendor_master": [
                    {"vendor_id": "V001", "name": "SteelCorp AG", "country": "DE",
                     "commodity_group": "raw_materials"},
                    {"vendor_id": "V002", "name": "EnergieWerk GmbH", "country": "DE",
                     "commodity_group": "energy"},
                ],
            },
            "FI_CO": {
                "cost_centers": [
                    {"cc": "CC-PROD", "name": "Production", "total_cost": Decimal("45000000")},
                    {"cc": "CC-ADMIN", "name": "Administration", "total_cost": Decimal("12000000")},
                    {"cc": "CC-LOGIS", "name": "Logistics", "total_cost": Decimal("8500000")},
                ],
            },
            "PM": {
                "equipment_list": [
                    {"equipment_id": "EQ-001", "type": "boiler", "fuel_type": "natural_gas",
                     "capacity_kw": 5000, "annual_consumption_kwh": Decimal("12000000")},
                    {"equipment_id": "EQ-002", "type": "chiller", "refrigerant": "R-410A",
                     "charge_kg": Decimal("85"), "annual_leak_rate_pct": Decimal("5")},
                ],
            },
            "HCM": {
                "employee_count": 12500,
                "locations": [
                    {"plant": "P001", "country": "DE", "headcount": 4500},
                    {"plant": "P002", "country": "US", "headcount": 3200},
                    {"plant": "P003", "country": "CN", "headcount": 2800},
                    {"plant": "P004", "country": "SG", "headcount": 2000},
                ],
            },
        },
    }


@pytest.fixture
def mock_oracle_response() -> Dict[str, Any]:
    """Build a mock Oracle ERP Cloud API response."""
    return {
        "business_unit": "GlobalBank Holdings",
        "ledger": "Primary Ledger",
        "procurement": {
            "spend_by_category": [
                {"category": "IT_Equipment", "amount_usd": Decimal("85000000")},
                {"category": "Professional_Services", "amount_usd": Decimal("120000000")},
                {"category": "Facilities", "amount_usd": Decimal("45000000")},
                {"category": "Travel", "amount_usd": Decimal("28000000")},
            ],
        },
        "financials": {
            "revenue_usd": Decimal("45000000000"),
            "operating_expenses_usd": Decimal("28000000000"),
            "capex_usd": Decimal("3500000000"),
        },
        "supply_chain": {
            "supplier_count": 25000,
            "active_pos": 180000,
        },
    }


@pytest.fixture
def mock_workday_response() -> Dict[str, Any]:
    """Build a mock Workday HCM API response."""
    return {
        "organization_name": "TechGlobal Inc",
        "workers": {
            "total_headcount": 42000,
            "by_location": [
                {"location": "San Francisco", "country": "US", "headcount": 8500,
                 "avg_commute_km": Decimal("22"), "remote_pct": Decimal("40")},
                {"location": "London", "country": "GB", "headcount": 5200,
                 "avg_commute_km": Decimal("18"), "remote_pct": Decimal("35")},
                {"location": "Berlin", "country": "DE", "headcount": 3800,
                 "avg_commute_km": Decimal("15"), "remote_pct": Decimal("30")},
                {"location": "Tokyo", "country": "JP", "headcount": 2500,
                 "avg_commute_km": Decimal("28"), "remote_pct": Decimal("20")},
                {"location": "Bangalore", "country": "IN", "headcount": 12000,
                 "avg_commute_km": Decimal("16"), "remote_pct": Decimal("25")},
            ],
        },
        "expenses": {
            "travel_spend_usd": Decimal("85000000"),
            "air_travel_pct": Decimal("65"),
            "rail_travel_pct": Decimal("20"),
            "car_travel_pct": Decimal("15"),
        },
    }


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
# Fixtures -- Mock ERP Clients
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sap_client(mock_sap_response):
    """Create a mock SAP S/4HANA API client."""
    client = MagicMock()
    client.get_company = AsyncMock(return_value={"company_code": "1000", "name": "GlobalManufact Corp"})
    client.get_purchase_orders = AsyncMock(return_value=mock_sap_response["modules"]["MM"]["purchase_orders"])
    client.get_vendor_master = AsyncMock(return_value=mock_sap_response["modules"]["MM"]["vendor_master"])
    client.get_cost_centers = AsyncMock(return_value=mock_sap_response["modules"]["FI_CO"]["cost_centers"])
    client.get_equipment = AsyncMock(return_value=mock_sap_response["modules"]["PM"]["equipment_list"])
    client.get_employees = AsyncMock(return_value=mock_sap_response["modules"]["HCM"])
    client.is_connected = MagicMock(return_value=True)
    client.refresh_token = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_oracle_client(mock_oracle_response):
    """Create a mock Oracle ERP Cloud client."""
    client = MagicMock()
    client.get_business_unit = AsyncMock(return_value={"name": mock_oracle_response["business_unit"]})
    client.get_spend_by_category = AsyncMock(return_value=mock_oracle_response["procurement"]["spend_by_category"])
    client.get_financials = AsyncMock(return_value=mock_oracle_response["financials"])
    client.get_supplier_count = AsyncMock(return_value=mock_oracle_response["supply_chain"]["supplier_count"])
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_workday_client(mock_workday_response):
    """Create a mock Workday HCM client."""
    client = MagicMock()
    client.get_organization = AsyncMock(return_value={"name": mock_workday_response["organization_name"]})
    client.get_workers = AsyncMock(return_value=mock_workday_response["workers"])
    client.get_expenses = AsyncMock(return_value=mock_workday_response["expenses"])
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
# Fixtures -- Scope 3 Activity Data
# ---------------------------------------------------------------------------


@pytest.fixture
def enterprise_scope3_data() -> Dict[str, Any]:
    """Build comprehensive Scope 3 activity data for all 15 categories."""
    return {
        "cat1_purchased_goods": {
            "total_spend_usd": Decimal("780000000"),
            "supplier_specific_pct": Decimal("45"),
            "average_data_pct": Decimal("35"),
            "spend_based_pct": Decimal("20"),
        },
        "cat2_capital_goods": {
            "total_spend_usd": Decimal("120000000"),
            "asset_categories": ["machinery", "vehicles", "it_equipment", "buildings"],
        },
        "cat3_fuel_energy": {
            "auto_calculated": True,
            "wtt_factor": Decimal("0.03"),
            "td_factor": Decimal("0.02"),
        },
        "cat4_upstream_transport": {
            "total_tkm": Decimal("450000000"),
            "mode_split": {"road": Decimal("60"), "rail": Decimal("15"),
                           "sea": Decimal("20"), "air": Decimal("5")},
        },
        "cat5_waste": {
            "total_tonnes": Decimal("85000"),
            "treatment_split": {"landfill": Decimal("20"), "incineration": Decimal("30"),
                                "recycling": Decimal("45"), "composting": Decimal("5")},
        },
        "cat6_business_travel": {
            "air_pkm": Decimal("320000000"),
            "rail_pkm": Decimal("45000000"),
            "car_km": Decimal("28000000"),
        },
        "cat7_employee_commuting": {
            "total_employees": 12500,
            "avg_commute_km": Decimal("18"),
            "working_days_per_year": 235,
            "remote_work_pct": Decimal("25"),
        },
        "cat8_upstream_leased": {
            "leased_area_sqm": Decimal("120000"),
            "energy_intensity_kwh_per_sqm": Decimal("150"),
        },
        "cat9_downstream_transport": {
            "total_tkm": Decimal("280000000"),
            "mode_split": {"road": Decimal("70"), "rail": Decimal("10"),
                           "sea": Decimal("15"), "air": Decimal("5")},
        },
        "cat10_processing": {
            "products_requiring_processing": ["intermediate_chemical_A", "steel_component_B"],
            "processing_energy_kwh": Decimal("45000000"),
        },
        "cat11_use_of_sold": {
            "product_categories": ["industrial_equipment"],
            "avg_energy_per_unit_kwh": Decimal("8500"),
            "avg_lifetime_years": 15,
            "units_sold": 12000,
        },
        "cat12_end_of_life": {
            "product_mass_tonnes": Decimal("180000"),
            "disposal_split": {"landfill": Decimal("30"), "incineration": Decimal("25"),
                               "recycling": Decimal("40"), "reuse": Decimal("5")},
        },
        "cat13_downstream_leased": {
            "leased_area_sqm": Decimal("85000"),
            "energy_intensity_kwh_per_sqm": Decimal("180"),
        },
        "cat14_franchises": {
            "franchise_count": 0,
        },
        "cat15_investments": {
            "financed_emissions_enabled": False,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures -- Carbon Pricing Data
# ---------------------------------------------------------------------------


@pytest.fixture
def carbon_pricing_data() -> Dict[str, Any]:
    """Build sample carbon pricing configuration."""
    return {
        "shadow_price_usd_per_tco2e": Decimal("85"),
        "escalation_rate_pct_per_year": Decimal("5"),
        "ets_price_usd_per_tco2e": Decimal("72"),
        "cbam_certificate_price_eur": Decimal("68"),
        "business_units": [
            {"name": "Manufacturing EU", "scope1_tco2e": Decimal("45000"),
             "scope2_tco2e": Decimal("18000"), "revenue_usd": Decimal("850000000")},
            {"name": "Manufacturing US", "scope1_tco2e": Decimal("38000"),
             "scope2_tco2e": Decimal("22000"), "revenue_usd": Decimal("720000000")},
            {"name": "Logistics", "scope1_tco2e": Decimal("12000"),
             "scope2_tco2e": Decimal("5000"), "revenue_usd": Decimal("350000000")},
        ],
        "investment_proposals": [
            {"name": "Solar Farm Phase 1", "capex_usd": Decimal("15000000"),
             "annual_reduction_tco2e": Decimal("8500"), "lifetime_years": 25},
            {"name": "Fleet Electrification", "capex_usd": Decimal("8000000"),
             "annual_reduction_tco2e": Decimal("4200"), "lifetime_years": 10},
            {"name": "Heat Pump Retrofit", "capex_usd": Decimal("3500000"),
             "annual_reduction_tco2e": Decimal("2800"), "lifetime_years": 20},
        ],
    }


# ---------------------------------------------------------------------------
# Fixtures -- Supplier Data
# ---------------------------------------------------------------------------


@pytest.fixture
def supplier_data() -> List[Dict[str, Any]]:
    """Build sample supplier data for supply chain mapping."""
    return [
        {"id": "S001", "name": "SteelCorp AG", "country": "DE", "tier": 1,
         "spend_usd": Decimal("120000000"), "emissions_tco2e": Decimal("85000"),
         "cdp_score": "B", "sbti_status": "committed", "dq_level": 2},
        {"id": "S002", "name": "ChemWorks Inc", "country": "US", "tier": 1,
         "spend_usd": Decimal("95000000"), "emissions_tco2e": Decimal("62000"),
         "cdp_score": "A-", "sbti_status": "validated", "dq_level": 1},
        {"id": "S003", "name": "PlastiPack Ltd", "country": "GB", "tier": 1,
         "spend_usd": Decimal("78000000"), "emissions_tco2e": Decimal("45000"),
         "cdp_score": "C", "sbti_status": "none", "dq_level": 3},
        {"id": "S004", "name": "EnergiePro GmbH", "country": "DE", "tier": 2,
         "spend_usd": Decimal("45000000"), "emissions_tco2e": Decimal("28000"),
         "cdp_score": "B", "sbti_status": "committed", "dq_level": 2},
        {"id": "S005", "name": "LogiTrans BV", "country": "NL", "tier": 2,
         "spend_usd": Decimal("32000000"), "emissions_tco2e": Decimal("18000"),
         "cdp_score": "D", "sbti_status": "none", "dq_level": 4},
        {"id": "S006", "name": "RawMat Asia Pte", "country": "SG", "tier": 2,
         "spend_usd": Decimal("28000000"), "emissions_tco2e": Decimal("35000"),
         "cdp_score": "Not Disclosed", "sbti_status": "none", "dq_level": 5},
        {"id": "S007", "name": "TechParts Co", "country": "KR", "tier": 3,
         "spend_usd": Decimal("15000000"), "emissions_tco2e": Decimal("12000"),
         "cdp_score": "C", "sbti_status": "none", "dq_level": 3},
        {"id": "S008", "name": "PackageCo SA", "country": "FR", "tier": 3,
         "spend_usd": Decimal("8000000"), "emissions_tco2e": Decimal("5500"),
         "cdp_score": "B-", "sbti_status": "committed", "dq_level": 3},
    ]


# ---------------------------------------------------------------------------
# Fixtures -- SBTi Validation Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sbti_baseline_data() -> Dict[str, Any]:
    """Build sample baseline data for SBTi target setting."""
    return {
        "base_year": 2024,
        "target_year_near_term": 2030,
        "target_year_long_term": 2050,
        "scope1_tco2e": Decimal("125000"),
        "scope2_location_tco2e": Decimal("85000"),
        "scope2_market_tco2e": Decimal("62000"),
        "scope3_total_tco2e": Decimal("680000"),
        "scope3_by_category": {
            1: Decimal("285000"), 2: Decimal("45000"), 3: Decimal("28000"),
            4: Decimal("52000"), 5: Decimal("8500"), 6: Decimal("22000"),
            7: Decimal("15000"), 8: Decimal("12000"), 9: Decimal("48000"),
            10: Decimal("35000"), 11: Decimal("85000"), 12: Decimal("18000"),
            13: Decimal("8000"), 14: Decimal("0"), 15: Decimal("18500"),
        },
        "total_emissions_tco2e": Decimal("867000"),
        "scope1_coverage_pct": Decimal("98"),
        "scope2_coverage_pct": Decimal("97"),
        "scope3_coverage_pct": Decimal("72"),
        "flag_emissions_pct": Decimal("3"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Scenario Modeling Data
# ---------------------------------------------------------------------------


@pytest.fixture
def scenario_parameters() -> Dict[str, Any]:
    """Build sample scenario parameters for Monte Carlo simulation."""
    return {
        "scenarios": {
            "aggressive_15c": {
                "carbon_price_2030_usd": Decimal("150"),
                "grid_decarbonization_rate_pct": Decimal("6"),
                "ev_adoption_rate_2030_pct": Decimal("80"),
                "re_procurement_2030_pct": Decimal("100"),
                "supplier_engagement_rate_pct": Decimal("70"),
            },
            "moderate_2c": {
                "carbon_price_2030_usd": Decimal("85"),
                "grid_decarbonization_rate_pct": Decimal("4"),
                "ev_adoption_rate_2030_pct": Decimal("50"),
                "re_procurement_2030_pct": Decimal("80"),
                "supplier_engagement_rate_pct": Decimal("45"),
            },
            "bau": {
                "carbon_price_2030_usd": Decimal("35"),
                "grid_decarbonization_rate_pct": Decimal("2"),
                "ev_adoption_rate_2030_pct": Decimal("20"),
                "re_procurement_2030_pct": Decimal("40"),
                "supplier_engagement_rate_pct": Decimal("20"),
            },
        },
        "monte_carlo_runs": 10000,
        "confidence_intervals": [10, 25, 50, 75, 90],
    }
