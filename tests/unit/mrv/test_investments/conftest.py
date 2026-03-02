# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-028: Investments (Scope 3 Category 15) Agent.

Provides comprehensive test fixtures for:
- Equity investment inputs (listed equity Apple, private equity)
- Debt investment inputs (corporate bond, project finance)
- Real asset inputs (CRE office, residential mortgage, motor vehicle loan)
- Sovereign bond inputs (US Treasury)
- Mixed portfolio inputs (multiple asset classes)
- Configuration objects and mock engines
- Database engine mocks, singleton resets

Usage:
    def test_something(sample_equity_input, mock_database_engine):
        result = calculate(sample_equity_input, mock_database_engine)
        assert result.financed_emissions > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest


# ============================================================================
# EQUITY INVESTMENT INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_equity_input() -> Dict[str, Any]:
    """Listed equity investment -- Apple Inc via EVIC attribution."""
    return {
        "asset_class": "listed_equity",
        "investee_name": "Apple Inc.",
        "isin": "US0378331005",
        "outstanding_amount": Decimal("100000000"),  # $100M
        "evic": Decimal("3000000000000"),  # $3T
        "investee_scope1": Decimal("22400"),  # tCO2e
        "investee_scope2": Decimal("9100"),  # tCO2e
        "sector": "information_technology",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 1,
    }


@pytest.fixture
def sample_private_equity_input() -> Dict[str, Any]:
    """Private equity investment -- unlisted company via equity share."""
    return {
        "asset_class": "private_equity",
        "investee_name": "GreenTech Solutions Ltd",
        "outstanding_amount": Decimal("50000000"),  # $50M equity
        "total_equity_plus_debt": Decimal("200000000"),  # $200M
        "investee_scope1": Decimal("15000"),  # tCO2e
        "investee_scope2": Decimal("8000"),  # tCO2e
        "sector": "industrials",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }


# ============================================================================
# DEBT INVESTMENT INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_corporate_bond_input() -> Dict[str, Any]:
    """Corporate bond investment -- EVIC attribution."""
    return {
        "asset_class": "corporate_bond",
        "investee_name": "Tesla Inc.",
        "isin": "US88160RAJ68",
        "outstanding_amount": Decimal("75000000"),  # $75M bond
        "evic": Decimal("500000000000"),  # $500B
        "investee_scope1": Decimal("30000"),  # tCO2e
        "investee_scope2": Decimal("12000"),  # tCO2e
        "sector": "consumer_discretionary",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 1,
    }


@pytest.fixture
def sample_project_finance_input() -> Dict[str, Any]:
    """Project finance -- solar project pro-rata cost attribution."""
    return {
        "asset_class": "project_finance",
        "project_name": "SunBelt Solar Farm",
        "outstanding_amount": Decimal("30000000"),  # $30M
        "total_project_cost": Decimal("100000000"),  # $100M
        "project_lifetime_years": 25,
        "annual_project_emissions": Decimal("500"),  # tCO2e/yr
        "sector": "utilities",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }


# ============================================================================
# REAL ASSET INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_cre_input() -> Dict[str, Any]:
    """Commercial real estate (CRE) -- office building via EUI method."""
    return {
        "asset_class": "commercial_real_estate",
        "property_name": "Downtown Office Tower",
        "outstanding_amount": Decimal("25000000"),  # $25M loan
        "property_value": Decimal("50000000"),  # $50M
        "floor_area_m2": Decimal("10000"),
        "property_type": "office",
        "epc_rating": "B",
        "climate_zone": "temperate",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }


@pytest.fixture
def sample_mortgage_input() -> Dict[str, Any]:
    """Residential mortgage -- LTV-weighted property emissions."""
    return {
        "asset_class": "mortgage",
        "property_name": "123 Oak Street",
        "outstanding_amount": Decimal("300000"),  # $300K loan
        "property_value": Decimal("400000"),  # $400K
        "floor_area_m2": Decimal("150"),
        "property_type": "residential",
        "epc_rating": "C",
        "climate_zone": "temperate",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 3,
    }


@pytest.fixture
def sample_motor_vehicle_input() -> Dict[str, Any]:
    """Motor vehicle loan -- passenger car via per-vehicle EFs."""
    return {
        "asset_class": "motor_vehicle_loan",
        "vehicle_description": "2024 Toyota Camry Hybrid",
        "outstanding_amount": Decimal("25000"),  # $25K loan
        "vehicle_value": Decimal("35000"),  # $35K
        "vehicle_category": "passenger_car",
        "fuel_type": "hybrid",
        "annual_mileage_km": Decimal("20000"),
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 3,
    }


# ============================================================================
# SOVEREIGN BOND INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_sovereign_bond_input() -> Dict[str, Any]:
    """Sovereign bond -- US Treasury via GDP-PPP attribution."""
    return {
        "asset_class": "sovereign_bond",
        "country": "US",
        "outstanding_amount": Decimal("500000000"),  # $500M
        "gdp_ppp": Decimal("25460000000000"),  # $25.46T
        "country_emissions": Decimal("5222000000"),  # 5.222 GtCO2e
        "include_lulucf": False,
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 4,
    }


# ============================================================================
# PORTFOLIO INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_portfolio_input(
    sample_equity_input,
    sample_corporate_bond_input,
    sample_cre_input,
    sample_sovereign_bond_input,
) -> Dict[str, Any]:
    """Mixed portfolio with multiple asset classes."""
    return {
        "portfolio_name": "GreenBank Balanced Portfolio",
        "reporting_year": 2024,
        "currency": "USD",
        "investments": [
            sample_equity_input,
            sample_corporate_bond_input,
            sample_cre_input,
            sample_sovereign_bond_input,
        ],
    }


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================


@pytest.fixture
def mock_database_engine() -> MagicMock:
    """Mock InvestmentDatabaseEngine with deterministic returns."""
    engine = MagicMock()
    engine.get_sector_ef.return_value = Decimal("0.450")
    engine.get_country_emissions.return_value = Decimal("5222000000")
    engine.get_grid_ef.return_value = Decimal("0.417")
    engine.get_building_benchmark.return_value = Decimal("200.0")
    engine.get_vehicle_ef.return_value = Decimal("0.120")
    engine.get_currency_rate.return_value = Decimal("1.0")
    engine.get_sovereign_data.return_value = {
        "gdp_ppp": Decimal("25460000000000"),
        "total_emissions": Decimal("5222000000"),
        "population": 332000000,
    }
    engine.get_pcaf_quality_criteria.return_value = {
        "score": 1,
        "description": "Reported emissions, audited",
        "data_type": "reported",
    }
    engine.get_carbon_intensity_benchmark.return_value = Decimal("15.5")
    return engine


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock InvestmentsConfig with default values."""
    config = MagicMock()
    config.general.enabled = True
    config.general.debug = False
    config.general.log_level = "INFO"
    config.general.agent_id = "GL-MRV-S3-015"
    config.general.agent_component = "AGENT-MRV-028"
    config.general.version = "1.0.0"
    config.general.api_prefix = "/api/v1/investments"
    config.general.max_batch_size = 1000
    config.general.default_gwp = "AR5"
    config.general.default_attribution_method = "EVIC"
    config.database.pool_size = 5
    config.database.max_overflow = 10
    config.equity.default_evic_source = "BLOOMBERG"
    config.equity.include_scope3 = False
    config.debt.green_bond_discount = Decimal("0.0")
    config.real_assets.default_eui_source = "CRREM"
    config.sovereign.include_lulucf = False
    config.compliance.get_frameworks.return_value = [
        "GHG_PROTOCOL_SCOPE3",
        "PCAF",
        "ISO_14064",
        "CSRD_ESRS_E1",
        "CDP",
        "SBTI_FI",
        "SB_253",
        "TCFD",
        "NZBA",
    ]
    config.compliance.strict_mode = False
    config.compliance.materiality_threshold = Decimal("0.01")
    config.provenance.chain_algorithm = "SHA-256"
    config.provenance.merkle_enabled = True
    config.metrics.prefix = "gl_inv_"
    config.cache.ttl_seconds = 3600
    return config


@pytest.fixture
def reset_singletons():
    """Reset all singleton engines before and after each test."""
    try:
        from greenlang.investments.investment_database import reset_database_engine
        reset_database_engine()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from greenlang.investments.investment_database import reset_database_engine
        reset_database_engine()
    except (ImportError, AttributeError):
        pass
