# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-018 Commodity Risk Analyzer.

Provides mock configuration, engine instance fixtures for all 8 engines,
and sample data fixtures for commodity profiling, derived product analysis,
price volatility, production data, and portfolio analysis testing.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.config import (
    CommodityRiskAnalyzerConfig,
    get_config,
    reset_config,
    set_config,
)
from greenlang.agents.eudr.commodity_risk_analyzer.models import (
    CommodityType,
    DerivedProductCategory,
    MarketCondition,
    ProcessingStage,
    RiskLevel,
    VolatilityLevel,
)
from greenlang.agents.eudr.commodity_risk_analyzer.commodity_profiler import (
    CommodityProfiler,
)
from greenlang.agents.eudr.commodity_risk_analyzer.derived_product_analyzer import (
    DerivedProductAnalyzer,
)
from greenlang.agents.eudr.commodity_risk_analyzer.price_volatility_engine import (
    PriceVolatilityEngine,
)
from greenlang.agents.eudr.commodity_risk_analyzer.production_forecast_engine import (
    ProductionForecastEngine,
)
from greenlang.agents.eudr.commodity_risk_analyzer.engines.substitution_risk_analyzer import (
    SubstitutionRiskAnalyzer,
)
from greenlang.agents.eudr.commodity_risk_analyzer.engines.regulatory_compliance_engine import (
    RegulatoryComplianceEngine,
)
from greenlang.agents.eudr.commodity_risk_analyzer.engines.commodity_due_diligence_engine import (
    CommodityDueDiligenceEngine,
)
from greenlang.agents.eudr.commodity_risk_analyzer.engines.portfolio_risk_aggregator import (
    PortfolioRiskAggregator,
)


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_config(monkeypatch):
    """Create and install a test CommodityRiskAnalyzerConfig singleton.

    Patches the global config singleton with test-appropriate values
    and resets it on teardown to avoid cross-test contamination.
    """
    test_cfg = CommodityRiskAnalyzerConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        enable_provenance=True,
        enable_metrics=False,
        batch_max_size=100,
        batch_concurrency=2,
        batch_timeout_s=60,
        retention_years=1,
        volatility_window_short_days=30,
        volatility_window_long_days=90,
        disruption_threshold=0.8,
        hhi_concentration_threshold=0.25,
        hhi_low_threshold=0.15,
        hhi_moderate_threshold=0.25,
        max_single_commodity_exposure=0.40,
        diversification_target=0.70,
        dd_default_level="standard",
        dd_completion_threshold=0.90,
        traceability_min_score=0.60,
        forecast_horizon_months=12,
        confidence_interval_width=0.95,
    )
    set_config(test_cfg)
    yield test_cfg
    reset_config()


# ---------------------------------------------------------------------------
# Engine instance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def commodity_profiler():
    """Create a fresh CommodityProfiler instance for testing."""
    return CommodityProfiler()


@pytest.fixture()
def derived_product_analyzer():
    """Create a fresh DerivedProductAnalyzer instance for testing."""
    return DerivedProductAnalyzer()


@pytest.fixture()
def price_volatility_engine():
    """Create a fresh PriceVolatilityEngine instance for testing."""
    return PriceVolatilityEngine()


@pytest.fixture()
def price_engine_with_history():
    """Create a PriceVolatilityEngine pre-loaded with synthetic cocoa history."""
    engine = PriceVolatilityEngine()
    base_price = Decimal("4800.00")
    today = datetime.now(timezone.utc).date()
    history: List[Dict[str, Any]] = []
    for i in range(120):
        d = today - timedelta(days=120 - i)
        # Small deterministic price walk
        noise = Decimal(str(((hash(str(d)) % 200) - 100) / 100.0))
        price = base_price + (noise * Decimal("50"))
        price = max(price, Decimal("3200.00"))
        history.append({"date": d.isoformat(), "price": price})
    engine.load_price_history("cocoa", history)
    return engine


@pytest.fixture()
def production_forecast_engine():
    """Create a fresh ProductionForecastEngine instance for testing."""
    return ProductionForecastEngine()


@pytest.fixture()
def substitution_risk_analyzer():
    """Create a fresh SubstitutionRiskAnalyzer instance for testing."""
    return SubstitutionRiskAnalyzer()


@pytest.fixture()
def regulatory_compliance_engine():
    """Create a fresh RegulatoryComplianceEngine instance for testing."""
    return RegulatoryComplianceEngine()


@pytest.fixture()
def commodity_dd_engine():
    """Create a fresh CommodityDueDiligenceEngine instance for testing."""
    return CommodityDueDiligenceEngine()


@pytest.fixture()
def portfolio_risk_aggregator():
    """Create a fresh PortfolioRiskAggregator instance for testing."""
    return PortfolioRiskAggregator()


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_commodity_profile():
    """Return sample commodity profile input data for oil_palm."""
    return {
        "commodity_type": "oil_palm",
        "country_data": {"ID": 60, "MY": 30, "NG": 10},
        "supply_chain_data": {
            "stages": 5,
            "intermediaries": 12,
            "countries": 3,
            "custody_models": ["mass_balance", "identity_preserved"],
        },
    }


@pytest.fixture()
def sample_processing_chain():
    """Return sample processing chain stages for cocoa chocolate production."""
    return {
        "product_id": "cocoa-choc-test-001",
        "source_commodity": "cocoa",
        "processing_stages": [
            "fermentation",
            "drying",
            "roasting",
            "conching",
            "moulding",
        ],
    }


@pytest.fixture()
def sample_price_data():
    """Return sample price history records for cocoa."""
    today = datetime.now(timezone.utc).date()
    records = []
    base = Decimal("4800.00")
    for i in range(60):
        d = today - timedelta(days=60 - i)
        noise = Decimal(str(((hash(str(d)) % 100) - 50) / 100.0))
        records.append({
            "date": d.isoformat(),
            "price": base + noise * Decimal("100"),
        })
    return records


@pytest.fixture()
def sample_production_data():
    """Return sample production data for soya in Brazil."""
    return {
        "commodity_type": "soya",
        "region": "BR",
        "year": 2025,
        "production_tonnes": Decimal("154000000"),
        "area_hectares": Decimal("44000000"),
        "yield_tonnes_per_ha": Decimal("3.50"),
    }


@pytest.fixture()
def sample_portfolio():
    """Return sample multi-commodity portfolio positions."""
    return {
        "positions": [
            {"commodity_type": "cocoa", "share": Decimal("0.30"), "risk_score": Decimal("65")},
            {"commodity_type": "coffee", "share": Decimal("0.25"), "risk_score": Decimal("45")},
            {"commodity_type": "oil_palm", "share": Decimal("0.20"), "risk_score": Decimal("80")},
            {"commodity_type": "soya", "share": Decimal("0.15"), "risk_score": Decimal("55")},
            {"commodity_type": "wood", "share": Decimal("0.10"), "risk_score": Decimal("40")},
        ],
    }


@pytest.fixture()
def all_seven_commodities():
    """Return all 7 EUDR commodity type strings."""
    return ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]
