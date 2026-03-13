# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-016 Country Risk Evaluator test suite.

Provides reusable fixtures for configuration, engine instances, sample data,
and shared test utilities used across all test modules.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.country_risk_evaluator.config import (
    CountryRiskEvaluatorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.country_risk_evaluator.provenance import (
    reset_provenance_tracker,
)
from greenlang.agents.eudr.country_risk_evaluator.country_risk_scorer import (
    CountryRiskScorer,
)
from greenlang.agents.eudr.country_risk_evaluator.commodity_risk_analyzer import (
    CommodityRiskAnalyzer,
)
from greenlang.agents.eudr.country_risk_evaluator.deforestation_hotspot_detector import (
    DeforestationHotspotDetector,
)
from greenlang.agents.eudr.country_risk_evaluator.governance_index_engine import (
    GovernanceIndexEngine,
)
from greenlang.agents.eudr.country_risk_evaluator.due_diligence_classifier import (
    DueDiligenceClassifier,
)
from greenlang.agents.eudr.country_risk_evaluator.trade_flow_analyzer import (
    TradeFlowAnalyzer,
)
from greenlang.agents.eudr.country_risk_evaluator.risk_report_generator import (
    RiskReportGenerator,
)
from greenlang.agents.eudr.country_risk_evaluator.regulatory_update_tracker import (
    RegulatoryUpdateTracker,
)
try:
    from greenlang.agents.eudr.country_risk_evaluator.setup import (
        CountryRiskEvaluatorService,
    )
except ImportError:
    CountryRiskEvaluatorService = None  # type: ignore[assignment,misc]
from greenlang.agents.eudr.country_risk_evaluator.models import (
    CommodityType,
    RiskLevel,
    DueDiligenceLevel,
    HotspotSeverity,
    GovernanceIndicator,
    TrendDirection,
    AssessmentConfidence,
    ReportFormat,
    ReportType,
    TradeFlowDirection,
    RegulatoryStatus,
    CertificationScheme,
    SUPPORTED_COMMODITIES,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
    DEFAULT_FACTOR_WEIGHTS,
    SUPPORTED_COUNTRIES,
)


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------


class DeterministicUUID:
    """Generate sequential identifiers for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:08d}"

    def reset(self):
        self._counter = 0


# ---------------------------------------------------------------------------
# Provenance hash helper
# ---------------------------------------------------------------------------


def compute_test_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for test assertions."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a CountryRiskEvaluatorConfig with test defaults."""
    cfg = CountryRiskEvaluatorConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        pool_size=2,
        # Country risk scoring
        deforestation_weight=30,
        governance_weight=20,
        enforcement_weight=15,
        corruption_weight=15,
        forest_law_weight=10,
        trend_weight=10,
        score_normalization="minmax",
        confidence_threshold=0.6,
        data_freshness_max_days=365,
        ec_benchmark_override=True,
        low_risk_threshold=30,
        high_risk_threshold=65,
        min_factor_weight=5,
        max_factor_weight=50,
        # Commodity analysis
        correlation_threshold=0.5,
        enable_seasonal_analysis=True,
        certification_weight=0.3,
        production_volume_weight=0.2,
        supply_chain_complexity_max=10,
        # Hotspot detection
        alert_threshold=70,
        clustering_min_points=3,
        clustering_radius_km=10.0,
        enable_fire_correlation=True,
        protected_area_buffer_km=10.0,
        min_deforestation_rate_alert=0.5,
        indigenous_territory_buffer_km=10.0,
        trend_window_years=5,
        # Governance
        wgi_weight=0.30,
        cpi_weight=0.30,
        forest_governance_weight=0.25,
        gov_enforcement_weight=0.15,
        enable_legal_framework_scoring=True,
        enable_judicial_scoring=True,
        # Due diligence
        simplified_threshold=30,
        enhanced_threshold=60,
        certification_credit_max=30,
        audit_frequency_multiplier=1.0,
        simplified_cost_min_eur=200,
        simplified_cost_max_eur=500,
        standard_cost_min_eur=1000,
        standard_cost_max_eur=3000,
        enhanced_cost_min_eur=5000,
        enhanced_cost_max_eur=15000,
        enable_time_to_compliance=True,
        # Trade flows
        min_trade_volume_tonnes=10.0,
        re_export_risk_threshold=0.7,
        hs_code_depth=6,
        enable_concentration_risk=True,
        enable_sanction_overlay=True,
        enable_fta_impact=True,
        # Reports
        default_language="en",
        report_retention_days=1825,
        max_report_size_mb=50,
        # Regulatory
        monitoring_interval_hours=24,
        grace_period_days=90,
        enable_enforcement_tracking=True,
        # Batch
        batch_max_size=500,
        batch_concurrency=4,
        batch_timeout_s=300,
        # Provenance
        enable_provenance=True,
        genesis_hash="GL-EUDR-CRE-016-TEST-GENESIS",
        chain_algorithm="sha256",
        # Metrics
        enable_metrics=False,
    )
    set_config(cfg)
    yield cfg
    reset_config()


@pytest.fixture(autouse=True)
def reset_state():
    """Reset config and provenance between tests."""
    yield
    reset_config()
    reset_provenance_tracker()


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def country_risk_scorer(mock_config):
    """Create a CountryRiskScorer instance for testing."""
    return CountryRiskScorer()


@pytest.fixture
def commodity_analyzer(mock_config):
    """Create a CommodityRiskAnalyzer instance for testing."""
    return CommodityRiskAnalyzer()


@pytest.fixture
def hotspot_detector(mock_config):
    """Create a DeforestationHotspotDetector instance for testing."""
    return DeforestationHotspotDetector()


@pytest.fixture
def governance_engine(mock_config):
    """Create a GovernanceIndexEngine instance for testing."""
    return GovernanceIndexEngine()


@pytest.fixture
def due_diligence_classifier(mock_config):
    """Create a DueDiligenceClassifier instance for testing."""
    return DueDiligenceClassifier()


@pytest.fixture
def trade_flow_analyzer(mock_config):
    """Create a TradeFlowAnalyzer instance for testing."""
    return TradeFlowAnalyzer()


@pytest.fixture
def report_generator(mock_config):
    """Create a RiskReportGenerator instance for testing."""
    return RiskReportGenerator()


@pytest.fixture
def regulatory_tracker(mock_config):
    """Create a RegulatoryUpdateTracker instance for testing."""
    return RegulatoryUpdateTracker()


@pytest.fixture
def service(mock_config):
    """Create CountryRiskEvaluatorService instance for testing."""
    if CountryRiskEvaluatorService is None:
        pytest.skip("CountryRiskEvaluatorService not available due to import issues")
    return CountryRiskEvaluatorService()


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_factor_values():
    """Standard 6-factor values for country risk scoring.

    Normalization (applied by CountryRiskScorer._normalize_factors):
      - deforestation_rate: clamp [0,10] then *10  -> 5.0 -> 50.0
      - governance_quality: 100 - value (invert)   -> 100-45 = 55.0
      - enforcement_effectiveness: 100 - value      -> 100-40 = 60.0
      - corruption_index: 100 - value (CPI invert) -> 100-55 = 45.0
      - forest_law_compliance: 100 - value          -> 100-50 = 50.0
      - historical_trend: (clamp[-10,10]+10)/20*100 -> (3+10)/20*100 = 65.0
    Weighted: 50*0.30 + 55*0.20 + 60*0.15 + 45*0.15 + 50*0.10 + 65*0.10
            = 15 + 11 + 9 + 6.75 + 5 + 6.5 = 53.25 (STANDARD risk)
    """
    return {
        "deforestation_rate": 5.0,     # annual % forest loss [0-10 scale]
        "governance_quality": 45.0,    # 0-100 scale (higher = better)
        "enforcement_effectiveness": 40.0,  # 0-100 (higher = better)
        "corruption_index": 55.0,      # CPI 0-100 (higher = less corrupt)
        "forest_law_compliance": 50.0,  # 0-100 (higher = better)
        "historical_trend": 3.0,       # trend slope [-10 to +10]
    }


@pytest.fixture
def sample_low_risk_factors():
    """Factor values that produce a LOW risk classification (<= 30).

    Normalization:
      - deforestation_rate: 0.5 -> clamp 0.5 * 10 = 5.0
      - governance_quality: 100 - 90 = 10.0
      - enforcement_effectiveness: 100 - 88 = 12.0
      - corruption_index: 100 - 85 = 15.0
      - forest_law_compliance: 100 - 90 = 10.0
      - historical_trend: (-8+10)/20*100 = 10.0
    Weighted: 5*0.30 + 10*0.20 + 12*0.15 + 15*0.15 + 10*0.10 + 10*0.10
            = 1.5 + 2 + 1.8 + 2.25 + 1 + 1 = 9.55 (LOW risk)
    """
    return {
        "deforestation_rate": 0.5,
        "governance_quality": 90.0,
        "enforcement_effectiveness": 88.0,
        "corruption_index": 85.0,
        "forest_law_compliance": 90.0,
        "historical_trend": -8.0,
    }


@pytest.fixture
def sample_standard_risk_factors():
    """Factor values that produce a STANDARD risk classification (31-65).

    Normalization:
      - deforestation_rate: 4.0 -> 40.0
      - governance_quality: 100 - 55 = 45.0
      - enforcement_effectiveness: 100 - 50 = 50.0
      - corruption_index: 100 - 50 = 50.0
      - forest_law_compliance: 100 - 55 = 45.0
      - historical_trend: (2+10)/20*100 = 60.0
    Weighted: 40*0.30 + 45*0.20 + 50*0.15 + 50*0.15 + 45*0.10 + 60*0.10
            = 12 + 9 + 7.5 + 7.5 + 4.5 + 6 = 46.5 (STANDARD risk)
    """
    return {
        "deforestation_rate": 4.0,
        "governance_quality": 55.0,
        "enforcement_effectiveness": 50.0,
        "corruption_index": 50.0,
        "forest_law_compliance": 55.0,
        "historical_trend": 2.0,
    }


@pytest.fixture
def sample_high_risk_factors():
    """Factor values that produce a HIGH risk classification (> 65).

    Normalization:
      - deforestation_rate: 9.0 -> 90.0
      - governance_quality: 100 - 15 = 85.0
      - enforcement_effectiveness: 100 - 20 = 80.0
      - corruption_index: 100 - 25 = 75.0
      - forest_law_compliance: 100 - 20 = 80.0
      - historical_trend: (8+10)/20*100 = 90.0
    Weighted: 90*0.30 + 85*0.20 + 80*0.15 + 75*0.15 + 80*0.10 + 90*0.10
            = 27 + 17 + 12 + 11.25 + 8 + 9 = 84.25 (HIGH risk)
    """
    return {
        "deforestation_rate": 9.0,
        "governance_quality": 15.0,
        "enforcement_effectiveness": 20.0,
        "corruption_index": 25.0,
        "forest_law_compliance": 20.0,
        "historical_trend": 8.0,
    }


@pytest.fixture
def sample_deforestation_events():
    """Sample deforestation events for hotspot detection."""
    base_lat = -3.5
    base_lon = -62.3
    return [
        {
            "latitude": base_lat + i * 0.01,
            "longitude": base_lon + i * 0.01,
            "date": f"2024-01-{15 + i:02d}",
            "area_ha": 10.0 + i * 2,
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_fire_alerts():
    """Sample fire alert data for correlation analysis."""
    base_lat = -3.5
    base_lon = -62.3
    return [
        {
            "latitude": base_lat + i * 0.012,
            "longitude": base_lon + i * 0.012,
            "date": f"2024-01-{16 + i:02d}",
            "confidence": 0.85 + i * 0.01,
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_wgi_data():
    """Sample World Bank WGI scores for governance testing."""
    return {
        "voice_accountability": 55.0,
        "political_stability": 40.0,
        "government_effectiveness": 48.0,
        "regulatory_quality": 52.0,
        "rule_of_law": 42.0,
        "control_of_corruption": 38.0,
    }


@pytest.fixture
def sample_trade_flow_data():
    """Sample trade flow data for testing."""
    return {
        "origin_country": "BR",
        "destination_country": "NL",
        "commodity_type": "soya",
        "volume_tonnes": 50000.0,
        "value_usd": 25000000.0,
        "direction": "export",
        "hs_codes": ["1201", "1208"],
        "quarter": "2025-Q4",
    }


@pytest.fixture
def sample_country_data():
    """Sample country assessment dataset for batch and comparison tests.

    Uses raw values aligned with CountryRiskScorer normalization logic:
      - deforestation_rate: [0-10] annual % forest loss
      - governance/enforcement/corruption/compliance: [0-100] higher = better
      - historical_trend: [-10,+10] slope

    BR: high risk (~84), ID: high risk (~79), SE: low risk (~10), DE: low risk (~17)
    """
    return [
        {
            "country_code": "BR",
            "country_name": "Brazil",
            "factor_values": {
                "deforestation_rate": 9.0,      # -> 90.0
                "governance_quality": 15.0,     # -> 85.0
                "enforcement_effectiveness": 20.0,  # -> 80.0
                "corruption_index": 25.0,       # -> 75.0
                "forest_law_compliance": 20.0,  # -> 80.0
                "historical_trend": 8.0,        # -> 90.0
            },
        },
        {
            "country_code": "ID",
            "country_name": "Indonesia",
            "factor_values": {
                "deforestation_rate": 8.0,      # -> 80.0
                "governance_quality": 20.0,     # -> 80.0
                "enforcement_effectiveness": 25.0,  # -> 75.0
                "corruption_index": 30.0,       # -> 70.0
                "forest_law_compliance": 25.0,  # -> 75.0
                "historical_trend": 6.0,        # -> 80.0
            },
        },
        {
            "country_code": "SE",
            "country_name": "Sweden",
            "factor_values": {
                "deforestation_rate": 0.3,      # -> 3.0
                "governance_quality": 92.0,     # -> 8.0
                "enforcement_effectiveness": 90.0,  # -> 10.0
                "corruption_index": 88.0,       # -> 12.0
                "forest_law_compliance": 93.0,  # -> 7.0
                "historical_trend": -9.0,       # -> 5.0
            },
        },
        {
            "country_code": "DE",
            "country_name": "Germany",
            "factor_values": {
                "deforestation_rate": 0.5,      # -> 5.0
                "governance_quality": 85.0,     # -> 15.0
                "enforcement_effectiveness": 82.0,  # -> 18.0
                "corruption_index": 80.0,       # -> 20.0
                "forest_law_compliance": 85.0,  # -> 15.0
                "historical_trend": -7.0,       # -> 15.0
            },
        },
    ]


@pytest.fixture
def sample_commodity_data():
    """Sample commodity risk analysis input."""
    return {
        "country_code": "BR",
        "commodity_type": "soya",
        "country_risk_score": 72.5,
        "region": "Mato Grosso",
        "certification_schemes": ["iscc"],
        "production_volume": 150000.0,
        "month": 11,
    }


@pytest.fixture
def sample_hotspot_data():
    """Comprehensive hotspot detection input with multiple clusters."""
    events = []
    # Cluster 1: Amazon region
    for i in range(5):
        events.append({
            "latitude": -3.5 + i * 0.005,
            "longitude": -62.3 + i * 0.005,
            "date": f"2024-01-{15 + i:02d}",
            "area_ha": 12.0 + i * 3,
        })
    # Cluster 2: Cerrado region
    for i in range(5):
        events.append({
            "latitude": -12.0 + i * 0.006,
            "longitude": -49.0 + i * 0.006,
            "date": f"2024-02-{10 + i:02d}",
            "area_ha": 8.0 + i * 2,
        })
    return events


# ---------------------------------------------------------------------------
# Helper fixtures for high-risk EUDR countries
# ---------------------------------------------------------------------------


EUDR_HIGH_RISK_COUNTRIES = [
    ("BR", "Brazil"),
    ("ID", "Indonesia"),
    ("CD", "Congo, Democratic Republic"),
    ("CI", "Cote d'Ivoire"),
    ("GH", "Ghana"),
    ("MY", "Malaysia"),
    ("CO", "Colombia"),
]


@pytest.fixture(params=EUDR_HIGH_RISK_COUNTRIES, ids=[c[0] for c in EUDR_HIGH_RISK_COUNTRIES])
def high_risk_country(request):
    """Parametrized fixture for EUDR high-risk countries."""
    return request.param
