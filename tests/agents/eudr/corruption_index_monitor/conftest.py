# -*- coding: utf-8 -*-
"""
Shared test fixtures for AGENT-EUDR-019: Corruption Index Monitor.

Provides mock configurations, sample data, and engine instances for all
test modules in the Corruption Index Monitor test suite. Fixtures cover
CPI monitoring, WGI analysis, bribery risk assessment, and institutional
quality evaluation.

Fixture Categories:
    - Configuration fixtures: mock_config, reset_singleton_config
    - CPI data fixtures: sample_cpi_data, sample_cpi_history
    - WGI data fixtures: sample_wgi_data, sample_wgi_dimensions
    - Bribery data fixtures: sample_bribery_data, sample_sector_data
    - Institutional data fixtures: sample_institutional_data
    - Country reference fixtures: sample_country_codes, eudr_countries
    - Trend data fixtures: sample_trend_data
    - Correlation data fixtures: sample_correlation_data
    - Engine fixtures: cpi_engine, wgi_engine, bribery_engine,
      institutional_engine
    - Provenance fixtures: mock_provenance
    - Metrics fixtures: mock_metrics

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
"""

import hashlib
import json
import uuid
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.config import (
    CorruptionIndexMonitorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.corruption_index_monitor.models import (
    AlertSeverity,
    BriberySector,
    ComplianceLevel,
    CorrelationStrength,
    CountryClassification,
    CPIScore,
    DataSource,
    GovernanceRating,
    RiskLevel,
    TrendDirection,
    WGIDimension,
    WGIIndicator,
    BriberyRiskAssessment,
    InstitutionalQualityScore,
    TrendAnalysis,
    DeforestationCorrelation,
    Alert,
    ComplianceImpact,
    CountryProfile,
    AuditLogEntry,
)
from greenlang.agents.eudr.corruption_index_monitor.cpi_monitor_engine import (
    CPIMonitorEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.wgi_analyzer_engine import (
    WGIAnalyzerEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.bribery_risk_engine import (
    BriberyRiskEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.institutional_quality_engine import (
    InstitutionalQualityEngine,
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
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a CorruptionIndexMonitorConfig with test defaults.

    Provides a configuration instance suitable for testing with sensible
    defaults that mirror production but with test database URLs and
    reduced rate limits.
    """
    return CorruptionIndexMonitorConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        cpi_high_risk_threshold=30,
        cpi_moderate_threshold=50,
        cpi_low_risk_threshold=70,
        wgi_risk_threshold=-0.5,
        wgi_low_risk_threshold=0.5,
        trend_min_years=5,
        trend_trajectory_window=10,
        trend_prediction_horizon=3,
        trend_min_r_squared=0.3,
        trend_reversal_sensitivity=0.15,
        correlation_min_data_points=10,
        correlation_significance_level=0.05,
        correlation_min_coefficient=0.3,
        alert_cpi_change_threshold=5,
        alert_wgi_change_threshold=0.3,
        alert_trend_reversal=True,
        alert_cooldown_hours=24,
        alert_max_per_country_per_day=10,
        art29_low_risk_cpi=60,
        art29_low_risk_wgi=0.5,
        art29_high_risk_cpi=30,
        art29_high_risk_wgi=-0.5,
        enable_provenance=True,
        genesis_hash="GL-EUDR-CIM-019-TEST-GENESIS",
        enable_metrics=False,
        pool_size=5,
        batch_max_size=100,
        batch_concurrency=2,
        retention_years=5,
    )


@pytest.fixture(autouse=True)
def reset_singleton_config():
    """Reset the singleton config after each test to avoid cross-test leaks."""
    yield
    reset_config()


@pytest.fixture
def uuid_gen():
    """Create a deterministic UUID generator."""
    return DeterministicUUID()


# ---------------------------------------------------------------------------
# CPI Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cpi_data():
    """Sample CPI scores for EUDR-relevant test countries.

    Data reflects approximate real-world CPI ranges:
        BRA (Brazil): ~38 (high corruption risk)
        IDN (Indonesia): ~34 (high corruption risk)
        MYS (Malaysia): ~50 (moderate)
        COL (Colombia): ~39 (high)
        CMR (Cameroon): ~26 (critical)
        GHA (Ghana): ~43 (moderate)
        CIV (Cote d'Ivoire): ~36 (high)
        COD (DR Congo): ~20 (critical)
        DNK (Denmark): ~90 (very low risk)
        FIN (Finland): ~87 (very low risk)
    """
    return {
        "BR": {"score": Decimal("38"), "rank": 104, "region": "americas", "year": 2024},
        "ID": {"score": Decimal("34"), "rank": 115, "region": "asia_pacific", "year": 2024},
        "MY": {"score": Decimal("50"), "rank": 57, "region": "asia_pacific", "year": 2024},
        "CO": {"score": Decimal("39"), "rank": 91, "region": "americas", "year": 2024},
        "CM": {"score": Decimal("26"), "rank": 142, "region": "sub_saharan_africa", "year": 2024},
        "GH": {"score": Decimal("43"), "rank": 72, "region": "sub_saharan_africa", "year": 2024},
        "CI": {"score": Decimal("36"), "rank": 110, "region": "sub_saharan_africa", "year": 2024},
        "CD": {"score": Decimal("20"), "rank": 162, "region": "sub_saharan_africa", "year": 2024},
        "DK": {"score": Decimal("90"), "rank": 2, "region": "eu_western_europe", "year": 2024},
        "FI": {"score": Decimal("87"), "rank": 4, "region": "eu_western_europe", "year": 2024},
    }


@pytest.fixture
def sample_cpi_history():
    """Time series CPI data for Brazil (2015-2024) for trend analysis."""
    return [
        {"year": 2015, "score": Decimal("38"), "rank": 76},
        {"year": 2016, "score": Decimal("40"), "rank": 79},
        {"year": 2017, "score": Decimal("37"), "rank": 96},
        {"year": 2018, "score": Decimal("35"), "rank": 105},
        {"year": 2019, "score": Decimal("35"), "rank": 106},
        {"year": 2020, "score": Decimal("38"), "rank": 94},
        {"year": 2021, "score": Decimal("38"), "rank": 96},
        {"year": 2022, "score": Decimal("38"), "rank": 94},
        {"year": 2023, "score": Decimal("36"), "rank": 104},
        {"year": 2024, "score": Decimal("38"), "rank": 104},
    ]


# ---------------------------------------------------------------------------
# WGI Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wgi_data():
    """Sample WGI indicators for test countries across all 6 dimensions.

    WGI estimates range from -2.5 (weakest) to +2.5 (strongest).
    Percentile ranks range from 0 (lowest) to 100 (highest).
    """
    return {
        "BR": {
            WGIDimension.VOICE_ACCOUNTABILITY: {
                "estimate": Decimal("0.37"), "percentile": Decimal("58.9"),
            },
            WGIDimension.POLITICAL_STABILITY: {
                "estimate": Decimal("-0.43"), "percentile": Decimal("28.8"),
            },
            WGIDimension.GOVERNMENT_EFFECTIVENESS: {
                "estimate": Decimal("-0.26"), "percentile": Decimal("38.9"),
            },
            WGIDimension.REGULATORY_QUALITY: {
                "estimate": Decimal("-0.15"), "percentile": Decimal("42.3"),
            },
            WGIDimension.RULE_OF_LAW: {
                "estimate": Decimal("-0.34"), "percentile": Decimal("35.1"),
            },
            WGIDimension.CONTROL_OF_CORRUPTION: {
                "estimate": Decimal("-0.48"), "percentile": Decimal("31.7"),
            },
        },
        "DK": {
            WGIDimension.VOICE_ACCOUNTABILITY: {
                "estimate": Decimal("1.64"), "percentile": Decimal("98.1"),
            },
            WGIDimension.POLITICAL_STABILITY: {
                "estimate": Decimal("0.88"), "percentile": Decimal("77.4"),
            },
            WGIDimension.GOVERNMENT_EFFECTIVENESS: {
                "estimate": Decimal("1.94"), "percentile": Decimal("99.5"),
            },
            WGIDimension.REGULATORY_QUALITY: {
                "estimate": Decimal("1.72"), "percentile": Decimal("96.6"),
            },
            WGIDimension.RULE_OF_LAW: {
                "estimate": Decimal("1.93"), "percentile": Decimal("99.0"),
            },
            WGIDimension.CONTROL_OF_CORRUPTION: {
                "estimate": Decimal("2.25"), "percentile": Decimal("99.5"),
            },
        },
        "CD": {
            WGIDimension.VOICE_ACCOUNTABILITY: {
                "estimate": Decimal("-1.35"), "percentile": Decimal("6.3"),
            },
            WGIDimension.POLITICAL_STABILITY: {
                "estimate": Decimal("-2.15"), "percentile": Decimal("1.9"),
            },
            WGIDimension.GOVERNMENT_EFFECTIVENESS: {
                "estimate": Decimal("-1.62"), "percentile": Decimal("3.8"),
            },
            WGIDimension.REGULATORY_QUALITY: {
                "estimate": Decimal("-1.43"), "percentile": Decimal("4.3"),
            },
            WGIDimension.RULE_OF_LAW: {
                "estimate": Decimal("-1.68"), "percentile": Decimal("2.4"),
            },
            WGIDimension.CONTROL_OF_CORRUPTION: {
                "estimate": Decimal("-1.54"), "percentile": Decimal("3.4"),
            },
        },
    }


# ---------------------------------------------------------------------------
# Bribery Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bribery_data():
    """Sample TRACE bribery risk scores for test countries.

    TRACE scores: 1 (lowest risk) to 100 (highest risk).
    Includes all 4 TRACE domains:
        - BIG: Business Interactions with Government
        - ABDE: Anti-Bribery Deterrence & Enforcement
        - GCST: Government & Civil Service Transparency
        - CCSO: Capacity for Civil Society Oversight
    """
    return {
        "BR": {
            "composite_score": Decimal("55"),
            "domains": {
                "big": Decimal("62"),
                "abde": Decimal("48"),
                "gcst": Decimal("54"),
                "ccso": Decimal("56"),
            },
            "risk_level": "HIGH",
        },
        "ID": {
            "composite_score": Decimal("62"),
            "domains": {
                "big": Decimal("68"),
                "abde": Decimal("55"),
                "gcst": Decimal("63"),
                "ccso": Decimal("60"),
            },
            "risk_level": "HIGH",
        },
        "DK": {
            "composite_score": Decimal("8"),
            "domains": {
                "big": Decimal("10"),
                "abde": Decimal("5"),
                "gcst": Decimal("8"),
                "ccso": Decimal("7"),
            },
            "risk_level": "LOW",
        },
        "CD": {
            "composite_score": Decimal("82"),
            "domains": {
                "big": Decimal("88"),
                "abde": Decimal("79"),
                "gcst": Decimal("82"),
                "ccso": Decimal("78"),
            },
            "risk_level": "VERY_HIGH",
        },
        "CM": {
            "composite_score": Decimal("74"),
            "domains": {
                "big": Decimal("78"),
                "abde": Decimal("71"),
                "gcst": Decimal("76"),
                "ccso": Decimal("72"),
            },
            "risk_level": "HIGH",
        },
    }


# ---------------------------------------------------------------------------
# Institutional Quality Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_institutional_data():
    """Sample institutional quality scores for test countries.

    All dimensions scored 0-100 where 100 = strongest institutions.
    Dimensions: judicial_independence, regulatory_enforcement,
    forest_governance, law_enforcement_capacity.
    """
    return {
        "BR": {
            "overall_score": Decimal("42"),
            "judicial_independence": Decimal("48"),
            "regulatory_enforcement": Decimal("40"),
            "forest_governance": Decimal("38"),
            "law_enforcement_capacity": Decimal("42"),
            "governance_rating": GovernanceRating.C,
        },
        "DK": {
            "overall_score": Decimal("92"),
            "judicial_independence": Decimal("95"),
            "regulatory_enforcement": Decimal("90"),
            "forest_governance": Decimal("88"),
            "law_enforcement_capacity": Decimal("93"),
            "governance_rating": GovernanceRating.A,
        },
        "CD": {
            "overall_score": Decimal("15"),
            "judicial_independence": Decimal("12"),
            "regulatory_enforcement": Decimal("10"),
            "forest_governance": Decimal("18"),
            "law_enforcement_capacity": Decimal("20"),
            "governance_rating": GovernanceRating.F,
        },
        "GH": {
            "overall_score": Decimal("50"),
            "judicial_independence": Decimal("52"),
            "regulatory_enforcement": Decimal("48"),
            "forest_governance": Decimal("45"),
            "law_enforcement_capacity": Decimal("55"),
            "governance_rating": GovernanceRating.C,
        },
        "ID": {
            "overall_score": Decimal("38"),
            "judicial_independence": Decimal("35"),
            "regulatory_enforcement": Decimal("40"),
            "forest_governance": Decimal("32"),
            "law_enforcement_capacity": Decimal("44"),
            "governance_rating": GovernanceRating.D,
        },
    }


# ---------------------------------------------------------------------------
# Country Reference Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_country_codes():
    """List of EUDR-relevant countries for testing."""
    return ["BR", "ID", "MY", "CO", "CM", "GH", "CI", "CD", "DK", "FI"]


@pytest.fixture
def eudr_high_risk_countries():
    """Countries considered high risk for EUDR due diligence."""
    return ["BR", "ID", "CD", "CM", "CI"]


@pytest.fixture
def eudr_low_risk_countries():
    """Countries considered low risk for EUDR due diligence."""
    return ["DK", "FI"]


# ---------------------------------------------------------------------------
# Trend Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trend_data():
    """Time series data for trend analysis testing.

    Provides 10 years of CPI-like data for multiple countries with
    different trend directions:
        - BRA: stable/slight deterioration
        - DNK: stable high scores
        - IDN: gradual improvement
    """
    return {
        "BR": [
            (2015, Decimal("38")), (2016, Decimal("40")), (2017, Decimal("37")),
            (2018, Decimal("35")), (2019, Decimal("35")), (2020, Decimal("38")),
            (2021, Decimal("38")), (2022, Decimal("38")), (2023, Decimal("36")),
            (2024, Decimal("38")),
        ],
        "DK": [
            (2015, Decimal("91")), (2016, Decimal("90")), (2017, Decimal("88")),
            (2018, Decimal("88")), (2019, Decimal("87")), (2020, Decimal("88")),
            (2021, Decimal("88")), (2022, Decimal("90")), (2023, Decimal("90")),
            (2024, Decimal("90")),
        ],
        "ID": [
            (2015, Decimal("36")), (2016, Decimal("37")), (2017, Decimal("37")),
            (2018, Decimal("38")), (2019, Decimal("40")), (2020, Decimal("37")),
            (2021, Decimal("38")), (2022, Decimal("34")), (2023, Decimal("34")),
            (2024, Decimal("34")),
        ],
    }


# ---------------------------------------------------------------------------
# Correlation Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_correlation_data():
    """Paired corruption-deforestation data for correlation testing.

    Format: list of (cpi_score, deforestation_km2) pairs by year.
    Expects inverse relationship (lower CPI -> higher deforestation).
    """
    return {
        "BR": [
            (Decimal("38"), Decimal("12000")),
            (Decimal("40"), Decimal("10000")),
            (Decimal("37"), Decimal("13000")),
            (Decimal("35"), Decimal("14000")),
            (Decimal("35"), Decimal("14500")),
            (Decimal("38"), Decimal("11000")),
            (Decimal("38"), Decimal("11500")),
            (Decimal("38"), Decimal("13000")),
            (Decimal("36"), Decimal("14000")),
            (Decimal("38"), Decimal("12500")),
        ],
        "ID": [
            (Decimal("36"), Decimal("9000")),
            (Decimal("37"), Decimal("8500")),
            (Decimal("37"), Decimal("8800")),
            (Decimal("38"), Decimal("8200")),
            (Decimal("40"), Decimal("7500")),
            (Decimal("37"), Decimal("8600")),
            (Decimal("38"), Decimal("8000")),
            (Decimal("34"), Decimal("9500")),
            (Decimal("34"), Decimal("9800")),
            (Decimal("34"), Decimal("10000")),
        ],
    }


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cpi_engine(mock_config):
    """Create a CPIMonitorEngine instance with mocked config.

    The config is set as the global singleton before engine creation and
    reset after the test via the autouse reset_singleton_config fixture.
    """
    set_config(mock_config)
    return CPIMonitorEngine()


@pytest.fixture
def wgi_engine(mock_config):
    """Create a WGIAnalyzerEngine instance with mocked config."""
    set_config(mock_config)
    return WGIAnalyzerEngine()


@pytest.fixture
def bribery_engine(mock_config):
    """Create a BriberyRiskEngine instance with mocked config."""
    set_config(mock_config)
    return BriberyRiskEngine()


@pytest.fixture
def institutional_engine(mock_config):
    """Create an InstitutionalQualityEngine instance with mocked config."""
    set_config(mock_config)
    return InstitutionalQualityEngine()


# ---------------------------------------------------------------------------
# Mock Provenance Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provenance():
    """Create a mock ProvenanceTracker for testing.

    Provides a MagicMock that simulates provenance chain operations
    including record(), get_hash(), and verify_chain().
    """
    tracker = MagicMock()
    tracker.record.return_value = hashlib.sha256(b"test-provenance").hexdigest()
    tracker.get_hash.return_value = hashlib.sha256(b"test-provenance").hexdigest()
    tracker.verify_chain.return_value = True
    tracker.genesis_hash = "GL-EUDR-CIM-019-TEST-GENESIS"
    return tracker


# ---------------------------------------------------------------------------
# Mock Metrics Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metrics():
    """Create a mock MetricsCollector for testing.

    Provides a MagicMock that simulates Prometheus metrics operations
    including inc(), observe(), and set().
    """
    metrics = MagicMock()
    metrics.inc = MagicMock()
    metrics.observe = MagicMock()
    metrics.set = MagicMock()
    metrics.labels = MagicMock(return_value=metrics)
    return metrics


# ---------------------------------------------------------------------------
# Shared Constants
# ---------------------------------------------------------------------------

#: SHA-256 hash length in hexadecimal characters.
SHA256_HEX_LENGTH = 64

#: All CPI corruption risk levels.
CPI_RISK_LEVELS = ["very_low", "low", "moderate", "high", "very_high"]

#: All WGI dimensions.
WGI_ALL_DIMENSIONS = [
    WGIDimension.VOICE_ACCOUNTABILITY,
    WGIDimension.POLITICAL_STABILITY,
    WGIDimension.GOVERNMENT_EFFECTIVENESS,
    WGIDimension.REGULATORY_QUALITY,
    WGIDimension.RULE_OF_LAW,
    WGIDimension.CONTROL_OF_CORRUPTION,
]

#: All governance ratings.
ALL_GOVERNANCE_RATINGS = [
    GovernanceRating.A,
    GovernanceRating.B,
    GovernanceRating.C,
    GovernanceRating.D,
    GovernanceRating.F,
]

#: EUDR Article 29 country classification values.
COUNTRY_CLASSIFICATIONS = [
    CountryClassification.LOW,
    CountryClassification.STANDARD,
    CountryClassification.HIGH,
]

#: All bribery sectors relevant to EUDR.
ALL_BRIBERY_SECTORS = [
    BriberySector.FORESTRY,
    BriberySector.CUSTOMS,
    BriberySector.AGRICULTURE,
    BriberySector.MINING,
    BriberySector.EXTRACTION,
    BriberySector.JUDICIARY,
]

#: CPI score boundary values for risk classification.
CPI_BOUNDARY_VALUES = [0, 19, 20, 39, 40, 59, 60, 79, 80, 100]

#: WGI estimate boundary values.
WGI_BOUNDARY_VALUES = [
    Decimal("-2.5"), Decimal("-1.0"), Decimal("-0.5"),
    Decimal("0.0"), Decimal("0.5"), Decimal("1.0"), Decimal("2.5"),
]
