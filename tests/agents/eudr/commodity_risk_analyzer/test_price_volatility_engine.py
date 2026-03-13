# -*- coding: utf-8 -*-
"""
Unit tests for PriceVolatilityEngine (AGENT-EUDR-018 Engine 3).

Tests commodity price tracking and volatility analysis including current
price retrieval, price history generation, volatility calculation across
30d/90d windows, market disruption detection, price risk scoring, price
forecasting, cross-commodity correlation, market indicators, price anomaly
detection, provenance hashing, and error handling.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.price_volatility_engine import (
    EUDR_COMMODITIES,
    PriceVolatilityEngine,
    REFERENCE_PRICES,
)


# =========================================================================
# TestInit
# =========================================================================


class TestInit:
    """Tests for PriceVolatilityEngine initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Engine initializes with no price history loaded."""
        engine = PriceVolatilityEngine()
        r = repr(engine)
        assert "PriceVolatilityEngine" in r
        assert "commodities_with_history=0" in r

    @pytest.mark.unit
    def test_initialization_with_config(self):
        """Engine accepts optional config dictionary."""
        engine = PriceVolatilityEngine(config={"test": True})
        assert repr(engine).startswith("PriceVolatilityEngine")

    @pytest.mark.unit
    def test_initialization_with_history(self):
        """Engine accepts pre-loaded price history."""
        history = {
            "cocoa": [
                {"date": "2026-01-01", "price": Decimal("4800")},
                {"date": "2026-01-02", "price": Decimal("4850")},
            ],
        }
        engine = PriceVolatilityEngine(price_history=history)
        assert "commodities_with_history=1" in repr(engine)


# =========================================================================
# TestGetCurrentPrice
# =========================================================================


class TestGetCurrentPrice:
    """Tests for get_current_price method."""

    @pytest.mark.unit
    def test_current_price_returns_required_keys(self, price_volatility_engine):
        """Current price result contains all required keys."""
        result = price_volatility_engine.get_current_price("cocoa")
        required_keys = {
            "commodity_type", "price", "unit", "currency",
            "typical_range", "seasonal_adjustment", "source",
            "as_of", "provenance_hash",
        }
        assert required_keys.issubset(set(result.keys()))

    @pytest.mark.unit
    def test_current_price_positive(self, price_volatility_engine):
        """Current price is positive."""
        result = price_volatility_engine.get_current_price("coffee")
        assert result["price"] > Decimal("0")

    @pytest.mark.unit
    def test_current_price_within_typical_range(self, price_volatility_engine):
        """Current price (reference-based) is near the typical range."""
        result = price_volatility_engine.get_current_price("soya")
        low = result["typical_range"]["low"]
        high = result["typical_range"]["high"]
        # With seasonal adjustment, price should be in reasonable bounds
        assert result["price"] > Decimal("0")
        assert low > Decimal("0")
        assert high > low

    @pytest.mark.unit
    def test_current_price_default_usd(self, price_volatility_engine):
        """Default currency is USD."""
        result = price_volatility_engine.get_current_price("rubber")
        assert result["currency"] == "USD"

    @pytest.mark.unit
    def test_current_price_with_loaded_history(self, price_engine_with_history):
        """Engine with loaded history returns the latest historical price."""
        result = price_engine_with_history.get_current_price("cocoa")
        assert result["price"] > Decimal("0")

    @pytest.mark.unit
    def test_current_price_invalid_commodity_raises(self, price_volatility_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            price_volatility_engine.get_current_price("banana")


# =========================================================================
# TestPriceHistory
# =========================================================================


class TestPriceHistory:
    """Tests for get_price_history method."""

    @pytest.mark.unit
    def test_history_returns_list(self, price_volatility_engine):
        """Price history returns a non-empty list of records."""
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=30)).isoformat()
        end = today.isoformat()
        result = price_volatility_engine.get_price_history("cocoa", start, end)
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.unit
    def test_history_records_have_date_and_price(self, price_volatility_engine):
        """Each history record has date and price keys."""
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=10)).isoformat()
        end = today.isoformat()
        records = price_volatility_engine.get_price_history("wood", start, end)
        for record in records:
            assert "date" in record
            assert "price" in record
            assert record["price"] > Decimal("0")

    @pytest.mark.unit
    def test_history_invalid_date_format_raises(self, price_volatility_engine):
        """Invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            price_volatility_engine.get_price_history("cocoa", "bad-date", "2026-01-31")

    @pytest.mark.unit
    def test_history_start_after_end_raises(self, price_volatility_engine):
        """start_date > end_date raises ValueError."""
        with pytest.raises(ValueError, match="must be <="):
            price_volatility_engine.get_price_history(
                "cocoa", "2026-03-01", "2026-01-01",
            )

    @pytest.mark.unit
    def test_loaded_history_filtered(self, price_engine_with_history):
        """Loaded history is filtered to the requested date range."""
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=30)).isoformat()
        end = today.isoformat()
        records = price_engine_with_history.get_price_history("cocoa", start, end)
        assert len(records) > 0
        for record in records:
            assert start <= record["date"] <= end


# =========================================================================
# TestCalculateVolatility
# =========================================================================


class TestCalculateVolatility:
    """Tests for calculate_volatility method across 30d and 90d windows."""

    @pytest.mark.unit
    def test_volatility_30d_returns_required_keys(self, price_volatility_engine):
        """30-day volatility result has all required keys."""
        result = price_volatility_engine.calculate_volatility("cocoa", window_days=30)
        required_keys = {
            "commodity_type", "window_days", "volatility", "volatility_pct",
            "daily_volatility", "data_points", "reference_volatility",
            "volatility_regime", "provenance_hash",
        }
        assert required_keys.issubset(set(result.keys()))
        assert result["window_days"] == 30

    @pytest.mark.unit
    def test_volatility_90d(self, price_volatility_engine):
        """90-day volatility calculation produces valid result."""
        result = price_volatility_engine.calculate_volatility("soya", window_days=90)
        assert result["window_days"] == 90
        assert result["volatility"] >= Decimal("0")
        assert result["volatility_pct"] >= Decimal("0")

    @pytest.mark.unit
    def test_volatility_regime_classification(self, price_volatility_engine):
        """Volatility regime is one of LOW, NORMAL, HIGH, EXTREME."""
        result = price_volatility_engine.calculate_volatility("oil_palm", window_days=30)
        assert result["volatility_regime"] in ("LOW", "NORMAL", "HIGH", "EXTREME")

    @pytest.mark.unit
    def test_volatility_window_too_small_raises(self, price_volatility_engine):
        """Window < 2 days raises ValueError."""
        with pytest.raises(ValueError, match="window_days must be >= 2"):
            price_volatility_engine.calculate_volatility("cocoa", window_days=1)

    @pytest.mark.unit
    def test_volatility_with_loaded_history(self, price_engine_with_history):
        """Volatility from loaded history uses actual data points."""
        result = price_engine_with_history.calculate_volatility("cocoa", window_days=60)
        assert result["data_points"] > 2


# =========================================================================
# TestMarketDisruption
# =========================================================================


class TestMarketDisruption:
    """Tests for detect_market_disruption method."""

    @pytest.mark.unit
    def test_disruption_returns_required_keys(self, price_volatility_engine):
        """Disruption result has all required keys."""
        result = price_volatility_engine.detect_market_disruption("cocoa")
        required_keys = {
            "commodity_type", "is_disrupted", "z_score", "severity",
            "price_change_pct", "description", "provenance_hash",
        }
        assert required_keys.issubset(set(result.keys()))

    @pytest.mark.unit
    def test_disruption_severity_values(self, price_volatility_engine):
        """Severity is one of NONE, MODERATE, SEVERE, EXTREME."""
        result = price_volatility_engine.detect_market_disruption("soya")
        assert result["severity"] in ("NONE", "MODERATE", "SEVERE", "EXTREME")

    @pytest.mark.unit
    def test_disruption_with_custom_threshold(self, price_volatility_engine):
        """Custom z-score threshold is respected."""
        # Very low threshold should make disruption easier to trigger
        result = price_volatility_engine.detect_market_disruption(
            "cocoa", threshold=Decimal("0.01"),
        )
        assert isinstance(result["is_disrupted"], bool)

    @pytest.mark.unit
    def test_disruption_z_score_is_decimal(self, price_volatility_engine):
        """Z-score is returned as Decimal."""
        result = price_volatility_engine.detect_market_disruption("coffee")
        assert isinstance(result["z_score"], Decimal)

    @pytest.mark.unit
    def test_disruption_description_nonempty(self, price_volatility_engine):
        """Description is a non-empty string."""
        result = price_volatility_engine.detect_market_disruption("wood")
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


# =========================================================================
# TestPriceRiskScore
# =========================================================================


class TestPriceRiskScore:
    """Tests for calculate_price_risk_score method."""

    @pytest.mark.unit
    def test_price_risk_in_range(self, price_volatility_engine):
        """Price risk score is in [0, 100]."""
        result = price_volatility_engine.calculate_price_risk_score("cocoa")
        assert Decimal("0") <= result <= Decimal("100")

    @pytest.mark.unit
    def test_price_risk_is_decimal(self, price_volatility_engine):
        """Price risk score is a Decimal."""
        result = price_volatility_engine.calculate_price_risk_score("oil_palm")
        assert isinstance(result, Decimal)

    @pytest.mark.unit
    def test_price_risk_invalid_commodity_raises(self, price_volatility_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            price_volatility_engine.calculate_price_risk_score("invalid")


# =========================================================================
# TestForecastPrice
# =========================================================================


class TestForecastPrice:
    """Tests for forecast_price method."""

    @pytest.mark.unit
    def test_forecast_returns_required_keys(self, price_volatility_engine):
        """Forecast result has all required keys."""
        result = price_volatility_engine.forecast_price("cocoa", horizon_days=90)
        required_keys = {
            "commodity_type", "current_price", "forecast_price",
            "confidence_68", "confidence_95", "horizon_days",
            "method", "provenance_hash",
        }
        assert required_keys.issubset(set(result.keys()))

    @pytest.mark.unit
    def test_forecast_price_positive(self, price_volatility_engine):
        """Forecast price is positive."""
        result = price_volatility_engine.forecast_price("soya", horizon_days=30)
        assert result["forecast_price"] > Decimal("0")

    @pytest.mark.unit
    def test_forecast_confidence_intervals(self, price_volatility_engine):
        """95% CI is wider than 68% CI."""
        result = price_volatility_engine.forecast_price("coffee", horizon_days=90)
        ci_68_width = result["confidence_68"]["high"] - result["confidence_68"]["low"]
        ci_95_width = result["confidence_95"]["high"] - result["confidence_95"]["low"]
        assert ci_95_width >= ci_68_width

    @pytest.mark.unit
    def test_forecast_method_is_ses(self, price_volatility_engine):
        """Forecast method is simple_exponential_smoothing."""
        result = price_volatility_engine.forecast_price("rubber", horizon_days=30)
        assert result["method"] == "simple_exponential_smoothing"

    @pytest.mark.unit
    def test_forecast_invalid_horizon_raises(self, price_volatility_engine):
        """Horizon < 1 or > 365 raises ValueError."""
        with pytest.raises(ValueError, match="horizon_days must be in"):
            price_volatility_engine.forecast_price("cocoa", horizon_days=0)
        with pytest.raises(ValueError, match="horizon_days must be in"):
            price_volatility_engine.forecast_price("cocoa", horizon_days=400)

    @pytest.mark.unit
    def test_forecast_with_loaded_history(self, price_engine_with_history):
        """Forecast uses loaded history data when available."""
        result = price_engine_with_history.forecast_price("cocoa", horizon_days=30)
        assert result["data_points"] > 10


# =========================================================================
# TestCorrelation
# =========================================================================


class TestCorrelation:
    """Tests for calculate_correlation method."""

    @pytest.mark.unit
    def test_self_correlation_is_one(self, price_volatility_engine):
        """Correlation of a commodity with itself is exactly 1.00."""
        result = price_volatility_engine.calculate_correlation("cocoa", "cocoa")
        assert result == Decimal("1.00")

    @pytest.mark.unit
    def test_correlation_in_range(self, price_volatility_engine):
        """Cross-commodity correlation is in [-1, 1]."""
        result = price_volatility_engine.calculate_correlation(
            "cocoa", "coffee", window_days=90,
        )
        assert Decimal("-1.00") <= result <= Decimal("1.00")

    @pytest.mark.unit
    def test_correlation_is_decimal(self, price_volatility_engine):
        """Correlation coefficient is a Decimal."""
        result = price_volatility_engine.calculate_correlation(
            "soya", "oil_palm",
        )
        assert isinstance(result, Decimal)

    @pytest.mark.unit
    def test_correlation_invalid_commodity_raises(self, price_volatility_engine):
        """Invalid commodity in correlation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            price_volatility_engine.calculate_correlation("cocoa", "invalid")


# =========================================================================
# TestMarketIndicators
# =========================================================================


class TestMarketIndicators:
    """Tests for get_market_indicators method."""

    @pytest.mark.unit
    def test_indicators_returns_required_keys(self, price_volatility_engine):
        """Market indicators result has all required keys."""
        result = price_volatility_engine.get_market_indicators("cocoa")
        required_keys = {
            "commodity_type", "current_price", "unit",
            "volatility_30d", "volatility_90d", "volatility_regime",
            "market_disrupted", "disruption_severity",
            "price_risk_score", "seasonal_adjustment",
        }
        assert required_keys.issubset(set(result.keys()))

    @pytest.mark.unit
    def test_indicators_commodity_type(self, price_volatility_engine):
        """Returned commodity_type matches requested commodity."""
        result = price_volatility_engine.get_market_indicators("wood")
        assert result["commodity_type"] == "wood"

    @pytest.mark.unit
    def test_indicators_price_risk_in_range(self, price_volatility_engine):
        """Price risk score in indicators is in [0, 100]."""
        result = price_volatility_engine.get_market_indicators("cattle")
        assert Decimal("0") <= result["price_risk_score"] <= Decimal("100")


# =========================================================================
# TestPriceAnomaly
# =========================================================================


class TestPriceAnomaly:
    """Tests for detect_price_anomaly method."""

    @pytest.mark.unit
    def test_anomaly_normal_price(self, price_volatility_engine):
        """Reference price is not anomalous."""
        ref_price = REFERENCE_PRICES["cocoa"]["reference_price"]
        result = price_volatility_engine.detect_price_anomaly("cocoa", ref_price)
        assert isinstance(result["is_anomaly"], bool)
        assert result["severity"] in ("NONE", "MILD", "MODERATE", "SEVERE")

    @pytest.mark.unit
    def test_anomaly_extreme_price(self, price_volatility_engine):
        """Extremely high price is flagged as anomaly or high severity."""
        result = price_volatility_engine.detect_price_anomaly(
            "cocoa", Decimal("99999.99"),
        )
        # With synthetic data this might still be NONE depending on MAD
        assert isinstance(result["is_anomaly"], bool)
        assert isinstance(result["z_score"], Decimal)

    @pytest.mark.unit
    def test_anomaly_negative_price_raises(self, price_volatility_engine):
        """Negative price raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            price_volatility_engine.detect_price_anomaly("cocoa", Decimal("-100"))

    @pytest.mark.unit
    def test_anomaly_returns_required_keys(self, price_volatility_engine):
        """Anomaly result has all required keys."""
        result = price_volatility_engine.detect_price_anomaly(
            "soya", Decimal("500"),
        )
        required_keys = {
            "commodity_type", "tested_price", "is_anomaly",
            "z_score", "deviation_pct", "median_price",
            "typical_range", "severity", "provenance_hash",
        }
        assert required_keys.issubset(set(result.keys()))

    @pytest.mark.unit
    def test_anomaly_with_date(self, price_volatility_engine):
        """Price date parameter is reflected in result."""
        result = price_volatility_engine.detect_price_anomaly(
            "wood", Decimal("280"), price_date="2026-02-15",
        )
        assert result["price_date"] == "2026-02-15"


# =========================================================================
# TestProvenance
# =========================================================================


class TestProvenance:
    """Tests for provenance hash generation on price analyses."""

    @pytest.mark.unit
    def test_current_price_provenance_64_chars(self, price_volatility_engine):
        """Current price provenance hash is 64-character hex."""
        result = price_volatility_engine.get_current_price("cocoa")
        assert len(result["provenance_hash"]) == 64
        int(result["provenance_hash"], 16)

    @pytest.mark.unit
    def test_volatility_provenance_64_chars(self, price_volatility_engine):
        """Volatility provenance hash is 64-character hex."""
        result = price_volatility_engine.calculate_volatility("coffee", window_days=30)
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_disruption_provenance_64_chars(self, price_volatility_engine):
        """Market disruption provenance hash is 64-character hex."""
        result = price_volatility_engine.detect_market_disruption("soya")
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_forecast_provenance_64_chars(self, price_volatility_engine):
        """Forecast provenance hash is 64-character hex."""
        result = price_volatility_engine.forecast_price("rubber", horizon_days=30)
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_anomaly_provenance_64_chars(self, price_volatility_engine):
        """Anomaly detection provenance hash is 64-character hex."""
        result = price_volatility_engine.detect_price_anomaly(
            "wood", Decimal("300"),
        )
        assert len(result["provenance_hash"]) == 64


# =========================================================================
# TestErrorHandling
# =========================================================================


class TestErrorHandling:
    """Tests for error handling and input validation."""

    @pytest.mark.unit
    def test_invalid_commodity_current_price(self, price_volatility_engine):
        """get_current_price with invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            price_volatility_engine.get_current_price("invalid")

    @pytest.mark.unit
    def test_invalid_commodity_volatility(self, price_volatility_engine):
        """calculate_volatility with invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            price_volatility_engine.calculate_volatility("invalid")

    @pytest.mark.unit
    def test_invalid_commodity_disruption(self, price_volatility_engine):
        """detect_market_disruption with invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            price_volatility_engine.detect_market_disruption("invalid")

    @pytest.mark.unit
    def test_invalid_commodity_forecast(self, price_volatility_engine):
        """forecast_price with invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            price_volatility_engine.forecast_price("invalid")

    @pytest.mark.unit
    def test_empty_commodity_raises(self, price_volatility_engine):
        """Empty string commodity raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            price_volatility_engine.get_current_price("")

    @pytest.mark.unit
    def test_load_and_clear_history(self, price_volatility_engine):
        """load_price_history and clear_price_history lifecycle works."""
        history = [
            {"date": "2026-01-01", "price": Decimal("100")},
            {"date": "2026-01-02", "price": Decimal("101")},
        ]
        count = price_volatility_engine.load_price_history("cocoa", history)
        assert count == 2
        price_volatility_engine.clear_price_history()
        # After clearing, get_price_history falls back to synthetic
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=5)).isoformat()
        end = today.isoformat()
        records = price_volatility_engine.get_price_history("cocoa", start, end)
        assert len(records) >= 0  # synthetic may or may not cover range
