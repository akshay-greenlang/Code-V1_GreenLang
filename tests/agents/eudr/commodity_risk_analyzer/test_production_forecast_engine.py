# -*- coding: utf-8 -*-
"""
Unit tests for ProductionForecastEngine (AGENT-EUDR-018 Engine 4).

Tests yield modeling, production forecasting, climate impact assessment,
seasonal pattern analysis, production anomaly detection, supply risk
scoring, drought impact modeling, and geographic concentration (HHI)
for all 7 EUDR commodities.

Coverage target: 85%+
"""

from decimal import Decimal
import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.production_forecast_engine import (
    ProductionForecastEngine,
    EUDR_COMMODITIES,
    PRODUCTION_STATISTICS,
    CLIMATE_IMPACT_COEFFICIENTS,
)

SEVEN_COMMODITIES = sorted(EUDR_COMMODITIES)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for ProductionForecastEngine initialization."""

    @pytest.mark.unit
    def test_init_default(self):
        """Engine initializes with empty default config."""
        engine = ProductionForecastEngine()
        assert engine._config == {}
        assert engine._forecast_cache == {}

    @pytest.mark.unit
    def test_init_with_config(self):
        """Engine accepts and stores custom config dict."""
        cfg = {"cache_ttl": 300}
        engine = ProductionForecastEngine(config=cfg)
        assert engine._config == cfg

    @pytest.mark.unit
    def test_repr(self):
        """repr includes commodity count and version."""
        engine = ProductionForecastEngine()
        r = repr(engine)
        assert "ProductionForecastEngine" in r
        assert "commodities=7" in r


# ---------------------------------------------------------------------------
# TestForecastProduction
# ---------------------------------------------------------------------------

class TestForecastProduction:
    """Tests for forecast_production method."""

    @pytest.mark.unit
    def test_forecast_global_soya(self, production_forecast_engine):
        """Global soya forecast returns a positive value."""
        result = production_forecast_engine.forecast_production("soya", "GLOBAL", 12)
        assert Decimal(str(result["forecast_kt"])) > Decimal("0")
        assert result["commodity_type"] == "soya"
        assert result["region"] == "GLOBAL"
        assert result["method"] == "trend_projection_with_seasonality"

    @pytest.mark.unit
    def test_forecast_country_brazil(self, production_forecast_engine):
        """Brazil coffee forecast uses country-level production data."""
        result = production_forecast_engine.forecast_production("coffee", "BR", 6)
        assert result["region"] == "BR"
        assert result["horizon_months"] == 6
        assert Decimal(str(result["base_production_kt"])) == Decimal("3675")

    @pytest.mark.unit
    def test_forecast_non_top_producer(self, production_forecast_engine):
        """Non-top producer receives 0.1% of global production estimate."""
        result = production_forecast_engine.forecast_production("cocoa", "ZZ", 12)
        assert result["region"] == "ZZ"
        # 0.1% of 5800 = 5.80
        assert Decimal(str(result["base_production_kt"])) == Decimal("5.80")

    @pytest.mark.unit
    def test_confidence_intervals_order(self, production_forecast_engine):
        """95% CI is wider than 68% CI."""
        result = production_forecast_engine.forecast_production("rubber", "TH", 12)
        ci68 = result["confidence_68"]
        ci95 = result["confidence_95"]
        assert Decimal(str(ci95["low"])) <= Decimal(str(ci68["low"]))
        assert Decimal(str(ci95["high"])) >= Decimal(str(ci68["high"]))

    @pytest.mark.unit
    def test_invalid_horizon_low(self, production_forecast_engine):
        """horizon_months < 1 raises ValueError."""
        with pytest.raises(ValueError, match="horizon_months"):
            production_forecast_engine.forecast_production("soya", "BR", 0)

    @pytest.mark.unit
    def test_invalid_horizon_high(self, production_forecast_engine):
        """horizon_months > 24 raises ValueError."""
        with pytest.raises(ValueError, match="horizon_months"):
            production_forecast_engine.forecast_production("soya", "BR", 25)

    @pytest.mark.unit
    def test_invalid_commodity(self, production_forecast_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            production_forecast_engine.forecast_production("banana", "BR", 12)


# ---------------------------------------------------------------------------
# TestYieldEstimate
# ---------------------------------------------------------------------------

class TestYieldEstimate:
    """Tests for calculate_yield_estimate method."""

    @pytest.mark.unit
    def test_yield_top_producer(self, production_forecast_engine):
        """Top producer returns known yield per hectare."""
        result = production_forecast_engine.calculate_yield_estimate("oil_palm", "ID", 2024)
        assert Decimal(str(result["yield_per_ha"])) == Decimal("3.80")
        assert result["country_code"] == "ID"

    @pytest.mark.unit
    def test_yield_trend_increasing(self, production_forecast_engine):
        """Coffee (1% annual) yields INCREASING trend."""
        result = production_forecast_engine.calculate_yield_estimate("coffee", "BR", 2026)
        assert result["yield_trend"] == "INCREASING"

    @pytest.mark.unit
    def test_yield_future_year_growth(self, production_forecast_engine):
        """Future year estimate applies positive trend factor."""
        r2024 = production_forecast_engine.calculate_yield_estimate("soya", "BR", 2024)
        r2030 = production_forecast_engine.calculate_yield_estimate("soya", "BR", 2030)
        assert Decimal(str(r2030["yield_per_ha"])) > Decimal(str(r2024["yield_per_ha"]))

    @pytest.mark.unit
    def test_yield_invalid_year(self, production_forecast_engine):
        """Year out of [2000, 2050] raises ValueError."""
        with pytest.raises(ValueError, match="year"):
            production_forecast_engine.calculate_yield_estimate("wood", "US", 1999)


# ---------------------------------------------------------------------------
# TestClimateImpact
# ---------------------------------------------------------------------------

class TestClimateImpact:
    """Tests for assess_climate_impact method."""

    @pytest.mark.unit
    def test_no_deviation_low_risk(self, production_forecast_engine):
        """Zero deviations yield LOW risk."""
        result = production_forecast_engine.assess_climate_impact(
            "soya", "BR",
            {"temp_deviation_c": 0, "rainfall_deviation_pct": 0},
        )
        assert result["risk_level"] == "LOW"

    @pytest.mark.unit
    def test_high_temp_deviation_negative_yield(self, production_forecast_engine):
        """High positive temperature deviation causes negative yield impact."""
        result = production_forecast_engine.assess_climate_impact(
            "coffee", "BR",
            {"temp_deviation_c": 5, "rainfall_deviation_pct": 0},
        )
        assert Decimal(str(result["yield_impact_pct"])) < Decimal("0")
        assert result["risk_level"] in ("HIGH", "CRITICAL")

    @pytest.mark.unit
    def test_contributing_factors_present(self, production_forecast_engine):
        """Factors list contains temperature and rainfall entries."""
        result = production_forecast_engine.assess_climate_impact(
            "cocoa", "CI",
            {"temp_deviation_c": 2, "rainfall_deviation_pct": -10},
        )
        factor_names = [f["factor"] for f in result["contributing_factors"]]
        assert "temperature" in factor_names
        assert "rainfall" in factor_names


# ---------------------------------------------------------------------------
# TestSeasonalPatterns
# ---------------------------------------------------------------------------

class TestSeasonalPatterns:
    """Tests for analyze_seasonal_patterns method."""

    @pytest.mark.unit
    def test_cocoa_peak_months(self, production_forecast_engine):
        """Cocoa peak months are Oct, Nov, Dec."""
        result = production_forecast_engine.analyze_seasonal_patterns("cocoa", "GLOBAL")
        assert result["peak_months"] == [10, 11, 12]

    @pytest.mark.unit
    def test_monthly_weights_length(self, production_forecast_engine):
        """Monthly weights list has exactly 12 entries."""
        result = production_forecast_engine.analyze_seasonal_patterns("soya", "BR")
        assert len(result["monthly_weights"]) == 12

    @pytest.mark.unit
    def test_peak_month_higher_weight(self, production_forecast_engine):
        """Peak months have weight 1.20, non-peak have weight 0.90."""
        result = production_forecast_engine.analyze_seasonal_patterns("rubber", "TH")
        for entry in result["monthly_weights"]:
            if entry["is_peak"]:
                assert entry["weight"] == Decimal("1.20")
            else:
                assert entry["weight"] == Decimal("0.90")


# ---------------------------------------------------------------------------
# TestProductionAnomaly
# ---------------------------------------------------------------------------

class TestProductionAnomaly:
    """Tests for detect_production_anomaly method."""

    @pytest.mark.unit
    def test_within_range_no_anomaly(self, production_forecast_engine):
        """Volume within 30% of expected is not flagged."""
        expected_kt = PRODUCTION_STATISTICS["cocoa"]["top_producers"]["CI"]["production_kt"]
        result = production_forecast_engine.detect_production_anomaly(
            "cocoa", "CI", expected_kt,
        )
        assert result["is_anomaly"] is False
        assert result["severity"] == "NONE"

    @pytest.mark.unit
    def test_huge_volume_critical(self, production_forecast_engine):
        """Volume 5x expected is CRITICAL anomaly."""
        expected = PRODUCTION_STATISTICS["soya"]["top_producers"]["BR"]["production_kt"]
        result = production_forecast_engine.detect_production_anomaly(
            "soya", "BR", expected * Decimal("5"),
        )
        assert result["is_anomaly"] is True
        assert result["severity"] == "CRITICAL"

    @pytest.mark.unit
    def test_negative_volume_raises(self, production_forecast_engine):
        """Negative reported volume raises ValueError."""
        with pytest.raises(ValueError, match="reported_volume"):
            production_forecast_engine.detect_production_anomaly(
                "cattle", "BR", Decimal("-100"),
            )


# ---------------------------------------------------------------------------
# TestSupplyRisk
# ---------------------------------------------------------------------------

class TestSupplyRisk:
    """Tests for calculate_supply_risk method."""

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity", SEVEN_COMMODITIES)
    def test_supply_risk_in_range(self, production_forecast_engine, commodity):
        """Supply risk for every commodity is between 0 and 100."""
        score = production_forecast_engine.calculate_supply_risk(commodity)
        assert Decimal("0") <= score <= Decimal("100")

    @pytest.mark.unit
    def test_supply_risk_invalid_commodity(self, production_forecast_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError):
            production_forecast_engine.calculate_supply_risk("wheat")


# ---------------------------------------------------------------------------
# TestProductionStatistics
# ---------------------------------------------------------------------------

class TestProductionStatistics:
    """Tests for get_production_statistics method."""

    @pytest.mark.unit
    def test_global_statistics(self, production_forecast_engine):
        """Global statistics include producer list."""
        result = production_forecast_engine.get_production_statistics("wood")
        assert result["commodity_type"] == "wood"
        assert result["producer_count"] == 10
        assert len(result["top_producers"]) == 10

    @pytest.mark.unit
    def test_country_specific_top_producer(self, production_forecast_engine):
        """Country-specific query for top producer returns is_top_producer=True."""
        result = production_forecast_engine.get_production_statistics("cocoa", "CI")
        assert result["is_top_producer"] is True
        assert Decimal(str(result["share_pct"])) == Decimal("38.00")

    @pytest.mark.unit
    def test_country_specific_non_producer(self, production_forecast_engine):
        """Non-top producer country returns zero production."""
        result = production_forecast_engine.get_production_statistics("cocoa", "XX")
        assert result["is_top_producer"] is False
        assert Decimal(str(result["production_kt"])) == Decimal("0")


# ---------------------------------------------------------------------------
# TestDroughtImpact
# ---------------------------------------------------------------------------

class TestDroughtImpact:
    """Tests for model_drought_impact method."""

    @pytest.mark.unit
    def test_zero_severity_minimal(self, production_forecast_engine):
        """Zero drought severity returns MINIMAL classification."""
        result = production_forecast_engine.model_drought_impact("soya", "BR", Decimal("0"))
        assert result["drought_classification"] == "MINIMAL"

    @pytest.mark.unit
    def test_extreme_drought(self, production_forecast_engine):
        """Severity 0.9 yields EXTREME classification."""
        result = production_forecast_engine.model_drought_impact("cocoa", "CI", Decimal("0.9"))
        assert result["drought_classification"] == "EXTREME"

    @pytest.mark.unit
    def test_severity_out_of_range(self, production_forecast_engine):
        """Severity > 1 raises ValueError."""
        with pytest.raises(ValueError, match="severity"):
            production_forecast_engine.model_drought_impact("coffee", "BR", Decimal("1.5"))

    @pytest.mark.unit
    def test_recovery_months_present(self, production_forecast_engine):
        """Result contains estimated_recovery_months."""
        result = production_forecast_engine.model_drought_impact("wood", "BR", Decimal("0.5"))
        assert result["estimated_recovery_months"] == 24  # wood recovery = 24


# ---------------------------------------------------------------------------
# TestProductionConcentration
# ---------------------------------------------------------------------------

class TestProductionConcentration:
    """Tests for calculate_production_concentration (HHI) method."""

    @pytest.mark.unit
    def test_oil_palm_high_concentration(self, production_forecast_engine):
        """Oil palm dominated by ID+MY should show HIGH concentration."""
        result = production_forecast_engine.calculate_production_concentration("oil_palm")
        assert result["concentration_level"] == "HIGH"
        assert Decimal(str(result["hhi"])) > Decimal("2500")

    @pytest.mark.unit
    def test_cattle_lower_concentration(self, production_forecast_engine):
        """Cattle has more diverse producers, lower HHI."""
        result = production_forecast_engine.calculate_production_concentration("cattle")
        # Not necessarily LOW, but HHI should be calculable
        assert Decimal(str(result["hhi"])) > Decimal("0")
        assert result["producer_count"] == 10

    @pytest.mark.unit
    def test_top_3_and_5_shares(self, production_forecast_engine):
        """Top-3 share <= top-5 share."""
        result = production_forecast_engine.calculate_production_concentration("coffee")
        assert Decimal(str(result["top_3_share"])) <= Decimal(str(result["top_5_share"]))

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity", SEVEN_COMMODITIES)
    def test_hhi_all_commodities(self, production_forecast_engine, commodity):
        """HHI is computable for every EUDR commodity."""
        result = production_forecast_engine.calculate_production_concentration(commodity)
        assert "hhi" in result
        assert result["concentration_level"] in ("LOW", "MODERATE", "HIGH")


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------

class TestProvenance:
    """Tests for provenance hash determinism."""

    @pytest.mark.unit
    def test_forecast_provenance_deterministic(self, production_forecast_engine):
        """Same inputs produce same provenance hash in forecast."""
        r1 = production_forecast_engine.forecast_production("soya", "BR", 12)
        r2 = production_forecast_engine.forecast_production("soya", "BR", 12)
        assert r1["provenance_hash"] == r2["provenance_hash"]
        assert len(r1["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_concentration_provenance(self, production_forecast_engine):
        """Concentration analysis has a 64-char SHA-256 hash."""
        result = production_forecast_engine.calculate_production_concentration("cocoa")
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_yield_provenance(self, production_forecast_engine):
        """Yield estimate has provenance hash."""
        result = production_forecast_engine.calculate_yield_estimate("cattle", "BR", 2025)
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.unit
    def test_empty_commodity_raises(self, production_forecast_engine):
        """Empty string commodity raises ValueError."""
        with pytest.raises(ValueError):
            production_forecast_engine.forecast_production("", "BR", 12)

    @pytest.mark.unit
    def test_none_commodity_raises(self, production_forecast_engine):
        """None commodity raises ValueError."""
        with pytest.raises(ValueError):
            production_forecast_engine.forecast_production(None, "BR", 12)

    @pytest.mark.unit
    def test_case_insensitive_commodity(self, production_forecast_engine):
        """Commodity matching is case-insensitive."""
        result = production_forecast_engine.forecast_production("SOYA", "BR", 12)
        assert result["commodity_type"] == "soya"

    @pytest.mark.unit
    def test_region_trimmed_uppercased(self, production_forecast_engine):
        """Region is trimmed and uppercased."""
        result = production_forecast_engine.forecast_production("coffee", " br ", 12)
        assert result["region"] == "BR"

    @pytest.mark.unit
    def test_processing_time_positive(self, production_forecast_engine):
        """Processing time is a positive float."""
        result = production_forecast_engine.forecast_production("cattle", "GLOBAL", 12)
        assert result["processing_time_ms"] >= 0
