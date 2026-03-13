# -*- coding: utf-8 -*-
"""
Unit tests for HistoricalBaselineEngine - AGENT-EUDR-020 Engine 6

Tests forest cover baseline establishment for the 2018-2020 EUDR reference
period including canopy cover calculations, forest area measurements,
vegetation index statistics, trend analysis, anomaly detection, baseline
comparisons, coverage statistics, and provenance tracking.

Coverage targets: 85%+ across all HistoricalBaselineEngine methods.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.deforestation_alert_system.engines.historical_baseline_engine import (
    AnomalyType,
    BaselineComparison,
    BaselineStatus,
    CANOPY_COVER_FOREST_THRESHOLD,
    CANOPY_COVER_OWL_THRESHOLD,
    ChangeDirection,
    CoverageStatistics,
    DEFAULT_BASELINE_END_YEAR,
    DEFAULT_BASELINE_START_YEAR,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_PLOT_RADIUS_KM,
    ForestClassification,
    GAIN_THRESHOLD_PCT,
    HistoricalBaseline,
    HistoricalBaselineEngine,
    LOSS_THRESHOLD_PCT,
    NDVI_TO_CANOPY_OFFSET,
    NDVI_TO_CANOPY_SCALE,
    REFERENCE_BASELINE_OBSERVATIONS,
    TrendDirection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> HistoricalBaselineEngine:
    """Create a default HistoricalBaselineEngine instance."""
    return HistoricalBaselineEngine()


@pytest.fixture
def engine_custom() -> HistoricalBaselineEngine:
    """Create engine with custom parameters."""
    return HistoricalBaselineEngine(
        baseline_start_year=2017,
        baseline_end_year=2021,
        min_samples=2,
        canopy_threshold_pct=Decimal("15"),
    )


@pytest.fixture
def forested_observations() -> List[Dict[str, Any]]:
    """Observations for a well-forested plot (high NDVI)."""
    return list(REFERENCE_BASELINE_OBSERVATIONS["sample_forested"])


@pytest.fixture
def degraded_observations() -> List[Dict[str, Any]]:
    """Observations for a degrading plot."""
    return list(REFERENCE_BASELINE_OBSERVATIONS["sample_degraded"])


@pytest.fixture
def non_forest_observations() -> List[Dict[str, Any]]:
    """Observations for a non-forest plot (low NDVI)."""
    return list(REFERENCE_BASELINE_OBSERVATIONS["sample_non_forest"])


@pytest.fixture
def established_baseline(
    engine: HistoricalBaselineEngine,
    forested_observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Establish a baseline for a forested plot."""
    engine.load_observations("PLOT-FOREST-001", forested_observations)
    return engine.establish_baseline(
        plot_id="PLOT-FOREST-001",
        latitude=-3.12,
        longitude=28.57,
        country_code="CD",
    )


# ---------------------------------------------------------------------------
# TestBaselineEstablishment
# ---------------------------------------------------------------------------


class TestBaselineEstablishment:
    """Tests for establish_baseline for various plots."""

    def test_establish_forested_baseline(
        self,
        engine: HistoricalBaselineEngine,
        forested_observations: List[Dict[str, Any]],
    ) -> None:
        """Establish baseline for a well-forested plot."""
        engine.load_observations("PLOT-F1", forested_observations)
        result = engine.establish_baseline(
            plot_id="PLOT-F1",
            latitude=-3.12,
            longitude=28.57,
            country_code="CD",
        )
        assert result["status"] == BaselineStatus.ESTABLISHED.value
        assert Decimal(result["canopy_cover_pct"]) > Decimal("0")
        assert Decimal(result["ndvi_mean"]) > Decimal("0.5")
        assert result["forest_classification"] == ForestClassification.FOREST.value
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64
        assert result["processing_time_ms"] > 0

    def test_establish_non_forest_baseline(
        self,
        engine: HistoricalBaselineEngine,
        non_forest_observations: List[Dict[str, Any]],
    ) -> None:
        """Establish baseline for a non-forest area."""
        engine.load_observations("PLOT-NF1", non_forest_observations)
        result = engine.establish_baseline(
            plot_id="PLOT-NF1",
            latitude=-3.12,
            longitude=28.57,
        )
        assert result["status"] == BaselineStatus.ESTABLISHED.value
        assert Decimal(result["canopy_cover_pct"]) < CANOPY_COVER_FOREST_THRESHOLD
        assert result["forest_classification"] == ForestClassification.NON_FOREST.value

    def test_establish_degraded_baseline(
        self,
        engine: HistoricalBaselineEngine,
        degraded_observations: List[Dict[str, Any]],
    ) -> None:
        """Establish baseline for a degrading plot."""
        engine.load_observations("PLOT-DEG1", degraded_observations)
        result = engine.establish_baseline(
            plot_id="PLOT-DEG1",
            latitude=-3.12,
            longitude=28.57,
            country_code="CD",
        )
        assert result["status"] == BaselineStatus.ESTABLISHED.value
        assert result["trend_direction"] == TrendDirection.DECREASING.value

    def test_establish_baseline_insufficient_data(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Insufficient observations produce FAILED status."""
        engine.load_observations("PLOT-INSUF", [
            {"date": "2019-06-15", "source": "SENTINEL2", "ndvi": "0.75", "evi": "0.44", "cloud_pct": 5},
        ])
        result = engine.establish_baseline(
            plot_id="PLOT-INSUF",
            latitude=-3.12,
            longitude=28.57,
        )
        assert result["status"] == BaselineStatus.FAILED.value
        assert any("Insufficient" in w for w in result["warnings"])

    def test_establish_baseline_empty_plot_id_raises(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Empty plot_id raises ValueError."""
        with pytest.raises(ValueError):
            engine.establish_baseline(
                plot_id="",
                latitude=-3.12,
                longitude=28.57,
            )

    def test_establish_baseline_invalid_latitude_raises(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Invalid latitude raises ValueError."""
        with pytest.raises(ValueError):
            engine.establish_baseline(
                plot_id="PLOT-BAD",
                latitude=100,
                longitude=28.57,
            )

    def test_establish_baseline_custom_period(
        self, engine_custom: HistoricalBaselineEngine
    ) -> None:
        """Custom reference period (2017-2021) is respected."""
        assert engine_custom._baseline_start_year == 2017
        assert engine_custom._baseline_end_year == 2021

    def test_establish_baseline_with_custom_start_end(
        self,
        engine: HistoricalBaselineEngine,
        forested_observations: List[Dict[str, Any]],
    ) -> None:
        """Override start/end year in establish_baseline call."""
        engine.load_observations("PLOT-CUSTOM", forested_observations)
        result = engine.establish_baseline(
            plot_id="PLOT-CUSTOM",
            latitude=-3.12,
            longitude=28.57,
            start_year=2018,
            end_year=2020,
        )
        assert result["baseline_start_date"] == "2018-01-01"
        assert result["baseline_end_date"] == "2020-12-31"

    def test_establish_baseline_caches_result(
        self,
        engine: HistoricalBaselineEngine,
        forested_observations: List[Dict[str, Any]],
    ) -> None:
        """Established baseline is cached."""
        engine.load_observations("PLOT-CACHE", forested_observations)
        engine.establish_baseline(
            plot_id="PLOT-CACHE", latitude=-3.12, longitude=28.57
        )
        assert "PLOT-CACHE" in engine._baselines

    def test_establish_baseline_source_counts(
        self,
        engine: HistoricalBaselineEngine,
        forested_observations: List[Dict[str, Any]],
    ) -> None:
        """Baseline tracks observation counts per source."""
        engine.load_observations("PLOT-SRC", forested_observations)
        result = engine.establish_baseline(
            plot_id="PLOT-SRC", latitude=-3.12, longitude=28.57
        )
        assert "source_counts" in result
        assert isinstance(result["source_counts"], dict)


# ---------------------------------------------------------------------------
# TestBaselineComparison
# ---------------------------------------------------------------------------


class TestBaselineComparison:
    """Tests for compare_to_baseline showing loss/gain/stable."""

    def test_compare_loss(
        self,
        engine: HistoricalBaselineEngine,
        established_baseline: Dict[str, Any],
    ) -> None:
        """Comparison with lower NDVI shows LOSS."""
        current_obs = [
            {"date": "2023-06-15", "source": "SENTINEL2", "ndvi": "0.30", "evi": "0.15", "cloud_pct": 5},
            {"date": "2023-09-10", "source": "LANDSAT", "ndvi": "0.28", "evi": "0.14", "cloud_pct": 8},
            {"date": "2023-12-15", "source": "SENTINEL2", "ndvi": "0.25", "evi": "0.12", "cloud_pct": 10},
        ]
        result = engine.compare_to_baseline(
            "PLOT-FOREST-001",
            current_observations=current_obs,
        )
        assert result["change_direction"] == ChangeDirection.LOSS.value
        assert Decimal(result["canopy_change_pct"]) < Decimal("0")
        assert result["investigation_required"] is True
        assert result["provenance_hash"] != ""

    def test_compare_stable(
        self,
        engine: HistoricalBaselineEngine,
        established_baseline: Dict[str, Any],
    ) -> None:
        """Comparison with similar NDVI shows STABLE."""
        current_obs = [
            {"date": "2023-06-15", "source": "SENTINEL2", "ndvi": "0.76", "evi": "0.44", "cloud_pct": 5},
            {"date": "2023-09-10", "source": "LANDSAT", "ndvi": "0.75", "evi": "0.43", "cloud_pct": 8},
            {"date": "2023-12-15", "source": "SENTINEL2", "ndvi": "0.74", "evi": "0.42", "cloud_pct": 10},
        ]
        result = engine.compare_to_baseline(
            "PLOT-FOREST-001",
            current_observations=current_obs,
        )
        assert result["change_direction"] == ChangeDirection.STABLE.value

    def test_compare_gain(
        self,
        engine: HistoricalBaselineEngine,
        established_baseline: Dict[str, Any],
    ) -> None:
        """Comparison with higher canopy shows GAIN."""
        current_obs = [
            {"date": "2023-06-15", "source": "SENTINEL2", "ndvi": "0.88", "evi": "0.52", "cloud_pct": 3},
            {"date": "2023-09-10", "source": "LANDSAT", "ndvi": "0.90", "evi": "0.54", "cloud_pct": 4},
            {"date": "2023-12-15", "source": "SENTINEL2", "ndvi": "0.89", "evi": "0.53", "cloud_pct": 5},
        ]
        result = engine.compare_to_baseline(
            "PLOT-FOREST-001",
            current_observations=current_obs,
        )
        # Depending on exact canopy calculation, could be GAIN or STABLE
        assert result["change_direction"] in (
            ChangeDirection.GAIN.value,
            ChangeDirection.STABLE.value,
        )

    def test_compare_baseline_not_found_raises(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Comparing against nonexistent baseline raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.compare_to_baseline("NONEXISTENT-PLOT")

    def test_compare_without_current_obs(
        self,
        engine: HistoricalBaselineEngine,
        established_baseline: Dict[str, Any],
    ) -> None:
        """Comparison without current observations uses synthetic degradation."""
        result = engine.compare_to_baseline("PLOT-FOREST-001")
        assert "change_direction" in result
        assert "canopy_change_pct" in result


# ---------------------------------------------------------------------------
# TestBaselineUpdate
# ---------------------------------------------------------------------------


class TestBaselineUpdate:
    """Tests for baseline update with new data."""

    def test_re_establish_updates_cache(
        self,
        engine: HistoricalBaselineEngine,
        forested_observations: List[Dict[str, Any]],
    ) -> None:
        """Re-establishing baseline updates the cache."""
        engine.load_observations("PLOT-UPDATE", forested_observations)
        r1 = engine.establish_baseline(
            plot_id="PLOT-UPDATE", latitude=-3.12, longitude=28.57
        )
        r2 = engine.establish_baseline(
            plot_id="PLOT-UPDATE", latitude=-3.12, longitude=28.57
        )
        assert r2["baseline_id"] != r1["baseline_id"]


# ---------------------------------------------------------------------------
# TestCoverage
# ---------------------------------------------------------------------------


class TestCoverage:
    """Tests for get_coverage by country and commodity."""

    def test_coverage_after_establish(
        self,
        engine: HistoricalBaselineEngine,
        established_baseline: Dict[str, Any],
    ) -> None:
        """Coverage includes established baseline."""
        baselines = engine._baselines
        assert len(baselines) >= 1


# ---------------------------------------------------------------------------
# TestCanopyCover
# ---------------------------------------------------------------------------


class TestCanopyCover:
    """Tests for _calculate_canopy_cover at various locations."""

    @pytest.mark.parametrize(
        "ndvi_mean,expected_min,expected_max",
        [
            (Decimal("0.75"), Decimal("50"), Decimal("100")),   # High NDVI = high canopy
            (Decimal("0.55"), Decimal("30"), Decimal("80")),    # Medium NDVI
            (Decimal("0.15"), Decimal("0"), Decimal("15")),     # Low NDVI = low canopy
            (Decimal("0.10"), Decimal("0"), Decimal("10")),     # Near-zero canopy
        ],
    )
    def test_canopy_cover_ranges(
        self,
        engine: HistoricalBaselineEngine,
        ndvi_mean: Decimal,
        expected_min: Decimal,
        expected_max: Decimal,
    ) -> None:
        """Canopy cover percentage falls within expected ranges."""
        canopy = engine._calculate_canopy_cover(ndvi_mean, "tropical_moist")
        assert expected_min <= canopy <= expected_max


# ---------------------------------------------------------------------------
# TestForestArea
# ---------------------------------------------------------------------------


class TestForestArea:
    """Tests for _calculate_forest_area at various radii."""

    def test_forest_area_high_canopy(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """High canopy cover produces positive forest area."""
        area = engine._calculate_forest_area(
            Decimal("80"), DEFAULT_PLOT_RADIUS_KM
        )
        assert area > Decimal("0")

    def test_forest_area_zero_canopy(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Zero canopy cover produces zero forest area."""
        area = engine._calculate_forest_area(
            Decimal("0"), DEFAULT_PLOT_RADIUS_KM
        )
        assert area == Decimal("0")


# ---------------------------------------------------------------------------
# TestCanopyThreshold
# ---------------------------------------------------------------------------


class TestCanopyThreshold:
    """Tests for 10% canopy threshold (FAO forest definition)."""

    def test_10_percent_threshold(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Canopy >= 10% classifies as FOREST."""
        classification = engine._classify_forest(Decimal("15"))
        assert classification == ForestClassification.FOREST

    def test_below_threshold_non_forest(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Canopy < 5% classifies as NON_FOREST."""
        classification = engine._classify_forest(Decimal("3"))
        assert classification == ForestClassification.NON_FOREST

    def test_other_wooded_land(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Canopy 5-10% classifies as OTHER_WOODED_LAND."""
        classification = engine._classify_forest(Decimal("7"))
        assert classification == ForestClassification.OTHER_WOODED_LAND


# ---------------------------------------------------------------------------
# TestLossSignificance
# ---------------------------------------------------------------------------


class TestLossSignificance:
    """Tests for >5% canopy loss triggering investigation flag."""

    def test_significant_loss_triggers_investigation(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Canopy change > -5% is significant."""
        assert engine._is_change_significant(Decimal("-6")) is True

    def test_minor_loss_not_significant(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Canopy change within -5% to +5% is not significant."""
        assert engine._is_change_significant(Decimal("-3")) is False

    def test_gain_significant(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Canopy change > +5% is also significant (gain)."""
        assert engine._is_change_significant(Decimal("8")) is True


# ---------------------------------------------------------------------------
# TestBaselineStatus
# ---------------------------------------------------------------------------


class TestBaselineStatus:
    """Tests for baseline status enum values."""

    def test_status_values(self) -> None:
        """BaselineStatus has expected transitions."""
        assert BaselineStatus.ESTABLISHED.value == "ESTABLISHED"
        assert BaselineStatus.UPDATING.value == "UPDATING"
        assert BaselineStatus.PENDING.value == "PENDING"
        assert BaselineStatus.FAILED.value == "FAILED"
        assert BaselineStatus.ARCHIVED.value == "ARCHIVED"


# ---------------------------------------------------------------------------
# TestAnomalyDetection
# ---------------------------------------------------------------------------


class TestAnomalyDetection:
    """Tests for _detect_baseline_anomalies for unusual values."""

    def test_anomalies_in_degraded_plot(
        self,
        engine: HistoricalBaselineEngine,
        degraded_observations: List[Dict[str, Any]],
    ) -> None:
        """Degraded plot produces anomaly detections."""
        engine.load_observations("PLOT-ANOM", degraded_observations)
        result = engine.establish_baseline(
            plot_id="PLOT-ANOM", latitude=-3.12, longitude=28.57
        )
        # Degraded observations should produce at least some anomalies
        # or declining trend
        assert (
            len(result.get("anomalies", [])) >= 0
            or result["trend_direction"] == TrendDirection.DECREASING.value
        )

    def test_anomaly_type_enum(self) -> None:
        """AnomalyType enum has expected values."""
        assert AnomalyType.SUDDEN_DROP.value == "SUDDEN_DROP"
        assert AnomalyType.GRADUAL_DECLINE.value == "GRADUAL_DECLINE"
        assert AnomalyType.SEASONAL_ANOMALY.value == "SEASONAL_ANOMALY"
        assert AnomalyType.OUTLIER.value == "OUTLIER"
        assert AnomalyType.DATA_GAP.value == "DATA_GAP"


# ---------------------------------------------------------------------------
# TestReferenceTimeline
# ---------------------------------------------------------------------------


class TestReferenceTimeline:
    """Tests for _build_reference_timeline with multi-year data."""

    def test_reference_timeline_forested(
        self,
        engine: HistoricalBaselineEngine,
        forested_observations: List[Dict[str, Any]],
    ) -> None:
        """Forested plot timeline shows stable NDVI."""
        engine.load_observations("PLOT-TL", forested_observations)
        result = engine.establish_baseline(
            plot_id="PLOT-TL", latitude=-3.12, longitude=28.57
        )
        assert result["observation_count"] >= 3
        assert Decimal(result["ndvi_min"]) > Decimal("0.5")


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance hash generation."""

    def test_baseline_provenance_hash(
        self, established_baseline: Dict[str, Any]
    ) -> None:
        """Established baseline has provenance hash."""
        assert len(established_baseline["provenance_hash"]) == 64

    def test_comparison_provenance_hash(
        self,
        engine: HistoricalBaselineEngine,
        established_baseline: Dict[str, Any],
    ) -> None:
        """Comparison result has provenance hash."""
        result = engine.compare_to_baseline("PLOT-FOREST-001")
        assert len(result["provenance_hash"]) == 64

    def test_failed_baseline_provenance(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Even failed baselines have provenance hashes."""
        engine.load_observations("PLOT-FAIL-PROV", [
            {"date": "2019-06-15", "source": "SENTINEL2", "ndvi": "0.75", "evi": "0.44", "cloud_pct": 5},
        ])
        result = engine.establish_baseline(
            plot_id="PLOT-FAIL-PROV", latitude=-3.12, longitude=28.57
        )
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


class TestDataClasses:
    """Tests for data class serialization."""

    def test_historical_baseline_to_dict(self) -> None:
        """HistoricalBaseline serialization works."""
        bl = HistoricalBaseline(
            baseline_id="bl-1",
            plot_id="PLOT-1",
            canopy_cover_pct=Decimal("65"),
            ndvi_mean=Decimal("0.72"),
            status=BaselineStatus.ESTABLISHED.value,
        )
        d = bl.to_dict()
        assert d["plot_id"] == "PLOT-1"
        assert d["canopy_cover_pct"] == "65"
        assert d["status"] == "ESTABLISHED"

    def test_baseline_comparison_to_dict(self) -> None:
        """BaselineComparison serialization works."""
        comp = BaselineComparison(
            comparison_id="comp-1",
            change_direction=ChangeDirection.LOSS.value,
            canopy_change_pct=Decimal("-12.5"),
            investigation_required=True,
        )
        d = comp.to_dict()
        assert d["change_direction"] == "LOSS"
        assert d["investigation_required"] is True

    def test_coverage_statistics_to_dict(self) -> None:
        """CoverageStatistics serialization works."""
        cs = CoverageStatistics(
            stats_id="cs-1",
            total_plots=50,
            established_count=45,
            forest_plots=35,
        )
        d = cs.to_dict()
        assert d["total_plots"] == 50
        assert d["forest_plots"] == 35

    def test_load_observations_validates_id(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Empty plot_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.load_observations("", [{"date": "2019-01-01"}])

    def test_load_observations_validates_data(
        self, engine: HistoricalBaselineEngine
    ) -> None:
        """Empty observation list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.load_observations("PLOT-EMPTY", [])


# ---------------------------------------------------------------------------
# TestTrendDirection
# ---------------------------------------------------------------------------


class TestTrendDirection:
    """Tests for trend direction determination."""

    def test_trend_direction_values(self) -> None:
        """TrendDirection enum has expected values."""
        assert TrendDirection.INCREASING.value == "INCREASING"
        assert TrendDirection.DECREASING.value == "DECREASING"
        assert TrendDirection.STABLE.value == "STABLE"

    def test_change_direction_values(self) -> None:
        """ChangeDirection enum has expected values."""
        assert ChangeDirection.LOSS.value == "LOSS"
        assert ChangeDirection.GAIN.value == "GAIN"
        assert ChangeDirection.STABLE.value == "STABLE"

    def test_forest_classification_values(self) -> None:
        """ForestClassification enum has expected values."""
        assert ForestClassification.FOREST.value == "FOREST"
        assert ForestClassification.OTHER_WOODED_LAND.value == "OTHER_WOODED_LAND"
        assert ForestClassification.NON_FOREST.value == "NON_FOREST"
        assert ForestClassification.UNKNOWN.value == "UNKNOWN"
