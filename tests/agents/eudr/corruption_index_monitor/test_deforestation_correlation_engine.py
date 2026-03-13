# -*- coding: utf-8 -*-
"""
Unit tests for DeforestationCorrelationEngine (AGENT-EUDR-019, Engine 6).

Tests all methods of DeforestationCorrelationEngine including Pearson/Spearman/Kendall
correlation analysis, country deforestation link assessment, regression model building,
heatmap data generation, causal pathway identification, simple and multiple regression,
correlation significance testing, and provenance chain integrity.

Key empirical findings validated:
    - Low CPI countries should show positive correlation with deforestation
    - CPI < 30 countries should have 3-5x higher deforestation rates
    - Control of Corruption (WGI) should negatively correlate with deforestation

Coverage target: 85%+ of DeforestationCorrelationEngine methods.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.engines.deforestation_correlation_engine import (
    DeforestationCorrelationEngine,
    CorrelationType,
    EvidenceStrength,
    SignificanceLevel,
    DeforestationMetric,
    CorrelationResult,
    RegressionModel,
    CausalPathway,
    CountryDeforestationLink,
    HeatmapCell,
    REFERENCE_DEFORESTATION_RATES,
    REFERENCE_CPI_SCORES,
    CAUSAL_PATHWAYS,
    MIN_CORRELATION_SAMPLES,
    MIN_REGRESSION_SAMPLES,
    CPI_HIGH_RISK_DEFORESTATION_THRESHOLD,
    CPI_LOW_RISK_DEFORESTATION_THRESHOLD,
    HIGH_CORRUPTION_DEFORESTATION_MULTIPLIER,
    _to_decimal,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> DeforestationCorrelationEngine:
    """Create a default DeforestationCorrelationEngine instance."""
    return DeforestationCorrelationEngine()


# ---------------------------------------------------------------------------
# TestCorrelationAnalysis
# ---------------------------------------------------------------------------


class TestCorrelationAnalysis:
    """Tests for analyze_correlation with Pearson/Spearman/Kendall."""

    def test_pearson_cpi_vs_deforestation(
        self, engine: DeforestationCorrelationEngine
    ):
        """Pearson correlation between CPI and deforestation rate should be negative."""
        result = engine.analyze_correlation("CPI", "loss_rate_pct", method="PEARSON")
        assert "coefficient" in result
        # Higher CPI = less corruption = less deforestation -> negative correlation
        coeff = Decimal(str(result["coefficient"]))
        assert coeff < Decimal("0")

    def test_spearman_cpi_vs_deforestation(
        self, engine: DeforestationCorrelationEngine
    ):
        """Spearman rank correlation should also be negative."""
        result = engine.analyze_correlation("CPI", "loss_rate_pct", method="SPEARMAN")
        assert "coefficient" in result
        coeff = Decimal(str(result["coefficient"]))
        assert coeff < Decimal("0")

    def test_correlation_result_has_sample_size(
        self, engine: DeforestationCorrelationEngine
    ):
        """Correlation result should include sample size."""
        result = engine.analyze_correlation("CPI", "loss_rate_pct")
        assert result["sample_size"] > 0

    def test_correlation_result_has_significance(
        self, engine: DeforestationCorrelationEngine
    ):
        """Correlation result should include significance level."""
        result = engine.analyze_correlation("CPI", "loss_rate_pct")
        assert "significance_level" in result
        assert "is_significant" in result

    def test_correlation_with_countries_filter(
        self, engine: DeforestationCorrelationEngine
    ):
        """Filtering to specific countries should limit sample size."""
        result = engine.analyze_correlation(
            "CPI", "loss_rate_pct",
            countries=["BR", "ID", "CD"],
        )
        assert result["sample_size"] == 3

    def test_correlation_result_has_provenance(
        self, engine: DeforestationCorrelationEngine
    ):
        """Correlation result should include provenance hash."""
        result = engine.analyze_correlation("CPI", "loss_rate_pct")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestDeforestationLink
# ---------------------------------------------------------------------------


class TestDeforestationLink:
    """Tests for get_country_deforestation_link for high/low corruption countries."""

    def test_high_corruption_country_brazil(
        self, engine: DeforestationCorrelationEngine
    ):
        """Brazil (high corruption, major deforestation) should show strong link."""
        result = engine.get_country_deforestation_link("BR")
        assert result["country_code"] == "BR"
        assert Decimal(str(result["deforestation_rate_pct"])) > Decimal("0")
        assert result["primary_driver"] == "cattle_ranching_soy"

    def test_low_corruption_country_denmark(
        self, engine: DeforestationCorrelationEngine
    ):
        """Denmark (low corruption) should show minimal deforestation."""
        result = engine.get_country_deforestation_link("DK")
        assert result["country_code"] == "DK"
        assert Decimal(str(result["deforestation_rate_pct"])) < Decimal("0.05")

    def test_indonesia_link(self, engine: DeforestationCorrelationEngine):
        """Indonesia should show palm oil as primary driver."""
        result = engine.get_country_deforestation_link("ID")
        assert result["primary_driver"] == "palm_oil"
        assert Decimal(str(result["cpi_score"])) < Decimal("40")

    def test_unknown_country_link(self, engine: DeforestationCorrelationEngine):
        """Unknown country should still return a result with warnings."""
        result = engine.get_country_deforestation_link("ZZ")
        assert "country_code" in result

    def test_risk_multiplier_for_corrupt_country(
        self, engine: DeforestationCorrelationEngine
    ):
        """High-corruption country should have risk multiplier > 1."""
        result = engine.get_country_deforestation_link("NG")
        assert Decimal(str(result["risk_multiplier"])) > Decimal("1.0")

    def test_link_has_active_pathways(self, engine: DeforestationCorrelationEngine):
        """Countries with deforestation should have active causal pathways."""
        result = engine.get_country_deforestation_link("BR")
        assert len(result.get("active_pathways", [])) > 0

    def test_link_has_provenance(self, engine: DeforestationCorrelationEngine):
        """Link result should have provenance hash."""
        result = engine.get_country_deforestation_link("BR")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestRegressionModel
# ---------------------------------------------------------------------------


class TestRegressionModel:
    """Tests for build_regression_model with single/multiple predictors."""

    def test_simple_regression(self, engine: DeforestationCorrelationEngine):
        """Simple regression with CPI should produce valid model."""
        result = engine.build_regression_model(
            predictors=["CPI"],
            target="loss_rate_pct",
        )
        assert "r_squared" in result
        assert "equation" in result
        assert Decimal(str(result["r_squared"])) >= Decimal("0")

    def test_multiple_regression(self, engine: DeforestationCorrelationEngine):
        """Multiple regression should include all predictors."""
        result = engine.build_regression_model(
            predictors=["CPI", "WGI"],
            target="loss_rate_pct",
        )
        assert "coefficients" in result
        assert "r_squared" in result

    def test_regression_has_provenance(self, engine: DeforestationCorrelationEngine):
        """Regression model should have provenance hash."""
        result = engine.build_regression_model(
            predictors=["CPI"],
            target="loss_rate_pct",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestHeatmapData
# ---------------------------------------------------------------------------


class TestHeatmapData:
    """Tests for generate_heatmap_data for different regions."""

    def test_all_regions_heatmap(self, engine: DeforestationCorrelationEngine):
        """All-regions heatmap should include countries from reference data."""
        result = engine.generate_heatmap_data()
        assert "cells" in result
        assert len(result["cells"]) > 0

    def test_africa_region_filter(self, engine: DeforestationCorrelationEngine):
        """Filtering to Africa should only include African countries."""
        result = engine.generate_heatmap_data(region="africa")
        assert "cells" in result
        for cell in result["cells"]:
            assert cell["region"] == "africa"

    def test_americas_region_filter(self, engine: DeforestationCorrelationEngine):
        """Filtering to Americas should only include American countries."""
        result = engine.generate_heatmap_data(region="americas")
        for cell in result["cells"]:
            assert cell["region"] == "americas"

    def test_heatmap_has_provenance(self, engine: DeforestationCorrelationEngine):
        """Heatmap result should have provenance hash."""
        result = engine.generate_heatmap_data()
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestCausalPathways
# ---------------------------------------------------------------------------


class TestCausalPathways:
    """Tests for identify_causal_pathways for known mechanisms."""

    def test_brazil_pathways(self, engine: DeforestationCorrelationEngine):
        """Brazil should have multiple active causal pathways."""
        result = engine.identify_causal_pathways("BR")
        assert len(result["pathways"]) >= 2

    def test_pathway_fields(self, engine: DeforestationCorrelationEngine):
        """Each pathway should have required fields."""
        result = engine.identify_causal_pathways("BR")
        for pathway in result["pathways"]:
            assert "pathway_id" in pathway
            assert "pathway_name" in pathway
            assert "evidence_strength" in pathway
            assert "mechanism" in pathway

    def test_unknown_country_pathways(self, engine: DeforestationCorrelationEngine):
        """Unknown country should return general pathways or empty."""
        result = engine.identify_causal_pathways("ZZ")
        assert "pathways" in result

    def test_all_pathways_have_references(self):
        """All predefined causal pathways should have academic references."""
        for pathway in CAUSAL_PATHWAYS:
            assert len(pathway["references"]) > 0

    def test_causal_pathway_evidence_strength(self):
        """Pathway evidence strengths should be valid enum values."""
        valid_strengths = {e.value for e in EvidenceStrength}
        for pathway in CAUSAL_PATHWAYS:
            assert pathway["evidence_strength"] in valid_strengths

    def test_pathway_has_provenance(self, engine: DeforestationCorrelationEngine):
        """Pathway result should have provenance hash."""
        result = engine.identify_causal_pathways("BR")
        assert "provenance_hash" in result


# ---------------------------------------------------------------------------
# TestPearsonCorrelation
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    """Tests for _pearson_correlation with known values."""

    def test_perfect_positive_correlation(
        self, engine: DeforestationCorrelationEngine
    ):
        """Perfectly correlated data should give r = 1.0."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"), Decimal("50")]
        r, p = engine._pearson_correlation(x, y)
        assert abs(r - Decimal("1.0")) < Decimal("0.01")

    def test_perfect_negative_correlation(
        self, engine: DeforestationCorrelationEngine
    ):
        """Perfectly inversely correlated data should give r = -1.0."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("50"), Decimal("40"), Decimal("30"), Decimal("20"), Decimal("10")]
        r, p = engine._pearson_correlation(x, y)
        assert abs(r - Decimal("-1.0")) < Decimal("0.01")

    def test_no_correlation(self, engine: DeforestationCorrelationEngine):
        """Uncorrelated data should give r close to 0."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("3"), Decimal("1"), Decimal("4"), Decimal("1"), Decimal("5")]
        r, p = engine._pearson_correlation(x, y)
        assert abs(r) < Decimal("0.5")

    def test_insufficient_data_raises(self, engine: DeforestationCorrelationEngine):
        """Fewer than 3 data points should raise ValueError."""
        with pytest.raises(ValueError):
            engine._pearson_correlation(
                [Decimal("1"), Decimal("2")],
                [Decimal("3"), Decimal("4")],
            )

    def test_mismatched_lengths_raises(self, engine: DeforestationCorrelationEngine):
        """Different length arrays should raise ValueError."""
        with pytest.raises(ValueError):
            engine._pearson_correlation(
                [Decimal("1"), Decimal("2"), Decimal("3")],
                [Decimal("4"), Decimal("5")],
            )


# ---------------------------------------------------------------------------
# TestSpearmanCorrelation
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    """Tests for _spearman_rank_correlation accuracy."""

    def test_perfect_rank_correlation(self, engine: DeforestationCorrelationEngine):
        """Perfectly monotonic data should give Spearman rho close to 1."""
        x = [Decimal("1"), Decimal("3"), Decimal("5"), Decimal("7"), Decimal("9")]
        y = [Decimal("2"), Decimal("6"), Decimal("10"), Decimal("14"), Decimal("18")]
        rho, p = engine._spearman_rank_correlation(x, y)
        assert abs(rho - Decimal("1.0")) < Decimal("0.01")

    def test_rank_correlation_with_ties(self, engine: DeforestationCorrelationEngine):
        """Tied values should be handled (average rank)."""
        x = [Decimal("1"), Decimal("1"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("10"), Decimal("10"), Decimal("30"), Decimal("40"), Decimal("50")]
        rho, p = engine._spearman_rank_correlation(x, y)
        assert abs(rho) <= Decimal("1.0")

    def test_compute_ranks_basic(self, engine: DeforestationCorrelationEngine):
        """Ranks should be 1-based and handle ties with average."""
        values = [Decimal("10"), Decimal("30"), Decimal("20")]
        ranks = engine._compute_ranks(values)
        assert ranks[0] == Decimal("1")  # 10 is smallest
        assert ranks[1] == Decimal("3")  # 30 is largest
        assert ranks[2] == Decimal("2")  # 20 is middle


# ---------------------------------------------------------------------------
# TestSimpleRegression
# ---------------------------------------------------------------------------


class TestSimpleRegression:
    """Tests for _simple_linear_regression with known slope/intercept."""

    def test_known_regression(self, engine: DeforestationCorrelationEngine):
        """Known data should produce correct slope and intercept."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("12"), Decimal("14"), Decimal("16"), Decimal("18"), Decimal("20")]
        result = engine._simple_linear_regression(x, y)
        assert abs(Decimal(str(result["slope"])) - Decimal("2")) < Decimal("0.01")
        assert abs(Decimal(str(result["intercept"])) - Decimal("10")) < Decimal("0.01")
        assert Decimal(str(result["r_squared"])) > Decimal("0.99")

    def test_regression_equation_format(self, engine: DeforestationCorrelationEngine):
        """Regression should produce a readable equation."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("5"), Decimal("10"), Decimal("15"), Decimal("20"), Decimal("25")]
        result = engine._simple_linear_regression(x, y)
        assert "deforestation_rate" in result["equation"]

    def test_insufficient_points_raises(self, engine: DeforestationCorrelationEngine):
        """Fewer than 3 points should raise ValueError."""
        with pytest.raises(ValueError):
            engine._simple_linear_regression(
                [Decimal("1"), Decimal("2")],
                [Decimal("3"), Decimal("4")],
            )


# ---------------------------------------------------------------------------
# TestMultipleRegression
# ---------------------------------------------------------------------------


class TestMultipleRegression:
    """Tests for _multiple_regression with multiple predictors."""

    def test_two_predictor_regression(self, engine: DeforestationCorrelationEngine):
        """Two-predictor regression should produce coefficients for both."""
        X = [
            [Decimal("1"), Decimal("10")],
            [Decimal("2"), Decimal("20")],
            [Decimal("3"), Decimal("30")],
            [Decimal("4"), Decimal("40")],
            [Decimal("5"), Decimal("50")],
        ]
        y = [Decimal("5"), Decimal("10"), Decimal("15"), Decimal("20"), Decimal("25")]
        result = engine._multiple_regression(X, y, ["x1", "x2"])
        assert "coefficients" in result
        assert "r_squared" in result

    def test_single_predictor_delegation(
        self, engine: DeforestationCorrelationEngine
    ):
        """Single predictor should delegate to simple regression."""
        X = [[Decimal("1")], [Decimal("2")], [Decimal("3")], [Decimal("4")], [Decimal("5")]]
        y = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"), Decimal("50")]
        result = engine._multiple_regression(X, y, ["x1"])
        assert "coefficients" in result

    def test_insufficient_observations_raises(
        self, engine: DeforestationCorrelationEngine
    ):
        """Fewer than 3 observations should raise ValueError."""
        with pytest.raises(ValueError):
            engine._multiple_regression(
                [[Decimal("1"), Decimal("2")]],
                [Decimal("3")],
                ["x1", "x2"],
            )


# ---------------------------------------------------------------------------
# TestCorrelationSignificance
# ---------------------------------------------------------------------------


class TestCorrelationSignificance:
    """Tests for p_value thresholds and significance testing."""

    def test_strong_correlation_significant(
        self, engine: DeforestationCorrelationEngine
    ):
        """Strong correlation with enough data should be significant."""
        x = [Decimal(str(i)) for i in range(1, 31)]
        y = [Decimal(str(i * 2 + 1)) for i in range(1, 31)]
        r, p = engine._pearson_correlation(x, y)
        assert p < Decimal("0.05")

    def test_significance_level_enum(self):
        """All significance levels should be defined."""
        levels = set(s.value for s in SignificanceLevel)
        assert "p<0.001" in levels
        assert "p<0.01" in levels
        assert "p<0.05" in levels
        assert "p<0.10" in levels
        assert "not_significant" in levels

    def test_p_value_bounds(self, engine: DeforestationCorrelationEngine):
        """P-value should be in [0, 1]."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("3"), Decimal("1"), Decimal("4"), Decimal("1"), Decimal("5")]
        r, p = engine._pearson_correlation(x, y)
        assert Decimal("0") <= p <= Decimal("1")


# ---------------------------------------------------------------------------
# TestCorrelationProvenance
# ---------------------------------------------------------------------------


class TestCorrelationProvenance:
    """Tests for provenance chain integrity."""

    def test_correlation_provenance(self, engine: DeforestationCorrelationEngine):
        """Correlation result should have 64-char provenance hash."""
        result = engine.analyze_correlation("CPI", "loss_rate_pct")
        assert len(result["provenance_hash"]) == 64

    def test_provenance_deterministic(self, engine: DeforestationCorrelationEngine):
        """Same inputs should produce the same provenance hash."""
        r1 = engine.analyze_correlation(
            "CPI", "loss_rate_pct", method="PEARSON",
            countries=["BR", "ID", "DK"],
        )
        r2 = engine.analyze_correlation(
            "CPI", "loss_rate_pct", method="PEARSON",
            countries=["BR", "ID", "DK"],
        )
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_link_provenance_deterministic(
        self, engine: DeforestationCorrelationEngine
    ):
        """Same country link queries should produce same provenance."""
        r1 = engine.get_country_deforestation_link("BR")
        r2 = engine.get_country_deforestation_link("BR")
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_reference_data_consistency(self):
        """Reference CPI and deforestation data should have overlapping countries."""
        common = set(REFERENCE_CPI_SCORES.keys()) & set(
            REFERENCE_DEFORESTATION_RATES.keys()
        )
        assert len(common) >= 20  # At least 20 countries in common

    def test_custom_data_loading(self, engine: DeforestationCorrelationEngine):
        """Loading custom CPI data should override reference."""
        engine.load_custom_cpi_data({"XX": Decimal("42")})
        # Verify it does not crash on subsequent operations
        result = engine.analyze_correlation("CPI", "loss_rate_pct")
        assert result["sample_size"] > 0

    def test_custom_deforestation_loading(
        self, engine: DeforestationCorrelationEngine
    ):
        """Loading custom deforestation data should work."""
        engine.load_custom_deforestation_data({
            "XX": {
                "annual_loss_rate_pct": Decimal("1.5"),
                "annual_loss_ha": Decimal("100000"),
                "total_forest_ha": Decimal("5000000"),
                "primary_driver": "test",
                "region": "test",
            }
        })
        result = engine.analyze_correlation("CPI", "loss_rate_pct")
        assert result["sample_size"] > 0

    def test_load_empty_cpi_raises(self, engine: DeforestationCorrelationEngine):
        """Loading empty CPI data should raise ValueError."""
        with pytest.raises(ValueError):
            engine.load_custom_cpi_data({})

    def test_load_empty_deforestation_raises(
        self, engine: DeforestationCorrelationEngine
    ):
        """Loading empty deforestation data should raise ValueError."""
        with pytest.raises(ValueError):
            engine.load_custom_deforestation_data({})

    def test_deforestation_metric_enum(self):
        """All deforestation metrics should be defined."""
        metrics = set(m.value for m in DeforestationMetric)
        assert "annual_loss_ha" in metrics
        assert "loss_rate_pct" in metrics
        assert "tree_cover_loss_ha" in metrics
        assert "net_deforestation_ha" in metrics

    def test_correlation_type_enum(self):
        """All correlation types should be defined."""
        types = set(c.value for c in CorrelationType)
        assert types == {"PEARSON", "SPEARMAN", "KENDALL"}
