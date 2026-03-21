# -*- coding: utf-8 -*-
"""
Unit tests for TemperatureScoringEngine (PACK-022 Engine 6).

Tests SBTi Temperature Rating v2.0 methodology including ARR calculation,
temperature mapping, entity scoring, portfolio aggregation, contribution
analysis, what-if scenarios, and the full assessment pipeline.
"""

import pytest
from decimal import Decimal

from engines.temperature_scoring_engine import (
    TemperatureScoringEngine,
    TemperatureScoringConfig,
    EmissionsTarget,
    PortfolioEntity,
    EntityTemperatureScore,
    PortfolioTemperatureScore,
    ContributionEntry,
    WhatIfScenario,
    WhatIfResult,
    TemperatureResult,
    ScoreType,
    TargetScope,
    TargetTimeframe,
    TemperatureBand,
    TargetValidityStatus,
    AggregationMethod,
    TEMPERATURE_MAPPING_TABLE,
    DEFAULT_TEMPERATURE_SCORE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Return a default TemperatureScoringEngine."""
    return TemperatureScoringEngine()


@pytest.fixture
def engine_custom():
    """Return an engine with custom configuration."""
    return TemperatureScoringEngine({"default_temperature": "3.50", "score_precision": 6})


@pytest.fixture
def ambitious_target():
    """Target with ~5% ARR (1.5C-aligned)."""
    return EmissionsTarget(
        entity_id="ent-a",
        scope=TargetScope.S1S2,
        timeframe=TargetTimeframe.NEAR_TERM,
        base_year=2020,
        target_year=2030,
        base_year_emissions=Decimal("1000"),
        target_year_emissions=Decimal("500"),
    )


@pytest.fixture
def moderate_target():
    """Target with ~2% ARR (below-2C)."""
    return EmissionsTarget(
        entity_id="ent-b",
        scope=TargetScope.S1S2,
        timeframe=TargetTimeframe.NEAR_TERM,
        base_year=2020,
        target_year=2030,
        base_year_emissions=Decimal("1000"),
        target_year_emissions=Decimal("800"),
    )


@pytest.fixture
def entity_with_target(ambitious_target):
    """PortfolioEntity with an ambitious S1S2 near-term target."""
    return PortfolioEntity(
        entity_id="ent-a",
        entity_name="Acme Corp",
        sector="Manufacturing",
        revenue=Decimal("500000000"),
        total_emissions=Decimal("100000"),
        targets=[ambitious_target],
    )


@pytest.fixture
def entity_no_target():
    """PortfolioEntity with no targets."""
    return PortfolioEntity(
        entity_id="ent-no",
        entity_name="NoTarget Inc",
        sector="Services",
        revenue=Decimal("200000000"),
        total_emissions=Decimal("50000"),
        targets=[],
    )


@pytest.fixture
def portfolio_engine(entity_with_target, entity_no_target):
    """Engine with two entities loaded."""
    eng = TemperatureScoringEngine()
    eng.add_entity(entity_with_target)
    eng.add_entity(entity_no_target)
    return eng


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestTemperatureScoringInit:
    """Tests for engine initialization and configuration."""

    def test_default_init(self, engine):
        assert isinstance(engine.config, TemperatureScoringConfig)
        assert engine.config.default_temperature == DEFAULT_TEMPERATURE_SCORE

    def test_custom_config_dict(self):
        eng = TemperatureScoringEngine({"default_temperature": "3.50", "score_precision": 6})
        assert eng.config.default_temperature == Decimal("3.50")
        assert eng.config.score_precision == 6

    def test_custom_config_object(self):
        cfg = TemperatureScoringConfig(default_temperature=Decimal("2.80"))
        eng = TemperatureScoringEngine(cfg)
        assert eng.config.default_temperature == Decimal("2.80")

    def test_none_config_uses_defaults(self):
        eng = TemperatureScoringEngine(None)
        assert eng.config.default_temperature == Decimal("3.20")


# ---------------------------------------------------------------------------
# Entity Management Tests
# ---------------------------------------------------------------------------


class TestEntityManagement:
    """Tests for adding, retrieving, and clearing entities."""

    def test_add_entity(self, engine, entity_with_target):
        result = engine.add_entity(entity_with_target)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_add_entities_batch(self, engine, entity_with_target, entity_no_target):
        count = engine.add_entities([entity_with_target, entity_no_target])
        assert count == 2

    def test_get_entity(self, engine, entity_with_target):
        engine.add_entity(entity_with_target)
        retrieved = engine.get_entity("ent-a")
        assert retrieved.entity_name == "Acme Corp"

    def test_get_entity_not_found(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.get_entity("nonexistent")

    def test_clear(self, portfolio_engine):
        portfolio_engine.clear()
        with pytest.raises(ValueError):
            portfolio_engine.get_entity("ent-a")


# ---------------------------------------------------------------------------
# ARR Calculation Tests
# ---------------------------------------------------------------------------


class TestARRCalculation:
    """Tests for annual reduction rate calculation."""

    def test_arr_50pct_over_10yrs(self, engine, ambitious_target):
        arr = engine.calculate_arr(ambitious_target)
        # (1 - 500/1000) / 10 * 100 = 5.0
        assert float(arr) == pytest.approx(5.0, rel=1e-3)

    def test_arr_20pct_over_10yrs(self, engine, moderate_target):
        arr = engine.calculate_arr(moderate_target)
        # (1 - 800/1000) / 10 * 100 = 2.0
        assert float(arr) == pytest.approx(2.0, rel=1e-3)

    def test_arr_zero_base_emissions(self, engine):
        target = EmissionsTarget(
            entity_id="x",
            scope=TargetScope.S1S2,
            timeframe=TargetTimeframe.NEAR_TERM,
            base_year=2020,
            target_year=2030,
            base_year_emissions=Decimal("0"),
            target_year_emissions=Decimal("100"),
        )
        assert engine.calculate_arr(target) == Decimal("0")

    def test_arr_zero_duration(self, engine):
        target = EmissionsTarget(
            entity_id="x",
            scope=TargetScope.S1S2,
            timeframe=TargetTimeframe.NEAR_TERM,
            base_year=2025,
            target_year=2025,
            base_year_emissions=Decimal("1000"),
            target_year_emissions=Decimal("500"),
        )
        assert engine.calculate_arr(target) == Decimal("0")

    def test_arr_negative_duration(self, engine):
        target = EmissionsTarget(
            entity_id="x",
            scope=TargetScope.S1S2,
            timeframe=TargetTimeframe.NEAR_TERM,
            base_year=2030,
            target_year=2020,
            base_year_emissions=Decimal("1000"),
            target_year_emissions=Decimal("500"),
        )
        assert engine.calculate_arr(target) == Decimal("0")

    def test_arr_100pct_reduction(self, engine):
        target = EmissionsTarget(
            entity_id="x",
            scope=TargetScope.S1S2,
            timeframe=TargetTimeframe.NEAR_TERM,
            base_year=2020,
            target_year=2030,
            base_year_emissions=Decimal("1000"),
            target_year_emissions=Decimal("0"),
        )
        arr = engine.calculate_arr(target)
        assert float(arr) == pytest.approx(10.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Temperature Mapping Tests
# ---------------------------------------------------------------------------


class TestTemperatureMapping:
    """Tests for ARR-to-temperature conversion and band classification."""

    def test_high_arr_gives_low_temp(self, engine):
        temp = engine.map_arr_to_temperature(Decimal("8.0"))
        assert float(temp) == pytest.approx(1.20, rel=1e-3)

    def test_zero_arr_gives_default(self, engine):
        temp = engine.map_arr_to_temperature(Decimal("0"))
        assert temp == engine.config.default_temperature

    def test_negative_arr_gives_default(self, engine):
        temp = engine.map_arr_to_temperature(Decimal("-1.0"))
        assert temp == engine.config.default_temperature

    def test_interpolation_mid_table(self, engine):
        # ARR=3.0 -> 1.70 (exact table entry)
        temp = engine.map_arr_to_temperature(Decimal("3.0"))
        assert float(temp) == pytest.approx(1.70, rel=1e-3)

    def test_interpolation_between_entries(self, engine):
        # ARR=4.6 is between 5.0->1.40 and 4.2->1.50
        temp = engine.map_arr_to_temperature(Decimal("4.6"))
        assert 1.40 <= float(temp) <= 1.50

    def test_band_1_5c(self, engine):
        assert engine.classify_temperature_band(Decimal("1.30")) == TemperatureBand.ALIGNED_1_5C

    def test_band_well_below_2c(self, engine):
        assert engine.classify_temperature_band(Decimal("1.70")) == TemperatureBand.WELL_BELOW_2C

    def test_band_below_2c(self, engine):
        assert engine.classify_temperature_band(Decimal("1.90")) == TemperatureBand.BELOW_2C

    def test_band_2c(self, engine):
        assert engine.classify_temperature_band(Decimal("2.30")) == TemperatureBand.ALIGNED_2C

    def test_band_above_2c(self, engine):
        assert engine.classify_temperature_band(Decimal("2.80")) == TemperatureBand.ABOVE_2C

    def test_band_no_target(self, engine):
        assert engine.classify_temperature_band(Decimal("3.20")) == TemperatureBand.NO_TARGET

    def test_boundary_1_5c(self, engine):
        assert engine.classify_temperature_band(Decimal("1.50")) == TemperatureBand.ALIGNED_1_5C


# ---------------------------------------------------------------------------
# Entity Scoring Tests
# ---------------------------------------------------------------------------


class TestEntityScoring:
    """Tests for scoring individual entities."""

    def test_score_entity_with_target(self, portfolio_engine):
        score = portfolio_engine.score_entity("ent-a", TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
        assert isinstance(score, EntityTemperatureScore)
        assert not score.is_default_score
        assert score.validity_status == TargetValidityStatus.VALID
        assert float(score.temperature_score) < 3.2
        assert len(score.provenance_hash) == 64

    def test_score_entity_no_target(self, portfolio_engine):
        score = portfolio_engine.score_entity("ent-no", TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
        assert score.is_default_score is True
        assert score.temperature_score == DEFAULT_TEMPERATURE_SCORE
        assert score.temperature_band == TemperatureBand.NO_TARGET
        assert score.validity_status == TargetValidityStatus.MISSING

    def test_score_all_entities(self, portfolio_engine):
        scores = portfolio_engine.score_all_entities(TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
        assert len(scores) == 2

    def test_score_entity_not_found(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.score_entity("bad-id", TargetScope.S1S2, TargetTimeframe.NEAR_TERM)


# ---------------------------------------------------------------------------
# Portfolio Aggregation Tests
# ---------------------------------------------------------------------------


class TestPortfolioAggregation:
    """Tests for portfolio-level temperature score aggregation."""

    def test_wats_aggregation(self, portfolio_engine):
        result = portfolio_engine.aggregate_portfolio(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert isinstance(result, PortfolioTemperatureScore)
        assert result.entities_count == 2
        assert result.entities_with_targets == 1
        assert result.entities_defaulted == 1
        assert len(result.provenance_hash) == 64

    def test_tets_aggregation(self, portfolio_engine):
        result = portfolio_engine.aggregate_portfolio(
            ScoreType.TETS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert result.score_type == ScoreType.TETS
        assert result.entities_count == 2

    def test_aggregate_all_methods(self, portfolio_engine):
        results = portfolio_engine.aggregate_all_methods(TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
        assert len(results) == 6  # WATS, TETS, MOTS, EOTS, ECOTS, AOTS

    def test_coverage_pct_with_mixed_targets(self, portfolio_engine):
        result = portfolio_engine.aggregate_portfolio(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert float(result.coverage_pct) > 0
        assert float(result.coverage_pct) < 100

    def test_portfolio_temp_between_entity_temps(self, portfolio_engine):
        result = portfolio_engine.aggregate_portfolio(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        # Should be between entity target temp and 3.2
        assert float(result.temperature_score) <= 3.2


# ---------------------------------------------------------------------------
# Contribution Analysis Tests
# ---------------------------------------------------------------------------


class TestContributionAnalysis:
    """Tests for entity contribution analysis."""

    def test_contributions_count(self, portfolio_engine):
        contribs = portfolio_engine.analyze_contributions(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert len(contribs) == 2

    def test_contributions_sum_to_portfolio(self, portfolio_engine):
        contribs = portfolio_engine.analyze_contributions(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        total = sum(c.contribution for c in contribs)
        portfolio = portfolio_engine.aggregate_portfolio(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert float(total) == pytest.approx(float(portfolio.temperature_score), rel=1e-2)

    def test_contributions_sorted_descending(self, portfolio_engine):
        contribs = portfolio_engine.analyze_contributions(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        for i in range(len(contribs) - 1):
            assert contribs[i].contribution >= contribs[i + 1].contribution

    def test_contribution_provenance_hash(self, portfolio_engine):
        contribs = portfolio_engine.analyze_contributions(
            ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        for c in contribs:
            assert len(c.provenance_hash) == 64
            assert all(ch in "0123456789abcdef" for ch in c.provenance_hash)


# ---------------------------------------------------------------------------
# What-If Analysis Tests
# ---------------------------------------------------------------------------


class TestWhatIfAnalysis:
    """Tests for what-if scenario evaluation."""

    def test_what_if_improvement(self, portfolio_engine):
        scenario = WhatIfScenario(
            entity_id="ent-no",
            new_annual_reduction_rate=Decimal("5.0"),
            scenario_name="Set 5% ARR",
        )
        result = portfolio_engine.run_what_if(
            scenario, ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert isinstance(result, WhatIfResult)
        assert float(result.modified_entity_temperature) < 3.2
        assert float(result.temperature_change) < 0

    def test_what_if_no_change(self, portfolio_engine):
        scenario = WhatIfScenario(entity_id="ent-no")
        result = portfolio_engine.run_what_if(
            scenario, ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert float(result.temperature_change) == pytest.approx(0.0, abs=1e-3)

    def test_what_if_batch(self, portfolio_engine):
        scenarios = [
            WhatIfScenario(entity_id="ent-no", new_annual_reduction_rate=Decimal("3.0")),
            WhatIfScenario(entity_id="ent-no", new_annual_reduction_rate=Decimal("6.0")),
        ]
        results = portfolio_engine.run_what_if_batch(
            scenarios, ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        assert len(results) == 2
        # Higher ARR should produce lower portfolio temp
        assert float(results[1].modified_portfolio_temperature) <= float(
            results[0].modified_portfolio_temperature
        )


# ---------------------------------------------------------------------------
# Full Assessment Tests
# ---------------------------------------------------------------------------


class TestFullAssessment:
    """Tests for the run_full_assessment pipeline."""

    def test_full_assessment_structure(self, portfolio_engine):
        result = portfolio_engine.run_full_assessment()
        assert isinstance(result, TemperatureResult)
        assert result.entities_assessed == 2
        assert len(result.entity_scores) == 2
        assert len(result.portfolio_scores) == 6
        assert len(result.contribution_analysis) == 2
        assert len(result.provenance_hash) == 64

    def test_full_assessment_with_what_if(self, portfolio_engine):
        scenarios = [
            WhatIfScenario(entity_id="ent-no", new_annual_reduction_rate=Decimal("4.0"))
        ]
        result = portfolio_engine.run_full_assessment(what_if_scenarios=scenarios)
        assert len(result.what_if_results) == 1


# ---------------------------------------------------------------------------
# Utility Method Tests
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """Tests for portfolio summary and distribution utilities."""

    def test_portfolio_summary(self, portfolio_engine):
        summary = portfolio_engine.get_portfolio_summary()
        assert summary["entity_count"] == 2
        assert summary["entities_with_targets"] == 1
        assert "sectors" in summary
        assert "provenance_hash" in summary

    def test_temperature_distribution(self, portfolio_engine):
        dist = portfolio_engine.get_temperature_distribution(
            TargetScope.S1S2, TargetTimeframe.NEAR_TERM
        )
        total = sum(dist.values())
        assert total == 2
        assert dist["no_target"] >= 1


# ---------------------------------------------------------------------------
# Edge Cases & Constants
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and reference data validation."""

    def test_mapping_table_is_descending_by_arr(self):
        for i in range(len(TEMPERATURE_MAPPING_TABLE) - 1):
            assert TEMPERATURE_MAPPING_TABLE[i][0] > TEMPERATURE_MAPPING_TABLE[i + 1][0]

    def test_default_temperature_score_value(self):
        assert DEFAULT_TEMPERATURE_SCORE == Decimal("3.20")

    def test_enum_values(self):
        assert ScoreType.WATS.value == "wats"
        assert TargetScope.S1S2S3.value == "s1s2s3"
        assert TargetTimeframe.LONG_TERM.value == "long_term"
        assert TemperatureBand.ALIGNED_1_5C.value == "1.5C"
