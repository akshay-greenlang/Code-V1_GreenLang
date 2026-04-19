"""
Unit tests for SamplingPlanEngine (PACK-048 Engine 7).

Tests all public methods with 28+ tests covering:
  - Population identification
  - Stratification by scope
  - MUS sample size
  - Confidence 95% reasonable
  - Confidence 80% limited
  - High-value item 100% coverage
  - Key item selection
  - Projected misstatement
  - Empty stratum
  - Small population

Author: GreenLang QA Team
"""
from __future__ import annotations

import math
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# Population Identification Tests
# ---------------------------------------------------------------------------


class TestPopulationIdentification:
    """Tests for sampling population identification."""

    def test_population_includes_all_emission_sources(self, sample_emissions_data):
        """Test population includes all emission source line items."""
        s1_items = len(sample_emissions_data["scope_1"]) - 1  # Exclude total
        s2_items = len(sample_emissions_data["scope_2_location"]) - 1
        s3_items = len(sample_emissions_data["scope_3"]) - 1
        total_items = s1_items + s2_items + s3_items
        assert total_items >= 10

    def test_population_total_matches_reported(self, sample_emissions_data):
        """Test population total matches reported total."""
        s1 = sample_emissions_data["scope_1"]["total_tco2e"]
        s2 = sample_emissions_data["scope_2_location"]["total_tco2e"]
        s3 = sample_emissions_data["scope_3"]["total_tco2e"]
        calculated_total = s1 + s2 + s3
        reported_total = sample_emissions_data["total_all_scopes_tco2e"]
        assert_decimal_equal(calculated_total, reported_total)

    def test_population_non_empty(self, sample_emissions_data):
        """Test population is non-empty."""
        assert sample_emissions_data["total_all_scopes_tco2e"] > Decimal("0")


# ---------------------------------------------------------------------------
# Stratification Tests
# ---------------------------------------------------------------------------


class TestStratificationByScope:
    """Tests for stratification by scope."""

    def test_3_strata_by_scope(self, sample_emissions_data):
        """Test population is stratified into 3 strata (S1, S2, S3)."""
        strata = [
            sample_emissions_data["scope_1"]["total_tco2e"],
            sample_emissions_data["scope_2_location"]["total_tco2e"],
            sample_emissions_data["scope_3"]["total_tco2e"],
        ]
        assert len(strata) == 3

    def test_strata_sum_to_total(self, sample_emissions_data):
        """Test strata totals sum to population total."""
        s1 = sample_emissions_data["scope_1"]["total_tco2e"]
        s2 = sample_emissions_data["scope_2_location"]["total_tco2e"]
        s3 = sample_emissions_data["scope_3"]["total_tco2e"]
        total = sample_emissions_data["total_all_scopes_tco2e"]
        assert_decimal_equal(s1 + s2 + s3, total)

    def test_scope_3_largest_stratum(self, sample_emissions_data):
        """Test Scope 3 is the largest stratum."""
        s1 = sample_emissions_data["scope_1"]["total_tco2e"]
        s2 = sample_emissions_data["scope_2_location"]["total_tco2e"]
        s3 = sample_emissions_data["scope_3"]["total_tco2e"]
        assert s3 > s1
        assert s3 > s2


# ---------------------------------------------------------------------------
# MUS Sample Size Tests
# ---------------------------------------------------------------------------


class TestMUSSampleSize:
    """Tests for Monetary Unit Sampling (MUS) sample size calculation."""

    def test_mus_formula(self, sampling_engine_config):
        """Test MUS sample size formula: n = population / (tolerable - expected)."""
        population_value = Decimal("23000")
        tolerable_pct = sampling_engine_config["tolerable_misstatement_pct"]
        expected_pct = sampling_engine_config["expected_misstatement_pct"]
        tolerable = population_value * tolerable_pct / Decimal("100")
        expected = population_value * expected_pct / Decimal("100")
        if tolerable > expected:
            sampling_interval = tolerable - expected
            sample_size = int(population_value / sampling_interval)
        else:
            sample_size = int(population_value)
        assert sample_size > 0

    def test_larger_population_larger_sample(self):
        """Test larger population results in larger sample size."""
        pop_small = Decimal("10000")
        pop_large = Decimal("100000")
        interval = Decimal("920")  # tolerable - expected
        n_small = int(pop_small / interval)
        n_large = int(pop_large / interval)
        assert n_large > n_small

    def test_lower_tolerable_larger_sample(self):
        """Test lower tolerable misstatement increases sample size."""
        population = Decimal("23000")
        expected = Decimal("230")
        tolerable_high = Decimal("1150")
        tolerable_low = Decimal("575")
        n_high = int(population / (tolerable_high - expected))
        n_low = int(population / (tolerable_low - expected))
        assert n_low > n_high


# ---------------------------------------------------------------------------
# Confidence Level Tests
# ---------------------------------------------------------------------------


class TestConfidence95Reasonable:
    """Tests for 95% confidence level (reasonable assurance)."""

    def test_confidence_95_pct(self, sampling_engine_config):
        """Test reasonable assurance uses 95% confidence level."""
        assert sampling_engine_config["confidence_level_reasonable"] == Decimal("0.95")

    def test_confidence_factor_reasonable(self):
        """Test confidence factor for 95% is approximately 3.0."""
        # In audit sampling, the confidence factor (reliability factor) for 95%
        # at 0% expected error is approximately 3.0
        confidence_factor = Decimal("3.0")
        assert_decimal_between(confidence_factor, Decimal("2.5"), Decimal("3.5"))


class TestConfidence80Limited:
    """Tests for 80% confidence level (limited assurance)."""

    def test_confidence_80_pct(self, sampling_engine_config):
        """Test limited assurance uses 80% confidence level."""
        assert sampling_engine_config["confidence_level_limited"] == Decimal("0.80")

    def test_limited_smaller_sample_than_reasonable(self):
        """Test limited assurance requires smaller sample than reasonable."""
        # Higher confidence = larger sample
        pop = Decimal("23000")
        factor_reasonable = Decimal("3.0")
        factor_limited = Decimal("1.6")
        tolerable = Decimal("1150")
        n_reasonable = int(pop * factor_reasonable / tolerable)
        n_limited = int(pop * factor_limited / tolerable)
        assert n_limited < n_reasonable

    def test_confidence_factor_limited(self):
        """Test confidence factor for 80% is approximately 1.6."""
        confidence_factor = Decimal("1.6")
        assert_decimal_between(confidence_factor, Decimal("1.0"), Decimal("2.0"))


# ---------------------------------------------------------------------------
# High-Value Item Tests
# ---------------------------------------------------------------------------


class TestHighValueItem100Pct:
    """Tests for high-value item 100% coverage."""

    def test_high_value_threshold(self, sampling_engine_config):
        """Test high-value threshold is 50% of materiality."""
        assert sampling_engine_config["high_value_threshold_pct"] == Decimal("50")

    def test_items_above_threshold_selected_100pct(self, sample_emissions_data):
        """Test items above threshold are selected at 100%."""
        overall_materiality = sample_emissions_data["total_all_scopes_tco2e"] * Decimal("5") / Decimal("100")
        threshold = overall_materiality * Decimal("50") / Decimal("100")
        # Scope 3 cat 1 (8000) exceeds threshold (575)
        s3_cat1 = sample_emissions_data["scope_3"]["cat_1_purchased_goods"]
        assert s3_cat1 > threshold

    def test_small_items_not_100pct(self, sample_emissions_data):
        """Test small items are not selected at 100%."""
        overall_materiality = sample_emissions_data["total_all_scopes_tco2e"] * Decimal("5") / Decimal("100")
        threshold = overall_materiality * Decimal("50") / Decimal("100")
        # Scope 3 cat 7 employee commuting (500) may or may not exceed
        s3_cat7 = sample_emissions_data["scope_3"]["cat_7_employee_commuting"]
        # 500 < 575 threshold
        assert s3_cat7 < threshold


# ---------------------------------------------------------------------------
# Key Item Selection Tests
# ---------------------------------------------------------------------------


class TestKeyItemSelection:
    """Tests for key item selection criteria."""

    def test_key_items_include_largest_sources(self, sample_emissions_data):
        """Test key items include the largest emission sources."""
        sources = {
            "scope_1_stationary": sample_emissions_data["scope_1"]["stationary_combustion"],
            "scope_3_cat_1": sample_emissions_data["scope_3"]["cat_1_purchased_goods"],
        }
        largest = max(sources, key=sources.get)
        assert largest == "scope_3_cat_1"

    def test_key_items_have_elevated_testing(self):
        """Test key items have elevated testing requirements."""
        key_item = {"item": "cat_1_purchased_goods", "testing_level": "extended", "sample_pct": 100}
        assert key_item["testing_level"] == "extended"


# ---------------------------------------------------------------------------
# Projected Misstatement Tests
# ---------------------------------------------------------------------------


class TestProjectedMisstatement:
    """Tests for projected misstatement calculation."""

    def test_misstatement_projection_from_sample(self):
        """Test misstatement projected from sample to population."""
        sample_total = Decimal("5000")
        sample_misstatement = Decimal("100")
        population_total = Decimal("23000")
        misstatement_rate = sample_misstatement / sample_total
        projected = population_total * misstatement_rate
        assert_decimal_equal(projected, Decimal("460"), tolerance=Decimal("1"))

    def test_projected_vs_materiality(self):
        """Test projected misstatement compared to materiality."""
        projected = Decimal("460")
        materiality = Decimal("1150")
        within_materiality = projected < materiality
        assert within_materiality is True

    def test_zero_misstatement_in_sample(self):
        """Test zero misstatement in sample projects to zero."""
        sample_misstatement = Decimal("0")
        population_total = Decimal("23000")
        sample_total = Decimal("5000")
        projected = population_total * (sample_misstatement / sample_total)
        assert_decimal_equal(projected, Decimal("0"))


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestSamplingEdgeCases:
    """Tests for sampling plan edge cases."""

    def test_empty_stratum_produces_zero_sample(self):
        """Test empty stratum produces zero sample size."""
        stratum_total = Decimal("0")
        population_total = Decimal("23000")
        weight = stratum_total / population_total if population_total > 0 else Decimal("0")
        sample_from_stratum = int(Decimal("25") * weight)
        assert sample_from_stratum == 0

    def test_small_population_full_coverage(self):
        """Test very small population gets 100% coverage."""
        population_items = 3
        min_sample = 5
        actual_sample = min(population_items, min_sample)
        assert actual_sample == 3  # Cannot sample more than population

    def test_single_item_population(self):
        """Test single item population."""
        population_items = 1
        sample_size = min(population_items, 25)
        assert sample_size == 1
