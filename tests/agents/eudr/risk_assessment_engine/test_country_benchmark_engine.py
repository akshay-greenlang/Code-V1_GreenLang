# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 CountryBenchmarkEngine.

Tests country lookup (known and unknown), batch retrieval, multiplier
lookups, low-risk classification, benchmark updates, and level-based
filtering including EU member state defaults.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    CountryBenchmark,
    CountryBenchmarkLevel,
    COUNTRY_BENCHMARK_MULTIPLIERS,
)


def _make_engine():
    """Instantiate CountryBenchmarkEngine with mocked metrics."""
    from greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine import (
        CountryBenchmarkEngine,
    )
    cfg = MagicMock(spec=RiskAssessmentEngineConfig)
    cfg.benchmark_low_multiplier = Decimal("0.70")
    cfg.benchmark_standard_multiplier = Decimal("1.00")
    cfg.benchmark_high_multiplier = Decimal("1.50")
    with patch(
        "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
    ):
        return CountryBenchmarkEngine(config=cfg)


class TestGetBenchmark:
    """Test single-country benchmark lookup."""

    def test_get_benchmark_known_country_low(self):
        """DE (Germany) should be classified as LOW risk."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            bm = engine.get_benchmark("DE")
        assert bm.level == CountryBenchmarkLevel.LOW

    def test_get_benchmark_known_country_high(self):
        """BR (Brazil) should be classified as HIGH risk."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            bm = engine.get_benchmark("BR")
        assert bm.level == CountryBenchmarkLevel.HIGH

    def test_get_benchmark_known_country_standard(self):
        """CN (China) should be classified as STANDARD risk."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            bm = engine.get_benchmark("CN")
        assert bm.level == CountryBenchmarkLevel.STANDARD

    def test_get_benchmark_unknown_country(self):
        """Unknown country code should default to STANDARD."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            bm = engine.get_benchmark("ZZ")
        assert bm.level == CountryBenchmarkLevel.STANDARD

    def test_get_benchmark_case_insensitive(self):
        """Country code lookup should be case-insensitive."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            bm = engine.get_benchmark("de")
        assert bm.level == CountryBenchmarkLevel.LOW


class TestGetBenchmarksBatch:
    """Test batch country benchmark retrieval."""

    def test_get_benchmarks_batch(self):
        """Batch lookup should return one benchmark per input code."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            benchmarks = engine.get_benchmarks(["DE", "BR", "ZZ"])
        assert len(benchmarks) == 3


class TestGetBenchmarkMultiplier:
    """Test multiplier lookups."""

    def test_get_benchmark_multiplier_low(self):
        """LOW countries should have 0.70 multiplier."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            multiplier = engine.get_benchmark_multiplier("DE")
        assert multiplier == Decimal("0.70")

    def test_get_benchmark_multiplier_standard(self):
        """STANDARD countries should have 1.00 multiplier."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            multiplier = engine.get_benchmark_multiplier("CN")
        assert multiplier == Decimal("1.00")

    def test_get_benchmark_multiplier_high(self):
        """HIGH countries should have 1.50 multiplier."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            multiplier = engine.get_benchmark_multiplier("BR")
        assert multiplier == Decimal("1.50")


class TestIsLowRiskCountry:
    """Test low-risk classification checks."""

    def test_is_low_risk_country_true(self):
        """EU member state should be classified as low risk."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            assert engine.is_low_risk_country("DE") is True
            assert engine.is_low_risk_country("FR") is True
            assert engine.is_low_risk_country("SE") is True

    def test_is_low_risk_country_false(self):
        """HIGH-deforestation country should not be low risk."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            assert engine.is_low_risk_country("BR") is False
            assert engine.is_low_risk_country("ID") is False


class TestUpdateBenchmarks:
    """Test benchmark update capability."""

    def test_update_benchmarks(self):
        """Updating a country benchmark should change its level."""
        engine = _make_engine()
        # Verify initial
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            assert engine.get_benchmark("BR").level == CountryBenchmarkLevel.HIGH

        # Update BR to LOW
        new_bm = CountryBenchmark(
            country_code="BR",
            level=CountryBenchmarkLevel.LOW,
            multiplier=Decimal("0.70"),
            source="test_update",
        )
        updated_count = engine.update_benchmarks([new_bm])
        assert updated_count >= 1

        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine.record_benchmark_lookup"
        ):
            assert engine.get_benchmark("BR").level == CountryBenchmarkLevel.LOW


class TestGetCountriesByLevel:
    """Test level-based filtering."""

    def test_get_countries_by_level(self):
        """Should return sorted country codes for a given level."""
        engine = _make_engine()
        low_countries = engine.get_countries_by_level(CountryBenchmarkLevel.LOW)
        assert isinstance(low_countries, list)
        assert "DE" in low_countries
        assert "FR" in low_countries

        high_countries = engine.get_countries_by_level(CountryBenchmarkLevel.HIGH)
        assert "BR" in high_countries

    def test_eu_members_are_low_risk(self):
        """All EU-27 member states should be classified as LOW."""
        engine = _make_engine()
        eu_members = [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        ]
        low_countries = engine.get_countries_by_level(CountryBenchmarkLevel.LOW)
        for code in eu_members:
            assert code in low_countries, f"{code} is not classified as LOW risk"
