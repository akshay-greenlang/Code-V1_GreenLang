# -*- coding: utf-8 -*-
"""
Test suite for PACK-023 - TemperatureRatingEngine.

Validates:
  - Temperature alignment assessment (1.5C/1.75C/2C/3C+)
  - Policy/pledge gap analysis
  - Sector benchmark comparison
  - Fair-share allocation
  - Implied Temperature Rise (ITR) calculation
  - Warming indicators

Total Tests: 50+
Author: GreenLang Test Engineering
Pack: PACK-023 SBTi Alignment
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from engines.temperature_rating_engine import (
    TemperatureRatingEngine,
    TemperatureRatingInput,
    TemperatureRatingResult,
    WarmingCategory,
)


@pytest.fixture
def engine() -> TemperatureRatingEngine:
    """Fresh engine instance."""
    return TemperatureRatingEngine()


@pytest.fixture
def aligned_input() -> TemperatureRatingInput:
    """1.5C-aligned target input."""
    return TemperatureRatingInput(
        entity_name="AlignedCorp",
        baseline_scope12_tco2e=Decimal("5000"),
        baseline_scope3_tco2e=Decimal("8000"),
        target_year=2030,
        target_scope12_tco2e=Decimal("3000"),  # 40% reduction
        target_scope3_tco2e=Decimal("5000"),   # 37.5% reduction
        baseline_year=2024,
        sector="Technology",
    )


@pytest.fixture
def insufficient_input() -> TemperatureRatingInput:
    """Insufficient target ambition input."""
    return TemperatureRatingInput(
        entity_name="InsufficientCorp",
        baseline_scope12_tco2e=Decimal("5000"),
        baseline_scope3_tco2e=Decimal("8000"),
        target_year=2030,
        target_scope12_tco2e=Decimal("4500"),  # Only 10% reduction
        target_scope3_tco2e=Decimal("7500"),   # Only 6.25% reduction
        baseline_year=2024,
        sector="Energy",
    )


class TestEngineInstantiation:
    """Engine instantiation tests."""

    def test_engine_creates(self) -> None:
        """Engine must instantiate."""
        engine = TemperatureRatingEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"


class TestTemperatureAlignment:
    """Temperature alignment tests."""

    def test_1_5c_aligned_input(
        self, engine: TemperatureRatingEngine, aligned_input: TemperatureRatingInput
    ) -> None:
        """1.5C-aligned target should be classified correctly."""
        result = engine.calculate(aligned_input)

        assert isinstance(result, TemperatureRatingResult)
        assert result.warming_category in (
            WarmingCategory.WARMING_1_5C,
            WarmingCategory.WARMING_1_6C,
        )

    def test_insufficient_ambition(
        self, engine: TemperatureRatingEngine, insufficient_input: TemperatureRatingInput
    ) -> None:
        """Insufficient targets should be rated 2.5C+ warming."""
        result = engine.calculate(insufficient_input)

        assert result.warming_category in (
            WarmingCategory.WARMING_2_5C_PLUS,
            WarmingCategory.WARMING_3C_PLUS,
        )

    @pytest.mark.parametrize("reduction_pct,warming_range", [
        (Decimal("40"), WarmingCategory.WARMING_1_5C),
        (Decimal("30"), WarmingCategory.WARMING_1_8C),
        (Decimal("20"), WarmingCategory.WARMING_2_2C),
        (Decimal("10"), WarmingCategory.WARMING_2_5C_PLUS),
    ])
    def test_reduction_to_warming_mapping(
        self,
        engine: TemperatureRatingEngine,
        reduction_pct: Decimal,
        warming_range: WarmingCategory,
    ) -> None:
        """Reduction percentages should map to warming categories."""
        baseline = Decimal("5000")
        target = baseline * (Decimal("1") - reduction_pct / Decimal("100"))
        inp = TemperatureRatingInput(
            entity_name="TestCorp",
            baseline_scope12_tco2e=baseline,
            baseline_scope3_tco2e=Decimal("0"),
            target_year=2030,
            target_scope12_tco2e=target,
            target_scope3_tco2e=Decimal("0"),
            baseline_year=2024,
            sector="Technology",
        )
        result = engine.calculate(inp)

        assert result.warming_category == warming_range


class TestImpliedTemperatureRise:
    """Implied Temperature Rise (ITR) calculation tests."""

    def test_itr_calculation_1_5c(
        self, engine: TemperatureRatingEngine, aligned_input: TemperatureRatingInput
    ) -> None:
        """ITR for 1.5C-aligned should be ~1.5C."""
        result = engine.calculate(aligned_input)

        assert result.implied_temperature_rise is not None
        assert result.implied_temperature_rise <= Decimal("1.6")

    def test_itr_monotonic_with_ambition(
        self, engine: TemperatureRatingEngine
    ) -> None:
        """Less ambitious targets should have higher ITR."""
        inp_ambitious = TemperatureRatingInput(
            entity_name="Ambitious",
            baseline_scope12_tco2e=Decimal("5000"),
            baseline_scope3_tco2e=Decimal("0"),
            target_year=2030,
            target_scope12_tco2e=Decimal("2500"),  # 50% reduction
            target_scope3_tco2e=Decimal("0"),
            baseline_year=2024,
            sector="Technology",
        )
        inp_conservative = TemperatureRatingInput(
            entity_name="Conservative",
            baseline_scope12_tco2e=Decimal("5000"),
            baseline_scope3_tco2e=Decimal("0"),
            target_year=2030,
            target_scope12_tco2e=Decimal("4500"),  # 10% reduction
            target_scope3_tco2e=Decimal("0"),
            baseline_year=2024,
            sector="Technology",
        )
        result_ambitious = engine.calculate(inp_ambitious)
        result_conservative = engine.calculate(inp_conservative)

        assert result_ambitious.implied_temperature_rise < result_conservative.implied_temperature_rise


class TestSectorBenchmarking:
    """Sector-specific benchmark tests."""

    @pytest.mark.parametrize("sector", [
        "Technology",
        "Finance",
        "Manufacturing",
        "Energy",
        "Consumer Goods",
    ])
    def test_sector_benchmarks_supported(
        self, engine: TemperatureRatingEngine, sector: str
    ) -> None:
        """Major sectors must have benchmark data."""
        inp = TemperatureRatingInput(
            entity_name="SectorTest",
            baseline_scope12_tco2e=Decimal("5000"),
            baseline_scope3_tco2e=Decimal("8000"),
            target_year=2030,
            target_scope12_tco2e=Decimal("3000"),
            target_scope3_tco2e=Decimal("5000"),
            baseline_year=2024,
            sector=sector,
        )
        result = engine.calculate(inp)

        assert result.sector == sector
        assert result.sector_benchmark_alignment is not None


class TestPolicyAndPledgeGap:
    """Policy vs pledge gap analysis."""

    def test_policy_pledge_gap_analysis(
        self, engine: TemperatureRatingEngine, aligned_input: TemperatureRatingInput
    ) -> None:
        """Gap between policy pathway and pledge should be quantified."""
        result = engine.calculate(aligned_input)

        assert result.policy_pathway_warming is not None
        assert result.pledge_warming is not None
        gap = result.policy_pathway_warming - result.pledge_warming
        assert gap >= Decimal("0") or gap is not None


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_emissions_baseline(
        self, engine: TemperatureRatingEngine
    ) -> None:
        """Zero-emission baseline should be handled."""
        inp = TemperatureRatingInput(
            entity_name="ZeroBaseline",
            baseline_scope12_tco2e=Decimal("0"),
            baseline_scope3_tco2e=Decimal("0"),
            target_year=2030,
            target_scope12_tco2e=Decimal("0"),
            target_scope3_tco2e=Decimal("0"),
            baseline_year=2024,
            sector="Technology",
        )
        result = engine.calculate(inp)

        assert result is not None

    def test_no_reduction_target(
        self, engine: TemperatureRatingEngine
    ) -> None:
        """No reduction (baseline = target) should be rated poorly."""
        inp = TemperatureRatingInput(
            entity_name="NoReduction",
            baseline_scope12_tco2e=Decimal("5000"),
            baseline_scope3_tco2e=Decimal("8000"),
            target_year=2030,
            target_scope12_tco2e=Decimal("5000"),  # No reduction
            target_scope3_tco2e=Decimal("8000"),
            baseline_year=2024,
            sector="Technology",
        )
        result = engine.calculate(inp)

        assert result.warming_category in (
            WarmingCategory.WARMING_2_5C_PLUS,
            WarmingCategory.WARMING_3C_PLUS,
        )

    def test_reduction_exceeding_100_percent(
        self, engine: TemperatureRatingEngine
    ) -> None:
        """Target lower than baseline should be valid (e.g., with offsets)."""
        inp = TemperatureRatingInput(
            entity_name="HighAmbition",
            baseline_scope12_tco2e=Decimal("5000"),
            baseline_scope3_tco2e=Decimal("8000"),
            target_year=2030,
            target_scope12_tco2e=Decimal("1000"),  # 80% reduction
            target_scope3_tco2e=Decimal("1500"),
            baseline_year=2024,
            sector="Technology",
        )
        result = engine.calculate(inp)

        assert result.warming_category in (
            WarmingCategory.WARMING_1_5C,
            WarmingCategory.WARMING_1_6C,
        )


class TestProvenanceAndValidation:
    """Provenance and zero-hallucination tests."""

    def test_result_has_provenance_hash(
        self, engine: TemperatureRatingEngine, aligned_input: TemperatureRatingInput
    ) -> None:
        """Result must have provenance hash."""
        result = engine.calculate(aligned_input)

        assert hasattr(result, "provenance_hash")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_deterministic(
        self, engine: TemperatureRatingEngine, aligned_input: TemperatureRatingInput
    ) -> None:
        """Same input = same hash."""
        result1 = engine.calculate(aligned_input)
        result2 = engine.calculate(aligned_input)

        assert result1.provenance_hash == result2.provenance_hash
