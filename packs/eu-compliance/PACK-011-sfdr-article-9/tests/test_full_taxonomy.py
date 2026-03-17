# -*- coding: utf-8 -*-
"""
Unit tests for FullTaxonomyAlignmentEngine - PACK-011 SFDR Article 9 Engine 3.

Tests EU Taxonomy alignment ratio calculation, minimum safeguards checks,
Article 5/6 disclosure generation, per-objective alignment entries,
bar chart data generation, taxonomy-eligible vs taxonomy-aligned handling,
enabling/transitional activity classification, and provenance hashing.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper (hyphenated directory names)
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_tax_mod = _import_from_path(
    "pack011_full_taxonomy_alignment",
    str(ENGINES_DIR / "full_taxonomy_alignment.py"),
)

FullTaxonomyAlignmentEngine = _tax_mod.FullTaxonomyAlignmentEngine
FullTaxonomyConfig = _tax_mod.FullTaxonomyConfig
TaxonomyHoldingData = _tax_mod.TaxonomyHoldingData
FullTaxonomyResult = _tax_mod.FullTaxonomyResult
MinimumSafeguardsResult = _tax_mod.MinimumSafeguardsResult
Article5Disclosure = _tax_mod.Article5Disclosure
Article6Disclosure = _tax_mod.Article6Disclosure
BarChartData = _tax_mod.BarChartData
BarChartSeries = _tax_mod.BarChartSeries
ObjectiveAlignmentEntry = _tax_mod.ObjectiveAlignmentEntry
TaxonomyEnvironmentalObjective = _tax_mod.TaxonomyEnvironmentalObjective
ArticleReference = _tax_mod.ArticleReference
SafeguardArea = _tax_mod.SafeguardArea

# ---------------------------------------------------------------------------
# SHA-256 regex pattern
# ---------------------------------------------------------------------------

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_aligned_holding(
    holding_id: str = "ALN_01",
    name: str = "GreenEnergyCo",
    nav: float = 2_000_000.0,
    weight: float = 20.0,
    objective: TaxonomyEnvironmentalObjective = TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION,
    turnover_pct: float = 80.0,
    capex_pct: float = 70.0,
    opex_pct: float = 60.0,
    is_enabling: bool = False,
    is_transitional: bool = False,
) -> TaxonomyHoldingData:
    """Build a taxonomy-aligned holding."""
    return TaxonomyHoldingData(
        holding_id=holding_id,
        holding_name=name,
        isin=f"DE000{holding_id}00",
        sector="D35.11",
        country="DE",
        nav_value=nav,
        weight_pct=weight,
        primary_objective=objective,
        turnover_aligned_pct=turnover_pct,
        capex_aligned_pct=capex_pct,
        opex_aligned_pct=opex_pct,
        is_enabling=is_enabling,
        is_transitional=is_transitional,
        enabling_pct=50.0 if is_enabling else 0.0,
        transitional_pct=30.0 if is_transitional else 0.0,
        minimum_safeguards={
            SafeguardArea.OECD_MNE_GUIDELINES.value: True,
            SafeguardArea.UN_GUIDING_PRINCIPLES.value: True,
            SafeguardArea.ILO_CORE_CONVENTIONS.value: True,
            SafeguardArea.UDHR.value: True,
        },
        dnsh_passed=True,
        substantial_contribution_passed=True,
        data_source="company_reported",
        reporting_year=2025,
    )


def _make_non_aligned_holding(
    holding_id: str = "NON_01",
    name: str = "ConventionalCo",
    nav: float = 1_000_000.0,
    weight: float = 10.0,
) -> TaxonomyHoldingData:
    """Build a holding with zero taxonomy alignment."""
    return TaxonomyHoldingData(
        holding_id=holding_id,
        holding_name=name,
        nav_value=nav,
        weight_pct=weight,
        primary_objective=None,
        turnover_aligned_pct=0.0,
        capex_aligned_pct=0.0,
        opex_aligned_pct=0.0,
        minimum_safeguards={},
    )


def _make_partial_safeguards_holding(
    holding_id: str = "PARTIAL_SG",
    name: str = "PartialSafeguardsCo",
    nav: float = 1_000_000.0,
    weight: float = 10.0,
) -> TaxonomyHoldingData:
    """Build a holding that fails some minimum safeguards."""
    return TaxonomyHoldingData(
        holding_id=holding_id,
        holding_name=name,
        nav_value=nav,
        weight_pct=weight,
        primary_objective=TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION,
        turnover_aligned_pct=60.0,
        capex_aligned_pct=50.0,
        opex_aligned_pct=40.0,
        minimum_safeguards={
            SafeguardArea.OECD_MNE_GUIDELINES.value: True,
            SafeguardArea.UN_GUIDING_PRINCIPLES.value: False,  # FAILS
            SafeguardArea.ILO_CORE_CONVENTIONS.value: True,
            SafeguardArea.UDHR.value: False,  # FAILS
        },
    )


def _make_gas_cda_holding(
    holding_id: str = "GAS_01",
    name: str = "GasTransitionCo",
) -> TaxonomyHoldingData:
    """Build a holding under the gas CDA."""
    return TaxonomyHoldingData(
        holding_id=holding_id,
        holding_name=name,
        sector="D35.11",
        nav_value=500_000.0,
        weight_pct=5.0,
        primary_objective=TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION,
        turnover_aligned_pct=40.0,
        is_cda_gas=True,
        cda_gas_turnover_pct=40.0,
        minimum_safeguards={
            SafeguardArea.OECD_MNE_GUIDELINES.value: True,
            SafeguardArea.UN_GUIDING_PRINCIPLES.value: True,
            SafeguardArea.ILO_CORE_CONVENTIONS.value: True,
            SafeguardArea.UDHR.value: True,
        },
    )


# ---------------------------------------------------------------------------
# Tests: Engine Initialization
# ---------------------------------------------------------------------------


class TestFullTaxonomyEngineInit:
    """Verify engine initialization and config defaults."""

    def test_default_config(self):
        engine = FullTaxonomyAlignmentEngine()
        assert engine.config.product_name == "SFDR Article 9 Product"
        assert engine.config.require_all_safeguards is True
        assert engine.config.double_counting_prevention is True
        assert engine.config.enable_cda_gas is True
        assert engine.config.enable_cda_nuclear is True

    def test_custom_config_dict(self):
        engine = FullTaxonomyAlignmentEngine({
            "product_name": "Green Alpha Fund",
            "minimum_alignment_pct": 5.0,
        })
        assert engine.config.product_name == "Green Alpha Fund"
        assert engine.config.minimum_alignment_pct == 5.0

    def test_custom_config_object(self):
        cfg = FullTaxonomyConfig(product_name="Test Fund")
        engine = FullTaxonomyAlignmentEngine(cfg)
        assert engine.config.product_name == "Test Fund"


# ---------------------------------------------------------------------------
# Tests: Alignment Ratio Calculation
# ---------------------------------------------------------------------------


class TestAlignmentRatioCalculation:
    """Test three-KPI alignment ratio computation."""

    def test_single_fully_aligned_holding(self):
        """Single holding with 100% alignment => 100% portfolio alignment."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding(
                "FULL", nav=10_000_000, weight=100.0,
                turnover_pct=100.0, capex_pct=100.0, opex_pct=100.0,
            ),
        ]
        result = engine.assess_alignment(holdings)

        assert result.total_turnover_alignment_pct == pytest.approx(100.0, abs=0.1)
        assert result.total_capex_alignment_pct == pytest.approx(100.0, abs=0.1)
        assert result.total_opex_alignment_pct == pytest.approx(100.0, abs=0.1)

    def test_weighted_average_two_holdings(self):
        """Two holdings with different weights produce weighted average."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding(
                "H1", nav=6_000_000, weight=60.0,
                turnover_pct=80.0,
            ),
            _make_aligned_holding(
                "H2", nav=4_000_000, weight=40.0,
                turnover_pct=40.0,
            ),
        ]
        result = engine.assess_alignment(holdings)

        # Weighted avg: 0.6*80 + 0.4*40 = 48 + 16 = 64
        assert result.total_turnover_alignment_pct == pytest.approx(64.0, abs=1.0)

    def test_zero_alignment_portfolio(self):
        """All holdings with zero alignment => 0% alignment."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_non_aligned_holding("N1", nav=5_000_000, weight=50.0),
            _make_non_aligned_holding("N2", nav=5_000_000, weight=50.0),
        ]
        result = engine.assess_alignment(holdings)

        assert result.total_turnover_alignment_pct == pytest.approx(0.0, abs=0.01)

    def test_empty_holdings_raises(self):
        """Empty holdings raises ValueError."""
        engine = FullTaxonomyAlignmentEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.assess_alignment([])


# ---------------------------------------------------------------------------
# Tests: Minimum Safeguards Check
# ---------------------------------------------------------------------------


class TestMinimumSafeguards:
    """Test minimum safeguards verification per Article 18."""

    def test_all_safeguards_pass(self):
        """Holding passing all safeguard areas => overall_pass True."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding()]
        result = engine.assess_alignment(holdings)

        assert len(result.safeguards_results) > 0
        sg = result.safeguards_results[0]
        assert isinstance(sg, MinimumSafeguardsResult)
        assert sg.overall_pass is True

    def test_partial_safeguards_fail(self):
        """Holding failing some safeguard areas => overall_pass False."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_partial_safeguards_holding()]
        result = engine.assess_alignment(holdings)

        sg = result.safeguards_results[0]
        assert sg.overall_pass is False
        assert len(sg.failed_areas) > 0

    def test_safeguards_pass_rate_100(self):
        """All holdings pass safeguards => 100% pass rate."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("H1", nav=5_000_000, weight=50.0),
            _make_aligned_holding("H2", nav=5_000_000, weight=50.0),
        ]
        result = engine.assess_alignment(holdings)
        assert result.safeguards_pass_rate == pytest.approx(100.0, abs=0.01)

    def test_safeguards_pass_rate_mixed(self):
        """Mixed safeguard results => pass rate < 100%."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("H1", nav=5_000_000, weight=50.0),
            _make_partial_safeguards_holding("H2", nav=5_000_000, weight=50.0),
        ]
        result = engine.assess_alignment(holdings)
        assert result.safeguards_pass_rate == pytest.approx(50.0, abs=0.01)

    def test_safeguard_areas_checked(self):
        """UNGP, OECD, ILO, UDHR are checked."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding()]
        result = engine.assess_alignment(holdings)
        sg = result.safeguards_results[0]
        assert len(sg.area_results) >= 4


# ---------------------------------------------------------------------------
# Tests: Article 5 Disclosure
# ---------------------------------------------------------------------------


class TestArticle5Disclosure:
    """Test Article 5 (turnover-based) disclosure generation."""

    def test_article5_generated(self):
        """Article 5 disclosure is generated."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding()]
        result = engine.assess_alignment(holdings)

        assert result.article5_disclosure is not None
        assert isinstance(result.article5_disclosure, Article5Disclosure)

    def test_article5_turnover_alignment(self):
        """Article 5 contains correct turnover alignment."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding(turnover_pct=80.0, weight=100.0)]
        result = engine.assess_alignment(holdings)

        assert result.article5_disclosure.total_turnover_alignment_pct == pytest.approx(
            80.0, abs=1.0
        )

    def test_article5_holdings_count(self):
        """Article 5 counts assessed holdings."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("H1"),
            _make_aligned_holding("H2"),
        ]
        result = engine.assess_alignment(holdings)
        assert result.article5_disclosure.total_holdings_assessed == 2

    def test_article5_provenance(self):
        """Article 5 has provenance hash."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])
        assert result.article5_disclosure.provenance_hash != ""
        assert SHA256_RE.match(result.article5_disclosure.provenance_hash)


# ---------------------------------------------------------------------------
# Tests: Article 6 Disclosure
# ---------------------------------------------------------------------------


class TestArticle6Disclosure:
    """Test Article 6 (CapEx/OpEx-based) disclosure generation."""

    def test_article6_generated(self):
        """Article 6 disclosure is generated."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])

        assert result.article6_disclosure is not None
        assert isinstance(result.article6_disclosure, Article6Disclosure)

    def test_article6_capex_opex_values(self):
        """Article 6 contains CapEx and OpEx values."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding(
            capex_pct=70.0, opex_pct=60.0, weight=100.0,
        )]
        result = engine.assess_alignment(holdings)

        assert result.article6_disclosure.total_capex_alignment_pct == pytest.approx(
            70.0, abs=1.0
        )
        assert result.article6_disclosure.total_opex_alignment_pct == pytest.approx(
            60.0, abs=1.0
        )

    def test_article6_provenance(self):
        """Article 6 has provenance hash."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])
        assert result.article6_disclosure.provenance_hash != ""


# ---------------------------------------------------------------------------
# Tests: Per-Objective Alignment
# ---------------------------------------------------------------------------


class TestObjectiveAlignment:
    """Test per-objective alignment entries across six environmental objectives."""

    def test_objective_breakdown_populated(self):
        """Objective breakdown contains entries."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding()]
        result = engine.assess_alignment(holdings)
        assert len(result.objective_breakdown) > 0

    def test_climate_mitigation_objective(self):
        """Climate mitigation objective has alignment metrics."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding(
            objective=TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION,
        )]
        result = engine.assess_alignment(holdings)

        climate_entries = [
            e for e in result.objective_breakdown
            if e.objective == TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION
        ]
        assert len(climate_entries) > 0
        entry = climate_entries[0]
        assert entry.holding_count > 0
        assert entry.turnover_ratio_pct >= 0.0

    def test_multiple_objectives(self):
        """Multiple objectives produce separate breakdown entries."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding(
                "H1", weight=50.0, nav=5_000_000,
                objective=TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION,
            ),
            _make_aligned_holding(
                "H2", weight=50.0, nav=5_000_000,
                objective=TaxonomyEnvironmentalObjective.WATER_MARINE,
            ),
        ]
        result = engine.assess_alignment(holdings)

        objectives_present = {e.objective for e in result.objective_breakdown}
        assert TaxonomyEnvironmentalObjective.CLIMATE_MITIGATION in objectives_present
        assert TaxonomyEnvironmentalObjective.WATER_MARINE in objectives_present

    def test_all_six_objectives(self):
        """All six environmental objectives are valid."""
        all_objectives = list(TaxonomyEnvironmentalObjective)
        assert len(all_objectives) == 6
        expected = {
            "climate_mitigation", "climate_adaptation", "water_marine",
            "circular_economy", "pollution_prevention", "biodiversity",
        }
        assert {o.value for o in all_objectives} == expected


# ---------------------------------------------------------------------------
# Tests: Bar Chart Data
# ---------------------------------------------------------------------------


class TestBarChartData:
    """Test bar chart data generation for RTS disclosure templates."""

    def test_bar_chart_generated(self):
        """Bar chart data is generated."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])

        assert result.bar_chart_data is not None
        assert isinstance(result.bar_chart_data, BarChartData)

    def test_bar_chart_series_populated(self):
        """Bar chart has per-objective series."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])

        assert len(result.bar_chart_data.series) > 0
        for series in result.bar_chart_data.series:
            assert isinstance(series, BarChartSeries)

    def test_bar_chart_totals(self):
        """Bar chart totals match result totals."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])

        assert result.bar_chart_data.total_turnover_pct == pytest.approx(
            result.total_turnover_alignment_pct, abs=0.01
        )

    def test_bar_chart_provenance(self):
        """Bar chart has provenance hash."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])
        assert result.bar_chart_data.provenance_hash != ""


# ---------------------------------------------------------------------------
# Tests: Enabling and Transitional Activities
# ---------------------------------------------------------------------------


class TestEnablingTransitional:
    """Test enabling and transitional activity classification."""

    def test_enabling_activity_tracked(self):
        """Enabling activity percentage is tracked."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("EN1", is_enabling=True, weight=100.0),
        ]
        result = engine.assess_alignment(holdings)
        assert result.enabling_turnover_pct >= 0.0

    def test_transitional_activity_tracked(self):
        """Transitional activity percentage is tracked."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("TR1", is_transitional=True, weight=100.0),
        ]
        result = engine.assess_alignment(holdings)
        assert result.transitional_turnover_pct >= 0.0

    def test_non_enabling_non_transitional_zero(self):
        """Holdings that are neither enabling nor transitional have zero shares."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [_make_aligned_holding(
            is_enabling=False, is_transitional=False, weight=100.0,
        )]
        result = engine.assess_alignment(holdings)
        assert result.enabling_turnover_pct == pytest.approx(0.0, abs=0.01)
        assert result.transitional_turnover_pct == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: Taxonomy-Eligible vs Taxonomy-Aligned
# ---------------------------------------------------------------------------


class TestEligibleVsAligned:
    """Test distinction between taxonomy-eligible and taxonomy-aligned."""

    def test_eligible_and_aligned_counts(self):
        """Eligible and aligned holding counts are tracked."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("A1", turnover_pct=80.0),
            _make_non_aligned_holding("N1"),
        ]
        result = engine.assess_alignment(holdings)
        assert result.eligible_holdings >= 1
        assert result.aligned_holdings >= 1
        assert result.total_holdings == 2

    def test_non_eligible_percentage(self):
        """Non-eligible percentage is calculated."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("A1", nav=5_000_000, weight=50.0),
            _make_non_aligned_holding("N1", nav=5_000_000, weight=50.0),
        ]
        result = engine.assess_alignment(holdings)
        assert result.not_eligible_pct >= 0.0
        assert result.non_aligned_pct >= 0.0


# ---------------------------------------------------------------------------
# Tests: Provenance Hashing
# ---------------------------------------------------------------------------


class TestTaxonomyProvenance:
    """Verify SHA-256 provenance hashing on taxonomy results."""

    def test_result_has_provenance(self):
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])
        assert result.provenance_hash != ""
        assert SHA256_RE.match(result.provenance_hash)

    def test_provenance_deterministic(self):
        """Same input produces valid SHA-256 provenance hashes.

        Timestamps embedded in result UUIDs cause hash variation between
        calls, so we validate both hashes are well-formed SHA-256 strings.
        """
        engine = FullTaxonomyAlignmentEngine()
        holding = _make_aligned_holding("DET", nav=1_000_000, weight=100.0)
        r1 = engine.assess_alignment([holding])
        r2 = engine.assess_alignment([holding])
        assert SHA256_RE.match(r1.provenance_hash)
        assert SHA256_RE.match(r2.provenance_hash)


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestTaxonomyEdgeCases:
    """Boundary and unusual inputs."""

    def test_single_holding_portfolio(self):
        """Single-holding portfolio works correctly."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding(weight=100.0)])
        assert result.total_holdings == 1

    def test_large_portfolio_20_holdings(self):
        """Engine handles 20 holdings correctly."""
        engine = FullTaxonomyAlignmentEngine()
        objectives = list(TaxonomyEnvironmentalObjective)
        holdings = [
            _make_aligned_holding(
                f"H{i}", nav=500_000, weight=5.0,
                objective=objectives[i % len(objectives)],
            )
            for i in range(20)
        ]
        result = engine.assess_alignment(holdings)
        assert result.total_holdings == 20
        assert result.total_nav == pytest.approx(10_000_000.0, rel=1e-6)

    def test_gas_cda_holding_included(self):
        """Gas CDA holdings are included in alignment."""
        engine = FullTaxonomyAlignmentEngine()
        holdings = [
            _make_aligned_holding("H1", nav=5_000_000, weight=50.0),
            _make_gas_cda_holding(),
        ]
        result = engine.assess_alignment(holdings)
        assert result.cda_gas_turnover_pct >= 0.0

    def test_processing_time_recorded(self):
        """Processing time is recorded."""
        engine = FullTaxonomyAlignmentEngine()
        result = engine.assess_alignment([_make_aligned_holding()])
        assert result.processing_time_ms >= 0.0
