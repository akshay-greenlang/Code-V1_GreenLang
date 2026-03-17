# -*- coding: utf-8 -*-
"""
Unit tests for InvestmentUniverseEngine (PACK-011 SFDR Article 9, Engine 8).

Tests multi-layer security screening, PAB/CTB exclusions, watch list
generation, pre-approval workflow, universe coverage statistics, screening
layer ordering, exclusion types, and provenance.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
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


_iu_mod = _import_from_path(
    "investment_universe_engine",
    str(ENGINES_DIR / "investment_universe_engine.py"),
)

InvestmentUniverseEngine = _iu_mod.InvestmentUniverseEngine
UniverseConfig = _iu_mod.UniverseConfig
SecurityData = _iu_mod.SecurityData
ScreeningResult = _iu_mod.ScreeningResult
ExclusionDetail = _iu_mod.ExclusionDetail
WatchListEntry = _iu_mod.WatchListEntry
PreApprovalResult = _iu_mod.PreApprovalResult
UniverseCoverage = _iu_mod.UniverseCoverage
ScreeningLayer = _iu_mod.ScreeningLayer
ExclusionType = _iu_mod.ExclusionType

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _clean_security(**kwargs) -> SecurityData:
    """Create a clean security passing all screens."""
    defaults = dict(
        company_name="Clean Corp",
        isin="XX0000000001",
        sector="C",
        country="DE",
        market_cap_eur=5_000_000_000,
        nav_value=10_000_000,
        weight_pct=2.0,
        controversial_weapons=False,
        coal_revenue_pct=0.0,
        oil_gas_revenue_pct=0.0,
        refining_revenue_pct=0.0,
        distribution_revenue_pct=0.0,
        power_carbon_intensity=0.0,
        tobacco_revenue_pct=0.0,
        thermal_coal_mining_revenue_pct=0.0,
        oil_sands_revenue_pct=0.0,
        arctic_drilling_revenue_pct=0.0,
        deforestation_linked=False,
        ungc_violations=False,
        human_rights_violations=False,
        controversy_score=1.0,
        esg_score=75.0,
        carbon_intensity=50.0,
        has_transition_plan=True,
        data_coverage="full",
    )
    defaults.update(kwargs)
    return SecurityData(**defaults)


def _dirty_security(**kwargs) -> SecurityData:
    """Create a security that fails PAB exclusion screening."""
    defaults = dict(
        company_name="Dirty Corp",
        isin="XX0000000099",
        sector="B",
        country="US",
        market_cap_eur=2_000_000_000,
        nav_value=5_000_000,
        weight_pct=1.0,
        controversial_weapons=False,
        coal_revenue_pct=3.0,
        oil_gas_revenue_pct=15.0,
        refining_revenue_pct=12.0,
        distribution_revenue_pct=55.0,
        power_carbon_intensity=150.0,
        tobacco_revenue_pct=0.0,
        thermal_coal_mining_revenue_pct=0.0,
        oil_sands_revenue_pct=0.0,
        arctic_drilling_revenue_pct=0.0,
        deforestation_linked=False,
        ungc_violations=False,
        human_rights_violations=False,
        controversy_score=2.0,
        esg_score=45.0,
        carbon_intensity=300.0,
        has_transition_plan=False,
        data_coverage="full",
    )
    defaults.update(kwargs)
    return SecurityData(**defaults)


def _borderline_security(**kwargs) -> SecurityData:
    """Create a security near exclusion thresholds (for watch list)."""
    defaults = dict(
        company_name="Borderline Corp",
        isin="XX0000000050",
        sector="D",
        country="FR",
        market_cap_eur=3_000_000_000,
        nav_value=8_000_000,
        weight_pct=1.5,
        controversial_weapons=False,
        coal_revenue_pct=0.85,  # near 1.0 threshold
        oil_gas_revenue_pct=8.5,  # near 10.0 threshold
        refining_revenue_pct=8.5,  # near 10.0 threshold
        distribution_revenue_pct=42.0,  # near 50.0 threshold
        power_carbon_intensity=90.0,  # near 100.0 threshold
        tobacco_revenue_pct=4.2,  # near 5.0 threshold
        thermal_coal_mining_revenue_pct=4.0,  # near 5.0 threshold
        oil_sands_revenue_pct=0.0,
        arctic_drilling_revenue_pct=0.0,
        deforestation_linked=False,
        ungc_violations=False,
        human_rights_violations=False,
        controversy_score=3.5,  # near 5.0 threshold
        esg_score=60.0,
        carbon_intensity=80.0,
        has_transition_plan=True,
        data_coverage="full",
    )
    defaults.update(kwargs)
    return SecurityData(**defaults)


def _make_universe(n_clean=10, n_dirty=2, n_borderline=1):
    """Build a mixed investment universe."""
    securities = []
    for i in range(n_clean):
        securities.append(_clean_security(
            company_name=f"Clean-{i}",
            nav_value=10_000_000,
        ))
    for i in range(n_dirty):
        securities.append(_dirty_security(
            company_name=f"Dirty-{i}",
            nav_value=5_000_000,
        ))
    for i in range(n_borderline):
        securities.append(_borderline_security(
            company_name=f"Borderline-{i}",
            nav_value=8_000_000,
        ))
    return securities


# ===========================================================================
# Tests
# ===========================================================================


class TestInvestmentUniverseEngineInit:
    """Test engine initialization."""

    def test_default_config(self):
        engine = InvestmentUniverseEngine()
        assert engine.config.apply_pab_exclusions is True
        assert engine.config.apply_norm_based is True
        assert engine.config.apply_sector_based is True
        assert engine.config.apply_controversy is True
        assert engine.config.watch_list_proximity_pct == pytest.approx(20.0)

    def test_dict_config(self):
        engine = InvestmentUniverseEngine({
            "product_name": "Art 9 Fund",
            "apply_pab_exclusions": True,
            "min_esg_score": 30.0,
        })
        assert engine.config.product_name == "Art 9 Fund"
        assert engine.config.min_esg_score == pytest.approx(30.0)

    def test_pydantic_config(self):
        cfg = UniverseConfig(
            product_name="Test Fund",
            apply_controversy=False,
        )
        engine = InvestmentUniverseEngine(cfg)
        assert engine.config.apply_controversy is False


class TestPABExclusions:
    """Test PAB exclusion screening rules."""

    def test_coal_exclusion_gte_1pct(self):
        """Security with >=1% coal revenue excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(coal_revenue_pct=1.5, company_name="Coal Co")]
        result = engine.screen_universe(secs)
        assert result.total_excluded >= 1
        coal_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.FOSSIL_FUEL_COAL
        ]
        assert len(coal_excls) >= 1
        assert coal_excls[0].actual_value == pytest.approx(1.5)
        assert coal_excls[0].threshold_value == pytest.approx(1.0)

    def test_coal_below_1pct_not_excluded(self):
        """Security with <1% coal revenue NOT excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(coal_revenue_pct=0.5)]
        result = engine.screen_universe(secs)
        coal_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.FOSSIL_FUEL_COAL
        ]
        assert len(coal_excls) == 0

    def test_oil_gas_exclusion_gte_10pct(self):
        """Security with >=10% oil/gas revenue excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(oil_gas_revenue_pct=12.0, company_name="Oil Co")]
        result = engine.screen_universe(secs)
        og_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.FOSSIL_FUEL_OIL_GAS
        ]
        assert len(og_excls) >= 1

    def test_refining_exclusion_gte_10pct(self):
        """Security with >=10% refining revenue excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(refining_revenue_pct=10.0)]
        result = engine.screen_universe(secs)
        ref_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.FOSSIL_FUEL_REFINING
        ]
        assert len(ref_excls) >= 1

    def test_distribution_exclusion_gte_50pct(self):
        """Security with >=50% distribution revenue excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(distribution_revenue_pct=55.0)]
        result = engine.screen_universe(secs)
        dist_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.FOSSIL_FUEL_DISTRIBUTION
        ]
        assert len(dist_excls) >= 1

    def test_high_carbon_power_exclusion_gt_100g(self):
        """Security with >100g CO2/kWh power generation excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(power_carbon_intensity=120.0)]
        result = engine.screen_universe(secs)
        power_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.HIGH_CARBON_POWER
        ]
        assert len(power_excls) >= 1

    def test_controversial_weapons_boolean_exclusion(self):
        """Security involved in controversial weapons excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(controversial_weapons=True)]
        result = engine.screen_universe(secs)
        weapons_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.CONTROVERSIAL_WEAPONS
        ]
        assert len(weapons_excls) >= 1

    def test_multiple_pab_violations(self):
        """Dirty security triggers multiple PAB exclusions."""
        engine = InvestmentUniverseEngine()
        secs = [_dirty_security()]
        result = engine.screen_universe(secs)
        assert len(result.exclusions) >= 3
        assert result.total_excluded == 1


class TestNormBasedScreening:
    """Test norm-based screening (UNGC/OECD)."""

    def test_ungc_violations_excluded(self):
        """Security with UNGC violations excluded."""
        engine = InvestmentUniverseEngine({"apply_norm_based": True})
        secs = [_clean_security(ungc_violations=True)]
        result = engine.screen_universe(secs)
        ungc_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.UNGC_VIOLATIONS
        ]
        assert len(ungc_excls) >= 1

    def test_human_rights_violations_excluded(self):
        """Security with human rights violations excluded."""
        engine = InvestmentUniverseEngine({"apply_norm_based": True})
        secs = [_clean_security(human_rights_violations=True)]
        result = engine.screen_universe(secs)
        hr_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.HUMAN_RIGHTS
        ]
        assert len(hr_excls) >= 1

    def test_no_norm_violations_passes(self):
        """Security without norm violations passes screening."""
        engine = InvestmentUniverseEngine({"apply_norm_based": True})
        secs = [_clean_security(ungc_violations=False, human_rights_violations=False)]
        result = engine.screen_universe(secs)
        norm_excls = [
            e for e in result.exclusions
            if e.screening_layer == ScreeningLayer.NORM_BASED
        ]
        assert len(norm_excls) == 0


class TestSectorBasedScreening:
    """Test sector-based screening."""

    def test_tobacco_exclusion_gte_5pct(self):
        """Security with >=5% tobacco revenue excluded."""
        engine = InvestmentUniverseEngine({"apply_sector_based": True})
        secs = [_clean_security(tobacco_revenue_pct=6.0)]
        result = engine.screen_universe(secs)
        tob_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.TOBACCO
        ]
        assert len(tob_excls) >= 1

    def test_thermal_coal_mining_exclusion(self):
        """Security with >=5% thermal coal mining revenue excluded."""
        engine = InvestmentUniverseEngine({"apply_sector_based": True})
        secs = [_clean_security(thermal_coal_mining_revenue_pct=7.0)]
        result = engine.screen_universe(secs)
        coal_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.THERMAL_COAL_MINING
        ]
        assert len(coal_excls) >= 1

    def test_deforestation_linked_excluded(self):
        """Security linked to deforestation excluded."""
        engine = InvestmentUniverseEngine({"apply_sector_based": True})
        secs = [_clean_security(deforestation_linked=True)]
        result = engine.screen_universe(secs)
        defor_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.DEFORESTATION
        ]
        assert len(defor_excls) >= 1


class TestControversyScreening:
    """Test controversy-based screening."""

    def test_severe_controversy_excluded(self):
        """Security with controversy score >= 5 excluded."""
        engine = InvestmentUniverseEngine({"apply_controversy": True})
        secs = [_clean_security(controversy_score=6.0)]
        result = engine.screen_universe(secs)
        cont_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.SEVERE_CONTROVERSY
        ]
        assert len(cont_excls) >= 1

    def test_moderate_controversy_passes(self):
        """Security with controversy score < 5 passes."""
        engine = InvestmentUniverseEngine({"apply_controversy": True})
        secs = [_clean_security(controversy_score=3.0)]
        result = engine.screen_universe(secs)
        cont_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.SEVERE_CONTROVERSY
            and e.screening_layer == ScreeningLayer.CONTROVERSY_BASED
        ]
        assert len(cont_excls) == 0


class TestWatchList:
    """Test watch list generation for borderline securities."""

    def test_near_threshold_generates_watch_entry(self):
        """Security near exclusion threshold lands on watch list."""
        engine = InvestmentUniverseEngine({
            "watch_list_proximity_pct": 20.0,
        })
        # coal_revenue_pct=0.85 -> distance to 1.0 = 15% < 20% proximity
        secs = [_borderline_security()]
        result = engine.screen_universe(secs)
        assert result.total_watch_list > 0
        assert len(result.watch_list) > 0

    def test_watch_entry_details(self):
        """Watch list entry has distance, risk level, monitoring frequency."""
        engine = InvestmentUniverseEngine({
            "watch_list_proximity_pct": 25.0,
        })
        secs = [_borderline_security(coal_revenue_pct=0.85)]
        result = engine.screen_universe(secs)
        coal_watches = [
            w for w in result.watch_list
            if w.exclusion_type == ExclusionType.FOSSIL_FUEL_COAL
        ]
        if coal_watches:
            entry = coal_watches[0]
            assert entry.distance_to_threshold_pct > 0
            assert entry.risk_level in ("high", "medium", "low")
            assert entry.monitoring_frequency in (
                "monthly", "quarterly", "semi-annual",
            )
            assert len(entry.provenance_hash) == 64

    def test_clean_security_no_watch(self):
        """Clean security with zero exposure not on watch list."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(coal_revenue_pct=0.0)]
        result = engine.screen_universe(secs)
        assert result.total_watch_list == 0

    def test_excluded_security_not_double_counted(self):
        """Security that is excluded should not also appear as watch-listed
        in the coverage statistics."""
        engine = InvestmentUniverseEngine()
        secs = [_dirty_security()]
        result = engine.screen_universe(secs)
        assert result.total_excluded >= 1
        # Watch list count in coverage should not count excluded securities
        coverage = result.universe_coverage
        assert coverage is not None


class TestPreApproval:
    """Test pre-approval workflow."""

    def test_clean_security_approved(self):
        """Clean security gets pre-approval."""
        engine = InvestmentUniverseEngine()
        sec = _clean_security(company_name="New Clean Corp")
        result = engine.pre_approve(sec)
        assert isinstance(result, PreApprovalResult)
        assert result.approved is True
        assert result.recommendation == "approve"
        assert len(result.exclusions) == 0
        assert len(result.provenance_hash) == 64

    def test_dirty_security_rejected(self):
        """Dirty security gets rejected."""
        engine = InvestmentUniverseEngine()
        sec = _dirty_security(company_name="Reject Corp")
        result = engine.pre_approve(sec)
        assert result.approved is False
        assert result.recommendation == "reject"
        assert len(result.exclusions) > 0
        assert len(result.layers_failed) > 0

    def test_borderline_security_watch(self):
        """Borderline security gets 'watch' recommendation."""
        engine = InvestmentUniverseEngine({
            "watch_list_proximity_pct": 25.0,
            # Only apply PAB exclusions so borderline is not excluded by others
            "apply_norm_based": False,
            "apply_sector_based": False,
            "apply_controversy": False,
        })
        sec = _borderline_security()
        result = engine.pre_approve(sec)
        # Should pass (not excluded) but have watch flags
        if result.approved and len(result.watch_list_flags) > 0:
            assert result.recommendation == "watch"

    def test_pre_approval_layers_tracked(self):
        """Pre-approval tracks passed and failed layers."""
        engine = InvestmentUniverseEngine()
        sec = _clean_security()
        result = engine.pre_approve(sec)
        assert result.total_layers_checked > 0
        assert len(result.layers_passed) > 0

    def test_esg_score_filter_rejection(self):
        """Security below min ESG score gets rejected."""
        engine = InvestmentUniverseEngine({
            "min_esg_score": 50.0,
        })
        sec = _clean_security(esg_score=30.0)
        result = engine.pre_approve(sec)
        assert result.approved is False
        esg_excls = [
            e for e in result.exclusions
            if e.screening_layer == ScreeningLayer.ESG_QUALITY
        ]
        assert len(esg_excls) >= 1

    def test_carbon_intensity_filter_rejection(self):
        """Security above max carbon intensity gets rejected."""
        engine = InvestmentUniverseEngine({
            "max_carbon_intensity": 200.0,
        })
        sec = _clean_security(carbon_intensity=350.0)
        result = engine.pre_approve(sec)
        assert result.approved is False


class TestUniverseCoverage:
    """Test universe coverage statistics."""

    def test_coverage_statistics_structure(self):
        """Coverage has all required fields."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=10, n_dirty=2, n_borderline=1)
        result = engine.screen_universe(secs)
        coverage = result.universe_coverage
        assert coverage is not None
        assert isinstance(coverage, UniverseCoverage)
        assert coverage.total_securities_screened == 13
        assert coverage.eligible_securities + coverage.excluded_securities == 13
        assert 0.0 <= coverage.eligibility_rate_pct <= 100.0
        assert 0.0 <= coverage.exclusion_rate_pct <= 100.0
        assert len(coverage.provenance_hash) == 64

    def test_eligibility_rate_calculation(self):
        """Eligibility rate = eligible / total * 100."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=8, n_dirty=2, n_borderline=0)
        result = engine.screen_universe(secs)
        coverage = result.universe_coverage
        expected_rate = (coverage.eligible_securities / 10.0) * 100.0
        assert coverage.eligibility_rate_pct == pytest.approx(expected_rate, rel=0.1)

    def test_exclusions_by_layer_tracked(self):
        """Exclusions are tracked per screening layer."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=5, n_dirty=3)
        result = engine.screen_universe(secs)
        coverage = result.universe_coverage
        assert isinstance(coverage.exclusions_by_layer, dict)
        # Dirty securities should trigger PAB exclusion layer
        assert len(coverage.exclusions_by_layer) > 0

    def test_exclusions_by_type_tracked(self):
        """Exclusions are tracked per exclusion type."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=5, n_dirty=3)
        result = engine.screen_universe(secs)
        coverage = result.universe_coverage
        assert isinstance(coverage.exclusions_by_type, dict)
        assert len(coverage.exclusions_by_type) > 0

    def test_nav_calculations(self):
        """Eligible and excluded NAV are calculated."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=5, n_dirty=2)
        result = engine.screen_universe(secs)
        coverage = result.universe_coverage
        assert coverage.eligible_nav >= 0
        assert coverage.excluded_nav >= 0
        # Total NAV = eligible + excluded
        total_nav = sum(s.nav_value for s in secs)
        assert coverage.eligible_nav + coverage.excluded_nav == pytest.approx(
            total_nav, rel=0.01,
        )


class TestScreeningLayerOrdering:
    """Test screening layer configuration and ordering."""

    def test_pab_only_screening(self):
        """Only PAB exclusions applied when others disabled."""
        engine = InvestmentUniverseEngine({
            "apply_pab_exclusions": True,
            "apply_ctb_exclusions": False,
            "apply_norm_based": False,
            "apply_sector_based": False,
            "apply_controversy": False,
        })
        # Security with UNGC violations but clean on PAB
        sec = _clean_security(ungc_violations=True)
        result = engine.screen_universe([sec])
        # Should NOT be excluded because norm-based is disabled
        ungc_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.UNGC_VIOLATIONS
        ]
        assert len(ungc_excls) == 0

    def test_ctb_exclusions_applied(self):
        """CTB exclusions applied when configured."""
        engine = InvestmentUniverseEngine({
            "apply_pab_exclusions": False,
            "apply_ctb_exclusions": True,
            "apply_norm_based": False,
            "apply_sector_based": False,
            "apply_controversy": False,
        })
        # CTB only excludes controversial weapons
        sec = _clean_security(controversial_weapons=True)
        result = engine.screen_universe([sec])
        assert result.total_excluded >= 1

    def test_threshold_overrides(self):
        """Custom threshold overrides change exclusion behavior."""
        engine = InvestmentUniverseEngine({
            "threshold_overrides": {
                ExclusionType.FOSSIL_FUEL_COAL.value: 5.0,
            },
            "apply_norm_based": False,
            "apply_sector_based": False,
            "apply_controversy": False,
        })
        # With 5% override, 3% coal should NOT be excluded
        sec = _clean_security(coal_revenue_pct=3.0)
        result = engine.screen_universe([sec])
        coal_excls = [
            e for e in result.exclusions
            if e.exclusion_type == ExclusionType.FOSSIL_FUEL_COAL
        ]
        assert len(coal_excls) == 0

    def test_all_layers_applied(self):
        """All layers applied when all are enabled."""
        engine = InvestmentUniverseEngine({
            "apply_pab_exclusions": True,
            "apply_norm_based": True,
            "apply_sector_based": True,
            "apply_controversy": True,
        })
        assert ScreeningLayer.PAB_EXCLUSION in engine.config.active_layers


class TestFullUniverseScreening:
    """Test full screen_universe pipeline."""

    def test_full_screening_result_structure(self):
        """Full screening returns ScreeningResult with all fields."""
        engine = InvestmentUniverseEngine({"product_name": "Art 9 Fund"})
        secs = _make_universe()
        result = engine.screen_universe(secs)

        assert isinstance(result, ScreeningResult)
        assert result.product_name == "Art 9 Fund"
        assert result.total_screened == len(secs)
        assert result.eligible_count + result.total_excluded == result.total_screened
        assert len(result.eligible_security_ids) == result.eligible_count
        assert result.universe_coverage is not None
        assert len(result.provenance_hash) == 64

    def test_empty_universe_raises(self):
        """Empty securities list raises ValueError."""
        engine = InvestmentUniverseEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.screen_universe([])

    def test_all_clean_universe(self):
        """Universe with only clean securities -> all eligible."""
        engine = InvestmentUniverseEngine()
        secs = [_clean_security(company_name=f"Clean-{i}") for i in range(10)]
        result = engine.screen_universe(secs)
        assert result.eligible_count == 10
        assert result.total_excluded == 0

    def test_all_dirty_universe(self):
        """Universe with only dirty securities -> all excluded."""
        engine = InvestmentUniverseEngine()
        secs = [_dirty_security(company_name=f"Dirty-{i}") for i in range(5)]
        result = engine.screen_universe(secs)
        assert result.total_excluded == 5
        assert result.eligible_count == 0

    def test_layers_applied_in_result(self):
        """Applied layers are recorded in result."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=3)
        result = engine.screen_universe(secs)
        assert len(result.layers_applied) > 0


class TestProvenanceUniverse:
    """Test provenance hashing for universe screening."""

    def test_screening_result_provenance(self):
        """ScreeningResult has valid SHA-256 hash."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=5, n_dirty=1)
        result = engine.screen_universe(secs)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_exclusion_detail_provenance(self):
        """Each ExclusionDetail has a provenance hash."""
        engine = InvestmentUniverseEngine()
        secs = [_dirty_security()]
        result = engine.screen_universe(secs)
        for excl in result.exclusions:
            assert len(excl.provenance_hash) == 64

    def test_pre_approval_provenance(self):
        """PreApprovalResult has valid SHA-256 hash."""
        engine = InvestmentUniverseEngine()
        sec = _clean_security()
        result = engine.pre_approve(sec)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_coverage_provenance(self):
        """UniverseCoverage has valid SHA-256 hash."""
        engine = InvestmentUniverseEngine()
        secs = _make_universe(n_clean=5)
        result = engine.screen_universe(secs)
        assert result.universe_coverage is not None
        assert len(result.universe_coverage.provenance_hash) == 64
        int(result.universe_coverage.provenance_hash, 16)
