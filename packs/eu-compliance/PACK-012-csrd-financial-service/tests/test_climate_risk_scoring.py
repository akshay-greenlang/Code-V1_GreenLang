# -*- coding: utf-8 -*-
"""
Unit tests for ClimateRiskScoringEngine (Engine 5)
====================================================

Tests physical risk scoring (acute + chronic), transition risk (5 channels),
NGFS 6 scenarios, 3 time horizons, sector heatmap, collateral risk,
credit risk impact (PD uplift, LGD), stranded assets, composite score 0-100,
and provenance hashing.

Target: 85%+ coverage, ~30 tests.
"""

import importlib.util
import os
import sys
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Dynamic import via importlib
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "engines",
)
_ENGINE_PATH = os.path.normpath(
    os.path.join(_ENGINE_DIR, "climate_risk_scoring_engine.py")
)

spec = importlib.util.spec_from_file_location(
    "climate_risk_scoring_engine", _ENGINE_PATH,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ClimateRiskScoringEngine = mod.ClimateRiskScoringEngine
ClimateRiskConfig = mod.ClimateRiskConfig
ExposureData = mod.ExposureData
ClimateRiskResult = mod.ClimateRiskResult
PhysicalRiskScore = mod.PhysicalRiskScore
TransitionRiskScore = mod.TransitionRiskScore
CreditRiskImpact = mod.CreditRiskImpact
StrandedAssetExposure = mod.StrandedAssetExposure
NGFSScenarioResult = mod.NGFSScenarioResult
NGFSScenario = mod.NGFSScenario
PhysicalHazard = mod.PhysicalHazard
TransitionChannel = mod.TransitionChannel
TimeHorizon = mod.TimeHorizon
RiskLevel = mod.RiskLevel
NGFS_SCENARIO_PARAMS = mod.NGFS_SCENARIO_PARAMS
SECTOR_TRANSITION_HEATMAP = mod.SECTOR_TRANSITION_HEATMAP
SECTOR_PHYSICAL_HEATMAP = mod.SECTOR_PHYSICAL_HEATMAP
RISK_LEVEL_THRESHOLDS = mod.RISK_LEVEL_THRESHOLDS
FOSSIL_FUEL_NACE_CODES = mod.FOSSIL_FUEL_NACE_CODES
DEFAULT_PHYSICAL_WEIGHT = mod.DEFAULT_PHYSICAL_WEIGHT
DEFAULT_TRANSITION_WEIGHT = mod.DEFAULT_TRANSITION_WEIGHT
_compute_hash = mod._compute_hash
_safe_divide = mod._safe_divide
_safe_pct = mod._safe_pct
_clamp = mod._clamp
_round_val = mod._round_val


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> ClimateRiskConfig:
    """Default engine configuration."""
    return ClimateRiskConfig(
        portfolio_name="Test Portfolio",
        scenarios=[
            NGFSScenario.NET_ZERO_2050,
            NGFSScenario.BELOW_2C,
            NGFSScenario.DELAYED_TRANSITION,
            NGFSScenario.CURRENT_POLICIES,
        ],
        time_horizons=[TimeHorizon.SHORT, TimeHorizon.MEDIUM, TimeHorizon.LONG],
    )


@pytest.fixture
def engine(default_config) -> ClimateRiskScoringEngine:
    """Engine instance."""
    return ClimateRiskScoringEngine(default_config)


@pytest.fixture
def sample_exposures() -> List[ExposureData]:
    """Create a small portfolio of diverse exposures."""
    return [
        ExposureData(
            exposure_id="exp-001",
            counterparty_name="EnergyCorpAlpha",
            nace_code="D35",
            country="DE",
            exposure_eur=50_000_000.0,
            weight_pct=25.0,
            base_pd=0.02,
            base_lgd=0.45,
            collateral_type="none",
            carbon_intensity=450.0,
            scope1_emissions=120_000.0,
            scope2_emissions=30_000.0,
            has_transition_plan=True,
            transition_plan_quality=60.0,
            fossil_fuel_revenue_pct=80.0,
            is_fossil_fuel_company=True,
            flood_severity=0.3,
            wildfire_severity=0.1,
            storm_severity=0.2,
            heatwave_severity=0.4,
            sea_level_rise_severity=0.15,
            heat_stress_severity=0.25,
            drought_severity=0.1,
        ),
        ExposureData(
            exposure_id="exp-002",
            counterparty_name="RealEstateHoldCo",
            nace_code="L68",
            country="NL",
            exposure_eur=80_000_000.0,
            weight_pct=40.0,
            base_pd=0.005,
            base_lgd=0.30,
            collateral_type="real_estate",
            collateral_value_eur=100_000_000.0,
            epc_label="C",
            carbon_intensity=50.0,
            scope1_emissions=5_000.0,
            scope2_emissions=8_000.0,
            has_transition_plan=False,
            flood_severity=0.6,
            sea_level_rise_severity=0.5,
            heat_stress_severity=0.3,
        ),
        ExposureData(
            exposure_id="exp-003",
            counterparty_name="TechStartup",
            nace_code="J62",
            country="IE",
            exposure_eur=20_000_000.0,
            weight_pct=10.0,
            base_pd=0.03,
            base_lgd=0.50,
            collateral_type="none",
            carbon_intensity=10.0,
            scope1_emissions=500.0,
            scope2_emissions=2_000.0,
            has_transition_plan=True,
            transition_plan_quality=80.0,
        ),
        ExposureData(
            exposure_id="exp-004",
            counterparty_name="CoalMineCo",
            nace_code="B05",
            country="PL",
            exposure_eur=30_000_000.0,
            weight_pct=15.0,
            base_pd=0.05,
            base_lgd=0.55,
            collateral_type="none",
            carbon_intensity=900.0,
            scope1_emissions=250_000.0,
            scope2_emissions=50_000.0,
            is_fossil_fuel_company=True,
            fossil_fuel_revenue_pct=95.0,
            has_transition_plan=False,
            heatwave_severity=0.3,
            drought_severity=0.4,
            storm_severity=0.35,
        ),
    ]


@pytest.fixture
def single_exposure() -> ExposureData:
    """Single exposure for unit-level tests."""
    return ExposureData(
        exposure_id="exp-solo",
        counterparty_name="SingleCo",
        nace_code="C24",
        country="DE",
        exposure_eur=10_000_000.0,
        weight_pct=100.0,
        base_pd=0.01,
        base_lgd=0.45,
        carbon_intensity=200.0,
        scope1_emissions=50_000.0,
        flood_severity=0.5,
        storm_severity=0.3,
        sea_level_rise_severity=0.2,
        heat_stress_severity=0.4,
        drought_severity=0.3,
    )


# ===================================================================
# Test Class: Helpers
# ===================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_safe_divide_normal(self):
        assert _safe_divide(10.0, 2.0) == 5.0

    def test_safe_divide_zero_denom(self):
        assert _safe_divide(10.0, 0.0) == 0.0

    def test_safe_divide_custom_default(self):
        assert _safe_divide(10.0, 0.0, default=-1.0) == -1.0

    def test_safe_pct(self):
        assert _safe_pct(25.0, 100.0) == 25.0

    def test_safe_pct_zero(self):
        assert _safe_pct(25.0, 0.0) == 0.0

    def test_clamp_within(self):
        assert _clamp(50.0) == 50.0

    def test_clamp_below(self):
        assert _clamp(-5.0) == 0.0

    def test_clamp_above(self):
        assert _clamp(150.0) == 100.0

    def test_round_val(self):
        assert _round_val(3.14159, 2) == 3.14

    def test_compute_hash_deterministic(self):
        data = {"a": 1, "b": "hello"}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2
        assert len(h1) == 64


# ===================================================================
# Test Class: Configuration
# ===================================================================


class TestClimateRiskConfig:
    """Tests for ClimateRiskConfig validation."""

    def test_default_config(self):
        cfg = ClimateRiskConfig()
        assert cfg.physical_weight == DEFAULT_PHYSICAL_WEIGHT
        assert cfg.transition_weight == DEFAULT_TRANSITION_WEIGHT
        assert abs(cfg.physical_weight + cfg.transition_weight - 1.0) < 0.001

    def test_custom_weights_valid(self):
        cfg = ClimateRiskConfig(physical_weight=0.6, transition_weight=0.4)
        assert cfg.physical_weight == 0.6
        assert cfg.transition_weight == 0.4

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            ClimateRiskConfig(physical_weight=0.3, transition_weight=0.3)

    def test_all_six_ngfs_scenarios_available(self):
        assert len(NGFSScenario) == 6
        for s in NGFSScenario:
            assert s.value in NGFS_SCENARIO_PARAMS


# ===================================================================
# Test Class: Engine Initialization
# ===================================================================


class TestEngineInitialization:
    """Tests for engine construction."""

    def test_engine_creates_with_default_config(self):
        engine = ClimateRiskScoringEngine(ClimateRiskConfig())
        assert engine.config is not None

    def test_engine_stores_config(self, default_config):
        engine = ClimateRiskScoringEngine(default_config)
        assert engine.config.portfolio_name == "Test Portfolio"


# ===================================================================
# Test Class: Physical Risk Scoring
# ===================================================================


class TestPhysicalRiskScoring:
    """Tests for physical risk (acute + chronic) scoring."""

    def test_physical_risk_score_structure(self, engine, single_exposure):
        score = engine._score_physical_risk(single_exposure)
        assert isinstance(score, PhysicalRiskScore)
        assert 0.0 <= score.acute_score <= 100.0
        assert 0.0 <= score.chronic_score <= 100.0
        assert 0.0 <= score.composite_physical_score <= 100.0
        assert isinstance(score.risk_level, RiskLevel)

    def test_zero_severity_yields_zero_score(self, engine):
        exp = ExposureData(nace_code="J62")
        score = engine._score_physical_risk(exp)
        assert score.acute_score == 0.0
        assert score.chronic_score == 0.0
        assert score.composite_physical_score == 0.0

    def test_high_severity_yields_higher_score(self, engine):
        low = ExposureData(nace_code="A01", flood_severity=0.1)
        high = ExposureData(nace_code="A01", flood_severity=0.9)
        low_score = engine._score_physical_risk(low)
        high_score = engine._score_physical_risk(high)
        assert high_score.composite_physical_score >= low_score.composite_physical_score

    def test_sector_vulnerability_affects_score(self, engine):
        agri = ExposureData(nace_code="A01", flood_severity=0.5)
        tech = ExposureData(nace_code="J62", flood_severity=0.5)
        agri_score = engine._score_physical_risk(agri)
        tech_score = engine._score_physical_risk(tech)
        assert agri_score.composite_physical_score >= tech_score.composite_physical_score

    def test_hazard_scores_dict_populated(self, engine, single_exposure):
        score = engine._score_physical_risk(single_exposure)
        assert len(score.hazard_scores) > 0
        for hazard_val, s in score.hazard_scores.items():
            assert s >= 0.0


# ===================================================================
# Test Class: Transition Risk Scoring
# ===================================================================


class TestTransitionRiskScoring:
    """Tests for transition risk (5 channels) scoring."""

    def test_transition_risk_score_structure(self, engine, single_exposure):
        score = engine._score_transition_risk(single_exposure)
        assert isinstance(score, TransitionRiskScore)
        assert 0.0 <= score.composite_transition_score <= 100.0
        assert hasattr(score, "policy_score")
        assert hasattr(score, "technology_score")
        assert hasattr(score, "market_score")
        assert hasattr(score, "reputation_score")
        assert hasattr(score, "legal_score")

    def test_high_carbon_sector_higher_transition_risk(self, engine):
        fossil = ExposureData(
            nace_code="B06", carbon_intensity=800.0,
            fossil_fuel_revenue_pct=90.0, is_fossil_fuel_company=True,
        )
        green = ExposureData(
            nace_code="J62", carbon_intensity=5.0,
            has_transition_plan=True, transition_plan_quality=90.0,
        )
        fossil_score = engine._score_transition_risk(fossil)
        green_score = engine._score_transition_risk(green)
        assert fossil_score.composite_transition_score > green_score.composite_transition_score

    def test_transition_plan_reduces_risk(self, engine):
        no_plan = ExposureData(
            nace_code="D35", carbon_intensity=400.0,
            has_transition_plan=False,
        )
        with_plan = ExposureData(
            nace_code="D35", carbon_intensity=400.0,
            has_transition_plan=True, transition_plan_quality=80.0,
        )
        no_score = engine._score_transition_risk(no_plan)
        plan_score = engine._score_transition_risk(with_plan)
        assert plan_score.composite_transition_score <= no_score.composite_transition_score


# ===================================================================
# Test Class: NGFS Scenario Analysis
# ===================================================================


class TestNGFSScenarioAnalysis:
    """Tests for NGFS 6 scenarios x 3 time horizons."""

    def test_six_scenarios_exist(self):
        assert len(NGFSScenario) == 6

    def test_three_time_horizons(self):
        assert len(TimeHorizon) == 3

    def test_scenario_params_populated(self):
        for scenario in NGFSScenario:
            params = NGFS_SCENARIO_PARAMS[scenario.value]
            assert "transition_severity" in params
            assert "physical_severity" in params
            assert "carbon_price_2030_usd" in params
            assert "temperature_2100_c" in params

    def test_portfolio_returns_scenario_results(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert len(result.scenario_results) > 0
        for sr in result.scenario_results:
            assert isinstance(sr, NGFSScenarioResult)
            assert 0.0 <= sr.composite_score <= 100.0

    def test_current_policies_highest_physical_risk(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        cp_results = [
            sr for sr in result.scenario_results
            if sr.scenario == NGFSScenario.CURRENT_POLICIES
        ]
        nz_results = [
            sr for sr in result.scenario_results
            if sr.scenario == NGFSScenario.NET_ZERO_2050
        ]
        if cp_results and nz_results:
            assert cp_results[0].physical_risk_score >= nz_results[0].physical_risk_score


# ===================================================================
# Test Class: Sector Heatmap
# ===================================================================


class TestSectorHeatmap:
    """Tests for sector-level heatmap generation."""

    def test_sector_heatmap_populated(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert len(result.sector_heatmap) > 0

    def test_fossil_sectors_high_transition_sensitivity(self):
        fossil_codes = ["B", "C19", "D35"]
        for code in fossil_codes:
            sensitivity = SECTOR_TRANSITION_HEATMAP.get(code, 0.0)
            assert sensitivity >= 0.85, f"{code} should have sensitivity >= 0.85"


# ===================================================================
# Test Class: Credit Risk Impact
# ===================================================================


class TestCreditRiskImpact:
    """Tests for PD uplift and LGD adjustment."""

    def test_credit_impacts_generated(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert len(result.credit_risk_impacts) == len(sample_exposures)

    def test_pd_uplift_non_negative(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        for ci in result.credit_risk_impacts:
            assert ci.adjusted_pd >= ci.base_pd

    def test_lgd_adjustment_non_negative(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        for ci in result.credit_risk_impacts:
            assert ci.adjusted_lgd >= ci.base_lgd

    def test_incremental_expected_loss_calculated(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        for ci in result.credit_risk_impacts:
            assert ci.adjusted_expected_loss >= ci.base_expected_loss

    def test_total_incremental_el(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert result.total_incremental_el_eur >= 0.0


# ===================================================================
# Test Class: Stranded Assets
# ===================================================================


class TestStrandedAssets:
    """Tests for stranded asset identification."""

    def test_stranded_asset_assessment_generated(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert result.stranded_asset_exposure is not None

    def test_fossil_fuel_nace_codes_detected(self):
        assert "B05" in FOSSIL_FUEL_NACE_CODES
        assert "B06" in FOSSIL_FUEL_NACE_CODES
        assert "C19" in FOSSIL_FUEL_NACE_CODES
        assert "D35" in FOSSIL_FUEL_NACE_CODES

    def test_stranded_ratio_between_0_and_100(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        sa = result.stranded_asset_exposure
        assert 0.0 <= sa.stranded_asset_ratio_pct <= 100.0

    def test_stranded_asset_disabled(self, sample_exposures):
        cfg = ClimateRiskConfig(include_stranded_assets=False)
        eng = ClimateRiskScoringEngine(cfg)
        result = eng.assess_portfolio(sample_exposures)
        assert result.stranded_asset_exposure is None


# ===================================================================
# Test Class: Composite Score and Collateral
# ===================================================================


class TestCompositeScoreAndCollateral:
    """Tests for composite 0-100 score and collateral risk."""

    def test_composite_score_in_range(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert 0.0 <= result.composite_risk_score <= 100.0

    def test_composite_equals_weighted_sum(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        expected = (
            engine.config.physical_weight * result.physical_risk_score
            + engine.config.transition_weight * result.transition_risk_score
        )
        assert abs(result.composite_risk_score - round(expected, 2)) < 0.5

    def test_risk_level_assigned(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert isinstance(result.composite_risk_level, RiskLevel)

    def test_collateral_at_risk(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert result.collateral_at_risk_eur >= 0.0
        assert 0.0 <= result.collateral_at_risk_pct <= 100.0


# ===================================================================
# Test Class: Provenance and Metadata
# ===================================================================


class TestProvenanceAndMetadata:
    """Tests for provenance hash and result metadata."""

    def test_provenance_hash_is_sha256(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # valid hex

    def test_provenance_hash_populated_for_scenarios(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        for sr in result.scenario_results:
            assert len(sr.provenance_hash) == 64

    def test_processing_time_positive(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert result.processing_time_ms > 0.0

    def test_total_exposures_counted(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert result.total_exposures == len(sample_exposures)

    def test_total_exposure_eur(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        expected = sum(e.exposure_eur for e in sample_exposures)
        assert abs(result.total_exposure_eur - expected) < 1.0

    def test_data_coverage_percentage(self, engine, sample_exposures):
        result = engine.assess_portfolio(sample_exposures)
        assert 0.0 <= result.data_coverage_pct <= 100.0

    def test_empty_portfolio(self, engine):
        result = engine.assess_portfolio([])
        assert result.total_exposures == 0
        assert result.composite_risk_score == 0.0
