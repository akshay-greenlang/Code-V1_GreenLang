# -*- coding: utf-8 -*-
"""
Unit tests for FSTransitionPlanEngine (Engine 7)
==================================================

Tests SBTi-FI assessment (SDA, Portfolio Coverage, Temperature Rating),
NZBA commitment tracking, sector decarbonization pathways, fossil fuel
phase-out, credibility score 0-100, and provenance hashing.

Target: 85%+ coverage, ~30 tests.
"""

import importlib.util
import os
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Dynamic import via importlib
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "engines",
)
_ENGINE_PATH = os.path.normpath(
    os.path.join(_ENGINE_DIR, "fs_transition_plan_engine.py")
)

spec = importlib.util.spec_from_file_location(
    "fs_transition_plan_engine", _ENGINE_PATH,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FSTransitionPlanEngine = mod.FSTransitionPlanEngine
TransitionPlanConfig = mod.TransitionPlanConfig
SectorTargetData = mod.SectorTargetData
TransitionPlanResult = mod.TransitionPlanResult
SBTiFIAssessment = mod.SBTiFIAssessment
NZBACommitment = mod.NZBACommitment
SectorDecarbPath = mod.SectorDecarbPath
PhaseOutCommitment = mod.PhaseOutCommitment
CredibilityScore = mod.CredibilityScore
SBTiMethod = mod.SBTiMethod
AllianceType = mod.AllianceType
SectorCategory = mod.SectorCategory
PhaseOutStatus = mod.PhaseOutStatus
CredibilityLevel = mod.CredibilityLevel
SECTOR_NZE_BENCHMARKS = mod.SECTOR_NZE_BENCHMARKS
PHASE_OUT_DEADLINES = mod.PHASE_OUT_DEADLINES
CREDIBILITY_WEIGHTS = mod.CREDIBILITY_WEIGHTS
_compute_hash = mod._compute_hash
_safe_pct = mod._safe_pct
_clamp = mod._clamp
_round_val = mod._round_val


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> TransitionPlanConfig:
    """Default engine configuration."""
    return TransitionPlanConfig(
        institution_name="Test Green Bank",
        alliance_type=AllianceType.NZBA,
        sbti_method=SBTiMethod.SDA,
        base_year=2020,
        current_year=2025,
        net_zero_target_year=2050,
        interim_target_year=2030,
        interim_reduction_target_pct=50.0,
        temperature_target_c=1.5,
    )


@pytest.fixture
def engine(default_config) -> FSTransitionPlanEngine:
    """Engine instance."""
    return FSTransitionPlanEngine(default_config)


@pytest.fixture
def sample_sector_targets() -> List[SectorTargetData]:
    """Create sector-level decarbonization targets."""
    return [
        SectorTargetData(
            sector=SectorCategory.POWER,
            current_intensity=300.0,
            target_intensity_2030=130.0,
            target_intensity_2050=0.0,
            baseline_intensity=460.0,
            baseline_year=2020,
            exposure_eur=2_000_000_000.0,
            weight_pct=30.0,
            unit="gCO2/kWh",
            methodology="SDA",
        ),
        SectorTargetData(
            sector=SectorCategory.REAL_ESTATE,
            current_intensity=42.0,
            target_intensity_2030=28.0,
            target_intensity_2050=0.0,
            baseline_intensity=55.0,
            baseline_year=2020,
            exposure_eur=3_000_000_000.0,
            weight_pct=45.0,
            unit="kgCO2/m2",
            methodology="SDA",
        ),
        SectorTargetData(
            sector=SectorCategory.STEEL,
            current_intensity=1.65,
            target_intensity_2030=1.30,
            target_intensity_2050=0.10,
            baseline_intensity=1.80,
            baseline_year=2020,
            exposure_eur=500_000_000.0,
            weight_pct=10.0,
            unit="tCO2/t_steel",
            methodology="SBTi",
        ),
        SectorTargetData(
            sector=SectorCategory.TRANSPORT,
            current_intensity=100.0,
            target_intensity_2030=70.0,
            target_intensity_2050=0.0,
            baseline_intensity=120.0,
            baseline_year=2020,
            exposure_eur=800_000_000.0,
            weight_pct=15.0,
            unit="gCO2/pkm",
            methodology="custom",
        ),
    ]


@pytest.fixture
def sample_nzba_commitment() -> NZBACommitment:
    """Create sample NZBA commitment data."""
    return NZBACommitment(
        alliance=AllianceType.NZBA,
        member_since="2022-01",
        targets_published=True,
        sectors_with_targets=["power", "real_estate", "steel"],
        sectors_required=9,
        annual_report_submitted=True,
        transition_plan_published=True,
        coal_phase_out_date=2030,
        oil_gas_policy="restricting",
    )


@pytest.fixture
def sample_phase_out_data() -> List[Dict[str, Any]]:
    """Create sample phase-out commitment data."""
    return [
        {
            "fuel_type": "thermal_coal_mining",
            "phase_out_target_year": 2028,
            "current_exposure_eur": 200_000_000.0,
            "baseline_exposure_eur": 800_000_000.0,
        },
        {
            "fuel_type": "thermal_coal_power",
            "phase_out_target_year": 2029,
            "current_exposure_eur": 150_000_000.0,
            "baseline_exposure_eur": 600_000_000.0,
        },
    ]


@pytest.fixture
def sample_engagement_data() -> Dict[str, Any]:
    """Create sample client engagement data."""
    return {
        "clients_engaged": 45,
        "engagement_target": 100,
        "engagements_completed": 45,
    }


@pytest.fixture
def sample_capex_data() -> Dict[str, float]:
    """Create sample CapEx alignment data."""
    return {
        "taxonomy_aligned_capex": 150_000_000.0,
        "total_capex": 500_000_000.0,
    }


# ===================================================================
# Test Class: Configuration
# ===================================================================


class TestTransitionPlanConfig:
    """Tests for TransitionPlanConfig validation."""

    def test_default_config(self):
        cfg = TransitionPlanConfig()
        assert cfg.net_zero_target_year == 2050
        assert cfg.temperature_target_c == 1.5

    def test_custom_alliance(self):
        cfg = TransitionPlanConfig(alliance_type=AllianceType.NZAOA)
        assert cfg.alliance_type == AllianceType.NZAOA

    def test_credibility_weights_present(self):
        cfg = TransitionPlanConfig()
        assert len(cfg.credibility_weights) == len(CREDIBILITY_WEIGHTS)
        total = sum(cfg.credibility_weights.values())
        assert abs(total - 1.0) < 0.01


# ===================================================================
# Test Class: Engine Initialization
# ===================================================================


class TestEngineInit:
    """Tests for engine construction."""

    def test_engine_creates(self, default_config):
        eng = FSTransitionPlanEngine(default_config)
        assert eng.config.institution_name == "Test Green Bank"


# ===================================================================
# Test Class: SBTi-FI Assessment
# ===================================================================


class TestSBTiFIAssessment:
    """Tests for SBTi Financial Institutions assessment."""

    def test_sbti_assessment_generated(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert result.sbti_assessment is not None

    def test_sda_sectors_assessed(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        sbti = result.sbti_assessment
        # Power, Real Estate, Steel, Transport all have NZE benchmarks
        assert sbti.sda_sectors_assessed >= 3

    def test_sda_alignment_percentage(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        sbti = result.sbti_assessment
        assert 0.0 <= sbti.sda_alignment_pct <= 100.0

    def test_portfolio_coverage_calculated(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        sbti = result.sbti_assessment
        assert 0.0 <= sbti.portfolio_coverage_pct <= 100.0

    def test_temperature_score_range(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        sbti = result.sbti_assessment
        assert 1.0 <= sbti.portfolio_temperature_score <= 6.0

    def test_temperature_aligned_flag(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        sbti = result.sbti_assessment
        if sbti.portfolio_temperature_score <= engine.config.temperature_target_c:
            assert sbti.temperature_aligned is True
        else:
            assert sbti.temperature_aligned is False

    def test_sbti_provenance_hash(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert len(result.sbti_assessment.provenance_hash) == 64


# ===================================================================
# Test Class: NZBA Commitment Tracking
# ===================================================================


class TestNZBACommitmentTracking:
    """Tests for NZBA commitment tracking."""

    def test_nzba_commitment_passthrough(
        self, engine, sample_sector_targets, sample_nzba_commitment,
    ):
        result = engine.assess_transition_plan(
            sample_sector_targets, nzba_commitment=sample_nzba_commitment,
        )
        assert result.nzba_commitment is not None
        assert result.nzba_commitment.alliance == AllianceType.NZBA

    def test_nzba_compliance_score(
        self, engine, sample_sector_targets, sample_nzba_commitment,
    ):
        result = engine.assess_transition_plan(
            sample_sector_targets, nzba_commitment=sample_nzba_commitment,
        )
        assert 0.0 <= result.nzba_commitment.compliance_score <= 100.0

    def test_nzba_none_when_not_provided(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert result.nzba_commitment is None


# ===================================================================
# Test Class: Sector Decarbonization Pathways
# ===================================================================


class TestSectorDecarbPathways:
    """Tests for sector-level decarbonization pathway assessment."""

    def test_pathways_generated(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert len(result.sector_pathways) == len(sample_sector_targets)

    def test_pathway_nze_benchmarks(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        power_path = next(
            p for p in result.sector_pathways
            if p.sector == SectorCategory.POWER
        )
        # NZE 2030 for power is 140 gCO2/kWh
        assert power_path.nze_benchmark_2030 == 140.0

    def test_reduction_achieved_calculated(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        power_path = next(
            p for p in result.sector_pathways
            if p.sector == SectorCategory.POWER
        )
        # baseline 460, current 300 => reduction = (460-300)/460 * 100
        expected_reduction = (460.0 - 300.0) / 460.0 * 100.0
        assert abs(power_path.reduction_achieved_pct - round(expected_reduction, 2)) < 0.5

    def test_on_track_2030_flag(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        for path in result.sector_pathways:
            if path.current_intensity <= path.target_2030:
                assert path.on_track_2030 is True
            else:
                assert path.on_track_2030 is False

    def test_gap_to_target_calculated(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        power_path = next(
            p for p in result.sector_pathways
            if p.sector == SectorCategory.POWER
        )
        expected_gap = 300.0 - 130.0
        assert abs(power_path.gap_to_target_2030 - expected_gap) < 1.0

    def test_pathway_provenance_hash(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        for path in result.sector_pathways:
            assert len(path.provenance_hash) == 64


# ===================================================================
# Test Class: Fossil Fuel Phase-Out
# ===================================================================


class TestFossilFuelPhaseOut:
    """Tests for fossil fuel phase-out commitment assessment."""

    def test_phase_out_commitments_generated(
        self, engine, sample_sector_targets, sample_phase_out_data,
    ):
        result = engine.assess_transition_plan(
            sample_sector_targets, phase_out_data=sample_phase_out_data,
        )
        assert len(result.phase_out_commitments) == len(sample_phase_out_data)

    def test_phase_out_deadlines_known(self):
        assert "thermal_coal_mining" in PHASE_OUT_DEADLINES
        assert "thermal_coal_power" in PHASE_OUT_DEADLINES
        assert PHASE_OUT_DEADLINES["thermal_coal_mining"] == 2030

    def test_no_phase_out_when_not_provided(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert len(result.phase_out_commitments) == 0


# ===================================================================
# Test Class: Credibility Score
# ===================================================================


class TestCredibilityScore:
    """Tests for overall credibility assessment 0-100."""

    def test_credibility_generated(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert result.credibility is not None

    def test_credibility_score_in_range(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        cred = result.credibility
        assert 0.0 <= cred.overall_score <= 100.0

    def test_credibility_level_assigned(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        cred = result.credibility
        assert isinstance(cred.credibility_level, CredibilityLevel)

    def test_credibility_weights_sum_to_one(self):
        total = sum(CREDIBILITY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_credibility_component_scores(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        cred = result.credibility
        assert len(cred.component_scores) > 0

    def test_credibility_provenance_hash(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert len(result.credibility.provenance_hash) == 64


# ===================================================================
# Test Class: Portfolio Metrics
# ===================================================================


class TestPortfolioMetrics:
    """Tests for portfolio-level emissions and reduction metrics."""

    def test_yoy_reduction_calculated(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(
            sample_sector_targets,
            portfolio_emissions_tco2e=80_000.0,
            prior_year_emissions_tco2e=100_000.0,
        )
        # (100000 - 80000) / 100000 * 100 = 20%
        assert abs(result.yoy_emission_reduction_pct - 20.0) < 0.5

    def test_yoy_reduction_zero_when_no_prior(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(
            sample_sector_targets,
            portfolio_emissions_tco2e=80_000.0,
            prior_year_emissions_tco2e=0.0,
        )
        assert result.yoy_emission_reduction_pct == 0.0

    def test_portfolio_waci_passthrough(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(
            sample_sector_targets,
            portfolio_waci=125.5,
        )
        assert abs(result.portfolio_waci - 125.5) < 0.1


# ===================================================================
# Test Class: Provenance and Metadata
# ===================================================================


class TestProvenanceAndMetadata:
    """Tests for provenance hash and result metadata."""

    def test_provenance_hash_is_sha256(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_processing_time_positive(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert result.processing_time_ms > 0.0

    def test_engine_version(self, engine, sample_sector_targets):
        result = engine.assess_transition_plan(sample_sector_targets)
        assert result.engine_version == "1.0.0"

    def test_empty_sector_targets(self, engine):
        result = engine.assess_transition_plan([])
        assert result.sbti_assessment is not None
        assert result.sbti_assessment.sda_sectors_assessed == 0
        assert len(result.sector_pathways) == 0
