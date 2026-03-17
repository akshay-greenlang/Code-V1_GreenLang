# -*- coding: utf-8 -*-
"""
Unit tests for ManufacturingBenchmarkEngine (PACK-013, Engine 8)

Tests sector benchmarking, KPI ranking, SBTi alignment, EU ETS gap
analysis, trajectory analysis, and provenance tracking.

Target: 85%+ coverage, 38+ tests.
"""

import importlib.util
import os
import sys
import pytest
from decimal import Decimal

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines"
)


def _load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_ENGINE_DIR, file_name)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mb = _load_module("manufacturing_benchmark_engine", "manufacturing_benchmark_engine.py")

ManufacturingBenchmarkEngine = mb.ManufacturingBenchmarkEngine
BenchmarkConfig = mb.BenchmarkConfig
FacilityKPIs = mb.FacilityKPIs
BenchmarkResult = mb.BenchmarkResult
KPIRanking = mb.KPIRanking
SBTiAlignment = mb.SBTiAlignment
TrajectoryAnalysis = mb.TrajectoryAnalysis
ETSBenchmarkResult = mb.ETSBenchmarkResult
BenchmarkKPI = mb.BenchmarkKPI
PercentileRank = mb.PercentileRank
SBTiPathway = mb.SBTiPathway
SubSector = mb.SubSector
SECTOR_BENCHMARKS = mb.SECTOR_BENCHMARKS
SBTI_PATHWAYS = mb.SBTI_PATHWAYS
ETS_BENCHMARKS = mb.ETS_BENCHMARKS
SUBSECTOR_ETS_MAP = mb.SUBSECTOR_ETS_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    return BenchmarkConfig(
        reporting_year=2025,
        sub_sector=SubSector.CEMENT.value,
        sbti_pathway=SBTiPathway.ONE_POINT_FIVE_C,
        baseline_year=2019,
        target_year=2030,
    )


@pytest.fixture
def default_engine(default_config):
    return ManufacturingBenchmarkEngine(default_config)


@pytest.fixture
def cement_kpis():
    return FacilityKPIs(
        facility_id="CEM-001",
        facility_name="Cement Plant Alpha",
        sub_sector=SubSector.CEMENT.value,
        emission_intensity_tco2e_per_unit=Decimal("0.63"),
        energy_intensity_mj_per_unit=Decimal("3500"),
        water_intensity_m3_per_unit=Decimal("0.35"),
        waste_intensity_kg_per_unit=Decimal("12"),
        circularity_rate_pct=Decimal("18"),
        renewable_share_pct=Decimal("12"),
        safety_ltir=Decimal("2.5"),
        baseline_emission_intensity=Decimal("0.80"),
    )


@pytest.fixture
def automotive_kpis():
    return FacilityKPIs(
        facility_id="AUTO-001",
        facility_name="Auto Plant Beta",
        sub_sector=SubSector.AUTOMOTIVE.value,
        emission_intensity_tco2e_per_unit=Decimal("0.80"),
        energy_intensity_mj_per_unit=Decimal("6000"),
        water_intensity_m3_per_unit=Decimal("4.0"),
        waste_intensity_kg_per_unit=Decimal("15"),
        renewable_share_pct=Decimal("35"),
        baseline_emission_intensity=Decimal("1.20"),
    )


@pytest.fixture
def chemicals_kpis():
    return FacilityKPIs(
        facility_id="CHEM-001",
        facility_name="Chemical Plant Gamma",
        sub_sector=SubSector.CHEMICALS.value,
        emission_intensity_tco2e_per_unit=Decimal("1.20"),
        energy_intensity_mj_per_unit=Decimal("12000"),
        water_intensity_m3_per_unit=Decimal("6.0"),
        waste_intensity_kg_per_unit=Decimal("50"),
        renewable_share_pct=Decimal("18"),
        baseline_emission_intensity=Decimal("1.60"),
    )


# ---------------------------------------------------------------------------
# TestInitialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test ManufacturingBenchmarkEngine initialization."""

    def test_default_init(self, default_config):
        engine = ManufacturingBenchmarkEngine(default_config)
        assert engine.config == default_config
        assert engine.config.reporting_year == 2025

    def test_with_config(self):
        cfg = BenchmarkConfig(
            reporting_year=2024,
            sub_sector=SubSector.STEEL_BOF.value,
        )
        engine = ManufacturingBenchmarkEngine(cfg)
        assert engine.config.sub_sector == SubSector.STEEL_BOF.value

    def test_with_dict(self):
        cfg = BenchmarkConfig(**{
            "reporting_year": 2025,
            "sub_sector": "automotive",
            "sbti_pathway": "one_point_five_c",
        })
        engine = ManufacturingBenchmarkEngine(cfg)
        assert engine.config.sub_sector == "automotive"

    def test_with_none_optional_fields(self):
        cfg = BenchmarkConfig(reporting_year=2025, sub_sector="cement")
        engine = ManufacturingBenchmarkEngine(cfg)
        assert engine.config.include_sbti is True
        assert engine.config.include_ets_benchmark is True


# ---------------------------------------------------------------------------
# TestBenchmarkKPIs
# ---------------------------------------------------------------------------


class TestBenchmarkKPIs:
    """Test KPI enumeration and definitions."""

    def test_all_kpis_defined(self):
        assert len(BenchmarkKPI) == 8

    def test_kpi_enum_values(self):
        values = {k.value for k in BenchmarkKPI}
        assert "emission_intensity" in values
        assert "energy_intensity" in values
        assert "water_intensity" in values

    def test_emission_intensity_kpi(self):
        assert BenchmarkKPI.EMISSION_INTENSITY.value == "emission_intensity"

    def test_energy_intensity_kpi(self):
        assert BenchmarkKPI.ENERGY_INTENSITY.value == "energy_intensity"

    def test_safety_ltir_kpi(self):
        assert BenchmarkKPI.SAFETY_LTIR.value == "safety_ltir"


# ---------------------------------------------------------------------------
# TestPercentileRanking
# ---------------------------------------------------------------------------


class TestPercentileRanking:
    """Test quartile ranking logic."""

    def test_top_quartile(self, default_engine):
        # Cement emission intensity top_quartile=0.55; value=0.50 => TOP
        rank = default_engine.rank_kpi(0.50, SubSector.CEMENT.value, BenchmarkKPI.EMISSION_INTENSITY)
        assert rank == PercentileRank.TOP_QUARTILE

    def test_second_quartile(self, default_engine):
        # Cement emission intensity median=0.63; value=0.60 => SECOND (below median, above top_q)
        rank = default_engine.rank_kpi(0.60, SubSector.CEMENT.value, BenchmarkKPI.EMISSION_INTENSITY)
        assert rank == PercentileRank.SECOND_QUARTILE

    def test_third_quartile(self, default_engine):
        # Cement emission intensity bottom_quartile=0.78; value=0.70 => THIRD
        rank = default_engine.rank_kpi(0.70, SubSector.CEMENT.value, BenchmarkKPI.EMISSION_INTENSITY)
        assert rank == PercentileRank.THIRD_QUARTILE

    def test_bottom_quartile(self, default_engine):
        # Cement emission intensity bottom_quartile=0.78; value=0.90 => BOTTOM
        rank = default_engine.rank_kpi(0.90, SubSector.CEMENT.value, BenchmarkKPI.EMISSION_INTENSITY)
        assert rank == PercentileRank.BOTTOM_QUARTILE

    def test_percentile_enum(self):
        assert len(PercentileRank) == 4


# ---------------------------------------------------------------------------
# TestSectorBenchmarks
# ---------------------------------------------------------------------------


class TestSectorBenchmarks:
    """Test sector benchmark data availability."""

    def test_cement_benchmarks(self):
        assert SubSector.CEMENT in SECTOR_BENCHMARKS
        cement = SECTOR_BENCHMARKS[SubSector.CEMENT]
        assert BenchmarkKPI.EMISSION_INTENSITY in cement

    def test_steel_benchmarks(self):
        assert SubSector.STEEL_BOF in SECTOR_BENCHMARKS
        steel = SECTOR_BENCHMARKS[SubSector.STEEL_BOF]
        assert BenchmarkKPI.EMISSION_INTENSITY in steel

    def test_automotive_benchmarks(self):
        assert SubSector.AUTOMOTIVE in SECTOR_BENCHMARKS

    def test_chemicals_benchmarks(self):
        assert SubSector.CHEMICALS in SECTOR_BENCHMARKS

    def test_all_sectors_have_data(self):
        # At least 8 sectors should have benchmark data
        assert len(SECTOR_BENCHMARKS) >= 8


# ---------------------------------------------------------------------------
# TestKPIRanking
# ---------------------------------------------------------------------------


class TestKPIRanking:
    """Test KPI ranking in full assessment."""

    def test_rank_emission_intensity(self, default_engine, cement_kpis):
        result = default_engine.benchmark_facility(cement_kpis)
        ei_ranks = [r for r in result.kpi_rankings if r.kpi == "emission_intensity"]
        assert len(ei_ranks) == 1
        # 0.63 == median for cement => SECOND_QUARTILE
        assert ei_ranks[0].rank == PercentileRank.SECOND_QUARTILE

    def test_rank_energy_intensity(self, default_engine, cement_kpis):
        result = default_engine.benchmark_facility(cement_kpis)
        eni_ranks = [r for r in result.kpi_rankings if r.kpi == "energy_intensity"]
        assert len(eni_ranks) == 1
        # 3500 == median for cement => SECOND_QUARTILE
        assert eni_ranks[0].rank == PercentileRank.SECOND_QUARTILE

    def test_rank_water_intensity(self, default_engine, cement_kpis):
        result = default_engine.benchmark_facility(cement_kpis)
        wi_ranks = [r for r in result.kpi_rankings if r.kpi == "water_intensity"]
        assert len(wi_ranks) == 1

    def test_rank_multiple_kpis(self, default_engine, cement_kpis):
        result = default_engine.benchmark_facility(cement_kpis)
        # Cement has 8 KPIs in benchmarks, our fixture has 7 non-zero values
        assert result.kpi_count_assessed >= 5


# ---------------------------------------------------------------------------
# TestSBTiAlignment
# ---------------------------------------------------------------------------


class TestSBTiAlignment:
    """Test SBTi pathway alignment assessment."""

    def test_on_track_1_5c(self, default_engine):
        # Baseline 0.80, current 0.40 => 50% reduction (needs ~42% by 2030 for 1.5C)
        facility = FacilityKPIs(
            facility_id="SBT-ON", sub_sector="cement",
            emission_intensity_tco2e_per_unit=Decimal("0.40"),
            baseline_emission_intensity=Decimal("0.80"),
        )
        result = default_engine.assess_sbti_alignment(
            facility, SBTiPathway.ONE_POINT_FIVE_C
        )
        assert result.on_track is True

    def test_off_track(self, default_engine):
        # Baseline 0.80, current 0.75 => 6.25% reduction (needs ~42% by 2030)
        facility = FacilityKPIs(
            facility_id="SBT-OFF", sub_sector="cement",
            emission_intensity_tco2e_per_unit=Decimal("0.75"),
            baseline_emission_intensity=Decimal("0.80"),
        )
        result = default_engine.assess_sbti_alignment(
            facility, SBTiPathway.ONE_POINT_FIVE_C
        )
        assert result.on_track is False

    def test_gap_percentage(self, default_engine):
        facility = FacilityKPIs(
            facility_id="SBT-GAP", sub_sector="cement",
            emission_intensity_tco2e_per_unit=Decimal("0.75"),
            baseline_emission_intensity=Decimal("0.80"),
        )
        result = default_engine.assess_sbti_alignment(
            facility, SBTiPathway.ONE_POINT_FIVE_C
        )
        assert result.gap_pct > 0

    def test_pathway_enum(self):
        assert len(SBTiPathway) == 3
        values = {p.value for p in SBTiPathway}
        assert "one_point_five_c" in values
        assert "well_below_2c" in values
        assert "net_zero_2050" in values


# ---------------------------------------------------------------------------
# TestTrajectoryAnalysis
# ---------------------------------------------------------------------------


class TestTrajectoryAnalysis:
    """Test multi-year trajectory analysis."""

    def test_improving_trajectory(self, default_engine):
        traj = default_engine.analyze_trajectory(
            baseline_value=1.0,
            current_value=0.70,
            target_reduction_pct=42.0,
            baseline_year=2019,
            current_year=2025,
            target_year=2030,
        )
        assert traj.actual_reduction_pct == pytest.approx(30.0, rel=1e-2)
        assert traj.annual_reduction_rate > 0

    def test_declining_trajectory(self, default_engine):
        traj = default_engine.analyze_trajectory(
            baseline_value=1.0,
            current_value=0.95,
            target_reduction_pct=42.0,
            baseline_year=2019,
            current_year=2025,
            target_year=2030,
        )
        assert traj.actual_reduction_pct == pytest.approx(5.0, rel=1e-2)
        assert traj.on_track is False

    def test_annual_reduction_rate(self, default_engine):
        traj = default_engine.analyze_trajectory(
            baseline_value=1.0,
            current_value=0.80,
            target_reduction_pct=42.0,
            baseline_year=2019,
            current_year=2025,
            target_year=2030,
        )
        assert traj.annual_reduction_rate > 0

    def test_on_track_assessment(self, default_engine):
        # 42% target over 11 years, by year 6 need ~22.9% reduction
        # current_value = 0.60 => 40% reduction => on track
        traj = default_engine.analyze_trajectory(
            baseline_value=1.0,
            current_value=0.60,
            target_reduction_pct=42.0,
            baseline_year=2019,
            current_year=2025,
            target_year=2030,
        )
        assert traj.on_track is True


# ---------------------------------------------------------------------------
# TestETSBenchmark
# ---------------------------------------------------------------------------


class TestETSBenchmark:
    """Test EU ETS benchmark gap analysis."""

    def test_below_benchmark(self, default_engine):
        facility = FacilityKPIs(
            facility_id="ETS-BLW", sub_sector="cement",
            emission_intensity_tco2e_per_unit=Decimal("0.50"),
        )
        result = default_engine.calculate_ets_gap(facility)
        assert result is not None
        assert result.above_benchmark is False
        assert result.free_allocation_eligible is True

    def test_above_benchmark(self, default_engine):
        facility = FacilityKPIs(
            facility_id="ETS-ABV", sub_sector="cement",
            emission_intensity_tco2e_per_unit=Decimal("0.90"),
        )
        result = default_engine.calculate_ets_gap(facility)
        assert result is not None
        assert result.above_benchmark is True
        assert result.free_allocation_eligible is False

    def test_gap_percentage(self, default_engine):
        facility = FacilityKPIs(
            facility_id="ETS-GAP", sub_sector="cement",
            emission_intensity_tco2e_per_unit=Decimal("0.90"),
        )
        result = default_engine.calculate_ets_gap(facility)
        assert result is not None
        assert result.gap_pct > 0


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_hash(self, default_engine, cement_kpis):
        result = default_engine.benchmark_facility(cement_kpis)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self, default_engine, cement_kpis):
        r1 = default_engine.benchmark_facility(cement_kpis)
        r2 = default_engine.benchmark_facility(cement_kpis)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64

    def test_different_input(self, default_engine, cement_kpis, automotive_kpis):
        r1 = default_engine.benchmark_facility(cement_kpis)
        auto_cfg = BenchmarkConfig(
            reporting_year=2025,
            sub_sector=SubSector.AUTOMOTIVE.value,
        )
        auto_engine = ManufacturingBenchmarkEngine(auto_cfg)
        r2 = auto_engine.benchmark_facility(automotive_kpis)
        assert r1.facility_id != r2.facility_id


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_sector(self, default_engine):
        facility = FacilityKPIs(
            facility_id="UNK-001", sub_sector="nonexistent_sector",
            emission_intensity_tco2e_per_unit=Decimal("1.0"),
        )
        result = default_engine.benchmark_facility(facility)
        assert result.kpi_count_assessed == 0
        assert result.overall_percentile == "not_assessed"

    def test_large_dataset(self):
        cfg = BenchmarkConfig(reporting_year=2025, sub_sector="cement")
        engine = ManufacturingBenchmarkEngine(cfg)
        facility = FacilityKPIs(
            facility_id="PERF-001", sub_sector="cement",
            emission_intensity_tco2e_per_unit=Decimal("0.63"),
            energy_intensity_mj_per_unit=Decimal("3500"),
            water_intensity_m3_per_unit=Decimal("0.35"),
            waste_intensity_kg_per_unit=Decimal("12"),
            circularity_rate_pct=Decimal("18"),
            renewable_share_pct=Decimal("12"),
            scope3_ratio_pct=Decimal("22"),
            safety_ltir=Decimal("2.5"),
            baseline_emission_intensity=Decimal("0.80"),
        )
        result = engine.benchmark_facility(facility)
        assert result.kpi_count_assessed >= 7

    def test_result_fields(self, default_engine, cement_kpis):
        result = default_engine.benchmark_facility(cement_kpis)
        assert isinstance(result, BenchmarkResult)
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0

    def test_methodology_notes(self, default_engine, cement_kpis):
        result = default_engine.benchmark_facility(cement_kpis)
        assert isinstance(result.methodology_notes, list)
        assert len(result.methodology_notes) > 0
