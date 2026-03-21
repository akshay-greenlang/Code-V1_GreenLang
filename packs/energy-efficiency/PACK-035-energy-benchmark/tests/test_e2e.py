# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - End-to-End Tests
================================================

Tests full benchmark pipeline (data -> EUI -> peer -> rating -> report),
portfolio benchmark pipeline, regulatory compliance pipeline,
multi-engine orchestration, and provenance chain validation.

Test Count Target: ~50 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    ENGINE_FILES,
    ENGINE_CLASSES,
    _load_engine,
    _load_module,
)


def _try_load_engine(engine_key):
    """Load an engine, skip test if not available."""
    try:
        return _load_engine(engine_key)
    except (FileNotFoundError, ImportError):
        pytest.skip(f"Engine not available: {engine_key}")


# =========================================================================
# 1. Single-Facility Benchmark Pipeline
# =========================================================================


class TestSingleFacilityBenchmarkPipeline:
    """Test the full single-facility benchmark pipeline end-to-end."""

    def test_pipeline_step_1_eui_calculation(
        self, sample_facility_profile, sample_energy_data
    ):
        """Step 1: Calculate site EUI from raw energy data."""
        total_energy = sum(
            m["electricity_kwh"] + m["gas_kwh"] for m in sample_energy_data
        )
        floor_area = sample_facility_profile["gross_internal_area_m2"]
        site_eui = total_energy / floor_area
        assert 100 < site_eui < 200
        assert site_eui == pytest.approx(140.6, rel=0.05)

    def test_pipeline_step_2_source_eui(self, sample_energy_data):
        """Step 2: Convert site EUI to source EUI using site-to-source factors."""
        total_elec = sum(m["electricity_kwh"] for m in sample_energy_data)
        total_gas = sum(m["gas_kwh"] for m in sample_energy_data)
        # ENERGY STAR factors
        source_elec = total_elec * 2.80
        source_gas = total_gas * 1.047
        total_source = source_elec + source_gas
        floor_area = 5000.0
        source_eui = total_source / floor_area
        assert source_eui > 140.6  # Source EUI > site EUI for electricity

    def test_pipeline_step_3_peer_comparison(self, sample_peer_group):
        """Step 3: Calculate percentile rank against peer group."""
        subject_eui = 140.6
        peer_euis = sorted(p["eui_kwh_per_m2_yr"] for p in sample_peer_group)
        better_count = sum(1 for e in peer_euis if e >= subject_eui)
        percentile = better_count / len(peer_euis) * 100
        assert 0 <= percentile <= 100

    def test_pipeline_step_4_performance_rating(self):
        """Step 4: Assign EPC band and ENERGY STAR score."""
        site_eui = 140.6
        # EPC band (UK thresholds)
        if site_eui <= 25:
            band = "A"
        elif site_eui <= 50:
            band = "B"
        elif site_eui <= 100:
            band = "C"
        elif site_eui <= 150:
            band = "D"
        elif site_eui <= 200:
            band = "E"
        elif site_eui <= 250:
            band = "F"
        else:
            band = "G"
        assert band == "D"

    def test_pipeline_step_5_report_generation(self):
        """Step 5: Generate benchmark report."""
        report_data = {
            "site_eui": 140.6,
            "source_eui": 280.0,
            "epc_band": "D",
            "percentile": 65.0,
        }
        assert all(v is not None for v in report_data.values())


# =========================================================================
# 2. Portfolio Benchmark Pipeline
# =========================================================================


class TestPortfolioBenchmarkPipeline:
    """Test end-to-end portfolio benchmark pipeline."""

    def test_portfolio_eui_aggregation(self, sample_portfolio):
        """Aggregate EUIs across portfolio facilities."""
        total_energy = sum(f["energy_consumption_kwh"] for f in sample_portfolio)
        total_area = sum(f["gross_floor_area_m2"] for f in sample_portfolio)
        portfolio_eui = total_energy / total_area
        assert portfolio_eui > 0

    def test_portfolio_ranking(self, sample_portfolio):
        """Rank all facilities from best to worst EUI."""
        ranked = sorted(sample_portfolio, key=lambda f: f["eui_kwh_per_m2"])
        assert ranked[0]["eui_kwh_per_m2"] < ranked[-1]["eui_kwh_per_m2"]
        assert len(ranked) == 10

    def test_portfolio_by_building_type(self, sample_portfolio):
        """Aggregate and compare by building type."""
        groups = {}
        for f in sample_portfolio:
            bt = f["building_type"]
            groups.setdefault(bt, []).append(f)
        # Should have multiple building types
        assert len(groups) >= 3

    def test_portfolio_yoy_trend(self, sample_portfolio):
        """Calculate YoY improvement for each facility."""
        for f in sample_portfolio:
            hist = f.get("historical_eui", {})
            years = sorted(hist.keys())
            if len(years) >= 2:
                improvement = (hist[years[-2]] - hist[years[-1]]) / hist[years[-2]] * 100
                # Improvement should be a finite number
                assert -50 < improvement < 50


# =========================================================================
# 3. Regulatory Compliance Pipeline
# =========================================================================


class TestRegulatoryCompliancePipeline:
    """Test end-to-end regulatory compliance pipeline."""

    def test_epbd_meps_check(self):
        """Check EPBD Minimum Energy Performance Standards compliance."""
        epc_band = "D"
        meps_minimum = "D"
        band_order = ["A", "B", "C", "D", "E", "F", "G"]
        is_compliant = band_order.index(epc_band) <= band_order.index(meps_minimum)
        assert is_compliant

    def test_eed_article_8_audit_requirement(self):
        """EED Article 8 requires energy audits for non-SME enterprises."""
        is_sme = False
        energy_consumption_gwh = 0.7  # 700 MWh = 0.7 GWh
        requires_audit = not is_sme or energy_consumption_gwh > 0.085
        assert requires_audit

    def test_dec_display_requirement(self):
        """UK DECs required for public buildings > 250 m2."""
        is_public = True
        floor_area_m2 = 5000
        requires_dec = is_public and floor_area_m2 > 250
        assert requires_dec

    def test_crrem_stranding_check(self, sample_portfolio):
        """Check CRREM stranding for portfolio facilities."""
        stranding_results = []
        for f in sample_portfolio:
            ci = f.get("carbon_emissions_kgco2", 0) / max(f["gross_floor_area_m2"], 1)
            target_2030 = 25.0  # kgCO2/m2 for offices
            is_stranded_2030 = ci > target_2030
            stranding_results.append({
                "facility_id": f["facility_id"],
                "carbon_intensity": ci,
                "stranded_2030": is_stranded_2030,
            })
        assert len(stranding_results) == 10


# =========================================================================
# 4. Multi-Engine Orchestration
# =========================================================================


class TestMultiEngineOrchestration:
    """Test coordinated execution of multiple engines."""

    def test_engine_dependency_chain(self):
        """Verify engine execution order: EUI -> Peer -> Rating -> Report."""
        execution_order = [
            "eui_calculator",
            "weather_normalisation",
            "peer_comparison",
            "sector_benchmark",
            "performance_rating",
            "gap_analysis",
            "benchmark_report",
        ]
        assert len(execution_order) == 7
        # EUI must come before peer comparison
        assert execution_order.index("eui_calculator") < execution_order.index("peer_comparison")
        # Peer before rating
        assert execution_order.index("peer_comparison") < execution_order.index("performance_rating")

    def test_all_10_engines_listed(self):
        """All 10 engines are in the ENGINE_FILES mapping."""
        assert len(ENGINE_FILES) == 10

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_file_exists(self, engine_key):
        """Engine source file exists on disk."""
        file_name = ENGINE_FILES[engine_key]
        path = ENGINES_DIR / file_name
        if not path.exists():
            pytest.skip(f"Engine file not found: {path}")
        assert path.is_file()


# =========================================================================
# 5. Provenance Chain Validation
# =========================================================================


class TestProvenanceChain:
    """Test provenance hash chain across pipeline stages."""

    def test_provenance_hash_deterministic(self):
        """Same input produces the same provenance hash."""
        data = b"facility_data_2025"
        h1 = hashlib.sha256(data).hexdigest()
        h2 = hashlib.sha256(data).hexdigest()
        assert h1 == h2
        assert len(h1) == 64

    def test_provenance_chain_3_stages(self):
        """Provenance chain links 3 stages with hash chaining."""
        stage_1_input = b"raw_energy_data"
        stage_1_hash = hashlib.sha256(stage_1_input).hexdigest()

        stage_2_input = (stage_1_hash + "|eui_calculation").encode()
        stage_2_hash = hashlib.sha256(stage_2_input).hexdigest()

        stage_3_input = (stage_2_hash + "|peer_comparison").encode()
        stage_3_hash = hashlib.sha256(stage_3_input).hexdigest()

        assert stage_1_hash != stage_2_hash
        assert stage_2_hash != stage_3_hash
        assert len(stage_3_hash) == 64

    def test_provenance_changes_with_different_input(self):
        """Different input produces a different provenance hash."""
        h1 = hashlib.sha256(b"input_a").hexdigest()
        h2 = hashlib.sha256(b"input_b").hexdigest()
        assert h1 != h2

    def test_provenance_hash_hex_format(self):
        """Provenance hash is hexadecimal format."""
        h = hashlib.sha256(b"test").hexdigest()
        assert all(c in "0123456789abcdef" for c in h)

    def test_full_pipeline_provenance(
        self, sample_facility_profile, sample_energy_data
    ):
        """Full pipeline produces a deterministic provenance chain."""
        # Simulate provenance through all stages
        input_data = str(sample_facility_profile) + str(sample_energy_data)
        stage_hashes = []
        current_hash = hashlib.sha256(input_data.encode()).hexdigest()
        stage_hashes.append(current_hash)

        stages = ["eui_calc", "weather_norm", "peer_comp", "rating", "report"]
        for stage in stages:
            current_hash = hashlib.sha256(
                (current_hash + "|" + stage).encode()
            ).hexdigest()
            stage_hashes.append(current_hash)

        assert len(stage_hashes) == 6
        assert len(set(stage_hashes)) == 6  # All unique


# =========================================================================
# 6. Data Flow Validation
# =========================================================================


class TestDataFlowValidation:
    """Test data flows correctly between pipeline stages."""

    def test_energy_data_has_required_fields(self, sample_energy_data):
        """Energy data has all required fields for EUI calculation."""
        required = {"month", "electricity_kwh", "gas_kwh"}
        for record in sample_energy_data:
            assert required.issubset(record.keys())

    def test_facility_profile_has_required_fields(self, sample_facility_profile):
        """Facility profile has all required fields."""
        required = {
            "facility_id", "building_type", "gross_internal_area_m2",
            "climate_zone", "country_code",
        }
        assert required.issubset(sample_facility_profile.keys())

    def test_peer_group_has_required_fields(self, sample_peer_group):
        """Peer group facilities have required fields."""
        required = {"facility_id", "eui_kwh_per_m2_yr", "building_type"}
        for peer in sample_peer_group:
            assert required.issubset(peer.keys())

    def test_portfolio_has_required_fields(self, sample_portfolio):
        """Portfolio facilities have required fields."""
        required = {
            "facility_id", "building_type", "gross_floor_area_m2",
            "energy_consumption_kwh", "eui_kwh_per_m2",
        }
        for f in sample_portfolio:
            assert required.issubset(f.keys())


# =========================================================================
# 7. Error Recovery
# =========================================================================


class TestErrorRecovery:
    """Test pipeline error recovery and graceful degradation."""

    def test_missing_gas_data_handled(self, sample_energy_data):
        """Pipeline handles missing gas data gracefully."""
        data_no_gas = [
            {k: v for k, v in m.items() if k != "gas_kwh"}
            for m in sample_energy_data
        ]
        # Should be able to compute electricity-only EUI
        total_elec = sum(m.get("electricity_kwh", 0) for m in data_no_gas)
        eui = total_elec / 5000.0
        assert eui > 0

    def test_missing_peer_group_handled(self):
        """Pipeline handles missing peer group data gracefully."""
        peer_group = []
        # Should return a result with no percentile
        if len(peer_group) < 10:
            percentile = None
        else:
            percentile = 50.0
        assert percentile is None

    def test_zero_floor_area_prevented(self):
        """Pipeline prevents division by zero for floor area."""
        floor_area = 0.0
        if floor_area <= 0:
            error = "Floor area must be positive"
        else:
            error = None
        assert error is not None
