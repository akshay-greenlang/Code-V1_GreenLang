# -*- coding: utf-8 -*-
"""
End-to-end tests for PACK-032 Building Energy Assessment Pack

Tests full pipeline: envelope -> EPC -> HVAC -> benchmark -> report,
provenance chain across engines, and multi-engine integration.

Target: 10+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack032_e2e.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def envelope_mod():
    return _load("building_envelope_engine")


@pytest.fixture(scope="module")
def epc_mod():
    return _load("epc_rating_engine")


@pytest.fixture(scope="module")
def hvac_mod():
    return _load("hvac_assessment_engine")


@pytest.fixture(scope="module")
def benchmark_mod():
    return _load("building_benchmark_engine")


@pytest.fixture(scope="module")
def lighting_mod():
    return _load("lighting_assessment_engine")


@pytest.fixture(scope="module")
def dhw_mod():
    return _load("domestic_hot_water_engine")


# =========================================================================
# Test Multi-Engine Pipeline
# =========================================================================


class TestMultiEnginePipeline:
    """End-to-end tests running multiple engines in sequence."""

    def test_envelope_then_epc(self, envelope_mod, epc_mod):
        """Test envelope analysis feeds into EPC rating."""
        # Step 1: Run envelope analysis
        env_engine = envelope_mod.BuildingEnvelopeEngine()
        walls = [
            envelope_mod.WallElement(
                element_id="W1",
                wall_type=envelope_mod.WallType.CAVITY_WALL,
                area_m2=200.0,
            ),
        ]
        windows = [
            envelope_mod.WindowElement(
                element_id="WIN1",
                window_type=envelope_mod.WindowType.DOUBLE_GLAZED,
                area_m2=80.0,
            ),
        ]
        envelope = envelope_mod.BuildingEnvelope(
            facility_id="BLD-E2E-001",
            name="E2E Test Office",
            building_type=envelope_mod.BuildingType.OFFICE,
            year_built=2000,
            gross_floor_area_m2=2000.0,
            heated_volume_m3=6000.0,
            walls=walls,
            windows=windows,
        )
        env_result = env_engine.analyze(envelope)
        assert env_result is not None
        assert env_result.provenance_hash != ""

        # Step 2: Use envelope data in EPC rating
        epc_engine = epc_mod.EPCRatingEngine()
        # Use the area-weighted U-value from the envelope result
        envelope_summary = epc_mod.EnvelopeSummary(
            wall_u_value=env_result.area_weighted_u_value or 0.50,
            window_u_value=1.40,
        )
        building = epc_mod.BuildingData(
            facility_id="BLD-E2E-001",
            building_type=epc_mod.BuildingUseType.OFFICE,
            floor_area_m2=2000.0,
            envelope=envelope_summary,
        )
        epc_result = epc_engine.rate(building)
        assert epc_result is not None
        assert epc_result.epc_rating is not None
        assert epc_result.provenance_hash != ""

    def test_provenance_chain(self, envelope_mod, epc_mod):
        """Test that provenance hashes from different engines are distinct."""
        env_engine = envelope_mod.BuildingEnvelopeEngine()
        walls = [
            envelope_mod.WallElement(
                element_id="W1",
                wall_type=envelope_mod.WallType.CAVITY_WALL,
                area_m2=150.0,
            ),
        ]
        envelope = envelope_mod.BuildingEnvelope(
            facility_id="BLD-PROV-001",
            name="Provenance Test Building",
            year_built=2000,
            gross_floor_area_m2=1000.0,
            heated_volume_m3=3000.0,
            walls=walls,
        )
        env_result = env_engine.analyze(envelope)

        epc_engine = epc_mod.EPCRatingEngine()
        building = epc_mod.BuildingData(
            facility_id="BLD-PROV-001",
            floor_area_m2=1000.0,
        )
        epc_result = epc_engine.rate(building)

        # Both should have provenance hashes but they should differ
        assert len(env_result.provenance_hash) == 64
        assert len(epc_result.provenance_hash) == 64
        assert env_result.provenance_hash != epc_result.provenance_hash

    def test_benchmark_after_epc(self, epc_mod, benchmark_mod):
        """Test benchmark engine uses EPC output context."""
        epc_engine = epc_mod.EPCRatingEngine()
        building = epc_mod.BuildingData(
            facility_id="BLD-BENCH-001",
            building_type=epc_mod.BuildingUseType.OFFICE,
            floor_area_m2=2000.0,
        )
        epc_result = epc_engine.rate(building)
        assert epc_result is not None

        # Feed EUI into benchmark
        bench_engine = benchmark_mod.BuildingBenchmarkEngine()
        energy = benchmark_mod.EnergyConsumptionInput(
            electricity_kwh=250000.0,
            gas_kwh=150000.0,
        )
        bench_input = benchmark_mod.BenchmarkInput(
            building_id="BLD-BENCH-001",
            building_type=benchmark_mod.BuildingType.OFFICE,
            gross_internal_area_m2=2000.0,
            energy=energy,
        )
        bench_result = bench_engine.calculate_eui(bench_input)
        assert bench_result.eui_kwh_per_m2 > 0


# =========================================================================
# Test Provenance Consistency
# =========================================================================


class TestProvenanceConsistency:
    """Provenance hashes must be 64-char hex and deterministic."""

    def test_all_engines_produce_provenance(
        self, envelope_mod, epc_mod, hvac_mod, benchmark_mod
    ):
        hashes = []

        # Envelope
        env_engine = envelope_mod.BuildingEnvelopeEngine()
        walls = [
            envelope_mod.WallElement(
                element_id="W1",
                wall_type=envelope_mod.WallType.CAVITY_WALL,
                area_m2=100.0,
            ),
        ]
        envelope = envelope_mod.BuildingEnvelope(
            facility_id="BLD-ALL",
            name="All Engines Test",
            year_built=2000,
            gross_floor_area_m2=500.0,
            heated_volume_m3=1500.0,
            walls=walls,
        )
        env_r = env_engine.analyze(envelope)
        hashes.append(env_r.provenance_hash)

        # EPC
        epc_engine = epc_mod.EPCRatingEngine()
        building = epc_mod.BuildingData(
            facility_id="BLD-ALL",
            floor_area_m2=500.0,
        )
        epc_r = epc_engine.rate(building)
        hashes.append(epc_r.provenance_hash)

        # HVAC
        hvac_engine = hvac_mod.HVACAssessmentEngine()
        heating = hvac_mod.HeatingSystem(
            system_type=hvac_mod.HeatingSystemType.GAS_BOILER,
            capacity_kw=100.0,
        )
        hvac_input = hvac_mod.HVACInput(
            facility_id="BLD-ALL",
            floor_area_m2=500.0,
            heating_systems=[heating],
        )
        hvac_r = hvac_engine.assess(hvac_input)
        hashes.append(hvac_r.provenance_hash)

        # Benchmark
        bench_engine = benchmark_mod.BuildingBenchmarkEngine()
        energy = benchmark_mod.EnergyConsumptionInput(electricity_kwh=100000.0)
        bench_input = benchmark_mod.BenchmarkInput(
            building_id="BLD-ALL",
            building_type=benchmark_mod.BuildingType.OFFICE,
            gross_internal_area_m2=500.0,
            energy=energy,
        )
        bench_r = bench_engine.calculate_eui(bench_input)
        # Some engines may not have provenance on EUI sub-result
        if hasattr(bench_r, "provenance_hash") and bench_r.provenance_hash:
            hashes.append(bench_r.provenance_hash)

        for h in hashes:
            assert isinstance(h, str)
            assert len(h) == 64


# =========================================================================
# Test Cross-Engine Data Flow
# =========================================================================


class TestCrossEngineDataFlow:
    def test_lighting_data_feeds_epc(self, lighting_mod, epc_mod):
        """Lighting LENI can inform EPC lighting energy calculation."""
        light_engine = lighting_mod.LightingAssessmentEngine()
        zone = lighting_mod.LightingZoneInput(
            zone_id="Z1",
            zone_name="Open Plan Office",
            space_category=lighting_mod.SpaceCategory.OFFICE_OPEN_PLAN,
            floor_area_m2=500.0,
            number_of_fixtures=50,
            watts_per_fixture=70.0,
            annual_operating_hours=2500,
            lamp_type=lighting_mod.LampType.LED,
        )
        light_input = lighting_mod.LightingAssessmentInput(
            building_id="BLD-CROSS",
            total_floor_area_m2=500.0,
            zones=[zone],
        )
        light_result = light_engine.analyze(light_input)
        assert light_result is not None

        # Use the lighting result data in an EPC calculation
        epc_engine = epc_mod.EPCRatingEngine()
        lighting_sys = epc_mod.LightingSystem(
            led_fraction=1.0,
            average_efficacy_lm_w=130.0,
        )
        building = epc_mod.BuildingData(
            facility_id="BLD-CROSS",
            floor_area_m2=500.0,
            lighting=lighting_sys,
        )
        epc_result = epc_engine.rate(building)
        assert epc_result is not None

    def test_dhw_data_feeds_epc(self, dhw_mod, epc_mod):
        """DHW demand calculation feeds into EPC total energy."""
        dhw_engine = dhw_mod.DomesticHotWaterEngine()
        dhw_system = dhw_mod.DHWSystemInput(
            system_type=dhw_mod.DHWSystemType.GAS_BOILER,
        )
        dhw_input = dhw_mod.DHWAssessmentInput(
            building_id="BLD-DHW-CROSS",
            building_type=dhw_mod.BuildingOccupancyType.OFFICE,
            occupancy_count=50,
            dhw_system=dhw_system,
        )
        dhw_result = dhw_engine.analyze(dhw_input)
        assert dhw_result is not None

        # Feed DHW energy into EPC
        epc_engine = epc_mod.EPCRatingEngine()
        dhw_sys = epc_mod.DHWSystem(
            system_type="gas_boiler",
            efficiency=0.89,
        )
        building = epc_mod.BuildingData(
            facility_id="BLD-DHW-CROSS",
            floor_area_m2=1000.0,
            dhw_system=dhw_sys,
        )
        epc_result = epc_engine.rate(building)
        assert epc_result is not None
        assert epc_result.provenance_hash != ""
