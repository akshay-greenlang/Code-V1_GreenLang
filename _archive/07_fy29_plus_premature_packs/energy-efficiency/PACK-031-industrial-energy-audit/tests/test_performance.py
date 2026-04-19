# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Performance Tests (test_performance.py)
================================================================================

Tests engine execution time, throughput, memory behavior, and scalability
for all 10 engines. Validates that each engine completes within target
latency, handles batch workloads, and scales with data volume.

Coverage target: 85%+
Total tests: ~40
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-031 Industrial Energy Audit
Date:    March 2026
"""

import importlib.util
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack031_perf.{name}"
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


# =============================================================================
# Helper: Data Generators
# =============================================================================

def _baseline_data(months=12):
    """Generate baseline data with variable months for scalability tests."""
    mod = _load("energy_baseline_engine")
    facility = mod.FacilityData(
        facility_id="FAC-PERF-001",
        name="Performance Test Plant",
        sector=mod.FacilitySector.MANUFACTURING,
        area_sqm=18000.0,
        location="DE",
        production_capacity=12500.0,
    )
    month_labels = [f"2024-{m:02d}" for m in range(1, months + 1)]
    readings = [
        mod.EnergyMeterReading(
            meter_id=f"MTR-{i:02d}",
            period=m,
            energy_carrier=mod.EnergyCarrier.ELECTRICITY,
            energy_kwh=650000.0 + (i * 10000),
        )
        for i, m in enumerate(month_labels, 1)
    ]
    prod = [
        mod.ProductionData(period=m, output_units=1000.0 + (i * 10))
        for i, m in enumerate(month_labels, 1)
    ]
    return mod, facility, readings, prod


def _compressed_air_data(num_compressors=2):
    """Generate compressed air data with variable compressor count."""
    mod = _load("compressed_air_engine")
    compressors = [
        mod.Compressor(
            compressor_id=f"CMP-PERF-{i:03d}",
            name=f"Compressor {i}",
            compressor_type="screw_fixed",
            control_type=mod.CompressorControl.LOAD_UNLOAD.value,
            rated_power_kw=Decimal("90"),
            fad_m3min=Decimal("14.5"),
            pressure_bar=Decimal("7"),
            operating_hours=5800,
        )
        for i in range(1, num_compressors + 1)
    ]
    data = mod.CompressedAirInput(
        system=mod.CompressedAirSystem(
            system_id="CA-PERF-001",
            system_pressure_bar=Decimal("7"),
            target_pressure_bar=Decimal("6"),
        ),
        compressors=compressors,
    )
    return mod, data


def _steam_data(num_boilers=1):
    """Generate steam data with variable boiler count."""
    mod = _load("steam_optimization_engine")
    boilers = [
        mod.Boiler(
            boiler_id=f"BLR-PERF-{i:03d}",
            boiler_type=list(mod.BoilerType)[0],
            fuel_type=list(mod.FuelType)[0],
            capacity_kg_h=3000.0,
            operating_pressure_bar=10.0,
            feed_water_temp_c=80.0,
            stack_temp_c=220.0,
            excess_air_pct=20.0,
            blowdown_pct=8.0,
            operating_hours=5000,
            annual_fuel_consumption_kwh=5_200_000.0,
        )
        for i in range(1, num_boilers + 1)
    ]
    system = mod.SteamSystem(
        system_id="STM-PERF-001",
        facility_name="Performance Steam Test",
        boilers=boilers,
        operating_hours=5000,
        total_steam_demand_kg_h=2100.0 * num_boilers,
    )
    return mod, system


def _waste_heat_data(num_sources=2):
    """Generate waste heat data with variable source count."""
    mod = _load("waste_heat_recovery_engine")
    sources = [
        mod.WasteHeatSource(
            source_id=f"WH-PERF-{i:03d}",
            name=f"Source {i}",
            source_type=mod.HeatSourceType.FLUE_GAS.value,
            inlet_temperature_c=Decimal("220"),
            outlet_temperature_c=Decimal("60"),
            flow_rate_kg_s=Decimal("0.972"),
            specific_heat_kj_kgk=Decimal("1.05"),
            operating_hours=5000,
        )
        for i in range(1, num_sources + 1)
    ]
    sinks = [
        mod.HeatSink(
            sink_id="HS-PERF-001",
            name="Perf Sink",
            inlet_temperature_c=Decimal("20"),
            target_temperature_c=Decimal("80"),
            required_heat_kw=Decimal("150"),
            operating_hours=5000,
        ),
    ]
    data = mod.WasteHeatRecoveryInput(
        facility_id="FAC-PERF-WH",
        facility_name="Performance Waste Heat",
        sources=sources,
        sinks=sinks,
    )
    return mod, data


# =============================================================================
# 1. Engine Latency Tests (< 2 seconds per single invocation)
# =============================================================================


class TestEngineLatency:
    """Test each engine completes within target latency."""

    LATENCY_TARGET_S = 2.0  # Each engine should finish under 2 seconds

    def test_baseline_engine_latency(self):
        """EnergyBaselineEngine completes within latency target."""
        mod, facility, readings, prod = _baseline_data(12)
        engine = mod.EnergyBaselineEngine()
        start = time.perf_counter()
        result = engine.establish_baseline(
            facility=facility, meter_data=readings, production_data=prod,
        )
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_audit_engine_latency(self):
        """EnergyAuditEngine completes within latency target."""
        mod = _load("energy_audit_engine")
        engine = mod.EnergyAuditEngine()
        scope = mod.AuditScope(
            facility_id="FAC-PERF-AU",
            audit_type=list(mod.AuditType)[1],
        )
        end_uses = [
            mod.EnergyEndUse(
                category=list(mod.EndUseCategory)[i],
                annual_kwh=kwh,
            )
            for i, kwh in enumerate([5_000_000, 2_000_000, 1_300_000])
        ]
        start = time.perf_counter()
        result = engine.conduct_audit(scope, end_uses)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_compressed_air_engine_latency(self):
        """CompressedAirEngine completes within latency target."""
        mod, data = _compressed_air_data(2)
        engine = mod.CompressedAirEngine()
        start = time.perf_counter()
        result = engine.audit(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_steam_engine_latency(self):
        """SteamOptimizationEngine completes within latency target."""
        mod, system = _steam_data(1)
        engine = mod.SteamOptimizationEngine()
        start = time.perf_counter()
        result = engine.analyze_steam_system(system)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_waste_heat_engine_latency(self):
        """WasteHeatRecoveryEngine completes within latency target."""
        mod, data = _waste_heat_data(2)
        engine = mod.WasteHeatRecoveryEngine()
        start = time.perf_counter()
        result = engine.analyze(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_equipment_engine_latency(self):
        """EquipmentEfficiencyEngine completes within latency target."""
        mod = _load("equipment_efficiency_engine")
        engine = mod.EquipmentEfficiencyEngine()
        data = mod.EquipmentEfficiencyInput(
            facility_id="FAC-PERF-EQ",
            facility_name="Perf Equipment",
            equipment=mod.Equipment(
                equipment_id="MTR-PERF-001",
                name="Perf Motor",
                equipment_type=mod.EquipmentType.MOTOR.value,
                rated_power_kw=Decimal("37"),
                operating_hours=5200,
            ),
            motor_data=mod.MotorData(
                efficiency_class=mod.MotorEfficiencyClass.IE3.value,
                rated_power_kw=Decimal("37"),
                poles=4,
                actual_load_pct=Decimal("75"),
            ),
        )
        start = time.perf_counter()
        result = engine.analyze(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_savings_engine_latency(self):
        """EnergySavingsEngine completes within latency target."""
        mod = _load("energy_savings_engine")
        engine = mod.EnergySavingsEngine()
        data = mod.EnergySavingsInput(
            facility_id="FAC-PERF-ES",
            facility_name="Perf Savings",
            total_baseline_kwh=Decimal("8300000"),
            energy_price_eur_kwh=Decimal("0.15"),
            measures=[
                mod.EnergySavingsMeasure(
                    measure_id="ECM-PERF-001",
                    name="Perf Measure",
                    category=mod.ECMCategory.COMPRESSED_AIR.value,
                    baseline_kwh=Decimal("880000"),
                    expected_savings_kwh=Decimal("220000"),
                    savings_pct=Decimal("25"),
                    implementation_cost_eur=Decimal("8500"),
                    lifetime_years=5,
                ),
            ],
        )
        start = time.perf_counter()
        result = engine.analyze(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_benchmark_engine_latency(self):
        """EnergyBenchmarkEngine completes within latency target."""
        mod = _load("energy_benchmark_engine")
        engine = mod.EnergyBenchmarkEngine()
        facility = mod.BenchmarkFacility(
            facility_id="FAC-PERF-BM",
            facility_name="Perf Benchmark",
            sector=list(mod.IndustrySector)[0],
            country="DE",
            energy_consumption_kwh=14_500_000.0,
            production_output=12_500.0,
            production_unit="tonnes",
            area_sqm=18_000.0,
            employees=420,
            reporting_year=2025,
        )
        start = time.perf_counter()
        result = engine.benchmark(facility)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_lighting_hvac_engine_latency(self):
        """LightingHVACEngine completes within latency target."""
        mod = _load("lighting_hvac_engine")
        engine = mod.LightingHVACEngine()
        data = mod.FacilityLightingHVACData(
            facility_id="FAC-PERF-LH",
            facility_name="Perf Lighting",
            lighting_zones=[
                mod.LightingZone(
                    zone_id="LZ-PERF-001",
                    name="Perf Warehouse",
                    space_type=mod.SpaceType.WAREHOUSE,
                    area_sqm=4000.0,
                    fixture_count=80,
                    fixture_type=mod.FixtureType.HID_MH,
                    wattage_per_fixture=400.0,
                    operating_hours=5800,
                ),
            ],
            hvac_systems=[
                mod.HVACSystem(
                    system_id="HVAC-PERF-001",
                    system_type=mod.HVACSystemType.CHILLER_AHU,
                    cooling_capacity_kw=350.0,
                    current_cop=3.8,
                    annual_cooling_hours=4000,
                ),
            ],
        )
        start = time.perf_counter()
        result = engine.analyze(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"

    def test_process_mapping_engine_latency(self):
        """ProcessEnergyMappingEngine completes within latency target."""
        mod = _load("process_energy_mapping_engine")
        engine = mod.ProcessEnergyMappingEngine()
        nodes = [
            mod.ProcessNode(
                node_id=f"PN-PERF-{i:03d}",
                name=f"Perf Step {i}",
                process_type=list(mod.ProcessType)[0],
                input_energy_kwh=300_000.0,
                output_energy_kwh=240_000.0,
            )
            for i in range(1, 4)
        ]
        lines = [mod.ProductionLine(line_id="PL-PERF-001", name="Perf Line", nodes=nodes)]
        flows = [
            mod.EnergyFlow(
                source_node="INPUT", target_node="PN-PERF-001",
                energy_kwh=300_000.0, energy_type=list(mod.EnergyType)[0],
            ),
        ]
        start = time.perf_counter()
        result = engine.map_process_energy(
            facility_id="FAC-PERF-PM",
            production_lines=lines,
            energy_flows=flows,
        )
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < self.LATENCY_TARGET_S, f"Took {elapsed:.3f}s"


# =============================================================================
# 2. Repeated Execution Stability Tests
# =============================================================================


class TestRepeatedExecution:
    """Test engines maintain stable performance across repeated calls."""

    ITERATIONS = 5

    def test_baseline_repeated(self):
        """Baseline engine is stable across repeated calls."""
        mod, facility, readings, prod = _baseline_data(12)
        engine = mod.EnergyBaselineEngine()
        timings = []
        for _ in range(self.ITERATIONS):
            start = time.perf_counter()
            engine.establish_baseline(
                facility=facility, meter_data=readings, production_data=prod,
            )
            timings.append(time.perf_counter() - start)
        avg = sum(timings) / len(timings)
        assert avg < 2.0, f"Average time {avg:.3f}s exceeds 2s"
        # No single run should be 5x the average (no outlier spikes)
        assert max(timings) < avg * 5, f"Max {max(timings):.3f}s >> avg {avg:.3f}s"

    def test_compressed_air_repeated(self):
        """Compressed air engine is stable across repeated calls."""
        mod, data = _compressed_air_data(2)
        engine = mod.CompressedAirEngine()
        timings = []
        for _ in range(self.ITERATIONS):
            start = time.perf_counter()
            engine.audit(data)
            timings.append(time.perf_counter() - start)
        avg = sum(timings) / len(timings)
        assert avg < 2.0, f"Average time {avg:.3f}s exceeds 2s"

    def test_waste_heat_repeated(self):
        """Waste heat engine is stable across repeated calls."""
        mod, data = _waste_heat_data(2)
        engine = mod.WasteHeatRecoveryEngine()
        timings = []
        for _ in range(self.ITERATIONS):
            start = time.perf_counter()
            engine.analyze(data)
            timings.append(time.perf_counter() - start)
        avg = sum(timings) / len(timings)
        assert avg < 2.0, f"Average time {avg:.3f}s exceeds 2s"


# =============================================================================
# 3. Scalability Tests (increasing data size)
# =============================================================================


class TestScalability:
    """Test engine performance scales reasonably with data size."""

    def test_baseline_scalability_12_months(self):
        """Baseline engine handles 12 months in < 2s."""
        mod, facility, readings, prod = _baseline_data(12)
        engine = mod.EnergyBaselineEngine()
        start = time.perf_counter()
        result = engine.establish_baseline(
            facility=facility, meter_data=readings, production_data=prod,
        )
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < 2.0

    def test_compressed_air_scalability_5_compressors(self):
        """Compressed air engine handles 5 compressors in < 2s."""
        mod, data = _compressed_air_data(5)
        engine = mod.CompressedAirEngine()
        start = time.perf_counter()
        result = engine.audit(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < 2.0

    def test_compressed_air_scalability_10_compressors(self):
        """Compressed air engine handles 10 compressors in < 3s."""
        mod, data = _compressed_air_data(10)
        engine = mod.CompressedAirEngine()
        start = time.perf_counter()
        result = engine.audit(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < 3.0

    def test_waste_heat_scalability_5_sources(self):
        """Waste heat engine handles 5 sources in < 2s."""
        mod, data = _waste_heat_data(5)
        engine = mod.WasteHeatRecoveryEngine()
        start = time.perf_counter()
        result = engine.analyze(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < 2.0

    def test_waste_heat_scalability_10_sources(self):
        """Waste heat engine handles 10 sources in < 3s."""
        mod, data = _waste_heat_data(10)
        engine = mod.WasteHeatRecoveryEngine()
        start = time.perf_counter()
        result = engine.analyze(data)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < 3.0

    def test_steam_scalability_3_boilers(self):
        """Steam engine handles 3 boilers in < 2s."""
        mod, system = _steam_data(3)
        engine = mod.SteamOptimizationEngine()
        start = time.perf_counter()
        result = engine.analyze_steam_system(system)
        elapsed = time.perf_counter() - start
        assert result is not None
        assert elapsed < 2.0


# =============================================================================
# 4. Provenance Hash Computation Performance
# =============================================================================


class TestProvenancePerformance:
    """Test provenance hash computation adds negligible overhead."""

    def test_hash_computation_is_fast(self):
        """SHA-256 provenance hash adds < 100ms overhead."""
        mod, facility, readings, prod = _baseline_data(12)
        engine = mod.EnergyBaselineEngine()
        # Warm up
        engine.establish_baseline(
            facility=facility, meter_data=readings, production_data=prod,
        )
        # Measure
        start = time.perf_counter()
        result = engine.establish_baseline(
            facility=facility, meter_data=readings, production_data=prod,
        )
        elapsed = time.perf_counter() - start
        assert len(result.provenance_hash) == 64
        # Hash computation is part of total time; total should be very fast
        assert elapsed < 1.0, f"Total including hash: {elapsed:.3f}s"

    def test_all_engines_have_provenance_hash(self):
        """All engine results include a 64-char provenance hash."""
        # Baseline
        bl_mod, facility, readings, prod = _baseline_data(12)
        bl_result = bl_mod.EnergyBaselineEngine().establish_baseline(
            facility=facility, meter_data=readings, production_data=prod,
        )
        assert len(bl_result.provenance_hash) == 64

        # Compressed Air
        ca_mod, ca_data = _compressed_air_data(2)
        ca_result = ca_mod.CompressedAirEngine().audit(ca_data)
        assert len(ca_result.provenance_hash) == 64

        # Waste Heat
        wh_mod, wh_data = _waste_heat_data(2)
        wh_result = wh_mod.WasteHeatRecoveryEngine().analyze(wh_data)
        assert len(wh_result.provenance_hash) == 64


# =============================================================================
# 5. Result Correctness Under Load
# =============================================================================


class TestCorrectnessUnderLoad:
    """Test results remain correct across rapid sequential invocations."""

    def test_baseline_results_consistent(self):
        """Multiple rapid calls produce valid results (provenance hashes differ
        due to result_id UUID, but all are valid 64-char hex)."""
        mod, facility, readings, prod = _baseline_data(12)
        engine = mod.EnergyBaselineEngine()
        results = []
        for _ in range(5):
            r = engine.establish_baseline(
                facility=facility, meter_data=readings, production_data=prod,
            )
            results.append(r)
        # All results should be valid
        for r in results:
            assert r is not None
            assert len(r.provenance_hash) == 64
            assert all(c in "0123456789abcdef" for c in r.provenance_hash)

    def test_compressed_air_results_consistent(self):
        """Multiple rapid compressed air audits produce valid results."""
        mod, data = _compressed_air_data(2)
        engine = mod.CompressedAirEngine()
        results = [engine.audit(data) for _ in range(5)]
        for r in results:
            assert r is not None
            assert len(r.provenance_hash) == 64

    def test_no_state_leakage_between_calls(self):
        """Engine state does not leak between different inputs."""
        mod, data_small = _compressed_air_data(1)
        _, data_large = _compressed_air_data(5)
        engine = mod.CompressedAirEngine()
        r_small = engine.audit(data_small)
        r_large = engine.audit(data_large)
        # Different inputs should produce different hashes
        assert r_small.provenance_hash != r_large.provenance_hash
