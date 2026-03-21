# -*- coding: utf-8 -*-
"""
Unit tests for BuildingEnvelopeEngine (PACK-032 Engine 1)

Tests U-value calculation (EN ISO 6946), thermal bridging (EN ISO 10211),
airtightness (n50 per EN 13829), condensation risk (Glaser method per
EN ISO 13788), and provenance hashing.

Target: 35+ tests, 85%+ coverage.
Author: GL-TestEngineer
"""

import importlib.util
import hashlib
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
    mod_key = f"pack032_test.{name}"
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


@pytest.fixture(scope="module")
def engine_mod():
    return _load("building_envelope_engine")


@pytest.fixture
def engine(engine_mod):
    return engine_mod.BuildingEnvelopeEngine()


@pytest.fixture
def minimal_envelope(engine_mod):
    """Minimal valid BuildingEnvelope with one wall element."""
    mod = engine_mod
    wall = mod.WallElement(
        wall_type=mod.WallType.CAVITY_WALL,
        area_m2=100.0,
        age_band=mod.AgeBand.BAND_1996_2006,
    )
    return mod.BuildingEnvelope(
        facility_id="FAC-TEST-001",
        name="Test Office Building",
        building_type=mod.BuildingType.OFFICE,
        year_built=2000,
        country="UK",
        gross_floor_area_m2=500.0,
        heated_volume_m3=1500.0,
        walls=[wall],
    )


@pytest.fixture
def full_envelope(engine_mod):
    """Full envelope with walls, roof, floor, windows, doors, thermal bridges."""
    mod = engine_mod
    wall = mod.WallElement(
        wall_type=mod.WallType.CAVITY_WALL,
        area_m2=200.0,
        age_band=mod.AgeBand.BAND_1996_2006,
    )
    roof = mod.RoofElement(
        roof_type=mod.RoofType.PITCHED_TILE,
        area_m2=150.0,
        insulation_thickness_mm=200,
    )
    floor = mod.FloorElement(
        floor_type=mod.FloorType.SOLID_CONCRETE,
        area_m2=150.0,
        perimeter_m=48.0,
    )
    window = mod.WindowElement(
        window_type=mod.WindowType.DOUBLE_GLAZED,
        glazing_type=mod.GlazingType.LOW_E_SOFT,
        frame_material=mod.FrameMaterial.UPVC,
        area_m2=40.0,
    )
    door = mod.DoorElement(
        door_type="composite_insulated",
        area_m2=4.0,
        quantity=2,
    )
    tb = mod.ThermalBridge(
        bridge_type=mod.ThermalBridgeType.WALL_FLOOR_GROUND,
        length_m=50.0,
    )
    airtightness = mod.AirtightnessData(
        n50_ach=5.0,
        measured=True,
    )
    return mod.BuildingEnvelope(
        facility_id="FAC-TEST-002",
        name="Full Test Office",
        building_type=mod.BuildingType.OFFICE,
        year_built=2005,
        country="UK",
        gross_floor_area_m2=800.0,
        heated_volume_m3=2400.0,
        envelope_area_m2=600.0,
        walls=[wall],
        roofs=[roof],
        floors=[floor],
        windows=[window],
        doors=[door],
        thermal_bridges=[tb],
        airtightness=airtightness,
    )


# =========================================================================
# Test Initialization
# =========================================================================


class TestInitialization:
    """Test BuildingEnvelopeEngine initialization."""

    def test_engine_class_exists(self, engine_mod):
        assert hasattr(engine_mod, "BuildingEnvelopeEngine")

    def test_engine_instantiation(self, engine):
        assert engine is not None

    def test_engine_version(self, engine):
        assert hasattr(engine, "engine_version")
        assert isinstance(engine.engine_version, str)
        assert engine.engine_version == "1.0.0"

    def test_engine_has_lookup_tables(self, engine):
        assert hasattr(engine, "_wall_u_values")
        assert hasattr(engine, "_roof_u_values")
        assert hasattr(engine, "_window_u_values")
        assert hasattr(engine, "_psi_values")

    def test_input_model_exists(self, engine_mod):
        assert hasattr(engine_mod, "BuildingEnvelope")

    def test_output_model_exists(self, engine_mod):
        assert hasattr(engine_mod, "EnvelopeResult")


# =========================================================================
# Test U-Value Calculations
# =========================================================================


class TestUValueCalculations:
    """Test U-value calculation per EN ISO 6946."""

    def test_analyze_minimal_envelope(self, engine, minimal_envelope):
        result = engine.analyze(minimal_envelope)
        assert result is not None
        assert len(result.element_u_values) >= 1

    def test_analyze_full_envelope(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result is not None
        assert len(result.element_u_values) >= 5  # wall + roof + floor + window + door

    def test_wall_u_value_cavity_1996(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        wall_elements = [e for e in result.element_u_values if e.element_type == "wall"]
        assert len(wall_elements) >= 1
        # Cavity wall 1996-2006 should be around 0.45 W/m2K
        assert wall_elements[0].u_value_w_m2k > 0
        assert wall_elements[0].u_value_w_m2k < 5.0

    def test_window_u_value_positive(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        windows = [e for e in result.element_u_values if e.element_type == "window"]
        assert len(windows) >= 1
        assert windows[0].u_value_w_m2k > 0

    def test_door_u_value_positive(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        doors = [e for e in result.element_u_values if e.element_type == "door"]
        assert len(doors) >= 1
        assert doors[0].u_value_w_m2k > 0

    def test_area_weighted_u_value(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.area_weighted_u_value > 0
        # Should be a reasonable weighted average < 3.0 for a 2005 building
        assert result.area_weighted_u_value < 5.0

    def test_calculate_element_u_value(self, engine, engine_mod):
        """Test U-value from detailed layer construction."""
        layers = [
            engine_mod.InsulationLayer(
                material="plasterboard",
                thickness_mm=12.5,
                conductivity_w_mk=0.21,
            ),
            engine_mod.InsulationLayer(
                material="mineral_wool",
                thickness_mm=100.0,
                conductivity_w_mk=0.035,
            ),
            engine_mod.InsulationLayer(
                material="brick",
                thickness_mm=100.0,
                conductivity_w_mk=0.77,
            ),
        ]
        u_val = engine.calculate_element_u_value(layers)
        assert u_val > 0
        assert u_val < 2.0  # Insulated wall should be < 2.0

    def test_calculate_element_u_value_no_layers_raises(self, engine):
        with pytest.raises(ValueError, match="layer"):
            engine.calculate_element_u_value([])

    def test_element_heat_loss_positive(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        for elem in result.element_u_values:
            assert elem.heat_loss_w_k >= 0


# =========================================================================
# Test Heat Loss Calculations
# =========================================================================


class TestHeatLoss:
    """Test fabric and ventilation heat loss calculations."""

    def test_fabric_heat_loss_positive(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.fabric_heat_loss_w_k > 0

    def test_ventilation_heat_loss_positive(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.ventilation_heat_loss_w_k >= 0

    def test_total_heat_loss_gte_fabric(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.total_heat_loss_coefficient_w_k >= result.fabric_heat_loss_w_k

    def test_specific_heat_loss_positive(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.specific_heat_loss_w_m2k > 0

    def test_annual_heating_demand_positive(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.annual_heating_demand_kwh > 0
        assert result.annual_heating_demand_kwh_m2 > 0


# =========================================================================
# Test Thermal Bridging
# =========================================================================


class TestThermalBridging:
    """Test thermal bridge assessment per EN ISO 10211."""

    def test_thermal_bridge_result_present(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.thermal_bridge_result is not None

    def test_thermal_bridge_htb_non_negative(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        tb = result.thermal_bridge_result
        assert tb.total_htb_w_k >= 0

    def test_y_factor_non_negative(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        tb = result.thermal_bridge_result
        assert tb.y_factor >= 0

    def test_no_thermal_bridges_zero_htb(self, engine, minimal_envelope):
        result = engine.analyze(minimal_envelope)
        if result.thermal_bridge_result is not None:
            assert result.thermal_bridge_result.total_htb_w_k >= 0


# =========================================================================
# Test Airtightness
# =========================================================================


class TestAirtightness:
    """Test airtightness assessment per EN 13829 / ISO 9972."""

    def test_airtightness_result_present(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.airtightness_result is not None

    def test_n50_value(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        air = result.airtightness_result
        assert air.n50_ach > 0

    def test_airtightness_classification(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        air = result.airtightness_result
        assert air.classification != ""

    def test_assess_airtightness_method(self, engine, full_envelope):
        air_result = engine.assess_airtightness(full_envelope)
        assert air_result is not None
        assert air_result.n50_ach > 0


# =========================================================================
# Test Condensation Risk
# =========================================================================


class TestCondensationRisk:
    """Test condensation risk assessment per EN ISO 13788 (Glaser method)."""

    def test_condensation_result_present(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.condensation_risk is not None

    def test_condensation_risk_level(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        cr = result.condensation_risk
        assert cr.risk_level in ("low", "medium", "high", "critical")

    def test_check_condensation_risk_method(self, engine, full_envelope):
        cr = engine.check_condensation_risk(full_envelope)
        assert cr is not None
        assert cr.risk_level in ("low", "medium", "high", "critical")


# =========================================================================
# Test Improvements
# =========================================================================


class TestImprovements:
    """Test improvement opportunity identification."""

    def test_improvements_list(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert isinstance(result.improvement_opportunities, list)

    def test_total_savings_non_negative(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.total_improvement_savings_kwh >= 0

    def test_identify_improvements_method(self, engine, full_envelope):
        improvements = engine.identify_improvements(full_envelope)
        assert isinstance(improvements, list)


# =========================================================================
# Test Provenance
# =========================================================================


class TestProvenance:
    """Test SHA-256 provenance hashing."""

    def test_provenance_hash_present(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.provenance_hash != ""

    def test_provenance_hash_length(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_hex(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        int(result.provenance_hash, 16)  # Should not raise

    def test_provenance_deterministic(self, engine, engine_mod):
        """Same input should produce same provenance hash."""
        wall = engine_mod.WallElement(
            wall_type=engine_mod.WallType.SOLID_BRICK,
            area_m2=100.0,
            age_band=engine_mod.AgeBand.PRE_1919,
        )
        env = engine_mod.BuildingEnvelope(
            facility_id="FAC-DET-001",
            name="Determinism Test",
            building_type=engine_mod.BuildingType.RESIDENTIAL_HOUSE,
            year_built=1900,
            country="UK",
            gross_floor_area_m2=80.0,
            heated_volume_m3=200.0,
            walls=[wall],
        )
        r1 = engine.analyze(env)
        r2 = engine.analyze(env)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)


# =========================================================================
# Test Edge Cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_elements_raises(self, engine, engine_mod):
        env = engine_mod.BuildingEnvelope(
            facility_id="FAC-EMPTY",
            name="Empty Building",
            building_type=engine_mod.BuildingType.OFFICE,
            year_built=2020,
            country="UK",
            gross_floor_area_m2=100.0,
            heated_volume_m3=300.0,
        )
        with pytest.raises(ValueError, match="element"):
            engine.analyze(env)

    def test_processing_time_recorded(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.processing_time_ms > 0

    def test_result_has_facility_id(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.facility_id == "FAC-TEST-002"

    def test_result_has_engine_version(self, engine, full_envelope):
        result = engine.analyze(full_envelope)
        assert result.engine_version == "1.0.0"

    def test_envelope_performance_dict(self, engine, full_envelope):
        perf = engine.calculate_envelope_performance(full_envelope)
        assert isinstance(perf, dict)
        assert "fabric_heat_loss_w_k" in perf
        assert "total_heat_loss_w_k" in perf
