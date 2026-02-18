# -*- coding: utf-8 -*-
"""
Unit tests for EquipmentComponentEngine (Engine 4 of 7) - AGENT-MRV-005

Tests component registration, listing, update, decommissioning,
component counts, AP-42 Chapter 7 tank loss calculations, pneumatic
inventory, repair history, facility inventory, and engine statistics.

Target: 50 tests, ~600 lines.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Dict
from unittest.mock import MagicMock

import pytest

from greenlang.fugitive_emissions.equipment_component import (
    EquipmentComponentEngine,
    ComponentType,
    ServiceType,
    TankType,
    PneumaticDeviceType,
    ComponentCondition,
    RimSealType,
    ComponentRecord,
    RepairRecord,
    TankParameters,
    PNEUMATIC_EMISSION_RATES,
    CH4_DENSITY_KG_PER_SCF,
    RIM_SEAL_FACTORS,
    DECK_FITTING_FACTORS,
    FIXED_ROOF_CONSTANTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return EquipmentComponentEngine()


@pytest.fixture
def populated_engine(engine):
    """Engine pre-loaded with components across types and services."""
    components = [
        ("V-101", "valve", "gas", "FAC-001"),
        ("V-102", "valve", "gas", "FAC-001"),
        ("V-103", "valve", "light_liquid", "FAC-001"),
        ("P-201", "pump", "light_liquid", "FAC-001"),
        ("P-202", "pump", "heavy_liquid", "FAC-001"),
        ("C-301", "compressor", "gas", "FAC-001"),
        ("F-401", "connector", "gas", "FAC-001"),
        ("F-402", "connector", "gas", "FAC-001"),
        ("F-403", "connector", "gas", "FAC-002"),
    ]
    for tag, ctype, stype, fid in components:
        engine.register_component({
            "tag_number": tag,
            "component_type": ctype,
            "service_type": stype,
            "facility_id": fid,
        })
    return engine


# ===========================================================================
# Component Registration (8 tests)
# ===========================================================================


class TestComponentRegistration:
    """Tests for register_component."""

    def test_register_basic(self, engine):
        result = engine.register_component({
            "tag_number": "V-001",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        })
        assert "component_id" in result
        assert result["tag_number"] == "V-001"
        assert result["component_type"] == "valve"
        assert result["service_type"] == "gas"
        assert result["facility_id"] == "FAC-001"

    def test_register_generates_unique_ids(self, engine):
        ids = set()
        for i in range(10):
            result = engine.register_component({
                "tag_number": f"V-{i:03d}",
                "component_type": "valve",
                "service_type": "gas",
                "facility_id": "FAC-001",
            })
            ids.add(result["component_id"])
        assert len(ids) == 10

    def test_register_missing_tag_raises(self, engine):
        with pytest.raises(ValueError, match="tag_number is required"):
            engine.register_component({
                "component_type": "valve",
                "service_type": "gas",
                "facility_id": "FAC-001",
            })

    def test_register_missing_facility_raises(self, engine):
        with pytest.raises(ValueError, match="facility_id is required"):
            engine.register_component({
                "tag_number": "V-001",
                "component_type": "valve",
                "service_type": "gas",
            })

    def test_register_includes_provenance(self, engine):
        result = engine.register_component({
            "tag_number": "V-PROV",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        })
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_register_id_prefix(self, engine):
        result = engine.register_component({
            "tag_number": "V-PFX",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        })
        assert result["component_id"].startswith("comp_")

    def test_register_increments_counter(self, engine):
        engine.register_component({
            "tag_number": "V-CNT",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        })
        assert engine._total_registrations == 1

    def test_register_all_component_types(self, engine):
        for ct in ComponentType:
            engine.register_component({
                "tag_number": f"TAG-{ct.value}",
                "component_type": ct.value,
                "service_type": "gas",
                "facility_id": "FAC-001",
            })
        assert engine._total_registrations == len(ComponentType)

    def test_register_invalid_component_type_raises(self, engine):
        with pytest.raises(ValueError, match="component_type must be one of"):
            engine.register_component({
                "tag_number": "V-BAD",
                "component_type": "not_a_type",
                "service_type": "gas",
                "facility_id": "FAC-001",
            })

    def test_register_invalid_service_type_raises(self, engine):
        with pytest.raises(ValueError, match="service_type must be one of"):
            engine.register_component({
                "tag_number": "V-BAD2",
                "component_type": "valve",
                "service_type": "not_a_service",
                "facility_id": "FAC-001",
            })


# ===========================================================================
# Component Listing and Retrieval (6 tests)
# ===========================================================================


class TestComponentListing:
    """Tests for list_components, get_component, update_component."""

    def test_list_components_pagination(self, populated_engine):
        result = populated_engine.list_components(
            facility_id="FAC-001", page=1, page_size=3
        )
        assert len(result["components"]) <= 3
        assert result["total"] >= 7

    def test_list_components_filter_type(self, populated_engine):
        result = populated_engine.list_components(
            facility_id="FAC-001", component_type="valve"
        )
        for comp in result["components"]:
            assert comp["component_type"] == "valve"
        assert result["total"] == 3

    def test_list_components_filter_service(self, populated_engine):
        result = populated_engine.list_components(
            facility_id="FAC-001", service_type="gas"
        )
        for comp in result["components"]:
            assert comp["service_type"] == "gas"

    def test_get_component_by_id(self, engine):
        reg = engine.register_component({
            "tag_number": "V-GET",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        })
        comp = engine.get_component(reg["component_id"])
        assert comp is not None
        assert comp["tag_number"] == "V-GET"

    def test_get_nonexistent_component_returns_none(self, engine):
        result = engine.get_component("comp_nonexistent")
        assert result is None

    def test_update_component(self, engine):
        reg = engine.register_component({
            "tag_number": "V-UPD",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        })
        updated = engine.update_component(
            reg["component_id"],
            {"condition": "poor", "location": "Area B"}
        )
        assert updated is not None
        assert updated["condition"] == "poor"
        assert updated["location"] == "Area B"


# ===========================================================================
# Component Counts (4 tests)
# ===========================================================================


class TestComponentCounts:
    """Tests for get_component_counts."""

    def test_counts_by_type(self, populated_engine):
        result = populated_engine.get_component_counts(facility_id="FAC-001")
        assert result["by_type"]["valve"] == 3
        assert result["by_type"]["pump"] == 2
        assert result["by_type"]["compressor"] == 1
        assert result["by_type"]["connector"] == 2

    def test_counts_by_service(self, populated_engine):
        result = populated_engine.get_component_counts(facility_id="FAC-001")
        assert result["by_service"]["gas"] >= 5
        assert result["by_service"]["light_liquid"] >= 2

    def test_counts_other_facility(self, populated_engine):
        result = populated_engine.get_component_counts(facility_id="FAC-002")
        assert result["by_type"]["connector"] == 1
        assert result["total_active"] == 1

    def test_counts_empty_facility(self, engine):
        result = engine.get_component_counts(facility_id="FAC-EMPTY")
        assert result["total_active"] == 0
        assert result["by_type"] == {}


# ===========================================================================
# Decommissioning (4 tests)
# ===========================================================================


class TestDecommissioning:
    """Tests for decommission_component."""

    def test_decommission(self, populated_engine):
        comps = populated_engine.list_components(facility_id="FAC-001")
        comp_id = comps["components"][0]["component_id"]
        result = populated_engine.decommission_component(
            component_id=comp_id, reason="End of service life"
        )
        assert result["condition"] == "decommissioned"
        assert result["is_active"] is False

    def test_decommission_not_in_active_list(self, populated_engine):
        comps = populated_engine.list_components(facility_id="FAC-001")
        comp_id = comps["components"][0]["component_id"]
        populated_engine.decommission_component(component_id=comp_id)
        active = populated_engine.list_components(
            facility_id="FAC-001", active_only=True
        )
        active_ids = [c["component_id"] for c in active["components"]]
        assert comp_id not in active_ids

    def test_decommission_nonexistent_returns_none(self, engine):
        result = engine.decommission_component(component_id="NONEXISTENT")
        assert result is None

    def test_decommission_increments_counter(self, populated_engine):
        comps = populated_engine.list_components(facility_id="FAC-001")
        comp_id = comps["components"][0]["component_id"]
        populated_engine.decommission_component(component_id=comp_id)
        assert populated_engine._total_decommissions >= 1


# ===========================================================================
# Tank Loss Calculations - AP-42 Chapter 7 (6 tests)
# ===========================================================================


class TestTankLossCalculations:
    """Tests for calculate_tank_losses (AP-42 Chapter 7)."""

    def test_fixed_roof_basic(self, engine):
        result = engine.calculate_tank_losses({
            "tank_id": "TK-001",
            "tank_type": "fixed_roof_vertical",
            "diameter_ft": 50.0,
            "height_ft": 40.0,
            "liquid_height_ft": 20.0,
            "vapor_pressure_psia": 1.5,
            "molecular_weight": 68.0,
            "annual_throughput_gal": 500000,
        })
        assert "breathing_loss_lb_yr" in result
        assert "working_loss_lb_yr" in result
        assert "total_loss_lb_yr" in result
        assert result["total_loss_lb_yr"] >= 0
        assert result["methodology"] == "AP-42 Chapter 7"

    def test_floating_roof_basic(self, engine):
        result = engine.calculate_tank_losses({
            "tank_id": "TK-002",
            "tank_type": "external_floating_roof",
            "diameter_ft": 100.0,
            "height_ft": 48.0,
            "liquid_height_ft": 40.0,
            "vapor_pressure_psia": 2.0,
            "molecular_weight": 72.0,
            "rim_seal_type": "mechanical_shoe",
            "fitting_counts": {
                "access_hatch": 1,
                "gauge_hatch": 1,
                "vacuum_breaker": 2,
            },
        })
        assert result["rim_seal_loss_lb_yr"] > 0
        assert result["fitting_loss_lb_yr"] > 0
        assert result["total_loss_lb_yr"] > 0

    def test_pressurized_zero_loss(self, engine):
        result = engine.calculate_tank_losses({
            "tank_id": "TK-003",
            "tank_type": "pressurized",
            "diameter_ft": 20.0,
        })
        assert result["total_loss_lb_yr"] == 0.0

    def test_underground_zero_loss(self, engine):
        result = engine.calculate_tank_losses({
            "tank_id": "TK-004",
            "tank_type": "underground",
        })
        assert result["total_loss_lb_yr"] == 0.0

    def test_tank_loss_has_kg_conversion(self, engine):
        result = engine.calculate_tank_losses({
            "tank_id": "TK-005",
            "tank_type": "fixed_roof_vertical",
        })
        assert "total_loss_kg_yr" in result
        expected_kg = round(result["total_loss_lb_yr"] * 0.453592, 4)
        assert result["total_loss_kg_yr"] == pytest.approx(expected_kg, abs=0.001)

    def test_tank_loss_increments_counter(self, engine):
        engine.calculate_tank_losses({"tank_id": "TK-006"})
        engine.calculate_tank_losses({"tank_id": "TK-007"})
        assert engine._total_tank_calculations == 2

    def test_tank_loss_provenance(self, engine):
        result = engine.calculate_tank_losses({"tank_id": "TK-PROV"})
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_unsupported_tank_type_raises(self, engine):
        with pytest.raises(ValueError, match="Unsupported tank_type"):
            engine.calculate_tank_losses({
                "tank_id": "TK-BAD",
                "tank_type": "open_pit",
            })


# ===========================================================================
# Pneumatic Inventory (3 tests)
# ===========================================================================


class TestPneumaticInventory:
    """Tests for get_pneumatic_inventory via pneumatic_device components."""

    def test_pneumatic_inventory_with_devices(self, engine):
        for i, ptype in enumerate(["high_bleed", "low_bleed", "zero_bleed"]):
            engine.register_component({
                "tag_number": f"PD-{i:03d}",
                "component_type": "pneumatic_device",
                "service_type": "gas",
                "facility_id": "FAC-001",
                "metadata": {"pneumatic_type": ptype},
            })
        result = engine.get_pneumatic_inventory(facility_id="FAC-001")
        assert result["total_devices"] == 3
        assert "high_bleed" in result["counts_by_type"]
        assert result["total_ch4_kg_yr"] > 0
        assert result["methodology"] == "EPA Subpart W Table W-1A defaults"

    def test_pneumatic_inventory_empty_facility(self, engine):
        result = engine.get_pneumatic_inventory(facility_id="FAC-EMPTY")
        assert result["total_devices"] == 0
        assert result["total_ch4_kg_yr"] == 0.0

    def test_pneumatic_emission_calculation(self, engine):
        """Verify: 1 high-bleed device = 37.3 scf/hr * 8760 hr * 0.0192 kg/scf."""
        engine.register_component({
            "tag_number": "PD-CALC",
            "component_type": "pneumatic_device",
            "service_type": "gas",
            "facility_id": "FAC-CALC",
            "metadata": {"pneumatic_type": "high_bleed"},
        })
        result = engine.get_pneumatic_inventory(facility_id="FAC-CALC")
        expected_scf = 37.3 * 8760
        expected_kg = expected_scf * 0.0192
        assert result["total_ch4_kg_yr"] == pytest.approx(expected_kg, rel=1e-3)


# ===========================================================================
# Repair History (4 tests)
# ===========================================================================


class TestRepairHistory:
    """Tests for add_repair and get_repair_history."""

    def test_add_repair(self, populated_engine):
        comps = populated_engine.list_components(facility_id="FAC-001")
        comp_id = comps["components"][0]["component_id"]
        result = populated_engine.add_repair({
            "component_id": comp_id,
            "repair_date": "2026-03-15",
            "leak_rate_before_ppmv": 15000,
            "leak_rate_after_ppmv": 50,
            "repair_method": "tightening",
            "cost_usd": 250.0,
        })
        assert "repair_id" in result
        assert result["repair_id"].startswith("repair_")
        assert result["provenance_hash"] is not None

    def test_repair_history_multiple(self, populated_engine):
        comps = populated_engine.list_components(facility_id="FAC-001")
        comp_id = comps["components"][0]["component_id"]
        populated_engine.add_repair({
            "component_id": comp_id,
            "repair_date": "2026-03-15",
        })
        populated_engine.add_repair({
            "component_id": comp_id,
            "repair_date": "2026-06-15",
        })
        history = populated_engine.get_repair_history(component_id=comp_id)
        assert history["total_repairs"] >= 2

    def test_repair_increments_counter(self, populated_engine):
        comps = populated_engine.list_components(facility_id="FAC-001")
        comp_id = comps["components"][0]["component_id"]
        populated_engine.add_repair({
            "component_id": comp_id,
            "repair_date": "2026-01-01",
        })
        assert populated_engine._total_repairs >= 1

    def test_repair_nonexistent_component_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.add_repair({
                "component_id": "comp_nonexistent",
                "repair_date": "2026-01-01",
            })


# ===========================================================================
# Facility Inventory (3 tests)
# ===========================================================================


class TestFacilityInventory:
    """Tests for get_facility_inventory."""

    def test_inventory_basic(self, populated_engine):
        result = populated_engine.get_facility_inventory(facility_id="FAC-001")
        assert result["facility_id"] == "FAC-001"
        assert result["total_active_components"] >= 7

    def test_inventory_has_breakdown(self, populated_engine):
        result = populated_engine.get_facility_inventory(facility_id="FAC-001")
        counts = result["component_counts"]
        assert "by_type" in counts
        assert "valve" in counts["by_type"]

    def test_inventory_empty_facility(self, engine):
        result = engine.get_facility_inventory(facility_id="EMPTY")
        assert result["total_active_components"] == 0

    def test_inventory_has_provenance(self, populated_engine):
        result = populated_engine.get_facility_inventory(facility_id="FAC-001")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Engine Statistics (2 tests)
# ===========================================================================


class TestEngineStatistics:
    """Tests for get_statistics."""

    def test_statistics_keys(self, engine):
        stats = engine.get_statistics()
        assert "total_components" in stats
        assert "active_components" in stats
        assert "total_registrations" in stats
        assert "total_decommissions" in stats
        assert "total_repairs" in stats
        assert "total_tank_calculations" in stats

    def test_statistics_after_operations(self, populated_engine):
        stats = populated_engine.get_statistics()
        assert stats["total_registrations"] == 9
        assert stats["total_components"] == 9
        assert stats["active_components"] == 9


# ===========================================================================
# Enumerations and Constants (6 tests)
# ===========================================================================


class TestEnumsAndConstants:
    """Verify enums and reference data."""

    def test_component_types(self):
        assert ComponentType.VALVE.value == "valve"
        assert ComponentType.PUMP.value == "pump"
        assert ComponentType.COMPRESSOR.value == "compressor"
        assert ComponentType.CONNECTOR.value == "connector"
        assert ComponentType.TANK.value == "tank"
        assert ComponentType.PNEUMATIC_DEVICE.value == "pneumatic_device"

    def test_service_types(self):
        assert ServiceType.GAS.value == "gas"
        assert ServiceType.LIGHT_LIQUID.value == "light_liquid"
        assert ServiceType.HEAVY_LIQUID.value == "heavy_liquid"
        assert ServiceType.HYDROGEN.value == "hydrogen"

    def test_tank_types(self):
        assert TankType.FIXED_ROOF_VERTICAL.value == "fixed_roof_vertical"
        assert TankType.EXTERNAL_FLOATING_ROOF.value == "external_floating_roof"
        assert TankType.PRESSURIZED.value == "pressurized"
        assert TankType.UNDERGROUND.value == "underground"

    def test_pneumatic_rates(self):
        assert PNEUMATIC_EMISSION_RATES["high_bleed"] == 37.3
        assert PNEUMATIC_EMISSION_RATES["low_bleed"] == 1.39
        assert PNEUMATIC_EMISSION_RATES["intermittent"] == 13.5
        assert PNEUMATIC_EMISSION_RATES["zero_bleed"] == 0.0

    def test_rim_seal_factors(self):
        assert "mechanical_shoe" in RIM_SEAL_FACTORS
        assert RIM_SEAL_FACTORS["mechanical_shoe"]["KRa"] == 5.8
        assert RIM_SEAL_FACTORS["mechanical_shoe"]["KRb"] == 0.3

    def test_ch4_density(self):
        assert CH4_DENSITY_KG_PER_SCF == pytest.approx(0.0192, rel=1e-3)

    def test_deck_fitting_factors(self):
        assert "access_hatch" in DECK_FITTING_FACTORS
        assert DECK_FITTING_FACTORS["access_hatch"]["KFa"] == 36.0
        assert DECK_FITTING_FACTORS["vacuum_breaker"]["KFa"] == 2.4
