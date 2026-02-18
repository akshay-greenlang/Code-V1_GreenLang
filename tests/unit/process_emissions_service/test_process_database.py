# -*- coding: utf-8 -*-
"""
Unit tests for ProcessDatabaseEngine - AGENT-MRV-004 Process Emissions Agent

Tests the comprehensive in-memory database of 25 industrial process types,
emission factors across 4 sources (IPCC, EPA, DEFRA, EU ETS), raw material
properties, carbonate stoichiometric factors, and GWP values for 8 greenhouse
gas species across 4 IPCC Assessment Report sources (AR4, AR5, AR6, AR6_20YR).

95 tests across 8 test classes.

Author: GreenLang QA Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.process_emissions.process_database import ProcessDatabaseEngine


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def db() -> ProcessDatabaseEngine:
    """Create a ProcessDatabaseEngine instance with provenance disabled."""
    return ProcessDatabaseEngine(config={"enable_provenance": False})


# =========================================================================
# Full set of the 25 process types for parametrized tests
# =========================================================================

ALL_PROCESS_TYPES = [
    "CEMENT", "LIME", "GLASS", "CERAMICS", "SODA_ASH",
    "AMMONIA", "NITRIC_ACID", "ADIPIC_ACID", "CARBIDE", "HYDROGEN",
    "PETROCHEMICAL_ETHYLENE",
    "IRON_STEEL_BF_BOF", "IRON_STEEL_EAF", "IRON_STEEL_DRI",
    "ALUMINUM_PREBAKE", "ALUMINUM_SODERBERG", "FERROALLOY_FESI",
    "MAGNESIUM",
    "SEMICONDUCTOR",
    "PULP_PAPER",
    "TITANIUM_DIOXIDE", "PHOSPHORIC_ACID", "ZINC", "LEAD", "COPPER",
]

MINERAL_TYPES = ["CEMENT", "LIME", "GLASS", "CERAMICS", "SODA_ASH"]
CHEMICAL_TYPES = ["AMMONIA", "NITRIC_ACID", "ADIPIC_ACID", "CARBIDE",
                  "HYDROGEN", "PETROCHEMICAL_ETHYLENE"]
METAL_TYPES = ["IRON_STEEL_BF_BOF", "IRON_STEEL_EAF", "IRON_STEEL_DRI",
               "ALUMINUM_PREBAKE", "ALUMINUM_SODERBERG", "FERROALLOY_FESI",
               "MAGNESIUM"]


# =========================================================================
# TestProcessInfo (25 tests) - one per process type
# =========================================================================

class TestProcessInfo:
    """Tests get_process_info() for all 25 industrial process types."""

    @pytest.mark.parametrize("process_type", ALL_PROCESS_TYPES)
    def test_get_process_info_returns_dict(
        self, db: ProcessDatabaseEngine, process_type: str
    ):
        """get_process_info() returns a non-empty dict for each process type."""
        info = db.get_process_info(process_type)
        assert isinstance(info, dict)
        assert len(info) > 0

    @pytest.mark.parametrize("process_type", ALL_PROCESS_TYPES)
    def test_process_info_has_required_keys(
        self, db: ProcessDatabaseEngine, process_type: str
    ):
        """Each process info dict has all required keys."""
        info = db.get_process_info(process_type)
        required_keys = [
            "category", "display_name", "description",
            "primary_gases", "applicable_tiers", "applicable_methods",
            "epa_subpart", "ipcc_code", "default_production_unit",
        ]
        for key in required_keys:
            assert key in info, f"Missing key '{key}' for {process_type}"

    @pytest.mark.parametrize("process_type", ALL_PROCESS_TYPES)
    def test_process_info_category_valid(
        self, db: ProcessDatabaseEngine, process_type: str
    ):
        """Category is one of the 6 valid categories."""
        info = db.get_process_info(process_type)
        valid_categories = {"MINERAL", "CHEMICAL", "METAL", "ELECTRONICS",
                            "PULP_PAPER", "OTHER"}
        assert info["category"] in valid_categories

    @pytest.mark.parametrize("process_type", ALL_PROCESS_TYPES)
    def test_process_info_primary_gases_non_empty(
        self, db: ProcessDatabaseEngine, process_type: str
    ):
        """Each process has at least one primary gas."""
        info = db.get_process_info(process_type)
        assert len(info["primary_gases"]) > 0

    @pytest.mark.parametrize("process_type", ALL_PROCESS_TYPES)
    def test_process_info_applicable_methods_non_empty(
        self, db: ProcessDatabaseEngine, process_type: str
    ):
        """Each process has at least one applicable calculation method."""
        info = db.get_process_info(process_type)
        assert len(info["applicable_methods"]) > 0

    def test_cement_category_is_mineral(self, db: ProcessDatabaseEngine):
        """Cement is classified under MINERAL category."""
        info = db.get_process_info("CEMENT")
        assert info["category"] == "MINERAL"
        assert info["ipcc_code"] == "2A1"

    def test_nitric_acid_primary_gas_is_n2o(self, db: ProcessDatabaseEngine):
        """Nitric acid primary gas is N2O."""
        info = db.get_process_info("NITRIC_ACID")
        assert "N2O" in info["primary_gases"]

    def test_aluminum_prebake_has_pfc_gases(self, db: ProcessDatabaseEngine):
        """Aluminum prebake lists CO2, CF4, and C2F6 as primary gases."""
        info = db.get_process_info("ALUMINUM_PREBAKE")
        assert "CO2" in info["primary_gases"]
        assert "CF4" in info["primary_gases"]
        assert "C2F6" in info["primary_gases"]

    def test_semiconductor_has_multiple_gases(self, db: ProcessDatabaseEngine):
        """Semiconductor lists multiple fluorinated GHGs."""
        info = db.get_process_info("SEMICONDUCTOR")
        assert len(info["primary_gases"]) >= 4

    def test_unknown_process_raises_key_error(self, db: ProcessDatabaseEngine):
        """Unknown process type raises KeyError."""
        with pytest.raises(KeyError, match="Unknown process type"):
            db.get_process_info("UNKNOWN_PROCESS")

    def test_case_insensitive_lookup(self, db: ProcessDatabaseEngine):
        """Process type lookup is case-insensitive."""
        info_upper = db.get_process_info("CEMENT")
        info_lower = db.get_process_info("cement")
        assert info_upper["display_name"] == info_lower["display_name"]


# =========================================================================
# TestEmissionFactors (20 tests)
# =========================================================================

class TestEmissionFactors:
    """Tests get_emission_factor() across process types, gases, and sources."""

    @pytest.mark.parametrize("process_type,gas,expected", [
        ("CEMENT", "CO2", Decimal("0.507")),
        ("LIME", "CO2", Decimal("0.785")),
        ("GLASS", "CO2", Decimal("0.208")),
        ("AMMONIA", "CO2", Decimal("1.500")),
        ("NITRIC_ACID", "N2O", Decimal("0.007")),
        ("ADIPIC_ACID", "N2O", Decimal("0.300")),
        ("IRON_STEEL_BF_BOF", "CO2", Decimal("1.900")),
        ("IRON_STEEL_EAF", "CO2", Decimal("0.400")),
        ("ALUMINUM_PREBAKE", "CO2", Decimal("1.500")),
        ("ALUMINUM_PREBAKE", "CF4", Decimal("0.040")),
    ])
    def test_ipcc_emission_factors(
        self, db: ProcessDatabaseEngine,
        process_type: str, gas: str, expected: Decimal
    ):
        """IPCC emission factors match authoritative values."""
        ef = db.get_emission_factor(process_type, gas, source="IPCC")
        assert ef == expected

    @pytest.mark.parametrize("process_type,gas,expected", [
        ("CEMENT", "CO2", Decimal("0.510")),
        ("IRON_STEEL_BF_BOF", "CO2", Decimal("1.850")),
        ("HYDROGEN", "CO2", Decimal("9.260")),
        ("ZINC", "CO2", Decimal("3.690")),
    ])
    def test_epa_emission_factors(
        self, db: ProcessDatabaseEngine,
        process_type: str, gas: str, expected: Decimal
    ):
        """EPA emission factors match values from 40 CFR Part 98."""
        ef = db.get_emission_factor(process_type, gas, source="EPA")
        assert ef == expected

    @pytest.mark.parametrize("process_type,gas,expected", [
        ("CEMENT", "CO2", Decimal("0.519")),
        ("AMMONIA", "CO2", Decimal("1.530")),
        ("SODA_ASH", "CO2", Decimal("0.415")),
    ])
    def test_defra_emission_factors(
        self, db: ProcessDatabaseEngine,
        process_type: str, gas: str, expected: Decimal
    ):
        """DEFRA emission factors match UK 2025 values."""
        ef = db.get_emission_factor(process_type, gas, source="DEFRA")
        assert ef == expected

    @pytest.mark.parametrize("process_type,gas,expected", [
        ("CEMENT", "CO2", Decimal("0.525")),
        ("IRON_STEEL_BF_BOF", "CO2", Decimal("1.900")),
    ])
    def test_eu_ets_emission_factors(
        self, db: ProcessDatabaseEngine,
        process_type: str, gas: str, expected: Decimal
    ):
        """EU ETS MRR emission factors match regulatory values."""
        ef = db.get_emission_factor(process_type, gas, source="EU_ETS")
        assert ef == expected

    def test_unknown_process_raises_key_error(self, db: ProcessDatabaseEngine):
        """Unknown process type raises KeyError."""
        with pytest.raises(KeyError):
            db.get_emission_factor("NONEXISTENT", "CO2", source="IPCC")

    def test_unknown_gas_raises_key_error(self, db: ProcessDatabaseEngine):
        """Unknown gas for a process raises KeyError."""
        with pytest.raises(KeyError):
            db.get_emission_factor("CEMENT", "SF6", source="IPCC")

    def test_emission_factor_is_decimal_type(self, db: ProcessDatabaseEngine):
        """Returned emission factor is always a Decimal."""
        ef = db.get_emission_factor("CEMENT", "CO2", source="IPCC")
        assert isinstance(ef, Decimal)


# =========================================================================
# TestRawMaterials (10 tests)
# =========================================================================

class TestRawMaterials:
    """Tests get_raw_material() for key raw material properties."""

    @pytest.mark.parametrize("material,expected_cc", [
        ("CALCIUM_CARBONATE", Decimal("0.1200")),
        ("MAGNESIUM_CARBONATE", Decimal("0.1423")),
        ("DOLOMITE", Decimal("0.1303")),
        ("COKE", Decimal("0.8700")),
        ("COAL", Decimal("0.7500")),
        ("IRON_ORE", Decimal("0.0000")),
        ("ANODE_CARBON", Decimal("0.8500")),
        ("NATURAL_GAS_FEEDSTOCK", Decimal("0.7300")),
        ("SCRAP_STEEL", Decimal("0.0050")),
        ("ALUMINA", Decimal("0.0000")),
    ])
    def test_raw_material_carbon_content(
        self, db: ProcessDatabaseEngine,
        material: str, expected_cc: Decimal
    ):
        """Raw material carbon contents match authoritative IPCC values."""
        mat_info = db.get_raw_material(material)
        assert mat_info["carbon_content"] == expected_cc

    def test_raw_material_has_required_keys(self, db: ProcessDatabaseEngine):
        """Raw material dict contains all required metadata keys."""
        mat_info = db.get_raw_material("CALCIUM_CARBONATE")
        required_keys = ["display_name", "formula", "molecular_weight",
                         "carbon_content", "category"]
        for key in required_keys:
            assert key in mat_info, f"Missing key '{key}'"

    def test_unknown_material_raises_key_error(self, db: ProcessDatabaseEngine):
        """Unknown material type raises KeyError."""
        with pytest.raises(KeyError):
            db.get_raw_material("UNOBTANIUM")


# =========================================================================
# TestCarbonateFactor (10 tests)
# =========================================================================

class TestCarbonateFactor:
    """Tests get_carbonate_factor() for stoichiometric CO2 factors."""

    @pytest.mark.parametrize("carbonate_type,expected_co2_factor", [
        ("CALCITE", Decimal("0.440")),
        ("MAGNESITE", Decimal("0.522")),
        ("DOLOMITE", Decimal("0.477")),
        ("SIDERITE", Decimal("0.380")),
        ("ANKERITE", Decimal("0.427")),
        ("SODA_ASH", Decimal("0.415")),
        ("WITHERITE", Decimal("0.223")),
        ("STRONTIANITE", Decimal("0.298")),
        ("RHODOCHROSITE", Decimal("0.383")),
        ("LITHIUM_CARBONATE", Decimal("0.596")),
    ])
    def test_carbonate_co2_factor(
        self, db: ProcessDatabaseEngine,
        carbonate_type: str, expected_co2_factor: Decimal
    ):
        """Carbonate CO2 factor matches stoichiometric calculation (rounded)."""
        factor_info = db.get_carbonate_factor(carbonate_type)
        assert factor_info["co2_factor_rounded"] == expected_co2_factor

    def test_carbonate_factor_has_molecular_weights(
        self, db: ProcessDatabaseEngine
    ):
        """Carbonate factor dict contains molecular weight data."""
        factor = db.get_carbonate_factor("CALCITE")
        assert "molecular_weight_carbonate" in factor
        assert "molecular_weight_oxide" in factor

    def test_unknown_carbonate_raises_key_error(self, db: ProcessDatabaseEngine):
        """Unknown carbonate type raises KeyError."""
        with pytest.raises(KeyError):
            db.get_carbonate_factor("FANTASY_CARBONATE")


# =========================================================================
# TestGWPValues (10 tests)
# =========================================================================

class TestGWPValues:
    """Tests get_gwp() for all gases across IPCC AR sources."""

    @pytest.mark.parametrize("gas,source,expected_gwp", [
        ("CO2", "AR6", Decimal("1")),
        ("CH4", "AR6", Decimal("29.8")),
        ("N2O", "AR6", Decimal("273")),
        ("CF4", "AR6", Decimal("7380")),
        ("C2F6", "AR6", Decimal("12400")),
        ("SF6", "AR6", Decimal("25200")),
        ("NF3", "AR6", Decimal("17400")),
        ("HFC_23", "AR6", Decimal("14600")),
    ])
    def test_ar6_gwp_values(
        self, db: ProcessDatabaseEngine,
        gas: str, source: str, expected_gwp: Decimal
    ):
        """AR6 GWP-100yr values match IPCC Sixth Assessment Report."""
        gwp = db.get_gwp(gas, source=source)
        assert gwp == expected_gwp

    @pytest.mark.parametrize("gas,expected_gwp", [
        ("CO2", Decimal("1")),
        ("CH4", Decimal("28")),
        ("N2O", Decimal("265")),
        ("SF6", Decimal("23500")),
    ])
    def test_ar5_gwp_values(
        self, db: ProcessDatabaseEngine,
        gas: str, expected_gwp: Decimal
    ):
        """AR5 GWP values match IPCC Fifth Assessment Report."""
        gwp = db.get_gwp(gas, source="AR5")
        assert gwp == expected_gwp

    @pytest.mark.parametrize("gas,expected_gwp", [
        ("CO2", Decimal("1")),
        ("CH4", Decimal("25")),
        ("N2O", Decimal("298")),
    ])
    def test_ar4_gwp_values(
        self, db: ProcessDatabaseEngine,
        gas: str, expected_gwp: Decimal
    ):
        """AR4 GWP values match IPCC Fourth Assessment Report."""
        gwp = db.get_gwp(gas, source="AR4")
        assert gwp == expected_gwp

    def test_gwp_is_decimal_type(self, db: ProcessDatabaseEngine):
        """GWP value is always a Decimal."""
        gwp = db.get_gwp("CO2", source="AR6")
        assert isinstance(gwp, Decimal)

    def test_unknown_gas_raises_key_error(self, db: ProcessDatabaseEngine):
        """Unknown gas raises KeyError."""
        with pytest.raises(KeyError):
            db.get_gwp("UNICORN_GAS", source="AR6")

    def test_unknown_source_raises_key_error(self, db: ProcessDatabaseEngine):
        """Unknown GWP source raises KeyError."""
        with pytest.raises(KeyError):
            db.get_gwp("CO2", source="AR99")


# =========================================================================
# TestProductionRoutes (5 tests)
# =========================================================================

class TestProductionRoutes:
    """Tests get_production_routes() for processes with multiple routes."""

    def test_iron_steel_bf_bof_routes(self, db: ProcessDatabaseEngine):
        """Iron/steel BF-BOF has multiple production routes."""
        routes = db.get_production_routes("IRON_STEEL_BF_BOF")
        assert len(routes) >= 2
        route_ids = [r["route_id"] for r in routes]
        assert "BF_BOF_INTEGRATED" in route_ids

    def test_iron_steel_dri_routes(self, db: ProcessDatabaseEngine):
        """DRI has natural gas, hydrogen, and coal routes."""
        routes = db.get_production_routes("IRON_STEEL_DRI")
        assert len(routes) >= 3
        route_ids = [r["route_id"] for r in routes]
        assert "DRI_NATURAL_GAS" in route_ids
        assert "DRI_HYDROGEN" in route_ids

    def test_aluminum_prebake_routes(self, db: ProcessDatabaseEngine):
        """Aluminum prebake has CWPB and SWPB routes."""
        routes = db.get_production_routes("ALUMINUM_PREBAKE")
        assert len(routes) >= 2
        route_ids = [r["route_id"] for r in routes]
        assert "PREBAKE_CWPB" in route_ids

    def test_cement_routes(self, db: ProcessDatabaseEngine):
        """Cement has Portland, blended, and white cement routes."""
        routes = db.get_production_routes("CEMENT")
        assert len(routes) >= 3

    def test_hydrogen_routes(self, db: ProcessDatabaseEngine):
        """Hydrogen has grey, blue, ATR, and coal gasification routes."""
        routes = db.get_production_routes("HYDROGEN")
        assert len(routes) >= 4
        route_ids = [r["route_id"] for r in routes]
        assert "SMR_GREY" in route_ids
        assert "SMR_BLUE" in route_ids


# =========================================================================
# TestListMethods (5 tests)
# =========================================================================

class TestListMethods:
    """Tests list_processes() and list_materials()."""

    def test_list_all_processes(self, db: ProcessDatabaseEngine):
        """list_processes() returns all 25 process types."""
        processes = db.list_processes()
        assert len(processes) == 25

    def test_list_processes_by_mineral_category(self, db: ProcessDatabaseEngine):
        """list_processes(category='MINERAL') returns 5 processes."""
        processes = db.list_processes(category="MINERAL")
        assert len(processes) == 5
        assert all(p["category"] == "MINERAL" for p in processes)

    def test_list_processes_by_metal_category(self, db: ProcessDatabaseEngine):
        """list_processes(category='METAL') returns 7 processes."""
        processes = db.list_processes(category="METAL")
        assert len(processes) == 7

    def test_list_materials_returns_non_empty(self, db: ProcessDatabaseEngine):
        """list_materials() returns all registered raw materials."""
        materials = db.list_materials()
        assert len(materials) > 0

    def test_list_processes_includes_process_type_key(
        self, db: ProcessDatabaseEngine
    ):
        """Each listed process includes a 'process_type' key."""
        processes = db.list_processes()
        for p in processes:
            assert "process_type" in p


# =========================================================================
# TestEdgeCases (10 tests)
# =========================================================================

class TestEdgeCases:
    """Tests for unknown types, missing factors, custom factors, defaults."""

    def test_register_custom_factor(self, db: ProcessDatabaseEngine):
        """Custom emission factors can be registered and retrieved."""
        db.register_custom_factor(
            process_type="CEMENT",
            gas="CO2",
            value=Decimal("0.520"),
            source="CUSTOM",
            unit="tCO2/t_clinker",
        )
        ef = db.get_emission_factor("CEMENT", "CO2", source="CUSTOM")
        assert ef == Decimal("0.520")

    def test_custom_factor_overrides_builtin(self, db: ProcessDatabaseEngine):
        """Custom factor takes precedence over built-in for CUSTOM source."""
        db.register_custom_factor(
            process_type="LIME",
            gas="CO2",
            value=Decimal("0.800"),
            source="CUSTOM",
            unit="tCO2/t_CaO",
        )
        ef = db.get_emission_factor("LIME", "CO2", source="CUSTOM")
        assert ef == Decimal("0.800")

    def test_builtin_factor_unaffected_by_custom(self, db: ProcessDatabaseEngine):
        """Registering a custom factor does not alter built-in sources."""
        db.register_custom_factor("CEMENT", "CO2", Decimal("0.999"), "CUSTOM", "t")
        ef_ipcc = db.get_emission_factor("CEMENT", "CO2", source="IPCC")
        assert ef_ipcc == Decimal("0.507")

    def test_get_factor_count(self, db: ProcessDatabaseEngine):
        """get_factor_count() returns a positive integer."""
        count = db.get_factor_count()
        assert isinstance(count, int)
        assert count > 0

    def test_get_emission_factor_with_unit(self, db: ProcessDatabaseEngine):
        """get_emission_factor_with_unit() returns (Decimal, str) tuple."""
        value, unit = db.get_emission_factor_with_unit("CEMENT", "CO2", "IPCC")
        assert isinstance(value, Decimal)
        assert isinstance(unit, str)
        assert value == Decimal("0.507")
        assert "tCO2" in unit

    def test_get_semiconductor_gas_params(self, db: ProcessDatabaseEngine):
        """Semiconductor gas parameters are retrievable for known gases."""
        params = db.get_semiconductor_gas_params("CF4")
        assert "default_utilization_rate" in params
        assert params["default_utilization_rate"] == Decimal("0.50")

    def test_default_source_is_ipcc(self, db: ProcessDatabaseEngine):
        """Default emission factor source is IPCC."""
        ef = db.get_emission_factor("CEMENT", "CO2")
        ef_ipcc = db.get_emission_factor("CEMENT", "CO2", source="IPCC")
        assert ef == ef_ipcc

    def test_factors_are_positive(self, db: ProcessDatabaseEngine):
        """All default emission factors are positive."""
        for pt in ALL_PROCESS_TYPES:
            info = db.get_process_info(pt)
            for gas in info["primary_gases"]:
                try:
                    ef = db.get_emission_factor(pt, gas, source="IPCC")
                    assert ef > 0, f"{pt}/{gas} EF should be positive, got {ef}"
                except KeyError:
                    pass  # Some gas/source combos may not exist

    def test_gwp_co2_always_one(self, db: ProcessDatabaseEngine):
        """CO2 GWP is always 1 regardless of AR source."""
        for source in ["AR4", "AR5", "AR6", "AR6_20YR"]:
            gwp = db.get_gwp("CO2", source=source)
            assert gwp == Decimal("1")

    def test_case_insensitive_gas_lookup(self, db: ProcessDatabaseEngine):
        """Gas lookup is case-insensitive."""
        ef_upper = db.get_emission_factor("CEMENT", "CO2", source="IPCC")
        ef_lower = db.get_emission_factor("CEMENT", "co2", source="IPCC")
        assert ef_upper == ef_lower
