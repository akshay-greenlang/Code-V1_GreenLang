# -*- coding: utf-8 -*-
"""
Unit tests for FugitiveSourceDatabaseEngine (Engine 1 of 7) - AGENT-MRV-005

Tests get_source_info (25 types), get_component_ef (20), get_screening_factor (10),
get_coal_methane_factor (10), get_wastewater_factor (10), get_pneumatic_rate (5),
get_gas_composition (5), list methods (5), custom factors, and edge cases.

Test Classes:
    - TestSourceInfo                  (25 tests - each source type)
    - TestComponentEFs                (20 tests)
    - TestScreeningFactors            (10 tests)
    - TestCoalFactors                 (10 tests)
    - TestWastewaterFactors           (10 tests)
    - TestPneumaticRates              (5 tests)
    - TestGasComposition              (5 tests)
    - TestListMethods                 (5 tests)
    - TestEdgeCases                   (5 tests)

Total: 95 tests, ~820 lines.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.fugitive_emissions.fugitive_source_database import (
    FugitiveSourceDatabaseEngine,
    SOURCE_TYPES,
    COMPONENT_EMISSION_FACTORS,
    SCREENING_RANGE_FACTORS,
    CORRELATION_COEFFICIENTS,
    COAL_METHANE_FACTORS,
    WASTEWATER_MCF,
    WASTEWATER_BO_KG_CH4_PER_KG_BOD,
    WASTEWATER_N2O_EF_KG_PER_KG_N,
    N2O_N_RATIO,
    PNEUMATIC_RATES,
    DEFAULT_GAS_COMPOSITION,
    DEFAULT_WEIGHT_FRACTIONS,
    GWP_VALUES,
    CH4_DENSITY_KG_PER_M3,
    POST_MINING_FRACTION,
    SourceCategory,
    CalculationMethod,
    GWPSource,
    CoalRank,
    WastewaterTreatmentType,
    CustomEmissionFactor,
    _compute_weight_fractions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> FugitiveSourceDatabaseEngine:
    """Create a default FugitiveSourceDatabaseEngine."""
    return FugitiveSourceDatabaseEngine()


@pytest.fixture
def db_ar5() -> FugitiveSourceDatabaseEngine:
    """Create engine with AR5 GWP default."""
    return FugitiveSourceDatabaseEngine(config={"default_gwp_source": "AR5"})


# ===========================================================================
# TestSourceInfo - 25 tests (one per source type)
# ===========================================================================


class TestSourceInfo:
    """Test get_source_info for all 25 source types."""

    ALL_SOURCE_TYPES = list(SOURCE_TYPES.keys())

    @pytest.mark.parametrize("source_type", ALL_SOURCE_TYPES)
    def test_source_type_found(self, db, source_type):
        result = db.get_source_info(source_type)
        assert result is not None
        assert result["source_type"] == source_type
        assert "name" in result
        assert "category" in result
        assert "primary_gas" in result
        assert "applicable_methods" in result
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_source_not_found_returns_none(self, db):
        result = db.get_source_info("NONEXISTENT_TYPE")
        assert result is None

    def test_equipment_leak_valve_gas_category(self, db):
        result = db.get_source_info("EQUIPMENT_LEAK_VALVE_GAS")
        assert result["category"] == "EQUIPMENT_LEAK"
        assert result["primary_gas"] == "CH4"
        assert "AVERAGE_EMISSION_FACTOR" in result["applicable_methods"]

    def test_coal_mine_underground_category(self, db):
        result = db.get_source_info("COAL_MINE_UNDERGROUND")
        assert result["category"] == "COAL_MINE"
        assert "ENGINEERING_ESTIMATE" in result["applicable_methods"]

    def test_wastewater_industrial_category(self, db):
        result = db.get_source_info("WASTEWATER_INDUSTRIAL")
        assert result["category"] == "WASTEWATER"
        assert "ENGINEERING_ESTIMATE" in result["applicable_methods"]

    def test_pneumatic_high_bleed_category(self, db):
        result = db.get_source_info("PNEUMATIC_HIGH_BLEED")
        assert result["category"] == "PNEUMATIC_DEVICE"

    def test_tank_fixed_roof_category(self, db):
        result = db.get_source_info("TANK_FIXED_ROOF")
        assert result["category"] == "TANK_STORAGE"
        assert result["primary_gas"] == "VOC"

    def test_direct_measurement_hiflow_category(self, db):
        result = db.get_source_info("DIRECT_MEASUREMENT_HIFLOW")
        assert result["category"] == "DIRECT_MEASUREMENT"
        assert "DIRECT_MEASUREMENT" in result["applicable_methods"]

    def test_source_info_provenance_hash_deterministic(self, db):
        r1 = db.get_source_info("EQUIPMENT_LEAK_VALVE_GAS")
        r2 = db.get_source_info("EQUIPMENT_LEAK_VALVE_GAS")
        assert r1["provenance_hash"] == r2["provenance_hash"]


# ===========================================================================
# TestComponentEFs - 20 tests
# ===========================================================================


class TestComponentEFs:
    """Test get_component_ef for all EPA component/service pairs."""

    @pytest.mark.parametrize("comp,svc,expected_ef", [
        ("valve", "gas", Decimal("0.00597")),
        ("valve", "light_liquid", Decimal("0.00403")),
        ("valve", "heavy_liquid", Decimal("0.00023")),
        ("pump", "light_liquid", Decimal("0.01140")),
        ("pump", "heavy_liquid", Decimal("0.00862")),
        ("compressor", "gas", Decimal("0.22800")),
        ("pressure_relief", "gas", Decimal("0.10400")),
        ("connector", "gas", Decimal("0.00183")),
        ("connector", "light_liquid", Decimal("0.00183")),
        ("connector", "heavy_liquid", Decimal("0.00183")),
        ("open_ended_line", "gas", Decimal("0.00170")),
        ("open_ended_line", "light_liquid", Decimal("0.00170")),
        ("sampling", "gas", Decimal("0.01500")),
        ("sampling", "light_liquid", Decimal("0.01500")),
        ("flange", "gas", Decimal("0.00083")),
        ("flange", "light_liquid", Decimal("0.00083")),
        ("flange", "heavy_liquid", Decimal("0.00083")),
    ])
    def test_builtin_ef_value(self, db, comp, svc, expected_ef):
        result = db.get_component_ef(comp, svc)
        assert result is not None
        assert result["ef_decimal"] == expected_ef
        assert result["is_custom"] is False
        assert result["source"] == "EPA-453/R-95-017"
        assert "provenance_hash" in result

    def test_unknown_component_returns_none(self, db):
        result = db.get_component_ef("nonexistent_component", "gas")
        assert result is None

    def test_unknown_service_returns_none(self, db):
        result = db.get_component_ef("valve", "nonexistent_service")
        assert result is None

    def test_component_ef_provenance_hash_is_sha256(self, db):
        result = db.get_component_ef("valve", "gas")
        assert len(result["provenance_hash"]) == 64
        int(result["provenance_hash"], 16)


# ===========================================================================
# TestScreeningFactors - 10 tests
# ===========================================================================


class TestScreeningFactors:
    """Test get_screening_factor for EPA leak/no-leak pairs."""

    @pytest.mark.parametrize("comp,svc", [
        ("valve", "gas"),
        ("valve", "light_liquid"),
        ("valve", "heavy_liquid"),
        ("pump", "light_liquid"),
        ("pump", "heavy_liquid"),
        ("compressor", "gas"),
        ("pressure_relief", "gas"),
        ("connector", "gas"),
    ])
    def test_screening_factor_found(self, db, comp, svc):
        result = db.get_screening_factor(comp, svc)
        assert result is not None
        assert "leak_ef_decimal" in result
        assert "no_leak_ef_decimal" in result
        assert result["leak_ef_decimal"] > result["no_leak_ef_decimal"]
        assert result["threshold_ppmv"] == 10000

    def test_valve_gas_screening_values(self, db):
        result = db.get_screening_factor("valve", "gas")
        assert result["leak_ef_decimal"] == Decimal("0.02680")
        assert result["no_leak_ef_decimal"] == Decimal("0.00006")

    def test_screening_factor_not_found(self, db):
        result = db.get_screening_factor("nonexistent", "gas")
        assert result is None


# ===========================================================================
# TestCoalFactors - 10 tests
# ===========================================================================


class TestCoalFactors:
    """Test get_coal_methane_factor for all 4 coal ranks."""

    @pytest.mark.parametrize("rank,expected_m3", [
        ("ANTHRACITE", Decimal("18")),
        ("BITUMINOUS", Decimal("10")),
        ("SUBBITUMINOUS", Decimal("3")),
        ("LIGNITE", Decimal("1")),
    ])
    def test_coal_rank_factor(self, db, rank, expected_m3):
        result = db.get_coal_methane_factor(rank)
        assert result is not None
        assert result["ef_m3_per_tonne_decimal"] == expected_m3
        assert result["coal_rank"] == rank
        assert "provenance_hash" in result

    def test_coal_kg_per_tonne_calculated(self, db):
        """Verify ef_kg_per_tonne = ef_m3 * CH4_density."""
        result = db.get_coal_methane_factor("BITUMINOUS")
        expected_kg = (Decimal("10") * CH4_DENSITY_KG_PER_M3).quantize(
            Decimal("0.00000001")
        )
        assert result["ef_kg_per_tonne_decimal"] == expected_kg

    def test_coal_post_mining_fraction_included(self, db):
        result = db.get_coal_methane_factor("ANTHRACITE")
        assert result["post_mining_fraction"] == str(POST_MINING_FRACTION)

    def test_coal_unknown_rank_returns_none(self, db):
        result = db.get_coal_methane_factor("UNKNOWN_RANK")
        assert result is None

    def test_coal_case_insensitive(self, db):
        result = db.get_coal_methane_factor("bituminous")
        assert result is not None
        assert result["coal_rank"] == "BITUMINOUS"

    def test_coal_uncertainty_pct(self, db):
        result = db.get_coal_methane_factor("ANTHRACITE")
        assert result["uncertainty_pct"] == "50"

    def test_coal_lignite_uncertainty_pct(self, db):
        result = db.get_coal_methane_factor("LIGNITE")
        assert result["uncertainty_pct"] == "75"


# ===========================================================================
# TestWastewaterFactors - 10 tests
# ===========================================================================


class TestWastewaterFactors:
    """Test get_wastewater_factor for all 10 treatment types."""

    @pytest.mark.parametrize("treatment,expected_mcf", [
        ("UNTREATED_DISCHARGE", Decimal("0.1")),
        ("AEROBIC_WELL_MANAGED", Decimal("0.0")),
        ("AEROBIC_POORLY_MANAGED", Decimal("0.3")),
        ("ANAEROBIC_REACTOR", Decimal("0.8")),
        ("ANAEROBIC_LAGOON_DEEP", Decimal("0.8")),
        ("ANAEROBIC_LAGOON_SHALLOW", Decimal("0.2")),
        ("FACULTATIVE_LAGOON", Decimal("0.2")),
        ("SEPTIC_SYSTEM", Decimal("0.5")),
        ("LATRINE_DRY", Decimal("0.1")),
        ("LATRINE_WET", Decimal("0.7")),
    ])
    def test_wastewater_mcf_value(self, db, treatment, expected_mcf):
        result = db.get_wastewater_factor(treatment)
        assert result is not None
        assert result["mcf_decimal"] == expected_mcf
        assert result["bo_decimal"] == WASTEWATER_BO_KG_CH4_PER_KG_BOD
        assert result["n2o_ef_decimal"] == WASTEWATER_N2O_EF_KG_PER_KG_N
        assert "provenance_hash" in result

    def test_wastewater_bo_value(self, db):
        result = db.get_wastewater_factor("ANAEROBIC_REACTOR")
        assert result["bo_kg_ch4_per_kg_bod"] == str(WASTEWATER_BO_KG_CH4_PER_KG_BOD)

    def test_wastewater_n2o_ratio_present(self, db):
        result = db.get_wastewater_factor("SEPTIC_SYSTEM")
        assert result["n2o_n_ratio"] == str(N2O_N_RATIO)

    def test_wastewater_unknown_type_returns_none(self, db):
        result = db.get_wastewater_factor("UNKNOWN_TREATMENT")
        assert result is None

    def test_wastewater_case_insensitive(self, db):
        result = db.get_wastewater_factor("anaerobic_reactor")
        assert result is not None
        assert result["treatment_type"] == "ANAEROBIC_REACTOR"


# ===========================================================================
# TestPneumaticRates - 5 tests
# ===========================================================================


class TestPneumaticRates:
    """Test get_pneumatic_rate for all device types."""

    @pytest.mark.parametrize("device_type,expected_m3", [
        ("high_bleed", Decimal("37.8")),
        ("low_bleed", Decimal("0.9440")),
        ("intermittent", Decimal("9.166")),
        ("zero_bleed", Decimal("0")),
    ])
    def test_pneumatic_rate(self, db, device_type, expected_m3):
        result = db.get_pneumatic_rate(device_type)
        assert result is not None
        assert result["rate_m3_per_day_decimal"] == expected_m3
        assert "provenance_hash" in result

    def test_pneumatic_unknown_device_returns_none(self, db):
        result = db.get_pneumatic_rate("nonexistent_device")
        assert result is None


# ===========================================================================
# TestGasComposition - 5 tests
# ===========================================================================


class TestGasComposition:
    """Test get_gas_composition and weight fraction helpers."""

    def test_default_composition_species(self, db):
        result = db.get_gas_composition()
        assert result["total_species"] == 4
        species_names = {s["species"] for s in result["species"]}
        assert species_names == {"CH4", "C2H6", "CO2", "N2"}

    def test_default_ch4_mole_fraction(self, db):
        result = db.get_gas_composition()
        assert result["ch4_mole_fraction"] == "0.950"
        assert result["is_custom"] is False

    def test_weight_fractions_sum_approximately_one(self, db):
        total = sum(DEFAULT_WEIGHT_FRACTIONS.values())
        assert abs(total - Decimal("1")) < Decimal("0.001")

    def test_get_weight_fraction_ch4(self, db):
        wf = db.get_weight_fraction("CH4")
        assert wf > Decimal("0.9")  # CH4 dominates

    def test_get_mole_fraction_co2(self, db):
        mf = db.get_mole_fraction("CO2")
        assert mf == Decimal("0.010")

    def test_get_mole_fraction_unknown_species(self, db):
        mf = db.get_mole_fraction("UNKNOWN_GAS")
        assert mf == Decimal("0")

    def test_custom_composition(self, db):
        custom = {
            "CH4": {"mole_fraction": Decimal("0.80"), "molecular_weight": Decimal("16.043")},
            "CO2": {"mole_fraction": Decimal("0.20"), "molecular_weight": Decimal("44.009")},
        }
        result = db.get_gas_composition(custom_composition=custom)
        assert result["total_species"] == 2
        assert result["is_custom"] is True


# ===========================================================================
# TestListMethods - 5 tests
# ===========================================================================


class TestListMethods:
    """Test list_sources, list_components, list_custom_factors."""

    def test_list_sources_all(self, db):
        result = db.list_sources()
        assert result["total"] == 25
        assert len(result["sources"]) == 25

    def test_list_sources_filtered_by_category(self, db):
        result = db.list_sources(category="EQUIPMENT_LEAK")
        assert result["total"] == 11
        assert all(s["category"] == "EQUIPMENT_LEAK" for s in result["sources"])

    def test_list_sources_coal_mine_category(self, db):
        result = db.list_sources(category="COAL_MINE")
        assert result["total"] == 3

    def test_list_components_all(self, db):
        result = db.list_components()
        assert result["total_builtin"] == len(COMPONENT_EMISSION_FACTORS)
        assert result["total"] >= result["total_builtin"]

    def test_list_custom_factors_empty_initially(self, db):
        result = db.list_custom_factors()
        assert result["total"] == 0
        assert result["custom_factors"] == []


# ===========================================================================
# TestEdgeCases - 5 tests
# ===========================================================================


class TestEdgeCases:
    """Test custom factor registration, statistics, GWP lookups, edge cases."""

    def test_register_custom_factor(self, db):
        result = db.register_custom_factor({
            "component_type": "valve",
            "service_type": "gas",
            "ef_kg_per_hr": "0.01200",
            "source": "Site-specific measurement",
        })
        assert "factor_id" in result
        assert result["ef_kg_per_hr"] == "0.01200"
        assert result["provenance_hash"]

    def test_custom_factor_takes_precedence(self, db):
        db.register_custom_factor({
            "component_type": "valve", "service_type": "gas",
            "ef_kg_per_hr": "0.05000", "source": "CUSTOM",
        })
        result = db.get_component_ef("valve", "gas")
        assert result["is_custom"] is True
        assert result["ef_decimal"] == Decimal("0.05000")

    def test_remove_custom_factor(self, db):
        reg = db.register_custom_factor({
            "component_type": "pump", "service_type": "gas",
            "ef_kg_per_hr": "0.999",
        })
        assert db.remove_custom_factor(reg["factor_id"]) is True
        assert db.remove_custom_factor("nonexistent_id") is False

    def test_register_negative_ef_raises(self, db):
        with pytest.raises(ValueError, match="ef_kg_per_hr must be >= 0"):
            db.register_custom_factor({
                "component_type": "valve", "service_type": "gas",
                "ef_kg_per_hr": "-1.0",
            })

    def test_register_missing_fields_raises(self, db):
        with pytest.raises(ValueError, match="component_type is required"):
            db.register_custom_factor({
                "service_type": "gas", "ef_kg_per_hr": "0.01",
            })
        with pytest.raises(ValueError, match="service_type is required"):
            db.register_custom_factor({
                "component_type": "valve", "ef_kg_per_hr": "0.01",
            })
        with pytest.raises(ValueError, match="ef_kg_per_hr is required"):
            db.register_custom_factor({
                "component_type": "valve", "service_type": "gas",
            })

    def test_gwp_ar6_ch4(self, db):
        result = db.get_gwp("CH4", "AR6")
        assert result is not None
        assert result["gwp_decimal"] == Decimal("27.9")
        assert result["horizon"] == "100-year"

    def test_gwp_ar6_20yr_ch4(self, db):
        result = db.get_gwp("CH4", "AR6_20YR")
        assert result["gwp_decimal"] == Decimal("81.2")
        assert result["horizon"] == "20-year"

    def test_gwp_ar4_n2o(self, db):
        result = db.get_gwp("N2O", "AR4")
        assert result["gwp_decimal"] == Decimal("298")

    def test_gwp_unknown_gas_returns_none(self, db):
        result = db.get_gwp("XenonFake", "AR6")
        assert result is None

    def test_gwp_unknown_source_returns_none(self, db):
        result = db.get_gwp("CH4", "AR99")
        assert result is None

    def test_get_all_gwps(self, db):
        result = db.get_all_gwps("AR6")
        assert "values" in result
        assert "CH4" in result["values"]
        assert result["gwp_source"] == "AR6"

    def test_get_all_gwps_unknown_source(self, db):
        result = db.get_all_gwps("NONEXISTENT")
        assert "error" in result

    def test_default_gwp_source_ar5(self, db_ar5):
        result = db_ar5.get_gwp("CH4")
        assert result["gwp_decimal"] == Decimal("28")
        assert result["gwp_source"] == "AR5"

    def test_statistics_query_count(self, db):
        db.get_source_info("EQUIPMENT_LEAK_VALVE_GAS")
        db.get_component_ef("valve", "gas")
        db.get_gwp("CH4", "AR6")
        stats = db.get_statistics()
        assert stats["total_queries"] >= 3
        assert stats["builtin_source_types"] == 25
        assert stats["builtin_component_efs"] == len(COMPONENT_EMISSION_FACTORS)

    def test_correlation_coefficients_valve_gas(self, db):
        result = db.get_correlation_coefficients("valve", "gas")
        assert result is not None
        assert result["a_decimal"] == Decimal("-6.36040")
        assert result["b_decimal"] == Decimal("0.79690")
        assert result["equation"] == "log10(kg/hr) = a + b * log10(ppmv)"

    def test_correlation_coefficients_not_found(self, db):
        result = db.get_correlation_coefficients("nonexistent", "gas")
        assert result is None
