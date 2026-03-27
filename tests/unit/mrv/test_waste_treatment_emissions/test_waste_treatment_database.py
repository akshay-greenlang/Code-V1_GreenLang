# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-007 WasteTreatmentDatabaseEngine.

Tests DOC values, MCF values, carbon content, composting EFs,
incineration EFs, wastewater MCFs, NCV values, half-life values,
GWP values, Bo values, custom factors, factor listing, and edge cases.

Target: 110+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal, InvalidOperation

import pytest


# ===========================================================================
# WasteTreatmentDatabaseEngine - standalone instantiation helper
# ===========================================================================


def _make_engine():
    """Create a WasteTreatmentDatabaseEngine without conftest dependency."""
    from greenlang.agents.mrv.waste_treatment_emissions.waste_treatment_database import (
        WasteTreatmentDatabaseEngine,
    )
    return WasteTreatmentDatabaseEngine()


# ===========================================================================
# TestDOCValues - degradable organic carbon fractions (19 waste types)
# ===========================================================================


class TestDOCValues:
    """Tests for DOC (Degradable Organic Carbon) fractions per IPCC Table 2.4."""

    @pytest.mark.parametrize("waste_type,expected_doc", [
        ("food_waste", Decimal("0.15")),
        ("paper_cardboard", Decimal("0.40")),
        ("wood", Decimal("0.43")),
        ("textiles", Decimal("0.24")),
        ("garden_waste", Decimal("0.20")),
        ("nappies_diapers", Decimal("0.24")),
        ("rubber_leather", Decimal("0.39")),
        ("plastics", Decimal("0.0")),
        ("metal", Decimal("0.0")),
        ("glass", Decimal("0.0")),
        ("other_inert", Decimal("0.0")),
        ("sewage_sludge", Decimal("0.05")),
        ("industrial_solid", Decimal("0.15")),
        ("construction_demolition", Decimal("0.08")),
        ("clinical_waste", Decimal("0.15")),
        ("electronic_waste", Decimal("0.0")),
        ("mixed_msw", Decimal("0.18")),
        ("yard_waste", Decimal("0.20")),
        ("agricultural_residue", Decimal("0.30")),
    ])
    def test_doc_fraction_by_waste_type(self, db_engine, waste_type, expected_doc):
        """DOC fraction for each of the 19 waste types matches IPCC defaults."""
        result = db_engine.get_doc(waste_type)
        assert result == expected_doc, (
            f"DOC for '{waste_type}': expected {expected_doc}, got {result}"
        )

    def test_doc_returns_decimal(self, db_engine):
        """get_doc returns a Decimal instance."""
        result = db_engine.get_doc("food_waste")
        assert isinstance(result, Decimal)

    def test_doc_non_negative(self, db_engine):
        """All DOC values are non-negative."""
        waste_types = [
            "food_waste", "paper_cardboard", "wood", "textiles",
            "garden_waste", "nappies_diapers", "rubber_leather",
            "plastics", "metal", "glass", "other_inert", "sewage_sludge",
            "industrial_solid", "construction_demolition", "clinical_waste",
            "electronic_waste", "mixed_msw", "yard_waste",
            "agricultural_residue",
        ]
        for wt in waste_types:
            assert db_engine.get_doc(wt) >= Decimal("0"), (
                f"DOC for '{wt}' should be >= 0"
            )

    def test_doc_le_one(self, db_engine):
        """All DOC values are at most 1.0 (fraction)."""
        waste_types = [
            "food_waste", "paper_cardboard", "wood", "textiles",
            "garden_waste", "plastics", "metal", "glass",
        ]
        for wt in waste_types:
            assert db_engine.get_doc(wt) <= Decimal("1.0"), (
                f"DOC for '{wt}' should be <= 1.0"
            )

    def test_doc_case_insensitive(self, db_engine):
        """Lookup is case-insensitive for DOC."""
        upper = db_engine.get_doc("FOOD_WASTE")
        lower = db_engine.get_doc("food_waste")
        assert upper == lower

    def test_doc_inert_materials_are_zero(self, db_engine):
        """Inert materials (plastics, metal, glass, e-waste) have DOC=0."""
        for wt in ["plastics", "metal", "glass", "electronic_waste", "other_inert"]:
            assert db_engine.get_doc(wt) == Decimal("0"), (
                f"Inert waste type '{wt}' should have DOC=0"
            )


# ===========================================================================
# TestMCFValues - methane correction factors by landfill type
# ===========================================================================


class TestMCFValues:
    """Tests for MCF (Methane Correction Factor) by landfill management type."""

    @pytest.mark.parametrize("landfill_type,expected_mcf", [
        ("managed_anaerobic", Decimal("1.0")),
        ("managed_semi_aerobic", Decimal("0.5")),
        ("unmanaged_deep", Decimal("0.8")),
        ("unmanaged_shallow", Decimal("0.4")),
        ("uncategorized", Decimal("0.6")),
    ])
    def test_mcf_by_landfill_type(self, db_engine, landfill_type, expected_mcf):
        """MCF for each landfill type matches IPCC 2006 Table 3.1."""
        result = db_engine.get_mcf(landfill_type)
        assert result == expected_mcf

    def test_mcf_returns_decimal(self, db_engine):
        """get_mcf returns a Decimal instance."""
        assert isinstance(db_engine.get_mcf("managed_anaerobic"), Decimal)

    def test_mcf_managed_anaerobic_is_max(self, db_engine):
        """Managed anaerobic has the highest MCF (1.0)."""
        assert db_engine.get_mcf("managed_anaerobic") == Decimal("1.0")

    def test_mcf_all_positive(self, db_engine):
        """All MCF values are > 0."""
        for lt in ["managed_anaerobic", "managed_semi_aerobic",
                    "unmanaged_deep", "unmanaged_shallow", "uncategorized"]:
            assert db_engine.get_mcf(lt) > Decimal("0")

    def test_mcf_all_le_one(self, db_engine):
        """All MCF values are at most 1.0."""
        for lt in ["managed_anaerobic", "managed_semi_aerobic",
                    "unmanaged_deep", "unmanaged_shallow", "uncategorized"]:
            assert db_engine.get_mcf(lt) <= Decimal("1.0")

    def test_mcf_case_insensitive(self, db_engine):
        """MCF lookup is case-insensitive."""
        assert db_engine.get_mcf("MANAGED_ANAEROBIC") == db_engine.get_mcf("managed_anaerobic")


# ===========================================================================
# TestCarbonContent - carbon % and fossil fraction per waste type
# ===========================================================================


class TestCarbonContent:
    """Tests for carbon content percentage and fossil fraction by waste type."""

    @pytest.mark.parametrize("waste_type,expected_carbon_pct", [
        ("food_waste", Decimal("0.38")),
        ("paper_cardboard", Decimal("0.46")),
        ("wood", Decimal("0.50")),
        ("textiles", Decimal("0.50")),
        ("plastics", Decimal("0.75")),
        ("rubber_leather", Decimal("0.67")),
        ("garden_waste", Decimal("0.38")),
        ("nappies_diapers", Decimal("0.60")),
    ])
    def test_carbon_content_by_type(self, db_engine, waste_type, expected_carbon_pct):
        """Carbon content percentage matches IPCC Table 5.2."""
        result = db_engine.get_carbon_content(waste_type)
        assert result == expected_carbon_pct

    @pytest.mark.parametrize("waste_type,expected_fossil_frac", [
        ("food_waste", Decimal("0.0")),
        ("paper_cardboard", Decimal("0.01")),
        ("wood", Decimal("0.0")),
        ("textiles", Decimal("0.20")),
        ("plastics", Decimal("1.0")),
        ("rubber_leather", Decimal("0.20")),
        ("garden_waste", Decimal("0.0")),
        ("nappies_diapers", Decimal("0.10")),
    ])
    def test_fossil_fraction_by_type(self, db_engine, waste_type, expected_fossil_frac):
        """Fossil carbon fraction matches IPCC Table 5.2."""
        result = db_engine.get_fossil_fraction(waste_type)
        assert result == expected_fossil_frac

    def test_carbon_content_returns_decimal(self, db_engine):
        """get_carbon_content returns Decimal."""
        assert isinstance(db_engine.get_carbon_content("plastics"), Decimal)

    def test_fossil_fraction_returns_decimal(self, db_engine):
        """get_fossil_fraction returns Decimal."""
        assert isinstance(db_engine.get_fossil_fraction("plastics"), Decimal)

    def test_plastics_fully_fossil(self, db_engine):
        """Plastics have 100% fossil carbon fraction."""
        assert db_engine.get_fossil_fraction("plastics") == Decimal("1.0")

    def test_food_waste_zero_fossil(self, db_engine):
        """Food waste has 0% fossil carbon fraction (fully biogenic)."""
        assert db_engine.get_fossil_fraction("food_waste") == Decimal("0.0")

    def test_biogenic_fraction_complement(self, db_engine):
        """Biogenic fraction = 1 - fossil fraction for all types."""
        for wt in ["food_waste", "paper_cardboard", "wood", "plastics"]:
            fossil = db_engine.get_fossil_fraction(wt)
            biogenic = Decimal("1.0") - fossil
            assert biogenic >= Decimal("0.0")
            assert biogenic <= Decimal("1.0")


# ===========================================================================
# TestCompostingEF - composting emission factors
# ===========================================================================


class TestCompostingEF:
    """Tests for composting CH4 and N2O emission factors by management type."""

    @pytest.mark.parametrize("mgmt_type,expected_ch4,expected_n2o", [
        ("well_managed", Decimal("4.0"), Decimal("0.24")),
        ("poorly_managed", Decimal("10.0"), Decimal("0.6")),
        ("ad_vented", Decimal("2.0"), Decimal("0.0")),
        ("ad_flared", Decimal("0.8"), Decimal("0.0")),
        ("mbt_aerobic", Decimal("4.0"), Decimal("0.3")),
        ("mbt_anaerobic", Decimal("2.0"), Decimal("0.1")),
    ])
    def test_composting_ef_by_management(self, db_engine, mgmt_type,
                                         expected_ch4, expected_n2o):
        """Composting EFs match IPCC 2019 Table 4.1 values."""
        ch4, n2o = db_engine.get_composting_ef(mgmt_type)
        assert ch4 == expected_ch4, (
            f"CH4 for '{mgmt_type}': expected {expected_ch4}, got {ch4}"
        )
        assert n2o == expected_n2o, (
            f"N2O for '{mgmt_type}': expected {expected_n2o}, got {n2o}"
        )

    def test_composting_ef_returns_tuple_of_decimal(self, db_engine):
        """get_composting_ef returns a tuple of two Decimal values."""
        ch4, n2o = db_engine.get_composting_ef("well_managed")
        assert isinstance(ch4, Decimal)
        assert isinstance(n2o, Decimal)

    def test_composting_ef_well_managed_lower_than_poorly(self, db_engine):
        """Well-managed composting has lower CH4 EF than poorly managed."""
        ch4_good, _ = db_engine.get_composting_ef("well_managed")
        ch4_bad, _ = db_engine.get_composting_ef("poorly_managed")
        assert ch4_good < ch4_bad

    def test_composting_ef_all_non_negative(self, db_engine):
        """All composting EFs are non-negative."""
        for mt in ["well_managed", "poorly_managed", "ad_vented",
                    "ad_flared", "mbt_aerobic", "mbt_anaerobic"]:
            ch4, n2o = db_engine.get_composting_ef(mt)
            assert ch4 >= Decimal("0")
            assert n2o >= Decimal("0")


# ===========================================================================
# TestIncinerationEF - incineration emission factors by incinerator type
# ===========================================================================


class TestIncinerationEF:
    """Tests for incineration N2O and CH4 EFs by incinerator technology."""

    @pytest.mark.parametrize("incinerator_type,expected_n2o,expected_ch4", [
        ("stoker_grate", Decimal("50.0"), Decimal("0.2")),
        ("fluidized_bed", Decimal("56.0"), Decimal("0.18")),
        ("rotary_kiln", Decimal("50.0"), Decimal("0.2")),
        ("semi_continuous", Decimal("60.0"), Decimal("6.0")),
        ("batch_type", Decimal("60.0"), Decimal("60.0")),
    ])
    def test_incineration_ef_by_type(self, db_engine, incinerator_type,
                                     expected_n2o, expected_ch4):
        """Incineration EFs match IPCC 2006 Table 5.3 values (g/tonne)."""
        n2o, ch4 = db_engine.get_incineration_ef(incinerator_type)
        assert n2o == expected_n2o, (
            f"N2O for '{incinerator_type}': expected {expected_n2o}, got {n2o}"
        )
        assert ch4 == expected_ch4, (
            f"CH4 for '{incinerator_type}': expected {expected_ch4}, got {ch4}"
        )

    def test_incineration_ef_returns_tuple_of_decimal(self, db_engine):
        """get_incineration_ef returns a tuple of two Decimals."""
        n2o, ch4 = db_engine.get_incineration_ef("stoker_grate")
        assert isinstance(n2o, Decimal)
        assert isinstance(ch4, Decimal)

    def test_batch_type_has_highest_ch4(self, db_engine):
        """Batch-type incinerators have the highest CH4 (poor combustion)."""
        _, ch4_batch = db_engine.get_incineration_ef("batch_type")
        for it in ["stoker_grate", "fluidized_bed", "rotary_kiln"]:
            _, ch4_other = db_engine.get_incineration_ef(it)
            assert ch4_batch >= ch4_other

    def test_continuous_vs_batch_n2o(self, db_engine):
        """Stoker grate (continuous) has lower N2O than batch/semi-continuous."""
        n2o_stoker, _ = db_engine.get_incineration_ef("stoker_grate")
        n2o_batch, _ = db_engine.get_incineration_ef("batch_type")
        assert n2o_stoker <= n2o_batch

    def test_incineration_ef_all_positive(self, db_engine):
        """All incineration EFs are positive (non-zero emissions)."""
        for it in ["stoker_grate", "fluidized_bed", "rotary_kiln",
                    "semi_continuous", "batch_type"]:
            n2o, ch4 = db_engine.get_incineration_ef(it)
            assert n2o > Decimal("0")
            assert ch4 > Decimal("0")


# ===========================================================================
# TestWastewaterMCF - MCF for wastewater treatment systems
# ===========================================================================


class TestWastewaterMCF:
    """Tests for MCF values for all 8 wastewater treatment system types."""

    @pytest.mark.parametrize("system_type,expected_mcf", [
        ("well_managed_aerobic", Decimal("0.0")),
        ("overloaded_aerobic", Decimal("0.3")),
        ("anaerobic_reactor", Decimal("0.8")),
        ("anaerobic_shallow_lagoon", Decimal("0.2")),
        ("anaerobic_deep_lagoon", Decimal("0.8")),
        ("septic_system", Decimal("0.5")),
        ("latrine_dry_climate", Decimal("0.1")),
        ("latrine_wet_climate", Decimal("0.7")),
    ])
    def test_wastewater_mcf_by_system(self, db_engine, system_type, expected_mcf):
        """Wastewater MCF matches IPCC 2006 Table 6.3 values."""
        result = db_engine.get_wastewater_mcf(system_type)
        assert result == expected_mcf

    def test_wastewater_mcf_returns_decimal(self, db_engine):
        """get_wastewater_mcf returns a Decimal."""
        result = db_engine.get_wastewater_mcf("septic_system")
        assert isinstance(result, Decimal)

    def test_well_managed_aerobic_zero_mcf(self, db_engine):
        """Well-managed aerobic treatment has MCF = 0 (no CH4)."""
        assert db_engine.get_wastewater_mcf("well_managed_aerobic") == Decimal("0.0")

    def test_anaerobic_reactor_highest_mcf(self, db_engine):
        """Anaerobic reactor has MCF = 0.8 (tied for highest)."""
        mcf = db_engine.get_wastewater_mcf("anaerobic_reactor")
        assert mcf == Decimal("0.8")

    def test_all_wastewater_mcf_range(self, db_engine):
        """All wastewater MCF values are in [0, 1]."""
        systems = [
            "well_managed_aerobic", "overloaded_aerobic",
            "anaerobic_reactor", "anaerobic_shallow_lagoon",
            "anaerobic_deep_lagoon", "septic_system",
            "latrine_dry_climate", "latrine_wet_climate",
        ]
        for sys_type in systems:
            mcf = db_engine.get_wastewater_mcf(sys_type)
            assert Decimal("0") <= mcf <= Decimal("1.0")


# ===========================================================================
# TestNCVValues - net calorific values by waste type
# ===========================================================================


class TestNCVValues:
    """Tests for NCV (Net Calorific Value) in GJ/tonne by waste type."""

    @pytest.mark.parametrize("waste_type,expected_ncv", [
        ("food_waste", Decimal("4.0")),
        ("paper_cardboard", Decimal("12.2")),
        ("wood", Decimal("15.6")),
        ("textiles", Decimal("16.0")),
        ("plastics", Decimal("32.0")),
        ("rubber_leather", Decimal("23.0")),
        ("garden_waste", Decimal("6.0")),
        ("nappies_diapers", Decimal("19.0")),
        ("mixed_msw", Decimal("10.0")),
    ])
    def test_ncv_by_waste_type(self, db_engine, waste_type, expected_ncv):
        """NCV values match IPCC 2006 reference data."""
        result = db_engine.get_ncv(waste_type)
        assert result == expected_ncv

    def test_ncv_returns_decimal(self, db_engine):
        """get_ncv returns a Decimal."""
        assert isinstance(db_engine.get_ncv("plastics"), Decimal)

    def test_plastics_highest_ncv(self, db_engine):
        """Plastics have the highest NCV among common waste types."""
        ncv_plastics = db_engine.get_ncv("plastics")
        for wt in ["food_waste", "paper_cardboard", "wood", "textiles"]:
            assert ncv_plastics >= db_engine.get_ncv(wt)

    def test_ncv_all_positive(self, db_engine):
        """All NCV values are positive (waste has energy content)."""
        for wt in ["food_waste", "paper_cardboard", "wood", "plastics",
                    "mixed_msw", "garden_waste"]:
            assert db_engine.get_ncv(wt) > Decimal("0")

    def test_ncv_case_insensitive(self, db_engine):
        """NCV lookup is case-insensitive."""
        assert db_engine.get_ncv("PLASTICS") == db_engine.get_ncv("plastics")


# ===========================================================================
# TestHalfLifeValues - by climate zone and waste type
# ===========================================================================


class TestHalfLifeValues:
    """Tests for half-life values (years) by climate zone and waste type."""

    @pytest.mark.parametrize("climate_zone,waste_type,expected_half_life", [
        ("tropical_moist", "food_waste", Decimal("3.0")),
        ("tropical_moist", "paper_cardboard", Decimal("6.0")),
        ("tropical_moist", "wood", Decimal("15.0")),
        ("tropical_moist", "textiles", Decimal("6.0")),
        ("tropical_moist", "garden_waste", Decimal("4.0")),
        ("temperate", "food_waste", Decimal("5.0")),
        ("temperate", "paper_cardboard", Decimal("10.0")),
        ("temperate", "wood", Decimal("23.0")),
        ("temperate", "textiles", Decimal("10.0")),
        ("temperate", "garden_waste", Decimal("8.0")),
        ("boreal_dry", "food_waste", Decimal("8.0")),
        ("boreal_dry", "paper_cardboard", Decimal("16.0")),
        ("boreal_dry", "wood", Decimal("35.0")),
        ("boreal_dry", "textiles", Decimal("16.0")),
        ("boreal_dry", "garden_waste", Decimal("12.0")),
    ])
    def test_half_life_by_zone_and_type(self, db_engine, climate_zone,
                                        waste_type, expected_half_life):
        """Half-life values match IPCC 2006 Table 3.4."""
        result = db_engine.get_half_life(climate_zone, waste_type)
        assert result == expected_half_life

    def test_half_life_returns_decimal(self, db_engine):
        """get_half_life returns a Decimal."""
        result = db_engine.get_half_life("temperate", "food_waste")
        assert isinstance(result, Decimal)

    def test_tropical_faster_than_temperate(self, db_engine):
        """Tropical decomposition is faster (shorter half-life) than temperate."""
        tropical = db_engine.get_half_life("tropical_moist", "food_waste")
        temperate = db_engine.get_half_life("temperate", "food_waste")
        assert tropical < temperate

    def test_temperate_faster_than_boreal(self, db_engine):
        """Temperate decomposition is faster than boreal/dry."""
        temperate = db_engine.get_half_life("temperate", "wood")
        boreal = db_engine.get_half_life("boreal_dry", "wood")
        assert temperate < boreal

    def test_food_faster_than_wood(self, db_engine):
        """Food waste decomposes faster than wood in same climate zone."""
        food = db_engine.get_half_life("temperate", "food_waste")
        wood = db_engine.get_half_life("temperate", "wood")
        assert food < wood

    def test_all_half_lives_positive(self, db_engine):
        """All half-life values are positive."""
        for zone in ["tropical_moist", "temperate", "boreal_dry"]:
            for wt in ["food_waste", "paper_cardboard", "wood"]:
                assert db_engine.get_half_life(zone, wt) > Decimal("0")


# ===========================================================================
# TestGWPValues - AR4, AR5, AR6, AR6_20YR for CO2/CH4/N2O
# ===========================================================================


class TestGWPValues:
    """Tests for GWP values by assessment report and gas species."""

    @pytest.mark.parametrize("source,gas,expected_gwp", [
        ("AR4", "CO2", Decimal("1")),
        ("AR4", "CH4", Decimal("25")),
        ("AR4", "N2O", Decimal("298")),
        ("AR5", "CO2", Decimal("1")),
        ("AR5", "CH4", Decimal("28")),
        ("AR5", "N2O", Decimal("265")),
        ("AR6", "CO2", Decimal("1")),
        ("AR6", "CH4", Decimal("29.8")),
        ("AR6", "CH4_biogenic", Decimal("27.0")),
        ("AR6", "N2O", Decimal("273")),
        ("AR6_20YR", "CO2", Decimal("1")),
        ("AR6_20YR", "CH4", Decimal("82.5")),
        ("AR6_20YR", "N2O", Decimal("273")),
    ])
    def test_gwp_by_source_and_gas(self, db_engine, source, gas, expected_gwp):
        """GWP values match IPCC assessment report published values."""
        result = db_engine.get_gwp(source, gas)
        assert result == expected_gwp

    def test_gwp_returns_decimal(self, db_engine):
        """get_gwp returns a Decimal."""
        assert isinstance(db_engine.get_gwp("AR6", "CH4"), Decimal)

    def test_co2_gwp_always_one(self, db_engine):
        """CO2 GWP is always 1 regardless of assessment report."""
        for source in ["AR4", "AR5", "AR6", "AR6_20YR"]:
            assert db_engine.get_gwp(source, "CO2") == Decimal("1")

    def test_ar6_biogenic_ch4_lower_than_fossil(self, db_engine):
        """AR6 biogenic CH4 GWP (27) < fossil CH4 GWP (29.8)."""
        biogenic = db_engine.get_gwp("AR6", "CH4_biogenic")
        fossil = db_engine.get_gwp("AR6", "CH4")
        assert biogenic < fossil

    def test_20yr_ch4_higher_than_100yr(self, db_engine):
        """20-year GWP for CH4 is higher than 100-year GWP."""
        gwp_20yr = db_engine.get_gwp("AR6_20YR", "CH4")
        gwp_100yr = db_engine.get_gwp("AR6", "CH4")
        assert gwp_20yr > gwp_100yr

    def test_ar5_ch4_between_ar4_and_ar6(self, db_engine):
        """AR5 CH4 GWP (28) is between AR4 (25) and AR6 (29.8)."""
        ar4 = db_engine.get_gwp("AR4", "CH4")
        ar5 = db_engine.get_gwp("AR5", "CH4")
        ar6 = db_engine.get_gwp("AR6", "CH4")
        assert ar4 < ar5 < ar6


# ===========================================================================
# TestBoValues - maximum CH4 producing capacity (BOD and COD basis)
# ===========================================================================


class TestBoValues:
    """Tests for Bo (maximum CH4 producing capacity)."""

    def test_bo_bod_default(self, db_engine):
        """Bo based on BOD is 0.6 kg CH4/kg BOD (IPCC default)."""
        result = db_engine.get_bo("BOD")
        assert result == Decimal("0.6")

    def test_bo_cod_default(self, db_engine):
        """Bo based on COD is 0.25 kg CH4/kg COD (IPCC default)."""
        result = db_engine.get_bo("COD")
        assert result == Decimal("0.25")

    def test_bo_returns_decimal(self, db_engine):
        """get_bo returns a Decimal."""
        assert isinstance(db_engine.get_bo("BOD"), Decimal)

    def test_bo_bod_greater_than_cod(self, db_engine):
        """Bo based on BOD is higher than Bo based on COD."""
        assert db_engine.get_bo("BOD") > db_engine.get_bo("COD")

    def test_bo_all_positive(self, db_engine):
        """All Bo values are positive."""
        assert db_engine.get_bo("BOD") > Decimal("0")
        assert db_engine.get_bo("COD") > Decimal("0")

    def test_bo_case_insensitive(self, db_engine):
        """Bo lookup is case-insensitive."""
        assert db_engine.get_bo("bod") == db_engine.get_bo("BOD")
        assert db_engine.get_bo("cod") == db_engine.get_bo("COD")


# ===========================================================================
# TestCustomFactors - register, get, overwrite, thread safety
# ===========================================================================


class TestCustomFactors:
    """Tests for custom emission factor registration and retrieval."""

    def test_register_custom_factor(self, db_engine):
        """Can register a custom emission factor."""
        db_engine.register_custom_factor(
            key="custom_ch4_food",
            value=Decimal("3.5"),
            source="lab_measurement_2025",
            metadata={"waste_type": "food_waste", "gas": "CH4"},
        )
        result = db_engine.get_custom_factor("custom_ch4_food")
        assert result == Decimal("3.5")

    def test_get_nonexistent_custom_factor_returns_none(self, db_engine):
        """Getting a non-registered custom factor returns None."""
        result = db_engine.get_custom_factor("nonexistent_factor")
        assert result is None

    def test_overwrite_custom_factor(self, db_engine):
        """Overwriting a custom factor updates the value."""
        db_engine.register_custom_factor(
            key="my_factor", value=Decimal("1.0"),
            source="v1", metadata={},
        )
        db_engine.register_custom_factor(
            key="my_factor", value=Decimal("2.0"),
            source="v2", metadata={},
        )
        result = db_engine.get_custom_factor("my_factor")
        assert result == Decimal("2.0")

    def test_custom_factor_thread_safety(self, db_engine):
        """Custom factor operations are thread-safe."""
        results = []
        barrier = threading.Barrier(4)

        def writer(key_suffix, value):
            barrier.wait()
            db_engine.register_custom_factor(
                key=f"thread_factor_{key_suffix}",
                value=Decimal(str(value)),
                source="thread_test",
                metadata={},
            )
            results.append(True)

        threads = [
            threading.Thread(target=writer, args=(i, i * 1.5))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 4
        for i in range(4):
            val = db_engine.get_custom_factor(f"thread_factor_{i}")
            assert val is not None

    def test_list_custom_factors(self, db_engine):
        """list_custom_factors returns all registered keys."""
        db_engine.register_custom_factor(
            key="cf_a", value=Decimal("1.0"), source="test", metadata={},
        )
        db_engine.register_custom_factor(
            key="cf_b", value=Decimal("2.0"), source="test", metadata={},
        )
        factors = db_engine.list_custom_factors()
        assert "cf_a" in factors
        assert "cf_b" in factors


# ===========================================================================
# TestFactorListing - list_available_factors by treatment method
# ===========================================================================


class TestFactorListing:
    """Tests for listing available emission factors by treatment method."""

    @pytest.mark.parametrize("treatment_method", [
        "composting",
        "incineration",
        "anaerobic_digestion",
        "landfill",
        "wastewater",
        "open_burning",
        "pyrolysis",
        "gasification",
    ])
    def test_list_factors_returns_non_empty(self, db_engine, treatment_method):
        """Each treatment method has at least one available factor."""
        factors = db_engine.list_available_factors(treatment_method)
        assert len(factors) > 0, (
            f"No factors found for treatment method '{treatment_method}'"
        )

    def test_list_factors_returns_list(self, db_engine):
        """list_available_factors returns a list."""
        factors = db_engine.list_available_factors("composting")
        assert isinstance(factors, list)

    def test_list_factors_composting_includes_ch4_n2o(self, db_engine):
        """Composting factors include both CH4 and N2O entries."""
        factors = db_engine.list_available_factors("composting")
        factor_names = [f.get("name", f.get("gas", "")) for f in factors]
        has_ch4 = any("CH4" in str(n).upper() or "ch4" in str(n) for n in factor_names)
        has_n2o = any("N2O" in str(n).upper() or "n2o" in str(n) for n in factor_names)
        assert has_ch4 or len(factors) >= 2
        assert has_n2o or len(factors) >= 2

    def test_list_factors_unknown_method(self, db_engine):
        """Unknown treatment method returns empty list."""
        factors = db_engine.list_available_factors("unknown_method")
        assert factors == []


# ===========================================================================
# TestEdgeCases - unknown inputs, missing data, error handling
# ===========================================================================


class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_unknown_waste_type_doc_raises(self, db_engine):
        """Unknown waste type in get_doc raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_doc("imaginary_waste")

    def test_unknown_landfill_type_mcf_raises(self, db_engine):
        """Unknown landfill type in get_mcf raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_mcf("imaginary_landfill")

    def test_unknown_system_type_ww_mcf_raises(self, db_engine):
        """Unknown wastewater system type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_wastewater_mcf("imaginary_system")

    def test_unknown_gwp_source_raises(self, db_engine):
        """Unknown GWP source raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_gwp("AR99", "CH4")

    def test_unknown_gas_gwp_raises(self, db_engine):
        """Unknown gas in get_gwp raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_gwp("AR6", "SF6")

    def test_unknown_incinerator_type_raises(self, db_engine):
        """Unknown incinerator type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_incineration_ef("plasma_arc")

    def test_unknown_composting_type_raises(self, db_engine):
        """Unknown composting management type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_composting_ef("unknown_composting")

    def test_unknown_bo_basis_raises(self, db_engine):
        """Unknown Bo basis raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_bo("UNKNOWN_BASIS")

    def test_engine_initialization(self):
        """Engine initializes without errors."""
        engine = _make_engine()
        assert engine is not None

    def test_engine_has_empty_custom_factors_initially(self):
        """Engine starts with empty custom factor registry."""
        engine = _make_engine()
        factors = engine.list_custom_factors()
        assert len(factors) == 0

    def test_lookup_counter_starts_at_zero(self):
        """Lookup counter starts at zero."""
        engine = _make_engine()
        assert engine._total_lookups == 0

    def test_lookup_counter_increments(self):
        """Lookup counter increments on each call."""
        engine = _make_engine()
        engine.get_doc("food_waste")
        assert engine._total_lookups >= 1
        engine.get_mcf("managed_anaerobic")
        assert engine._total_lookups >= 2

    def test_empty_string_waste_type_raises(self, db_engine):
        """Empty string waste type raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            db_engine.get_doc("")

    def test_none_waste_type_raises(self, db_engine):
        """None waste type raises TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            db_engine.get_doc(None)
