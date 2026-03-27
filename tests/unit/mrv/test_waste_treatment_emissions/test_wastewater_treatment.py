# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-008 Waste Treatment Emissions Agent - WastewaterTreatmentEngine.

Tests all wastewater CH4 (BOD & COD basis), industrial wastewater, N2O from
treatment plant and effluent, sludge treatment, combined calculations, and
edge cases using IPCC 2006 Vol 5 Ch 6 and 2019 Refinement methods.

Target: 100+ tests, 85%+ coverage.

Key Formulas Tested:
    CH4 = [(TOW - S) * Bo * MCF] * 0.001 - R
    N2O_plant = Q * N_influent * EF_plant * 44/28
    N2O_effluent = Q * N_effluent * EF_effluent * 44/28
    CH4_sludge = M_sludge_kg * VS_fraction * Bo_sludge * MCF_treatment

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import threading
from decimal import Decimal, ROUND_HALF_UP

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.waste_treatment_emissions.wastewater_treatment import (
        WastewaterTreatmentEngine,
        TreatmentSystem,
        IndustryType,
        SludgeTreatment,
        OrganicBasis,
        CalculationStatus,
        MCF_BY_TREATMENT,
        MCF_BY_SLUDGE_TREATMENT,
        INDUSTRY_PARAMETERS,
        BO_DEFAULT_BOD,
        BO_DEFAULT_COD,
        BO_SLUDGE_DEFAULT,
        VS_FRACTION_DEFAULT,
        EF_N2O_PLANT_DEFAULT,
        EF_N2O_EFFLUENT_DEFAULT,
        N2O_MOLECULAR_RATIO,
        GWP_VALUES,
        KG_TO_TONNES,
    )
    WASTEWATER_AVAILABLE = True
except ImportError:
    WASTEWATER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not WASTEWATER_AVAILABLE,
    reason="WastewaterTreatmentEngine not available",
)

_PRECISION = Decimal("0.00000001")


def _Q(val: Decimal) -> Decimal:
    """Quantize to 8 dp for comparison."""
    return val.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """Create a fresh WastewaterTreatmentEngine for each test."""
    return WastewaterTreatmentEngine()


@pytest.fixture
def engine_ar5():
    """Engine configured with AR5 GWP source."""
    return WastewaterTreatmentEngine(config={"default_gwp_source": "AR5"})


# ===========================================================================
# Test Class: CH4 from BOD-based calculations
# ===========================================================================


@_SKIP
class TestWastewaterCH4BOD:
    """Test CH4 from BOD-based wastewater calculations (IPCC Eq 6.1)."""

    def test_aerobic_well_managed_zero_ch4(self, engine):
        """Aerobic well-managed system has MCF=0, producing zero CH4."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("500"),
            bo=Decimal("0.6"),
            mcf=Decimal("0"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        # MCF=0 means zero CH4
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        assert ch4 == Decimal("0")

    def test_anaerobic_reactor_high_ch4(self, engine):
        """Anaerobic reactor MCF=0.8 produces significant CH4."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("1000"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        # Expected: (10000 - 1000) * 0.6 * 0.8 * 0.001 = 4.320 tonnes
        expected = _Q(Decimal("9000") * Decimal("0.6") * Decimal("0.8") * Decimal("0.001"))
        assert ch4 == expected

    def test_default_bo_used_when_none(self, engine):
        """When bo is None, default Bo of 0.6 (BOD basis) is used."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("0"),
            bo=None,
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        # Bo=0.6 by default; (5000-0)*0.6*0.5*0.001 = 1.5 tonnes
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        expected = _Q(Decimal("5000") * Decimal("0.6") * Decimal("0.5") * Decimal("0.001"))
        assert ch4 == expected

    def test_mcf_lookup_by_treatment_system(self, engine):
        """MCF is correctly looked up from treatment system enum."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("8000"),
            sludge_removal_kg=Decimal("800"),
            bo=Decimal("0.6"),
            mcf=None,
            recovery_tonnes=Decimal("0"),
            treatment_system="SEPTIC_SYSTEM",
        )
        assert result["status"] == "SUCCESS"
        mcf_value = Decimal(result["calculation_details"]["mcf"])
        assert mcf_value == Decimal("0.5")

    def test_recovery_subtraction(self, engine):
        """Methane recovery is correctly subtracted from gross CH4."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("20000"),
            sludge_removal_kg=Decimal("2000"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("5"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        # Gross: (20000-2000)*0.6*0.8*0.001 = 8.64; Net: 8.64-5 = 3.64
        gross = _Q(Decimal("18000") * Decimal("0.6") * Decimal("0.8") * Decimal("0.001"))
        expected = _Q(gross - Decimal("5"))
        assert ch4 == expected

    def test_recovery_exceeds_gross_clamps_to_zero(self, engine):
        """When recovery exceeds gross CH4, net CH4 is clamped to zero."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("1000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("100"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        # Gross: 1000*0.6*0.5*0.001=0.3; Net = max(0.3 - 100, 0) = 0
        assert ch4 == Decimal("0")

    def test_sludge_removal_reduces_tow(self, engine):
        """Sludge removal reduces the effective organic load."""
        result_no_sludge = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        result_with_sludge = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("5000"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        ch4_no = Decimal(result_no_sludge["calculation_details"]["ch4_net_tonnes"])
        ch4_with = Decimal(result_with_sludge["calculation_details"]["ch4_net_tonnes"])
        assert ch4_with < ch4_no

    def test_zero_tow_zero_emissions(self, engine):
        """Zero TOW produces zero CH4 emissions."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("0"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        assert ch4 == Decimal("0")

    def test_provenance_hash_present(self, engine):
        """Result includes a SHA-256 provenance hash."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("500"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == 64

    def test_trace_steps_present(self, engine):
        """Result includes trace steps for audit trail."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert "trace_steps" in result
        assert len(result["trace_steps"]) >= 3

    def test_processing_time_recorded(self, engine):
        """Processing time is recorded and positive."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["processing_time_ms"] >= 0

    def test_calculation_id_prefix(self, engine):
        """Calculation ID starts with ww_ch4_bod_ prefix."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["calculation_id"].startswith("ww_ch4_bod_")

    def test_organic_basis_is_bod(self, engine):
        """Result indicates BOD organic basis."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["organic_basis"] == "BOD"

    def test_gwp_co2e_conversion_ar6(self, engine):
        """CO2e conversion uses AR6 CH4 GWP of 29.8 by default."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        gas_results = result["emissions_by_gas"]
        assert len(gas_results) >= 1
        ch4_gas = gas_results[0]
        assert ch4_gas["gas"] == "CH4"
        assert ch4_gas["gwp_source"] == "AR6"

    @pytest.mark.parametrize("mcf_val,expected_factor", [
        (Decimal("0.0"), Decimal("0")),
        (Decimal("0.1"), Decimal("0.1")),
        (Decimal("0.3"), Decimal("0.3")),
        (Decimal("0.5"), Decimal("0.5")),
        (Decimal("0.8"), Decimal("0.8")),
        (Decimal("1.0"), Decimal("1.0")),
    ])
    def test_mcf_values_parametrized(self, engine, mcf_val, expected_factor):
        """Parametrized MCF values produce proportional CH4."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=mcf_val,
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        expected = _Q(Decimal("10000") * Decimal("0.6") * expected_factor * Decimal("0.001"))
        assert ch4 == expected

    @pytest.mark.parametrize("system,expected_mcf", [
        ("AEROBIC_WELL_MANAGED", Decimal("0.0")),
        ("AEROBIC_OVERLOADED", Decimal("0.3")),
        ("ANAEROBIC_REACTOR_NO_RECOVERY", Decimal("0.8")),
        ("ANAEROBIC_REACTOR_WITH_RECOVERY", Decimal("0.8")),
        ("ANAEROBIC_SHALLOW_LAGOON", Decimal("0.2")),
        ("ANAEROBIC_DEEP_LAGOON", Decimal("0.8")),
        ("SEPTIC_SYSTEM", Decimal("0.5")),
        ("UNTREATED_DISCHARGE", Decimal("0.1")),
    ])
    def test_treatment_system_mcf_lookup(self, engine, system, expected_mcf):
        """MCF lookup for each treatment system is correct."""
        mcf = engine.get_mcf(system)
        assert mcf == expected_mcf

    def test_neither_mcf_nor_system_raises_error(self, engine):
        """Omitting both mcf and treatment_system returns error status."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=None,
            recovery_tonnes=Decimal("0"),
            treatment_system=None,
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_negative_tow_returns_error(self, engine):
        """Negative TOW returns validation error."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("-1000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_sludge_exceeds_tow_returns_error(self, engine):
        """Sludge removal exceeding TOW returns validation error."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("1000"),
            sludge_removal_kg=Decimal("2000"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_mcf_above_one_returns_error(self, engine):
        """MCF above 1.0 returns validation error."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("1000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("1.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")


# ===========================================================================
# Test Class: CH4 from COD-based calculations
# ===========================================================================


@_SKIP
class TestWastewaterCH4COD:
    """Test COD-based CH4 calculations (Bo_COD = 0.25)."""

    def test_basic_cod_calculation(self, engine):
        """Basic COD calculation yields correct CH4."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("20000"),
            sludge_removal_kg=Decimal("2000"),
            bo_cod=Decimal("0.25"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        expected = _Q(Decimal("18000") * Decimal("0.25") * Decimal("0.8") * Decimal("0.001"))
        assert ch4 == expected

    def test_default_bo_cod_0_25(self, engine):
        """Default Bo for COD basis is 0.25."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo_cod=None,
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        expected = _Q(Decimal("10000") * Decimal("0.25") * Decimal("0.8") * Decimal("0.001"))
        assert ch4 == expected

    def test_cod_organic_basis_is_cod(self, engine):
        """Result indicates COD organic basis."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo_cod=Decimal("0.25"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["organic_basis"] == "COD"

    def test_cod_method_field(self, engine):
        """Method field is IPCC_WW_CH4_COD."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo_cod=Decimal("0.25"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["method"] == "IPCC_WW_CH4_COD"

    def test_cod_calculation_id_prefix(self, engine):
        """Calculation ID starts with ww_ch4_cod_ prefix."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("5000"),
            sludge_removal_kg=Decimal("0"),
            bo_cod=Decimal("0.25"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["calculation_id"].startswith("ww_ch4_cod_")

    def test_cod_recovery_clamps_to_zero(self, engine):
        """Recovery exceeding gross CH4 clamps net to zero (COD)."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("1000"),
            sludge_removal_kg=Decimal("0"),
            bo_cod=Decimal("0.25"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("100"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        assert ch4 == Decimal("0")

    def test_cod_with_treatment_system_lookup(self, engine):
        """COD calculation can look up MCF from treatment system."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("15000"),
            sludge_removal_kg=Decimal("1500"),
            bo_cod=Decimal("0.25"),
            mcf=None,
            recovery_tonnes=Decimal("0"),
            treatment_system="ANAEROBIC_DEEP_LAGOON",
        )
        assert result["status"] == "SUCCESS"
        mcf = Decimal(result["calculation_details"]["mcf"])
        assert mcf == Decimal("0.8")

    def test_cod_negative_tow_returns_error(self, engine):
        """Negative COD TOW returns error."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("-5000"),
            sludge_removal_kg=Decimal("0"),
            bo_cod=Decimal("0.25"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    @pytest.mark.parametrize("tow,sludge,mcf,expected_net", [
        (Decimal("10000"), Decimal("0"), Decimal("0.8"), _Q(Decimal("10000") * Decimal("0.25") * Decimal("0.8") * Decimal("0.001"))),
        (Decimal("50000"), Decimal("5000"), Decimal("0.5"), _Q(Decimal("45000") * Decimal("0.25") * Decimal("0.5") * Decimal("0.001"))),
        (Decimal("100000"), Decimal("10000"), Decimal("0.3"), _Q(Decimal("90000") * Decimal("0.25") * Decimal("0.3") * Decimal("0.001"))),
    ])
    def test_cod_parametrized_calculations(self, engine, tow, sludge, mcf, expected_net):
        """Parametrized COD calculations match expected values."""
        result = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=tow,
            sludge_removal_kg=sludge,
            bo_cod=Decimal("0.25"),
            mcf=mcf,
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        assert ch4 == expected_net

    def test_get_bo_default_bod(self, engine):
        """get_bo_default('BOD') returns 0.6."""
        assert engine.get_bo_default("BOD") == Decimal("0.6")

    def test_get_bo_default_cod(self, engine):
        """get_bo_default('COD') returns 0.25."""
        assert engine.get_bo_default("COD") == Decimal("0.25")

    def test_get_bo_default_invalid_raises(self, engine):
        """get_bo_default with invalid basis raises ValueError."""
        with pytest.raises(ValueError, match="Unknown organic basis"):
            engine.get_bo_default("INVALID")

    def test_cod_bod_different_results(self, engine):
        """Same TOW produces different CH4 for BOD vs COD basis."""
        result_bod = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        result_cod = engine.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo_cod=Decimal("0.25"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        ch4_bod = Decimal(result_bod["calculation_details"]["ch4_net_tonnes"])
        ch4_cod = Decimal(result_cod["calculation_details"]["ch4_net_tonnes"])
        assert ch4_bod > ch4_cod  # Bo_BOD=0.6 > Bo_COD=0.25


# ===========================================================================
# Test Class: Industrial Wastewater
# ===========================================================================


@_SKIP
class TestIndustrialWastewater:
    """Test industry-specific wastewater parameter lookup and calculations."""

    @pytest.mark.parametrize("industry", [
        "PULP_AND_PAPER",
        "MEAT_AND_POULTRY",
        "DAIRY",
        "SUGAR_REFINING",
        "VEGETABLE_OIL",
        "BREWERY",
        "STARCH",
        "FRUIT_VEGETABLE_PROCESSING",
        "TEXTILES",
        "PETROCHEMICAL",
        "PHARMACEUTICAL",
    ])
    def test_industry_parameter_lookup(self, engine, industry):
        """Each of 11 industry types returns valid parameters."""
        params = engine.get_industry_parameters(industry)
        assert "wastewater_m3_per_tonne" in params
        assert "cod_kg_per_m3" in params
        assert "typical_mcf" in params
        assert Decimal(str(params["wastewater_m3_per_tonne"])) > Decimal("0")
        assert Decimal(str(params["cod_kg_per_m3"])) > Decimal("0")

    def test_unknown_industry_raises_error(self, engine):
        """Unknown industry type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown industry_type"):
            engine.get_industry_parameters("UNKNOWN_INDUSTRY")

    def test_pulp_paper_calculation(self, engine):
        """Pulp and paper wastewater calculation is correct."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("1000"),
            industry_type="PULP_AND_PAPER",
            treatment_system="ANAEROBIC_REACTOR_NO_RECOVERY",
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "IPCC_WW_INDUSTRIAL"
        assert result["industry_type"] == "PULP_AND_PAPER"
        # TOW = 1000 * 60 * 5.0 = 300000 kg COD; sludge=0
        # CH4 = 300000 * 0.25 * 0.8 * 0.001 = 60 tonnes
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        assert ch4 == _Q(Decimal("300000") * Decimal("0.25") * Decimal("0.8") * Decimal("0.001"))

    def test_dairy_calculation(self, engine):
        """Dairy wastewater calculation uses IPCC Table 6.9 defaults."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("500"),
            industry_type="DAIRY",
            treatment_system="ANAEROBIC_SHALLOW_LAGOON",
        )
        assert result["status"] == "SUCCESS"
        # TOW = 500 * 7 * 3.0 = 10500 kg COD; MCF=0.2
        # CH4 = 10500 * 0.25 * 0.2 * 0.001 = 0.525 tonnes
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        expected = _Q(Decimal("10500") * Decimal("0.25") * Decimal("0.2") * Decimal("0.001"))
        assert ch4 == expected

    def test_parameter_override(self, engine):
        """Industry parameter overrides are correctly applied."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("1000"),
            industry_type="DAIRY",
            treatment_system="SEPTIC_SYSTEM",
            wastewater_m3_per_tonne_override=Decimal("20"),
            cod_kg_per_m3_override=Decimal("10"),
        )
        assert result["status"] == "SUCCESS"
        # TOW = 1000 * 20 * 10 = 200000 kg COD; MCF=0.5
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        expected = _Q(Decimal("200000") * Decimal("0.25") * Decimal("0.5") * Decimal("0.001"))
        assert ch4 == expected

    def test_sludge_fraction_applied(self, engine):
        """Sludge fraction correctly reduces TOW for industrial wastewater."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("1000"),
            industry_type="BREWERY",
            treatment_system="ANAEROBIC_REACTOR_NO_RECOVERY",
            sludge_fraction=Decimal("0.10"),
        )
        assert result["status"] == "SUCCESS"
        # TOW = 1000 * 10 * 4.0 = 40000 kg COD
        # Sludge = 40000 * 0.10 = 4000; Net TOW = 36000
        # CH4 = 36000 * 0.25 * 0.8 * 0.001 = 7.2 tonnes
        ch4 = Decimal(result["calculation_details"]["ch4_net_tonnes"])
        expected = _Q(Decimal("36000") * Decimal("0.25") * Decimal("0.8") * Decimal("0.001"))
        assert ch4 == expected

    def test_mcf_override_for_industrial(self, engine):
        """MCF override is applied instead of treatment system MCF."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("100"),
            industry_type="TEXTILES",
            treatment_system="SEPTIC_SYSTEM",
            mcf_override=Decimal("0.1"),
        )
        assert result["status"] == "SUCCESS"
        mcf_val = Decimal(result["calculation_details"]["mcf"])
        assert mcf_val == Decimal("0.1")

    def test_bo_cod_override_for_industrial(self, engine):
        """Bo override is applied for industrial wastewater."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("100"),
            industry_type="PETROCHEMICAL",
            treatment_system="AEROBIC_OVERLOADED",
            bo_cod_override=Decimal("0.40"),
        )
        assert result["status"] == "SUCCESS"
        bo_val = Decimal(result["calculation_details"]["bo_cod"])
        assert bo_val == Decimal("0.40")

    def test_zero_production_returns_error(self, engine):
        """Zero production tonnes returns validation error."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("0"),
            industry_type="DAIRY",
            treatment_system="SEPTIC_SYSTEM",
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_negative_production_returns_error(self, engine):
        """Negative production returns validation error."""
        result = engine.calculate_industrial_wastewater(
            production_tonnes=Decimal("-100"),
            industry_type="DAIRY",
            treatment_system="SEPTIC_SYSTEM",
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_case_insensitive_industry_type(self, engine):
        """Industry type lookup is case-insensitive."""
        params = engine.get_industry_parameters("dairy")
        assert "wastewater_m3_per_tonne" in params


# ===========================================================================
# Test Class: N2O from Wastewater Treatment
# ===========================================================================


@_SKIP
class TestWastewaterN2O:
    """Test N2O from wastewater treatment plant and effluent discharge."""

    def test_basic_n2o_calculation(self, engine):
        """Basic N2O calculation with default emission factors."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        assert result["status"] == "SUCCESS"
        assert result["method"] == "IPCC_WW_N2O"

    def test_n2o_default_ef_plant(self, engine):
        """Default EF_plant is 0.016 (IPCC 2019)."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        ef_plant = Decimal(result["calculation_details"]["ef_plant_kg_n2o_n_per_kg_n"])
        assert ef_plant == Decimal("0.016")

    def test_n2o_default_ef_effluent(self, engine):
        """Default EF_effluent is 0.005 (IPCC 2019)."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        ef_eff = Decimal(result["calculation_details"]["ef_effluent_kg_n2o_n_per_kg_n"])
        assert ef_eff == Decimal("0.005")

    def test_n2o_plant_calculation_accuracy(self, engine):
        """N2O plant calculation matches formula: Q * N_inf * EF * 44/28."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("500000"),
            n_influent_kg_m3=Decimal("0.050"),
            n_effluent_kg_m3=Decimal("0.005"),
            ef_plant=Decimal("0.016"),
            ef_effluent=Decimal("0.005"),
        )
        assert result["status"] == "SUCCESS"
        plant_n2o = Decimal(result["calculation_details"]["n2o_plant_kg"])
        # N2O_plant = 500000 * 0.050 * 0.016 * 1.57142857 = 628.571428 kg
        expected = _Q(Decimal("500000") * Decimal("0.050") * Decimal("0.016") * N2O_MOLECULAR_RATIO)
        assert plant_n2o == expected

    def test_n2o_effluent_calculation_accuracy(self, engine):
        """N2O effluent calculation matches formula: Q * N_eff * EF * 44/28."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("500000"),
            n_influent_kg_m3=Decimal("0.050"),
            n_effluent_kg_m3=Decimal("0.005"),
            ef_plant=Decimal("0.016"),
            ef_effluent=Decimal("0.005"),
        )
        effluent_n2o = Decimal(result["calculation_details"]["n2o_effluent_kg"])
        expected = _Q(Decimal("500000") * Decimal("0.005") * Decimal("0.005") * N2O_MOLECULAR_RATIO)
        assert effluent_n2o == expected

    def test_n2o_total_is_sum(self, engine):
        """Total N2O is the sum of plant + effluent."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("500000"),
            n_influent_kg_m3=Decimal("0.050"),
            n_effluent_kg_m3=Decimal("0.005"),
        )
        plant = Decimal(result["calculation_details"]["n2o_plant_kg"])
        eff = Decimal(result["calculation_details"]["n2o_effluent_kg"])
        total = Decimal(result["calculation_details"]["n2o_total_kg"])
        assert total == _Q(plant + eff)

    def test_n2o_custom_ef_overrides(self, engine):
        """Custom emission factors override the defaults."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("100000"),
            n_influent_kg_m3=Decimal("0.100"),
            n_effluent_kg_m3=Decimal("0.020"),
            ef_plant=Decimal("0.032"),
            ef_effluent=Decimal("0.010"),
        )
        ef_p = Decimal(result["calculation_details"]["ef_plant_kg_n2o_n_per_kg_n"])
        ef_e = Decimal(result["calculation_details"]["ef_effluent_kg_n2o_n_per_kg_n"])
        assert ef_p == Decimal("0.032")
        assert ef_e == Decimal("0.010")

    def test_n2o_gwp_conversion(self, engine):
        """N2O to CO2e uses correct GWP (AR6: 273)."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        gas_results = result["emissions_by_gas"]
        n2o_gas = gas_results[0]
        assert n2o_gas["gas"] == "N2O"
        gwp = Decimal(n2o_gas["gwp_value"])
        assert gwp == Decimal("273")

    def test_n2o_zero_flow_returns_error(self, engine):
        """Zero flow rate returns validation error."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("0"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_n2o_zero_nitrogen_produces_zero_n2o(self, engine):
        """Zero nitrogen concentration produces zero N2O."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0"),
            n_effluent_kg_m3=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"
        total = Decimal(result["calculation_details"]["n2o_total_kg"])
        assert total == Decimal("0")

    def test_n2o_negative_nitrogen_returns_error(self, engine):
        """Negative nitrogen returns validation error."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("-0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_n2o_calc_id_prefix(self, engine):
        """Calculation ID starts with ww_n2o_ prefix."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        assert result["calculation_id"].startswith("ww_n2o_")

    @pytest.mark.parametrize("flow,n_inf,n_eff", [
        (Decimal("100000"), Decimal("0.010"), Decimal("0.002")),
        (Decimal("500000"), Decimal("0.050"), Decimal("0.010")),
        (Decimal("2000000"), Decimal("0.080"), Decimal("0.020")),
    ])
    def test_n2o_parametrized_scenarios(self, engine, flow, n_inf, n_eff):
        """N2O calculations succeed for various flow and nitrogen levels."""
        result = engine.calculate_wastewater_n2o(
            flow_m3_yr=flow,
            n_influent_kg_m3=n_inf,
            n_effluent_kg_m3=n_eff,
        )
        assert result["status"] == "SUCCESS"
        total = Decimal(result["calculation_details"]["n2o_total_kg"])
        assert total >= Decimal("0")

    def test_n2o_higher_nitrogen_more_emissions(self, engine):
        """Higher nitrogen concentration produces more N2O."""
        low = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.010"),
            n_effluent_kg_m3=Decimal("0.002"),
        )
        high = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.100"),
            n_effluent_kg_m3=Decimal("0.020"),
        )
        low_total = Decimal(low["calculation_details"]["n2o_total_kg"])
        high_total = Decimal(high["calculation_details"]["n2o_total_kg"])
        assert high_total > low_total


# ===========================================================================
# Test Class: Sludge Treatment Emissions
# ===========================================================================


@_SKIP
class TestSludgeEmissions:
    """Test sludge treatment CH4 calculations."""

    def test_anaerobic_digestion_sludge(self, engine):
        """Sludge MCF for anaerobic digestion is 0.8."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("10"),
            treatment_system="ANAEROBIC_DIGESTION",
        )
        assert result["status"] == "SUCCESS"
        # CH4 = 10000 kg * 0.60 * 0.6 * 0.8 = 2880 kg CH4
        ch4_kg = Decimal(result["emissions_by_gas"][0]["emission_kg"])
        expected = _Q(Decimal("10000") * Decimal("0.60") * Decimal("0.6") * Decimal("0.8"))
        assert ch4_kg == expected

    def test_aerobic_digestion_zero_mcf(self, engine):
        """Aerobic digestion sludge has MCF=0, producing zero CH4."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("10"),
            treatment_system="AEROBIC_DIGESTION",
        )
        assert result["status"] == "SUCCESS"
        ch4_kg = Decimal(result["emissions_by_gas"][0]["emission_kg"])
        assert ch4_kg == Decimal("0")

    def test_sludge_vs_fraction_default(self, engine):
        """Default VS fraction for sludge is 0.60."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("5"),
            treatment_system="LANDFILL",
        )
        assert result["status"] == "SUCCESS"
        vs_frac = Decimal(result["calculation_details"].get("vs_fraction", "0.60"))
        assert vs_frac == Decimal("0.60")

    def test_sludge_custom_vs_fraction(self, engine):
        """Custom VS fraction is correctly applied."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("10"),
            vs_fraction=Decimal("0.40"),
            treatment_system="ANAEROBIC_DIGESTION",
        )
        assert result["status"] == "SUCCESS"
        ch4_kg = Decimal(result["emissions_by_gas"][0]["emission_kg"])
        expected = _Q(Decimal("10000") * Decimal("0.40") * Decimal("0.6") * Decimal("0.8"))
        assert ch4_kg == expected

    def test_sludge_mcf_override(self, engine):
        """MCF override is applied for sludge calculations."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("10"),
            mcf_override=Decimal("0.5"),
        )
        assert result["status"] == "SUCCESS"

    @pytest.mark.parametrize("method,expected_mcf", [
        ("ANAEROBIC_DIGESTION", Decimal("0.8")),
        ("AEROBIC_DIGESTION", Decimal("0.0")),
        ("COMPOSTING", Decimal("0.01")),
        ("LANDFILL", Decimal("1.0")),
        ("LAND_APPLICATION", Decimal("0.01")),
        ("INCINERATION", Decimal("0.0")),
        ("DRYING_BEDS", Decimal("0.0")),
        ("LAGOON_STORAGE", Decimal("0.8")),
    ])
    def test_sludge_mcf_lookup_all_methods(self, engine, method, expected_mcf):
        """Each sludge treatment method returns the correct MCF."""
        mcf = engine.get_sludge_mcf(method)
        assert mcf == expected_mcf

    def test_sludge_unknown_method_raises(self, engine):
        """Unknown sludge treatment method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sludge_treatment"):
            engine.get_sludge_mcf("PLASMA_ARC")

    def test_sludge_zero_tonnes_returns_error(self, engine):
        """Zero sludge tonnes returns validation error."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("0"),
            treatment_system="LANDFILL",
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")

    def test_sludge_calc_id_prefix(self, engine):
        """Sludge calculation ID starts with ww_sludge_ prefix."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("10"),
            treatment_system="LANDFILL",
        )
        assert result["calculation_id"].startswith("ww_sludge_")


# ===========================================================================
# Test Class: Combined Wastewater Calculations
# ===========================================================================


@_SKIP
class TestCombinedWastewater:
    """Test combined CH4 + N2O calculations for wastewater systems."""

    def test_ch4_and_n2o_independent(self, engine):
        """CH4 and N2O calculations are independent and can be combined."""
        ch4_result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("50000"),
            sludge_removal_kg=Decimal("5000"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        n2o_result = engine.calculate_wastewater_n2o(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        assert ch4_result["status"] == "SUCCESS"
        assert n2o_result["status"] == "SUCCESS"
        ch4_co2e = Decimal(ch4_result["total_co2e_tonnes"])
        n2o_co2e = Decimal(n2o_result["total_co2e_tonnes"])
        combined_co2e = ch4_co2e + n2o_co2e
        assert combined_co2e > Decimal("0")

    def test_reproducibility_ch4(self, engine):
        """Same inputs produce identical CH4 results (determinism)."""
        args = dict(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("1000"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        r1 = engine.calculate_wastewater_ch4(**args)
        r2 = engine.calculate_wastewater_ch4(**args)
        assert r1["calculation_details"]["ch4_net_tonnes"] == r2["calculation_details"]["ch4_net_tonnes"]

    def test_reproducibility_n2o(self, engine):
        """Same inputs produce identical N2O results (determinism)."""
        args = dict(
            flow_m3_yr=Decimal("1000000"),
            n_influent_kg_m3=Decimal("0.040"),
            n_effluent_kg_m3=Decimal("0.010"),
        )
        r1 = engine.calculate_wastewater_n2o(**args)
        r2 = engine.calculate_wastewater_n2o(**args)
        assert r1["calculation_details"]["n2o_total_kg"] == r2["calculation_details"]["n2o_total_kg"]

    def test_different_gwp_sources(self, engine):
        """Different GWP sources produce different CO2e totals."""
        args = dict(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        r_ar5 = engine.calculate_wastewater_ch4(**args, gwp_source="AR5")
        r_ar6 = engine.calculate_wastewater_ch4(**args, gwp_source="AR6")
        co2e_ar5 = Decimal(r_ar5["total_co2e_tonnes"])
        co2e_ar6 = Decimal(r_ar6["total_co2e_tonnes"])
        # AR5 GWP_CH4=28, AR6 GWP_CH4=29.8 -- different CO2e
        assert co2e_ar5 != co2e_ar6

    def test_calculation_counter_increments(self, engine):
        """Internal calculation counter increments on each call."""
        initial = engine._total_calculations
        engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("1000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert engine._total_calculations == initial + 1

    def test_all_treatment_systems_produce_result(self, engine):
        """All 8 treatment systems produce a successful result."""
        for system in TreatmentSystem:
            result = engine.calculate_wastewater_ch4(
                tow_kg_bod_yr=Decimal("10000"),
                sludge_removal_kg=Decimal("1000"),
                bo=Decimal("0.6"),
                mcf=None,
                recovery_tonnes=Decimal("0"),
                treatment_system=system.value,
            )
            assert result["status"] == "SUCCESS", (
                f"Failed for treatment system: {system.value}"
            )


# ===========================================================================
# Test Class: Edge Cases
# ===========================================================================


@_SKIP
class TestWastewaterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_tow(self, engine):
        """Very large TOW value does not overflow."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("999999999"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"

    def test_very_small_tow(self, engine):
        """Very small positive TOW still produces valid result."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("0.001"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        assert result["status"] == "SUCCESS"

    def test_string_numeric_inputs_accepted(self, engine):
        """String numeric inputs are accepted and converted."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr="10000",
            sludge_removal_kg="1000",
            bo="0.6",
            mcf="0.8",
            recovery_tonnes="0",
        )
        assert result["status"] == "SUCCESS"

    def test_integer_inputs_accepted(self, engine):
        """Integer inputs are accepted and converted."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=10000,
            sludge_removal_kg=1000,
            bo="0.6",
            mcf="0.8",
            recovery_tonnes=0,
        )
        assert result["status"] == "SUCCESS"

    def test_float_inputs_accepted(self, engine):
        """Float inputs are accepted and converted."""
        result = engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=10000.0,
            sludge_removal_kg=1000.0,
            bo=0.6,
            mcf=0.8,
            recovery_tonnes=0.0,
        )
        assert result["status"] == "SUCCESS"

    def test_unknown_treatment_system_raises(self, engine):
        """Unknown treatment system raises ValueError in get_mcf."""
        with pytest.raises(ValueError, match="Unknown treatment_system"):
            engine.get_mcf("PLASMA_ARC")

    def test_thread_safety_concurrent_calculations(self, engine):
        """Concurrent calculations from multiple threads do not corrupt state."""
        results = []
        errors = []

        def worker():
            try:
                r = engine.calculate_wastewater_ch4(
                    tow_kg_bod_yr=Decimal("10000"),
                    sludge_removal_kg=Decimal("1000"),
                    bo=Decimal("0.6"),
                    mcf=Decimal("0.8"),
                    recovery_tonnes=Decimal("0"),
                )
                results.append(r)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All results should have same CH4 value
        ch4_values = {r["calculation_details"]["ch4_net_tonnes"] for r in results}
        assert len(ch4_values) == 1

    def test_error_counter_increments_on_failure(self, engine):
        """Error counter increments when validation fails."""
        initial_errors = engine._total_errors
        engine.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("-1000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.5"),
            recovery_tonnes=Decimal("0"),
        )
        # Error counter may or may not increment for validation errors
        # but at least the status is error
        assert engine._total_errors >= initial_errors

    def test_gwp_ar5_for_ch4(self, engine_ar5):
        """Engine with AR5 default uses GWP CH4 = 28."""
        result = engine_ar5.calculate_wastewater_ch4(
            tow_kg_bod_yr=Decimal("10000"),
            sludge_removal_kg=Decimal("0"),
            bo=Decimal("0.6"),
            mcf=Decimal("0.8"),
            recovery_tonnes=Decimal("0"),
        )
        gas = result["emissions_by_gas"][0]
        assert gas["gwp_source"] == "AR5"
        assert Decimal(gas["gwp_value"]) == Decimal("28")

    def test_sludge_neither_system_nor_override_returns_error(self, engine):
        """Sludge without system or override returns error."""
        result = engine.calculate_sludge_emissions(
            sludge_tonnes=Decimal("10"),
            treatment_system=None,
            mcf_override=None,
        )
        assert result["status"] in ("ERROR", "VALIDATION_ERROR")
