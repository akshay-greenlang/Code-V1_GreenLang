# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-009 SteamHeatCoolingEngine.

Tests steam emission calculations, heating emission calculations, cooling
emission calculations, boiler efficiency adjustments, CHP allocation,
unit conversions, and input validation.

Target: 40+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

try:
    from greenlang.agents.mrv.scope2_location.steam_heat_cooling import (
        SteamHeatCoolingEngine,
        SteamType,
        HeatingType,
        CoolingType,
        EnergyType,
        CHPMethod,
        CalculationStatus,
        _DEFAULT_STEAM_EFS,
        _DEFAULT_HEATING_EFS,
        _DEFAULT_COOLING_EFS,
        _MMBTU_TO_GJ,
        _THERM_TO_GJ,
        _KWH_TO_GJ,
        _MWH_TO_GJ,
        _GJ_TO_MWH,
        _KG_TO_TONNES,
        _D,
        _quantize,
        _quantize_3,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a default SteamHeatCoolingEngine."""
    return SteamHeatCoolingEngine()


@pytest.fixture
def engine_biogenic_disabled():
    """Create engine with biogenic tracking disabled."""
    return SteamHeatCoolingEngine(
        config={"enable_biogenic_tracking": False}
    )


@pytest.fixture
def engine_with_custom_efs():
    """Create engine with custom emission factor overrides."""
    return SteamHeatCoolingEngine(
        config={
            "custom_steam_efs": {"geothermal": "12.50"},
            "custom_heating_efs": {"solar_thermal": "0.00"},
            "custom_cooling_efs": {"ice_storage": "15.00"},
        }
    )


# ===========================================================================
# TestSteamEmissions
# ===========================================================================


@_SKIP
class TestSteamEmissions:
    """Tests for calculate_steam_emissions."""

    def test_natural_gas_steam_500_gj(self, engine):
        """500 GJ x 56.10 kgCO2e/GJ = 28,050 kgCO2e."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("500"),
            steam_type="natural_gas",
        )
        assert result["status"] == "SUCCESS"
        assert Decimal(result["total_co2e_kg"]) == _quantize(
            Decimal("500") * Decimal("56.10")
        )

    def test_coal_steam_1000_gj(self, engine):
        """1000 GJ x 94.60 kgCO2e/GJ = 94,600 kgCO2e."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("1000"),
            steam_type="coal",
        )
        assert result["status"] == "SUCCESS"
        expected_kg = _quantize(Decimal("1000") * Decimal("94.60"))
        assert Decimal(result["total_co2e_kg"]) == expected_kg

    def test_biomass_steam_is_biogenic(self, engine):
        """Biomass steam emissions are flagged as biogenic."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("100"),
            steam_type="biomass",
        )
        assert result["status"] == "SUCCESS"
        assert result["is_biogenic"] is True
        # Biogenic means Scope 2 total is 0
        assert Decimal(result["total_co2e_kg"]) == Decimal("0")
        # Biogenic CO2 is tracked separately
        assert Decimal(result["biogenic_co2_kg"]) == Decimal("0")

    def test_biomass_biogenic_disabled(self, engine_biogenic_disabled):
        """With biogenic disabled, biomass emissions stay in Scope 2."""
        result = engine_biogenic_disabled.calculate_steam_emissions(
            consumption_gj=Decimal("100"),
            steam_type="biomass",
        )
        assert result["status"] == "SUCCESS"
        # biomass EF is 0.00, so total is 0 regardless
        assert Decimal(result["total_co2e_kg"]) == Decimal("0")

    def test_oil_steam(self, engine):
        """Oil steam uses 73.30 kgCO2e/GJ factor."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("200"),
            steam_type="oil",
        )
        assert result["status"] == "SUCCESS"
        expected = _quantize(Decimal("200") * Decimal("73.30"))
        assert Decimal(result["total_co2e_kg"]) == expected

    def test_custom_ef_overrides_default(self, engine):
        """Custom EF overrides the default steam EF."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("100"),
            steam_type="natural_gas",
            custom_ef=Decimal("70.00"),
        )
        assert result["status"] == "SUCCESS"
        assert Decimal(result["ef_applied"]) == Decimal("70.00")
        expected = _quantize(Decimal("100") * Decimal("70.00"))
        assert Decimal(result["total_co2e_kg"]) == expected

    def test_steam_result_contains_provenance(self, engine):
        """Steam result includes provenance_hash."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("100"),
            steam_type="natural_gas",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_steam_result_has_calculation_trace(self, engine):
        """Steam result includes non-empty calculation trace."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("100"),
            steam_type="natural_gas",
        )
        assert len(result["calculation_trace"]) > 0

    def test_zero_consumption_returns_zero(self, engine):
        """Zero consumption produces zero emissions."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("0"),
            steam_type="natural_gas",
        )
        assert result["status"] == "SUCCESS"
        assert Decimal(result["total_co2e_kg"]) == Decimal("0")

    def test_steam_tonnes_conversion(self, engine):
        """Total CO2e tonnes = kg / 1000."""
        result = engine.calculate_steam_emissions(
            consumption_gj=Decimal("500"),
            steam_type="natural_gas",
        )
        kg = Decimal(result["total_co2e_kg"])
        tonnes = Decimal(result["total_co2e_tonnes"])
        assert tonnes == _quantize_3(kg * _KG_TO_TONNES)


# ===========================================================================
# TestHeatingEmissions
# ===========================================================================


@_SKIP
class TestHeatingEmissions:
    """Tests for calculate_heating_emissions."""

    def test_district_heating_300_gj(self, engine):
        """300 GJ x 43.50 kgCO2e/GJ = 13,050 kgCO2e."""
        result = engine.calculate_heating_emissions(
            consumption_gj=Decimal("300"),
            heating_type="district",
        )
        assert result["status"] == "SUCCESS"
        expected = _quantize(Decimal("300") * Decimal("43.50"))
        assert Decimal(result["total_co2e_kg"]) == expected

    def test_gas_boiler_heating(self, engine):
        """Gas boiler uses 56.10 kgCO2e/GJ factor."""
        result = engine.calculate_heating_emissions(
            consumption_gj=Decimal("200"),
            heating_type="gas_boiler",
        )
        assert result["status"] == "SUCCESS"
        expected = _quantize(Decimal("200") * Decimal("56.10"))
        assert Decimal(result["total_co2e_kg"]) == expected

    def test_biomass_heating_is_biogenic(self, engine):
        """Biomass heating is flagged as biogenic (Scope 2 total = 0)."""
        result = engine.calculate_heating_emissions(
            consumption_gj=Decimal("100"),
            heating_type="biomass",
        )
        assert result["is_biogenic"] is True
        assert Decimal(result["total_co2e_kg"]) == Decimal("0")

    def test_heat_pump_low_ef(self, engine):
        """Heat pump uses 18.50 kgCO2e/GJ factor."""
        result = engine.calculate_heating_emissions(
            consumption_gj=Decimal("100"),
            heating_type="heat_pump",
        )
        expected = _quantize(Decimal("100") * Decimal("18.50"))
        assert Decimal(result["total_co2e_kg"]) == expected

    def test_heating_with_custom_ef(self, engine):
        """Custom EF overrides default heating factor."""
        result = engine.calculate_heating_emissions(
            consumption_gj=Decimal("100"),
            heating_type="district",
            custom_ef=Decimal("50.00"),
        )
        assert Decimal(result["ef_applied"]) == Decimal("50.00")


# ===========================================================================
# TestCoolingEmissions
# ===========================================================================


@_SKIP
class TestCoolingEmissions:
    """Tests for calculate_cooling_emissions."""

    def test_absorption_cooling_200_gj(self, engine):
        """200 GJ x 32.10 kgCO2e/GJ = 6,420 kgCO2e."""
        result = engine.calculate_cooling_emissions(
            consumption_gj=Decimal("200"),
            cooling_type="absorption",
        )
        assert result["status"] == "SUCCESS"
        expected = _quantize(Decimal("200") * Decimal("32.10"))
        assert Decimal(result["total_co2e_kg"]) == expected

    def test_district_cooling(self, engine):
        """District cooling uses 28.50 kgCO2e/GJ factor."""
        result = engine.calculate_cooling_emissions(
            consumption_gj=Decimal("100"),
            cooling_type="district",
        )
        expected = _quantize(Decimal("100") * Decimal("28.50"))
        assert Decimal(result["total_co2e_kg"]) == expected

    def test_free_cooling_zero(self, engine):
        """Free cooling has zero emission factor."""
        result = engine.calculate_cooling_emissions(
            consumption_gj=Decimal("500"),
            cooling_type="free_cooling",
        )
        assert Decimal(result["total_co2e_kg"]) == Decimal("0")

    def test_cooling_with_custom_ef(self, engine):
        """Custom EF overrides default cooling factor."""
        result = engine.calculate_cooling_emissions(
            consumption_gj=Decimal("100"),
            cooling_type="absorption",
            custom_ef=Decimal("40.00"),
        )
        assert Decimal(result["ef_applied"]) == Decimal("40.00")
        expected = _quantize(Decimal("100") * Decimal("40.00"))
        assert Decimal(result["total_co2e_kg"]) == expected


# ===========================================================================
# TestEfficiency
# ===========================================================================


@_SKIP
class TestEfficiency:
    """Tests for calculate_steam_with_efficiency."""

    def test_85_percent_efficiency_adjustment(self, engine):
        """Adjusted EF = 56.10 / 0.85 = 66.00 kgCO2e/GJ."""
        result = engine.calculate_steam_with_efficiency(
            consumption_gj=Decimal("100"),
            boiler_efficiency=Decimal("0.85"),
            steam_type="natural_gas",
        )
        assert result["status"] == "SUCCESS"
        ef_adjusted = _quantize(Decimal("56.10") / Decimal("0.85"))
        assert Decimal(result["ef_adjusted"]) == ef_adjusted

    def test_efficiency_1_no_adjustment(self, engine):
        """Efficiency of 1.0 produces same EF as default."""
        result = engine.calculate_steam_with_efficiency(
            consumption_gj=Decimal("100"),
            boiler_efficiency=Decimal("1.00"),
            steam_type="natural_gas",
        )
        assert result["status"] == "SUCCESS"
        assert Decimal(result["ef_adjusted"]) == _quantize(Decimal("56.10"))

    def test_low_efficiency_below_min_errors(self, engine):
        """Boiler efficiency below 0.50 returns error."""
        result = engine.calculate_steam_with_efficiency(
            consumption_gj=Decimal("100"),
            boiler_efficiency=Decimal("0.30"),
            steam_type="natural_gas",
        )
        assert result["status"] == "ERROR"

    def test_efficiency_includes_boiler_field(self, engine):
        """Result includes boiler_efficiency field."""
        result = engine.calculate_steam_with_efficiency(
            consumption_gj=Decimal("100"),
            boiler_efficiency=Decimal("0.90"),
            steam_type="natural_gas",
        )
        assert result["boiler_efficiency"] == str(Decimal("0.90"))


# ===========================================================================
# TestCHP
# ===========================================================================


@_SKIP
class TestCHP:
    """Tests for CHP emission allocation."""

    def test_chp_efficiency_method_exists(self, engine):
        """Engine has allocate_chp_emissions method."""
        assert hasattr(engine, "allocate_chp_emissions") or hasattr(
            engine, "calculate_chp_allocation"
        )

    def test_chp_method_enum_values(self):
        """CHPMethod enum has expected allocation methods."""
        assert CHPMethod.EFFICIENCY_METHOD.value == "efficiency_method"
        assert CHPMethod.ENERGY_METHOD.value == "energy_method"
        assert CHPMethod.EXERGY_METHOD.value == "exergy_method"


# ===========================================================================
# TestUnitConversions
# ===========================================================================


@_SKIP
class TestUnitConversions:
    """Tests for mmbtu_to_gj, therms_to_gj, normalize_to_gj."""

    def test_mmbtu_to_gj_1(self, engine):
        """1 MMBtu = 1.05506 GJ."""
        result = engine.mmbtu_to_gj(Decimal("1"))
        assert result == _quantize(Decimal("1") * _MMBTU_TO_GJ)

    def test_mmbtu_to_gj_10(self, engine):
        """10 MMBtu = 10.5506 GJ."""
        result = engine.mmbtu_to_gj(Decimal("10"))
        expected = _quantize(Decimal("10") * _MMBTU_TO_GJ)
        assert result == expected

    def test_mmbtu_negative_raises(self, engine):
        """Negative MMBtu raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.mmbtu_to_gj(Decimal("-1"))

    def test_therms_to_gj_1(self, engine):
        """1 therm = 0.105506 GJ."""
        result = engine.therms_to_gj(Decimal("1"))
        assert result == _quantize(Decimal("1") * _THERM_TO_GJ)

    def test_therms_to_gj_100(self, engine):
        """100 therms = 10.5506 GJ."""
        result = engine.therms_to_gj(Decimal("100"))
        expected = _quantize(Decimal("100") * _THERM_TO_GJ)
        assert result == expected

    def test_therms_negative_raises(self, engine):
        """Negative therms raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.therms_to_gj(Decimal("-5"))

    def test_kwh_to_gj(self, engine):
        """1000 kWh = 3.6 GJ."""
        result = engine.kwh_to_gj(Decimal("1000"))
        expected = _quantize(Decimal("1000") * _KWH_TO_GJ)
        assert result == expected

    def test_normalize_gj_identity(self, engine):
        """normalize_to_gj with 'GJ' returns the same value."""
        result = engine.normalize_to_gj(Decimal("42.5"), "GJ")
        assert result == _quantize(Decimal("42.5"))

    def test_normalize_mwh_to_gj(self, engine):
        """normalize_to_gj with 'MWh' converts correctly."""
        result = engine.normalize_to_gj(Decimal("1"), "MWh")
        expected = _quantize(Decimal("1") * _MWH_TO_GJ)
        assert result == expected

    def test_normalize_unsupported_unit_raises(self, engine):
        """Unsupported unit raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported energy unit"):
            engine.normalize_to_gj(Decimal("100"), "barrels")


# ===========================================================================
# TestValidation
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for validate_steam_input, validate_cop."""

    def test_valid_steam_input_no_errors(self, engine):
        """Valid steam input returns empty error list."""
        errors = engine.validate_steam_input(
            consumption_gj=Decimal("100"),
            steam_type="natural_gas",
        )
        assert errors == []

    def test_negative_consumption_error(self, engine):
        """Negative consumption is flagged."""
        errors = engine.validate_steam_input(
            consumption_gj=Decimal("-10"),
            steam_type="natural_gas",
        )
        assert any("must be >= 0" in e for e in errors)

    def test_unknown_steam_type_error(self, engine):
        """Unknown steam type is flagged."""
        errors = engine.validate_steam_input(
            consumption_gj=Decimal("100"),
            steam_type="plutonium",
        )
        assert any("Unknown" in e or "unknown" in e.lower() for e in errors)

    def test_validate_cop_in_range(self, engine):
        """COP within range produces no errors."""
        errors = engine.validate_cop(
            cop=Decimal("3.5"),
            system_type="electric_chiller",
        )
        assert errors == []

    def test_validate_cop_out_of_range(self, engine):
        """COP outside range is flagged."""
        errors = engine.validate_cop(
            cop=Decimal("15.0"),
            system_type="electric_chiller",
        )
        assert any("outside" in e.lower() for e in errors)

    def test_validate_cop_zero_error(self, engine):
        """COP of zero is flagged."""
        errors = engine.validate_cop(
            cop=Decimal("0"),
            system_type="heat_pump",
        )
        assert any("must be > 0" in e for e in errors)

    def test_custom_ef_steam_type_registered(self, engine_with_custom_efs):
        """Custom EF types are recognized in validation."""
        errors = engine_with_custom_efs.validate_steam_input(
            consumption_gj=Decimal("100"),
            steam_type="geothermal",
        )
        assert errors == []

    def test_list_available_efs(self, engine):
        """list_available_efs returns steam, heating, and cooling dicts."""
        result = engine.list_available_efs()
        assert "steam" in result
        assert "heating" in result
        assert "cooling" in result
        assert "natural_gas" in result["steam"]
        assert "district" in result["heating"]
        assert "absorption" in result["cooling"]

    def test_get_steam_ef_valid(self, engine):
        """get_steam_ef returns correct factor for natural_gas."""
        ef = engine.get_steam_ef("natural_gas")
        assert ef == Decimal("56.10")

    def test_get_steam_ef_invalid_raises(self, engine):
        """get_steam_ef raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown steam type"):
            engine.get_steam_ef("plutonium")

    def test_get_heating_ef_valid(self, engine):
        """get_heating_ef returns correct factor for district."""
        ef = engine.get_heating_ef("district")
        assert ef == Decimal("43.50")

    def test_get_cooling_ef_valid(self, engine):
        """get_cooling_ef returns correct factor for absorption."""
        ef = engine.get_cooling_ef("absorption")
        assert ef == Decimal("32.10")
