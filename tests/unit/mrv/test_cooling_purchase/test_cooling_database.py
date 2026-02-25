"""
test_cooling_database.py - Tests for CoolingDatabaseEngine

Tests database lookups and reference data for AGENT-MRV-012 (Cooling Purchase Agent).
Validates technology specifications, emission factors, and unit conversions.

Test Coverage:
- Singleton pattern
- 18 cooling technologies
- COP, IPLV, energy source lookups
- Technology classification (electric/absorption/free/TES)
- 12 district cooling regions
- 11 heat sources
- 11 refrigerants (AR5/AR6 GWP)
- Efficiency and unit conversions
- Custom technology management
"""

import pytest
from decimal import Decimal
from typing import Dict, Tuple, Optional

try:
    from greenlang.cooling_purchase.database import CoolingDatabaseEngine
except ImportError:
    pytest.skip("cooling_purchase not available", allow_module_level=True)


@pytest.fixture
def db():
    """Fresh database engine instance for each test."""
    engine = CoolingDatabaseEngine()
    engine.reset()
    return engine


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, db):
        """Test singleton returns same instance."""
        db1 = CoolingDatabaseEngine()
        db2 = CoolingDatabaseEngine()
        assert db1 is db2

    def test_reset_clears_custom_data(self, db):
        """Test reset clears custom technologies and factors."""
        db.add_custom_technology(
            "custom_chiller", {"cop": Decimal("7.0"), "type": "electric"}
        )
        db.reset()

        with pytest.raises((KeyError, ValueError)):
            db.get_technology_spec("custom_chiller")


class TestTechnologySpecifications:
    """Test get_technology_spec() for all 18 technologies."""

    # Electric chillers (6)
    def test_get_technology_spec_air_cooled_screw(self, db):
        """Test air_cooled_screw technology spec."""
        spec = db.get_technology_spec("air_cooled_screw")
        assert isinstance(spec, dict)
        assert "cop" in spec
        assert "iplv" in spec
        assert spec["type"] == "electric"

    def test_get_technology_spec_water_cooled_screw(self, db):
        """Test water_cooled_screw technology spec."""
        spec = db.get_technology_spec("water_cooled_screw")
        assert spec["type"] == "electric"
        assert "cop" in spec

    def test_get_technology_spec_air_cooled_centrifugal(self, db):
        """Test air_cooled_centrifugal technology spec."""
        spec = db.get_technology_spec("air_cooled_centrifugal")
        assert spec["type"] == "electric"

    def test_get_technology_spec_water_cooled_centrifugal(self, db):
        """Test water_cooled_centrifugal technology spec."""
        spec = db.get_technology_spec("water_cooled_centrifugal")
        assert spec["type"] == "electric"

    def test_get_technology_spec_magnetic_bearing(self, db):
        """Test magnetic_bearing_centrifugal technology spec."""
        spec = db.get_technology_spec("magnetic_bearing_centrifugal")
        assert spec["type"] == "electric"

    def test_get_technology_spec_scroll_chiller(self, db):
        """Test scroll_chiller technology spec."""
        spec = db.get_technology_spec("scroll_chiller")
        assert spec["type"] == "electric"

    # Absorption chillers (4)
    def test_get_technology_spec_single_effect_libr(self, db):
        """Test single_effect_libr technology spec."""
        spec = db.get_technology_spec("single_effect_libr")
        assert spec["type"] == "absorption"
        assert "cop" in spec

    def test_get_technology_spec_double_effect_libr(self, db):
        """Test double_effect_libr technology spec."""
        spec = db.get_technology_spec("double_effect_libr")
        assert spec["type"] == "absorption"

    def test_get_technology_spec_triple_effect_libr(self, db):
        """Test triple_effect_libr technology spec."""
        spec = db.get_technology_spec("triple_effect_libr")
        assert spec["type"] == "absorption"

    def test_get_technology_spec_ammonia_absorption(self, db):
        """Test ammonia_absorption technology spec."""
        spec = db.get_technology_spec("ammonia_absorption")
        assert spec["type"] == "absorption"

    # Free cooling (4)
    def test_get_technology_spec_waterside_economizer(self, db):
        """Test waterside_economizer technology spec."""
        spec = db.get_technology_spec("waterside_economizer")
        assert spec["type"] == "free_cooling"

    def test_get_technology_spec_airside_economizer(self, db):
        """Test airside_economizer technology spec."""
        spec = db.get_technology_spec("airside_economizer")
        assert spec["type"] == "free_cooling"

    def test_get_technology_spec_cooling_tower(self, db):
        """Test cooling_tower technology spec."""
        spec = db.get_technology_spec("cooling_tower")
        assert spec["type"] == "free_cooling"

    def test_get_technology_spec_dry_cooler(self, db):
        """Test dry_cooler technology spec."""
        spec = db.get_technology_spec("dry_cooler")
        assert spec["type"] == "free_cooling"

    # Thermal Energy Storage (3)
    def test_get_technology_spec_ice_storage(self, db):
        """Test ice_storage technology spec."""
        spec = db.get_technology_spec("ice_storage")
        assert spec["type"] == "tes"

    def test_get_technology_spec_chilled_water_storage(self, db):
        """Test chilled_water_storage technology spec."""
        spec = db.get_technology_spec("chilled_water_storage")
        assert spec["type"] == "tes"

    def test_get_technology_spec_phase_change_material(self, db):
        """Test phase_change_material technology spec."""
        spec = db.get_technology_spec("phase_change_material")
        assert spec["type"] == "tes"

    # District cooling (1)
    def test_get_technology_spec_district_cooling(self, db):
        """Test district_cooling technology spec."""
        spec = db.get_technology_spec("district_cooling")
        assert spec["type"] == "district"

    def test_invalid_technology_raises_error(self, db):
        """Test invalid technology raises error."""
        with pytest.raises((KeyError, ValueError)):
            db.get_technology_spec("invalid_tech")


class TestCOPLookups:
    """Test COP (Coefficient of Performance) lookups."""

    def test_get_default_cop_returns_decimal(self, db):
        """Test get_default_cop returns Decimal."""
        cop = db.get_default_cop("air_cooled_screw")
        assert isinstance(cop, Decimal)
        assert cop > 0

    def test_get_default_cop_electric_chiller(self, db):
        """Test get_default_cop for electric chiller."""
        cop = db.get_default_cop("water_cooled_centrifugal")
        assert cop >= Decimal("5.0")  # Reasonable range

    def test_get_default_cop_absorption(self, db):
        """Test get_default_cop for absorption chiller."""
        cop = db.get_default_cop("single_effect_libr")
        assert Decimal("0.6") <= cop <= Decimal("0.8")

    def test_get_cop_range_returns_tuple(self, db):
        """Test get_cop_range returns (min, max) tuple."""
        cop_range = db.get_cop_range("air_cooled_screw")
        assert isinstance(cop_range, tuple)
        assert len(cop_range) == 2
        assert cop_range[0] < cop_range[1]

    def test_get_cop_range_electric(self, db):
        """Test get_cop_range for electric chiller."""
        min_cop, max_cop = db.get_cop_range("magnetic_bearing_centrifugal")
        assert isinstance(min_cop, Decimal)
        assert isinstance(max_cop, Decimal)
        assert min_cop > 0

    def test_get_cop_range_absorption(self, db):
        """Test get_cop_range for absorption chiller."""
        min_cop, max_cop = db.get_cop_range("double_effect_libr")
        assert min_cop < max_cop


class TestIPLVLookups:
    """Test IPLV (Integrated Part Load Value) lookups."""

    def test_get_iplv_returns_decimal(self, db):
        """Test get_iplv returns Decimal for electric chillers."""
        iplv = db.get_iplv("air_cooled_screw")
        assert isinstance(iplv, Decimal)
        assert iplv > 0

    def test_get_iplv_electric_chiller(self, db):
        """Test get_iplv for electric chiller."""
        iplv = db.get_iplv("water_cooled_screw")
        cop = db.get_default_cop("water_cooled_screw")
        # IPLV typically 10-20% higher than COP
        assert iplv > cop

    def test_get_iplv_free_cooling_none(self, db):
        """Test get_iplv returns None for free cooling."""
        iplv = db.get_iplv("waterside_economizer")
        assert iplv is None

    def test_get_iplv_absorption_none(self, db):
        """Test get_iplv returns None for absorption."""
        iplv = db.get_iplv("single_effect_libr")
        assert iplv is None

    def test_get_iplv_all_electric_technologies(self, db):
        """Test get_iplv for all electric technologies."""
        electric_techs = [
            "air_cooled_screw",
            "water_cooled_screw",
            "air_cooled_centrifugal",
            "water_cooled_centrifugal",
            "magnetic_bearing_centrifugal",
            "scroll_chiller",
        ]
        for tech in electric_techs:
            iplv = db.get_iplv(tech)
            assert iplv is not None
            assert isinstance(iplv, Decimal)


class TestEnergySourceLookups:
    """Test energy source lookups."""

    def test_get_energy_source_electric(self, db):
        """Test get_energy_source for electric chiller."""
        source = db.get_energy_source("air_cooled_screw")
        assert source == "electricity"

    def test_get_energy_source_absorption(self, db):
        """Test get_energy_source for absorption chiller."""
        source = db.get_energy_source("single_effect_libr")
        assert source in ["heat", "thermal"]

    def test_get_energy_source_free_cooling(self, db):
        """Test get_energy_source for free cooling."""
        source = db.get_energy_source("waterside_economizer")
        assert source in ["ambient", "electricity"]  # Fan power still needs electricity


class TestTechnologyClassification:
    """Test technology classification methods."""

    def test_is_electric_technology_air_cooled_screw(self, db):
        """Test is_electric_technology for air_cooled_screw."""
        assert db.is_electric_technology("air_cooled_screw") is True

    def test_is_electric_technology_water_cooled_screw(self, db):
        """Test is_electric_technology for water_cooled_screw."""
        assert db.is_electric_technology("water_cooled_screw") is True

    def test_is_electric_technology_centrifugal(self, db):
        """Test is_electric_technology for centrifugal chillers."""
        assert db.is_electric_technology("air_cooled_centrifugal") is True
        assert db.is_electric_technology("water_cooled_centrifugal") is True

    def test_is_electric_technology_magnetic_bearing(self, db):
        """Test is_electric_technology for magnetic_bearing."""
        assert db.is_electric_technology("magnetic_bearing_centrifugal") is True

    def test_is_electric_technology_scroll(self, db):
        """Test is_electric_technology for scroll_chiller."""
        assert db.is_electric_technology("scroll_chiller") is True

    def test_is_electric_technology_absorption_false(self, db):
        """Test is_electric_technology returns False for absorption."""
        assert db.is_electric_technology("single_effect_libr") is False

    def test_is_absorption_technology_single_effect(self, db):
        """Test is_absorption_technology for single_effect_libr."""
        assert db.is_absorption_technology("single_effect_libr") is True

    def test_is_absorption_technology_double_effect(self, db):
        """Test is_absorption_technology for double_effect_libr."""
        assert db.is_absorption_technology("double_effect_libr") is True

    def test_is_absorption_technology_triple_effect(self, db):
        """Test is_absorption_technology for triple_effect_libr."""
        assert db.is_absorption_technology("triple_effect_libr") is True

    def test_is_absorption_technology_ammonia(self, db):
        """Test is_absorption_technology for ammonia_absorption."""
        assert db.is_absorption_technology("ammonia_absorption") is True

    def test_is_absorption_technology_electric_false(self, db):
        """Test is_absorption_technology returns False for electric."""
        assert db.is_absorption_technology("air_cooled_screw") is False

    def test_is_free_cooling_technology_waterside(self, db):
        """Test is_free_cooling_technology for waterside_economizer."""
        assert db.is_free_cooling_technology("waterside_economizer") is True

    def test_is_free_cooling_technology_airside(self, db):
        """Test is_free_cooling_technology for airside_economizer."""
        assert db.is_free_cooling_technology("airside_economizer") is True

    def test_is_free_cooling_technology_cooling_tower(self, db):
        """Test is_free_cooling_technology for cooling_tower."""
        assert db.is_free_cooling_technology("cooling_tower") is True

    def test_is_free_cooling_technology_dry_cooler(self, db):
        """Test is_free_cooling_technology for dry_cooler."""
        assert db.is_free_cooling_technology("dry_cooler") is True

    def test_is_tes_technology_ice_storage(self, db):
        """Test is_tes_technology for ice_storage."""
        assert db.is_tes_technology("ice_storage") is True

    def test_is_tes_technology_chilled_water(self, db):
        """Test is_tes_technology for chilled_water_storage."""
        assert db.is_tes_technology("chilled_water_storage") is True

    def test_is_tes_technology_pcm(self, db):
        """Test is_tes_technology for phase_change_material."""
        assert db.is_tes_technology("phase_change_material") is True


class TestDistrictCoolingFactors:
    """Test district cooling emission factors for 12 regions."""

    def test_get_district_cooling_factor_north_america(self, db):
        """Test get_district_cooling_factor for North America."""
        factor = db.get_district_cooling_factor("north_america")
        assert isinstance(factor, Decimal)
        assert factor > 0

    def test_get_district_cooling_factor_europe_west(self, db):
        """Test get_district_cooling_factor for Western Europe."""
        factor = db.get_district_cooling_factor("europe_west")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_europe_north(self, db):
        """Test get_district_cooling_factor for Northern Europe."""
        factor = db.get_district_cooling_factor("europe_north")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_asia_pacific(self, db):
        """Test get_district_cooling_factor for Asia Pacific."""
        factor = db.get_district_cooling_factor("asia_pacific")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_middle_east(self, db):
        """Test get_district_cooling_factor for Middle East."""
        factor = db.get_district_cooling_factor("middle_east")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_china(self, db):
        """Test get_district_cooling_factor for China."""
        factor = db.get_district_cooling_factor("china")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_india(self, db):
        """Test get_district_cooling_factor for India."""
        factor = db.get_district_cooling_factor("india")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_australia(self, db):
        """Test get_district_cooling_factor for Australia."""
        factor = db.get_district_cooling_factor("australia")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_southeast_asia(self, db):
        """Test get_district_cooling_factor for Southeast Asia."""
        factor = db.get_district_cooling_factor("southeast_asia")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_latin_america(self, db):
        """Test get_district_cooling_factor for Latin America."""
        factor = db.get_district_cooling_factor("latin_america")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_africa(self, db):
        """Test get_district_cooling_factor for Africa."""
        factor = db.get_district_cooling_factor("africa")
        assert isinstance(factor, Decimal)

    def test_get_district_cooling_factor_global_default(self, db):
        """Test get_district_cooling_factor for global default."""
        factor = db.get_district_cooling_factor("global")
        assert isinstance(factor, Decimal)

    def test_get_district_ef_returns_decimal(self, db):
        """Test get_district_ef returns Decimal."""
        ef = db.get_district_ef("north_america")
        assert isinstance(ef, Decimal)


class TestHeatSourceFactors:
    """Test heat source emission factors for 11 sources."""

    def test_get_heat_source_factor_natural_gas(self, db):
        """Test get_heat_source_factor for natural_gas."""
        factor = db.get_heat_source_factor("natural_gas")
        assert isinstance(factor, Decimal)
        assert factor > Decimal("50.0")  # Typical natural gas EF

    def test_get_heat_source_factor_waste_heat(self, db):
        """Test get_heat_source_factor for waste_heat."""
        factor = db.get_heat_source_factor("waste_heat")
        assert factor == Decimal("0.0")

    def test_get_heat_source_factor_chp(self, db):
        """Test get_heat_source_factor for chp."""
        factor = db.get_heat_source_factor("chp")
        assert isinstance(factor, Decimal)

    def test_get_heat_source_factor_steam(self, db):
        """Test get_heat_source_factor for steam."""
        factor = db.get_heat_source_factor("steam")
        assert isinstance(factor, Decimal)

    def test_get_heat_source_factor_hot_water(self, db):
        """Test get_heat_source_factor for hot_water."""
        factor = db.get_heat_source_factor("hot_water")
        assert isinstance(factor, Decimal)

    def test_get_heat_source_factor_biomass(self, db):
        """Test get_heat_source_factor for biomass."""
        factor = db.get_heat_source_factor("biomass")
        assert isinstance(factor, Decimal)

    def test_get_heat_source_factor_solar_thermal(self, db):
        """Test get_heat_source_factor for solar_thermal."""
        factor = db.get_heat_source_factor("solar_thermal")
        assert factor == Decimal("0.0")

    def test_get_heat_source_factor_geothermal(self, db):
        """Test get_heat_source_factor for geothermal."""
        factor = db.get_heat_source_factor("geothermal")
        assert factor == Decimal("0.0") or factor < Decimal("10.0")

    def test_get_heat_source_factor_district_heat(self, db):
        """Test get_heat_source_factor for district_heat."""
        factor = db.get_heat_source_factor("district_heat")
        assert isinstance(factor, Decimal)

    def test_get_heat_source_factor_oil(self, db):
        """Test get_heat_source_factor for oil."""
        factor = db.get_heat_source_factor("oil")
        assert isinstance(factor, Decimal)

    def test_get_heat_source_factor_electric_resistance(self, db):
        """Test get_heat_source_factor for electric_resistance."""
        factor = db.get_heat_source_factor("electric_resistance")
        assert isinstance(factor, Decimal)

    def test_is_zero_emission_heat_source_waste_heat(self, db):
        """Test is_zero_emission_heat_source for waste_heat."""
        assert db.is_zero_emission_heat_source("waste_heat") is True

    def test_is_zero_emission_heat_source_solar(self, db):
        """Test is_zero_emission_heat_source for solar_thermal."""
        assert db.is_zero_emission_heat_source("solar_thermal") is True

    def test_is_zero_emission_heat_source_geothermal(self, db):
        """Test is_zero_emission_heat_source for geothermal."""
        assert db.is_zero_emission_heat_source("geothermal") is True

    def test_is_zero_emission_heat_source_natural_gas_false(self, db):
        """Test is_zero_emission_heat_source returns False for natural_gas."""
        assert db.is_zero_emission_heat_source("natural_gas") is False


class TestRefrigerantGWP:
    """Test refrigerant GWP values for 11 refrigerants."""

    def test_get_refrigerant_gwp_r134a_ar5(self, db):
        """Test get_refrigerant_gwp for R-134a with AR5."""
        gwp = db.get_refrigerant_gwp("R-134a", assessment_report="AR5")
        assert gwp == 1430

    def test_get_refrigerant_gwp_r134a_ar6(self, db):
        """Test get_refrigerant_gwp for R-134a with AR6."""
        gwp = db.get_refrigerant_gwp("R-134a", assessment_report="AR6")
        assert gwp == 1530

    def test_get_refrigerant_gwp_r410a(self, db):
        """Test get_refrigerant_gwp for R-410A."""
        gwp = db.get_refrigerant_gwp("R-410A", assessment_report="AR5")
        assert gwp == 2088

    def test_get_refrigerant_gwp_r32(self, db):
        """Test get_refrigerant_gwp for R-32."""
        gwp = db.get_refrigerant_gwp("R-32", assessment_report="AR5")
        assert gwp == 675

    def test_get_refrigerant_gwp_r407c(self, db):
        """Test get_refrigerant_gwp for R-407C."""
        gwp = db.get_refrigerant_gwp("R-407C", assessment_report="AR5")
        assert isinstance(gwp, int)

    def test_get_refrigerant_gwp_r404a(self, db):
        """Test get_refrigerant_gwp for R-404A."""
        gwp = db.get_refrigerant_gwp("R-404A", assessment_report="AR5")
        assert isinstance(gwp, int)

    def test_get_refrigerant_gwp_r507a(self, db):
        """Test get_refrigerant_gwp for R-507A."""
        gwp = db.get_refrigerant_gwp("R-507A", assessment_report="AR5")
        assert isinstance(gwp, int)

    def test_get_refrigerant_gwp_r1234yf(self, db):
        """Test get_refrigerant_gwp for R-1234yf (low GWP)."""
        gwp = db.get_refrigerant_gwp("R-1234yf", assessment_report="AR5")
        assert gwp < 10  # Very low GWP

    def test_get_refrigerant_gwp_r1234ze(self, db):
        """Test get_refrigerant_gwp for R-1234ze (low GWP)."""
        gwp = db.get_refrigerant_gwp("R-1234ze", assessment_report="AR5")
        assert gwp < 10

    def test_get_refrigerant_gwp_r717_ammonia(self, db):
        """Test get_refrigerant_gwp for R-717 (ammonia)."""
        gwp = db.get_refrigerant_gwp("R-717", assessment_report="AR5")
        assert gwp == 0  # Natural refrigerant

    def test_get_refrigerant_gwp_r744_co2(self, db):
        """Test get_refrigerant_gwp for R-744 (CO2)."""
        gwp = db.get_refrigerant_gwp("R-744", assessment_report="AR5")
        assert gwp == 1  # CO2 baseline


class TestEfficiencyConversions:
    """Test efficiency metric conversions."""

    def test_convert_efficiency_cop_to_eer(self, db):
        """Test convert_efficiency COP to EER (multiply by 3.412)."""
        cop = Decimal("6.0")
        eer = db.convert_efficiency(cop, from_unit="COP", to_unit="EER")
        expected = cop * Decimal("3.412")
        assert abs(eer - expected) < Decimal("0.01")

    def test_convert_efficiency_eer_to_cop(self, db):
        """Test convert_efficiency EER to COP."""
        eer = Decimal("20.0")
        cop = db.convert_efficiency(eer, from_unit="EER", to_unit="COP")
        expected = eer / Decimal("3.412")
        assert abs(cop - expected) < Decimal("0.01")

    def test_convert_efficiency_cop_to_kw_ton(self, db):
        """Test convert_efficiency COP to kW/ton."""
        cop = Decimal("6.0")
        kw_ton = db.convert_efficiency(cop, from_unit="COP", to_unit="kW/ton")
        # kW/ton = 3.517 / COP
        expected = Decimal("3.517") / cop
        assert abs(kw_ton - expected) < Decimal("0.01")

    def test_convert_efficiency_kw_ton_to_cop(self, db):
        """Test convert_efficiency kW/ton to COP."""
        kw_ton = Decimal("0.6")
        cop = db.convert_efficiency(kw_ton, from_unit="kW/ton", to_unit="COP")
        expected = Decimal("3.517") / kw_ton
        assert abs(cop - expected) < Decimal("0.1")

    def test_convert_efficiency_same_unit_returns_same(self, db):
        """Test convert_efficiency with same unit returns same value."""
        cop = Decimal("6.0")
        result = db.convert_efficiency(cop, from_unit="COP", to_unit="COP")
        assert result == cop


class TestIPLVCalculation:
    """Test IPLV calculation with AHRI weights."""

    def test_calculate_iplv_with_ahri_weights(self, db):
        """Test calculate_iplv with AHRI 550/590 weights."""
        cops = {
            "cop_100": Decimal("6.0"),
            "cop_75": Decimal("7.0"),
            "cop_50": Decimal("8.0"),
            "cop_25": Decimal("9.0"),
        }
        iplv = db.calculate_iplv(**cops)
        # IPLV = 0.01*6.0 + 0.42*7.0 + 0.45*8.0 + 0.12*9.0
        expected = (
            Decimal("0.01") * cops["cop_100"]
            + Decimal("0.42") * cops["cop_75"]
            + Decimal("0.45") * cops["cop_50"]
            + Decimal("0.12") * cops["cop_25"]
        )
        assert abs(iplv - expected) < Decimal("0.01")

    def test_calculate_iplv_higher_than_full_load(self, db):
        """Test calculate_iplv typically higher than full load COP."""
        cops = {
            "cop_100": Decimal("5.0"),
            "cop_75": Decimal("6.0"),
            "cop_50": Decimal("7.0"),
            "cop_25": Decimal("8.0"),
        }
        iplv = db.calculate_iplv(**cops)
        assert iplv > cops["cop_100"]


class TestCoolingUnitConversions:
    """Test cooling unit conversions."""

    def test_convert_cooling_units_ton_hour_to_kwh_th(self, db):
        """Test convert_cooling_units ton_hour to kwh_th."""
        ton_hours = Decimal("100.0")
        kwh_th = db.convert_cooling_units(ton_hours, from_unit="ton_hour", to_unit="kwh_th")
        # 1 ton_hour = 3.517 kWh_th
        expected = ton_hours * Decimal("3.517")
        assert abs(kwh_th - expected) < Decimal("0.1")

    def test_convert_cooling_units_kwh_th_to_ton_hour(self, db):
        """Test convert_cooling_units kwh_th to ton_hour."""
        kwh_th = Decimal("351.7")
        ton_hours = db.convert_cooling_units(kwh_th, from_unit="kwh_th", to_unit="ton_hour")
        expected = kwh_th / Decimal("3.517")
        assert abs(ton_hours - Decimal("100.0")) < Decimal("0.1")

    def test_convert_cooling_units_mwh_th_to_kwh_th(self, db):
        """Test convert_cooling_units MWh_th to kWh_th."""
        mwh_th = Decimal("1.0")
        kwh_th = db.convert_cooling_units(mwh_th, from_unit="MWh_th", to_unit="kwh_th")
        assert kwh_th == Decimal("1000.0")


class TestCustomTechnologyManagement:
    """Test custom technology addition and management."""

    def test_add_custom_technology(self, db):
        """Test add_custom_technology."""
        custom_spec = {
            "cop": Decimal("8.0"),
            "iplv": Decimal("9.0"),
            "type": "electric",
        }
        db.add_custom_technology("custom_high_efficiency", custom_spec)

        spec = db.get_technology_spec("custom_high_efficiency")
        assert spec["cop"] == Decimal("8.0")

    def test_add_custom_technology_retrieval(self, db):
        """Test custom technology can be retrieved."""
        custom_spec = {"cop": Decimal("7.5"), "type": "electric"}
        db.add_custom_technology("custom_chiller_2", custom_spec)

        cop = db.get_default_cop("custom_chiller_2")
        assert cop == Decimal("7.5")

    def test_reset_custom_factors_clears_custom(self, db):
        """Test reset_custom_factors clears custom technologies."""
        db.add_custom_technology("temp_tech", {"cop": Decimal("5.0")})
        db.reset_custom_factors()

        with pytest.raises((KeyError, ValueError)):
            db.get_technology_spec("temp_tech")


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_technology_raises_keyerror(self, db):
        """Test invalid technology raises KeyError."""
        with pytest.raises((KeyError, ValueError)):
            db.get_default_cop("nonexistent_technology")

    def test_invalid_region_raises_error(self, db):
        """Test invalid region raises error."""
        with pytest.raises((KeyError, ValueError)):
            db.get_district_cooling_factor("invalid_region")

    def test_invalid_heat_source_raises_error(self, db):
        """Test invalid heat source raises error."""
        with pytest.raises((KeyError, ValueError)):
            db.get_heat_source_factor("invalid_source")

    def test_invalid_refrigerant_raises_error(self, db):
        """Test invalid refrigerant raises error."""
        with pytest.raises((KeyError, ValueError)):
            db.get_refrigerant_gwp("invalid_refrigerant")

    def test_invalid_assessment_report_raises_error(self, db):
        """Test invalid assessment report raises error."""
        with pytest.raises(ValueError):
            db.get_refrigerant_gwp("R-134a", assessment_report="AR7")
