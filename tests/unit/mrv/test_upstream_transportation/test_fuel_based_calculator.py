"""
Unit tests for FuelBasedCalculatorEngine - AGENT-MRV-017 Upstream Transportation & Distribution.

Tests fuel-based emission calculations for transportation.
"""

import pytest
from decimal import Decimal
from typing import List

from greenlang.mrv.upstream_transportation.engines.fuel_based_calculator import (
    FuelBasedCalculatorEngine,
)
from greenlang.mrv.upstream_transportation.engines.transport_database import (
    FuelType,
    EmissionScope,
)
from greenlang.mrv.upstream_transportation.models import (
    FuelBasedInput,
    FuelBasedOutput,
    FuelBlend,
    GasBreakdown,
)


@pytest.fixture
def calculator():
    """Create FuelBasedCalculatorEngine instance."""
    return FuelBasedCalculatorEngine()


class TestDieselCalculations:
    """Test diesel fuel calculations."""

    def test_calculate_diesel_100_litres_ttw(self, calculator):
        """Test diesel TTW calculation for 100 litres."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        # 100 litres × 2.6868 kgCO2e/litre = 268.68 kgCO2e
        expected_emissions = Decimal("100") * Decimal("2.6868")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.scope == EmissionScope.TTW

    def test_calculate_diesel_100_litres_wtt(self, calculator):
        """Test diesel WTT (upstream) calculation for 100 litres."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.WTT,
        )

        output = calculator.calculate(input_data)

        # WTT (upstream) = 0.5001 kgCO2e/litre
        # 100 litres × 0.5001 = 50.01 kgCO2e
        expected_emissions = Decimal("100") * Decimal("0.5001")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.scope == EmissionScope.WTT

    def test_calculate_diesel_100_litres_wtw(self, calculator):
        """Test diesel WTW calculation for 100 litres (TTW + WTT)."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.WTW,
        )

        output = calculator.calculate(input_data)

        # WTW = TTW + WTT = 2.6868 + 0.5001 = 3.1869 kgCO2e/litre
        # 100 litres × 3.1869 = 318.69 kgCO2e
        expected_emissions = Decimal("100") * Decimal("3.1869")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.scope == EmissionScope.WTW
        assert output.ttw_emissions_kgco2e == Decimal("268.68")
        assert output.wtt_emissions_kgco2e == Decimal("50.01")


class TestOtherFuels:
    """Test calculations for other fuel types."""

    def test_calculate_jet_kerosene(self, calculator):
        """Test jet kerosene calculation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.JET_KEROSENE,
            quantity=Decimal("1000"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        # 1000 litres × 2.5392 kgCO2e/litre = 2539.2 kgCO2e
        expected_emissions = Decimal("1000") * Decimal("2.5392")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_hfo(self, calculator):
        """Test heavy fuel oil (HFO) calculation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.HFO,
            quantity=Decimal("500"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        # 500 litres × 3.1144 kgCO2e/litre = 1557.2 kgCO2e
        expected_emissions = Decimal("500") * Decimal("3.1144")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_vlsfo(self, calculator):
        """Test very low sulphur fuel oil (VLSFO) calculation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.VLSFO,
            quantity=Decimal("1000"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        # VLSFO similar to HFO, ~3.10 kgCO2e/litre
        assert output.total_emissions_kgco2e > Decimal("3000")
        assert output.total_emissions_kgco2e < Decimal("3200")

    def test_calculate_mgo(self, calculator):
        """Test marine gas oil (MGO) calculation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.MGO,
            quantity=Decimal("1000"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        # MGO similar to diesel, ~2.7 kgCO2e/litre
        assert output.total_emissions_kgco2e > Decimal("2600")
        assert output.total_emissions_kgco2e < Decimal("2800")

    def test_calculate_lng_marine(self, calculator):
        """Test liquefied natural gas (LNG) marine fuel calculation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.LNG,
            quantity=Decimal("100"),
            unit="tonnes",  # LNG typically measured in tonnes
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        # LNG: ~2750 kgCO2e/tonne (TTW)
        expected_emissions = Decimal("100") * Decimal("2750")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_cng(self, calculator):
        """Test compressed natural gas (CNG) calculation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.CNG,
            quantity=Decimal("500"),
            unit="kg",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        # CNG: ~2.69 kgCO2e/kg
        expected_emissions = Decimal("500") * Decimal("2.69")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_electricity(self, calculator):
        """Test electricity calculation (grid emissions)."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.ELECTRICITY,
            quantity=Decimal("1000"),
            unit="kWh",
            scope=EmissionScope.WTW,  # Electricity is inherently WTW
            region="UK",
        )

        output = calculator.calculate(input_data)

        # UK grid: ~0.193 kgCO2e/kWh (2024)
        expected_emissions = Decimal("1000") * Decimal("0.193")
        assert output.total_emissions_kgco2e == expected_emissions

    def test_calculate_hydrogen(self, calculator):
        """Test hydrogen calculation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.HYDROGEN,
            quantity=Decimal("100"),
            unit="kg",
            scope=EmissionScope.WTW,  # H2 production emissions
            hydrogen_source="grey",  # Steam methane reforming
        )

        output = calculator.calculate(input_data)

        # Grey H2: ~10 kgCO2e/kg H2
        expected_emissions = Decimal("100") * Decimal("10")
        assert output.total_emissions_kgco2e == expected_emissions


class TestBiofuelBlends:
    """Test biofuel blend calculations."""

    def test_calculate_with_blend_b20(self, calculator):
        """Test B20 blend (80% diesel + 20% biodiesel)."""
        blend = FuelBlend(
            fossil_fuel_type=FuelType.DIESEL,
            fossil_percent=Decimal("80"),
            biofuel_type=FuelType.BIODIESEL,
            biofuel_percent=Decimal("20"),
        )

        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
            fuel_blend=blend,
        )

        output = calculator.calculate(input_data)

        # B20: 80% fossil + 20% biogenic
        # Fossil CO2: 100 × 2.6868 × 0.80 = 214.944 kgCO2e
        # Biogenic CO2: 100 × 2.6868 × 0.20 = 53.736 kgCO2 (non-GHG)
        # Total GHG emissions: 214.944 kgCO2e
        expected_emissions = Decimal("100") * Decimal("2.6868") * Decimal("0.80")
        assert output.total_emissions_kgco2e == expected_emissions
        assert output.biogenic_co2_kgco2 == Decimal("100") * Decimal("2.6868") * Decimal("0.20")

    def test_calculate_with_blend_b100(self, calculator):
        """Test B100 (100% biodiesel)."""
        blend = FuelBlend(
            fossil_fuel_type=None,
            fossil_percent=Decimal("0"),
            biofuel_type=FuelType.BIODIESEL,
            biofuel_percent=Decimal("100"),
        )

        input_data = FuelBasedInput(
            fuel_type=FuelType.BIODIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
            fuel_blend=blend,
        )

        output = calculator.calculate(input_data)

        # B100: 100% biogenic CO2 (reported separately, not in GHG total)
        # GHG emissions: 0 kgCO2e (only CH4/N2O from combustion)
        # Biogenic CO2: 100 × 2.6868 = 268.68 kgCO2 (non-GHG)
        assert output.total_emissions_kgco2e < Decimal("10")  # Only CH4/N2O
        assert output.biogenic_co2_kgco2 == Decimal("100") * Decimal("2.6868")

    def test_calculate_with_blend_hvo(self, calculator):
        """Test HVO (Hydrotreated Vegetable Oil) blend."""
        blend = FuelBlend(
            fossil_fuel_type=FuelType.DIESEL,
            fossil_percent=Decimal("70"),
            biofuel_type=FuelType.HVO,
            biofuel_percent=Decimal("30"),
        )

        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.WTW,  # Include upstream
            fuel_blend=blend,
        )

        output = calculator.calculate(input_data)

        # HVO has lower WTW emissions than fossil diesel
        pure_diesel_wtw = Decimal("100") * Decimal("3.1869")
        assert output.total_emissions_kgco2e < pure_diesel_wtw


class TestUnitConversions:
    """Test fuel unit conversions."""

    def test_convert_fuel_units_litres_to_kg(self, calculator):
        """Test litres to kg conversion for diesel."""
        kg = calculator.convert_fuel_units(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            from_unit="litres",
            to_unit="kg",
        )

        # Diesel density: ~0.832 kg/litre
        # 100 litres × 0.832 = 83.2 kg
        assert kg == Decimal("83.2")

    def test_convert_fuel_units_litres_to_gallons(self, calculator):
        """Test litres to gallons conversion."""
        gallons = calculator.convert_fuel_units(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            from_unit="litres",
            to_unit="gallons_us",
        )

        # 100 litres × 0.264172 = 26.4172 gallons
        assert gallons == Decimal("26.4172")

    def test_convert_fuel_units_kg_to_litres(self, calculator):
        """Test kg to litres conversion for diesel."""
        litres = calculator.convert_fuel_units(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("83.2"),
            from_unit="kg",
            to_unit="litres",
        )

        # 83.2 kg / 0.832 kg/litre = 100 litres
        assert litres == Decimal("100")

    def test_get_fuel_density_diesel(self, calculator):
        """Test fuel density lookup for diesel."""
        density = calculator.get_fuel_density(FuelType.DIESEL)
        assert density == Decimal("0.832")  # kg/litre

    def test_get_fuel_density_hfo(self, calculator):
        """Test fuel density lookup for HFO."""
        density = calculator.get_fuel_density(FuelType.HFO)
        assert density == Decimal("0.960")  # kg/litre (heavier than diesel)

    def test_get_fuel_heating_value(self, calculator):
        """Test net calorific value lookup."""
        ncv = calculator.get_fuel_heating_value(FuelType.DIESEL)
        assert ncv == Decimal("43.0")  # MJ/kg (IPCC default)


class TestGasSplitting:
    """Test gas breakdown calculations."""

    def test_split_by_gas_diesel(self, calculator):
        """Test gas splitting for diesel (CO2, CH4, N2O)."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        gas_breakdown = output.gas_breakdown
        assert gas_breakdown.co2_kgco2e > Decimal("0")
        assert gas_breakdown.ch4_kgco2e > Decimal("0")
        assert gas_breakdown.n2o_kgco2e > Decimal("0")

        # Sum equals total
        total_from_gases = (
            gas_breakdown.co2_kgco2e +
            gas_breakdown.ch4_kgco2e +
            gas_breakdown.n2o_kgco2e
        )
        assert total_from_gases == output.total_emissions_kgco2e

    def test_split_by_gas_lng(self, calculator):
        """Test gas splitting for LNG (includes methane slip)."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.LNG,
            quantity=Decimal("10"),
            unit="tonnes",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        gas_breakdown = output.gas_breakdown
        # LNG has higher CH4 component due to methane slip in engines
        assert gas_breakdown.ch4_kgco2e > Decimal("0")


class TestBiogenicSplit:
    """Test biogenic vs fossil CO2 splitting."""

    def test_calculate_biogenic_fraction_diesel_zero(self, calculator):
        """Test biogenic fraction for 100% fossil diesel (zero biogenic)."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        assert output.biogenic_co2_kgco2 == Decimal("0")
        assert output.fossil_co2_kgco2e == output.gas_breakdown.co2_kgco2e

    def test_calculate_biogenic_fraction_b100_full(self, calculator):
        """Test biogenic fraction for B100 (100% biogenic)."""
        blend = FuelBlend(
            fossil_fuel_type=None,
            fossil_percent=Decimal("0"),
            biofuel_type=FuelType.BIODIESEL,
            biofuel_percent=Decimal("100"),
        )

        input_data = FuelBasedInput(
            fuel_type=FuelType.BIODIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
            fuel_blend=blend,
        )

        output = calculator.calculate(input_data)

        # All CO2 is biogenic
        assert output.biogenic_co2_kgco2 > Decimal("250")  # ~268.68 kgCO2
        assert output.fossil_co2_kgco2e == Decimal("0")

    def test_calculate_biogenic_co2(self, calculator):
        """Test biogenic CO2 calculation for B20 blend."""
        blend = FuelBlend(
            fossil_fuel_type=FuelType.DIESEL,
            fossil_percent=Decimal("80"),
            biofuel_type=FuelType.BIODIESEL,
            biofuel_percent=Decimal("20"),
        )

        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
            fuel_blend=blend,
        )

        output = calculator.calculate(input_data)

        # 20% of CO2 is biogenic
        total_co2 = Decimal("100") * Decimal("2.6868")
        expected_biogenic = total_co2 * Decimal("0.20")
        assert output.biogenic_co2_kgco2 == expected_biogenic

    def test_calculate_methane_slip_lng(self, calculator):
        """Test methane slip calculation for LNG engines."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.LNG,
            quantity=Decimal("10"),
            unit="tonnes",
            scope=EmissionScope.TTW,
            engine_type="dual_fuel",  # Higher methane slip
        )

        output = calculator.calculate(input_data)

        # Dual-fuel engines have ~2-5% methane slip
        # Should see elevated CH4 emissions
        assert output.gas_breakdown.ch4_kgco2e > Decimal("100")


class TestCargoAllocation:
    """Test cargo allocation methods."""

    def test_allocate_to_cargo_mass(self, calculator):
        """Test allocation of vessel fuel to cargo by mass."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.HFO,
            quantity=Decimal("1000"),
            unit="litres",
            scope=EmissionScope.TTW,
            cargo_mass_tonnes=Decimal("5000"),
            total_vessel_capacity_tonnes=Decimal("10000"),
        )

        output = calculator.calculate(input_data)

        # Allocate 50% of emissions to cargo (5000/10000)
        total_vessel_emissions = Decimal("1000") * Decimal("3.1144")
        allocated_emissions = total_vessel_emissions * Decimal("0.5")
        assert output.allocated_emissions_kgco2e == allocated_emissions

    def test_allocate_to_cargo_volume(self, calculator):
        """Test allocation of vessel fuel to cargo by volume."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.HFO,
            quantity=Decimal("1000"),
            unit="litres",
            scope=EmissionScope.TTW,
            cargo_volume_teu=Decimal("2000"),  # TEU (Twenty-foot Equivalent Units)
            total_vessel_capacity_teu=Decimal("5000"),
        )

        output = calculator.calculate(input_data)

        # Allocate 40% of emissions to cargo (2000/5000)
        total_vessel_emissions = Decimal("1000") * Decimal("3.1144")
        allocated_emissions = total_vessel_emissions * Decimal("0.4")
        assert output.allocated_emissions_kgco2e == allocated_emissions

    def test_calculate_per_tkm(self, calculator):
        """Test conversion to per-tonne-km intensity."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
            cargo_mass_tonnes=Decimal("20"),
            distance_km=Decimal("500"),
        )

        output = calculator.calculate(input_data)

        # Total emissions: 100 × 2.6868 = 268.68 kgCO2e
        # Intensity: 268.68 / (20 × 500) = 0.026868 kgCO2e/tonne-km
        expected_intensity = Decimal("268.68") / (Decimal("20") * Decimal("500"))
        assert output.intensity_kgco2e_per_tkm == expected_intensity


class TestBatchCalculations:
    """Test batch processing capabilities."""

    def test_batch_calculate(self, calculator):
        """Test batch calculation of multiple fuel consumptions."""
        inputs = [
            FuelBasedInput(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("100"),
                unit="litres",
                scope=EmissionScope.TTW,
            ),
            FuelBasedInput(
                fuel_type=FuelType.JET_KEROSENE,
                quantity=Decimal("500"),
                unit="litres",
                scope=EmissionScope.TTW,
            ),
            FuelBasedInput(
                fuel_type=FuelType.HFO,
                quantity=Decimal("1000"),
                unit="litres",
                scope=EmissionScope.TTW,
            ),
        ]

        outputs = calculator.batch_calculate(inputs)

        assert len(outputs) == 3
        assert all(o.total_emissions_kgco2e > Decimal("0") for o in outputs)

    def test_aggregate_by_fuel_type(self, calculator):
        """Test aggregation of emissions by fuel type."""
        inputs = [
            FuelBasedInput(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("100"),
                unit="litres",
                scope=EmissionScope.TTW,
            ),
            FuelBasedInput(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("200"),
                unit="litres",
                scope=EmissionScope.TTW,
            ),
            FuelBasedInput(
                fuel_type=FuelType.HFO,
                quantity=Decimal("500"),
                unit="litres",
                scope=EmissionScope.TTW,
            ),
        ]

        outputs = calculator.batch_calculate(inputs)
        aggregated = calculator.aggregate_by_fuel_type(outputs)

        assert FuelType.DIESEL in aggregated
        assert FuelType.HFO in aggregated

        # Diesel: (100 + 200) × 2.6868 = 806.04 kgCO2e
        expected_diesel = Decimal("300") * Decimal("2.6868")
        assert aggregated[FuelType.DIESEL] == expected_diesel


class TestDataQualityScoring:
    """Test data quality scoring."""

    def test_get_data_quality_score_with_actual_fuel(self, calculator):
        """Test data quality score for actual fuel data (high quality)."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
            data_source="actual",  # High quality
        )

        output = calculator.calculate(input_data)

        assert output.data_quality_score >= Decimal("0.9")


class TestValidation:
    """Test input validation and error handling."""

    def test_validate_fuel_input_valid(self, calculator):
        """Test valid fuel input passes validation."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        # Should not raise
        calculator.validate_input(input_data)

    def test_validate_fuel_input_negative_quantity(self, calculator):
        """Test negative fuel quantity raises ValueError."""
        with pytest.raises(ValueError, match="Quantity must be greater than zero"):
            input_data = FuelBasedInput(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("-100"),
                unit="litres",
                scope=EmissionScope.TTW,
            )
            calculator.validate_input(input_data)

    def test_estimate_fuel_from_distance(self, calculator):
        """Test fuel estimation from distance (when fuel data unavailable)."""
        estimated_fuel = calculator.estimate_fuel_from_distance(
            vehicle_type="articulated_hgv",
            distance_km=Decimal("1000"),
            fuel_type=FuelType.DIESEL,
        )

        # Typical consumption: ~35 litres/100km = 350 litres for 1000km
        assert Decimal("300") < estimated_fuel < Decimal("400")


class TestWTWValidation:
    """Test well-to-wheel (WTW) validation."""

    def test_wtw_equals_ttw_plus_wtt(self, calculator):
        """Test that WTW = TTW + WTT."""
        # Calculate TTW
        ttw_input = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.TTW,
        )
        ttw_output = calculator.calculate(ttw_input)

        # Calculate WTT
        wtt_input = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.WTT,
        )
        wtt_output = calculator.calculate(wtt_input)

        # Calculate WTW
        wtw_input = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.WTW,
        )
        wtw_output = calculator.calculate(wtw_input)

        # WTW should equal TTW + WTT
        assert wtw_output.total_emissions_kgco2e == (
            ttw_output.total_emissions_kgco2e + wtt_output.total_emissions_kgco2e
        )

    def test_biofuel_lower_wtw_than_fossil(self, calculator):
        """Test that biofuels have lower WTW emissions than fossil fuels."""
        # Fossil diesel WTW
        fossil_input = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.WTW,
        )
        fossil_output = calculator.calculate(fossil_input)

        # B100 biodiesel WTW
        biofuel_blend = FuelBlend(
            fossil_fuel_type=None,
            fossil_percent=Decimal("0"),
            biofuel_type=FuelType.BIODIESEL,
            biofuel_percent=Decimal("100"),
        )
        biofuel_input = FuelBasedInput(
            fuel_type=FuelType.BIODIESEL,
            quantity=Decimal("100"),
            unit="litres",
            scope=EmissionScope.WTW,
            fuel_blend=biofuel_blend,
        )
        biofuel_output = calculator.calculate(biofuel_input)

        # Biofuel should have lower WTW GHG emissions
        assert biofuel_output.total_emissions_kgco2e < fossil_output.total_emissions_kgco2e


class TestDecimalPrecision:
    """Test Decimal arithmetic precision."""

    def test_decimal_precision(self, calculator):
        """Test that all calculations use Decimal (not float)."""
        input_data = FuelBasedInput(
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("100.123456789"),
            unit="litres",
            scope=EmissionScope.TTW,
        )

        output = calculator.calculate(input_data)

        assert isinstance(output.total_emissions_kgco2e, Decimal)
        assert isinstance(output.gas_breakdown.co2_kgco2e, Decimal)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_zero_quantity_raises(self, calculator):
        """Test that zero fuel quantity raises ValueError."""
        with pytest.raises(ValueError, match="Quantity must be greater than zero"):
            input_data = FuelBasedInput(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("0"),
                unit="litres",
                scope=EmissionScope.TTW,
            )
            calculator.calculate(input_data)

    def test_invalid_fuel_type_raises(self, calculator):
        """Test that invalid fuel type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid fuel type"):
            input_data = FuelBasedInput(
                fuel_type="invalid_fuel",
                quantity=Decimal("100"),
                unit="litres",
                scope=EmissionScope.TTW,
            )
            calculator.calculate(input_data)
