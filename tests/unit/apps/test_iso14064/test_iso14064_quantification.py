# -*- coding: utf-8 -*-
"""
Unit tests for QuantificationEngine -- ISO 14064-1:2018 Clause 6.1.

Tests all three calculation methods (emission factor, direct measurement,
mass balance), multi-gas quantification, GWP conversion, data quality scoring,
emission factor registry, and per-source aggregation with 30+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    DataQualityTier,
    GHGGas,
    GWPSource,
    ISOCategory,
    QuantificationMethod,
)
from services.quantification_engine import QuantificationEngine


class TestEmissionFactorMethod:
    """Test the emission factor (calculation-based) method."""

    def test_basic_calculation(self, quant_engine):
        result = quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("1000"),
            emission_factor=Decimal("2.5"),
            gwp=Decimal("1"),
        )
        assert result.tco2e == Decimal("2500.0000")
        assert result.method == QuantificationMethod.CALCULATION_BASED

    def test_calculation_with_gwp(self, quant_engine):
        result = quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("100"),
            emission_factor=Decimal("0.5"),
            gwp=Decimal("28"),
            gas=GHGGas.CH4,
        )
        # 100 * 0.5 * 28 = 1400
        assert result.tco2e == Decimal("1400.0000")

    def test_zero_activity_data(self, quant_engine):
        result = quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("0"),
            emission_factor=Decimal("2.5"),
        )
        assert result.tco2e == Decimal("0.0000")

    def test_negative_activity_data_raises(self, quant_engine):
        with pytest.raises(ValueError, match="negative"):
            quant_engine.calculate_emission_factor_method(
                activity_data=Decimal("-100"),
                emission_factor=Decimal("2.5"),
            )

    def test_negative_emission_factor_raises(self, quant_engine):
        with pytest.raises(ValueError, match="negative"):
            quant_engine.calculate_emission_factor_method(
                activity_data=Decimal("100"),
                emission_factor=Decimal("-2.5"),
            )

    def test_biogenic_flag(self, quant_engine):
        result = quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("1000"),
            emission_factor=Decimal("1"),
            is_biogenic=True,
        )
        assert result.biogenic_co2 == result.tco2e

    def test_non_biogenic_default(self, quant_engine):
        result = quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("1000"),
            emission_factor=Decimal("1"),
        )
        assert result.biogenic_co2 == Decimal("0")

    def test_result_stored(self, quant_engine):
        result = quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("100"),
            emission_factor=Decimal("1"),
        )
        retrieved = quant_engine.get_result(result.id)
        assert retrieved is not None
        assert retrieved.tco2e == result.tco2e

    def test_source_stored(self, quant_engine):
        result = quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("100"),
            emission_factor=Decimal("1"),
            source_name="Test Boiler",
            inventory_id="inv-test",
        )
        source = quant_engine.get_source(result.source_id)
        assert source is not None
        assert source.inventory_id == "inv-test"


class TestDirectMeasurement:
    """Test direct measurement method."""

    def test_basic_measurement(self, quant_engine):
        result = quant_engine.calculate_direct_measurement(
            concentration=Decimal("100"),
            flow_rate=Decimal("500"),
            time_period_hours=Decimal("8760"),
        )
        assert result.tco2e > Decimal("0")
        assert result.method == QuantificationMethod.DIRECT_MEASUREMENT

    def test_negative_concentration_raises(self, quant_engine):
        with pytest.raises(ValueError, match="negative"):
            quant_engine.calculate_direct_measurement(
                concentration=Decimal("-10"),
                flow_rate=Decimal("500"),
                time_period_hours=Decimal("100"),
            )

    def test_zero_time_period_raises(self, quant_engine):
        with pytest.raises(ValueError, match="positive"):
            quant_engine.calculate_direct_measurement(
                concentration=Decimal("100"),
                flow_rate=Decimal("500"),
                time_period_hours=Decimal("0"),
            )

    def test_different_gases_have_different_density(self, quant_engine):
        result_co2 = quant_engine.calculate_direct_measurement(
            concentration=Decimal("100"),
            flow_rate=Decimal("100"),
            time_period_hours=Decimal("1"),
            gas=GHGGas.CO2,
            gwp=Decimal("1"),
        )
        result_ch4 = quant_engine.calculate_direct_measurement(
            concentration=Decimal("100"),
            flow_rate=Decimal("100"),
            time_period_hours=Decimal("1"),
            gas=GHGGas.CH4,
            gwp=Decimal("1"),
        )
        # Different density factors means different raw emissions
        assert result_co2.raw_emissions_tonnes != result_ch4.raw_emissions_tonnes


class TestMassBalance:
    """Test mass balance method."""

    def test_basic_mass_balance(self, quant_engine):
        inputs = [
            {"name": "Coal", "mass_tonnes": Decimal("1000"), "carbon_fraction": Decimal("0.75")},
        ]
        outputs = [
            {"name": "Ash", "mass_tonnes": Decimal("100"), "carbon_fraction": Decimal("0.05")},
        ]
        result = quant_engine.calculate_mass_balance(
            inputs=inputs,
            outputs=outputs,
        )
        # Input carbon: 1000*0.75 = 750, Output carbon: 100*0.05 = 5
        # Emitted carbon: 745, CO2 = 745 * 44/12 = 2731.67
        assert result.tco2e > Decimal("2700")
        assert result.method == QuantificationMethod.MASS_BALANCE

    def test_no_inputs_raises(self, quant_engine):
        with pytest.raises(ValueError, match="input"):
            quant_engine.calculate_mass_balance(inputs=[], outputs=[])

    def test_negative_result_becomes_zero(self, quant_engine):
        inputs = [
            {"name": "In", "mass_tonnes": Decimal("10"), "carbon_fraction": Decimal("0.1")},
        ]
        outputs = [
            {"name": "Out", "mass_tonnes": Decimal("100"), "carbon_fraction": Decimal("0.5")},
        ]
        result = quant_engine.calculate_mass_balance(
            inputs=inputs, outputs=outputs,
        )
        assert result.tco2e == Decimal("0.0000")


class TestMultiGasQuantification:
    """Test quantify_multi_gas batch method."""

    def test_multi_gas(self, quant_engine):
        gas_data = [
            {"gas": "CO2", "activity_data": 1000, "emission_factor": 2.5},
            {"gas": "CH4", "activity_data": 100, "emission_factor": 0.5},
        ]
        result = quant_engine.quantify_multi_gas(gas_data)
        assert result["total_tco2e"] > Decimal("0")
        assert "CO2" in result["by_gas"]
        assert len(result["results"]) == 2

    def test_multi_gas_uses_gwp_from_table(self, quant_engine):
        gas_data = [
            {"gas": "CH4", "activity_data": 1, "emission_factor": 1},
        ]
        result = quant_engine.quantify_multi_gas(gas_data, gwp_source=GWPSource.AR5)
        # AR5 CH4 GWP = 28, so 1 * 1 * 28 = 28
        assert result["total_tco2e"] == Decimal("28.0000")


class TestGWPConversion:
    """Test GWP conversion and lookup."""

    def test_co2_gwp_is_1(self, quant_engine):
        gwp = quant_engine.get_gwp_value(GHGGas.CO2, GWPSource.AR5)
        assert gwp == Decimal("1")

    def test_ch4_ar5_gwp(self, quant_engine):
        gwp = quant_engine.get_gwp_value(GHGGas.CH4, GWPSource.AR5)
        assert gwp == Decimal("28")

    def test_ch4_ar6_gwp(self, quant_engine):
        gwp = quant_engine.get_gwp_value(GHGGas.CH4, GWPSource.AR6)
        assert gwp == Decimal("27.9")

    def test_convert_gwp(self, quant_engine):
        result = quant_engine.convert_gwp(
            Decimal("10"), GHGGas.CH4, GWPSource.AR5,
        )
        assert result == Decimal("280.0000")


class TestDataQualityScoring:
    """Test 5-dimension data quality scoring."""

    def test_default_scores(self, quant_engine):
        dqi = quant_engine.score_data_quality()
        assert dqi.overall_score == Decimal("3")

    def test_high_quality_scores(self, quant_engine):
        dqi = quant_engine.score_data_quality(
            completeness=Decimal("5"),
            accuracy=Decimal("5"),
            consistency=Decimal("5"),
            timeliness=Decimal("5"),
            methodology=Decimal("5"),
        )
        assert dqi.overall_score == Decimal("5")

    def test_dqi_to_tier_mapping(self, quant_engine):
        high_dqi = quant_engine.score_data_quality(
            completeness=Decimal("5"),
            accuracy=Decimal("5"),
            consistency=Decimal("4"),
            timeliness=Decimal("4"),
            methodology=Decimal("5"),
        )
        tier = quant_engine._dqi_to_tier(high_dqi)
        assert tier in (DataQualityTier.TIER_3, DataQualityTier.TIER_4)


class TestEmissionFactorRegistry:
    """Test emission factor registration and lookup."""

    def test_register_ef(self, quant_engine):
        ef = quant_engine.register_emission_factor(
            ef_id="IPCC-2019-NG-CO2",
            gas=GHGGas.CO2,
            value=Decimal("56.1"),
            unit="kgCO2/GJ",
            source="IPCC",
            source_year=2019,
        )
        assert ef["ef_id"] == "IPCC-2019-NG-CO2"
        assert ef["provenance_hash"] is not None
        assert len(ef["provenance_hash"]) == 64

    def test_get_registered_ef(self, quant_engine):
        quant_engine.register_emission_factor(
            ef_id="TEST-001", gas=GHGGas.CO2,
            value=Decimal("10"), unit="kg/unit",
            source="Test", source_year=2024,
        )
        ef = quant_engine.get_emission_factor("TEST-001")
        assert ef is not None
        assert ef["value"] == "10"

    def test_list_emission_factors_by_gas(self, quant_engine):
        quant_engine.register_emission_factor(
            ef_id="CO2-EF", gas=GHGGas.CO2,
            value=Decimal("10"), unit="kg/unit",
            source="Test", source_year=2024,
        )
        quant_engine.register_emission_factor(
            ef_id="CH4-EF", gas=GHGGas.CH4,
            value=Decimal("5"), unit="kg/unit",
            source="Test", source_year=2024,
        )
        co2_factors = quant_engine.list_emission_factors(gas=GHGGas.CO2)
        assert len(co2_factors) >= 1
        assert all(f["gas"] == "CO2" for f in co2_factors)


class TestAggregation:
    """Test per-source aggregation."""

    def test_aggregate_empty_inventory(self, quant_engine):
        result = quant_engine.aggregate_by_source("empty-inv")
        assert result["total_tco2e"] == Decimal("0")
        assert result["source_count"] == 0

    def test_aggregate_with_sources(self, quant_engine):
        quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("100"), emission_factor=Decimal("2"),
            inventory_id="inv-agg", category=ISOCategory.CATEGORY_1_DIRECT,
        )
        quant_engine.calculate_emission_factor_method(
            activity_data=Decimal("50"), emission_factor=Decimal("1"),
            inventory_id="inv-agg", category=ISOCategory.CATEGORY_2_ENERGY,
        )
        result = quant_engine.aggregate_by_source("inv-agg")
        assert result["source_count"] == 2
        assert result["total_tco2e"] == Decimal("250.0000")
        assert ISOCategory.CATEGORY_1_DIRECT.value in result["by_category"]
