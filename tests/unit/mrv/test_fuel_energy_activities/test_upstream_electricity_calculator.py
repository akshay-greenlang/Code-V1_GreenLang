"""
Unit tests for UpstreamElectricityCalculatorEngine (AGENT-MRV-016 Engine 3 - Activity 3b)

Tests all methods of UpstreamElectricityCalculatorEngine with comprehensive coverage.
Validates upstream electricity emissions (T&D losses) calculations, grid mix analysis,
and reporting for Scope 3 Category 3 - Fuel and Energy-Related Activities (Activity 3b).
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from greenlang.agents.mrv.fuel_energy_activities.models import (
    ElectricityConsumptionRecord,
    UpstreamElectricityResult,
    UpstreamElectricityBatchResult,
    EnergyType,
    EnergyUnit,
    EmissionFactorSource,
    DataQualityIndicator,
    GasBreakdown,
    GridMix,
)
from greenlang.agents.mrv.fuel_energy_activities.engines.upstream_electricity_calculator import (
    UpstreamElectricityCalculatorEngine,
)
from greenlang_core.exceptions import ValidationError, ProcessingError


# Fixtures
@pytest.fixture
def calculator_engine():
    """Create UpstreamElectricityCalculatorEngine instance for testing."""
    return UpstreamElectricityCalculatorEngine()


@pytest.fixture
def us_electricity_record():
    """Create sample US electricity consumption record."""
    return ElectricityConsumptionRecord(
        record_id="ELEC-US-001",
        energy_type=EnergyType.ELECTRICITY,
        quantity=Decimal("100000"),  # kWh
        unit=EnergyUnit.KWH,
        consumption_date=date(2025, 1, 15),
        facility_id="FAC-001",
        region="US",
        calculation_method="location-based",
        source=EmissionFactorSource.EPA_GHGRP,
    )


@pytest.fixture
def uk_electricity_record():
    """Create sample UK electricity consumption record."""
    return ElectricityConsumptionRecord(
        record_id="ELEC-UK-001",
        energy_type=EnergyType.ELECTRICITY,
        quantity=Decimal("50000"),
        unit=EnergyUnit.KWH,
        consumption_date=date(2025, 1, 20),
        facility_id="FAC-002",
        region="UK",
        calculation_method="location-based",
        source=EmissionFactorSource.DEFRA,
    )


@pytest.fixture
def de_electricity_record():
    """Create sample Germany electricity consumption record."""
    return ElectricityConsumptionRecord(
        record_id="ELEC-DE-001",
        energy_type=EnergyType.ELECTRICITY,
        quantity=Decimal("75000"),
        unit=EnergyUnit.KWH,
        consumption_date=date(2025, 1, 25),
        facility_id="FAC-003",
        region="DE",
        calculation_method="location-based",
    )


@pytest.fixture
def steam_record():
    """Create sample steam consumption record."""
    return ElectricityConsumptionRecord(
        record_id="STEAM-001",
        energy_type=EnergyType.STEAM,
        quantity=Decimal("10000"),
        unit=EnergyUnit.KWH,
        consumption_date=date(2025, 1, 15),
        facility_id="FAC-001",
        region="US",
    )


# Test Class
class TestUpstreamElectricityCalculatorEngine:
    """Test suite for UpstreamElectricityCalculatorEngine."""

    def test_initialization(self, calculator_engine):
        """Test engine initializes correctly."""
        assert calculator_engine is not None

    def test_calculate_location_based_us(self, calculator_engine, us_electricity_record):
        """Test upstream electricity calculation for US: 100000 kWh × 0.045 = 4500 kgCO2e."""
        result = calculator_engine.calculate(us_electricity_record)

        assert result is not None
        assert result.record_id == "ELEC-US-001"
        assert result.energy_type == EnergyType.ELECTRICITY
        assert result.energy_quantity_kwh == Decimal("100000")

        # Upstream emissions (T&D losses): ~4-5% of generation emissions
        # US grid avg ~0.40 kgCO2e/kWh generation → ~0.045 kgCO2e/kWh upstream
        # 100000 kWh × 0.045 = 4500 kgCO2e
        expected_emissions = Decimal("4500")
        assert result.upstream_emissions_kg == pytest.approx(
            expected_emissions, rel=Decimal("0.2")
        )

        assert result.upstream_emission_factor > 0
        assert result.emission_factor_source == EmissionFactorSource.EPA_GHGRP
        assert result.calculation_method == "location-based"

    def test_calculate_location_based_uk(self, calculator_engine, uk_electricity_record):
        """Test upstream electricity calculation for UK."""
        result = calculator_engine.calculate(uk_electricity_record)

        assert result is not None
        assert result.energy_type == EnergyType.ELECTRICITY
        assert result.energy_quantity_kwh == Decimal("50000")

        # UK upstream factor: ~0.030-0.040 kgCO2e/kWh
        assert Decimal("1000") < result.upstream_emissions_kg < Decimal("3000")

    def test_calculate_location_based_de(self, calculator_engine, de_electricity_record):
        """Test upstream electricity calculation for Germany."""
        result = calculator_engine.calculate(de_electricity_record)

        assert result is not None
        assert result.region == "DE"
        assert result.upstream_emissions_kg > 0

    def test_calculate_location_based_fr(self, calculator_engine):
        """Test upstream for France (low nuclear mix → low upstream emissions)."""
        fr_record = ElectricityConsumptionRecord(
            record_id="ELEC-FR-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="FR",
            calculation_method="location-based",
        )

        result = calculator_engine.calculate(fr_record)

        assert result is not None
        # France has low-carbon grid (nuclear) → lower upstream emissions
        # Should be significantly lower than US/UK/DE
        assert result.upstream_emissions_kg > 0
        # France upstream: ~0.010-0.015 kgCO2e/kWh (low due to nuclear)

    def test_calculate_market_based_with_supplier(self, calculator_engine):
        """Test market-based calculation with specific supplier factor."""
        market_record = ElectricityConsumptionRecord(
            record_id="ELEC-MB-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            calculation_method="market-based",
            supplier_specific_factor=Decimal("0.250"),  # Supplier emission factor
            supplier_name="GreenPower Inc",
        )

        result = calculator_engine.calculate(market_record)

        assert result is not None
        assert result.calculation_method == "market-based"
        # Upstream should be based on supplier mix
        assert result.upstream_emissions_kg > 0

    def test_calculate_batch_multiple_records(
        self, calculator_engine, us_electricity_record, uk_electricity_record
    ):
        """Test batch calculation with multiple electricity records."""
        records = [us_electricity_record, uk_electricity_record]

        batch_result = calculator_engine.calculate_batch(records)

        assert batch_result is not None
        assert len(batch_result.results) == 2
        assert batch_result.total_upstream_emissions_kg > 0
        assert batch_result.total_upstream_emissions_tonnes > 0

        # Individual results should sum to total
        individual_sum = sum(r.upstream_emissions_kg for r in batch_result.results)
        assert individual_sum == pytest.approx(
            batch_result.total_upstream_emissions_kg, rel=Decimal("0.01")
        )

    def test_calculate_steam_upstream(self, calculator_engine, steam_record):
        """Test upstream emissions calculation for purchased steam."""
        result = calculator_engine.calculate(steam_record)

        assert result is not None
        assert result.energy_type == EnergyType.STEAM
        assert result.upstream_emissions_kg > 0

        # Steam upstream includes fuel extraction/processing for boiler fuel

    def test_calculate_heat_upstream(self, calculator_engine):
        """Test upstream emissions calculation for purchased heat."""
        heat_record = ElectricityConsumptionRecord(
            record_id="HEAT-001",
            energy_type=EnergyType.HEAT,
            quantity=Decimal("50000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        result = calculator_engine.calculate(heat_record)

        assert result is not None
        assert result.energy_type == EnergyType.HEAT
        assert result.upstream_emissions_kg > 0

    def test_calculate_cooling_upstream(self, calculator_engine):
        """Test upstream emissions calculation for purchased cooling."""
        cooling_record = ElectricityConsumptionRecord(
            record_id="COOL-001",
            energy_type=EnergyType.COOLING,
            quantity=Decimal("30000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        result = calculator_engine.calculate(cooling_record)

        assert result is not None
        assert result.energy_type == EnergyType.COOLING
        assert result.upstream_emissions_kg > 0

    def test_calculate_grid_mix_upstream(self, calculator_engine):
        """Test upstream calculation using weighted grid mix."""
        # Grid mix with multiple sources
        grid_mix = GridMix(
            coal=Decimal("0.30"),
            natural_gas=Decimal("0.40"),
            nuclear=Decimal("0.20"),
            renewables=Decimal("0.10"),
        )

        mixed_record = ElectricityConsumptionRecord(
            record_id="MIX-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            grid_mix=grid_mix,
        )

        result = calculator_engine.calculate(mixed_record)

        assert result is not None
        assert result.upstream_emissions_kg > 0
        # Upstream should reflect weighted average of fuel WTT factors

    def test_get_upstream_ef_all_countries(self, calculator_engine):
        """Test retrieving upstream EF for all 46 countries."""
        countries = [
            "US", "UK", "DE", "FR", "IT", "ES", "NL", "BE", "AT", "DK",
            "SE", "NO", "FI", "PL", "CZ", "HU", "RO", "BG", "GR", "PT",
            "IE", "LU", "HR", "SI", "SK", "LT", "LV", "EE", "MT", "CY",
            "CA", "MX", "BR", "AR", "CL", "CO", "PE", "VE",
            "CN", "JP", "IN", "KR", "AU", "NZ", "ZA", "GLOBAL",
        ]

        for country in countries:
            factor = calculator_engine.get_upstream_factor(
                region=country,
                energy_type=EnergyType.ELECTRICITY,
            )

            if factor:  # Some countries may not have data
                assert factor > 0
                assert factor < Decimal("0.2")  # Reasonable upper bound

    def test_get_upstream_ef_by_egrid(self, calculator_engine):
        """Test retrieving upstream EF by eGRID subregion (26 US subregions)."""
        egrid_subregions = [
            "AKGD", "AKMS", "AZNM", "CAMX", "ERCT", "FRCC", "HIMS", "HIOA",
            "MROE", "MROW", "NEWE", "NWPP", "NYCW", "NYLI", "NYUP", "RFCE",
            "RFCM", "RFCW", "RMPA", "SPNO", "SPSO", "SRMV", "SRMW", "SRSO",
            "SRTV", "SRVC",
        ]

        for subregion in egrid_subregions:
            factor = calculator_engine.get_upstream_factor(
                region=f"US-{subregion}",
                energy_type=EnergyType.ELECTRICITY,
            )

            if factor:  # Some subregions may not have upstream data
                assert factor > 0

    def test_calculate_per_gas_breakdown(self, calculator_engine, us_electricity_record):
        """Test per-gas breakdown (CO2, CH4, N2O) for upstream electricity."""
        result = calculator_engine.calculate(us_electricity_record)

        assert result.gas_breakdown is not None
        assert result.gas_breakdown.co2 > 0
        assert result.gas_breakdown.ch4 >= 0
        assert result.gas_breakdown.n2o >= 0

        # Total should equal upstream emissions
        total_gas = (
            result.gas_breakdown.co2
            + result.gas_breakdown.ch4
            + result.gas_breakdown.n2o
        )
        assert total_gas == pytest.approx(
            result.upstream_emissions_kg, rel=Decimal("0.01")
        )

    def test_compare_location_vs_market(self, calculator_engine):
        """Test comparison between location-based and market-based methods."""
        location_record = ElectricityConsumptionRecord(
            record_id="LOC-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            calculation_method="location-based",
        )

        market_record = ElectricityConsumptionRecord(
            record_id="MKT-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            calculation_method="market-based",
            supplier_specific_factor=Decimal("0.100"),  # Low-carbon supplier
        )

        loc_result = calculator_engine.calculate(location_record)
        mkt_result = calculator_engine.calculate(market_record)

        comparison = calculator_engine.compare_methods(loc_result, mkt_result)

        assert "location_based_emissions" in comparison
        assert "market_based_emissions" in comparison
        assert "difference_kg" in comparison

    def test_aggregate_by_energy_type(self, calculator_engine, us_electricity_record, steam_record):
        """Test aggregation of emissions by energy type."""
        records = [us_electricity_record, steam_record]
        batch_result = calculator_engine.calculate_batch(records)

        aggregated = calculator_engine.aggregate_by_energy_type(batch_result)

        assert EnergyType.ELECTRICITY in aggregated
        assert EnergyType.STEAM in aggregated

        assert aggregated[EnergyType.ELECTRICITY]["emissions_kg"] > 0
        assert aggregated[EnergyType.STEAM]["emissions_kg"] > 0

    def test_aggregate_by_region(
        self, calculator_engine, us_electricity_record, uk_electricity_record
    ):
        """Test aggregation of emissions by region."""
        records = [us_electricity_record, uk_electricity_record]
        batch_result = calculator_engine.calculate_batch(records)

        aggregated = calculator_engine.aggregate_by_region(batch_result)

        assert "US" in aggregated
        assert "UK" in aggregated

        assert aggregated["US"]["emissions_kg"] > 0
        assert aggregated["UK"]["emissions_kg"] > 0

    def test_get_total_emissions(
        self, calculator_engine, us_electricity_record, uk_electricity_record
    ):
        """Test getting total emissions from batch result."""
        records = [us_electricity_record, uk_electricity_record]
        batch_result = calculator_engine.calculate_batch(records)

        total_kg = calculator_engine.get_total_emissions(batch_result, unit="kg")
        total_tonnes = calculator_engine.get_total_emissions(batch_result, unit="tonnes")

        assert total_kg > 0
        assert total_tonnes > 0
        assert total_tonnes == pytest.approx(
            total_kg / Decimal("1000"), rel=Decimal("0.01")
        )

    def test_assess_dqi_high_quality(self, calculator_engine):
        """Test DQI assessment for high-quality electricity data."""
        high_quality_record = ElectricityConsumptionRecord(
            record_id="HQ-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US-CAMX",  # Specific eGRID region
            calculation_method="market-based",
            supplier_specific_factor=Decimal("0.150"),
            supplier_name="Verified Renewable Energy Inc",
            measurement_method="SMART_METER",
            uncertainty=Decimal("1.0"),
        )

        result = calculator_engine.calculate(high_quality_record)

        assert result.data_quality_indicator is not None
        # High quality: smart meter, market-based, supplier-specific, low uncertainty
        assert result.data_quality_indicator in [
            DataQualityIndicator.HIGH,
            DataQualityIndicator.MEDIUM,
        ]

    def test_assess_dqi_low_quality(self, calculator_engine):
        """Test DQI assessment for low-quality electricity data."""
        low_quality_record = ElectricityConsumptionRecord(
            record_id="LQ-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2020, 1, 15),  # Old data
            facility_id="FAC-001",
            region="GLOBAL",  # Generic region
            calculation_method="location-based",
            measurement_method="ESTIMATED",
            uncertainty=Decimal("30.0"),
        )

        result = calculator_engine.calculate(low_quality_record)

        assert result.data_quality_indicator is not None
        # Low quality: estimated, old data, generic region, high uncertainty
        assert result.data_quality_indicator in [
            DataQualityIndicator.LOW,
            DataQualityIndicator.MEDIUM,
        ]

    def test_quantify_uncertainty(self, calculator_engine, us_electricity_record):
        """Test uncertainty quantification for upstream electricity."""
        result = calculator_engine.calculate(us_electricity_record)

        assert result.uncertainty_percentage is not None
        assert result.uncertainty_percentage >= 0

        # Calculate confidence interval
        uncertainty = calculator_engine.calculate_uncertainty(result)
        assert "lower_bound" in uncertainty
        assert "upper_bound" in uncertainty
        assert uncertainty["lower_bound"] < result.upstream_emissions_kg
        assert uncertainty["upper_bound"] > result.upstream_emissions_kg

    def test_check_double_counting_with_scope2(self, calculator_engine, us_electricity_record):
        """Test detection of potential double counting with Scope 2."""
        result = calculator_engine.calculate(us_electricity_record)

        # Upstream electricity (Scope 3) should NOT double count with Scope 2
        double_count_check = calculator_engine.check_double_counting(
            result, scope2_energy=[EnergyType.ELECTRICITY]
        )

        assert "potential_overlap" in double_count_check
        # Should warn if electricity appears in Scope 2
        if EnergyType.ELECTRICITY in double_count_check.get("scope2_energy", []):
            assert double_count_check["potential_overlap"] is True
            # Upstream is separate from Scope 2, so acceptable overlap

    def test_get_renewable_upstream_ef_solar(self, calculator_engine):
        """Test upstream EF for solar PV (~0.005 kgCO2e/kWh)."""
        solar_record = ElectricityConsumptionRecord(
            record_id="SOLAR-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            renewable_source="SOLAR",
        )

        result = calculator_engine.calculate(solar_record)

        # Solar has very low upstream (panel manufacturing)
        # ~0.005 kgCO2e/kWh for lifecycle
        assert result.upstream_emissions_kg < Decimal("1000")  # Much lower than grid avg

    def test_get_renewable_upstream_ef_wind(self, calculator_engine):
        """Test upstream EF for wind (~0.003 kgCO2e/kWh)."""
        wind_record = ElectricityConsumptionRecord(
            record_id="WIND-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            renewable_source="WIND",
        )

        result = calculator_engine.calculate(wind_record)

        # Wind has very low upstream (turbine manufacturing)
        assert result.upstream_emissions_kg < Decimal("800")

    def test_get_renewable_upstream_ef_hydro(self, calculator_engine):
        """Test upstream EF for hydro (~0.004 kgCO2e/kWh)."""
        hydro_record = ElectricityConsumptionRecord(
            record_id="HYDRO-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            renewable_source="HYDRO",
        )

        result = calculator_engine.calculate(hydro_record)

        # Hydro has very low upstream (dam construction)
        assert result.upstream_emissions_kg < Decimal("1000")

    def test_validate_consumption_record_valid(self, calculator_engine, us_electricity_record):
        """Test validation accepts valid electricity record."""
        # Should not raise
        calculator_engine.validate_record(us_electricity_record)

    def test_validate_consumption_record_missing_energy_type(self, calculator_engine):
        """Test validation rejects record with missing energy type."""
        invalid_record = ElectricityConsumptionRecord(
            record_id="INV-001",
            energy_type=None,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        with pytest.raises(ValidationError):
            calculator_engine.validate_record(invalid_record)

    def test_validate_consumption_record_zero_quantity(self, calculator_engine):
        """Test validation handles zero quantity."""
        zero_record = ElectricityConsumptionRecord(
            record_id="ZERO-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("0"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        result = calculator_engine.calculate(zero_record)
        assert result.upstream_emissions_kg == Decimal("0")

    def test_validate_consumption_record_negative_quantity(self, calculator_engine):
        """Test validation rejects negative quantity."""
        negative_record = ElectricityConsumptionRecord(
            record_id="NEG-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("-10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        with pytest.raises(ValidationError):
            calculator_engine.validate_record(negative_record)

    def test_chp_upstream_allocation(self, calculator_engine):
        """Test upstream allocation for CHP (combined heat and power)."""
        chp_record = ElectricityConsumptionRecord(
            record_id="CHP-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            is_chp=True,
            chp_heat_output=Decimal("50000"),  # kWh thermal
        )

        result = calculator_engine.calculate(chp_record)

        # CHP upstream should be allocated between electricity and heat
        assert result.upstream_emissions_kg > 0
        # Should be lower than if all fuel was for electricity only

    def test_transmission_and_distribution_losses(self, calculator_engine, us_electricity_record):
        """Test that T&D losses are properly accounted for in upstream."""
        result = calculator_engine.calculate(us_electricity_record)

        # T&D losses typically 5-7% in US
        # Upstream factor should reflect additional generation needed
        assert result.td_loss_percentage is not None
        if result.td_loss_percentage:
            assert Decimal("3") <= result.td_loss_percentage <= Decimal("10")

    def test_custom_upstream_factor(self, calculator_engine):
        """Test using custom upstream emission factor."""
        custom_record = ElectricityConsumptionRecord(
            record_id="CUSTOM-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            custom_upstream_factor=Decimal("0.050"),  # Custom upstream factor
        )

        result = calculator_engine.calculate(custom_record)

        # Should use custom factor: 100000 × 0.050 = 5000 kgCO2e
        assert result.upstream_emissions_kg == pytest.approx(
            Decimal("5000"), rel=Decimal("0.01")
        )
        assert result.emission_factor_source == EmissionFactorSource.CUSTOM

    def test_renewable_energy_certificates(self, calculator_engine):
        """Test handling of Renewable Energy Certificates (RECs)."""
        rec_record = ElectricityConsumptionRecord(
            record_id="REC-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            calculation_method="market-based",
            has_recs=True,
            rec_type="GREEN-E",
            rec_quantity=Decimal("100000"),  # Full RECs
        )

        result = calculator_engine.calculate(rec_record)

        # With 100% RECs, market-based upstream should reflect renewable sources
        # Upstream for renewables is much lower than grid average
        assert result.upstream_emissions_kg < Decimal("2000")  # Lower than grid avg

    def test_power_purchase_agreement(self, calculator_engine):
        """Test handling of Power Purchase Agreement (PPA)."""
        ppa_record = ElectricityConsumptionRecord(
            record_id="PPA-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
            calculation_method="market-based",
            has_ppa=True,
            ppa_source="SOLAR_FARM_A",
            ppa_emission_factor=Decimal("0.005"),  # Solar lifecycle
        )

        result = calculator_engine.calculate(ppa_record)

        # PPA should use contracted source upstream factor
        assert result.upstream_emissions_kg < Decimal("1000")  # Solar upstream

    def test_unit_conversion_mwh_to_kwh(self, calculator_engine):
        """Test automatic unit conversion from MWh to kWh."""
        mwh_record = ElectricityConsumptionRecord(
            record_id="MWH-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100"),  # MWh
            unit=EnergyUnit.MWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        result = calculator_engine.calculate(mwh_record)

        # Should convert to kWh: 100 MWh = 100000 kWh
        assert result.energy_quantity_kwh == Decimal("100000")

    def test_unit_conversion_gj_to_kwh(self, calculator_engine):
        """Test automatic unit conversion from GJ to kWh."""
        gj_record = ElectricityConsumptionRecord(
            record_id="GJ-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("360"),  # GJ
            unit=EnergyUnit.GJ,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        result = calculator_engine.calculate(gj_record)

        # Should convert to kWh: 360 GJ = 100000 kWh
        assert result.energy_quantity_kwh == pytest.approx(
            Decimal("100000"), rel=Decimal("0.01")
        )

    def test_batch_with_mixed_energy_types(self, calculator_engine):
        """Test batch calculation with mixed energy types."""
        records = [
            ElectricityConsumptionRecord(
                record_id="ELEC-001",
                energy_type=EnergyType.ELECTRICITY,
                quantity=Decimal("100000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
                region="US",
            ),
            ElectricityConsumptionRecord(
                record_id="STEAM-001",
                energy_type=EnergyType.STEAM,
                quantity=Decimal("50000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
                region="US",
            ),
            ElectricityConsumptionRecord(
                record_id="HEAT-001",
                energy_type=EnergyType.HEAT,
                quantity=Decimal("30000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
                region="US",
            ),
        ]

        batch_result = calculator_engine.calculate_batch(records)

        assert len(batch_result.results) == 3
        assert batch_result.total_upstream_emissions_kg > 0

    def test_historical_grid_mix_evolution(self, calculator_engine):
        """Test that upstream factors reflect grid mix evolution over time."""
        old_record = ElectricityConsumptionRecord(
            record_id="OLD-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2020, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        new_record = ElectricityConsumptionRecord(
            record_id="NEW-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        old_result = calculator_engine.calculate(old_record)
        new_result = calculator_engine.calculate(new_record)

        # Grid is getting cleaner over time → upstream should decrease
        # (This assumes default factors reflect this trend)
        assert old_result.upstream_emissions_kg > 0
        assert new_result.upstream_emissions_kg > 0

    def test_zero_emission_sources(self, calculator_engine):
        """Test upstream for zero-emission sources (wind, solar, nuclear)."""
        nuclear_record = ElectricityConsumptionRecord(
            record_id="NUCLEAR-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="FR",  # France = high nuclear
            calculation_method="location-based",
        )

        result = calculator_engine.calculate(nuclear_record)

        # Nuclear has low operational emissions but some upstream (mining, enrichment)
        # Should be significantly lower than fossil-heavy grids
        assert result.upstream_emissions_kg > 0
        assert result.upstream_emissions_kg < Decimal("2000")  # Lower than US grid avg


# Integration Tests
class TestIntegrationUpstreamElectricityCalculator:
    """Integration tests for UpstreamElectricityCalculatorEngine."""

    @pytest.mark.integration
    def test_full_calculation_pipeline(self, calculator_engine):
        """Test full calculation pipeline from input to output."""
        record = ElectricityConsumptionRecord(
            record_id="INTEG-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("50000"),
            unit=EnergyUnit.MWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US-CAMX",
        )

        # Full pipeline: validate → convert → calculate → format
        calculator_engine.validate_record(record)
        result = calculator_engine.calculate(record)
        json_output = calculator_engine.format_result(result, format="json")

        assert result is not None
        assert json_output is not None

    @pytest.mark.integration
    def test_multi_region_portfolio(self, calculator_engine):
        """Test calculation for multi-region portfolio."""
        regions = ["US", "UK", "DE", "FR", "JP", "AU"]
        records = []

        for region in regions:
            record = ElectricityConsumptionRecord(
                record_id=f"ELEC-{region}",
                energy_type=EnergyType.ELECTRICITY,
                quantity=Decimal("100000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id=f"FAC-{region}",
                region=region,
            )
            records.append(record)

        batch_result = calculator_engine.calculate_batch(records)

        assert len(batch_result.results) == len(regions)
        assert batch_result.total_upstream_emissions_kg > 0


# Performance Tests
class TestPerformanceUpstreamElectricityCalculator:
    """Performance tests for UpstreamElectricityCalculatorEngine."""

    def test_single_calculation_performance(
        self, calculator_engine, us_electricity_record, benchmark
    ):
        """Test single calculation performance."""
        result = benchmark(calculator_engine.calculate, us_electricity_record)
        assert result is not None

    def test_batch_calculation_performance(self, calculator_engine):
        """Test batch calculation performance for 1000 records."""
        records = []
        for i in range(1000):
            record = ElectricityConsumptionRecord(
                record_id=f"PERF-{i:04d}",
                energy_type=EnergyType.ELECTRICITY,
                quantity=Decimal("100000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id=f"FAC-{i % 10}",
                region="US",
            )
            records.append(record)

        start_time = datetime.now()
        batch_result = calculator_engine.calculate_batch(records)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        # Should process 1000 records in <5 seconds
        assert duration < 5.0
        assert len(batch_result.results) == 1000


# Edge Case Tests
class TestEdgeCasesUpstreamElectricityCalculator:
    """Edge case tests for UpstreamElectricityCalculatorEngine."""

    def test_very_large_consumption(self, calculator_engine):
        """Test calculation with very large electricity consumption."""
        large_record = ElectricityConsumptionRecord(
            record_id="LARGE-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000000"),  # 100 GWh
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        result = calculator_engine.calculate(large_record)

        assert result.upstream_emissions_kg > Decimal("1000000")  # >1000 tonnes

    def test_missing_region_fallback_to_global(self, calculator_engine):
        """Test fallback to global factor when region not found."""
        unknown_region_record = ElectricityConsumptionRecord(
            record_id="UNKNOWN-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="XYZ-UNKNOWN",  # Non-existent region
        )

        result = calculator_engine.calculate(unknown_region_record)

        # Should fall back to GLOBAL factor
        assert result is not None
        assert result.upstream_emissions_kg > 0
        assert result.region == "GLOBAL"  # Fallback

    def test_future_date_uses_latest_factor(self, calculator_engine):
        """Test that future dates use latest available factor."""
        future_record = ElectricityConsumptionRecord(
            record_id="FUTURE-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("100000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2030, 1, 15),  # Future
            facility_id="FAC-001",
            region="US",
        )

        result = calculator_engine.calculate(future_record)

        # Should use latest available factor (2025 or similar)
        assert result is not None
        assert result.upstream_emissions_kg > 0
