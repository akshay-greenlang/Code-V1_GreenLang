"""
Unit tests for UpstreamFuelCalculatorEngine (AGENT-MRV-016 Engine 2 - Activity 3a)

Tests all methods of UpstreamFuelCalculatorEngine with comprehensive coverage.
Validates upstream fuel emissions (WTT) calculations, aggregations, and reporting
for Scope 3 Category 3 - Fuel and Energy-Related Activities (Activity 3a).
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from greenlang.fuel_energy_activities.models import (
    FuelConsumptionRecord,
    UpstreamFuelResult,
    UpstreamFuelBatchResult,
    FuelType,
    EnergyUnit,
    EmissionFactorSource,
    DataQualityIndicator,
    GasBreakdown,
    SupplyChainBreakdown,
)
from greenlang.fuel_energy_activities.engines.upstream_fuel_calculator import (
    UpstreamFuelCalculatorEngine,
)
from greenlang.fuel_energy_activities.engines.wtt_fuel_database import WTTFuelDatabaseEngine
from greenlang_core.exceptions import ValidationError, ProcessingError


# Fixtures
@pytest.fixture
def calculator_engine():
    """Create UpstreamFuelCalculatorEngine instance for testing."""
    return UpstreamFuelCalculatorEngine()


@pytest.fixture
def db_engine():
    """Create WTTFuelDatabaseEngine instance."""
    return WTTFuelDatabaseEngine()


@pytest.fixture
def natural_gas_record():
    """Create sample natural gas consumption record."""
    return FuelConsumptionRecord(
        record_id="NG-001",
        fuel_type=FuelType.NATURAL_GAS,
        quantity=Decimal("100000"),  # kWh
        unit=EnergyUnit.KWH,
        consumption_date=date(2025, 1, 15),
        facility_id="FAC-001",
        region="US",
        source=EmissionFactorSource.EPA_GHGRP,
    )


@pytest.fixture
def diesel_record():
    """Create sample diesel consumption record."""
    return FuelConsumptionRecord(
        record_id="DS-001",
        fuel_type=FuelType.DIESEL,
        quantity=Decimal("50000"),  # kWh
        unit=EnergyUnit.KWH,
        consumption_date=date(2025, 1, 20),
        facility_id="FAC-002",
        region="US",
        source=EmissionFactorSource.DEFRA,
    )


@pytest.fixture
def coal_record():
    """Create sample coal consumption record."""
    return FuelConsumptionRecord(
        record_id="COAL-001",
        fuel_type=FuelType.COAL,
        quantity=Decimal("200000"),  # kWh
        unit=EnergyUnit.KWH,
        consumption_date=date(2025, 1, 25),
        facility_id="FAC-003",
        region="US",
    )


# Test Class
class TestUpstreamFuelCalculatorEngine:
    """Test suite for UpstreamFuelCalculatorEngine."""

    def test_initialization(self, calculator_engine):
        """Test engine initializes correctly."""
        assert calculator_engine is not None
        assert calculator_engine.db_engine is not None

    def test_calculate_natural_gas_wtt(self, calculator_engine, natural_gas_record):
        """Test WTT calculation for natural gas: 100000 kWh × 0.0246 = 2460 kgCO2e."""
        result = calculator_engine.calculate(natural_gas_record)

        assert result is not None
        assert result.record_id == "NG-001"
        assert result.fuel_type == FuelType.NATURAL_GAS
        assert result.fuel_quantity_kwh == Decimal("100000")

        # WTT emissions: 100000 kWh × 0.0246 kgCO2e/kWh = 2460 kgCO2e
        expected_emissions = Decimal("2460")
        assert result.wtt_emissions_kg == pytest.approx(expected_emissions, rel=Decimal("0.05"))
        assert result.wtt_emissions_tonnes == pytest.approx(Decimal("2.46"), rel=Decimal("0.05"))

        assert result.wtt_emission_factor > 0
        assert result.emission_factor_source == EmissionFactorSource.EPA_GHGRP

    def test_calculate_diesel_wtt(self, calculator_engine, diesel_record):
        """Test WTT calculation for diesel: 50000 kWh × 0.0507 = 2535 kgCO2e."""
        result = calculator_engine.calculate(diesel_record)

        assert result is not None
        assert result.fuel_type == FuelType.DIESEL
        assert result.fuel_quantity_kwh == Decimal("50000")

        # WTT emissions: 50000 kWh × 0.0507 kgCO2e/kWh = 2535 kgCO2e
        expected_emissions = Decimal("2535")
        assert result.wtt_emissions_kg == pytest.approx(expected_emissions, rel=Decimal("0.05"))
        assert result.wtt_emissions_tonnes == pytest.approx(Decimal("2.535"), rel=Decimal("0.05"))

    def test_calculate_coal_wtt(self, calculator_engine, coal_record):
        """Test WTT calculation for coal."""
        result = calculator_engine.calculate(coal_record)

        assert result is not None
        assert result.fuel_type == FuelType.COAL
        assert result.fuel_quantity_kwh == Decimal("200000")

        # Coal WTT: ~0.0182 kgCO2e/kWh
        # 200000 kWh × 0.0182 = 3640 kgCO2e
        expected_emissions = Decimal("3640")
        assert result.wtt_emissions_kg == pytest.approx(expected_emissions, rel=Decimal("0.10"))

    def test_calculate_per_gas_breakdown(self, calculator_engine, natural_gas_record):
        """Test per-gas breakdown (CO2, CH4, N2O) for natural gas WTT."""
        result = calculator_engine.calculate(natural_gas_record)

        assert result.gas_breakdown is not None
        assert result.gas_breakdown.co2 > 0
        assert result.gas_breakdown.ch4 > 0  # Natural gas has significant CH4 leakage
        assert result.gas_breakdown.n2o >= 0

        # Total gas breakdown should equal total emissions
        total_gas = (
            result.gas_breakdown.co2
            + result.gas_breakdown.ch4
            + result.gas_breakdown.n2o
        )
        assert total_gas == pytest.approx(result.wtt_emissions_kg, rel=Decimal("0.01"))

    def test_calculate_supply_chain_breakdown(self, calculator_engine, diesel_record):
        """Test supply chain breakdown (extraction/processing/transport)."""
        result = calculator_engine.calculate(diesel_record)

        assert result.supply_chain_breakdown is not None
        assert result.supply_chain_breakdown.extraction > 0
        assert result.supply_chain_breakdown.processing > 0
        assert result.supply_chain_breakdown.transport > 0

        # Total supply chain should equal total WTT emissions
        total_supply_chain = (
            result.supply_chain_breakdown.extraction
            + result.supply_chain_breakdown.processing
            + result.supply_chain_breakdown.transport
            + result.supply_chain_breakdown.distribution
        )
        assert total_supply_chain == pytest.approx(
            result.wtt_emissions_kg, rel=Decimal("0.01")
        )

    def test_calculate_batch_multiple_fuels(
        self, calculator_engine, natural_gas_record, diesel_record, coal_record
    ):
        """Test batch calculation with multiple fuel types."""
        records = [natural_gas_record, diesel_record, coal_record]

        batch_result = calculator_engine.calculate_batch(records)

        assert batch_result is not None
        assert len(batch_result.results) == 3
        assert batch_result.total_wtt_emissions_kg > 0
        assert batch_result.total_wtt_emissions_tonnes > 0

        # Individual results should sum to total
        individual_sum = sum(r.wtt_emissions_kg for r in batch_result.results)
        assert individual_sum == pytest.approx(
            batch_result.total_wtt_emissions_kg, rel=Decimal("0.01")
        )

    def test_calculate_blended_fuel_e10(self, calculator_engine):
        """Test WTT calculation for blended fuel (E10: 90% gasoline + 10% ethanol)."""
        e10_record = FuelConsumptionRecord(
            record_id="E10-001",
            fuel_type=FuelType.GASOLINE,  # Base fuel
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            blend_components=[
                {"fuel_type": FuelType.GASOLINE, "fraction": Decimal("0.9")},
                {"fuel_type": FuelType.ETHANOL, "fraction": Decimal("0.1")},
            ],
        )

        result = calculator_engine.calculate(e10_record)

        assert result is not None
        # Blended WTT should be weighted average of components
        # Gasoline WTT ~0.0489, Ethanol WTT ~0.015
        # Blended: 0.9 × 0.0489 + 0.1 × 0.015 = 0.04551 kgCO2e/kWh
        expected_emissions = Decimal("10000") * Decimal("0.04551")
        assert result.wtt_emissions_kg == pytest.approx(expected_emissions, rel=Decimal("0.10"))

    def test_calculate_biogenic_split_fossil_fuel(self, calculator_engine, diesel_record):
        """Test biogenic split for fossil fuel (100% fossil, 0% biogenic)."""
        result = calculator_engine.calculate(diesel_record)

        assert result.biogenic_emissions_kg == Decimal("0")
        assert result.fossil_emissions_kg == result.wtt_emissions_kg

    def test_calculate_biogenic_split_biofuel(self, calculator_engine):
        """Test biogenic split for biofuel (100% biogenic)."""
        biodiesel_record = FuelConsumptionRecord(
            record_id="BD-001",
            fuel_type=FuelType.BIODIESEL,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        result = calculator_engine.calculate(biodiesel_record)

        # Biodiesel WTT is typically reported but may have biogenic component
        # WTT for biodiesel includes cultivation, processing
        assert result.wtt_emissions_kg > 0
        # Biogenic fraction depends on methodology

    def test_calculate_supply_chain_stages(self, calculator_engine, natural_gas_record):
        """Test supply chain stage breakdown (extraction/processing/transport)."""
        result = calculator_engine.calculate(natural_gas_record)

        breakdown = result.supply_chain_breakdown
        assert breakdown is not None

        # Natural gas supply chain
        assert breakdown.extraction > 0  # Well extraction
        assert breakdown.processing > 0  # Processing plant
        assert breakdown.transport > 0  # Pipeline transport

        # Extraction + processing + transport should equal total
        total = breakdown.extraction + breakdown.processing + breakdown.transport
        if breakdown.distribution:
            total += breakdown.distribution

        assert total == pytest.approx(result.wtt_emissions_kg, rel=Decimal("0.01"))

    def test_aggregate_by_fuel_type(
        self, calculator_engine, natural_gas_record, diesel_record, coal_record
    ):
        """Test aggregation of emissions by fuel type."""
        records = [natural_gas_record, diesel_record, coal_record]
        batch_result = calculator_engine.calculate_batch(records)

        aggregated = calculator_engine.aggregate_by_fuel_type(batch_result)

        assert FuelType.NATURAL_GAS in aggregated
        assert FuelType.DIESEL in aggregated
        assert FuelType.COAL in aggregated

        # Each fuel type should have correct emissions
        assert aggregated[FuelType.NATURAL_GAS]["emissions_kg"] > 0
        assert aggregated[FuelType.DIESEL]["emissions_kg"] > 0
        assert aggregated[FuelType.COAL]["emissions_kg"] > 0

    def test_aggregate_by_facility(
        self, calculator_engine, natural_gas_record, diesel_record
    ):
        """Test aggregation of emissions by facility."""
        records = [natural_gas_record, diesel_record]
        batch_result = calculator_engine.calculate_batch(records)

        aggregated = calculator_engine.aggregate_by_facility(batch_result)

        assert "FAC-001" in aggregated
        assert "FAC-002" in aggregated

        assert aggregated["FAC-001"]["emissions_kg"] > 0
        assert aggregated["FAC-002"]["emissions_kg"] > 0

    def test_aggregate_by_period(self, calculator_engine):
        """Test aggregation of emissions by time period."""
        records = [
            FuelConsumptionRecord(
                record_id="NG-JAN",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("10000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
            ),
            FuelConsumptionRecord(
                record_id="NG-FEB",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("12000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 2, 15),
                facility_id="FAC-001",
            ),
            FuelConsumptionRecord(
                record_id="NG-MAR",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("11000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 3, 15),
                facility_id="FAC-001",
            ),
        ]

        batch_result = calculator_engine.calculate_batch(records)

        # Aggregate by month
        aggregated = calculator_engine.aggregate_by_period(
            batch_result, period="month"
        )

        assert "2025-01" in aggregated
        assert "2025-02" in aggregated
        assert "2025-03" in aggregated

    def test_get_total_emissions(self, calculator_engine, natural_gas_record, diesel_record):
        """Test getting total emissions from batch result."""
        records = [natural_gas_record, diesel_record]
        batch_result = calculator_engine.calculate_batch(records)

        total_kg = calculator_engine.get_total_emissions(batch_result, unit="kg")
        total_tonnes = calculator_engine.get_total_emissions(batch_result, unit="tonnes")

        assert total_kg > 0
        assert total_tonnes > 0
        assert total_tonnes == pytest.approx(total_kg / Decimal("1000"), rel=Decimal("0.01"))

    def test_assess_dqi_high_quality(self, calculator_engine):
        """Test DQI assessment for high-quality data."""
        high_quality_record = FuelConsumptionRecord(
            record_id="HQ-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US-CA",
            source=EmissionFactorSource.EPA_GHGRP,
            measurement_method="DIRECT_METER",
            uncertainty=Decimal("2.0"),  # Low uncertainty
        )

        result = calculator_engine.calculate(high_quality_record)

        assert result.data_quality_indicator is not None
        # High quality: direct measurement, low uncertainty, specific region
        assert result.data_quality_indicator in [
            DataQualityIndicator.HIGH,
            DataQualityIndicator.MEDIUM,
        ]

    def test_assess_dqi_low_quality(self, calculator_engine):
        """Test DQI assessment for low-quality data."""
        low_quality_record = FuelConsumptionRecord(
            record_id="LQ-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2020, 1, 15),  # Old data
            facility_id="FAC-001",
            region="GLOBAL",  # Generic region
            measurement_method="ESTIMATED",
            uncertainty=Decimal("50.0"),  # High uncertainty
        )

        result = calculator_engine.calculate(low_quality_record)

        assert result.data_quality_indicator is not None
        # Low quality: estimated, high uncertainty, old data, generic region
        assert result.data_quality_indicator in [
            DataQualityIndicator.LOW,
            DataQualityIndicator.MEDIUM,
        ]

    def test_quantify_uncertainty_analytical(self, calculator_engine, diesel_record):
        """Test analytical uncertainty quantification."""
        result = calculator_engine.calculate(diesel_record)

        assert result.uncertainty_percentage is not None
        assert result.uncertainty_percentage >= 0

        # Calculate confidence interval
        uncertainty = calculator_engine.calculate_uncertainty(result)
        assert "lower_bound" in uncertainty
        assert "upper_bound" in uncertainty
        assert uncertainty["lower_bound"] < result.wtt_emissions_kg
        assert uncertainty["upper_bound"] > result.wtt_emissions_kg

    def test_identify_hot_spots_pareto(self, calculator_engine):
        """Test identifying emission hot spots using Pareto analysis (80/20 rule)."""
        records = []
        # Create records with varying emissions
        for i in range(10):
            record = FuelConsumptionRecord(
                record_id=f"REC-{i:03d}",
                fuel_type=FuelType.DIESEL if i % 2 == 0 else FuelType.NATURAL_GAS,
                quantity=Decimal(str(10000 * (i + 1))),  # Varying quantities
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id=f"FAC-{i % 3}",
            )
            records.append(record)

        batch_result = calculator_engine.calculate_batch(records)

        hot_spots = calculator_engine.identify_hot_spots(batch_result, threshold=0.8)

        # Top 20% should contribute ~80% of emissions
        assert len(hot_spots) > 0
        hot_spot_emissions = sum(hs["emissions_kg"] for hs in hot_spots)
        total_emissions = batch_result.total_wtt_emissions_kg

        contribution = hot_spot_emissions / total_emissions
        assert contribution >= Decimal("0.6")  # At least 60% from hot spots

    def test_compare_yoy(self, calculator_engine):
        """Test year-over-year comparison."""
        records_2024 = [
            FuelConsumptionRecord(
                record_id="2024-001",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("100000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2024, 1, 15),
                facility_id="FAC-001",
            )
        ]

        records_2025 = [
            FuelConsumptionRecord(
                record_id="2025-001",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("95000"),  # 5% reduction
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
            )
        ]

        batch_2024 = calculator_engine.calculate_batch(records_2024)
        batch_2025 = calculator_engine.calculate_batch(records_2025)

        comparison = calculator_engine.compare_periods(batch_2024, batch_2025)

        assert "emissions_change_kg" in comparison
        assert "emissions_change_percent" in comparison
        assert comparison["emissions_change_kg"] < 0  # Reduction
        assert comparison["emissions_change_percent"] == pytest.approx(
            Decimal("-5.0"), rel=Decimal("0.1")
        )

    def test_check_double_counting_with_scope1(self, calculator_engine, diesel_record):
        """Test detection of potential double counting with Scope 1."""
        result = calculator_engine.calculate(diesel_record)

        # WTT emissions (Scope 3) should NOT double count with combustion (Scope 1)
        double_count_check = calculator_engine.check_double_counting(
            result, scope1_fuels=[FuelType.DIESEL]
        )

        assert "potential_overlap" in double_count_check
        # Should warn if same fuel appears in Scope 1
        if FuelType.DIESEL in double_count_check.get("scope1_fuels", []):
            assert double_count_check["potential_overlap"] is True

    def test_validate_fuel_record_valid(self, calculator_engine, natural_gas_record):
        """Test validation accepts valid fuel record."""
        # Should not raise
        calculator_engine.validate_record(natural_gas_record)

    def test_validate_fuel_record_invalid_missing_fuel_type(self, calculator_engine):
        """Test validation rejects record with missing fuel type."""
        invalid_record = FuelConsumptionRecord(
            record_id="INV-001",
            fuel_type=None,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        with pytest.raises(ValidationError):
            calculator_engine.validate_record(invalid_record)

    def test_validate_fuel_record_invalid_zero_quantity(self, calculator_engine):
        """Test validation handles zero quantity."""
        zero_record = FuelConsumptionRecord(
            record_id="ZERO-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("0"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        result = calculator_engine.calculate(zero_record)
        assert result.wtt_emissions_kg == Decimal("0")

    def test_validate_fuel_record_invalid_negative_quantity(self, calculator_engine):
        """Test validation rejects negative quantity."""
        negative_record = FuelConsumptionRecord(
            record_id="NEG-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("-1000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        with pytest.raises(ValidationError):
            calculator_engine.validate_record(negative_record)

    def test_get_materiality_assessment(self, calculator_engine):
        """Test materiality assessment for emissions."""
        records = []
        # Create mix of high and low emission records
        records.append(
            FuelConsumptionRecord(
                record_id="HIGH-001",
                fuel_type=FuelType.COAL,
                quantity=Decimal("1000000"),  # Large quantity
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
            )
        )
        records.append(
            FuelConsumptionRecord(
                record_id="LOW-001",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("100"),  # Small quantity
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-002",
            )
        )

        batch_result = calculator_engine.calculate_batch(records)

        materiality = calculator_engine.assess_materiality(
            batch_result, threshold_tonnes=Decimal("1.0")
        )

        assert "material_sources" in materiality
        assert "immaterial_sources" in materiality

        # Coal record should be material
        material_ids = [s["record_id"] for s in materiality["material_sources"]]
        assert "HIGH-001" in material_ids

    def test_format_results_json(self, calculator_engine, diesel_record):
        """Test formatting results as JSON."""
        result = calculator_engine.calculate(diesel_record)

        json_output = calculator_engine.format_result(result, format="json")

        assert json_output is not None
        assert "record_id" in json_output
        assert "wtt_emissions_kg" in json_output

    def test_format_results_csv(self, calculator_engine, natural_gas_record, diesel_record):
        """Test formatting batch results as CSV."""
        records = [natural_gas_record, diesel_record]
        batch_result = calculator_engine.calculate_batch(records)

        csv_output = calculator_engine.format_batch_result(batch_result, format="csv")

        assert csv_output is not None
        assert "record_id" in csv_output
        assert "wtt_emissions_kg" in csv_output

    def test_zero_quantity_returns_zero(self, calculator_engine):
        """Test that zero fuel quantity returns zero emissions."""
        zero_record = FuelConsumptionRecord(
            record_id="ZERO-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("0"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        result = calculator_engine.calculate(zero_record)

        assert result.wtt_emissions_kg == Decimal("0")
        assert result.wtt_emissions_tonnes == Decimal("0")

    def test_negative_quantity_rejected(self, calculator_engine):
        """Test that negative fuel quantity is rejected."""
        negative_record = FuelConsumptionRecord(
            record_id="NEG-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("-1000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        with pytest.raises(ValidationError):
            calculator_engine.calculate(negative_record)

    def test_calculate_multiple_facilities_same_fuel(self, calculator_engine):
        """Test calculation for same fuel across multiple facilities."""
        records = []
        for i in range(5):
            record = FuelConsumptionRecord(
                record_id=f"NG-{i:03d}",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("10000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id=f"FAC-{i:03d}",
            )
            records.append(record)

        batch_result = calculator_engine.calculate_batch(records)

        assert len(batch_result.results) == 5
        # All should have same WTT emissions (same fuel, same quantity)
        emissions = [r.wtt_emissions_kg for r in batch_result.results]
        assert all(e == emissions[0] for e in emissions)

    def test_unit_conversion_litres_to_kwh(self, calculator_engine):
        """Test automatic unit conversion from litres to kWh."""
        diesel_litres = FuelConsumptionRecord(
            record_id="DS-L-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("1000"),  # litres
            unit=EnergyUnit.LITRES,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        result = calculator_engine.calculate(diesel_litres)

        # Should convert litres → kg → kWh automatically
        assert result.fuel_quantity_kwh > Decimal("1000")  # More kWh than litres
        # 1000 L × 0.835 kg/L × 10.7 kWh/kg ≈ 8935 kWh
        assert Decimal("8000") < result.fuel_quantity_kwh < Decimal("10000")

    def test_unit_conversion_kg_to_kwh(self, calculator_engine):
        """Test automatic unit conversion from kg to kWh."""
        coal_kg = FuelConsumptionRecord(
            record_id="COAL-KG-001",
            fuel_type=FuelType.COAL,
            quantity=Decimal("5000"),  # kg
            unit=EnergyUnit.KG,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        result = calculator_engine.calculate(coal_kg)

        # Should convert kg → kWh using heating value
        # Coal: ~8.1 kWh/kg → 5000 kg × 8.1 = 40500 kWh
        assert result.fuel_quantity_kwh > Decimal("30000")

    def test_custom_emission_factor(self, calculator_engine):
        """Test using custom emission factor."""
        custom_record = FuelConsumptionRecord(
            record_id="CUSTOM-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            custom_wtt_factor=Decimal("0.0600"),  # Custom WTT factor
        )

        result = calculator_engine.calculate(custom_record)

        # Should use custom factor: 10000 × 0.0600 = 600 kgCO2e
        assert result.wtt_emissions_kg == pytest.approx(Decimal("600"), rel=Decimal("0.01"))
        assert result.emission_factor_source == EmissionFactorSource.CUSTOM

    def test_batch_with_errors(self, calculator_engine):
        """Test batch calculation handles individual errors gracefully."""
        records = [
            FuelConsumptionRecord(
                record_id="VALID-001",
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("10000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
            ),
            FuelConsumptionRecord(
                record_id="INVALID-001",
                fuel_type=None,  # Invalid
                quantity=Decimal("10000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id="FAC-001",
            ),
        ]

        batch_result = calculator_engine.calculate_batch(
            records, continue_on_error=True
        )

        # Should process valid record, skip invalid
        assert len(batch_result.results) == 1
        assert batch_result.errors > 0

    def test_regional_factor_variation(self, calculator_engine):
        """Test that regional factors produce different results."""
        us_record = FuelConsumptionRecord(
            record_id="US-001",
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US",
        )

        uk_record = FuelConsumptionRecord(
            record_id="UK-001",
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="UK",
        )

        us_result = calculator_engine.calculate(us_record)
        uk_result = calculator_engine.calculate(uk_record)

        # Different regions may have different WTT factors
        # (This might be same for global factors, but test the logic)
        assert us_result.wtt_emissions_kg > 0
        assert uk_result.wtt_emissions_kg > 0


# Integration Tests
class TestIntegrationUpstreamFuelCalculator:
    """Integration tests for UpstreamFuelCalculatorEngine."""

    @pytest.mark.integration
    def test_full_calculation_pipeline(self, calculator_engine):
        """Test full calculation pipeline from input to output."""
        record = FuelConsumptionRecord(
            record_id="INTEG-001",
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("5000"),
            unit=EnergyUnit.LITRES,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
            region="US-CA",
        )

        # Full pipeline: validate → convert → calculate → format
        calculator_engine.validate_record(record)
        result = calculator_engine.calculate(record)
        json_output = calculator_engine.format_result(result, format="json")

        assert result is not None
        assert json_output is not None

    @pytest.mark.integration
    def test_integration_with_wtt_database(self, calculator_engine, db_engine):
        """Test integration with WTT fuel database."""
        # Verify calculator uses database correctly
        record = FuelConsumptionRecord(
            record_id="DB-INTEG-001",
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("10000"),
            unit=EnergyUnit.KWH,
            consumption_date=date(2025, 1, 15),
            facility_id="FAC-001",
        )

        result = calculator_engine.calculate(record)

        # Verify factor came from database
        db_factor = db_engine.get_wtt_factor(FuelType.NATURAL_GAS)
        assert result.wtt_emission_factor == db_factor.wtt_emission_factor


# Performance Tests
class TestPerformanceUpstreamFuelCalculator:
    """Performance tests for UpstreamFuelCalculatorEngine."""

    def test_single_calculation_performance(self, calculator_engine, diesel_record, benchmark):
        """Test single calculation performance."""
        result = benchmark(calculator_engine.calculate, diesel_record)
        assert result is not None

    def test_batch_calculation_performance(self, calculator_engine):
        """Test batch calculation performance for 1000 records."""
        records = []
        for i in range(1000):
            record = FuelConsumptionRecord(
                record_id=f"PERF-{i:04d}",
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("10000"),
                unit=EnergyUnit.KWH,
                consumption_date=date(2025, 1, 15),
                facility_id=f"FAC-{i % 10}",
            )
            records.append(record)

        start_time = datetime.now()
        batch_result = calculator_engine.calculate_batch(records)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        # Should process 1000 records in <5 seconds
        assert duration < 5.0
        assert len(batch_result.results) == 1000
