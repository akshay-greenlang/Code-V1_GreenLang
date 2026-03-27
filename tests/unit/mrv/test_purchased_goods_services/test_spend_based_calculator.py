"""
Test suite for SpendBasedCalculatorEngine - AGENT-MRV-014

This module tests the SpendBasedCalculatorEngine for the Purchased Goods & Services Agent.
Tests cover spend-based EEIO calculations, NAICS resolution, currency conversion,
margin removal, DQI scoring, aggregation, and uncertainty quantification.

Coverage target: 85%+
Test count: 60+ tests
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional

from greenlang.agents.mrv.purchased_goods_services.spend_based_calculator import (
    SpendBasedCalculatorEngine,
    SpendRecord,
    SpendCalculationResult,
    SpendAggregation,
    DQIScore,
    UncertaintyRange
)
from greenlang.agents.mrv.purchased_goods_services.procurement_database import (
    ProcurementDatabaseEngine
)


class TestSpendBasedCalculatorSingleton:
    """Test singleton pattern for SpendBasedCalculatorEngine."""

    def test_singleton_creation(self):
        """Test that engine can be created."""
        engine = SpendBasedCalculatorEngine()
        assert engine is not None
        assert isinstance(engine, SpendBasedCalculatorEngine)

    def test_singleton_identity(self):
        """Test that multiple calls return same instance."""
        engine1 = SpendBasedCalculatorEngine()
        engine2 = SpendBasedCalculatorEngine()
        assert engine1 is engine2

    def test_singleton_reset(self):
        """Test that singleton can be reset."""
        engine1 = SpendBasedCalculatorEngine()
        SpendBasedCalculatorEngine._instance = None
        engine2 = SpendBasedCalculatorEngine()
        assert engine1 is not engine2


class TestSingleCalculation:
    """Test single spend-based calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_basic_usd_spend(self, engine):
        """Test basic spend calculation in USD."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-001",
            description="Electronic components"
        )
        result = engine.calculate(record)
        assert result is not None
        assert result.total_emissions > Decimal("0")
        assert result.currency == "USD"
        assert result.naics_code == "334111"
        assert result.calculation_method == "spend_based_eeio"

    def test_eur_conversion(self, engine):
        """Test EUR to USD conversion in calculation."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="EUR",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-002",
            description="Electronic components"
        )
        result = engine.calculate(record)
        assert result is not None
        assert result.total_emissions > Decimal("0")
        assert result.original_currency == "EUR"
        assert result.currency == "USD"  # Converted
        assert result.spend_amount_usd > Decimal("10000.00")  # EUR > USD

    def test_margin_removal(self, engine):
        """Test that purchaser margins are removed."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",  # Has retail/transport margins
            spend_year=2023,
            supplier_id="SUP-003",
            description="Electronic components"
        )
        result = engine.calculate(record)
        assert result.producer_price < result.spend_amount_usd
        assert result.margin_removed > Decimal("0")
        assert result.margin_rate > Decimal("0")

    def test_cpi_deflation(self, engine):
        """Test CPI deflation to EEIO base year."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-004",
            description="Electronic components"
        )
        result = engine.calculate(record)
        # 2023 dollars deflated to 2017 (USEEIO base year)
        assert result.deflated_spend < result.producer_price
        assert result.base_year == 2017

    def test_full_pipeline(self, engine):
        """Test complete calculation pipeline."""
        record = SpendRecord(
            spend_amount=Decimal("50000.00"),
            currency="EUR",
            naics_code="331110",  # Steel manufacturing
            spend_year=2022,
            supplier_id="SUP-005",
            description="Steel products"
        )
        result = engine.calculate(record)

        # Verify all steps executed
        assert result.original_currency == "EUR"
        assert result.currency == "USD"
        assert result.spend_amount_usd > Decimal("50000.00")
        assert result.producer_price < result.spend_amount_usd
        assert result.deflated_spend < result.producer_price
        assert result.emission_factor > Decimal("0")
        assert result.total_emissions > Decimal("0")

        # Verify formula: emissions = deflated_spend * emission_factor
        expected_emissions = result.deflated_spend * result.emission_factor
        tolerance = Decimal("0.01")
        assert abs(result.total_emissions - expected_emissions) < tolerance

    def test_zero_spend(self, engine):
        """Test handling of zero spend amount."""
        record = SpendRecord(
            spend_amount=Decimal("0.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-006",
            description="Zero spend"
        )
        result = engine.calculate(record)
        assert result.total_emissions == Decimal("0")

    def test_missing_naics(self, engine):
        """Test handling of missing NAICS code."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code=None,
            spend_year=2023,
            supplier_id="SUP-007",
            description="Unknown category"
        )
        with pytest.raises(ValueError, match="NAICS code required"):
            engine.calculate(record)

    def test_unknown_currency(self, engine):
        """Test handling of unknown currency."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="XYZ",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-008",
            description="Unknown currency"
        )
        with pytest.raises(ValueError, match="currency not supported|Exchange rate not found"):
            engine.calculate(record)

    def test_negative_spend_credit(self, engine):
        """Test handling of negative spend (credit/return)."""
        record = SpendRecord(
            spend_amount=Decimal("-5000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-009",
            description="Return credit"
        )
        result = engine.calculate(record)
        # Negative spend should produce negative emissions
        assert result.total_emissions < Decimal("0")

    def test_very_large_spend(self, engine):
        """Test handling of very large spend amounts."""
        record = SpendRecord(
            spend_amount=Decimal("100000000.00"),  # $100M
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-010",
            description="Large purchase"
        )
        result = engine.calculate(record)
        assert result.total_emissions > Decimal("0")
        # Should handle large numbers without overflow
        assert result.total_emissions < Decimal("1000000000")  # Sanity check

    def test_high_precision_calculation(self, engine):
        """Test calculation maintains high precision."""
        record = SpendRecord(
            spend_amount=Decimal("12345.67"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-011",
            description="Precision test"
        )
        result = engine.calculate(record)
        # Result should preserve precision
        assert result.total_emissions.as_tuple().exponent <= -2  # At least 2 decimal places


class TestNAICSResolution:
    """Test NAICS code resolution from various classification systems."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_explicit_naics(self, engine):
        """Test calculation with explicit NAICS code."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-020",
            description="Explicit NAICS"
        )
        result = engine.calculate(record)
        assert result.naics_code == "334111"

    def test_resolve_from_nace(self, engine):
        """Test NAICS resolution from NACE code."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            nace_code="26.11",  # Electronics
            spend_year=2023,
            supplier_id="SUP-021",
            description="NACE code"
        )
        result = engine.calculate(record)
        assert result.naics_code.startswith("334")

    def test_resolve_from_unspsc(self, engine):
        """Test NAICS resolution from UNSPSC code."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            unspsc_code="43211503",  # Computer equipment
            spend_year=2023,
            supplier_id="SUP-022",
            description="UNSPSC code"
        )
        result = engine.calculate(record)
        assert result.naics_code.startswith("334")

    def test_resolve_fallback_6_to_4_digit(self, engine):
        """Test fallback from 6-digit to 4-digit NAICS."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334119",  # Specific 6-digit
            spend_year=2023,
            supplier_id="SUP-023",
            description="Fallback test"
        )
        result = engine.calculate(record)
        # Should resolve to 4-digit or 2-digit if 6-digit not found
        assert result.naics_code is not None
        assert len(result.naics_code) <= 6

    def test_resolve_description_keyword(self, engine):
        """Test NAICS resolution from description keywords."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            description="Computer hardware and electronics",
            spend_year=2023,
            supplier_id="SUP-024"
        )
        result = engine.calculate_with_keyword_matching(record)
        # Should identify electronics category
        assert result.naics_code.startswith("334") or result.naics_code.startswith("3")

    def test_multiple_resolution_methods(self, engine):
        """Test priority of resolution methods."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",  # Explicit NAICS
            nace_code="26.11",
            unspsc_code="43211503",
            spend_year=2023,
            supplier_id="SUP-025",
            description="Multiple codes"
        )
        result = engine.calculate(record)
        # Should prioritize explicit NAICS
        assert result.naics_code == "334111"


class TestEEIOFactorResolution:
    """Test EEIO emission factor resolution and fallback."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_exact_match_6digit(self, engine):
        """Test exact match for 6-digit NAICS."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-030",
            description="Exact match"
        )
        result = engine.calculate(record)
        assert result.naics_code == "334111"
        assert result.emission_factor > Decimal("0")

    def test_progressive_fallback(self, engine):
        """Test progressive fallback 6→4→2 digit."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="999999",  # Non-existent
            spend_year=2023,
            supplier_id="SUP-031",
            description="Fallback test"
        )
        result = engine.calculate(record)
        # Should fallback to 2-digit or economy-wide
        assert result.naics_code is not None
        assert len(result.naics_code) <= 2

    def test_missing_ef_economy_wide(self, engine):
        """Test economy-wide fallback when all else fails."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="000000",
            spend_year=2023,
            supplier_id="SUP-032",
            description="Economy-wide fallback"
        )
        result = engine.calculate(record)
        # Should use economy-wide average
        assert result.naics_code == "00"
        assert result.emission_factor > Decimal("0")

    def test_database_selection(self, engine):
        """Test selection of appropriate EEIO database."""
        # US spend should use USEEIO
        record_us = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-033",
            country="USA"
        )
        result_us = engine.calculate(record_us)
        assert result_us.eeio_database == "USEEIO"

    def test_base_year_selection(self, engine):
        """Test base year selection for EEIO database."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-034",
            description="Base year test"
        )
        result = engine.calculate(record)
        # USEEIO base year is 2017
        assert result.base_year == 2017


class TestBatchCalculation:
    """Test batch calculation of multiple spend records."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_multiple_items(self, engine):
        """Test batch calculation of multiple items."""
        records = [
            SpendRecord(
                spend_amount=Decimal("5000.00"),
                currency="USD",
                naics_code="334111",
                spend_year=2023,
                supplier_id="SUP-040",
                description="Item 1"
            ),
            SpendRecord(
                spend_amount=Decimal("8000.00"),
                currency="USD",
                naics_code="331110",
                spend_year=2023,
                supplier_id="SUP-041",
                description="Item 2"
            ),
            SpendRecord(
                spend_amount=Decimal("3000.00"),
                currency="USD",
                naics_code="325211",
                spend_year=2023,
                supplier_id="SUP-042",
                description="Item 3"
            )
        ]
        results = engine.calculate_batch(records)
        assert len(results) == 3
        assert all(r.total_emissions > Decimal("0") for r in results)

    def test_mixed_currencies(self, engine):
        """Test batch calculation with mixed currencies."""
        records = [
            SpendRecord(
                spend_amount=Decimal("5000.00"),
                currency="USD",
                naics_code="334111",
                spend_year=2023,
                supplier_id="SUP-043"
            ),
            SpendRecord(
                spend_amount=Decimal("5000.00"),
                currency="EUR",
                naics_code="334111",
                spend_year=2023,
                supplier_id="SUP-044"
            ),
            SpendRecord(
                spend_amount=Decimal("500000.00"),
                currency="JPY",
                naics_code="334111",
                spend_year=2023,
                supplier_id="SUP-045"
            )
        ]
        results = engine.calculate_batch(records)
        assert len(results) == 3
        # All should be converted to USD
        assert all(r.currency == "USD" for r in results)

    def test_sector_mix(self, engine):
        """Test batch calculation across diverse sectors."""
        sectors = ["334111", "331110", "325211", "311111", "541511"]
        records = [
            SpendRecord(
                spend_amount=Decimal("10000.00"),
                currency="USD",
                naics_code=sector,
                spend_year=2023,
                supplier_id=f"SUP-{i+50}"
            )
            for i, sector in enumerate(sectors)
        ]
        results = engine.calculate_batch(records)
        assert len(results) == 5
        # Emission factors should vary by sector
        efs = [r.emission_factor for r in results]
        assert len(set(efs)) > 1  # Not all the same

    def test_batch_performance(self, engine):
        """Test batch calculation performance (100+ records)."""
        records = [
            SpendRecord(
                spend_amount=Decimal("1000.00"),
                currency="USD",
                naics_code="334111",
                spend_year=2023,
                supplier_id=f"SUP-PERF-{i}"
            )
            for i in range(100)
        ]
        import time
        start = time.time()
        results = engine.calculate_batch(records)
        elapsed = time.time() - start

        assert len(results) == 100
        # Should complete in reasonable time (< 5s for 100 records)
        assert elapsed < 5.0

    def test_empty_batch(self, engine):
        """Test handling of empty batch."""
        results = engine.calculate_batch([])
        assert results == []


class TestDQIScoring:
    """Test Data Quality Indicator (DQI) scoring."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_default_dqi_scores(self, engine):
        """Test default DQI scores for spend-based method."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-060"
        )
        result = engine.calculate(record)

        assert result.dqi is not None
        # Spend-based has medium quality (Tier 3)
        assert result.dqi.tier == 3
        assert 2.0 <= result.dqi.composite_score <= 4.0

    def test_dqi_dimensions(self, engine):
        """Test individual DQI dimensions."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-061"
        )
        result = engine.calculate(record)

        dqi = result.dqi
        # All dimensions should be scored 1-5
        assert 1 <= dqi.technological_representativeness <= 5
        assert 1 <= dqi.geographical_representativeness <= 5
        assert 1 <= dqi.temporal_representativeness <= 5
        assert 1 <= dqi.completeness <= 5
        assert 1 <= dqi.reliability <= 5

    def test_composite_dqi_calculation(self, engine):
        """Test composite DQI calculation formula."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-062"
        )
        result = engine.calculate(record)

        dqi = result.dqi
        # Composite = average of 5 dimensions
        expected = (
            dqi.technological_representativeness +
            dqi.geographical_representativeness +
            dqi.temporal_representativeness +
            dqi.completeness +
            dqi.reliability
        ) / Decimal("5")

        tolerance = Decimal("0.01")
        assert abs(dqi.composite_score - expected) < tolerance

    def test_quality_tier_mapping(self, engine):
        """Test quality tier mapping (1=best, 5=worst)."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-063"
        )
        result = engine.calculate(record)

        # Spend-based is Tier 3 (medium quality)
        assert result.dqi.tier == 3
        assert 2.0 <= result.dqi.composite_score <= 4.0

    def test_old_data_temporal_score(self, engine):
        """Test temporal score degrades for old data."""
        record_recent = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-064"
        )
        record_old = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2010,  # 13 years old
            supplier_id="SUP-065"
        )

        result_recent = engine.calculate(record_recent)
        result_old = engine.calculate(record_old)

        # Older data should have worse temporal score
        assert result_old.dqi.temporal_representativeness >= \
               result_recent.dqi.temporal_representativeness

    def test_6digit_naics_completeness(self, engine):
        """Test completeness score improves with detailed NAICS."""
        record_6digit = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",  # 6-digit
            spend_year=2023,
            supplier_id="SUP-066"
        )
        record_2digit = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="33",  # 2-digit
            spend_year=2023,
            supplier_id="SUP-067"
        )

        result_6digit = engine.calculate(record_6digit)
        result_2digit = engine.calculate(record_2digit)

        # More detailed NAICS should have better completeness
        assert result_6digit.dqi.completeness <= result_2digit.dqi.completeness


class TestAggregation:
    """Test aggregation and summarization of spend calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    @pytest.fixture
    def sample_records(self):
        """Create sample spend records."""
        return [
            SpendRecord(
                spend_amount=Decimal("5000.00"),
                currency="USD",
                naics_code="334111",
                spend_year=2023,
                supplier_id="SUP-070",
                description="Electronics"
            ),
            SpendRecord(
                spend_amount=Decimal("8000.00"),
                currency="USD",
                naics_code="334111",
                spend_year=2023,
                supplier_id="SUP-071",
                description="Electronics"
            ),
            SpendRecord(
                spend_amount=Decimal("10000.00"),
                currency="USD",
                naics_code="331110",
                spend_year=2023,
                supplier_id="SUP-072",
                description="Steel"
            ),
            SpendRecord(
                spend_amount=Decimal("3000.00"),
                currency="USD",
                naics_code="331110",
                spend_year=2023,
                supplier_id="SUP-073",
                description="Steel"
            )
        ]

    def test_aggregate_by_sector(self, engine, sample_records):
        """Test aggregation by NAICS sector."""
        results = engine.calculate_batch(sample_records)
        aggregation = engine.aggregate_by_sector(results)

        assert "334111" in aggregation
        assert "331110" in aggregation

        # Electronics: 5000 + 8000 = 13000
        assert aggregation["334111"].total_spend == Decimal("13000.00")
        # Steel: 10000 + 3000 = 13000
        assert aggregation["331110"].total_spend == Decimal("13000.00")

    def test_aggregate_by_supplier(self, engine, sample_records):
        """Test aggregation by supplier."""
        results = engine.calculate_batch(sample_records)
        aggregation = engine.aggregate_by_supplier(results)

        assert "SUP-070" in aggregation
        assert "SUP-071" in aggregation
        assert "SUP-072" in aggregation
        assert "SUP-073" in aggregation

        # Each supplier has one record
        assert aggregation["SUP-070"].total_spend == Decimal("5000.00")

    def test_aggregate_totals(self, engine, sample_records):
        """Test total aggregation across all records."""
        results = engine.calculate_batch(sample_records)
        totals = engine.aggregate_totals(results)

        # Total spend: 5000 + 8000 + 10000 + 3000 = 26000
        assert totals.total_spend == Decimal("26000.00")
        assert totals.total_emissions > Decimal("0")
        assert totals.record_count == 4

    def test_coverage_calculation(self, engine, sample_records):
        """Test data coverage calculation."""
        results = engine.calculate_batch(sample_records)
        coverage = engine.calculate_coverage(results)

        assert 0.0 <= coverage.spend_coverage <= 1.0
        assert 0.0 <= coverage.emission_coverage <= 1.0
        assert coverage.records_with_data == 4
        assert coverage.total_records == 4

    def test_sector_breakdown(self, engine, sample_records):
        """Test sector-level breakdown."""
        results = engine.calculate_batch(sample_records)
        breakdown = engine.get_sector_breakdown(results)

        assert len(breakdown) == 2  # Two sectors
        # Each sector should have spend and emissions
        for sector in breakdown:
            assert sector.sector_code is not None
            assert sector.spend_amount > Decimal("0")
            assert sector.emissions > Decimal("0")

    def test_top_emitters(self, engine, sample_records):
        """Test identification of top emitting sectors."""
        results = engine.calculate_batch(sample_records)
        top_emitters = engine.get_top_emitters(results, top_n=2)

        assert len(top_emitters) <= 2
        # Should be sorted by emissions descending
        if len(top_emitters) == 2:
            assert top_emitters[0].emissions >= top_emitters[1].emissions

    def test_spend_intensity(self, engine, sample_records):
        """Test spend intensity calculation (emissions per dollar)."""
        results = engine.calculate_batch(sample_records)
        intensities = engine.calculate_spend_intensity(results)

        for sector, intensity in intensities.items():
            # Intensity = emissions / spend
            assert intensity > Decimal("0")

    def test_weighted_average_dqi(self, engine, sample_records):
        """Test weighted average DQI calculation."""
        results = engine.calculate_batch(sample_records)
        avg_dqi = engine.calculate_weighted_dqi(results)

        # Weighted by spend
        assert 1.0 <= avg_dqi <= 5.0


class TestUncertainty:
    """Test uncertainty quantification for spend-based calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_base_uncertainty_range(self, engine):
        """Test base uncertainty range for spend-based method."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-080"
        )
        result = engine.calculate(record)

        assert result.uncertainty is not None
        # Spend-based typically has ±50% uncertainty
        lower_bound = result.total_emissions * Decimal("0.5")
        upper_bound = result.total_emissions * Decimal("1.5")

        assert result.uncertainty.lower_bound <= result.total_emissions
        assert result.uncertainty.upper_bound >= result.total_emissions
        assert abs(result.uncertainty.lower_bound - lower_bound) < Decimal("1.0")
        assert abs(result.uncertainty.upper_bound - upper_bound) < Decimal("1.0")

    def test_pedigree_adjustment(self, engine):
        """Test pedigree matrix adjustment to uncertainty."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-081"
        )
        result = engine.calculate(record)

        # Uncertainty adjusted by DQI pedigree
        assert result.uncertainty.pedigree_adjusted is not None

    def test_sector_specific_uncertainty(self, engine):
        """Test that uncertainty varies by sector."""
        record_electronics = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-082"
        )
        record_steel = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="331110",
            spend_year=2023,
            supplier_id="SUP-083"
        )

        result_electronics = engine.calculate(record_electronics)
        result_steel = engine.calculate(record_steel)

        # Uncertainty ranges should differ
        assert result_electronics.uncertainty.relative_uncertainty != \
               result_steel.uncertainty.relative_uncertainty

    def test_aggregated_uncertainty(self, engine):
        """Test uncertainty aggregation across multiple records."""
        records = [
            SpendRecord(
                spend_amount=Decimal("5000.00"),
                currency="USD",
                naics_code="334111",
                spend_year=2023,
                supplier_id=f"SUP-084-{i}"
            )
            for i in range(5)
        ]
        results = engine.calculate_batch(records)
        aggregated = engine.aggregate_totals(results)

        # Aggregated uncertainty should be lower than individual (law of large numbers)
        individual_rel_unc = results[0].uncertainty.relative_uncertainty
        aggregated_rel_unc = aggregated.uncertainty.relative_uncertainty

        assert aggregated_rel_unc <= individual_rel_unc


class TestFormulas:
    """Test exact formula implementation for spend-based calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_spend_pipeline_formula(self, engine):
        """Test complete spend processing pipeline formula."""
        spend_amount = Decimal("10000.00")
        currency = "EUR"
        naics_code = "334111"
        spend_year = 2023

        record = SpendRecord(
            spend_amount=spend_amount,
            currency=currency,
            naics_code=naics_code,
            spend_year=spend_year,
            supplier_id="SUP-090"
        )
        result = engine.calculate(record)

        # Manual calculation to verify
        # Step 1: Currency conversion
        db = ProcurementDatabaseEngine()
        spend_usd = db.convert_currency(
            amount=spend_amount,
            from_currency=currency,
            to_currency="USD",
            conversion_date=datetime(spend_year, 6, 15)
        )

        # Step 2: Margin removal
        producer_price = db.remove_margin(
            purchaser_price=spend_usd,
            naics_code=naics_code
        )

        # Step 3: CPI deflation
        deflated = db.deflate_to_base_year(
            amount=producer_price,
            currency="USD",
            current_year=spend_year,
            base_year=2017
        )

        # Step 4: Apply emission factor
        ef = db.lookup_eeio_factor(
            naics_code=naics_code,
            database="USEEIO",
            base_year=2017
        )
        expected_emissions = deflated * ef.emission_factor

        # Verify
        tolerance = Decimal("0.01")
        assert abs(result.total_emissions - expected_emissions) < tolerance

    def test_margin_removal_formula(self, engine):
        """Test margin removal formula: producer_price = purchaser_price * (1 - margin_rate)."""
        record = SpendRecord(
            spend_amount=Decimal("1000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-091"
        )
        result = engine.calculate(record)

        expected_producer = result.spend_amount_usd * (Decimal("1") - result.margin_rate)
        tolerance = Decimal("0.01")
        assert abs(result.producer_price - expected_producer) < tolerance

    def test_cpi_deflation_formula(self, engine):
        """Test CPI deflation formula: deflated = amount * (CPI_base / CPI_current)."""
        record = SpendRecord(
            spend_amount=Decimal("1000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2020,
            supplier_id="SUP-092"
        )
        result = engine.calculate(record)

        # Deflated should be less than producer price (2020 -> 2017)
        assert result.deflated_spend < result.producer_price

    def test_emission_calculation_formula(self, engine):
        """Test emission calculation: emissions = deflated_spend * emission_factor."""
        record = SpendRecord(
            spend_amount=Decimal("1000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-093"
        )
        result = engine.calculate(record)

        expected = result.deflated_spend * result.emission_factor
        tolerance = Decimal("0.01")
        assert abs(result.total_emissions - expected) < tolerance

    def test_currency_conversion_precision(self, engine):
        """Test currency conversion maintains precision."""
        record = SpendRecord(
            spend_amount=Decimal("1234.56"),
            currency="EUR",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-094"
        )
        result = engine.calculate(record)

        # Should preserve at least 2 decimal places
        assert result.spend_amount_usd.as_tuple().exponent <= -2

    def test_decimal_arithmetic_no_floating_point(self, engine):
        """Test that all arithmetic uses Decimal, not float."""
        record = SpendRecord(
            spend_amount=Decimal("10000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-095"
        )
        result = engine.calculate(record)

        # All monetary values should be Decimal
        assert isinstance(result.spend_amount_usd, Decimal)
        assert isinstance(result.producer_price, Decimal)
        assert isinstance(result.deflated_spend, Decimal)
        assert isinstance(result.emission_factor, Decimal)
        assert isinstance(result.total_emissions, Decimal)

    def test_negative_emission_credit_formula(self, engine):
        """Test negative emissions for credits/returns."""
        record = SpendRecord(
            spend_amount=Decimal("-1000.00"),
            currency="USD",
            naics_code="334111",
            spend_year=2023,
            supplier_id="SUP-096"
        )
        result = engine.calculate(record)

        # Negative spend should produce negative emissions
        assert result.total_emissions < Decimal("0")
        # Formula still holds
        expected = result.deflated_spend * result.emission_factor
        tolerance = Decimal("0.01")
        assert abs(result.total_emissions - expected) < tolerance

    def test_zero_margin_formula(self, engine):
        """Test formula when margin is zero."""
        # Use NAICS with zero margin
        record = SpendRecord(
            spend_amount=Decimal("1000.00"),
            currency="USD",
            naics_code="211111",  # Oil extraction (low margin)
            spend_year=2023,
            supplier_id="SUP-097"
        )
        result = engine.calculate(record)

        # When margin ~ 0, producer_price ≈ purchaser_price
        if result.margin_rate < Decimal("0.01"):
            tolerance = Decimal("10.0")
            assert abs(result.producer_price - result.spend_amount_usd) < tolerance


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SpendBasedCalculatorEngine()

    def test_health_check_returns_dict(self, engine):
        """Test health check returns valid response."""
        health = engine.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_all_fields(self, engine):
        """Test health check contains all expected fields."""
        health = engine.health_check()
        expected_fields = [
            "status",
            "database_connection",
            "eeio_factors_loaded",
            "currency_rates_loaded",
            "margin_adjustments_loaded",
            "last_calculation_time_ms"
        ]
        for field in expected_fields:
            assert field in health
