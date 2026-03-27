"""
Test suite for SpendBasedCalculatorEngine.

This module tests the spend-based calculation method for capital goods emissions,
including currency conversion, CPI deflation, margin removal, EEIO factor lookup,
emissions calculation, uncertainty estimation, and aggregation capabilities.

Test Coverage:
- Singleton pattern enforcement
- Full calculation pipeline for single and batch records
- Currency conversion to USD for 20+ currencies
- CPI-based deflation to base year
- Margin removal for capital goods sectors
- EEIO factor lookup with progressive NAICS fallback
- Emissions calculation formula: spend × factor
- Gas breakdown (CO2, CH4, N2O) summing to total
- Data quality indicator scoring (1-5 per dimension)
- Aggregation by sector, category, year
- Top emitters identification
- Coverage reporting
- Input validation and error handling
- Uncertainty estimation with Monte Carlo
- Provenance hash calculation
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.mrv.capital_goods.engines.spend_based_calculator import (
    SpendBasedCalculatorEngine,
    SpendBasedRecord,
    SpendBasedResult,
    GasBreakdown,
)
from greenlang.agents.mrv.capital_goods.models import (
    EmissionFactorSource,
    DataQualityDimension,
    DataQualityScore,
)


@pytest.fixture
def calculator_engine():
    """Provide fresh SpendBasedCalculatorEngine instance."""
    # Reset singleton
    SpendBasedCalculatorEngine._instance = None
    engine = SpendBasedCalculatorEngine()
    return engine


@pytest.fixture
def sample_record():
    """Provide sample spend-based record."""
    return SpendBasedRecord(
        record_id="REC-001",
        description="CNC Machining Center",
        naics_code="333517",
        spend_amount=Decimal("100000.00"),
        currency="USD",
        purchase_year=2023,
        region="US",
    )


@pytest.fixture
def mock_database_engine():
    """Mock CapitalAssetDatabaseEngine."""
    mock_db = MagicMock()
    return mock_db


class TestSingletonPattern:
    """Test singleton pattern enforcement."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        # Arrange & Act
        engine1 = SpendBasedCalculatorEngine()
        engine2 = SpendBasedCalculatorEngine()

        # Assert
        assert engine1 is engine2

    def test_singleton_persists_state(self, calculator_engine):
        """Test that singleton preserves state across references."""
        # Arrange
        calculator_engine._base_year = 2020

        # Act
        new_ref = SpendBasedCalculatorEngine()

        # Assert
        assert new_ref._base_year == 2020


class TestCalculateSingleRecord:
    """Test calculate() method for single record."""

    def test_calculate_full_pipeline(self, calculator_engine, sample_record, mock_database_engine):
        """Test complete calculation pipeline."""
        # Arrange
        mock_database_engine.get_eeio_factor.return_value = MagicMock(
            ef_co2e_per_usd=Decimal("0.525"),
            ef_co2_per_usd=Decimal("0.450"),
            ef_ch4_per_usd=Decimal("0.015"),
            ef_n2o_per_usd=Decimal("0.060"),
            source=EmissionFactorSource.USEEIO,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate(sample_record)

            # Assert
            assert result is not None
            assert result.record_id == "REC-001"
            assert result.total_emissions_kg_co2e == Decimal("52500.00")  # 100000 * 0.525
            assert result.ef_source == EmissionFactorSource.USEEIO
            assert result.provenance_hash is not None

    def test_calculate_with_currency_conversion(self, calculator_engine, mock_database_engine):
        """Test calculation with currency conversion to USD."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-002",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="EUR",
            purchase_year=2023,
            region="EU",
        )

        mock_database_engine.convert_currency.return_value = Decimal("110000.00")
        mock_database_engine.get_eeio_factor.return_value = MagicMock(
            ef_co2e_per_usd=Decimal("0.500"),
            ef_co2_per_usd=Decimal("0.430"),
            ef_ch4_per_usd=Decimal("0.014"),
            ef_n2o_per_usd=Decimal("0.056"),
            source=EmissionFactorSource.EXIOBASE,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate(record)

            # Assert
            assert result.spend_usd == Decimal("110000.00")
            assert result.total_emissions_kg_co2e == Decimal("55000.00")

    def test_calculate_with_deflation(self, calculator_engine, mock_database_engine):
        """Test calculation with CPI deflation."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-003",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("120000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )

        mock_database_engine.deflate_to_base_year.return_value = Decimal("100000.00")
        mock_database_engine.get_eeio_factor.return_value = MagicMock(
            ef_co2e_per_usd=Decimal("0.500"),
            source=EmissionFactorSource.USEEIO,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate(record)

            # Assert
            assert result.deflated_spend_usd == Decimal("100000.00")

    def test_calculate_with_margin_removal(self, calculator_engine, mock_database_engine):
        """Test calculation with margin removal."""
        # Arrange
        record = sample_record

        mock_database_engine.remove_margin.return_value = Decimal("75000.00")
        mock_database_engine.get_eeio_factor.return_value = MagicMock(
            ef_co2e_per_usd=Decimal("0.500"),
            source=EmissionFactorSource.USEEIO,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate(record)

            # Assert
            assert result.producer_price_usd == Decimal("75000.00")

    def test_calculate_no_emission_factor_found(self, calculator_engine, mock_database_engine):
        """Test calculation when no emission factor found."""
        # Arrange
        record = sample_record
        mock_database_engine.get_eeio_factor.return_value = None

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act & Assert
            with pytest.raises(ValueError, match="No emission factor found"):
                calculator_engine.calculate(record)


class TestCalculateBatch:
    """Test calculate_batch() method for multiple records."""

    def test_calculate_batch_multiple_records(self, calculator_engine, mock_database_engine):
        """Test batch calculation for multiple records."""
        # Arrange
        records = [
            SpendBasedRecord(
                record_id=f"REC-{i:03d}",
                description=f"Equipment {i}",
                naics_code="333517",
                spend_amount=Decimal("10000.00"),
                currency="USD",
                purchase_year=2023,
                region="US",
            )
            for i in range(1, 6)
        ]

        mock_database_engine.get_eeio_factor.return_value = MagicMock(
            ef_co2e_per_usd=Decimal("0.500"),
            source=EmissionFactorSource.USEEIO,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            results = calculator_engine.calculate_batch(records)

            # Assert
            assert len(results) == 5
            assert all(r.total_emissions_kg_co2e == Decimal("5000.00") for r in results)

    def test_calculate_batch_preserves_order(self, calculator_engine, mock_database_engine):
        """Test batch calculation preserves input order."""
        # Arrange
        records = [
            SpendBasedRecord(
                record_id=f"REC-{i:03d}",
                description=f"Equipment {i}",
                naics_code="333517",
                spend_amount=Decimal(f"{i * 1000}.00"),
                currency="USD",
                purchase_year=2023,
                region="US",
            )
            for i in range(1, 4)
        ]

        mock_database_engine.get_eeio_factor.return_value = MagicMock(
            ef_co2e_per_usd=Decimal("0.500"),
            source=EmissionFactorSource.USEEIO,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            results = calculator_engine.calculate_batch(records)

            # Assert
            assert results[0].record_id == "REC-001"
            assert results[1].record_id == "REC-002"
            assert results[2].record_id == "REC-003"

    def test_calculate_batch_handles_errors(self, calculator_engine, mock_database_engine):
        """Test batch calculation handles individual record errors."""
        # Arrange
        records = [
            SpendBasedRecord(
                record_id="REC-001",
                description="Valid",
                naics_code="333517",
                spend_amount=Decimal("10000.00"),
                currency="USD",
                purchase_year=2023,
                region="US",
            ),
            SpendBasedRecord(
                record_id="REC-002",
                description="Invalid",
                naics_code="999999",
                spend_amount=Decimal("10000.00"),
                currency="USD",
                purchase_year=2023,
                region="US",
            ),
        ]

        def side_effect(naics, region, year):
            if naics == "333517":
                return MagicMock(ef_co2e_per_usd=Decimal("0.500"))
            return None

        mock_database_engine.get_eeio_factor.side_effect = side_effect

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            results = calculator_engine.calculate_batch(records, skip_errors=True)

            # Assert
            assert len(results) == 1
            assert results[0].record_id == "REC-001"


class TestCurrencyConversion:
    """Test convert_to_usd() method."""

    def test_convert_to_usd_from_eur(self, calculator_engine, mock_database_engine):
        """Test conversion from EUR to USD."""
        # Arrange
        mock_database_engine.convert_currency.return_value = Decimal("1100.00")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.convert_to_usd(
                Decimal("1000.00"),
                "EUR",
                datetime(2023, 12, 1)
            )

            # Assert
            assert result == Decimal("1100.00")

    def test_convert_to_usd_already_usd(self, calculator_engine):
        """Test conversion when already in USD."""
        # Act
        result = calculator_engine.convert_to_usd(
            Decimal("1000.00"),
            "USD",
            datetime(2023, 12, 1)
        )

        # Assert
        assert result == Decimal("1000.00")

    @pytest.mark.parametrize("currency,amount", [
        ("EUR", Decimal("1000.00")),
        ("GBP", Decimal("1000.00")),
        ("JPY", Decimal("100000.00")),
        ("CNY", Decimal("7000.00")),
        ("CAD", Decimal("1000.00")),
    ])
    def test_convert_to_usd_various_currencies(
        self,
        calculator_engine,
        mock_database_engine,
        currency,
        amount
    ):
        """Test conversion from various currencies."""
        # Arrange
        mock_database_engine.convert_currency.return_value = Decimal("1000.00")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.convert_to_usd(amount, currency, datetime(2023, 12, 1))

            # Assert
            assert result is not None
            mock_database_engine.convert_currency.assert_called_once()


class TestCPIDeflation:
    """Test deflate_spend() method."""

    def test_deflate_spend_to_base_year(self, calculator_engine, mock_database_engine):
        """Test deflation to base year."""
        # Arrange
        mock_database_engine.deflate_to_base_year.return_value = Decimal("1000.00")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.deflate_spend(
                Decimal("1200.00"),
                from_year=2023,
                base_year=2020,
                region="US"
            )

            # Assert
            assert result == Decimal("1000.00")

    def test_deflate_spend_same_year(self, calculator_engine):
        """Test deflation with same source and base year."""
        # Act
        result = calculator_engine.deflate_spend(
            Decimal("1000.00"),
            from_year=2023,
            base_year=2023,
            region="US"
        )

        # Assert
        assert result == Decimal("1000.00")


class TestMarginRemoval:
    """Test remove_margin() method."""

    def test_remove_margin_machinery_sector(self, calculator_engine, mock_database_engine):
        """Test margin removal for machinery sector."""
        # Arrange
        mock_database_engine.remove_margin.return_value = Decimal("7500.00")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.remove_margin(
                Decimal("10000.00"),
                "333517"
            )

            # Assert
            assert result == Decimal("7500.00")

    def test_remove_margin_no_margin_data(self, calculator_engine, mock_database_engine):
        """Test margin removal when no margin data available."""
        # Arrange
        mock_database_engine.remove_margin.return_value = Decimal("10000.00")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.remove_margin(
                Decimal("10000.00"),
                "999999"
            )

            # Assert
            assert result == Decimal("10000.00")


class TestEEIOFactorLookup:
    """Test lookup_eeio_factor() method."""

    def test_lookup_eeio_factor_found(self, calculator_engine, mock_database_engine):
        """Test successful EEIO factor lookup."""
        # Arrange
        mock_factor = MagicMock(
            ef_co2e_per_usd=Decimal("0.525"),
            source=EmissionFactorSource.USEEIO,
        )
        mock_database_engine.get_eeio_factor.return_value = mock_factor

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.lookup_eeio_factor("333517", "US", 2023)

            # Assert
            assert result is not None
            assert result.ef_co2e_per_usd == Decimal("0.525")

    def test_lookup_eeio_factor_not_found(self, calculator_engine, mock_database_engine):
        """Test EEIO factor not found."""
        # Arrange
        mock_database_engine.get_eeio_factor.return_value = None

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.lookup_eeio_factor("999999", "US", 2023)

            # Assert
            assert result is None


class TestEmissionsCalculation:
    """Test calculate_emissions() formula."""

    def test_calculate_emissions_basic_formula(self, calculator_engine):
        """Test basic emissions calculation: spend × factor."""
        # Arrange
        spend = Decimal("100000.00")
        ef = Decimal("0.525")

        # Act
        result = calculator_engine.calculate_emissions(spend, ef)

        # Assert
        assert result == Decimal("52500.00")

    def test_calculate_emissions_zero_spend(self, calculator_engine):
        """Test emissions calculation with zero spend."""
        # Act
        result = calculator_engine.calculate_emissions(Decimal("0"), Decimal("0.525"))

        # Assert
        assert result == Decimal("0")

    def test_calculate_emissions_zero_factor(self, calculator_engine):
        """Test emissions calculation with zero factor."""
        # Act
        result = calculator_engine.calculate_emissions(Decimal("100000.00"), Decimal("0"))

        # Assert
        assert result == Decimal("0")

    def test_calculate_emissions_precision(self, calculator_engine):
        """Test emissions calculation maintains precision."""
        # Arrange
        spend = Decimal("123456.78")
        ef = Decimal("0.525432")

        # Act
        result = calculator_engine.calculate_emissions(spend, ef)

        # Assert
        expected = Decimal("64871.073")  # Rounded to 3 decimal places
        assert abs(result - expected) < Decimal("0.001")


class TestGasBreakdown:
    """Test split_gas_breakdown() method."""

    def test_split_gas_breakdown_sums_to_total(self, calculator_engine):
        """Test gas breakdown components sum to total."""
        # Arrange
        total_co2e = Decimal("52500.00")
        ef = MagicMock(
            ef_co2e_per_usd=Decimal("0.525"),
            ef_co2_per_usd=Decimal("0.450"),
            ef_ch4_per_usd=Decimal("0.015"),
            ef_n2o_per_usd=Decimal("0.060"),
        )
        spend = Decimal("100000.00")

        # Act
        result = calculator_engine.split_gas_breakdown(total_co2e, ef, spend)

        # Assert
        total = result.co2_kg + result.ch4_kg_co2e + result.n2o_kg_co2e
        assert total == total_co2e

    def test_split_gas_breakdown_individual_gases(self, calculator_engine):
        """Test individual gas calculations."""
        # Arrange
        total_co2e = Decimal("52500.00")
        ef = MagicMock(
            ef_co2e_per_usd=Decimal("0.525"),
            ef_co2_per_usd=Decimal("0.450"),
            ef_ch4_per_usd=Decimal("0.015"),
            ef_n2o_per_usd=Decimal("0.060"),
        )
        spend = Decimal("100000.00")

        # Act
        result = calculator_engine.split_gas_breakdown(total_co2e, ef, spend)

        # Assert
        assert result.co2_kg == Decimal("45000.00")  # 100000 * 0.450
        assert result.ch4_kg_co2e == Decimal("1500.00")  # 100000 * 0.015
        assert result.n2o_kg_co2e == Decimal("6000.00")  # 100000 * 0.060

    def test_split_gas_breakdown_no_breakdown_available(self, calculator_engine):
        """Test gas breakdown when individual gas factors not available."""
        # Arrange
        total_co2e = Decimal("52500.00")
        ef = MagicMock(
            ef_co2e_per_usd=Decimal("0.525"),
            ef_co2_per_usd=None,
            ef_ch4_per_usd=None,
            ef_n2o_per_usd=None,
        )
        spend = Decimal("100000.00")

        # Act
        result = calculator_engine.split_gas_breakdown(total_co2e, ef, spend)

        # Assert
        # Should allocate all to CO2 when breakdown not available
        assert result.co2_kg == total_co2e
        assert result.ch4_kg_co2e == Decimal("0")
        assert result.n2o_kg_co2e == Decimal("0")


class TestDataQualityScoring:
    """Test score_dqi() method."""

    def test_score_dqi_all_dimensions(self, calculator_engine):
        """Test DQI scoring returns valid scores for all dimensions."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )
        ef_source = EmissionFactorSource.USEEIO

        # Act
        result = calculator_engine.score_dqi(record, ef_source)

        # Assert
        assert result is not None
        assert 1 <= result.technological_representativeness <= 5
        assert 1 <= result.temporal_representativeness <= 5
        assert 1 <= result.geographical_representativeness <= 5
        assert 1 <= result.completeness <= 5
        assert 1 <= result.reliability <= 5

    def test_score_dqi_high_quality_useeio(self, calculator_engine):
        """Test DQI scoring for high-quality USEEIO source."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )

        # Act
        result = calculator_engine.score_dqi(record, EmissionFactorSource.USEEIO)

        # Assert
        # USEEIO should score high on reliability
        assert result.reliability >= 4

    def test_score_dqi_custom_source_lower_reliability(self, calculator_engine):
        """Test DQI scoring for custom source has lower reliability."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )

        # Act
        result = calculator_engine.score_dqi(record, EmissionFactorSource.CUSTOM)

        # Assert
        # Custom source should have lower reliability
        assert result.reliability <= 3

    def test_score_dqi_temporal_scoring(self, calculator_engine):
        """Test temporal representativeness scoring."""
        # Arrange
        recent_record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )

        old_record = SpendBasedRecord(
            record_id="REC-002",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2010,
            region="US",
        )

        # Act
        recent_score = calculator_engine.score_dqi(recent_record, EmissionFactorSource.USEEIO)
        old_score = calculator_engine.score_dqi(old_record, EmissionFactorSource.USEEIO)

        # Assert
        assert recent_score.temporal_representativeness >= old_score.temporal_representativeness


class TestAggregations:
    """Test aggregation methods."""

    def test_aggregate_by_sector(self, calculator_engine):
        """Test aggregation by NAICS sector."""
        # Arrange
        results = [
            SpendBasedResult(
                record_id="REC-001",
                naics_code="333517",
                total_emissions_kg_co2e=Decimal("10000.00"),
                spend_usd=Decimal("20000.00"),
            ),
            SpendBasedResult(
                record_id="REC-002",
                naics_code="333517",
                total_emissions_kg_co2e=Decimal("15000.00"),
                spend_usd=Decimal("30000.00"),
            ),
            SpendBasedResult(
                record_id="REC-003",
                naics_code="336120",
                total_emissions_kg_co2e=Decimal("8000.00"),
                spend_usd=Decimal("16000.00"),
            ),
        ]

        # Act
        aggregated = calculator_engine.aggregate_by_sector(results)

        # Assert
        assert len(aggregated) == 2
        assert aggregated["333517"]["total_emissions"] == Decimal("25000.00")
        assert aggregated["333517"]["total_spend"] == Decimal("50000.00")
        assert aggregated["336120"]["total_emissions"] == Decimal("8000.00")

    def test_aggregate_by_category(self, calculator_engine, mock_database_engine):
        """Test aggregation by asset category."""
        # Arrange
        results = [
            SpendBasedResult(
                record_id="REC-001",
                naics_code="333517",
                total_emissions_kg_co2e=Decimal("10000.00"),
            ),
            SpendBasedResult(
                record_id="REC-002",
                naics_code="336120",
                total_emissions_kg_co2e=Decimal("8000.00"),
            ),
        ]

        mock_database_engine.classify_asset.side_effect = ["MACHINERY", "VEHICLES"]

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            aggregated = calculator_engine.aggregate_by_category(results)

            # Assert
            assert "MACHINERY" in aggregated
            assert "VEHICLES" in aggregated
            assert aggregated["MACHINERY"]["total_emissions"] == Decimal("10000.00")

    def test_aggregate_by_year(self, calculator_engine):
        """Test aggregation by purchase year."""
        # Arrange
        results = [
            SpendBasedResult(
                record_id="REC-001",
                purchase_year=2023,
                total_emissions_kg_co2e=Decimal("10000.00"),
            ),
            SpendBasedResult(
                record_id="REC-002",
                purchase_year=2023,
                total_emissions_kg_co2e=Decimal("15000.00"),
            ),
            SpendBasedResult(
                record_id="REC-003",
                purchase_year=2022,
                total_emissions_kg_co2e=Decimal("8000.00"),
            ),
        ]

        # Act
        aggregated = calculator_engine.aggregate_by_year(results)

        # Assert
        assert len(aggregated) == 2
        assert aggregated[2023]["total_emissions"] == Decimal("25000.00")
        assert aggregated[2022]["total_emissions"] == Decimal("8000.00")


class TestTopEmitters:
    """Test get_top_emitters() method."""

    def test_get_top_emitters_default_limit(self, calculator_engine):
        """Test getting top emitters with default limit."""
        # Arrange
        results = [
            SpendBasedResult(
                record_id=f"REC-{i:03d}",
                total_emissions_kg_co2e=Decimal(f"{i * 1000}.00"),
            )
            for i in range(1, 21)
        ]

        # Act
        top = calculator_engine.get_top_emitters(results)

        # Assert
        assert len(top) == 10  # Default limit
        assert top[0].total_emissions_kg_co2e == Decimal("20000.00")
        assert top[-1].total_emissions_kg_co2e == Decimal("11000.00")

    def test_get_top_emitters_custom_limit(self, calculator_engine):
        """Test getting top emitters with custom limit."""
        # Arrange
        results = [
            SpendBasedResult(
                record_id=f"REC-{i:03d}",
                total_emissions_kg_co2e=Decimal(f"{i * 1000}.00"),
            )
            for i in range(1, 21)
        ]

        # Act
        top = calculator_engine.get_top_emitters(results, limit=5)

        # Assert
        assert len(top) == 5
        assert top[0].total_emissions_kg_co2e == Decimal("20000.00")

    def test_get_top_emitters_sorted_descending(self, calculator_engine):
        """Test top emitters are sorted in descending order."""
        # Arrange
        results = [
            SpendBasedResult(record_id="REC-001", total_emissions_kg_co2e=Decimal("5000.00")),
            SpendBasedResult(record_id="REC-002", total_emissions_kg_co2e=Decimal("15000.00")),
            SpendBasedResult(record_id="REC-003", total_emissions_kg_co2e=Decimal("10000.00")),
        ]

        # Act
        top = calculator_engine.get_top_emitters(results, limit=3)

        # Assert
        assert top[0].record_id == "REC-002"
        assert top[1].record_id == "REC-003"
        assert top[2].record_id == "REC-001"


class TestCoverageReport:
    """Test get_coverage_report() method."""

    def test_get_coverage_report_basic_stats(self, calculator_engine):
        """Test coverage report provides basic statistics."""
        # Arrange
        results = [
            SpendBasedResult(
                record_id=f"REC-{i:03d}",
                total_emissions_kg_co2e=Decimal(f"{i * 1000}.00"),
                spend_usd=Decimal(f"{i * 2000}.00"),
            )
            for i in range(1, 11)
        ]

        # Act
        report = calculator_engine.get_coverage_report(results)

        # Assert
        assert report["total_records"] == 10
        assert report["total_emissions"] == Decimal("55000.00")
        assert report["total_spend"] == Decimal("110000.00")

    def test_get_coverage_report_ef_source_distribution(self, calculator_engine):
        """Test coverage report includes EF source distribution."""
        # Arrange
        results = [
            SpendBasedResult(
                record_id="REC-001",
                total_emissions_kg_co2e=Decimal("10000.00"),
                ef_source=EmissionFactorSource.USEEIO,
            ),
            SpendBasedResult(
                record_id="REC-002",
                total_emissions_kg_co2e=Decimal("15000.00"),
                ef_source=EmissionFactorSource.USEEIO,
            ),
            SpendBasedResult(
                record_id="REC-003",
                total_emissions_kg_co2e=Decimal("8000.00"),
                ef_source=EmissionFactorSource.EXIOBASE,
            ),
        ]

        # Act
        report = calculator_engine.get_coverage_report(results)

        # Assert
        assert "ef_source_distribution" in report
        assert report["ef_source_distribution"]["USEEIO"] == 2
        assert report["ef_source_distribution"]["EXIOBASE"] == 1


class TestValidation:
    """Test validate_record() method."""

    def test_validate_record_valid(self, calculator_engine):
        """Test validation passes for valid record."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )

        # Act
        is_valid, errors = calculator_engine.validate_record(record)

        # Assert
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_record_negative_spend(self, calculator_engine):
        """Test validation fails for negative spend."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("-100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )

        # Act
        is_valid, errors = calculator_engine.validate_record(record)

        # Assert
        assert is_valid is False
        assert "spend_amount" in str(errors)

    def test_validate_record_invalid_naics(self, calculator_engine):
        """Test validation fails for invalid NAICS code."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="ABC123",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )

        # Act
        is_valid, errors = calculator_engine.validate_record(record)

        # Assert
        assert is_valid is False
        assert "naics_code" in str(errors)

    def test_validate_record_future_year(self, calculator_engine):
        """Test validation fails for future purchase year."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2030,
            region="US",
        )

        # Act
        is_valid, errors = calculator_engine.validate_record(record)

        # Assert
        assert is_valid is False
        assert "purchase_year" in str(errors)


class TestUncertaintyEstimation:
    """Test estimate_uncertainty() method."""

    def test_estimate_uncertainty_returns_bounds(self, calculator_engine):
        """Test uncertainty estimation returns lower and upper bounds."""
        # Arrange
        result = SpendBasedResult(
            record_id="REC-001",
            total_emissions_kg_co2e=Decimal("52500.00"),
            spend_usd=Decimal("100000.00"),
        )

        # Act
        uncertainty = calculator_engine.estimate_uncertainty(result)

        # Assert
        assert "lower_bound" in uncertainty
        assert "upper_bound" in uncertainty
        assert uncertainty["lower_bound"] < result.total_emissions_kg_co2e
        assert uncertainty["upper_bound"] > result.total_emissions_kg_co2e

    def test_estimate_uncertainty_confidence_interval(self, calculator_engine):
        """Test uncertainty estimation respects confidence interval."""
        # Arrange
        result = SpendBasedResult(
            record_id="REC-001",
            total_emissions_kg_co2e=Decimal("52500.00"),
            spend_usd=Decimal("100000.00"),
        )

        # Act
        uncertainty_95 = calculator_engine.estimate_uncertainty(result, confidence=0.95)
        uncertainty_68 = calculator_engine.estimate_uncertainty(result, confidence=0.68)

        # Assert
        # 95% interval should be wider than 68% interval
        range_95 = uncertainty_95["upper_bound"] - uncertainty_95["lower_bound"]
        range_68 = uncertainty_68["upper_bound"] - uncertainty_68["lower_bound"]
        assert range_95 > range_68


class TestProvenanceHash:
    """Test compute_provenance_hash() method."""

    def test_compute_provenance_hash_deterministic(self, calculator_engine):
        """Test provenance hash is deterministic."""
        # Arrange
        record = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )
        result = SpendBasedResult(
            record_id="REC-001",
            total_emissions_kg_co2e=Decimal("52500.00"),
        )

        # Act
        hash1 = calculator_engine.compute_provenance_hash(record, result)
        hash2 = calculator_engine.compute_provenance_hash(record, result)

        # Assert
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_compute_provenance_hash_different_inputs(self, calculator_engine):
        """Test different inputs produce different hashes."""
        # Arrange
        record1 = SpendBasedRecord(
            record_id="REC-001",
            description="Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )
        record2 = SpendBasedRecord(
            record_id="REC-002",
            description="Different Equipment",
            naics_code="333517",
            spend_amount=Decimal("100000.00"),
            currency="USD",
            purchase_year=2023,
            region="US",
        )
        result = SpendBasedResult(
            record_id="REC-001",
            total_emissions_kg_co2e=Decimal("52500.00"),
        )

        # Act
        hash1 = calculator_engine.compute_provenance_hash(record1, result)
        hash2 = calculator_engine.compute_provenance_hash(record2, result)

        # Assert
        assert hash1 != hash2
