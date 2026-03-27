"""
Test suite for CapitalAssetDatabaseEngine.

This module tests the database lookup and classification capabilities
of the Capital Goods agent, including EEIO factor matching, physical EF lookup,
progressive NAICS fallback, currency conversion, and taxonomy mapping.

Test Coverage:
- Singleton pattern enforcement
- EEIO factor lookup with progressive NAICS matching (6→5→4→3→2 digit)
- Physical emission factor lookup for construction materials
- Supplier-specific emission factor lookup
- Asset classification and capitalization threshold checks
- Useful life estimation
- Currency conversion with 20+ currencies
- CPI-based deflation to base year
- Margin removal for capital goods sectors
- Taxonomy mapping (NAICS↔ISIC, NACE→ISIC, UNSPSC→NAICS)
- Custom factor registration and override
- Database validation and statistics
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional, List
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.mrv.capital_goods.engines.capital_asset_database import (
    CapitalAssetDatabaseEngine,
    EEIOFactor,
    PhysicalEmissionFactor,
    SupplierEmissionFactor,
    AssetCategory,
    MaterialType,
    DatabaseStats,
)
from greenlang.agents.mrv.capital_goods.models import (
    EmissionFactorSource,
    DataQualityDimension,
)


@pytest.fixture
def database_engine():
    """Provide fresh CapitalAssetDatabaseEngine instance."""
    # Reset singleton
    CapitalAssetDatabaseEngine._instance = None
    engine = CapitalAssetDatabaseEngine()
    return engine


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn, mock_cursor


class TestSingletonPattern:
    """Test singleton pattern enforcement."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        # Arrange & Act
        engine1 = CapitalAssetDatabaseEngine()
        engine2 = CapitalAssetDatabaseEngine()

        # Assert
        assert engine1 is engine2

    def test_singleton_persists_state(self, database_engine):
        """Test that singleton preserves state across references."""
        # Arrange
        custom_factor = EEIOFactor(
            naics_code="999999",
            description="Custom Test",
            ef_co2e_per_usd=Decimal("1.5"),
            region="US",
            year=2023,
            source=EmissionFactorSource.USEEIO,
        )
        database_engine.register_custom_eeio_factor(custom_factor)

        # Act
        new_ref = CapitalAssetDatabaseEngine()
        result = new_ref.get_eeio_factor("999999", "US", 2023)

        # Assert
        assert result is not None
        assert result.ef_co2e_per_usd == Decimal("1.5")


class TestEEIOFactorLookup:
    """Test EEIO emission factor lookup with progressive NAICS matching."""

    def test_get_eeio_factor_exact_6digit_match(self, database_engine, mock_db_connection):
        """Test exact 6-digit NAICS code match."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "naics_code": "333611",
            "description": "Turbine and Turbine Generator Set Units Manufacturing",
            "ef_co2e_per_usd": Decimal("0.525"),
            "ef_co2_per_usd": Decimal("0.450"),
            "ef_ch4_per_usd": Decimal("0.015"),
            "ef_n2o_per_usd": Decimal("0.060"),
            "region": "US",
            "year": 2023,
            "source": "USEEIO",
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("333611", "US", 2023)

            # Assert
            assert result is not None
            assert result.naics_code == "333611"
            assert result.ef_co2e_per_usd == Decimal("0.525")
            assert result.source == EmissionFactorSource.USEEIO
            assert mock_cursor.execute.call_count >= 1

    def test_get_eeio_factor_progressive_fallback_5digit(self, database_engine, mock_db_connection):
        """Test progressive fallback to 5-digit NAICS when 6-digit not found."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        # First call (6-digit) returns None, second call (5-digit) returns result
        mock_cursor.fetchone.side_effect = [
            None,  # 6-digit not found
            {
                "naics_code": "33361",
                "description": "Engine, Turbine, and Power Transmission Equipment",
                "ef_co2e_per_usd": Decimal("0.480"),
                "ef_co2_per_usd": Decimal("0.410"),
                "ef_ch4_per_usd": Decimal("0.012"),
                "ef_n2o_per_usd": Decimal("0.058"),
                "region": "US",
                "year": 2023,
                "source": "USEEIO",
            },
        ]

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("333619", "US", 2023)

            # Assert
            assert result is not None
            assert result.naics_code == "33361"
            assert result.ef_co2e_per_usd == Decimal("0.480")
            assert mock_cursor.execute.call_count >= 2

    def test_get_eeio_factor_progressive_fallback_4digit(self, database_engine, mock_db_connection):
        """Test progressive fallback to 4-digit NAICS."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.side_effect = [
            None,  # 6-digit
            None,  # 5-digit
            {
                "naics_code": "3336",
                "description": "Engine, Turbine, and Power Transmission Equipment",
                "ef_co2e_per_usd": Decimal("0.450"),
                "region": "US",
                "year": 2023,
                "source": "USEEIO",
            },
        ]

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("333698", "US", 2023)

            # Assert
            assert result is not None
            assert result.naics_code == "3336"
            assert mock_cursor.execute.call_count >= 3

    def test_get_eeio_factor_progressive_fallback_3digit(self, database_engine, mock_db_connection):
        """Test progressive fallback to 3-digit NAICS."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.side_effect = [
            None, None, None,  # 6, 5, 4-digit
            {
                "naics_code": "333",
                "description": "Machinery Manufacturing",
                "ef_co2e_per_usd": Decimal("0.420"),
                "region": "US",
                "year": 2023,
                "source": "USEEIO",
            },
        ]

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("333999", "US", 2023)

            # Assert
            assert result is not None
            assert result.naics_code == "333"
            assert mock_cursor.execute.call_count >= 4

    def test_get_eeio_factor_progressive_fallback_2digit(self, database_engine, mock_db_connection):
        """Test progressive fallback to 2-digit NAICS (final fallback)."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.side_effect = [
            None, None, None, None,  # 6, 5, 4, 3-digit
            {
                "naics_code": "33",
                "description": "Manufacturing",
                "ef_co2e_per_usd": Decimal("0.385"),
                "region": "US",
                "year": 2023,
                "source": "USEEIO",
            },
        ]

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("339999", "US", 2023)

            # Assert
            assert result is not None
            assert result.naics_code == "33"
            assert mock_cursor.execute.call_count >= 5

    def test_get_eeio_factor_all_fallbacks_fail(self, database_engine, mock_db_connection):
        """Test when all progressive fallbacks fail, returns None."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None  # All queries return None

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("999999", "US", 2023)

            # Assert
            assert result is None
            assert mock_cursor.execute.call_count >= 5

    def test_get_eeio_factor_invalid_naics_code(self, database_engine):
        """Test invalid NAICS code format returns None."""
        # Act
        result = database_engine.get_eeio_factor("ABC123", "US", 2023)

        # Assert
        assert result is None

    def test_get_eeio_factor_different_regions(self, database_engine, mock_db_connection):
        """Test EEIO factor lookup for different regions."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "naics_code": "333611",
            "description": "Turbine Manufacturing",
            "ef_co2e_per_usd": Decimal("0.620"),
            "region": "EU",
            "year": 2023,
            "source": "EXIOBASE",
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("333611", "EU", 2023)

            # Assert
            assert result is not None
            assert result.region == "EU"
            assert result.source == EmissionFactorSource.EXIOBASE

    def test_get_eeio_factor_different_years(self, database_engine, mock_db_connection):
        """Test EEIO factor lookup for different years."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "naics_code": "333611",
            "description": "Turbine Manufacturing",
            "ef_co2e_per_usd": Decimal("0.550"),
            "region": "US",
            "year": 2020,
            "source": "USEEIO",
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_eeio_factor("333611", "US", 2020)

            # Assert
            assert result is not None
            assert result.year == 2020


class TestPhysicalEmissionFactorLookup:
    """Test physical emission factor lookup for materials."""

    def test_get_physical_ef_structural_steel(self, database_engine, mock_db_connection):
        """Test lookup for structural steel."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "material_type": "structural_steel",
            "description": "Structural Steel",
            "ef_co2e_per_kg": Decimal("2.35"),
            "ef_co2_per_kg": Decimal("2.10"),
            "unit": "kg",
            "region": "GLOBAL",
            "source": "ICE",
            "year": 2023,
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_physical_ef("structural_steel", "GLOBAL", 2023)

            # Assert
            assert result is not None
            assert result.material_type == MaterialType.STRUCTURAL_STEEL
            assert result.ef_co2e_per_kg == Decimal("2.35")
            assert result.source == EmissionFactorSource.ICE

    def test_get_physical_ef_concrete_25mpa(self, database_engine, mock_db_connection):
        """Test lookup for 25 MPa concrete."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "material_type": "concrete_25mpa",
            "description": "25 MPa Concrete",
            "ef_co2e_per_kg": Decimal("0.145"),
            "unit": "kg",
            "region": "GLOBAL",
            "source": "ICE",
            "year": 2023,
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_physical_ef("concrete_25mpa", "GLOBAL", 2023)

            # Assert
            assert result is not None
            assert result.ef_co2e_per_kg == Decimal("0.145")

    def test_get_physical_ef_aluminum_primary(self, database_engine, mock_db_connection):
        """Test lookup for primary aluminum."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "material_type": "aluminum_primary",
            "description": "Primary Aluminum",
            "ef_co2e_per_kg": Decimal("11.5"),
            "unit": "kg",
            "region": "GLOBAL",
            "source": "ECOINVENT",
            "year": 2023,
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_physical_ef("aluminum_primary", "GLOBAL", 2023)

            # Assert
            assert result is not None
            assert result.ef_co2e_per_kg == Decimal("11.5")
            assert result.source == EmissionFactorSource.ECOINVENT

    def test_get_physical_ef_not_found(self, database_engine, mock_db_connection):
        """Test material not found returns None."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_physical_ef("unknown_material", "GLOBAL", 2023)

            # Assert
            assert result is None

    def test_get_physical_ef_regional_variations(self, database_engine, mock_db_connection):
        """Test regional variations in physical EF."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "material_type": "structural_steel",
            "description": "Structural Steel (China)",
            "ef_co2e_per_kg": Decimal("2.85"),
            "unit": "kg",
            "region": "CN",
            "source": "DEFRA",
            "year": 2023,
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_physical_ef("structural_steel", "CN", 2023)

            # Assert
            assert result is not None
            assert result.region == "CN"
            assert result.ef_co2e_per_kg == Decimal("2.85")


class TestSupplierEmissionFactorLookup:
    """Test supplier-specific emission factor lookup."""

    def test_get_supplier_ef_valid_supplier(self, database_engine, mock_db_connection):
        """Test lookup for valid supplier."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "supplier_id": "SUPP-001",
            "supplier_name": "ABC Steel Corp",
            "material_type": "structural_steel",
            "ef_co2e_per_kg": Decimal("1.85"),
            "verification_status": "VERIFIED",
            "valid_from": datetime(2023, 1, 1),
            "valid_until": datetime(2024, 12, 31),
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_supplier_ef("SUPP-001", "structural_steel")

            # Assert
            assert result is not None
            assert result.supplier_id == "SUPP-001"
            assert result.ef_co2e_per_kg == Decimal("1.85")
            assert result.verification_status == "VERIFIED"

    def test_get_supplier_ef_not_found(self, database_engine, mock_db_connection):
        """Test supplier not found returns None."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_supplier_ef("SUPP-999", "structural_steel")

            # Assert
            assert result is None


class TestAssetClassification:
    """Test asset classification functionality."""

    def test_classify_asset_buildings(self, database_engine):
        """Test classification of building assets."""
        # Act
        result = database_engine.classify_asset(
            "Office Building Construction",
            naics_code="236220"
        )

        # Assert
        assert result == AssetCategory.BUILDINGS

    def test_classify_asset_machinery(self, database_engine):
        """Test classification of machinery assets."""
        # Act
        result = database_engine.classify_asset(
            "CNC Machining Center",
            naics_code="333517"
        )

        # Assert
        assert result == AssetCategory.MACHINERY

    def test_classify_asset_vehicles(self, database_engine):
        """Test classification of vehicle assets."""
        # Act
        result = database_engine.classify_asset(
            "Delivery Truck Fleet",
            naics_code="336120"
        )

        # Assert
        assert result == AssetCategory.VEHICLES

    def test_classify_asset_it_equipment(self, database_engine):
        """Test classification of IT equipment."""
        # Act
        result = database_engine.classify_asset(
            "Server Infrastructure",
            naics_code="334111"
        )

        # Assert
        assert result == AssetCategory.IT_EQUIPMENT

    def test_classify_asset_furniture(self, database_engine):
        """Test classification of furniture."""
        # Act
        result = database_engine.classify_asset(
            "Office Furniture",
            naics_code="337211"
        )

        # Assert
        assert result == AssetCategory.FURNITURE

    def test_classify_asset_other(self, database_engine):
        """Test classification defaults to OTHER for unknown."""
        # Act
        result = database_engine.classify_asset(
            "Unknown Asset",
            naics_code="999999"
        )

        # Assert
        assert result == AssetCategory.OTHER


class TestCapitalizationCheck:
    """Test capitalization threshold checking."""

    def test_check_capitalization_above_threshold(self, database_engine):
        """Test asset value above capitalization threshold."""
        # Act
        result = database_engine.check_capitalization(
            asset_value=Decimal("10000.00"),
            threshold=Decimal("5000.00")
        )

        # Assert
        assert result is True

    def test_check_capitalization_at_threshold(self, database_engine):
        """Test asset value at capitalization threshold."""
        # Act
        result = database_engine.check_capitalization(
            asset_value=Decimal("5000.00"),
            threshold=Decimal("5000.00")
        )

        # Assert
        assert result is True

    def test_check_capitalization_below_threshold(self, database_engine):
        """Test asset value below capitalization threshold."""
        # Act
        result = database_engine.check_capitalization(
            asset_value=Decimal("2500.00"),
            threshold=Decimal("5000.00")
        )

        # Assert
        assert result is False

    def test_check_capitalization_default_threshold(self, database_engine):
        """Test default capitalization threshold."""
        # Act
        result = database_engine.check_capitalization(
            asset_value=Decimal("6000.00")
        )

        # Assert
        assert result is True  # Default threshold is typically $5000


class TestUsefulLifeEstimation:
    """Test useful life estimation for asset categories."""

    def test_get_useful_life_buildings(self, database_engine):
        """Test useful life for buildings."""
        # Act
        result = database_engine.get_useful_life(AssetCategory.BUILDINGS)

        # Assert
        assert result is not None
        assert result["min_years"] >= 20
        assert result["max_years"] <= 50
        assert result["typical_years"] >= result["min_years"]
        assert result["typical_years"] <= result["max_years"]

    def test_get_useful_life_machinery(self, database_engine):
        """Test useful life for machinery."""
        # Act
        result = database_engine.get_useful_life(AssetCategory.MACHINERY)

        # Assert
        assert result is not None
        assert 5 <= result["min_years"] <= 20
        assert result["typical_years"] in range(7, 16)

    def test_get_useful_life_vehicles(self, database_engine):
        """Test useful life for vehicles."""
        # Act
        result = database_engine.get_useful_life(AssetCategory.VEHICLES)

        # Assert
        assert result is not None
        assert result["typical_years"] in range(5, 11)

    def test_get_useful_life_it_equipment(self, database_engine):
        """Test useful life for IT equipment."""
        # Act
        result = database_engine.get_useful_life(AssetCategory.IT_EQUIPMENT)

        # Assert
        assert result is not None
        assert result["typical_years"] in range(3, 8)

    def test_get_useful_life_furniture(self, database_engine):
        """Test useful life for furniture."""
        # Act
        result = database_engine.get_useful_life(AssetCategory.FURNITURE)

        # Assert
        assert result is not None
        assert result["typical_years"] in range(7, 13)


class TestCurrencyConversion:
    """Test currency conversion with 20+ currencies."""

    @pytest.mark.parametrize("from_currency,to_currency,amount,expected_min,expected_max", [
        ("USD", "EUR", Decimal("1000"), Decimal("800"), Decimal("950")),
        ("EUR", "USD", Decimal("1000"), Decimal("1050"), Decimal("1200")),
        ("GBP", "USD", Decimal("1000"), Decimal("1200"), Decimal("1400")),
        ("JPY", "USD", Decimal("100000"), Decimal("650"), Decimal("850")),
        ("CNY", "USD", Decimal("7000"), Decimal("950"), Decimal("1050")),
        ("INR", "USD", Decimal("83000"), Decimal("950"), Decimal("1050")),
        ("CAD", "USD", Decimal("1000"), Decimal("700"), Decimal("800")),
        ("AUD", "USD", Decimal("1000"), Decimal("600"), Decimal("700")),
        ("CHF", "USD", Decimal("1000"), Decimal("1100"), Decimal("1200")),
        ("SEK", "USD", Decimal("10000"), Decimal("900"), Decimal("1100")),
    ])
    def test_convert_currency_various_pairs(
        self,
        database_engine,
        mock_db_connection,
        from_currency,
        to_currency,
        amount,
        expected_min,
        expected_max
    ):
        """Test currency conversion for various currency pairs."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        # Mock exchange rate lookup
        rate = Decimal("1.10") if from_currency == "EUR" else Decimal("0.90")
        mock_cursor.fetchone.return_value = {
            "rate": rate,
            "date": datetime(2023, 12, 1),
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.convert_currency(
                amount,
                from_currency,
                to_currency,
                datetime(2023, 12, 1)
            )

            # Assert
            assert result is not None
            # Allow reasonable range for exchange rate variations
            assert expected_min <= result <= expected_max

    def test_convert_currency_same_currency(self, database_engine):
        """Test conversion with same source and target currency."""
        # Act
        result = database_engine.convert_currency(
            Decimal("1000"),
            "USD",
            "USD",
            datetime(2023, 12, 1)
        )

        # Assert
        assert result == Decimal("1000")

    def test_convert_currency_historical_rate(self, database_engine, mock_db_connection):
        """Test conversion with historical exchange rate."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "rate": Decimal("1.15"),
            "date": datetime(2020, 1, 1),
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.convert_currency(
                Decimal("1000"),
                "EUR",
                "USD",
                datetime(2020, 1, 1)
            )

            # Assert
            assert result == Decimal("1150.00")


class TestCPIDeflation:
    """Test CPI-based deflation to base year."""

    def test_deflate_to_base_year_2023_to_2020(self, database_engine, mock_db_connection):
        """Test deflation from 2023 to 2020 base year."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        # Mock CPI index lookups: 2023 CPI = 120, 2020 CPI = 100
        mock_cursor.fetchone.side_effect = [
            {"cpi_index": Decimal("120.0")},  # 2023
            {"cpi_index": Decimal("100.0")},  # 2020
        ]

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.deflate_to_base_year(
                Decimal("1200"),
                from_year=2023,
                base_year=2020,
                region="US"
            )

            # Assert
            assert result == Decimal("1000.00")

    def test_deflate_to_base_year_same_year(self, database_engine):
        """Test deflation with same source and base year."""
        # Act
        result = database_engine.deflate_to_base_year(
            Decimal("1000"),
            from_year=2023,
            base_year=2023,
            region="US"
        )

        # Assert
        assert result == Decimal("1000")

    def test_deflate_to_base_year_inflation_adjustment(self, database_engine, mock_db_connection):
        """Test deflation correctly adjusts for inflation."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        # Older year has lower CPI (deflation increases value in real terms)
        mock_cursor.fetchone.side_effect = [
            {"cpi_index": Decimal("90.0")},   # 2019
            {"cpi_index": Decimal("100.0")},  # 2020
        ]

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.deflate_to_base_year(
                Decimal("900"),
                from_year=2019,
                base_year=2020,
                region="US"
            )

            # Assert
            assert result == Decimal("1000.00")


class TestMarginRemoval:
    """Test margin removal for capital goods sectors."""

    def test_remove_margin_machinery_sector(self, database_engine, mock_db_connection):
        """Test margin removal for machinery sector."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "margin_percentage": Decimal("0.25"),  # 25% margin
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.remove_margin(
                Decimal("10000"),
                naics_code="333611"
            )

            # Assert
            assert result == Decimal("7500.00")  # 10000 * (1 - 0.25)

    def test_remove_margin_construction_sector(self, database_engine, mock_db_connection):
        """Test margin removal for construction sector."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "margin_percentage": Decimal("0.18"),  # 18% margin
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.remove_margin(
                Decimal("20000"),
                naics_code="236220"
            )

            # Assert
            assert result == Decimal("16400.00")

    def test_remove_margin_no_margin_data(self, database_engine, mock_db_connection):
        """Test margin removal when no margin data available."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act - should return original amount
            result = database_engine.remove_margin(
                Decimal("10000"),
                naics_code="999999"
            )

            # Assert
            assert result == Decimal("10000")


class TestTaxonomyMapping:
    """Test taxonomy mapping between classification systems."""

    def test_map_naics_to_isic(self, database_engine, mock_db_connection):
        """Test NAICS to ISIC mapping."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "isic_code": "2811",
            "description": "Manufacture of engines and turbines",
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.map_naics_to_isic("333611")

            # Assert
            assert result == "2811"

    def test_map_nace_to_isic(self, database_engine, mock_db_connection):
        """Test NACE to ISIC mapping."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "isic_code": "2811",
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.map_nace_to_isic("28.11")

            # Assert
            assert result == "2811"

    def test_map_unspsc_to_naics(self, database_engine, mock_db_connection):
        """Test UNSPSC to NAICS mapping."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {
            "naics_code": "333611",
        }

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.map_unspsc_to_naics("26111501")

            # Assert
            assert result == "333611"

    def test_map_naics_to_isic_not_found(self, database_engine, mock_db_connection):
        """Test NAICS to ISIC mapping when not found."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.map_naics_to_isic("999999")

            # Assert
            assert result is None


class TestCustomFactorRegistration:
    """Test custom emission factor registration and override."""

    def test_register_custom_eeio_factor(self, database_engine):
        """Test registering custom EEIO factor."""
        # Arrange
        custom_factor = EEIOFactor(
            naics_code="888888",
            description="Custom Test Sector",
            ef_co2e_per_usd=Decimal("2.5"),
            region="US",
            year=2023,
            source=EmissionFactorSource.CUSTOM,
        )

        # Act
        database_engine.register_custom_eeio_factor(custom_factor)
        result = database_engine.get_eeio_factor("888888", "US", 2023)

        # Assert
        assert result is not None
        assert result.ef_co2e_per_usd == Decimal("2.5")
        assert result.source == EmissionFactorSource.CUSTOM

    def test_register_custom_eeio_factor_override(self, database_engine, mock_db_connection):
        """Test custom factor overrides built-in factor."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        custom_factor = EEIOFactor(
            naics_code="333611",
            description="Custom Turbine Factor",
            ef_co2e_per_usd=Decimal("0.999"),
            region="US",
            year=2023,
            source=EmissionFactorSource.CUSTOM,
        )

        # Act
        database_engine.register_custom_eeio_factor(custom_factor)
        result = database_engine.get_eeio_factor("333611", "US", 2023)

        # Assert - should return custom factor, not query database
        assert result is not None
        assert result.ef_co2e_per_usd == Decimal("0.999")
        assert mock_cursor.execute.call_count == 0  # No DB query

    def test_register_custom_physical_ef(self, database_engine):
        """Test registering custom physical emission factor."""
        # Arrange
        custom_ef = PhysicalEmissionFactor(
            material_type=MaterialType.CUSTOM,
            description="Custom Material",
            ef_co2e_per_kg=Decimal("5.5"),
            unit="kg",
            region="US",
            source=EmissionFactorSource.CUSTOM,
            year=2023,
        )

        # Act
        database_engine.register_custom_physical_ef(custom_ef)
        result = database_engine.get_physical_ef("custom_material", "US", 2023)

        # Assert
        assert result is not None
        assert result.ef_co2e_per_kg == Decimal("5.5")

    def test_register_custom_supplier_ef(self, database_engine):
        """Test registering custom supplier emission factor."""
        # Arrange
        custom_supplier_ef = SupplierEmissionFactor(
            supplier_id="CUSTOM-001",
            supplier_name="Custom Supplier Inc",
            material_type=MaterialType.STRUCTURAL_STEEL,
            ef_co2e_per_kg=Decimal("1.5"),
            verification_status="VERIFIED",
            valid_from=datetime(2023, 1, 1),
            valid_until=datetime(2024, 12, 31),
        )

        # Act
        database_engine.register_custom_supplier_ef(custom_supplier_ef)
        result = database_engine.get_supplier_ef("CUSTOM-001", "structural_steel")

        # Assert
        assert result is not None
        assert result.ef_co2e_per_kg == Decimal("1.5")


class TestEmissionFactorHierarchy:
    """Test emission factor selection hierarchy."""

    def test_select_best_ef_supplier_priority(self, database_engine):
        """Test supplier-specific EF has highest priority."""
        # Arrange
        supplier_ef = Decimal("1.5")
        physical_ef = Decimal("2.0")
        eeio_ef = Decimal("0.5")

        # Act
        result = database_engine.select_best_ef(
            supplier_ef=supplier_ef,
            physical_ef=physical_ef,
            eeio_ef=eeio_ef
        )

        # Assert
        assert result == supplier_ef
        assert result == Decimal("1.5")

    def test_select_best_ef_physical_second_priority(self, database_engine):
        """Test physical EF has second priority when supplier not available."""
        # Arrange
        supplier_ef = None
        physical_ef = Decimal("2.0")
        eeio_ef = Decimal("0.5")

        # Act
        result = database_engine.select_best_ef(
            supplier_ef=supplier_ef,
            physical_ef=physical_ef,
            eeio_ef=eeio_ef
        )

        # Assert
        assert result == physical_ef
        assert result == Decimal("2.0")

    def test_select_best_ef_eeio_fallback(self, database_engine):
        """Test EEIO EF is fallback when others not available."""
        # Arrange
        supplier_ef = None
        physical_ef = None
        eeio_ef = Decimal("0.5")

        # Act
        result = database_engine.select_best_ef(
            supplier_ef=supplier_ef,
            physical_ef=physical_ef,
            eeio_ef=eeio_ef
        )

        # Assert
        assert result == eeio_ef
        assert result == Decimal("0.5")

    def test_select_best_ef_none_available(self, database_engine):
        """Test returns None when no EF available."""
        # Act
        result = database_engine.select_best_ef(
            supplier_ef=None,
            physical_ef=None,
            eeio_ef=None
        )

        # Assert
        assert result is None


class TestNAICSValidation:
    """Test NAICS code validation."""

    def test_validate_naics_code_valid_6digit(self, database_engine):
        """Test validation of valid 6-digit NAICS code."""
        # Act
        result = database_engine.validate_naics_code("333611")

        # Assert
        assert result is True

    def test_validate_naics_code_valid_5digit(self, database_engine):
        """Test validation of valid 5-digit NAICS code."""
        # Act
        result = database_engine.validate_naics_code("33361")

        # Assert
        assert result is True

    def test_validate_naics_code_valid_4digit(self, database_engine):
        """Test validation of valid 4-digit NAICS code."""
        # Act
        result = database_engine.validate_naics_code("3336")

        # Assert
        assert result is True

    def test_validate_naics_code_valid_3digit(self, database_engine):
        """Test validation of valid 3-digit NAICS code."""
        # Act
        result = database_engine.validate_naics_code("333")

        # Assert
        assert result is True

    def test_validate_naics_code_valid_2digit(self, database_engine):
        """Test validation of valid 2-digit NAICS code."""
        # Act
        result = database_engine.validate_naics_code("33")

        # Assert
        assert result is True

    def test_validate_naics_code_invalid_format(self, database_engine):
        """Test validation rejects invalid format."""
        # Act
        result = database_engine.validate_naics_code("ABC123")

        # Assert
        assert result is False

    def test_validate_naics_code_too_long(self, database_engine):
        """Test validation rejects too long code."""
        # Act
        result = database_engine.validate_naics_code("3336111")

        # Assert
        assert result is False

    def test_validate_naics_code_empty(self, database_engine):
        """Test validation rejects empty code."""
        # Act
        result = database_engine.validate_naics_code("")

        # Assert
        assert result is False


class TestDatabaseStatistics:
    """Test database statistics retrieval."""

    def test_get_database_stats(self, database_engine, mock_db_connection):
        """Test retrieving database statistics."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.side_effect = [
            {"count": 25000},  # EEIO factors
            {"count": 850},    # Physical EFs
            {"count": 150},    # Supplier EFs
            {"count": 45},     # Regions
            {"count": 5},      # Years
        ]

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_database_stats()

            # Assert
            assert result is not None
            assert result.total_eeio_factors == 25000
            assert result.total_physical_ef == 850
            assert result.total_supplier_ef == 150
            assert result.regions_covered == 45
            assert result.years_covered == 5

    def test_get_database_stats_empty_database(self, database_engine, mock_db_connection):
        """Test statistics for empty database."""
        # Arrange
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = {"count": 0}

        with patch.object(database_engine, "_get_db_connection", return_value=mock_conn):
            # Act
            result = database_engine.get_database_stats()

            # Assert
            assert result.total_eeio_factors == 0
            assert result.total_physical_ef == 0
